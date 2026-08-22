use std::{cell::Cell, path::Path};

use marian_model::{Architecture, LexicalShortlist, MAXIMUM_POSITION, sinusoidal_positions};
use objc2::rc::autoreleasepool;

use crate::MetalConfig;
use crate::metal_runtime::{Buffer, Commands, MetalRuntime, MetalStorage};
use crate::tuning::MetalTuning;
use crate::workspace::{MetalWorkspace, TransientFrame};

mod decode;
mod model;
mod ops;

pub(crate) use decode::BatchOutput;
use model::{EncoderLayer, FeedForwardWeights, ModelWeights};
use ops::AttentionView;

const MAXIMUM_BATCH: usize = 256;

struct CrossCache {
    key_value: Buffer,
}

struct EncodedBatch {
    source_length: usize,
    lengths: Vec<usize>,
    length_buffer: Buffer,
    cross: Vec<CrossCache>,
}

pub(crate) struct MetalEngine {
    runtime: MetalRuntime,
    healthy: Cell<bool>,
    model: ModelWeights,
    positions: Buffer,
    shortlist: LexicalShortlist,
    dummy_bias: Buffer,
    dim: usize,
    heads: usize,
    ffn_dim: usize,
    source_vocab: usize,
    max_length_factor: usize,
    workspace: MetalWorkspace,
    storage: MetalStorage,
    tuning: MetalTuning,
}

impl MetalEngine {
    pub(crate) fn load(
        weights_path: &Path,
        shortlist_path: Option<&Path>,
        architecture: &Architecture,
        config: &MetalConfig,
    ) -> Result<Self, String> {
        let runtime = MetalRuntime::new(config.precision)?;
        let tuning = MetalTuning::resolve(runtime.device_name(), config)?;
        runtime.validate_execution_plan(tuning.decode.selection_threads())?;
        let storage = runtime.storage();
        let model = ModelWeights::load(&runtime, weights_path, architecture)?;
        let positions = runtime.upload(&sinusoidal_positions(architecture.model_dim)?)?;
        let shortlist = LexicalShortlist::load(
            shortlist_path,
            architecture.source_vocab_size,
            architecture.target_vocab_size,
        )?;
        let dummy_bias = runtime.upload(&[0.0_f32])?;
        Ok(Self {
            runtime,
            healthy: Cell::new(true),
            model,
            positions,
            shortlist,
            dummy_bias,
            dim: architecture.model_dim,
            heads: architecture.attention_heads,
            ffn_dim: architecture.ffn_dim,
            source_vocab: architecture.source_vocab_size,
            max_length_factor: architecture.max_length_factor.max(1),
            workspace: MetalWorkspace::default(),
            storage,
            tuning,
        })
    }

    pub(crate) fn device_name(&self) -> &str {
        self.runtime.device_name()
    }

    pub(crate) fn precision(&self) -> &'static str {
        self.storage.label()
    }

    pub(crate) fn attention_label(&self) -> String {
        self.tuning.attention_label()
    }

    pub(crate) fn tuning_profile(&self) -> String {
        self.tuning.profile_label()
    }

    pub(crate) fn duplicate_batch_width(&self) -> usize {
        self.tuning.duplicate_batch_width()
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.healthy.get()
    }

    pub(crate) fn warmup(&self) -> Result<(), String> {
        self.translate(&[2, 0], &[0, 2], &[4]).map(|_| ())
    }

    pub(crate) fn translate(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
    ) -> Result<BatchOutput, String> {
        self.validate_batch(tokens, offsets, max_output_tokens)?;
        // Once device execution begins, any error leaves request-scoped state
        // potentially incomplete. Fail closed; validation errors above remain
        // ordinary caller errors and do not poison readiness.
        // MPS creates autoreleased descriptor objects while encoding matrix
        // operations. The backend runs on a long-lived Rust worker thread, so
        // it has no event-loop boundary that would otherwise drain those
        // objects. Give every inference its own pool to keep idle memory tied
        // to the model and reusable workspace instead of cumulative traffic.
        autoreleasepool(|_| {
            self.execute_request(tokens, offsets, max_output_tokens)
                .inspect_err(|_| self.healthy.set(false))
        })
    }

    fn execute_request(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
    ) -> Result<BatchOutput, String> {
        self.workspace.begin_request(&self.runtime)?;
        let encoded = self.encode(tokens, offsets)?;
        self.decode(tokens, offsets, max_output_tokens, &encoded)
    }

    fn validate_batch(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
    ) -> Result<(), String> {
        if offsets.len() < 2
            || offsets[0] != 0
            || offsets.last().copied() != Some(tokens.len() as u32)
        {
            return Err("invalid packed batch offsets".into());
        }
        let batch = offsets.len() - 1;
        if batch > MAXIMUM_BATCH {
            return Err(format!(
                "batch contains {batch} sentences; maximum is {MAXIMUM_BATCH}"
            ));
        }
        if max_output_tokens.len() != batch {
            return Err("max_output_tokens does not match packed batch".into());
        }
        for pair in offsets.windows(2) {
            if pair[1] <= pair[0] {
                return Err("each input must contain at least one token".into());
            }
        }
        for &token in tokens {
            if token < 0 || token as usize >= self.source_vocab {
                return Err(format!(
                    "source token {token} is outside vocabulary {}",
                    self.source_vocab
                ));
            }
        }
        Ok(())
    }

    fn encode(&self, tokens: &[i32], offsets: &[u32]) -> Result<EncodedBatch, String> {
        let batch = offsets.len() - 1;
        let lengths = offsets
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as usize)
            .collect::<Vec<_>>();
        let source_length = lengths.iter().copied().max().unwrap_or(0);
        if source_length > MAXIMUM_POSITION {
            return Err(format!(
                "source has {source_length} tokens; maximum is {MAXIMUM_POSITION}"
            ));
        }
        let mut padded = vec![0_i32; checked_mul(batch, source_length)?];
        for row in 0..batch {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            padded[row * source_length..row * source_length + lengths[row]]
                .copy_from_slice(&tokens[start..end]);
        }
        let gpu_lengths = lengths
            .iter()
            .map(|&length| to_u32(length, "source length"))
            .collect::<Result<Vec<_>, _>>()?;
        let token_buffer = self.upload_buffer(&padded)?;
        let length_buffer = self.upload_buffer(&gpu_lengths)?;

        // Flash attention does not retain the classic path's quadratic score
        // buffers, so the complete encoder and cross-cache projection can be
        // submitted in one command buffer. This removes seven CPU/GPU waits
        // from every request batch while preserving the layer order recorded
        // in the compute encoder.
        if self
            .tuning
            .attention
            .use_flash(source_length, source_length, self.dim / self.heads)
        {
            let _scratch = self.begin_scratch()?;
            let commands = self.commands("encode+cross-cache")?;
            let mut encoder = self.embedding(&commands, &token_buffer, batch, source_length)?;
            for layer in &self.model.encoder {
                encoder = self.encode_layer(
                    &commands,
                    &encoder,
                    layer,
                    &length_buffer,
                    batch,
                    source_length,
                )?;
            }
            let cross = self.build_cross_cache(&commands, &encoder, batch, source_length)?;
            self.finish_commands(commands)?;
            return Ok(EncodedBatch {
                source_length,
                lengths,
                length_buffer,
                cross,
            });
        }

        let commands = self.commands("encode-embedding")?;
        let mut encoder = self.embedding(&commands, &token_buffer, batch, source_length)?;
        self.finish_commands(commands)?;

        for (index, layer) in self.model.encoder.iter().enumerate() {
            // Finish each encoder layer before dispatching the next one. A
            // retained command buffer keeps every temporary attention score
            // allocation alive; one command buffer for all six layers would
            // otherwise multiply the O(sequence^2) scratch requirement.
            let commands = self.commands(&format!("encode-layer-{}", index + 1))?;
            encoder = self.encode_layer(
                &commands,
                &encoder,
                layer,
                &length_buffer,
                batch,
                source_length,
            )?;
            self.finish_commands(commands)?;
        }

        let commands = self.commands("cross-cache")?;
        let cross = self.build_cross_cache(&commands, &encoder, batch, source_length)?;
        self.finish_commands(commands)?;
        Ok(EncodedBatch {
            source_length,
            lengths,
            length_buffer,
            cross,
        })
    }

    fn encode_layer(
        &self,
        commands: &Commands<'_>,
        encoder: &Buffer,
        layer: &EncoderLayer,
        lengths: &Buffer,
        batch: usize,
        source_length: usize,
    ) -> Result<Buffer, String> {
        let rows = checked_mul(batch, source_length)?;
        let qkv = self.matmul(
            commands,
            encoder,
            &layer.attention.qkv,
            Some(&layer.attention.qkv_bias),
            rows,
            checked_mul(self.dim, 3)?,
            self.dim,
        )?;
        let attended = self.attend(
            commands,
            AttentionView {
                buffer: &qkv,
                stride: self.dim * 3,
                offset: 0,
            },
            AttentionView {
                buffer: &qkv,
                stride: self.dim * 3,
                offset: self.dim,
            },
            AttentionView {
                buffer: &qkv,
                stride: self.dim * 3,
                offset: self.dim * 2,
            },
            lengths,
            batch,
            source_length,
            source_length,
        )?;
        let normalized = self.linear_residual_norm(
            commands,
            &attended,
            &layer.attention.output.weight,
            &layer.attention.output.bias,
            encoder,
            &layer.attention.output.norm_scale,
            &layer.attention.output.norm_bias,
            rows,
            self.dim,
        )?;
        self.feed_forward(commands, &normalized, &layer.ffn, rows)
    }

    fn build_cross_cache(
        &self,
        commands: &Commands<'_>,
        encoder: &Buffer,
        batch: usize,
        source_length: usize,
    ) -> Result<Vec<CrossCache>, String> {
        let rows = checked_mul(batch, source_length)?;
        let mut cross = Vec::with_capacity(self.model.decoder.len());
        for layer in &self.model.decoder {
            let key_value = self.matmul_persistent(
                commands,
                encoder,
                &layer.context.key_value,
                Some(&layer.context.key_value_bias),
                rows,
                checked_mul(self.dim, 2)?,
                self.dim,
            )?;
            cross.push(CrossCache { key_value });
        }
        Ok(cross)
    }

    fn finish_commands(&self, commands: Commands<'_>) -> Result<(), String> {
        commands.finish()
    }

    fn commands(&self, label: &str) -> Result<Commands<'_>, String> {
        self.runtime.commands_labeled(label)
    }

    fn begin_scratch(&self) -> Result<TransientFrame<'_>, String> {
        self.workspace.begin_transient()
    }

    fn feed_forward(
        &self,
        commands: &Commands<'_>,
        input: &Buffer,
        weights: &FeedForwardWeights,
        rows: usize,
    ) -> Result<Buffer, String> {
        let hidden = self.matmul_with_activation(
            commands,
            input,
            &weights.w1,
            Some(&weights.b1),
            rows,
            self.ffn_dim,
            self.dim,
            true,
            false,
        )?;
        self.linear_residual_norm(
            commands,
            &hidden,
            &weights.w2,
            &weights.b2,
            input,
            &weights.norm_scale,
            &weights.norm_bias,
            rows,
            self.ffn_dim,
        )
    }

    fn upload_buffer<T: crate::metal_runtime::MetalPod>(
        &self,
        values: &[T],
    ) -> Result<Buffer, String> {
        self.workspace.request_upload(&self.runtime, values)
    }
}

fn checked_mul(lhs: usize, rhs: usize) -> Result<usize, String> {
    lhs.checked_mul(rhs)
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Metal tensor shape {lhs} x {rhs} is zero or overflows"))
}

fn checked_product(values: &[usize]) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1_usize, |product, &value| checked_mul(product, value))
}

fn to_u32(value: usize, label: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{label} {value} exceeds u32"))
}
