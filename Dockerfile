# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM rust:1.98-bookworm@sha256:e70e2eec3d495fd5c8e0be74adda86507dfac7f51a724fbf9813ff59b2b247c7 AS builder

ARG VCS_REF=unknown

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY vendor ./vendor
RUN MARIAN_EDGE_BUILD_GIT_SHA="$VCS_REF" \
    cargo build --locked --release -p marian-server --features cpu && \
    install -D -m 0755 target/release/marian-edge-server /out/bin/marian-edge-server

FROM debian:bookworm-slim@sha256:60eac759739651111db372c07be67863818726f754804b8707c90979bda511df AS runtime

ARG VERSION=0.7.0
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Marian Edge CPU" \
      org.opencontainers.image.description="Local Marian translation service using the pure-Rust Q8 CPU backend" \
      org.opencontainers.image.source="https://github.com/malusama/marian-edge" \
      org.opencontainers.image.url="https://github.com/malusama/marian-edge" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="$VERSION" \
      org.opencontainers.image.revision="$VCS_REF"

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gzip util-linux && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 65532 marian-edge && \
    useradd --uid 65532 --gid 65532 --no-create-home --home-dir /nonexistent \
      --shell /usr/sbin/nologin marian-edge && \
    install -d -o 65532 -g 65532 /models /usr/share/licenses/marian-edge

COPY --from=builder /out/bin/ /usr/local/bin/
COPY --chmod=0755 docker/prepare-model.sh /usr/local/bin/marian-edge-prepare-model
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/marian-edge-entrypoint
COPY LICENSE LICENSE-APACHE-2.0 THIRD_PARTY_NOTICES.md /usr/share/licenses/marian-edge/

ENV MARIAN_EDGE_BACKEND=cpu \
    MARIAN_EDGE_BIND=0.0.0.0:3000 \
    MARIAN_EDGE_MODEL_DIR=/models/en-zh

USER 65532:65532
VOLUME ["/models"]
EXPOSE 3000
ENTRYPOINT ["/usr/local/bin/marian-edge-entrypoint"]
CMD []
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=6 \
  CMD curl -fsS http://127.0.0.1:3000/readyz || exit 1
