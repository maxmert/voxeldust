# syntax=docker/dockerfile:1
#
# Cloud-ready k3d — Slice 0: the ONE multi-bin SERVER image.
#
# HR3 (one shard binary; a shard type is a ShardProfile config, never a per-kind image):
# this single image carries all three server bins — vd-orchestrator / vd-gateway / vd-shard.
# A pod's ROLE is which binary its `command:` runs + its VD_* env, NOT a distinct image.
# (The prior repo's per-shard-kind images were deleted for exactly this HR3 violation.)
#
# Built Bevy-FREE: `-p vd-bins` with DEFAULT features pulls no `render`/`dev-control` — the
# ~8.8 GB Bevy dep is behind the optional `render` feature (client only), so the server image
# stays small. The agent `--capture` client is a SEPARATE, heavier image (a later slice).
#
# Multi-stage by necessity: the dev host is macOS (Mach-O), so a host binary cannot run in a
# Linux pod — the release build happens INSIDE a Linux glibc toolchain, and the runtime stage
# is debian-slim with a MATCHING glibc (D3: glibc/debian-slim chosen over musl/distroless for
# the first bring-up — lowest risk that ring/quinn/rustls build cleanly; distroless is a later
# additive optimization).

# ---- builder --------------------------------------------------------------------------------
FROM rust:1.94.1-slim-bookworm AS builder
WORKDIR /src
# ring (rustls' crypto backend, via quinn) needs a C toolchain + perl at build time; redb is
# pure Rust. pkg-config is harmless-but-conventional. No openssl (rustls uses ring, not openssl).
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config perl make && \
    rm -rf /var/lib/apt/lists/*
# The whole workspace (the .dockerignore keeps target/ (~70 GB), .git, runs/, scripts/ out of
# the build context). rust-toolchain.toml pins 1.94.1 — the base image tag matches it exactly.
COPY . .
# Release build of ONLY the three server bins (default features → no Bevy, no dev-control).
RUN cargo build --release -p vd-bins \
        --bin vd-orchestrator --bin vd-gateway --bin vd-shard

# ---- runtime --------------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime
# ca-certificates is defensive (no TLS-to-public-CA path today — inter-node mTLS uses the
# mounted DER bundle, /metrics is plain HTTP — but cheap insurance against a future dep).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin vd
COPY --from=builder /src/target/release/vd-orchestrator /usr/local/bin/vd-orchestrator
COPY --from=builder /src/target/release/vd-gateway      /usr/local/bin/vd-gateway
COPY --from=builder /src/target/release/vd-shard        /usr/local/bin/vd-shard
# S4b: the k3d entrypoint. VD_PEERS is SocketAddr-ONLY (no k8s DNS names), so this DNS-resolves the three
# headless-Service pod names to literal IPs, builds the id=IP:port book for the role, then `exec`s the bin
# (so the bin is PID 1 and receives SIGTERM directly — the graceful-drain path). debian-slim ships sh + getent.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh
# Non-root by default (k8s runAsNonRoot); PVC mounts get an fsGroup in the manifests (Slice 4).
USER vd
# Deliberately NO ENTRYPOINT/CMD: each workload's manifest sets
#   command: ["entrypoint.sh", "vd-orchestrator" | "vd-gateway" | "vd-shard"]  (local dev/docker-run may still
#   invoke the bin directly with a literal VD_PEERS, bypassing entrypoint.sh — the image serves both).
# so the one image serves every server role by command + env (HR3).
