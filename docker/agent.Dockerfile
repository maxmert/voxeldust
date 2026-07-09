# syntax=docker/dockerfile:1
# S5b/S5a — the agent-HR6 client image: `client` + `vdctl`, built `--features dev-control` (Bevy-FREE).
# SEPARATE from server.Dockerfile ON PURPOSE: cargo unifies features per `-p` build, so adding `dev-control`
# to the server build would link the loopback dev-control listener into the server bins (the "absence is a
# compile fact" contract). `dev-control` does NOT pull `render` (a separate feature), so no wgpu/Bevy here —
# this stays a small debian-slim image. The agent logs an avatar into the LIVE cluster gateway over the pod
# network (QUIC) and is driven by vdctl over loopback TCP; it asserts on DevState (no GPU capture).
FROM rust:1.94.1-slim-bookworm AS builder
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config perl make && \
    rm -rf /var/lib/apt/lists/*
COPY . .
# Bevy-FREE: `--features dev-control` links the loopback listener (client) + the vdctl wire, NOT `render`.
RUN cargo build --release -p vd-bins --features dev-control --bin client --bin vdctl

FROM debian:bookworm-slim AS runtime
# jq parses the typed `vdctl state` JSON in the scenario (no brittle grep of pos[0]); ca-certificates is insurance.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates jq && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin vd
COPY --from=builder /src/target/release/client /usr/local/bin/client
COPY --from=builder /src/target/release/vdctl  /usr/local/bin/vdctl
# The scenario scripts MUST live under docker/ — .dockerignore excludes deploy/ + scripts/, so a script placed
# there would be silently absent from the build context and the ENTRYPOINT would crash at container start.
COPY docker/agent-entrypoint.sh  /usr/local/bin/agent-entrypoint.sh
COPY docker/scenario-boundary.sh /usr/local/bin/scenario-boundary.sh
RUN chmod 0755 /usr/local/bin/agent-entrypoint.sh /usr/local/bin/scenario-boundary.sh
USER vd
ENTRYPOINT ["agent-entrypoint.sh"]
