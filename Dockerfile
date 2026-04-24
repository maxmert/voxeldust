# Multi-stage Dockerfile for all voxeldust binaries.
# Build with: docker build --target <name> -t voxeldust-<name>:latest .
# Targets: orchestrator, gateway, stub-shard, system-shard, ship-shard, planet-shard

# ---- Builder stage ----
FROM rust:1.93-bookworm AS builder
RUN apt-get update && apt-get install -y clang libclang-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /build

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY core/ core/
COPY shard-common/ shard-common/
COPY orchestrator/ orchestrator/
COPY gateway/ gateway/
COPY stub-shard/ stub-shard/
COPY system-shard/ system-shard/
COPY ship-shard/ ship-shard/
COPY planet-shard/ planet-shard/
COPY e2e-tests/ e2e-tests/
COPY protocol/ protocol/
COPY galaxy-shard/ galaxy-shard/
# client: stub so workspace resolves (client-only crate, not built in server image)
COPY client/Cargo.toml client/Cargo.toml
RUN mkdir -p client/src && echo "fn main() {}" > client/src/main.rs

# Build all server binaries in release mode
RUN cargo build --release \
    -p orchestrator \
    -p gateway \
    -p stub-shard \
    -p system-shard \
    -p ship-shard \
    -p planet-shard \
    -p galaxy-shard

# ---- Orchestrator ----
FROM debian:bookworm-slim AS orchestrator
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/orchestrator /usr/local/bin/
ENTRYPOINT ["orchestrator"]

# ---- Gateway ----
FROM debian:bookworm-slim AS gateway
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/gateway /usr/local/bin/
ENTRYPOINT ["gateway"]

# ---- Stub Shard ----
FROM debian:bookworm-slim AS stub-shard
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/stub-shard /usr/local/bin/
ENTRYPOINT ["stub-shard"]

# ---- System Shard ----
FROM debian:bookworm-slim AS system-shard
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/system-shard /usr/local/bin/
ENTRYPOINT ["system-shard"]

# ---- Ship Shard ----
FROM debian:bookworm-slim AS ship-shard
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/ship-shard /usr/local/bin/
ENTRYPOINT ["ship-shard"]

# ---- Planet Shard ----
FROM debian:bookworm-slim AS planet-shard
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/planet-shard /usr/local/bin/
ENTRYPOINT ["planet-shard"]

# ---- Galaxy Shard ----
FROM debian:bookworm-slim AS galaxy-shard
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/galaxy-shard /usr/local/bin/
ENTRYPOINT ["galaxy-shard"]
