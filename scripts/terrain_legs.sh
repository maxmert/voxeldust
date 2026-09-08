#!/bin/sh
# ★ THE GOLDEN LEGS ON ANOTHER TARGET (SL10 clause 3: no drift is MEASURED on every shipped target).
# Runs the generator's and the leaf's tests — the golden table of 2 592 digests included — inside the
# pinned toolchain image on another platform. The x86-64 leg on an Apple host is EMULATED: it can find
# a drift and can never prove its absence; a real x86-64 machine is owed before "no drift" is called
# satisfied (ruling V6 D, V9 S5-5; DEFERRED D-TERRAIN-1). The aarch64 Linux leg is the k3d image's
# target triple and base image, without the image build itself.
#
#   scripts/terrain_legs.sh linux/amd64
#   scripts/terrain_legs.sh linux/arm64
#
# The host's target/ is never touched: the leg builds in target/legs-<platform>/ and keeps its own
# crate cache beside it.
set -eu
PLATFORM="${1:?usage: terrain_legs.sh linux/amd64|linux/arm64}"
TAG=$(echo "$PLATFORM" | tr '/' '-')
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=rust:1.94.1-slim-bookworm
mkdir -p "$REPO/target/legs-$TAG/tgt" "$REPO/target/legs-$TAG/registry"
docker run --rm --platform "$PLATFORM" \
  -v "$REPO":/src:ro \
  -v "$REPO/target/legs-$TAG/tgt":/tgt \
  -v "$REPO/target/legs-$TAG/registry":/usr/local/cargo/registry \
  -w /src \
  -e CARGO_TARGET_DIR=/tgt \
  -e RUSTUP_TOOLCHAIN=1.94.1 \
  "$IMAGE" \
  sh -c 'echo "terrain-legs: $(uname -m) $(rustc -vV | grep host)"; \
         cargo test --release -p vd-seed -p vd-terrain --lib --tests -- --quiet 2>&1 \
           | grep -E "test result|FAILED|panicked|^error"'
echo "terrain-legs: $PLATFORM PASS"
