#!/usr/bin/env bash
# visual-run.sh — boot the STATIC single-shard cluster on THE world + open an interactive client
# window showing the home star system. Reuses dev-cluster.sh (robust bring-up) + client.sh --window.
# Closing the window (or Ctrl-C) tears the cluster down.
#
# Usage:
#   scripts/visual-run.sh          # boot THE world's home system, open the client window
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Resolve the real target dir (honours CARGO_TARGET_DIR / .cargo config).
TARGET="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')/debug"

[[ -x "$TARGET/vd-devcluster" ]] || cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins

# 1. Boot the static cluster. THE world is not selected — there is only one. The client draws its
#    scene from the SERVER STREAM (the composed level on join + deltas + the per-tick composed
#    datagrams) — NO --realm-boxes file exists any more (Slice C1, window_lane.md §2.11, D-LANE-6).
"$ROOT/scripts/dev-cluster.sh" up

# 2. Tear the cluster down when the client window closes / on Ctrl-C.
cleanup() {
    "$ROOT/scripts/dev-cluster.sh" down >/dev/null 2>&1 || true
}
trap cleanup EXIT

# 3. Open the interactive client window (WASD + mouse fly the dot; the planets ORBIT because the
#    composed feed ships their poses each tick).
echo "launching the client window — WASD + mouse to fly the dot; the planets ORBIT. Ctrl-C to stop."
"$ROOT/scripts/client.sh" --window --name visual
