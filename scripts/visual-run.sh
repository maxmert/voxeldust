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

# vd-devcluster carries the emit-world-scene subcommand this launcher needs.
[[ -x "$TARGET/vd-devcluster" ]] || cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins

# 1. Dump THE world's regions.json for the client's --realm-boxes — the ONE scene emitter (SL5):
#    the home shard's own boot neighbourhood, so the client draws EXACTLY the geometry the shard's
#    detector evaluates. There is nothing to choose and no scale to pass.
SCENE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vd-visual-scene.XXXXXX")"
eval "$("$TARGET/vd-devcluster" emit-world-scene "$SCENE_DIR")" # sets VD_WORLD_SCENE
echo "world scene → $VD_WORLD_SCENE"

# 2. Boot the static cluster. THE world is not selected — there is only one.
"$ROOT/scripts/dev-cluster.sh" up

# 3. Tear the cluster down + clean the scene when the client window closes / on Ctrl-C.
cleanup() {
    "$ROOT/scripts/dev-cluster.sh" down >/dev/null 2>&1 || true
    rm -rf "$SCENE_DIR" || true
}
trap cleanup EXIT

# 4. Open the interactive client window pointed at the world scene (WASD + mouse fly the dot; the
#    planets ORBIT because the shard authors their poses each tick and ships them over the realm
#    feed — the static file only seeds the boxes).
echo "launching the client window — WASD + mouse to fly the dot; the planets ORBIT. Ctrl-C to stop."
"$ROOT/scripts/client.sh" --window --name visual --realm-boxes "$VD_WORLD_SCENE"
