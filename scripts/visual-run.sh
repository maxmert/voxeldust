#!/usr/bin/env bash
# visual-run.sh — FA-5 visual-universe test (VU-0): boot the VISUAL universe cluster + open an
# interactive client window showing the orbiting BLUE planet spheres. Reuses dev-cluster.sh (robust
# bring-up) + client.sh --window. Closing the window (or Ctrl-C) tears the cluster down.
#
# Usage:
#   scripts/visual-run.sh          # boot visual System 7 + orbiting planets, open the client window
#
# What you SEE: a translucent System-7 sphere inside the Galaxy box, with N BLUE planet spheres ORBITING
# the star. WASD + mouse fly the dot. The planets orbit because the shard AUTHORS their poses each tick
# (VD_UNIVERSE_SCALE=visual) and SHIPS them over the realm feed; the client overlays the live poses onto
# the boot scene. No meshes — realm spheres only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Resolve the real target dir (honours CARGO_TARGET_DIR / .cargo config).
TARGET="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')/debug"

# vd-devcluster carries the emit-visual-fixtures subcommand this launcher needs.
[[ -x "$TARGET/vd-devcluster" ]] || cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins

# 1. Dump the VISUAL regions.json for the client's --realm-boxes — SINGLE-SOURCED with what the
#    VD_UNIVERSE_SCALE=visual shard plants (realm_regions_for_config(seed, visual_scale())), so the client
#    draws EXACTLY the system the shard authors + ships.
SCENE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vd-visual-scene.XXXXXX")"
eval "$("$TARGET/vd-devcluster" emit-visual-fixtures "$SCENE_DIR")" # sets VD_VISUAL_SCENE
echo "visual scene → $VD_VISUAL_SCENE"

# 2. Boot the cluster with the shard in VISUAL scale. spawn_node inherits the parent env (no env_clear),
#    so the System-7 shard's resolve_universe_scale reads VD_UNIVERSE_SCALE=visual and authors the orbiting
#    planets (the gateway/orchestrator inherit it too but ignore it).
# THE WORLD IS NO LONGER SELECTED, so there is nothing to export here. This line used to set a scale,
# and a live cluster was read process by process with the orchestrator on one world and its own gateway on
# another — from THIS script, in one launch. A knob that exists can be set twice; the fix was to delete it.
"$ROOT/scripts/dev-cluster.sh" up

# 3. Tear the cluster down + clean the scene when the client window closes / on Ctrl-C.
cleanup() {
    "$ROOT/scripts/dev-cluster.sh" down >/dev/null 2>&1 || true
    rm -rf "$SCENE_DIR" || true
}
trap cleanup EXIT

# 4. Open the interactive client window pointed at the visual scene (WASD + mouse fly the dot).
echo "launching the client window — WASD + mouse to fly the dot; the planets ORBIT as blue spheres. Ctrl-C to stop."
"$ROOT/scripts/client.sh" --window --name visual --realm-boxes "$VD_VISUAL_SCENE"
