#!/usr/bin/env bash
# crossing-playground.sh — LAUNCH the Visual Crossing Playground (NODE-PER-REALM seed forest): a SIX-shard
# dev cluster, ONE shard per realm — System 7, Planet 7, Station 7, Area 7, the GALAXY between-space, and
# System 8 — each hosting its realm by coordinates (NO co-hosting), plus a WINDOWED node-agnostic pure-
# renderer client that draws the seed forest — the GALAXY containing box with System 7 / System 8 inside it.
# WASD-walk your dot +X through the FULL chain and back and watch the coordinate-driven re-home flip the HUD
# realm live: System 7 → Planet 7 → Area 7 → Planet 7 → System 7 → Galaxy → System 8 → back — always inside
# the Galaxy box, never orphaned. Because each realm is its OWN node, EVERY crossing is a uniform CROSS-NODE
# saga (the source==dest co-hosted case never arises).
#
#   scripts/crossing-playground.sh          # build, boot the --forest cluster, open the window
#
# Closing the window (or Ctrl-C) tears the cluster down. Requires a working GPU (Bevy/wgpu window).
# The scene (regions.json) is the SAME `worldgen::realm_regions_for` geometry the shards' detector
# evaluates AND the `triple_cluster_crossing_smoke` GPU gate proves — so what you walk is exactly the test.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')/debug"

echo "crossing-playground: building the node binaries + the windowed client…" >&2
cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins
cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins --bin client --features dev-control,render

# Emit the SEED-FOREST scene (regions.json — the canonical forest, INCLUDING the Galaxy containing box).
FIXDIR="$(mktemp -d "${TMPDIR:-/tmp}/vd-crossing-XXXXXX")"
eval "$("$TARGET/vd-devcluster" emit-seed-fixtures "$FIXDIR")" # → VD_SEED_SCENE

# Bring up the --forest NODE-PER-REALM cluster: System 7 (login) + Planet 7 + Station 7 + Area 7 + Galaxy
# (between-space) + System 8, each on its OWN node booting its seed neighbourhood by coordinates — NO authored
# boundary override, NO co-hosting. Every child realm has its own shard, so walking into one is a uniform
# CROSS-NODE re-home that never orphans the dot. Tear the cluster down (and drop the scene) on any exit.
trap '"$ROOT/scripts/dev-cluster.sh" down >/dev/null 2>&1 || true; rm -rf "$FIXDIR"' EXIT
"$ROOT/scripts/dev-cluster.sh" up --forest

cat >&2 <<'BANNER'

  ┌─ Visual Crossing Playground (NODE-PER-REALM seed forest) ──────────────────
  │ A window opens with the GALAXY as a big translucent CONTAINING box, with
  │ System 7 (at the origin) and System 8 (off to the +X side) nested inside it.
  │ Your dot spawns inside System 7. WALK it (WASD / arrows, mouse to look) +X:
  │   System 7 → Planet 7 → Area 7 → Planet 7 → System 7 → Galaxy → System 8.
  │ The HUD realm label flips as you cross — authority re-homes by COORDINATES,
  │ node-agnostically, EACH realm on its own shard (every crossing a cross-node
  │ saga); the dot keeps rendering and is ALWAYS inside the Galaxy box (never
  │ orphaned). Close the window (or Ctrl-C here) to tear it down.
  └───────────────────────────────────────────────────────────────────────────

BANNER

# The windowed client (foreground) — WASD/mouse feed the SAME input mailbox vdctl drives. It renders the
# seed forest (Galaxy containing box + the systems) and is a NODE-AGNOSTIC pure renderer: it never learns
# which shard owns the dot, it just draws the authoritative coordinates the server streams.
"$ROOT/scripts/client.sh" --window --name walker --realm-boxes "$VD_SEED_SCENE"
