#!/usr/bin/env bash
# crossing-playground.sh — LAUNCH the Visual Crossing Playground (V4): a dual-shard dev cluster with a
# walk-into crossing trigger + a WINDOWED client that draws two colored realm boxes. WASD-walk your dot
# from box A across the boundary into box B and watch the PROVEN cross-shard transfer re-home it live.
#
#   scripts/crossing-playground.sh          # build, boot the dual cluster, open the window
#
# Closing the window (or Ctrl-C) tears the cluster down. Requires a working GPU (Bevy/wgpu window).
# The fixtures (walk-into trigger + two-box scene) are the SAME `vd_bins::crossing_playground` geometry
# the `render_crossing_smoke` gate proves, so what you see is exactly what the test asserts in pixels.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')/debug"

echo "crossing-playground: building the node binaries + the windowed client…" >&2
cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins
cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins --bin client --features dev-control,render

# Emit the single-sourced playground fixtures (walk-into shard trigger + the client's two-box scene).
FIXDIR="$(mktemp -d "${TMPDIR:-/tmp}/vd-crossing-XXXXXX")"
eval "$("$TARGET/vd-devcluster" emit-crossing-fixtures "$FIXDIR")" # → VD_CROSSING_TRIGGER + VD_CROSSING_SCENE

# Bring up the DUAL cluster with the SOURCE shard's crossing trigger injected via the launcher hook
# (VD_DEVCLUSTER_BOUNDARIES). Tear it down (and drop the fixtures) on any exit.
trap '"$ROOT/scripts/dev-cluster.sh" down >/dev/null 2>&1 || true; rm -rf "$FIXDIR"' EXIT
VD_DEVCLUSTER_BOUNDARIES="$VD_CROSSING_TRIGGER" "$ROOT/scripts/dev-cluster.sh" up --dual

cat >&2 <<'BANNER'

  ┌─ Visual Crossing Playground ──────────────────────────────────────────────
  │ A window opens with TWO translucent colored boxes: box A (System 7) at the
  │ origin, box B (System 8) off to the +X side. Your dot spawns inside box A.
  │ WALK it (WASD / arrows, mouse to look) toward box B — as it crosses the
  │ boundary the shard transfer re-homes its authority (System 7 → System 8)
  │ seamlessly; the dot keeps rendering, now computed by the other shard.
  │ Close the window (or Ctrl-C here) to tear the cluster down.
  └───────────────────────────────────────────────────────────────────────────

BANNER

# The windowed client (foreground) — WASD/mouse feed the SAME input mailbox vdctl drives.
"$ROOT/scripts/client.sh" --window --name walker --realm-boxes "$VD_CROSSING_SCENE"
