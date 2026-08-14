#!/usr/bin/env bash
# crossing-playground.sh — LAUNCH the human crossing playground on THE WORLD, demand-driven: an
# orchestrator + gateway with NO shard pre-booked (`up --demand`), plus a WINDOWED node-agnostic
# pure-renderer client. Your login demands the home star system into existence; every realm you
# approach spins up AHEAD of you and is reaped behind you — so a free-flying human can NEVER strand
# on a realm no shard hosts (a static cluster hosts a SUBSET of THE world, and an off-corridor
# crossing there is a PERMANENT drop with no reply — the source latch stays standing; D-WORLD-2).
#
#   scripts/crossing-playground.sh          # build, boot the --demand cluster, open the window
#
# Closing the window (or Ctrl-C) tears the cluster down. Requires a working GPU (Bevy/wgpu window).
# NO --realm-boxes file: the scene STREAMS — the gateway ships the realm registry and the live realm
# poses, so what you see is exactly what the shards author, with nothing loaded from disk (SL5).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "crossing-playground: building the node binaries + the windowed client…" >&2
cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins
cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins --bin client --features dev-control,render

# Bring up the DEMAND cluster: orchestrator + gateway only. The armed reconciler spins the home
# realm up on your login, and every realm your visibility demands after that — warp as a
# consequence of movement, never a mode. Tear it down on any exit.
trap '"$ROOT/scripts/dev-cluster.sh" down >/dev/null 2>&1 || true' EXIT
"$ROOT/scripts/dev-cluster.sh" up --demand

cat >&2 <<'BANNER'

  ┌─ Crossing Playground (THE world, demand-driven) ───────────────────────────
  │ A window opens at the HOME STAR: a translucent 150 m system shell with five
  │ planets ORBITING inside it. WASD + mouse fly your dot; SPACE/CTRL go up and
  │ down; hold LEFT SHIFT to BOOST (full 500 m/s — release it to cruise at
  │ 15 m/s for manoeuvring). The proven corridor is the ±Z POLE (straight "up"
  │ or "down" out of the orbital plane):
  │   · fly −Z ~220 m  → out of the home shell; the HUD label flips to the
  │     galaxy between-space (the re-home is a directory CAS, not a teleport);
  │   · fly back to −60 m → the label flips home again;
  │   · park by a planet's path and let it sweep over you → you board it;
  │   · the sibling star is ~12.8 km down the ±X ring — hold BOOST ~26 s and
  │     its planets stream in AHEAD of you before you arrive.
  │ Realms spin up ahead of your visibility and are reaped behind you, so free
  │ flight anywhere is safe — nothing you can reach is unhosted. Close the
  │ window (or Ctrl-C here) to tear it down.
  └───────────────────────────────────────────────────────────────────────────

BANNER

# The windowed client (foreground) — WASD/mouse feed the SAME input mailbox vdctl drives. It is a
# NODE-AGNOSTIC pure renderer: it never learns which shard owns the dot, it draws the streamed
# scene + the authoritative coordinates the server ships.
"$ROOT/scripts/client.sh" --window --name walker
