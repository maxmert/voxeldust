#!/usr/bin/env bash
# demand-visual-run.sh — fly the DEMAND LOOP in a window (VU visual-arc, first watchable increment).
#
# Unlike visual-run.sh (a STATIC single system whose spheres are fed from a file), this boots the
# DEMAND cluster: orchestrator + gateway only, NO pre-booked world. The ONLY way a realm exists is the
# armed reconciler spinning one up on your login / your AoI demand — the exact cluster the headless
# rlm_demand_login + demand-walk proofs build in-harness, now with a HUMAN in a window instead of vdctl.
# The client draws its scene from the SERVER STREAM — the COMPOSED picture the gateway stacks per
# observer (window lane, 2026-08-16): a scene level on join + deltas as realms enter/leave your AoI,
# every row already in your own realm's frame. The client composes NOTHING. There is no boot file at
# all any more (`--realm-boxes` and its whole loading path are deleted), and no scale to select:
# THE world is the only world.
#
# Usage:
#   scripts/demand-visual-run.sh          # boot the demand cluster, open the client window
#   scripts/demand-visual-run.sh --fast   # ^ with Bevy linked as a shared library (faster relink)
#
# What you SEE: your HOME system streams in on login (it is demand-spawned the moment you connect —
# expect a ~1-2s warm-up while the home shard boots). WASD + mouse fly the dot. As you move, realms your
# AoI reaches stream in and ones you leave evict — the demand loop, live. Since the window lane landed
# you also see: the OTHER systems as points of light placed by the galaxy (no server runs for them),
# dormant planets shining by reflected starlight, the body of the realm you are INSIDE drawn around you,
# and — flying at another system — the point of light growing continuously into a live self-drawn system
# while the one behind shrinks back to a dot. That is the warp, and it is gated in pixels
# (`just warp-pixels`). Closing the window (or Ctrl-C) tears the cluster down.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')/debug"

# `--fast` links Bevy as ONE shared library instead of statically — the edit-rebuild-fly loop's
# build-speed knob. It changes NOTHING about the world, the cluster or the client's behaviour;
# only how the client binary is linked.
#
# The pre-build below and the client.sh launch at the bottom MUST agree on this: they are two
# cargo invocations of the SAME binary, and a mismatched feature set means each one undoes the
# other's Bevy build every single run. That is why one flag drives both, rather than being
# passed by hand in two places.
FAST=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast) FAST=1; shift ;;
        *) echo "demand-visual-run.sh: unexpected argument '$1'" >&2; exit 1 ;;
    esac
done

CLIENT_FEATURES="dev-control,render"
CLIENT_ARGS=(--window --name demand-walker)
if [[ "$FAST" == "1" ]]; then
    CLIENT_FEATURES="dev-control,render-dylib"
    CLIENT_ARGS+=(--fast)
fi

# ALWAYS build the SERVER side. This used to be `[[ -x "$TARGET/vd-devcluster" ]] ||` — skip if the binary
# merely EXISTS — which silently flew whatever server was built last, however old. That is the worst
# possible failure mode for a fix-then-fly loop: the change appears to do nothing, and the obvious
# conclusion (the fix is wrong) is the wrong one. It nearly cost a day on 2026-08-06, and only did not
# because the binaries happened to be current from an unrelated test run. Cargo is already incremental —
# when nothing changed this is a no-op, so the skip bought nothing and risked everything.
cargo build --manifest-path "$ROOT/Cargo.toml" -p vd-bins

# Pre-build the WINDOWED client with progress visible. (This used to compensate for client.sh building
# with `-q`; that flag is gone — it also hid "Blocking waiting for file lock on build directory", which
# turned a build QUEUED behind another cargo into a silent stall indistinguishable from compiling.) The
# pre-build stays: it keeps the heavy compile above the cluster bring-up, so a long first build does not
# sit behind a booted cluster waiting on it.
echo "building the windowed client (Bevy — the FIRST build is heavy, ~15-25 min on a slow box; progress below)…"
cargo build --manifest-path "$ROOT/Cargo.toml" -p vd-bins --bin client --features "$CLIENT_FEATURES"

# Boot the demand cluster on THE world (the demand-spawned shards boot `UniverseConfig::world`;
# `resolve_universe_scale` / `VD_UNIVERSE_SCALE` were deleted with the scale knob, Stage-C batch 1).
# ★THROWAWAY (test instrument, owner-ordered 2026-08-20): cross the world faster than the
# three-minute traverse policy allows. It multiplies the CRUISE ceiling of whatever realm you are
# inside — and NOTHING else. The approach governor keeps its lawful values, so boundary crossings
# behave exactly as they will in the real game: you still arrive slowly and cannot fly through
# anything. Override it per run, e.g. `VD_TEST_OVERDRIVE=200 scripts/demand-visual-run.sh`.
# 1.0 is the law; this whole knob retires with the keyboard throttle when the ship realm lands.
# spawn_node inherits this environment, so every demand-spawned shard reads the same number.
export VD_TEST_OVERDRIVE="${VD_TEST_OVERDRIVE:-64}"
echo "cruise overdrive: ${VD_TEST_OVERDRIVE}x (1 = the lawful three-minute traverse)"

# THE WORLD IS NO LONGER SELECTED, so there is nothing to export here. This block used to set a scale,
# and a live cluster was read process by process with the orchestrator on one world and its own gateway on
# another — from THIS script, in one launch. A knob that exists can be set twice; the fix was to delete it.

# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ THROWAWAY — DELETE WITH THE TINY WORLD. Not a feature, and deliberately not one.                  ║
# ║                                                                                                  ║
# ║ There is nothing to aim at, BY DESIGN: a star is invisible until it is close enough to wake, and  ║
# ║ the neighbours are placed beyond that on purpose so the waking can be watched. In the real game   ║
# ║ knowing where anything is will be a MECHANIC — charts, scanners, an earned HUD — never something  ║
# ║ the server volunteers. So this prints coordinates instead of building a marker stream that would  ║
# ║ give that information away for free and then have to be taken back.                              ║
# ║                                                                                                  ║
# ║ Fly along X and watch the position readout in the corner.                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
cat <<NAV

  ── flying THE world (seed 2298 — your chosen home) ────────────────────
    controls
      W A S D        move             mouse   look
      SPACE / CTRL   up / down
      [  and  ]      throttle tier DOWN / UP   (10 tiers; you start at 10)

    the throttle
      Full throttle crosses whatever realm you are INSIDE in three minutes
      — at every level, from a planet's own space to the whole galaxy. The
      tiers are spaced by RATIO, so the top two are your travel speeds and
      the lower ones are for close work. The ship gathers way and loses it
      instead of starting and stopping dead.

      This run also carries a test overdrive of ${VD_TEST_OVERDRIVE}x on the CRUISE
      ceiling, so you can get places quickly. It does NOT touch the approach
      governor: near anything you are still slowed to the lawful speed, so
      you still arrive gently and cannot fly through a world.

    where you are
      A yellow sun, 1.03 solar masses. Your home planet is 6,516 km across
      with 10.2 m/s^2 of gravity, and it holds its air. Nine planets share
      the system; 22 moons among them. Both neighbour stars sit 0.238 light
      years away — a lawful warp of ~142 s, faster with the overdrive.

    what to watch for
      · the sun draws as a BODY with its own colour, not a dot
      · planets keep their own picture as you leave, shrinking smoothly
      · moons draw too — they are realms like everything else
      · aim at a neighbour star and hold forward: the OTHER star should
        sweep sideways across your view as you travel. That is real
        parallax from real 3-D placement, ~23 degrees of it.
      · nothing should pop, blink, or vanish at any handover
  ───────────────────────────────────────────────────────────────────────

NAV
"$ROOT/scripts/dev-cluster.sh" up --demand

# Tear the cluster down when the client window closes / on Ctrl-C. `down` kills the RECORDED node groups;
# a DEMAND-SPAWNED shard is spawned into its own group at runtime and can outlive that record, so also
# best-effort reap any node process from THIS worktree's target dir (path-scoped — never touches another
# worktree's cluster).
cleanup() {
    "$ROOT/scripts/dev-cluster.sh" down >/dev/null 2>&1 || true
    pkill -f "$TARGET/vd-shard" >/dev/null 2>&1 || true
    pkill -f "$TARGET/vd-orchestrator" >/dev/null 2>&1 || true
    pkill -f "$TARGET/vd-gateway" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Open the interactive client window. NO --realm-boxes: the scene STREAMS from the server (RealmRegistry
# on join). --window builds with the `render` feature (Bevy) — the FIRST such build is heavy (incremental
# after). WASD + mouse fly the dot; the dev-control listener stays up so vdctl can also drive/inspect it.
echo "launching the client window on the DEMAND cluster — your home system streams in on login (give the"
echo "home shard ~1-2s to spawn); WASD + mouse to fly. Ctrl-C or close the window to stop."
"$ROOT/scripts/client.sh" "${CLIENT_ARGS[@]}"
