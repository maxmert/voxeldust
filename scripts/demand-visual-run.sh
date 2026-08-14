#!/usr/bin/env bash
# demand-visual-run.sh — fly the DEMAND LOOP in a window (VU visual-arc, first watchable increment).
#
# Unlike visual-run.sh (a STATIC single system whose spheres are fed from a file), this boots the
# DEMAND cluster: orchestrator + gateway only, NO pre-booked world. The ONLY way a realm exists is the
# armed reconciler spinning one up on your login / your AoI demand — the exact cluster the headless
# rlm_demand_login + demand-walk proofs build in-harness, now with a HUMAN in a window instead of vdctl.
# The client draws its scene from the SERVER STREAM (RealmRegistry on join + RealmSceneDelta as realms
# enter/leave your AoI) — NO --realm-boxes file. Scale = visual (window-friendly).
#
# Usage:
#   scripts/demand-visual-run.sh          # boot the demand cluster, open the client window
#   scripts/demand-visual-run.sh --fast   # ^ with Bevy linked as a shared library (faster relink)
#
# What you SEE (and what's still owed): your HOME system streams in on login (it is demand-spawned the
# moment you connect — expect a ~1-2s warm-up while the home shard boots). WASD + mouse fly the dot. As
# you move, realms your AoI reaches stream in and ones you leave evict — the demand loop, live. This is
# the FIRST VU-arc window on the demand loop; the surrounding STAR FIELD, warp targeting, and the seamless
# warp fly-by are the following slices (scripts/visual_universe_arc_plan.md, VU-3..VU-7). Closing the
# window (or Ctrl-C) tears the cluster down.
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
cat <<'NAV'

  ── flying this test world ─────────────────────────────────────────────
    controls
      W A S D        move            mouse   look
      SPACE / CTRL   up / down       SHIFT   hold for WARP speed

    two speeds, because there are two scales
      cruise (no shift)   15 m/s   — crossing a star system takes ~20 s
      warp   (hold shift) 500 m/s  — crossing to the next star ~24 s

    where the stars are
      you start at   x = 0
      a star sits at x = +12031
      another at     x = -12031
      a star wakes when you are within ~11460 m of it, so a neighbour
      lights up shortly after you set off and grows as you close.

    what to watch for
      planets are visible from ANYWHERE inside their system now (~318 m
      reach against a 150 m system), so arriving at a star should show
      you a populated system, not an empty box. they should fade in as
      small dots and grow, not pop in at full size.
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
