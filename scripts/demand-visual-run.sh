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

# ALWAYS build the SERVER side. This used to be `[[ -x "$TARGET/vd-devcluster" ]] ||` — skip if the binary
# merely EXISTS — which silently flew whatever server was built last, however old. That is the worst
# possible failure mode for a fix-then-fly loop: the change appears to do nothing, and the obvious
# conclusion (the fix is wrong) is the wrong one. It nearly cost a day on 2026-08-06, and only did not
# because the binaries happened to be current from an unrelated test run. Cargo is already incremental —
# when nothing changed this is a no-op, so the skip bought nothing and risked everything.
cargo build --manifest-path "$ROOT/Cargo.toml" -p vd-bins

# Pre-build the WINDOWED client with progress VISIBLE (client.sh builds it with `-q`, which hides the
# heavy Bevy first-build and makes it look frozen). Doing it here with output means you see the compile;
# once done, client.sh's own `-q` build is a no-op and the window opens immediately.
echo "building the windowed client (Bevy — the FIRST build is heavy, ~15-25 min on a slow box; progress below)…"
cargo build --manifest-path "$ROOT/Cargo.toml" -p vd-bins --bin client --features dev-control,render

# Boot the demand cluster in VISUAL scale. spawn_node inherits the parent env (no env_clear), so the
# DEMAND-SPAWNED home shard's resolve_universe_scale reads VD_UNIVERSE_SCALE=visual and authors the
# window-friendly orbiting system when the reconciler spins it up on your login.
export VD_UNIVERSE_SCALE=visual
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
"$ROOT/scripts/client.sh" --window --name demand-walker
