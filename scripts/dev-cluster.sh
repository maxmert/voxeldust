#!/usr/bin/env bash
# dev-cluster.sh — the agent-facing entry to the LOCAL-PROCESS dev cluster (HR6).
#
#   scripts/dev-cluster.sh up              # spawn orchestrator+gateway+shard, wait ready
#   scripts/dev-cluster.sh status
#   scripts/dev-cluster.sh env             # print the eval-able cluster contract
#   scripts/dev-cluster.sh down
#
# The slot DEFAULTS to a stable per-worktree value (so this worktree and another
# never collide on slot 0); pass `--slot N` to override. ALL bring-up logic lives
# in the tested Rust launcher (crates/bins/src/bin/vd-devcluster.rs), which uses
# the canonical DevPortScheme (vd-devproto) — no port is hand-computed in bash.
# Distinct from the root k3d dev-cluster (this one needs no Docker/kubectl).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve the real target dir (honours CARGO_TARGET_DIR / .cargo config), not a
# hardcoded target/debug.
TARGET="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')/debug"

# The launcher spawns its sibling node binaries, so all of them must exist.
need_build=0
for bin in vd-devcluster vd-orchestrator vd-gateway vd-shard vd-slot; do
    [[ -x "$TARGET/$bin" ]] || need_build=1
done
if [[ "$need_build" == 1 ]]; then
    echo "dev-cluster.sh: building the node binaries…" >&2
    # NOT `-q`, deliberately. Cargo takes an EXCLUSIVE lock on the build directory, so a
    # second cargo anywhere on the machine (an agent session's check/test, another script)
    # makes this call sit and wait — and `-q` suppresses the one line that says so
    # ("Blocking waiting for file lock on build directory"). That turned a queued build
    # into a silent 30-minute stall indistinguishable from compiling. Measured 2026-08-11:
    # a 33m52s client build that wrote ZERO artifacts because it was queued the whole time.
    cargo build --manifest-path "$ROOT/Cargo.toml" -p vd-bins
fi

# Default the slot to this worktree's stable value unless the caller set --slot.
if [[ "$*" != *--slot* ]]; then
    eval "$("$TARGET/vd-slot" --worktree "$ROOT")" # sets VD_SLOT (sh-quoted)
    set -- "$@" --slot "$VD_SLOT"
fi

exec "$TARGET/vd-devcluster" "$@"
