#!/usr/bin/env bash
# worktree-remove.sh — remove a worktree made by worktree-new.sh, AND its slot's realm stores.
#
#   scripts/worktree-remove.sh <name>
#
# `git worktree remove` alone leaves the worktree's slot stores in the temporary folder (the removed perf worktree left
# slot 8's that way). This script:
#   1. refuses while any process holds a file in the worktree's realm stores (its cluster runs);
#   2. runs `git worktree remove` WITHOUT --force, so a worktree with uncommitted or untracked work is refused and kept;
#   3. only then deletes the slot's realm stores (the worktree's own solves and berths; the checkout it was cloned from
#      keeps its own). The branch stays.
#
# Example: `scripts/worktree-remove.sh weather` — the weather worktree, its build folder and slot stores are gone; the
# branch `weather` stays for a merge.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ $# -eq 1 && "$1" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "usage: scripts/worktree-remove.sh <name>" >&2; exit 2; }
NAME="$1"

fail() { echo "worktree-remove.sh: $*" >&2; exit 1; }

MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
DEST="$MAIN/.claude/worktrees/$NAME"
[[ -d "$DEST" ]] || fail "no worktree at $DEST"
[[ "$(cd "$DEST" && pwd)" != "$ROOT" ]] || fail "run this from another checkout than the one it removes"

PROFILE="${VD_BUILD:-debug}"
TARGET_DIR="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')"
SLOT_BIN="$TARGET_DIR/$PROFILE/vd-slot"
[[ -x "$SLOT_BIN" ]] || fail "no slot tool at $SLOT_BIN: cargo build -p vd-bins --bin vd-slot"
SLOT="$(eval "$("$SLOT_BIN" --worktree "$DEST")"; echo "$VD_SLOT")"
STORES_OUT="$("$SLOT_BIN" --slot "$SLOT" --stores)" || fail "the slot tool has no --stores: build it again"
STORES="$(eval "$STORES_OUT"; echo "$VD_REALM_STORE_DIR")"

if [[ -d "$STORES" ]]; then
    holders="$(lsof +D "$STORES" 2>/dev/null | tail -n +2 || true)"
    [[ -z "$holders" ]] || fail "these processes hold files in $STORES (stop the worktree's cluster first):
$holders"
fi

git -C "$ROOT" worktree remove "$DEST"

if [[ -d "$STORES" ]]; then
    echo "worktree-remove.sh: deleting slot $SLOT's realm stores at $STORES" >&2
    rm -rf -- "$STORES"
fi
echo "worktree-remove.sh: removed $DEST (slot $SLOT); the branch stays"
