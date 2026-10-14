#!/usr/bin/env bash
# worktree-new.sh — a new git worktree that STARTS WITH THIS CHECKOUT'S SOLVED WORLD.
#
#   scripts/worktree-new.sh <name> [--branch <branch>] [--from <ref>] [--no-target]
#
# A worktree gets its own dev-cluster slot, and a new slot has empty realm stores: its first login solves the home
# planet again (2 245 s on the shard's threads, MEASURED 2026-10-12), every far planet (hours each) and every fine
# tile. The solved ground is a pure result of the seed, the body and the generator's version words (SL10), so this
# script gives the new slot a COPY of this checkout's stores instead:
#
#   1. It refuses while any process holds a file in this checkout's realm stores (a store copied while a shard writes
#      it is torn; a far planet's first solve keeps its shard alive after the login ends, ruling B91), or while a
#      cargo build holds this checkout's build folder.
#   2. `git worktree add` at <main checkout>/.claude/worktrees/<name>, on a new branch (default <name>) from <ref>
#      (default this checkout's HEAD).
#   3. It clones this checkout's realm stores into the new worktree's slot, and this checkout's build folder into the
#      new worktree (skipped with --no-target). The clones are the Mac file system's copy-on-write clones: they take
#      no disk until one side changes a block.
#
# The copies are independent afterwards: a tile one side solves later, the other side solves again; a generator
# change in the new worktree (a new key generation or fine version) solves again in ITS stores only. To take this
# checkout's newer ground later, remove the worktree's slot stores and clone again (the worktree's own berths go too).
#
# Example: `scripts/worktree-new.sh weather` — the weather worktree's first login lands aboard hull 17 on the home
# planet's hills, and the planet's shard ships the artifact and every 128 m and 16 m tile from the cloned store.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() { echo "usage: scripts/worktree-new.sh <name> [--branch <branch>] [--from <ref>] [--no-target]" >&2; exit 2; }

NAME=""
BRANCH=""
FROM="HEAD"
CLONE_TARGET=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch) [[ $# -ge 2 ]] || usage; BRANCH="$2"; shift 2 ;;
        --from) [[ $# -ge 2 ]] || usage; FROM="$2"; shift 2 ;;
        --no-target) CLONE_TARGET=0; shift ;;
        -*) usage ;;
        *) [[ -z "$NAME" ]] || usage; NAME="$1"; shift ;;
    esac
done
[[ -n "$NAME" ]] || usage
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "worktree-new.sh: the name '$NAME' must be letters, digits, '.', '_' or '-'" >&2; exit 2; }
BRANCH="${BRANCH:-$NAME}"

fail() { echo "worktree-new.sh: $*" >&2; exit 1; }

MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
DEST="$MAIN/.claude/worktrees/$NAME"
[[ ! -e "$DEST" ]] || fail "$DEST exists"
! git -C "$ROOT" show-ref --verify --quiet "refs/heads/$BRANCH" || fail "the branch '$BRANCH' exists"
git -C "$ROOT" rev-parse --verify --quiet "$FROM^{commit}" >/dev/null || fail "'$FROM' names no commit"

# The slot tool, from this checkout's build (the profile as dev-cluster.sh reads it).
PROFILE="${VD_BUILD:-debug}"
case "$PROFILE" in
    debug) PROFILE_FLAG="" ;;
    release) PROFILE_FLAG="--release" ;;
    *) fail "VD_BUILD must be 'debug' or 'release', got '$PROFILE'" ;;
esac
TARGET_DIR="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')"
SLOT_BIN="$TARGET_DIR/$PROFILE/vd-slot"
if [[ ! -x "$SLOT_BIN" ]]; then
    echo "worktree-new.sh: building the slot tool…" >&2
    # shellcheck disable=SC2086 — an empty flag must vanish, not become an empty argument.
    cargo build --manifest-path "$ROOT/Cargo.toml" -p vd-bins --bin vd-slot $PROFILE_FLAG
fi
slot_of() { (eval "$("$SLOT_BIN" --worktree "$1")"; echo "$VD_SLOT"); }
stores_of() {
    local out
    out="$("$SLOT_BIN" --slot "$1" --stores)" \
        || fail "the slot tool has no --stores: build it again (cargo build -p vd-bins --bin vd-slot $PROFILE_FLAG)"
    (eval "$out"; echo "$VD_REALM_STORE_DIR")
}

# The slot a worktree's dev-cluster.sh derives from its own root path: the same string here, before the tree exists.
SRC_SLOT="$(slot_of "$ROOT")"
DEST_SLOT="$(slot_of "$DEST")"
[[ "$SRC_SLOT" != "$DEST_SLOT" ]] || fail "'$NAME' hashes to this checkout's own slot $SRC_SLOT: choose another name"
SRC_STORES="$(stores_of "$SRC_SLOT")"
DEST_STORES="$(stores_of "$DEST_SLOT")"
if [[ -e "$DEST_STORES" ]]; then
    fail "slot $DEST_SLOT already has stores at $DEST_STORES (an older worktree's?): look at them, remove them, then run again"
fi

# 1. Nothing may write what is copied.
if [[ -d "$SRC_STORES" ]]; then
    holders="$(lsof +D "$SRC_STORES" 2>/dev/null | tail -n +2 || true)"
    [[ -z "$holders" ]] || fail "these processes hold files in $SRC_STORES (stop the cluster first):
$holders"
fi
if [[ "$CLONE_TARGET" == 1 ]]; then
    [[ -z "${CARGO_TARGET_DIR:-}" ]] || fail "CARGO_TARGET_DIR is set: every worktree shares that build folder; run with --no-target"
    for lock in "$TARGET_DIR"/debug/.cargo-lock "$TARGET_DIR"/release/.cargo-lock; do
        if [[ -e "$lock" ]] && [[ -n "$(lsof "$lock" 2>/dev/null | tail -n +2 || true)" ]]; then
            fail "a cargo build holds $lock: wait for it to end"
        fi
    done
    # A clone needs one volume; across volumes `cp -c` silently makes a full copy.
    [[ "$(stat -f %d "$TARGET_DIR")" == "$(stat -f %d "$MAIN")" ]] \
        || fail "the build folder $TARGET_DIR is on another volume than $MAIN: run with --no-target"
fi

# 2. The worktree. From here a failure leaves a worktree without its clones: say so, and how to start again.
git -C "$ROOT" worktree add -b "$BRANCH" "$DEST" "$FROM"
trap 'echo "worktree-new.sh: STOPPED after the worktree was added: $DEST (branch $BRANCH) holds no complete clone;
  start again with: git worktree remove --force \"$DEST\" && git branch -D \"$BRANCH\" && rm -rf \"$DEST_STORES\"" >&2' ERR

# 3. The clones.
if [[ -d "$SRC_STORES" ]]; then
    echo "worktree-new.sh: cloning the realm stores of slot ${SRC_SLOT} into slot ${DEST_SLOT}…" >&2
    mkdir -p "$(dirname "$DEST_STORES")"
    cp -c -R "$SRC_STORES" "$DEST_STORES"
else
    echo "worktree-new.sh: slot $SRC_SLOT has no realm stores at $SRC_STORES: the new slot starts empty" >&2
fi
if [[ "$CLONE_TARGET" == 1 ]]; then
    echo "worktree-new.sh: cloning the build folder (copy-on-write; this can take a few minutes)…" >&2
    cp -c -R "$TARGET_DIR" "$DEST/target"
    # The GPU crate's own build folder, when this checkout has one.
    if [[ -d "$ROOT/crates/recipe-gpu/target" ]]; then
        cp -c -R "$ROOT/crates/recipe-gpu/target" "$DEST/crates/recipe-gpu/target"
    fi
fi

trap - ERR

cat <<EOF

The worktree:   $DEST  (branch $BRANCH from $FROM)
Its slot:       $DEST_SLOT  (this checkout's: $SRC_SLOT)
Its stores:     $DEST_STORES
Build it first: cd "$DEST" && cargo build -p vd-bins   (add --release for VD_BUILD=release; the cloned binaries are
                this checkout's, and dev-cluster.sh never builds a binary that exists)
Start it:       scripts/dev-cluster.sh up --demand
Remove it:      scripts/worktree-remove.sh $NAME
EOF
