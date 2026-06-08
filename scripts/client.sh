#!/usr/bin/env bash
# client.sh — launch ONE dev-control client window against a running dev cluster
# (HR6). The dev-control + client-QUIC ports follow the canonical per-slot/per-agent
# scheme (vd-devproto via `vd-slot`), so several clients per worktree never collide
# — what the P2 two-client visual scenario (client2 screenshots client1 crossing a
# boundary) needs.
#
#   scripts/client.sh --name walker  --agent-index 0
#   scripts/client.sh --name watcher --agent-index 1
#
# The slot DEFAULTS to this worktree's stable value (matching what
# `dev-cluster.sh up` derived); pass `--slot N` to override. The real wgpu client
# binary lands in P1.5 Slice 3; until then this resolves and prints the exact
# conventions it WILL use (and fails loud if the cluster is not up).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')/debug"

SLOT=""
AGENT=0
NAME="client"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slot) SLOT="${2:?--slot needs a value}"; shift 2 ;;
        --agent-index) AGENT="${2:?--agent-index needs a value}"; shift 2 ;;
        --name) NAME="${2:?--name needs a value}"; shift 2 ;;
        *) echo "client.sh: unexpected argument '$1'" >&2; exit 1 ;;
    esac
done

[[ -x "$TARGET/vd-slot" && -x "$TARGET/vd-devcluster" ]] \
    || cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins

# Default the slot to this worktree's stable value (same derivation dev-cluster.sh used).
if [[ -z "$SLOT" ]]; then
    eval "$("$TARGET/vd-slot" --worktree "$ROOT")" # sets VD_SLOT
    SLOT="$VD_SLOT"
fi

# The cluster contract (gateway addr, login key, trust dir) + this client's ports.
# Values are sh-quoted by the emitters, so eval can neither word-split nor inject.
eval "$("$TARGET/vd-devcluster" env --slot "$SLOT")"
eval "$("$TARGET/vd-slot" --slot "$SLOT" --agent "$AGENT")"

CLIENT_BIN="$TARGET/client"
if [[ ! -x "$CLIENT_BIN" ]]; then
    cat >&2 <<EOF
client.sh: the wgpu client binary lands in P1.5 Slice 3 — not built yet.
Resolved conventions for --name '$NAME' (slot $SLOT, agent $AGENT):
  gateway          $VD_GW_ADDR
  client QUIC port $VD_CLIENT_QUIC_PORT
  dev-control port $VD_DEVCTL_PORT  (vdctl --port $VD_DEVCTL_PORT)
  trust dir        $VD_TRUST_DIR
  login key        (VD_AUTH_SIGNING_KEY from the cluster env)
EOF
    exit 0
fi

exec "$CLIENT_BIN" \
    --name "$NAME" \
    --gateway "$VD_GW_ADDR" \
    --client-quic "$VD_CLIENT_QUIC_PORT" \
    --trust-dir "$VD_TRUST_DIR" \
    --dev-control "$VD_DEVCTL_PORT"
