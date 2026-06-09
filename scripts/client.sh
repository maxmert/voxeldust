#!/usr/bin/env bash
# client.sh — launch ONE dev-control client window against a running dev cluster
# (HR6). The dev-control + client-QUIC ports follow the canonical per-slot/per-agent
# scheme (vd-devproto via `vd-slot`), so several clients per worktree never collide
# — what the P2 two-client visual scenario (client2 screenshots client1 crossing a
# boundary) needs.
#
#   scripts/client.sh --name walker  --agent-index 0            # headless (vdctl-driven)
#   scripts/client.sh --name walker  --agent-index 0 --window   # Bevy window (walk a dot)
#   scripts/client.sh --name watcher --agent-index 1
#
# The slot DEFAULTS to this worktree's stable value (matching what
# `dev-cluster.sh up` derived); pass `--slot N` to override. By default this launches
# the HEADLESS dev-control client (net + deterministic core + the cfg-gated `vdctl`
# listener). With `--window` (Slice-3 T4) it ALSO opens the Bevy window — WASD + mouse
# walk the dot, the HUD shows the player's location/stats — built with the `render`
# feature (pulls Bevy); the dev-control listener stays up so `vdctl` still drives it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')/debug"

SLOT=""
AGENT=0
NAME="client"
WINDOW=0
CAPTURE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slot) SLOT="${2:?--slot needs a value}"; shift 2 ;;
        --agent-index) AGENT="${2:?--agent-index needs a value}"; shift 2 ;;
        --name) NAME="${2:?--name needs a value}"; shift 2 ;;
        --window) WINDOW=1; shift ;;
        --capture) CAPTURE=1; shift ;;
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
# `set -a` (allexport) is REQUIRED: the client reads VD_AUTH_SIGNING_KEY from its
# process ENV (a secret belongs in env, not argv where `ps` would expose it), so the
# contract vars must be EXPORTED to cross the `exec` boundary into the client — a
# plain eval leaves them shell-local and the client dies at boot with a missing key.
set -a
eval "$("$TARGET/vd-devcluster" env --slot "$SLOT")"
eval "$("$TARGET/vd-slot" --slot "$SLOT" --agent "$AGENT")"
set +a

CLIENT_BIN="$TARGET/client"
# `dev-control` is NON-default (a release build links no listener at all). `--window`
# additionally enables `render` (the Bevy window) — a heavier build (pulls Bevy), so it
# is opt-in. Builds are incremental — effectively a no-op once current.
FEATURES="dev-control"
{ [[ "$WINDOW" == "1" ]] || [[ "$CAPTURE" == "1" ]]; } && FEATURES="dev-control,render"
cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins --bin client --features "$FEATURES"

# The agent drives this client over the dev-control listener: input injection at
# the same seam the keyboard uses, `state`/`wait-until` reads of the decoded
# delivered world. `--allow-dev-control` enables the privileged (mutating) commands.
# `--window` opens the Bevy window in addition (WASD/mouse feed the SAME input mailbox).
CMD=(
    "$CLIENT_BIN"
    --name "$NAME"
    --agent-index "$AGENT"
    --gateway "$VD_GW_ADDR"
    --client-quic "$VD_CLIENT_QUIC_PORT"
    --trust-dir "$VD_TRUST_DIR"
    --dev-control "$VD_DEVCTL_PORT"
    --allow-dev-control
)
[[ "$WINDOW" == "1" ]] && CMD+=(--window)
[[ "$CAPTURE" == "1" ]] && CMD+=(--capture)
exec "${CMD[@]}"
