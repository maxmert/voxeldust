#!/usr/bin/env bash
# client.sh — launch ONE dev-control client window against a running dev cluster
# (HR6). The dev-control + client-QUIC ports follow the canonical per-slot/per-agent
# scheme (vd-devproto via `vd-slot`), so several clients per worktree never collide
# — what the P2 two-client visual scenario (client2 screenshots client1 crossing a
# boundary) needs.
#
#   scripts/client.sh --name walker  --agent-index 0            # headless (vdctl-driven)
#   scripts/client.sh --name walker  --agent-index 0 --window   # Bevy window (walk a dot)
#   scripts/client.sh --name walker  --agent-index 0 --window --fast   # ^ + fast relink
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
FAST=0
REALM_BOXES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slot) SLOT="${2:?--slot needs a value}"; shift 2 ;;
        --agent-index) AGENT="${2:?--agent-index needs a value}"; shift 2 ;;
        --name) NAME="${2:?--name needs a value}"; shift 2 ;;
        --window) WINDOW=1; shift ;;
        --capture) CAPTURE=1; shift ;;
        # Link Bevy as ONE shared library instead of statically into the client binary
        # (see the FAST block below). Pure build-speed knob; changes no behaviour.
        --fast) FAST=1; shift ;;
        # Boot-load a colored-box render scene (the Visual Crossing Playground): a boxes.json =
        # Vec<RealmBoundary>, drawn as translucent realm boxes the dot walks between.
        --realm-boxes) REALM_BOXES="${2:?--realm-boxes needs a value}"; shift 2 ;;
        *) echo "client.sh: unexpected argument '$1'" >&2; exit 1 ;;
    esac
done

# NOT `-q` (here or at the client build below), deliberately — see the note in
# dev-cluster.sh: `-q` hides "Blocking waiting for file lock on build directory", which is
# what a build queued behind another cargo prints, and without it a queued build looks
# exactly like a compiling one for as long as the queue lasts.
[[ -x "$TARGET/vd-slot" && -x "$TARGET/vd-devcluster" ]] \
    || cargo build --manifest-path "$ROOT/Cargo.toml" -p vd-bins

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

# `--fast`: swap the statically-linked Bevy for the dylib one (`render-dylib`). This is the
# EDIT-REBUILD-LOOK loop's knob — with the graph warm, the client bin's link is what you
# wait on, and linking against one shared library instead of relocating all of Bevy is the
# bulk of it.
#
# OPT-IN, and it stays opt-in: `render-dylib` is a DIFFERENT feature set, so alternating
# between `--fast` and a plain `--window`/`--capture` in the SAME build directory makes
# cargo rebuild Bevy each way. Pick one per session. It is also NEVER what a gate or a
# container builds — those link exactly what ships (`render`), which is why this is a
# script flag and not a default.
if [[ "$FAST" == "1" ]]; then
    if [[ "$FEATURES" == "dev-control" ]]; then
        echo "client.sh: --fast only affects the Bevy build; add --window or --capture" >&2
        exit 1
    fi
    FEATURES="dev-control,render-dylib"
fi

cargo build --manifest-path "$ROOT/Cargo.toml" -p vd-bins --bin client --features "$FEATURES"

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
[[ -n "$REALM_BOXES" ]] && CMD+=(--realm-boxes "$REALM_BOXES")

# `--fast` only: hand the dynamic loader the search path it needs. A dylib build leaves the
# binary asking for `@rpath/libstd-*.dylib` while carrying NO LC_RPATH at all, so running it
# directly dies at load with "no LC_RPATH's found". `cargo run` hides this by injecting the
# path into the child; we exec the binary ourselves (the env-var contract above requires it),
# so we must do the same. Rust's own dylibs live in the toolchain; libbevy_dylib is referenced
# by absolute path and needs no help, but deps/ is included so a future @rpath dep also
# resolves. Set at LAUNCH, not build: adding `-C rpath` would change RUSTFLAGS and cost a full
# rebuild for something the loader can be told at zero cost. Non-fast builds are static and
# deliberately get no such variable.
if [[ "$FAST" == "1" ]]; then
    RUST_SYSROOT="$(rustc --print sysroot)"
    RUST_HOST="$(rustc -vV | sed -n 's/^host: //p')"
    export DYLD_FALLBACK_LIBRARY_PATH="$RUST_SYSROOT/lib/rustlib/$RUST_HOST/lib:$TARGET/deps${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

exec "${CMD[@]}"
