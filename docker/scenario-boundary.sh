#!/bin/sh
# S5a scenario #1 — boundary crossing (position DISPLACEMENT). The FIRST agent-HR6 scenario: it proves the FULL
# agent-operable loop end-to-end against the LIVE cluster, GPU-free:
#   login -> inject forward input -> gateway -> shard AUTHORITATIVE sim -> snapshot -> the avatar's pos advances.
# The pass signal is the EUCLIDEAN DISPLACEMENT magnitude, NOT a single axis: the shared movement convention
# (vd_core::kinematics::local_axes_from_movement) maps `move 1 0 0` (forward) to world -Z, so an x-only pos[0]
# check would never see it. Magnitude == "the avatar moved past a threshold" == the honest boundary signal,
# convention-agnostic (no hand-re-encoded axis drift — the sim owns the ONE movement convention).
# In P1.5 (one realm, SystemSpace{1}) "crossing a boundary" == the authoritative position advances past a
# threshold under injected input; the cross-REALM `location` flip is a NO-OP here (one realm) so it is logged
# as a diagnostic ONLY, never a pass signal. (S5b walk-to + P2 transfers grow this into a real realm crossing.)
# Exit codes: 0 crossed | 1 never-crossed | 11 client-died | 12 never-Active | 13 no-authority.
set -eu

GW="${VD_GATEWAY_ADDR:?agent-entrypoint must export VD_GATEWAY_ADDR}"
QUIC="${VD_CLIENT_QUIC:-9000}"
DEVCTL="${VD_DEVCTL_PORT:-7777}"
TRUST="${VD_TRUST_DIR:-/etc/vd/trust}"
IDX="${VD_AGENT_INDEX:-0}"
THRESH="${VD_CROSS_THRESHOLD:-1.0}"
MAXPOLL="${VD_MAX_POLLS:-300}"
export VD_DEVCTL_PORT="$DEVCTL" # vdctl reads this fallback; no --port needed

echo "[agent] launching client --gateway $GW --client-quic $QUIC --bind 0.0.0.0 --dev-control $DEVCTL --agent-index $IDX" >&2
# --bind 0.0.0.0: the client's QUIC endpoint MUST bind all interfaces to reach the gateway over the pod
# network (the default 127.0.0.1 is loopback-only, correct for the local dev cluster but unroutable in k3d).
client --name agent --agent-index "$IDX" --gateway "$GW" \
  --client-quic "$QUIC" --bind 0.0.0.0 --trust-dir "$TRUST" \
  --dev-control "$DEVCTL" --allow-dev-control &
CLIENT_PID=$!
trap 'vdctl close >/dev/null 2>&1 || true; kill "$CLIENT_PID" 2>/dev/null || true' EXIT
alive() { kill -0 "$CLIENT_PID" 2>/dev/null || { echo "[agent] FATAL client exited early" >&2; exit 11; }; }

# 1) LOGIN -> Active, with retry (absorbs the listener bind + admission timing + shard-holds-realm).
#    `vdctl wait active eq 1 400` blocks IN the client listener up to 400 client step-ticks, returns State
#    (exit 0) once Active or Timeout (exit 3); vdctl exit 2/1 = listener not up yet -> retry.
i=0
while [ "$i" -lt 30 ]; do
  alive
  if vdctl wait active eq 1 400 >/dev/null 2>&1; then break; fi
  echo "[agent] not yet Active (retry $i)" >&2
  sleep 2
  i=$((i + 1))
done
[ "$i" -lt 30 ] || { echo "[agent] FATAL never became Active (login race / admission / key mismatch)" >&2; vdctl state || true; exit 12; }

# 2) Require own_entity + a landed frame BEFORE reading pos (the shard-holds-realm / authority guard).
vdctl wait own_entity_set eq 1 200 >/dev/null || { echo "[agent] FATAL no authority (own_entity unset)" >&2; vdctl state; exit 13; }
vdctl wait snapshots_applied ge 1 200 >/dev/null || { echo "[agent] FATAL no snapshot landed" >&2; vdctl state; exit 13; }
echo "[agent] ACTIVE + authority + first frame:" >&2
vdctl state

# 3) RECORD the OWN entity's FULL start pos (typed jq parse, own-entity-keyed — DevResponse is serde tag=resp,
#    so the DevState is under .state; DevState fields are snake_case). `.pos` is the [x,y,z] array; we track the
#    Euclidean displacement magnitude from it (see the header — convention-agnostic).
S0="$(vdctl state)"
OWN="$(printf '%s' "$S0" | jq -r '.state.own_entity')"
P0="$(printf '%s' "$S0" | jq -c --arg e "$OWN" '.state.entities[] | select(.entity==$e) | .pos')"
LOC0="$(printf '%s' "$S0" | jq -r '.state.location')"
echo "[agent] start own=$OWN pos=$P0 location=$LOC0 threshold=$THRESH" >&2

# 4) INJECT forward ONCE. apply_input_action sets a LEVEL-HELD InputState (the sim integrates it every tick until
#    reset), so a single Move keeps the avatar moving — no per-poll re-inject needed.
vdctl move 1 0 0 >/dev/null

# 5) POLL the OWN pos DISPLACEMENT magnitude until >= THRESH or MAXPOLL exhausted. Position is the ONLY honest
#    v1 pass signal; magnitude (sqrt of per-axis delta² sum) makes it axis-convention-agnostic.
p=0
while [ "$p" -lt "$MAXPOLL" ]; do
  alive
  S="$(vdctl state)"
  P="$(printf '%s' "$S" | jq -c --arg e "$OWN" '.state.entities[] | select(.entity==$e) | .pos')"
  crossed="$(jq -n --argjson p "$P" --argjson p0 "$P0" --argjson t "$THRESH" \
    '( ( ($p[0]-$p0[0]) as $dx | ($p[1]-$p0[1]) as $dy | ($p[2]-$p0[2]) as $dz | ($dx*$dx)+($dy*$dy)+($dz*$dz) ) | sqrt ) >= $t')"
  if [ "$crossed" = "true" ]; then
    echo "[agent] PASS crossed: pos $P0 -> $P (displacement >= $THRESH)" >&2
    printf '%s\n' "$S"
    exit 0
  fi
  p=$((p + 1))
  sleep 0.1
done
echo "[agent] FAIL timeout: pos displacement never reached $THRESH from $P0 (last $P) in $MAXPOLL polls" >&2
vdctl state
exit 1
