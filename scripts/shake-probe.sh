#!/usr/bin/env bash
# shake-probe.sh — MEASURE the wobbling-horizon bug before anyone writes a fix for it.
#
# WHY THIS EXISTS. The recorded explanation for the shake was "the player is smoothed between server
# updates while the ground is drawn from the newest value". Reading the code REFUTES that: the client
# keeps only the two most recent poses of anything (about one tick apart under per-tick delivery) but
# deliberately draws a moment ~6 ticks in the past, so it always falls off the end of that window and
# returns the older sample verbatim. Nothing on the client is interpolated today, for anything. Both
# the player and the ground are step functions, so the wobble is ARRIVAL-driven, and there are three
# candidate causes left. Only a measurement separates them, and the fix spans four crates on a ~50-min
# build — so guessing is the expensive option.
#
# THE THREE CANDIDATES, and what each looks like in the output below:
#   (A) RENDER PATH — the two feeds' ticks track each other closely, but the player-to-planet vector
#       still jitters. Then the cause is on the client (what is sampled when) and slice 6 fixes it.
#   (B) TWO CLOCKS — the two feeds' ticks DRIFT apart. The ground's placement is authored by the
#       parent star system's shard while the player's pose is composed on the planet's own shard, and
#       each shard advances its sense of universe time ONLY when a clock sync arrives (there is no
#       local per-tick advance). Two independently-stepping clocks meeting on one screen. That fix is
#       server-side and is NOT slice 6.
#   (C) NEITHER — the vector is steady and the shake is elsewhere (camera, projection, the box mesh).
#
# Usage:
#   scripts/shake-probe.sh [seconds]        # default 12
#
# Output: a TSV series to stdout plus a summary. Attach the series to task #176.
set -euo pipefail

SECONDS_TO_SAMPLE="${1:-12}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["target_directory"])')/debug"

cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins --bin client --features dev-control
cargo build -q --manifest-path "$ROOT/Cargo.toml" -p vd-bins

# The demand cluster at VISUAL scale — the exact configuration the shake was reported in. `spawn_node`
# inherits this env, so the demand-spawned home shard authors the window-friendly orbiting system.
# THE WORLD IS NO LONGER SELECTED, so there is nothing to export here. This line used to set a scale,
# and a live cluster was read process by process with the orchestrator on one world and its own gateway on
# another — from THIS script, in one launch. A knob that exists can be set twice; the fix was to delete it.
"$ROOT/scripts/dev-cluster.sh" up --demand

cleanup() {
    [[ -n "${CLIENT_PID:-}" ]] && kill "$CLIENT_PID" 2>/dev/null || true
    "$ROOT/scripts/dev-cluster.sh" down >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

"$ROOT/scripts/client.sh" --name shake-probe --agent-index 0 >/tmp/shake-probe-client.log 2>&1 &
CLIENT_PID=$!

eval "$("$TARGET/vd-slot" --worktree "$ROOT")"
eval "$("$TARGET/vd-slot" --slot "$VD_SLOT" --agent 0)"
VDCTL=("$TARGET/vdctl" --port "$VD_DEVCTL_PORT")

# The client needs a moment to link, boot its net thread and open the dev-control listener. Poll the
# port rather than assuming — calling vdctl too early gets "connection refused", and under `set -e`
# that tears the whole cluster down and wastes the boot.
# Poll with a REAL request, not a bare TCP connect: the dev-control listener expects one request line
# per connection, so a `nc -z`-style open-and-close leaves it waiting for a line that never comes and
# the NEXT genuine request reads back a closed connection.
echo "waiting for the client's dev-control listener on ${VD_DEVCTL_PORT}…"
ready=0
for _ in $(seq 1 120); do
    if "${VDCTL[@]}" state >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
        echo "the client EXITED before serving dev-control — its log:" >&2
        cat /tmp/shake-probe-client.log >&2
        exit 1
    fi
    sleep 0.5
done
[[ "$ready" == "1" ]] || {
    echo "the client never served dev-control on ${VD_DEVCTL_PORT} within 60s — its log:" >&2
    cat /tmp/shake-probe-client.log >&2
    exit 1
}

# Wait for the session to go live AND for BOTH feeds to have delivered at least one frame — a probe
# that starts sampling before the realm feed exists would report a frozen box and prove nothing.
#
# NEVER discard these: `wait` exits non-zero on TIMEOUT and prints the last state to STDOUT, so
# redirecting to /dev/null under `set -e` kills the run silently and destroys the one piece of
# evidence explaining why. Report the state and say WHICH precondition failed.
# NOTE: `vdctl wait` closes the connection on this build instead of replying (plain `state` answers
# fine), so the preconditions are polled with `state` — the request that demonstrably works. That
# tool defect is real but is NOT the shake; it is recorded on the task rather than chased here.
await_json() {
    local jq_expr="$1" why="$2" last=""
    for _ in $(seq 1 240); do
        last="$("${VDCTL[@]}" state 2>&1 || true)"
        if printf '%s' "$last" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
s=d.get('state', d)
sys.exit(0 if ($jq_expr) else 1)
" 2>/dev/null; then return 0; fi
        sleep 0.5
    done
    echo "PRECONDITION FAILED after 120s: $why" >&2
    printf '%s\n' "$last" | head -40 >&2
    return 1
}
await_json "s.get('phase')=='active'" \
    "the client never became Active — login/subscribe did not complete"
await_json "(s.get('realm_frames_applied') or 0) >= 1" \
    "the realm pose feed never delivered a frame — the home realm did not spin up, or its feed never reached this client"

echo "# sampling ${SECONDS_TO_SAMPLE}s — every column is read from ONE dev-control snapshot"
echo -e "t_s\tcursor\tuniverse_tick\tentity_tick\trealm_tick\ttick_gap\tlocation\tdist_to_planet"

python3 - "$SECONDS_TO_SAMPLE" "$TARGET/vdctl" "$VD_DEVCTL_PORT" <<'PY'
import json, math, subprocess, sys, time

secs, vdctl, port = float(sys.argv[1]), sys.argv[2], sys.argv[3]

def state():
    # vdctl PRETTY-prints, so the reply spans many lines — parse the whole document, never a line.
    # It is also an envelope: {"kind":"state","state":{...}} (a timed-out wait carries the last state
    # under the same key), so unwrap one level rather than assuming the state is at the top.
    out = subprocess.run([vdctl, "--port", port, "state"], capture_output=True, text=True)
    try:
        doc = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(doc, dict) and isinstance(doc.get("state"), dict):
        return doc["state"]
    return doc if isinstance(doc, dict) else None

rows, t0 = [], time.time()
# Poll FASTER than the 50 Hz tick, so a per-tick step is resolvable rather than aliased.
while time.time() - t0 < secs:
    s = state()
    if s:
        own_id = s.get("own_entity")
        own = next((e for e in s.get("entities", []) if e["entity"] == own_id), None)
        # The realm the player is standing in, by label match against the drawn boxes.
        loc = s.get("location") or ""
        # The NEAREST drawn box, not a label match: at login the player stands in a system whose shell
        # is ambient (never drawn), so requiring `location == box` would report nothing on a perfectly
        # good run. The wobble we care about is between the player and whatever ground is nearest.
        d, near = None, ""
        boxes = s.get("realm_boxes", [])
        if own and boxes:
            nb = min(boxes, key=lambda b: math.dist(own["pos"], b["center"]))
            d, near = math.dist(own["pos"], nb["center"]), nb["realm"]
        loc = f"{loc}|near={near}"
        et, rt = s.get("entity_feed_newest_tick"), s.get("realm_feed_newest_tick")
        gap = (et - rt) if (et is not None and rt is not None) else None
        rows.append((time.time() - t0, s.get("render_cursor"), s.get("universe_tick"), et, rt, gap, loc, d))
        f = lambda v, p="": "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) and p != "" else str(v))
        print(f"{rows[-1][0]:.3f}\t{f(s.get('render_cursor'),'2')}\t{f(s.get('universe_tick'))}\t{f(et)}\t{f(rt)}\t{f(gap)}\t{loc}\t{f(d,'4')}")
    time.sleep(0.005)

print(f"\n# {len(rows)} samples over {secs:.0f}s")
gaps = [r[5] for r in rows if r[5] is not None]
ds   = [r[7] for r in rows if r[7] is not None]
ets  = [r[3] for r in rows if r[3] is not None]
rts  = [r[4] for r in rows if r[4] is not None]

if ets: print(f"# entity feed tick: {min(ets)} -> {max(ets)}  (advanced {max(ets)-min(ets)})")
if rts: print(f"# realm  feed tick: {min(rts)} -> {max(rts)}  (advanced {max(rts)-min(rts)})")
if gaps:
    print(f"# TICK GAP (entity - realm): min {min(gaps)}  max {max(gaps)}  spread {max(gaps)-min(gaps)}")
    print("#   spread 0-1  => the two feeds are locked together     => candidate (A) render path")
    print("#   spread > 1  => the two shards' clocks are DRIFTING   => candidate (B) server-side, NOT slice 6")
else:
    print("# TICK GAP: never computable — one of the feeds delivered nothing. The run proves NOTHING; fix the setup.")
if ds:
    mean = sum(ds)/len(ds)
    var  = sum((x-mean)**2 for x in ds)/len(ds)
    print(f"# player-to-planet distance: mean {mean:.4f}  sd {var**0.5:.4f}  min {min(ds):.4f}  max {max(ds):.4f}")
    print("#   a LARGE sd with a locked tick gap is the render-path signature;")
    print("#   a LARGE sd tracking the tick gap is the two-clocks signature.")
else:
    print("# player-to-planet distance: never computable (no box matched the player's location label).")
    print("#   The run proves NOTHING about the shake — the label match or the AoI scene is the problem.")
PY
