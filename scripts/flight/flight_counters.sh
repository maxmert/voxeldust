#!/bin/bash
# Poll the gateway's admin counters once a second; print a line whenever a hand-over lands or a
# hold/refusal counter moves. Usage: flight_counters.sh <gateway admin port>
PORT=${1:-7561}
KEYS="exterior_moves_applied window_compose_hold_ticks window_misauthored_body window_unresolved_standing scene_levels_sent window_rotated_refused window_instant_mismatch frame_sub_desync window_t_monotone_stalled"
prev=""
while true; do
  line=$(curl -s --max-time 2 http://127.0.0.1:$PORT/admin/snapshot | python3 -c '
import sys,json
try: d=json.load(sys.stdin)
except Exception: sys.exit()
g=d.get("gateway") or {}
keys="'"$KEYS"'".split()
print(" ".join(f"{k.replace(chr(119)+chr(105)+chr(110)+chr(100)+chr(111)+chr(119)+chr(95),chr(119)+chr(46))}={g.get(k,0)}" for k in keys), "tick=%s" % d.get("universe_tick"))
')
  if [ -n "$line" ]; then
    cur=$(echo "$line" | sed -E 's/ tick=[0-9]+//')
    if [ "$cur" != "$prev" ]; then echo "$(date +%T) $line"; prev="$cur"; fi
  fi
  sleep 1
done
