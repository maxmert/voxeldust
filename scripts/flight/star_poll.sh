#!/bin/bash
# 10 Hz: the client's own view of the star and the sky, plus the client's CPU — a render freeze
# shows as the wall clock jumping between samples while the client burns or sleeps. `camera=` is the
# camera-mode instrument (2026-09-06): poll it through a hand-over, and a one-second flip to
# first-person shows as a sample that says so, instead of a memory nobody can check.
V="$(cd "$(dirname "$0")/../.." && pwd)/target/release/vdctl"
while true; do
  t=$(python3 -c 'import time; print("%.2f" % time.time())')
  cpu=$(ps -o pcpu= -p $(pgrep -f "target/release/client" | head -1) 2>/dev/null | tr -d ' ')
  $V --port 7562 state 2>/dev/null | python3 -c '
import sys,json
try:
  d=json.load(sys.stdin).get("state",{})
  b=d.get("realm_boxes",[])
  star=[r for r in b if r["realm"].startswith("Star")]
  own=d.get("own_entity"); pos=None
  for e in d.get("entities",[]):
    if e.get("entity")==own: pos=[round(x,1) for x in e.get("pos",[])]
  print("camera=%s boxes=%d star=%s star_center=%s star_kind=%s stars_drawn=%s tick=%s cursor=%s loc=%s pos=%s" % (d.get("camera_mode","?"), len(b), len(star), ([round(x,1) for x in star[0].get("center",[])] if star else "-"), (star[0]["body_kind"] if star else "-"), d.get("stars_drawn"), d.get("universe_tick"), d.get("render_cursor"), (d.get("location") or "")[:24], pos))
except Exception as e: print("no state")' | sed "s/^/$t cpu=$cpu /"
  sleep 0.1
done
