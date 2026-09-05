#!/usr/bin/env python3
"""Turn the gateway's per-tick trace (`VD_TRACE_REALM`) into a per-tick table for ONE realm kind.

Usage: trace_star.py <vd-gateway.log> [realm-substring]
Prints: per fold tick — stamp, source (fresh/held), distance, the closing rate per tick, the
change of that rate (the hitch signal), the chain's hops/prefix/covers/pending — and a summary of
every run of held ticks and every tick whose rate deviates from its neighbours' median.
"""
import re, sys, statistics
path = sys.argv[1]; want = sys.argv[2] if len(sys.argv) > 2 else ""
pat = re.compile(r'(\S+)\s+INFO vd_trace: trace row t=(\d+) stamp=(\d+) client=\S+ realm=(\S+) source="?(\w+)"? stratum=(\d+) x=(\S+) y=(\S+) z=(\S+) vx=(\S+) vy=(\S+) vz=(\S+) dist=(\S+) hops=(\d+) prefix=(\d+) covers_lineage=(\w+) hop_pending=(\w+)')
rows = []
for line in open(path, errors="replace"):
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    m = pat.search(line)
    if not m: continue
    if want and want not in m.group(4): continue
    rows.append(dict(ts=m.group(1), t=int(m.group(2)), stamp=int(m.group(3)), realm=m.group(4), src=m.group(5),
                     dist=float(m.group(13)), hops=int(m.group(14)), prefix=int(m.group(15)), covers=m.group(16), pending=m.group(17),
                     v=(float(m.group(10)), float(m.group(11)), float(m.group(12)))))
if not rows:
    print("no trace rows (is VD_TRACE_REALM set on the gateway, and did a session draw that kind?)"); sys.exit(1)
rows.sort(key=lambda r: r["t"])
# per-tick closing rate and its change
prev = None; table = []
for r in rows:
    rate = None; drate = None
    if prev and r["t"] > prev["t"]:
        rate = (r["dist"] - prev["dist"]) / (r["t"] - prev["t"])
        if prev.get("rate") is not None: drate = rate - prev["rate"]
    r["rate"] = rate; r["drate"] = drate; table.append(r); prev = r
print(f"{len(table)} ticks for {table[0]['realm']} from t={table[0]['t']} to t={table[-1]['t']}")
# held runs
runs = []; start = None
for r in table:
    if r["src"] == "held" and start is None: start = r["t"]
    if r["src"] == "fresh" and start is not None: runs.append((start, r["t"] - 1)); start = None
if start is not None: runs.append((start, table[-1]["t"]))
print("held runs (ticks):", runs)
# the hitch scan: |drate| against the median |drate| of the 20 neighbours
rates = [r for r in table if r["drate"] is not None]
print("\n  tick  src    dist         rate/tick     d(rate)      hops/prefix covers pending")
flagged = 0
for i, r in enumerate(rates):
    win = [abs(x["drate"]) for x in rates[max(0, i-10):i+10]]
    med = statistics.median(win) if win else 0.0
    flag = abs(r["drate"]) > 5 * med and abs(r["drate"]) > 1e-6 * max(abs(r["rate"]), 1.0)
    if flag or r["src"] == "held":
        flagged += 1
        print(f"  {r['t']:6d} {r['src']:5s} {r['dist']:12.4e} {r['rate']:13.4e} {r['drate']:12.4e}  {r['hops']}/{r['prefix']}      {r['covers']:5s}  {r['pending']:5s} {'<-- hitch' if flag else ''}")
print(f"\n{flagged} lines shown (held ticks and rate hitches); the rest of the {len(rates)} ticks are steady.")
