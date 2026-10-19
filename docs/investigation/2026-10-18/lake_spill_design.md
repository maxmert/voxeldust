# The spill law at the fine levels — a short design (slice W2; the owner, 2026-10-18: "Short design next")

Binding context read first: W11 (a lake is water that stands; Fill–Spill–Merge), W16 (one shape at every distance), B23 /
B38 (no caps, no knobs, published laws), B84 (the 128 m level holds each node's mean, no correction), B86 (the final
ground from the first frame; the advance calculation), B105 Q1 (b) and Q2 (every standing water's points under its level
are outlets; the water map crosses with the tile), SL6 (ask before new data crosses), SL10 (one generator, the same bytes).
The one-volume revision §3.4 item 6 names this case and its cure.

## 0. For the owner

### 0.1 What the bench found

- **G-DIVIDE** (`planet_stand water-gates 3,97,21`): south of the basin lake, a lake at −341 m reaches 53 131 cells of
  the −2 002 m lake under it. The finish then puts that water over the lower lake in a cut's band: 456 steps of 1.66 km.
- **THE SPILL COUNT** (`planet_stand spill-count … 40`): 40 of the home planet's 1 009 spilling lakes (the 20 largest,
  20 spread over the rest by size). For each lake the bench solves the 128 m tile that holds the lake's BRIM node (the first
  node its water's own receivers reach outside the lake) and finds the lowest 128 m path from the lake's cells to ground of
  a lower node. RESULT: **10 of 40 lakes spill under their 8 km level**; the drop is 4.0 / 28.8 / 502 / 773 m (p10 / p50 /
  p90 / most); none drains whole inside the tile. The count reads the brim's tile only, so it is a LOWER BOUND: a rim cut in
  another tile of the lake is not seen. The closed lakes (155) are not counted yet.

| Lake | Area | 8 km level | Its 128 m spill | Drop |
|---|---|---|---|---|
| #8 | 11 148 km² | −1 265 m | −1 767 m | 502 m |
| #16 | 4 701 km² | −163 m | −936 m | 773 m |
| #12 | 9 881 km² | −889 m | −925.5 m | 36.5 m |
| #366 | 129 km² | 159 m | 152 m | 7 m |
| #563 | 64 km² | 2 980 m | 2 970.8 m | 9.2 m |

- ★ **THE FULL COUNT** (ruling B115 item 3; `planet_stand spill-full`: every lake, its tiles' own squares joined, 1 854
  tiles solved in 857 s on 12 threads; 13 lakes across a cube face not read; `lakes.csv` outside the repo):
  - **Spilling lakes: 261 of 1 000** spill under their 8 km level on the 128 m ground (375 490 km²); the drop 0.5 / 14.2 /
    502 / 1 214 m (p10 / p50 / p90 / most); 6 drain whole.
  - **Closed lakes: 5 of 151**; the drop p50 408 m, most 1 700 m; none drains whole.
  - Drops over 1 / 10 / 50 / 100 / 500 m: 210 / 143 / 91 / 74 / 29 lakes. The largest drops are HIGH MOUNTAIN LAKES of the
    belts (levels 2 500–3 900 m, 8 km floors about a kilometre under them), whose 128 m spill lies near their floor: lake
    #688 stands at 3 005 m, its floor at 1 765 m, its 128 m spill at 1 791 m.
  - THE BASIN LAKE THE OWNER LOOKED AT (lake #525, 255 nodes, closed, −2 002 m; `planet_stand spill-lake 3,97,21`): its own
    tile's raster joins the lake's floor to a dry pocket node (row −3 007 m) through a rim node's 128 m valley; the pass
    −2 721.25 m, the same by raster and on the joined map — 719 m under the lake's level.

### 0.2 Why it happens

The 8 km world's lake stands at its SPILL, the height of its brim node (W11). The 128 m level holds only each node's MEAN to
its row (B84). So the brim node's own 128 m ground holds valleys under its row, and its lowest valley cuts the rim under the
lake's level. The water map spreads each water through every cell under its level that a path joins (B105 Q1 (b)), so the
lake runs out through the cut and stands over all the lower ground it reaches.

**Example.** Lake #16 stands at −163 m. Its brim node's row is −163 m, but one 128 m valley of that node runs down to the
basin at −936 m. Today the water map spreads the lake at −163 m down that valley, and over every lower cell that joins it.

### 0.3 The law

W11 already rules it: **a still water stands at most at its lowest outlet** (Fill–Spill–Merge; Barnes, Callaghan &
Wickert 2021). On the 128 m ground the lowest outlet is the lowest PASS of the 128 m ground between the lake and lower
ground. So a lake's FINE LEVEL is the lower of its 8 km level and its 128 m spill. No new law; the same law read on the
finer ground.

### 0.4 The options

**(a) ONE FINE LEVEL PER LAKE, IN THE SHARD (recommended).** The planet's shard computes each lake's 128 m spill once per
world version, over the finished 128 m squares of the lake's tiles, keeps it in its store, and writes each tile's water
map with it. **No new data crosses**: the water map already crosses with each tile (B105 Q2), and it carries the lake's
level in each wet cell. So this needs NO SL6 ask. The price:
- A tile's water map waits until each lake that touches it has its fine level, so the lake's tiles are solved together.
  The largest spilling lake (#0, 231 098 km²) covers about 56 tiles of 64 km; at 13–49 s a tile on 14 cores, about 1–3
  minutes at the first visit near it (once per world version; B86 item 6's advance calculation orders it).
- THE GROUND STAYS AS SOLVED. The solve took the ground under the 8 km level as the lake's floor (the smooth 8 km surface,
  B105 Q1 (b)). Where the level drops, that floor stands dry: a "bathtub ring" of smooth, uneroded ground between the two
  levels. Lake #12's ring is 36.5 m high; lake #16's is 773 m high.

**(a′) THE SAME, AND THE LAKE'S TILES SOLVED AGAIN at the fine level.** The fine level becomes the lake's outlets' level
in a second solve, so the ground under the old water erodes like other land and the ring goes. The price: the lake's tiles
solve twice; the second solve can cut a new pass, so the level is solved again until it holds (a fixed point; the number of
rounds to MEASURE). Recommended only if the side views show the ring.

**(b) EACH RASTER ITS OWN SPILL — refused.** Two tiles see different parts of one rim and give one lake two levels: a seam
in the water (G-SEAM-WATER).

**(c) THE 128 m RIM HELD AT THE LAKE'S LEVEL — refused.** A clamp on the ground (B23).

### 0.5 The far view (W16)

From rung 13 up the far map reads the pyramid's water words (slice W2 item 8, built 2026-10-18), and they hold the 8 km
level. A lake whose fine level drops shows two levels: the near level and the far level. At the rung 12 → 13 swap its
water steps by the drop (36.5 m for lake #12, 773 m for lake #16). Two ways:
- **Register the step** until the advance calculation holds a whole body; then the pyramid becomes the exact fold of the
  tiles (the one-volume revision §3.6, Q10). G-SWAP reads it meanwhile. No new data.
- **A per-realm table of fine lake levels** to the client (an SL6 ask): one word per spilling or closed lake, about 1 164
  lakes × 8 bytes ≈ 9 KB a body, sent when the advance calculation has it.

### 0.6 The questions

- **Q1.** (a) now and (a′) only if the side views show the ring? (recommended) — or (a′) at once?
- **Q2.** The far step: register it until the advance calculation (recommended), or the SL6 table now?
- **Q3.** Count the closed lakes too and read every lake's whole rim (all its tiles, not the brim's alone) before the build?
  (recommended; a bench of about 30 minutes on the cores)

## 1. How the fine level is computed (a)

1. **The lakes.** Each lake of the 8 km world: the connected nodes whose water word stands over their row and are not the
   sea. Its id: its lowest node index (one answer on every host).
2. **Its tiles.** Every 128 m tile that holds a lake node or a node next to one.
3. **The spill.** On the finished 128 m squares of those tiles (the ground the client draws), a priority flood from the
   lake's own cells: each path's height is the highest ground along it; the first cell of a node outside the lake whose
   row stands under the lake's level ends the flood, and the path's height there is the spill. Integers in quarter metres;
   ties by the cell's index.
4. **The fine level** = the lower of the 8 km level and the spill.
5. **The water map** of each tile: the lake's identity spreads at its fine level (the one rule, `fine::water::raster_water`,
   with the lake's level replaced).
6. **The 16 m level** reads the 128 m water's identity (§3.4 item 5), so it takes the fine level with no change.

**Example.** The −341 m lake south of the basin lake: its flood meets the −2 002 m basin's node at a pass of the 128 m
ground; if that pass stands at −1 500 m, the lake's fine level is −1 500 m, its water stays inside its own hollow, and the
−2 002 m lake's cells hold the −2 002 m water again.

## 2. The checks

| Gate | Checks | Failing control |
|---|---|---|
| G-DIVIDE | No cell two standing waters reach, on every solved tile around the stand | Today's map (53 136 cells at the basin) |
| G-SPILL (new) | Every lake's level at or under its 128 m spill on the finished ground | The 8 km level (10 of 40 lakes) |
| G-SEAM-WATER | No wet/dry disagreement across a cut | Option (b) |
| Side views | The −341 m lake's outlet, lakes #12 and #16 from 1.5 km, judged by the owner from the side (B93 item 2) | — |

## 3. The price (estimates)

- The flood: one pass over the lake's finished cells, about 0.26 million cells a tile — small next to a tile's solve.
- The first visit near a large lake: its tiles solved together, about 1–3 minutes for the largest (once per world version).
- The store: one word per lake, about 9 KB a body.
- The world move: FINE_VERSION 6 (every water map written again; the ground does not change under (a)).

## 4. Form (a′) in the shard, as built (ruling B115 item 1; 2026-10-18)

1. **A lake's fine level is the fixed point of its OWN rounds.** Its tiles (each tile whose raster holds one of its cells:
   the tile of each lake node and of each node beside one) are solved with that lake ALONE at a trial level (the others at
   their 8 km words); each raster's spill for the lake is read (`L1Tile::spills`); the lowest spill lowers the level, and
   the tiles are solved again, until the level holds (bound: 8 rounds). The rounds read nothing but the artifact, so the
   level is one answer on every host, in any order of visits.
2. **The store keeps it**: one five-byte record per lake under the fine family's level number 0, keyed by the 128 m
   level's input word.
3. **A tile's final solve** waits until every lake in its domain (the tile and its two-node buffer) has its level, and
   names the lakes whose level is not their 8 km word (`solve_l1_levels`): the outlets, the base levels and the water map
   read them through `Waters`; a lake node whose row stands at or over its lake's fine level is dry land.
4. **The 16 m water** spreads only inside the 128 m water's reach (the column rule at each 16 m cell), so it refines the
   128 m shore and never runs out through a 16 m valley (W16).
5. **The price**: every lake's first round reads ONE shared plain solve per tile (every lake at its 8 km word), and that
   solve is the tile's final raw where every lake of its domain holds its word; a lowered lake's tiles solve once more per
   later round and once for the final solve. MEASURED: the basin lake held after one round on the bench; the window's
   warm-up at the basin lake ran 3 finished tiles a minute with a round per lake, 8.5 with the shared plain solve (the
   shard's solve threads in the low class, ruling B91).

**Example.** A pilot lands beside the basin lake. The planet's shard finds the lake in the domains of the tiles it needs,
solves the lake's 18 tiles at −2 002 m, reads their spills (the lowest −2 721.25 m), solves them again at −2 721.25 m,
reads the same spill, keeps −2 721.25 m in its store, and then solves the pilot's tiles with the lake at that level: the
lake stands lower, among eroded slopes, and the pocket beside it stays dry.
