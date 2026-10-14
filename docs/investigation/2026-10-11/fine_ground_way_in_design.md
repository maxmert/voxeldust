# The fine ground's way into the game — design (B80 step 3), revision 4

**Revision 4 (2026-10-11) folds in rulings B82–B84 and the bench B0's measurements; §11 lists every change with its source.
Where this revision and an older passage below disagree, §11 and the passages marked "rev 4" win.** Ruling B85 (the
same day) amends §3.4 and §4.4 — the shard ships the FINISHED tile, joined by the corner rule; its passages are marked
"B85" and win over rev 4's. The build starts
from this revision (the owner: *"Ok, please proceed. Make sure that ground rules and laws are not broken"*).

Status (revision 3): a design for the owner's answers. Nothing here is built. Revision 1 (2026-10-11) had three adversarial
reviews — A the rules and rulings, B the speed and robustness, C the fit with the code (kept in the session scratchpad's
`fine_reviews/`); revision 2 answers every finding (§10). Revision 3 follows ruling B81 (the owner accepted the eight recommendations
"if they are not breaking ground rules and laws and if we don't build seams — all our solutions should be generic
and reusable"): §0.2 names the one generic machinery, §3.8 lists every place two pieces of ground meet and its one
treatment, and three seam sources of revision 2 are gone (the far levels' straight edge, the rising region's edge,
the coast). Owner's brief: *"We need very performant, robust and reliable
solution please"*; *"make sure you have ground rules and laws in your context, as this is core and we should not break
them — especially one where realm compute and draw itself"*; and mid-design: *"Current rungs and all mechanic works
quite good now (but not very performant). If we can simply change the terrain generation algos, that would be good
enough. If we can optimize without compromising quality, that would be even better."*

## 0. The idea in one paragraph

The planet realm computes and draws its own ground (SL3, B35). Below its 8 km macro nodes it solves its ground in TILES
at two LEVELS by B80's laws — L1 (points at the rung-7 cell, 128 m on the home planet) and L2 (points at the rung-4 cell,
16 m) — each tile a pure function of its INPUTS (the seed, the charter, the macro artifact, the present rise, the fine
version), reading only the coarser level's STORED bytes and never a sibling tile, so the result does not depend on who
came first, on which host, or on how many cores. The planet's shard stores each tile once and ships it through the
artifact's own tile path. A chunk that reads a fine level WAITS until the tiles it needs are live, exactly as a chunk
waits for its artifact tiles today; meanwhile the coarser rung stands whole. Each tile carries its own mean pyramid, so
rung `r` reads the level whose cell is `2^r` m and a rung swap or a tile's arrival REFINES the ground and never
contradicts it.

**Example.** A pilot flies a hull down onto the home planet's active belt at 3° S, 121° W. When the hull enters the
planet realm it becomes the planet's occupied child; the planet's shard queues the L1 tiles under its path and solves
them on its share of the cores long before the touchdown; at 7 km the L2 tile under the landing site is live. The
pilot's client draws both from the stored bytes; until each lands, the coarser rung stands. A second pilot who lands
there a year later gets the same bytes from the store.

## 0.1 What stays exactly as it is

- **Unchanged:** the rung ladder, the 62-cell chunk, the extractor, the crossfade band and the morph, the arrival ramp,
  the stand-ins, the bounded ask (F9), the client's worker share, the GPU seam, the artifact's own head, rows, pyramid
  and tiles, the "wait for your data, the coarser rung stands" rule (`chunks.rs:2194-2245`), the ship's paths, and the
  gateway's undecoded relay.
- **Changed — where a column's height comes from** (`ColumnRead::column`, `height::field_column`, one fine read for
  both): on a body whose fine levels apply, rungs 0–9 read the finest LIVE fine level instead of the macro field plus
  the noise octaves (§4.4).
- **Changed — the data behind it:** the shard's fine worker and store family, ONE new bulk part variant with a level
  field, the client's fine tile book, and the ladder's step bound for solved ground (§4.5). Five crates change; the
  mechanics do not.
- **Faster where the ground is held** (gate P-READ, measured per rung): a held tile replaces the octaves between
  48.8 m and 25 km (the live fine octaves, `body.rs:185`) by one integer Catmull-Rom read of 16 samples (32 or 64 at a
  cut, §3.4). At rungs 7–9, where today sums only 1–3 octaves, it may not be faster; P-READ says.

## 0.2 One generic machinery (B81 item 4)

ONE tile machinery serves every level, every body and every landform family: a LEVEL is a rung cell and a LAW; a TILE
is solved from the coarser level's stored bytes, stored, made live, shipped, read and stitched the same way at every
level. The fluvial law (B80) is its first family. The families still owed — craters on an airless moon, dunes, the sea
floor's abyssal hills, the cliff and bed family — plug into the same machinery as other laws over the same tiles,
never as a second path. The stitch (§3.4) is one function for every border of every level.

## 1. The ground rules this design keeps

| Rule | What it asks | How the design keeps it |
|---|---|---|
| SL3, B35 — a realm draws itself | the planet authors its own surface | the planet's own shard solves, stores and ships its tiles; nothing else computes them |
| SL7 — the realm decides from itself and one level down | what to solve comes from the realm's own occupants and its OCCUPIED DIRECT CHILDREN at the placements it authored | §4.1: only those reasons; a viewer in another realm (path B) is served STORED tiles only and never queues a solve |
| SL2 — no occupant pose crosses | no foreign pose warms the realm | path B's view ask (today's, `session_flow.rs:195-199`) never enters the queue |
| SL9 — unbounded children | no per-tick walk of every reason | the queue is fed by interest CHANGES (enter, leave, a new tile in reach), never rebuilt from all reasons each tick (§4.1) |
| SL6 — ask before new data crosses | the fine tiles to clients are new data | ASK Q3, with the SL6 default (every client solves) priced beside it (Q2) |
| SL10 — the static shape is a function of seed, address, charter; one generator; no drift | the shape never a function of time or state | each tile is a pure function of its inputs; a chunk never draws a stand-in shape "until a tile lands" at its own rung (it waits); the server and the client read the same live set; G-NO-DRIFT and the golden self-check cover the fine read (§6) |
| F7 — the recipe is integer-only | no float in the recipe | the READING (`column_height`) is integer and lives in `vd-recipe`'s kernels beside the macro field's; the SOLVE runs once in the shard under the float fence `Gf`, as the macro solve does, and its stored output is integers |
| B64 — the same bytes on any number of cores | — | every sum over points runs over FIXED blocks of 4 096 joined in index order, whatever the thread count (§3.7); P-SAME on 1 and 4 cores and on aarch64 and x86-64 |
| F6, B20 — a share of the cores | — | the shard's fine worker uses the solve's share (half by default); the owner's Mac runs the shard and the client at once: the frame is measured with both (§6) |
| F9, B18 — the bounded ask; a hole and a pop are defects | — | a chunk waits for live tiles; the coarser rung stands; a late chunk rises over 0.5 s, as today |
| W16, B16, B27 — one shape at every distance; no change between rungs | — | each tile's mean pyramid (§3.3): rung `r` reads the `2^r` m mean level of one surface; **rev 4 (B84):** every swap between two levels stays under ONE PIXEL at its switch (measured: 128 m → 8 km rows ≤ 0.15 px; 16 m → 128 m ground 0.4 px at its max) |
| T7, B55 — the ladder's step bounds; the band one value for the flight | — | **rev 4 (B83):** the band widens by the law's bound ±0.70 × the node's half-diagonal (±4.1 km on the home planet), one value per body, before any tile is solved; at L1 every measured step is under its coarser cell's own pixel distance, so no rung's switch moves |
| B23, B38, B39, T9 | no caps, no knobs; the world converges | every stop is a convergence test (N against 2N, B39); every constant is named with its law and calibration body; **rev 4:** every SOLVER number (the rounds, the settle tolerance, the damping) is named in §7 with its convergence reading, and the rounds' step clamp is replaced by a damped step in B1 |
| Own code — no new library without the owner | the bench's `fastlem`/`terrain-graph` are dev-dependencies of the bench only | **rev 4:** the game's point graph is our own spherical Delaunay (§3.1) inside `vd-terrain`; no new dependency enters the game |
| T3 — time never moves the land | — | a tile reads no tick; a new fine version is ONE world move with new pins, never a lazy drift of visited ground (§4.6) |
| F1, F2 — a built area keeps its shape | — | a tile's key holds its version; a built patch pins its tiles; F2's half-metre bound does not hold for a new fine LAW: ASK Q8 (§4.6) |
| B21, B22 | one volume; edits are diffs | the tiles are the SOLID pass's surface; the fill and the edits read them like any surface |
| B23 item 3, B79 Q4 — every body; nothing retires before the look | — | a LAW decides where the levels apply (§3.6): no water → no fluvial level; the moon and the dry sea floor keep today's ground until their own families exist |
| HR4 | features run on any shard kind | the fine worker is a capability of any realm with ground (a planet, a moon, a hull with terrain), the G-IDENTICAL fixture on two kinds |
| HR5 | 100 % coverage on Tier-A | the new code in `vd-terrain`, `vd-recipe`, `vd-sim`, `vd-wire`, `vd-client`, `vd-connection-plane`; tests on small domains (B74) |
| B74, B76 | tests on the owner's word; the pictures decide | every heavy run asked; the owner's look is the judge at every step |

## 2. The levels and the tiles

The levels are defined in RUNG CELLS, so they hold on every body whatever its node size: L1's points and raster at the
rung-7 cell, L2's at the rung-4 cell. On the home planet (node 8 192 m) that is 128 m and 16 m.

| Level | Points and raster | Tile (solved, stored, shipped as one unit) | Solved domain | Its mean pyramid (rung → level) |
|---|---|---|---|---|
| L0 (today) | the artifact's rows | 64 × 64 nodes | — | rungs 10–18 read the artifact's pyramid (unchanged) |
| L1 | rung-7 cell | 8 × 8 nodes = 512 × 512 cells (home) | **rev 4:** + TWO nodes each side (the measured seam rule, §3.4) | **rev 4 (B83):** rung 7 the raster, rungs 8–12 its 2 × 2 … 32 × 32 means; rung 13 the node's row |
| L2 | rung-4 cell | 1 node = 512 × 512 cells (home) | **rev 4:** + 2 km (128 cells) each side | rung 4 the raster, rung 5 and 6 its means; rungs 0–3 the raster by Catmull-Rom |

- **Where a chunk at rung `r ≤ 12` reads (rev 4).** Rungs 7–12 read L1 (its mean level for the rung). Rungs 0–6 read L2 where
  L2 applies (§3.6), else L1's raster by Catmull-Rom. A chunk WAITS until every tile its stencil needs is live.
- **Under 16 m.** Today no octave exists under 48.8 m (`body.rs:185, 203`), so under the L2 raster the ground is its
  smooth integer Catmull-Rom — the look of Earth's 30 m data at 60 m height (B80's views). The law down to 2–4 m (B80
  item 3) is a later level L3 (§9 B5), not noise (B35, B80): ASK Q1.
- **The home planet's numbers.** About 133 million km² of land: ~31 000 L1 tiles if all were solved; L2 only where the
  land rises today (~3.5 %).

## 3. The solve of one tile

### 3.1 The points and the graph: the same in every tile, on the sphere

- **rev 4 (B83): THE HASHED POISSON-DISK SET** — Matérn's hard-core process, type II: four candidates a cell, each at a
  place and with a priority drawn from `corner_hash(seed, level, the cell, the candidate)` (the leaf's integer hash); a
  candidate is kept unless a candidate of higher priority (ties by the cell and the candidate's index) stands within
  `r = cell/√π` — the radius that keeps one point a cell on average (`(1 − e^(−4π r²))/(π r²)` = 0.98). A candidate's
  fate reads only the eight cells around its own, so every tile that holds a cell keeps the same points. MEASURED (B0):
  no lean (22 %), the straight share of B80's relaxed set (belt 7.1 %, plain 12.4 %).
- **The graph is the Delaunay graph ON THE SPHERE**, built from the points' integer directions (the face bend's own
  integer direction, `vd_seed::bend`) with exact predicates in `i128`: the orientation of four directions and the
  in-sphere test. No flat unfolding, so the cube's edges and its eight corners need no special case (B-I5). A point's
  neighbours lie within about 2.8 cells; a tile's buffer (§3.4) is far wider, so every tile builds the same edges for
  the points it draws.
- **Lengths** are the chords between the directions times the radius; Voronoi areas and the cotangent weights come
  from the same triangles.

### 3.2 The laws (B80's), with their names

| Level | The law | Its calibration (B38: named) |
|---|---|---|
| L1 | the joint law's steady state on the flow tree: `S = (U/(K·Q))^(1/2.2)`, `Q` = rain × the area draining through the point (Adams et al. 2020); the threshold law: no link steeper than `S_c`; each node's `U/K` solved by B80's rounds so the law's mean meets the node's (rev 4: from a coarse start, §3.3) | `n = 2.2` (Adams et al.); **rev 4 (B82): `S_c` = tan 35° = 0.70 at L1** (Burbank et al. 1996, threshold hillslopes on 90 m relief; amends A6: 1.25 gave 52 % of the 128 m ground over 45°) |
| L2 | **rev 4: the steady form** — each point over its receiver by the lower of the joint law's slope and Roering's creep steady slope at `ℓ = A/width` (Roering et al. 2007), never steeper than `S_c`; the water's area shared over the lower neighbours (Freeman 1991); UNDER L1's CHANNELS (§3.3); the relaxation of revision 3 measured 15 min a tile and retires | `D` = 0.003 m²/yr, `S_c` = 1.25 (Roering et al. 1999, Oregon, at the hillslope's own scale; B82); the ABSOLUTE rise from the planet's present rise, `K` so L1's `U/K` holds |

Where the solve has no water (`Q = 0`: an airless moon, a dry body) the joint law does not apply and no fluvial level is
solved there (§3.6) — never a slope from a division by nothing (A11, B-I1).

### 3.3 The mean pyramid and the mean hold

**The mean pyramid.** Each stored tile holds its raster and the box means of the raster over 2 × 2 and 4 × 4 cells —
the levels rungs `r + 1` and `r + 2` read (§2). A chunk reads the level whose cell equals its own cell; so a rung swap
compares two box-mean levels of ONE solved surface, and a finer level's arrival changes a coarser rung's ground by
nothing (A1).

**rev 4 — what replaces the hold below (B84, B83 item 5; the passage after this one is revision 3's and is refuted):**

- **L1 IS THE LAW ALONE.** Each node's `U/K` comes from B80's rounds: the same law first on points four times further
  apart (a coarse start, ~1 s), then three rounds at L1 whose tree passes stop when no node's mean moves more than 5 m;
  NO correction after it. MEASURED: no pit (every height is computed along the flow tree); each node's mean differs from
  its row by at most 1 233 m on the roughest benched tile — 0.15 of a pixel where the rows take over (7 510 km). The
  design's linear hold (below) took 6 min a tile and made cliffs; the additive hold dug hollows up to 1.2 km deep; a
  fill loop never converged; the slope hold at L1 steepened 10–22 % of the points past 45°. **G-MEAN is "the swap to the
  coarser level under a pixel at its switch".**
- **L2 FOLLOWS L1's CHANNELS AND IS HELD PER L1 CELL BY ITS SLOPES.** (1) An L1 link is a CHANNEL where L2's own law says
  its water cuts (the joint law's slope under the creep's at `ℓ = A/width`, the comparison the steady form makes at
  every point); the L2 points within ¾ of a spacing of a channel link are outlets at L1's height there. (2) After the
  steady form, on its FIXED tree, each link's slope is scaled by a factor per L1 point's own VORONOI cell (irregular, so
  no grid prints), read with Gaussian weights (σ half an L1 spacing), half the logarithmic step a sweep, until each cell's
  mean meets L1's surface over the same points; a cell with fewer than four free points or under a metre of movable
  relief is left to the law. Every link still falls, so no hollow forms. MEASURED: the hand-over to L1 974 → 48 m (0.4 px
  at its switch), L2's own swaps 41–42 m; two tiles and one joint solve AGREE EXACTLY in the overlap (each height depends
  only on its own catchment between two channels): L2 is seamless by construction.

**B96 (2026-10-14) — L2 IS TODAY'S L1 GROUND PLUS THE LAW'S FINER DETAIL (amends the hold by slopes above).** Benched
on hull 17's hills, the hold fought L1's ground (two laws at two levels) and made walls. L2's law is now solved ALONE —
the steady form under L1's channels, settled by B93's rule one level down (the drainage built again every second pass,
stopped when no L1 cell's mean moves 1 m) — and the stored raster is L1's drawn curve plus the law's raster less its own
L1 version (its box means over each L1 cell, read back by the same Catmull-Rom). MEASURED: each L1 cell's mean within
5.1 m of L1's, 1.49 % of the links over 35°, the joins 12 m at the 99th percentile after the one cut, 3.7 s a tile.

Revision 3's text: **The mean hold** makes every level's mean over each coarser cell equal that cell's value: L1 over
each macro node equals the node's row; L2 over each L1 cell equals that L1 cell. It is solved, never iterated by a knob
(B-B4):

- **L1.** On a FIXED flow tree, a point's height is its outlet's height plus a sum of `S·L` along its path, and every
  `S` below the threshold is `V(x)·Q^(−1/2.2)` with `V = (U/K)^(1/2.2)`, read at the point as a weighted sum of the
  nodes' `V` by the partition of unity — LINEAR in the nodes' `V`. The node means are then a linear function of them: one non-negative least-squares solve of ~100 unknowns (the
  tile and its buffer nodes) gives the `V` that holds every mean; links at the threshold join an active set. The tree is
  then found again from the new heights; the loop stops when the tree repeats (a CYCLE is detected and its lowest-mean-
  error tree kept), with N against 2N passes as its convergence test (B39).
- **L2.** After the relaxation, the residual of each L1 cell is removed by Tobler's mean-keeping smooth correction
  (1979: one sparse solve that keeps every cell's mean exactly and is smooth across cells), then any pit it made on a
  channel is breached along the flow tree and the cell's mean held again (B-I9); G-PIT counts the pits left (none).

### 3.4 The seams between tiles

**rev 4 — measured (B0 steps 2–2b):** the buffer is TWO nodes at L1 and 2 km at L2 (with one node / 1 km a cut ran along a
valley the two halves placed apart); the cut is free over HALF the buffer; the blend across it is 8 cells; a river that
leaves a domain leaves at the lower of its link's two node means (a property of the link, the same on both sides); the
domain's region reads every 8 km link that crosses its rim and each node's water from the artifact's own discharge. Two
L1 tiles: the slope break across the cut 1.33 against 1.25 inside (the cut not found in the picture); four L2 tiles at one
corner: 2.31 against 2.30; L2 under L1's channels: identical to the joint solve.

**B85 — the corner rule, measured:** each tile edge's cut runs in TWO HALVES, from each of its corner points to its
midpoint, with the ends fixed; a corner point is the cell within the cut's band where the four tiles meeting there
agree best (the least sum of the six pairs' squared differences; ties to the lower row, then column), a midpoint the cell
on the edge's middle line where its two tiles agree best (ties west, or north). Every cut is then a pure function of the
two tiles beside it and its two ends, and a drawn cell depends only on its tile and the eight around it (`fine::stitch::
tile_cuts`, `finished_tile`): a cell is joined within its row of tiles across the nearer column cut, then across the
nearer row cut, each by the 8-cell blend. The cut's band is a node less the blend (±56 cells at L1), so no blend reads past
a raster that reaches one node beyond its tile. MEASURED on sixteen solved belt tiles: the same slope breaks as the
bench's two-step stitch (1.669 / 1.669 across the cuts at the 99th, 1.304 / 1.307 at the four-tile corner), and the
finished tiles' straight edges read as the stitched ground on the same lines (the step 9.17 against 9.25 m at the
median) — the two tiles that meet there were drawn apart and agree. The order this replaces (each row of tiles along its
column cuts, then the rows along one row cut) ran one shortest path along a whole row of tiles, so a tile's ground
depended on tiles without end.

A tile's DOMAIN is the tile plus a BUFFER that is solved but never drawn as-is. Three rules make two neighbours agree in
the overlap, and a CUT decides who draws each point (B-B2):

1. **The same points and edges** (§3.1).
2. **Only the coarser rivers leave the domain.** The domain's rim is closed except where the coarser level's stored
   flow tree crosses it (a crossing list in the coarser tile's head: the place, the water, the height on the coarser
   river's own profile). Both neighbours read the same crossings, so their valleys run through the overlap to the same
   place. The rim's other points are not outlets: no drainage runs to an artificial edge.
3. **The same means** (§3.3).

**The minimum-error cut** (Efros & Freeman 2001) — the ONE stitch of every border (B81): inside the overlap, a path from one corner of the shared edge to the
other that follows where the two solutions differ least — mostly along ridges, where both agree on the divide. Each side
draws its own solution up to the cut; a blend only a few cells wide crosses the cut. The cut is an integer shortest-path
search over the two stored tiles (~33 000 samples an L2 edge) with ties broken by the cell index; the shard stores it
with the later tile, so the client reads it and never searches. **G-SEAM** reads, beside a tile's interior: the pits per
km², the relief's spread, the valleys that end at the cut, the height step and the slope break at the 99th percentile;
and it compares the stitched pair with ONE solve of both tiles together (the reference with no border at all). **G-SEAM
is BLOCKING (B81):** a seam the eye sees in the pictures, or readings outside the interior's own spread, means no build.

**The far levels are made from the stitched ground** (B81): a tile's 2 × 2 and 4 × 4 mean levels are the box means of
the DRAWN surface — its own solution up to each cut — so they exist once the tile and its stored neighbours are cut;
until then the far chunk waits (the coarser rung stands). No straight tile edge exists at any rung.

**Example.** The L2 tiles under nodes (3, 184, 556) and (3, 185, 556) both solve the 2 km overlap. L1's river crosses
their shared edge at one stored place; both solutions carry it there. The cut runs along the divide north of that river,
so the gullies on each side are whole and none is drawn twice.

### 3.5 What the solve must show about its scale

Perron's length `ℓc = (D^n·U^(1−n)/(K·R))^(1/(n+2))` (Perron, Kirchner & Dietrich 2009; Theodoratos 2018 for `n ≠ 1`)
is computed per L2 tile from its rise and rain and stated in the tile's head (a reading). Where `ℓc` is under the L2
point spacing, the law's valleys are finer than the points (B80's "feather"); the bench counts such tiles on the home
planet; on the fastest belt `ℓc` is 24–30 m, above the 16 m points.

### 3.6 Where the levels apply: a law, never a flag on a body

- **L1** where the macro solve's water exists (the joint law's `Q > 0`). A body with no water (an airless moon) has no
  fluvial level; its ground stays today's until its own families (craters, Neukum & Ivanov) exist (B79 Q4).
- **AMENDED by B95 item 3 (2026-10-13): L2 ON ALL LAND A PLAYER WALKS ON** — a still plain and old hills get their 16 m
  gullies by the same law (each node's rate read back from L1's ground by the joint law); the cost stays local (solved
  around occupants and their arrival places). The paragraph below is the superseded rule.
- **L2** in every tile where the present rise is above zero ANYWHERE in its domain. The rise is read from the map by
  the same smooth weights as `V` (§3.3), so it falls to zero smoothly toward the still ground, and the LAW fades the
  gullies with it (B81: no line where the gullies stop). A tile whose whole domain is still is not solved: its L2 is
  L1's raster by Catmull-Rom — the same surface the fading neighbour reaches at its still edge, stitched by the one cut.
  About 96 % of the L2 jobs are saved (B-I1). The still world's own cure is the macro laws' (the rain, B78; uplift
  outside the belts), not this design's.
- **The coast** (B81): the sea floor keeps today's ground until its own family (abyssal hills) exists. The land's
  solve treats every sea point as an outlet whose height IS the sea floor's own drawn height there (the same column
  read), so the land meets the sea floor at exactly the sea floor's surface — no step at the coast. When the sea floor's
  family arrives it joins the same machinery and the same stitch.

### 3.7a Every place two pieces of ground meet (B81: no seams)

| Where | Its one treatment | Measured by |
|---|---|---|
| two tiles of one level | the shared points and edges, the coarser rivers as the only rim outlets, the same means, the minimum-error cut | G-SEAM against one joint solve (blocking) |
| two levels (L1 under L2, L0 under L1) | the mean hold: each finer level's mean over each coarser cell is that cell (§3.3) | G-MEAN |
| two rungs | the mean pyramid (rung `r` reads the `2^r` m level of ONE drawn surface) and today's crossfade | P-SWITCH, the flight's pop detector |
| a rising region and still ground | the rise fades smoothly; the law fades the gullies; the cut stitches the last tile | G-SEAM |
| the land and the sea floor | the coast points' outlets take the sea floor's own drawn heights | G-SEAM at the coast |
| a live tile and a waiting one | today's rule: the waiting chunk is not drawn; the coarser rung stands and is crossfaded | the flight's pop detector |
| the cube's faces and corners | the Delaunay graph on the sphere (§3.1): no face, no corner exists for the solve | G-GRID at a corner |
| a built patch and the new ground | F1's blend band (ruled) | F2 as amended (B81 Q8) |

### 3.7 Determinism (B64, SL10)

- The solve runs under `Gf` (IEEE-exact `+ − × ÷ sqrt`, the fenced `exp`, `ln`, `powf`; `terrain/clippy.toml`).
- EVERY SUM over points (the water's accumulation is a tree walk in a fixed order; the conjugate gradient's dot
  products; the means) runs over FIXED BLOCKS of 4 096 points in index order and joins the blocks' partial sums in
  index order — never over ranges cut by the thread count (`par::ranges`, `par.rs:42`, cuts by threads; a new
  `par::blocks` cuts by a fixed size) (B-B1).
- The heaps order ties by the point's index; the cut breaks ties by the cell index; the Delaunay predicates are exact.
- The stored tile is integers (quarter metres). P-SAME compares a tile's bytes on 1 and 4 cores and on aarch64 and
  x86-64 before the build.

## 4. The path: solve, store, go live, ship, read

### 4.1 Who solves, and when (SL7, SL9)

**The fine worker** is a capability of any realm with ground (HR4), run in the realm's own shard beside today's artifact
worker, on the solve's share of the cores (`VD_SOLVE_CORES`, half by default). It starts once the artifact is whole and
the present-rise map is stored (§4.2). It runs one tile per core of the share at once; tiles are independent.

**The reasons** (the realm's own, SL7) — each a set of tiles, changed only on an EVENT, never rebuilt each tick (SL9):
1. **Its own occupants** at their own poses, and **its occupied direct children** (a landing hull) at the placements the
   realm authored for them: the tiles within each one's reach (§5.3), the reach growing with the closing speed (the
   movement answers, item 2). An event: an occupant or child enters, leaves, or crosses a tile's edge.
2. **Idle** (Q4): L1 tiles around the realm's own arrival places (its berths and its spawn places — places the realm
   holds), only while the realm is alive for other reasons; idle work never keeps a realm alive (SL7).
3. **Path B is never a reason.** A viewer in another realm is served tiles already stored (§4.4) and never queues a
   solve (SL2: no foreign pose warms the realm).

**The order.** By the time to reach (distance over closing speed), the coarser level first, ROUND-ROBIN across the
reasons so a fast flier never starves a walker (B-I8). An L2 job waits until every L1 tile its domain meets is stored.
A job whose reasons have all left is dropped before it starts; a started job runs to its end (a half tile is never
stored). The worker states its capacity (tiles a second) as a reading.

### 4.2 The store

- One row family in the realm's redb store (`vd-sim`'s `built_store.rs`, prefix 9, free today), one tile one commit,
  durable.
- **The key:** `[9, input digest, level, face, tx, ty]`. The INPUT DIGEST folds the artifact's digest, the charter's
  words, the present-rise map's digest and `FINE_VERSION` (B-I6, A5), so a world change (generator 27, B78) never serves
  an old tile.
- **The head:** the input digest, the tile's own digest, its range (lowest, highest), the CROSSING LIST of its flow tree
  at the domain's rim (the place, the water, the height), its nodes' `V` (§3.3), its `ℓc`, the cut paths to its
  neighbours already stored. L2 reads only stored L1 bytes (B-I4).
- **The present-rise map** is solved once per planet (177 s on the bench) and stored beside the artifact, keyed by the
  artifact's digest.
- Every read verifies the tile's digest, on the shard and on the client; a mismatch solves it again (the bytes are the
  same, §3.7). A store lost or moved to another host is rebuilt by solving again.

### 4.3 Going live (SL10, V1: the player stands on what the player sees)

A tile becomes LIVE when it is stored AND every session within its reach holds it (each client acknowledges a held tile,
as `ArtifactHeld` does for the artifact, `channels.rs:107-115`), or after a stated wait for a laggard (the laggard's
chunks keep waiting; its coarser rung stands). From that tick the realm's colliders (owed; none exist today) and every
client read the tile. Before it, every host reads the coarser rung's ground: one live set, one shape.

**Under a standing occupant** (Q6): if a tile goes live where someone stands (a fast landing that beat the queue), the
ground under them changes by the fine detail around the same mean. Recommended: the realm lifts or lowers each body
standing in it with the ground over the arrival ramp (0.5 s). The reaches of §5.3 make this rare.

### 4.4 The ship

- **One** new bulk part variant, `BulkMsg::FineTilePart { level, key, part, parts, bytes }` (HR3: one variant with a
  level, not one per level), relayed undecoded by the gateway as `ServerControlMsg::ArtifactPart` already is — SL6 ASK
  Q3; `PROTO_MINOR` + 1 (36 → 37), the floor unchanged.
- **Parts of 32 KB** (as the pyramid's), paced by a BYTE budget per occupant per tick beside the artifact's (B-I2, C-I4);
  the coder states its worst case (4 bytes a sample), so no part passes the 1 MiB frame (`framing.rs:17`).
- **B85 — THE FINISHED TILE ships, and nothing else:** the tile's own square after its cuts (512 × 512 at L1 on the home
  planet) and its mean levels (rungs 8–12), each through the coder below; no raw buffer and no cut paths, so a client
  never stitches and every host draws one surface. A tile is finished once its eight neighbours are solved (the reach's
  rings solve them; the coarser rung stands meanwhile). MEASURED on the belt: 303 KB (1.16 bytes a sample, 10 parts of
  32 KB) and 115 KB for the five mean levels, every level decoded back exactly. The rev-4 item below is replaced.
- **What a tile ships, by distance (rev 4, replaced by B85).** Near: the raster over the tile and HALF its buffer — 640 × 640 samples on the
  home planet (+56 %; the cut lies inside it) — and the cut paths. Far (rungs 8–12): only the DRAWN surface's 2 × 2 …
  32 × 32 mean levels over the tile, made after the cuts — no straight tile edge at any rung (B81).
- **The coder (rev 4, B83): THE ADAPTIVE RICE CODE** (JPEG-LS's; Weinberger, Seroussi & Sapiro 2000): each sample in
  quarter metres predicted by LOCO-I's median rule from its decoded left, upper and upper-left neighbours; the residual
  zigzagged and written as its quotient by `2^k` in ones ended by a zero and its `k` low bits, `k` from the running mean
  of the residuals' size (halved every 64 samples); an escape at a quotient of 8 to a plain 24-bit word, so no sample
  costs over 4 bytes. A tile decodes alone. Our own code. MEASURED (every level decoded back exactly): belt L1 468 KB near
  (1.17 bytes a sample, 15 parts of 32 KB) and 105 KB far; plain L1 353 / 82 KB; belt L2 347 / 83 KB — within 4 % of the
  residuals' entropy. (Revision 3's byte-wise code measured 634 / 145 KB on the belt and failed P-SIZE.)
- **Path A** ships to each occupant and occupied child the tiles of its reach that it does not hold (the gateway's held
  record). **Path B:** the gateway caches stored tiles once per realm, BOUNDED IN BYTES (newest refused, as today's held
  bulk), and serves a session the far levels in its view's reach.

### 4.5 The reading and the ladder's bounds

- **One fine read for every reader** (C-I6): `ColumnRead::column`, `height::field_column`, the morph target's coarser
  field, `field_normal`, the eye's `drawn_surface_m` and the shard's `ground_m` all read the column through ONE function,
  so the shard's tile ring and the client's ring never part (W17, B28).
- **The smallest change** (C's): one method on `ZField` — `fine_z(face, rung, i, j) -> Option<Gi>`, `None` by default;
  `ColumnRead::column` and `field_column` pass the fine height in place of `Z` and start the octaves under the level's
  cell (today: none under 48.8 m); `column_surface_at`, the terrace, the biome and the extractor unchanged; `field_for`
  answers `Waiting` until the live tiles are held, naming the missing ones as it names missing artifact tiles today.
- **B1 (g) — as built:** the read is `fine::read::FineBook::z(face, draw_rung, cell_rung, i, j)` → Held / Missing /
  Outside; it reads level `max(draw_rung − 7, 0)` at the centre of a cell of any rung through the macro field's OWN
  Catmull-Rom kernel (`vd_recipe::macro_field::catmull_rom_16`, so no new kernel enters `vd-recipe`); a column at rung
  `7 + k` stands on its own sample and reads the mean exactly. `ZField::fine_z` answers Outside by default;
  `ColumnRead::column` and `field_column` take Held with no octave and refuse on Missing. B85 retired the cut's side test
  (the client never stitches) and moved the mean levels to the shard alone (`fine::stitch::mean_levels`). Measured on
  the belt: exact at rungs 7 and 12, no step across a tile edge at 1 m, 0.08 µs a read.
- Revision 4's text — **The integer kernels** (the Catmull-Rom over a fine raster, the cut's side test, the mean levels) live in `vd-recipe`
  beside the macro field's kernel (F7); the tile book and its lookups in `vd-terrain`.
- **rev 4 — P-SWITCH and P-BAND MEASURED (B83):** at L1 every step from rung 7 to the node is under the coarser cell's own
  one-pixel distance (belt: the max 128 m at 7 → 8 subtends a pixel at 118 km against the 235 km switch; 594 m at 11 → 12
  at 545 km against 3 755 km); the law's bound `S_c·0.71·(cell_r + cell_r+1)` is 0.75 of the coarser cell at tan 35°, so
  NO rung's switch moves and no chunk is added. At L2 (after the fix): 4 → 5 max 41 m (99th 14 m; 1.3 px at its switch at
  the worst point — owed in B3), 5 → 6 42 m, 6 → 7 48 m. THE BAND: the fine ground over / under its node's mean +3.2 /
  −2.5 km at most on the benched belt tiles, ±0.24 km on the plain; the band widens by the law's ±4.1 km (B83 item 3).
- Revision 3's text — **The fine step bound** (C-B3, A1). Every ladder bound reads the octave table today (`step_bound_m`, `sink_m`,
  `peak_m`, the span's `column_bound`, the tile rings). For a body whose fine levels apply, each rung pair `(r, r+1)`
  states ONE fine step bound for the whole body (as B55 states one band): the most two box-mean levels of one surface
  can differ at a point under the threshold law — `S_c` × 0.71 × (cell_r + cell_{r+1}) — 21 m at rungs 3→4 and 680 m at
  8→9 on the home planet. Rule 3 would then floor rung 3's switch at ~9 km (today ~7 km) and rung 8's at ~296 km (today
  222 km): more chunks at those rungs. **P-SWITCH** measures the bound against `rung_disagreement`'s maximum on benched
  tiles and the chunk counts at each; the owner chooses with the numbers (Q7).
- **The band (B55)** needs the fine levels' bound before any tile is solved. The law's bound (`S_c` × a node's diagonal,
  ~14.5 km) costs ~41 % more reach for far peaks and ~2.6 ms a frame of shadows (B-I7, C-I7). **P-BAND** measures max
  |F − row| on benched tiles of the three stands; the owner chooses (Q7).
- **The identity check** (C-I8): `golden_self_check` gains a small golden fine literal, so a drifting fine reader fails
  at login; a reader change bumps `GENERATOR_VERSION` (the column rule's own version, `tag.rs:348`); `FINE_VERSION`
  enters only the store key.

### 4.6 Versions and the frozen patch (T3, F1, F2)

- A new fine law is ONE WORLD MOVE (A4): `FINE_VERSION` moves, the pins move with it, every tile's input digest changes,
  and every visited tile is solved again as wanted — never a silent drift of ground a player stood on between visits
  under one version.
- A built patch (F1) records its tiles' input digests (one per level); the store keeps those tiles' bytes, and the column
  read inside the patch reads them (a frozen patch smaller than a tile: the tile's other part reads the new version, the
  F1 band blends the two).
- F2's half-metre bound was set for an ADDED OCTAVE; a new fine law moves unbuilt ground by metres — ASK Q8.

### 4.7 What changes, by crate

| Crate | Change |
|---|---|
| `vd-recipe` | the fine raster's integer Catmull-Rom, the cut's side test, the mean levels |
| `vd-terrain` | the `fine` module: the points and the spherical Delaunay, the L1 and L2 solves, the mean hold, the cut, the coder, the tile book's lookups, `fine_z`, the fine step and band bounds; the present-rise reading (from the bench); `par::blocks` |
| `vd-sim` | the store family (`built_store.rs`), path A's per-occupant and per-child ship by bytes |
| `vd-bins` | the fine worker beside `artifact_worker`, its reasons and queue |
| `vd-wire` | `BulkMsg::FineTilePart`, the client's acknowledgement; `PROTO_MINOR` 37 |
| `vd-connection-plane` | path B's byte-bounded cache and serve, the held record |
| `vd-client` | the fine tile book (an LRU bounded in bytes; F1-pinned tiles kept), the `Waiting` rule for fine tiles, the acknowledgement |
| `client-render` | nothing |

## 5. Performance

### 5.1 Measured on the bench (one core of the M4 Pro, platform floats, 2026-10-11)

| Job | Points | Time | Where it goes |
|---|---|---|---|
| L1, 98 km region | 590 000 | 48–94 s | 10 rounds of the iterated hold × up to 60 tree passes (the tree never settled) |
| L2, 8 km window at 16 m | 250 000 | 100–140 s | 400 steps × (tree + Newton + 2 conjugate-gradient solves) |
| The present-rise map | 8.9 M nodes | 177 s | once per planet |

The bench used the platform's `powf`; the fence's `Gf::powf` costs about five times more (about twenty dependent
divisions, `gf.rs:177-227`), and L2's Newton calls it ~8 × 10⁸ times a tile (B-B3). So the bench's numbers are a floor,
not a forecast, until P-COST measures under `Gf`.

### 5.2 The levers (each measured on the bench; none may change the bench's pictures)

| Lever | For | Expected |
|---|---|---|
| the mean hold as one linear solve on a fixed tree (§3.3) | L1 | the ten rounds become a few tree passes |
| the tree's passes stop on a repeat or a detected cycle | L1, L2 | no 60-pass tails |
| a FINE POWER: `x^(1/2.2)` and `x^2.2` as the fence's `exp(c·ln x)` with the constant `c` precomputed and `ln`, `exp` as fixed polynomials (no division) | L1, L2 | near the platform's cost, the same bytes on every host |
| Newton warm-started from the last step's root | L2 | 1–2 iterations, not 5 |
| fewer, longer implicit steps, N against 2N deciding how few (B39) | L2 | 400 → ~100 |
| an incomplete-Cholesky preconditioner for the creep's solve; an absolute tolerance of a quarter metre | L2 | about half the iterations |
| barycentric interpolation on the Delaunay triangles in place of natural-neighbour | rasters | ~50 × cheaper |
| no L2 where nothing rises (§3.6) | L2 | ~96 % of the L2 jobs gone |

**Targets, per tile on ONE core of the share** (B-B3: one target, the arrivals derived from it): L1 ≤ 5 s, L2 ≤ 20 s.

**rev 4 — MEASURED on the bench (B0 step 3; one core of the M4 Pro):** an L1 tile (605 000 points with its buffer) 2.3–6.2 s
(65 s before); an L2 tile 13.5 s with the fix (16.9 s before). The levers that carried it, each proved byte for byte or by
equal readings: the flow tree in LINEAR TIME on ground with no pit (each point to its steepest lower neighbour, the order a
walk up the donors; any pit falls back to the priority flood), the graph in flat arrays, the neighbour scan with selects,
the coarse start, the rounds' passes stopped when the node means settle (5 m), the hold's weights built once. REFUTED:
the linear hold (6 min a tile), the one loop (river captures never settle), a tie tolerance, reused lists, a slope cache.
THE FENCE (P-COST): the fence's own `powf` costs 2.7× the platform's (not 5×); the FINE POWER (the fence's series with each
division by a whole number replaced by a multiplication by its reciprocal written once) costs 4.2 ns, worst relative error
8.3 × 10⁻¹⁵, +7 % a tile — the build's power. ONE TILE A CORE: 4 L1 tiles at once 6.6 s of wall time (each 3.3–6.6 s) —
~0.9 tiles a second on 4 cores (the need 0.8), ~1.3 on 7.

### 5.3 What the reaches need (4 cores of the share; L1 0.8 tiles a second, L2 0.2)

| Who | Speed | Needs | Time on a never-solved place |
|---|---|---|---|
| A walker on a rising belt | 5 m/s | the L2 tiles within rung 6's reach (~56 km → only the rising ones), the near L1 ring (rung 7, ~111 km: ~12 tiles), the far L1 levels to rung 9 (445 km: ~145 tiles) | near L1 in ~15 s, L2 under the eye in ~25 s, the far ring in ~3 min (rung 10 stands there meanwhile) |
| A rover or a low flight | 100 m/s | a new L2 row (3 tiles) every 82 s | 15 s a row: holds |
| A fast low pass | 1 km/s | a new L1 row every 65 s; L2 not asked (the bounded ask, F9: detail drops at speed, never a hole) | 4 s a row: holds |
| A descent from orbit | 2 km/s down | the L1 rings under the descent from ~450 km up | done during the descent |
| Many rovers on new ground at once | — | ~5 rovers fill 4 cores (B-I8) | round-robin keeps each one's nearest first; the capacity is a reading |

### 5.4 Bytes, store and memory

- **A near tile:** ~400–600 KB (576² samples at ~1.2–1.8 bytes, to be measured — P-SIZE). **A far tile:** ~100 KB.
- **An arrival on a rising belt:** ~12 near L1 + ~9 L2 + ~145 far L1 ≈ 12–25 MB, near ones first, paced by bytes.
- **The store:** only solved tiles; a busy region holds hundreds (~100–300 MB).
- **The client:** the decoded tiles in reach plus a margin, an LRU bounded in bytes; pinned (F1) tiles kept.

## 6. The gates — on the bench first (B0); each heavy run asked first (B74); the owner's look decides (B76)

| Gate | What it reads | Passes when |
|---|---|---|
| P-COST | L1 and L2 tile time on ONE core under `Gf` | §5.2's targets |
| P-SAME | a tile's bytes on 1 and 4 cores, and on aarch64 and x86-64 | identical |
| G-SEAM | across the cut and at the far levels' straight edge: pits per km², the relief's spread, valleys that end at the cut, the height step and the slope break at the 99th percentile — beside the tile's interior | within the interior's own spread; the far step under a pixel at its distance |
| G-MEAN (rev 4, B84) | each swap between a level and the next coarser: the step at the 99th percentile and the max | under one pixel at the swap's switch distance |
| G-PIT | pits after the solve (L1: the law alone; L2: after the slope hold) | none |
| P-SIZE | bytes a near and a far tile, and the coder's worst case | ≤ 600 KB and ≤ 120 KB; no part over 32 KB |
| P-READ | the chunk build's time with the fine read, per rung, against today (`chunk_budget`) | not slower at any rung |
| P-SWITCH | the fine step bound against `rung_disagreement`'s maximum; the chunks per rung it adds | the owner's choice (Q7) |
| P-BAND | max \|F − row\| on benched tiles; the reach and the shadows' cost of each band | the owner's choice (Q7) |
| G-NO-DRIFT | the column bytes from the same tiles on the server's and the client's builds; the golden fine literal | byte for byte |
| G-GRID | the straight-line share and the lean on solved tiles; the three point sets | near Earth's, as B80's maps |
| G-IDENTICAL | the fine worker's fixture on two shard kinds | identical |
| F6 frame | the window's frame with the shard and the client on the owner's Mac | not worse than today |
| G-LOOK | the belt stand, a plain, a seam, in the window (opened only on the owner's word) | the owner's word |

## 7. Stated choices and interims (B23, B38: each named)

- **The point set:** the hashed Poisson-disk (B83); its radius `cell/√π` is the one that keeps one point a cell.
- **The buffers** (rev 4: two nodes at L1, 2 km at L2) and the cut's band (half the buffer), the blend (8 cells): G-SEAM
  decided them; widened only by a measurement.
- **rev 4 — THE SOLVER'S NUMBERS** (numerics, never physics; each with its convergence reading): the coarse start's
  points four times further apart; three L1 rounds (three warm rounds gave the readings of ten cold ones: the straight
  share, the lean, the pits, the steep shares within the twins' spread); the rounds' passes stop when no node mean moves
  over 5 m; the L2 hold's half logarithmic step, its four-free-point and one-metre rules, its stop at 1 m. **THE ROUNDS'
  STEP CLAMP** (each node's rise changes by a factor between ¼ and 4 a round) is a clamp inside an operator (B23): B1
  replaces it by the damped logarithmic step and measures it before the generator takes it.
- **The block size** of the fixed sums (4 096): any fixed number gives the same bytes on any core count; this one is a
  cost choice.
- **The laws' constants:** `n` = 2.2 (Adams et al. 2020), `S_c` = 1.25 and `D` = 0.003 m²/yr (Roering et al. 1999,
  Oregon Coast Range) — Earth calibrations of laws, as the macro solve's (B41: Earth fixes a constant no law computes yet).
- **The creep past `S_c`** flows at its rate at `S_c` (numerics under a fixed buffer edge; the threshold law holds the
  slope).
- **The present rise** leaves out the rebound of the eroded rock (the macro solve's own).
- **The rock's scatter** stays off (its spread is assumed, not sourced).
- **The levels' cells** (rung 7 and rung 4): judged by the look and by Perron's length (§3.5).

## 8. The owner's questions — ANSWERED (B81: every recommendation accepted; Q1 amends B77, Q8 amends F2, Q3 is SL6 yes)

1. **Q1 — the tile solve, and the ground under 16 m.** B77 said "no bake: forms per point". B80's method solves a tile
   at once. Accept the TILE SOLVE (each tile solved once per world version and stored) as the way below 8 km? Under the
   L2 raster the ground is its smooth curve until a later level L3 brings the law down to 2–4 m (B80 item 3); no noise
   is added there (B35). Recommended: yes to both.
2. **Q2 — where the solve runs.** (a) The planet's shard solves, stores and ships (recommended: the realm computes and
   draws itself; one solve serves every visitor; the client's cores stay with the game). (b) Every client solves its own
   tiles (the SL6 default: no new data crosses; each client spends ~5 s an L1 tile and ~20 s an L2 tile of its own share,
   ~145 far L1 tiles on an arrival, and waits; the realm's colliders must solve them too).
3. **Q3 — SL6.** With (a): the fine tiles cross from the planet's shard to its own clients through the gateway — ONE new
   bulk part variant; ~12–25 MB on an arrival on a rising belt, then a few tiles as one moves; the client acknowledges
   held tiles. Yes?
4. **Q4 — the idle pre-solve.** While the realm is alive, solve L1 around its own arrival places when the share is idle
   (recommended), or strictly on demand?
5. **Q5 — the first visit.** On a never-solved place the coarser rung stands until the tiles are live: the near ground
   in ~15–25 s, the far ring in ~3 min (rung 10's 1 km ground meanwhile). Once per place per world version. Accept?
6. **Q6 — a tile live under a standing body.** Recommended: the realm moves each standing body with the ground over the
   0.5 s ramp.
7. **Q7 — the two bounds.** The fine step bound per rung pair (pushes rung 3's switch ~7 → ~9 km and rung 8's 222 →
   ~296 km) and the band (the law's ~14.5 km, +41 % reach): take the law's bounds, or decide on P-SWITCH's and P-BAND's
   measured costs first (recommended: measure first)?
8. **Q8 — F2 for a new fine law.** F2's half metre was set for an added octave. A new fine law moves unbuilt ground by
   metres; built ground stays pinned (F1). Amend F2 so: "a new fine law may move unbuilt ground; built ground never
   moves"?

## 9. The build order (each step ends with the owner's look; tests only on the owner's word)

1. **B0 — the bench, no game change:** the levers under `Gf` (P-COST); the point sets and the spherical Delaunay
   (P-SAME, G-GRID); the linear mean hold (G-MEAN, G-PIT); two neighbour tiles, their crossings and the cut (G-SEAM); the
   coder (P-SIZE); the column read (P-READ); the bounds (P-SWITCH, P-BAND). Pictures: the belt stand, a plain, a seam.
2. **B1 — the generator's fine part** (`vd-recipe`, `vd-terrain::fine`, Tier-A, under the fence; rev 4's list, in order):
   (a) the Poisson-disk point set on the sphere from the leaf's hash; (b) OUR OWN spherical Delaunay with exact `i128`
   predicates and its Voronoi areas; (c) the fine power and the fenced reads (the Gaussian partition, the Catmull-Rom of
   the macro field, the rain) in `Gf`; (d) the L1 solve: the linear-time tree with the flood's fallback, the steady law at
   tan 35°, the coarse start, the rounds with the damped step (measured on the bench first), fixed-block sums; (e) the mean
   pyramid to 32 × 32 and the cut; (f) the Rice coder and the tile's head; (g) `fine_z` and the integer read in
   `vd-recipe`; unit tests on small domains (run on the owner's word, B74).
3. **B2 — L1 in the game:** the fine worker on the solve's share, the store family, going live, path A, the wire part
   (`BulkMsg::FineTilePart`, B81 Q3), the client's tile book and `Waiting` rule, the band ±4.1 km. The owner's look at the
   belt stand in the window (opened on the owner's word).
4. **B3 — L2 in the game:** the steady form under L1's channels with the hold per L1 cell (rev 4), its two owed cures
   (cells with no slope left to the law; the 4 → 5 worst point), its tiles, the reaches.
5. **B4 — the rest:** path B's far levels, the idle pre-solve (Q4), the client's eviction, the F1 record (with slice 9).
   **B86 (2026-10-12) re-orders B2's end and B4** (the final ground from the first frame): (1) the face-edge cut — every
   tile finishes, at a face's edge and the cube's corner; (2) the safety net — the coarser rung fills only empty places,
   never over arrived fine ground (amends the own-height stand-in of 2026-09-30 for the fine level; §4.5's waiting rule);
   the owner's look; (3) far copies first — a tile's levels ship apart, by the distance that reads them (§4.4); (4) the
   send-ahead by the closing speed (§4.1, already this design's reach rule); (5) the loading screen OVER the running game,
   at login only — no teleport, no special path; (6) the advance calculation — the whole body once per world version,
   after §4.1's reasons (the unvisited planet's case open); (7) the player's disk cache keyed by the input word, the held
   list in the acknowledgement of §4.3. **Built (2026-10-12, not run):** (1) the face-edge cut — each corner point, midpoint
   and half in the chart of its lowest tile, turned into each tile's chart; a cube corner's three tiles by a symmetric
   product of their pair weights (`vd_terrain::fine::finish`, `stitch::Meet`); (2) the safety net — a chunk whose fine
   tiles have not landed is built on the rough view and built again when they land (`FieldPick::Rough`,
   `ChunkLane::upgrade`), replacing §4.5's waiting rule for the fine level; (3) far copies first — each tile shipped as
   the range of its levels from the finest its view reads (`fine::tile::range_record`, PROTO_MINOR 38), the finer
   levels following when the occupant nears (measured at the stand: 702 tiles in ~15 s, 1 776 parts against ~9 100).
   **B3 built (2026-10-14, not run; B96):** the level descriptor (`vd_terrain::fine::level`) and every tile path by level —
   the 16 m tile one node, its raster 64 cells past it, two mean levels (rungs 4–6); the planar triangulation on the cube
   face for its points (`delaunay::triangulate_on_face`: the sphere's in-circle test is at the directions' rounding at
   16 m); the shard's fine work keyed by level, a 16 m tile solved once its 3 × 3 finished L1 tiles are held; the gateway's
   needs per level; the client's second book and the COARSE state (a chunk on the L1 ground until its 16 m tiles land,
   built again then).
6. **B5 — L3:** the law at 2–4 m in a small ring around a walker (its cost measured first).

## 10. The reviews' findings and their answers (revision 1 → 2)

| Finding | Answer |
|---|---|
| A1 a coarse rung reads points, not means; noise then gullies is a contradiction | §3.3 the mean pyramid (rung `r` reads the `2^r` m level); §2 the waiting rule (no octave-drawn stand-in at a fine level's rung); §4.5 the fine step bound |
| A2 SL7/SL2: who queues a solve; a hull's pilot gets no L2 | §4.1: own occupants and occupied direct children at authored placements; path B never a reason |
| A3 SL9: the queue walks every reason | §4.1: reasons change only on events |
| A4 T3/V1: the shard's live ground and the client's differ; a lazy re-solve drifts | §4.3: live when every session in reach holds it; §4.6: a new fine law is one world move |
| A5 F1/F2: versions, patches smaller than tiles, no artifact digest in the key | §4.2: the input digest in the key; §4.6: patches pin digests; Q8 |
| A6 B55: two slope limits break the band's bound | §3.2: one `S_c` at both levels; §4.5: P-BAND; Q7 |
| A7 SL6 and the owner's limit: §0.1 said the gateway stays | §0.1 corrected (five crates change, the mechanics do not); Q2 prices the SL6 default |
| A8 knobs: "0.1 %", "3 rounds", the creep clamp, unnamed constants | §3.3 convergence tests; §7 every constant named |
| A9 plain noise under 8 m | §2 and Q1: no noise; the smooth curve until L3 |
| A10 the jittered lattice keeps its axes | §3.1: three point sets benched; the owner chooses |
| A11 other bodies; division by `Q = 0` | §3.6: a law decides where each level applies |
| A minors (F7, V1, HR5, HR4, SL7 liveness, B76, F6) | §4.5, §6, §1, §4.1, §6 |
| B-B1 sums move with the core count | §3.7: fixed blocks of 4 096, `par::blocks` |
| B-B2 the crossfade ghosts valleys; edges drain to the rim | §3.4: a solved buffer, only the coarser rivers as outlets, the minimum-error cut; G-SEAM reads pits, spread, valley ends |
| B-B3 the targets disagree; `Gf::powf` is ~5× | §5.1–5.3: one target per core; the fine power; P-COST under `Gf` |
| B-B4 loops with no proven end | §3.3: the hold as one linear solve; a cycle detector; N against 2N |
| B-I1 still land costs most; airless moon | §3.6 |
| B-I2 the ship's bytes and the frame | §4.4: 32 KB parts paced by bytes; the coder's worst case |
| B-I3 a cliff where held tiles end | the waiting rule: a chunk never mixes a held and an unheld level |
| B-I4 L2 needs L1 data not stored | §4.2: the head's crossings and `V` |
| B-I5 cube corners | §3.1: Delaunay on the sphere |
| B-I6 versions and corruption | §4.2: the input digest, verified on every read |
| B-I7 the wide band's cost | Q7, P-BAND |
| B-I8 the queue under load; camps; path B's cache | §4.1 round-robin and a capacity reading; §4.3 and Q6; §4.4 a byte-bounded cache |
| B-I9 the hold makes pits | §3.3: breach and hold again; G-PIT |
| B minors (raster cost, tolerance, reads per column, GPU, durability, neighbour reach) | §5.2, §6, §4.2, §3.1 |
| C-B1 the fallback makes the shape a function of state | the waiting rule (§0, §0.1, §2, §4.5) |
| C-B2 no rebuild path, no ramp | removed by the waiting rule: a fine-level chunk is built first when its tiles are live and ramps as any late chunk |
| C-B3 every ladder bound reads the octave table | §4.5: the fine step bound per rung pair; P-SWITCH; Q7 |
| C-I1 no octave under 48.8 m | §2, Q1 |
| C-I2 the means move when L1 lands | the waiting rule: rungs 7–9 never draw the octaves on a body whose L1 applies |
| C-I3 a hull's pilot gets no L2 | §4.1 |
| C-I4 the tile does not fit the lane | §4.4 |
| C-I5 the blend needs halo bytes | §4.4: tile + half the buffer (576²) |
| C-I6 a second column path | §4.5: one fine read for every reader |
| C-I7 the band's runtime cost | Q7, P-BAND |
| C-I8 the identity check; version names | §4.5: a golden fine literal; `GENERATOR_VERSION` for the reader, `FINE_VERSION` in the key |
| C minors (store crate, one variant, `PROTO_MINOR`, rung cells, P-READ per rung, decode alone) | §4.7, §4.4, §2, §6 |

## 11. Revision 4's changes (B82–B84 and the bench B0, 2026-10-11)

| Where | Revision 3 | Revision 4 | Source |
|---|---|---|---|
| §3.1 the points | three sets to bench | the hashed Poisson-disk (Matérn II, `r = cell/√π`) | B83 item 1; B0 step 4 |
| §3.2 `S_c` | 1.25 at both levels (A6) | tan 35° at L1, 1.25 at L2 | B82 item 1 (52 % over 45° at 1.25) |
| §3.2 L2's law | the relaxation (FastScape splitting) | the steady form (river or creep, the lower; MFD area) | B0: 15 min → 7 s a tile, the same look |
| §3.3 L1's hold | one linear solve on a fixed tree | none: the law alone from a coarse start and three rounds | B82 item 3, B84 (the linear hold refuted; the additive hold's hollows) |
| §3.3 G-MEAN | exactly zero | every swap under a pixel at its switch | B84 item 2 |
| §3.3 L2's hold | Tobler's correction per L1 cell, then breach | L1's channels as outlets + the slope hold per L1 Voronoi cell | B83 item 5 (974 → 48 m at the hand-over; seamless by construction) |
| §3.4 the buffers | one node / 64 cells | two nodes / 2 km; the cut free over half; an 8-cell blend; exits at the link's lower mean | B0 steps 2–2b |
| §2 L1's rungs | 7–9 | 7–12 (rung 13 the row) | B83 item 2 |
| §4.4 the near tile | 576² | 640² (half the measured buffer) | B0 step 4 |
| §4.4 the coder | LOCO-I + varint bytes | LOCO-I + the adaptive Rice code | B83 item 4 (P-SIZE passes) |
| §4.5 the band | the law's 14.5 km or measured | the law's ±4.1 km at tan 35° | B83 item 3 |
| §4.5 the step bound | may push rung switches out | no switch moves at L1 | B0 step 4 (P-SWITCH) |
| §5 the speed | targets only | measured: L1 2.3–6.2 s, L2 13.5 s, ~0.9 tiles/s on 4 cores; the fine power +7 % | B0 step 3 |
| §7 the solver's numbers | — | named; the rounds' clamp replaced in B1 | B23, B38 |
| §1 own code | — | the game's point graph our own; the bench's library stays a bench dev-dependency | the owner's rule on new libraries |
| §9 B1 | the read and the tile book | the whole fine part, in order (a)–(g) | this revision |
| §3.4 the corners (B85) | each row of tiles along its column cuts, then the row cut | each edge's cut in two halves, corner point → midpoint; a tile depends on its eight neighbours only | B85 items 2–3 (measured: the same slope breaks as the bench's stitch) |
| §4.4 what ships (B85) | the raw 640² + the cut paths, the client stitches | the finished 512² + its mean levels, the client only reads | B85 item 1 (303 KB against 463 KB on the belt) |

**Kept unchanged (checked against the rules):** SL3 (the planet's shard solves its own tiles), SL7 and SL9 (§4.1's reasons
and events), SL2 (path B never a reason), SL6 (the one approved bulk part, B81 Q3), SL10 and F7 (the read integer, the solve
under `Gf`, the stored tile integers), B64 (fixed-block sums, one tile one core), F6 (the solve's share), F9 and B18 (the
waiting rule), T3 (a tile reads no tick), F1/F2 as amended (B81 Q8), HR3/HR4 (one worker, any realm with ground, one wire
variant with a level), HR5 (Tier-A coverage of every new line), B74 (every test run on the owner's word).
