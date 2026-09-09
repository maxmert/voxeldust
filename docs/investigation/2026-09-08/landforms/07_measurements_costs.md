# 07 — Landforms: the cost model, and the measurements that must run before the build

**Date:** 2026-09-08. **Revision 3** (after the second pair of refutations, A and B). **Domain:** costs
and the measurement plan for the landform work (domain 07).
**Status:** an investigation report for the owner. It measures, it estimates, and it decides nothing.
**Binding law read first:** CLAUDE.md (HR1–HR6, SL1–SL10), `owner_decisions_2026-09-07_voxels.md`
(V1 SL10, V2.1–V2.9, V4, V6, V8, V9, V10, V11, V12), `owner_decisions_2026-08-27_seed_and_secrecy.md`,
`owner_decisions_2026-09-02_reach.md`, and the standing rule **NEVER ASSUME — MEASURE IT**.

Every number below carries one of three marks:

- **MEASURED** — a run that could have failed. The command is named.
- **DERIVED** — arithmetic on measured numbers. The arithmetic is shown.
- **ESTIMATED** — an op count times a unit cost. The op count is shown, and the bench that replaces
  the estimate is named in §13.

No number in this document is a result until §13's table has a green row beside it.

**What revision 3 changed. The recommendation itself moved.** Two refuters read revision 2. Nine of
their findings moved the design, not only the words:

1. **The cost document priced a design nobody proposed** (A-B4). Domain 03 owns the erosion design, and
   its revision 2 recommends **ONE GLOBAL macro lattice of 640 nodes per face edge, 8 224 m per node,
   40 passes, solved ONCE EVER per body and cached** — not a pyramid of on-demand patches. Revision 3
   prices THAT design first, and **recommends it** (§5.5 shape B, §16 D-C1). My pyramid becomes the
   REFINEMENT domain 03 defers, priced here so the owner can buy it later.
2. **The composition rule was never stated** (A-B1). Domain 03 states it: the eroded field **REPLACES**
   the coarse octaves; it does not add to them. Revision 2 priced an addition, which would have doubled
   the home planet's long-wave relief. §7.1 is re-derived (§5.7).
3. **An implicit sweep carries the base-level signal down the whole basin in ONE pass**, so revision 2's
   "the iteration count grows with the grid's diameter" law was written for the wrong solver (A-D4,
   B-B2). The pass count is the landscape's **erosional age**, not a convergence radius (§5.3). The same
   fact is the deepest argument AGAINST patch refinement: flow is a global structure, so no halo is ever
   wide enough (§5.6).
4. **The document never priced THE VISTA** (B-B4), which is the owner's own question. New §7.5:
   **about 4 200 chunks, 16.3 s of one-core work, 1.04 GB of mesh** for a 50 km vista.
5. **The drawable floor is 1.1506 mrad, not 1.0908** (A-D1). Every staging distance is re-derived.
6. **The build record named `f32`, which the fence bans by name** (B-B1). Every accumulator is `Gf` or
   an integer, and rule **E10** names the control (§12).
7. **There is no deposition and no ice** (B-B3), and **isostasy was taught and never implemented**
   (A-W1, B-W2). All three are now passes with prices (§4.1), and the two landforms this domain still
   cannot make are named at the top.
8. **The extraction curve mixed two measured rows** (A-D2). The corrected model is
   **0.97 ms + 61 to 104 ns per vertex**, and under it the densest measured chunk at twice the vertices
   is **over** the 8 ms budget (§3, §7.1).
9. **Mean-pinning conserves mass, so a refinement invents a raised shoulder beside every valley**
   (A-W1, B-D8). It is stated, bounded, measured (M-L21) and put to the owner (D-C17).

The answer to every finding is §20.

---

## 1. The recommendation, in short

**Erode each body ONCE, on ONE global lattice of about 8 200 m per node, and keep the answer. Do not
erode per player, per patch or per visit.**

The cost of eroding a lattice of `M` nodes per cube-face edge is

```
   cost  =  6 · M²  nodes  ×  the PASS SCHEDULE  ≈  6 · M² × 4.93 µs
```

on the home planet's own lattice (`M = 640`, 8 224 m per node), which is **12.1 s on one core and
128 MB of solve state, ONCE EVER per body** (DERIVED, §5.3; domain 03's independent op model says
12–20 s and 125 MB, and the two agree). The answer kept is **9.83 MB**.

The cost grows as the SQUARE of the resolution, so the same lattice at 257 m per node — the resolution a
walkable valley needs — is **3.6 hours and 100 GB** (DERIVED, §5.5 shape A). That one line kills a fine
global solve for ever, and it is why the fine detail must be closed-form (domain 03) or a refinement
(§5.6).

**The per-chunk budget survives, and the headroom is thin.** A rung-0 surface chunk costs **3.30 ms**
today (MEASURED, `just terrain-cost`) against the owner's **8 ms** budget
(`crates/bins/examples/terrain_cost.rs:41`). The landform work adds **0.45 ms** of column-pass
arithmetic and makes the surface rougher, which makes the EXTRACTION dearer:

```
   a surface chunk           3.30 -> 4.34 ms   at twice the vertices           DERIVED + ESTIMATED
   the cave-dense SEAM chunk 6.11 -> 7.42 ms   at twice the vertices
   the DENSEST measured chunk 5.22 -> 8.07 ms  at twice the vertices  OVER the 8 ms budget
```

**The vista is the cost nobody had priced, and it is a hundred times the chunk budget.** A 50 km vista
on the home planet holds about **4 200 chunks**: **16.3 s of one-core work, 1.94 s on the client's
fourteen threads, and 1.04 GB of mesh** (DERIVED, §7.5). The landform work adds 18 % to that. The macro
artifact — the thing this whole document is about — is **9.83 MB, which is under one per cent of the
mesh the same picture holds.**

```
   WHAT A 50 km VISTA ON THE HOME PLANET COSTS          DERIVED, §7.5
   ------------------------------------------------------------------
   the macro artifact (once per body, cached)   9.83 MB   |
   the mesh resident for the picture             1.04 GB  |##############################
   worker time to fill it, one core             16.3 s
   worker time to fill it, 14 threads (8.4x)     1.94 s
```

**Three honest limits, stated at the top because the owner asked about the picture.**

- **This domain does not fix the band below the macro node.** I replayed the crate's own octave law on
  the home planet (§4.2): the five octaves under 800 m carry **15.15 m of amplitude in total**. A macro
  lattice that stops at 8 224 m puts big, correct valleys on a surface that is still glass-smooth at the
  scale of a cliff. The cliffs, the rock pillars and the mesa rims of the reference picture come from
  the roughness, from the closed-form terms and from an overhang mechanism, which are domains 01, 02
  and 03. §4.2 prices the alternatives, so this document's kill rows have somewhere to land.
- **This domain delivers no glacial forms and no flat alluvial valley floor unless two passes are
  bought.** The reference picture's headline is a snow-capped ridge over a U-shaped valley, and its
  middle ground is a flat green floor with a river on it. A stream-power law alone makes neither
  (refutation B B3, right). §4.1 prices a deposition pass at **+30 ns per node per sweep** and an ice
  pass at **+30 ns per node over the solve**; D-C21 puts them to the owner.
- **Weather is not climate, and only climate is seed-derived.** §10.2 states what weather is, why the
  client may not derive it, what it costs to cross (**1 936 bytes per player per update** over a 50 km
  window, thirteen times revision 2's figure) and that it is an SL6 ask the owner must answer.

**The single measurement that decides everything else** is M-L1: the pass schedule's true cost per node.
Every solve figure here is that schedule times a node count. At half the estimate the home planet solves
in 6 s; at twice it, in 24 s, and the Earth-sized body's lattice coarsens by one divisor step. **Run
M-L1 first, on THE world's home planet, before any landform code is designed further.**

---

## 2. The words, explained once

Each term is the industry's own. Each example is in the game's words.

| Term | Plain words | The example |
|---|---|---|
| **Tectonic uplift** | The slow rise of the crust where two plates push together. It ADDS height. | The home planet's seed draws its plates. Where two plates converge, the macro solve raises the ground by about a hundred metres each pass, and after forty passes a range stands there, which the rivers then cut back down. |
| **Isostasy** | The crust floats on the mantle. Take mass off the top and the crust rises. | The rivers cut a valley on the home planet. The block is lighter, so the block rises, so the peaks beside the valley stand higher than the uplift alone put them. Domain 03 runs it every five passes; §4.1 prices it. |
| **Denudation** | The net REMOVAL of material from a surface. It is what erosion does over a whole landscape. | After the home planet's solve, the mean height of a mature basin is lower than it was. If nothing is removed, nothing eroded. |
| **Hydraulic erosion** | Water removes rock. | The rain that falls on the home planet's highland runs downhill and takes the ground with it. |
| **Thermal erosion (hillslope diffusion)** | Rock falls down a slope that is too steep. It smooths the land. | A cliff on the home planet's mesa keeps its face, and the scree below it makes the slope the pilot can walk. |
| **Stream power law** | The rule that says how fast a river cuts: `erosion = K · A^m · S^n`, where `A` is the land AREA that drains through this point and `S` is the slope. | A macro node with a big catchment and a steep slope cuts fast, so a gorge forms. |
| **Detachment-limited** | A law where the rock's resistance sets the rate. It only REMOVES. | Stream power alone. It cuts the home planet's gorge and lays no floor in it. |
| **Transport-limited (deposition)** | A law where the water's carrying capacity sets the rate. Material the water cannot carry is DROPPED. | The river leaves the home planet's range, the slope falls, the water drops its load, and the pilot walks on a flat valley floor with a field on it. |
| **Alluvium** | The material a river drops. A flat valley floor is made of it. | The reference picture's green middle ground. |
| **Glacial forms** | What ice leaves: a U-shaped valley, a bowl (a cirque) at its head, a knife ridge (an arête) between two bowls. | The reference picture's snow-capped ridge. Stream power makes a V, not a U, so a geologist sees the error in one glance. |
| **Flow accumulation / drainage area** | For each node, how much land upstream drains through it, in square metres. | A node high on the ridge drains only itself. The node at the river mouth drains ten thousand nodes, so the river there is wide. |
| **Drainage basin** | Every node that drains to one outlet. A tree. | The valley the pilot lands in is one basin. The basin next door drains to the other side of the range, and the divide between them is the ridge she walks over. |
| **Drainage density** | Channel length per unit of area. It is the fingerprint of a landscape's climate and rock. | One channel every 40 km on the home planet's highland reads as a desert. One every 2 km reads as a wet, dissected range. |
| **Base level** | The lowest height a river can cut to. Usually the sea. | The home planet's sea sits 5 297 m under the ladder radius (MEASURED, slice 5 results), so no river on it cuts below that. |
| **Priority flood / depression filling** | A pit with no outlet stops the flow. The pass fills the pit, or it carves an outlet, so every node drains somewhere. It uses a HEAP: a queue that always hands back its lowest node. | Without it a crater on the home planet swallows its river, and the vista shows a channel that stops in mid-air. |
| **Basin labelling** | The pass that gives every node the name of the outlet it drains to. It must run before any per-basin work. | Every node of the pilot's valley carries the name of the river mouth at the coast, so the whole valley is one job. |
| **Pointer jumping** | How a label reaches a whole tree fast: every node replaces its parent with its grandparent, and after `log2` rounds every node points at the root. | Round one, the pilot's node points two nodes downstream. Round two, four. After eleven rounds it points at the river mouth. |
| **Topological order (the stack)** | An order in which every node comes after everything that drains into it. | The solve adds the areas in that order, so a node's own area is complete before anything reads it. |
| **Donor** | A node that drains INTO this one. The opposite of a receiver. | The pilot's node has three donors: the two gullies above her and the snowfield. |
| **Implicit sweep** | A solver that writes each node's new height in terms of its RECEIVER's new height, walking the stack downstream to upstream. One sweep carries a change at the river mouth to every headwater. | The sea drops. In ONE pass every valley on the home planet knows about it. An explicit solver would need one pass per node of distance. |
| **Stencil** | The small fixed set of neighbours a pass reads. | The hillslope pass reads the four nodes around it, and nothing else. |
| **Erosional age (the pass count)** | How MATURE a landscape is, set by the number of passes. It is not a convergence count. | Forty passes on the home planet make a landscape whose trunk valleys are cut and whose headwaters are still young. Eighty would wear it flat. |
| **Hypsometry / the hypsometric curve** | How much of a body's surface stands at each height. A signature of a planet. | Earth has two humps: the continents and the abyssal plains. The home planet has almost no sea, so it can never grow the second hump (§13, M-L13). |
| **Hypsometric integral** | One number from that curve: the mean height as a fraction of the range. A young land is near 0.6; a worn land is near 0.3. | If forty passes move the home planet's integral from 0.50 to 0.42, the erosion did visible work. |
| **Orographic lift** | Air rises over a range, it cools, and it drops its rain on the near side. The far side is dry. | The pilot walks over the home planet's range: forest on the wind side, desert behind it. That is the rain shadow. |
| **Advection** | Carrying a quantity along with a flow. The wind ADVECTS moisture across the map. | The moisture pass starts at the home planet's coast and walks inland along the wind, dropping rain where the ground rises. |
| **Hadley cell** | The big loop of air between the equator and about thirty degrees. It makes wet equators and dry deserts at thirty degrees. | The home planet's equatorial belt is wet, and its deserts sit in two bands north and south of it — not scattered by a noise. |
| **Coriolis** | A spinning body bends its winds. | The home planet's spin decides which way the wind crosses the range, so it decides which SIDE of the range is the desert. |
| **Obliquity** | The tilt of a body's spin axis against its orbit. It makes seasons, and it sets the size of the ice caps. | Today `POLE_AXIS` is `+Z`, the orbit's own axis (`crates/terrain/src/height.rs:35`), so the home planet has no tilt and no seasons (§10.4). |
| **Insolation** | How much starlight a patch of ground receives. | The home planet's poles receive light at a glancing angle, so they are cold. The star's luminosity, not a constant, sets the whole scale. |
| **Albedo** | How much light a surface throws back. Ice throws back a lot. | Snow on the home planet's peaks keeps them cold, which keeps the snow. |
| **Lapse rate** | How fast the air cools with height. It is the surface gravity divided by the air's heat capacity. | The pilot climbs 3 000 m up the mesa, and the biome turns from grassland to tundra with no change of latitude. |
| **Climate** | The long-run AVERAGE state of the air over a place. It does not change with the tick. | The home planet's highland is cold and dry, this year and every year. Both hosts derive it from the seed. |
| **Season** | A SLOW, planet-wide change with the tick, from the obliquity and the orbit. | The snow line on the home planet's ridge walks down in winter and back up in summer. It is neither climate nor weather (§10.4). |
| **Weather** | The state of the air at THIS tick: cloud, rain, a gust. It changes with time and with live state. | It is raining on the pilot's visor right now. The planet's shard owns that fact, and it must SHIP it (§10.2). |
| **Aerial perspective** | Distant things turn pale and blue, because air scatters light. It is the eye's main distance cue. | The reference picture's ridge 40 km away is blue. The home planet has no atmosphere in the code today, so it cannot do this yet (§10.3). |
| **Nyquist** | A grid can only carry a wave at least twice its own spacing. A shorter wave becomes a lie. | The 8 224 m lattice may hold the 25 km octave. It may NOT hold the 12.5 km one, which would read as a false pattern. |
| **Bicubic / bilinear** | Two ways to read a value between grid points. Bilinear uses 4 taps and its SLOPE jumps at every cell edge; bicubic uses 16 taps and is smooth in slope. | Bilinear prints an 8 km grid of creases across the whole vista, which is a seam. Bicubic does not. |
| **Minimax polynomial** | A polynomial chosen so its LARGEST error against the true function is as small as possible. It is how a fenced crate computes a sine. | The insolation needs a cosine. The fence forbids one, so the crate carries a committed polynomial with a stated maximum error (§10.1, E11). |
| **Mip pyramid** | One field stored at many resolutions, each a fixed factor coarser than the one below. | The artifact's own coarse pyramid, 19 KB at the top, is what a planet seen from a window is drawn from. |
| **Refinement (multigrid)** | Solving a coarse problem first, then adding detail INSIDE each coarse cell without moving the coarse answer. | The deferred slice: the 8 224 m solve says the pilot's basin averages 1 200 m; a 1 285 m patch puts a valley and a shoulder inside it. |
| **Mean-pinning** | The rule that makes refinement lawful: after every pass a child block's average is forced back to its parent node's value. | The pilot flies up. Her valley smooths into the hill that always stood there. **It also conserves mass, which invents a raised shoulder** (§5.6, D-C17). |
| **Partition of unity** | A set of smooth weights that add to one everywhere, used to blend overlapping patches with no seam. | Two patches overlap over the pilot's landing site. Each contributes a weight; the weights add to one; nobody can point at the join. |
| **The macro artifact** | Our name for the per-body field the solve produces: one height, one receiver, one discharge per node, plus a lake table and a coarse pyramid. | It is what rungs 0 to 11 all read, and what the coarse rungs draw almost alone. |
| **A patch** | A square piece of one refinement level, computed on demand from its own ADDRESS, then cached. Deferred; priced in §5.6. | The pilot lands. The patch around her landing site is computed once, cached, and every chunk of her vista reads it. |
| **The halo** | The extra ring of nodes a patch computes so that its inside does not know it has an edge. **It is not the extractor's halo**: `HALO` in `crates/terrain/src/lattice.rs:48` is ONE cell, because a stencil reaches one cell; a patch's is forty, because an iterative field reaches further. | The patch computes 400 × 400 nodes and keeps the middle 320 × 320. |
| **ulp** | "Unit in the last place": the smallest step a floating-point number can take at its own size. Two hosts that differ by one ulp differ by the least amount that exists. | The float fence exists so that no ulp ever differs between the server's chunk and the client's chunk. |
| **L1 cache / last-level cache** | Small fast memories in front of the main memory. A read that misses every cache costs a trip to main memory, about 80 ns. | A chunk gathers its 4 × 4 macro taps once; after that they sit in registers and all 4 096 columns read them for free. |
| **Slope predictor + entropy coder** | Compression: predict each node from its neighbours, then store the small surprise with short codes for common values. | Shipping the artifact (§8.1) would use it: 9.83 MB raw, an ESTIMATED 3 to 5 MB coded. |

---

## 3. What the code costs today (the measured ground)

Every row is MEASURED by `just terrain-cost`
(`cargo run --release -p vd-bins --example terrain_cost`), release, on this Mac (Apple M4 Pro,
14 logical cores, 10 of them performance cores — MEASURED, `sysctl hw.ncpu hw.perflevel0.logicalcpu`),
on THE world's home planet: seed 7 701 581 858 760 374 086, look radius 3 351 154 m, ladder radius
3 350 759 m, 12 rungs, 14 octaves, relief 14 304 m.

| What | Cost | Source |
|---|---|---|
| The column pass, rung 0, 62 x 62 columns, 14 octaves | **710 µs** | slice 5 results table |
| The column pass, rung 11, 3 octaves | **266 µs** | slice 5 results table |
| The cell pass, rung 0, 62³ cells | **716 µs** | slice 6 results ("from 578 µs to 716 µs with the site plumbing") |
| The sample box, rung 0 (chunk plus halo: 64³ cells, 64² columns) | **1.98 ms** | slice 6 results table |
| **A rung-0 surface chunk (3, 5): box 1.98 + extraction 1.32, 5 650 vertices** | **3.30 ms** | slice 6 results, "THE BUDGET, decided" |
| A rung-0 seam chunk (84 892, 7): box 1.89 + extraction 1.34, 6 057 vertices | 3.23 ms | slice 6 results table |
| A rung-0 corner chunk: box 1.66 + extraction 1.22, 3 359 vertices | 2.88 ms | same |
| **A cave-dense chunk (rung 2, chunk (3, 5)): box 2.83 + extraction 2.39, 23 084 vertices** | **5.22 ms** | same |
| **A cave-dense SEAM chunk (rung 2, chunk (21 223, 7)): box 4.38 + extraction 1.73, 8 234 vertices** | **6.11 ms** | same |
| A rung-11 chunk: box 1.05 + extraction 1.28, 4 055 vertices | **2.33 ms** | slice 6 results |
| The synthetic checkerboard bound, 250 046 vertices | **26 ms** | same |
| The halo's own cost | **+35 %** over the bare chunk | slice 6 results |
| The snap error on surface vertices | p50 0.0039 cells, max 0.0090 | slice 6 results |
| **Client geometry, rung 0, one thread** (box + extraction + positions + normals) | **4.18 ms per chunk** (239 chunks/s) | slice 7 M7-2 |
| Client geometry, fourteen threads (independent chunk jobs) | 2 004 chunks/s, **8.4x** | slice 7 M7-2 |
| Per rung-0 chunk handed to the engine | 8 577 vertices, 16 615 triangles, **405 228 bytes** | slice 7 M7-3 |
| One radial column at rung 0 | 470 chunks, 463 skipped, **417 ms** | slice 5 results |
| The boot self-check (8 chunks) | **13.5 ms** | slice 5 results |
| **The budget** | **8 ms of worker time per chunk** | owner 2026-09-08; `crates/bins/examples/terrain_cost.rs:41` (`EXTRACT_BUDGET_US = 8_000.0`) |

**Revision 2 built one row out of two, and every headroom figure rested on it** (refutation A D2,
right). Slice 6 has TWO cave-dense rows: the chunk (3, 5) with 23 084 vertices and 2.39 ms of
extraction, and the SEAM chunk (21 223, 7) with 8 234 vertices and 1.73 ms. Revision 2 took the time
from one and the vertex count from the other and printed "75 ns per vertex", which no measurement
supports. The table above keeps the two rows apart.

**THE EXTRACTION MODEL, re-derived from the measured rows.** A straight line through the rung-0 surface
chunk and the cave-dense chunk (3, 5):

```
   extraction  =  0.973 ms  +  61.4 ns per vertex          DERIVED from two MEASURED rows

   vertices    measured    the model    the measured per-vertex figure
   ---------   ---------   ----------   ------------------------------
     3 359      1.22 ms     1.18 ms       363 ns
     4 055      1.28 ms     1.22 ms       316 ns
     5 650      1.32 ms     1.32 ms       234 ns      <- the fit's first point
     6 057      1.34 ms     1.34 ms       221 ns
     8 234      1.73 ms     1.48 ms       210 ns      <- the model UNDER-predicts a seam chunk
    23 084      2.39 ms     2.39 ms       104 ns      <- the fit's second point
   250 046     26.00 ms    16.32 ms       104 ns      <- the model UNDER-predicts by 60 %
```

**So the model is a LOWER bound at both extremes**, and §7.1 therefore prices the extraction growth
TWICE: once at the fit's marginal cost of **61 ns per vertex**, and once at the measured high-count cost
of **104 ns per vertex**. Revision 2's "it settles near 100 ns" was right about the value and wrong
about which rows produced it.

**Two facts of the ladder carry every cost below** (`crates/seed/src/ladder.rs:21`, `:23`):

- `CHUNK_EDGE = 62` cells per chunk edge, at every rung.
- `TOP_RUNG_CHUNKS = 64`: the top rung is the FIRST rung with at most 64 chunks along a face edge. So
  **the top rung's cell grid never exceeds 64 x 62 = 3 968 cells per face edge, for any body in THE
  world.** The worst case is `6 x 3 968² = 94 470 144` cells. The home planet holds 39 629 400.

**THE DRAWABLE FLOOR, from the code and not from a memory** (refutation A D1, right).
`crates/core/src/geometry.rs:1164-1173` computes it as `2 · tan(fov/2) / rows`:

```
   the code:      2 * tan(22.5 deg) / 720  =  0.8284271 / 720  =  0.00115059 rad = 1.1506 mrad
   revision 2:        (pi/4)         / 720  =  0.7853982 / 720  =  0.00109083 rad
   MEASURED check: a one-metre extent is drawable to 1 / 0.00115059 = 869 m, which is the
   "~870 m" the reach ruling itself states. Revision 2's figure gives 917 m and is 5.2 % wrong.
```

Every staging distance in §8.3 is re-derived from **1.1506 mrad**.

---

## 4. The unit costs this document uses

### 4.1 The pass schedule, not one number

**Revision 2 priced the erosion as one number — "160 ns per cell per iteration" — and both refuters
showed that no such number exists** (A-D9, B-W6). A heap's cost grows with the grid; a labelling pass's
cost grows with the basin's length; a flood does not run every pass. Revision 3 prices a SCHEDULE.

| Unit | Value | How it is obtained |
|---|---|---|
| One gradient-noise evaluation (`noise3`: hash, fade, blend) | **at most 11.5 ns** | DERIVED from MEASURED, and an UPPER bound. The rung-0 column pass is 710 µs for 62² = 3 844 columns, and it also runs `site_of`, `site_dir` and a min/max fold per column, and `biome_at` returns early. Dividing 710 µs by 3 844 x 16 charges the noise for work the noise did not do, so **the true per-evaluation cost is LOWER, and this figure is safe as a COST and unsafe as a SAVING.** §7.1 books every saving it implies at ZERO. |
| One column of the base height at rung 11 (3 octaves plus the biome noises) | **69 ns** | DERIVED: 266 µs / 3 844 columns. |
| One fenced arithmetic operation (`Gf` add, multiply, divide, `sqrt`) | **1 ns**, conservative | ESTIMATED. A dependent double-precision chain on this chip runs at about 3 to 4 cycles per operation near 4 GHz. 1 ns is the pessimistic end, used deliberately. |
| One irregular memory read that misses the last-level cache | **80 ns** | ESTIMATED, the usual figure for a main-memory round trip on this class of chip. UNMEASURED here; M-L1 measures the composite instead. |
| One irregular read that HITS the last-level cache | **12 ns** | ESTIMATED. |
| One heap compare | **1.5 ns** | ESTIMATED: a compare, a branch and a swap on an 8-byte key that is usually resident. |

**THE SCHEDULE**, per node, following domain 03's own loop
(`03_erosion_rivers.md:812-826`: `PASSES = 40`, `FLOOD_EVERY = 10`, `ISOSTASY_EVERY = 5`,
`TALUS_PASSES = 8`):

```
   EVERY SWEEP (40 of them)                                         ns per node
      receivers and slope, with the METRIC (§5.4)                       18
      donor counting                                                     8
      the stack build                                                   15
      the flow accumulation, in stack order                             12
      the IMPLICIT incision (one sqrt, eight operations)                10
      the hillslope diffusion (a four-neighbour stencil)                 8
      the DEPOSITION term (a flux down the stack, a settling term)      30   <- NEW, refutation B B3
      the quantisation to millimetres                                    2
                                                                     -----
                                                                       103 ns per node per sweep

   EVERY TENTH SWEEP (4 of them) — the two passes revision 1 forgot and revision 2 under-priced
      PRIORITY FLOOD: 2·log2(n) compares at 1.5 ns, plus 25 ns of irregular traffic
      BASIN LABELLING: ceil(log2(basin length)) rounds of pointer jumping at 6 ns
                                            n =    24 576   ->  111 ns
                                            n =   160 000   ->  137 ns
                                            n = 2 457 600   ->  155 ns
                                            n = 2.5 x 10^9  ->  215 ns

   EVERY FIFTH SWEEP (8 of them)
      the isostatic rebound on the artifact's own pyramid                12   <- NEW, refutation A W1

   ONCE, AFTER THE SWEEPS
      the talus relaxation, 8 passes at 8 ns                             64
      the ice pass (mask, depth proxy, over-deepening) on the
        cells above the snow line, about 20 passes on 15 % of them       30   <- NEW, refutation B B3
```

**The whole solve, per node, on the home planet's lattice** (`n = 2 457 600`):

```
   40 x 103  +  4 x 155  +  8 x 12  +  64  +  30  =  4 929 ns per node       DERIVED
```

**Two independent models agree.** Domain 03 built its own op model and printed **12 to 20 s** for the
same lattice (`03_erosion_rivers.md:1450-1458`); mine prints **12.1 s**. Two models are not a
measurement, and M-L1 is still the gate — but they do not disagree, which is worth stating.

**Where the estimate is weakest, and it matters.** The 80 ns memory figure assumes an access that misses
every cache. The home planet's solve state is 128 MB and streams from main memory every sweep, so the
true figure may be higher there; a 6 x 64² refinement level holds 1 MB and fits in cache, so it may be
lower. **The estimate is optimistic for the big grids and pessimistic for the small ones, which is the
wrong way round for safety.** M-L1 measures at three sizes, and §5.3's table carries the size-dependent
flood cost rather than one number.

### 4.2 The band table — what owns which scale, and what this domain does NOT fix

The reference picture holds landforms from 50 km down to a hand's width. Here is who owns each band on
the home planet, and what each band carries TODAY.

```
   400 km ---- 82 km ---- 8.2 km --- 1.3 km --- 257 m --- 49 m ---- 1 m
     |           |           |          |          |         |        |
     |<-- THE MACRO SOLVE, this domain -|          |         |        |
     |           |           |<-- the DEFERRED refinement -->|        |
     |                                                       |        |
     |<----------------- octaves 1-14 ---------------------->|        |
                                                   |         |        |
                                                   |<-- THE BROKEN BAND -->|
                                                     (domains 01, 02 and 03's closed-form terms)
```

| Band | Owner | What it carries today (MEASURED, by replaying `crates/terrain/src/body.rs`'s octave draw in `python3`) |
|---|---|---|
| 400 km – 25 km | octaves 1–5 | 7 606 m, 3 562 m, 1 668 m, 781 m, 366 m of amplitude — the continents |
| **400 km – 8.2 km** | **THE MACRO SOLVE, this domain** | today: nothing. The ranges, the basins, the trunk valleys and the rivers this domain adds, REPLACING octaves 1–5 (§5.7) |
| 12.5 km – 1.6 km | octaves 6–9 | 171 m, 80 m, 38 m, 18 m |
| **8.2 km – 257 m** | **the DEFERRED refinement (§5.6), or domain 03's closed-form terms** | the side valleys and the ridge every 1–3 km |
| **800 m – 49 m** | **octaves 10–14** | **8.24 m, 3.86 m, 1.81 m, 0.85 m, 0.40 m — 15.15 m in TOTAL** |
| under 49 m | nothing | the octave loop's bound is `while wave_m > 30`, so the finest octave the home planet draws is 48.8 m, and 49 m to 30 m is empty too (refutation A N7, right) |

**The measured fact the owner's worry rests on.** Across the whole band from 800 m down to 49 m the
home planet's recipe carries **15.15 metres of relief**. A macro solve that adds three hundred metres of
valley at 8 224 m spacing sits on a surface that is glass-smooth at the scale of a cliff. **The picture
does not get a cliff; it gets a bigger, smoother hill.** Domain 01 measures the same thing from the
other side (its maximum slope of 36°, its spectral exponent β = 3.19 against Earth's ≈ 2); my replay of
the amplitudes agrees with its amplitudes exactly.

**So this domain cannot be the whole answer, and the alternative must have a price.** Here it is:

| The alternative | Cost per rung-0 sample box (4 096 columns) | Mark |
|---|---|---|
| One extra octave | 4 096 x 11.5 ns = **47 µs = 0.047 ms** | DERIVED from §4.1's upper bound |
| Four extra octaves (the table reaching 3 m instead of 49 m) | **0.19 ms** | DERIVED |
| A different roughness (no new octave) | **0 ms** — it re-weights amplitudes the pass already sums | DERIVED |
| Both, plus the extraction growth they cause | 0.19 ms + §7.1's extraction row | ESTIMATED |

**Two warnings, so nobody reads that alternative as free.**

1. A roughness change with the crate's present SUM normalisation is violent. I replayed it: at
   `k_rough = 0.707` the 781 m octave's amplitude jumps from 8.24 m to **186 m**, and the summed
   per-octave slope reaches a tangent of 20.3 — a vertical planet. **The normalisation must change
   with the roughness**, and that arithmetic is domain 01's and 02's, not this document's.
2. The short-wave floor and the roughness both live in the recipe, so changing either **opens a world
   epoch** (`crates/terrain/src/tag.rs:19`). So does the macro solve. See D-C14.

**In the game's words.** The pilot stands in the home planet's highland. This domain gives her the
valley she is standing in, the river running through it, and the desert on the far side of the range.
It does NOT give her the cliff at her elbow. That comes from the octaves, from domain 03's closed-form
terms and from the cell pass, and this document's job is to say so plainly and to price it.

### 4.3 Threading — and why every ceiling in this document is a ONE-CORE ceiling

`crates/terrain/Cargo.toml` has exactly one dependency: `vd-seed`. There is no `rayon`, no thread pool
and no `std::thread` on the generator path. The crate ships as `crate-type = ["lib", "staticlib"]`,
because "the staticlib is what a client on another engine links (SL10 clause 2)", and a static library
linked into another engine cannot assume a Rust thread pool.

- **The kernel stays single-threaded inside `vd-terrain`.** It exposes a step function, so a HOST that
  has a thread pool may drive whole BODIES in parallel — one job per body, no shared state.
- Parallelism INSIDE one solve needs a pool inside the generator. That is **D-C11**, an owner decision,
  and until it is taken every solve figure in this document is a one-core figure.
- **The CLIENT's chunk work is already parallel and already measured**: slice 7 ran 14 independent chunk
  jobs at **8.4x**. §7.5 uses that factor for the vista, and only for the vista, because chunk jobs are
  independent and a solve's passes are not.

The 8.4x figure is MEASURED on independent chunk jobs. Basins are wildly unequal — one continent's basin
dwarfs an island's — so even with a pool the solve's factor would be lower. UNMEASURED (§18 C8).

---

## 5. Component 1 — the macro solve

### 5.1 How a fine column addresses a macro node (the rule revision 1 got wrong)

The home planet's face edge at rung 0 is **N = 5 263 360 cells** (DERIVED: `Ladder::for_radius` snaps
`R·π/2` to a multiple of `2^(rungs−1) = 2 048`; `2 570 x 2 048 = 5 263 360`, and
`5 263 360 x 2/π = 3 350 759 m`, which is the radius `terrain_cost` prints — the arithmetic checks
against a MEASURED number). Its factors are **`N = 2^12 x 5 x 257`**.

> **THE ADDRESS RULE. A macro lattice is lawful when a fine column can name its macro node and its
> interpolation weight with EXACT INTEGER ARITHMETIC. That is true exactly when the lattice's node
> spacing is a WHOLE NUMBER of rung-0 cells — that is, when its nodes-per-face-edge `M` divides `N`.**

```
   FAMILY 1 — A RUNG.  M = N >> L, so the node is 2^L metres.
      node index = i >> L                    one shift
      weight     = (i & (2^L − 1)) / 2^L     a DYADIC fraction: exact in binary

   FAMILY 2 — ANY WHOLE DIVISOR.  M | N, so the node is s = N/M metres, a whole number.
      node index = i / s                     one i64 division (exact)
      weight     = (i mod s) / s             an exact rational; ONE correctly-rounded
                                             IEEE division, identical on both hosts
      On the home planet: M in {1,2,4,...,4096} x {1,5} x {1,257}
      -> 640 (8 224 m), 320 (16 448 m), 1 280 (4 112 m), ...

   FORBIDDEN — a lattice whose node is NOT a whole number of rung-0 cells. Then a node
   boundary falls INSIDE a rung-0 cell at a fraction neither host can name the same way
   twice, and each rounds it alone. That is the drift class SL10 removes.
```

**Family 2 is not a drift risk, and the crate already proves it.** `tube_region`
(`crates/terrain/src/carve.rs:66-75`) indexes a lattice of **512 metre** regions by dividing a metre
coordinate and flooring it, and the golden legs measure that lattice equal on four legs today. IEEE
division and `floor` are in the fence's own allowed set. The numerator and the divisor are both `i64`,
the division is integer, and only the final weight is one correctly-rounded IEEE division of two exactly
representable integers.

**What family 2 costs.** One `i64` division per column, about 20 to 40 cycles, plus one `f64` division:
about **10 ns per column** — and it is hoistable, because inside one 62-cell chunk the weight numerator
rises by exactly one per column, so the whole chunk needs ONE division and 61 integer increments. §7.1
charges 5 ns per column, which is generous.

**Domain 03 chose `M = 640` on the home planet** (`03_erosion_rivers.md:57-59`): 8 224 m per node,
2 457 600 nodes, 0.4 % over its 8 192 m target, and 640 divides `N` exactly. That is family 2, and this
document prices it.

**THE PER-BODY LATTICE — the measurement revision 2 owed** (refutation A D5, right: "any whole divisor"
is not a usable answer until somebody runs the chooser over the world's bodies). I ran it, in `python3`,
over the seven body radii `01_grid_family.md:302-309` prints, applying `Ladder::for_radius`'s own snap:

```
   MEASURED (a replay that could have failed: the home planet's row reproduces
   N = 5 263 360 and M = 640 = 8 224 m, which is domain 03's own choice)

   body            N            factorisation           M      node m    nodes      solve s   state MB
   -------------   ----------   ---------------------   -----  --------  ---------  --------  --------
   asteroid 500 m         785   5 · 157                   157       5     147 894      0.7       7.7
   moon 200 km        314 112   2^8 · 3 · 409              64   4 908      24 576      0.1       1.3
   Ceres              742 912   2^9 · 1451                128   5 804      98 304      0.5       5.1
   Luna             2 728 960   2^10 · 5 · 13 · 41        328   8 320     645 504      3.2      33.6
   HOME PLANET      5 263 360   2^12 · 5 · 257            640   8 224   2 457 600     12.1     127.8
   Mars             5 324 800   2^14 · 5^2 · 13           650   8 192   2 535 000     12.5     131.8
   Earth           10 006 528   2^12 · 7 · 349            698  14 336   2 923 224     14.4     152.0
   super-Earth     20 013 056   2^13 · 7 · 349            698  28 672   2 923 224     14.4     152.0
```

**Two facts the table forces, and both are new.**

1. **Every body in THE world has a usable divisor.** The nearest divisor to the 8 192 m target is inside
   20 % on every body except the tiny ones, where the body itself is smaller than the target node.
2. **A constant node SIZE does not survive a big body.** At 7 168 m an Earth-sized planet would hold
   11 692 896 nodes: **58 s and 608 MB**; a super-Earth would hold 35 809 494 nodes: **180 s and
   1.86 GB**. Both are dead. So the chooser must walk COARSER until the ceilings hold, which gives Earth
   14 336 m and a super-Earth 28 672 m — **a big planet gets a coarser landscape, and the owner must
   know that** (D-C2).

### 5.2 Memory

Two byte budgets, because the artifact has two lives.

**While it is SOLVED** (the working state), per node. **Revision 2 wrote `f32` here, and the fence bans
`f32` by name** (refutation B B1, right: `crates/terrain/src/gf.rs`'s header ends its list of deliberate
omissions with "and `f32` anywhere", and `clippy.toml` repeats it in a lint reason). Every accumulator
is `Gf` (8 B) or an exact integer:

```
   z_terrain, i32 millimetres              4 B     exact, and it bounds the state (E4)
   z_flood,   i32 millimetres              4 B
   receiver direction, u8                  1 B     which of the eight neighbours it drains to
   drainage area, Gf square metres         8 B     AREA, not a node count (E8); Gf, never f32 (E10)
   discharge Q, Gf                         8 B
   lake id, u32                            4 B
   precipitation, i16                      2 B
   ice depth, i32                          4 B
   the talus second buffer, i32            4 B
                                          ----
                                           39 B  ->  40 B with alignment
   plus the heap (8 B per node) and the flood's stack (4 B per node)
```

**The metric is NOT a per-node field** (refutation B N3, right that three quantities do not fit in 4 B).
The cell area and the two spacings are a function of the face coordinate alone, so a row scan holds ONE
row of them: `O(M)`, not `O(M²)`. That removes revision 2's 4-byte row instead of packing it.

**While it is READ** (what a chunk samples), per node — domain 03's own artifact
(`03_erosion_rivers.md:301-308`):

```
   eroded height Z, i16 metres             2 B
   D8 receiver + facies                    1 B
   discharge, a log byte                   1 B
                                          ----
                                            4 B    plus a lake table and a coarse pyramid (19 KB top)
```

**Why `i16` METRES is enough here, and was not enough for revision 2's 257 m pyramid** (refutation A D6
and refutation B W1, both right about revision 2). A one-metre quantum on an 8 224 m node is a slope
quantum of **0.012 %**, against the 781 m octave's characteristic slope of about 1.1 % — a hundredth of
the natural slope, and invisible. On revision 2's 257 m grid the same quantum was 0.39 %, a third of the
natural slope, which is a visible terrace. **The record problem was made by the fine grid and it goes
away with the coarse one.** If the refinement of §5.6 is ever bought, its read record must be `i32`
millimetres (+2 B per node), and §5.6 prices that.

| Grid | m per node | nodes | read, 4 B | solve, 52 B (state + heap + stack) |
|---|---|---|---|---|
| the home planet's artifact: 6 x 640² | 8 224 | 2 457 600 | **9.83 MB** | 127.8 MB |
| the climate: 6 x 256² | 20 560 | 393 216 | 1.57 MB | 1.57 MB (no solve state) |
| Earth's artifact: 6 x 698² | 14 336 | 2 923 224 | 11.69 MB | 152.0 MB |
| Luna's artifact: 6 x 328² | 8 320 | 645 504 | 2.58 MB | 33.6 MB |
| rung 12 globally: 6 x 1 285² | 4 096 | 9 907 350 | 39.6 MB | 515 MB |
| **6 x 20 480² (257 m, shape A)** | 257 | 2 516 582 400 | **10.1 GB** | **131 GB** |

All DERIVED. **The resident total for a pilot standing on the home planet** is the body's artifact plus
the climate plus the artifact's coarse pyramid: **9.83 + 1.57 + 1.6 = 13.0 MB**. Against the **1.04 GB
of mesh the same vista holds** (§7.5) that is **1.2 %**, which is the right way to read it: the macro
map is not the memory problem, the mesh is.

### 5.3 The solve cost, and why the pass count is an AGE

Two stages. The base height is the existing octave field, evaluated once per macro node. Which octaves
are live at a node is decided by NYQUIST, and **revision 2 both mis-stated the rule and attributed it to
the crate** (refutation A D3, right). Here is the truth:

> **`octaves_at` does NOT implement a Nyquist rule.** `crates/terrain/src/body.rs:252-258` drops exactly
> ONE octave per rung, so at rung `r` the finest live octave has a wavelength of `48.83 · 2^r` m — about
> **49 times** the cell, not twice it. The macro lattice needs its own rule, and it is a NEW rule, not a
> generalisation of the crate's.
>
> **THE MACRO NYQUIST RULE: a lattice carries only the octaves whose wavelength is at least TWICE its
> node spacing.** On the home planet at 8 224 m that is `λ ≥ 16 448 m`, which is **octaves 1 to 5**
> (400, 200, 100, 50 and 25 km). Octave 6 is 12.5 km, which is 1.5 times the node, and folding it in
> would print a false pattern.

**One octave of disagreement with domain 03, and it is OWED to a joint decision.**
`03_erosion_rivers.md:418` says "octaves 0 to 5", which is six octaves, so it folds the 12.5 km octave
(171 m of amplitude) into a lattice that cannot carry it. Mine says five. The difference is 171 m of
relief moving between the solved field and the closed-form field, and one of the two documents must give
way (§17 item 10).

The erosion is the schedule of §4.1. The plate layout and the uplift field are one pass each, about
30 ns and 20 ns per node (ESTIMATED); at forty passes they are under 0.5 % and they are folded in as a
rounding. **The uplift RATE is not a few millimetres** (refutation A W2, right: a few millimetres times
a hundred passes is a kerb, not a range). For uplift and incision to reach a steady state inside forty
passes on a body whose relief is 14 304 m, the uplift must be of order `relief / PASSES ≈ 350 m` per
pass. §2's example now says so.

**THE PASS COUNT IS NOT A CONVERGENCE COUNT.** Revision 2 wrote a law — "the iteration count scales with
the grid's DIAMETER in cells, because the signal travels one cell per iteration" — and then exempted its
own patches from it. Both refuters caught the contradiction (A-D4, B-B2), and both are right that
revision 2 could not have it both ways. **The law itself is wrong for the solver domain 03 designs.**

```
   AN EXPLICIT SWEEP writes z_new(i) from z_old(i) and z_old(receiver(i)).
      A change at the river mouth reaches a headwater D nodes away after D sweeps.
      -> the diameter law, and 640 nodes would need about 1 280 sweeps.

   AN IMPLICIT SWEEP writes z_new(i) from z_new(receiver(i)), walking the stack from
   the outlet upwards. Every node is solved AFTER its receiver, in the same pass.
      A change at the river mouth reaches EVERY headwater in ONE sweep.
      -> no diameter law. `03_erosion_rivers.md:816` names the implicit sweep.

   SO WHAT DOES `PASSES` BUY? The landscape's EROSIONAL AGE. Forty passes cut the trunk
   valleys and leave the headwaters young. Eighty wear the range flat. Domain 03's D9
   sets it by MEASUREMENT against the per-basin hypsometric integral, never by taste.
```

**In the game's words.** The home planet's sea level falls. With an implicit sweep every valley on the
planet knows about it in the same pass, and the pilot's river starts cutting at once. With an explicit
sweep the news would crawl inland one 8 km node per pass, and after forty passes it would have travelled
329 km — so only the coast would have responded.

| Lattice | m per node | nodes | solve, 40 passes | read | solve state |
|---|---|---|---|---|---|
| 6 x 128² | 41 120 | 98 304 | 0.5 s | 0.39 MB | 5.1 MB |
| 6 x 320² | 16 448 | 614 400 | 3.0 s | 2.46 MB | 32.0 MB |
| **6 x 640² (domain 03's, recommended)** | **8 224** | **2 457 600** | **12.1 s** | **9.83 MB** | **127.8 MB** |
| 6 x 1 024² | 5 140 | 6 291 456 | 31.0 s | 25.2 MB | 327 MB |
| 6 x 1 285² | 4 096 | 9 907 350 | 48.9 s | 39.6 MB | 515 MB |
| 6 x 2 570² | 2 048 | 39 629 400 | 196 s | 158 MB | 2.06 GB |
| **6 x 20 480² (257 m, shape A)** | **257** | **2 516 582 400** | **13 007 s = 3.6 h** | **10.1 GB** | **131 GB** |

All DERIVED from §4.1's schedule, ON ONE CORE (§4.3).

**Read the table with three lines drawn across it.**

- **A first visit to a body.** The solve runs ONCE EVER and is cached in the shard's store keyed by
  `(world tag, body seed)` (domain 03's own rule). So the ceiling is not a boot ceiling; it is what a
  player waits for the FIRST time anybody visits a body. A defensible ceiling is **20 s on one core**,
  and it is the owner's number, not a fact (D-C4).
- **The solve state.** 128 MB on the home planet is affordable on a shard and uncomfortable on a client.
  **160 MB** is the ceiling the chooser reads (D-C4).
- **The emulated x86-64 leg** (§11). Twenty times the native cost, UNMEASURED.

### 5.4 One core, the passes, the cube net, and the metric

The rule the code must obey is: **the same bytes on every host, every time.** Each pass has an order,
and each order has to be stated, because an order that changes is a drift SL10 forbids.

```
   PASS                       ORDER                       WHY IT IS BYTE-IDENTICAL
   -----------------------    ------------------------    --------------------------------------
   base height                a fixed row scan            pure per node; no node reads another
   the metric (area, spacing) a fixed row scan            pure per row, from the bend
   plate layout, uplift       a fixed row scan            pure per node
   receivers and slope        a fixed row scan            reads neighbours, writes its own node;
                                                          ties broken by the LOWER node index (E2)
   PRIORITY FLOOD             a heap keyed by             the key's second half is a total order,
                              (height, node index)        so two equal heights never tie
   BASIN LABELLING            pointer jumping; a fixed    the forest is fixed before the pass;
                              round count per pass        the round count is a recipe constant
   the stack build            one basin at a time, in     a basin is a tree; its order is its own
                              basin-label order
   flow accumulation          summed in the basin's own   floating addition is not associative,
                              stack order                 so the order IS the answer
   the IMPLICIT incision      stack order, outlet first   each node reads its receiver's NEW value
   the deposition term        the same stack order        the flux is carried down the same walk
   hillslope diffusion        a fixed row scan            a stencil read of the PREVIOUS pass
   the isostatic rebound      restrict, smooth, prolong   a fixed pyramid, a fixed order
   the talus relaxation       a fixed row scan, 8 passes  a stencil read of the previous pass
   the ice pass               a fixed row scan            pure per node from the climate
   the digest                 a fixed node order          the fold is the world identity
```

- **Every pass writes to a DIFFERENT array from the one it reads**, except the implicit incision, which
  reads its receiver's NEW value BY DESIGN and is safe because the stack order guarantees the receiver
  is already final.
- **No atomics anywhere on the arithmetic path.** An atomic add over floats is order-dependent, and it
  is the classic way a parallel field drifts.
- **If D-C11 ever grants a thread pool**, the split must be into a FIXED number of bands whose count is
  a recipe constant, never `available_parallelism()`, and E3's control is the gate.
- **The pointer-jumping round count must be large enough, and it is a constant** (refutation A D9,
  right). A basin on a 640-node face can be a thousand nodes long, so the count is `ceil(log2 1 280) =
  11`, not four. A count that is too small leaves nodes with the wrong basin label, which changes the
  ANSWER and not only the cost. E1's control fixes it in the recipe; §4.1 charges eleven rounds.

**THE CUBE NET.** The macro lattice lives on six faces bent by `W(a)` (`crates/seed/src/bend.rs`). Every
pass above reads neighbours.

```
   THE NEIGHBOUR RULE (rule E7)

   A macro node's 8 neighbours are named by the CUBE NET, not by the array.
   At a face edge the (a, b) axes rotate into the next face's axes; the mapping is
   the same one HR4's G-MAPPING-TABLE already gates (`crates/seed/src/seam.rs`:
   24 directed edges, generated from the face basis and pinned by a digest).
   At each of the cube's EIGHT corners only three faces meet, so a corner node has
   SEVEN neighbours, never eight. The rule names it; it is not an accident.

              face +X            face +Y
          +---------------+---------------+
          |               |               |     a node on this edge reads three of
          |               |<-- the edge --|     its neighbours from the face on the
          |               |               |     other side, with a and b swapped
          +---------------+---------------+     and one of them negated

   WHY IT MATTERS, in the game's words: without the rule, a river that reaches a
   face edge finds no receiver and STOPS. Twelve edges of 5 263 km each would ring
   the home planet with dead drainage, and the divide between two basins would land
   ON the cube edge — a straight ridge 5 263 km long, the most recognisable
   "this is a cube sphere" artefact there is.
```

**THE METRIC.** Stream power reads an AREA in square metres and a SLOPE in metres per metre. A
cube-sphere node is not the same size everywhere. I measured the variation by replaying the crate's own
bend constants (`K1 = π/4`, `K2 = 0.15`, `K3 = 1 − K1 − K2`) in `python3`, using the area element
`W'(a)·W'(b) / (1 + W(a)² + W(b)²)^{3/2}`:

```
   MEASURED (a run that could have failed: the six faces' integral came to 12.566386
   against 4π = 12.566371, a relative agreement of 1.2 x 10^-6)

      largest node area factor   0.61685   (at the face centre)
      smallest node area factor  0.43274
      RATIO                      1.425
```

**So a node near a face edge is 30 % smaller than a node at the face centre.** Counting nodes instead of
summing areas would make the drainage area wrong by up to 43 %, biased systematically toward the face
edges — a six-fold pattern on the planet, exactly the artefact the cube net must never show. Hence
**E8**: the accumulation carries square metres, and the slope divides by the metric spacing, both
computed once per row from the bend (about 10 ns per node, once, not per pass).

### 5.5 The three shapes, priced

```
   SHAPE A — ONE GLOBAL SOLVE AT THE RESOLUTION A WALKABLE VALLEY NEEDS (257 m)
   ---------------------------------------------------------------------------
   6 x 20 480² = 2 516 582 400 nodes. 40 passes on §4.1's schedule = 13 007 s
   (3.6 hours) on one core. 10.1 GB read, 131 GB of solve state.
   KILL, at any measured schedule cost. Nothing here is a boot, a ship or a cache.

   SHAPE B — ONE GLOBAL SOLVE AT 8 224 m, ONCE EVER, CACHED  (RECOMMENDED)
   ---------------------------------------------------------------------------
   6 x 640² = 2 457 600 nodes. 40 passes = 12.1 s on one core, 127.8 MB of solve
   state, 9.83 MB kept, plus a 1.6 MB coarse pyramid.
   It is domain 03's own recommendation (`03_erosion_rivers.md:1601`), and this
   document's independent op model agrees with its 12-20 s.
   WHO PAYS IT, AND WHEN:
      the shard   once ever per body, cached in its store by (world tag, body seed)
      the client  either the same 12.1 s once (and cached on disk, D-C7), or
                  nothing at all if the owning realm SHIPS the artifact (D-C20)
   Verdict: it fits every ceiling except a cold first visit, which is 12.1 s and
   needs a stated answer (§8.1). Its risks are the cube seam (M-L17), the fenced
   climate polynomials (M-L23) and the pass count (M-L2).

   SHAPE C — A PYRAMID OF ON-DEMAND PATCHES  (revision 2's recommendation)
   ---------------------------------------------------------------------------
   Now priced as the REFINEMENT domain 03 defers (§5.6), not as the shape.
   Its own arithmetic killed it as a replacement for shape B:
      - flow routing is a GLOBAL structure, and an implicit sweep carries a
        base-level change across a whole basin in one pass, so NO halo is ever
        wide enough to make a patch independent (§5.6);
      - the arrival census is 10 to 11 patches, ESTIMATED 6.5 s per arrival, and
        it is paid AGAIN every time the pilot travels (§5.6);
      - mean-pinning conserves mass, so a patch invents a raised shoulder beside
        every valley it cuts (§5.6, D-C17).
   As a refinement UNDER a global solve, all three shrink: the routing is
   inherited, the pinning is against a solved parent, and the patch adds detail
   only. That is the lawful shape, and it is a later slice.
```

**The comparison that decides it, and it is not the one revision 2 made.** Revision 2 killed a global
map because it priced one at 257 m. At 8 224 m a global solve costs 12.1 s ONCE PER BODY; a patch
pyramid costs an estimated 6.5 s PER ARRIVAL, for ever. A pilot who lands in four places on the home
planet has already paid more for the pyramid than for the whole planet.

### 5.6 The deferred refinement, and what it would cost

Domain 03's D2 answers "do we ever refine near a player?" with **"(a) never, now; (b) a fixed tiling
refined to about 2 km with a halo, in a later slice"**. This section is that later slice's price, and it
carries the contract revision 2 wrote, corrected where the refuters were right.

**THE GEOMETRY, fixed** (refutation A B3, right: revision 2's 256-cell patch did not tile a parent grid
whose ratio was five, so `256 / 5 = 51.2` and three clauses of its own contract could not all hold). A
patch's kept edge and its halo must both be whole multiples of EVERY ratio in the pyramid:

```
   ratios 8, 8, 5     ->  the kept edge must divide by 40, and so must the halo
   kept 320 nodes     320 / 8 = 40      320 / 5 = 64      whole
   halo  40 nodes      40 / 8 =  5       40 / 5 =  8      whole
   built 400 nodes    400 / 8 = 50      400 / 5 = 80      whole

   AND the level-3 patch then spans 320 x 257 m = 82.24 km, which is EXACTLY one
   level-0 node — so a patch is addressed by naming a parent node, not by arithmetic
   that could round differently on two hosts.
```

| Patch | nodes built | passes | one core | kept, 5 B | kept, 9 B (finest) | solve state, 40 B |
|---|---|---|---|---|---|---|
| 320² kept, 40 halo | 160 000 | 32 | **0.62 s** | 0.51 MB | 0.92 MB | 6.4 MB |
| 320² kept, 40 halo | 160 000 | 64 | 1.24 s | 0.51 MB | 0.92 MB | 6.4 MB |
| 256² kept, 64 halo (revision 2's) | 147 456 | 32 | 0.57 s | — | — | 5.9 MB |

The halo waste is `400² / 320² = 1.5625`, so **36 % of the work is thrown away** — better than revision
2's 56 %, because a wider kept square amortises the ring.

**THE CONTRACT, five clauses.**

```
   C1  A PATCH IS A REFINEMENT, NEVER A SECOND EROSION.
       A patch starts from the artifact's height, bicubically upsampled. Its uplift,
       its climate, its base level AND ITS ROUTING are READ from the parent.
       WHY IT MATTERS MORE THAN REVISION 2 SAID: an implicit sweep carries a
       base-level change across a whole basin in ONE pass (§5.3), so a patch can
       never be independent of the land outside it, at any halo width. Revision 2
       sized a 64-node halo from a 32-iteration explicit argument; refutation B B2 is
       right that the argument does not hold. The cure is not a wider halo. It is
       C3: import the routing instead of recomputing it.

   C2  MEAN-PINNING, AND WHAT IT COSTS. After every pass, each parent node's block of
       children is shifted so that the block's mean height equals the parent's height.
       IT BUYS: the ladder's coarse-answer law, true by construction.
       IT COSTS: mass conservation, which is NOT what erosion does.
          a valley of depth h over a fraction f of the block raises the rest by
          f·h / (1 − f);  at f = 0.30 and h = 300 m that is 129 m of INVENTED shoulder
       Refutation A W1 and refutation B D8 are both right, and revision 2 printed the
       artefact in §2 as if it were a feature. A real interfluve sits near the
       pre-incision surface, never above it. THE FORK IS D-C17:
          (i)  pin the mean, accept the shoulder, and bound it by the level's own
               detail amplitude (M-L21 measures it in metres);
          (ii) pin the mean to the parent MINUS the level's own modelled denudation,
               which removes the shoulder and makes the coarse level the fine level's
               mean plus a stated per-level offset (M-L14 measures the step).
       A SECOND CONSEQUENCE the measurement plan must carry: under (i) a patch's
       hypsometric integral CANNOT move, by construction, so M-L13's fourth statistic
       is measurable only on the global solve.

   C3  THE FLOW SEED. Every node on the patch's outer halo ring is seeded with the
       DRAINAGE AREA and the RECEIVER its parent node carries, rescaled by the node-area
       ratio, before the first accumulation runs.
       WHY: without it a river entering the patch from a 400 km catchment arrives with
       the area of the halo alone, so the incision is wrong by the square root of that
       ratio and the vista shows a stream where a river belongs.

   C4  THE SHARED EDGE IS EXACT, NOT TOLERANT. Patches tile on a fixed grid derived
       from the address (E6). The nodes ON a tile boundary are computed by a rule BOTH
       neighbouring patches evaluate identically: the parent's bicubic value, with no
       erosion applied. A chunk on either side reads the same integer MILLIMETRE, which
       is why a refinement needs the i32 read record (+2 B per node, §5.2).
       WHY: slice 6 spent four shape fixes making the extractor's seams EQUAL, never
       merely close. A p99 tolerance is the wrong quantity: one column in the tail is
       one HOLE, and the collider rides the same shape. M-L4's bound is ZERO quanta.

   C5  WHAT C4 COSTS. Pinning the boundary fades the carved detail to nothing at a tile
       line: the height agrees exactly, but the SLOPE has a crease, and a crease every
       82 km is a grid on the planet. The alternative is a PARTITION OF UNITY over four
       overlapping patches with address-derived weights: exact AND creaseless, at 4x
       the build (2.48 s per patch, not 0.62 s). M-L4 measures the crease in pixels
       and the owner chooses (D-C3).
```

**THE ARRIVAL CENSUS — the number revision 2 never computed** (refutation A B2, right: revision 2
counted one patch per level and a descent needs many). Two staging rules were tried, and the second is
the right one:

```
   RULE 1 (revision 2's): a level is needed while its NODE subtends a drawn pixel.
      level 3 at 257 m  ->  needed inside 223 km      -> about 20 patches
   RULE 2 (this revision's): a level is needed while its own DETAIL AMPLITUDE subtends
   a drawn pixel — the same law the reach ruling itself uses (an EXTENT, not a cell).
      level 3, detail amplitude ESTIMATED 60 m   ->  needed inside  52 km  -> 4 patches
      level 2, detail amplitude ESTIMATED 300 m  ->  needed inside 261 km  -> 4 patches
      level 1, detail amplitude ESTIMATED 1500 m ->  needed inside 1304 km -> 2.5 patches
                                                     ---------------------------------
                                                     ABOUT 10.5 PATCHES = 6.5 s per arrival
```

The amplitudes are **UNMEASURED**; M-L20 measures them on the built levels and pins them per level in
the recipe, so no host evaluates them at run time. **A cold login on the ground pays the same 6.5 s**,
because from a surface every level is inside its own range at once (refutation B D5, right) — there is
no approach to spread it over.

**So the refinement's honest price is about 6.5 s of one-core work per arrival, for ever, against
12.1 s once per body for the global solve.** That comparison is why §16 D-C1 recommends the global solve
and defers the refinement.

### 5.7 THE COMPOSITION RULE — how the solved field joins the octaves

**Revision 2 never stated it, and its sibling had already decided it the other way** (refutation A B1,
right). `03_erosion_rivers.md:834` states it in four words:

> **`Z` REPLACES the coarse octaves; it does not add to them.**

```
   h(dir, rung)  =  radius
                 +  Z(dir)                        the solved field, bicubic, ONE macro read
                 +  Σ octaves FINER than the lattice's Nyquist limit
                 −  C(dir, rung)                  the channel and floodplain carve
                 +  T(dir, rung)                  scree, ridges, gullies (domain 03)
```

**Why it must be a replacement.** The solve's own base height IS the coarse octave field. If a column
added both, the home planet's long-wave relief would double from 14 304 m to about 26 000 m, every band
in the ladder would move, and every stored edit would be at the wrong height. Revision 2 priced the
addition (§7.1 charged the macro sample as an extra on a column that still summed fourteen octaves) and
would have shipped exactly that defect.

**What the replacement costs, and what it saves.** At rung 0 today a column sums 14 octaves. Under the
rule, octaves 1 to 5 are inside `Z`, so the column sums 9:

```
   REMOVED   5 octave evaluations per column   at most 5 x 11.5 ns = 57.5 ns   (§4.1: an UPPER bound)
   ADDED     one bicubic macro read                                45 ns
             the channel and floodplain term                       45 ns
             the carve blend                                       10 ns
             the climate read (it replaces at most 23 ns of noise)  10 ns
```

**§7.1 books the saving at ZERO**, because 11.5 ns is an upper bound on a noise evaluation and using an
upper bound as a saving is the unsafe direction. The saving is at most 0.24 ms per chunk, and M-L5
measures it.

---

## 6. Component 2 — the river field

### 6.1 From a channel to a distance field

Revision 1 stored the channels as polylines and asked each fine column to search a 3 x 3 box of regions
for segments. Refutation B showed the arithmetic did not hold: the search charged 2.2 ns per lookup
against §4.1's own 80 ns for an irregular read, and `carve.rs`'s own precedent asks a wider box than
3 x 3 (`tubes_near` walks `lo−2 ..= hi+1`, `crates/terrain/src/carve.rs:124-131`).

**The solve already walks the receiver tree, so it extracts the channels once and stores a FIELD.** The
artifact's own discharge byte plus a distance transform over the macro lattice gives every column what
it needs with one read:

```
   PER MACRO NODE, in the artifact
      the discharge, a log byte                 1 B    (already in domain 03's 4 B record)
      the D8 receiver + facies                  1 B    (already there)
   PER MACRO NODE, added for the carve
      distance to the nearest channel, u16 m    2 B
      channel width and depth, u16 packed       2 B
                                               ----
                                                 8 B per node -> the artifact grows 9.83 -> 19.7 MB

   BUILD COST: one channel extraction (a threshold test and a link per node) plus one
   distance transform (two sweeps, about 50 ns per node):
      2 457 600 nodes x ~50 ns = 123 ms per body                 ESTIMATED
   READ COST: ONE bicubic sample of two fields per column, one gather per chunk.
```

**The carve's own resolution is the open question.** An 8 224 m node cannot place a 30 m channel; it can
only say a channel passes near. Domain 03 solves that with a closed-form `segment_distance_m` term at
the fine rungs, seeded by the macro receiver chain. **This document prices both and does not choose:**
the field costs 9.9 MB of artifact and 45 ns per column; a segment search costs an ESTIMATED 150 to
250 ns per column and no artifact bytes. **M-L6 measures both on THE world**, and reports drainage
density in channel-kilometres per square kilometre, which is the number a geomorphologist would ask for.

**The frame.** During the solve a channel point is `(i32 millimetres, i32 millimetres)` **in the FACE's
own frame**, not the body's: an `i32` in millimetres reaches ±2 147 km while the home planet's radius is
3 351 km, so a body-frame millimetre position would overflow (refutation B, right). Nothing here is an
SL1 question: a lattice address is an address, not a realm's position.

### 6.2 The carve at the fine rungs, and where it stops

The carve is a HEIGHT change, so it belongs in the **column pass**, never the cell pass: 62² = 3 844
columns per chunk instead of 62³ = 238 328 cells. Moving it to the cell pass would cost
238 328 x 150 ns = **35.7 ms per chunk**, over four times the whole budget.

**What that decision COSTS, stated plainly**: a column pass produces one height per column, so **it
cannot make an overhang**. No undercut cliff, no free-standing rock pillar with a waist narrower than
its head, no river bank that leans over the water. The reference picture's foreground is full of exactly
those. They must come from the CELL pass, where the cave carvers already make overhangs today
(`cavern_hollow_m`, `crates/terrain/src/carve.rs:55-64`), and pricing that mechanism is domain 02's job.
This document's contribution is the number that forbids the easy route.

**Rivers stop at the coarse rungs, exactly as caves do**, and the code carries the rule:
`tubes_carve_at` carves only where `cell_m <= tube_radius_m * 2` (`crates/terrain/src/carve.rs:34-37`).
A river 30 m wide has a radius of 15 m, so it is carved at rung 4 (16 m cells) and **dropped at rung 5
(32 m cells)**.

**And the difference between a dropped cave and a dropped river must be bounded.** A cave that stops at
a coarse rung was hidden underground; nobody sees it go. A river that stops is a visible notch in the
ground that vanishes.

> **The dropped carve's height step is at most the channel depth the artifact states at that node.**
> M-L14 measures it in pixels at the rung boundary. If it shows, the cure is the crossfade the octaves
> already use: the carve fades over the crossfade band instead of switching off.

---

## 7. Component 3 — the per-chunk budget, and the vista

### 7.1 The rung-0 accounting

```
   ---------------------------------------------------------------- 8.00 ms BUDGET
                                                                    (owner; terrain_cost.rs:41)
   TODAY, a surface chunk (3, 5)             3.30 ms   MEASURED
     sample box (columns 0.71 + cells 0.72 + halo and sites 0.55)   1.98 ms
     extraction (5 650 vertices)                                    1.32 ms

   ADDED BY THE LANDFORM WORK, per sample box (4 096 columns), under the
   COMPOSITION RULE of §5.7 (Z REPLACES octaves 1-5):
     the macro height read: bicubic, taps gathered once per chunk,
       plus the family-2 address (5 ns, hoisted to an increment)  0.18 ms  ESTIMATED (45 ns/col)
     the channel distance + width read: bicubic on two fields     0.18 ms  ESTIMATED (45 ns/col)
     the carve blend into the height                              0.04 ms  ESTIMATED (10 ns/col)
     the climate read (it REPLACES at most 23 ns of noise)        0.04 ms  ESTIMATED (10 ns/col)
     the FIVE octaves the macro field replaces                   -0.00 ms  booked at ZERO (§5.7);
                                                                 --------  at most -0.24 ms
                                                                  0.45 ms

     THE EXTRACTION GROWS, because the surface gets rougher. Priced at BOTH measured
     marginal costs (§3): the fit's 61 ns per vertex and the high-count 104 ns.
```

| Chunk | today | vertices | +column | +extraction at 1.5x | at 2.0x | **NEW at 2.0x** |
|---|---|---|---|---|---|---|
| surface (3, 5) | 3.30 ms | 5 650 | 0.45 | +0.17 to +0.29 | +0.35 to +0.59 | **4.10 to 4.34 ms** |
| cave-dense SEAM (21 223, 7) | 6.11 ms | 8 234 | 0.45 | +0.25 to +0.43 | +0.51 to +0.86 | **7.07 to 7.42 ms** |
| **cave-dense (3, 5)** | 5.22 ms | 23 084 | 0.45 | +0.71 to +1.20 | +1.42 to **+2.40** | **7.09 to 8.07 ms** |
| the checkerboard bound | 26.00 ms | 250 046 | — | — | — | synthetic; no chunk of THE world reaches it |

```
   THE BUDGET                  8.00 ms          |#####################################|
   surface chunk, 2.0x         4.34 ms          |####################                 |
   cave-dense seam, 2.0x       7.42 ms          |##################################   |
   cave-dense (3,5), 2.0x      8.07 ms          |######################################|  OVER
```

**The document must report a fail, not a pass** (refutation B D1, right that revision 2's 2.0× row was
arithmetically wrong and that the corrected number is over budget). **At twice the vertices, on the
densest chunk slice 6 measured, the landform work puts the chunk over the 8 ms budget by 0.07 ms.** The
kill row (§14) fires there, and its answer is the owner's ruling V10: measure it, then decide.

**The extraction growth factor is the whole headroom and it is UNMEASURED.** M-L18 prints the vertex
count before and after on the named set.

### 7.2 The trick that makes the macro read nearly free

A rung-0 chunk spans 62 m. A macro node spans 8 224 m. So **every column of the chunk reads the SAME
small square of macro taps.** The chunk gathers a 4 x 4 tap tile once (16 reads, possibly 16 cache
misses, about 1.3 µs), and every column then does arithmetic on values already in registers.

| Rung | chunk spans | tap tile at 8 224 m | bytes at 4 B (8 B with the channel field) |
|---|---|---|---|
| 0 | 62 m | 4 x 4 | 48 to 96 B |
| 5 | 1 984 m | 4 x 4 | 48 to 96 B |
| 8 | 15 872 m | 5 x 5 | 100 to 200 B |
| 11 | 126 976 m | 19 x 19 | 1.4 to 2.9 KB |
| above the top rung | — | the artifact's coarse pyramid | 19 KB at the top |

Every tile fits in L1 at every rung. **The cost is one gather per chunk plus arithmetic per column, at
every rung.**

**Bicubic, not bilinear.** Bilinear interpolation of a height field is continuous, but its SLOPE jumps
at every node edge. On the home planet that prints an 8 224 m grid of creases across the whole vista — a
seam, and SL8 says a seam is a defect. Bicubic (16 taps) is smooth in slope and costs about 40
operations per column against about 12. Both are gathered once per chunk, so the difference is 28 ns per
column = 0.11 ms per chunk. **Pay it.** M-L5 measures both, and the pop detector judges.

### 7.3 What must never happen

The river carve must never move into the cell pass (§6.2: 35.7 ms per chunk). It is recorded here
because it is the easy mistake to make when somebody wants an overhanging bank.

### 7.4 The solve is not a rate; the REFINEMENT would be

The global solve runs ONCE EVER per body and is cached, so it has no rate: a pilot who flies around the
home planet for a week pays it once. That is the strongest cost property of shape B and it is why §5.5
recommends it.

**If the refinement of §5.6 is ever bought, it DOES have a rate, and revision 2's own table fired its
own kill row without saying so** (refutation A D7, right):

```
   a level-3 patch spans 320 x 257 m = 82.24 km and costs 0.62 s to build (§5.6)
   crossing a patch boundary enters a new ROW, so at least THREE new patches are needed

   speed        one row every    build share of one core     is that speed possible at that altitude?
   ---------    -------------    ------------------------    ---------------------------------------
   1.4 m/s      16.3 hours       0.003 %                     a walk
   240 m/s      5.7 minutes      0.54 %                      a low pass
   3 000 m/s    27.4 seconds     6.8 %                       a fast low pass
   30 000 m/s   2.74 seconds     68 %                        NO: at that speed the pilot is far above
                                                             the ground, and every chunk in her view is
                                                             drawn at a rung that reads a COARSER level
```

**The kill row is at 10 % of one core (M-L15), so the refinement fails it above about 4 500 m/s at low
altitude.** The lawful answer is not to stand in a coarse level while a fine one builds — that would be
a shape that changes with time, which V1 clause 1 forbids by name. The answers are: build ahead along
the occupant's own address chain (§7.4's lead is always inside the patch she is leaving), or let the
host build whole patches in parallel (D-C11), or refuse the refinement above a stated speed, which is a
gameplay statement and the owner's to make.

**THE SHARD'S COLD START, which revision 2 had no answer for** (refutation A D8, right). A lead exists
only when there is a previous position. It does not exist for a login, a transfer in from a hull, a warp
arrival, or a shard restart after a kill-9 — and ruling V1 clause 5 says the shard computes collision on
the same shape, so until the shape exists there is no collider and the pilot falls through the world.

> **THE RULE: an occupant is not ADMITTED to a body's realm until the body's artifact is loaded and, if
> the refinement is bought, the patch under the occupant's own address is built.** Admission waits; it
> never admits and then builds. The wait is 12.1 s on a cold first visit to a body (or nothing, from the
> shard's cache), and 0.62 s for a patch. That is a REFUSAL, which SL8 permits, and it is not a seam,
> which SL8 does not.

### 7.5 THE VISTA — the cost the document was written for

**No section of revision 2 multiplied a per-chunk cost by the number of chunks the reference picture
holds** (refutation B B4, right; this is the owner's own question). Here is the arithmetic.

```
   ASSUMPTIONS, each named
     - a 90 degree horizontal field of view over flat-ish ground
     - the ladder picks the rung whose cell subtends about one drawn pixel:
          cell = 1.1506 mrad x d               (the code's own floor, §3)
     - a chunk is 62 cells (ladder.rs:21), so a chunk edge = 0.07134 x d
       and a chunk footprint = 0.005089 x d^2
     - a ring from d to 2d in a 90 degree wedge covers 2.356 x d^2

   chunks in one doubling ring   =  2.356 / 0.005089   =   463
   rings from 100 m to 51.2 km   =  9 doublings        =   4 167 chunks      DERIVED
```

| What | one core | at slice 7's MEASURED 8.4x | mesh bytes |
|---|---|---|---|
| the vista today, server cost 3.30 ms | 13.8 s | 1.64 s | — |
| the vista with the landform work, 3.92 ms | **16.3 s** | **1.94 s** | — |
| the vista today, CLIENT cost 4.18 ms (MEASURED, slice 7 M7-2) | 17.4 s | 2.07 s | — |
| **the vista with the landform work, client 4.80 ms** | **20.0 s** | **2.38 s** | — |
| resident mesh, at 250 KB average per chunk | — | — | **1.04 GB** |

**Three consequences, and each is new to this document.**

1. **The landform work costs +2.5 s of one-core work per vista, not +0.45 ms.** At the client's fourteen
   threads it is +0.3 s. That is the honest way to read §7.1's per-chunk delta.
2. **The first full vista is 2.38 s at fourteen threads, over the 2 s line SL8 draws for a stall.** The
   macro solve's own 12.1 s sits BESIDE that, not inside it: a cold first visit to a body is
   12.1 + 2.4 s unless the artifact is cached or shipped (§8.1).
3. **1.04 GB of mesh is resident for one vista.** Slice 7 measured 205 MB for a 13 x 13 x 3 patch of
   chunks, so the order is right. **The client's own mesh residency ceiling is not stated anywhere in
   this project**, and it must be (D-C4). Against it, the 9.83 MB artifact is 1 %.

**M-L19 is the measurement**: on the shipped flight, standing on the home planet and looking at a 50 km
horizon, count the chunks resident, their vertices, their bytes, and the wall time from arrival to a
full vista, with the landform work off and on.

**THE BUDGET'S OWNER IS UNSTATED, and it changes the answer** (refutation B D2, right). The 8 ms
constant lives in a SERVER example (`crates/bins/examples/terrain_cost.rs:41`), and the client's own
measured chunk is 4.18 ms against the server's 3.30 ms, because SL10 makes the client build the same
shape and then also make positions and normals. A missed budget is an ARRIVAL RATE on the server and a
STALL on the client. **D-C18 asks the owner which the 8 ms governs.**

---

## 8. Component 4 — the client's first visit, the world identity, and the resolution rule

### 8.1 Derive, or receive

**Revision 2 killed shipping with one line — "deriving costs 0.58 s, so shipping buys nothing" — and
that line was true only of a 82 km lattice too coarse to hold the picture.** At the lattice domain 03
designs, the comparison is real:

| What the client holds or receives | raw | after a slope predictor and an entropy coder (ESTIMATED) | seconds at 10 Mbit/s | the alternative: derive it |
|---|---|---|---|---|
| the home planet's artifact, 6 x 640² at 4 B | **9.83 MB** | 3 to 5 MB | 2.9 to 4.2 s | **12.1 s on one core, 128 MB peak** |
| the artifact's coarse pyramid (the approach) | 19 KB | 8 KB | 0.01 s | free with the artifact |
| the climate, 6 x 256² at 4 B | 1.57 MB | 0.6 MB | 0.5 s | 0.08 s on one core |
| a whole body at 257 m (shape A) | 10.1 GB | 5 GB | 66 hours | 3.6 hours, 131 GB |

**So the fork is real and it is the owner's** (D-C20):

```
   DERIVE (SL10's own shape: one generator, two hosts)
      + no new lane, no SL6 ask, no wire arm; the picture depends on the seed alone
      + a disk cache makes every session after the first free (D-C7)
      - 12.1 s and a 128 MB peak on the CLIENT, once per body
      - it needs D-C9: the body's physical facts must be drawn INSIDE `vd-terrain`,
        because `vd-physics` is unfenced and the generator may not read it (§10.1)

   RECEIVE (domain 03's revision-2 recommendation, `03_erosion_rivers.md:1486`)
      + no compute and no 128 MB peak on the client; the coarse pyramid covers the approach
      + it removes the cross-host drift in the astrophysics entirely
      - an SL6 ask, a wire arm, and about 4 MB per body per player
      - the picture depends on a transfer, which is what D-C7's cache must then also
        carry a CONTENT digest for (refutation B N1, right that revision 2 used one
        objection to kill shipping and then recommended a disk cache with the same shape)
```

**This document recommends DERIVE with D-C9**, because it keeps one generator on both hosts and needs no
new lane, and because a disk cache removes the repeat cost. **Domain 03 recommends RECEIVE.** Neither is
a cost question alone, so §16 D-C20 states both prices and hands it over.

**The cold-start stall, answered without breaking SL10.** Revision 1 offered "build the coarse field in
the background, while the rungs draw from the octave field alone". **That is DELETED.** A shape that is
the octave field before the solve lands and the solved field after it is a **function of time**, which
V1 clause 1 forbids by name; the pilot would walk on a hill the shard does not have, which V1 clause 5
forbids; and the whole difference would pop into being mid-session, which SL8 forbids.

**The lawful answer is a REFUSAL plus a STAGING BY DISTANCE.** No ground is drawn, and no occupant is
admitted (§7.4), until the artifact that rung reads exists. On an approach the coarse pyramid arrives
first and the full artifact is needed only inside the near gate (§8.3), so the wait spreads. On a cold
login it does not spread, and the wait is the honest 12.1 s (or the ship's 3 s, or nothing from a
cache). **M-L7 flies both legs.**

### 8.2 The world identity: the call site revision 1 forgot

`crates/terrain/src/tag.rs` folds `golden_self_check(home)` into `WorldIdentity::of`, and that generates
**eight chunks of the home body**. It is called at boot by `crates/bins/src/bin/gateway.rs:193`, by
`crates/bins/src/bin/client.rs:118` (both through `vd_bins::world_identity`,
`crates/bins/src/lib.rs:1047`) and by `crates/bins/tests/home_body_pin.rs`.

**Once a chunk reads the macro artifact, every one of those call sites needs the artifact first.** The
gateway would pay 12.1 s at boot although it draws nothing. Every test binary that touches the home body
would pay it, three times over in `just terrain-pin` (debug, release, and `target-cpu=native`), and again
under coverage instrumentation, where a debug build is tens of times slower. This is the exact shape of
the defect the project already survived once — every process building the forest at boot, 5.3 GB in the
gateway.

> **THE FIX (D-C13): the golden self-check's macro input is a FIXTURE, not the home planet's artifact.**
> The self-check folds eight chunks over a small, named, self-contained lattice whose size is a recipe
> constant — 6 x 16², about 1 ms — and the home planet's own artifact is folded into the MACRO LEG
> (§11), which runs where it belongs and not in every process. The world identity keeps exactly the
> property it has today: two binaries that would draw different ground state different numbers.

### 8.3 Staging by distance — two gates, not one

A field only matters once what it carries subtends more than one drawn pixel. **Revision 2 had two rules
that disagreed with each other and left the climate out of both** (refutation B D7, right). Here are the
two gates, and each says which field it governs:

| Gate | The rule | On the home planet | What rides it |
|---|---|---|---|
| **THE FAR GATE** | the body's own COLOUR is legible at one pixel, so it is needed as soon as the body is drawn at all | the whole time the body is drawn | **the climate** (1.57 MB, 0.08 s): the deserts, the ice caps and the vegetation belts a planet shows from a window |
| **THE NEAR GATE** | the field's own RELIEF subtends a drawn pixel: `relief / 1.1506 mrad` | 14 304 m / 1.1506 mrad = **12 432 km** | **the artifact** (9.83 MB, 12.1 s): the ranges and the valleys |
| the artifact's coarse pyramid | its own top level's relief, ESTIMATED at half the body's | about 6 200 km | the displaced-sphere proxy above the top rung (§9.1) |

**Why the climate must ride the FAR gate.** At 20 000 km the home planet is a ball 469 pixels across.
Under revision 2's single rule it would be a UNIFORM ball with no deserts and no ice caps, and it would
grow them at 12 432 km — an arrival pop, which the seam taxonomy names (refutation A W4, right). The
climate is cheap (0.08 s, 1.57 MB), so it costs almost nothing to have it always.

**How many bodies need an artifact at once?** A body's artifact matters only inside 12 432 km, so **in a
star system at most one planet and its moons are that close at any moment.** That is a DERIVED answer
resting on a rule the code does not have yet, so **M-L16 counts the bodies inside that distance on the
flight the slice-7 gate already flies**, and the rule itself is D-C12.

### 8.4 The resolution rule, without a magic number

```
   MACRO_LATTICE(body) is chosen ONCE, OFFLINE, and PINNED PER BODY IN THE RECIPE.
   The chooser lists the WHOLE DIVISORS of N and takes the one whose node spacing is
   nearest MACRO_CELL_TARGET_M; while either ceiling fails it steps to the next
   COARSER divisor:
        (a) M divides N                                  -- exact integer addressing (§5.1)
        (b) 6 · M² · 52 bytes  <=  SOLVE_STATE_CEILING
        (c) 6 · M² · SCHEDULE_NS  <=  SOLVE_TIME_CEILING
   The chosen M rides GENERATOR_VERSION, so no host ever evaluates (b) or (c) at run
   time and two chips can never disagree.
```

**The four constants, stated ONCE for the world, and every one of them is the owner's** (D-C4):

```
   MACRO_CELL_TARGET_M    =  8 192 m         domain 03's own target
   SOLVE_STATE_CEILING    =  160 MB          UNSOURCED: chosen so the home planet's 128 MB fits
   SOLVE_TIME_CEILING     =  20 s, one core  UNSOURCED: chosen so the home planet's 12.1 s fits
   MESH_RESIDENCY_CEILING =  not stated anywhere in this project, and §7.5 needs it
```

**Two of them have no source, and revision 2 dressed one as a fact** (refutation A W5 and refutation B
W3, both right: "a pod is not promised more" cited §4.3, which is about a thread pool and says nothing
about a pod). They are stated here as CHOSEN and owner-owed, not as measurements.

**The tension the chooser carries, stated instead of claimed away** (refutation A W6, right). `SCHEDULE_NS`
is a measurement on one machine, so the resolution of every planet's landforms is decided by how fast one
Mac ran one benchmark in 2026. That is not a seed-derived number and it is not a per-entity property.
**The honest framing: it is an operational parameter that MOVES THE WORLD, so it is frozen once, recorded
in the tag, and never re-derived.** §15's no-magic-numbers row says that, rather than claiming a pass.

---

## 9. Component 5 — the coarse rungs, and the view from orbit

### 9.1 The ladder still coarsens for free

At the top rung the octave field is already cheap (266 µs per chunk column pass, 3 octaves, MEASURED),
and the new work is cheaper still: the river carve is off (§6.2), the caves are off (`caverns_carve_at`,
`crates/terrain/src/carve.rs:40-43`), and the macro tile is one gather of at most 19 x 19 taps (§7.2).
So **the ladder's law — rung L costs less than rung L−1 — survives the landform work**, and
`terrain_cost`'s existing assertion keeps guarding it.

The whole home planet at the top rung, DERIVED from MEASURED per-chunk figures (rung 11: box 1.05 ms
plus extraction 1.28 ms = 2.33 ms; 4 055 vertices per chunk, slice 6):

```
   chunks on the body at rung 11 = 6 x 42^2 = 10 584        (42 = ceil(2 570 / 62))
   the visible half                        =  5 292
   worker time to mesh the visible half    = 12.3 s on one core
```

and, as a SEPARATE calculation over the WHOLE body (10 584 chunks, refutation A N1's point that revision
2 mixed the two inside one block):

```
   mesh bytes if the whole body were resident at rung 11
      10 584 chunks x 4 055 vertices x 47.24 B per vertex  =  2.03 GB
```

**The conclusion is unchanged: a planet is never fully meshed, at any rung.**

Above the top rung the proxy takes over. **The artifact's own coarse pyramid is exactly the right
source**: a displaced sphere plus a normal map baked from its top level, which is a fraction of a
millisecond and 19 KB. **The macro solve pays for itself twice: once for the vista's landforms, and once
for the planet's look from a window.**

The task named "a planet at rung 15 = the macro map alone". On the home planet the ladder stops at
rung 11 (`rungs = 12`), and rung 15 is only the address's maximum (`RUNG_MAX`,
`crates/seed/src/ladder.rs:27`). Restated in the code's words: **above the top rung there is no chunk,
and the artifact's pyramid is what the proxy draws.**

### 9.2 Which field a rung reads

**With ONE global artifact the question nearly disappears, and that is a cost property worth naming.**

> **`Z` is rung-independent: one field at 8 224 m, read identically at every rung**
> (`03_erosion_rivers.md:1427`). So there is no level-to-level boundary, no crossfade to design and no
> pop to hide. What varies with the rung is only which OCTAVES are live, which is the machinery
> `octaves_at` already has and `terrain_cost` already gates.

If the refinement of §5.6 is ever bought, the question returns, and the rule is:

> **A rung reads the FINEST refinement level whose node is at least twice the rung's own cell size, and
> never a level whose patch does not exist.** The global artifact always exists, so the fallback is
> always there.

| Rung | cell | level read (if the refinement is bought) | m per node |
|---|---|---|---|
| 0–7 | 1–128 m | 3 | 257 |
| 8–9 | 256–512 m | 2 | 1 285 |
| 10–11 | 1 024–2 048 m | 1 | 10 280 |
| above the top rung | — | the global artifact | 8 224 |

And the step across those boundaries is bounded by C2's mean-pinning — with the invented shoulder D-C17
must first settle. **M-L14 measures it in pixels, because a law with no measurement is an argument.**

---

## 10. Component 6 — the climate, the weather, the season, and the sky

### 10.1 The climate field (static, seed-derived, both hosts)

The climate is the cheapest component and it buys the most believability, because it makes a biome a
CONSEQUENCE instead of a noise. It runs at its own global grid, 6 x 256² (20 560 m per cell).

**Its orography must be the ERODED height, not the octave field** (refutation B D3, right). Revision 2
took the climate's ranges from the un-eroded octaves, so the rain shadow landed on a range the rivers had
already moved: the pilot would walk over the range she can SEE, read the wind, walk to the lee, and find
grassland. The fix is one line and it costs nothing extra, because the solve already runs the climate on
its own schedule (`CLIMATE_EVERY = 10`, `03_erosion_rivers.md:813`):

> **The climate reads the artifact's height, and it is re-solved every tenth pass, so the rain carves
> the range and the range steers the rain.** The residual is the 8 224 m node: a rain shadow finer than
> that is not resolved, and the lapse rate recovers the SNOW line per column at any rung.

| Pass | Op count per cell | Cost at 393 216 cells |
|---|---|---|
| The base height at 20 560 m (octaves 1–4, the macro Nyquist rule) | — | **22 ms** DERIVED (69 ns per column is a rung-11 THREE-octave figure, so this row is an under-count; refutation A N4, right) |
| Insolation from latitude, obliquity, orbit and the star's luminosity | about 30 (a fenced polynomial) | **12 ms** ESTIMATED |
| Lapse rate: temperature falls with height | 2 | 0.8 ms |
| Prevailing wind from the spin and the Hadley bands | about 10 | 4 ms |
| Orographic moisture: an ADVECTION sweep along the wind, 4 passes | about 20 per pass | **31 ms** ESTIMATED |
| Ocean distance (a source term for the moisture) | about 15 | 6 ms |
| **Total, SIX passes** | | **about 76 ms on one core** |

(Revision 2 called this "5 passes" in one place and listed six here; it is six — refutation A N8, right.)

Per column at rung 0 the climate read is **about 10 ns**, and it replaces at most 23 ns of noise (§7.2).

**THE FENCE FORBIDS EVERY TRANSCENDENTAL, AND THE CLIMATE IS MADE OF THEM** (refutation B D9, right).
`crates/terrain/clippy.toml` bans `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `exp`, `ln`,
`log`, `powf`, `powi`, `cbrt`, `hypot`, every hyperbolic, `to_radians`, `to_degrees`, `mul_add`, `min`
and `max` by name, and `Gf` offers no `f32` at all. Each climate quantity therefore needs a COMMITTED
LITERAL polynomial or table, and **the error is the finding, not the cost**, because the error moves a
biome boundary and a biome boundary is the picture:

| Quantity | Its natural form | Under the fence | Its error |
|---|---|---|---|
| daily-mean insolation by latitude and obliquity | `sin`, `cos`, `acos` of the hour angle | a minimax polynomial in `sin(latitude)`, degree about 7, ~14 operations | **UNSTATED — M-L23** |
| the Coriolis parameter | `2 Ω sin(latitude)` | the same polynomial, reused | **UNSTATED — M-L23** |
| saturation vapour pressure (what "moisture" means) | Clausius–Clapeyron, an exponential | a committed 256-entry table on temperature, one array read | **UNSTATED — M-L23** |
| the atmosphere's density with height (§10.3's haze) | `exp(−h/H)` | a committed table on height | **UNSTATED — M-L23** |
| the channel depth law | `Q^0.4` | a committed 256-entry table on the log-discharge byte (domain 03's D4) | one quantisation step |

**M-L23 states, for each: the polynomial's degree, its maximum error over the argument's range, and the
BIOME-BOUNDARY DISPLACEMENT in kilometres that error causes.** A 1 % error in the insolation near the
pole moves the ice cap's edge by degrees of latitude, and nobody has measured it.

**THE BLOCKER.** The insolation scale needs the star's luminosity and the orbit's semi-major axis. They
exist — **in a crate the generator may not read**:

- `crates/terrain/Cargo.toml` has exactly ONE dependency, `vd-seed`.
- `BodyDefinition`'s doc comment carries the invariant: a body is drawn from a seed and never assembled
  from numbers computed elsewhere, "so no float from an unfenced crate can enter the recipe as a body".
- `vd-physics` is not fenced: its luminosity, semi-major axis and mass are ordinary `f64` computed with
  `powf` and `exp` (`crates/physics/src/worldgen/generate.rs:571`, `:598`, `:633`, `:1727`, `:1880`).

**The only lawful path is that `vd-terrain` DRAWS the body's physical facts from the seed itself**, and
`vd-physics` reads THEM rather than the reverse. That keeps one generator (SL10) and one draw. It moves
the body draw under the fence, which touches every body in THE world, so it is **D-C9, before the first
saved world.** The facts needed, and their state today: mass and radius exist; luminosity and the orbit
exist in `vd-physics`; **spin, obliquity and atmosphere do not exist anywhere**, and
`crates/terrain/src/height.rs:35` confirms the pole is the orbit's own axis.

### 10.2 The weather (live, the realm's, and an SL6 ask)

The owner's task says: *"It should be very believable, as we also should simulate the weather."*

```
   CLIMATE                                WEATHER
   ------------------------------------   ------------------------------------
   the long-run average                   the state at THIS tick
   a function of (seed, address)          a function of TIME and of live state
   BOTH HOSTS DERIVE IT (SL10)            the client MAY NOT derive it (SL10 clause 7:
                                            "the client never derives a pose, a velocity
                                            or any state"; revision 2 cited clause 3,
                                            which is the no-drift MEASUREMENT — A-N6, right)
   costs 76 ms once per body              costs per tick, on the owning shard
   nothing crosses                        it CROSSES, as a one-hop diff (V1 c.6)
                                          -> an SL6 ask, default NO
```

**What weather would cost, so the ask has a number.** Weather is a field of continents, not of valleys:
one state per 10 280 m cell is finer than any cloud front needs.

```
   PER SAMPLE (ESTIMATED)
      wind, two i16 (cm/s) 4 B | pressure anomaly i16 2 B | cloud cover u8 1 B
      precipitation rate u8 1 B | air temperature i16 2 B | humidity u8 1 B
      the instant it was stated, u32 4 B  (SL1 clause 6: a stale reading is refused)
      pad 1 B                                                          = 16 B

   THE WINDOW MUST COVER THE VISTA, NOT THE PILOT (refutation B D6, right)
      3 x 3 cells  =  30.8 km across  <  the 50 km vista: a front would appear
                      15.4 km away, which is a pop, and SL8 refuses it
      11 x 11 cells = 113 km across, covering a 50 km radius
         121 x 16 B = 1 936 B per update
         at 1 Hz  = 1.9 kB/s per player;  at 10 Hz = 19.4 kB/s per player

   THE CHEAPER SPLIT, and it is a design decision, not a rounding:
      the NEAR field (3 x 3) is live and crosses; the FAR field is drawn from the
      seed-derived CLIMATE, which both hosts already have. Then only 144 B crosses,
      and the far cloud is climatological rather than live.

   WHAT THE SHARD COMPUTES: an advection step over the cells inside its occupants'
      reach, NEVER over the planet (SL9: no per-tick walk of all children).
      ESTIMATED at about 200 ns per cell per step; 1 000 cells at 1 Hz is 0.2 ms/s.
```

**The client's part is style, not derivation.** The shard states the wind, the cover and the rate; the
client draws a cloud layer from a seed-derived noise ADVECTED by the stated wind at the stated instant.
That is the VU streaming contract's own shape — "a pose plus a bag of signals" — and it adds no new lane
if the samples ride the existing signal bag. **This is an SL6 ask, and it is D-C15.**

### 10.3 The blue haze, and the roads

The reference picture's depth read over 50 km is **aerial perspective**: distant ridges turn pale and
blue. That needs an atmosphere with a scale height, which is D-C9's third missing fact. **Until D-C9
lands, this domain cannot deliver the reference picture's depth cue**, however good the landforms are,
and no measurement in §13 can hide that.

The reference also shows a **road** through the valley — the strongest "somebody lives here" orienter,
and the owner's complaint was "no orienters". A road is authored or live, so it belongs to the edit
pyramid (`crates/terrain/src/compose.rs`, rank 2) and to a server skeleton, never to the seed. It is
domain 06's, and it is named here so nobody looks for it in a macro map.

### 10.4 The season — the field that is neither climate nor weather

**Revision 2 taught obliquity, asked the owner to draw it, and then had no place for a season**
(refutation B D4, right). Climate does not change with the tick. Weather is local and lives inside an
occupant's reach. A season is neither: it is planet-wide, it changes slowly, and it is a closed-form
function of `(seed, universe_tick)` — **Category A in CLAUDE.md's determinism rule, the same class as
the celestial math the client already reads the universe clock for.**

```
   THE SEASON, if the owner buys it (D-C19)
      one scalar per body per tick: the sub-stellar latitude, from the obliquity and
      the orbit's true anomaly, both closed-form on (seed, universe_tick)
      the column applies it as an offset to the climate's temperature and precipitation
      COST: 2 fenced operations per column  =  under 0.01 ms per chunk       ESTIMATED
      BYTES: none. Both hosts have the universe clock already.
      LAWFULNESS: it changes the LOOK (the snow line, the ice cap edge), never the
      SHAPE, so SL10's "never a function of time" for the static shape still holds.
```

**In the game's words.** The snow line on the home planet's ridge walks down the slope in winter and
back up in summer, and the server and the client agree on where it is, because both compute it from the
same tick and the same seed. Without it the ice cap the owner's obliquity buys has one extent for ever.

---

## 11. Component 7 — the no-drift legs

The gate today (`scripts/terrain_legs.sh`, `just terrain-legs`) runs the leaf's and the generator's
tests inside `rust:1.94.1-slim-bookworm` on two platforms:

```
   G1  aarch64 macOS, debug                       MEASURED, equal   (just terrain-pin)
   G2  aarch64 macOS, release, target-cpu=native  MEASURED, equal
   G3  aarch64 Linux (the k3d image's triple)     MEASURED, equal   (just terrain-legs)
   G4  x86-64 Linux, EMULATED on this Mac         MEASURED, equal   -- a smoke test only
   G5  the client binary against the server       LIVE since slice 7
   -- a REAL x86-64 machine is still owed: DEFERRED.md D-TERRAIN-1
```

**`terrain-legs` is NOT in `just gate`.** `justfile:408`'s `gate` recipe lists `terrain-pin`,
`terrain-link-scan` and `terrain-fence-control`; `terrain-legs` is at `justfile:756` with its own comment
saying it needs Docker. It is run by hand, and `DEFERRED.md` D-TERRAIN-1 records it as run once, on
2026-09-08.

> **The macro leg's size matters BECAUSE somebody must be willing to run it.** A leg that takes half an
> hour gets run once and then never, and SL10 clause 3 stops being measured. Whether the leg joins
> `gate` is D-C16; either way the cost below decides whether it is bearable.

**The new leg: a macro digest.** It must fold the WHOLE artifact, because a drift in one node of an
iterative field spreads to that node's whole basin by the next pass, and a sampled table would miss it.

```
   MACRO LEG = fold(the artifact's heights, receivers and discharge, in node order)  1 digest
             + fold(the climate grid)                                                1 digest
             + fold(the extracted channel polylines of three NAMED basins)           3 digests
```

| Leg content | native, one core | at 20x under emulation (ESTIMATED) |
|---|---|---|
| the home planet's solve (40 passes) plus the climate | 12.2 s | 4.1 min |
| the digest folds | about 30 ms | 0.6 s |
| **the macro leg, total** | **12.2 s** | **about 4.1 min** |
| a SMALL body instead: Ceres at 6 x 128² | 0.5 s | 10 s |

**The emulation factor of 20x is UNMEASURED.** M-L11 measures it by timing today's leg, which does a
known amount of work. At 5x the leg is 1 min; at 50x it is 10 min.

**A cheaper leg exists and it should be preferred**: fold a SMALL body of THE world (Ceres, 0.5 s
native) plus THREE NAMED basins of the home planet, rather than the whole home planet. It is the same
code on the same world, it needs no test-only body (SL5 holds), and it runs in ten seconds under
emulation. **D-C16 chooses.**

---

## 12. Determinism rules for an iterative field (each with a control)

SL10's clause-4 rules were written for a pure per-address function. An iterative field over a graph on
a bent grid adds failure modes the existing fence does not catch. Each needs a rule and a control that
goes red. **Each row carries an example in the game's words**, because these are the rules a later
implementer will break.

| # | Rule | Why | The control that must go red | In the game's words |
|---|---|---|---|---|
| **E1** | **Every pass count is a constant of the recipe, folded into `GENERATOR_VERSION`, never a convergence test.** That includes `PASSES`, `FLOOD_EVERY`, `ISOSTASY_EVERY`, `TALUS_PASSES` and the pointer-jumping round count. | A convergence test makes the cost data-dependent, so one host may stop a step earlier under a different scheduler. A pointer-jump count that is too small changes the ANSWER. | A build with a different count produces a different `declared_world_tag`, and a store written under the old tag REFUSES to open — the machinery `crates/terrain/src/tag.rs` already has. Plus a test that the longest basin on a named body needs no more rounds than the constant. | The pilot's saved cabin was written when the recipe said 40 passes. Somebody changes it to 41. Her store refuses to open and the world is rebuilt — instead of her cabin floating three metres above a valley that moved. |
| **E2** | **Every tie is broken by a total order.** Two neighbours with the same slope: the lower node index wins. The priority flood's heap key is **(height, node index)**. The donor enumeration order is fixed, because it fixes the stack order, which fixes the addition order E3 depends on. | `min` and `max` are already banned by the float fence. An unbroken tie makes the receiver — or the pop order — depend on the run. | A fixture with a flat plateau (every slope equal) and a fixture with a flat-bottomed crater (every flood key equal), both with pinned digests. | Two nodes on the home planet's mesa top are exactly level. Without the rule, one run sends the water north and the next sends it east, and the pilot's river moves between the server and her client. |
| **E3** | **Sums run in the basin's stack order — never with atomics, never in a reduction that depends on a thread count.** | Floating-point addition is not associative. | The same fixture with the recipe's band count set to 1 and to 8, asserting an EQUAL digest. | The pilot's whole valley drains to one river mouth. The area at that mouth is a sum of ten thousand nodes; add them in another order and the river is a hand's width wider on one host. |
| **E4** | **The state is quantised to whole millimetres after every pass.** It buys MEMORY and a BOUNDED state. **It does not buy determinism.** | An `i32` in millimetres halves the state against a `Gf`, and it stops forty passes from walking into a range where the `i16` metre read record overflows. | A test that the state after every pass is a whole number of millimetres and inside the `i16` metre range. | The home planet's peak is 14 304 m up. After forty passes of adding and subtracting, the number is still a whole millimetre and still fits the two bytes the chunk reads. |
| ~~E4, revision 1~~ | ~~"it stops any ulp difference from compounding"~~ | **DELETED, because it was false.** Rounding does not remove a one-ulp difference; near a rounding boundary it AMPLIFIES it to a whole millimetre, which then spreads through the node's basin. **What keeps two hosts equal is the float fence** (`crates/terrain/clippy.toml`, `Gf`, the link scan). | — | — |
| **E5** | **The stream power exponents are `m = 1/2` and `n = 1`**, so the law is `K · sqrt(A) · S`: one `sqrt` and multiplies. Every other exponent is a COMMITTED LITERAL table. | `powf` is libm, and the fence bans it by name. | The existing `terrain-fence-control` recipe, extended with a `powf` in the erosion kernel; clippy must go red. Plus a digest over each committed table. | A river on the home planet cuts as the square root of what drains through it. Nothing in that sentence needs a function the fence forbids. |
| **E6** | **If the refinement lands: a patch's address, its halo width and its pass count are derived from the CHUNK address by a fixed rule**, so a patch is a function of `(seed, address)`. | Otherwise a patch computed "around the player" is a function of where the player stood, which is live state, and SL10 breaks. | A test that the same chunk asks for the same patch id from two different observer positions and from two different arrival directions. | Two pilots land a kilometre apart. They get the SAME patch, not two patches centred on their own boots. |
| **E7** | **The neighbourhood is the CUBE NET's, not the array's.** A face edge rotates the axes; a cube corner node has SEVEN neighbours, never eight. | Without it a river stops at a face edge and a divide lands on a 5 263 km straight line (§5.4). | An exhaustive table test at a small `M` — the shape of HR4's G-MAPPING-TABLE at `N = 62` — asserting 8 neighbours for every node except the 8 corner nodes, which have 7, and that neighbour-of-neighbour returns home. | The pilot follows a river east. It crosses from the `+X` face to the `+Y` face and keeps going, because the nodes on both sides know each other. |
| **E8** | **The accumulation carries SQUARE METRES and the slope divides by the METRIC spacing**, both from the bend, never a node count and never an index distance. | MEASURED: the node-area ratio across a face is **1.425** (§5.4), so node counting is a 43 % error, biased toward the face edges. | A test that summing every node's area over the six faces equals `4πR²` to a stated relative bound (the replay above reached 1.2 x 10^-6). | Two rivers on the home planet drain the same land. One is near a face centre, one near a face edge. With the metric they are the same size. Without it one is a third bigger, and the pattern repeats six times around the planet. |
| **E9** | **If the refinement lands: mean-pinning is part of the pass, not a post-pass** — and its DENUDATION arm is settled first (D-C17). | It is what makes a coarse level the lawful coarse ANSWER of a fine one. Its unsettled half invents a raised shoulder (§5.6 C2). | A test that for every parent node the children's mean equals the parent's stated value to the millimetre, after every pass; plus M-L21's measurement of the invented shoulder in metres. | The pilot climbs out of her valley. At the rung where the level changes, the ground she sees is the average of the ground she walked on. |
| **E10** | **NEW. Every accumulator on the solve path is `Gf` or an exact integer. `f32` appears nowhere in the crate.** | The fence's own header ends its list of omissions with "and `f32` anywhere"; `clippy.toml` repeats it. But clippy and the link scan catch banned METHOD calls, and a plain `f32` add calls nothing, **so no control catches this today** (refutation B B1, right). | A new arm on `just terrain-link-scan`: the crate's source contains no `f32` token, and its object file exports no `f32` arithmetic symbol. Plus a `compile_fail` doc test beside `Gf`'s four. | The pilot's valley drains ten thousand nodes. If that sum were `f32`, the first host to round differently would give her a river of a different size, and nothing would go red. |
| **E11** | **NEW. Every transcendental is a COMMITTED LITERAL polynomial or table, with its maximum error stated in the source.** Never computed at build time, never fitted at run time. | The fence bans them all (§10.1), and a table computed by a build script would be computed with the very functions the fence forbids. | A digest over every committed table, plus M-L23's error measurement recorded beside it. | The home planet's ice cap edge sits where the insolation polynomial says it does. Both hosts read the same committed numbers, so they draw the same edge. |

**The coverage cost (HR5).** **Revision 2 called the kernel "straight-line monomorphic code over arrays",
which stopped being true when revision 2 itself added a heap and a pointer chase** (refutation A W7,
right). The honest statement:

- The sweeps ARE straight-line code over arrays: the easy case.
- **The priority flood is not.** A binary heap has sift-up and sift-down loops with data-dependent
  branches. Fixtures: an empty heap, a single element, a full re-heapify, a pop that walks to a leaf, and
  the `(height, node index)` tie arm.
- **The basin labelling is not.** Pointer jumping has a per-round convergence structure. Fixtures: a
  chain of length one, a chain of the maximum length, and a node that is its own root.
- The other deliberate fixtures: a closed pit, an all-sea body, a dry body, the face edge and the cube
  corner (E7), a node with no receiver, a glaciated node, a coastal node.
- **ESTIMATED cost: one afternoon of tests per pass, and two for the flood.** The RUNNING cost is real
  too: `just terrain-pin` runs three builds and coverage runs instrumented, which is why §8.2's fixture
  matters. Domain 03 makes the same point from the other side: a 200 km moon has 24 576 nodes and a 50 km
  body has fewer, **so a FULL solve runs inside a unit test in milliseconds, even instrumented.**

---

## 13. The measurement plan

**The owner's method: measurements first, on THE world, never a proxy.** Every measurement runs on the
home planet (seed 7 701 581 858 760 374 086), release, on an unloaded machine — the standing rule is
ONE JOB AT A TIME, and a measurement taken while a build runs is not a measurement. **Every bound below
reads §8.4's four constants, and no others.**

| # | Measurement | How | Pass bound | What it decides |
|---|---|---|---|---|
| **M-L0** | The baseline, re-taken | `just terrain-cost` on an idle machine | the numbers of §3 reproduce within 10 % | that every later delta is real |
| **M-L1** | **The pass schedule: nanoseconds per node, per PASS KIND**, at 6 x 64², 6 x 256² and 6 x 640², with the flood and the labelling in the loop | a new `landform_cost` example beside `terrain_cost` | **the schedule, and its spread across the three sizes.** The design passes if the home planet's lattice solves inside `SOLVE_TIME_CEILING` | every solve figure here; the per-body lattice of §5.1 |
| **M-L2** | **The pass count as an AGE**: the per-basin hypsometric integral against `PASSES`, **and the fraction of nodes whose BASIN LABEL changed in the last pass** | the same example, printing both distributions | the integral inside the published mature band 0.4–0.6; the label churn under 1 % | `PASSES` (E1), and whether the routing has settled |
| **M-L3** | **The solve's time and state**, on the home planet, on Luna and on an Earth-sized body | the same example | **at most 20 s on one core and 160 MB** after the chooser has walked | the per-body lattice; derive against ship |
| **M-L4** | **The cube seam, and (only if the refinement lands) the patch tile seam.** (i) the height at a shared column from each side. (ii) the extracted channel polyline crossing it | a test that builds both sides and compares the shared strip | **(i) MAX difference = ZERO quanta** — not a p99, because one column in the tail is one hole in the collider. **(ii) the channel enters within one node** | whether the field is lawful under SL8 at all |
| **M-L5** | **The per-chunk budget with the landform work on**: the costliest named chunk, box plus extraction, **and its vertex count** | `terrain_cost` extended with the macro read, the channel field and the climate, over the SAME named set | **under 8 ms**, the top rung still cheaper than rung 0, and the chunks-per-second fall stated | which components ship; whether bicubic survives; the descent's arrival rate |
| **M-L6** | **The river field against a segment search**: nanoseconds per column for each, the channel-node fraction, the DRAINAGE DENSITY in channel-km per km², and the rung at which the carve stops | the same example | **at most 100 ns per column**; density between 0.5 and 5 km per km² | the distance field against the search; the artifact's byte record |
| **M-L7** | **The client's first visit, TWO LEGS**: (i) an approach from outside 12 432 km; (ii) a COLD LOGIN standing on the home planet | the window flight (`scripts/client.sh --window`) with a timer | **no black frame on either leg; the ground refused, never wrong**; the wait stated in seconds | the derive-or-ship fork (D-C20); whether a cache is required |
| **M-L8** | The shipped size of the artifact, coded | a one-off slope predictor and entropy coder over the built artifact | at most 5 MB | D-C20's ship side |
| **M-L9** | **The coarse-rung law**: rung L's column pass strictly cheaper than rung L−1's, with the landform work on | the assertion `terrain_cost` already makes, kept | the existing assertion holds | that the ladder still coarsens for free |
| **M-L10** | **The climate passes**: milliseconds for the whole global climate grid, on the ERODED height | the same example | **at most 200 ms on one core** | that the climate stays global at 6 x 256² |
| **M-L11** | **The emulation factor**: the wall time of today's `terrain-legs` on `linux/amd64` against `linux/arm64` | time both legs | the factor, and the macro leg **at most 5 min** | whether the macro leg is bearable (D-C16) |
| **M-L12** | **No drift on the iterative field**: the macro digest equal on G1, G2, G3 and G4 | `scripts/terrain_legs.sh` with the macro leg added, plus a tampered-table control | every digest equal; the tampered control red | whether an iterative field may enter the generator at all |
| **M-L13** | **Believability, on a DRY planet.** Four statistics before and after the solve: the slope distribution's p95, the flat-area fraction (slope under 2°), the drainage density, and the hypsometric integral | a printout from the same example | **the flat fraction rises by at least half; the p95 slope rises; the hypsometric integral moves by at least 0.05** | whether the erosion is worth its cost AT ALL |
| **M-L14** | **The pop**: at a rung boundary, at a dropped river carve, and (if the refinement lands) at a level boundary, in pixels | the HR6 readback pop detector, at a walk and at 240 m/s | no visible step at the drawn-pixel floor (**1.1506 mrad: 57.5 m at the 50 km vista, 1.15 cm at 10 m**) | the crossfade band's width |
| **M-L15** | **The refinement's arrival rate**, if it is ever bought: patches per minute per pilot and per shard, at a walk, at 240 m/s and at the hull's rated cruise | the shipped flight with a counter | **the build share of one core under 10 %**, and no tick hitch on the shard | whether a patch may be built on a shard worker at all |
| **M-L16** | **Bodies inside macro range**: how many bodies are within `relief / 1.1506 mrad` of the pilot at once | a counter on the slice-7 flight | **at most 4** | whether §8.3's near gate holds, or the per-body cost multiplies |
| **M-L17** | **The cube net and the metric**: every node has 8 neighbours except the 8 corners (7); the six faces' node areas sum to `4πR²`; no channel terminates on a face edge | an exhaustive table test at small `M`, plus a channel-termination histogram | exact neighbour counts; area within 1e-5 relative; **zero channels terminating on a face edge** | E7 and E8; whether the cube shows |
| **M-L18** | **The extraction growth**: vertices per chunk before and after the landform work, on the named set | `terrain_cost`'s own printout | the ratio, and §7.1's estimate replaced | the real headroom under the 8 ms budget |
| **M-L19** | **NEW. THE VISTA CENSUS**: standing on the home planet looking at a 50 km horizon — chunks resident, total vertices, total bytes, and the wall time from arrival to a full vista, with the landform work off and on | the shipped flight with a counter and a timer | **the bytes under `MESH_RESIDENCY_CEILING`** (which D-C4 must first state) and the fill under 2 s at the client's real thread count | whether the reference picture is affordable at all |
| **M-L20** | **NEW. The refinement's arrival census**, if it is ever bought: each level's DETAIL AMPLITUDE in metres, and the patches resident and built against altitude | the same example, plus a counter on the flight | the amplitudes pinned per level in the recipe; the arrival under 2 s at the host's real thread count | §5.6's staging rule, and whether the refinement is affordable |
| **M-L21** | **NEW. Denudation and the invented shoulder**: the mean height change over the whole solve, and — under mean-pinning — the interfluve rise in metres | the same example, before and after | denudation strictly negative; the shoulder under the level's own detail amplitude | D-C17, and whether the erosion erodes at all |
| **M-L22** | **NEW. The biome edge width**: the distance in metres over which the forest-to-desert boundary crosses, at a range the pilot walks over | a transect printed from the climate and the artifact | **under 5 km**, because the reference picture's forest ENDS at a ridge | whether a 20 560 m climate grid can draw the picture's own edge |
| **M-L23** | **NEW. The fenced polynomials**: for each climate quantity, the degree, the maximum error over its argument's range, and the BIOME-BOUNDARY DISPLACEMENT that error causes | an offline fit against the true function, then a transect | **the displacement under one macro node (8 224 m)** | E11; whether the fence can carry the climate at all |

**The order is not free.** M-L1 gates M-L3, which gates M-L7. M-L17 gates M-L4, because a seam test on a
broken net measures nothing. M-L19 gates the whole picture question. **M-L13 gates everything**: if forty
passes of stream power do not move the slope distribution and the flat fraction on THE world's home
planet, the cheapest answer is §4.2's alternative — spend the milliseconds on octaves and on the biome
rule — and this whole domain closes.

---

## 14. The kill criteria

Each candidate is dropped at the number stated. The number is the decision. The argument is not.

| Candidate | KILL at | Then what |
|---|---|---|
| A GLOBAL solve at the vista's own resolution (shape A) | already dead: 13 007 s and 131 GB at 257 m (§5.5) | the 8 224 m solve (shape B) |
| **The global solve at 8 224 m (shape B)** | M-L3 over **20 s** or over **160 MB** on the home planet after the chooser has walked | the next coarser divisor: 6 x 320² at 16 448 m, 3.0 s, 32 MB — and the trunk valleys are 32 km apart, which reads as a lumpy planet (domain 03's D1(a)) |
| The macro solve at ALL | M-L13 shows no change in the flat fraction, the p95 slope or the hypsometric integral | §4.2's alternative: four more octaves (0.19 ms per chunk), a re-normalised roughness (free), and the biome rule — **priced, so this row lands somewhere** |
| The DEPOSITION and ICE passes | M-L1 shows them over **+40 %** of the schedule, or M-L13 shows the flat fraction unmoved without them | drop them, and state in the acceptance document that the flat valley floors and the U-shaped snowy valleys of the reference picture are NOT delivered |
| The channel DISTANCE FIELD | M-L6 over **150 ns per column**, or the artifact's growth to 19.7 MB fails D-C20's ship side | domain 03's closed-form `segment_distance_m` term seeded by the macro receiver chain |
| Bicubic macro reading | M-L5 over budget | bilinear plus one extra octave to hide the slope break, then re-run M-L14 |
| An iterative field in the generator at all | M-L12 shows one digest differing on any leg, and E1 to E11 do not close it | the landforms must be a closed-form function of `(seed, address)` — no erosion, and the vista is bought with shaped octaves alone |
| **The client DERIVING the artifact** | M-L7's cold-login leg over **20 s**, or the 128 MB peak fails a client budget | ship it: D-C20's option (ii), domain 03's own recommendation |
| **The whole domain's per-chunk cost** | M-L5's densest chunk over 8 ms — **and §7.1 already estimates 8.07 ms at twice the vertices** | ask the owner to raise the budget, with the measured number in hand. His own words, ruling V10: *"I think 8 should be ok. If that over time will become a problem, we can rethink and reimplement"* |
| **The vista** | M-L19's mesh bytes over the client's residency ceiling, or its fill over 2 s at the real thread count | fewer rings: draw the far ground at a coarser rung than the pixel floor allows, which is a detail-by-distance seam and must then be measured by M-L14 |
| The REFINEMENT (§5.6), if it is ever started | M-L20's arrival over **2 s** at the host's real thread count, or M-L15's build share over 10 % of one core, or M-L21's invented shoulder over the level's own amplitude | stay with the global solve alone, and buy the mid-scale with domain 03's closed-form terms |

---

## 15. How this passes each law gate

| Gate | How |
|---|---|
| **SL10 — one generator, two hosts, no drift** | The artifact is a pure function of `(seed, address)`: the seed draws the plates, the uplift and the climate; the address names the node; the pass counts are recipe constants folded into `GENERATOR_VERSION` (E1); the lattice is addressed by exact integer arithmetic (§5.1); **nothing reads time or live state, and the background build revision 1 offered is deleted** (§8.1). Both hosts run the SAME crate, so a port cannot exist. E1 to E11 close the failure modes an ITERATIVE field on a BENT grid adds, and M-L12 is the measurement on four legs. **Two parts are NOT derived: the weather, which §10.2 states as live state that crosses, and — if D-C20 chooses to ship — the artifact itself, which SL10 permits a realm to ship.** |
| **The seed ruling (2026-08-27)** | Everything derived here is SHAPE: heights, channels, biomes, moisture. All of it is safe to publish — a wiki of the home planet's rivers is a map, not a treasure map. **No deposit, no ore and no value is placed by the macro solve**; domain 03's facies decides sand, gravel and clay and refuses a placer by name. A block's substance is still not a pure function of `(position, seed)`. |
| **SL5 — one world** | The lattice is chosen once, offline, from the body's own `N`, and pinned in the recipe (§8.4). There is no scale knob, no preset and no test-only lattice. Every measurement in §13 runs on THE world. **TWO exceptions are named, not one** (refutation B N4, right): the golden self-check's fixture grid (§8.2) and E7's exhaustive small-`M` table. Both are test INPUTS, not worlds. |
| **No magic numbers** | Four constants only — the cell target and three ceilings — stated once in §8.4. **Two of the three ceilings are UNSOURCED and are put to the owner as D-C4**, and §8.4 states the tension rather than claiming a pass: a resolution chosen by a benchmark is neither seed-derived nor a per-entity property, so it is frozen once and recorded in the tag. **Where three physical facts do not exist (spin, obliquity, atmosphere) and two live in a crate the generator may not read (§10.1), that is D-C9, not papered over.** |
| **The 8 ms per-chunk budget** | §7.1: 4.10 to 4.34 ms for a surface chunk and **7.09 to 8.07 ms for the densest measured chunk, which is OVER the budget at the top of the range.** Reported as a fail, not a pass. M-L5 and M-L18 are the gates, and D-C8 is the owner's call. |
| **The ladder (V8, V9): rung L cheaper than rung 0, and its coarse answer** | §9.1: the rivers stop at rung 5, the caves at their own rungs, and the macro tile is one gather of at most 19 x 19 taps at every rung. §9.2: **`Z` is rung-independent, so there is no level boundary and no crossfade to design.** M-L9 keeps the existing assertion; M-L14 measures the pop that remains at the dropped carve. |
| **The record (12 bytes, the density byte, the biome/object param)** | The **PROPOSED** record (ruling V4 keeps it OPEN; today's cell is 2 bytes, `crates/terrain/src/chunk.rs`) is untouched. The macro artifact is an INPUT to the column pass, never a cell field. Nothing new is stored per cell. |
| **The edit pyramid as the authoring override** | Unchanged, and it sits ABOVE the macro artifact in the composition order (`crates/terrain/src/compose.rs`: the generated shape is rank 1, edits rank 2). A player who fills a river bed fills it. |
| **The collider on the same shape** | The artifact changes the HEIGHT the column pass produces, so the extracted surface changes with it, so the collider changes with it. Nothing forks. **This is why M-L4's bound is exact equality and not a tolerance, and why §7.4 refuses admission until the shape exists rather than admitting an occupant onto ground the shard has not built.** |
| **SL8 — seamless** | The pop risks are named and measured, never argued: the node interpolation (bicubic, M-L14), the cube face edge (M-L4, M-L17), the dropped river carve (§6.2, M-L14), the climate's arrival pop (fixed by the far gate, §8.3), and the weather front's window edge (§10.2). Each over tolerance kills its candidate (§14). |
| **HR5 — 100 % coverage in Tier-A** | §12's coverage note, **rewritten**: the sweeps are the easy case, the heap and the pointer chase are not, and each names its fixtures. A full solve of a small body of THE world runs inside a unit test in milliseconds even instrumented, which is what makes 100 % reachable. |
| **V4 — vegetation are art assets from a server skeleton** | The macro artifact supplies the biome, the moisture and domain 03's facies that the skeleton reads, and draws no tree, no grass and no primitive. **The skeleton's own per-chunk cost is domain 06's and is NOT priced here** — for a reference picture that is mostly forest, that is a line this document owes and does not carry. |
| **SL6 — ask before new data crosses** | **The landforms ask for nothing to cross IF the client derives them** (§8.1). **They DO cross if D-C20 chooses to ship**, at about 4 MB per body per player, and that is an SL6 ask. **The WEATHER crosses either way** (§10.2), at 1 936 bytes per player per update over a 50 km window, and it is an SL6 ask the owner must answer (D-C15). |
| **SL9 — a parent's child count is unbounded** | Nothing here walks a realm's children. The weather advection runs over the cells inside the occupants' reach, never over the planet (§10.2). |
| **HR4 — the geometry seam** | E7's cube-net neighbour rule is the macro lattice's own mapping, and it reuses `crates/seed/src/seam.rs`'s 24 directed edges. Its gate has the same shape as G-MAPPING-TABLE: exhaustive at a small `M` (M-L17). |

---

## 16. What you decide

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| **D-C1** | The macro map's shape. **THIS ANSWER CHANGED IN REVISION 3.** | A: one global solve at the vista's resolution. **B: ONE global solve at about 8 200 m, once ever per body, cached.** C: a pyramid of on-demand patches. | **B** | A costs 3.6 hours and 131 GB (§5.5). C costs an estimated 6.5 s per ARRIVAL, for ever, and its patches cannot be independent because flow is a global structure (§5.6). B costs 12.1 s ONCE per body and 9.83 MB. It is also what domain 03 designs, and a cost document should price the design that exists. |
| **D-C2** | The lattice per body. | (a) a constant node SIZE everywhere. (b) **a constant node size, with the chooser walking coarser until the ceilings hold.** | **(b)** | (a) gives an Earth-sized body 11.7 M nodes: 58 s and 608 MB, and a super-Earth 180 s and 1.86 GB (§5.1, MEASURED replay). (b) gives Earth 14 336 m and a super-Earth 28 672 m. **A big planet gets a coarser landscape, and you should know that before it ships.** |
| **D-C3** | The patch edge, IF the refinement is ever bought. | (i) pinned to the parent: exact, with a slope crease every 82 km. (ii) a partition of unity over four overlapping patches: exact AND creaseless, 4x the build. | **(i), pending M-L4** | (i) costs 0.62 s per patch, (ii) costs 2.48 s. If M-L4 shows the crease at the vista distance, (ii) is the answer. |
| **D-C4** | **The four constants.** | the cell target, the solve state ceiling, the solve time ceiling, **and the client's mesh residency ceiling, which this project has never stated** | **8 192 m, 160 MB, 20 s, and a mesh ceiling you must set** | The first is domain 03's. The next two are CHOSEN so the home planet fits, and revision 2 wrongly presented one as a fact about a pod. The fourth decides whether §7.5's **1.04 GB vista** is affordable, and no document has it. |
| ~~**D-C5**~~ | ~~Must every body have an exact coarse macro rung?~~ | — | **WITHDRAWN** | §5.1's family 2 (any whole divisor of `N`) removes the problem without touching the ladder, and the MEASURED per-body replay shows every body has a usable divisor. |
| **D-C6** | The seam tolerance M-L4 must meet. | a pixel tolerance; exact equality | **exact equality (ZERO quanta) on a shared column, plus the pixel floor for the CREASE** | A shared chunk edge is a MAX property: one column in a p99 tail is one hole in the collider. The pixel floor still governs what the eye sees: 57.5 m at the 50 km vista, 1.15 cm at the pilot's feet. |
| **D-C7** | May the client cache a built artifact on disk? | (i) no, rebuild every session. (ii) yes, keyed by the world identity **and a CONTENT digest**. | **(ii)** | It removes the 12.1 s from every session after the first. The content digest answers the objection that killed shipping: a cache is only lawful if a corrupted or edited one is REFUSED, and the store stamp checks the recipe version, not the bytes (refutation B N1, right). |
| **D-C8** | The 8 ms budget, now that §7.1 estimates 8.07 ms on the densest chunk. | keep it and shrink; raise it | **decide when M-L5 and M-L18 are in hand** | Your own words, ruling V10: *"I think 8 should be ok. If that over time will become a problem, we can rethink and reimplement"*. |
| **D-C9** | Spin, obliquity, atmosphere — and WHERE a body's physical facts are drawn. | (i) leave them out; the climate is latitude and height only. (ii) draw mass, radius, luminosity, orbit, spin, obliquity and atmosphere **inside `vd-terrain` from the seed**, and let `vd-physics` read THEM. | **(ii), before the first saved world** | Your task names them: biomes should depend on position, spin, trajectory, size and gravity. Three of the five do not exist, and two live in an UNFENCED crate the generator may not read (§10.1). It is an epoch. **It is also what makes D-C20's derive side possible at all.** |
| **D-C10** | The erosion's pass counts. | recipe constants (E1); a per-body draw from the seed | **recipe constants, set by MEASUREMENT** | Domain 03's D9 sets `PASSES` against the per-basin hypsometric integral and freezes it as the world's erosional age. A per-body draw makes the cost per planet unpredictable. |
| **D-C11** | May the generator use a thread pool? | (i) no: the kernel stays single-threaded and the HOST parallelises whole bodies. (ii) yes: adopt a pool inside `vd-terrain`. | **(i)** | `vd-terrain` has ONE dependency today and ships as a `staticlib` another engine links. (ii) is a new library, which is yours to approve, never ours to assume. **Note what (i) costs: the 12.1 s solve stays 12.1 s, because its passes are not independent.** |
| **D-C12** | When is a body's field built? | (i) one gate for everything. (ii) **TWO gates: the CLIMATE always, the ARTIFACT inside `relief / 1.1506 mrad`.** | **(ii)** | Colour is legible at one pixel and relief is not. Under one gate a planet 20 000 km away is a uniform ball that grows its deserts at 12 432 km — an arrival pop. The climate costs 0.08 s, so it is cheap to have always. |
| **D-C13** | What macro input does the golden self-check fold? | (i) the home planet's artifact: 12.1 s in EVERY process, three times in `terrain-pin`, and again under coverage. (ii) **a small named FIXTURE lattice, about 1 ms**, with the home planet folded into the macro LEG instead. | **(ii)** | (i) repeats the measured 5.3 GB-at-boot defect in a new place. (ii) keeps the identity's meaning: two binaries that would draw different ground still state different numbers. |
| **D-C14** | The epoch. | — | **The landform work bumps `GENERATOR_VERSION` and opens a world epoch**, because a macro solve moves every byte of every chunk of every body. So does any roughness change (§4.2). | It must land before the first saved world, exactly like D-C9. |
| **D-C15** | Weather crossing a realm boundary. | (i) no weather. (ii) a 3 x 3 window: 144 B per update, and a front pops into view 15 km away. (iii) **an 11 x 11 window covering the 50 km vista: 1 936 B per update.** (iv) the SPLIT: 3 x 3 live, the far field from the seed-derived climate. | **(iii) or (iv), and it is YOUR SL6 ask** | Your task asks for weather. The client may not derive it. (ii) is a pop SL8 refuses. (iii) is 1.9 kB/s per player at 1 Hz. (iv) is 144 B/s and a far sky that is climate, not weather. |
| **D-C16** | Should `terrain-legs` join `just gate`, and what does the macro leg fold? | (i) hand-run, folding the home planet (12.2 s native, about 4 min emulated). (ii) **in `gate`, folding a SMALL body of THE world plus three named basins (0.5 s native, about 10 s emulated).** | **(ii) once M-L11 measures the factor** | SL10 clause 3 is a MEASUREMENT, and a measurement nobody runs is not one. A ten-second leg gets run; a four-minute one does not. |
| **D-C17** | **NEW.** Mean-pinning against denudation, IF the refinement is bought. | (i) pin the mean and accept an invented shoulder of up to `f·h/(1−f)` — 129 m at f = 0.3, h = 300 m. (ii) pin the mean to the parent MINUS the level's own modelled denudation, and accept a stated per-level offset. | **(ii), pending M-L21** | (i) prints valleys with paired raised rims, which a geologist names on sight, and it makes the patch's hypsometric integral unable to move by construction. (ii) costs one extra field per level and one measured offset. |
| **D-C18** | **NEW.** Whose budget is the 8 ms? | (i) the server's. (ii) the client's. (iii) both. | **(iii), with the client's own measured baseline** | The constant lives in a server example, and the client's measured chunk is 4.18 ms against the server's 3.30 ms. A missed budget is an arrival rate on the server and a STALL on the client, and they are not the same failure. |
| **D-C19** | **NEW.** The season. | (i) none: one ice cap extent for ever. (ii) **a third field: the sub-stellar latitude, closed-form on `(seed, universe_tick)`, offsetting the climate's temperature and precipitation.** | **(ii)** | You asked for obliquity, and obliquity with no season is a word. (ii) costs two operations per column and ZERO bytes, because both hosts already read the universe clock for the celestial math. It changes the LOOK, never the shape, so SL10 holds. |
| **D-C20** | **NEW. Does the client DERIVE the artifact or RECEIVE it?** | (i) **derive** (this document's recommendation, and SL10's own shape). (ii) **receive** (domain 03's revision-2 recommendation). | **(i), and it depends on D-C9** | (i): no lane, no SL6 ask, the picture depends on the seed; costs 12.1 s and a 128 MB peak once per body, and a disk cache removes the repeat. (ii): no client compute; costs an SL6 ask, a wire arm and about 4 MB per body per player. **The two domains disagree, and this row exists so you settle it with both prices in hand.** |
| **D-C21** | **NEW.** Deposition and ice. | (i) stream power alone. (ii) **add a transport/deposition term and an ice pass.** | **(ii)** | (i) cannot make a flat alluvial valley floor or a U-shaped snowy valley, and the reference picture is made of both. (ii) costs +30 ns per node per sweep and +30 ns per node once — **+33 % on the schedule**, so the home planet's solve goes from 9.1 s to 12.1 s (DERIVED: 3 700 ns against 4 929 ns per node). |

---

## 17. Open questions

1. **Does erosion actually change the picture on THE world?** M-L13 is the honest test, and it must run
   before one more line of landform design. The pass bound asks for four statistics a DRY planet can
   move, because the home planet's sea covers about one column in a hundred (MEASURED, slice 5), so it
   can never grow Earth's second hypsometric hump.
2. **Where does the sea come from after erosion?** Today the sea radius is a seed draw between −40 % and
   +30 % of the relief (`crates/terrain/src/body.rs:186-190`), and the home planet drew about −0.37, so
   it is nearly dry. Erosion needs the sea as its base level, and a believable planet needs the sea to
   cover a believable fraction. **This is the same question as item 1.**
3. **How many passes does the solve need, and is the routing settled?** M-L2 measures both the
   hypsometric integral and the basin-label churn, because a landscape whose height has stopped moving
   may still be re-organising its rivers.
4. **What is the emulation factor?** Every leg figure rests on a guessed 20x. M-L11 measures it in ten
   minutes of wall time, and nobody has spent them.
5. **Does a 20 560 m climate grid draw the picture's own biome edge?** The reference shows a forest that
   ENDS at a ridge. M-L22 measures the edge's width, and if it is 40 km wide the climate needs refining
   where the artifact already is (refutation A W8, right that no level refines the climate).
6. **Can the fence carry the climate?** Every insolation and moisture quantity is naturally
   transcendental, and the fence bans every one. M-L23 measures the polynomial error AS A BIOME
   DISPLACEMENT, which is the quantity that matters.
7. **Can a hull or a station hold a macro artifact?** Ruling V4 says a hull may hold terrain. A hull has
   no seed (`ShipLocal` carries an entity id, `crates/core/src/pose.rs`), so it has no artifact, and its
   terrain must come from saved blocks. Two paths produce ground by different means and somebody must
   check the seam where they meet.
8. **Where does the artifact live in the workspace?** It is a per-body field with a build phase and a
   CACHE, which is a different shape from the pure per-address functions `vd-terrain` holds today.
   **The recommendation: `vd-terrain` stays pure and exposes `solve(seed, body) -> Artifact`; the HOST
   owns the cache, its residency and its eviction** (refutation A W9, right that a cache inside a Tier-A
   `staticlib` is new and undesigned). That keeps 100 % coverage reachable and keeps the crate linkable
   into another engine.
9. **What does the vegetation skeleton cost per chunk?** The reference picture is mostly forest, and
   ruling V4 makes vegetation art assets placed by a server skeleton from the biome. This document
   prices the biome the skeleton reads and not the skeleton. Domain 06 owes the line.
10. **Five octaves or six inside the solved field?** §5.3's Nyquist rule says five on an 8 224 m lattice;
    `03_erosion_rivers.md:418` says six. The difference is the 12.5 km octave and its 171 m of amplitude,
    and one of the two documents must give way before either is built.

---

## 18. What is UNMEASURED

| # | Unmeasured | How it would be measured | Consequence if it is wrong |
|---|---|---|---|
| C1 | **The pass schedule's cost per node.** Every solve number is that schedule times a node count. | M-L1 | A 2x error moves the home planet's lattice one divisor step, to 16 448 m. |
| C2 | The pass count the world needs, and whether the routing has settled. | M-L2 | 80 passes doubles every solve figure. |
| C3 | The channel-node fraction and the drainage density. | M-L6 | The artifact's memory, and whether the vista shows a river network at all. |
| C4 | Whether a distance field on an 8 224 m lattice can place a 30 m channel at all, against domain 03's closed-form term. | M-L6 | The river design, and 9.9 MB of artifact. |
| C5 | The macro tap gather's real cost on a cold cache. | M-L5 | It is assumed to be one gather per chunk. If the tile does not stay in L1, it becomes per column. |
| C6 | The compressed size of the artifact (about 0.5 B per node assumed). | M-L8 | D-C20's ship side, and the SL6 ask's size. |
| C7 | The emulation factor on `linux/amd64` (20x assumed). | M-L11 | The macro leg's bearability, and D-C16. |
| C8 | The per-body parallel factor if D-C11 ever grants a pool (8.4x borrowed from chunk jobs). | M-L1 at 1 and at 14 threads | Only the host-side arrival rate; no ceiling in this document depends on it. |
| C9 | Whether erosion moves the four statistics of M-L13 on THE world. | M-L13 | The existence of this whole domain. |
| C10 | The cube-face-edge behaviour of the drainage (E7 is designed, never run). | M-L17, M-L4 | A ring of dead drainage and a 5 263 km straight ridge per face edge. |
| C11 | The 1 ns per fenced operation figure. | folded into M-L1's composite | Only the op-count estimates of §6.1 and §10.1 rest on it alone. |
| C12 | Whether a client's disk cache needs a CONTENT digest as well as a version stamp. | a design read of `crates/core/src/store_stamp.rs` | A corrupted or edited cache would draw a world the server does not have. |
| C13 | **The extraction growth factor** (1.5x to 2.0x vertices assumed). | M-L18 | The whole headroom: the densest chunk is estimated at 7.09 to **8.07 ms** against an 8 ms budget. |
| C14 | **The vista's chunk count, bytes and fill time**, and the client's own residency ceiling. | M-L19, D-C4 | Whether the reference picture is affordable at all. 1.04 GB is DERIVED from a ring model, not counted. |
| C15 | **The detail amplitudes of the refinement's levels** (60 m, 300 m, 1 500 m assumed). | M-L20 | The refinement's staging rule, and its 6.5 s arrival. |
| C16 | **The invented shoulder under mean-pinning**, and the solve's true denudation. | M-L21 | D-C17, and whether a refined landscape looks embossed. |
| C17 | **The fenced polynomials' error, as a biome displacement.** | M-L23 | The ice cap's edge and the forest's edge — the picture itself. |
| C18 | The weather's per-tick cost and its real byte rate. | belongs with D-C15 | The size of the SL6 ask. |
| C19 | The deposition and ice passes' true cost (+30 % of the schedule assumed). | M-L1 with them in and out | D-C21, and whether the picture's valley floors exist. |

---

## 19. The recommendation in one paragraph

Erode each body ONCE, on one global cube-sphere lattice of about 8 200 metres per node — for the home
planet 640 nodes per face edge, 8 224 m, 2 457 600 nodes, which divides its own `N = 5 263 360` exactly
so every fine column names its node with integer arithmetic — and keep the answer: **12.1 s on one core,
127.8 MB of solve state, 9.83 MB kept, once ever per body, cached.** That is domain 03's own design, and
my independent operation model lands on its own 12-to-20-second estimate, which is worth stating even
though two models are not a measurement. The same lattice at 257 m would cost **3.6 hours and 131 GB**,
so a fine global solve is dead for ever; a pyramid of on-demand patches, which the previous revision
recommended, costs an estimated **6.5 s per arrival, for ever**, invents a raised shoulder beside every
valley it cuts, and cannot make its patches independent at any halo width, because an implicit sweep
carries a base-level change across a whole basin in ONE pass — so refinement is a later slice, priced
here, under the global solve and never instead of it. The solved field **REPLACES** the coarse octaves
rather than adding to them, which is the rule the previous revision never stated and its sibling had
already decided; the fine rungs gather their macro taps once per chunk, so the added work per column is
arithmetic: **0.45 ms per chunk**, plus an extraction that grows with the roughness, taking a rung-0
surface chunk from a MEASURED 3.30 ms to **4.10–4.34 ms** and the densest measured chunk from 5.22 ms to
**7.09–8.07 ms against the owner's 8 ms — a fail at the top of the range, reported as a fail.** Eleven
determinism rules cover an iterative field on a bent grid, and two are new: every accumulator is `Gf` or
an integer, because the fence bans `f32` by name and no control catches a plain `f32` add today; and
every transcendental is a committed literal polynomial whose error must be measured **as a biome
displacement**, because the fence forbids the sine the insolation is made of. Twenty-four measurements
carry their pass bounds and their kill numbers, and five decide everything: the schedule's true cost, the
cube seam at ZERO quanta, whether erosion moves a DRY planet's slope and flatness at all, whether the
client derives the artifact or receives it, and **the vista census — about 4 200 chunks, 16.3 s of
one-core work and 1.04 GB of mesh for the owner's own 50 km picture, against which the 9.83 MB artifact
is one per cent.** **Four things this domain does not deliver, and says so: the band below the macro
node, where the home planet carries a measured 15.15 metres of relief across five octaves and where the
reference picture's cliffs and rock pillars live; the flat alluvial valley floor and the U-shaped snowy
valley, unless the deposition and ice passes are bought at +30 % of the schedule; the blue haze, which
needs an atmosphere no body in the code has; and the weather, which is live state, may not be derived on
the client, and crosses at 1 936 bytes per player per update over a 50 km window — an SL6 ask the owner
must answer.**

---

## 20. Refutation answers

Every finding from `verdicts/07_measurements_costs_refutation_a.md` and
`verdicts/07_measurements_costs_refutation_b.md` (both against revision 2) is answered. **FIXED** means
the section was rewritten, not annotated. **KEPT** means the refuter is wrong, with the evidence beside
it. **OWED** means it is real, outside this domain's reach, and it names who carries it.

### Refutation A

| # | Finding | Severity | Answer |
|---|---|---|---|
| A-B1 | The composition rule is never stated, and the sibling states the opposite one | Blocker | **FIXED.** New §5.7 states domain 03's own rule — **`Z` REPLACES the coarse octaves** — shows what an addition would have done (the home planet's relief doubling to about 26 000 m), and re-derives §7.1's column arithmetic against it: five octaves removed, four reads added, the saving booked at ZERO because 11.5 ns is an upper bound. |
| A-B2 | The arrival cost counts one patch per level; a descent needs tens | Blocker | **FIXED, and it changed the recommendation.** §5.6 carries the arrival census: about 10.5 patches, **6.5 s per arrival, for ever**, under a corrected staging rule (a level's own DETAIL AMPLITUDE, not its node, subtends a pixel). Against 12.1 s ONCE per body for the global solve, the pyramid loses, so §16 D-C1 now recommends the global solve and defers the refinement. New M-L20 measures the census and the amplitudes. |
| A-B3 | A level-3 patch does not tile the level-2 parent grid (256 / 5 = 51.2) | Blocker | **FIXED.** §5.6's geometry is 320 kept, 40 halo, 400 built — every one a whole multiple of both ratios (5 and 8) — and a level-3 patch then spans exactly one level-0 node, so a patch is addressed by naming a parent node. The halo waste falls from 56 % to 36 %. |
| A-B4 | The kernel priced is not the kernel the design needs, and the sibling's grid is never priced | Blocker | **FIXED, and it is the largest change in revision 3.** §4.1 replaces one number with a SCHEDULE that carries the deposition, the isostatic rebound, the talus and the ice passes domain 03 names. §5.3 prices domain 03's own lattice — 640 nodes, 8 224 m, 40 passes: **12.1 s and 127.8 MB**, against its own independent estimate of 12–20 s and 125 MB. §5.5 makes it shape B and **recommends it**. |
| A-D1 | The drawable floor is 1.1506 mrad in the code, not 1.0908 | Defect | **FIXED.** §3 quotes `crates/core/src/geometry.rs:1164-1173` and its own worked example (a one-metre extent at 869 m, the "~870 m" the reach ruling states). Every staging distance, and M-L14's pixel conversion (57.5 m at the 50 km vista), is re-derived. |
| A-D2 | §3's cave-dense row mixes two source rows, and §7.1's headroom rests on the result | Defect | **FIXED.** §3 keeps slice 6's two cave-dense rows apart (23 084 vertices at 2.39 ms, and 8 234 at 1.73 ms) and replaces "75 ns per vertex" with a fitted model, **0.973 ms + 61.4 ns per vertex**, whose two under-predictions are printed beside it. §7.1 prices the growth at BOTH marginal costs. |
| A-D3 | The Nyquist rule is mis-applied three times, and it is not the rule `octaves_at` implements | Defect | **FIXED.** §5.3 states plainly that `body.rs:252-258` drops ONE octave per rung — about 49 times the cell, not twice it — so the macro rule is NEW, not a generalisation. On the 8 224 m lattice it keeps octaves 1 to 5, and the one-octave disagreement with domain 03 is §17 item 10. |
| A-D4 | The patch iteration count breaks the document's own iteration law | Defect | **FIXED, by retiring the law.** §5.3 shows the law was written for an EXPLICIT sweep. Domain 03's solver is IMPLICIT (`03_erosion_rivers.md:816`), and an implicit sweep in stack order carries a base-level change to every headwater in ONE pass. So the pass count is the landscape's **erosional age**, not a convergence radius. The refuter is right that revision 2 could not have it both ways; the resolution is that neither number was the right one. |
| A-D5 | "Any whole divisor" does not give a usable pyramid on every body (SL5) | Defect | **FIXED with a MEASUREMENT.** §5.1 carries a `python3` replay of `Ladder::for_radius` over all seven body radii of `01_grid_family.md:302-309`: every body has a usable divisor, and **an Earth-sized body at a constant node size would cost 58 s and 608 MB**, so the chooser must walk coarser. That is now D-C2, and the owner is told that a big planet gets a coarser landscape. |
| A-D6 | The coarse-answer law is proven on the build record and consumed from the read record | Defect | **FIXED by the design change, with the arithmetic shown.** §5.2: a one-metre quantum on an 8 224 m node is a slope quantum of **0.012 %**, against the 781 m octave's 1.1 %. The problem was made by revision 2's 257 m grid, where the same quantum was 0.39 %. If the refinement is ever bought, its read record is `i32` millimetres (+2 B), and §5.6 C4 prices it. |
| A-D7 | The rate table fires its own kill row, and the text says the opposite | Defect | **FIXED.** §7.4 states that the global solve has NO rate (it is once ever, cached), and that the refinement's rate table **fails the 10 % kill row above about 4 500 m/s at low altitude**. The table now carries an "is that speed possible at that altitude?" column, because at 30 000 m/s every chunk in view is drawn at a rung that reads a coarser field. |
| A-D8 | The arrival case has no lead, so the collider's shape does not exist | Defect | **FIXED.** §7.4 states the rule: **an occupant is not ADMITTED to a body's realm until the artifact is loaded and, if the refinement is bought, the patch under its own address is built.** Admission waits; it never admits and then builds. That covers the login, the transfer from a hull, the warp arrival and the kill-9 restart. A refusal is lawful under SL8; a fall through the world is not. |
| A-D9 | The op counts for the flood and the labelling are optimistic and internally inconsistent | Defect | **FIXED, all three points.** §4.1 charges the flood over the WHOLE grid, at `2·log2(n)` compares (29 at 24 576 nodes, 35 at 2.46 M), and §5.4 states the pointer-jumping round count as `ceil(log2 1 280) = 11`, with E1's control testing that the constant is large enough on a named body. The "doubling is conservative" claim is gone: §4.1 prices a schedule, and M-L1 replaces it. |
| A-W1 | Mean-pinning conserves mass, so the erosion removes nothing; isostasy is defined and never used | Weakness | **FIXED, both halves.** §5.6 C2 states the invented shoulder with its arithmetic (`f·h/(1−f)` = 129 m at f = 0.3, h = 300 m), names it as an artefact rather than a feature, and makes it D-C17 with M-L21 measuring it. Isostasy is now a PASS in §4.1's schedule (every fifth sweep, 12 ns per node) and in §5.4's pass table, following domain 03's `ISOSTASY_EVERY = 5`. |
| A-W2 | §2's tectonic-uplift example is wrong by about four orders of magnitude | Weakness | **FIXED.** §5.3 states the rate as `relief / PASSES ≈ 350 m` per pass on the home planet, and §2's example says "about a hundred metres each pass, and after forty passes a range stands there, which the rivers then cut back down". |
| A-W3 | The stream-power example promises deposition the chosen law cannot produce | Weakness | **FIXED.** §4.1 adds a **transport/deposition term at 30 ns per node per sweep**, §5.4 puts it in the pass table in the same stack order, §2 explains detachment-limited against transport-limited and alluvium, and D-C21 puts the pass to the owner with its price. |
| A-W4 | §8.3's rule reads relief only, so a planet loses its continents before it loses its bumps | Weakness | **FIXED.** §8.3 now has TWO gates: the CLIMATE always (colour is legible at one pixel), the ARTIFACT inside `relief / 1.1506 mrad` = 12 432 km. D-C12 records it. The climate costs 0.08 s and 1.57 MB, so it is cheap to have always. |
| A-W5 | `BOOT_CEILING = 2 s` has no source | Weakness | **FIXED.** §8.4 states the constants as CHOSEN and owner-owed, marks two of them **UNSOURCED**, and drops the "a pod is not promised more" claim, which cited a section about a thread pool. D-C4 carries all four, including the mesh residency ceiling this project has never stated. |
| A-W6 | The world's shape becomes a function of a benchmark | Weakness | **FIXED.** §8.4 states the tension instead of claiming a pass: the schedule cost is an operational number that MOVES THE WORLD, so it is frozen once, pinned per body and recorded in the tag, and no host evaluates it at run time. §15's no-magic-numbers row says the same. |
| A-W7 | §12 calls the kernel "straight-line code over arrays" after adding a heap and a pointer chase | Weakness | **FIXED.** §12's coverage note is rewritten: the sweeps are the easy case, **the heap and the pointer chase are not**, and each names its own fixtures (an empty heap, a single element, a full re-heapify, a pop that walks to a leaf, the tie arm; a chain of length one, a maximal chain, a self-root). The running cost and domain 03's "a full solve of a small body runs inside a unit test" are both stated. |
| A-W8 | The climate stays global at 20 560 m, which cannot resolve the picture's own biome edge | Weakness | **FIXED in part, OWED in part.** FIXED: §10.1 moves the climate's orography onto the ERODED height and re-solves it every tenth pass, so the shadow is on the right range (this also answers B-D3). OWED: no level refines the climate, so the edge's width is bounded by the 20 560 m grid, and **new M-L22 measures it with a pass bound of 5 km.** If it fails, the climate must be refined where the artifact already is. |
| A-W9 | The patch cache is nowhere designed, and it is the crate's first mutable state | Weakness | **FIXED.** §17 item 8 answers it: **`vd-terrain` stays pure and exposes `solve(seed, body) -> Artifact`; the HOST owns the cache, its residency and its eviction.** That keeps 100 % branch coverage reachable and keeps the crate linkable as a `staticlib` into another engine. |
| A-N1 | §9.1 mixes chunk counts inside one block | Note | **FIXED.** §9.1 prints the visible half (5 292 chunks, 12.3 s) and the whole-body byte figure (10 584 chunks, 2.03 GB) as two separate calculations. |
| A-N2 | §8.1 prints a patch as 0.26 MB in a column headed 2 B | Note | **FIXED.** §8.1's table is rewritten around the artifact's 4 B record, and every cell is derived from it. |
| A-N3 | §5.5 and §8.1 price the same two grids at 1.25 MB and 0.84 MB | Note | **FIXED.** One record (4 B), one number (9.83 MB), used everywhere. |
| A-N4 | §10.1 derives its base height from a rung-11 THREE-octave figure | Note | **FIXED.** §10.1's row states plainly that the 69 ns figure is a three-octave measurement and that the row is therefore an UNDER-count. |
| A-N5 | §5.5 prices shape A at `I = 100` while §5.3's law gives `I = 2M` | Note | **FIXED.** The diameter law is retired (A-D4), and shape A is priced at the same 40 passes as every other row: 13 007 s and 131 GB. |
| A-N6 | §10.2 cites "SL10 c.3" for the client not deriving | Note | **FIXED.** §10.2 cites clause 7 ("the client never derives a pose, a velocity or any state") and names clause 3 as the no-drift MEASUREMENT. |
| A-N7 | The band diagram ends the octave table at 30 m | Note | **FIXED.** §4.2's table says the loop's bound is `while wave_m > 30`, so the finest octave the home planet draws is **48.8 m**, and the 49 m to 30 m band carries nothing either. |
| A-N8 | §5.5 calls the climate "5 passes"; §10.1 lists six | Note | **FIXED.** It is six, and §10.1's total row says so. |
| A-N9 | The chooser's "coarsest first / take the finest" describes opposite scans | Note | **FIXED.** §8.4: the chooser LISTS the whole divisors, takes the one nearest the target, and steps to the next COARSER divisor while either ceiling fails. |
| A-N10 | Terms used before they are explained | Note | **FIXED.** §2 adds pointer jumping, topological order, donor, stencil, bicubic and bilinear, Nyquist, denudation, detachment-limited and transport-limited, alluvium, glacial forms, implicit sweep, erosional age, minimax polynomial and season. |

### Refutation B

| # | Finding | Severity | Answer |
|---|---|---|---|
| B-B1 | The build record uses `f32`, and the fence bans `f32` on this path | Blocker | **FIXED.** §5.2's record carries `Gf` and exact integers only (39 B → 40 B), and the finding's real half — that **no control catches a plain `f32` add**, because clippy and the link scan catch banned METHOD calls — becomes **rule E10**, with a new link-scan arm (no `f32` token in the crate, no `f32` arithmetic symbol in the object file) and a `compile_fail` beside `Gf`'s four. |
| B-B2 | The patch's iteration count contradicts the document's own law, and the halo is sized from the wrong one | Blocker | **FIXED, and the refuter's deeper point is adopted.** §5.3 retires the diameter law for an IMPLICIT sweep. The refuter's own argument — that flow routing is non-local and `C3` proves it — is exactly right, and §5.6 C1 now states the consequence the refuter did not: **no halo is ever wide enough, at any width**, so a patch must IMPORT its routing rather than recompute it. That is one of the three reasons the pyramid is demoted to a deferred refinement. M-L2 also reports the basin-label churn, as asked. |
| B-B3 | There is no deposition pass, and no ice; the reference picture is made of both | Blocker | **FIXED.** §4.1 adds the deposition term (+30 ns per node per sweep) and the ice pass (+30 ns per node once), §5.4 puts both in the pass table, §2 explains alluvium and the glacial forms, **§1's honest-limits list grows from two entries to four**, and D-C21 puts the two passes to the owner with the +30 % they cost. |
| B-B4 | The document that owns cost never prices THE VISTA | Blocker | **FIXED.** New §7.5: about **4 200 chunks, 16.3 s of one-core work, 1.94 s at slice 7's measured 8.4x, and 1.04 GB of mesh**, with the client's own 4.18 ms baseline used for the client column. New **M-L19** is the census the refuter asked for, and D-C4 adds the mesh residency ceiling this project has never stated. |
| B-D1 | The 2.0x cave-dense row is arithmetically wrong, and the corrected number is over budget | Defect | **FIXED, and reported as a FAIL.** §3's corrected extraction model and §7.1's table give the densest measured chunk **7.09 to 8.07 ms** at twice the vertices — over the 8 ms budget at the top of the range. §1, §15 and §14 all say so instead of reporting a pass. |
| B-D2 | The budget is applied to the server's chunk cost; the client's measured cost is higher | Defect | **FIXED.** §3 carries both measured rows, §7.5 uses the client's 4.18 ms for the client column, and **D-C18** asks the owner whether the 8 ms governs the server, the client or both — because a missed budget is an arrival rate on one and a stall on the other. |
| B-D3 | The climate's ranges are not the erosion's ranges | Defect | **FIXED.** §10.1: the climate reads the ARTIFACT's height and is re-solved every tenth pass (`CLIMATE_EVERY = 10`), so the rain carves the range and the range steers the rain. The residual — a shadow finer than 8 224 m is not resolved — is stated, and M-L22 measures the edge. |
| B-D4 | Obliquity is taught, and nothing in the design makes a season | Defect | **FIXED.** New §10.4 defines the season as a third field: the sub-stellar latitude, closed-form on `(seed, universe_tick)`, Category A, **two operations per column and zero bytes**, changing the LOOK and never the shape. D-C19 puts it to the owner. |
| B-D5 | Arriving on a surface is not staged, and it costs 2.86 s | Defect | **FIXED.** §5.6 states that from a surface every level is inside its own range at once, so nothing spreads; §8.1 states the cold-login wait honestly (12.1 s derived, about 3 s shipped, nothing from a cache); **M-L7 now flies TWO legs**, an approach and a cold ground login. |
| B-D6 | The weather window is smaller than the vista, so the byte count is understated | Defect | **FIXED.** §10.2 prices an 11 x 11 window covering a 50 km radius: **1 936 B per update, 1.9 kB/s at 1 Hz, 19.4 kB/s at 10 Hz**, thirteen times revision 2's figure. The cheaper SPLIT (a live near field, a climatological far field) is stated as a design decision, and D-C15 carries all four options. |
| B-D7 | §8.3's two rules disagree, and the climate is covered by neither | Defect | **FIXED.** §8.3 is rewritten as TWO gates, and the climate rides the FAR one for exactly the refuter's reason: colour is legible at one pixel and shape is not. D-C12 records it. |
| B-D8 | Mean-pinning invents a raised shoulder beside every valley | Defect | **FIXED.** §5.6 C2 carries the refuter's own arithmetic (129 m at f = 0.3, h = 300 m), states that a real interfluve never sits above the pre-incision surface, states the second consequence (a patch's hypsometric integral cannot move by construction, so M-L13's fourth statistic is measurable only on the global solve), and makes the fork **D-C17** with **M-L21** measuring it. |
| B-D9 | The fence bans every transcendental, and the climate is built from transcendentals | Defect | **FIXED.** §10.1 carries a table of every climate quantity, its natural form, its fenced replacement and its error — all marked **UNSTATED** — plus **rule E11** (committed literal polynomials and tables, never fitted at run time) and **M-L23**, whose pass bound is the BIOME-BOUNDARY DISPLACEMENT in kilometres, because the error is the finding and not the cost. |
| B-W1 | The read record is `i16` metres and C4 claims millimetres; the cure is cheap and unpriced | Weakness | **FIXED.** §5.2 shows the quantum is 0.012 % on an 8 224 m node against 0.39 % on revision 2's 257 m one, so the global solve does not need the cure; and if the refinement lands, §5.6 C4 states the `i32` millimetre record and prices its +2 B per node. |
| B-W2 | Isostasy is taught in §2 and appears in no pass | Weakness | **FIXED.** Same as A-W1: it is a pass in §4.1's schedule and in §5.4's table, at 12 ns per node every fifth sweep. |
| B-W3 | `BOOT_CEILING = 2 s` is presented as a fact and is unsourced | Weakness | **FIXED.** Same as A-W5: §8.4 marks it CHOSEN and UNSOURCED and hands it to D-C4. |
| B-W4 | §4.2's band table contradicts the rest of the document about what this domain owns | Weakness | **FIXED.** §4.2's table now says **400 km – 8.2 km, the macro solve, this domain**, with a separate row for the deferred refinement's 8.2 km – 257 m, and the diagram matches. |
| B-W5 | "257 m, the resolution the reference picture's valley needs" is never derived | Weakness | **FIXED by removing the claim and deriving what replaces it.** 257 m is no longer the recommendation. Where a resolution is claimed, §5.6 derives it: a valley 1 to 2 km wide needs at least four nodes across its cross-section, so 250 to 500 m — which is what the deferred refinement's finest level would be, and it is a result rather than a convenient divisor. |
| B-W6 | The 160 ns kernel is size-independent, and the priority flood is not | Weakness | **FIXED.** §4.1 prices the flood at four sizes (111, 137, 155 and 215 ns per node), and §5.3's table carries the size-dependent figure rather than one number. M-L1 measures at three sizes. |
| B-N1 | The client disk cache is the objection used to kill shape B | Note | **FIXED.** §8.1 states the derive-and-ship comparison honestly instead of killing shipping with one line, and **D-C7 now requires a CONTENT digest** as well as the version stamp, which is the refuter's own cure. |
| B-N2 | Shape A is priced at `I = 100` while the document's own law says `I = 2M` | Note | **FIXED.** Same as A-N5: the law is retired and shape A is priced at 40 passes. |
| B-N3 | Three quantities in four bytes, with no quantisation stated | Note | **FIXED.** The metric leaves the per-node record entirely: it is a function of the face coordinate, so a row scan holds one ROW of it — `O(M)`, not `O(M²)`. |
| B-N4 | Two test-only grids, and §15's SL5 row names one | Note | **FIXED.** §15's SL5 row names both: the golden self-check's fixture and E7's exhaustive small-`M` table. |
| B-N5 | The word "halo" carries two sizes 64x apart | Note | **FIXED.** §2's halo row states it: the extractor's `HALO` is ONE cell because a stencil reaches one cell; a patch's is forty because an iterative field reaches further. |

### What both refuters got right that changed the design, in one list

1. The cost document was pricing a design its own sibling had already replaced. **The recommendation is
   now the global solve at 8 224 m, once ever per body, cached.**
2. An implicit sweep carries a base-level change across a whole basin in ONE pass, so the "iterations
   grow with the diameter" law is wrong — and the same fact means **no patch halo is ever wide enough**.
3. The composition rule is a REPLACEMENT, not an addition, and nobody had written it down.
4. The vista — the owner's actual question — had never been multiplied out: **4 200 chunks and 1.04 GB**.
5. The kernel was missing deposition, ice and isostasy, and the picture is made of the first two.
6. Mean-pinning conserves mass, so a refinement invents a raised shoulder beside every valley.
7. The fence bans every transcendental the climate is made of, and the error moves a biome boundary.
8. `f32` is banned by name and no control catches it.
9. The drawable floor in the code is 1.1506 mrad, and the extraction curve came from two mixed rows.





