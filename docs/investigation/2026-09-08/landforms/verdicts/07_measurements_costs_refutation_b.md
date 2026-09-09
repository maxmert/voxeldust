# Refutation B of `07_measurements_costs.md` (revision 2)

**Date:** 2026-09-08. **Refuter:** b. **Lens:** believability, cost, and the owner's own question.
**Method:** I read revision 2 in full, then the code it cites (`crates/terrain`, `crates/seed`,
`crates/bins/examples/terrain_cost.rs`, `justfile`), then slice 5, 6 and 7 for the measured rows.
**Verdict in one line:** the arithmetic of the pyramid is much better than revision 1, but the
document still hides three costs (the patch's true iteration count, the vista, the client's own
per-chunk figure), it specifies a build record the float fence forbids, and it cannot make the
landforms the reference picture is actually made of.

Every number below shows its arithmetic. Where I estimate, I write ESTIMATED and I name the run that
would replace the estimate.

---

## Blockers

### B1 — The build record uses `f32`, and the float fence bans `f32` on this path

`§5.2` states the erosion's working state as, per cell:

```
   flow accumulation, f32 square metres   4 B
```

and `§5.4` builds a determinism rule on it: *"floating addition is not associative, so the order IS
the answer."*

The code refuses the type. `crates/terrain/src/gf.rs`'s own header lists what the fence deliberately
omits and ends the list with **"and `f32` anywhere"**. `crates/terrain/clippy.toml` carries the same
words in a lint reason: *"SL10 clause 4: no f32 on the seed-shaped path at all"*. `Gf` is the only
float the generator may compute with, it is `f64`, and it never gives its number away
(`gf.rs`, the four `compile_fail` controls).

So the accumulation must be `Gf` (8 B) or an exact integer (`i64` square millimetres, 8 B). Both add
4 B to every build cell:

```
   build record            29 B  ->  33 B          (+13.8 %)
   a 384² patch's state   4.28 MB -> 4.87 MB
   level 0's state        0.71 MB -> 0.81 MB
   shape A's state          73 GB  ->  83 GB
```

The cost change is small. The finding is not the bytes; it is that **the document's central data
structure was written without reading the fence it claims to pass**, and `§15`'s SL10 row asserts the
design passes. An `f32` accumulator would also escape the gate: the link scan and clippy catch banned
METHOD calls, and a plain `f32` add calls nothing. So this is a rule a reviewer must carry, not a rule
a control carries. State the accumulator's type, and name the control that goes red if somebody writes
`f32`.

**In the game's words.** The pilot's valley drains ten thousand cells. If that sum is `f32` on one
host, the fence's own control never fires, and the first host to round differently gives her a river
of a different size — the exact failure `Gf` exists to make impossible.

---

### B2 — The patch's iteration count contradicts the document's own iteration law, and the halo is sized from the wrong one

`§5.3` states the law and calls it physical:

> *"The iteration count is not a constant of the world; it scales with the grid's DIAMETER in cells,
> because the erosion signal travels from an outlet to a divide one cell per iteration."* — `I = c · M`
> with `c` of order 2.

Level 0 obeys it: `M = 64`, `I = 128`.

A patch does not. `§5.5` gives every patch `I = 32` while it builds 384 cells per edge, and defends it:

> *"A patch is a REFINEMENT: its long wavelengths are already solved by its parent and are held fixed.
> Only the detail INSIDE a parent cell has to grade, and that detail spans five to eight child cells."*

That argument is true of the HEIGHT detail. It is false of the FLOW ROUTING, and the flow routing is
what the passes actually compute. `§5.4`'s pass list rebuilds the receiver graph, the priority flood,
the basin labels, the stack and the accumulation over **all 147 456 cells, every iteration**. A
channel network is a non-local structure: which way a fine cell drains can change which basin it
belongs to, and that change must travel to the patch's outlet. The document's own `C3` (the flow seed)
proves the non-locality — it exists because the incoming drainage area cannot be computed locally.

Apply the document's own law to a patch:

```
   I = 2M with M = 384        ->  I = 768
   147 456 cells x 768 x 160 ns  =  18.11 s per patch      (one core)
   three levels                  =  54.3 s per arrival
   even I = 2 x 256 (the KEPT edge) = 512 -> 12.08 s per patch, 36.2 s per arrival
```

against the document's headline of **0.76 s per patch and 2.28 s per arrival**. That is a factor of
24. `§17` item 3 admits only *"if M-L2 says 128, the arrival cost triples"*; the document's own law
says 768, not 128, and no kill row covers it — `§14`'s patch row kills at 64 iterations and falls back
to a 128² patch, which by the same law needs `I = 2·192 = 384`, not 32.

**The halo is welded to the same number.** `§5.5` builds a 64-cell halo. A halo is wide enough only if
it is at least as wide as the distance an edge's influence travels, which is `I` cells. At `I = 32` a
64-cell halo is exactly 2×, which is why it looks right. At `I = 768` the halo must be 768 cells, the
patch becomes `(256 + 1536)² = 3 211 264` cells, and shape C is dead — it becomes shape A with extra
steps.

**The other half of the same fact, and it is a believability finding.** At `I = 32` the fine field can
move information only 32 cells = **8.2 km at level 3**, while the patch is **65.8 km** across. So the
level-3 channel network is not decided by the fine physics at all; it is the parent's routing
upsampled, with an 8 km-wide fringe of fine adjustment. In the game's words: the pilot's river is drawn
by the 1 285 m map and merely roughened by the 257 m one. A geomorphologist reading M-L6's drainage
density would get the parent's density, not the fine one.

**What the document must do.** State `I` for a patch from the same law it states for level 0, or state
a different law and defend it. Then re-price shape C. M-L2 is not enough: it measures convergence of
the HEIGHT change, and `§13`'s bound is *"the p99 per-iteration height change under a millimetre"* — a
routing that is still re-organising can sit under a millimetre of height change per iteration for a
long time. **M-L2 must also report the fraction of cells whose BASIN LABEL changed in the last
iteration**, because that is the quantity the argument is about.

---

### B3 — There is no deposition pass, and no ice; the reference picture is made of both

`§5.4`'s pass table is the complete list of what the erosion does:

```
   base height, metric, plate layout and uplift, receivers and slope,
   priority flood, basin labelling, stack build, flow accumulation,
   incision, hillslope diffusion, mean-pinning, quantise, digest
```

**Nothing in that list puts material down.** Incision removes; diffusion moves rock one cell downslope;
the flood fills a pit with an artificial surface, not with sediment. A pure detachment-limited
stream-power model erodes everywhere and never builds an alluvial fan, a floodplain, a terrace, a delta
or a valley fill.

Look at the reference picture. Its middle ground is **a flat, wide, green valley floor with a river
meandering across it and fields on it**. That floor is alluvium. A stream-power-only landscape gives a
V-notch with the channel at the vertex and no floor to put a field on. The document even promises the
landform it cannot make: `§2`'s stream-power row says *"A flat cell with the same catchment lays a
floodplain instead."* No pass lays one.

And the picture's headline is **snow-capped ridges above carved valleys**. Those are glacial forms — a
U-shaped cross-section, cirques at the heads, arêtes between them, a hanging valley where a tributary
joins. Stream power makes none of them. The document mentions snow only as albedo (`§2`) and never as
an eroding agent. A geologist looking at a snow-capped range with a sharp V-notch under it would say
the picture is wrong, and would say it in one glance.

**The cost this omits.** An erosion–deposition term (a transport-limited law, or a `phi`-style
erosion–deposition rule) adds one more pass down the same stack — ESTIMATED **+25 to 40 ns per cell per
iteration** on `§4.1`'s own op scale (one flux carried down the stack plus one settling term). That is
**+16 % to +25 % on the 160 ns kernel**: level 0 goes from 0.50 s to 0.58–0.63 s, and a patch from
0.76 s to 0.88–0.95 s. A glacial pass (an ice-thickness field from the climate's temperature plus a
sliding-law abrasion term) is a second field and a second loop, ESTIMATED at the same order again, and
only on cells above the snow line.

**What the document must do.** Either add the two passes and re-price, or state plainly — the way it
already states the 30 m–257 m band — that **the flat valley floors, the fields and the U-shaped snowy
valleys of the reference picture are NOT delivered by this domain**, and name who carries them. Today
`§1`'s "two honest limits" list has two entries and needs four.

---

### B4 — The document that owns cost never prices THE VISTA

The owner's question is about a picture: *"detail to the horizon over ~50 km"*. `§7.1` prices ONE
chunk. `§7.4` prices ONE patch against travel. **No section anywhere multiplies a per-chunk cost by the
number of chunks the reference vista contains**, so the document cannot answer the question it was
written for.

Here is the arithmetic it owes. ESTIMATED; every assumption is named.

```
   Assumptions
     - a 90 degree horizontal field of view over flat-ish ground
     - the ladder picks the rung whose cell subtends about one drawn pixel:
       cell = 1.0908 mrad x d          (the reach ruling's floor, the document's own §8.3)
     - a chunk is 62 cells (ladder.rs:21), so a chunk edge = 0.0676 x d
       and a chunk footprint = 0.00457 x d^2
     - a ring from d to 2d in a 90 degree wedge covers about 2.36 x d^2

   chunks in one ring       = 2.36 / 0.00457   =  about 516
   rings from 100 m to 51.2 km (nine doublings) = about 4 600 chunks

   worker time to fill the vista once, at §7.1's NEW 4.08 ms
      4 600 x 4.08 ms  =  18.8 s on ONE core
      at slice 7's MEASURED 8.4x on fourteen threads  =  2.2 s

   resident mesh bytes, from slice 7 M7-3 (405 228 B at 8 577 vertices = 47 B per vertex)
      near chunks about 8 500 vertices  -> 400 KB
      far chunks about 4 055 vertices (slice 6, rung 11) -> 192 KB
      take 250 KB as an average:  4 600 x 250 KB  =  about 1.15 GB
```

Three consequences the document never states:

1. **The vista's fill is 18.8 s on one core.** The document's own seam bound is 2 s (`§5.3`: *"more
   than about 2 s of stall before the first drawn ground is a seam under SL8"*). The pyramid's 0.58 s
   boot is the small part of that stall; the chunks are the big part, and only the client's thread pool
   (slice 7 measured 8.4×) brings it to 2.2 s.
2. **The landform work's 24 % chunk-cost rise is 24 % of 18.8 s, not 24 % of 4.08 ms.** `§7.1` notes
   *"a descent fills its vista 24 % more slowly"* and leaves it as a sentence. It is +3.6 s of one-core
   work per vista, or +0.4 s at 8.4×.
3. **About 1.15 GB of mesh is resident for one vista.** That number decides whether the reference
   picture is affordable at all, and it appears nowhere in a document whose title is "the cost model".

**The measurement this needs, and it is missing from `§13`.** Add **M-L19, the vista census**: on the
shipped flight, standing on the home planet's surface and looking at a 50 km horizon, count the chunks
resident, their total vertices, their total bytes, and the wall time from arrival to a full vista, with
the landform work off and on. Pass bound: the bytes under the client's own resident ceiling (which the
document also never states) and the fill under 2 s at the client's real thread count.

---

## Defects

### D1 — The 2.0× cave-dense row is arithmetically wrong, and the corrected number is OVER budget

`§7.1` and `§1` both print:

```
   NEW, a cave-dense chunk     7.83 ms  ESTIMATED  (2.0x)
   THE BUDGET                  8.00 ms
```

Back out the increment: `6.11 + 0.40 + X = 7.83`, so `X = 1.32 ms`.

Now do it from the document's own MEASURED per-vertex figure. `§3`: the cave-dense chunk is 23 084
vertices and 1.73 ms of extraction = **75 ns per vertex**.

```
   at 2.0x the vertices:  46 168 x 75 ns  =  3.46 ms
   the increment          3.46 - 1.73     =  +1.73 ms      (not +1.32)
   the chunk              6.11 + 0.40 + 1.73  =  8.24 ms   OVER the 8 ms budget
```

And `§3` itself says the per-vertex figure *"settles near 100 ns"* as the count rises. At 100 ns:

```
   46 168 x 100 ns = 4.62 ms;  increment +2.89 ms;  the chunk = 9.40 ms
```

So under the document's own two measured per-vertex numbers the 2.0× case is **8.24 ms or 9.40 ms**,
not 7.83 ms. `§1`'s headline *"The budget survives; the headroom does not"* is false at the upper end
of the document's own estimate range. The 1.5× row checks out (23 084 × 1.5 = 34 626 × 75 ns = 2.60 ms,
increment +0.87, total 7.38 against the printed 7.41), which makes the 2.0× row look like an
unexplained substitution of the SURFACE chunk's 1.32 ms extraction time as the cave-dense chunk's
increment.

**Consequence.** `§14`'s last kill row ("ask the owner to raise the budget") fires at 2.0×, and the
document should say so at the top instead of reporting a pass.

### D2 — The budget is applied to the SERVER's chunk cost; the CLIENT's measured cost is higher

`§3` carries both numbers and `§7.1` uses only one:

| | measured | source |
|---|---|---|
| server, rung-0 surface chunk: box + extraction | **3.30 ms** | slice 6 |
| client, rung-0 chunk: box + extraction + positions + normals | **4.18 ms** | slice 7 M7-2 |

The client pays a measured **+0.88 ms** per rung-0 chunk, because SL10 makes it build the same shape
and then it must also make positions and normals. Every figure in `§7.1` is built on 3.30 ms.

```
   the client, surface chunk, with the landform work
      4.18 + 0.40 + 0.21..0.38   =   4.79 to 4.96 ms
   the client, cave-dense chunk (positions and normals scale with the vertex count,
      so +0.88 ms is a floor, not the figure)
      6.11 + 0.88 + 0.40 + 1.73  =   9.12 ms       OVER the 8 ms budget
```

The budget constant lives in a SERVER example (`crates/bins/examples/terrain_cost.rs:41`). The document
never says whether the owner's 8 ms is a server budget, a client budget, or both. **It must, because a
missed budget is a stall on the client and an arrival rate on the server.** M-L5 measures
`terrain_cost`, which is the server path only.

### D3 — The climate's ranges are not the erosion's ranges

`§10.1` builds the climate on a global grid of 6 × 256² (20 560 m per cell) and takes its orography
from *"The base height at 20 560 m (octaves 1–5)"* — the un-eroded octave field.

The erosion's ranges do not live there:

```
   level 0 (global)   82 240 m per cell   too coarse to hold a range that casts a shadow
   level 1 (a PATCH)  10 280 m per cell   exists only where somebody stands
   level 2 (a PATCH)   1 285 m per cell   exists only where somebody stands
   level 3 (a PATCH)     257 m per cell   exists only where somebody stands
```

So the rain shadow is computed against a range the erosion has not carved, at a place the erosion may
have moved. `§10.1`'s own justification — *"finer than the erosion's level 0 because a rain shadow
needs a range resolved"* — resolves the wrong range.

**In the game's words.** The pilot walks over the range she can see. The desert is on the far side of a
DIFFERENT range, 40 km away, that the octave field put there before the rivers moved it. She reads the
wind, walks to the lee, and finds grassland.

The loop is also broken physically: real orographic rain feeds the rivers that carve the range, and the
carved range then steers the rain. This climate is one-way AND attached to a different height field.

**The cheap repair, and it should be priced.** Run the climate on level 0's ERODED height (82 240 m)
and apply the fine correction per column from the level the chunk already reads (the lapse rate is
already per column). Then the shadow is on the right hill at the scale the global grid can hold. State
the residual error.

### D4 — Obliquity is taught, and nothing in the design makes a season

`§2` teaches obliquity as *"the tilt of a body's spin axis against its orbit. It makes seasons, and it
sets the size of the ice caps."* `§16` D-C9 asks the owner to draw spin, obliquity and atmosphere,
*"because your task names them"*.

Then the design has no place for a season:

- **Climate** is defined in `§2` and `§10.1` as *"the long-run AVERAGE ... It does not change with the
  tick"*, and it is seed-derived.
- **Weather** is `§10.2`'s per-tick field of wind, cloud, rain and temperature, computed by the shard
  *"over the cells inside its occupants' reach, NEVER over the planet"*.

A season is neither. It is planet-wide, it changes with time slowly, and it is a closed-form function
of `(seed, universe_tick)` — Category A in CLAUDE.md's determinism rule, the same class as the
celestial math. It cannot be climate (climate does not change with the tick) and it cannot be weather
(weather is local and lives inside a reach). So the ice cap the owner's obliquity buys has one fixed
extent forever, and the snow line on the reference picture's ridge never moves.

**The decision this needs.** A third field — a seasonal modulation both hosts derive from
`(seed, universe_tick)`, offsetting the climate's temperature and precipitation by latitude. It is
lawful under SL10 only if the client may read the universe tick, which it already does for the
celestial math. **The document must name it, decide it, or refuse it.** Today `§16` has sixteen
decision rows and none of them is the season.

### D5 — Arriving on a surface is not staged, and it costs 2.86 s

`§5.5` and `§8.3`:

```
   BOOT          = 0.58 s   (level 0 + the climate)
   FIRST ARRIVAL = 2.28 s more, staged by DISTANCE   (levels 1, 2, 3)
```

Staging works for an APPROACH: level 1 inside 9 424 km, level 2 inside 1 178 km, level 3 inside 236 km,
so a descending pilot pays them at different times.

It does not work for the two cases that matter most:

1. **A login on the surface.** The standing rule is that login gets no bespoke path. A pilot who logs
   out on the home planet's highland and logs in there needs level 0, the climate and all three patches
   at once: `0.58 + 2.28 = 2.86 s` on one core — **over the document's own `BOOT_CEILING = 2 s`** and
   over M-L7's *"at most 2 s to the first ground"*.
2. **A crossing into the planet realm at close range.** A hull already inside 236 km when the pilot's
   client attaches pays the same 2.86 s.

Neither case appears in `§8.3`, in M-L7 (which flies `scripts/client.sh --window`), or in `§14`. The
distances make it worse, not better: the home planet's radius is 3 351 km, so from any point on its
surface the whole body is inside level 1's 9 424 km trigger and most of the visible ground is inside
level 2's 1 178 km. **From the surface, every level is always needed.** Staging by distance only helps
somebody who arrives from space, slowly.

M-L7 must fly two legs: an approach from outside 9 424 km, and a cold login on the ground.

### D6 — The weather window is smaller than the vista, so the byte count is understated

`§10.2` sizes the weather crossing as *"her own macro cell and its eight neighbours = 9 samples =
144 B per update"* at level 1's 10 280 m cell.

```
   3 x 3 cells at 10 280 m  =  30.84 km across
   the reference picture's vista  =  50 km
```

The pilot sees 19 km beyond the last cell she is told about, and cloud and rain in the far distance are
the most visible weather in the reference picture. A front entering the window would appear at 15.4 km
from her — a pop, which SL8 refuses.

```
   to cover a 50 km radius at 10 280 m needs 11 x 11 = 121 samples
      121 x 16 B  =  1 936 B per update      13.4x the stated 144 B
      at 1 Hz  = 1.9 kB/s per player;  at 10 Hz = 19.4 kB/s per player
```

19 kB/s per player is a real lane cost and it changes the shape of D-C15's SL6 ask. Either state the
bigger window, or state that the far field is drawn from the seed-derived climate and only the near
field is live — which is a design decision, not a rounding.

### D7 — `§8.3`'s two rules disagree about when a body's map exists, and the climate is not covered

`§8.3` gives two rules and never reconciles them:

| Rule | Says |
|---|---|
| the level table | level 0 is needed closer than **75 400 km**, level 1 inside 9 424 km |
| D-C12 | a body's macro map is built only when its RELIEF subtends a pixel — **13 100 km** for the home planet |

Between 75 400 km and 13 100 km a rung wants level 0 and D-C12 has not built it. `§9.2`'s fallback rule
(*"never a level whose patch does not exist. Level 0 always exists"*) assumes level 0 exists; under
D-C12 it does not.

The gap matters because `§9.1` makes level 0 the source of the above-the-top-rung PROXY: *"a displaced
sphere plus a normal map baked from 24 576 texels"*. So a planet approached from 40 000 km is drawn
from a level the design has not built.

**And the climate is in neither rule.** The climate grid (1.18 MB, 6 × 256²) is what gives a planet its
COLOUR from a window — the deserts, the ice caps, the vegetation belts. Colour is not sub-pixel at
75 400 km; the planet is 178 pixels across there and its continents are visible. If the climate rides
D-C12's 13 100 km gate, a planet approached from far away is a plain ball that suddenly grows
continents. `§8.3`'s defence — *"Beyond that a planet is a smooth ball, and still 469 pixels across"* —
argues about SHAPE and forgets colour.

Decide it: the climate is cheap (0.08 s, 1.18 MB) and belongs at the FAR gate, while the erosion levels
ride the near one.

### D8 — Mean-pinning invents a raised shoulder beside every valley

`§5.6` C2 forces each parent block's child mean back to the parent's height *"after every iteration"*,
and `§2` gives the example: *"Level 2 says the pilot's basin averages 1 200 m. Level 3 puts a 300 m
valley and a 300 m shoulder inside it, and their mean is still 1 200 m."*

The second half of that sentence is the artefact. Real erosion is not zero-mean at any scale: it is net
denudation. A refinement that carves a valley and is then forced to hold the block mean must put the
removed material back on the interfluves.

```
   valley occupies a fraction f of the parent block, cut to depth h
   the rest of the block must rise by   f x h / (1 - f)
   at f = 0.30 and h = 300 m  ->  129 m of INVENTED shoulder
```

Repeated on every parent cell, that prints a landscape of valleys with paired raised rims — an embossed
look, and a look a geologist names on sight, because real interfluves sit near the pre-incision
surface, never above it.

There is a second consequence the measurement plan trips over: **a patch's hypsometric integral cannot
move, by construction.** Mean-pinning fixes the mean; the hypsometric integral is the mean as a
fraction of the range. So M-L13's fourth statistic can only ever be measured at level 0, and `§13` does
not say so.

The repair is standard: pin the block mean to the parent's height MINUS the parent's own modelled
denudation for this level, or let the mean drift and re-pin the PARENT instead. The document must state
which, because C2 is the clause that buys the ladder's coarse-answer law.

### D9 — The fence bans every transcendental, and the climate is built from transcendentals

`crates/terrain/clippy.toml` bans, by name, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
`exp`, `ln`, `log`, `powf`, `powi`, `cbrt`, `hypot`, every hyperbolic, `to_radians`, `to_degrees`,
`mul_add`, `min`, `max` — for `f64`; and `f32` is absent from `Gf` entirely.

`§10.1` prices the climate's insolation as *"about 30 (a polynomial; no `exp`, no `pow`)"* and says
nothing more. Every quantity in that table is naturally transcendental:

| Quantity | Its natural form | Under the fence |
|---|---|---|
| daily-mean insolation by latitude and obliquity | `sin`, `cos`, `acos` of the hour angle | a fenced polynomial, error UNSTATED |
| the Coriolis parameter | `2 Ω sin(latitude)` | a fenced polynomial, error UNSTATED |
| saturation vapour pressure (what "moisture" means) | Clausius–Clapeyron, an exponential | a fenced polynomial, error UNSTATED |
| the atmosphere's density with height (`§10.3`'s blue haze) | `exp(−h/H)` | a fenced polynomial, error UNSTATED |

Each replacement is possible. None is free of error, and **the error moves a biome boundary**, which is
the picture. A 1 % error in the insolation polynomial near the pole moves the ice cap's edge by degrees
of latitude.

State, for each climate quantity: the polynomial's degree, its maximum error over the argument's range,
and the biome-boundary displacement that error causes. `§4.1`'s 1 ns per fenced operation makes that
cheap to price — a degree-7 polynomial is about 14 operations — but the ERROR is the finding, not the
cost. `§15`'s "no magic numbers" row and its SL10 row both pass this over.

---

## Weaknesses

### W1 — The read record is `i16` METRES, and C4 claims millimetres; the cure is cheap and unpriced

`§5.2`'s read record is *"height, `i16` metres — 2 B"*. `§5.6` C4 says *"A chunk on either side reads
the same integer millimetre."* Those differ by a factor of 1 000. The seam property survives (both
sides read the same `i16`), but the QUANTUM matters:

```
   1 m on a 257 m cell  =  a slope quantum of 0.39 %
   the 781 m octave's own characteristic slope  =  about 1.1 %
   the extractor's own quantum (ruling S5-2, tolerance ZERO)  =  8 mm
```

`§18` C15 names this and leaves it UNMEASURED. What the document never does is **price the cure**,
although the cure fits its own ceiling:

```
   read record i32 millimetres instead of i16 metres: +2 B per cell
      levels 1 and 2 kept:  65 536 x 5 B  =  0.328 MB each   (was 0.197)
      level 3 kept:         65 536 x 9 B  =  0.590 MB        (was 0.459)
      the worst-case pilot: 0.07 + 1.18 + 4x0.328 + 4x0.328 + 4x0.590 = 6.23 MB
      READ_CEILING = 8 MB   ->  it FITS
```

A finding that a table row can close should be closed, not deferred to a measurement.

### W2 — Isostasy is taught in `§2` and appears in no pass

`§2` teaches isostasy with a game example: *"The rivers cut a valley on the home planet. The block is
lighter, so the block rises, so the peaks beside the valley stand higher."* `§5.4`'s pass table has no
isostasy pass, `§4.1`'s op count has no isostasy line, and no measurement tests for it.

The owner is taught a term the design does not implement. Either add the pass — a smoothed load times a
compensation factor, ESTIMATED 15 ns per cell per iteration, +9 % on the kernel — and price it, or move
the row out of `§2` and say it is not modelled. As written, a reader believes the peaks rise.

### W3 — `BOOT_CEILING = 2 s` is presented as a fact and is unsourced

`§5.3` and `§8.4` justify the ceiling with *"a pod is not promised more (§4.3)"*. `§4.3` is about the
absence of a thread pool in `vd-terrain`; it says nothing about a pod's CPU request or limit, and no
k3d manifest is cited anywhere. The standing rule is that such a claim is a measurement or it is marked
UNMEASURED.

The number is put to the owner as D-C4, which is right. The JUSTIFICATION should say "chosen, and your
call" rather than borrow the authority of a fact.

### W4 — `§4.2`'s band table contradicts the rest of the document about what this domain owns

`§4.2`'s diagram runs the macro pyramid from 400 km down to 257 m. The table one line below says:

| Band | Owner |
|---|---|
| 82 km – 1.3 km | **the macro pyramid, this domain** |

Level 3 (257 m) is a whole level below 1.3 km, and `§1`, `§5.5`, `§9.2` and `§19` all build it. One
line of one table understates the domain's own reach by a factor of five. It matters because that table
is what the owner reads to learn who owns which scale.

### W5 — "257 m, the resolution the reference picture's valley needs" is never derived

`§1` states it twice as a fact and `§5.5` prices shape A against it. Nothing derives it. The divisor
list in `§5.1` shows where it came from: `N = 2^12 × 5 × 257`, so 20 480 cells per face is the
convenient whole divisor that lands near 257 m. The number came from arithmetic convenience and was
then labelled as the picture's requirement.

A derivation exists and is short: a valley the pilot can walk needs a cross-section resolved by at
least four cells; the reference picture's valleys are 1 to 2 km wide; therefore 250 to 500 m per cell.
**Write that**, and 257 m becomes a result rather than a coincidence.

### W6 — The 160 ns kernel is size-independent, and the priority flood is not

`§4.1` charges the priority flood 55 ns for *"about 19 compares"*. A binary heap's compare count per
push-pop pair is about `2·log2(n)`:

```
   level 0,  n =    24 576  ->  log2 = 14.6  ->  about 29 compares
   a patch,  n =   147 456  ->  log2 = 17.2  ->  about 34 compares
   shape A,  n = 2.5 x 10^9 ->  log2 = 31.2  ->  about 62 compares
```

So the flood alone varies by more than 2× across the sizes the document prices with ONE number. `§4.1`
acknowledges the memory side of this (*"the estimate is optimistic for the big grids and pessimistic
for the small ones, which is the wrong way round for safety"*) and does not carry it into the tables.
M-L1 measures at three sizes, which is right; the tables should then carry three kernel numbers, not
one.

---

## Notes

### N1 — The client disk cache is the objection the document used to kill shape B

`§8.1` kills shape B partly because it would give *"a client picture that depends on a transfer instead
of on the seed"*. D-C7 then recommends the client cache a built level on disk, keyed by the world
identity. A disk cache is a picture that depends on a stored artefact instead of on the seed, and the
stamp (`crates/core/src/store_stamp.rs`) checks the recipe VERSION, not the content, so a corrupted or
edited cache passes it. Reconcile the two positions in one sentence, and give the cache a content
digest.

### N2 — Shape A is priced at `I = 100` while the document's own law says `I = 2M`

`§5.5` prices shape A at 6 × 20 480² with `I = 100` = 40 265 s. The document's own law (`§5.3`) gives
`I = 2M = 40 960`, which is 4.6 × 10^6 s — 53 days. The kill stands either way, so this weakens only
the internal consistency; but `§5.3`'s table marks the `I = 2M` cell of that row as "—", which reads as
"not applicable" rather than "unaffordably large".

### N3 — Three quantities in four bytes, with no quantisation stated

`§5.2`'s build record has *"the metric (cell area and two spacings) 4 B"*. A cell area at level 0 is
6.76 × 10^9 m², and the two spacings are metres. Three quantities do not fit in 4 B without a stated
quantisation, and a quantised metric is exactly the rounding E8 exists to fix. State the packing, or
make the row 12 B (build state 29 → 37 B, a patch 4.28 → 5.46 MB).

### N4 — Two test-only grids, and `§15`'s SL5 row names one

`§15`'s SL5 row names the golden self-check's fixture as *"the one exception"*. M-L17 and E7 also need
*"an exhaustive table test at small `M`"*. That is a second test-only grid. Both are lawful — they are
test inputs, not worlds — but the row should say "two", so the exception list stays honest.

### N5 — The word "halo" carries two sizes 64× apart

`§2` says the patch's halo is *"the same word as the extractor's halo
(`crates/terrain/src/lattice.rs:48`)"*. The code's `HALO` is **1** cell; the patch's is 64. The document
is not wrong, and it does say "the same word", but a reader who greps `HALO` after reading `§2` finds a
different number. One clause removes the trap: the extractor's halo is one cell, a patch's is
sixty-four, because an iterative field's edge reaches further than a stencil's.

---

## Checked, and sound

Verified against the code or the arithmetic. These need no second look.

1. `N = 5 263 360 = 2^12 × 5 × 257` for the home planet, and 64, 512, 4 096 and 20 480 all divide it
   (82 240 / 10 280 / 1 285 / 257 m). The address rule (`§5.1`) and the withdrawal of D-C5 are right.
2. `CHUNK_EDGE = 62` at `ladder.rs:21`, `TOP_RUNG_CHUNKS = 64` at `:23`, `RUNG_MAX = 15` at `:27`,
   `HALO` at `lattice.rs:48`. Every citation checks.
3. `6 × 3 968² = 94 470 144`, and the home planet's top-rung grid is 39 629 400 cells.
4. `EXTRACT_BUDGET_US = 8_000.0` at `crates/bins/examples/terrain_cost.rs:41`, with the owner's own
   words in the doc comment. The quotation in `§14` and `§16` matches the file.
5. `terrain-legs` is NOT in `just gate`: `justfile:408` lists `terrain-pin`, `terrain-link-scan` and
   `terrain-fence-control`; `terrain-legs` is `justfile:756`. `§11`'s retraction is right.
6. The composition order: the generated shape is rank 1, an edit is rank 2 (`compose.rs:13`, `:46-48`).
7. The sea draw: between 40 % of the relief below the ladder radius and 30 % above it
   (`body.rs:186-190`). `§17` item 2 states it correctly.
8. The reach floor: 45° over 720 rows = 1.0908 mrad; 54.5 m at 50 km; 1.09 cm at 10 m; 469 px for the
   home planet's diameter at 13 100 km. All four check.
9. Level 0 at `I = 128`: 24 576 × 128 × 160 ns = 0.503 s. A patch at `I = 32`: 147 456 × 32 × 160 ns =
   0.755 s. Shape A at `I = 100`: 40 265 s, 7.55 GB at 3 B. The halo waste 81 920 / 147 456 = 55.6 %.
   All arithmetic correct as printed.
10. The resident total 0.07 + 1.18 + 0.79 + 0.79 + 1.84 = 4.67 MB, and the level-3 patch at 7 B =
    0.459 MB. Correct.
11. The cube-net neighbour rule (E7), the seven-neighbour cube corner, and the metric measurement (area
    ratio 1.425, the six-face integral agreeing with 4π to 1.2 × 10^-6). E8 is the right consequence.
12. The tie-breaking rules (E2, the heap key `(height, cell index)`), the fixed pass order, the
    read-one-array-write-another rule and the ban on atomics are correct for byte-identical hosts. The
    deletion of revision 1's E4 claim is right: quantisation does not remove a one-ulp difference.
13. The leg table: `rust:1.94.1-slim-bookworm` on two platforms, G4 emulated, and D-TERRAIN-1 recording
    the real x86-64 machine as owed.
14. The rung-11 mesh figure re-scaled to 2.03 GB from slice 6's measured 4 055 vertices, and the
    conclusion that a planet is never fully meshed at any rung.
15. `§6.2`'s carve rule: `tubes_carve_at` carves while `cell_m <= tube_radius_m * 2`
    (`carve.rs:34-37`), so a 15 m-radius channel is carved at rung 4 and dropped at rung 5.
16. The column-pass-not-cell-pass decision, its 35.7 ms alternative, and the honest statement that a
    column pass cannot make an overhang — so the reference picture's pillars and undercuts belong to
    the cell pass. This is the document's best passage.
17. The refusal of the background build (a shape that is a function of time) and its replacement by a
    refusal plus staging. Correct under V1 clause 1.
18. D-C13: the golden self-check must fold a small fixture, not the home planet's map. The call sites
    are real (`gateway.rs:193`, `client.rs:118`, `lib.rs:1047`, `tag.rs:47`).
19. `§4.2`'s measured octave amplitudes (15.2 m across the five octaves under 800 m) and the plain
    statement that this domain does not fix that band. Naming the limit at the top is the right shape.
20. The seed-ruling gate: the pyramid places shape, never value, and no ore rides it.
