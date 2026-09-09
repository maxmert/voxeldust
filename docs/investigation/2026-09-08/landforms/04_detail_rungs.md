# 04 — THE FINE RUNGS AND THE LADDER: the shape at a point, at every rung

**Date:** 2026-09-08. **Domain 04** of the landform investigation. **REVISION 2**, after round 2 of the
refutations (`verdicts/04_detail_rungs_refutation_a.md`, `_b.md`). §14 answers every finding of both
rounds. **Status:** an investigation report for the owner. It designs the per-point height and density
function. It decides nothing by itself.
**Reads from:** DOMAIN 02 (the macro layer stack), DOMAIN 03 (the erosion solve and the river graph)
and DOMAIN 05 (climate and weather), **in their revision-2 texts, which this revision re-read**.
**Binding law read first:** CLAUDE.md (HR1–HR6, SL1–SL10), `owner_decisions_2026-09-07_voxels.md`
(V1 SL10, V2 the nine requirements, V4 the review, V6 the formats, V8, V9 the generator, V10 the
extractor and THE VISTA, V11, V12), `owner_decisions_2026-08-27_seed_and_secrecy.md`,
`docs/design/DEFERRED.md` (D-TERRAIN-1..4).

Every number is marked **MEASURED** (with the bench that produced it), **COMPUTED** (arithmetic done
in this document, from a law in the code or from a measured unit cost) or **ESTIMATED** /
**UNMEASURED**. Every claim about the code cites `file:line`. Every citation in this revision was
re-opened; the six slips refuter A found are corrected.

**What revision 2 changed, in seven lines.**

1. **The octave anchor moved to the LADDER.** Revision 1 anchored the table to DOMAIN 03's erosion
   spacing. DOMAIN 03 has since declared a lattice of 640 nodes and 8 224 m that is **not a rung and
   never appears in an address**. Both refuters caught it. §4.3 now anchors the table to the ladder's
   own cell, which is this domain's own object, and the anchor needs nothing from a neighbour.
2. **The relief law goes back to DOMAIN 02.** A bare gravity ceiling refuses every body under about
   313 km of radius (COMPUTED, §4.3). DOMAIN 02 owns the ceiling, states it as a **scale, not a wall**,
   and keeps a size term. This domain withdraws its own version.
3. **The modulation became ONE PER-COLUMN factor.** The per-octave derivative damping is withdrawn.
   The amplitude of octave `i` is now `A_i × r(dir)`, which makes the coarse rung's bound EXACT again,
   obeys DOMAIN 02's binding multiplier rule, and costs one divide per column instead of one per octave.
4. **The layer stack is a VENEER, not a periodic stack.** A stack that repeats forever downward
   unbounds `max_depth_m()` and kills the below-skip, which writes 463 of the 470 chunks in a radial
   column today. §4.5 keeps the shape law radial and the SUBSTANCE inside today's envelope.
5. **The terrace's falloff is measured against HALF the band spacing**, so it reaches zero before the
   next band top. Without that the terrace jumps by 12 m at a mid-band radius and `Lip(T)` is unbounded.
6. **The SL8 headline is corrected and it no longer says "tighter everywhere".** Against the roughness
   the home planet actually drew, the proposal is tighter on smooth ground and LOOSER on steep ground.
   §5.2 states the gate as a **bound in cells of the rung**, which is resolution-independent.
7. **The budget verdict is UNPROVEN, not FAILED.** Revision 1's "FAILED" mixed rungs and used a cost
   the document had itself struck. §9 gives a range and names the three measurements that close it.

---

## 0. The recommendation, in one page

**The task.** The owner showed a Crimson Desert vista: snow ridges, carved valleys, cliffs, rock
pillars, forest as a mass, a river, roads, haze, detail to the horizon over about 50 km. The owner
looked at our own pictures and said *"no orienters and no details at all"*, and *"the only thing I'm
worried about is that the surface will not be interesting enough"*.

**The measured cause of "no details".** The home planet's HEIGHT FIELD is one global sum of octaves
(`crates/terrain/src/height.rs:17-27`). The amplitudes fall by one seed-drawn factor `k_rough` in
`[0.45, 0.55)` (`body.rs:155,176`), and their sum is the body's whole relief (`body.rs:180-184`). The
home planet drew 14 octaves and a relief of 14 304.887 m (MEASURED, `terrain_cost`, quoted in
`docs/investigation/2026-09-07/slice_05_generator.md:368`). DOMAIN 02 MEASURED the consequence on the
real body: the home planet drew `k_rough = 0.468 338 1`, its root-mean-square slope is **1.6° to 2.0°
at every baseline from 2 m to 8 km**, and the steepest slope in 4 900 samples is **7.1°**
(DOMAIN 02 §1.2). COMPUTED from the same law and the same two drawn numbers, the finest octave has a
wavelength of 48.83 m and an amplitude of **0.397 m** on this body. So the height field carries nothing
smaller than 48.8 m across, and it carries the same roughness on a salt flat and on an alpine wall,
because `k_rough` is one draw per body and not a field.

*(The CAVES are smaller: a tube radius is drawn in `2..5` m and a cavern wavelength in `24..48` m,
`body.rs:221,228`. The sentence above is true of the height field only.)*

**Why there are no orienters, stated correctly.** `octaves_at` keeps the COARSEST octaves and drops the
finest (`body.rs:253-257`), so the orienters were never the octaves a coarse rung drops. The real cause
is that **an isotropic sum of band-limited noise has no lines**. It has no ridge crest, no coastline, no
scarp, no basin rim and no valley axis, at any scale, because every direction is statistically the same.
A pilot has nothing to steer by, and a coarse rung has nothing to draw.

**And the second half of that sentence, which revision 1 did not answer.** Refuter B counted what
carries a LINE at each scale under revision 1 and found the band from about **100 m to 3 km empty** —
the exact band the reference picture's spurs, gullies and ridge lines live in. That is right. §4.4 now
carries a **ridged, flow-aligned octave band**: in that band the octave is `1 − |noise|`, which makes
creases instead of lumps, and it is sampled in a frame stretched along the macro flow direction that
DOMAIN 03's D8 receiver already holds. Lines at the vista's own scale come from that term, and they
agree with the drainage instead of fighting it.

**The recommendation, in one sentence.** Replace the one global octave sum with a **fold**: start from
DOMAIN 02's macro layer stack plus DOMAIN 03's eroded macro field `Z` at the point, find the distance to
the nearest channel and the flow direction there, then add octaves whose amplitude is the body's own
spectrum times ONE per-column factor read from the macro state, ridged and flow-aligned in the middle
band, then bench the rock into terraces, then cut the channel — and anchor the octave table to the
LADDER, so the free-coarsening law becomes exact instead of approximate.

```text
   h(dir, rung)  =  a FOLD, coarse to fine, stopped early at a coarse rung
   +----------------------------------------------------------------------+
   | s0 = MACRO(dir)          the layer stack L1..L5 (DOMAIN 02)          |  every rung
   |      + Z(dir)            the eroded macro field (DOMAIN 03), bicubic |  every rung
   | dc, flow = CHANNEL(dir)  distance to the nearest channel, and the    |  every rung
   |                          D8 flow direction at the macro node         |
   | r  = MODULATION(dir)     ONE per-column factor: rock, macro slope,   |  every rung
   |                          the channel damping. Rung-independent.      |
   | s1 = s0 + A0*r*n0(dir)   octave 0, wavelength C * cell_m(top rung)   |  rungs 0..10
   | s2 = s1 + A1*r*n1(dir)   ridged and flow-aligned inside the middle   |  rungs 0..9
   | ...                      band; plain value noise outside it          |
   | s+ = s11 + FINE_FLOOR    one biome octave at 8 m                     |  rung 0 only
   | h  = CARVE(TERRACE(s+))  the benching, then the channel              |  every rung
   +----------------------------------------------------------------------+
   gap(cell) = (r - h) / (cell * sqrt(1 + |grad h|^2))  +  the 3-D removals
```

**The eight parts, and what each buys in the picture.**

| # | The part | What the eye gets | Where |
|---|---|---|---|
| 1 | The macro layer stack and the eroded field `Z` as the coarse term (DOMAIN 02, DOMAIN 03) | Ridges, basins, coasts, mesas, scarps, trunk valleys — the ORIENTERS, and they are LINES | §4.2 |
| 2 | One octave per rung, from `C` cells of the top rung down to `C` cells of rung 0 | Detail at every distance, at a constant cost per rung | §4.3 |
| 3 | The octave amplitude times ONE per-column factor read from the macro state | An alpine wall is rough; a farm plain is flat; both from ONE law, and the ladder bound stays exact | §4.4 |
| 4 | A ridged, flow-aligned band between the macro node and the hillslope length | Spurs, gullies and ridge crests — LINES between 100 m and 3 km | §4.4 |
| 5 | Radial layers and a terrace | Mesas, cliff bands, hogbacks, caprock — the Crimson Desert look | §4.5 |
| 6 | The channel carve from the river graph (DOMAIN 03) | Valleys, banks and a bed a pilot can follow | §4.6 |
| 7 | A 3-D REMOVAL term in a thin band at the surface | Arches, undercuts, cave mouths in a cliff — which DOMAIN 02 says by name no layer of its own can make | §6.2 |
| 8 | A fine floor and a per-biome surface form | Dunes, rills, hummocks — no glass-flat ground | §4.7 |

**Cost: the budget row is UNPROVEN.** The budget is 8 ms of worker time per chunk, at any rung
(ruling V10, `owner_decisions_2026-09-07_voxels.md:441-443`; revision 1 wrongly added "rung-0").
Revision 1 said the row FAILED at 9.60 ms. Refuter A showed that number applies deltas to a rung-2
chunk that this document's own rung rules switch off at rung 2, and refuter B showed one of its rows
uses a cost the document had itself struck as unavailable. Rebuilt coherently (§9.2), the rung-0 line
is **about 4.9 ms of box time** and the worst measured cave-dense chunk's line is **about 6.9 ms**,
both plus an extraction that grows with a vertex count nobody has measured, and both plus an undercut
term whose cost is UNKNOWN. **The honest sentence is: the budget is neither passed nor failed, and
M4-1, M4-2 and M4-11 must run before the fold is written.**

**The free-coarsening law survives, and the gate is now in CELLS, not pixels** (§5). The fold runs
coarse to fine and the modulation is one per-column factor, so rung `L`'s answer is exactly the first
`n − L` steps of rung 0's fold and the coarse rung can STATE its own bound exactly. COMPUTED, the law
is scale-free: **the rung change measured in cells of that rung is about 3.3 times the per-octave slope
of the rung's finest octave** (§5.2). At a per-octave slope of 0.3 that is 0.98 cells; at 0.6 it is
1.96 cells. In pixels that is 1.0 to 2.0 under the conservative residency rule and 0.6 to 1.1 under the
code's own drawable factor — and the ratio is resolution-independent, which refuter B is right to say
the owner should be told.

**And the comparison with today is corrected.** Revision 1 said the proposal is "tighter everywhere"
and gave today's rung-3 bound as 6.11 m. That is the number at `k_rough = 0.5`. At the roughness the
home planet actually drew, today's `dropped_bound_m(3)` is **3.05 m** and `dropped_bound_m(6)` is
**32.76 m** (COMPUTED from `body.rs:176,182,274-277` with the drawn `k = 0.468 338 1` and
`relief = 14 304.887 m`). A proposal that puts real roughness on the mountains **loosens** the bound
there and tightens it on the plains. That is the price of the picture, and §0 states it instead of the
opposite.

**Weather, plainly.** This domain gives a weather model its inputs — elevation, slope, aspect, shelter,
distance to open water, upwind ridge height — and it **builds no weather**. DOMAIN 05 owns the model,
and DOMAIN 02 has already opened the SL6 ask for the lane it would need (its §5.5). A static climate
mean is seed shape and crosses nothing; today's storm is live state and crosses as a diff.

**What must be decided before the first world is saved.** The world identity has tolerance ZERO
(ruling V9 S5-2): a change that moves one rung-0 byte opens a world epoch. Most of this document moves
bytes. It is free today and expensive later. §12 is the decision table.

---

## 1. What the code holds today (the only current truth)

| Fact | Where | What it means here |
|---|---|---|
| The height is one sum of octaves along a direction, with the `rung` finest octaves dropped. | `crates/terrain/src/height.rs:17-27`; `body.rs:252-258` (`octaves_at`) | There is no macro stack, no river, no terrace and no 3-D surface term. Everything this document adds is new code. |
| The relief is 0.4 % of the radius, clamped to `200..12 000` m, times a seed factor in `[0.5, 1.5)`. | `body.rs:146-149` | A body-wide constant, not a field. DOMAIN 02 replaces the CAP with a gravity ceiling and keeps the size term. §4.3. |
| The coarsest wavelength is a quarter to a half of the radius, clamped to `20..400` km. The floor is 30 m. | `body.rs:151-153,169`; `LONG_WAVE_CAP_M` and `SHORT_WAVE_M` at `body.rs:24-25` | Two magic numbers in metres, tied to nothing. §4.3 replaces both with the ladder's own cell. |
| The amplitude ratio `k_rough` is ONE draw per body, in `[0.45, 0.55)`; the home planet drew 0.468 338 1. | `body.rs:155,176`; the draw MEASURED by DOMAIN 02 §1.2 | One roughness for a whole planet. A desert plain and an alpine face read the same number. |
| The amplitudes are scaled so their SUM is the relief. | `body.rs:180-184` | The envelope is exact, so a spectrum may be redistributed inside it without moving the band (DOMAIN 03's §4.2 rests on this, and so does §4.3 here). |
| `octaves_at` clamps to **at least one octave**, and a landed test asserts the count falls STRICTLY at every rung from 1 to `rungs − 1`. | `body.rs:255-256`; the test at `body.rs:355-365`; the clamp's own test at `body.rs:368-379` | §4.3's table keeps the strict fall. The clamp must go, because at the top rung the right octave count is zero: `Z` carries the shape there. §12 D4-15. |
| `dropped_bound_m(rung)` is the tier-agreement bound: the sum of the dropped amplitudes, a per-body constant. | `body.rs:274-277`; the property test at `height.rs:88-96` | COMPUTED on the home planet's own drawn numbers: 0.40 m at rung 1, 3.05 m at rung 3, 32.76 m at rung 6, 321.97 m at rung 9, 1 469.16 m at rung 11. §5.2 compares against these, not against a `k = 0.5` body. |
| The ladder BAND is part of the ADDRESS. `crust_m = relief + strata.max_depth_m() + caves.max_depth_m + 64`, and `Ladder::for_radius` turns it into `floor_m = surface − crust`. `for_radius` returns `None` when `crust >= surface`. | `body.rs:233-236`; `crates/seed/src/ladder.rs:87-97,132-139`; the refusal at `ladder.rs:89` | **Move the relief or `max_depth_m()` by one metre and every cell's `k` index on the planet shifts by one.** §4.3 and §4.5 are both written to leave both alone. §12 D4-14. |
| `max_depth_m()`'s written contract: *"The deepest metre the strata change at; below it a cell is bedrock without a lookup."* | `strata.rs:165-169` | A layer stack that repeats forever downward has no such number. §4.5 keeps the veneer. |
| The below-skip writes a whole chunk without a cell pass only when every cell is bedrock. | `chunk.rs:12-17` | MEASURED: one radial column at rung 0 holds 470 chunks, **463 of them are skipped**, and the column still costs 417 ms (`docs/investigation/2026-09-07/slice_05_generator.md:387-389`). Any stratigraphy that changes substance at depth deletes that skip. §4.5. |
| The biome comes from latitude, height over the sea and two slow noises, and the CELL PASS reads it to choose the substance. | `height.rs:39-66`; computed at `chunk.rs:183`, carried in `ColumnField`, read at `chunk.rs:637` | It is a pure function of `(body, dir, surface)`, so BOTH hosts compute it under SL10. §3.1 keeps ONE biome source and §7 states what its signature must grow to. |
| The pole axis is `+Z`, the orbit's axis. There is no obliquity, no spin and no atmosphere on a body today. | `height.rs:29-35` | The owner asked for biomes that depend on position, spin, trajectory, size and gravity. None of those five reaches `biome_at` today. §3.3, §13 item 13. |
| The gap byte is the RADIAL difference `(r − h)/cell`, **clamped to one cell**, quantised to 1/128 cell. | `chunk.rs:630`; the convention at `chunk.rs:19-23` | The clamp, not the ratio, is what a cliff breaks. §6.1. |
| The extractor's crossing is a RATIO of two gap magnitudes, and its only surface is the rock/air sign. | `extract.rs:182-187`; `is_rock(gap) = gap < 0` at `extract.rs:70-74` | On a RADIAL edge both cells share one column, so a per-column factor divides out exactly and moves no vertex. And a WATER cell is on the air side, so it emits no triangle. §4.6, §6.1. |
| The gap byte and the stratum code are both folded into the world identity. | `digest.rs:38-42` | Any change to the density byte's meaning is a world epoch on the day it lands, not only when Format B freezes. §6.1, §12 D4-8. |
| The cave field is a value-noise lattice on a GLOBAL node grid, one node per 4 cells, trilinear, read only inside the depth band, and it can only OPEN air. | `carve.rs:24,49-66`; `chunk.rs:435-520`; the band gate `chunk.rs:641`; the fold `chunk.rs:641-649` (`greater` at `:646`, `Stratum::Air` at `:647`) | This is the machinery the 3-D REMOVAL term reuses (§6.2). It cannot ADD rock. |
| Tube caves are gathered per REGION and pruned against the chunk's bounding sphere before any cell pays. | `chunk.rs:320-370`; `carve.rs:127-146` (`tubes_near`) | The channel carve reuses this shape — **per region, not per chunk**, which is what makes a halo column agree with its neighbour's core. §4.6. |
| The strata are ONE topsoil (1–4 m), ONE subsoil (2–8 m), ONE sediment band of `20..80` m, then bedrock forever; the substance is chosen by DEPTH under the surface. | `strata.rs:166-178,188-209`; drawn at `body.rs:196-201` | A depth band follows the hill, so it can never make a bench. §4.5 makes the SHAPE radial and leaves the substance inside this envelope. |
| The sea IS cells: `fluid_at` returns `Stratum::Water` under `sea_radius_m`, and `finish_cell` writes it. | `chunk.rs:207-213`; `chunk.rs:633-635`; `Stratum::Water = 1` at `strata.rs:18` | A river must be cells too, or the game has two mechanisms for one job. But nothing outside `vd-terrain` reads `Stratum::Water` today, and no buoyancy exists. §4.6 states both halves. |
| The mesh holds vertices and triangles only. No material rides with it. | `extract.rs:62-69` (`ChunkMesh`) | The client cannot texture a cliff band differently from the soil above it. §7 adds the stratum per vertex. |
| **Base terrain has NO record.** A record exists only for a cell a player changed, and the `object` byte is the tree's shape parameter. | `docs/investigation/2026-09-07/topic_00_format_sitting.md:34-36,55`; ruling V6 B-5 | Nothing in this document mints a record. §7. |
| `seat_eighths` reads the cell and exactly ONE neighbour — above if the cell is rock, below if it is air. | `seat.rs:32-62` | It is a SECOND reader of the density byte, and §6.1 owes it the same argument the extractor gets. |
| The composition order is fixed: generated shape, then cell edits, then catalogue blocks, then sub-metre blocks, then attachments. | `compose.rs:1-19,60-88` | An authored delta is live state and rides row 2's lane, ON TOP of the generated shape. §8. |
| The physics crate computes a body's surface gravity, `G·M ÷ R²`, from a mass drawn through `powf`. | `crates/physics/src/taxonomy.rs:750-752`; `segmented_power_law` at `:616` | `powf` is libm, outside the float fence, and the terrain crate states in writing that no unfenced float may enter a body (`body.rs:86`). So gravity must be SHIPPED as a quantised integer, which is DOMAIN 02's SL6 ask. §4.3. |
| The home planet: look radius 3 351 154 m, ladder radius **3 350 759 m**, `N = 5 263 360 = 2¹² · 5 · 257` cells per face edge at rung 0, **12 rungs** (cells 1 m to 2 048 m), **14 octaves**, relief **14 304.887 m**. | MEASURED, `terrain_cost`; `docs/investigation/2026-09-07/slice_05_generator.md:341,368`; `crates/terrain/src/home.rs:35-41` | COMPUTED: 84 892.9 chunks along a face edge at rung 0. The two literals in `home.rs` are pinned by `crates/bins/tests/home_body_pin.rs`; a third input adds a third literal and a third pin (§13 item 12). |
| The reference view: one pixel is `2·tan(22.5°)/720 = 1.1506 × 10⁻³` rad, and a thing of extent `e` is drawable to `e × 2/θ = e × 1 738` m. | `crates/core/src/geometry.rs:1156-1173` | COMPUTED: a 1 m cell read as a half-metre extent is drawable to 869 m; read as a cube (extent `√3/2`) it reaches 1 505 m. §5.2 states the gate so that the choice only scales it. |
| Column pass by rung: 710 µs at rung 0 (14 octaves), 266 µs at rung 11 (3 octaves), over **3 844 columns** (62 × 62). The sample BOX is **4 096 columns** (64 × 64, the chunk plus one halo cell). | MEASURED, `docs/investigation/2026-09-07/slice_05_generator.md:374-380`; `chunk.rs:24-25`; `lattice.rs:1-2` | COMPUTED unit cost: **10.5 ns per column-sample per octave**, so **40.4 µs** per octave over 3 844 columns and **43.0 µs** over the box's 4 096, plus **144 µs** of fixed cost. §9 prices every per-column row over 4 096 and says so once. |
| Sample box 1.98 ms, extraction 1.32 ms, 5 650 vertices at rung 0. The worst MEASURED chunk (+X rung 2, (21 223, 7), a seam, cave-dense): 4.38 ms and 1.73 ms, 8 234 vertices. The cell pass alone: 579 µs. | MEASURED, `docs/investigation/2026-09-07/slice_06_extractor.md:346-352`; `slice_05_generator.md:381` | The extraction slope of 61 ns per vertex is a regression over two rows at two rungs; §9.1 states exactly why that one is defensible and the cave delta is not. |
| The client's own per-chunk geometry cost: **4.18 ms** on one thread, **405 228 bytes** per rung-0 chunk at 8 577 vertices and 16 615 indices. | MEASURED, `docs/investigation/2026-09-07/slice_07_client_link.md:235-236` | COMPUTED: `8 577 × 12 + 8 577 × 12 + 16 615 × 12 = 405 228` exactly — positions, normals and indices, all four bytes a component. §5.4 uses that decomposition for the lever. |
| The snap error on surface vertices: p50 0.0039 cells, max 0.0090 cells. | MEASURED; `docs/investigation/2026-09-07/slice_06_extractor.md:356-357` | Measured on gentle ground. §6.1 owes the cliff measurement (M4-9). |

---

## 2. The words, explained once

Each word below is the industry's own. Each explanation carries an example in the game's words.

- **Hypsometry** — how much of a body's surface stands at each height. Earth's curve has two humps: a
  broad continental platform near sea level, a broad abyssal plain about 4 km down, and a steep step
  between them. *Example:* the home planet's sea stands 5 297 m under the ladder radius and covers
  about one column in a hundred (MEASURED, `slice_05_generator.md:391`). That is a dry world with a
  puddle, not an Earth-like one. The curve is DOMAIN 02's and DOMAIN 03's; this domain must not fight it.
- **fBm (fractional Brownian motion)** — a sum of noise octaves whose wavelength halves and whose
  amplitude falls by a fixed ratio. *Example:* the home planet's hills today, `height.rs:17-27`.
- **Lacunarity** — the wavelength ratio between two neighbouring octaves. Ours is 2 (the halving at
  `body.rs:177`). *Example:* on the home planet octave 3 has a 2 048 m wave and octave 4 has a 1 024 m
  wave. It stays 2, because only a ratio of 2 lets one rung drop exactly one octave.
- **Gain, also called persistence** — the amplitude ratio between two neighbouring octaves. Ours is
  `k_rough` (`body.rs:155`). *Example:* the home planet drew 0.468 338 1, so its 1 024 m hills are 47 %
  as tall as its 2 048 m hills, everywhere on the body.
- **Slope spectrum** — the same table read as SLOPE instead of amplitude: `s(o) = 2π·a(o)/λ(o)`. It is
  the honest quantity, because slope is what the eye and the boot feel. *Example:* DOMAIN 03's §4.2
  states the law as a slope that peaks at the valley spacing, and this domain reads that table.
- **Hurst exponent `H`** — the roughness exponent, where `gain = 2^(−H)`. A gain of 0.47 puts `H` above
  1, which no fractal surface has. *This document quotes the GAIN and the SLOPE, never `H`.*
- **Multifractal** (Musgrave, 1993) — an fBm whose octave amplitude is modulated by the field already
  accumulated, so different places get different roughness from ONE table. *Example:* a ridge column of
  the home planet keeps its full amplitude; the plain beside it gets a fraction of it.
- **Ridged noise** — the transform `1 − |n|`, which turns a smooth wave into a sharp crest with a
  rounded trough. It is how the trade makes a LINE out of an isotropic field. *Example:* a spur running
  down from the ridge above the pilot's landing site, instead of a lump.
- **Anisotropic** — different along different directions. *Example:* the fine octaves inside a valley on
  the home planet are stretched along the flow direction the D8 receiver holds, so the gullies run down
  the slope instead of across it. A dune field is anisotropic for the same reason, with the wind
  standing in for the flow.
- **Domain warping** — displacing the sample point by another noise before the evaluation, so ridges
  bend instead of running straight. *Example:* a ridge line on the home planet stops looking like a row
  of equal waves.
- **Hillslope diffusion**, also called soil creep — soil moves downhill and smooths everything below a
  length of about 50 to 150 m. It is why real terrain has a **spectral break**: rough above the break,
  smooth below it. *Example:* it is the falling limb of DOMAIN 03's slope spectrum, and it is why the
  fine octaves on the home planet must not carry the same slope as the 2 km ones.
- **Drainage density** — how much channel length a unit of area holds, read in the field as the spacing
  between neighbouring valleys. A mountain belt has a channel every 100 to 500 m. *Example:* DOMAIN 03's
  solve gives the home planet one channel every 8 224 m; everything between them is this domain's
  ridged band, and that is the whole of §4.4's second rule.
- **Angle of repose** — the steepest angle loose material stands at, about 34° for dry rock rubble. It
  is a friction property and it does **not** change with gravity. *Example:* the scree cone under a
  cliff on the home planet stands at the same angle as one on Earth, even at 0.35 g.
- **Talus, also called scree** — the cone of broken rock at the foot of a cliff, standing at the angle
  of repose. *Example:* every cliff in the owner's reference picture stands on one. §4.7 says where it
  comes from, and it is not an octave.
- **Stream power law** — the rate at which a river cuts down, `dz/dt = U − K·A^m·S^n`. It is DOMAIN 03's
  tool. This domain only READS the channel that law produces.
- **Flow accumulation and drainage basin** — how much land drains through a point, and the region that
  drains to one outlet. Together they set a channel's width and depth. *Example:* a river on the home
  planet is 4 m wide near a ridge and 200 m wide at its mouth, because the accumulation says so.
- **D8** — the rule that sends all the water leaving a node to the lowest of its eight neighbours, so a
  channel is a chain of nodes and every node has exactly ONE outgoing direction. *Example:* DOMAIN 03
  keeps that direction in three bits per node, and §4.4 reads it as the anisotropy direction.
- **Base level** — the height a river cannot cut below, normally the sea. *Example:* a channel on the
  home planet never cuts under the sea radius.
- **Knickpoint** — a step in a river's long profile: a waterfall or a rapid. *Example:* a hard radial
  band that a channel crosses on the home planet makes one. §3.2 asks DOMAIN 03 to allow it.
- **Isostasy** — the crust floats on the mantle, so a thick mountain root holds a high surface.
  *Example:* the home planet's highest range stands high because DOMAIN 02's layer L2 gave it a thick
  low-density root, not because a noise octave was large there.
- **Tectonic uplift** — the rate at which a region rises. *Example:* on the home planet DOMAIN 02's
  layer L3 raises the ground along a plate boundary, and that is where the alpine columns sit.
- **Crustal strength and the relief ceiling** — rock at the base of a mountain fails when the weight
  above it passes the rock's strength, so the tallest a mountain can stand is about
  `strength ÷ (density × gravity)`. *Example:* DOMAIN 02 fixes the strength at 243 MPa from Everest and
  reads the relation as a SCALE with a ±15 % spread, not as a wall. §4.3 explains why this domain
  stopped trying to own it.
- **Structural datum and dip** — the surface a rock layer was laid on, and the angle it now sits at.
  Flat-lying layers make mesas; tilted layers make hogbacks. *Example:* the flat-topped mesas in the
  owner's vista.
- **Hogback** — a sharp ridge made where a hard tilted layer stands out of softer rock, so both its
  faces are steep. *Example:* on the home planet a hogback stands where DOMAIN 02's dip channel is large
  and the terrace's hard band reaches the surface.
- **Caprock** — a hard layer on top of soft rock. The soft rock erodes back under it, so the hard layer
  stands out over the edge. *Example:* the flat lid of a mesa in the owner's vista.
- **Terracing, also called benching** — pulling the height toward the top of a hard layer, so the slope
  steps. It turns a smooth hill into a mesa.
- **Veneer** — a thin cover of one material over a different basement. *Example:* the home planet's
  topsoil, subsoil and sediment together are a veneer at most 92 m deep, and under it every cell is
  bedrock without a lookup (`strata.rs:165-169`). §4.5 keeps that veneer, because the below-skip lives
  on it.
- **Nyquist rate** — a wave must be sampled at least twice per wavelength to exist at all, and about six
  times to look smooth. *Example:* a 16 m wave on 1 m cells has 16 samples; a 2 m wave has 2 and aliases
  into hash.
- **Catmull-Rom bicubic** — a smooth interpolation through four samples per axis, whose weights are
  cubic polynomials. *Example:* DOMAIN 03's eroded field `Z` between its 8 224 m nodes.
- **Smoothstep** — the polynomial `u²(3 − 2u)`, which rises from 0 to 1 with a flat start and a flat
  end. It is add, subtract and multiply only, so it is safe under the float fence. *Example:* the
  terrace's falloff and the channel's bank both use it.
- **Lipschitz constant** — how much a function can multiply a difference in its input. It matters here
  because the terrace step multiplies the disagreement between two rungs (§5.2).
- **Hadley cell and Coriolis** — the tropical overturning of an atmosphere, and the sideways deflection
  a body's spin gives to moving air. Together they set which way the wind blows at each latitude.
  *Example:* on the home planet the trade winds run east to west near the equator; §3.3 makes the dunes
  read that, so the wind never becomes a stored channel.
- **Orographic lift and rain shadow** — air forced up a mountain cools and rains on the windward side,
  and the far side stays dry. *Example:* a desert on the home planet sits behind a range, not at a
  random place a noise chose.
- **Aerial perspective, also called haze** — the blue that distance adds. It belongs to the client's
  atmosphere, and it is what hides the last of a rung change (ruling V10, THE VISTA).

---

## 3. The interface this domain needs (DOMAIN 02, DOMAIN 03, DOMAIN 05)

This is a CONTRACT, not an assumption. **Revision 2 re-read both neighbours in their revision-2 texts,
because refuter A showed that revision 1's contract quoted a lattice DOMAIN 03 had already withdrawn.**
Where a neighbour now says something different, this section takes the neighbour's word and §14 records
what moved.

### 3.1 From DOMAIN 02: a closed-form layer stack, a ceiling, and a binding rule

DOMAIN 02 recommends **six layers, each a pure function of the unit direction, the body's seed and the
body's physical facts. No grid, no bake, no cache except a short plate list inside `BodyDefinition`.**

```text
   macro_at(dir) -> MacroSample          a FUNCTION, evaluated per column, nothing stored
   +-------------------+----------------------------------------+----------------+
   | channel           | meaning                                | from           |
   +-------------------+----------------------------------------+----------------+
   | height_m          | the surface over the ladder radius      | L1..L3 + L5    |
   | gradient          | its slope, from the same closed form    | the same call  |
   | rough_q8          | the roughness factor r(dir), 0..255     | L1..L3         |
   | break_log2        | the hillslope length, as log2 metres    | L5 (climate)   |
   | hardness_q8       | the rock's resistance                   | L2 (crust)     |
   | datum_m, dip_q8   | the structural surface and its angle    | L1 + L3        |
   | water_m           | the basin's water surface, or none      | L4 + the sea   |
   +-------------------+----------------------------------------+----------------+
```

Four requirements this domain places on that stack, and one rule it accepts FROM the stack.

1. **The macro step is DERIVED, never stored.** The step over one macro node is `|gradient| × node_m`,
   taken from the gradient the same closed form already returns. It has no ceiling, no quantisation and
   no second source to keep consistent. Revision 1's `step_cm` channel stays deleted: refuter B showed a
   `u16` in centimetres saturates at a slope of 17.7°, which is every mountain front in the picture.
2. **Every channel is a pure function of `(seed, dir)` and of the body's facts.** No tile address may
   appear in any of them. A tile address makes a node on a tile boundary carry two values, which is a
   crease 8 km long, and it breaks SL10 no-drift and SL8 seamless in one line.
3. **The gradient comes back with the height, from the same polynomial.** §4.4 and §6.1 both need it,
   and a finite difference would cost two more evaluations per column.
4. **There is ONE biome source.** It is `height::biome_at` (`height.rs:39-66`), the function the cell
   pass already reads (`chunk.rs:183,637`). Two biome fields on one planet would be a fork — the terrace
   would read one and the substance under a player's boots the other. **But the freeze is on the NUMBER
   of sources, not on the signature**: refuter B is right that today's arguments `(body, dir, surface_m)`
   have no room for insolation, obliquity, an upwind ridge or a distance to a coast, and the owner asked
   for biomes that follow position, spin, trajectory, size and gravity. §7 states the signature the
   function must grow to, and DOMAIN 05 owns what fills it.

**The rule this domain ACCEPTS from DOMAIN 02, and now obeys by construction:**

> *A MULTIPLIER may read only layers that are evaluated at EVERY rung. A layer that is cut off at a
> coarse rung may only ADD, never MULTIPLY.* (DOMAIN 02 §4.6, binding.)

§4.4's modulation is ONE per-column factor read from L1–L3 and from the macro gradient, all
rung-independent, so `dropped_bound_m(L) × r` stays exact. Revision 1's per-octave derivative damping
broke this rule in spirit and made the coarse rung unable to state its own bound. It is WITHDRAWN.

**The ceiling belongs to DOMAIN 02.** Its change 3 fixes `σ_y = 243 MPa` from Everest, gives the home
planet a ceiling of 16.8 to 23.6 km against today's drawn relief of 14 305 m, and states the relation as
a **scale with a ±15 % spread, not a wall**. This domain withdraws its own strength draw. §4.3 says why.

**The cost this places on the budget.** DOMAIN 02's own costed table states **416 ns per column without
rivers and 672 ns with them**. Over the sample box's 4 096 columns that is **1.70 ms or 2.75 ms per
chunk**. Revision 1 used 180 ns, which was the low end of an older table; refuter A was right, and §9
now carries the neighbour's own number.

### 3.2 From DOMAIN 03: an eroded macro field, a river graph, and a SHIPPED artifact

DOMAIN 03 revision 2 runs the erosion ONCE per body on **its own uniform cube-sphere lattice**:
`n_macro = 640` nodes per face edge, **8 224 m per node**, 2 457 600 nodes on the home planet, chosen
because 640 divides `N` exactly and 642 does not. **It is not a rung of the ladder and it never appears
in an address.** It keeps an eroded macro height `Z` (`i16`, metres), one byte of D8 receiver and
facies, one byte of quantised discharge and a lake table: **9.83 MB per body**, plus a coarse pyramid
whose top level is 19 KB.

```text
   Z_at(dir)          -> correction_m        bicubic over a 4x4 node stencil, 8 224 m apart
   flow_at(dir)       -> the D8 receiver direction at the nearest node
   channels_near(dir) -> [Channel]
   Channel { start_dir, end_dir, bed_m_start, bed_m_end, half_width_m, valley_m, water_m }
```

Six rows, and revision 2 rewrote the first, the third and the sixth.

1. **The macro lattice is DOMAIN 03's own, not a rung, and this domain no longer anchors anything to
   it.** Revision 1 wrote *"the grid's spacing is a RUNG of the ladder"* and built the whole octave table
   on `S_e = 8 192 m`. Both refuters showed that 8 192 rung-0 cells do not divide `N = 5 263 360`
   (COMPUTED: `N / 8 192 = 642.5`), that 8 192 m is two rungs above the top rung, and that DOMAIN 03 had
   already chosen 8 224 m for exactly that reason. **The octave table is re-anchored to the LADDER
   instead** (§4.3). The only thing this domain now asks of the lattice is a WAVELENGTH SPLIT: DOMAIN 03
   already selects its coarse octaves by wavelength (its §4.2), so everything coarser than the node is
   inside `Z` and everything finer is in the fold. On the home planet the split falls between 8 224 m and
   the fold's coarsest octave of 8 192 m, a gap of 0.4 %, which no eye and no bound can see.
2. **`t` is not dyadic, and the no-drift argument is restated.** Revision 1 argued that the bicubic's
   offset inside a node cell is a fraction with a power of two below the line, so the weights are exact.
   On an 8 224-cell node that is false: `8 224 = 2⁵ · 257`, so `t` is an odd multiple of `1/16 448`.
   **The conclusion survives and the argument does not.** `Z`'s node values are INTEGERS shipped by the
   owning realm, the weights are `Gf` polynomials in one fixed operation order, and IEEE-754
   round-to-nearest in a fixed order is identical on x86-64 and on aarch64 — which is what the crate
   already proves on three legs (`just terrain-legs`). Exactness is not needed; identical rounding is.
3. **At most one outgoing channel per node.** D8 gives exactly one, and the same byte gives the flow
   DIRECTION that §4.4 reads for the anisotropy.
4. **A monotone bed, EXCEPT at a declared knickpoint.** The bed never rises downstream, and a DROP is
   allowed and flagged, so the carve can make a plunge pool instead of a smooth ramp.
5. **The bed sits under the macro height** at every node of the channel, by at least the channel's own
   depth. Otherwise the carve makes a trench in the air.
6. **The artifact is COMPUTED ON THE SERVER and SHIPPED.** DOMAIN 03 states this three times and calls
   it an SL6 ask (its §1 item 7 and §12 D3), because the client links no motion crate and cannot compute
   the gravity, temperature, obliquity and spin the solve needs. Revision 1 wrote *"derived on both
   hosts"* and marked SL6 PASSED on it. **That was false, and refuter B is right.** §11's SL6 row now
   reads OPEN, and §5.4 prices the 9.83 MB the client must hold before it draws its first chunk.

**One row is still OPEN across the three documents.** DOMAIN 02's layer L4 (hydrology) and DOMAIN 03's
channel carve both shape a valley, and DOMAIN 03 also proposes a fine DETAIL term (*"scree, ridged
crests, flow-warped gullies"*, its §7.3, 0.15–0.25 ms) that covers the same band as §4.4's ridged
octaves. If two of them run, a valley is cut twice and the budget pays twice. **§3.4 states the merge
this domain recommends. D4-17 and D4-18.**

### 3.3 From DOMAIN 05: what the shape OWES the weather, and what it READS back

The owner asked for weather in the same sentence as the picture. **This domain builds no weather.** It
is the place where the shape and the climate meet, so it states both directions and nothing else.

**What the shape OWES a weather model, all DERIVED, none stored:**

| Field | How this domain gives it | Why weather needs it |
|---|---|---|
| Elevation | `h(dir, rung)` | Air cools with height; the snow line is a height |
| Slope and **aspect** (which way a slope faces) | `grad h`, already in the fold | Insolation on a slope, and where snow survives the summer |
| Shelter and sky view | a query of `h` on a ring at a coarse rung | Cold air pools in a valley floor; frost forms there first |
| Distance to open water | DOMAIN 02's `water_m` mask at a coarse rung | Humidity falls inland; a continental interior is dry |
| Upwind ridge height | a query of `h` along the wind direction, at a coarse rung | Orographic lift and the rain shadow behind it |

**What this domain READS back, and how it stays lawful.** The wind direction is **a LAW, not a
channel**: a function of latitude and of the body's spin, through the Hadley cell and the Coriolis
deflection. So §4.7's dunes get their direction from `(dir, spin)` with no data crossing and no new
channel.

**The SL6 reading, stated plainly for the owner.** The static CLIMATE — the mean insolation, the mean
rain, the permanent snow line — is a function of the seed and the address, so it is SHAPE and it crosses
nothing. **A storm is LIVE STATE.** It cannot be derived; it must cross from the planet's realm to a
window, and that is a new lane. DOMAIN 02 has already opened that ask (its §5.5) and DOMAIN 05 owes the
design. **This domain adds no weather crossing and builds no weather, and §0 says so in one line so the
owner is not left thinking otherwise.**

### 3.4 ★ ONE `body.rs`, THREE DOCUMENTS: the collision, and the merge this domain recommends

Refuter A found the sharpest structural problem in the whole round: **three documents are editing the
same octave table, and the owner is being asked to approve all three.**

```text
   what                        DOMAIN 02 says          DOMAIN 03 says        DOMAIN 04 (this) says
   -------------------------   ---------------------   -------------------   ---------------------
   the coarsest wavelength     ~50 km (L1-L3 own       coarse octaves are    C * cell_m(top rung)
                               everything above)       EATEN by the solve    = 16 384 m on the home
                                                       and replaced by Z     planet
   the finest wavelength       SHORT_WAVE_M = 2 m      untouched             C * cell_m(0) = 8 m
   the count                   15 octaves              untouched             11 octaves
   the per-octave amplitude    one r(column) times     THE SLOPE SPECTRUM,   the same spectrum,
                               today's geometric       peaked at the valley  times r(column)
                               fall                    spacing
   the clamp at body.rs:256    must never fire         silent                must be REMOVED: at the
                                                                             top rung zero octaves is
                                                                             the right answer
```

**The merge.** Each document owns the part its own subject decides, and one table results:

| The part | Owner | Why |
|---|---|---|
| Everything coarser than the macro node (8 224 m) | **DOMAIN 03's `Z`** | The solve consumed those octaves; drawing them again would double the shape |
| The per-octave AMPLITUDES inside the fold's table | **DOMAIN 03's slope spectrum** | It is erosion that makes roughness peak at the valley spacing; this domain has no law for it |
| The table's ENDS and COUNT | **DOMAIN 04** (this document) | They are the ladder's own, and the free-coarsening law is what they exist to keep |
| The per-column multiplier `r(dir)` | **DOMAIN 02** states the layers it reads; **DOMAIN 04** states the slope and channel terms and the rung rule | DOMAIN 02's binding rule governs it, and §4.4 obeys it |
| The ridged, flow-aligned band | **DOMAIN 04**, reading DOMAIN 03's D8 direction | It is a shape in the fold; the direction is the solve's |
| The relief ceiling and the size term | **DOMAIN 02** | It fixed `σ_y` from a measurement and read the relation as a scale |

**What the owner must decide, once, is D4-18: whether this merge stands, or whether one document is
asked to hand its part to another.** Nothing else in this document is safe until it is settled, because
`body.rs` cannot hold two tables.

---

## 4. The height at a point and a rung

### 4.1 The fold

```rust
// The shape of the recommendation. It is not the final code.
fn height_m(body, dir, rung) -> Gf {
    let m = macro_at(body, dir);              // 4.2  DOMAIN 02's closed form + its gradient
    let mut s    = m.height + z_at(body, dir);                     // 4.2  DOMAIN 03's eroded field
    let mut grad = m.gradient + z_grad;                            //      per metre of ground
    let (dc, flow) = channel_at(body, dir);   // 4.6  ONE gather per region, one distance per column
    let r = modulation(&m, dc);               // 4.4  ONE per-column factor. Rung-independent.
    let n = body.octave_count;                // 4.3  one octave per rung
    for i in 0..n.saturating_sub(rung) {      //      the fold, coarse to fine, stopped early
        let a = body.octaves[i].amplitude_m * r;                   // 4.4  the table times r, and
        let (v, dv) = octave_sample(body, i, dir, flow);           //      nothing else
        s    += a * v;
        grad += tangent(dir, a * dv / body.octaves[i].wave_m);     // 4.1  per METRE of ground
    }
    if rung == 0 { s += fine_floor(body, dir, m.biome); }          // 4.7
    let s = terrace(body, &m, s, rung);       // 4.5  the benching, BEFORE the carve, with a rung rule
    river_carve(body, dir, s, dc, rung)       // 4.6  the carve is LAST, so the bed survives
}
```

Five properties of this shape carry the whole design.

- **It is a prefix fold.** Every term reads only the state above it. So rung `L` computes exactly the
  first `n − L` steps of rung 0's fold. §5 turns that into the free-coarsening law.
- **The modulation is ONE per-column factor.** `a_i(dir) = A_i · r(dir)`, so the amplitude of octave `i`
  is the same at every rung that keeps octave `i`, and the sum of the dropped amplitudes is
  `r(dir) × (a per-body constant)`. That is why the coarse rung can STATE its own bound exactly.
  Refuter A showed that revision 1's per-octave damping made this impossible: a coarse rung cannot know
  a damping factor that reads octaves it never evaluated, so it could only state a geometric envelope
  three times wider than the truth. **The per-column form is the cure, and it also obeys DOMAIN 02's
  binding multiplier rule by construction.**
- **Every per-point number comes from the macro state, from the body's physical facts, or from the
  seed.** No tuning constant with a metre in it survives. That is the "no magic numbers" law.
- **The order is fixed and it matters.** The channel distance is found BEFORE the loop, so the octaves
  can be damped inside the channel. The terrace runs BEFORE the carve, so the terrace cannot pull a
  carved bed back up toward a layer top and put a step in the water. The carve is the only term that may
  write the bed.
- **The terrace now takes the rung.** Refuter B is right that every other term had a rung rule and the
  terrace did not, and that a 40 m bench evaluated on a 2 048 m cell is 0.02 samples per band. §4.5
  gives it one, and §5.2 carries its contribution to the bound.

**The gradient's units.** `noise3` is sampled at `dir × frequency` where `frequency = radius_m ÷ wave_m`
(`body.rs:171`) and `dir` is a UNIT direction, so `dv` is a derivative per unit of the NOISE domain and
not per metre of ground. Moving one metre along the surface moves `dir` by `1/radius` and the sample
point by `frequency/radius = 1/wave_m`. So the conversion is a multiply by `1 ÷ wave_m`, and the result
must then be projected onto the tangent plane, because a derivative with respect to `dir` has a radial
part that is not a slope at all. Without both, a 16 384 m wave and a 16 m wave of the same amplitude
would contribute the same slope, which is wrong by a factor of 1 024.

### 4.2 The macro term, and how it is sampled

**Layers L1 to L5 need no interpolation.** They are closed forms of the unit direction (DOMAIN 02), so
they are exact at every point at every rung, and they cost what DOMAIN 02 measures. There is no node
grid, no stencil and no seam rule for them at all.

**Only DOMAIN 03's `Z` is a table**, at 8 224 m on the home planet. It is an `i16` per node, so it is
interpolated, and only there does a stencil exist.

```text
          the 4x4 stencil around one column, over the macro nodes
              n-1      n       n+1      n+2      (nodes at 8 224 m on the home planet)
          +--------+--------+--------+--------+
   m-1    |        |        |        |        |
          +--------+--------+--------+--------+
   m      |        |   P0---+--*     |        |    *  = the column's direction
          +--------+--------+--------+--------+    t  = the offset inside the node cell:
   m+1    |        |        |        |        |         an odd multiple of 1/16 448 at rung 0,
          +--------+--------+--------+--------+         which is NOT a power of two
   m+2    |        |        |        |        |
          +--------+--------+--------+--------+
```

**Why it is still byte-identical, without the dyadic claim.** The node values are integers, shipped by
the owning realm. The weights are cubic polynomials of `t` computed in `Gf` with `+ − × ÷` only, in one
fixed operation order. The fence already forbids fused multiply-add, libm and reassociation
(`crates/terrain/src/gf.rs`, the crate's `clippy.toml`, the link scan). Rounding happens, and it is the
SAME rounding on both hosts, which is the property SL10 needs — the same property the crate already
proves on three legs. **ESTIMATED cost: about 15 ns per sample** (an operation count, not a measurement).
An exact `i128` rational form is the belt and braces at an ESTIMATED 60 ns; take it only if a leg ever
disagrees.

**The gradient is free.** The derivative of a cubic is a quadratic over the same stencil.

**Seams.** The stencil crosses a cube edge. It uses THE crossing rule, `vd_seed::seam::across`, the same
one the halo uses (`crates/terrain/src/lattice.rs:11-12`). At a corner the missing node takes the corner
node's value, and a test proves the three faces agree, in the shape of `extract::prism_vertex` — the
mechanism ruling V10 landed for the corner, not an argument from ruling A5, which says nothing about
symmetry (`topic_00_format_sitting.md:23`).

**The seam measurement.** Every column belongs to exactly one face, and a halo column across a seam is
computed from the PARTNER's own `(face, i, j)` by `dir_of` (`chunk.rs:139-142`) after `across`, so it is
bit-identical to the partner's copy by construction. There is no step between two evaluations to
measure. What a seam can really show is a **slope kink** between adjacent columns on either side of the
edge, and the gate is stated in slope. **M4-6.**

### 4.3 The octave table, anchored to the LADDER; and why the relief law goes back to DOMAIN 02

**Proposal 1 — the table is anchored to the ladder's own cell, and to nothing else.**

```text
   the finest octave     lambda_0min = C * cell_m(0)        C = cells per wavelength
   the coarsest octave   the first C * cell_m(L) at or BELOW the macro node size
   the count             n = (that rung) + 1                one octave dropped per rung
   rung L keeps octaves  0 .. n-L-1                         zero at the rungs above n-1
```

COMPUTED for the home planet with `C = 8`, 12 rungs and DOMAIN 03's 8 224 m node: the coarsest fold
octave is `8 × 2¹⁰ = 8 192 m` (the next one up, 16 384 m, is above the node and belongs to `Z`), the
finest is `8 × 1 = 8 m`, and **`n = 11` octaves**. `LONG_WAVE_CAP_M` and `SHORT_WAVE_M`
(`body.rs:24-25`) both disappear, and **nothing in the table reads a neighbour's lattice**. That is the
whole cure for the anchor both refuters broke: the ladder is this domain's own object, and it cannot be
withdrawn by a neighbour.

**Why this shape.** At rung `L` the finest live octave has the wavelength `C · 2^L` metres and the cell
is `2^L` metres. So **every rung samples its own finest octave at about `C` cells per wavelength**, and
no rung ever aliases. "About", not "exactly": the face bend makes a cell 0.935 of a face-centre cell at
a cube corner, so `C = 8` reads 8.55 there. COMPUTED from `bend.rs:22-30` with `K1 = π/4`, `K2 = 0.15`
and `K3 = 1 − K1 − K2`: `W'(1) = K1 + 3K2 + 5K3 = 1.5584`, and the corner's own tangent metric adds
`√(1 − 1/3)/√3 = 0.4714` against 1 at the face centre, so the cell is `1.5584 × 0.4714 / 0.7854 = 0.935`
of a centre cell. Both refuters were right that revision 1 printed the answer without that second step.

**The count falls strictly, and the clamp must go.** COMPUTED: the counts run 11, 10, … 1, 0 across
rungs 0 to 11, so `every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it` (`body.rs:355-365`)
passes as written **once the "never fewer than one octave" clamp (`body.rs:255-256`) is removed**. The
clamp must go for a reason the merge makes clear: **at the top rung the right octave count is ZERO**,
because DOMAIN 03's `Z` carries the shape there and is present at every rung. Keeping one 8 192 m octave
at a 2 048 m cell would be 4 samples per wavelength at the top rung and 2 at the rung above — the
aliasing `C` exists to prevent. §12 D4-15.

**Choosing `C`, and the disagreement with DOMAIN 02 stated in full.** DOMAIN 02's change 1 sets
`SHORT_WAVE_M = 2 m` so the spectrum "reaches the cell". This domain recommends **`C = 8`, a floor of
8 m**, and it owes the owner the arithmetic behind the difference:

| `C` | Finest wave | Samples per wavelength at rung 0 | What it buys | What it costs |
|---|---|---|---|---|
| 2–3 | 2–3 m | 2–3 | the metre scale is in the height field | a gradient-noise lattice at Nyquist reads as hash, and the amplitude there is ESTIMATED under 0.05 m, which is 6 quanta of the gap byte — a texture, not a shape |
| **8** | **8 m** | **8** | the finest feature that reads cleanly on 1 m cells | nothing under 8 m in the height field; §4.7 answers that band with objects |
| 16 | 16 m | 16 | no aliasing at all | an 8 m arch and a 12 m channel are no longer cut anywhere (refuter A, and he is right) |

**RECOMMENDED: `C = 8`, with M4-3 as the gate, and the decision taken together with DOMAIN 02's
`SHORT_WAVE_M`.** They are the same number, and only one of them may live in `body.rs`.

**The law is total on a small body.** A 10 km rock has `rungs = 3` and a macro lattice of 8 nodes per
face edge (DOMAIN 03's own minimum). COMPUTED: the fold's coarsest octave is `8 × 2² = 32 m`, the finest
is 8 m, `n = 3` octaves, and `Z` carries everything above. That is believable — a 10 km rock has little
scale range — and it is one world with no variant (SL5).

**Proposal 2 — the relief law and its ceiling go back to DOMAIN 02, and this domain withdraws its
version.** Revision 1 proposed `relief_ceiling_m = crustal_strength ÷ (rock_density × surface_gravity)`
with the strength as a per-body seed draw, and §11 said the proposal DELETED the `0.4 % of the radius`
share with its clamps. Both refuters killed it, and they are right:

- **A bare ceiling grows as the body shrinks.** For a body of uniform density,
  `g = (4/3)·π·G·ρ·R`, so `ceiling = 3S / (4πGρ²R)`. COMPUTED at 200 MPa and 2 700 kg/m³:

```text
   radius     surface gravity    relief ceiling      verdict
   3 351 km      2.53 m/s2           29.3 km         accepted
     500 km      0.377 m/s2         196.3 km         accepted, and absurd
     313 km      0.236 m/s2         313.5 km         the ladder refuses the body
     100 km      0.0755 m/s2        981.4 km         the ladder refuses the body
       5 km      0.0038 m/s2      19 627 km          the ladder refuses the body
```

  `Ladder::for_radius` returns `None` when `crust >= surface` (`ladder.rs:89`) and
  `crust_m = relief + strata + caves + 64` (`body.rs:234`). **So a bare ceiling deletes every moon and
  every asteroid under about 313 km of radius**, including the 10 km asteroid revision 1 used as its own
  worked example, and the landed test `BodyDefinition::from_seed(5, 3_000.0).expect("a rock")`
  (`body.rs:386`) would panic.
- **DOMAIN 02 already owns the ceiling and states it better.** It fixes `σ_y = 243 MPa` from Everest,
  checks Mars at 22 594 m against Olympus Mons' 21 900 m and Venus at 9 788 m against Maxwell Montes'
  11 000 m, and reads the relation as **a scale with a ±15 % spread, not a wall**. Refuter B is also
  right that Everest is set by uplift against glacial erosion and Olympus Mons is a volcanic construct
  on a cold lithosphere, so the relation is an ORDER-OF-MAGNITUDE ceiling and only the RATIO tests
  anything. A per-body strength draw over 100–300 MPa on top of that would move the home planet's
  ceiling between 10.7 km and 32.2 km, which is a three-fold move on a number the ladder band reads.
- **The size term stays.** Nothing stands 39 % of its own radius high. `relief = min(share × radius,
  ceiling) × a seed factor` keeps the law total on every body, and it is DOMAIN 02's to state.

**What this domain keeps from the gravity idea.** Gravity is still the right physical input, and the
owner named it. It reaches the generator as **a quantised integer in DOMAIN 02's `BodyFacts` record**
(§3.1), never as a float from an unfenced crate. §12 D4-13 is now a vote for that ask, not a second
version of it.

**What gravity does NOT change**, so that a geologist is not offended: the angle of repose is a friction
property and it does not move with gravity. A scree cone on the home planet stands at the same 34° as
one on Earth. Only the ceiling moves.

### 4.4 The modulation: ONE per-column factor, and the band that carries the LINES

**Rule 1 — the amplitude is the body's spectrum times ONE per-column factor.**

```text
   a_i(dir) = A_i * r(dir)
   A_i      = DOMAIN 03's slope spectrum, normalised so the octaves sum to the fine relief share
   r(dir)   = r_rock(L1..L3) * damp(|macro gradient|) * channel_damp(dc)      in [0, r_max]
```

- `r_rock` is DOMAIN 02's `rough_q8`: hard rock and an active belt are rough, a basin fill is smooth.
- `damp(|∇macro|) = 1 ÷ (1 + c·|∇macro|²)` reads the MACRO gradient, which is rung-independent, so the
  rule stays a lawful multiplier. A place the macro stack already stands steep gains less; a valley
  floor keeps gaining. It is what makes the surface look eroded rather than crumpled.
- `channel_damp(dc)` falls toward the channel axis, so a fine octave cannot put a metre-high lump in
  the water.

*Why not the body's relief?* Because that is what the code does today, and it is the defect: the whole
body gets one amplitude, so a salt flat is as rough as a ridge.

*Why not the per-octave derivative damping of revision 1?* Because a coarse rung cannot evaluate it, so
it could not state its own bound; because it broke DOMAIN 02's binding multiplier rule in spirit; and
because refuter A proved the table revision 1 printed from it was not that recipe at all but an ensemble
average (see §4.8). One per-column factor keeps the physics and costs one divide per column instead of
one per octave.

**Rule 2 — the RIDGED, FLOW-ALIGNED BAND: where the LINES come from.** This is the largest new part of
revision 2, and it answers refuter B's strongest finding: under revision 1 the whole band from about
100 m to 3 km was isotropic noise, and that band is the reference picture's spurs, gullies and ridge
lines.

```text
   inside the band  lambda in [break_length, macro node]  (128 m .. 8 192 m on the home planet)

      value noise                          ridged, flow-aligned
      n(x)                                 1 - |n(M x)|,  M stretches along the D8 flow direction

         ___     ___     ___                  /\      /\      /\        a CREST, and a rounded floor
        /   \   /   \   /   \                /  \    /  \    /  \       between two crests: a spur
       -     ---     ---     -              -    ----    ----    -      and a gully, not a lump
```

Three things make it lawful and cheap:

1. **The direction is DOMAIN 03's, not a new datum.** The D8 receiver is already one of the three bits
   per node in the 9.83 MB artifact, and the column already looks the node up for the channel distance.
   No new channel, no SL6 ask.
2. **It is prefix-safe.** The transform reads the octave's own sample and a per-column matrix, never a
   finer octave.
3. **It costs almost nothing.** The ridge transform is one absolute value and one subtract. The
   stretch is three multiplies per sample. COMPUTED at 10.5 ns per sample per octave today, the stretch
   adds an ESTIMATED 30 % to the six octaves in the band: `6 × 43.0 µs × 0.3 = 77 µs` per chunk.

**What it buys, and the number that is still short.** DOMAIN 03 measured the gap: the recipe spends
47 m of amplitude at 3 km and 23 m at 1.6 km, and *"the reference picture needs 300–600 m there"*. Its
slope spectrum answers the AMPLITUDE (606 m at 6.25 km, 377 m at 3.13 km, 151 m at 1.56 km). This
domain's ridged band answers the SHAPE — that the amplitude arrives as crests and gullies rather than
as lumps. **Neither alone is enough, and this document says plainly that the two must be taken
together.**

**One cross-domain check this domain owes back, with the arithmetic.** Read as sinusoids, DOMAIN 03's
three published amplitudes give per-octave root-mean-square slopes of `2π·a/(λ·√2)` = 0.431, 0.540 and
0.419. Summed in quadrature those three alone reach 0.808, which is **38.9°** — above the angle of
repose, from three octaves out of eleven, before this domain's `r` multiplies anything. Either the
spectrum's peak is high by about a factor of two, or the two documents mean different things by
"amplitude at a wavelength". **This is not a refutation; it is a question that must be answered before
either table is built. M4-13, and §13 item 4.**

**Rule 3 — the spectral break belongs to the spectrum, not to a second gain.** Revision 1 carried two
gains, one above a 128 m break and one below it. DOMAIN 03's spectrum already falls away below its peak,
which is the same physics (hillslope diffusion) stated once. This domain keeps only the REQUIREMENT: the
spectrum must fall below the hillslope length, and `break_log2` is the channel DOMAIN 02 states it on.

**Where every constant comes from.** This is the honest table, and it is shorter than revision 1's
because three of its rows moved to a neighbour.

| Number | Value or range | Where it comes from | Status |
|---|---|---|---|
| `C`, cells per wavelength | 8 | the ladder, and the free-coarsening identity (§4.3) | argued; **M4-3** is the gate, taken with DOMAIN 02's `SHORT_WAVE_M` |
| `A_i`, the octave amplitudes | DOMAIN 03's slope spectrum | erosion peaks the roughness at the valley spacing | **DOMAIN 03's, and its own M9 is the gate** |
| `r_max`, the modulation ceiling | seed draw, range OWED | none yet | **OWED a source; M4-4** |
| `c`, the macro-slope damping | seed draw, range OWED | chosen so the slope histogram lands where real alpine ground is | **M4-4 is the gate** |
| the ridged band's ends | `[break_log2, macro node]` | both are channels the neighbours already state | derived |
| `q`, the terrace strength | seed draw in `[0.4, 0.7)` | §4.5 derives the Lipschitz constant it implies | **COMPUTED, §4.5** |
| the fine floor amplitude | per biome, `[0.1, 0.5]` m | the topsoil depth the strata already draw (`body.rs:196`) | §4.7 |
| the relief and its ceiling | **DOMAIN 02's** | a physical relation read as a scale, plus a size term | §4.3 |

**Domain warping** fits in the same fold and is also prefix-safe. RECOMMENDED as a second step, after
the pictures, because it costs two more noise evaluations for every octave it warps: COMPUTED at
`2 × 10.5 ns × 4 096 = 86 µs` per warped octave.

### 4.5 Terraces: the strata make the cliffs, WITHOUT touching the veneer

Today the strata are ONE topsoil (1–4 m), ONE subsoil (2–8 m), ONE sediment band (20–80 m) and then
bedrock forever, and `strata.at` picks the substance by DEPTH under the surface
(`strata.rs:166-178,188-209`). A depth band follows the hill, so it can never make a bench.

**Revision 1 proposed a stack that REPEATS in radius forever. That is refuted and withdrawn.** Refuter B
found the fatal consequence in the code's own words: `max_depth_m()`'s contract is *"the deepest metre
the strata change at; below it a cell is bedrock without a lookup"* (`strata.rs:165-169`), and the
below-skip writes a whole chunk without a cell pass only when every cell is bedrock (`chunk.rs:12-17`).
MEASURED: one radial column at rung 0 holds 470 chunks and **463 of them are skipped**, and it still
costs 417 ms (`slice_05_generator.md:387-389`). A stack that changes substance forever downward has no
`max_depth_m()` at all, so those 463 chunks would each run a full cell pass over 238 328 cells.
COMPUTED at the measured cell-pass cost of 579 µs per chunk: **about 268 ms more per radial column, a
64 % rise**, and it breaks a landed invariant with its own test.

**The proposal, rebuilt: the SHAPE is radial, the SUBSTANCE stays in the veneer.**

```text
   depth bands today                     the proposal
        _____                                 _____
       /     \                               /     \        hard band top  <- the terrace pulls here
      /       \___                       ___/_______\___    soft
     /            \                     /               \   hard band top  <- and here
    ---------------- topsoil            -----------------   soft
    the layer bends with the hill        the HARDNESS is flat; the hill cuts it,
                                         but the SUBSTANCE is still the veneer
```

1. **A radial HARDNESS function, not a table.** `H(r − datum_m)` is a pure function: the band index is
   `floor((r − datum_m) / band_thickness_m)`, and the hardness is a fenced integer hash of that index
   and the seed, read as a `q8`. There is no table to bound, no memory, and it is defined at every
   radius. `datum_m` and the dip come from the macro stack (§3.1), so flat-lying bands in a basin give
   mesas and tilted bands in a belt give hogbacks.
2. **The terrace reads it per COLUMN, and shapes the height.** That is the whole of the cliff-band look,
   and it costs one hash and one smoothstep per column.
3. **The SUBSTANCE keeps the veneer.** `strata.at(biome, depth)` gains the band index as a third
   argument and uses it **only within `max_depth_m()`**. Below that depth every cell is bedrock, exactly
   as today. So `max_depth_m()` does not change, `crust_m` does not change, **the ladder band does not
   move**, the below-skip keeps its 463 chunks and its landed test still passes. This is also what
   DOMAIN 03 asks for by name: *"the band does not move … revision 1's proposal to grow the crust is
   DELETED"*.
4. **The per-cell cost is charged.** Refuter A is right that revision 1 charged the terrace per column
   and never charged the substance per cell. The band index inside the veneer is a subtract, a multiply
   by a reciprocal, a floor and a mask. COMPUTED at 2 ns per cell over the cells a surface chunk's cell
   pass already visits: **+0.48 ms per surface chunk**, and §9.2 carries it.
5. **The soil is stripped on the MACRO slope, not on the fold's slope.** Soil does not stay on a 60°
   wall, so `topsoil_m` and `subsoil_m` are multiplied by a falloff that reaches zero above the angle of
   repose. **The falloff reads the MACRO gradient only.** Refuter A found the trap in revision 1:
   stripping on the fold's own gradient makes the SUBSTANCE differ by rung, so a mesa flank would be
   bare rock at rung 0 and soil at rung 3, and §7's stratum byte would pop the material on exactly the
   cliffs the picture is made of. The macro gradient is rung-independent, so the substance is the same
   at every rung and no bound is owed.
6. **The terrace itself, with a falloff that reaches zero.**

```text
   S           = the spacing between two HARD band tops        (two thicknesses or more)
   d           = h - nearest hard top
   u           = |d| / (S/2)                                   so u runs over the WHOLE [0, 1]
   terrace(h)  = h - q * H * d * f(u),   f(u) = 1 - u^2*(3 - 2u)
```

**Why the half-spacing matters, and what revision 1 got wrong.** Refuter B proved that with `u` measured
against ONE thickness the falloff never reaches zero before the next band top, so the height JUMPS
where the nearest top changes: COMPUTED at `q = 0.6`, `H = 1` and a 40 m band, the jump is **12 m**,
which is six cells at rung 1 and an unbounded `Lip(T)`. With `u` measured against HALF the spacing,
`f(1) = 0` and `f'(1) = 0`, so the pull fades to nothing at the midpoint from BOTH sides. The terrace is
then continuous and smooth, and the derived constant is measured over a range the function can reach.

**The Lipschitz constant, DERIVED over the reachable range.** `d(terrace)/dh = 1 − q·H·(1 − 9u² + 8u³)`,
whose extreme over `u ∈ [0, 1]` is at `u = 0.75`, giving `1 + 0.6875·q·H`:

| `q` | `max |d terrace / dh|` |
|---|---|
| 0.4 | **1.275** |
| 0.6 | **1.413** |
| 0.8 | **1.550** |

**The terrace's rung rule.** The terrace runs at a rung only while the hard-band spacing `S` is at least
`C` cells of that rung, and over the last rung it FADES: `q` is scaled linearly to zero as `S/cell`
falls from `2C` to `C`. So the terrace never appears or vanishes between two rungs, and its own
contribution to the coarse-answer bound is the fade's own step, not the whole bench. COMPUTED on the
home planet with `S = 200 m` and `C = 8`: the terrace is full strength to rung 4 (16 m cells), fades
through rung 4 to 5, and is gone by rung 5 (32 m cells). §5.2 carries it.

**The extractor consequence.** Ruling V10 records that surface nets round a crease by half a cell of the
rung drawn, and that dual contouring keeps it sharp. A terrace lip IS a crease. So the terrace is the
strongest argument on the extractor question, and the pictures that decide it (ruling V10, slices 7 and
8) should include a mesa.

### 4.6 The channel carve, and what "the river is cells" really buys

A channel is sharp. It must not be interpolated on a node lattice, or the river blurs into a dent. It is
evaluated per column, like a tube cave.

```text
   gather ONCE per REGION                  pay per column only if the region kept a segment
   +-----------------------------+        +--------------------------------------------+
   | the macro cells the REGION  |  --->  | dc = the least distance to the kept segments|
   | touches, plus the valley    | prune  | w  = the channel's half width there         |
   | width; at most one channel  |        | h  = blend(bed, h, smoothstep(dc / valley)) |
   | per node (D8)               |        | and the octaves were damped by dc already   |
   +-----------------------------+        +--------------------------------------------+
       the same shape as `tubes_near` (`carve.rs:127-146`, `chunk.rs:320-370`)
```

Five properties, and refuter B corrected two of them.

1. **The gather is per REGION, and its window is a function of the rung.** Revision 1 wrote "at most
   nine segments" as if it were general. COMPUTED: a chunk is `62 × 2^L` metres, so it passes the
   8 224 m macro node at **rung 7** and spans 15.5 macro cells at rung 11. The window must therefore be
   stated per rung, and the gather must be per REGION — which is exactly how the tube carver earns its
   agreement across a chunk edge. A halo column gathered by a neighbouring chunk must find the same set,
   or the two meshes crack.
2. **A chunk far from every channel pays nothing.** The prune compares the region's bounding sphere
   against each segment plus its valley width (`chunk.rs:356-364`). At rung 0 a chunk is 62 m and a macro
   cell is 8 224 m, so the prune usually keeps at most one or two segments. §9 prices it from that.
3. **The bed is the floor, and the carve is LAST.** The carved height never goes under `bed_m`, and no
   later term may lift it, so the channel stays monotone downstream.
4. **The rung rule reads the VALLEY, not the water.** Revision 1 said "a channel is cut at a rung only
   while its own width is at least `C` cells", and refuter A showed that at `C = 16` this deletes every
   headwater stream, because a 4 m half-width channel is 8 m wide and 8 m is never 16 cells. The rule is
   corrected: **the VALLEY is what is cut at a rung while its width is at least `C` cells**, and a
   valley is tens to hundreds of metres wide even where the water is 4 m. At `C = 8` the water's own
   8 m trench is cut at rung 0 (8 cells) and dropped above it, and its contribution to the coarse-answer
   bound is its own depth — a metre or two, which is under the gate by §5.2's arithmetic.
5. **The water IS cells, and here is exactly what that buys and what it does not.** `fluid_at` returns
   `Stratum::Water` under `sea_radius_m` (`chunk.rs:207-213`) and `finish_cell` writes it
   (`chunk.rs:633-635`), so the sea is cells today. `fluid_at` gains the local water surface —
   `fluid_at(body, r, water_m)` — and a river is cells like the sea. **That is HR3: one machinery, not
   two.** But refuter A is right that revision 1 oversold it:
   - the extractor's only surface is the rock/air sign (`extract.rs:70-74`), and a water cell has
     `gap ≥ 0`, so **it emits no triangle**;
   - nothing outside `vd-terrain` reads `Stratum::Water` today (a grep over `crates` finds it only in
     `terrain/src/{chunk,strata}.rs`), and **no buoyancy exists anywhere**.

   So the bed COLLIDES, because it is rock, and the water is DRAWN by DOMAIN 03's second extractor run
   over the water field (its item 9). **A pilot who ditches in the river today lands on the bed.**
   Floating is not built, it is not this domain's, and §13 item 9 names it as owed.

### 4.7 The fine floor, the surface form, and the honest limit at the metre

**The limit, stated first.** Nothing in this document makes height detail between 1 m and 8 m, and the
fix is not more octaves: at the fine end the octave amplitudes are tenths of a metre, so an extra octave
at 4 m would carry a few centimetres and would be invisible. **On 1 m cells the finest height feature
that reads cleanly is about 8 m across, and that is a property of the cell, not of the table.** Refuter B
called this the strongest paragraph of revision 1 and it stands unchanged.

So the metre-scale ground comes from four places, and each is named:

| What the eye needs at 1–8 m | Where it comes from | Status |
|---|---|---|
| Hummocks, frost heave, rills, a soil that is not glass | **the fine floor**, one biome octave at 8 m | this document, below |
| Dunes, striation, badland rills | **the surface form**, one biome-gated term | this document, below |
| Boulders, talus, rock outcrops, stream cobbles | **placed rock objects**, with their own colliders, on the block and attachment machinery | **NOT this domain; §13 item 8** |
| Grass, trees, decoration | **art assets**, blended dynamically (ruling V4) | not this domain |

**The fine floor**, with a law and a number. Amplitude = a per-biome fraction of the topsoil depth the
strata already draw (`body.rs:196`, 1–4 m), wavelength 8 m:

| Biome | Amplitude | The real thing it stands for |
|---|---|---|
| Desert | 0.10–0.25 m | ripple and lag gravel |
| Grassland | 0.10–0.30 m | tussock and bioturbation |
| Tundra | 0.20–0.50 m | frost heave, thufur |
| Highland | 0.30–0.60 m | blockfield |

COMPUTED cost: one octave, **43.0 µs** per sample box, at rung 0 only.

**It is dropped without a crossfade, and the arithmetic says that is safe.** An 8 m wave is 8 cells at
rung 0 and 4 cells at rung 1, so it lives at rung 0 only. Its amplitude is at most 0.6 m, which is
**0.6 of a cell** — and §5.2's gate is stated in cells, so it is 0.6 on the same scale as every other
term, well inside one cell. It needs no band.

**The surface form** is one biome-selected extra term, and it is what makes a desert read as a desert.

| Biome | The form | How |
|---|---|---|
| Desert, sand | dunes | one anisotropic wave along the LAW's wind direction (§3.3), with a sharp lee face |
| Karst, limestone | sinkholes | a sparse negative cone field on the caves' own node lattice |
| Glacial | striation and moraine | a directional low-amplitude ridge field; the ice flow direction is DOMAIN 03's, and this term waits for it |
| Badlands | rills | a high-frequency ridged term, gated to steep soft rock |

Each is one term, gated by the biome, so a body with no desert pays nothing for dunes.
**RECOMMENDED: build dunes and rills in the first slice.** They are the two the reference picture shows,
and both read their direction from a law, not from a channel.

### 4.8 The numbers, and the METHOD that produces them

**Revision 1's table is withdrawn.** Refuter A reproduced it exactly and proved it was not the recipe
§4.1 described: the damping in it was a STATISTICAL one over the accumulated variance, not the pointwise
gradient the text named, so the table described an average column and a slope histogram would not
reproduce it. Refuter B showed the undamped comparison figure did not reproduce either: from the same
stated gains the undamped added slope is **56.7°**, not the 44.3° revision 1 printed. Both are right,
and rather than print a third table this domain now states the METHOD and the CONSTRAINT, and leaves
the amplitudes to DOMAIN 03's spectrum (§3.4).

**The method, stated once so a reader can check any number in this family.**

```text
   per-octave slope, RMS      s(o) = 2*pi*a(o) / (lambda(o) * sqrt(2))      a sinusoid's RMS slope
   the column's added slope   S    = sqrt( SUM over o of ( s(o) * r )^2 )   octaves are independent
   the angle                  theta = atan(S)
```

**The constraint this domain places on whatever spectrum DOMAIN 03 lands.**

| Landform class | `r` | Required p50 added slope | Why |
|---|---|---|---|
| Alpine belt | near `r_max` | **28°–34°** | above 34° nothing walks and no scree stands; below 28° it is not alpine |
| Hills | mid | 8°–16° | the reference picture's middle ground |
| Plain | near 0 | under 5° at p99 | a farm plain must be flat, and today the whole planet is 1.6°–2.0° |

**M4-4 is the gate**, and it fixes `r_max` and `c` by measurement, not by argument. Until it runs, no
number in this family may be quoted as COMPUTED.

**What is already known and does not need M4-4.** MEASURED today, from DOMAIN 02 §1.2: the home planet's
RMS slope is 1.6° to 2.0° at every baseline from 2 m to 8 km, and the steepest slope in 4 900 samples is
7.1°. Any `r` field that separates a plain from a belt is an improvement on one number for a whole
planet; the question M4-4 answers is how far it may go before the ground stops being walkable.

---

## 5. The free-coarsening law, restated

### 5.1 The prefix property

**The claim.** Write the fold as `s_{i+1} = s_i + A_i·r(dir)·n_i(dir)`. Then
`h(dir, L) = R_L(T_L(s_{n−L}))` and `h(dir, 0) = R_0(T_0(s_n))`, where `T` is the terrace and `R` is the
channel carve, and both read only the state handed to them.

**The consequence.** Rung `L` runs the SAME code as rung 0 and stops `L` steps early. It is not an
approximation of rung 0. It is a prefix of it.

**What breaks it.** Any term that reads a FINER octave. Four examples, each easy to write by accident:

- a normalisation over the whole octave sum;
- a multiplier that reads the accumulated octaves (revision 1's derivative damping, §4.4);
- a channel distance found after the loop instead of before it;
- a soil-stripping falloff on the fold's own gradient instead of the macro gradient (§4.5).

**The fence.** The fold is one function, the accumulator is passed forward, and the modulation is
computed once before the loop. A test evaluates rung `L` two ways — the direct call, and rung 0's fold
halted at step `n − L` — and asserts equality.

### 5.2 The bound, per column, and the gate in CELLS

```text
   |h(dir,0) - h(dir,L)|  <=  Lip(T) * r(dir) * SUM over i in [n-L, n-1] of A_i    the octaves
                          +   the terrace's own fade step at its last rung          the terrace
                          +   |R_0(dir) - R_L(dir)|                                 the carve
```

**Every term is STATABLE by the coarse rung.** `Σ A_i` over the dropped octaves is a per-body constant
the body already holds; `r(dir)` is one number the coarse rung computed before its loop; the terrace
holds the `q` it faded to; and the carve holds the depth it applied. That is the whole payoff of the
per-column modulation, and it is what revision 1's per-octave damping made impossible.

**The gate is the bound in CELLS of the rung, and it is resolution-independent.** Refuter B found this
and is right that the owner should be told: the client draws rung `L` out to a distance proportional to
`cell_m(L)`, and one pixel subtends a fixed angle, so the two cancel and every "pixels" figure is the
bound divided by the cell, times a constant near one. A finer screen moves the switch distance out by
the same factor it shrinks the pixel by.

**The law, scale-free.** The dropped sum is dominated by its coarsest term, `Σ_{i≥m} A_i ≈ A_m/(1−k)`,
and `A_m = s_m·λ_m/2π` with `λ_m = C · cell_m(L)`. So

```text
   bound / cell  =  Lip(T) / (1 - k)  *  C / (2*pi)  *  s_m
```

COMPUTED at `Lip(T) = 1.413`, `k = 0.45` on the spectrum's falling limb and `C = 8`:
**`bound / cell = 3.27 × s_m`**, where `s_m` is the per-octave slope of the rung's finest live octave.

| `s_m` at the finest live octave | bound in cells | pixels, code's cube convention (×0.58) | pixels, half-metre convention (×1.00) |
|---|---|---|---|
| 0.1 (5.7°) | 0.33 | 0.19 | 0.33 |
| 0.3 (16.7°) | 0.98 | 0.57 | 0.98 |
| 0.6 (31.0°) | **1.96** | **1.13** | **1.96** |

The two pixel columns are the two readings of the code's own drawable factor: a 1 m cell taken as a
half-metre extent reaches 869 m, and taken as a cube (extent `√3/2`) it reaches 1 505 m
(`crates/core/src/geometry.rs:1156-1173`). **The residency rule that picks between them is slice 8's,
not this domain's**, which is why the gate is stated in cells.

**The comparison with today, corrected.** Revision 1 said "tighter everywhere" and quoted today's
rung-3 bound as 6.11 m. That is the value at `k_rough = 0.5`. COMPUTED at the roughness the home planet
actually drew (`k = 0.468 338 1`, relief 14 304.887 m, 14 octaves):

| rung | today's `dropped_bound_m` | today's bound in cells of that rung |
|---|---|---|
| 1 | 0.40 m | 0.20 |
| 3 | 3.05 m | 0.38 |
| 6 | 32.76 m | 0.51 |
| 9 | 321.97 m | 0.63 |
| 11 | 1 469.16 m | 0.72 |

Today's ladder sits at half a cell to three quarters of a cell everywhere, **because the ground is
uniformly gentle**. A proposal that puts a real alpine slope on the mountains lands at 1.96 cells there
and at 0.33 cells on the plain. **So the honest sentence is: the bound tightens where the ground is
smooth and loosens where it is steep, and the loosening is the price of the picture.** Refuter A is
right that §0 said the opposite, and §0 now says this.

**Two other uses of the bound:**

1. **The ladder band.** `BodyDefinition::from_seed` derives the band from the relief
   (`body.rs:233-236`). §4.3 and §4.5 both leave the relief and `max_depth_m()` alone, so **the band does
   not move** and no address changes. That is also DOMAIN 03's requirement.
2. **The crossfade (SL8).** A per-column bound lets the band stay narrow on a plain and go wide on a
   ridge, which is what "no visible jump" needs without paying everywhere.

### 5.3 The rung where the octaves vanish

With eleven octaves the home planet's rung 11 holds none. The height there is the macro stack, `Z` and
the erosion correction alone. COMPUTED cost at rung 11: the fixed 144 µs, one macro evaluation and one
bicubic per column — **1.8 ms at DOMAIN 02's 416 ns per column, and 2.9 ms at its 672 ns** — against
266 µs today. **The coarse rungs get MUCH more expensive, not cheaper**, because the macro stack is not
free. Rung 11 is still cheaper than rung 0, so the landed ladder assert holds
(`slice_05_generator.md:383-386`), but refuter A is right that an eight-fold rise on the rung that draws
the most chunks in a vista is a fact the owner must be given, and M4-1 must print the whole curve.

```text
   rung   cell     finest live octave      what carries the shape
   ----   ------   ------------------      -----------------------------------------------
    0       1 m    8 m    (8 cells)        macro + Z + 11 octaves + fine floor + terrace + carve + 3-D
    3       8 m    64 m   (8 cells)        macro + Z + 8 octaves + terrace + carve
    6      64 m    512 m  (8 cells)        macro + Z + 5 octaves + carve
   10    1 024 m   8 192 m (8 cells)       macro + Z + 1 octave
   11    2 048 m   none                    the macro stack and Z alone
```

### 5.4 What it means for the vista

**First, an honest constraint the owner should know.** COMPUTED on the home planet's ladder radius, the
ground horizon from an eye 1.8 m up is `√(2·R·h) = 3 473 m` — **3.5 km, 27 % closer than Earth's
4 789 m**, because the planet is small. A 50 km vista is a HIGH-VANTAGE case: from a ridge, from a
hull, or looking at a 14 km peak from far away. The chunk counts below are that case.

COMPUTED chunk counts for one shell of each rung, on the surface only:

```text
   rung 0  out to    500 m :  204 chunk columns   (62 m per chunk)
   rung 3  out to  5 000 m :  316 chunk columns   (496 m per chunk)
   rung 6  out to 50 000 m :  494 chunk columns   (3 968 m per chunk)
   -------------------------------------------------------------
   about 1 014 surface chunk columns
```

**The vertical stack.** A surface chunk column crosses more than one chunk wherever the relief inside the
column's own footprint passes one chunk height. COMPUTED at rung 0: a chunk is 62 m across and 62 m
tall, so a column crosses two chunks wherever the ground falls more than 62 m over 62 m — a slope over
45°. On alpine ground the mean crossing is ESTIMATED at 1.6 to 2.0 chunks, and on a plain it is 1.0.
Revision 1 wrote "about 2 000 chunks" with no derivation; refuter A was right to ask. **The honest range
is 1 000 to 2 000 chunks, and M4-2 measures it.**

**Time.** At the MEASURED client cost of 4.18 ms per chunk on one thread, 1 000 to 2 000 chunks is
**4.2 s to 8.4 s on one thread** and, at the MEASURED 2 004 chunks/s on 14 threads, **0.5 s to 1.0 s**.
Every delta in §9.2 lands on the client too, so this is a floor, not an answer.

**Bytes, and this is the harder number.** MEASURED (`slice_07_client_link.md:236`): 405 228 bytes per
rung-0 chunk at 8 577 vertices and 16 615 indices, which decomposes exactly as
`8 577 × 12 + 8 577 × 12 + 16 615 × 12` — positions, normals and indices, four bytes a component. So the
vista is **400 MB to 810 MB of mesh data**, and this proposal's rougher ground raises the vertex count by
an ESTIMATED 1.4 to 2.9 times. **This domain does not solve that.** It hands slice 8 three things: the
chunk counts, the vertex multiplier from M4-2, and the lever, priced correctly this time:

| Lever | Bytes per chunk | COMPUTED |
|---|---|---|
| today | 405 228 | `8 577×12 + 8 577×12 + 16 615×12` |
| `i16` positions and `u16` indices | **254 076** | the two changes revision 1 named |
| plus normals packed to 4 bytes | **185 460** | the figure revision 1 quoted, which needs this third change |

Refuter A is right that revision 1 quoted 185 KB for two changes that reach 254 KB. (`ChunkMesh` already
stores `[i16; 3]` vertices, `extract.rs:67`, so the position half is a client-side choice.)

**And the artifact, which revision 1 priced at nothing.** DOMAIN 03 ships `Z` and the river graph as a
**9.83 MB artifact per body**, with a coarse pyramid whose top level is 19 KB. The client must hold the
19 KB on approach and the full 9.83 MB for the body it lands on, **before it draws its first chunk**. A
pilot who flies to a second planet holds a second one. §13 item 3.

**The orienters.** A ridge line, a mesa, a scarp and a trunk valley are all macro-stack or `Z` features,
so they are present at EVERY rung, the coarsest included. Between them, the ridged flow-aligned band
(§4.4) carries the spurs and gullies down to the hillslope length. That is the structural answer to "no
orienters", and it is an answer about LINES at two scales, not about octave counts.

---

## 6. From the height to the cells

### 6.1 The gap byte, and the slope correction

Today `gap = ((r − h) / cell).clamp(−1, 1)` (`chunk.rs:630`).

**The defect is the CLAMP, not the ratio.** The extractor's crossing is a RATIO of two gap magnitudes
(`extract.rs:182-187`), so on a RADIAL edge both cells belong to the same column, a per-column slope
factor divides out exactly, and no vertex moves. But on a 70° face the neighbouring column's surface is
`tan 70° = 2.75` m away in radius per 1 m of lateral step, so the LATERAL neighbour's gap saturates at
±1 cell and the ratio degenerates toward `m0 / (m0 + 127)`. That is a systematic bias, on exactly the
cliffs the owner's picture is made of. Dividing by the slope factor un-saturates it:

```text
   gap = (r - h) / (cell * sqrt(1 + |grad h|^2))
```

`grad h` is already in hand from the fold. `Gf::sqrt` is fenced and already used (`gf.rs:101-105`,
`carve.rs:160`). The factor is a column property, so it costs one square root per column. COMPUTED:
about 20 ns times 4 096 columns, so **82 µs per chunk**.

**Four consequences, and revision 2 adds two of them.**

1. **The cave hollow folds into the SAME byte** by `greater` (`chunk.rs:646`) and is a true radial hollow
   in metres. The hollow must be scaled by the same factor, or the two halves are in different units.
2. **The bound must be restated for the byte, not for the height.** The gradient differs by rung, so
   §5.2's bound does not by itself bound the DENSITY BYTE. That is new work, and it is in M4-9.
3. **`seat_eighths` is a SECOND reader** (`seat.rs:32-62`). It computes a sub-metre block's seat from
   the ratio `8·|g| / (|g| + up)` over the cell and exactly ONE neighbour. That neighbour is the cell
   ABOVE or BELOW — the same column — so the per-column factor divides out there too, exactly as it does
   on a radial extractor edge. **The argument holds, and revision 1 never made it.**
4. **On a LATERAL edge the two cells belong to two columns with two different factors**, so after the
   correction the crossing ratio interpolates two quantities in slightly different units. The clamp is
   un-saturated, which is the gain; a new per-edge inconsistency appears, which is the cost. **M4-9 must
   measure both.**

**This changes the meaning of the density byte**, which ruling V6 row B-3 already APPROVED as *"the
signed radial gap at the cell centre, 1/128 of a cell"*. **The owner must be told he is being asked to
reverse an approved row.** And the gap byte is folded into the world identity (`digest.rs:38-42`), so
this is a world epoch **on the day it lands**, not only when Format B freezes. **D4-8 reads MEASURE
FIRST (M4-9), then decide.**

### 6.2 The 3-D terms: what they can and cannot make

A height field cannot make an overhang: along one direction there is one surface. DOMAIN 02 says the same
thing about its own layers by name: *"no layer in this report can make a rock pillar, an arch, a sea
stack or an undercut cliff"*. So this term is this domain's, and nobody else's.

| Feature | Needs 3-D? | How | Status |
|---|---|---|---|
| A cliff, a mesa wall, the shaft of a rock pillar | **No** | The height jumps between neighbouring columns. On 1 m cells that is a vertical wall, which is right. | §4.5 |
| An arch, an undercut, a cave mouth in a cliff | **Yes, a REMOVAL** | A 3-D term in the band around the surface, on the cave lattice's own machinery | RECOMMENDED |
| A caprock cap wider than its shaft | **Yes, an ADDITION** | The cave machinery cannot do it | **WITHDRAWN** |
| A cave | already built | `carve.rs`, the cavern lattice and the tubes | landed |

**The caprock cap is withdrawn.** The fold at `chunk.rs:641-649` takes `greater` (`:646`) and sets
`Stratum::Air` (`:647`): it can only REMOVE. Rock standing where the height field says air cannot be put
there by any `greater` rule. Adding rock needs a second fold with the opposite comparison and its own
stratum, bounded so it can never fill a cave — new machinery, and it needs the owner's word.

**The removal term.** ONE more node lattice, of exactly the shape `NodeLattice` already has
(`chunk.rs:435-520`), evaluated only inside a stated band around the column's surface, and only where the
macro state allows an undercut (hard rock, a steep macro slope, the right biome). Three reasons to reuse
that machinery: it is proved continuous across every chunk edge and every face seam
(`chunk.rs:359-364,508-520`); it is skipped for free outside its band (`CaveRule::in_band`,
`chunk.rs:611-614`); and it costs what the caves cost — **which is the one number this document may not
guess. §9 marks it UNKNOWN and M4-11 measures it.**

**The rung range.** An undercut is cut at a rung only while its own size is at least `C` cells of that
rung, and it FADES over the last rung exactly as the terrace does (§4.5), so it never appears or
vanishes between two rungs. At `C = 8` an 8 m arch lives at rung 0 alone and fades out through it. Its
own bound is its depth, and a fade over a rung keeps that inside the §5.2 gate.

**RECOMMENDED stride: measure 2 and 4 before choosing.** An arch needs 2 m features and a 4 m node grid
cannot make one, but halving the stride costs eight times the nodes. Ruling V6 D-3 already says the cave
lattice step is decided on pictures; the undercut step joins that decision. **M4-5.**

### 6.3 What does not change, and what does

**Does not change:** the extractor's topology, the halo, the seam rule, the corner prism, the vertex
quantum (`extract.rs:37`), the composition order, the digest shape — **and, in revision 2, the LADDER
BAND**. §4.3 leaves the relief law to DOMAIN 02, which keeps a size term, and §4.5 keeps the substance
inside today's `max_depth_m()`. So `crust_m` (`body.rs:234`) is untouched, `floor_m` is untouched, and
**no cell's `k` index moves on any body**. Revision 1 proposed the opposite and DOMAIN 03 had already
deleted the same proposal; both refuters caught it.

**Does change:** the numbers the extractor reads, at every rung; the octave table's ends and count
(§4.3); the density byte's meaning, if and only if D4-8 is taken after M4-9; and one `u8` per vertex in
the mesh (§7).

---

## 7. The mesh and the biome: one addition, two withdrawals, one signature

| # | The proposal | Verdict |
|---|---|---|
| 1 | **The stratum id per vertex** — one `u8` beside `ChunkMesh::vertices` (`extract.rs:62-69`), taken from the group's lowest-index ROCK corner in the fixed corner order | **KEPT.** The client cannot texture a cliff band, a soil, a snow cap and a river bank apart without it, and §4.5's macro-slope soil-stripping rule is what makes the byte tell the truth on a cliff AND keeps it the same at every rung. COMPUTED cost: 5 650 bytes per rung-0 chunk, **+3.4 %** on the vertices and indices, **+1.4 %** on the 405 KB the client holds |
| 2 | The biome in the surface cell's record | **WITHDRAWN.** Base terrain has NO record (`topic_00_format_sitting.md:34-36`), so the row would mint a twelve-byte record for every surface cell of every planet — COMPUTED 46 to 92 KB per rung-0 chunk on ground nobody has touched. And the `object` byte is already the tree's shape (ruling V6 B-5). The local formulation exists: `biome_at` is a pure function both hosts compute |
| 3 | The seat rule's "tie" for an arch | **WITHDRAWN.** `seat_eighths` (`seat.rs:32-62`) reads the cell and exactly ONE neighbour, so an arch already gets a correct local answer today |

**Is the stratum id in the identity?** Ruling V10 S6-5 keeps normals and colours out, because the client
derives them. The stratum id is DERIVED from cells that are already in the identity, by a stated rule, so
it need not be pinned. **RECOMMENDED: pin it anyway**, in the mesh digest. It turns "the client drew the
wrong rock" into a gate failure instead of a picture the owner has to catch.

**★ The biome signature, which revision 1 froze by accident.** §3.1 requirement 4 keeps ONE biome
source. Refuter B is right that today's signature has no room for what the owner asked for. The
requirement is therefore stated in full:

```text
   today   biome_at(body, dir, surface_m)                  latitude about +Z, height, two noises
   owed    biome_at(body, facts, macro, dir, surface_m)    where facts carries the obliquity, the
                                                            spin, the insolation and the atmosphere
                                                            (DOMAIN 02's BodyFacts), and macro
                                                            carries the upwind ridge and the
                                                            distance to open water
```

**DOMAIN 05 owns what fills it.** This domain owns only the rule that there is ONE of them.

**And the biome SET is too small for the picture.** The enum has four members — Desert, Grassland,
Tundra, Highland (`strata.rs`). The reference picture shows a forest MASS and cultivated fields, and
neither is any of the four. **Whoever owns the set must grow it before the art assets have anything to
key on. §13 item 11.**

---

## 8. The authored override: the pyramid, and the law it must obey

The owner will want to place a landmark: a named canyon, a crater, a mountain that a story needs. The
seed cannot be asked for it, and an edit at 1 m cells cannot build it.

**Revision 1 put the authored delta INSIDE the fold, below the octaves, so the octaves would roughen it.
That is refuted and withdrawn.** Refuter A read ruling S5-3: *"the far view draws the recipe's coarse
hill PLUS the change"* — the delta is added ON TOP of the generator's answer, after it. Putting live
state inside the recipe has three costs revision 1 did not carry: `height_m` stops being a function of
`(seed, address)` on both hosts, which is SL10 clause 1; the world identity's golden self-check
(`digest.rs`, `tag.rs:44-49`) is computed from the seed shape, so a body with an authored delta no
longer matches its own golden table; and every column pass must query the delta store.

**The proposal, rebuilt as TWO kinds, one law each.**

```text
   kind A  AN AUTHORED DELTA           live state. A pyramid entry at a coarse rung, holding a height
                                       delta in metres. Applied ON TOP of h(dir, rung), after the fold,
                                       one hop, never in the identity, exactly as ruling S5-3 says.
                                       The author gets a smooth hill with the terrain's own texture
                                       under it, because the fold ran first.

   kind B  AN AUTHORED LANDFORM        seed shape. A named kind and its parameters, held in the
                                       GENERATOR as a pure function of the address, evaluated inside
                                       the fold below the octaves, so the octaves roughen it.
                                       It is a world-epoch change and it enters the identity, because
                                       it IS part of what the seed decides.
```

**The rule that separates them, in one sentence:** *what the octaves must roughen is SEED SHAPE and rides
a world epoch; what may sit on top is LIVE STATE and rides the diff lane.* That keeps SL10 clause 1
whole, keeps the golden self-check meaningful, and gives the owner both behaviours without a fork.

**Composition.** Kind B lives inside composition row 1. Kind A rides the same lane as every other terrain
diff (`compose.rs:1-19`), and its arrival seam is §13 item 7.

---

## 9. The cost, and the bench that measures it

### 9.1 The unit costs we already own — and the two we do NOT

| Unit | Value | Source |
|---|---|---|
| One octave, per column-sample | **10.5 ns** — so **40.4 µs** per 3 844-column pass and **43.0 µs** per 4 096-column sample box | COMPUTED from 710 µs at 14 octaves and 266 µs at 3 (`slice_05_generator.md:374-380`) |
| The fixed part of a column pass | **144 µs** | the same regression |
| The sample box at rung 0, surface chunk | **1.98 ms** | MEASURED, `slice_06_extractor.md:348` |
| The sample box, the worst MEASURED chunk (+X rung 2, (21 223, 7), a seam, cave-dense) | **4.38 ms**, extraction **1.73 ms**, 8 234 vertices | MEASURED, `slice_06_extractor.md:352` |
| The cell pass alone, a surface chunk | **579 µs** for 238 328 cells | MEASURED, `slice_05_generator.md:381` |
| The extraction, per vertex above a fixed part of 973 µs | **61 ns** | COMPUTED from 1.32 ms at 5 650 and 2.39 ms at 23 084 — see the rule below |
| The macro layer stack L1–L5, per column | **416 ns** without rivers, **672 ns** with | DOMAIN 02's own revision-2 table |
| The halo | **+35 %** on the bare chunk | MEASURED, `slice_06_extractor.md:354` |
| ~~The cavern machinery: +0.85 ms~~ | **NOT AVAILABLE** | `2.83 − 1.98` subtracts a rung-0 chunk from a rung-2 chunk. **M4-11** is owed |
| ~~The undercut lattice: +0.60 ms~~ | **UNKNOWN** | Refuter B is right: revision 1 struck the cave cost as unavailable and then used it to price the undercut, and the whole budget verdict turned on it. **M4-11 measures both** |

**★ The subtraction rule, stated precisely, because revision 1 applied it inconsistently.** Refuter A
found that the same table forbids `2.83 − 1.98` and then uses `2.39 − 1.32` across the same two rows.
Both are subtractions across two rungs; only one of them is sound, and the difference is what the rung
does to the work:

- **Forbidden** when the rung changes the WORK PER UNIT. A rung-0 chunk sums 14 octaves and a rung-2
  chunk sums 12, and the cell is four times wider, so the cave field is sampled on a different lattice
  density. A cave delta taken across them measures three changes at once.
- **Sound** when the rung changes only the COUNT of units and the unit is the regressor. The extraction
  runs the same code per vertex; the rung changes how many vertices there are, which is the `x` of the
  regression. **61 ns per vertex is therefore ESTIMATED, not COMPUTED**, and **M4-2 confirms it.**

### 9.2 The estimate, built coherently at ONE rung at a time

Revision 1 built one line by applying rung-0 deltas to a rung-2 chunk. Refuter A showed three of those
deltas are switched off at rung 2 by this document's own rung rules. Here are two coherent lines, and a
range.

**Line 1 — the rung-0 surface chunk (MEASURED baseline 1.98 ms, `slice_06_extractor.md:348`).** Every
per-column row is priced over the sample box's 4 096 columns, once and for all (refuter A and refuter B
both caught the 3 844 / 4 096 mix).

```text
   sample box, MEASURED rung-0 surface chunk                                  1.98 ms   MEASURED
   octaves 14 -> 11 (-3 x 43.0 us)                                           -0.13
   analytic derivatives, +80 % on the 11 that remain                         +0.38     ESTIMATED
   the macro layer stack L1-L5 (416 ns x 4 096, no rivers)                   +1.70     DOMAIN 02
   Z: one bicubic and its gradient (15 ns x 4 096)                           +0.06     ESTIMATED
   the ridged, flow-aligned band (+30 % on six octaves)                      +0.08     ESTIMATED
   the modulation r: one divide and two multiplies per column                +0.02     COMPUTED
   the slope correction (one Gf::sqrt per column)                            +0.08     COMPUTED
   the terrace: one hash, one smoothstep per column                          +0.06     ESTIMATED
   the channel carve: two segments x 20 ns x 4 096 after the prune           +0.16     ESTIMATED
   the strata band index inside the veneer (2 ns x the cells the pass visits)+0.48     ESTIMATED
   the fine floor (one octave, rung 0 only)                                  +0.04     COMPUTED
   ---------------------------------------------------------------------------------
   sample box                                                                 4.91 ms
   the undercut lattice                                                       UNKNOWN   M4-11
   extraction at 12 000 vertices (1.32 ms + 6 350 x 61 ns)                    1.71 ms   ESTIMATED
   ---------------------------------------------------------------------------------
   TOTAL against the 8 ms budget                                              6.62 ms + UNKNOWN
```

**Line 2 — the worst MEASURED chunk, at ITS rung (4.38 ms box, rung 2).** At rung 2 the fine floor is
off, the undercut is off (an 8 m arch is 2 cells), and the carve cuts only valleys 32 m wide or more, so
its cost falls rather than rises. The macro stack, `Z`, the modulation, the slope correction, the terrace
and the strata index all still run.

```text
   sample box, MEASURED worst chunk (rung 2, cave-dense, a seam)              4.38 ms   MEASURED
   octaves 12 -> 9                                                           -0.13
   analytic derivatives on the 9 that remain                                 +0.31     ESTIMATED
   the macro layer stack, Z, r, the slope correction, the terrace            +1.92     as above
   the strata band index                                                     +0.48     ESTIMATED
   the channel carve at rung 2 (fewer valleys qualify)                       +0.08     ESTIMATED
   ---------------------------------------------------------------------------------
   sample box                                                                 7.04 ms
   extraction at 12 000 vertices                                              1.71 ms  ESTIMATED
   ---------------------------------------------------------------------------------
   TOTAL against the 8 ms budget                                              8.75 ms  OVER
```

**Three costs are still outside both lines, and each is owed:** the undercut lattice (UNKNOWN, M4-11);
domain warping, if it is taken (+0.09 ms per warped octave); and the macro stack at its river-bearing
672 ns (**+1.05 ms**, which by itself takes line 2 to 9.80 ms). **That last one is why D4-17 matters:
DOMAIN 02's river work and DOMAIN 03's carve must not both run.**

**The two levers, priced:**

| Lever | Saving | Where it lands |
|---|---|---|
| ONE lattice for the caves AND the undercut: a second threshold on the same nodes | the whole UNKNOWN row | it removes the risk rather than reducing a number |
| DOMAIN 02's L4 stops where DOMAIN 03's carve starts (D4-17) | −1.05 ms on the river-bearing line | it is a design decision, not an optimisation |

**The honest verdict: UNPROVEN.** Line 1 passes with room; line 2 passes only if the undercut is free,
which nobody has measured. Revision 1 said FAILED on a number that mixed rungs and used a struck cost;
that verdict is withdrawn, and it is replaced by a range and three named measurements. **M4-1, M4-2 and
M4-11 run before the fold is written.**

### 9.3 The measurements owed, in order

| # | What | How | The gate |
|---|---|---|---|
| **M4-1** | The real cost of the fold at every rung | extend `terrain_cost` with the new fold on the named chunk set, INCLUDING the worst measured chunk and the top rung | every chunk stays under 8 ms, at every rung |
| **M4-2** | The vertex count and the vertical stack on rough ground | extract an alpine chunk and a mesa chunk; report vertices, triangles, bytes, and how many chunks a surface column crosses | the extraction stays under 3 ms; the bytes and the stack go to slice 8 |
| **M4-3** | Aliasing at `C = 16`, `8` and `4`, together with DOMAIN 02's `SHORT_WAVE_M` | compare the rung-0 and rung-1 surfaces over 10 000 columns; report the p99 disagreement against §5.2's bound, and a picture at each | the measured disagreement stays under the stated bound, and no picture shows hash |
| **M4-4** | The slope histogram, and `r_max` and `c` | evaluate the fold over 100 000 columns in each landform class at three values of each | p50 alpine 28–34°, p99 plain under 5° |
| **M4-5** | The undercut lattice at stride 2 and stride 4 | the same bench row as the caves | the added box cost stays under 1 ms |
| **M4-6** | The seam SLOPE KINK | for each of the twelve cube edges, take the adjacent column pairs on either side and report the largest difference in the surface SLOPE across the edge, at rung 0 | the kink stays under the slope one vertex quantum implies over one cell |
| **M4-7** | The macro stack's real answer time | DOMAIN 02 owns it; this domain states the need: `macro_at` for a whole sample box | under 1.7 ms per chunk, which is its own 416 ns claim |
| **M4-8** | The no-drift legs | `just terrain-legs` with a golden table covering one column per landform class, a seam column, a corner column and a column inside a channel | every digest equal on every leg |
| **M4-9** | The slope-corrected density byte | extract a mesa chunk and an alpine chunk with and without the correction; report the distance from each vertex to the true surface, the byte's own coarse-answer bound, AND the lateral-edge inconsistency the correction introduces | the corrected form is measurably closer on both counts, or D4-8 is refused |
| **M4-10** | The home planet's own numbers | the bench prints the drawn `k_rough`, every octave amplitude, the drawn mass and the surface gravity | the "before" picture carries numbers, not ranges |
| **M4-11** | The cave machinery's real cost, and the undercut's | the SAME chunk with the term on and off, at the SAME rung | replaces two struck numbers; it gates §9.2 |
| **M4-12** | The coverage fixtures | for every new branch — the four surface forms, the undercut gate, the channel prune's kept and dropped arms, the terrace's falloff and fade arms, the corner rule, the veneer band index — a fixture that finds a live chunk for it | HR5: 100 % region and branch in `vd-terrain` |
| **M4-13 ★ NEW** | The spectrum's slope budget, across DOMAIN 03 and DOMAIN 04 | sum DOMAIN 03's published per-octave amplitudes as RMS slopes by §4.8's method | the total added slope at `r_max` lands in 28°–34°, or the spectrum's peak moves |

**The order the owner has set before** (rulings V9, V10, V12): the measurements run FIRST, then the code,
then the refuter.

### 9.4 The client's half, stated separately

The 8 ms is the SERVER's worker time per chunk. The client computes the same shape under SL10, and its
MEASURED cost is 4.18 ms per chunk on one thread (`slice_07_client_link.md:235`). Every delta in §9.2
lands on the client too, plus the positions and normals, which grow with the vertex count, plus the
9.83 MB artifact it must hold before it draws (§5.4). **This document does not propose a client budget.
It states that one is owed, and that it is slice 8's.**

---

## 10. No drift: every operation under the fence

| The operation | How it stays byte-identical | Status |
|---|---|---|
| The macro layer stack L1–L5 | DOMAIN 02 owes its own fence row; it is closed-form and it must name every operation | **OPEN, held by DOMAIN 02** |
| The erosion solve's arithmetic | DOMAIN 03 owes it. Its revision 2 answers it by SHIPPING the artifact: the solve runs on the server only, so the client never repeats it | **OPEN until the SL6 ask lands** |
| `Z`'s node values | integers, in metres, in the shipped artifact | by construction |
| The bicubic weights | cubics in `t`; `Gf` only (`+ − × ÷`), a fixed order, no FMA, no libm. `t` is NOT dyadic and does not need to be: identical rounding, not exactness, is the property | §4.2 |
| The bicubic across a seam | `vd_seed::seam::across`, the ONE crossing rule the halo uses | `lattice.rs:11-12` |
| The bicubic at a corner | the stated missing-node rule, TESTED for three-face agreement | a new test, the shape of `prism_vertex` |
| The octave amplitudes | `A_i × r`, one multiply, one order | §4.4 |
| The ridged transform and the flow stretch | one absolute value, one subtract, three multiplies | §4.4 |
| The analytic derivative | the same lerp chain as the value; `+ − ×` only; then one divide by `wave_m` and the tangent projection | `noise.rs:85-121`, extended |
| The channel distance | `segment_distance_m`, already fenced, already using `sqrt` | `carve.rs:160-175` |
| The terrace | one integer hash, one floor, one multiply, one smoothstep | §4.5 |
| The slope correction | one `Gf::sqrt` per column | `gf.rs:101-105` |
| The body's physical facts | quantised INTEGERS in DOMAIN 02's `BodyFacts`, from outside the fence, snapped inside it; a landed gate must move each fact by a thousand ulps and require the same body | §4.3; DOMAIN 02's M13 |
| The golden table | extended to one column per landform class, plus a seam column, a corner column and a channel column | `crates/terrain/tests/golden_home.txt`, `golden_home_mesh.txt` |

**No new float source enters this domain.** No libm call, no `powf`, no trigonometry, no external noise
crate. That is the same discipline slice 5 already proved on three legs, and the real x86-64 leg stays
owed as `D-TERRAIN-1`.

---

## 11. The law gates, one by one

| Law | Verdict | How this proposal stands against it |
|---|---|---|
| **SL10 — one generator, two hosts, no drift** | **OPEN, held by DOMAIN 02 and DOMAIN 03** | Everything in §4 to §6 is a function of `(seed, address)`, of the body's shipped facts and of the macro state, and the fold is ONE function in `vd-terrain` compiled into both hosts. But DOMAIN 02's arithmetic is not fenced yet, and DOMAIN 03's artifact is SHIPPED rather than derived, so the no-drift chain runs through a lane that is not open. M4-8 is the measurement |
| **The seed ruling (2026-08-27)** | **PASSED** | Everything the seed decides here is SHAPE: heights, slopes, layers, channels, biomes. A wiki may publish all of it and give nothing of value away. No ore, no deposit and no indicator material is added |
| **SL5 — one world** | **PASSED** | One fold, one macro stack, one table. No preset, no test-only variant, no scale knob. §4.3's octave law is total on a 10 km rock, and §4.3 withdraws the relief law that was NOT total |
| **No magic numbers** | **PASSED for what this domain still owns** | It deletes `LONG_WAVE_CAP_M` and `SHORT_WAVE_M` (`body.rs:24-25`) and replaces them with the ladder's own cell. The relief and its ceiling go back to DOMAIN 02, which derives them from a measured relation and a size term. Every remaining constant has a range and a named measurement in §4.4's table; two are marked OWED |
| **The 8 ms budget (V10, per chunk at any rung)** | **UNPROVEN** | §9.2: 6.62 ms on a coherent rung-0 line, 8.75 ms on the worst measured chunk's own line, both plus an UNKNOWN undercut. Revision 1's FAILED verdict is withdrawn as unsound, not as wrong |
| **The ladder (V8, V9)** | **PASSED, with one landed test changed** | §4.3 anchors the table to the ladder itself, so the count falls 11, 10 … 1, 0 and `every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it` passes as written ONCE the "never fewer than one octave" clamp is removed. §5.2's bound is per column, statable by the coarse rung, and gated in cells |
| **The record (12 bytes, the density byte, the biome and object param)** | **PASSED, after a withdrawal** | Nothing here mints a record. The density byte keeps its width and its 1/128-cell step; its DEFINITION changes only if D4-8 is taken, and D4-8 says MEASURE FIRST and states plainly that it reverses approved row B-3 and opens a world epoch on the day it lands (`digest.rs:38-42`) |
| **The edit pyramid as the override** | **PASSED, after a correction** | §8 puts an authored DELTA on top of the generator's answer, which is what ruling S5-3 says, and puts an authored LANDFORM in the generator as seed shape. Revision 1 put live state inside the recipe |
| **The collider on the same shape** | **PASSED, with one claim withdrawn** | The collider reads the extractor's triangles, and the extractor reads the same cells at every rung. §4.6 withdraws "a river that COLLIDES": water cells emit no triangle and no buoyancy exists. The BED collides, which is what a valley needs |
| **SL8 — seamless** | **PASSED for the octaves, with two terms fading** | §5.2 gives a per-column bound in cells: 0.33 to 1.96 cells at the rung change, which is 0.2 to 2.0 pixels depending on slice 8's residency rule. The terrace and the undercut each fade over one rung rather than switching (§4.5, §6.2) |
| **HR5 — 100 % coverage** | **PLANNED, not claimed** | Every new branch sits in `vd-terrain`, a Tier-A crate, in the `carve_at` shape the caves already use. The hard part is finding a live chunk for every arm on a whole planet, and **M4-12** is that plan |
| **V4 — vegetation are art assets** | **PASSED, with a gap named** | Nothing here draws a tree. `biome_at` is what the server's scatter skeleton reads. But the four-member biome set cannot carry the picture's forest and fields (§7), and talus and boulders are NOT covered by V4 (§13 item 8) |
| **SL6 — ask before new data crosses** | **NOT PASSED. It rides two open asks** | Revision 1 said "no new crossing" while its own recommendation needed gravity on the wire and its neighbour ships a 9.83 MB artifact. The truth: this domain needs DOMAIN 02's `BodyFacts` ask (56 bytes, a new tag) and DOMAIN 03's artifact ask (§12 D3). **One ask should carry both.** The wind stays a law and the biome stays a function, so this domain adds no THIRD crossing |

---

## 12. What you decide

| # | The decision | The recommendation | The cost of deciding late |
|---|---|---|---|
| **D4-1** | Does the height field become a FOLD over the macro stack and `Z`, replacing the single global octave sum? | **Yes.** It is the only change that makes a plain flat and a ridge rough from one law. | After the first saved world it is a world epoch. |
| **D4-3** | The octave table runs from `C · cell_m` at the rung below the macro node down to `C · cell_m(0)`, anchored to the LADDER and to no neighbour's lattice, deleting `LONG_WAVE_CAP_M` and `SHORT_WAVE_M`. | **Yes**, with `C = 8`, decided together with DOMAIN 02's `SHORT_WAVE_M`. M4-3 is the gate. | A world epoch. |
| **D4-4** | The octave amplitude is the body's spectrum times ONE per-column factor `r(dir)`; the per-octave derivative damping is withdrawn. | **Yes.** It obeys DOMAIN 02's binding multiplier rule, it keeps the coarse rung's bound exact, and it costs one divide per column instead of one per octave. | A world epoch. |
| **D4-5 ★ CHANGED** | A RIDGED, FLOW-ALIGNED octave band between the hillslope length and the macro node. | **Yes.** It is the answer to "the surface is not interesting enough" in the band the eye actually reads — 100 m to 3 km — and the direction is already a byte in DOMAIN 03's artifact. | A world epoch. |
| **D4-6 ★ CHANGED** | The terrace reads a RADIAL HARDNESS function on a structural datum; the falloff is measured against HALF the hard-band spacing; the terrace fades over one rung; the SUBSTANCE stays inside today's `max_depth_m()` veneer. | **Yes.** The periodic stack of revision 1 is withdrawn: it unbounds `max_depth_m()`, deletes the below-skip's 463 chunks per column and moves the ladder band. | A world epoch for the shape; the veneer rule keeps every address still. |
| **D4-7** | A 3-D REMOVAL term in a thin band at the surface, for arches and undercuts, with a rung range and a fade. | **Yes to removals**, and only after M4-11 prices it. DOMAIN 02 says by name that no layer of its own can make an arch. | A world epoch. |
| **D4-7b** | A 3-D ADDITION term, for a caprock cap wider than its shaft. | **No, not in the first slice.** The cave machinery can only remove (`chunk.rs:646`). | Cheap later; it is additive. |
| **D4-8** | Does the density byte become the SLOPE-CORRECTED distance instead of the radial gap? | **MEASURE FIRST (M4-9), then decide.** It reverses APPROVED ruling row V6 B-3, and because the byte is in the digest (`digest.rs:38-42`) it opens a world epoch on the day it lands. | Format B freezes with the first saved world. |
| **D4-9** | Does the mesh carry a stratum id per vertex, and does the mesh digest cover it? | **Yes** to both. +3.4 % on the mesh bytes, and D4-6's macro-slope soil stripping is what makes it truthful at every rung. | Cheap later; it is additive. |
| **D4-10 ★ CHANGED** | An authored DELTA rides on top of the fold (live state, ruling S5-3); an authored LANDFORM lives in the generator as seed shape. | **Yes, both, kept apart.** Revision 1 put live state inside the recipe, which breaks SL10 clause 1 and the golden self-check. | Cheap for kind A; kind B is a world epoch. |
| **D4-11** | Which surface forms are built first? | **Dunes and rills.** Both are in the picture and both read their direction from a law. | Cheap later; each is one gated term. |
| **D4-12** | Does the seasonal snow line stay OUT of the generated cells? | **Yes.** Time may not enter the static shape (SL10 clause 1). A permanent ice cap from the climate mean is seed shape; a season's snow is the client's style or a live diff. | Cheap now, a law breach later. |
| **D4-13 ★ CHANGED** | The relief ceiling and the size term. | **Take DOMAIN 02's version, not this domain's.** A bare ceiling refuses every body under about 313 km of radius (COMPUTED, §4.3). Gravity reaches the generator as a quantised integer in `BodyFacts`. **This is a vote for DOMAIN 02's SL6 ask, not a second version of it.** | A world epoch, and it is DOMAIN 02's to open. |
| **D4-14 ★ CHANGED** | Does the ladder BAND move? | **No, and this domain now guarantees it does not.** §4.3 leaves the relief to DOMAIN 02's size-term law and §4.5 keeps the substance inside `max_depth_m()`. Revision 1 said "accept the move"; DOMAIN 03 had already deleted the same proposal. | Nothing moves, so nothing is owed. |
| **D4-15** | `octaves_at`'s "never fewer than one octave" clamp (`body.rs:255-256`) is REMOVED, and its test (`body.rs:368-379`) changes with it. | **Yes.** At the top rung the right count is ZERO, because `Z` carries the shape there. Keeping one 8 192 m octave against a 2 048 m cell aliases. | A world epoch. |
| **D4-16** | Where does the 1 m to 8 m relief come from? | **The fine floor plus PLACED ROCK OBJECTS with their own colliders.** Not more octaves. Talus, boulders and outcrops are landform, ruling V4's "trees, grass and decoration" does not cover them, and they need a collider. **It needs its own slice.** | Cheap later; but the owner should know NOW that the height field alone does not answer his sentence. |
| **D4-17** | Does DOMAIN 02's flow proxy (its layer L4) stop where DOMAIN 03's channel carve starts? | **Yes, they must.** If both run, a valley is cut twice and the budget pays 1.05 ms twice. | Cheap now, a double-cut valley and 1.05 ms later. |
| **D4-18 ★ NEW** | **The merge in §3.4: who owns which part of the one octave table in `body.rs`.** | **Take the merge:** DOMAIN 03 owns the amplitudes and everything above the macro node; DOMAIN 04 owns the table's ends, count and the modulation's rung rule; DOMAIN 02 owns `r`'s layers, the ceiling and the size term. | **Nothing else in this document is safe until this is settled**, because `body.rs` cannot hold two tables and three documents currently propose one each. |

---

## 13. Open questions, and what is UNMEASURED

1. **The macro stack's determinism and cost (DOMAIN 02).** Closed-form is the right shape, but the float
   fence row is not written and 416 ns is a per-operation model, not a measurement. UNMEASURED.
   **M4-7, M4-8.**
2. **The erosion solve's determinism (DOMAIN 03).** Its revision 2 removes the cross-host risk by
   shipping the artifact, which converts a determinism problem into an SL6 ask. Both must land.
3. **The client's boot with a 9.83 MB artifact per body.** DOMAIN 03 ships `Z`, the graph and the lake
   table, with a 19 KB pyramid top for the approach. Nothing prices the arrival, the residency of two
   bodies at once, or what the client draws while it waits. **OPEN, and it is slice 8's with DOMAIN 03.**
4. **The slope budget across DOMAIN 03's spectrum and this domain's `r`.** COMPUTED in §4.4: three of
   DOMAIN 03's published amplitudes alone give 38.9° of added RMS slope, above the angle of repose, before
   `r` multiplies anything. Either the peak is high or the two documents mean different things by
   amplitude. **M4-13.**
5. **The double-cut valley.** DOMAIN 02's L4 against DOMAIN 03's carve, and DOMAIN 03's detail term
   against §4.4's ridged band. **OPEN, cross-domain. D4-17, D4-18.**
6. **The vista's BYTES.** 400 MB to 810 MB for a 50 km high-vantage vista at today's vertex counts, times
   an ESTIMATED 1.4 to 2.9. This domain hands slice 8 the chunk counts, M4-2's multiplier and the lever
   table (405 KB → 254 KB → 185 KB).
7. **The authored delta's arrival seam.** A delta moves a hill by hundreds of metres. A client that draws
   the seed's hill and then receives the delta sees the hill move, and SL8 calls a seam a defect. The
   likely answer is that the delta arrives with the realm's surface statement, before the first chunk is
   drawn. That is a question about WHEN, not about new data.
8. **What the reference picture shows and this document does not build.**
   - **Talus and boulders.** Every cliff in the picture stands on a scree cone. D4-16 names them and puts
     them with the block machinery. V4 does not cover them.
   - **Alluvial fans, floodplains and river terraces.** DOMAIN 03's, and it now proposes strath terraces
     from the channel's own depth rather than from the strata.
   - **A ROAD.** A built feature, not a landform; it belongs with the block store.
   - **A GLACIER.** A moving surface with its own erosion. It is not a static shape.
9. **Floating.** Making the river cells is the HR3 answer for the SHAPE, and it gives nothing else today:
   the extractor emits no triangle for a water cell and no buoyancy exists in the workspace. A pilot who
   ditches lands on the bed. **Owed to a later slice, and named here so nobody reads §4.6 as a promise.**
10. **The cave machinery's and the undercut's real cost.** Both struck. UNMEASURED. **M4-11.**
11. **The biome set and signature.** Four members cannot carry a forest mass and cultivated fields, and
    the signature has no room for obliquity, insolation, an upwind ridge or a coast. **DOMAIN 05's, and
    §7 states what this domain needs from it.**
12. **The home planet's third literal.** `home.rs` holds two literals and `crates/bins/tests/home_body_pin.rs`
    proves the forest draws them. A shipped `BodyFacts` adds more inputs and needs its own pin and its own
    thousand-ulp gate.
13. **The owner's five words.** He asked for biomes that follow *position, spin, trajectory, size and
    gravity*. Today `biome_at` reads latitude about a fixed `+Z` axis, height and two noises
    (`height.rs:29-35,39-66`). None of the five reaches it. This domain took only gravity, and used it for
    the relief ceiling, which DOMAIN 02 now owns. **The other four are owed by DOMAIN 02's `BodyFacts` and
    DOMAIN 05's climate, and neither has landed.**

---

## 14. Refutation answers

Round 2 refuted revision 1. Between them the two refuters raised **45 items**. **Thirty-one are FIXED**,
**four are KEPT with the reason stated**, and **ten are OWED to a named neighbour or a measurement**.
Six proposals of revision 1 are withdrawn outright: the erosion-spacing anchor, the gravity relief law,
the per-octave derivative damping, the periodic radial stratigraphy, the authored delta inside the fold,
and the claim that a river collides.

### Round 2, refutation A — the laws and the code

| # | Finding | Severity | Answer |
|---|---|---|---|
| A-B1 | The octave table is anchored to a lattice DOMAIN 03 has withdrawn; `S_e = 8 224 m` makes `n` non-integer and `t` non-dyadic; §3's contract row is not met and §13 does not record it | BLOCKER | **FIXED.** Verified in `03_erosion_rivers.md:52-63,345-376`: `n_macro = 640`, 8 224 m, not a rung, never in an address. §4.3 re-anchors the table to the LADDER (`λ = C · cell_m(rung)`), which needs nothing from a neighbour; §3.2 row 1 is rewritten; §4.2 replaces the dyadic argument with the identical-rounding one and says the old sentence is withdrawn |
| A-B2 | The relief law is deleted and only a ceiling replaces it; on a small body the ceiling is unbounded and a landed test panics; DOMAIN 02 fixes `σ_y = 243 MPa` and calls it a scale | BLOCKER | **FIXED — the law is WITHDRAWN and CEDED.** §4.3 prints the ceiling against radius (313 km break-even, COMPUTED) and the `ladder.rs:89` refusal, and hands the relief, the ceiling and the size term to DOMAIN 02. D4-13 becomes a vote for DOMAIN 02's version |
| A-B3 | SL6 is marked PASSED for a datum that must cross; the snap analogy fails; `powf` is outside the fence; DOMAIN 02 is already asking | BLOCKER | **FIXED.** §11's SL6 row now reads **NOT PASSED**. §4.3 states that gravity reaches the generator only as a quantised integer in DOMAIN 02's `BodyFacts`, and §10 adds the thousand-ulp gate row. §1 cites `taxonomy.rs:616`'s `powf` and `body.rs:86`'s written rule |
| A-B4 | §5.2's comparison uses `k = 0.5`, not the drawn 0.468; today's bound is 3.05 m at rung 3, so the proposal is LOOSER from rung 3 up and §0 tells the owner the opposite | BLOCKER | **FIXED.** Re-derived independently: 0.40 / 1.24 / 3.05 / 32.76 / 321.97 / 1 469.16 m. §0 and §5.2 now say the bound **tightens on smooth ground and loosens on steep ground, and that is the price of the picture** |
| A-D1 | The bound a coarse rung can STATE is a geometric envelope, not the exact dropped sum, because the damping reads octaves it never evaluated | DEFECT | **FIXED at the source.** §4.4 withdraws the per-octave damping. With `a_i = A_i · r(dir)` the dropped sum is a per-body constant times one number the coarse rung already holds, so the stated bound IS the exact bound |
| A-D2 | The budget uses DOMAIN 02's withdrawn 180–520 ns; its current table says 416 ns and 672 ns | DEFECT | **FIXED.** §3.1 and §9 use 416 ns and 672 ns over 4 096 columns: +1.70 ms, or +2.75 ms with rivers. The river-bearing case is what makes D4-17 a budget decision |
| A-D3 | §9.2 mixes rungs: three deltas are switched off at rung 2 by this document's own rules, so "FAILED" is as unproven as revision 0's "passes" | DEFECT | **FIXED.** §9.2 now prints TWO coherent lines — a rung-0 line at 6.62 ms and the worst measured chunk's own rung-2 line at 8.75 ms — and the verdict is **UNPROVEN**, with three measurements named |
| A-D4 | The forbidden subtraction is used anyway for the 61 ns per vertex | DEFECT | **FIXED by stating the rule precisely.** §9.1 distinguishes a subtraction that changes the WORK PER UNIT (forbidden: the cave delta) from one that changes only the COUNT of units when the unit is the regressor (sound: the extraction). The 61 ns is re-labelled **ESTIMATED** and M4-2 confirms it |
| A-D5 | The cell pass pays nothing for the periodic radial strata: 0.48–1.19 ms per chunk | DEFECT | **FIXED.** §4.5 replaces the periodic stack with a veneer rule, and §9.2 charges **+0.48 ms** for the band index inside the veneer on every surface chunk |
| A-D6 | Soil stripping on the fold's slope makes the SUBSTANCE rung-dependent, and D4-9 puts it on every vertex | DEFECT | **FIXED.** §4.5 item 5 strips on the MACRO gradient, which is rung-independent, so the substance is identical at every rung and no bound is owed. §5.1 lists the fold-gradient version as a way to break the prefix property |
| A-D7 | With `C = 16` the rung rules delete every headwater channel and the 8 m arch | DEFECT | **FIXED twice.** `C` drops to 8 (§4.3), and §4.6 property 4 rewrites the rule to read the VALLEY width, not the water's. The water's own trench is a rung-0 term with a bound of its own depth |
| A-D8 | "A river that COLLIDES" and "a ditched pilot floats" are not in the code | DEFECT | **FIXED — the claim is WITHDRAWN.** Verified at `extract.rs:70-74`: a water cell is on the air side and emits no triangle; nothing outside `vd-terrain` reads `Stratum::Water`; no buoyancy exists. §4.6 says the BED collides and §13 item 9 names floating as owed |
| A-D9 | D4-8's consequence list misses `seat_eighths`, `digest.rs:39` and the lateral-edge inconsistency | DEFECT | **FIXED.** §6.1 gains all three: the seat's two cells share a column so the factor divides out there too (the argument revision 1 never made), the byte is in the identity so D4-8 is a world epoch on the day it lands, and M4-9 must measure the lateral-edge cost as well as the gain |
| A-D10 | §8's authored delta is not ruling S5-3's entry and puts live state inside the seed shape | DEFECT | **FIXED.** §8 is rewritten into two kinds: a DELTA on top of the fold (live state, S5-3) and a LANDFORM inside the generator (seed shape, a world epoch). The rule that separates them is stated in one sentence |
| A-D11 | §4.8's damping is a statistical variance model, not the pointwise gradient §4.1 describes; the histogram will not reproduce the table | DEFECT | **FIXED.** The table is withdrawn. §4.8 now states the METHOD (per-octave RMS slope, quadrature sum) and a CONSTRAINT on whatever spectrum DOMAIN 03 lands, and M4-4 is the gate. No third table is printed |
| A-W1 | The pixel gate pairs the wrong bound with the wrong distance; the neighbour-rung swap gives 1.39 px | WEAKNESS | **FIXED.** §5.2 states the gate as **bound ÷ cell of the rung**, which is what a neighbour-rung swap actually compares, and gives the pixel figure under both readings of the code's drawable factor |
| A-W2 | The pixel gate's denominator is an assumed residency rule slice 8 has not made | WEAKNESS | **FIXED.** The gate is stated in cells; §5.2 says in writing that the residency rule is slice 8's and that it only scales the pixel column |
| A-W3 | The ×2 vertical stack is never derived, and the 185 KB lever needs a third change | WEAKNESS | **FIXED.** §5.4 derives the stack (a column crosses two chunks where the ground falls more than one chunk height over one chunk width) and gives the range 1 000–2 000 with M4-2 as the gate; the lever table prints 405 KB → 254 KB → 185 KB with the third change named |
| A-W4 | The column count is 3 844 in one place and 4 096 in another | WEAKNESS | **FIXED.** §1 states the unit as **10.5 ns per column-sample**, and §9 prices every per-column row over the box's **4 096** columns, once and for all |
| A-W5 | The 50 km vista is not visible from the ground on this planet | WEAKNESS | **FIXED.** §5.4 opens with the COMPUTED horizon: 3 473 m from an eye 1.8 m up, 27 % closer than Earth's 4 789 m, and states the vista as a HIGH-VANTAGE case |
| A-W6 | The corner cell width is right, but not by the working shown | WEAKNESS | **FIXED.** §4.3 shows the metric step: `1.5584 × 0.4714 / 0.7854 = 0.935` |
| A-W7 | The ownership split is not what DOMAIN 02 states: it designs L6, the biome, the ceiling and the water surface, and two documents give one crate two octave tables | WEAKNESS | **FIXED, and promoted.** New **§3.4** prints the three documents' tables side by side and recommends a merge; **D4-18** is the decision, and §12 says nothing else is safe until it is taken |
| A-N1 | Six citation slips against a claim that none remain | NOTE | **FIXED.** `body.rs:171` (frequency), `body.rs:177` (the halving), `body.rs:196` (topsoil), `chunk.rs:641` (the band gate), `chunk.rs:646` (`greater`), and every slice document now carries its `docs/investigation/2026-09-07/` path |
| A-N2 | "(1 984, 3 968] for EVERY body" is false at the small end | NOTE | **FIXED by deletion.** The paragraph argued against a proposal that is already withdrawn, and §3.1 no longer contains it |
| A-N3 | The finest-octave amplitude is quoted as a range where a measured value exists | NOTE | **FIXED.** §0 states the home planet's own **0.397 m** at 48.83 m |
| A-N4 | Two decisions left implicit: the third `home.rs` literal and its pin, and the coarse rungs' cost rise | NOTE | **FIXED.** §13 item 12 owes the pin; §5.3 prints the top rung at 1.8–2.9 ms against today's 266 µs and says M4-1 must print the curve |

### Round 2, refutation B — believability, cost and the owner

| # | Finding | Severity | Answer |
|---|---|---|---|
| B-F1 | The gravity ceiling refuses every body under about 313 km, and the home planet's band grows by half | BLOCKER | **FIXED — WITHDRAWN and CEDED**, as A-B2. §4.3 prints B's own table and the `ladder.rs:89` refusal, and keeps DOMAIN 02's size term so the band does not move at all |
| B-F2 | `S_e = 2^(rungs+1)` cannot tile a cube face and DOMAIN 03 says so by name | BLOCKER | **FIXED**, as A-B1. The anchor is the ladder; the wavelength SPLIT against the macro node is all that remains, and the 0.4 % offset between 8 192 m and 8 224 m is stated |
| B-F3 | The terrace is DISCONTINUOUS — `u` never passes 0.5, the band switch jumps 12 m, `Lip(T)` is unbounded — and the terrace has no rung rule | BLOCKER | **FIXED.** §4.5 measures `u` against HALF the hard-band spacing, so `f(1) = f'(1) = 0` and the pull fades to nothing from both sides; the derived `Lip(T) = 1.413` is then taken over a reachable range. The terrace also gains a rung rule and a one-rung fade, and §5.2 carries its term |
| B-F4 | DOMAIN 03's solve is computed on the SERVER and SHIPPED, so the SL6 row is false and the client's boot is unpriced | BLOCKER | **FIXED.** §3.2 row 6 quotes DOMAIN 03's three statements, §11's SL6 row reads NOT PASSED, and §5.4 and §13 item 3 price the 9.83 MB artifact and the 19 KB pyramid top the client must hold before its first chunk |
| B-F5 | Below the erosion grid there are still NO LINES, and that band is the whole vista | BLOCKER | **FIXED, and it is the largest new part of revision 2.** §4.4 rule 2 adds the RIDGED, FLOW-ALIGNED band between the hillslope length and the macro node, using the D8 direction already in DOMAIN 03's artifact. §4.4 also states the amplitude gap in DOMAIN 03's own numbers and the slope-budget question the two documents must settle (M4-13) |
| B-F6 | D4-13 needs a new datum and the local formulation is not looked for | DEFECT | **FIXED.** §4.3 routes gravity through DOMAIN 02's `BodyFacts` ask, and §11 says one ask should carry both domains. The local formulation B proposes — gravity from the look radius through the mass-radius law — is the fallback if the owner refuses the ask, and it needs a fenced integer form because `powf` is outside the fence |
| B-F7 | The periodic radial stack unbounds `max_depth_m()` and kills the below-skip; +270 ms per radial column | DEFECT | **FIXED — the periodic stack is WITHDRAWN.** §4.5 keeps the SHAPE radial and the SUBSTANCE inside today's veneer, so `max_depth_m()`, `crust_m`, the below-skip's 463 chunks and its landed test are all untouched. Verified against `strata.rs:165-169` and `chunk.rs:12-17` |
| B-F8 | D4-14 moves the ladder band; DOMAIN 03 has DELETED the same proposal | DEFECT | **FIXED.** D4-14 now reads **the band does not move**, and §6.3 states the guarantee with both reasons. The two documents now agree |
| B-F9 | The 44.3° figure does not reproduce (56.7°), and the RMS method is never stated | DEFECT | **FIXED.** Re-derived independently: 56.7° from the stated gains. §4.8 prints the METHOD in three lines and withdraws both tables rather than printing a third that a reader cannot check |
| B-F10 | The +0.60 ms undercut row uses the cost the document forbids itself to use | DEFECT | **FIXED.** §9.1 marks the undercut **UNKNOWN**, §9.2 carries it as an UNKNOWN row rather than a number, and M4-11 measures the cave and the undercut together |
| B-F11 | §9.2 counts the columns two ways | DEFECT | **FIXED**, as A-W4: everything over 4 096 |
| B-F12 | The pixel gate is really "the bound in cells" — good news that should be stated — and 870 m contradicts the code | DEFECT | **FIXED, and the good news is stated.** §5.2 states the gate in cells and says in writing that it is resolution-independent. §1 and §5.2 carry both readings of `geometry.rs:1156-1173`: 869 m for a half-metre extent, 1 505 m for a cube cell |
| B-F13 | The relief ceiling is a one-parameter fit presented as a validation | WEAKNESS | **FIXED.** §2 and §4.3 call it an order-of-magnitude ceiling read as a SCALE, quote B's geological objections to Everest and Olympus Mons as evidence, and say only the RATIO tests anything. The law itself is DOMAIN 02's now |
| B-F14 | "At most nine channel segments" holds only while a chunk is small against a macro cell | WEAKNESS | **FIXED.** §4.6 property 1 states the window as a function of the rung (a chunk passes the macro node at rung 7 and spans 15.5 cells at rung 11) and makes the gather **per REGION**, which is how the tube carver earns agreement across a chunk edge |
| B-F15 | The owner's biome sentence is answered for gravity only, and `biome_at` is frozen while its inputs are handed away | WEAKNESS | **FIXED.** §3.1 requirement 4 says the freeze is on the NUMBER of sources, not the signature; §7 prints the signature the function must grow to; §13 item 13 records that four of the owner's five words reach nothing today |
| B-F16 | Weather is named, listed and not simulated | WEAKNESS | **FIXED.** §0 says in one line that this domain gives the weather its inputs and builds none of it; §3.3 names DOMAIN 05 as the owner and DOMAIN 02's §5.5 as the ask that is already open |
| B-F17 | The budget is 8 ms per CHUNK, not per rung-0 chunk | NOTE | **FIXED.** §0, §9 and §11 all say per chunk at any rung, and cite `owner_decisions_2026-09-07_voxels.md:441-443` |
| B-F18 | The 0.935 m corner cell is right; the shown working does not produce it | NOTE | **FIXED**, as A-W6 |
| B-F19 | The reference picture's forest and fields have no biome to hang on | NOTE | **FIXED.** §7 states that the four-member set cannot carry the picture and §13 item 11 hands the set to DOMAIN 05 |

### The four findings KEPT, with the reason

| # | The refuter's point | Why it is kept |
|---|---|---|
| A-D4, the rule itself | *"Either the rule is wrong, or the slope is; the document must say which."* | **KEPT as answered, not as refuted.** The rule was too broad, not wrong. §9.1 now states it precisely: a cross-rung subtraction is forbidden when the rung changes the work per unit, and sound when the rung changes only the count of units and the unit is the regressor. The 61 ns row survives as ESTIMATED |
| A-B1, item 2 | *"the bicubic is still deterministic … so the conclusion survives; the stated argument does not"* | **KEPT exactly as the refuter states it.** §4.2 keeps the conclusion, deletes the dyadic argument by name, and replaces it with identical rounding in a fixed order |
| B-F12, the 870 m | *"using the person's number here is conservative by 1.7×, so no conclusion moves"* | **KEPT as both.** The gate is stated in cells, so neither number is load-bearing; §5.2 prints both columns and says slice 8 picks |
| A-N2 | *"(1 984, 3 968] for EVERY body is false at the small end … the paragraph argues against a proposal that is already withdrawn"* | **KEPT as deleted.** The refuter says nothing turns on it; the paragraph is removed rather than corrected, so no reader can build on it |

### The ten items OWED to a neighbour or a measurement

| # | The item | Owed to | Where |
|---|---|---|---|
| 1 | The relief law, the ceiling and the size term | **DOMAIN 02** | §4.3, D4-13 |
| 2 | The macro stack's float fence and its measured cost | **DOMAIN 02** | §10, §13 item 1, M4-7 |
| 3 | The `BodyFacts` SL6 ask, and its thousand-ulp gate | **DOMAIN 02** | §4.3, §10, §11 |
| 4 | The per-octave amplitudes (the slope spectrum) | **DOMAIN 03** | §3.4, §4.4 |
| 5 | The artifact's SL6 ask, and the client's boot with 9.83 MB | **DOMAIN 03 and SLICE 8** | §3.2, §5.4, §13 item 3 |
| 6 | The slope budget across the two documents' tables | **DOMAIN 03 and DOMAIN 04 together** | §4.4, M4-13 |
| 7 | The double-cut valley, and the detail-term overlap | **DOMAIN 02 and DOMAIN 03** | §3.2, D4-17, D4-18 |
| 8 | The biome set and the `biome_at` signature | **DOMAIN 05** | §7, §13 items 11 and 13 |
| 9 | The vista's residency in BYTES, and the client's own budget | **SLICE 8** | §5.4, §9.4 |
| 10 | The cave and undercut costs, and the vertex count on rough ground | **M4-11 and M4-2** | §9.1, §9.2 |

### Round 1 (against revision 0), the retained record

Round 1 refuted revision 0 and revision 1 answered its 46 items. The refuters state that those findings
stay on the record here, so the two tables are kept below unchanged. **Six of those answers are
superseded by round 2**, and the list is stated first so no reader builds on a superseded answer.

| Round 1 answer | What round 2 changed |
|---|---|
| A-B1 / B-D2, the octave anchor | The anchor moved again, from DOMAIN 03's erosion spacing to the LADDER (§4.3) |
| A-D3 / A-D8 / B-D10, the terrace and the strata | The periodic stack is withdrawn; the veneer rule and the half-spacing falloff replace it (§4.5) |
| A-D10, gravity | The relief law is ceded to DOMAIN 02; this domain proposes no ceiling of its own (§4.3) |
| A-W1, the SL8 arithmetic | The comparison used `k = 0.5`; against the drawn roughness the bound loosens on steep ground (§5.2) |
| B-D5, the damping | The per-octave damping is withdrawn for a per-column factor, and the 44.3° figure did not reproduce (§4.4, §4.8) |
| A-D4 / B-D4, the river | "A river that COLLIDES" is withdrawn: a water cell emits no triangle (§4.6) |

#### Round 1, refutation A — the laws and the code

| # | Finding | Severity | Answer |
|---|---|---|---|
| A-B1 | §11 claims a landed test still holds; `n = 9` on a 12-rung body makes `every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it` go red at rung 9, and the cost is the SAME at rungs 9, 10, 11 | BLOCKER | **FIXED.** Verified at `body.rs:355-365` and the clamp at `body.rs:255-256`; A's rung 9 is right and B's rung 10 is off by one. §4.3 re-anchors the octave table to DOMAIN 03's erosion rung, giving `n = rungs − 1 = 11`, so the test passes as written once the clamp is removed (D4-15). §5.3 now states honestly that the coarse rungs get MORE expensive than today, because the macro stack is not free |
| A-B2 | §7 row 2 writes a biome into a record that generated terrain does not have; the object byte is already the tree's; the cost is not zero; and "the biome is thrown away" is false | BLOCKER | **FIXED — the row is WITHDRAWN.** Verified against `topic_00_format_sitting.md:34-36,55` and ruling V6 B-5, and against `chunk.rs:183,637` (the biome decides every cell's substance). §7 states the local formulation instead: `biome_at` is a pure function both hosts compute |
| A-B3 | The SL10 gate is asserted for a map §13 calls UNMEASURED; two code paths would be a port; a river network does not refine per tile; an iterative solve is where no-drift dies | BLOCKER | **FIXED.** §11's SL10 row now reads **OPEN, held by DOMAIN 02 and DOMAIN 03**, and §10 carries two OPEN rows for their arithmetic. §3 is rewritten against DOMAIN 02's closed form and DOMAIN 03's single coarse solve, so no tile refinement exists and the server takes the same path as the client. §13 items 1 and 2 |
| A-B4 | The ladder band derives from a GLOBAL property the client cannot hold; the citation is wrong; §12 has no row | BLOCKER | **FIXED.** Citation corrected to `body.rs:233-236` throughout. §4.3's gravity law makes the ceiling a per-body number that needs no maximum over any map, which removes the two-address failure entirely. **D4-14** is the new decision row |
| A-D1 | The 0.85 ms cave delta subtracts a rung-2 chunk from a rung-0 chunk | DEFECT | **FIXED.** The unit row is struck from §9.1 and marked NOT AVAILABLE. §9.2 is rebuilt on the worst MEASURED chunk. **M4-11** is owed |
| A-D2 | `greater` can only open air, so it cannot make a hoodoo cap | DEFECT | **FIXED — the cap is WITHDRAWN** (§6.2, D4-7b). Verified at `chunk.rs:641-649`. *One step of A's reasoning is refuted:* the one-cell gap clamp and `cavern_scale_m = 20` do NOT bound a feature to 20 m — the clamp is per cell and a 200 m cave exists today under it. A's conclusion is right; that step is not, and §6.2 says so |
| A-D3 | The terrace runs after the carve and re-benches the bed; and the octave damping cannot know the channel distance | DEFECT | **FIXED.** §4.1 reorders: the channel distance is found BEFORE the loop, the terrace runs before the carve, and the carve is last. §4.6 property 1 carries the new cost model, and §9.2's lever prices the prune |
| A-D4 | §4.5 says the sea is not a cell today; it is | DEFECT | **FIXED.** Verified at `chunk.rs:207-213,633-635` and `strata.rs:18`. §4.6 makes the river CELLS too, through `fluid_at(body, r, water_m)`, which is HR3 and which gives the river a collider |
| A-D5 | §4.2 leans on ruling A5 for something A5 does not say | DEFECT | **FIXED.** Verified at `topic_00_format_sitting.md:23`. The argument is withdrawn in §4.2 and replaced by the prism mechanism ruling V10 actually landed, plus the stated corner rule and its test |
| A-D6 | §6.1 ignores the clamp on the line it cites; the stated reason is wrong; two consequences unaddressed | DEFECT | **FIXED.** Verified at `extract.rs:182-187`. §6.1 now makes A's argument (the clamp saturates the lateral neighbour), carries both consequences (the cave hollow's units, the bound stated for the wrong quantity), and **D4-8 changes from "Yes" to "MEASURE FIRST"** with M4-9 |
| A-D7 | §4.8's table cannot be reproduced: three columns use three different gain pairs | DEFECT | **FIXED.** §4.8 is recomputed with ONE gain pair, ONE damping constant and three macro steps. Every row now follows from the stated inputs alone |
| A-D8 | The terraces cannot stand on the strata table; the cliff band reads as soil | DEFECT | **FIXED.** Verified at `strata.rs:166-178,188-209`. §4.5 rebuilds it as a PERIODIC radial stack with a band count, a thickness law and a hardness draw, plus the soil-stripping rule that makes a cliff read as rock. D4-6 restated |
| A-D9 | The §9.2 budget omits four of the document's own proposals, and two rows are internally wrong | DEFECT | **FIXED.** §9.2 is rebuilt from the worst MEASURED chunk, lists the three costs still outside it, prices the two levers, and states the verdict as **FAILED** |
| A-D10 | Gravity, which the owner named, appears nowhere | DEFECT | **FIXED.** §4.3 makes the relief ceiling `strength ÷ (density × gravity)`, checked against Everest and Olympus Mons; §2 explains the term; §4.3 also states what gravity does NOT change (the angle of repose). **D4-13** is the new decision row |
| A-W1 | The SL8 arithmetic goes the wrong way and §0 calls it a strengthening | WEAKNESS | **FIXED.** Two errors combined in the 23 m figure: the damping was off, and `Lip(T)` was guessed. §4.5 DERIVES `Lip(T) = 1.413` at `q = 0.6`; §5.2's alpine rung-3 bound is 3.58 m against today's 6.11 m, and the whole ladder is gated in PIXELS (worst 1.26 px) |
| A-W2 | The client's boot cost for the 50 km vista is never computed | WEAKNESS | **FIXED.** §5.4 computes the chunk counts per rung shell, the time (1.0 s on 14 threads, MEASURED unit) and the bytes (about 810 MB, which is the real problem), and hands the bytes to slice 8 |
| A-W3 | The fine floor is the whole answer on a plain and has no number | WEAKNESS | **FIXED.** §4.7 gives it a law (a fraction of the drawn topsoil depth), a per-biome amplitude table, a wavelength, a cost (40.4 µs) and a pixel argument for why it needs no crossfade |
| A-W4 | The design leaves two biome sources standing | WEAKNESS | **FIXED.** §3.1 requirement 4 deletes the biome channel. `biome_at` is THE biome; DOMAIN 02 and DOMAIN 05 own what it READS |
| A-W5 | §4.7 needs two channels §3.1's contract does not carry | WEAKNESS | **FIXED, by deleting the need.** §3.3 makes the wind direction a LAW of latitude and spin (Hadley cells and Coriolis), so no channel is needed. The ice flow is DOMAIN 03's and the glacial form waits for it |
| A-W6 | M4-6's seam gate is ill-posed and its threshold is finer than the map's quantum; the citation is wrong | WEAKNESS | **FIXED.** A is right that a halo column is bit-identical to the partner's by construction (`chunk.rs:139-142`). §4.2 and M4-6 now measure the SLOPE KINK between adjacent columns across the edge. Citation corrected to `extract.rs:37` |
| A-N1 | D4-8 reopens an APPROVED ruling without saying so | NOTE | **FIXED.** D4-8 now names ruling V6 row B-3 and says plainly that the owner is being asked to reverse it |
| A-N2 | §7 addition 3 fixes a problem `seat.rs` does not have | NOTE | **FIXED — the row is WITHDRAWN.** Verified at `seat.rs:32-62`: it reads the cell and exactly one neighbour, so an arch already gets a correct local answer |
| A-N3 | Eight citation slips | NOTE | **FIXED.** All eight corrected and re-verified in the source: `body.rs:233-236`, `body.rs:355-365`, `lattice.rs:11-12`, `chunk.rs:183`, `extract.rs:37`, `chunk.rs:642`, `extract.rs:62-69`, `body.rs:252-258` |
| A-N4 | Jargon left unexplained; five §2 entries with no example | NOTE | **FIXED.** §2 adds dyadic rational, D8 and D-infinity, hogback, anisotropic, uber noise, smoothstep, caprock, talus, angle of repose, knickpoint, crustal strength, Hadley cell and Coriolis, orographic lift; and lacunarity, gain, stream power, isostasy and tectonic uplift now each carry an example in the game's words |
| A-N5 | "Nothing in the recipe is smaller than 48.8 m" overstates | NOTE | **FIXED.** §0 says "the height field", and names the cave numbers (`body.rs:221,228`) in a parenthesis |
| A-N6 | §4.3's "every rung" claim contradicts §5.3 | NOTE | **FIXED.** With `n = rungs − 1` the claim now holds for rungs 0 to 10, and §5.3's table says so. §4.3 also says "about `C`", not "exactly" (see B-N1) |
| A-N7 | Weather, which the owner named, is not addressed | NOTE | **FIXED.** §3.3 is new: what the shape OWES a weather model (elevation, slope, aspect, shelter, distance to water, upwind ridge height), what it READS back (the wind as a law), and the SL6 reading (static climate is shape; a storm is live state and would be a new lane) |
| A-N8 | A label contradiction: an operation count is ESTIMATED, not COMPUTED | NOTE | **FIXED.** §4.2's 15 ns and 60 ns are both marked ESTIMATED |

#### Round 1, refutation B — believability, cost and the owner

| # | Finding | Severity | Answer |
|---|---|---|---|
| B-B1 | The biome byte does not exist and its cost is not zero | BLOCKER | **FIXED — WITHDRAWN.** Same as A-B2. B's "46 to 92 KB per chunk on untouched ground" is the right way to state the cost, and §7 uses it |
| B-B2 | The budget gate fails on the worst MEASURED chunk | BLOCKER | **FIXED.** §9.2 is rebuilt on exactly that chunk and the verdict is stated as FAILED. §9.4 states the client's half separately, as B asked |
| B-B3 | The macro map is the hard part; §3.1 delegates it with a tile-edge crease, a size understated by 2.4 times, and an eroded map that is not local | BLOCKER | **FIXED.** §3 is rewritten. The tile-address rule is deleted and forbidden by name (§3.1 requirement 2). B's size range `(1 984, 3 968]` cells per top-rung face edge, giving 284 MB to 1 134 MB per body, is COMPUTED and recorded in §3.1 as what the closed form avoids. §13 items 1–3 hold what remains OPEN |
| B-D1 | "The orienters are made of octaves the coarse rungs drop" is false | DEFECT | **FIXED.** Verified at `body.rs:253-257` and `height.rs:1-8`. §0 states the real cause: an isotropic sum of band-limited noise has no LINES — no ridge crest, no coast, no scarp, no basin rim, at any scale |
| B-D2 | The proposal breaks a landed test; and the "at least one octave" clamp is not decided | DEFECT | **FIXED.** Same as A-B1, plus **D4-15** decides the clamp. *One correction to B:* the test goes red at rung 9, not rung 10, and the surviving octave at rung 11 would be the COARSEST (16 384 m) and not one "shorter than a cell" — it aliases at 2 samples per wavelength at the rung above, which is B's conclusion by a different route |
| B-D3 | The fold's gradient is dimensionally wrong, and two load-bearing terms read it | DEFECT | **FIXED.** Verified at `body.rs:168` and `noise.rs:85-121`. §4.1 multiplies by `1 ÷ wave_m` and projects onto the tangent plane, and states the factor of 1 024 that was missing |
| B-D4 | The river and the undercut sit outside the coarse-answer bound, and one of them pops | DEFECT | **FIXED.** §5.2's bound gains the carve term `|R_0 − R_L|`. §4.6 property 3 gives the carve a rung rule. §6.2 gives the undercut a rung range and states that it is the ONE term needing a crossfade band (2.1 px), which also keeps the collider and the picture on the same shape |
| B-D5 | The headline is computed WITHOUT the term the document calls load-bearing, and 44° RMS is above the angle of repose | DEFECT | **FIXED.** §4.8 is recomputed WITH the damping at `c = 1`: alpine 33.8° RMS, median near 29°. §4.8 quotes B's geologist reading and states the 44° figure as what happens without the damping. `c` gets a range and M4-4 |
| B-D6 | Eight numbers are typed in, and §11 claims they are not | DEFECT | **FIXED.** §4.4 has a new table naming every constant, its range, its source and its status; three are marked ESTIMATED and one UNMEASURED. §13 item 12 keeps the gain pair open |
| B-D7 | `step_cm` saturates at 17.7° on exactly the terrain the owner wants | DEFECT | **FIXED, by deleting the channel.** §3.1 requirement 1: the step is `|macro gradient| × S_e`, derived from the gradient the same closed form already returns. No ceiling, no quantisation, no second source |
| B-D8 | Nothing makes height detail between 1 m and 16 m; talus is missing | DEFECT | **FIXED, and B is right that this is the owner's real sentence.** §4.7 states the limit honestly (about 8 m on 1 m cells, a property of the cell) and names the three places the metre scale comes from. **D4-16** is a new decision, and talus, boulders and outcrops are in §13 item 8 with the note that ruling V4 does not cover them |
| B-D9 | The client's bytes are never priced, and the proposal multiplies them | DEFECT | **FIXED.** §5.4 prices the vista at about 810 MB and states the 1.4× to 2.9× multiplier; §9.4 states the client budget as owed; §13 item 5 hands it to slice 8 with a free lever |
| B-D10 | The terrace needs a stratigraphy the code does not have, and the band change is not stated | DEFECT | **FIXED.** §4.5 gives the stack a count, a thickness law and a hardness draw, and makes it PERIODIC so it spans any relief with a bounded table. §6.3 and **D4-14** now state the ladder-band consequence that revision 0's "what does not change" list omitted |
| B-D11 | The octave-count law breaks on small bodies | DEFECT | **FIXED.** §4.3's law is `max(0, …)` and anchors to the erosion rung. Worked on a 10 km asteroid: `rungs = 3`, `S_e = 16 m`, `n = 2` octaves, and the macro stack carries the rest. Total, believable, one world |
| B-D12 | The slope-corrected density byte is a tolerance-zero format change with no measurement owed | DEFECT | **FIXED.** **M4-9** is that measurement, and D4-8 now reads MEASURE FIRST |
| B-D13 | The weather the owner asked for is absent, and §4.7 reads a channel that does not exist | DEFECT | **FIXED.** §3.3 is new, as in A-N7, and it also states the gravity dependence B asked for (§4.3) |
| B-W1 | `Lip(T) ≈ 1/(1 − q)` is asserted, and SL8 rests on it | WEAKNESS | **FIXED.** §4.5 derives it from the smoothstep falloff and tabulates it at three values of `q`: 1.275, **1.413**, 1.550 — not 1.67, 2.50, 5.00 |
| B-W2 | The seat rule claim is false and the proposed fix would make it worse | WEAKNESS | **FIXED — WITHDRAWN.** Same as A-N2 |
| B-W3 | Two citation slips | WEAKNESS | **FIXED.** `chunk.rs:183` and `strata.rs:13,166-178` |
| B-W4 | HR5 is claimed, not planned | WEAKNESS | **FIXED.** §11's HR5 row reads PLANNED, and **M4-12** names the fixture plan for every new arm |
| B-W5 | The cavern delta is a subtraction across two rungs | WEAKNESS | **FIXED.** Same as A-D1. B's note that the 61 ns per vertex IS defensible across the same two rows is accepted, and §9.1 keeps it |
| B-N1 | "Exactly C cells per wavelength" is exact only at a face centre | NOTE | **FIXED.** §4.3 says "about", and quotes B's number: the bend gives a 0.935 m cell at a cube corner (COMPUTED from `bend.rs:23-30`), so `C = 16` reads 17.1 there — a 6.5 % effect |
| B-N2 | The Hurst exponent is used outside its range | NOTE | **FIXED.** §2 explains why a gain above `H = 1` is not a fractal surface and states that this document quotes the GAIN throughout |

#### Round 1: the three findings KEPT, with the reason

| # | The refuter's point | Why it is kept |
|---|---|---|
| A-D2, second half | *"a 30 m hoodoo cannot be cut out of a plateau by this machinery"*, argued from `cavern_scale_m = 20` and the one-cell gap clamp | **KEPT as refuted.** The clamp is PER CELL (`chunk.rs:630,647`), and a chain of fully-hollow cells makes a cavity of any size — a 200 m cave exists today under the same clamp. The 20 m is the scale of the hollow FIELD. A's CONCLUSION (that `greater` cannot add rock) is right and the cap is withdrawn; that step of the reasoning is not, and §6.2 says so in the source's own words |
| B-D2, second half | *"three rungs hold one octave of a wavelength shorter than a cell, which aliases"* | **KEPT as corrected.** `octaves_at` keeps the COARSEST octaves (`body.rs:253-257`), so the surviving octave is 16 384 m, not one shorter than a cell. It still aliases, at 2 samples per wavelength at the rung above the top. The conclusion stands; the mechanism is different, and §4.3 states the right one |
| B-D2 / A-B1, the rung | B says the test goes red at rung 10; A says rung 9 | **A is right, and §14 says so.** With `n = 9` and the clamp, `octaves_at(8).len() = 1` and `octaves_at(9).len() = 1`, so the assertion `here < below` reads `1 < 1` at rung 9 |

#### Round 1: the four findings OWED to a neighbour

| # | The item | Owed to | Where |
|---|---|---|---|
| 1 | The macro layer stack's float fence and its measured cost | **DOMAIN 02** | §10, §13 items 1 and 3, M4-7 |
| 2 | The erosion solve's fence, its convergence test and its no-drift legs | **DOMAIN 03** | §10, §13 item 2, M4-8 |
| 3 | The double-cut valley: DOMAIN 02's flow proxy against DOMAIN 03's carve | **DOMAIN 02 and DOMAIN 03 together** | §3.2 requirement 5, §13 item 4, D4-17 |
| 4 | The vista's residency in BYTES, and the client's own per-chunk budget | **SLICE 8** | §5.4, §9.4, §13 item 5 |
