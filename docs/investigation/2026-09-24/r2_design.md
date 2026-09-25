# R2 — one eroded level at 1 024 m in the artifact, and the noise only below it (believability step 3)

> **THE DESIGN OF PLAN STEP 3** (ruling B6 item 1 step 3; ruling B7 item 3, the second track). Nothing in this file is
> built. Nothing in it is measured. It was written while step 2 changes `crates/terrain/src/{lakes,solve,land,climate}.rs`
> and `crates/recipe/src/{height,plan}.rs` (uncommitted), so it designs against the code at `a985e5c` and names each
> place where step 2 touches it (§9). Line numbers in those files move with step 2's edits.
> - **How it was made.** Three read-only angles (bake and ship; kernel and rung continuity; operators and gates) wrote
>   their designs. This file joins them. Where they disagreed, the key lines were read again by hand and the code
>   decided (§2). The reference implementation's shaders were read from the copies the operators angle fetched on
>   2026-09-24 (`H-Schott/MultiScaleErosion`, MIT; session scratchpad, outside the repo).
> - **Marks.** `path:line` = read in the code. DOC = a number from a ruling or an investigation file, not re-measured.
>   ARITH = hand arithmetic. UNMEASURED = nobody printed it.
> - **The binding texts.** B6, B7 (`docs/design/owner_decisions_2026-09-22_believability.md`); W16, W17, W6 (the water
>   rulings); T9, T7 (the terrain rulings); F6, F7, F8, F9 (the frozen-patch rulings); SL10, SL6, SL5, HR5, SL8.

---

## 0. The short answer

- **What step 3 builds.** The planet's shard erodes the macro field once more, at 1 024 m, and stores the result as a
  new product beside the rows: **A0**, one whole-metre height per 1 024 m cell. The client reads A0 at rungs 0–10.
  The noise stays only under A0, as a smooth skin.
- **Why.** The dunes come from noise that owns every wave under 32 km (status §5 D1–D8). A0 gives every wave down to
  about 2 km to a solved drainage field. The bench (R1) measured the cure on one tile: Hack's law failed on the
  bicubic field (n = 0.883) and passed after erosion (n = 0.499) (DOC R1).
- **The loop** (one level, §4): the rows' own Catmull-Rom read at each 1 024 m cell, then 128 iterations of MFD
  routing, clamped stream power (m = ½, n = 1, the macro's own Earth law), the macro's talus law and a mass-conserving
  deposit, then THE MEAN HOLD (each 8 km node's mean equals its row), then THE MULTI-SCALE BREACH (4 km, 2 km, 1 km),
  then whole metres.
- **The bake** (§6): the planet's shard bakes one row tile (524 km) at a time, nearest first, on a share of the cores.
  Each tile is a pure function of the stored rows and a fixed halo. A prebake tool runs the same function for the home
  planet. The home planet costs **14–25 core-hours** (ARITH; the halo decides).
- **The ladder** (§7): rungs 0–10 read A0; rungs 11 and up read the head, unchanged. The only field swap moves from
  9 → 10 to 10 → 11, and its step stays under one rung-11 cell by a stated ceiling.
- **The ship** (§8): parts of 128 × 128 cells (32 KiB) on one new wire arm; about **10 MB** on a 150 km approach and
  **5 MB** on the reference stand (ARITH).
- **The prediction** (§10): the shipped dunes number at rung 7 goes from 2.84 closed basins ≥ 10 m per 100 km² toward
  the solve's own 0.003–0.1; G-HACK n from 0.22 toward 0.45–0.60. UNMEASURED.
- **Ten owner questions** (§14). The biggest is the skin (status §11 item 5).

**Example.** After step 3 a pilot at 150 km over the belt sees ridges 3–8 km apart and valleys that join, out to
890 km. Past 890 km the ground is the 16 km head, as today. When the ring at 890 km passes under her, the ground
moves by less than one 2 km cell.

---

## 1. What the code says now (the facts this design stands on)

1. **The noise owns every wave under 32 768 m.** `FINE_ABOVE_M = 4 × 8 192` (`crates/terrain/src/body.rs:94-96`).
   `first_fine_octave` reads that constant, not the body's lattice (`body.rs:778-790`). The same index sets the anchor
   band `S_PEAK` is solved over (`body.rs:804-814`, `:990-991`) and `slope_ref` (`body.rs:1011`).
2. **A column reads five words from its field** (`crates/recipe/src/plan.rs:138-156`): `z`, `first`, `slope_share`,
   `water`, `side`. The host picks `first`, so a new first octave under a new field needs no new kernel word
   (`crates/recipe/src/height.rs:300-331`).
3. **The roughness factor is the greater of a placeholder noise and the slope share**
   (`crates/recipe/src/height.rs:318`), and the share divides by `slope_ref` (`crates/terrain/src/artifact.rs:1705-1783`).
4. **A rung reads the rows or one pyramid level.** `PyramidField::level_for` gives the rows to every rung whose chunk
   is narrower than four nodes, else the finest level whose node is at least the cell (`artifact.rs:739-770`). On the
   home planet: rows at rungs 0–9, level 1 (16 km) at rungs 10–14.
5. **The pyramid is the nested floor mean of 2 × 2 blocks** (`artifact.rs:1443-1467`, `div_euclid(4)`). The head
   folds from the rows. Nothing finer feeds it.
6. **The tile is 64 × 64 rows of 10 B = 40 KB** (`artifact.rs:79`). The tile ring is rung 9's, pushed to ~627 km by
   W17's floor (`artifact.rs:825-943`).
7. **The four operators exist, integer, unwired** (`crates/recipe/src/amplify.rs:1-6`). `H_BITS` is 16
   (`amplify.rs:81`); the module doc still says 1/256 m (`amplify.rs:47`, stale). `SLOPE_N` is 2 and `drainage_power`
   is `a^{3/4}` (`amplify.rs:107`, `:233-240`).
8. **The bench is not a bake.** It upsamples in `f64` (`crates/bins/examples/erosion_bench.rs`, `build` and
   `catmull_rom`), reads the discharge through an `f64` `unlog`, sets `k` from the tile's own percentiles
   (`calibrate`), uses one chord pair for the whole tile, and pulls its tile off the face's rim because `node_at`
   reads at most two nodes past a face edge.
9. **The bench's deposit does not conserve mass.** The incision stores the cut (`sed_next[i] = c` in
   `pass_incise`), and `deposit` adds only `share × cut` to the load (`amplify.rs:347`, share 0.1). Nine tenths of
   every cut leave the field. (The operators angle said the bench's deposit conserves mass. The code says it does not.)
10. **The shipped pace counts messages, not bytes**: `TILES_PER_TICK` 4 (`crates/sim/src/stub/artifact_ship.rs:46`),
    `ARTIFACT_PARTS_PER_BEAT` 32 (`crates/connection-plane/src/gateway/artifact.rs:43`), `PARKED_TILES_MAX` 256
    (`crates/client/src/artifact_book.rs:327`). The gateway's note says "the most tiles one view reaches is NINE … a
    finer lattice would make the break live again: restore it" (`gateway/artifact.rs:470-490`).
11. **The client never rebuilds a chunk when a later tile lands.** Only the first artifact of a realm drops its
    chunks (`crates/client/src/chunks.rs:2195-2219`). A chunk at a rows rung waits for every tile of its stencil
    (`artifact_book.rs:162-176`).
12. **The store scans every tile at boot** (`crates/bins/src/artifact_store.rs:211`), and the head is refused on
    another world tag, which folds `GENERATOR_VERSION` (`artifact_store.rs:191-203`; `crates/terrain/src/tag.rs:174-183`).
13. **The macro law is Earth's:** `K₀ = 2 × 10⁻⁶ /yr` at m = ½, n = 1 over the discharge
    (`crates/terrain/src/solve.rs:83-88`); the talus moves half the excess a pass, split by excess, with a tangent
    from 25° (wet) to 35° (dry) (`solve.rs:93-107`, `:1139`); the sweep multiplies the province's erodibility
    (`solve.rs:980`; `crates/terrain/src/strata.rs:214-227`).
14. **The reference's released code** (read from the copies): the deposition is its own pass with rain 2.6, the stream
    to the power 0.3, and sediment made as `0.1 × streamPower` (`deposition.glsl:28`, `:179-193`); the thermal pass
    moves a fixed quantum with a noise-varied angle (`thermal.glsl:16-18`, `:86-110`); the erosion shader declares a
    hardness buffer and never reads it (`erosion.glsl:13-14`). The release holds no retargeting and no breach.

**Example.** Today a pilot at 120 km over the belt sees the 3 125 m ridged octave at full height on every hillside,
because rung 7 keeps it and the solve under it is 8 km wide. The code has no drainage anywhere under 8 km.

---

## 2. Where the three angles disagreed, and what the code decided

| # | question | bake angle | kernel angle | operators angle | DECIDED | why |
|---|---|---|---|---|---|---|
| 1 | levels in the loop | one, ×8 | — | three, ×2 each | **one, ×8** (the bench compares three, §11 slice 0) | each level adds its own halo (§6); R1 measured Hack passing at ×8 (DOC) |
| 2 | the law | bench's m = ¾, n = 2 | — | m = ½, n = 1 | **m = ½, n = 1** | the macro's own law (`solve.rs:83-88`); θ = 0.5 on both levels (W16) |
| 3 | iterations | 300 | — | 8 / 32 / 128 | **128 at 1 024 m** | the response-time law (§4.8), no fitted count |
| 4 | the mean hold | exact integer δ after rounding | smooth, then integer rest | 16 Catmull-Rom passes | **4 Chebyshev passes, ≤ 1 m** | the breach must run last (§4.9); the head reads the rows, never A0 (`artifact.rs:1443-1467`), so an exact fold buys nothing |
| 5 | when the hold and the breach run | once | once | every level | **once, at the end** | the survey's quote of the paper: "take a few seconds … once, at the end" (DOC) |
| 6 | which rungs read A0 | 0–6 A0, 7–11 A1, 12 A2 | 0–10 A0 | — | **0–10 A0** | `level_for`'s own rule, "the finest level whose node is at least the cell" (`artifact.rs:752-769`); a pilot at 150 km sees 1 km features to 890 km (8 px at 111 km, ARITH) |
| 7 | levels shipped | A0 + two folds | A0 | — | **A0 only** | one product, one swap, no fold-level steps |
| 8 | tile shape | 512² bake tile, 16 parts | 128² tiles | — | **bake by the row tile, ship by 128² parts** | the halo wants a big bake unit; the ask wants a small ship unit |
| 9 | wire | a new arm | a new arm or fused into `ArtifactTile` | — | **a new arm** (Q7) | a fused `rows` field is a repurposed message; 552 KB control messages block the lane |
| 10 | the skin | — | from 3 125 m, smooth, 128 m | from ≤ 2 048 m | **recommend ≤ 2 048 m** (Q1) | the 3 125 m octave is the measured dune signature (status §5 D3) |
| 11 | the drainage in the loop | rim inflow | — | exact sorted pass every 16 | **the Jacobi gather + rim inflow** | a global pass has no finite halo; the clamp makes the gather local (§4.3) |
| 12 | incision floor | — | — | lowest neighbour, sill, base level | **as operators** | no slot, no tie rule |
| 13 | deposition | — | — | "the bench's, conserving" | **conserving (the bench's is not)** | §1 fact 9 |
| 14 | thermal | — | — | macro law, 32-pass phase | **macro law, one pass per iteration** | the bench's loop shape; "32 passes clear any excess" holds for one pair only |
| 15 | hardness at a province edge | nearest node | a noise boundary | nearest node (Q) | **a ramp between node centres** | no staircase, no new draw (T9) |
| 16 | displacement ceiling | — | 1 024 m | — | **768 m** | Catmull-Rom's 2-D weight norm is 1.5625 (ARITH, §7.3) |
| 17 | store key | world tag | apart from `GENERATOR_VERSION` | — | **macro digest + `BAKE_VERSION`** | a skin-only change must not re-bake 20 core-hours |
| 18 | digest | own per tile | the artifact digest folds A0 | — | **own per tile** | a lazy product cannot sit in the macro digest |
| 19 | halo | H = N = 300 | — | — | **the seam gate decides; predicted 256 at N = 128** | §6.2 |

**Example.** The bake angle wanted the pilot at 150 km to see a 2 km mean from 55 km out. The code's own rule says a
cell can show a node no finer than itself, so rung 8 at 222 km reads the 1 km field, and the pilot sees the side
valleys the bake cut.

---

## 3. The field model and the data

**D1. A0 IS A NEW PRODUCT BESIDE THE ROWS, AND THE FINEST HEIGHT LEVEL.**
- **The lattice.** The macro lattice with its edge times f. f is the largest power of two, at most 8, that divides
  `cells_per_node` (`crates/terrain/src/macro_lattice.rs:122`). The home planet: f = 8, edge 9 728, node 1 024 m (one
  rung-10 cell), 567.8 M cells (ARITH). A body with odd `cells_per_node` gets f = 1 and no A0 (one rule, SL5; the count
  of such bodies in the home system is UNMEASURED, slice 1 prints it).
- **No head field.** Both hosts derive f from the body. `MACRO_EDGE_CEILING` 2 048 bounds the SOLVE's working state
  (`macro_lattice.rs:54-56`). The bake works in tiles, so A0's edge stands over it lawfully; A0 states its own
  ceiling (the bake's region memory, §6.4).
- **The word.** i16, whole metres over the ladder radius, the rows' own unit (`artifact.rs:97-100`). The loop works
  at `H_BITS` 16 and quantises once, by floor.
- **The record words stay on the rows.** Water, side, receiver, discharge, climate and province stay 8 km words. A
  column at rungs 0–10 reads Z from A0 and the rest from its nearest row, as today (`artifact.rs:1027-1159`).

**D2. THE BAKE READS THE STORED ARTIFACT AND NOTHING ELSE.** The input is the `Arc<Artifact>` the shard holds
(`crates/bins/src/artifact_boot.rs:98-108`) and the body. The solve's working state is gone after a reboot, so a bake
that read it could not resume. Every derived input (the runoff, the closed basins, the sea's reach) is a pure function
of the rows (§4, §9).

**D3. THE FOLD IS HELD TO ONE METRE, NOT TO THE BYTE.** Each 8 km node's nested floor mean of its 64 A0 cells stands
within 1 m of the row's `z_m` (G-FOLD, §10). The head and the far view read the rows, so they do not move by one byte
whatever A0 holds (`artifact.rs:1443-1467`). An exact fold would need the δ step after the breach, and δ re-opens
sills (§4.9).

**D4. A0 IS NOT IN THE MACRO DIGEST.** Each stored A0 tile carries its own digest and the macro digest it was baked
from. `TAG_ARTIFACT` stays the macro digest.

**Example.** The home planet's node (161, 1 204) on face −Z stands 8 306 m over the ladder radius. After the bake its
64 cells of 1 024 m hold a ridge at 8 700 m and a valley floor at 7 950 m, and their mean reads 8 306 m ± 1 m.

---

## 4. The operators and the loop (the bake of one region)

The loop is Schott, Galin, Guérin, Peytavie & Paris 2024's (ACM TOG 43(4), 10.1145/3658200), on the fenced integer
word, in this order.

### 4.1 The start: the stand-in

- **S = the shipped reader at each A0 cell centre:** `sample_z` over the rows (`artifact.rs:1641-1660`,
  `vd_recipe::macro_field::catmull_rom_16`), never the bench's `f64` Catmull-Rom.
- S is also the client's stand-in for a part not yet baked (§8.4). So before any erosion, A0 is exactly the ground a
  client draws today at those points, and R2's change is one clean difference.

### 4.2 The operators (`crates/recipe/src/amplify.rs`, pure stencils, no allocation, no float)

1. **Routing: MFD at p = 2** (Hyväluoma 2017's isotropy band 1.3–4.1, DOC R1). A method choice; G-GRID-OP checks it.
2. **Stream power: `E = K · √A · S`**, m = ½, n = 1, `K = K₀` with the macro's own discharge unit (`solve.rs:83-88`).
   One root, not two: `drainage_power` becomes `√a`. The clamps come from laws:
   - `s_max` = the repose tangent of the cell's node (past it the talus owns the slope);
   - `a_max` = ONE MACRO NODE'S CATCH (s² cells, s = 8). A river wider than that is the macro's; R2 refines it (W16).
   - The reference's `max_spe` is dropped.
3. **Incision floor: the LOWEST NEIGHBOUR'S HEIGHT** (a value, so no slot and no tie rule), plus two floors:
   - **the sill floor:** a cell within one macro node of a closed basin's divide may not cut under that basin's spill
     (B7 item 1; the spills come from a flood of the rows, §9);
   - **the base level:** a sea cell or a lake cell does not incise.
4. **Talus: the macro's own law** (`solve.rs:1139`): half the largest excess over the node's repose tangent leaves a
   cell, split by excess. One pass per iteration. It replaces the reference's 0.57 and its noise-varied angle.
5. **Deposition, mass-conserving.** The whole cut enters the load. The load rides the same MFD weights. Where the
   load exceeds the local stream power, a share settles (the reference's form: strength 1.0, share 0.1, stated method
   constants, Q2). A sea or lake cell takes load up to its water's level less one word, never over it. What is still in
   suspension after the last iteration leaves the field (the macro trunk carries it on); the mean hold returns the
   node's mass.
6. **Hardness: the province's erodibility share** (`strata.rs:214-227`), the share the macro sweep reads
   (`solve.rs:980`), RAMPED bilinearly between the four nearest node centres. The belt is one province (28/256), so
   the ramp matters only at province edges. No noise hardness (T9): a scatter enters only if G-GRID-OP is red, and
   then with its source named (Q2).
7. **Chords and areas per cell,** from the A0 lattice's own bend (`macro_lattice.rs:254`, `:272`), never one pair per
   level. The face bend makes a corner node 0.76 of a centre node's area (DOC operators angle).

### 4.3 The water: the local catch and the rim inflow

- **The local catch** of a cell = its node's RUNOFF × the cell's area, in the macro's own unit (mm·m²/yr). The runoff
  is the Turc–Pike share of the rain (`crates/terrain/src/climate.rs:616`). Its ratio to the rain depends only on the
  aridity, so the bake reads it from the row's rain and aridity bytes through one integer table of 256 entries.
  B7 item 2 rules the runoff; a desert then cuts little.
- **The drainage is the Jacobi gather only** (the bench's `gather`), from zero. No sorted accumulation runs inside the
  loop. The reason: with `a_max` at one node's catch, a cell's cut depends only on whether its own near catchment
  reaches that catch, which a Hack length of about 20–40 cells decides (ARITH), well inside 128 iterations.
- **The rim inflow:** on a region's rim, a cell whose outside node drains INWARD takes that node's discharge share in
  its catch (R1's corrected seeding, DOC R1). This is "MFD seeded from the row's discharge". After step 2 the row
  carries the runoff over the final receivers (B7 item 2).
- **No fluvial pass on a dry body.** A cell whose node's runoff is zero (the airless moon, a hyper-arid pan) neither
  incises nor breaches. Its fine level is the stand-in, the talus and the mean hold.

### 4.4 The dose: 128 iterations, from the law (T9)

- n = 1 makes a knickpoint travel at `c = K·√A`. At CFL ½ for the largest channel (`a_max`), the phase lasts the
  response time of the smallest channel (one cell) over one macro node: `N = 2·(node/cell)·√(A_node/A_cell) = 2s² = 128`
  (the operators angle's derivation, checked by hand). K, the runoff and the hardness cancel in the ratio.
- The factor 2 is the upwind scheme's CFL margin: at CFL 1 a trunk cell falls onto its floor in one step and makes a
  flat, and a flat stops the MFD water.
- In years (ARITH): `Δt = ½·Δx / (K₀·√(0.27 m/yr)·8·1 024 m) ≈ 60 000 yr`, so the dose is ≈ 7.7 Myr at Earth's mean
  runoff (the source of 0.27 m/yr is to be re-read at first hand). The fine valleys equilibrate with the macro field;
  they do not re-run the planet's age.
- G-DOSE and G-STEP (§10) test that the answer does not hang on the count.

### 4.5 The mean hold (the paper's height retargeting, our form)

- **The law:** every macro node's mean over its 64 cells equals its row's `z_m` (D3). The paper's own §5.1 is unread
  (UNVERIFIED); the survey reads it as "a diffusion that restores chosen points". Means, not points, for three reasons:
  the pyramid's fold is a mean (`artifact.rs:1463`); a pinned point every 8 km draws an 8 km lattice (G-GRID); and a
  diffusion run until it converges is a tolerance loop.
- **The method:** `Δ_N = z_N − mean_N`; add the Catmull-Rom of Δ (through the seam table). Four passes of a Chebyshev
  semi-iteration with coefficients from the spectrum of "the node mean of the Catmull-Rom". That spectrum lies in
  [0.39, 1] (ARITH: the 1-D Nyquist gain is 0.625, `2∫₀^½(4t³ − 6t² + 1)dt`; squared in 2-D). Four passes leave
  about 0.6 % of Δ (ARITH), so a 100 m residue falls under 1 m. Each pass reaches 2 nodes, so the hold reaches 8 nodes
  (64 cells), against 32 nodes for 16 plain passes.

### 4.6 The multi-scale breach (our form)

- **The terminals** (the macro's facts, never the node mask's square):
  - a SEA cell: under the sea's level AND reached by the flood from the connected ocean over cells under it;
  - a LAKE cell: in a lake node's footprint and under that lake's level (the row's `water_m`);
  - a CLOSED-BASIN cell: under its basin's spill, inside the basin's footprint (B7 item 1: the breach never opens a
    basin the rows hold closed);
  - in a tiled bake, a RIM cell whose node drains out of the region.
- **The flood:** a labelled Priority-Flood (Barnes, Lehman & Mulla 2014) keyed on (spill level, GLOBAL cell index):
  one total order on every host and every thread count.
- **The carve:** complete breaching along the flood's paths (Lindsay 2016's form; the attribution is to be confirmed):
  `h(qᵢ) = min(h(qᵢ), h_pit − i·ε)`, ε one `H_BITS` word (15 µm). It only lowers.
- **Multi-scale:** fold the field to 4 km and 2 km by the pyramid's rule; breach at 4 km first, prolong the carve to
  1 024 m as the Catmull-Rom of the 4 km carve clipped at zero (it only lowers); refold; breach at 2 km; then at
  1 024 m. A pit breaches at the coarsest scale where it exists, so its outlet is a valley 4 km wide, not a one-cell
  slot. The paper's §5.2 is unread (UNVERIFIED); this is our reading of "without one-cell canyon artifacts".
- **Only fluvial nodes:** a pit in a node with zero runoff stays (a crater on the moon, a playa in a dry interior).

### 4.7 The ceiling, the quantisation, the order

1. S (§4.1).
2. 128 iterations: route, gather, incise, talus, deposit (§4.2–4.4).
3. The mean hold (§4.5).
4. The multi-scale breach (§4.6).
5. **The displacement ceiling D_max = 768 m** on `|A0 − S|` (§7.3). The census counts its hits; the design wants 0.
6. Quantise to whole metres by FLOOR (monotone, so a descending path stays non-increasing).

The breach runs last among the shape passes, so the stored words drain; the carve moves node means by little
(G-CARVE, G-FOLD, §10).

### 4.8 One level, not three

The paper amplifies ×2 per level. This design upsamples ×8 once, as R1 did, for two reasons: each extra level adds its
own halo (4 km × its iterations, then 2 km × its iterations, about 96 cells at 1 024 m, ARITH), and R1 measured Hack's
law passing at ×8 (DOC). The breach stays multi-scale on the field's own folds. **Slice 0 runs three levels beside one
on the three stands' tiles; three levels win only if they move G-HACK n or G-BREACH where one level does not.**

**Example.** A hollow 30 m deep forms on the floor of a belt valley after the loop. At 4 km it is no pit; at 2 km it is
one, and the breach lowers a 2 km-wide sill by 12 m toward the trunk. The pilot sees a valley that runs on, not a pond
and not a slot.

---

## 5. The calibration (T9)

**D5. NO FITTED NUMBER.**
- K₀ is the macro's Earth calibration. m and n are the macro's law. The talus is the macro's repose law. The
  erodibility is Sklar & Dietrich's (`strata.rs`). The dose is the response time (§4.4).
- **Hack's law is the acceptance test, never the fit target.** B3 says a gate is never tuned to pass. Hack's n also
  hardly depends on the dose once a network forms (R1: 0.883 → 0.499), so a fit would say little.
- **The Earth stand (Q3):** our own instruments on public-domain Earth DEM tiles (GMTED2010 or SRTM): an orogen front,
  a passive coast, an interior plain. Fold each to 8 192 m by our pyramid's rule, run R2, and compare with the same
  tile's real 1 024 m field. That gives Earth's band IN OUR UNITS for G-HACK (with our D8 L bias in it), the spill
  census, G-BREACH, the spectrum, the hill spacing and the skyline. The tiles stay outside the repo; only numbers
  are recorded.
- **The fallback (Q3):** if the derived dose fails the Earth stand, one multiplier on the dose, fitted on the Earth
  orogen tile only, by the 2–16 km band power. The other Earth tiles and our four stands are the test set.

**Example.** The bake cuts the belt with the same K₀ that cut the belt's 8 km trunks. If the Earth orogen tile,
folded and re-cut, reads Hack's n = 0.52 and its real field reads 0.55, the law holds in our units; the home planet's
belt reading is then a reading, not a fit.

---

## 6. The bake: unit, halo, determinism, store, worker

### 6.1 The unit

**D6. THE BAKE UNIT IS THE ROW TILE:** 64 × 64 nodes = 512 × 512 A0 cells (524 km), keyed `(face, tx, ty)`, with a
fixed halo H. The region is every cell within H steps of the tile on the A0 lattice's neighbour graph, walked through
the seam table, so a face edge and a cube corner (seven neighbours) are exact. The square path inside a face is a
fast path and must equal the graph path byte for byte (a test). 432 of the 2 166 tiles have a halo over a face edge
and 24 touch a corner (DOC bake angle).

### 6.2 Determinism and the seam

- **The bytes:** a tile is a pure function of (the rows, the body, the tile id, H, the bake constants). Any order, any
  thread count and any machine give the same bytes. The four Jacobi passes split across threads exactly; the flood is
  sequential inside a region.
- **The seam:** two tiles meet along a 524 km line, each computed in its own region. The seam is exactly zero only if
  H covers every operator's reach. The reaches (ARITH): the loop up to 3 cells an iteration by construction, about 1
  measured (R1: halo 64 byte-identical at 64 iterations, DOC); the hold 64 cells; the breach is region-global in
  principle. **Predicted H = 256 cells (262 km)**, and G-AMP-HALO decides (a tile baked at H and at 2H: 0 differing
  stored words). If the strict gate needs more than 256, Q5 asks for a stated tolerance.
- **The partition and H are part of the bake's identity** (`BAKE_VERSION`).

### 6.3 Store

**D7. THE STORE HOLDS A0 PER TILE, AND AN INDEX.**
- A new row family `artifact_amp_tile(face, tx, ty)`: 512 KiB of cells (under the 1 MiB field cap), the tile's own
  digest, the macro digest it was baked from, and the bake tag (`BAKE_VERSION` folded with the halo and the constants).
- One AMP INDEX row: per tile, baked or not, and its digests (2 166 entries, ~87 KB, ARITH), written in the same commit
  as each tile. Boot reads the index, never a scan of 1 GiB (§1 fact 12).
- **The key is the macro digest and the bake tag, not the world tag.** A `GENERATOR_VERSION` bump that changes only
  the skin re-solves the macro (two minutes) to the SAME digest, and the baked tiles stand. A new macro digest deletes
  the amp rows by key from the index.
- `GENERATOR_VERSION` still bumps on any change of one chunk byte (`tag.rs:3`), bake changes included; `BAKE_VERSION`
  then bumps beside it.
- A tile row whose own digest is wrong: Q8.

### 6.4 The worker, the share, the prebake

**D8. THE PLANET'S SHARD BAKES, IN THE BACKGROUND.**
- A `BakeWorker` seam in the shape of `ArtifactWorker` (`crates/bins/src/artifact_worker.rs:52-133`): an inline worker
  for the Tier-A tests, a threaded one in the binary. Nothing bakes at boot and nothing on the tick thread.
- **Order:** the tiles a live view names (an occupant's or a gateway's want), nearest first; then an outward fill while
  the realm is alive (SL7). A planet nobody visits costs nothing.
- **Share (F6):** a quarter of the cores, at least one, at the lowest OS priority (Q4).
- **The prebake tool:** the same `bake_tile`, run outside play on the whole machine, writing the same rows. G-AMP-PIN
  proves the bytes equal. The home planet is prebaked; every other body bakes on demand (one machinery, SL5).
- **Memory per region:** ~1.05 M cells × ~72 B (R1's buffers) + the neighbour table and the per-cell chords on a seam
  tile ≈ 125 MB (ARITH).

**Example.** The home planet's shard boots, reads its index, and finds the belt's nine tiles baked by the prebake
tool. A pilot then flies to a far continent nobody has seen. The shard bakes the tile under her first (about 41 s on
one core), then its eight neighbours, while she flies over the stand-in.

---

## 7. The kernel and the rungs

### 7.1 The ladder

**D9. RUNGS 0–10 READ A0; RUNGS 11 AND UP READ THE HEAD, UNCHANGED.** The rule is `level_for`'s own second clause
carried one level finer: a rung reads A0 while its cell is no wider than A0's node.

| rungs (home planet) | ring (`869 m × 2^r`) | Z from | water, level, side, province | skin | shipped as |
|---|---|---|---|---|---|
| 0–5 | to 28 km | A0 | the nearest row | yes (§7.4) | A0 parts + row tiles |
| 6–10 | 28–890 km | A0 alone (no skin octave lives, `body.rs:203-232`) | the nearest row | none | A0 parts + row tiles |
| 11–14 | 890–14 000 km | head level 1 (16 km) | pyramid water, mask counts | none | head |
| 15+ | as today | as today | as today | none | head |

- **One home for the readers.** About forty readers compute `level_for`, the lattice and `coarser(level)` on their own.
  A new `FieldLadder` answers, for a rung, the height lattice and level and the record lattice and level. A reader that
  takes the rows' lattice for A0 is wrong by a factor of 8: the parent-frame unit trap again.
- **The tile rung** (`artifact.rs:825`) becomes 10 for both products, and `tile_reach_m` reads rung 10's ring:
  890 km × 1.1 × 1.17 ≈ 1 145 km (ARITH), bounded by the reach (`artifact.rs:803-806`).
- **`step_bound` under a field sums from the field's first octave.** Today `amplitude_bound` sums every live octave
  (`body.rs:1505`). Under A0 the octaves over 2 km are not drawn; if the bound still counts them it states steps of
  hundreds of metres at 7 → 8, 8 → 9 and 9 → 10 that nobody draws (DOC kernel angle) and pushes those rings out.

### 7.2 The swaps

- **0 → 1 … 5 → 6:** one skin octave drops at each, under one cell by the ladder clip (§7.4).
- **6 → 10:** nothing drops. W17's 9 → 10 swap (1 442.5 m, 1.41 cells, DOC) is gone: rungs 9 and 10 read one field.
- **10 → 11:** the one field swap, 1 km → 16 km. Its bound is §7.3.
- **The water words** swap at 10 → 11 too (rows → pyramid water). W6's lake seam (345 lakes vanish at 9 → 10, DOC step
  1) moves there; step 2's "small lake at pyramid level 1" must key on `FieldLadder`, not on the label "9 → 10".

### 7.3 The step at 10 → 11, and D_max

- The step = `CR₁ₖₘ(A0) − CR₁₆ₖₘ(L1)` = the displacement part + the fold part.
  - **The fold part** (rows → level 1) keeps W17's law: `slope_ref · 16 384 / (2√2)` = 722 m (`artifact.rs:871-902`),
    with `slope_ref` = 0.1247 unchanged (§7.4 keeps the anchor band).
  - **The displacement part** is the Catmull-Rom of `A0 − S`. Catmull-Rom's weights sum in absolute value to at most
    1.25 in 1-D (at t = ½: 1/16 + 9/16 + 9/16 + 1/16), so 1.5625 in 2-D (ARITH).
- **D10. THE DISPLACEMENT CEILING D_max = 768 m** (3/8 of a rung-11 cell): `1.5625 × 768 + 722 = 1 922 m`, under one
  rung-11 cell (2 048 m). T7's floor never pushes the ring out. Every host holds D_max, so no word crosses (W17's own
  property). R1's worst displacement was 226 m (DOC), so the ceiling should never bind; the census counts its hits.
  (The kernel angle's 1 024 m forgot the Catmull-Rom norm: `1.5625 × 1 024 + 722 = 2 322 m`, over one cell.)
- **Do not let W17 read the skin's slope.** With the skin's slope it states several kilometres and the ring goes to
  thousands of kilometres (DOC kernel angle).

### 7.4 R4's second half: the skin under A0 (Q1)

**D11 (recommended; the owner decides, status §11 item 5).**
- **Two thresholds, not one.** `FINE_ABOVE_M` does three jobs today (§1 fact 1). Split it:
  - **the anchor band stays 32 768 m:** `S_PEAK` stays 0.3998, no amplitude grows, `slope_ref` stays 0.1247
    (`body.rs:1011`), W17 unchanged;
  - **the skin under A0 starts under 2 × A0's node = 2 048 m** (the level's Nyquist): the first skin octave is
    1 562.5 m on the home planet. The host passes it as `first` (§1 fact 2); `FieldRead` needs no new index word.
- **Smooth only.** Under a field the ridged fold retires: its crest rings are the round closed pits of D5, its positive
  mean is the +287 m lift of D7 (status §5), and its lines cross solved valleys. R3 owns the 1–4 km lines later.
- **The ladder clip.** Each skin octave's amplitude at m = 1 is the lesser of today's spectrum and one cell of the rung
  that drops it: 1 562.5 m 80 → 64 m (dropped at 5 → 6); 781 m 25 m; 390 m 7.8 m; 195 m 2.5 m; 97.7 m 0.9 m; 48.8 m
  0.3 m. The skin's RMS slope at m = 1 is ≈ 0.37 (ARITH).
- **m = min(1, |∇A0| / TAN_REPOSE)** by the existing central difference at one A0 node. The skin's slope is then about
  half the ground's own slope, so on average it cannot turn a hillside's water uphill. On a floodplain m is near zero,
  and at a valley's axis the difference cancels, so the floor stays the solved floor.
- **The placeholder retires under a field (T9).** One word in `FieldRead`, all ones with no field and zero under one,
  masks both the placeholder and the ridged kind — a mask in the data, never a branch, as the card needs. A body with no
  artifact keeps today's arithmetic, so `gpu-drift` compares like with like.
- **The octave offset.** A per-octave integer offset now (every address under A0 moves at step 3 anyway); a rotation
  only if G-GRID on the skin alone (rung 3) shows a lean.
- **The stand-in uses the same skin,** so a stand-in column and a baked one differ only by the displacement.

**The option not recommended:** the code's own 4-samples rule (`MACRO_SAMPLES_PER_WAVE`, `body.rs:94-95`) gives
4 096 m, and the skin keeps the 3 125 m octave (199 → 128 m by the clip). It is the measured dune octave (status §5 D3:
the shipped hill spacing 3.2 km is that octave), and the reference stand's picture at 28–55 km would carry it. The 4
came from the macro design (03 §4.2.1); the owner's words for R4 are "the noise only below the solved level".

**Example.** A walker on a belt hillside where A0's slope is 0.3 feels a 1.5 km swell of about 27 m and a fine texture.
On the valley floor below her the ground is the solved floor, and it drains to the river.

---

## 8. The ship and the client

### 8.1 The parts

**D12. A PART IS 128 × 128 A0 CELLS (16 × 16 nodes, 131 km), 32 KiB,** keyed `(face, px, py)` on the A0 lattice and
aligned so that 4 × 4 parts are one row tile. The home planet has 76 parts a face edge and 34 656 in all. The bake
unit is the tile; the ship, the ask and the cache unit is the part.

### 8.2 The wire (Q7, SL6)

- **One appended arm:** `BulkMsg::ArtifactAmp { realm, face, px, py, from_digest, state }`, where `state` is `Baked
  { digest, cells }` or `Pending`. Discriminant 7, `PROTO_MINOR` 35 (`crates/wire/src/channels.rs:398-486`). An old
  client cannot log in anyway: `GENERATOR_VERSION` moves the declared world tag (`tag.rs:177-183`). The floor follows
  `crates/wire/src/version.rs`'s own ledger.
- **No new lane and no new want.** `ArtifactWant { view }` already names the view; the shard answers the parts within
  rung 10's reach beside the tiles, on the same realm audience (`artifact_ship.rs:1-30`).
- **Pending** tells the client to draw the stand-in. The shard keeps the want and pushes `Baked` when the bake lands.
  The gateway caches `Pending` until `Baked` replaces it.
- **Rejected: A0 fused into `ArtifactTile.rows`** (552 KB a tile): a repurposed field, and 552 KB messages on the
  Control lane block every other control message behind them.

### 8.3 The pace and the bounds (Q7)

- **The pace in BYTES:** 160 KB a tick a want at the shard (today's 4 × 40 KB) and 1 MiB a beat a session at the
  gateway (32 × 32 KB). The gateway's "restore the break" note (§1 fact 10) comes true with A0 and is restored.
- **Byte-bounded caches, least recently used out:** the client's book, the gateway's per-realm cache, the parked parts
  (proposed 128 MiB, 512 MiB, 8 MiB). Today no cache empties (§1 facts 10, 11), and a flight round the planet at
  150 km would pull the whole 1.06 GiB.
- **The shard** reads a tile row, cuts its 16 parts once, and keeps the encoded parts in a byte-bounded LRU
  (the "encoded once" lesson, `crates/bins/src/artifact_source.rs:56-63`). The tick cost of a read is measured the way
  the 90 ms digest cost was.

### 8.4 The client

- **A column at rungs 0–10** reads A0 where its parts are `Baked` and held, and S (the rows' Catmull-Rom at the A0
  node) where a part is `Pending`. Per A0 node, so the interpolant stays continuous across a baked–pending line.
- **A chunk waits** (F9: the coarser rung stands) for a part that is neither held nor known `Pending`.
- **Per-part rebuild (new code, §1 fact 11):** when a `Baked` part replaces `Pending`, only the chunks whose stencil
  reads it rebuild, and each vertex morphs radially from S to A0 over a stated time (about 2 s), the rung geomorph's own
  trick. A first arrival of a baked part while the chunk waits needs no morph.

**Example.** The pilot at 150 km over a far continent sees smooth 8 km hills (the stand-in, no dunes: the skin under it
is small because the stand-in's slope is small). Two minutes later the tile under her lands, and in two seconds the
hills settle into valleys. The globe past 890 km does not move by one metre.

---

## 9. Where step 2 touches R2 (ruling B7)

1. **The discharge row becomes the runoff over the final receivers** (B7 item 2). The rim inflow reads it. W10's
   1 304 002 disagreeing nodes (DOC step 1) must read 0 first, or the rim puts water where the tree sends none.
2. **The local catch reads the node's runoff** from the row's rain and aridity (§4.3). If step 2 changes those bytes'
   meaning, the table follows.
3. **Inland pockets become closed basins** (B7 item 1: 148 pockets, 3.2 M km², DOC). The bake finds every closed
   basin, its spill and its footprint by one priority flood of the ROWS from the connected ocean — a pure function of
   the rows, once per body, 8.9 M nodes. The breach treats them as terminals; the sill floor keeps their rims. It must
   agree with step 2's hierarchy: G-SILL (§10) reads both.
4. **More lakes** (the watershed what-if: 1.914 % of the land, DOC) mean more lake terminals and more sills.
5. **The lake seam moves to 10 → 11** (§7.2). Step 2's pyramid-level-1 lake must follow `FieldLadder`.
6. **One water (Q9).** The macro sweep still cuts with the rain (`solve.rs:940-1000`, status W10). If R2 cuts with the
   runoff, the two levels read two waters. That is step 2's to settle; this design flags it.
7. **Merge order.** Step 2 lands first with its own version bumps. R2's operators go into `amplify.rs` and a NEW
   `crates/terrain/src/amp.rs` (no conflict). R2 touches `artifact.rs` (`FieldLadder`, the tile rung, the swap law) only
   after step 2 merges. The bake reads the rows, never `lakes.rs`'s types, so a renamed `Lakes::deps` costs nothing.
8. **Pins.** Every A0 tile carries its macro digest, so each re-pin makes the baked tiles stale by key.
9. **The inland side word.** Step 2 adds `SIDE_INLAND` to the shore law (uncommitted, `crates/recipe/src/height.rs`
   and `plan.rs`): a column whose row says land and whose "own read stencil" holds no sea node keeps its own ground
   under the sea's level. Under R2 a column at rungs 0–10 reads Z through A0's stencil and the side through the ROWS.
   `FieldLadder` must name the rows' 16-node stencil as the one that decides `SIDE_INLAND`, or the word changes at the
   rung where A0 starts. And a basin floor under the sea's level is a CLOSED-BASIN terminal in the breach, never a sea
   terminal (§4.6), so the bake and the shore law read one fact.

**Example.** A pocket 2 km under the sea's level in a dry interior is, after step 2, a salt pan with a small lake at
its bottom. The bake's flood finds its spill, the breach leaves its floor alone, and the sill floor keeps its rim
whole, so the pan does not drain through a new 1 km notch.

---

## 10. The gates

Every line is a census line that can go red. "Today" is generator 19 (DOC `step1_census_gen19.txt`). Every predicted
number is UNMEASURED.

### 10.1 Identity and determinism

| gate | band | predicted |
|---|---|---|
| G-AMP-PIN: the digests of six tiles (belt, coast, plain, reference stand, one face-edge tile, one corner tile) on every server target | equal | equal |
| G-AMP-ORDER: the moon and one home tile, one thread and four, two tile orders | identical bytes | identical |
| G-AMP-HALO: a tile at H and at 2H | 0 differing stored words | H = 256 cells; the gate decides |
| G-AMP-PATH: the square fast path against the graph path | identical | identical |
| G-DRIFT (SL10): the golden fields gain the A0 nodes under the six rung-0 keys (~216 words, DOC) | byte for byte | pass |
| G-FOLD: each node's nested mean against `z_m` | within 1 m on ≥ 99.9 % of nodes, max ≤ 3 m | ≥ 99.9 % within 0.5 m |
| G-CLAMP: the D_max hits | 0 | 0 |

### 10.2 Continuity (W16, T7)

| gate | band | today | predicted |
|---|---|---|---|
| `rung_swap` 10 → 11 | ≤ 1 cell of rung 11 | (9 → 10: 1.41 cells, DOC) | 0.4–0.7 cells |
| `rung_swap` 0 → 1 … 5 → 6 | ≤ 1 cell | ≤ 0.64 px (DOC T7) | ≤ 1 cell by the clip |
| G-RUNG-LAKE / G-RUNG-LIP, at `FieldLadder`'s swap | 0 | 529 nodes / 116 nodes RED | step 2's cure, then 0 |
| G-BAND: every A0 node inside the ladder's band and under the relief cap | all | — | all |

### 10.3 Believability (B3 and step 1's lines; shipped ground, rung 7, ~206 km squares; belt / coast / plain)

| gate | band | today | predicted | the risk |
|---|---|---|---|---|
| G-HACK n (shipped) | 0.45–0.7 | 0.224 / 0.205 / 0.251 RED | 0.45–0.60 | MEDIUM: rung-7 sampling moves n (the solve read 0.735 at rung 7, 0.883 on the bench's 1 km grid, DOC) |
| G-HACK c (shipped) | 1–6 | 2.63 / 2.46 / 2.95 | 3–6.5 | MEDIUM: R1's 5.50 sits near the top |
| closed basins ≥ 10 m per 100 km² | a reading | 2.84 / 2.55 / 2.00 | 0.003–0.1: the solve's own basins, B7's, and the Catmull-Rom's diagonal sills | LOW at rung 7 (no skin lives there) |
| closed basins ≥ 100 m per 100 km² | a reading | 0.52 / 0.51 / 0.35 | 0 outside real basins | LOW |
| dry ground inside a basin ≥ 1 m | a reading | 35.6 / 41.3 / 9.8 % | ≤ 5 % | LOW |
| G-BREACH, shipped over solve | falls | 28× / 59× / 4.3× | ≤ 2× | LOW |
| hill spacing | reference 1–5 km | 3.2 / 3.2 / 4.2 km (the noise's peak) | 3–8 km, different per stand | the report says R2 moves it AWAY from 1–5 km on purpose; R3 owes 1–5 km (B6 item 2) |
| the fine relief's mean / RMS at rung 7 | 0 | +287 m / 330 m | 0 / 0 | none |
| the skin's own pits at rung 5 (new line) | a reading | — | < 0.2 per 100 km² | the skin decides it (Q1) |
| G-SPECTRUM (new): one power law over 2–32 km, no bump | β from the Earth stand | a bump at 3.125 km | no bump at rungs ≥ 6 | HIGH at rungs 0–5 if Q1 keeps 3 125 m |
| G-GRID long axis | 1.006 | 1.284 RED | > 16 km ≈ 1.28 (W16 forbids re-routing the macro D8); 2–16 km 1.03–1.15 | CERTAIN at the macro band: step 5's D8 decision stays open |
| G-HYPSO | two humps | GREEN, 55.12 % over 1 000 m | ± 1 % (the mean hold) | LOW |
| G-RIVER end / descent | GREEN | GREEN | GREEN | LOW |
| the teeth (`shore_teeth`) | a reading | median 2.9 km / 309 m at rung 7 | unchanged or re-patterned (the side stays the 8 km word) | step 4's; the report says so |
| the skyline, reference stand, breaks at 0.25° per 60° | 2.5–5.5 | 0 | 0.5–3 | HIGH: the stand rule picks a summit (status §9 ⚠) |
| the skyline's largest rise / range | 0.43–1.07° / 4.87–7.49° | 0.200° / 1.21° | 0.3–0.6° / 1.3–2.0° | HIGH |

### 10.4 Method checks (new)

| gate | band | predicted |
|---|---|---|
| G-BREACH-RESIDUAL: pits on the stored words outside terminals | none deeper than 1 m | 0 by construction |
| G-CARVE: the breach's lowering | p99 ≤ 10 m, max ≤ 50 m (stated, Q5) | p99 2–5 m; max 50–200 m where a trunk crosses a held ridge |
| G-SILL: a closed basin whose fine spill stands under the rows' spill by > 1 m | 0 | 0; MEDIUM on the pockets' long rims |
| G-REFINE-BASIN: cells > 1 node from a macro divide that keep their macro terminal | ≥ 99.5 % | 99.7–99.95 % |
| G-RIVER-FOLLOW: each macro trunk's fine thalweg within 1 node | ≥ 95 % of its length | ≥ 97 % |
| G-FLINT: the slope–area concavity θ | 0.35–0.6 | 0.40–0.55 |
| G-STEP / G-DOSE: (Δt/2, 2N) and (Δt, 2N) against (Δt, N) | < 5 % of the cut on 99 % of cells | UNMEASURED; a fail means the dose is a hidden knob, and the owner hears it |
| G-GRID-OP: the whole pipeline on `grid_bias`'s cone, Hyväluoma's rotation score | ≥ 0.95 at every angle | 0.93–0.99 |

**Example.** On the belt at rung 7, G-HACK n reads 0.52 and the closed basins ≥ 10 m read 0.02 per 100 km². The report
shows the belt picture beside the step-1 "before", and the hill spacing line says 5.1 km: further from the reference's
1–5 km band than the noise's 3.2 km, which R3 owes.

---

## 11. The order of work, in slices

Each slice ends with the census numbers and a picture (B6 item 2): the three stands and the REFERENCE STAND with its
skyline numbers. Where a slice changes nothing drawn, its picture is the bench's own (named in the slice). Every
slice updates `terrain_believability_status.md` in the same change (B6 item 4). One job at a time on the machine (B6.5).

| # | slice | what lands | ends with |
|---|---|---|---|
| 0 | **Bench II** (starts the day step 2 lands) | The operators in their homes, unwired: `amplify.rs` (n = 1, √a, the lowest-neighbour and sill floors, the macro talus, the conserving deposit, per-cell chords); `crates/terrain/src/amp.rs` (`bake_tile`: the region walk, S, the loop, the hold, the breach, the ceiling, floor). HR5 unit tests on stated tiny fields. The bench runs the belt, coast, plain and reference-stand tiles: ns a cell, H by G-AMP-HALO, G-HACK, G-BREACH ratio, spill census, G-FOLD, the displacement, the clamp hits, G-CARVE, G-STEP/G-DOSE, one level against three, G-GRID-OP. | The numbers; the bench's hillshade of the reference stand's 3 × 3 tiles beside S; the skyline marched over A0 from the reference stand's pose (the `skyline.rs` rule). The drawn world is unchanged. |
| 1 | **The bake as a service** | `BakeWorker`, the amp rows and the index, the nearest-first order, the share, the prebake tool; G-AMP-PIN, G-AMP-ORDER, G-AMP-PATH; the moon baked whole (runoff 0: no fluvial pass). | The prebake times; the four stands' tiles baked; the same bench picture and skyline, now read from the store. |
| 2 | **The kernel and the ladder** (after step 2 merges; `GENERATOR_VERSION` +1) | `FieldLadder`; `level_for` gives A0 to rungs 0–10; the tile rung, the reach, the swap law with D_max; `step_bound` from the field's first octave; the skin (Q1's answer), the field mask, the octave offset; S as the stand-in; the golden fields; the instruments: `shipped_drainage` variants (b) rows, (b′) A0, (a) shipped, and a rung-5 line; `rung_swap` at 10 → 11; `lake_census` on `FieldLadder`; a new `amp_census` (§10.1, §10.4). | THE DUNES NUMBER on the shipped kernel at rung 7 (the instruments bake in process), the swap numbers, the teeth reading. The picture is still the bench's (the client does not read A0 yet). |
| 3 | **The ship and the client** (`PROTO_MINOR` 35) | The arm, the source's parts, the byte pace, the gateway's bounded cache, the client's book, `Pending`, the per-part rebuild and the arrival morph. | THE PICTURES: the belt, the coast and the plain at 150 km, and the reference stand at 400 m, each "no far shadow"; the skyline numbers; the bytes and seconds of a 150 km approach; the frame cost with `VD_CLIENT_UNPACED=1`. The owner's look (G-LOOK for step 3). |
| 4 | **The gates in the gate** | HR5 at 100 % on `vd-recipe` and `vd-terrain`; the process-tier gates in release; DEFERRED entries (R3 reads A0; step 4 owns the 1 km side; the card's path, G-steps). | The full census and the status file's plan row 3 ✅ with its commit. |

**Example.** At the end of slice 2 the census prints `G-HACK-n:belt:shipped 0.52` and the swap line prints 0.6 cells
at 10 → 11, but the pilot's window still shows the dunes, because the client does not read A0 until slice 3. The
report says exactly that.

---

## 12. The costs (home planet)

| item | value | mark |
|---|---|---|
| A0 lattice | 9 728 cells a face edge, 567.8 M cells, 2 166 tiles, 34 656 parts | ARITH |
| the kernel's cost | ~280 ns a cell an iteration on one core (R1's 331.6 ns less one of pass ②'s two roots) | ARITH from DOC |
| one interior tile, H = 256 / H = 128 | region 1 024² / 768² cells; ~41 s / ~23 s on one core | ARITH |
| the whole planet, H = 256 / H = 128 | **~25 / ~14 core-hours** (halo-free: 5.7; the survey's 17 at 300 iterations) | ARITH |
| wall, 2 workers / 14-core prebake | ~12 h / ~1.8 h (H = 256) | ARITH |
| the first tile under a pilot, and its 3 × 3 | ~41 s; ~3 min on 2 workers | ARITH |
| memory per region in flight | ~125 MB | ARITH |
| store | 1.06 GiB of A0 + the ~87 KB index, beside the macro artifact's ~90 MB | ARITH |
| wire, 150 km approach | A0 ~9 MB + rows ~1.2 MB ≈ 10 MB, ~3.3 s at 3.2 MB/s (today ~0.4 MB) | ARITH |
| wire, reference stand (400 m) | ring = the reach, ~740 km: ~4.7 MB, ~1.5 s | ARITH |
| client memory for one view | ~10 MB, bounded by the book's LRU | ARITH |
| a stands dev loop | the four stands' ~60–80 tiles: ~4 min on 14 cores | ARITH |

**Rejected, with its number:** R2 derived on both hosts. The card does a tile in about 0.37 s, but the CPU fallback on
an average machine is about 81 core-minutes a view (DOC bake angle). So A0 ships.

---

## 13. The risks

1. **The skin decides the near picture.** At rungs 0–5 (the reference stand's first 28 km) the skin, not R2, sets the
   hill spacing and the pits. Q1 is the lever.
2. **G-GRID stays red at the macro band** by W16's own law. Step 5 owns the D8 decision.
3. **The dose may leave valleys too shallow or too alike.** G-STEP/G-DOSE and the Earth stand detect it.
4. **Hack's c near the top of its band** (R1: 5.50).
5. **The paper's §5.1 and §5.2 are unread at first hand.** The hold and the breach here are our own forms, each named.
6. **The tile seam.** The breach is region-global in principle. G-AMP-HALO measures; Q5 is the fallback.
7. **The first visit.** A planet nobody baked shows the stand-in for minutes, then each tile morphs in (Q6).
8. **The collider (future).** A tile that lands moves the ground under an occupant by up to D_max. When the collider
   lands, a bake landing must use the edit path. Today there is no server terrain collision, so only the picture moves.
9. **A rebuild storm** on the client from per-part invalidation (new code).
10. **The teeth stay until step 4.** The 8 km side against a 1 km ground may raise new walls; `shore_teeth` reports.
11. **Diagonal sills.** A channel that steps diagonally on the 1 km lattice gets a sill in the Catmull-Rom between its
    nodes; the rung-7 census may count basins the breach never saw. A candidate cure (4-neighbour carve paths) is
    UNMEASURED.
12. **Dry interiors stay smooth.** With the runoff as the catch (B7), a hyper-arid plain barely erodes at 1 km: honest,
    and possibly bland in the plain's picture.
13. **Face edges.** The slope share uses the face-centre node size (`artifact.rs:1705-1731`); near an edge m is off by
    the bend's ratio. UNMEASURED.
14. **The reference stand's rule picks a summit** (status §9 ⚠): its skyline may not show what the belt holds.
15. **HR5 is owed** for every changed line in the two Tier-A crates.

**Example.** If slice 0 prints G-HACK n = 0.30 on the belt with three levels as with one, the dose or the law is short;
the report stops the step and asks, and no tile is baked for the window.

---

## 14. The owner questions (each with a recommendation)

1. **Q1 — THE SKIN (status §11 item 5).** Which octaves stay under A0, how tall, and in what form?
   **Recommendation:** from under 2 048 m (1 562.5 m on the home planet); smooth only; clipped by the ladder (80 → 64 m);
   m = A0's slope over the angle of repose; the placeholder and the ridged fold retire UNDER A FIELD ONLY; a per-octave
   offset now, a rotation only if measured. The other option (the 4-samples rule, 3 125 m kept at 128 m) keeps the
   measured dune octave in the near field.
2. **Q2 — THE OPERATORS' LAWS.** m = ½, n = 1 (the macro's own, θ = 0.5) in place of the bench's ¾ and 2; the
   deposition made mass-conserving with the reference's constants (strength 1.0, share 0.1) kept as stated method
   constants; a hardness scatter only if G-GRID-OP is red, with its source named.
   **Recommendation:** yes to all three. (Davy & Lague 2009's ξ–q deposition law is the lawful replacement if the
   floodplains read wrong; to be read at first hand.)
3. **Q3 — THE CALIBRATION.** No fitted number; Hack's law as the acceptance test; the Earth stand on public-domain DEM
   tiles kept outside the repo; if the dose fails, one multiplier fitted on the Earth orogen tile only.
   **Recommendation:** yes.
4. **Q4 — THE BAKE'S SHARE (F6).** A quarter of the server's cores, at least one, at the lowest OS priority; and a
   whole-machine prebake tool outside play for the home planet (same function, same rows).
   **Recommendation:** yes; ~25 core-hours once for the home planet.
5. **Q5 — TWO STATED NUMBERS UNTIL EARTH'S BAND EXISTS.** The seam's fallback if strict byte identity at H and 2H needs
   H > 256 (then ≤ 1 m on ≤ 0.01 % of edge cells, and no carve path differs); and G-CARVE's tolerance (p99 ≤ 10 m,
   max ≤ 50 m).
   **Recommendation:** ask only if slice 0 measures a need.
6. **Q6 — THE FIRST VISIT.** On a tile not yet baked, the stand-in shows, each tile morphs in over ~2 s when it lands,
   and a shape line stands at the edge between baked and unbaked tiles while the bake runs. May this be a registered
   transient?
   **Recommendation:** yes; it is the only road for a planet nobody visited. The home planet is prebaked.
7. **Q7 — THE WIRE (SL6: a new arm).** `BulkMsg::ArtifactAmp` with `Baked`/`Pending`, `PROTO_MINOR` 35; the pace in
   bytes (160 KB a tick a want, 1 MiB a beat a session); the byte bounds (client book 128 MiB, gateway 512 MiB, parked
   8 MiB). The data is "more of an approved row"; the arm is new.
   **Recommendation:** yes. Fusing A0 into the tile's `rows` repurposes a field.
8. **Q8 — A CORRUPT A0 TILE.** Refuse, as a lying store refuses today (`artifact_boot.rs:9-12`), or re-bake it?
   **Recommendation:** today's law: the shard names the tile and refuses; never a quiet re-bake.
9. **Q9 — ONE WATER (for step 2's track).** Should the macro sweep also cut with the runoff, so both levels read one
   water?
   **Recommendation:** yes, and in step 2, before R2's first bake.
10. **Q10 — THE DISPLACEMENT CEILING.** D_max = 768 m, derived from the ladder's one-cell tolerance at 10 → 11.
    **Recommendation:** yes; it binds nowhere on R1's tile (worst 226 m).

**Not asked, decided:** no 1 km drainage byte ships (R3 and 8e route A0 themselves, a pure function of shipped data);
the province hardness ramps between node centres (no new draw); the A0 store is keyed on the macro digest and
`BAKE_VERSION`.

---

## 15. Corrections to the record found while writing

1. **The bench's deposit does not conserve mass** (§1 fact 9): one tenth of each cut enters the load. The operators
   angle said it does.
2. **The survey's diff table** says thermal and deposition are "the same operator; their constants carried verbatim"
   (`docs/investigation/2026-09-22/fine_terrain_models_and_crates.md`, R1 "The diff against the reference"). The
   reference's released shaders differ (§1 fact 14).
3. **`amplify.rs:47`** says a height is 1/256 m; `H_BITS` is 16 (`amplify.rs:81`). Its lines 3–6 ("the bench only")
   change when slice 0 lands.
4. **The kernel angle's D_max of 1 024 m** misses the Catmull-Rom norm (§7.3).
5. **The kernel angle's "the artifact digest folds the 1.14 GB field in"** is replaced by per-tile digests (D4).
6. **The bake angle's exact fold** keeps nothing the head does not already keep, and it would re-open the breach's
   sills (D3).
7. **"32 thermal passes bring any excess under one word"** holds for one pair of cells, not for a chain.

**Example.** A reader of the survey who takes the bench's deposit as mass-conserving would build a bake that loses nine
tenths of every valley's rock and blame the mean hold for lifting the ridges. This section names the line instead.

---

## 16. THE SKEPTICS' FINDINGS (2026-09-24, workflow `wf_dfee77e9-dca`) — OPEN, and they bind the design above

Two adversarial readers attacked this design against the laws and against the measurements. Every finding below is OPEN.
Where a finding contradicts a decision above, THE FINDING WINS until slice 0 measures otherwise. Nothing is built.

| # | severity | finding | the fix it asks | sections |
|---|---|---|---|---|
| 1 | HIGH | The stand-in and the late bake change the land's shape in time | 1 | §8.2, §8.4, §6.4 D8, §13 risks 7–8, Q6 |
| 2 | HIGH | The bake's water and the kernel's side are two facts, and the shore law turns A0 ground inside out | Before the client reads A0 (slice 3), make one fact say where the water is: 1 | §3 D1, §4.2 item 5, §4.6, §9 item 9, §13 risk 10 |
| 3 | HIGH | The mass-conserving deposit gives each channel back its own cut, so the valleys stop growing about 20 km below their heads | Keep the bench's deposit (sediment made as a share of the cut) | §4.2 item 5 (deposition); §2 row 13; §15 item 1 |
| 4 | HIGH | The dose derivation holds on no belt cell: one Δt cannot cancel both the hardness and the runoff | Decide and state Δt before slice 0, and give the owner the consequence in numbers | §4.4 (the dose); §4.2 items 2 and 6 |
| 5 | HIGH | No measurement supports the predicted cure: R1 passed Hack's law with a different loop | In slice 0, run R1's own configuration on the same belt tile as a CONTROL beside the design's loop | §0; §2 rows 2, 3, 13, 14; §5 D5; §10.3 |
| 6 | MEDIUM | The tile seam is tested on one tile, and Q5 makes a seam lawful | 1 | §6.2, §4.6, §4.7, §10.1, Q5 |
| 7 | MEDIUM | The dose's time step is not stated, and each possible reading breaks a law | State ONE physical duration T for the fine level, computed once per body from the rows (region-free), and state the law it comes from | §4.2 items 2 and 6, §4.3, §4.4, §10.4 G-STEP/G-DOSE |
| 8 | MEDIUM | Two groups of physical numbers have no law: the deposit's constants and the skin's height | 1 | §1 fact 14, §4.2 item 5, §7.4 D11, Q1, Q2 |
| 9 | MEDIUM | The A0 store stands outside the world epoch, and only a hand bump guards it | 1 | §6.3 D7, §3 D4, §10.1 G-DRIFT |
| 10 | MEDIUM | D9 is not level_for's rule; the new 16× swap has no true bound and no shade or colour line | 1 | §2 row 6, §7.1 D9, §7.3 D10, §10.2 |
| 11 | MEDIUM | The bake's core share is per shard, not per machine (F6) | Use one bake budget per HOST (a pool for the whole host, or the pod's CPU quota) that every shard on the host shares | §6.4 D8, Q4 |
| 12 | MEDIUM | The closed-basin line falls with R4 alone, so it cannot show that A0 works, and the plan reads A0 at rung 7 only after it builds the bake service | Print S (the stand-in) beside A0 on every line in every slice | §10.3; §11 slices 0–2 and its example |
| 13 | MEDIUM | At 100–150 km, 1 024 m removes the dunes, but the ground has no detail between 1 px and 12–18 px | Put the expected look into §0 and into the slice-3 picture captions | §0; §7.1; §10.3 (hill spacing, fine relief) |
| 14 | MEDIUM | At 1–10 km altitude, the recommended skin makes a new one-peaked bump: isotropic, and with no drainage under 2 km | Add a G-SPECTRUM line at rungs 0–5 for the recommended skin, and a hill-spacing line at rung 3 on the belt | §7.4 D11; Q1; §10.3 G-SPECTRUM |
| 15 | MEDIUM | Diagonal thalwegs hold closed basins in the rung-7 reading, and the breach never sees them | In slice 0, read A0 through the shipped kernel at 128 m and count the closed basins at diagonal steps | §10.3 closed-basin row (rated LOW); §13 risk 11; §4.6 |
| 16 | MEDIUM | A far planet needs 25–36 tiles for one view: 8–15 minutes at the average machine's share, and a fast pilot flies faster than the bake | In §12 and Q6, state the first-visit time for one view, not for a 3 × 3 block | §6.4 D8 and its example; §12; Q6 |
| 17 | MEDIUM | The bake share is per shard, but F6 gives the terrain a share of the MACHINE | Give the bake one budget for each HOST, set by the deployment (for example the processes' CPU limit), not by each shard | §6.4 D8; Q4 |
| 18 | LOW | G-FOLD cannot pass as written, and the clamp after the breach can make new sills | Put the clamp before the breach, so that the breach is the last shape pass | §3 D3, §4.7, §10.1 G-FOLD, §10.4 G-BREACH-RESIDUAL and G-CARVE |
| 19 | LOW | Two region walks for one job | Keep one walk, the graph path, unless slice 0 measures the walk as a stated share of a tile's bake time | §6.1 D6, §10.1 G-AMP-PATH |
| 20 | LOW | The rim inflow never reaches the tile, so the macro discharge changes no stored byte | Say it plainly in §4.3 and §9: the fine loop reads the runoff, not the discharge | §4.3; §6.2; §9 item 1 |
| 21 | LOW | The cost table uses a rate below R1's measured wall rate, and it misses the halo as the main cost lever | Use R1's measured wall rate in §12, and measure the fused interior loop in slice 0 | §12; §6.1 D6 |
| 22 | LOW | The G-FOLD band and the G-CARVE prediction cannot both hold | Predict G-FOLD's maximum from G-CARVE's, or state that the fold is not held after the breach (the head reads the rows, so no byte outside A0 moves) | §4.7; §10.1 G-FOLD; §10.4 G-CARVE |
| 23 | LOW | The 10 → 11 handover goes from 1 km to 16 km in one ring, and the gate bounds only the height | Add a shade (normal) step to rung_swap at 10 → 11 | §7.2; §7.3; §10.2 rung_swap |

**What they change before any code (the order of work, §11):**
- Slice 0 (the bench) must run R1's own loop as a CONTROL beside this design's, change ONE thing at a time (the law, the rate,
  the count, the deposit, the talus), and read every line THROUGH THE SHIPPED Catmull-Rom at 128 m columns, printing the
  rows alone (S) beside A0 on every line — G-HACK c AND n, the hill spacing, G-BREACH, the spill-depth census. No bake
  service (slice 1) is built before that table exists (finding: no measurement supports the cure with this loop).
- Keep the bench's deposit (a share of the cut) or a transport law with a calibration body; the mass-conserving form stops
  the valleys 20-30 km below their heads (HIGH).
- State Δt and the dose ONCE per body, and tell the owner what the hardness and the runoff then shape (HIGH).
- Never draw a stand-in that A0 later replaces: an unbaked tile's rungs WAIT and the coarser rung stands (F9), and the bake
  belongs to the realm's warm-up, on a stated share of the HOST's cores (F6), never the land's shape in time (T3, HIGH).
- The bake takes its sea, lake and basin terminals from the words the kernel reads (one fact for the water, W16, HIGH).
- Show the owner a hillshade of A0 at the 100-150 km pixel (115-173 m) BEFORE slice 1: at 1 024 m the far view loses the
  dunes but keeps no detail between one pixel and ~2 km — valleys 2-8 km apart with smooth walls; the 1-2 km band and the
  near field (the skin under 2 km, still isotropic noise) wait for R3 (MEDIUM, and the owner's target picture).
