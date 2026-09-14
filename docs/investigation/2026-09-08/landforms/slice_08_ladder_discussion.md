# Slice 8 — the detail ladder on the client (with 8p, the picture instrument)

Written 2026-09-09 for the owner, before the build. ASD-STE100, in-game examples, drawings where they
help. The order is ruling V13's addendum (foundation first): slice 8 is the first step after the
step-0 measurements, and it lands with the instrument (8p) and the light fix (8L, already landed in
step 0).

## 1. What this slice decides, and what it leaves to later slices

**Decides.**
- **Which rung a chunk column is drawn at**, from ONE angular rule and ONE reference view. Today one
  rung per realm is named by a dev flag (`D-TERRAIN-3`, red). This slice deletes the flag and the
  one-rung assertion.
- **How two rungs meet** without a visible step: coarse before fine, and a crossfade band whose width
  comes from the ladder's own bound.
- **How much ground the client holds around the eye** (the residency band), sized from SERVER-STATED
  data only, and the distance at which a body's whole-body data must be complete.
- **What replaces the realm's proxy outline** when terrain is drawn: the rung ABOVE the coarsest one
  drawn, from the same crate, so the globe is never black beyond the patch.
- **How a picture is judged** (8p): a measured stamp on every frame, a probe that says what drew each
  pixel, a ruler of known size, and the light stated. Every later slice's pictures use it.

**Leaves to later slices.**
- The shape itself (the spectrum, the relief law, the charter): slice 8a.
- The macro artifact and its transfer deadline: slice 8b and 8c. This slice only STATES the distance
  at which whole-body data must be complete (the deadline), because nothing names it today.
- The pop detector under the real character (slice 16); the legs here are flown by a hull.
- The paint, the haze, the sky: after the freeze (V13 addendum).

**Example, in the game's words.** A pilot dives at the home planet in her hull. At 900 km the planet is
drawn at rung 10 (1 km cells). At 470 km the rung-9 chunks arrive under the rung-10 ones and take over
through a short crossfade that no frame shows as a step. At 1.8 km the rung-1 chunks arrive; at 500 m
she sees rung 0. She never sees a hard edge, a hole, or a jump, at 240 m/s or at 1.4 m/s.

## 2. The words, explained once

| Term | What it means, and an example here |
|---|---|
| **Rung** | One level of the ladder: rung `L` has cells of `2^L` metres. Rung 0 is the metre; rung 11 on the home planet is 2 048 m. |
| **Tier rule / angular rule** | The rule that picks the rung for a chunk column from how big one cell LOOKS: a cell that subtends less than one pixel at the reference view is drawn one rung coarser. One rule for every realm kind, never a branch on what the realm is. |
| **Reference view** | The one camera the rules are stated against: 45° vertical field of view over 720 rows. One pixel is 45° / 720 = 0.0625°. Every screen scales from it; a finer screen moves the switch distance out and shrinks the pixel by the same factor, so the pixel figures do not change. |
| **Switch distance** | The distance at which one cell of rung `L + 1` is one pixel high: `cell_m(L + 1) / pixel_rad`. On the home planet: 1.8 km for rung 1, 14.7 km for rung 4, 939 km for rung 10 (MEASURED, `rung_disagreement`). |
| **Free-coarsening law** | Rung `L`'s height is rung 0's height with the finest `L` octaves left out, so the far view is the near hill minus the small bumps. The BOUND is the sum of the dropped amplitudes; the coarse rung can state it. |
| **Rung disagreement** | How far apart the ground stands at rung `L` and rung `L + 1` along the same direction. MEASURED today: at most 0.37 cells and 0.37 pixels at the switch distance, at every rung (M8-0). |
| **Crossfade band** | The range of distances over which two rungs are both drawn and blended, so the switch is a fade, not a step. |
| **Dither** | The blend done per pixel by a fixed pattern that drops pixels of the vanishing rung and keeps pixels of the arriving one, instead of transparency. It costs no sorting and no second pass. |
| **Coarse before fine** | The coarser rung of a column is drawn first; a finer rung may replace it only when it has arrived. So there is never a hole where a fine chunk is still building. |
| **Residency band** | How much ground around the eye the client keeps built: enough that the ground under the eye is never missing for one frame, at the speed the OWNING shard states. |
| **Server-stated lead** | The number the realm's shard states about how far ahead to build. The client never works a speed out from two delivered positions: SL10 clause 7 forbids a derived velocity. |
| **Hysteresis** | A switch that flips at one distance on the way in and at a different distance on the way out, so a hover on the boundary does not flap. |
| **Drawable floor** | The smallest thing the reference view can show: one pixel. A realm's reach and a rung's switch both read it (`vd_core::geometry`). |
| **Pop detector** | A gate that flies a hull through the ladder and asserts that no pixel on a rung boundary changes by more than the dither's own noise between two frames. |
| **The stamp** (8p) | The measured facts written on every judged picture and its state file: altitude, horizon, radius drawn, rungs and chunk counts, the star's angle, the biome, the identity, the tick. |
| **The probe** (8p) | A second buffer aligned to the picture in which each pixel says WHAT drew it (which realm, which rung, which mesh) and HOW FAR it is. It replaces the colour classifier of slice 7. |
| **The ruler** (8p) | A subject of known size in frame, measured in pixels against its prediction. |

## 3. The path from a delivered row to a drawn ladder

```text
   THE WINDOW (delivered)           THE CLIENT LIBRARY (Tier-A)              THE ENGINE (Tier-B)
   ----------------------           ---------------------------              -------------------
   the realm's row:                 for each realm with a body:
     placement, facing                 eye in the body's frame  (display arithmetic)
     surface statement                 |
     reach, the interpolation          v
     buffer, the STATED LEAD  -----> the RESIDENCY BAND: how far out, per rung
                                       |
                                       v
                                    the TIER RULE: for each chunk column in the band,
                                    the rung whose cell is >= one pixel at the reference
                                    view, with hysteresis
                                       |
                                       v
                                    the WANTED SET, coarse before fine ------> the workers build
                                       |                                       (threads, the seam)
                                       v
                                    the CROSSFADE: per column, the blend weight
                                    between the arriving rung and the leaving one,
                                    from the distance and the band ---------> the dither, in the
                                       |                                       material
                                       v
                                    the RUNG ABOVE: the coarsest drawn rung
                                    + 1, from the same crate, at the body's
                                    radius, replaces the proxy outline ------> the globe, never black
                                                                               |
                                                                               v
                                                                   8p: the stamp, the probe,
                                                                       the ruler, the light stated
```

Nothing new crosses a realm boundary in this slice. The lead is the one open question (§9 D8-3).

## 4. The tier rule

**One rule.** A chunk column at distance `d` from the eye is drawn at the finest rung `L` whose cell
still stands one pixel high at the reference view: `cell_m(L) ≥ d · pixel_rad`. Every realm — a planet,
a moon, a hull with terrain — reads the same rule. No branch on the realm's kind (HR3, HR4).

**Hysteresis.** A column switches to the finer rung at `0.9 · switch distance` on the way in and back
to the coarser at `1.1 · switch distance` on the way out. The two factors are ONE named constant pair,
and the pop detector measures them; a hover on the boundary then never flaps.

**The coarsest rung drawn, and the rung above it.** Beyond the band's outer edge the body is drawn at
the rung above the coarsest one in the band, from the same crate at the body's own radius: a coarse
mesh of the whole visible cap, built once and kept. Today the proxy outline is hidden while terrain is
drawn, and the globe is black beyond the patch (`D-TERRAIN-3`). After this slice the globe is the
recipe's own coarse rung.

**Example.** The pilot stands on the ground. Columns within 1.8 km are rung 0, out to 3.7 km rung 1,
out to 7.3 km rung 2, and so on to the horizon at 3.5 km on this small planet (COMPUTED, `04` §5.4).
From 60 km up, the ground under her is rung 5 and the cap of the planet is rung 11.

## 5. The crossfade

**The width comes from the bound.** The coarse rung can state how far it may differ from the fine one
(the sum of the dropped amplitudes). MEASURED today the disagreement is at most 0.37 pixels at the
switch distance, so a band of a few percent of the switch distance hides it; under the spectrum
(8a) the bound reaches about two cells on alpine ground (`04` §5.2, COMPUTED) and the band widens
there and stays narrow on a plain. The band's width per column is therefore a function of the bound
per column, which the fold states before its loop.

**The dither.** Inside the band both rungs' chunks are drawn; the material discards pixels of the
leaving rung by a fixed screen-space pattern and keeps pixels of the arriving one by the complementary
pattern, with the weight from the column's distance inside the band. No transparency, no sorting, no
second pass. The pattern is the same on every frame, so a still camera sees a still picture.

**Coarse before fine.** A column's finer rung is requested when the column enters the band; until it
arrives, the coarser rung stays. A finer chunk that arrives is placed with weight zero and fades in;
the coarser one fades out and is released when its weight reaches zero. A chunk is never released
while it is the only one drawn for its column.

## 6. The residency band, the lead, and the deadline

**Sized from server-stated data only.** The band's outer edge is the realm's own REACH plus the STATED
LEAD, and the inner edge is the eye. The lead is how far ahead the owning shard says to build. The
client reads that number; it never subtracts two delivered rows and calls the difference a speed
(SL10 clause 7).

**What data exists today.** The realm's reach (on its row), the interpolation buffer (on the wire's
tuning), and NO lead. So one of two things is true: (a) the reach plus the buffer is enough for every
speed a hull can fly inside a planet's realm, which M8-1 measures; or (b) a lead is needed and the ask
R-18 (a realm states a lead beside its reach) goes to the owner (§9 D8-3). The slice starts with (a)
and measures.

**The deadline this slice states.** The band names the distance at which a body's WHOLE-BODY data must
be complete before the client builds a chunk of it. Today that data is the surface statement alone.
From slice 8b it is the macro artifact, and this distance becomes its transfer deadline. Nothing names
that distance today (`06` U-L4); after this slice the band does.

**Example.** A hull approaches the home planet at 528 m/s. The planet's reach is its look radius; the
interpolation buffer is 120 ms (`INTERP_BUFFER_MS`; this text first said 150 ms); at 528 m/s that is
63 m of lead, which is one rung-0 chunk. If M8-1 shows the
band goes incomplete at that speed, the realm must state a lead, and the ask is made.

## 7. What the client holds, and the bytes

COMPUTED (`04` §5.4) for a 50 km vista from a high stand on the home planet: about 1 000 surface chunk
columns across rungs 0, 3 and 6, and 1 000 to 2 000 chunks with the vertical stack. MEASURED per
rung-0 chunk today: 405 228 bytes (positions, normals and indices at four bytes a component), so a
vista is 400 to 810 MB of mesh in the client's memory. That is too much for a shipped client and this
slice owns the lever:

| Lever | Bytes per chunk | Source |
|---|---|---|
| today: f32 positions, f32 normals, u32 indices | 405 228 | MEASURED (M7-3) |
| i16 positions (the extractor already stores them) and u16 indices | 254 076 | COMPUTED |
| plus normals packed to four bytes | 185 460 | COMPUTED |

The packing is a client-side choice; the extractor's vertices are `i16` already. §9 D8-4 asks which.

## 8. The measurements the slice owes, FIRST

| # | What | How | Gate | State |
|---|---|---|---|---|
| **M8-0** | The rung disagreement in pixels at the switch distance | `cargo run --release -p vd-bins --example rung_disagreement` | p99 ≤ 1 px, max ≤ 2 px | **MEASURED: max 0.37 px, p99 0.24 px** |
| M8-1 | The chunk arrival rate during a scripted descent at 1.4, 240 and 528 m/s, against the thread budget | a hull leg through the harness, the lane's pending count as the instrument | the band is never incomplete for one frame; the deepest queue reported | **MEASURED (§16.2):** the walk holds; 240 m/s at 1 km up is ON the wall (three red of six); 528 m/s breaks on every frame |
| **M8-2a** | THE THREE RATES on the M8-1 flight (ruling V15): what the workers build, what the engine harvests, the frames — and the phases of one build (`chunk_phases`) | the stamp's counters read across each leg; `cargo run --release -p vd-bins --example chunk_phases` | which rate binds, named | **MEASURED (§16.5):** the WORKERS bind — 60–64 ms a chunk on the flight against 5 ms with the parent meshes warm; the harvest never fills its cap at 240 m/s |
| M8-2 | The vista census: chunks, vertices, bytes, fill time at a 50 km stand, rock only | the capture client's counters | under a stated ceiling (§9 D8-4); the fill under 2 s at the client's thread count | owed |
| M8-3 | The pop detector on three hull legs | frame-to-frame difference on rung boundaries | no pixel changes by more than the dither's own noise | owed |
| M8-4 | The stamp's own truth | the stamp's altitude and horizon against the state file and the ruler's pixels against its prediction | equal within one pixel | owed |

## 9. What you decide

| # | Question | Options | Recommendation | Why |
|---|---|---|---|---|
| D8-1 | The switch rule | (A) one cell = one pixel at the reference view; (B) one cell = a stated fraction of a pixel | **(A)** | It is the drawable floor the reach already uses; one rule, one constant, and M8-0 says the disagreement at it is a third of a pixel |
| D8-2 | The hysteresis factors | (A) 0.9 / 1.1; (B) measured by the pop detector and then fixed | **(B), starting from (A)** | A flap on the boundary is a seam (SL8); the detector decides the width |
| D8-3 | The residency lead | (A) the reach plus the interpolation buffer, no new data; (B) the ask R-18: a realm states a lead beside its reach | **(A) first, measured by M8-1; (B) only on a measured failure** | SL6: no new data crosses a boundary without a measured need |
| D8-4 | The mesh bytes | (A) keep f32/u32 (405 KB); (B) i16 positions + u16 indices (254 KB); (C) B plus packed normals (185 KB) | **(B) now, (C) when the vertex count rises with the spectrum** | The extractor already holds i16 vertices; B is a copy-time choice with no precision loss at 1/256 cell |
| D8-5 | The pop detector's legs | (A) a hull at 1.4, 240 and 528 m/s; (B) only the two fast legs, the walk waits for the character | **(A)** | The 1.4 m/s regime is the one the player sees most; the suit ruling lets a hull fly it |
| D8-6 | The globe beyond the band | (A) the rung above the coarsest drawn, from the crate; (B) keep the proxy outline | **(A)** | A realm draws itself (SL3) from its own recipe; the outline was a placeholder |
| D8-7 | The stamp's fields (8p) | the list in §2 | **the list** | Each field is measured, never typed; the owner asked for orienters |
| D8-8 | **The far-rung renderer** (owner, 2026-09-09: Enshrouded draws its far levels as CUBES, not triangles) | (A) every rung is a surface-nets MESH, as today; (B) the near rungs are meshes and the far rungs are the ladder's own VOXELS drawn as cubes or splats, one cell per pixel or more; (C) decided by a measurement in the look phase | **(C), and this slice keeps the seam OPEN for it:** the client library hands the engine a chunk's CELLS beside its mesh (it holds both today), the tier rule and the crossfade are written per column and not per triangle, and the dither works on either. Nothing in slice 8 chooses a triangle. | The far rungs ARE voxel grids already (rung L is the recipe at 2^L m cells, edits folded in); a cell drawn at one pixel is the same picture as a triangle, so the mesh's extraction (60 % of a chunk's cost) and its 254–405 KB are the price of a smoothness no far pixel shows. The choice is a LOOK decision (V13 addendum: after the freeze), and the foundation must not close it → **MEASURED 2026-09-11 (§19):** two-cell splats of the ladder's own vertices shade within ONE level of the mesh over the whole far ground and differ only on a one-pixel silhouette; zero cracks at two cells (3 183 at one). The look instrument is in the tree; the product form (vertex pulling, then per-rung merging, DEFERRED item 17) waits on the owner's look acceptance |

### 9.1 What the far view must keep open (owner, 2026-09-09)

The owner's target is a picture indistinguishable from reality, with everything players build visible
from very far. Three things in THIS slice decide whether that stays reachable, and each is kept open:

1. **The ladder is voxels, not triangles.** Rung `L` is the recipe's own cell grid at `2^L` metres with
   the edit pyramid's deltas folded in (ruling V4 row 7; S5-3). A tower a player built is a set of
   cells at rung 0 and a set of coarser cells at every rung above, so it is visible from as far as
   its reach says, drawn from the same rung as the ground around it. This slice's tier rule and
   residency band are stated per COLUMN of cells, so they serve a mesh renderer and a voxel
   renderer alike (D8-8).
2. **The mesh is one of two lowerings.** The client library already holds a chunk's cells (it runs
   the recipe) and turns them into a mesh. It keeps both. Which one the engine draws at a far rung
   is a look decision, measured after the freeze.
3. **The metre is the shape's floor, not the picture's.** On 1 m cells the finest clean feature of
   the height field is about 8 m (`04` §4.7); 1 m to 8 m relief comes from the fine floor and from
   PLACED ROCK with its own collider; below 1 m the detail is the surface's MATERIAL (textures,
   detail normals, displacement) and small objects — client work, after the freeze, on a seam the
   shape does not touch.

**Example.** A pilot builds a stone tower 60 m tall on the new home planet. At rung 0 it is 60
cells of stone; at rung 6 it is one cell. From 100 km her friend sees a one-pixel speck where the
tower stands, drawn from the rung-6 cells the pyramid holds — as a cube, if D8-8's measurement says
cubes, as a triangle if not — and the picture is the same either way at that distance.

## 10. Laws

- **SL8 (a seam is a defect).** The crossfade, the hysteresis, coarse-before-fine and the pop detector
  exist for this law. The detector runs on three speeds.
- **SL10 clause 7 (no derived velocity).** The band reads a stated lead or none; it never differences
  two rows.
- **SL3 (a realm draws itself).** The globe beyond the band is the realm's own recipe at a coarse rung.
- **The 2026-09-01 visibility ruling.** A dormant realm is never drawn; the client derives only for a
  realm that holds a row in the composed scene.
- **SL9.** Nothing per child; the wanted set is per column of the realm the eye is in.
- **HR3, HR4.** One rule for every realm kind; the fixture runs on a planet and on a hull with terrain.
- **HR5.** The client library stays Tier-A at 100 %.
- **SL6.** No new data crosses; R-18 is asked only on M8-1's failure.
- **`D-TERRAIN-3`** flips to green: the flag, the assertion and the one-rung code are deleted.

## 11. How it is built

1. 8p first: the stamp, the probe, the ruler, the light stated — on the existing one-rung pictures, so
   the instrument is proven before the ladder moves anything.
2. The tier rule and the wanted set per rung in the client library, with the one-rung assertion
   replaced by the multi-rung test; the coarse-before-fine order; the rung above the band.
3. The crossfade weight in the library and the dither in the material.
4. The residency band from the reach and the buffer; M8-1; the ask if it fails.
5. The mesh packing (D8-4) and M8-2.
6. The pop detector on three hull legs (M8-3); the refuter; the gates; the pictures; the owner's word.

## 12. The pictures

Every picture from this slice on carries the stamp, the probe and the ruler, with the star at 12°–18°
over the horizon and 100°–140° off the camera's nose, one light and one shadow (landed in step 0). The
set for this slice: the ground at 1.8 m with the ladder to the horizon; the same from a hill; the
approach sequence from 10 000 km to the ground, for the pop detector; and the globe from 2 000 km,
where the rung above the band is what the owner sees.

## 13. 8p — the picture instrument, BUILT (2026-09-09)

Step 1 of §11 is done on the one-rung pictures. Every judged picture now carries the stamp, the
probe, the ruler and one light at a stated angle, and the gate `terrain_pictures` asserts M8-4.

### 13.1 What was built

| Piece | Where | What it is |
|---|---|---|
| The stamp | `vd_devproto::DevTerrainStamp` on `DevState::terrain_stamp`; written by the renderer (`terrain.rs` §6, published by `place_chunks`); shown as four HUD lines on the picture | realm, rung and cell, the recipe's surface under the eye and the eye's height over it, the horizon and its dip, the radius drawn, nearest and farthest chunk, chunks drawn and pending, the star's elevation and azimuth off the nose, the biome, the world identity, the tick, the ruler |
| The probe | `probe.wgsl` + `ProbeMaterial` (a code instead of a colour), a second camera on render layer 1 (no tonemapping, no dither, no MSAA, black clear), a second `ImageCopier` slot, `shots/<label>.probe.png` beside the picture | per pixel: `R = kind << 5 \| rung`, `G,B` = distance from the eye in cells of the rung; the codec is Tier-A (`vd_client_harness::probe`) and the renderer's uniform is built by it |
| The twins | `terrain.rs`: every chunk gets a twin on layer 1 with the probe material; the ruler gets one too | what the probe camera sees; despawned with their chunk |
| The ruler | `vd_client::chunks::ruler_on_surface` (Tier-A): the eye's centre ray marched one cell at a time to the drawn rung's surface, refined by bisection; a ball of radius `hit distance × tan 2°` (never under half a cell), hovering one radius clear of the ground | a subject of known size in every picture, at the same angular size on every stand; its shadow is the second orienter |
| The light | the stamp's `star`, from `star_angles` (Tier-A) on the brightest row, the local up and the camera's nose | asserted 12°–18° up and 100°–140° off the nose |
| The gate | `crates/bins/tests/terrain_pictures.rs` | the probe replaces the paint classifier of slice 7; M8-4 below |

### 13.2 M8-4, MEASURED (three stands, the green flight of 2026-09-09)

| Stand | Altitude (stamp = state) | Horizon | Ruler: predicted / measured radius | Probe cells under the ruler / stamp |
|---|---|---|---|---|
| ground, rung 0 | 3.382 m | 6.57 km, dip 0.06° | 30.65 px / 30.55 px | 28..29 / 28.2 |
| hill, rung 3 | 301.54 m | 62.0 km, dip 0.56° | 30.91 px / 30.85 px | 140..145 / 139.7 |
| aloft, rung 9 | 60 001.5 m | 876.6 km, dip 7.83° | 30.83 px / 30.75 px | 461..479 / 461.4 |

Every terrain pixel of every probe states the flag's rung (a bit-exact reading of the probe's
channel through the sRGB target); the star reads 15.00° up and 120.00° off the nose on all three;
the terrain share in the lower band is 1.000 on all three; the ruler's centroid stands under 0.8 px
from its projection; the probe's far rim under the ball reads `√(d² − r²)/cell` within one cell.
THE PICTURE ITSELF is judged where the probe points (the refuter's finding 1): the ground's lit
paint under the probe's ground reads 1.000 / 1.000 / 0.999, the ball's red hue under its disc
0.981 / 0.985 / 0.988 — a black picture with a perfect probe fails.

The refutation and its answers: `verdicts/slice_08p_refutation.md` (19 findings; 15 fixed, 4
answered with a measurement or a bound). The row's delivered FACING now rides the state
(`DevRealmBox::facing`), so the gate rotates the eye into the planet's frame instead of assuming
the identity.

### 13.3 What the instrument found on its first flights

1. **The stamp read the wrong planet.** The window holds every planet of the system with a
   surface statement; "the body under the eye" was the FIRST by id — a sibling 29 000 km away —
   and the stamp read an altitude of 29 416 km. The work light had carried the same choice
   silently since M8-L. Now: the body whose surface is nearest the eye.
2. **A capture showed the frame before its request.** The readback runs one or two frames behind
   the world, and the serve took the LATEST readback. The gate waited for "no chunk pending", asked
   for the picture, and the hill picture showed a black gap on the ridge with three chunks
   pending. Now every readback carries the main-world frame it was extracted at, a job remembers
   the frame it arrived in, and it is served only by a readback stamped at or after it. Every
   capture gate in the tree inherits this.
3. **The horizon from the ladder's floor was 184 km for an eye 3.4 m up.** The floor radius stands
   2 650 m under this planet's recipe. The horizon is now the sphere through the surface under
   the eye.

### 13.4 What the pictures say now (the owner's orienters)

On the ground the eye is 3.4 m over the recipe and the ball 29 m ahead is 2 m across; the drawn
patch ends 0.37 km out, far inside the 6.6 km horizon — the black edge is the one rung's radius
(`D-TERRAIN-3`), which step 2 replaces with the rungs beyond. From the hill the ball is 82 m
across at 1.2 km. From 60 km up the ball is 17 km across at 245 km and the drawn patch reaches
548 km, past the 877 km horizon's dip.

## 14. Step 2 — the tier rule and the ladder to the horizon, BUILT (2026-09-09)

`D-TERRAIN-3` is 🟩: the one-rung flag, its radius, the lane's one-rung refusal and `chunks_around`
are deleted. The client draws THE LADDER for every body in its window, no flag.

### 14.1 What was built

| Piece | Where | What it is |
|---|---|---|
| The tier rule | `vd_client::ladder_view::rung_for_distance` | the finest rung whose cell is one pixel at the reference view, `cell(L) ≥ d · pixel`, the pixel being `vd_core::geometry::drawable_theta_min_rad` (the reach's own floor); switch distances 869 m, 1.7 km, 3.5 km, 7.0 km … |
| The descent | `LadderView::wanted` | a quadtree from the top rung: every top-rung column within the REACH (the horizon from the eye's height plus the horizon of the recipe's tallest ground, its amplitude sum, 17 km on this planet) is a node; a node is SPLIT into its four children when the rule at its centre asks for a finer rung, DRAWN otherwise. The children tile the parent exactly, so the rings have no hole and no double. MEASURED before the descent: each rung's columns chosen by their own centres left black rectangles at every ring boundary (4 818 hole pixels on the hill, 16 297 aloft); a tiling test (400 directions from under the eye to the horizon, exactly one wanted column each) found the last gap, at the horizon's edge — a column is inside the horizon by its NEAREST point, not its centre |
| Peaks past the horizon | `sightline_drop_m`, `surface_column` | past the geometric horizon a column is wanted only when its peak clears the sightline's drop `(d − d_h)²/2R` — from the ground a 14 m hill at 20 km, a 700 m mountain at 100 km. MEASURED before it: 6 556 chunks from the ground, most hidden by the planet's curve |
| The column's own bound | `vd_terrain::digest::column_bound_m`, `surface_column` | the surface is sampled on a 5 × 5 grid per column and the span is widened by the recipe's own bound (each octave's amplitude × `(π·s/λ)²/2`, capped), instead of a whole chunk each way. MEASURED before it: three chunks per column, two empty; after: 692 chunks over 659 columns at rung 0 |
| Coarse before fine | `WantedSet::keys` order, `overlapping_missing` | the wanted set lists the coarsest ring first; a chunk no longer wanted is released only when every wanted chunk over its footprint has arrived — a lookup on a per-face-and-rung index, never a scan (SL9) |
| The globe beyond the band | the outermost ring | from 2 000 km the nadir is at the top rung, so the whole visible cap is one ring of 1 189 chunks and the limb is in frame: D8-6 with no proxy outline |
| The gate | `terrain_pictures`, four stands (ground, hill, aloft, orbit) | every terrain pixel's rung is the rule's rung at the pixel's own distance within one rung; the finest rung on screen is the rule's rung at the eye's height; the reach passes the horizon; the centre column's topmost drawn pixel reaches the horizon's predicted row; NO BLOCK OF NOTHING under drawn ground (`ground_holes`: a hole that survives a one-pixel erosion is a missing chunk; a hairline crack is not) |

### 14.2 MEASURED (the tenth ladder flight of 2026-09-09, after the refutation)

| Stand | Rungs on screen | Chunks | Reach | Pixels one rung off the rule | Crack pixels / blocks |
|---|---|---|---|---|---|
| ground, 3.4 m | 0..9 | 4 107 (712, 519, 524, 606, 560, 495, 372, 210, 108, 1) | 465 km | 398 of 632 482 | 90 / 0 |
| hill, 301 m | 0..9 | 4 514 | 465 km | 25 303 of 744 688 | 506 / 0 |
| aloft, 60 km | 7..11 | 2 112 (480, 543, 469, 476, 144) | 1 335 km | 35 976 of 582 959 | 450 / 0 |
| orbit, 2 000 km | 12 | 1 381 | 5 889 km | 0 of 463 413 | 0 / 0 |

Before the refutation's fixes the orbit cap held 1 277 chunks: the 104 more are the columns the
one-face fold had dropped (finding 1), and the far rings of the ground and hill grew by the peaks
the coarse field had culled (finding 2). In every column of every picture the topmost drawn pixel
stands at or above the horizon of the lowest ground the recipe can raise (from orbit three rows
under the eye's own horizon — the limb is complete). The ruler stands at the rule's rung for its
distance: rung 0 on the ground, rung 1 from the hill (2 m cells), rung 9 from aloft, rung 12 from
orbit (a 124 km ball at 3 479 km).

The refutation of this step and its answers: `verdicts/slice_08_step2_refutation.md` (19
findings; the fold, the coarse-field peak, the one-metre march to 466 km, the never-withdrawn jobs,
the unbounded span cache, and the argued bound were real, and each is fixed and measured).

### 14.3 What is still owed in this slice, and what the pictures show

- The boundary between two rings is a HARD EDGE with a hairline CRACK where a fine chunk meets a
  coarse one along their shared edge (the two meshes do not share vertices): MEASURED 151 crack
  pixels on the ground, 524 on the hill, 450 aloft, none from orbit, none of them a block. That is
  step 3, the crossfade with the dither and the hysteresis pair (`HYSTERESIS_IN`/`OUT` are stated
  in the view and applied there): inside the band both rungs are drawn, so a crack in one is
  covered by the other, and the gate then asserts zero crack pixels.
- The rings are full circles around the eye (residency around the eye, not the view wedge), so a
  look-around never builds. The cost is the honest cost of D8-1 A: about 4 000 chunks from the
  ground, 1.6 GB of mesh at today's bytes — M8-2's census, and step 5's packing is the lever.
- Past two body radii the proxy outline still stands in for the globe; the handover is a seam the
  pop detector measures (step 6).
- The hill's ruler rim read 576 cells against 574.8 at 2 m cells: the probe's low byte is exact to
  within one step of the target's 8-bit rounding, and the gate's tolerance says so (two cells).

## 15. Step 3 — the crossfade, BUILT (2026-09-09)

`D-TERRAIN-5` item 1 is 🟩: the picture gate asserts ZERO pixels of nothing under drawn ground on
every stand — not a block, not a crack — and every stand is green.

### 15.1 What was built

| Piece | Where | What it is |
|---|---|---|
| The bands | `vd_client::ladder_view::fade_bands`, `HYSTERESIS_IN/OUT` (0.9, 1.1) | around every switch distance `s_L` a band `(0.9 s_L, 1.1 s_L)`; rung `L` is drawn from `0.9 s_{L−1}` to `1.1 s_L`, split while some point of it lies short of `1.1 s_{L−1}`. The band's two edges ARE the hysteresis: nothing flips at either |
| THE GEOMORPH | `ChunkGeometry::morph`, `ATTRIBUTE_MORPH`, `ladder_fade.wgsl` (vertex) | every vertex carries where its own radial meets the next coarser rung's MESH (`ParentMesh`: the parent chunk's whole surface, every crossed edge of its box, `extract_all_edges`, bucketed by lattice cell; the eight parents a chunk faces, shared through the lane's bounded `ParentCache`); across its fade-out band the vertex slides from its own position to that one, on its own distance from the eye. At `1.1 s_L` the finer vertices lie on the coarser triangles, and the finer rung ends there (the fragment stage discards it past the edge). A hit farther than the sink bound is another surface (a cave) and the field is the target instead |
| THE SINK | `chunks::sink_m`, `ChunkGeometry::sink`, `ATTRIBUTE_SINK` | nearer than its fade-in edge `1.1 s_{L−1}` — the finer rung's fade-out edge, the same line — a rung drops along each vertex's radial by the recipe's bound on the gap between the two fields plus a cell of each rung (the extractor's placement), from nothing at the line to the whole drop at `0.9 s_{L−1}`. Under that drop the coarser mesh is certainly below the finer one: it never shows through, and where the finer is still building the sunk coarser shows instead of a hole (coarse before fine, SL8). Nothing is discarded on the near side |
| The shadow pass | `ladder_fade_prepass.wgsl`, `MaterialExtension::prepass_vertex_shader` | the same morph and sink, so the shadows fall from the surface the picture shows; every distance is read from the render frame's origin (the floating origin, where the eye stands), because the shadow pass's view is the light's |
| THE SKYLINE | `vd_client::skyline` (Tier-A) | what the near ground hides, per azimuth, as a rigorous lower bound: every column inside the eye's horizon raises a WALL — its guaranteed floor over its own footprint quad on the eye's chart — on a fan of 3 600 rays; a column past the horizon is wanted only when its peak bound can show over the LOWEST wall on every ray its circumscribed disc touches. Replaces step 2's sightline drop against the eye's own sphere |
| The sampled low | `vd_terrain::digest::surface_column` | a column's lowest sample is now a comparison (`lesser`), not a subtraction from the largest float that absorbed every sample: it read 6.9 km, and nothing had read it before the skyline |
| One cell of margin below | `surface_column` | the extractor gives an edge to the chunk that owns its LOWER cell, so a crossing of a chunk's bottom boundary edge is drawn by the chunk below it; the span's low end steps one cell down |
| THE SKIRTS | `chunks::add_skirts`, `SKIRT_CELLS` (2) | a strip two cells deep under every boundary edge of a chunk's mesh, facing outward, with its edge's normals, morphing and sinking with its edge. Two neighbours state a shared vertex from one set of quanta, but the engine adds each chunk's own origin to its own single-precision offsets, and the sums differ by a rounding (a tenth of a millimetre at a kilometre) — a hairline crack a pixel's centre can fall into. MEASURED on the hill: one pixel of nothing at 970 m where four rung-1 chunks meet, the same pixel on two flights |
| The probe | `probe.wgsl` | the same morph, sink and far edge, so what the probe reads is what the picture shows; the ruler ball carries still attributes |
| The gate | `terrain_pictures` | zero hole pixels (not only zero blocks); the rung rule judged outside the bands (≥ 0.9 exact); the finest rung ON SCREEN (the probe's) is the rule's rung at the eye's height or, inside a band, the finer one still fading; the finest RESIDENT rung at most one under it |

### 15.2 What was measured on the way, and refused

| Attempt | MEASURED | Why it was refused |
|---|---|---|
| A complementary dither on each fragment's own distance, both rungs drawn in the band | 116 holes, all in bands | two surfaces that stand apart along a pixel's ray cannot share one weight |
| The same, the finer rung's weight read on the pixel's ray against a reference sphere | 788 then 895 holes on the horizon rows | ground with relief is not the sphere: a hill seen over a crest reads as the sphere's horizon |
| The coarser rung whole beneath, the finer dithering out | 47 holes at 13–15 km | a finer hill seen over a near crest dithers out onto a coarser surface under the sightline |
| The geomorph alone, the coarser whole beneath | 47 holes, the same pixels | not a crossfade defect at all: the ray tracer (`examples/ray_hole`) found the culled column |
| Learned spans: a built chunk whose surface leaves through a face names its neighbour | 11 705 chunks from the ground against 4 107, 47 holes still | three times the chunks (sealed caves under the surface chain the exits) and no hole removed; taken out again |
| The sightline drawn against the sphere lowered by the whole relief bound (16 km) | 8 775 chunks from the ground, every far ring a full annulus | rigorous, and culls nothing inside the reach |
| The skyline over the disc INSCRIBED in each column | nothing culled (8 779) | adjacent discs leave an open bin at every corner, and a far column always found one |
| A partition by distance on the coarser rung's fragments, once the finer footprint had arrived | 1 hole on the ground stand | two different meshes are not identical between vertices: on a grazing ray the finer read just past the line and the coarser just before it, and both discarded |
| The skyline's first flight | 34 dark specks on the hill at 950 m, 1.9 km, 3.8 km | not holes: shadows. The shadow pass ran the engine's unmorphed vertex stage (fixed by the prepass stage), and — the specks unchanged after that — the finer rung morphed onto the coarser FIELD while the coarser MESH lies within a cell of it, so the two surfaces crossed each other along every fade-out edge and the coarser's bumps shadowed the finer. Hence the parent MESH as the target |
| The parent mesh with the parent alone | halo vertices morphed 33 m down into a cave | a lower half's halo stands over the parent's own halo column, whose radial crossings no box can build; the lateral neighbours hold that column as their own last one |
| The refutation's fixes (the cap as angles, the corners' ray span, the lowered floors, the sink's end past the edge, the seam rule, the alpha-masked prepass, the grown bounds) | 1 hole on the hill at 970 m, the same pixel twice | not any of them: a hairline crack where four rung-1 chunks meet, the engine's own single-precision rounding through two origins, that a pixel's centre sampled. Hence the skirts |

The root cause of the 47 holes, found with the ray tracer: from the ground stand a ridge 15.5 km
out peeked over one 13 km out, and the rung-4 column of the valley floor between them (its peak
90 m UNDER the eye's surface, the sphere's tangent 3.9 m OVER it) was culled by step 2's sightline
test — one row of sky between two crests. The ground between the eye and that valley lies below the
eye's sphere, so the eye sees over it into a valley the sphere would hide: the sphere test was
unsound wherever the ground has relief.

### 15.3 MEASURED (the twenty-second flight of 2026-09-09, after the refutation and the skirts)

| Stand | Rungs on screen | Chunks resident | Reach | Pixels one rung off the rule (in the bands) | Morph fallbacks of vertices | Hole pixels / blocks |
|---|---|---|---|---|---|---|
| ground, 3.4 m | 0..5 | 5 918 (949, 818, 831, 917, 809, 661, 544, 181, 114, 75, 19) | 465 km | 2 155 of 632 628 | 7 451 of 45 529 074 | 0 / 0 |
| hill, 301 m | 0..9 | 7 056 (864, 819, 833, 950, 1 052, 904, 851, 382, 217, 154, 30) | 520 km | 77 363 of 749 175 | 8 197 of 49 386 184 | 0 / 0 |
| aloft, 60 km | 7..11 | 3 526 (rung 6: 166 … rung 11: 300, rung 12: 16) | 1 335 km | 98 553 of 583 371 | 2 145 of 15 436 881 | 0 / 0 |
| orbit, 2 000 km | 12 | 1 457 (29 at rung 11, 1 428 at rung 12) | 5 889 km | 0 of 463 413 | 14 of 6 651 714 | 0 / 0 |

The chunks grew against step 2 (4 107 / 4 514 / 2 112 / 1 381): the bands hold two rungs over a
fifth of every ring, and the skyline wants every column the eye can see into where the sphere
test culled valleys it could see; the refutation's lowered floors (a cell and the sink under the
field's bound) admit a few hundred more far columns (ground: 5 648 → 5 918). The far rings still
shrink under the skyline (rung 8: 114 on the ground, against 1 090 with no culling at all). In
every column of every picture the topmost drawn pixel reaches the horizon of the lowest ground;
the ruler stands at the rule's rung on every stand. No seam vertex on any stand (all four look at
mid-face ground). The dark specks at the fade-out edges fell from 34 to 2 pixels on the hill (of
749 175: shading at the residual crease where a finer triangle cuts across a coarser edge); the
five dark pixels on the orbit picture are the ruler ball's own shadow at that scale.

### 15.4 What is still owed in this slice

- At a band's far edge a finer triangle that spans a coarser crease is the chord under it; the sink
  ramp ends past the edge so one finer cell of sink remains there (`sink_end_m`). The pop detector
  (step 6) measures the edges on a moving eye; splitting the straddling triangles is the exact cure.
- A vertex on a cube-face edge reads the coarser field from both faces (one target, one row of
  field-vs-mesh crease along the twelve cube edges); the neighbouring face's parent mesh is the
  exact cure (D-TERRAIN-5 item 6).
- Every measurement is a STILL stand (D-TERRAIN-5 item 4): the morph on a moving eye, the arrival
  of a finer rung over a sunk coarser one, and the skyline's recompute per half metre are owed to
  M8-1 and step 6.
- The skyline's cost is not measured on a moving eye: a recompute raises about 36 rays per 62 m
  column at 1 km (about 2 000 near columns from the ground; MEASURED before the refutation's fix,
  every wall walked the whole fan of 3 600). M8-2 measures it with the bytes.
- A sealed cave under the surface is never wanted from above, and a cave that opens sideways is
  drawn only with its own chunk: the column span is a surface model. The block system's own
  residency (slice 9) owns caves; registered in `DEFERRED`.

## 16. Step 4 — the residency band, BUILT and MEASURED (2026-09-09)

D8-3 (A) stands on the walk and at 240 m/s, MEASURED: the reach plus the interpolation buffer, no
new data, and the band is complete on every sample. At 528 m/s the band fails on every sample —
and the failure is a THROUGHPUT wall, not a lead shortfall. The ask that goes to the owner is
therefore NOT R-18 (a stated lead): a lead of one chunk cannot bridge a gap of 1 800 chunks. §16.3
names the levers.

### 16.1 What was built

| Piece | Where | What it is |
|---|---|---|
| THE LEAD | `RenderSnapshot::lead_cursor`, `rendered_at`; `client-render::terrain::sync_terrain` | the wanted set is computed for the eye at the LEAD cursor: the render cursor plus the buffer's ticks — at or past the freshest tick the shards have delivered, one interpolation buffer ahead of the picture; the sampler FREEZES at the newest pose and never coasts, so the lead eye steps forward as rows arrive (the stamp's lead is a sawtooth, 14–30 m at 240 m/s). Every chunk is asked for one buffer before the picture needs it, from delivered poses alone (SL10 clause 7: the client never subtracts two rows and calls it a speed). The lead eye in the body's frame is the body's centre at the lead scene plus the own pose at the lead cursor — a pilot's own pose stands still inside a flying hull while the planet's row moves through the hull's frame. The lead carries the TRANSLATION only: the scene overlay keeps the boot facing at every cursor, so a turning hull's rotation is not in the lead (nor in the drawn scene; `DEFERRED` D-TERRAIN-5 item 11) |
| THE TERRITORY | `WantedSet` (`Margin`, `Urgent`, `Revealed`), `urgent_missing`, `urgent_missing_per_rung`, `revealed_missing` | a wanted chunk inside the horizon whose column lies in its rung's OWN territory (nearer than the rung's switch distance and farther than its fade-in edge) is URGENT: the picture draws it now, and one that is not resident is the band's gap. A wanted chunk in a band (a coarser column entering at its fade-in edge, still under the finer rung) is MARGIN: the finer rung covers it. A wanted chunk past the horizon (a peak the skyline admits) is REVEALED: an eye sees it over a crest before any lead can predict it — counted apart, the pop detector's (step 6) |
| THE SKYLINE'S HYSTERESIS | `LadderView::kept`, `culled`, `KEEP_MARGIN_RAD` (0.05), `WANT_MARGIN_RAD` (0.01), `HORIZON_HYSTERESIS` (0.25) | a column the last descent kept clears the skyline with a wider margin than a new one, and a column the skyline culled stays skyline-judged until it lies a quarter inside the horizon — the horizon moves with the eye's height (a walker over a bump moves it from 4.8 to 5.7 km), and MEASURED on the walk a column flipped between "hidden" and "wanted unconditionally" step by step and was built for nothing |
| The stamp | `DevTerrainStamp::{chunks_urgent, chunks_revealed, urgent_per_rung, lead_m}` | the band's gap, the reveals' gap, the gap per rung, and the lead the band ran on (the distance from the drawn eye to the lead eye in the frame of the body UNDER the eye — the widest over every body is a moon's: in the spinning planet's frame a moon moves kilometres per buffer) |
| THE GATE | `terrain_moving_eye` (`just terrain-moving-eye`) | the walk at 1.4 m/s (the gate finds the stick's share by FEEDBACK against the measured walk; the minute covers 84 m, a rung-0 chunk and a third; ASSERTED: zero frames with a gap), a hull berthed in the planet's realm 500 m over the HIGHEST ground along the whole flight (read from the recipe every 100 m; between two samples the surface may stand higher, and the clearance carries that), the character boards it (the crossing commits), pushes to 240 m/s and coasts, pushes to 528 m/s and coasts (both hull legs MEASURED and reported, never asserted — §16.2's five readings). The gate measures every speed from the planet's row moving through the hull's frame; it reads the band's gap every 40 ms from the stamp, with a time course every second, and the verdict is the renderer's own count of FRAMES with a gap across the leg (no frame escapes a poll). The gate refuses to start beside a process of an earlier run's fixture |

### 16.2 MEASURED (the eighth run of 2026-09-09)

| Leg | Samples | The band's gap (max urgent; samples with a gap) | Reveals (max; samples) | Queue peak | Fewest drawn | Eye over the ground |
|---|---|---|---|---|---|---|
| walk, 1.40 m/s | 1 287 | **0; 0** | 2; 9 | 7 | 6 652 | 4–6 m |
| hull, 240 m/s | 1 274 | **0; 0** | 3; 118 | 102 | 7 129 | 1 400 → 990 m |
| hull, 528 m/s commanded, 540 measured (the probe) | 1 276 | **1 822; 1 276** | 31; 1 276 | 2 798 | 5 896 | 858 → 1 713 m |

The probe's gap by rung, at 17.6 s: rung 0: 242, rung 1: 708, rung 2: 508, rung 3: 239, rung 4:
85, rung 5: 13. The queue stood at 1 173 when the probe began (the 240 m/s leg ended at 990 m over
rising ground, rung 0 entering) and never fell under 900. The ground stayed on screen throughout
(the coarser rungs cover what the finer ones miss): no hole, but a picture one to five rungs
coarser than the rule for a minute, which SL8 calls a seam.

The ninth run (the gate as it stands: the walk and 240 m/s asserted, 528 m/s reported) repeats it:
walk 0 gap on 1 379 samples, the lead 0.1–0.2 m; 240 m/s 0 gap on 1 270 samples, the queue at 144,
the lead 14–30 m (one buffer of the hull's motion); the probe 1 804 urgent on all 1 229 samples, the
queue at 2 780, the lead up to 64 m. The tenth run, after the refutation's fixes (pass B reads the
true horizon, the verdict counts FRAMES with a gap, the clearance reads the whole flight): walk 0
frames with a gap on 1 284 samples; 240 m/s 0 frames with a gap on 1 234 samples, the queue at 109,
the lead up to 29 m; the probe 1 796 urgent at the worst on all 1 227 samples and 1 105 frames with
a gap (the renderer drew about 18 frames a second under that queue), the queue at 2 762, the lead
up to 64 m. The eleventh run (the full gate chain's, after thirty minutes of builds and flights):
walk 0 frames with a gap on 1 291 samples; 240 m/s 7 urgent at the worst, 24 FRAMES with a gap on
17 of 1 278 samples, the queue at 213 — all in the leg's last two seconds, when the eye fell under
990 m over the rising ground and rung 0 entered (the queue went 37 → 94 → 147 → 194 in three
seconds; runs 8 to 10 ended at 967–990 m and stayed green); the probe 1 830 urgent on 2 144 frames
with a gap, the queue at 2 815. The twelfth run (the final gate, straight after the chain): walk 0
frames with a gap on 1 289 samples; 240 m/s 34 urgent at the worst, 59 FRAMES with a gap on 75 of
1 218 samples, the queue at 338. **So 240 m/s at one kilometre up sits ON the wall — three red
readings of six — and breaks for certain the moment the finest ring joins**; the gate asserts the
walk alone and REPORTS both hull legs (a leg that flaps at the machine's edge is a measurement, not
a gate). Runs one to seven, on the way:

| Run | MEASURED | What it taught |
|---|---|---|
| 1 | the walker's nose 8° down: under the surface after 13 m, horizon zero, 1 930 chunks pending | the walker faces LEVEL; with the eye under the surface everything within the reach is wanted (owed: a stated floor, `DEFERRED` D-TERRAIN-5 item 12) |
| 2 | the naive stick share walked at 0.52 m/s | the foot speed by feedback against the measured walk |
| 3–5 | 4, then 2 urgent chunks on the walk | the skyline's keep/want margins, the horizon's hysteresis, and the territory rule (a coarser column entering at its fade-in edge is under the finer rung: margin, not urgent) → 0 |
| 6 | the hull at 300 m / 244 m/s: the queue ~2 400 steady, rung 0 at 200 of 950 resident; then a collapse to a dozen chunks 12 km on | a skim at 300 m is past the throughput wall; and the hull flew INTO rising ground (847 m along the path): the altitude is read from the recipe |
| 7 | the hull at 1 347 m / 240 m/s: 0 urgent for 44 s, then a queue of 333 and 68 urgent | the same leg is green in run 8 (queue 102); a shard of run 6's kept fixture was found running beside run 7 (UNMEASURED whether it was the cause). 240 m/s sits near this machine's edge |

### 16.3 What it means, and the ask

**The lead is applied and is not the lever.** One buffer (120 ms) at 528 m/s is 63 m — one rung-0
chunk (62 m), half a rung-1 chunk; the ninth run read the lead at up to 64 m. The gap is 1 800
chunks across six rungs and the queue 2 800 deep: the eye sweeps more chunks per second than the
workers build. An estimate from the geometry, NOT a measurement: at 1 000 m over the ground rung
1's territory is a disc about 2.8 km across (a radius of 1.4 km, from its switch distance of
1 738 m), rung 2's 5.7 km, rung 3's 11.5 km; at 528 m/s the eye uncovers a strip of each per
second, about 100 + 50 + 25 chunk columns a second at rungs 1 to 3 alone (the strip's width times
the speed, over the chunk's footprint), two to three chunks a column. The build rate is UNMEASURED
as a number; M8-2's census (step 5) owes it. The probe began with 1 173 chunks already queued (the
240 m/s leg ended over rising ground with rung 0 entering; a coasting hull cannot settle between
two legs), so its first seconds carry that debt.

**Where the wall stands, MEASURED:** a walk holds everywhere; 240 m/s at one kilometre over the
ground is ON the wall (three red readings of six: the queue at 213–338 against 102–144 on the green
ones), breaks for certain the moment the finest ring enters (run 11's last two seconds), and never
held on a skim at 300 m (run 6); 528 m/s breaks throughout. Example: a hull
crossing a plain at 240 m/s and 1 200 m up draws whole ground; the same hull over a ridge that rises
to 300 m under it sees the finest ring arrive late for a few frames, and the coarser ring stands in.

**The levers are the owner's:**

1. **Throughput.** Step 5 shrinks the bytes per chunk (D8-4) and M8-2 measures the build rate and
   the parent cache; the worker count is a client setting. Then the probe is flown again. This is
   the recommended first move: a measurement, no new rule.
2. **A rule under a deep queue** — the finest rungs are not asked for while the queue is deeper than
   the workers can drain in one buffer. It keeps the ground whole at the cost of a coarser picture
   at speed, which is the seam the probe already shows, only chosen. It needs no new data; it needs
   the owner's word, because SL8 calls it a seam.
3. **A stated lead (R-18).** Refused by this measurement: a lead is a distance, and no distance
   builds chunks faster.

Nothing new crosses a realm boundary in step 4 (SL6). Both hull legs stay in the gate as
measurements, reported on every run, so the wall is seen the day it moves; the walk is the gate's
assertion, and the owner names which hull legs it asserts once the lever is chosen.

### 16.4 The rig, for the next flights

- The walker faces level; a nose tilted down walks under the ground, and under the ground the
  horizon is zero and everything within the reach is wanted.
- The foot speed and the hull's speed are commanded by feedback and measured from rows, never
  computed from a constant the shard does not share.
- The hull's altitude is read from the recipe along the whole path, plus a clearance; nothing in
  the gate states a height.
- A failed run KEEPS its fixture, and its shards keep running: stop them before the next run (one
  job at a time). Run 7 flew beside one.
- The stamp's lead is the body's under the eye; the moon's lead is kilometres per buffer in the
  spinning planet's frame, and it is the moon's ladder that recomputes every frame for it — the
  same as before the lead (the moon moved that much through the eye's frame already). The stamp's
  gap sums every body's ladder; a body past two of its radii wants nothing, so today the sum is
  the planet's.
- The release reads the LEAD eye's wanted set, so a chunk leaves one buffer before the drawn eye
  would drop it: at its fade-out edge a rung-0 chunk goes at 64 % of its morph at 528 m/s (63 m
  of a 174 m band), a difference under one pixel by the tier rule (a cell at the switch distance
  is one pixel; the residual is a third of a cell). Step 6's pop detector measures it; the exact
  form keeps the drawn eye's set for the release (`DEFERRED` D-TERRAIN-5 item 11).
- The refutation of step 4 (`verdicts/slice_08_step4_refutation.md`, 30 findings): the horizon's
  hysteresis sent ground inside the horizon to the "revealed" class and the gap under-read it
  (fixed: pass B reads the true horizon); the hull's clearance was read over the probe leg alone
  (fixed: the whole flight); the gate's verdict is now the renderer's count of frames with a gap.

### 16.5 M8-2a — THE THREE RATES, MEASURED (2026-09-10, the thirteenth run and `chunk_phases`)

Ruling V15 asked for the three rates before anyone touched the pool. The stamp now counts frames,
chunks built (and the nanoseconds the workers spent), chunks harvested and harvests that filled
their cap; the gate reads each across a leg.

| Leg | Frames/s | Built/s | ms per chunk | Harvested/s | Harvest at its cap |
|---|---|---|---|---|---|
| walk, 1.4 m/s | 26.1 | 2 | 52 | 2 | 0 of 1 563 frames |
| hull, 240 m/s | 28.6 | 151 | 64 | 151 | 0 of 1 715 |
| hull, 528 m/s | 20.7 | 233 | 60 | 219 | 88 of 1 242 |

**The harvest hypothesis is REFUTED.** The engine harvests everything the workers build; the cap of
24 a frame fills on no frame at 240 m/s and on 7 % of frames at 528 m/s. **The workers bind:** 14
threads at 60 ms of WALL time a chunk (the stamp's clock runs around the whole build, a wait on a
sibling's parent included) are about 230 chunks a second, and the probe's demand is more.

**Where the 60 ms goes** (`chunk_phases`, release, the home planet, face +X chunk (3, 5), five
rounds each):

| Rung | Box + extract | Vertices | Parents | One parent mesh | Whole build, parents WARM | Whole build, parents COLD | The morph's own cost |
|---|---|---|---|---|---|---|---|
| 0 | 2.76 ms | 4 256 | 8 | 4.34 ms | 5.33 ms | 61.1 ms | 2.57 ms |
| 1 | 2.75 ms | 4 279 | 8 | 4.33 ms | 5.61 ms | 79.2 ms | 2.86 ms |
| 2 | 2.68 ms | 4 207 | 8 | 3.82 ms | 4.52 ms | 50.0 ms | 1.84 ms |
| 3 | 8.34 ms | 4 041 | 8 | 3.43 ms | 10.39 ms | 39.7 ms | 2.05 ms |
| 4–8 | 2.3–2.6 ms | ~4 200 | 8 | 3.8–4.1 ms | 4.0–4.4 ms | 26–28 ms | 1.7–1.8 ms |
| 9–11 | 2.1–2.2 ms | ~4 100 | 4 | 3.6–3.7 ms | 3.8–4.0 ms | 18 ms | 1.7–1.8 ms |

A chunk with its parents in the cache costs 4 to 5.6 ms (the generator's 2.7 ms plus the geomorph's
2.6 ms: the ray against the parent's triangles per vertex). A chunk whose eight parents all miss
costs 28 to 79 ms. The flight measured 60 to 64 ms: on a moving eye the parent cache (48 entries,
shared by 14 workers, each job needing 8) misses almost every time — the jobs of one ring arrive in
the order the descent emitted them, spread around the eye, and no two neighbours build back to back.
Rung 3's 8 ms box is the rung where the recipe's octaves cross a boundary (the V10 budget is 8 ms).

**The lever, named by the measurement.** THE PARENT CACHE'S HIT RATE, not the harvest and not the
thread count: with every parent warm the same 14 workers build about 2 800 chunks a second, twelve
times today's 233. Three parts, in the V15 order's second item (the priority queue), now shaped by
this:

1. **Order the jobs by parent**: within a class and a rung, the four children of one parent and
   their lateral neighbours build back to back, parents nearest the lead eye first — so the working
   set of parents at any moment is the workers' current jobs' parents, not the whole ring's.
2. **Size the cache to that working set**: the workers' count times the parents a chunk reads,
   with a margin, in the one config struct — 14 × 8 × 2 = 224 entries, about 110 MB at today's
   parent mesh (the packing of step 5 halves it).
3. **Build each parent once**: with siblings adjacent, four workers miss the same parent at the
   same moment; a build in flight is waited for, never repeated.

Also MEASURED, for M8-2's census: the renderer draws 26 frames a second on a STILL walk with 6 700
chunks on screen — the frame cost is the draw count, one mesh per chunk, and step 5's packing and
batching own it.

### 16.6 THE PARENT CACHE'S HIT RATE — the lever, built and MEASURED (2026-09-10, runs 14 to 17)

What ruling V15's second item became after §16.5 named the cache:

| Piece | Where | What it is |
|---|---|---|
| The request order | `ladder_view::RequestOrder`, `morton` | the wanted set's keys sort by class (urgent, revealed, margin), then the coarser rung first, then the PARENT column along a Morton curve over its face, then the chunk; the renderer requests each key at its index as the job's priority |
| The priority pool | `client-render::terrain::ThreadedWorkers` | one pool, one queue ordered by (priority, arrival), a condition variable; a worker takes the first; a withdrawn job is skipped when taken (as before) |
| The cache's capacity | `ParentCache::with_capacity`, `set_capacity`; `PARENTS_PER_CHUNK` (8) × workers × `PARENT_WORKING_SETS` | sized to the workers, not a constant of 48 |
| Single flight | `ParentCache::claim`, `finish`, `Claim` | a parent one reader is building is WAITED for by the next, never built twice; a wait counts on the stamp |
| The counters | `ParentStats` (hits, builds, waits), `ParentMesh::bytes`; the stamp's `parent_hits/builds/waits` | the gate prints the hit rate per leg; `chunk_phases` prints a parent mesh's bytes (about 500 KB) |

MEASURED, the 240 m/s leg and the 528 m/s probe on the M8-1 flight (14 workers):

| Run | Order | Cache entries | 240 m/s: ms/chunk, hit, frames/s, queue peak | 528 m/s: ms/chunk, hit, built/s, harvested/s, queue peak, worst gap |
|---|---|---|---|---|
| 13 (§16.5) | arrival | 48 | 64 ms, —, 29, 42 | 60 ms, —, 233, 219, 2 776, 1 804 |
| 14 | class, rung, parent nearest-first | 224 | 36 ms, —, 36, 31 | 46 ms, —, 273, 272, 1 383, 959 |
| 15 | class, rung, parent Morton | 224 | 35 ms, 53 %, 40, 30 | 42 ms, 52 %, 294, 294, 1 202, 819 |
| 16 | the same | 896 (≈450 MB) | 16 ms, 89 %, 43, 23 | 21 ms, 89 %, 418, 345, 2 015, 1 234 — and from 27 s on, over one kilometre up, ZERO urgent at 528 m/s (the queue 14–26) |
| 17 | the same | 448 (≈225 MB) | 22 ms, 76 %, 40, 40 | 29 ms, 74 %, 348, 347, 752, 461 — zero urgent over one kilometre |
| 18 | the same | 448, the upload timed | **22 ms, 76 %, 40, 39** | **28 ms, 75 %, 340, 339, 832, 524** — zero urgent over one kilometre; 0.07 ms of the main thread per upload |

The "ms a chunk" column is the worker's WALL time per job, a wait on a sibling's parent build
included (the stamp's clock runs around the whole build; the lib reads no clock, so the wait is not
split out). **THE SHIPPED SETTING is a memory budget of 256 MB** (`TerrainConfig::parent_cache_bytes`,
about 512 entries at an estimated 512 KB a mesh — between runs 17 and 16), never fewer than one
working set of the workers.

The order alone bought a fifth; the capacity bought the rest. The cache turns over (ESTIMATED from
the stamp's parent builds and the ring geometry, not printed as such): the pool builds about 600
parents a second at 224 entries, so an entry lives 0.4 s, and the next column of the same ring
arrives about half a second later. The working set is every ring's LEADING EDGE, about 300 to 500
parents on this flight by that estimate, not one worker's neighbourhood.

**What the probe still shows.** Over one kilometre the band holds at 528 m/s at every setting from
run 16 on. On the low pass (730 to 860 m up, inside the finest ring's territory) the gap peaks at
461–1 234 across rungs 0 to 4. At 896 entries the workers outran the harvest (418 built against 345
harvested a second, the cap of 24 a frame full on 744 of 1 683 frames); at the shipped size the two
are balanced (340 built, 339 harvested, the cap full on 325 of 1 632 frames), and the harvest loop
itself costs the main thread 0.07 ms an upload — the frame's cost sits in the engine's own GPU
upload of the changed assets and in the draw count, outside that loop. The byte budget for the
harvest (V15 item 2) therefore belongs with step 5's packing, where the bytes per upload are the
unit, and is NOT built here.

**The price and the ask.** A parent mesh is about 500 KB by `ParentMesh::bytes` (an ESTIMATE over
the vectors' capacities and a map node per bucket: positions as `f64`, triangles as `u32`, the
buckets); 896 entries is 450 MB, a measurement setting. The shrink — positions as `f32` from the
parent's origin, `u16` triangle indices (fewer than 65 536 vertices), the buckets as one flat table
— cuts it to about 190 KB, so 896 entries is 170 MB and 448 is 85 MB (D-TERRAIN-5 item 10).

**After the refutation** (`verdicts/slice_08_throughput_refutation.md`, 31 findings): a waiting
job now moves to each frame's priority (T-1), the priority is a global order — class, depth, index —
the same across realms (T-2), a cancel removes the job from the queue (T-3, T-4), a claim is a guard
that a panic drops (T-5), a forgotten realm refuses a late landing (T-6), a hit refreshes a use
counter and never scans (T-7), and the cache's size is a memory budget in the one config struct
with the harvest cap beside it (T-8, T-9). **Run 19, the shipped form (the 256 MB budget, 512
entries, the re-keyed queue), MEASURED:** the walk 0 frames with a gap; 240 m/s 0 frames with a gap,
80 % hits, 20 ms of wall time a job, 41 frames a second, the queue at 35; the probe 388 jobs and 374
harvests a second at 27.5 frames a second, 81 % hits, 24 ms a job, the gap at the low pass 770 and
the queue 1 414, zero urgent over one kilometre. The probe's low pass varies by run (461 to 1 234 at
the worst sample over runs 16 to 19): it is the finest ring's territory at 528 m/s, and step 5's
packing is its lever.

## 17. Step 5 — the packing, ruled V16: bytes change, the picture does not

### 17.1 M8-2's baseline census, MEASURED (2026-09-10, before any packing)

The stamp now carries the drawn chunks' mesh bytes as the engine uploads them; the picture gate
prints the census per stand and compares every new picture with the one on disk, pixel by pixel.

| Stand | Chunks | Vertices | On the GPU | Per chunk | Fill | Frames/s, still |
|---|---|---|---|---|---|---|
| ground, 3.4 m | 6 659 | 48.3 M | 3 374 MB | 495 KB | 11.9 s | 26.0 |
| hill, 301 m | 7 360 | 50.7 M | 3 541 MB | 470 KB | 10.9 s | 23.5 |
| aloft, 60 km | 3 529 | 15.5 M | 1 082 MB | 299 KB | 2.7 s | 51.9 |
| orbit, 2 000 km | 1 457 | — | — | — | — | — |

A vertex costs 48 bytes today (position, normal, morph target and sink, four vectors of three
`f32`), and a triangle's three indices 12. The frame rate falls with the vertices and the bytes on
screen: 48 million vertices at 26 frames a second on the ground, 15 million at 52 aloft. The
packing is the first lever tried; what it bought is in §17.3 (an eighth of the frame for 44 % of
the bytes), so the vertex count and the draw count hold the rest.

**The noise floor (flight pair: the free-tick flight against the flight before it; the rule:
every pixel).** Two flights of ONE code, each capturing at its own universe tick, differed by
277 to 593 pixels of 924 480 per stand, the widest channel step 59 to 140. The star moves between
the ticks and the silhouettes shift by a pixel. The gate now captures every stand at the first
multiple of 1 200 ticks after the settle, so two flights capture the same moment of the world;
the remaining difference between two flights of one code is the noise floor a packing is measured
against. MEASURED, two flights of one code at the fixed ticks (flight pair A → B, unpacked; the
rule: every pixel, no mask):

| Stand | Pixels that differ | Widest channel step |
|---|---|---|
| ground | **0** of 924 480 | 0 |
| hill | 76 | 59 |
| aloft | 162 | 137 |
| orbit | 71 | 137 |

The ground stand is exact. The other three keep a residual of tens of pixels at the widest step
(a silhouette's edge, the ruler ball or a star sprite, UNMEASURED which): the capture fires on the
first frame at or past the tick, and that frame's cursor stands anywhere inside the tick. The
gate's flag refuses one differing pixel; a packing is judged per stand against this floor.

### 17.2 The exact packing, BUILT

| Piece | Where | What it is |
|---|---|---|
| THE MORPH METRE | `ChunkGeometry::morph_m`, `morph_target`, `morph_targets`, `sink_of`, `bounds` | the target lies on the vertex's radial by construction (a radial hit on the parent mesh, or the coarser field in the same direction), so it is ONE signed metre along the radial from the vertex; the target is `vertex + radial × morph_m`, exact to float rounding. A skirt vertex carries its top's metre (both drop by the same along the radial) |
| THE SINK AS A UNIFORM | `ChunkGeometry::sink_m`; `LadderFade::centre_sink`, `ProbeParams::centre_sink` | the rung's sink is one number per chunk, the material's uniform; no attribute |
| THE RADIAL PER VERTEX | `ChunkGeometry::radials`, `RADIAL_STEP_RAD`; `ATTRIBUTE_RADIAL` (`Float32x3`, location 9) | a vertex's radial rides the vertex as three `f32` (12 bytes) in the realm's frame, turned into the world like a normal; it stands within 2e-7 rad of the exact one, `f32`'s own rounding (a unit test holds the bound over every vertex of a chunk). MEASURED as four signed 16-bit quanta first (8 bytes, 3e-5 rad): on the hill picture 18 content pixels of 701 472 differed from the float form, 13 by one level and 5 by 3 to 19 levels — isolated pixels in the far ground whose centre fell on the neighbouring triangle at a crease. A change of the picture; V16 refuses it, and the 16-bit form stands as the owner's option with its numbers (4 bytes a vertex, about 150 MB on the ground stand). The first form read the radial off a BODY CENTRE in the material's uniform; MEASURED, that rewrote every material as the eye moved and the engine re-prepared them: 26 → 19 frames a second on the walk, 41 → 18 at 240 m/s, with a millimetre's tolerance, and no better at 240 m/s with a millionth of the distance (the rewrite rate grows with the speed and each rewrite is a hitch). The uniform keeps the rung's sink alone and is never rewritten |
| THE SHADERS | `ladder_fade.wgsl`, `ladder_fade_prepass.wgsl`, `probe.wgsl` | the morph in world space: `mix(own + radial × morph_m, own, whole(d)) − radial × sink × (1 − risen(d))`; the motion vector carries this frame's displacement onto last frame's own position |
| 16-BIT INDICES | `packed_indices` | exact where a chunk has at most 65 535 vertices (every chunk today), 32-bit otherwise |
| The bounds | `ChunkGeometry::bounds` | computed once in the library from the vertices, the targets and the sunk positions |

A vertex is 40 bytes (position 12, normal 12, the morph metre 4, the radial 12) against 48; a
triangle's indices 6 against 12. Normals and positions stay full width in this half (V16: a packed form that changes a
pixel is refused; the pixel measurement comes first).

### 17.3 The exact packing, MEASURED (2026-09-10)

| Stand | On the GPU, before → after | Per chunk | Frames/s, still, before → after |
|---|---|---|---|
| ground | 3 374 → 1 880 MB | 495 → 276 KB | 26.0 → 29.4 |
| hill | 3 541 → 1 974 MB | 470 → 262 KB | 23.5 → 26.9 |
| aloft | 1 082 → 603 MB | 299 → 167 KB | 51.9 → 53.4 |
| orbit | — → 260 MB | 174 KB | — → 52.9 |

44 % fewer bytes on the GPU, and the still frame rate up an eighth on the near stands. Two things
on the way: (1) the first packed flight wrote every material's centre uniform EVERY frame and the
engine re-prepared them — 26 → 18 frames a second on the still ground stand; the write is now made
only when the centre moved, and the frame rate came back and rose. (2) `target` is a reserved word
in WGSL; the shader failed to compile and the first packed flight drew no ground at all (a red gate,
"terrain share 0.000").

**The pixels.** The first packed flight, compared with flight B's unpacked pictures at the same
ticks, differed by 70 / 54 / 181 / 62 pixels (ground / hill / aloft / orbit), the widest channel
step 137 — the size of the noise floor. A second packed flight, compared with the first, differed
on the ground stand by 152 pixels, and the difference image put EVERY one of them inside an 18 × 11
patch at the top middle: the stats overlay's counter, whose digits differ between runs. Not one
terrain pixel differed. The gate now compares only the CONTENT — the pixels either probe marks as
terrain or ruler — so the overlay never counts, and it keeps the previous picture and writes a
difference image beside the run whenever content differs.

**What is NOT measured, stated plainly.** The gate overwrote flight B's unpacked pictures before
the mask existed, so the masked comparison of packed against UNPACKED is lost; a masked comparison
of packed against packed (the packed noise floor) is what the next flight measures. What stands as
the exactness proof of the geometry is a unit test in the library: over every vertex of two chunks
the target the vector form carried and the target the metre form reconstructs differ by the
narrowing alone, under a tenth of a millimetre (the gate's bound; the measured widest difference
is printed by the test). A tenth of a millimetre at the switch distance is a thousandth of a pixel.

**The packed noise floor (flight pair: two packed flights at the same ticks; the rule: the probe's
content pixels, the overlay still counted):**
ground 5 of 635 560 (the widest step 1), hill 158 of 752 164 (step 59), aloft 0 of 586 341, orbit 6
of 466 445 (step 1). The hill's 158: its difference image put 157 of them inside the HUD's own text
— the stamp line's "tick 2399" against "tick 2400", the captured frame's tick readout, which
stands OVER the hill's far ground on that stand and so passed the probe mask — and about ten
isolated pixels at one channel step. The renderer now reports the HUD's rectangle on the stamp and
the compare leaves it out. With the overlay excluded (the next flight, packed against packed):
ground 7 of 635 560, hill 0 of 703 652, aloft 2 of 586 341, orbit 5 of 466 445 — every residual at
one channel step, a rounding flicker of a few pixels between two runs, never a change of the ground.


**On a MOVING eye (the M8-1 flight, frames a second):**

| Leg | Before the packing | The centre as a uniform, a millimetre's tolerance | The same, a millionth's tolerance | The radial on the vertex |
|---|---|---|---|---|
| walk, 1.4 m/s | 25.6 | 19.3 | 29.7 | 29.9 |
| hull, 240 m/s | 41.0 | 18.3 | 18.6 | 45.1 |
| hull, 528 m/s | 27.5 | 15.7 | — | 27.7 (the gap at the low pass 704, the queue 1 340) |

A material rewritten is a material the engine re-prepares, and a centre in the uniform is
rewritten as often as the eye moves past the tolerance — every frame at 240 m/s under any
tolerance small enough to keep the radial exact. The radial on the vertex (8 bytes) needs no
centre: no rewrite, no hitch, and the frame rate rises above the pre-packing one on every leg.

### 17.4 The refutation of step 5's first half, answered (2026-09-10)

`verdicts/slice_08_step5a_refutation.md`, 17 findings. SUPERSEDED by the radial on the vertex
(§17.2): P-1 (a material born without the centre), P-2 (the rewrite's cost on a moving eye —
MEASURED, then removed with the centre itself), P-13 (the tolerance) and P-14 (the ruler's
rewrite) — no material carries a centre any more. FIXED: the compare masks the UNION of both pictures' overlay rectangles, grown by the glyphs'
antialiasing, the reference's rectangle kept in a sidecar beside it (P-3); every stand captures at
a CONSTANT tick and a late settle is a red gate (P-4); the two `&&` asserts are split and the bounds
test covers a chunk that sinks (P-6, P-7); the worker builds the culling box and carries it on the
geometry (P-8); `centre_is` compares within a millimetre (P-13); the ruler's probe material is not
rewritten (P-14); the shaders state why the radial never normalises a zero (P-15); the census says
what its two counts count under the flat switch (P-17); the three stale comments and the three doc
numbers are rewritten (P-9, P-11, P-12); the noise-floor tables name their flight pair and rule
(P-10). STANDS: the moving-eye cost of a material rewritten every frame is MEASURED on this chain's
moving-eye flight (P-2; the fallback is an octahedral radial per vertex); the packed-against-
unpacked measurement V16 asks for is LOST, stated to the owner (P-5) — from here the gate keeps
every compared picture under `target/terrain_pictures`; the main-world copy of every mesh
(`RenderAssetUsages::default()`, P-16) is owed a resident-memory measurement with step 5's second
half.

### 17.5 The first half, FINAL (2026-09-10, the gate chain on the float radial)

Coverage 100 %, lint, combos, the pin, the pictures (zero holes on four stands) and the moving-eye
flight green. The census with the radial as three `f32` (a vertex of 40 bytes, the indices at 16
bits):

| Stand | Chunks | On the GPU (before → now) | Per chunk | Frames/s, still (before → now) |
|---|---|---|---|---|
| ground | 6 659 | 3 374 → 2 460 MB | 495 → 361 KB | 26.0 → 28.0 |
| hill | 7 360 | 3 541 → 2 582 MB | 470 → 343 KB | 23.5 → 26.4 |
| aloft | 3 529 | 1 082 → 788 MB | 299 → 218 KB | 51.9 → 52.4 |
| orbit | 1 457 | — → 339 MB | 227 KB | — → 52.4 |

The pictures against the 16-bit radial's: the ground and orbit stands exact, aloft within its
one-step floor, the hill on the same 18 pixels — the 16-bit form's own footprint flipping back. On
the moving eye: the walk 26.9, 240 m/s 43.8, 528 m/s 24.4 frames a second (25.6, 41.0 and 27.5
before the packing), the walk and the 240 m/s leg with zero frames with a gap; the probe's low
pass at a gap of 752 and a queue of 1 723 (the same wall, §16.6).

### 17.6 The parent mesh's shrink and the client's memory, MEASURED (2026-09-10, ruling V17 item 1)

**What was built.** A parent mesh is the coarser chunk's surface that a finer chunk's morph targets
are shot against (§16.6). It held its positions as `f64`, its triangles as `u32` and its buckets as
a map of vectors: about 500 KB a mesh. Now it holds its positions as `f32` offsets from one `f64`
origin (the parent's own centre): the offset reaches half the box, 31 cells, so the `f32` step is
about 62 × 2⁻²³ of the cell — 15 µm at rung 1 (cell 2 m), 7.8 mm at the home planet's coarsest
rung 11 (cell 2 048 m, offset 65 km), and 12.5 cm at the
largest rung the ladder can name (rung 15, cell 32 768 m). Against the cell that step is 7.4e-6
at every rung, which is what the morph reads it at. Its triangles are at 16 bits when the parent
has fewer than 65 536 vertices and at 32 bits otherwise, and its buckets one flat table: a start
index per cell of the 64 × 64 grid and one triangle list. The ray test that reads a parent widens
its slack to that step (`RAY_SLACK` at 1e-5, against 7.4e-6 per component): at the old 1e-9 a ray
through a narrowed vertex missed the triangle it stood on — the unit test that reads a real
parent's radials over its cells found that (`a_parent_mesh_answers_the_radials_over_its_cells…`).
An accepted hit stays within 1e-5 of an edge of its own triangle (2 cm at rung 11, 20 µm at rung
1), and the candidate set is unchanged, so the slack can neither pull in another cell's triangle
nor return a wild radius; a miss falls back to the field and the census counts it (unchanged,
below).

**Example.** The finest chunk under the eye on the hill stand shoots its 5 000 morph rays at the
rung-1 parent above it. Before the shrink the cache held 512 of those parents in 256 MB; now the
same budget holds about a thousand — the 89 % hit-rate setting of §16.6 — and the cache bounds
itself by the meshes' OWN bytes (refutation finding 2: a count of entries at an estimated size
was not a bound; a parent larger than the estimate overshot the budget silently), never fewer
than the workers' working set.

**MEASURED, `chunk_phases`:** 500 → 239 KB a parent mesh; a cold chunk build 61 → 42 ms (the
parent's build is most of a cold chunk, and it writes half the bytes).

**MEASURED, the picture flight (the four stands, Docker on, the cache at 512 entries — the
byte bound came with the refutation and flies in the final chain):**
the pictures within the one-step floor on every stand (ground 11, hill 1, aloft 6, orbit 5 content
pixels at a channel step of 1 — the gate's own noise), the morph fallbacks unchanged (ground 7 985
of 48 M vertices).

**The client's resident memory, and why the first reading is not trusted.** The census prints the
operating system's resident count (`ps`, RSS). With the shrink the ground stand read 3 122 MB
against 3 646 MB before it, but the hill stand read 4 097 MB against 1 904 MB. A resident count
falls when the machine is short of memory: the machine held 7.5 GB in swap during these flights
(Docker on), so a still-growing client can read SMALLER than it is, and the hill's 1 904 MB was
such a reading. The census now prints the physical footprint too (`footprint`, what the process
holds in RAM PLUS what the system compressed or swapped on its behalf), which does not fall under
pressure; the numbers that count are the footprint ones, below. The readers live in
`vd_bins::memory` (resident, footprint by category, the heap summary, the region table) so the
picture gate prints them and the moving-eye gate measures their growth per leg.

**THE MAIN-WORLD COPY, DELETED (ruling V17 item 1).** The engine keeps a mesh in the main world
too by default, and nothing on the client reads a chunk's mesh back (the culling box is the
library's own, the geometry stays on the lane), so a chunk's mesh is now render-world only. The
same four stands, both in the footprint's unit, Docker on:

| Stand | Drawn | Footprint before → after | Small malloc blocks before → after | Resident before → after |
|---|---|---|---|---|
| ground | 2 460 MB | 8 591 → 6 611 MB | 3 892 → 1 967 MB | 3 418 → 2 052 MB |
| hill | 2 582 MB | 8 247 → 6 463 MB | 3 476 → 1 698 MB | 2 270 → 2 025 MB |
| aloft | 788 MB | 2 963 → 2 493 MB | 1 243 → 784 MB | 1 765 → 1 123 MB |
| orbit | 339 MB | 1 674 → 1 610 MB | 524 → 491 MB | 1 112 → 1 066 MB |

The pictures unchanged on every stand (within the one-step floor). The frame rates unchanged
(ground 28.4, hill 27.0, aloft 52.4, orbit 52.4 on the still stand).

**WHERE THE FOOTPRINT GOES NOW (the ground stand, 6 611 MB), MEASURED by category.**

| Category | MB | What it is |
|---|---|---|
| graphics buffers (owned, unmapped) | 2 870 | the chunks on the GPU: 2 460 MB drawn plus the slab rounding, the depth and prepass targets; 145 regions, the engine's slabs |
| owned, unmapped, not graphics | 1 546 | 12 371 regions — about two per chunk of the fill; the upload path's working set (below) |
| small malloc blocks | 1 967 | the heap holds 712 MB IN USE (`heap -s`: 161 143 blocks); the rest is freed pages the allocator kept — it fell to 860 MB after the walk and 351 MB after the 528 m/s leg |
| large malloc blocks | 161 | the parent cache and the lane |
| the binary and its libraries | 372 | resident text |

**THE UPLOAD PATH DOES NOT LEAK, MEASURED on the moving eye (three legs, the footprint at both
ends of each leg):**

| Leg | Chunks harvested | Footprint at the end | Growth: total / graphics / unmapped / small malloc |
|---|---|---|---|
| walk 1.4 m/s | 135 | 5 569 MB | −225 / +1 / 0 / −227 MB |
| hull 240 m/s | 9 767 | 5 561 MB | +528 / +507 / 0 / +4 MB |
| hull 528 m/s | 21 544 | 4 713 MB | −901 / −198 / 0 / −707 MB |

The owned-unmapped memory grew by ZERO over 31 000 uploads: it is a pool bounded by the fill's
peak, not a per-upload cost. Its size follows the drawn bytes on a still stand (0.64 × drawn
across the four stands) and its region count is about two per chunk of the fill, which fits the
two writes an upload makes (the vertex slab and the index slab) through the engine's staging path
— that attribution is a HYPOTHESIS consistent with the count, UNMEASURED. The graphics buffers grew
with the drawn set (6 677 → 7 564 chunks on the 240 m/s leg) and shrank when it shrank.

**Example.** A pilot at 528 m/s for a minute has the client upload 21 544 chunks — three times
the ground stand's fill — and the client ends the minute 900 MB SMALLER than it started it,
because the allocator returned the pages the fill had dirtied and the drawn set shrank.

**What is left, and where it lands.** (1) The graphics buffers are the drawn bytes: the lossy
packing (V17 item 2, the tolerance decision) is the only lever, and it waits on the owner. (2) The
upload pool at 0.6 × the drawn bytes: a persistent staging ring would bound it by the frame's
uploads instead of the fill's — an engine-side change, deferred until the number matters
(D-TERRAIN-5 item 14). (3) The allocator's retained pages: a fill dirties 2 GB of small blocks
that the heap does not hold; they return over the next minute of flight. An allocator with a
different policy is a new dependency and the owner's call, not this slice's.

### 17.7 The second half, FINAL (2026-09-10, the gate chain on the refuted tree, Docker off)

Coverage 100 % (the byte-bounded cache and the bucket passes included), lint, the combos, the
pin, the pictures (four stands within the one-step floor, zero holes) and the moving eye green.
The census with the byte-bounded parent cache (256 MB of meshes, never fewer than the workers'
working set), the shrunk parent mesh and the render-world-only chunk mesh:

| Stand | Chunks | On the GPU | Footprint (graphics / unmapped / small malloc) | Frames/s, still (§17.5 → now) |
|---|---|---|---|---|
| ground | 6 659 | 2 460 MB | 6 525 MB (2 870 / 1 543 / 1 885) | 28.0 → 29.4 |
| hill | 7 360 | 2 582 MB | 6 235 MB (2 885 / 1 645 / 1 472) | 26.4 → 27.5 |
| aloft | 3 529 | 788 MB | 2 815 MB (1 166 / 462 / 970) | 52.4 → 51.9 |
| orbit | 1 457 | 339 MB | 1 604 MB (751 / 225 / 424) | 52.4 → 52.4 |

The moving eye (the run that passed; frames a second, §17.5 → now): the walk 26.9 → 29.5 with
zero frames with a gap; 240 m/s 43.8 → 45.1 with zero frames with a gap and the parent cache at
90 % (76 % before the shrink freed the budget); 528 m/s 24.4 → 32.1, the probe's low pass at a
gap of 1 026 chunks and a queue of 1 727 (the same wall, §16.6), the cache at 89 %. The memory
over the legs: the unmapped pool grew by zero on every leg (0 / 0 / 0 MB over 136, 10 013 and
21 291 uploads); the graphics buffers followed the drawn set (+628 MB as the screen grew to 7 622
chunks, −144 MB as it shrank).

**THE BOARDING THAT NEVER SETTLED (one run of two, OPEN — D-TERRAIN-5 item 15).** The first
run of the final chain went red at "hull, aboard": after the crossing into the hull (which ran
its saga twice, attempt 0 and attempt 1 ten seconds apart, on the failed run AND on the run that
passed) the ladder never settled in three minutes. Its last state: the drawn eye 20 km above the
surface by the planet's delivered box, the LEAD eye 5 591 km from it and 41 km UNDER the surface
by the stamp, zero chunks drawn, 4 846 pending, 3 066 urgent, 238 265 chunk builds over the
client's life — a thrash: the lead eye far from the drawn one wants everything within its reach,
under-surface (item 12), and the wanted set flips. The second run settled at once and flew both
legs green, so the state is a race at the boarding, not the tree. UNMEASURED: what the lead did
between the origin swap and the stall — the settle wait was one blocking call and kept only its
last state. The wait now POLLS and prints its course every two seconds (drawn, pending, urgent,
lead, altitude, origin, own pose, the entity windows), so the next occurrence is a diagnosis.
The lead's own formula is the suspect: the offset between the own pose composed at the drawn
cursor and at the lead cursor, which straddle the origin swap for one buffer.

## 18. The lossy packing under the tolerance (ruling V18)

### 18.1 The tolerance gate, BUILT and MEASURED on the exact tree (2026-09-10)

The exact pictures of the four stands are frozen beside the owner's (`pictures/exact/`, the final
chain of `ef2b33a`). Every flight compares its picture against the frozen one over the content
pixels (the probe's terrain and ruler, the overlays' union rectangle left out) and a channel step
above ONE level is a red gate. The exact tree against itself: ground 0, hill 9, aloft 2, orbit 11
pixels at one level — the gate's own noise floor, inside the tolerance.

### 18.2 The packed normal, MEASURED (2026-09-10)

**What it is.** A unit normal as two signed 16-bit numbers on the octahedron (the sphere folded
onto a square) — four bytes against twelve; the shader unfolds it. The encoder picks, of the four
quanta around the folded point, the one whose unfolded normal lies nearest the true one, judged by
the cross product in 64-bit (a 32-bit dot near one cannot tell two candidates a ten-thousandth of
a radian apart — MEASURED: the search picked a worse quantum until the comparison moved to the
cross product). Over a real rung-1 chunk the widest angle is 4.2e-5 rad precise against 6.3e-5
plain.

**Example.** The hill's crease at (641, 300) has a normal 31° off the radial; packed and unpacked
it is 31° off by less than three thousandths of a degree.

**THE LIMB.** With every rung packed, three stands stayed within one level and the orbit stand
moved ONE pixel of 466 445 by three levels — at the planet's limb, (807, 305), the same pixel on
every run and under both roundings. The lighting at a limb divides by the view angle's cosine,
which is near zero there, so it reads a normal's error a hundredfold: no 16-bit normal passes a
one-level rule at a limb. The near stands never show a limb (the horizon is rough ground, not a
smooth sphere's edge). So THE NORMAL IS PACKED BY RUNG: rungs 0 to 8 packed, rungs 9 and up (cells
of 512 m and up, the far view of a body from high) exact. The mesh's own layout tells the shader
which form it carries (a shader define), one shader source serves both.

**MEASURED, four stands, Docker off, against the frozen exact pictures:**

| Stand | GPU bytes, exact → packed by rung | Content pixels at one level | Widest step | Frames/s, still (runs vary ±5 on the far stands) |
|---|---|---|---|---|
| ground | 2 460 → 2 080 MB | 1 721 of 635 560 | 1 | 29.4 |
| hill | 2 582 → 2 186 MB | 2 665 of 701 472 | 1 | 27.4 |
| aloft | 788 → 732 MB | 1 208 of 586 341 | 1 | 47.8–52.4 |
| orbit | 339 → 339 MB (all exact) | 7 of 466 445 | 1 | 47.3–52.9 |

The frame rate on the still stands did not move with the bytes: the ground stand reads 29.4 before
and after. The still stand is the draw count (§16.5), and the packing owns the bytes only. The
far stands' rate varied by five frames a second between flights of ONE binary (46.8 then 52.4
aloft; 47.8 then 52.9 orbit), and the orbit stand — whose render path is the exact tree's, every
rung exact — read 47.9 on the answered tree against 52.4 on the exact one. MEASURED beside it:
two other sessions' processes held 27 % and 20 % of a core during those flights (load 6.7). A
far stand's frame is the main thread's, and the main thread shares the machine; a far stand's
rate is quoted as a range until the census takes more than one sample on a quiet machine.

**MEASURED on the moving eye (Docker off, the worker packing, the answered tree):** the walk
29.9 frames a second with zero frames with a gap; 240 m/s 46.1 (45.1 before the packing) with
zero frames with a gap and the parent cache at 90 %; 528 m/s 30.9 (32.1 before, inside the
run-to-run range) at the same wall, the queue at 1 967. The harvest's cost on the main thread
stayed at 0.02–0.03 ms a chunk: the packing runs on the worker (refutation N-2), and the harvest
never saw it.

**Two lessons the flights taught.** (1) The ruler ball is lit by the engine's standard material,
which reads the engine's normal; with the packed one in its place the ball became a flat dark
disc (145 levels). The ball now carries both forms, its probe twin reads the packed one. (2) A
difference count without the gate's masks misleads: 48 "terrain" pixels at 138 levels were the
HUD's tick readout, which the gate's overlay rectangle already leaves out.

### 18.3 The 16-bit position, NOT BUILT — the recommendation

The position at 16 bits over the chunk's box would take the vertex from 32 to 28 bytes: about
200 MB at the ground stand, and by the measurement above no frames a second on a still stand. Its
risk is the crease flip (a point moved a millimetre at rung 0 hands a pixel to the other slope —
the 16-bit radial's own failure, 19 levels), and its machinery is a per-mesh scale in the
transform, which bends the radial's rotation-only path. Against that, the far-rung voxel renderer
(D8-8) moves the bytes AND the draw count. RECOMMENDED: the position stays exact; the packing ends
with the normal; the next measurement is D8-8's.

## 19. THE FAR-RUNG VOXEL RENDERER — the LOOK, MEASURED (D8-8, 2026-09-11)

### 19.1 The fact the tier rule sets

The tier rule draws every cell at one to two pixels: rung `r` serves the distances where its cell
is between one and two pixels high, and rung 0 alone grows past that near the eye (34 pixels at
17 m). So on every stand every drawn cell above rung 0 is one or two pixels on the screen, and a
mesh's smoothness inside a cell is a smoothness no pixel shows. At the ground stand 86 % of the
chunks are rungs 1 and up; at the aloft and orbit stands, all of them.

### 19.2 The instrument

A LOOK switch, `VD_TERRAIN_SPLATS=<rung>`: from that rung up a chunk is drawn as one
camera-facing square per surface vertex — the extractor's own vertex (surface nets place one per
surface cell), with its normal, its morph metre and its radial, so the crossfade and the light
are the mesh's own — instead of the extracted triangles. The picture gate in report-only mode
(`VD_PICTURE_REPORT_ONLY=1`) captures on a three-times coarser tick grid (the splat form fills
slower, §19.4), withholds the tolerance's verdict, keeps its pictures beside the run and leaves the
owner's untouched; an exact flight on the same grid is the reference, and an offline compare
(`look_compare.py`) counts the content pixels by their channel step.

**The quad form is a look instrument, not a product form:** four copies of every vertex, so a
splat chunk weighs 1.5 to 3.8 times its mesh. The product form (§19.5) is a different build.

### 19.3 MEASURED: the look

Two-cell splats from rung 3 up (every chunk of the far stands, 61 % of the near stands'), against
the exact look on the same tick:

| Stand | Content pixels | Differ | At one level | Over five levels | Where |
|---|---|---|---|---|---|
| ground | 637 061 | 6 478 (1.0 %) | 2 276 | 2 014 | the skyline band, one pixel high |
| hill | 723 903 | 52 643 (7.3 %) | 29 725 | 2 921 | the skyline and the far slopes' shade |
| aloft | 588 206 | 279 054 (47 %) | 260 031 (44 %) | 2 852 | the whole ground at ONE level; the limb over five |
| orbit | 467 582 | 171 217 (37 %) | 147 836 (32 %) | 1 834 | the same |

Read: a splat's normal is flat across its square where the mesh's is interpolated, so the far
ground shades ONE level differently over broad waves (the aloft difference image is white waves
over the whole ground: `pictures/look/aloft_diff_white_1_level_red_over_5.png`). Every difference
past five levels sits on a SILHOUETTE: the skyline seen from the ground, the planet's limb seen
from high — a square's edge against the sky is not a curve's. To the eye the two looks are the
same picture (`pictures/look/*_mesh_above_splats_below.png`).

**The crack, and its cure.** At ONE cell wide the aloft stand showed 3 183 pixels of sky through
the ground between splats (the hole census): surface-nets vertices stand up to 1.7 cells apart on
a diagonal. At TWO cells wide, zero holes on all four stands. Two cells is the width.

### 19.4 MEASURED: what this form costs, and what was not measured

GPU bytes at the ground stand 2 080 → 4 023 MB (four copies of every rung-3-and-up vertex), at
the aloft stand 732 → 2 769 MB, at the orbit stand 339 → 1 279 MB; the fill of the ground stand
10 → 45 s. The frame rates of these flights are NOT quoted: Docker restarted during them (its
machine held 9.6 GB, the swap 10.6 GB, the load average reached 104), the exact look itself read
14 frames a second at the ground against 29 before, and the near stands missed their capture tick
twice. The rate of the product form is measured when it exists, on a quiet machine.

### 19.5 What the measurement decides, and what it leaves to the owner

1. **The look is the same.** A far cell drawn as a two-cell splat with the vertex's own normal
   shades within one level of the mesh over the whole ground and differs only on a one-pixel
   silhouette. The far-rung voxel renderer is a LOOK the owner can accept from these pictures.
2. **The product form is vertex pulling, not quads:** one record per surface vertex (position
   12 bytes, packed normal 4, morph 4, radial 12 — or less) in a storage buffer, the square
   spread by the vertex stage from the vertex index; no index buffer, no skirts, no triangles.
   Bytes: about a third of a mesh chunk's. The draw count falls only when a rung's chunks merge
   into one buffer, which is the second step. Both are engineering with a measured look behind
   them, not a look decision.
3. **The silhouette.** A splat skyline is a row of squares; at two cells wide it reads as the
   mesh's within a pixel. A softer edge (a round splat, a per-splat depth) is a look refinement
   the owner may ask for after seeing the product form.
4. **The near rungs stay meshes.** Rung 0 cells are up to 34 pixels wide; a splat there is a
   visible square. The rung the splats begin at is the owner's look choice, from 1 up; the
   measurement above shows rung 3 and up.

**Example.** A pilot at 60 km sees 3 529 chunks, every cell one or two pixels. Today that is
3 529 meshes of 15 million triangles and 732 MB. As splats it is 15 million squares from one
record each, about 250 MB, and the same picture within one brightness level.

### 19.6 THE STILL STAND'S WALL, NAMED BY ABLATION (2026-09-11) — it is the shadow, not the draw count

Before the product form was built, the frame was taken apart: the engine's frame-time diagnostic
and its render diagnostics on the stamp (CPU milliseconds per render pass; the GPU side records
nothing on Metal, which offers no timestamps inside passes), the GPU's busy share from the
driver's own statistics (`vd_bins::memory::gpu_busy`, ten readings while the stand holds still),
and two dev switches: the sun's shadows off (`VD_TERRAIN_SHADOWS=0`) and the chunks from a rung
up spawned hidden (`VD_TERRAIN_HIDE_RUNG`). A census-only flight (`VD_PICTURE_CENSUS_ONLY=1`)
prints the numbers and judges no picture. Docker was on (its load steady in this hour); every
flight is relative to the baseline of the same hour.

| Flight | Ground frames/s (ms) | Ground GPU busy | Hill frames/s (ms) | Aloft | Orbit |
|---|---|---|---|---|---|
| baseline | 28.9 (34.1) | 86 % | 26.9 (37.3) | 48.8, 37 % | 52.9, 31 % |
| shadows OFF | **53.9 (18.2)** | 72 % | **51.9 (18.6)** | 53.0 | 52.9 |
| rungs 6 and up hidden | 29.0 (34.6) | 88 % | 27.9 (36.2) | 52.4 | 52.4 |
| rungs 3 and up hidden | 31.9 (30.7) | 82 % | 31.4 (32.9) | 52.9 | 52.9 |
| rungs 1 and up hidden | **52.9 (17.3)** | 66 % | **54.0 (18.2)** | 51.9 | 51.9 |

Read, in order of weight:

1. **The near stands are GPU-bound** (86 % busy with the client's CPU at 36 %; the render graph's
   encoding costs 0.2 ms), and **the shadow is the whole wall**: with the sun's shadows off both
   near stands sit at the frame runner's cap (about 53 frames a second at 18 ms — the headless
   runner ticks at 60 Hz). The encoding of the four cascade passes costs nothing; their GPU work
   costs 16 ms: every caster in the shadow's reach (rungs 0 to 2, about 2 600 chunks, 7 000
   vertices each) runs the full morphing vertex stage FOUR times a frame — about 73 million vertex
   runs for the shadow against about 10 million for the picture.
2. **The far rungs cost the near stands nothing.** Rungs 6 and up hidden: no change. Rungs 3 and
   up hidden: three frames a second. So the far-rung voxel renderer (§19.2–19.5) would not move
   the still stand's frame rate; its gain is bytes (a fifth to a third) and the far stands, which
   already sit at the cap.
3. **Rungs 1 and 2 are the shadow's casters.** Hiding them (1 649 chunks with cells of 2 to 4 m)
   reaches the cap exactly as shadows-off does, because the shadow's reach ends at rung 2's switch
   distance: they are the casters the four cascades draw.
4. **The far stands are capped, not bound**: 22 to 40 % busy at the runner's 60 Hz.
5. **The GPU has a 30 % baseline from other applications** (the browser, Docker's machine,
   Spotlight) in these readings; the client's own share on a near stand is about 55 %.

**Example.** A pilot standing on the ground sees 6 659 chunks. The picture costs the GPU 18 ms.
The sun's four shadow maps cost it another 16 ms, drawing the nearest 2 600 chunks four more
times each, and the frame is 34 ms. Take the shadow's cost away and the frame is the runner's
cap; take the whole far view away and it is 34 ms still.

**What this decides.** The far-rung renderer's product form is not the frame-rate lever the plan
took it for (§18.3 and the D8-8 row said "moves the draw count"; MEASURED: it does not). The
lever is the shadow: its cascade count, its reach, its casters' vertex path and its map size —
each a measurement of the same kind, none built yet. The owner decides the order.

### 19.7 THE SHADOW'S COST, TAKEN APART (2026-09-11, rounds two and three)

Round two flew the engine's own knobs (Docker's machine on, steady; every flight against its own
baseline of the hour): the cascade count, the reach, the map size.

| Flight | Ground frames/s | Hill frames/s |
|---|---|---|
| baseline | 28.9 | 27.5 |
| two cascades (four today) | 32.9 | 28.9 |
| the reach at rung 1 (rung 2 today) | 34.0 | 30.9 |
| both | 36.0 | 32.9 |
| maps of 1 024 pixels (2 048 today) | 28.0 | 27.9 |
| two cascades and 1 024 | 33.5 | 28.5 |

None reaches the cap that shadows-off reaches (54). Round three split the shadow into its halves
on the quiet machine (Docker off, load 10): the casters (chunks drawn INTO the maps) and the
receivers (pixels that READ the maps).

| Flight | Ground | Hill |
|---|---|---|
| baseline, quiet | 26.0 | 25.4 |
| no casters (every chunk `NotShadowCaster`: the maps stay empty) | **55.0** | **45.4** |
| no receivers (every chunk `NotShadowReceiver`: the maps are drawn, never read) | 27.9 | 26.9 |
| shadows off | 54.9 | 49.9 |

**Read.** Drawing the casters into the maps is the whole cost; reading the maps costs two frames a
second. The map's size costs nothing: the fill of the maps is not it. The cascade count and the
reach each recover a fifth of it: the cost follows the casters' VERTEX work, and every cascade's
light-space frustum at a sun 15° over the horizon stretches across most of the reach, so halving
the cascades does not halve the casters drawn, and shortening the reach removes only the outer
ring. About 2 600 chunks of 7 000 vertices run the ground's full morphing vertex stage into each
cascade: the vertex throughput, not the pixels.

**Example.** A pilot on the ground at a low sun. The sun's maps are empty in one flight and the
frame runs at the cap; they are full and unread in the next and the frame is 26 frames a second.
The pixels never asked the maps; the vertices filled them.

**The levers that remain, each a measurement:** which rungs cast (round four: only rung 0; rungs
0 and 1), a lighter caster vertex stage (the morph and the sink left out beyond the first
cascade), and the terrain horizon map, which draws no caster at all for the ground's own shadow.

### 19.8 THE CASTERS BY RUNG (round four, quiet machine, 2026-09-11)

| Casters | Ground frames/s (ms) | Hill frames/s (ms) |
|---|---|---|
| none | 55.0 (18.2, the cap) | 45.4 |
| rung 0 only (949 chunks, 1 m cells) | 45.9 (21.8) | 40.9 |
| rungs 0 and 1 (+818 chunks, 2 m) | 35.9 (27.9) | 33.4 |
| rungs 0 to 2, today (+831 chunks, 4 m) | 28.9 (34.6) | 26.9 |

The cost per rung GROWS with the rung: rung 0 costs about 4 ms, rung 1 about 6, rung 2 about 11
(the ground stand; the baselines of rounds three and four, 26.0 and 28.9, bound the noise at
three frames a second). The reason is the sun's height: at 15° over the horizon a hill 3 km out
shadows the ground at the eye's feet, so every cascade's light-space frustum — the near 55 m one
included — must draw the far ring toward the sun, and the far ring is drawn four times. This is
what a cascaded shadow map costs at a low sun on open ground, in every engine.

**The levers, with what each is MEASURED or ESTIMATED to give at the ground stand:**

| Lever | Frames/s | What it costs the look | Kind |
|---|---|---|---|
| cast rungs 0 and 1 only | 35.9 MEASURED | hills past 1.7 km throw no shadow | one number |
| two cascades at rung 1 | 36.0 MEASURED | a coarser near shadow, and the same loss past 1.7 km | two numbers |
| a lighter caster vertex stage (no morph past the first cascade) | UNMEASURED | none the eye sees at a map's texel | one shader define |
| a coarser shadow ladder: the far rings cast with meshes two rungs coarser, on a render layer the sun sees and the camera does not | ESTIMATED 16× fewer caster vertices for the far ring | none at the map's 1.7 m texel | a second wanted set |
| the terrain horizon map: no caster for the ground's own shadow | ESTIMATED the cap, plus one small cascade for movers | a shadow with no fixed reach at all | a slice of its own |

## 20. The harvest's byte budget and the boardings (2026-09-11)

### 20.1 The byte budget (ruling V15 item 2), MEASURED on the 528 m/s leg

The harvest now stops at a byte budget or at a count, whichever comes first (`poll_within`; the
first chunk always comes). The count cap rose from 24 to 48 and the budget stands at 24 near
chunks of 400 KB, so a frame of near chunks uploads what it uploaded before and a frame of far
ones (200 KB) uploads up to 48.

| Setting | Harvested/s | Harvest full, frames | Gap peak (chunks) | Frames with a gap | Queue |
|---|---|---|---|---|---|
| count 24 (before) | 355–358 | 736 of 1 852 | 1 026–1 181 | 597–673 | 1 700–1 967 |
| bytes 24 × 305 KB, count 48 | 300 | 1 184 of 1 846 | 1 616 | 1 025 | 2 551 |
| bytes 24 × 400 KB, count 48 | **367** | 646 of 1 927 | **693** | **460** | **1 323** |

**Read.** A budget at the mean near chunk's bytes bound BELOW the old cap (a near chunk with its
skirts weighs about 316 KB), throttled the harvest to 300 a second and let 2 551 finished chunks
wait — 2.2 GB of small allocations in a minute, because a finished chunk waits with its whole
geometry (DEFERRED item 19: the done queue needs a bound). A budget above the cap's worth, with
the count cap doubled, gave the best 528 m/s leg yet: the harvest at 367 a second, the gap's peak
and the frames with a gap down by a third, the queue down by a third. The 240 m/s leg and the
walk unchanged (zero frames with a gap).

**Example.** At 528 m/s over the hill stand the ring ahead is mostly rung 3 to 6 chunks of 200
to 250 KB. Under the count cap of 24 a frame uploaded 24 of them and left the rest; under the
byte budget the same frame uploads 38 to 48, and the queue drains faster than it fills.

### 20.2 The boardings, MEASURED (DEFERRED item 15)

A diagnosis flight boards `VD_BOARDINGS` times before the legs, each pilot fresh, each on its own
hull (the shipyard stand-in's `--seq`), the berths 200 m apart along the path. The first form
put five pilots on ONE hull: the third never crossed within five minutes — the two boarded
pilots' bodies stood at the berth, and players collide. With one hull per boarding, five of five
boarded and settled in two to six seconds each, the settle course printing a normal refill after
the origin swap (the lead at zero, the altitude sane). The race stands at one failure in ten
boardings today, still unseen with the course in the log.

### 19.9 THE LIGHT CASTER, MEASURED (round five, quiet machine, 2026-09-11)

The shadow passes' vertex stage without the morph and the sink (`VD_TERRAIN_LIGHT_CASTER=1`, one
define in the prepass shader):

| Flight | Ground frames/s | Hill frames/s |
|---|---|---|
| baseline | 30.0 | 27.0 |
| light caster | 29.4 | 27.5 |
| light caster, two cascades at rung 1 | 39.4 | 35.4 |

**Read.** The work per caster vertex is not the cost: taking the morph and the sink out of four
passes over 2 600 chunks changed nothing. The COUNT of caster vertices is — with two cascades at
rung 1 the light caster adds three frames a second over the same setting with the full stage,
and no more. So the lever is fewer caster vertices per area: the coarser shadow ladder (the far
rings cast with meshes two rungs coarser, sixteen times fewer vertices, on a render layer the
sun alone sees), or no ground casters at all (the terrain horizon map). The light caster stays
as an instrument; it ships with the ladder if the pictures allow it, since alone it buys nothing.

### 19.10 THE SHADOW LADDER, BUILT (option A, owner 2026-09-11)

**What it is.** From rung 0 up (the shipped default after round seven, below; the first form
started at rung 1 with two rungs), a drawn chunk casts no shadow itself. The chunk ONE rung
coarser that holds its volume (`coarse_key`: each axis quartered, the same halving `parent_keys`
does one step at a time) is asked from the lane at the margin class's priority, built by the same
workers, and spawned on THE SHADOW LAYER — a render layer the sun sees and the camera does not
(`RenderLayers::from_layers(&[0, SHADOW_LAYER])` on the sun) — with a LIGHT-CASTER material: the
rung's own crossfade material under a pipeline key that makes the shadow pass's vertex stage skip
the morph and the sink and sink the vertex by THE CASTER'S SINK instead. A drawn chunk holds the key
of the caster it asked for; the caster leaves with its last wanter (a count per caster key). A
coarse caster shares one key space and ONE residency with the drawn chunks (refutation of the
ladder, findings 1–3): a key the ladder draws casts itself while a finer chunk wants it as a
caster and is a non-caster otherwise; a key held as a caster that the ladder comes to want is
dropped and rebuilt as a drawn chunk; a drawn key's residency is never released by a caster's
want ending. A caster's culling box grows by its own sink (finding 4); a caster's request sorts
behind every margin chunk of its rung (finding 6); the stamp's nearest and farthest skip casters
(finding 7).

**The caster's sink.** The coarse surface and the fine one disagree by the recipe's own bound
between their rungs (`dropped_bound_m`, the octaves the coarser rung drops) plus a cell of each
for the extractors' placement. The caster stands that far UNDER the drawn ground along every
vertex's radial, so the fine ground never shades itself against a surface that stands above it
(shadow acne); what remains is a shadow that floats off its hill by up to that bound — light
leaks — which the pictures measure. Example: a rung-2 chunk casts from a rung-4 caster sunk by
the two dropped octaves' amplitude plus 20 m; at 3 km that is a few pixels at the shadow's edge.

**Sixteen times fewer caster vertices per area:** the rung-2 ring, 831 chunks, casts from about
52 rung-4 chunks. The stamp carries the casters' count and bytes (`shadow_casters`,
`shadow_bytes`); the census prints them. Switches: `VD_TERRAIN_SHADOW_COARSE_STEP` (0 turns the
ladder off; the default 2) and `VD_TERRAIN_SHADOW_COARSE_FROM` (the default 1).

**MEASURED (2026-09-11, quiet machine, Docker off):**

| Flight | Ground frames/s | Hill | Coarse casters at the ground (MB) |
|---|---|---|---|
| ladder off (every drawn chunk casts) | 29.4 | 27.4 | 0 |
| ladder, two rungs from rung 1 (the default) | **44.8** | **39.8** | 486 (93 MB) |
| ladder, one rung from rung 1 | 44.8 | 39.3 | 737 (183 MB) |

One rung and two rungs read the same: the coarse casters are no longer the cost, and what
remains over the cap (55) is rung 0 casting its own ring, about 4 ms, plus the picture. The far
stands are unchanged (at the runner's cap).

**The look, against the exact look on the same tick (the overlay masked at the hill's wider
line):** ground 22 pixels of 635 560 differ, 2 of them by more than five levels; hill 39 of
701 137, 12 by more than five (the widest 18); aloft 14 at one level; orbit 4 at one level. The
pictures are under `pictures/look/hill_shadow_ladder_*`. The dozen pixels are far shadow edges,
as §19.9's estimate said; no acne (the caster's sink holds), no floating shadow the eye finds.
Under ruling V18's letter the twelve pixels are past the tolerance; the shadow is a look the
owner accepts or refuses from the pictures, and on acceptance the exact references are frozen
again with the ladder in them.

**What the casters cost.** 486 casters at the ground stand hold 93 MB and were built by the
workers beside the drawn chunks (the fill 8.6 → 8.1 s: no slower). Casters are asked for every
drawn rung from 1 up, the rings past the shadow's reach included; those never cast (the cascades
end at 3.5 km) and could be left unasked — a lever for the memory, DEFERRED item 17's neighbour.

**The near ring too (round seven).** With the ladder from rung 0 — the ring at the eye's feet
cast by a coarser mesh as well — the ground stand reaches the runner's cap:

| Flight | Ground frames/s | Hill | Casters at the ground (MB) |
|---|---|---|---|
| from rung 1, two rungs (the default) | 44.5 | 39.8 | 486 (93 MB) |
| from rung 0, one rung (2 m cells cast for the 1 m ring) | **56.2** | **46.9** | 858 (221 MB) |
| from rung 0, two rungs (4 m cells for the 1 m ring) | 56.3 | 47.8 | 572 (146 MB) |

The near ring's own casting was the last four milliseconds. Whether a 2 m caster at the eye's
feet keeps the look — a rock a metre wide, a step, a doorway's edge throw shadows a 2 m mesh
does not know — is the look flight's question and the owner's call.

**The near caster's look, MEASURED (the exact look on the same tick, the overlay masked):**
from rung 0 with one rung — ground 17 pixels of 635 560 differ (2 past five levels, the widest
22), hill 52 of 701 137 (12 past five, the widest 18), aloft 5 at one level, orbit 4 at one
level: the same handful of far-shadow-edge pixels as the default, and nothing new at the eye's
feet on these stands (smooth ground; a metre-wide rock is a picture the stands do not hold yet).


### 19.11 ACCEPTED, REFUTED, FROZEN (2026-09-11)

**The owner's acceptance.** *"I've checked hill_shadow_ladder_exact_above_ladder_below.png, looks
ok."* The default is the accepted setting: from rung 0, one rung coarser (`SHADOW_COARSE_STEP = 1`,
`SHADOW_COARSE_FROM_RUNG = 0`). The exact references were frozen again with the ladder in them
(`VD_PICTURE_FREEZE=1`, a mode exclusive with the look grid and the census: ground 58.4 frames a
second, hill 48.4 on the freeze flight).

**The refutation** (`slice_08_shadow_refutation.md`): nine findings, all answered. The one that
mattered — a caster and a drawn chunk shared ONE residency in the lane, so a key built as a caster
blocked the ladder's own request for it, and a crossfade band could lose its coarse rung. The
invariant now: a key the ladder draws casts itself while a finer chunk wants it as a caster; a key
held as a caster that the ladder comes to want is dropped and rebuilt as a drawn chunk; a caster's
last unwant never releases a drawn key; a replaced drawn chunk is despawned with its twin and its
counts returned. Example: the hill's rung-1 chunk under the eye draws AND casts while the rung-0
ring at the eye's feet wants it; when the eye walks off and the rung-0 ring drops, the chunk keeps
drawing and stops casting.

**The gates on the answered tree** (Docker off, one job at a time): coverage 100 % with zero real
misses, lint, combos and the terrain pin green; the moving eye — the walk 51.3 frames a second and
240 m/s 46.3 with ZERO frames with a gap, 528 m/s 31.3 with a gap on 447 of 1 876 frames (460 on
the flight before the fixes: the throughput wall, unchanged), the boarding settled.

**The picture gate after the fixes** read red on the ground stand by TWO content pixels of
635 560, each a single far 2 m cell near the horizon about 20 levels darker: a drawn chunk that a
finer chunk wants as its caster now casts, where before it cast nothing (finding 3). The hill,
aloft and orbit stands matched their frozen pictures. The crops are
`pictures/look/ground_refuter_pixel{0,1}_exact_left_fixed_right.png`. The owner accepted the two
pixels (*"1. yes"*) and the four stands were frozen again on the answered tree; the picture gate
then flew against the new references.

**The picture gate against the new references, MEASURED:** ground 0 pixels differ, hill 11 of
701 472 at one level, aloft 17 of 586 341 at one level, orbit 3 of 466 445 at one level — green
under ruling V18. The still stands on the gate flight: ground 50.4 frames a second, hill 42.5,
aloft 53.0, orbit 52.4.

## 21. THE 16-BIT POSITION, MEASURED AND REFUSED (ruling V18, 2026-09-11)

**What was built, for the measurement.** Every vertex's offset from its chunk's origin as four
signed 16-bit quanta of the rung's own position lattice (a cell over 256, the extractor's own
quantum: 3.9 mm at rung 0), the fourth lane zero; the chunk's origin snapped onto that lattice so
two chunks state a shared vertex at ONE lattice point (no crack of the packing's own — a unit test
held it on two neighbours); a vertex past the lattice's reach (±128 cells) fell back to the float
form and was counted (zero fallbacks on every stand); the engine's own position attribute carried
under its own id in the 16-bit format (the engine reads a mesh's format from the mesh's layout,
so every pipeline specialised to it), the shader scaling the quanta by the rung's quantum from the
material's uniform; the culling box grown by half a quantum. The picture gate learned to judge
every stand at the flight's end (kept: one flight now reports all four).

**MEASURED, every rung packed, against the frozen exact pictures:**

| Stand | Bytes on screen | Frames a second | Pixels over one level | Widest step | Where |
|---|---|---|---|---|---|
| Ground | 1 886 MB (from 2 080) | 50.0 (from 50.4) | 351 of 5 068 changed | 19 | rungs 0–5, mostly rung 0 |
| Hill | 1 983 MB (from 2 186) | 42.0 (from 42.5) | 166 of 5 357 | 22 | rungs 0–5 |
| Aloft | 670 MB (from 732) | 53.5 (from 53.0) | 142 of 3 778 | 17 | rungs 9–11 |
| Orbit | 313 MB (from 339) | 52.4 (from 52.4) | 76 of 2 191 | 31 | rung 12 |

The changed pixels spread over every rung in proportion to the rung's share of the picture
(the probe's rung per pixel), so no per-rung rule saves it — the half-quantum shift is the same
fraction of a pixel at every rung, and a shading edge that crosses a pixel's centre hands the
pixel to the other slope (§18.3's crease flip, the 16-bit radial's own failure). Thousands more
pixels moved one level (within the floor). The gain: a tenth of the GPU's bytes on every stand
and no frame rate on any.

**RULED (owner, 2026-09-11): REFUSED** — *"1. yes, refuse."* Ruling V18's tolerance stands: the
position stays the engine's float. The packed-position path is removed from the tree (nothing
inert stays); this record and the four difference images (`pictures/look/*_packed_position_diff_*`)
are what remains of it. Step 5's packing ends with the normal.

## 22. STEP 6 — THE POP DETECTOR (ruling V14 M8-3, D8-2, D8-5; 2026-09-11)

### 22.1 The instrument

**A pop** is a change of the picture in ONE frame instead of across a band. The ruled instrument
is a frame-to-frame difference on the rung boundaries. Built:

- **THE PAIR.** Every two seconds of a leg the client records two CONSECUTIVE frames — the
  picture, the probe and the frame's own stamp — through `Record { consecutive: true }`: the
  first frame is the freshest rendered at or after the request, the second is asked for BY ITS
  INDEX (the first's plus one) from a readback ring of the last eight frames (`READBACK_RING`);
  a frame the ring has dropped is refused, never stood in for. The PNG encode runs on its own
  thread (MEASURED on the walk with the encode on the main thread: 51 → 42 frames a second; off
  it: 51). The terrain stamp travels with the readback FROM THE EXTRACT, so the dump beside a
  frame states the drawn camera of the very pixels beside it. Three wrong forms were MEASURED on
  the way: two requests paced by the clock (frames two to six apart); the stamp remembered under
  the capture counter in the serving system (the placement system TOOK the stamp out of the
  resource to publish it, so the extract found none and the dump fell back to the poll's stamp
  of a random later frame: pairs 0, +2, +3 apart in the terrain's own count while the capture
  frames were consecutive); the stamp kept in the resource (exact).
- **THE STAMP** carries the drawn eye in the body's frame and the camera's rotation in that
  frame (`eye_body_m`, `camera_body_xyzw`), and a dump beside a capture carries the renderer's
  capture frame index (`capture_frame`).
- **THE JUDGE** (`vd_client_harness::pop`, Tier-A): each terrain pixel of the second frame
  becomes a point in the body's frame (the probe's distance, the camera's inverse projection —
  `CaptureCamera::unproject`, the inverse of the ruler gate's projection) and projects into the
  first frame; pixels nearer than the near limit (a stated distance is half a cell coarse:
  `near_limit_m` bounds the miss to a quarter pixel) are left out. A pixel whose rung differs
  between the two frames is a boundary pixel. Every pixel — boundary and still alike —
  compares against the BEST MATCH in the reprojected pixel's 3 × 3 neighbourhood (MEASURED with
  the nearest pixel alone at 240 m/s: a floor of 43 levels, the ground's shading changing by tens
  of levels from one pixel to the next at a crease; with the neighbourhood: 7). The floor is the
  step under which 99.9 % of the still pixels lie; the reading is the boundary pixels past it
  and the widest, per boundary.
- **THE PICTURE GATE** holds every stand's verdict to the flight's end (§21).
- **AFTER THE REFUTATION** (`slice_08_pop_refutation.md`): ONE MOMENT PER FRAME — the camera
  samples the display time and the snapshot once (`RenderEye::moment`) and the wanted set, the
  stamp and the chunks' placement read that sample (three samples a frame had put the drawn
  ground a metre ahead of the stamped eye at 528 m/s); a reading carries HISTOGRAMS and a leg's
  sum pools them, so the floor, the count past it and the widest are one reading; a pixel is a
  boundary pixel when any terrain pixel of its neighbourhood was drawn by another rung; the
  box's facing is f64 end to end (a narrowed facing moved the stamped eye 0.7 m a frame on a
  turning hull); a capture with no stamp says so; the readback ring lives only around a capture.

### 22.2 The legs, MEASURED (pairs every two seconds, thirty a leg; the final flight, 2026-09-12)

| Leg | Frames/s | Band | Pixels compared | Beside a boundary | Floor | Past the floor | Widest | Where |
|---|---|---|---|---|---|---|---|---|
| Walk 1.4 m/s | 51.7 | held | 14.3 M | 105 634 | 1 | 243 | 15 | rung 0→1 (161, 11), 2→3 (56, 15): the picture's own shimmer at thin features, at the still pixels' rate |
| Hull 240 m/s | 44.6 | held | 13.3 M | 248 597 | 1 | 6 718 | 62 | rung 1→2 (4 340 of 33 850, 48), rung 2→3 (1 473 of 54 094, 62); rungs 3–8 under 10 |
| Hull 528 m/s | 34.9 | a gap on 166 frames | 13.2 M | 381 870 | 6 | 2 416 | 72 | rung 1→2 (1 032, 72), 2→3 (1 287, 68), 6→7 (39, 62) |
| Hull turning, last, at 528 m/s | 45.7 | held | 16.0 M | 269 417 | 66 | 174 | 86 | the turn's chord pooled into the leg (item 21), not a pop reading |

"Beside a boundary" counts every pixel whose 3 × 3 neighbourhood holds another rung (after the
refutation's finding 8), so it is the boundary's whole width on every pair. With ONE MOMENT PER
FRAME (finding 1) the floor at 240 m/s fell from 9 levels to 1, and the seam stands out: at the
rung 1→2 handover 13 % of the pixels step past the floor, up to 48 levels.

**THE SEAM THE DETECTOR FOUND.** At 240 m/s the pixels that cross the rung 1→2 and 2→3
boundaries step by up to 62 levels, on every flight (450 / 367 / 566 / 600 past the floor with
the centre-rung count, 6 718 with the neighbourhood's). The crossfade morphs POSITIONS onto the
coarser surface and keeps the finer rung's NORMALS, so at the fade-out edge, where the coarser
rung takes over, the shade jumps by the two rungs' slope difference — a crease of the finer
rung under a smooth face of the coarser. Example: a rung-1 chunk at 1.7 km lies on the rung-2
surface at its far edge; its vertices agree with the coarser ones to the millimetre, but a
gully the rung-1 normals still tilt into is flat to the rung-2 normals, and the pixel steps from
shade to lit in one frame. The cure is the normal's own morph: a finer vertex carries the
coarser surface's normal beside its own and the shader blends the two across the band as it
blends the positions (packed: four bytes a vertex; ruling V18's tolerance judges it) — a slice-8
item for the owner's word, recorded as D-TERRAIN-5 item 20.

**The 528 m/s leg, UNATTRIBUTED:** on this flight the band's gap peaked at 35 urgent chunks on
166 frames with the queue at 210, against 707 / 458 / 1 341 on every flight before. The two
changes since are the one sampling moment per frame and the readback ring's 62 MB; no ablation
separates them yet.

### 22.3 The turning leg, and what it found

**THE FIRST TURNING LEG (a five-second hold, no counter-turn) broke the world**: the lead eye
jumped 2 946 km, the ground on screen fell to ZERO chunks, 6 511 urgent, a gap on 853 frames,
18 frames a second with the workers building 1 600 chunks a second for a wanted set that never
drew — and it stayed broken for the 528 m/s leg after it. The kept run's dumps named the
mechanism: the planet's box in the pilot's window carried a LIVE centre (the per-tick track,
interpolated at the display cursor) and the LEVEL's facing (`overlaid_at` kept `..*boot`,
D-TERRAIN-5 item 11's "boot facing"), and the eye those two imply in the planet's frame —
the facing's inverse on the centre's negative — stood up to 280 km from the hull (one tick of
spin over a 6 371 km lever). ★ FIXED: the overlay takes the facing from the SAME live sample as
the centre (`live.orient`, slerped by the track as the centre is lerped). MEASURED after, on the
final flight: the band held on every frame of the turn, 6 378 chunks on screen at the least, the
lead back under 40 m once the spin was cancelled (during the turn it ran to 8 831 m: the chord,
item 21).

**The turn axis is a TORQUE**, and the hull keeps spinning after the axis is released
(MEASURED: about fifty degrees a second for the rest of the leg); the leg now turns for two
seconds, counter-turns for two, then cancels the residual in rounds (`cancel_spin`; on the
final flight the residual was −0.2°/s at 5.2 s). The product's own answer is
the ship's safety block (slowing is gameplay, ruling 2026-08-27 item 4); the instrument closes
the loop itself. The angular acceleration of a held axis was MEASURED at 37°/s per second
(`TURN_ACCEL_DEG_S2`).

**Two residues of a spinning parent, for the record (D-TERRAIN-5 item 21):**
- The track LERPS the centre and SLERPS the facing, so a parent spinning in the window cuts the
  chord of its arc between two ticks: during the fast turn the stamp's altitude dipped to −8 km
  for a frame and the detector's floor rose to 73 levels. Bounded by the spin over one tick;
  the exact form composes the centre from the interpolated placement (rotate, then subtract).
- The speed reading from the planet's centre in the window is a rotation's victim (587 km/s at
  5°/s of spin); the moving eye now reads the hull's place in the planet's frame
  (`hull_in_planet`), rotation-invariant.

**The order of legs**: a turn before the 528 m/s leg made that leg's push fire along the turned
nose (the hull climbed, the ground left the view, no pixel compared), so the turning leg flies
LAST, at the speed the hull has then.

**The spin's rest band**: the shortest hold the round trip allows (0.05 s) changes the rate by
about two degrees a second, so the loop stops there (MEASURED: 1.3 → 2.7°/s on a 0.04 s hold);
the residual spin's chord still lifts the detector's floor on the turning leg (65 levels at
2.7°/s: four metres of eye, four pixels at a kilometre), which is item 21's own measurement.

### 22.5 Ruling V14 D8-5, for the owner's word

D8-5 reads *"the pop detector flies a HULL at 1.4, 240 and 528 m/s (the suit ruling: never a
walking dot)"*. The moving eye's slow leg walks a character at 1.4 m/s — the band's own gate from
step 4 — and the detector reads it (floor 1, no pop). A hull flown at 1.4 m/s is a fifth leg if
the ruling's letter is wanted; the refuter raised it (finding 13), and it waits for the owner.

BUILT 2026-09-12 (the hardening arc, §23): the hull's legs are 1.4, 240 and 528 m/s, the slow one
first, from rest. MEASURED: the push quantum (one tick at the hull's rating) takes the hull to
2.8 m/s, past the target; the band held on every frame, the detector's floor 1 with 231 pixels
past it (the widest 23) at the rung 2→3 edge — the hull stands 1.35 km up, so rung 2 is its
finest. The ruling's letter is met.

### 22.4 The under-surface floor (D-TERRAIN-5 item 12)

An eye the recipe's surface stands over (a dip where the mesh cuts under the field, a cave, the
frame of a hard landing) read a ZERO horizon. Now the wanted set's altitude is floored at the
eye's own height (`EYE_HEIGHT_M`, the one datum the pilot camera lifts the eye by — the harness's
offset reads it from here): the horizon and the reach of an eye standing on the surface. The
skyline keeps its own truth below: MEASURED in the unit test, an eye ten metres under the field
wants the same reach and finest rung as a standing eye, its coarsest ring no coarser and its
chunks no more than the standing eye's (the far rings walled off by the ground around it: rung 3
against 10 on the fixture, a comment in the test, not its gate) — never the reach-wide flood.

## 23. THE HARDENING (2026-09-12, the owner: "build an extremely strong foundation first")

### 23.1 Item 21(a), the arc — DONE

A realm row is stated in the origin's frame: its centre is where that realm's origin stands as
seen from the pilot's own realm. When the origin turns, every row's centre swings on an arc
around the pilot, and a straight blend of two centres cuts the chord (§22.3: 8 km inside the arc
at fifty degrees a second). Now a row is blended as THE ORIGIN'S OWN PLACEMENT IN THE ROW'S
FRAME — the pilot's position there lerped, the pilot's rotation there slerped — and the row's
centre and facing are recomposed at the cursor: on the arc, with no new data and no knowledge of
which rows are the pilot's ancestors (every row is treated alike; the residue moves to the row's
own spin over one tick, half a millimetre on a planet's radius). Where nothing turns the plain
blend runs, byte-identical.

MEASURED: the four stands within one level (7 / 11 / 10 / 0 pixels); the turning leg at up to
fifty degrees a second — the lead 15 to 62 m through the turn (2 741 to 8 831 m before), the
altitude flat at 1 700 m (dips to −8 km before), the band held on every frame with 6 433 chunks
on screen at the least. The detector's floor on the turning leg stayed at 71 levels: a second
cause moves the whole picture while the hull turns (the shadow cascades re-fitted to a rotating
frustum are the suspect; an ablation with the shadows off measures it, §23.2).

The 528 m/s leg on the same flight: a gap on 567 frames, the queue at 1 666 — and 166 / 210 on
the flight before it. The wall's own variance is that wide; §22.2's "unattributed improvement"
was the variance, not a change.

### 23.2 The turning leg's floor, FOUND — the overlay's readouts (2026-09-12)

With the arc in place the turning leg's detector floor stayed at 70 levels, with the shadows on
and off alike (an ablation flight: the walk and the straight legs read the same without shadows,
the 240 m/s handover's count halved — the shadow caster changes rung at the same edge as the
normals do). A kept flight (`VD_KEEP_FIXTURE=1`) put the turning pairs under an offline reader:
the misses had no global shift (a pose or timing error would show one), and the step image
showed them: THE OVERLAY. The stamp's readouts stand inside the terrain the probe marks, the
judge never masked them, and while a hull turns the readouts change every frame ("the planet
turned …°", the lead, the light off the nose) — thousands of digit pixels at tens of levels,
enough to set the floor at the still pixels' 99.9th percentile. On a straight leg the readouts
barely change, so the floor read 1. The judge now masks the stamp's own rectangle
(`hud_rect_px`, grown by the glyphs' antialiasing as the picture gate grows it) in both frames.
The same reader showed a frozen pair inside a turn (two frames with one eye, no step): a
delivery stall at the render cursor, the tick-hitch class, seen once; the census counters of the
window's tracks are the instrument for it when it recurs.

MEASURED with the mask (the flight after §23.3's fix): the turning leg's floor fell from 70 to
14 levels (234 pixels past it of 18.7 million compared, the widest 48 at the rung 2→3
handover); the walk and the straight legs read as before (floors 1, 1, 2). The residual 14,
ATTRIBUTED by the step image of a kept turning pair: the pilot's own sphere (the stand-in body
drawn at the own pose, which the probe marks as its own kind, so the judge skips it) and ITS
CAST SHADOW on the ground — a dark wedge under the overlay that swings across the terrain as
the hull turns. The shadow's pixels are terrain in both frames, shaded by a moving object: a
true change of the picture, not a pop of the ladder. It sets the still pixels' 99.9th
percentile only on the turning leg, where the sphere swings.

### 23.3 A drive that outlived its realm — the boarding push, FOUND AND FIXED (2026-09-12)

The flight that carried the mask panicked before its turning leg: "the hull never reached
528 m/s (at 448.2 m/s)". Twelve pushes of 0.8 s each (a push adds 80 m/s at the hull's rating)
left the reading where it was. The kept fixture (`VD_KEEP_FIXTURE`) told the story from its
state dumps:

- During the walk the hull's box stood still at the berth (0.0 m/s over sixty seconds, facing
  identity). Before the pilot boarded, nothing pushed it.
- The first leg started with no push: the reading was already 448 m/s. From inside the hull
  the eye moved along the planet's body `−Y` axis at 443 m/s in a straight line — no gravity
  bent it, and the radial part (23 % of it) climbed the hull from 2 343 m to 7 248 m. And the
  planet's box facing in the window turned about the hull's own `Y` between every dump: THE
  HULL SPUN. A push along a spinning nose averages to nothing, which is why twelve pushes read
  as none.

What pushed it. The boarding is a crossing leg: `WalkTo` the berth, a point stated in the
planet's frame, in chunks of 400 ticks. In this flight the first crossing attempt was refused
(the hand-off found the pilot 5.16 m outside the hull, "the occupant has left the destination")
and the second attempt committed ten seconds later — INSIDE a walk chunk. From that tick the
drive read its planet-frame target from the pilot's new pose in the hull's frame: a point
6 200 km away, so a full stick on all three axes, every tick, for the rest of the chunk. Inside
a hull the pilot's stick is the hull's drive (the temporary control seam): a push on three
axes and a torque. 443 m/s along one body axis and a spin about it is exactly that stick held
for four and a half seconds. The green flights had committed while the pilot stood parked
(arrived, throttle cut, waiting), so the same chunk never straddled the commit there — the
difference between the flights was timing, not code.

The fix, at the harness seam (HR6, the product untouched): a closed-loop drive (`WalkTo`,
`LookAt`) remembers the location label of its first delivered pose. When a later pose carries
another label the drive pushes its release (a zero `Move` for the walk, a zero `Look` delta for
the look) and returns a new reply, `Crossed { state }` — its target was stated in the realm the
entity left, so continuing is not a walk any more. `vdctl` maps it to exit code 4; the crossing
legs read the reached label straight from the reply. Two findings this leaves open, both under
item 15: the first crossing attempt is refused on EVERY boarding (both green flights show two
saga starts ten seconds apart; the scan decides "inside", the hand-off seven ticks later finds
the pilot 5.16 m outside — two shapes, or a pilot still moving); and the §17.7 boarding that
never settled (the lead eye 5 591 km off, under the surface for three minutes) has the shape
of THIS mechanism at a higher spin — a chunk that straddled the swap, the hull driven and spun,
the lead eye swung on the chord — which the boarding storm can now confirm or refute.

MEASURED on the fix: the boarded hull stood at rest (the first push ran 2.43 s to 234 m/s, as
on the green flights), the 528 m/s push ran 2.91 s to 525 m/s, every leg flew, the band held on
every frame of the walk, the 240 m/s leg and the turn (the 528 m/s leg's wall: 527 gap frames,
1 731 queued — inside its measured variance), and the flight passed.

THE BOARDING STORM on the fix (`VD_BOARDINGS=10`, item 15): ten of ten pilots crossed and
settled, each in two to eight seconds, every refill course normal (the lead at zero, the
altitude sane, the urgent count falling to zero within six seconds); the boarding that never
settled (§17.7) did not recur. Nineteen saga starts for ten boardings: nine first attempts were
refused ("the occupant has left the destination", 5 m outside the hull seven ticks after the
scan found it inside) and one was accepted at once — the refusal is the rule, not a race, and
it costs ten seconds a boarding. It stays open under item 15 as its own question: the scan's
shape against the hand-off's.

### 23.4 Item 20, the morph normal — BUILT AND MEASURED (2026-09-12)

The crossfade morphed positions and kept each vertex's own normal (§22.2): a finer rung's
crease under a smooth coarser face shaded one way until the coarser rung took over, then the
other — at 240 m/s, 13 % of the pixels crossing the rung 1→2 boundary stepped by up to 45
levels in one frame. Now the parent mesh keeps its smooth normals (packed as the drawn chunks
pack theirs), the radial hit returns the parent's normal interpolated over the triangle it met,
and every vertex carries a MORPH NORMAL beside its own — the shade the next coarser rung draws
at its morph target (its own where no parent triangle stands on its radial, and at the top
rung). The shaders blend the two with the same weight that blends the positions, in the main
pass, the prepass and the probe alike. Four bytes a vertex (36 at the packed rungs).

MEASURED, the moving eye (pixels past the floor at the boundary, and the widest step):

| boundary                | before          | with the morph normal |
|-------------------------|-----------------|-----------------------|
| walk, rung 0→1          | 239 (20 levels) | 164 (13)              |
| 240 m/s, rung 1→2       | 3 843 (45)      | 376 (46)              |
| 240 m/s, rung 2→3       | 1 038 (32)      | 1 065 (26)            |
| 528 m/s, all boundaries | 7 122 (51)      | 3 998 (48)            |
| 528 m/s, rung 2→3       | 5 656 (51)      | 2 845 (48)            |

The rung 1→2 handover at 240 m/s fell tenfold. The rung 2→3 handover did not move: the shadows-
off ablation (§23.2) had already put that edge on the shadow caster, which changes rung at the
same boundary (item 18's territory, the caster's own handover). The widest steps stay (one
pixel in a million at a crease the parent's smooth normal cannot carry); the floors are
unchanged (1, 1, 2, 15).

MEASURED, the stands (the picture gate in report-only mode, against the frozen references):
ground 5 301 of 635 560 content pixels differ (the widest step 7), hill 116 713 of 701 472
(13), aloft 131 469 of 586 341 (8), orbit 6 946 of 466 445 (2). The change is the crossfade
bands' shading, which now hands over with the shape; on the hill the band's crease that stood
across the reference is gone. THE OWNER'S LOOK IS OWED before the references freeze: the
before/after crops of each stand's most-changed window are
`docs/investigation/2026-09-07/pictures/look/<stand>_morph_normal_before_left_after_right.png`
(the reference on the left, the morph normal on the right, three times life size).

THE OWNER ACCEPTED THE LOOK (2026-09-12, "the look is fine"): every stand's exact reference is
frozen on the gate's own grid with the morph normal in it — the four older stands and the seam
— and the strict gate reads green against them.

### 23.5 Item 19, the done queue — BOUNDED AND MEASURED (2026-09-12)

The workers handed finished chunks to the harvest over an unbounded channel: when the harvest
lagged, every finished chunk waited in memory with its whole geometry (2 551 of them on the
528 m/s leg, 2.2 GB of small allocations over the minute). The channel is now bounded from the
harvest's own rate — four frames of the harvest cap, 192 chunks — and a worker that finishes a
chunk while that many wait pauses on the hand-over until the harvest takes one.

MEASURED on the 528 m/s leg (the flight before, with the morph normal, against the flight
with the bound):

| the 528 m/s leg                    | unbounded  | bounded |
|------------------------------------|------------|---------|
| small allocations, growth          | +2 424 MB  | +14 MB  |
| footprint, growth                  | +2 743 MB  | +262 MB |
| resident, growth                   | +2 641 MB  | +3 MB   |
| chunks harvested                   | 23 000     | 23 389  |
| the queue's peak (pending)         | 2 027      | 1 044   |
| urgent chunks missing at the worst | 1 216      | 313     |
| frames with a gap                  | 671        | 502     |

The wall itself moved with it: the workers no longer build ahead into memory nobody drains, so
the queue and the worst gap fell by half or more (one flight; the wall's variance is wide,
§23.1, so the gap numbers are indicative, the memory numbers are not in doubt).

### 23.6 Item 18, the casters past the reach — BOUNDED AND MEASURED (2026-09-12)

Every drawn chunk from the coarse rung up asked for its coarse caster, wherever it stood: the
ground stand held 858 casters (221 MB), most of them past the sun's cascades and never cast.
Now the ladder view decides which drawn chunks may ask (Tier-A, from the eye): a chunk asks
while its column's nearest point lies within THE CASTER BOUND — the cascades' reach, plus the
longest shadow its caster can throw onto ground within the reach, plus the caster's own
diagonal — and the renderer wants and unwants casters as chunks enter and leave that set on
every recompute (a change of the sun's tangent past a tenth recomputes too). The sun's tangent
is read at every placement of the sun (the bias read it once, at its birth).

MEASURED, the first bound — the whole relief over the sun's tangent: the ground stand 858 →
838 casters (236 MB), the hill 858, the aloft 182 (45 MB), the orbit 0; the pictures within
run noise of the flight before (ground 5 301 → 5 301 pixels against the reference, hill
116 713 → 116 713, aloft 131 469 → 131 461, orbit 6 946 → 6 949 — the run-to-run noise is a
dozen pixels, §23.1), so no shadow was lost. The bound barely bit: a 5 km relief at a 15° sun
reaches 40 km, farther than the ladder's own drawn ground at the stands. The bound now reads
THE CASTER COLUMN'S OWN PEAK over the lowest ground within the reach (a caster whose peak
stands under that ground shades none of it, whatever the sun); its measurement follows.

MEASURED, the peak bound (the same stands, the same 15° sun):

| stand  | casters before | with the peak bound | the picture against the reference |
|--------|----------------|---------------------|-----------------------------------|
| ground | 858 (221 MB)   | 576 (182 MB)        | 5 301 pixels, as before           |
| hill   | 858 (238 MB)   | 556 (174 MB)        | 116 713, as before                |
| aloft  | 182 (45 MB)    | 0                   | 131 461, as before                |
| orbit  | 0              | 0                   | 6 949, as before                  |

No shadow was lost on any stand (the diffs against the frozen references are the morph
normal's, unchanged to the pixel on the ground and the hill). From 60 km up nothing lies within
the cascades' reach, so the aloft stand casts nothing at all now. On the ground a third of the
casters and forty megabytes went; the rest stand within the reach or hold a peak that can
shade it at a 15° sun — an honest bound, not a cap. The 528 m/s leg on the same flight: the
queue peaked at 1 113, 302 urgent chunks missing at the worst sample, 559 frames with a gap
(against 2 027 / 1 216 / 671 before items 18 and 19), the footprint grew 34 MB over the leg.

THE CASTER'S OWN NEAREST POINT (after the refutation, §23.8): the bound had tested the drawn
chunk's nearest point against the reach plus the shadow plus the CASTER'S DIAGONAL, and a
coarse caster's diagonal is hundreds of kilometres (a rung-12 caster: 227 km), so every coarse
chunk within that asked — the savings were the fine rungs' alone. Now the caster column's own
geometry gives its nearest point (floored at the eye's height over the relief, because a
column wider than the eye is high reads under the eye by the disc bound), and that is tested
against the reach plus the shadow alone. The caster's rung is clamped to the body's top rung
and a top-rung chunk never asks (it has no coarser rung).

MEASURED, the caster's own nearest point (the same stands, the same 15° sun):

| stand  | casters, unbounded | the peak bound | the caster's own nearest point |
|--------|--------------------|----------------|--------------------------------|
| ground | 858 (221 MB)       | 576 (182 MB)   | 414 (158 MB)                   |
| hill   | 858 (238 MB)       | 556 (174 MB)   | 394 (150 MB)                   |
| aloft  | 182 (45 MB)        | 0              | 0                              |
| orbit  | 0                  | 0              | 0                              |
| seam   | 616 (188 MB)       | —              | 488 (161 MB)                   |

Every picture unchanged against its reference to the pixel (ground 5 301, hill 116 708 against
116 713 — the run noise, aloft 131 461, orbit 6 946, seam 5 227): no shadow lost. Half the
ground stand's casters and sixty megabytes went. The 528 m/s leg on the same flight: the
queue peaked at 976, 276 urgent chunks missing at the worst sample, 416 frames with a gap
(2 027 / 1 216 / 671 before items 18 and 19), the small allocations grew 142 MB over the leg
(2.4 GB before item 19), and the turning leg's lead stayed within 26 to 45 m with the arc on
the origin's ancestors alone.

### 23.7 Two more stands — the seam kept, the feature refused by the world (2026-09-12)

THE SEAM: the eye on one of the cube's twelve edges, looking along it tilted as the ground
stand is, at the point where the star stands inside the gate's elevation band (12°–18°) and
nearest over the shoulder. The face bend must not show (SL8): a picture along the seam,
frozen, keeps it in the gate. MEASURED on the way there: the spot's own face's nearest point
by elevation put the star straight behind (177.75° off the nose); the nearest by the sum of
both misses put it 37.65° up; and over all twelve edges no point has the star both low and
100°–140° off a nose along the edge — the edges cross the star's low ring where they run
toward it (17.96° up and 177.71° off the nose at best). So the off-nose band is now the
picture's own: the four stands that choose their nose keep 100°–140°, the seam accepts the
star anywhere behind the shoulder (100°–180°; the light from behind, the ground lit, no
glare). A per-stand freeze (`VD_PICTURE_FREEZE=seam`, a comma list; `1` or empty freezes
every stand) lets a new stand take its first reference while the older stands' look stays the
owner's to accept.

THE FEATURE, refused: the stand was to picture the sharpest metre-wide bump or pit within
60 m of the spot from 4 m, with the low sun throwing its shadow (the 4 m caster at the feet
loses a metre-wide rock's shadow, §19.10). MEASURED from the recipe: the sharpest cell's
relief over its neighbours two metres off is 0.01 m — the recipe's finest octave has no
metre-wide feature there, and the picture would have been flat ground (the ruler's centroid
gate also read 1.07 px off at a 21° nose, the disc's own perspective). The stand comes back
with the block store's features, which are the metre-wide things the world will have.

### 23.8 The refutation of the hardening arc (2026-09-12)

A read-only refuter over the whole change set returned sixteen findings. What changed, and
what stands:

- **The caster's rung clamped at the global `RUNG_MAX`, not the body's top rung** (a defect):
  every top-rung chunk sampled a caster column on a rung the ladder does not have — a phantom
  span of twenty-five height evaluations per column per descent, kept in the span cache, for a
  verdict the renderer then discarded. Now the caster's rung is clamped to the body's top and a
  chunk with no coarser rung never asks. Tested.
- **The arc blamed the origin for a turn the ROW made** (a defect): a sibling realm spinning on
  its own (a tumbling hull ten kilometres away, fifty degrees a second) would have had its
  centre pulled toward the eye by the chord's deficit, 2.4 m per window, because one sampled
  facing cannot tell the row's own spin from the origin's turn. Now the arc blends the ORIGIN'S
  ANCESTORS alone (the origin and its parents up the scene's parent links — their facing change
  in the window IS the origin's own turn), and every other row keeps the plain blend, which is
  exact for a centre that does not move. The exact form for a sibling — its placement blended in
  the PARENT's frame and recomposed through the parent row's arc — needs no new data and is owed
  (item 21(a), below). Tested (the chain, a cycle, an absent parent).
- **A finite non-unit facing** (a zero quaternion, which the finiteness sanitizer passes) went
  through `normalize` to a NaN centre. It reads as identity now. Tested.
- **The crossing's release could be shed** by the full mailbox — the one push with no next tick.
  It is now offered until the mailbox takes it (two hundred polls at most). A look drive
  releases nothing (a zero look delta was a no-op).
- **The "no ground within the reach" arm was untested** (HR5): an eye 60 km up now tests it
  (no chunk asks for a caster). **The reach's hysteresis** was asymmetric and zero-width at a
  zenith sun: symmetric now, with a floor. Tested.
- **The caster-delta pass walked every realm's chunks** on every recompute (SL9): it ranges the
  realm's own keys now.
- **A zero overlay rectangle masked a 2×2 corner**: it masks nothing.
- **The packed vertex grew by four bytes and the harvest's byte budget did not** (an 11 % cut
  of the harvest under the byte bound): the budget grows by the same eighth (24 × 450 KB).
- **The parent's shade was decoded for every candidate triangle**: once, for the winner.
- Stands as measured, not changed: the bounded done queue parks every worker while the harvest
  lags (an urgent chunk may wait four harvest frames, 66 ms) — the flights read the worst gap
  falling 1 216 → 313 and the queue 2 027 → 1 044, so no inversion showed; the arc's absolute-
  metre arithmetic at a coarse tier (a galaxy row) resolves to ten kilometres at 10²⁰ m, an
  angle of 10⁻¹⁶, invisible.
- Confirmed by the refuter, not refuted: the bounded channel cannot deadlock (the send is
  outside the queue lock; the receiver drops after `Drop`, so a parked send returns); the bound
  is the live-geometry bound (nothing parks between the channel and the world); every ground
  mesh carries the morph normal; the blend weight is the position's; the published stamp cannot
  be stale; `Crossed` cannot fire on the first pose; the seam search cannot produce a NaN or a
  night-side stand; the caster bound over-estimates in every term; the four hull legs are
  wired in order.
- The refuter also held that item 20 is not 🟩 (the references are not refrozen; the owner's
  look is owed) and that the seam's reference was not yet frozen when it read the tree — both
  true; the seam is frozen now and item 20 waits for the owner, as its entry says.

## 24. THE FOUNDATION'S LAST FOUR (2026-09-12, the owner: "make sure we have a stable foundation")

### 24.1 The soak — a ten-minute walk, MEASURED

`VD_WALK_S=600` lengthens the walk, and every leg now prints THE MEMORY once a minute, so a slow
growth reads as a slope. The ten-minute walk, the client's growth since the leg began:

| minute | footprint | of it: graphics | unmapped | malloc large | resident |
|--------|-----------|-----------------|----------|--------------|----------|
| 1      | +95 MB    | +12             | 0        | +80          | +83 MB   |
| 2      | +98       | +12             | +3       | +80          | +84      |
| 3      | +112      | +14             | +14      | +80          | +83      |
| 4      | +777      | +175            | +518     | +80          | +85      |
| 5      | +785      | +175            | +524     | +80          | +86      |
| 6      | +788      | +175            | +527     | +80          | +86      |
| 7      | +797      | +175            | +535     | +80          | +87      |
| 8      | +799      | +175            | +537     | +80          | +88      |
| 9      | +807      | +175            | +543     | +80          | +88      |
| end    | +824      | +177            | +552     | +80          | +97      |

Not a leak: two steps and a plateau. The large allocations' 80 MB come in the first minute and
never move (a working set). The step of minute four (graphics +160 MB, unmapped +500 MB) is the
walk reaching new ground — the screen went from 6 677 to 7 151 chunks over the leg and 1 755
chunks were harvested — and it is item 14's staging pool: about three bytes of unmapped
staging per byte of geometry uploaded, kept after the upload. Over the last five minutes the
footprint grew 22 MB. Example: a player who walks a valley for ten minutes holds about 800 MB
more than when they landed, and holds it; they do not lose more with every step. Resident
memory grew 97 MB in ten minutes. (The soak's boarding met a race seen once in sixteen flights:
the first speed poll read the window before the realm feed had re-delivered the planet's row
after the origin swap; the boarding now waits for the row, bounded at twenty seconds.)

The long walk's detector: 300 pairs, the floor 5, 25 868 pixels past it at the rung 0→1 edge
with the widest step 111 levels — the finest handover on a long walk shows steps the minute-long
walk does not; item 5 of the list (the caster's crossfade) is the first suspect, unmeasured.

### 24.2 The flights recipe

`just flights` = the picture gate, the moving eye and the boarding storm, on release binaries,
one after the other. The rule: no slice lands before it is green. `just terrain-pictures` and
`just boarding-storm` are its parts; `VD_TERRAIN_WORKERS=<n>` sets the chunk workers' thread
count (the throughput measurement's one lever).

### 24.3 The refused first attempt — THE PENDING FLUSH (item 15, 2026-09-12)

Why every boarding took two saga attempts and ten seconds, from the code: the containment scan
decides a crossing on the LED point (the dot's position projected a request-ttl ahead, so the
destination is awake in time); the orchestrator starts the saga and asks the source to flush the
subject's pose; the source RE-VALIDATES the entry at that instant and finds the pilot still on
its way (5 m short of the hull's shell, seven ticks after the scan); it refuses, ships nothing,
and the saga waits for a flush that never comes until its freeze deadline (200 orchestrator
ticks, ten seconds), aborts, clears the latch, and the scan fires again with the pilot now
inside. The refusal was built for a fast pass-through (an arrival that crosses the destination
within a tick), and it is right for that; a walking pilot is not that.

The cure, at the source and with no new data crossing a realm boundary (SL6): a flush the
re-validation refused as STALE (the departure not yet true, or the entry not yet true) is KEPT,
keyed by its transfer with the tick it was asked, and tried again every tick from the next one;
it ships the tick the entry or the departure becomes true (the same `SourceFlushed`, later); it
is dropped when the crossing is no longer in flight (an abort or a commit cleared the latch),
when the saga's ttl lapses, or when the subject vanished (a fault, as before). A refusal that
was a fault (an unplaceable child, a subject this shard does not hold) is never kept. Example:
a pilot walks into a hull; the scan fires with the pilot 20 m out; the flush is refused and
kept; twelve ticks later the pilot stands inside, the kept flush ships, the saga commits — one
attempt, a quarter of a second. Three counters read it (`flush_kept`, `flush_retry_shipped`,
`flush_retry_dropped`). Two rig tests pin every arm: the kept-and-shipped path, the drop on an
abort, on the ttl, on a vanished subject, and on a disarmed ttl. The storm's number follows.

MEASURED, the first storm on the retry: 19 saga starts for 10 boardings — unchanged. The
traced storm (the kept flush's life in the planet shard's log) then showed the whole shape:

- The scan fires with the pilot 4.0 m outside the hull's shell. The flush is refused and KEPT.
- The pilot stands at 4.0 m for twenty-six ticks (the walk arrived at its aim on the delivered
  pose and cut the stick; the body glides to a stop), then 2.2 m, then inside.
- The kept flush SHIPS on the retry, 28 ticks (0.56 s) after it was kept, 0.70 s after the saga
  started. The retry works.
- Sixty milliseconds later the scan fires AGAIN, attempt 1: the saga was already dead.

Why the saga was dead, in two layers. First the saga's patience for the flush is the ABORT
deadline (24 orchestrator ticks, 0.48 s at the dev cluster's rate) and the flush shipped at
0.70 s. A trial that waited the request ttl instead (57 ticks) was flown and REVERTED: the storm
still read two attempts a boarding, and a patience equal to the ttl breaks the invariant that the
ttl strictly outlasts the saga (the source's re-drive would race a live saga by a tick).

Second, and the real shape: on entering the freezing state the saga tells the GATEWAY to freeze
the pilot's input at a marker. From that instant the pilot cannot walk. The twenty-six ticks at
a constant 4.0 m outside are that freeze, not a glide; the move to 2.2 m and inside is the THAW
after the abort, when the cut buffer drains at once; and attempt 1 then commits because the
pilot now stands inside. So for an OCCUPANT — a pilot whose motion is their own input — a
crossing decided while they are still outside the destination can never complete on that
attempt: the freeze holds them where the decision found them. The kept flush is right and
stays (it ships the moment the entry is true, and it is bounded), but it cannot move a frozen
pilot. The LED point (the projection a request-ttl ahead) is used for the EXTERIOR lane alone —
a hull the parent flies keeps flying under the freeze — so the occupant's decision was made on
the pilot's own swept pose, and yet the flush found the pilot 4.0 m outside seven ticks later.
Two hypotheses remained, one measurement apart: (a) the dev-control walk at 27 m/s passes
through the hull's box before the freeze lands (a test-rig artefact a player at 1.4 m/s never
meets); (b) the scan and the flush measure against different centres or books. THE MEASUREMENT
(the containment scan at debug level, a kept storm): at the decision, tick 4425, the pilot's
pose in the planet's frame is the hull's own box centre to four centimetres, and the scan reads
it 2.98 m INSIDE the hull's region — a box of 12 by 6 by 40 m in the hull's own frame, so 3 m is
the smallest half-width. At the refused flush, tick 4434, the flush's conversion lands the same
pilot 18.8 m from the hull's origin and 5.7 m OUTSIDE the box. A pilot walks 0.54 m a tick at
the dev speed; nine ticks are 4.9 m, not 18.8. So (a) looked dead and (b) looked alive. The
next kept run printed, on the refusal line, the pose the flush read, the book's tick, the book's
placement of the hull and the landed position — and it reversed that reading.

**THE REFUSAL LINE'S NUMBERS (the second kept storm).** At the refused flush (tick 4431, the
book at 4431) the book's placement of the hull is IDENTICAL to the placement the scan read at
tick 4424, to the last decimal: the scan and the flush agree about where the hull is, and
hypothesis (b) is dead. What differs is THE PILOT'S OWN POSE: the scan read it at the hull's
centre; seven ticks later the flush read it 16.2 m away (1.9 m across, 7.6 m up, 14.3 m along
the hull), 4.6 m outside the box; and the flush that committed at tick 4472 read it 1.3 m from
the centre again. The pilot moved 16 m in seven ticks: 2.3 m a tick, about 115 m/s. The
"0.54 m a tick" above was the WALK leg's share, not the boarding's: the boarding walked at the
full dev stick (500 m/s, braked over the last 40 m), and the closed-loop drive steers from the
delivered pose, which lags the server by the interpolation buffer — at metres a tick the
pilot overshoots the aim by tens of metres and comes back. So (a) was right after all, with the
true speed: the pilot passed THROUGH the hull's 12 m box in five ticks, the saga froze the pilot
on the far side, and the flush refused the fast pass-through it was built to refuse. Then the
thaw let the walk return the pilot to the centre and attempt 1 committed.

**THE FIX IS THE RIG'S.** The product is right: a player walks 1.4 m/s, 0.028 m a tick, and is
0.2 m further at the flush — inside; any occupant slower than the box's width over the saga's
seven ticks (about 85 m/s for this hull) boards at the first attempt, and a faster one meets
one abort and a second attempt a second later, which is the crossing's own conduct, not a seam.
No special path for hulls or boardings and no longer deadline: the closed-loop walk gains a
STICK SHARE (`WalkTo.speed_share`, zero = the full stick and every earlier caller), the crossing
leg takes one (`cross_leg_at`), and the moving eye boards at the share its own walk leg measured
for the foot speed (`FOOT_SHARE`). Every other crossing leg walks as before. The kept flush and
its retry stay: bounded, tested, and right for any occupant that arrives within the ttl.

**THE STORM ON THE FIX (ten boardings at the foot speed, 2026-09-12):** ten boardings, TEN saga
starts, zero refused flushes, zero kept flushes; the same storm started nineteen sagas before.
The flight's other legs are unchanged and green (1 039 s in all).

(The round's coverage gate found one line the proptests hit only by a random draw — the saga's
cut-timeout arm — against the FSM's own rule that coverage must never depend on a draw; it has
its deterministic twin now, like the cancel arms.)

### 24.4 The wall's throughput — the worker count as the only change, MEASURED

The moving eye's 528 m/s leg, once with the machine's fourteen workers and once with seven
(`VD_TERRAIN_WORKERS=7`), the same flight otherwise:

| the 528 m/s leg              | 14 workers | 7 workers |
|------------------------------|------------|-----------|
| chunks built and harvested   | 399 / s    | 374 / s   |
| one build, wall time         | 17.2 ms    | 14.6 ms   |
| the workers' utilisation     | 49 %       | 78 %      |
| parent-cache waits / builds  | 3 561 / 22 549 | 3 565 / 18 624 |
| frames per second            | 30.7       | 43.3      |
| frames with a gap            | 388        | 964       |
| urgent chunks missing, worst | 178        | 218       |
| the queue's peak             | 596        | 1 095     |

Three readings. (1) At fourteen workers the workers idle half the time: the build is not the
wall at that count. The 400 chunks a second are what the eye ASKS for; both counts deliver it.
(2) At seven the queue doubles and the gap frames with it: seven is under the ask. (3) The
frame rate is HIGHER with seven workers (43 against 31): fourteen builders starve the render
thread of cores, and the pilot pays in frames. So the worker count has a middle worth finding
(ten or eleven: enough for the ask, cores left for the picture) — one more flight each, the
owner's lever. What fourteen workers cannot cure is the gap that stays: with the ask met, an
urgent chunk is missing because it was asked LATE — the lead of 60 m is a tenth of a second at
528 m/s, five frames, and a build takes one — so the next lever is the ASK's timing (a lead in
time, or the wanted set computed a buffer ahead), not more builders. Example: a hull at
528 m/s over a plain; the finest ring's chunks are built within a frame of being asked and
still arrive a few frames late, because the ask itself came when the ground was already on the
screen's edge.

**THE HARVEST CAP AS THE ONLY CHANGE (2026-09-12, chain 17).** The 528 m/s leg at fourteen
workers with the harvest cap at its default (48 chunks and 13.5 MB a frame) against the same
flight with the cap doubled (`VD_TERRAIN_HARVEST_PER_FRAME=96 VD_TERRAIN_HARVEST_BYTES=22118400`,
96 chunks and 21.6 MB a frame):

| the 528 m/s leg, 14 workers  | cap 48 / 13.5 MB | cap 96 / 21.6 MB |
|------------------------------|------------------|------------------|
| chunks built and harvested   | 386 / s          | 413 / s          |
| one build, wall time         | 17.1 ms          | 17.7 ms          |
| the workers' utilisation     | 47 %             | 52 %             |
| parent-cache hit rate        | 89 %             | 88 %             |
| frames per second            | 30.9             | 29.6             |
| frames on which the cap filled | 551 of 1 853   | 86 of 1 776      |
| frames with a gap            | 397              | 48               |
| urgent chunks missing, worst | 268              | 15               |
| the queue's peak             | 845              | 173              |
| the pop detector, widest     | 54               | 48               |

The harvest cap WAS the wall at fourteen workers: doubling it cuts the gap frames by eight in
nine (397 to 48), the worst gap from 268 urgent chunks to 15 and the queue's peak from 845 to
173, for one frame a second (30.9 to 29.6) and 27 more chunks a second harvested. The
residue — 48 frames, 15 chunks at the worst — is the late ask named above (the lead of 60 m
at 528 m/s), which no cap cures. So the lever order is settled by measurement: the harvest cap
first (a config change in `TerrainConfig`, the default owed a decision: 96 chunks and 21.6 MB
a frame cost one frame a second on this machine and buy a band that stays whole), the ask's
timing second, the worker count's middle third.

**TEN WORKERS, THE CAP AT ITS DEFAULT (the same chain):** 37.2 frames a second (against 30.9
at fourteen and 43.3 at seven), 416 chunks a second built and harvested, 15.7 ms a build, the
workers busy 65 % of the time, the cap filled on 417 of 2 233 frames, 368 frames with a gap
(42 urgent chunks at the worst), the queue's peak 306. The middle count buys six frames a
second over fourteen and keeps the ask met, but the gap frames stay (368 against 397): the cap
is the wall at ten too. So the two levers add: the cap doubled clears the gap, the count at ten
gives the frames back. The next flight is the pair together, and the defaults follow the owner's
choice between frames and a whole band on the owner's own machine.

**THE PAIR TOGETHER (ten workers AND the cap doubled, chain 18):** 38.7 frames a second, 406
chunks a second, 15.5 ms a build, the workers busy 63 %, the cap filled on ONE of 2 322 frames
— and 380 frames with a gap (45 urgent chunks at the worst), the queue's peak 331. The cap never
filled and the gap stayed. That contradicts the reading above, so the reading is withdrawn until
measured again: the four flights are ONE each, and three of them (cap 48 at fourteen, cap 48 at
ten, cap 96 at ten) sit at 368 to 397 gap frames while one (cap 96 at fourteen) sits at 48. A
single flight at 48 against three at about 380 is either the lever or the scatter, and only a
second flight of the same pair (fourteen workers, the cap doubled) tells which. That flight is
queued after the coverage gate; until it lands the wall's lever is UNMEASURED, and the earlier
"the cap IS the wall" is one sample.

**THE TWIN LANDED (chain 20, fourteen workers + the cap doubled, the second flight):** 28.9
frames a second, 421 chunks a second, 18.2 ms a build, the cap filled on 94 of 1 733 frames,
103 frames with a gap (21 urgent chunks at the worst), the queue's peak 180. So the six flights
read, gap frames at the 528 m/s leg:

| workers | cap 48 / 13.5 MB | cap 96 / 21.6 MB |
|---------|------------------|------------------|
| 14      | 397, 388 (§24.4) | 48, 103          |
| 10      | 368              | 380              |
| 7       | 964              | —                |

The reading that survives two samples: at FOURTEEN workers the doubled cap cuts the gap frames
by four to eight times and the worst gap by an order (268 → 15, 21). At TEN workers the doubled
cap changes nothing (368 → 380) although the cap never fills there: with fewer builders the
chunks are late for a different reason (each urgent chunk waits behind the queue, 306 to 331
deep, not behind the harvest), so the count and the cap are not substitutes — the count keeps
the queue short, the cap lets the finished chunks onto the screen the frame they are done. The
frames cost: 29 to 31 a second at fourteen against 37 to 39 at ten. THE OWNER'S CHOICE is now
between two measured points on this machine: fourteen workers and the doubled cap (a whole band
at 528 m/s but 30 frames a second) or ten workers (38 frames a second with 370 gap frames a
minute at 528 m/s). The ask's timing (the lead in time) is the lever that could give both, and
it is the next one to build; the defaults stay as they are until the owner picks.

**THE AVERAGE MACHINE'S SHARE, MEASURED (ruling F6, 2026-09-13):** the moving eye at THREE
workers (this Mac's share: a quarter of fourteen cores), every other setting the default:

| leg | frames/s | chunks built/s | frames with a gap | worst gap (urgent chunks) | the queue's peak |
|---|---|---|---|---|---|
| walk | 49.9 | 2 | — | — | — |
| hull 1.4 m/s | 49.1 | 112 | — | — | — |
| hull 240 m/s | 45.1 | 161 | 551 | 13 | 159 |
| hull 528 m/s | 43.2 | 195 | 2 594 (every frame) | 974 | 2 187 |
| hull turning | — | — | 908 | 108 | 1 412 |

At the share the walk and the slow hull are whole and the frames are at the ceiling; at 240 m/s
the band already breaks on one frame in two (at fourteen workers that leg was whole); at 528 m/s
every frame has a gap and the queue holds two thousand chunks — three builders make 195 chunks a
second against the eye's ask of about 400. This is the baseline the GPU chain must beat on the
average machine, and the number behind the throughput-bounded wanted set: at speed the finest
ring must not be asked for what three builders cannot deliver, so the next rung is drawn whole
and the finest fades in where it can.

**A TRANSIENT SEEN ON THE WAY (three of eight flights):** the hull's WALKING leg reports a gap
right after the boarding — 10, 51 and 76 frames, the worst sample missing 1, 293 and 1 265
urgent chunks, the queue at 2 700 to 4 100 pending, the lead under a metre. That is the
arrival's whole-band ask: the moment the pilot's origin moves into the hull, every chunk of the
planet's band around the hull is wanted at once, at a speed where the lead cannot help. It is
not new to this round (it shows on flights before and after every change here), it is bounded
(the band fills within about a second), and it is the same "ask's timing" lever as the fast
legs' residue: the wanted set computed a buffer ahead of a KNOWN crossing (the boarding's
commit is known to the client one snapshot before the origin moves). Listed with the owed
measurements; not a defect of the crossing (the pilot's own hull is drawn whole throughout —
the missing chunks are the planet's band seen through the hull's windows).

### 24.5 The caster's crossfade, MEASURED (item 3, 2026-09-12)

The change: a coarse caster's fade bands are the DRAWN rung's (the rung it casts for), and in the
shadow pass its sink scales with that rung's wholeness — full where the finer chunk stands whole,
zero at the band's end where the finer chunk has morphed onto the caster's surface. So the caster
that leaves and the drawn chunk that takes over cast the same surface, and the shadow's edge moves
with the crossfade instead of at it (`shadow_material` builds the fade from `drawn_rung`; the
prepass multiplies the sink by `whole(d)` under `LIGHT_CASTER`).

The measurement is the pop detector's widest step per rung boundary, five flights before the
change against two after it, the moving eye flown twice (the storm on item 15's fix and this chain's moving eye both carry
the change):

| leg | before (five flights) | after (two flights) |
|---|---|---|
| walk, rung 0 boundary | 12, 12, 16, 12, 12 | 16, 12, 12 |
| walk, rung 1 boundary | 23, 20, 19, 19, 19 | 11, 10, 20 |
| walk, widest of all | 23, 20, 19, 19, 19 | 16, 12, 20 |
| hull 1.4 m/s, widest | 13, 17, 16, 19, 19 | 17, 16 |
| hull 240 m/s, widest | 48, 37, 41, 53, 53 | 41, 45 |
| hull 528 m/s, widest | 50, 47, 54, 47, 47 | 49, 54 |
| hull turning, widest | 51, 50, 28, 61, 61 | 48, 53 |

The walk's rung 1 handover read ten and eleven on the first two flights after the change and
twenty on the third (the harvest-cap flight, whose walk leg never fills the cap): NO MEASURABLE
CHANGE — the detector judges every pixel that crossed a rung boundary, and the shadow's edge is
a small share of those, so a change in the shadow alone sits under this detector's floor. The
hull legs are unchanged within their own scatter (the turning leg alone ranges 28 to 61 across
five flights before the change): at 240 to 528 m/s the widest steps are not the caster's — they
sit where the harvest wall's gap frames sit (§24.4), which is item 2's lever, not the shadow's.
The still stands in report mode sit inside the report-mode scatter (§23), so the owner's frozen
pictures need no re-freeze for this change on the evidence so far; the gate's own run decides.
The change stands on its reasoning (the caster that leaves and the chunk that takes over cast
the same surface at the handover) and on "no degradation" measured; a detector that judges the
SHADOW'S edge alone (a mask on the lit-versus-shaded step, not on every rung crossing) is the
measurement it still owes, and it is listed with the owed measurements below.

### 24.6 The sibling's exact blend (item 4, 2026-09-12)

A realm that is not the origin's ancestor (a sibling hull parked beside the pilot's hull, a
moon seen from a ship in the same system) is drawn where its parent's ARC puts its parent at
the frame's instant and where the sibling's own track puts it within that parent — the same
blend the origin's chain uses, through `sample_via_parent` and `realm_pose_blended`. Before it,
a sibling was drawn at its last shipped pose under its parent's arc-blended pose, and the two
instants could differ by one snapshot. Unit tests in `vd-client` (248 green, the still-parent
and moving-parent cases) prove the blend; the in-flight measurement (a second hull in view of
the moving eye, judged by the pop detector) is owed with the storm's next extension.

## 25. ★ THE BOUNDED ASK (ruling F9 item 1, 2026-09-13)

### 25.1 The rule, and why it is one number per rung

Ruling F9 item 1: *the client measures its builders' throughput as it goes and asks for the finest
ring only as far ahead as the builders can deliver it before the ground reaches the screen; beyond
that it asks for the next rung, which stands whole, and the ladder's crossfade blends the finer
rung in as it lands.*

The measurement behind it (§24.4, F6's baseline): on this machine's SHARE of three workers, an eye
at 528 m/s asks for about 400 chunks a second and three workers build 195. Every frame of that leg
has a gap and the queue holds two thousand chunks. The unbounded ask cannot be met, so the picture
shows holes and late pops; the bounded ask trades detail the eye cannot see at speed for a picture
that stands whole.

**The form.** The bound is ONE NUMBER PER RUNG — the rung's EFFECTIVE SWITCH DISTANCE
(`vd_client::ladder_view::AskBound`). Every part of the ladder already reads a rung's switch
distance:

| what reads it | where | what the bound does to it |
|---|---|---|
| the descent's SPLIT | `Sweep::descend`, `geo.near < fade_in[1]` | a column splits into the finer rung only inside that rung's horizon |
| the rung's TERRITORY | `descend`, `geo.near < switch_m(rung)` | the rung is urgent out to its horizon, not to the tier rule's radius |
| the CROSSFADE's bands | `AskBound::fade_bands` → the three materials' uniform | the band sits on the horizon, so the handover is a crossfade and never a cut |
| the SINK's ramp | `AskBound::sink_end_m` | a rung with no finer rung under it sinks nowhere, exactly as rung 0 does |

So moving that one number moves all four together. A column of the finest rung INSIDE the horizon
is still asked at the tier rule's own rung; one BEYOND it is asked at the next rung, whose territory
now reaches in to the horizon — the parent was already wanted under the crossfade, and now it is
wanted whole. No column is ever left unasked (the invariant test asserts exactly that).

### 25.2 The arithmetic

```
  rate    = the builders' CAPACITY, chunks a second   (the worker count ÷ the mean wall time of a
                                                       build, an exponential average over 10 s;
                                                       one reading carries at most half of it)
  v       = the eye's speed through the body, m/s     (the lead's metres ÷ the buffer's seconds:
                                                       two DELIVERED poses, SL10 clause 7)
  h       = the eye's height over the surface, m
  z       = the chunks a column holds                 (the last descent's own keys ÷ columns)

  for rung L:   R_L      = HYSTERESIS_OUT · switch(L)        the distance the split admits rung L
                g(R)     = √(max(0, R² − h²))                the ground circle at slant R
                A_L      = (62 · cell(L))²                   a column's footprint
                q_L(R)   = 2 · g(R) · v · z / A_L            the chunks a second the ask adds

  walk COARSEST FIRST with a budget b = rate:
      q_L(R_L) ≤ b  ⇒  the rung keeps the tier rule's radius,  b −= q_L(R_L)
      otherwise     ⇒  the rung keeps the reach b pays for,    b = 0
  then, from the top down, every finer rung is at most HALF its coarser neighbour — the tier rule's
  own shape, so no two rungs ever land on one distance (two rungs sharing a crossfade band would
  draw a half-transparent shell).
```

Three readings follow from the shape, and each is a unit test:

- **A still stand, a walk or a strong machine never binds.** `v = 0` makes every `q` zero; a large
  `rate` covers every rung; either way `ask_bound` returns `AskBound::unbounded`, which IS the tier
  rule's own radii — so the picture gate's still stands are byte for byte what they were.
- **A ring that reaches no ground costs nothing.** An eye a kilometre up has no ground within
  869 m, so `g(R_0) = 0` and the finest ring is free. That is why the bound barely touches a hull
  at altitude and bites hardest on a low, fast pass — which is where §16.2's probe measured the
  wall.
- **The horizon follows both levers.** More builders push it out; a faster eye pulls it in.

**What the estimate is honest about.** `q_L` counts the WHOLE leading edge of a ring, where the
skyline and the horizon cull part of it, and it counts translation only, where a turn uncovers
ground too. So the bound binds a little sooner than the true ask needs. Ruling F9 puts completeness
first, and the flight is the judge.

**The hysteresis.** The horizon IN FORCE slides toward the one the measurement asks for until the
two stand within half a percent (`ASK_BOUND_HYSTERESIS` = 0.005) of each other, and then it holds.
Without it the measured rate and the measured speed would walk the crossfade bands in and out for
ever, and every material's uniform with them; half a percent of a 491 m horizon is two and a half
metres, which no eye can see, and any real move is far larger.

★ **AND IT GATES THE DISTANCE, NEVER THE STEP.** The first writing compared the horizon before a
frame's step with the horizon after it, and discarded the step when the two were within the
hysteresis. A step is `0.2678 · dt` of the horizon, so the gate refused EVERY step below about
18.7 ms a frame — above 53.6 frames a second the bound never left the tier rule's own radii and the
whole feature silently did nothing. §25.9 tells that story.

**The knob.** `VD_TERRAIN_BOUND=0` switches the bound off for the comparison flight; the product
default is ON.

### 25.3 What it cost the still stands, MEASURED (the picture gate, 2026-09-13)

The bound must not bind on a still stand: a still eye has no speed, so every `q` is zero, the
budget covers everything and `ask_bound` returns `AskBound::unbounded` — which IS the tier rule's
own radii. The picture gate is that measurement, and it ran at the whole machine's fourteen
workers, as the harness always does.


| stand | content pixels differing | widest channel step |
|---|---|---|
| ground | 0 of 635 557 | 0 |
| hill | 10 of 701 472 | 1 |
| aloft | 0 of 586 341 | 0 |
| orbit | 18 of 466 445 | 1 |
| seam | 0 of 672 789 | 0 |

Every stand sits inside the gate's OWN run-to-run noise (§23's incidental measurement: two runs of
one binary read 0 to 19 content pixels differing at a widest step of ONE), and three of the five read
zero. So the bound bound nothing on a still stand, and the owner's frozen pictures need no second
look. `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in
129.07s`.

### 25.4 THE SAWTOOTH — the first bounded flight found the speed, not the bound (2026-09-13)

The first flight of the bounded ask, at the average machine's three workers, read `the ask was
bound on 0 of 1228 samples` on the 528 m/s leg, with 968 urgent chunks missing at the worst sample.
The bound was inert, and the reason is the SPEED, not the arithmetic.

The lead eye FREEZES at the freshest DELIVERED pose and never coasts (§16.1; SL10 clause 7), so the
lead's METRES are a SAWTOOTH: zero the instant a row lands, and the whole buffer's travel just
before the next one. MEASURED (§16.2): at 240 m/s the lead reads 14 to 30 m and at 528 m/s up to
64 m, against a buffer of 0.12 s — so the sawtooth's PEAK is the true speed (64 ÷ 0.12 = 533 m/s)
and its mean is about half of it. The flight's own time course showed it plainly: on the 528 m/s
leg the eye read 236 m/s most frames and 483 m/s at the peak. A speed half the truth halves every
`q`, the budget then covered the ask on most frames, and the horizon slewed back out as fast as it
slewed in.

**The cure, and it is still a measurement.** The client HOLDS the peak: `speed_mps` is the
reading, or what is left of the last peak after this frame's share of a one-second hold
(`TerrainConfig::speed_hold_s`). The sawtooth's period is the snapshot interval (0.05 s at the
20 Hz universe tick), so one second holds twenty of its teeth, and a hull that stops reads a still
eye within a second. Nothing is extrapolated: the peak is a MAXIMUM OVER READINGS, each of them a
difference of two delivered poses.

### 25.5 THE JUDGE — the moving eye at the average machine's three workers, the bound ON and OFF
### (2026-09-13)

Two flights of the SAME binary, `VD_TERRAIN_WORKERS=3`, one leg after another, the only difference
`VD_TERRAIN_BOUND=0`. The walk leg is the gate's assertion; the hull legs are reported.

| the leg | frames/s OFF → ON | worst gap OFF → ON | frames with a gap OFF → ON | queue's peak OFF → ON | pop's widest OFF → ON | the ask bound |
|---|---|---|---|---|---|---|
| walk, 1.4 m/s | 50.1 → 49.8 | 0 → 0 | 0 → 0 | 4 → 6 | 18 → 24 | never |
| hull, 1.4 m/s | 48.8 → 47.9 | 3 695 → 0 | 709 → 0 | 6 504 → 4 | 13 → 15 | never |
| hull, 240 m/s | 45.4 → 45.4 | 11 → 11 | 436 → 478 | 132 → 181 | 46 → 36 | never |
| **hull, 528 m/s** | **43.5 → 39.1** | **981 → 449** | **2 612 → 2 348** | **2 199 → 1 611** | **48 → 47** | **on 1 150 of 1 214 samples** |
| hull, turning | 47.1 → 47.4 | 102 → 119 | 942 → 873 | 1 408 → 1 403 | 49 → 53 | on 70 of 1 213 |

The slow hull's 3 695 chunks on the OFF flight are §24.4's known BOARDING TRANSIENT (the whole
band of the planet wanted the instant the pilot's origin moves into the hull); it shows on some
flights and not on others, before and after this change, and it is not the bound's.

**The tightest horizons the 528 m/s leg reached** (metres, finest rung first):
`[436, 872, 3476, 6953, …]` — the tier rule's own are `[869, 1738, 3477, 6953, …]`, so rung 0 and
rung 1 came in by half and every rung from 2 up stood where the tier rule puts it. At 766 m over
the ground neither of those two rings holds any ground at all, so what the pilot sees is the
four-metre ring reaching in to the eye's own foot instead of the one- and two-metre rings arriving
late.

**Against ruling F9's own bar, read honestly:**

- ✅ **THE BAND'S COMPLETENESS AT THE DRAWN RUNG IMPROVES AT 528 m/s** — the worst sample's gap
  falls from 981 urgent chunks to 449 and the queue's peak from 2 199 to 1 611. With the bound in
  force the ask no longer HOLDS a finer chunk the picture is not waiting for, so this count IS the
  drawn rung's, which is the number the owner ruled on.
- ✅ **THE POP DETECTOR'S WIDEST STEP DOES NOT RISE** at 528 m/s (48 → 47), nor on the walk's own
  terms (the walk and the slow hull move by a few levels inside their own scatter). The leg's
  historical range over seven earlier flights is 47 to 54 (§24.5), and both readings sit in it.
- 🟥 **THE FRAMES FALL AT 528 m/s: 43.5 → 39.1**, and the same leg read 38.7 and 38.9 on two
  earlier flights of the bound, so the fall is real and not scatter. **The cause is UNMEASURED.**
  The chunks DRAWN are the same (5 945 against 5 993 at the same moment of the leg), the chunks
  BUILT are the same (195 a second), and the render passes report zero milliseconds on this GPU, so
  the frame's anatomy could not be read. What was tried and did NOT recover the frames: rewriting
  the crossfade materials only when a horizon moves a twentieth (`ASK_BOUND_REBIND`), and widening
  the throughput's window from three seconds to ten. The next probe is the lane's own request and
  cancel counts: the horizon still wanders, and a finest-ring chunk asked and cancelled frame after
  frame costs the queue's lock, not the picture.
- 🟥 **THE 240 m/s LEG DOES NOT REACH EVERY FRAME** (478 frames with a gap, 11 urgent chunks at the
  worst). The bound is INERT there — 160 chunks a second built against an ask the arithmetic puts
  near 100 — so this residue is the ASK'S TIMING, which §24.4 already named and which no bound
  cures: the chunk is built within a frame of being asked and the ask itself came late.


### 25.6 What the bounded ask does NOT cure, and what it still owes

- **A TURN is not in it.** The ask rate counts the eye's TRANSLATION only; a hull that turns on the
  spot uncovers ground too, and the turning leg's gap is untouched by this bound. The lead itself
  carries no rotation either (`DEFERRED` D-TERRAIN-5 item 11), so the two owe one measurement
  together.
- **THE ARRIVAL'S WHOLE-BAND ASK** (§24.4's transient): the moment a pilot's origin moves into a
  hull, every chunk of the planet's band around the hull is wanted at once. A bound sized from a
  steady ask cannot see that coming; the lever named there — the wanted set computed a buffer ahead
  of a KNOWN crossing — is still the right one.
- **THE ESTIMATE'S OWN SLACK.** `q_L` counts a ring's whole leading edge where the skyline and the
  horizon cull part of it. A measured ask rate (the keys that ENTER the wanted set per second, per
  rung) would replace the geometry with a measurement, but it feeds back on itself — once the bound
  binds, the measured ask is the BOUNDED one — so it needs an unbounded counter to divide by, and
  that is a second descent. Left as it stands until a flight says the slack matters.

### 25.7 ★ THE FRAME BAR — the cause MEASURED, and the cure (2026-09-14)

§25.5 left the frame rate red at 528 m/s (43.5 with the bound off, 39.1 with it on) and the cause
UNMEASURED. The GPU reports zero milliseconds for every render pass on this machine and the drawn
and built chunk counts were equal, so the time had to be the MAIN THREAD's own. The flight now
carries a wall-clock timer around each of the pieces the bounded ask added — the builders'
throughput read, the bound's arithmetic (the speed hold, `ask_bound`, the slew, the two hysteresis
tests), the wanted set's DESCENT, and the crossfade materials' rewrite — and prints them per leg as
`THE FRAME'S WORK`.

**THE MEASUREMENT, the 528 m/s leg, two flights of one binary:**

| the piece | bound OFF | bound ON |
|---|---|---|
| the throughput read | 0.000 ms a frame | 0.000 ms |
| the bound's arithmetic | 0.002 ms | 0.003 ms |
| the materials' rewrite | 0.000 ms, ran 0 times | 0.000 ms, ran 187 times |
| **the DESCENT** | **9.723 ms a frame**, ran 7 721 times over 2 624 frames | **18.117 ms a frame**, ran 8 390 times over 2 339 frames |
| frames a second | 43.7 | 39.1 |

**The bound's own arithmetic is free.** Every millisecond is the DESCENT, and it is dearer two ways:
it ran 8 390 times against 7 721 (+9 %), and each run cost 5.05 ms against 3.30 ms (+53 %). The
horizon SLIDES every frame, and a slid horizon forced a descent for EVERY body in the window — the
planet's own descent runs every frame at 528 m/s anyway (the eye moves 13 m a frame), so the extra
runs are the OTHER bodies, whose reaches are vast and whose descents are the expensive ones.

**THE CURE, in three steps, each measured.**

1. **THE DESCENT AND THE DRAWN BANDS MOVE AT DIFFERENT RATES** (the coordinator's hypothesis, and
   the measurement's). The drawn bands still slide every frame — a jumped band is a pop — but the
   descent re-runs only when the horizon has left the ring it last asked for (`ASK_BOUND_BRACKET`,
   a tenth), and the descent's own ask is WIDENED (`AskBound::with_slack`), so what the picture
   draws always lies inside what the descent asked for. **MEASURED: 39.1 → 41.8 frames a second,
   the descent 18.1 → 11.5 ms a frame.**

   ★ **AND THE WIDENING IS A DERIVATION, NOT THE BRACKET** (found by the adversarial review, cured
   2026-09-14). The bracket alone does not cover the case: the picture and the descent read the
   horizon at TWO tolerances — the materials follow within `ASK_BOUND_REBIND` (a twentieth), the
   descent within `ASK_BOUND_BRACKET` (a tenth) — so at the worst stand of both at once the drawn
   base stands `1 / ((1 − 0.05)(1 − 0.10)) = 1.170` times the asked base, against an asked outer
   edge of 1.10. The picture could draw a band the descent never asked for. `ASK_BOUND_SLACK` is
   now computed from the two (0.17), asserted against them at COMPILE TIME, and the unit test walks
   the worst stand at every rung and asserts that the bracket alone falls short.
2. **A RATE LIMIT ON TOP CHANGED NOTHING** (half a second per body): **40.1 frames a second**, worse
   than the bracket alone, because the widened bracket it came with cost more than the crossings it
   saved. The crossings were never the cost; the guard stays in the config at zero as the lever it
   is.
3. **THE SLACK GOES ONLY WHERE THE HORIZON MOVES.** A rung the bound left at the tier rule's own
   radius never slides, so widening it buys nothing and costs the descent every column of the
   widening — and the coarse rungs, whose rings are the widest of all, never move. **MEASURED:
   45.0 frames a second and the descent 8.565 ms a frame — BELOW the unbounded flight's own 9.723,
   because the bounded ladder visits fewer columns once it is not paying for a widening it does not
   need.**

**And the slew slowed with it**, from half a length a second to a quarter: the band's traversal is
then about 0.8 s, softer than before and not harsher — and it halves how often the bracket is
crossed.

### 25.8 ★ THE JUDGE, RE-FLOWN ON THE CURE — the frames are back (2026-09-14)

Two flights of ONE binary, `VD_TERRAIN_WORKERS=3`, the only difference `VD_TERRAIN_BOUND=0`.

| the leg | frames/s OFF → ON | worst gap OFF → ON | frames with a gap OFF → ON | queue's peak OFF → ON | pop's widest OFF → ON | the ask bound |
|---|---|---|---|---|---|---|
| walk, 1.4 m/s | 50.0 → 50.1 | 0 → 0 | 0 → 0 | 5 → 5 | 20 → 20 | never |
| hull, 1.4 m/s | 48.1 → 48.0 | 0 → 0 | 0 → 0 | 4 → 3 | 21 → 17 | never |
| hull, 240 m/s | 45.5 → 45.9 | 11 → 14 | 449 → 540 | 160 → 177 | 48 → 36 | never |
| **hull, 528 m/s** | **43.8 → 45.0** | **974 → 401** | 2 625 → 2 695 | **2 189 → 1 768** | 29 → 43 | on 1 222 of 1 222 |
| hull, turning | 47.4 → 47.0 | 129 → 108 | 884 → 778 | 1 439 → 1 496 | 54 → 44 | on 1 200 of 1 200 |

**THE FRAME'S WORK on the 528 m/s leg**, the same pair: the throughput read and the materials'
rewrite cost 0.000 ms a frame either way, the bound's arithmetic 0.002 against 0.003 ms, and the
DESCENT **9.805 ms a frame with the bound off against 8.565 ms with it on** — the bounded ladder's
descent is now CHEAPER than the unbounded one, because it visits fewer columns and pays for a
widening only where a horizon actually moves.

**Against ruling F9's own bar:**

- ✅ **THE FRAMES ARE BACK.** 45.0 against 43.8 on the same binary — the bound now runs FASTER than
  the unbounded ask at 528 m/s, and inside the OFF flights' own spread of 43.2 to 45.1 across five
  flights of this leg. Every other leg is within a tenth of a frame of its unbounded pair.
- ✅ **THE COMPLETENESS GAIN SURVIVED AND GREW.** The worst sample's gap at the DRAWN rung falls
  from 974 urgent chunks to **401** (it was 449 before the cure) and the queue's peak from 2 189 to
  **1 768**.
- 🟨 **THE POP DETECTOR is mixed, and one pair does not settle it.** Four legs of five improve or
  hold (240 m/s 48 → 36, turning 54 → 44, the slow hull 21 → 17, the walk 20 → 20); the 528 m/s leg
  reads 43 against this pair's OFF value of **29**. That 29 is the LOWEST reading that leg has ever
  produced — the eight readings of it are 54, 50, 48, 48, 47, 43, 35 and 29 — and 43 sits below its
  own median. The honest statement is that the detector's scatter on this leg is wider than the
  difference, and a second pair is owed.
- 🟥 **THE 240 m/s LEG is unchanged and still short of every frame** (449 → 540 frames with a gap,
  11 → 14 urgent at the worst). The bound is INERT there — 0 of 1 227 samples — so this residue is
  the ask's TIMING (§24.4), which no bound cures.

### 25.9 ★ THE BLOCKER — the bounded ask never bound above 54 frames a second (2026-09-14)

An adversarial review of the whole slice found a fault that every flight had hidden, and it is the
most useful thing in this section: **the feature did nothing on a fast machine, and no test could
have said so.**

**WHAT WAS WRONG.** The horizon in force slides toward the one the measurement asks for. The step it
may take in one frame is `(horizon + column width) · 0.25 · dt` — proportional to THE FRAME'S OWN
SECONDS. The first writing then asked "is the step worth taking?" by comparing the horizon before
the step with the horizon after it, and discarding the step when the two stood within the hysteresis
(half a percent) of each other:

```
    let next = self.bound_target.slewed_toward(target, rungs, dt_s);
    if self.bound_target.same_as(&next, rungs, fraction) { return; }   // ← the fault
    self.bound_target = next;
```

A step of `0.2678 · dt` relative is smaller than half a percent whenever `dt < 0.0187 s`. **Above
53.6 frames a second EVERY step was refused, at every rung, for ever** — the horizon never left the
tier rule's own radii, `AskBound::unbounded` was what the descent read, and the bounded ask was a
no-op. It is not a slow convergence; it is a cliff with nothing on the other side of it.

**WHY NOTHING CAUGHT IT.** The unit tests exercised `slewed_toward` (the step) and `ask_bound` (the
arithmetic), both of which were correct. The GATE between them lived in the render crate, which is
Tier-B and whose only tests are the workers'. And the judge flights ran at 39 to 47 frames a second
on this machine — under the cliff, by luck. A fast machine, the one the player is most likely to
have, would have flown the whole slice and measured nothing.

**THE FIX.** The hysteresis gates THE DISTANCE FROM THE TARGET, never the step. The horizon slides
every frame until it stands within half a percent of what the measurement asks for, and then it
holds:

```
    if self.held.same_as(want, rungs, hysteresis) { return; }          // arrived: hold
    self.held = self.held.slewed_toward(want, rungs, dt_s);
```

The arrival then takes the same WALL TIME at any frame rate, which is what a rate means. Two unit
tests assert exactly that: the horizon arrives inside five seconds at 144 frames a second AND at 30,
and the two times agree within a fifth.

**AND THE REAL CURE IS WHERE THE CODE NOW LIVES.** The gate was in Tier-B because the pace had grown
there piece by piece — the slew, the descent's bracket, the materials' rebind, the throughput's
smoothing, the speed's hold. None of them is renderer work; all of them are arithmetic with two
arms. They are now `vd_client::ask_pace` (`AskPace`, `PeakHold`, `Throughput`, `FrameClock`) in the
Tier-A library at 100 % region and branch coverage, and the render crate only wires them. Three more
defects fell out of the move, each now a test:

- **THE SPEED'S "PEAK HOLD" WAS A DECAY.** It kept a fraction of the last peak each frame, so a hull
  that stopped dead from 528 m/s still read **194 m/s a second later** and the bound went on
  coarsening ground the pilot stood still on. `PeakHold` is a true maximum over a window of eight
  slots: the hull that stops reads zero.
- **THE DESCENT'S SLACK DID NOT COVER THE DRAWN BAND** — the derivation above (§25.7), now a
  compile-time assertion.
- **A LONG IDLE THREW THE CAPACITY'S WINDOW AWAY.** The smoothing weighed a reading by the share of
  the ten-second window it covered, and a reading after a minute's gap covered all of it: the
  average collapsed onto one sample of one frame's luck. A reading now carries at most half
  (`THROUGHPUT_ALPHA_MAX`).

**AND TWO SMALLER ONES.** `VD_TERRAIN_BOUND=false` used to mean ON, because only the exact string
`0` switched a knob off; every terrain switch now reads `0`, `false`, `off` and `no` in any case.
And the flight's frame-anatomy line printed the worst frame OF THE WHOLE RUN against each leg's own
mean; the stamp's peak now rolls over one second, the flight samples every 40 ms, so the largest
sample of a leg IS that leg's worst frame.

### 25.10 ★ THE SECOND PAIR, ON THE CURED CODE — and the pop question is answered (2026-09-14)

The review's blocker (§25.9) changed the shipped behaviour: the horizon now slides at any frame rate,
the slack is derived, the speed hold is a true maximum and the descent asks a ring that is wider only
where the horizon moves. So the judge was flown again — one binary, `VD_TERRAIN_WORKERS=3`, the only
difference `VD_TERRAIN_BOUND=0`.

| the leg | frames/s ON → OFF | worst gap ON → OFF | frames with a gap | queue's peak | pop's widest ON → OFF |
|---|---|---|---|---|---|
| walk, 1.4 m/s | 50.1 → 50.0 | 0 → 0 | 0 → 0 | — | 17 → 18 |
| hull, 1.4 m/s | 48.0 → 49.0 | **0** → 3 710 | 0 → 714 | — → 6 519 | 28 → 11 |
| hull, 240 m/s | 45.9 → 45.2 | 15 → 9 | 1 027 → 576 | 294 → 160 | **40 → 47** |
| **hull, 528 m/s** | **45.6 → 43.3** | **207 → 972** | 2 613 → 2 600 | **1 512 → 2 194** | **37 → 40** |
| hull, turning | 47.7 → 47.1 | **0** → 102 | **0** → 964 | — → 1 409 | **44 → 48** |

**THE FRAME'S WORK on the 528 m/s leg**, the same pair, now with a PER-LEG worst frame (the stamp's
peak rolls over one second and the flight samples every 40 ms, so the largest sample of a leg is that
leg's own worst frame): the throughput read and the materials' rewrite cost 0.000 ms a frame either
way, the bound's arithmetic 0.002 ms, and the DESCENT **7.695 ms a frame with the bound ON against
10.345 with it OFF**, its worst frame **33.4 ms against 36.7**. The bounded descent ran 6 628 times
over 2 736 frames; the unbounded one 8 099 times over 2 600.

**Against ruling F9's own bar — every line is green now.**

- ✅ **THE FRAMES.** 45.6 against 43.3 at 528 m/s: the bounded ask is FASTER than the unbounded one,
  and the margin grew after the cure (45.0 against 43.8 last round). Every other leg is within a
  frame of its pair.
- ✅ **THE COMPLETENESS.** The worst sample's gap at the DRAWN rung falls from 972 urgent chunks to
  **207** — better than the 401 of the first cured round and the 449 of the round before it — and the
  queue's peak from 2 194 to **1 512**. ★ The TURNING leg now **holds the band on every frame** with
  the bound on, against 102 urgent chunks at the worst and 964 frames with a gap without it.
- ✅ **THE POP IS ANSWERED, and it improves.** The 528 m/s leg reads **37 against 40**, the 240 m/s
  leg **40 against 47**, the turning leg **44 against 48**, the walk 17 against 18. Only the slow
  hull reads higher (28 against 11), and that leg's OFF reading came with a 3 710-chunk gap and a
  6 519-deep queue — the OFF run entered it still filling the ring from the walk (its workers ran 112
  jobs a second against the bounded run's 4), so its picture was coarse for a different reason. §25.8
  left this owed on a single pair whose OFF value was the lowest that leg had ever produced; the
  second pair settles it: **the bound does not widen the pop.**
- 🟨 **THE 240 m/s LEG'S TIMING RESIDUE STANDS** (1 027 frames with a gap against 576), and the bound
  binds there now that the speed is read honestly. It is the ask's TIMING (§24.4), and its worst gap
  is 15 chunks — a fifteenth of the 528 leg's.

## 26. ★ THE CARD AS A BUDGETED SECOND BUILDER (ruling F9 item 2, 2026-09-14)

Ruling F9 item 2 wires the GPU chain — built and proven byte for byte in Steps 12 and 13 — into the
client BESIDE the CPU share, taking chunks from the same wanted list under a fixed slice of each
frame, so the card keeps the rest of the frame to DRAW. The ruling's own order is: prove the two
client seams first, then build the builder, then judge it on the three-worker flight.

★ **THIS SECTION IS THE SECOND WRITING.** The first builder was measured, found wanting on the
judge's own leg, and then REVIEWED; the review named twelve items and two likely causes, and both
causes are now measured and cured. What the first writing got wrong is kept below, because the two
wrong numbers are the whole lesson.

### 26.1 THE TWO SEAMS, MEASURED FIRST (`just gpu-seam`)

Two things stood unproven, and both are wgpu questions rather than game questions. The renderer's
device is ONE object and the card is ONE queue, so a worker that waits on the device can wait on the
renderer's own frames.

1. A WORKER THREAD SUBMITTING COMPUTE to the renderer's own device while the renderer draws.
2. A WORKER'S `device.poll(wait)` while the renderer keeps submitting frames.

**The instrument** (`crates/bins/tests/gpu_seam.rs`, `vd_client_render::gpu_check::spawn_seam_probe`).
One capture client on the real cluster, in TWO PHASES on one binary: for fifteen seconds nothing but
the renderer touches the card, then a worker thread runs THE BUILDER'S OWN GEAR — the same pooled
buffers, the same timestamps, the same read back — over the eight golden boxes for the rest of the
run. THE PROBE ITSELF states the renderer's frames, because it is the only thing on the right thread
at the right moment (`vd_client_render::terrain::FrameMeter`).

★ **THE FIRST QUIET WINDOW IS DROPPED BY NAME** (review item 5). The client's own start is inside it
— the near ladder is uploading and the shaders are compiling — and its worst frame reads 76.6 ms
against a steady 20.8. The gate drops exactly one window, says so, and reports what it dropped.

**MEASURED on this machine (Apple M4 Pro; the capture client's own sixty-a-second loop):**

| | frames a second | the worst single frame | the card's own work |
|---|---|---|---|
| the client's own start (1 window, DROPPED) | 50.6 | 76.6 ms | — |
| the renderer alone (6 windows) | **51.5** | **20.8 ms** | — |
| a worker building on the same device (9 windows) | **52.2** | **20.9 ms** | 549 boxes a second, **0.08 ms a box by the device's own clock**, the worst box 6.57 ms of wall time, **0 stalls** |

✅ **BOTH SEAMS HOLD.** The frames held 101.4 % of the quiet rate and the worst single frame is the
SAME 20.9 ms against 20.8 — inside the quiet phase's own spread, which the gate now ASSERTS (review
item 11: the busy phase's worst frame must stand within a fifth of the quiet phase's). No submit and
no poll stalled: the longest box took 6.57 ms against a deadline of half a second, over 9 936 boxes.

★★ **AND THE FIRST WRITING'S CENTRAL NUMBER WAS WRONG.** It said the card's box costs 1.83 ms, and
it rationed the budget by that. It was measuring THE ROUND TRIP — the submit, the wait, the map and
the read back — with the host's wall clock, because no timestamp query existed. The device's own
clock says **0.08 ms**: the card's three compute passes are twenty times cheaper than the trip that
carries their answer home. So the budget was rationing the WRONG SECONDS, and the bench's old
reading of "the card is worth about two of this machine's cores" was a reading of LATENCY, not of
the card's arithmetic.

### 26.2 THE BUILDER — one queue, one priority, two stages

**One queue, one priority, two builders.** `take_job` is the ONE rule by which a builder takes work:
the job at a given place in the wanted set's own priority order, waiting while there is none,
leaving when the workers close. The CPU workers take the first; the card takes the one after them
(§26.4's head-of-line rule). ★ A defect fell straight out of that: the queue's wake was
`notify_one`, and a wake that happened to reach the card — which cannot take a queue of one — left
the CPU workers asleep beside a job they could have taken. MEASURED: the walk's settle hung for ever
on ONE pending chunk. Two kinds of waiter on one queue means every wake is a BROADCAST.

**Two stages, because the budget must ration the CARD** (review item 1c):

- THE CARD THREAD plans a box's topology, uploads it, submits the three passes, waits and reads the
  bytes back. Nothing else — no decode, no mesh.
- THE GEOMETRY THREAD decodes those bytes into the box (`BoxPlan::box_of`) and runs the SAME
  geometry step the CPU workers run (`vd_client::chunks::geometry_from`), then sends the chunk down
  the same done channel to the same harvest.

**The gear** (`vd_client_render::gpu_check::BoxGear`, review items 1a and 1b): fourteen pooled
buffers and three bind groups, GROWN and never rebuilt, written through ONE reused byte vector — the
steady state of a box allocates nothing — and a query set that reads THE CARD'S OWN SECONDS from the
device where the device offers timestamps (this Mac does; the stamp states which clock measured).

**The time budget** (`vd_client::card_budget::CardBudget`, Tier-A, 100 % covered). Every frame GRANTS
the card a share of its own seconds (`VD_TERRAIN_GPU_BUDGET`, a quarter by default); the card SPENDS
its measured device time before it dispatches. The allowance never holds more than one frame's grant
(or one box, where the grant is smaller), so a quiet second cannot be banked into a burst.

★ **AND THE CAPACITY IS THE RATE THE CAP ALLOWS, not the ratio** (review item 2). The naive
`fraction / per_box` overstates by up to a factor of two, because the anti-banking cap throws away
every grant past one frame's worth: a frame that grants two and a half boxes buys TWO, and a frame
that grants six tenths of a box buys ONE box every two frames. The formula is derived, and a
thousand-frame simulation in the unit tests checks the arithmetic against what the budget really
delivers, on both sides of the cap.

**Every failure is an answer** (review item 4): a poll or a map that fails detaches the card — the
budget states zero capacity from that moment, the job goes back to the queue for a CPU worker, the
client goes on drawing, and the failure is told once.

### 26.3 THE FLIGHTS — six runs of one binary, three workers

| the 528 m/s leg | the card OFF | quarter, capacity SUMMED, skip 1 | quarter, NOT summed, skip 1 (THE RULE) | quarter, NOT summed, skip 0 | quarter, NOT summed, skip 3 | THE WHOLE FRAME |
|---|---|---|---|---|---|---|
| frames a second | 45.8 | 43.3 | **45.7** | 45.8 | 45.6 | 46.0 |
| the band's worst gap | 125 | 442 | **87** | 113 | 97 | 84 |
| the queue's peak | 1 442 | 1 900 | **529** | 615 | 501 | 486 |
| the pop's widest step | 44 | 38 | 46 | — | 53 | 41 |
| the engine harvested | 204 chunks/s | 250 | **256** | — | 252 | 257 |
| the card built | — | 84 boxes/s (33 %) | **95 boxes/s (36 %)** | 91 (35 %) | 92 (35 %) | 95 |
| the card's own box | — | 0.10 ms | 0.10 ms | 0.10 ms | 0.47 ms | 0.10 ms |

**EVERY LEG, the card OFF against THE RULE, and against the worker-count rule:**

| the leg | frames/s OFF → skip 1 → skip 3 | the band's worst gap OFF → skip 1 → skip 3 |
|---|---|---|
| walk, 1.4 m/s | 48.2 → 49.1 → 47.6 | 0 → 0 → 0 |
| hull, 1.4 m/s | 53.4 → 48.1 → 53.5 | 3 714 → **0 (the band HELD on every frame)** → 3 394 |
| hull, 240 m/s | 45.9 → 45.0 → 45.0 | 10 → 10 → 0 |
| hull, 528 m/s | 45.8 → **45.7** → 45.6 | 125 → **87** → 97 |
| hull, turning | 46.4 → 47.2 → 47.8 | 14 → 50 → 58 |

★ A card that skips THREE builds NOTHING on the walk (0 boxes over sixty seconds), because that
leg's queue is never four deep — which is the whole reading of the skip-3 column: that card stands
down where the queue is short, and the leg then reads what NO CARD reads.

### 26.4 THE TWO CAUSES, ISOLATED — and the rule the measurement picks

The first writing named ONE cause and could not separate it from a second. Three ablation flights
separate them, and BOTH turn out to matter.

★ **CAUSE ONE: THE BOUND SPENDS THE CARD'S CHUNKS ON A WIDER RING.** Summed into the bounded ask,
the card's capacity takes the builders' reading from 177–219 chunks a second to 222–445 and pulls the
finest ring's horizon out from 405 m to 569 m — and the band then goes MORE incomplete, not less
(442 urgent chunks against 125 with no card at all), the queue deeper (1 900 against 1 442) and the
frames slower (43.3 against 45.8). NOT summed, the very same card, the very same budget and the very
same chunks read **87 urgent chunks, a queue of 529 and 45.7 frames a second**. The card's chunks are
worth more spent on the ask the eye already has than on a wider one. **So the capacity is measured,
stated on the stamp and NOT summed** (`VD_TERRAIN_GPU_BOUND=1` sums it).

★ **CAUSE TWO: HEAD-OF-LINE BLOCKING IS REAL, AND SMALLER.** With the capacity out of the bound
either way, a card that takes the SINGLE most urgent request reads 113 urgent chunks at 528 m/s and
65 at 240 m/s; a card that leaves that one request to the CPU workers reads **87 and 10**. The card
holds a chunk for a round trip where a worker finishes it in arithmetic, so the most urgent request
is exactly the one it must not take. **So the card skips one** (`VD_TERRAIN_GPU_SKIP`).

★ **AND SKIPPING MORE IS WORSE, MEASURED.** A card that skips THE WORKER COUNT was flown, because
such a card stands down wherever the CPU share can empty the queue by itself — the cure the still
stand asks for. It reads 97 urgent chunks at 528 m/s against 87, 58 turning against 50, and on the
SLOW HULL's leg it reads **3 394 urgent chunks against a band that held on every frame**: it never
enters that leg's queue, so the leg reads what no card at all reads. It is the better rule on one
reading only — the queue's peak, 501 against 529. **So the shipped rule is ONE.**

**AGAINST RULING F9's OWN BARS, with the rule in force:**

- ✅ **THE WORST GAP FALLS**, 125 → **87** (and far below Step 14's 207).
- ✅ **THE QUEUE FALLS**, 1 442 → **529** (far below Step 14's 1 512).
- ✅ **THE SLOW HULL'S BAND NOW HOLDS ON EVERY FRAME**, against 3 714 urgent chunks and a 6 523-deep
  queue without the card.
- 🟨 **THE FRAMES: 45.7 against 45.8.** A tenth of a frame, where this leg's own eight readings span
  43.3 to 45.8. Inside the spread, not above it.
- 🟨 **THE POP: 46 against 44.** Two levels, where this leg's readings span 29 to 54. Inside the
  spread.
- 🟥 **THE TURNING LEG'S GAP RISES**, 14 → 50, on a queue of 445 against 565. It is the one leg the
  card makes worse, and it is the leg whose heading the lead never asked for.

★ **AND THEN THE PICTURE GATE MEASURED THE OTHER HALF OF THE ANSWER: THE CARD COSTS A STILL
STAND.** With the card building, the hill stand's terrain settles at tick **2 408** against **2 081**
with no card — past that stand's own capture tick of 2 400, so the gate is RED. The cause is the
same round trip that pays at speed: a builder that holds a chunk for 1.8 ms of submit-and-wait helps
a queue of hundreds and hurts a queue of three, and the tail of a settle is a queue of three. The
head-of-line rule was flown at THE WORKER COUNT because of it (2 446 → 2 408 ticks), which helps,
does not cure, and costs the slow hull's whole band — so the shipped rule stayed ONE.

**SO THE CARD SHIPS AS A KNOB (`VD_TERRAIN_GPU=1`), NOT AS THE DEFAULT.** It is built, proven byte
for byte, and MEASURED both ways: it pays at speed and it costs a still stand, and a shipped default
must be right for both. `VD_TERRAIN_GPU_BOUND=1` puts its capacity back into the bound and
`VD_TERRAIN_GPU_SKIP=<n>` names another head-of-line rule. ⚠ **OWED BY THE OWNER:** whether a third
of the 528 m/s leg's gap and two thirds of its queue are worth eighteen seconds of a still stand's
settle — or whether the card should simply stand down where the queue is short, which is the cure
the measurement points at and which nobody has measured yet.

### 26.5 THE STILL STANDS, AND WHAT IS OWED

`just terrain-pictures-card` is the automated run that exercises THE CARD AS A BUILDER (review item
6). What it proves it proves: the ground stand reads **0 of 635 557 content pixels differing**, so
the card's box makes the CPU's picture. ⚠ It is RED on the hill stand for the settle above, which is
why it is NOT in `just flights` — a run that measures the card, never a green light. The shipped
path (`just terrain-pictures`, the card off) is green: 0 / 10 / 7 / 18 / 0 content pixels at a widest
channel step of one.

Owed, in the order a measurement would take them:

1. **THE STILL STAND.** The card should stand down where the queue is short — the tail of a settle
   and a walk are exactly where a round trip is pure latency. ⚠ Raising the head-of-line rule is
   NOT that floor, and the flight says so: at the worker count the card also stands down on a leg
   whose queue is six thousand deep the moment the CPU workers are keeping up, and the slow hull's
   band goes from holding on every frame to 3 394 urgent chunks. A floor ON THE QUEUE'S DEPTH, by
   name, is the cure, and it is unmeasured.
2. THE TURNING LEG (14 → 50 urgent chunks). The other leg the card costs.
3. THE ROUND TRIP, now that the card's own arithmetic is known to be 0.08 ms and the trip 1.8 ms:
   two boxes in flight at once would hide the wait behind the next box's compute, and would shorten
   exactly the latency that costs the settle.
4. WHETHER A DAMPED SUM IS BETTER THAN NO SUM: the card's capacity is real, and the bound refusing
   it entirely is a blunt answer to a measured harm.
