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
