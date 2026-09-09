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
interpolation buffer is 150 ms; at 528 m/s that is 79 m of lead, which is one chunk. If M8-1 shows the
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
| M8-1 | The chunk arrival rate during a scripted descent at 1.4, 240 and 528 m/s, against the thread budget | a hull leg through the harness, the lane's pending count as the instrument | the band is never incomplete for one frame; the deepest queue reported | owed |
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
| D8-8 | **The far-rung renderer** (owner, 2026-09-09: Enshrouded draws its far levels as CUBES, not triangles) | (A) every rung is a surface-nets MESH, as today; (B) the near rungs are meshes and the far rungs are the ladder's own VOXELS drawn as cubes or splats, one cell per pixel or more; (C) decided by a measurement in the look phase | **(C), and this slice keeps the seam OPEN for it:** the client library hands the engine a chunk's CELLS beside its mesh (it holds both today), the tier rule and the crossfade are written per column and not per triangle, and the dither works on either. Nothing in slice 8 chooses a triangle. | The far rungs ARE voxel grids already (rung L is the recipe at 2^L m cells, edits folded in); a cell drawn at one pixel is the same picture as a triangle, so the mesh's extraction (60 % of a chunk's cost) and its 254–405 KB are the price of a smoothness no far pixel shows. The choice is a LOOK decision (V13 addendum: after the freeze), and the foundation must not close it |

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
  exact cure (D-TERRAIN-5 item 8).
- Every measurement is a STILL stand (D-TERRAIN-5 item 4): the morph on a moving eye, the arrival
  of a finer rung over a sunk coarser one, and the skyline's recompute per half metre are owed to
  M8-1 and step 6.
- The skyline's cost is not measured on a moving eye: a recompute raises about 36 rays per 62 m
  column at 1 km (about 2 000 near columns from the ground; MEASURED before the refutation's fix,
  every wall walked the whole fan of 3 600). M8-2 measures it with the bytes.
- A sealed cave under the surface is never wanted from above, and a cave that opens sideways is
  drawn only with its own chunk: the column span is a surface model. The block system's own
  residency (slice 9) owns caves; registered in `DEFERRED`.
