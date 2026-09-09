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
