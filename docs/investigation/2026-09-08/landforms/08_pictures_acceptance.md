# 08 — THE PICTURES AND THE ACCEPTANCE: the stamp, the ruler, the stands, and the verdicts a machine holds

**Date:** 2026-09-08. **Domain 08** of the landforms investigation. **REVISION 3**, after a second
round of refutation A (the laws and the code) and refutation B (believability, cost and the owner).
Every finding of both rounds is answered in §17.
**Status:** an investigation report for the owner. It designs the picture protocol. It decides nothing by
itself.
**Binding law read first:** CLAUDE.md (HR1–HR6, SL1–SL10), the 2026-09-07 voxel ruling
(`docs/design/owner_decisions_2026-09-07_voxels.md`: V1 SL10, V2 the nine requirements, V4 the review,
V6 the format, V8 the plant, V9 the generator, V10 the extractor and THE VISTA, V11 up is the server's,
V12 slice 7), the 2026-08-27 seed ruling, and SL8 (a seam is a defect).

**The task, in the owner's words.** *"Make sure that we reach that quality on the picture for
earth-like planets (biomes can be different of course, should be dependent on the planet position, spin,
trajectory, size and gravity, etc.). It should be very believable, as we also should simulate the
weather."* And of the three pictures slice 7 delivered: *"no orienters and no details at all, so I have
no idea how far we see or how above from the surface we are."* *"The only thing I'm worried about is
that the surface will not be interesting enough."*

Every number below is marked **MEASURED** (with how) or **ESTIMATED** (with the arithmetic) or
**UNMEASURED** (with how it would be measured). Every claim about the code cites `file:line`. Where a
number is recomputed outside the repository, the recomputation is cross-checked against a number the
repository already published, and the cross-check is shown.

---

## 0. The recommendation, in one page

A picture the owner cannot judge is a picture the slice did not deliver. This revision changes five of
the last revision's answers, because measurement refuted them.

### 0.1 What the pictures are missing

1. **THE STAMP.** Every picture carries an on-frame readout and the same numbers in its aligned state
   file: the realm, the stand's address, the eye's altitude over the surface, the geometric horizon,
   the DRAWN horizon, the radius of terrain drawn, every rung drawn with its chunk count, the sun's
   elevation and azimuth, the biome under the eye, the height over the sea, the ruler and its measured
   pixels, the world identity and the universe tick. One struct in `vd-devproto`, filled by `vd-client`
   from the delivered state and the linked generator, formatted by `vd-client` as lines, drawn by the
   renderer, and asserted by the gate. **Three consumers, one source: measured, never typed.**
2. **THE PICTURE PROBE.** Every judged capture writes a second, aligned buffer in which each pixel
   states WHAT drew it and HOW FAR it is. Every geometric verdict then reads that buffer instead of
   guessing at colour. **The colour classifier in the slice-7 gate is a fitted heuristic**
   (`crates/bins/tests/terrain_pictures.rs:279` calls a pixel ground when `r >= g >= b && r >= b + 8 &&
   r > 24`). The probe replaces it with an exact reading. The probe's pixel holds **an author AND a
   mesh instance**, because an author alone cannot refuse an impostor (§6.2).
3. **THE RULER.** A subject of stated size stands in frame, and the gate measures its painted footprint
   in pixels against a prediction from data it holds. The avatar marker is the near ruler and it is
   exact inside **145 m** (MEASURED, §5.2). It is an INSTRUMENT, not the reference picture's scale
   figure: it paints a 0.5 m ball, and a ball carries no size to a human eye (§5.4).

### 0.2 The five measurements that changed the recommendation

Each was taken for this revision, outside the repository or by reading the code, and each carries a
cross-check that could have failed.

**(M-A) The pilot does not stand on a hill. He stands on a plane tilted about one and a half degrees.**
I rebuilt the generator's noise outside the repository — the same sixteen gradients, the same quintic
fade (`crates/terrain/src/noise.rs:20-36,67-72,84-120`) — and measured **one octave's standard
deviation at 0.2701** over 150 000 points, not 1.0. I then built the whole home-planet height field from
the octave table and measured it over 4 000 directions: **the surface's standard deviation is 2 295 m,
and 0.92 % of the surface lies under the sea offset of −5 297 m.** *The cross-check:* the repository's
own MEASURED comment says the sea *"covers about one column in a hundred"*
(`crates/terrain/src/chunk.rs:851-853`), and sibling document `01_reference_target.md:262-268` replays
200 000 columns and reports **0.99 %**. The three agree. The larger sample is the better number; this
document uses its own 0.92 % where it names its own measurement and cites 0.99 % beside it.

The measured relief, on real profiles (§3.3):

```text
  view length     mean peak-to-peak   mean tilt   RMS wobble after the tilt is removed
      400 m            12.2 m           1.74 deg           0.5 m
    3 473 m            99.4 m           1.60 deg           5.5 m      <- a standing pilot's horizon
   44 839 m           968.1 m           1.15 deg          97.5 m
```

**A pilot on the home planet sees a plane tilted 1.6°, with five metres of wobble on it.** The eye reads
that as level ground. Only a new landform mechanism makes a hill; the picture protocol cannot.

**(M-B) The reference picture is a GROUND picture, and the last revision's flagship altitude was a
misreading.** The last revision inverted the horizon formula, said "50 km needs a 373 m eye", and moved
the flagship there. That drops the reference picture's own mechanism. In the reference the 50 km comes
from RELIEF: the far ridges are visible because they are TALL. A point at ground distance `d` is visible
from a 1.8 m eye when it stands above the horizon plane by `h' = (d − 3 473)² / 2R`:

```text
   distance   the height a thing needs, to be seen from a 1.8 m eye
     10 km          6.4 m
     20 km         40.8 m
     33 km        130.1 m
     50 km        323.0 m        <- the reference picture's mid ground
    100 km      1 390.3 m
    145 km      2 988.9 m        <- the reference picture's tallest ridges
```

(MEASURED arithmetic, `R = 3 350 759 m`, `crates/terrain/src/home.rs`; the reference's *"1 500–3 000 m
ridges at 30–60 km"* is `01_reference_target.md:428`.) **So a standing pilot on our own planet would see
the reference picture's ridges out to 100–145 km — if the body held any.** The flagship therefore moves
back to the pilot's eye, and beside it the acceptance prints this threshold curve against the body's own
measured prominence. **A flat world then FAILS the flagship instead of being flown over.**

**(M-C) The renderer casts NO shadows, so eight of the last revision's sixteen pictures bought almost
nothing.** Every light in the client is created with `shadows_enabled: false`: the terrain sun
(`crates/client-render/src/terrain.rs:457-459`), the terrain FILL light (`:471-473`) and the stub scene's
key light (`crates/client-render/src/lib.rs:847-848`). Worse for the argument, the fill light is aimed
from the OPPOSITE side of the sun on purpose — *"The shadow side is never pure black"*,
`terrain.rs:466-478`, `FILL_SHARE = 0.08`, `terrain.rs:56` — which flattens exactly the raking contrast
the doubling was for. On a surface whose median slope is 1.37° a low sun moves only the Lambert term. The
set drops the two-light doubling and names the shadow pass as its precondition.

**(M-D) Three verdicts used a tolerance that cannot fail, and one of them PASSED the picture the whole
document exists to refuse.** `relief_bound_m(rung)` sums the live amplitudes, and the amplitudes are
scaled so their SUM IS THE RELIEF (`crates/terrain/src/body.rs:180-186,263-271`), so on the home planet
it is **14 304.9 m at every rung that matters**. As a tolerance at 300 m of eye height that is 277 px
against a 69.4 px signal. The cure is a tolerance from the field the body actually has:

```text
  view length   3 x RMS wobble   tolerance in pixels   note
      400 m          1.5 m            3.26 px
    3 473 m         16.5 m            4.13 px          a standing pilot
   44 839 m        292.5 m            5.67 px          300 m up
  636 939 m      3 858.0 m            5.26 px          60 km up
```

**The tolerance is about four to six pixels at EVERY scale**, because the field is nearly self-similar
(the wobble and the horizon both grow with the view length). With it, the slice-7 hill picture fails by
69.4 px against 5.67 px — twelve times over — and the ground picture's 3.0 px passes, which is honest.

**(M-E) The set as specified cannot be rendered, the vertical chunk count follows the RUNG, and the
ladder has no rung for an orbit picture.** The residency price is rebuilt from the code in §11. Two
facts changed it. First, the band's own height caps the chunks per column: `surface_chunk_span` adds one
chunk of margin each way and clamps to the band's top chunk index
(`crates/terrain/src/digest.rs:89-130`), so a column holds three chunks wherever the band is three
chunks tall — **and the home planet's band is ONE chunk tall at rung 9 and above** (band ≈ 28 800 m; at
rung 9 that is 56 cells, and 56 cells is under one 62-cell chunk). Second, the residency must reach past
the geometric horizon, because high ground beyond it rises into view (§11).

```text
  stand                          chunks     mesh       triangles   14 threads
  ground, 1.8 m, horizon only     1 506     610 MB      25.0 M       1.3 s
  ground, 1.8 m, visible relief   2 805   1 137 MB      46.6 M       2.4 s
  drone, 373 m, visible relief    3 134   1 270 MB      52.1 M       2.7 s
  orbit 300 km   (rung 9)         5 752   2 331 MB      95.6 M       5.0 s
  orbit 10 000 km (rung 11)       3 277   1 328 MB      54.4 M       2.9 s
```

At 10 000 km one pixel covers 11 506 m of the body, so a two-pixel cell would be 23 012 m. **The
ladder's coarsest rung is 11, whose cell is 2 048 m — eleven times finer than the picture needs**
(`crates/seed/src/ladder.rs:15-27,109-111`; the home planet's `rungs = 12`, `crates/terrain/src/home.rs`).
There is no coarser answer to ask for, and the two reasons are structural, not oversights (§11).

### 0.3 What this domain therefore recommends

The protocol stands: a stamp, a probe, a ruler, found stands, measured bounds, a recorded approach. Six
things change:

- **The flagship picture returns to the pilot's eye at 1.8 m**, and the acceptance prints the
  visible-prominence threshold beside it (M-B). The 373 m stand stays as a DIAGNOSTIC, because it shows
  the tilt cleanly and it makes residency the dominant defect (87.6 px of missing ground, §3.2).
- **Every tolerance comes from the body's own MEASURED field, never from the aligned relief bound**
  (M-D). The measured pass that produces it is a separate call the gate makes, never part of
  `BodyDefinition::from_seed` and never part of the body's identity (§7.2).
- **The two-light doubling is dropped until a shadow pass exists** (M-C). Fourteen judged pictures and
  two recordings, not sixteen and one.
- **Stands are found by a SATISFYING predicate at a stated scale, never by an argmax, and the scan
  strides the address space with a co-prime step** so the stands spread over the body instead of
  crowding one corner of one face (§7.2).
- **Six stands are marked NOT BUILDABLE TODAY**, with their measurements: the river bend (no
  hydrology), the plateau edge (the SMOOTH height field's largest 1 m slope over 3 000 samples is
  7.23°), the coast (the sea covers 0.92 % and is not drawn), the two-body pair, the windward/leeward
  pair, and the whole weather set.
- **The owner's five facts get a table, not a sentence.** SIZE already has an acceptance, because every
  bound in §9 is a function of the body's radius. POSITION gets an ORDERING verdict over the world's own
  bodies (V13). SPIN, TRAJECTORY and GRAVITY get named owners and named verdicts (§10.2(d), §16 Q9).

**What the protocol cannot fix, and says so.** The amplitude-to-wavelength ratio falls from 0.019 to
0.008 across fourteen octaves (§3.3). A surface like that is *self-similar*: every scale looks like
every other scale, so it makes swells and never a carved valley, a sharp ridge, a cliff or a plateau.
**The one number this acceptance owes the landform domains is therefore M-B's threshold curve**: the
body must hold prominences of 323 m at 50 km and 1 390 m at 100 km, or a standing pilot sees a plain.

---

## 1. What the harness holds today — the only current truth

| Fact | Where | What it means for the protocol |
|---|---|---|
| The HUD holds six lines: a title, the phase, the realm label, the entity id, the world position, the count of drawn rows. Nothing else. | `crates/client-render/src/lib.rs:2032-2037` | This is the whole "orienter" the owner had. No altitude, no horizon, no rung, no scale. §4 replaces it. |
| The SAME HUD function draws the windowed frame and the headless capture. | `crates/client-render/src/lib.rs:2004-2007` (the doc line: "so the captured overlay is byte-for-byte the same HUD") | A stamp added to the HUD lands in every captured picture with no second path. |
| Every capture already writes an aligned `state/<stem>.json` dump of the whole `DevState`, and a manifest row that names it. **A recording writes one per FRAME.** | `crates/bins/src/bin/client.rs:1231-1240,1332-1372,1391`; `crates/client-harness/src/capture.rs:87-93` (`state_rel_for`) | The picture's record already exists, so a stamp rides into it for free. And a 1 545-frame recording writes 1 545 PNGs AND 1 545 pretty-printed JSON dumps, which §8.2 now prices. |
| **Every light in the client has `shadows_enabled: false`, and a FILL light is aimed from the far side of the sun on purpose.** | `crates/client-render/src/terrain.rs:457-459,466-478`, `:56` (`FILL_SHARE = 0.08`); `crates/client-render/src/lib.rs:847-848` | **There are no cast shadows.** A low sun buys a Lambert term only. §7.4 drops the two-light doubling and names the shadow pass as a precondition. |
| `DevState` carries `terrain_chunks_drawn` and `terrain_chunks_pending`, and nothing else about the terrain. | `crates/devproto/src/state.rs:192,196`; `crates/devproto/src/predicate.rs:32-36` | The gate can wait for the terrain to settle. It cannot read the altitude, the rung, the drawn radius or the biome. |
| The renderer computes the nearest and the farthest placed chunk and prints them to a debug log every 60 frames. **The loop has no frustum test and no facing test**, so a chunk behind the pilot counts. | `crates/client-render/src/terrain.rs:522-546,556-566` | `farthest` is the RESIDENCY radius, not the picture's reach. §9 V2 reads the probe's distance buffer instead. The counter is still worth promoting, under its true name. |
| **The client holds no biome and no paint table.** The whole terrain is one material: `base_color: Color::srgb(0.55, 0.50, 0.42)`. The word "biome" does not occur in `crates/client-render/src` or `crates/client/src`. | `crates/client-render/src/terrain.rs:362` (grep of both crates) | V7 (the picture shows the biome the stamp names) CANNOT RUN. It is owed, beside the atmosphere. §9 marks it so. |
| **There is no sky dome and no atmosphere.** The clear colour is deep-space black, and the star sky is a point cloud. | `crates/client-render/src/lib.rs:78` (`CLEAR_SRGB`); slice 7 §6 | An unauthored pixel is LAWFUL today wherever the sky shows. §9 V3 is written for that. |
| **The chunk lane refuses a second rung of a realm that already holds one, and counts the refusal.** | `crates/client/src/chunks.rs:28-30,335-341` (`counters.second_rung`); `docs/design/DEFERRED.md` `D-TERRAIN-3` 🟥, deleted by slice 8 | Six verdicts and both recordings are DEAD until slice 8 flips that row. §13.1 gates the work on it by name. |
| **A column holds at most as many chunks as the BAND is tall.** `surface_chunk_span` takes five heights, adds one chunk of margin each way, and clamps to the band's top chunk index. | `crates/terrain/src/digest.rs:89-130`; `crates/seed/src/ladder.rs:114-118` (`cells_in_band`) | Three chunks per column at rungs 0–7, two at rung 8, **one at rungs 9 and above** on the home planet (band ≈ 28 800 m). §11 prices every rung with its own vertical count. |
| The slice-7 gate decides "this pixel is ground" by colour, outside a HUD corner mask of `x < 320 && y < 180`. | `crates/bins/tests/terrain_pictures.rs:266-289` | A fitted heuristic, and a mask a ten-line stamp does not fit inside. §6 replaces the classifier; §13.1 puts the probe BEFORE the stamp so the gate is never red in between. |
| The straddle discipline exists: a screenshot is bracketed by two state polls and retried until they agree, with the subject's own apparent radius as the tolerance. | `crates/bins/src/pixel.rs:1-24,204` | Every geometric verdict in §9 inherits it. No verdict may compare a frame with a state poll taken at another moment. |
| The pixel instrument reports, per watched realm, its author (`SelfLook` or `ParentMarker`), its drawn centre, its projected rectangle and its footprint radius in pixels — **from `extent_m` alone, as a bounding sphere**. | `crates/bins/src/pixel.rs:47-95,148,158-183` | It cannot predict a box. §5 states what the diagnosis row must gain before a hull can be a ruler. |
| `DevRealmBox.extent_m` is *"a sphere's radius; a box's half-diagonal length"*, produced by `shape_extent_m = half.length()`. The row carries **no half-extents and no facing**. | `crates/devproto/src/state.rs:63`; `crates/client/src/realm_scene.rs:532-541` | The default hull's streamed extent is `√(6² + 3² + 20²) = 21.095 m`, not 20 m and not 40 m. |
| A realm's look draws as an ORIENTED CUBOID, scaled by its three half-extents and turned by the facing its parent authored — and **translucent** (`AlphaMode::Blend`). | `crates/client/src/realm_scene.rs:831-843`; `crates/client-render/src/lib.rs:1823,1849-1856` | A hull's painted width swings by a factor of 3.3 with its facing, and a blended surface has no single author per pixel. §5 answers both. |
| The marker sprite has a **3-pixel apparent-size floor**, applied by ONE function the renderer and the gates share, over a 0.5 m figure extent. | `crates/client-harness/src/camera.rs:254,264-266`; `crates/core/src/look.rs:100` | **The avatar marker lies about its size past 144.9 m** (MEASURED arithmetic, §5.2). It is an exact instrument inside that, a presence dot beyond it, **and a ball at every distance** — never a human silhouette. |
| The reference camera is 45° vertical over 720 rows; the window is 1280 × 720. | `crates/core/src/geometry.rs:1160`; `crates/client-render/src/lib.rs:71-72` | One pixel is 0.06592° (MEASURED arithmetic). Every angular verdict derives from this pair, never fitted. **Sibling document 01 computes with 1 080 rows**, so its pixel angle is 1.5× smaller and its cost tables are not directly comparable (§11). |
| The harness holds pure pixel verdicts already: magenta count, content fraction, mean luminance, region non-empty, differing-pixel count, NaN count. `differing_pixel_count` is an EXACT byte-inequality count whose doc names its purpose as a STILL scene. | `crates/client-harness/src/assert.rs:35-118`, `:98-105` | It saturates on a moving camera. §9 V9 measures a magnitude instead. |
| **The float fence is a per-crate `clippy.toml`.** Ten crates carry one; `vd-client-harness` carries NONE. The fence's own text names the operations it TRUSTS on every target: *"add, subtract, multiply, divide, square root, the round-to-integral family, comparison, abs, clamp"*, and it bans every transcendental, `min`/`max` and `mul_add`. | `ls crates/*/clippy.toml`; `crates/terrain/clippy.toml:1-45` | A comparison of two fenced numbers cannot flip between chips, so the search's home is NOT an arithmetic argument (§7.2). It is a LINT argument: an unfenced crate has nothing that stops a later slice writing `.powf()`. And a predicate may not name a degree. |
| `vd-client` already depends on `vd-terrain` and `vd-seed`; `vd-client-harness` does not. | `crates/client/Cargo.toml:10-12`; `crates/client-harness/Cargo.toml` | The stamp's terrain numbers need no new crate edge. The buffer verdicts need no terrain at all. |
| `vd-client`, `vd-client-harness` and `vd-devproto` are Tier-A at 100 % region and branch. `vd-client-render` and `vd-bins` are not. | `justfile:12` | Every pure verdict and every stamp expression goes above the line; the renderer's probe pass and the gate go below it. |
| The stands are placed by the `VD_SPAWN_POSES` stand-in, **one entry per account**, which also states a facing. **The gate itself computes the surface height with `height_m` and hands it to the server as a literal.** | `crates/bins/tests/terrain_pictures.rs:457-472`; `docs/design/DEFERRED.md` `D-TERRAIN-4` 🟥 | So a seating check compares one binary with itself today (§9 V10). And every extra figure in a frame is another LOGIN, which §11 M8-7 counts. |
| **`D-TERRAIN-1` is 🟧, not 🟩.** The golden table is equal on three legs of ONE target (aarch64 macOS) plus aarch64 Linux; the x86-64 leg (G4) is **EMULATED**, and its own row says *"it can find a drift and can never prove its absence"*. | `docs/design/DEFERRED.md:7721-7745` | The only no-drift gate is itself owed a leg. §7.2 and §14 say so instead of claiming an existing cross-chip measurement. |
| The gate waits 1 800 client ticks at 20 Hz for the terrain, and a recording is capped at 3 600 frames. | `crates/bins/tests/terrain_pictures.rs:63`; `crates/bins/src/bin/client.rs:705` | The terrain deadline is **90 s**, which §11 prices every stand against. A 1 545-frame recording fits the cap and fills the disk (§8.2). |
| The three slice-7 pictures are `ground.png`, `hill.png`, `aloft.png`. | `docs/investigation/2026-09-07/pictures/` | `ground.png` is a brown plain under a black sky. The patch ends 403 m away; the horizon is 3 473 m away; the gap is 3.0 pixels of frame (§3.2). |

---

## 2. The words, explained once

Each row explains the term in plain words and gives an example in the game's own words.

| Industry term | What it means here, in the game's words |
|---|---|
| **Horizon distance** | How far you see before the body curves away. `d = √(h·(h + 2R))` for eye height `h` over a body of radius `R`. On the home planet (`R = 3 350 759 m`, `crates/terrain/src/home.rs`) a standing pilot at 1.8 m sees **3 473 m**; from a hull at 300 m, **44.8 km**; from 373 m, **50.0 km**; from 60 km, **637 km** (MEASURED arithmetic). |
| **Horizon dip** | How far below level the horizon sits, in degrees: `acos(R / (R + h))`. At 1.8 m it is 0.059° — under one pixel. At 60 km it is 10.76°. **A picture from 60 km with a LEVEL nose still shows ground:** the horizon lands at row `360 + 869.12·tan(10.76°) = 525` of 720, so the ground fills 27 % of the frame. The gate tilts 15° anyway, for framing (`ALOFT_TILT_DEG`, `crates/bins/tests/terrain_pictures.rs:49`). |
| **Prominence** | How far a hill stands above the highest saddle that joins it to higher ground — plainly, how much it STANDS OUT, not how high it is. A 3 000 m peak on a 2 900 m plateau has 100 m of prominence and reads as a bump. **On the home planet a standing pilot needs 323 m of prominence at 50 km to see anything at all (§0.2 M-B), and the body's measured departure from the horizon plane at that range is about 293 m — a tilt, not a peak.** |
| **Angular size (subtense)** | How big a thing looks: its size divided by its distance. The avatar's 0.5 m figure at 10 m subtends 0.05 rad, which is 43 pixels at our camera. This is what a RULER measures. |
| **Quantile (p10, p90, p99)** | Sort every measurement and take the one that stands at a stated share of the way up. The p90 slope of the home planet is the slope that 90 % of the sampled ground is gentler than: **2.96° at a 1 km spacing** (MEASURED, §3.3). A quantile is a fact about the body; a typed threshold is not. |
| **Standard deviation** | The usual size of a departure from the average. The home planet's surface has a standard deviation of **2 295 m** about its ladder radius (MEASURED). Earlier drafts of this document wrote "standing deviation"; the industry word is standard deviation, and one word means one meaning. |
| **Hypsometry** | The statistics of height over a body: how much of the surface stands at each altitude. On the home planet the curve is one smooth hump around the ladder radius, standard deviation 2 295 m, with 0.92 % of columns under the sea (MEASURED, §0.2). Earth's curve has TWO steps — the continents and the ocean floors — because two kinds of crust exist. The curve is a *number the owner can read* for "believable". |
| **Slope histogram** | How much of the drawn ground stands at each steepness. On the home planet, over 3 000 samples at one-metre spacing, the median slope is **1.37°** and the largest is **7.23°** (MEASURED, §3.3). A picture of that ground cannot show a cliff, because the SMOOTH height field holds none — the carvers and the edit pyramid still can (§3.3). |
| **Relief profile** | The surface height along one line on the ground, plotted against distance. The strip the owner reads beside a picture: a valley draws a V, a plateau draws a step, a swell draws a curve, and the home planet draws a straight tilted line with five metres of wobble (MEASURED, §3.3). |
| **Aerial perspective (haze)** | Distant things lose contrast and turn toward the sky's colour, because air scatters light. The blue distance in the owner's reference picture is this. Our client has no atmosphere, so a far chunk today is as sharp as the pilot's boots. |
| **Terminator** | The line between the lit half and the dark half of a body. From the 300 km stand it crosses the frame and is the strongest orienter in the picture. |
| **Obliquity** | The tilt of a body's spin axis away from its orbit's axis. It is what makes seasons. `crates/terrain/src/height.rs:29-35` pins every body's pole to `+Z`, the orbit's own axis, and names obliquity *"a later slice"*. So the home planet has no seasons and no tilted ice caps. |
| **Insolation** | How much sunlight one square metre of the surface receives — it falls with the square of the orbital distance and rises with the star's luminosity. `biome_at` reads NEITHER (`crates/terrain/src/height.rs:38-65`). That is why the owner's *"biomes dependent on the planet position"* has no acceptance yet (§9 V13). |
| **Albedo** | The share of light a surface throws back. Snow is high, bare rock is low. The worldgen already draws a seed-drawn albedo per body (`crates/physics/src/worldgen/body.rs:80-104`), and the terrain's paint reads none of it. |
| **Surface gravity** | The pull at a body's surface, `G·M/r²`, from the mass and the radius the worldgen already draws (`taxon.mass_kg`, `taxon.radius_m`, `crates/physics/src/worldgen/generate.rs:1206,1227`). It sets the atmosphere's **scale height** — how fast the air thins with altitude — so it sets the haze, the snow line and the dust. Nothing in `crates/terrain` reads it. |
| **Flow accumulation** | For each point on the ground, how much upstream land drains through it. A river is where that number is large. The generator computes nothing of the kind, so it has no rivers (§7.3, stand 8). |
| **Stream power** | How fast a river cuts down, as a function of its flow and its slope. It is the mechanism that turns a swell into a valley with a V-shaped cross-section. Our height field has no such mechanism, which is exactly why its profile is a straight tilted line. |
| **Orographic lift** | Air forced up a ridge cools, and cool air holds less water, so it rains on the way up. The windward side is wet and green; the leeward side is dry and brown. **The dry adiabatic lapse rate is 9.8 K per kilometre of rise**, and air condenses at its lifting condensation level — about 125 m of rise per kelvin of dew-point depression. So a visible rain shadow needs roughly **1 000–1 500 m of rise inside 10–20 km** (a 3° to 8° slope). The home planet rises 1 000 m across 50 km, a slope of 1.15°: a rain shadow here is UNMEASURED and probably invisible (§7.3, stand 15). |
| **Hadley cell** | The great loop of air that rises at the equator, moves poleward high up, and falls at about 30° of latitude. Falling air is dry air, which is why Earth's great deserts sit in a band at 15–35° and the equator is rainforest. **Our `biome_at` does the opposite** (§10.2(d)). |
| **Coriolis deflection** | A planet's spin turns a moving air mass sideways. It is what makes a storm a spiral. The home planet has no stated spin, so nothing can turn. |
| **Isostasy** | Thick crust floats high, like a deep iceberg. It is one of the two mechanisms that give Earth's hypsometric curve its second step. |
| **Base level** | The height below which a river cannot cut — the sea. On the home planet the sea covers 0.92 % of the surface, so nearly the whole body has no base level within reach. |
| **LOD pop** | A visible jump when a chunk swaps rungs. Our word for a detail level is a RUNG. SL8 calls a visible pop a defect. |
| **Second difference** | The change in the change. If a frame differs from the last one by the same amount every frame, the motion is smooth and the second difference is near zero. A rung swap makes one frame differ much more than its neighbours, so the second difference spikes. §9 V9 measures exactly this. |
| **Geomorphing** | Moving a coarse chunk's vertices smoothly onto the fine chunk's positions across a short band, so the swap has no instant. Ruling V10 names it the fallback if the crossfade's measured pop stays visible. |
| **Surface nets** | The extractor's way of making a mesh: it puts ONE vertex inside each cell that straddles the surface, then joins the vertices of four neighbouring cells into a quad (`crates/terrain/src/extract.rs:435-455`). It makes smooth ground cheaply, and it ROUNDS a sharp crease by up to half a cell — which is ruling V10's crease question, asked at stand 11. |
| **Impostor** | A far object drawn as a flat card that always faces you. Ruling V10 **refuses** them. A picture verdict must therefore never be satisfied by one. |
| **Stencil / id buffer** | A second picture of the same frame in which each pixel holds a number saying what drew it, not a colour. Free of taste, exact for geometry. This report calls the pair (id + distance) **the picture probe**. |
| **Straddle, and straddle drift** | A screenshot is bracketed between two state polls and retried until they agree, so the reconstructed camera and the pixels describe the same instant (`crates/bins/src/pixel.rs:1-24`). The **drift** is how far the subject moved between the two polls, reported in pixels (`pixel.rs:113`). It is the honest tolerance of every geometric verdict, because nothing can be measured tighter than the frame's own uncertainty. |
| **e-fold** | One step of growing or shrinking by the factor 2.718. Falling from 10 000 km to 1.8 m is 15.53 e-folds. At 1 % per frame each frame is one hundredth of an e-fold, so the fall takes `15.53 / ln(1/0.99) = 1 545` frames (MEASURED arithmetic; the last revision printed 1 553, which divided by 0.01 instead). |
| **A stand** | A named place and facing on a body from which a picture is taken. In this report a stand is **found** by a rule over the recipe, never written down as a coordinate. |

---

## 3. Why the three pictures could not be judged

### 3.1 The picture holds no quantity

`ground.png` shows brown ground, a black sky, and a HUD naming the realm, the entity and a world
position of `−2 180 953.5, 2 551 968.8, 30 026.8`. That position is in the planet's frame and says
nothing a person can use: it does not say the eye is 1.8 m over the ground, and it does not say the far
edge is 403 m away and not 3 473 m. The owner's sentence is exactly right, and it is a design fault, not
a taste: **the picture states nothing that can be checked.**

### 3.2 The hard edge is 3 pixels — the black is the missing SKY

The ground picture drew 13 × 13 columns at rung 0 (`GROUND_RADIUS: i32 = 6`,
`crates/bins/tests/terrain_pictures.rs:55`). A rung-0 chunk is 62 m across
(`crates/seed/src/ladder.rs:21`). So the drawn patch reaches `6.5 × 62 = 403 m`, and the geometric
horizon at a 1.8 m eye is 3 473 m. The picture holds **12 % of the distance the pilot should see.**

**But that 88 % of DISTANCE is 3 pixels of FRAME.** MEASURED arithmetic, on the sphere. Every depression
below is the EXACT sphere value: the patch edge sits at arc distance `s`, so its central angle is `s/R`
and its depression is `asin(−v_z/|v|)` for `v` the vector from the eye to that point. (The last revision
used the flat approximation `atan((h + s²/2R)/s)`, which is 0.02° low at 60 km. Refuter A caught it.)

```text
  stand              eye        patch      horizon   horizon dip   patch-edge dip     gap      gap px
  ground            1.8 m        403 m     3 473 m      0.0594 deg     0.2594 deg   0.2000     3.0
  hill (slice 7)  300   m      3 224 m    44 839 m      0.7667         5.3435       4.5768    69.4
  aloft (slice 7)  60   km    396 800 m   636 939 m    10.7628        11.9061       1.1432    17.3
  drone           373   m      3 224 m    49 998 m      0.8549         6.6267       5.7718    87.6
```

(The patch radii are the slice-7 gate's own: rung 3 × 6.5 columns for the hill, rung 9 × 12.5 for aloft;
the drone row shows what the hill's patch would leave at the 373 m stand. One pixel is 0.06592°.)

**So the owner's complaint splits in two, and the two have different cures.**

- At a 1.8 m eye the black is the SKY. There is no sky dome, no atmosphere and no horizon line
  (`CLEAR_SRGB = [0,0,0]`, `crates/client-render/src/lib.rs:78`). The picture reads as a void because
  nothing draws the void. **The cure is the atmosphere and the stamp, not residency.**
- At 300 m and above the black IS missing terrain: 69 pixels at the slice-7 hill stand, 88 pixels at the
  373 m drone stand. **The cure there is residency**, and §11 prices it.

A verdict that reads "the skyline sits where the body's radius says" catches the 300 m case by name and
reports the 1.8 m case as 3 pixels, which is honest (§9, V2).

### 3.3 What relief the recipe actually offers — MEASURED

The octave table of the home planet, recomputed outside the repository from
`crates/terrain/src/body.rs:140-190` and `crates/seed/src/rng.rs:22-28,70-74`:

```text
 relief 14 304.9 m   coarsest wave 400 000 m   roughness k = 0.4683   14 octaves

  i   wavelength      amplitude   amp/wave
  0     400 000 m      7 605.5 m    0.0190
  1     200 000 m      3 562.0 m    0.0178
  2     100 000 m      1 668.2 m    0.0167
  3      50 000 m        781.3 m    0.0156
  4      25 000 m        365.9 m    0.0146
  5      12 500 m        171.4 m    0.0137
  6       6 250 m         80.3 m    0.0128
  7       3 125 m         37.6 m    0.0120
  8       1 562 m         17.6 m    0.0113
  9         781 m          8.2 m    0.0106
 10         391 m          3.9 m    0.0099
 11         195 m          1.8 m    0.0093
 12          98 m          0.8 m    0.0087
 13          49 m          0.4 m    0.0081
```

**The cross-check that makes this MEASURED and not a guess.** The same recomputation produces the sea
offset `floor((u·0.7 − 0.4)·relief) = −5 297 m`. The owner's own ruling publishes that number
(`docs/design/owner_decisions_2026-09-07_voxels.md:393`). The two agree exactly.

**The relief a stand can show — MEASURED, not modelled.** One octave of this noise has a standard
deviation of **0.2701** (MEASURED over 150 000 points on the rebuilt noise), not 1.0. These are the
measured numbers, from 60 real profiles per row (40 for the longest), each sampled at 64 points along a
great circle:

| Ground distance `L` | Mean peak-to-peak | RMS end-to-end change | Mean tilt | RMS wobble after the tilt is removed |
|---|---|---|---|---|
| 400 m | **12.2 m** | 14.8 m | 1.74° | **0.5 m** |
| 3 473 m (a pilot's horizon) | **99.4 m** | 120.5 m | 1.60° | **5.5 m** |
| 44 839 m (from 300 m up) | **968.1 m** | 1 048.5 m | 1.15° | **97.5 m** |
| 636 939 m (from 60 km up) | 5 937 m | 3 286 m | 0.31° | 1 286 m |

**Read the last two columns together.** Almost all of the height is TILT, and almost none is SHAPE. The
eye reads a tilted plain as level ground. The four longest waves (400 km down to 50 km) are 14 to 115
times longer than a standing pilot's whole view, so they cannot make a shape inside it — they only tilt
it.

**The slope distribution, MEASURED over 3 000 random directions on the home planet.** The quantity is
the TWO-POINT slope: the height difference between two points a stated distance apart, divided by that
distance.

```text
  spacing    median    p90     p99     largest seen
     1 m     1.37 deg  3.29    5.06     7.23 deg
    62 m     1.31      3.21    4.78     6.60
     1 km    1.22      2.96    4.72     5.64
```

**The largest two-point slope measured anywhere is 7.23°.** A cliff, a pillar, a mesa rim and a canyon
wall are all slopes over 45°.

**The narrower claim, corrected.** The last revision wrote *"the home planet holds no cliff"*. That is a
statement about `height_m` presented as a statement about the DRAWN surface, and refuter A refuted it
from three code facts. The true statement is:

- **The SMOOTH HEIGHT FIELD holds no slope over about 8°.** That is the measurement above.
- **The carvers open vertical faces.** A cavern opens rooms and a tube cuts passages
  (`crates/terrain/src/carve.rs:1-16`), and the extractor emits a quad wherever `is_rock` differs across
  a cell edge (`crates/terrain/src/extract.rs:435-455`). A cave mouth is a hole with a wall around it.
- **The generator's own code contemplates a cliff.** `crates/terrain/src/digest.rs:92-94` names the two
  cases slice 8's residency band exists for: *"a slope steeper than the margin (a cliff of more than 62
  cells inside one column)"* and *"a cave mouth deeper than the margin"*.

So everything vertical in the world today comes from the carvers or from the edit pyramid. That
strengthens stand 11 (the authored crease) and stand 12 (the cave mouth) rather than weakening them.

**Two numbers this domain and its sibling report differently, named apart.** Sibling document
`01_reference_target.md:280-286` reports a median **local tilt** of 2.28° at a 50 m ring and 2.05° at a
1 000 m ring, over 200 000 columns with a PLANE fitted to a RING. This document reports a median
**two-point slope** of 1.37° at 1 m and 1.22° at 1 km. **They are different quantities and both may be
right.** A plane fitted to a ring of radius `r` keeps the wavelengths up to the ring's circumference
`2πr`; a two-point slope over spacing `L` keeps the wavelengths up to `L`. At `r = 1 000 m` the ring
reaches 6 283 m of wavelength, where the octave amplitude is 80.3 m; a 1 km two-point slope reaches
1 000 m, where it is about 10 m. The ring therefore reads a larger number by construction, and the same
mechanism explains the wobble rows (08 measures 5.5 m over a 3 473 m profile; 01 measures 7.05 m around
a 1 000 m ring). **The reconciliation is STATED, not proved: M8-10 puts ONE bench in `vd-terrain` that
both domains cite, so the world has one answer.** Until it runs, the last revision's claim that the two
documents *"agree in the same words"* is withdrawn: they agree on the no-cliff CONCLUSION only.

**The conclusion.** The picture protocol recovers a **tilted plain**, more honestly framed and better
instrumented than the slice-7 picture. It does not recover a hill. Only a new landform mechanism does.
The acceptance measures the gap and reports it (§10.2); closing it belongs to the landform domains.

---

## 4. THE STAMP — the on-frame readout, measured and never typed

### 4.1 The rule

> Every number the stamp states is computed by ONE expression in a Tier-A crate, from the delivered
> state, the picture probe and the linked generator. The renderer draws the strings that expression
> produced. The gate asserts on the same struct, out of the picture's own state file. **No number is
> typed into a picture, and no number is computed twice.**

This is the star-sky lesson, applied to terrain: *"Asserting the client's drawn count against the
client's held count would be checking one source against itself, and two numbers from one source always
agree"* (`crates/bins/tests/star_sky_pixels.rs:14-17`). So the stamp also carries **pairs that come from
different sources and must agree** — the world's geometry against the picture's pixels (§9, V2).

### 4.2 The fields, with a REAL example

The stamp below is the FLAGSHIP stand: a standing pilot, 1.80 m over the surface. It is labelled
`ground-01`, because the eye is on the ground. (The last revision headed the same numbers `vista-01`,
which was the wrong label; refuter B caught it.)

**How to read the address.** The face and the two face parameters are an address ON the home planet,
printed so the reader can check every line of arithmetic below. **It is not a found address.** The
search of §7.2 strides the address space with a co-prime step and writes whatever it reaches first into
the picture's own state file. A worked example must state one address; the recipe states the rule.

```text
  VOXELDUST — picture stamp
  realm      Planet 7701581858760374086          <- delivered (own location frame)
  stand      ground-01  face +Y  (0.0121, -0.8553)   <- the stand recipe, and the address it found
  eye        1.80 m over the surface, 3 351 890 m from the centre
  ground     +6 426 m over the sea      biome Grassland
  horizon    geometric 3 474 m | drawn 3 410 m | terrain radius resident 3 476 m
  rungs      0: 462 chunks | 1: 348 | 2: 348 | 3: 348   (four rungs, 1 506 chunks)
  sun        elevation 25.0 deg, azimuth +0.0 deg from the nose
  ruler      avatar 0.50 m at 10.0 m -> predicted 43.5 px, measured 43 px
  camera     45.0 deg over 720 rows, one pixel = 4.0 m at the horizon
  world      declared 0x91af..  measured 0x33c7..   universe tick 41 208
```

The internal checks the reader can run: `3 351 890 − 1.8 − 3 350 759 = 1 129 m` of relief, and
`3 350 759 − 5 297 = 3 345 462 m` of sea radius, so `3 351 888 − 3 345 462 = 6 426 m` over the sea —
the row's own figure. `√(1.8 × (1.8 + 2 × 3 351 888)) = 3 474 m`. `0.0011506 × 3 474 = 4.00 m` per pixel
at the horizon. `0.5 / 10 / 0.0011506 = 43.5 px`. **The rung row: `154 + 116 + 116 + 116 = 502` columns
times 3 chunks per column = 1 506 chunks, split `462 : 348 : 348 : 348` — the exact split §11 prices.**
(The last revision printed 1 505 and `463 : 347 : 347 : 348`. Both refuters caught the arithmetic; both
are corrected here and in §11.)

**One word about the world row, because it reads like a failure and is not.** `declared` and `measured`
are two DIFFERENT quantities: the declared tag folds the version and the seed, and the measured half
folds the client's own computed chunk and mesh digests (`crates/client/src/net.rs:55-56,146`). They are
never equal, and a mismatch is refused by name at login (`WorldRefused`), never by a reader comparing two
hex strings.

| Field | Where the number comes from | Why it cannot be typed |
|---|---|---|
| realm | `DevState.location`, the own entity's delivered frame label (`crates/client/src/net.rs:886-892`) | The server states it. |
| stand + address | The stand recipe (§7) plus `vd_seed::bend::face_of` / `face_coords` on the delivered position | The address is FOUND, so it moves when the recipe moves. |
| eye altitude | `‖delivered position − realm centre‖ − height_m(body, dir, 0)`, with `body` the one the chunk lane already holds (`crates/client/src/chunks.rs:294`) | It reads the delivered pose through the same chokepoint the renderer draws with (`world_pos`), and the one generator (SL10). |
| ground height over sea | `height_m(body, dir, 0) − body.sea_radius_m` | The generator's, not a constant. |
| biome | `vd_terrain::height::biome_at(body, dir, surface)` (`crates/terrain/src/height.rs:38`) | The generator's. |
| geometric horizon | `√(h(h + 2R))`, `R` = the body's ladder radius (`crates/seed/src/ladder.rs:103`) | Derived from two measured quantities. |
| **drawn horizon** | **the farthest TERRAIN pixel in the picture probe's distance buffer** | **The independent half, and it is IN THE FRUSTUM by construction.** The renderer's `farthest` is not used for this: its loop has no frustum test (`crates/client-render/src/terrain.rs:522-546`), so it measures residency, not reach. |
| terrain radius resident | the chunk lane's resident set (`ChunkLane::resident`, `crates/client/src/chunks.rs:390`), the farthest key's centre distance | The library's own book. Named RESIDENT, so it is never mistaken for the picture's reach. |
| rungs and counts | the resident set, grouped by rung | Slice 8 draws several rungs; the field is a list from the first day, never one number. |
| sun elevation and azimuth | the brightest luminous row in the window — the very row the renderer lights from (`crates/client-render/src/terrain.rs:430-436`) — projected onto the local up and the nose | The star's direction is the world's, through the delivered window. |
| ruler | §5 | Two numbers from two sources. |
| camera | `REFERENCE_VIEW_FOV_Y_RAD` and the frame's rows; one pixel at the horizon through `one_pixel_world_m` (`crates/client-harness/src/camera.rs:273`) | The one camera expression the gates reconstruct with. |
| world identity, tick | the world hello's two halves (`crates/client/src/net.rs:55-56,146`) and `DevState.universe_tick` | Two pictures of two builds are then never confused. |

### 4.3 Where the code goes

```text
   vd-terrain (Tier-A, FENCED)   the one generator: height, biome, sea, ladder radius, THE STAND SEARCH,
        |                        and BodyStats (the gate calls it; a login never does)
   vd-client (Tier-A)       PictureStamp::of(delivered state, body, chunk census, probe reading)
                            PictureStamp::lines()                                  -> the strings
        |                                    |
   vd-devproto (Tier-A)     DevState.stamp: Option<PictureStamp>   (serde; rides the state file)
        |                                    |
   vd-client-harness (A)    the pure verdicts over the probe buffers (skyline, holes, ring fit, spikes)
        |
   vd-client-render (B)     draw_hud() draws stamp.lines()          <- draws, never computes
                            the probe render pass                   <- a device, not a number
   vd-bins tests (B)        reads state.stamp                       <- asserts, never recomputes
```

`vd-client` already depends on `vd-terrain` and `vd-seed` (`crates/client/Cargo.toml:10-12`), so the
stamp adds no crate edge. `vd-client-harness` depends on neither and needs neither: its verdicts read
buffers and numbers, never the generator.

One test in `vd-client` asserts `lines()` is a pure rendering of the struct. One test in the gate asserts
the stamp in the picture's state file matches the stamp polled at the straddle.

### 4.4 CLIMATE and WEATHER are two different things — the split, and the SL6 ASK this domain owes

The owner wrote one sentence about weather. The laws split it in two, and **making the split explicit is
the most useful thing this domain hands the weather domain.**

```text
   CLIMATE  = a static function of the body: latitude, insolation, the windward side of a ridge,
              the snow line. It is SHAPE. Under SL10 the client MAY derive it, it crosses no
              boundary, and it costs nothing on the wire.
                 example: the leeward side of the ridge at stand 15 is dry on every tick, for
                 every pilot, forever.

   WEATHER  = live state: THIS hour's cloud, THIS hour's rain, THIS hour's wind. The seed ruling
              forbids deriving live state, so it must be SHIPPED from the realm that owns it.
                 example: two pilots stand on the same ridge at universe tick 41 208. They must
                 see the SAME cloud. A client that invents its own cloud is a second source.
```

The last revision filed the windward/leeward pair (pure climate) under weather, and filed the ask (pure
weather) beside it. This revision separates them: stand 15 and V18 are CLIMATE; V15, V16 and V17 are
WEATHER, and each needs the shipped data.

> **SL6 ASK (owed to the owner).** *What data:* three scalars per realm per change — cloud cover
> (0..1), precipitation rate, visibility in metres. *From which realm to which:* from the planet's own
> shard down its window lane to a client that draws it. *Why the receiver cannot compute it:* weather
> is LIVE STATE, not seed-shape; the 2026-08-27 seed ruling forbids deriving anything valuable from the
> seed alone, and SL10 admits only the static shape on the client. *What doing without costs:* the
> client would invent its own weather, which is a second source and a guaranteed disagreement between
> two players standing in the same rain — the exact thing V15 measures. *This domain's recommendation:*
> ASK, because the alternative is worse; but the ask belongs to the weather domain's design and this
> document only names it.

Until that ask is answered, V6 (the haze verdict) and V15–V17 (the weather verdicts) do not run, and the
stamp carries no weather field.

---

## 5. THE RULER — what stands beside the eye

### 5.1 Why a ruler and not a number

The stamp says "the eye is 1.8 m up". A ruler proves it, because it is drawn by the same pipeline the
ground is drawn by. If the world's scale is wrong anywhere between the shard and the pixel, the ruler is
the wrong size and the stamp is not.

### 5.2 The avatar marker is an exact ruler inside 145 m — MEASURED

The marker's world radius is `max(base_radius_m, 3 px × one_pixel_world_m(dist))`
(`crates/client-harness/src/camera.rs:264-266`), with `base_radius_m = 0.5 m`
(`crates/core/src/look.rs:100`). One pixel at distance `d` is `2·d·tan(22.5°)/720 = 0.0011506·d`
(MEASURED arithmetic). The floor takes over when `3 × 0.0011506 × d > 0.5`, that is at
**d > 144.9 m**. Inside that the painted radius is exactly `0.5 / d / 0.0011506` pixels; past it the dot
stops shrinking, so its pixels no longer state a size.

**Verdict:** the avatar marker is an exact near-field ruler inside 145 m and a presence dot beyond it.
The stamp states which of the two it is, and a gate never sizes anything from a marker past the floor.

### 5.3 Why the hull cannot be a ruler today — MEASURED

An earlier draft made a hull the far-field ruler and predicted its footprint as `40 m / d / 0.0011506`.
Three facts from the code refute it.

1. `DevRealmBox.extent_m` is a **bounding-sphere radius**: a box's half-diagonal
   (`crates/devproto/src/state.rs:63`; `shape_extent_m = half.length()`,
   `crates/client/src/realm_scene.rs:532-541`). The default hull is half-extents 6 / 3 / 20 m
   (`crates/bins/src/bin/build-ship.rs:164-166`), so its streamed extent is `√(36+9+400) = 21.095 m`.
2. The hull DRAWS as an oriented cuboid, scaled by the three half-extents and turned by the facing its
   parent authored (`crates/client/src/realm_scene.rs:831-843`). The diagnosis row carries **no
   half-extents and no facing**.
3. The prediction, the instrument and the paint therefore give three different answers at 60 m:

```text
  the old draft's prediction     40.000 m / 60 / 0.0011506  = 579.4 px
  the instrument's rectangle     2 x 21.095 / 60 / 0.0011506 = 611.1 px   (the bounding sphere)
  the painted cuboid, broadside     40 m wide x 12 m tall     = 579 px x 174 px
  the painted cuboid, end-on        12 m wide x  6 m tall     = 174 px x  87 px
```

The painted answer swings by a factor of 3.3 with a facing nobody states, against a bound of "the
straddle drift plus one pixel". **A hull ruler as specified fails on a correct picture.** And the hull's
look draws TRANSLUCENT (`AlphaMode::Blend`, `crates/client-render/src/lib.rs:1823,1849-1856`), so a
single-author id buffer cannot label its pixels either.

### 5.4 What the ruler is instead — and what it is NOT

**The correction this revision makes.** The last revision said *"The reference picture uses a character
for scale; so do we"*, and it said three avatars *"cost nothing"*. Both are false, and both refuters
found them.

1. **The avatar is an INSTRUMENT, never the picture's scale figure.** It paints a BALL of
   `OCCUPANT_FIGURE_EXTENT_M = 0.5 m` (`crates/core/src/look.rs:100`), scaled through
   `marker_world_radius` (`crates/client-harness/src/camera.rs:264-266`); §1's own table calls it *"a
   presence dot"*. A person reads a HUMAN SILHOUETTE without thinking, and reads a ball not at all. So
   the avatar carries V1 exactly, and it carries the owner's sense of scale not at all. **The reference
   picture's own device — a character in frame — is OWED to the character slice (16)**, and this
   document registers the dependency rather than pretending a dot is a person.
2. **An extra figure is an extra LOGIN, and it is priced.** The camera sits AT the own avatar, so a
   figure 10 m ahead is a SECOND OCCUPANT: another account, another `VD_SPAWN_POSES` entry, another
   login (`crates/bins/tests/terrain_pictures.rs:457-472`, one spawn entry per account). §11 M8-7 counts
   them: the calibration stand takes three logins, every other stand takes one.
3. **The far ruler needs one diagnosis-row change.** `DevRealmBox` gains `half_extent_m: [f64; 3]` and
   `facing: [f64; 4]` — read from the same `BoxShape` and `RealmBox.facing` the renderer already draws
   with (`crates/client/src/realm_scene.rs:110-115,831-843`). Then the projected cuboid is exact, and
   V1 compares it against the painted pixels. **This is the harness's own diagnosis lane, not a realm
   boundary**, so SL6 does not apply; `DevState` already carries `extent_m`, `luma` and `body_kind` by
   the same right.
4. **The rulers ride in the STAMPED twin, never in the clean picture.** Q5 proposes two captures from
   one straddle. A 40 m hull at 60 m fills 45 % of the frame's width; that is an instrument, not a
   picture.

**The ruler's verdict (V1).** The subject's **painted** footprint, read out of the picture probe's id
buffer, agrees with the footprint predicted from its delivered geometry and its delivered distance,
within the straddle drift plus one pixel. Two sources: the wire's geometry, and the pixels.

### 5.5 The scale ladder in one frame

The CALIBRATION stand — and only that stand — carries **three avatars** along the line of sight, at
10 m, 40 m and 120 m: predicted footprints 43.5 px, 10.9 px and 3.6 px (MEASURED arithmetic,
`0.5 / d / 0.0011506`), all inside the 145 m floor. It is a 1.8 m ground stand, so the three stand ON the
ground and not in the air. (The last revision put them at the 373 m stand, where they would have hung
370 m up. Refuter B caught it.) **Three logins, counted in M8-7.** No other stand carries them, and the
flagship carries none: a ball in the middle of the vista is in the way of the picture the owner judges.

---

## 6. THE PICTURE PROBE — how a machine reads a picture

### 6.1 The defect the probe removes

The slice-7 gate classifies ground by colour (`crates/bins/tests/terrain_pictures.rs:266-289`). Two
things are true about it.

- **It is a fitted heuristic.** A rule that wants red at least green, green at least blue, red at least
  blue plus eight, and red over twenty-four is a rule tuned to one material's paint. It cannot be
  derived from anything and it can only ever be re-tuned.
- **It survives today, because there is no biome paint at all.** The whole terrain uses one material,
  `base_color: Color::srgb(0.55, 0.50, 0.42)` (`crates/client-render/src/terrain.rs:362`), and the word
  "biome" occurs nowhere in `crates/client-render/src` or `crates/client/src`. So the classifier is not
  failing today; it is a rule that will fail the day a paint table lands.

The probe's case therefore rests on what a colour cannot say: a colour cannot say how far a pixel is,
WHICH MESH drew it, where the skyline is, or whether a far ground is a mesh or a card.

### 6.2 What the probe is — an author AND a mesh instance

Every JUDGED capture writes **two** aligned buffers:

```text
   colour.png     what the owner judges       RGBA8, 1280 x 720, the HUD and stamp on it
   probe.bin      what the machine judges     per pixel: u16 id + f32 distance in metres
```

**The id is TWO fields inside one `u16`, and the last revision defined it two ways.** Refuter A found the
contradiction: §6.2 said the id names an AUTHOR, and V8(b) needs to know WHICH MESH drew the pixel. Seven
authors is a three-bit set, under which every terrain pixel — a near chunk, a far chunk, or a flat card
standing in for a far band — carries the identical id, and V8(b) is unmeasurable.

```text
   bits 15..13   the AUTHOR   unauthored | terrain | water | hull | marker | star | HUD | atmosphere
   bits 12..0    the MESH INSTANCE, 0..8 191, assigned per drawn mesh in the frame
```

Eight authors and 8 192 mesh instances. The largest stand in §11 draws 5 752 chunks, so `u16` holds it
and the price stays six bytes per pixel. `water` is reserved and unwritten until a sea surface exists
(§7.3, stand 10); `atmosphere` is reserved for the day a sky is drawn — V6 bins terrain pixels against
the sky's colour, and the sky needs an id of its own then. Today nothing writes either: the clear colour
is deep-space black, so the verdicts treat an unauthored pixel above the skyline as lawful (§9, V3).

The probe is written on the capture path only, and only on frames the manifest STAMPS. At 6 bytes per
pixel one probe is `1 280 × 720 × 6 = 5.53 MB` (MEASURED arithmetic), so a 1 545-frame recording would
be 8.5 GB if every frame carried one. It does not: §8 stamps about twenty frames, which is 111 MB.

### 6.3 What the probe makes exact

| Question | With colour | With the probe |
|---|---|---|
| Where is the skyline? | a fitted threshold | the topmost terrain pixel per column, exactly |
| Is there a hole in the ground? | invisible under a black sky | an unauthored pixel below the skyline, counted |
| Is a whole COLUMN missing? | invisible | a column with no terrain pixel at all, counted (§9 V3(b)) |
| How far is that pixel? | unknowable | read |
| How far does the picture actually SEE? | unknowable | the farthest terrain pixel, in the frustum by construction |
| How big is the subject on screen? | a colour guess | the id's pixel extent |
| Does the haze grow with distance? | not measurable | luminance binned by distance |
| Is the far ground a mesh or a card? | not measurable | **the mesh-instance field says how many meshes drew the band** |

### 6.4 Cost and coverage

The probe pass is a second render target in the capture path only. Its cost is UNMEASURED; it is
measured by M8-2 (§11). The verdicts over the buffer are pure functions in `vd-client-harness` (Tier-A,
100 %), tested on synthetic buffers exactly as `assert.rs` is
(`crates/client-harness/src/assert.rs:1-6`). The pass itself is Tier-B renderer code.

---

## 7. THE STANDS — found, never typed, and satisfying, never argmax

### 7.1 The rule

> A stand is a SATISFYING PREDICATE over the body's own recipe, at a stated scale, evaluated over a
> stated set of ladder addresses in a stated order. The first address that satisfies it wins. It is
> never a written-down face and cell, and it is never an argmax.

Four reasons, each binding.

1. **No magic numbers.** A typed coordinate is a world parameter that came from nowhere. So is a typed
   length and a typed angle (§7.2 derives every one).
2. **The stand must survive the recipe.** The landform slices will change the height field. A typed
   ridge stand would then stand on a plain, and the picture set would silently stop testing what it was
   made to test. A found stand still stands on a ridge.
3. **The comparison over time.** The same predicate re-run after slice 14 puts the eye in the same kind
   of place, so the owner can compare "the ridge, before the forest" with "the ridge, after the forest".
4. **An argmax is not a landform on a homogeneous field, and it is unaffordable.** The home planet's
   height field is statistically the same everywhere: no tectonics, no drainage, no uplift. So "the
   largest 1 km prominence on the planet" is a lottery ticket, not a ridge — and finding it needs a
   sample spacing of 500 m, which is `4πR² / 500² = 5.6 × 10⁸` directions; at nine samples per direction
   and about 390 ns per column that is **about 33 minutes per predicate per run** (ESTIMATED). A
   satisfying predicate costs nothing: MEASURED, **9.4 %** of directions hold a 1 km slope over 3°, so a
   first-hit scan ends after about eleven addresses and about 100 height evaluations — **39 µs**
   (ESTIMATED at the same rate).

**The rate this arithmetic uses, sourced honestly.** The ~390 ns per column is NOT published in the
repository. It is DERIVED from the published column pass — *"the column pass at rung 0 ~1.5 ms per
62 × 62 columns with 14 octaves"* (ruling V10's measured budget) — as `1.5 ms / 3 844 = 390 ns`. That
pass is AMORTISED over a whole chunk's columns; a scattered `height_m` call at a strided address has no
such amortisation and is likely SLOWER. **M8-1 measures the scattered call**, and until it does, every
microsecond figure in this section is ESTIMATED, never measured. (Refuter B is right that the last
revision called a derived rate a published one.)

### 7.2 The search, under the float fence

**Where it lives, and the REAL reason.** In `vd-terrain`, under the fence. The last revision gave an
arithmetic reason — *"a prominence computed in plain `f64` could flip between x86-64 and aarch64"* — and
**that reason is false.** The fence's own text names the operations IEEE-754 fixes on every target:
*"add, subtract, multiply, divide, square root, the round-to-integral family, comparison, abs, clamp"*
(`crates/terrain/clippy.toml:1-8`). A comparison of two such numbers cannot flip. Refuter A is right, and
the true reason is a LINT reason, which is sufficient on its own:

> An unfenced crate has NOTHING THAT STOPS a later slice writing `.sin()` or `.powf()` into the search.
> `vd-client-harness` carries no `clippy.toml`, no compile-fail control (`crates/terrain/src/gf.rs`) and
> no link scan (`just terrain-link-scan`). In `vd-terrain` all three layers stand on the first day.

**How it names a direction, with no transcendental.** The fence bans `f64::sin`, `cos`, `acos`, `powf`,
`to_radians`, `to_degrees`, `min`, `max` and the rest (`crates/terrain/clippy.toml:9-45`), so a
golden-angle spiral is FORBIDDEN. The search set is an **address stride on the ladder**: `(face, i, j)`
at a stated rung, turned into a unit direction by `vd_seed::bend::direction`
(`crates/seed/src/bend.rs:251-259`), which is already fenced and is already the generator's own way of
naming a direction.

**The stride is CO-PRIME, not a raster scan.** The last revision said *"face index, then `i`, then `j`;
first hit wins"*. At a 9.4 % hit rate that ends within about eleven addresses of the scan's START, so
every stand on every body would stand in the same corner of the same face, and "the comparison over
time" would compare one corner forever. Refuter A is right. The corrected rule:

```text
   let M = 6 * C * C            all the addresses of the search rung (C chunks per face edge)
   let S = a stride drawn from the body's seed, raised until gcd(S, M) == 1
   the k-th address is  (k * S) mod M ,  k = 0, 1, 2, ...
   first hit wins.
```

Every operation is integer. A co-prime stride visits every address exactly once, so the scan is
exhaustive if nothing satisfies, and it spreads the found stands over the whole body. No tie-break exists
to need, because the order is total.

**How its numbers are derived, not typed — and NOT in degrees.** Refuter A's second point stands: the
fence bans `to_radians`, `to_degrees` and `atan`, so a predicate inside the fence **cannot name an
angle**. Every threshold is stored and compared as a RISE OVER RUN.

| The predicate needs | It reads | On the home planet that is |
|---|---|---|
| a neighbourhood at "the 1 km scale" | `octaves_at(0)[8].wavelength` | 1 562 m |
| a neighbourhood at "the 5 km scale" | octave 6's wavelength | 6 250 m |
| a step "within 500 m" | octave 10's wavelength | 391 m |
| "flat" | the body's own p10 slope RATIO at that spacing | rise/run at most 0.0049 (0.28°, when the stamp prints it) |
| "steep" | the body's own p90 slope RATIO at that spacing | rise/run at least 0.0517 over 1 562 m (2.96°) |
| a tolerance in pixels | the body's own measured RMS wobble at the view length, times three | 5.67 px at a 44 839 m view (§0.2 M-D) |

**The degrees are printed by the STAMP, outside the fence.** The stamp lives in `vd-client`, which is
Tier-A but not fenced, and it may call `atan`. The stand recipe never does.

**The sun rule is NOT part of the fenced search.** A sun elevation needs `to_radians`, `cos` and `sin`,
and the slice-7 gate already computes one outside the fence
(`crates/bins/tests/terrain_pictures.rs:444-460`). It stays there. The fenced search finds the PLACE; the
gate chooses the MOMENT.

**Where the body's measured field table lives, and what it costs.** The quantiles and the wobble are a
function of the recipe: one pass of a stated sample count over strided ladder addresses. Refuter A asked
three questions the last revision did not answer. Here are the three answers.

- **Who computes it.** A separate function, `BodyStats::measure(body, sample_count)`, that the GATE
  calls. **NOT `BodyDefinition::from_seed`.** `from_seed` runs on the CLIENT for every planet the chunk
  lane meets (`crates/client/src/chunks.rs:277`) and in the shard (`crates/bins/src/bin/shard.rs:404`);
  a million-address pass there would make a pilot's login wait about four tenths of a second for a table
  he never reads.
- **Does it enter the body's identity.** **No.** `BodyDefinition` is compared with `PartialEq`
  (`crates/terrain/src/body.rs:39`) and pinned by `crates/bins/tests/home_body_pin.rs`. `BodyStats` is a
  separate type, outside the pin, outside the golden table and outside the world identity.
- **What HR5 does with it.** The million-address pass NEVER runs in a unit test. The unit tests measure a
  small stated sample on the home planet and assert the SHAPE (the quantiles rise in order; the wobble is
  positive), which covers every branch. The million-address pass runs in the bench (M8-8) and in the
  gate.

**The stride's determinism, stated honestly.** A stride over integer addresses through a fenced
`direction` is exactly reproducible by construction. **The no-drift gate `D-TERRAIN-1` would extend to
it — but that gate is itself 🟧, not 🟩:** its x86-64 leg (G4) is EMULATED, and its own row says *"it can
find a drift and can never prove its absence"* (`docs/design/DEFERRED.md:7721-7745`). So the honest
sentence is: **M8-3 adds the found addresses to the golden table, and the cross-CHIP half of both is
UNMEASURED until a real x86-64 machine runs the legs.** The last revision wrote *"exactly as the golden
chunk digests already are"*, which asserted a measurement nobody has taken.

A stand's full recipe is five things, and all five are derived:

```text
   stand = { name, predicate, altitude rule, nose rule, ruler placement }
```

The altitude rule is "eye height over the surface at the found direction", so a stand is 1.8 m or 373 m
up on every body, not a body-specific number. The nose rule is a direction derived from the local
surface (down the slope, along the coast), never a typed azimuth.

### 7.3 The stands

Seventeen stands. Six are marked **OWED** with the measurement that refuses them today.

| # | Stand | The predicate that finds it | What it must SHOW | Verdicts |
|---|---|---|---|---|
| 1 | **Calibration** | the first address whose slope ratio over octave 10's wavelength is under the body's p10 | three avatars at 10 m, 40 m and 120 m on flat ground; the stamped twin adds the far ruler. **Three logins.** | V1 on all three (43.5 / 10.9 / 3.6 px); V2; V3 |
| 2 | ★ **THE PILOT'S VISTA, 1.8 m** | the same address as stand 3, seen from a standing pilot's eye | **THE PICTURE the owner judges.** The reference picture is a ground picture (§0.2 M-B): what a person standing on our world sees, with the visible-prominence curve printed beside it | V2, V3, V8, **V11**; the relief profile strip |
| 3 | **The high ground, 1.8 m** | the first address whose slope ratio over octave 8's wavelength exceeds the body's p90 at that spacing (MEASURED: 9.4 % of the body qualifies) | a slope falling away, ground between the eye and the horizon | V1, V2, V3, V8, V11; the relief profile |
| 4 | **The drone diagnostic, 373 m** | the same address as stand 2, from 373 m | NOT a vista: a DIAGNOSTIC. It shows the tilt cleanly, and it makes residency the dominant defect (87.6 px of missing ground, §3.2) | V2, V3, V8 |
| 5 | **The low ground, 1.8 m** | the first address whose height over the sea is under the body's p10, with the p90 slope ratio inside octave 6's wavelength | walls on both sides where the recipe allows any, the floor running away | V2, V3, V11; the cross-profile |
| 6 | **The polar cap** | the first address inside a stated latitude band whose biome is `Tundra` (`POLE_AXIS = +Z`, `crates/terrain/src/height.rs:29-35`; MEASURED: 45.5 % of the body is Tundra) | the cap's edge, and the low sun's Lambert term — **no long shadows, because the renderer casts none** (§0.2 M-C) | V2, V3; V7 **owed** (no paint table) |
| 7 | **The desert** | the first address whose biome is `Desert` (MEASURED: 3.1 % of the body, so the scan ends in about 32 addresses) | the dune field the sibling domain asks for, when dunes exist | V2, V3; V7 **owed**; **V12 reads its latitude** |
| 8 | **The highland** | the first address whose biome is `Highland` — the surface over `highland_above_m` = 7 868 m (MEASURED: 14.3 % of the body) | the snow-capped ridge of the reference picture, when a snow line exists | V2, V3; V7 **owed** |
| 9 | **The river bend** | **OWED — NOT BUILDABLE TODAY.** The recipe holds no hydrology: no flow accumulation, no stream power. The predicate for the day it exists: the largest flow accumulation within a stated basin, at a bend of stated curvature | a channel, a bank, the valley the channel cut | owed; §16 Q4 |
| 10 | **The coast** | **OWED — NOT BUILDABLE TODAY.** Two facts refuse it: `Stratum::Water` is not solid, so the extractor makes no face for it and **the sea is not drawn at all** (`crates/terrain/src/strata.rs:91-95`; `crates/terrain/src/extract.rs:73-75,444-446`); and the sea covers **0.92 %** of the body (MEASURED; sibling 01 replays 0.99 % over 200 000 columns) | the waterline, the beach, water to the horizon | owed; §16 Q8 |
| 11 | **The plateau edge** | **OWED — NOT BUILDABLE TODAY.** MEASURED: the largest two-point slope over 3 000 samples at one-metre spacing is **7.23°**, in the SMOOTH height field. A cliff is over 45°. The predicate would find the steepest SWELL and call it a cliff | a cliff face, a flat top, the drop | owed; and see stand 12 |
| 12 | **The authored crease** | the crease question of ruling V10 asked on a subject that HAS a crease: one column of cells set to `Empty` through the edit pyramid, on a hull's own terrain or on the planet | does surface nets round a sharp crease by half a cell of the rung drawn, and is that visible | V3, V5; **this is where ruling V10's crease question is answered**, not stand 11 |
| 13 | **The cave mouth** | the first address near stand 3 where the carver's field opens the surface (a surface cell with a cavern cell under it inside the caves' own `min_depth_m`, `crates/terrain/src/body.rs:217-229`) | the mouth, the dark inside, the lattice step or its absence — **a real vertical face, from the carvers** (§3.3) | V3, V5; the cave lattice step is judged here (ruling V10) |
| 14 | **The face seam, along and across** | the ladder's own bound: the face parameter at its limit; two pictures, the nose along the seam and across it | no line, no step, no change of detail across the seam | V5, V9 |
| 15 | **The windward/leeward pair** | **OWED — CLIMATE, and probably refused twice.** Two stands across one ridge. MEASURED: the body rises 1 000 m across 50 km, a slope of 1.15°, and a visible rain shadow needs about 1 000–1 500 m inside 10–20 km (§2, orographic lift). **So this body may hold no rain shadow even after the weather lands** | a wet green side and a dry brown side | V18, owed on BOTH the paint table and a rise the body does not have |
| 16 | **Orbit at 300 km** | the radial over stand 2, the nose down by the horizon dip (23.39°) | the limb, the terminator, the hypsometry | V2, V4 (92.64 px of sag), V8 |
| 17 | **Orbit at 10 000 km** | the same radial, the nose down by 75.47° | **the whole body in frame** — a disc 451 px across (MEASURED by exact projection). The 2 000 km stand is DROPPED as a still: its disc is 1 396 px across and does not fit a 1 280-wide frame (§9 V4) | V2, V8; V4 fits the LIMB CIRCLE, not the sag |

**The two-body pair is no longer a stand.** It was one in the last revision, and refuter B showed it
would go green for the wrong reason: two bodies draw different `temperature_seed` and `humidity_seed`
(`crates/terrain/src/body.rs:203-215`), so two pictures already look different for a reason that has
nothing to do with the owner's sentence. It becomes **V13, an ORDERING verdict over the world's own
bodies** (§9), which can fail.

### 7.4 One light per stand, and why the doubling is dropped

The last revision asked for TWO pictures of four stands, on the argument that *"the reference picture's
believability comes largely from raking light"*. **The renderer casts no shadows** (§0.2 M-C):
`shadows_enabled: false` on the terrain sun, the terrain fill and the scene key light
(`crates/client-render/src/terrain.rs:457-459,471-473`; `crates/client-render/src/lib.rs:847-848`), and
a deliberate FILL light from the opposite side at 8 % (`FILL_SHARE`, `terrain.rs:56`). On ground whose
median slope is 1.37° a 10° sun changes the Lambert cosine by about 2 %. **A picture pair that differs
by 2 % of brightness answers a weaker question than the one it was asked.**

So: **one light per stand, at the body's own low daylight elevation**, chosen by the gate from the
planet's own orbit as it does today (`crates/bins/tests/terrain_pictures.rs:444-460`) — never by moving a
light. **The doubling returns the day a shadow pass exists**, and that pass is named here as the
precondition, with no owner yet. The picture count falls from sixteen to fourteen.

**The assumption that carries every ground stand, stated.** Those same gate lines warn: *"The planet's
frame is its parent's while the planet does not spin — ASSUMED here … a planet that spins one day makes
this gate red with 'the ground is not in the picture', and the fix is to read the row's facing off the
state, which does not carry it yet."* The owner's task asks for biomes that depend on SPIN. **So the day
a planet spins, every ground stand in this set breaks, and the fix is a facing on the diagnosis row.**
That is a dependency of this domain on the spin slice, and it is registered here, not hidden.

---

## 8. THE APPROACH — the seamless sequence

### 8.1 Why a recording and not four screenshots

SL8 says a seam is a defect. A seam lives BETWEEN two frames. Four screenshots at four altitudes cannot
hold one. So the approach is a **recording**: `DevRequest::Record { fps, secs }`
(`crates/devproto/src/dispatch.rs:47-51`), which writes one PNG per frame, one state dump per frame and
a manifest row for each (`crates/bins/src/bin/client.rs:1231-1240`;
`crates/client-harness/src/capture.rs:60-93`).

### 8.2 TWO recordings, because one path cannot answer both questions

The last revision asked for one recording that falls from 10 000 km to 1.8 m at 1 % of the altitude per
frame. **Nothing in the world can fly that path.** At 30 fps the first frame moves the eye 100 km, which
is 3 000 km/s — one hundredth of the speed of light — and the last frame moves it 18 mm. A hull states
its rated cruise and acceleration as facts about what it IS (the 2026-08-27 ruling), and a suit states
its own (the 2026-09-05 ruling). Neither states anything near that. Both refuters are right, and the cure
is to split the recording in two, because it was answering two different questions.

**(1) THE FLIGHT — the SL8 gate.** The eye descends on the SHIPPED path: a hull under its own rated
acceleration and cruise, driven through the movement contract's own lane (acceleration and torque per
tick, in the child's own frame). This is what a player does, so this is what SL8 judges. Its verdicts are
V3 (no hole), V9 (no pop) and **V14 (the realm crossing)**.

> **V14 exists because the last revision never named the one seam this architecture really has.** An eye
> at 10 000 km is not inside the planet's realm; an eye at 1.8 m is. Somewhere on the path the occupant
> RE-HOMES from the star system into the planet — the transfer machinery, HR2, and exactly the moment a
> player would see a seam. §9 states the verdict.

**Marked OWED, with its blocker named.** The stands are placed by `VD_SPAWN_POSES`, which is a SPAWN and
not a flight (`D-TERRAIN-4` 🟥). A re-spawn per frame is a teleport, and SL8 calls a teleport a seam, so
a pop detector fed by teleports measures the teleports. **The flight recording lands when a hull can be
flown down under the movement contract inside the picture gate.** UNMEASURED: how long our own rated
descent takes, and therefore how many frames the recording holds.

**(2) THE RUNG SWEEP — a rung-coverage instrument, and it is named one.** The fractional descent stays,
under an honest name. Its only job is to cross every rung boundary in one file, so V5's rung halves and
V8's ladder half have a subject. It is not the SL8 gate and it is not a player's view.

```text
   from 10 000 km to 1.8 m is ln(10 000 000 / 1.8) = 15.53 e-folds
   at 1 % of the altitude per frame:  15.53 / ln(1/0.99) = 1 545 frames = 51.5 s at 30 fps
   (the last revision printed 1 553: it divided by 0.01 instead of by ln(1/0.99))

   10 000 km   the whole body in frame, a disc 451 px across; the ladder's TOP rung
        |      the altitude falls 1 % per frame, so every frame looks like the last one, scaled
    2 000 km   the disc is 1 396 px across; the limb leaves the frame's sides
      300 km   the horizon is 1 449 km away; the terminator crosses the frame
      100 km   the horizon is 825 km away; the largest landforms resolve
       10 km   the horizon is 260 km away; the tilt of the plain resolves
        1 km   the horizon is 82 km away
      373 m    the drone diagnostic's altitude; the horizon is 50.0 km away
        1.8 m  the pilot stands; the horizon is 3 473 m away
```

**What the sweep costs on disk, priced.** Every recorded frame writes a PNG **and** a pretty-printed
`DevState` JSON dump — the capture path makes no exception for a record sequence
(`crates/bins/src/bin/client.rs:1231-1240,1332-1372`; `write_state_dump`, `:1391`). So 1 545 frames are
1 545 PNGs plus 1 545 state dumps, and the dumps grow when the stamp joins them. The frame count is
inside the harness's own cap (`MAX_RECORD_FRAMES = 3 600`, `crates/bins/src/bin/client.rs:705`), so the
run will not truncate — it will fill the disk quietly. The probe is written on about twenty stamped
frames only (111 MB), never on all 1 545 (8.5 GB). **M8-11 measures the whole run's bytes before it runs
on the owner's machine.**

### 8.3 What the two recordings prove

| Recording | Verdict | What it proves |
|---|---|---|
| The flight | V3 | A chunk that arrives late is a hole in the ground, and a descent at a REAL rate is exactly the case that makes chunks arrive late. |
| The flight | V9 | No pop, at the speed a player actually travels. SL8 is about what a PLAYER sees. |
| The flight | **V14** | The realm crossing is invisible: the frame before and the frame after differ no more than the smooth-motion distribution M8-5 measures. |
| The sweep | V5 (a), (b) | Every rung boundary crosses the frame at least once, so the seam verdicts have a subject. |
| The sweep | V8 | The skyline at 10 km and at 1 km is the same skyline at two rungs; their difference stays inside `dropped_bound_m` (`crates/terrain/src/body.rs:274`) projected to pixels. |
| The sweep | V2 | The horizon grows as the radius says, every frame. |

**Neither recording can run until `D-TERRAIN-3` is deleted** (one rung per realm,
`crates/client/src/chunks.rs:335-341`). Slice 8 owns that. The flight also waits on `D-TERRAIN-4`.

---

## 9. THE VERDICTS A MACHINE HOLDS

Each verdict states its inputs, its bound, where the bound comes from, and where it is INAPPLICABLE. A
verdict with a fitted bound is not a verdict; where a bound must be measured first, the measurement is
listed in §11 and the verdict does not run until it exists.

**THE TOLERANCE RULE, stated once and used everywhere.** No verdict uses `relief_bound_m` as a
tolerance. That number sums the live amplitudes and the amplitudes are scaled so their SUM IS THE RELIEF
(`crates/terrain/src/body.rs:180-186,263-271`), so it is the height the surface would reach if all
fourteen octaves aligned — which the field never does. §0.2 M-D shows it is 277 px against a 69.4 px
signal at 300 m: a tolerance that cannot fail. **Every tolerance below is instead three times the body's
own MEASURED RMS wobble at the view length, projected to pixels — about four to six pixels at every
scale** (§0.2 M-D). It is a fact about this body, it moves when the landform slices move the body, and
it can fail today.

### V1 — The ruler is the right size

**Inputs.** The subject's delivered geometry (an avatar's `OCCUPANT_FIGURE_EXTENT_M`, or a box's three
half-extents and facing once the row carries them, §5.4), its delivered distance, the camera, and its
painted pixel extent from the probe's id buffer.
**Verdict.** `|measured radius px − predicted radius px| ≤ straddle drift px + 1`.
**Bound's source.** The straddle already reports its own drift in pixels (`crates/bins/src/pixel.rs:113`).
The `+1` is the pixel grid.
**Inapplicable.** For a marker past 144.9 m (§5.2), and for any translucent subject until the id buffer
states a rule for blending. The verdict refuses to run and says so.

### V2 — The picture reaches as far as the world says

**Inputs.** The eye altitude (stamp), the body's ladder radius, the body's measured wobble table, the
skyline row per column and the farthest terrain distance (both from the probe).
**Verdict, two halves.**
(a) **The residency self-check:** the farthest terrain distance in the probe is at least the resident
radius minus one chunk edge at the coarsest rung drawn. **This half is a SELF-CHECK and it says so in
its own text** — the residency chose the picture's reach, so both numbers come from one `chunks_around`
decision (`crates/client-render/src/terrain.rs:338`; `crates/client/src/chunks.rs:364,390`).
(b) **The world check, and the real verdict:** for each column, the skyline row lies within the measured
tolerance of the row the geometric horizon projects to. The horizon's row comes from the body's radius
and the eye's altitude; the skyline's row comes from the pixels. **That pair is two sources.**
**Bound's source.** Three times the body's measured RMS wobble at the view length, projected: **4.13 px
at a 1.8 m eye, 5.67 px at 300 m** (§0.2 M-D).
**What it catches, MEASURED.** On the slice-7 hill picture (300 m up, a 3 224 m patch) the skyline sits
**69.4 px** below the horizon row against a 5.67 px tolerance, and (b) FAILS by twelve times. On the
slice-7 ground picture it sits **3.0 px** below against 4.13 px and (b) passes — which is the honest
answer, because at a 1.8 m eye the missing three kilometres really are three pixels (§3.2).
**What changed.** The last revision set this tolerance from `relief_bound_m`, which is 277 px at 300 m,
**so the verdict as specified REPORTED A PASS on the very picture the document exists to refuse.**
Refuter A found it. It is the most serious defect either round found, and it is fixed here.
**The residency this verdict compares against is NOT the geometric horizon.** See the residency rule in
§11: high ground beyond the horizon rises into view, so a picture that stops at the horizon hides every
mountain, and half (a) would pass it.

### V3 — No hole in the ground

**Inputs.** The probe's id buffer, the skyline row per column, the projected horizon row.
**Verdict, two halves, both counts that must be zero.**
(a) The count of unauthored pixels strictly BELOW the skyline row.
(b) **The count of columns that hold NO terrain pixel at all below the projected horizon row.**
**Bound's source.** None needed; both are counts that must be zero.
**Why (b) exists.** Refuter A found two blind spots in the last revision's single half. A column with no
terrain pixel has no skyline row, so half (a) skips it silently — and that is every column at the
frame's edge during a descent, and every column where a whole chunk column is late. Half (b) catches
both, and it needs no tolerance.
**What is lawful.** The clear colour is deep-space black (`crates/client-render/src/lib.rs:78`), there is
no sky dome, and the star sky is a point cloud — so the space between stars is lawfully unauthored, and
the whole sky is. A frame-wide "no unauthored pixel" verdict is **owed to the atmosphere**.
**Note.** A cave mouth is not a hole: a cave's inside is terrain and carries the terrain id. A hole is
the absence of an author below the skyline.

### V4 — The horizon curves like a sphere of this radius

**Inputs.** The skyline row per column across the frame, the eye altitude, the radius, the camera.
**Verdict.** The skyline's LOWER envelope (the 10th-percentile row per column band) fits the projected
horizon ring of a sphere of radius `R` at altitude `h`, with a mean absolute deviation under the measured
tolerance.
**Bound's source.** The ring is exact geometry; the tolerance is three times the body's measured wobble
at the view length, about 5.3 to 5.7 px at every altitude this verdict runs at.
**Where it applies — MEASURED by exact projection** (the sag between the frame's centre column and its
edge column, camera nosed down by the dip):

```text
  eye height     sag across the frame     tolerance     verdict can fail?
      1.8 m         0.22 px                4.13 px           no
    300   m         2.81                   5.67             no
      1.2 km        5.67                   5.67             borderline
      4.9 km       11.34                   5.67             YES (twice the tolerance)
     60   km       40.10                   5.26             yes
    300   km       92.64                   5.3              yes
  2 000   km      319.28                   5.3              yes
 10 000   km      UNDEFINED — the ring never reaches the frame's edge column
```

**The verdict runs above 4 869 m of eye altitude**, where the sag reaches twice the measured tolerance
(MEASURED by bisection), **and below 3 144 km**, where the horizon ring stops reaching the frame's edge
(MEASURED by bisection). Outside that band it is INAPPLICABLE and says so.
**What changed.** The last revision said the verdict runs above 151.6 m, where the sag passes 2 px. That
answered "where is the sag two pixels", not "where can this verdict fail": at 151.6 m the tolerance was
about 390 px against a 2 px signal. Refuter A is right, and the low end moves by a factor of thirty-two.
**Above the upper limit the picture's geometric fact is the LIMB'S OWN RADIUS**, and here too the last
revision was wrong. It printed 588 px at 2 000 km and 220 px at 10 000 km, which is the angular radius
divided by the pixel angle — a small-angle formula used at 38.8°. The EXACT rectilinear projection,
`f · tan(asin(R/(R+h)))` with `f = 869.12 px`:

```text
  altitude      limb angular radius   exact px   the disc across   fits a 1280 px frame?
    300 km          66.610 deg          2 009.4      4 019 px            no
  2 000 km          38.772               698.1      1 396 px            no
 10 000 km          14.535               225.3        451 px            YES
```

**So the 2 000 km still is DROPPED and the whole-body stand moves to 10 000 km** (§7.3 stand 17). The
last revision promised *"the same where the whole limb is in frame"* at 2 000 km, which its own §8.2
already contradicted. Refuter A found it.

### V5 — No visible edge

**Inputs.** The probe's id and distance buffers; the projected line of the face seam or the rung
boundary.
**Verdict, three parts.**
(a) **No hard edge in distance:** along the projected seam, the distance buffer's step across the seam is
under one cell of the rung drawn there.
(b) **No hard edge in the skyline:** the skyline row's step across the seam is under the measured
tolerance (three times the body's RMS wobble at the view length, projected).
(c) **No straight luminance line:** a line detector over the colour frame finds no straight edge within
2 px of the projected seam that is not also present in the distance buffer.
**Bound's source.** The cell size at the rung, the body's measured wobble, the pixel grid.
**Inapplicable.** Where no seam and no rung boundary projects into the frame. **A rung boundary cannot
project into any frame until `D-TERRAIN-3` is deleted** (`crates/client/src/chunks.rs:335-341`), so
part (a) and part (b) run from slice 8.

### V6 — The haze grows with distance and never jumps

**Inputs.** The probe's distance buffer, the colour frame, the sky's colour.
**Verdict.** Bin the terrain pixels by distance. The mean contrast against the sky must fall
monotonically across the bins, and no bin-to-bin fall may exceed a stated multiple of the median fall.
**Bound's source.** UNMEASURED until the atmosphere exists. The multiple is set by M8-4 (§11), measured
on the first atmosphere and then frozen with its measurement recorded.
**Inapplicable today.** There is no atmosphere and no sky (slice 7 §6). **No slice owns the atmosphere
yet**, which §16 Q6 hands to the owner. The probe reserves an `atmosphere` id for the day it lands
(§6.2).

### V7 — The picture shows the biome the stamp names

**Inputs.** The stamp's biome, the probe's terrain pixels, the client's biome-to-paint table.
**Verdict.** The dominant paint class in the near band (inside one chunk edge) is the paint the client's
table gives for the stamp's biome.
**Bound's source.** The client's own table — so this checks the CHAIN (generator → stamp → paint), not a
colour taste.
**INAPPLICABLE TODAY, and owed.** **There is no paint table.** The terrain is one material
(`crates/client-render/src/terrain.rs:362`) and neither client crate names a biome. So the polar-cap,
desert and highland stands (6, 7, 8) each paint the same beige until a paint table lands.

### V8 — The far ground is a mesh, and its scale matches the world

**Inputs.** The stamp's drawn horizon, the probe's id and distance buffers.
**Verdict, two halves.**
(a) The maximum distance over terrain pixels agrees with the stamp's drawn horizon within one chunk edge.
(b) **The MESH-INSTANCE field names several distinct meshes across every far terrain band.** Ruling V10
refuses impostors, and this is the honest test: a card is ONE mesh instance across a whole band; a chunk
band is many.
**Bound's source.** The chunk edge; the id set is exact. The count of distinct instances a band must hold
is derived from §11's own column count for that band, not typed.
**What changed twice.** The first draft tested for an impostor by the distance histogram's spread, which
is weak in both directions. The last revision replaced it with "the id names a real chunk mesh" — but its
own §6.2 defined the id as an AUTHOR, under which every terrain pixel carries one value and the test is
unmeasurable. Refuter A found the contradiction. §6.2 now splits the `u16` into an author and a mesh
instance, and this verdict reads the second field.

### V9 — No pop across a recording

**Inputs.** The recorded frames, their manifest rows, and the delivered poses.
**Verdict.** For every frame `i`, the **second difference in time** of the mean absolute per-pixel
change — `|(F(i) − F(i−1)) − (F(i−1) − F(i−2))|` averaged over pixels — must not exceed a factor `F`
times the median of its neighbours.
**Why not the differing-pixel count.** `differing_pixel_count` is an exact byte-inequality count whose
own doc names its purpose as a STILL scene (`crates/client-harness/src/assert.rs:98-105`). On a moving
camera nearly every terrain pixel changes by at least one least-significant bit, so the fraction sits at
about 1.0 on a smooth frame AND on a frame with a rung pop. A verdict that cannot fail is worse than no
verdict.
**Which recording it judges.** THE FLIGHT (§8.2), because SL8 is about what a PLAYER sees, and a player
flies a rated hull. It also runs on the rung sweep, where it is a diagnostic and not the gate.
**Bound's source. MEASURED FIRST, then frozen.** M8-5 (§11) records the distribution on a flight in which
the rung never changes. `F` is set above that distribution's maximum.
**Inapplicable.** On the first two frames, where the second difference is undefined.

### V10 — The stand really stands on the ground (a SEATING check, not a no-drift measurement)

**Inputs.** The delivered pose, the client's own surface height at that direction.
**Verdict.** `‖delivered position‖ − height_m(body, dir, 0)` equals the stand's stated eye height within
one seating quantum (an eighth of a cell, ruling S6-7).
**What it is NOT.** The GATE computes the surface itself and hands it to the server as a literal through
the `VD_SPAWN_POSES` stand-in: `let h = vd_terrain::height::height_m(&body, dir, 0).to_f64();` then
`stand(d, EYE_HEIGHT_M, h, …)` then `spawn_entry(...)`
(`crates/bins/tests/terrain_pictures.rs:457-471`; `D-TERRAIN-4` 🟥). So the chain is `height_m` in the
gate process → a literal → the server → the wire → the client → `height_m` in the client process: **the
same crate, the same build, the same chip.** That is the star-sky self-check this document quotes in
§4.1, and it agrees to the last bit whether or not two hosts agree.
**What would make it a no-drift measurement, stated as two preconditions.** (1) Ruling V11's server-side
seating lands, so the SERVER computes the surface. (2) The picture runs on a cluster whose shard sits on
a different chip from the client. Until both hold, **the golden table (`D-TERRAIN-1`) remains the only
no-drift gate — and that gate is itself 🟧, with its x86-64 leg EMULATED** (§7.2, §14).

### V11 — THE VISIBLE PROMINENCE (the flagship's own verdict) — NEW

**Inputs.** The skyline row per column, the projected horizon row, and the probe's distance buffer.
**Verdict, two numbers reported and one bound.**
(a) **The count of skyline pixels that stand ABOVE the projected horizon row** by more than the pixel
grid. Each one is ground beyond the horizon, rising into view — a landform.
(b) **The largest such rise, converted to metres of prominence** at its own measured distance:
`h' = (d − d_horizon)² / 2R + (row rise) × d × 0.0011506`.
**The bound is the reference picture's own demand**, printed beside the picture: from a 1.8 m eye, ground
at 50 km needs 323 m and ground at 100 km needs 1 390 m (§0.2 M-B).
**What it does today.** It reports a number near zero, because the home planet is a tilted plain. **That
is the point.** This is the one verdict in the set that FAILS the world we have, and it fails it with a
number the landform domains can aim at.
**Why it exists.** The last revision moved the flagship to a 373 m eye to "get 50 km of ground". Refuter
B showed that reads the reference picture backwards: its 50 km comes from RELIEF, not from altitude. The
flagship returns to the ground, and this verdict is what makes a flat world fail it instead of being
flown over.

### V12 — The desert sits in a circulation band, not on the equator — NEW

**Inputs.** The biome of a stated sample of the body's addresses, and each address's latitude.
**Verdict.** The median |latitude| of the `Desert` addresses lies inside the body's own stated descending
band, and the median |latitude| of the `Tundra` addresses lies outside its own stated band.
**Bound's source.** The Hadley cell's descending latitude, which follows from the body's rotation rate
and radius. **The body states no spin today**, so the BAND is OWED to the spin slice. **The MEASUREMENT
runs today, and it already refuses the body:**

```text
   temperature = 1 − |z| + 0.35·n_t − 0.3·above_sea/(highland_above_m + 1)   height.rs:50-52
   |z| is sin(latitude), not the latitude.

   the typical column stands 5 297 m over the sea (the sea offset is −5 297 m and the surface's
   mean is the ladder radius), and highland_above_m = 7 867.7 m, so the height term is
   0.3 × 5 297 / 7 868.7 = 0.202 — NOT small.

   Tundra needs temperature < 0.35  ->  |z| > 0.448 + 0.35·n_t  ->  typically |z| > 0.448
                                     ->  the tundra begins at 26.6 deg of latitude
   Desert needs temperature > 0.75  ->  |z| < 0.048 + 0.35·n_t  ->  typically |z| < 0.14
                                     ->  the desert exists only within 8.2 deg of the EQUATOR
```

**Three cross-checks that could have failed.** `|z| > 0.448` covers 55.2 % of a sphere, and the highland
takes 14.3 % of the body first: measured Tundra 45.5 %. `|z| < 0.14` covers 14.0 %, times the humidity
condition's ~36 %: 5.1 %, against measured Desert 3.1 %. And `above_sea > 7 868 m` is 1.12 standard
deviations over the mean at a standard deviation of 2 295 m: 13.1 %, against measured Highland 14.3 %.
The model reproduces all three.

**What this says to the owner, in the game's words.** On the home planet the tundra begins at the
latitude of Florida, and the only deserts sit on the equator, where a real world puts rainforest. **That
is the exact opposite of the Hadley cell.** Refuter B found the mechanism; this document corrects the
numbers, because the refuter's version dropped the height term and put the tundra line at 40.5° instead
of 26.6°. **This is the cheapest believability verdict available, it costs one pass of the sample §10.2
already takes, and it is the owner's own first sentence.**

### V13 — Biomes follow the body's place, over the world's own bodies — NEW

**Inputs.** The biome shares of several planets of one system, and each planet's orbital distance and
its star's luminosity (`crates/physics/src/worldgen/generate.rs:193-207,1206,1227`).
**Verdict.** Rank the system's planets by their INSOLATION (luminosity divided by the square of the
orbital distance). The tundra share must fall monotonically as insolation rises, and the desert share
must rise.
**Bound's source.** The ordering itself — a monotone rank test needs no tolerance.
**Why it is an ORDERING and not a pair.** The last revision asked for two pictures of two bodies, with
the verdict *"the difference must move when the orbital distance moves"*. **Nobody may move a planet's
orbit: SL5 gives one world and the seed draws the orbit.** And two bodies already look different today
for an unrelated reason — each draws its own `temperature_seed` and `humidity_seed`
(`crates/terrain/src/body.rs:203-215`) — so a difference verdict would go GREEN on a generator that
reads nothing the owner named. Refuter B is right on both counts. An ordering over many bodies of THE
world runs on the shipped world and it can fail.
**INAPPLICABLE TODAY, and owed.** `biome_at` reads no insolation, no spin, no obliquity and no gravity
(`crates/terrain/src/height.rs:38-65`). The verdict is written now so the slice that lands it inherits
its gate.

### V14 — The realm crossing is invisible — NEW

**Inputs.** The flight recording's frames, and the transfer's own record of the tick at which the
occupant re-homed.
**Verdict.** The frame before the crossing and the frame after it differ by no more than the smooth-motion
distribution M8-5 measures — the same instrument V9 uses, applied at one named instant.
**Why it exists.** An eye at 10 000 km is in the star system's realm; an eye at 1.8 m is in the planet's.
Somewhere on the descent the occupant crosses, which is the transfer machinery (HR2) and **the one moment
a player would actually see a seam in this architecture**. The last revision's approach judged rung pops
only and never named the crossing. Refuter B found it.
**Inapplicable** until the flight recording exists (§8.2), which waits on `D-TERRAIN-4`.

### V15 — The weather is SHIPPED, not invented — NEW, owed

**Inputs.** Two clients, at the same stand, at the same universe tick, each with its own probe.
**Verdict.** The two cloud masks agree within the straddle drift.
**Why it is the weather's most important verdict.** It is the only one that proves the SL6 ask of §4.4
was needed. A client that invents its own cloud passes every other weather verdict and fails this one.
**Inapplicable** until weather is shipped.

### V16 — The weather MOVES — NEW, owed

**Inputs.** The same stand at two universe ticks a stated interval apart.
**Verdict.** The cloud mask is displaced by the shipped wind times the interval, within the measured
tolerance.
**Why a still cannot accept a simulation.** Nothing in this section had a TIME axis except V9, which
detects pops. The owner asked for a SIMULATION, and a simulation is judged across ticks. Refuter B found
the gap.
**Inapplicable** until weather is shipped.

### V17 — The wind agrees with the body's spin — NEW, owed

**Inputs.** The mean cloud displacement direction from V16, and the body's stated spin axis and rate.
**Verdict.** The displacement turns with the Coriolis deflection the body's own spin implies, in the
right hand.
**Inapplicable** until the body states a spin AND weather is shipped. Two owners, both named.

### V18 — The windward side is wetter than the leeward side — NEW, owed (CLIMATE, not weather)

**Inputs.** The two stands of §7.3 stand 15, and the client's paint table.
**Verdict.** The mean paint of the windward stand's near band differs from the leeward stand's by a
stated amount, in the wet direction.
**Bound's source.** UNMEASURED, and probably refused by the body itself: a visible rain shadow needs
about 1 000–1 500 m of rise inside 10–20 km, and this body rises 1 000 m across 50 km (§2, orographic
lift; §7.3 stand 15). **The weather domain learns on day one whether this body can show a rain shadow at
all**, which is exactly what refuter B asked this document to state.
**Inapplicable** until a paint table AND a prevailing wind exist.

### V19 — The vegetation is placed lawfully, and it is not an impostor — NEW, owed

The reference picture is, by area, mostly forest, and the last revision wrote fifteen stands and ten
verdicts and left the picture's largest object with none. Refuter A found it. Ruling V4 makes trees,
grass and decoration ART ASSETS placed by a SERVER SKELETON from the biome, so the skeleton raises three
questions this domain owns:

- **(a) SL10.** The skeleton's placement is a function of seed and address only, and two hosts compute it
  byte-identically — or it crosses as a one-hop diff. The verdict is a digest comparison, exactly like
  `D-TERRAIN-1`'s.
- **(b) The chain.** The drawn stem density in the near band agrees with the density the biome the stamp
  names implies. This is V7, one level up.
- **(c) No impostor.** Every far forest band names several distinct MESH INSTANCES in the probe's id
  buffer. A far forest is where an impostor is most tempting, and V8's instrument is the only one against
  it.

**Inapplicable** until slice 14. Re-taking a stand after slice 14 is a COMPARISON, not a verdict; these
three are the verdicts.

### The sentinels that already exist and stay

Zero magenta (`magenta_pixel_count`), content present (`content_present_fraction`), no NaN in a float
readback (`nan_count_f32`), and the presence law (`crates/bins/src/pixel.rs:57-65`).

### 9.4 The numbers this section rests on — MEASURED

```text
  R = 3 350 759 m (the home planet's ladder radius, crates/terrain/src/home.rs)
  camera 45.0 deg vertical over 720 rows; 1280 x 720; focal length 869.12 px
  one pixel = 0.06592 deg = 0.0011506 rad; one pixel at distance d is 0.0011506 * d metres

  eye height    horizon        dip        sag (exact ring projection)   tolerance (3x wobble)
      1.8 m      3 473 m     0.059 deg       0.22 px                       4.13 px
    300   m     44 839 m     0.767           2.81                          5.67
     60   km   636 939 m    10.763          40.10                          5.26
    300   km  1 449 295 m   23.390          92.64                          ~5.3
  2 000   km  4 171 695 m   51.228         319.28                          ~5.3
 10 000   km 12 923 435 m   75.465         UNDEFINED (limb radius 225.3 px, disc 451 px)

  the sag reaches the tolerance at         1 218 m of eye height   (bisection)
  the sag reaches twice the tolerance at   4 869 m of eye height   (bisection)  <- V4 starts here
  the ring leaves the frame at             3 144 km of eye height  (bisection)  <- V4 stops here
```

The horizons and the dips reproduce `d = √(h(h + 2R))` and `dip = acos(R/(R + h))` exactly. The sags are
projected, not approximated: the horizon ring is the cone of half-angle `90° + dip` about the local up,
and the sag is the row difference between the frame's centre column and its edge column. **I re-derived
every sag row independently for this revision and reproduced the last revision's table to the printed
digit (0.22 / 2.81 / 40.10 / 92.64 / 319.28), and refuter A reproduced it a third time.** The limb rows
are the ones that were wrong, and they are corrected in V4.

---

## 10. WHAT ONLY THE OWNER JUDGES, AND THE NUMBERS THAT RIDE BESIDE THE PICTURE

### 10.1 The three words, and what a machine can and cannot do with them

| The owner's word | What a machine can hold | What only the owner holds |
|---|---|---|
| **Vast** | the picture reaches as far as the world says (V2); the far ground is a mesh (V8); **the count of prominences above the horizon (V11)** | whether it FEELS far |
| **Interesting** | the slope histogram's tail; the hypsometric curve's shape; **the visible-prominence count (V11)**; the tilt-versus-wobble split of the relief profile | whether there is somewhere to go |
| **Believable** | the horizon curves right (V4); no seam (V5); the haze is monotone (V6, owed); no impostor (V8); **the desert's latitude against a circulation band (V12)**; **the biome order over the world's bodies (V13)** | whether it looks like a world |

**What changed.** The last revision offered the owner only geometry under "believable" — the ring fit,
the seam, the haze. Refuter B is right that a climatologist refuses this planet on sight and that the
document held the tools to say so. V12 and V13 are now in the table, and V12 runs today.

### 10.2 The four numbers printed beside every ground stand

These are computed by the generator on the CPU, with no GPU and no picture.

**(a) The relief profile, split into TILT and WOBBLE.** The surface height along the nose, out to the
drawn radius, sampled at rung 0. Printed as a strip beside the picture, with a straight line fitted and
REMOVED, because the two halves look nothing alike:

```text
  ground-01, along the nose, 0 -> 50 000 m
  raw           tilt 1.15 deg  (the ground falls 1 004 m across the view)
  after the fit  RMS wobble 97.5 m, largest departure 214 m
   +200 |        __                                 _
      0 |__/\__/  \___/\____/\__________/\____/\__/  \____
   -200 |
        +---------+---------+---------+---------+---------
        0       10 km     20 km     30 km     40 km   50 km
```

A self-similar swell draws a straight tilted line with a small wobble on it — MEASURED, that is what the
home planet draws (§3.3). A carved landscape draws a V for a valley, a step for a plateau, a spike for a
ridge. **The owner reads the shape of the world in one line, without a GPU, and the split says at once
whether he is looking at a shape or at a slope.**

**(b) The slope histogram** over the drawn disc, in bins whose top is the body's own measured p99 slope,
so the axis is a fact about the body and not a typed number. MEASURED today: median 1.37°, p99 5.06°,
largest 7.23° (§3.3). The number is reported, not argued.

**(c) The hypsometric curve** — the fraction of the surface at each height over the sea. It is a STATED
SAMPLE at rung 0: 1 000 000 ladder addresses through the co-prime stride, fourteen octaves each, at about
390 ns per column ≈ **0.39 s single-threaded** (ESTIMATED; M8-8 measures it). It is NOT a coarse-rung
pass: `octaves_at(11)` keeps only `14 − 11 = 3` octaves (`crates/terrain/src/body.rs:251-258`), so a
curve taken there is the hypsometry of a three-octave surface and is biased by construction, and a full
rung-11 pass is `6 × 2 570² ≈ 39.6 million` columns.

MEASURED today, from 4 000 directions: one hump, standard deviation **2 295 m**, **0.92 %** under the
sea. Earth's two steps come from two kinds of crust (isostasy) and a 71 % ocean.

**(d) THE BIOME LATITUDE HISTOGRAM — new, and it is the one the owner asked for.** The share of each
biome in each 5° band of latitude, from the same sample as (c). It is what V12 reads, and it already says
this body puts the tundra at 26.6° and the desert on the equator (§9 V12). One pass, one strip, and the
owner sees his own first sentence answered or refused.

### 10.3 The comparison over time

A stand recipe is stable. So the picture set is re-taken at every terrain slice, and the owner sees the
same places change. That is the golden-image discipline without golden pixels: **the recipe is golden,
the picture is not.** Pictures are stored under
`docs/investigation/<date>/<slice>/pictures/<stand>.png`, with the state file beside them. The co-prime
stride (§7.2) is what makes this honest: a raster scan would compare one corner of one face forever.

---

## 11. THE MEASUREMENTS OWED FIRST

Nothing in §12 is built before these run. Each says what would make it fail.

| # | What | How | What it decides |
|---|---|---|---|
| **M8-1** | The stand search's cost: the stride, the address count scanned before the first hit, and the milliseconds — **measured on a SCATTERED `height_m` call, not on an amortised column pass** | a bench in `vd-terrain` | whether the search runs in the gate or is cached per body per recipe version. Every microsecond in §7.1 is ESTIMATED until this runs |
| **M8-2** | The picture probe's cost: milliseconds and bytes per capture, and whether the extra render target changes the colour frame by one pixel | the capture path, twice, with `differing_pixel_count` between the two colour frames | the probe must not change what the owner judges |
| **M8-3** | The stand search's CROSS-HOST determinism: the found addresses of every stand, byte for byte | extend the `D-TERRAIN-1` golden table — **and note that D-TERRAIN-1's own x86-64 leg is EMULATED** | without a real x86-64 leg, both this and the golden table are UNMEASURED on the target that matters |
| **M8-4** | The haze's measured profile on the first atmosphere | the vista stand with the atmosphere on | V6's multiple |
| **M8-5** | The second-difference distribution on a FLIGHT where no rung changes | one flight recording with the rung pinned | V9's factor `F`, and V14's crossing bound |
| **M8-6** | The residency every stand needs, in chunks, bytes, triangles and SECONDS | arithmetic over the ladder, checked against `terrain_cost` | the demand this set puts on the ladder slice (the table below) |
| **M8-7** | The wall-clock cost of the whole picture set: **logins (three for calibration, one per other stand), cluster boots, GPU seconds** | the gate, once | whether the set runs per slice or per milestone |
| **M8-8** | The hypsometric and biome sample's cost at rung 0 for a stated sample count | a bench in `vd-terrain` | §10.2(c) and (d)'s sample count |
| **M8-9** | **The body's own p99 RISE above the horizon plane at each range** — the number the residency rule below solves against, and the number V11 compares to M-B's threshold curve | a bench in `vd-terrain` over strided addresses | the drawn radius of every ground stand, and the landform domains' target |
| **M8-10** | **ONE slope-and-wobble bench both domains cite**, so 08's two-point slope and 01's ring tilt stop being two answers | a bench in `vd-terrain`, with both quantities named apart | §3.3's reconciliation, which is stated and not proved today |
| **M8-11** | The rung sweep's total bytes: 1 545 PNGs plus 1 545 state dumps plus 20 probes | one run, on a machine with room | whether the sweep can run on the owner's own Mac |

### M8-6, PRICED NOW for every stand, because it gates the whole set

**Rule 1 — how far each rung is drawn.** Draw each rung out to where its cell subtends `k` pixels:
`d_r = 2^r / (k · 0.0011506)`. Then the columns in one annulus are the SAME at every rung:
`0.75π / (62 · k · 0.0011506)²` — **115.7 columns at k = 2**, 28.9 at k = 4.

> **`k` IS AN OPEN OWNER DECISION, and this document borrows it.** Sibling document
> `01_reference_target.md:1543` hands the owner decision **D15** with the options priced: 1 px, 2 px or
> 4 px. This table uses k = 2 and prints k = 4 beside it, because **the whole price falls by about four
> when the owner answers (c)**. A gate that swings four times on an unanswered question is not a gate.
> Refuter B is right, and the byte half of GATE B waits on D15. (Note also that domain 01 computes with
> 1 080 rows and this document with the shipped 720, so its pixel angle is 1.5× smaller; the two cost
> tables are not directly comparable.)

**Rule 2 — how many chunks a column holds. It follows the RUNG, and the code says why.**
`surface_chunk_span` returns `(lo − 1, hi + 1)` clamped to the band's top chunk index
(`crates/terrain/src/digest.rs:89-130`). So a column holds three chunks wherever the band is three chunks
tall, and fewer where it is not. The home planet's band is about 28 800 m
(`crust = relief + strata + caves + 64`, `above = relief + 64`, `crates/terrain/src/body.rs:231-236`):

```text
   rung  cell     cells in the band   chunks in the band   chunks per column
    0     1 m         28 800                 465                  3
    3     8 m          3 600                  58                  3
    7   128 m            225                   4                  3
    8   256 m            112                   2                  2
    9   512 m             56                   1                  1
   11  2 048 m            14                   1                  1
```

(Refuter B is right that the last revision's flat "×3 at every rung" is wrong, and wrong about the
mechanism: it is the BAND's own height that caps the count, not the local relief.)

**Rule 3 — how far the residency must reach. NOT the geometric horizon.** High ground beyond the horizon
rises INTO the picture, which is the reference picture's whole far half. The drawn radius solves

```text
   d  =  √(2R·h_eye)  +  √(2R·P(d))          P(d) = the body's p99 rise above the horizon plane at d
```

On the home planet, using the measured wobble as a LOWER bound for `P` (`P ≈ 3 × 0.00217 · d`):

```text
   a 1.8 m eye     the geometric horizon is  3 473 m   the visible-relief radius is  ~50 400 m   (14x)
   a 373 m eye     the geometric horizon is 49 998 m   the visible-relief radius is ~123 500 m   (2.5x)
```

**So even on a near-flat body a standing pilot must be shipped terrain to about 50 km, not 3.5 km.** This
is UNMEASURED in its exact form — the wobble is a departure from a fitted line, not from the eye's own
horizon plane — and **M8-9 measures `P` properly.** Until it does, the visible-relief rows below are a
LOWER bound. Refuter B found the rule; this document refuses the refuter's own tolerance
(`relief_bound_m`, which would demand `√(2R·14 305) = 309 km`) for the same reason V2 refuses it.

```text
  k = 2 px (D15 option b)
  stand                          columns   chunks    mesh      triangles   1 thread   14 threads
  ground 1.8 m, horizon only        502     1 506    610 MB     25.0 M      18.3 s      1.3 s
  ground 1.8 m, visible relief      935     2 805  1 137 MB     46.6 M      34.2 s      2.4 s
  drone 373 m, visible relief     1 089     3 134  1 270 MB     52.1 M      38.2 s      2.7 s
  orbit 300 km    (rung 9)        5 752     5 752  2 331 MB     95.6 M      70.1 s      5.0 s
  orbit 10 000 km (rung 11)       3 277     3 277  1 328 MB     54.4 M      39.9 s      2.9 s

  k = 4 px (D15 option c) — the same stands
  ground 1.8 m, horizon only        154       463    188 MB      7.7 M       5.6 s      0.4 s
  drone/vista to 50 km              263       767    311 MB     12.7 M       9.3 s      0.7 s
  orbit 300 km    (rung 10)       1 438     1 438    583 MB     23.9 M      17.5 s      1.2 s
```

**Where every number comes from.** The bytes and triangles per chunk are the repository's own MEASURED
M7-3 (`docs/investigation/2026-09-07/slice_07_client_link.md:236`: 405 228 bytes, 16 615 triangles, and
*"a chunk is 62³ cells at every rung, so the bytes per chunk are the same order at every rung"*). The
seconds are 8 ms of generator budget (ruling V10) plus 4.18 ms of client geometry per chunk on one thread
(M7-2, `slice_07_client_link.md:234`), so 12.18 ms per chunk, over M7-2's own 14 threads. The orbit
columns are the visible cap `f = h / (2(R + h))` times the body's area, divided by the chunk edge squared
— **and the last revision's 6 793 / 1 978 / 3 964 were 18–21 % above what that method gives.** Refuter B
recomputed them; this table adopts the corrected 5 752 / 1 635 / 3 277 and drops the 2 000 km still
(§9 V4).

**The gate's own terrain deadline is 90 s** (`TERRAIN_WAIT_TICKS = 1 800` at 20 Hz,
`crates/bins/tests/terrain_pictures.rs:63`). The 300 km orbit stand sits at 70 s single-threaded, inside
the deadline only because the workers are threaded. **And 2.3 GB of mesh is a MEMORY statement, not a
disk one:** Bevy holds the mesh asset and the GPU buffer, so an orbit picture 451 pixels across asks a
developer's machine for gigabytes of vertex data. On the owner's own swap-starved Mac that is the
difference between a picture and a wedged run.

**Two demands on the ladder slice, and they are different demands.**

1. **The ground and vista stands are a BYTE problem.** The cross-check: M7-3 already publishes *"a
   13 × 13 × 3 patch is 507 chunks ≈ 205 MB"* — that patch reaches 403 m, so an honest ground picture is
   five and a half times it. Either the residency band, the vertex packing, or `k` must move.
2. **The orbit stands are a LADDER problem, and the two reasons are structural.**
   - **The top rung is a PACKING RULE, not an oversight:** *"The top rung is the first rung with at most
     this many chunks along a face edge"*, `TOP_RUNG_CHUNKS = 64` (`crates/seed/src/ladder.rs:22-23,
     47-57`). A coarser rung means fewer than 64 chunks per face edge, which is a decision about the
     ADDRESS SPACE, and `RUNG_MAX = 15` caps it anyway (`ladder.rs:27`).
   - **A coarser rung has no shape left to draw:** `octaves_at(rung)` keeps `octave_count − rung`
     octaves (`crates/terrain/src/body.rs:251-258`). The home planet has 14. Rung 11 keeps 3. Rung 13
     keeps 1 — the single 400 km wave. So the coarse answer DEGENERATES before it gets coarse enough,
     and `dropped_bound_m` grows to the whole relief.

   **So "no rung answers an orbit picture" is a LANDFORM demand as much as a ladder demand**, and this
   document states both rather than assuming a rung will appear. Refuter A is right that the last
   revision named neither fact.

---

## 12. THE PICTURE SET, AND THE ORDER

The order is not arbitrary: each picture answers a question the next one depends on. **Fourteen judged
pictures and two recordings.** One light each, because the renderer casts no shadows (§7.4).

| Order | Picture | Sun | The question it closes |
|---|---|---|---|
| 1 | **Calibration** (three logins) | high | Is the ruler right? Is the stamp right? Nothing later can be believed if this is wrong. |
| 2 | ★ **THE PILOT'S VISTA, 1.8 m** | low | THE PICTURE. What does a person standing on our world see? V11 prints the prominence beside it, against the reference picture's own 323 m at 50 km. |
| 3 | **The high ground, 1.8 m** | low | Is there any relief at all inside a pilot's horizon? MEASURED today: a plane tilted 1.6°. |
| 4 | **The drone diagnostic, 373 m** | low | The tilt, cleanly — and the residency defect at its largest (87.6 px). NOT a vista. |
| 5 | **The low ground, 1.8 m** | low | Is there anywhere to be, not just to look at? |
| 6 | **The polar cap** | low | Does the biome reach the picture? (V7 owed until a paint table exists.) |
| 7 | **The desert** | high | The second biome — and V12 reads its latitude against the Hadley band. |
| 8 | **The highland** | low | The snow-capped ridge of the reference picture, when a snow line exists. |
| 9 | **The authored crease** | low | THE CREASE QUESTION (ruling V10), asked on a subject that HAS a crease. |
| 10 | **The cave mouth** | any | THE CAVE LATTICE STEP (ruling V10), on the one real vertical face the world holds. |
| 11 | **The face seam, along** | low | Is the cube-sphere seam invisible? |
| 12 | **The face seam, across** | low | The same, in the harder direction. |
| 13 | **Orbit at 300 km** | — | Does the body read as a body? Is the hypsometry visible from outside? |
| 14 | **Orbit at 10 000 km** | — | The whole body in one frame — a disc 451 px across. (2 000 km is dropped: its disc is 1 396 px and does not fit.) |
| R1 | **The flight**, recorded | — | SL8: is there a pop, a hole, or a visible REALM CROSSING between orbit and a boot on the ground? |
| R2 | **The rung sweep**, recorded | — | A rung-coverage instrument: does every rung boundary cross the frame at least once? |
| — | **The river bend** | — | OWED: hydrology. §16 Q4. |
| — | **The coast** | — | OWED: a drawn sea, and a sea worth standing beside. §16 Q8. |
| — | **The plateau edge** | — | OWED: a body that holds a slope over 45°. |
| — | **The windward/leeward pair** | — | OWED: a paint table, a wind, and a ridge this body may not have. §16 Q6. |

**One picture the set does NOT take.** No picture with vegetation before slice 14 — but every stand is
re-taken after slice 14 with the same recipe, and **slice 14 also brings V19's three verdicts**, because
a re-taken picture is a comparison and not a verdict (§9 V19).

**A picture of a SECOND BODY is lawful**, and SL5 does not forbid it: SL5 forbids a variant WORLD (*"no
scale knob, no preset, no reduced or test-only variant, no second generator"*), and the world already
holds planets and moons (`crates/physics/src/worldgen/generate.rs`). The set no longer takes a two-body
PAIR, because the pair would pass for the wrong reason; it takes V13, an ordering over the system's own
planets (§9 V13).

---

## 13. HOW IT IS BUILT

### 13.1 The order of work, with its gates named

```text
   GATE A: slice 8 deletes D-TERRAIN-3 (one rung per realm, crates/client/src/chunks.rs:335-341)
   GATE B: the owner answers D15 (the pixels per cell), and the ladder slice answers M8-6
   GATE C: a hull can be FLOWN down inside the picture gate (D-TERRAIN-4), or R1 does not exist
```

1. **M8-2, M8-1 and M8-9** — the probe's cost, the search's real cost, and the body's rise profile.
   Nothing is built if the probe moves a pixel of the colour frame, and no residency is priced before
   M8-9.
2. **The probe**: the second render target with the author-plus-instance id; the buffer verdicts in
   `vd-client-harness`. **Before the stamp**, because §6 deletes `paint_share`, whose HUD mask is
   `x < 320 && y < 180` (`crates/bins/tests/terrain_pictures.rs:275`) and cannot hold a ten-line stamp.
   Building the stamp first would leave the gate red in between.
3. **The stamp**: `PictureStamp` in `vd-devproto`, filled and formatted in `vd-client`, drawn by
   `draw_hud`, asserted in the gate. The terrain census becomes a counter under its true name
   (RESIDENT radius, not drawn horizon).
4. **The ruler**: the avatar as the instrument; the two new diagnosis-row fields (§5.4); V1.
5. **The search, `BodyStats`, and the stands**: in `vd-terrain` under the fence, with the co-prime
   stride; M8-3 extends the golden table to the found addresses. **Ground stands only until GATE A and
   GATE B.**
6. **V11 and V12 next**, because they are the two verdicts that can FAIL the world we have, and the
   landform domains need their numbers first.
7. **AFTER GATE A:** the multi-rung stands, V5's rung halves, and the rung sweep (R2); M8-5, then V9.
8. **AFTER GATE B:** the vista's full residency and the two orbit stands.
9. **AFTER GATE C:** the flight (R1), V14.
10. **The refuter**, every finding answered; the gates; the report; the owner's word.

### 13.2 Where each piece lives

| Piece | Crate | Tier |
|---|---|---|
| `PictureStamp` (serde), the two new `DevRealmBox` fields | `vd-devproto` | A, 100 % |
| `PictureStamp::of` and `::lines` | `vd-client` | A, 100 % |
| **The stand search, the stand predicates and `BodyStats`** | **`vd-terrain`** (the fence; no new crate edge for anyone) | A, 100 % |
| The verdicts over the buffers (skyline, holes, ring fit, second difference, id extent, prominence) | `vd-client-harness` | A, 100 % |
| The resident-radius counter | `vd-client-render` → `DevCounters` | B |
| The probe render pass | `vd-client-render` | B |
| The gate and the picture set | `crates/bins/tests/terrain_pictures.rs` | B, GPU-required, LOCAL |

The split is the one the crate list already draws (`justfile:12`). **The search does NOT go in
`vd-client-harness`**, for the lint reason of §7.2, not for an arithmetic one.

**Two HR5 notes, stated rather than assumed.**

- A search that returns "no address satisfies the predicate" needs that arm covered. On the one world
  (SL5) some predicates may never take it — the desert predicate finds a hit in about 32 addresses out
  of a whole planet. Where the arm is unreachable it takes a named row in `coverage-exemptions.toml`,
  with the measurement beside it.
- **The expensive arm is `BodyStats`.** Its unit tests measure a SMALL stated sample and assert the
  shape; the million-address pass runs only in the bench and the gate (§7.2). A quantile pass over a
  million addresses inside a coverage build would be minutes per run.

### 13.3 The one-source discipline, stated as tests

- `vd-client`: `lines()` is a pure rendering of the struct (no second formatting path).
- `vd-client`: the stamp's altitude uses `world_pos`, the same chokepoint the renderer draws with
  (`crates/client/src/net.rs:838` and the lesson at `:866-873`).
- The gate: the stamp polled at the straddle equals the stamp in the picture's state file.
- The gate: V2(b) compares the pixels with the body's radius, which are two sources. V2(a) is declared
  a residency self-check in its own text, so nobody mistakes it for the second one.
- The gate: **every tolerance is read from `BodyStats`, never typed** — a test asserts that no verdict
  names a numeric literal as a bound.

### 13.4 The stand-in this deepens, and the shape that survives it

Every stand is placed by `VD_SPAWN_POSES` (`D-TERRAIN-4` 🟥). The stand recipe is therefore written as
**a direction, an altitude over the surface, and a nose** — never as a pose. When ruling V11's
server-side up lands, only the delivery changes: the recipe hands the same three quantities to whatever
seats the character. The picture set does not need re-authoring, and the owner's comparison over time
survives the change. That must be stated in the recipe's own doc comment so a later slice cannot quietly
re-anchor it. **And V10 becomes a real no-drift measurement at exactly that moment** (§9, V10), while
R1 becomes possible at the same time (GATE C).

---

## 14. THE LAW GATES

| Law | How this design passes it |
|---|---|
| **SL10** (one generator, two hosts, no drift) | Every terrain number in the stamp comes from `vd-terrain`, linked into the client. **The stand search lives in `vd-terrain` under the float fence and names no transcendental and no degree**: it strides integer ladder addresses with a co-prime step and turns them into directions through the already-fenced `vd_seed::bend::direction`; thresholds are rise-over-run ratios. No verdict re-implements a height. **This design does NOT claim to add a no-drift measurement**: V10 compares one binary with itself until ruling V11's seating lands and the picture runs across two chips. **And the golden table it defers to is itself 🟧** — `D-TERRAIN-1`'s x86-64 leg is EMULATED (`docs/design/DEFERRED.md:7721-7745`), so the cross-chip half is UNMEASURED for the found addresses AND for the chunk digests until a real x86-64 machine runs the legs. |
| **The seed ruling** (a seed-derived map of VALUE is a treasure map) | Every stand predicate names SHAPE only: slope, height over the sea, biome, a cave opening. **A stand may never be found by a deposit, an ore or any value.** That is a rule on the predicate set, not a habit. **And the climate/weather split (§4.4) is the same law applied twice:** climate is shape and may be derived; weather is live state and must be shipped. |
| **SL5** (one world) | Every picture is of THE world under THE universe seed. There is no test body, no scale knob, no reduced world, no second generator. A second BODY is lawful and V13 uses many of them; **nobody moves a planet's orbit**, which is why V13 is an ORDERING over the world's own bodies and not a moved-orbit pair. |
| **No magic numbers** | Every stand address is FOUND, and **every RULER used to find it is derived**: neighbourhood lengths are named octaves' wavelengths from the body's own table; thresholds are quantiles of the body's own measured slope distribution, as RATIOS because the fence bans degrees; every verdict's tolerance is three times the body's own measured RMS wobble; sun elevations come from the body's own orbit, outside the fence. **The one number still typed is `k`, the pixels per cell — and it is not typed here: it is owner decision D15, cited and priced at two values** (§11). |
| **The 8 ms per-chunk budget** (ruling V10) | The picture path adds nothing to a chunk. The probe pass is a render target, not a generator call. `BodyStats` runs once per body per recipe version IN THE GATE, never in `from_seed` and never on a login. |
| **The ladder** (rung L is the coarse answer of rung 0) | V8's difference across the sweep is bounded by `dropped_bound_m` at the rung. The stamp lists every rung drawn, so a picture can never hide which rung answered. **And §11 states where the ladder does NOT answer, with both structural reasons: `TOP_RUNG_CHUNKS = 64` is a packing rule, and `octaves_at` leaves three octaves at rung 11 and one at rung 13.** |
| **The record** (12 bytes, the density byte, a biome/object param) | Untouched. The stamp reads the biome through `biome_at`, not through a record field. The probe's id buffer is a render-time artifact and never a stored record. |
| **The edit pyramid** (the authoring override) | The stamp states whether a diff was applied to the drawn chunks. **The authored-crease stand (12) uses the pyramid as its subject**, which is the lawful way to put a sharp crease in front of the camera on a body whose seed shape has none. |
| **The collider on the same shape** | V10 asserts the delivered pose stands on the client's own surface, on the shipped path, on every picture — as a SEATING check today and as a no-drift check when its two preconditions hold. |
| **SL8 seamless** | V5 (no visible edge), V9 (the second-difference spike detector) and **V14 (the realm crossing)** are the SL8 gate for terrain, and they run on THE FLIGHT — the shipped path a player can actually fly — not on a 3 000 km/s sweep no suit and no hull can reach. The rung sweep stays as an instrument and is named one. The set cannot judge a rung boundary at all before `D-TERRAIN-3` is deleted (GATE A). |
| **HR5** (100 % in Tier-A) | The stamp's expressions, the stand predicates, `BodyStats` and every buffer verdict are pure functions in `vd-devproto`, `vd-client`, `vd-terrain` and `vd-client-harness`, tested on synthetic inputs and on SMALL stated samples. Only the probe pass and the gate sit below the line. **Two arms are named: "no address satisfies" may need an exemption row, and `BodyStats`' million-address pass never runs in a unit test** (§13.2). |
| **V4** (vegetation is an art asset placed by a server skeleton) | No verdict draws or judges a primitive tree. **V19 gives the reference picture's largest object its own three verdicts** — the skeleton's placement under SL10, the density against the biome, and no impostor by the mesh-instance id. The stands are re-taken after slice 14, unchanged. |
| **SL3** (a realm draws itself) | The stamp reports what the realm stated (its surface statement, its look, its luminous rows). It never composes a realm's appearance from a chain. |
| **SL6** (ask before new data crosses) | **Two claims, separated.** (a) *The stamp and the two new diagnosis-row fields cross no realm boundary*: they ride `DevState`, the harness's own diagnosis lane, which already carries `extent_m`, `luma` and `body_kind` by the same right. (b) **The weather fields DO cross a realm boundary, and this document carries the ASK, not a denial** (§4.4). |

---

## 15. WHAT YOU DECIDE

| # | Question | Recommendation |
|---|---|---|
| **P8-1** | The stamp | **Every picture carries the on-frame readout of §4.2, computed once in Tier-A, drawn by the renderer, asserted by the gate out of the picture's own state file. No number is typed into a picture.** |
| **P8-2** | The picture probe | **Every judged capture writes an aligned id + distance buffer, and the id is an AUTHOR (3 bits) plus a MESH INSTANCE (13 bits). Every geometric verdict reads it. The colour classifier is deleted, and the probe is built BEFORE the stamp.** |
| **P8-3** | The drawn horizon | **The picture's reach is the farthest terrain pixel in the probe, which is in the frustum by construction. The renderer's `farthest` is promoted to a counter under its true name — the RESIDENT radius — and never called the drawn horizon.** |
| **P8-4** | The ruler | **The avatar is an INSTRUMENT, exact inside 145 m, and it stands only in the calibration picture. It is NOT a scale figure: it paints a ball. The reference picture's character is OWED to slice 16. The far ruler needs two new diagnosis-row fields.** |
| **P8-5** | Stands are found, satisfying, fenced, and SPREAD | **A stand is a satisfying predicate at a stated scale over a CO-PRIME stride of ladder addresses, in `vd-terrain` under the float fence, first hit wins. Every threshold is a rise-over-run ratio, because the fence bans degrees. An argmax is refused: on a homogeneous field it is not a landform and it costs 33 minutes.** |
| **P8-6** | ★ **The flagship picture returns to the PILOT'S EYE** | **The reference picture is a GROUND picture: its 50 km comes from RELIEF, not from altitude. The flagship is 1.8 m, and V11 prints the visible-prominence threshold beside it — 323 m at 50 km, 1 390 m at 100 km. The 373 m stand stays as a DIAGNOSTIC. A flat world then fails the flagship instead of being flown over.** |
| **P8-7** | ★ **Every tolerance comes from the body's MEASURED field** | **No verdict may use `relief_bound_m`: it is the aligned bound, it is 277 px at 300 m, and the last revision's V2(b) therefore PASSED the very picture this document exists to refuse. Every bound is three times the body's own measured RMS wobble at the view length — four to six pixels at every scale.** |
| **P8-8** | ★ **The two-light doubling is dropped** | **The renderer casts NO shadows and adds a fill light from the opposite side. A low sun buys a 2 % Lambert change on 1.37° ground. Fourteen judged pictures, one light each. The doubling returns when a shadow pass exists, and that pass has no owner.** |
| **P8-9** | Six stands are OWED, with their measurements | **The river bend (no hydrology), the coast (the sea is not drawn and covers 0.92 %), the plateau edge (the smooth field's largest slope is 7.23°), the windward/leeward pair (no wind, and probably not enough rise), and the whole weather set. Ruling V10's crease question moves to an AUTHORED crease, which is buildable.** |
| **P8-10** | The nineteen verdicts | **V1–V19 of §9, each with a derived bound or a measured-then-frozen one. V6, V7 and V15–V19 are INAPPLICABLE today and say so. V11 and V12 CAN FAIL TODAY, and they are the two the landform and biome slices should aim at.** |
| **P8-11** | The four numbers beside the picture | **The relief profile split into TILT and WOBBLE, the slope histogram, a hypsometric curve taken as a stated sample at rung 0, and THE BIOME LATITUDE HISTOGRAM — which already shows the tundra beginning at 26.6° and the desert only on the equator.** |
| **P8-12** | V10 is a seating check, and the golden table is itself owed a leg | **The gate computes the surface itself today, so V10 compares one binary with itself. And `D-TERRAIN-1` is 🟧 with an EMULATED x86-64 leg, so "no drift" is UNMEASURED on the target that matters. Say so rather than citing it as an existing cross-chip measurement.** |
| **P8-13** | ★ **The set is gated on three slices, and on ONE owner answer** | **GATE A: slice 8 deletes `D-TERRAIN-3`. GATE B: the owner answers D15 (`k` pixels per cell — the whole price swings four times on it) and the ladder slice answers M8-6. GATE C: a hull can be flown down in the gate, or the SL8 recording does not exist.** |
| **P8-14** | ★ **TWO recordings, because one path answered two questions** | **THE FLIGHT is the SL8 gate: the shipped path, a rated hull, V3 + V9 + V14 (the realm crossing — the one seam this architecture really has). THE RUNG SWEEP keeps the fractional descent under an honest name: a rung-coverage instrument, 1 545 frames, and 1 545 state dumps that M8-11 prices before it runs.** |
| **P8-15** | Weather crosses a boundary, so this is an ASK — and CLIMATE does not | **§4.4 splits them. Climate is shape and the client may derive it. Weather is live state: cloud cover, precipitation and visibility, shipped from the realm. V15 (two clients agree) is the verdict that proves the ask was needed.** |
| **P8-16** | The set's stability over slices | **The stand recipes are frozen and re-run every terrain slice; the pictures are stored per slice with their state files. The recipe is golden; the pixels are not. The co-prime stride is what keeps the stands spread over the body instead of in one corner of one face.** |
| **P8-17** | Where the set runs | **A LOCAL, GPU-required gate, as every capture gate is. M8-7 decides whether it runs per slice or per milestone, and it counts the LOGINS: three for calibration, one per other stand.** |

---

## 16. OPEN QUESTIONS

**Q1 — Does the probe's second render target belong in the shipped renderer at all?**
It is only ever written on the capture path. A cleaner shape puts it behind the capture feature so no
shipped frame can pay for it. The cost is that the gate then measures a renderer the player does not run.
UNMEASURED: whether the extra target changes the colour frame by one pixel (M8-2 answers it, and the
answer decides this question).

**Q2 — How many stands can one cluster boot serve?**
Slice 7 took three pictures from one cluster with three logins. Fourteen pictures, two recordings and
sixteen logins are more. UNMEASURED: the wall-clock cost (M8-7). If it is large, the set splits into a
fast half per slice and a full half per milestone — which is a decision, not a detail.

**Q3 — Is the prominence count a fair proxy for "vast"?**
V11 counts skyline pixels above the horizon row. It rewards a spiky world and says nothing about the
ground between here and there. A second proxy — the count of distinct distance bands that hold terrain —
is proposed inside V8. The owner should say whether either number is worth anything to him, or whether
"vast" stays entirely his eye.

**Q4 — The river bend has no predicate, because the recipe has no water flow.**
There is no flow accumulation and no stream power in `crates/terrain`. Two honest answers exist: the
landform slices build hydrology and the stand ships with it, or the stand is dropped and the owner is
told that the world has no rivers. **A river faked by a noise channel would pass a picture and fail a
player who walks upstream**, so it is refused here.

**Q5 — Should the stamp be in the picture, or beside it?**
Drawn on the frame, it is unmissable and it is also in the way of the vista the owner is judging. The
recommendation is BOTH, with a switch: the set takes each judged picture twice, once with the stamp and
the far ruler, once clean, from the same straddle. UNMEASURED: whether two captures from one straddle
are byte-identical apart from the overlay.

**Q6 — WHO OWNS THE ATMOSPHERE, AND WHO OWNS THE SHADOW PASS?**
Three things have no slice and each blocks a picture. **No slice owns the atmosphere**, so V6's bound
(M8-4) has nothing to measure on and the sky in every picture stays black. **No slice owns the shadow
pass**, so §7.4's two-light doubling stays dropped and the reference picture's raking light cannot be
reproduced at all. **The SL6 ask of §4.4 is unanswered**, so V15–V17 cannot run. The weather ACCEPTANCE
is written here as stands and verdicts (V15, V16, V17, V18), so whichever slice lands the mechanism
inherits a gate and not a paragraph.

**Q7 — Does the owner want the sweep's stamps on round altitudes or on round frames?**
The rung sweep descends at a constant fractional rate (P8-14), so the listed altitudes do not land on
frame boundaries. The recommendation is to stamp the nearest frame to each listed altitude and print the
true altitude in the stamp.

**Q8 — Should the sea be drawn at all, and should it be bigger? Here is the number that answers it.**
`Stratum::Water` is not solid, so the extractor makes no face for it and the sea is not drawn
(`crates/terrain/src/strata.rs:91-95`; `crates/terrain/src/extract.rs:73-75,444-446`). And the ocean
share follows in one line from two numbers this document already measured:

```text
   ocean share ≈ Φ(sea offset / surface standard deviation) = Φ(−5 297 / 2 295) = Φ(−2.31) = 1.0 %
      (MEASURED 0.92 % here; sibling 01 replays 0.99 % over 200 000 columns — the model reproduces)

   an Earth-like 71 % needs z = +0.553  ->  offset = +1 270 m
   the offset is (u·0.7 − 0.4) · relief, so that needs u > 0.698
   ->  about 30 % of the world's bodies ALREADY draw an ocean world, and the home planet drew a puddle
```

**So the owner can be asked the thing he can actually judge**: what share of an earth-like body should be
ocean, and how wide a spread across bodies — not a range in metres he cannot picture. Refuter B is right
that the last revision stopped one step short.

**Q9 — The owner's FIRST sentence, fact by fact.**
*"Biomes … should be dependent on the planet position, spin, trajectory, size and gravity."*

| The owner's word | What the acceptance does today | Who owes the rest |
|---|---|---|
| **Size** | **Already tested.** Every bound in §9 is a function of the body's radius: the horizon, the dip, the sag, the prominence threshold, the tolerance. A body of another size gets different numbers and the same verdicts. | nobody — it works |
| **Position** | **V13**, an ordering of the system's planets by insolation. It cannot run: `biome_at` reads no insolation (`crates/terrain/src/height.rs:38-65`). | the biome slice |
| **Spin** | **V17** (the wind turns with the Coriolis deflection) and **V12's band** (the Hadley cell's descending latitude follows from the rotation rate). And §7.4: the day a planet spins, every ground stand breaks until the diagnosis row carries a facing. | the spin slice |
| **Trajectory** | Obliquity is named *"a later slice"* in the code (`height.rs:29-35`), so there are no seasons and no tilted ice caps. No verdict is written, because there is nothing yet to write one against. | the orbit slice |
| **Gravity** | `g = G·M/r²` from the drawn mass and radius (`crates/physics/src/worldgen/generate.rs:1206,1227`). It sets the atmosphere's scale height, hence the haze's fall-off, the snow line and the dust. **Nothing in `crates/terrain` reads it, and no verdict can be written before the atmosphere exists.** | the atmosphere slice |

**Q10 — Two documents, two rulers.**
This document reports a two-point slope; sibling 01 reports the tilt of a plane fitted to a ring; this
document computes with 720 rows and sibling 01 with 1 080. Both pairs are defensible and neither is
reconciled. **M8-10 puts one bench in `vd-terrain` that both cite.** Until it runs, the owner should read
the two documents' slope and wobble numbers as two DIFFERENT quantities, and not as a disagreement.

---

## 17. REFUTATION ANSWERS

**Round 1** (refutation A: the laws and the code; refutation B: believability, cost and the owner) was
answered in revision 2. Its twenty-four findings were all FIXED and the fixes stand in this revision:
the V2 split, the hull ruler's refutation, the fenced search, the derived predicates, V10's demotion to
a seating check, GATE A, the corrected sea and noise measurements, the OWED stands, the SL6 ask, the
second-difference detector, the exact ring projection, and the SL5 reading that a second BODY is lawful.
Nothing from round 1 is re-opened here.

**Round 2** is answered below. **FIXED** = the section was rewritten. **FIXED, refuter corrected** = the
finding is right and the fix goes further or differs from the refuter's own proposal, with the evidence.
**KEPT** = the refuter is wrong, with the evidence. **OWED** = it is real and it belongs to a named
slice, a named measurement or the owner.

### Round 2 — Refutation A (the laws and the code)

| # | Finding | Severity | Answer |
|---|---|---|---|
| A-B1 | V2(b), V4 and V5(b) use `relief_bound_m`, which is the aligned bound: 277 px against a 69.4 px signal at 300 m. **V2(b) as specified PASSES the slice-7 hill picture**, and V4's applicability band is wrong at its low end by two orders of magnitude. | BLOCKER | **FIXED.** This is the most serious finding of either round, and the refuter is right in every part. §9 now opens with THE TOLERANCE RULE: no verdict may use `relief_bound_m`. Every bound is three times the body's own MEASURED RMS wobble at the view length — **4.13 px at a 1.8 m eye, 5.67 px at 300 m, and about 4 to 6 px at every scale, because the field is nearly self-similar** (§0.2 M-D). With it the hill picture fails by twelve times. V4's band moves from 151.6 m to **4 869 m** of eye height (MEASURED by bisection at twice the tolerance). §0.2 M-D carries the whole finding to the owner's first page. |
| A-B2 | Eight of the sixteen judged pictures rest on cast shadows the renderer switches off; the word "shadow" appears once in the document and never as a fact about the code. | BLOCKER | **FIXED**, and verified independently: `shadows_enabled: false` at `crates/client-render/src/terrain.rs:459` (sun), `:473` (fill) and `crates/client-render/src/lib.rs:848` (key), with a deliberate opposite-side fill at `FILL_SHARE = 0.08` (`terrain.rs:56,466-478`). §1 carries it as a row, §0.2 M-C as a headline measurement, §7.4 drops the doubling, and §12 falls from sixteen judged pictures to fourteen. §16 Q6 names the shadow pass as a thing with no owner. |
| A-B3 | The probe cannot run V8(b): §6.2's id names an AUTHOR and V8(b) needs a MESH. The law gate for ruling V10's "no impostors" is passed by an instrument defined two ways. | BLOCKER | **FIXED**, and the refuter's own one-line cure adopted. §6.2 splits the `u16`: **three bits of author, thirteen bits of mesh instance** (8 192 instances; the largest stand draws 5 752 chunks, so it fits and the 6-byte price is unchanged). §9 V8(b) reads the instance field and states the count of distinct instances a band must hold, derived from §11's own column count. §14's ruling-V10 row now names the field that carries it. |
| A-D4 | *"compared byte for byte on x86-64 and aarch64, exactly as the golden chunk digests already are"* is false: `D-TERRAIN-1` is 🟧 and its x86-64 leg is EMULATED. | DEFECT | **FIXED.** Verified at `docs/design/DEFERRED.md:7721-7745`: G4 is *"EMULATED 2026-09-08 … a smoke test: it can find a drift and can never prove its absence"*. §1 carries it as a row, §7.2 states that **the only no-drift gate is itself owed a leg**, and §14's SL10 row says the cross-chip half is UNMEASURED for the found addresses AND for the chunk digests. P8-12 says the same to the owner. |
| A-D5 | The float-fence argument for the search's home is an unmeasured determinism claim: the fence's own text says comparison is IEEE-fixed on every target. | DEFECT | **FIXED**, and the refuter's own replacement adopted by name. Verified at `crates/terrain/clippy.toml:1-8`. §7.2 now states the LINT reason: an unfenced crate has no lint, no compile-fail control (`crates/terrain/src/gf.rs`) and no link scan (`just terrain-link-scan`), so nothing stops a later slice writing `.powf()` into the search. §1's fence row states the trusted-operation list too, so the false claim cannot be re-derived from the document. |
| A-D6 | The predicates are stated in degrees, and the fence bans `to_radians`, `to_degrees` and `atan`; a sun elevation cannot be computed inside the fence at all. | DEFECT | **FIXED**, in both halves the refuter names. §7.2's table now states every threshold as a RISE OVER RUN (p10 = 0.0049, p90 = 0.0517 over 1 562 m), and says the degrees are printed by the STAMP in `vd-client`, which is not fenced. And: *"The sun rule is NOT part of the fenced search … The fenced search finds the PLACE; the gate chooses the MOMENT"*, which is where `crates/bins/tests/terrain_pictures.rs:444-460` already computes it. §14's no-magic-numbers row repeats both. |
| A-D7 | The slope-quantile table is put in the body's derived table with no cost, no identity and no coverage analysis; `from_seed` runs on every login. | DEFECT | **FIXED**, and all three questions answered by name. §7.2: the table is `BodyStats::measure(body, sample_count)`, **called by the GATE, never by `from_seed`** (verified: `from_seed` runs at `crates/client/src/chunks.rs:277` and `crates/bins/src/bin/shard.rs:404`); it is **outside** the body's `PartialEq`, the pin and the golden table; and its million-address pass **never runs in a unit test** — the unit tests use a small stated sample and assert the shape. §13.2 repeats the coverage half as the second named HR5 arm. |
| A-D8 | *"the home planet holds no cliff"* is a statement about `height_m` presented as a statement about the drawn surface: the carvers open vertical faces and the generator's own code contemplates a cliff. | DEFECT | **FIXED**, and it strengthens two stands. Verified: `crates/terrain/src/carve.rs:1-16`, `crates/terrain/src/extract.rs:435-455` (a quad wherever `is_rock` differs across a cell edge), `crates/terrain/src/digest.rs:92-94` (*"a cliff of more than 62 cells inside one column"*). §3.3 now claims only that **the SMOOTH HEIGHT FIELD holds no slope over about 8°**, and says everything vertical comes from the carvers or the pyramid. §2's slope-histogram row carries the same narrowing. |
| A-D9 | The limb radii in V4 are equiangular in a section that says "exact projection", and the 2 000 km picture does not fit the frame. | DEFECT | **FIXED**, and it changed the picture set. My own exact projection: `f·tan(asin(R/(R+h)))` gives **2 009.4 px at 300 km, 698.1 px at 2 000 km, 225.3 px at 10 000 km** — so the disc is 1 396 px at 2 000 km against a 1 280 px frame. §9 V4 prints all three with the disc widths, and **§7.3 drops the 2 000 km still and moves the whole-body stand to 10 000 km**, where the disc is 451 px. §12 order 14 says so. (I also reproduce the refuter's sag column independently: 0.22 / 2.81 / 40.10 / 92.64 / 319.28 and both bisections.) |
| A-D10 | The ground stand's chunk count does not add up (1 505 against 502 × 3 = 1 506), and §4.2's rung row does not match §11's columns. | DEFECT | **FIXED.** §4.2 prints `462 : 348 : 348 : 348 = 1 506` and shows the multiplication; §11 prints 502 columns × 3 = 1 506; §0.2 M-E carries 1 506. The five other internal checks the refuter verified are unchanged. |
| A-W11 | The approach's path is not flyable — 3 000 km/s on the first frame — so V3 and V9 measure the harness, not the game. And 15.53 e-folds at 1 % is 1 545 frames, not 1 553. | WEAKNESS | **FIXED**, and the refuter's own two-recording shape adopted. §8.2 is rewritten as **THE FLIGHT** (the shipped path, a rated hull, the SL8 gate) and **THE RUNG SWEEP** (the fractional descent, named an instrument). V9 judges the flight; the sweep is a diagnostic. The flight is marked OWED on `D-TERRAIN-4` and becomes GATE C in §13.1. The frame count is corrected to **1 545** everywhere, with the arithmetic shown in §2's e-fold row. |
| A-W12 | The near ruler is a featureless dot, not a scale figure, and three of them need three logins nobody prices. | WEAKNESS | **FIXED.** §5.4 item 1 now says the avatar is an **INSTRUMENT, never the picture's scale figure**, because it paints a 0.5 m ball and a person reads a silhouette; **the reference picture's character is OWED to slice 16**. §5.4 item 2 prices the logins (one `VD_SPAWN_POSES` entry per account, `terrain_pictures.rs:457-472`) and M8-7 counts them: three for calibration, one per other stand. §5.5 puts the three avatars on the ground at the 1.8 m calibration stand, not in the air. |
| A-W13 | The wobble measurement disagrees with domain 01's, and neither reconciles it. | WEAKNESS | **FIXED as far as an argument can go; the measurement is OWED to M8-10.** §3.3 now names the two quantities apart (a two-point slope over a spacing; the tilt of a plane fitted to a ring) and gives the mechanism: a ring of radius `r` keeps wavelengths up to `2πr`, a profile of length `L` keeps them up to `L`, and at `r = 1 000 m` that is 6 283 m against 1 000 m, where the octave amplitudes are 80.3 m and about 10 m. **The last revision's claim that the two documents "agree in the same words" is withdrawn in the text.** M8-10 puts one bench in `vd-terrain` that both cite. §16 Q10 hands the owner the state of it. |
| A-W14 | The reference picture's largest mass — the forest — has no acceptance, and its absence is written as compliance. | WEAKNESS | **FIXED.** §9 **V19** is new and has the three parts the refuter names: (a) the server skeleton's placement is a function of seed and address and is byte-identical on two hosts, or it crosses as a diff (SL10); (b) the drawn density agrees with the biome the stamp names (V7, one level up); (c) every far forest band names several MESH INSTANCES, because a far forest is where an impostor is most tempting. §12 says a re-taken picture is a COMPARISON and V19 is the verdict. §14's V4 row now names V19. |
| A-W15 | The first-hit raster scan puts every stand within about eleven addresses of the scan's start — one corner of one face — and §4.2's showcase address cannot come from it. | WEAKNESS | **FIXED**, and the refuter's own cure adopted. §7.2 replaces the raster order with a **CO-PRIME STRIDE** drawn from the body's seed: `(k·S) mod M` with `gcd(S, M) = 1`, all integer, exhaustive, total, and spread over the body. §4.2 now says in full that the printed address is **a worked example and not a found address**, and that the found one is written into the picture's own state file. §10.3 says the stride is what makes the comparison over time honest. |
| A-W16 | *"No rung answers an orbit picture"* is stated without the two facts that decide it, and the orbit prices change method silently. | WEAKNESS | **FIXED**, and both facts adopted by name. §11 now states that `TOP_RUNG_CHUNKS = 64` is a **packing rule about the address space** (`crates/seed/src/ladder.rs:22-23,47-57`, with `RUNG_MAX = 15`), and that `octaves_at` leaves **three octaves at rung 11 and one at rung 13** (`body.rs:251-258`), so the coarse answer degenerates before it gets coarse enough. The conclusion is restated as *"a LANDFORM demand as much as a ladder demand"*. The vertical count is now stated per rung with its own code reason (Rule 2), so no row changes method silently. |
| A-W17 | V3 hides the two holes a descent makes most often: a column with no terrain pixel has no skyline row, and a missing far band does not put an unauthored pixel below the skyline. | WEAKNESS | **FIXED**, exactly as the refuter specified. §9 V3 gains half (b): **the count of columns that hold NO terrain pixel below the projected horizon row**, a count that must be zero and needs no tolerance. §6.3 adds the row *"Is a whole COLUMN missing?"*. The second half of the finding — that V2(b) could not catch the far band — is cured by A-B1's tolerance fix. |
| A-N18 | One code claim in §4.2 cites a slice item, not a `file:line`. | NOTE | **FIXED.** §4.2's sun row cites `crates/client-render/src/terrain.rs:430-436`. |
| A-N19 | The 60 km row of §3.2 does not reproduce: the exact sphere gives 11.9061° and a 17.3 px gap. | NOTE | **FIXED**, and the refuter is right about the cause. §3.2 now computes every depression by the EXACT sphere formula and names the flat approximation the last revision used. The table reads 11.9061° and 17.3 px at 60 km, and the 373 m row is recomputed to 87.6 px on the same patch radius. |

### Round 2 — Refutation B (believability, cost and the owner)

| # | Finding | Severity | Answer |
|---|---|---|---|
| B-B1 | The 373 m flagship is the wrong reading of the reference picture: its 50 km comes from RELIEF, not altitude. The set loses the human vantage, and the landform domains lose their number. | BLOCKER | **FIXED**, and it is the largest change in this revision. §0.2 M-B computes the threshold curve `h' = (d − 3 473)²/2R` and prints it (6.4 m at 10 km, 323 m at 50 km, 1 390 m at 100 km, 2 989 m at 145 km), so the reference picture's own 1 500–3 000 m ridges would be visible from our ground at 100–145 km. **The flagship returns to the 1.8 m pilot's eye** (§7.3 stand 2, §12 order 2, P8-6); the 373 m stand stays as a DIAGNOSTIC. **§9 V11 is new and is the verdict that makes a flat world FAIL the flagship**, and §0.3 names the threshold curve as the one number this acceptance owes the landform domains. |
| B-B2 | The approach recording has no mechanism, and it never names the realm crossing — the one seam this architecture really has. | BLOCKER | **FIXED**, both halves. §8.2 says what moves the eye: a rated hull on the movement contract's own lane, marked OWED on `D-TERRAIN-4` (a spawn is not a flight, and a re-spawn per frame is a teleport). **§9 V14 is new**: the frame before the crossing and the frame after differ by no more than M8-5's smooth-motion distribution. §8.3 lists it, §13.1 makes it GATE C, and §14's SL8 row names it. |
| B-B3 | GATE B is *"a number, not an opinion"* and is neither reproducible nor sourced: (a) the 2-pixel rule is open owner decision D15; (b) three chunks per column is a rung-0 measurement applied at every rung; (c) the three orbit rows do not reproduce. | BLOCKER | **FIXED on (a) and (c); FIXED with the refuter corrected on (b).** (a) §11 cites **D15** (`01_reference_target.md:1543`) by name, prices the whole set at k = 2 AND k = 4, and P8-13 makes the owner's answer part of GATE B. §1 also records that domain 01 computes with 1 080 rows, so the two tables are not directly comparable (B-W2). (c) The refuter's arithmetic is adopted exactly: **5 752 / 1 635 / 3 277** columns. (b) **The conclusion is right and the mechanism is not.** The vertical count is capped by the BAND's own height, not by the local relief: `surface_chunk_span` adds one chunk of margin each way and clamps to the band's top chunk index (`crates/terrain/src/digest.rs:89-130`), so a column holds **three chunks at rungs 0–7, two at rung 8 and one at rungs 9 and above** on the home planet — three even at rung 7, where the refuter's relief argument would give one. §11 Rule 2 prints the whole table with `cells_in_band` as its source. |
| B-D1 | The renderer casts no shadows, so "raking light" buys almost nothing and the set doubles for it. | DEFECT | **FIXED** — the same fix as A-B2, and both refuters found it independently. |
| B-D2 | The two-body verdict would PASS today for the wrong reason; *"move the orbital distance"* is unbuildable under SL5; and SIZE and GRAVITY get nothing at all. | DEFECT | **FIXED**, and the refuter is right on the law and on the code. Verified: each body draws its own `temperature_seed`, `humidity_seed` and wavelengths (`crates/terrain/src/body.rs:203-215`), so two bodies already differ for an unrelated reason. **§9 V13 replaces the pair with an ORDERING over the system's own planets, ranked by insolation** — it runs on THE world and it can fail. **§16 Q9 is now a five-row table**, one row per word the owner used: SIZE is **already tested** (every bound in §9 is a function of the radius); POSITION is V13; SPIN is V12's band and V17; TRAJECTORY has no verdict and says why (obliquity is *"a later slice"*); GRAVITY is `G·M/r²` from `taxon.mass_kg` and `taxon.radius_m`, it sets the atmosphere's scale height, and it waits on the atmosphere. §2 gains a surface-gravity row. |
| B-D3 | The biome mix is measured and never judged; the deserts sit where the rainforest belongs, and the document owns the tools to say so. | DEFECT | **FIXED, refuter corrected — and it became one of the two verdicts that can fail today.** **§9 V12** bins the desert and tundra addresses by latitude against a stated circulation band. **My arithmetic differs from the refuter's and is worse for the body:** the refuter dropped the height term, which is not small — the typical column stands 5 297 m over the sea against `highland_above_m = 7 868 m`, so the term is 0.202. **The tundra therefore begins at 26.6° of latitude, not 40.5°, and the desert exists only within 8.2° of the equator, not 20°.** Three cross-checks reproduce the measured shares (Tundra 45.5 %, Desert 3.1 %, Highland 14.3 %). §10.2(d) adds the biome latitude histogram as a fourth number beside every picture, and §10.1 puts V12 under "believable". |
| B-D4 | Weather and CLIMATE are never separated, and a still picture cannot accept a simulation. | DEFECT | **FIXED**, and the refuter is right that the split is the most useful thing this domain hands the weather domain. §4.4 is rewritten as the split, with an example in the game's own words on each side: climate is SHAPE (SL10, derivable, no boundary), weather is LIVE STATE (shipped, the SL6 ask). **Four verdicts replace the paragraph:** V15 (two clients at one stand and one tick agree — the verdict that proves the ask was needed), V16 (the cloud moves between two ticks), V17 (the wind turns with the Coriolis deflection), V18 (the windward contrast, filed under CLIMATE where it belongs). §16 Q6 states the three missing owners. |
| B-D5 | The rulers: *"costs nothing"* is false, three of them hang in mid-air at the new flagship, and a dot is not a scale figure. | DEFECT | **FIXED** — the same fix as A-W12, plus the mid-air half: §5.5 puts the three avatars at the **1.8 m calibration stand**, on the ground. |
| B-D6 | The arithmetic does not close in four places, and two of them were marked FIXED last round. | DEFECT | **FIXED, all four.** 1 506 chunks and `462 : 348 : 348 : 348` (§4.2, §11, §0.2). The picture count is now **fourteen judged and two recordings**, stated once in §12 and repeated nowhere else. The light sets agree because there is only one light per stand (§7.4, §12). |
| B-D7 | The showcase stamp is headed `vista-01` and reads a 1.80 m eye. | DEFECT | **FIXED.** §4.2 is headed `ground-01`, which is now the flagship stand, so the numbers and the label agree. |
| B-D8 | No picture is priced in TIME or in memory, though the repository publishes both inputs. | DEFECT | **FIXED**, and the refuter's own method adopted. §11's table carries seconds on one thread and on 14 (8 ms of generator budget plus M7-2's 4.18 ms of client geometry = 12.18 ms per chunk), states **the gate's own 90 s terrain deadline** (`TERRAIN_WAIT_TICKS = 1 800` at 20 Hz), and says in as many words that **2.3 GB of mesh is a MEMORY statement, not a disk one** — Bevy holds the asset and the GPU buffer, and the owner's own machine is swap-starved. |
| B-D9 | *"Residency out to the geometric horizon"* is the wrong rule for any world with relief, so both V2(a) and the price are wrong. | DEFECT | **FIXED, refuter corrected on the tolerance.** §11 Rule 3 states the fixed point `d = √(2R·h_eye) + √(2R·P(d))` and solves it with the body's MEASURED wobble: **a 1.8 m eye must be shipped terrain to about 50 km, not 3.5 km — fourteen times the geometric horizon.** V2(a) compares against that radius. **But the refuter's own `P = relief_bound_m` is refused for the same reason A-B1 refuses it:** `√(2R·14 305) = 309 km` is the aligned bound, not the field. **M8-9** measures `P` properly, and until it runs the visible-relief rows are marked a LOWER bound. |
| B-W1 | Domain 08 and domain 01 give two different slopes and 08 claims they agree. | WEAKNESS | **FIXED** — the same fix as A-W13. The two quantities are named apart in §3.3, the claim of agreement is withdrawn in the text, and M8-10 owes one bench. |
| B-W2 | The two documents use different frames (720 against 1 080 rows), so their costs cannot be compared. | WEAKNESS | **FIXED.** §1's camera row states it, and §11 repeats it inside the D15 note, where the borrowing happens. |
| B-W3 | Q8 asks the sea question and stops one step before the number that answers it. | WEAKNESS | **FIXED**, and the refuter's own z-score adopted and re-derived. §16 Q8 now prints `Φ(−5 297 / 2 295) = 1.0 %` against the measured 0.92 % and 01's 0.99 %, and `u > 0.698` for an Earth-like 71 % — **so about 30 % of the world's bodies already draw an ocean world, and the home planet drew a puddle.** The owner is asked for a target share and a spread, not a range in metres. |
| B-W4 | Stand 15 may be refused a second time, and the document does not measure it. | WEAKNESS | **FIXED as far as arithmetic goes; the measurement is OWED.** §2's orographic-lift row now states the dry adiabatic lapse rate (9.8 K/km) and the lifting condensation level (~125 m of rise per kelvin of dew-point depression), and concludes that a visible rain shadow needs about **1 000–1 500 m of rise inside 10–20 km**. The home planet gives 1 000 m across 50 km. §7.3 stand 15 and §9 V18 both carry it: **this body may hold no rain shadow even after the weather lands.** |
| B-W5 | The recording's own bytes are unpriced; only the probe is counted. | WEAKNESS | **FIXED.** Verified: `record_capture` writes a state dump for every captured frame with no exception for a record sequence (`crates/bins/src/bin/client.rs:1231-1240,1332-1372`, `write_state_dump` at `:1391`). §8.2 prices 1 545 PNGs plus 1 545 pretty-printed JSON dumps plus 20 probes, notes the run fits inside `MAX_RECORD_FRAMES = 3 600` and will *"fill the disk quietly"*, and **M8-11 measures it before it runs on the owner's machine.** §1 carries the per-frame state dump as a row. |
| B-W6 | The argmax refusal leans on a rate the repository does not publish. | WEAKNESS | **FIXED.** §7.1 now says the ~390 ns is **DERIVED**, shows the derivation (`1.5 ms / 3 844 columns` from ruling V10's published column pass), warns that a scattered `height_m` call has no amortisation and is likely slower, and marks **every microsecond in the section ESTIMATED until M8-1 measures the scattered call**. M8-1's row says the same. |
| B-W7 | Terms the owner is meant to learn are used and never explained; and the document says "standing deviation" where the industry says standard deviation. | WEAKNESS | **FIXED.** §2 gains **prominence, quantile, standard deviation, second difference, surface nets, straddle drift, e-fold and surface gravity**, each in plain words with an example in the game's own words. **"Standing deviation" is replaced by "standard deviation" throughout**, with a line in §2 saying why. |
| B-N1 | The 0.92 % sea share comes from 4 000 directions; the sibling replays 200 000 and reports 0.99 %. | NOTE | **FIXED.** §0.2 M-A cites both and says the larger sample is the better number; §7.3 stand 10 and §16 Q8 carry both figures. |
| B-N2 | §6.2's id set has no id for the atmosphere. | NOTE | **FIXED.** §6.2 reserves `atmosphere` beside `water` and says why V6 will need it; §9 V6 points back. |
| B-N3 | §4.2's stamp shows two world-identity halves that are never equal, and a reader reads it as a failure. | NOTE | **FIXED.** §4.2 carries one short paragraph naming the two quantities and stating that a real mismatch is refused at login by `WorldRefused`, never by a reader comparing hex strings. |

### What both refuters checked and found sound, and which is unchanged here

Each of these was re-derived for this revision and each reproduces: the horizon and dip table; the sag
column by exact ring projection (three independent computations agree to the printed digit) and both
bisections; the pixel angle and focal length; the 144.9 m marker floor and the 43.5 / 10.9 / 3.6 px
predictions; the hull refutation and its 3.3× swing; the octave table, the −5 297 m sea cross-check and
the 0.2701 noise deviation; the 12 % patch diagnosis and the 3.0 / 69.4 px gaps; the argmax arithmetic;
`highland_above_m = 7 868 m` and the 14.3 % share; the five OWED stands' refusals; the V2 split, V8's id
test, V9's second difference and V10's demotion; the one-rung refusal as GATE A; and the frustum-free
renderer loop renamed to the RESIDENT radius.
