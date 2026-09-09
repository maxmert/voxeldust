# Slice 8p — the picture instrument: REFUTATION

Written 2026-09-09 by the refuter, on the uncommitted tree (`git diff` + `crates/client-harness/src/probe.rs`
+ `crates/client-render/src/probe.wgsl`). READ-ONLY: no build, no test and no coverage run was made. Every
claim below is read off the source, or it is marked as a PREDICTION to measure.

**Laws, checked first.** No SL violation found. The ruler and the stamp read the DELIVERED row (the composed
scene the gateway already ships), the eye the renderer already stands at, and the recipe — which SL10 lets
the client compute. Nothing derives a pose or a velocity: `ruler_on_surface` makes a client-local render
subject, never an occupant's placement, and it is stated to nobody. No new realm-boundary datum and no new
`InterShardFlow` arm, so SL6 does not bite: `DevState` is the dev-control surface, not a realm lane. The
client states no placement about itself (SL1 clause 3). One caveat, style and not law: the RULER BALL is a
harness object drawn into the shipped capture path's picture, so from 8p on the owner judges a picture with
an instrument standing in it.

---

## 1. DEFECT (high) — after 8p the gate asserts almost nothing about THE PICTURE. A black picture passes.

`crates/bins/tests/terrain_pictures.rs:406` is now the only assert that reads the captured PNG
(`magenta_pixel_count(&rgba) == 0`). Everything else — the ground share (`:433`), the rung on every pixel
(`:448`), the ruler's disc and centroid (`:526`, `:534`) — reads `probe`, a SECOND image drawn by a SECOND
camera (`crates/client-render/src/lib.rs:2424-2441`) from SEPARATE entities (the twins,
`crates/client-render/src/terrain.rs:512-525`) with a SEPARATE material.

Failure scenario: the picture camera's terrain `StandardMaterial` handle is dropped, or the sun and the work
light are both absent, or `place_camera` returns early and the picture camera keeps last frame's transform.
The picture comes back BLACK. `magenta_pixel_count` counts only the exact missing-shader magenta
(`crates/client-harness/src/assert.rs:34-42`), so a black frame scores zero magenta. The probe is untouched
by any of those faults — its camera, its transform copy (`follow_probe_camera`), its material and its clear
colour are all separate. **The gate is green on a picture that renders nothing.**

Slice 7's `paint_share` could not do this: it demanded warm paint IN THE PICTURE, which demanded the
picture's own camera, material and light. The diff deletes it and puts nothing in the picture's place.

Fix: keep a picture-side floor beside the probe-side share — e.g. `content_present_fraction` over the
probe's terrain mask (already in `vd_client_harness::assert`), or assert that the pixels the probe calls
TERRAIN are not all one colour in the picture. One line, and it restores what the classifier bought.

## 2. DEFECT (high) — the probe cannot see occlusion, so the defect class that motivated it is now invisible.

The probe camera carries `RenderLayers::layer(PROBE_LAYER)` (`lib.rs:2436`), and only the terrain twins and
the ruler twin live on layer 1 (`terrain.rs:522`, `terrain.rs:772`). The dots, the avatar's own marker, the
realm boxes, the star sky, the reference scaffold and the egui HUD are all on layer 0 and are therefore
ABSENT from the probe.

So the probe's own header claim — *"a second picture, aligned to the first, in which every pixel says WHAT
drew it"* (`crates/client-harness/src/probe.rs:10-11`) — is false. It says what the terrain-and-ruler layer
drew there, and stays silent about anything the picture draws in front.

Failure scenario, measured once already in this arc: `crates/client-harness/src/camera.rs:376-379` records
that on the first ground picture *"the marker filled the lower half of the frame"*. Re-introduce that today —
the avatar's own marker sprite, or a realm box outline, covering the lower band — and the picture shows a
marker while the probe still reports `share = 1.000` terrain at the right rung. The gate is green.

Fix: a THIRD kind in the codec (`PROBE_KIND_*` has spare codes) written by a probe material on the marker,
the dot and the box prims, so the probe covers every drawn thing. Until then, the header and §13.1 of the
slice document must say what the probe does NOT see.

## 3. DEFECT (medium) — `assert_eq!(stamp.chunks_drawn, state.terrain_chunks_drawn)` is tautological AND a flake.

`terrain_pictures.rs:457`. Both sides come from one expression, `terrain.entities.len()`: `terrain.rs:530-533`
stores it into `net.terrain_drawn` (the atomic behind `state.terrain_chunks_drawn`) inside `sync_terrain`, and
`terrain.rs:711` copies the same `terrain.entities.len()` into the stamp, also inside `sync_terrain`. The
assert therefore proves nothing about the picture.

Worse, it can fail. The two values leave the render thread at DIFFERENT points of the frame — the atomic in
`sync_terrain`, the stamp only in `place_chunks` (`terrain.rs:807`, `:854-856`). The core thread samples
`DevState` from another thread, so a sample taken between the two reads the atomic of frame N and the stamp
of frame N−1. While chunks are still arriving the counts differ and the gate goes red with the message "the
stamp's chunk count is the state's" — which names no cause.

Fix: delete the assert, or make it real by putting the frame number on both sides.

## 4. DEFECT (medium) — the stamp the gate judges is NOT from the captured frame, and nothing asserts the stand is still.

The state file is written by `record_capture` (`crates/bins/src/bin/client.rs:1355-1373`) AFTER the capture is
served, from `current(handles)` — a fresh poll on the core thread. Its own comment already calls it *"a
best-effort diagnostic snapshot of the delivered world AROUND the capture"* (`client.rs:1364-1368`).

8p now hangs all of M8-4 on it. The gate reads `stamp` from that file, builds `camera` from that file's pose
(`terrain_pictures.rs:472`), and compares both against a probe rendered at frame F, which the new pairing
rule only guarantees is *at or after the request* (`lib.rs:2503-2515`). Three different instants.

It is green only because the three stands are motionless, and nothing in the gate says so. The moment a
picture is taken during a leg — D8-5's pop detector flies a hull at 240 and 528 m/s and will want the stamp —
the ruler prediction is built from a pose the disc was not drawn at, and the 1 px assert fails for something
that is not a defect.

Fix: carry the render frame number in `DevTerrainStamp` and in `CaptureResult`, and assert they match. That
turns "the stamp belongs to this picture" from an assumption into the measurement item 3 pretends to be.

## 5. DEFECT (latent, medium) — the gate cannot correct for a spinning planet, and `DevRealmBox` carries no facing.

`terrain_pictures.rs:479` computes `eye_body = camera.eye - centre` with no rotation. The renderer does the
opposite: `terrain.rs:456-460` builds the eye in the body's frame through `body_frame_point(..., facing)`.
`DevRealmBox` (`crates/devproto/src/state.rs:50-76`) has `center` and no orientation field, so the gate
CANNOT do what the renderer does.

Failure scenario: the home planet spins — the terrain module's own header says the chunks "turn with the
planet because they are children of its row". The stamp reads the recipe at direction `facing⁻¹·d`, the gate
reads it at `d`. Altitude, surface and biome diverge by the terrain's own relief, hundreds of metres, and the
assert fires with "the stamp's altitude 3.382 m vs the state's 812.5 m". The RULER assert stays GREEN, because
`DevRuler.centre_m` is already facing-applied (`terrain.rs:684`). Half the gate breaks, the other half agrees,
and no message names rotation or facing.

Fix: state the facing on `DevRealmBox` (the renderer already reads that datum off the composed row) and use
it; or put the assumption in the assert message — *"the gate assumes the planet's facing is the identity; a
spinning body needs the row's facing on DevRealmBox"*.

## 6. DEFECT (latent, medium) — the ruler's prediction is the wrong formula; the 1 px tolerance hides it only at the three stands measured.

`projected_point_aabb` (`crates/client-harness/src/verdict.rs:243-248`) sizes the rectangle by projecting a
point offset along the camera RIGHT axis. That offset is perpendicular to the forward, so both points share a
view depth and `predicted = f·r/d`. The true silhouette of a sphere is `f·r/√(d²−r²)`.

At the three stands `r/d ≈ tan 2° = 0.0349`, so the error is `(r/d)²/2 ≈ 6·10⁻⁴` → 0.02 px on a 30 px disc,
and the 1 px tolerance is honest. It stops being honest the moment the `cell·0.5` FLOOR binds instead of the
angle (`crates/client/src/chunks.rs`, `let radius_m = (hi * RULER_TAN_HALF_ANGLE).max(cell * 0.5);`). The
floor binds for a hit closer than `0.5 / tan 2° = 14.3` cells — a rung-0 stand looking at a wall 10 m away, or
any picture taken indoors. At a hit of 4 m, `r/d ≈ 0.11`: the true disc is ~0.6 % larger than the prediction,
which on a ~100 px radius is 0.6 px, and at 2 m it is past the tolerance. The gate then goes red saying "the
ruler's disc measures X px, its projection Y px" for a FORMULA error, not a picture error.

Fix: predict `f·r/√(d²−r²)` (one line beside `projected_point_aabb`, or a `projected_sphere_radius_px` in
`vd_client_harness::probe` beside the other stamp formulas), or assert `radius_m/distance_m < 0.05` so the
gate refuses a stand where the approximation stops holding.

## 7. DEFECT (latent, low) — the bisection bracket is unsound when the FIRST sample is already below the surface.

`crates/client/src/chunks.rs`, in `ruler_on_surface`:

```rust
let mut above = 0.0_f64;
let mut t = cell;
while t <= reach_m && !below(t) { above = t; t += cell; }
...
let mut lo = above;   // assumed to be ABOVE the surface; never tested at t = 0
```

`below(0.0)` is never evaluated. When the eye itself stands at or under the DRAWN rung's surface — an eye
inside a hill, an eye at a coarse rung whose surface rises above the rung-0 surface the eye's height was set
from, an eye a hair under the mesh — the first sample at `t = cell` is already below, the loop exits at once
with `above = 0.0`, and the bisection refines a bracket whose "above" end is not above. It converges to
`hi ≈ lo ≈ 0`, so the hit lands ON THE EYE: `radius_m = max(0·tan2°, cell/2) = cell/2`, and the ball is
planted half a cell in front of the pilot's nose. The stamp then states a ruler the gate checks against
itself — both the disc and the probe distance come from that same ball — so nothing goes red.

Not hypothetical at the coarse rungs: the aloft test's own comment says the rung-9 surface "lies within the
dropped octaves' bound of the rung-0 surface the eye's height was set from", and that bound is what decides
the sign.

Secondary: the `below` closure divides by `p.length()` with NO `positive()` guard, unlike `eye_surface`,
which guards the identical expression. A ray through the body's centre yields a NaN direction, and
`Gf::from_f64` (`crates/terrain/src/gf.rs:69-71`) is a plain wrapper that passes NaN straight through — so the
answer is a silently wrong ruler, not a `None`.

Fix: test `below(0.0)` first and return `None` (the eye is not above the drawn ground, so there is no ground
to stand a ruler on), and guard the closure's length the way `eye_surface` does.

## 8. UNMEASURED — "THE BYTES ARE EXACT THROUGH AN sRGB TARGET" is measured for the R byte only.

`crates/client-render/src/probe.wgsl:9-16` states the round-trip argument and closes it with "(MEASURED by
the picture gate: every terrain pixel's rung byte must equal the rung the flag named)". That measurement
(`terrain_pictures.rs:448`) reads the R byte. **No assert anywhere reads a G or B byte for exactness.** The
only distance assert is `disc.cells_min` against the stamp within `RULER_CELLS_TOLERANCE = 2.0` cells
(`terrain_pictures.rs:551`) — a tolerance that absorbs a ±1 byte error by construction, and at rung 9 absorbs
1 024 m.

The argument is also weaker than the text: a driver's sRGB encode is specified to within a fraction of an
8-bit ULP, not exactly, so a G/B byte off by one is permitted by the platform and would be seen by nobody.
Per the standing ground rule the header should say "MEASURED for the kind and rung byte; the distance channel
is UNMEASURED" until a gate reads a known distance byte-exactly.

Cheap fix: the ruler's near edge is a known quantity. On the GROUND stand a cell is one metre — assert
`disc.cells_min == round((distance − radius)/cell)` exactly there, instead of within two cells.

## 9. UNMEASURED — three doc-comment numbers are contradicted by this same commit's own measurements.

- `crates/client/src/chunks.rs`, the `RULER_TAN_HALF_ANGLE` doc: *"on the ground it is a half-metre ball
  twelve metres off, from sixty kilometres up it is a kilometre ball two hundred kilometres off"*. The slice
  document's §13.4, written from the green flight, says the ground ball is **2 m across at 29 m** and the
  aloft ball is **17 km across at 245 km**. `200 000 × tan 2° = 6 986 m`, not 1 000 m.
- `crates/client-harness/src/probe.rs:32-33`: *"a ruler ball 16 m off covers 33 px of radius — which is what a
  half-metre ball at 16 m projects to."* At 45° over 720 rows the focal length is 869 px, so a half-metre ball
  at 16 m projects to **27.2 px**, and the three measured discs are 30.55 / 30.85 / 30.75 px. Neither number
  in that sentence is right.
- `crates/client/src/chunks.rs`: `RULER_TAN_HALF_ANGLE: f64 = 0.034_920_769_491_747_67` is a bare literal. It
  IS `tan(2°)`, but no test pins it, so a typo in the last digits moves every ruler in every picture and no
  gate notices. This is also the "no magic numbers" rule — the constant states a number, not a derivation.

Fix: correct the three sentences from the measured table, and add
`assert!((RULER_TAN_HALF_ANGLE - 2f64.to_radians().tan()).abs() < 1e-15)` to the chunks tests.

## 10. UNMEASURED — the ground-share floor was calibrated for the classifier 8p retired.

`terrain_pictures.rs:677, 691, 705` still read `min_share: 0.30`, unchanged by the diff, while the classifier
under it changed from "warm paint in the picture" to "terrain kind in the probe". §13.2 records the new
measurement: **the share is 1.000 on all three stands.** A floor at 0.30 now leaves 70 % of the band free to
go wrong before anything fires. It is not the same test at the same number.

Fix: raise the floor to what the new instrument measures (0.95 or so, with the measured 1.000 on record), or
state in the constant why 0.30 is still the right bar for the new quantity.

## 11. UNMEASURED — nothing checks that there is drawn ground under the ruler ball.

`ruler_on_surface` marches the RECIPE (`vd_terrain::height::height_m`); the picture draws the EXTRACTOR's
mesh; the ball is lifted two radii along the radial and is drawn whether or not its chunk has landed. The
ruler asserts (`terrain_pictures.rs:526-556`) compare the ball's disc against the ball's own stated placement
and the probe's distance against the ball's own stated distance — a closed loop. A ball floating over a hole
(a chunk still pending, or the ray leaving the square patch through a corner, where
`reach_m = radius·cell·CHUNK_EDGE` is the half-width and not the diagonal) passes every one of them.

The probe already holds the answer for free. Fix: read the terrain cells in a ring just outside the disc and
assert they sit within a stated tolerance of `disc.cells_max`. That measures the ball's contact with the
ground, which is the whole reason the ruler exists.

## 12. COVERAGE-TRAP (prediction — verify with `just coverage-fast`) — two derived `PartialEq` bodies in Tier-A are instantiated but never executed.

`vd-client` is Tier-A (`justfile:12`). `EyeSurface` and `Ruler` (`crates/client/src/chunks.rs`) both derive
`PartialEq`. The new tests compare them ONLY through `Option`, and only in the `None` arm:
`assert_eq!(eye_surface(&body, [0.0,0.0,0.0]), None)`, `assert_eq!(eye_surface(&body,[NAN,..]), None)`, and
three `assert_eq!(ruler_on_surface(...), None)`. `Option::eq` reaches `EyeSurface::eq` / `Ruler::eq` only in
the `(Some, Some)` arm, which no test takes — so both derived bodies are codegen'd (the impls are public and
required by the `Option` monomorphisation) with zero hits.

Contrast `DevTerrainStamp` in `vd-devproto`, which is safe: `state.rs:398-399` round-trips and does
`assert_eq!(back, state)` with both sides `Some`, so its derived body runs. `ProbeBlob` is safe for the same
reason.

Fix: one extra line per type — compare two equal `Some` values, plus one `assert_ne!` so the false side is
taken too.

## 13. COVERAGE-TRAP — the bisection's two `if under` arms are covered only by whatever the seed's terrain happens to do.

```rust
let under = below(mid);
hi = if under { mid } else { hi };
lo = if under { lo } else { mid };
```

Both arms must be taken for HR5. Whether they are depends entirely on where the height field puts the
crossing inside the bracket `[above, above+cell]` on the two successful calls in
`the_ruler_stands_on_the_drawn_ground_ahead_and_is_absent_for_a_sky_ray`. A crossing above every midpoint (or
below every midpoint) exercises one arm only. Today it is presumably fine; a change to the recipe, the
spectrum or the seed — slice 8a is exactly that — can turn `coverage-fast` red with no change to this file,
and the miss will read as an unreachable branch instead of as what it is.

Fix: a second unit test that brackets a KNOWN crossing (march from two chosen offsets so the mid falls on
either side), so both arms are taken by construction and not by luck.

## 14. NIT — `under_eye` ranks bodies by the LADDER FLOOR radius while calling it "the surface".

`terrain.rs:576-590`: `let over = DVec3::from_array(eye_body).length() - body.ladder().radius_m();`, under a
comment that says *"THE BODY UNDER THE EYE is the one whose surface is NEAREST the eye"*. Fifteen lines later
the same function's comment records the measurement that the floor radius stands **2 650 m under this
planet's recipe**. Two names, one quantity, and they are not the same quantity.

Failure scenario: two candidate bodies whose floor-to-recipe offsets differ by more than the gap between the
eye's two altitudes — a moon the eye is 1 km above whose recipe sits at its floor, against a planet the eye is
2 km above whose recipe rises 4 km. The wrong body wins and the stamp reads the wrong world. Unlikely at
29 000 km; a defect the moment the eye is between two close bodies.

Fix: rank by `eye_surface(body, eye_body).altitude_m` — the function is right there, and it is the quantity
the comment names.

## 15. NIT — the HUD states the wrong reason for an absent ruler.

`lib.rs:2166` prints `"none (the centre ray meets no drawn ground)"` whenever `stamp.ruler` is `None`. In
Windowed mode `Assets<ProbeMaterial>` does not exist (the plugin is added only in `run_capture`,
`lib.rs:2332-2333`), so `terrain.rs:669` never computes a ruler and the line always lies to the player at the
window: the ray met the ground perfectly well; there was no probe to read it with.

Fix: two arms — "none (no probe in this mode)" and "none (the centre ray meets no drawn ground)".

## 16. NIT — two `assert!(a && b)` in the gate, against the stated test discipline.

`terrain_pictures.rs:494-497` and `:501-504` each assert a band as `x >= lo && x <= hi`. CLAUDE.md HR5 (d)
says to split those. The coverage denominator excludes `/tests/` (`justfile:32`), so no gate fires — but the
rule exists because the short-circuit false side is the arm you want to see, and here it is the arm that says
WHICH end of the band the star fell out of. Splitting also improves the message.

## 17. NIT — the probe's distance saturates silently at 65 535 cells.

`probe.wgsl:45`: `clamp(floor(d/cell + 0.5), 0.0, 65535.0)`. Beyond `65 535 × cell` every pixel reads the same
number, and nothing distinguishes "saturated" from "exactly 65 535 cells". At rung 0 the ceiling is 65 km,
above the 372 m drawn today but below a 60 km stand at rung 0. `PROBE_MAX_CELLS` is exported but no reader
treats it as a sentinel.

Fix: reserve `65535` as "farther than this rung can say" and have `probe_blob` report it separately, or state
the ceiling in `ProbeBlob`'s doc so a future reader does not take a saturated max for a distance.

## 18. NIT — the probe and the picture do not rasterise the same silhouettes.

The probe camera declares `Msaa::Off` (`lib.rs:2439`); the picture camera declares no `Msaa` component and so
takes Bevy's default (4×). "Aligned to the first" therefore holds in the interior of a surface and not at its
edges. This is the RIGHT choice for byte-exactness — but it means the probe's edge pixels are not the
picture's edge pixels, and `equivalent_radius_px` measures a hard-edged disc against a soft-edged one. Worth
one sentence in `probe.rs`'s header: the difference is the size of the tolerance the ruler assert spends.

## 19. NIT — one quantity, two derivations of the eye's direction.

`eye_surface` and the `below` closure inside `ruler_on_surface` each compute `p / |p|` into three
`Gf::from_f64` calls, with different robustness (one guards, one does not — item 7). One helper, called twice,
removes both the duplication and the asymmetry.

---

## Verdict

The instrument is a real advance, and the two defects it found on itself — the sibling planet under the eye,
the readback older than its request — are exactly what it was built for. The stamp's formulas correctly live
in one Tier-A home shared by the renderer and the gate, and no law is broken. But 8p buys its precision by
narrowing what the gate looks at, and the narrowing is stated nowhere. After 8p the picture gate makes ONE
assertion about the picture the owner judges — that it holds no magenta — while everything it calls a
measurement is read off a second image, drawn by a second camera, from second copies of two kinds of object,
and compared against a stamp that comes from a state file the client polls after the frame is gone. A black
picture, an unlit picture, and a picture with the avatar's marker across the lower half all pass. That is the
instrument certifying its own silence, and items 1, 2 and 3 should close before 8p is called done. Items 5, 6
and 7 are latent, and each fires on a stand slice 8 already plans to take — a spinning body, a ruler under the
cell floor, an eye at a coarse rung's surface — with a message that names no cause. Items 8, 9 and 10 are the
honesty ledger: a byte-exactness claim measured on one of three channels, three doc numbers this same commit's
own table contradicts, and a threshold left at a value calibrated for a classifier that no longer exists.

---

## The answers (2026-09-09, after the refutation)

Every finding is answered below. "FIXED" names the change; "ANSWERED" gives the measurement or the
reason no change is made. The fourth flight of `terrain_pictures` ran after these changes.

| # | Answer |
|---|---|
| 1 | **FIXED.** The picture is judged WHERE THE PROBE POINTS: `paint_under_probe` reads the picture's own pixels under the probe's ground (slice 7's lit-paint classifier, ≥ 0.95) and under its ball (red over both other channels by 40, ≥ 0.90). A black or unlit picture with a perfect probe fails; so does a marker over the ground. |
| 2 | **FIXED by 1, and the claim corrected.** The probe knows the terrain and the ruler, and the gate's header now says so; what draws over them is caught by the picture's paint under the probe. Adding the marker, boxes and sky to the probe is not this slice's need (the pop detector reads terrain), and would be a second twin set per drawing. |
| 3 | **FIXED.** The tautology is deleted; the gate asserts `stamp.chunks_pending == 0` (the stamp was taken settled). |
| 4 | **FIXED for this gate.** The gate polls the state again after reading the file and asserts the delivered position did not move (1 µm): the stamp, the pose and the probe describe one moment. A moving leg (D8-5) uses the straddle, as the gate's constant says. |
| 5 | **FIXED.** `DevRealmBox` now carries the row's delivered `facing` (the same four numbers the renderer turns the terrain by), and the gate rotates the eye into the planet's frame by it. The identity is no longer assumed anywhere in the gate. |
| 6 | **ANSWERED, bounded.** The projection offsets a point along the right axis, `f·r/d`; the silhouette is `f·r/√(d²−r²)`. At the ball's `r/d = tan 2°` the difference is 0.06 % — a fiftieth of a pixel at 31 px — and it stays under one pixel until `r/d` passes 0.2, a hit under three cells where the half-cell floor binds. Written on the tolerance's doc comment. The prediction stays the marker gates' pair on purpose: one projection for every subject. |
| 7 | **FIXED.** `ruler_on_surface` tests the eye itself first: an eye at or under the drawn surface plants nothing (`below(0.0)` ⇒ `None`), with a test (an eye 5 m under the surface). The stamp's altitude states why. The `below` closure's division by a zero length yields a NaN comparison that reads "not below", which is a miss, never a plant. |
| 8 | **FIXED.** The G/B channel is now measured on both edges of the ball: the nearest pixel against `(d − r)/cell` and the rim against `√(d² − r²)/cell`, each within ONE cell. A wrong high byte (256 cells) or a wrong low byte fails. |
| 9 | **FIXED.** The doc numbers are the measured ones (a 1 m ball at 29 m; an 8.7 km ball at 245 km; 30.55 px), and `RULER_TAN_HALF_ANGLE` is pinned to `tan(2°)` by a test. |
| 10 | **FIXED.** The floor is 0.95 on every stand (measured 1.000). |
| 11 | **FIXED.** The probe's pixels three to eight rows under the ball's rim, at its centroid's column, must be terrain. |
| 12 | **ANSWERED by the gate.** `coverage-fast` on the first version reported two misses, both assert-message expressions (hoisted), not the derived `eq` bodies. The tests now also compare `Some(read)` and `Some(ruler)` by equality, so the derived bodies run either way. |
| 13 | **FIXED.** The bisection is branchless: the reading is a weight (`hi += (mid − hi)·u`, `lo += (mid − lo)·(1 − u)`), so no arm depends on where the seed puts the crossing. |
| 14 | **FIXED (comment).** The ranking is by the ladder floor, and the comment says why that is the same answer as the surface between bodies thousands of kilometres apart. |
| 15 | **FIXED.** The HUD names both reasons. |
| 16 | **FIXED.** The two band asserts use `RangeInclusive::contains`. |
| 17 | **ANSWERED (documented).** The two bytes saturate at 65 535 cells; the constant's doc says so, and the gate's distance check would read the saturation. |
| 18 | **ANSWERED.** The picture's MSAA blends the ball's edge; the probe's one-sample edge is the disc the gate measures against the projection, and the picture's paint under that disc is judged at ≥ 0.90 for this reason. |
| 19 | **ANSWERED.** `eye_surface` guards the zero length; `below` in the march cannot see a zero length except on a ray through the body's exact centre, where the NaN reads as a miss. |
