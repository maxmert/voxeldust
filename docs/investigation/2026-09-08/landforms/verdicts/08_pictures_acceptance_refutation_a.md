# Refutation A of domain 08 — the laws and the code

**Date:** 2026-09-08. **Target:** `docs/investigation/2026-09-08/landforms/08_pictures_acceptance.md`,
revision 2. **Lens:** is every claim about the repository true, is every law gate passed, is every number
sourced?

**Method.** I read the code the document cites, line by line. I rebuilt the document's geometry
independently (the horizon, the dip, the exact projection of the horizon ring, the limb) from the camera
constants the code states. I compared the document's measurements with the sibling domain's. Every number
below carries its arithmetic or its `file:line`.

**The short verdict.** The protocol's SHAPE is right, and most of its arithmetic reproduces. Three things
break it. The verdicts that are advertised as the two-source half use a tolerance the document itself
calls never-reached, so they cannot fail. Eight of the sixteen judged pictures rest on cast shadows the
renderer switches off. And the instrument that is supposed to refuse an impostor cannot see one, because
the probe's id set names an author and the verdict needs a mesh.

---

## BLOCKER 1 — V2(b), V4 and V5(b) use a tolerance that cannot fail; V2(b) PASSES the very picture the document says it catches

**Where.** §9 V2 ("the relief bound is `relief_bound_m` at the rung,
`crates/terrain/src/body.rs:263`"), §9 V4 ("a mean absolute deviation under the relief's projected
bound"), §9 V5(b) ("under the relief bound's projected angle").

**The fact about the code.** `relief_bound_m(rung)` sums the live amplitudes
(`crates/terrain/src/body.rs:263-271`), and `from_seed` scales the amplitudes so that their SUM IS THE
RELIEF (`body.rs:180-186`). So on the home planet `relief_bound_m(0) = 14 304.9 m`, and
`relief_bound_m(3) = 14 301.9 m`. It is the same number the document prints as the octave table's total
in §3.3.

**The arithmetic.** One pixel is 0.0011506 rad (the document's own figure, and it reproduces from
`REFERENCE_VIEW_FOV_Y_RAD`, `crates/core/src/geometry.rs:1160`, over 720 rows). The tolerance in pixels
at eye height `h` is `(14 304.9 / horizon(h)) / 0.0011506`. I computed the sag by exact ring projection
at the same time:

```text
  eye height    the sag V4 measures   the stated tolerance   tolerance / signal
      300 m          2.80 px               277 px                  99
    1 000 m          5.12                  152                     30
   10 000 m         16.19                   48                      3.0
   30 000 m         28.12                   28                      1.0
   60 000 m         39.93                   20                      0.5
  300 000 m         92.25                    9                      0.1
```

**Two consequences, both fatal to the section.**

1. **V2(b) passes the slice-7 hill picture.** §9 V2 states: *"On the slice-7 hill picture (300 m up, a
   3 224 m patch) the skyline sits 69.4 pixels below the horizon row, and (b) fails."* At 300 m the
   stated tolerance is 277 px. 69.4 px is inside it. The verdict as specified REPORTS A PASS on the
   picture the whole document is built to refuse. This is the document's one claimed two-source verdict
   (§13.3: *"V2(b) compares the pixels with the body's radius, which are two sources"*), and it is inert.
2. **V4's applicability band is wrong at its low end by two orders of magnitude.** §9 V4 says *"The
   verdict runs above 151.6 m of eye altitude, where the sag passes 2 px."* At 151.6 m the tolerance is
   about 390 px against a 2 px signal. The tolerance only falls under the signal above about 30 km of eye
   height. The bisection that found 151.6 m answered "where is the sag two pixels", which is not "where
   can this verdict fail".

**Why the document should have caught it.** §3.3 says, of the same quantity: *"The theoretical bound
where all fourteen octaves align is far larger, and the measurement never approaches it; a bound the
field never reaches is not a fact about the picture."* The document then uses that bound as the tolerance
of three verdicts.

**The fix the document owes.** A tolerance from the MEASURED field, not from the aligned bound — for
example the p99 of the surface's height over the sea at the drawn rung, which the document already
measures (§10.2(c): standing deviation 2 295 m), and which at 300 m gives `2 295 × 3 / 44 839 / 0.0011506
≈ 133 px` at three deviations and still does not catch 69.4 px. The honest cure is a tolerance derived
from the relief WITHIN THE DRAWN RADIUS, not the whole body's.

---

## BLOCKER 2 — eight of the sixteen judged pictures rest on cast shadows the renderer switches off

**Where.** §7.4 (*"The reference picture's believability comes largely from raking light … the set asks
for TWO pictures of stands 2, 3, 4 and 7"*), §7.3 stand 5 (*"the low sun, LONG SHADOWS, the cap's
edge"*), §12 (sixteen judged pictures, *"stands 2, 3, 4 and 5 take two lights each"*).

**The fact about the code.** The terrain's sun and its fill light are both created with
`shadows_enabled: false` (`crates/client-render/src/terrain.rs:459` and `:473`), and so is the stub
world's key light (`crates/client-render/src/lib.rs:848`). Worse for the argument, the renderer adds a
FILL light deliberately aimed from the opposite direction, *"so the shadow side is never pure black"*
(`terrain.rs:466-478`), which flattens exactly the raking contrast the section asks for.

**What that means for the set.** A low sun on a shadowless renderer moves only the Lambert term on a
surface whose median slope is 1.37° (§3.3). A slope of 1.37° at a 10° sun changes the cosine by about
2 %. There are no long shadows, no shade in a hollow and no rim light on a ridge. The doubling of four
stands buys two near-identical pictures at different overall brightness.

**Why this is a code claim, not a taste.** The document's own rule (§0, line 22) is *"Every claim about
the code cites `file:line`."* The word "shadow" occurs ONCE in the whole document (line 634, stand 5's
promise) and never as a fact about the renderer. The document checked the clear colour, the material, the
frustum test and the alpha mode, and did not check the one switch its believability argument depends on.

**What is owed.** Either a row in §1 stating `shadows_enabled: false` and marking the two-light half OWED
beside V6 and V7, or a named owner of the shadow pass. The picture-set count falls from sixteen to twelve
until then.

---

## BLOCKER 3 — the probe cannot run V8(b), because §6.2's id names an AUTHOR and V8(b) needs a MESH

**Where.** §6.2 (*"The source id names the author: unauthored, terrain, water, hull, marker, star,
HUD"*) against §9 V8(b) (*"The id buffer names a real chunk mesh for every far terrain pixel … a card
has one id and one mesh handle, a chunk band has many"*).

**The contradiction.** Seven authors is a three-bit set. Under it every terrain pixel in the frame — near
chunk, far chunk, or a flat card standing in for a far band — carries the identical id `terrain`, and
V8(b) is unmeasurable. To count "how many meshes drew this band" the buffer must carry a PER-CHUNK id: at
the vista stand that is 2 805 distinct ids (§11), which fits `u16` but is a different artifact with a
different meaning. The change cascades: V3's "unauthored" arm, V5's "which mesh drew across the seam",
and §6.2's own 5.53 MB-per-frame price all read the id.

**Why it matters beyond the section.** §14's law-gate table passes ruling V10's no-impostor law with
exactly this verdict: *"V8's id test refuses an impostor by reading which mesh drew the pixel, which is
the machine half of ruling V10's 'no impostors'."* The law gate is passed by an instrument the document
defines two ways.

**One line fixes it.** State the id as `(author, mesh instance)` — three bits and thirteen — and re-price
§6.2. The document never states the decision, which is what a refuter must report.

---

## DEFECT 4 — "compared byte for byte on x86-64 and aarch64, exactly as the golden chunk digests already are" is false about the repository

**Where.** §7.2 (*"The no-drift gate (`D-TERRAIN-1`) extends to it: the found addresses of every stand
are compared byte for byte on x86-64 and aarch64, exactly as the golden chunk digests already are"*), and
§14's SL10 row (*"The byte-for-byte golden table (`D-TERRAIN-1`) stays the only no-drift gate"*).

**The fact.** `docs/design/DEFERRED.md:7721` marks D-TERRAIN-1 🟧, not 🟩, and its own table says the
x86-64 leg (G4) is **EMULATED** under the Mac and that *"it can find a drift and can never prove its
absence"*; the proper leg is *"a REAL x86-64 machine before SL10 clause 3 is called satisfied"*. The
golden table is equal on three legs of ONE target (aarch64 macOS) plus aarch64 Linux in the pinned image.

**Why it matters.** The document's SL10 pass rests on the golden table being an existing cross-chip
measurement. It is an existing SAME-CHIP measurement plus an emulation. The document hedges correctly two
sentences later (*"until it runs, the determinism is UNMEASURED"*), but the sentence a reader carries
away — "exactly as the golden chunk digests already are" — asserts a measurement that has not been taken
on the target that matters. The SL10 row must say that the only no-drift gate is itself owed a leg.

---

## DEFECT 5 — the float-fence argument for the search's home is an unmeasured determinism claim

**Where.** §7.2: *"a prominence computed there in plain `f64` and compared against another could flip
between x86-64 and aarch64, and the found stand would move between the developer's Mac and the cluster."*

**The fact.** The fence's own text refutes it. `crates/seed/clippy.toml:1-8` and
`crates/terrain/clippy.toml:1-8` say: *"Everything … must be a function of its inputs computed with
operations IEEE-754 FIXES ON EVERY TARGET: add, subtract, multiply, divide, square root, the
round-to-integral family, comparison, abs, clamp."* Those are exactly the operations a prominence uses. A
comparison of two such numbers cannot flip between the two chips. What the fence bans is the
transcendentals, `min`/`max` on signed zero, and `mul_add`.

**The true argument, which the document does not make.** An unfenced crate has NOTHING THAT STOPS a later
slice from writing `.sin()` or `.powf()` into the search, and no compile-fail control and no link scan
(`crates/terrain/src/gf.rs`, `just terrain-link-scan`). That is a real and sufficient reason to put the
search in `vd-terrain`. The reason printed is a claim about arithmetic that is not true, in a document
that says "NEVER a guess presented as a result".

---

## DEFECT 6 — the predicates are stated in degrees, and the fence bans the conversion

**Where.** §7.2's derivation table (*"'flat' — the body's own p10 slope at that spacing — 0.28°"*,
*"'steep' — p90 — 2.96° at 1 km"*, *"the sun's elevation — a low stand and a high stand at the body's own
p25 and p75 daylight elevations"*), §7.1 (*"9.4 % of directions hold a 1 km slope over 3°"*), §13.2 (the
stand search AND the stand predicates live in `vd-terrain`).

**The fact.** `crates/terrain/clippy.toml:9-45` bans `f64::to_radians`, `f64::to_degrees`, `f64::atan`,
`f64::atan2`, `f64::sin`, `f64::cos`, `f64::asin`, `f64::acos` and `f64::powf`. A predicate that says
"steeper than 2.96°" cannot be written inside the fence as written. A sun elevation cannot be computed
there at all; the slice-7 gate computes its own with `to_radians`, `cos` and `sin`
(`crates/bins/tests/terrain_pictures.rs:453-460`), which is lawful only because `vd-bins` is outside the
fence.

**What is owed.** Two sentences the document does not write. First: a slope threshold is stored and
compared as a RISE OVER RUN, never as an angle — "the rise over 1 562 m exceeds the body's own p90 ratio"
— and the degrees are printed by the stamp outside the fence. Second: the sun rule is not part of the
fenced search; it stays in the gate, as it is today. Without them §13.2's placement and §14's "no magic
numbers" row do not both hold.

---

## DEFECT 7 — the slope-quantile table is put in the body's derived table with no cost, no identity and no coverage analysis

**Where.** §7.2: *"The slope quantiles are themselves a function of the recipe: one pass of a stated
sample count over the ladder's addresses, computed once per body per recipe version and carried in the
body's own derived table."*

**The facts.** `BodyDefinition::from_seed` is called on the CLIENT for every planet the chunk lane meets
(`crates/client/src/chunks.rs:277`) and in the shard (`crates/bins/src/bin/shard.rs:404`). Today it is a
few dozen integer-hash draws and some fenced arithmetic. The document prices a comparable pass at
**1 000 000 addresses × ~390 ns = 0.39 s single-threaded** (§10.2(c)). A body is compared with
`PartialEq` (`body.rs:39`, the derive) and pinned by `crates/bins/tests/home_body_pin.rs`.

**Three questions left unanswered.** Who computes the table — `from_seed`, so every login pays it once
per planet, or a separate call the gate makes? Does it enter the body's identity, and therefore the pin
and the golden gate? And under HR5, does a quantile pass over a million addresses run inside
`vd-terrain`'s own unit tests, where the coverage build is many times slower? The document names one
HR5 arm ("no address satisfies") and misses the expensive one.

**The example, in the game's words.** A pilot logs in beside the berthed hull. The client's chunk lane
meets the home planet and builds its body. If the quantile table rides in `from_seed`, that login now
waits about four tenths of a second for a table the picture gate wanted and the pilot never reads.

---

## DEFECT 8 — "the home planet holds no cliff" is a statement about `height_m` presented as a statement about the drawn surface

**Where.** §3.3 (*"The largest slope measured anywhere is 7.23°. A cliff, a pillar, a mesa rim and a
canyon wall are all slopes over 45°. The home planet holds none of them"*), §7.3 stand 10 (*"OWED — NOT
BUILDABLE TODAY … The seed's shape holds none"*), §0.3.

**The facts that refute the generalisation.**

- The carvers open the surface. `crates/terrain/src/carve.rs:1-16`: *"a tunnel mouth is round"*, and the
  cavern field *"opens rooms"*. The document's own stand 12 finds *"a surface cell with a cavern cell
  under it"* — that is a hole in the ground with a wall around it.
- `crates/terrain/src/digest.rs:92-94` names, as the two cases slice 8's residency band exists for, *"a
  slope steeper than the margin (a cliff of more than 62 cells inside one column)"* and *"a cave mouth
  deeper than the margin"*. The generator's own code contemplates both.
- The drawn surface is the extracted mesh over one-metre cells, not the analytic height field. Where a
  cell is `Empty` or hollow, the extractor makes a vertical face (`crates/terrain/src/extract.rs:440-450`
  emits a quad wherever `is_rock` differs across a cell edge).

**What is true, and should be written instead.** The SMOOTH HEIGHT FIELD holds no slope over about 8°.
Everything vertical in the world today comes from the carvers or from the edit pyramid. That is a
narrower claim and it strengthens stand 11 and stand 12 rather than weakening them.

---

## DEFECT 9 — the limb radii in V4 are equiangular, in a section that says "exact projection"; the 2 000 km picture does not fit the frame

**Where.** §9 V4 (*"Where it applies — MEASURED by exact projection"*, then *"the picture's geometric
fact is the LIMB'S OWN RADIUS … 588 px at 2 000 km, 220 px at 10 000 km, against a whole body 441 px
across at 10 000 km"*), §0.2 (*"a disc 1 176 px across"*), §8.2 (*"the body is a disc 29.1 deg across
(441 px)"*).

**My reproduction.** With the document's own focal length of 869.12 px (720 rows at 45° vertical):

```text
  altitude      limb angular radius   exact rectilinear px   the document's px   error
  2 000 km          38.772 deg             698.1                  588.1          -19 %
 10 000 km          14.535 deg             225.3                  220.5           -2 %
```

588.1 and 220.5 are exactly `angle / 0.06592°`, the equiangular conversion. It is a small-angle formula
used at 38.8°.

**Why it changes what the picture shows.** The exact disc at 2 000 km is 1 396 px across, against a
1 280 × 720 frame. The limb leaves the frame on all four sides. So §12 order 14's *"The same where the
whole limb is in frame"* is false, and it already contradicts §8.2's own *"the limb leaves the frame's
sides"*. Stand 17's promise, *"the body as a body, the day/night line, the largest landforms"*, needs a
higher stand or a wider field, and the document should say which.

**Credit where it is due.** I re-derived the SAG column by independent exact ring projection and it
reproduces: 0.22 / 2.81 / 39.99 / 92.38 / 318.20 px against the document's 0.22 / 2.81 / 40.10 / 92.64 /
319.28. At 10 000 km the ring's greatest horizontal excursion is 233 px, well inside the 640 px edge
column, so UNDEFINED is right; at 3 144 km it is 656 px, so the bisection limit is right. The fault is in
the limb rows only.

---

## DEFECT 10 — the ground stand's own chunk count does not add up, and the showcase stamp does not match the priced table

**Where.** §11 M8-6 (*"502 columns x 3 chunks = 1 505 chunks = 610 MB = 25.0 M triangles"*) and §4.2
(*"rungs 0: 463 chunks | 1: 347 | 2: 347 | 3: 348 (four rungs, 1 505 chunks)"*, and *"The four rung
counts sum to 1 505, which is §11's priced residency"*).

**The arithmetic.** 154 + 116 + 116 + 116 = 502. 502 × 3 = **1 506**, not 1 505. Three times the M8-6
per-rung columns is 462 / 348 / 348 / 348, not 463 / 347 / 347 / 348. The two tables disagree with each
other and each disagrees with its own multiplication.

**Why it is worth a finding at all.** §4.2 says *"The internal checks the reader can run"* and prints six
of them. Five reproduce exactly (the 1 129 m of relief, the 6 426 m over the sea, the 3 474 m horizon,
the 4.00 m per pixel, the 43.5 px ruler — I checked each). The sixth is the one that ties the stamp to
the priced residency, and it is the one that does not.

---

## WEAKNESS 11 — the approach's path is not flyable, so V3 and V9 measure the harness rather than the game

**Where.** §8.2 (*"the descent falls by the same PERCENTAGE of its altitude each frame … 1 553 frames,
52 seconds at 30 fps"*), §8.3, §9 V9.

**The arithmetic.** One per cent of 10 000 km is 100 km. At 30 fps the first frame of the recording moves
the eye **3 000 km/s**, which is one hundredth of the speed of light. The last frame moves it 18 mm, at
0.5 mm/s. No suit and no hull in the world flies that path: a hull states its rated cruise and
acceleration as facts about what it is, and a suit states its own (the 2026-08-27 and 2026-09-05
rulings).

**What that does to the two verdicts.**

- **V3 (no hole in the ground, every frame).** At 3 000 km/s the chunk lane cannot feed the frame, so the
  recording's first hundred frames measure the residency lane's throughput, not a seam. §8.3 says the
  opposite: *"a chunk that arrives late is a hole in the ground, and a descent is exactly the case that
  makes chunks arrive late."* It is exactly the case — at a speed nothing can reach.
- **V9 (no pop).** SL8 is about what a PLAYER sees. A pop measured on optical flow no player can produce
  is not the SL8 gate for terrain, whatever its second difference does.

**The cheaper honest form.** Two recordings: one at a hull's own rated descent (the movement contract
already ships the forces), which is the SL8 gate; and the fractional sweep as a RUNG-COVERAGE instrument,
named so, because its only real job is to cross every rung boundary in one file.

**A small one on the way past.** 15.53 e-folds at 1 % per frame is `15.53 / ln(1/0.99) = 1 545` frames,
not 1 553; the document divided by 0.01.

---

## WEAKNESS 12 — the near ruler is a featureless dot, and three of them need three logins nobody prices

**Where.** §5.4 (*"The reference picture uses a character for scale; so do we"*), §5.5 (*"three avatars
along the line of sight, at 10 m, 40 m and 120 m"*), §17's answer to B-W9c (*"no hull realm is booted for
a ruler any more"*).

**The facts.** The avatar is a point-source sprite, not a figure:
`POINT_SOURCE_BASE_RADIUS_M = OCCUPANT_FIGURE_EXTENT_M = 0.5 m`
(`crates/client/src/realm_scene.rs:473`; `crates/core/src/look.rs:100`), scaled through
`marker_world_radius` (`crates/client-render/src/lib.rs:1275-1285`). §1's own table calls it *"a presence
dot"*. A sphere carries no size to a human eye — which is precisely why the reference picture uses a
human silhouette and not a ball. The document's own §5.2 measurement is sound; the leap from "an exact
instrument for the gate" to "the picture's scale figure for the owner" is not.

**And the cost the answer removed comes back.** Three avatars in one frame are three occupants, and the
harness places an occupant by logging a client in with `VD_SPAWN_POSES`
(`crates/bins/tests/terrain_pictures.rs:466-472`, one spawn entry per account). So the calibration stand
and the vista stand each need three logins, not one. M8-7 counts *"logins, cluster boots, GPU seconds"*
and the set's login count is never restated after §5.5 replaced one hull with three avatars.

---

## WEAKNESS 13 — the wobble measurement disagrees with the sibling domain's, and neither reconciles it

**Where.** §3.3 (a 3 473 m profile: mean tilt 1.60°, **RMS wobble 5.5 m** after the line is removed; a
44 839 m profile: 97.5 m) against `01_reference_target.md:275-280` (a ring of radius 1 000 m, a plane
fitted: **rms departure 7.049 m**, max 13.232 m, and 1.204 m at a 200 m ring).

**The arithmetic.** Domain 08's own two rows scale linearly with length (5.5 m at 3 473 m, 97.5 m at
44 839 m — a factor 12.9 for a factor 12.9), so the field behaves with `H ≈ 1`. Scaled down to a 2 000 m
baseline — the diameter of domain 01's 1 km ring — domain 08 predicts **3.2 m** against domain 01's
**7.0 m**. The residual should GROW with the baseline, and here the longer baseline reports the smaller
number. Both are labelled measured. Both come from separate rebuilds of the generator outside the
repository.

**Why it matters.** The 5.5 m figure carries the document's headline: *"A pilot on the home planet sees a
plane tilted 1.6°, with five metres of wobble on it."* If the sibling's number is the right one the
sentence reads "with ten to fifteen metres of wobble on it", which is a different picture for the owner.

**What is owed.** One reconciliation paragraph, or one bench inside `vd-terrain` that both domains cite.
The slope quantiles show the method works: domain 08's median 1.37° / p90 3.29° at 1 m and domain 01's
1.36° / 3.29° agree to two figures, and domain 01 even reconciles its ring tilt with its bearing slope
(*"a random bearing samples its cosine"*). The wobble rows got no such treatment.

---

## WEAKNESS 14 — the reference picture's largest mass has no acceptance, and its absence is written as compliance

**Where.** §14's V4 row: *"No verdict draws or judges a primitive tree."* §12: *"No picture with
vegetation before slice 14 — but every stand is re-taken after slice 14 with the same recipe."*

**The gap.** The reference vista is, by area, mostly forest. Ruling V4 makes trees, grass and decoration
ART ASSETS blended dynamically and placed by a SERVER SKELETON from the biome. That skeleton raises
exactly the questions this domain owns and does not name:

- Is the skeleton's placement a function of seed and address only, and is it byte-identical on two hosts,
  or does it cross as a diff (SL10)?
- Does the drawn density agree with the biome the stamp names — the V7 chain, one level up?
- Ruling V10 refuses impostors; a far forest is where an impostor is most tempting, and V8's id test is
  the only instrument against it (see BLOCKER 3).

Re-taking a stand after slice 14 is a COMPARISON, not a verdict. The document writes fifteen stands and
ten verdicts and leaves the picture's largest object with none.

---

## WEAKNESS 15 — the first-hit scan puts every stand in one corner of one face, and §4.2's address cannot come from it

**Where.** §7.2 (*"The scan order is face index, then `i`, then `j`. First hit wins"*), §7.1 (*"MEASURED:
9.4 % of directions hold a 1 km slope over 3°, so a first-hit scan ends after about eleven addresses"*),
§4.2 (*"stand vista-01 face +Y (0.0121, −0.8553) … the stand recipe, and the address it found"*).

**The consequence the document does not state.** At a 9.4 % hit rate the first satisfying address lies
within about eleven addresses of the scan's START — the same corner of the same face, for every predicate
and every body. The high ground, the low ground, the desert, the highland, the cave mouth and the vista
therefore all stand within a few strides of each other, and the face-seam stand is their neighbour. "The
comparison over time" then compares one corner of one face of one planet across every terrain slice.

**And the showcase contradicts it.** Face `+Y` at `(0.0121, −0.8553)` is not reachable by "face index,
then `i`, then `j`, first hit wins" at a 9.4 % density — an earlier face would have hit first. §4.2 is
honest that the address *"is the one the earlier draft printed"*, but the same line calls it *"the
address it found"*.

**The cheap cure the document should have chosen.** A stride derived from the body's own recipe (a
co-prime step over the face addresses) instead of a raster order, so the found stands spread over the
body and stay reproducible. That is still no argmax and still first-hit.

---

## WEAKNESS 16 — "no rung answers an orbit picture" is stated without the two facts that decide whether the ladder can answer, and the orbit prices change method silently

**Where.** §0.2, §11 M8-6, §14's ladder row, P8-11.

**What is missing.**

1. The top rung is not an oversight, it is a PACKING RULE: *"The top rung is the first rung with at most
   this many chunks along a face edge"*, `TOP_RUNG_CHUNKS = 64` (`crates/seed/src/ladder.rs:23,48-52`).
   A coarser rung means fewer than 64 chunks per face edge, which is a decision about the address space,
   not a missing number.
2. A coarser rung has no shape left to draw. `octaves_at(rung)` keeps `octave_count − rung` octaves
   (`body.rs:253-259`). The home planet has 14. Rung 11 keeps 3. Rung 13 keeps 1 — the single 400 km
   wave. `RUNG_MAX = 15` caps the address anyway (`ladder.rs:27`). So the coarse answer degenerates
   before it gets coarse enough, and the ladder's own bound `dropped_bound_m` grows to the whole relief.
   That is the real demand, and it is a landform demand, not a ladder one.

**And the prices change method between rows.** The ground and vista rows use THREE chunks per column (the
slice-7 shape, `13 × 13 × 3 = 507`, M7-3). The orbit rows do not: on the ladder's own cell counts, rung 9
has `6 × (10 280 / 62)² = 164 940` chunk columns and the visible cap is `h / (2(R + h)) = 4.11 %`, giving
**6 779** columns against the document's 6 793 chunks — one chunk per column. Rung 11 gives 1 927 against
1 978, and 3 865 against 3 964, again one per column. The document never says the depth factor changes.
If an orbit column needs two or three chunks like a ground column, the priced 2.75 GB at 300 km is 5.5 to
8.3 GB.

---

## WEAKNESS 17 — V3's definition hides the two holes a descent makes most often

**Where.** §9 V3: *"The count of unauthored pixels strictly BELOW the skyline row is zero"*, with the
skyline defined in §6.3 as *"the topmost terrain pixel per column"*.

**The two blind spots.**

- A column with NO terrain pixel at all has no skyline row, so the verdict skips it silently. That is
  every column at the frame's edge during a descent, and every column where a whole chunk column is late.
- A missing FAR BAND does not put an unauthored pixel below the skyline: the near patch's far edge simply
  BECOMES the topmost terrain pixel. That is exactly the slice-7 ground picture (§3.2), and V3 reports
  zero on it.

The document assigns the far band to V2(b), which BLOCKER 1 shows cannot fail. So today no verdict in the
set catches the defect the owner actually complained about. V3 needs a second half: the count of columns
with no terrain pixel below the projected horizon row, which is a count that must be zero and needs no
tolerance at all.

---

## NOTE 18 — one code claim in §4.2 carries no `file:line`, against the document's own rule

The stamp's sun row cites *"the brightest luminous row in the window — the very row the renderer lights
from (`S7-7`)"*. The claim is TRUE — `crates/client-render/src/terrain.rs:430-436` picks `brightest` and
aims the sun from it — but it is cited to a slice item, not to a line, in the one section that says every
number is computed and every claim is cited.

## NOTE 19 — one row of §3.2 does not reproduce

I recomputed the table with the same sphere and the same camera. Every row reproduces except the 60 km
one: I get a patch-edge depression of 11.9061° and a gap of 1.1432° = **17.3 px**, against the printed
11.8828 / 1.1200 / 17.0 px. The likely cause is arc against chord in the 396 800 m patch radius. It
changes nothing, but the table is labelled MEASURED arithmetic.

---

## Checked, and sound

Each of these I re-derived or read in the code, and each holds.

- **The horizon and dip table (§9.4).** `d = √(h(h + 2R))` and `dip = acos(R/(R+h))` with
  `R = 3 350 759 m` reproduce every row to the printed digit (`crates/terrain/src/home.rs`;
  `ladder.rs:103`).
- **The sag column (§9.4).** My own exact ring projection gives 0.22 / 2.81 / 39.99 / 92.38 / 318.20 px.
  The UNDEFINED at 10 000 km is right (the ring's greatest |x| is 233 px, inside the 640 px edge column)
  and so is the 3 144 km bisection limit (656 px there).
- **One pixel = 0.0011506 rad = 0.06592°**, from `REFERENCE_VIEW_FOV_Y_RAD`
  (`crates/core/src/geometry.rs:1160`) over 720 rows, and `one_pixel_world_m`
  (`crates/client-harness/src/camera.rs:273`).
- **The 144.9 m marker floor (§5.2).** `DOT_MIN_APPARENT_RADIUS_PX = 3.0`
  (`camera.rs:254`), base 0.5 m: `0.5 / (3 × 0.0011506) = 144.85 m`. The 43.5 / 10.9 / 3.6 px
  predictions all reproduce.
- **The hull refutation (§5.3).** Half extents 6 / 3 / 20 (`crates/bins/src/bin/build-ship.rs:164-166`)
  through `half.length()` (`crates/client/src/realm_scene.rs:537-540`) give 21.095 m; the cuboid draws
  scaled by the three half extents and turned by the parent's facing (`realm_scene.rs:831-843`) and
  blends when it is not luminous (`crates/client-render/src/lib.rs:1849-1856`). The 3.3× swing is real.
- **The octave table and the sea cross-check (§3.3).** The relief 14 304.9 m, the 400 km coarsest wave,
  `k = 0.4683` and fourteen octaves all follow from `body.rs:140-190`; the amplitudes sum to the relief
  by construction; the −5 297 m sea offset is published at
  `docs/design/owner_decisions_2026-09-07_voxels.md:393` and in `crates/terrain/src/chunk.rs:852`. The
  2 295 m standing deviation is consistent with `0.2701 × √(Σ a²) = 2 325 m`, and the 8 609 m the earlier
  draft implied is exactly `√(Σ a²)`.
- **The 12 % patch diagnosis (§3.2).** `GROUND_RADIUS = 6` and `CHUNK_EDGE = 62` give 403 m
  (`terrain_pictures.rs:55`; `ladder.rs:21`), and the 3.0 / 69.4 / 80.4 px gaps reproduce exactly.
- **The argmax arithmetic (§7.1).** `4πR² / 500² = 5.64 × 10⁸`; × 9 samples × 390 ns = 1 980 s = 33 min.
  The satisfying scan: 11 × 9 = 99 evaluations = 39 µs. The 9.4 % hit rate agrees with the document's own
  p90 of 2.96° at 1 km.
- **The residency products (§11).** 1 505 × 405 228 B = 610 MB and × 16 615 = 25.0 M triangles; the
  vista, the three orbit rows and the M7-3 citation all multiply out
  (`docs/investigation/2026-09-07/slice_07_client_link.md:236`).
- **Every code fact in §1 that I checked.** No frustum and no facing test in the terrain diag loop
  (`crates/client-render/src/terrain.rs:522-546`); one terrain material
  (`terrain.rs:362`); the word "biome" in neither client crate; `CLEAR_SRGB = [0,0,0]`
  (`lib.rs:78`); the six HUD lines shared by window and capture (`lib.rs:2004-2037`); the
  `second_rung` refusal (`crates/client/src/chunks.rs:335-341`); the aligned `state/<stem>.json` dump and
  `CaptureEntry.state_path` (`crates/bins/src/bin/client.rs:1331-1392`;
  `crates/client-harness/src/manifest.rs:22-38`); ten `clippy.toml` with none in `vd-client-harness`;
  `tier_a` at `justfile:12`; `vd-client` already depending on `vd-terrain` and `vd-seed`
  (`crates/client/Cargo.toml:10-12`); `DevRequest::Record` (`crates/devproto/src/dispatch.rs:46-51`);
  `differing_pixel_count`'s still-scene doc (`crates/client-harness/src/assert.rs:98-105`); the straddle
  and the presence law (`crates/bins/src/pixel.rs:1-24,148-183`).
- **V10 demoted to a seating check.** The chain in `terrain_pictures.rs:457-471` is exactly as quoted:
  the gate calls `height_m` itself and hands the number to the server as a literal. The two preconditions
  are the right two.
- **The SL6 ask for the three weather scalars (§4.4).** It states what data, from which realm to which,
  why the receiver cannot compute it, and what doing without costs. That is the whole gate, correctly
  applied.
- **Deleting the colour classifier.** `paint_share` is a fitted rule with a HUD mask of
  `x < 320 && y < 180` (`terrain_pictures.rs:266-289`), exactly as described, and a ten-line stamp does
  not fit inside it. The build order that puts the probe first is right.
