# Refutation B of domain 08 — the pictures and the acceptance (revision 2)

**Date:** 2026-09-08. **Lens:** believability, cost and the owner. **Target:**
`docs/investigation/2026-09-08/landforms/08_pictures_acceptance.md` (revision 2).

Revision 2 is much better than revision 1. It measures where it used to argue, and it marks five stands
OWED with the number that refuses them. This refutation does not repeat that praise. It states what is
still wrong.

Every finding below carries the file, the line and the arithmetic. Every number I recompute is shown.

---

## BLOCKERS

### B-B1 — The flagship at 373 m is the wrong reading of the reference picture, and it hides the failure it was made to find

**Where.** §0.3 first bullet, §7.3 stand 2, §12 orders 2 and 3, P8-6.

The document reads the reference picture as *"50 km of ground"*, inverts the horizon formula, and stands
the camera at 373 m. That step drops the reference picture's own mechanism. In the reference picture the
50 km comes from RELIEF, not from altitude: the far ridges are visible because they are TALL, and the
camera sits at a person's eye on a high place. The sibling document says the same thing in its own words:
*"Snow-capped ridges on the skyline — 1 500–3 000 m tall, 30–60 km away"*
(`01_reference_target.md:428`).

**The arithmetic the document owes.** From an eye at 1.8 m the horizon is 3 473 m. A point 50 km away is
visible when it stands above the horizon plane by

```text
   h' = (50 000 − 3 473)² / (2 R) = 46 527² / 6 701 518 = 323 m
```

So a 323 m prominence at 50 km is visible to a STANDING PILOT on the home planet. The reference picture's
1 500–3 000 m ridges are visible from the ground at 100–200 km. **The reference picture is a ground
picture.** Standing the camera at 373 m does not reproduce it; it produces a drone picture of a tilted
plain, and it makes the missing landform invisible to the very acceptance built to find it.

**Two consequences the document does not carry.**

1. The set loses the human vantage. The owner's picture holds a character on the ground. Ours would hold
   nothing on the ground at all (see B-D5).
2. The landform domains lose their number. The honest demand is a LANDFORM demand, not a camera demand:
   *the body must hold prominences of at least 323 m at 50 km from a 1.8 m eye, and 1 500 m ones to match
   the reference*. That single line is the most useful thing this acceptance could hand the landform
   slices, and P8-6 replaces it with a camera altitude.

**What to do.** Keep the 373 m stand as a DIAGNOSTIC (it shows the tilt cleanly). Make the FLAGSHIP the
pilot's own eye at 1.8 m, and print beside it the visible-prominence threshold (323 m at 50 km, 138 m at
33 km, 46 m at 20 km) against the body's own measured prominence at those distances. Then a flat world
FAILS the flagship instead of being flown over.

---

### B-B2 — The approach recording has no mechanism, and it never names the one seam this architecture really has: the realm crossing

**Where.** §8 in full, §12 order 15, P8-14.

§8.2 descends from 10 000 km to 1.8 m in 1 553 frames and 52 seconds. It never says WHAT MOVES THE EYE.
Three facts refuse every unstated option.

1. **The stands are placed by `VD_SPAWN_POSES`** (`crates/bins/tests/terrain_pictures.rs:457-471`,
   `D-TERRAIN-4`). That is a SPAWN, not a flight. A re-spawn per frame is a teleport, and SL8 calls a
   teleport a seam. A pop detector fed by teleports measures the teleports.
2. **A flight needs a rating.** The owner's suit ruling (2026-09-05) says a character moves in space only
   because it WEARS a suit that states its acceleration and cruise. The descent's mean speed is
   `10 000 km / 52 s = 192 km/s`. No suit and no hull states a rating near that, and the document asks
   for no measurement of what our rating gives.
3. **The descent CROSSES A REALM BOUNDARY.** An eye at 10 000 km is not inside the planet's realm; an eye
   at 1.8 m is. Somewhere on that path the occupant re-homes from the star system into the planet, which
   is the transfer machinery — HR2, and the seam SL8 exists for. **The document's approach never names
   the crossing.** It judges rung pops only (§8.3: no pop, no hole, the ladder's coarse answer, the
   horizon). The one moment a player would actually see a seam in THIS architecture is unjudged.

**What to do.** Say how the eye moves, on the shipped path, and add the crossing to §8.3 as a verdict:
the frame before the crossing and the frame after it differ by no more than the smooth-motion
distribution M8-5 measures. If nothing can fly the path yet, mark the approach OWED with its blocking
slice, exactly as the river bend is marked.

---

### B-B3 — GATE B is presented as *"a number, not an opinion"*, and it is neither reproducible nor sourced

**Where.** §0.2 (M-C), §11 M8-6, §13.1 GATE B, P8-11.

The document makes the whole picture set wait on GATE B, and GATE B rests on the table in §11. Three
faults in that table.

**(a) The two-pixel rule is an OPEN OWNER DECISION, borrowed silently.** §11 states the rule
*"draw each rung out to where its cell subtends two pixels"*, and §14 defends it under no-magic-numbers as
*"the camera's own pixel angle"*. The camera gives the ANGLE; the number 2 is typed. The sibling document
hands exactly this choice to the owner as decision **D15**, with the options priced:
*"(a) 1 px — 8.4 GB; (b) 2 px — 2.1 GB; (c) 4 px — 0.5 GB"* (`01_reference_target.md:1543`). Domain 08
picks (b) without citing D15.

The sensitivity is exact and cheap. The band radius is `d_r = 2^r / (k · θ)`, the chunk edge is
`62 · 2^r`, so the columns per annulus are independent of the rung:

```text
   columns per annulus = 0.75 π / (62 · k · θ)²      θ = 0.0011506 rad
     k = 2 px  ->  115.7 columns   (the document's 116)
     k = 4 px  ->   28.9 columns
```

**The cost falls by four when the owner answers D15 with (c).** The vista drops from 1 137 MB to about
285 MB. A gate that swings by 4× on an unanswered owner question is not a gate.

**(b) The three chunks per column are measured at rung 0 only, and are applied at every rung.** The
citation is M7-2/M7-3, and both measure the GROUND patch: *"9 × 9 columns, three chunks each"* and
*"a 13 × 13 × 3 patch"* (`slice_07_client_link.md:234,236`). At rung 0 a chunk is 62 m tall. At rung 7 a
chunk is `62 × 128 = 7 936 m` tall, and the document's own measurement says the relief across the whole
50 km vista is 968 m peak to peak (§3.3). **One chunk per column covers it.** Re-priced with a vertical
count that follows the rung, the vista falls from 2 805 chunks to about 1 620 chunks (657 MB) at k = 2,
and to about 400 chunks (165 MB) at k = 4. The headline *"1 137 MB"* is high by up to seven times.

**(c) The three orbit rows do not reproduce.** The document states the cap fraction and then a chunk
count, with no step between. The cap fraction is right (`f = h / (2(R + h))`), so I finished the
arithmetic:

```text
  stand        cap fraction   cap area (m²)   chunk edge   columns = area/edge²   the document says
  300 km          4.11 %        5.799e12       31 744 m         5 755                 6 793   (+18 %)
  2 000 km       18.69 %        2.637e13      126 976 m         1 636                 1 978   (+21 %)
  10 000 km      37.45 %        5.284e13      126 976 m         3 277                 3 964   (+21 %)
```

Every row is 18–21 % above the stated method, and no vertical multiplier is stated for these rows while
the ground and vista rows carry ×3. A reader cannot check the number that gates the set.

**What survives.** The LADDER half of GATE B survives untouched and is the document's best finding: at
10 000 km one pixel covers 11 506 m and rung 11's cell is 2 048 m, so no rung is coarse enough. Keep
that. Withdraw the byte half until D15 is answered and the vertical count follows the rung.

---

## DEFECTS

### B-D1 — The renderer casts NO shadows, so "raking light" buys almost nothing, and the set doubles for it

**Where.** §7.4 (*"The reference picture's believability comes largely from raking light"*), §12 rows 2/3,
§10.1.

The code refutes the premise. Every light in the client is
`shadows_enabled: false` — the terrain sun (`crates/client-render/src/terrain.rs:457-459`), the terrain
FILL light (`:471-473`) and the scene light (`crates/client-render/src/lib.rs:847-848`). Worse for the
purpose, the fill light is aimed from the OPPOSITE side of the sun and is on by design:
*"The shadow side is never pure black: a weak FILL light from the opposite direction"*
(`terrain.rs:467-470`, `FILL_SHARE`, `terrain.rs:55`).

So a low sun gives a Lambert term only, flattened by a fill. It cannot draw the long cast shadows that
carry the reference picture. §7.4 doubles four stands for this, which is up to eight extra pictures and
eight extra cluster logins, and §11 M8-7 then prices a wall clock nobody has run.

**What to do.** State the fact. Add one row to §1's table (shadows are off, the fill is on). Either make
the shadow map a named precondition of the two-light stands, or drop the doubling until it exists.
A picture pair that differs only by a Lambert term does not answer *"is it still interesting without
raking light"* — it answers a weaker question.

### B-D2 — The two-body verdict would PASS today for the wrong reason, and "move the orbital distance" is unbuildable under SL5

**Where.** §7.3 stand 14, §16 Q9, P8-7.

Two errors, one of law and one about the code.

**The law.** Q9 states the verdict as *"the difference must move when the orbital distance moves"*. SL5
gives one world. Nobody may MOVE a planet's orbit — the seed draws it
(`crates/physics/src/worldgen/generate.rs`). The verdict as written needs a variant world, which is the
thing SL5 forbids. The buildable form is an ORDERING over the bodies the world already holds: rank the
system's planets by orbital distance and require the tundra share (or the desert share) to fall
monotonically with insolation. That runs on THE world, over many bodies, and it can fail.

**The code.** Q9 says *"the two pictures MUST look alike today"*. That is false. Every body draws its own
biome field: `temperature_seed`, `temperature_wavelength_m` in 30–120 km, `humidity_seed`,
`humidity_wavelength_m` in 20–80 km, and `highland_above_m = 0.55 × relief`
(`crates/terrain/src/body.rs:203-215`). Two bodies therefore look DIFFERENT today — by an unrelated
draw. **So the pair verdict as written passes on a generator that ignores every fact the owner named.**
That is the worst kind of acceptance: one that goes green for the wrong reason.

**And two of the owner's five facts get nothing at all.** The owner named position, spin, trajectory,
SIZE and GRAVITY. §7.3 stand 14 tests orbital distance only. Size and gravity have no stand, no verdict
and no owed row. Gravity is cheap to state: surface gravity is `G·M/r²` from the drawn mass and radius,
and it sets the atmospheric scale height, hence the snow line and the dust. Say so, or say it is out of
scope, but do not leave the owner's own word unanswered in the table that answers his sentence.

### B-D3 — The biome mix is measured and never judged; a climatologist refuses this planet, and the document owns the tools to say so

**Where.** §7.3 stands 5/6/7, §2 (the Hadley cell row), §10.1.

The document measures the shares — Tundra 45.5 %, Grassland 37.0 %, Highland 14.3 %, Desert 3.1 % — and
never states that the mix is unbelievable. Read the code and the arithmetic falls out.

```text
   temperature = 1 − |z| + 0.35·n_t − 0.3·(above_sea / highland_above_m)      height.rs:50-52
   Tundra  when temperature < 0.35   ->  |z| > 0.65 + 0.35·n_t
   Desert  when temperature > 0.75 AND humidity < −0.1                        height.rs:53-64
```

`|z|` is `sin(latitude)`, not the latitude. Two consequences:

- **The tundra reaches 40.5° of latitude** (`|z| = 0.65` gives 40.5°), and takes 45.5 % of the body. On
  Earth that band holds Madrid and New York.
- **The desert can exist ONLY near the equator.** `temperature > 0.75` needs `|z| < 0.25 + 0.35·n_t`;
  with the noise's measured deviation of 0.27 that is `|z| ≲ 0.35`, or within about 20° of the equator.
  **That is the exact opposite of the Hadley cell**, which the document explains in §2 and then does not
  use: Earth's great deserts sit at 15–35°, and the equator is rainforest. Our world puts the desert
  where the rainforest belongs.

Add no rainforest, no savanna, no ice sheet distinct from tundra, and a sea covering 0.92 %, and a
climatologist refuses the body on sight. The document's §10.1 promises the owner a machine reading of
*"believable"* and offers only geometry (the ring fit, the seam, the haze). **The cheapest climate
verdict available today is missing: the desert's latitude histogram against a stated Hadley band.** It
costs one pass of the same sample §10.2(c) already takes, it can fail today, and it is exactly the
owner's first sentence.

### B-D4 — Weather and CLIMATE are never separated, and a still picture cannot accept a simulation

**Where.** §4.4, §7.3 stand 15, §16 Q6, P8-13.

The owner wrote *"we also should simulate the weather"*. The document's whole weather acceptance is one
still pair across a ridge (stand 15) plus three shipped scalars (§4.4). Two things are wrong.

**The split is never made, and it is the split the laws demand.** CLIMATE is a static function of the
body — latitude, insolation, the windward side of a ridge. Under SL10 climate is SHAPE, so the client may
derive it and it costs no boundary crossing. WEATHER is LIVE STATE — this hour's cloud, this hour's rain.
The seed ruling forbids deriving live state, so weather must be shipped, which is the SL6 ask of §4.4.
The document mixes them: it files the wet-green / dry-brown ridge pair (pure climate) under weather, and
it files the ask (pure weather) beside it. **Making the split explicit is the most useful thing this
domain can hand the weather domain**, and it is absent.

**A still picture cannot accept a simulation.** Nothing in §9 has a TIME axis except V9, which detects
pops. A weather acceptance needs at least:

- the same stand at two universe ticks, with the cloud moved and the rain moved;
- TWO clients at the same stand at the same tick, whose pictures agree — the one verdict that proves the
  weather is shipped rather than invented, and the exact reason §4.4 gives for the ask;
- the wind's direction against the body's spin (the Coriolis deflection the document explains in §2).

None of those is written. §16 Q6 lists what a fuller acceptance "would add" in prose. Write it as stands
and verdicts, like the rest of §9, so the weather slice inherits a gate instead of a paragraph.

### B-D5 — The rulers: "costs nothing" is false, three of them hang in mid-air, and a dot is not a scale figure

**Where.** §5.4 item 1, §5.5, §12 order 1.

Three faults.

1. **"This costs nothing: the avatar is already in every picture" is false.** The avatar in the picture is
   the OWN entity, and the camera sits at it — it is at the eye, not 10 m ahead. A scale figure 10 m
   ahead is a SECOND OCCUPANT: another account, another login, another `VD_SPAWN_POSES` entry, another
   seat. §5.5 asks for THREE of them. That is exactly the cost the document says it removed when it
   deleted the hull rulers (answer B-W9c: *"no hull realm is booted for a ruler any more"*). It moved the
   cost; it did not delete it.
2. **In the flagship they float.** The vista stand is 373 m over the surface (P8-6). Three avatars at
   10 m, 40 m and 120 m along the line of sight are three dots hanging 370 m up in the air. The document
   never notices the contradiction between §5.5 and its own new flagship altitude.
3. **A 0.5 m dot is not the reference picture's character.** §5.4 says *"The reference picture uses a
   character for scale; so do we"*. The reference uses a HUMAN SILHOUETTE, which a person reads without
   thinking. Our marker is a dot of `OCCUPANT_FIGURE_EXTENT_M = 0.5 m`
   (`crates/core/src/look.rs:100`), and at 10 m it paints a blob 87 px across in the middle of the frame.
   It is a fine INSTRUMENT for V1. It is not a scale figure, and the document should not claim it is.

### B-D6 — The arithmetic does not close, in four places, and two of them were marked FIXED

**Where.** §4.2, §11, §12, §7.4.

- **§11 ground stand:** `154 + 116 + 116 + 116 = 502` columns; `502 × 3 = 1 506` chunks. The document
  states **1 505** in three places (§0.2, §4.2, §11).
- **§4.2 stamp rung row:** `463 : 347 : 347 : 348`. From §11's own columns the split is
  `462 : 348 : 348 : 348`. The showcase stamp does not match the price it claims to sum to. Answer A-D7
  says this was FIXED.
- **§12 count:** the table lists fourteen judged pictures and one recording (rows 1–14 plus the
  approach). The text below it says *"Sixteen judged pictures and one recording"*, and §16 Q2 repeats
  sixteen. M8-7 prices the wrong count.
- **§12 versus §7.4:** §7.4 asks two lights for stands *"2, 3, 4 and 7"*; §12 says *"stands 2, 3, 4 and 5
  take two lights each"* AND its table gives one light each to rows 4, 5, 6 and 8. Three statements, three
  sets. Answer B-W3 says this was FIXED.

### B-D7 — The showcase stamp is the wrong stand

**Where.** §4.2.

The stamp is headed `stand vista-01` and then reads `eye 1.80 m over the surface` and
`horizon geometric 3 474 m`. The VISTA is the 373 m stand with a 50 km horizon (§7.3 stand 2, P8-6). The
document's own flagship change did not reach its own showcase. The internal checks below the stamp all
reproduce, so the numbers are right for a GROUND stand and the label is wrong.

### B-D8 — No picture is priced in TIME or in memory, though the repository publishes both inputs

**Where.** §11 M8-6, M8-7, §16 Q2.

M8-6 prices chunks, bytes and triangles, and defers the clock to M8-7 (*"the gate, once"*). The clock is
derivable now, from two measurements the repository already publishes: the 8 ms per rung-0 chunk budget
(ruling V10) and M7-2's client geometry at **4.18 ms per chunk on one thread, 2 004 chunks/s on 14**
(`slice_07_client_link.md:234`).

```text
  stand              chunks   one thread (12.18 ms/chunk)   14 threads   mesh held
  ground, 1.8 m       1 506        18.3 s                     ~1.3 s      610 MB
  vista, 373 m        2 805        34.2 s                     ~2.4 s    1 137 MB
  orbit 300 km        6 793        82.7 s                     ~5.9 s     2.75 GB
```

Two consequences the document should carry.

- The gate's own terrain deadline is **90 s** (`TERRAIN_WAIT_TICKS = 1_800` at 20 Hz,
  `crates/bins/tests/terrain_pictures.rs:63`). The 300 km orbit stand sits at 83 s single-threaded, inside
  the deadline only because the workers are threaded. That is worth stating rather than discovering.
- **2.75 GB of mesh is a memory statement, not a disk statement.** Bevy holds the mesh asset and the GPU
  buffer, so the orbit stand asks a developer's machine for several gigabytes of vertex data for a picture
  441 pixels across. The document says *"MB"* throughout and never says where the bytes live. On the
  owner's own machine (the swap-starved Mac) that is the difference between a picture and a wedged run.

### B-D9 — "Residency out to the geometric horizon" is the wrong rule for any world with relief, so both the verdict and the price are wrong

**Where.** §9 V2(a), §11 M8-6, §4.2 (`horizon geometric | drawn | resident`).

V2(a) and the whole M8-6 price stop terrain at the GEOMETRIC horizon. That rule is only correct on a
perfectly smooth ball. High ground beyond the horizon rises INTO the picture — which is the reference
picture's entire far half.

```text
   the visible distance to ground of height h' above the local surface:
   d = √(2 R h_eye) + √(2 R h')
   1.8 m eye, h' = 1 000 m  ->  3 473 + 81 861 = 85 334 m
```

So a pilot must be shipped terrain to **85 km**, not to 3.5 km, the moment the body holds a kilometre of
coarse relief — and the body's own relief bound at the coarse rungs is far more than a kilometre
(`relief_bound_m`, `crates/terrain/src/body.rs:263`). Under the document's own k = 2 rule the extra bands
(rungs 4 to 8) add about 580 columns, roughly 235 MB. That is a modest cost and a large correctness point:

- **V2(a) would pass a picture that hides every mountain**, because the residency it compares against
  already stopped at the horizon.
- **M8-6's ground price is too small in one direction while §11's headline is too large in another**
  (B-B3). Neither error is visible while the body is a tilted plain, and both bite on the first day the
  landform slices work.

The correct rule is derived, not typed: draw to `√(2R·h_eye) + √(2R·relief_bound_m(rung))`, which is the
recipe's own bound. State it in §11 and in V2.

---

## WEAKNESSES

### B-W1 — Domain 08 and domain 01 give the owner two different slopes for one body, and 08 claims they agree

§3.3 reports a median slope of 1.37° at 1 m, 1.31° at 62 m and 1.22° at 1 km, over 3 000 samples. The
sibling reports a median LOCAL TILT of **2.28° at a 50 m ring, 2.20° at 200 m and 2.05° at 1 000 m**,
over 200 000 columns with a fitted plane (`01_reference_target.md:280-286`). The two differ by up to
1.7×. One is a two-point slope; the other is the gradient of a fitted plane; the document never says
which it uses, and §3.3 states it *"agrees with sibling document 01_reference_target.md:190 in the same
words"*. It agrees on the no-cliff conclusion only. Reconcile the two, or name the two quantities apart
— *one word, one meaning*.

### B-W2 — The two documents use different frames, so their costs cannot be compared

Domain 08 uses 720 rows (`crates/client-render/src/lib.rs:71-72`, correct for the shipped window). Domain
01 computes with 1 080 rows (`01_reference_target.md:1451`). The pixel angle differs by 1.5×, so the two
cost tables and the two rung choices are not comparable. 08 should say so where it borrows from 01 (the
373 m target, the D15 options).

### B-W3 — Q8 asks the sea question and stops one step before the number that answers it

The document holds both halves already: the surface deviation is 2 295 m (§0.2) and the offset is
`(u·0.7 − 0.4) · relief` with `relief = 14 304.9 m` (`crates/terrain/src/body.rs:185-188`). The ocean
share is then a z-score, in one line:

```text
   ocean share = P(surface < sea)  ≈  Φ(offset / 2 295)
   offset −5 297 m  ->  z = −2.31  ->  ~1 %      (MEASURED 0.92 %; the sibling replays 0.99 %)
   an Earth-like 71 % needs  z = +0.55  ->  offset ≈ +1 263 m  ->  u > 0.698
```

**So about 30 % of bodies already draw an ocean world, and the home planet drew a puddle.** The owner can
then be asked for the thing he can actually judge — a target ocean share and a spread — instead of a
metres range he cannot picture. Q8 asks *"should it be bigger"* without this line.

### B-W4 — Stand 15 may be refused a second time, and the document does not measure it

Orographic lift needs a rise that cools the air enough to rain. This body rises about 1 000 m across
50 km (§3.3), a slope of 1.15°. Whether that produces a windward/leeward contrast a picture can SHOW is
UNMEASURED, and the document treats the stand as blocked only by the missing weather. State the lift the
contrast needs (a rise in metres over a stated distance, from the dry adiabatic lapse rate), so the
weather domain learns on day one whether this body can show a rain shadow at all.

### B-W5 — The recording's own bytes are unpriced; only the probe is counted

§6.2 prices 111 MB of probe on twenty stamped frames and calls the recording priced. The other 1 533
frames each write a PNG **and a full pretty-printed `DevState` JSON dump** — the capture path writes one
per capture, with no exception for a record sequence
(`crates/bins/src/bin/client.rs:1331-1366`, `write_state_dump`; `crates/client-harness/src/capture.rs:83-90`,
`state_rel_for`). A 1 553-frame descent is therefore 1 553 PNGs plus 1 553 state dumps, and the state dump
grows when the stamp is added to it. The frame count is inside the harness's own cap
(`MAX_RECORD_FRAMES = 3600`, `crates/bins/src/bin/client.rs:705`), so the run will not truncate — it will
just fill the disk unpriced.

### B-W6 — The argmax refusal leans on a rate the repository does not publish

§7.1 and §11 use *"the repository's own ~390 ns per column"* four times. I could not find that figure in
the repository. It is derivable from the column pass (1.5 ms per 62 × 62 columns = 390 ns), but the
document presents a derived rate as a published one, and it prices a single scattered `height_m` call at
the rate of an amortised 3 844-column pass. Cite the derivation, or measure the scattered call.

### B-W7 — Terms the owner is meant to learn are used without an explanation

§2 is good, and B-W8 of the last round is honoured. These are still used and unexplained: **prominence**
(§7.1, and it carries the argmax argument), **quantile / p10 / p90** (§7.2, everywhere), **e-fold**
(§8.2), **surface nets** (§7.3 stand 11), **second difference** (§9 V9), **straddle drift** (§9 V1).
And the document says **"standing deviation"** throughout where the industry says **standard deviation**;
one word for one meaning means the industry's word.

---

## NOTES

- **B-N1.** The 0.92 % sea share comes from 4 000 directions. The sibling replays 200 000 and reports
  0.99 % (`01_reference_target.md:262-268`). Cite the larger sample, or say why the smaller one is used.
- **B-N2.** §6.2's id set has no id for the atmosphere. V6 bins terrain pixels *"against the sky's
  colour"*, and the day the sky exists the probe needs an `atmosphere` id, exactly as it reserves `water`
  today.
- **B-N3.** §4.2's stamp shows `world declared 0x91af.. measured 0x33c7..`. The two halves are different
  quantities (`crates/client/src/net.rs:55-56,146`), so unequal values are lawful — but a reader who does
  not know that reads the showcase stamp as a failed world check. One word of explanation fixes it.

---

## CHECKED, SOUND

- The octave table (§3.3). Recomputed from `crates/terrain/src/body.rs:145-190`: the fourteen amplitudes
  sum to **14 304.9 m**, exactly the relief, and the relief itself is
  `clamp(0.004 R, 200, 12 000) × 1.192 = 12 000 × 1.192`. The coarsest wave is the 400 km cap. The
  roughness 0.4683 is inside [0.45, 0.55). Every row reproduces.
- The −5 297 m sea cross-check, and the standing-deviation model. `0.2701 × √(Σ a_i²) = 0.2701 × 8 608 =
  2 325 m` against the document's measured 2 295 m, and the 8 609 m it names as the refuted old model is
  exactly `√(Σ a_i²)`. Self-consistent.
- The camera arithmetic. `f = 360/tan(22.5°) = 869.12 px`; one pixel `= 0.0011506 rad = 0.06592°`;
  the 145 m marker floor (`0.5 / (3 × 0.0011506) = 144.85 m`, `crates/client-harness/src/camera.rs:254,264-266`);
  43.5 px at 10 m.
- The horizons, the dips and the 3.0 px gap at a 1.8 m eye (§3.2, §9.4). All reproduce, including the
  patch-edge depression `atan((1.8 + 403²/2R)/403) = 0.2593°`.
- V2 split into a residency self-check and a world check; V4's applicability band; V8's id test; V9's
  second difference; V10 demoted to a seating check. Each is right, and each names the code line that
  makes it right.
- The five OWED stands and their measurements: no hydrology, `Water` is not solid
  (`crates/terrain/src/strata.rs:91-95`), the largest slope is 7.23°, `biome_at` reads no insolation
  (`crates/terrain/src/height.rs:38-65`), no weather. The refusals are correct and honest.
- `highland_above_m = 0.55 × relief = 7 867.7 m` (`crates/terrain/src/body.rs:214`) matches the
  document's 7 868 m, and the 14.3 % highland share follows from it.
- The stand search moved into `vd-terrain` under the fence, striding integer ladder addresses through
  `vd_seed::bend::direction`, with M8-3 extending the no-drift gate. Correct, and correctly marked
  UNMEASURED until the gate runs.
- The one-rung refusal (`crates/client/src/chunks.rs:335-341`) as GATE A, and the frustum-free renderer
  loop renamed to the RESIDENT radius. Both verified in the code.
