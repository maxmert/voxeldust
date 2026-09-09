# REFUTATION A of 01_reference_target.md (REVISION 2) — the laws and the code

**Date:** 2026-09-08. **Refuter:** agent A, domain 01. **Lens:** is every claim about the repository
true, is every law gate passed, does every number carry a source.
**Target:** revision 2 of `docs/investigation/2026-09-08/landforms/01_reference_target.md`, which
answers 56 findings from refutation A and refutation B of revision 1. This refutation reads revision 2
only. It repeats no finding revision 2 answers.

**Method.** I read the document in full. I then read `crates/terrain/src/{body,height,noise,chunk,
strata,lattice,gf}.rs`, `crates/seed/src/{ladder,rng}.rs`, `crates/client/src/chunks.rs`,
`crates/client-render/src/terrain.rs`, `crates/physics/src/taxonomy.rs`,
`crates/physics/src/worldgen/generate.rs`, `crates/core/src/frame.rs`,
`crates/sim/src/stub/placement.rs`, `crates/bins/examples/terrain_cost.rs`, `justfile`,
`docs/design/owner_decisions_2026-09-07_voxels.md`, `docs/design/DEFERRED.md` and
`docs/investigation/2026-09-07/slice_0{5,6,7}_*.md`.

**My own replay, written from the source and run independently of the document's.** I re-implemented
`SplitMix64`, `child_seed`, `top_rung_for`, `snap`, `Ladder::for_radius`, `draw_unit`,
`BodyDefinition::from_seed`, `noise3` and `height_m` in Python. It produces:

| Quantity | My replay | The document |
|---|---|---|
| Ladder radius | 3 350 759.045 m | the same |
| N at rung 0 | 5 263 360 | the same |
| Rungs / octaves | 12 / 14 | the same |
| Relief | 14 304.887 m | 14 304.9 m |
| `k_rough` | 0.4683 | the same |
| Coarsest wave | 400 000 m (the cap saturates) | the same |
| Sea offset | −5 297 m | the same |
| Amplitudes 0–3 | 7 605.55 / 3 561.97 / 1 668.21 / 781.28 | the same |
| Slope at 1 m over 20 000 columns | median 1.35°, p90 3.29°, p99 5.04° | 1.36° / 3.29° / 5.10° |
| Skew / kurtosis of the altitude | −0.009 / 2.811 | −0.031 / 2.795 |
| Share under the sea | 0.87 % (20 000 columns) | 0.99 % (200 000) |

**So the document's arithmetic is sound, and its diagnosis of a smooth planet is sound.** Every finding
below is about EVIDENCE, LAW, CODE STATE, a MISSING PART or a DECISION LEFT IMPLICIT — never about the
sums. Two of my findings come from measurements the document did not run; I state each one's method.

---

## Blocker

### B1 — P1, the headline gate, is not reproducible, and its headline claim does not survive a change of camera

**What the document claims.** §0, §1.4 and §6.1 print one number as the headline of the whole
investigation: *"the skyline's largest rise over its own local trend is 0.040°"* and *"the count of
breaks is zero at every threshold down to 0.05°"*. §6.1 makes P1 a GATE (D2 option (a)) at
**≥ 6 breaks per 60°**.

**What is wrong, in three parts.**

1. **The camera is not named.** §6.1 says *"from a camera 373 m above the surface at a named column"*
   and §1.4 says *"Ray-marched from a camera 373 m over the surface"*. No face, no chunk key and no
   column index appears anywhere in the document, and §7.1's stations are described as *"named by its
   chunk key"* while carrying no key. The 200 000-column samples of §1.3 and §1.4 name no RNG and no
   seed either. Slice 5 and slice 6 both name every chunk they time
   (`slice_05_generator.md:368-370`: *"the column is face PosX, chunk (3, 5)"*). This document's own
   §7.3 rule — a picture without a readout cannot be judged — applies to its own instrument, and the
   instrument fails it.
2. **The claim is camera-dependent, and I measured it.** METHOD: my own replay of `height_m`, the
   document's own method of §6.1 (720 bearings, 500 steps from 300 m to 90 km, the local trend as the
   median inside ±5° of bearing), at four camera columns, each 373 m over its own surface:

   | Camera direction | Skyline elevation range | Largest rise over the local trend | Crosses 0.05°? |
   |---|---|---|---|
   | (0.941, 0.188, 0.282) | −1.884° … +1.504° | **0.0536°** | **yes** |
   | (0.268, 0.894, −0.358) | −0.595° … +1.032° | 0.0439° | no |
   | (−0.531, 0.379, 0.758) | −1.919° … +1.189° | **0.0599°** | **yes** |
   | (0.097, −0.966, 0.241) | −2.192° … +0.072° | **0.0913°** | **yes** |

   The document's *"zero breaks at EVERY threshold down to 0.05°"* holds at its own unnamed column and
   fails at three of my four. The largest rise spans 0.044°–0.091°, a factor of 2.1, and the document's
   0.040° is the low end of that spread, not the planet's number. The document's own elevation range
   (−1.92° … +3.63°) does not appear at any of my four cameras either.
3. **The gate's band and its subject are both undefined.** §6.9 states plainly that the skyline break
   count has NO law from a body's facts. The band ≥ 6 therefore comes from §2's *"8–12 breaks in 60°,
   counted by eye"* off a screenshot of another game — the exact instrument §1.1 forbids one page
   earlier, *"a picture cannot carry a diagnosis"*. And the gate is written *"on an Earth-like
   planet"*, while §12 Q9 admits no body is known to be Earth-like and `vd-terrain` cannot reach a
   body's facts at all (§4.6). A gate needs a subject, an instrument and a band. P1 has none of the
   three fixed.

**Why this is a blocker and not a defect.** D2 recommends P1 as the FIRST numeric gate, and §0 calls
it *"the headline number"*. A gate built on an unnamed camera measures whichever column the author
picked; a later change that moves the camera by a kilometre will read as a terrain regression.

**Game example.** The pilot lands his hull on the home planet and walks to the station-2 ridge. The
gate says the skyline holds no break. He walks two hundred kilometres, stands on another column, and
the same gate now says a break exists. Nothing about the planet changed.

**What would fix it.** Name the column (face, rung, x, y) and the step count in the document. State the
sampling RNG and its seed for §1.3–§1.6. Then state the gate over a SET of named stations with a
stated spread, never over one column. This is U1's work, and U1 must inherit the names.

---

## Defects

### D1 — The strata cannot make a mesa or a hoodoo, because the code lays them parallel to the surface

**What the document claims.** §0 item 6: *"The strata … already exist and belong in the target … a hard
cap over soft rock is what a mesa and a hoodoo ARE"*, and §3.5: *"Half of the cheapest believability in
this document is already built and unused"*. D12 recommends the owner adopt it.

**What the code says.** `crates/terrain/src/strata.rs:187-210`, `StrataTable::at(biome, depth_m)`,
selects the substance by **whole metres UNDER THE SURFACE**, and the file's own first line says so:
*"which common substance a cell holds, by its depth under the surface and the biome"*. The order is
topsoil (1–4 m), subsoil (2–8 m), sediment (20–80 m), then bedrock forever (`body.rs:194-202`).

Three consequences the document misses:

1. **The layers drape; they do not bed.** Real bedding sits at a constant ALTITUDE. This table sits at
   a constant DEPTH under whatever the height field draws, so every layer follows the hill instead of
   cutting it.
2. **The hard layer is always at the BOTTOM.** A mesa and a hoodoo are a HARD cap over a SOFT layer.
   This table is soft over hard by construction — soil, then subsoil, then sediment, then bedrock. No
   erosion rule of any kind can leave a cap standing on it, because the cap material is never on top.
   The document names the mesa and the hoodoo as strata effects (§3.1, §6.8, D12) on a table that
   makes both impossible.
3. **A body draws ONE sediment and ONE bedrock.** `body.rs:198-201` draws one of three sediments and
   one of five bedrocks. So a cliff can expose at most FOUR distinct substances, not *"several"*
   (§6.8), and §2's reference row asks for *"bands 1–20 m thick"* against a single 20–80 m sediment
   block.

**Severity.** Defect: a false claim about the repository, carried into a recommendation the owner is
asked to adopt. The underlying idea is right — bedding is cheap believability — but it is NEW WORK
(altitude-anchored beds with a hardness per bed), not an unused asset.

**Game example.** The player stands under the red-rock rim the reference shows. On our planet the rim's
material is whatever sits 1 m under the local surface, which is the same dirt as the plain behind him,
and the granite is 90 m below his boots wherever he walks.

### D2 — The maximum slope is a sample extremum used as a property of the field; a two-second search beats it by 28 %

**What the document claims.** §1.4: *"The steepest column in 200 000 stands at 8.74°. The home planet
CANNOT hold a cliff, a pillar, a mesa rim, a fjord wall or a canyon … the field does not reach a third
of that [45°]."* §6.6 prints *"0 in 200 000 columns"* as the census answer.

**What is wrong.** 200 000 samples is 1.2 × 10⁻⁹ of the body's rung-0 columns
(6 N² = 6 × 5 263 360² = 1.66 × 10¹⁴). A sampled maximum is a lower bound on the supremum, and it grows
with the sample. My own replay shows both halves:

- At 20 000 columns I measure a maximum of 7.60° and a span of 15 577 m; the document at 200 000
  measures 8.74° and 17 003 m. Ten times the samples buys 1.1° and 1.4 km. The extremum is a function
  of the sample size, not of the planet.
- METHOD: I took the 20 steepest of 30 000 random columns and ran a plain local hill-climb on the full
  gradient (shrinking steps from 200 m to 5 cm). In **2 seconds** it reached **11.17°** — 28 % steeper
  than the document's stated maximum over ten times as many columns.

So *"the steepest column stands at 8.74°"* is false as written about the field; 8.74° is the steepest
of one sample. The document's own analytic estimate (§1.4: `G·ΣA/L = 2.97 × 0.1801 = 0.535` → 28.1°) is
the only support the "no cliff" conclusion has, and the document marks its `G` as a sampled maximum and
therefore UNMEASURED (U2). §1.3's *"realised span 17 003 m against Earth's 19 900 m"* compares a
200 000-sample extremum with Earth's true global extremum, which is the same error again.

**The conclusion survives; the evidence does not.** Even 28.1° is under 45°, so "no cliff" is very
probably right. The document must state it as the analytic bound with U2 attached, not as a measured
maximum. **"Never assume, measure"** cuts both ways: a sampled extremum dressed as a supremum is the
same defect as an argument dressed as a result.

### D3 — §1.2 re-counts the three radius-only quantities as independent checks, and nothing validates the replay's `noise3`

**What the document claims.** §1.2: *"Both refuters showed that test asserts only the rung count, the
octave count and the radius, all of which are functions of the radius alone and insensitive to every
draw. They were right. The pins above are the ones that could have failed."* It then lists: *"14 304.887
m, 12 rungs, 14 octaves, 3 350 759.045 m, a sea offset of −5 297 m, and 0.99 % … Five independent
checks, all equal."*

**What is wrong.** Three of the six listed values — 12 rungs, 14 octaves, 3 350 759.045 m — are exactly
the three the sentence above has just declared insensitive to every draw. `Ladder::for_radius`
(`ladder.rs:73-99`) reads the radius alone, and the octave count follows from `long_wave_m` saturating
at `LONG_WAVE_CAP_M` (`body.rs:152-153`), which the document itself proves happens for every seed on
this body. So the count of checks that could have failed is **three**, not five, and one of them (the
1-in-100 share) is a coarse statistic.

**The bigger half.** Every headline of revision 2 — 0.040°, 0.26 m, 8.74°, β = 3.19, the hypsometry —
is a function of `noise3`, and **NOT ONE of the listed checks tests `noise3` against the crate.** The
crate offers a ready pin: `golden_self_check` prints the home body's digest
(`slice_05_generator.md:388`: `0x9331e1fdfd902272`), and `terrain_cost` prints the surface height of a
named column. Reproducing either would have made the replay's noise a measurement that could have
failed. (My own independent replay of `noise3` agrees with the document's statistics, which is
evidence, but two replays that agree are still two replays.)

### D4 — The rain shadow is exactly as non-local as flow accumulation, and the document treats it as free

**What the document claims.** §4.7: *"The STANDING climate … is a function of the seed and of the body's
facts, and it belongs in the generator, because the terrain's SHAPE depends on it (a rain shadow has a
different shape, not only a different colour). **SL10 allows exactly that split.**"* D8 recommends it.

**What is wrong.** §5.6 is careful and honest about the river: *"how much water passes here depends on
the whole basin above, and a basin can be continental … it is not a bounded local question"*, and it
names three unknowns. Orographic precipitation has the same shape. To know the rain at a column you
must integrate the air's moisture along the wind path from the ocean, over every ridge the air crossed
— tens to hundreds of kilometres upwind. At rung 0 (62 m chunks) a 200 km fetch is 3 200 chunks in one
direction. That is not a bounded stencil, and SL10 does not "allow" it any more than it allows the
river; it is the same problem wearing a different word.

The document therefore prices the river's memory (§8.3: 0.24 ms to 2.1 ms, and it may not fit) and
prices the climate's at zero. **A wind-shadow field needs its own coarse-rung memory, its own stencil
and its own halo**, and the document must say so before D8 is put to the owner as *"SL10 permits it"*.

**Game example.** The pilot flies over the lee side of a range. If the rain there is a local noise, the
green stops at a random line. If it is a real shadow, it stops at the crest — and to know where the
crest was, the generator had to read 200 km of ground the chunk does not hold.

### D5 — D10 routes the spin through the placement lane, which SL1 makes a stamped, staleable reading

**What the document claims.** §4.5 and D10 (a), recommended: the spin and the obliquity *"ride the
PLACEMENT lane that already carries `angular_velocity` (`frame.rs:66`)"*, and §4.5 says the terrain
would receive it *"the way a placement arrives — one hop, stamped, read-only (SL1 clause 2)"*.

**What is wrong.** SL1 clause 6 says a stamped reading carries its instant and **a stale reading is
REFUSED, never used**. SL10 says the client may derive the static shape from `(seed, address)` alone,
*"never a function of time or live state"*, and the no-drift gate compares server-built and
client-built chunks byte for byte. A datum that (a) arrives on a lane, (b) carries an instant, and
(c) may be refused as stale is the exact opposite of an input to a byte-identical seed-derived shape.
If the desert belt of P7a is a function of a placement reading, then two hosts holding different
readings build different ground, and `just terrain-legs` cannot even express the comparison, because
one leg has no lane at all.

The document sees this problem clearly for the OTHER route — §4.6 shape 1 carries the whole
unfenced-arithmetic argument and the world-identity growth — and then loses it when the same fact
arrives by placement instead. The two routes need the same test: **a fact that shapes the ground must
be a seed draw or a fenced constant of the body, never a reading with an instant on it.** A lawful
form exists and is not named: the body DRAWS its spin from its own seed (like its relief), and the
placement lane merely reports the same number to whoever moves.

### D6 — §4.6 and §9's SL6 row miss the client lane, which is where the facts must actually cross

**What the code says.** `crates/client/src/chunks.rs:266-289`: the client builds its `BodyDefinition`
from a `SurfaceStmt` (the generator tag and `FrameRef::PlanetCentered { planet_seed }`) plus the look
shell's radius `r`. So the client already learns a body's seed and radius **over the wire**, and it
refuses a foreign generator tag.

**What the document misses.** §4.6 frames the whole facts question as a CRATE boundary
(`vd-terrain` may not name `vd-physics`), and §9's SL6 row lists three consequences: the facts to the
generator, the spin, and the weather. None of them is the one that actually binds: a `BodyFacts` record
that shapes the ground must reach the CLIENT too, which means a new field on the wire statement (HR1:
inter-shard bytes exist only as reviewed arms; SL6: ask before new data crosses), a postcard encoding
that is bit-stable, and a place in the world tag the handshake refuses on (V6 Part D). Without it, the
client derives one planet and the server another, which is D-TERRAIN-1's whole subject.

**Severity.** Defect: the hardest architectural consequence in the document is stated one boundary
short of where it bites.

### D7 — Four gates that are RED today have no home, and `just gate` would go permanently red

**What the document claims.** D2: make P1, P5, P6 and P8 numeric GATES now, and §6 states all four fail
today. §0: *"they read the crate's own `height_m` with no renderer in the path, and they are RED today
with the numbers above."*

**What is wrong.** Three things the document never decides:

1. **A gate that cannot pass blocks every merge.** `justfile:408` lists the pre-merge gate:
   `fmt-check lint … terrain-pin … test … coverage`. A gate that is red on the day it lands makes
   `just gate` red for every unrelated change until slice 8 or later. The document says *"a gate that
   cannot fail proves nothing"* and never asks the opposite question. A ledger row
   (`DEFERRED.md`, an expected-red with a named closing slice) is the shape the tree already uses, and
   it is not proposed.
2. **Where the code lives is unstated, and it decides two laws.** If the proxies live in
   `crates/bins/examples` beside `terrain_cost.rs`, they run only under `just`, NOT under
   `cargo test --workspace` — which contradicts §6's own reason for refusing a picture gate
   (*"it could not run in `cargo test --workspace`, which CLAUDE.md defines as the fast deterministic
   suite"*). If they live in `vd-terrain`, they are Tier-A and every branch of a histogram, a
   percentile and a log–log fit needs 100 % region and branch coverage (HR5).
3. **`terrain-cost` and `terrain-legs` are not in `just gate` today** (`justfile:408`), so "adding a
   gate" is itself a change to the pre-merge recipe that the document does not name.

### D8 — D2 gates on β while the document proves β is the wrong target, and the band is unreachable

**What the document claims.** §6.5 recommends *"GATE on β, because it is one number and it is cheap"*,
with the band 1.6–2.6, and D2 lists P5 among the four gates to adopt now. The same section then says
*"read §1.6 first: β alone is a poor target, because the β = 2 variant of our own field is unwalkable
and still holds no skyline. β is necessary and nowhere near sufficient."* §6.9 says β has **no law**
from a body's facts.

**Why that cannot stand together.** §1.6 measures that setting `k_rough` to 0.707 reaches β ≈ 2 and
puts **30.1 %** of the surface over 45° with **0.8** breaks per 60°. A gate on β therefore PASSES the
unwalkable gravel heap and FAILS a mechanism that buys landform without changing the spectrum much. A
gate must be able to fail the bad answer, not reward it.

Second, the band is unreachable by the recipe as a whole, not only by the home planet: `k_rough` is
drawn in `[0.45, 0.55)` (`body.rs:155`), so β runs 2.72–3.30 across every body the world can draw
(§1.5). A band of 1.6–2.6 is red for every planet at once, which makes it a version-bump alarm rather
than a landform test.

**Recommendation.** β belongs in the REPORT column with the break wavelength, exactly as §6.5's own
second sentence says.

### D9 — §9's record row states something the code does not do

**What the document claims.** §9, the record row: *"The target asks for no new per-cell field. Shape
rides the density byte, as today. **Climate rides the biome/object parameter, as today.**"*

**What the code says.** A generated cell is `chunk.rs:64-69`: `Cell { stratum: Stratum, gap: i8 }`.
There is no biome field and no climate field in a generated cell. `biome_at` (`height.rs:40-66`) is a
per-column FUNCTION whose only output is one of four `Biome` values, and that value is consumed
immediately to choose a `Stratum` (`strata.rs:187-209`). Ruling V6 Part B's approved twelve bytes are a
16-bit kind, the density byte, the rotation bits, the tree's shape byte and the attachment key — no
biome parameter is listed (`owner_decisions_2026-09-07_voxels.md:186-193`).

**Why it matters.** §9's V4 row says the terrain must give the asset skeleton *"a believable biome, a
slope and a **moisture** value to place assets by"*, and D8 asks for a standing climate with rain, wind
and a snow line. None of those has a home in the record or in the generated cell today. The document
closes the record question by asserting it is already answered. It is not; it is the one place the
target does touch the format, and it must say so as an ask.

---

## Weaknesses

### W1 — §8.2's example budget spends 32 of the 35 evaluations on prices nobody has measured

§8.2: *"an analytic derivative on every octave (about +14), a ridged and warped variant (about +14),
and a climate field of four slow noises (+4) — 32 of the 35."*

- **The warp is under-priced.** Domain warping moves the sample point with another noise. A 3-D warp
  needs a 3-D offset, which is **three** noise evaluations per warped octave: 42 for fourteen octaves,
  not 14. Warping once at a coarse level costs 3. Neither is 14, and the document does not say which
  it means.
- **The derivative's price is the document's own U7 (UNMEASURED)** and it is spent here as a result.
  A gradient-noise derivative re-uses the eight corner dots but adds the fade's derivative and eight
  more multiply-adds per axis; +1× is optimistic and nobody has run `noise3_d`.
- Ridging itself is free (a fold and a subtraction on a value already computed), so the row mixes a
  free operation with an unpriced one under one number.

The headroom of 1.89 ms and the 13.2 ns per octave-column are both sound (I re-derived 710 µs ÷
(3 844 × 14) = 13.19 ns, and `column_field` does walk `CHUNK_EDGE² = 3 844` columns,
`chunk.rs:172-186`). Only the shopping list on top of them is unsourced.

### W2 — §8.4 promises a BOOT cost and prices none

The section is titled *"What a FRAME costs, and what a BOOT costs"*, and it prices only the frame. The
tree has a measured boot number the document never uses: the golden self-check over 8 chunks costs
13.5 ms (`slice_05_generator.md:386`), it runs before a body is trusted, and `terrain_cost.rs:332-338`
runs it. A richer field raises it in proportion to the field work. The client's own boot — how long
after login until the first chunk is on screen — is the number a player feels, and it is neither
measured nor listed in §10.

### W3 — The gates' own arithmetic must be host-stable, and nothing says so

`just terrain-legs` runs the golden gate on aarch64 Linux and on emulated x86-64, byte for byte,
because D-TERRAIN-1 exists. The proposed proxies are threshold verdicts computed from unfenced `f64`:
P1 uses `atan2`, P4 a log–log fit, P5 a spectrum. Two legs may compute 5.99 and 6.01 breaks near the
band edge and disagree about the build's colour. The document never states that a gate's VERDICT must
be host-independent, nor that its margin must exceed the platform's own variance. This is cheap to fix
(quantise the statistic, or state the margin) and expensive to discover later.

### W4 — P6 counts "closed basins (a lake site)", and the world has no local base level

§6.6 counts *"closed basins (a hole with no outlet — a lake site)"* and §3.1 defines the base level as
*"the sea, or a closed basin floor"*. The code draws ONE sea radius per body (`body.rs:186-190`), and
nothing else holds water. A closed basin above that radius holds no water at all, so the proxy counts
holes, not lakes, and §3.1's *"or a lake"* names a mechanism (a per-basin base level) that neither
exists nor appears in §10 or §11.

### W5 — The 20 000 m long-wave floor's reach is stated wrongly

§9: *"the 20 000 m floor beside it, which binds on any body under 80 km of radius"*. `body.rs:151-153`
draws `long_share` in `[0.25, 0.5)` and then clamps. The floor binds when `radius × long_share <
20 000`, so it ALWAYS binds under 40 km of radius, and binds under 80 km only when the draw is near the
low end. "Any body under 80 km" is true only for the smallest draw.

---

## Notes

### N1 — The ladder's "exact" bound rests on an unproven property of the noise

§12 Q4 says *"Today `dropped_bound_m` is exact because the amplitudes are known"*. It is exact only if
`|noise3| ≤ 1`. The crate's own comment says the value is *"in about `[−1, 1]`"* (`noise.rs:85`), and
`relief_bound_m` claims exactness on the stronger reading (`body.rs:261-262`). My replay's sampled
maximum is 0.905 over 200 000 points; the document's is 0.98 over 400 000. Both are below 1 and neither
is a proof. The crate's own band test samples 400 directions (`height.rs:83-106`). The supremum of
`noise3` is UNMEASURED and belongs beside U2 — it is one paragraph of algebra on a quintic-faded
trilinear form, and the ladder's contract and the chunk band both lean on it.

### N2 — The tree line is missing from the vocabulary

§3.1 lists the snow line and the dune field, and §2 asks for *"forest as a mass"*. The reference
picture's strongest single orienter after the ridge is the line where the forest STOPS on the slope.
The tree line is a biome boundary the existing `biome_at` could almost draw today (temperature falls
with height at `height.rs:52-53`), and it is the cheapest scale reference on a hillside: a player reads
a mountain's height from where the trees give up. It appears nowhere in §3, §6 or §11.

### N3 — Four of §7.1's stations are unshootable, and §6's gates depend on them

§7.4 says stations 3, 4 and 5 cannot be shot and station 7's landform does not exist. §6.1's gate is
written *"from the ridge camera station"* (station 2), which §7.4 marks *"partly"* shootable. The
dependency is stated in §7 and not carried into §6, so a reader of §6 alone would take four gates as
runnable today. Only P2 and P3 are.

---

## Checked and sound

Every one of these I re-derived or re-read in this worktree, and each could have failed:

- **Every code citation in §13.** `body.rs` 147-149, 151-155, 158-184, 186-190, 194-202, 219, 253, 274;
  `height.rs` 16-27, 32, 35, 38-66; `home.rs` 16-23; `chunk.rs` 377-383; `lattice.rs:50` (`BOX_EDGE`);
  `noise.rs` 20-120; `ladder.rs` 73, 103, 109; `client-render/src/terrain.rs` 54 (`DEFAULT_RADIUS = 2`)
  and 210-214 (`ThreadedWorkers::start(available_parallelism())`); `taxonomy.rs` 324, 332, 387, 389,
  519, 681, 750, 756, 762, 794, 814 — all eleven land on the exact function named; `frame.rs` 66 and
  243-273; `placement.rs:149`; `terrain_cost.rs` 41, 345, 357. Revision 1's citation defect is fully
  repaired.
- **The three grep counts.** `obliquity|axial_tilt|rotation_period|spin_rate|day_length` over `crates/`
  returns exactly 2 hits, both comments. `powf|ln()|cos()|sin()|sqrt()` in `taxonomy.rs` returns 35.
  `atmosphere|haze|fog|scatter` over `client-render/src` and `client/src` returns 0.
- **The body replay**, to the last digit (the table at the head of this document).
- **The ladder arithmetic.** N = 5 263 360, 12 rungs, radius 3 350 759.045 m, 2 570 cells per face edge
  at rung 11, 42 chunks per edge, 10 584 chunks over the body, 39.6 million coarse cells,
  Σ_{L=1..11} 4^(−L) = 0.3333, 4¹¹ = 4 194 304.
- **The geometry.** Surface area 141.09 million km²; horizon 3.375 km at 1.7 m and 50.0 km at 373 m;
  `h = d²/2R` gives 373.05 m for 50 km; 1 m at 50 km is 0.0275 px at 45° over 1 080 rows, so a cell must
  be 36 m — rung 5 or 6, and rung 9 would put 14 px on a cell.
- **The frame integral of §8.4.** 2π/(62·c·θ_px)² · ln(50 000/62) gives 20 690 chunks at 1 px per cell
  against the document's 20 685; 3 448 in a 60° field; 8.4 GB at 405 KB per chunk.
- **The cost base.** The 8 ms budget and its gate on the costliest NAMED chunk at any rung
  (`terrain_cost.rs:41,345,357`); the named set really does include a rung-2 cave-dense seam chunk; that
  chunk really is 4.38 + 1.73 = 6.11 ms (`slice_06_extractor.md:352`); the headroom is 1.89 ms; the
  column pass is a printed 710 µs over 3 844 columns and 14 octaves, so 13.2 ns per octave-column, which
  agrees with the noise bench's 12.19 ns.
- **The strength ceiling.** σ = 2 700 × 9.81 × 8 850 = 234.4 MPa; the seven-body ceiling table; the
  code-against-ceiling table (0.02 % at 100 km, 32–97 % at home with the guessed density, 200–599 % at
  12 000 km). The withdrawal of the `1/g` law and the "relief is set by the process" reading are right.
- **The rulings.** V6 A5 (a landform at all eight cube corners) is at
  `owner_decisions_2026-09-07_voxels.md:184` and is approved; V6 Part D's radius move is at :203 and is
  decided-not-built (`body.rs:6-8` still calls the radius an input); S5-2 sets the identity tolerance to
  ZERO at :372, so D6's one-way-door arithmetic is correct even though Part D at :202 still says the
  tolerance is owed. D-TERRAIN-3 is at `DEFERRED.md:7775-7792` and says what the document says.
- **The cave-mouth defect.** `chunk.rs:377-383` builds the rule and `chunk.rs:613` applies
  `depth ≥ cave_min & depth ≤ cave_max` to caverns AND tubes alike, with `cave_min` drawn in 8..30 m
  (`body.rs:219`). Ruling V5's picture genuinely cannot be taken.
- **The overhang limit.** `height_m` returns one radius per direction (`height.rs:17-28`), so the arch,
  the undercut and the karst tower are impossible in the height field, and the collider consequence is
  right.
- **§1.6's direction.** My replay agrees that the field is band-limited and that its slope is the same
  at 1 m and at 1 000 m; the conclusion that roughness is not landform is sound.
