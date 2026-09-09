# Refutation B of `01_reference_target.md` (revision 2)

**Date:** 2026-09-08. **Refuter:** b. **Lens:** believability, cost and the owner.
**Questions asked:** would the result look like the reference picture; would a geologist or a
climatologist accept it; does it fit the budget and the boot; does it answer what the owner asked
(biomes from position, spin, trajectory, size and gravity; the weather simulated); is it written so
the owner learns the terms.

**Method.** I read the document in full, then read the code it names: `crates/terrain/src/body.rs`,
`height.rs`, `chunk.rs`, `strata.rs`, `noise.rs`, `crates/seed/src/ladder.rs`,
`crates/client-render/src/terrain.rs`, `crates/bins/examples/terrain_cost.rs`, and ruling V6 Parts A
and B in `docs/design/owner_decisions_2026-09-07_voxels.md`. I re-did the arithmetic that carries the
document's conclusions. Revision 2 is a large improvement on revision 1: most of the numbers are now
sound (the last section lists what I checked and could not break). The findings that remain are the
ones that still mislead the owner.

**Count:** 3 blockers, 8 defects, 6 weaknesses, 4 notes.

---

## Blockers

### B1 — The strata cannot make a mesa, a hoodoo or a bedded cliff, and the document says they can

**Where.** §0 item 6, §3.5, §6.8, §11 D12, §14.2 B-D12.

**The claim.** *"The crate draws 19 strata and a per-body table of topsoil over subsoil over sediment
over bedrock … Bedding is what makes a cliff read as rock instead of as a brown wall, and a hard cap
over soft rock is what a mesa and a hoodoo ARE … Half of the cheapest believability in this document
is already built and unused."* D12 recommends adopting it.

**What the code says.** `crates/terrain/src/chunk.rs:631` reads

```rust
let depth = h - r;                       // h = the COLUMN's surface radius, r = the cell's radius
… .at(site.biome, depth.floor().to_i64_floor().max(0) as u32)
```

and `crates/terrain/src/strata.rs:189-210` returns topsoil, then subsoil, then the sediment, then the
bedrock, **by depth under the surface**. So every layer is DRAPED parallel to the local ground, at a
constant thickness over the whole body.

Three consequences the document never states:

1. **A drape cannot cap anything.** A mesa exists because a hard layer lies at a fixed RADIUS and the
   soft rock under it wore back from the edge. A layer that follows the hill down its own side caps
   nothing: the cap is on the valley floor too. The same kills the hoodoo. `body.rs:194-202` draws
   `topsoil_m` in 1..4, `subsoil_m` in 2..8, `sediment_m` in 20..80 — **one number each, for the
   whole planet** — so there is no geometry in which a hard layer ends.
2. **A cliff exposes at most four bands, never bedding.** One sediment kind is drawn
   (`sediments[strata_rng.range_u64(0, 3)]`) and one bedrock kind (`Bedrock::ALL[…range_u64(0, 5)]`).
   So a cliff face can show topsoil, subsoil, one sediment and one bedrock. The reference picture's
   cliffs show bands 1–20 m thick, many of them. §6.8's proxy — *"print how many distinct strata the
   face exposes"* — has an answer bounded at 4 by construction, and the document does not say so.
3. **"19 strata" is a substance palette, not strata.** `Stratum::ALL` (`strata.rs:42`) holds 19
   entries, of which `Air`, `Water` and `Empty` are not rock at all, and 5 are the bedrock choices.
   The rock and soil vocabulary a column can ever hold is four cells deep.

**Why it is a blocker.** It is item 6 of the six things the document recommends the owner adopt, and
it is the only recommendation the document calls *already built*. A geologist looking at a
draped-strata cliff sees layers bending over a hilltop, which no sedimentary sequence does. The owner
would approve D12 believing he is buying bedding for free, and would receive a repainted brown wall.
D12 must be rewritten: the strata as they stand are a per-column soil profile; bedding and a hard cap
are a NEW mechanism (layers at constant radius, with per-layer thickness varying over the body), and
they cost octaves and a record decision the document has not priced.

### B2 — The headline number is measured on a ray-march that is truncated, and no convergence check is reported

**Where.** §0 first bullet, §1.4, §6.1, and every restatement of "0 breaks, 0.040°".

**The claim.** *"Ray-marched from a camera 373 m over the surface, over 720 bearings and out to
90 km … the skyline's largest rise over its own local trend is 0.040°, and the count of breaks is
zero at every threshold down to 0.05°. The skyline stands 86.4 km away (the median over bearings)."*

**The arithmetic that refutes it.** The march stops at 90 km, and the reported MEDIAN skyline sits at
86.4 km — 96 % of the cut-off. That is the signature of a truncated ray, not of a horizon. Two checks:

- The coarsest octave has a 400 000 m wavelength (§1.2). Ninety kilometres is **0.225 of one
  wavelength**, so over the whole march the ground can rise monotonically and never turn over. A
  march that never sees the far flank of the dominant wave has not found the skyline.
- Take a point 8 km above the ladder radius at ground distance `d` from a camera 373 m up. Its
  elevation angle is about `(8000 − d²/2R − 373)/d`. At `d` = 90 km that is
  `(8000 − 1208 − 373)/90 000` = 4.09°; at 120 km, 2.61°; at 150 km, 1.63°. The measured maximum,
  +3.63°, sits exactly where the 90 km wall is. The far skyline of this planet is being cut off.

**Why it is a blocker.** §0's first sentence, §1.4's conclusion and §6.1's whole gate rest on this one
number, and the document elsewhere insists that a statistic must be a measurement that could have
failed. This one could not fail in the way it was run: the instrument's range is shorter than the
feature it measures. Re-run at 180 km and at 360 km and report that the count and the 0.040° are
stable, or withdraw the number. The direction of the error is not obvious — truncation removes far
breaks and also flattens the local trend — which is exactly why the check is owed.

### B3 — The gate's target and the gate's measurement come from two different instruments

**Where.** §2's table, §6.1's table, §0's "at least six silhouette breaks", §11 D2.

**The claim.** §6.1 prints *"Reference picture (counted by eye) | 8–12 per 60°"* beside *"Our world
today, computed | 0 per 60°"*, and then proposes **≥ 6 per 60°** as a numeric GATE that D2 recommends
adopting now.

**What is wrong.** The 8–12 was counted by a person looking at a hand-authored game screenshot. The 0
came from a rule the document states precisely: *a local maximum of the skyline that stands 0.5° over
the median of the skyline within ±5° of bearing*. **That rule was never applied to the reference
picture.** A person counting "orienters" by eye counts a snow cap, a colour change, a haze band and a
tree line as well as a silhouette step; the rule counts none of those. So 8–12 and 0 are not the same
quantity, the ratio between them means nothing, and 6 is a number chosen between two incommensurable
readings.

The document's own marking scheme makes this visible: the 8–12 carries no mark. It is not MEASURED,
not REPLAYED, not PUBLISHED, not ESTIMATED. §13's own rule — *"No PUBLISHED number may enter a recipe
before a named source is attached to it"* — is broken by the headline gate's own calibration.

**What would fix it.** Ray-march the reference. Take the picture's own camera height and field of
view, trace the skyline by hand as a curve of elevation against bearing, and run the SAME break rule
on it. Then 8–12 becomes a number the gate can be set against. Until that is done, D2 asks the owner
to adopt a gate whose target is a guess.

---

## Defects

### D1 — P7a's "law" cannot compute the quantity it claims, and §6.9 lists it as a law that exists

**Where.** §6.7 P7a, §6.9's last row, §4.5.

**The claim.** P7a: *"compare the desert share's peak with the latitude `f = 2Ω sin φ` predicts for
the body's own spin."* §6.9: *"The desert latitude — **yes** — `f = 2Ω sin φ`, once a body draws a
spin."*

**What is wrong.** `f = 2Ω sin φ` is the **Coriolis parameter**: the sideways acceleration per unit
speed that a body's spin gives to air at latitude `φ`. It is a function OF the latitude. It does not
have a latitude as its output, and no rearrangement of it yields one. Solving `f = 2Ω sin φ` for `φ`
needs a value of `f` that nothing supplies.

The quantity the document wants is the **Hadley cell edge** — the latitude at which the air that rose
at the equator sinks again. On Earth that is near ±30°, and it comes from an angular-momentum
argument (the Held–Hou scaling), in which the cell's half-width goes roughly as the square root of the
thermal Rossby number and therefore SHRINKS as the spin rate rises. That is a real law with real
inputs (gravity, the scale height, the equator-to-pole temperature contrast, the radius and the spin),
and every one of those inputs is already in §4's list.

**Why it matters.** §6.9 exists precisely to separate "there is a law" from "we borrowed Earth's
number". Putting a false law in the "yes" column is worse than an empty row, because it tells the
owner that his own headline requirement — *biomes dependent on the spin* — is already solved on
paper. It is not. And P7a as written is not implementable: a programmer handed that row cannot
compute anything.

### D2 — §9's record row is false about the code, and the target does need new per-cell data

**Where.** §9's row *"The record (12 bytes, the density byte, the biome/object param)"*: *"The target
asks for no new per-cell field. Shape rides the density byte, as today. Climate rides the biome/object
parameter, as today."*

**What the code says.** The generated cell is (`crates/terrain/src/chunk.rs:64-69`):

```rust
pub struct Cell {
    pub stratum: Stratum,
    pub gap: i8,          // the signed radial gap in 1/128 of a cell
}
```

There is no biome, no moisture, no temperature and no wind in it. `biome_at` is called once per
COLUMN (`chunk.rs:637`) and its only effect is to pick which topsoil `strata.at` returns. The biome
value itself is thrown away and never leaves the generator. Ruling V6 Part B's twelve bytes are a
16-bit kind, the density byte, a 6-bit rotation, the tree's shape byte and the attachment slot — no
climate parameter is named in it.

**Why it matters.** The document's own §2 asks for *"forest as a mass"*, and V4 says vegetation is an
art asset placed by a server skeleton **from the biome**. §4.7 asks for a rain shadow, which needs
moisture. §6.7 P7c reads moisture on the windward and the lee side. A slope-and-moisture asset kit —
which §5.1 recommends by name from Star Citizen's ecosystems — needs at least a biome, a moisture and
a slope readable at a site. None of that is in the record or on the wire today. The law-gate table is
the one section whose job is to prove the target costs nothing it has not declared, and its record row
is unchecked.

### D3 — "Detail to the horizon" is answered with a pixel table, not a content table

**Where.** §3.4 *"What this costs in rungs — corrected"*, and §2's last row.

**The claim.** One metre at 50 km covers 0.0275 of a pixel, so a cell must be about 36 m, so **rung 5
(32 m) or rung 6 (64 m)**. The pixel arithmetic is right; I re-did it.

**What it misses.** `BodyDefinition::octaves_at` (`body.rs:253`) keeps *"the coarsest
`octave_count − rung`"*. At rung 5 the five finest octaves are gone, so the finest live wavelength is
**1 562 m** (§1.2's own table). At rung 6 it is 3 125 m. So a ridge 50 km away is drawn on a 32 m grid
carrying no shape smaller than a kilometre and a half — and D6's finer octave floor changes nothing,
because the drop is tied to the rung, not to the floor.

The reference's far ridges are the opposite: at 30–60 km they show sub-ridges and gullies of
200–500 m, which subtend 0.2–0.6° and therefore 5–14 pixels. Those are exactly the features the ladder
removes.

**The deeper observation the document never makes.** The finest live octave is always about 48 cells
wide, at EVERY rung: 48.8 m over a 1 m cell at rung 0, 1 562 m over a 32 m cell at rung 5. So the
ladder guarantees the same relative smoothness at every distance. That is a second, independent
statement of §1.4's "the slope is the same at every baseline" — and it means "detail to the horizon"
is not a rung question at all. It is a question about whether a coarse rung may carry structure the
free coarsening does not remove, which is a change to the ladder's contract and therefore an owner
decision the document does not raise.

### D4 — The four recommended gates are red on the shipped world, and no landing plan is given

**Where.** §11 D2 (*"Which proxies become numeric GATES now? … (a) P1, P5, P6, P8"*), §6.1, §6.2,
§6.5, §6.8.

**What is wrong.** All four gates fail today by the document's own numbers: 0 breaks against ≥ 6,
β = 3.19 against 1.6–2.6, 0 cliff segments against ≥ 1, and no A5 corner landform exists (§7.4 says
so). D2 recommends adopting them **now**. CLAUDE.md makes `just gate` (fmt + clippy + tests +
coverage) the pre-merge gate, and the project rule is "don't commit broken". A gate that is red from
the moment it lands blocks every unrelated merge until the whole landform mechanism ships — which, by
the document's own §5.6, is unsolved.

The document must say HOW they land: as a printed report that becomes an assertion when a mechanism
claims it, as an expected failure with a named ticket, or as a ratchet that only forbids getting
worse. It says none of these. That is the difference between a recommendation the owner can act on and
one he cannot.

### D5 — "Twelve times under even a 0.05° test" is arithmetically wrong, in the headline

**Where.** §0 second paragraph, repeated in §6.1 (*"it fails today by a factor of twelve even against
the finest threshold"*).

**The arithmetic.** 0.040° against the proposed 0.5° gate is 12.5× under. Against a 0.05° threshold it
is **1.25×** under — 20 %, not twelve times. The sentence attaches the factor of twelve to the one
threshold it does not apply to. What is true, and is the stronger statement, is that the BREAK COUNT
is zero at every threshold down to 0.05°.

### D6 — The headline statistic disagrees with itself between sections

**Where.** §0, §1.4 and §6.1 say the largest rise over the local trend is **0.040°**. §1.6's "Today"
row of the variant table says **0.045°**.

Both are presented as REPLAYED measurements of the same quantity on the same body. One of them used a
different camera, a different bearing set or a different march, and the document does not say which.
The number carries the whole diagnosis, so it must be one number with its parameters stated: the
column address, the eye height, the bearing count, the march limit and the step.

### D7 — The weather, the owner's second sentence, gets no target, no proxy, no cost and no decision

**Where.** §4.7, §12 Q6, §11 (no D row for the weather itself; D8 covers only the standing climate).

**What the owner asked.** *"It should be very believable, as we also should simulate the weather."*

**What the document delivers.** A five-row vocabulary table, a recommended split between standing
climate and moving weather, and four questions. §4.7 item 4 says *"What does it cost, at what rate, in
how many bytes? **Nothing is known.**"* No proxy in §6 measures weather — P7a to P7d all measure
standing climate. §11 holds sixteen decision rows and none of them is about the weather.

For a document whose stated job is *the TARGET* — *"what the picture must contain and how a gate
measures it"* — the owner's own second requirement leaves with no target at all.

**And one interaction nobody names.** The world's dormant-world design keeps 99 % of the galaxy off. A
planet realm that is not running cannot simulate anything. So the weather a pilot sees on approach
must either be a function of `(seed, time)` that needs no running realm, or the realm must spin up
before any weather is visible. That is a hard fork, it is the same fork SL10 forces on the shape, and
it is not asked.

### D8 — The named camera set has no light, so its pictures are not reproducible

**Where.** §7.1's seven stations, §7.2's overlay, §11 D1.

**What is missing.** The overlay states the realm, the seed, the world tag, the camera station, the
chunk key, the altitude, the latitude, the biome, the horizon, the rungs in frame, the draw radius,
the P1 count and a scale bar. It does not state **the sun's elevation and azimuth, or the time of
day**.

**Why that sinks D1's purpose.** Relief reads almost entirely through shading. At the measured median
slope of 1.36° a ridge under a sun 80° up is invisible; the same ridge under a sun 5° up throws a
kilometre of shadow and reads at once. Two shots of the identical terrain two hours apart are not
comparable, which is exactly what a regression set must be. And the reference picture is lit by a low
sun with long shadows — a large part of what the owner means by "believable" is the light, and this
document, which correctly names the haze as unowned (§2.1, D13), does not name the light at all.

---

## Weaknesses

### W1 — The strength ceiling is broken by the second row of its own table, and it is the wrong physics

**Where.** §4.2: *"The ceiling is a real bound: no body passes it by more than the uncertainty in σ."*

Its own table gives Venus's Maxwell Montes at **112 %** of the ceiling. A bound that its own
calibration set exceeds is not a bound; it is a fit with one outlier explained away. If σ may flex
12 % to absorb Venus, the headline consequence — *"600 % of the ceiling on a super-Earth"* — carries
the same 12 % of slack, which the document does not state.

The deeper objection a planetary scientist would make: `h ≤ σ/(ρ_c g)` is the crushing limit of a rock
COLUMN under its own weight. That is not what limits a mountain. Olympus Mons stands 21.9 km because a
thick, cold, stiff lithosphere carries the load in flexure, not because basalt is strong; Everest
stands where it does because uplift, isostatic rebound and glacial erosion balance. The document
half-admits this (*"relief is set by the PROCESS, and gravity is one term"*) and then keeps the
ceiling as the only physical anchor it offers. Label it what it is: an order-of-magnitude sanity
bound, good to a factor of two, not a law.

Two datum errors also ride in the table. Everest's 8 850 m is above sea level, Maxwell Montes's
11 000 m is above Venus's mean radius, and Olympus Mons's 21.9 km is above the Martian datum. Three
different rulers in one column.

### W2 — Earth's 19.9 km is a land-plus-ocean span; ours is 17.0 km of almost pure land

**Where.** §0 (*"its realised surface spans 17.0 km … against Earth's 19.9 km"*), §1.3.

Earth's 19 900 m runs from Everest (+8 850 m) to Challenger Deep (−10 990 m); over half of that span
is sea floor. The home planet has **0.99 %** of its columns under the sea, by the document's own
measurement, so its 17 003 m is essentially all dry land. Earth's dry-land span is about 9 300 m
(Everest to the Dead Sea shore). On the same ruler our planet holds close to **twice** Earth's land
relief — which strengthens the document's conclusion ("the relief is there; the problem is where it
sits") while showing that the comparison as printed is not the one that supports it.

### W3 — The single hump is blamed on the central limit theorem, which the document's own table rules out

**Where.** §0 fourth bullet (*"because the recipe adds fourteen independent noises and the central
limit theorem does the rest"*), §6.3.

§1.2 measures 78.1 % of the amplitude in the two coarsest octaves. Two dominant terms is not a
central-limit regime — there are about two effective degrees of freedom, not fourteen. The measured
kurtosis of **2.795** is *below* 3, which is the opposite of what a sum of many bounded independent
terms converging on a Gaussian gives; it is what a sum of a few bounded terms gives.

The finding (one hump, no shelf, no plain, no tail) is right. The reason offered is not, and the
correct reason is more useful to the owner: **the height distribution is the distribution of one or
two long waves.** It stays one hump however many fine octaves are added, and only a mechanism that
separates a base level from an uplifted block can make a shelf.

### W4 — A river needs water above the sea, and nothing in the world can hold it

**Where.** §2's river row, §3.1's base-level and delta rows, §12 Q5.

`BodyDefinition` holds ONE `sea_radius_m` (`body.rs:190,242`), and a cell holds a `Stratum` and a gap.
`Stratum::Water` exists, but nothing places water above the sea radius. So a river surface, a lake, a
tarn and a valley-floor pond have no representation at all — they are not expensive, they are
inexpressible.

Q5 asks whether the ocean is water the player swims in. It never asks the harder question: **what
carries the water surface of a channel that runs five kilometres above the sea?** That is a record
question (a per-column water level, or a `Water` stratum the generator writes), a collider question
(does the pilot's hull float on it), and an SL10 question (a lake level derived from the seed, or a
one-hop diff). The reference picture's river is one of its strongest landforms, and the document asks
for the channel without asking for the water.

### W5 — Four biomes, no snow, no forest, and no decision row for the biome vocabulary

**Where.** §4.5, §6.7 P7b, §7.1 station 6, §11 (no row).

`Biome` (`strata.rs:114-123`) has exactly four arms: `Desert`, `Grassland`, `Tundra`, `Highland` — and
`Highland` is a HEIGHT class, not a climate. There is no forest, no alpine, no ice cap and no wetland.
The reference picture needs at least forest, field, bare rock and snow in one frame.

Two consequences the document does not draw:

- P7b (*"the height at which the biome turns to permanent snow"*) cannot be computed for a reason
  simpler than the missing lapse rate: **there is no snow biome to turn to.** The document blames only
  the absent facts.
- §7.1 station 6 asks the picture to show *"the ice caps"*. There are none; the poles return `Tundra`.

The industry's own vocabulary for this is a **Whittaker classification** — biomes laid out on two
axes, the mean temperature against the mean rainfall. It is the natural home for §4.7's aridity and
for V4's asset kits, and it is absent from a §3 whose whole job is the vocabulary. §11 holds no
decision row asking the owner how many biomes there are and on what axes, which is a prerequisite for
every one of P7a to P7d.

### W6 — Six terms the owner is meant to learn are never explained; §8 still carries no game example

**Where.** §0, §1.3, §3.1, §6.3, §8; §14.2's answer to B-W2.

Glossed nowhere, though each carries an argument: **skew**, **kurtosis** (the two numbers that carry
§0's fourth bullet and all of §6.3), **rms**, **p99**, the **central limit theorem**, and the **angle
of repose** — which §3.1 uses inside its own definition (*"piles at its foot at the angle of repose"*)
without saying it is the steepest angle loose rock holds before it slides.

And §14.2 answers B-W2 with *"FIXED. Examples added to §1.6, §5.1, §5.6, §6.2 and §6.4"*, while B-W2
named *"§5, §6 and §8"*. §8 — the whole cost case, five subsections and six tables — still carries no
example in the game's words. A "FIXED" that does not cover what the finding named should not be marked
FIXED.

---

## Notes

### N1 — §8.4's table mixes a 60° count with 360° columns

The columns "One thread at 4.18 ms", "14 threads at 2 004/s" and "Mesh bytes at 405 KB" are computed
on the 360° chunk count, not on the "In a 60° field" column that sits immediately to their left:
20 685 × 4.18 ms = 86.5 s; 20 685 ÷ 2 004 = 10.3 s; 20 685 × 405 KB = 8.4 GB. The prose says "a full
360° vista", so the intent is right, but a reader taking the row across gets a number six times too
large for a frame. Label the columns.

### N2 — The 405 KB per chunk is a rung-0 measurement applied to every rung

`slice_07_client_link.md:235` measured 4.18 ms and 405 KB over 243 chunks at **rung 0** around the
ground spot. A rung-5 chunk carries five fewer octaves and a much smoother surface, so it should
produce fewer triangles and fewer bytes. §8.4 multiplies every chunk in the vista by 405 KB, so the
8.4 GB and the 2.1 GB are upper bounds of unknown tightness, presented as the wall. One measurement
settles it: extract one chunk at each rung and print its byte count.

### N3 — U1's "under 30 s" carries no arithmetic, and the fast suite builds debug

§10 U1 estimates the whole proxy bench at *"under 30 s of run time"*. My own arithmetic says release
is comfortable: P2 at 10⁶ columns × 3 baselines × 2 points × 14 octaves × 13.2 ns ≈ 1.1 s; P3 at 10⁶
directions ≈ 0.2 s; P5 at 100 000 samples ≈ 0.02 s; P1 at 720 × 500 ≈ 0.07 s. But §6's own rule says
the proxies must run in `cargo test --workspace`, which CLAUDE.md defines as the fast deterministic
suite and which builds **debug**; the project's own record says debug binaries miss release deadlines
by a wide margin. State the release requirement, or state the sample counts a debug build can afford.

### N4 — The ridge station is hand-picked, and D2 adopts the gate for a class Q9 says is undefined

§6.1's gate reads *"≥ 6 per 60° from the ridge camera station, on an Earth-like planet"*. The station's
address is not given and no rule chooses it, so a station picked after the mechanism lands fits the
gate to its own answer. And Q9 asks *"Which body is the reference for Earth-like?"*, noting that the
home planet may not be one. D2 recommends adopting the gate now, for a class the document leaves open,
from a viewpoint it does not name. Give the station a RULE — for example, the highest-prominence
column inside a named 100 km patch at a named address — so the address is a consequence and not a
choice.

---

## Checked, sound

These I re-derived and could not break:

- The relief clamp holds the PRE-factor: `clamp(0.004 R, 200, 12 000) × [0.5, 1.5)` gives 100 m to
  18 000 m (`body.rs:147-149`), and the home planet's 14 304.9 m is 0.427 % of its radius.
- The octave table. The weight sum `(1 − k¹⁴)/(1 − k)` at k = 0.4683 is 1.8807, so octave 0 is
  14 304.9 ÷ 1.8807 = 7 606 m; the two coarsest hold 78.1 %; everything from 5 km down holds 70.35 m;
  everything from 200 m down holds 3.06 m. All reproduce.
- `LONG_WAVE_CAP_M` saturates on this body: `radius × 0.25` = 837 690 m > 400 000 m, for every seed.
  Naming it as the magic number that causes the headline is correct.
- The horizon table. `√(2Rh)` gives 3.38 km at 1.7 m and 50.0 km at 373 m; `h = d²/2R` gives 373 m.
- The pixel arithmetic. `θ_px` = 45° ÷ 1 080 = 7.272 × 10⁻⁴ rad; 1 m at 50 km is 0.0275 px; a cell
  must be 36 m; rung 5 gives 0.88 px and rung 6 gives 1.76 px; rung 9 at 14.08 px is a staircase.
- The ladder. `n >> 11` = 2 570 cells per face edge, 42 chunks per edge, 42² × 6 = 10 584 chunks,
  39.6 M coarse cells; the 4:1 nesting is exact (2 570 × 2¹¹ = 5 263 360); `Σ 4⁻ᴸ` over L = 1..11 is
  0.333. Ruling V6 A4's "at most 64 × 64 chunks per face" is satisfied at 42.
- §8.4's chunk-count integral: `2π/(62·c·θ_px)² × ln(50 000/62)` = 20 686 at one pixel per cell, and
  the ÷4 and ÷16 rows follow. The 60° division is right.
- The budget. `terrain_cost.rs:41` sets 8 000 µs and `:356-360` asserts it on the **costliest named
  chunk at any rung**; 8.00 − 6.11 = 1.89 ms, and 1.89 ms ÷ (4 096 × 13.2 ns) = 35 extra evaluations
  per column.
- The client's worker pool is real (`client-render/src/terrain.rs:210-214`,
  `available_parallelism()`), and the 4.18 ms and 2 004 chunks/s are MEASURED at `slice_07:235` and
  used correctly (the 14-thread figure is 8.4×, not 14×, and the document does not pretend otherwise).
- No overhang: `height_m` returns one radius per direction (`height.rs:16-27`). No cave mouth: the
  carve band starts 8–30 m under the surface (`body.rs:219`, gated at `chunk.rs:377-383`). Both
  findings stand.
- `POLE_AXIS = 2` is hardwired (`height.rs:35`), and `biome_at` reads only the latitude, the height
  above the sea and two slow noises (`height.rs:38-66`). §4.5's statement is exact.
- `DEFAULT_RADIUS = 2` (`client-render/src/terrain.rs:54`) gives 5 × 5 chunks = 310 m.
- σ = 8 850 × 2 700 × 9.81 = 234 MPa, inside granite's published range. The calibration arithmetic is
  right even where the physics behind it (W1) is not.
- The cube corner's latitude: `asin(1/√3)` = 35.264°. Ruling V6 A5 is quoted correctly.
- The refusal of the learned/GAN approach under SL10, the SL5 handling of §1.6's four bench variants,
  and the move of the headline gate off a rendered frame onto `height_m` are all right. The last is
  the single best change in revision 2.
