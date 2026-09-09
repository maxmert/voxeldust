# Refutation A — domain 02, `02_planet_layout.md` **revision 2**

**Date:** 2026-09-08. **Lens:** the laws and the code. Is every claim about the repository true? Is
every law gate passed? Is every number sourced?

**Method.** I read the document in full. I read `crates/terrain/src/{body,height,chunk,noise,gf,strata,home,carve}.rs`,
`crates/seed/src/{ladder,rng,bend}.rs`, `crates/core/src/look.rs`, `crates/wire/src/version.rs`,
`crates/physics/src/{taxonomy,celestial}.rs`, `crates/client/Cargo.toml`,
`crates/client/src/chunks.rs`, `tests/tests/crate_isolation.rs`, `docs/design/DEFERRED.md`,
`docs/design/owner_decisions_2026-09-07_voxels.md` and the three slice reports the document cites.
I re-implemented `SplitMix64`, `child_seed`, `corner_hash`, the gradient table, the quintic fade,
`noise3`, the ladder snap and `BodyDefinition::from_seed` in Python, and I checked my replication
against the crate's own noise pin. **My replication returns
`noise3(2298, [0.25, 0.5, 0.75]).to_bits() = 13 827 097 110 060 728 320`, equal to
`NOISE_PIN` at `crates/terrain/src/noise.rs:235`.** Every number below marked COMPUTED comes from
that replication and could have failed.

This refutation replaces the revision-1 refutation at this path. The revision-1 findings F1–F30 are
answered in the document's §17 and I do not repeat them. The findings below are new, and I number
them **R1–R24** so they never read as the report's own D-rows or W-doors.

---

## Blockers

### R1 — The two headline recommendations are never composed, and their product is a world nobody can walk

**Where:** §4.6 change 1 (D3, the re-anchor) against §1.3 and §4.6 change 2 (D2, the roughness
modulation). **Severity: blocker.**

The report recommends both changes and prices each one alone. It never evaluates the two together.
I did. COMPUTED, on the home planet's own seed, with the crate's own noise:

| Table | Octaves | Coarsest amplitude ÷ wavelength | Finest octave | RMS slope @2 m | @32 m | @512 m |
|---|---|---|---|---|---|---|
| today (400 km, 30 m, `k = 0.4683`) | 14 | 7 605 m ÷ 400 000 m = **0.0190** | 48.83 m, 0.397 m | **2.01°** | 2.00° | 1.78° |
| re-anchored (50 km, 2 m, `k` unchanged) | 15 | 7 605 m ÷ 50 000 m = **0.1521** | 3.05 m, 0.186 m | **15.01°** | 14.35° | 12.20° |
| re-anchored **and** `k = 0.62` (D2's own target) | 15 | 5 440 m ÷ 50 000 m = 0.1088 | 3.05 m, **6.75 m** | **56.22°** | 40.03° | 21.25° |

Three consequences, none of them in the report:

1. **The re-anchor is not a spectrum fix that costs one octave.** The report sells it as "+13 ns (one
   octave)" and as "the direct cure for cause 2". The amplitudes are re-normalised to the SAME relief
   (`body.rs:176-184` divides by `weight_sum`), so shortening the coarsest wavelength by eight times
   multiplies the amplitude-to-wavelength ratio by eight at EVERY scale. The re-anchor alone takes the
   world from 2° to 15°.
2. **The two changes therefore do the same job twice.** Applying both gives 56° RMS at a 2 m baseline
   and a **6.75 m** octave at a **3.05 m** wavelength — a vertical wall under every boot. The report's
   §1.3 says `k = 0.70` "makes the whole planet unwalkable" at 40.8°; the recommended pair is worse
   than that everywhere.
3. **§1.3's evidence no longer applies to the report's own table.** §4.6 writes *"At an `r`
   corresponding to `k_rough = 0.62` the RMS slope is 14.92°"* and cites §1.3. §1.3 was computed on the
   400 km / 30 m table. On the re-anchored table the same `r` gives 56°. The report validates its
   recommendation against a table it also recommends deleting.

**What is missing, and it is one sentence.** The report never states the octave sum's new AMPLITUDE
budget. Today the octaves carry the whole relief (14 304.887 m). If the macro layers "own everything
above 50 km" (§Summary row 1), the octaves must carry far less — but §10.5 replaces the relief rule
with `h_max = σ_y/(ρ_c·g)`, which on the home planet is 16 808–23 564 m, that is MORE, and §9.3's band
rule still lists "Σ octave amplitudes" as a term with no value.

*Game example: the pilot lands on the home planet under the recommended pair. Every 3 m of ground
rises and falls by nearly 7 m. There is no walk, only a climb, and the collider stops the boots at
every step.*

**The fix the report owes:** state the octave relief budget after the macro layers land, re-compute
§1.3 and §1.6 on the re-anchored table, then put D2 and D3 to the owner as ONE decision with ONE
measured picture. M12 today decides only the floor, 2 m against 3 m.

---

### R2 — L4b is called "a lookup, never a search", but the data it looks up must be grown from the coast, and that growth is priced at zero

**Where:** §4.4b, §4.0's hydrology cell, §8.1's memory row, §9.1's `+256`. **Severity: blocker.**

The report deletes the 32-step march because a per-column search cannot fit the budget, and that
deletion is right. It replaces the march with `branches(body, cell)`, "a pure function of the cell's
address", and prices a column at "≤ 32 branch segments × ~8 ns = 256 ns".

**That price is the price of READING the list. The list exists nowhere.** There is no store: §8.1 says
"no cache required" and "0 new drift classes", and SL10 requires the client to derive the same answer
from `(seed, address)` alone. So `branches(body, cell)` must be COMPUTED, and §4.4b says how: *"seed
river mouths, then grow a tree inland, each node's elevation set by its PARENT'S plus a slope from the
stream power law"*.

A node's elevation is defined by its parent's. So the branches of one hydrology cell need the whole
chain from that cell back to a river mouth. On the home planet a hydrology cell is about 2 km, so a
1 000 km river is a 500-node chain, and a chunk far inland pays that chain before it reads its 32
segments. That is a search whose cost grows with the distance to the coast — the same class of cost
the march had, moved from the column to the cell and then left unpriced.

**The arithmetic the report owes.** Its allowance is 1 223 ns per column, and that frame is correct. A
500-node chain at even 50 ns a node is 25 µs. It is amortised over the 3 844 columns of the chunk
column only if the harvest is cached, and §8.1 calls the memory "an ESTIMATED 32–320 KB of harvested
hydrology cells" — a cache the report elsewhere says it does not need. Neither the build cost nor the
eviction cost appears in §9.1's table, and W9 hands the residency question to slice 8 without a number.

**Q2 does not cover this.** Q2 admits that seeding the MOUTHS is unsolved. It does not admit that
growing the TREE from a solved mouth to an inland cell is unbounded. That is a second and larger hole,
and D11 already recommends option (iii), which carries L4b.

---

## Defects

### R3 — The budget table does not add up: the rows sum to 436 ns, not 416

**Where:** §9.1, and the summary's layer table. **Severity: defect.**

§9.1's rows: 40 + 5 + 10 + 78 + 220 + 20 + 30 + 20 + 13 = **436 ns**. The table states **416**. The
missing 20 is L7, the biome: the row says `+20` and the subtotal does not carry it. With rivers the
total is 436 + 256 = **692**, not 672.

The summary's table disagrees again. Its rows are 5 + 40 + 88 + 240 + 256 + 30 + 20 + 195, and it
states "TOTAL ADDED: 416 ns without rivers". Without rivers those rows sum to **618 ns**, because the
L6 row carries **195 ns**, which is the octaves' cost TODAY (the MEASURED 185 ns per column,
`slice_05_generator.md:372-386`), not the 13 ns the change ADDS. A reader who checks the summary's own
arithmetic gets a third number.

Corrected by the report's own method: a cold surface chunk is 3.30 + 436 × 3 844 ns = **4.98 ms**
(report: 4.90); a cave-dense chunk with rivers is 6.10 + 692 × 3 844 ns = **8.76 ms** (report: 8.68).
The verdict does not flip. The headline number is wrong, and it is quoted in the summary, in §9.1, in
§9.2, in §9.4 and in §15.

### R4 — §6.2 and §9.5 contradict each other about L4a, and the rung-11 verdict is computed from the contradiction

**Where:** §6.2's cut-off table against §9.5's rung table. **Severity: defect.**

§6.2 states: *"L4a valley transfer | it MULTIPLIES the octaves, so it is never dropped (§4.6's rule) |
**never**"*.

§9.5 states: *"At rung 11 the ridge lines, the channel incision and the climate operators are all past
their derived cut-offs (§6.2), so only L1, L2, L3's profile and one extra octave remain"*, and prices
rung 11 at +261 µs = 68 ns per column = 40 + 5 + 10 + 13.

L4a's 240 ns is dropped there, and L7's 20 ns with it. If §6.2 is right, rung 11 costs
266 + (40 + 5 + 10 + 220 + 20 + 20 + 13) × 3 844 ns = 266 + 1 261 = **1 527 µs**, and the top rung is
**1.5 times** cheaper than rung 0, not the claimed **4.4 times**. The bench's own assertion (the top
rung cheaper than rung 0) still passes, so this is not fatal. But §9.5 is the ladder gate's evidence,
and it is computed from the arm of the contradiction that gives the nicer answer.

### R5 — The Mars ceiling row uses a crustal density the table says it does not use

**Where:** §4.3's ceiling table. **Severity: defect.**

The table declares: *"COMPUTED, with `ρ_c = 2 800 kg/m³` for every body so the only variable is `g`"*.
With `σ_y = 243 091 800 Pa`, which is the report's own Everest calibration and which I reproduce
exactly (2 800 × 9.81 × 8 850):

- Venus: 243 091 800 ÷ (2 800 × 8.87) = **9 788 m** ✓ matches the table.
- the Moon: 243 091 800 ÷ (2 800 × 1.62) = **53 592 m** ✓ matches.
- the super-earth: 243 091 800 ÷ (2 800 × 28.9) = **3 005 m** ✓ matches.
- **Mars: 243 091 800 ÷ (2 800 × 3.71) = 23 401 m.** The table says **22 594 m**.

22 594 m is what `ρ_c = 2 900` gives (243 091 800 ÷ 10 759 = 22 594). The Mars row silently uses a
different density from the rows above and below it. With the declared 2 800, Olympus Mons reaches
21 900 ÷ 23 401 = **93.6 %** of the ceiling, not 97 %. So the headline for D4 — *"the relation then
gives Mars 22 594 m against Olympus Mons's 21 900 m"* — does not reproduce from the report's own
stated method.

### R6 — `h_smooth` is priced twice, at 55 ns and at 135 ns, in one document

**Where:** §4.4a against §7.1. **Severity: defect.**

§4.4a: *"**`h_smooth` costs 55 ns** (L1's 40 + L2's 5 + the profile's 10), so a 4-point
central-difference gradient on the tangent plane costs **220 ns**."*

§7.1: *"Evaluate `h_smooth` (L1+L2+L3's profile) at each. COMPUTED cost: 4 096 × **135 ns** =
**0.55 ms**"*.

The same quantity, the same definition, two prices, a factor of 2.5 apart. If 135 ns is right, the L4a
gradient costs 540 ns and the no-river subtotal becomes 756 ns. If 55 ns is right, §7.1's boot cost is
0.23 ms. The report levelled exactly this charge at revision 1 (B-D8: *"a COMPUTED number that does
not reproduce inside its own document is an estimate"*) and then repeated it.

### R7 — Two MEASURED counts of unfenced calls do not reproduce

**Where:** §0's row, the summary's "the one law hole", §5.4 leg 2, and D9. **Severity: defect.**

The report states, four times and marked MEASURED: *"39 `powf`/`ln`/`exp`/`sqrt`/`powi` calls in
`taxonomy.rs`, 22 in `celestial.rs`"*, and D9 sums them as *"61 unfenced calls across two files"*.

MEASURED by me, 2026-09-08, on this worktree:

```
  grep -oE "\.powf\(|\.ln\(|\.exp\(|\.sqrt\(|\.powi\("   taxonomy.rs  →  43 calls on 39 lines
                                                         celestial.rs →  11 calls on  9 lines
```

39 is the LINE count of `taxonomy.rs`, so the first number reproduces if "calls" means "lines".
**No counting rule I can find gives 22 for `celestial.rs`** — the call count is 11, the line count is
9, and counting every bare token including comments gives 43. The conclusion — `vd-physics` computes
outside the fence, so an incoming fact is an unfenced float — is CORRECT and it is the report's most
important law finding. The number is not, and the report's own standard (F25, where it corrected "four
hits" to "two") applies to itself.

### R8 — Quantisation is called a cure. It is a probability reduction, and M13 cannot prove it

**Where:** §5.4's grid table, *"Then a drifting input is HARMLESS"*, and M13. **Severity: defect.**

The cure is right in direction and it is the report's best law fix. The claim around it is too strong.

Quantising `g` to whole mm/s² does not remove a drift. It turns a continuous disagreement into a
discrete one. A body whose true `g` sits within one drift width of a mm/s² boundary quantises to
`4 684` on one host and to `4 685` on the other. Then EVERY mountain on that body differs and the
golden self-check disagrees between two servers — the exact failure §5.4 describes. The report states
no residual and no bound on the host-to-host drift of `powf`, and that drift is not fixed by IEEE-754:
it is a property of each target's libm, which is why `libm` is refused by name at
`tests/tests/crate_isolation.rs:324-331`.

**Why the radius is different, and the report half-says it.** `Ladder::for_radius` snaps the radius to
a multiple of `2^(rungs−1)` cells — on the home planet a snap unit of **1 303 m**
(`crates/seed/src/ladder.rs:60-70`; `home.rs:48` moves 1 mm and 1 000 ulps and still gets the same
body). A 1 303 m quantum against a drift of about 1e−9 m is a margin of twelve orders of magnitude. A
1 mm/s² quantum against a `powf` chain may hold a similar margin, but nobody has measured that chain's
drift, so the margin is asserted, not measured.

**M13 as written cannot close it.** It moves the home planet's facts by a thousand ulps and requires
the same body. That proves the home planet is not near a boundary. It says nothing about any other
body of THE world, and the failure is per body. The honest gate is a bench over many seeds that
prints, for each body, the DISTANCE from every fact to its nearest quantisation boundary against a
stated floor — plus a measured bound on `surface_gravity_mps2`'s drift between the shipped targets,
which is the same measurement `D-TERRAIN-1`'s G4 leg still owes.

### R9 — The harvest law covers the nearest-site query only; the river lookup is ADDITIVE and needs a stronger law

**Where:** §4.0's HARVEST LAW, §10.6's SL8 bullet. **Severity: defect.**

THE HARVEST LAW reads: *"the candidate list of a hydrology cell must be a strict SUPERSET of every site
that could be the NEAREST site for any direction inside that cell"*. With a fixed-order scan and a tie
broken by site index, that is sufficient for L1, because the query is a MINIMUM: an extra candidate
loses and changes nothing.

L4b's query is not a minimum. A column "tests ≤ 32 branch segments" and each one contributes a
channel, a water surface or a fill. A sum is not a minimum. Two cells across a hydrology-cell edge with
different branch lists agree only if BOTH of these hold, and the report states neither:

1. every branch outside a cell's list contributes **exactly `+0.0`** at every direction inside that
   cell — a clamped polynomial can do this, but it must be REQUIRED;
2. the summation ORDER is fixed across lists of different length — float addition is not associative,
   and §3.3 reason 3 refuses a competing design for exactly this hazard (*"`gf.rs` fences ARITHMETIC;
   it fences no ordering"*).

Until both are written, §10.6's claim that all three SL8 directions are covered is not established for
the third, and M6(b) tests the superset property of SITES only.

### R10 — "The bend stays irrelevant to the layout" is false as soon as the hydrology cell exists, and its cost is not in the table

**Where:** §6.3's last paragraph, §4.0. **Severity: defect.**

§6.3 states the law *"evaluate on `dir`, never on `(face, a, b)`"* and then: *"The bend `W(a)` stays
irrelevant to the layout"*.

§4.0 then makes every bounded list a function of the **hydrology cell**, which is *"a cell of the
ladder at a fixed rung"*. A ladder cell IS `(face, a, b)`. Getting from a direction to that address
runs `face_of`, then `unbend` twice — and `unbend` is **four Newton steps, each with a division**
(`crates/seed/src/bend.rs:47-66`, `INVERSE_STEPS = 4`). Two consequences:

- **The layout now depends on the bend's world identity.** `bend.rs`'s own doc says it: *"the
  constants, the evaluation order (Horner form), the inverse's FIRST GUESS and its STEP COUNT are part
  of the generator's world identity"*. Every plate, every river and every biome moves if
  `INVERSE_STEPS` or `K2` ever changes. That is a real coupling, and the report says the opposite.
- **The cost is missing.** §9.1 prices L1 at 40 ns for the site tests and prices the address
  derivation at nothing. Two `unbend` calls are eight divisions plus about forty multiplies and adds,
  and a probe of the harvested cell sits on top. On a budget that already holds a red row at 8.68 ms,
  an unpriced per-column step is not a footnote.

### R11 — Airy isostasy omits the water load, and the sea-level solve is not self-consistent with it

**Where:** §4.2's table and *"Earth's real figure is a 4.5 km mean step"*; §7.1. **Severity: defect.**

`e = t(1 − ρ_c/ρ_m)` is the elevation of a crust block that floats with AIR above it. The design puts
an OCEAN over the oceanic block, and water loads it down. Write the pressure balance at a common
compensation depth with the report's own densities (`ρ_m = 3 300`, `ρ_c = 2 800`, `ρ_o = 2 900`,
`ρ_w = 1 030`, `t_c = 35 000`, `t_o = 7 000`) and the step for an ocean of depth `d` is:

```
   step (continent surface to ocean floor)  =  4 454.5 m  +  0.312 · d
```

With a 4 km ocean the step is **5 703 m**, 28 % more than the 4 455 m the report reports and compares
with Earth. The omitted term is the same order as the effect the report claims to reproduce, so the
agreement with Earth is not evidence for the formula as written.

The consequence is not cosmetic. §7.1 bisects for the sea level on a hypsometry built from `h_smooth`,
and `h_smooth` carries no water load — but the water load depends on where the sea lands. The two are
coupled: the correct solve is implicit, and the report presents it as one pass of a fixed 24-step
bisection.

### R12 — The sea level is solved on a field that omits most of the height variance, and the 4 096 directions have no fenced construction

**Where:** §7.1 steps 2 and 3. **Severity: defect.**

Two separate problems in one recipe.

1. **The field.** Step 2 evaluates `h_smooth` — L1 + L2 + L3's profile — at 4 096 directions and calls
   the result *"the hypsometric curve"*. It is not. The octave sum alone contributes a standard
   deviation of **2 284 m** (COMPUTED over 4 000 directions, and it agrees with the report's own
   2 297 m in §1.4), and L3's ridged folds add more. A level solved on the smooth field alone puts an
   unknown area of land under water and an unknown area of sea floor above it. The promise — *"a
   water-rich earth-like planet gets 60–75 % ocean"* — is not established by a solve that omits the
   term carrying most of the variance. Either the octaves enter the 4 096 evaluations, which
   multiplies the boot cost by the octave count, or the report must state the error.
2. **The directions.** *"Draw 4 096 directions from the body's own seed, in a FIXED order, with the
   same `SplitMix64`"* — but a UNIFORM direction on a sphere needs either a Gaussian (`ln` and `cos`,
   both refused) or rejection sampling in a cube followed by a normalise. The rejection loop is fenced
   and deterministic, but it has a data-dependent trip count, it is a new HR5 branch, and the report
   never writes it. This is the class of omission the report itself found in revision 1 (F8: the
   hotspot rotation, the Voronoi area, the wind's turn), and §10.1's fence table lists the hypsometry
   as *"4 096 fixed-order evaluations, compare | **yes**"* without the sampler.

---

## Weaknesses

### R13 — L4a's 220 ns gradient cannot resolve a valley, because `h_smooth` holds no valley-scale content

**Where:** §4.4a. **Severity: weakness. It carries 53 % of the added budget.**

`h_smooth = L2 + L3's PROFILE only`. L2's affinity field has a wavelength near 3 000 km (§4.2) and
L3's profile is bounded by a transition width `W < 500 km` (§4.1). So `∇h_smooth` varies over hundreds
of kilometres and is effectively CONSTANT across a rung-0 chunk, whose 62 × 62 columns span 62 m.

Three consequences the report does not state:

- it cannot tell a valley floor from a divide at the 1 km scale the reference vista shows — the only
  field with content at that scale is the octave sum, which `h_smooth` deliberately excludes;
- what it CAN do is modulate roughness by the macro slope, which is what §4.6's `r` already does from
  L1–L3 for **free**;
- because it is constant across a chunk, the four extra evaluations are a per-CHUNK cost, not a
  per-column cost. Charging 220 ns per column over-prices it by nearly the chunk's column count, so
  the budget verdict rests on an error in the safe direction.

The report should either state which feature size the gradient resolves, or price it per chunk and
find the valley shape somewhere else.

### R14 — Every per-layer nanosecond figure is ESTIMATED, and the verdict already holds a red row

**Where:** §9.1's "Why that number" column, §9.2, D1, D11. **Severity: weakness.**

Only two numbers in §9 are MEASURED: 185 ns per column and 13 ns per octave. Everything added is an
estimate, and one is optimistic on its face. L1's *"~5 ns per dot-and-compare"* must cover a bisector
normal `n̂ = (s₂ − s₁)/|s₂ − s₁|` — a subtract, a dot, a `sqrt` and three divides — for each candidate
pair, not a dot product. If L1 is three times its estimate, the gradient (four `h_smooth` evaluations)
triples with it, the added cost rises by about 320 ns per column, and the cold surface chunk moves
from 4.98 ms to **6.2 ms** while the cave-dense chunk moves to **10.0 ms**. M3 is written correctly
("it must run BEFORE the design is accepted"), but D1 and D11 already carry recommendations, so the
owner is asked to decide ahead of the measurement that decides.

### R15 — The ridges and the drainage are independent fields, so the crests will not be the divides

**Where:** §4.3's ridge lines against §4.4. **Severity: weakness (believability).**

L3 draws ridge lines as a domain-warped `1 − |noise|` fold. L4b grows a river tree from the coast.
Nothing couples them. In real land a ridge crest IS a drainage divide, and that is what makes a range
read as a range rather than as corrugated noise. Under this stack a river may run along a crest and a
divide may sit in a valley floor. §11's oracle measures the slope–area exponent and the drainage
density; it does not measure crest-divide agreement, which is the property this gap breaks.

### R16 — `σ_y` is calibrated against a different material property from the one the formula names

**Where:** §4.3. **Severity: weakness.**

`h_max = σ_y/(ρ_c·g)` names `σ_y` the crust's yield strength. The check is *"granite's measured
unconfined compressive strength is 100–250 MPa"*. Those are two properties. An unconfined compressive
strength is measured on a free sample at the surface; the rock under a mountain root sits at hundreds
of MPa of confining pressure, where the limit is ductile flow and gravitational spreading, not
crushing. The `1/g` trend is a real physical consequence and the relation is the standard textbook
estimate. The claim that the calibration "produces a real rock property, in range" reads as a
validation and is a coincidence of magnitude.

### R17 — A landform the reference vista shows is missing from both the stack and the limit table

**Where:** §4.8. **Severity: weakness.**

§4.8 states the stack's limit as four overhang-class features (pillar, arch, undercut, sea stack), and
that is a real service. It misses a class the height field CAN make and this stack does not: the
**layered mesa, butte and cuesta** — the caprock signature a desert vista is mostly made of. It comes
from DIFFERENTIAL EROSION on horizontal strata: a hard bed resists, the soft bed under it retreats, and
a flat top with a cliff rim survives. The crate already holds horizontal strata (`StrataTable::at`,
`strata.rs:189`), and the report's transfer curve is driven by SLOPE only, never by the substance at
that height, so no mesa forms except by luck. §4.8 should say so, beside the pillars.

### R18 — The half-cell cut-off rule's SL8 safety is asserted, and the rule it depends on is not built

**Where:** §6.2, §10.6's SL8 bullet. **Severity: weakness.**

The half-cell rule is a good rule and the right shape. Its SL8 claim — *"under half a cell by
construction, which is under the extractor's own resolution, so it cannot be seen"* — mixes two
resolutions. Half a cell at rung 8 is **128 m**. Whether 128 m is invisible depends entirely on the
tier rule that decides WHICH rung is drawn at which distance, and that rule is slice 8's. Today the
world runs one rung per realm behind a dev flag, which is `D-TERRAIN-3`, marked 🟥 at
`docs/design/DEFERRED.md:7775`. The claim therefore rests on an unbuilt rule. It should read: *the
cut-off adds under half a cell, and slice 8's tier rule must keep a cell near a pixel for that to be
invisible; the measurement is owed.*

### R19 — A 4 KB `Copy` body definition is a real change and no decision states it

**Where:** §4.0's sizing. **Severity: weakness.**

`plates: [Plate; 64]` at an ESTIMATED 64 bytes a row is **4 KB**, against today's whole
`BodyDefinition` of roughly 500 bytes (`octaves: [Octave; 16]` at 24 bytes a row plus the small
tables, `body.rs:89-105`). The struct is `Copy` and `PartialEq` (`body.rs:88`), `from_seed` returns it
BY VALUE inside an `Option`, `home_planet()` returns it by value, and the client holds one per realm
(`crates/client/src/chunks.rs:275-279`). M10 measures the size, which is right, but no D-row asks
whether the plate list should sit behind a reference or be re-drawn on demand, and §8.1's memory row
does not carry it.

### R20 — "No magic numbers" is claimed passed while the functions that decide the picture are unwritten

**Where:** §10.5's second table, §2's property 5. **Severity: weakness.**

The second table derives ten new numbers, and that answers F20 properly. It does not cover anything
shaped rather than scalar: the transfer curve's polynomial, `f(c)` and `g(a)` in §4.3's uplift,
`profile(x)`'s rise-crest-backslope-foreland shape, the domain-warp amplitude, the ridged octave count
(given as 3 with no derivation), the influence-cap radius of §4.0's harvest, the hydrology cell's "near
2 km" target, the "≤ 32 branch segments", and the finite-difference step `δ` of §4.4a — a length the
report never names anywhere. These are the parts that decide what a player sees. Stating the gate as
passed is premature; "passed for every scalar, owed for every shape" would be honest.

### R21 — The snow-line table's inputs are asserted inside a table marked COMPUTED

**Where:** §4.5's Earth check. **Severity: weakness.**

The arithmetic reproduces exactly: (299 − 273.15)/6.5 = 3 977 m, (303 − 273.15)/6.5 = 4 592 m, and so
on for every row. The `T_warm` column — 299 K at the equator, 303 K at 23°, 293 K at 45°, 288 K at 60°,
283 K at 70° — is an input with no citation. The whole table is labelled COMPUTED, so a reader takes
the agreement with Earth's real snow line as a result of the model. It is a result of five chosen
inputs. One line of source (a warmest-month climatology) fixes it.

### R22 — "The cell is still 12 bytes" is a ruling's format, not a code fact

**Where:** §4.4c, §10.6's record bullet, §Summary row 5. **Severity: weakness.**

The generated cell in the crate is `Cell { stratum: Stratum, gap: i8 }` — **two bytes**
(`crates/terrain/src/chunk.rs:64-69`). The twelve-byte record is the STORED record of ruling V6 part B
(`owner_decisions_2026-09-07_voxels.md:186-187`) and slice 9 has not built it. The statement *"the
12-byte record is genuinely unchanged"* is true about the ruling and reads as a check against the code.
The report should name which of the two it checks, because the water-surface change (W3b) does move
stored cells, and that is the sentence's whole point.

### R23 — The producer of `BodyFacts` is never named, and every candidate touches a law

**Where:** §5.4's SL6 ask. **Severity: weakness.**

The ask states the data, the direction, the reason and the cost — four of SL6's parts, properly. It
never says WHICH code computes the record. §5.2 says the draw belongs "in the forest, not in the
generator", and §5.4 says the record travels "from the realm ITSELF, about ITSELF". Between those two
sentences sits the question: the planet's own shard must link `vd-physics` and derive its own mass, its
own orbit and its own insolation. Insolation is a function of the body's own semi-major axis, and SL1
clause 3 says *"A child NEVER derives, adjusts, computes or states its own placement"*. Deriving a
scalar from your own orbital elements is not stating a placement, so this is probably lawful — but SL1
clause 5 says such a fence is settled structurally and never by care, and the report leaves it
implicit.

---

## Notes

### R24 — Small corrections

- **Cite drift.** The noise pin's assertion is `crates/terrain/src/noise.rs:231` and `NOISE_PIN` is at
  `:235`; §1.1's code block says `:228`. `Stratum::Water`'s VARIANT is `strata.rs:18`; `:44` is its row
  in `Stratum::ALL`. `D-TERRAIN-3` is "ONE RUNG PER **REALM**", not "one rung per session"
  (`DEFERRED.md:7775`).
- **The compile-time assertion is integer arithmetic.** §1.6 and §4.6 write
  `50 000 / 65 536 = 0.76 < 2 ✓`. `LONG_WAVE_CAP_M` and `SHORT_WAVE_M` are `u64` (`body.rs:24-26`), so
  the expression evaluates to `0 < 2`. The assertion still holds and the conclusion is unchanged; the
  arithmetic shown is not the code's.
- **"A per-body draw near 50 km" is a constant for nearly every body.** The long wave is
  `radius × [0.25, 0.5)` CLAMPED to `[20 km, cap]` (`body.rs:151-153`). With a 50 km cap every body
  above a 200 km radius sits exactly on the cap, so the coarsest wavelength stops being a per-body
  number. Today's 400 km cap does the same above a 1 600 km radius, so this is not new — but the report
  calls it a draw.
- **"The clamp arm still never fires in production."** True for the home planet (15 − 11 = 4). The arm
  at `body.rs:256` still fires for a body with 16 rungs, which the ladder accepts up to a 42 723 km
  radius. D13 sends those bodies to a cloud shell, so the claim survives through D13, not through the
  octave count.
- **D7's precedent.** `L_SUN_W` (`taxonomy.rs:588`) and `RHO_ROCK_KGM3` (`:831`) are real and the
  citation is right, but they live in the UNFENCED crate. A physical constant inside `vd-terrain` is a
  world-identity datum in a way those two are not.
- **§9.3's ceiling row.** The label `(2 × 18 536 + 2 000)` is unexplained. 18 536 m is `h_max` at
  `g = 4.684`, and the band arithmetic then doubles the whole expression again
  (2 × 39 072 + 495 = 78 639), which is how the row's number is reached. It reproduces; it needs one
  sentence to be readable.

---

## Checked, and sound

Everything below I tried to break and could not. Each is a measurement I ran.

1. **The Python replication is real.** My independent re-implementation returns the crate's own noise
   pin bit for bit: `13 827 097 110 060 728 320` (`noise.rs:235`). §1.1's validation stands, and it is
   the right pin to have chosen.
2. **The home planet's drawn numbers.** COMPUTED and equal to the report's: ladder radius
   3 350 759.045 m, `N = 5 263 360`, 12 rungs, relief **14 304.887 m**, `relief_whole` **14 305**,
   `long_wave = 400 000 m` (at the cap), `k_rough = 0.468 338 118`, **14** octaves, finest wavelength
   **48.83 m**, sea offset **−5 297 m**. §1.1's octave table (7 605.55 m at 400 km down to 0.397 m at
   48.83 m; ratios 0.0190 → 0.0081) reproduces row for row.
3. **§1.2's diagnosis.** My own sampling gives RMS slope 1.96° at 2 m, 2.01° at 32 m, 1.75° at 512 m
   and 1.59° at 8 192 m, against the report's 1.95 / 2.00 / 1.80 / 1.62. The 50 m window median is
   1.11 m against the report's 1.18 m. One slope at every scale, one metre inside fifty: correct.
4. **§1.4's hypsometry.** COMPUTED over 4 000 directions: mean +49 m, σ **2 284 m**, σ ÷ relief
   **0.160**, p05 −3 728 m, p50 +49 m, p95 +3 761 m, and **0.9 %** of the surface under the drawn sea,
   which matches the crate's MEASURED "about one column in a hundred"
   (`slice_05_generator.md:388-392`).
5. **§1.6 is the report's best finding.** The octave loop at `body.rs:169` halts at 48.83 m and the
   cell is 1 m. The height field holds nothing between them. That is the most direct answer in the
   report to *"no details at all"*.
6. **§9.3's band arithmetic.** `band_m = 2 × relief_whole + strata + caves + 128` follows exactly from
   `body.rs:233-236` and `ladder.rs:88-98`, and 2 × 14 305 + 367 + 128 = **29 105**, the crate's
   MEASURED `cells_in_band(0)`. W1b is real: `floor_m` moves, so every cell address moves.
7. **The ladder's refusal.** `N_MAX = 2²⁶` (`ladder.rs:25`) caps the radius at 2 × 2²⁶/π =
   **42 723 km**, so Jupiter and Saturn are already refused and the gap is Neptune and Uranus. D13 asks
   the right question.
8. **The super-earth chain.** From `R ∝ M^0.27`: a 12 000 km radius is **10.43 M⊕**, mean density
   **8 606 kg/m³**, `g` **28.87 m/s²**, ceiling **3 005 m**. Every figure reproduces.
9. **Held & Hou.** `sin φ_H = sqrt(5·9.81·15 000·(1/3) / (3·(7.292e−5)²·(6.371e6)²)) = 0.6155`, so
   φ_H = 38.0°; the doubled and quadrupled spins give 0.308 and 0.154. Every row reproduces, and the
   report states its own 27 % error rather than hiding it. Testing `|dir · pole_axis| < sin φ_H` is a
   genuinely good fence move.
10. **The lapse rate and the snow line arithmetic.** 9.81/1005 = 9.76 K/km, × 0.666 = 6.50 K/km; the
    home planet 2.44 and 3.43 K/km at the two densities; every snow-line row divides correctly.
11. **The visibility arithmetic.** √(2 R h) with R = 3 350 759 m: 3 473 m for a 1.8 m eye, 25 887 m for
    a 100 m butte, 81 863 m for a 1 000 m crest; the curvature drop is 373 m at 50 km. Adding the
    observer's own horizon is the right fix.
12. **The bisection and the sampling error.** 40 000 ÷ 2²⁴ = **2.4 mm**; √(0.25/4 096) = **0.78 %**.
13. **The code facts of §0.** I read all of these and all are true: `height_m` is `radius + Σ octaves`
    and takes a unit direction (`height.rs:17-28`); `fluid_at` is one rule per body
    (`chunk.rs:204-212`); the bedrock is one draw per body (`body.rs:200`) and `StrataTable::at` takes
    no direction (`strata.rs:189`); `POLE_AXIS` is `+Z` with obliquity named as a later slice
    (`height.rs:30-35`); `Gf` grants `lesser`, `greater`, `clamp` and `is_finite` and refuses
    `min`/`max`, `mul_add`, `powi` and every transcendental (`gf.rs:10-14, 102-148`); the density byte
    is the radial gap clamped to one cell (`chunk.rs:630`); `TAG_SURFACE` and the 1 200-byte self-look
    budget exist (`look.rs:56, 69-75, 79`); the wire's rule is "new data rides a new trailing variant"
    (`version.rs:13-14`); the client links `vd-terrain` and `vd-seed` and carries `vd-physics` under
    `[dev-dependencies]` only (`crates/client/Cargo.toml:24-26`), with the edge refused at
    `tests/tests/crate_isolation.rs:321-331`; today's refusal path compares only the generator tag
    (`crates/client/src/chunks.rs:269-271`).
14. **The spin grep.** Two hits, exactly as the report says: `height.rs:32` (a comment) and
    `crates/physics/tests/celestial_ephemeris_pin.rs:29` (a test constant). The world has no spin.
15. **`D-TERRAIN-1`'s x86-64 leg.** The G4 row reads EMULATED, and *"a REAL x86-64 machine before SL10
    clause 3 is called satisfied"* (`DEFERRED.md:7731`). §3.3 reason 3 quotes it correctly.
16. **The refusals that are right.** The bake is refused on four structural reasons rather than on a
    scary corner, and the honest 512-edge cost (6.3 MB, 0.19 s on 14 threads) is stated — 1.6 s ÷ 8.4
    reproduces from the report's own table. The passive-margin cure (a crust FIELD, never a per-plate
    label) is correct geology and the best believability move in the document. §4.8's statement of the
    stack's own limit is the honesty the earlier revision lacked.
