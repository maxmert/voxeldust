# Slice 8c — THE SOLVE: the implementation design

**Status:** the design for the owner's discussion, 2026-09-19. Nothing here is built. Ruling T1 item 5
(`owner_decisions_2026-09-16_terrain.md`) names 8c: *"the macro layout (plates, isostasy, orogeny), the
drainage, the priority flood, discharge and stream power, isostasy, talus whose angle reads the aridity,
ice, craters, the coast; the climate inside the schedule (insolation, the lapse rate, the Hadley edge,
orographic lift, the rain shadow). Once per body; the artifact kept."* Ruling T8 makes 8c the cure of
the drowning: *"8a's pictures are judged WITH THE SEA OFF — the dry shape — until 8c gives the ground its
second hump; the coast is judged after 8c."* Ruling T3 forbids erosion IN THE GAME: the solve computes
an eroded shape ONCE and the land never changes after it.

**Where it comes from.** The investigation of 2026-09-08 designed this in three documents and their
refutations: `02_planet_layout.md` (the initial land: plates, isostasy on a crust field, orogeny under
the relief law), `03_erosion_rivers.md` (the macro solve on its own lattice, the artifact, how the fine
rungs read it) and `05_climate_biomes_weather.md` (the climate as a closed form over the same lattice),
integrated by `06_laws_integration.md` (§4.1 ship or derive, §4.3 the lattice's resolution, §6.1 the
climate INSIDE the solve). This design binds those to the code as it stands after 8a, 8b and 8s and to
the rulings since. Where the three documents say more than fits here, this design points at the
section and does not restate it.

**The one-line summary.** ONCE per body, on a coarse lattice of about eight kilometres, a solve turns
the seed's plates into continents and ocean floors, routes the rain the climate gives it into rivers,
cuts valleys with the stream power law, rebounds the crust, relaxes the slopes, marks the ice and the
coast, and keeps a small table — the ARTIFACT. Every chunk then reads that table as its coarse layer,
under the spectrum 8a built and the words 8b stated. Nothing runs on the tick, nothing changes with
time, and the sea finally has a basin to sit in.

**Every explanation carries a game-word example, as the owner asked.**

---

## 0. THE LAWS THIS DESIGN OBEYS, AND HOW

| law | how 8c obeys it |
|---|---|
| SL10 one generator, no drift | the solve is `f(seed, charter)`; its every pass works on INTEGERS between passes and names its one float (`sqrt`, granted by name); the artifact's bytes are digest-pinned on every target (§7) |
| the seed ruling (shape public, value not) | the artifact decides where a valley, a beach or a gravel bar lies — SHAPE; it decides no deposit's value (03 rec 11: *"a seed-derived placer bar is a treasure map"*) |
| T3 no erosion in the game | the solve runs once at the body's creation; the land's shape after it is static; the only later change is the authored override (8d) |
| T8 the sea after the second hump | the sea radius is re-solved over the solve's hypsometry (§6); the recipe's sea DRAW dies with 8c |
| T9 no physical number drawn | every rate is a published law with a calibration body (§4); the pass counts are COST knobs and named so; the erosional age is the system's age (§4.9) |
| SL6 ask before new data crosses | ONE row proposed (§8): the artifact itself, if the owner chooses "ship" — the measurement of §9 C1 decides whether it is needed |
| HR5 100 % | the solve is Tier-A code in `vd-terrain` with its own tests; the generic parts are branchless shims |
| SL5 one world | no reduced lattice, no test-only solve: the driver test runs the SAME solve on the smallest body the world holds (a 50 km moon: 384 nodes, 06 §3.3) |
| F1 the frozen patch | the artifact's version is part of the world identity; a built area keeps its artifact version like its octave count |
| SL8 a seam is a defect | the macro field is interpolated on the TRUE lattice with the seam and corner rules of 03 §5.4; the rung-to-rung disagreement under the new coarsening stays under a pixel (03 §7.1) |

---

## 1. THE LAND TODAY (measured from the code, 2026-09-19)

- **The height** (`crates/recipe/src/height.rs`, `crates/terrain/src/body.rs`): the ladder radius plus
  the relief-shaped octave sum — 8a's slope spectrum on the fine octaves, the ridged middle band, the
  cap-rock bench, the roughness factor on a PLACEHOLDER slow octave (`salt::ROUGHNESS`, ledgered) — and
  the sea as a DRAW (`sea_radius`, kept until 8c by ruling T8). The COARSE octaves (the longest ~400 km)
  carry the whole relief budget the relief law allows (8 276 m on the home planet). There is no
  continent, no ocean basin, no range: the same statistics everywhere. MEASURED in 8b: the water
  inventory drowns the whole surface (ocean share 1.000), because the hypsometry has one hump.
- **The charter** (`crates/core/src/look.rs`, 8b): twenty words stated once by the realm; 8c reads the
  gravity, the bulk density, the escape velocity, the insolation, the equilibrium and surface
  temperatures, the greenhouse depth, the day, the obliquity's cosine, the water inventory, the elastic
  thickness `T_e` (stated *"8c's input: it sets the flexural wavelength"*, 8b §2.6), and the flags.
- **The ladder** (`crates/seed/src/ladder.rs`): `N = 5 263 360` rung-0 cells per face edge on the home
  planet; the crossfade and the wanted set of slice 8; one aliased top octave at the rungs past the
  table (T6 ask 3, *"8c named as the fix"*).
- **The store**: slice 9 lands the block store; the free window for rung-0 bytes closes when the first
  world is saved (06 §6.1). 8c moves every byte, so it lands before slice 9, as T1 orders.
- **The census** (`vd-physics`): the system's age (`SYSTEM_AGE_GYR`), the star's class and
  luminosity, the body's mass and radius — the facts the solve's laws are calibrated against.

---

## 2. THE MACRO LATTICE (06 §4.3, adopted)

A lattice is a grid of points laid over the whole globe. The solve's lattice is its OWN: a uniform
cube-sphere grid whose node count per face edge is a DIVISOR of `N`, so a node covers an exact integer
number of rung-0 cells and a chunk finds its node by one integer division — no float, no seam.

```
MACRO_CELL_TARGET_M  a stated METRIC resolution of the world (recommended 8 192 m; ask 1)
n_macro  = the divisor of N whose node N / n_macro is nearest MACRO_CELL_TARGET_M
           (ties to the smaller; at least 8; a stated MEMORY CEILING above)
nodes    = 6 · n_macro²
```

Home planet: `n_macro = 640`, a node of 8 224 m, 2 457 600 nodes. A 200 km moon: 32 per edge, 6 144
nodes. A 50 km body: the clamp at 8, 384 nodes. Why it is lawful: the target is a metric statement about
THE world — the drainage skeleton is resolved at about eight kilometres everywhere — like `SHORT_WAVE_M`;
the ceiling is a cost knob and named as one.

Example: a pilot looks 50 km down a valley on the home planet. Six macro nodes span that view. The solve
decided which way the valley runs and where its river is; the spectrum and the carve draw the spurs and
the gullies inside each node. Nothing the eye resolves in that frame is a node; the nodes are the
skeleton.

---

## 3. THE INITIAL LAND: THE MACRO LAYOUT (02 §4.1–4.3, adopted)

The land before water touched it, as a closed form per node from the seed and the charter:

- **L1 plates.** A spherical Voronoi partition: the seed places plate seeds on the sphere; every node
  belongs to the nearest one; the distance to the nearest boundary `b` is written out and fenced
  (02 §4.1). The count of plates follows the body's size and its age (a law, calibration Earth: about
  a dozen large plates at one Earth radius).
- **L2 isostasy on a CRUST FIELD, with the water load.** The crust's thickness and density are a
  smooth field, not a per-plate label, so a continent and an ocean floor can share a plate and a
  passive margin exists. Airy isostasy — thick light crust floats high — with the water column's weight
  on the low side. COMPUTED in 02 §4.2 for the home planet: the ocean floor 6 476 m under the
  continental surface with the water load, 4 455 m without. THIS is the second hump ruling T8 waits for.
- **L3 orogeny under the relief law.** Ranges where plates converge, trenches, rifts, arcs, hotspot
  chains — an uplift field from the boundary distance and the boundary kind, capped by 8b's relief law
  `σ_y / (ρ_c g)` (already landed as the ceiling).

What 8c ADDS to 02's layers: nothing; it uses them as the solve's starting surface. What it DROPS: 02's
L4a multifractal weight and L5 climate operators are superseded by the solve's own products (the
roughness field, the snow line) — 02 §4.4 handed the rivers to domain 03 and this design follows that.

---

## 4. THE SOLVE (03 §4, adopted; every law with its calibration body)

The solve is a short list of passes over the macro lattice, each a known landscape-evolution operator,
run a FIXED number of times. A pass is a sweep over every node in a stated order. Integers between
passes; `sqrt` the one float, granted by name.

### 4.1 Flow routing — D8, with an exact tie-break
Each node's water goes to the steepest of its eight neighbours (D8). Steepness compares
cross-multiplied integers; a tie breaks on the node index. No float.

### 4.2 The priority flood — pits, lakes and flats
A pit is a node lower than all its neighbours; real water fills it. The priority flood raises every pit
to its spill level in one pass over a heap ordered by `(height, node index)`, and keeps TWO surfaces:
the terrain and the flooded surface. Where they differ there is a lake, with its spill level as the base
level for everything upstream (03 §4.4).

### 4.3 Discharge — the rain, not the area
Discharge is how much water passes a node. It is the RAIN accumulated downstream through the D8 tree —
the rain the CLIMATE gives each node (§5), not the drainage area. 06 §6.1: *"a desert basin and a
rainforest basin of the same area carry a different water"*; with area as the proxy both are cut to the
same depth and the rain-shadow gate would measure paint on ground carved as if both sides were wet.

### 4.4 The stream power law
```
dz/dt = U − K · Q^½ · S        (m = ½, n = 1)
```
`U` the uplift (from L3), `K` the erodibility (calibrated on Earth's measured denudation rates, the
calibration stated in the code), `Q` the discharge, `S` the slope. `m = ½` makes the one transcendental
a `sqrt`; `n = 1` makes the implicit update one division per node in a single downstream-to-upstream
sweep, no inner iteration, no stability test (03 §4.6). A lake node is skipped; its spill level is the
base level of its inlet.

Example: on the home planet's windward range the rain is heavy, the discharge high, and the trunk
valley cuts deep with a wide floor. Behind the range, in the rain shadow, the same-sized basin carries
little water and stays a high, gentle upland. The player sees the two sides of one ridge differ in
SHAPE, not only in colour.

### 4.5 The sediment budget
One number per basin: what the cuts removed, deposited at the basin's outlet and the coast (03 §4.11).
It decides where a flood plain and a delta are — SHAPE, never a deposit's value.

### 4.6 Flexural isostasy on a pyramid
The crust bends under load like a plate: the response is spread over the flexural wavelength, which
the charter's `T_e` sets (8b §2.6). The rebound is applied on a coarse pyramid of the height field, a
few levels, not a box blur (03 §4.7).

### 4.7 Talus — the angle reads the aridity
Nothing stands steeper than its angle of repose; the excess mass slides to the foot and is conserved.
The angle is a law of the material and the climate: loose dry rock stands at 35°, and wet or vegetated
slopes at a lower angle read from the node's aridity (§5). 8a's `TAN_REPOSE` becomes this per-node
value's ceiling.

### 4.8 Ice
Above the equilibrium line altitude (ELA, from the climate) snow outlives the summer and there is no
liquid water: stream power does not act, and glaciers over-deepen the trunk valleys and leave cirques.
The solve marks the ice (a mask and a depth proxy) and over-deepens the trunks; the fine pass draws
the cirques and the U-shape (03 §4.9).

### 4.9 Craters
Craters are the record of impacts over the surface's age, screened by the air: a thick atmosphere burns
the small impactors, so a body's smallest crater grows with its surface pressure. Two published laws
with calibration bodies: the lunar crater production function (Neukum) for the count per area and
size against the AGE — the system's age from the census, never a drawn number — and the atmospheric
screening threshold against the charter's pressure (the Moon keeps everything; Earth keeps only the
large and recent; Venus keeps none under a few kilometres). Erosion then removes the old craters where
the stream power acted, so a cratered highland and a smooth wet lowland come out of the same passes.
**Ask 4:** include craters in 8c's solve as ruled, or place the small ones in 8f with the third
dimension (a crater's rim and floor are shape; its bowl below the surface is 8f's kind of shape).

### 4.10 The coast
Where the terrain meets the sea's radius (§6) the solve marks the coastal band: the shelf, the beach,
the cliff class; the fine pass (8d) draws them.

### 4.11 The schedule, and the erosional age
```
for pass in 0 .. PASSES:
    if pass % CLIMATE_EVERY  == 0:  the climate over the relief as it stands (§5)
    if pass % FLOOD_EVERY    == 0:  the priority flood, the flats, the D8 receivers
    the discharge; one stream-power sweep; the sediment
    if pass % ISOSTASY_EVERY == 0:  the flexural rebound
TALUS_PASSES relaxations; the ice pass; the coastal band; THE ENVELOPE ASSERTION
```
The counts are numerical resolution — how finely the age is stepped — and are COST KNOBS, part of the
world tag so a change is a new world (03 §5.5). The AGE the passes step through is the body's own: the
system's age from the census (4.54 Gyr on the home planet's system). T3 holds: the age is stepped once,
at creation; the game's clock never steps it again. The per-basin hypsometric integral is the gate that
says the eroded shape is a shape of that age (03 §12 D9).

### 4.12 The envelope
`|Z| ≤ relief_m − A_fine` as an integer assertion at the end: the solve's field plus the fine spectrum
never leaves the relief law's band, so the containment band of slice 8 does not move (03 §4.13). The
downward budget under the surface — channel 32 m, glacial floor 16 m, detail 16 m — is stated in full.

---

## 5. THE CLIMATE INSIDE THE SCHEDULE (05 §5, adopted; 06 §6.1 the loop)

The climate is a closed form over the same lattice — no heap, no order — recomputed every
`CLIMATE_EVERY` passes on the relief AS IT THEN STANDS, because relief and rain are a loop: mountains
lift the air and make rain, rain cuts the mountains. Its rows, each from charter words by a law:

- **Temperature.** The insolation by latitude from the charter's insolation and the obliquity's cosine
  (the seasonal mean); the greenhouse depth sets the surface mean (8b's thermostat); the LAPSE RATE —
  air cools with height, about 6.5 K per kilometre on Earth, a law of the gas and the gravity — sets the
  temperature on the relief as it stands.
- **The circulation.** The spin decides how many belts of rising and sinking air a planet has: the
  Hadley edge (the border of the tropics) from the day length and the radius (the Held–Hou scaling,
  calibrated on Earth's 30°). A slow-spinning body has one cell pole to pole; a fast one has several.
- **Precipitation at three scales.** The zonal belt (wet at the rising belts, dry under the sinking
  ones); OROGRAPHIC LIFT — air pushed up a windward slope cools and rains; the RAIN SHADOW behind it.
  The wind's direction comes from the circulation belt and the spin's sense.
- **The snow line and the ELA.** Where the surface temperature stays under freezing over the year:
  the ice's edge for §4.8, the sea ice for the coast.
- **The aridity.** Precipitation against the potential evaporation (the temperature): the talus's
  angle (§4.7), and 8e's biome later.

The climate rows are KEPT in the artifact (§7), because 8e's biome classifier reads them and the
weather's state model (18) starts from them.

Example: the reference picture's crimson desert is ONE zone of ONE planet: a basin in the rain shadow of
a coastal range, dry, warm, with cap-rock benches and little soil. The same planet holds a wet windward
forest on the other side of that range. The climate rows put both where the physics says; 8e paints
them.

---

## 6. THE SEA, RE-SOLVED (ruling T8's cure)

8b solved the sea level over the ladder's one-humped hypsometry and found the water drowns everything.
The solve's hypsometry has two humps (§3, L2). The SAME 8b machinery (`crates/bins/src/sea.rs`, the
bisection over the water inventory) runs over the artifact's height field and gives the sea radius
that holds the inventory: MEASURED then as the ocean share on the home planet, with the expected answer
in the tens of percent, never 1.000. The result is stored in the charter's `sea_offset_mm` (8b's word),
the recipe's `sea_radius` DRAW is deleted, and the bathymetry under the sea is KEPT in the artifact
(ruling T2: the sea floor exists). The coast (§4.10) is judged from this radius.

---

## 7. THE ARTIFACT: WHAT IS KEPT, HOW IT IS READ

**Per node** (03 §4.14, plus the climate rows):

| row | type | what |
|---|---|---|
| `Z` | i16, metres over the ladder radius | the eroded macro height (bathymetry included) |
| water level | i16, the same encoding, a sentinel for dry | lakes and the sea's local level |
| receiver + facies | u8 | the D8 direction (3 bits) and the surface class (rock, talus, alluvium, ice, shelf, beach…) |
| discharge | u8, quantised log | for the carve's channel width (8d) |
| climate | u8 temperature class, u8 precipitation class, u8 aridity/snow flags | for 8e and 18 |

About 9 bytes a node: 22 MB on the home planet, plus a pyramid of `Z` (coarse levels, 4.8 KB at the
top). A 200 km moon: 55 KB.

**How a chunk reads it.** `height_m` gains the ADDRESS (it holds one already at every production
caller, 03 §7.3) and reads: the ladder radius + `Z` interpolated on the TRUE lattice (the seam and the
corner rules, 03 §5.4) + the fine octaves the rung keeps on 8a's spectrum × the ROUGHNESS FIELD read
off the solve (the facies and `|∇Z|`: a craton is a plain, an orogen is rough) − the carve (8d) + the
detail. **The coarse octaves are REPLACED by `Z`**: the octave table is re-anchored so its longest
wavelength is about the macro node (~50 km), and 8a's placeholder roughness octave DIES (its owed law
lands here). The pyramid answers the top rungs (T6 ask 3: the aliased octave's replacement).

**Where it lives.** The owning realm's store (the shard's redb, slice 9's row kind), keyed by the body
and the artifact version; the home planet's artifact is pinned in the build like `home.rs`, so the
world hello works before the first bulk arrives (03 rec 10).

**Derive or ship** (06 §4.1). The client MAY derive it (SL10), but the solve was ESTIMATED at 12–40 s of
one core for the home planet at 8 224 m — too long for a login. 06 recommends the realm SHIPS it on the
bulk lane, coarse pyramid first (the star field's precedent). **This is decided by a measurement, not
by the estimate**: §9 C1 runs the solve and reads its wall time; under a few seconds the client derives
(no new wire), over it the realm ships (one SL6 row, ask 2).

---

## 8. WHAT CROSSES (SL6)

Nothing new if the client derives. If the owner chooses to ship: ONE new bulk payload kind — the artifact
(digest, version, the pyramid first, then the rows), a one-hop statement from the owning realm, refused
on a digest mismatch. The charter's `sea_offset_mm` already crosses.

---

## 9. THE ORDER OF WORK

| stage | what lands | proof |
|---|---|---|
| C1 THE BENCH | the solve's core (lattice, D8, flood, discharge, stream power) on the home planet at 8 224 m, single-threaded, off the tick: WALL TIME and PEAK MEMORY; the same on a 200 km moon and a 50 km body | M-L1 measured; the ship/derive decision (ask 2) |
| C2 THE INITIAL LAND | plates, the crust field, isostasy with the water load, orogeny under the relief law | the LAND hypsometric curve is BIMODAL (a gate that can fail); the slope histogram |
| C3 THE SOLVE | the schedule with the climate inside; sediment, flexure, talus, ice, coast, craters (ask 4); the envelope assertion | the per-basin hypsometric integral = the erosional age; every land node drains to a lake or the sea |
| C4 THE ARTIFACT AND THE READ | the rows, the pyramid, the digest, the home pin; `height_m` reads `Z` and the roughness field; the coarse octaves replaced; the ladder's coarsening re-measured; the version bump | no-drift on the artifact (byte-identical on every target); the seam/corner gate; the rung-disagreement judge under a pixel |
| C5 THE SEA | the sea re-solved over the new hypsometry; the draw deleted; the coast marked | the ocean share MEASURED (T8's cure); the water sheet judged wet |
| C6 THE LOOK | the seven stands + the VISTA stand (the largest-relief chunk, chosen by measurement), re-taken | the owner's look; the freeze on his word |

Coverage runs after C3 and after C5. Every gate flight waits for a quiet machine.

---

## 10. THE GATES

| gate | what it measures | red when |
|---|---|---|
| G-LAND-HYPSOMETRY | the land's height distribution has two humps (continents, ocean floors) at the depth the water load gives | one hump, or the floors under the surface by less than the law's step |
| G-DRAINAGE | every land node has a D8 path to a lake or the sea after the flood | an undrained node |
| G-AGE | the per-basin hypsometric integral against the system's age | outside the published band for that age |
| G-SLOPE | the slope histogram of the solve plus the spectrum | past the angle of repose anywhere; a 53° planet |
| G-SEAM | the interpolated field across a face seam and at a cube corner | any step past one integer |
| G-DRIFT | the artifact's digest on every target | any byte differs |
| G-SEA | the ocean share over the new hypsometry | 1.000 (the drowning), or the inventory not held |
| G-COST | the solve's wall time and memory (M-L1); the per-column recipe cost with the artifact read | outside the stated budget (8 ms a chunk) |
| the pictures | the seven stands re-frozen on the look; the vista stand as a candidate | as today |

---

## 11. THE ASKS (the owner decides; the recommendation first)

1. **The lattice's resolution.** Recommend `MACRO_CELL_TARGET_M = 8 192` (8 224 m nodes on the home
   planet, 22 MB kept). The finer 5 140 m costs 2.6× the memory and the time; the coarser 16 448 m loses
   the trunk valleys of a 50 km frame.
2. **Ship or derive.** Decided by C1's measurement: derive under a few seconds of one core, ship above.
   Recommend deciding after the number exists, not before.
3. **The erosional age.** Recommend the system's own age from the census (derived, T9), with the
   erodibility `K` calibrated on Earth's denudation rates. The alternative — a stated world constant —
   is a typed number.
4. **Craters.** Recommend in 8c as ruled: the production function by age and the air's screening by
   pressure, erased where the stream power acted. The alternative: the small craters in 8f.
5. **The pass counts.** Recommend the 03 values (40 passes, the climate every 10, the flood every 10,
   the rebound every 5, 8 talus passes) as COST knobs in the world tag, re-measured in C1.
6. **Two integer transcendentals.** The climate's laws need `exp` and `ln` (the vapour pressure, the
   lapse rate's integral); the recipe has the root and the reciprocal only. Recommend a bench first, the
   same shape as the integer bench of ruling F8, before C3.
7. **The vista stand.** Recommend it as the look's new candidate, chosen by measurement (the chunk with
   the largest relief in a window), so the reference picture's kind of view is judged where it exists.

---

## 12. WHAT 8c DOES NOT DO

Rivers as drawn things, lakes, the water sheet, the channel carve, the rock map and the authored override
(8d); the biome, the soil, the tree line, the paint table (8e); arches, overhangs, cave mouths (8f); the
ocean's surface, tides and sailing (8o); the almanac and the live weather (18). 8c gives every one of
them the skeleton and the rows they read.
