# Refutation B — domain 05, climate, biomes and weather

**Date:** 2026-09-08. **Target:** `docs/investigation/2026-09-08/landforms/05_climate_biomes_weather.md`, revision 1.
**Lens:** believability, cost and the owner. Would the result look like the reference picture? Would a
geologist or a climatologist accept it? Does it fit the budget and the boot? Does it answer what the owner
asked? Is it written so the owner learns the terms?

Every number below I computed from the cited file or from the shipped code. Where I ran arithmetic I state
the inputs, so the author can reproduce it or show me wrong.

---

## Blockers

### B1 — The ladder gate is claimed with the wrong bound: a metre bound does not bound a CATEGORY, and the biome flips by up to 1 499 m of height

§11, the ladder row, states: *"The biome at rung L is the biome of the coarsened hill, which is the coarse
answer of rung 0 within the same dropped-amplitude bound `dropped_bound_m` already proves
(`body.rs:264-270`)."*

`dropped_bound_m` bounds a HEIGHT in metres (`crates/terrain/src/body.rs:265-270`: the sum of the dropped
amplitudes). A biome is a CLASSIFICATION with hard thresholds. A bounded height error puts NO bound on a
classified output: one metre of height flips a column across the snow line, the sea line or the highland cut.
The law gate is therefore claimed with a bound that does not apply to the quantity it is claimed for.

The size is not small. The home planet draws 14 octaves with a roughness near 0.5, and its relief is
12 000 m × [0.5, 1.5) (`body.rs:145-148`: the 0.4 % share clamps at 12 000 for any radius over 3 000 km).
Computed here for relief 12 000 m and roughness 0.5:

| rung | dropped share of the relief | metres |
|---|---|---|
| 3 | 0.00043 | 5.1 |
| 6 | 0.00385 | 46.1 |
| 8 | 0.01556 | **186.8** |
| 11 | 0.12495 | **1 499.4** |

A vista to the horizon draws its far ground at rung 8 and beyond (ruling V10 §12). At rung 8 the height a
column reads is up to 187 m away from its rung-0 height. §5.6 makes the permanent snow line a comparison on
exactly that height. So the far ridge is white and the near ridge is not — or the reverse — and the white
cap MOVES as the pilot flies in and the rung changes. That is the "detail-by-box" and "arrival-pop" seam of
the eleven-seam taxonomy, on the single most visible feature of the owner's reference picture.

It is not hypothetical. The shipped `biome_at` already does it: `crates/terrain/src/height.rs:44-46` returns
`Highland` on `above_sea > highland_above_m`, reading `surface_m` at whatever rung the caller built
(`chunk.rs:180-181` passes the rung's own height). The defect exists today and the document's law-gate row
says the opposite.

**What is owed.** Either a stated CATEGORICAL bound (a hysteresis band around every threshold, sized from
`dropped_bound_m(rung)`, so a column near a line keeps its rung-0 class), or a measurement: the share of
columns whose biome differs between rung 0 and rung L, on the home planet, per rung. Neither exists.

### B2 — Lying snow as a per-cell live diff is unbudgeted, and the arithmetic breaks the diff lane

§8.2: *"lying snow is a LIVE diff on the surface cells of the realm — the same diff lane a player's edit
uses, and bounded by the same store. It applies where a player IS."*

Computed here. A rung-0 cell is 1 m. A snowfall over the ground a pilot can see from a hillside — call it
1 km × 1 km — is 1 000 000 surface cells. At the record's 12 bytes (ruling V6) that is **12 MB** of diff, for
one snowfall, on one planet, and it must reach every client in the realm. A snowfall over a 10 km × 10 km
valley is 1.2 GB. The document states no area, no rate, no cap and no measurement, and calls it "bounded by
the same store" — the store the format sitting sized for the cells a PLAYER changed by hand.

Three further consequences are not addressed at all:

- **The melt.** Every diff must be deleted when the snow goes. The diff store is the authoring override
  (ruling V6, the edit pyramid). A mechanism that writes and then un-writes a million authoring records per
  season is a different mechanism from a player's edit, not the same one.
- **The collision with a player's edit.** A player digs a trench; snow falls; the snow diff writes the same
  cells. Which record wins, and does the trench survive the thaw? Undecided.
- **SL10 clause 7.** A diff is the lawful way live state crosses. It is also the ONE hop the owning realm
  must pay for. §5.7's budget table has no row for it.

This is the point where the document's own load-bearing rule ("a season must never invalidate a chunk",
C-8) bends, and the bend is exactly the cost the rule exists to avoid — localised, but per storm and per
season rather than four times a year.

### B3 — The owner asked to SIMULATE THE WEATHER, and the live layer is one paragraph with no model, no cost, no determinism and no decision row

The owner's words: *"It should be very believable, as we also should simulate the weather."*

The document's answer to the second half is §8.1's third box and §8.3. In full, the live layer is: *"Storms
and fronts, gusts, the real cloud field, rain now, the depth of lying snow, the river's flow now,
lightning. WHO: the realm's own shard."* That is a LIST OF NOUNS. There is no model, no state, no
resolution, no tick rate, no bytes per tick, no shard budget line, no determinism story (a shard that
crashes and restarts must resume the same storm, or a checkpoint must carry it — the Category C rule), no
wire arm, and no measurement in §12.

§13 has fourteen decision rows. **Not one of them asks the owner to approve a weather simulation.** C-9 asks
for the SPLIT (static / almanac / live). C-10 asks that a dormant world runs on the almanac. The thing the
owner named is the one thing the owner is not asked about.

For scale: the static climate gets a derived grid, a cost table, an anchor set, a fence technique and eleven
measurements. The live weather gets six lines. On the "does it answer what the owner asked" lens, the
document answers "biomes from position, spin, trajectory, size and gravity" well, and answers "simulate the
weather" not at all.

---

## Defects

### D1 — Every cost and memory figure in §5.1 and §5.7 is computed on the body §10 of the same document says is the wrong one; the real numbers are four times larger

§5.1 works the grid on the CURRENT home planet (radius 3 350 759 m) and gets C = 128, 98 304 samples,
1.18 MB, about 30 ms. §10 and C-11 then say — correctly, in my reading — that the home planet must become
seed 2298's Earth-like body at 6 515.5 km, "and first".

Recomputed here for that body:

```
face edge = R * pi/2 = 6 515 500 * 1.5708 = 10 234 523 m
target_cell_m = 50 000            (long_wave_m clamps at 400 000; see D2)
10 234 523 / 128 = 79 957 m  > 50 000   -> C = 128 refused
10 234 523 / 256 = 39 979 m <= 50 000   -> C = 256
samples = 6 * 256^2 = 393 216      (4x the document's figure)
memory  = 393 216 * 12 B = 4.72 MB (the document's own C = 256 row)
build   ~ 4 x 30 ms = about 120 ms (every step scales with the sample count)
```

So the document's headline "1.18 MB and 30 ms once per body" becomes **4.72 MB and about 120 ms** on the
body it itself recommends. §5.7's summary table, the SL9 row of §11 and U3 all carry the wrong number.

The consequence the document never draws: this grid is built ON THE CLIENT too (§5.1: "on the server and on
the client alike"), on the critical path before the first chunk of a body is drawn. A pilot who approaches a
planet pays about 120 ms of worker time and 4.72 MB before the ground appears — per body. The home system
holds 9 planets and 19 moons (`census.rs:57-63`). Nothing in §5.7 budgets client memory or names the arrival
hitch as an SL8 seam.

### D2 — "The resolution is DERIVED, not a magic number" is degenerate: `long_wave_m` clamps to 400 km on every planet worth a vista

§5.1: *"the recipe already draws the coarsest wavelength: `long_wave_m`, a quarter to a half of the radius,
clamped to 20–400 km. Sample it eight times per wavelength ... the climate cell's target size falls out ...
There is no rounding to argue about."*

The code (`crates/terrain/src/body.rs:150-153`) clamps at `LONG_WAVE_CAP_M = 400_000`. The clamp binds
whenever `radius * 0.25 > 400 000`, that is for **every body with a radius above 1 600 km**. Both candidate
home planets clamp: 3 350 759 × 0.25 = 837 690, and 6 515 500 × 0.25 = 1 628 875.

So on every planet a player will ever stand on, `long_wave_m` is the constant 400 000 and `target_cell_m` is
the constant 50 000. The "derivation" reads a clamp, not a draw. The 50 km is a magic number wearing a
derivation's coat, and the document's §11 no-magic-numbers row rests on it.

§11 then concedes the other half in passing: *"The remaining literals are named and justified: the samples
per wavelength (8) ..."* — which contradicts §0 item 5 and §5.1's "not a magic number" framing. Eight
samples per wavelength is a choice (a Nyquist argument needs two), and it is the whole of the derivation once
the clamp binds.

### D3 — §5.3's circulation anchor table does not reproduce under its own formula and its own constant: Mars gives 2 cells, not 1

§5.3 states the Held & Hou form and *"ESTIMATED calibration, computed here from published body values at
`dH = 0.388`"*, then marks Earth, Mars, Venus and Jupiter each with a ✓.

I evaluated `sin(phi_H) = sqrt(5 dH g H / (3 Omega^2 a^2))`, `Omega = 2 pi / day_s`, at `dH = 0.388`, on the
published body values:

| Body | g | H | day | a | phi_H computed here | cells | the document says |
|---|---|---|---|---|---|---|---|
| Earth | 9.81 | 8 500 | 86 400 s | 6.371e6 | **30.08°** | 3 | 30°, 3 ✓ |
| Mars | 3.71 | 11 100 | 88 642 s | 3.3895e6 | **42.78°** | **2** | 90°, **1** ✗ |
| Venus | 8.87 | 15 900 | 2.0995e7 s | 6.0518e6 | **90°** (sin ≥ 1) | 1 | 90°, 1 ✓ |
| Jupiter | 24.79 | 27 000 | 35 640 s | 6.9911e7 | **3.06°** | 6 (clamped) | **1.9°**, 6 |

Two of four rows do not reproduce. Mars is not a rounding difference: 42.78° gives `round(90/42.78) = 2`,
which is a two-cell planet, and the document's ✓ claims a one-cell planet. Mars's scale height is robust
(`H = R T / (mu g) = 8.314 × 210 / (0.044 × 3.71) = 10 696 m`, so 10.8 km or 11.1 km both give 42–43°). A
✓-marked table labelled "computed here" that I cannot reproduce is an unmeasured claim dressed as a result —
the exact thing the standing ground rule forbids.

The knock-on matters for the picture: the cell count decides the desert belts, and §5.4 says *"This one term
places every large desert."* A model that puts Mars in the wrong cell class has not been checked at all.

### D4 — V12 is mis-cited: the ruling puts the light on the CLIENT deriving from a placement already shipped, not on a new almanac row

§8.1 justifies shipping the almanac with: *"Computed by the OWNING realm and shipped in a small row, because
V12 already put the light on the ship-it side"*, and again *"Not derived on the client: V12 put the light on
the ship side."*

The ruling text (`docs/design/owner_decisions_2026-09-07_voxels.md:484-503`) says the opposite of what is
claimed. V12's recorded answer is *"the parent authors the spin on the row, the recipe lives in the planet's
own frame, only the light changes on the surface"*, and S7-7 rules: *"One directional light from the
brightest luminous row in the window; normals derived on the client."* The light is DERIVED ON THE CLIENT
from the star's placement, which the window already carries. V12 adds no row and blesses no almanac lane; it
is a precedent for NOT adding one.

This is load bearing. C-9 asks the owner to approve a three-layer split on a precedent that does not exist.

### D5 — SL6 is asked for the charter and not asked for the two new lanes the same document invents

§3.4 gives a proper SL6 statement for the charter (what data, which realm to which, how often, why the
receiver cannot compute it, what doing without costs) and correctly says "This is an ASK. The default is NO."

The almanac row (§8.1) and the live weather diff (§8.1, §8.3) are also new data crossing a boundary, and
both are new wire arms. SL6 covers both by its own words: *"ASK BEFORE NEW DATA CROSSES A REALM BOUNDARY,
and before adding a wire arm. Default NO."* Neither gets a statement, a size, a rate, or an HR1 note on
which `InterShardFlow` arm carries it. The almanac's contents are given as a list — *"the season's phase,
the sub-solar point, the day's phase, the temperature now, the nominal cloud cover, the nominal river level,
the seasonal snow line"* — of which at least three (the temperature now, the nominal river level, the
seasonal snow line) are FIELDS over the body, not scalars, and cannot ride "a small row" at all.

### D6 — §11's SL1 row is false: the charter tells a child its own orbital distance and hands it the machinery to compute its own placement forever

§11, the SL1 row: *"The charter states what a body IS (its mass, its air, its spin), never where it is."*

Four rows of the §3.2 charter table are orbit facts, not body facts: `insolation_q12`, `ecc_q16`,
`arg_periapsis_q16`, `year_s`. Two consequences follow from the shipped laws:

- **Insolation IS a distance.** `sma = 0.4 · sqrt(L) · 1.7^n` and `insolation = 6.25 / 2.89^n`
  (`generate.rs:170-181`, `taxonomy.rs:1108-1110`; verified: n = 2 gives 0.748, which the code's own comment
  pins). A child holding its insolation and its star's class holds its own orbital radius. That is half of
  its own placement, stated to it.
- **The elements are a placement generator.** With `ecc`, `arg_periapsis`, `year_s` and the tick, a child can
  integrate its own orbit for every future instant. SL1 clause 3 forbids a child to derive, adjust or compute
  its own placement, and clause 6's staleness refusal has nothing to bite on when the child can extrapolate.

This may still be the right answer — a body arguably must know how much light it gets — but it is an OWNER
DECISION and the document denies that a decision exists. Compounding it, §8.1 and §11 disagree on who owns
the one datum that is plainly a placement: §8.1 says *"the OWNING realm computes"* the sub-solar point, and
§11 says *"the sub-solar point is the PARENT's to author, and the parent authors it."* Both cannot hold.

### D7 — Obliquity re-defines the body's own frame and breaks a shipped cross-pin, and the document never says so

§1 cites `crates/bins/tests/home_body_pin.rs:44-60` as evidence that obliquity is zero today: the pole axis
is cross-pinned against the orbits' axis. I read the test; the citation is correct. `height.rs:32-38` states
the same by design: *"the axis the world's orbits turn about ... so ice caps face away from the orbital
plane."*

C-2 then adds obliquity. The moment obliquity is non-zero, the spin axis is NOT the orbit's axis, so `+Z` in
the body's own frame must be re-defined from "the orbit's normal" to "the spin axis". That re-definition:

- makes `home_body_pin.rs:44-60` false as written (it asserts `POLE_AXIS == 2` *because* the perifocal plane
  is `z = 0`);
- changes what the parent authors as the child's ORIENTATION, which is a placement datum, so it changes the
  row, not just the recipe;
- makes `crates/bins/src/lib.rs:1007`'s stated method for the picture gate — *"the star's direction from the
  planet's centre is minus the planet's position in the system, in the planet's own frame while the planet
  does not spin"* — wrong, and every existing picture gate reads it.

§5.2's box asserts the opposite is free: *"OBLIQUITY DOES NOT BEND THE BIOME BANDS ... The spin axis is +Z in
that frame whatever the tilt is."* That is true of the RECIPE. It is not free of the FRAME, the row and the
pin. The cost is real and unstated.

### D8 — Precipitation is never computed in millimetres, yet the classifier's axis is millimetres

§5.4 gives `P(d, h) = supply(...) * cell_factor(x) * orographic(...) * continental(...)` — a product of four
dimensionless-looking factors with no unit-bearing constant anywhere. §6.1's Whittaker chart then reads an
axis labelled "MEAN ANNUAL PRECIPITATION (mm)" with cut lines at 200, 500, 1 000, 2 000, 3 000 and 4 000 mm.

Nothing in the document converts one into the other. `supply` is described as "the saturation vapour
pressure at the source temperature" — a PRESSURE in pascals, not a rainfall in millimetres per year. The
missing piece is a calibration constant with units (millimetres per year per pascal of supply, on an
Earth-like body), and it is the one constant the whole biome map's realism depends on.

Contrast the care taken elsewhere: the greenhouse gets four anchor bodies and a golden test (U7), the
circulation gets four anchors (U8), the vapour polynomial gets an error bound (U9). Precipitation — the
axis that decides forest against desert — gets **no anchor, no unit and no measurement row at all.** A
climatologist reads the chart and asks "in what units?", and the document has no answer.

### D9 — A 2-pass chamfer is the wrong algorithm on a closed cube sphere, and the iterative parts have no determinism statement

§5.1 and §5.4 build the distance to the sea by *"two chamfer passes over the cube sphere"*, and the moisture
by *"the moisture relaxation, 4 sweeps"*. ("Chamfer pass" is never explained — see W4.)

A two-pass (forward then backward raster) chamfer distance transform is correct on a bounded RECTANGLE,
where every shortest path is monotone in the scan order. A cube sphere is a CLOSED surface: it has no
boundary, it wraps, and a column on the `+X` face can have its nearest sea across the `−X` face. Two raster
passes cannot propagate a distance that wraps around the body; the result carries a visible discontinuity
somewhere on every face ring. The correct shapes are a multi-round sweep to convergence, or a bucketed
Dijkstra over `lattice::site_of`'s neighbour graph.

"Sweep to convergence" then collides with the document's own §4 rule. §4 refuses a Newton iteration by name
because *"The term count becomes a branch on a float, the branch is a coverage hole (HR5), and a value near
the branch is a drift risk."* A convergence test on a distance field is the identical hazard, and the
document does not apply its own rule to its own iteration. The 4 relaxation sweeps are never stated as a
FIXED count in a FIXED face order with no early exit — which is what SL10's byte-identity needs and what
HR5's branch coverage needs.

### D10 — The biome set does not add up, and the chart names a biome the set does not hold

- §0 item 6, §6.2's title and C-5 all say **sixteen** biomes.
- §6.2's enumeration lists **seventeen** names: Ocean, SeaIce, Beach, Ice, Tundra, Taiga, TemperateForest,
  TemperateRainforest, TropicalRainforest, Grassland, Savanna, Shrubland, Desert, ColdDesert, Wetland,
  Alpine, Regolith.
- §6.2's prose says *"Fourteen chart biomes plus two overrides"* = 16, and the chart of §6.1 draws
  **eleven** chart biomes.
- §6.1's chart draws **WOODLAND**, which appears in no enumeration and in no soil row of §7.1.
- *"Four bits hold sixteen and leave NO room"* — with seventeen values four bits do not hold them at all.

The five-bit conclusion happens to survive, but every count that supports it is wrong, and one biome exists
only in a picture. §7.1's soil table then covers 11 of the 17 and silently leaves Savanna, Shrubland,
ColdDesert, TemperateForest, TemperateRainforest and Ocean with no topsoil.

### D11 — The charter names no condensable volatile, yet §5.4 and §5.6 both depend on one

§5.6 clause 1: *"Where the annual mean surface temperature is below the freezing point of the volatile..."*
Clause 3: *"If the mean is above the volatile's boiling point at the surface pressure..."* §5.4's supply term
is Clausius–Clapeyron for the condensable.

The §3.2 charter carries `mu_q8` (the atmosphere's MEAN molecular weight) and `p_surf_pa`. Neither names the
condensable. On Titan the rain is methane; on Earth it is water; on a hot rocky world it is nothing. The mean
molecular weight of the bulk air does not determine the condensable: Earth's air is mostly N₂ (mu 28) and it
rains H₂O (mu 18); Titan's air is mostly N₂ too and it rains CH₄ (mu 16).

So the classifier cannot decide sea ice, the snow line, or a frozen world, on any body but the one the author
had in mind. An eighteenth charter field (the condensable's identity, with its freezing point and its latent
heat) is owed, and its DRAW is owed — and that draw is another owner decision C-3 does not ask.

### D12 — Nothing in the model couples climate to SHAPE, so a desert and a grassland have identical relief; §7.3's claim about cliffs and pillars does not follow

§11 makes it explicit and treats it as a virtue: *"The climate changes materials and cover, not the extracted
surface — except through the snow line and sea ice."*

That is the deepest believability gap in the document. Arid and humid landscapes are not the same hills with
different paint. An arid landscape has dunes, yardangs, mesas, alluvial fans and sharp scarps, because there
is no soil creep and no vegetation to hold the regolith; a humid one has rounded convex hilltops and
soil-mantled slopes. A geologist tells an arid range from a humid one in one photograph, at any distance.
In this model the home planet's desert is its grassland with a sand-coloured topsoil byte.

§7.3 nonetheless claims the reference picture's *"dry cliffs and rock pillars"* come from *"the rain shadow
and the aridity, plus the extractor's crease."* They do not. A rock pillar (a hoodoo) is DIFFERENTIAL
erosion of layered rock of unequal hardness. The shipped strata are horizontal layers of a single drawn
thickness for the whole body (`body.rs:186-200`: one `topsoil_m`, one `subsoil_m`, one `sediment_m`, one
sediment kind, one bedrock), so there is no lithological contrast to erode differentially anywhere on any
planet. The aridity term changes no cell's hardness and no cell's position.

The honest form is either a stated coupling (the climate as an input to the erosion domain's rate laws,
which domain 03 owns), or an explicit deferral in §14. There is neither: §14's seven questions do not
include it.

### D13 — At 41 km per climate cell the whole reference vista is one cell, so the biome varies only with height — the model's default output is contour banding

The reference picture is a vista to roughly 50 km. The climate grid is 41.1 km per cell on the current home
planet and 40.0 km on the Earth-like one (D1's arithmetic). So across the WHOLE picture, the grid terms — the
rain shadow, the distance to the sea, the upwind ocean fetch — are one or two samples, that is effectively
constant.

What is left varying inside the picture is the per-column closed form: latitude (which barely moves over
50 km), height (which moves a lot), and *"a small slow noise, for texture"* (§5.2). Therefore the biome inside
any single vista is, to first order, **a function of height alone**. The visible result is biomes as
elevation contour bands: a green band, then a brown band, then a white band, each following a height isoline
around every hill.

The document names this as U12 and Q7 and leaves it to the owner as a judgement on pictures. It is not a
judgement; it is a consequence of the design as specified, and it is predictable before any picture is
taken. The reference picture's most striking property — a green windward valley beside a dry leeward cliff
at THE SAME HEIGHT, kilometres apart — is exactly what a 41 km grid plus a height-driven closed form cannot
produce. §5.4's "local form" alternative does not rescue it either: the coarse height it samples uses 3
octaves, whose finest wavelength on the home planet is 100 km.

**What is owed.** A stated mechanism that varies moisture at the 1–5 km scale of the picture, with its cost;
or an honest statement that the picture's slope-to-slope contrast is out of reach, and what is offered
instead.

### D14 — §10's headline rests on an inference the cited code does not support

§10 and §0 item 10 lead with **"THE HOME PLANET IS THE WRONG BODY"** in bold, and C-11 makes it the FIRST
thing to do. The chain begins: *"`home_body` takes the FIRST planet row of the home system
(`crates/bins/src/lib.rs:992-1003`), which is the innermost."*

The code says something narrower. `home_body` is a `find_map` that returns the first planet **whose look
radius the ladder accepts**, and the doc comment says so by name: *"A planet the ladder refuses (above the
address) is passed over."* So the picked body is the innermost ACCEPTED planet, not the innermost planet.
The whole §10 estimate chain (0.093 M⊕, g 3.29 m/s², v_esc 4 693 m/s, 0.187 S⊕ retention against 6.2 S⊕
insolation) depends on the picked body being ladder rung n = 0.

Everything downstream of that assumption is arithmetically sound — I reproduced it: `insolation(n) =
6.25 / 2.89^n` from `sma = 0.4 · sqrt(L) · 1.7^n` (`generate.rs:180`, `taxonomy.rs:1108`) gives 6.25 S⊕ at
n = 0 and 0.748 at n = 2, which matches the document's `0.748 × 1.7^4 = 6.2`; the Chen–Kipping inverse gives
about 0.10 M⊕ at R/R⊕ = 0.526; `g = 3.30 m/s²`; `v_esc = 4 701 m/s`; `6.0229 × (4 701/11 180)^4 = 0.187 S⊕`.
The chain is right IF n = 0.

The document does flag U1 and says "Do this first: it decides §10". Good. But §0's bullet and §10's heading
state the conclusion in bold and unhedged, as the reason the vista cannot exist — and C-11's cost if the
owner says no is written as certain. An unmeasured inference should not carry a ★ and a "YES, and first".

---

## Weaknesses

### W1 — There is no ice-albedo feedback, so every polar cap is a perfect circle of latitude

`bond_albedo_q10` is a per-body charter CONSTANT (§3.2). The temperature field is diagnostic: the latitude
profile, the lapse rate, the continental term and a noise (§5.2). Sea ice then appears wherever that smooth
field crosses freezing (§5.6 clause 1).

On Earth the ice line is set by the ice-albedo feedback — ice is bright, bright ground stays cold, so the
line is sharp, irregular and hysteretic, and it is why a Snowball Earth is possible at all. With a constant
albedo and a smooth analytic profile, this model's sea-ice edge is a circle of constant `|dir.z|` plus a
small noise, on every ocean of every world. It will read as a drawn line. Also lost: the runaway states §9's
table wants ("a frozen world", "a boiled world") arrive as a threshold on the mean, rather than as a
bistable outcome, so a marginal world sits half frozen with no history.

To state this as a designed limit is fine. Not to name it is not.

### W2 — "This one term places every large desert on Earth" is false, and a climatologist will say so

§5.4, the cell factor: *"This one term places every large desert on Earth."*

The subtropical-high (Hadley sinking) term places the Sahara, the Arabian, the Kalahari and the Australian
deserts. It does not place: the Gobi and the Taklamakan (a continental interior plus a rain shadow),
Patagonia (a rain shadow), the Atacama and the Namib (cold upwelling coastal currents plus subsidence), or
the polar deserts (cold, not subsidence). Roughly half of the large deserts by area come from the other
three mechanisms, two of which this model has (the rain shadow, continentality) and one of which it does not
have at all (ocean currents).

The claim as written is the kind a reviewer checks in one minute and then distrusts the rest. Say instead:
"the cell term places the subtropical desert belts; the rain shadow and the continental term place most of
the rest; cold coastal currents are out of scope."

### W3 — §5.7's per-column climate cost is roughly three times faster than the document's own measured rate implies

§5.7 gives *"the climate at a column, the closed-form part (about 60 operations), 3 844 columns, 0.05 ms"*.

Computed here: 0.05 ms / (3 844 × 60) = **0.217 ns per operation** = about 0.65 cycles at 3 GHz, on a chain
that holds divisions. The document's own measured constant is 27.9 ns per octave evaluation (§5.1, derived
from the measured 1.5 ms / 3 844 columns / 14 octaves — I reproduced it). One octave of the shipped 3-D noise
(`crates/terrain/src/noise.rs`) is well over 60 operations including eight hash calls, so the code's own
measured rate is nearer 0.5 ns per operation. On that rate 60 operations per column cost about 0.13 ms per
chunk, not 0.05 ms.

The conclusion (0.15 ms against 4.7 ms of headroom) survives at three times the cost — but the row states no
operations-per-nanosecond and no derivation, in a document that is otherwise careful to source every number.

### W4 — Six terms are used and never explained, against the brief's own rule

The §2 glossary is genuinely good — twenty-two terms with a plain meaning and a game example each. Six terms
used later are missing from it and are never explained on first use:

| Term | Where |
|---|---|
| thermal Rossby number | §5.3:518 |
| chamfer pass | §5.1:382, §5.4:592 |
| Legendre coefficient, `P2(x)` | §3.2:215, §5.2:453-457 |
| Tetens form | §4:319 |
| biotemperature | §6.1:658 |
| aerial perspective | §10:912 |

Two of these carry the design: "chamfer pass" IS the distance-to-sea algorithm (see D9), and the "Legendre
coefficient" IS the charter's most important row (§3.2: "The last row is the important one"). The owner is
asked to approve C-1 and C-4 on words the document does not define.

Also: §4's techniques 2 and 3 are the only major parts whose examples are not in the game's own words (the
saturation vapour pressure, the air density against height). Each is one sentence away from an example about
the pilot's valley or the home planet's haze.

### W5 — U5 is a measurement that cannot fail, and it does not measure the thing at risk

U5: *"Perturb every charter float by 1 000 units in the last place before quantising, over a census of 1 000
bodies, and assert that no charter integer moves."*

Two problems.

- **It cannot go red usefully.** With a quantum near 10⁻³ of the value and a perturbation near 10⁻¹³ of it,
  the test passes for every body except one that sits within 10⁻¹³ of a quantum boundary. Over 1 000 bodies
  the chance of that is about 10⁻¹⁰. The test asserts a thing that is true by construction and reports it as
  evidence.
- **It does not measure the risk.** The risk is not "the same libm, perturbed"; it is "a DIFFERENT libm on a
  different target" — glibc against Apple's, x86-64 against aarch64 — where a chained `powf`, `ln` and `cos`
  expression can differ by far more than one unit in the last place of the final value. The honest
  measurement is the one D-TERRAIN-1 already does for chunks: compute the charter on both targets and
  compare the integers. U6 gestures at this for the climate GRID but not for the charter.

A related question is left implicit: §3.4 says the parent computes the charter and states it, while
`crates/physics/src/worldgen/body.rs:50-56` says *"every process derives the whole forest from the seed at
boot ... a planet shard computes its own mass, gravity, temperature and atmosphere locally, from the seed,
with no message."* Both a parent's stated charter and a shard's own derivation would then exist. Which one
wins on a disagreement, and what happens on a mismatch, is never said. §3.2 also folds the charter into the
world identity while §3.4 ships the charter to the client — so the client folds a number the server just
sent it, and the check can never disagree.

### W6 — "byte-identical" and "the required result is zero" are false precision for a rendered sky

§8.4: *"At T the live layer contributes exactly zero, so the sky at T is byte-identical to the sky at T-1."*
U10: *"measure the change in the rendered sky over the waking tick. The required result is zero."*

Between tick T−1 and tick T the star has moved, the camera has moved, the exposure has adapted, and every
float in the shading chain has changed. No two consecutive rendered frames of a live sky are byte-identical,
before any weather exists. The claim as written makes U10 a gate that must go red for reasons unrelated to
waking.

The measurable version: run the SAME tick twice, once with the realm dormant and once with it just woken,
from an identical camera, and require the two images to match within a stated per-pixel tolerance. That is a
comparison that can fail for the right reason.

### W7 — The haze the owner's picture shows needs more than a scale height

§7.3's row: *"blue haze with distance ← the atmosphere's scale height, a charter number."*

Aerial perspective is set by the optical depth along the line of sight: the surface number density, times a
scattering cross-section, times a path length, and it is wavelength-dependent (Rayleigh scattering goes as
1/λ⁴, which is WHY it is blue). The scale height gives only the vertical falloff. The surface density comes
from `p_surf`, `mu` and the temperature — all of which the charter carries — but the scattering
cross-section depends on WHICH GAS, which the charter does not carry (see D11). On a CO₂ world the haze is a
different colour and a different depth from an N₂ world at the same pressure.

The right row is: haze from `p_surf`, `mu`, the temperature and the gas identity, over the scale height. As
written the row is incomplete physics presented as an answer.

### W8 — The seed ruling is checked for ore and not for the things the climate actually makes valuable

§11's seed-ruling row is correct as far as it goes, and I verified its code claim: `strata.rs:1-5` does say
*"bulk stock only; every ore is live state; no indicator material"*, which matches ruling V4 row 15.3
(*"Default: NOT public"*).

Unconsidered: the climate is a seed-derived, publishable map of where the FORESTS are (timber), where fresh
water is, where arable ground is, and — on a tidally locked world — where the single habitable terminator
ring is. The 2026-08-27 ruling's own resolution probably covers it (*"an expensive gate buys ROOM, not
treasure: a new galaxy is worth reaching because nobody has taken it yet, which is live state"*): land
quality is room, and who took it is live state. But the document asserts the gate passes with *"no ore, no
deposit and no reward is a function of the climate"*, having checked ore only. State the argument; do not
assert the conclusion.

### W9 — HR5 and "every test runs on the home planet" cannot both hold for a 17-biome, charter-driven model

§11's SL5 row: *"every test runs on the home planet, as the crate already requires (`home.rs:5`)."*
§11's HR5 row: *"Every law is a straight-line expression over `Gf` ... the overrides are an ordered chain of
comparisons."*

An ordered chain of six overrides plus a chart is a lot of branches, and HR5 requires 100 % region AND branch
coverage in Tier-A. One body cannot drive them: if the home planet has air, the `has_air = 0` arm is dead; if
it is not tidally locked, the locked arms are dead; if `s2 < 0`, the high-tilt arms are dead; and a sixteen-row
Whittaker table needs a body that shows all sixteen classes, while §14's Q7 says a single Earth-like world
shows perhaps ten.

The existing crate already meets this by FORCING inputs rather than sampling (`height.rs:112-140`: "Forced
cases, so every arm is driven whatever the seed draws"). The climate's equivalent is to construct charters
directly in tests — which is lawful, and is NOT an SL5 variant, because a charter is per-body data of the one
world. The document should say so, because as written its two law-gate rows contradict each other, and a
reader could conclude that a second test world is needed.

### W10 — Two small sourcing lapses, in a document that set the standard itself

- **"12 bytes" per grid sample** (§5.1's memory column) has no source and no field list. Every memory number
  in §5.1 and §5.7 scales with it. It appears to borrow the record's 12 bytes (ruling V6), which is a
  different thing entirely.
- **§1's table cites a RULING for a row headed "what the code holds today"**: *"The client's realm box
  already carries the recipe tag and the look shell's radius"* is sourced to
  `owner_decisions_2026-09-07_voxels.md` V12 / S7-3, not to code. The document's own header says *"Every
  claim about the code cites `file:line`"*, and the standing rule is "read the code, not the docs". The claim
  is in fact true — S7-3's landing note says it is built — but the row is sourced the way the document
  forbids.

---

## Checked, sound

These I tried to break and could not.

- **The fence claim.** `crates/terrain/src/gf.rs` offers exactly add, subtract, multiply, divide, negate,
  square root, floor, truncate, absolute value and comparison, with four `compile_fail` controls and a
  private field. The §1 row and §4's premise are exact.
- **The client links no motion crate.** `crates/client/Cargo.toml:24-26` puts `vd-physics` under
  `[dev-dependencies]`, with the comment *"DEV-ONLY ... the shipped client never names a motion (SL4)"*.
  §3.1's wall is real.
- **The look-radius precedent.** `crates/terrain/src/home.rs:44-77` is a real measurement: +1e-3 m, −1e-3 m
  and +1 000 units in the last place all give the identical body, and the test also pins the snap unit that
  DOES move it. The document reads it correctly and does not overstate it.
- **The 27.9 ns per octave.** 1.5 ms / 3 844 / 14 = 27.87 ns. Reproduced.
- **The refusal of a per-column upwind march.** 16 × 3 × 27.9 ns = 1.34 µs per column, × 3 844 = 5.15 ms
  against 8 ms with 3.3 ms already spent. Reproduced. To refuse it by measurement is the right shape.
- **`long_wave_m`'s draw and clamp, the relief law, the sea offset, the biome field's parameters.** Every §1
  row about `body.rs:139-232` matches the code line for line.
- **The column pass.** `chunk.rs:143-200` does compute the direction, the height and the biome per column,
  and it keeps the lowest and the highest. The climate does slot in there.
- **`Empty` appended at code 18, 19 strata, `Biome::ALL.len() == 4`.** All three verified in `strata.rs` and
  its tests; §7.1's append rule is the shipped pattern.
- **`POLE_AXIS` is cross-pinned to the orbits' axis.** `home_body_pin.rs:44-60` verified, exactly as §1 says
  (which is why D7 bites).
- **The insolation ladder arithmetic.** `0.748 × 1.7⁴ = 6.25 S⊕` is the correct n = 0 value under
  `sma = 0.4 · sqrt(L) · 1.7ⁿ`; I reproduced 0.748 at n = 2 independently.
- **The Earth lapse-rate check.** `9.81 / 1004 = 9.77 K per km`, and `c_p = 3.5 R / mu = 1004.8 J/kg/K` for
  mu = 0.02896. Both reproduce. Earth's 6.5 / 9.77 = 0.665 is right.
- **Held & Hou for Earth.** 30.08° at `dH = 0.388`, giving 3 cells. Reproduced exactly. (Mars and Jupiter do
  not — see D3.)
- **`s2 ≈ −0.48` for Earth, and the sign change near a 54° tilt.** Both are the standard energy-balance
  results, and both are correctly stated.
- **The eccentricity arithmetic.** `((1+e)/(1−e))² = 1.069` at e = 0.0167, 1.49 at e = 0.1, and 2.25 at
  e = 0.2. Reproduced.
- **The vegetation anchor arithmetic.** 62² = 3 844 m² per rung-0 chunk; × 0.06 stems per m² = 231 anchors;
  × 8 B = 1.8 KB against 405 KB of measured geometry. Reproduced, and 600 stems per hectare is a real
  temperate forest density.
- **The three fence techniques, and the refusal of a data-dependent iteration.** §4 is the strongest section
  in the document. The per-body constant, the fixed-degree polynomial and the fixed table are the right
  three, and to refuse Newton by naming both the HR5 hole and the drift risk is exactly right. (D9 is that
  the document then breaks its own rule elsewhere.)
- **The three-layer split as a SHAPE.** Static, then a closed-form almanac, then a live diff is the right
  decomposition, and "a season must never invalidate a chunk" is a genuinely load-bearing rule, correctly
  identified.
- **The owner's five words are all answered** by the static model: position (insolation), spin (Coriolis and
  the cell count), trajectory (eccentricity and `arg_periapsis`), size (radius, grid, gravity) and gravity
  (`Γ = g / c_p`, and the snow line with it). This is the document's real achievement, and it is the reason
  the refutations above are worth fixing rather than starting over.
