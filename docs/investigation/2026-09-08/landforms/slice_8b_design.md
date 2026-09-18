# Slice 8b — THE CHARTER: the implementation design

**Written 2026-09-18, on `601a1e1` (Step 17) plus the uncommitted 8a stages 1–5 and the T7 step.**
This document is the build plan for 8b, the third item of ruling T1
(`docs/design/owner_decisions_2026-09-16_terrain.md`). It changes no code, no test and no picture.

**What binds it.** T1 (the whole of 8 before 9; 8b's scope), **T2** (the sea level and the water
inventory in the charter, stored), **T3** (no time in the shape), **T6 ask 1** (the relief law moves
to 8b) and **★ T6 ask 2** (*"not based on the percentage, but based on the physical laws or some
models; let's see what numbers it will produce"* — NO interim ocean fraction). Above them: V13 **L12**
(the charter: about twenty quantised integers, authored once and stored), **A1** (the charter rides
the realm's own surface statement to its observers' clients), **A2** (the orbit-derived facts,
quantised, from the star system to its planet, one hop down), **L27** (the home planet's facts),
SL1, SL3, SL5, SL6, SL10, HR3 and HR5.

**What it is NOT.** It is not the macro solve (8c), not the sky (8s), not the water sheet or the rock
map (8d), not the biome (8e), not the third dimension (8f) and not the ocean (8o). Where 8b carries a
number that only a later slice reads, this document says so by name.

**The one-line summary.** 8b gives the generator the body's physical facts as whole numbers, replaces
two lotteries — the relief draw and the sea draw — with two physical laws, and ships the numbers to
the client on the lane the surface statement already uses. It opens **no new wire arm**.

---

## 1. THE CHARTER TODAY (measured from the code, 2026-09-18)

Every claim in this section is read from the tree. Line numbers are of the working tree with 8a
stages 1–5 and the T7 step applied.

### 1.1 There is no body charter in the code

**MEASURED.** `vd_terrain::BodyDefinition::from_seed(seed: u64, look_radius_m: f64)`
(`crates/terrain/src/body.rs:724`) takes a seed and a radius, and nothing else.
`crates/terrain/Cargo.toml` lists exactly two dependencies, `vd-seed` and `vd-recipe`. A search for
`gravity`, `BodyTaxon` and `vd_physics` under `crates/terrain/src/` returns nothing. So no physical
fact of a body reaches the recipe today. The only census number that crosses into the generator is
the **look radius**, handed over by the caller at `crates/bins/src/lib.rs:1003`.

The word "charter" is already taken in the code, and it means something else: it is the **word block
the kernels read**, not the body's physical facts. The two must not be confused. §1.2 and §1.3
inventory the word blocks; §1.4 inventories the physical facts, which live in another crate and cross
nothing.

### 1.2 `CellCharter` — the per-cell word block

`crates/recipe/src/cell.rs:256-292`. `#[repr(C)]`, every field a `Gi` (the fenced integer word,
`crates/recipe/src/gi.rs:53`). **18 words, 144 bytes**, asserted at `crates/terrain/src/gpu.rs:687`.

| # | Field | Line | Unit | Quantum | The kernel that reads it |
|---|---|---|---|---|---|
| 0 | `sea_radius` | 261 | gap steps | `LENGTH_BITS` = 28 (`cell.rs:37`) | `fluid_code` (311-317) |
| 1 | `cave_min` | 263 | gap steps of depth | 28 | `in_band` (305-308) |
| 2 | `cave_max` | 264 | gap steps of depth | 28 | `in_band` |
| 3 | `cavern_threshold` | 267 | a share | `NOISE_BITS` = 28 | `cavern_hollow_steps` (339-346) |
| 4 | `cavern_scale_steps` | 268 | gap steps per noise unit | whole | `cavern_hollow_steps` |
| 5 | `carve_any` | 271 | 0 or 1 | whole | `in_band` |
| 6 | `rung` | 274 | a shift count | whole | `cell_word` (376, 382, 397) |
| 7 | `topsoil_m` | 277 | whole metres | whole | `stratum_code` (323-338) |
| 8 | `subsoil_end_m` | 278 | whole metres | whole | `stratum_code` |
| 9 | `strata_end_m` | 279 | whole metres | whole | `stratum_code` |
| 10 | `air_code` | 282 | a host code byte | whole | `fluid_code`, `cell_word:400` |
| 11 | `water_code` | 283 | a host code byte | whole | `fluid_code` |
| 12 | `bedrock_code` | 284 | a host code byte | whole | `stratum_code:336`, `below_cell_word:415` |
| 13 | `box_edge` | 288 | cells | whole | the GPU shell only (`crates/recipe-gpu/src/lib.rs:78`) |
| 14-17 | `strata: [Gi; 4]` | 291 | three code bytes a word | rows at 0, 8, 16 (`cell.rs:53-57`) | `stratum_code:324` |

It is filled by `charter_of` (`crates/terrain/src/chunk.rs:242-273`).

**★ The one row 8b touches is row 0.** The sea radius is ALREADY a kernel word. 8b changes how it is
computed; it does not add a word to this block.

### 1.3 `PlanCharter` — the per-column word block

`crates/recipe/src/plan.rs:41-71`. `#[repr(C)]`, **173 words, 1 384 bytes**, asserted at `gpu.rs:688`.

`seed` (45), `cavern_recip` (48), `cavern_shift` (49), `inv_n` (52), `radius` (54), `octave_count`
(56), `key_face` (58), then `biome: BiomeCharter` (60, 20 words, `crates/recipe/src/height.rs:281-295`),
`roughness: Roughness` (64, 10 words, `height.rs:121-134`), `terrace: Terrace` (68, 8 words,
`crates/recipe/src/terrace.rs:59-78`) and `octaves: [Octave; 16]` (70, 128 words, `height.rs:56-76`).

An `Octave` carries `seed`, `frequency_int`, `frequency_frac` (at `NOISE_BITS` = 28), `amplitude` (at
`AMP_BITS` = 8, so 1/32 768 m at the metre rung), `kind` (smooth 0 / ridged −1) and `fine`
(coarse 0 / fine −1).

**No word in this block is a physical fact of the body.** Every one is a seed draw or a derived
integer.

### 1.4 Where a body's physical facts live today, and what they never touch

**MEASURED.** Three records hold every physical fact, all inside `crates/physics`, and **none of them
ever reaches the wire or the generator**.

- **`StarPhotometrics`** — `crates/physics/src/worldgen/body.rs:66-78`: `mass_msun` (68), `class`
  (71), `luma_lsun` (77).
- **`BodyTaxon`** — `crates/physics/src/taxonomy.rs:894-909`: `class` (897), `mass_kg` (899),
  `radius_m` (901), `insolation_rel` (903), `t_eq_k` (905), `bond_albedo` (907), `atmosphere` (908).
  `Atmosphere` (`taxonomy.rs:881-888`) holds `mean_molecular_weight`, `scale_height_m` and
  `reference_density_kgm3`. The doc at `taxonomy.rs:890-893` states that surface gravity and escape
  velocity are **not stored**; every caller recomputes them.
- **`OrbitalElements`** — `crates/physics/src/celestial.rs:136-152`: `sma` (138), `ecc` (141),
  `inclination` (143), `raan` (145), `arg_periapsis` (147), `mean_anomaly_epoch` (149),
  `central_mass` (151), with `period()` at 170.

`crates/physics/src/worldgen/body.rs:46-51` states the rule in the code's own words: the taxon is
**never lowered onto a `RealmRegion` and never on the wire**; every process derives it from the seed.
`to_regions` (`body.rs:206`) drops both `taxon` and `photometrics` and keeps the look.

### 1.5 How a realm states its surface today, and how the client decodes it

- **The type.** `vd_core::look::SurfaceStmt` (`crates/core/src/look.rs:69`) has exactly **two**
  fields: `frame: FrameRef` (70) and `generator: u64` (72).
- **The budget.** `SELF_LOOK_BUDGET_BYTES = 1200` (`crates/core/src/look.rs:79`), tied by assertion to
  `vd_wire::channels::CONSERVATIVE_DATAGRAM_BUDGET = 1200`
  (`crates/wire/src/channels.rs:646`, `1155-1157`).
- **★ The measured occupancy.** `crates/core/src/look.rs:504` pins the WIDEST possible self-look bag —
  the widest frame, a full outline, the widest luma, `generator = u64::MAX` — at **exactly 124 bytes**:
  `assert_eq!(len, 124, "the widest self-look bag moved")`. **So 1 076 of the 1 200 bytes are free, and
  that is a measurement, not an estimate.**
- **The tags.** `TAG_LOOK = 1` (31), `TAG_LUMA = 2` (35), `TAG_EXTENT = 3` (48), `TAG_SURFACE = 4`
  (56), `TAG_BODIES = 5` (63, reserved with no producer). The codec is `surface_look_bag`
  (`look.rs:210-219`) and `surface_of` (`look.rs:223-225`), postcard inside a TLV field, with
  skip-unknown at `look.rs:336-346`.
- **The producer.** `current_bodies` (`crates/sim/src/stub/window.rs:477`) fills exactly one row, the
  realm's own `BodyStmt::SelfLook`; the per-child marker loop was deleted on 2026-09-04
  (`window.rs:512-520`). `emit_window_rosters` (386) sends **on change, by digest** (414-417).
- **Where the statement is built.** `crates/bins/src/bin/shard.rs:398-413`. A realm states a surface
  if and only if its frame carries a planet seed AND the ladder accepts its look radius.
- **The wire.** `ShardToGateway::WindowBody { … stmt: BodyStmt … }`
  (`crates/wire/src/session_flow.rs:327-335`), `BodyStmt::SelfLook { bag: Vec<u8> }` (667), relayed
  sealed inside `InterShardFlow::WindowRelay` (`crates/wire/src/intershard.rs:475`), composed by the
  gateway (`crates/connection-plane/src/window.rs:155, 282-292, 1187-1191`), delivered as
  `SceneRow.bag` (`crates/wire/src/channels.rs:522`).
- **The client.** `crates/client/src/realm_scene.rs:404` decodes it into `RealmBox.surface` (119).
  `crates/client-render/src/terrain.rs:2410-2412` is the one production caller of
  `state_surface`. `crates/client/src/chunks.rs:1497-1511` is the whole build: a foreign generator tag
  is refused and counted (`foreign_generator`, 1501); anything but
  `(FrameRef::PlanetCentered, Boundary::Shell)` is refused and counted (`no_body`); otherwise the
  client calls `BodyDefinition::from_seed(planet_seed, r)`.

**★ So the client builds a planet from exactly two numbers: the seed and the shell radius.**

### 1.6 How the charter reaches the card

`crates/terrain/src/gpu.rs`: `CHARTER_WORDS = 18` (47) with `charter_words()` (401-425);
`PLAN_CHARTER_WORDS = 173` (58-59) with `plan_charter_words()` (442-469). The size gate is at
`gpu.rs:687-696`. The card reads the same words the CPU reads, in `repr(C)` order, and that is what
makes the no-drift gate meaningful.

### 1.7 The two lotteries 8b replaces, as the code writes them

| The thing | Where | What it is today |
|---|---|---|
| the relief | `body.rs:731-733` | `min(0.004·R, [200, 12 000]) × (0.5 + u)`. **`relief_m` is never stored** — it is a local inside `from_seed` |
| the sea | `body.rs:861-864` | `sea_radius = radius + floor((u·0.7 − 0.4) · relief_m)`; a uniform draw over −40 % to +30 % of the relief |
| the band | `body.rs:1028-1032` | `relief_whole = floor(Σ|a| + pull) + 1`; the ladder is taken a second time from it |

**MEASURED on the current home planet** (read from the tree's own comments): `relief_m = 16 454 m`,
and the fine spectrum's own sum is **1 532 m**, so the `fine_scale` arm at `body.rs:782` is never
taken on this body.

### 1.8 The digest folds the CELLS, not the charter

`digest_of` (`crates/terrain/src/digest.rs:33-47`) folds the chunk key and then, for every cell, the
stratum code and the gap. **It folds no charter word.** So a charter word that moves a cell is caught
by the golden tables and the no-drift gate; **a charter word that no kernel reads is caught by
nothing.** §6 owes those words their own pin.

---

## 2. THE DRAWS 8b ADDS

Every draw below is authored ONCE, by the body's own realm, with full `libm` precision, and floored to
a whole number in a stated unit. **The integer is the fact.** Nothing downstream ever re-derives it.
That is the authoring rule `02_planet_layout.md` §5.4 adopted after both refuters refused
"quantisation makes a drift harmless", and it is the rule 8b builds.

### 2.0 The float→integer door, and where it stands

The 8a stages put every new term through one door in the DRAW (`length_of`, `share_of`, `draw_m`,
`rounded`, all in `crates/terrain/src/units.rs` and `body.rs:549-593`). 8b uses the same shape, one
step earlier: **the `libm` arithmetic happens in `vd-physics`, the floor happens in the authoring
shard, and `vd-terrain` receives whole numbers.** The generator's fence (`Gf`, `crates/terrain/src/gf.rs:1-25`)
never sees a `powf`, an `ln` or a `cos`. The invariant already written at `body.rs:85-87` —
*"no float from an unfenced crate can enter the recipe as a body"* — is **kept**, because what enters
is an integer, not a float.

### 2.1 The relief law (T6 ask 1)

**The law.** `relief_m = draw × min(σ_y / (ρ_c · g), 0.077 · R)`, with `draw ∈ [0.5, 1.0]`.

**The published source.** The strength bound is the crustal-strength relation of
`02_planet_layout.md` §4.3: a mountain stands until the pressure at its root passes the crust's
long-term yield stress and the root flows. `σ_y = 243.1 MPa` is a **calibration on one body** —
Everest, `2 800 × 9.81 × 8 850` — in the same house style as the project's other calibrations. What is
physical is the `1/g` trend, and the spread over the three bodies that have built mountains is
Earth 100 %, Mars 94 %, Venus 112 %. The shape bound `0.077 · R` is cited to **Vesta**, an observed
small differentiated body (`06_laws_integration.md` §3.4). COMPUTED there: the two bounds cross at a
radius of about **939 km**, so a planet is strength-limited and a moon is shape-limited.

**The inputs, and where each comes from.**

| Input | Source | New? |
|---|---|---|
| `g` | `surface_gravity_mps2(mass_kg, radius_m)`, `taxonomy.rs:750` | **already in the census** |
| `ρ_bulk` | `mass_kg / ((4/3)π·radius_m³)`, computed on the spot by `home_body_facts.rs:28-29` | **already in the census** |
| `ρ_c` | **nowhere.** §8 ask 3 | NEW, a model choice |
| `σ_y` | a stated constant, calibrated on Everest | NEW, one constant |
| `R` | the census radius | already |

**Quantisation.** `gravity_mm_s2: u32` (whole mm/s², 1 part in 9 821 on the home planet) and
`bulk_density_kgm3: u32` (whole kg/m³, 1 part in 5 515). One step of either moves the relief cap by
about 0.01 %, which is under a metre.

**What it does to the home planet.** COMPUTED BY HAND, from L27's facts and
`home_body_pin.rs:10-11`:

- census radius `6 370 747.312 696 504 m` (asserted, `home_body_pin.rs:11`), mass `1.000 M⊕`;
- `g = G·M/R² = 9.821 m/s²`; `ρ_bulk = M / ((4/3)πR³) = 5 514.7 kg/m³`;
- with `ρ_c = 2 800 kg/m³` (Earth's, which this body's bulk density reproduces exactly):
  **the strength bound is `243.1e6 / (2 800 × 9.821) = 8 840 m`**;
- the shape bound is `0.077 × 6 370 747 = 490 545 m`, so **the strength bound binds**;
- today's relief is **16 454 m** (MEASURED). Under the law with `draw ∈ [0.5, 1.0]` the relief is at
  most **8 840 m**, and at the same stream position (`u = 0.871`, implied by 16 454 / 12 000) about
  **8 271 m**.

**★ So the home planet's relief falls by about half: 16 454 m → about 8 271 m (ESTIMATED; the stage's
test prints the real number).** The band is re-derived from `Σ|a| + pull` (`body.rs:1028-1032`), so
`Ladder::floor_m` and `Ladder::band_m` move again, every chunk's radial index `z` moves, and every
golden row is re-recorded. `Ladder::n`, `Ladder::rungs` and every `(face, rung, x, y)` do **not**
move, for the reason `slice_8a_design.md` §3.1 gives: `Ladder::for_radius`
(`crates/seed/src/ladder.rs:82-125`) reads neither `crust_m` nor `above_m`.

### 2.2 The spin, under tidal locking

**The published source.** Two bounds, both physical.

- **The lower bound is break-up**: a body cannot turn faster than the period at which a point on its
  equator flies off, `P_min = 2π√(R/g)`. COMPUTED for the home planet: `2π√(6 370 747 / 9.821)` =
  **5 060 s = 1.406 h**.
- **The upper bound is tidal locking.** The despinning time scales as `a⁶`, so the locking distance is
  a sharp threshold. From the standard despinning law, `a_lock ∝ t^(1/6) · M★^(1/3) · ρ^(−1/6) ·
  P₀^(−1/6)`; the two sixth roots are near one, so the law is `M★^(1/3) · t^(1/6)` with a
  **calibration on one body**, exactly as `σ_y` is calibrated on Everest. The calibration body is
  **Mercury**, despun at 0.387 AU around the Sun in 4.5 Gyr.

**The draw.** `day_s` log-uniform over **[6 h, 100 h]**, the band anchored on the solar system's rocky
bodies (`05_climate_biomes_weather.md` §3.3 B), refused below `P_min`. Then the locking test: a body
inside `a_lock` is set to `day_s = year_s` and carries a `tidally_locked` flag.

**The inputs.** `sma` and `central_mass` from `OrbitalElements` (`celestial.rs:138, 151`) —
**already in the census**. The age: `SYSTEM_AGE_GYR = 5.0` (`taxonomy.rs:718`) is a stated constant
today, so `age_myr` is carried as `5 000` and is not yet a draw. `ρ_bulk` — already.

**Quantisation.** `day_s: u32`, whole seconds. One second in an Earth-like day is 1 part in 86 400;
the Hadley edge moves under 0.01°.

**COMPUTED for the home system.** `a_lock = 0.387 × (0.953)^(1/3) × (5.0/4.5)^(1/6) = 0.3876 AU`.
The orbit ladder is `sma = 0.4 · √L · 1.7ⁿ AU` (`config.rs:58-59`, `generate.rs:178-180`), so for the
home star (`L = 0.823 L☉`) `a_n = 0.362 876 × 1.7ⁿ AU`:

| rung n | `a_n` (AU) | locked? |
|---|---|---|
| 0 | 0.362 88 | **YES** (0.3629 < 0.3876) |
| 1 | 0.616 89 | no |
| **2 (the home planet)** | **1.048 71** | **no** |
| 3 | 1.782 81 | no |
| 4–8 | 3.03 … 25.31 | no |

**So exactly one planet of the home system's nine is tidally locked** — the innermost. A permanent day
side and a permanent night side is a major gameplay fact, and it falls out of the law, not out of a
switch.

### 2.3 The damped obliquity

**The published source.** Laskar, Joutel & Robutel 1993: Earth's 23.4° tilt is stable **because of the
Moon**; without it the tilt wanders chaotically. The physical control is the ratio of the satellite's
torque on the equatorial bulge to the star's:

```
        Λ  =  (m_moon / M★) · (a_orbit / a_moon)³
```

COMPUTED for Earth: `(7.35e22 / 1.989e30) × (1.496e11 / 3.844e8)³ = 3.696e-8 × 5.896e7 = 2.18`.
That is the published lunisolar torque ratio, which is why this form is offered and not another.

**The draw, then the damping.** Draw `cos ε` from the isotropic prior in a stated band; then pull it
toward the median by `1/(1 + Λ)`. A body with a large close moon keeps a familiar tilt; a moonless
body wanders. The **prior's band** (`05` §3.3 A recommends `cos ε` uniform over [0.2, 1.0] for nine
bodies in ten and over [−1.0, 0.2] for the tenth) is an owner decision, and so is the damping's shape:
**§8 ask 4**.

**The inputs.** The moon list and each moon's mass and orbit — **already in the census**
(`append_moons`, `generate.rs:1659`). The planet's own orbit — already.

**Quantisation.** `obliquity_cos_q1024: i32`, the **COSINE and never an angle**, because `sin` and
`cos` are banned inside the fence (`06_laws_integration.md` §4.2). One step is 0.002 46 rad at
Earth's tilt, which is 8.2 km of ice-cap edge on a body this size — coarse, and it is the right
coarseness because the ice-cap edge is 8e's, not 8b's.

**`POLE_AXIS` does not move.** The obliquity is the angle between the spin axis and the orbit's
normal, and **the parent already owns that relation** (SL1 clause 1); it is carried by the
parent-authored orientation (`StampedPose.orient`, `crates/core/src/pose.rs:919`). The recipe needs
only the scalar. This resolves the disagreement between `02` §5.4 (which wanted two angles) and `06`
§4.2 (which wants one scalar): **06 is later, it reasons from the code, and 8b takes one scalar.**

### 2.4 ★ THE WATER INVENTORY — the physical model (T6 ask 2)

The owner refused a percentage. This is the model, and **every law it uses is either already in the
code or is a published relation with one calibration body.**

#### 2.4.1 The formation zone, against the star's snow line

**★ The snow line is already in the code**, and it is already the right law:
`frost_line_radius_au(luminosity_lsun, frost_coeff_au) = 2.7 · √L` — `taxonomy.rs:519-521`, with
`FROST_COEFF_AU = 2.7` at `taxonomy.rs:340`. It is computed for planets at `generate.rs:1168` and for
moons at `generate.rs:1680`, and it already decides a rock core against an ice core. **8b adds no
second snow line.**

The physics: inside the snow line, water is a vapour in the disc and the body accretes dry rock;
outside it, water condenses and the body accretes solar-composition condensates, which are about
**half ice by mass** (Lodders 2003). A body between the two gets what its feeding zone swept in.

**The model.** A log-linear ramp in the ratio `r = a / a_frost`, anchored at **two published points**:

```
        r_E  = 1.0 AU / 2.7 AU   = 0.370 37      Earth's own ratio
        f_E  = 2.26e-4                           Earth's ocean over Earth's mass
        f_ice = 0.5                              solar-composition ice fraction, at r = 1

        log10 f(r) = log10 f_E + (r − r_E)/(1 − r_E) · (log10 f_ice − log10 f_E)
                   = −3.6459 + 5.3125 · (r − 0.370 37),   capped at −0.301 03 for r ≥ 1
```

**One calibration body (Earth) and one physical endpoint (the condensate ratio).** No tuned exponent.

#### 2.4.2 The retention, against the star's heat

**★ Both retention laws are already in the code.** 8b adds neither.

- **The cosmic shoreline** (Zahnle & Catling 2017) — `cosmic_shoreline_retains` with
  `SHORELINE_COEFF = 6.0229` (`taxonomy.rs:789`) and `xuv_rel_of` (`taxonomy.rs:802`). It is the same
  predicate that already decides whether a body keeps an atmosphere at all. **A body that fails it
  keeps no water.** This is thermal and hydrodynamic escape, in the form the census already trusts;
  the `jeans_retains` gauge beside it (`taxonomy.rs:814`) is the classical Jeans check, recorded.
- **The runaway greenhouse** — `KOPPARAPU_FLUX_CONSERVATIVE = (0.53, 1.10)` (`taxonomy.rs:700`).
  A body above 1.10 S⊕ holds its water as steam, the steam is photolysed, and the hydrogen leaves.
  **A body above the inner edge keeps no ocean.**

**The verdict is HARD, not a ramp** — retained or stripped — because that is what the census's own
atmosphere draw does, and a soft ramp needs a steepness nobody has measured. §8 ask 5 states the
alternative.

#### 2.4.3 The result, and the unit

```
        V_water  =  f(r) · M_body / ρ_w          ρ_w = 1 000 kg/m³
        V_water  =  0                            if the shoreline strips it, or S > 1.10 S⊕
```

**Quantisation: `water_km3: u64`, whole cubic kilometres.** One km³ over the home planet's surface is
**two microns** of sea level — free precision in a `u64`.

*The alternative in `02` §5.4, "whole parts per million of the body's mass", is REFUSED by arithmetic:
one ppm of the home planet is 5.97e18 kg of water, which is **11.7 km of mean sea depth per step**.
One step would move the coast off the planet.*

#### 2.4.4 The numbers this model produces

**★ The orbit ladder is seed-free, so the water FRACTION of every planet of the home system can be
computed by hand today. Only the masses are owed.** COMPUTED here from `config.rs:58-59`
(`a_n = 0.4·√L·1.7ⁿ`), `taxonomy.rs:519-521` (`a_frost = 2.7·√L = 2.4494 AU` at `L = 0.823`) and
`insolation_rel = L/a²` (`taxonomy.rs:932`), which reduces to `S_n = 6.25 / 2.89ⁿ` and is therefore
**the same for every star at every seed** — which is exactly why the code's own pin reads
`0.748_314_795` at rung 2 (`worldgen/tests.rs:3291`).

| rung | `a` (AU) | `r = a/a_frost` | `S` (S⊕) | `f(r)` | the verdict |
|---|---|---|---|---|---|
| 0 | 0.3629 | 0.1482 | 6.250 | 1.49e-5 | **dry** — far over the runaway edge, and tidally locked |
| 1 | 0.6169 | 0.2519 | 2.163 | 5.30e-5 | **dry** — over the runaway edge |
| **2 = the home planet** | **1.0487** | **0.4282** | **0.7483** | **4.58e-4** | **wet** — both verdicts pass |
| 3 | 1.7828 | 0.7279 | 0.2589 | 1.79e-2 | **very wet**, and cold |
| 4 | 3.0308 | 1.2374 | 0.0896 | 0.5 | **ice world** — half ice by mass; no sea (§3.5) |
| 5–8 | 5.15 … 25.31 | 2.10 … 10.33 | 0.031 … 0.0013 | 0.5 | **ice worlds** |

**THE HOME PLANET, COMPUTED IN FULL (ESTIMATED until the code runs it):**

```
        r        = 1.048 71 / 2.449 4          = 0.428 15
        log10 f  = −3.6459 + 5.3125 × 0.057 78 = −3.3389
        f        = 4.58e-4                     (2.03 × Earth's water fraction)
        M        = 1.000 M⊕ = 5.9722e24 kg
        V_water  = 4.58e-4 × 5.9722e24 / 1 000 = 2.737e18 m³ = 2.737e9 km³
        shoreline: the body HAS an N₂-like atmosphere (L27), so it is retained
        runaway:   0.748 S⊕ < 1.10 S⊕, so no steam loss
        ⇒ the home planet holds about TWICE Earth's ocean.
```

**The moon.** A moon forms in its planet's own sub-disc, so **its formation zone is its planet's
orbit**, and the code already computes the frost line for a moon from the same star at
`generate.rs:1680`. A moon of the home planet therefore reads the same `r = 0.4282` and the same
`f = 4.58e-4`. Then the shoreline decides, and it decides against a small body at 0.748 S⊕ — our own
Moon, at 1 S⊕ with an escape velocity of 2.38 km/s, is bare. **So a moon of the home planet comes out
DRY, holds no water, runs no bisection and gets no sea.** The moon's own mass and radius are
**OWED** — they are a seed draw, and no golden file holds them (§9).

**The largest and the smallest planet.** Their masses are drawn between
`PLANET_MASS_LO_MEARTH = 0.0553` (`generate.rs:109`) and `planet_mass_hi_mearth`
(`generate.rs:1594`), and **no test or golden file holds the home system's table**. So the volumes are
OWED. What is NOT owed is the verdict, because the verdict reads the orbit rung, which is seed-free:
**the smallest and the largest, wherever they sit, follow the table above** — dry at rungs 0 and 1,
ocean at 2 and 3, ice from 4 out. The first thing 8b does is run
`crates/bins/examples/home_body_facts.rs`, extended to print the water model's answer for all nine
planets and fifteen moons (§6, M-W).

### 2.5 The optical depth

**Two different optical depths are wanted, and the code has neither.**

- **`tau_vis_q12`** — the **Rayleigh** optical depth at 550 nm, for the sky and the aerial perspective
  (8s's input). The published relation: the scattering optical depth is the column number density over
  the fourth power of the wavelength, and the column is `p_surf / (g · μ)`. Calibrated on Earth
  (Bodhaine et al. 1999: Earth's Rayleigh depth at 550 nm is **0.0973**):

  ```
        τ_vis = 0.0973 · (p/p⊕) · (g⊕/g) · (μ⊕/μ)
  ```

  COMPUTED for the home planet at one bar, `μ = 28` (N₂-like): `0.0973 × 1 × (9.81/9.821) ×
  (28.96/28) = 0.1005`. In `q12` that is **412**.

- **`tau_ir_q12`** — the **grey greenhouse** optical depth, which sets the surface temperature:
  `T_s = T_eq · (1 + ¾·τ_ir)^(1/4)`. Calibrated on Earth: `(288 / 254.3)⁴ = 1.6452`, so
  `τ_ir⊕ = 0.860`.

**The input the code does not have.** `p_surf` has **no derivation** and the code says so
(`taxonomy.rs:876-879`, D-TAX-1). So 8b must draw it. `05` §3.3 C recommends a log-uniform draw over
**[1 000 Pa, 10 000 000 Pa]** where a secondary atmosphere is retained, a band that holds Mars,
Titan, Earth and Venus. **It is a DRAW, not a derivation, and the code must say so.** The band is
**§8 ask 6**.

**Quantisation.** `p_surf_pa: u32` (whole pascals), `tau_vis_q12: u32`, `tau_ir_q12: u32` (1/4096).

**★ AND HERE IS A NUMBER THE OWNER MUST SEE.** COMPUTED with Earth's own `τ_ir`:

```
        T_s(home) = 236.8 K × (1 + 0.75 × 0.860)^(1/4) = 236.8 × 1.1325 = 268.2 K
```

**That is five kelvin below the freezing point of water.** With an Earth-like greenhouse the home
planet is a **snowball**. This is §8 ask 2, and the recommendation there is the
carbonate–silicate thermostat (Walker, Hays & Kasting 1981), which is the law the habitable zone
itself is built on: on a cold planet the thermostat lets CO₂ build up until the surface warms.
COMPUTED with `τ_ir = 2.0`: `T_s = 236.8 × 2.5^(1/4) = 297.7 K`, a warm, liquid world. The thermostat
is a LAW and not a tuning, and it makes the census's own earth-like predicate self-consistent.

### 2.6 The elastic thickness

**The published source.** The effective elastic thickness of the lithosphere (Watts 2001, *Isostasy
and Flexure of the Lithosphere*) tracks the depth to a fixed isotherm, so it falls as the surface heat
flow rises. Radiogenic heat scales with mass and escapes through the surface area, so `q ∝ ρ·R`,
falling with age.

```
        T_e = T_e,ref · (q_ref / q),      q ∝ ρ·R / (1 + t/τ)
```

with `T_e,ref = 35 km` at Earth's `ρ·R` and 4.54 Gyr. The decay is written as a rational and not as an
exponential, because `exp` is banned inside the fence and because the producer should use the same
shape the recipe would.

**COMPUTED for the home planet:** `ρ·R` equals Earth's to four figures; the age is 5.0 Gyr against
Earth's 4.54, so `q/q⊕ = 0.9516` and **`T_e ≈ 36.8 km`**.

**What the recipe does with it, and when.** Nothing, in 8b. `T_e` is 8c's input: it sets the
**flexural parameter** `α = (4D / (Δρ·g))^(1/4)` with `D = E·T_e³ / (12(1−ν²))`, which is how far a
load's support spreads and therefore whether a shelf exists. **The fourth root is two square roots,
and the fence has the square root**, so the recipe derives `α` itself and 8b need carry only `T_e`.
COMPUTED for the home planet with `E = 100 GPa`, `ν = 0.25`, `Δρ = 3 300`: **`α ≈ 86 km`**, which sits
inside Earth's observed 50–130 km. That agreement is a sanity check and not a validation.

**Quantisation.** `elastic_thickness_m: u32`, whole metres.

**★ Stated plainly: `T_e`, `day_s`, `obliquity_cos_q1024` and both optical depths are carried in 8b
and read by NO kernel in 8b.** They are carried now because the charter is authored once, and because
appending a word after slice 14 is a promise the freeze takes away. §6 gives them the pin they need,
since §1.8 shows no digest can catch them.

---

## 3. THE SEA FROM THE VOLUME

### 3.1 The mechanism, as `slice_8a_design.md` §1.6 and `02` §7.1 describe it

1. **The sample directions come from the LADDER, never from a random sampler.** The cell centres of a
   coarse rung: **26 cells a face edge gives 26² × 6 = 4 056 directions**, through the bend the crate
   already has. A fixed list, a fixed order, a fixed trip count, no rejection loop, no new branch.
   Each sample carries its cell's own area factor as a weight, which is a fenced expression in the
   cube-sphere Jacobian.
2. **Each sample evaluates the FULL field** — the octaves, the ridge, the roughness and the terrace —
   because that is the surface the water stands on.
3. **Bisect for the radius at which the water volume above the shape equals `V_water`**, in a
   **fixed 24 steps**, never "until converged" (a convergence comparison is a drift class). Twenty-four
   halvings of a 40 km range reach **2.38 mm**.

### 3.2 Where the volume is measured, and the error

**The resolution.** 4 056 samples over the home planet's ladder sphere (`radius_m = 6 341 670.0 m`,
`home.rs:54`) is one sample per `1.246e11 m²`, that is **a patch about 353 km on a side**. So the
bisection sees the two coarsest octaves properly and averages everything finer.

**What that biases, and what it does not.** Each sample evaluates the full field, so the sampled
heights carry the full field's spread; the sampling loses the spatial CORRELATION, not the
distribution. So it biases the shape of a coastline, which 8b does not draw, and not the LEVEL.

**The level's error, COMPUTED.** `02` §7.1 gives the sampling error of an ocean fraction near 0.5 from
about 4 000 samples as **0.78 % of the surface**. Turned into a level through the field's own density:

- near an even split the level is good to about **45 m**;
- on a nearly drowned body (a fraction near 0.999) the level is good only to about **600 m**,
  because the distribution's tail is flat.

**RECOMMENDATION:** run the coarse rung first; if the solved fraction falls outside [0.05, 0.95], run
once more at the next finer rung (52 cells an edge, 16 224 samples, four times the cost, half the
error) and state which rung answered. That is a bounded, stated escalation and not a loop.

**The water load is NOT in 8b's loop.** `02` §7.1 puts the isostatic water load inside the bisection.
**8b has no crust field**, so there is nothing for the load to press on. 8b solves the dry volume;
**8c adds the load term, one multiply, inside the same loop.** The mechanism does not change, which
is the whole point of building the bisection once.

### 3.3 The sea radius as a stored charter integer (T2)

T2 rules the sea level **in the charter, stored**. 8b obeys, and the shape is:

- the **owning realm** runs the bisection once, in `from_seed`, and stores the answer;
- it states the answer in its own surface statement, as **`sea_offset_mm: i32`**, the offset of the
  sea from the ladder radius in whole millimetres (the offset is always under the relief, so `i32`
  covers ±2 147 km);
- the **client does not re-bisect.** It reads the integer and converts it with `length_of`, exactly
  as `body.rs:861-864` does today.

**Two things this buys.** The client's message path loses the bisection's cost (`02` §7.1 COMPUTES
2.1 ms a body, and a star system stating a planet and twenty moons in one frame would pay it twenty-one
times — `tick-hitch` is a named seam). And 8d, 8o and the collider read one number instead of
re-running a solve.

**One thing it owes.** A stored number that both hosts could also derive is a second source, and SL5
and HR3 refuse a second source that can disagree. **So the gate is a measurement:** the stored offset
must equal the offset the bisection derives, on both hosts, for every pinned body (§6, G-SEA).

### 3.4 What the recipe reads, before and after

| | today | after 8b |
|---|---|---|
| where the sea comes from | `body.rs:861-864`, a uniform draw in [−0.4, +0.3] × relief | the bisection over the shape, against `water_km3` |
| where the recipe reads it | `CellCharter.sea_radius` (`cell.rs:261`) → `fluid_code` (311-317) | **unchanged** |
| the word block | 18 words | **unchanged** |

**So the kernels do not change at all.** Only the number in row 0 changes, and only `from_seed`
changes to compute it. That is what makes 8b's kernel risk small and its DRAW risk large.

### 3.5 The rule an ice world needs

A body beyond the snow line holds half its mass as water, and **that water is not a sea**: it is an
ice mantle. The sea exists only where the condensable is liquid at the surface, which is the
condensable table of `05` §3.4 read at the charter's surface temperature and pressure.

**The rule, stated:** the bisection runs only when the condensable is liquid at the surface. Otherwise
the body carries `water_km3` with **no sea offset**, the `has_sea` flag is clear, and `fluid_code`
answers rock everywhere. *In the game's words: the pilot lands on the fourth planet and walks on a
shell of ice a hundred kilometres thick. There is water under his boots and there is no coast, because
nothing there is liquid.*

### 3.6 ★ THE PREDICTED OCEAN FRACTION, AND THE FINDING THE OWNER MUST JUDGE

COMPUTED BY HAND (ESTIMATED; the code has not run it):

```
        ladder radius              6 341 670 m        (home.rs:54, MEASURED)
        surface area  A = 4πR²  =  5.0538e14 m²
        V_water                 =  2.737e18 m³        (§2.4.4)
        mean water thickness    =  V/A = 5 416 m
```

The field's own spread: **MEASURED 2 297 m on the OLD home planet** (`02` §1.4). On the current body
it is **UNMEASURED**; scaling by the relief gives about **2 640 m today**, and about **1 400–1 700 m
after the relief law halves the relief** (§2.1).

For a one-humped field the water covers a share `Φ(z)` where `σ·[φ(z) + z·Φ(z)] = 5 416 m`:

| the field's σ | z | **the ocean's share** |
|---|---|---|
| 2 640 m (today's relief) | 2.03 | **97.9 %** |
| 1 500 m (after the relief law) | 3.58 | **99.98 %** |

**★ THE HOME PLANET DROWNS.** Under the physical model it is a water world with a few islands.

**And the cause is NOT the water model.** COMPUTED, as the control that could have refuted it: put
**Earth's own ocean** (a mean thickness of 2 647 m) on the same one-humped field at σ = 1 500 m, and
the share is still **96 %**. Earth reaches 71 % only because 71 % of its surface stands **4.5 km below**
the other 29 % — the crustal dichotomy, which is **isostasy on two crust types, and that is 8c's**
(`02` §4.2; `discussion` L8). Our planet has one hump, so **any** Earth-like inventory drowns it.

Turned round, the size of the gap is exactly measurable: to leave 30 % of the surface dry on a
one-humped field at σ = 1 500 m the planet may hold only **5.4e17 m³**, which is **0.40 of Earth's
ocean and one fifth of what the model gives**. So the arc must close a factor of five, and 8c closes
it with a mechanism the arc already owns.

**This is the number T6 ask 2 asked for, and §8 ask 1 puts the decision back to the owner.**

### 3.7 What the picture stands do

**MEASURED.** The stands do **not** hold a fixed altitude. `eye_surface`
(`crates/client/src/chunks.rs:1708-1720`) reads the recipe's own surface under the eye and returns the
altitude from it; the ground stand stands 1.8 m over that surface
(`crates/bins/tests/terrain_pictures.rs:16`). The far and aloft stands stand at `R / sin(fov·share/2)`,
a function of the radius alone.

So:

- **the eye follows the ground automatically** when the relief halves — no stand needs re-siting for
  the relief law;
- **the skyline gate's tolerance moves.** Its floor is "the sphere of the surface under the eye minus
  the recipe's relief bound" (`terrain_pictures.rs:130-133`), and both terms move. The MEASURED
  28 rows / 1.75° allowance was read on the old shape and is re-read;
- **the chunk counts move** (MEASURED today: 3 855 / 4 300 / 2 012 / 1 277, `terrain_pictures.rs:137`);
- **★ and the ground stand may end up under water.** At a 98–100 % ocean share the ground stand, the
  hill stand and the seam stand are all very likely at sea. The sea is not drawn until 8d, so what the
  owner would see in 8a's pictures is **dry ground below a sea level nothing draws** — a picture that
  looks right and is a lie. §8 ask 1 is therefore not optional: it must be answered before the
  pictures are judged.

---

## 4. A1 AND A2 — THE TWO CROSSINGS

### 4.1 (a) A1 — the charter in the realm's own surface statement

**This crossing is real, it is needed, and 8b builds it.**

**Why the client cannot compute it.** MEASURED: `crates/client/Cargo.toml:9-16` links `vd-terrain`,
`vd-seed`, `vd-core`, `vd-devproto`, `vd-wire` and `vd-sim`; **`vd-physics` is under
`[dev-dependencies]` only** (line 25-26), with the comment *"the shipped client never names a motion"*.
The client holds the planet's seed and its shell radius and nothing else
(`chunks.rs:1506-1511`). It cannot derive `g` without the mass, and it may not link the crate that
holds the mass.

**The record 8b proposes.** Twenty whole numbers, in one new TLV field beside `TAG_SURFACE`.

| # | field | type | unit | source | read by a kernel in 8b? |
|---|---|---|---|---|---|
| 1 | `gravity_mm_s2` | `u32` | mm/s² | `surface_gravity_mps2`, `taxonomy.rs:750` | through the relief DRAW |
| 2 | `bulk_density_kgm3` | `u32` | kg/m³ | mass / volume | through the relief DRAW |
| 3 | `escape_velocity_mps` | `u32` | m/s | `escape_velocity_mps`, `taxonomy.rs:756` | no — 8c |
| 4 | `insolation_q12` | `u32` | 1/4096 S⊕ | `BodyTaxon.insolation_rel` | no — 8e |
| 5 | `t_eq_mk` | `u32` | mK | `BodyTaxon.t_eq_k` | no — 8e |
| 6 | `t_surface_mk` | `u32` | mK | `t_eq` and `tau_ir` (§2.5) | no — 8e |
| 7 | `bond_albedo_q12` | `u32` | 1/4096 | `BodyTaxon.bond_albedo` | no — 8s |
| 8 | `mu_q8` | `u32` | 1/256 u | `Atmosphere.mean_molecular_weight` | no — 8s, 8e |
| 9 | `scale_height_m` | `u32` | m | `Atmosphere.scale_height_m` | no — 8s |
| 10 | `p_surf_pa` | `u32` | Pa | **a new draw** (§2.5) | no — 8s |
| 11 | `tau_vis_q12` | `u32` | 1/4096 | **new** (§2.5) | no — 8s |
| 12 | `tau_ir_q12` | `u32` | 1/4096 | **new** (§2.5) | no — 8e |
| 13 | `day_s` | `u32` | s | **a new draw** (§2.2) | no — 8c, 18 |
| 14 | `obliquity_cos_q1024` | `i32` | 1/1024 | **a new draw** (§2.3) | no — 8c, 8e |
| 15 | `water_km3` | `u64` | km³ | **new** (§2.4) | through the sea bisection |
| 16 | `sea_offset_mm` | `i32` | mm | **new**, the bisection's answer (§3.3) | **YES — `CellCharter.sea_radius`** |
| 17 | `elastic_thickness_m` | `u32` | m | **new** (§2.6) | no — 8c |
| 18 | `ecc_q16` | `u32` | 1/65536 | `OrbitalElements.ecc` | no — 18 |
| 19 | `year_s` | `u64` | s | `OrbitalElements::period()` | no — 18 |
| 20 | `flags` | `u32` | bits | `has_air`, `solid_surface`, `tidally_locked`, `has_sea`, the condensable code, `star_class` | through the sea rule (§3.5) |

**The byte count, against the measured budget.**

```
        the raw record                     17 × 4 B + 2 × 8 B + 1 × 4 B  =  88 B
        postcard varints, worst case       17 × 5 B + 2 × 10 B + 1 × 5 B = 110 B
        the TLV tag and length                                           +  3 B
        ------------------------------------------------------------------------
        the field, worst case                                     ESTIMATED 113 B

        the widest self-look bag today, MEASURED (look.rs:504)            124 B
        the widest bag after 8b                          ESTIMATED about  237 B
        SELF_LOOK_BUDGET_BYTES (look.rs:79)                             1 200 B
        headroom left                                    ESTIMATED about  963 B
```

**So the charter costs about a fifth of the free space, and 963 bytes stay free.** The stage's
failing-first test pins the real number the way `look.rs:504` pins 124 today — **the estimate above
is not a measurement and must not be quoted as one.**

**The decode on the client.** `surface_of` grows a sibling, `charter_of_bag`, under a new
`TAG_CHARTER = 6` (`TAG_BODIES = 5` is reserved and must not be reused). `RealmBox` grows
`charter: Option<BodyCharter>` beside `surface`. `state_surface` (`chunks.rs:1497`) grows **one
branch**: a realm whose bag carries a surface but no charter is **REFUSED and counted** —
`no_charter`, beside the existing `foreign_generator` and `no_body`. **No default is ever
substituted**, which is the project's own rule ("decode-to-Default is BANNED for Durable kinds") and
SL1 clause 6's shape.

**The no-drift consequence.** `BodyDefinition::from_seed(seed, look_radius_m)` becomes
`from_seed(seed, look_radius_m, &charter)`. Both hosts then read **the same integers**, so the
client's chunks equal the server's by construction. The old proof stands unchanged: the look radius is
an input from outside the fence and the ladder snaps it, measured at
`crates/terrain/src/home.rs:49-77`. **The charter's integers need no snap, because nobody rounds them
twice: the author floors them once, and every reader reads the whole number.**

**Where the straddle risk actually sits, and it is real.** The author evaluates `powf`, `ln` and `cos`
with `libm`. If the same body is drawn on an x86-64 shard and on an aarch64 shard, one may land either
side of a grid line. **That risk is UNMEASURED**, and §6's M-T measures it — it is the same
measurement `D-TERRAIN-1`'s G4 leg already owes.

**The SL6 statement, in the law's own four parts.**

- **What data.** Twenty whole numbers, about 113 bytes, per drawable round body, on change, retained.
- **From which realm to which.** **It is not realm to realm.** It is a realm's statement ABOUT ITSELF,
  on the bag that already carries its surface tag, to the clients the gateway composes for. The
  connection plane is not a realm (SL2's 2026-08-24 clarification). The precedent is exact:
  `TAG_SURFACE` is already the realm's own statement about what shapes it (ruling V6 row R-8), and the
  2026-09-05 suit ruling blesses the pattern by name — a thing states its rating as a fact about what
  it IS.
- **Why the receiver cannot compute it.** Measured above: the shipped client links no motion crate.
- **What doing without costs.** No relief law, no water inventory, no sea, no climate — the owner's
  requirement by name.

**What it is NOT.** It is not on the parent's per-child bag. SL3 says a parent's per-child message
carries a placement and nothing else, and the one-radius law says the same
(`crates/core/src/look.rs:41-47`). **A dormant realm sends no self-look, so a dormant body ships no
charter — and a dormant realm is never drawn, so nobody needs one.**

### 4.2 (b) A2 — the orbit half, one hop down

**★ MEASURED FINDING: A2 needs no lane, and 8b should open none.**

V13 A2 approved the orbit-derived facts crossing from the star system to its planet, one hop down.
The approval was given on the belief that the planet cannot see its own orbit. **The code says it
can.**

- A shard boots **its own subtree AND its lineage**: `shard_boot_world(seed, config, held, hosted,
  lineage)` (`crates/physics/src/worldgen/forest_query.rs:116-125`) calls `realm_subtree(seed, config,
  held, lineage)`. So a planet's shard holds its parent chain up to its star system.
- `shard_boot_world_lit` (`forest_query.rs:145-152`) returns, in one build, the region rows, **the
  mover roster `Vec<(RealmId, OrbitalElements)>`** and **the photometric draw
  `Vec<(RealmId, StarPhotometrics)>`**.
- The shard binary links `vd-physics`. `body_params` (`taxonomy.rs:921-960`) derives the whole
  `BodyTaxon` from the seed.

**So the planet's own shard already holds its orbit, its star and its mass, and it derives its charter
locally. Nothing crosses a realm boundary.** This is SL6's own instruction — *"find the local
formulation first; there usually is one"* — and `05` §3.1 reached the same place: *"the child needs
no message, and the parent's bag may not carry one."*

**The exact wire arms, measured, and why none of them fits anyway.** The full `InterShardFlow` list is
`crates/wire/src/intershard.rs:131-634`, 45 arms, discriminants pinned by
`crates/wire/tests/intershard_closed.rs:806-858`. Only three run **parent → one direct child**:

| arm | disc | line | payload | fits? |
|---|---|---|---|---|
| `RealmInterest` | 35 | 508 | one optional distance | no |
| `LineageStated` | 39 | 568 | a coord and a fence | no |
| `ChildFelt` | 44 | 633 | `felt: [i64; 3]`, micro m/s² | no — closed, and **it has no producer** |

`ChildFacts` (37, line 545) runs **upward**, child → parent, on change — it is the movement contract's
"what I AM" lane (`mass_g`, `cross_section_mm2`, `drag_micro`, `declared`;
`crates/wire/src/intershard.rs:1455-1477`, produced by `emit_child_facts`,
`crates/sim/src/stub/drive.rs:661-704`, sent strictly on change at line 678). **Its shape is the right
shape for a facts lane, and its direction is the wrong direction.** `ReachStated` (43, line 610) is
the other on-change up-lane, and every realm sends it.

**THE RULE 8b WRITES.**

1. **The planet's own shard derives and quantises its charter at boot, from the subtree it already
   holds.** No message, no arm, no new field on any inter-shard lane.
2. **A shard that cannot derive its charter states NO surface.** It does not guess and it does not
   default. That is `shard.rs:398-413`'s existing shape: a realm states a surface if and only if the
   recipe defines its body; 8b adds "and the charter is in hand". SL1 clause 6: *a stale reading is
   refused, never used*; an absent one likewise.
3. **A client that holds no charter for a realm refuses that realm's surface and counts the refusal**
   (§4.1). One realm's ground is missing; the world stands.
4. **V13 A2's approval is kept in reserve, unspent.** The day a body's facts stop being a pure function
   of the seed — a terraformed atmosphere, a stripped ocean, live state — the local formulation dies
   and A2's lane is the answer. **On that day a new arm at discriminant 45 is the shape**, a twin of
   `ChildFelt` carrying the quantised integers, and it is already approved. 8b does not build it.

**The cost of the recommendation, named.** Every planet's shard pays `body_params` for its own body at
boot. That is one taxonomy row, not a forest fold, and the boot already builds the subtree. **The cost
is UNMEASURED and M-B measures it.** If it is not free, the fallback is the boot plant — the route
`ChildLuma` already takes at `crates/bins/src/lib.rs:3103-3113`, where the parent's photometric datum
is planted at boot rather than sent on a wire — which still crosses no realm boundary.

---

## 5. WHAT MOVES

### 5.1 Addresses — no

`Ladder::for_radius` (`crates/seed/src/ladder.rs:82-125`) reads neither `crust_m` nor `above_m`, so
the home planet keeps `N = 9 961 472`, 19 rungs and a ladder radius of 6 341 670 m through the whole of
8b. **Every chunk's `(face, rung, x, y)` is the chunk it is today.** A chunk's radial index `z` moves,
because the floor moves.

### 5.2 The version — one bump, shared with 8a

T6's refined order puts 8b's draws between 8a's stage 5 and 8a's stage 7. **So there is ONE version
bump for 8a and 8b together**, at the end: `GENERATOR_VERSION` **3 → 4** (`crates/terrain/src/tag.rs:25`),
with `DECLARED_PIN` re-recorded (`tag.rs:99`, today 10 363 069 377 454 183 796). One epoch, not two.

This is lawful for the reason `slice_8a_design.md` §3.4 gives: T1 states the free window in the
owner's own words — *"during that time the ground may move freely — nothing is built on it"* — and F1's
append-only promise begins at slice 14.

### 5.3 The digests

- `crates/terrain/tests/golden_home.txt` (1 458 × 3 rows) and `golden_home_mesh.txt` — **re-recorded at
  every stage that moves a byte**, with the commit stating which stage moved it and why
  (`crates/terrain/tests/terrain_pin.rs:29-41`, `mesh_pin.rs`).
- `WorldIdentity::measured` (`tag.rs:46-56`, `digest.rs:351`) moves with them; it is stored nowhere and
  every process re-folds it.
- `crates/recipe` `NOISE_PIN` / `VALUE_PIN` — **no**. The noise kernel does not change.
- `crates/bins/tests/home_body_pin.rs` — **grows**. It pins two literals today
  (`home_body_pin.rs:10-11`) and must pin the home body's **whole charter**, cross-checked against the
  forest's own draw, exactly as it cross-checks the radius. `crates/terrain/src/home.rs`'s
  `home_planet()` (line 29-32) must state the charter too, because the client's binary folds the world
  identity **before it connects to anything** (`crates/bins/src/bin/client.rs:117-118`), so for the
  home body there is no author to state one yet.

### 5.4 The pictures

The five frozen references and the far candidate move again
(`crates/bins/tests/terrain_pictures.rs:183-190`). They are re-frozen **once**, with 8a's, on the
owner's single look — which is what T1 promises. **§3.7 is the warning that goes with them: the stands
may be at sea.**

---

## 6. THE MEASUREMENTS FIRST, AND THE GATES

### 6.1 The numbers the owner judges (T6 ask 2)

These run **before** the sea is wired into a picture.

**M-W — THE WATER TABLE.** Extend `crates/bins/examples/home_body_facts.rs` to print, for all nine
planets and fifteen moons of the home system: mass, radius, `g`, bulk density, escape velocity,
insolation, `T_eq`, the atmosphere, `a`, `a_frost`, `r = a/a_frost`, `f(r)`, the shoreline verdict, the
runaway verdict, **`water_km3`**, the relief cap, the drawn relief, **the solved sea offset** and **the
ocean share**. One afternoon, no gate. **This is the report T6 ask 2 asks for, and the owner judges it.**
§2.4.4's four rows are the hand computation it must reproduce or refute.

**M-R — THE RELIEF CAP.** Print, for the same bodies, today's relief against the new cap, and the
change in `Ladder::floor_m` and `Ladder::band_m`. The home planet's hand number to beat:
**16 454 m → about 8 271 m**.

**M-S — THE SEA'S COST AND ITS ERROR.** The bisection's wall time per body on the quiet machine
(`02` §7.1 COMPUTES 2.1 ms on the old octave count, before the ridge and the terrace — **the real
figure is higher**), and the solved level at 4 056 samples against the level at 16 224, on five bodies.

**M-B — THE BOOT.** The added cost of `body_params` on a planet's shard at boot (§4.2), against the
MEASURED 13.5 ms of the boot self-check.

**M-T — THE AUTHOR'S STRADDLE.** Draw the home system on both shipped target legs and compare **every
charter integer**. This is the one measurement that can refute the authoring rule, and it is the same
leg `D-TERRAIN-1`'s G4 owes. A pass is "every integer equal on both legs"; a fail names the field and
its distance from the grid line.

### 6.2 The pins

- `just terrain-pin` — `terrain_pin` and `mesh_pin`, three ways (debug, release, and release with
  `-C target-cpu=native`), over 1 458 chunk columns × 3 rows (`justfile:797`). Re-record at every
  stage that moves a byte.
- `just terrain-link-scan` (`justfile:801`) and `just terrain-fence-control` (`justfile:814`) —
  **unchanged and must stay green**. They are how a float that arrives through a dependency is caught,
  and 8b adds a whole new class of numbers that must not arrive as floats.
- **NEW: `charter_pin`.** §1.8 measures that no digest folds a charter word, so **`day_s`,
  `obliquity_cos_q1024`, `tau_vis_q12`, `tau_ir_q12`, `elastic_thickness_m`, `ecc_q16`, `year_s` and
  `escape_velocity_mps` are invisible to every existing gate.** `charter_pin` states the home body's
  twenty integers as literals and cross-checks them against the forest's live draw, in
  `crates/bins/tests/home_body_pin.rs`. Without it those eight numbers are untested data that four
  later slices will trust.
- **NEW: the bag's size pin**, beside `look.rs:504`: `assert_eq!(len, <measured>, "the widest
  self-look bag moved")` on a bag that carries the charter.

### 6.3 The drift gate

**One kernel word changes: `CellCharter.sea_radius`.** So `crates/bins/tests/gpu_no_drift.rs` and
`just terrain-legs` (`justfile:809`) cover 8b's kernel change already, because the sea moves cells and
the cells are folded into every digest.

**G-SEA, the gate the stored level owes (§3.3).** On every pinned body, the **stored** `sea_offset_mm`
must equal the offset the bisection **derives** from `water_km3` and the shape, on the CPU and on the
card. It is a measurement that can fail: a bisection that drifts by one step on one host moves the
coast, and the digest would catch it only where a cell changes.

**G-REFUSE.** A surface statement carrying no charter is refused and counted, and the counter goes up
— the observed-failing control, in the shape S7-3 already built.

### 6.4 Coverage

`just coverage-fast` (`justfile:30`) — Tier-A at 100 % region and branch, and `vd-terrain`,
`vd-recipe`, `vd-core`, `vd-sim` and `vd-client` are all in it. **Four branches 8b adds that the world
itself never drives**, and which therefore need a synthetic body in a unit test:

1. the relief law's **shape bound** arm (`0.077·R` under the strength bound) — it needs a body under
   939 km of radius, and the home system may hold none;
2. the **stripped** arm of the water model (the shoreline refuses, so `water_km3 = 0`);
3. the **runaway** arm (`S > 1.10 S⊕`);
4. the **no-sea** arm of §3.5 (the condensable is not liquid), and the `has_sea` flag clear.

Plus the three refusal branches: `no_charter` on the client, "no charter, no surface" on the shard, and
the bisection's coarse-then-fine escalation (§3.2).

`coverage-exemptions.toml` holds `(none yet)` and must still hold it when 8b lands.

### 6.5 The flights

`just flights` = `terrain-pictures` + `terrain-moving-eye` + `boarding-storm` (`justfile:320`), about
twenty-five minutes on a quiet machine with Docker off. **No stage lands before it is green.** 8b
halves the relief, so the residency band and the pop detector should get EASIER, not harder — and if
they do not, that is a finding.

---

## 7. THE STAGES

**The order's reason, in one line:** the charter must exist before any law can read it; the relief law
must land before the sea, because the sea solves against the shape the relief law sets; and the
version bumps once, at the end, with 8a's.

| # | The stage | The failing-first test | The gate |
|---|---|---|---|
| **1** | **The record and the lane.** `vd_core::look::BodyCharter`, `TAG_CHARTER = 6`, `charter_of_bag`, the encode in `surface_look_bag`, the `RealmBox.charter` field, the `no_charter` refusal and its counter. No producer yet; no number changes. | the widest bag's byte pin goes red at 124 and green at the measured number; a bag with no charter is refused and the counter reads one | `coverage-fast`, `terrain-pin` unchanged (no byte moves) |
| **2** | **The author.** The shard derives and quantises its own charter at boot from its subtree (§4.2) and states it. `home.rs` states the home body's charter; `home_body_pin.rs` cross-checks it against the forest (`charter_pin`). Still nothing reads it. | `charter_pin` goes red on a wrong integer; the home body's gravity reads 9 821 mm/s² | `charter_pin`, M-T on both target legs |
| **3** | **The relief law.** `from_seed` takes the charter; `relief_m = draw × min(σ_y/(ρ_c·g), 0.077·R)`. The band, the floor and `z` move. | a synthetic 400 km body takes the shape-bound arm; the home body's cap reads 8 840 m | `terrain-pin` re-recorded, M-R published, `terrain-legs` |
| **4** | **The remaining draws.** Spin under locking, the damped obliquity, `p_surf`, the two optical depths, the elastic thickness, the surface temperature. Carried and pinned; **no kernel reads them**. | the innermost planet of the home system reads `tidally_locked`; a moonless body's tilt is undamped | `charter_pin` widened; no byte of any chunk moves, so `terrain-pin` stays green — **and that is the assertion** |
| **5** | **The water inventory.** `f(r)` against the code's own frost line, the shoreline and the runaway verdicts, `water_km3`. **M-W runs and is published.** No sea yet. | a body over 1.10 S⊕ reads `water_km3 = 0`; the home body reads about 2.737e9 km³ | M-W published for the owner |
| **6** | **The sea by bisection.** The 4 056-direction ladder sample, the full-field evaluation, the fixed 24 steps, the escalation, `sea_offset_mm` stored and stated, the `has_sea` rule, and **`from_seed` moves off the message path on both hosts**. | the solved level puts the target volume over the shape to under 3 mm; a no-condensable body runs no bisection | `terrain-pin` re-recorded, **G-SEA**, M-S published, `flights` |
| **7** | **The version, the pins, the pictures.** `GENERATOR_VERSION` 3 → 4 **once, with 8a's**; `DECLARED_PIN` re-recorded; the five references and the far candidate re-frozen; the cave stand joins. | the world epoch opens; a store labelled with the old tag refuses to open | `just gate`, `just flights`, the owner's single look |

Each stage is under a day of agent work. **That judgement comes from the size of each diff and is not
a record of anything** (§9).

---

## 8. THE ASKS

Each is a question with a recommendation. **None of them is decided here.**

**★ ASK 1 — THE HOME PLANET DROWNS. What do you want done about it?**
The physical model gives the home planet about twice Earth's water (§2.4.4), and on a one-humped field
that is a **97.9–99.98 % ocean** (§3.6). The control shows the water model is not the culprit: **Earth's
own ocean on the same one-humped field is still 96 %.** The missing term is the crustal dichotomy —
isostasy on two crust types — and that is **8c's**, the very next item after 8s.
Three answers:
(a) accept it, judge 8a's pictures **with the sea off** (the dry shape), and judge the coast after 8c;
(b) accept it, judge the pictures **with the sea on**, and accept that the ground stand is a seascape;
(c) **pull 8c's two-crust isostasy forward**, ahead of 8s, so the sea has a basin to sit in.
**RECOMMENDATION: (c) if you want a coast in the pictures you judge once; otherwise (a).** (b) shows
you a picture whose shape you cannot judge. The arc must close a factor of five between the physical
inventory and what a one-humped planet can hold, and 8c is where that factor lives.

**★ ASK 2 — THE HOME PLANET IS COLD: 268 K with an Earth-like greenhouse (§2.5).**
At 0.748 S⊕ and `T_eq` 236.8 K, Earth's own greenhouse gives a mean surface temperature **five kelvin
below freezing**. Two answers:
(a) fix the greenhouse at Earth's `τ_ir`, and the home planet is a frozen world;
(b) implement the **carbonate–silicate thermostat** (Walker, Hays & Kasting 1981) — on a cold planet
CO₂ builds up until the surface warms; it is the law the habitable zone itself is built on, and it
gives **297.7 K** at `τ_ir = 2.0`.
**RECOMMENDATION: (b).** It is a law, not a tuning, and it makes the census's own earth-like predicate
self-consistent — every body that passes `KOPPARAPU_FLUX_CONSERVATIVE` then has a liquid surface by
construction. Note that (b) does **not** fix ask 1: a warm drowned world is still drowned.
The third answer, **(c) re-pick the home planet**, stays free until slice 14 (L27), and should be held
in reserve until ask 1 and ask 2 are both answered.

**ASK 3 — THE CRUST'S DENSITY: derived from the bulk, or a constant?**
The relief cap is `σ_y / (ρ_c · g)`, and the census holds the BULK density, never the crust's.
(a) `ρ_c = 2 800 kg/m³` for every body — the table `02` §4.3 actually uses;
(b) `ρ_c = ρ_bulk × 0.507 74`, the Earth ratio, so a denser body has a denser crust.
**RECOMMENDATION: (b).** `02` §4.3's own super-earth row says the constant **overstates** that body's
ceiling by exactly this term, and calls the number conservative on purpose. On the home planet the two
answers agree to four figures, so nothing visible changes today and the law is right for the next body.

**ASK 4 — THE OBLIQUITY'S PRIOR AND ITS DAMPING.**
Two model choices, each defensible.
(a) the prior: `cos ε` uniform over [0.2, 1.0] for nine bodies in ten and over [−1.0, 0.2] for the
tenth (`05` §3.3 A), so one world in ten is a Uranus-like curiosity — **this split is a taste, not a
physics result, and the code must say so**;
(b) the damping: `1/(1 + Λ)` with `Λ` the lunisolar torque ratio (Earth's is 2.18), against a hard
threshold at `Λ > 1`.
**RECOMMENDATION: the prior as (a) with the split stated as your number; the damping as `1/(1+Λ)`,
because a hard threshold makes two nearly identical moons give two different worlds.**

**ASK 5 — THE WATER RETENTION: a hard verdict or a ramp?**
(a) hard — retained or stripped, using the shoreline and the runaway edge the census already holds;
(b) a smooth retained fraction that falls across the line.
**RECOMMENDATION: (a).** It is what the census's own atmosphere draw does, it introduces no steepness
nobody has measured, and a body sitting exactly on the line is rare. The cost, named: two bodies a
hair apart across the line get very different worlds, and a player may one day find the pair.

**ASK 6 — THE SURFACE PRESSURE'S BAND (D-TAX-1).**
`p_surf` has no derivation in the code and the code says so (`taxonomy.rs:876-879`). `05` §3.3 C
recommends a log-uniform draw over **[1 000 Pa, 10 000 000 Pa]** where a secondary atmosphere is
retained — a band that holds Mars, Titan, Earth and Venus.
**RECOMMENDATION: take that band, and write in the code that it is a DRAW and not a derivation**, in
the same honest style as `GEOMETRIC_ALBEDO_BOUNDS` (`worldgen/body.rs:84-90`). Doing without it costs
8s its sky, because the aerial perspective reads the column.

**ASK 7 — THE CHARTER'S FIVE UNREAD WORDS.**
`day_s`, `obliquity_cos_q1024`, `tau_vis_q12`, `tau_ir_q12` and `elastic_thickness_m` are carried in 8b
and read by no kernel until 8c, 8s, 8e and 18. Carrying them now costs about 20 bytes on the bag and a
pin; **not** carrying them means appending to the record later, which is free before slice 14 and
impossible after it.
**RECOMMENDATION: carry them now, and give them `charter_pin`**, because §1.8 measures that no digest
can catch them and four later slices will trust them.

**ASK 8 — A2's LANE, UNSPENT.**
§4.2 measures that the planet's own shard already holds its orbit and its star, so the approved
crossing is not needed. **RECOMMENDATION: spend nothing. Keep A2's approval in hand** for the day a
body's facts stop being a pure function of the seed. This is the only ask where the recommendation is
to decline something you already granted, and it is SL6's own instruction.

---

## 9. UNMEASURED

Everything below is an argument, an estimate or a hand calculation. **None of it is a measurement, and
none of it may be quoted as one.**

1. **Every water volume, every sea level and every ocean share in §2.4.4 and §3.6.** They are hand
   arithmetic on the model of §2.4. M-W is what turns them into measurements.
2. **The home planet's facts other than the seed and the look radius.** `1.000 M⊕`, `9.82 m/s²`,
   `0.748 S⊕`, `236.8 K`, the N₂-like atmosphere with a 7 161 m scale height, the G star at
   `0.953 M☉` and `0.823 L☉`, nine planets and fifteen moons — **these appear only as prose in the doc
   comment at `crates/core/src/worldgen.rs:42-49`, and no test asserts them.** They are a record of one
   run. Only the seed and the radius bits are asserted (`home_body_pin.rs:10-11`), and only the
   insolation rung is pinned (`worldgen/tests.rs:3291`).
3. **The current home planet's field spread σ.** MEASURED at 2 297 m on the **old** 3 351 km home
   planet (`02` §1.4). The 2 640 m and 1 400–1 700 m figures in §3.6 are scalings by the relief, not
   readings.
4. **The relief falling to about 8 271 m.** It assumes the new draw's `[0.5, 1.0]` range maps onto the
   same stream position as today's `[0.5, 1.5)`. The stage prints the real number.
5. **The charter's 113 bytes and the bag's 237 bytes.** Postcard varint arithmetic by hand. The byte
   pin measures it.
6. **`a_lock = 0.3876 AU`, and that exactly one home-system planet is locked.** The `M★^(1/3) · t^(1/6)`
   scaling is standard; the Mercury calibration is mine, and the two sixth-root correction terms are
   dropped as near one.
7. **`τ_ir = 0.860` giving 268.2 K, and `τ_ir = 2.0` giving 297.7 K.** A grey-slab greenhouse
   calibrated on Earth. `τ_ir = 2.0` is a stand-in for the thermostat's answer, not the thermostat's
   answer.
8. **`T_e ≈ 36.8 km` and `α ≈ 86 km`.** The `T_e` law's rational decay is a modelling choice; the
   agreement of `α` with Earth's observed 50–130 km is a sanity check, not a validation.
9. **The sea level's 45 m and 600 m errors.** Derived from `02` §7.1's COMPUTED 0.78 % fraction error
   through a Gaussian density. The field is not Gaussian.
10. **The bisection's cost.** `02` §7.1's 2.13 ms was computed on the old body's octave count, before
    the ridge and the terrace. M-S measures the real figure.
11. **That `body_params` at a planet's boot is cheap (§4.2).** An argument from "one row, not a fold".
    M-B measures it.
12. **The seven stages each fitting under a day.** A judgement from the size of each diff.
13. **Whether the home planet has a moon at all**, and every number about the home system's largest and
    smallest planets. The masses are seed draws and no golden file holds them; `home_body_facts` is the
    one afternoon that answers it.
14. **The author's cross-target straddle.** Nobody has drawn the home system on two architectures and
    compared the integers. M-T is the only thing that answers it, and it is the one measurement that
    could refute the whole authoring rule.
