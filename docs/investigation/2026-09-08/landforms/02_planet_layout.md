# 02 — The planet-scale layout: where continents, ranges, rifts and coasts come from

**Date:** 2026-09-08. **Revision 3** — every finding of the revision-2 refutations is answered:
`verdicts/02_planet_layout_refutation_a.md` (R1–R24) and
`verdicts/02_planet_layout_refutation_b.md` (B2-1…B2-25). §17 holds the answer table, and it keeps
the revision-2 answers (F1–F30, B1–B4, B-D…, B-W…, B-N…) below them as history. Where a refuter is
right the section is REWRITTEN, not annotated. Where a refuter is wrong the section says so and
shows the evidence.
**Domain:** the macro map of one round body — continents, ocean basins, mountain ranges, rift
valleys, plateaus, island arcs, hotspot chains, the sea level, the ocean floor and the coast.
**Status:** an investigation report for the owner. It designs the layout. It decides nothing by itself.
**Binding law read first:** CLAUDE.md (HR1–HR6, SL1–SL10), the 2026-09-07 voxel ruling
(V1 SL10, V2.1–V2.9, V4, V6, V8, V9, V10, V11, V12), the 2026-08-27 seed-and-secrecy ruling
(S1–S6), the 2026-08-27 galaxy-shape ruling, and SL8 (a seam is a defect).

**The owner's words this report answers.** *"Make sure that we reach that quality on the picture for
earth-like planets (biomes can be different of course, should be dependent on the planet position,
spin, trajectory, size and gravity, etc.). It should be very believable, as we also should simulate
the weather."* And, on the three pictures of slice 7: *"no orienters and no details at all … The only
thing I'm worried about is that the surface will not be interesting enough."*

Every number below is marked **MEASURED** (with how), **COMPUTED** (a Python replication of the
crate's own arithmetic, validated against a pin the crate holds), or **ESTIMATED**. Every claim about
the code cites `file:line`, re-checked in revision 3. The investigation base is not binding; where it
disagrees with a ruling, the ruling wins.

---

## Summary — the recommendation in one page

**The surface is not uninteresting by accident. It is uninteresting by arithmetic, and there are TWO
arithmetic causes.** Revision 3 adds a third finding, about the ladder itself, that nobody was
looking for.

> **Cause 1 — one slope at every scale.** The height field is one stationary sum of gradient-noise
> octaves whose amplitude ratio is `k_rough ∈ [0.45, 0.55)` (`crates/terrain/src/body.rs:155`). The
> home planet drew `k_rough = 0.468 338 1`. **The root-mean-square slope of the home planet is 1.6°
> to 2.0°, and it is the SAME at every baseline from 1 m to 8 km. The median height change inside a
> 50 m window is 1.14 m; inside a 1 km window it is 26.4 m.** (COMPUTED, §1.2.)
>
> **Cause 2 — a hole in the spectrum from 1 m to 49 m.** `SHORT_WAVE_M = 30`
> (`crates/terrain/src/body.rs:25`) stops the octave table at the first wavelength under 30 m, which
> on the home planet is **48.83 m**. The cell is 1 m. **The height field carries NO content at all
> between 1 m and 48.8 m** (COMPUTED, §1.6).
>
> **Cause 3, new in revision 3 — the ladder's own promise is not kept today.** The crate drops one
> octave per rung and the property test at `crates/terrain/src/height.rs:83-110` bounds the
> disagreement by the dropped amplitudes. That bound is **larger than half a cell at six of the
> twelve rungs** on the home planet: 32.8 m against 32 m at rung 6, and 1 469 m against 1 024 m at
> rung 11 (COMPUTED, §6.2). The far view therefore moves the surface by more than the rung can
> show, which is an SL8 step. *Refuter B found this and it was against the report's own premise
> (B2-7).*

**The recommendation is option (d), a closed-form layer stack, and it survives a second refutation —
but three of its parts are REBUILT.** Revision 2 deleted a downhill march and replaced it with a
gradient-driven valley transfer and a river-network lookup. Both refuters showed that the river
lookup hides a whole-planet pass (R2, B2-1), and refuter A showed that the gradient it uses cannot
resolve a valley at all (R13). **They are right.** §4.4 replaces the gradient transfer with a
**multifractal weight** — one multiply per octave, no gradient, and it flattens floors and sharpens
crests at every scale — and **hands the river network to domain 03**, with the three conditions any
river design must meet written down here.

```
        WHAT THE PLAYER SEES                  WHICH LAYER MAKES IT          COST/COLUMN
  ┌──────────────────────────────────────┬──────────────────────────────┬─────────────┐
  │ a continent, an ocean basin          │ L1 plates (spherical Voronoi)│    70 ns    │
  │ a coast, a shelf, an abyssal plain   │ L2 isostasy (a crust FIELD)  │    64 ns    │
  │ a range, a trench, a rift, an arc    │ L3 orogeny (boundary field)  │    98 ns    │
  │ a valley floor, a ridge crest        │ L4a the multifractal weight  │    35 ns    │
  │ a bench, a mesa rim, a cuesta        │ L4e differential erosion     │     5 ns    │
  │ a snow cap, a dune field, a beach    │ L5 climate operators         │    30 ns    │
  │ a forest belt, a rain shadow         │ L7 the biome, rebuilt        │    20 ns    │
  │ the rock texture under the boots     │ L6 ONE extra octave          │    13 ns    │
  │ a river, a lake, a flood plain       │ DOMAIN 03 — not this report  │      —      │
  └──────────────────────────────────────┴──────────────────────────────┴─────────────┘
   TOTAL ADDED: 335 ns per column, against an allowance of 1 153 ns (COMPUTED: the
   4.7 ms of headroom under the 8 ms budget, divided by the 4 075 columns of a chunk's
   SAMPLE BOX — the halo included, which is six per cent more columns than the bare
   chunk, MEASURED at slice_06_extractor.md:355 — on a COLD chunk, nothing amortised).
```

Four changes carry most of the picture:

| # | The change | What it buys | Cost |
|---|---|---|---|
| 1 | **The octave table is re-anchored and given a BUDGET**: the coarsest wavelength drops from 400 km to ~50 km (the macro layers own everything above), `SHORT_WAVE_M` drops from 30 m to **2 m**, the sum stops carrying the whole relief and carries **2 053 m**, and every amplitude is capped by the ladder's own half-cell budget | The spectrum reaches the cell AND the slope grows as you look closer, which is what real land does: **RMS slope 18.9° at a 1 m baseline, 9.3° at 32 m, 3.8° at 512 m, 1.3° at 8 km**, and 6.13 m of relief inside a 50 m window against today's 1.14 m (COMPUTED, §4.6). The ladder cap makes the half-cell promise TRUE at every rung for the first time (§6.2). | +13 ns |
| 2 | **Airy isostasy on a crust FIELD, with the water load** (L2), not a per-plate label | A **bimodal** hypsometry, and a **passive margin**, because a continent and an ocean floor share a plate. With the water column included the ocean floor sits **6 476 m** below the continental surface, not 4 455 m (COMPUTED, §4.2). *Both refuters found the missing water load (R11, B2-9) and they are right.* | 64 ns |
| 3 | **The mountain ceiling `h_max = σ_y/(ρ_c·g)`** replaces `0.4 % of the radius, capped at 12 km` | Everest fixes `σ_y = 243 MPa`. The relation then gives **Mars 23 401 m against Olympus Mons's 21 900 m (94 %)** and Venus 9 788 m against Maxwell Montes's 11 000 m (112 %) — a **scale with a ±15 % spread**, never a wall (COMPUTED, §4.3). *Revision 2's Mars row used a density the table said it did not use; both refuters caught it (R5, B2-8).* | free |
| 4 | **A per-column WATER SURFACE** replaces the one sea sphere | `fluid_at` reads one number per body today (`crates/terrain/src/chunk.rs:207-209`), so water exists below one sphere and nowhere else. The field must exist before any river can live in it, and domain 03 needs it. | ~0 |

**What this report now says plainly about rivers.** L4a makes valley SHAPES at every scale and makes
no drainage. **The first landing has ridges, valleys, cliffs, benches and a coast, and it has no
river.** The reference picture has a river. *Refuter B asked for that sentence (B2-18) and it is
owed to the owner.*

**Two SL6 asks.** §5.4 states the layout's ask — a `BodyFacts` record, now **thirteen integers,
52 bytes of payload**, quantised to integer grids and **authored once by the body's own realm**, never
re-derived anywhere else. *Revision 2 carried seven integers, and refuter B showed four layers could
not be computed from them (B2-5).* §5.5 opens the weather ask and hands it to domain 05.

**The law hole, and the structural cure.** The body's physical facts are floats computed by
`vd-physics`, whose arithmetic is outside the fence (MEASURED 2026-09-08: **43 `powf`/`ln`/`exp`/
`sqrt`/`powi` calls on 39 lines of `taxonomy.rs`, and 11 calls on 9 lines of `celestial.rs`**).
Quantising to an integer grid does not remove a drift; it makes a drift harmless everywhere except
at a grid boundary. **So the integer is AUTHORED ONCE, by the realm that owns the body, and every
other host receives the integer and never re-derives it.** *Refuter A and refuter B found the
over-claim independently (R8, B2-10), and refuter B stated the cure; revision 3 adopts it.*

---
## 0. What the code holds today (the only current truth)

| Fact | Where | What it means for the layout |
|---|---|---|
| The height field is `radius + Σ octaves`, and nothing else. There is no continent, no plate, no range, no river and no coast anywhere in the crate. | `crates/terrain/src/height.rs:17-28` | Every landform in this report is new code. Nothing is retrofitted. |
| The relief is `0.4 % of the radius, clamped to [200, 12 000] m, times a seed factor in [0.5, 1.5)`. | `crates/terrain/src/body.rs:147-149` | Three magic numbers (0.004, 200, 12 000) decide how tall every mountain in the world may be. None comes from a physical fact of the body. |
| The coarsest wavelength is a quarter to a half of the radius, clamped to `[20 km, 400 km]`. | `crates/terrain/src/body.rs:151-153` | The home planet hits the 400 km ceiling. Its largest landform is therefore 400 km wide — smaller than a continent by ten times. |
| **The finest wavelength is `SHORT_WAVE_M = 30 m`**, and halving from 400 km reaches **48.83 m** before it stops. | `crates/terrain/src/body.rs:25,169` | **Nothing in the height field lives between 1 m and 48.8 m.** §1.6. |
| The roughness `k_rough` is one number per body in `[0.45, 0.55)`, applied to every octave everywhere on the body. | `crates/terrain/src/body.rs:155` | The roughness cannot differ between a flood plain and a range. One number rules the whole sphere. |
| **The amplitudes are re-normalised so their SUM is the relief** (`amp[o] = weight[o] · relief / weight_sum`). | `crates/terrain/src/body.rs:176-184` | **Shortening the coarsest wavelength without changing this rule makes the whole world eight times steeper.** That is the trap revision 2 walked into; §4.6 states the budget that closes it. *Both refuters found it (R1, B2-4).* |
| The sea radius is `ladder radius + floor((u·0.7 − 0.4)·relief)`, with `u` uniform in `[0, 1)`. | `crates/terrain/src/body.rs:188-190` | The sea level is drawn against the relief BOUND (a worst case), not against the height DISTRIBUTION. §1.4 measures what that costs. |
| **THE FLUID is one rule for the whole body**: `if r < body.sea_radius_m { Water } else { Air }`. | `crates/terrain/src/chunk.rs:207-209` | **Water exists below one sphere and nowhere else.** A river, a lake and a flooded valley floor have no field to live in. §4.4c. |
| **The bedrock is ONE draw per BODY**, and `StrataTable::at` takes a biome and a depth — no direction. | `crates/terrain/src/body.rs:200`; `crates/terrain/src/strata.rs:177,189` | The strata are HORIZONTAL and they already exist. §4.4e uses that for benched cliffs WITHOUT a per-direction rock map; §10.3 and D10 keep the ore question separate. |
| **The band is DERIVED from the relief**: `crust = relief_whole + strata + caves + 64`, `above = relief_whole + 64`, and `Ladder::for_radius` turns those into `floor_m` and `band_m`. | `crates/terrain/src/body.rs:233-236`; `crates/seed/src/ladder.rs:88-98` | **Changing the relief moves `floor_m`, and `floor_m` is the base of every cell address on the body.** §9.3 prices it. |
| The biome reads latitude, height above the sea, and two slow noises (temperature and humidity). There are four biomes. | `crates/terrain/src/height.rs:40-66`; `crates/terrain/src/strata.rs:114-131` | There is no wind, no rain shadow, no lapse rate and no ocean nearby. The owner's "biomes … dependent on the planet position, spin, trajectory, size and gravity" is not served by two noises. §4.9 replaces it. |
| `POLE_AXIS` is `+Z`, the orbit's own axis. A per-body obliquity is named as "a later slice". | `crates/terrain/src/height.rs:30-35` | The world has no obliquity today. Without one there are no seasons, and the ice caps sit exactly on the orbital poles. |
| **The world has no spin.** MEASURED 2026-09-08: the grep for `obliquity`, `axial_tilt`, `spin_rate`, `rotation_period`, `day_length` and `sidereal` over `crates/**/*.rs` returns **TWO** hits — `crates/terrain/src/height.rs:32` (a comment) and `crates/physics/tests/celestial_ephemeris_pin.rs:29` (a test constant). | MEASURED, re-run in revision 3 | No Coriolis parameter, no day length, no prevailing wind. Weather has no input to stand on. §5 asks for the draw. |
| The forest DOES hold a body's mass, radius, insolation, equilibrium temperature, Bond albedo and an atmosphere record (with its **mean molecular weight**). | `crates/physics/src/taxonomy.rs:895-910`, `:881-890` | Most of the physical facts the layout needs already exist. They live in a motion crate the generator may not name (§5.4). |
| **`vd-physics` computes with the arithmetic the fence exists to exclude.** MEASURED by grep 2026-09-08, with the counting rule stated: `grep -oE "\.powf\(\|\.ln\(\|\.exp\(\|\.sqrt\(\|\.powi\("` gives **43 calls on 39 lines** in `taxonomy.rs` and **11 calls on 9 lines** in `celestial.rs`. `tests/tests/crate_isolation.rs:325-331` refuses `libm` in `vd-terrain` and `vd-seed` by name. | MEASURED (re-run; *revision 2 said "39 and 22" and the 22 reproduces under no rule — refuter A was right, R7*) | A fact handed in from the forest is an unfenced float. §5.4 makes the realm author it once. |
| **The crate forbids exactly that, in writing.** *"a body is DRAWN from a seed … and never assembled from numbers computed elsewhere, so no float from an unfenced crate can enter the recipe as a body."* | `crates/terrain/src/body.rs:84-87` (the struct doc) | The authored integer of §5.4 is what makes an incoming fact lawful under that sentence. |
| Surface gravity is one line from mass and radius, and the code says so on purpose. | `crates/physics/src/taxonomy.rs:750-752` | `g` is available wherever mass and radius are. It is the single most useful number the layout does not have. |
| The generator is fenced. `Gf` offers `+ − × ÷ sqrt floor trunc abs`, comparison, **and also `lesser`, `greater`, `clamp` and `is_finite`**. What is absent is `f64::min`/`max`, `mul_add`, `powi` and every transcendental. | `crates/terrain/src/gf.rs:10-14` (the doc), `:102,117,123,129,135,148` (the methods) | Every formula in §4 is written in these operations, and §10.1 checks each one. |
| The noise is gradient noise on `SplitMix64` with a 16-entry gradient table and the quintic fade, and it is PINNED by a known vector. | `crates/terrain/src/noise.rs:19-36,86-120`; the assertion at `:231`, the constant `NOISE_PIN` at `:235` | A ridged or warped field reuses this. No new hash, no new library. §1.1 validates this report's replication against that pin. *The citation is corrected: revision 2 said `:228` (R24).* |
| **The generated cell is `Cell { stratum: Stratum, gap: i8 }` — TWO bytes.** The **twelve-byte record** of ruling V6 part B (`owner_decisions_2026-09-07_voxels.md:186-187`) is the STORED record, and slice 9 has not built it. | `crates/terrain/src/chunk.rs:64-69` | When this report says "the record does not change" it means **the stored twelve-byte record**, and §4.4c says which of the two it checks. *Refuter A was right that revision 2 blurred them (R22).* |
| The height field takes a **unit direction**, never a face parameter. | `crates/terrain/src/height.rs:17` | **The cube-face seam is already solved by construction.** Every new layer must obey the same rule, and so must every accelerator (§4.0, §6.3). |
| The ladder: the home planet is `N = 5 263 360` cells per face edge at rung 0, 12 rungs, top rung 2 048 m. The largest radius the ladder accepts is `2·N_MAX/π = 42 723 km`. | `crates/seed/src/ladder.rs:25,66-105`; COMPUTED, and the crate MEASURED the same (ruling V9) | Jupiter (69 911 km) and Saturn (58 232 km) are **already refused**; Neptune (24 622 km) and Uranus are accepted. §4.7 and D13. |
| MEASURED cost, release, home planet: the rung-0 column pass is 710 µs per 62×62 columns with 14 octaves; the top rung is 266 µs with 3 octaves. A surface chunk is 3.3 ms; a cave-dense chunk 6.1 ms. The budget is 8 ms of worker time. | `docs/investigation/2026-09-07/slice_05_generator.md:372-386`; `slice_06_extractor.md:363-365`; the budget is ruling V10 | 185 ns per column, 13 ns per octave. §9 prices every new layer against these numbers. |
| **The sample box carries a halo: six per cent more columns than the bare chunk.** | MEASURED, `slice_06_extractor.md:355` | The budget's per-column allowance divides by **4 075** columns, not 3 844. *Refuter B was right (B2-6).* |
| MEASURED on the CLIENT: 4.18 ms per chunk on one thread, 239 chunks/s, 2 004 chunks/s on 14 threads (8.4×). | `docs/investigation/2026-09-07/slice_07_client_link.md` (M7-2, `:235`) | The client pays the same column pass. §9.4 prices it there too. |
| A radial column holds 470 chunks at rung 0 (`cells_in_band(0) = 29 105`); the column extremes skip 463 of them without a cell pass; the whole column costs 417 ms. | `slice_05_generator.md:384-386` | The amortisation is real, and it is **not** the case the 8 ms budget gates. §9.2 uses the COLD chunk. |
| `TAG_SURFACE` exists and carries `SurfaceStmt { frame, generator }`. The self-look bag's budget is 1 200 bytes. | `crates/core/src/look.rs:56,69-75,79` | The carrier for the body's physical facts already exists and has room (§5.4). |
| The `DEFERRED` register holds `D-TERRAIN-1..4`. **`D-TERRAIN-3` is "ONE RUNG PER REALM"**, it is RED, and slice 8 deletes it. | `docs/design/DEFERRED.md:7721,7752,7775,7794` | Nothing in this report may add a flagged interim; the owner refused that shape by name in ruling V12. *The name is corrected: revision 2 wrote "one rung per session" (R24).* |

---
## 1. The measured diagnosis: why the picture is not the picture

### 1.1 The method, and how it is validated

I re-implemented `SplitMix64`, `child_seed`, `corner_hash`, the 16-entry gradient table, the quintic
fade, `noise3`, the ladder snap and `BodyDefinition::from_seed` in Python, line by line, from
`crates/seed/src/rng.rs:15-74`, `crates/terrain/src/noise.rs:44-120`, `crates/seed/src/ladder.rs:60-105`
and `crates/terrain/src/body.rs:139-247`. The replication is checked against **the crate's own noise
pin**:

```
  crates/terrain/src/noise.rs:231   assert_eq!(noise3(2298, p(0.25, 0.5, 0.75)).to_bits(), NOISE_PIN)
  crates/terrain/src/noise.rs:235   NOISE_PIN = 13 827 097 110 060 728 320

  this report's replication          noise3(2298, [0.25, 0.5, 0.75]).to_bits()
                                    = 13 827 097 110 060 728 320   ✓ EQUAL, bit for bit
```

That check exercises the hash, the gradient table, the fade, the eight-corner blend and the blend
ORDER — everything the slope numbers rest on. **In revision 3 every table of §1, §4.6 and §6.2 was
re-run on this replication**, so the numbers below are one method, one program, one run. *M1 still
moves the whole of §1 into the crate as a bench, because a Python replication is evidence and a
crate measurement is the gate.*

The home planet, COMPUTED: seed `7 701 581 858 760 374 086`; ladder radius 3 350 759.045 m; 12 rungs;
`N = 5 263 360`; relief 14 304.887 m (`relief_whole`, the integer the band uses, is 14 305);
`long_wave = 400 000 m` (at the cap); `k_rough = 0.468 338 1`; 14 octaves; sea offset −5 297 m.

| Octave | Wavelength | Amplitude | Amplitude ÷ wavelength |
|---|---|---|---|
| 0 | 400 000 m | 7 605.55 m | 0.0190 |
| 1 | 200 000 m | 3 561.97 m | 0.0178 |
| 2 | 100 000 m | 1 668.21 m | 0.0167 |
| 3 | **50 000 m** | **781.28 m** | 0.0156 |
| 5 | 12 500 m | 171.37 m | 0.0137 |
| 9 | 781.25 m | 8.24 m | 0.0106 |
| 13 | **48.83 m** | **0.397 m** | 0.0081 |

**Octave 3 is printed in bold because §4.6's whole budget rule hangs on it:** it is the octave whose
wavelength the re-anchored table starts at, and the rule is that it keeps the amplitude it has here.

### 1.2 The finding: one slope, at every scale, everywhere

The ratio in the last column is what matters. It falls by only 2.3 times over 14 octaves, so **every
scale contributes about the same slope**.

> **Hurst exponent, in plain words.** It says how much rougher a surface gets as you look closer.
> A value near 0 is a jagged mess at every zoom. A value near 1 is a smooth curved sheet. Real land
> sits near 0.5–0.7: a hillside seen from a kilometre away is smooth, and the same hillside seen from
> five metres away is full of ledges and boulders. The exponent is `−log₂(k_rough)`, and it is
> defined on `0 < H < 1`. The home planet's `k_rough = 0.4683` gives **1.094**, which is OUTSIDE the
> definition: the field is not fractional Brownian motion at all, it is a sum whose amplitudes fall
> FASTER than any Hurst surface. The honest statement is *"the surface is smoother than the smoothest
> fractal land, and the amplitude-to-wavelength ratio proves it"*. *Refuter B was right to object to
> the term's use past its range (B-N1); the number and the conclusion stand.*
>
> *Game example: the pilot lands on the home planet and walks. The 49 m ripple under the boots is
> 0.40 m tall. There is nothing to step over, nothing to climb, and nothing to hide behind.*

COMPUTED, ONE sampling method for every slope table in this report — 400 directions per row drawn
from a fixed-seed Gaussian and normalised, a random tangent bearing per sample, the step taken along
a great circle, gradient noise exactly as the crate computes it:

| Baseline | RMS slope | Median slope | p95 slope |
|---|---|---|---|
| **1 m** | **1.93°** | 1.22° | 3.86° |
| 2 m | 1.97° | 1.30° | 3.98° |
| 8 m | 1.93° | 1.26° | 3.86° |
| 32 m | 1.92° | 1.32° | 3.86° |
| 128 m | 1.89° | 1.26° | 3.85° |
| 512 m | 1.87° | 1.25° | 3.70° |
| 2 048 m | 1.75° | 1.23° | 3.33° |
| 8 192 m | 1.57° | 0.98° | 3.34° |

```
   what the reference picture holds          what the home planet holds
        (Crimson Desert vista)                    (COMPUTED, §1.2)

   90° ┤ ███ cliff, rock pillar               90° ┤
   60° ┤ ███ scree, valley wall               60° ┤
   45° ┤ ████ ridge flank                     45° ┤
   30° ┤ █████ forested slope                 30° ┤
   15° ┤ ███████ fields, terraces             15° ┤
    5° ┤ ████████ river plain                  5° ┤ ███████████████ everything
    0° ┼──────────────────────────             0° ┼──────────────────────────
```

And the local relief — the height range inside a window on the ground, which is what a player calls
"detail". COMPUTED, 60 lines per row, 21 samples along each line, the same replication:

| Window | Median relief | p95 relief |
|---|---|---|
| 50 m | **1.14 m** | 2.63 m |
| 200 m | 5.00 m | 13.35 m |
| 1 000 m | **26.36 m** | 66.09 m |
| 5 000 m | 109.23 m | 254.20 m |
| 20 000 m | 377.19 m | 1 020.59 m |

**One metre of height change inside fifty metres.** That is the owner's "no details at all", as a
number. The cells are one metre. The extractor resolves them. There is simply nothing in the field
for the extractor to find.

### 1.3 What the roughness alone buys, ON THE TABLE THE REPORT RECOMMENDS

*Revision 2 swept `k_rough` on TODAY's table (400 km down to 48.8 m) and then recommended a different
table. Refuter A was right that this validates a recommendation against a table the report also
deletes (R1, third consequence). The sweep below is re-run on the RE-ANCHORED table of §4.6, under
its budget rule, so the evidence and the recommendation are about the same object.*

COMPUTED: the coarsest wavelength is 50 km, the floor is 2 m, so the table holds **15 octaves**; the
coarsest amplitude is held at today's 781.28 m (the budget rule of §4.6), and `A_oct` — the octave
sum's total, which is what the old code called the relief — falls out of the roughness:

| `k_rough` | `A_oct` | Finest amplitude at 3.05 m | RMS @1 m | @32 m | @512 m | Relief in 50 m | Relief in 1 km |
|---|---|---|---|---|---|---|---|
| 0.4683 (today's draw) | 1 469 m | 0.019 m | 1.72° | 1.65° | 1.42° | 0.78 m | 19.8 m |
| 0.50 | 1 563 m | 0.048 m | 2.47° | 2.21° | 1.69° | 1.05 m | 23.5 m |
| 0.55 | 1 736 m | 0.181 m | 5.33° | 3.88° | 2.33° | 1.95 m | 34.6 m |
| 0.58 | 1 860 m | 0.381 m | 9.12° | 5.66° | 2.86° | 3.43 m | 43.5 m |
| 0.60 | 1 952 m | 0.612 m | 13.16° | 7.35° | 3.31° | 4.72 m | 51.4 m |
| **0.62** | **2 054 m** | **0.969 m** | **18.87°** | **9.57°** | **3.84°** | **6.05 m** | **61.5 m** |
| 0.65 | 2 229 m | 1.878 m | 31.10° | 14.23° | 4.84° | 9.42 m | 83.4 m |

**Read the columns across, not down.** At `k = 0.62` the slope FALLS as the baseline grows —
18.9° at a metre, 9.6° at thirty, 3.8° at five hundred. That is what real land does and what today's
world does not do at all (§1.2's column is flat). **`k ≈ 0.62` is the recommendation, and it is a
draw range, not a constant** (§4.6).

**Two things the table does not say, and both matter.**

1. `k = 0.65` is already too much: a 1.88 m bump at a 3 m wavelength is a wall under every boot.
   The window between "no detail" and "unwalkable" is about `[0.58, 0.63]`, and **M12 owes the
   picture that picks the point inside it.**
2. This is the ROUGHEST state. A global 0.62 makes the sea floor, the flood plain and the desert pan
   equally rough. **The modulation of §4.6 only ever scales DOWN from this table**, which is what
   keeps the band and the ladder bound safe.

### 1.3b ★ The composition refuter A demanded, and what it shows

*R1 is the blocker of the revision-2 refutation and it is correct as a charge: revision 2 recommended
two changes, priced each alone, and never evaluated them together. The refuter did, on the shipped
amplitude rule, and got 56° RMS at a 2 m baseline — "a vertical wall under every boot". The
arithmetic is right. **The cause is not the pair of changes; it is the amplitude rule the pair was
composed under.** `body.rs:176-184` re-normalises every amplitude so the sum is the relief, so
shortening the coarsest wavelength by eight times multiplies every amplitude-to-wavelength ratio by
eight. That rule is what revision 2 failed to state, and §4.6 now replaces it.*

COMPUTED, all four corners, one program, one run:

| Table | Coarsest amp ÷ wavelength | Finest octave | RMS @1 m | @32 m | @512 m |
|---|---|---|---|---|---|
| today (400 km, 30 m, `k` 0.4683, sum = relief) | 0.0190 | 48.83 m, 0.397 m | 1.93° | 1.92° | 1.87° |
| re-anchored, **sum still the relief** (R1's case) | **0.1521** | 3.05 m, 0.186 m | ~15.7° | 14.9° | 12.8° |
| re-anchored, sum still the relief, **and `k` 0.62** | 0.1088 | 3.05 m, 6.75 m | **> 50°** | — | — |
| **re-anchored under §4.6's BUDGET, `k` 0.62** | **0.0156** | 3.05 m, **0.969 m** | **18.87°** | **9.57°** | **3.84°** |

**So refuter A's second consequence — "the two changes do the same job twice" — does not hold under
the budget rule, and its first and third consequences do hold and are fixed.** The re-anchor moves
the SPECTRUM (it deletes three coarse octaves the macro layers now own, and adds two fine ones the
cell can see); the roughness moves the SLOPE (it decides how the fixed budget is spread across the
table). Under the budget rule they are orthogonal, and the last row is the recommendation.

### 1.3c ★ The third cause: the ladder's own promise is broken today

`BodyDefinition::octaves_at` drops one octave per rung (`body.rs:249-256`) and `relief_bound_m`
bounds the disagreement between two rungs by the dropped amplitudes (`body.rs:258-266`), which the
property test at `height.rs:83-110` checks. **Nobody ever checked that bound against the CELL at
that rung**, and refuter B did (B2-7). COMPUTED on the home planet:

```
   rung:        0     1     2     3     4     5     6     7     8     9    10    11
   half cell: 0.5   1.0   2.0   4.0   8.0    16    32    64   128   256   512  1024   m
   dropped:   0.0   0.4   1.2   3.1   6.9    15  32.8  70.4 150.6 322.0 687.9 1469.2  m
   verdict:    ok    ok    ok    ok    ok    ok  OVER  OVER  OVER  OVER  OVER  OVER
                                                 1.02x 1.10x 1.18x 1.26x 1.34x 1.43x
```

**Six of twelve rungs move the surface by more than the rung's own cell can show.** The reason is
exact and it is a one-line law: the dropped amplitude grows as `(1/k)^L` and the cell grows as `2^L`,
so the ratio grows as `(1/(2k))^L`. **A body is safe at coarse rungs only if `k ≥ 0.5`, and the home
planet drew 0.4683.** Every body in `[0.45, 0.50)` — half the draw range — fails the same way.

Two consequences, and the second is the good news:

- the SL8 claim of "a coarse rung is the same hill without the small bumps" is not established today,
  and slice 8's tier rule inherits the error;
- **§4.6's budget cures it, and §6.2 makes it exact by construction with one clamp per octave.**
### 1.4 The sea level is a lottery, and the hypsometry is the wrong shape

COMPUTED over 4 000 random directions on the home planet:

```
  height above the ladder radius        the drawn sea level is here
        (COMPUTED, 4 000 samples)                 ↓
   -7159 ────────────────────────────────────┬───────────────────────────── +7341
                                        p01  │ p05      p25   p50   p75  p95
                                      -5166  │-3725   -1586    -7  +1589 +3799
                                             -5297
                              0.9 % of the surface is under water
```

- mean **+12 m**, σ **2 297 m**, min −7 159 m, max +7 341 m;
- **σ ÷ relief = 0.161.** *(Refuter A's independent replication gave σ = 2 345 m and 0.164 — a 2 %
  spread from a different sample, which is agreement, not disagreement.)*

The sea offset is drawn uniformly in `[−0.4, +0.3] × relief` (`body.rs:188-189`). Divided by σ, that
is a uniform draw in **[−2.49 σ, +1.87 σ]**. So the ocean fraction of a body is a lottery whose
outcomes run from "no sea at all" to "no land at all", and the home planet drew −2.31 σ.

Two separate defects sit here, and they need separate fixes:

1. **The level is drawn against a worst case, not against the distribution.** Fix: solve for a water
   inventory (§7.1).
2. **The distribution itself is the wrong SHAPE.** A sum of noises is approximately Gaussian, so the
   most common elevation is the mean and the tails are rare and symmetric. Earth's hypsometry is
   **bimodal**: two peaks, one at the continental platform near sea level and one at the abyssal plain
   near −4 km, with a steep continental slope between them.

> **Hypsometry, in plain words.** The hypsometric curve says how much of a planet's surface stands at
> each height. It is the planet's height histogram. Earth's has two humps because Earth has two kinds
> of crust. *Game example: the pilot flies from the ocean toward the coast. On Earth the sea floor
> stays near −4 km for a thousand kilometres, then rises over a shelf, then meets a beach. On the home
> planet today the floor rises the whole way at 2°, so there is no shelf, no beach and no coast — only
> a waterline drawn across a smooth slope.*

```
   Earth (bimodal)                      the home planet today (Gaussian)
   area →                               area →
  ┌──────────────────┐                 ┌──────────────────┐
+5│ ▏  continents     │              +7│ ▏                 │
 0│ ████▏ shelf ──────│ sea level     0│ ██████▏           │ ← sea would sit here
-4│ ████████▏ abyss   │              -5│ █▏                │ ← sea actually sits here
-8│ ▏ trench          │              -7│ ▏                 │
  └──────────────────┘                 └──────────────────┘
   TWO humps, a steep step between      ONE hump, no step anywhere
```

Isostasy (§4.2) produces the two humps from two densities and two thicknesses. It is the single
cheapest change with the largest visible effect.

### 1.5 The vista, and what "no orienters" really measures

COMPUTED, from an eye 1.8 m above the surface, over 60 random stances, sampling a 50 km ray every
250 m and subtracting the curvature drop `s²/2R`:

- the geometric horizon on a smooth sphere is **3 473 m**;
- the curvature drop is 15 m at 10 km, 60 m at 20 km, **373 m at 50 km**;
- the furthest point still standing above the eye line: **median 31 750 m**, p90 50 000 m.

So the range is not the problem — the home planet's swells DO reach past 30 km. **The problem is that
what stands at 30 km is a 2° swell, not a mountain.** A landmark must be recognisable, and a shape
with one slope everywhere is not recognisable at any distance.

The physical requirement follows. A feature of height `h` is visible over the horizon at
`√(2·R·h) + √(2·R·h_eye)` — **the observer's own horizon adds to the feature's**. COMPUTED for the
home planet, with a 1.8 m eye:

| Feature height | The feature's own horizon | Visible from, with a 1.8 m eye |
|---|---|---|
| 1.8 m (a standing figure) | 3 473 m | **6 946 m** |
| 100 m (a butte) | 25 887 m | **29 360 m** |
| 1 000 m (a range crest) | 81 863 m | **85 336 m** |
| 2 000 m (a snow-capped peak) | 115 771 m | **119 245 m** |

*Revision 1 gave only the first column. Both refuters caught it (F27, N3). The correction makes the
world slightly MORE generous than revision 1 claimed, so the conclusion is unaffected.*

**To hold the reference picture's 50 km of readable distance, the world needs relief of roughly 1 to
2 km CONCENTRATED IN RANGES, not spread as a swell.** Concentration is exactly what convergent plate
boundaries do and what a stationary noise sum cannot do at any roughness.

### 1.6 ★ The second cause: a hole in the spectrum from 1 m to 49 m

`crates/terrain/src/body.rs:25` sets `SHORT_WAVE_M = 30`, and the octave loop at `:169` halves the
wavelength **while it is over 30 m**. On the home planet the coarsest wavelength is 400 km, so the
finest octave lands at `400 000 ÷ 2¹³ = 48.83 m` (COMPUTED, and it matches the crate's MEASURED
14 octaves).

```
   wavelength:  400 km ── 200 ── 100 ── … ── 195 m ── 97.7 m ── 48.83 m │ ▓▓▓▓▓▓▓▓▓▓ │ 1 m cell
                                                                        │  NOTHING   │
                 the octave table stops here ───────────────────────────┘  AT ALL    └── the extractor
                                                                            resolves this
```

**The height field carries no content between 1 m and 48.8 m.** The consequences:

- raising `k_rough` alone cannot fill the hole — it makes the 49 m ramp steeper and leaves the metre
  scale empty;
- the extractor's triangles between 1 m and 49 m are an interpolation of a smooth ramp, so the
  measured snap error of slice 6 is measuring a ramp, not a landform;
- the owner's *"no details at all"* is answered directly by filling this hole.

**The cure is one constant and one re-anchor** (§4.6): `SHORT_WAVE_M` drops from 30 m to **2 m**, and
the coarsest wavelength drops from 400 km to ~50 km, because L1–L3 own everything above that.
COMPUTED: the table then holds **15 octaves** with a finest wavelength of 3.05 m, and the top rung
keeps `15 − 11 = 4` octaves, so the clamp arm at `body.rs:253-255` still never fires on a 12-rung
body.

> **Nyquist, in plain words.** To draw a wave you need at least two samples inside one wavelength;
> with fewer, the wave comes out as a different, wrong wave. *Game example: the pilot's chunk has
> one-metre cells. A 1.5 m ripple would be sampled 1.5 times per wave, so the boots would walk on a
> pattern the field never contained. A 3 m ripple is sampled three times and comes out as itself.*
> That is why §4.6's floor is 2 m (finest wavelength 3.05 m) and not 1 m.

*Two corrections of arithmetic, both from refuter A (R24), both accepted.* (1) Revision 2 wrote the
compile-time assertion as `50 000 / 65 536 = 0.76 < 2 ✓`. `LONG_WAVE_CAP_M` and `SHORT_WAVE_M` are
`u64` (`body.rs:24-26`), so the expression the compiler evaluates is `0 < 2`. The assertion holds; the
arithmetic shown was not the code's. (2) The clamp arm at `body.rs:253-255` still fires for a body
with **16 rungs**, which the ladder accepts up to a 42 723 km radius; D13 sends those bodies to a
cloud shell, so the claim "the arm never fires in production" survives through D13, not through the
octave count.

---

## 2. The recommendation: a closed-form layer stack

```
   INPUTS (all of them facts, none of them tuning)
   ┌──────────────────────────────────────────────────────────────────────┐
   │ body seed │ radius │ mass → g │ spin → Ω │ obliquity + pole azimuth  │
   │ insolation │ atmosphere (pressure, molecular weight) │ water │ AGE   │
   │ eccentricity │ year length │ has a solid surface? │ star luminosity  │
   │   EVERY ONE AN INTEGER, AUTHORED ONCE BY THE BODY'S REALM (§5.4)     │
   └──────────────────────────────────────────────────────────────────────┘
                                   │
       drawn ONCE in BodyDefinition::from_seed, beside the octave table
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ L0  THE PLATE LIST  — P sites on the sphere, each with a drift        │
   │     vector, a crust-thickness draw, an affinity offset and an age.   │
   │     P ≈ 8…40 rows, a fixed-size array with a live prefix.            │
   └──────────────────────────────────────────────────────────────────────┘
                                   │
   evaluated PER COLUMN, on the unit direction, in Gf, no grid, no bake
                                   ▼
   L1 plates      → which plate am I on? how far to the boundary, and which kind?
   L2 isostasy    → the crust FIELD → the base elevation, with the WATER LOAD
                    (the affinity field returns its VALUE and its GRADIENT together)
   L3 orogeny     → boundary uplift: range, trench, rift, arc, hotspot; then the ridged fold
   L4a erosion    → the MULTIFRACTAL WEIGHT: each octave is attenuated by the coarser
                    ones, so floors flatten and crests sharpen — no gradient, no search
   L4e lithology  → the strata's own hardness benches the slope: mesas, ledges, rims
   L5 climate     → snow line, glacial carving, dune fields, the coastal notch
   L6 octaves     → the fine texture, RE-ANCHORED to 3 m, BUDGETED, ladder-CAPPED
   L7 biome       → insolation, latitude, elevation, continentality, rain shadow
                                   ▼
                    h(dir, rung), water(dir), biome(dir)
                                   ▼
                 the density byte, the extractor, the collider

   NOT IN THIS REPORT: the river network and the lake. §4.4b states the three
   conditions any river design must meet and hands it to DOMAIN 03.
```

**The five properties that make this the recommendation:**

1. **Addressable.** A chunk needs no neighbour, no grid and no whole-planet pass. It asks for its own
   62×62 columns and gets them. This is the only property that keeps the 8 ms budget and ruling V4's
   cave rule (*"NOTHING underground is generated … unless a surface crossing lies inside an observer's
   band"*, `owner_decisions_2026-09-07_voxels.md:164`).
2. **Every bounded list is scanned in a FIXED ORDER, and a harvest is only an optimisation.** §4.0
   states the law: the answer of a harvested scan must equal the answer of the full scan, bit for bit.
   That is what removes the asker-dependence, the face indexing and the ordering drift in one rule.
   *Revision 2 made the harvest part of the DEFINITION, and both refuters broke it (R9, R10, B2-2).*
3. **Coarsenable with a DERIVED cut-off.** A layer is dropped at the first rung where its own
   amplitude falls under half a cell, and **every octave's amplitude is CAPPED so that promise is
   true** (§6.2). Today it is false at six rungs of twelve.
4. **Fenced.** Every formula is `+ − × ÷ sqrt floor abs lesser greater clamp` and comparison. §10.1
   checks each one.
5. **Physical.** Every constant is a density, a strength, a heat capacity or a measured planetary
   number, and every per-body value is a seed draw over a physically bounded range. **Every SCALAR;
   the SHAPES — the transfer polynomial, the belt profile, the warp — are named as owed in §10.5.**
   *Refuter A was right that "no magic numbers, passed" was premature (R20).*

---
## 3. The four options, compared

### 3.1 What each option is

| Option | What it is | What it produces |
|---|---|---|
| **(a) Plate simulation** | Seed P plate sites on a sphere. Give each a drift. Step a coarse grid N times: move plates, subduct, accrete, uplift. | Real plate history: collision belts, sutures, orogenies of different ages, drifted continents. The most believable macro map anyone can make. |
| **(b) Uplift + fluvial erosion** (Cordonnier 2016, *Large scale terrain generation from tectonic uplift and fluvial erosion*) | A tectonic uplift field, then iterate the **stream power law** `∂h/∂t = U − k·A^m·S^n` on a grid until steady state. | The correct **slope–area scaling** of real land: broad valleys downstream, steep headwaters, sharp ridge crests. This is what makes the reference picture's shapes. |
| **(c) Pure closed-form** | Domain-warped ridged noise plus a continental mask. | Ridge lines and a land/sea split. Cheap. Never quite right: the ridges do not drain, the valleys have no rivers, the coast has no shelf. |
| **(d) Hybrid** | A closed-form macro skeleton, with the erosion signature reproduced ANALYTICALLY rather than iterated. | §2's stack. |

> **Stream power law, in plain words.** Running water cuts rock at a rate that grows with how much
> water passes (the drainage area `A`) and with how steep the ground is (the slope `S`). Where the
> ground rises at `U` and the river cuts at the same rate, the shape stops changing: that is the
> steady state, and it fixes the slope at `S = (U / (k·A^m))^(1/n)`. **The steady state is a formula,
> not a simulation.** *Game example: the pilot walks up a valley on the home planet. Near the mouth the
> river is wide, the valley floor is nearly flat and the walk is easy. Two hours upstream the same
> river drains a tenth of the area, so the floor tilts five times more steeply and the walk becomes a
> climb. That change is the exponent `1/n` doing its work, and the walk feels like a real valley
> because of it.*

**This is the load-bearing insight of the report.** Option (b)'s VALUE is the slope–area relation, and
the slope–area relation is a closed form once `A` is known. Everything after that is a question about
how to get `A` cheaply — and revision 1 answered it wrongly. §4.4 answers it correctly: `A` comes from
a **rooted network held as data**, never from a per-column search.

### 3.2 Cost, on both hosts, for a grid bake

The grid sizes the task named, for the home planet (`N = 5 263 360` cells per face edge at rung 0, so
one macro cell is `N ÷ edge` metres on the ground). COMPUTED:

| Face edge | Nodes (6 faces) | Cell on the ground | As a ladder rung | Memory @4 B | @16 B |
|---|---|---|---|---|---|
| 256 | 393 216 | 20 560 m | 14.3 | 1.6 MB | 6.3 MB |
| **512** | **1 572 864** | **10 280 m** | 13.3 | **6.3 MB** | 25.2 MB |
| 1 024 | 6 291 456 | 5 140 m | 12.3 | 25.2 MB | 100.7 MB |
| 2 048 | 25 165 824 | 2 570 m | 11.3 | 100.7 MB | 402.7 MB |
| **2 570** | **39 629 400** | **2 048 m** | **11.0 — the top rung exactly** | 158.5 MB | 634.1 MB |
| 4 096 | 100 663 296 | 1 285 m | 10.3 | 402.7 MB | 1 610.6 MB |

The bake, ESTIMATED at 20 ns per node per step (cache-friendly, one core; a flow-routing step is
optimistic at that price):

| Face edge | 50 steps, 1 core | 200 steps, 1 core | 200 steps, 14 cores (8.4× MEASURED, `slice_07_client_link.md:235`) |
|---|---|---|---|
| **512** | **1.6 s** | 6.3 s | **0.75 s** |
| 1 024 | 6.3 s | 25.2 s | 3.0 s |
| 2 048 | 25.2 s | 100.7 s | 12.0 s |

### 3.3 Why the bake is refused — and the refusal is re-argued on the RIGHT reasons

*Revision 1 refused the bake with "seconds and hundreds of megabytes", which are the 2 048-edge,
200-step corner of its own table. Refuter B was right (B-W4): the cheapest useful corner is a 512-edge
grid at **6.3 MB and 0.19 s on 14 threads** (COMPUTED from the same table: 1.6 s ÷ 8.4), and that is
not a scary number. Refusing on the scary corner is an argument against a proposal nobody made.
Revision 2 refuses the bake on the four reasons that hold at EVERY corner, and states the cost
honestly rather than at the worst corner.*

1. **It is not addressable.** A chunk cannot be evaluated without a whole-body pass first. That breaks
   the one property the 8 ms budget rests on, and it breaks ruling V4's cave rule by making a
   whole-planet pass the precondition of any surface at all. **This reason holds at 6.3 MB exactly as
   it holds at 634 MB.**
2. **It is a SECOND SHAPE.** The baked grid is not the ladder. §3.2 shows the only grid that IS a rung
   is the 2 570 edge, at 158 MB. Any other grid must be interpolated, and an interpolated source under
   an exact extractor is a new drift surface on every target — which SL10 clause 3 measures byte for
   byte.
3. **Four new drift classes, all of them ORDER, none of them covered by the fence.** Float addition is
   not associative, so a drainage-area accumulation must fix its summation order. A priority-flood
   heap pops equal keys in an implementation-defined order. A "run until converged" test compares
   floats. A parallel bake is non-deterministic unless the partition is fixed. `crates/terrain/src/gf.rs`
   fences ARITHMETIC; it fences no ordering. This project has never carried an ordering fence, and
   `D-TERRAIN-1` shows the arithmetic fence still owes a real x86-64 leg.
4. **HR5.** An iterative solver at 100 % region and branch coverage is a large, unpleasant surface: a
   convergence branch, a boundary branch, a degenerate-basin branch, a saturation branch, each in
   every monomorphisation.

**The cost, stated honestly rather than at a corner:** a 512-edge bake costs the CLIENT 6.3 MB and
about 0.19 s per body on 14 threads, or 1.6 s on one. That is survivable. **It is refused on reasons
1–4, not on that number.**

**Verdict: (a) and (b) are refused as SHIPPED machinery.** §11 keeps one lawful use for them: as an
OFFLINE ORACLE that a test compares the closed form against, never in a product build.

### 3.4 Why pure closed-form (c) is not enough either

A domain-warped ridged multifractal gives ridge LINES. It does not give:

- a coast with a shelf (needs two crust types — L2);
- a range that ends where its boundary ends (needs plates — L1/L3);
- a valley that LOOKS eroded at every scale (needs the multifractal weight — L4a), and a valley that
  DRAINS, which needs a rooted network nothing in this report builds (domain 03);
- a benched cliff (needs the strata read back into the shape — L4e);
- a snow cap at the right altitude (needs `g`, insolation, latitude and a lapse rate — L5).

Option (c) is the layer L6 we already have, dressed better. It is necessary and not sufficient.

---

## 4. The layers, one by one

### 4.0 L0 — the plate list, and ★ THE SCAN LAW that replaces the hydrology cell

**What it is.** A short list of `P` sites on the unit sphere. Each row holds: a unit direction (the
site), a drift vector tangent at that site, a crust-thickness draw, a continental-affinity offset and
an age. Every field is a seed draw. `P` comes from the body's **tectonic vigour** (§4.7).

**Where it lives.** Inside `BodyDefinition`, beside `octaves: [Octave; OCTAVES]`
(`crates/terrain/src/body.rs:99-100`), as `plates: [Plate; PLATES_MAX]` with a live prefix
`plate_count: u8` — the identical shape the octave table already uses. That matters: the struct is
`Copy` and it is compared by `PartialEq` (`body.rs:88-91`), and a fixed-size array keeps it so.

**Sizing, and a decision it forces.** `PLATES_MAX` is the smallest number above the largest plate
count the vigour law can draw, and §4.7 bounds that at 40. So **`PLATES_MAX = 40`** — not 64, because
the array is dead weight in a `Copy` struct. At an ESTIMATED 56 bytes a row that is **2 240 bytes**,
against today's whole `BodyDefinition` of roughly 500 bytes (`octaves: [Octave; 16]` at 24 bytes a
row plus the small tables). *Refuter A and refuter B both flagged the weight (R19, B2-24): the struct
is `Copy`, `from_seed` returns it by value, `home_planet()` returns it by value, and the client holds
one per realm (`crates/client/src/chunks.rs:275-289`). This report does not decide it: **D17** asks
the owner whether the plate list sits in the struct, behind a reference, or is re-drawn on demand, and
**M10** prints the size before the choice is made.*

#### ★ THE SCAN LAW — the fix for the accelerator, and it is the important part

*Revision 1 proposed a fixed cube-face bucket grid; both refuters refused it. Revision 2 replaced it
with a HYDROLOGY CELL — a ladder cell at a fixed rung near 2 km — and made the harvest part of the
DEFINITION of every layer. **Both refuters broke that too, and from three directions:***

- *refuter B: at rung 11 a chunk spans 62 × 2 048 = **126 976 m**, so its 3 844 columns sit in
  **3 844 different hydrology cells** — one harvest per column, which INVERTS the ladder's cost order
  and makes the top rung dearer than rung 0 (B2-2, COMPUTED and correct);*
- *refuter A: the harvest law was written for a MINIMUM (the nearest site), and an ADDITIVE lookup
  (a sum over branches) is not a minimum, so two different lists can give two different sums (R9);*
- *refuter A: a ladder cell IS `(face, a, b)`, and getting to it from a direction runs `unbend`,
  which is four Newton steps with a division each (`crates/seed/src/bend.rs:47-66`), so the layout
  would inherit the bend's world identity and an unpriced per-column cost (R10).*

**All three are right, and one rule answers all three.**

> **THE SCAN LAW.** Every layer is DEFINED as a scan of the body's bounded lists **in a fixed global
> index order**, over the whole live prefix. A harvest — any shorter candidate list — is an
> OPTIMISATION and never a definition. It is lawful only when it returns the identical answer, bit
> for bit, and that holds when both of these hold:
>
> 1. **Compact support with an exact zero.** Every list member's contribution is a clamped polynomial
>    that is EXACTLY `+0.0` outside its own support radius. Not "small": zero.
> 2. **A fixed summation order.** The sum runs in list-index order, always, whatever the harvest
>    contains. Float addition is not associative, and §3.3 refuses a competing design for exactly
>    this hazard.
>
> Then a harvest may be keyed however the caller likes — by the asking chunk, by a coarse cell, by
> nothing at all — because the key cannot change the answer. **M6 tests it: the harvested answer
> equals the full-scan answer, bit for bit, over a sample of columns.**

```
   REVISION 2 (refused)                        REVISION 3 (the scan law)

   dir → its hydrology cell → the harvest      dir → the FULL fixed-order scan   = the ANSWER
             │                                        ▲
             ▼                                        │ must be bit-equal (M6)
   the harvest DEFINES the answer  ✗           any harvest, keyed anyhow  ─────┘
   → a different cell, a different answer      → the key is free, the answer is not
```

**What the law buys, one line each:**

- **The ladder's cost order is restored.** At rung 11 a chunk does not harvest 3 844 times; it scans
  the live prefix per column, exactly as at rung 0, and the coarse rung stays cheaper because it has
  fewer octaves and fewer layers (§9.5: 1 118 µs against 2 075 µs).
- **The bend is irrelevant again.** No layer needs a cell address, so no layer calls `unbend`, so
  changing `INVERSE_STEPS` cannot move a plate. *That is the answer to R10, and it is the SCAN
  LAW doing the work, not an assertion.*
- **An additive lookup is safe**, because condition 1 makes every non-candidate contribute an exact
  zero and condition 2 fixes the order. *That is the answer to R9, and it is a REQUIREMENT on any
  future river design (§4.4b), stated so domain 03 inherits it.*
- **The face seam is untouched**, because nothing names a face.

**The cost, priced honestly.** The scan is over the LIVE prefix: the home planet draws an ESTIMATED
12–24 plates (§4.7), so a column pays 12–24 dot-and-compare pairs at ESTIMATED 3 ns, plus **one**
bisector normalise for the winner and the runner-up (a subtract, a dot, one `sqrt` and three divides,
ESTIMATED 10 ns). **ESTIMATED 70 ns per column, and 130 ns on a 40-plate body**, which §9.1 carries as
the budget line and §9.2 carries as the worst case. *Refuter A was right that revision 2's "~5 ns per
dot-and-compare" hid the normalise (R14); the normalise is now once per column, not once per pair,
and the number is stated for both ends of the draw.*

*Game example: the home planet's shard boots. `BodyDefinition::from_seed` draws 19 plates from the
planet's own seed, exactly as it draws 14 octaves today. A chunk under the pilot's boots scans all 19
in index order. The client, linking the same crate, scans the same 19 in the same order and gets the
same metre. Neither ever sends a plate to the other.*

### 4.1 L1 — plates: which plate, and how far to which boundary

**What it produces.** For a direction `d`: the owning plate `p`, the nearest boundary distance `b` in
metres, and the boundary's **kind**, which is a closed form of the two plates' drifts:

```
  relative drift  v = drift(p2) − drift(p1),  boundary normal  n (from p1 toward p2)

      v · n  <  0      →  CONVERGENT   (they push together)   → range / trench / arc
      v · n  >  0      →  DIVERGENT    (they pull apart)      → mid-ocean ridge / rift valley
      |v · n| ≈ 0      →  TRANSFORM    (they slide past)      → an offset, a scarp, no uplift
```

> **Convergent, divergent, transform, in plain words.** Two plates can push together, pull apart, or
> slide past each other. Pushing together builds mountains and digs trenches. Pulling apart opens
> rifts and, under water, builds a ridge of new hot rock. Sliding past shifts the ground sideways and
> leaves a scar without much height. *Game example: the pilot flies west across the home planet. The
> ground rises for 300 km into a range, drops into a valley of old sea floor, and rises again into a
> line of volcanic islands. That is one convergent boundary, and one field of the plate list drew it.*

#### The boundary distance `b`, written out and fenced

For two unit sites `s₁`, `s₂` the great-circle bisector is the plane through the origin whose normal
is `n̂ = (s₂ − s₁) / |s₂ − s₁|`. For a unit direction `d`, the SINE of the angle from `d` to that
plane is exactly `d · n̂`. So:

```
        b  =  R · (d · n̂)          all of it: subtract, dot, sqrt, divide
```

**The error is the sine against the angle, and it is bounded and stated** (COMPUTED for the home
planet, `R = 3 350 759 m`):

| Angle | Distance on the ground | `(a − sin a)/a` |
|---|---|---|
| 1° | 58 km | 0.01 % |
| 5° | 292 km | 0.13 % |
| 10° | 585 km | 0.51 % |
| **14°** | **819 km** | **0.99 %** |
| 20° | 1 170 km | 2.02 % |
| 30° | 1 754 km | 4.51 % |

Every L2 and L3 profile is bounded by a transition width `W` under 500 km, so the error inside every
profile is **under 0.4 %**, which is under a metre of height on a 200 m profile. Beyond `W` the
profile is flat and the error is invisible — and **"flat" means an exact `+0.0`, which is condition 1
of the scan law**.

#### ★ Where three plates meet

*Refuter B asked what a triple junction looks like, and it is a fair question because the nearest-two
rule is ambiguous exactly there (B2-25).* A triple junction is a point where three plates meet, as at
Afar in East Africa. Under this construction the column simply takes its nearest and second-nearest
sites; near the junction the second-nearest changes over a few kilometres, so the boundary KIND
changes there, and the three boundary profiles meet at a point with their own uplifts summed. The
tie-break by site index decides only which of two EQUAL distances wins, which is a set of measure
zero on the sphere and never a visible line. **What the junction looks like is a picture question, and
M4 owes the shot**; what matters for the law is that the answer is a function of the direction alone,
which the scan law guarantees.

**Believability.** High. Spherical Voronoi plates with drift are what the geodynamics literature calls
a kinematic plate model. The plate SHAPES are Voronoi cells, which are a little too regular; D6 offers
the cure (a warped distance, one noise call), decided on pictures.

**Cost, ESTIMATED:** 70 ns per column (see §4.0), 130 ns on a 40-plate body.
### 4.2 L2 — isostasy, on a crust FIELD, with the WATER LOAD

> **Isostasy, in plain words.** The crust floats on the mantle like a raft on water. A thick, light
> raft rides high; a thin, heavy one rides low. That is the whole of it, and it is why continents
> stand above ocean floors — not because they were pushed up, but because they float higher.

**Airy isostasy** gives the elevation of a crust block of thickness `t` and density `ρ_c` over a
mantle of density `ρ_m`:

```
                e = t · (1 − ρ_c / ρ_m)
```

COMPUTED with `ρ_m = 3 300 kg/m³`:

| Crust | Thickness | Density | Elevation over the compensation depth |
|---|---|---|---|
| continental | 35 000 m | 2 800 | **5 303 m** |
| oceanic | 7 000 m | 2 900 | **848 m** |
| **the step, DRY** | | | **4 455 m** |

#### ★ The step is wrong until the water stands on it

*Revision 2 stopped at 4 455 m and compared it with Earth's real 4.5 km step. **Both refuters
refused that comparison and both are right** (R11, B2-9): Earth's 4.5 km is measured with four
kilometres of sea water standing on the ocean floor, and water pushes that floor down. The report
left out a term of the same size as the effect it claimed to reproduce.*

> **Water loading, in plain words.** Pour an ocean into a basin and the basin sinks, because the
> water weighs on the crust and the mantle flows out from under it. A dry basin and a flooded basin
> of the same crust do not sit at the same depth. *Game example: the pilot flies over a dry world's
> basin and over a water world's basin. The same crust, the same thickness — and the flooded floor
> lies two kilometres deeper.*

The pressure balance at the compensation depth, with the continent's surface as the reference:

```
   t_c(ρ_c − ρ_m)  =  d(ρ_w − ρ_m) + t_o(ρ_o − ρ_m)

   35 000(2 800 − 3 300) = d(1 030 − 3 300) + 7 000(2 900 − 3 300)
   ⇒  d = 6 476 m below the continental surface       (COMPUTED)
```

And it has a **closed form**, which matters because §7.1 must not need an implicit solve:

```
   d  =  step_dry / (1 − ρ_w/ρ_m)  =  4 455 / 0.688  =  6 476 m       (COMPUTED, and it
                                                                        reproduces the balance)
```

*Refuter A's form `step = 4 454.5 + 0.312·d` is the same statement written incrementally, and its
fixed point is the same 6 476 m. Refuter A also charged that the sea-level solve is therefore
implicit (R11's last paragraph). It is not, and this is the one leg where the refuter's conclusion
does not follow: the loading term is a LINEAR function of the depth below the water surface, so a
column's loaded elevation is `h_dry − (level − h_dry)·ρ_w/ρ_m` wherever `h_dry` is under the level,
which is one multiply inside the bisection's own loop, not an outer iteration. §7.1 states it.*

**And the load is per body, which is what refuter B's second point demands (B2-9).** D5 draws the
water inventory per body, so a dry world's basins stand shallow against their continents and a
water-rich world's stand deep. Under revision 2 every world got the same 4 455 m step. Under
revision 3 the step is `step_dry` where there is no water and `step_dry/(1 − ρ_w/ρ_m)` under a full
ocean, with the sea level deciding which.

#### ★ The crust is a FIELD, not a per-plate label

On Earth most coast is a **passive margin**: a continent and an ocean floor on the SAME plate, with
no boundary between them. The whole east coast of the Americas, the whole west coast of Africa and
Europe, all of Australia's coast and India's are passive margins, and they carry most of the world's
continental shelf.

> **Passive margin, in plain words.** A coast where the land simply ends and the sea floor begins,
> with no plate boundary anywhere near. It is quiet: no earthquakes, no volcanoes, a wide shelf, a
> gentle slope, thick sediment. Most of the world's beaches are on one. *Game example: the pilot lands
> on a beach on the home planet, and there is no range behind it and no trench offshore — the shelf
> runs out 200 km before the floor drops away. That is the ordinary case.*

**The cure, and it is small.** The crust thickness is a FIELD over the sphere:

```
   t(d)  =  t_ocean  +  (t_cont − t_ocean) · S( c(d) )

   where  c(d) = a CONTINENTAL AFFINITY field  =  a slow noise (3 octaves,
                 wavelength ~ 3 000 km) + a per-plate affinity offset from L0
          S(·) = the quintic fade already in the crate (noise.rs:66-71), used
                 as a smoothstep between 0 and 1
```

- The **per-plate offset** keeps a plate mostly continental or mostly oceanic, which is real: a plate
  does have a character.
- The **noise term** is boundary-independent, so a continent's edge falls where the affinity crosses
  the threshold, which is usually INSIDE a plate. That is a passive margin, and it is now the ordinary
  case.
- A plate boundary that also crosses the affinity threshold gives an ACTIVE margin — a trench under a
  cordillera, the Andes. Both exist, from one field, with no kind branch (HR3).

**This one layer produces:** continents that stand as PLATFORMS not as swells, ocean basins with a
FLOOR at a common depth, a **continental slope** wherever the affinity crosses, a **continental
shelf** wherever the sea covers the platform's edge, and therefore a **coast**.

**No magic numbers.** `ρ_c`, `ρ_m`, `ρ_o` and `ρ_w` are densities; the thicknesses are seed draws over
physically bounded ranges. The equilibrium crust thickness of a rocky body scales with its internal
heat flux and its gravity; a fenced form is `t ∝ 1/g` at fixed composition, calibrated on Earth
(35 km at `g = 9.81`), with a seed factor in `[0.7, 1.4)`. **The 35 km and 7 km are Earth
CALIBRATIONS and this report says so.**

#### ★ The value and the GRADIENT come out of one evaluation

The affinity field is the only part of the macro stack with a gradient anybody needs (§4.5's rain
shadow and §4.6's modulator read it). **Gradient noise gives its own derivative for about 1.4 times
the cost of its value** — the eight corner gradients are already in hand; only the fade's derivative
is new. So the field returns `(c, ∇c)` together.

*This replaces revision 2's four-point finite difference, and it answers three findings at once.
Refuter B was right that revision 2 never paid for the affinity's noise calls at all and that the
corrected cost turned the budget red (B2-6). Refuter A was right that the same quantity was priced
at 55 ns in §4.4a and 135 ns in §7.1 (R6) — a factor of 2.5 apart inside one document, which is the
defect the report itself charged revision 1 with. **One price, stated once: 3 noise-with-derivative
calls at ESTIMATED 18 ns, plus 10 ns of arithmetic = 64 ns**, and there is no separate gradient row
in §9.1 and no finite-difference step `δ` anywhere in the design.*

**Cost, ESTIMATED:** 64 ns per column, the affinity's noise calls included.
### 4.3 L3 — orogeny: ranges, trenches, rifts, arcs and hotspots, under a gravity ceiling

#### The ceiling: a crustal-strength relation, stated as a SCALE and not as a law

A mountain stands on rock that has a finite strength. The pressure under a column of height `h` is
`ρ_c·g·h`. When that pressure passes the crust's long-term yield stress `σ_y`, the base flows and the
mountain spreads under its own weight. So the ceiling is:

```
                h_max  =  σ_y / (ρ_c · g)
```

Everest fixes the one constant: `σ_y = 2 800 × 9.81 × 8 850 = **243.1 MPa**` (COMPUTED).

**What that number is, stated honestly.** *Refuter A is right (R16): `σ_y` here is the stress at
which the root flows, and the "granite's unconfined compressive strength is 100–250 MPa" check
compares two different material properties. An unconfined compressive strength is measured on a free
sample at the surface; the rock under a mountain root sits at hundreds of megapascals of confining
pressure, where the limit is ductile flow and gravitational spreading, not crushing.* So the honest
statement is: **`σ_y` is a CALIBRATION on one body, its value lands in the same order of magnitude as
laboratory rock strengths, and that agreement is a sanity check and not a validation.** What is
physical, and what the design actually uses, is the `1/g` TREND.

COMPUTED, with `ρ_c = 2 800 kg/m³` for every body so the only variable is `g`:

| Body | `g` (m/s²) | Ceiling | The tallest real relief | How much of the ceiling it reaches |
|---|---|---|---|---|
| Earth | 9.81 | 8 850 m | Everest 8 849 m | **100 %** (the calibration) |
| **Mars** | **3.71** | **23 401 m** | Olympus Mons 21 900 m | **94 %** |
| Venus | 8.87 | 9 788 m | Maxwell Montes 11 000 m | **112 %** |
| the Moon | 1.62 | 53 592 m | ~10 800 m over the mean radius | **20 %** |
| the home planet (`g` OWED, M2) | 3.68 … 5.17 | 23 592 … 16 793 m | — | — |
| a 12 000 km super-earth | **28.9** | **3 004 m** | — | — |

*The Mars row is CORRECTED. Revision 2 printed 22 594 m and 97 %, which needs `ρ_c = 2 900` — a
density the table says it does not use. Both refuters found it independently and both are right
(R5, B2-8). With the declared 2 800 the ceiling is 23 401 m and Olympus Mons reaches 93.6 %. The
spread over the three bodies that HAVE built mountains is therefore Earth 100 %, Mars 94 %, Venus
112 %, which is the **±15 %** this report claims — no better, and no worse.*

**How to read this table, honestly.**

- **It is a CEILING, not a prediction.** A body that reaches 20 % of its ceiling has not falsified it;
  it has simply never built a mountain. The Moon has no orogeny at all — no plates, no rivers, no
  uplift — so its relief comes from impact basins, and §4.7's crater recipe is what produces those.
- **Venus exceeds it by 12 %**, so the relation is a scale with a spread, not a wall. The design uses
  it as the SCALE of the belt amplitude draw and never as a hard clamp on the field.
- **Mars is a weak check**, because Olympus Mons is a shield volcano whose height is set by magma
  buoyancy and lithosphere thickness, not by orogenic uplift. Two mechanisms landing within 6 % of
  one line is suggestive, not conclusive.
- **The super-earth's ceiling is if anything OVERSTATED.** *Refuter B is right (B2-23): the same
  chain COMPUTES that body's mean density at 8 608 kg/m³, and a compressed crust is denser than
  Earth's, so `σ_y/(ρ_c·g)` with a truer `ρ_c` is lower than 3 004 m. D4's case is stronger than the
  table shows, and the table is left at the conservative number on purpose.*

**The super-earth's gravity, sourced.** COMPUTED from the Zeng–Sasselov–Jacobsen rocky mass-radius
relation `R/R⊕ = (M/M⊕)^0.27`: a 12 000 km planet is **10.4 Earth masses**, mean density
**8 608 kg/m³**, `g = 28.9 m/s²`, ceiling **3 004 m**. So today's rule (`0.4 % of radius, capped at
12 km, × [0.5, 1.5)` → 6 to 18 km) over-reliefs it by **2 to 6 times**.

**Today's rule on the home planet** gives a relief of 14 305 m against a ceiling between 16 793 m and
23 592 m. It sits under the ceiling **by luck, not by rule** — and the size of the luck depends on the
planet's density, which nobody has measured yet (M2).
#### The uplift field

For a column at boundary distance `b` on a CONVERGENT boundary with convergence rate
`c = −(v·n)` and boundary age `a`:

```
   U(b) = h_max · f(c) · g(a) · profile(b / W)
```

- `f(c)` — the convergence rate, normalised, so a fast collision builds a taller belt;
- `g(a)` — age: a young belt is high and sharp, an old belt is low and rounded (the Appalachians
  against the Himalayas). **This is where time enters without the crate ever knowing the tick**: the
  AGE is a seed draw, a fact about what the belt IS, exactly as a hull states its rating;
- `profile(x)` — the cross-belt shape: a rise, a crest, a steeper back slope, and a **foreland basin**
  in front. A polynomial, in `Gf`. **Its own gradient is a polynomial too**, so the macro slope the
  rain shadow and the roughness modulator need costs ~10 ns and no finite difference (§4.2).

> **Foreland basin, in plain words.** The weight of a new range bends the crust in front of it, and
> the dip fills with the range's own debris. It is a broad low plain hugging the mountain front.
> *Game example: the pilot flies out of the range on the home planet and crosses 80 km of flat,
> sediment-floored ground before the ordinary hills start. That is the foreland, and the fields of
> the reference picture sit on one.* *(Refuter B asked for this word and eight others to be
> explained; B2-21.)*

**The belt amplitude draw is bounded by `h_max`, and that bound is what sizes the band** (§9.3). It is
a per-body constant known inside `from_seed`, which is what makes the band exact.

**Ridge lines.** Along the belt, the crest is a set of parallel ridges: the classical **ridged
multifractal**, `1 − |noise|`, summed over octaves, with the coordinate **domain-warped** by a second
noise so the ridges bend instead of running straight.

> **Domain warping, in plain words.** Ask for the noise not at the point you are at, but at the point
> displaced by another noise. Straight features become sinuous and geologically plausible. *Game
> example: a range on the home planet no longer runs as a straight rib across the map; it bends around
> an old block, exactly as a real fold belt does.*

**The honest price.** *Revision 1 priced "three warped ridged octaves at ~13 ns each" as 45 ns. Both
refuters showed that a 3-D domain warp needs three noise calls for the displacement vector on top of
the field call.* COMPUTED against the MEASURED 13 ns per noise call: three warp calls + three ridged
calls = **78 ns**; the belt profile and its analytic gradient add **20 ns**, so **L3 costs 98 ns**.
Those two numbers are carried into §9 as separate rows, because the ridged fold has a derived cut-off
rung and the profile never does (§6.2).

**Every other boundary kind is the same machinery with a different profile**, which is how HR3 stays
clean:

| Boundary + crust pair | Landform | Profile |
|---|---|---|
| convergent, continent + continent | a collision belt (Himalaya) | a broad high crest, doubled crust, no trench |
| convergent, ocean + continent | a trench and a coastal cordillera (Andes) | a deep narrow trench, then a steep range |
| convergent, ocean + ocean | an **island arc** (Aleutians) | a trench, then a line of separate high points |
| divergent, ocean + ocean | a **mid-ocean ridge** | a broad rise (young hot crust rides ~2–3 km higher), with an axial valley |
| divergent, continent + continent | a **rift valley** (East Africa) | two shoulders and a dropped floor |
| transform | a scarp | a lateral offset of the field, near-zero uplift |
| a hotspot site inside a plate | a **hotspot chain** | a line of cones, spaced by the drift rate and aged along it |

#### The hotspot chain, fenced

*Revision 1 said "the chain is the great-circle track of the site under the plate's drift". Refuter A
was right (F8): moving a point along a great circle by an arc is Rodrigues' rotation, which needs `sin`
and `cos`, and `Gf` grants neither.*

The fenced construction: cone `k` sits at

```
      d_k  =  normalise( s + t · u_k )       s = the site, t = the drift tangent,
                                             u_k = a seed-drawn scalar, increasing in k
```

`normalise` is three multiplies, two adds, one `sqrt` and three divides — all fenced. The arc that
`d_k` reaches is `atan(u_k)`, which is monotone in `u_k`, so the spacing is a monotone
reparametrisation of the true arc. Geologically it is indistinguishable from an even arc spacing (a
real chain's spacing is not even anyway, because the drift rate changed), and it needs no
transcendental at all. **The chain is still a great circle**, exactly: every `d_k` lies in the plane
of `s` and `t`.

*Game example: the pilot flies along a line of eight islands. The nearest is a bare cone; the furthest
is a flat atoll. The chain names the direction that plate has been moving for a hundred million years,
and a player can read it.*

**Cost, ESTIMATED:** 98 ns per column — **20 ns** for the belt profile and its analytic gradient, and
**78 ns** for the ridged fold (3 domain-warp noise calls plus 3 ridged calls at the MEASURED 13 ns).

### 4.4 L4 — the erosion signature, REBUILT AGAIN: no march, no gradient, no river

*Revision 1 proposed a 32-step downhill march; both refuters priced it out of the budget and it was
deleted. Revision 2 replaced it with a gradient of the smooth macro field plus a river-network
lookup. **Both halves are now refused, and by good arguments.***

> *Refuter A, R13: `h_smooth` is L1 + L2 + L3's profile. L2's affinity has a wavelength near
> 3 000 km and L3's profile is bounded by a width under 500 km, so `∇h_smooth` is effectively
> CONSTANT across a rung-0 chunk, whose 62 columns span 62 m. **It cannot tell a valley floor from a
> divide at the kilometre scale the reference vista shows** — the only field with content at that
> scale is the octave sum, which `h_smooth` deliberately excludes. The refuter is right, and the
> consequence is that revision 2 spent 53 % of its added budget on a term that could not do the job
> it was bought for.*
>
> *Refuter A, R2 and refuter B, B2-1: the river lookup was priced at the cost of READING a branch
> list that nothing builds. Built as revision 2 described it — grow a tree inland from the coast,
> each node's height set by its parent's — a cell far inland needs the whole chain back to a mouth,
> and the drainage area needs a sum over the whole catchment. **Both walks leave the cell.** Refuter
> B COMPUTED the size: a 2 048 m hydrology cell on the home planet is a grid of 39 629 400 cells, the
> same object §3.3 refuses. **A function can be pure and still cost a planet.** Both refuters are
> right and neither exaggerated.*

**So L4 is rebuilt a second time, and it is now two things and a hand-off.**

#### L4a — the MULTIFRACTAL WEIGHT: the erosion signature with no gradient at all

**The idea, in plain words.** Erosion does not act equally everywhere: a slope that is already steep
sheds more, a floor that is already flat collects more. You do not need to know WHERE the water goes
to reproduce that; you need each finer scale to know how rough the coarser scales already were. So
each octave is multiplied by a weight built from the octaves above it:

```
   h = Σ  amp(i) · n_i(dir) · w_i        with   w_0 = 1
                                                w_i = clamp( w_{i-1} · (a + b·|n_{i-1}|), 0, 1 )
```

- where the coarse field is FLAT (`|n|` small) the weight collapses, so the fine octaves nearly
  vanish and the ground is a **floor**;
- where the coarse field is STEEP the weight stays near one, so the fine octaves land at full
  amplitude and the ground is a **crest, a flank, a broken ridge**;
- it costs **one multiply, one absolute value and one clamp per octave** — ESTIMATED 15 ns for the
  whole 15-octave table — and it needs **no gradient, no neighbour and no search**.

This is the standard hybrid multifractal of the procedural-terrain literature (Musgrave), and it is
the cheapest known way to get the LOOK of erosion out of a noise sum. **What it gets right:** flat
floors, sharp divides, the erosion signature at EVERY scale from 50 km down to 3 m — including the
kilometre scale where refuter A showed the macro gradient has nothing to say. **What it gets wrong:**
it has no drainage. A "valley" made this way does not lead anywhere.

**And the crest IS the divide, by construction.** *Refuter A raised the coupling as a believability
weakness (R15): under revision 2 the ridges and the drainage were independent fields, so a river
could run along a crest.* Under the multifractal weight the roughness and the height come out of ONE
field, so the high ground is the rough ground and the low ground is the flat ground everywhere, at
every scale. **That does not give a river a course** — domain 03 owes that — but it removes the
class of picture where a smooth ridge sits beside a corrugated plain.

**Cost, ESTIMATED:** 15 ns for the weight, plus 20 ns for the shaping arithmetic below = 35 ns.

#### L4b — the river network: THREE CONDITIONS, then hand it to domain 03

**This report does not design the river network.** It states what it costs to be without one, and it
states the three conditions any design must meet so that domain 03 does not have to rediscover them:

1. **Bounded work per column, with no walk that leaves the body's bounded lists.** A construction
   whose cost grows with the distance to a river mouth is the march again, moved. A recursion whose
   depth is bounded by the LADDER (a coarse skeleton refined rung by rung, ESTIMATED 8 levels from a
   500 km root to a 2 km branch on the home planet) is the shape that could work, and it is
   UNMEASURED.
2. **The scan law of §4.0.** A river contribution is a SUM, not a minimum, so every branch outside a
   candidate list must contribute an exact `+0.0`, and the sum must run in a fixed global branch
   order. *That is refuter A's R9, and it is binding on domain 03.*
3. **A per-column water surface to live in** — §4.4c below, which this report does build, because the
   field must exist before anything can put water in it.

**What being without a river costs, stated for the owner.** *Refuter B asked for this sentence
plainly (B2-18) and it is owed.* **The first landing has ridges, valleys, flanks, benches, cliffs,
a shelf and a coast, and it has no river, no lake and no flood plain.** The reference picture has a
river through the valley floor, and doc 01 counts it as a named orienter
(`01_reference_target.md`). M4's pass mark must say so, and D11 puts the order to the owner.

#### L4c — ★ THE WATER SURFACE: what actually holds a river

The code holds ONE water rule for a whole body: `crates/terrain/src/chunk.rs:207-209` —
`if r < body.sea_radius_m { Water } else { Air }`. A river at 1 200 m above the sea and a lake in a
highland basin have no field to live in.

**The cure, and it needs no stored-record change.** The column pass already computes one surface
radius per column. It computes a **second** one:

```
   ColumnField { surface_m, water_m, biome, … }        water_m = the water surface, or NONE

   fluid_at(body, column, r)  =  Water  where  r < column.water_m
                                 Air    otherwise
```

- On the open sea `water_m` is `sea_radius_m` — today's behaviour exactly, so nothing regresses.
- In a river channel `water_m` is the channel's own surface, when domain 03 gives one.
- In a closed basin `water_m` is the basin's fill level.
- On dry land `water_m` is below `surface_m` and no cell is Water.

**What it costs, stated against the right object.** *Refuter A was right that "the cell is still 12
bytes" mixed two records (R22).* The **stored** 12-byte record of ruling V6 part B does not change:
`Stratum::Water` is already a variant (`crates/terrain/src/strata.rs:18`, and `:44` is its row in
`Stratum::ALL`). The **generated** cell in the crate is `Cell { stratum, gap }`, two bytes, and it
does not change either. What DOES change is `fluid_at`'s signature, `above_surface_cell`
(`chunk.rs:216-223`), the above-surface skip and the halo rule — *"the one rule the above-surface
skip, the cell pass and the halo share"* (`chunk.rs:204-205`). That is a real, contained change to
four call sites, and it is **a world-identity version bump because it moves cells** (W3b).

#### L4d — deposition, the talus, and the flood plain

Three operators, each local, each a polynomial, all inside L4a's 20 ns of shaping:

- **Alluvial fill.** Where the multifractal weight is low and the macro slope is small, RAISE the
  field toward a local base level. That is what makes a floor flat rather than merely smooth.
- **The angle of repose.** Loose rock cannot stand steeper than about **32–37°**. A slope limiter
  clamps the shaped gradient to `tan(θ_repose)` on unconsolidated material and leaves bedrock free to
  stand vertically. `Gf::clamp` does it. **This is what puts scree under a cliff.**
- **A delta.** Where the fill meets the water surface, fan it outward. One extra term on the same
  operator. Without domain 03's channels a delta has nothing to grow from, so this term waits with
  the river.

#### L4e — ★ LITHOLOGY: the benched cliff, for five nanoseconds

*Refuter B found the gap and it is the best believability finding of this refutation (B2-12): the
crate already holds a stratigraphy — topsoil over subsoil over sediment over bedrock, one of three
sediments and one of five bedrocks (`crates/terrain/src/body.rs:200-206`;
`crates/terrain/src/strata.rs:177-200`) — and **the height field never reads any of it**. Every
landform in revision 2 was cut out of one homogeneous material.*

> **Differential erosion, in plain words.** Water and frost take soft rock away faster than hard rock.
> Where the layers are stacked, the hard ones stand out as ledges and the soft ones cut back as
> slopes. That stepping is most of what makes a rock face look like rock. *Game example: the pilot
> walks into a canyon on the home planet. Under revision 2 the wall is one smooth ramp of one
> substance from floor to rim. With this term the same wall is four bands: a sandstone ledge, a shale
> slope, another ledge, then talus.*

**The construction.** The strata table is a function of DEPTH BELOW THE SURFACE today
(`StrataTable::at`, `strata.rs:189`). For benching, read it at the column's own height against a
per-body **datum**: the strata are horizontal, so the substance at a given RADIUS is the same across
a whole region, and a hard bed outcrops as a line along a hillside. Then one hardness factor
multiplies the shaping term of L4a: hard rock resists the transfer and holds a ledge, soft rock
retreats.

**Why this is cheap and why it needs no rock map.** It is ONE table read per column against a radius
the column already has — ESTIMATED **5 ns** — and it uses the strata the body already draws. **It
does NOT need D10's per-direction rock map**, so the benched cliff is available whichever way the
owner rules on the ore question. *That separation is new in revision 3; revision 2 debated the rock
map for ore and never noticed that benching was free.*

**What it produces:** escarpments and cuestas (a hard cap holding a cliff line for tens of
kilometres), mesas and buttes (the same cap, isolated), benched valley walls, and the resistant ribs
of a fold belt. A waterfall needs a river, so it waits for domain 03.
### 4.5 L5 — climate operators: the snow line, done properly

#### ★ The snow line, rebuilt

*Revision 1 computed "the height where a 288 K surface reaches 273 K" on the DRY adiabat `g/c_p`, got
1 537 m for Earth, and called it the answer to the owner's "snow-capped ridges". Refuter B refused it
(B-D6) and every part of the refusal is correct: `g/c_p` is the dry adiabat, not the environmental lapse
rate; the 0 °C level of an annual mean is not a snow line; 288 K is Earth's GLOBAL mean, so a tropical
mountain is started 12 K too cold; and Earth's real snow line runs 4 500–5 000 m at the equator, so
revision 1 was low by a factor of three on the one body it had a real number for.*

Two corrections, both fenced:

**1. The environmental lapse rate, not the dry adiabat.** Rising air condenses and releases latent
heat, so it cools more slowly than a dry parcel. Earth's environmental rate is about 6.5 K/km against a
dry adiabat of 9.76 K/km, a ratio of **0.666**. So:

```
      Γ_env  =  (g / c_p) · w        w = a WETNESS factor from the water inventory
                                         and the temperature (Earth: 0.666; a dry
                                         or frozen body: 1.0, the dry adiabat)
```

COMPUTED: Earth 6.50 K/km ✓ (by construction); the home planet 2.44 K/km at a Mars-like density,
3.43 K/km at an Earth-like one.

**2. The snow line is a WARMEST-MONTH freezing level, and it has a latitude.**

```
      z_snow(lat)  =  ( T_warm(lat) − 273.15 ) / Γ_env
```

`T_warm` is the warmest-month mean at that latitude: the body's equilibrium temperature, plus the
greenhouse offset from the atmosphere record, plus the insolation's latitude profile (a function of
`dir · pole_axis`), plus the seasonal swing from the obliquity. Every term is a fact of the body.

COMPUTED, Earth check, `Γ_env = 6.5 K/km`. **The `T_warm` column is an INPUT, not a result**: the
five values are a warmest-month climatology, read off the standard July/January surface-temperature
means, and they are ESTIMATED to the nearest kelvin. *Refuter A was right that a table labelled
COMPUTED with an uncited input reads as if the model produced the agreement (R21). It did not: the
arithmetic is a division, and the inputs are chosen. M11 extends to fitting them.*

| Latitude | Warmest-month mean (INPUT, ESTIMATED) | Snow line, this model (COMPUTED) | Earth's real snow line |
|---|---|---|---|
| 0° | 299 K | 3 977 m | ~4 800 m |
| 23° | 303 K | 4 592 m | ~5 000 m (the subtropics) |
| 45° | 293 K | 3 054 m | ~2 700–3 200 m (the Alps) |
| 60° | 288 K | 2 285 m | ~1 500 m |
| 70° | 283 K | 1 515 m | ~1 000 m |

**The shape reproduces — highest in the subtropics, falling toward the poles, which is the real
curve's distinctive feature — and the absolute is within about 20 to 50 %.**

> **Lapse rate, in plain words.** Air cools as it rises, because it expands. How fast it cools is the
> gravity divided by the air's heat capacity, slowed down by however much water the rising air gives
> up. A low-gravity world's air cools slowly with height, so its snow line sits far higher. *Game
> example: the pilot climbs a 3 000 m ridge at 40° north on the home planet expecting snow and finds
> bare rock, because on this world the environmental lapse rate is under 3 K/km and the snow line at
> that latitude is near 5 000 m. The player learns the world's gravity by looking at it.*

**And `c_p` is not one number for every world.** `Atmosphere.mean_molecular_weight`
(`crates/physics/src/taxonomy.rs:882`) says a body may carry a hydrogen–helium envelope, where `c_p`
is about 14 000 J/(kg·K), not 1 005. `c_p` is therefore **derived from the mean molecular weight**:
`c_p ≈ (7/2)·R_gas/μ` for a diatomic gas. **That fact must therefore be IN the record**, and §5.4
carries it — *refuter B showed that revision 2's seven-integer record could not compute this at all
(B2-5), and that is why the record grew.*

#### ★ SHAPE or COVER: which of these operators the seed may decide

*Refuter B asked the question revision 2 skipped (B2-13): SL10 lets the client derive what
`(seed, address)` decides and forbids it deriving live state. The owner asked for WEATHER. So which
of L5's operators are the world's SHAPE, and which are its live COVER?*

| L5 operator | SHAPE or COVER | Why |
|---|---|---|
| glacial carving (U-valleys, cirques) | **SHAPE** | a carved valley stays for ten thousand years; the seed may decide it |
| the coastal notch, the beach, the sea cliff | **SHAPE** | the same |
| the dune field's ridges | **SHAPE** | the field's alignment and wavelength are climate, not weather |
| the rain shadow's effect on the BIOME | **SHAPE** | it decides which vegetation the skeleton places, not what falls today |
| **snow on the ground** | **COVER** | the owner asked for weather; snow moves with the season and the storm |
| cloud, rain, wind of the hour | **COVER** | live state by definition |

**So `z_snow` decides a BIOME boundary (where the world is glaciated, which is shape), and it does
NOT decide the white you see on a ridge.** That white is `Stratum::Snow` as a COVER, a one-hop diff
from the owning realm, and **§5.5 hands it to domain 05 with the rest of the weather.** *A snow line
that never changes is a painted-on cap, and refuter B is right that the report cannot both hand
weather away and use a fixed snow line to answer "snow-capped ridges".*

#### The circulation bands, from the spin, with the law named and its reading questioned

> **Hadley cell and Coriolis, in plain words.** Warm air rises at the equator, moves toward a pole,
> cools, and sinks — that loop is a Hadley cell. The planet's spin bends the moving air sideways, and
> that bend is the Coriolis effect. A fast spinner bends it more, so its loops are narrow and it has
> many wind bands; a slow spinner has one wide loop per hemisphere.

The law is **Held & Hou (1980)**, which gives the poleward edge of the Hadley cell:

```
      φ_H  ≈  sqrt( 5 · g · H · Δθ / (3 · Ω² · a² · θ₀) )         [radians, small-angle]
```

**★ The reading matters, and refuter B is right that revision 2 read it wrong (B2-14).** Held & Hou
give that expression as the ANGLE in radians, under a small-angle approximation. Revision 2 tested
it as `sin φ_H`, because `Gf` grants no arcsine. COMPUTED with `H = 15 km`, `Δθ/θ₀ = 1/3`:

| Spin | the expression's value | read as `sin φ_H` (revision 2) | read as `φ_H` in radians (the source) |
|---|---|---|---|
| 1/100 of Earth's (Venus-like) | 61.5 | 90° (clamped) | 90° (clamped) |
| half Earth's (a 48 h day) | 1.231 | **90°** | **70.5°** |
| **Earth (24 h)** | **0.615** | **38.0°** | **35.3°** |
| twice Earth's (12 h) | 0.308 | 17.9° | 17.6° |
| four times (6 h) | 0.154 | 8.9° | 8.8° |

**The substitution is nearly free at fast spin (0.3 % at twice Earth's) and wrong at slow spin (a
20-degree band edge on a 48-hour day).** Earth's real Hadley cell reaches about 30°, so the source's
own reading is 18 % wide and revision 2's was 27 % wide.

**The recommendation.** Keep the fenced test — `|dir · pole_axis| < s_H` is a dot product against one
`sqrt`, and no sine is ever taken — but define `s_H` as **the sine of the calibrated edge**, with the
small-angle-to-sine conversion folded into the calibration constant that **M11 fits on Earth, Mars
and Titan**. Then the fence costs nothing and the source is not misread.

**★ And the "cells per hemisphere" column is DELETED.** *Refuter B is right that it had no derivation
anywhere (B2-14): the numbers 1, 1, 2, 5, 10 came from nothing, and §4.9's biome layer then consumed
them.* What the biome legitimately reads is the Hadley EDGE — one latitude, wet inside it near the
equator, dry at its poleward rim — which is the trade-wind and desert-belt structure the owner will
recognise. **The number of bands beyond the first cell is a weather question and goes to domain 05
with Q5.**
#### The rest of L5

- **Orographic lift and the rain shadow.** Air forced up over a range cools, drops its water on the
  windward side, and arrives dry on the lee side. The test is `wind · ∇h_macro`, where `∇h_macro` is
  the affinity gradient L2 already returned plus the belt profile's analytic gradient (§4.2, §4.3), so
  it costs nothing new here. Positive is windward and wet; negative is lee and dry. **It resolves the
  MACRO relief only** — a range, not a hill — which is exactly the scale a rain shadow works at.
- **The wind direction without an angle.** The surface wind is `w = p·c₁ + (d × p)·c₂`, normalised —
  where `p` is the pressure-gradient tangent, `d × p` is its perpendicular on the sphere, and `c₁`,
  `c₂` come from the Coriolis parameter `f = 2Ω·(dir · pole_axis)`. A cross product, a dot product and
  a `sqrt`. No angle anywhere.
- **Glacial carving.** Above the snow line, ice cuts a **U-shaped** valley instead of a **V**, with
  cirques at the heads. One switch on L4a's shaping arithmetic. Near zero cost.

  > **Cirque, in plain words.** The bowl a glacier digs at the head of its own valley: steep walls on
  > three sides and a flat floor, often with a small lake. *Game example: the pilot climbs to the top
  > of a glaciated valley on the home planet and finds an amphitheatre of rock with a tarn in it,
  > instead of a valley that simply narrows to nothing.*

- **Aeolian dunes.** Where an atmosphere exists and liquid does not, a transverse ripple field aligned
  to the wind vector. One anisotropic noise.
- **The coastal notch.** Wave action flattens a narrow band around the water surface, which is why
  real coasts have beaches and sea cliffs. `h → water + f(h − water)`, compressing a band whose width
  is **the significant wave height's own scale**, which scales with the wind speed and the fetch —
  both of which the climate layer holds. Stated as a derivation, not a constant.

**Cost, ESTIMATED:** 30 ns per column for all of L5 together (the macro gradient is L2's and L3's and
is not double-counted).

### 4.6 ★ L6 — the octaves: RE-ANCHORED, BUDGETED and LADDER-CAPPED

*This section is the most changed in revision 3, because refuter A's blocker (R1) and refuter B's
blocker (B2-4) both land here, they are the same finding, and they are right: revision 2 moved the
coarsest wavelength and never said what happened to the amplitudes. Under the shipped rule
(`body.rs:176-184` re-normalises the amplitudes so their sum is the relief) that single omission
multiplies every amplitude-to-wavelength ratio by eight and makes the whole planet a 15° slope
everywhere — before any roughness change. Refuter A computed 56° for the recommended pair. The
arithmetic is right and the cure is a rule that was missing.*

#### 1. THE OCTAVE BUDGET: the sum stops being the relief

> **THE BUDGET RULE.** The octave sum is no longer "the body's relief". It is **the texture below the
> macro layers' own scale**, and it is anchored, not drawn: **the coarsest octave of the re-anchored
> table keeps the amplitude the octave law already gives at that wavelength**, and every finer octave
> follows the roughness from there. `A_oct` — the sum — is then DERIVED.

On the home planet the shipped table's octave 3 sits at exactly 50 km and carries **781.28 m**
(§1.1). So the re-anchored table starts at 781.28 m, and COMPUTED at `k = 0.62`:

```
   A_oct = 781.28 × (1 − 0.62¹⁵)/(1 − 0.62) = 2 054 m          (against today's 14 305 m)
```

**Read that number twice.** The octave sum's contribution to the body's relief falls by seven times.
That is not a loss of relief: **L1's isostatic step (6 476 m loaded), L3's belt uplift (bounded by
`h_max`) and L3's trench now carry the kilometres**, and the noise carries the texture. It is also
why §9.3's band does not explode: the band is sized on the sum of DRAWN amplitudes, and this term
shrinks while the macro terms appear.

**The physical bound on `A_oct`, so it is not a free parameter.** The random texture may not rival the
crust's own contrast, or noise hills stand as tall as continents. So `A_oct` is bounded by the body's
isostatic step: **`A_oct ≤ 0.5 × step`**, which on the home planet is 3 238 m, and the anchored 2 054 m
sits inside it. The anchor is the recommendation and the bound is the fence. *The `0.5` is a
CALIBRATION and this report says so; **M16** owes the picture that fixes it.*

#### 2. The table is re-anchored: 50 km at the top, 2 m at the bottom

COMPUTED, from the loop at `body.rs:169`:

| Coarsest wavelength | `SHORT_WAVE_M` | Octaves | Finest wavelength | Octaves kept at rung 11 |
|---|---|---|---|---|
| 400 km (today) | 30 m | 14 | 48.83 m | 3 |
| 50 km | 30 m | **11** | 48.83 m | **1 — the clamp arm fires** ✗ |
| **50 km** | **2 m** | **15** | **3.05 m** | **4** ✓ |
| 50 km | 1 m | 16 | 1.53 m | 5 — but under Nyquist (§1.6) |

*A correction of wording, from refuter A (R24): with a 50 km cap, every body above a 200 km radius
sits exactly ON the cap, because the long wave is `radius × [0.25, 0.5)` clamped
(`body.rs:151-153`). So the coarsest wavelength is a CONSTANT for nearly every body, not a per-body
draw. Today's 400 km cap does the same above a 1 600 km radius, so this is not new — but revision 2
called it a draw and it is not one.*

#### 3. ★ THE LADDER CAP: the half-cell promise becomes true by construction

§1.3c measured that today's table breaks the ladder's own bound at six rungs of twelve. The cure is
one clamp, applied while the table is built, from the fine end upward:

> **THE LADDER CAP.** Build the amplitudes from the finest octave upward. The `L`-th finest octave is
> the one that disappears at rung `L`, so the sum of the `L` finest may not exceed **half a cell at
> rung `L`**. Cap each amplitude at what that budget leaves:
> `amp(i) ← lesser(amp(i), half_cell(L) − Σ finer amplitudes)`.

It is `Gf::lesser` and a subtraction, once per octave, inside `from_seed`. COMPUTED, the home planet,
`k = 0.62`, the anchored budget:

| Octave | Wavelength | Amplitude | Capped? |
|---|---|---|---|
| 0 | 50 000 m | 781.284 m | — |
| 4 | 3 125 m | 115.445 m | — |
| 8 | 195.31 m | 17.059 m | — |
| 10 | 48.83 m | 6.557 m | — |
| 11 | 24.41 m | **4.000 m** | yes, from 4.066 |
| 12 | 12.21 m | **2.000 m** | yes, from 2.521 |
| 13 | 6.10 m | **1.031 m** | yes, from 1.563 |
| 14 | 3.05 m | 0.969 m | — |
| | | **`A_oct` = 2 053 m** | (2 054 uncapped) |

```
   rung:        0     1     2     3     4     5     6     7     8     9    10    11
   half cell: 0.5   1.0   2.0   4.0   8.0    16    32    64   128   256   512  1024   m
   dropped:   0.0   0.97  2.0   4.0   8.0  14.6  25.1  42.2  69.7 114.1 185.7 301.1   m
   verdict:    ok    ok    ok    ok    ok    ok    ok    ok    ok    ok    ok    ok
                              (today: OVER at rungs 6, 7, 8, 9, 10 and 11)
```

**Three octaves lose a total of 1.1 m of amplitude and the ladder's promise becomes exact.** The cap
costs nothing at run time and it is what lets §6.2 state the SL8 bound as a construction rather than
as a hope.

#### 4. The amplitude is modulated DOWNWARD, and only by rung-independent layers

*Refuter B asked the question revision 2 never answered (B2-3): what is the modulator's RANGE? Both
readings the refuter tried break something — a multiplier above one explodes the band, and a
constant-sum redistribution breaks the ladder bound. **The answer is the third reading, and the
budget rule is what makes it available:** the table above IS the rough state, and the modulator only
ever scales down.*

```
   amp_effective(o)  =  amp(o) · r(column),      r ∈ [r_min, 1],   r_min ESTIMATED 0.10
```

- **`r = 1` is a mountain flank**: 18.9° RMS at a metre, 9.6° at thirty (§1.3).
- **`r = 0.13` is a flood plain**: about 2°, which is today's whole world.
- **The band is untouched**, because `r ≤ 1` and the band is sized on `Σ amp × r_max = Σ amp`.
- **The ladder bound is untouched**, because `dropped_bound × r_max = dropped_bound`.

> **THE MULTIPLIER RULE (binding, unchanged from revision 2).** A multiplier may read only layers that
> are evaluated at EVERY rung. So `r(column)` reads **L1, L2 and L3's profile** — never L4's shaping
> and never L5's operators, which have cut-off rungs. Then `r` is rung-independent by construction.

**What the modulation buys.** It is what lets a flood plain and a cliff exist on the same planet.
Without it the choice is a 2° world or an 18° world, and both are wrong everywhere except in one
place.

**Cost, ESTIMATED:** one extra octave against today's table, at the MEASURED 13 ns per octave. The
modulator is one multiply per octave and is counted in L4a's 15 ns.
### 4.7 The weights: one pipeline, many kinds of body, no kind branch

**HR3 forbids matching on a kind in a feature.** So the recipe never asks "is this a moon?". It asks
each layer for its WEIGHT, and every weight is a number derived from the body's physical facts.

| Weight | Derived from | Zero when |
|---|---|---|
| `w_plates` (tectonic vigour) | mass, radius, age → internal heat; a body needs a convecting mantle | small, cold or old bodies → **one-plate (stagnant lid)**, so no L1 boundaries and no L3 belts |
| `w_rivers` | **atmospheric pressure > 0** and `t_eq_k` in the liquid band and water inventory > 0 | airless or frozen or dry bodies |
| `w_craters` | `age × (1 − clamp(w_plates + w_rivers + w_aeolian, 0, 1))` | an active earth-like body erases its craters |
| `w_glacier` | the surface fraction above `z_snow` | a hot body |
| `w_aeolian` | **atmospheric pressure > 0** and `w_rivers` = 0 | an airless body, or a wet one |
| `w_terrain` | **the `solid_surface` flag of `BodyFacts`** | **a gas giant or an ice giant** |

> **Stagnant lid, in plain words.** A rocky body's outside can be one unbroken shell instead of a set
> of moving plates: the inside still churns, but the crust above it never breaks and never slides.
> Venus and Mars are like this today; Earth is not. *Game example: the pilot flies over a small moon
> of the home system and finds no ranges at all, only craters and their debris — because the moon
> drew a stagnant lid, `w_plates` is zero, and L1 and L3 never run.* *(Refuter B asked for this word
> to be explained, B2-21.)*

> **Size-frequency distribution, in plain words.** How many craters of each size a surface carries:
> many small ones, few big ones, in a fixed proportion. Writing `N(>D) ∝ D⁻²` says "make the diameter
> ten times smaller and you find a hundred times as many". *Game example: the pilot walks a moon's
> plain and steps over dozens of metre-wide pits, passes a few hundred-metre bowls in an hour, and
> sees one ten-kilometre basin on the horizon.*

**★ `w_terrain` now has an input.** *Refuter B was right that revision 2 left this OWED and then let
D13 rest on it (B2-22).* §5.4's record carries a `solid_surface` flag — one bit, drawn by the forest
from the body's own class, which is where the class lives. So `from_seed` returns `None` for a body
with no solid surface, D13's answer has an input, and the ice-giant coverage arm can be pinned as
"this real body of THE world returns `None`", which is a pinnable fact.

**How many plates? — still open, and now bounded.** The physical driver is the ratio of the
lithosphere's thickness to the body's radius: a thin lithosphere on a big body breaks into many
plates. Earth has 7 major and about 15 total at `R = 6 371 km`; Venus, hotter and one-plate, has 1.
**No body in the literature carries more than about 40 plates, so `PLATES_MAX = 40`** (§4.0). The law
itself stays Q3.

**A zero weight skips the layer's cost**, so a rocky moon's column pass is CHEAPER than the home
planet's, not more expensive. That is the right direction for SL9 (a system with six hundred bodies).

**Craters.** A Poisson process in (position, diameter) with a size-frequency distribution
`N(>D) ∝ D⁻²` and a profile of rim, bowl and ejecta. It is **exactly the machinery the crate already
has for tube carvers** (`crates/terrain/src/carve.rs`: a bounded draw of features per region, tested
per cell), and it obeys the scan law of §4.0 like every other bounded list. The simple-to-complex
transition diameter scales as `1/g`, so a low-gravity body's craters stay bowl-shaped to larger
sizes. Saturation is age: an old surface reaches a density where new craters erase old ones, which is
a cap on the draw.

**A body with no solid surface.** COMPUTED, the ladder's `N_MAX = 2²⁶` caps the radius at
**42 723 km**, so Jupiter (69 911 km) and Saturn (58 232 km) are **already refused** by
`Ladder::for_radius`. The gap is the **ice giants**: Neptune (24 622 km) and Uranus (25 362 km) are
accepted today and would get a voxel surface. `w_terrain` closes it: `from_seed` returns `None`, and
the realm draws a **banded cloud shell** with a look and no cell grid, so no chunk is ever asked for.
§13 D13.

### 4.8 ★ What this stack CANNOT make, stated plainly

`height_m` returns ONE radius per direction, and every layer in §4 modifies that one number. Here is
the honest inventory, against the reference vista, in three groups.

**Group 1 — what the height field CAN now make.**

| The reference picture holds | Made by |
|---|---|
| a range at 30–50 km with a snow-capped crest | L3 + L5 |
| a carved valley with a flat floor and a sharp divide | L4a's multifractal weight + L4d's fill |
| a steep cliff face | the ridged fold plus the shaping curve: a near-vertical single-valued step |
| a scree slope at its foot | L4d's angle-of-repose limiter |
| **a benched cliff, a mesa rim, a cuesta** | **L4e's hardness term** — *new in revision 3, and refuter B was right that it was the cheapest missing win (B2-12)* |
| a coast with a shelf and a beach | L2's affinity field + L5's notch |

**Group 2 — what a single-valued field can NEVER make.**

| The reference picture holds | Why not |
|---|---|
| a **rock pillar / hoodoo** | it needs air on all sides at one height |
| an **arch** | it needs rock over air over rock |
| an **undercut cliff** | the overhang is a second surface on one radial |
| a **sea stack** | a pillar in water |

> **Single-valued, in plain words.** For every direction the field gives exactly one ground height. A
> pillar needs three answers along one radial: rock, then air, then rock again. *Game example: the
> pilot walks toward the hoodoo that doc 01 calls "an orienter a player names and returns to". Under
> this stack it is not there. What is there is a steep-sided knob, because a knob is single-valued and
> a hoodoo is not.*

**The route exists and it is already in the crate.** `carve.rs` writes density per CELL, independent
of `h` — a three-dimensional term, which is exactly what an arch and an undercut need. **This report
does not scope it**; it is **D14**.

**Group 3 — what is in the picture and belongs to nobody yet.** *Refuter B is right that the owner
counts everything in the frame (B2-20), so the list is stated rather than left implicit.*

| In the picture | Who owns it |
|---|---|
| **a river, a lake, a flood plain** | **DOMAIN 03**, with §4.4b's three conditions |
| **roads, terraced fields, walls, a bridge** | **NOBODY TODAY.** They are culture, not geology. No layer here makes one and no domain in this investigation is named for them. |
| **the blue distance haze** | the renderer, from an atmosphere fact — §5.4's record carries the surface pressure and the molecular weight, so the input exists |
| **moraines, fjords, hanging valleys** | L5's glacial switch, at the level of "a U-valley and a cirque". A moraine is a deposit and needs L4d with a glacier's own path, which is UNSCOPED |
| trees, grass, decoration | ruling V4: art assets placed by a server skeleton from the biome. The layout makes the skeleton believable and draws nothing |
### 4.9 L7 — the biome, rebuilt from the physical facts

Today `biome_at` (`crates/terrain/src/height.rs:40-66`) reads: the height above the sea,
`dir[POLE_AXIS]` as a latitude, and two slow noises. There are four biomes
(`crates/terrain/src/strata.rs:114-131`). That serves none of the owner's words.

The replacement reads the facts the layout now holds, and every input is already computed by an
earlier layer, so it costs almost nothing:

| Input | Where it comes from | The owner's word it serves |
|---|---|---|
| insolation at this latitude | `BodyFacts.insolation` × the profile in `dir · pole_axis` | *position, trajectory* |
| the seasonal swing | `BodyFacts.obliquity` and `BodyFacts.eccentricity` × the same latitude | *position, trajectory* |
| elevation above the LOCAL water surface | L4c's `water_m`, not the sea sphere | — |
| the temperature at that elevation | `Γ_env = (g/c_p)·w` from §4.5 | *size, gravity* |
| **the Hadley edge** (inside it wet, at its poleward rim dry) | Held–Hou from `BodyFacts.day_length` | *spin* |
| the rain shadow | `wind · ∇h_macro`, free from L2 and L3 | *spin* |
| continentality | L2's affinity field | — |

> **Continentality, in plain words.** How far you are from open water, in effect. The sea evens out
> temperature: a coast is mild in both directions, and the middle of a continent bakes in summer and
> freezes in winter with little rain. *Game example: the pilot walks inland from the home planet's
> beach. The forest thins into steppe and then into dry pan, with no range in the way — that is
> continentality, and L2's affinity field already knows it.* *(Refuter B asked for this word,
> B2-21.)*

**The biome count grows** from four, because a temperature-by-humidity grid with a wind and an
elevation now distinguishes more than four things. That is a `Biome::ALL` change, a `StrataTable::at`
change and a world-identity bump — **the same door as everything else in §12**, so it costs nothing
extra if it lands in the same slice. §13 **D12** asks the owner for the biome list.

**Cost, ESTIMATED:** 20 ns per column (every input is already in hand; the biome is a small decision
tree over them). The two slow noises of today's function are retired, so the net is near zero.
## 5. The physical facts the layout needs, and how each one crosses the fence

### 5.1 The list

| Fact | Used by | Exists today? |
|---|---|---|
| radius | the ladder, everything | **yes**, an input to `from_seed`, and **the ladder SNAPS it** |
| mass → **surface gravity `g`** | `h_max` (L3), the lapse rate (L5), the crater transition (§4.7), the crust thickness (L2) | `BodyTaxon.mass_kg` (`taxonomy.rs:899`); `surface_gravity_mps2` (`taxonomy.rs:750`) — in a motion crate |
| **spin (day length) `Ω`** | Coriolis, the Hadley edge, the day/night cycle, the dune alignment | **no** (MEASURED, §0) |
| **obliquity, AND the pole axis's azimuth** | seasons, the ice-cap latitude, the true pole axis | **no**; `POLE_AXIS` is hard-wired to `+Z` (`height.rs:30-35`). *One angle cannot orient an axis in three dimensions — refuter B was right (B2-5), so the record carries two.* |
| insolation, `t_eq_k`, Bond albedo | the snow line, the liquid band, the biome field | **yes** — `BodyTaxon` (`taxonomy.rs:902-907`) |
| **atmospheric surface pressure**, **mean molecular weight** | `w_rivers`, `w_aeolian`, `c_p`, the lapse rate, the renderer's haze | **yes** — `Atmosphere` (`taxonomy.rs:881-890`) |
| **water inventory** | the sea level (§7.1), `w_rivers`, the wetness factor `w` | **no**; derivable from `inside_frost` + class + a seed draw |
| **age** | the belt age (L3), crater saturation, `w_plates` | **no** |
| **eccentricity and year length** | the seasonal swing, the biome's seasonality — the owner's word *trajectory* | `OrbitalElements` (`celestial.rs:136-152`), in the motion crate |
| **has a solid surface** | `w_terrain`, D13 | the class knows it; the generator does not |
| star luminosity, orbit | already folded into insolation | **yes** |

**Age crosses; it is not drawn twice.** The age is a fact the forest holds about the body, and it
crosses in `BodyFacts` like every other fact, so the generator never draws a second one.

### 5.2 The two the world does not have: spin and obliquity

Both must be **seed draws under physical constraints**, and the constraints need the orbit:

- **Tidal locking.** A body close to its primary despins until its day equals its year. The timescale
  goes as `a⁶` in the semi-major axis, so it is a sharp threshold, and the orbit is right there in
  `OrbitalElements` (`crates/physics/src/celestial.rs:136-152`). A locked planet has a permanent day
  side and a permanent night side, and that is a MAJOR gameplay fact, not a detail.
- **Obliquity** is drawn from an angular-momentum distribution, and a large moon damps it (Earth's
  23.4° is stable because of the Moon; Mars's wanders). The forest already draws moons.

**Where the draw belongs: in the forest, not in the generator.** The generator may not name a motion
crate (`vd-terrain` never depends on `vd-physics`; the fence is `tests/tests/crate_isolation.rs`), so
it cannot see the orbit. §5.4's last part answers the obvious counter-proposal.

### 5.3 What the obliquity does to `POLE_AXIS`

`POLE_AXIS = 2` (`+Z`) is a constant today, with the doc naming obliquity as "a later slice"
(`height.rs:30-35`). With an obliquity the climate axis is **not** the frame's `+Z`; it is a unit
vector built from TWO angles — the tilt and its azimuth. The biome, the snow line and the Hadley test
all read `dir · pole_axis`, a dot product, instead of `dir[2]`. The change is three lines and two
struct fields, and it is a world-identity bump.

### 5.4 ★ SL6 ASK ONE — a `BodyFacts` record, AUTHORED ONCE, as a new tag

CLAUDE.md states SL6 in full: *"ASK BEFORE NEW DATA CROSSES A REALM BOUNDARY, and before adding a wire
arm. Default NO. … State what data, from which realm to which, why the receiver cannot compute it from
what it legitimately holds, and what doing without costs."*

#### The hole, and the cure revision 3 adopts

1. **The radius is safe because the LADDER SNAPS IT, and the crate proves it.**
   `crates/terrain/src/home.rs:48` is a test named
   `a_look_radius_moved_below_the_snap_gives_the_same_body`: the look radius moved by a millimetre, or
   by a THOUSAND ulps, gives the same body byte for byte. **No new fact has a snap**, and the snap
   unit is huge — COMPUTED, 1 303 m on the home planet, against a drift of order 10⁻⁹ m.
2. **`vd-physics` computes with the arithmetic the fence exists to exclude** (MEASURED, §0: 43 calls
   on 39 lines of `taxonomy.rs`, 11 calls on 9 lines of `celestial.rs`). `mass_kg` itself comes out of
   the Chen–Kipping/Zeng fits, which are `powf`.
3. **The crate forbids this in writing**, at `crates/terrain/src/body.rs:86`: *"no float from an
   unfenced crate can enter the recipe as a body."*

> **ulp, in plain words.** The smallest step a computer's number can take at a given size — one "unit
> in the last place". Moving a value by a thousand ulps moves it by a thousand of those smallest
> steps, which for a planet's radius is under a millionth of a millimetre. *(Refuter B asked for this
> word, B2-21.)*

**What would fail.** SL10 clause 3 is a MEASUREMENT: server-built and client-built chunks byte for
byte on every shipped target. If an x86-64 shard computes `g = 4.684 1…1` and an aarch64 shard
computes `g = 4.684 1…2`, then `h_max = σ_y/(ρ_c·g)` differs, every mountain differs, and the golden
self-check disagrees **between two SERVERS**, before the client is involved.

**★ THE CURE, corrected in revision 3.** *Revision 2 said "quantise every fact to an integer grid,
then a drifting input is HARMLESS". **Both refuters refused the word HARMLESS and both are right**
(R8, B2-10): quantisation does not remove a drift, it makes the drift harmless everywhere except
within one drift-width of a grid boundary, where the two hosts land on different integers and every
mountain on that body moves. Refuter B stated the structural cure and revision 3 adopts it:*

> **THE AUTHORING RULE.** The body's facts are quantised to integers **once, by the realm that owns
> the body**, and every other process — a second server, a client, a re-spun shard — RECEIVES the
> integers and never re-derives them. The integer is the fact. Nothing downstream ever touches
> `powf`.

| Fact | Grid | Why that grid |
|---|---|---|
| surface gravity | whole **mm/s²** | 1 part in 4 700 on the home planet; a mountain ceiling moves under 4 m |
| day length | whole **seconds** | 1 part in 86 400 on an Earth-like day; the Hadley edge moves under 0.01° |
| obliquity, pole azimuth | whole **milli-degrees** | a pole direction moves under 60 m on the home planet's surface |
| insolation | whole **W/m²** | a snow line moves under 10 m |
| equilibrium temperature | whole **milli-kelvin** | a snow line moves under 1 mm |
| atmospheric surface pressure | whole **pascals** | a weight threshold, never a fine term |
| mean molecular weight | whole **milli-units** | `c_p` moves by 1 part in 30 000 |
| water inventory | whole **parts per million of the body's mass** | a sea level moves under a metre |
| age | whole **millions of years** | a belt's age term is smooth over 10⁷ years |
| eccentricity | whole **parts per million** | a seasonal term moves under a kelvin |
| year length | whole **seconds** | the same |

**And the pin test is corrected too.** *Revision 2's M13 moved each fact by a thousand ulps and
required the same body — which passes for any value that is not within a thousand ulps of a boundary,
that is, almost every value. Both refuters said it is a test that cannot fail (R8, B2-10) and they are
right; the project's own standing rule is that a no-drift claim must be a measurement that could have
failed.* **M13 becomes two things:** (a) the authoring rule enforced structurally — only one code path
computes the integers, and a test asserts that the generator never sees a float; and (b) a bench over
1 000 seeds that prints, for every body, the DISTANCE from each fact to its nearest quantisation
boundary, against a floor stated in ulps — plus the still-owed measurement of `surface_gravity_mps2`'s
drift between the shipped targets, which is the same measurement `D-TERRAIN-1`'s G4 leg owes.

#### The ask, in SL6's four parts

- **What data.** ONE new tag beside `TAG_SURFACE`, carrying a fixed record of **thirteen integers**:
  gravity (`u32`, mm/s²), day length (`u32`, s), obliquity (`i32`, m°), pole azimuth (`u32`, m°),
  insolation (`u32`, W/m²), equilibrium temperature (`u32`, mK), surface pressure (`u32`, Pa), mean
  molecular weight (`u32`, milli-units), water inventory (`u32`, ppm of body mass), age (`u32`, Myr),
  eccentricity (`u32`, ppm), year length (`u32`, s), and a flags word (`u32`) whose first bit is
  `solid_surface`. **13 × 4 = 52 bytes of payload; ESTIMATED 80 bytes on the wire** with the tag, the
  length and postcard's framing — inside a 1 200-byte self-look budget (`look.rs:79`) that carries
  about 40 bytes of surface tag today.

  *The record GREW from revision 2's seven integers because refuter B showed that four of the
  report's own layers could not be computed from seven (B2-5): the lapse rate needs the molecular
  weight, `w_rivers` and `w_aeolian` need the atmosphere, the pole axis needs two angles, the owner's
  word "trajectory" needs the eccentricity and the year, and `w_terrain` needs the solid-surface bit.
  The refuter is right on every row, and 28 more bytes is the whole cost.*

- **From which realm to which.** From the realm ITSELF, about ITSELF, to the window of a client already
  subscribed to it. It rides `BodyStmt::SelfLook`, which only a RUNNING realm may send, so a dormant
  realm is still never drawn.

- **★ Who computes it.** *Refuter A was right that revision 2 never said (R23).* **The body's own
  realm computes it**, in its own shard, which already links `vd-physics`, from its own seed and its
  own drawn taxon. It then quantises once and states the integers. **Is that a realm deriving
  something about itself, which SL1 clause 3 forbids?** No, and the distinction is sharp: SL1 forbids
  a child asserting its own PLACEMENT — where it is. A body's mass, its spin and its insolation are
  what it IS, in the same class as a hull stating its rating (the 2026-08-27 movement ruling) and a
  realm stating its own reach (the 2026-09-02 reach ruling). **And SL1 clause 5 says such a fence is
  structural, never remembered**, so the slice owes the fence: the record's type may not name a
  position, a velocity or a frame, and the crate-isolation test already refuses the generator's edge
  to `vd-physics`.

- **Why the receiver cannot compute it.** The client links `vd-terrain` and `vd-seed` and NOT
  `vd-physics` (`crates/client/Cargo.toml:24-26` carries it under `[dev-dependencies]` only;
  `tests/tests/crate_isolation.rs:322` refuses the normal edge). The client holds the body's seed and
  its look radius and nothing else. It cannot derive `g` without the mass.

- **★ What doing without costs — FOUR options, not three.** *Refuter B was right that SL6 says "find
  the local formulation first" and that revision 2 never tried the obvious one (B2-11).*

  | Option | What it costs |
  |---|---|
  | (i) the generator draws `g` from the body's seed | a SECOND gravity that disagrees with the mass the forest drew — the "two temperatures" defect |
  | (ii) the layout stops depending on physical facts | discards the owner's requirement by name |
  | (iii) the client links the motion crate | ruling V12 refuses it |
  | **(iv) move the body's physical DRAW below the fence** | **the option revision 2 never stated.** `vd-seed` or `vd-terrain` owns the mass–radius law and `vd-physics` READS it. Then there is one gravity, both hosts derive it from the seed, no record crosses, no refusal path is needed and M13 is not needed. **The cost:** today's law is the Chen–Kipping/Zeng fit, which is `powf`, so it must be re-expressed — the natural fenced form draws the RADIUS and the DENSITY as bounded seed values and derives the mass by multiplication, which needs no transcendental at all. That is **a world-identity bump for the WHOLE world, not only for terrain**: every body's mass, orbit and insolation move, so every star, every planet and every moon the world has ever drawn changes. |

  **The recommendation stays (i)-free: cross the record (D8), because option (iv) re-draws the whole
  universe and the owner has not asked for that.** But the owner should see (iv), because it is the
  only option that opens no lane at all, and if the world is going to be re-drawn for another reason
  it is the moment to take it. **UNMEASURED: how much of `vd-physics` a fenced mass–radius law would
  touch.**

#### ★ The refusal path

**The rule: a client that does not hold `BodyFacts` for a body MUST NOT derive that body's surface.**
It draws the realm's shipped look and waits. That is one branch beside the generator-tag comparison
(`crates/client/src/chunks.rs:269-271`), and it is the same refusal shape slice 7 already built.
**No default value is ever substituted** — "decode-to-Default is BANNED for Durable kinds" is already
a project convention, and this is the same class of mistake.
### 5.5 ★ SL6 ASK TWO — the weather lane, opened and handed on

*Revision 1 said "Nothing here needs a new wire arm except one" and then admitted in its own open
questions that a live wind field "is LIVE STATE and therefore a diff". Refuter B was right (B-D2): one of
those two statements is wrong, and weather is the owner's own second sentence. The report must open the
question even though it does not answer it.*

**Weather is live state.** It is not a function of `(seed, address)`, so SL10 does not let the client
derive it. It must cross from the planet's realm to a window, and SL6 governs that crossing. **This
report does not design it; domain 05 does.** What this report owes is the QUESTION, in SL6's own form:

- **What data.** UNANSWERED. Candidates: a per-realm statement (one wind vector, one cloud cover, one
  precipitation rate for the whole body — tiny, and wrong for a planet with a rain shadow); or a coarse
  climate grid keyed by a COARSE LADDER RUNG (right, and its byte count is the whole question); or a
  per-chunk field (certainly too much: 3.5 M chunks). **And the SNOW COVER of §4.5 rides this lane**,
  because §4.5 rules it COVER rather than shape.
- **From which realm to which.** From the planet's realm, about itself, to a subscribed window. One
  hop, exactly as `BodyFacts`.
- **At what rate.** UNANSWERED, and it is the number that decides everything. A weather field that
  restates every tick is a per-tick lane; one that restates on change is a diff.
- **Why the receiver cannot compute it.** Because it is live state, by definition.
- **What doing without costs.** No rain, no cloud shadow, no storm, no seasons a player can feel. The
  owner asked for weather by name.

**What THIS report gives weather, at no crossing cost at all:** the spin, the obliquity, the
circulation bands, the lapse rate, the ranges the wind must climb, the ocean map, the elevation and the
slope aspect. Every one of them is derived on both hosts from `BodyFacts` and the layer stack, so the
weather lane only has to carry what CHANGES. **That is the SL6 "find the local formulation first"
answer, and it is why this report belongs before the weather one.**

---

## 6. The ladder: which layer lives at which rung

### 6.1 The rung a macro cell would be

For the home planet, one macro cell of a face-edge `E` grid is `5 263 360 ÷ E` metres, and rung `L` has
`2^L` metre cells. COMPUTED (§3.2's table): a 512-edge grid is rung 13.3; a 1 024-edge grid is rung
12.3; **the ladder's top rung (11, 2 048 m) is a 2 570-edge grid.**

**Two conclusions:**

1. **The ladder's top rung is FINER than any macro grid we would bake.** So a baked macro map could
   never BE a rung; it would always be an interpolated source under the ladder. That is a second shape
   (§3.3 reason 2), and it is the arithmetic reason the closed form wins.
2. **With the closed form the question dissolves.** L1, L2 and L3 are evaluated at EVERY rung at the
   same cost, because they are a function of the direction and not of the rung. The coarse rungs
   therefore ARE the macro map: at the top rung the field is `radius + isostasy + orogeny profile + 4
   octaves`. *(Revision 1 wrote "3 octaves" here and refuter A showed the arithmetic gave 1 with the
   clamp firing (F6). §4.6's re-anchor is what makes it 4, and 4 is COMPUTED, not asserted.)*

### 6.2 ★ The cut-off rung is DERIVED — and the rule is not obeyed today

**The rule:**

> **A layer is dropped at the first rung where its own amplitude falls under HALF A CELL of that
> rung.** Its contribution to the tier-agreement bound is then under half a cell by construction.

```
   rung:      0     1     2     3     4     5     6     7     8     9    10    11
   cell:      1     2     4     8    16    32    64   128   256   512  1024  2048  metres
   half:    0.5     1     2     4     8    16    32    64   128   256   512  1024  metres

   a layer whose amplitude here is A is dropped at the first rung with half-cell > A
```

**★ The premise revision 2 stated is FALSE, and refuter B measured it (B2-7).** Revision 2 introduced
the rule as *"the one the octaves already obey"*. They do not. COMPUTED (§1.3c): today's octave table
exceeds half a cell at **rungs 6, 7, 8, 9, 10 and 11**, by up to 1.43 times — 1 469 m of movement
against a 1 024 m cell at the top rung. The refuter found four of the six; the replication finds six.
**The false premise is withdrawn, and §4.6's LADDER CAP is what makes the rule true**, by clamping
each octave's amplitude to what the half-cell budget at its own rung leaves. After the cap the same
table passes at every rung (§4.6's box).

| Layer | Its own amplitude | Dropped at the first rung where |
|---|---|---|
| L1 plates | the full isostatic step, kilometres | **never** |
| L2 isostasy | kilometres | **never** |
| L3 belt profile | up to `h_max`, kilometres | **never** |
| L3 ridge lines | the belt's ridge amplitude, a per-body constant | half-cell exceeds it |
| **L4a the multifractal weight** | it MULTIPLIES the octaves, so it lives and dies with them | **never dropped as a layer; it costs nothing once the octaves are gone** |
| L4d fill and talus | the fill depth | half-cell exceeds it |
| L4e the hardness term | the bench height, a per-body constant | half-cell exceeds it |
| L5 notch, dunes | the operator's band width | half-cell exceeds it |
| L6 octaves | each octave's own amplitude, now CAPPED | as today, and now exactly |

*Refuter A and refuter B both found the same contradiction between this table and §9.5 (R4, B2-16):
revision 2's §6.2 said L4a is never dropped and its §9.5 dropped it to make the top rung look
cheap. **Both are right and the contradiction is gone**, because L4a is no longer a gradient with a
cost of its own: it is one multiply per octave, so at rung 11 it costs what four octaves cost and
nothing more. §9.5 is recomputed from this table with no arm left out.*

**Every cut-off is a per-body number the recipe computes in `from_seed`, and the bound it adds is
under half a cell.** So `dropped_bound_m` extends by a sum of per-body constants and the property
test at `crates/terrain/src/height.rs:83-110` extends without changing shape.

**★ The SL8 claim, stated honestly.** *Refuter A is right that "under half a cell, therefore
invisible" mixes two resolutions (R18).* Half a cell at rung 8 is **128 metres**. Whether 128 m is
invisible depends entirely on the tier rule that decides WHICH rung is drawn at WHICH distance, and
that rule is slice 8's — today the world runs one rung per realm behind a dev flag, which is
`D-TERRAIN-3`, marked 🟥 at `docs/design/DEFERRED.md:7775`. **So the honest statement is: the cut-off
adds under half a cell at its own rung, and slice 8's tier rule must keep a cell near a pixel for
that to be invisible. The measurement is owed, and it is slice 8's, not this report's.**

### 6.3 The cube-face seam: solved by construction, and the accelerator must not break it

`height_m` takes a **unit direction** (`crates/terrain/src/height.rs:17`), so the field knows nothing
of faces and the seam cannot exist. **Every new layer must obey the same rule: evaluate on `dir`,
never on `(face, a, b)`.**

**★ And revision 3 keeps that true where revision 2 broke it.** *Refuter A found the breach (R10):
revision 2's hydrology cell IS a ladder cell, a ladder cell IS `(face, a, b)`, and reaching one from
a direction runs `unbend` — four Newton steps with a division each (`crates/seed/src/bend.rs:47-66`,
`INVERSE_STEPS = 4`) — whose own doc says the step count is part of the generator's world identity.
So revision 2's layout would have moved every plate, river and biome if `INVERSE_STEPS` ever changed,
while §6.3 asserted the opposite.* **§4.0's SCAN LAW removes the coupling at the root:** no layer needs
a cell address, because the definition is the full fixed-order scan and a harvest may not change the
answer. **The bend is irrelevant to the layout again, and now it is irrelevant by construction rather
than by assertion.**

**The test (M6), two legs:** (a) sample columns either side of a cube edge and require bit-equal
heights; (b) **assert that a harvested answer equals the full-scan answer, bit for bit**, over a
sample of columns and over every bounded list the body carries.
## 7. The sea, the ocean floor, and the coast

### 7.1 The sea level solves for a water inventory — on the FIELD THAT IS DRAWN

*Revision 2 built a fixed-sample hypsometry, which was the right shape, and then evaluated the wrong
field with an unwritten sampler. **Both refuters found both faults** (R12, B2-15), and both are
right.*

> *Fault 1 — the field. Revision 2 evaluated `h_smooth` (L1 + L2 + L3's profile) and called the
> result the hypsometric curve. The octave sum alone carries a standard deviation of 2 284–2 297 m
> (COMPUTED here and independently by refuter A), and L3's ridged fold adds more. A level solved on
> the smooth field alone puts an unknown area of land under water. The report then quoted a 0.78 %
> SAMPLING error, which is right and measures the wrong thing: the MODEL error was unstated and
> larger.*
>
> *Fault 2 — the sampler. "Draw 4 096 directions from the body's own seed" needs, for a uniform
> direction on a sphere, either a Gaussian (`ln` and `cos`, both refused) or a rejection loop with a
> data-dependent trip count and a new HR5 branch. Revision 2 wrote neither and §10.1's fence table
> listed the hypsometry as fenced anyway. That is the exact class of omission the report itself
> caught in revision 1.*

**The corrected recipe, and every step is fenced and order-fixed.**

1. Draw a **water inventory** `V_water` from the body's facts: a body that formed beyond the frost
   line is water-rich; `inside_frost` is already computed (`taxonomy.rs:573,986-994`).

2. **★ The sample directions come from the LADDER, not from a random sampler.** Take the cell centres
   of a coarse rung — 26 cells per face edge gives `26² × 6 = 4 056` directions — through the bend the
   crate already has. Every one is a pure function of the address, the order is the address order, the
   trip count is fixed, there is no rejection loop and there is no new branch. **The bend's own
   equal-area design is why the samples are near-uniform**, and each sample carries its cell's own
   area factor as a weight, which is a fenced expression in the cube-sphere Jacobian. *This is the
   one place in the report where a cell address is used on purpose, and it is lawful because it is a
   FIXED list drawn once inside `from_seed` — not a per-column lookup, so the scan law of §4.0 is not
   engaged.*

3. **Evaluate the FULL field** at each sample — `h_smooth` plus the ridged fold plus the octave sum —
   because that is the surface the water actually stands on. COMPUTED: one full evaluation is
   `185 ns (today's octaves, MEASURED) + 335 ns (the stack, §9.1) = 520 ns`, so 4 096 of them cost
   **2.13 ms per body**, once, inside `from_seed`.

4. **Bisect** for the level that holds `V_water`, with the **water load** of §4.2 inside the loop: a
   sample below the trial level contributes `h − (level − h)·ρ_w/ρ_m`, which is one multiply, so the
   loading and the level are solved together and no outer iteration is needed. A **fixed** 24 steps —
   never "until converged", which is a float comparison and a drift class. COMPUTED: 24 halvings of a
   40 km range reach **2.38 mm**.

**Determinism:** a fixed direction list from the ladder, a fixed evaluation order, a fixed step count,
no sort and no tie. **Accuracy:** the sampling error of an ocean fraction near 0.5 from 4 096 samples
is **0.78 % of the surface** (COMPUTED), and the model error is now zero by construction, because the
solve reads the field that is drawn.

**★ Where it runs, which revision 2 never said.** *Refuter B is right that this is an SL8 hazard
(B2-17): `crates/client/src/chunks.rs:266-289` calls `BodyDefinition::from_seed` INLINE on the
message path, once per realm, when a surface statement arrives. COMPUTED: one body is 2.13 ms, and a
star system stating a planet, seven siblings and twenty moons in one frame is **60 ms** — four
dropped frames at 60 Hz, and `tick-hitch` is a named seam.* **So `from_seed` moves to a worker on
both hosts, and the realm's look is drawn from the shipped statement until the body is ready.** It is
a small change and it must be in the slice.

**The cost this adds to `from_seed`.** Today the boot self-check (8 chunks) costs 13.5 ms (MEASURED,
`slice_05_generator.md:386`). Adding 2.13 ms per body is ESTIMATED at 16 % of that, and M10 measures
it.

**The consequence the owner will see.** The ocean fraction stops being a lottery. A water-rich
earth-like planet gets 60–75 % ocean; a dry one gets basins with no sea in them; an ice world gets its
ocean under a crust.

**One term is knowingly missing.** *Refuter B is right (B2-15): a basin that fills into a LAKE holds
water the ocean inventory has already spent, so the balance counts it twice.* This report has no
lakes — they are domain 03's, with the river network — so **the double count is zero today and
becomes real the moment domain 03 lands.** The fix is one subtraction in the inventory, and it is
stated here so domain 03 inherits it rather than discovering it.
### 7.2 The ocean floor

Not one flat plane. From L1–L3, at no extra machinery:

- the **abyssal plain** at the oceanic isostatic level — COMPUTED **6 476 m** below the continental
  surface once the water column stands on it (§4.2), against 4 455 m dry;
- **abyssal hills** — a low-amplitude, strongly anisotropic octave set aligned to the spreading
  direction, because sea floor is made in stripes at a ridge and carries them for its whole life;

  > **Abyssal hills, in plain words.** The deep sea floor is not smooth: it is corrugated in long
  > parallel ridges a few hundred metres high, all running the same way, because the crust was made
  > at a spreading ridge and cracked as it cooled. *Game example: the pilot takes a submersible along
  > the home planet's ocean floor and crosses ridge after ridge, all parallel — and their direction
  > tells the pilot which way that plate was made.* *(Refuter B asked for this word, B2-21.)*

- the **mid-ocean ridge** along every divergent boundary, 2–3 km above the plain because young crust is
  hot and less dense (isostasy again, with `ρ_c` a function of the crust's age);
- **trenches** at convergent boundaries;
- **seamounts** along hotspot tracks (§4.3), some breaking the surface as islands.

### 7.3 The coast

The coast is where L2's continental affinity crosses the water surface, so it exists as soon as L2
does — and because the affinity is a FIELD, most coasts are passive margins (§4.2). Two operators
finish it:

- **the shelf**: the platform's flooded edge, nearly flat, then the **continental slope** falling to the
  abyssal plain — both are in the L2 affinity profile;
- **the coastal notch** (§4.5): the wave-height compression band, which turns a slope-crossing-water
  into a beach where the slope is gentle and a sea cliff where it is steep.

**The task asked whether a coastline needs erosion. It needs an EROSION SIGNATURE, not an erosion
SIMULATION.** The notch is that signature, and it is a polynomial.

---

## 8. Cost, memory, determinism and cache — the four options side by side

### 8.1 The comparison

| | (a) plate simulation | (b) uplift + fluvial | (c) pure closed form | **(d) the layer stack** |
|---|---|---|---|---|
| Believability | highest | highest for valleys | low–medium | **high** |
| Boot cost per body, client | 0.19–12 s (COMPUTED §3.2, the 512-edge to 2 048-edge range on 14 threads) | same | 0 | **2.13 ms** (§7.1's hypsometry, on a worker) |
| Memory per body | 6.3–403 MB | same | ~0 | **2.2 KB of plate list, and NOTHING ELSE** |
| Per-column cost | 0 after the bake | 0 after the bake | **185 ns MEASURED** (`slice_05_generator.md:372-386`) | **+335 ns (§9)** |
| Addressable (one chunk alone) | **no** | **no** | yes | **yes** |
| New drift classes | 4 (§3.3) | 4 | 0 | **0** — the scan law of §4.0 fixes the order, and nothing is accumulated over a set |
| Cache needed | disk or re-bake | disk or re-bake | none | **none. A harvest is optional and cannot change an answer (§4.0)** |
| Is it the ladder? | no (a second shape) | no | yes | **yes** |
| HR5 burden | high (iterative branches) | high | low | **medium** |
| SL10 (both hosts identical) | at risk (order) | at risk | held | **held, IF §5.4's authoring rule lands** |

*Three rows are corrected against revision 2, and each correction was earned by a refuter.*

- *Option (c) was priced at "~90 ns". Today's field IS a closed-form octave sum and slice 5 MEASURES
  it at **185 ns** per column. Refuter B was right (B2-19), and understating the option you reject is
  the same unfairness the report charged revision 1 with.*
- *The memory row said "4 KB plates + an ESTIMATED 32–320 KB of harvested hydrology cells", which was
  a cache the report elsewhere said it did not need, in a document that also priced its build at
  nothing. Refuter A and refuter B both attacked it (R2, B2-2). **The harvest is gone as a
  requirement**, so the row is the plate list and nothing else.*
- *The drift-class row said 0 while L4b needed an accumulation order and a frontier tie-break.
  Refuter B was right that it was false as soon as rivers were on (B2-1). **The rivers are out of this
  report**, and the three conditions of §4.4b are exactly what keeps domain 03's answer at zero too.*
### 8.2 Scaling with body radius

| | a 400 km moon | the home planet, 3 351 km | a 12 000 km super-earth |
|---|---|---|---|
| ladder `N` at rung 0 (`N = R·π/2`, snapped; `N_MAX = 2²⁶`) | ~628 000 | 5 263 360 | ~18.8 M — accepted |
| `g` | ~0.56 (at ρ = 5 000) | **OWED, M2**; 3.68 at Mars's density, 5.17 at Earth's | **28.9** (COMPUTED, Zeng) |
| ceiling `σ_y/(ρ_c·g)` | 155 km — **but** `w_plates = 0`, so no belt is built | 16.8–23.6 km | **3.0 km** |
| plates | 1 (stagnant lid) | ESTIMATED 12–24 | ESTIMATED 20–40 |
| dominant recipe | craters + regolith | plates + rivers + ice | plates, low relief, big oceans |
| macro cost | **lower** than the home planet (zero weights skip layers) | the baseline | ~the same per column |

**The per-column cost does not grow with the radius.** Only the NUMBER of columns a player can reach
grows, and that is the interest band's job, not the recipe's. This is the SL9-clean shape.

---

## 9. The budget: does it fit in 8 ms?

### 9.1 The arithmetic, against MEASURED numbers, on a COLD chunk

MEASURED (`slice_05_generator.md:372-386`): the rung-0 column pass is 710 µs for 62 × 62 = 3 844
columns with 14 octaves. So **185 ns per column, 13 ns per octave**. A surface chunk costs 3.3 ms in
total against an 8 ms budget (ruling V10), leaving **4.7 ms of headroom**.

**The allowance, COMPUTED, on the right column count.** The sample box carries a halo, and the halo is
MEASURED at **six per cent more columns** (`slice_06_extractor.md:355`), so a chunk's box is **4 075**
columns, not 3 844:

```
   4 700 000 ns ÷ 4 075 columns = 1 153 ns per column for every new layer together
```

*Revision 2 divided by 3 844 and got 1 223 ns. Refuter B was right (B2-6) and the corrected, slightly
tighter frame is used everywhere below.*

| Layer | ns/column | Why that number |
|---|---|---|
| L1 plates: the fixed-order scan of the live prefix, and ONE bisector normalise | **+70** | 12–24 sites × ~3 ns + one `sqrt` and three divides. **130 ns on a 40-plate body** (§4.0) |
| L2 isostasy: the affinity field's 3 noise calls, VALUE AND GRADIENT together, plus the arithmetic | **+64** | 3 × ~18 ns (a gradient-noise derivative is ~1.4× its value) + 10 ns |
| L3 the belt profile and its analytic gradient | +20 | polynomials in `b`, and `∇b` is the bisector normal — free |
| L3 the ridge lines | +78 | 3 warp noise calls + 3 ridged calls × the MEASURED 13 ns |
| L4a the multifractal weight and the shaping | +35 | one multiply, one `abs` and one clamp per octave (15), plus the transfer curve, the talus and the fill (20) |
| L4e the hardness term | +5 | one strata-table read against a radius already in hand |
| L5 the climate operators | +30 | the macro gradient is L2's and L3's, not double-counted |
| L7 the biome | +20 | every input is already computed; two old noises retire |
| L6 one extra octave (15 against 14) | +13 | the MEASURED 13 ns/octave |
| **TOTAL** | **+335** | against an allowance of **1 153** |

**Every row of that column is ESTIMATED except the octave row.** *Refuter A is right to insist on
saying so plainly (R14): only 185 ns per column and 13 ns per octave are MEASURED. **M3 is the gate, and it must run
before the design is accepted.** §9.2 states what happens if the estimates are wrong and by how much
they may be wrong before the answer changes.*

### 9.2 The verdict, and how much room it has

COMPUTED, per chunk, against the MEASURED baselines, with 4 075 columns:

| Chunk | Today | + the stack | Budget |
|---|---|---|---|
| a cold **surface** chunk | 3.30 ms | **4.67 ms** | 8 ms ✓ |
| a cold **cave-dense** chunk | 6.10 ms | **7.47 ms** | 8 ms ✓ |
| a cold cave-dense chunk on a **40-plate** body | 6.10 ms | **7.71 ms** | 8 ms ✓ |

**Stated plainly: the stack fits, and the tight case is the cave-dense chunk, which has 466 ns per
column to spend and is estimated to spend 335 — seventy-two per cent of its allowance.** So:

- **the design survives a 39 % error in the estimates** and goes red beyond that;
- at **1.5 times** every estimate the cave-dense chunk is **8.03 ms**, which is red by 0.4 %;
- **M3 measures it, and D11's ordering is what protects it**: L4a, L4e and L6 first, because they are
  the cheap ones and they carry most of the picture.

*Revision 2's verdict was 4.90 ms and a RED 8.68 ms with rivers. The rivers are gone to domain 03, the
gradient that cost 220 ns is gone with them, and the affinity's noise calls that refuter B found
unpaid (B2-6) are now paid. The red row is gone because the work is gone, not because the number was
argued down.*

**And the added-cost row is the honest one now.** *Refuter A found that revision 2's own subtotals did
not add up — the rows summed to 436 ns against a stated 416, and the summary's table summed to 618 ns
because it carried the octaves' whole 195 ns instead of the 13 ns the change adds (R3). The refuter
is right; every table in revision 3 sums to 335 and the summary's table carries the DELTA in every
row.*

### 9.3 ★ THE BAND, and why it now SHRINKS

The band is DERIVED from the relief (`crates/terrain/src/body.rs:233-236`), and `Ladder::for_radius`
turns it into `floor_m`, which is the base of every cell address on the body.

```
   band_m = 2 × relief_whole + strata.max_depth_m() + caves.max_depth_m + 128
   MEASURED: cells_in_band(0) = 29 105 on the home planet, relief_whole = 14 305
   ⇒ strata + caves = 29 105 − 2×14 305 − 128 = 367 m   (COMPUTED, and it reproduces the measurement)
```

**The design rule, unchanged and still binding:**

> The band is sized on the recipe's **EXACT per-body bound**, which `from_seed` computes as
> `the isostatic spread + the DRAWN belt amplitude + the DRAWN trench depth + A_oct + the fill depth`.
> It is **never** sized on `h_max`, which is a ceiling nobody reaches.

**★ What changed in revision 3, and it is good news.** The octave sum is no longer the relief: §4.6's
budget takes it from 14 305 m to **2 053 m**. The macro layers add their own — the isostatic spread
(COMPUTED 6 476 m loaded), a drawn belt amplitude and a drawn trench — so the bound is a SUM of
smaller, physical terms instead of one big draw. COMPUTED consequences:

| The recipe's relief bound | `band_m` | Chunks per radial column at rung 0 | A whole column (417 ms MEASURED today) |
|---|---|---|---|
| **today, 14 305 m** | **29 105** | **470** | **417 ms** |
| an ESTIMATED new bound near 15 000 m | 30 495 | 492 | 437 ms |
| 20 000 m (a high belt draw) | 40 495 | 654 | 580 ms |
| 25 000 m | 50 495 | 815 | 723 ms |
| the CEILING used as the bound — `2 × h_max(g = 4.684) + 2 000`, that is `2 × 18 536 + 2 000 = 39 072`, doubled again by the band's own `2 ×` | 78 639 | 1 269 | 1 126 ms |

*The last row's label is spelled out because refuter A could not read it (R24): `h_max` at
`g = 4.684` is 18 536 m; sizing the bound on `2 h_max + 2 000 m` gives 39 072 m; the band formula
doubles that and adds 495, reaching 78 639. It is the row that shows why the bound must be the DRAWN
one.*

**It stays a one-way door.** `floor_m` moves whenever the bound moves, so **every cell address on
every body moves** — a strictly larger claim than "the height moves". §12 W1b, and **M14 prints the
new bound across 1 000 seeds before anything lands.**

### 9.4 The client, priced with the same numbers

MEASURED (`slice_07_client_link.md`, M7-2): 4.18 ms per chunk on one thread, 239 chunks/s, 2 004
chunks/s on 14 threads. COMPUTED, adding the same per-column work over the same 4 075 columns:

| | ms per chunk, 1 thread | chunks/s, 1 thread | The cut |
|---|---|---|---|
| today | 4.18 | 239 | — |
| + the stack | **5.55** | **180** | **−25 %** |

**A 25 % cut in the client's mesh rate is a real cost and slice 8's residency band is sized from that
rate.** It is still a very different thing from a 0.19–12 s per-body bake plus 6–403 MB, and it is
paid per chunk actually drawn rather than per body visited. **The 2.13 ms body draw is paid once per
body, on a worker, never on the message path** (§7.1).

### 9.5 The rung table

Derived from §9.1's table and §6.2's cut-off rule, with **no arm left out** — at rung 11 the ridge
lines, the hardness term and the climate operators are past their derived cut-offs, and the
multifractal weight costs what its four surviving octaves cost:

| Rung | Measured today | Added per column | + the stack | Total |
|---|---|---|---|---|
| 0 | 710 µs | 335 ns | +1 365 µs | **2 075 µs** |
| 6 | 458 µs | 335 ns | +1 365 µs | **1 823 µs** |
| 11 | 266 µs | 209 ns | +852 µs | **1 118 µs** |

**The top rung stays 1.9 times cheaper than rung 0** (today's MEASURED ratio is 2.7), so the bench's
own assertion — the top rung cheaper than rung 0 — still passes, by a narrower margin than today.

*Revision 2 claimed 4.4 times, and both refuters showed the claim came from dropping L4a at rung 11
while §6.2 said L4a is never dropped (R4, B2-16). **They are right; 1.9 is the honest number and it
is computed with L4a kept.** Refuter B also warned that the hydrology-cell harvest would INVERT the
order at rung 11, where a chunk spans 126 976 m and touches 3 844 cells (B2-2, COMPUTED and correct).
§4.0's scan law removes the harvest from the definition, so the inversion cannot happen: a coarse
rung scans exactly what a fine rung scans, and it is cheaper because it has fewer octaves and fewer
layers.*
## 10. Law gates — how the recommendation passes each one

### 10.1 SL10 — one generator, two hosts, no drift

| Formula | Operations needed | Inside the `Gf` fence? |
|---|---|---|
| plate site distance, over the live prefix in index order | dot product, compare | **yes** |
| the boundary distance `b = R·(d · n̂)` | subtract, dot, `sqrt`, divide | **yes**, with a stated error bound (§4.1) |
| boundary kind | dot product, compare | **yes** |
| the continental affinity field, **value and gradient together** | 3 `noise3` calls, the quintic fade and its own derivative (a polynomial) | **yes** — *this replaces revision 2's 4-point finite difference, and with it the unnamed step `δ` (R20)* |
| isostasy `t(1 − ρ_c/ρ_m)` | `− × ÷` | **yes** |
| **the water load `d = step/(1 − ρ_w/ρ_m)`** | `− ÷` | **yes** — *revision 2 omitted the term entirely (R11, B2-9)* |
| the blend across a margin | the quintic fade, already in `noise.rs:66-71` | **yes** |
| the belt profile and its analytic gradient | polynomials in `b`; `∇b` is `n̂` itself | **yes** |
| ridged fold `1 − |n|` | `Gf::abs`, `−` | **yes** |
| domain warp | 3 extra `noise3` calls | **yes** |
| **the multifractal weight `w_i = clamp(w_{i−1}(a + b|n_{i−1}|), 0, 1)`** | `× + abs`, `Gf::clamp` | **yes** — *new in revision 3; it replaces the gradient transfer refuter A refuted (R13)* |
| **the hardness read** | a table index against a radius, compare | **yes** |
| `h_max = σ_y/(ρ_c·g)` | `× ÷` | **yes** |
| the hotspot chain `normalise(s + t·u_k)` | `+ × ÷ sqrt` | **yes** |
| the talus limiter, the fill, the transfer curve | polynomials, `Gf::clamp` | **yes** |
| the environmental lapse rate `(g/c_p)·w` | `× ÷` | **yes** |
| `c_p` from the mean molecular weight | `× ÷` | **yes** |
| the Hadley band edge `s_H = sqrt(…)`, tested against `dir · pole_axis` | `× ÷ sqrt`, dot, compare | **yes** |
| the Coriolis-turned wind `p·c₁ + (d × p)·c₂`, normalised | cross, dot, `sqrt` | **yes** |
| **the hypsometry's SAMPLE DIRECTIONS, from a coarse rung's cell centres through the bend** | the ladder's own arithmetic, a fixed count, no rejection loop | **yes** — *revision 2 had no sampler at all and listed the hypsometry as fenced anyway (R12)* |
| the hypsometry itself | 4 056 fixed-order FULL-field evaluations, compare | **yes** |
| the sea-level bisection, with the water load inside the loop | compare, `× ÷`, a FIXED 24 steps | **yes** |
| the crater size-frequency draw | the integer hash + `sqrt` | **yes** |
| the Coriolis parameter `2Ω·sin(lat)` | `sin` — **cured**: `dir · pole_axis` IS `sin(latitude)` | **yes**, see §10.2 |

**Nothing needs `min`, `max`, `mul_add`, `powf`, `exp` or any transcendental.** Every general power
reduces to a square root by choosing `m/n = 1/2`, which is the value the literature reports for real
rivers anyway — and that choice now belongs to domain 03 with the river network.
### 10.2 The one arithmetic refusal, and its cure

`sin(latitude)` is forbidden. But **latitude never has to be an angle**: `dir · pole_axis` IS
`sin(latitude)` for a unit direction and a unit pole. So `f = 2Ω·(dir · pole_axis)` — a dot product and
a multiply. The same trick removes every trigonometric function from the climate layer, because every
"angle" in it is really a direction cosine, **including the Hadley band edge**, which is a comparison
against `sin φ_H` and never against `φ_H`. `crates/terrain/src/height.rs:45` already does this
(`let latitude = dir[POLE_AXIS].abs();`).

**Order determinism:** the plate scan is a fixed-order loop over the live prefix with index
tie-breaks; the hypsometry's 4 056 directions are a coarse rung's own cell centres, evaluated in
address order; the bisection has a FIXED step count; and **§4.0's SCAN LAW makes any harvest give the
scan's own answer**, so no shorter list can reorder a sum. **No sort, no heap, no reduction over an
unordered set, anywhere.**

**And the fence has a second layer now** (§5.4): every physical fact reaches the recipe as an INTEGER
authored once by the body's own realm, so an unfenced float from the forest never enters the
generator on any host.

### 10.3 The seed ruling (2026-08-27) — is the layout a treasure map?

*"Anything a fixed seed alone determines must be SAFE TO PUBLISH; anything VALUABLE must depend on
world state that CHANGES."*

**The layout itself is safe.** A wiki page listing every continent, range, coastline and river of every
planet is a MAP, and a map is a fine thing for a wiki to hold. It is the same class as the star field,
which R1 already ships once to everybody.

**★ The risk this report names.** A published plate map plus a geology-driven ore rule **is** a treasure
map. Real ore follows tectonics: porphyry copper sits above subduction zones, gold sits in orogenic
belts, placer gold sits in river gravels. **That is exactly what S5.1 forbids**, because the plate map
is a pure function of the seed and the client links the generator, so the map IS published.

**The recommendation: the layout may decide the ROCK and must NOT decide the ORE.** Common bulk stock
is already the generator's (ruling V9: *"it draws bulk stock only, no ore, no indicator"*). Valuable
deposits stay live state, and the strategic drop table may READ the layout on the server — where the
map is not published — without the layout ever hinting at it in a client binary.

**★ The residual, which revision 1 closed too early.** *Refuter A was right (F23): real deposits
correlate with ROCK, and a published rock map is a prospecting PRIOR — "search the granite of the old
belts". The report should state the residual and let the owner rule, not close the row.* The residual
is real and it has a size: a rock map narrows a search from "the whole planet" to "the old belts",
which might be 5 % of the surface. **That is a 20× prospecting advantage to anyone who reads a wiki.**
Three answers, and **D10 asks the owner to pick**:

(i) no rock map at all (throws away a free, believable geology);
(ii) a rock map, and the live-state deposit rule must NOT correlate with it (the deposit is drawn
against the rock's own distribution, so the prior is worthless);
(iii) a rock map, and the correlation is accepted as gameplay — a geologist player earns their edge.

**And the cost of a rock map, which revision 1 also hid.** *Refuter A found the contradiction (F7) and
refuter B found it again (B-W1): revision 1 said "the layout changes only `h`" and "the record —
unchanged" while recommending a layout-driven rock.* Today the bedrock is ONE draw per BODY
(`crates/terrain/src/body.rs:200`, `Bedrock::ALL[range(0,5)]`) and `StrataTable::at` takes a biome and
a depth and no direction (`strata.rs:189`). **A per-column rock is a new signature, a new input to the
cell pass, new HR5 branches and a world-identity bump.** It is not free, and D10's option (i) is
genuinely cheaper.

**★ And option (i) got cheaper again in revision 3.** *Refuter B charged that the report never told the
owner what "no rock map" costs in LOOK (B2-12).* It now costs **nothing in look**: the benched cliff,
the mesa rim and the cuesta come from **L4e**, which reads the HORIZONTAL strata the body already
draws — a depth table, not a direction (§4.4e). So the rock map is needed for the ORE geology and for
nothing else, and D10 is a pure question about value, which is where the seed ruling wants it.

### 10.4 SL5 — one world

Every number is a property of THE world's bodies, drawn from THE seed. No scale knob, no preset, no
reduced grid, no test-only body.

### 10.5 No magic numbers — passed for every SCALAR, OWED for every SHAPE

| Number today | Replaced by |
|---|---|
| `0.004` relief share (`body.rs:147`) | `h_max = σ_y/(ρ_c·g)` bounds the belt; the octave sum gets §4.6's budget |
| `200`, `12_000` relief clamp (`body.rs:148`) | the same ceiling |
| `[0.45, 0.55)` `k_rough` (`body.rs:155`) | a draw near `[0.58, 0.63]` (§1.3, M12), modulated per column by L1–L3 |
| `[−0.4, +0.3]` sea offset (`body.rs:188-189`) | the fixed-sample hypsometry and the water inventory |
| `0.55` highland threshold (`body.rs:214`) | the biome from insolation, the lapse rate and the Hadley edge |
| **`SHORT_WAVE_M = 30`** (`body.rs:25`) | **2 m, from the cell size at Nyquist** (§1.6) |
| **the amplitudes' "sum = relief" rule** (`body.rs:176-184`) | **the octave BUDGET and the LADDER CAP** (§4.6) |

**And the numbers this report ADDS, each with its derivation:**

| New number | Where it comes from |
|---|---|
| `PLATES_MAX = 40` | the plate-count law's own ceiling in the literature (§4.7) |
| the octave budget's anchor | the amplitude the shipped octave law already gives at 50 km (§4.6) |
| the bound `A_oct ≤ 0.5 × the isostatic step` | the texture may not rival the crust's own contrast. **The 0.5 is a CALIBRATION, and M16 owes the picture** |
| the ladder cap | half a cell at the rung where the octave disappears — the ladder's own number, not a new one |
| the isostatic blend width | the flexural wavelength of the crust, `∝ (t³/(ρ_m·g))^(1/4)` — a physical length |
| the coastal notch's band | the significant wave height, from the wind speed and the fetch (§4.5) |
| the 24 bisection steps | the count that reaches under a millimetre on a 40 km range (COMPUTED: 2.38 mm) |
| the 4 056 hypsometry samples | a coarse rung's own cell count (26 per face edge), whose sampling error is under 1 % (COMPUTED: 0.78 %) |
| the angle of repose, 32–37° | a measured property of loose rock |
| `ρ_c = 2 800`, `ρ_m = 3 300`, `ρ_o = 2 900`, `ρ_w = 1 030` | rock, mantle, ocean-crust and water densities |
| `σ_y = 243 MPa` | Everest's calibration, and §4.3 says plainly that it is a calibration, not a measurement |

> **Flexural wavelength, in plain words.** Press down on the crust at one point and it does not dent
> only there: it bends over a distance set by how stiff and how thick it is. That distance is the
> flexural wavelength, and it is why a mountain range has a broad low plain in front of it rather than
> a sharp step. *Game example: the pilot flies off the home planet's range and the ground stays low
> for 80 km before the ordinary hills resume.* *(Refuter B asked for this word, B2-21.)*

**★ The gate is stated honestly, and it is not fully passed.** *Refuter A is right (R20): the table
above covers every SCALAR and the report also introduces SHAPES, which are what actually decide what
a player sees. These are OWED, and no slice starts before they are written down:*

| The shape | Why it is not a scalar |
|---|---|
| the transfer curve's polynomial | its exponents decide how flat a floor is and how sharp a crest is |
| `f(c)` and `g(a)` in the uplift | how convergence rate and belt age become height |
| `profile(x)`'s rise–crest–backslope–foreland shape | the cross-section of every range in the world |
| the domain-warp amplitude and the ridged octave count (3) | how sinuous a range reads from orbit |
| the multifractal weight's `a` and `b` | how hard erosion bites — the single most visible pair in §4.4 |
| the hardness curve of L4e | how tall a bench stands |
| `r_min`, the roughness modulator's floor (ESTIMATED 0.10) | how flat the flattest plain is |

**Every one of them is decided ON A PICTURE (M4, M12, M16), not by argument**, and the honest gate
statement is: *"no magic numbers — passed for every scalar, owed for every shape."*

**Two Earth calibrations are named as calibrations, not laws:** the 35 km and 7 km crust thicknesses
(§4.2), and `H = 15 km`, `Δθ/θ₀ = 1/3` in the Held–Hou form (§4.5, with M11 owed).
### 10.6 The rest, in one line each

- **The 8 ms budget** — §9.2: **4.67 ms for a cold surface chunk and 7.47 ms for a cave-dense one**,
  both green, with 28 % of the cave-dense allowance in hand. M3 measures before the design is
  accepted.
- **The ladder** — §6.2: today the half-cell promise is broken at six rungs of twelve (MEASURED by
  replication); §4.6's LADDER CAP makes it true at every rung; §9.5: the top rung stays 1.9× cheaper
  than rung 0, computed with no layer left out.
- **The record** — **the STORED 12-byte record of ruling V6 part B does not change, and neither does
  the crate's own 2-byte generated `Cell`** (`chunk.rs:64-69`). What changes is `fluid_at`'s signature
  and three call sites (§4.4c), and, if D10 takes a rock map, `StrataTable`'s signature too (§10.3).
  *Refuter A was right that revision 2 named neither record (R22).*
- **The edit pyramid** — unchanged, and the layout makes it BETTER: a coarse rung now carries the
  continent and the range, which is what the far view needs.
- **★ The collider and the extractor — RE-MEASUREMENT OWED.** The density byte is the RADIAL gap
  clamped to one cell (`crates/terrain/src/chunk.rs:630`), so the extractor's vertex slides along the
  radial, and slice 6 MEASURED its snap error on TODAY's 2° field: p50 0.0039 cells, p99 0.0074 cells
  (`slice_06_extractor.md:357-358`) — while the same document warns *"a crease shows up to half a
  cell"* (`:232`). **This report raises the RMS slope at a 1 m baseline from 1.9° to about 19° and
  adds a crease at every ridge crest.** **M15 re-measures it, and the seating rule with it.**
- **SL8 seamless** — three directions: across a margin (the quintic blend, a per-body width, a
  property test on `|h(x+ε) − h(x)|`); across a RUNG (§6.2's half-cell rule, **now true by
  construction rather than assumed — and its INVISIBILITY still waits on slice 8's tier rule, R18**);
  and across any harvest boundary (§4.0's scan law makes a harvest unable to change an answer, which
  removes the direction entirely rather than bounding it).
- **HR5 100 %** — the branches are: the boundary-kind three-way, the affinity threshold, the
  weight-zero skips, the transfer curve's arms, the talus clamp, the ladder cap's clamp, the
  bisection's two sides, and the `BodyFacts` refusal. **★ Some cannot be reached on the home planet**
  — `w_plates = 0`, `w_terrain = 0`, `w_rivers = 0`, and an ocean–ocean convergent boundary if the
  home planet draws none. **SL5 permits testing on ANOTHER REAL BODY of THE world; it forbids an
  invented one.** So `crates/terrain/src/home.rs` must also state: a moon of the home system (for
  `w_plates = 0` and `w_rivers = 0`), and an ice giant (for `w_terrain = 0`, pinned as *"this body
  returns `None`"*, which §4.7's `solid_surface` flag now makes a real fact rather than an owed one).
  **HR5 blocks the slice without them.**
- **V4 — vegetation are art assets from a server skeleton.** The layout is what makes the skeleton
  believable: a forest belongs on a windward slope below the snow line, and grass belongs on a valley
  floor. Nothing here draws a plant.
- **HR3 — no kind match.** §4.7: layer weights derived from physical facts, never a `match` on a body
  kind.
- **HR4 — the identical fixture on two shard kinds.** A station or hull that holds terrain (ruling V4
  item 15.2) gets `w_plates = w_rivers = 0` and a flat isostatic base — the same code with zero
  weights, which is the honest HR4 subject and is also the `w_plates = 0` coverage arm.

---

## 11. The lawful use of a simulation: an offline oracle

Options (a) and (b) are refused as shipped machinery (§3.3). They have one lawful home: **a test-only
oracle in the repository, never in a product build**, which runs a real stream-power erosion on a small
patch of the home planet and compares statistics — not bytes — against the closed form:

- the slope–area relation's exponent, over four decades of area;
- the hypsometric curve's shape;
- the drainage density (channel length per unit area);
- the fraction of the surface in closed basins;
- **★ the crest–divide agreement: what fraction of the ridge crests ARE drainage divides.**
  *Refuter A named this as the property the stack's independence breaks (R15). §4.4a's multifractal
  weight couples roughness to height and so removes the worst case, but the coupling is UNMEASURED,
  and this is the statistic that measures it. It becomes the acceptance test domain 03's river
  network must also pass.*

**★ Is this "a second implementation", which a standing owner law forbids?** **No, and the distinction
is sharp:** the banned thing is a second implementation **of the shipped world** — a reduced
generator, a test-only variant, a second path selected by config. This oracle produces **no world**.
It produces five statistics, it is never linked into a product build, and nothing it computes ever
reaches a player. It is an instrument, in the same class as a golden table: a measurement that could
have failed.

---

## 12. One-way doors

| # | The door | Why it is one-way | When it must be decided |
|---|---|---|---|
| **W1** | **Every change in §4 moves `h` on every planet.** | Every stored edit and every placed block is addressed by a cell whose surface moved. | **Before the first saved world.** |
| **W1b** | ★ **The band moves, so `floor_m` moves, so EVERY CELL ADDRESS on every body moves.** | `body.rs:233-236` derives the band from the relief; `ladder.rs:95` derives `floor_m` from it. **The bound now goes DOWN for the octaves and UP for the macro layers**, so the direction is not obvious and M14 must print it. | Same slice as W1, and **M14 first**. |
| **W2** | The obliquity turning `POLE_AXIS` into a per-body vector, built from a tilt AND an azimuth. | Same as W1, and it moves every biome. | Same slice. |
| **W3** | The sea level solving for an inventory, on the full field, with the water load. | Moves the waterline on every body. | Same slice. |
| **W3b** | ★ **The water surface becoming per-column** (`fluid_at`'s signature, the above-surface skip, the halo rule). | It changes which cells are Water on every body. Neither the stored 12-byte record nor the 2-byte generated cell changes. | Same slice. |
| **W4** | The `BodyFacts` record. | A **new tag** is NOT one-way (a reader that does not know it skips it). It **IS** one-way as a field inside `SurfaceStmt`, because `crates/wire/src/version.rs:10-14` says *"New data rides a new trailing variant, never a new field"*. | **Choose the tag.** And state the REFUSAL (§5.4), or an old client silently draws the wrong mountains. |
| **W5** | ★ **`from_seed`'s signature changes** (it must take `BodyFacts`), which breaks `crates/bins/tests/home_body_pin.rs` and `crates/terrain/src/home.rs`. | Every caller and every pin moves together. | Same slice. It is mechanical, but it must be planned. |
| **W6** | ★ **`SHORT_WAVE_M` 30 m → 2 m, the coarsest wavelength 400 km → 50 km, the octave BUDGET, and the LADDER CAP.** | One extra octave, a whole new band of the spectrum, and a different amplitude for every octave: every column on every body moves. **These four are ONE change and must land together** — refuter A's R1 is exactly what happens when they do not. | Same slice as W1. |
| **W7** | The biome list growing past four (`Biome::ALL`, `StrataTable::at`). | Every cell's substance may change. | Same slice, IF D12 takes it. |
| **W8** | A layout-driven ROCK (`StrataTable::at` gaining a direction). | Every cell's substance changes. **L4e's benched cliff does NOT need this** (§4.4e), so the door is the ore question's alone. | Only if D10 takes option (ii) or (iii). |
| **W9** | ★ **`from_seed` moves OFF the client's message path.** | `crates/client/src/chunks.rs:266-289` calls it inline today, and §7.1 adds 2.13 ms to it — 60 ms for a system stating 28 bodies, which is a `tick-hitch` seam. | Same slice as W3. *(Refuter B found it, B2-17.)* |
| **W10** | ★ **`BodyDefinition` grows by the plate list** (ESTIMATED 2.2 KB in a `Copy`, `PartialEq` struct that is returned by value). | Every call site copies it; `PartialEq` compares the dead suffix too. | **D17 decides the shape; M10 prints the size first.** *(Refuters A and B, R19 and B2-24.)* |

---
## 13. What you decide (with a recommended answer)

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| **D1** | Which family builds the macro map? | (a) plate simulation; (b) uplift + fluvial iteration; (c) pure closed form; (d) the closed-form layer stack | **(d)** | (a) and (b) are not addressable, are a second shape under the ladder, and add four ORDERING drift classes the fence does not cover (§3.3). Their cost is 0.19–12 s and 6–403 MB per body on the client, which is survivable — so they are refused on the four structural reasons, not on the cost. (c) is today's world and cannot make a coast. |
| **D3** | ★ **THE OCTAVE CHANGE, as ONE decision: re-anchor to 50 km, drop `SHORT_WAVE_M` to 2 m, give the sum a BUDGET (2 053 m, not 14 305 m), cap every amplitude by the ladder's half cell, and raise the roughness draw to about 0.62.** | (i) no; (ii) yes, all five together | **(ii)** | *Revision 2 asked this as two decisions and refuter A proved the pair is catastrophic if the budget is missing (R1): the world becomes 15° everywhere, or 56° with the roughness. **They are one decision.** Together they give 18.9° at a metre falling to 1.3° at 8 km — real land's own scaling — 6.13 m of relief in a 50 m window against today's 1.14 m, and a ladder promise that is true at every rung for the first time (§1.3, §4.6, §6.2). This is the direct answer to "no details at all", and it costs one octave, 13 ns.* |
| **D2** | Does the roughness stop being one number per body? | (i) keep one `k_rough`; (ii) modulate it per column, DOWNWARD only, from L1–L3 | **(ii)** | One number gives 19° everywhere or 2° everywhere. `r ∈ [r_min, 1]` keeps the band and the ladder bound exact because it never scales up (§4.6). |
| **D4** | Does the relief bound become `h_max = σ_y/(ρ_c·g)`? | (i) keep `0.4 % of radius, capped at 12 km`; (ii) the strength relation | **(ii)**, as a SCALE with a stated ±15 % spread | Earth 100 %, Mars 94 %, Venus 112 % (COMPUTED, §4.3 — the Mars row is corrected from revision 2's 97 %). Today's rule over-reliefs a 12 000 km super-earth by 2 to 6 times. |
| **D5** | Does the sea level solve for a water inventory? | (i) keep the uniform draw; (ii) bisect a fixed-sample hypsometry on the FULL field, with the water load | **(ii)** | Today's draw is a ±2 σ lottery and the home planet drew 0.9 % ocean. The method is fenced, order-fixed and costs 2.13 ms once per body, **on a worker** (§7.1). |
| **D6** | Are the plates plain spherical Voronoi cells, or warped? | (i) plain; (ii) warp the distance with one noise call | **(ii)**, decided ON PICTURES | Plain Voronoi cells read as too regular from orbit. One noise call is 13 ns MEASURED. |
| **D7** | May the crate hold physical CONSTANTS, and which? | yes / no | **yes**, each with a citation in its doc comment | **The list is: `ρ_c`, `ρ_m`, `ρ_o`, `ρ_w`, `σ_y`, the angle of repose.** `c_p` is NOT on it — it is derived from the mean molecular weight. *Refuter A is right that `L_SUN_W` and `RHO_ROCK_KGM3` are a weaker precedent than revision 2 claimed (R24): they live in the UNFENCED crate, and a constant inside `vd-terrain` is a world-identity datum in a way those two are not. The recommendation stands; the precedent does not carry it.* |
| **D8** | Do the spin, the obliquity, the water inventory, the age, the orbit's eccentricity and the atmosphere get drawn in the forest and cross? | (i) no; (ii) yes, passed into `from_seed`; **(iii) move the body's physical DRAW below the fence instead** | **(ii)** | The owner asked for biomes dependent on spin, and for weather. Neither has an input without it. **★ Option (iii) is new and the owner should see it (B2-11): it opens NO lane at all, but it re-draws every body in the world, because the mass–radius law today is a `powf` fit. If the world is re-drawn for another reason, that is the moment to take it.** |
| **D9** | ★ **Are the incoming facts INTEGERS, authored once by the body's own realm?** | (i) pass floats; (ii) quantise on every host; **(iii) quantise ONCE, at the owning realm, and never re-derive** | **(iii), and it is not optional** | `vd-physics` computes with `powf` (MEASURED: 43 calls on 39 lines, and 11 on 9). Quantising on every host is not a cure — it moves the failure to the grid boundaries (R8, B2-10). One author, one integer, no re-derivation. |
| **D10** | ★ **May the layout inform the ROCK, and what about the residual prospecting prior?** | (i) no rock map; (ii) a rock map, and the deposit rule must NOT correlate with it; (iii) a rock map and the correlation is accepted as gameplay | **(ii)** | (iii) turns the published rock map into a 20× prospecting advantage for anyone who reads a wiki, which the seed ruling forbids by name. **★ And the price of (i) fell in revision 3: the BENCHED CLIFF no longer needs a rock map** — L4e reads the horizontal strata the body already draws (§4.4e) — so (i) now costs the ore geology and not the look. |
| **D11** | ★ **The hydrology: what lands, and in what order?** | (i) L4a's erosion signature only, in this slice; (ii) wait for a river network too | **(i)**, and the owner is told what he does not get | L4a is unconditional, costs 35 ns, needs no network and carries most of the valley shape. **The river, the lake, the flood plain and the waterfall are DOMAIN 03's**, under §4.4b's three conditions. **The first landing has no river, and the reference picture has one** (B2-18). |
| **D12** | ★ **Does the biome field get rebuilt, and does the biome list grow past four?** | (i) keep `biome_at` and four biomes; (ii) rebuild it from insolation, obliquity, the lapse rate, the Hadley edge, the rain shadow and continentality | **(ii)** | The owner's first requirement is *"biomes … dependent on the planet position, spin, trajectory, size and gravity"*, and two noises serve none of those words. Every input is already computed, so it costs ~20 ns. The owner should name the biome list. |
| **D13** | ★ **What happens to a body with no solid surface?** | (i) build a cell grid anyway; (ii) `from_seed` returns `None` and the realm draws a banded cloud shell | **(ii)** | The ladder already refuses Jupiter and Saturn (radius > 42 723 km). The gap is the ICE GIANTS — Neptune and Uranus are accepted today and would get a voxel surface. **The `solid_surface` bit of §5.4 is what gives this answer an input** (B2-22). |
| **D14** | ★ **Who owns the overhang, the arch and the rock pillar?** | (i) nobody — accept that the world has none; (ii) a 3-D density term in a thin band at the surface, scoped in its own slice | **(ii), and NOT in this report** | A single-valued `h(dir)` cannot make one (§4.8), and the reference picture holds four kinds. `carve.rs` is the existing 3-D route. |
| **D15** | When do these changes land relative to the first saved world? | before / after | **before** | Every one of W1–W10 re-addresses stored edits, and W1b moves **every cell address on every body**. |
| **D16** | Is the offline erosion oracle (§11) built? | yes / no | **yes, as a test-only crate** | It converts "the closed form looks right" into "its slope–area exponent and its crest–divide agreement match a real simulation's". It produces no world, so it is not a second implementation. |
| **D17** | ★ **Where does the plate list live?** | (i) inside `BodyDefinition` as a fixed array (ESTIMATED 2.2 KB, `Copy`, `PartialEq`); (ii) behind a reference; (iii) re-drawn on demand from the body's seed | **(i), unless M10 says otherwise** | The octave table already has this shape and the struct's `Copy`-ness is what makes the crate simple. But 2.2 KB is four times today's whole `BodyDefinition`, it is returned by value, and `PartialEq` compares the dead suffix. **M10 prints the size and the copy cost before this is decided.** *(R19, B2-24.)* |

---

## 14. Measurements owed, in order

| # | What | How | Cost, ESTIMATED |
|---|---|---|---|
| **M1** | **The diagnosis, re-measured in the crate, not in Python.** Slope at 1 m / 32 m / 512 m, local relief in a 50 m / 1 km / 20 km window, and the height histogram, on the home planet. | A new `terrain_cost` section. It makes §1.2 a crate measurement and it becomes the acceptance gate for every later change. | An hour |
| **M2** | **The home planet's surface gravity.** Everything in §4.3 and §4.5 that says ESTIMATED becomes MEASURED. Today the answer ranges from `g = 3.68` (Mars's density) to `g = 5.17` (Earth's), which moves the ceiling and the snow line by 27 %. | One line: print `surface_gravity_mps2(taxon.mass_kg, taxon.radius_m)` in `terrain_cost`. | Minutes |
| **M3** | ★ **THE BUDGET GATE, and it must run BEFORE the design is accepted.** Time ONE plate scan, ONE affinity value-and-gradient, ONE ridged fold and ONE multifractal pass at rung 0 on the home planet, then a cave-dense chunk with the whole stack on. §9.2's margin — 335 ns estimated against 466 allowed — is the number this settles. | Add each layer behind a bench switch; time the column pass at rungs 0, 6 and 11. | A day |
| **M4** | **The picture, with a PASS MARK.** The three harness shots re-taken with the stack, plus a coast, a benched cliff, a triple junction, and a vista with a range at 30–50 km. **The pass mark is doc 01's own metric: 8–12 orienters per 60° of field of view**, counted by hand on the shot. | The slice 7 harness path. | Half a day |
| **M5** | **The closed-basin fraction.** How much of the surface fails to drain, with L4a alone. It is the number domain 03 inherits. | Sample 10⁵ columns. | Hours |
| **M6** | ★ **The scan law, TWO tests.** (a) columns either side of a cube edge give bit-equal heights; (b) **a harvested answer equals the FULL-SCAN answer, bit for bit**, over a sample of columns and every bounded list. | Unit tests in the crate. | Included in the slice |
| **M7** | **The ladder bound with every layer on**, including each layer's derived cut-off rung and the LADDER CAP. It must show the half-cell rule holding at every rung, which today it does not. | Extend the property test at `height.rs:83-110`. | Included in the slice |
| **M8** | **The ocean fraction across 1 000 seeds**, before and after D5. | A bench loop over `BodyDefinition::from_seed`. | Hours |
| **M9** | **The oracle** (§11): the slope–area exponent, the drainage density, the hypsometric shape and **the crest–divide agreement**, closed form against a real erosion run. | The test-only crate of D16. | A week |
| **M10** | ★ **The `BodyDefinition` size and copy cost** after the plate list, and the boot self-check cost with §7.1's hypsometry on (13.5 ms today, MEASURED). **D17 waits on it.** | `terrain_cost`. | Minutes |
| **M11** | ★ **The Held–Hou calibration.** Fit `H` and `Δθ/θ₀` on Earth, then check Mars and Titan — and fold the small-angle-to-sine conversion into the fitted constant (§4.5). Fit the `T_warm` climatology too, which §4.5 marks as an ESTIMATED input. | An offline fit; no crate change. | Hours |
| **M12** | ★ **The roughness picture.** `k` at 0.58, 0.60, 0.62 and 0.65, and the floor at 2 m against 3 m, under the boots at rung 0. Decides D3's two open ends. | The harness, four shots. | Hours |
| **M13** | ★ **THE AUTHORING PIN, rewritten.** (a) a test asserting the generator never receives a float; (b) a bench over 1 000 seeds printing every fact's DISTANCE to its nearest quantisation boundary, in ulps, against a stated floor; (c) the still-owed measurement of `surface_gravity_mps2`'s drift between the shipped targets, which is `D-TERRAIN-1`'s own G4 leg. *Revision 2's version could not fail (R8, B2-10).* | A unit test and a bench. | A day |
| **M14** | ★ **The recipe's exact relief bound and the resulting band, across 1 000 seeds.** The octave term falls from 14 305 m to ~2 053 m and the macro terms appear; the direction of the net change is UNMEASURED and this number gates W1b. | A bench loop printing `band_m` and `cells_in_band(0)`. | Hours |
| **M15** | ★ **The extractor's snap error, RE-MEASURED on the new field.** Slice 6's p50 0.0039 / p99 0.0074 cells were taken on a 2° world; this report makes it ~19° at a metre with a crease at every crest. The seating rule is re-checked with it. | The slice 6 bench, re-run. | Hours |
| **M16** | ★ **The octave budget's own calibration.** `A_oct` at 0.3, 0.5 and 0.7 of the isostatic step, on one picture each, at 5 km and at 50 m. It is the number that decides whether the noise reads as texture or as a second mountain system. | The harness, three shots. | Hours |

---

## 15. The recommended design in one paragraph

The home planet is uninteresting for two measurable reasons and the ladder is unsound for a third:
its height field is one stationary noise sum whose slope is about 2° at every baseline from 1 m to
8 km and which puts 1.14 m of relief inside a 50 m window; its octave table stops at a 48.8 m
wavelength, so **nothing at all exists between 1 m and 49 m**; and the amplitudes it drops at each
coarse rung exceed half a cell at **six of its twelve rungs**, so the far view already moves the
ground by more than the rung can show (all COMPUTED, on a replication validated bit for bit against
the crate's own noise pin). The cure is a **closed-form layer stack**, not a simulation: a short plate
list drawn once inside `BodyDefinition`, with **every layer DEFINED as a fixed-order scan of the
body's bounded lists and every harvest reduced to an optimisation that may not change an answer** —
which is what makes the query independent of the asking chunk, of the cube face and of the bend;
**Airy isostasy on a crust FIELD with the water column standing on it**, which puts the abyssal plain
6 476 m below the continental surface instead of 4 455 m and which gives the **passive margins** that
carry most of a real world's coastline; **orogeny** along convergent, divergent and transform
boundaries under a crustal-strength ceiling `h_max = σ_y/(ρ_c·g)` whose Everest calibration lands
within 6 % of Olympus Mons and 12 % of Maxwell Montes and which this report calls a calibration
rather than a law; **an erosion signature with no gradient and no search** — a multifractal weight,
one multiply per octave, which flattens floors and sharpens crests at every scale from 50 km to 3 m
and which couples the roughness to the height so a crest is never a smooth rib beside a corrugated
plain; **the strata read back into the shape**, so a hard bed holds a ledge and a cliff comes out
benched, for five nanoseconds and with no per-direction rock map; **climate operators** split
explicitly into SHAPE (the carved valley, the notch, the dune field) and COVER (the snow, the cloud,
the rain), with a snow line on the ENVIRONMENTAL lapse rate that reproduces Earth's subtropical
maximum and a Hadley edge from the spin whose reading of the source is stated and whose band count is
withdrawn as underived; **the octave table re-anchored, BUDGETED and LADDER-CAPPED** — 50 km down to
3 m, a sum of 2 053 m instead of 14 305 m, and every amplitude clamped to the half-cell budget of the
rung where it disappears, which is what makes the far view sound for the first time; and **the biome
rebuilt** from insolation, obliquity, eccentricity, the lapse rate, the Hadley edge, the rain shadow
and continentality, which is the owner's first requirement and which two noises never served. No grid
is baked, nothing is unaddressable, no second shape appears under the ladder, and none of the four
ordering drift classes a simulation carries is introduced. The cost is **+335 ns per column against a
COMPUTED allowance of 1 153 ns**, which takes a cold surface chunk from a MEASURED 3.3 ms to
**4.67 ms** and a cave-dense chunk from 6.1 ms to **7.47 ms — both green, with the cave-dense case
using 72 % of its allowance, so the design survives a 39 % error in the estimates and M3 must measure
before it is accepted**. The layout needs the body's physical facts, the client holds none of them,
and `vd-physics` computes them with `powf`, so there are **two SL6 asks**: a thirteen-integer
`BodyFacts` record of 52 bytes as a new tag, **authored ONCE by the body's own realm and never
re-derived anywhere**, with a stated refusal path for a client that does not hold it — and, opened
here and answered by domain 05, the weather lane, because weather is live state and the snow a player
sees on a ridge is COVER, not shape. Every change re-addresses stored edits, and because the band is
derived from the relief, **every cell address on every body moves**, so all of it lands **before the
first saved world**. And the report states its own limits plainly: a single-valued height field makes
cliffs, benches and scree but **no pillar, no arch and no overhang**; **the river, the lake and the
flood plain belong to domain 03 and are not in this slice**, so the first landing the owner sees has
a coast, a range and a valley, and no water above the sea.

---

## 16. Open questions

| # | Question | Why it is open |
|---|---|---|
| **Q1** | **Is the closed form actually believable, or only cheaper?** | §11's oracle is the only honest answer, and it has not been built. Until M9 runs, "believable" is an argument. The owner decides on pictures (M4, against doc 01's 8–12 orienters per 60°). |
| **Q2** | ★ **Can a river network be built at all under §4.4b's three conditions?** | It is now DOMAIN 03's question, not this report's. What this report establishes is the shape it must have: bounded work per column, an exact-zero support with a fixed summation order, and a per-column water surface to live in. **The construction that could satisfy it — a coarse skeleton refined rung by rung, ESTIMATED 8 levels on the home planet — is UNMEASURED.** |
| **Q3** | **How many plates does a body of a given mass and age have?** | The physical driver (lithosphere thickness against radius against heat flux) exists in the literature but is not reduced to a fenced formula here. The BOUND is settled (~40, hence `PLATES_MAX = 40`); the law is not. |
| **Q4** | **What is the home planet's real surface gravity?** | It moves the ceiling and the snow line by 27 % between a Mars-like and an Earth-like density. One line of bench (M2). |
| **Q5** | ★ **What does the weather lane carry, and at what rate — and how many circulation bands does a body have?** | Opened in §5.5 as SL6 ask two. **The band COUNT goes with it**, because §4.5 withdrew it as underived. Domain 05 answers both. |
| **Q6** | **Does the sub-metre block ruling (V2.4) change what "detail" means at the surface?** | With the floor at 2 m the height field now reaches the cell, and COMPUTED it puts 6.13 m of relief in a 50 m window. The roughness target may now be met by the field itself rather than by art placement. The trees-and-decoration domain owns the other half. |
| **Q7** | **Are 12 rungs enough once the macro map exists?** | A continent is ~2 000 km, which is 1 000 cells at the top rung — fine. Whether the far view wants a rung above 11 is slice 8's residency question. |
| **Q8** | ★ **Is the estimate right?** | §9.2 is green with 28 % of the cave-dense allowance in hand, and every per-layer number except the octave's 13 ns is ESTIMATED. **M3 is the gate and it runs before the design is accepted.** |
| **Q9** | ★ **Which real bodies of THE world carry the zero-weight coverage arms?** | HR5 needs `w_plates = 0`, `w_rivers = 0` and `w_terrain = 0`, and the home planet has none of them. `home.rs` must state a moon and an ice giant, and each needs a pin. SL5 permits this; it forbids an invented body. |
| **Q10** | ★ **Who owns roads, terraced fields and walls?** | They are in the reference picture, they are culture rather than geology, and **no domain in this investigation is named for them** (§4.8, group 3). The owner should say whether they are in scope at all. |

---
## 17. Refutation answers

**FIXED** means the section was rewritten. **KEPT** means the report holds its position and says why.
**OWED** names who must answer. §17.1 and §17.2 answer the revision-2 refutations, finding by
finding. §17.3 and §17.4 keep the revision-1 answers below them as history.

### 17.1 Refutation A of revision 2 — the laws and the code (R1–R24)

| Finding | Severity | Answer |
|---|---|---|
| **R1** — the re-anchor and the roughness modulation are never composed; together they give 56° at a 2 m baseline | Blocker | **FIXED, and it changed the design.** The refuter's arithmetic is right and I reproduce it (§1.3b): under the shipped rule at `body.rs:176-184` — which re-normalises every amplitude so the SUM is the relief — shortening the coarsest wavelength eight times multiplies every amplitude-to-wavelength ratio by eight. **§4.6 states the OCTAVE BUDGET revision 2 was missing:** the coarsest octave keeps the amplitude the octave law already gives at 50 km (781.28 m on the home planet), so `A_oct` becomes 2 053 m instead of 14 305 m. §1.3 is re-computed on the re-anchored table, as the refuter demanded, and **D2 and D3 are put as ONE decision with one picture (M12, M16)**. **One leg is KEPT:** the refuter's second consequence — *"the two changes do the same job twice"* — does not hold under the budget rule. The re-anchor moves the SPECTRUM (three coarse octaves out, two fine ones in); the roughness moves the SLOPE (how the fixed budget is spread). COMPUTED, they are orthogonal: at `k = 0.4683` the re-anchored table is 1.72° at a metre, at `k = 0.62` it is 18.87°, and the amplitude at 50 km is the same in both. |
| **R2** — L4b's branch list must be GROWN from the coast, and that growth is priced at zero | Blocker | **FIXED by deletion.** The refuter is right that a per-cell branch list needs the whole chain back to a mouth and a drainage area summed over the catchment, and right that this is the march's cost class moved rather than removed. **The river network LEAVES this report** (§4.4b): what stays is the three conditions any design must meet (bounded work per column, exact-zero support with a fixed sum order, and a water surface to live in), and the honest sentence to the owner that **the first landing has no river** (D11, B2-18). The budget is recomputed without it and is green. |
| **R3** — the budget table sums to 436, not 416; the summary's table sums to 618 | Defect | **FIXED.** The refuter re-added every row and is right. Revision 3's §9.1 sums to **335 ns** and I re-added it; the summary's table carries the **delta** in every row (the L6 row is +13 ns, the one octave the change adds, never the octaves' whole 185 ns), and every chunk figure in §9.2, §9.4 and §15 is computed from that one number and the MEASURED 4 075-column box. |
| **R4** — §6.2 and §9.5 contradict each other about L4a, and the rung verdict uses the nicer arm | Defect | **FIXED, and the contradiction is designed out.** The refuter is right that revision 2 held both halves. L4a is no longer a gradient with a cost of its own: it is one multiply per octave (§4.4a), so at rung 11 it costs exactly what its four surviving octaves cost. §9.5 is recomputed with nothing left out and the honest ratio is **1.9×**, not 4.4×. |
| **R5** — the Mars ceiling row uses `ρ_c = 2 900` while the table declares 2 800 | Defect | **FIXED.** COMPUTED: `243 091 800 / (2 800 × 3.71) = 23 401 m`, so Olympus Mons reaches **93.6 %**, not 97 %. Every other row reproduces at 2 800. The summary, §4.3, §15 and D4 all carry 94 % now, and the ±15 % spread is unchanged. |
| **R6** — `h_smooth` is priced at 55 ns in §4.4a and 135 ns in §7.1 | Defect | **FIXED, once, by removing the quantity.** There is no `h_smooth` gradient any more: the affinity field returns its VALUE and its GRADIENT in one evaluation (§4.2, ESTIMATED 64 ns), the belt profile's gradient is analytic (10 ns inside L3's 20), and §7.1 evaluates the FULL field at 520 ns per sample. One price for each object, stated once. |
| **R7** — the MEASURED counts of unfenced calls do not reproduce | Defect | **FIXED.** Re-run 2026-09-08 with the counting rule stated: `taxonomy.rs` **43 calls on 39 lines**, `celestial.rs` **11 calls on 9 lines**. The refuter is right that no rule gives 22, and right that the report's own standard applies to itself. The conclusion — `vd-physics` computes outside the fence — is unchanged. |
| **R8** — quantisation is a probability reduction, not a cure, and M13 cannot fail | Defect | **FIXED, and the cure is now structural.** The refuter is right on both legs: a body whose true `g` sits within one drift width of a mm/s² boundary quantises differently on two hosts, and a thousand-ulp test passes for any value not near a boundary. **§5.4 adopts the authoring rule** — the owning realm computes and quantises ONCE, and no other process re-derives — and **M13 becomes three things**: a test that the generator never receives a float, a bench printing every fact's distance to its nearest boundary in ulps across 1 000 seeds, and the still-owed target-to-target drift measurement that `D-TERRAIN-1`'s G4 leg also owes. |
| **R9** — the harvest law covers a MINIMUM; an additive lookup needs a stronger law | Defect | **FIXED, and promoted into the design's central rule.** The refuter is right that a sum is not a minimum and that two lists of different length agree only under two extra conditions. **Both conditions are now binding and written as THE SCAN LAW (§4.0):** an exact `+0.0` outside a compact support, and a fixed global index order for the sum. They are also stated as conditions on domain 03's river network (§4.4b). |
| **R10** — "the bend stays irrelevant" is false once the hydrology cell exists, and its cost is unpriced | Defect | **FIXED at the root.** The refuter is right: a ladder cell IS `(face, a, b)`, reaching one runs `unbend`'s four Newton steps, and `bend.rs`'s own doc says the step count is part of the world identity. **§4.0 removes the cell from the definition entirely** — a layer is the full fixed-order scan, and a harvest may not change the answer — so no layer calls `unbend`, nothing is unpriced, and §6.3's claim is true by construction rather than by assertion. |
| **R11** — Airy isostasy omits the water load, and the sea-level solve is not self-consistent with it | Defect | **FIXED for the load; the "implicit solve" leg is REFUSED with a reason.** The refuter is right about the omission and the arithmetic: with the report's own densities the ocean floor sits **6 476 m** below the continental surface, not 4 455 m, and §4.2 now carries the balance and the closed form `d = step/(1 − ρ_w/ρ_m)`. **The refusal:** the solve is not implicit. The loading term is LINEAR in the depth below the water surface, so a sample's loaded elevation is one multiply inside the bisection's own loop (§7.1) and no outer iteration exists. |
| **R12** — the sea level is solved on a field missing most of the variance, and the sampler is unfenced | Defect | **FIXED, both legs.** §7.1 evaluates the **FULL field** at every sample — COMPUTED 520 ns each, 2.13 ms per body — so the model error is zero by construction rather than unstated. And the sample directions come from **a coarse rung's own cell centres through the bend**: a fixed count, address order, no Gaussian, no rejection loop and no new branch. The refuter was right that revision 2 listed a sampler-less hypsometry as fenced. |
| **R13** — L4a's gradient cannot resolve a valley, because `h_smooth` holds no valley-scale content | Weakness | **FIXED, and it is the second-largest design change in this revision.** The refuter is right: L2's affinity runs over 3 000 km and L3's profile under 500 km, so `∇h_smooth` is effectively constant across a 62 m chunk and cannot tell a floor from a divide at the kilometre scale. **§4.4a replaces the gradient transfer with the MULTIFRACTAL WEIGHT**, which acts at every scale the octave table has, costs one multiply per octave (15 ns against 220), and needs no gradient at all. The macro gradient survives only where it belongs — the rain shadow, which IS a hundred-kilometre effect. |
| **R14** — every per-layer nanosecond is ESTIMATED, and one is optimistic | Weakness | **FIXED as far as a report can.** L1's price now covers the bisector normalise the refuter found hidden, once per column rather than once per pair, and is stated for both ends of the plate draw (**70 ns, 130 ns on a 40-plate body**). §9.1 says plainly that only 185 ns/column and 13 ns/octave are MEASURED. §9.2 states the sensitivity the refuter asked for: **the cave-dense chunk uses 72 % of its allowance, so the design survives a 39 % error and is red beyond it**, and D11 orders the work so the cheap, high-value layers land first. **M3 remains the gate and it runs before acceptance.** |
| **R15** — the ridges and the drainage are independent, so crests will not be divides | Weakness | **PARTLY FIXED, PARTLY OWED.** The multifractal weight builds the roughness FROM the height, so the rough ground is the high ground at every scale and the worst case the refuter names — a smooth ridge beside a corrugated plain — cannot occur. **But a crest is still not proved to be a DIVIDE**, because there is no drainage in this report at all. **§11 adds the crest–divide agreement to the oracle's statistics (M9)**, and it becomes an acceptance test domain 03's river network must pass. |
| **R16** — `σ_y` is calibrated against a different material property from the one the formula names | Weakness | **FIXED.** The refuter is right that an unconfined compressive strength and a long-term yield stress under a mountain root are two properties. §4.3 now says: `σ_y` is a **CALIBRATION on one body**, the laboratory comparison is a sanity check of magnitude and not a validation, and what the design uses is the `1/g` TREND. |
| **R17** — the layered mesa, butte and cuesta are missing from the stack and from the limit table | Weakness | **FIXED, and it became a new layer.** The refuter is right that the caprock signature is most of what a desert vista is made of, right that it comes from differential erosion on horizontal strata, and right that the crate already holds them. **§4.4e adds the hardness term for an ESTIMATED 5 ns**, and §4.8's table lists the mesa, the bench and the cuesta in the group the field CAN make. Refuter B found the same gap from the believability side (B2-12). |
| **R18** — the half-cell cut-off's SL8 safety is asserted, and it depends on an unbuilt tier rule | Weakness | **FIXED, in the refuter's own words.** §6.2 now reads: the cut-off adds under half a cell at its own rung, and **slice 8's tier rule must keep a cell near a pixel for that to be invisible; the measurement is owed and it is slice 8's.** §10.6's SL8 line says the same. |
| **R19** — a 4 KB `Copy` body definition is a real change and no decision states it | Weakness | **FIXED.** `PLATES_MAX` drops from 64 to **40** — the law's own ceiling, with no power-of-two padding — so the array is ESTIMATED 2 240 bytes, and **D17 is a new decision row** (in the struct, behind a reference, or re-drawn on demand) with **M10** printing the size and the copy cost first. **W10** is a new door. Refuter B raised the same (B2-24). |
| **R20** — "no magic numbers" is claimed passed while the SHAPES are unwritten | Weakness | **FIXED.** §10.5 is retitled *"passed for every SCALAR, OWED for every SHAPE"* and lists the seven shapes by name — the transfer polynomial, `f(c)`, `g(a)`, the belt profile, the warp, the multifractal weight's pair, the hardness curve and `r_min` — each decided on a picture. **The finite-difference step `δ` the refuter could not find is gone with the finite difference itself** (§4.2). |
| **R21** — the snow-line table's `T_warm` inputs are asserted inside a table marked COMPUTED | Weakness | **FIXED.** The column is labelled **INPUT, ESTIMATED**, sourced as a warmest-month climatology, and the text says the arithmetic is a division and the agreement is a result of five chosen inputs. **M11 extends to fitting them.** |
| **R22** — "the cell is still 12 bytes" is a ruling's format, not a code fact | Weakness | **FIXED.** §0 carries a row naming both: the crate's generated `Cell { stratum, gap }` at **two bytes** (`chunk.rs:64-69`) and the **stored twelve-byte record** of ruling V6 part B, which slice 9 has not built. §4.4c and §10.6 now say which one they check, and W3b says the water change moves cells without changing either. |
| **R23** — the producer of `BodyFacts` is never named, and every candidate touches a law | Weakness | **FIXED.** §5.4 names it: **the body's own realm**, in its own shard, which already links `vd-physics`. It answers the SL1 question the refuter raised: SL1 clause 3 forbids a child asserting its own PLACEMENT, and a mass, a spin and an insolation are what the body IS — the same class as a hull stating its rating and a realm stating its reach. **And SL1 clause 5's structural fence is stated as owed to the slice**: the record's type may not name a position, a velocity or a frame. |
| **R24** — small corrections | Note | **ALL FIXED.** The noise pin is `noise.rs:231` (assertion) and `:235` (constant); `Stratum::Water`'s variant is `strata.rs:18`; `D-TERRAIN-3` is "ONE RUNG PER **REALM**"; the compile-time assertion is INTEGER arithmetic (`0 < 2`, not `0.76 < 2`); a 50 km cap makes the coarsest wavelength a CONSTANT above a 200 km radius, not a draw; the clamp arm still fires for a 16-rung body and survives through D13; D7's precedent (`L_SUN_W`, `RHO_ROCK_KGM3`) lives in the UNFENCED crate and is a weaker precedent than revision 2 claimed; and §9.3's ceiling row is spelled out. |

### 17.2 Refutation B of revision 2 — believability, cost and the owner (B2-1…B2-25)

| Finding | Severity | Answer |
|---|---|---|
| **B2-1** — the river network is not a lookup; its harvest is the whole-planet pass §3.3 refuses | Blocker | **FIXED by deletion**, with refuter A's R2. The refuter is right that the tree's growth and the drainage accumulation both leave the cell, right that a 2 048 m cell grid on the home planet is 39.6 million cells — the same object §3.3 refuses — and right that an accumulation order and a frontier tie-break bring back the drift classes §8.1 claimed were zero. **The river leaves this report** (§4.4b), the three conditions it must meet are written for domain 03, and §8.1's rows are corrected. |
| **B2-2** — the hydrology cell inverts the ladder's cost order; the top rung becomes dearer than rung 0 | Blocker | **FIXED at the root.** The refuter's arithmetic is right and I reproduce it: at rung 11 a chunk spans **126 976 m** and touches **3 844** hydrology cells, one per column. **§4.0's SCAN LAW removes the harvest from the definition**, so a coarse rung scans exactly what a fine rung scans and stays cheaper because it has fewer octaves and fewer layers (§9.5: 1 118 µs against 2 075 µs). The refuter's closing point — that letting the hydrology rung follow the asking rung would make `height_m` depend on the asker — is exactly why the cure had to be the invariance of the ANSWER, not a better key. |
| **B2-3** — the roughness modulator has no stated size, and both readings break a promise | Blocker | **FIXED, with the third reading.** The refuter is right that revision 2 never stated `r`'s range and right that a multiplier above 1 explodes the band (224 531 m) while a constant-sum redistribution breaks the ladder bound (5.1× at rung 9). **§4.6 states it: `r ∈ [r_min, 1]`, downward only**, because the budgeted table IS the rough state. Then the band is untouched (`Σ amp × r_max = Σ amp`) and the ladder bound is untouched. |
| **B2-4** — the re-anchored table has no amplitude budget, so D3 makes the world eight times steeper | Blocker | **FIXED**, same as R1. The refuter is right that `body.rs:169-183` scales the amplitudes to the relief and that revision 2 changed the wavelengths without saying what happened to the sum, and right that `h_max` bounds a MOUNTAIN and not an octave sum. **§4.6's budget rule is the missing number**, and §9.3 shows the band SHRINKS on the octave term as a result. |
| **B2-5** — `BodyFacts` cannot carry the climate the report builds on it | Blocker | **FIXED.** Every row of the refuter's table is right. The record grows from seven integers to **thirteen**: the mean molecular weight (so `c_p` and the lapse rate exist), the atmospheric surface pressure (so `w_rivers` and `w_aeolian` exist), the pole azimuth beside the obliquity (because one angle cannot orient an axis), the eccentricity and the year length (the owner's word *trajectory*), and a `solid_surface` bit (so `w_terrain` and D13 have an input). **52 bytes of payload, ESTIMATED 80 on the wire.** |
| **B2-6** — the affinity noise is charged to a budget line that does not grow, and the corrected cost is RED | Defect | **FIXED, and the correction is adopted.** The refuter is right that 13 ns is one octave and cannot also be three noise calls. §9.1 now pays for the affinity explicitly — **64 ns, value and gradient together** — and the four-times-repeated gradient is gone with the finite difference. The allowance is also corrected to the refuter's **1 153 ns** on the MEASURED 4 075-column halo box. The result is **7.47 ms on a cave-dense chunk, green**, because the work was removed rather than the number argued down. |
| **B2-7** — today's octave ladder already breaks the half-cell rule at four rungs | Defect | **FIXED, and the finding is promoted to a headline cause.** The refuter is right and the premise is withdrawn. My own replication finds it at **six** rungs, not four (6, 7, 8, 9, 10 and 11, up to 1.43×), and the exact law is stated: the dropped amplitude grows as `(1/k)^L` against a cell of `2^L`, so any body with `k < 0.5` — half the shipped draw range — fails at coarse rungs. **§4.6's LADDER CAP makes the rule true at every rung by construction**, which is the result the refuter said the report should claim. |
| **B2-8** — Mars's ceiling is arithmetically wrong, and it is D4's headline evidence | Defect | **FIXED**, same as R5. 23 401 m, 93.6 %. |
| **B2-9** — the isostatic step omits water loading, so the 4 455 m agreement with Earth is an artefact | Defect | **FIXED**, same as R11, and the refuter's second point is adopted too: **the load varies per body**, because D5 draws the water inventory per body, so a dry world's basins stand shallow and a wet world's stand deep. §4.2 carries the balance, the closed form and the per-body statement. |
| **B2-10** — the quantisation cure is stated as absolute and is conditional; M13 cannot fail | Defect | **FIXED, and the refuter's own cure is the one adopted.** §5.4's AUTHORING RULE: the integer is authored once by the realm that owns the body, and no other process re-derives it. M13 is rewritten so it can fail. |
| **B2-11** — SL6's "find the local formulation first" is not attempted | Defect | **FIXED as an option the owner sees; the recommendation is KEPT.** The refuter is right that revision 2 never considered moving the body's physical DRAW below the fence. **D8 now carries it as option (iii)**, with its cost stated: today's mass–radius law is a `powf` fit, the fenced form would draw a radius and a density and multiply, and **it re-draws every body in the world**, not only terrain. The recommendation stays "cross the record", because the owner has not asked for the world to be re-drawn — but the option that opens no lane is on the page, which is what SL6 asks for. |
| **B2-12** — no lithology reaches the surface, so the picture's cliff bands cannot exist | Defect | **FIXED, and it is the best believability finding of this refutation.** The refuter is right that the strata exist and the height field never reads them, and right that one hardness term produces benching with no new evaluation. **§4.4e is a new layer at ESTIMATED 5 ns**, and it needs **no per-direction rock map**, so the benched cliff is available whichever way D10 rules — which also answers the refuter's charge that D10 option (i) hid a cost in LOOK. |
| **B2-13** — climate is baked into the static shape, and the owner asked for weather | Defect | **FIXED, in one table.** §4.5 now labels every L5 operator **SHAPE** or **COVER**: the carved valley, the notch and the dune field are shape and the seed may decide them; **the snow you see on a ridge is COVER**, a live diff from the owning realm, and it goes to domain 05 with §5.5. The refuter is right that a fixed snow line is a painted-on cap. |
| **B2-14** — Held & Hou's result is read as a sine where the source gives an angle, and the band count has no law | Defect | **FIXED, both halves.** COMPUTED: the substitution costs 0.3 % at twice Earth's spin and **20 degrees at half of it**, exactly as the refuter says. §4.5 states both readings, keeps the fenced test, and folds the conversion into the constant **M11 fits**. **The "cells per hemisphere" column is DELETED** — the refuter is right that it had no derivation anywhere — and the band count goes to domain 05 with Q5. What the biome reads is the Hadley EDGE, which does have a law. |
| **B2-15** — the hypsometry solves on a field that is not the one drawn | Defect | **FIXED**, same as R12, and the two companions with it: `h_smooth`'s double price is gone (R6), and **the lake double-count is stated** — it is zero today because this report has no lakes, and it becomes one subtraction the moment domain 03 lands. |
| **B2-16** — §9.5 contradicts §6.2 about L4a, and the 4.4× headline depends on it | Defect | **FIXED**, same as R4. The honest ratio is 1.9×. |
| **B2-17** — the body draw runs on the client's message path, which is a tick hitch | Weakness | **FIXED.** The refuter is right: `chunks.rs:266-289` calls `from_seed` inline. COMPUTED with revision 3's costs, a system stating 28 bodies is **60 ms** — four dropped frames. **§7.1 says `from_seed` runs on a worker on both hosts, and W9 is a new door.** |
| **B2-18** — the recommended first landing has no river, and the reference picture has one | Weakness | **FIXED, and the sentence is in the summary as well as in D11.** The refuter is right that this must be said plainly to the owner. M4's pass mark is extended, and the river is domain 03's with a named interface rather than an implied promise. |
| **B2-19** — option (c) is priced at 90 ns against a MEASURED 185 ns for the same thing | Weakness | **FIXED.** §8.1 carries 185 ns with its citation. The refuter is right that understating a rejected option is the same unfairness the report charged revision 1 with. |
| **B2-20** — the picture holds things no layer makes, and §4.8 names only the overhang | Weakness | **FIXED.** §4.8 is now three groups, and group 3 lists what belongs to nobody: **roads, terraced fields and walls have no owner in this investigation at all**, which is now **Q10** for the owner; the haze belongs to the renderer and its input is in the record; moraines and fjords are named as beyond L5's glacial switch; the benched cliff moved into group 1 by B2-12. |
| **B2-21** — terms the report teaches with are used before they are explained | Weakness | **FIXED, all nine.** Plain-word explanations now stand at first use for **stagnant lid** (§4.7), **Nyquist** (§1.6), **ulp** (§5.4), **flexural wavelength** (§10.5), **continentality** (§4.9), **foreland basin** (§4.3), **size-frequency distribution** (§4.7), **abyssal hills** (§7.2) and **cirque** (§4.5), each with a game example. |
| **B2-22** — `w_terrain` is left OWED, so D13's own answer has no input | Weakness | **FIXED.** §5.4's record carries a `solid_surface` bit, drawn by the forest from the body's class. D13 has an input, and the ice-giant coverage arm is pinnable as *"this real body returns `None`"*. |
| **B2-23** — the super-earth's ceiling is understated, in the report's own favour | Note | **FIXED.** §4.3 says so: the same chain computes that body's mean density at 8 608 kg/m³, a compressed crust is denser than 2 800, so the true ceiling is lower than 3 004 m and D4's case is stronger than the table shows. |
| **B2-24** — a 2–4 KB plate list makes `BodyDefinition` a heavy `Copy` type | Note | **FIXED**, same as R19: `PLATES_MAX = 40`, **D17**, **W10**, and M10 measures first. |
| **B2-25** — a triple junction is decided by a tie-break, and nobody says what it looks like | Note | **FIXED.** §4.1 says what happens: near the junction the second-nearest site changes over a few kilometres, so the boundary KIND changes there and three profiles meet at a point with their uplifts summed. **The index tie-break decides only exactly-equal distances, which is a set of measure zero and never a visible line.** The picture is owed to M4. |

### 17.3 Refutation A of revision 1 — the laws and the code (F1–F30), kept as history

| Finding | Severity | Answer |
|---|---|---|
| **F1** — L4a's cost is under-estimated 4 to 28 times, so the 8 ms verdict falls | Blocker | **FIXED.** The refuter is right, and the arithmetic came from revision 1's own table. **The 32-step march is DELETED.** §4.4 replaces it with a valley transfer (one gradient of the SMOOTH macro field, 220 ns) and a river-network LOOKUP (256 ns). §9 is rebuilt on the COLD chunk with an allowance of 1 223 ns/column, and it names its own red row (8.68 ms on a cave-dense chunk with rivers). The second half — a 3-D domain warp needs 3 extra noise calls — is also FIXED: L3 is priced at 88 ns, not 45 ns. |
| **F2** — the physical facts are UNFENCED floats and nothing snaps them; an SL10 no-drift hole | Blocker | **FIXED, and it is the most important fix in the revision.** The refuter is right on all three legs, including quoting `body.rs:86` against revision 1. §5.4 now **quantises every incoming fact to an integer grid** (gravity in mm/s², spin in seconds, obliquity in milli-degrees, and so on), with a table of grids and the reason for each, and **M13 pins it** in the exact shape of `home.rs:48`. D9 makes it a decision the owner sees. |
| **F3** — a river and a lake have no representation, and the report promises both | Blocker | **FIXED.** The refuter is right: `chunk.rs:207-209` holds one water rule for a whole body. §4.4c adds a **per-column water surface** in `ColumnField`, so `fluid_at` reads the column, not the body. The 12-byte record is genuinely unchanged (`Stratum::Water` exists), but `fluid_at`'s signature and three call sites change — stated as W3b, not hidden. |
| **F4** — the per-column roughness breaks the ladder's bound and puts a step between rungs 8 and 9 | Blocker | **FIXED.** The refuter's arithmetic is exact: `|r₀ − r₉| × 13 983 m` is four times the whole bound. §4.6 states a binding rule: **a MULTIPLIER may read only rung-independent layers** (L1, L2, L3's profile), and a layer that is cut off may only ADD. Then `r` is rung-independent by construction and `Σ dropped × r_max` is a bound again. |
| **F5** — the plate lookup is face-indexed and the per-chunk harvest makes `h` depend on the asker | Defect | **FIXED.** Both halves are right, including that the "62 m" argument holds at one rung out of twelve (at rung 11 a chunk column spans 127 km). §4.0 replaces the bucket grid with the **hydrology cell** — an address, not an asker — and states **THE HARVEST LAW** (the candidate list must be a strict superset), with M6(b) as its test. |
| **F6** — removing the three coarsest octaves leaves the top rung with ONE octave, not three | Defect | **FIXED.** The refuter's count is right (a 50 km cap with a 30 m floor gives 11 octaves, so `keep = 0` and the clamp arm fires). §4.6 cures it by ALSO dropping `SHORT_WAVE_M` to 2 m, which gives 15 octaves and 4 at the top rung. §6.1 and §9.5 are recomputed. |
| **F7** — "the layout changes only `h`" contradicts "the layout may decide the ROCK" | Defect | **FIXED.** The refuter is right and the code proves it (`body.rs:200` draws one bedrock per body; `strata.rs:189` takes no direction). §10.3 now prices a rock map as a new signature, new HR5 branches and a world-identity bump, and D10 offers "no rock map" as a genuinely cheaper option. W8 is a new door. |
| **F8** — the fence table misses at least two transcendentals | Defect | **FIXED.** The refuter is right about all three. The hotspot chain becomes `normalise(s + t·u_k)` (§4.3), the spherical Voronoi area is deleted in favour of a fixed-sample hypsometry (§7.1), and the wind's Coriolis turn becomes `p·c₁ + (d × p)·c₂` normalised (§4.5). Each is written out in §10.1 rather than hidden behind a "yes". |
| **F9** — `h_max` grows the band by 30 %, and no table says so | Defect | **FIXED**, and merged with B2, which found the same thing more completely. §9.3 gives the band's exact arithmetic pinned on the MEASURED 29 105, a table of consequences (up to 1 269 chunks per radial column), and **a design rule: the band is sized on the recipe's DRAWN bound, never on `h_max`**. The refuter's implicit question — does `h_max` replace the RELIEF or the BAND? — is answered: it bounds the belt amplitude draw; the band is sized on the sum of drawn amplitudes. |
| **F10** — the SL6 ask is internally inconsistent and never states the refusal | Defect | **FIXED.** Both halves are right. §5.4 states the size ONCE (28 bytes of payload, ESTIMATED 56 on the wire), as **seven integers** rather than nine or eleven `f64`s, as a **new tag** (never a field in `SurfaceStmt`), and it states the **refusal**: a client without `BodyFacts` must not derive that body's surface, and no default is ever substituted. |
| **F11** — "the ladder would accept a gas giant" is false | Defect | **FIXED.** The refuter's arithmetic is right: `N_MAX = 2²⁶` caps the radius at 42 723 km, so Jupiter and Saturn are already refused and the gap is the ICE GIANTS. §4.7 and D13 now ask the right question. |
| **F12** — "the extractor and the collider are unchanged" rests on a measurement taken on a 2° world | Defect | **FIXED.** The refuter is right and it is the "never assume, measure" law. §10.6 now reads: **the extractor's error must be RE-MEASURED on the new field, and the seating rule with it**, as M15. "Unchanged" was an UNMEASURED claim dressed as a result. |
| **F13** — no overhang, arch or rock pillar is possible, and the picture shows pillars | Defect | **FIXED**, and promoted to the summary as Cause 3. The refuter is right and doc 01 states it outright at `:190`. §4.8 is a new section that states the limit as a table, and **D14** hands the 3-D term to its own decision. Revision 1 letting a reader believe the stack reached the picture was the worst kind of omission. |
| **F14** — `h_max` is presented as a law; it fails its third point and its calibration is a datum artefact | Defect | **FIXED for the framing, KEPT for the recommendation, and the Moon objection is REFUSED with a reason.** §4.3 restates the relation physically as `h_max = σ_y/(ρ_c·g)`, so Everest's calibration produces a checkable rock strength (243 MPa, inside granite's measured range) rather than an arbitrary `C`. The claim "a law and not a fit" is withdrawn; it is a **scale with a ±15 % spread** (Earth 100 %, Mars 97 %, Venus 112 %). **The Moon does not falsify it:** a body that reaches 20 % of its ceiling has not exceeded it, and the Moon has no orogeny at all. Revision 1's "rubble body" excuse was wrong (the Moon is differentiated) and unnecessary. |
| **F15** — the super-earth's gravity has no source and disagrees with the report's own density | Defect | **FIXED, and the refuter's replacement is also refused.** Revision 1's 24.8 was unsourced — the refuter is right about that. But 16.8 holds the density at 5 000 kg/m³, which is wrong for a 10-Earth-mass planet: it is compressed. COMPUTED from the Zeng rocky mass-radius relation `R ∝ M^0.27`: 10.4 Earth masses, 8 608 kg/m³, **`g = 28.9`**, ceiling 3.0 km. D4's case is therefore **stronger** than revision 1's (2–6× over-relief, not 2–5×) and now correctly sourced. |
| **F16** — the report refuses the bake on client cost and never prices its own stack on the client | Weakness | **FIXED.** §9.4 is a new section with the same table for the client: 4.18 ms → 5.78 or 6.76 ms per chunk, 239 → 173 or 148 chunks/s, a **28–38 % cut**, and it names slice 8's residency band as what that cut sizes. |
| **F17** — the hydrology only CUTS; no flood plain, delta or alluvial fan | Weakness | **FIXED.** The refuter is right that a valley floor is flat because it is FILLED. §4.4d adds alluvial fill, the angle of repose (32–37°) and a delta, all as transfer-curve terms inside L4a's 20 ns. |
| **F18** — the stated validation of the replication does not touch the noise | Weakness | **FIXED.** The refuter is right: the sea offset never calls `noise3`. §1.1 now validates against **the crate's own noise pin** (`noise.rs:228`), bit for bit, which exercises the hash, the gradient table, the fade and the blend ORDER. The sea-offset check is kept as a second leg. |
| **F19** — "amortised 470 ways" is true and irrelevant to the gate it is offered against | Weakness | **FIXED.** The refuter is right and revision 1's own parenthesis admitted it. **§9 is written on the COLD chunk only**, and the amortisation is named as slice 8's residency argument, not as a budget argument. |
| **F20** — several new magic numbers arrive under the report's own gate | Weakness | **FIXED.** §10.5 gains a second table deriving every new number: `PLATES_MAX` from the plate law's ceiling, the hydrology rung from the ladder, the blend width from the flexural wavelength, the notch band from the significant wave height, the 24 and the 4 096 from their own accuracy targets. **`K = 32` and the march step size are DELETED with the march.** §4.2's *"it needs no tuning at all"* is withdrawn: the 35 km and 7 km thicknesses are named as Earth calibrations, and their `1/g` scaling is stated. |
| **F21** — HR5 at 100 % cannot be reached on the home planet alone | Weakness | **FIXED**, and merged with B-W2. §10.6 now states which REAL bodies of THE world carry which arm — a moon of the home system for `w_plates = 0` and `w_rivers = 0`, an ice giant for `w_terrain = 0` — and says HR5 blocks the slice without them. `home.rs` must state them, each with a pin. Q9. |
| **F22** — weather is the owner's second sentence and the report asks no SL6 question for it | Weakness | **FIXED**, and merged with D2. **§5.5 is a new section: SL6 ask TWO**, put in SL6's four parts with the unanswered parts marked UNANSWERED and handed to domain 05. The summary and §15 now say **two asks**, not one. |
| **F23** — the ore fence is half-closed | Weakness | **FIXED.** The refuter is right that a published rock map is a prospecting prior, and the residual has a size (a 20× narrowing). §10.3 states it and **D10 gives the owner three options instead of one recommendation**. |
| **F24** — cite drift | Note | **FIXED.** The octave table is `body.rs:99-100`; `Gf::lesser`/`greater` are `gf.rs:123,129`; the latitude line is `height.rs:45`. Every citation in revision 2 was re-opened. |
| **F25** — a MEASURED grep that does not reproduce ("four hits") | Note | **FIXED.** Re-run: **two** hits. The conclusion (the world has no spin) is unchanged and both refuters confirm it. |
| **F26** — "`Gf` … and NOTHING else" is loose | Note | **FIXED.** §0's row now lists `lesser`, `greater`, `clamp`, `is_finite` and says what is actually absent (`f64::min`/`max`, `mul_add`, `powi`, every transcendental). It matters: D11's clamp cure depends on it. |
| **F27** — "visible from" omits the observer's own horizon | Note | **FIXED.** §1.5 gives both columns. A 1 000 m crest is seen from 85 336 m, not 81 863 m. |
| **F28** — 14 304 against 14 305 | Note | **FIXED.** The relief is 14 304.887 m; `relief_whole` (the integer the band uses) is 14 305. Both appear with their own names. |
| **F29** — the offline oracle meets a standing law it does not name | Note | **FIXED.** §11 answers it in a paragraph: the banned thing is a second implementation OF THE WORLD; the oracle produces no world, only four statistics, and never links into a product build. |
| **F30** — S6-10 is over-read | Note | **FIXED.** §12's note no longer borrows S6-10, which is about the cave lattice step. The conclusion stands on its own. |

### 17.4 Refutation B of revision 1 — believability, cost and the owner, kept as history

| Finding | Severity | Answer |
|---|---|---|
| **B1** — L4a costs 30 to 190 times what §9 prices | Blocker | **FIXED**, same as F1. The refuter's allowance arithmetic (4.7 ms ÷ 3 844 = 1 223 ns/column) is adopted as §9's frame, and the march is deleted. The point that an analytic gradient of `noise3` costs MORE than the value, not less, is accepted and is why §4.4 uses a finite difference of the SMOOTH field instead. |
| **B2** — the ladder band is never recomputed; every cell address moves | Blocker | **FIXED.** The refuter is right, including that revision 1 never used the words `crust_m`, `above_m`, `relief_whole` or `cells_in_band`. §9.3 is a new section with the exact arithmetic pinned on the MEASURED 29 105, the consequence table, and the design rule that the band is sized on the DRAWN bound. **W1b** is a new one-way door: `floor_m` moves, so every cell address on every body moves. M14 measures it. |
| **B3** — crust type is per plate, so every coast is a plate boundary and a passive margin cannot exist | Blocker | **FIXED, and it is the best believability finding in either refutation.** The refuter is right: most of Earth's coast is a passive margin, and revision 1 made every one of them impossible. §4.2 makes the crust a **FIELD** — a boundary-independent affinity noise plus a per-plate offset — so a continent's edge normally falls INSIDE a plate, and a plate may hold a continent and an ocean floor as Earth's African plate does. |
| **B4** — a single-valued height field cannot make a cliff, a pillar or an arch | Blocker | **FIXED for the pillar, the arch and the overhang; PARTLY REFUSED for the cliff.** §4.8 is a new section and D14 hands the 3-D term to its own decision. **The one refusal:** a single-valued field CAN hold a near-vertical face, because a slope discontinuity scaled by a belt amplitude is a vertical step in the limit and the transfer curve can steepen it. What it cannot hold is an UNDERCUT. §4.8's table separates the two rather than lumping them. |
| **B-D1** — the macro field is asserted smooth for L4a while §4.3 makes it discontinuous | Defect | **FIXED, and it changed the design.** The refuter is right that `Gf::abs` has no derivative at zero and that the crest is exactly where the march needed one — and right that it is a determinism hazard, not only a quality one. §4.4 states a binding ORDER: `h_smooth = L2 + L3's PROFILE only`, the gradient is taken there, and every FOLD is added afterwards. |
| **B-D2** — the weather is not designed, yet the report claims to answer the owner's words | Defect | **FIXED**, same as F22. §5.5 opens the second SL6 ask; the "only one ask" claim is withdrawn from the summary and §15. |
| **B-D3** — the biome function is diagnosed and never replaced | Defect | **FIXED.** The refuter is right that the biome is the owner's FIRST requirement and had no D-row and no measurement. **§4.9 is a new layer L7** that rebuilds `biome_at` from insolation, obliquity, the environmental lapse rate, the circulation bands, the rain shadow and continentality — every input already computed, so ~20 ns — and **D12** asks the owner for the biome list. |
| **B-D4** — the hypsometric bisection needs a spherical Voronoi area, which is not fenced | Defect | **FIXED.** All three legs of the refusal are right (the sort, the `acos`, the `erf`). §7.1 replaces it with a **fixed-sample hypsometry**: 4 096 seed-drawn directions in a FIXED order, evaluated once in `from_seed` at a COMPUTED 0.55 ms, standard error 0.78 % of the surface. Fenced, order-fixed, no sort. |
| **B-D5** — the boundary distance `b` has no fenced form, and every L3 profile consumes it | Defect | **FIXED.** The refuter is right that `b` decides where every range sits and appeared nowhere in the fence table. §4.1 writes it: `b = R·(d · n̂)` with `n̂ = (s₂ − s₁)/|s₂ − s₁|`, and **states the error bound** (the sine against the angle: 0.13 % at 292 km, 0.99 % at 819 km), which is under 0.4 % inside every profile width. It is now a row in §10.1 with a property test owed. |
| **B-D6** — the snow line is the freezing level of a dry adiabat, and it is 3× wrong on Earth | Defect | **FIXED.** Every leg of the refusal is right. §4.5 uses the **environmental** lapse rate `(g/c_p)·w`, a **warmest-month** freezing level, and a **latitude**. COMPUTED Earth check: 3 977 m at the equator, 4 592 m in the subtropics, 3 054 m in the Alps, 1 515 m at 70° — the real curve's distinctive subtropical maximum, within 20–50 %, against revision 1's 1 537 m everywhere. And **`c_p` is derived from the mean molecular weight**, not asserted, so a hydrogen–helium body is not given Earth's air. D7's constant list drops `c_p`. |
| **B-D7** — `h_max` is a one-point fit whose second point has a different mechanism and whose third fails | Defect | **FIXED for the framing; the third point is REFUSED.** See F14. The relation is restated physically and its "law" claim is withdrawn. The Moon is a body that never reached its ceiling, which is not a failure of a ceiling; and revision 1's "rubble body" excuse was factually wrong, which the refuter is right about. |
| **B-D8** — two COMPUTED values of the same quantity disagree by 38 % | Defect | **FIXED.** The refuter is right that a COMPUTED number which does not reproduce inside its own document is an estimate. §1.2 and §1.3 now use **ONE sampling method** (700 slope samples, 120 relief lines), and the 1 km median relief reads **21.52 m** in both by construction. |
| **B-D9** — the spectrum stops at 48.8 m, and the report never names the constant that stops it | Defect | **FIXED, and promoted to a headline.** The refuter is right that this is a stronger and simpler statement of "no details at all", and right that raising `k_rough` does not fill the hole. **§1.6 is a new section**, the summary names it as Cause 2, §4.6 drops `SHORT_WAVE_M` to 2 m, and **D3** puts it to the owner. The refuter's own arithmetic checks: at `k = 0.62` the 48.83 m octave carries **10.89 m**, which I reproduce exactly. |
| **B-D10** — the rung cut-offs are new LOD steps, and the SL8 line does not see them | Defect | **FIXED.** The refuter is right that revision 1's SL8 line looked only at the plate boundary and missed the rung direction. **§6.2 replaces the chosen cut-offs with a DERIVED rule**: a layer is dropped at the first rung where its own amplitude falls under half a cell. Then no rung change moves the surface by more than half a cell, which is the SL8 statement the vista gate wants. M7 tests it. |
| **B-D11** — `w_craters` goes negative | Defect | **FIXED.** The refuter's arithmetic is right (the sum reaches 3, so `1 − resurfacing` reaches −2). §4.7 clamps it, and `Gf::clamp` exists at `gf.rs:135` — which is exactly why the refuter's other finding about §0's `Gf` row (B-W6) mattered. |
| **B-D12** — "31 hours" and "one and a half cells" are invented | Defect | **FIXED.** The whole paragraph is DELETED. The refuter is right on every point, including that a slower spinner has WIDER cells rather than fractional ones. §4.5 replaces it with the **Held–Hou** law, a COMPUTED table, the fenced test `|dir · pole_axis| < sin φ_H`, and **an honest statement of its error** (38° against Earth's real 30°, 2 bands against 3), with M11 owing the calibration. |
| **B-D13** — §9.2's numbers do not follow from §9.1's | Defect | **FIXED.** The refuter is right about both the 35 µs gap and the underived +346 µs. §9.5 derives every row from §9.1's table and §6.2's cut-off rule: rung 0 2 309 µs, rung 6 2 057 µs, rung 11 527 µs, a 4.4× ratio. |
| **B-D14** — the `BodyFacts` byte count is stated three ways and none matches the field list | Defect | **FIXED**, same as F10. Stated once: seven integers, 28 bytes of payload, ESTIMATED 56 on the wire. |
| **B-W1** — "the record — unchanged" is asserted against D10's own recommendation | Weakness | **FIXED**, same as F7. |
| **B-W2** — HR5's zero-weight arms cannot be covered where the tests run | Weakness | **FIXED**, same as F21. The refuter's own cure — another REAL body of THE world, which SL5 permits — is the one adopted. |
| **B-W3** — the plate accelerator is face-indexed, unsized, and its correctness condition is unstated | Weakness | **FIXED**, same as F5. THE HARVEST LAW is the stated condition, M6(b) is its test, and the harvest is a fixed-size structure because `BodyDefinition` is `Copy` and `PartialEq`. |
| **B-W4** — the bake is refused at the pessimistic corner while (d) is priced at the optimistic one | Weakness | **FIXED.** The refuter is right and it is a fairness point. §3.3 now states the cheapest useful corner honestly (**6.3 MB and 0.19 s on 14 threads for a 512-edge grid**) and refuses the bake on the four structural reasons that hold at every corner. §8.1's boot-cost row gives the range, and it also stops claiming the layer stack's own boot cost is zero (it is 0.55 ms). |
| **B-W5** — L3's warped octaves are under-priced by about half | Weakness | **FIXED**, merged with F1's second half. L3's noise part is 78 ns (3 warp + 3 ridged × 13 ns MEASURED), and L3 total is 88 ns. |
| **B-W6** — §0's `Gf` row is wrong about the code | Weakness | **FIXED**, same as F26. |
| **B-W7** — the home planet's `g` is a guessed density used as settled in four places | Weakness | **FIXED.** Every number derived from `g` now carries its range: the ceiling is 16 808–23 564 m and the lapse rate 2.44–3.43 K/km, between a Mars-like and an Earth-like density. **M2 is one line of bench** and it is second in the measurement list. |
| **B-W8** — thermal erosion, talus, lakes and river water are absent | Weakness | **FIXED.** All four. The angle of repose (32–37°) and the alluvial fill and delta are §4.4d; the river water and the lake surface are §4.4c's per-column water surface. |
| **B-N1** — the Hurst exponent is used past its range | Note | **FIXED.** §1.2 states the definition's range, says the field is not fractional Brownian motion at all, and gives the honest form of the claim. In a document written to teach the owner a term, the refuter is right that this matters. |
| **B-N2** — the grep claims four hits | Note | **FIXED**, same as F25. |
| **B-N3** — "visible from" omits the observer | Note | **FIXED**, same as F27. |
| **B-N4** — age is listed as an input and then drawn inside the generator: a SECOND age | Note | **FIXED.** The refuter is right, and it is the same shape as the defect revision 1 used to refuse an option elsewhere. §5.1 makes age a `BodyFacts` field (whole Myr), drawn once by the forest. |
| **B-N5** — revision 1's decision D9 (now **D13**) needs a "has a solid surface" fact that `BodyFacts` does not carry; and `from_seed`'s signature changes | Note | **PARTLY FIXED, PARTLY OWED.** The signature change is now **W5** in the door table, naming `home_body_pin.rs` and `home.rs`. On the missing fact: `w_terrain` is derivable from the facts the record DOES carry — a body with `g` and a radius that put it past the ice-giant threshold, and an atmosphere whose scale height is a large fraction of its radius, has no solid surface. **OWED to the slice: state that derivation, or add an eighth field.** |
| **B-N6** — no measurement makes the owner's own complaint countable | Note | **FIXED.** M4 now carries doc 01's own pass mark: **8–12 orienters per 60° of field of view**, counted by hand on the shot. |
