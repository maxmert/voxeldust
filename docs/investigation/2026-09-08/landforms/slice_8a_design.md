# Slice 8a — THE SPECTRUM: the implementation design

**Written 2026-09-16, on `601a1e1` (Step 17).** This document is the build plan for 8a, the second
item of ruling T1 (`docs/design/owner_decisions_2026-09-16_terrain.md`). It is precise enough that an
agent can build each stage without reading the arc again. It changes no code.

**What binds it.** T1 (the whole of 8 before 9; 8a's scope; L25 first; the reference traced; the cave
stand), F1/F2/F7/F8 (`owner_decisions_2026-09-12_frozen_patches.md`), V13 L5/L12/L25
(`owner_decisions_2026-09-07_voxels.md`), and SL10 (one generator, two hosts, no drift).

**What it is NOT.** It is not the macro solve (8c), not the charter (8b), not the rock map or the
authored override (8d), not the biome (8e), not the third dimension (8f) and not the ocean (8o).
Where 8a needs something one of those owns, this document names a PLACEHOLDER, says exactly how the
later slice replaces it, and proves the replacement moves no address.

---

## 1. THE SCOPE

### 1.0 What the code holds today (read from the source, 2026-09-16)

Every claim in this section is read from the tree, not from a document.

| The thing | Where it lives | What it is today |
|---|---|---|
| The octave table | `crates/terrain/src/body.rs:246-297` | `wave_m` halves from `long_wave_m` while it is over `SHORT_WAVE_M`; `weights[o] = k_rough^o`; `amplitude_m = weights[o] · relief_m / Σ weights`, so **Σ a(o) = relief_m exactly** |
| The ends | `crates/terrain/src/body.rs:40-42` | `LONG_WAVE_CAP_M = 400_000`, `SHORT_WAVE_M = 30`; `const _: () = assert!((LONG_WAVE_CAP_M >> OCTAVES) < SHORT_WAVE_M)` |
| The cap | `crates/recipe/src/height.rs:30`, `crates/terrain/src/body.rs:34-35` | `OCTAVES_CAP = 16`, pinned equal to `vd_terrain::body::OCTAVES` |
| The octave sum | `crates/recipe/src/height.rs:56-100` | `relief()` / `relief_of_table()`: an index loop, `h += octave_term(...)`, one floor at the end |
| One octave's term | `crates/recipe/src/height.rs:97-100` | `(o.amplitude * noise3(o.seed, p)) >> AMP_BITS` — smooth gradient noise, no transform |
| The coarsening | `crates/terrain/src/body.rs:441-449` | `octaves_at(rung)` returns the coarsest `count − rung`, **never fewer than one** |
| The bounds | `crates/terrain/src/body.rs:451-464` | `relief_bound(rung)` = Σ of the live amplitudes; `dropped_bound(rung)` = `relief_bound(0) − relief_bound(rung)` |
| The column bound | `crates/terrain/src/digest.rs:122-146` | a second-order curvature bound per octave, `min(1, (π·edge·f/R)²)` × amplitude, summed |
| The band | `crates/terrain/src/body.rs:368-373` | `crust_m = relief_whole + strata.max_depth_m() + caves.max_depth_m + 64`, `above_m = relief_whole + 64`, where `relief_whole = floor(relief_m) + 1` — **derived from `relief_m`, not from the amplitude sum** |
| The relief law | `crates/terrain/src/body.rs:242-245` | `min(0.004·R, [200, 12 000]) × (0.5 + draw)` |
| The sea | `crates/terrain/src/body.rs:299-303` | `sea_radius = radius + floor((draw·0.7 − 0.4)·relief_m)` — a uniform draw, no solve |
| The draw's fence | `crates/terrain/src/gf.rs:1-25` | `Gf` has add, subtract, multiply, **divide**, negate, **square root**, floor, truncate, abs and comparison, and nothing else |
| The body's build | `crates/client/src/chunks.rs:1491-1515`, `crates/bins/src/bin/shard.rs:404` | `BodyDefinition::from_seed` runs **INLINE on the message path** in `state_surface` |

**The home planet, from `crates/terrain/src/home.rs:47-57`:** seed 4 030 111 653 607 004 909, look
radius 6 370.7 km, ladder `N = 9 961 472`, radius 6 341 670 m, **19 rungs, 14 octaves**.

**Two numbers derived by hand from those constants, owed a one-line test.** The long wavelength
clamps: `6 341 670 × [0.25, 0.5)` is 1.59–3.17 Mm, over the 400 km cap, so **`long_wave_m` is exactly
400 000 m** and `λ(o) = 400 000 / 2^o`. The loop stops at `o = 13` (48.83 m > 30 m; 24.41 m is not),
so **count = 14**, which is what `home.rs:54` asserts.

★ **TODAY'S COARSENING IS ALREADY BY WAVELENGTH, AT A CONSTANT RATIO.** At rung `L` (for `L ≤ 13`)
the finest live octave is index `13 − L`, so its wavelength is `400 000 / 2^(13−L)` and the cell is
`2^L` m. The ratio is `400 000 / 2^13` = **48.83 cells per wavelength at EVERY rung** — a constant,
because a strict halving table dropped one per rung is exactly a wavelength rule. So "drop one octave
per rung" and "an octave survives while `λ ≥ K · cell`" are the SAME rule today, at `K = 48.83`.

★ **AND THE CLAMP ARM BREAKS IT AT THE TOP.** The home planet has 19 rungs and 14 octaves, so from
rung 14 upward `octaves_at` returns the single 400 km octave (`body.rs:446-448`, the
`if keep == 0 { 1 }` arm). The ratio then falls: **24.4 cells per wavelength at rung 14, 12.2, 6.1,
3.05, and 1.53 at rung 18.** Under two cells per wavelength the field aliases. This is a defect the
arc names (`00_proposed_landforms.md` §4.1) and it is in the tree now.

### 1.1 The slope spectrum

**What 8a builds.** The fine octaves' amplitudes stop being a geometric ladder and become a rational
bump in the octave INDEX, anchored on the angle of repose (V13 L5; `03_erosion_rivers.md` §4.2.3):

```
   s(o)  =  S_PEAK / ( 1 + ((o − O_PEAK)·LN_2)² / SIGMA² )      the per-octave RMS slope
   a(o)  =  s(o) · λ(o) / TAU                                    slope to amplitude
```

`o` is the octave index (0 is the coarsest). `O_PEAK = 7`, `SIGMA = 1.4`. The **ANCHOR** fixes
`S_PEAK`: `sqrt( Σ over the fine octaves of s(o)² ) = TALUS_RMS · tan(θ_loose)`, with
`tan(θ_loose) = 0.70` and `TALUS_RMS = 1.0`, which gives `S_PEAK = 0.3998` on a 400 km table with the
fine band 4..13. The **CONSTRAINT** `Σ over all octaves of a(o) ≤ relief_m` is an inequality; spending
less is lawful.

**Which octaves are FINE.** `03` §4.2.1 selects by wavelength: an octave is COARSE when
`λ(o) ≥ 4 · macro_node_m`. The macro node is 8c's. **8a has no macro node**, so 8a selects the fine
band by the same wavelength in metres, stated once as a body constant: the first octave whose
wavelength is under `FINE_ABOVE_M`. On the home planet `FINE_ABOVE_M = 32 896 m` gives exactly
octaves 4..13, which is the band the whole arc's arithmetic was computed on. **8c replaces the
constant with `4 · macro_node_m` and the band is unchanged on the home planet by construction.**

**What it is worth, MEASURED (`09_step0_results.md` M1b, on the OLD home planet, 2026-09-09).** The
amplitudes the formula produces — 496.2 / 401.6 / 319.4 / 198.8 / 79.8 / 25.1 m at 25 km / 12.5 km /
6.25 km / 3.13 km / 1.56 km / 781 m — sum to **1 532 m of fine relief**, at a fine RMS slope of
**0.700 = tan 35°**. The four-station skyline march's largest rise moved from **0.054°–0.067° to
0.138°–0.168°**, a gain of **2.2–2.7×**; the breaks per 60° at the 0.10° threshold moved from **0 to
0.83–1.83**. The stop line (0.10°) is passed. ★ Those wavelengths depend only on the 400 km clamp and
the halving, and BOTH home planets clamp, so **the amplitude table is the same on the new body**; the
break counts are not, and M1b re-runs (§4).

### 1.2 The per-column roughness factor, on a STATED PLACEHOLDER

**What it is.** ONE downward-only multiplier per column on the FINE octaves alone (V13 L5; `03`
§4.2.4). It is what makes a plain flat and a range rough with one table:

```
   m(site)  =  M_MIN + (1 − M_MIN) · smoothstep(0, 1, min(r_raw, 1))        m ∈ [M_MIN, 1]
   a(o, site)  =  a(o) · m(site)                                            the FINE octaves only
```

`M_MIN = 0.06`, so a craton's fine RMS slope is `0.70 × 0.06 = 0.042 = 2.4°` — Earth's plains.

**The placeholder, stated exactly.** `03` sets `r_raw = |∇Z| / SLOPE_REF`, and `Z` is 8c's macro
field. 8a has no `Z`. **8a's `r_raw` is ONE MORE SLOW OCTAVE of the recipe's own noise**, drawn from
its own seed salt at a continental wavelength (the band the biome noises already use — 20 to 120 km,
`body.rs:320-325`), mapped to `[0, 1)` and passed through the smoothstep:

```
   r_raw  =  ( one_octave(roughness, dir) + 1 ) / 2         the noise, in [0, 1)
   m      =  M_MIN + (1 − M_MIN) · fade(r_raw)              `fade` IS the quintic smoothstep
```

`vd_recipe::noise::fade` (`crates/recipe/src/noise.rs:73-80`) is already the quintic
`t³(t(6t−15)+10)`, so no new polynomial enters the fence.

**How 8c replaces it WITHOUT MOVING AN ADDRESS.** An address is `(face, rung, x, y, z)`. The factor
changes the surface's HEIGHT inside a column; it never touches the grid. The grid's `n` and `rungs`
read the look radius alone (`crates/seed/src/ladder.rs:82-125`); `floor_m` and `band_m` read
`crust_m` and `above_m`, which 8a derives from the amplitude sum **at `m = 1`** — the upper bound,
which holds for the placeholder and for 8c's field alike, because both are bounded above by one. So
8c swaps one function, bumps `GENERATOR_VERSION`, and moves not one address. ★ **That property is
built in at 8a by deriving the band from `Σ|a|` at `m = 1`; it is not free.**

### 1.3 The ridged middle band

**What it is.** In the middle of the spectrum the octave stops being a smooth wave and becomes a
crest: `1 − |n|`, re-centred so the amplitude keeps its meaning (`04_detail_rungs.md` §4.4 rule 2;
V13 L5). A smooth octave makes a lump; a ridged one makes a LINE, which is what a spur and a gully
are.

**Which octaves.** `04` sets the band as `λ ∈ [break_length, macro node]` — 128 m to 8 192 m on the
home planet. The M1b-2 instrument (`crates/bins/examples/skyline_march.rs:52-54`) used
`RIDGED_BAND = 5..=9`, which is 12.5 km down to 781 m. **8a states the band in METRES, not in
indices**, so the band does not move when the table's ends move: `λ ∈ [RIDGE_LO_M, RIDGE_HI_M]` with
the pair chosen to reproduce the measured band on the home planet.

**What 8a does WITHOUT the flow alignment.** `04` stretches the ridged octave's sample frame along
the D8 receiver direction, which is the solve's skeleton and belongs to 8c. **8a ships the ridge
ISOTROPIC** — `1 − |n|`, no stretch. That is exactly what M1b-2 measured, so 8a's picture is the one
the owner will have already seen a prediction of. **8c adds the stretch**: one 3×3 matrix per column
from the artifact's D8 bits, three multiplies per sample, no new datum crossing a boundary (`04`
§4.4 leg 1), and no address moves — the same argument as §1.2.

**What it is worth, MEASURED (M1b-2).** The largest rise grew another **1.7–5×** over the smooth
spectrum, to 0.233°–0.689°, and the first breaks at the 0.25° and 0.50° thresholds appeared.

★ **THE CAVEAT THE OWNER MUST SEE AGAIN.** Ridged noise is ONE-SIDED — its mean stands above zero —
so it **lifts the ground by about the band's summed amplitude, roughly 800 m, where the crests
stand** (`09` M1b-2). That is not an error; it is what makes a crest. Three consequences 8a must
carry: the band must hold the lift (§2.6); the sea must be solved against the lifted field, not the
smooth one (§1.6); and the picture stands' altitudes move.

### 1.4 The cap-rock bench

**What 8a builds — and which of the two proposals it is.** There are two in the arc, and 8a takes the
cheap one. `04_detail_rungs.md` §4.5 proposes a radial HARDNESS function (a per-band hash, a bed dip
from the macro stack, the substance changed inside the veneer). That reads the rock map and the dip,
which are 8d's. `03_erosion_rivers.md` §7.3 rule 2 proposes the SHAPE alone, "one comparison per
column", and says it is buildable today. **8a builds the SHAPE: a terrace toward the nearest bed top
at a fixed RADIUS, one hardness for the whole body, no dip, no rock map, and the substance table
untouched.**

```
   S           = the bed spacing, a per-body draw
   k           = the bed index of the column's height against a per-body DATUM radius
   d           = h − (the nearer of the two bed tops)
   u           = |d| / (S/2)                                so u runs over the whole [0, 1]
   terrace(h)  = h − q · d · f(u)             f(u) = 1 − u²(3 − 2u)
```

**Why the HALF spacing.** With `u` taken against one whole thickness the falloff never reaches zero
before the next bed top, so the height JUMPS where the nearest top changes — COMPUTED in `04` §4.5 at
`q = 0.6` on a 40 m band, the jump is **12 m**, six cells at rung 1. With half the spacing `f(1) = 0`
and `f'(1) = 0`, so the pull fades to nothing at the midpoint from both sides.

**The Lipschitz constant, and why it matters to the ladder.** `d(terrace)/dh = 1 − q(1 − 9u² + 8u³)`,
whose extreme over `u ∈ [0, 1]` is at `u = 0.75` and equals `1 + 0.6875·q` — **1.275 at `q = 0.4`,
1.413 at 0.6, 1.550 at 0.8**. The terrace AMPLIFIES every variation below it, so
`relief_bound`, `dropped_bound` and `column_bound` must each be multiplied by it (§2.7). A terrace
landed without that is a bound quietly broken.

**Its rung rule.** The terrace runs while `S ≥ C · cell_m(rung)`, and over its last rung `q` is
scaled linearly to zero as `S/cell` falls from `2C` to `C`. At `C = 8` and `S = 200 m` on the home
planet: full strength to rung 4 (16 m cells), fading through rung 4 to 5, gone by rung 5.

**What it does NOT deliver.** No overhang: the cell fold takes `greater` and can only REMOVE
(`crates/recipe/src/cell.rs:400-405`), so a caprock cap wider than its shaft is **withdrawn**
(`04` D4-7b), and 8f owns the question. No mesa whose LID is a different rock: that is 8d's rock map.
A vertical face still comes from the column-to-column height jump, which on 1 m cells is a wall.

### 1.5 The relief law

**Today (`body.rs:242-245`):** `min(0.004·R, [200, 12 000]) × (0.5 + draw)`. `03`/`discussion` L4
propose `draw × min(strength bound, shape bound)` with the strength bound `σ_y / (ρ · g)`.

**8a cannot build it, and here is why.** `BodyDefinition::from_seed(seed, look_radius_m)` holds the
seed and the radius and nothing else. The strength bound needs **surface gravity and bulk density**,
which the FOREST draws and the CHARTER carries (ruling V13 L12; SL6 ask A1, approved). Deriving a
density inside `from_seed` would be a second source for a number the world already has, which SL5 and
HR3 refuse.

**RECOMMENDATION: the relief law moves to 8b, with the charter.** 8a keeps `0.004·R` and says so.
This departs from `00_proposed_landforms.md` §6.2's 8a row, which moved D6 forward — it is **ASK 1**
in §7. The cost of waiting is small: the spectrum's fine amplitudes do not read `relief_m` at all
(they read `λ(o)`), and the CONSTRAINT `Σ a ≤ relief_m` is slack by a factor of nine on the home
planet, so the relief law moves only the COARSE octaves, which 8c replaces anyway.

### 1.6 The sea

**Today:** a uniform draw in `[−0.4, +0.3] × relief_m`. MEASURED (`02_planet_layout.md` §1.4, old home
planet): the field's own σ is 2 297 m, the draw landed at **−2.31 σ**, and **0.9 % of the surface is
under water**. The sea is a lottery.

**What 8b owes it:** the WATER INVENTORY, a charter integer (T2, T1 item 3). **8a has no inventory.**

**THE INTERIM, and its owner.** 8a solves the sea for a stated OCEAN FRACTION drawn from the seed in
a stated band, by the recipe `02` §7.1 gives, with the volume half replaced by an area half:

1. the sample directions are the LADDER's own cell centres at a coarse rung — **26 cells per face
   edge is 26² × 6 = 4 056 directions**, through the bend the crate already has. A fixed list, a
   fixed order, a fixed trip count, no rejection loop, no new branch;
2. each sample evaluates the **FULL field** — the octaves, the ridge, the roughness and the terrace —
   because §1.3's lift and §1.4's benches both move the hypsometry;
3. **bisect for the level at which the weighted share of samples under it equals the target
   fraction**, each sample weighted by its cell's own area factor, in a **fixed 24 steps**, never
   "until converged" (a float comparison is a drift class). Twenty-four halvings of a 40 km range
   reach 2.38 mm.

**The owner owns the band.** `discussion` L26 recommends **55 %–70 % for the median body**; the band
itself is **ASK 2** in §7. When 8b lands, the target fraction's DRAW is replaced by the charter's
inventory and the **bisection body does not change** — which is what makes this an interim and not a
second mechanism.

**What the interim does NOT buy.** A shelf, a beach and a bimodal hypsometry come from ISOSTASY
(`02` §1.4: "the single cheapest change with the largest visible effect"), which is 8c's. At 8a the
sea solves honestly against a one-hump field, so it gives a sea at the right LEVEL and a coast with
no shelf.

**The cost, and where it lands.** `02` §7.1 COMPUTED one full field evaluation at 520 ns, so 4 056 of
them cost **about 2.1 ms per body, once, inside `from_seed`**. MEASURED in the tree today:
`from_seed` runs **inline on the message path** (`crates/client/src/chunks.rs:1502`,
`crates/bins/src/bin/shard.rs:404`). A star system stating a planet and its moons in one frame would
pay that per body on the client's own thread, and `tick-hitch` is a named seam (SL8). **So `from_seed`
moves off the message path in the same stage as the sea, on both hosts.**

### 1.7 The coarsening by wavelength

**What exists today** is §1.0's finding: the count rule IS the wavelength rule at a constant 48.83
cells per wavelength for rungs 0..13, and the clamp arm breaks it at rungs 14..18 (down to 1.53
cells at rung 18).

**What changes.**

1. `octaves_at(rung)` stops counting and reads the wavelength: an octave survives while
   `λ(o) ≥ LADDER_SAMPLES_PER_WAVE · cell_m(rung)`. `00` §4.2 recommends **4**; `04` §4.3 recommends
   `C = 8` cells per wavelength for the TABLE'S FINEST END, which is a different number for a
   different question. 8a takes **4** for the survival rule, which is Nyquist with a factor of two of
   margin, and leaves the table's ends where they are.
2. The clamp arm (`body.rs:446-448`) stops returning one octave at rungs past the table. `04` §4.3
   says the right count at the top rung is ZERO — but ZERO is only right once `Z` carries the shape,
   and `Z` is 8c's. **8a's answer: the rule returns whatever survives, which at rungs 17 and 18 on
   the home planet is nothing, and 8a therefore keeps ONE octave there with an explicit comment
   naming 8c as the fix.** Keeping one aliased octave is a known, bounded, measured wrong; drawing a
   bare sphere at the top two rungs is a visible one. **ASK 3.**
3. **The table's ends do NOT move in 8a.** `04` §4.3 wants the table anchored to the ladder, which
   deletes `LONG_WAVE_CAP_M` and `SHORT_WAVE_M` — but the anchor's TOP end is "the first
   `C · cell_m(L)` at or below the macro node", and the macro node is 8c's.

   ★ **And the arithmetic says extending the bottom end buys nothing at 8a.** Under the spectrum, the
   two octaves 8a would add at `SHORT_WAVE_M = 8` carry `a(14) = 0.119 m` at 24.4 m and
   `a(15) = 0.047 m` at 12.2 m — **0.17 m together**, under two gap steps, invisible, and they cost
   about 14 % of the column pass. Going further, `SHORT_WAVE_M = 4` needs **17 octaves**, which
   overflows `OCTAVES_CAP = 16` and forces a cap change on the CPU and on the card. So 8a leaves
   `SHORT_WAVE_M` at 30 and the cap at 16, and the metre-to-49-metre hole (`02` §1.6) stays open
   until 8c anchors the table and 04 §4.7 answers the band with objects. **This is stated, not
   hidden.**

---

## 2. THE INTEGER FORMS

Every new term below is fenced integer arithmetic in the recipe's own types. The rules each must obey
are the ones the tree already enforces and MEASURED faults already found.

### 2.0 The kernel rules, restated from the code

| The rule | Where it is enforced, and the fault that found it |
|---|---|
| No `/`, no `%`, no float arithmetic on the recipe's path | `crates/recipe/src/lib.rs:33-35` (three `deny`s); `crates/recipe/clippy.toml` bans every float method |
| No negation, no `abs` on `Gi`; a magnitude is `Gi::unsigned_abs` | `crates/recipe/src/gi.rs:1-45`, with four `compile_fail` controls |
| A shift masks; add, subtract and multiply wrap | `crates/recipe/src/gi.rs:137-185` |
| **Two roundings, and a kernel states which it uses** | `gi.rs:14-28`: `mul_shr` TRUNCATES toward zero (use it where a signed product passes 63 bits); `>>` FLOORS toward −∞ |
| No derived `PartialOrd` — an `Ordering` is an 8-bit word the SPIR-V back end refuses | `gi.rs:55-78` (MEASURED 2026-09-13) |
| No const array indexed at run time — naga's Metal made a private copy PER USE | `crates/recipe/src/noise.rs:22-28` (MEASURED: 274–348 ms became 37 ms) |
| No iterator — rust-gpu refuses its pointer arithmetic; write an INDEX loop | `crates/recipe/src/height.rs:54`, `cell.rs:220-221` |
| **No loop whose value a BRANCH assigns** — rust-gpu spills it and reads it one step stale | `crates/recipe/src/root.rs:25-34` (`isqrt(1)` read 0 on the card) and `cell.rs:242-248` (the hollow read 0 on 87 of 200 probe points). The cure is an **ADDITIVE** accumulator, which leaves in a phi |
| The root is 32 steps WRITTEN OUT, no loop, no branch | `crates/recipe/src/root.rs:46-97` |
| Every buffer read is guarded, and a failed guard WRITES NOTHING | `crates/recipe-gpu/src/lib.rs:165-170` |

### 2.1 The spectrum's amplitudes — a per-body TABLE, computed ONCE in the DRAW

★ **THIS IS THE MOST IMPORTANT DESIGN POINT IN 8a. The spectrum is a DRAW, not a kernel.**

`Octave::amplitude` is already "gap steps at `AMP_BITS = 8` below the step" — 1/32 768 m at the metre
rung (`crates/recipe/src/height.rs:43-44`). `octave_term` already computes
`(o.amplitude * noise3(...)) >> AMP_BITS`. **So changing HOW the amplitude is drawn changes not one
line of any kernel, and the card reads the same 32-byte `repr(C)` row it reads today.** The spectrum
costs the GPU nothing.

The draw runs in `BodyDefinition::from_seed`, where `Gf` is lawful (`gf.rs:6-8`, caller 1):

```
   x(o)  = (o − O_PEAK) · LN_2                    o is an integer; LN_2 is a CONSTANT, not a call
   s(o)  = S_PEAK / (1 + x(o)·x(o) / (SIGMA·SIGMA))
   a(o)  = s(o) · λ(o) / TAU                      λ(o) is `waves[o]`, already a Gf in the loop
```

Each step is `Gf` add, subtract, multiply or divide. No `powf`, no `ln`, no `exp` — the fence holds.

**`S_PEAK` is SOLVED, not typed.** 8a computes it from the anchor so the 35° promise survives a table
whose ends move:

```
   let base(o)  = 1 / (1 + x(o)² / SIGMA²)                          the bump, at unit peak
   let norm     = sqrt( Σ over the fine octaves of base(o)² )       Gf::sqrt is lawful
   S_PEAK       = TALUS_RMS · TAN_REPOSE / norm
```

A pin asserts `S_PEAK` lands at 0.3998 on the home planet, so a table change that moves it is a red
test and not a surprise.

**The constraint is one `Gf::lesser`.** `scale = lesser(1, relief_m / Σ a(o))`, applied to every
amplitude. On the home planet `Σ a(fine) = 1 532 m` against `relief_m ≥ 6 000 m`, so the scale is one
and the branch is never taken in the world — which is why the test must drive it with a body that
does take it.

**THE ONE ROUNDING** stays where `body.rs:283` has it:
`Gi::new(rounded(amplitude_m · Gf::from_i64(STEPS_PER_M << AMP_BITS)))`.

**How the card reads it.** `PlanCharter` (`crates/recipe/src/plan.rs:42-62`) already carries
`octaves: [Octave; OCTAVES_CAP]` as `repr(C)`, and `crates/terrain/src/gpu.rs` already writes that
charter as one storage buffer. Sixteen octaves × 32 bytes = **512 bytes, unchanged**.

### 2.2 The ridged noise in integers

`1 − |n|`, re-centred so the amplitude keeps its meaning, at `NOISE_BITS = 28`:

```
   let n       = noise3(o.seed, p);                         in about [−1, 1] at 28 bits
   let mag     = Gi::new(n.unsigned_abs() as i64);          the fence's only magnitude
   let ridged  = ((NOISE_ONE - mag) << 1) - NOISE_ONE;      (1 − |n|)·2 − 1
```

This is **exactly** the transform `crates/bins/examples/skyline_march.rs:84` measured
(`(1.0 - n.abs()) * 2.0 - 1.0`), so 8a ships what M1b-2 predicted.

**It is EXACT.** There is no shift down, so no rounding is added — the error budget gains nothing
from the ridge.

**It stays inside `[−1, 1]`**, because `|n| ≤ 1` gives `ridged ∈ [−1, 1]`. Therefore
`relief_bound` and `dropped_bound` (Σ of the amplitudes) hold in FORM, unchanged.

**How a kernel chooses.** The selection must not be a branch that assigns a loop-carried word. Two
shapes, and 8a takes the second:

- *Refused:* `let v = if ridge { ridged } else { n };` inside the sum's loop. It assigns a LOCAL, not
  the accumulator, so it is probably safe — but `root.rs` and `cell.rs` both record a shape a CPU
  compiler folds away and a shader compiler mis-spells, and "probably" is not a measurement.
- **Taken:** an arithmetic select on a MASK. The octave carries a word `kind` that is 0 (smooth) or
  −1 (ridged); the term is `v = n + (kind & (ridged - n))`. One `and`, one add, one subtract, no
  branch, no shape for a compiler to get wrong, and the accumulator stays `h += ...`.

**Where the flag lives, and what it costs the bus.** `Octave` grows from four words to **eight** —
`seed, frequency_int, frequency_frac, amplitude, kind`, and three words of explicit padding so the
`repr(C)` stride is 64 bytes, a power of two the card's addressing likes. The kernel signature
`relief(&[Octave], dir)` does not change, so `crates/recipe-gpu/src/lib.rs`'s three entry points do
not change. The charter's octave block grows from 512 to 1 024 bytes. Nothing measurable.

★ **THE ONE BOUND THAT DOES NOT SURVIVE.** `column_bound` (`crates/terrain/src/digest.rs:122-146`)
bounds how far the surface can stand from the samples that measure it, with a SECOND-ORDER
(curvature) argument: `min(1, (π·edge·f/R)²) × amplitude`, summed. **A ridged octave has a KINK at
`n = 0` — that is what a crest IS — so its second derivative is unbounded there and the curvature
argument does not apply.** The first derivative doubles (the transform's slope is `∓2` where the
noise's is `±1`). The cure is to bound a ridged octave by its FIRST-order term instead:
`min(1, 2·π·edge·f/R) × amplitude`, which is larger and honest. Without it a surface chunk's column
span can be short and a chunk is missed — a hole, which is a defect.

### 2.3 The roughness factor in integers

★ **ONE multiply per COLUMN, not per octave.** `03` §4.2.4 writes `a(o, site) = a(o) · m(site)` for
every fine octave. Because `m` has no octave argument, that is algebraically
`Σ_coarse a·n + m · Σ_fine a·n`, so the kernel splits the sum at the first fine octave and multiplies
ONCE. Sixteen multiplies become one.

```
   let r_raw = (one_octave(&charter.roughness, dir) + NOISE_ONE) >> 1;      in [0, 1) at 28 bits
   let m     = M_MIN + ((NOISE_ONE - M_MIN) * fade(r_raw) >> NOISE_BITS);   in [M_MIN, 1]
   ...
   h = coarse_sum + fine_sum.mul_shr(m, NOISE_BITS);
```

**The bits, and why `mul_shr` and not `* >>`.** `fine_sum` is gap steps at `NOISE_BITS`. On the home
planet the fine amplitudes reach 1 532 m, which is `1 532 × 128 × 2²⁸ ≈ 5.3 × 10¹³`; times `m` at
`2²⁸` is about `1.4 × 10²²`, past `2⁶³`. **So the product MUST go through the two-word
`Gi::mul_shr`** (`gi.rs:116-127`), which truncates toward zero — stated here so a reader never
guesses. `(NOISE_ONE - M_MIN) * fade(...)` is safe as a plain `*` because both factors are under
`2²⁸`.

**`M_MIN` at 28 bits:** `round(0.06 · 2²⁸) = 16 106 127`. `m` therefore lives in
`[16 106 127, 268 435 456]`.

**The placeholder's octave** is one more `Octave` in the charter, drawn from its own salt
(`salt::ROUGHNESS`, a new constant beside the seven in `body.rs:59-67` — a new salt never moves an
existing draw). One noise call per column: **+7 % on a 14-octave column pass**.

### 2.4 The cap-rock bench in integers

Three per-body words, drawn once, and no divide anywhere on the path:

```
   spacing_steps   = S, whole gap steps                       a per-body draw
   spacing_recip   = recip_pow2(S, 62)                        crates/recipe/src/root.rs:109
   datum_steps     = the per-body datum radius, whole gap steps
   q               = the terrace strength at NOISE_BITS, a draw in [0.4, 0.7)
```

Per column, after the octave sum and before the carve (8d's carve does not exist yet, so at 8a the
terrace is simply last):

```
   let rel  = h - datum;                                       gap steps at LENGTH_BITS
   let k    = (rel >> LENGTH_BITS).mul_shr(spacing_recip, 62);  the bed index — `>>` FLOORS, both signs
   let lo   = datum + (k * spacing) << LENGTH_BITS;             the bed top below
   let hi   = lo + (spacing << LENGTH_BITS);                    the bed top above
   let d    = if (h - lo) < (hi - h) { h - lo } else { h - hi };  the nearer, signed
   let mag  = Gi::new(d.unsigned_abs() as i64);
   let u    = (mag >> LENGTH_BITS).mul_shr(spacing_recip << 1, 62);   |d| / (S/2), at 28 bits
   let f    = NOISE_ONE - smoothstep(u);                        1 − u²(3 − 2u)
   h        = h - q.mul_shr(d, NOISE_BITS).mul_shr(f, NOISE_BITS);
```

Every step is an add, a subtract, a shift, a compare or a two-word multiply. The `k` floor uses `>>`
deliberately, so a column BELOW the datum indexes the bed under it and not the one above.

**`smoothstep`.** `04` states the cubic `u²(3 − 2u)`. `vd_recipe::noise::fade` is the QUINTIC
`u³(u(6u − 15) + 10)`, whose derivative is zero at BOTH ends — strictly better for the half-spacing
argument, and already in the crate and already covered. **8a reuses `fade` and states the change**,
so no new polynomial enters the fence and the Lipschitz constant is re-derived for the quintic in the
stage's own test rather than copied from `04`.

**The rung fade** is one comparison of `spacing_steps` against `C · cell_m(rung) · STEPS_PER_M` and
one linear scale of `q` — both words the charter can carry, so the kernel neither divides nor loops.

### 2.5 What each new term must obey, as a checklist

| Term | Index loop? | Const array at run time? | Divide? | Loop value a branch assigns? | New float? |
|---|---|---|---|---|---|
| The amplitudes (the draw) | no loop in a kernel — runs in `from_seed` | no | `Gf` divide, lawful in the draw | no | no new one; `Gf` only |
| The ridge | inside the existing index loop | no | no | **no — an arithmetic mask** | none |
| The roughness | one call before the loop | no | no | no | none |
| The terrace | straight line, no loop | no | no — one per-body reciprocal | no | none |
| The sea bisection | a FIXED 24-step loop in the DRAW, not a kernel | no | `Gf` divide, lawful | no | none |

### 2.6 The band, and the one line that must change

`body.rs:368-373` derives `crust_m` and `above_m` from `relief_whole = floor(relief_m) + 1`. That is
correct today **only because `Σ a(o) = relief_m` exactly** (`body.rs:280-284` normalises it so).

Under the spectrum the two part company. COMPUTED by hand from the code's own constants: the coarse
four octaves keep `relief_m × (1 + k + k² + k³) / Σ₀¹³ kᵒ`, which for `k ∈ [0.45, 0.55)` is 0.96 to
0.99 of `relief_m`; the fine ten become 1 532 m. So `Σ|a|` **exceeds `relief_m` by roughly 1 300 to
1 500 m** on the home planet, and the ridge's one-sided lift (about 800 m, §1.3) stands inside that
same sum. A surface that leaves the grid is a chunk with no ground in it.

**THE CHANGE:** `relief_whole` reads the amplitude sum at `m = 1`, in whole metres — the number
`relief_bound(0)` already computes (`body.rs:451-461`). One line, and it is what `02` §4.6 states
("the band is sized on the sum of DRAWN amplitudes").

### 2.7 The bounds the terrace multiplies

`relief_bound`, `dropped_bound` and `column_bound` each gain a `Lip(T)` factor, a per-body word at
`NOISE_BITS` computed in the draw as `1 + 0.6875·q` for the cubic (re-derived for the quintic in
stage 4). `dropped_bound` is what the client's crossfade sizes its sink from
(`crates/client/src/ladder_view.rs:96-108`), so a bound left un-multiplied shows as a coarse mesh
through a finer one at a fade edge — the defect the sink ramp was built to cure.

### 2.8 THE ERROR BUDGET, against F2's half a metre

F2 sets the added-octave tolerance at **half a metre at the 1 m rung** (ten band cells × a slope of
1/20). The sub-millimetre budget 8a must stay inside:

| Source | Worst error | Where it is measured |
|---|---|---|
| The octave sum today, integer against the float recipe | **1.06 mm widest, 0.13 mm mean** over 4 million columns | `crates/recipe/src/noise.rs:10-12`, `height.rs:6-12` |
| The amplitude's ONE rounding, per octave | 1/32 768 m = **0.0305 mm**; 14 octaves all rounding the same way, **0.43 mm** | `height.rs:24-25` |
| The ridge transform | **0** — no shift down, no rounding | §2.2 |
| The roughness `mul_shr` | one unit at 28 bits of a gap step = **2.9 × 10⁻¹¹ m** | §2.3 |
| The terrace's four `mul_shr`s | four of the same, **1.2 × 10⁻¹⁰ m** | §2.4 |
| The terrace's AMPLIFICATION | × `Lip(T)` = **1.413 at `q = 0.6`** on everything above | §1.4 |

**Total, ESTIMATED: under 2.2 mm, against F2's 500 mm — a margin of about 220.**

★ **And F2 is not the gate here.** SL10's no-drift gate is BYTE-FOR-BYTE between the server's CPU,
the client's CPU and the card (`just gpu-drift`, `just terrain-pin`). F2's half-metre governs a
DIFFERENT question — how far a frozen patch's surface may move when an octave is added to a world
that is already built on. Nothing is built on the ground yet (T1), so F2 binds 8a only as a
statement of how much room the arc has: **220 times what 8a spends.**

---

## 3. WHAT MOVES AND WHAT DOES NOT

### 3.1 The ladder's `N` and the rung count do NOT move

`Ladder::for_radius` (`crates/seed/src/ladder.rs:82-125`) computes `n_ideal = radius_m · π/2`, snaps
it with `top_rung_for(n_ideal)`, and sets `rungs = top + 1`. **None of that reads `crust_m` or
`above_m`.** So the home planet keeps `N = 9 961 472`, 19 rungs, and a ladder radius of 6 341 670 m
through the whole of 8a. Every chunk's `(face, rung, x, y)` is the chunk it is today.

### 3.2 What DOES move

| The thing | Moves? | Why |
|---|---|---|
| `Ladder::n`, `Ladder::rungs`, the radius | **no** | §3.1 |
| `Ladder::floor_m`, `Ladder::band_m` | **YES**, by about 1 300–1 500 m | §2.6: the band is re-derived from `Σ|a|` |
| A chunk's radial index `z` | **YES** | the floor moved, so the same radius is a different `k` |
| A chunk's `(face, rung, x, y)` | **no** | §3.1 |
| Every chunk's bytes | **YES** | the amplitudes, the ridge, the roughness and the terrace all move the surface |
| `crates/terrain/tests/golden_home.txt` (1 458 × 3 rows) | re-recorded **at every stage that moves a byte** | `crates/terrain/tests/terrain_pin.rs:29-41` |
| `crates/terrain/tests/golden_home_mesh.txt` | the same | `mesh_pin.rs` |
| `tag.rs::GENERATOR_VERSION` | **3 → 4**, ONCE, at the end of 8a | `crates/terrain/src/tag.rs:25` |
| `tag.rs` `DECLARED_PIN` (10 363 069 377 454 183 796) | **YES**, once, with the version | `tag.rs:99` |
| `WorldIdentity::measured` (`golden_self_check`) | **YES**, but it is stored nowhere — every process re-folds it | `tag.rs:46-56`, `digest.rs:351` |
| `crates/recipe` `NOISE_PIN` / `VALUE_PIN` | **no** | the noise kernel itself does not change |
| `home.rs`'s `octave_count == 14` assert | **no** | §1.7: `SHORT_WAVE_M` stays at 30 |
| The five frozen pictures + the far candidate | **YES** — re-frozen once, on the owner's look | `crates/bins/tests/terrain_pictures.rs:183-190` |
| `crates/bins/tests/home_body_pin.rs` | check; the seed and the look radius do not move, the ladder's band does | |

### 3.3 `GENERATOR_VERSION`: one bump, at the end

`tag.rs:1-14` states the rule: bump by hand on ANY edit that moves one byte of one chunk; bumping
opens a world epoch, every store labelled with the old tag refuses to open, every saved edit is
rebuilt. **Nothing is stored on the ground yet** (T1: "the ground may move freely — nothing is built
on it"), so a bump costs nothing except the pins.

Bumping once per stage would be seven epochs in a fortnight and seven re-recordings of
`DECLARED_PIN`. **RECOMMENDATION: one bump, `3 → 4`, in the last stage, with the pictures.** The
golden pin tables are re-recorded at each stage that moves a byte, which is what keeps every stage
independently green.

### 3.4 F1 and the frozen patch, reconciled

**F1 says** a built area keeps the octave COUNT it was built on — a small integer stored with the
realm, shipped as a block-store row — and that the list is APPEND-ONLY.

**The apparent conflict.** 8a does not append to the octave list. It REWRITES the amplitudes of
octaves 4..13, turns five of them ridged, and multiplies the fine half by a per-column factor. A
stored count of "14" would name a different surface before and after.

**The reconciliation, and it is already ruled, not asked.** T1 sets the order: the arc lands, then 9,
10, 11 (the block store, the diff lane, the collider), **then 14 THE FREEZE**. F1's append-only
promise is a promise about what happens AFTER the freeze, because the freeze is the moment the world
identity is pinned with the feature anchors in it. T1 states the window in the owner's own terms:
"during that time the ground may move freely — nothing is built on it — which is the freedom the
freeze takes away at 14". **So 8a is a world VERSION change inside the free window: `GENERATOR_VERSION`
3 → 4, a new epoch, and F1's list begins at 14 from whatever table 8f leaves behind.**

**What 8a OWES step 14, written down here so nobody re-derives it.** A bare octave COUNT identifies a
surface only while the table is append-only. After 8a the table also carries a per-octave `kind`
word, a per-column factor and a terrace. **RECOMMENDATION for 14: the stored integer becomes a PAIR —
the recipe's version and the octave count — so a patch names its surface completely.** That is a
decision for 14, listed as **ASK 4** so the owner sees it once now rather than as a surprise then.

---

## 4. THE MEASUREMENTS FIRST

T1 item 2 is judged by pictures, but T1's L25 paragraph is unambiguous: **"8a measures its densest
chunk FIRST."** None of §6's stages starts before §4 is in hand.

### M-A. THE SKYLINE MARCH, RE-RUN ON THE NEW BODY AND ON 8a's EXACT CONSTANTS

**The instrument:** `crates/bins/examples/skyline_march.rs`, already written, already carrying the
spectrum (`:45-47`) and the ridged band (`:52-54`). Both marches are **the same 1.5 s program**
(`09` M1b-2). `cargo run --release -p vd-bins --example skyline_march`.

**Why it re-runs.** `09`'s M1b and M1b-2 were measured on 2026-09-09, on the OLD home planet
(3 351 km, seed 7 701 581 858 760 374 086). The tree's home planet is now
`Planet(4030111653607004909)` at 6 341 670 m (`crates/terrain/src/home.rs:22-26`). The amplitude
table is unchanged (§1.1: both bodies clamp to a 400 km long wave), but the STATIONS, the relief
draw, the `k_rough` draw and therefore every break count are not.

**What it must print, and what the owner sees:** the amplitude table today and proposed; the fine
sum and the fine RMS slope against `tan 35° = 0.700`; the largest rise and the breaks per 60° at the
four thresholds, for all THREE fields (today, spectrum, spectrum + ridged), at the four stations.

**The stop line, from `discussion` §8:** if the largest rise stays under **0.10°**, the spectrum is
not the cure and the arc stops. It passed on the old body at 0.138°–0.168°.

**Two changes the instrument needs before it runs:** the ridged band is stated by INDEX (`5..=9`) and
must be stated in METRES (§1.3); and the roughness factor is at its maximum everywhere, which is the
prediction's stated assumption and must be printed as such.

### M-B. THE DENSEST CHUNK, AGAINST L25's EIGHT MILLISECONDS

**Which instrument answers which question — they are not the same:**

- `crates/bins/examples/terrain_cost.rs` measures **the generator's own cost**: the sample box plus
  the extraction, per rung, against `EXTRACT_BUDGET_US = 8_000` (`:38-40`). **This is L25's budget.**
- `crates/bins/examples/chunk_phases.rs` measures **the whole build**, the parent meshes the geomorph
  reads included, warm and cold. **This is where T1's 111–112 ms comes from.**

**Run both.** `terrain_cost` says whether the recipe fits 8 ms; `chunk_phases` says what the pilot
actually waits for.

★ **Where the cost really is, and it is not the arithmetic.** COMPUTED: a sample box is `BOX_EDGE = 64`
cells a side (`crates/terrain/src/lattice.rs:47-51`), so 4 096 columns × 14 octaves = 57 344 octave
sums. 8a adds one roughness noise (+7 %), the ridge on five octaves (a mask, an add and a subtract
each), one column multiply and one terrace (about one octave's work). **ESTIMATED: the column pass
grows 20–25 %, which is a fifth of a millisecond of a box.** The risk is the EXTRACTOR: rougher
ground makes more surface, and `04` §5.4 ESTIMATES the vertex count rises **1.4 to 2.9 times**. A
rung-0 chunk MEASURED at 405 228 bytes and 8 577 vertices (`04` §5.4) becomes 12 000 to 25 000.

**So M-B measures, on the walk's path AND on the six picture stands:** milliseconds of box, of
extract and of the whole build, per rung; vertices and triangles per chunk; and the WORST chunk of
each set. The owner then sets the budget (T1: "the budget itself stays the owner's call when 8a's
numbers are in").

### M-C. THE SLOPE HISTOGRAM (proxy P2)

The proxy has no instrument in the tree (`grep` finds no slope histogram in `crates/bins/examples`).
**8a writes one**, as a new example beside the others, reading `vd_terrain::height::height_m` on a
stated grid at stated baselines.

**The band, from `discussion` §8 (M9) and `04` §4.8:** median slope **3°–12°**, p95 slope
**25°–40°**, and — the half that proves the roughness factor works — **the plain's p99 under 5°**
while the range's p50 sits at 28°–34°. Today the whole planet measures **1.6°–2.0° at every baseline
from 2 m to 8 km** (`02` §1.2), so a single-humped histogram after 8a means the per-column factor is
not modulating anything.

### M-D. THE RUNG DISAGREEMENT

**The instrument:** `crates/bins/examples/rung_disagreement.rs`, already written. The pass line is
slice 8's own: **p99 ≤ 1 pixel, max ≤ 2 pixels** (`:9-11`).

**Today, MEASURED (`09` M8-0):** worst over the whole ladder **0.37 px max, 0.24 px p99** — a margin
of five. **After 8a, `04` §5.2 COMPUTES 1.96 cells at an alpine slope**, which is where the crossfade
band earns its width. The instrument re-runs after every stage that touches the height field, and its
number is the one that says whether the ladder still holds.

★ **And it must be re-read against the corrected claim.** `discussion` §7.2 says the ladder's promise
is broken today at six of twelve rungs (32.8 m against 32 m at rung 6). `09` M8-0 lines 191–197
RETRACTS that: "that compared the BOUND with HALF a cell, which is a stricter promise than the ladder
makes, and it was not a measurement."

---

## 5. THE GATES

### 5.1 The picture protocol

**Today, from `crates/bins/tests/terrain_pictures.rs:1548-1670`:** six stands — **ground** (1.8 m),
**hill** (300 m), **aloft** (60 km), **orbit** (2 000 km), **seam** (on a cube edge, found by a
search over all twelve edges) and **far** (the globe a quarter of the frame high, about 10.2 body
radii). Five are frozen against `docs/investigation/2026-09-07/pictures/exact/`; the far stand is a
CANDIDATE (`:114-118`) which the gate compares and REPORTS but never turns red on, because the owner
has not looked at it.

**Every picture carries** the STAMP (what the renderer measured), the PROBE (a second buffer in which
every pixel says what drew it, at which rung, how far), the RULER (a ball of known size) and ONE
LIGHT at **15° elevation, 120° off the nose** (`:76-85`).

**8a adds a SEVENTH stand: THE CAVE** (T1, "the cave in the crossfade"). The eye stands at a cave
mouth in a cliff, found — never typed — by a search over ladder addresses for a column where the
cavern field opens a room within a stated depth of the surface and the column's neighbour stands a
stated height below it. It joins as a CANDIDATE, like the far stand, and freezes on the owner's look.

**The candidate pattern, restated:** a new stand writes to `candidate/`, the gate prints its verdict
against the last candidate, and `VD_PICTURE_FREEZE=<name>` moves it to `exact/` when the owner
accepts it (`:180-193`).

**The owner looks ONCE, after 8a** (T1). Not per stage. Each stage refreezes the references so the
next stage's diff is readable; the owner's judgement is taken on the finished seven.

### 5.2 The skyline gate P1, and the band it waits for

P1 counts breaks per 60° from the four named stations (`01_reference_target.md` §6.1). **Its band is
not honest yet**, and `01` §2.2 says why in the sharpest finding of the round: the reference's
"8–12 breaks per 60°" was **counted by eye off a screenshot**, and our "0" was **computed by a stated
rule over a height field**. Those are two different quantities, and the 8–12 carries no measurement
mark at all.

**So P1 lands as a RATCHET** — neither the break count nor the largest rise may fall below the last
recorded value at any of the four stations — **and becomes a FLOOR when U16 lands.**

★ **U16 is being traced now, by another agent, and it will write
`docs/investigation/2026-09-08/landforms/11_reference_trace.md`.** That file traces the reference
picture's skyline by hand as elevation against bearing, at its own camera height and field of view,
and runs OUR break rule over it. 8a's P1 band is calibrated from that file and from nowhere else.

⚠ **Two discrepancies 8a must settle when it sets the band.** `01` §6.1 defines a BREAK at **0.5°**
over the ±5° median and marches to **360 km**; `09`'s M1b tabulates four thresholds and treats
**0.10°** as the stop line, marching to **540 km**. Under the strict 0.5° rule only one of the four
stations scores anything at all even with the ridged band. **The threshold and the march limit must
be one pair, stated once, and the same pair must be used on the reference trace.** That is **ASK 5**.

### 5.3 The no-drift gate on the card

SL10 is a MEASUREMENT, not an argument. Three gates already exist and every stage runs them:

- **`just gpu-drift`** — `crates/bins/tests/gpu_no_drift.rs`: the card's box against the CPU's on
  **every box a real stand draws** (about 6 049 on the seam stand, ~30 s), not on eight golden ones.
  It exists in that shape because a pooled buffer that grew and never shrank drifted **1 022 of
  6 049 boxes** and was invisible to a fresh-buffer check.
- **`just gpu-seam`** — the card as a second builder beside the renderer, frames a second before and
  after.
- **`just terrain-pictures-card`** (`VD_TERRAIN_GPU=1`) — the same stands with the card building.
  Green as of 2026-09-14 at 9 / 8 / 7 / 1 / 13 content pixels against the frozen references, at a
  widest channel step of ONE.
- **The boot self-check** — `WorldIdentity::of` folds the DECLARED tag with the MEASURED half
  (`golden_self_check`, eight golden chunks) and the gateway refuses a client whose pair differs
  (`crates/terrain/src/tag.rs:36-56`).

★ **Stage 2 is the only stage that touches a kernel**, so `gpu-drift` is that stage's headline gate.
Every other stage changes the DRAW, which runs on the CPU once per body; the card reads the charter
it produces.

**The shader must be rebuilt for stage 2:** `just recipe-gpu` (cargo-gpu, rust-gpu's pinned nightly,
`Int64` declared).

### 5.4 The pins

**`just terrain-pin`** runs `terrain_pin` and `mesh_pin` three ways — debug, release, and
`RUSTFLAGS="-C target-cpu=native"` release — over 1 458 chunk columns × 3 rows. Re-record it at every
stage that moves a byte, and **state in the commit which stage moved it and why.**

**`just terrain-link-scan`** (the generator's object code names no platform math symbol) and
**`just terrain-fence-control`** (the observed-failing control: three disallowed methods must go red)
do not change and must stay green — they are how a float that sneaks in through a dependency is
caught.

### 5.5 Coverage

**`just coverage-fast`** — Tier-A at 100 % region and branch, over
`-p vd-recipe -p vd-seed -p vd-terrain -p vd-core -p vd-physics -p vd-devproto -p vd-wire -p vd-sim -p vd-node -p vd-connection-plane -p vd-harness -p vd-client -p vd-client-harness`
(`justfile:12`). `vd-recipe` and `vd-terrain` are both in it, so **every new branch 8a writes needs a
test that drives both arms**, and HR5's generic-code gotcha applies: a branch inside a generic
function must be exercised for every instantiated type and in every test binary.

**Three branches 8a adds that the world itself never drives**, and which therefore need a synthetic
body in a unit test:
1. the constraint `Σ a ≤ relief_m` taking its `lesser` arm (slack by nine on the home planet);
2. the terrace's rung fade at the rung where `q` reaches zero;
3. the wavelength rule returning nothing at a rung past the table.

`coverage-exemptions.toml` holds `(none yet)` and must still hold it when 8a lands.

### 5.6 The moving eye: the band and the pop

**`just terrain-moving-eye`** flies three legs — a walk at 1.4 m/s, a hull at 240 m/s, a hull at
528 m/s — and reads the residency band's gap every frame, with the POP DETECTOR taking a pair of
frames every 2 s at 60 Hz (`crates/bins/tests/terrain_moving_eye.rs:87-105`). Rougher ground means
more vertices per chunk and a slower build, so the band is the first thing 8a can break.

**`just flights`** = `terrain-pictures` + `terrain-moving-eye` + `boarding-storm`, about
twenty-five minutes, on a quiet machine with Docker off. The standing rule: **no slice lands before
`just flights` is green.**

### 5.7 The card's SECOND stand-down trigger

T1: "the card's stand-down gets a second trigger — a chunk over budget — judged on the same flights."

**Today (`crates/client/src/card_gate.rs:97-101`)** the card takes a request only when the queue is
DEEP: `pending > CARD_DEEP_MARGIN · workers_finish()`, with `CARD_DEEP_MARGIN = 2.0`, and a still eye
never sees the card at all.

**The second trigger, as 8a would write it:** the rule also stands down when the CPU workers' own
measured per-chunk time at the queue's finest rung exceeds the budget — because a chunk that takes
three times the budget on a worker takes longer still through the card's round trip and its single
geometry thread, and the card's backlog then lands every chunk behind it later. The MEASURED
precedent is exactly that shape: the hill stand's card took 960 chunks, ten seconds of its own
geometry stage, and moved the settle from tick 2 081 to 2 509.

★ **It is built ONLY if M-B says so.** If 8a's densest chunk stays under the budget, there is no
trigger to add and the rule stays as it is. **The measurement decides; this design does not.**

---

## 6. THE STAGES

Seven stages. Each is under about a day of agent work, each is independently green, each re-records
the pins it moves, and each names what the owner would see if asked.

**Why this order.** (1) The amplitudes first, because they change only the DRAW and touch no kernel,
so the GPU gates cannot go red and the biggest single picture change is judged alone. (2) The ridge
next, because it is the ONE kernel change in the whole slice, and `gpu-drift` must run on it with
nothing else moving. (3) The roughness third, because it multiplies the two before it and is
meaningless until they exist. (4) The bench fourth, because it reads the height the three before it
produce and multiplies every bound. (5) The wavelength rule fifth, because the table must be final
before the survival rule is rewritten. (6) The sea sixth, because it evaluates the WHOLE field and
must see the ridge's lift and the benches. (7) The version, the pins and the pictures last, once.

---

### Stage 1 — THE SPECTRUM IN THE DRAW

**Files.** `crates/terrain/src/body.rs` (the fine amplitudes from `s(o)·λ(o)/TAU`; `S_PEAK` solved
from the anchor; the constraint as one `lesser`; `FINE_ABOVE_M`; **the band from `Σ|a|`, §2.6**);
`crates/terrain/tests/golden_home.txt`; `crates/terrain/tests/golden_home_mesh.txt`.

**Failing-first test** (`body.rs`, unit): `the_fine_octaves_carry_the_slope_spectrum_and_the_band_holds_their_sum`
— each fine `a(o)` within one gap step of `s(o)·λ(o)/TAU`; `sqrt(Σ_fine s²)` equal to
`TALUS_RMS · TAN_REPOSE` within a stated tolerance; `S_PEAK` at its pin; `Σ a(o) ≤ relief_m`; and
`ladder.floor_m + crust ≥ radius + Σ|a|`. RED today, because the amplitudes are `k_rough^o` and the
band reads `relief_m`.

**Gate.** `cargo test -p vd-terrain -p vd-recipe`; `just terrain-pin` (re-recorded);
`just coverage-fast`; M-A and M-D re-run.

**The owner sees.** The skyline march's new amplitude table beside today's, and the break count.

---

### Stage 2 — THE RIDGED BAND

**Files.** `crates/recipe/src/height.rs` (`Octave` grows a `kind` word and pads to eight words;
`octave_term` gains the arithmetic mask of §2.2); `crates/terrain/src/body.rs` (mark the band by
WAVELENGTH in metres, `RIDGE_LO_M`/`RIDGE_HI_M`); `crates/terrain/src/digest.rs` (`column_bound`
bounds a ridged octave by its FIRST-order term, §2.2); `crates/bins/examples/skyline_march.rs` (state
the band in metres, so the instrument and the world agree); the two golden tables.

**Failing-first tests.** In `vd-recipe`: `a_ridged_octave_is_one_minus_the_magnitude_recentred` —
the term equals `amplitude × ((1 − |n|)·2 − 1) >> AMP_BITS` on named lattice points, and stays inside
`[−amplitude, +amplitude]` over a scan. And `the_sum_is_additive_with_a_ridged_octave_in_it` — the
whole sum equals the sum of the terms taken one at a time, which is the accumulator shape the card
needs. In `vd-terrain`: `a_ridged_octave_widens_the_column_bound`.

**Gate.** `just recipe-gpu` (the shader rebuilds); **`just gpu-drift`** (the headline);
`just gpu-seam`; `just terrain-pin`; `just coverage-fast`; M-A and M-D.

**The owner sees.** M1b-2's numbers, re-run on the shipped field instead of on a prediction.

---

### Stage 3 — THE ROUGHNESS FACTOR, ON THE PLACEHOLDER

**Files.** `crates/terrain/src/body.rs` (a roughness octave from a new `salt::ROUGHNESS`);
`crates/recipe/src/height.rs` (`relief` splits at the first fine octave and applies ONE `mul_shr`);
`crates/recipe/src/plan.rs` (`PlanCharter` gains `roughness: Octave`, `m_min: Gi`,
`first_fine: Gi`); `crates/terrain/src/digest.rs` (the bounds read `m = 1`); the golden tables.

**Failing-first tests.** `the_roughness_factor_never_leaves_its_band` (a scan asserting
`M_MIN ≤ m ≤ 1`, both ends reached); `a_smooth_column_and_a_rough_column_differ_by_the_factor_alone`
(two columns whose coarse sums are equal and whose fine sums differ by exactly the ratio of their
factors); `the_band_is_sized_at_the_factors_ceiling` (the ladder's crust covers `Σ|a|` at `m = 1`,
which is what lets 8c swap the field).

**Gate.** `just gpu-drift`; `just terrain-pin`; `just coverage-fast`; **M-C, the slope histogram** —
this is the stage the histogram exists for.

**The owner sees.** A slope histogram with two humps: a plain under 5° at p99 and a range at 28°–34°
p50, against today's single 1.6°–2.0° spike.

---

### Stage 4 — THE CAP-ROCK BENCH

**Files.** `crates/recipe/src/height.rs` or a new `crates/recipe/src/terrace.rs` (the kernel of
§2.4); `crates/terrain/src/body.rs` (the datum, the spacing, `q`, the reciprocal);
`crates/recipe/src/plan.rs` (the charter words, and the rung fade's own word);
`crates/terrain/src/digest.rs` (`Lip(T)` into `relief_bound`, `dropped_bound` and `column_bound`);
the golden tables.

**Failing-first tests.** `the_terrace_fades_to_nothing_at_the_midpoint_from_both_sides` (the pull and
its first difference both zero at `u = 1`, approached from each bed); `the_terrace_lipschitz_constant_is_the_one_the_bounds_use`
(a dense scan of `d(terrace)/dh` against the per-body word — RED if the quintic's constant is taken
from `04`'s cubic); `the_terrace_fades_out_over_its_last_rung` (`q` reaches zero at the stated rung,
and the two neighbouring rungs differ by the fade's step, not by the whole bench).

**Gate.** `just gpu-drift`; `just terrain-pin`; `just coverage-fast`; **M-D** (the bench is the term
most likely to break the rung pass line); `just terrain-moving-eye`.

**The owner sees.** A bench in the hill stand — a flat step with a vertical face along a hillside at
one height, which is the thing the reference picture has and this world has never had.

---

### Stage 5 — THE LADDER'S WAVELENGTH RULE

**Files.** `crates/terrain/src/body.rs` (`octaves_at` reads the wavelength, not the count; the clamp
arm at the top rungs carries its comment naming 8c); the golden tables if any byte moves.

**Failing-first test.** `every_rung_samples_its_finest_live_octave_at_four_cells_per_wavelength_or_more`
— **RED TODAY at rungs 17 and 18 of the home planet** (3.05 and 1.53 cells per wavelength, §1.0), so
the test is a real refusal and not a restatement. Plus
`the_wavelength_rule_and_the_count_rule_agree_below_the_clamp` — for rungs 0..13 the two rules select
the same octaves, which is what makes this stage move no byte there.

**Gate.** `just terrain-pin`; `just coverage-fast`; **M-D**; `just terrain-pictures` (the far and
orbit stands draw the top rungs).

**The owner sees.** The far and orbit stands, where the top rungs live.

---

### Stage 6 — THE SEA, AND `from_seed` OFF THE MESSAGE PATH

**Files.** `crates/terrain/src/body.rs` (the 4 056-sample, fixed-24-step bisection of §1.6);
`crates/client/src/chunks.rs` (`state_surface` hands the seed and the radius to a worker and the
realm has no body until it is ready); `crates/bins/src/bin/shard.rs:404` (the same); the golden
tables.

**Failing-first tests.** `the_sea_solves_to_the_targets_ocean_fraction` (the measured share of the
4 056 weighted samples under the level, within the sampling error of 0.78 %);
`the_bisection_takes_a_fixed_number_of_steps` (an instrumented count equal to 24, never a convergence
test); `the_sample_directions_are_the_ladder_s_own_cells_in_address_order` (no rejection loop, no
sort, no tie); and in `vd-client`,
`stating_a_surface_does_not_build_the_body_on_the_message_path`.

**Gate.** `just terrain-pin`; `just coverage-fast`; `just flights` (the tick-hitch half is what the
moving eye and the pictures measure); M-B re-run, because `from_seed` got 2.1 ms heavier.

**The owner sees.** The orbit and far stands with a sea at the right level instead of the puddle
today's uniform draw produced (0.9 % of the surface under water, MEASURED).

---

### Stage 7 — THE VERSION, THE PINS, THE PICTURES AND THE CAVE STAND

**Files.** `crates/terrain/src/tag.rs` (`GENERATOR_VERSION` 3 → 4, the doc comment saying what moved,
`DECLARED_PIN`); `crates/bins/tests/terrain_pictures.rs` (the CAVE stand, found by a search, joining
as a candidate); the six picture references + the two candidates.

**Failing-first test.** `the_declared_tag_folds_the_version_and_the_seed_and_nothing_else` already
asserts `GENERATOR_VERSION == 3` and the `DECLARED_PIN` (`tag.rs:70-84`), so it goes red on the bump
and green on the new pin — an observed failure, by construction. Plus
`the_cave_stand_is_found_and_holds_a_cave_mouth_in_frame`.

**Gate.** `just gate` (the whole pre-merge gate); `just flights`; `just terrain-pictures-card`;
`just terrain-legs` if Docker is up.

**The owner sees.** ALL SEVEN STANDS, ONCE — the five frozen ones moved, the far candidate, and the
new cave candidate — under the raking light, with the stamps beside them. This is T1's "the owner
looks ONCE after 8a; the far stand's candidate is judged then too."

---

## 7. THE ASKS

Each is a question with a recommendation. **None of them is decided here.**

**ASK 1 — Does the RELIEF LAW stay in 8a, or move to 8b with the charter?**
`00_proposed_landforms.md` §6.2's 8a row moved D6 forward into 8a. But the law
`draw × min(σ_y/(ρ·g), 0.077·R)` needs surface gravity and bulk density, and `from_seed` holds only a
seed and a radius; the charter that carries them is 8b's (V13 L12, SL6 ask A1, approved). Deriving a
density inside the generator would be a second source for a number the forest already draws.
**RECOMMENDATION: move the relief law to 8b.** The cost is small — the spectrum's fine amplitudes do
not read `relief_m` at all, and 8c replaces the coarse octaves the relief law scales.

**ASK 2 — What is the SEA's interim target, and its band?**
8a cannot use a water inventory (8b's). It CAN solve for a stated ocean FRACTION by the same
bisection, so the mechanism does not change when the inventory arrives. `discussion` L26 recommends
**55 %–70 % for the median body**. **RECOMMENDATION: a seed draw in a stated band, with the band the
owner's number**, and the bisection's body unchanged at 8b. The alternative — keep today's uniform
draw for one more slice — leaves every 8a picture with a puddle on it, which makes the coast
unjudgeable.

**ASK 3 — At a rung past the octave table, does the ladder keep ONE aliased octave, or none?**
`04` §4.3 says the right count is ZERO, and it is right — once `Z` carries the shape at every rung.
`Z` is 8c's. On the home planet rungs 17 and 18 today sample the 400 km octave at 3.05 and 1.53 cells
per wavelength, which aliases. Dropping to zero draws a bare sphere at the far and orbit stands.
**RECOMMENDATION: keep ONE, with the ratio printed by M-D and a comment naming 8c as the fix.** A
measured, bounded wrong is better than a visible one.

**ASK 4 — At the FREEZE (step 14), does F1's stored integer become a PAIR?**
F1 stores a built area's octave COUNT and promises the list is append-only. After 8a the table also
carries a per-octave `kind`, a per-column factor and a terrace, so a bare count no longer names a
surface. **RECOMMENDATION: at 14, store (the recipe's version, the octave count).** It is one extra
small integer on a block-store row, and it makes the frozen patch's promise complete. Nothing is
built yet, so this changes nothing today — it is written down so it is not re-derived at 14.

**ASK 5 — What is P1's break THRESHOLD and its march LIMIT?**
`01_reference_target.md` §6.1 defines a break at **0.5°** over the ±5° median and marches to
**360 km**. `09`'s M1b uses four thresholds and treats **0.10°** as the stop line, marching to
**540 km**. Under the strict 0.5° rule only one of the four stations scores anything at all even with
the ridged band. **RECOMMENDATION: state ONE pair, use it on the reference trace (`11_reference_trace.md`)
and on our own field, and publish all four thresholds as information.** A gate whose rule differs
from the reference's rule compares nothing — `01` §2.2's own finding.

**★ No SL6 ask is needed by 8a, and no new wire arm.** Every term 8a adds is a function of the seed
and the address, computed on both hosts from the same crate. The macro field's gradient, the D8 flow
direction and the charter's integers — the three things that WOULD cross a boundary — are all
deliberately replaced by placeholders (§1.2, §1.3) or deferred (§1.5, §1.6). **That is the main
reason 8a can be built now.**

---

## 8. UNMEASURED

Everything below is an argument, an estimate or a hand calculation. None of it is a measurement, and
none of it may be quoted as one.

1. **The current home planet's `relief_m` and `k_rough`.** Both are seed draws
   (`body.rs:242-250`). `03`'s table (relief 14 304.887 m, `k_rough = 0.468 338 1`) was computed on
   the OLD 3 351 km home planet. On the current body `relief_share = 6 341 670 × 0.004 = 25 366.7 m`
   clamps to 12 000, so `relief_m ∈ [6 000, 18 000)` — **the actual draw is UNMEASURED.** M-A prints
   it.
2. **That `long_wave_m` is exactly 400 km and `λ(o) = 400 000/2^o` on the current body.** Derived by
   hand from `body.rs:248-250` and the clamp; consistent with `home.rs:54`'s `octave_count == 14`,
   but not run.
3. **The constant 48.83 cells per wavelength at every rung, and 3.05 / 1.53 at rungs 17 / 18.**
   Arithmetic on `body.rs:441-449` and `ladder.rs:48-51`. Stage 5's failing-first test is what turns
   it into a measurement.
4. **That `Σ|a|` exceeds `relief_m` by 1 300–1 500 m under the spectrum.** Hand arithmetic on
   `k ∈ [0.45, 0.55)`; it moves with the actual `k_rough` draw.
5. **That the two octaves a `SHORT_WAVE_M = 8` would add carry 0.17 m together.** Hand evaluation of
   `s(o)·λ(o)/TAU` at `o = 14, 15`; the same formula reproduced `03`'s published table to the decimal
   at `o = 4..9`, which is why it is offered — but it is not run.
6. **The column pass grows 20–25 %.** An operation count, not a bench. M-B measures it.
7. **The extractor's vertex count rises 1.4–2.9×.** `04` §5.4's ESTIMATE, on the old body, before the
   ladder was extended. This is the largest open risk in 8a and M-B is the only thing that answers it.
8. **The error budget's 2.2 mm total.** A sum of published per-stage errors times the terrace's
   Lipschitz constant. The per-stage numbers are measured; the SUM is not, and worst-case alignment
   of every rounding is assumed.
9. **`Lip(T) = 1 + 0.6875·q` for the QUINTIC falloff.** `04` §4.5 derives it for the CUBIC
   `1 − u²(3 − 2u)`. §2.4 recommends reusing `vd_recipe::noise::fade`, which is the quintic; its
   constant is re-derived in stage 4's own test and is expected to be smaller, not larger.
10. **The ridge's ~800 m lift on the current body.** MEASURED on the OLD body (`09` M1b-2) with the
    ridged band at indices 5..=9. The band in metres is the same; the body is not.
11. **That the placeholder roughness noise gives a two-humped slope histogram.** It is a noise, not a
    macro gradient, so its contrast is smooth and continental rather than following a range. M-C is
    the only thing that says whether it is enough to judge 8a's pictures by, or whether the owner is
    looking at a plain that is merely less rough.
12. **That 2.1 ms of sea bisection per body is the right price.** `02` §7.1's COMPUTED figure, on the
    old body's octave count, before the ridge and the terrace were added to the evaluated field. The
    real figure is higher and M-B measures it.
13. **The seven stages each fitting under a day.** A judgement from the size of each diff, not a
    record of anything.
14. **Whether the card needs a second stand-down trigger at all.** §5.7 states the rule's shape; L25's
    measurement decides whether it is built, and that measurement has not run.
