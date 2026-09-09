# Refutation B of `02_planet_layout.md` — revision 2

**Date:** 2026-09-08. **Lens:** believability, cost and the owner.
**Target:** `docs/investigation/2026-09-08/landforms/02_planet_layout.md` (revision 2).
**Method:** I read the report in full. I then re-opened every file it cites and re-ran every
arithmetic step I could check. Each finding names the section, the file line and the arithmetic.
Numbers I computed are marked COMPUTED and their inputs are stated, so the author can reproduce them
or refute them.

Revision 2 is a large improvement. The march is gone, the passive margin is in, the band is priced,
the spectral hole is named. This refutation is therefore about what revision 2 ADDED, not about what
revision 1 got wrong. The findings are numbered `B2-…` so they never read as revision 1's `B-…`.

---

## Blockers

### B2-1 — The river network is NOT a lookup. Its harvest is the whole-planet pass §3.3 refuses.

§4.4b says the tree is *"harvested per hydrology cell … `branches(body, cell)` is a pure function of
the cell's address"*, and §9.1 prices only the READ (≤ 32 segments × 8 ns = 256 ns). The report never
prices the HARVEST. It cannot be cheap, and the reason is the report's own construction.

§4.4b builds the tree **top-down from the coast**: *"seed river mouths, then grow a tree inland, each
node's elevation set by its parent's plus a slope from the stream power law"*, and the slope needs
the drainage area `A`, which is a sum over the whole catchment upstream. So:

- to know a branch in MY cell you must know its parent, and its parent's parent, back to the mouth;
- to know its slope you must know `A`, which is a sum over every cell that drains into it.

Both walks leave the cell. A function can be pure and still cost a planet.

**The size, COMPUTED.** The hydrology cell is *"a cell of the ladder at a fixed rung `H` … so its
edge is near 2 km"*. On the home planet that is rung 11 (2 048 m), which is 2 570 cells per face
edge, so **39 629 400 cells** — the report's own §3.2 table names that exact grid at 158 MB. Growing
a rooted tree over tens of millions of nodes is the same object §3.3 refuses under reason 1 (*"a
chunk cannot be evaluated without a whole-body pass first"*).

**It also brings back the drift classes.** §3.3 reason 3 refuses a bake because of ORDER: a drainage
accumulation must fix its summation order, and a priority queue pops equal keys in an
implementation-defined order. A tree grown inland from many mouths needs a frontier and a tie-break,
and the accumulation of `A` is a sum over a set. So §8.1's row **"New drift classes: 0"** for
option (d) is false as soon as L4b is on. The fence covers arithmetic only; it covers no order.

**Three claims of §8.1 fall with it**, and all three are stated as facts:

| §8.1 claim for option (d) | What L4b makes of it |
|---|---|
| boot cost per body 0.55 ms | UNSTATED. A tree over ~4×10⁷ cells at even 20 ns a node is ~0.8 s on one core. |
| memory 4 KB + 32–320 KB | UNSTATED. See B2-2: one rung-11 chunk alone touches 3 844 cells. |
| new drift classes 0 | at least two (accumulation order, frontier tie-break). |

**Q2 does not cover this.** Q2 names only the seeding of the mouths. The growth, the accumulation and
the memory are a separate and larger hole, and §4.4b puts L4b *"on the critical path"* while pricing
one sixth of it.

*Game example: the pilot's chunk under the boots wants the river in the valley. To answer, the
planet's shard must first know where every river on the whole planet runs, because this river's floor
height is its mouth's height plus every step between. That is a whole-body pass, and the report
refused whole-body passes on its first page.*

**What would settle it:** price `branches(body, cell)` — its walk, its memory and its ordering — or
delete L4b and tell the owner that rivers wait for a slice that can afford a global structure.

---

### B2-2 — The hydrology cell inverts the ladder's cost order. The top rung becomes DEARER than rung 0.

The harvest amortises only while a chunk sits inside few hydrology cells. At coarse rungs it does
not, and the report never checks.

COMPUTED, from `crates/seed/src/ladder.rs:21` (`CHUNK_EDGE = 62`) and a 2 048 m hydrology cell:

| Rung | A chunk's edge on the ground | Hydrology cells a chunk touches | Harvests per column |
|---|---|---|---|
| 0 | 62 m | **1** | ~1/3 844 — free |
| 6 | 3 968 m | ~4 | ~1/1 000 |
| 11 | **126 976 m** | **3 844** | **1 per column** |

At rung 11 every column sits in its own hydrology cell. A harvest is a cap-intersection test against
the live plate list — up to 64 sites at ~5 ns = **320 ns per column** — and, with L4b, a river-branch
harvest as well. §9.5's rung-11 row adds **+261 µs** and counts none of it.

**The arithmetic, with the harvest, and with §6.2's own cut-off table honoured** (§6.2 says L4a is
*"never"* dropped, because it multiplies the octaves; §9.5 drops it):

```
   rung 11, per column:  L1 40 + L2 5 + affinity 39 + profile 10
                       + L4a gradient 4 x 94 = 376 + shaping 20
                       + one octave 13 + harvest 320             =  823 ns
   3 844 columns x 823 ns = 3 164 us,  + the MEASURED 266 us     = 3 430 us

   rung 0, per column (B2-6's corrected price):  611 ns
   3 844 x 611 = 2 349 us, + the MEASURED 710 us                 = 3 059 us
```

**The top rung costs MORE than rung 0.** That breaks the ladder's own gate: slice 5's bench asserts
*"the top rung cheaper than rung 0"* (`slice_05_generator.md:381-383`), and ruling V8/V9 requires
rung L to be cheaper than rung 0. The report's headline *"the top rung stays 4.4 times cheaper"*
(§9.5) rests on dropping a layer its own §6.2 says is never dropped, and on a harvest it never
counted.

**The memory claim falls with it.** §8.1 says *"an ESTIMATED 32–320 KB of harvested hydrology
cells"*. One rung-11 chunk touches 3 844 cells. At even 1 KB a cell (32 branch segments of 32 bytes)
that is **3.8 MB for ONE chunk**, and the client meshes many.

**The trap is closed on purpose.** Letting the hydrology rung follow the asking rung would cure the
cost and would make `height_m` depend on WHICH CHUNK ASKED — the exact defect §4.0 forbids.

---

### B2-3 — The roughness modulator has no stated size, and both readings break a promise the report makes.

§4.6 writes `amp_effective(o) = amp(o) · r(column)` and never states `r`'s range. §1.3 gives the
target: *"`k_rough ≈ 0.62` gives a walkable, interesting hillside"*, an RMS slope of 14.92° against
today's 1.95°. There are two ways to read the modulator, and arithmetic refutes both.

**Reading one — `r` is one multiplier over all octaves (what §4.6 writes).**
A uniform scale multiplies the tangent of the slope. COMPUTED:
`r = tan(14.92°) / tan(1.95°) = 0.2665 / 0.03405 = 7.83`.
The octave sum in a rough column is then `7.83 × 14 305 m = 112 000 m`. The band is derived from that
sum (`crates/terrain/src/body.rs:233-236`), so, in §9.3's own formula:

```
   band_m = 2 x 112 018 + 367 + 128 = 224 531 m
   chunks per radial column = 224 531 / 62 = 3 621     (today: 470)
   one radial column at rung 0 = 417 ms x 7.7          = 3.2 s
```

**That is 2.9 times worse than the worst row of §9.3's own table** (1 269 chunks, from sizing the
band on `h_max`), and §9.3's comforting sentence — *"if the drawn bound stays near today's 14 km, the
band and the store are unchanged"* — cannot hold while D2 is recommended.

**Reading two — `r` redistributes the amplitudes at a constant sum (a per-column `k_rough`).**
The band is then safe and the LADDER breaks instead. COMPUTED with the crate's own table rule at
`k = 0.62`, 14 octaves, relief 14 304.887 m:

```
   dropped bound at rung 9  = 1 294 m      half a cell at rung 9  =   256 m  ->  5.1x over
   dropped bound at rung 11 = 3 396 m      half a cell at rung 11 = 1 024 m  ->  3.3x over
```

§6.2 promises *"no rung change can move the surface by more than half a cell at that rung"*. Under
reading two it moves the surface by five cells.

**Neither reading is named, and the two are different designs.** The single most important number in
the report's answer to *"the surface is not interesting enough"* is absent, and every value it can
take costs something the report says is safe. It needs its own decision row and its own measurement,
beside M14.

---

### B2-4 — The re-anchored octave table has no amplitude budget, so D3 makes the whole world eight times steeper.

§4.6 moves the coarsest wavelength from 400 km to ~50 km. The code scales the amplitudes so their SUM
is the relief (`crates/terrain/src/body.rs:169-183`). The report changes the wavelengths and never
says what happens to the relief. Under the only rule that exists today:

| | coarsest wavelength | its amplitude | amplitude ÷ wavelength |
|---|---|---|---|
| today | 400 000 m | 7 605 m (§1.1) | 0.0190 |
| after D3, relief unchanged | **50 000 m** | **~7 600 m** | **0.152 — eight times steeper** |

§1.2's whole diagnosis rests on that ratio. Multiplying it by eight puts the RMS slope near 15°
**everywhere on the body at every scale** — on the abyssal plain, on the flood plain, on the desert
pan. §1.3 refuses that outcome by name for a global `k_rough` of 0.70: *"0.70 makes the whole planet
unwalkable"*.

So D3 as written either (i) makes the planet unwalkable, or (ii) silently needs a new, unstated rule
for the octave relief — the number that also sizes the band (W1b), the store and the column cost.
**§10.5 deletes the `0.4 %` relief law and puts `h_max` in its place, but `h_max` bounds a MOUNTAIN,
not an octave sum.** Nothing in the report says what the octave table's total amplitude becomes. That
is the largest unstated decision in the design.

---

### B2-5 — `BodyFacts` cannot carry the climate the report builds on it. Four layers become uncomputable.

§5.4 fixes the record at **seven integers**: gravity, day length, obliquity, insolation, equilibrium
temperature, water inventory, age. Now read what the layers consume:

| Consumer | The fact it needs | In the seven? |
|---|---|---|
| §4.5 `c_p ≈ (7/2)·R_gas/μ`, hence `Γ_env`, hence the snow line | **mean molecular weight** (`crates/physics/src/taxonomy.rs:882`) | **NO** |
| §4.7 `w_rivers`, `w_aeolian` (*"atmosphere present"*) | the atmosphere's presence | **NO** |
| §4.9 L7, whose temperature term IS `Γ_env` | the same | **NO** |
| §5.3 `pole_axis` as a unit vector | obliquity **and the axis's azimuth** — one angle cannot orient an axis in three dimensions | **NO** |
| the owner's word *trajectory* | orbital eccentricity, and a year length for the seasons | **NO** |
| §4.7 `w_terrain` (B-N5, still OWED) | *"has a solid surface"* | **NO** |

§5.1's own fact list names the atmosphere as needed and marks it *"yes — exists today"* in the
forest. §5.4 then does not carry it. The two sections disagree, and D7 makes it worse: it **drops
`c_p` from the constant list** because `c_p` is derived from a fact the record does not send.

**Consequence.** With the record as specified, neither host can compute the environmental lapse rate.
The snow line, the glacial switch, the dune weight, the river weight and the whole rebuilt biome —
the owner's FIRST requirement — have no input. This is the SL6 ask the report puts to the owner, and
it is short by the facts four of its own layers read.

*Game example: the pilot climbs a ridge on the home planet and asks whether there is snow at 3 000 m.
The client holds the seed, the look radius and seven integers. It cannot answer, because the air's
molecular weight — the number that decides how fast the air cools with height — was never sent.*

---

## Defects

### B2-6 — The affinity noise is charged to a budget line that does not grow, and the corrected cost is RED without rivers.

§4.2 ends: *"5 ns per column for the isostatic arithmetic, plus the affinity field's 2–3 noise calls,
**which are counted in L6's octave budget** because they use the same machinery."* §9.1's L6 row is
`+13 | one extra octave (15 against 14)`. Thirteen nanoseconds is one octave. It cannot also be three
noise calls. The affinity calls are unpaid, and the gradient pays them four more times, because
§4.4a's `h_smooth` contains L2.

COMPUTED, at the report's own MEASURED 13 ns per `noise3` call:

```
   h_smooth per evaluation = L1 40 + L2 5 + affinity 3 x 13 = 39 + profile 10  =  94 ns
                                                       (the report uses 55 ns)
   the gradient = 4 evaluations                    = 376 ns   (the report: 220)
   the centre evaluation                           =  94 ns   (the report:  55)
   unpaid                                          = +195 ns per column
```

| | report | corrected | cold surface chunk | cold cave-dense chunk |
|---|---|---|---|---|
| no rivers | 416 ns | **611 ns** | 3.30 + 2.35 = **5.65 ms** | 6.10 + 2.35 = **8.45 ms — RED** |
| with rivers | 672 ns | **867 ns** | **6.63 ms** | 6.10 + 3.33 = **9.43 ms — 18 % over** |

**The red row moves from "with rivers" to "always".** §9.2's three cures all aim at the river lookup,
which is no longer the cause, and §12's ordering (L4a first, L4b behind M3) no longer saves the
budget, because L4a alone is over on a cave-dense chunk.

The allowance is also slightly generous. §9.1 divides 4.7 ms by 3 844 columns, while the MEASURED
sample box carries the halo, which is *"six per cent more columns"* (`slice_06_extractor.md:355`).
COMPUTED: 4 075 columns → **1 153 ns**, not 1 223 ns.

---

### B2-7 — §6.2's premise is false: today's octave ladder already breaks the half-cell rule at four rungs.

§6.2 introduces the half-cell rule as *"the one the octaves already obey"* and concludes *"no rung
change can move the surface by more than half a cell at that rung"*. COMPUTED from
`crates/terrain/src/body.rs:169-183` and `octaves_at` (`body.rs:249-256`), on the home planet
(14 octaves, `k = 0.468 338 1`, relief 14 304.887 m):

| Rung | Dropped amplitude | Half a cell | Verdict |
|---|---|---|---|
| 5 | 15.2 m | 16 m | ok |
| 8 | **150.6 m** | 128 m | **1.18× over** |
| 9 | **322.0 m** | 256 m | **1.26× over** |
| 10 | **687.9 m** | 512 m | **1.34× over** |
| 11 | **1 469.2 m** | 1 024 m | **1.43× over** |

The 322 m at rung 9 is the report's own number (§4.6 calls it *"the whole tier-agreement bound of
322 m"*), so the report holds both halves of the contradiction. The SL8 promise in §10.6 — the third
of its *"three directions, all now covered"* — is therefore not established by the rule it cites.

The re-anchor of §4.6 happens to cure it. COMPUTED: with 15 octaves the dropped bound is 70.5 m at
rung 8 and 688 m at rung 11, both under half a cell. **The report should claim that result and
withdraw the false premise** — and it must then survive B2-3's modulator, which multiplies these
numbers.

---

### B2-8 — Mars's ceiling is arithmetically wrong, and the wrong number is the headline evidence for D4.

§4.3 states `σ_y = 2 800 × 9.81 × 8 850 = 243 MPa` and `ρ_c = 2 800` for every body. COMPUTED with
those exact inputs:

```
   h_max(Mars) = 243 091 800 / (2 800 x 3.71) = 23 401 m       the report says 22 594 m
   Olympus Mons 21 900 m -> 93.6 % of the ceiling              the report says 97 %
```

Earth (8 850), Venus (9 788) and the Moon (53 592) all reproduce exactly, so the formula and the
constants are right and the Mars row alone is wrong. The value 22 594 needs `g = 3.84`, which is not
Mars's gravity.

**Why it matters.** The 97 % appears in the summary table, in §4.3, in §15 and in D4's justification
as the strongest independent check of the relation. At 93.6 % the spread over three bodies is
Earth 100 %, Mars 94 %, Venus 112 % — still a usable scale, and the report's stated *"±15 %"* is then
the honest bound, but the Mars agreement is no longer *"within 3 %"*. This is the defect class the
report accepted as B-D8: a COMPUTED number that does not reproduce inside its own document.

---

### B2-9 — The isostatic step omits water loading, so the 4 455 m agreement with Earth is an artefact.

§4.2's table is unloaded Airy: `e = t(1 − ρ_c/ρ_m)`, continental 5 303 m, oceanic 848 m, step
4 455 m, followed by *"Earth's real figure is a 4.5 km mean step"*. Earth's 4.5 km is measured with
**four kilometres of sea water standing on the ocean floor**, and water loading pushes that floor
down. The report's model carries no water term at all.

COMPUTED, the same numbers with the water column included (mass balance at the compensation depth,
`ρ_w = 1 030`, the continent at sea level):

```
   t_c(rho_c - rho_m) = d(rho_w - rho_m) + t_o(rho_o - rho_m)
   35 000(2 800 - 3 300) = d(1 030 - 3 300) + 7 000(2 900 - 3 300)
   d = 6 476 m below the continent            the report's step: 4 455 m
```

The correct treatment of the report's own thicknesses gives a **2 021 m deeper** ocean. The agreement
with Earth comes from leaving a term out, not from the physics falling out. A geologist finds this at
once, because freeboard and water loading are the first pair of numbers in the subject.

The error does not even stay constant. D5 makes the **water inventory a per-body draw**, so the
loading term varies from body to body: a dry world's basin should stand shallower against its
continents and a water-rich world's deeper. Under §4.2 every world gets the same 4 455 m step,
whatever its ocean weighs.

---

### B2-10 — The quantisation cure is stated as absolute and is conditional, and M13 cannot fail.

§5.4 says *"Then a drifting input is HARMLESS"*, and M13 is *"the gate that makes this ask lawful"*.
Quantisation does not remove drift. It removes drift except at a grid boundary. If two hosts compute
`g` and one lands at `4.684 500 0 − ε` and the other at `4.684 500 0 + ε`, the millimetre integer
differs, `h_max` differs, and every mountain on that body differs — which is the failure the section
itself describes.

M13 as written (*"move each incoming fact by a thousand ulps and require the same body"*) passes for
any value that is not within a thousand ulps of a boundary, which is almost every value. **It is a
test that cannot fail for a randomly chosen body**, and the project's standing rule is that a
no-drift claim must be a measurement that could have failed.

The structural cure exists and is not stated: **one host computes the fact, quantises it, and every
other host receives the INTEGER.** That is already true of the client, which is sent `BodyFacts`. It
is not true of a second server, which §5.4 names as the real hazard (*"the golden self-check
disagrees between two SERVERS"*). The honest form is: the integer is authored once per body by the
realm that owns the body, and no other process re-derives it.

---

### B2-11 — SL6's "find the local formulation first" is not attempted: the draw is never moved below the fence.

SL6 says *"Find the local formulation first; there usually is one."* §5.4's *"what doing without
costs"* gives three options, and option (i) is *"the generator draws `g` from the body's seed, which
creates a SECOND gravity that disagrees with the mass the forest drew"*.

That refutes a DUPLICATE draw. It does not consider the obvious fourth option: **move the body's
physical draw below the fence**, so `vd-seed` or `vd-terrain` owns the mass–radius law and
`vd-physics` READS it. There is then one gravity, not two; both hosts derive it from the seed; no
record crosses; no refusal path is needed; the seven grids are not needed; and M13 is not needed.

The report supplies the reason this is the natural shape: SL10 already says the static world is a
function of `(seed, address)`, and a body's mass is a function of its seed. The mass law today is the
Chen–Kipping/Zeng fit in `crates/physics/src/taxonomy.rs` — real work to fence, and a world-identity
bump. **That cost should be stated and compared, not skipped.** The ask asks the owner to open a wire
lane; the owner should see the option that opens none.

---

### B2-12 — No lithology reaches the surface, so the picture's cliff bands cannot exist either.

§4.8 lists what the stack cannot make and names one cause: the field is single-valued. That is true
for a pillar, an arch and an undercut. It is NOT the reason for what else is missing.

The crate already holds a stratigraphy: `StrataTable` with topsoil, subsoil, sediment and bedrock,
one of three sediments and one of five bedrocks (`crates/terrain/src/body.rs:200-206`;
`crates/terrain/src/strata.rs:177-200`). **The height field never reads any of it.** `height_m` is
the radius plus noise (`crates/terrain/src/height.rs:17-28`). So every landform in the report is cut
out of one homogeneous material.

Real land is not. Hard rock over soft rock is what makes:

- an **escarpment** and a **cuesta** — a hard cap layer holding a cliff line for tens of kilometres;
- a **mesa** and a **butte** — the same cap, isolated;
- **benched valley walls** — the stepped cliff-and-ledge profile the reference picture shows in every
  gorge;
- a **waterfall** — a river crossing a hard band;
- **differential ridges** — the resistant beds standing out of a fold belt.

> **Differential erosion, in plain words.** Water and frost take soft rock away faster than hard rock.
> Where the layers are stacked, the hard ones stand out as ledges and the soft ones cut back as
> slopes. That stepping is most of what makes a rock face look like rock. *Game example: the pilot
> walks into a canyon on the home planet. Under this report the wall is one smooth ramp of one
> substance from floor to rim. In the reference picture the same wall is four bands: a sandstone
> ledge, a shale slope, another ledge, then talus.*

§10.3 and D10 debate a layout-driven ROCK for the ore question and never notice that a rock map is
also the cheapest believability win available: the strata exist, the biome already selects among
them, and one hardness term in §4.4a's transfer curve produces benching with no new evaluation.
D10's option (i) (*"no rock map at all"*) closes this door too, and the report never tells the owner
what option (i) costs in LOOK.

**A geologist names this gap in one glance**, and it sits under the owner's exact words: *"the
surface will not be interesting enough"*.

---

### B2-13 — Climate is baked into the static shape, and the owner asked for weather. The split is never stated.

SL10 lets the client derive what `(seed, address)` decides and forbids deriving live state. §4.5 puts
five climate operators into the seed-shaped field: the snow line, glacial carving, dune fields, the
coastal notch and the rain shadow. Two questions follow, and the report answers neither:

1. **Which of them are shape, and which are surface state?** Glacial carving is shape: a U-valley
   stays for ten thousand years. Snow is not: the owner asked for weather, so snow cover must move
   with the season and with the storm. §4.5 gives one `z_snow(lat)` and never says whether it decides
   GEOMETRY, the surface stratum (`Stratum::Snow` already exists, `strata.rs:114-131`), or both.
2. **If snow is live, what crosses?** §5.5 opens the weather ask and hands it to domain 05, which is
   fair. But the report cannot both hand weather away and use a snow line to answer the owner's
   *"snow-capped ridges"* in its summary. A snow line that never changes is a painted-on cap; a snow
   line that changes is live state and needs the lane §5.5 leaves UNANSWERED.

**One sentence fixes it:** name each L5 operator as SHAPE (static, seed-derived, SL10-legal) or COVER
(live, a diff from the owning realm), and say that the report designs the first and hands the second
to domain 05.

---

### B2-14 — Held & Hou's result is read as a sine where the source gives an angle, and the band count has no law.

§4.5 writes `sin(φ_H) = sqrt(5·g·H·Δθ/(3·Ω²·a²·θ₀))` and tests `|dir · pole_axis| < sin φ_H`. Held &
Hou (1980) give the cell edge as `φ_H` in radians, a small-angle result, not as its sine. The
substitution is made because `Gf` grants no arcsine — a good instinct, and nearly free at fast spin.
It is not free at slow spin. COMPUTED with the report's own `H = 15 km`, `Δθ/θ₀ = 1/3`:

| Spin | the value | read as `sin φ_H` (the report) | read as `φ_H` in radians (the source) |
|---|---|---|---|
| Earth | 0.615 | 38.0° | 35.3° |
| half Earth's | 1.231 | **90° — one cell to the pole** | **70.5°** |
| twice Earth's | 0.308 | 17.9° | 17.6° |

The report's table says a 48-hour day gives *"1 cell per hemisphere"* reaching the pole. That is the
substitution, not the physics, and it decides the biome map of every slow rotator in the world.

**And the "cells per hemisphere" column has no derivation at all.** It reads 1, 1, 2, 5, 10. Nothing
in the report says how a cell edge becomes a band count. §4.9's L7 then CONSUMES that count (*"the
circulation band (wet belt / dry belt) from Held–Hou"*). The one number the biome layer reads out of
the spin has no law, and M11 calibrates `H` and `Δθ`, not the count.

*Game example: the pilot lands on a world with a 48-hour day and expects a wide wet belt at the
equator and a dry belt above it. Under §4.5 the whole hemisphere is one cell, there is no dry belt,
and the planet has no deserts at all — from a substitution made for the fence.*

---

### B2-15 — The hypsometry solves for a sea level on a field that is not the one drawn.

§7.1 evaluates **`h_smooth` (L1 + L2 + L3's profile)** at 4 096 directions and bisects for the water
inventory. The surface the player stands on is `h_smooth + L3's ridged octaves + L4 + L6`. The
excluded part is not small: the octave sum alone is the whole relief (14 305 m, §1.1), and §1.4
MEASURES its spread at σ = 2 297 m.

So the solved level is the level that holds `V_water` **against the smooth field**, and the flooded
fraction of the real field differs. The report's accuracy claim covers the sampling only:
*"the standard error of an ocean fraction near 0.5 from 4 096 samples is 0.78 %"* — COMPUTED
`sqrt(0.25/4 096) = 0.0078`, which is right, and which measures the wrong thing. The model error is
unstated and is the larger of the two.

Two companions in the same section:
- `h_smooth` is priced at **55 ns** in §4.4a and at **135 ns** in §7.1 (`4 096 × 135 ns = 0.55 ms`).
  One price is wrong, and 0.55 ms is quoted five times, including in §8.1's boot row.
- A basin that fills into a lake (§4.4b) holds water the ocean inventory has already spent. The
  balance counts it twice.

---

### B2-16 — §9.5 contradicts §6.2 about L4a, and the 4.4× headline depends on the contradiction.

§6.2's cut-off table: *"L4a valley transfer | it MULTIPLIES the octaves, so it is never dropped
(§4.6's rule) | **never**"*.
§9.5: *"At rung 11 the ridge lines, the channel incision and the climate operators are all past their
derived cut-offs, so only L1, L2, L3's profile and one extra octave remain"* — L4a's 240 ns is gone.

Both cannot hold. With L4a kept, as §6.2 requires, rung 11 gains 3 844 × 240 ns = 923 µs, so the row
reads 1 450 µs rather than 527 µs even before B2-2's harvest, and *"4.4 times cheaper"* becomes 2.1×.
With the harvest it inverts (B2-2).

---

## Weaknesses

### B2-17 — The 0.55 ms body draw runs on the client's message path, which is an SL8 tick hitch.

`crates/client/src/chunks.rs:266-289`: `state_surface` calls `BodyDefinition::from_seed` inline, once
per realm, when a surface statement arrives. §7.1 adds 4 096 field evaluations to `from_seed`.

COMPUTED: one body is 0.55 ms; a star system stating a planet, seven siblings and twenty moons in one
frame is **28 × 0.55 = 15.4 ms**, a dropped frame at 60 Hz. The seam taxonomy names `tick-hitch` as a
seam, and SL8 says a seam is a defect. The report prices the hypsometry per body and never says where
it runs. The fix is easy — draw the body on a worker — but it must be said.

### B2-18 — The recommended first landing has no river, and the reference picture has one.

D11 recommends *"(iii), and L4a first"*, and §4.4a is honest that L4a *"has no drainage at all. A
'valley' made this way does not lead anywhere."* L4b is red on the budget (B2-6) and unsolved on the
mouths (Q2). So the first thing the owner sees has valleys that lead nowhere and no water above the
sea sphere. M4's pass mark asks for *"a coast and a vista with a range at 30–50 km"* and never asks
for a river. The owner's reference picture holds a river through the valley floor, and doc 01 treats
it as a named orienter. **Say it plainly:** the first slice reaches the ridges, the valleys and the
coast, and does not reach the river.

### B2-19 — Option (c) is priced at 90 ns per column against a MEASURED 185 ns for the same thing.

§8.1's row for *(c) pure closed form* reads `~90 ns`. Today's field IS a closed-form octave sum, and
slice 5 MEASURES it at **185 ns per column** (`slice_05_generator.md:372-386`, 710 µs ÷ 3 844). The
comparison table understates the option it rejects by half, in a document that corrected revision 1
for refusing the bake at its pessimistic corner (B-W4).

### B2-20 — The picture holds things no layer makes, and §4.8 names only the overhang.

§4.8 is a good section and its list is short. Against the reference vista the stack also does not
make: **roads and terraced fields** (culture, not geology — no layer owns them and no section says
they are out of scope), **the blue distance haze** (an atmosphere the renderer must draw, and §5.4
sends no atmospheric fact — see B2-5), **moraines, fjords and hanging valleys** (glaciation is *"one
switch on L4a's transfer curve"*), and **the benched cliff** (B2-12). The owner counts everything in
the frame. §4.8's table is where the honest list belongs.

### B2-21 — Terms the report teaches with are used before they are explained.

The brief says every term is explained in plain words the first time. These are not: **stagnant lid**
(§4.7's table), **Nyquist** (§4.6 — the sentence that justifies the 2 m floor), **ulps** (M13, the
gate of the SL6 ask), **flexural wavelength** (§10.5), **continentality** (§4.9's table), **foreland
basin** (§4.3), **size-frequency distribution** (§4.7), **abyssal hills** (§7.2), **cirque** (§4.5).
Several carry a decision, so the owner is asked to rule on words the report has not given him.

### B2-22 — `w_terrain` is left OWED, so D13's own answer has no input.

B-N5 is answered *"PARTLY FIXED, PARTLY OWED"*. D13 asks the owner what an ice giant does, and the
answer (`from_seed` returns `None`) rests on a fact §5.4's record does not carry. A decision the owner
is asked to take should not rest on a derivation the report says it still owes. The same gap makes
§10.6's HR5 cure awkward: `home.rs` is asked to state an ice giant as a pinned body, and an ice giant
has no `BodyDefinition` to pin.

---

## Notes

### B2-23 — The super-earth's ceiling is understated, in the report's own favour.

§4.3 uses `ρ_c = 2 800` for a body whose mean density it COMPUTES at 8 608 kg/m³. A compressed crust
is denser than Earth's, so `h_max = σ_y/(ρ_c·g)` is lower still than 3 006 m, and D4's case is
stronger than stated. One sentence, because the report elsewhere refuses a number that is right by
luck.

### B2-24 — A 4 KB plate list makes `BodyDefinition` a heavy `Copy` type.

§4.0 puts `plates: [Plate; 64]` at an ESTIMATED 64 bytes a row inside a struct that is `Copy` and
`PartialEq` (`crates/terrain/src/body.rs:88-91`). Today the octave table is `16 × 24 = 384 bytes` and
the whole struct is well under a kilobyte. A 4 KB `Copy` type is a different object: every
pass-by-value copies it and `PartialEq` compares 64 rows including the dead suffix. The client holds
it behind an `Arc` (`chunks.rs:284`) and is safe; the crate's own call sites are the question. M10
prints the size, which is right.

### B2-25 — A triple junction is decided by a tie-break, and nobody says what it looks like.

§4.1 decides the boundary kind from the two nearest plates' drifts. Nothing states what happens where
THREE plates meet, where the nearest-two rule is ambiguous and §4.0's tie-break by site index picks
the landform. A triple junction is a real place with a real look — three boundaries meeting at a
point, as at Afar. One sentence and one picture would settle whether an index comparison is an
acceptable answer there.

---

## Checked, and sound

I tried to break each of these and could not. Each was re-derived from the code or from the
arithmetic.

1. **`SHORT_WAVE_M = 30` and the 48.83 m floor.** `body.rs:25` and the loop at `:169` confirm it, and
   `400 000 / 2¹³ = 48.83 m` reproduces the crate's MEASURED 14 octaves. The spectral hole is real and
   is the report's best single finding.
2. **The re-anchored table's count.** `50 000 / 2¹⁴ = 3.05 > 2` and `50 000 / 2¹⁵ = 1.53 < 2` give 15
   octaves, and `keep = 15 − 11 = 4` at the top rung, so the clamp arm at `body.rs:253-255` still
   never fires. The compile assertion `50 000 / 2¹⁶ = 0 < 2` holds.
3. **The band arithmetic.** `2 × 14 305 + 367 + 128 = 29 105` reproduces the MEASURED
   `cells_in_band(0)`, and `body.rs:233-236` with `ladder.rs:88-98` confirms that `floor_m` moves with
   the relief. W1b is correctly raised as a one-way door.
4. **The ladder refuses the gas giants.** `N_MAX = 2²⁶` (`ladder.rs:25`) gives `2N/π = 42 723 km`, so
   Jupiter and Saturn are refused and the ice giants are not. The correction of revision 1 is right.
5. **`Gf`'s surface.** `gf.rs:102-148` grants `sqrt`, `floor`, `trunc`, `abs`, `lesser`, `greater`,
   `clamp` and `is_finite`, and the module doc names `min`/`max`, `mul_add`, `powi` and every
   transcendental as deliberately absent. §0's row is now correct.
6. **`fluid_at` is one rule per body.** `chunk.rs:204-213` confirms it, and `ColumnField` already
   carries per-column data (`chunk.rs:6-14`), so §4.4c's second radius is the right shape and the
   12-byte record genuinely does not change.
7. **The client holds a seed and a radius only.** `chunks.rs:276-280` builds the body from
   `FrameRef::PlanetCentered { planet_seed }` and `Boundary::Shell { r }`, and
   `tests/tests/crate_isolation.rs:322-324` refuses the `vd-physics` edge. The SL6 premise is true.
8. **A new tag is the right carrier.** `crates/wire/src/version.rs:10-14` says a trailing field is not
   additive in postcard. W4's ruling is correct and the refusal path is the right cure.
9. **`b = R·(d · n̂)` and its error.** The construction is fenced and the sine-against-angle bound
   (0.99 % at 819 km) is arithmetically right.
10. **`σ_y = 2 800 × 9.81 × 8 850 = 243.1 MPa`**, and Earth, Venus and the Moon reproduce exactly from
    it. Only the Mars row fails (B2-8).
11. **The super-earth's mass and gravity.** `R/R⊕ = (M/M⊕)^0.27` gives 10.4 M⊕, 8 604 kg/m³ and
    `g = 28.9`, and the 3 006 m ceiling follows. Correctly sourced.
12. **The Hadley test needs no sine.** `dir · pole_axis` IS `sin(latitude)`, and `height.rs:45`
    already does exactly that. The trick is sound; only the reading of the source is not (B2-14).
13. **The hypsometry's sampling error.** `sqrt(0.25 / 4 096) = 0.78 %` is right, and 24 halvings of a
    40 km range reach 2.4 mm. The order-fixed construction is genuinely fenced.
14. **The passive-margin cure.** A crust FIELD does put most coasts inside a plate, and it lets one
    plate hold a continent and an ocean floor. This is the best change in revision 2.
15. **The single-valued limit.** `height.rs:17` returns one radius, and `carve.rs` is the only
    three-dimensional term. §4.8 and D14 are the honest treatment.
16. **The client's measured baseline.** `slice_07_client_link.md:235` confirms that 4.18 ms per chunk
    covers box, extraction, positions and normals, so §9.4 adds the right work to the right number.
