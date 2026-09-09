# REFUTATION A of DOMAIN 04 — `04_detail_rungs.md`, REVISION 1 (round 2)

**Date:** 2026-09-08. **Lens:** THE LAWS AND THE CODE. Is every claim about the repository true? Is
every law gate passed? Is every number sourced?

**This file replaces round 1.** Round 1 refuted revision 0. The target document answers round 1 item
by item in its own §14, so those findings stay on the record there. This file refutes **revision 1**.

**Method.** I re-opened every file the document cites: `crates/terrain/src/{body,height,chunk,carve,
strata,extract,seat,compose,gf,noise,lattice,home,tag,digest}.rs`, `crates/seed/src/{ladder,bend}.rs`,
`crates/core/src/look.rs`, `crates/physics/src/taxonomy.rs`. I re-read the rulings
(`owner_decisions_2026-09-07_voxels.md` A4, A5, B-1..B-13, S5-2, S5-3, S5-6, V10's budget line), the
format sitting (`topic_00_format_sitting.md` §B1), the measurement tables
(`docs/investigation/2026-09-07/slice_05_generator.md`, `slice_06_extractor.md`,
`slice_07_client_link.md`), and — because §3 is a CONTRACT — the two neighbour documents the contract
names: `02_planet_layout.md` and `03_erosion_rivers.md` **in their current revisions**. I re-did every
piece of arithmetic, in `python3`, from the drawn constants.

**Verdict in one line.** Revision 1 fixed most of round 1, and its octave table, its Lipschitz
derivation and its damping arithmetic now reproduce exactly — but the whole octave anchor stands on an
erosion spacing that DOMAIN 03 has since **withdrawn and refuted by name**; the relief law is deleted
and replaced by a ceiling that is unbounded on a small body and kills a landed test; SL6 is marked
PASSED for a datum the neighbour document is formally ASKING the owner for; and the SL8 headline
("tighter everywhere", "1.26 pixels") reverses when today's bound is computed with the roughness the
home planet actually drew.

---

## BLOCKERS

### B1 — The octave table is anchored to a lattice DOMAIN 03 has withdrawn, and the anchor is refuted by name

§3.2 requirement 1: *"The grid's spacing is a RUNG of the ladder, never a number of metres. On the
home planet the top rung is 2 048 m and the erosion grid is 8 192 m, which is `2^(rungs + 1)`."*
§4.3 builds everything on it: `λ₀ = 2·S_e = 16 384 m`, `n = rungs + 3 − log2(C)`, and therefore
`n = rungs − 1 = 11` at `C = 16` — which is the single argument that keeps the landed ladder test
(D4-15), the whole §4.8 table, every bound in §5.2 and the pixel gate in §0.

`03_erosion_rivers.md` §4.1, in the revision now in the tree, refutes exactly this:

> *"Revision 1 said three things that cannot all be true: that the macro grid is the ladder's own
> lattice at rung `M` … COMPUTED: `N = 5 263 360`, and `642 × 8 192 = 5 259 264`, which leaves 4 096 m
> over. … it is NOT a multiple of 8 192. … Revision 2 takes the second reading and DECLARES it."*

DOMAIN 03's answer is `n_macro = 640`, a node of **8 224 m**, 2 457 600 nodes, **not a rung, and it
never appears in an address** (its own D1 row). Three consequences for DOMAIN 04, each COMPUTED:

1. `S_e = 8 224 m`, so `λ₀ = 16 448 m` and
   `n = log2(16 448 / 16) + 1 = log2(1 028) + 1 = 11.006`. **The octave count is no longer an
   integer**, so "one octave dropped per rung", `n = rungs − 1`, and the argument that
   `every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it` (`body.rs:355-365`) survives all
   collapse together.
2. §4.2's *"`t` is always a dyadic rational … on the home planet at rung 0, `t` is an odd multiple of
   `1/16384`"* is false under the neighbour's lattice: one node spans `5 263 360 / 640 = 8 224` rung-0
   cells, so `t` is an odd multiple of `1/16 448`, and `16 448 = 2⁶ · 257` is **not** a power of two.
   The stated reason for the bicubic's determinism therefore does not hold as written. (The bicubic is
   still deterministic — IEEE round-to-nearest in a fixed order is identical on both legs — so the
   *conclusion* survives; the *stated argument* does not, and the sentence "Both refuters re-derived
   this and it holds" is now false.)
3. §3.2's whole framing — "a RUNG of the ladder, never a number of metres" — is the opposite of what
   the neighbour landed. The document's §3 is written as a CONTRACT that "if a neighbour cannot meet a
   row, the consequence changes and §13 records it as open". The row is not met, and §13 does not
   record it.

**What is owed:** either re-derive `n` from `n_macro` (a divisor of `N`, not a power of two), or state
that the octave table is anchored to the LADDER's top rung (2 048 m) and re-run every number in §4.8,
§5.2 and §9.

### B2 — The relief law is deleted and only a CEILING replaces it; on a small body the ceiling is unbounded and a landed test panics

§4.3 proposal 1 replaces the relief with `relief_ceiling_m = crustal_strength / (rock_density ×
surface_gravity)`, and §11's "no magic numbers" row states that the proposal **DELETES** *"the relief's
0.4 % with its 200 m and 12 000 m clamps (`body.rs:146-149`)"*. The document never says what draws the
relief afterwards. Two readings, and both fail:

- **If `relief = ceiling × a seed factor`,** the law has no upper reference to the body's own size.
  COMPUTED for a 10 km rock of density 2 700: `g = (4/3)πGρR = 7.55 × 10⁻³ m/s²`, so
  `ceiling = 200 × 10⁶ / (2 700 × 7.55 × 10⁻³) = 9.81 × 10⁶ m` — **9 810 km of relief on a 10 km
  body**. `crust_m = relief + strata + caves + 64` (`body.rs:234`) then exceeds the surface radius, and
  `Ladder::for_radius` returns `None` (`ladder.rs:89-91`, the `crust >= surface` refusal). The landed
  test `BodyDefinition::from_seed(5, 3_000.0).expect("a rock")` (`body.rs:386`) **panics**, and every
  small body in THE world stops existing. §11's "SL5 PASSED … the law is TOTAL … a 10 km asteroid" is
  argued for the OCTAVE COUNT only (§4.3's last paragraph) and never for the relief.
- **If the ceiling only CLAMPS a share-of-radius draw,** then the 0.4 % is not deleted, §11's
  no-magic-numbers row is wrong about what it deletes, and the home planet's relief moves from
  `min(0.4 % · R, 12 000) × [0.5, 1.5)` to `min(0.4 % · R, 21 500) × [0.5, 1.5)` — up to 20 105 m
  instead of 14 305 m, which is a different band, a different `floor_m` and a different address for
  every cell (D4-14 says a world epoch, but §4.3's "the new law gives the same order on this body" hides
  the size of the move).

The neighbour has already answered this and DOMAIN 04 does not read it: `02_planet_layout.md` change 3
calls the same relation *"a **scale** with a ±15 % spread, **not a wall**"*, and fixes
`σ_y = 243 MPa` from Everest — against DOMAIN 04's per-body seed draw in 100–300 MPa. Two documents
now own one law with two different constants, and the law moves the ladder band on every body.

### B3 — SL6 is marked PASSED for a datum that must cross a realm boundary, and the "the snap makes it harmless" argument does not transfer

§11: *"**SL6 — ask before new data crosses: PASSED.** This domain adds NO new crossing."*
§4.3: the surface gravity *"becomes an INPUT to `BodyDefinition::from_seed`, as an INTEGER in mm/s²,
exactly as the look radius is an input today (`body.rs:140-142`; the snap makes an input drift
harmless, `home.rs:45-60`)."*

Four facts from the code and one from the neighbour:

1. **Nothing on the wire carries gravity.** `TAG_SURFACE` carries `SurfaceStmt { frame, generator }`
   and nothing else (`crates/core/src/look.rs:56-75`). The realm's extent rides `TAG_LOOK`. A new fact
   therefore needs a new field or a new tag — which is SL6's second clause ("and before adding a wire
   arm") word for word.
2. **The terrain crate may not compute it.** `body.rs:86` states in writing: *"no float from an
   unfenced crate can enter the recipe as a body."* `surface_gravity_mps2` lives in `vd-physics`
   (`taxonomy.rs:750-752`, correctly cited) and its mass comes through `segmented_power_law`, which is
   `coeff * x.powf(exponent)` (`taxonomy.rs:616`) — libm, outside the fence, not correctly rounded, and
   free to differ between two targets.
3. **The snap analogy fails.** The look radius is harmless because the LADDER SNAPS IT: the test at
   `home.rs:47-76` moves it by a thousand ulps and gets the same body, and the snap unit is
   `2¹¹ · 2/π ≈ 1 303 m`. A gravity in mm/s² has **no snap inside the recipe**. COMPUTED: one mm/s²
   moves the ceiling by `21 500 / 3 450 = 6.2 m`, which moves the relief, every octave amplitude, the
   sea offset (`body.rs:189`) and the highland threshold (`body.rs:214`) — so two hosts that disagree by
   one mm/s² produce different bytes in every chunk on the planet. SL10 clause 3 is a byte-for-byte
   measurement; it fails.
4. **The neighbour is already asking.** `02_planet_layout.md` §5.4 is titled *"★ SL6 ASK ONE — a
   `BodyFacts` record, QUANTISED, as a new tag"*, carries gravity as `u32` mm/s², and owes an M13 gate
   ("move each incoming fact by a thousand ulps and require the same body"). DOMAIN 04 needs that ask
   granted and that gate green; it may not mark SL6 PASSED while it is open.

### B4 — §5.2's comparison with today is wrong by a factor of two, and with the right number the headline reverses

§5.2: *"Today's per-body constant `dropped_bound_m(3)` is COMPUTED at **6.11 m** … The proposal's
alpine column at rung 3 is 3.58 m and its plain column is 0.33 m: **tighter everywhere**, and 18 times
tighter on the plain."* §0 and §14 (answer A-W1) both repeat it.

`dropped_bound_m` is `relief_bound_m(0) − relief_bound_m(rung)` (`body.rs:274-277`), and the
amplitudes are `relief × k^i / Σk^i` (`body.rs:176,182`). The home planet drew `k_rough = 0.468 338 1`
and `relief = 14 304.887 m` — both MEASURED, and both quoted by this document's own §0 and §1 (via
`02_planet_layout.md:171`). COMPUTED (`python3`, 14 octaves):

| rung | today's `dropped_bound_m` | the proposal, alpine (§5.2) | verdict |
|---|---|---|---|
| 1 | **0.40 m** | 0.22 m | tighter |
| 2 | **1.24 m** | 0.98 m | tighter |
| 3 | **3.05 m** | 3.58 m | **17 % LOOSER** |
| 6 | **32.76 m** | 73.60 m | **2.2× LOOSER** |
| 9 | **321.97 m** | 611.03 m | **1.9× LOOSER** |
| 11 | **1 469.16 m** | 1 965.62 m | **1.34× LOOSER** |

The document's 6.11 m is the number you get at `k = 0.5` (COMPUTED: 6.112 m), not at the roughness the
home planet drew. So "tighter everywhere" is false on the alpine column at every rung from 3 up, and
"18 times tighter on the plain" is 9.2 times (3.05 / 0.33). In pixels, on the document's own
convention, today's rung-6 bound is `32.76 / 55 680 / 1.091 mrad = 0.54 px` against the proposal's
1.21 px: **the proposal roughly doubles the worst rung change the eye can see**, and §0 tells the owner
the opposite. The honest sentence is: the bound tightens where the ground is smooth and loosens where
it is steep, which is the price of putting the roughness where the mountains are.

---

## DEFECTS

### D1 — The bound a coarse rung can actually STATE is not the bound in the table

§5.2: *"`a_{n−L}(dir)` is the amplitude the rung-`L` evaluation stopped at. The coarse rung already
holds it. So every rung states its own bound for its own column at the cost of two operations."*

The table's numbers are the EXACT sum of the dropped rung-0 amplitudes. A coarse rung cannot compute
that sum: every dropped amplitude is `a_{i+1} = a_i · k(dir) · damp(grad s_i) · ch(dc)`, and
`damp(grad s_i)` reads octaves the coarse rung never evaluated (§4.1, §4.4 rule 3). What it can state
is a geometric envelope with the largest lawful gain, `a_m / (1 − k_max)`; §4.4 gives
`k_max < 0.68`, so the factor is 3.125. COMPUTED, alpine at rung 9:
`206.846 × 3.125 × 1.413 = 913 m`, which at 445 440 m is `2.05 mrad = 1.88 px` — not 1.26 px.
The crossfade is sized on the STATED bound, so the SL8 gate reads 1.9 px, not 1.26 px.

### D2 — The budget uses DOMAIN 02's withdrawn cost numbers

§3.1: *"Its costed table gives layers L1 to L5 a total of **180 ns to 520 ns** per column."*
`02_planet_layout.md`'s current table reads L1 40, L2 5, L3 88, L4a 240, L4b 256, L5 30 ns per column,
and states the total itself: **"416 ns without rivers, 672 ns with them"** (L1–L5 alone: 403 and 659).
COMPUTED against §9.2, which charges the "LOW end" of 180 ns × 3 844 = 0.71 ms:

| macro stack | per chunk | §9.2 total | with both levers |
|---|---|---|---|
| §3.1's low end, 180 ns | 0.71 ms | 9.60 ms | 8.38 ms |
| the neighbour's no-river total, 403 ns | 1.55 ms | **10.44 ms** | **9.22 ms** |
| the neighbour's with-river total, 659 ns | 2.53 ms | **11.42 ms** | **10.20 ms** |

The budget row is worse than the document says, and the two named levers no longer reach it.

### D3 — §9.2 mixes rungs, and the FAILED verdict is as unproven as revision 0's PASSED

The baseline is the MEASURED **rung-2** chunk (+X, (21 223, 7), 4.38 ms box). Three of the deltas
applied to it are excluded at that rung **by this document's own rules**:

- the undercut lattice (+0.60 ms): §6.2 says an undercut is cut only while its own size is at least
  `C` cells of that rung; at rung 2 the cell is 4 m, so an 8 m arch is 2 cells and is not cut;
- the channel carve (+0.82 ms): §4.6 property 3, same rule — at rung 2 only channels 64 m wide or
  more are cut, so the carve's cost falls, not rises;
- the fine floor (+0.04 ms) is rung-0 only, and is correctly left outside — but the 25 000-vertex
  extraction guess is a rung-0 rough-ground figure applied to an 8 234-vertex rung-2 chunk.

COMPUTED, the coherent rung-0 line from the MEASURED rung-0 surface chunk (1.98 ms box,
`slice_06_extractor.md:348`), with the same deltas: `1.98 − 0.12 + 0.29 + 0.71 + 0.06 + 0.08 + 0.03 +
0.82 + 0.60 + 0.04 = 4.49 ms` box. The document never prints this line. "The budget FAILS" is
therefore no better established than revision 0's "the budget passes"; the honest statement is a range
whose ends are 4.49 ms and 6.85 ms of box time, both UNMEASURED.

Also: §9 says *"the budget is 8 ms of worker time per **rung-0** chunk (ruling V10)"*. The ruling says
per chunk, and its own worst case was the rung-2 cave-dense chunk at 6.1 ms
(`owner_decisions_2026-09-07_voxels.md:441-443`). The qualifier is added by this document.

### D4 — The forbidden subtraction is used anyway, for the extraction slope

§9.1 strikes the cave unit cost because *"`2.83 − 1.98` subtracts a rung-0 chunk from a rung-2 chunk …
The two rows may not be subtracted."* The same table then states **"the extraction, per vertex above
the fixed part of 973 µs: 61 ns — COMPUTED from 1.32 ms at 5 650 and 2.39 ms at 23 084"**. Those are
the extraction halves of the **same two rows**: +X rung 0 chunk (3, 5) and +X rung 2 chunk (3, 5)
(`slice_06_extractor.md:348-349`). `(2.39 − 1.32) / (23 084 − 5 650) = 61.4 ns` is arithmetically what
the document says, and it is the forbidden subtraction. Every extraction estimate in §9.2 and §5.4
rests on it. Either the rule is wrong, or the slope is; the document must say which.

### D5 — The cell pass pays nothing for the periodic radial strata

§4.5 replaces the depth-band strata with a PERIODIC RADIAL STACK: the band index is
`floor((r − datum_m) / band_thickness) mod m`. Today `strata.at` is three integer comparisons
(`strata.rs:189-209`) and it runs **per cell**: `finish_cell` calls it at `chunk.rs:637`, and a rung-0
chunk holds **238 328 cells** (`chunk.rs:24-25`). §9.2 charges the terrace *"one radial band lookup and
one smoothstep: +0.03 ms"* — a per-COLUMN cost, over 3 844 columns. The per-CELL substance is never
charged. COMPUTED at 2 to 5 ns per cell (a subtract, a divide-or-multiply, a floor, a modulo, in `Gf`):
**0.48 to 1.19 ms per chunk**, added to a table that already fails. The soil-stripping rule (§4.5
item 4) adds a per-column slope falloff on top.

### D6 — The soil stripping makes the SUBSTANCE rung-dependent, and no bound covers it

§4.5 item 4 multiplies `topsoil_m` and `subsoil_m` by a falloff in the SLOPE. §6.1 consequence 2 says,
correctly, that *"the gradient differs by rung"*. Therefore the substance of a surface cell differs by
rung: a mesa flank is bare rock at rung 0 and soil at rung 3. §5.2's bound is stated for the HEIGHT
only, §11's SL8 row counts only the height and the undercut, and D4-9 puts the stratum id **on every
vertex** and pins it in the mesh digest — so a rung change would show a material pop on exactly the
cliffs the owner's picture is made of. Neither the bound nor the crossfade nor the measurements owe a
row for it.

### D7 — With `C = 16` the two rung rules delete the features they promise

§4.6 property 3 and §6.2 both state the rule as *"the same rule as everything else"*: a feature is cut
at a rung only while its own size is at least `C` cells of that rung, and §4.3 recommends `C = 16`.
COMPUTED:

- **A channel.** At rung 0 the cell is 1 m, so a channel must be at least 16 m wide to be cut
  anywhere. The document's own example is *"a 4 m half-width channel"* — 8 m wide — which is then
  never cut at any rung, and every headwater stream in the reference picture disappears.
- **An arch.** §6.2 says *"an 8 m arch lives at rungs 0 and 1 and is gone by rung 2"*. Under `C = 16`
  an 8 m feature is 8 cells at rung 0 and is not cut there either. The stated range needs `C = 4`
  (8 m ≥ 4 × 2 m at rung 1, and 8 m < 4 × 4 m at rung 2), which is a second, unstated constant.

### D8 — "A river that COLLIDES" and "a ditched pilot floats" are not in the code

§0 part 6 promises *"a river that COLLIDES"*, and §4.6 property 4 says making the water cells *"is what
gives a ditched pilot something to float in, because the collider reads the same cells"*.
The collider reads the extractor's triangles, and the extractor's only surface is the ROCK/AIR sign
(`extract.rs:73` `is_rock(gap) = gap < 0`; used at `:162,183,445`). A water cell has `rock_gap ≥ 0`
(`chunk.rs:633-635`), so it sits on the AIR side and emits no triangle. Nothing in the workspace
outside `vd-terrain` reads `Stratum::Water` (grep over `crates`: five hits, all inside
`terrain/src/{chunk,strata}.rs`, plus one unrelated `RlmWater` store key). No buoyancy exists anywhere.
Making the river cells is the right HR3 answer, and the CONSEQUENCE the document sells is not in the
code and is not costed as new work.

### D9 — D4-8's consequence list misses two more readers of the density byte

§6.1 carries two consequences (the cave hollow's units, the bound stated for the wrong quantity). Two
more readers exist:

- `seat_eighths` (`seat.rs:39-58`) computes a sub-metre block's seat from the ratio
  `8·|g| / (|g| + up)`. A slope-corrected byte changes the seat of every placed block on a slope. The
  "the per-column factor divides out" argument has to be MADE for the seat (its two cells share a
  column, so it does divide out) — the document makes it for the extractor only, and states nothing
  about the seat at all.
- `digest.rs:39` folds `cell.gap` into the world identity, so D4-8 is a Format-B change AND a world
  epoch on the day it lands, not only "when Format B freezes".

Also unstated: on a LATERAL edge the two cells belong to two columns with two different slope factors,
so after the correction the crossing ratio interpolates two quantities in different units. The clamp
is un-saturated (which is the gain) and a new per-edge inconsistency is introduced (which is the
cost). M4-9 must measure both, not only the distance to the true surface.

### D10 — §8's authored delta is not ruling S5-3's entry, and it puts live state inside the seed shape

§8: *"Why the pyramid, and not a new store. Ruling V9 S5-3 already makes a pyramid entry hold 'the
average change from the seed and the dominant substance' … Nothing new is invented."*
S5-3 reads (`owner_decisions_2026-09-07_voxels.md:373`): *"the far view draws the recipe's coarse hill
**plus** the change"* — the delta is added ON TOP of the generator's answer, after it. §8 puts the
authored delta INSIDE the fold, below the octaves, because *"the octaves must roughen it"*. That is a
different mechanism, and it has three costs the document does not carry:

1. `height_m` stops being a function of `(seed, address)` and becomes a function of live state, on
   both hosts. SL10 clause 1 lets the client derive the STATIC shape; the diff is clause 5 and it is
   applied to the RESULT, not fed into the recipe.
2. The world identity's golden self-check (`digest.rs`, `tag.rs:44-49`) is computed from the seed
   shape. A body with an authored delta no longer matches its own golden table, and the no-drift gate
   has nothing to compare.
3. Every column pass must query the delta store. §9 charges nothing.

### D11 — §4.8's damping is not the damping §4.1 describes

§4.4 rule 3 damps on *"the per-metre tangent gradient of §4.1"* — a POINTWISE gradient, which is what
the fold in §4.1 accumulates. §4.8's table cannot be produced that way. I reproduced it exactly, and
the damping it uses is a STATISTICAL one: with `s_i = 2π·a_i/λ_i` (which is exactly the table's own
"max slope" column: `atan(2π·600/16 384) = 12.96°` ✓ against its printed 13.0°),

```text
   a_{i+1} = a_i · gain · 1/(1 + c·G_i)      with   G_i = ½·( macro_slope² + Σ_{j≤i} s_j² )
```

Every one of the 33 amplitudes and the 33.8° RMS follow from that and from nothing else (COMPUTED).
`G_i` is a variance, not `|grad s|²` at a point. At a point where the octaves happen to align, the
real gradient is larger and the fold damps harder; on a flat patch it damps less. So the table
describes an AVERAGE column, and the document presents it as what the recipe in §4.1 produces. The
slope histogram M4-4 will not reproduce this table.

---

## WEAKNESSES

### W1 — The pixel gate pairs the wrong bound with the wrong distance

§5.2 divides `|h₀ − h_L|` by the FAR end of rung `L`'s band (`2^L × 870 m`). The jump the eye sees is
between two NEIGHBOURING rungs at the distance where the swap happens. COMPUTED on the same alpine
column and the same `Lip(T) = 1.413`: the rung-6→7 swap at `2⁶ × 870 = 55 680 m` moves the surface by
`1.413 × 59.548 = 84.1 m`, which is `1.511 mrad = 1.385 px`. Worst over the ladder: **1.39 px**, not
1.26 px. Together with D1 (the statable bound) the number the owner should be given is nearer 2 px.

### W2 — The pixel gate's denominator is an assumed residency rule

*"The client draws rung `L` out to about `2^L × 870 m`"* is stated as COMPUTED. The 870 m drawable
floor is owner-ruled (2026-09-06); the doubling per rung is not — slice 8's residency band does not
exist yet, and §5.4 hands slice 8 the byte problem that will decide it. If memory forces
`2^L × 500 m`, every pixel figure multiplies by 1.74 and the worst rung change is 2.4 px on the
document's own convention. The SL8 verdict is therefore held by a decision this document does not own.

### W3 — §5.4's bytes rest on an unstated multiplier and on lever arithmetic that does not reach 185 KB

Two numbers:

- *"about 2 000 chunks with the vertical stack"* — the shells give 1 014 surface chunk COLUMNS
  (204 + 316 + 494, each COMPUTED correctly). The ×2 is never derived. 810 MB is linear in it.
- *"the client keeps `i16` positions and `u16` indices instead of `f32` positions and `u32` indices,
  which COMPUTED takes 405 KB to 185 KB"*. M7-3's 405 228 B = `8 577 × 12` (positions) `+ 8 577 × 12`
  (normals) `+ 16 615 × 12` (indices) — I re-derived it exactly. With ONLY the two named changes:
  `8 577 × 6 + 8 577 × 12 + 16 615 × 6 = ` **254 KB**. 185 KB needs the NORMALS compressed to 4 bytes
  as well, which the sentence does not name. (`ChunkMesh` already stores `[i16; 3]` vertices,
  `extract.rs:67`, so the "i16 positions" half is a client-side choice, not a mesh change.)

### W4 — The column count is 3 844 in one place and 4 096 in another

§9.1 and §4.7 price everything per 3 844 columns (62 × 62); §6.1 prices the slope correction at
*"about 20 ns times 4 096 columns, so 82 µs"*. The sample box is 64 × 64 columns with its halo
(`lattice.rs:1-2`). Both cannot be right in one budget. If the pass really covers 4 096 columns, the
per-octave unit is 9.9 ns, not 10.5, and every macro-stack delta in §9.2 is 6.5 % low.

### W5 — The 50 km vista is not visible from the ground on this planet, and the document never says so

§5.4 sizes the vista at 50 km "to the horizon". COMPUTED on the home planet's ladder radius:
`√(2 R h) = √(2 × 3 350 759 × 1.8) = 3 473 m` — the ground horizon from an eye 1.8 m up is **3.5 km**,
27 % closer than Earth's 4 787 m. The 50 km shell is reachable only from height, or as a 14 km peak
seen from 220 km. The reference picture is a Crimson Desert vista from a high vantage; the document
should say that the home planet's small radius is itself a believability constraint on the owner's
picture, and that the chunk shells are a HIGH-VANTAGE case.

### W6 — The corner cell width is right, but not by the working shown

§4.3: *"the bend makes a cell 0.935 m wide at a cube corner (COMPUTED from `bend.rs:23-30`:
`W'(1) = K1 + 3K2 + 5K3 = 1.5584`)"*. `W'(1) = 1.5584` is correct (`bend.rs:24-30,49`), but it alone
gives 1.56, not 0.935. The 0.935 needs the metric factor as well:
`W'(1)·√(1 − 1/3)/√3 ÷ W'(0) = 1.5584 × 0.8165 / 1.7321 / 0.7854 = 0.9353`. The number is RIGHT; the
shown arithmetic does not produce it. Note also that `03_erosion_rivers.md` §4.1 states the corner AREA
density as 0.758 of the face centre, whose mean linear factor is `√0.758 = 0.871` → `C = 16` reads
18.4 samples there, because the two tangent directions at a corner are not orthogonal.

### W7 — §3.1's ownership split is not what the neighbour states

*"DOMAIN 02 owns L1 to L5; this domain owns L6 and everything below it."* `02_planet_layout.md`'s
current revision designs **L6** (change 1: the coarsest wavelength drops to ~50 km, `SHORT_WAVE_M`
drops from 30 m to **2 m**, 15 octaves, and *"the clamp arm at `body.rs:256` still never fires"*),
**L7** (the biome), the **relief ceiling** and the **per-column water surface**. DOMAIN 04 gives the
same crate a different table (11 octaves, a 16 m floor, and the clamp REMOVED, D4-15) and a different
answer to the metre scale (§4.7: "on 1 m cells the finest height feature that reads cleanly is about
8 m across"). Two documents, one `body.rs`, two incompatible octave tables, and the owner is being
asked to approve both.

---

## NOTES

### N1 — Citation slips remain, against an explicit claim that none do

The header states *"every citation in this revision was re-opened and re-read"*. Re-read:

| The document says | The code says |
|---|---|
| `frequency = radius_m ÷ wave_m` (`body.rs:168`) | `body.rs:171` (168 is a comment) |
| lacunarity 2 (`body.rs:172`) | the halving is `body.rs:177`; 172 is the octave's seed |
| the topsoil depth (`body.rs:197`, 1–4 m) | topsoil is `body.rs:196`; 197 is the subsoil, 2–8 m |
| the band gate (`chunk.rs:642`) | `chunk.rs:641` (`if rule.in_band(depth)`) |
| the fold's `greater` (`chunk.rs:647`) | `chunk.rs:646`; 647 sets `Stratum::Air` |
| `slice_05_generator.md`, `slice_06_extractor.md`, `slice_07_client_link.md` | all live under `docs/investigation/2026-09-07/`; the bare filenames do not resolve from the repository root |

### N2 — "(1 984, 3 968] for EVERY body" is false at the small end

§3.1's size paragraph. `top_rung_for` returns 0 when a face is already at most 64 × 64 chunks
(`ladder.rs:50-56`), so a body under about 1 264 m of ladder radius has `n_top = n < 1 984`. The
paragraph argues against a proposal that is already withdrawn, so nothing turns on it.

### N3 — The finest-octave amplitude is quoted as a range where a measured value exists

§0: *"the finest octave has a wavelength of 48.8 m and an amplitude between 0.24 m and 2.71 m"*. The
range is over `k_rough ∈ [0.45, 0.55)`; the home planet's own value at `k = 0.468 338 1` is
**0.397 m** (COMPUTED). The wavelength 48.8 m is the home planet's (`400 000 / 2¹³ = 48.83`), so the
sentence mixes one body's number with a range over all bodies.

### N4 — Two decisions are left implicit

- `home.rs` states the home planet as two literals and `crates/bins/tests/home_body_pin.rs` proves the
  forest draws them. A gravity input adds a THIRD literal and a third pin. Neither §12 nor §9.3 owes it.
- §5.3 says the coarse rungs get more expensive, and the bench's landed assert is only "the top rung
  cheaper than rung 0" (`slice_05_generator.md:383-386`). At DOMAIN 02's high end rung 11 costs
  `3 844 × 535 ns + 144 µs = 2.2 ms` against today's 0.266 ms — an eight-fold rise on the rung that
  draws the most chunks in a vista. §5.4 nonetheless prices the whole vista at the rung-0 client cost
  of 4.18 ms per chunk, for every rung.

---

## CHECKED, SOUND

Each of these I re-derived or re-read, and each holds:

1. **The Lipschitz derivation** (§4.5). `d(terrace)/dh = 1 − q·H·(1 − 9u² + 8u³)`, extremum at
   `u = 0.75`, so `max = 1 + 0.6875 q`: 1.275, 1.413, 1.550 at `q =` 0.4, 0.6, 0.8. Exact.
2. **§4.8's table reproduces**, every row, from `a₀ = g·step`, gain 0.62 above the 128 m break and
   0.42 below it, and the damping described in D11 — including the 33.8° RMS and the 44.3° without
   damping.
3. **§5.2's octave sums** are the correct dropped sums of that table times 1.413, and the pixel
   column's divisions are right on its own convention.
4. **The unit costs**: 40.4 µs per octave and 144 µs fixed, from 710 µs at 14 octaves and 266 µs at 3
   (`slice_05_generator.md:374-380`); 61 ns per vertex over 973 µs (subject to D4).
5. **The gravity check numbers**: 7.6 km (Earth), 20.0 km (Mars), 21.5 km (home) at 200 MPa and
   2 700 kg/m³; ratio 2.63 against the mountains' 2.49.
6. **The home planet's ladder arithmetic**: `N = 2¹² · 5 · 257 = 5 263 360`, 84 892.9 chunks per face
   edge at rung 0, 2 570 cells at rung 11, `top_rung_for` → 12 rungs. All re-derived from
   `ladder.rs:50-66,77-98`.
7. **Every withdrawal is correct against the code**: base terrain has no record
   (`topic_00_format_sitting.md:34-36`), the object byte is the tree's (B-5), `seat_eighths` reads one
   neighbour (`seat.rs:32-62`), `greater` can only open air (`chunk.rs:641-649`), the sea IS cells
   (`chunk.rs:207-213,633-635`), the extractor's crossing is a ratio (`extract.rs:182-187`), ruling A5
   says nothing about symmetry (`topic_00_format_sitting.md:23`), and the corner is a prism (ruling
   V10's landed note).
8. **The gradient-units fix** (§4.1) is right: `frequency = radius_m / wave_m` (`body.rs:171`) and
   `dir` is a unit direction, so one metre of ground moves the sample point by `1/wave_m`, and the
   radial part must be projected out.
9. **The vista's chunk shells**: 204, 316, 494 columns — all three re-derived.
10. **`n = 11` would keep the landed ladder test** (`body.rs:355-365`) once the clamp
    (`body.rs:255-256`) is removed: the counts fall 11, 10, … 1, 0 and both asserts pass. The
    arithmetic is right; only its ANCHOR is gone (B1).
