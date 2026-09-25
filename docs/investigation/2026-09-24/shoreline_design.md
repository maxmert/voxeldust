# One shoreline from one field — design v2 of believability step 4 (ruling B8 item 1)

> **DESIGN v2 OF PLAN STEP 4** (ruling B8 item 1: *"the next build step is ONE SHORELINE FROM ONE FIELD — the sea's
> shore and the lake's edge where the ground crosses the water's level, never a node square, the lake's level held by
> its dry neighbours (W6's owed law) — on today's ground"*). Nothing in this file is built. Nothing in it is measured.
> - **Written** 2026-09-25 against `HEAD` `4442fa7` (GENERATOR_VERSION 20, ARTIFACT_VERSION 8, `PROTO_MINOR` 34).
>   Another agent builds step 2b in `crates/terrain/src/{solve,lakes}.rs` at the same time. This design reads no
>   receiver byte and no discharge row. Every `path:line` is `HEAD`'s; the working tree may differ.
> - **What v2 replaces.** v1 (written the same day) let the fine octaves decide the side inside a taper. Two skeptics
>   (workflow `wf_d8fb34af-0aa`) found 23 defects, 6 of them HIGH. v2 replaces v1's body. v1's decisions D1–D12 are
>   retired or restated in §3–§5. §18 keeps v1's findings table and adds the answer to each row.
> - **Two variants.** v2 carries TWO laws for the side: **VARIANT N** (the node-scale side) and **VARIANT O** (the
>   octave side under a lateral hold). Every other part is common (§3). Slice 0 measures both in the instruments, with
>   no world change (§7). The owner chooses on the numbers and the hillshades. This file recommends N (§8).
> - **The findings' numbers.** "F1" … "F23" are the workflow's own numbers, the numbers the task text uses. v1's §18
>   table listed the same findings in severity order; §18 below prints both numbers on each row.
> - **How it was made.** One reader. The key lines were read in the code at `HEAD`. The rulings read: B6, B7, B8
>   (`docs/design/owner_decisions_2026-09-22_believability.md`); W6, W10, W15, W16, W17, W18
>   (`docs/design/owner_decisions_2026-09-21_water.md`; W9 has no section in that file, §17); T7, T9; F7, F8, F9; SL5,
>   SL6, SL10, HR5. The numbers come from the status file (`terrain_believability_status.md` §4–§8b) and
>   `step2_census_gen20.txt`.
> - **Marks.** `path:line` = read in the code at `4442fa7`. DOC = a number from a ruling or an investigation file, not
>   re-measured. ARITH = hand arithmetic. UNMEASURED = nobody printed it. PREDICTED = what this design expects a line
>   to print; a prediction is never a result.

---

## 0. The short answer

- **THE ROOT OF v1's HIGH FINDINGS.** Inside v1's taper the fine octaves decided the side. Their spread at rung 7
  (163–270 m, ARITH from the census's mean and RMS) is larger than the lakes are deep (mean 56 m; the median lake is
  ONE node, DOC). So a small lake became a cluster of ponds in the noise's pits, water crossed sills, walls stood where
  the upper hold switched off on a node line, the far rungs drew a lake at its node area while the near rungs drew a
  smaller one, and the sideways step of the shore at a swap had no bound.
- **VARIANT N — THE NODE-SCALE SIDE.** A column's side comes from ONE field at every rung: the solved rows read by the
  recipe's own Catmull-Rom weights, each one-axis pass held inside its two middle nodes (the maximum principle), on node
  words that the coast mask decides, against the water's own flat level. No fine octave decides a side at any rung.
  W6 and T7 stand as written. A pond is a smooth rounded shape at node scale; a coast is a smooth curve at node scale.
  Sub-node bays come later, from R2's 1 024 m level read by the SAME law.
- **VARIANT O — THE OCTAVE SIDE UNDER A LATERAL HOLD.** N's side, plus the ground's own live fine octaves
  (conditioned on the nodes, so no lift). An octave may flip a column only within ONE CELL OF THE COARSER RUNG of the
  coarser rung's line (F1's fix, in the law), and never into a cell with a corner beyond the basin (F16's fix). It gives
  bays and headlands up to about 2 km (ARITH) now. It amends W6's letter, so it needs the owner's word.
- **THE COMMON PARTS.** (1) ONE CONTINUOUS HOLD: the fine relief FADES to nothing at the side's line over one node
  (the whole fine part in N, the lift only in O), and W6's quarter holds the rest on its side; every term that one water
  adds is faded to zero before the column's read stencil loses that water, so nothing switches on a node line, at a
  divide or at a sill. (2) THE FAR FOLD: the side is point-sampled up to the first rung that reads the pyramid (rung 10
  on the home planet); every coarser rung takes the wet majority of those samples. So the far view draws a lake at the
  area the near view draws, and a pond folds away once and stays folded. (3) THE SHORE BAND on the wire (an SL6 ask):
  the rows' heights near every side change, so the client can draw the same side past 704 km.
- **SLICE 0** measures both variants in `shore_teeth` and `lake_census` on six stands (the belt, the coast 300 km
  inland, the plain, the reference stand, the pocket stand, and a GENTLE coast of slope ≤ 1/50): the sideways step at
  every swap, each lake's wet area at rungs 10–14, perched water, islands, pond clusters, walls on every node line where
  a term changes, G-POCKET-HELD between node centres, grid lines judged against a null (never against the drawn line
  itself), the band's size, the cost, and HILLSHADES of today, N and O side by side.
- **THE RECOMMENDATION: N.** It keeps W6 and T7 unamended, its continuity bounds hold by construction, it is the law R2
  reads unchanged, and it is cheaper. The price: until R2 a coast is a smooth 8 km curve that still leans on the grid,
  and every shore has a smooth band about one node wide where the dunes fade out. Slice 0's hillshades decide whether
  the owner accepts that until R2 (§8 names what would change the recommendation).
- **THE ORDER.** Slice 0 (measure, both variants) → slice 1 (the kernel, new entries, no world change) → slice 2 (the
  band on the wire, after the owner's SL6 word) → slice 3a (the far fold derived and checked, read by nobody) →
  slice 3b (every reader switches at once; GENERATOR_VERSION + 1; THE PICTURES) → slice 4 (the beach, the gates
  restated, DEFERRED closed). Every slice ends with its numbers, the five pictures, the status file, and HR5.
- **ELEVEN OWNER QUESTIONS** (§15). None is answered in this file. §16 lists what no owner said.

**Example.** A tarn fills one node of a pass on the belt, 20 m deep, with its rim 200 m over its level. Today a pilot
at 450 km sees an 8 km blue tile. With N she sees a round pond about 3.3 km across (ARITH), set in a smooth dish; at
704 km it is the same pond, four pixels wide; at about 1 800 km (rung 12) it folds away and does not come back. With
O the pond's edge is ragged by up to a kilometre where the dune crests cross it.

---

## 1. What the code does now (the facts this design stands on)

- **C1. The side is the nearest node's word at rungs 0–13.** `sample_side` rounds the column's fraction to the nearest
  fine node and reads its mask word (`crates/terrain/src/artifact.rs:1337-1362`). The footprint level is zero while a
  cell is no wider than a node (`coast_level`, `:502-511`; `footprint_level`, `:1367-1370`). From rung 14 the side is
  the wet majority of the footprint (`CoastCounts::side`, `:465-480`), and a parent's count is the sum of its children
  (`CoastCounts::of`, `:384-415`).
- **C2. The mask word is one of four** (`Side`, `:251-261`): land 0, sea 1, pocket 2 (a dry node under the sea's
  level, `side_word`, `:319-322`), lake 3. A lake node is a node whose water stands STRICTLY over it
  (`water_z[i] > z[i]`, `crates/terrain/src/solve.rs:1486`). A spilling lake's level is its spill node's height
  (`crates/terrain/src/lakes.rs:496`, `:587`), so the sill node is a LAND node at exactly the level.
- **C3. One reader names a column's water and its hold** (`column_water`, `artifact.rs:1209-1255`): the sea for a sea
  cell, the row's own level for a lake cell, and for a land cell the sea's hold where a sea node stands in its read
  stencil (`stencil_holds_sea`, `:1271-1303`), else INLAND (`:1231-1234`). A land column beside a lake is never held
  over the lake.
- **C4. The shore law has a lower bound only** (`shore`, `crates/recipe/src/height.rs:412-434`, `SHORE_SHIFT` 2 at
  `:336`). It holds the surface on the stated side by at least `|base − water| >> 2`, and `base` is the node-scale
  ground `radius + z + coarse` (`crates/recipe/src/plan.rs:235-236`). `SIDE_INLAND` under the water returns the
  surface unchanged (`height.rs:417-419`).
- **C5. The fine relief carries a lift.** A ridged octave's mean stands over zero (`crates/terrain/src/body.rs:132`).
  At rung 7 the fine relief's mean / RMS / RMS slope read +287 m / 330 m / 0.103 on the belt, +283 m / 334 m / 0.109 on
  the coast, +325 m / 423 m / 0.218 on the plain (DOC `step2_census_gen20.txt`, the `G-FINE-*` lines). So a dry column
  at a node square's edge stands hundreds of metres over the water: the wall of the teeth (DOC 1 396 m median on the
  coast stand at rung 7, status §6).
- **C6. The lakes are shallow and small.** 67 233 lake nodes, 3.883 × 10⁶ km², 2.1873 × 10⁵ km³ (DOC): a mean depth of
  56 m (ARITH). 3 883 patches, median ONE node; 2 084 patches are one node; 111 patches of 64 nodes or more hold about
  79 % of the lake nodes (ARITH from the census's size classes).
- **C7. The Catmull-Rom read passes through its nodes and can overshoot between them.** At a fraction of zero the
  weights are `(0, 1, 0, 0)` (`crates/recipe/src/macro_field.rs:29-40`); at a half they are
  `(−1/16, 9/16, 9/16, −1/16)` (the recipe's own test, `:80-90`). So land nodes at `W+100, W+1, W+1, W+100` read
  `W − 11.4 m` between the two middle nodes (ARITH, F11's own example): land nodes alone can put the read under the
  water.
- **C8. No column stands on a node centre below rung 13 on the home planet.** A rung-`r` column centre is at
  `2^r·(2i+1)` half-cells and a node centre at `8 192·(2c+1)` (`node_and_fraction`, `artifact.rs:1808-1815`). On a body
  whose node is not a power of two in cells (the code's own 714-cell node, `crates/terrain/src/macro_lattice.rs:79-82`),
  no rung's cell is one node: the only rung whose columns can stand on node centres is set by the node's largest
  power-of-two factor (rung 1 for 714 = 2 · 357, ARITH). So "rung 13 = one node" has no counterpart there (F10).
- **C9. Rungs 0–9 read the rows; the first rung that reads a pyramid level is where a chunk (62 cells,
  `crates/terrain/src/chunk.rs:68`) spans four nodes** (`PyramidField::level_for`, `artifact.rs:844-867`). On the home
  planet that is rung 10, and rungs 10–14 read level 1. This file calls that rung **R₁**. It is a rule of every body,
  not a number of the home planet.
- **C10. The ground drops its fine octaves by one rule** (T7 rule 1, `octave_survives`, `body.rs:226`): at 48.83 cells a
  wave (a crest at half that). Rung 7 keeps 4 fine octaves, rung 9 keeps 2, rung 10 keeps none (DOC status §5 D8). The
  code states that nothing else selects an octave (`body.rs:156`) and that the world never reads
  `NYQUIST_CELLS_PER_WAVE` (`body.rs:207-213`).
- **C11. The chunk holds the nodes two before and three after every column's own node** (`nodes_of_chunk`,
  `artifact.rs:1413-1424`), because the slope share reads the field one node either way (`slope_share`,
  `:1976-2003`). So a column can read a 6 × 6 block of nodes with no new tile.
- **C12. The water sheet takes a vertex's level from its columns' standing water** (`water_sheet_sided`,
  `crates/terrain/src/position.rs:384-498`: the highest word among the group's four columns, `:406-440`; a triangle
  at the highest of its three vertices, `:443-455`). The water's morph target reads the FINER rung's side and level
  (`crates/client/src/chunks.rs:1418-1428`).
- **C13. The card holds no artifact.** Its column pass reads `FieldRead::none` and `SIDE_UNKNOWN`
  (`plan.rs:127-132`, `:161-175`). `gpu-drift` compares the card and the CPU on seed-only boxes (DOC 5 979 of 5 979).
- **C14. The instruments read the shipped column through public parts.** `RungReader::column`
  (`crates/bins/src/shore_geometry.rs:204-270`) re-runs the kernel's last lines and keeps the base (`:251`). The
  `image` crate is already a dependency of `vd-bins` (`crates/bins/Cargo.toml:123`), so a hillshade needs no new
  library.

**Example.** On the plain stand forty small lakes stand as blue tiles (DOC step-2 pictures). Each tile is one node
square: its side comes from one node (C1), its dry neighbour is held over the sea and not over the lake (C3), and the
lift stands on the dry side (C5). So each lake is a flat square pit with a wall on each node line.

---

## 2. The constraints any v2 law must meet

v1 failed because each constraint below was met at node centres only, or by a fallback, or by an argument. v2 meets
each one BY CONSTRUCTION between node centres, and slice 0 has a line that can fail for each one.

| # | the constraint | its law | v1's failure (§18) |
|---|---|---|---|
| K1 | A swap moves the drawn shore sideways by at most one cell of the coarser rung. | T7's line (W17.1: p99 ≤ 1 px, max ≤ 2 px, one pixel being half a coarser cell at the swap), W16 | F1, F17 |
| K2 | A water is never drawn past its basin: not past a sill, not into a pocket, not as a perched pond — between node centres too. | B7, W11, W16 | F2, F16 |
| K3 | No wall: the surface is continuous where the side changes, where a water enters or leaves the read stencil, at a divide and at a sill. | W6, W16 | F3, F15, F18 |
| K4 | The far view draws what the near view draws: a lake's area agrees across every swap, and water never appears as the pilot climbs. | W16 items 2–3 | F4, F21 |
| K5 | A node centre reads its own mask word, every tie included. | W10, W15 | F8 |
| K6 | One rule for every body; no "rung 13 = one node". | SL5 | F10 |
| K7 | No new tuned number. A bound is a law, the read's own reach, or a stated calibration the owner rules. | T9 | F11, F18 |
| K8 | One integer source for the server, the client and the card; no loop shape the card's ledger records as faulty. | SL10, F7, F8 | F22 |
| K9 | No owner word is recorded as given when it is not. | B6 item 1 | F6, F13, F23 |

**Example.** A pilot climbs from the belt's coast to 3 000 km. K1 says the coast never jumps sideways by more than one
cell of the ring that takes over. K4 says a bay she saw at 900 km is still there at 150 km. K3 says she never sees a
straight cliff along a node line. K2 says no pond appears in a hollow the budget never filled.

---

## 3. The common parts (both variants)

### D1. The candidate waters and their flat levels

A WATER is the sea, or one lake level (two lake nodes of one level are one water). Its level `W_b` is ONE flat word:
the body's sea, or the row's own `water_m`. The water's level is continuous in space inside the water. It is never
interpolated between two waters: a blend names a level no water stands at (the rule `sample_row`'s doc already states,
`artifact.rs:1126-1130`). A column's CANDIDATES are the waters that have a wet node within two nodes of one of its
cell's four corner nodes: the 6 × 6 block the chunk already holds (C11). The kernel carries at most FOUR candidates; a
column with more is a FAILING census line (`CHECK-SHORE-CANDIDATES-OVER-4`, band 0), never a silent drop (F11).

### D2. The node-scale side field `S_b` — the maximum principle

For each candidate `b`, the node words of `b`'s side field are the rows' heights, decided by the MASK WORD (K5, F8):

```text
ẑ^b_n = min(z_n, W_b − 1 m)            n is a wet node of b (its mask word is b's water)
      = W_b + max(1 m, W_b − z_n)       n is not b's and z_n ≤ W_b  (a sill at the level, an outlet, another
                                        basin's floor, a pocket's floor, another water's node): THE MIRROR
      = z_n                              n stands over W_b
S_b(x) = CRh(ẑ^b)(x) − W_b              wet for b where S_b < 0
```

- **The mirror.** A node beyond `b`'s basin reads dry by at least a metre (the rows' own unit) and by as much as it is
  deep. So `|S_b|` measures how far the ground is from `b`'s level on BOTH sides of a divide, and a deep pocket reads
  far over the sea in the sea's field. v1 floored such a node to `W + 1 m`; that made `|S|` small beside every pocket
  and every outlet, which is where v1's hold pinched.
- **The held Catmull-Rom `CRh`.** The recipe's own weights (`macro_field.rs:29-40`), with each one-axis pass held
  inside its own two middle words: `r_k = clamp(Σ w(ta)·p_k, min(p_k1, p_k2), max(p_k1, p_k2))` for each of the four
  rows, then the same along the other axis over `r_0 … r_3`. On a cell edge the value depends only on that edge's own
  nodes and their neighbours along it, so `CRh` is continuous across every node line (§10 test 2 checks it). And
  it keeps THE MAXIMUM PRINCIPLE: the value inside a cell lies between the least and the greatest of its four corner
  words.
- **Why the maximum principle matters (K2).** A cell whose four corners are all dry for `b` (over the level, or
  mirrored) reads dry at EVERY point, not only at its centre. So `b`'s wet set lies inside the cells that touch a wet
  node of `b`; no pond stands between two land nodes (C7's overshoot), and no water crosses into a cell past a sill or
  into a pocket. A cell whose four corners are all `b`'s reads wet at every point: no islet inside a lake where no node
  says land.
- **Why not the plain Catmull-Rom.** C7: land nodes alone read `W − 11.4 m` between them. That happens between two
  nodes one metre over the level (two sill nodes, a sill and a low rim), so the plain read floods past a sill (F2).
  The plain read stays the GROUND's read (`sample_z`, `artifact.rs:1862-1881`); `CRh` is the SIDE's read. A slice-0
  control prints the plain read's perched count.
- **The pure path.** Where a cell's four corners carry one word, the side is that word and no read is needed. That is
  an exact shortcut of the same law, not a second rule.

### D3. The presence `Ω_b` — every term of a water ends inside the stencil

`ω_b(n) = 1` where a wet node of `b` stands within two nodes of `n` (through the seam table), else 0. `Ω_b(x)` is the
bilinear read of `ω_b` at the cell's four corners. Properties (§10 test 3):
- `Ω_b = 1` at every point within ONE NODE of `b`'s side line: the line lies in cells that touch a wet node of `b`
  (D2), and every point within one node of such a cell stands in a cell whose four corners are within two nodes of
  that wet node.
- `Ω_b > 0` only in a cell with a corner within two nodes of a wet node of `b`, and every such cell's 6 × 6 block holds
  that node. So every term that `b` adds is zero BEFORE the column's read loses `b`, and no term switches on a node
  line (K3, F15).
- `Ω` reads mask words only. It is available at every rung on both hosts.

`Ω` is node-aligned. It is NOT a taper of the side (v1's D4 was): it never decides a side. It only ends a water's hold
terms. Inside the region where it is 1 it changes nothing. It can show only where the distance estimate of D4 errs by
more than a node; the WALL lines of §7 read that.

### D4. The fade — the fine relief ends at the side's line

For each candidate `b`:

```text
d_b  = |S_b| / max(|∇S_b|, 1 m per node)              in nodes (first-order distance to b's side line)
φ_b  = 1 − Ω_b · (1 − fade(min(1, d_b)))              fade = the recipe's own quintic (noise.rs:75-81)
φ    = the least φ_b over the column's candidates
```

- `∇S_b` is the Catmull-Rom's own derivative weights over the same sixteen words (unheld: the held read's slope is
  zero where it holds). The floor of one metre per node is the rows' own unit: the rows cannot state a smaller slope.
- The reach is ONE NODE because that is the widest reach the read can carry: `Ω_b = 1` is guaranteed only within one
  node of the line (D3). It is derived from the reads, not typed (K7). It is the body's node, not "8 km" (K6).
- What the fade multiplies is variant-specific: in N the WHOLE fine part (§4); in O the LIFT only (§5).
- The fade adds at most `1.875 · |f| / node` to the slope (the quintic's steepest slope is 15/8): about 0.07–0.10 on
  the three stands (ARITH from C5). No bank rises at four times anything (F18).

### D5. The one continuous hold

After the fade the surface is `h' = terrace(B + …)` (variant-specific, §4, §5), with `B = radius + CR(z) + coarse` the
rung's own node-scale ground: the rows at rungs under R₁, the pyramid level from R₁ up (as today). Then:

```text
WET (the column's water b is the LOWEST water it is wet for):
    s = W_b − max(W_b − h', k · |S_b|)                                  (W6's quarter, k = 1/4)
DRY:
    s = max(h', LB),   LB = max( LB_sea , max over lake candidates b of LB_b )
    λ(g) = k · g   for g ≥ 0        λ(g) = K · g   for g < 0            (K = 1/k = 4)
    LB_sea = sea + λ(B − sea)                                          (every column, every rung: no stencil)
    LB_b   = W_b + λ(B − W_b) − (1 − Ω_b) · A                          (A: see below)
```

A body with no sea has no `LB_sea` (today's rule: a column with no water is untouched, `height.rs:413-415`).

- **`A` is the body's own bound** on how far the fine part and the bench can move the surface at the rung: the live
  fine amplitudes at full roughness (times the read's norm, 1.5625, for O's lift term), plus the bench's reach
  (`relief_bound`'s parts, `body.rs:1522`). It is a charter fact, not a tuned number.
- **Why it is continuous (K3).** `λ` is continuous (it is zero at zero from both sides), so every `LB` term is
  continuous in `B`; a max of continuous terms is continuous. `λ(g) ≤ g` for every `g`, so `LB ≤ B`, and at `Ω_b = 0`
  the term stands under `B − A ≤ h'`: a lake's term is inert before the stencil loses the lake. At the side's line the
  fade makes `h' → B ≈ W`, the wet bound `→ W`, and `LB → W`: both sides meet the water.
- **Why no wall at a divide or a sill (F3).** A dry column BEYOND a lake's divide stands physically under `W_b`. Its
  term `W_b + K·(B − W_b)` falls four times as fast as the ground, so it never lifts the column toward the lake's level
  (no dam), and it is continuous where `B` crosses `W_b` (no switch of "which water holds me"). v1 switched the held
  water there and pinned the upper bound near it.
- **Why no mesa in a pocket (K2).** For `B < sea`, `LB_sea = sea − K·depth ≤ B − 3·depth`: the law never lifts a
  pocket's column over the sea (G-POCKET-HELD 0 by construction, between node centres too), and it never lifts it over
  its own node-scale ground.
- **Why no stencil.** Today a land column is held over the sea only where a sea node stands in its read stencil, and
  the stencil widens with the pyramid level (C3). That is the mechanism of G-RUNG-HOLD (DOC 5 columns at 9 → 10,
  1.43 px). `LB_sea` holds every column by its own ground and reads no stencil, so the word cannot change at a swap.
- **The two stated words.** `k` is W6's quarter (owed, W6). `K` is new: how deep the fine octaves may dig a DRY column
  that stands physically under a water — at most `K − 1` times its own node-scale depth. Continuity needs only `K ≥ 1`;
  `K = 1/k` is the recommendation. `K` touches no visible bank: it acts only beyond divides and in pockets. §15 asks both
  words, apart (F18).
- **At rungs from R₁ up** there is no fine octave (C10), so there is nothing to fade; the wet magnitude is `|S_b|`
  where the tiles or the band hold the rows and `|B − W_b|` in a pure wet stencil (as today). A dry dip under the sea's
  level can stand beside a steep coast there (F7's case); the side still says land, and the dip is far under one pixel
  at 704 km and more. The restated G-RUNG-HOLD reads it in pixels (§7).
- **What is retired.** v1's upper bound `4g` (D6) and its taper (D4). Today's `shore()` stays byte for byte for its
  callers until slice 3b (F12); the new hold is a new entry.

### D6. The sampling rule and THE FAR FOLD (K4, K6)

- **Rungs 0 … R₁ point-sample the side** at the column's centre, by the variant's law. All these rungs read one field
  (N) or one field plus the octaves the rung draws (O).
- **Every rung coarser than R₁ takes the wet majority of its footprint's rung-R₁ samples** — W15's own rule, with the
  rung-R₁ sample in place of the fine node. Rung cells always nest (a rung-`r+1` cell is exactly four rung-`r` cells) on
  every body, so the fold is exact on a body whose node is not a power of two (K6, F10).
- **The counts.** A node whose 3 × 3 block holds one mask word contributes all its samples with that word. A MIXED node
  (its 3 × 3 holds two words) contributes its own samples' counts (wet, lake, pocket; a pocket sample is dry with
  `B < sea`). On the home planet a node holds 8 × 8 rung-10 samples, so a node's count is 0–64 (ARITH), and
  `CoastCounts` folds these per-node counts exactly as it folds bits today: a parent's count is the sum of its children
  (W15.1 item 2), `u32` still holds the top level (4¹¹ · 64 = 2²⁸). Rungs whose cell is under a node (11, 12 on the
  home planet) fold the node's own samples directly. On a body whose node is not a power of two, a node's count is the
  samples whose centres lie in its square, and a cell of a node or wider reads the node counts by today's footprint rule
  rounded outward (`coast_level`, `artifact.rs:490-498`).
- **Why this answers F4 and F21.** A parent cell is wet only if some child is wet (a majority of four ≥ 2 ⇒ one child
  holds ≥ half), so water never APPEARS as the pilot climbs. A pond that is dry at rung `r` stays dry at every coarser
  rung (a pond in at most four cells, each under half, sums under two cells). The far view draws the lake at the area
  the near rungs draw, never at its node area. Rung 13 is no longer "the node word".
- **The far view's water level** stays today's: the block's highest lake level, else the sea (`block_water`,
  `artifact.rs:1606-1614`). Two lakes in one block keep today's 0.016 px step (DOC; §15 Q9).

### D7. The standing water and the sheet (F9, F5)

- A DRY column's standing water (`ColumnWater::water`, `artifact.rs:1156-1165`) stays today's land rule: the sea where
  a sea node stands in its read stencil, else none. It is NEVER a lake's level. The hold's waters (D5) and the standing
  water are two different words, as `hold` and `water` already are.
- The sheet takes a vertex's level from its WET columns only; a vertex with no wet column takes the sea where one of its
  columns stands the sea, else nothing. So the sea never climbs to a lake's level beside a barrier (F9). A new line
  counts sea vertices drawn over the sea's level (band 0).
- **The morph target (F5).** A water vertex's target reads the COARSER rung's side and level through the ONE reader at
  the rung-`r+1` cell the vertex falls in, at the vertex's moved radius (`water_radius`, `position.rs:361-379`). So a
  column that changes side at a swap fades with its ground. `water_edge_step` measures the flips at every swap.

### D8. The shore band on the wire (an SL6 ask, §15 Q2)

- **What.** The rows' height `z_m` of every fine node within TWO nodes (through the seam table) of a mixed node, and
  `water_m` for every lake node among them. Two is the reach of the rung-R₁ samples' own stencils: a sample in a mixed
  node's square reads nodes at most two away.
- **Derived on both hosts from the mask**, in node order, so no index crosses. The receiver checks the assembled size
  against the generator's own count (the W16.2 lesson). A far chunk that needs a band node waits for it; the coarser
  rung stands meanwhile (F9).
- **The size.** UNMEASURED. ARITH: if a twentieth to a fifth of the 8 871 936 nodes stand in the band, 0.9–3.5 MB of
  heights plus at most 0.13 MB of lake levels (67 233 × 2 B). The mask is 2.2 MB (DOC W16.2). Slice 0 prints
  `BAND-NODES` and `BAND-BYTES`.
- **Why the client cannot compute it.** The rows are the solve's output; the solve runs only on the shard.
- **What doing without costs.** From R₁ up the side would stay the node word: the 704 km swap would change a smooth
  pond into a node square (up to half a node, about 5 px, ARITH v1 §9) — a W16 breach — and the far fold of D6 could not
  be built. The other road is the tiles out to where half a node is one pixel, about 3.7 Mm (ARITH, v1).
- **The shape.** v1's arm: `BulkMsg::ArtifactShore { realm, part, parts, bytes }`, parts of at most 32 KiB, a
  `shore_parts` count in `ArtifactHead`, `PROTO_MINOR` 34 → 35 (R2 also plans 35; whichever lands first takes it). The
  gateway caches the parts as it caches the mask. No new artifact product and no digest word: the band is a schedule of
  rows that already cross in the tiles.
- **The same band serves both variants**: O's octaves at R₁ are none on the home planet (C10), and on another body the
  client computes them from the seed.

### D9. The beach

A dry column is a beach where its hold water is the sea and its node-scale ground stands within the storm band over it
(`coast_band`, `solve.rs:1591`), read at the column, not at the nearest node (today a node square,
`crates/terrain/src/chunk.rs:412-417`). No pixel shows it until 8e (status §6 T8). §15 Q11.

**Example.** A pilot lands on a lake shore in the belt's foothills. The lake stands at 1 200 m; its node is 40 m under
it, the rim nodes 60 m over it. The side's line runs where the held read of those nodes crosses 1 200 m (D2). Within a
node of that line the dunes shrink to nothing (D4), so the ground under her boots rises gently out of the water. Her
dry rim is held over 1 200 m, never over the sea 3 km below (D5). Her boots, the collider and the mesh read one column.

---

## 4. VARIANT N — the node-scale side

### 4.1 The law

```text
side(x)  = wet for the LOWEST candidate b with S_b(x) < 0; else dry            (every rung up to R₁; the fold above)
h'(x)    = terrace(B + φ · f)                                                  (f = the column's fine part, parts.fine)
surface  = the one continuous hold of D5
```

No fine octave enters the side at any rung. The side at rung 0 and the side at rung R₁ read the same field at two
sample spacings. W6 holds in its own words ("the fine octaves may not carry a column across its water"), and T7's one
selector stays the only selector (F6).

### 4.2 The static shape at every rung (home planet; every rung by its rule on another body)

| rungs | cell (face middle, DOC) | the drawn ground | the side | the water's level | the hold |
|---|---|---|---|---|---|
| 0–9 | 0.95–485 m | the rows (tiles), CR, the live fine octaves | point sample of `S_b` (tiles) | the sea; the lowest wet lake's own level | fade (whole fine part) + D5 |
| 10 = R₁ | 971 m | level 1, no fine octave | point sample of `S_b` (the band; the node word in a pure 3 × 3) | as above | D5, no fade |
| 11–12 | 1.9–3.9 km | level 1 | majority of the cell's rung-10 samples | as above; the block's water in a pure lake | D5 |
| 13–18 | 7.8–250 km | levels 1–5 | majority of the footprint's samples (per-node counts in `CoastCounts`) | today's block water or the sea | D5 |

- **The sea.** The coast is the zero line of `CRh(ẑ^sea) − sea`: a smooth curve at node scale. It follows the solve,
  so it keeps the solve's own lean on the grid (the base's own waterline reads G-TEETH-GRID-GROUND 2.3–3.1 at rungs
  5–13, DOC). A pocket is mirrored: the sea never enters it.
- **A lake.** Its edge is the zero line of `CRh(ẑ^b) − W_b` inside the cells that touch its wet nodes. A one-node lake
  is a rounded pond at its node's centre. ARITH (a symmetric pit, depth `d`, rim `r`, the tensor read of one node):
  `d = 20 m, r = 200 m` → about 3.3 km across (8.7 km², 0.13 of the node); `d = 56 m, r = 100 m` → about 7.2 km across;
  a deep lake with a low rim reaches past the node's half and turns slightly square (the tensor read's own shape; the
  square test of §7 reads it).

### 4.3 The lake's extent between node centres (F2)

By the maximum principle (D2) the wet set of `b` lies inside the cells that touch a wet node of `b`, and a cell with no
such corner is dry at every point. So:
- the channel past a sill is dry at every column, not only at the node centre: the sill node reads `W + 1 m`, the
  outlet node `W + its depth`, and the cell between them has no wet corner;
- a trough beyond the rim under the lake's level is dry at every column;
- a pocket beside the sea is dry at every column (mirrored in the sea's field);
- no column inside a cell whose four corners are the lake's is dry: no islet where no node says land;
- the lake stops just before its sill (the held read crosses `W` between the last lake node and the sill's `W + 1 m`).

What N does NOT do: it draws no sub-node bay, inlet or islet (T3). A sub-node hollow under the level inside the lake's
cells fills with the lake; one outside them stays dry.

### 4.4 The hold (F3, F15, F18)

D5 with `h' = terrace(B + φ·f)`. At the side's line `φ = 0`, so the drawn surface is the node-scale ground there and
meets the water on both sides. Within a node of the line the dunes shrink smoothly to nothing (the quintic fade). No
upper bound exists, so no bound switches off on a node line (F15) and no bank rises at four times anything (F18).

### 4.5 The far rungs' lake area (F4, F21)

D6. The near rungs draw the pond at the held read's area; rung 10 samples it every kilometre; every coarser rung folds
those samples. The 3.3 km pond of §4.2 (about 9 rung-10 samples, ARITH) is wet at rung 11 (majority of four in at least
one block) and folds away where no block of its footprint holds half its samples — at rung 12 (ARITH) — and stays folded.
The per-lake lines of §7 read each lake's area at rungs 7, 9, 10, 11, 12, 13 and 14.

### 4.6 The swap bound as a sideways step (F1, F17)

Both rungs of a swap point-sample ONE field. The drawn edge of a rung lies within half its own cell of the field's
line. So the sideways step at `r → r+1` is at most `½·cell(r) + ½·cell(r+1) = ¾` of a coarser cell, on every coast,
at every slope, for every swap from 0 → 1 to R₁ − 1 → R₁. From R₁ up the fold moves the edge by at most one coarser
cell (a majority of four nested cells). No octave height enters the step, so the gentle coast (slope ≤ 1/50) is no
worse than a steep one. PREDICTED: G-SHORE-SWAP GREEN at every swap on both coasts, except where a sub-cell pond folds
(read by the fold lines, not by the step).

### 4.7 The pocket

Mirrored in the sea's field (D2) and never lifted over the sea (D5). G-POCKET-HELD is 0 by construction at every rung
and between node centres. The fade keeps a deep pocket's relief (its `|S_sea|` is its depth, so `d_sea` is large).

### 4.8 The cost (UNMEASURED; slice 0 benches the what-if, slice 1 the kernel)

- A column with no candidate within two nodes: today's arithmetic plus one purity test per chunk (the chunk's node block
  is one word ⇒ every column takes the pure path; F22's per-chunk choice).
- A column near water: per candidate one held read (16 words), one derivative read (32 words), four presence words,
  one reciprocal and one quintic; then one multiply for the fade and the hold's max. ARITH: +100–200 ns a column near
  water, against a column pass whose own cost this file does not re-measure (v1 used about 450 ns at rung 7, not
  re-measured). With a tenth to a third of a coastal frame near water: +3–10 % of the column pass (ARITH).
- The client's one-time far fold: every mixed node × 64 samples × one or two held reads: ARITH 1–6 s on one core at the
  band's arrival, on the terrain workers' share (F6 of the frozen-patch rulings), while the coarser rung stands (F9).

### 4.9 What a pilot sees with N until R2

From a hull at 150 km over the belt's coast: no staircase, no 90° turns, no wall. The coast is a smooth curve with
bends of about a node and more; it still leans on the grid where the solve does. Along every coast and around every lake
a smooth band about 8 km wide where the dunes fade into a gentle coastal plain. The small lakes are round ponds 2–8 km
across, each in a smooth dish; the chains of lakes along a grid line are smooth ribbons. At 704 km and past it the same
outlines, folding away one by one as they fall under half a cell. No pond looks dune-shaped: B8 accepted a dune-shaped
edge; N does not draw one, and the owner judges that on the hillshade.

### 4.10 What R2 changes for N

The law's words do not change: "the finest held lattice, read by the recipe's own held Catmull-Rom, decided by its own
side words, against the water's flat level". Under R2 the finest lattice at rungs 0–10 is A0 (1 024 m,
`r2_design.md` §0), so the side gets 1–2 km bays and rias. A0 needs its OWN side words at 1 024 m (a drowned valley
through a land node is sea): that is R2's own finding 2 ("one fact says where the water is"), and this law is that fact.
The far fold then folds A0's rung-10 samples; §13 prices the wire for it.

**Example.** The belt's coast from a hull at 150 km, with N: a smooth coastline bending every 8–20 km, the dunes
shrinking to a flat coastal strip about 8 km wide, and no step anywhere as the rings sweep under the hull.

---

## 5. VARIANT O — the octave side under a lateral hold

### 5.1 The law

```text
G_s(x)  = W_b + S_b(x) + m(x) · [ Σ_s(x) − CR(Σ_s(n))(x) ]         the side field at rung s: N's field plus the
                                                                    ground's OWN live fine octaves at rung s (T7's one
                                                                    selector), conditioned on the nodes (no lift)
side_R₁ = N's side                                                  (no fine octave lives at R₁ on the home planet)
for s = R₁ − 1 down to the column's rung r, for each candidate b (the lowest wet water wins, as in N):
    side_s(x) = sign(G_s(x) − W_b)  where  |G_{s+1}(x) − W_b| < (cell(s+1) / node) · max(|∇S_b(x)|, 1 m per node)
                                    and a DRY → WET flip stands in a cell with a wet corner of b and no mirrored corner
              = side_{s+1}(x)       elsewhere
h'(x)   = terrace(B + f − (1 − φ) · L),   L = m(x) · CR(Σ_r(n))(x)   (the fade takes the LIFT only)
```

- **The lateral hold (F1's fix, in the law).** An octave may flip a column only where the column stands within one
  cell of the coarser rung of that rung's line, measured laterally by the node-scale slope. The hold telescopes: each
  rung may depart from the next coarser rung by one of that rung's cells, so the reach at rung 7 is at most
  `cell(8) + cell(9) + cell(10) ≈ 1.8 km` from N's line (ARITH), and at rung 0 about 2 km.
- **The basin bound (F16's fix).** A dry → wet flip is refused in a cell with a MIRRORED corner (a node beyond the
  basin), and a wet set can exist only in cells that touch a wet node of its water (N's own set, plus flips in those
  cells). So no pond stands past a sill or in another basin. A pond inside the lateral band but cut off from the lake by a
  dry strip is still possible: the PERCHED line reads it (band 0).
- **The octaves are the ground's own** (T7's one selector, C10), so no second selector exists (F6). The side's residual
  is the drawn ground's own residual, so the drawn ground crosses the water where the side does.

### 5.2 The static shape at every rung

| rungs | the side | what O adds to N |
|---|---|---|
| 0–9 | the telescoped lateral hold over the live octaves | bays, headlands, points and ponds up to about 2 km from N's line; the edge follows the dune crests and pits |
| 10 = R₁ | N's side (no octave lives) | nothing |
| 11–18 | the fold of rung-10 samples | nothing: the far view is N's |

### 5.3 The lake's extent between node centres (F2)

N's extent (§4.3), widened by at most the lateral reach inside the cells that touch the lake's wet nodes and have no
mirrored corner. The outlet, the trough and the pocket stay dry at every column. Islands inside all-lake cells are
possible within one coarser cell of N's line (the ISLAND line reads them).

### 5.4 The hold (F3, F15, F18)

D5 with `h' = terrace(B + f − (1 − φ)·L)`. The residual texture stays at the shore; only the lift fades. At O's line
the drawn ground stands at `W + φ·L`, and `φ` is measured from N's line (so it is continuous). Where O's line departs
from N's by the full reach, `φ ≈ fade(0.22) ≈ 0.07`: a step of about 20 m on the plain's lift (ARITH). PREDICTED
G-TEETH-WALL near 2–3 × the slope's own step.

### 5.5 The far rungs' lake area (F4, F21)

N's (§4.5): O adds nothing at R₁ and up, so its far view is N's. The near area is N's plus the octave ragging; the
fold sees only rung 10.

### 5.6 The swap bound as a sideways step (F1, F17)

The flips of rung `r` lie within one coarser cell of rung `r+1`'s line (first-order). With the two samplings the
worst sideways step is `1 + ¾ = 1¾` coarser cells (ARITH), over T7's max line of one cell; the p99 is PREDICTED under
half a cell because most flips are small. If the gentle coast's max reads over one cell, the hold must shrink to a
quarter of a coarser cell, and O's detail shrinks to about 450 m at rung 7 (ARITH). This is O's main risk and slice 0's
main test of O.

### 5.7 The pocket

N's: every cell that touches a pocket node has a mirrored corner, so no flip floods it. G-POCKET-HELD 0 by construction.

### 5.8 The cost (UNMEASURED)

N's cost, plus: the octave loop keeps one snapshot accumulator per distinct live count between the rung and R₁ (four at
rung 7, nine at rung 0, all ADDITIVE, masks in the data — the only shape the card's ledger accepts,
`height.rs:195-203`); per snapshot one 16-word read of the node sums (computed once a chunk for its node block) and one
compare. ARITH: +150–400 ns more per column near water, so +10–30 % of a coastal frame's column pass.

### 5.9 What it amends

W6's letter ("the fine octaves may not carry a column across its water"). O lets them, within one coarser cell of the
coarser line. That is §15 Q10. T7 is not amended.

### 5.10 What a pilot sees with O until R2

From a hull at 150 km: N's smooth curve with a ragged edge — points, coves and small ponds up to 1–2 km off it where
the dune crests and pits cross the level; the dunes reach the water; no smooth coastal band, but a gentle lowering of
the ground toward the shore (the lift fades). From 704 km on, N's smooth outline.

### 5.11 What R2 changes for O

R2 retires the ridged kind under a field and keeps only a smooth skin under 2 km (`r2_design.md` §7.4). O's octave
term then carries a few metres, so O becomes N-on-A0. The octave machinery of O is thrown away at R2.

**Example.** A tarn in a pass seen from 450 km, with O: rung 9 draws its edge bent by the 12.5 km and 25 km octaves
within one rung-10 cell of N's pond; at 704 km rung 10 draws N's pond, and the edge moves by less than one kilometre.

---

## 6. The two variants side by side

| | N | O |
|---|---|---|
| The side reads | the node-scale field only | the node-scale field + the ground's live octaves, laterally held |
| W6 / T7 | both as written | W6 amended (Q10); T7 as written |
| Sub-node shore detail now | none | up to about 2 km |
| The shore's look | smooth curves, a smooth band ~one node wide | ragged edges following the dunes |
| Sideways swap step, worst (ARITH) | ¾ coarser cell | 1¾ coarser cells (hold of one cell) |
| Perched water, islands | none by construction | possible inside the lateral band |
| The far view | the fold of rung-R₁ samples | the same (N's) |
| The wire | the band (D8) | the same band |
| Extra cost near water (ARITH) | +100–200 ns a column | +250–600 ns a column |
| Under R2 | the same law on A0 | becomes N on A0; its octave machinery is thrown away |

**Example.** The owner opens two hillshades of the plain side by side: in N forty round ponds in smooth dishes; in O
forty ponds with ragged rims, a few with a pond beside them.

---

## 7. Slice 0 — a MEASURED what-if of both variants (no world change)

### 7.1 What is built

- **The what-if** lives in `crates/bins/src/shore_geometry.rs` beside `RungReader`: a `ShoreLaw` word (`Today`, `N`,
  `O`) and a column read that computes the law from the public parts (`vd_recipe::macro_field::{weights,
  catmull_rom_16}`, `vd_recipe::noise::fade`, `vd_recipe::height::relief_parts_from`, `vd_terrain::artifact::{sample_z,
  node_and_fraction, node_at, coast_side}`), in the recipe's integers where a public function exists and in `f64`
  otherwise (a what-if, not the shipped path; slice 3b re-measures on the shipped path).
- **CHECK-WHATIF-TODAY:** the what-if under `Today` must equal the shipped column bit for bit on every column of every
  square (the `G-TEETH-SELFCHECK` shape). It proves the what-if reads the shipped inputs. Band: all columns.
- **The instruments** `shore_teeth` and `lake_census` gain the lines of §7.3 for the three laws. The step-2 numbers of
  today's law come back as the control (they must equal `step2_census_gen20.txt`).
- **The hillshades** are written with the `image` crate (C14), no new library.

### 7.2 The stands (status §8b's poses, plus one chosen here)

| stand | the square the instrument reads | why |
|---|---|---|
| THE BELT | `shipped_drainage`'s belt square (~206 km at rung 7) | ~15 small lakes on dunes (the step-2 picture) |
| THE COAST 300 KM INLAND | the coast stand's shore (node 4 718 858, face NegY, a STEEP coast), the square the coast300 picture looks at | the teeth |
| THE PLAIN | the plain square | ~40 small lakes, a chain along a grid line |
| THE REFERENCE STAND | node 8 857 509's square | the owner's target frame; inland |
| THE POCKET STAND | the largest pocket's deepest node (28 913 nodes, 11 867 m under the sea) and its coast ridge | K2 at a pocket |
| **THE GENTLE COAST (chosen here)** | W6's own coast: the belt's coast at the owner's stand `0.617270 −0.437286 −0.654033` (W15.2's lines), taken ONLY if the instrument measures its median node-scale slope across the shore at ≤ 1/50; otherwise the coast edge within 1 000 km of it with the least slope and at least 16 km of shore in the square. The instrument prints the slope it found. | F1: the sideways step is largest where the coast is flat |

Plus the largest lake's edge (`shore_teeth`'s lake stand) and, for the per-lake lines, every lake of the globe.

### 7.3 The lines (each can fail; the band and its source beside it)

| line | what it reads | band (source) | TODAY (DOC or PREDICTED) | N PREDICTED | O PREDICTED |
|---|---|---|---|---|---|
| G-SHORE-SWAP@r-(r+1), every swap 0→1 … 17→18, sea and lake, on the gentle coast and the coast stand | the drawn shore's sideways step (`shore_step`'s method, first crossing along lines across the shore), in coarser cells | p99 ≤ ½, max ≤ 1 (T7's line, W17.1) | 0 m at 10→13 (DOC W15.2); 52.6 m median at 9→10 (DOC W17.1) | GREEN at every swap | p99 GREEN; max may be RED on the gentle coast (§5.6) |
| G-SHORE-SWAP-AREA@r-(r+1) | the wet area that changes side per km of shore (the mean step) | a reading | — | ≈ 0 | small |
| G-SHORE-OFFSET-NODE.<water>.<stand>.rNN | the drawn shore against N's line `S = 0` | N: ½ column (a self-check); O: its lateral reach at that rung | — | ≤ ½ column | ≤ reach |
| G-TEETH-OFFSET (kept, against the base's waterline) | as today | ½ column | 23 525.8 m RED at r07 (DOC) | GREEN except where the mirror moves the line near a sill | RED by the reach (by design) |
| G-TEETH-GRID vs THE NULL (never vs GRID-GROUND) | the drawn shore's direction histogram, 2 km windows | the flat null at that count (W13.3's method) | 2.950 RED at r07 (DOC) | RED on the coast stand (≈ the base's 2.3–3.1, DOC); nearer the null on the gentle coast | lower than N |
| G-LAKE-SQUARE.<stand> | each drawn lake outline's direction histogram, windows of a quarter node | the flat null at that count | squares (RED) | near the null for 1-node ponds; multi-node lakes may lean | near the null |
| G-TEETH-WALL / WALL10 on the drawn shore | the surface step across the drawn shore against the square's median step | no larger than the steepest neighbour step among the square's columns more than a node from any shore (a wall is a step the ground itself never makes) | 213.06× at r07 (DOC) | ≤ 2× | ≤ 3× |
| G-WALL-LINES.<stand>.rNN | the WALL method on every pair of neighbour columns across a node line where a candidate enters or leaves the 6 × 6, where `Ω` leaves 1 or 0, on every divide and sill line (`B = W_b` beyond a lake), on every pocket rim (`B = sea`), and at the band's edge | the same band; and no more pairs over 10× than today's law on the same pairs | — | 0 | 0 |
| G-PERCHED.<water>.<stand>.rNN | wet columns not joined through wet columns (4-neighbour) to a cell that touches a wet node of their own water | 0 (B7: a hollow the budget never saw is dry) | 0 | 0 (a few saddle wiggles possible) | > 0 near shallow lakes |
| G-LAKE-ISLAND.rNN | dry columns inside cells whose four corners are one lake's | 0 for N; a reading for O | 0 | 0 | a few |
| G-LAKE-CLUSTER.rNN | per connected lake body, its wet components | 1 | 1 | 1 | ≥ 1 for some shallow lakes |
| G-LAKE-AREA@r, r = 7, 9, 10, 11, 12, 13, 14 | each lake's wet area (every lake of the globe at 10–14; the stands' lakes at 7, 9) | a reading | the node area at 0–13 | the held read's area, then the fold | N's at ≥ 10 |
| G-LAKE-APPEAR@r-(r+1) | lakes wet at `r+1` with no wet column at `r` | 0 (W16 item 2) | 0 (PREDICTED: a lake is its node square up to rung 13) | 0 by construction from R₁ up | 0 from R₁ up |
| G-LAKE-FLICKER@r | lakes wet at `r−1` and `r+1`, dry at `r` | 0 | — | 0 or a few below R₁ (sampling) | the same |
| G-LAKE-AREA-STEP@r-(r+1) | per lake `|ΔA|` in coarser cells | ≤ 1 for a lake under 4 coarser cells; larger lakes judged by G-SHORE-SWAP on lake lines | LOST 3 698 / 558 at 14→15 / 15→16 (DOC) | GREEN | GREEN |
| G-LAKE-DRAWN | the lakes' drawn share of the land at rung 7, beside G-LAKE-SHARE 3.510 % | a reading (§15 Q5) | the node share | 2.8–3.4 % (79 % of lake nodes stand in 111 large lakes, C6) | a little over N |
| G-POCKET-HELD@r, every rung 0–18, sampled at every column of the square (between node centres) | pocket columns the law lifts over the sea | 0 (ruling B7) | 0 at node-centre cells (DOC) | 0 by construction | 0 by construction |
| G-RUNG-HOLD@r-(r+1) as a SURFACE gate (F7) | dry columns that stand over the sea at `r` and under it at `r+1` (or the reverse) with no water, in pixels at the swap | ≤ 1 px | 5 columns, 1.43 px at 9→10 (DOC) | 0 word changes; a few dips, each ≪ 1 px | the same |
| G-SHORE-TWO-WATERS / G-SHEET-SEA-OVER | wet pairs of two waters; sea vertices drawn over the sea's level | a reading / 0 | — | rare / 0 | rare / 0 |
| CHECK-SHORE-NODE | at every node centre (sampled directly), the side equals the mask word; the sills counted | 0 | — | 0 | 0 |
| CHECK-SHORE-CANDIDATES-OVER-4 | columns with more than four candidates | 0 | — | 0 | 0 |
| CHECK-PLAIN-CR-PERCHED (control) | G-PERCHED with the PLAIN Catmull-Rom in place of the held one | a reading: the control that shows why D2 holds the read | — | > 0 PREDICTED | — |
| BAND-NODES, BAND-BYTES, MIXED-NODES | the wire's size (D8) | a reading for §15 Q2 | — | 0.9–3.5 MB | the same |
| G-LOOK-DUNES | G-HACK, closed basins ≥ 10 m per 100 km², hill spacing on each square | a reading: how much the fade moves the dunes numbers | 0.226 / 2.94 / 3.2 km on the belt (DOC) | moves on lake-rich squares | moves less |
| BENCH | ns per column (pure; near water; mixed) and per chunk (a pure-sea chunk, a coastal chunk) for each law; the client's far-fold time | a reading | — | — | — |

### 7.4 The hillshades (for the owner's look)

For each of the six stands: three panels side by side — TODAY, N, O — of the rung-7 surface over the ~206 km square,
grey Lambert shade under the stand's own §8b sun, the wet columns flat blue. Plus one close panel per stand at rung 3
over about 8 km around one small lake (a pilot landing on its shore), and one far panel at rung 11 over about 1 000 km
(the fold). Each captioned with the law, the rung and "no far shadow". The pictures of the five flight stands are NOT
retaken in slice 0: nothing drawn moved, so step 2's pictures stand as "before".

### 7.5 What decides the variant

The owner chooses on §7.3's numbers and §7.4's panels. The numbers that decide:
- K1: G-SHORE-SWAP's max on the gentle coast (O's risk).
- K2: G-PERCHED, G-LAKE-ISLAND, G-LAKE-CLUSTER (O's risk).
- K3: G-WALL-LINES and G-TEETH-WALL (both).
- K4: G-LAKE-APPEAR, G-LAKE-FLICKER, G-LAKE-AREA-STEP (both).
- The look: G-LAKE-SQUARE, G-TEETH-GRID vs the null, and the panels.
- The cost: BENCH.

The owner's words on §15 Q2–Q9 are asked HERE, beside these numbers.

**Example.** The owner opens the plain's three panels. Today: forty blue tiles. N: forty round ponds in smooth
dishes. O: forty ragged ponds, some with a small pond beside them. Beside the panels the census prints, for each law,
G-PERCHED on the plain and G-SHORE-SWAP's max on the gentle coast, so the look and the law are judged together.

---

## 8. The recommendation, and what would change it

**RECOMMENDED: VARIANT N.** The reasons, in order:
1. **It keeps the rulings as written.** W6 ("the fine octaves may not carry a column across its water") and T7 (one
   selector) stand. O needs an amendment of W6.
2. **Its bounds hold by construction.** The sideways step (¾ cell), the basin (the maximum principle), the pocket (the
   mirror and `λ`), the far area (the fold). O's sideways step needs a first-order estimate and can exceed the max line.
3. **It is the law R2 reads.** Under R2 the same law on A0 gives the 1–2 km bays. O's octave machinery is thrown away
   when R2 retires the ridged kind under a field.
4. **It is cheaper**, near water and on the card.
5. **B8 asked for the edge "where the ground crosses the water's level"**, and W6 defines the shore's ground as "the
   solved field plus the coarse octaves" (W6, water.md:103-106). N is exactly that crossing.

**What slice 0 result would change the recommendation to O:**
- N's hillshades show the lakes as rounded squares (G-LAKE-SQUARE RED on one-node ponds), or the fade's band reads as a
  visible halo or rim that the owner rejects; AND
- O's G-SHORE-SWAP max on the gentle coast stands at or under one coarser cell at every swap, with G-PERCHED 0 and
  G-LAKE-ISLAND near 0, at no more than +15 % of a coastal frame; OR
- R2 moves further away than the next step (its own 23 findings may reorder it), so the interim look weighs more.

**What would change it to NEITHER:** N's G-WALL-LINES red on the divide or sill lines (the hold is wrong, not the side),
or G-LAKE-APPEAR above 0 from R₁ up (the fold is wrong). Then the design returns to the owner before slice 1.

**Example.** If the owner looks at N's plain and says the ponds in their smooth dishes look like craters of paint, and
O's gentle coast holds its swap under one cell with no perched pond, O is built instead; its octave side is then
retired at R2.

---

## 9. The order of work after slice 0

Each slice ends with: ITS NUMBERS (the census lines it moves); THE FIVE PICTURES — the belt, the coast 300 km inland,
the plain, the reference stand, the pocket stand — same poses, same sun, each captioned "no far shadow", plus the
reference stand's skyline numbers (B6 item 2); the living status file updated in the same change (§6 T1–T5 and T8,
§8 row 4, §12); DEFERRED.md D-TERRAIN-9 in the same change; and **HR5**: `cargo llvm-cov clean --workspace` first, then
`just coverage-fast`, with 0 real misses on every line the slice touched in `vd-recipe`, `vd-terrain`, `vd-client`
and `vd-connection-plane`, the vd-terrain baseline (113 real misses at generator 20, DOC status §12) no larger (F14).

| # | slice | what it builds | it ends with |
|---|---|---|---|
| 0 | **Measure, no world change** | §7 | the numbers and the panels; the owner's word on Q1–Q9 |
| 1 | **The kernel** (`vd-recipe`), new entries only | the held Catmull-Rom and its derivative weights; the continuous hold (a new entry; today's `shore()` byte for byte for its callers, F12); the fade; for O the snapshot accumulators and the lateral test; the per-chunk choice of the loop shape (F22) | recipe tests (§10); `gpu-drift` 5 979 of 5 979 (MEASURED) plus a NEW box that states candidates and a fade through the card's own gears; the kernel's ns (pure-sea chunk and coastal chunk); `terrain_pin`/`mesh_pin` unmoved. No picture: nothing drawn moves. |
| 2 | **The band on the wire** (after Q2) | the band's derivation in `vd-terrain` (one function, both hosts); the arm, the gateway cache, the client's book; the far chunk waits | the band's bytes and parts on the home planet, its arrival time on login, CHECK-SHORE-BAND (the membership both hosts derive). A CONTROL PICTURE on the five stands equal to step 2's (`frame_diff` ≈ 0 %): nobody reads the band yet. |
| 3a | **The far fold, derived and checked, read by nobody** | the per-node sample counts, `CoastCounts` over them, the in-node fold for rungs under a node; on both hosts | `far_side_probe`'s invariant on the new counts (0 disagreements, parent = sum of children); the per-lake area lines of §7 on the derived counts; a test on a stated body whose node is not a power of two (K6). A control picture. |
| 3b | **Every reader switches at once** (GENERATOR_VERSION + 1) | one reader at every rung: the chunk, the collider, `field_column`, the morph target (D7), the instruments; the sheet's level from wet columns; pins re-recorded | THE CENSUS of §7.3 on the shipped path; `rung_swap` (the handover's height step, F6); `water_edge_step` at every swap (F5); the frame cost with `VD_CLIENT_UNPACED=1` on the coast leg; THE PICTURES; the owner's look (G-LOOK for step 4). |
| 4 | **The beach and the gates' restatement** | D9; the far-swap gates restated as §7's lake lines; DEFERRED's far-swap and question-5/6 entries closed by the owner's words | the beach count (no pixel before 8e); the status file's §6 T1–T5 and T8 flipped |

Slices 3a and 3b are split so that no build ships a smooth near shore beside a node-square far shore: that would put a
new seam at 890 km between two slices. If the owner chooses Q5's solved drawn level, it is its own slice after 4 (an
ARTIFACT_VERSION bump). Step 2b lands first; this step takes the GENERATOR_VERSION after 2b's, and its slice 0 runs on
the HEAD of that day (the band depends on the mask, which 2b does not move — PREDICTED, and slice 0 checks it).

**Example.** At the end of slice 3b the owner opens the coast300 picture beside step 2's. The staircase is gone; the
coast bends with the solved ground; the small lakes near it are ponds, not tiles. The report says the reference stand's
skyline did not move (it is inland), and that `coverage-fast` reads 0 misses on the lines 3b touched.

---

## 10. The tests (HR5, Tier-A; each with a FAILING CONTROL)

1. **The held read** (`vd-recipe`): at a node it reads the node; on a plane it reads the plane; between four corners it
   stays inside their range on a million random stencils (the control: the plain read leaves the range, C7's example).
2. **Continuity across a node line** (`vd-recipe`): the held read of two neighbour cells agrees on their shared edge,
   bit for bit, on a million random stencils (the control: a clamp to the four CORNERS instead of each pass's two
   middle words disagrees).
3. **The presence** (`vd-terrain`): `Ω = 1` at every column within one node of a stated line; `Ω = 0` where the 6 × 6
   holds no wet node within two of a corner (the control: a 1-node dilation leaves `Ω < 1` inside the fade).
4. **The continuous hold** (`vd-recipe`): over a sweep of `B` across `W_b` on a dry column the surface is continuous;
   `LB ≤ B`; `LB_b` at `Ω = 0` stands under `B − A` (the control: v1's upper bound `4g` steps where it ends).
5. **The mirror and the extent** (`vd-terrain`): a stated basin with the home planet's octaves — a lake node, a sill at
   the level, an outlet under it, a trough beyond the rim, a pocket by the sea. Sampled at every column between node
   centres at every rung 0–10: the outlet, the trough and the pocket stay dry; the lake stops before the sill (the
   control: v1's floor at `W + 1 m` with the plain read floods the outlet).
6. **The node word** (`vd-terrain`): at every node centre of the home golden fields and a stated coast the side equals
   the mask word, the sills included (F8).
7. **The fold** (`vd-terrain`): a parent's count is the sum of its children's; a cell is wet only if a child is wet; a
   stated pond folds at one rung and stays folded (the control: the node-word rule at rung 13 brings it back, F21).
8. **SL5** (`vd-terrain`): tests 5 and 7 on a stated body whose node is 714 cells (the divisor rule's own example).
9. **The sheet** (`vd-terrain`): `the_ground_never_stands_over_the_water_it_is_under` stays green on a box of the new
   columns; a sea vertex beside a lake-held rim stands at the sea (the control: the level from all columns lifts it).
10. **The morph target** (`vd-client`): a column that changes side at a swap has a water vertex whose target is the
    coarser side (the control: the finer side's target keeps the water through the band).
11. **The band** (`vd-terrain`, `vd-client`, `vd-connection-plane`): the membership both hosts derive is equal; a
    far column read from the band equals the same column read from the tiles; a wrong size is refused; a far chunk
    waits for a missing band node.
12. **O only** (`vd-recipe`, `vd-terrain`): a flip is refused beyond the lateral hold and in a cell with a mirrored
    corner; the snapshot accumulators equal the separate sums read alone.

**Example.** Test 5 states a lake at 500 m in one node, a sill node at 500 m, and past it an outlet node at 470 m.
With the plain read and v1's floor (the sill left at 500 m, the outlet floored to 501 m), the columns between the sill
and the outlet read within a metre of 500 m or under it, and they flood: the control fails. With the held read and the mirror (the outlet reads 530 m), every column past the sill is dry.

---

## 11. The kernel, the card and one integer source (F7, F8, F22)

- **Where the arithmetic lives.** The held read, its derivative weights, the fade's distance (the recipe's own
  reciprocal, `crates/recipe/src/root.rs:132`), the quintic fade, and the continuous hold are integer functions in
  `vd-recipe`, branchless (`cell::greater`/`lesser`), masks in the data. The host (terrain) gathers the words; the
  kernel computes. The card will compile the same source.
- **The column's new words.** `FieldRead` gains, per candidate (at most four): the water's level, the side field's
  value, its distance, its presence; and the column's side, its water, its fade. `FieldRead::none` (the card, a body
  with no artifact) holds no candidate and reads today's arithmetic.
- **The loop shape.** The charter (and so the loop count) is chosen per CHUNK, never per column: a chunk whose node
  block holds any water runs the candidate path for all its columns, with the masks zero where a term does not apply
  (F22; the card's ledger, `height.rs:195-203`).
- **The card's own proof.** The card holds no artifact today, so its seed-only boxes cannot exercise the new path.
  Slice 1 adds a `gpu-drift` box that STATES candidates, levels and fades as data and runs them through the card's own
  gears.
- **Two hosts, one generator (SL10).** The band and the far fold are derived on both hosts from the same words; the
  drift gate gains a far chunk near a coast built from the band against the same chunk built from the full rows.

**Example.** The shard builds a coastal chunk at rung 7 for a pilot's boots. Its node block holds the sea, so every
column runs the candidate path. The columns more than two nodes from the sea read a presence of zero and a fade of one:
their dunes stand whole and only the sea's lower bound holds them, which is today's land arithmetic. The columns near
the shore fade their dunes and hold their surface by D5. The client's builder runs the same integers.

---

## 12. SL5 — every rule for every body (F10)

- R₁ is the first rung that reads a pyramid level, by `level_for`'s own rule (C9); it is rung 10 on the home planet and
  another rung elsewhere. Its cell is between a sixteenth and an eighth of a node on every body (ARITH from the rule:
  `62 · 2^R₁ ≥ 4` nodes).
- The fold nests by rung cells, which always nest; the node counts use today's footprint rule rounded outward on a body
  whose node is not a power of two.
- The fade's reach is one node of the body's own lattice; the slope floor is the rows' one metre.
- No rule names rung 13, 8 km or 8 192 cells. Test 8 runs on the 714-cell body.

**Example.** On a 5 km moonlet whose node is 714 m, R₁ falls where a chunk spans four nodes; a pond in a crater folds
away at the rung where its samples lose their majority, exactly as on the home planet.

---

## 13. What R2 changes, and its price on the wire (F19)

- **The law.** Unchanged (§4.10): the side reads the finest held lattice. Under R2 that is A0 at rungs 0–10, with A0's
  own side words at 1 024 m (a connected flood at A0's resolution: R2's finding 2, "one fact says where the water is").
- **The far view under R2.** The fold needs the side at rung R₁'s samples over the whole globe. Two ways:
  - A0's heights near every side change: 64 A0 cells a node × 2 B = 128 B a band node plus its context — about 64 times
    the rows' band (F19's arithmetic: about 110 MB on a tenth of the nodes);
  - **the side's rung-R₁ samples themselves**: 2 bits × 64 = 16 B a mixed node, plus one "mixed" bit a node for the
    whole globe (1.1 MB), because under A0 a mixed node no longer follows from the node mask (a ria crosses land nodes).
    ARITH: 0.5 million mixed nodes → 8 MB + 1.1 MB.
  The second is about seven times cheaper and needs no A0 read on the client for the far view. Recommended for R2; the
  owner decides it with R2's own SL6 ask.
- **The 10 → 11 swap under R2.** The fold keeps a ria at rung 11 as long as it holds half a block's samples, and folds
  it away after, never back. So the rias do not "vanish at 890 km": they fold with the rung, like any sub-cell water.
  R2's slice 0 should still measure G-SHORE-SWAP@10-11 on A0 before the owner answers R2's wire question.
- **What this step does NOT build that R2 must undo:** no stored side lattice, no per-node word that assumes 8 km, no
  change to the pyramid. The per-node sample counts of D6 are derived and change with the lattice.

**Example.** After R2 a pilot at 800 km over the belt's coast sees a drowned valley 5 km long as a thin bay two
kilometre cells wide. She climbs past 890 km; at rung 11 the bay is one cell wide, and at rung 12 it folds into the
coast's smooth line. It never comes back as she climbs higher.

---

## 14. The risks

- **Risk 1. N's look.** Smooth node-scale curves and a smooth band at every shore may read as "paint" or as rounded
  squares. Slice 0's panels judge it before any build.
- **Risk 2. O's sideways step.** The worst case is 1¾ coarser cells (§5.6). If red, O's hold shrinks to a quarter cell and
  O's detail to about 450 m.
- **Risk 3. The distance estimate.** `|S| / |∇S|` is first-order. Where it errs by a node, `Ω` ends the fade on a node
  line (a soft node-aligned edge). G-WALL-LINES reads it.
- **Risk 4. The mirror moves a sill's line slightly** toward the lake (the outlet's depth enters the sill cell's read
  through the outer weights). A reading in slice 0 (CHECK-SHORE-NODE counts the sills; the extent test reads the rest).
- **Risk 5. The outlet's drop.** The lake stops at its sill; the channel past it is dry ground under the lake's level: a
  small dip at every spilling lake's mouth until step 6 carves the spill (§15 Q7).
- **Risk 6. The band may be larger than the ARITH** (3 883 lakes, 236 pockets, fractal coasts). The fallback: the
  side's samples (16 B a mixed node, §13) in place of the heights.
- **Risk 7. The client's one-time fold** may take several seconds (§4.8). Fallback: fold per far chunk on demand.
- **Risk 8. The fade moves the dunes numbers** on lake-rich squares (G-LOOK-DUNES). That is a look change near water, not a
  cure of the dunes; the report says so.
- **Risk 9. O on the card.** Nine snapshot accumulators at rung 0 widen the loop body; the card's ledger has failed on
  looser shapes. Slice 1's stated box is the gate.

**Example.** On a delta coast the ground stands 5 m over the sea for 30 km inland. N draws one smooth shoreline and a
flat plain; O draws a maze of lagoons within one coarser cell of it, and at each swap the maze's edge may move by a
cell and a half. That is risk 2, and G-SHORE-SWAP on the gentle coast is the line that says how much.

---

## 15. The owner questions (each with a recommendation; none is answered here)

1. **THE VARIANT.** N (the node-scale side: smooth curves now, bays from R2) or O (the octave side under a lateral
   hold: ragged kilometre detail now, thrown away at R2)? **Recommended: N**, decided on slice 0's numbers and panels
   (§8). B8 expected the edges to "follow the fine noise"; N does not, so this word also confirms B8's reading.
2. **THE SHORE BAND ON THE WIRE (SL6).** May the planet realm ship, once beside its head, the rows' height of every node
   within two nodes of a side change, and the level of each lake node among them (ARITH 0.9–3.5 MB, slice 0 measures;
   derived from the mask on both hosts, so no index crosses)? **Recommended: YES.** Without it the 704 km swap turns a
   pond into a node square (about 5 px) and the far fold cannot be built. Alternative shape: the side's rung-R₁
   samples (16 B a mixed node, about 8 MB ARITH), which is also R2's cheaper road (§13).
3. **W6's QUARTER `k`** (the shelf: how far a column stays on its side, as a share of its node-scale distance from the
   water). **Recommended: a quarter, as built**, judged on slice 3b's pictures. v1's question 2 had its parenthetical
   backwards for the bank: with one shift for both bounds, a half gave a bank at 2× the ground's slope (flatter) and an
   eighth at 8× (steeper). v2 has no bank bound at all (the fade replaces it), so the quarter now sets only the shelf.
4. **THE NEW WORD `K`** (how deep the fine octaves may dig a DRY column that stands under a water's level — beyond a
   divide, in a pocket — as a multiple of its node-scale depth). **Recommended: `K = 1/k = 4`.** It makes the hold
   continuous at a divide, a sill and a pocket rim; any `K ≥ 1` does; it shapes no visible bank.
5. **A LAKE'S DRAWN LEVEL.** Keep the budget's level (the drawn area follows the node-scale ground; PREDICTED 2.8–3.4 %
   of the land against 3.510 % by nodes) or solve each partial lake's drawn level so its drawn area equals its budget's
   (Langbein's balance fixes an area; ARTIFACT_VERSION + 1)? **Recommended: keep the budget's level now**, decided on
   slice 0's G-LAKE-DRAWN and panels. W16 is met by the fold either way (D6), so this is a question of amount, not of a
   seam (F4's point, answered).
6. **THE FAR FOLDS.** A pond smaller than half a coarser cell folds away at that rung and stays folded. **Recommended:
   lawful**, by W16's own example (*"the tarn in the pass … holds its own water until it is smaller than a cell, and
   then it simply folds away"*), judged by the per-lake lines that can fail (APPEAR 0, FLICKER 0, AREA-STEP).
7. **THE OUTLET'S DROP** (§14 risk 5). **Recommended: accept until step 6**, which carves the spill and lays the delta.
8. **PLAN QUESTION 5, THE DRY-LAND HOLD** (DEFERRED: *"the owner's word on question 5 (the dry-land hold) — before step
   4's one shoreline"*, `docs/design/DEFERRED.md:10081-10091`). v2 replaces the stencil rule with ONE continuous hold
   over the sea by each column's own node-scale ground, at every rung, with no stencil (D5). MEASURED today: 5 columns
   at 9 → 10, 1.43 px (DOC). **Recommended: yes, as D5 states it** (PREDICTED 0 word changes; a few dips far under one
   pixel, read by the restated G-RUNG-HOLD).
9. **PLAN QUESTION 6, TWO LAKES IN ONE BLOCK** (DEFERRED `:10076-10080`; `lake_census`'s own line states the bound as
   proposed). The far view's water is the block's highest lake level; the lower lake's surface steps at a far swap.
   MEASURED: 211 columns at 14 → 15, the largest step 0.016 px (DOC). **Recommended: the one-pixel bound**; a level per
   node on the wire (an SL6 ask) only if a picture shows the step.
10. **(ONLY IF Q1 = O) AMEND W6**: the fine octaves may carry a column across its water within one cell of the coarser
    rung of the coarser rung's line. **Recommended: not needed under N.**
11. **THE BEACH FROM THE SIDE FIELD** (D9). **Recommended: yes, in slice 4** — no pixel shows it until 8e, and 8e would
    otherwise read node squares.

**Example.** The owner reads Q2 beside the far panel at rung 11: the left half is today's node squares past 704 km, the
right half is N's fold. The report says the right half needs the band, and what the band costs in bytes.

---

## 16. What this file does NOT decide, and what no owner said (F13, F23, K9)

- No owner word is recorded here. Every §15 question is open. B8 decided the ORDER (the edges before R2) and accepted a
  dune-shaped edge; it did not choose a variant, the band, `k`, `K`, the drawn level, the folds, plan question 5 or plan
  question 6.
- W6's quarter is still owed (W6, W10, W15, W16.9 item 7 all say so).
- The owner's look at W17's and W18's stands is still owed (W17.10, W18.6).
- DEFERRED D-TERRAIN-9's entries for question 5, question 6 and the far-swap lake losses stay 🟥 until the owner's words
  and slice 4.
- v1's "decided here and stated for the record (no question)" line (its §16) is WITHDRAWN.

**Example.** A reader who sees "D5 replaces the stencil rule" must not read it as the owner's word on plan question 5.
It is the design's proposal; Q8 asks it.

---

## 17. Where the docs and the code disagree (found while writing; fix when the step touches them)

- The status file's §6 cites line numbers of the generation-19 `artifact.rs`: `sample_side` is now `:1337-1362` (cited
  `:1134-1159`), the nearest-node water is `sample_row` `:1132-1146` (cited `:1027-1041`), the land rule `:1231-1234`
  (cited `:1084-1087`) with its stencil test `:1271-1303`; `fold_water` is gone (`block_water`, `:1606-1614`). T8's
  beach is `chunk.rs:412-417` (cited `:404-411`).
- `FieldRead::side`'s doc (`plan.rs:150-158`) says the side is *"the same word at every rung, because it is the same
  fine node's bit"*. Since W15 a far rung reads the footprint's majority, and since step 2 a land column's word depends
  on its read stencil (`SIDE_INLAND`).
- `column_surface_from`'s doc (`plan.rs:187-190`) and `shore`'s doc (`height.rs:375`) say the shoreline stands where
  the ground without the fine octaves crosses the water, *"the same place at every rung"*. Since W10 the side is the
  mask's, so the shoreline stands on the node square's edge, not where the ground crosses.
- `shore_teeth`'s header (`shore_teeth.rs:76-90`) and its printed legend (`:1053`) state its predictions as
  UNMEASURED; they were measured at generations 19 and 20. Its line *"at rungs 10–13 most small lakes have no edge at all (T5)"* is stale since step 2
  (`block_water` keeps a lone lake).
- `ColumnRead`'s doc (`chunk.rs:290-292`) says a column takes *"its water and coast from the nearest row"*; since W16
  the water follows the side word through `column_water`, and only the coast bit is the nearest row's.
- `body.rs:86-92` says the fine threshold is 8 224 m × 4 = 32 896 m; the code reads `MACRO_CELL_TARGET_M` (status §10).
- The CLAUDE.md summary of the water rulings names a **W9** ("the on-foot ground", the coast mask as an SL6 ask);
  `owner_decisions_2026-09-21_water.md` has no W9 section (it goes from W8 to W10).
- v1 claimed at its §2 that *"the crossfade morph carries the ground and its water across the band (W18.7), so the
  change is a fade"*. The water's morph target reads the FINER rung's side (`chunks.rs:1418-1428`), so a side change at
  a swap is a pop (F5). D7 answers it.
- v1's §18 table listed the findings in severity order while its own closing example cited the workflow's numbers
  ("finding 16", "finding 15"). §18 below prints both numbers.

**Example.** A reader who follows the status file's `artifact.rs:1148` to find the nearest-node rule lands inside
`column_water`'s doc. The rule lives at `artifact.rs:1345-1352` today; this design cites `HEAD`.

---

## 18. v1's findings (2026-09-25) and how v2 answers them

Two adversarial readers attacked v1 (workflow `wf_d8fb34af-0aa`): one against the laws and the continuity, one against
the pictures and the cost. The rows keep v1's severity order. "F#" is the workflow's own number (the task text's);
"v1 row" is v1's §18 row. Every answer is a DESIGN answer: nothing is built, and slice 0 must measure each one. A row
is closed only when its line in §7.3 reads green on the shipped path (slice 3b).

| F# | v1 row | sev. | finding | how v2 answers it (or why it is not real, with evidence) | where |
|---|---|---|---|---|---|
| F1 | 1 | HIGH | The swap bound limits a feature's width, not the shore's sideways step. | REAL. N: no octave enters the side, so both rungs of a swap sample ONE field and the sideways step is at most ¾ of a coarser cell on every coast and slope, by construction; the far rungs fold (≤ 1 cell). O: the lateral hold is IN the law (a flip only within one coarser cell of the coarser line), worst case 1¾ cells, measured. G-SHORE-SWAP on a GENTLE coast (slope ≤ 1/50, W6's own coast by a stated rule) at every swap 0 → 18, beside W15.2's zeros. | §4.6, §5.1, §5.6, §7.2, §7.3 |
| F2 | 2 | HIGH | The extent law holds only at node centres: the lake floods its outlet, and the sea enters a pocket and lifts its floor. | REAL. v2's side field is the HELD Catmull-Rom (the maximum principle) on node words decided by the mask with the MIRROR: a cell with no wet corner is dry at every point, so the outlet, the trough and the pocket stay dry between node centres at every rung. The hold never lifts a column over a water it stands under (`λ` with `K`), so G-POCKET-HELD is 0 by construction. Test 5 uses the home planet's octaves and samples between centres; G-POCKET-HELD reads every column at every rung 0–18. | D2, D5, §4.3, §4.7, §5.3, §10 test 5, §7.3 |
| F3 | 3 | HIGH | The two-way hold builds a wall at every lake's divide and sill. | REAL. v2 has no upper bound. The fade removes the lift at the side's line; the lower bound is ONE continuous function of the node-scale ground (`λ`, continuous at zero), a max over every candidate water, each lake's term faded by its presence, so the held water never switches at `B = W_b` beyond a divide. G-WALL-LINES reads every divide, sill and pocket-rim line. | D4, D5, §4.4, §5.4, §7.3 |
| F4 | 4 | HIGH | The far rungs draw a lake at its node area and the near rungs at G's area: water appears as the pilot climbs. | REAL, and treated as a W16 requirement. D6's far fold: every rung coarser than R₁ takes the majority of the rung-R₁ samples of the SAME side, so the far area is the near area; a parent is wet only if a child is (no water appears on the climb). Gates: G-LAKE-AREA at rungs 7–14 for every lake, G-LAKE-APPEAR (0), G-LAKE-FLICKER (0), G-LAKE-AREA-STEP. The drawn level (Q5) stays a question of amount. | D6, §4.5, §7.3, §15 Q5 |
| F15 | 5 | HIGH | The upper hold ends where the stencil loses the water: a wall on a node line and a 32 km square around each small lake. | REAL. v2 has no upper hold. Every term a water adds (its lower bound, its share of the fade) is multiplied by the presence `Ω_b`, which is 1 within a node of the line and 0 before the 6 × 6 block loses the water (D3), so no term switches on a node line. G-WALL-LINES reads every node line where a candidate enters or leaves, where `Ω` leaves 0 or 1, and the band's edge. | D3, D5, §7.3 |
| F16 | 6 | HIGH | Near a shallow lake the noise decides the side: a square cluster of ponds, and water past its own sill. | REAL (the root). N: no octave decides a side, so no pond stands in a noise pit; the maximum principle keeps the lake inside the cells that touch its nodes. O: the lateral hold and the basin bound (no dry → wet flip in a cell with a mirrored corner). Slice 0 measures BEFORE the kernel: PERCHED (band 0), ISLAND, CLUSTER, the wet share per lake, G-LAKE-DRAWN; Q5 is asked with measured numbers. | §4.1, §4.3, §5.1, §5.3, §7.3 |
| F5 | 7 | MEDIUM | A side change at a swap is a pop, not a fade. | REAL (the code: `chunks.rs:1418-1428` reads the finer side). v2 withdraws v1's claim and gives the water its own target: the coarser rung's side and level through the one reader at the vertex's rung-`r+1` cell. `water_edge_step` at every swap in slice 3b. | D7, §9 slice 3b, §10 test 10, §17 |
| F6 | 8 | MEDIUM | A second octave selector (Nyquist) amends T7 and W6, and no owner question asks it. | REAL for v1. v2 has no Nyquist selector: N reads no octave in the side; O reads the ground's OWN live set (T7's one selector). W6's amendment exists only under O and is asked (Q10). `rung_swap` runs in slice 3b; the fade's change of the drawn ground at a swap is bounded by the ground's own dropped octaves (N) or by the lift over the lateral reach (O, about 66 m at 9 → 10, ARITH), far under one coarser cell. | §4.1, §5.1, §5.9, §15 Q10, §9 |
| F7 | 9 | MEDIUM | D7 reads the fine stencil while the far ground reads level 1: dry pits under the sea at rungs 10–12, unseen by the census. | REAL for v1. v2's lower bound reads no stencil at all: every column is held over the sea by its OWN rung's node-scale ground (D5), so no word changes at a swap. Where a level-1 mean stands under the sea beside a steep coast, the column is still drawn dry (the side is the rows'), and its dip is far under one pixel at 704 km and more. G-RUNG-HOLD is restated as a SURFACE gate in pixels. | D5, §7.3 |
| F8 | 10 | MEDIUM | The tie at a node centre is not the facies' tie: every spilling lake's sill reads wet in G and land in the mask. | REAL (`solve.rs:1486` is strict; `lakes.rs:496,587` put the level at the sill). v2 decides every node by its MASK WORD: the lake's own nodes read `min(z, W − 1 m)`, every other node at or under the level reads `W + max(1 m, W − z)`, so the sill reads `W + 1 m`, dry. CHECK-SHORE-NODE counts the sills. | D2, §10 test 6, §7.3 |
| F9 | 11 | MEDIUM | The sheet lifts the sea to a lake's level beside a barrier that holds the lake. | REAL for v1 (its dry column carried the lake's word). v2 keeps a dry column's standing water as today's land rule (never a lake's level) and takes the sheet's level from WET columns only. New line: sea vertices drawn over the sea's level, band 0. | D7, §7.3 |
| F10 | 12 | MEDIUM | "Rung 13 = one node" is true only on the home planet (SL5). | REAL (one side claim is slightly off: on the 714-cell node rung 1's columns CAN stand on node centres, because 714 = 2 · 357, C8; the conclusion stands, no rung's cell is one node there). v2 never names rung 13: R₁ is `level_for`'s first pyramid rung on every body; the fold nests by rung cells, which always nest; the node counts use today's footprint rule rounded outward. Test 8 runs on the 714-cell body. | D6, §12, §10 test 8 |
| F11 | 13 | MEDIUM | The taper and the candidate cap are stated numbers, and the taper puts the grid back into the coast. | PARTLY REAL, partly retired. v2 has no taper on the side (N: none; O: a lateral hold measured in the ladder's own cells). The presence `Ω` is node-aligned, but its reach comes from the reads the chunk already holds (C11) and it only ENDS a water's hold terms; inside the region where it is 1 (every point within a node of the line) it changes nothing, and it can show only where the first-order distance errs (§14 risk 3, G-WALL-LINES). The cap of four candidates stays, and an overflow is a FAILING line (band 0), never a silent drop. G-TEETH-GRID on the gentle coast is judged against the null. | D1, D3, §14 risk 3, §7.3 |
| F17 | 14 | MEDIUM | The side's Nyquist rule moves the shore at three swaps that read 0 m today, by the amount W6 measured. | REAL for v1. N: the swaps 9 → 10 … 12 → 13 sample one field (¾ cell worst, no octave height); O: the octaves live only up to rung 9 (C10) and their flips are held within one rung-10 cell. The slice-0 table puts each swap's p99 and max beside W15.2's zeros; if O's p99 is over half a cell, the owner chooses N or a quarter-cell hold. | §4.6, §5.6, §7.3 |
| F18 | 15 | MEDIUM | The dry bank rises at four times G's slope from a flat shelf, and question 2 states a half and an eighth backwards. | REAL. v2 has no bank bound: the fade replaces it, and it adds at most `1.875·|f|/node` (0.07–0.10, ARITH) to the slope. Q3 corrects the parenthetical. The owner gets two words apart: `k` (the shelf) and `K` (the depth under a water a dry column may be dug), and `K` shapes no visible bank. | D4, D5, §15 Q3, Q4 |
| F19 | 16 | MEDIUM | Under R2 the rias vanish at 10 → 11; both exits cost the rias or about 64 times the band. | REAL, and priced. Under v2 the far fold keeps a ria while it holds half a block's samples and folds it after, never back, so it does not "vanish" at 890 km. The wire: the side's rung-R₁ samples (16 B a mixed node + a 1.1 MB mixed map, about 9 MB ARITH) instead of A0's heights (about 110 MB, F19's ARITH). R2's slice 0 measures G-SHORE-SWAP@10-11 on A0 before R2's wire question. | §13, §15 Q2 |
| F20 | 17 | MEDIUM | The teeth gates judge the drawn line against itself; GRID contradicts R2 on the steep coast; no line reads the new squares or walls. | REAL. The judging lines never read the drawn line as their own reference: GRID against the null (N PREDICTED RED on the steep coast stand, like the base's own 2.3–3.1), a square test per lake outline, PERCHED, CLUSTER, ISLAND, WALL-LINES, and a GENTLE coast stand. OFFSET-NODE is labelled a self-check for N. | §7.2, §7.3 |
| F12 | 18 | LOW | Slice 1's "no world change" is proven by pins that never read a stated side. | REAL. Slice 1 adds NEW entries and keeps today's `shore()` byte for byte for its callers; the world changes only in 3b, with GENERATOR_VERSION + 1 and re-recorded pins. | §9 slice 1, D5 |
| F13 | 19 | LOW | The restated fold gate cannot fail, and two owner words are taken as decided. | REAL. The fold is gated per lake by lines that can fail: APPEAR (0), FLICKER (0), AREA-STEP. Plan questions 5 and 6 are §15 Q8 and Q9; v1's "decided for the record" is withdrawn (§16). | §7.3, §15, §16 |
| F14 | 20 | LOW | HR5 coverage is not an exit criterion of any slice. | REAL. Every slice ends with `cargo llvm-cov clean --workspace` and `just coverage-fast`: 0 real misses on the touched lines of `vd-recipe`, `vd-terrain`, `vd-client`, `vd-connection-plane`; the vd-terrain baseline (113, DOC) no larger. | §9 |
| F21 | 21 | LOW | No column stands on a node centre below rung 13, so a small pond can vanish at 11–12 and come back at 13. | REAL for v1 (C8). v2's rungs above R₁ are majority folds of rung-R₁ samples, never a node-centre point sample, so a folded pond stays folded (§4.5). G-LAKE-APPEAR and G-LAKE-FLICKER at every swap. | D6, §4.5, §7.3 |
| F22 | 22 | LOW | The mixed path cannot be measured on the card, and its per-column charter choice is the loop shape the card's ledger records as faulty. | REAL. The charter is chosen per CHUNK (every column of a chunk that holds water runs the candidate path, masks zero where a term does not apply). Slice 1 adds a `gpu-drift` box that states candidates and fades through the card's gears, and benches a pure-sea chunk and a coastal chunk. | §11, §9 slice 1 |
| F23 | 23 | LOW | Plan questions 5 and 6 are recorded as decided, but DEFERRED and the census still name the owner's word as owed. | REAL. Both are §15 Q8 and Q9, each with its recommendation and its measured number (5 columns / 1.43 px; 211 columns / 0.016 px). | §15, §16 |

**Example.** On the plain stand a one-node tarn 20 m deep would draw, by v1 as written, as a soft 16–32 km square of
ponds in the noise's pits (F16), fenced by a straight scarp on the node line where the stencil loses the lake (F15),
and it would vanish at rungs 11–12 and come back as a full square at rung 13 (F21). By v2's variant N it draws as one
round pond about 3.3 km across in a smooth dish, with no scarp, and it folds away once at the rung where it holds
under half a cell, and stays folded.

## 19. v2's FINDINGS (2026-09-25, workflow `wf_ff164ced-d5d`) — OPEN, and they bind v2

Two adversarial readers attacked v2 (one on the laws and the continuity, one on the pictures and the cost): 24 findings,
6 HIGH, 20 of them against BOTH variants. THE FINDING WINS until a measurement says otherwise. Their common root: the side
is read from one field (the held, mirrored node read) and the ground and the sheet are drawn from another (the plain read
plus the faded fine relief), so a wall or a drowned strip stands wherever the two disagree — and slice 0's lines, as
written, read the law's own side, not the DRAWN water, so they cannot see it. Also: ruling B8 item 1 accepted "a ragged
outline that follows the ground", and variant N draws smooth node-scale ponds (finding on the recommendation).

| # | severity | variant | finding | the fix it asks | sections |
|---|---|---|---|---|---|
| 1 | HIGH | both | The side's line is not where the drawn ground crosses the water: a wall and a drowned strip wherever a mirrored node is in the read | Make the drawn node-scale ground and the side ONE field inside the presence. Blend B toward W_b + S_b with the same fade, for example h' = terrace(B + Ω_b·(1−φ_b)·(W_b + S_b − B) + φ·f), so the ground stands exactly at W_b on the line. Or mirror only the four CORNER words: the maximum principle needs only the corners and the per-pass clamp. Keep the twelve outer words raw, so S_b = B − W_b wherever no corner is mirrored. Add slice-0 line G-SIDE-VS-GROUND.<stand>.rNN: /B − W_b/ on the side line, band half the rung's extractor step. Correct risk 4's sign. Example: from a hull at 150 km over the pocket stand's coast ridge, the pilot must see the shore where the ridge's ground meets the sea. She must not see the sea 6 km inland under a 55 m cliff. | D2, D5 (lines 260-263), §4.2, §8 reason 5, §14 risk 4, §7.3 G-TEETH-OFFSET |
| 2 | HIGH | both | A side-dry column whose ground stands under the sea gets no hold, and D7 draws the sea over it: W10's cure is reversed from R₁ up, and a pocket's rim takes the sea | State the dry hold per water as W6 and W10 state it for the sea. A column that the side calls LAND for the sea (not mirrored) stays held over the sea by k·/B − sea/, even where B < sea. λ with K applies only where the column is a pocket (mirrored in the sea's field) or beyond a lake's divide. Prove continuity where those two meet, inside the fade band at the pocket's rim. State D7 for a dry column between node centres: no standing water where the mirror makes it a pocket, and never the sea over a column the side calls land. Add slice-0 line G-SEA-OVER-DRY.<stand>.rNN: dry-side columns under a drawn sea or lake sheet, band 0, at every rung 0–18. Restate G-RUNG-HOLD without 'with no water'. Example: a hull that descends over the belt's coast must not see the sea spread 11 km over the land when the 704 km ring sweeps under it. | D5 (λ, lines 249-253, 268-270, 278-281), D7 (lines 309-313), §7.3 G-RUNG-HOLD and G-POCKET-HELD, §18 F7 |
| 3 | HIGH | O | O's lateral hold tests the coarser rung's FIELD, not its drawn SIDE, so the telescoped reach and the 1¾-cell worst case do not follow | Gate a flip on rung s+1's DRAWN side, not on its field. Permit a flip at rung s only inside a rung-(s+1) cell that is a boundary cell of side_{s+1} (its 3 × 3 of rung-(s+1) sides holds both words). Or carry each rung's distance to its own drawn line. Then prove the telescoping again and state the new worst case. Example: seen from 450 km, a tarn in a pass must not grow a second pond 1.6 km off its shore at 300 km that is gone again at 450 km. | §5.1 (lines 470-479), §5.6 |
| 4 | HIGH | both | The drawn water is not the side's water: a dry column sinks under the sea that D7 names as its water, so the sea returns inland at 704 km and over pockets | Hold a dry column over every water that STANDS over it, by the side's own field: LB = W + k·S_b with S_b ≥ 0 on the dry side. This is continuous at S_b = 0 and never reads B's level mean. Keep λ with K only for columns with NO standing water. Give a pocket floor no standing water at every rung, as HEAD's Pocket arm does. Add CHECK-DRAWN-IS-SIDE at every rung 0-18 on every stand: a dry column with its surface under its standing water, and a wet column with its surface over its water. The band is 0. Compute it with water_sheet_sided's own s < l rule. Paint every hillshade and every wet-area line from that drawn set, and keep the side set only as a control. | D5 (lines 240-283), D7 (307-317), §7.3 G-RUNG-HOLD / G-POCKET-HELD, §7.4 |
| 5 | HIGH | both | The surface is built on B and the side on the mirrored held read, so a wall stands at the shore beside every deep mirrored node | Use one field for the ground and the side near a shore. Either (a) build the surface on B_side = W_b + S_b, blended into B by Ω_b and the fade, so that B_side = B wherever ẑ = z. Or (b) state that the mirror also moves the drawn ground past sills and pockets, and show it in the panels. Then gate the step across the drawn shore at rung 0 and rung 3, against the column's own slope step, on the pocket stand and on every spilling lake's outlet. Correct risk 4's direction. | D2 (lines 174-201), D5 (lines 260-263), §14 risk 4 (823-824) |
| 6 | HIGH | O | Variant O as written draws the coast on the lateral band's EDGE, with walls there and slivers on pit rims: slice 0 would test a broken rule, not the octave side | Restate O BEFORE slice 0 measures it, as a continuous and bounded displacement: side_s = sign(S_b + clamp(m·residual_s, −c_s·/∇S_b/, +c_s·/∇S_b/)), with c_s = one coarser cell in nodes. The line then moves at most one cell, continuously, with no band edge, and it has no zero set away from N's line (so it makes no slivers). Measure φ from THAT line, and apply the hold with that same field. Then O's panels and its G-SHORE-SWAP numbers test the octave side itself. | §5.1 (lines 464-485), §5.4, §5.6 (513-519), §7.3 O column |
| 7 | MEDIUM | both | Below R₁ the side is a point sample, so a small pond appears on the climb and flickers at 9 → 10, where today reads 0 | Make every rung that the fold does not cover monotone too. One way: fold rungs F … R₁ from one fine sample lattice, where rung F is the finest rung the per-node count budget allows, and point-sample only below F. Another way: derive the least drawn pond from the read itself, so that every drawn pond covers every sample of each rung up to the rung where it folds. Make G-LAKE-APPEAR and G-LAKE-FLICKER hard lines with band 0 at every swap 0 → 1 … R₁−1 → R₁. Example: a pilot who climbs out of the belt must not see a shallow tarn blink on at 704 km and off again at 900 km. | D6 (lines 287-291), §4.5, §4.6, §7.3 G-LAKE-FLICKER (line 618), §18 F21 |
| 8 | MEDIUM | both | The fold does not bound the sideways step by one coarser cell: W15.2 measured the same majority rule at 3.7 cells, and DEFERRED registers the fold's losses as a W16 breach | Split the line in two. (1) A sideways step: the nearest crossing, or the first crossing only where both rungs draw water within one coarser cell. Band: max ≤ 1. (2) A fold-away line: the features lost at each swap, their width, and the swap distance, for the owner to judge under Q6. State Q6 as an amendment of DEFERRED's W16 reading. Beside it, give slice 0's count of the lakes that fold at each of 10 → 11 … 13 → 14, and the distance of each swap. Correct the PREDICTED column. Example: a lake ribbon 1 km wide and 20 km long on the plain's grid line vanishes whole at 890 km. The owner must see that number, not a green 'sideways step'. | §4.6 (lines 417-420), §7.3 G-SHORE-SWAP (line 605), §15 Q6, §4.9 |
| 9 | MEDIUM | both | SL5: D6 claims an exact fold on a body whose node is not a power of two, then specifies node counts 'rounded outward', which breaks it | On every body, fold by RUNG CELLS: a count pyramid over rung-R₁ cells, with each rung-(r+1) cell the sum of its four rung-r cells, and never over node blocks. Or state that the fold is exact only where the node is a power of two in cells, and ask the owner (SL5). Test 8 must then check 'parent wet ⇒ a child wet' on the 714-cell body at every rung. Example: on a moonlet whose node is 714 m, a crater pond must fold away once as the pilot climbs, and it must never come back one rung higher. | D6 (lines 289-299), §12, §10 test 8, §18 F10 |
| 10 | MEDIUM | both | The lake's hold ends on a node-aligned ramp: F15's square around each small lake comes back where the quarter binds the fine pits | End the lake's term with a smooth function of the side's own distance, for example the share 1 − fade(clamp(d_b − 1, 0, 1)). Keep Ω only as a guard, and prove that Ω is 1 wherever that share is above zero. Or state the held-floor ring as a look and put it to the owner. Add slice-0 line G-HOLD-RAMP.<stand>.rNN: the columns where LB_b binds inside cells with 0 < Ω_b < 1. Add a hillshade crop of one plain lake at rung 7. Example: a pilot at 5 km over the plain must not see a 16 km square of flat-floored dune pits around each small pond. | D3 (lines 208-220), D5 (LB_b, line 252; A, lines 257-259), §7.3 G-WALL-LINES and G-LAKE-SQUARE |
| 11 | MEDIUM | both | The held read's per-pass clamp depends on the pass order, so the side and the wet hold crack on every face seam that swaps the axes | Make the held read symmetric: evaluate both pass orders and take their mean. On a node line both orders give the 1-D held read along that line, so node-line continuity stays. The mean does not change under a transpose, so the two faces agree at a seam. Both terms lie inside the corners' range, so the maximum principle stays. Add test 2b: the held read on the two sides of every axis-swapping seam agrees, with the one-order read as the failing control. Example: a lake on the seam between two cube faces must not have its shore jog at the seam. | D2 (lines 188-193), §10 test 2 |
| 12 | MEDIUM | O | O's wet hold reads N's /S_b/, so every O edge has a step of k/S_b/ + φL, and O cannot meet K3 | In O, take the wet magnitude from O's own field, k·/G_r − W_b/, so both sides meet W_b on O's line. Restate §5.4 and the O column of §7.3 with the step O still makes (φL wherever the lift is not faded to zero on O's line). Example: a pilot who lands on a ragged O shore of a plain lake must not step off a 30 m ledge into the water. | D5 (line 247), §5.4 (lines 503-506), §7.3 O column |
| 13 | MEDIUM | both | Both variants amend W10 and W15, but the file says N 'keeps the rulings as written' | In §16, list the amendments that BOTH variants make to W10 items 1–2 and W15.1 items 1, 3 and 5, and ask them in §15 (or put them into Q1 and Q2). Take the pocket's far word from the rows (the band), not from B. Remove 'keeps the rulings as written' as a reason that separates N from O. Example: before he chooses, the owner reads Q1 and knows that N and O BOTH stop reading a pond's side from its node's word. | §4.1, §6, §8 reason 1, §16, D6 |
| 14 | MEDIUM | both | From R₁ up, the wet hold's rule reads which rows the host holds (SL10) | State the rule as a function of the mask on both hosts: /S_b/ exactly where the column's 4 × 4 stencil holds two words (the band always holds those nodes, D8), and /B − W_b/ otherwise, whatever tiles a host holds. In slice 3a's drift line, add one rung-10 chunk built with the pilot's tiles held and built again without them. Example: a far chunk of a belt lake must read the same bytes on the shard and on a client that has just landed beside it. | D5 (lines 278-281) |
| 15 | MEDIUM | N | N's small lakes are not round ponds: four in five spill and point an 8 km tongue at their sill, and on flat ground a pond grows wider than today's tile | Before slice 0, print per lake: closed or spilling, depth, rim over the level, the drawn area, and the principal axis's bearing against the lattice. Add G-LAKE-AXIS (the axis bearings against the flat null). Restate G-LAKE-DRAWN's prediction per stand. Put the teardrop to the owner beside Q5 as a picture, not as prose. | §0 example, §4.2 (lines 380-384), §7.3 G-LAKE-DRAWN (620), §14 risk 4 |
| 16 | MEDIUM | both | On flat ground the lake's floor and the fade both end on Ω's node-aligned square, so square halos remain around the plain's lakes | Give the floor's and the fade's reach a law from the ground or from the basin, not from Ω's node square. For example, weight LB_b by the side field over a stated share of the water's depth, or by the lake's own hollow. Keep Ω only as the stencil guard. Add a line for the extent of the floored troughs and of φ < 1 around each lake, with its outline's grid lean against the flat null. | D3 (lines 206-220), D4 (227-230), D5 LB_b (252) |
| 17 | MEDIUM | both | The cost is under-estimated: the per-chunk loop puts every column of a lake country on the candidate path, and the card cost does not exist to compare | Bench the WHOLE rung-7 and rung-9 rings over the plain and the belt (every chunk the ladder asks for), with VD_CLIENT_UNPACED=1, not only 'a pure-sea chunk and a coastal chunk'. Print a histogram of candidates per column and per chunk and CANDIDATES-OVER-4 on the plain before slice 1. State what a column with a fifth water does. Remove the card claim from §8, or state that the card holds no artifact today. | §4.8 (lines 427-436), §5.8, §11 (760-762), §8 reason 4, §7.3 BENCH |
| 18 | MEDIUM | both | The recommendation for N rests on claims the code and the newest ruling do not support | Restate O first (see the O finding). Fix the drawn-water and mirrored-wall defects in both variants. Then write the recommendation again. Quote B8 item 1 word for word in Q1. Price the A0 side word as a new product in §4.10 and §13. In §8, state which slice-0 result for a FAIR O would change the choice. | §8 (lines 657-682), §4.10, §13, §15 Q1 |
| 19 | MEDIUM | both | Slice 0's lines mostly read the law's own side, so the defects that show in the picture cannot turn them red | Define every wet-set line on the DRAWN set: a column's surface under its standing water, which is water_sheet_sided's own rule. Keep the side set as a control, and add a line for their difference with a band of 0. Make the WALL band a quantile with a count, read at rungs 0, 3 and 7. Paint the hillshades from the drawn set. Measure a lake's swap against that lake's own level. | §7.1, §7.3 (lines 603-629), §7.4 |
| 20 | MEDIUM | both | A pilot on a lake shore in N stands on bare 8 km Catmull-Rom ground, and in both variants every water sits in a smooth dish about 300 m deep | State the dish's depth and width in §4.9 and §5.10, with the numbers. Consider fading only the lift and the octaves that can carry a column across the water, and keep the walking-scale octaves (the hold already keeps them on their side). Add a line: the fine RMS within 1 km of the shore against two nodes away. Make the rung-3 close panel of §7.4 part of the owner's question. | D4 (lines 224-238), §4.1, §4.9, §5.1, §5.10 |
| 21 | LOW | both | The fade's slope bound is true only if the first-order distance has a gradient of at most one | State the bound with /∇d_b/ and the largest /f/, or cap the gradient of d_b. Add slice-0 line FADE-SLOPE-MAX per stand: the largest slope that the fade adds. Example: a belt lake's shore must not grow a crease steeper than its own dunes. | D4 (lines 227-238), §18 F18 |
| 22 | LOW | both | The HR5 exit criterion names a gate that is red and omits a Tier-A crate that slice 2 touches | Record the 113 misses as an open HR5 defect, with a DEFERRED entry and the slice that cures them (or cure them before slice 1). State each slice's exit as coverage_gate.py with 0 real misses on every touched Tier-A crate, vd-wire included (and vd-sim if the store changes). | §9 (lines 691-693), §18 F14 |
| 23 | LOW | both | Two kernel cases have no rule: more than four candidates, and a lake cell from the fold with no wet candidate at its centre | State both rules and test them. For example: take the four candidates by the nearest wet node, and count the columns over four on the WHOLE globe in slice 0. Give a folded lake cell the level of the lake whose samples carry its majority. Example: a far cell over two tarns in one pass must draw one of their two levels by a stated rule, never a guessed one. | D1 (lines 169-170), §4.2 (row 11–12, line 374) |
| 24 | LOW | both | The far fold waits for a heights band at login, and slice 2 ships a wire shape that R2 replaces | Ship the side samples, or the per-node counts, derived on the shard now, beside the mask. That is one arm whose content R2 later changes, and no client fold. Or state the login picture and measure its pop. | D6 (lines 285-305), D8 (319-339), §4.8, §13 (795-802), §14 risk 7 |

**WHAT FOLLOWS (2026-09-25).** Two paper rounds found 47 faults; each round's cure added a mechanism the next round
broke. The next round is not a third paper: slice 0 becomes a PROTOTYPE BENCH that paints the DRAWN water (the sheet's own
`s < l` rule) and the drawn ground as top-down maps and oblique hillshades at several rungs, on the six stands, for each
candidate rule, with every line read on the drawn set (a wall, a drowned strip, a square, a pond that appears on the climb,
a shore that moves at a swap); the rule is iterated in the bench, measured, and shown to the owner before any kernel work.

**Example.** The bench paints the plain's forty lakes at rungs 7, 10 and 13 for rule N and for rule O; the owner sees
round dishes or ragged dune-shaped shores, and the census beside each panel says how many columns of dry ground carry
water, how tall the tallest step on the shore is, and which ponds appear between rung 12 and rung 13.
