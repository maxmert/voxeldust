# Verdict — the grid family and the geometry seam

**Lens:** feasibility refutation.
**Target:** `docs/investigation/2026-09-07/01_grid_family.md`.
**Date:** 2026-09-07.
**Result: REFUTED.** The report's recommendation survives. Three of its load-bearing claims do not.

The cube-sphere family is sound, and I confirm it with my own measurements below. The refutation is
narrower and sharper: the report shuts a one-way door on an iteration it never bounded, it writes an
address that cannot hold a hull's own cells, and it leaves out the owner's tree.

---

## 1. What I re-measured, and what stands

I ran my own script. It is not the report's script. The numbers agree.

| Claim | My measurement | Method | Verdict |
|---|---|---|---|
| The tangential cell edge runs 0.708–1.005 m | min 0.708484, max 1.005012 | 400 × 400 samples of the face, arc length `= (4/π)·W'(a)·√(W(b)²+1)/(W(a)²+W(b)²+1)`, MEASURED 2026-09-07 | STANDS |
| A cell at the middle of a cube edge is 0.71 × 1.00 m | 0.708 × 0.992 m | same script, at `a = 0, b = −1` | STANDS |
| The grid directions run 90° to 120° | 90.0000° at the face centre, 119.9999° at the corner | dot product of the two tangent vectors, MEASURED | STANDS |
| `k₁ + k₂ + k₃ == 1.0` exactly in `f64` | the sum minus one is `0.0` | one `f64` add, MEASURED | STANDS |
| The eight corners sit at latitude ±35.264° | `asin(1/√3) = 35.2644°` | arithmetic | STANDS |
| `R = 2N/π` gives an exact 1.000 m cell at the face centre | the arc formula gives 1.000000 at `a = b = 0` | derivation plus the script | STANDS |
| The §2.3 ladder table's radii and errors | every row reproduces: moon −30 m, Ceres −48 m, Luna +310 m, Mars −127 m, Earth −646 m, super-Earth −1,293 m; the chunk counts 40, 47, 43, 42, 39, 39 reproduce | recomputed each row from `N = πR/2`, the snap to `2^(T−1)`, and `T − 1 = ceil(log2(N/3968))` | STANDS, with one label defect (§4, finding 10) |
| `rapier` is not a dependency | `grep -c rapier Cargo.lock` returns `0` | MEASURED | STANDS |
| `noise = "=0.9.0"` is declared and unused | `Cargo.toml:78`; `grep -rn "noise::" crates` returns nothing | MEASURED | STANDS |

I also checked every code citation the report makes. They resolve. Three drift by a few lines:
`RealmId::Area` is `crates/core/src/pose.rs:47`, not 44; `FrameRef::AreaLocal` is `pose.rs:105`, not 101;
the second `Shell` fixture is `crates/core/src/built.rs:180`, not 179. None of that changes an argument.

One code claim I expected to break holds. The doc comment at `crates/core/src/pose.rs:635-638` says no
production code calls `normalize`. That comment is stale: `crates/physics/src/motion.rs:62` calls
`translated`, which normalises inside itself (`pose.rs:624-629`), and `LatticePos::local` is
`pub(crate)` (`pose.rs:572`) while the public metre constructor `from_metres` normalises
(`pose.rs:585-586`). The report's §5.1 row is right and the doc comment is wrong.

No new library is adopted. D7 gives the owner two rapier options and takes neither. That is correct.

---

## 2. The refutation — three load-bearing claims that fail

### Finding 1 — The three-step Newton inverse is not bounded at a production body, and the report shuts a door on it

The report keeps the investigation's *"fixed three-step Newton inverse"* (§1.3) and lists *"the warp:
`VD-ASC5` constants, evaluation order, `const` inverse step count"* as a one-way door that shuts at the
first golden digest (§8). It never states the inverse's FIRST GUESS, and it never bounds the residual.

I measured it. 100,001 samples of `a` in `[−1, 1]`, forward `W`, then the inverse, then the error:

| first guess | steps | worst error in `a` | in CELLS at Earth's `N = 2.0 × 10⁷` |
|---|---|---|---|
| `a₀ = t` | 3 | 2.14 × 10⁻¹¹ | 0.00021 cells |
| `a₀ = t` | 4 | 2.22 × 10⁻¹⁶ | 0.0000000022 cells |
| `a₀ = t/k₁`, clamped | 3 | 1.04 × 10⁻⁸ | **0.104 cells** |
| `a₀ = t/k₁`, clamped | 4 | 2.22 × 10⁻¹⁶ | 0.0000000022 cells |

MEASURED 2026-09-07 by the script above, in `f64`, on this machine.

Two facts follow. First, three steps do NOT reach the `f64` floor; four steps do. So the answer depends
on the first guess, and the report does not say which guess it means. Second, the two natural guesses
differ by five orders of magnitude in the answer. `a₀ = t/k₁` — invert the linear term, the obvious
choice — leaves a tenth of a cell at Earth's size. A cell centre is half a cell from its boundary, so the
round trip survives by a factor of five, and no more. The report's own §5.3 says the address must reach a
42,700 km radius, which is `N ≈ 4.3 × 10⁷`; there the same guess leaves 0.22 cells and the margin is
2.4. The report never computes that margin.

*Example: a pilot lands on the home planet and mines a cell. The shard runs the inverse warp on the
pilot's own position and names cell `(Planet 7, face +Y, tier 0, i 8 471 002, j 6 004 331, k 8 400)`. A
client built with the other first guess names `i 8 471 001`. The player mines the cell beside the one the
crosshair covered. The world-generation tag does not refuse that client, because the crate version is the
same — only the arithmetic inside one function differs.*

**Verdict: UNMEASURED_AS_FACT.** The step count is presented as settled and is written into a one-way
door. **Fix:** state the first guess as part of the frozen warp; measure the worst residual, in CELLS, at
the largest `N` the address admits (`4.3 × 10⁷`); shut the door on four steps unless three are proven
with a stated margin. Add the residual bound to §7 as measurement 1b.

### Finding 2 — The proposed gate cannot see finding 1

`G-MAPPING-EXHAUSTIVE` runs *"Exhaustive at `N = 62`, spot-checked at two production `N`"* (§4.3), and
one of its invariants is `addr_of(cell_center(x)) == x` for every cell.

At `N = 62` a cell is 2/62 = 0.032 wide in `a`. A residual of 1.0 × 10⁻⁸ is 3 × 10⁻⁷ cells there — the
test passes with a margin of 1.5 million (MEASURED: the residual above divided by the cell width). The
exhaustive run therefore CANNOT fail on an inverse that is too coarse for a real planet.

§1.5 says the `N = 62` run is *"a complete proof of the table for every body"*. That is true of the seam
TABLE, which depends on the cube net alone. It is NOT true of the round trip, which depends on `N`. The
report puts both invariants in the same gate and gives the whole gate the table's `N`-independence.

**Verdict: UNMEASURED_AS_FACT.** **Fix:** split the invariants. The seam table, `neighbor` involution and
the corner-loop closure are `N`-independent and belong at `N = 62`. The round trip and `addr_of`'s
`None`-outside-domain arm are `N`-dependent and must run at the largest legal `N`, on cells chosen at the
worst `a` my table names (`a ≈ ±0.85`), not spot-checked.

### Finding 3 — The address cannot hold a hull's own cells

Three statements in the report do not fit together:

- §1.4: `i, j: u32`, `k: u32`.
- §4.2: `IdentityGrid { half_cells: IVec3 }` — the hull's slot box, in whole cells, centred on the
  hull's own frame origin (`FrameRef::ShipLocal`, `crates/core/src/pose.rs:91`).
- §5.2: `addr_of(p) = p.cell() >> 10`.

A hull's cells lie on both sides of its own origin, because the slot box is centred there. The shift
gives a NEGATIVE index for every cell behind or below the origin, and `u32` holds none of them. On a
planet `k` is biased by `floor_radius_m`, which the report states; on a hull there is no floor radius and
the report states no bias.

*Example: an engineer walks to the stern of a hull and places a thruster block one metre aft of the
hull's own origin. The shard shifts the position and gets `−1`. The address field is unsigned, so the
block is filed at `4 294 967 295` — the far bow — or the placement is refused.*

**Verdict: WRONG.** **Fix:** bias the identity arm by `half_cells`, exactly as the shell arm biases by
`floor_radius_m`, and say so in §1.4 and §5.2. It is a one-line change now and a record migration later,
because §8 shuts the address-width door before the first edit is written.

---

## 3. Law conflicts the report declares clean and is not

### Finding 4 — `CornerDefect` is a "tier refusal" seam, and SL8 is signed off without it

§1.2 says a blueprint walk *"returns a typed `CornerDefect` instead of overwriting a cell"*. SL8's eleven
seam kinds include TIER REFUSAL. A refusal to build is a refusal the player sees, at eight known places
on every planet, at latitude ±35.264° (MEASURED above). §6.5's four SL8 bullets cover the terrain field,
the corner landform, the ladder snap and the partial edge chunk. None of them covers the refusal.

*Example: a settler stamps a prefab landing pad on the home planet at latitude 35.3°. The pad lands
everywhere else on the planet and is refused here. Nothing on screen explains why. That is a seam.*

**Verdict: BREAKS_LAW.** **Fix:** either state the refusal as an owner decision with its player-facing
answer (the generator puts a mountain on all eight corners so no pad is ever sited there), or design the
walk to bend across the corner. Add it to §9 as a decision, not to §6.5 as a settled bullet.

### Finding 5 — HR4 names the `FrameSpace` seam; the report renames it without asking

`CLAUDE.md:110` states HR4 as *"capability DAG + `FrameSpace` seam"*. `docs/design/PLAN.md:42` states
*"The geometry seam is a stateful `FrameSpace`"* and *"Permanent CI gate G-IDENTICAL ... (with one
fixture forcing a reanchor)"*. `docs/design/DEFERRED.md:299` records that the reanchor half of D-38 is
the only owed half.

§4.2 replaces that seam with a stateless `GridMapping` and deletes the anchor. The argument is good — the
anchor existed for an `f32` physics world, and there is no `f32` on the position base
(`crates/core/src/pose.rs:548-551`). But HR4 is a HARD RULE, and the report escalates only the GATE SPLIT
(D3). It does not escalate the seam's rename, and it does not escalate the deletion of the
reanchor-forcing fixture that HR4's own sentence names.

**Verdict: BREAKS_LAW (an unflagged conflict, not a wrong design).** **Fix:** add decision D8 — "HR4's
text names `FrameSpace`; this report renames the seam to `GridMapping` and moves the anchor below it. May
the hard rule's wording change?" — and mark `DEFERRED.md` D-38's reanchor half as superseded only after
the owner answers.

---

## 4. What the domain needs and the report leaves out

### Finding 6 — The owner's tree (V2.2) appears nowhere

`grep -c "V2.2" 01_grid_family.md` returns **0** (MEASURED). The requirements table in §1.1 lists V2.1,
V2.3, V2.4, V2.5, V2.8 and SL8. V2.2 is missing, and V2.2 is a geometry requirement: *"they should be
placed as one object/block on the surface … then the collisions are calculated on the server, so server
somehow should know the shape."*

A tree contradicts the report's own rule. §3.1 fixes the 1 m cell as the unit of space-taking and says a
cell is `Empty`, or `Block(shape, orient)`, or `SubGrid`. A tree is one block whose canopy fills twenty
cells. Under §3.1 either the tree takes one cell and its collider leaves that cell — which the report
forbids for a sub-grid and never permits for a block — or the tree takes twenty cells and is not one
object.

*Example: a pilot flies a hull through the canopy of a pine on a moon. The moon's shard owns one cell at
the trunk. The hull's sweep test asks which cells it crossed and gets twenty empty ones.*

**Verdict: MISSING.** **Fix:** add a fourth cell arm, or a "large feature" record anchored at one cell
with a seed-derived shape the shard expands for collision, and name its door. It also raises the case the
report never writes: a tree on a MINED edge — the cell under the trunk is dug out by a diff while the
trunk's shape still comes from the seed.

### Finding 7 — The cell state has no room for smooth terrain (V2.1)

§4.3 puts *"the smooth-surface extractor (V2.1)"* above the seam and gives it
`static_shape(seed, CellAddr) -> Cell`. §3.1 defines `Cell` as `Empty | Block(shape, orient) | SubGrid`.
There is no field in which an isosurface value can live, and none that mining or placing can move
continuously. The owner asked for *"smooth realistic terrain, but still build with voxels"* and for the
terrain to *"change accordingly"* when a voxel is placed. A three-arm occupancy enum gives a Minecraft
surface.

*Example: a prospector fires a mining beam at a hillside on the home planet. Under §3.1 the cell becomes
`Empty` and a 1 m cube disappears from the hill. Under V2.1 the slope should sag.*

**Verdict: MISSING.** **Fix:** state whether a terrain cell carries a density, and if it does, put it in
`CellAddr`'s companion record and name its door beside the sub-cell width. If the answer is the
saved-record report's, say so explicitly; today the grid report claims the extractor and denies it a
field.

### Finding 8 — Sub-metre blocks were chosen without re-reading the warp's own anisotropy

D2 recommends 1/8 m. §3.3 says the collider is *"the union of its micro-boxes … scaled by 2⁻ˢ"* and that
*"the result never leaves the cell's box"*. That is identity-grid language. A planet's cell is a frustum,
0.708 × 0.992 m at the middle of a cube edge (MEASURED above), so a 1/8 m micro-cell there is
0.0885 × 0.124 m and is not a box.

At 1 m the report's own answer holds: *"the player cannot see the change; a survey tool could measure
it."* At 1/8 m it does not. A 29 % compression on a 12.5 cm part is 3.5 cm on a hand rail, held at arm's
length.

The report never says whether a sub-metre block may be placed on a planet cell at all. §3.3's example is
a railing on a hull; §1.1's example is a stone wall on a planet.

**Verdict: MISSING.** **Fix:** state the rule. Either sub-metre blocks are hull-and-station only — an
owner decision, because it removes detail from every base on every planet — or they are planet-legal and
§3.3 must say what a micro-cell's collider is on a frustum cell.

### Finding 9 — A rotating joint has no address

§3.5 reduces V2.5 to *"an attachment is addressed by `(CellAddr, face: u8)`"*. The owner's V2.5 asks for
more than a HUD: *"a rotation joint between two blocks that turns what is built on it by a signal (a
manipulator, a remote-controlled turret)"*. What is built on the joint is a set of CELLS at a continuous
angle to the hull's grid. `(body, face, tier, i, j, k)` cannot express it.

This is a grid-family question, not a record question: a turret is a second grid inside a hull, which is
exactly what HR4 exists to prevent. It touches the D1 door.

*Example: a gunner turns a turret 37° on a hull's dorsal mount. The turret's four armour blocks were
placed on the hull's integer grid. At 37° they sit on no cell of it.*

**Verdict: MISSING.** **Fix:** name the answer before D1 shuts — a turret is its own realm with its own
identity grid and a parent-authored orientation (which the code already supports: a parent stamps
`orient`, `pose.rs:919-925`), or a turret is off-grid rigid geometry. Either answer changes what "one
grid family" means.

### Finding 10 — The §2.3 table's `N` column is the un-snapped `N`

The ladder rule is `N = q·2^(T−1)`. The table prints Earth's `N` as 10,007,543, which is odd and is not a
multiple of 4,096, while the same row's snapped radius (6,370,354 m) comes from `N = 10,006,528` — the
figure the §2.3 example gives three lines later. Every other number in the table reproduces exactly (I
recomputed all seven rows). Only the column is mislabelled.

**Verdict: MISSING (a label).** **Fix:** print both columns, or label the one printed "ideal `N` before
the snap".

### Finding 11 — Determinism is argued about the function, never about the inputs

§5.2's example says *"The client used the same function on the same told position."* The report proves the
function is identical on both hosts. It never establishes that the client HOLDS the server's exact
`LatticePos` bits. One `floor` per tangential axis turns a sub-ulp input difference into a one-cell
disagreement, and the no-drift gate of §7 compares a FIXED SAMPLE of addresses — it cannot see an input
difference at all.

The consequence is bounded and the report is lucky: chunk geometry comes from the address, not from a
pose, so collision and drawing still agree. Only aiming is exposed. But the report states this as covered
by the gate.

**Verdict: UNMEASURED_AS_FACT.** **Fix:** say plainly that the gate proves the FUNCTION and that
targeting agreement is a separate measurement — the server names the cell and the client draws the
highlight from the server's own answer, or the gate must compare a client-held pose with a server-held
pose.

### Finding 12 — No frame budget anywhere

The report chooses the rung rule in D6 (*"top rung = a face is at most 64 × 64 chunks"*) and never states
what that rule costs to draw. From the report's own table: an Earth-sized planet has 39 chunks per face
edge at its top rung, so 6 × 39 × 39 = **9,126 top-rung chunks**, about **4,500 in the visible
hemisphere**, each 62 × 4,096 m = 254 km on a side (all ESTIMATED by me from the report's §2.3 table).

Nothing in the report says on which thread a chunk is generated, how many chunks a landing at flight
speed must produce per second, or what a planet at 10,000 km in the window costs. `addr_of`'s cost is
owed as measurement 3; the meshing budget is owed by nobody.

**Verdict: MISSING.** **Fix:** either state the budget here, because D6 sets it, or name the report that
owns it and record the dependency in §7 and §9.

### Finding 13 — An edit during a crossing is not written

§6.1 rules that an occupant's pose never enters another realm's grid. It does not say what happens to an
edit in flight when the player crosses out of the realm that owns the cell.

*Example: a miner stands on a hull's ramp, aims at the moon's surface, and fires as the hull lifts. The
crossing hands the miner to the hull's realm on the same tick the moon's shard receives the edit.*

**Verdict: MISSING (small).** **Fix:** one sentence — the owning realm refuses an edit whose author is no
longer inside it, on the same fence the transfer already carries.

### Finding 14 — The §2.1 grep list is not complete

§2.1 says the grep *"returns only doc comments: `crates/sim/src/capability.rs:11,18,35`,
`crates/core/src/fence.rs:18`"*. My grep also returns `crates/sim/src/stub/transient.rs:239`, which names
`SphericalSpace` in a comment. The conclusion — no such type exists — is right. The list is presented as
exhaustive and is not.

**Verdict: MISSING (small).** **Fix:** add the line, or say "among them".

---

## 5. What the owner should do with this

The report's recommendation is right and should be promoted, with these changes first:

1. State the Newton inverse's first guess and bound its residual in CELLS at `N = 4.3 × 10⁷` before the
   warp's door shuts (finding 1). This is the only finding that can cost a world-format epoch.
2. Split `G-MAPPING-EXHAUSTIVE` into its `N`-independent half and its `N`-dependent half (finding 2).
3. Bias the identity arm's address, or make it signed (finding 3).
4. Add three decisions to §9: the corner refusal against SL8 (finding 4); HR4's seam rename (finding 5);
   whether a sub-metre block is legal on a planet cell (finding 8).
5. Write the tree, the smooth-terrain field and the rotating joint into the domain or hand each one to a
   named report with the dependency recorded (findings 6, 7, 9).

No new library is needed for any of it. No new data crosses a realm boundary because of any of it.
