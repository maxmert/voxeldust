# 09 — Physics on the terrain (P5): the collider, the character, the hull, the placement, the sweep

**Date:** 2026-09-07. **Revision 2** (after the law refutation and the feasibility refutation).
**Domain:** P5 in the decision board's order.
**Status:** an investigation result for the owner. Not binding until the owner rules and the result
moves into `docs/design/`.

**What binds this report.** CLAUDE.md HR1–HR6 and SL1–SL10; `owner_decisions_2026-09-07_voxels.md`
(SL10 and V2.1–V2.9); the reach ruling (2026-09-02); the suit ruling (2026-09-05); the movement rulings
(2026-08-26, 2026-08-27); SL8 (a seam is a defect). The investigation base (`docs/investigation/`) is an
input only. Where this report and the base disagree, the base is stale and §8 names the stale claim.

**How to read the numbers.** MEASURED means somebody ran a command or read a line of code, and the
report says which. ESTIMATED means arithmetic on stated inputs, and this revision states the inputs.
An ESTIMATED number is never a gate.

**Three words this report keeps apart.** The 2026-09-02 ruling makes REACH a law word, so revision 2
stops using it for three things:

- **reach** — a realm's own stated number, the radius inside which an observer sees it
  (`owner_decisions_2026-09-02_reach.md:66-81`). It rides the slow lane beside mass.
- **the arm's length** — how far a player may place or mine from where the player stands. A placement
  check.
- **the physics ring radius** — how far around a dynamic body the shard builds colliders (§1.4).

---

## 0. What exists in the code today (MEASURED by reading, 2026-09-07; citations re-run in revision 2)

| Claim | Evidence |
|---|---|
| No physics engine is in the tree. `rapier` and `parry` appear in no `Cargo.lock` entry. | `grep -n 'name = "rapier\|name = "parry' Cargo.lock` returns nothing; the only mention is a profile comment at `Cargo.toml:96`. |
| No collider, rigid body or character controller type exists. | `grep -rn "Collider\|RigidBody\|CharacterController" crates --include='*.rs'` returns only the ghost registry (`crates/sim/src/stub/ghost.rs:55`), which holds no collision behaviour and says so (`ghost.rs:9-10`). |
| The movement lane exists: a child states a push and a turn per tick and its body facts on change. | `crates/wire/src/intershard.rs:520` (`ChildDrive`), `:538` (`ChildFacts`); `crates/core/src/built.rs:37-54` (`BuiltFacts`: mass in grams, cross-section, drag, max push, max turn). |
| The parent integrates a driven child: rotate the push, add the realm's pull and drag, semi-implicit step. The pull is one constant vector per realm, never a function of position. There is NO angular term of any kind. | `crates/sim/src/stub/drive.rs:139-182` (`advance_driven`); `drive.rs:112-118` (`Ambient { pull_mps2, density_kgpm3 }` — two fields, no angular velocity). |
| Drag is exactly zero where the medium is zero, with no special case. | `drive.rs:116-118` (the `density_kgpm3` doc), `drive.rs:161-163`. |
| A walking occupant's input still becomes a pose per datagram, through the old governor and ramp. | `crates/sim/src/stub/session.rs:227` → `crates/sim/src/stub/dot.rs:238` (`apply_input`) → `dot.rs:442-486` (`walk`, which calls `ramp_cap_mps` at `:457` and `governed_ceiling_for_frame` at `:465`). This is the placeholder the suit ruling defers (S6). |
| The realm time multiplier dilates the occupant's step and the realm's own tick length. | `dot.rs:475-482` (`move_speed · dt · time_multiplier`); `crates/sim/src/stub/placement.rs:59`. |
| Containment is SWEPT for `Shell` and `Aabb`, on integers; `Obb` is not swept. | `crates/core/src/geometry.rs:1527` (`region_verdict`), `:1545-1553` (the shape fork and the `Obb` arm); the exclusion list `geometry.rs:8-27`. |
| The child index answers a segment, not only a point. | `crates/core/src/child_index.rs:171` (`candidates_segment`). |
| The re-clamp on a transient after a crossing is deleted. | `crates/sim/src/stub/transient.rs:230-236`. |
| The DEV cluster ticks at 50 Hz. The tick rate is a per-shard knob, not one shipped constant. | `crates/bins/src/lib.rs:413-414` (`pub const DEV: DevClusterParams { tick_hz: 50, … }`); `crates/core/src/kinematics.rs:148-149` (*"`tick_hz` is a per-shard knob passed in (a global `TICKS_PER_SECOND` const would be a magic number silently splitting a 10 Hz vs 50 Hz shard)"*). |
| The finest lattice cell is 2⁻¹⁰ m = 0.9765625 mm. | `crates/core/src/pose.rs:255` (`FINE_CELL_EDGE_M`). |
| A planet's mass and radius exist as generator data; surface gravity is a function of both. | `crates/physics/src/taxonomy.rs:750` (`surface_gravity_mps2(mass_kg, radius_m)`), `:653` (`planet_radius_m`). |
| A PLANET REALM'S BOUND IS ITS OWN GRAVITATIONAL SPHERE OF INFLUENCE, not its surface. A moon is a planet with a planet as its parent. | `crates/physics/src/worldgen/generate.rs:1796-1798` (`shape: Boundary::Shell { r: celestial::planet_soi(...) }`); `generate.rs:1793` (*"A MOON IS A PLANET (the ruling, literally): same kind, parent = a planet"*). |
| A planet's DRAWN size is a separate, smaller number. | `generate.rs:1819` (`look: Some(Boundary::Shell { r: taxon.radius_m })`). |
| No planet spins today: the Kepler arm writes position and velocity through `FramePlacement::moving`, which sets identity orientation and zero angular velocity. | `crates/physics/src/motion.rs:49-52`; `crates/core/src/frame.rs:92-101`. |
| The frame machinery ALREADY carries the `ω × r` term at a crossing, both ways. | `crates/core/src/frame.rs:64-66` (`angular_velocity`, doc: *"drives the velocity transform's Coriolis term for rotating frames (planet surface, spinning ship)"*); `frame.rs:251` and `frame.rs:273`. |
| `FrameSpace`, `SphericalSpace`, `SurfaceAnchor`, `AnchorGen` do not exist. Six doc-comment mentions in three files. | `grep -rn "FrameSpace\|SphericalSpace\|SurfaceAnchor\|AnchorGen" crates --include='*.rs'` → `crates/core/src/fence.rs:18`, `crates/sim/src/capability.rs:11,18,35,40`, `crates/sim/src/stub/transient.rs:239`. |
| HR4's structural half is LEDGERED AS OWED for exactly this domain. | `crates/sim/src/capability.rs:13-18`: *"KNOWN LIMIT (DEFERRED [[D-38]]): this is HR4's STRUCTURAL half … The variant forcing a `reanchor()` stays owed at P5 (`FrameSpace` does not exist yet)."* The two-shard-kind fixture that DOES exist is named there: `drive_swept_crossing_feature` (D-PLACE-1). |
| The grid seam is named and has two arms. | `capability.rs:34-42`: `VoxelGeometry::Spherical` / `Cartesian`, *"the ONE seam where spherical planets and Cartesian ship grids differ"*. |
| A worker pool is already in the dependency tree; `rayon` is not. | `grep -n 'name = "bevy_tasks"\|name = "rayon"' Cargo.lock` → `bevy_tasks` at `Cargo.lock:1570`, no `rayon`. No crate uses `bevy_tasks` directly today (`grep -rn "bevy_tasks" crates --include='*.rs'` returns nothing). |
| The noise benchmark is the one measurement in the base: 12.19 ns per evaluation, 0.885 ms per 62³ chunk, Apple M4 Pro. | `scripts/noisebench/src/main.rs:1-13`; `docs/investigation/README.md`. |

**Consequence.** Every physics number in this report is design arithmetic. The P5.1 spike (§7) turns
the load-bearing ones into measurements before any collider code is written.

**A stated assumption this report uses for arithmetic.** The chunk edge is not decided in this domain.
Every "per chunk" figure below assumes a **32 m chunk edge** and says so. If the grid domain picks
another edge, every such figure moves and none of the rules do.

---

## 1. The collider source: the server builds colliders from the SAME surface the client draws

### 1.1 What SL10 and V2.1 change

The investigation base built its collision design on two premises that the 2026-09-07 rulings
remove:

- The base said generated terrain is `Cube`-only at every tier (`block_system_design.md` §4.9.4),
  so physics only ever met one-metre cells and the planet lane was a greedy-box decomposition of cube
  cells (§4.6.1). **V2.1 says the terrain is smooth.** A hillside is a continuous surface extracted
  from a density field over voxels, not a staircase of cubes. The greedy-box terrain lane has no
  surface to decompose.
- The base said terrain never crosses the wire and the client only renders what the server ships.
  **SL10 says the client derives the static shape from the seed with the ONE generator crate, and the
  server computes collision on that same shape (clause 5).** So the collision surface and the drawn
  surface come from one function on both hosts.

So the question is not "which of four collider lanes" but "which rapier shape holds a smooth,
deterministic, per-chunk surface, and what tolerance does that give a boot".

### 1.2 The three candidates, compared

| | **A. Trimesh from the same extractor** (recommended for terrain) | **B. Heightfield** | **C. Voxel / SDF collider** |
|---|---|---|---|
| Identity with the drawn surface | **The generator's canonical output is identical; each host then builds its own structure from it.** See §1.5 for the exact tolerance, which is NOT zero. | Not exact. A heightfield is a function of (x, y). A cube-sphere face is not a height function at the face edges, and a cave, an overhang or a mined tunnel cannot be a heightfield at all. | Not exact. The base says parry's `Voxels` is an axis-aligned cube lattice that cannot express the cube-sphere shear (`block_system_design.md` §4.6.1), and that parry has no signed-distance-field shape. **BOTH CLAIMS ARE UNMEASURED** — neither crate is in `Cargo.lock` (§0) — and §7 row S15 reads the pinned version's shape list before either decides anything. If they hold, a voxel collider under a smooth drawn surface puts the boot on a staircase the player cannot see: a detail-by-box seam (SL8). |
| Cost per chunk, **f32 flavour** (ESTIMATED at a 32 m chunk with one surface crossing, ~1,000 vertices and ~2,000 triangles) | vertices 12 KB, indices 24 KB, BVH ~100 KB ⇒ **~140 KB** | ~32² samples × 4 B = 4 KB. Cheapest, and wrong on a sphere. | ~1 B per voxel ⇒ 32 KB per chunk; cheap, and wrong under a smooth surface. |
| Cost per chunk, **f64 flavour** (the recommended one — ESTIMATED by doubling every float field of the row above) | vertices 24 KB, indices 24 KB, BVH ~200 KB ⇒ **~250 KB** | — | — |
| Build time per chunk | ESTIMATED 0.2–0.5 ms on top of the extraction, by analogy with the base's cube remesh. **UNMEASURED.** Every downstream figure in §5 and §7 inherits this estimate and says so. | — | — |
| Memory per active occupant's physics ring (ESTIMATED) | A lone walker holds the surface chunks its physics ring touches: at a 32 m chunk, a 2 m arm's length and a walking speed, the ring is one chunk edge, and a capsule near a chunk corner touches at most 8 chunks ⇒ **1.0–2.0 MB in f64**, 0.6–1.2 MB in f32. | — | — |
| Placed blocks | Not this shape. A placed square block keeps its own convex collider (§1.3). | — | The candidate for the 1 m cube lane on a HULL (flat grid), if S15 confirms the base. |
| Sub-metre blocks | Not this shape. | — | Per-cell compound of boxes at the sub-metre grid (§1.3). |
| Rapier's own warning | Internal-edge snagging on a trimesh. `TriMeshFlags::FIX_INTERNAL_EDGES` plus the controller's skin; the chunk-boundary walk gate (§7 row S4) proves it. | — | — |

**Recommendation: A for the terrain surface, with the base's cube and shape lanes kept ONLY for placed
blocks.** The terrain is one `TriMesh` collider per chunk, built from the extractor's output by the
realm's shard, static in the body-fixed surface frame, and entered into the realm's rapier world under
the rule §2.5 decides.

**Example.** A player walks up a hill on a moon. The client evaluated the moon's seed for that chunk and
drew the slope. The moon's shard evaluates the same crate for the same chunk address, gets the same
canonical vertex list, and builds a `TriMesh` from it. The boot stands on the triangle the player sees,
to within the tolerance §1.5 states.

### 1.3 The four collider arms

A cell is either a terrain-field cell or a built cell. It is never both. A seed-placed FEATURE is a
third thing and needs its own arm, which revision 1 did not have.

- **A terrain-substance voxel placed on a planet** joins the density field. The extractor reshapes the
  surface around it (V2.1). Its collider is the reshaped trimesh; there is no separate block collider.
  *Example: a player stacks three dirt voxels on a slope; the hill grows a bump, and the boot walks over
  the bump.*
- **A building block on a hull** (Cartesian grid): the 1 m cube lane is one collider per chunk from the
  base's voxel shape, and the ~20 shapes are `Compound` parts from a baked template table, `Arc`-shared
  per variant (base §4.6.1, still valid on a flat grid, subject to S15).
- **A building block on a planet** (cube-sphere grid): one convex part per cell over the cell's eight
  warped corners, as the base's spherical shaped-cell lane. The base's own arithmetic still holds: the
  chord error of a warped 1 m cell against the true bowed face is far below one FINE cell. (The
  base's 8-cell greedy box cap was for cube TERRAIN; it is no longer needed for terrain.)
- **A sub-metre block** (V2.4) sits inside one 1 m cell on a sub-grid the grid domain fixes. Its
  collider is a box part in the cell's compound, greedy-merged with its sub-metre neighbours inside the
  same cell. Below the smallest catalogue extent there is no collider.
- **An attachment** (V2.5: a HUD, a joint) takes no space and has no collider. It is a raycast target
  for interaction only, never a contact. **A joint that MOVES what is built on it is not an attachment
  for physics purposes — see §3.5.**
- **A SEED-PLACED FEATURE (V2.2: a tree, a boulder, a rock arch)** is the fourth arm, and revision 1
  had no arm for it. SL10 rule 1 already names it: the static shape includes *"the geometry of
  seed-placed features (a tree's trunk and canopy from its seed parameters)"*
  (`owner_decisions_2026-09-07_voxels.md:20-25`). So the rule is the same rule as the terrain's:
  1. The feature's shape is a function of `(seed, address)`, evaluated by the ONE generator crate.
  2. The client derives the drawn tree from it. The shard derives the COLLISION shape from it: a
     capsule for the trunk, and a shape the catalogue names for the canopy — the feature's own
     parameters pick both. The generator emits the collision shape as a small canonical record, so the
     server never re-derives it from the drawn mesh.
  3. The features of a chunk are demanded, built, cached and evicted **with that chunk**, by the same
     physics ring (§1.4). They are not a separate lifetime.
  4. Felling a tree, or mining the terrain cell under its root, is a LIVE-STATE change. It crosses as a
     one-hop diff from the owning realm (SL10 rule 6), the same as any edit, and the chunk's feature
     colliders rebuild with the chunk.
  *Example: a player lands a hull in a forest on a moon. The moon's shard holds the ground trimesh and,
  for the same chunks, a hundred trunk capsules. The landing legs stop on a trunk, and a player who
  walks into a pine he can see is stopped by it.*
- **The density field sees a built cell as fully solid.** So the smooth surface meets a placed cube's
  faces without a gap. The trimesh and the cube's collider touch at the shared boundary. This is the
  "cube next to a slope" seam the base flagged as unvalidated (§4.6.2); §7 keeps it in the spike.

**Support after the fact — a rule revision 1 did not state.** A player mines the terrain cell UNDER a
placed block. In v1 the block does not move, does not fall and does not refuse the mine: SUPPORT IS A
PLACEMENT-TIME CHECK ONLY (§4.1's list), never a running simulation. There is no settling pass in the
foundation. *Example: a player digs out the hillside under his own landing pad; the pad hangs, and the
boots still walk on it.* A structural settling pass is a large commitment with its own tick cost, and
§6 decision 9 puts it to the owner rather than assuming it.

### 1.4 The collider cache rule (SL9: cost grows with occupants and edits, never with the planet)

1. **Colliders exist only around dynamic bodies.** Each dynamic body in a realm demands the chunks its
   PHYSICS RING touches. The ring radius is `max(arm's length + body extent, |v| · dt · k_lead)` with a
   floor of one chunk edge, where `k_lead` covers the collider build latency in ticks (M-B's shape:
   grow the radius with speed, never cap the speed). The ring is a swept ring: it covers the line from
   the last pose to the current pose plus the lead (§5).
2. **THE RING IS ALTITUDE-AWARE.** A chunk is demanded only if the swept tube touches it AND the chunk
   contains a surface crossing. The generator answers the second question from a coarse density
   bracket, without extracting anything. Revision 1 omitted this and then costed orbital flight as if
   every chunk of a 280 m tube held terrain; §5 restates the arithmetic. *Example: a hull in a 100 km
   orbit over a moon demands no collider at all, because no chunk of its tube holds the moon's surface.*
3. **One collider per chunk per realm, reference-counted.** Two walkers in one chunk share one collider.
   This deletes O38's "cluster merge/split rule": there are no per-body colliders to merge, so no cell is
   ever in the rapier world twice and no contact impulse doubles.
4. **Build on demand, from the shape plus the diff.** The shard evaluates the generator for the chunk,
   applies the chunk's edit diff from the edit store, extracts, builds. An edited chunk with no occupant
   nearby costs storage only (the diff) and no collider.
5. **Evict on last leave, after a cooldown.** The cooldown is the same hysteresis idea as realm
   liveness: a walker pacing across a chunk edge must not rebuild the chunk every tick.
6. **The build runs OFF the tick thread — see §5.3.** A 0.5 ms build inside a 20 ms tick is a tick
   hitch, which SL8 names as a seam.
7. **The cost model, stated so it can be measured:** collider bytes ≈ (distinct surface chunks under all
   rings) × (bytes per chunk) + (feature colliders in those chunks) + (child-realm shells, §3.4). It
   depends on how many occupants there are, how far apart they stand, and how many child realms the
   realm holds. It does not read the planet's radius anywhere. §7 measures this on a 500 m asteroid and
   an Earth-sized body with the same occupants and demands equal bytes.

**Example.** One hundred players stand in one spaceport. Their rings overlap. The realm holds about the
same eight chunk colliders it would hold for one player. One hundred players spread over a continent
hold about eight hundred. A million players' worth of dug tunnels on the far side of the planet cost the
realm nothing until someone walks there.

### 1.5 The tolerance, stated physically — and the vertex rule this report PROPOSES

**Revision 1 said the geometric error is "zero by construction" and attributed the rule to SL10 rule 4.
Both refuters are right, and both halves are withdrawn.**

- **SL10 rule 4 says nothing about a vertex lattice and nothing about an index order.** MEASURED: I read
  it at `docs/design/owner_decisions_2026-09-07_voxels.md:36-40` and at `CLAUDE.md:227-236`. It names
  five things — integer hashing, fixed evaluation order, no fast-math flag, no FMA contraction, no
  platform transcendental call — and none of them is a vertex rule. The FINE-lattice vertex rule and
  the fixed index order are **this report's proposal**, and §6 decision 10 gives them their own one-way
  door.
- **"Zero by construction" is an argument, and SL10 rule 3 says no-drift is a MEASUREMENT, never an
  argument.** The claim is withdrawn. What replaces it is a chain of three statements, each with its own
  error and its own gate:

| Link | What it says | Error | Gate |
|---|---|---|---|
| 1. The generator's canonical output | Both hosts evaluate the same crate for the same chunk address and get the same canonical record: FINE-lattice integer vertices in a fixed index order, plus the feature records | **Zero differing bytes, MEASURED**, or the gate is red | S2 |
| 2. The extractor's snap | A surface crossing between two density samples lands BETWEEN two FINE lattice points. The extractor must snap it. The snap moves the surface | ESTIMATED at up to **half a FINE cell, 0.488 mm**, in the direction normal to the surface | S2c |
| 3. Each host's own structure | The shard builds a rapier `TriMesh`; the client builds whatever vertex buffer its engine wants (V2.8 says the engine may be Unreal). Two different structures never share bytes | The host's float type decides it. In f64 the step at 6,371 km is ~1 nm (ESTIMATED). **In f32 the step at 6,371 km is 0.76 m** (ESTIMATED: 6.371 × 10⁶ × 2⁻²³) | S2b |

- **THE CHUNK-LOCAL ORIGIN RULE, which revision 1 was missing.** Both hosts re-express a chunk's FINE
  integers relative to a CHUNK-LOCAL origin before handing them to a solver or to a renderer. A GPU
  pipeline is f32 on every engine, so without this rule the drawn hill and the collided hill disagree by
  most of a metre at a planet's radius, and the boots float or sink. With it, the f32 step inside a 32 m
  chunk is 32 × 2⁻²³ = 3.8 µm (ESTIMATED), far below one FINE cell. This rule is not optional and it is
  not a rendering detail; it is what makes link 3 small.
- **The boot rests one skin above the triangle.** The controller's skin is a constant of the CHARACTER
  record, not of the terrain. The base derived 1/64 m = 15.6 mm and called it f32 quantisation at a
  4,096 m anchor; **that derivation does not hold** — the f32 step at 4,096 m is 4,096 × 2⁻²³ = 0.49 mm
  (ESTIMATED), which is 32× smaller. The base's number has some other origin, so revision 2 does not
  carry it.
- **THE TOLERANCE, and what still has to be measured before it is a gate.** The design target is: a boot
  never sinks below the drawn surface by more than one FINE cell (0.977 mm, MEASURED constant
  `pose.rs:255`) and never floats above it by more than the skin. **Every contact solver keeps its own
  allowed penetration (its "slop") to stay stable, and it is of the same order as a millimetre.** So
  S3's pass number cannot be set until the solver's slop is read off the pinned version and stated as an
  input. The gate sits ABOVE the solver's slop plus the snap error of link 2, or it is red for the
  engine's default and not for the design.
- **A placed cube's collider is exactly the cell.** The base's render-side rim warp (≤ 0.20 m inward,
  §4.9.4) stays render-only, so the visual hull is a subset of the collision hull: a chipped-looking
  edge still holds you up, never the reverse.

### 1.6 The library, said plainly

rapier3d is the engine CLAUDE.md names ("rapier state is checkpoint-carried"), and the owner's tech
memory names it for the client greenfield. It is NOT in `Cargo.lock` (§0). Adding it is the owner's
word, not this report's. Two flavours exist and the choice is a one-way door (§6, decision 1):

- **rapier3d (f32)** needs a local anchor per collider cluster so a 6,371 km planet's colliders sit
  near an origin (the `FrameSpace`/`reanchor()` seam that D-38 and D-41 owe and that does not exist
  today, `capability.rs:13-18`).
- **rapier3d-f64** can hold every body of a planet realm in the realm's own frame: at 10⁷ m the f64
  step is ~2 nm, far below one FINE cell. The physics side then needs no anchor and no reanchor at all.
  The cost is ESTIMATED at 1.5–2× per step against f32, and the per-chunk collider is about 1.8× the
  bytes (§1.2). Both are UNMEASURED.

The recommendation is rapier3d-f64. It deletes the ANCHOR half of the owed seam, and the cost is
measurable in P5.1 before anything is built on it. **What f64 does NOT delete is the GRID half** — see
§2.7, which revision 1 owed to HR4 and did not pay.

No other library is proposed. The worker pool §5.3 needs is `bevy_tasks`, which is already in the
dependency tree through `bevy_ecs` (`Cargo.lock:1570`) and is used by no crate today. `rayon` is NOT in
the tree; if the owner prefers it, it is a new library and needs the owner's word.

---

## 2. The character controller on a cube-sphere

### 2.1 Gravity toward the centre from the realm's authored mass

- The planet realm is centred on itself (SL1). An occupant's pose is in the planet's own frame, so the
  vector from the origin to the occupant IS the radial. No placement is needed to know "down".
- The realm's pull at a position is `g(r) = G · M / r²` toward the origin, with `M` the body's authored
  mass. `taxonomy.rs:750` already computes the surface value from mass and radius. Today
  `Ambient.pull_mps2` is one vector per realm (`drive.rs:115`). It becomes a function of position:
  `pull_at(pos)`.
- The same function serves a hull in low orbit and a boot on the ground. No kind test.

**Correction from revision 1.** Revision 1 called this *"the only change to the parent-side
integrator"*. That sentence is true ONLY if the realm's frame does not rotate. §2.5 is where that is
decided, and under one of its three options `advance_driven` gains three more terms. The sentence is
withdrawn until decision 7 is answered.

**Example.** A hull descends toward a moon. At 200 km up the moon's pull is weaker than at the surface;
the moon's shard computes both from the same mass and the same formula. The pilot inside feels nothing
change in the code, only in the fall.

### 2.2 Up per position

- `up = normalize(pos)` in the planet's frame, re-read every tick. The controller's capsule and the
  camera's frame follow it. Smoothly, never snapped: the base records the blocky-planet author's
  stutter from per-frame snapping (addendum 1 A.1), and SL8 calls that stutter a seam.
- On a hull (a flat grid), `up` is the hull's deck normal. O38 said nobody declares one. The character
  record does not need it: the realm states it (a realm authors how it looks and what its floor is,
  SL3). A zero-g interior states "no deck": the walk becomes a suit push (the suit ruling S2).

### 2.3 Slopes on smooth terrain (this is new)

The base derived every slope threshold on the premise that natural terrain has only 0° and 90°
surfaces, so thresholds "only ever judge player-built ramps" (§4.7, C1). Under V2.1 natural terrain
has every slope, and the climb and slide angles are gameplay on every hillside.

- **The thresholds are character data, never constants.** A character record carries `max_slope_climb`
  and `min_slope_slide` (and the capsule radius, height, step height, skin, jump clearance), the same
  way a hull carries its `BuiltFacts` and a suit will carry its rating. Two characters may differ; a
  suit or a boot upgrade may change them. This is the no-magic-numbers rule applied where the base had
  a derived constant.
- **Climb and slide stay separate, with a gap.** Equal thresholds give the sticky-then-slippery
  flip-flop the base named. The gap is derived from the normal error, and the normal error now has a
  stated source: the extractor's snap of link 2 in §1.5. P5.3 states the gap after the extractor exists.
- **The warp ratio `w` on placed shapes still holds** (base §4.7's table: a `Wedge` spans 43.55° to
  56.10° across a planet). It binds only placed shapes; terrain slopes are what the extractor says.

**Example.** A player walks up a 40° meadow on the starter world and slides back down a 60° scree
face. The same character in a hull walks up a 45° built ramp everywhere on the hull, because the hull's
`w` is exactly 1.

### 2.4 Step height with sub-metre blocks

- Autostep clears the largest sub-metre step a player expects to walk (a half slab, 0.5 m) and never a
  full 1 m block: the base's 544 FINE = 0.53125 m stands. The minimum tread of 0.25 m stands.
- Smooth terrain makes no steps of its own. A mined cell reshapes the surface (V2.1) into a pit with
  sloped sides, so the walk out of a one-cell pit is a slope test, not a step test.
- The controller runs against the chunk's colliders — the terrain trimesh, the built compound and the
  feature shapes — in one query; it does not know which one it touched.

### 2.5 The realm's frame when the body spins — REVISION 1'S DECISION 7 IS WITHDRAWN

**What revision 1 said, and why it is wrong.** Revision 1 stated *"the planet realm's own frame is
BODY-FIXED"*, then measured the fictitious force at ONE place — the equator, `ω²·r` = 0.034 m/s² — and
made "ignore it in v1" a realm-wide rule. Both refuters refuted this, and the code agrees with them:

- **A planet realm's frame reaches to its SOI, not to its surface.** `generate.rs:1796-1798` writes
  `shape: Boundary::Shell { r: planet_soi(...) }` (MEASURED). The realm's frame is the frame in which
  the planet authors every child's placement, out to that bound. A moon is one of those children
  (`generate.rs:1793`, MEASURED).
- **The force the report ignored is not small out there.** ESTIMATED from Earth's rotation rate
  ω = 7.2921 × 10⁻⁵ rad/s:

| Place inside the planet realm | Centrifugal `ω²·r` |
|---|---|
| The equator, 6,371 km (revision 1's only case) | 0.034 m/s² |
| The Moon's orbit, 384,000 km | 2.04 m/s² |
| Near the Hill-class SOI, ~1.5 × 10⁹ m | **~8.0 m/s², about 0.8 g** |

- **Coriolis never appeared in revision 1 at all.** `2·ω·v` is 0.036 m/s² for a hull at 250 m/s
  (ESTIMATED), which is about 66 m of sideways error over a one-minute descent, and 1.0 m/s² at
  7 km/s. `advance_driven` has no such term: it sums exactly three (`drive.rs:159-163`, MEASURED), and
  `Ambient` carries no angular velocity (`drive.rs:112-118`, MEASURED).
- **Revision 1's own example refuted it.** It said a moon's placement moves 20 m per tick. In a
  body-fixed frame the spin alone sweeps that placement 560 m per tick (ESTIMATED:
  7.2921 × 10⁻⁵ × 3.84 × 10⁸ × 0.02 s). The band-sizing rule `band > v_rel · dt · K_SAFETY`
  (`geometry.rs:8-9`, MEASURED) would then have to cover an apparent speed no engine made.
- **The code is right at the boundary and would be wrong inside it.** `transfer_frame` already carries
  `ω × r` both ways (`frame.rs:251`, `frame.rs:273`, MEASURED). So a hull's velocity converts correctly
  at the shell and then integrates wrongly for every tick after — which shows up as a jump at the ruler
  switch on the way out. SL8 names that a seam.

**What the report now puts to the owner (decision 7, rewritten).** Three lawful options. Each names the
DYNAMICS frame (the frame the rapier world integrates in, and the frame the parent authors children's
placements in) and the COLLIDER frame (the frame the terrain is static in).

| | Dynamics frame | Terrain collider | What it costs | What it deletes |
|---|---|---|---|---|
| **A. Recommended — one non-rotating frame, the terrain is a kinematic rotating set** | Non-rotating, centred on the body, axes fixed | A kinematic body carrying the realm's own spin (`ω`, phase from the seed); the trimeshes are static ON it | A boot standing on Earth's equator moves 465 m/s in the realm's frame — 9.29 m per tick at 50 Hz (ESTIMATED) — so the physics ring must sweep with it, and the character controller must hold contact against a fast kinematic surface. **UNMEASURED**, and rapier's CCD against a MOVING trimesh is a known limitation that S16 must read | Every fictitious term. `advance_driven` keeps its three terms plus contact. The band sizing sees true relative speed. A spinning station gets its artificial gravity from a floor that pushes on the feet, which is the real physics, not a fake outward `pull_at` |
| **B. One body-fixed frame, with the terms added** | Body-fixed (rotates with the body) | Static, no motion at all | `Ambient` gains an angular velocity, and `advance_driven` gains centrifugal `ω×(ω×r)`, Coriolis `2ω×v` and the Euler term. The band sizing must then cover apparent speeds up to 560 m/tick for a moon (ESTIMATED). A circular orbit inside the realm is integrated in a spinning frame, so its ground track is only as good as those three terms | The fast-walker problem: the boot is at rest and the terrain is at rest |
| **C. Split the frames at an altitude** | Non-rotating above the split, body-fixed below | Static below the split | A NEW crossing at the split, with its own band, its own hysteresis and its own conversion. SL8 says a seam is a defect, and this is a new one | Both problems, at the price of a new mechanism |

**The recommendation is A**, because it adds NO new frame, NO new crossing and NO new integrator term:
it only makes the terrain collider a kinematic body, which rapier already has. Its two risks are named
above and both get a gate (S16 and S17). If S16 says rapier cannot sweep a fast body against a moving
trimesh, the recommendation falls back to B with the three terms written out, and §6 decision 7 says so.

**What is NOT in doubt.** Whichever option wins, the occupant's pose changes only by the occupant's own
physics inside the realm; the gateway composes the picture from the planet's composed placement plus the
occupant's own-frame pose; and the occupant is never told its parent's placement (SL1 clause 4). And the
question is not blocking today: no planet spins (`motion.rs:49-52`, MEASURED). It blocks D-MOVE-4.

**Where the spin datum lives, under option A.** The spin is the realm's OWN statement, not the parent's
placement orientation — a realm authors how it looks and how its floor turns (SL3). The client already
receives the planet's composed placement, and the drawn chunk vertices turn with the realm's stated spin
phase; no new data crosses.

**Example.** A moon orbits a planet at 1 km/s. A player stands on the moon. The moon's shard steps the
player's capsule against the moon's terrain in the moon's own frame. The planet's shard moves the moon's
placement 20 m per tick, and only 20 m, because the planet's frame does not turn. The gateway adds the
two, and the client draws the player standing on a moon that sweeps across the sky.

### 2.6 Subjective time

The realm's rapier world steps with `dt = tick_dt · time_multiplier` (the multiplier the occupant walk
already applies, `dot.rs:475-482`, and the placement pass already applies, `placement.rs:59`). Gravity,
the fall, the walk and every contact dilate together inside the realm. The parent-authored orbit does
not read the multiplier. This is the standing law with rapier substituted for the hand-written step.

### 2.7 HR4: the second shard kind, the fixture, and what happens to `FrameSpace` — the debt revision 1 did not pay

HR4 says a feature *"passes the identical fixture on ≥2 shard kinds (G-IDENTICAL) or it doesn't land"*
(`CLAUDE.md:110-111`). Revision 1 named neither the shard kind nor the fixture, and it proposed deleting
a seam the code has ledgered as HR4's owed structural half without saying so. Both are repaired here.

- **The second shard kind is a HULL REALM** — `VoxelGeometry::Cartesian` against a planet realm's
  `VoxelGeometry::Spherical` (`capability.rs:34-42`, MEASURED: *"the ONE seam where spherical planets
  and Cartesian ship grids differ"*).
- **The G-IDENTICAL fixture is ONE body run twice: place a block, then walk over it.** One identical
  fixture places a block at a stated grid address, then walks a capsule from a stated start to a stated
  end across that block, and records the contact heights. It runs once on a planet realm and once on a
  hull realm. Only the realm's stated geometry may differ. This follows the shape the code already has
  for the crossing feature: `drive_swept_crossing_feature` is named at `capability.rs:16-17` as the
  existing two-shard-kind run (D-PLACE-1).
- **What f64 deletes, and what survives.** `capability.rs:13-18` says the `reanchor()` variant is owed
  at P5. Under the f64 recommendation the **ANCHOR half of `FrameSpace` dies**: no cluster needs a local
  origin, so nothing re-anchors and `AnchorGen` never exists. **The GRID half survives and is the seam**:
  a cube-sphere chunk address with warped cell corners on a planet, a flat chunk address with square
  cells on a hull. Everything above it — the ring, the cache, the controller, the placement checks —
  is one machinery. If the owner picks f32 instead, the anchor half must be BUILT, and it is a new
  mechanism, not a deletion. §6 decision 1 carries this consequence.

---

## 3. Hulls and vehicles on terrain

### 3.1 A landed hull is an occupant of the planet realm

- The hull is a child realm of the planet. The planet's rapier world holds ONE dynamic rigid body for
  the hull. The hull's shard keeps simulating its interior: the crew walk on the hull's floors in the
  hull's own rapier world, and never enter the planet's world (SL2).
- **ONE integrator, not two — a gap revision 1 left open.** On a realm with a rapier world, RAPIER is
  the integrator. `advance_driven`'s three terms become the forces the parent feeds the hull's rigid
  body each tick: the child's own-frame push, rotated by the parent (the parent is the only one who
  knows where the nose points); the realm's `pull_at(pos)` as gravity; drag from the hull's
  `BuiltFacts` and the realm's density. The numbers and the order are `advance_driven`'s
  (`drive.rs:139-182`); the solver is rapier's. `advance_driven` stays the integrator on a realm with no
  rapier world, which every stub realm is today. **The risk this creates is a path that bends at a
  crossing**, because semi-implicit Euler and rapier's integrator are not the same. §7 row S18 measures
  it: a hull crossing from a stub realm to a rapier realm under a constant drive, and the path's
  curvature must not step at the shell.
- The planet never sets the hull's speed (A4). Rapier's contact solver adds contact impulses, which are
  forces the world applies. That is lawful: the ceiling leaves the flight path (M-D); the ground does
  not.

### 3.2 What the planet needs that it does not hold: the hull's exterior shell (SL6 ask)

The planet holds the hull's `bound` (`built.rs:76`) and `look` (`built.rs:78`) and its `BuiltFacts`
(`built.rs:80`). It does not hold the hull's SHAPE. A hull resting on its bounding box floats
above a hill and sinks into a hollow, its landing legs never touch, and a lopsided hull rests level.
Each of those is a seam a player sees.

**The ask (data, direction, why, cost of doing without) is in the structured summary and §6.3.** In one
line: the hull states its EXTERIOR touchable shell — a convex decomposition of the outside of the hull,
in the hull's own frame — on change (rebuild), on the reliable slow lane beside mass and reach. The
exterior only: a parent can touch a hull only from outside, so internal voids never cross, and O28's
"free hull scan at a player-owned station" is closed by construction.

A second ask rides the same lane: the hull's centre of mass and inertia tensor. M2a said collisions
need mass; a contact on a slope also makes a torque, and rapier needs the inertia to answer it.

**The stated shell covers a moving part's SWEPT envelope, not its current pose** — see §3.5.

### 3.3 The crossing on landing and take-off — corrected

**Revision 1 said the planet takes the hull "a kilometre up". That is wrong.** A planet realm's bound is
its SOI (`generate.rs:1796-1798`, MEASURED), which for an Earth-class planet is ESTIMATED at about
1.5 × 10⁹ m. So:

- **There is NO crossing at the ground.** The hull became the planet's child at the SOI, hundreds of
  thousands of kilometres up, long before the atmosphere. The whole descent, the landing and the
  take-off happen INSIDE one realm, with one authority, one integrator and one collider world. This is
  simpler than revision 1 claimed and it removes a seam nobody has to build.
- **The crossing that does exist is at the SOI**, and it is airborne by definition. The swept
  containment test (`geometry.rs:1527`) reads the line the hull travelled; the system hands the hull
  down one hop. From the next tick the hull sends its six numbers to the planet.
- **What crosses: nothing new.** The placement re-frames through `transfer_frame` as today. The hull's
  shell and facts arrive on the slow lane after the crossing exactly as `ChildFacts` do today
  (`drive.rs:159-166`: a child without facts coasts and is pushed until they land). No occupant pose
  crosses. The crew stay in the hull realm.
- **The consequence for the collider ring.** Because the hull is the planet's child from the SOI, the
  ring is alive for the whole descent, and §5's arithmetic covers a fall from any altitude.

### 3.4 A realm with many children — the SL9 rule revision 1 was missing

SL9 says a parent may hold six hundred ships and stations and forbids *"a per-tick walk of all of
them"* (`CLAUDE.md:214-225`). §3.1 puts every landed child realm's shell into the parent's rapier world
as a rigid body, and a rapier world steps every body it holds, every tick. Revision 1 stated no rule and
measured no case.

**The rule.** A child realm's shell body SLEEPS the moment its authored velocity and its contact set
stop changing, and it wakes on a contact, on a stated drive, or on a stated shell change. A sleeping
body is not stepped, and its collider stays in the broadphase only. The per-tick cost must therefore
grow with the number of MOVING children, never with the number of parked ones.

**Example.** Six hundred hulls sit parked at a spaceport on a moon. Each stated its exterior shell to
the moon once, on the slow lane. The moon steps none of them. A pilot fires one hull's thrusters; that
one hull wakes, and the ones its shell touches wake with it.

**The gate is S12**, and it is the one that can fail: 1 / 100 / 600 parked hulls in one realm, with the
per-tick core time demanded not to grow with the parked count.

### 3.5 A rotating joint, a turret and a piston (V2.5) — the case revision 1 did not name

The owner asked for *"a rotation joint between two blocks that turns what is built on it by a signal (a
manipulator, a remote-controlled turret)"* (V2.5). Two rules of this report collide on it: §2.5's
requirement that structure is static in its collider frame, and §3.2's "state the shell on change".
A turret that turns would restate the shell every tick, which is a lane flood — SL8's eleventh seam.

**The rule, so neither breaks.**

1. **Inside the hull's own world**, a turret is a KINEMATIC part of the hull's collider set, whose
   isometry the joint's signal drives. It is not a separate realm and it is not an attachment: it holds
   blocks, so it holds colliders. This is the same mechanism §2.5 option A uses for a spinning planet's
   terrain: a static set on a kinematic body.
2. **Toward the parent**, the hull's stated exterior shell covers the turret's SWEPT ENVELOPE — the
   volume the turret can reach through its joint's whole range — and not its current pose. The envelope
   changes only when a player BUILDS on the turret, which is a rebuild, which is exactly the "on change"
   rate §3.2 already asks for. Nothing crosses per tick.
3. The cost of the envelope is that a parked hull's outside is slightly larger than the metal. A block
   placed into the swept volume of a turret is refused. That is the right answer anyway: the turret
   would have hit it.

**Example.** A player builds a manipulator arm on his hull and parks on a moon. The moon holds one shell
that includes the whole arc the arm can swing through. Another player cannot place a block inside that
arc. The arm swings inside the hull's own world all day, and the moon hears nothing.

### 3.6 Vehicles

- The base's finding that a one-metre staircase makes wheels impossible (`collidable_decoration.md`
  §8.5) is gone with V2.1: a smooth surface is what a wheel wants. The base's V-R6 (a generator field
  for a smooth surface skin) is what SL10 and V2.1 already make the default.
- A wheel is a ray or a shape cast against the same trimesh; the base's swept-sampling rule (a wheel
  crosses cells between ticks) stays. Suspension compliance stays the right answer to sub-vehicle
  geometry under a 100–150 ms buffer.
- A vehicle is a hull (a realm) or an entity of the planet realm. Which one is a P8 decision and does
  not change the collider.

---

## 4. Placement validation

### 4.1 A block where another player stands

- The client ships the resolved target (cell, face, shape, orientation, nonce) in the frame of the
  realm that owns the cell. The server validates (the base's list §4.9.2 stands: the arm's length, line
  of sight, occupancy, permission, shape row, orientation, placeable, support, inventory, batch cap,
  rate, nonce).
- **The occupancy test against people is a rapier query**: intersect the new block's collider with the
  realm's dynamic bodies. Microseconds. A capsule inside the cell ⇒ refuse with a reason. The bodies
  it can see are the realm's own occupants and the exterior shells of child realms it holds. So a block
  placed into a parked hull is refused by the hull's shell, and a crew member inside that hull is
  protected without the planet ever holding the crew member's pose (SL2).
- **A sleeping child's shell still answers a query.** Sleeping (§3.4) removes a body from the STEP, not
  from the broadphase. A block may not be placed inside a parked hull that has not moved for a week.
- **The refusal path.** The two-stage protocol of the base stands: an off-tick, effect-free receipt
  (`accepted_for_tick` or `refused(reason)`), then the authoritative echo at the applying tick. The
  client never applied anything, so a refusal deletes a preview and rolls back nothing. No prediction.
- **Support is checked here and never again** (§1.3). Mining the cell under a placed block later does
  not move the block in v1.

**Example.** Two players build a wall together on a moon. One places a block into the cell where the
other stands. The moon's shard finds the capsule in the cell and answers "refused: occupied". The
placer's preview vanishes. Nothing on either screen moves.

### 4.2 A block placed while the hull moves

- The cell is a hull-frame cell. The hull's motion does not change it. The player inside the hull is
  an occupant of the hull realm, so the arm's length and line of sight are tested in the hull's frame,
  at rest.
- The edit is applied on the hull's shard. The hull's exterior shell changes only if the edit touches
  the outside; then the hull restates its shell on the slow lane, and the parent replaces the hull's
  body collider.
- **THE SHELL SWAP MUST NOT POP — revision 1's answer is withdrawn.** Revision 1 said the pop is *"at
  most one block of lift — below the SL8 tolerance the P6 slice must state"*. A block is one metre, a
  one-metre lift in one tick is a JUMP, and SL8 forbids shipping a capability without its continuity.
  **The rule, stated now:** the parent replaces the collider and lets the SOLVER resolve the new resting
  contact, with rapier's penetration-correction velocity CAPPED so the change is a settle and not a
  jump. At a settle speed of 1 m/s that is 20 mm per tick at a 50 Hz shard (ESTIMATED from the cap and
  the tick), which reads as a hull sinking onto its new belly plate — physics, not an authored
  interpolation. The cap is a field of the realm's own record, not a constant.
- **The gate is S14:** weld a plate onto the belly of a parked hull with a crew member on the ramp, and
  demand that no tick moves the hull vertically by more than the stated cap.
- **A player standing on the ground editing a parked hull** edits another realm's cell. That is a
  command crossing a boundary — the transfer machinery, not the edit lane. v1 refuses it with a reason;
  P8 routes it (§6 decision 6).

### 4.3 The latency budget

At a 50 Hz shard (the DEV constant, `lib.rs:413-414`; the rate is a per-shard knob,
`kinematics.rs:148-149`), ESTIMATED means, at an 80 ms round trip:

| Stage | Mean | Where the number comes from |
|---|---|---|
| Uplink | 40 ms | half of the stated 80 ms round trip |
| Gateway drain wait + hop | 11 ms | the base's figure, ESTIMATED |
| Shard wait for the applying tick | 10 ms | half a tick at 50 Hz |
| Downlink | 40 ms | the other half of the round trip |
| **Click to the shard's echo arriving** | **~101 ms** | ESTIMATED, sum of the rows above |
| Client re-extract + remesh the edited chunk | **NOT A MEASUREMENT.** The base's 33 ms is a BUDGET (*"edit echo → pixels ≤ 33 ms"*, `block_system_design.md:6193`), and it was a budget for remeshing CUBE faces. Under V2.1 the client extracts a smooth surface from a density field, which is a different and heavier job | UNMEASURED; S19 measures it |
| **Click to pixel** | ~101 ms + the client extract | quotable only after S19 |
| If the edit is drawn at its STAMPED tick on the render clock | + the interpolation buffer (100–150 ms) | the flight document |

Revision 1 summed a budget into a cost and quoted ~145 ms. That number is withdrawn.

The base zeroed the buffer for edits. The flight document's RS-6 argues the opposite: draw the edit at
its stamped tick, or the terrain changes ahead of the drawn player (a jump seam, SL8). Consistency wins
under SL8: **the edit is drawn at its stamped tick**, and the placement preview (a drawing, not state)
hides the wait. The preview is O39(k)'s D2, it needs the owner's word, and revision 2 moves it into the
law-conflicts list where the owner will read it (§6.3 item 5).

The flight document also measured (2026-08-03, by exact replication) that the client's buffer was inert
and the picture stepped once per tick. That state is UNMEASURED today; the P5.1 spike re-runs the
one-test falsifier it named before any latency figure is quoted to the owner.

---

## 5. The swept containment test on terrain

Two sweeps exist and must not be confused.

### 5.1 The two sweeps

1. **Containment (which realm holds me) is already swept** for `Shell` and `Aabb`
   (`geometry.rs:1527`, `:1545-1553`) and the index answers a segment (`child_index.rs:171`). Nothing
   here changes. The `Obb` hole (D-MOVE-3 piece 1) stays ledgered (`geometry.rs:14-18`).
2. **Collision (did I hit the ground) is a separate sweep, inside rapier.** A hull descending at
   250 m/s moves 5 m per tick. A hull in low orbit over an Earth-sized body moves about 140 m per tick
   at ~7 km/s. (Revision 1 attached that figure to *"orbital speed near a small moon"*; that is wrong.
   Orbital speed at a 500 m asteroid is ESTIMATED at 0.42 m/s, from a 2,500 kg/m³ density and
   `v = √(GM/r)`.) A discrete step can pass through a 1 m thick ridge. Two rules close it:
   - **Continuous collision detection on every fast body.** Rapier's CCD sweeps the body's shape along
     its motion against static colliders and clamps the step at the first hit. Cost per fast body per
     step: ESTIMATED at tens of microseconds against a trimesh BVH; measured in S7. Slow bodies pay
     nothing. **Open risk:** under §2.5 option A the terrain is a MOVING kinematic set, and rapier's CCD
     against a moving trimesh is a known limitation. S16 reads the pinned version and says whether it
     works.
   - **The collider ring is swept and altitude-aware.** The chunks demanded this tick are the chunks the
     segment `prev_pos → pos + v · dt · k_lead` touches AND that contain a surface crossing (§1.4 rule
     2). `k_lead` is the collider build latency in ticks, measured, never guessed; and it grows the ring
     with speed, never caps the speed (M-B).
   - A body must never be stepped into a chunk whose collider is not built. §5.3 says how that stays
     true when the build runs on another thread.

### 5.2 The cost of fast flight — revision 1's arithmetic was optimistic and its escape hatch was false

**What was wrong.** Revision 1 costed a 7 km/s hull at *"~700 builds per second ≈ 350 ms per second:
heavy, bounded, and only while a body flies at orbital speed inside the surface ring, which the realm's
air (drag, M2a) ends quickly"*. Two errors:

1. **The tube's cross-section is two-dimensional.** At 7,000 m/s with `dt` = 0.02 s and `k_lead` = 2,
   the ring radius is 280 m, which is 8.75 chunks at a 32 m edge, so the tube is about 17.5 chunks
   wide. It advances 7,000 / 32 = 219 chunk lengths per second. Where the tube skims the surface the
   sheet inside it is ESTIMATED at 17.5 × 219 ≈ **3,800 surface chunks per second**, which at 0.5 ms per
   build is about **1.9 seconds of one core per second of flight**. That is over one core, not 350 ms.
2. **"The realm's air ends it quickly" is false on a moon.** `Ambient.density_kgpm3` is zero in space
   and the drag term is then exactly zero, with no special case (`drive.rs:116-118`, MEASURED). A moon
   has no air. Low orbit over an airless moon never decays. Both of this report's worked examples put
   the player on a moon.

**What the altitude test fixes, and what it does not.** With §1.4 rule 2, a hull in a 100 km orbit
demands NOTHING: no chunk of its tube holds surface. The 3,800-per-second figure therefore applies only
to a hull SKIMMING an airless moon at orbital speed, a few tens of metres above the rock. That case is
real and it is not closed by the altitude test.

**What closes it, put to the owner as decision 8b.** The COLLIDER TIER follows the body's speed, the
same way the drawn terrain already coarsens by dropping octaves (the standing LOD position). A hull
skimming at 7 km/s collides against a coarse trimesh, and each halving of the resolution divides the
chunk count by four. **The continuity rule that keeps it lawful:** the coarse surface is the OUTER
ENVELOPE of the fine surface — never below it — so a fast body stops early rather than late and can
never pass through a crater rim the player can see. The cost is that a very fast hull may clip an
invisible metre of air above a ridge, at a speed at which nobody can see a metre. **The alternative is
to state a build budget and let the frontier fall behind, which means a hull tunnels through a rim. That
is not acceptable and this report does not offer it.**

**The corrected mid-speed figure.** At 250 m/s the ring radius is 10 m, below one chunk edge, so the
floor applies and the tube is 3 chunks wide, advancing 7.8 chunk lengths per second: about **23 surface
chunks per second on flat ground** and ESTIMATED **70–120 per second over strong relief**, because a
cliff face puts three to five surface chunks in one column. Revision 1 said 24 and ignored the relief
term.

### 5.3 The thread and the tick budget — a gap revision 1 did not name

A shard tick at 50 Hz is 20 ms. A descent demands tens of chunk builds in one tick at an ESTIMATED
0.5 ms each. If the build runs on the tick thread that is a TICK HITCH, which SL8 lists among the eleven
seams. So:

1. **The generator evaluation, the diff application and the extraction run on a worker pool**, off the
   tick thread. The pool that already exists in the dependency tree is `bevy_tasks` (`Cargo.lock:1570`);
   no crate uses it yet (§0). Adopting `rayon` instead is a new library and needs the owner's word.
2. **A chunk enters the rapier world only at a tick boundary, by the tick thread, from a COMPLETED
   build.** Nothing half-built is ever visible to the solver.
3. **The body is never stepped past the frontier of built chunks.** The ring demands `k_lead` ticks
   ahead precisely so the frontier stays in front. If the frontier falls behind, that is a GATE FAILURE
   recorded by S6, not a stall of the body and not a silent tunnel.
4. **The per-tick budget the tick thread spends on physics is a stated number of the shard's profile**,
   and S20 reports the tick-time distribution under a descent so a hitch can be seen.

**Example.** A pilot dives a hull at a mesa at 250 m/s. The star system handed the hull to the planet at
the SOI, long before. The planet's ring builds the mesa's chunks a hundred metres ahead of the nose, on
a worker thread, and the tick thread only inserts finished ones. Rapier's sweep finds the cliff face
inside the tick's 5 m step and stops the hull ON the face, never inside it. The crew inside feel the
stop as a placement the planet authored, one hop down.

---

## 6. Decisions for the owner, one-way doors, and law conflicts

### 6.1 Open decisions (recommended answer first)

1. **rapier3d-f64 in the realm frame** vs rapier3d (f32) with a per-cluster anchor. Recommend f64: it
   deletes the ANCHOR half of the owed `FrameSpace` seam (`capability.rs:13-18`) while the GRID half
   survives as HR4's seam (§2.7). Under f32 the anchor half must be BUILT. P5.1 measures the step cost
   and the collider bytes first, in both flavours.
2. **Slope thresholds as character-record fields** (per character, like a hull's facts), with reference
   values the owner picks for the starter character after walking the starter world. Recommend fields;
   the numbers are a feel decision (as O43 is), not architecture.
3. **Terrain under a surface realm (a spaceport Area, a station on the ground).** Recommend the LOCAL
   formulation first: an area realm's floor is BUILT, so the area holds only its own built cells and
   needs no terrain and no reading. If the owner wants a bare-rock area realm, then the area evaluates
   the generator for the cells under it from its stamped placement as an instrument, and two further
   things must be true — the stale rule and the diff ask in §6.3 items 3 and 6.
4. **Draw an edit at its stamped tick (consistent, pays the buffer) and adopt the placement preview.**
   Recommend both. The preview is now in the law-conflicts list (§6.3 item 5).
5. **The hull's exterior shell: convex decomposition of the outside** vs coarse voxel occupancy of the
   outside. Recommend the decomposition (exact contact, no voids exposed). It covers a moving part's
   swept envelope (§3.5).
6. **Cross-realm edits (a player on the ground editing a parked hull).** Recommend refuse in v1; route
   as a crossing command at P8.
7. **The realm's frame when the body spins — REWRITTEN.** Recommend option A of §2.5: one NON-ROTATING
   dynamics frame, with the terrain as a kinematic set carrying the body's spin. Fall back to option B
   (body-fixed plus centrifugal, Coriolis and Euler terms in `advance_driven`, and an angular velocity
   in `Ambient`) if S16 says rapier cannot sweep a fast body against a moving trimesh. Option C (split
   the frames at an altitude) is a new seam and is not recommended. Revision 1's "body-fixed, ignore the
   terms" is withdrawn.
8. **CCD: rapier's built-in sweep** vs an own ray pre-check. Recommend rapier's; measure it (S7, S16).
   **8b — the collider tier follows speed, with the outer-envelope rule** (§5.2). Recommend yes; it is
   the only offered answer to a hull skimming an airless moon at orbital speed.
9. **Structural settling after a support is mined.** Recommend NO settling in v1: support is a
   placement-time check only, and a block whose ground is dug out hangs. A settling pass is a per-tick
   cost over an unbounded structure and belongs to a later slice with its own budget.
10. **The FINE-lattice vertex rule and the fixed index order** — THIS REPORT'S PROPOSAL, not an owner
    ruling. SL10 rule 4 names determinism rules and no vertex rule (MEASURED, §1.5). Recommend adopting
    it, together with the CHUNK-LOCAL ORIGIN rule, and stating the extractor's snap error rather than
    claiming zero.

### 6.2 One-way doors

| Door | Deadline | Cost if wrong |
|---|---|---|
| The extractor's output IS the collision surface: vertex lattice, index order, apron rule, chunk-local origin frozen under the world-generation tag | before the first chunk is saved or the first client ships | a boot that sinks into what is drawn, forever, on every saved world |
| rapier f32 + anchor vs f64 in the realm frame | P5.1, before the first collider line | rewrite every physics call site; build or delete the anchor half of `FrameSpace` |
| The realm's dynamics frame: non-rotating vs body-fixed (decision 7) | before the first spinning body ships, and before D-MOVE-4 pins the day | every band width, every authored child placement and every integrator term re-derived on a shipped world |
| The hull's exterior shell as a wire arm (positional postcard: a field order freezes when it ships) | when the arm ships (P8's first landing) | a protocol migration on the slow lane |
| Slope and step thresholds as character data, not constants | the character record format (P5.3) | players build ramps around a constant; changing it breaks or trivialises builds |
| The physics step `dt = tick_dt · time_multiplier` and every derived controller constant | before P5.3 pins the constants | every constant and every quantisation grid re-derived on a shipped world |
| No structural settling (decision 9) | before the first structure is saved | adding settling later moves every hanging structure on every saved world at once |

### 6.3 Law conflicts and SL6 asks

1. **SL6 ask — the hull's exterior shell.** DATA: a convex decomposition of the hull's OUTSIDE, covering
   a moving part's swept envelope, in the hull's own frame. FROM: the hull (child) TO: its parent (the
   planet, the system). RATE: on change, reliable slow lane beside mass and reach. WHY THE RECEIVER
   CANNOT COMPUTE IT: HR1 seals the hull's block field. WITHOUT IT: a hull lands on its bounding box; it
   floats over hills, sinks into hollows, rests level when it should tip; landing legs never touch. Each
   is a seam (SL8).
2. **SL6 ask — the hull's centre of mass and inertia tensor.** Same lane, same rate. WHY: a contact on a
   slope makes a torque; the parent cannot sum the hull's blocks. WITHOUT IT: the parent assumes a
   uniform box from the bound; a hull rests wrong and tips wrong.
3. **SL1 clarification, not new data — a surface realm reads its stamped placement to evaluate the
   generator under itself.** The reading already exists (clause 2). The new thing is that a consumer
   outside the placement machinery uses it as an address into the seed. Clause 5 forbids the placement
   and crossing MACHINERY from naming a realm's position; the terrain sampler is neither.
   **CLAUSE 6, which revision 1 did not answer:** *"A STALE reading is REFUSED, never used."* The
   degrade, stated so the owner can rule on it: when the area's stamped placement goes past its bound,
   the area KEEPS the colliders it already built from the last fresh reading, STOPS building new ones,
   and SAYS SO. A collider is a thing already made, not a fresh statement. Without this the pad under a
   parked hull would vanish the moment a shard stumbles, and the hull would fall through a world the
   client is still drawing.
4. **No conflict, recorded so nobody reopens it:** no occupant pose crosses for collision. A crew member
   inside a hull is protected from a planet-side placement by the hull's shell, not by the crew
   member's pose (§4.1).
5. **THE PLACEMENT PREVIEW — a law question, moved here in revision 2.** DATA: none crosses; the client
   draws a ghost block at the cell it resolved, before any shard has stated it. AGAINST: CLAUDE.md's
   non-negotiable *"NO client-side prediction"*, and SL10 rule 7 (*the client never derives state*).
   FOR: the ghost is a DRAWING and never state — nothing is applied, so a refusal deletes the ghost and
   rolls nothing back, and no pose, velocity or entity state is derived. WITHOUT IT: the player waits
   ~101 ms plus the buffer with no feedback at all on every block. The owner rules.
6. **SL6 ask, only if decision 3 takes the bare-rock branch — the edit diff for the cells under a
   surface realm.** DATA: the planet's edit diff for the terrain cells beneath the area's footprint.
   FROM: the planet TO: its child area realm. RATE: on change, one hop. WHY THE RECEIVER CANNOT COMPUTE
   IT: the seed does not know about a tunnel a player dug last week, and the planet owns the cell.
   WITHOUT IT: the area builds its pad from the seed alone, so the pad is solid where the client draws
   a hole, and a hull rests on rock that is not there. If decision 3 takes the built-floor branch, this
   ask disappears.

---

## 7. UNMEASURED items and the P5.1 collider spike

**The spike runs before any collider architecture is committed (the board's P5.1 rule stands).** It
measures, and its pass numbers are stated here so a run can fail. Rows S12–S20 are new in revision 2.

| # | Measurement | How | Pass |
|---|---|---|---|
| S1 | Trimesh chunk collider: build time and bytes, from the extractor's output | 1,000 seed chunks of the starter world at the chosen chunk edge, release build, on the reference machine, **in BOTH float flavours** | build ≤ 1.0 ms per chunk; ≤ 256 KB per chunk in f32 and ≤ 512 KB in f64 (ESTIMATED targets from §1.2; the owner may move them, never the method) |
| S2 | No-drift of the GENERATOR'S CANONICAL OUTPUT | the generator crate's canonical chunk record (integer vertices, index order, feature records) built on the server target and the client target, x86-64 and aarch64 | zero differing bytes (SL10 rule 3). **Rewritten in revision 2:** the old row compared a rapier `TriMesh` with a GPU vertex buffer, which are different structures that never share bytes |
| S2b | The drawn triangle against the collided triangle | drop a capsule on the SERVER's collider and read the same surface point off the CLIENT's uploaded vertex buffer, at a planet's radius, on both targets and in BOTH float flavours, with and without the chunk-local origin rule | the two points agree to within one FINE cell WITH the chunk-local origin rule. Without it, the f32 run is expected to fail by ESTIMATED ~0.76 m, and that failure is the rule's justification |
| S2c | The extractor's snap error | for 10,000 surface crossings, compare the true density-zero crossing with the snapped FINE-lattice vertex | report the distribution; the maximum sets the normal-error input for §2.3's climb/slide gap |
| S3 | The boot on the drawn surface | drop a capsule on 10,000 random surface points; read the contact depth against the triangle. **First read the pinned solver's allowed penetration and state it as an input** | never below the surface by more than (the solver's slop + S2c's snap error); never above by more than the skin |
| S4 | Chunk-boundary walk | walk a capsule across a chunk boundary at 20 sampled offsets, and across a trimesh-to-cube seam and a trimesh-to-feature seam | no vertical velocity impulse above one FINE cell per tick |
| S5 | SL9 on the collider cache, by OCCUPANTS | the same 1 / 10 / 100 occupants, co-located and spread, on a 500 m asteroid and an Earth-sized body | bytes equal across the two bodies to within one chunk; bytes grow with spread occupants and not with radius |
| S6 | Collider build rate under a fast hull | a hull at 30, 250 and 7,000 m/s over the surface with the swept, altitude-aware ring, and at 100 km altitude | the frontier is never late (no body stepped into an unbuilt chunk); core time per second reported at each speed; the 100 km run demands ZERO chunks |
| S7 | CCD cost | a hull with CCD at 250 m/s against the trimesh, 10,000 steps | per-step cost reported; no tunnel through a 1 m ridge in 10,000 dives |
| S8 | rapier snapshot/restore (D-19 SPIKE-6a) | serialise the planet's world with 100 bodies, restore on a fresh process and on the other target, step 1,000 ticks from the same inputs | on the same target: bit-identical poses; across targets: no pose differs by more than one FINE cell at the restore tick, and the drift over 1,000 ticks is reported (Category C forbids cross-host re-simulation, so this number sets how a checkpoint is carried, not a gate on the solver) |
| S9 | f64 vs f32 step cost and bytes | S7's scene in both flavours | the ratio, reported; the owner rules on it |
| S10 | The interpolation buffer falsifier | the flight document's one-test check on today's client | reported before any latency figure is quoted |
| S11 | Placement occupancy query | 4,096-edit batch against 128 capsules, including sleeping child shells | ≤ 1 ms per batch |
| **S12** | **SL9 on the collider cache, by CHILD REALMS** | 1 / 100 / 600 parked hulls in one planet realm, each with a stated shell, then wake one | bytes reported; **per-tick core time must NOT grow with the parked count**; the woken hull and its contacts wake and nothing else does |
| **S13** | **Seed-placed features (V2.2)** | a forest chunk against a bare chunk: build time, bytes, and the walk over a trunk | the feature colliders are demanded, cached and evicted with the chunk; the added bytes reported; a capsule is stopped by a trunk it can see |
| **S14** | **The shell swap does not pop (SL8)** | weld one plate onto the belly of a parked hull with a capsule on the ramp; 100 repeats | no tick moves the hull vertically by more than the realm's stated settle cap (ESTIMATED 20 mm at 1 m/s and 50 Hz) |
| **S15** | **What the pinned parry actually has** | read the pinned version's shape list and its voxel shape's documented layout | report whether a voxel shape exists, its bytes per voxel, whether it classifies seams, and whether any signed-distance shape exists. **§1.2's column C is UNMEASURED until this runs** |
| **S16** | **CCD against a MOVING kinematic trimesh** | the terrain set on a kinematic body carrying Earth's spin; a hull at 250 m/s and a capsule walking, 10,000 steps | no tunnel, and the contact is stable. **If this fails, decision 7 falls back to option B** |
| **S17** | **A standing boot on a spinning planet (decision 7 option A)** | a capsule standing still on the equator for 10,000 ticks, terrain kinematic at Earth's rate | the capsule's position in the body-fixed sense does not drift by more than one FINE cell per 1,000 ticks; the ring is never late at 465 m/s |
| **S18** | **One integrator, no bend at a crossing** | a hull under a constant drive crossing from a stub realm (`advance_driven`) into a rapier realm | the path's second derivative does not step at the shell beyond a stated bound |
| **S19** | **The client's smooth re-extract** | remesh one edited chunk on the client, at the chosen chunk edge, smooth extractor, on the reference machine | reported. **No click-to-pixel figure is quoted to the owner before this runs** |
| **S20** | **The tick budget and the hitch** | a 250 m/s descent over strong relief, 10,000 ticks, builds on the worker pool | the tick-time distribution reported; no tick exceeds the shard's stated physics budget |

Everything the base states as a physics number (27 MB per cluster, 64 m physics radius, 860 KB per
trimesh chunk, 0.53125 m autostep, 3.000° separation, every "at 20 Hz" derivation) is ESTIMATED until
one of these rows measures it. **Revision 2 withdraws the claim that every "at 20 Hz" figure is wrong by
2.5×:** `tick_hz` is a per-shard knob (`kinematics.rs:148-149`, MEASURED), and `tick_hz: 50` is a field
of the DEV cluster constant (`lib.rs:413-414`), so there is no single shipped rate to divide by. The
base's figures must be re-derived at whatever rate the shard in question runs, not scaled by a constant.

---

## 8. Stale claims in the investigation base, and what supersedes them

| Claim | Where | Superseded by |
|---|---|---|
| Per-chunk `TriMesh` is rejected; the planet cube-cell lane is a greedy-box decomposition | `block_system_design.md` §4.6.1 | V2.1 (smooth terrain) + SL10 clause 5: the terrain collider is a trimesh from the same extractor. The cube and shape lanes survive for placed blocks only |
| Generated terrain is `Cube`-only at every tier | §4.9.4; `decision_board.md` P5.2 | V2.1 |
| Slope thresholds cost nothing on terrain because natural terrain is 0° / 90° (C1) | §4.7; `decision_board.md` O39(b) | V2.1: slopes bind on every hillside; thresholds become character data |
| A 1 m staircase makes wheeled vehicles impossible; hover is the shipped answer; reserve V-R6 | `collidable_decoration.md` §8.5, §12 V-R6, UD-16 | V2.1: the surface is smooth by default |
| Terrain never crosses the wire; only the seed crosses | `high_speed_flight_latency.md` §4, §11.5; roadmap P4 DoD | SL10 rule 6: edits, placed blocks and live deposits cross as a one-hop diff. The base shape still does not |
| The collider cluster needs a merge/split rule; a bubble radius per body | `decision_board.md` O38 | §1.4: one collider per chunk per realm, reference-counted; the ring is swept, speed-sized and altitude-aware |
| The client draws 2 m cells at 1.1 km, so it cannot adjudicate collision | `high_speed_flight_latency.md` §7.4 | Still true that the client adjudicates nothing (SL10 rule 7); the tier argument is moot because the client may evaluate any tier it wants from the seed |
| Every physics figure "at 20 Hz" (snap-to-ground, physics radius, budget per tick) | §4.6.1, §4.7; `collidable_decoration.md` §8.4 | The rate is a per-shard knob (`kinematics.rs:148-149`); the DEV cluster runs 50 Hz (`lib.rs:413-414`). Each figure must be re-derived at the shard's own rate, not scaled by one constant |
| The 33 ms edit-to-pixel figure is a cost | `block_system_design.md:6193` | It is a BUDGET, and it was a budget for remeshing CUBE faces. The smooth extractor is a different job (§4.3, S19) |
| Motion is slaved to the command rate (40 % of configured speed) | `high_speed_flight_latency.md` §6.1 A2 | For a HULL the parent now holds the last drive and applies it each tick until it goes stale (`drive.rs:340-350`). For the walking dot the per-datagram shape remains (`dot.rs:238`, `session.rs:227`), deferred with the suit (S6) |
| The approach governor is a ceiling on the flight path | `crates/core/src/flight.rs:1-19` header; `dot.rs:465` | M-D: the ceiling leaves the flight path. The walk keeps it as the placeholder the owner deferred (suit S6) |
| The realm's pull is one vector | `drive.rs:112-118` | §2.1: a function of position from the authored mass |
| The character skin of 1/64 m is "f32 quantisation at a 4,096 m anchor" | base §4.7 | The f32 step at 4,096 m is 0.49 mm (ESTIMATED), 32× smaller. The derivation does not hold and revision 2 does not carry the number |

---

## 9. Summary for the owner

The server builds each terrain chunk's collider as a trimesh from the same extractor the client draws
with, so the boot stands on the drawn triangle to within a stated, measured tolerance — the generator's
integer output is byte-identical, the extractor's snap costs up to half a millimetre, and each host must
re-express the chunk near a chunk-local origin or an f32 renderer is out by most of a metre. Placed
blocks keep their square colliders, and a seed-placed tree states its own trunk and canopy shape from
the same crate. Colliders live only around bodies, one per chunk per realm, demanded by a swept,
altitude-aware ring, built on a worker thread. Gravity is a function of position from the realm's
authored mass; up is the radial; slopes and steps are character data. A landed hull is one rigid body in
the planet's world, and it became the planet's child at the SOI, not at the ground, so there is no
crossing at the landing. Parked hulls sleep, or six hundred of them cost a tick each. Two things must
cross from a hull on the slow lane: its exterior shell, covering any moving part's swept envelope, and
its inertia. The one open question this report cannot settle is what frame a spinning planet's realm
integrates in; revision 1 answered it wrongly and revision 2 puts three options to the owner. Twenty
measurements gate the architecture before it is built.

---

## Revision log

Every finding from `verdicts/physics_law.md` (LAW) and `verdicts/physics_feasibility.md` (FEAS), and
what revision 2 did with it.

| # | Finding | Verdict | What I did |
|---|---|---|---|
| 1 | LAW §1 + FEAS §3 — the body-fixed realm frame breaks the realm's own children; a planet realm's bound is its SOI; no Coriolis anywhere; `advance_driven` has no angular term | BREAKS_LAW / WRONG | ACCEPTED. Rewrote §2.5 completely. Withdrew "body-fixed, ignore the terms". Added the SOI evidence (`generate.rs:1796-1798`), the force table out to the SOI (~8.0 m/s²), the Coriolis figures, and the refuter's own 560 m/tick refutation of my example. Rewrote decision 7 as three options with costs, recommending a non-rotating dynamics frame with the terrain as a kinematic rotating set, and naming the two risks it carries (S16, S17). Withdrew §2.1's "the only change to the parent-side integrator". |
| 2 | LAW §2 + FEAS §2 — "zero by construction" is an argument, the gate does not test it, it is false on f32, and SL10 rule 4 says nothing about a vertex lattice | UNMEASURED_AS_FACT / WRONG | ACCEPTED both. MEASURED SL10 rule 4 myself (`owner_decisions_2026-09-07_voxels.md:36-40`): it names five determinism rules and no vertex rule. Moved the FINE-lattice vertex rule and the index order into §6.1 as decision 10 with a one-way door. Replaced §1.5's "zero" with a three-link chain, each with its own error and gate. Added the CHUNK-LOCAL ORIGIN rule and the 0.76 m f32 figure. Rewrote S2, added S2b and S2c. |
| 3 | LAW §3 + FEAS §7.1 — V2.2 trees have no collider; the word "tree" appears once | MISSING | ACCEPTED. Added the fourth collider arm in §1.3, anchored on SL10 rule 1, which already names seed-placed features as static shape. Stated the shape source, the cache lifetime, the felling case, the cost row in §1.4 rule 7, and gate S13. |
| 4 | LAW §4 — HR4: no second shard kind, no G-IDENTICAL fixture, and f64 silently deletes HR4's owed structural half | BREAKS_LAW | ACCEPTED. Added §2.7. Named the hull realm as the second shard kind, named the place-then-walk fixture, cited `capability.rs:13-18` and `:16-17`, and stated exactly what f64 deletes (the anchor half) and what survives (the grid half). Carried the consequence into decision 1. |
| 5 | LAW §5 + FEAS §8 tail — SL9: many child realms are never costed; no sleeping rule; S5 measures occupants only | BREAKS_LAW | ACCEPTED. Added §3.4 with the sleeping rule and gate S12. Added the child-shell term to §1.4's cost model. Added the "a sleeping shell still answers a placement query" clause to §4.1. |
| 6 | LAW §6 — SL8: a one-block lift pop is accepted and its tolerance is deferred | BREAKS_LAW | ACCEPTED. Rewrote §4.2's shell-swap paragraph. Stated the continuity now: a capped penetration-correction velocity, so the change is a settle (ESTIMATED 20 mm per tick at 1 m/s and 50 Hz), with the cap as realm data, not a constant. Added gate S14. |
| 7 | LAW §7 — SL1 clause 6 unanswered for a surface realm's stale reading; and the area never gets the planet's diff | MISSING | ACCEPTED. Wrote the degrade into §6.3 item 3 (keep what is built, stop building, say so). Added the diff as SL6 ask item 6. Also added a simpler local formulation first in decision 3: an area realm's floor is BUILT, which dissolves the whole case. |
| 8a | LAW §8 — "handed the hull to the planet a kilometre up" | WRONG | ACCEPTED. Rewrote §3.3: the hull becomes the planet's child at the SOI, so there is NO crossing at the landing. Fixed the §5 example. |
| 8b | LAW §8 — "at orbital speed near a small moon, 140 m per tick" | WRONG | ACCEPTED. Corrected in §5.1: 140 m/tick is ~7 km/s, low orbit at an Earth-sized body. Orbital speed at a 500 m asteroid is ESTIMATED 0.42 m/s, method stated. |
| 8c | LAW §8 — "the code ticks at 50 Hz" and "every 20 Hz figure is wrong by 2.5×" | OVER-CLAIM | ACCEPTED in substance. Corrected §0, §4.3, §7 and §8: `tick_hz` is a per-shard knob (`kinematics.rs:148-149`) and 50 is the DEV constant. Withdrew the 2.5× scaling. **DISPUTED: the refuter cites `crates/wire/src/channels.rs:785` as "a 20 Hz `UniverseRate`"** — that line is inside a unit test (`let rate = ServerControlMsg::UniverseRate { tick_hz: 20 };`, a round-trip encode assertion), so it is a test fixture and not evidence of a shipped 20 Hz rate. The correction stands on `kinematics.rs:148-149` alone. |
| 8d | LAW §8 — parry `Voxels` bytes and "parry has no SDF shape" are stale base claims stated as fact | UNMEASURED_AS_FACT | ACCEPTED. Marked both UNMEASURED in §1.2 column C and in §1.3, and added gate S15 to read the pinned version's shape list. |
| 8e | LAW §8 — "reach" used for three different things, and REACH is a law word | AMBIGUOUS | ACCEPTED. Added a three-word glossary at the head of the report and replaced every use: "reach" (the 2026-09-02 ruling's number), "the arm's length" (a placement distance), "the physics ring radius" (§1.4). |
| 9 | LAW §9 + FEAS §1 — six citations point a few lines away | WRONG (citations) | ACCEPTED. Re-ran every one. Fixed: `built.rs:37-54` (BuiltFacts), `built.rs:76` / `:78` (bound / look), `drive.rs:139-182`, `ghost.rs:55` and `ghost.rs:9-10`, `motion.rs:49-52`, six `FrameSpace` mentions in three files, `drive.rs:340-350`, `dot.rs:465` for the governor, `geometry.rs:8-27` and `:1545-1553`, `placement.rs:59`, `transient.rs:230-236`, `child_index.rs:171`. Also dropped `celestial.rs:151` as evidence for a planet's own mass: FEAS is right that `central_mass` is the PARENT's mass; `taxonomy.rs:750` alone carries the claim. |
| 10 | FEAS §4 — the fast-flight ring is not bounded; the arithmetic is optimistic by ~5×; the airless-moon escape hatch does not exist; growing the ring makes it worse | REFUTED | ACCEPTED. Rewrote §5.2 with the two-dimensional tube (ESTIMATED 3,800 chunks/s and 1.9 core-seconds per second at 7 km/s), removed the false drag sentence with the code evidence (`drive.rs:116-118`), corrected the 250 m/s figure to 23 flat / 70–120 over relief, added the ALTITUDE-AWARE ring rule (§1.4 rule 2) which deletes the orbital case entirely, and put the remaining skimming case to the owner as decision 8b — a speed-following collider tier with an outer-envelope continuity rule. Stated plainly that letting the frontier fall behind is not offered. |
| 11 | FEAS §5 — the trimesh sizes are f32 numbers under an f64 recommendation, and S2 compares two structures that never share bytes | REFUTED | ACCEPTED. §1.2 now costs both flavours (f32 ~140 KB, f64 ~250 KB per chunk), the walker's ring in both, and S1's pass is split per flavour so the f64 run has margin. S2 rewritten to compare the generator's canonical output; S2b added for the drawn-vs-collided point. |
| 12 | FEAS §6.1 — the 1 FINE cell skin ignores the solver's own slop; the base's 15.6 mm is not the stated derivation | UNMEASURED_AS_FACT | ACCEPTED. §1.5 now demands the solver's slop as a stated input before S3's pass exists, and records that the f32 step at 4,096 m is 0.49 mm (ESTIMATED), so the base's derivation fails. Added the base row to §8. |
| 13 | FEAS §6.2 — the 33 ms client re-extract is a BUDGET for cube remeshing, used as a cost | UNMEASURED_AS_FACT | ACCEPTED. §4.3's table now marks it as a budget with its citation, withdraws the ~145 ms total, quotes ~101 ms to the shard's echo only, and gate S19 must run before any click-to-pixel figure reaches the owner. |
| 14 | FEAS §6.3 — the 0.2–0.5 ms build carries everything downstream | UNMEASURED_AS_FACT | ACCEPTED. Marked at its source in §1.2 with the sentence that §5 and S6 inherit the estimate, and repeated at each use. |
| 15 | FEAS §6.4 — the 8-chunk walker ring has no stated chunk edge and no stated arm's length | UNMEASURED_AS_FACT | ACCEPTED. Added the stated-assumption paragraph at the end of §0 (32 m chunk edge) and spelled the derivation in §1.2's memory row. |
| 16 | FEAS §6.5 — the 51 ms downlink row is unexplained | WRONG | ACCEPTED. Rebuilt §4.3's table with a "where the number comes from" column: 40 up, 11 gateway, 10 half-tick, 40 down. |
| 17 | FEAS §7.2 — no thread and no per-tick budget; a 0.5 ms build on the tick thread is a tick hitch | MISSING | ACCEPTED. Added §5.3 with four rules, named `bevy_tasks` as the pool already in the tree (`Cargo.lock:1570`) and `rayon` as a NEW library needing the owner's word, and added gate S20 for the tick-time distribution. |
| 18 | FEAS §7.3 — the rotating joint and the turret (V2.5) break §2.5's own rule and flood the shell lane | MISSING | ACCEPTED. Added §3.5: a turret is a kinematic part of the hull's own collider set, and the STATED shell covers its swept envelope, not its pose, so nothing crosses per tick. Carried into §3.2 and decision 5. |
| 19 | FEAS §7.4 — support removed after placement is never answered | MISSING | ACCEPTED. Stated the rule at the end of §1.3 (support is a placement-time check only; no settling in v1), repeated in §4.1, and added decision 9 and a one-way door, because adding settling later moves every hanging structure on every saved world. |
| 20 | FEAS §7 small gap — two integrators for one hull | MISSING | ACCEPTED. §3.1 now says rapier is the integrator on a rapier realm and `advance_driven`'s three terms become its forces; `advance_driven` stays on stub realms. Added gate S18 for the path bending at a crossing. |
| 21 | FEAS §7 small gap — the per-datagram walk against a fixed-step solver | MISSING | ACCEPTED in part. §0 and §8 record the per-datagram shape with its lines; the migration stays deferred with the suit (S6), and the report now names it as a seam rather than only a deferral. |
| 22 | LAW §11 — the placement preview belongs in the law-conflicts list | MISSING | ACCEPTED. Moved into §6.3 as item 5, with the data, the law it touches, the argument for, and the cost of doing without. |
| 23 | LAW §10 and FEAS §8 — what both refuters say must be kept | — | KEPT UNCHANGED: SL1 (the realm is centred on itself, no chain folds an absolute), SL2 (no occupant pose crosses for collision), SL4 (the crossing path untouched, gravity inside the integrator), the movement contract (no velocity crosses up, the ring grows with speed and never caps it), SL5 (no reduced world), the seed ruling, the trimesh over a heightfield, the SL9 occupant half of the cache rule, gravity from the authored mass, up as the radial, thresholds as character data, the two SL6 asks with all four fields, and rapier named as an owner choice rather than adopted. |
