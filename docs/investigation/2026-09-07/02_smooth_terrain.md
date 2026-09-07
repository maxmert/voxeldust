# 02 — Smooth terrain on voxels, and the square blocks that stand on it

**Domain:** the terrain representation (V2.1) together with the square building blocks (V2.3).
**Date:** 2026-09-07. **Status:** an investigation report. It proposes. The owner decides.
**Revision:** REVISED 2026-09-07 after two refutations (`verdicts/terrain_law.md`,
`verdicts/terrain_feasibility.md`). Section 12 is the revision log. Thirty findings came in. This report
accepts them all in substance, and disputes a detail inside three of them.
**Law read:** CLAUDE.md (HR1–HR6, SL1–SL10), `owner_decisions_2026-09-07_voxels.md` (V1 = SL10, V2.1–V2.9,
V3), every earlier `owner_decisions_*.md`, `DEFERRED.md` (P4/P5/P6, `FrameSpace`, `reanchor`), and SL8
(a seam is a defect; eleven seam kinds).
**Base read:** `block_system_design.md` §2.7.1 (the persisted record), §4 (shapes), §5.1 (the mesher),
§5.6 (the un-square edge), §3.5 (generation), §4.9.4 (what terrain may emit), Appendix A (the measured
noise numbers); `block_system_design_addendum_2.md` (the detail ladder); `decision_board.md` (O1, O2,
O10, O28, O39, the collision rule, §3 doors, §4 P4a/P5/P4b); `stunning_look_plan.md` §1, §5, §7.1;
`collidable_decoration.md` §3, §8.

**How to read a number in this report.** MEASURED means somebody ran a program and the report cites
where. ESTIMATED means arithmetic on top of a measured number or a published number. UNMEASURED means
nobody has run it. Every claim about the code cites `file:line`.

---

## 0. The answer in one page

The owner asked for terrain that looks smooth and realistic, but is still made of voxels; that you mine
like Minecraft; and that reshapes itself when you place a voxel (V2.1). The owner also asked that
building blocks stay square, with about twenty shapes (V2.3).

**The recommendation.**

1. **A terrain cell holds one substance and one RADIAL GAP sample.** The gap is `r − h`: the cell
   centre's radius minus the terrain height the generator states for that direction. It is negative
   inside solid, which is the workspace's existing sign (`crates/core/src/geometry.rs:440-446`, a shell
   reads `p.length() - r`). It is clamped to one cell and quantised to `i8` in steps of **1/128 cell**.
   The seed decides it for every unedited cell (SL10). An edit stores it as a diff. *Example: a dirt
   cell on a hillside holds `substance = dirt, gap = −0.30 cell` ("I am three tenths of a cell under the
   surface").*
   It is a RADIAL GAP and not a true signed distance. On a slope the true distance to the surface is the
   gap times the cosine of the slope, so the clamp bites at a different depth on a plain and on a
   mountainside. The zero set is still exactly the surface, which is what the extractor reads.
2. **A smooth extractor turns the gap lattice into the surface.** The recommended extractor is naive
   surface nets over cell-centred samples: one vertex per 2×2×2 group of cells with a sign change,
   placed at the mean of the edge crossings. It uses add, subtract, multiply, divide and one square root
   for the normal. SL10 clause 4 is NOT satisfied "by construction": it is satisfied by a named fence
   (§2.4) and PROVED by measurement M3. Until M3 runs, the claim is UNMEASURED. *Example: the moon's
   shard and the client both run the extractor over the same chunk; M3 is the gate that says the two
   triangle lists are the same bytes.*
3. **The same extracted surface is the collider, and one invariant ties the two hosts together.** The
   realm's shard extracts the tier-0 surface over the body's SWEPT SEGMENT for this tick (§5.4), and the
   client holds tier 0 over a band that strictly contains that segment (§5.5). Without that invariant
   "what you see is what you stand on" is a hope. *Example: a hull dives at its rated cruise onto a moon;
   the client's tier-0 band leads the hull by the closing speed times the interpolation buffer
   (`crates/wire/src/channels.rs:328`, `INTERP_BUFFER_MS = 120.0`) plus one tick, so the boots land on
   the slope the eyes see.*
4. **Square blocks stay square, in the same grid.** A cell holds ONE form: a terrain form (smooth, with
   a gap) or a catalogue form (a cube, a wedge, a slab). The mesher has three lanes selected by the
   cell's form, never by the realm's kind: the smooth lane, the cube lane, the shaped lane. There is ONE
   collider builder per lane, never one per realm geometry. *Example: a hull block placed on a hillside
   is a cube in the cube lane; the dirt around it stays in the smooth lane; a steel cube in a station's
   wall and a steel cube on a moon's hillside go through the same builder.*
5. **The terrain gap survives under a placed square block.** The planet cell record carries the gap
   beside the block identity, so the hill runs through the block, is hidden inside it, and returns
   unchanged when the block is removed. *Example: a player sets a steel foundation into a slope, later
   breaks it, and the slope is exactly what it was before.*
6. **The detail ladder stays.** Tier L drops the L finest octaves of the same generator, the same
   extractor runs over the coarser lattice, a downward apron of a CONSTANT depth in cells closes the
   crack (§5.2), and a dither crossfade removes the pop. The server never coarsens: the collider is
   tier 0 by type. *Example: a mountain seen from a descending hull grows detail without a step; the
   shard under the landing site extracts the tier-0 surface only.*
7. **THIS DOMAIN MAKES AN SL6 ASK.** The earlier draft said it asked nothing. That was wrong. A player's
   edit must reach a client, and no lane carries a cell today. Section 9.1 states the ask in SL6's own
   shape: two new wire surfaces and two protocol-minor bumps.

**What this shuts.** Generated terrain is no longer cube-only (the base's §4.9.4 is reversed). The
terrain cell record needs a gap byte from the first saved world (a one-way door), and the persisted edit
record grows from five bytes to six. Iso-surface extraction is no longer "rejected, one-way" (§5.6.5) for
terrain; it stays rejected for built blocks. The rim warp (§5.6) stops being the terrain's look and
stays, at most, a look for natural-substance square blocks. Trees as leaf cells (R13, R14) are gone under
V2.2, and the collision rule gains a fourth arm for them (§2.3 item 4).

---

## 1. What exists in the code today

The code is the only current truth. This is what it holds for this domain.

| Claim | Where | What it means for this design |
|---|---|---|
| **No terrain, chunk, voxel field or mesher exists.** A grep for `marching`, `surface net`, `dual contour`, `isosurface`, `density`, `voxel`, `chunk`, `terrain` across `crates/` finds only a containment SDF for a box and a capability tag. | `crates/core/src/geometry.rs:440-446` (`Boundary::signed_distance`; `Shell { r } => p.length() - r`, NEGATIVE inside); `crates/sim/src/capability.rs:36-43` (`VoxelGeometry { Spherical, Cartesian }`, a profile tag) | The whole terrain representation is new. There is no code to reuse and no code to fight. "Negative inside" is the workspace's existing sign, and this report's gap byte takes the same sign. |
| **`FrameSpace`, `reanchor()`, `AnchorGen` do not exist**, and building the seam is TERRAIN'S FIRST SLICE by the owner's word. | `crates/sim/src/capability.rs:11,13-18` (comments); `docs/design/DEFERRED.md:309-311` ("★ THE FRAME-SPACE SEAM … is TERRAIN'S FIRST SLICE by the owner's word (2026-09-05, 'yes to all')") | Nothing today turns a cell address into a radial direction on a moon. The generator cannot run without it. It is slice 0 of §11, not an aside. |
| **rapier3d / parry3d are NOT dependencies.** | `Cargo.toml:24-90` (no rapier, no parry); `crates/core/src/pose.rs:719,808`; `crates/physics/src/motion.rs:31,60` (comments) | Every collider statement below is a design against a library the owner has not yet adopted. USER DECISION 4-B of the base is still open (D8). |
| **`noise = "=0.9.0"` is declared in the workspace and used by no crate.** | `Cargo.toml:78`; no `crates/*/Cargo.toml` lists it; no `noise::` in `src` | The generator will be hand-written on integer hashing (SL10 clause 4 forbids a library's transcendental calls anyway). The declaration is dead weight to remove. |
| **`SplitMix64` is the one STREAM generator; the noise bench's hash is NOT the same function.** | `crates/core/src/rng.rs:11-29` (`next_u64` performs the state step `state.wrapping_add(0x9E37_79B9_7F4A_7C15)` FIRST); `scripts/noisebench/src/main.rs:22-35` (`mix64` is the FINALIZER only; `hash3` adds its own per-axis multiplies) | A stateful stream and a stateless position hash are two different functions. The generator needs the POSITION HASH, and its exact definition sits inside the pinned-digest door. Name it exactly in slice 1; do not write "reuses SplitMix64" and leave it. |
| **The fine lattice is `2⁻¹⁰ m` per cell; `LatticePos` = i64 cell + f64 offset.** | `crates/core/src/pose.rs:475,513,526` (`Tier::Fine => -10`, `cell_edge_m`), D-41 in `DEFERRED.md:3382-3420` | A gap step of 1/128 cell is `2⁻⁷ m` at tier 0 = exactly 8 fine cells, and `2^(L+3)` fine cells at tier L. A step of 1/127 is not a whole number of them (§2.3). |
| **The store stamp already refuses a mismatched world generation.** It folds f64 world constants only. | `crates/core/src/store_stamp.rs:117-123,301-307` | SL10 clause 3 says the tag must also name the generator crate's version and the target's arithmetic profile. Two more inputs to an existing fold. |
| **`BlockEdit` is a WORD IN A COMMENT, not a wire arm, and `InterShardFlow` is the shard-to-shard contract in any case.** | `crates/wire/src/intershard.rs:34` ("RESERVED (variant lands with its consumer): `BlockEdit` (P6)") | The earlier draft said the terrain diff "rides the reserved `BlockEdit` arm". It cannot: there is no variant, and that contract does not reach a client. §9.1 states the real ask. |
| **The client-facing lane carries no cell and no block.** Its arms are `Welcome`, `SubscriptionOpened`, `SubscriptionClosing`, `AuthorityChanged`, `RequestCut`, `CutConfirmed`, `TransferCosmetic`, `TransferRejected`, `Ping`, `Close`, `UniverseRate`, `OwnEntity`, `RealmRegistry`, `RealmSceneDelta`, `Event(EventMsg)`, `StarCatalogue`, `SkyAlive`. | `crates/wire/src/channels.rs:88-195`; `EventMsg` at `crates/wire/src/channels.rs:303-320` (`Notice`, `EntityRemoved` only) | The cheapest lawful carrier is an appended `EventMsg` variant on the existing `Event` arm: its own doc already rules that P9's signal deliveries ride it that way — "one carrier, two consumers, built once" (`crates/wire/src/channels.rs:189-194`). |
| **The live shard→gateway scene lanes are `WindowBody` and `WindowMembership`; `ShardToGateway::RealmSceneDelta` is a TOMBSTONE.** | `crates/wire/src/session_flow.rs:315,337` (live); `crates/wire/src/session_flow.rs:236-248` ("nothing produces it … Do not revive") | A terrain diff cannot ride the tombstone. The shard→gateway leg needs its own appended arm, mirroring `ShardToGateway::EntityRemoved` (`crates/wire/src/session_flow.rs:259`), which is exactly the shape the gateway already fans out as `ServerControlMsg::Event`. |
| **`coarsen_level: u8` survives only in a TOMBSTONE payload.** | `crates/wire/src/intershard.rs:1173-1186` (`OccupantInterest`, "nothing produces it") | Addendum 2 §A's claim that "line 656 already carries a `coarsen_level` precision ladder" is stale: that lane is dead. The tier of a chunk address is a new field. |
| **The client's mesh seam is renderer-free AND ALREADY CARRIES NORMALS.** `Vertex` is `pos: [f32; 3], normal: [f32; 3]`; `MeshPrim` is a vertex buffer + colour + transform; the renderer builds a mesh from it with no shape branch. | `crates/client/src/realm_scene.rs:770-773` (`Vertex`), `791-795` (`MeshPrim`), `887` (the proxy already writes a normal); `crates/client-render/src/lib.rs:1828-1841` (`mesh_from_prim` / `mesh_from_vertices`) | V2.8 is already the shape of the seam. The smooth lane emits `MeshPrim`s AS THEY ARE — **nothing is widened**. (The earlier draft cited `crates/client-render/src/lib.rs:820-821`; those lines build the stub reference ground plate, a `Cuboid`. The citation was wrong; the seam claim was right.) |
| **The realm proxy is a fixed 12×8 UV sphere and calls itself the coarsest rung.** | `crates/client/src/realm_scene.rs:800-806,893` | The ladder's top is the proxy today; the terrain ladder plugs in below it. |
| **The existing galaxy generator calls libm.** `celestial.rs` uses `sin`, `cos`, `atan2`, `powf`. | `crates/physics/src/celestial.rs:66-93,103,208-240` | SL10 clause 4 forbids these INSIDE the terrain generator. The terrain generator is therefore a separate crate under the fence of §2.4. The star field is shipped once (SL10 clause 8), so the galaxy's math stays where it is. |
| **A profile has `surfaces: bool`, and the code ALREADY allows `surfaces` on a Cartesian realm.** `ShardProfile::build` refuses `surfaces` only without a voxel realm, and `ship()` sets `Cartesian` + `surfaces: true`. | `crates/sim/src/capability.rs:52,111,134` (`SurfacesNeedVoxel`); `crates/sim/src/capability.rs:260-272` (`ship()`) | The smooth lane is a capability of a voxel realm, never a shard-kind test (HR3). A hull MAY hold terrain-form cells, and §5.6 shows why it must, for HR4 to be real. |

---

## 2. Question 1 — the representation of a terrain cell

### 2.1 The two candidates

- **(A) A gap sample with a material, meshed by a smooth extractor.** Each terrain cell holds a
  substance and a signed radial gap to the surface. An extractor (surface nets, dual contouring or
  marching cubes) turns the lattice into triangles.
- **(B) A blocky cell with a smoothing pass.** Each terrain cell is a cube. A pass after meshing rounds
  the cubes: either a vertex relaxation over the cube mesh, or a surface extractor run over the BINARY
  occupancy.

Candidate (B) is one thing in two costumes. A vertex relaxation over a cube mesh converges toward the
same result as an extractor over binary occupancy: every one-metre step becomes a 45° ramp, every
overhang becomes a bulge, and a cliff becomes a slope. The base already names this failure for surface
nets over a binary field (`block_system_design.md:8998-9001`). So (B) is "(A) with one bit of gap".

### 2.2 The comparison

| Criterion | (A) gap + smooth extractor | (B) blocky + smoothing pass |
|---|---|---|
| **Mining feel** (V2.1: "mine, same as in Minecraft") | One click removes one cell: its gap becomes fully air. The extractor re-runs on the 3×3×3 neighbourhood. The hole is a rounded one-metre bowl. *Example: a player digs into a sandbank; each click takes one cell's worth of sand and leaves a scoop, never a cube-shaped socket.* | One click removes one cube. The hole is a cube with softened rims. The socket reads as a box in a smooth hill. |
| **Placing reflows the terrain** (V2.1) | Placing sets the cell fully solid. The extractor blends it: a rounded bump merged into the slope. *Example: a player heaps three dirt voxels against a hillside and the hill grows a knoll, not three cubes.* | The placed cube stays a cube with rounded edges. It does not merge into the slope. **Fails V2.1.** |
| **Sharp features: cliffs, caves** | A cliff is a jump in the gap between two neighbouring columns; the extractor places the crossing at the column boundary and the face is vertical. Creases get a chamfer of at most half a cell with surface nets; dual contouring keeps them sharp but needs gradient data and a small solver. Caves: the carver's field IS a gap, so a tunnel mouth is round. **But the base's cheap cave lattice caps a cave's shape at four metres — see M1 and D12.** | Cliffs become 45° ramps under any smoothing. Overhangs bulge. **Fails the "realistic" half of V2.1.** |
| **The collider on the SAME surface** (SL10 clause 5) | The shard runs the same extractor over the same tier-0 lattice and hands the triangles to the physics engine as a static mesh. It holds only WITH the residency invariant of §5.5; it is NOT free. | The collider is the cube (base §4.9.4). The smoothed picture and the cube collider differ by up to half a cell everywhere — the "maximum-hazard" divergence the base itself names (`block_system_design.md:9003-9004`). **Fails SL10 clause 5** unless the collider is also smoothed, which is (A). |
| **Determinism on two hosts** (SL10 clause 4) | Surface nets: an edge crossing is `t = d0 / (d0 − d1)` (one divide), a vertex is a sum of up to 12 crossings divided by the count, in a fixed order. Dual contouring's QEF is the risk: practical solvers reach for `atan2`/`hypot` (`block_system_design.md:5425-5426`). Marching cubes: a 256-case table plus the same edge crossings. All three need the fence of §2.4 and the M3 measurement; none is safe by its operation list alone. | Also needs the same fence. Tie. |
| **Cost per chunk** | Generation: strategy C is MEASURED at 0.885 ms per 62³ chunk with caves (`block_system_design.md:17414-17423`), of which the height field is 0.252 ms and the fill-by-comparison 0.037 ms. **That number depends on sampling the cave field every 4th cell** (`block_system_design.md:17428-17439`), which under a smooth extractor becomes visible geometry (M1, D12). Writing a signed difference instead of a bit is the same class of work: ESTIMATED +0.05 ms, UNMEASURED. Extraction: UNMEASURED. ESTIMATED 0.3–1.5 ms per surface chunk. | Binary greedy meshing: 65 µs per chunk (a published figure the base cites, `block_system_design.md:7506`; not our measurement). Cheaper. |
| **The seam with square built blocks** (V2.3) | Needs a rule (§3). The hybrid is what 7 Days to Die ships; its documented defect ("do not blend perfectly", `block_system_design.md:9005-9007`) is a placement-height defect this design closes with the gap-under-block rule. | Free: cubes meet cubes. |
| **Memory per chunk** | One byte more per cell than a palette-packed cube chunk when stored dense (238,328 B for 62³). Unedited terrain is never stored (SL10); only diffs carry the byte. Client-side scratch only. | Baseline. |

### 2.3 The recommendation, and what it shuts

**Recommend (A) with naive surface nets over cell-centred samples.**

- **The sample, named exactly.** One cell = one substance + one RADIAL GAP. The gap is `r − h`: the cell
  centre's radius minus the terrain height the generator states along that direction. It is NEGATIVE
  INSIDE SOLID, which matches `crates/core/src/geometry.rs:442`. The carver enters as a `max` (air wins,
  and air is the positive side): `gap = max(r − h, carver_gap)`.
  Two honest limits, stated because the digest pins this meaning forever:
  1. A radial gap is not a true distance. On a 45° hillside the two differ by about 1.41, so the clamp
     truncates the field at a different true depth on a plain and on a mountainside. The zero set is
     still exactly the surface.
  2. A `max` of two fields is not a field of that kind near the seam, so the byte is least trustworthy
     exactly where a tunnel mouth meets a hillside. The surface stays continuous there; only the
     vertex's placement along an edge loses accuracy, and the loss is bounded by the cell.
  *Example: a cell one metre under a moon's plain reads about `−1.0` cell and clamps; a cell one metre
  above it reads about `+1.0` cell and clamps; the cell the surface passes through reads the fraction.
  The pin test that could fail: a cell under the surface reads NEGATIVE, a cell above it reads
  POSITIVE.*
- **The quantum is 1/128 cell, not 1/127.** An `i8` in steps of 1/128 spans `[−1.0, +0.9921875]`
  exactly. One step is `2⁻⁷ m` at tier 0 = **exactly 8 fine cells** of `Tier::Fine`
  (`crates/core/src/pose.rs:513`), and `2^(L+3)` fine cells at tier L. A step of 1/127 is 7.874 mm,
  which is 8.06 fine cells — not a whole number at any rung, and it contradicts this report's own rule
  that every quantised terrain length is an exact count of fine cells. The top code `+127` is one step
  short of a full cell; the extractor needs only the sign and the near-zero magnitude, so that costs
  nothing.
- **Cell-centred, not corner-centred.** The cell is the unit of matter (`block_system_design.md:5633-5637`).
  Mining and placing act on a cell. With cell-centred samples an edit changes exactly the cells the
  player clicked. *Example: a player mines the cell under a rock ledge; that one cell's sample becomes
  air; the ledge's underside sags by the interpolation, and the eight cells around are untouched.*
- **The extractor.** Naive surface nets: for every 2×2×2 group of cells whose eight signs are not all
  equal, one vertex at the mean of the edge crossings; one quad per sign-changing edge between two cell
  centres, joining the four vertices around it. Normals from the gap gradient by central differences, or
  from the triangle itself. No case table. About 150 lines.
- **Why not dual contouring first.** It keeps sharp creases, but it needs Hermite data and a
  least-squares solve per vertex; the solve is the one place a platform `libm` call creeps in. Surface
  nets first; dual contouring is a reserved upgrade. Switching changes what a player stands on for the
  SAME saved data, so it is a content door: it must close before the first player flies a planet in the
  window.
- **Why not marching cubes.** Correct, but it emits up to five triangles per cell group, makes long thin
  triangles that hurt the collider, and needs a 256-entry table to cover.

**What (A) shuts.**

1. **Generated terrain stops being cube-only at every tier.** `block_system_design.md:7072-7074` is
   reversed.
2. **The terrain cell record gains a gap byte before the first world is saved** (one-way door, §7), and
   the base's 5-byte persisted edit record becomes 6 bytes (§3.1).
3. **The rim warp (§5.6) is no longer the terrain's look.** It may survive for natural-substance SQUARE
   blocks (D4).
4. **The collision rule is restated, with FOUR arms.** The decision board's "a thing collides iff it
   occupies a cell or is an entity" (`decision_board.md:132-142`) is too narrow twice over: it has no arm
   for the extracted surface, and no arm for a tree. The restated rule:
   *a thing collides if and only if it is a square cell, an entity, the extracted tier-0 terrain surface,
   or a SEED-SHAPED SURFACE OBJECT whose collider the realm's shard derives from the same generator crate
   the client drew it from.*
   The fourth arm is V2.2's home. It needs no diff while the object's shape is pure static shape, and it
   DOES need one the moment growth or damage touches it, because those are live state (SL10 clause 6).
   *Example: a player walks into a pine on a moon and stops, because the moon's shard evaluated the same
   crate the client drew the pine from and got the same trunk; when a player cuts that pine half through,
   the damage arrives on the diff lane of §9.1.*
5. **The skirt-versus-transvoxel argument loses its premise.** The base chose skirts because "our
   surfaces are axis-aligned quads at every tier" (`decision_board.md:1511-1513`). With a smooth surface
   two rungs meet along a curve. Skirts are still the recommendation (§5.2), on the derived bound.
6. **The mesher has three lanes.** The base's two (cube, shaped) plus the smooth lane. Lane selection is
   by the cell's FORM, a data field (HR3).

### 2.4 The determinism fence, in three parts, and what proves it

SL10 clause 4 names four dangers: a fast-math flag, a fused multiply-add contraction, a call into the
platform's transcendental functions, and an evaluation order that is not fixed. A float newtype answers
one of the four. The earlier draft wrote "holds by construction". That is an argument where the standing
rule of method demands a measurement. The fence is:

1. **The newtype.** The generator crate's float type exposes `+`, `−`, `×`, `÷` and `sqrt` BY NAME and
   nothing else. It withholds `mul_add` explicitly, because `f64::mul_add` is a fused multiply-add and
   the clause forbids the contraction it performs. A newtype hides a library call; it does not hide a
   method on the primitive.
2. **The crate's own build profile.** The generator crate pins its optimisation profile and carries an
   allowed-flag list; a crate-level test refuses a build whose profile is not on the list. A newtype
   cannot govern a `RUSTFLAGS` value somebody adds later; only a pinned profile plus a gate can.
3. **The measurement.** M3 (§6) is the guarantee: the same `(seed, chunk key, tier)` produces the same
   `i8` field and the same vertex list on `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`, at two
   optimisation levels. Until M3 runs, no-drift is **UNMEASURED**.

*Example: a player on a Mac and a player on a Linux box stand on the same ridge of the same moon; M3 is
the test that says the two ridges are the same bytes, and nothing else says it.*

---

## 3. Question 2 — the junction: one grid, two forms

### 3.1 The record, with its arithmetic closed

The base's block identity is a triple index `BlockTypeId(u16) → (substance, form, function)`
(`block_system_design.md:676-700`). This design adds ONE shape family to the FORM axis: **`Terrain`**,
the smooth form. A `(dirt, Terrain)` row exists; a `(steel, Terrain)` row does not, so steel can never be
smooth, and the manifest is the permission (`block_system_design.md:700-706`).

**The persisted EDIT record, with the sum.** The base fixes it at 5 bytes exactly
(`block_system_design.md:2033-2050`):

```
local_index 18 b │ block_type 16 b │ orient 5 b │ placed 1 b                      = 40 bits = 5 bytes  (the base)
local_index 18 b │ block_type 16 b │ orient 5 b │ placed 1 b │ terrain_gap 8 b    = 48 bits = 6 bytes  (this design)
```

- **Orient is 5 bits in BOTH places in this report.** The earlier draft said 6 bits in the field table
  and 5 bits in §4.1. Five is right, and the sixth bit is NOT free to reserve again: the base already
  spends it on `placed`, the generator-authored/player-placed bit that keeps structural support from
  collapsing every natural cave roof (`block_system_design.md:2046-2050`).
- **The gap byte takes the record from 5 bytes to 6.** There is no slack to hide it in. This is the door,
  stated with its arithmetic, and the formats domain freezes it.
- **The gap is present on every planet cell, not only on terrain-form cells.** That one rule decides
  every junction case below. On a realm whose cells are all catalogue forms the byte reads "fully air"
  and the smooth lane emits nothing for them.

The base's §4.1.3 argument against a gap field — "block identity dissolves; a functional block needs an
id, an owner, a damage state, a signal address" (`block_system_design.md:5416-5420`) — does not apply:
the identity stays in the same record beside the gap. A terrain cell is still "this cell, here, is dirt".
Only its SURFACE is smooth.

### 3.2 The three lanes, selected by the cell's form

| Lane | Selected when | Emits | Collider |
|---|---|---|---|
| **Smooth** | the cell's form is `Terrain` — and, for the extractor's sample lattice, EVERY cell contributes its gap byte | surface-nets quads with normals, as `MeshPrim` (no widening: `Vertex` already carries a normal, `crates/client/src/realm_scene.rs:770-773`) | the same quads as a static triangle mesh |
| **Cube** | the form's `is_cube` (base §4.5.1) | greedy-merged axis quads | **ONE builder: greedy boxes in index space through `cell_point`.** On a Cartesian realm `cell_point` is the identity map. |
| **Shaped** | any other catalogue form | instanced baked templates through `cell_point` | **ONE builder:** baked convex templates / parts, placed through the same `cell_point`. |

**One builder per lane, never one per realm geometry.** The earlier draft carried the base's split — a
library voxel shape on Cartesian, greedy boxes through `cell_point` on spherical
(`block_system_design.md:6465-6472`). `VoxelGeometry` is a config field
(`crates/sim/src/capability.rs:36-43`, set from `CapRequest`), and the FINAL-backend law forbids a second
implementation selected by config. If a library voxel shape is adopted later, it is an OPTIMISATION that
must produce the IDENTICAL collider, and slice 4's digest gate is the measurement that proves it.
*Example: a steel cube in a station's wall and a steel cube on a moon's hillside are built into a
collider by the same code, and one digest test covers both.*

The dispatch reads a field of the cell. Nothing tests the shard's kind (HR3), and the same fixture runs
on both profiles (HR4, made real in §5.6).

### 3.3 The junction cases

**Case 1 — a square hull block placed on a smooth hillside.**

- Placement targets the first cell on the air side of the pick ray whose gap is above the "placeable"
  threshold (recommended: gap ≥ 0, the cell centre is on the air side). A cell whose centre is inside
  rock refuses the placement: "solid ground, mine first."
- The placed cell becomes `(steel, Cube)`; **its gap byte is kept as it was.**
- The smooth lane keeps reading that byte. The hill's surface runs INTO the cube's volume, is hidden
  inside the opaque cube, and exits its side faces along a clean cut line. *Example: a foundation set
  into a 30° slope shows dirt up to the cut line on its uphill face and a clean face on its downhill
  side, which is what a foundation cut into a slope looks like.*
- The collider: the terrain mesh inside the cube is covered by the cube's box, so two colliders overlap.
  Whether the character walks that cleanly is **UNMEASURED**. The earlier draft called it "harmless",
  which is an argument, not a result. M9 (§6) measures it: a character sweep across a foundation cut into
  a 30° slope, at 20 offsets, asserting no vertical impulse and no step the player can climb (the base's
  own assertion shape, `block_system_design.md:6502-6505`).
- The cheaper cure, priced and reserved: clip the terrain quads against square-cell faces so no terrain
  triangle exists inside a square cell. It also removes the hidden collider outright. Cost: a per-quad
  clip against up to six planes, ESTIMATED under 10 % of the extraction time, UNMEASURED. Build it if M9
  goes red.

Two alternatives were tested and lose:
- *Treat a square cell as fully SOLID for the extractor.* The terrain climbs the cube's sides as a
  half-cell fillet outside the cube's footprint, and a player walks up a 0.5 m ramp against every wall.
- *Treat a square cell as fully AIR for the extractor.* The terrain bends away and leaves a visible gap
  around every placed block.

**Case 2 — a smooth terrain voxel placed against a square wall.** The placed cell becomes
`(dirt, Terrain)` with the gap fully solid. The bump runs into the wall's cube cell and is hidden there;
the visible part is a heap against the wall. Correct without a rule.

**Case 3 — a mined cell under a placed block.** Mining sets the terrain cell's gap fully air. The smooth
lane reflows: the surface under the block sags into the hole. The block does not move. Whether it falls
is the structural-support mechanic (`block_provenance_collapse.md`, R6), not this domain. The support
flood reads the gap byte's sign as "solid or not" for terrain cells.

**Case 4 — placing a terrain voxel into a cell that holds a square block.** Refused. One cell, one form.

**Case 5 — removing a square block.** The cell returns to form `Terrain` with the gap byte it kept.
*Example: a player breaks a foundation; the slope under it is exactly the slope before the foundation.*

**Case 6 — a sub-metre block on a slope (V2.4).** A `SubGrid` cell is a catalogue form like `Wedge` is
(§4.3), so it obeys Case 1 exactly: **it keeps its gap byte**, the smooth surface runs through it and is
hidden inside whichever sub-cells are solid, and the collider is the union of the cell's merged sub-boxes
plus the terrain surface under them. The defect to watch is the reverse of Case 1: a `SubGrid` cell is
mostly EMPTY, so the hidden terrain shows through the empty sub-cells. **Recommend: the terrain quad clip
of Case 1 is REQUIRED for `SubGrid` cells, not optional.** *Example: a player details a landing pad's edge
with quarter-metre trim on a 30° hillside; the trim's own boxes hold the boots, and the hill stops at the
trim's faces instead of poking through the gaps.* The sub-metre domain owns the sub-cell record; this
report owns the rule that the gap byte survives under it.

**Case 7 — a tree on a mined edge (the V2.2 interface).** The terrain owes the tree domain exactly two
statements, and no more:
1. The anchor cell's gap gives the trunk's base height.
2. **If a player mines the anchor cell, the anchor is GONE** and the tree domain decides what happens (it
   falls, it is destroyed, it drops to the next solid cell). The terrain does not decide it, and the
   terrain must not silently leave a tree standing on air.
*Example: a player digs the cell under a pine's trunk; the moon's shard tells the tree domain that the
anchor cell changed, and the pine is not left hanging in the air while the hill reflows around it.*

**Case 8 — an edit lands while a player crosses out of the moon.** The diff lane is reliable and one hop,
but a subscription flips during a crossing. **Rule: a diff is addressed to a CHUNK, never to a session,
and a client that (re)subscribes to a chunk receives that chunk's whole diff set BEFORE any incremental
diff for it** (§9.1, the baseline). So a diff lost across a flip is repaired by the baseline on the next
subscribe, and the client's shape can never stay behind the shard's collider.

### 3.4 Mining yield on a partial cell

The base rules "one full cube yields one unit wherever it is mined" (`block_system_design.md:5633-5637`).
A surface cell now holds a fraction. Recommend: mining yields `fill` units, where `fill` is the cell's
volume fraction derived from its gap and quantised to 1/8 unit; placing costs one unit and sets the cell
fully solid. Mass is then conserved to 1/8 unit. *Example: a player scrapes the top of a dune; each click
on a half-full cell yields half a unit of sand.* This is an owner decision (D6).

### 3.5 What is NOT this domain's, and is named so nobody assumes it

- **A hull standing on a planet.** A ship is its own realm with its own Cartesian grid. Its contact with
  the planet's terrain is a collision between a child realm's body and the parent's surface. The planet's
  shard needs the hull's collider shape for that, and that shape crossing child → parent is an SL6 ask
  the base already flags (`decision_board.md:814-817`). This report does not make that ask. The landing
  domain does.
- **Trees.** V2.2 makes a tree ONE object on the surface with a seed-decided shape. The terrain owes it
  the collision arm of §2.3 item 4 and the two statements of Case 7. The tree domain owns the rest.
- **The saved record's exact bit layout and the edit-pyramid entry.** This report states the fields this
  domain needs (the gap byte, the `Terrain` form, the 6-byte sum, a gap-DELTA arm in the pyramid, §5.3).
  The formats domain freezes them.

---

## 4. Question 3 — the ~20 shapes, re-validated

### 4.1 One identity, one orientation

The base's catalogue is 23 topologies (`block_system_design.md:5661-5686`): Cube; Slab, Plate, Panel;
Wedge, RampLow, RampHigh; CornerOut, CornerIn, RampLowOut, RampLowIn, RampHighOut, RampHighIn; Tetra,
TetraIn; Quarter, Post, Nub; Stairs; Wall, Fence, Railing, Conduit. That is "about twenty" and it matches
V2.3.

**Identity:** the shape lives inside the `BlockTypeId` triple. Re-validated: it stands.

**Orientation: 5 bits, everywhere in this report.** The base contradicts itself — §2.1.1 says 5 bits with
mirror as a shape row (`block_system_design.md:710-720`); §4.1.2 says 6 bits with a mirror bit
(`block_system_design.md:5379-5382`). Take §2.1.1. Every shape in the catalogue is achiral today
(`block_system_design.md:5739-5740`); a future chiral shape gets its twin as a ROW, which is the only way
the collider hull and the winding stay baked rather than branched. **The freed sixth bit is already
spent** on `placed` (`block_system_design.md:2046-2050`); this report does not reserve it a second time.
*Example: a player mirrors a wing built of wedges; each wedge's mirror is the same wedge under another of
the 24 rotations, and the bake asserts that closure.*

**Rotations are integer matrices** (a 24-entry constant table, `block_system_design.md:5389-5392`), so a
rotated template, a rotated coverage mask and a rotated collider are exact. Re-validated: it stands.

### 4.2 The collision per shape

| Class | Shapes | Collider |
|---|---|---|
| Full cube | Cube | the cube lane's ONE builder (greedy boxes in index space through `cell_point`) |
| Convex | Slab, Plate, Panel, Wedge, RampLow, RampHigh, CornerOut, RampLowOut, RampHighOut, Tetra, Quarter, Post, Nub (13) | ONE builder: one convex hull baked in index space and mapped per cell through `cell_point` (the base's Lane B collider, `block_system_design.md:6469-6470`). `cell_point` is the identity map on a Cartesian realm, so there is one code path, not two. |
| Non-convex | CornerIn, RampLowIn, RampHighIn, TetraIn, Stairs, Wall, Fence, Railing, Conduit (9) | a compound of convex parts, about 1.6 parts average (`block_system_design.md:6480-6483`) |
| Pass-through | (was: bushes R13, leaves R14) | an EMPTY collider. **Under V2.2 leaves are not cells any more.** The class survives only if a harvestable bush is ever a cell; today it holds nothing. |

Every walkable class derives its thresholds from the catalogue and the warp bounds, not from typed
constants (`block_system_design.md:6690-6711`). The one transcendental (`atan` for the controller's
angle) runs once per realm at activation and rides the checkpoint (`block_system_design.md:6752-6757`).
Re-validated: SL10 clause 4 binds the GENERATOR; the controller's configuration is not the generator.

**The one thing V2.1 changes here:** the walkability decision (the base's 4-C) was cheap because
"generated terrain is cube-only, so the thresholds only ever judge player-built ramps"
(`block_system_design.md:6719-6722`). With smooth terrain the thresholds judge every hillside. Recommend
the same answer (C1, everything but vertical, climb ≈ 67°) and note that natural slopes above it become
cliffs the player must dig into. This is an owner decision (D7).

### 4.3 Do the shapes and the sub-metre blocks (V2.4) overlap in purpose?

Partly, and the overlap must be settled before the catalogue freezes (`ShapeId` is append-only and
persisted, `block_system_design.md:6033-6034`).

| Catalogue shape | What a 2×2×2 half-metre sub-grid gives | What a 4×4×4 quarter-metre sub-grid gives |
|---|---|---|
| Nub (0.5 m cube) | exactly | exactly |
| Quarter (0.5 × 0.5 × 1) | exactly | exactly |
| Post (0.5 × 0.5 × 1, other axis) | exactly | exactly |
| Slab (½ thick) | exactly | exactly |
| Plate (¼ thick) | no | exactly |
| Panel (⅛ thick) | no | no (needs 8×8×8) |
| Wedge, ramps, corners, tetra, stairs | never (a sub-grid gives staircases, not slopes) | never |
| Wall, Fence, Railing, Conduit | no (connect families are derived from neighbours) | no |

**Recommend:** the catalogue keeps what a sub-grid cannot express — the slopes, corners, tetra, stairs,
the thin plates and the connect families — and the sub-metre mechanism (V2.4) owns the axis-aligned small
boxes. Drop Nub, Quarter and Post from the catalogue if a half-metre sub-grid ships; keep Slab because it
is a plate, not a box. The base rejected sub-grids for wire and storage reasons
(`block_system_design.md:5400-5412`) and states flatly that "the base cell is never subdividable"
(`decision_board.md:132-142`); the owner now asks for them "if possible" (V2.4), so that base ruling is
STALE (§10) and the wire cost is the sub-metre domain's to price. The interface this domain needs: **a
cell holds ONE form**, `SubGrid` is a form like `Wedge` is, and Case 6 of §3.3 rules that a `SubGrid` cell
keeps its gap byte. This is an owner decision (D5).

---

## 5. Question 4 — detail levels for smooth terrain, without a pop

### 5.1 The ladder stays, and gets simpler

Addendum 2's rule — tier L drops the L finest octaves of the same generator — is about the FIELD, not
about cubes. It applies to the gap field unchanged: `h(dir, L)` is the same function with a shorter sum
(`block_system_design.md:3861-3880`), the carver contributes iff its radius is at least `2^L` metres
(`block_system_design.md:3583-3591`), and the error is bounded: `|h(dir, L) − h(dir, 0)| < 4·k_rough·2^L`
METRES (derived, `block_system_design.md:3890-3898`; the gate `terrain_tier_agreement` is a property test,
not a measurement). At tier L the smooth lane runs the same surface-nets code over a lattice of
`2^L`-metre cells. No second extractor, no per-tier code path (HR3). *Example: a player in a descending
hull sees a ridge at tier 4 (16 m cells); the client evaluated the seed with four octaves dropped,
extracted it, and draws it. As the hull descends, tier 3 chunks under it arrive and take over.*

Smooth terrain removes one of the base's own worries: the coarse cube rungs made a hillside a staircase
of `2^L`-metre risers that transvoxel could not stitch. A coarse smooth rung is a smooth surface, and its
difference from the fine one is a bounded vertical offset, which is the easy case for every seam
technique.

### 5.2 The seam and the pop, against SL8's eleven kinds

**The apron's depth is a CONSTANT NUMBER OF CELLS at every rung.** The base's bound is in METRES:
`4·k_rough·s_L` metres, where `s_L = 2^L` metres is the rung's cell size
(`block_system_design.md:3890-3898`, which states the consequence itself: "**two cells**, whether the
cells are 1 m or 128 m"). A rung-L chunk's lateral apron must hide the crack against a rung-(L+1)
neighbour, whose relief error is `4·k_rough·s_{L+1}` metres `= 8·k_rough·s_L` metres `= 8·k_rough` CELLS
of rung L. So:

> **apron depth = `8 · k_rough · skirt_safety` cells of the chunk's OWN rung** — 4 cells at
> `k_rough = 0.5` and `skirt_safety = 1`, at EVERY rung.

The earlier draft wrote `(4·k_rough·2^(L+1)) · skirt_safety` **cells**, which mixes the metre form with
the cell unit and gives 64 cells at rung 4 instead of 4. That error would have sized the residency budget
sixteen times too high.

| Seam kind (SL8) | Where it can appear here | The mechanism | Tolerance, physical |
|---|---|---|---|
| **detail-by-box** | a chunk at tier L beside one at tier L+1 | tier is per chunk COLUMN, adjacent columns differ by at most one rung (`block_system_design.md:6409-6419`); the crack is closed by the constant-depth apron above | no visible hole at any camera pose above the surface; MEASURED by the pop detector, UNMEASURED today |
| **arrival pop** | a fine chunk arrives over a coarse one | coarse-before-fine ordering (`block_system_design.md:7606-7616`) + a dither crossfade across a band (Addendum 2 §C.6) | the vertical disagreement at the switch distance, in pixels. At 2 px per cell and `k_rough = 0.5` the bound is 2 cells = **4 px** (ESTIMATED from the derived bound). That is visible. **M4 RUNS BEFORE SLICE 3 BUILDS THE LADDER.** If M4's residual exceeds the pop detector's threshold, geomorphing is PART OF slice 3, not a reserve. |
| **brightness pop** | normals change between rungs | normals from the gap gradient at the chunk's own tier; the crossfade blends them | sub-threshold by the same dither |
| **jump / black frame / flicker** | never from this domain — the placement is the parent's (SL1), the terrain is static shape | — | — |
| **tick hitch** | a chunk generation or extraction on the main thread; **and the ARRIVAL RATE during a descent** | never on the main thread (base §5.1.3); two job classes with caps in the render tuning struct; **M10 prices the descent's chunk rate against a named thread budget** | the 1 % low; the base's frame budget |
| **re-state rate / lane flood** | terrain diffs, the baseline on subscribe, and the standing pyramid | a diff is per edited cell, one hop, reliable lane; the baseline is per chunk on subscribe (§9.1); M8 measures live digging, **M11 measures the STANDING cost of a long-dug moon** | bytes per second per player, and bytes at first sight; UNMEASURED |
| **tier refusal** | a request for a rung above the body's `tier_depth` | the top rung is the realm proxy that already exists (`crates/client/src/realm_scene.rs:800-806`) | never refused; the proxy is always resident |
| **sprite cull** | not this domain | — | — |

### 5.3 Edits at coarse tiers: the pyramid stores DELTAS, never absolutes

The base's edit pyramid stores an octant mask + a fill + a dominant substance per coarse cell for player
BUILDS (`block_system_design.md:5776-5806`). For terrain edits the entry is **the mean of the DELTAS**,
plus the dominant substance. A delta is `edited gap − seed gap` at the SAME rung.

**Why a delta and not the absolute.** At rung L an UNEDITED cell's sample comes from `h(dir, L)`, the
generator with L octaves dropped. An absolute summary of eight rung-0 samples comes from the FINE world.
The two bases differ by up to `4·k_rough·2^L` metres even where nobody dug
(`block_system_design.md:3890-3898`). Laying one over the other makes a STEP at the edited cell's rim, up
to four metres at rung 2 — a detail-by-box seam, which SL8 forbids. A delta adds onto whatever base the
rung supplies, so the rim is flat by construction. *Example: a tunnel a player dug last week shows a
DIMPLE in the ridge at rung 2, not a four-metre ledge around the dimple.*

**The integer arithmetic, corrected.** Eight `i8` values sum to as much as ±1024, so the accumulator is
`i16`, not `i8`. And a delta is measured in the cells of its OWN rung, so summarising to the next rung
halves it. The walk-up is:

```
delta(L+1) = (sum of the eight children's delta(L), as i16, + 8) >> 4      // the mean, halved; arithmetic shift
```

The result cannot leave the `i8` range (|delta| ≤ 128 ⇒ |result| ≤ 64), so no clamp is needed, and the
shift is exact on every target. The walk-up terminates on summary equality, as the base rules
(`block_system_design.md:5922-5929`). **Slice 5's gate measures the RIM STEP, not only the tunnel's
visibility.** The pyramid entry needs a gap-delta arm; the formats domain owns its layout.

### 5.4 What the server never coarsens, and how much collider it holds

- **Collision is tier 0, by type.** The base's `Tier0Key` newtype (`block_system_design.md:6573-6601`)
  stands: a collider, a mass property, a raycast, an edit and a record are defined over tier-0 addresses
  and nothing else. The smooth lane's collider builder takes a `Tier0Key`.
- **The collider's residency is a SWEPT SEGMENT, not a fixed 64 m ball.** The base's 64 m was "derived as
  'greater than any body's per-tick swept motion at 20 Hz plus the character's reach'"
  (`block_system_design.md:6485-6488`). **That quantity no longer has a value.** The owner took the
  ceiling off the flight path and made containment swept
  (`owner_decisions_2026-08-27_movement_answers.md`, section M-D, rulings 1 and 3), so no per-tick motion
  bound exists. A hull that closes at two kilometres a second covers a hundred metres in one tick at
  20 Hz and passes clean through a 64 m ball of collider.
  > **The rule: the shard extracts tier-0 collider over the body's OWN swept segment for this tick — the
  > line from the last pose to this pose — fattened by the body's bounding radius plus the character's
  > reach.** It is a per-body set, never one number.
  *Example: a hull dives at a moon; the moon's shard holds collider along the whole line the hull travels
  this tick, so the hull cannot be on the far side of the ground when the tick ends.*
  M5 measures the worst case with a LANDING HULL, not a walking player.
- **The physics engine's own state is Category C** (checkpoint-carried, never re-simulated cross-host,
  CLAUDE.md "Determinism"). What must be byte-identical is the collider's INPUT — the extracted triangle
  list — and SL10's gate (M3) measures exactly that.
- **The server never runs the crossfade, the skirt or the tier selection.** Those are the client's
  presentation of the same shape.

### 5.5 THE RESIDENCY INVARIANT — the missing half of SL10 clause 5

The earlier draft said "what the player sees is what the player stands on, by construction". It is not
construction: the client picks a rung by an angular rule and the shard collides rung 0, and nothing tied
them together. Where they disagree the drawn surface and the collider differ by the derived bound — two
cells at `k_rough = 0.5` (`block_system_design.md:3890-3898`). The boots float or sink. That is an
arrival pop and a broken clause in one event.

> **The invariant: the CLIENT's resident tier-0 band STRICTLY CONTAINS the shard's collider set for every
> body the client's observer can touch, at every camera pose and every closing speed.**

Three parts, each with a home:

1. **The floor.** Every chunk inside a body's collider set (§5.4) is resident at tier 0 on the client.
   This report does NOT inherit the base's "one chunk-width of the camera" floor
   (`block_system_design.md:6408-6410`): 62 m is narrower than the base's own 64 m collider ball, which
   is how the invariant went missing in the first place.
2. **The lead.** The band grows with the closing speed, and the speed is NEVER capped — the owner's own
   answer of 2026-08-27, item (2). The lead is the closing speed times the interpolation buffer
   (`crates/wire/src/channels.rs:328`, `INTERP_BUFFER_MS = 120.0`, a constant the two ends already share)
   plus one tick.
3. **The gate.** A LANDING FIXTURE. A hull descends at its rated cruise onto a moon and touches down; a
   character walks off the ramp. Assert: the drawn surface height and the collider surface height agree
   at the contact point, at every approach speed the suit or the hull states
   (`owner_decisions_2026-09-05_suit.md`: a suit and a hull each state their own rating). M6b measures
   the see-versus-stand vertical gap directly. It is a DIFFERENT quantity from M4, which measures the pop
   between two rungs.

### 5.6 The client derives terrain only for a realm it is DRAWING, and HR4 needs a real second kind

**The derivation is gated on the drawn set.** SL3 says a realm that is not running cannot be drawn, and
the ruling of 2026-09-01 says a dormant realm is never drawn by anybody. SL10 clause 1 permits the client
to derive static shape; it lifts neither. So:

> **Rule: the client derives terrain for a realm ONLY while that realm holds a row in the composed
> scene** (`ServerControlMsg::RealmRegistry` / `RealmSceneDelta`, `crates/wire/src/channels.rs:158,179`),
> which is exactly the reach ruling's in-range set.

*Example: a moon two star systems away is not in the window, so the client evaluates nothing for it, even
though it holds the crate that could.*

**The HR4 fixture must not be empty.** The earlier draft's gate was "the same fixture on a Spherical and a
Cartesian profile (the Cartesian one emits nothing)". A run that produces nothing is not the feature
running on a second shard kind; it is the feature being absent there, which is the fork HR3 and HR4 exist
to prevent. Nothing in the code refuses terrain on a Cartesian realm: `ShardProfile::build` accepts
`surfaces: true` with `VoxelGeometry::Cartesian` (`crates/sim/src/capability.rs:134`), and `ship()`
already sets that pair (`crates/sim/src/capability.rs:260-272`).

> **Recommend: a Cartesian realm MAY hold terrain-form cells.** The game already has the example — a
> station's growing bay with a soil floor, or a hull with a dirt-filled ballast hold.

Then the fixture is truly identical: plant the same gap pattern in index space on a moon and inside a
station hull, run the same extractor, and assert the SAME quad list on both. The extractor reads
cell-centred samples in index space, so it does not care which grid it stands on. *Example: a player digs
a scoop in a station's growing bay and gets the same scoop the same click makes in a moon's dune.* This is
an owner decision (D13), because it adds soil inside hulls to the game.

---

## 6. Question 5 — what is UNMEASURED, and the smallest bench for each

The precedent is `scripts/noisebench/` (its own workspace root; MEASURED on an Apple M4 Pro, `rustc
1.94.1`, opt-level 3, fat LTO, single core; `docs/investigation/README.md:74-76`). Every bench below
extends that rig, and every one is owed BEFORE the slice it gates. The rig is a measurement stand, not a
second world (SL5): when the generator crate exists, the bench MOVES INTO THE CRATE, the gates run the
crate's own bench, and the standalone copy is deleted.

| # | Unmeasured | Smallest bench | Gates |
|---|---|---|---|
| M1 | The gap chunk's generation cost against the blocky fill's 0.885 ms (MEASURED) — **and the cave lattice step, which is now VISIBLE geometry** | `noisebench` bin `gapfield`: strategy C, writing `clamp(r − h)` combined with the carver term as an `i8` per cell. **Run it at cave-lattice steps 1, 2 AND 4** (the base's own table for the cave field alone: 11.63 / 1.95 / 0.625 ms, `block_system_design.md:17428-17439`). Report ms per 62³ chunk for each, and a picture of a lava-tube mouth at each step. | the generator slice's throughput gate (≤ 1.20 ms per tier-0 chunk, `block_system_design.md:17461-17475`) AND the pinned digest, because A.5 conclusion 2 already puts the step inside the one-way door |
| M2 | Surface-nets extraction time and output size per chunk, typical and worst | `noisebench` bin `surfacenets`: run over M1's chunk with a one-cell apron; report ms, vertices, quads, bytes at 20 B per vertex. Worst case: a 3-D checkerboard gap field. | the mesher slice's throughput gate; the expensive-queue threshold |
| M3 | **No drift** (SL10 clause 3) — the guarantee behind §2.4's fence | the same bin prints a 128-bit digest of the `i8` field AND of the vertex list, per `(seed, chunk key, tier)`, for ~64 keys covering all six faces, twelve edges, eight corners, at every legal tier. Run on `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu` at opt 0 and 3. Diff. | SL10's no-drift gate; the generator's DoD. **Until it runs, no-drift is UNMEASURED, not "by construction".** |
| M4 | The vertical disagreement between rung L and rung L+1, in pixels at the switch distance | a property sample: 10⁵ directions, `|surface(L) − surface(L+1)|` in cells, converted at the pixel target. Report the max and the 99th percentile. | **RUNS BEFORE SLICE 3 STARTS.** It decides whether geomorphing is part of slice 3 (§5.2, D9). |
| M5 | The collider build time and memory of one terrain chunk, and the WORST-CASE swept set for a landing hull | BLOCKED on the owner's library decision (rapier3d / parry3d are not dependencies, `Cargo.toml:24-90`). When decided: a standalone bench with that crate as its only dependency; build from M2's output; report ms and bytes; then sweep a LANDING HULL at its rated cruise and report the chunk count of the swept set per tick (§5.4). | the collider slice |
| M6 | Edit reflow latency: one mined cell → re-extract the 3×3×3 neighbourhood → upload | M2's bin with a dirty set of 8 chunks; report wall time on a 4-thread pool. Budget: "edit echo → pixels ≤ 33 ms" (`block_system_design.md:7644-7653`). | the edit slice |
| **M6b** | **The see-versus-stand vertical gap** (the §5.5 invariant) | the landing fixture: a hull descends at its rated cruise onto a moon; at the contact point compare the CLIENT's drawn surface height with the SHARD's collider surface height. Repeat at every approach speed a suit or a hull states. | slices 3 and 4. **A different quantity from M4.** |
| M7 | Resident triangles and bytes for a 100 km view of smooth terrain | M2's per-chunk figures × Addendum 2's chunk counts per tier. ESTIMATED ~7,700 triangles per surface chunk (2 per surface cell); UNMEASURED. | the residency governor's byte cap |
| M8 | The terrain diff lane's bytes per second under LIVE digging | a fixture: one player mines 10 cells per second for a minute on a moon; count bytes on the reliable lane to a second client. | the lane-flood tolerance (SL8) |
| **M9** | **Whether two overlapping colliders under a foundation are walkable** (§3.3 Case 1) | a character sweep across a foundation cut into a 30° slope, at 20 offsets; assert no vertical impulse and no climbable step (the base's assertion shape, `block_system_design.md:6502-6505`). | slice 6. If red, the quad clip becomes required, not optional. |
| **M10** | **The chunk ARRIVAL RATE during a descent** — the tick-hitch answer for V2.9 | a scripted descent at the hull's stated cruise through four rungs. Count chunk requests per second and the deepest queue, against a NAMED thread budget. | slice 3. Without it the report cannot say a landing is free of a tick hitch. |
| **M11** | **The STANDING cost of a long-dug moon** — pyramid bytes a client must hold and receive at first sight | seed a moon with N edited cells scattered over its whole surface, N across four orders of magnitude. Report the pyramid bytes held per rung, and the bytes on the wire when the moon first enters the window (the baseline of §9.1). | the lane-flood and re-state-rate tolerances (SL8); SL9's rule that a growing cost is measured on a realm that has many, never argued |

---

## 7. One-way doors this domain opens or moves

| Door | Must shut | Cost if wrong |
|---|---|---|
| **The terrain gap byte on the planet cell record** — 8 bits, `i8`, a RADIAL GAP `r − h` in cell units, NEGATIVE inside solid, in steps of **1/128 cell** | before the first world is saved | every saved planet migrates; a change of resolution, of sign, or of the byte's MEANING reinterprets every edit ever made |
| **The persisted edit record grows from 5 bytes to 6** (`local_index 18 + block_type 16 + orient 5 + placed 1 + gap 8 = 48 bits`) | with the formats domain's freeze | the WAL's element width is persisted; a later change re-serialises every stored edit |
| **The gap's meaning is `r − h` combined with the carver by `max`, at the cell centre, AND the cave lattice's power-of-two step** — all named in the generator's pinned digest | with the generator's first pinned digest (the base's P4.1 DoD); the step is inside the same door by A.5 conclusion 2 | the digest changes = a world-format epoch, and the inherited fail-safe on an epoch mismatch is DISCARD (`decision_board.md:1162-1165`) |
| **The generator's exact POSITION HASH** (not "SplitMix64" loosely — the stream generator and the bench's `hash3` are different functions, §1) | with the same pinned digest | the same as above: a different hash is a different world |
| **The `Terrain` shape family on the FORM axis**, and the rule "one cell, one form" | before the manifest allocates its first row | a row's triple may never change once allocated (`block_system_design.md:700-706`) |
| **Generated terrain is SMOOTH** — the base's reserved "generated-surface-skin" field (door 52, O10) is taken, value `Smooth` | before the first planet is saved | turning it on later changes terrain under existing player builds (`decision_board.md:403-413`) |
| **The terrain diff's WIRE SHAPE and the two protocol-minor bumps** (§9.1) | with the first shipped diff | postcard discriminants are positional and are never renumbered (`crates/wire/src/session_flow.rs:241-243`); a wrong shape is reserved forever |
| **The extractor (surface nets)** | before the first player flies a planet in the window | a content door: switching changes what players stand on for the same data; no migration |
| **Which mechanism owns the half-metre cube** (catalogue shapes vs `SubGrid`) | before the catalogue freezes | a retired `ShapeId` must bake, collide and render forever |
| **The pyramid entry's gap-DELTA arm** (a delta, never an absolute; §5.3) | with the edit pyramid's format (another domain's door 43) | derived data, but permanent; an absolute arm bakes a four-metre rim step into every coarse rung |

---

## 8. Open decisions for the owner

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| D1 | Which smooth extractor? | (a) naive surface nets; (b) dual contouring with Hermite data; (c) marching cubes | **(a)** | Fewest operations, no case table, no solver, quad output for the collider. (b) is the reserved upgrade if rock creases must be sharp. |
| D2 | Where does the gap sample live? | (a) at the cell centre; (b) at the lattice corners | **(a)** | Mining and placing act on the cell the player clicked and nothing else; yield accounting stays per cell. |
| D3 | Does the terrain gap survive under a placed square block? | (a) yes, the byte is on every planet cell; (b) no, the square block replaces it and removal leaves air | **(a)** | The only rule that makes a foundation cut into a slope look right and restores the slope exactly on removal. Cost: one byte on every cell record, and 5 bytes → 6 on the edit record. |
| D4 | May a natural substance be placed as a SQUARE block? | (a) yes, as a separate `(dirt, Cube)` row; (b) no, dirt is terrain-form only | **(b) for the first slice; (a) as a later row** | (b) keeps the rim-warp machinery out of the first slice entirely. (a) is additive: a new manifest row. |
| D5 | Which mechanism owns Nub, Quarter, Post? | (a) the catalogue; (b) the sub-metre `SubGrid` form | **(b)** if V2.4 ships a half-metre grid; else (a) | Two mechanisms for one 0.5 m cube is the fork HR3 exists to prevent. |
| D6 | Mining yield on a partial surface cell | (a) always one unit; (b) the cell's fill, quantised to 1/8 unit; (c) nothing below half | **(b)** | Mass is conserved; a player who scrapes a dune gets sand in proportion. |
| D7 | Walkability on natural slopes | (a) C1: everything but vertical (climb ≈ 67°); (b) C2: the 1:2 pair is the steepest | **(a)** | Under smooth terrain the threshold judges every hillside. (a) keeps mountains walkable and leaves cliffs as cliffs. |
| D8 | Collision library and the terrain collider's form | (a) rapier3d + a static triangle mesh per tier-0 chunk with internal-edge fixing; (b) rapier3d + a heightfield, mesh only in caves; (c) another library | **(a)**, decided together with the base's 4-B | Terrain is static in its realm's frame, so the base's rejection of triangle meshes (for DYNAMIC bodies, `block_system_design.md:6494-6496`) does not apply. (b) is a second representation. **Nothing is adopted here; the library is the owner's call.** |
| D9 | The tier switch's pop remedy if M4 exceeds the threshold | (a) narrow the pixel target; (b) geomorph; (c) accept | **(b), inside slice 3** | (a) raises the chunk count everywhere; (c) breaks SL8. M4 runs first, so the choice is made on a number. |
| **D10** | **What carries the terrain diff to a client?** (§9.1) | (a) an appended `EventMsg` variant on the existing `ServerControlMsg::Event` arm; (b) a brand-new `ServerControlMsg` arm | **(a)** | The `Event` arm's own doc already rules it the carrier for appended variants — "one carrier, two consumers, built once" (`crates/wire/src/channels.rs:189-194`). |
| **D11** | **How does a client that arrives after a week of digging learn the week?** | (a) the whole diff set per chunk on subscribe; (b) a per-chunk digest first, and the client pulls only where its own digest differs | **(a) for the first slice, (b) when M11 says the bytes hurt** | (a) is simple and correct on every join. (b) is an optimisation whose need must be a measurement, not a guess. |
| **D12** | **The cave lattice's step, which is now visible geometry** | (a) every 4th cell (0.885 ms MEASURED, a cave mouth resolved to 4 m); (b) every 2nd cell; (c) every cell (over the 1.20 ms gate) | **decide on M1's pictures, before the digest is pinned** | Under a blocky field the step only chose which cubes were air. Under a smooth extractor it IS the cave wall a player walks into. |
| **D13** | **May a Cartesian realm hold terrain-form cells?** (a station's growing bay, a hull's ballast hold) | (a) yes; (b) no, terrain is for spherical realms only | **(a)** | Without it the smooth lane's HR4 fixture is an empty run, and the feature lives on one shard kind. The code already permits the pair (`crates/sim/src/capability.rs:134,260-272`). It also adds soil in hulls to the game, which is why it is the owner's call. |

---

## 9. Law conflicts and SL6 asks

### 9.1 ★ THE SL6 ASK — the terrain diff needs new wire surfaces

**The earlier draft said this domain asked nothing. That was wrong.** SL6 has two halves and the draft
answered one: *"ASK BEFORE NEW DATA CROSSES A REALM BOUNDARY, and before adding a wire arm. Default NO."*
The diff crosses no realm boundary — it goes from the owning realm's shard through the connection plane,
which is not a realm (SL2, clarified 2026-08-24). But it needs arms, and there are none.

**What the code says.** `BlockEdit` is a word in a doc comment, not a variant
(`crates/wire/src/intershard.rs:34`), and `InterShardFlow` is the shard-to-shard contract in any case.
The client-facing lane carries no cell (`crates/wire/src/channels.rs:88-195`). The shard→gateway lane's
`RealmSceneDelta` is a TOMBSTONE that nothing produces and that says "Do not revive"
(`crates/wire/src/session_flow.rs:236-248`).

**THE ASK, in SL6's own shape.**

- **WHAT.** One edited cell: the chunk key in the realm's OWN frame, the local cell index, the block
  identity, the orientation, the `placed` bit, and the gap byte — the 6-byte record of §3.1, plus the
  chunk key. And the same record set, per chunk, as the BASELINE a client receives when it subscribes to
  a chunk (D11).
- **FROM WHICH REALM TO WHICH.** From the moon's own shard, to the gateway, to the clients that hold the
  moon in their composed scene. **No realm learns another realm's cells.**
- **WHY THE RECEIVER CANNOT COMPUTE IT.** The seed decides the hill. It does not decide the tunnel a
  player dug last week, because a player is live state (SL10 clause 6).
- **THE SHAPE — two surfaces, and the minor bumps.**
  1. **Client-facing:** an appended `EventMsg` variant on the existing `ServerControlMsg::Event` arm
     (`crates/wire/src/channels.rs:195,303-320`). Its doc already rules that appended variants ride it —
     "one carrier, two consumers, built once" (`crates/wire/src/channels.rs:189-194`). One client
     protocol-minor bump.
  2. **Shard→gateway:** one appended `ShardToGateway` arm, mirroring `ShardToGateway::EntityRemoved`
     (`crates/wire/src/session_flow.rs:259`), which is already the shape the gateway fans out as
     `ServerControlMsg::Event`. One mesh protocol-minor bump.
- **WHAT DOING WITHOUT COSTS.** A player digs a scoop into a dune and nobody else ever sees it; and the
  client's derived shape disagrees with the shard's collider on every edited cell, which breaks SL10
  clause 5 outright.
- **DISPUTED (a detail inside the finding, not the finding).** The feasibility refuter asked for THREE
  surfaces, including an `InterShardFlow` arm. This domain needs no such arm: no sibling shard reads
  another realm's cells, and the reserved `BlockEdit` line (`crates/wire/src/intershard.rs:34`) stays
  reserved for whatever P6 consumer eventually needs a shard-to-shard edit. Two surfaces are asked. The
  same refuter cited `ShardToGateway::RealmSceneDelta` (`session_flow.rs:244`) as the existing path; that
  arm is a TOMBSTONE which "nothing produces" and which says "Do not revive"
  (`crates/wire/src/session_flow.rs:236-248`), so it can carry nothing.

### 9.2 ★ THE SEED / SUBSTANCE BOUNDARY — stated here, because this report owns the substance field

The ruling of 2026-08-27 binds the block system by name: *"a block's SUBSTANCE may not be a pure function
of (position, seed)"* (`owner_decisions_2026-08-27_seed_and_secrecy.md:59`), because a shipped generator
inverts to its seed. SL10 makes the danger sharper, not softer: clause 2 ships the generator crate to
EVERY client, so a seed-decided ore vein is a public map the moment the first client runs it.

SL10 clause 1 nevertheless names *"the seed-decided common materials"* as part of the static shape. The
two are reconciled by drawing the line, and the line is:

> **The seed decides only the COMMON STRUCTURAL SUBSTANCES — rock, dirt, sand, ice, water. Every VALUABLE
> deposit is live state, is never a function of `(seed, position)` at all, and arrives on the diff lane of
> §9.1.**

- **The gate, which could fail:** the generator crate's substance table contains no row the manifest marks
  valuable. A test asserts it, and slice 1 carries it.
- **The tension, named.** The seed ruling's own table says COMPOSITION — "what a place is MADE OF" — may
  NEVER be client-computed (`owner_decisions_2026-08-27_seed_and_secrecy.md:38-40`). SL10 clause 1 is
  later and explicitly permits the COMMON materials. The boundary above is what makes both true at once:
  what the client computes cannot pay, and what pays the client never computes.

*Example: two players land on the same moon and both see the same grey rock face; only the one who
surveyed it knows which seam holds the ore, because the moon's shard decided that seam and told nobody
else.*

### 9.3 The other laws

- **The hull-on-terrain contact** needs a child's collider shape at its parent. This is the SL6 ask the
  base already names (`decision_board.md:814-817`). It belongs to the landing/collision domain and is NOT
  made here. Doing without it: a hull cannot land on a planet's surface.
- **SL10 clause 4 versus the existing generator.** `crates/physics/src/celestial.rs:66-93,103,208-240`
  calls `sin`, `cos`, `atan2`, `powf`. Those are lawful there (the star field ships once). The terrain
  generator is a separate crate under §2.4's three-part fence. **One unpriced item stays open:** the
  spherical profile is a *"tangent-anchored spherical projection with re-anchoring"*
  (`crates/sim/src/capability.rs:39-40`), and turning a cell address into a direction on a moon is where
  `atan` and `tan` normally live — INSIDE the generator's path. The projection's exact operation list is
  UNRESOLVED. It is slice 0's business and M3 gates it.
- **"The client only renders" (SL10 clause 7).** The extractor runs on the client. It derives geometry
  from the static shape and from diffs the owning realm shipped. It derives no pose, no velocity, no
  entity state. **And it derives nothing at all for a realm the client is not drawing** (§5.6).
- **SL3 — a realm draws itself.** The planet realm's shard owns its cells and ships their diffs; the
  parent (the star system) ships the planet's placement and nothing else. Unchanged.
- **SL5 — one world.** The noise bench is a stand-in rig. The moment the generator crate exists, the bench
  moves INTO the crate, the gates run the crate's bench, and the standalone copy is deleted.
- **The FINAL-backend law.** ONE collider builder per lane (§3.2). No second implementation selected by
  `VoxelGeometry`.
- **HR3 / HR4.** Lane selection by the cell's form (data); the same three lanes and the same NON-EMPTY
  fixture on a Spherical and a Cartesian profile (§5.6, D13).
- **HR5.** The extractor, the quantisation and the pyramid's delta mean are Tier-A integer or
  IEEE-primitive code, coverable to 100 %; the exactness pins (digests) live in `tests/` like
  `celestial_ephemeris_pin.rs`.
- **The movement contract.** No velocity crosses upward and no parent sets a speed. The residency rules of
  §5.4 and §5.5 READ the body's swept line and the closing speed; they never state or clamp a speed, which
  is what the 2026-08-27 answer (M-D, rulings 1 and 2) requires.

---

## 10. Stale claims in the investigation base, with what supersedes each

| Claim | Where | Superseded by |
|---|---|---|
| "Generated terrain is `Cube`-only at P4 and P6, and at every rung of the ladder." | `block_system_design.md:7072-7096` | V2.1 (smooth terrain). This report §2.3. |
| "Iso-surface extraction — rejected, and it is a ONE-WAY DOOR." | `block_system_design.md:8998-9007` | V2.1 for terrain-form cells. The rejection stands for catalogue-form cells (V2.3). |
| "Sub-metre smooth extraction is the wrong target"; the look is "PHYSICAL, BLOCKY, OLD". | `stunning_look_plan.md:1135-1150,183` | V2.1: the terrain is smooth; the BUILT world stays blocky. The look statement needs a re-cut, which is the look domain's. |
| "SDF / dual contouring: block identity dissolves." | `block_system_design.md:5416-5420` | §3.1: the identity stays in the record beside the gap. |
| "The user's ask — dirt should not have perfectly lined edges — is answered by §5.6's render-side rim warp." | `block_system_design.md:7082-7084` | V2.1: it is answered by the smooth surface. |
| "Our surfaces are axis-aligned quads at every tier, so two rungs meet in a vertical STEP." | `decision_board.md:1511-1513`; `block_system_design.md:9013-9019` | §5.1–5.2: the surfaces are smooth; skirts stay, at a CONSTANT depth in cells; geomorphing is decided by M4. |
| "The wire already carries a `coarsen_level: u8` precision ladder at line 656." | `block_system_design_addendum_2.md` §A | `crates/wire/src/intershard.rs:1173-1186`: the field lives only in a tombstone payload that nothing produces. |
| Bushes and leaves are real cells with empty colliders (R13, R14). | `block_system_design.md` owner-rulings table | V2.2: a tree is ONE object; leaves are not cells. The pass-through class is empty today. |
| "A thing collides iff it occupies a CELL, or it is an ENTITY." | `decision_board.md:132-142` | §2.3 item 4: FOUR arms — a square cell, an entity, the extracted tier-0 terrain surface, and a seed-shaped surface object (V2.2's tree). |
| **"The base cell is never subdividable"; "below the catalogue's smallest extent there is no cell and therefore no collider, ever."** | `decision_board.md:132-142` | **V2.4 (blocks smaller than one metre, fitting inside the 1 m cell). §4.3 makes `SubGrid` a form; §3.3 Case 6 rules it keeps its gap byte. The sub-metre domain owns the wire and storage cost the base rejected it for.** |
| "The collider is the unwarped tier-0 cell … `render_hull ⊆ collider_hull`." | `block_system_design.md:7104-7110` | SL10 clause 5: the collider is the extracted surface, EQUAL to the drawn one — and equality needs §5.5's invariant, not a superset rule. |
| **"Collider residency is 64 m, derived as greater than any body's per-tick swept motion at 20 Hz plus reach."** | `block_system_design.md:6485-6492` | **`owner_decisions_2026-08-27_movement_answers.md` M-D rulings 1 and 3: the ceiling left the flight path and containment became swept, so no per-tick motion bound exists. §5.4 replaces the ball with the tick's swept segment.** |
| **"Everything inside one chunk-width of the camera is tier 0" as the client's rung-0 floor (62 m).** | `block_system_design.md:6408-6410` | **§5.5: the floor is the body's collider set, not a camera distance. 62 m is narrower than the base's own 64 m collider ball.** |
| **Strategy C's 0.885 ms is "the" per-chunk generation cost.** | `block_system_design.md:17414-17423` | **Still MEASURED, but it depends on sampling the cave field every 4th cell (`:17428-17439`), which under a smooth extractor becomes the cave wall a player sees. M1 re-measures at three steps; D12 decides; the step sits inside the digest door.** |
| "P4 commits to the binary-greedy-meshing adapter on 62³ chunks." | `block_system_design.md:5422-5423`; `roadmap.json` P4 | Already contradicted by the base's own O39(a) (`decision_board.md:1086`); now a three-lane mesher. |
| The catalogue's orientation is 6 bits with a mirror bit. | `block_system_design.md:5379-5382` | `block_system_design.md:710-720`: 5 bits, mirror as a shape row, and the freed sixth bit is already spent on `placed` (`:2046-2050`). §3.1 and §4.1 of this report both say 5. |
| "Walkability thresholds only ever judge player-built ramps." | `block_system_design.md:6719-6722` | §4.2: they judge every hillside now (D7). |
| The horizon fill of 1.19 s on one core. | `block_system_design.md:17444-17453` | Addendum 2's ladder (already flagged in the base); and the per-chunk cost must be re-measured for the gap path (M1, M2). |
| `parry Voxels` for the Cartesian cube lane, "the whole API", with a DIFFERENT builder on a spherical realm. | `block_system_design.md:6465-6472` | Two things: rapier/parry are not dependencies (`Cargo.toml:24-90`, D8 open); and §3.2 collapses the two builders into ONE, because a second implementation selected by `VoxelGeometry` breaks the FINAL-backend law. |
| **"`BlockEdit` is a reserved wire arm."** | `crates/wire/src/intershard.rs:34` reads as if it were | **It is a doc comment, not a variant, and `InterShardFlow` does not reach a client. §9.1 states the real ask.** |
| **"The noise bench inlines exactly the repo's `SplitMix64`."** | `scripts/noisebench/src/main.rs:22-35` says "inlined" | **`crates/core/src/rng.rs:23-28` performs a state step first; the bench's `mix64` is the finaliser only and its `hash3` adds its own per-axis multiplies. Two different functions. The generator's position hash must be named exactly, inside the digest door.** |

---

## 11. The slice this domain owes, in order

Each slice names its gate and its measurement. The formats domain and the sub-metre domain fix the
records these slices write; this list assumes their doors shut first.

0. **★ THE FRAME-SPACE SEAM.** `FrameSpace` / `SphericalSpace` / `CartesianSpace`, `reanchor()`,
   `AnchorGen`, and the address→direction map the generator needs on a moon. **This is terrain's first
   slice by the owner's word** (`docs/design/DEFERRED.md:309-311`, 2026-09-05 "yes to all"): it IS voxel
   geometry, so it lands with the first voxel. Nothing in the code turns a cell address into a radial
   direction today, so slice 1 cannot start without it. *Gate:* the still-owed D-38 reanchor-forcing HR4
   fixture (`DEFERRED.md:331-334`; the known limit is written into `crates/sim/src/capability.rs:13-18`).
   *Measurement:* the cell-address round trip stays EXACT across a re-anchor; and the projection's
   operation list is written down and passes §2.4's fence (§9.3).
1. **The generator crate, with the gap output.** One crate, the float newtype of §2.4, a NAMED integer
   position hash (§1, §7), the height stack with octave dropping, the carver, the strata, water as a
   material, and the COMMON-substances-only rule of §9.2. Output: an `i8` gap + a substance per cell, per
   `(seed, chunk key, tier)`. *Gate:* M1 throughput at three cave-lattice steps; M3 digests equal across
   two targets at two optimisation levels, at every legal tier; the no-valuable-substance test.
   *Measurement:* ms per chunk, and the digest diff.
2. **The smooth lane.** Surface nets over the gap with a one-cell apron; normals; `MeshPrim` output
   (unchanged type); skirts at the constant cell depth of §5.2. *Gate:* M2 throughput; a pinned quad list
   for a fixture chunk; **the NON-EMPTY HR4 fixture — the same gap pattern planted in index space on a
   moon AND in a station's growing bay, asserting the same quad list** (§5.6, D13). *Measurement:* ms,
   quads, bytes per chunk.
3. **The tier ladder on the client.** **M4 runs FIRST.** Then: tier per column from the angular rule;
   coarse-before-fine; the crossfade band; geomorphing if M4 says so; the rule that the client derives
   only for a realm in its composed scene (§5.6). *Gate:* the no-pop detector at walking and flight speed;
   M10's descent arrival rate against a named thread budget. *Measurement:* max px at the switch; chunks
   per second and the deepest queue.
4. **The terrain collider (after the library decision).** Tier-0 extraction over the body's SWEPT SEGMENT
   (§5.4); `Tier0Key` typing; the §5.5 residency invariant as a gate. *Gate:* M5 with a LANDING HULL;
   M6b's see-versus-stand gap at every rated approach speed; the collider input's digest equals the
   client's mesh digest for the same chunk. *Measurement:* ms and bytes per chunk; the swept chunk count
   per tick; the vertical gap at contact.
5. **Mining and placing on terrain.** The diff lane on the NEW surfaces of §9.1, with the per-chunk
   BASELINE on subscribe (D11); the gap under a placed square block (D3); the yield (D6); the pyramid's
   gap-DELTA mean (§5.3). *Gate:* M6 reflow ≤ 33 ms; M8 lane bytes; M11 standing bytes; the
   fly-away-and-back test, which now asserts **no RIM STEP** around a coarse-rung dimple, not only that
   the tunnel is visible; the crossing case (§3.3 Case 8). *Measurement:* ms, bytes per second, bytes at
   first sight, the rim step in cells.
6. **The square lanes on a planet.** The cube and shaped lanes over the same grid, through the ONE builder
   of §3.2; a foundation cut into a slope (Case 1); a heap against a wall (Case 2); a `SubGrid` cell on a
   slope with the required quad clip (Case 6). *Gate:* the run-anywhere fixture (place a wedge, break it,
   re-place it as a ramp, walk it) on both profiles with the terrain present; **M9's sweep across the
   foundation**. *Measurement:* the per-cell collider bytes on a planet (the base's 800 B per shaped cell
   is ESTIMATED); the sweep's vertical impulses.

*Example of the whole path:* a player lands a hull on a moon (slice 4 holds the ground under the hull's
whole swept line), walks down the ramp onto a smooth hillside (slices 1–3 drew it, with no pop as the hull
descended and no hitch as the chunks arrived), digs a scoop into the sand (slice 5 sends the diff on the
newly asked arm and reflows the slope on every client that holds the moon), and sets a steel foundation
into the slope beside the scoop (slice 6 cuts the hill cleanly at the block's faces, and M9 says the boots
do not catch on it).

---

## 12. Revision log

Two refuters reviewed the first draft. Their findings, and what this revision did with each.

### From `verdicts/terrain_law.md`

| # | Finding | What I did |
|---|---|---|
| 1 | BREAKS_LAW (SL6): "no new wire arm is asked for" is false | **FIXED.** §9.1 is a new section that makes the ask in SL6's shape. §0 item 7, §1 rows 8–10 and §10's last rows carry the correction. Disputed in part — see F1 below. |
| 2 | WRONG: the sign contradicts itself (`h − r` against "negative inside") | **FIXED.** The datum is `r − h` everywhere: §0 item 1, §2.3, §7 doors 1 and 3. §2.3 names the pin test that could fail. |
| 3 | WRONG: 1/127 is not an exact count of fine cells | **FIXED.** The quantum is 1/128 cell in §0, §2.3 and §7, with the fine-cell arithmetic (`2⁻⁷ m` = 8 fine cells at tier 0, `2^(L+3)` at tier L). |
| 4 | BREAKS_LAW (HR4): the Cartesian fixture is empty | **FIXED.** §5.6 rules that a Cartesian realm MAY hold terrain-form cells, cites the code that already permits it (`capability.rs:134,260-272`), makes the fixture assert the SAME quad list on both kinds, and adds D13 because it changes the game. |
| 5 | BREAKS_LAW (FINAL backend): two collider builders behind a config field | **FIXED.** §3.2 and §4.2 state ONE builder per lane through `cell_point`; §10 records the base's split as stale. |
| 6 | BREAKS_LAW (SL10 clause 5): nothing ties the client's tier to a body's contact | **FIXED.** §5.5 is a new section: the invariant, its lead (tied to `INTERP_BUFFER_MS`), and the landing gate. M6b is a new bench. |
| 7 | The restated collision rule has no arm for a tree | **FIXED.** §2.3 item 4 has four arms; the fourth is a seed-shaped surface object whose collider the shard derives from the same crate. §3.3 Case 7 states the two facts the terrain owes the tree domain. |
| 8 | BREAKS_LAW (the 2026-08-27 seed ruling): substance as a pure function of seed | **FIXED.** §9.2 draws the boundary (common structural substances only), gives the gate, and names the tension with the ruling's "COMPOSITION NEVER" table row. Slice 1 carries the test. |
| 9 | MISSING: the client may derive only while the realm is drawn | **FIXED.** §5.6 states the rule and cites the composed-scene arms. |
| 10 | UNMEASURED_AS_FACT: "two colliders overlap … harmless" | **FIXED.** §3.3 Case 1 marks it UNMEASURED and adds bench M9; the quad clip becomes the cure if M9 is red. |
| 11 | WRONG: two code citations (the `MeshPrim` widening, `lib.rs:820-821`) | **FIXED.** §1 gives `crates/client/src/realm_scene.rs:770-773` and `crates/client-render/src/lib.rs:1828-1841`, and says plainly that nothing is widened. §2.3, §3.2 and slice 2 no longer say "widened". |
| 12 | UNMEASURED_AS_FACT (SL8): a known-visible pop is deferred | **FIXED.** §5.2 and slice 3 put M4 BEFORE the ladder is built; D9 makes geomorphing part of slice 3 if M4 says so. |
| 13 | MISSING: the standing pyramid bytes are never measured | **FIXED.** M11 is a new bench (N edited cells across four orders of magnitude; bytes held per rung and bytes at first sight). |
| 14 | MISSING (low): the "base cell is never subdividable" ruling is not in the stale table | **FIXED.** §10 has the row, with V2.4 as the superseder and the sub-metre domain as the owner. |

### From `verdicts/terrain_feasibility.md`

| # | Finding | What I did |
|---|---|---|
| F1 | BREAKS_LAW: three new wire surfaces, not zero | **FIXED, and DISPUTED in part.** §9.1 makes the ask. **DISPUTED:** the ask is TWO surfaces. This domain needs no `InterShardFlow` arm, because no sibling shard reads another realm's cells; `crates/wire/src/intershard.rs:34` stays reserved for a later P6 consumer. **Also corrected:** the refuter cited `ShardToGateway::RealmSceneDelta` (`session_flow.rs:244`) as the existing path; that arm is a TOMBSTONE — "nothing produces it … Do not revive" (`crates/wire/src/session_flow.rs:236-248`). The live shard→gateway scene lanes are `WindowBody` and `WindowMembership` (`session_flow.rs:315,337`), and neither can carry a cell either. |
| F2 | BREAKS_LAW: the 64 m residency rests on a deleted bound | **FIXED.** §5.4 replaces the ball with the tick's swept segment, cites the ruling (M-D 1 and 3), and M5 now measures a landing hull. §10 records the base's 64 m as stale. |
| F3 | MISSING: SL10 clause 5 has no residency invariant | **FIXED.** §5.5, plus M6b. §5.5 also says plainly that this report does NOT inherit the base's 62 m floor, which is the arithmetic the refuter used. |
| F4 | MISSING: the FrameSpace seam is terrain's first slice and is not in the list | **FIXED.** §11 slice 0, with the D-38 reanchor-forcing fixture as its gate and the exact-round-trip measurement. §1 row 2 carries the ruling's citation. |
| F5 | WRONG: the apron depth mixes metres and cells | **FIXED.** §5.2 derives `8 · k_rough · skirt_safety` CELLS of the chunk's own rung — 4 cells at `k_rough = 0.5`, constant at every rung — and says where the old form went wrong. |
| F6 | WRONG: the byte is a radial gap, not a signed distance | **FIXED, and DISPUTED in part.** The byte is named a RADIAL GAP throughout, and §2.3 states both honest limits (the cosine on a slope, the `max` near a tunnel seam). **DISPUTED:** I do NOT divide by the local gradient magnitude. That needs a square root over neighbouring samples, so the stored byte would depend on an apron's evaluation order, which puts MORE inside the pinned digest, not less. The refuter's own first option — name the byte what it is — is the cheaper and safer one. |
| F7 | WRONG: the quantisation contradicts the lattice claim | **FIXED** with law finding 3. |
| F8 | MISSING: the coarse summary and the coarse seed base are different quantities | **FIXED.** §5.3 stores DELTAS, states why an absolute makes a four-metre rim step, corrects the accumulator to `i16`, adds the rung rescale (`(sum + 8) >> 4`), and makes slice 5's gate measure the RIM STEP. |
| F9 | MISSING: the diff lane has no baseline | **FIXED.** §9.1 includes the per-chunk baseline in the ask; D11 is the open decision; §3.3 Case 8 covers an edit during a crossing; M11 budgets the bytes. |
| F10 | WRONG: the measured 0.885 ms does not transfer | **FIXED.** §2.2's cost row, M1 (three lattice steps, with pictures), D12 and §10's row all carry it. The step is named inside the digest door in §7. |
| F11 | UNMEASURED_AS_FACT: "by construction" | **FIXED.** §2.4 is a new section with the three-part fence (a newtype that withholds `mul_add` by name, a pinned crate profile with an allowed-flag list, and M3 as the guarantee). §0 item 2 now says UNMEASURED until M3 runs. |
| F12 | MISSING: the spherical address→direction step is unpriced | **FIXED.** §9.3 states it as an open item and cites `crates/sim/src/capability.rs:39-40`; §11 slice 0 owns it, with M3 as the gate. |
| F13 | MISSING: the frame budget for a landing at flight speed | **FIXED.** M10 is a new bench (a scripted descent; chunk requests per second; the deepest queue; a named thread budget). Slice 3 is gated on it; §5.2's tick-hitch row names it. |
| F14 | WRONG: three code claims | **FIXED.** The citation (`lib.rs:1828-1841`); the needless widening (deleted); and the hash — §1 and §10 now say the bench's `mix64` is the FINALISER only, that `hash3` adds its own multiplies, and that the generator's POSITION HASH must be named exactly inside the digest door (§7). |
| F15 | WRONG: the record's bit arithmetic contradicts itself | **FIXED.** §3.1 shows the sum, `18 + 16 + 5 + 1 + 8 = 48 bits = 6 bytes`; orient is 5 bits in §3.1 AND §4.1; the freed sixth bit is named as already spent on `placed` (`block_system_design.md:2046-2050`); §7 carries the 5→6 byte door. |
| F16 | BREAKS_LAW: the smooth lane's HR4 gate is empty | **FIXED** with law finding 4, by the stronger of the two offered cures: the Cartesian realm holds real terrain cells, so the fixture asserts the SAME output on both kinds instead of an absence. |

### What both refuters agreed still stands, and this revision keeps

- The representation choice: a substance plus one signed sample per cell, meshed by naive surface nets.
- Surface nets before dual contouring, now with the §2.4 fence written properly.
- No library adopted: rapier3d and parry3d stay absent, and D8 stays the owner's call.
- The dead `noise` dependency (`Cargo.toml:78`).
- SL1, SL2, SL3, SL4, SL7, SL9, HR1 and the movement contract are unbroken by this design.
- The earlier stale-claim rows; this revision adds seven more.
