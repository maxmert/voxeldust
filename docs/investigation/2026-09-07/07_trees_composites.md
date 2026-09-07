# 07 — Trees, large plants and composites: ONE object on the surface

**Domain:** V2.2 (a tree is one object with seed and height parameters inside; the client draws the
full tree; the server computes collision on a shape it knows), with R4 (composites, OPEN), R12–R14
(grass and bushes are decoration or pass-through; leaves pass through) and R6–R8 (provenance,
regrowth, planted growth stage).
**Date:** 2026-09-07. **Revision 2** (after the law refutation and the feasibility refutation; see
§14, the revision log). **Status:** investigation output. Not binding until the owner rules.
**Law read first:** `CLAUDE.md` (HR1–HR6, SL1–SL10), `owner_decisions_2026-09-07_voxels.md` (V1 =
SL10, V2.1–V2.9, V3), the rulings of 2026-09-05, 09-02, 09-01, 08-27 (three files), 08-26, 08-24,
`DEFERRED.md`, and the seamless law SL8 (eleven seam kinds).
**Base read second, not trusted:** `collidable_decoration.md`, `block_provenance_collapse.md`,
`block_system_design.md` §4.6, §6 and the R4 note, `decision_board.md` (§1, O24–O26, O57, §3, §4).

Every number below is marked MEASURED (with how) or ESTIMATED. Every claim about the code cites a
file and a line. The code is the only truth for what exists today.

---

## 0. What exists today (the code, MEASURED by grep on 2026-09-07)

- There is no voxel, chunk, block, tree or decoration module anywhere. `ls crates/core/src` shows no
  `block/`, `grid/` or `voxel/` (`crates/core/src/` holds **26 files**, `built.rs` … `worldgen.rs`,
  MEASURED by `ls crates/core/src | wc -l` on 2026-09-07; none of them is a voxel module, and I read
  every name). The generator makes bodies, orbits, stars and regions only
  (`crates/physics/src/worldgen.rs:1-70`; the module list at lines 44-56 has no terrain stage).
- The generator is ONE crate and its draw stream is append-only by owner law
  (`crates/physics/src/worldgen.rs:17-36`, THE DISCOVERY-PERMANENCE LAW). Every seed-derived thing in
  this report must obey that law: a new draw goes AFTER every existing draw.
- The RNG is `SplitMix64` with `child_seed(parent, salt, index)`
  (`crates/core/src/rng.rs:12-27`, `:66-72`). **The integer core is integer-only, and the module is
  not.** `SplitMix64::next_u64` (`:22`) and `child_seed` (`:70`) use integer operations alone.
  `SplitMix64::next_f64` (`:31`) returns `f64` and `SplitMix64::chance` (`:39`) takes `f64`; the
  generator draws floats from them today (`crates/physics/src/worldgen/generate.rs:159-163`,
  `:460-466`, `:765-767`). The integer core is the hash SL10 V1.4 asks for. `expand` (§1.3) must use
  the integer core alone, and a lint must refuse `next_f64` and `chance` inside it.
- **The one generator crate calls libm at EIGHT production sites today** (MEASURED by
  `grep -n "powf|\.exp()|\.ln()|\.sqrt()|\.cos()|\.sin()|\.tan()" crates/physics/src/worldgen/` on
  2026-09-07; six more sites are in `tests.rs` and do not ship):
  `generate.rs:571` (`.ln()` twice), `:598` (`.ln()`), `:633` (`.ln()`), `:638` (`.sqrt()`),
  `:639` (`.cos()` and `.sin()`), `:807` (`.tan()`), `:1880` (`.powf()`), and
  `scale.rs:572` (`.powf()`). SL10 V1.4 allows `sqrt` and forbids `ln`, `cos`, `sin`, `tan` and
  `powf` in a crate a client links. **A tree lives in that crate. Retiring these seven forbidden
  sites is owed work for this domain, not somebody else's problem** (§9 item 11).
- Neither `rapier3d` nor `parry3d` is a dependency: `grep -c 'name = "rapier3d"|name = "parry3d"'
  Cargo.lock` returns **0** (MEASURED 2026-09-07). `noise = "=0.9.0"` is declared at the workspace
  level (`Cargo.toml:78`) but no crate uses it: `grep -rn "noise::" crates` returns nothing and
  `Cargo.lock` has no `noise` package. The block design's USER DECISION 4-B (adopt rapier) is still
  open. **So the capsule compound of §1.3 has no implementation anywhere** (§11 item 9).
- The entity registry has seven kinds: `Player, Ship, NamedConstruction, Debris, DroppedBlock,
  Projectile, Rocket` (`crates/core/src/entity_kind.rs:33-41`). There is no `FallingGroup`. The
  `RealmAnchored` continuity model is **not built**: `crates/core/src/entity_kind.rs:23` says
  "Guided/RealmAnchored/InBand land at P6/P10". Only two behaviour triples exist today.
- The bulk lane has three kinds, `ChunkSnapshot, ChunkDelta, Catalog`, and one opaque `Blob`
  (`crates/wire/src/channels.rs:283-296`). It is RELIABLE and per-subscription. The world state rides
  a separate UNRELIABLE 20 Hz datagram (`crates/wire/src/channels.rs:450`, `SnapshotDatagram`).
  **The two lanes have no shared order.** `BlockEdit` is a reserved arm name, not a variant
  (`crates/wire/src/intershard.rs:34-35`).
- The client holds every entity 100–150 ms in an interpolation buffer before it draws it
  (`crates/client/src/tuning.rs:8-18`, `ClientInterpTuning::interp_buffer_ms`;
  `crates/client/src/view.rs:49-55`; `CLAUDE.md:261`). There is no client-side prediction.
- The render seam today is `MeshPrim { vertices, color_rgba, transform }` with `Vertex { pos, normal }`
  and `PrimTransform { translation, scale, rotation }` (`crates/client/src/realm_scene.rs:770-796`).
  The renderer turns it into a mesh with no shape branch (`crates/client-render/src/lib.rs:1824-1830`,
  "PURE glue: no shape branch, no logic — exactly the H4 seam"). A realm's look is one `Boundary`
  (`Shell`, `Aabb`, `Obb`; `crates/core/src/geometry.rs:338-348`) shipped as `TAG_LOOK`
  (`crates/core/src/look.rs:31`). Nothing draws a tree today.
- The reach helper the nesting fence is built from is `Boundary::circumscribed_extent`
  (`crates/core/src/geometry.rs:350-361`), paired with `inscribed_extent`; the boot-time fence uses
  the pair (`crates/core/src/geometry.rs:4327-4331` names it: "the pair the boot-time nesting fence is
  built from").
- The DRAWABLE FLOOR exists in the code: `drawable_theta_min_rad()` is `2·tan(fov/2)/rows` at the
  reference view of 45° over 720 rows (`crates/core/src/geometry.rs:1156-1173`). MEASURED by
  arithmetic on those constants: **0.0011506 rad**, and the reach factor `cot(θ/2)` is **1738**. A
  one-metre extent is one pixel at 1.74 km, which the code's own doc states.
- The finest coordinate rung is `2⁻¹⁰ m` (`crates/core/src/pose.rs:474-478`, `Tier::Fine`).
- `FrameSpace` and `reanchor()` do not exist yet; they are terrain's first slice
  (`docs/design/DEFERRED.md:299-353`, D-38).

So this report designs on paper against the laws. Nothing here describes running code.

---

## 1. The tree object

### 1.1 The ruling this serves

V2.2, owner's words: *"they should be placed as one object/block on the surface, but simply rendered
on the client as a tree (with different height and structure depends on the seed and height params
inside this block), but then the collisions are calculated on the server, so server somehow should
know the shape."*

SL10 V1.1 says the geometry of a seed-placed feature is static shape: *"a tree's trunk and canopy from
its seed parameters"*. SL10 V1.5 says the server computes collision on the SAME shape.

### 1.2 The record: one surface cell holds one object, and the growing part sits beside it

A tree is ONE record on ONE cell of the block field, plus — for a tree that is still growing — ONE row
in the per-cell side table. The cell is the air cell directly above the ground cell the tree stands on
(the base doc reached this for a stone: `collidable_decoration.md` §4.1, lines 235-275).

| Field | Where it lives | Seed tree (a forest the seed placed) | Planted tree (a player planted it) |
|---|---|---|---|
| kind | the record's `kind` field (16 bits) | derived from the seed | stored |
| variant / style | the record's `variant` field (6 bits) | derived | stored (V2.6) |
| instance seed | nowhere | derived: `hash(realm seed, cell address, kind)` | derived the SAME way |
| growth stage | the SIDE TABLE row | not stored: the kind's baseline stage, a constant | stored |
| plant tick | the SIDE TABLE row | not stored | stored |
| yaw | nowhere | derived from the instance seed | derived from the instance seed |
| provenance | the record's `provenance` field (2 bits) | `Feature` (R6) | `Feature` (R6-10) |

**CORRECTED (law finding F1).** The first revision said the tree's growth stage rides "the one state
byte" of the saved block record, and cited the investigation base
(`block_provenance_collapse.md` §3.1, lines 137-147). **Domain 04 does not recommend that record.**
The record it recommends has no state byte at all: `cell 18 | kind 16 | orient 6 | provenance 2 |
variant 6 | sub_scale 2 | sub_addr 9 | reserved 5`
(`04_block_record_registry.md:112-136`, and `:247-256` records where the fifteen spare bits went; the
five that stay reserved must decode as zero). The eight-bit state field exists only in the
investigation base, which is not binding.

So the growth stage has no home in the eight-byte record, and this report no longer claims one. It
takes the side-table slot it already asked domain 04 to reserve (§2.5) and puts TWO fields there: the
growth stage and the plant tick. Both are live state, both are already excluded from the static shape
by SL10 V1.1, and both go away when the tree is mature.

**A seed tree still costs zero bytes.** A felled tree still costs exactly the eight-byte record
(§2.2), because a felled cell is `Air` and has no side row. Only a GROWING tree carries a side row.
ESTIMATED side-row cost: stage 1 B + plant tick 8 B + the side table's own cell key = **on the order
of 16 B while the tree grows**, and 0 B once it matures and prunes (§2.4).

**Example.** The seed plants an oak at cell `(face 3, i 1200, j 880, k 161 671)` of a moon. No byte is
stored. A player plants a pine two cells away. One eight-byte record is stored — the pine kind,
provenance `Feature` — plus one side row holding stage 0 and the tick she planted it. Both trees get
their exact shape from the same function.

### 1.3 The expansion: one function, two outputs, one skeleton between them

The generator crate carries one function per kind:

```
expand(kind, instance_seed, growth_stage, size_class) -> Skeleton
```

A `Skeleton` is integer data:

- **structural segments**: a list of `(start, end, radius)` in the anchor cell's own frame, on the
  **1/1024 m integer lattice** — the finest rung of the coordinate ruler
  (`crates/core/src/pose.rs:474-478`). The trunk, its taper, the first-order limbs, and any limb
  thicker than the kind's `min_collide_radius`.
- **thin segments**: the same shape for twigs below that radius. Drawn, never collided.
- **canopy volumes**: a list of `(centre, half-extents)` ellipsoids. Drawn as foliage. Pass-through
  per R14 by default (see §11 item 6 for a per-kind flag).
- **yield rows**: units of wood per structural segment and units of leaf per canopy volume, so a cut
  drops exactly what the shape held (conservation, §3).

**CORRECTED (law finding F6). One unit, everywhere: 1/1024 m.** The first revision said "a 1/1024 m
integer lattice" in one place and "integer millimetres" in another. A millimetre and 1/1024 m differ,
and the difference reached 42 mm on a limb 1.8 m up a birch — enough to put the moon's capsule and the
client's tube in different places. Every length, every radius and every offset in a skeleton is an
integer count of 1/1024 m. There are no millimetres anywhere in this design.

Two consumers read one skeleton:

- **The server** builds the COLLISION SHAPE: one capsule per structural segment, as a compound. It
  never builds a mesh. The character controller, the sweep test and a hull's landing gear collide with
  the capsules. The canopy volumes are not in the compound.
- **The client** builds the MESH: a tube around every segment (structural and thin), foliage inside
  every canopy volume, at the detail the rung asks for. The tube radius is at most the capsule radius,
  so the drawn trunk lies inside the collider. This is the divergence contract of
  `collidable_decoration.md` §10 (lines 1087-1136): a phantom hit is allowed, a phantom miss is not.

**NEW (law finding F4). A phantom hit is allowed, and it is BOUNDED.** SL8 says a tolerance is
physical. "Allowed" without a number lets a player stop a metre short of the bark, in open air, and
that is the jump seam. **The rule: the collider surface lies within `PHANTOM_HIT_MAX_M` of the drawn
surface everywhere a player, a suit or a landing gear can touch it.** Proposed value **0.10 m**
(ESTIMATED: a player notices a stop that is a hand's width from the bark, and does not notice a stop
that is a finger's width). Measurement 3 asserts the DISTANCE, not membership (§9). An art-pack
archetype table that cannot meet the bound is refused (§5.1 route (b)).

Both hosts compile the same crate (SL10 V1.2). The skeleton is the thing the gate compares (§1.5).

**Example.** A player walks into a birch. The moon's shard tests the walk against seven capsules (a
three-segment trunk, four limbs). The player's client drew the same seven segments as tubes plus forty
twigs and three canopy ellipsoids. The player stops within ten centimetres of the drawn bark, and
walks through the leaves.

### 1.4 Determinism inside `expand` (SL10 V1.4)

- Every random draw is `SplitMix64::next_u64` from `hash(realm seed, cell, kind)`
  (`crates/core/src/rng.rs:22`, `:70`). `next_f64` (`:31`) and `chance` (`:39`) are BANNED inside
  `expand`, because both produce or take an `f64`, and a lint enforces the ban.
- Branch angles come from an integer sine table in the crate (256 entries, 1/65536 units). No `sin`,
  `cos`, `exp`, `tan`, `ln` or `powf` from libm. V1.4 forbids them, and the crate breaks that rule at
  seven shipped sites today (§0, §9 item 11).
- Lengths and radii are integer counts of 1/1024 m, scaled by the growth stage with integer multiply
  and shift.
- The evaluation order is fixed: trunk, then limbs by index, then twigs by index.
- The skeleton at growth stage `s` is a prefix and a scale of the skeleton at stage 255: the same
  segments in the same order, shorter and thinner. A stage step changes every point by a bounded
  distance (a gate in §9). This is what makes growth pop-free (SL8).

### 1.5 No drift is a measurement — of the SHAPE and of the PLACEMENT

**CORRECTED (feasibility finding R2).** The first revision gated the skeleton and nothing else. The
question that decides whether a collider exists at all is which CELLS hold trees, and stage 8 answers
it inside the same crate that breaks V1.4 today. A gate on the shape alone lets a client relocate one
draw by one cell and leaves the player walking into an invisible trunk.

The SL10 gate for trees has TWO arms:

1. **The shape arm.** For every kind, a fixed set of `(instance seed, stage, size)` inputs produces a
   skeleton; the gate compares the skeleton bytes from the server build and the client build on
   x86-64 and aarch64. One differing byte is red.
2. **The placement arm.** For a fixed list of chunks, the SET of `(cell, kind, size class)` that a
   host derives from stage 8 is compared the same way, on the same four builds. One differing tuple is
   red. This arm requires stage 8's draws, its support test and its bound test (§2.1) to be
   integer-only, and it is the arm that refuses the seven libm sites.

The mesh is NOT in either arm: two clients may tessellate differently (a Bevy client and an Unreal
client, V2.8). Collision reads the skeleton, never the mesh, so mesh differences cannot move a player.

The world-generation tag that refuses a client whose generator disagrees (SL10 V1.3) covers the tree
kinds, the skeleton format and stage 8's draw position, because they live in the same crate.

### 1.6 Cost (ESTIMATED; every number is owed a measurement, §9)

**CORRECTED (law finding F5, feasibility finding R4).** The first revision compared a 64 m disc with
the base's 64 m physics CLUSTER, which is 2.7 times the area and 2.7 times the trees, and it stated
two capsule sizes that differ by 3.6 times. Both are fixed. **One capsule size: 28 bytes**, which is
two endpoints of three `i32` on the 1/1024 m lattice plus one `i32` radius (ESTIMATED by arithmetic on
the field widths, not measured; the RUNTIME cost of a broad phase over those capsules is separate and
UNMEASURED, §9 item 5). The comparison below is normalised PER TREE and PER SQUARE METRE, so the two
sides describe the same thing.

| Quantity | Estimate | Basis |
|---|---|---|
| structural segments per mature tree | 5–16 | trunk taper 1–4, limbs 4–12 |
| thin segments per mature tree | 30–120 | drawn only |
| **stored collision bytes per mature tree** | **448 B** | 16 capsules × 28 B |
| trees per 64 m disc at one per 100 m² | 129 | π·64²/100 = 128.7 (the base's density) |
| stored collision bytes per 64 m disc (12 868 m²) | 57.8 KB | 129 × 448 B |
| stored collision bytes per square metre | 4.49 B/m² | 57.8 KB ÷ 12 868 m² |
| **the base's number, normalised per tree** | **19.9–65.0 KB** | 6.9–22.5 MB ÷ 346 trees (`collidable_decoration.md` §9.4, lines 1004-1025, a cluster of nine surface chunk columns, 34 596 m²) |
| **the base's number, normalised per m²** | **199–650 B/m²** | 6.9–22.5 MB ÷ 34 596 m² |
| **the gain** | **44× to 145×** | either normalisation gives the same pair |
| `expand` per tree, mature | ~2–10 µs | a few hundred integer draws and multiplies; UNMEASURED |
| a 800 m object-horizon disc of ~20 100 seed trees on the client | ~40–200 ms once, off the main thread | 20 100 × the line above; UNMEASURED |

The collider budget moves from "the largest line in the cluster" to a rounding error. The reason is
structural, not a tuning: a capsule holds a 10 m limb in 28 bytes, where the cell model held it in ten
eight-byte cells and a greedy box per cell. **The 35× to 112× the first revision claimed came from
comparing two different areas; the true gain is 44× to 145×, which is larger, and the conclusion is
unchanged.**

### 1.7 What this changes for the block field and for placement

- A tree occupies ONE cell. Its shape reaches beyond that cell. **Space taking is by shape, not by
  cells:** block placement validation tests the new cell's cube against the object collision shapes of
  the chunk (a few capsules, cheap). A player cannot place a block through a trunk. A player CAN build
  inside a canopy, because a canopy is pass-through. That is the treehouse, and it needs no rule.
- A tree's shape may cross a chunk edge and a cube-face seam. That is lawful for a shape (it was a
  problem only for a cell set, `block_provenance_collapse.md` §4.4 rejection 2). The realm's shard
  keeps object shapes in a per-chunk list keyed by the anchor cell and tests neighbours' lists too.
- A tree's shape may NOT reach outside its realm's bound. The bound test uses
  `Boundary::circumscribed_extent` (`crates/core/src/geometry.rs:350-361`), the reach helper the
  boot-time nesting fence is built from (`crates/core/src/geometry.rs:4327-4331`).
  **CORRECTED (law finding F13):** the first revision cited lines 350-368 as the fence itself; those
  lines hold the helper, and the fence is assembled from the helper pair elsewhere.
  **CORRECTED (feasibility finding R11): the bound test runs on the SEED PATH too, not only on the
  player's placement path.** Stage 8 tests every drawn object's skeleton extent against the realm's
  bound and drops a draw that pokes out, deterministically, on both hosts (the realm's bound is realm
  data both hosts hold, so the drop is identical). Without that, the seed draws a 20 m spruce on a
  station's garden deck, the spruce grows through the station's hull, and the station's parent sees a
  child that pokes out of its bound.
- The anchor cell's "up" is the face the object stands on, through `gravity_dir` and `FrameSpace`
  (the base's `up_face` rule, `block_system_design.md` §6.3). On a planet that is radial; in a hull it
  is the deck; in zero g it is the face the player placed it on. A potted tree in a hull is the same
  record, the same expansion and the same capsules. HR4 holds with no branch on shard kind.

---

## 2. Seed forests and planted trees

### 2.1 A seed forest is static shape: zero bytes

The generator's stage 8 (the provenance base named it, `block_provenance_collapse.md` §4.4, lines
431-490) runs on BOTH hosts as part of the one crate:

1. Per feature region (one 62-cell tangential cell of the direction sphere), draw `0..n` instances
   from `SplitMix64(child_seed(realm seed, FEATURE_SALT, region))`: an anchor cell, a kind, a size
   class. Integer draws only (§1.4).
2. Test the anchor's support cell in the post-carver, post-water field. Air or water below: relocate
   the draw deterministically. A tree is never planted over a cave mouth.
3. Test the drawn skeleton's extent against the realm's bound (§1.7). It pokes out: drop the draw
   deterministically.
4. Emit the record `(cell, kind, provenance Feature, variant)` with the kind's BASELINE STAGE — a
   constant — into the chunk's DERIVED object list. Nothing is written to any store.

**CORRECTED (feasibility finding R12).** The first revision's §1.2 said a seed tree's stage is
*"derived (the seed says 'mature', or the kind's age curve)"*. An age curve is a function of time, and
SL10 V1.1 says the static shape is never a function of time, of a tick or of any live state. **A seed
tree's stage is the kind's baseline stage, a constant in the registry.** The age-curve wording is
deleted. A tree whose stage moves is a tree with a side row (§1.2), and that side row is state that
crosses as a diff.

The draws are appended AFTER every existing draw of the region's stream, by THE DISCOVERY-PERMANENCE
LAW (`crates/physics/src/worldgen.rs:17-36`). A tree added by a content patch appears after the trees
already there; no tree that a player has seen moves.

**NEW (law finding F11). The seed ruling fences the DROP, not the draw.** The 2026-08-27 ruling says a
block's substance may not be a pure function of `(position, seed)`
(`owner_decisions_2026-08-27_seed_and_secrecy.md:59`), because a fixed seed plus a position is a
public treasure map, and it gives the test: *if this were printed on a public wiki tomorrow, would the
game still work?* Stage 8 draws the kind purely from `(seed, region)`, and §3.4 turns kinds into drops.
So the fence is: **a seed-drawn object kind may drop COMMON substance only.** Wood, leaf, fibre and
stone are common and may be seed-derived. Anything that pays — an ore, a crystal, a deposit — is live
state on the object's side row, it is decided by the world and not by the seed, and it arrives as a
diff. A crystal spire on a themed planet (V2.7) may stand exactly where the seed says; what it HOLDS is
not in the seed. Domain 03 owns the same fence for terrain; this is its statement for objects.

**Example.** A pilot lands on a moon nobody has visited. The moon's shard and the pilot's client both
evaluate stage 8 for the chunks in view and get the same 214 trees at the same cells — a claim the
placement arm of the gate (§1.5) MEASURES rather than asserts. Zero bytes crossed for the forest. The
shard holds 214 capsule compounds; the client holds 214 meshes.

### 2.2 Cutting a seed tree is a diff

A cut writes the anchor cell as `Air`: one eight-byte removal record in the chunk's authored set
(`block_system_design.md` §3.9.3, lines 4476-4530: a `ChunkRecord` is always the set of authored
cells). The client applies the diff over its derived forest: the generator says "oak here", the diff
says "air here", and the diff wins. **A felled tree costs 8 bytes of permanent record, not 1 845.** A
felled cell is `Air`, so it has no side row and the side table costs nothing here.

The base's storage alarm — *"a quarter of a thousandth of a planet, clear-cut, fills the planet's
delta budget"* (`block_provenance_collapse.md` §5.4, lines 802-818) — is superseded. Its arithmetic
re-run with 8 B per tree: a clear-cut 786 m disc is 19 406 × 8 B = **155 KB** (ESTIMATED from the
base's own count), against the base's 35.8 MB. That is 230× smaller (35.8 MB ÷ 155 KB = 231). R7
regrowth stays wanted as a world feature; it is no longer a storage requirement.

### 2.3 Growth is a stored plant tick and a fixed schedule, not a per-tick walk (R8, revised)

**CORRECTED (law finding F7, feasibility findings R6 and R7).** The first revision said three things
that cannot all hold: growth is memoryless; the shard draws elapsed stages "from a binomial over the
chunk's last-ticked stamp"; and "no TIMESTAMP is stored". A per-chunk last-ticked stamp IS a stored
timestamp, it was costed nowhere, the binomial had no named integer sampler for a count in the
billions, and the slow tick that walks every diverged cell had no budget and no measurement.

**The recommended mechanism, which stores less and walks nothing:**

- A planted tree's side row holds the PLANT TICK and, for a tree whose growth a player has altered
  (fertiliser, damage), the current stage. Nothing else.
- The stage is a closed-form function the owning shard evaluates on demand:
  `stage = kind.schedule(universe_tick − plant_tick, instance_seed)`, where `schedule` is an integer
  step table per kind and the instance seed supplies a bounded per-tree jitter, so a stand of pines
  does not step in unison. No loop over elapsed ticks. No binomial. No sampler to name.
- **The shard walks nothing on a slow tick.** It evaluates a tree's stage when a subscriber asks for
  that chunk, and when a player acts on the tree. A moon with a million growing trees and no observers
  costs zero. That is the SL9 shape: the cost follows observers and edits, never the size of the realm.
- **The client never advances a stage itself** (SL10 V1.7: growth is state, never derived). The shard
  ships the stage in the chunk's object diff.

**This bounds the growth lane (feasibility finding R7).** The first revision emitted a diff on every
stage change to every subscribed client, which for one recovering 786 m disc is
19 406 × 256 = **4.97 million messages** (ESTIMATED by arithmetic on the report's own counts). Under
the schedule the rules are:

1. A stage change emits a diff ONLY to a subscriber that holds that chunk at a rung where the step is
   drawn. A chunk nobody holds emits nothing at all.
2. A subscriber that arrives receives the CURRENT stage once, inside the chunk's object diff. A whole
   dormant recovery collapses to one row.
3. A step that moves no drawn point by more than the rung's own pixel emits nothing at that rung.

ESTIMATED under those rules for the same disc: one row per tree per visit, so **19 406 rows**, not
4.97 million. Owed a measurement (§9 item 12).

**Marked as a change of intent.** R8 in the investigation base says growth is MEMORYLESS. This report
recommends a DETERMINISTIC schedule instead, because memorylessness forced either a per-tick walk or a
binomial nobody could sample deterministically. The owner should rule (§11 item 10).

### 2.4 Regrowth toward baseline, and what "permanent" really means (R7, R8)

A cut seed tree leaves `Air` where the baseline says `oak, baseline stage`. Natural recovery is the
shard reading the baseline (it is the generator, which the shard holds) and stepping the cell toward
it, on the same on-demand path as §2.3:

1. `Air` → `(oak, provenance Feature)` with a side row holding a plant tick, after the kind's recovery
   delay, as one edit.
2. The stage follows the schedule of §2.3.
3. When the record equals the baseline exactly — kind, variant, provenance — AND the schedule says the
   stage equals the baseline stage, the prune-on-equality rule deletes the record and the side row
   (`block_provenance_collapse.md` §3.4 rule 2, lines 246-258). The bytes come back.

What "stores nothing" means exactly: no TARGET is stored (the generator is the target). **A plant tick
IS stored, in the side row, and this report costs it** (ESTIMATED 16 B while the tree grows, 0 B after
it prunes). The first revision's claim that regrowth stores no timestamp was false.

**CORRECTED (law finding F8). A planted tree is NOT permanent, and this report no longer says it is.**
The first revision argued that a planted tree never equals the baseline, so its record never prunes.
That argument fails on its own rules: §11 item 1 recommends deriving the instance seed from the cell
address, and a planted tree carries provenance `Feature`, the same as a seed tree. So a logger who
fells the seed oak and plants an oak back in the same hole grows a tree that is byte-identical to the
baseline, and the prune rule correctly deletes its record.

**That is the right behaviour, and it is now stated as such.** A replanted tree that matches the
baseline in every respect IS the baseline tree; nothing in the world changes when its record goes, and
the bytes come back. The claim that is retired is "a planted tree is permanent". What survives is the
narrower and true clause: **a `Placed` cell can never equal a `Feature` baseline**, so a player's
BUILT things never prune. A player who wants a permanent tree that the world will not reclaim plants a
kind or a variant the baseline does not hold at that cell, and then it never matches.

**Example.** A logger clears twenty oaks near a landing pad. Twenty removal records exist. Far from
the pad, nobody returns; over weeks the moon's shard regrows the oaks. Each oak's record and side row
delete themselves when the oak matches the seed's oak again. Near the pad, players keep cutting, so
the land stays clear (R7's emergent geography).

### 2.5 The object horizon: what the record (04), the pyramid (03) and the diff lane (06) must hold

**CORRECTED (feasibility finding R1). The first revision's rule "seed objects are derived at every
rung" and "seed trees are derived for the whole visible disc" is IMPOSSIBLE and is withdrawn.**
Neither rule named a distance at which a tree stops being an object, so the work grew with the area of
the planet a pilot can see. ESTIMATED by arithmetic on the drawable floor
(`crates/core/src/geometry.rs:1156-1173`, θ = 0.0011506 rad, reach factor 1738): a 20 m oak of
circumscribed extent 10 m is one pixel at 17.4 km, a disc of that radius holds
π·17400²/100 = **9.51 million trees**, and at 2 µs each the client needs **19 seconds** for one frame.
At flight altitude the first revision's rule is worse still. A per-frame budget that never finishes IS
the arrival pop it was meant to prevent.

**The recommended rule: an OBJECT LADDER of three rungs, with the crossovers set by measurement.**

| Rung | What the client holds | What it draws | Cost follows |
|---|---|---|---|
| near (inside `r_full`) | the full skeleton | tubes for every segment, foliage in every canopy volume | the object count inside `r_full` |
| mid (`r_full` … `r_object`) | the coarsened skeleton | the trunk and one canopy ellipsoid per tree — a real coarsening, never a box | the object count inside `r_object` |
| far (beyond `r_object`) | NO per-tree data at all | a CANOPY layer on the terrain surface the rung already draws: per coarse cell, a canopy occupancy, a mean canopy height and a mean canopy colour | the coarse cell count, which the pyramid already bounds |

`r_object` is the object horizon. ESTIMATED from §1.6's own expand cost: at one tree per 100 m² a disc
of **800 m** holds π·800²/100 = 20 100 trees, which is 40–200 ms of `expand` once, off the main
thread. That is the affordable number, and the horizon is therefore a BUDGET, measured, not a taste.
UNMEASURED today. A 20 m oak at 800 m subtends 0.025 rad, which is **22 pixels** at the reference
view, so the mid-to-far crossover is plainly visible and must be a genuine crossfade with a measured
no-pop assertion (§9 item 13), not a switch. That risk is real, it is named here, and §11 item 11 puts
the crossover to the owner.

**Where the far rung gets its diff.** The canopy layer must know that a forest was felled, or a pilot
at 4 km sees trees that are gone — the arrival pop the provenance base already named
(`block_provenance_collapse.md` §7.3, lines 961-964). **NEW REQUEST to domains 03 and 06: the object
diff FOLDS into the pyramid as a per-coarse-cell canopy occupancy**, exactly the way the pyramid
already folds `fill_256` and `occ_mask` (`block_system_design.md` §3.9.8, lines 4698-4790). One number
per coarse cell, not one row per tree. That is what makes the far rung affordable AND correct.

- **The record (domain 04):** a kind id space large enough for tree kinds and themed kinds (V2.7)
  inside the 16-bit `kind` field; provenance `Feature`; the 6-bit `variant` for V2.6 style. **One
  request, now load-bearing: domain 04 must reserve a per-cell SIDE-TABLE slot (a parameter blob keyed
  by cell) that holds the growth stage and the plant tick**, because domain 04's record has no state
  byte and this report no longer pretends it does (§1.2). The eight-byte record cannot grow.
- **The pyramid (domain 03):** a tree is one cell in the field but a 20 m object in the picture. A
  one-cell tree in an 8 m coarse cell is one bit and 1/512 of fill, which draws nothing. **So the
  pyramid must not carry trees as cells.** It carries the CANOPY FOLD above instead, which is a
  separate small field beside `fill_256`.
- **The diff lane (domain 06):** objects are a SPARSE layer. A client that holds a chunk at a rung
  inside `r_object` also holds that chunk's object diff. Beyond `r_object` it holds the canopy fold
  instead. Domain 06 owes one rule: **the object diff of a chunk ships to a subscriber at every rung
  inside the object horizon, and the canopy fold ships beyond it.**

**NEW (law finding F10). A derived forest is drawn only where a LIVE realm ships its diff.** SL3 says
a realm that is not running cannot be drawn, and SL10 now lets a client derive a moon's static shape
on its own. Nothing joined the two, so a client could draw a dormant moon's seed forest with no diff
to correct it, and every tree players felled would stand again. **The rule: a client draws a realm's
derived objects only for a chunk whose object diff it holds, and only a live realm ships one.** Where
the client holds no diff for a chunk, it draws the far rung's canopy fold from the LAST fold it
received, and where it has none it draws no objects at all. A moon that nobody woke shows terrain, not
a resurrected forest. Measurement 9 asserts it.

**Example.** A pilot at 3 km looks at a hillside. The hillside is beyond the 800 m object horizon, so
the client draws no trees there: the terrain rung carries the hillside's canopy occupancy and its mean
green. The moon's shard is live and shipped the fold, and the fold says three cells of the hillside
were cleared. The pilot sees a clearing. She descends; at 800 m the hillside's chunks cross into the
object rung, the client expands the hillside's 60 seed trees, applies the chunk's object diff, and
draws 57. The crossfade holds the canopy's mean colour, so nothing changes brightness. When she lands,
the same 57 trees gain twigs and leaves. No tree appears or vanishes on approach.

---

## 3. Collision when a tree is cut or when the ground under it is mined

### 3.1 The rule (R6, provenance Feature)

Two events, two outcomes, both from `block_provenance_collapse.md` §4.1 (lines 317-348), re-stated for
an object that is one cell:

| Event | What the shard decides | Result |
|---|---|---|
| **A chop** at height `h` on a structural segment | the object breaks at `h`; the part below stays as the object at a `stump` state; the part above becomes a FALLING GROUP entity | the intact-but-broken case: the crown falls |
| **The ground cell under the anchor is mined** | the object is intact but unfooted; the whole object becomes a falling group, cut height 0; the anchor cell becomes `Air` | the intact-but-unfooted case: the whole tree falls |
| **A placed block is removed elsewhere** | nothing; a tree's SUPPORT is its anchor cell and the one cell beneath it | no dirty set beyond that pair |

Who decides: the realm's shard that owns the cell, and only it (HR1; the provenance base's rule
"support runs only on the realm owner", lines 723-725). A ghost or overlap shard never runs it.

The chop is a tool action on the object, not a cell edit: the shard resolves the tool's hit point on
the capsule compound and takes the height along the trunk. The client sends the ordinary tool input;
the server does the math.

### 3.2 The falling group carries parameters, not cells

The provenance base sized a falling group at 8 192 B of cells (`block_provenance_collapse.md` §4.8,
line 619; §8 row 12). Under one-object the falling group carries **the parameters**: kind, instance
seed, growth stage, size, cut height — about 16 bytes — plus the closed-form topple schedule the base
already designed (§4.5, lines 493-523). The client expands the same skeleton at the cut height and
draws it at the entity's pose.

**The entity kind is new and it is an SL6 request, not a free tag** (§12 request 1). `FallingGroup`
would take a tag in the transient band 10..20 (`crates/core/src/entity_kind.rs:27-41`; the table is
exhaustive and `from_tag` rejects an unknown tag rather than defaulting). Its continuity model would
be `RealmAnchored`, **which is NOT BUILT**: `crates/core/src/entity_kind.rs:23` says
"Guided/RealmAnchored/InBand land at P6/P10", and only two behaviour triples exist today
(`{Durable, Frozen, Always}` and `{Transient, BallisticReadvance, NeverTransferOnly}`). So the falling
group costs a new continuity model, which the file itself calls "first-class engineering and a
crash-matrix multiplication". The first revision presented it as four lines. It is not.

### 3.3 The fall: a SCRIPTED pose and a ONE-WAY pusher

**CORRECTED (law finding F3, feasibility finding R5). The first revision wanted both a colliding fall
and a path-independent landing, and those two cannot both hold.** If the crown stops on a parked hull,
the final pose depends on the path, and the property that makes a dormant moon and a live moon write
identical bytes is destroyed. If the crown does not stop, it is a phantom miss. The first revision also
misread the base: line 503 of `block_provenance_collapse.md` reads "No physics body, no collider, no
float" and gives no size reason; the base's reason was the closed-form determinism of the schedule
(lines 500-515), not the collider's cost.

**The recommendation: the topple is SCRIPTED, and its capsules are a ONE-WAY PUSHER.**

1. The crown's pose is a closed-form function of the hinge, the cut height, the instance seed and the
   TERRAIN under the hinge. Terrain is static shape, so both hosts and both paths agree, and a dormant
   moon and a live moon write identical bytes. The crown never sinks into the ground, because the
   schedule reads the ground.
2. The capsules under that pose PUSH an occupant they overlap. The containing realm's shard is the
   physics authority for its occupants, so it applies the push when it authors the occupant's pose.
   Nothing about the tree changes.
3. The crown does NOT react to anything. It passes through another tree's trunk, through a parked
   hull that cannot be pushed, and through a wall.

**Say the cost plainly. This accepts a phantom miss against IMMOVABLE drawn things** — a neighbouring
trunk, a building, a landed hull that is anchored — and it refuses a phantom miss against a player, a
suit or a free occupant, because those get pushed. That is the honest trade, and it is the trade that
keeps the dormant world byte-identical.

**And say the lag plainly (feasibility finding R5).** The crown's pose rides the entity lane through
the client's 100–150 ms interpolation buffer (`crates/client/src/tuning.rs:8-18`; `CLAUDE.md:261`), and
there is no client prediction. So the collider is always AHEAD of the drawn crown. ESTIMATED from the
report's own fall (fifty ticks at 20 Hz = 2.5 s) with a 20 m crown whose tip moves at roughly 16 m/s
near landing: a 150 ms lag puts the collider about **2.4 m** ahead of what the player sees. **Neither
option removes the divergence; option (a) moves it into TIME.** The choice is between a push the player
has not yet seen and a miss the player has seen. §9 item 14 owes the measurement of that gap on the
shipped lane, and §11 item 3 puts it to the owner with this cost stated.

**Example.** A pilot parks a hull under an oak and chops the trunk. The crown falls and pushes the
hull aside, because the planet's shard authors the hull's pose and applies the push. A second pilot
lands a hull on legs and locks it down; the crown passes through that hull's plating, because a locked
hull is immovable. The report states which of those two happens; the first revision did not.

### 3.4 Landing: conservation with no lying tree

On landing the group resolves per skeleton row from the kind's yield table:

- structural segments → `DroppedBlock` stacks of wood units (the existing kind, tag 11);
- canopy volumes → leaf drops at the kind's declared yield, no tool gate (R6-7's amendment).

`units in == units out` is a unit assertion over the skeleton's yield rows. Nothing is created and
nothing vanishes. A lying-tree object (`FallenTree`, a kind with a horizontal skeleton) is reserved as
a later kind; v1 itemises (§11 item 4). The landing delivers the ordinary `Hit` to the cells it lands
on, with the base's `FALL_DP_PER_KJ` (R6-11) unchanged.

**NEW (feasibility finding R13). A crown that leaves the realm while it falls.** On a small body a
felled crown can topple past the realm's bound before it lands. This report does NOT design that
crossing, and it does not pretend to. The lawful shape is the ordinary one: the crown is an occupant of
the realm that owns the cell, and if it leaves that realm's bound it re-homes through THE crossing, the
same machinery every occupant uses. That costs the `RealmAnchored` continuity model (§3.2) and a
crossing test. **The simple alternative, recommended for v1: the topple schedule CLAMPS the crown's
travel to the realm's bound, so a crown never leaves.** A 20 m oak on a moon whose bound is kilometres
wide never reaches the clamp; a 20 m oak on a 30 m garden asteroid does, and it lands short instead of
flying off. §11 item 12.

### 3.5 What the client sees: the atomic chop is a MECHANISM, not a declaration

**CORRECTED (law finding F2). The first revision DECLARED that a diff and a spawn sharing a tick are
applied atomically, and the two lanes in the code cannot keep that promise.** The chunk diff rides the
RELIABLE per-subscription bulk lane (`crates/wire/src/channels.rs:283-296`). The entity spawn rides the
UNRELIABLE 20 Hz world-state datagram (`crates/wire/src/channels.rs:450`). The client then holds every
entity 100–150 ms in the interpolation buffer (`crates/client/src/tuning.rs:8-18`). So the stump
arrives on a reliable stream and draws at once, and the falling crown arrives on a datagram and draws a
tenth of a second later. **That is the black-frame seam and the flicker seam this report claims to
prevent.**

**The mechanism that makes it true:**

1. **The object diff carries the TICK it belongs to.** Domain 06's object row gains an `apply_tick`
   field. It is the shard's tick, the same clock the snapshot's `universe_tick` stamps
   (`crates/wire/src/channels.rs:455-459`).
2. **The client QUEUES an object diff instead of applying it on arrival.** It applies the diff when
   the render cursor reaches `apply_tick` — the same instant the interpolator first draws the falling
   group. The interpolation buffer is what makes this possible: the diff arrives EARLY on the reliable
   lane and waits, so the reliable lane's head start becomes the schedule, not the seam.
3. **A diff whose `apply_tick` is already past applies at once.** A client that joins mid-fall sees the
   stump and the crown in the same frame, because both come from the same snapshot's state.
4. **The same rule governs the landing:** the despawn and the drop spawns share one tick.

At a coarse rung inside the object horizon the falling group is still an entity and rides the same
lane, so a felled tree at 2 km falls at 2 km (the base's AoI rule, §7.3 lines 966-972, holds
unchanged). Beyond the object horizon there is no falling group at all; the canopy fold changes and
the crossfade covers it.

**Measurement 15 makes it a result rather than a rule:** chop an oak 1 000 times on the shipped lanes
and assert that no frame shows a stump without a crown and no frame shows two crowns.

**Example.** A player chops an oak at 1.2 m. The moon's shard writes the oak's record to `stump` (its
skeleton is now the trunk to 1.2 m, two capsules) with `apply_tick = 40 812`, and spawns a falling
group carrying `(oak, seed, stage 255, cut 1.2 m)` in tick 40 812. The diff reaches the client 90 ms
before the snapshot does and waits in the queue. When the render cursor reaches tick 40 812, the client
shows a stump and a leaning crown in the same frame. Fifty ticks later the crown lands, despawns, and
eleven wood stacks and a leaf stack spawn in the same tick.

---

## 4. Composites (R4): the same object with a size parameter, not pattern recognition

### 4.1 The two candidates

| | **A. One object mechanism with a size parameter** (recommended) | **B. Pattern recognition over placed cells** (the base's O26) |
|---|---|---|
| "more tree blocks → a bigger tree" | the tree's growth stage and size class; a player feeds material into the object or waits | a recogniser matches a wood column plus leaf cells and swaps in an art mesh |
| "rock blocks → a non-blocky rock" | a `Rock` object kind with a size parameter; adding a stone unit increments size and the expansion grows the boulder | a recogniser over stone cell patterns with rank, isolation shell and orientation clauses |
| collision | the object's expansion: capsules for trunks, an ellipsoid set for rocks, all of it collides | the cells; an art mesh larger than its cells ("all of it must collide") has no server-side shape — the base itself refused meshes on the server (`block_provenance_collapse.md` §6.3, lines 858-863) |
| what breaks it | remove a unit → size − 1; remove the support → it falls; nothing else | any edit in the pattern's dirty set; the felled-log-is-a-new-tree exploit (§6.4 line 869); the wall-becomes-boulder exploit (`collidable_decoration.md` §4.4, lines 344-354) |
| stored bytes | one 8 B record plus a side row | one 8 B record PER CELL, ~998 for a tree (UD-3) |
| new subsystems | none: it is §1's mechanism | a recogniser, a per-realm recognition budget, an emit-exclusion hook in the mesher, a bake tool, a coverage gate |
| where V2.2 stands | V2.2 says a tree is ONE object; A is that sentence applied to rocks | B contradicts V2.2 for trees outright |

**Recommendation: A.** V2.2 already decided it for trees. A rock that is "not blocky" is the same
mechanism: a `Rock` kind whose `expand` returns an ellipsoid set (a convex-ish lump from the instance
seed) with a flat, lattice-snapped crown so a player can stand on it (the base's crown rule,
`collidable_decoration.md` §11.1 item 5). Collision is the ellipsoid set; the mesh lies inside it,
within the same `PHANTOM_HIT_MAX_M` bound as a tree (§1.3).

### 4.2 The size parameter

For an inorganic object the SIZE — units of material the object holds — takes the same side-table slot
the growth stage takes for a tree (§1.2). **CORRECTED (law finding F1): it is not the record's state
byte, because domain 04's record has no state byte.** Placing a stone block onto an existing rock
object adds one unit (the placement path recognises "the target cell holds an object of a kind that
accepts this substance" and increments, instead of placing a cell). Mining the rock removes one unit
and drops it. At zero the record is `Air`. The expansion grows monotonically with size, so adding a
unit never moves the crown down (no phantom miss under a standing player).

Small stones stay CELLS: a `Nub`, `Plate` or `Quarter` from the shape catalogue
(`collidable_decoration.md` §4.1 table) for anything the catalogue holds. The object begins where the
catalogue ends — a rock bigger than one cell.

### 4.3 The pattern-broken rule, stated for the object

There is no pattern, so the rule is short:

1. **An object's EXPANSION depends on its own record and its own side row only, never on a
   neighbouring cell. Its SUPPORT set is its anchor cell and the ONE cell beneath the anchor's up
   face.** An edit two cells away changes nothing.
   **CORRECTED (feasibility finding R3).** The first revision said an object's dirty set is its anchor
   cell alone, and three lines later said the object falls when the ground cell under the anchor is
   mined. Those cannot both hold: the ground cell under the anchor IS a neighbouring cell.
2. **DISPUTED: the refuter says this costs a reverse index the report does not design. It does not.**
   The per-chunk object list is ALREADY keyed by the anchor cell (§1.7). When a miner removes cell `C`,
   the shard asks that same map whether an object is anchored at each of `C`'s six face neighbours,
   and keeps only an object whose own up face points back at `C`. That is at most six map lookups per
   edit, on a map that exists for another reason. It is O(1)-ish, it needs no new structure, and it
   respects SL9 ("Finding which child holds a point is a LOOKUP, never a scan", `CLAUDE.md:221-222`).
   The refuter's cost is real only if the object list is not keyed by cell, and §1.7 says it is.
3. An object whose size parameter is reduced shrinks by the expansion at the new size. At size 0 it is
   air.
4. A structural segment that is cut splits the object (§3). A canopy volume is never structural.
5. A felled object is drops (v1). A drop is never re-recognised into anything. The base's requirement
   "recognition must consume orientation" (`block_provenance_collapse.md` §6.4) is discharged by
   having no recognition.

**Example.** A miner digs a trench under a row of oaks on a moon. Each removed cell asks the chunk's
object map about its six neighbours, finds the oak anchored on the cell above, and turns that oak into
a falling group. A builder stacks a 2×2×2 wall of stone blocks. Under B the wall is a boulder and stops
being a wall. Under A it is eight cells and stays a wall. To make a boulder the builder places a rock
object and feeds it seven more units; the boulder is one record of size 8.

### 4.4 What A does not give, said plainly

A does not make a tree from tree BLOCKS placed by hand (the owner's literal 2026-08-03 sentence). A
player who wants a bigger tree plants a bigger sapling (size class), feeds it, or waits for growth. If
the owner wants hand-stacked wood and leaf blocks to become a tree, that is B and it reopens every
cost in the right-hand column. Open decision §11 item 5.

---

## 5. Art: a generated mesh or an art-pack asset — and what the render seam exposes

### 5.1 The two ways to draw one skeleton

| | **(a) Procedural mesh from the skeleton** | **(b) An art-pack asset picked by the parameters** |
|---|---|---|
| what the engine receives | the skeleton and the parameters; the client library tessellates tubes and foliage into `MeshPrim` (`crates/client/src/realm_scene.rs:791-796`), the seam that exists today | the parameters and the skeleton; the engine maps `(kind, instance seed, stage)` to an archetype from a pack, with per-instance scale, lean and hue from the seed |
| collision agreement | by construction: the tube radius ≤ the capsule radius; the skeleton is the mesh's source | by authoring: the archetype's skeleton is a TABLE baked from the asset; `expand` for a pack kind is a lookup plus a scale, not a procedure. **The bake must meet `PHANTOM_HIT_MAX_M` (§1.3): the capsule surface within 10 cm of the drawn bark everywhere a player, a suit or a landing gear can touch. An archetype table that cannot meet it is REFUSED** |
| variety | infinite, from the seed | the pack's archetype count × per-instance variation; a 20 m oak and a 22 m oak are the same asset scaled |
| cost | ESTIMATED weeks of graphics work; "procedural trees look procedural" is the risk the base named (`block_system_design.md` §6.8, USER DECISION 6-2 option D) | €46–200 for greybox packs, €15–40 k commissioned (§6.8); a bake tool; two live licence clauses (source files never leave the team, never feed an AI tool); the FBX→glTF pipeline; a per-engine importer for an Unreal client |
| engine independence (V2.8) | complete: the seam carries geometry an engine only uploads | partial: each engine ships its own asset build of the same pack |
| HR6 (agent-operable, no assets in CI) | the harness renders trees with no pack | the harness needs the pack or a greybox substitute |
| themes (V2.7) | a stylised kind is a different `expand` | a stylised kind is a different archetype table |

### 5.2 What the seam must expose either way

The render seam (the engine-free client library's output, V2.8) carries, per object:

```
ObjectDraw {
    kind,                      // registry id; an engine on route (b) may pick an asset by it
    params: { instance_seed, growth_stage, size, yaw },   // an engine may vary an asset by them
    skeleton,                  // the crate's integer skeleton, the same bytes the server collided
    style,                     // V2.6 parameters the client picks, never stored on the server
    placement,                 // the anchor cell's position IN THE REALM'S OWN FRAME
    rung,                      // the detail the client asked for
}
```

**The engine gets BOTH the parameters and a reference mesh.** The client library always produces a
`MeshPrim` set from the skeleton (route (a)); an engine may ignore it and draw an asset for a kind
whose skeleton is table-authored (route (b)).

**CORRECTED (feasibility finding R10). Route (b) is NOT free, and this report no longer calls it
free.** The code states the opposite invariant: `crates/client/src/realm_scene.rs:788` says the
renderer "consumes ONLY this — it never learns the word 'sphere' or 'box' (adversary H4)", and
`crates/client-render/src/lib.rs:1826-1827` says the mesh build is "PURE glue: no shape branch, no
logic — exactly the H4 seam". **An engine that maps a kind id to an art asset IS a shape branch in the
renderer**, and it is the branch that seam exists to prevent. That may be the right trade for V2.8, and
the trade is a real cost: route (b) puts a per-kind table in every engine, and the H4 harness must
either carry the pack or fall back to route (a) and draw a different picture from the shipped client.
Route (a) keeps one seam for the Bevy client, the Unreal client and the HR6 harness. Route (b) does
not, and §11 item 2 now says so.

**CORRECTED (law wording note).** The first revision wrote *"the anchor cell's composed position"*.
"Composed" is the gateway's word. The seam carries the anchor cell's position **in the realm's own
frame**; the gateway composes it. A seed tree's anchor cell is static shape and the client derives that
CELL (SL10 V1.1). A planted tree's anchor arrives in the diff. A falling tree's pose arrives from the
server every tick. **The client never derives a pose** (SL10 V1.7).

### 5.3 Themes (V2.7)

A stylised tree, a mushroom forest, a crystal spire on another planet: each is a registry KIND with
its own `expand`. Kinds are an append-only table in the generator crate, the same discipline as the
draw stream. A new kind bumps the generator tag, so a client without it is refused at the door (SL10
V1.3), never shown a wrong tree. A kind's variant (V2.6) selects client-side details the server never
stores. **Every themed kind obeys the seed fence of §2.1: it may drop common substance only.**

**Recommendation.** Ship route (a) as the reference and the harness path; take route (b) for a kind
only when the owner picks a pack AND the archetype bake passes the phantom-hit bound (UD 6-2 stays the
owner's). Do not adopt any tree-generation library silently; the crate's own expansion is small and
must obey V1.4, which a third-party library will not.

---

## 6. Decoration stays derived and never collides

Grass, tufts, moss, pebbles, sand ripples: the base's rule stands and SL10 now makes it lawful in
one sentence. Decoration is a function of `(seed, address, the received diffs)`. The client derives it;
the server may sample it at a point (`vd_decor::query`, `block_system_design.md` §6.6, lines
11310-11346). It never owns a collider, a raycast hit, a fence or a byte of storage
(`collidable_decoration.md` §3, lines 152-198; the promotion boundary, `block_system_design.md`
lines 11347-11383).

**Nothing in this report relaxes the boundary.** The tree object sits on the PROMOTED side: it is
persisted (a record), it collides (its capsules), a player acts on it (chop, plant, feed). The test
the base wrote still decides every case: *can a player act on it? Then it is promoted.* A tuft is not
acted on, so it stays scatter. A bush a player harvests is an object kind with no structural segments
(pass-through, R13) — the same mechanism, not a relaxation. A bush nobody harvests is scatter (R12).

**Example.** A meadow of 50 000 blades is drawn by the client from the soil cells and the seed. A
player walks through it; the moon's shard slows the walk from the soil's `traction`, not from any
blade. A berry bush in the meadow is one object record; the player picks it, and a diff clears its
`fruit` bit in the bush's side row.

---

## 7. Seams to watch (SL8), named for trees

| Seam kind | Where a tree could cause it | The rule that prevents it | Status |
|---|---|---|---|
| detail-by-box | a far tree drawn as a box | the mid rung is a real coarsening of the skeleton: trunk plus one canopy ellipsoid, never a box | rule stated, §9 item 8 measures it |
| arrival pop | a felled tree returns at 4 km, or a forest appears on approach | the object diff ships at every rung INSIDE the object horizon, and the canopy fold ships beyond it (§2.5); a derived forest draws only where a live realm ships a diff (§2.5) | rule stated, §9 item 9 measures it |
| **mid-to-far crossover** | **a 20 m oak is 22 pixels tall at the 800 m object horizon; switching to the canopy fold there is plainly visible** | **a measured crossfade over a band, with the canopy fold's mean colour and mean height matched to the mid rung's silhouette** | **OPEN — the honest risk of the object horizon (§2.5, §11 item 11), owed §9 item 13** |
| brightness pop | foliage at the rung crossfade | the canopy ellipsoid's lit colour equals the foliage's mean colour, asserted per kind | rule stated |
| flicker / black frame | a chop, a landing | the tick-stamped object diff queued to the render cursor (§3.5) — a MECHANISM, not a declaration | mechanism stated, §9 item 15 measures it |
| tick hitch | expanding a disc of trees on arrival | the object horizon bounds the count at ~20 100 (§2.5); the expansion runs on the task pool with a per-frame budget; the disc is expanded from the outside in | bound stated, §9 item 4 measures it |
| re-state rate | a growth stage change | a stage step moves any point by less than the gate's bound (§9 item 2) | rule stated |
| **lane flood** | **a recovering forest emitting a diff per stage per tree** | **a stage change reaches only a subscriber that holds the chunk at a rung where the step is drawn; a dormant recovery collapses to one row on arrival (§2.3)** | **NEW — the first revision had no answer, §9 item 12 measures it** |
| jump | a falling tree | the scripted topple's pose is on the entity lane through the interpolation buffer, and the collider leads the drawn crown by an ESTIMATED 2.4 m (§3.3) — a stated, measured gap, not a solved one | OPEN, §9 item 14, §11 item 3 |

---

## 8. Stale claims in the investigation base (superseded by V2.2, SL10, or this design)

1. R13/R14's "bushes and canopy cells are real cells that hold the tree's 120:1 spread"
   (`block_system_design.md` lines 78-79; `collidable_decoration.md` §7.1, lines 707-717) —
   superseded by V2.2: a tree is one cell; the spread is the skeleton. R14's pass-through canopy
   survives as a property of the canopy volumes.
2. UD-3 / O25 "≈998 cells per tree" and the four numbers derived from it
   (`collidable_decoration.md` §7.4, lines 766-805; `decision_board.md:709-719`) — no cell count exists.
3. O26 "composites are RECOGNISED, never overlaid" and the P4.2 emit-exclusion hook
   (`decision_board.md:723-750`, `:1522`) — superseded by §4: no recogniser, no hook.
4. The falling-group blob of 8 192 B and `feature_max_cells = 1 024`
   (`block_provenance_collapse.md` §4.8 line 619, §8 row 12) — a falling group carries ~16 B of
   parameters. Its CONTINUITY MODEL is not free, though: `RealmAnchored` is unbuilt (§3.2).
5. Stage 8's "attachment cells 1..=9 per instance" (`block_provenance_collapse.md` §4.4, lines
   358-364) — one anchor cell per object.
6. The clear-cut storage alarm, 1 845 B per felled tree and 0.025 % of a planet
   (`block_provenance_collapse.md` §5.2-5.4, lines 744-818) — 8 B per felled tree. A GROWING tree adds
   an ESTIMATED 16 B side row, which a felled tree does not have.
7. The forest collider line, 6.9–22.5 MB per cluster (`collidable_decoration.md` §9.4, lines
   1004-1025) — ESTIMATED 4.49 B/m² under capsules against the base's 199–650 B/m², a 44× to 145×
   gain; owed a measurement (§9 item 5). **The first revision's 35×–112× compared two different
   areas.**
8. `block_system_design.md` §6.3 worked example 3 (lines 10934-11021): the leaf distance field
   (3 bits + persistent), `CubeWithShell`, shell clusters — superseded for trees. Leaf BLOCKS as
   building blocks (V2.3) may keep a cube look; they are not trees.
9. USER DECISION 6-1 "client-derived decoration collides with the server-does-all-math law and is
   a one-way door" (`block_system_design.md` §6.1, lines 10243-10288) — settled by SL10 for static
   shape; the code-shape door (the crate graph) still stands.
10. O57 / R6-12 "give the Cartesian test profile a TEST-ONLY generator emitting three boulders"
    (`decision_board.md:1128`; `block_provenance_collapse.md` §4.9, R6-12) — forbidden by SL5 (no
    test-only world) and by "test exactly production". The HR4 fixture PLANTS an object through the
    shipped placement path: a player places a rock object in a hull and on a planet.
11. R4's own text "placing several tree blocks yields a real tree from an art pack" — the owner's
    2026-09-07 V2.2 supersedes the mechanism; the intent (a bigger tree from more material) is served
    by §4.2.
12. `collidable_decoration.md` §11.1 items 1–4 (the conservative footprint, no member thinner than 1 m,
    the 1 024-cell cap, the solid simply-connected canopy) — art constraints of the cell model; under
    capsules a thin branch is a drawn thin segment, and the constraint is the phantom-hit bound of
    §1.3, which is a NUMBER, not a shape rule.
13. **NEW: `block_provenance_collapse.md` §3.1's eight-bit "state" field (lines 137-147).** Domain 04's
    recommended record has no state byte (`04_block_record_registry.md:112-136`). A tree's growth stage
    and a rock's size live in the side table instead (§1.2).
14. **NEW: this report's own first revision, in the places §14 lists.**

---

## 9. Measurements owed

1. `tree_skeleton_golden` — for every kind, 4 096 `(instance seed, stage, size)` inputs → skeleton
   bytes, identical on the server build and the client build, x86-64 and aarch64. The SL10 gate's tree
   SHAPE arm.
2. `tree_growth_step_bound` — for every kind and every stage `s`, the largest displacement of any
   skeleton point between `s` and `s+1` is below a stated physical bound (proposed 5 cm; ESTIMATED).
3. `tree_render_inside_collider` — **a DISTANCE, not a membership.** Every tube vertex lies inside its
   capsule AND within `PHANTOM_HIT_MAX_M` (proposed 0.10 m) of the capsule surface, everywhere a
   player, a suit or a landing gear can reach. Every canopy vertex lies inside its ellipsoid volume.
   Red on one vertex further in than the bound. (NEW — law finding F4.)
4. `bench-expand` — `expand` per tree per kind at stage 255 on the reference CPU; an object-horizon
   disc of seed trees on a client, wall time and main-thread time.
5. `bench-collider-forest` re-issued — stored collision bytes AND broad-phase query time per 64 m disc
   at one tree per 100 m² under the capsule compound, on both geometry lanes, against the chosen
   implementation of §11 item 9.
6. `felled_tree_bytes` — fell 1 000 seed trees; assert 8 B ± 0 of permanent record per tree in the
   WAL after compaction, and 0 B of side row.
7. `object_placement_validation` — placement against a chunk's object shapes, p99 per placement; and
   the six-neighbour support lookup on a mining sweep, p99 per edit (NEW — feasibility finding R3).
8. `tree_frame_cost` — a forest at one per 100 m² at each rung of the object ladder, GPU time on
   VD-REF (the base's §6.10 budgets are for card clusters and do not carry over).
9. `object_diff_at_every_rung` — a felled forest observed at 4 km, then approached: zero trees appear
   or vanish. **Plus: a DORMANT moon's forest observed with no diff held — assert no felled tree
   returns** (NEW — law finding F10).
10. `assert_feature_anywhere(object)` — plant a rock object and a tree object on a Spherical planet
    profile and inside a Cartesian hull profile through the shipped placement path, with a forced
    `reanchor()`; identical records, identical skeletons, identical capsule sets.
11. **NEW: `tree_placement_golden` — the SL10 gate's PLACEMENT arm.** For a fixed list of chunks, the
    SET of `(cell, kind, size class)` stage 8 derives is byte-identical on the server build and the
    client build, on x86-64 and aarch64. Gated on retiring the seven forbidden libm sites listed in §0.
    (Feasibility finding R2.)
12. **NEW: `growth_lane_rate` — a recovering 786 m disc: assert the growth lane emits one row per tree
    per subscribed visit, never one per stage.** (Feasibility finding R7.)
13. **NEW: `object_horizon_crossfade` — a 20 m oak driven across the mid-to-far crossover: assert no
    frame changes the drawn silhouette area or the mean colour by more than a stated bound.**
    (Feasibility finding R1.)
14. **NEW: `falling_collider_lag` — over one fall on the shipped lanes, the largest distance between
    the collider pose and the drawn pose.** ESTIMATED 2.4 m; the measurement decides whether §11 item 3
    can ship. (Feasibility finding R5.)
15. **NEW: `atomic_chop` — chop an oak 1 000 times on the shipped lanes; assert no frame shows a stump
    with no crown, and no frame shows two crowns.** (Law finding F2.)
16. **NEW: `slow_tick_absent` — a moon holding one million growing trees and no observers: assert the
    shard does zero per-tree work per tick.** The SL9 shape of §2.3. (Law finding F7.)

---

## 10. One-way doors

| Door | Must shut | Cost if wrong |
|---|---|---|
| The tree's use of domain 04's record: the kind id in the 16-bit `kind` field, provenance `Feature`, the 6-bit `variant` for style, and **the reserved per-cell side-table slot that holds the growth stage and the plant tick** | with domain 04's record freeze, before the first world is saved | a record migration over every saved planet |
| The instance seed rule `hash(realm seed, cell, kind)` and stage 8's draw position in the region stream | before the first world is saved | every tree in the universe re-rolls; a player's remembered forest changes (the discovery-permanence law) |
| The skeleton format — **segment lattice in 1/1024 m, radii in 1/1024 m, canopy volumes** — as part of the generator tag | before the first client links the crate | a client refusal fleet-wide, and a changed collider under an existing treehouse |
| Objects collide by SHAPE; placement validation reads object shapes from the first edit slice (P6) | with the first block placement | builds interpenetrate trunks forever, or every such build must be re-validated |
| Object kinds as an append-only registry with themed kinds in reserve (V2.7) | with the first kind | renumbering a kind reinterprets every saved tree |
| The object diff ships at every rung inside the object horizon, and the canopy fold folds into the pyramid beyond it (domains 03 and 06) | with the first coarse rung | a felled forest pops back at distance — a seam that is then a protocol change to fix |
| **The object diff's `apply_tick` field, and the client's queue-to-render-cursor rule** (§3.5) | with the first object diff on the wire | every chop shows a black frame or a double tree; adding the field later is a protocol change |
| **The phantom-hit bound `PHANTOM_HIT_MAX_M`** (§1.3) | before the first art-pack archetype table is baked | a pack is baked to a looser bound and every asset must be rebaked |

---

## 11. Open decisions for the owner (each with a default that ships if nothing is said)

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| 1 | Where does a planted tree's instance seed come from? | (a) derived from the cell address, zero bytes; (b) stored per instance in the side row | **(a)** | eight bytes per tree, no new record; and a replanted cell growing the same tree is CORRECT, not a defect — it is what lets the record prune (§2.4); (b) becomes additive later |
| 2 | The art route | (a) procedural reference mesh in the client library; (b) art-pack archetype tables; (c) both, engine's choice | **(c) with (a) shipped first** | V2.8 needs an engine-free seam and HR6 needs assetless rendering. **(b) is NOT free: it puts a shape branch back in the renderer, which the H4 seam exists to prevent** (§5.2), and its bake must meet the phantom-hit bound |
| 3 | Does a falling tree collide while it falls? | (a) yes, a ONE-WAY pusher under the scripted topple pose (§3.3); (b) no, as the provenance base chose | **(a)**, with the cost stated | **Neither option is clean.** (a) pushes a player an ESTIMATED 2.4 m before the player sees contact (§3.3), and it still passes through immovable things. (b) is a miss the player has seen. Measurement 14 decides |
| 4 | What does a landed tree become? | (a) drops only; (b) a `FallenTree` object lying on the ground; (c) (a) now, (b) reserved | **(c)** | conservation is a unit assertion under (a); (b) is an additive kind |
| 5 | Do hand-stacked wood and leaf BLOCKS ever become a tree (the literal R4)? | (a) no: a bigger tree is a bigger sapling, feeding, or growth; (b) yes: build the recogniser | **(a)** | V2.2 already made a tree one object; (b) reopens every cost in §4.1's right column |
| 6 | Is the canopy pass-through for every kind? | (a) always, R14; (b) a per-kind `canopy_collides` flag, default false | **(b)** | one data bit; a dense conifer or a themed crystal spire may want a solid crown; the default keeps R14 |
| 7 | Small rocks: cells or objects? | (a) cells from the shape catalogue up to one cell, objects above; (b) objects always | **(a)** | a `Nub` is already a cell with a collider and a drop; an object is worth its record only past the catalogue's largest extent |
| 8 | The HR4 object fixture | (a) plant through the shipped placement path on a planet and in a hull; (b) a test-only generator (O57) | **(a)** | SL5 and "test exactly production" forbid (b) |
| 9 | **NEW: how does a shard collide a capsule compound?** | (a) `rapier3d`/`parry3d` — an OWNER OPTION, not adopted here (neither is in `Cargo.lock`, MEASURED §0); (b) a hand-written capsule sweep on the `glam` types the workspace already has (`DVec3`, `DQuat`) | **(b) for v1** | a tree needs capsule-vs-capsule and capsule-vs-capsule-sweep only, which is a page of arithmetic; (a) brings a large dependency and its own determinism profile into the crate SL10 pins. **Never adopt (a) silently** |
| 10 | **NEW: is growth memoryless (base R8) or a deterministic schedule?** | (a) memoryless, per-tick probability — needs a stored per-chunk stamp and an integer binomial sampler nobody has named; (b) a stored plant tick plus a fixed integer step table with per-instance jitter | **(b)** | (b) stores one tick, walks nothing, costs zero on a moon with no observers (SL9), and needs no sampler. It changes R8's stated intent, so the owner should say |
| 11 | **NEW: where is the object horizon, and what is beyond it?** | (a) ~800 m, and beyond it a canopy fold on the terrain surface (recommended); (b) further out, with a measured budget; (c) per-tree impostors beyond it | **(a), with §9 item 13 as the gate** | (a) bounds the expand cost at an ESTIMATED 20 100 trees. Its cost is honest: a 20 m oak is 22 pixels at 800 m, so the crossover is visible and the crossfade must be measured, not assumed |
| 12 | **NEW: a crown that would topple out of its realm** | (a) the topple CLAMPS to the realm's bound, so a crown never leaves; (b) the crown re-homes through THE crossing | **(a) for v1** | (b) needs the `RealmAnchored` continuity model, which `crates/core/src/entity_kind.rs:23` says lands at P6/P10. (a) matters only on a body a few tens of metres wide |

---

## 12. Law conflicts and SL6 requests

**CORRECTED (law finding F9). The first revision answered "None found", and it read only half of SL6.**
`CLAUDE.md:210-213` says: *"SL6 — ASK BEFORE NEW DATA CROSSES A REALM BOUNDARY, **and before adding a
wire arm.** Default NO."* This design adds wire-visible things, and it now asks for them by name.

**No new data crosses a REALM boundary.** The object record lives in the realm that owns the cell. The
diff goes from that realm's shard to the gateway to the client, which SL2's 2026-08-24 clarification
says is not a realm crossing. The falling group is an entity of the same realm. A seed forest crosses
nothing. An object's shape may not reach outside its realm's bound (§1.7 refuses it on BOTH the seed
path and the placement path), so no object shape ever needs a neighbouring realm's frame.

**Four wire additions, each an SL6 request, default NO until the owner answers:**

1. **A new entity kind tag for `FallingGroup`.** Data: one tag in the transient band 10..20. From: the
   realm's shard. To: the gateway, then the client. Why the receiver cannot compute it: a falling crown
   is live state; nothing derives it. What doing without costs: a chopped crown cannot be an entity, so
   it either vanishes at the cut (a black-frame seam) or the whole fall becomes a chunk animation that
   the diff lane cannot carry at 20 Hz. **Extra cost the first revision hid: it also needs the
   `RealmAnchored` continuity model, which is not built** (`crates/core/src/entity_kind.rs:23`).
2. **The falling group's parameter blob.** Data: kind, instance seed, growth stage, size, cut height —
   ESTIMATED 16 B. From: the realm's shard. To: the client. Why the receiver cannot compute it: the cut
   height is a player's action. What doing without costs: the client cannot expand the crown's
   skeleton, so it draws nothing that falls. Constraint: entity blobs are TLV-framed and
   decode-to-Default is BANNED for Durable kinds (`CLAUDE.md:265-267`); a `Transient` kind still owes a
   named decode error, never a default.
3. **New TLV tags for object rows inside `BulkKind::ChunkDelta`.** Data: an object row (cell, kind,
   variant, provenance) and its side-row fields (stage, plant tick), plus the `apply_tick` field of
   §3.5. From: the realm's shard. To: the client. Why the receiver cannot compute it: an edit is state.
   What doing without costs: no chop, no planting and no growth reaches a client. Note: domain 04
   already places cell, attachment and box TLV tags inside `ChunkDelta`
   (`04_block_record_registry.md:645-647`), so these are new tags on an existing lane, not a new arm —
   but they are still new wire, and SL6 covers them.
4. **The canopy fold in the pyramid entry.** Data: a per-coarse-cell canopy occupancy, a mean canopy
   height and a mean canopy colour. From: the realm's shard. To: the client. Why the receiver cannot
   compute it: the fold must include what players felled, which is state. What doing without costs: the
   far rung either draws a resurrected forest or draws no forest, and both are the arrival-pop seam.

`ObjectDraw` (§5.2) is the client library's own output, not a wire arm, and it needs no request.

**One note for the record, not a request:** a hull that lands in a forest is an occupant of the planet
realm, and the planet's shard collides it with the planet's capsules. The hull's own realm learns
nothing about the trees. That is SL2 working as written.

---

## 13. Summary of the recommended design

A tree is one eight-byte record on the air cell above the ground, plus one small side row while it
grows. One function in the one generator crate expands `(kind, instance seed, growth stage, size)` into
an integer skeleton on the 1/1024 m lattice. The server turns the skeleton into a capsule compound and
collides with it, within ten centimetres of the drawn bark. The client turns the same skeleton into a
mesh and draws it, or hands the parameters to an engine that draws an asset and pays a shape branch for
it. The gate compares the skeleton bytes AND the derived placement set. A seed forest costs zero bytes
on both hosts, out to an object horizon of an ESTIMATED 800 m; beyond it the terrain rung carries a
canopy fold that the object diff folds into. A cut is an 8 B removal diff. A planted tree stores a
plant tick and reads its stage from a fixed schedule, so a moon with no observers costs nothing. A
regrown tree deletes its own record when it matches the seed's tree again. A chop or an undermined
anchor makes a falling-group entity that carries parameters, topples on a scripted schedule that reads
only static shape, pushes what it can push and passes through what it cannot, and itemises on landing —
with the diff tick-stamped and applied at the render cursor so the stump and the crown appear in one
frame. A rock is the same object with a size parameter; there is no pattern recogniser. Grass stays
derived and never collides.

**Example, end to end.** A pilot flies toward a moon. At 20 km the moon's forests are a green tint on
the terrain, folded per coarse cell, and three cleared patches show where loggers worked. At 800 m the
patches' chunks cross the object horizon; the client expands 20 100 skeletons off the main thread,
applies the moon's object diff, and the trunks fade in without a change of colour. The moon's shard
holds the same forest as capsules and stops the hull's landing gear on an oak's trunk. The pilot steps
out, walks to within ten centimetres of the bark, walks through the leaves, and chops the oak. A stump
and a leaning crown appear in the same frame, because the diff carried the crown's tick and waited for
it. The crown pushes her suit aside and lands as wood stacks. She plants a pine in the clearing: one
record and one side row holding the tick she planted it. Weeks later, with nobody there, the shard does
no work at all; when she returns it reads the pine's stage from the schedule, finds the oak matches the
seed's oak again, and deletes the oak's record.

---

## 14. Revision log

Two refuters reviewed revision 1. This revision answers every finding. "Fixed" means the report now
says something different. "Disputed" means the claim stays and carries my evidence.

### The law refuter (`verdicts/trees_law.md`)

| # | Verdict | What I did |
|---|---|---|
| F1 | WRONG — the tree record does not fit domain 04's record | **Fixed.** §1.2 is rewritten. Domain 04's record has no state byte (`04_block_record_registry.md:112-136`). The growth stage and the plant tick move to the per-cell side table; the record carries kind, variant, provenance. §4.2 follows for a rock's size. The storage arithmetic of §2.2 is unchanged, because a FELLED tree has no side row; §1.2 and §8 item 6 now cost the side row at an ESTIMATED 16 B while a tree grows. §10 and §12 restate the door and the request. |
| F2 | BREAKS_LAW (SL8) + WRONG — the chop cannot be atomic on the lanes that exist | **Fixed.** §3.4 became §3.5 and now BUILDS the mechanism: the object diff carries `apply_tick`, the client queues it and applies it when the render cursor reaches that tick, so the reliable lane's head start becomes the schedule. The two lanes and the buffer are cited. New measurement 15, new one-way door. |
| F3 | WRONG — a falling tree cannot both collide and land where the seed says | **Fixed.** New §3.3 chooses: the topple is SCRIPTED and reads only static shape, and its capsules are a ONE-WAY pusher that never changes the tree's own pose. The accepted cost — a phantom miss against immovable drawn things — is stated. The misreading of `block_provenance_collapse.md` line 503 is corrected in place. |
| F4 | BREAKS_LAW (SL8) — an unbounded phantom hit | **Fixed.** §1.3 states `PHANTOM_HIT_MAX_M`, proposed 0.10 m ESTIMATED. Measurement 3 now asserts a DISTANCE, not membership. §5.1 refuses an archetype table that cannot meet it. New one-way door. |
| F5 | WRONG — the collider comparison is not like for like; two capsule sizes | **Fixed.** §1.6 is rebuilt: one capsule size (28 B, ESTIMATED by arithmetic on the field widths), and the comparison is normalised per tree and per square metre. The gain is 44×–145×, not 35×–112×. §8 item 7 restated. |
| F6 | WRONG — two units for one length, inside a frozen door | **Fixed.** 1/1024 m everywhere; the millimetre wording is deleted from §1.3 and §1.4; §10's door names the unit. |
| F7 | WRONG + UNMEASURED_AS_FACT — a denied timestamp is used; the growth tick has no budget | **Fixed.** §2.3 is rewritten. A plant tick IS stored, named, and costed. The per-chunk stamp and the binomial are gone: the stage is a closed-form schedule the shard evaluates on demand, so no slow tick walks anything. New measurement 16 asserts zero per-tree work on an unobserved moon. §11 item 10 puts the change of R8's intent to the owner. |
| F8 | WRONG — a planted tree is not permanent under the report's own rule | **Fixed.** §2.4 retires the permanence claim. A replanted tree that matches the baseline prunes, and that is stated as correct behaviour. The narrower true clause — a `Placed` cell never equals a `Feature` baseline — survives. §11 item 1's reasoning is corrected. |
| F9 | BREAKS_LAW (SL6) — "None found" reads half of SL6 | **Fixed.** §12 is rewritten. Four wire requests, each with the data, the lane, why the receiver cannot compute it, and the cost of doing without. `CLAUDE.md:210-213` is quoted. |
| F10 | MISSING — nothing ties the derived forest to a LIVE realm | **Fixed.** §2.5 states the rule: a client draws a realm's derived objects only for a chunk whose object diff it holds, and only a live realm ships one. Measurement 9 gains the dormant-moon arm. |
| F11 | MISSING — the seed ruling is never applied to the object KIND | **Fixed.** §2.1 states the fence: a seed-drawn object kind may drop COMMON substance only; anything that pays is live state on the side row. §5.3 repeats it for themed kinds. |
| F12 | WRONG — the file count | **Fixed.** 26, MEASURED by `ls crates/core/src \| wc -l` on 2026-09-07. |
| F13 | WRONG — the nesting-fence citation | **Fixed.** §1.7 cites `circumscribed_extent` at `crates/core/src/geometry.rs:350-361` as the reach helper and `:4327-4331` as where the fence's pair is named. |
| §2 wording note (SL1) | "composed position" | **Fixed.** §5.2's seam field now reads "in the realm's own frame". |

### The feasibility refuter (`verdicts/trees_feasibility.md`)

| # | Verdict | What I did |
|---|---|---|
| R1 | "derive the whole visible disc" is impossible | **Fixed, and it is the biggest change.** §2.5 withdraws the every-rung and whole-disc rules and states a THREE-RUNG object ladder with an object horizon at an ESTIMATED 800 m and a canopy fold beyond it. The refuter's arithmetic is reproduced and extended from the code's own drawable floor (`crates/core/src/geometry.rs:1156-1173`). The honest cost — a 22-pixel oak at the crossover — is named in §7 and put to the owner as §11 item 11 with measurement 13. |
| R2 | the no-drift gate tests SHAPE and never PLACEMENT | **Fixed.** §1.5 has two arms now, and the placement arm is named. §0 lists the seven forbidden libm sites (plus one allowed `sqrt`) with file and line, MEASURED, and calls retiring them owed work for this domain. New measurement 11. |
| R3 | an object's dirty set cannot be its anchor cell alone | **Fixed on the contradiction, DISPUTED on the cost.** §4.3 rule 1 now says the SUPPORT set is the anchor cell and the one cell beneath its up face, and §3.1's third row matches. I dispute that this needs a new reverse index: the per-chunk object list is already keyed by the anchor cell (§1.7), so a mined cell asks that map about its six face neighbours — six lookups, no new structure, and exactly the shape SL9 asks for (`CLAUDE.md:221-222`, "a LOOKUP, never a scan"). The dispute is written into §4.3 item 2. Measurement 7 gains the lookup's p99. |
| R4 | the collider budget rests on two numbers that differ by 3.6× | **Fixed.** One capsule size, 28 B. §8 item 7 restated. New §11 item 9 puts the collision implementation to the owner: `rapier3d`/`parry3d` as an option never adopted here, or a hand-written capsule sweep on the workspace's existing `glam`. §0 restates that neither is in `Cargo.lock`, MEASURED. |
| R5 | the falling collider moves the divergence into time | **Fixed.** §3.3 states the ESTIMATED 2.4 m lead of the collider over the drawn crown, says neither option satisfies the divergence contract, and names the real choice. §7's jump row and §11 item 3 carry the cost. New measurement 14. |
| R6 | "no timestamp" and "the chunk's last-ticked stamp" | **Fixed** with F7, above. The binomial is gone; a plant tick is stored and costed. |
| R7 | the growth lane has no bound | **Fixed.** §2.3 states three lane rules and the ESTIMATED collapse from 4.97 million rows to 19 406. §7 gains a lane-flood row. New measurement 12. |
| R8 | "Integer only" is false of `crates/core/src/rng.rs` | **Fixed.** §0 now separates the integer core (`next_u64` at `:22`, `child_seed` at `:70`) from `next_f64` (`:31`) and `chance` (`:39`), cites the generator's float draws, and §1.4 BANS the two float methods inside `expand` with a lint. |
| R9 | the file count inside a MEASURED heading | **Fixed** with F12. |
| R10 | route (b) puts back the shape branch the code exists to prevent | **Fixed.** §5.2 quotes both code comments and states the cost. §11 item 2's recommendation stands and now carries the cost. |
| R11 | the seed forest has no bound fence | **Fixed.** §2.1 step 3 tests every SEED draw's skeleton against the realm's bound, deterministically on both hosts, and §1.7 says the fence runs on both paths. |
| R12 | "the kind's age curve" is a function of time | **Fixed.** §1.2 and §2.1 use the kind's BASELINE STAGE, a constant. The age-curve wording is deleted. |
| R13 | a falling group that leaves its realm is not designed; its continuity class is not built | **Fixed.** §3.2 states that `RealmAnchored` is unbuilt and cites `crates/core/src/entity_kind.rs:23`; §12 request 1 carries that cost. §3.4's new paragraph and §11 item 12 recommend clamping the topple to the realm's bound for v1. |

### What both refuters left standing, and I did not change

The core answer to V2.2: a tree is one record, one integer skeleton, capsules on the shard, a mesh on
the client. The refusal of the pattern recogniser (§4). The disc counts (129 and 19 408). The 230×
storage win of §2.2. The stale-claim list items 1–6 and 8–12. The SL1–SL5, SL7, SL9, HR3, HR4 and
movement-contract readings the law refuter tried to break and could not.
