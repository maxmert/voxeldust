# 01 — The grid family and the geometry seam

**Domain:** the format the owner freezes FIRST. Every other voxel decision presupposes it.
**Date:** 2026-09-07. **Status:** an investigation report, REVISED after two refutations
(`verdicts/grid_law.md`, `verdicts/grid_feasibility.md`). It becomes binding only when the owner
promotes it into `docs/design/`.
**Read against:** `CLAUDE.md` (HR1–HR6, SL1–SL10), `docs/design/owner_decisions_2026-09-07_voxels.md`
(SL10, V2.1–V2.9), the 2026-09-02 REACH ruling, the 2026-09-01 visibility ruling, the 2026-08-24
rulings, SL8 (a seam is a defect).
**Method:** the code is the only truth for what exists today. Every claim about the code cites a file
and a line. Every number says MEASURED (and how), DERIVED (and from what) or ESTIMATED.
**Revision log:** §11.

---

## 0. The answer in one page

1. **The grid family stays the cube-sphere for a planet and a moon, and the plain Cartesian grid for a
   hull, a station and an asteroid.** Re-validated against the new rulings (§1). One index space serves
   both (HR4). The eight cube corners and the twelve edge bands are the known price, and the price is
   paid on the planet only. A hull is never built on planet cells, so a player who builds a SHIP never
   meets a corner defect. A player who builds a BASE on a planet does meet one, and that refusal is a
   named seam the owner must rule on (§6.5, D9).

2. **The address of a cell is realm-local, integer and SIGNED:** `(body, face, tier, i, j, k)` with
   `i, j, k: i32`. The body is the `RealmId` the picture already names. The face is `0..=5` on a planet
   and always `0` on a hull. The tier is the detail rung. On a planet `i, j` count cells along the face
   and are never negative; `k` counts whole metres from the body's floor radius and is never negative.
   On a hull the origin is the hull's own frame origin, so all three run negative, and a signed field is
   what holds them. No parent frame is in the address. The server and the client compute the same
   integer from the same `LatticePos` with the same crate, so SL10's "identical on both hosts" holds by
   construction and is then MEASURED by the no-drift gate (§7). *Example: the shard of the home planet
   and the pilot's client both name the cell under the spaceport's landing pad as `(Planet 7, face +Y,
   tier 0, i 126 971, j 127 004, k 8 400)`. On the pilot's hull, the cell one metre aft of the hull's
   own origin is `(Ship 44, face 0, tier 0, i 0, j 0, k −1)`. Neither realm asked its parent where it
   is.*

3. **The radius ladder is re-derived.** The investigation's 39.47 m step came from forcing WHOLE CHUNKS
   along a face edge. The radial axis already tolerates a partial chunk at the band's top, so the
   tangential axes can too. The cell count per face edge `N` is then any multiple of `2^(T−1)`, where `T`
   is the body's rung count. A body with 13 rungs (Earth-sized) lands within ±0.65 km of its seed-drawn
   radius (0.01 %); a 500 m asteroid with 1 rung lands within ±0.3 m. Today's planet radius is a
   continuous number from the mass–radius law and is on no ladder at all
   (`crates/physics/src/worldgen/generate.rs:1232`, `crates/physics/src/taxonomy.rs:653`); snapping it is
   a world-generation-tag change and is free before the first chunk is saved.

4. **The sub-metre lattice is the fine lattice's own high bits.** The position lattice already counts in
   `2⁻¹⁰ m` (`crates/core/src/pose.rs:255`). A sub-block at `1/2ˢ m` is addressed by the `s` bits below
   the whole metre: `cell = fine_cell >> 10`, `sub = (fine_cell >> (10 − s)) & (2ˢ − 1)`. No division,
   no float. **Recommended finest step: 1/8 m (s = 3), with the address reserving s ≤ 4 (1/16 m).** A
   1 m cell is EITHER one block OR one sub-grid; space-taking and collision stay at the 1 m cell (§3).
   The one-way door is the address width, not the shipped step. On a planet cell a micro-cell inherits
   the cell's own warp anisotropy, so a 1/8 m part measures 0.0884–0.1256 m depending on where it stands
   (MEASURED, §3.6). Whether that is acceptable is an owner decision (D11).

5. **The geometry seam is `GridMapping`, a stateless value in `vd-core`, with two arms** (`Identity`,
   `CubeSphere`) and six operations. Nothing above it may see the arm. The stateful `FrameSpace` of
   `docs/design/PLAN.md:42` and `docs/design/sealed_shards.md:219-234` (a tangent anchor, `reanchor()`,
   `AnchorGen`) exists for a physics library that wants `f32`; that library is not a dependency today
   (§4). **HR4's own wording names `FrameSpace`, so renaming the seam is an owner decision (D8), not a
   recommendation this report may take by argument.** The smallest landing slice is one module, one
   generated seam table with its pinned literal, one exhaustive test at `N = 62`, one `N`-dependent
   round-trip test at the largest legal `N`, and one `build_app` wiring line.

6. **The warp inverse is FOUR Newton steps from the first guess `a₀ = t`.** Three steps do not reach the
   `f64` floor and their residual depends on the first guess: `a₀ = t` leaves 7.2 × 10⁻⁴ cells at the
   largest legal body, and the obvious alternative `a₀ = t/k₁` leaves 0.35 cells — a margin of only 1.4
   against half a cell (MEASURED, §1.3). Four steps reach the `f64` floor from either guess. The first
   guess AND the step count are both part of the frozen warp.

7. **Re-centering math is already in the tree.** Every position normalises inside `translated`
   (`crates/core/src/pose.rs:623-630`), the one subtraction is integer-exact (`pose.rs:709-715`), and the
   FINE lattice spans ±0.476 ly (DERIVED from `pose.rs:455`). An Earth-sized body is `2^32.6` fine cells
   across. There is no planet-scale limit from `i64`. The only float in the cell address is the
   tangential `floor` after the inverse warp, and it is quantised once, in one function (§5).

---

## 1. The grid family, re-validated

### 1.1 What the rulings demand of a grid

| Requirement | Source | What it demands of the grid |
|---|---|---|
| The static shape is `f(seed, address)`; both hosts compute it | SL10 V1.1–V1.3 | The address must be an INTEGER the client computes without any parent frame |
| No `sin/exp/pow`; `+ − × ÷ √` only | SL10 V1.4 | The index→position map must be polynomial plus square root |
| The server collides on the same shape | SL10 V1.5 | The collider comes from the same cells the client draws |
| Smooth, realistic terrain, still built of voxels; mining and placing reshape the surface | V2.1 | A rectilinear cell lattice, AND a per-cell surface datum the extractor can read (§3.1) |
| A tree is ONE object on the surface; the server knows its shape | V2.2 | A landscape-object arm anchored at ONE cell whose collision volume may leave that cell (§3.1) |
| Building blocks stay square, ~20 shapes | V2.3 | Rectilinear cells everywhere a player builds |
| Sub-metre blocks fit inside the 1 m cell | V2.4 | A cell that subdivides by powers of two |
| Attachments that take no space, INCLUDING a rotation joint | V2.5 | A cell plus a face as an address, AND an answer for geometry that turns off-grid (§3.5) |
| Block parameters and client-side style | V2.6 | Nothing from the grid; the address carries no material (the record report owns it) |
| Renderer-agnostic | V2.8 | The seam is data and pure functions, not an engine type |
| A realm is told where it is; one hop | SL1 | The address must not name any position outside the realm, and must not carry a told placement onto a lane |
| A realm draws itself | SL3 | The realm's own grid parameters are the realm's own statement |
| A parent's child count is unbounded; lookups not scans | SL9 | The address→chunk map must be O(1) |
| One radius per realm, tested by its parent | REACH R3 | The grid must not change the bound; the bound is the SOI shell today (`generate.rs:1219`) |
| A seam is a defect | SL8 | No visible line between two charts; no gap at a corner; no unexplained refusal |
| One world; never a reduced variant | SL5 | One generator, one grid, the same on every body |

### 1.2 The families, re-scored against those rows

The investigation compared four families (`block_system_design_addendum_1.md` §A.2). One more must be
named now, because a realm kind for it exists in the code.

| Family | SL10 address | V2.3 square blocks | HR4 one grid with a hull | SL8 seams | Verdict |
|---|---|---|---|---|---|
| **Cube-sphere with a warp** | integer `(face, i, j, k)`; the map is a polynomial and one `sqrt` | yes, everywhere except 8 corners (rhombi) | yes: same index space, `face = 0` | none in the world; a build REFUSAL at eight corners (D9) | **keep** |
| Literal cube planet | integer, trivial | yes, exactly | yes | a cube from orbit; "down" tilts 54.7° at a corner | refused (look and "down") |
| Hex / Goldberg | integer, but a second lattice | no | **no** — a second mesher, a second catalogue | 12 pentagons | refused (HR4 dies on day one) |
| Flat octree with an analytic surface | integer | yes | yes | a wall at latitude 40° is a staircase | refused (building) |
| **NEW: a planet as a set of flat tangent-plane Area realms** (`RealmId::Area`, `crates/core/src/pose.rs:47`; `FrameRef::AreaLocal`, `pose.rs:105`) | integer per area | yes inside an area | yes | **a visible seam at every area border** — two flat grids do not line up on a sphere; terrain between areas has no grid | refused (SL8) |

**The cube-sphere stands.** The price, stated once more so the owner signs with it in view:

- Tangential cell edge `0.7071–1.0050 m` under the recommended warp. MEASURED 2026-09-07: 401 × 401
  samples per face of the arc-length derivative of `normalize(1, W(a), W(b))`, scaled by `4/π`. The
  minimum sits at the middle of a cube edge, the maximum near a corner.
- The angle between the two grid directions is exactly 90° at a face centre and departs from 90° by up
  to 30° at a cube corner (MEASURED: the sampled range is 60.0000°–120.0000°, the same angle read from
  its two orientations). This departure is a consequence of the cube net, not of the warp: three squares
  meet at a cube corner, so a four-fold pattern cannot close there. No warp removes it.
- Eight corners per body, at latitude ±35.2644° (DERIVED: `asin(1/√3)`).
- A box prefab cannot be stamped across a corner. A blueprint is a relative walk from an origin cell,
  and the walk returns a typed `CornerDefect` instead of overwriting a cell. **That refusal is a seam
  under SL8 ("tier refusal"), it is not cured by this report, and it is D9.**

*Example: a player builds a wall of stone blocks along the equator of the home planet. Each block is a
square 1.000 m at the face centre and a 0.707 × 0.992 m rectangle where the wall crosses the middle of a
cube edge, 40 km later (MEASURED at `a = 0, b = −1`). The player cannot see the change; a survey tool
could measure it. If instead the wall runs toward latitude 35.26°, it reaches a cube corner and the
stamp refuses — which is the seam D9 must answer. A hull built in the shipyard next to that wall is a
separate Cartesian realm, and every one of its blocks is exactly 1 m.*

### 1.3 The warp, re-validated under SL10

The investigation recommends `VD-ASC5`: `W(a) = k₁a + k₂a³ + k₃a⁵` with `k₁ = π/4`, `k₂ = 0.15`,
`k₃ = 1 − k₁ − k₂` (`block_system_design.md` §3.3.2). Under SL10 V1.4 this is the right shape: the
forward map is five multiplies and two adds; the normalisation is one `sqrt` and one divide; the inverse
is `+ − × ÷` with a `const` step count. No `libm` call anywhere. `π/4` is a literal `f64`, never
`std::f64::consts::FRAC_PI_4` read through a function.

**The three constants sum to exactly 1.0.** MEASURED 2026-09-07 in IEEE `f64` (python3, aarch64 darwin):
`(k₁ + k₂) + k₃ − 1.0` is `0.0`, and `k₁ + (k₂ + k₃) − 1.0` is `0.0`. Both orders are exact, so
`W(1) = 1` holds and the face edge lands on the cube edge. This is a measurement in another language on
one target; **the `const` assertion inside the grid crate, run on x86-64 AND aarch64, is still owed
(§7 item 9).**

**The inverse: FOUR steps from `a₀ = t`.** The investigation said "a fixed three-step Newton inverse"
and never stated the first guess. The first guess decides the answer. MEASURED 2026-09-07, 200 001
samples of `a` in `[−1, 1]`, forward `W` then inverse then error, in IEEE `f64`; a cell spans `2/N` in
`a`, so "cells" is `|Δa|·N/2`:

| first guess | steps | worst `|Δa|` | cells at Earth (`N = 1.0007 × 10⁷`) | cells at the largest legal body (`N = 2²⁶ = 6.711 × 10⁷`) |
|---|---|---|---|---|
| `a₀ = t` | 3 | 2.14 × 10⁻¹¹ | 1.07 × 10⁻⁴ | 7.18 × 10⁻⁴ |
| **`a₀ = t`** | **4** | **2.22 × 10⁻¹⁶** | **1.1 × 10⁻⁹** | **7.5 × 10⁻⁹** |
| `a₀ = t/k₁`, clamped | 3 | 1.04 × 10⁻⁸ | 0.052 | **0.348** |
| `a₀ = t/k₁`, clamped | 4 | 2.22 × 10⁻¹⁶ | 1.1 × 10⁻⁹ | 7.5 × 10⁻⁹ |

Three steps from `a₀ = t/k₁` leave a third of a cell at the largest body the address admits. A cell
centre is half a cell from its boundary, so that arm survives by a factor of 1.4 and nothing else.
**Recommendation: freeze `a₀ = t` AND four steps.** Four steps reach the `f64` floor from either guess,
so the choice of guess stops being load-bearing, and the margin argument disappears instead of being
carried. The extra step costs about nine multiplies, four adds, one divide and one subtract per
tangential axis per `addr_of` call (ESTIMATED from the expression), and `addr_of` is not on the
containment path (§5.2).

*Example: a pilot lands on the home planet and mines a cell. The shard runs the inverse warp on the
pilot's own position and names one cell. A client built with a different first guess or a different step
count could name the neighbouring cell, and the world-generation tag would not refuse that client,
because the crate version is the same and only the arithmetic inside one function differs. That is why
the guess and the count are both inside the frozen warp and inside the version tag (§6.3).*

The angle-proportional constraint `W'(0) = π/4` is what makes "1.000 m at the face centre" exact
(MEASURED: the arc formula returns 1.000000 at `a = b = 0`) and the radius arithmetic below closed-form.
It is kept.

### 1.4 The address

```text
CellAddr {
  body:  RealmId      // the realm whose grid this is: Planet(seed) | Ship(id) | Station(seed) | ...
  face:  u8           // 0..=5 on a cube-sphere; 0 on an identity grid
  tier:  u8           // detail rung L: a cell is 2^L m; tier 0 is the only writable rung
  i, j:  i32          // tangential cell index at this tier
  k:     i32          // radial index at this tier
}
```

- **The index fields are SIGNED.** On a cube-sphere `0 <= i, j < N / 2^L` and `k >= 0`, because `k`
  counts from `floor_radius_m`; the sign is never used. On an identity grid the origin IS the hull's own
  frame origin (`FrameRef::ShipLocal`, `crates/core/src/pose.rs:91`), and a hull's slot box is centred
  there, so every cell aft of or below the origin has a negative index. An unsigned field would file a
  thruster placed one metre aft of the origin at `4 294 967 295` — the far bow — or refuse the placement.
  A signed field is one bit and it removes the fault. *Example: an engineer walks to the stern of a hull
  and places a thruster one metre aft of the hull's own origin. The address is `k = −1`, and the block
  lands where the engineer stood.*
- **A bias was the alternative, and it is worse.** Biasing the identity arm by `half_cells` would make
  the indices non-negative, but then no reader can decode an address without knowing `half_cells` — and
  `half_cells` comes from the built realm's `bound` (`crates/core/src/built.rs:76`), which does NOT
  reach the client today (§6.1). The signed address keeps the decode local and dissolves an SL6 request.
- **Identical on both hosts.** The client reads `body` from the row it draws (`SceneRow.realm`,
  `crates/wire/src/channels.rs:378-379`). `face, i, j, k` come from the same `GridMapping` function on
  the same lattice input. No frame conversion exists between the address and the realm's own
  `LatticePos`, because the cube-sphere's origin IS the `PlanetCentered` frame's origin (`pose.rs:89`),
  and the identity grid's origin IS the `ShipLocal` / `StationLocal` origin (`pose.rs:91,102`).
- **Stable forever** as long as six things do not move: the face basis table, the warp constants, the
  inverse's FIRST GUESS and step count, `N` for the body, `floor_radius_m` for the body, and the fine
  step `2⁻¹⁰ m`. The last is already fenced by `coordinate_generation`
  (`crates/core/src/store_stamp.rs:118,256-261`); the others join the generator crate's version tag
  (SL10 V1.3).
- **Realm-local (SL1).** A planet is centred on itself; `k` counts up from ITS floor radius; `face`
  points along ITS axes. The star system's placement of the planet is nowhere in the address. A cell
  never learns where its planet is. *Example: the star system moves the planet along its orbit every
  tick. The landing pad's cell address does not change by one bit.*
- **Spin.** A planet's cells co-rotate with the body, so the `PlanetCentered` frame is body-fixed. The
  star system authors the planet's placement AND orientation per tick (the stamped `StampedPose` carries
  `orient`, `crates/core/src/pose.rs:919,923`), exactly as it does today. The grid adds no new statement.

**The chunk is a packing, not part of the address.** A chunk key groups `62³` cells for storage and
streaming (`docs/design/PLAN.md:42` names `62³` as the shared addressing above the seam). The cell
address does not depend on the chunk edge; a chunk key is derived from it by an integer divide. Whether
`62` survives is the saved-record report's door, not this one. This report only states the bit budget a
packing must carry (§5.3).

### 1.5 The seam table and the apron

Twelve directed edge pairings, 24 records of `(dst_face, dst_edge, reverse)`, generated from the face
basis table and pinned by a committed literal; eight corner columns; 24 corner-diagonal apron slots
filled with the nearest in-face cell (`block_system_design.md` §3.3.3).

**The table depends on the cube net only** — not on `N`, not on the tier — so an exhaustive test at
`N = 62` (1,488 seam cases + 24 corners + 24 slots) proves THE TABLE for every body. That claim covers
the table and nothing else. The round trip `addr_of(cell_center(x)) == x` is `N`-DEPENDENT and a run at
`N = 62` cannot see a coarse inverse: a cell there spans 0.032 in `a`, so the worst three-step residual
of 1.04 × 10⁻⁸ is 3 × 10⁻⁷ of a cell and the test passes by a factor of 1.5 million (MEASURED: the
residual above divided by the cell width). The gate therefore splits by `N`-dependence, not by
convenience (§4.3).

*Example: the mesher gathers the apron for the chunk at the north-east corner of face `+X`. Two of its
apron strips come from face `+Y` and face `+Z` through the seam table with an axis swap; the diagonal
slot where a fourth face would be does not exist, and it is filled with the corner cell of `+X` itself,
so no triangle is emitted into a hole.*

---

## 2. The radius ladder, re-derived

### 2.1 What the code does today

- A planet's radius is `taxon.radius_m`, a continuous `f64` from the Chen–Kipping mass–radius law
  (`crates/physics/src/taxonomy.rs:653`), and the planet DRAWS itself at that radius
  (`crates/physics/src/worldgen/generate.rs:1232`, `look: Some(shell(taxon.radius_m))`). A moon does
  the same (`generate.rs:1819`).
- A planet's BOUND is its sphere of influence, not its surface (`generate.rs:1219`, `shape: shell(soi_m)`).
  The grid never touches the bound. REACH R3 and SL9 are untouched by anything in this report.
- No `ChunkKey`, no `BlockAddr`, no `GridMapping` and no `FrameSpace` exist in any crate. A grep over
  `crates/**/*.rs` for `FrameSpace|SphericalSpace|CartesianSpace|SurfaceAnchor|AnchorGen|reanchor`
  returns comments and one unrelated test name, among them `crates/sim/src/capability.rs:11,18,35,40`,
  `crates/core/src/fence.rs:18`, `crates/sim/src/stub/transient.rs:239` and
  `crates/node/src/saga_runtime/tests.rs:3513` (which is a re-home budget test, not this seam). No type
  and no module. `rapier` is absent from `Cargo.lock` (MEASURED: `grep -c rapier Cargo.lock` returns 0).
  `noise = "=0.9.0"` is declared in the workspace (`Cargo.toml:78`) and used by no crate (MEASURED:
  `grep -rn "noise::" crates/` returns nothing).

### 2.2 Why the 39.47 m step is an artefact

The investigation fixed `N = 62·m` cells per face edge so that whole chunks tile the face, giving
`R = 124·m/π` and a 39.47 m step (`block_system_design_addendum_1.md` §C.2). But the radial axis already
has a partial chunk at the top of the band: `V = ceil((D_crust + H_scale) / 62)` (`block_system_design.md`
§3.3.5). A partial chunk is a chunk whose last cells are `Outside` the domain — the same arm every
edit-admission test already needs. Nothing in the seam table or the apron cares: seams are cell-to-cell,
and a partial edge chunk gathers its apron across the seam exactly as a full one does.

**What must still hold** is that tier-L CELLS tile the face exactly, or a partial COARSE CELL sits on the
seam (the investigation's option C, refused for good reason). That needs `2^L | N` for every rung the
body carries: **`N` is a multiple of `2^(T−1)` for a `T`-rung body.**

So the ladder is: `R = 2·N/π` metres, `N = q·2^(T−1)` for any integer `q`. The face-centre cell is
exactly `1.000 m` (from `W'(0) = π/4`). The step is `2^T/π` metres — 0.64 m for a one-rung body, 2.6 km
for a thirteen-rung body.

### 2.3 How many rungs a body carries

The investigation derived the rung count from the face-covering rung (`tier_depth = ceil(log2 m) + 1`,
`block_system_design.md` §3.4), which forces `m` to be a power of two and therefore forbids a realistic
radius (Earth would land at 5,341 km or 10,683 km). That rider collides with the owner's own requirement
of realistic, differing sizes (`addendum_1` header, point (c)). It is refused here.

**Recommended rule:** the top rung is the one at which a face is at most `64 × 64` chunks, i.e.
`T − 1 = ceil(log2(N / (62·64)))`, clamped to `[0, 15]`. Beyond the top rung the realm draws its own
look (SL3: the realm authors how it looks) — the same smooth sphere-plus-height it draws as a dot at its
reach. This is per-body DATA, derived from the radius, never authored.

The table prints BOTH `N` values, because the ladder rule snaps `N` and the previous revision printed
only the un-snapped one under a snapped heading:

| body | seed radius | `N` ideal (`πR/2`) | `N` snapped | rungs `T` | top cell | snapped radius | error | chunks per face edge at top |
|---|---|---|---|---|---|---|---|---|
| asteroid | 500 m | 785 | 785 | 1 | 1 m | 500 m | −0.3 m | 12.7 |
| moon | 200 km | 314,159 | 314,112 | 8 | 128 m | 199,970 m | −30.1 m | 39.6 |
| Ceres | 473 km | 742,987 | 742,912 | 9 | 256 m | 472,952 m | −47.5 m | 46.8 |
| Luna | 1,737 km | 2,728,473 | 2,728,960 | 11 | 1,024 m | 1,737,310 m | +309.9 m | 43.0 |
| Mars | 3,390 km | 5,325,000 | 5,324,800 | 12 | 2,048 m | 3,389,873 m | −127.0 m | 41.9 |
| Earth | 6,371 km | 10,007,543 | 10,006,528 | 13 | 4,096 m | 6,370,354 m | −646.4 m | 39.4 |
| super-Earth | 12,742 km | 20,015,087 | 20,013,056 | 14 | 8,192 m | 12,740,707 m | −1,292.8 m | 39.4 |

MEASURED 2026-09-07 by a script that applies the rule above (nearest-multiple snap) to each seed radius;
it is a measurement of the RULE, not of the crate, because the crate does not exist. The worst error is
0.01 % of the radius. The look shell moves by that amount on THE world when the ladder lands, which is a
`world_generation` change (`crates/core/src/store_stamp.rs:122`) and is free before the first chunk exists.

**The rule sets a draw budget this report does not own.** At the top rung an Earth-sized planet has 39.4
chunks per face edge, so `6 × 39.4² ≈ 9,300` top-rung chunks, about 4,650 in the visible hemisphere, each
`62 × 4,096 m ≈ 254 km` on a side (all ESTIMATED from the table). Nothing here says on which thread a
chunk is built, how many a landing at flight speed needs per second, or what a planet at 10,000 km in the
window costs. That budget belongs to the render report (`08_render_seam_seamless.md`) and the terrain
report (`02_smooth_terrain.md`); the dependency is recorded in §7 item 12 and in D6's row.

*Example: the home planet's seed draws a radius of 6,371,000 m. The generator snaps `N` from 10,007,543
to 10,006,528 (a multiple of 4,096) and the planet states its look at 6,370,354 m. A pilot who watches
the planet grow from a dot sees the planet's own look; when the first chunks arrive at rung 12, they sit
exactly on that same sphere, so nothing pops (SL8).*

### 2.4 The radial band

`k` counts whole metres from `floor_radius_m = R_snapped − D_crust` (an integer number of metres) up to
`ceiling_m = R_snapped + H_scale`, both derived from the body's mass (`addendum_1` §D.2; the derivation
is the body-physics report's business). A radial cell is exactly `2^L` m tall at every altitude and every
tier, so `k` is an integer subtraction and a shift — no float on the radial axis (`block_system_design.md`
§3.1.3 holds). A chunk is a frustum, not a box, and only index-space code ever knows it.

---

## 3. The cell's state and the sub-metre lattice

### 3.1 What a cell may hold — and the scope of this report

**This report fixes the PLACEABLE state of a cell only.** The previous revision wrote "Nothing else is
placeable" and both refuters read it as a closed list of everything a cell may hold. It is not. Two of
the owner's requirements need a cell arm that this report does not design, and shutting the address door
without naming them would shut it ON them:

| arm | what it holds | who designs it | does it change §1.4's address? |
|---|---|---|---|
| `Empty` | nothing | here | no |
| `Block(shape, orient)` | one of the ~20 square shapes (V2.3) | the saved-record report (04) | no |
| `SubGrid` | a 1/2ˢ m micro-lattice inside the cell (V2.4) | here, §3.2–3.4 | no |
| **`Terrain(surface datum)`** | the per-cell scalar a smooth-surface extractor reads, so mining sags a slope instead of removing a cube (V2.1) | the smooth-terrain report (02); the record width is the saved-record report's door | **no** |
| **`Landscape(object)`** | ONE seed-parameterised object anchored at this cell whose volume may leave it — a tree (V2.2) | the trees report (07); the record is the saved-record report's | **no** |

The address in §1.4 is unchanged by either new arm, so §8's one-way door stays a door about the ADDRESS
and about the sub-cell width, not about the cell's payload.

**What this report DOES rule on the two new arms**, because they are geometry:

- A `Terrain` cell keeps the 1 m cell as the unit of MINING and of chunk membership. The surface datum
  moves the drawn and collided surface inside the cell; it never moves the cell.
- A `Landscape` object is anchored at ONE cell and its collision volume MAY leave that cell. This is the
  single exception to "a cell's geometry stays inside the cell", and it exists because V2.2 demands it.
  The shard expands the object's shape from its seed parameters for the sweep test; the client expands
  the same shape from the same crate for drawing (SL10). *Example: a pilot flies a hull through the
  canopy of a pine on a moon. The moon's shard holds one `Landscape` cell at the trunk, expands the
  pine's canopy from its seed parameters, and the hull's sweep test hits the canopy — not twenty empty
  cells.*
- The case where the two meet: a tree on a MINED edge. The trunk's cell is dug out by a diff while the
  trunk's shape still comes from the seed. The owning realm's admission rule must delete or fell the
  object when its anchor cell is emptied. That rule is the trees report's, and the dependency is recorded
  in §7 item 13.

### 3.2 The owner's words on sub-metre blocks, and what they fix

*"Smaller blocks than 1 m … those smaller blocks should fit into 1 m space of the bigger blocks to
simplify collision and space taking."* (V2.4). Three consequences are fixed by that sentence:

1. A sub-block never crosses a 1 m cell boundary.
2. The unit of SPACE-TAKING stays the 1 m cell. A full block goes only into an `Empty` cell. A sub-block
   goes into an `Empty` cell (which becomes a `SubGrid`) or into a free micro-slot of a `SubGrid` cell.
3. Everything that reasons at 1 m — the seam table, the apron, the corner defect, the flood fills, the
   edit pyramid, the AoI grids — never learns that sub-blocks exist.

### 3.3 The finest step: options and the recommendation

The position lattice already counts in `2⁻¹⁰ m` (`crates/core/src/pose.rs:255`, `FINE_CELL_EDGE_M =
1.0 / 1024.0`), chosen a power of two so `normalize` is exactly idempotent. A sub-lattice at `1/2ˢ m`
is therefore the fine lattice's own high bits:

```text
whole-metre cell axis  =  fine_cell >> 10
sub-cell axis          = (fine_cell >> (10 − s)) & ((1 << s) − 1)
```

No division, no float, and the sub-cell of a `LatticePos` on an identity grid is bit-deterministic on
every host. On a cube-sphere the same shift runs in INDEX space after `addr_of` (`i + u/2ˢ`), so the
warp reaches the micro-boxes exactly as it reaches the cell (the "index-space contract",
`block_system_design.md` §4.2).

| step `s` | size | micro-cells per cell | address bits (3·s) | what it resolves | mesh worst case per cell (ESTIMATED) |
|---|---|---|---|---|---|
| 1 | 1/2 m | 8 | 3 | a half-step, a thick console | 8 boxes |
| 2 | 1/4 m | 64 | 6 | a railing post, a window frame | 64 boxes, ~12 quads after merging |
| **3** | **1/8 m** | **512** | **9** | a rail, a keyboard, a pipe, a stair tread | 512 boxes, ~30 quads after merging |
| 4 | 1/16 m | 4,096 | 12 | a switch, a cable | 4,096 boxes, ~100 quads |

**Recommendation: ship `s = 3` (1/8 m) as the finest placeable step; reserve `s ≤ 4` in the address.**
1/8 m resolves every "little detail" the owner named (joints, HUDs, rails) at one eighth of the cost of
1/16 m in micro-cells, and the ~20 shapes (a slope, a corner, a half) scaled to 1/8 m give the finer
silhouette a 1/16 m cube would give. 1/16 m stays reachable by widening nothing: the level field is
4 bits and the index is 12 bits from the first record.

**The one-way door is the address width** (level 4 bits + index 3×4 bits = 16 bits per sub-cell), not
the shipped step. The saved-record report freezes the bits; this report states the requirement.

### 3.4 How collision stays at the 1 m cell

- The realm's shard builds the collider for a `SubGrid` cell from the union of its micro-boxes, merged
  by the same greedy pass the 1 m mesher uses, at tier 0 only (collision never coarsens —
  `addendum_2` §C.7). The result never leaves the cell.
- The character controller and the sweep test read that collider through the same path as a full block's
  collider. No branch on "is this a sub-grid".
- A `SubGrid` cell is never seed-generated. It is always a DIFF from the owning realm (SL10 V1.6 names
  "sub-metre blocks" as diff data), so the client applies it over the derived shape and the server
  collides on the same union.

*Example: an engineer builds a railing on the bridge of a hull with 1/8 m posts and a 1/8 × 1/8 m rail.
The four cells the railing passes through become `SubGrid` cells; a crewmate cannot place a full crate
into any of them, but can walk past and lean on the rail, because the shard's collider is the rail's own
boxes, not the four whole cells.*

### 3.5 A sub-grid on a PLANET cell — the rule the previous revision left out

A planet's cell is a frustum, not a box. A micro-cell is the cell's own eighth in INDEX space, so it
inherits the cell's shape exactly and "it fits inside the 1 m cell" stays true everywhere. What changes
across the planet is the micro-cell's ABSOLUTE size:

| where | 1 m cell (MEASURED §1.2) | 1/8 m micro-cell (DERIVED: ÷8) |
|---|---|---|
| face centre | 1.0000 × 1.0000 m | 0.1250 × 0.1250 m |
| middle of a cube edge | 0.7071 × 0.9921 m | 0.0884 × 0.1240 m |
| near a cube corner | 0.9354 × 0.9354 m at 120° | 0.1169 × 0.1169 m at 120° |

At 1 m the report's own answer holds: a player cannot see the difference, a survey tool can. At 1/8 m a
0.0884 m rail beside a 0.1250 m rail is a 3.7 cm difference on a part held at arm's length, which a
player CAN see. Two lawful answers, and the owner picks (D11):

- **(A) planet-legal (recommended).** A sub-metre block may go in any cell on any body. The anisotropy is
  the same anisotropy the 1 m block already carries, and refusing sub-metre blocks on a planet would give
  a base less detail than a hull — a "detail-by-box" seam under SL8, and a rule the player would meet as
  "my workshop looks worse on the ground than in orbit".
- **(B) identity-grid only.** Sub-metre blocks on hulls, stations and asteroids only. This removes the
  visible anisotropy and removes detail from every planet base.

*Example: a colonist builds a control desk in a surface habitat at latitude 35°, then builds the same
desk in the hull parked outside. Under (A) the two desks differ by a few centimetres across and the
colonist may notice. Under (B) the desk cannot be built on the ground at all.*

### 3.6 Attachments that take no space (V2.5), and the rotation joint

An attachment that has no volume — a HUD, a lamp, a sensor, a rail clamp — is addressed by
`(CellAddr, face: u8)`. It never changes the cell's arm. That much is settled here; the attachment
record is the saved-record report's business.

**A rotation joint is not that, and it is a grid question.** V2.5 asks for *"a rotation joint between two
blocks that turns what is built on it by a signal (a manipulator, a remote-controlled turret)"*. What is
built on the joint is a set of CELLS at a continuous angle to the hull's grid, and `(body, face, tier,
i, j, k)` cannot express a cell at 37°. Three answers, and the owner picks (D10):

- **(A) a MOUNT: a second identity grid inside the same realm (recommended).** The joint carries its own
  `IdentityGrid` with its own origin and its own orientation inside the hull's realm. Every block on the
  turret is addressed on the MOUNT's grid, at integer indices, and the mount's orientation is authored by
  the hull each tick from the signal. This is the SAME arm, instantiated twice — it forks no code, so
  HR4 holds. The cost is real and must be stated: the hull's collider now holds two rigid bodies that
  can touch each other, and the mesher must draw a chunk that straddles two grids.
- **(B) a child REALM.** The turret is its own built realm and the hull authors its orientation, exactly
  as a star system authors a planet's (`StampedPose.orient`, `crates/core/src/pose.rs:923`). Lawful with
  no new machinery, but a hull with forty joints becomes forty realms; SL9 permits it and the spin-up
  cost does not.
- **(C) off-grid rigid geometry.** The turret is not built of cells at all. This contradicts V2.3 and
  V2.5's own words ("what is built on it").

*Example: a gunner turns a turret 37° on a hull's dorsal mount. Under (A) the four armour blocks stay at
integer indices on the mount's own grid, and the hull authors the mount's orientation from the fire-control
signal. Under (C) the gunner could not have built the turret from blocks in the first place.*

### 3.7 The cost in record bits and in mesh work

- **Record bits.** A full-block cell keeps the record the saved-record report freezes. A `SubGrid` cell
  needs one extra record per occupied micro-cell: `sub address (16 bits reserved, 9 used) + the same
  identity/orientation fields as a block`. A dense 1/8 m cell is at most 512 such records (≈4 KiB at
  8 bytes each) — ESTIMATED. A per-chunk cap on `SubGrid` cells is an abuse bound and a tuning field, not
  a literal.
- **Mesh work.** A `SubGrid` cell costs roughly what an `8³` chunk of full blocks costs
  (ESTIMATED: the same greedy pass over 512 cells). Culling between a sub-grid and its 1 m neighbours
  uses the six face masks at the cell's boundary: a micro-box touching the cell's face contributes to
  that face's mask, so a full block next to a sub-grid still culls its hidden face.

---

## 4. The geometry seam

### 4.1 What exists

- `VoxelGeometry::{Spherical, Cartesian}` as a capability VALUE on the profile
  (`crates/sim/src/capability.rs:37-42`; the doc comment above it is `capability.rs:34-36`), with an
  exact-match rule so a Cartesian realm never re-homes onto a Spherical shard.
- The profiles as data: `planet` is `Spherical`; `ship`, `station`, `asteroid` are `Cartesian`
  (`capability.rs`, `mod profiles`; the same list is written out in `docs/design/sealed_shards.md:213`).
- The HR4 harness `assert_feature_anywhere` runs one crossing fixture on a planet (Shell) and a station
  (Aabb) profile (`docs/design/DEFERRED.md:299-355`, D-38). The `reanchor()` variant is owed at P5
  (`DEFERRED.md:331-334`).
- `FrameSpace`, `SphericalSpace`, `CartesianSpace`, `SurfaceAnchor`, `AnchorGen` do not exist in code
  (§2.1). `rapier` is not in `Cargo.lock`. `noise = "=0.9.0"` is declared (`Cargo.toml:78`) and unused.

### 4.2 The seam, stated — and the rename the owner must approve

Two halves, deliberately separate, as `block_system_design.md` §3.1.2 already argued:

**Half one — `GridMapping`, stateless, pure, in `vd-core` (Tier-A, no I/O, no `libm`):**

```text
enum GridMapping { Identity(IdentityGrid), CubeSphere(ShellGrid) }

IdentityGrid { half_cells: IVec3 }                       // the hull's slot box, in whole cells
ShellGrid    { n: u32, floor_r_m: u32, ceiling_r_m: u32, rungs: u8 }   // per-body, seed-derived

addr_of      (&self, local: LatticePos, tier: Tier) -> Option<CellAddr>
cell_center  (&self, CellAddr) -> LatticePos
cell_corners (&self, CellAddr) -> [LatticePos; 8]
neighbor     (&self, CellAddr, Dir6) -> Same(CellAddr) | AcrossSeam { addr, axis_remap } | Outside
up           (&self, CellAddr) -> DVec3          // the one "down" seam: constant on a hull, radial on a planet
domain       (&self) -> GridDomain
```

`Identity` never returns `AcrossSeam`. `CubeSphere` returns it only at the twelve face edges. The two
arms are the ONLY `match` on geometry in the workspace; each arm delegates to a monomorphic
`identity::*` or `shell::*` function (the `core/src/tlv.rs` HR5 shape). Matching on a geometry VALUE the
realm carries is what HR3's no-shard-fork lint exists to force (`crates/sim/src/capability.rs:9-11`);
matching on a shard KIND stays forbidden.

**★ THIS RENAMES A HARD RULE'S OWN WORDING, AND THAT IS AN OWNER DECISION (D8).** `CLAUDE.md:110` states
HR4 as *"capability DAG + `FrameSpace` seam"*, and `docs/design/PLAN.md:42` states *"The geometry seam is
a stateful `FrameSpace`"* with *"one fixture forcing a reanchor"* inside G-IDENTICAL. The previous
revision replaced that seam by argument, inside its answer-in-one-page, and escalated only the LIBRARY
question. The argument is still good and is repeated below. The decision is the owner's.

**Half two — the anchor, deferred to the physics library decision.** The published `FrameSpace`
(`docs/design/sealed_shards.md:219-234`) carries `to_local_flat -> (Vec3 f32, AnchorGen)`,
`reanchor(centroid)` and a `StaleAnchor` fence, because the old planet shard ran one `f32` Rapier world
on a tangent plane. Today's position base has no `f32` anywhere on the authoritative path
(`LatticePos` is `i64` cells + `f64` offset, `crates/core/src/pose.rs:547-551`), so the anchor's only
remaining reason is a physics library that wants `f32`. That is a library decision the owner has not
taken (rapier3d was the plan; it is not a dependency). Two lawful outcomes:

- (a) `rapier3d-f64` (the same crate, `f64` feature): no anchor, no `reanchor`, no `AnchorGen`; the
  D-38 reanchor-forcing fixture becomes "a block edit on a planet AND on a hull" with nothing to force.
- (b) `rapier3d` `f32` with an anchor per physics island: the anchor is a physics-island detail BELOW
  the grid, and `GridMapping` never sees it.

Either way the grid seam is half one, and half one can land now. This report does not choose the
library (the reuse rule says: research options, the owner decides). **D8's answer decides the fate of the
owed reanchor fixture at `DEFERRED.md:331`; nothing there may be marked superseded until the owner rules.**

### 4.3 What runs once on both (HR4), and what the gate is

Above the seam, written once and blind to the arm: the chunk container, the palette, the mesher, the
smooth-surface extractor (V2.1), the edit pipeline, the collider derivation, the diff codec, the
pyramid, the signal walk. The static-shape function `static_shape(seed, CellAddr) -> Cell` runs on both
arms: on a hull it returns `Empty` for every cell (a hull is 100 % diff), on a planet it samples the
terrain. One function, two arms, identical signature.

**The gate splits, and the split is by `N`-DEPENDENCE, not by convenience:**

- `G-IDENTICAL` binds every feature ABOVE `GridMapping`: one fixture, a planet profile and a hull
  profile, equal control-plane results. This is HR4 as written.
- `G-MAPPING-TABLE` — the `N`-INDEPENDENT half, exhaustive at `N = 62`: `neighbor` is involutive; a
  four-step loop closes everywhere except at the eight corners, where it closes in three; every cell has
  four lateral neighbours; the apron is watertight at 24 strips and 24 slots; the seam table is
  byte-identical across tiers. These depend on the cube net, and `N = 62` proves them for every body.
- `G-MAPPING-ROUNDTRIP` — the `N`-DEPENDENT half, run at the LARGEST legal `N` (`2²⁶`), on cells chosen
  at the worst `a` the inverse's residual table names (`a ≈ ±0.85`, MEASURED §1.3), never spot-checked:
  `addr_of(cell_center(x)) == x` for every sampled cell; `addr_of` is `None` outside `domain`; and the
  residual stays under a stated fraction of a cell. **A run at `N = 62` cannot fail on a coarse inverse
  (§1.5), so this half is what catches the fault §1.3 found.**

HR4 is a hard rule, so the split is escalated to the owner (open decision D3), not ruled here.

### 4.4 The smallest code that lands the seam

One slice, before any mesher or generator code:

1. `crates/core/src/grid/{mod.rs, addr.rs, warp.rs, seam.rs, identity.rs, shell.rs}` — `CellAddr`,
   `GridMapping`, the six operations, the warp with `const` constants, the `const` first guess and step
   count, the generated seam table plus its pinned literal.
2. `GridParams` is DERIVED AT EACH USE, not stored: `ShellGrid` from the body's seed through the same
   taxonomy call the look already uses (`crates/physics/src/taxonomy.rs:653`), and `IdentityGrid` from
   the built realm's `bound` (`crates/core/src/built.rs:76`). **No field is added to any record in this
   slice.** `BuiltBody` derives `Serialize`/`Deserialize` (`built.rs:59-61`) and is round-tripped through
   postcard by its own test (`built.rs:167-171`); a new field on it is a STORE change and the store
   refuses a file written under a different stamp (`crates/core/src/store_stamp.rs:256-267`). If the
   owner later wants the params cached on the record, that is a separate slice with its stamp move.
3. One line in `build_app`: `voxel().geometry` selects the arm, the realm's derived params fill it
   (`docs/design/sealed_shards.md:200-212` sketches the site; the `feature::voxel` module does not exist
   yet).
4. `G-MAPPING-TABLE` at `N = 62`, `G-MAPPING-ROUNDTRIP` at `N = 2²⁶`, and the no-drift golden digest of
   `cell_center` over a sampled set of addresses on both targets (§7).

With item 2 written this way, nothing in this slice touches a wire arm, a store or a realm boundary.

### 4.5 A hull's slot as its grid domain

A built realm's bound is "the slot it was sold, not the size of its hull"
(`crates/core/src/built.rs:76`, owner 2026-09-01), and the fixtures state it as a `Shell`
(`built.rs:148` for a body, `built.rs:179` for a berth). *DISPUTED: `verdicts/grid_feasibility.md` §1
corrects the second fixture line to `built.rs:180`; line 180 is that fixture's `look`, and its `bound` is
line 179, which is the field this row is about.* An identity grid needs a BOX domain, or a block at the corner of the circumscribing
box would stand outside the realm's own shell. Two answers: sell slots as `Aabb` boxes
(`crates/core/src/geometry.rs:341`; the D-38 fixture already pairs Cartesian with `Aabb`), or take the
cube inscribed in the shell (`half = r/√3`). This is open decision D5; the recommendation is the box.

**The domain is a SERVER-SIDE admission rule and it does not need to reach the client.** The server
decides whether a placement is inside the slot; the client draws the diffs it is given, and no diff for a
cell outside the domain is ever produced. With the SIGNED address of §1.4 the client also decodes an
address without knowing `half_cells`. So the domain crosses nothing, and the SL6 request the law refuter
correctly demanded of the previous revision does not arise. It arises again the moment the address is
biased instead of signed, and §6.1 keeps that conditional request written out in full.

---

## 5. Re-centering math and the fine grid

### 5.1 What the lattice guarantees today (code, not doc)

| Fact | Where | Number |
|---|---|---|
| The FINE step is `2⁻¹⁰ m`, not 1 mm | `crates/core/src/pose.rs:255` | 0.9766 mm |
| Three rungs live: Fine `2⁻¹⁰`, Galaxy `2¹`, Universe `2¹⁵` | `pose.rs:475-486, 501, 511-517` | — |
| Every position normalises on the one in-frame move; the offset stays in `[0, edge)` | `pose.rs:623-630, 640-660` | exact, idempotent |
| The one subtraction is integer-exact | `pose.rs:709-715` | — |
| The clamp domain is `±i64::MAX/2` cells | `pose.rs:455` | ±0.476 ly at FINE (DERIVED, not measured: `2⁶² · 2⁻¹⁰ m = 2⁵² m`) |
| `Separation::metres` is exact up to `2⁵³` cells per axis | `pose.rs:816-818` | 8.8 × 10¹² m ≈ 59 AU |
| A rotation folds through `f64` only within `2⁴²` m | `pose.rs:913-916` | 29.4 AU |
| The store refuses a file written at another step | `store_stamp.rs:118, 256-261` | — |

One doc comment in that file is STALE and a reader should not trust it: `pose.rs:635-638` says no
production code calls `normalize`, but `crates/physics/src/motion.rs:62` calls `translated`, which
normalises inside itself (`pose.rs:623-630`), and the public metre constructor `from_metres` normalises
(`pose.rs:584-586`) while the raw `local` is `pub(crate)` (`pose.rs:572`).

### 5.2 A cell address against the realm's `LatticePos`

- **Identity grid:** `cell_center(a) = LatticePos::at(cell = (a.i, a.j, a.k) << 10 + 512, offset = 0)`
  and `addr_of(p) = p.cell() >> 10`, with `i, j, k` SIGNED (§1.4) so the arithmetic shift carries the
  hull's own negative half correctly. Integer both ways, bit-identical on every host. The sub-lattice is
  the same shift with `10 − s` (§3.3).
- **Cube-sphere grid:** `cell_center(a) = LatticePos::from_metres(dir(face, i, j) · r(k))`. `dir` is the
  warped, normalised face direction (polynomial + `sqrt` + divide); `r(k)` is an integer number of
  metres, exactly `1024·r` fine cells. At an Earth-sized radius the `f64` product rounds at
  `1.4 × 10⁻⁹ m` (ESTIMATED: `6.4 × 10⁶ m × 2.2 × 10⁻¹⁶`), five orders under the fine step, and
  `from_metres` normalises it (`pose.rs:584-586`). The result is deterministic because both hosts run
  the same IEEE operations in the same order on the same input (SL10 V1.4), not because it is exact.
- **The inverse (`addr_of`) on a cube-sphere** runs the four-step Newton inverse from `a₀ = t` (§1.3)
  and ONE `floor` per tangential axis. That `floor` is a physics→control quantisation boundary. It runs
  in exactly one function, and every consumer takes the integer it returns. It is NOT on the containment
  path: containment reads a `Boundary` (`crates/core/src/geometry.rs:338-347`, and
  `region_signed_distance`, `geometry.rs:1442`) and never asks which cell anything is in.

**The gate proves the FUNCTION, not the INPUTS, and targeting is the one place that matters.** The
no-drift gate of §7 compares a FIXED SAMPLE of addresses, so it cannot see a case where the client holds
a slightly different `LatticePos` than the server and the `floor` then lands one cell apart. The
consequence is bounded: chunk geometry comes from the ADDRESS, not from a pose, so collision and drawing
still agree everywhere. Only AIMING is exposed. The cure is a rule, not a gate: **the server names the
cell and the client draws the highlight from the server's own answer.** *Example: a player on the home
planet aims at a boulder. The planet's shard turns the player's own-frame `LatticePos` into a cell
address once and ships that address back with the highlight; the client draws the cell the server named,
so the mined cell and the highlighted cell are the same by construction.*

### 5.3 The planet-scale limits

- **From `i64`:** none. An Earth-sized body is `2^32.6` fine cells across (ESTIMATED: `6.371 × 10⁶ m ×
  1024`); the domain is `2⁶²`. A super-Earth, a gas giant's cloud deck and the whole radial band all fit
  with 29 bits to spare.
- **From `f64` offsets:** none inside a body; the offset is bounded to one fine cell by `normalize`.
- **From rotation:** a hull leaving a spinning planet converts its separation through
  `Separation::rotated`, which is exact-to-one-cell within `2⁴²` m (`pose.rs:889-916`). A planet's
  radius is at most `~10⁷` m; the fold is lawful by five orders.
- **From the address width:** with SIGNED fields, `i, j` need 27 bits to reach `N = 2²⁶ = 67,108,864`
  (a 42,723 km radius — MEASURED: `2·2²⁶/π = 42,722,830 m`), `k` needs 18 bits to reach a 65 km band,
  `face` 3 bits, `tier` 4 bits (`T ≤ 16`), and the sub-cell 16 bits (§3.3): **95 bits**. A cell address
  does not fit a `u64` once the sub-lattice and an Earth-sized `N` are both inside it. The packing
  therefore splits into a chunk key (`u64`) and a cell-in-chunk index plus sub-cell (`u32`) — the
  saved-record report owns the split; this report owns the total.

  *DISPUTED: `verdicts/grid_feasibility.md` finding 1 names `N ≈ 4.3 × 10⁷` as the largest `N` the
  address admits for a 42,700 km radius. `N = πR/2`, so a 42,723 km radius is `N = 2²⁶ = 6.711 × 10⁷`.
  The correction makes the finding WORSE, not weaker: the three-step residual from `a₀ = t/k₁` is 0.348
  cells there, not 0.22, so the margin against half a cell is 1.4 rather than 2.4. I have adopted the
  finding with my own measurement (§1.3).*

  *DISPUTED: the same verdict reports 0.104 cells for `a₀ = t/k₁` at three steps at Earth's `N`; I
  measure 0.052 cells with a cell width of `2/N` in `a`. The factor of two is a cell-width convention.
  Either number leaves the same conclusion: three steps from that guess are not safe, four steps are.*

### 5.4 Re-centering: the tangent anchor is not needed for the grid

The investigation planned a `SurfaceAnchor` re-centred on the player cluster with `AnchorGen` fencing
(`sealed_shards.md:219-234`; D-41 plant item 4, `DEFERRED.md:3405,3446`) because Rapier wanted `f32`
within a 10 km budget. On today's base every occupant pose is exact to the fine cell anywhere on the
body, and every chunk's vertices come from `cell_center` in the realm's own frame. The client narrows to
`f32` once, per frame, relative to the camera — the same fold the star cloud already does. That is a
render-only floating origin and is lawful (the client only renders). No anchor state exists on the server
for the grid, and no `AnchorGen` rides an address.

The D-41 tripwires that guard "more than one anchor per body" therefore guard a physics-island
question, not a grid question, and they land with the physics library slice. **This paragraph is the
argument behind D8; it is not the decision.**

---

## 6. Laws and conflicts

### 6.1 SL6 — does any new data cross a realm boundary? **No, provided the identity address is signed.**

| Candidate | From → to | Verdict |
|---|---|---|
| A body's grid parameters (`N`, floor, ceiling, rungs) | planet → gateway → client | Not a realm boundary (SL2 clarification 2026-08-24). Also derivable: `SceneRow.realm` carries `RealmId::Planet(seed)` (`crates/wire/src/channels.rs:378-379`), and SL10 lets the client run the same crate on that seed. |
| A hull's grid domain (`bound`) | hull → gateway → client | **NOT NEEDED, and NOT shipped today.** The field is `BuiltBody.bound` (`crates/core/src/built.rs:76`) — the previous revision named `built.rs:78`, which is `look`, a different field with a different meaning. What reaches the client is the `TAG_LOOK` bag (`crates/core/src/look.rs:31,84-90`) on `SceneRow`, and it carries the realm's LOOK; `RealmShape` no longer reaches the client at all (`crates/wire/src/channels.rs:350-353`). The domain stays server-side: it is an ADMISSION rule (§4.5), and the signed address (§1.4) lets the client decode without it. |
| An area's offset in its planet's cell space | planet → area | Already crosses as the parent's stamped berth. The area's frame even carries the planet's seed today (`crates/core/src/pose.rs:105`, `AreaLocal { planet_seed, area_seed }`). No new datum — but what the area may DO with it is D4a. |
| A child's cells to the parent | never | A parent positions its children and never draws them (REACH R2). The grid adds nothing upward. |
| An occupant's pose into another realm's grid | never | SL2. A ship resting on terrain is a P8 contact-ownership question, not a grid one. **An edit in flight during a crossing follows the same rule: the owning realm REFUSES an edit whose author is no longer inside it, on the same fence the transfer already carries.** *Example: a miner stands on a hull's ramp, aims at the moon's surface and fires as the hull lifts. The crossing hands the miner to the hull's realm on the same tick the moon's shard receives the edit, and the moon refuses it.* |

**THE CONDITIONAL SL6 REQUEST, written out in full so the owner can rule if the owner prefers a biased
address.** *Data:* a built realm's `bound` (`crates/core/src/built.rs:76`), one `Boundary`. *From → to:*
the built realm's shard, through the gateway, to the client. *Why the receiver cannot compute it:* a slot
is SOLD by a spaceport to a player. It is live state, not a seed draw, so SL10 gives the client no way to
derive it. *What doing without costs:* nothing, if the address is signed — the client never needs the
domain. Everything, if the address is biased by `half_cells` — the client then cannot decode a single
address. **Recommendation: keep the signed address and make no request.**

### 6.2 SL1 — an area realm on a planet

An `Area` is its own realm with its own frame (`crates/core/src/pose.rs:47,105`). Its cells are the
PLANET's cells: the static shape is `f(planet seed, address)`, and an area holds the diffs for the cells
inside its box. Two SL1 clauses bite, and the previous revision tested only the first.

**Clause 2 — the told berth as an instrument.** To evaluate the terrain under itself, the area's shard
runs the planet's `GridMapping` on `(told berth + local)`. The told berth is the stamped, read-only
reading SL1 clause 2 allows (`CLAUDE.md:138-139`). Reading it is lawful.

**Clause 4 — one hop.** *"What you are told about yourself, you NEVER pass on."* If the area STORES its
diffs at planet-global cell addresses, or ships them on a lane at planet-global addresses, then the
berth is folded into `i` and `j` and the area has passed its own placement on in a disguised form. That
is a defect, and the previous revision did not name it.

**The lawful local formulation (recommended, D4a option C).** The area USES the berth to evaluate the
planet's static shape for itself, and STORES and SHIPS every diff at an AREA-LOCAL address
`(Area seed, face 0, tier, i', j', k)`. No told placement leaves the area. The conversion from
area-local to planet-global happens where SL1 clause 1 says conversions happen — in the PARENT — and for
the client's picture it happens in the gateway, which holds both sides and is not a realm (SL2
clarification 2026-08-24). *Example: a spaceport district's shard holds an edit for the cell under its
landing pad at its own `(i' 40, j' 12, k 3)`. The planet's shard never sees that address, and the
district never states where it sits on the planet.*

**Single-writer ownership (D4b).** `RealmId` is *"a persistence/ownership realm: the unit of
single-writer durable state"* (`crates/core/src/pose.rs:34-35`), and SL10 V1.6 says a diff comes from
*"the realm that owns the cell"*. With area-local addresses the rule is clean and enforceable: **the AREA
is the single writer of every cell inside its box, and the planet is the single writer of every cell
outside every area box.** The planet's edit admission must refuse a cell that lies inside a live area's
box. The client then takes each cell's diff from exactly one realm's rows, and there is nothing to
reconcile. *Example: a player digs a trench that runs out of the spaceport district and onto open ground.
The district writes the cells inside its box; the planet writes the cells outside it; no cell has two
authors and the trench is continuous in the picture the gateway composes.*

### 6.3 SL10 — the address is the generator's input

The generator crate's version tag must fold: the face basis table, the warp constants, **the inverse's
first guess and step count**, the ladder rule, and the fine step. Two of those already fold into
`coordinate_generation` and `world_generation` (`crates/core/src/store_stamp.rs:118,122`). The others
join the same stamp when the grid module lands. A client whose grid disagrees is refused at the door,
which is the existing handshake rule.

**The shipped client does not link the generator today.** `crates/client/Cargo.toml:20-22` puts
`vd-physics` under `[dev-dependencies]`, with the comment *"DEV-ONLY: scene tests build THE world; the
shipped client never names a motion (SL4)"*. Only the client's own scene tests link it. SL10 requires the
link, and the link is a real, unmeasured cost: binary size, cold-build time, and the aarch64 no-drift
gate. It is owed as §7 item 10. *Example: a pilot's client on an aarch64 laptop must carry the same
generator crate as the moon's shard, or the boots and the drawn slope disagree. Nothing in the tree
carries it there yet.*

### 6.4 SL5 — one world

The grid module has no scale knob, no test-only `N`, no reduced body. `G-MAPPING-TABLE` at `N = 62` is
not a reduced world: it is a proof over the seam TABLE, which is `N`-independent; the production `N` runs
the same body in `G-MAPPING-ROUNDTRIP`. Every fixture that builds a planet builds it from THE world's
seed.

### 6.5 SL8 — seams

- The terrain field is a function of the 3-D direction, so the world has no seam at a face edge; only
  the index space does.
- The ladder snap happens BEFORE the first chunk, so no player ever sees a radius change.
- A partial edge chunk (§2.2) emits no triangle into the `Outside` region; the apron test proves it.
- **AN ACCEPTED SEAM, PENDING D9: the corner build refusal.** At eight known addresses per body, at
  latitude ±35.2644°, a prefab stamp returns `CornerDefect` and the build tool says no. That is SL8's
  "tier refusal" kind: a refusal the player did not cause and cannot see the reason for. It is not cured
  by this report. The previous revision offered "the generator MAY put a landform there" as a prose
  aside; a maybe is not a cure, and this report now books the refusal as a seam and asks the owner to
  rule (D9). *Example: a colonist lays a straight wall out from a spaceport. Forty kilometres later the
  wall crosses a cube corner. The stamp refuses, the wall stops, and nothing in the world explains why.*
- **SL10 does NOT let the client draw a realm nobody is drawing.** The 2026-09-01 visibility ruling says
  a dormant realm is never drawn by anybody. SL10 says the client MAY derive the static shape. Read
  together: **the client derives the static shape ONLY for a realm the gateway has already put in the
  window.** The running-realm rule still decides WHAT is drawn; SL10 decides only WHO computes the
  pixels of a realm already admitted. *Example: a pilot's client holds the generator and the seed of a
  moon three systems away. It draws nothing of that moon, because no row for it is in the window.* If
  the owner wants the client to derive AHEAD of a realm's spin-up, that is a new ruling and it is not
  proposed here.

---

## 7. Measurements owed

1. **The no-drift gate for the grid math:** a golden digest of `cell_center` and `addr_of` over a fixed
   sample of addresses on every body kind, byte-for-byte equal between an x86-64 build and an aarch64
   build, debug and release. Red on one differing byte (SL10 V1.3). Includes a check that no
   fused-multiply-add contraction happened in `warp()` (Rust does not contract by default; UNMEASURED on
   aarch64 in this tree).
1b. **The inverse warp's residual, in CELLS, at the largest legal `N` (`2²⁶`)**, re-run inside the crate
   in Rust rather than in python, for the frozen first guess `a₀ = t` and the frozen step count. The
   python figures of §1.3 are the requirement; the crate's own number is the gate.
2. **The tangential cell-edge spread** under `VD-ASC5` re-measured by OUR crate (§1.2's 0.7071–1.0050 m
   is MEASURED 2026-09-07 by a python script, not by this tree).
3. **`addr_of` cost** on a cube-sphere at `N = 10⁷`: microseconds per call with four Newton steps, and
   calls per tick on a planet with a hundred lookers (block targeting + chunk residency), to show it is
   O(1) and off the containment path.
4. **`G-MAPPING-TABLE` at `N = 62`** — all `N`-independent invariants at every rung, the seam table
   byte-identical across tiers.
4b. **`G-MAPPING-ROUNDTRIP` at `N = 2²⁶`** — the round trip and the `None`-outside-domain arm, sampled at
   the worst `a` (`≈ ±0.85`), on both grid arms, at every legal tier.
5. **A partial edge chunk's apron**: mesh a face-edge chunk whose last row is `Outside` and assert zero
   emitted quads into the outside and zero holes across the seam.
6. **The sub-lattice identity**: a proptest that `(fine_cell >> (10 − s)) & mask` equals the index-space
   sub-cell for every `s ≤ 4` on an identity grid, INCLUDING negative indices (the hull's own half), and
   the round trip through `cell_center` on a cube-sphere.
7. **A `SubGrid` cell's mesh cost** at 1/8 m: quads after merging for a dense cell, a rail, and a console;
   time per cell; the per-chunk cap that keeps a hostile chunk under the tick budget.
8. **The look-shell move on THE world** when the ladder snaps: the largest radius change over the
   generated forest, to confirm it stays under one top-rung cell for every body (the table in §2.3 is
   MEASURED against the rule, not against the crate).
9. **The warp constants' sum, inside the crate**: a `const` assertion that `(k₁ + k₂) + k₃` is exactly
   `1.0` in the order the warp evaluates it, run on x86-64 AND aarch64. §1.3's `0.0` is python3 on one
   target.
10. **The cost of linking the generator into the shipped client**: binary size and cold-build time with
   `vd-physics` (or its successor generator crate) moved from `[dev-dependencies]` to `[dependencies]` in
   `crates/client/Cargo.toml`, on both shipped targets.
11. **The identity arm's negative half**: an exhaustive test that every cell of a hull's slot box, on
   both sides of the origin on all three axes, round-trips through `addr_of`/`cell_center`.
12. **The top-rung draw budget** (owed by the render report, dependency recorded here): chunks built per
   second during a landing at flight speed, and the cost of an Earth-sized planet at 10,000 km in the
   window, against D6's `64 × 64` chunks-per-face rule.
13. **The mined-anchor rule for a landscape object** (owed by the trees report): what happens to a pine
   when the cell holding its anchor is dug out.

---

## 8. One-way doors (this domain)

| Door | Must shut | Cost if wrong |
|---|---|---|
| The grid family: cube-sphere for a body, identity for a built realm | before any P4 code | every planet's shape, every block boundary, and whether one block system exists |
| The warp: `VD-ASC5` constants, evaluation order, **the inverse's first guess `a₀ = t`**, and the `const` step count (four) | at the first golden digest (P4.1) | every stored edit on every planet re-addresses; a world-format epoch. A wrong guess or a short count is a one-cell aiming disagreement no version tag can catch (§1.3) |
| The face basis table and the seam table's orientation conventions | same digest | every seam and corner re-pairs; every apron gather is wrong |
| **The index fields are SIGNED** | with the address | a hull cannot address its own stern; a record migration over every built realm |
| The ladder rule `N = q·2^(T−1)`, `R = 2N/π`, and the radial floor as whole metres from `R − D_crust` | before the first body is saved | a body cannot be re-radiused without regenerating it under player builds |
| The rung-count rule (top rung = a face is ≤ 64 chunks) | same | the persisted tier ceiling per body, and the top-rung draw budget (§7 item 12) |
| The sub-cell address width: level 4 bits + index 3×4 bits | with the saved record (before P6 writes an edit) | a record migration over every planet, hull, station and blueprint |
| The fine step `2⁻¹⁰ m` | ALREADY SHUT (`store_stamp.rs:118`) | a 1,024× misplacement, refused at the door |
| The chunk edge (`62`) as a packing | the saved-record report's door | a re-pack, not a re-address, if the cell address is the primitive |
| The CELL PAYLOAD (the `Terrain` and `Landscape` arms) | NOT this report's door | the address does not change; §3.1 keeps the room |

---

## 9. Open decisions for the owner

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| D1 | The grid family | (A) cube-sphere + identity; (B) literal cube; (C) hex; (D) flat octree; (E) tangent-plane areas | **(A)**, signed with D9 in view | the only family that keeps blocks square, planets round, one grid with a hull, and no visible seam — its one price is the corner refusal (§1.2, D9) |
| D2 | The finest sub-block step | 1/2, 1/4, **1/8**, 1/16 m | **1/8 m shipped, 1/16 m reserved in the address** | resolves every named detail at one eighth of 1/16's cost; the door is the width, not the step (§3.3) |
| D3 | What HR4's gate means for the seam itself | (A) split: `G-IDENTICAL` above, `G-MAPPING-TABLE` + `G-MAPPING-ROUNDTRIP` on the mapping; (B) hold HR4 literally and leave the mapping ungated | **(A)** | (B) leaves the one piece everything stands on without a permanent gate; and the split must be by `N`-dependence or it cannot catch a coarse inverse (§4.3) |
| D4a | May an area name its parent's cells through its told berth, against SL1 clause 4? | (A) yes, planet-global addresses stored and shipped; (B) an area is a shard partition of the planet realm, not a grid of its own; **(C) the berth is an instrument for evaluating the shape, and every stored or shipped address is AREA-LOCAL** | **(C)** | (A) folds the told placement into a stored address and passes it on, which clause 4 refuses; (C) keeps the reading local and puts the conversion in the parent, where clause 1 says it belongs (§6.2) |
| D4b | Which realm is the single writer of a cell inside an area's box? | (A) the area, and the planet refuses cells inside a live area's box; (B) the planet, and the area holds none; (C) both, with a precedence rule | **(A)** | `RealmId` is the unit of single-writer state (`pose.rs:34-35`); (C) gives one cell two authors and the gateway nothing to compose (§6.2) |
| D5 | A built realm's slot is a box | (A) sell slots as `Aabb` boxes; (B) keep `Shell` slots and inscribe a cube | **(A)** | an identity grid's domain is a box; the D-38 fixture already pairs Cartesian with `Aabb` (§4.5) |
| D6 | The rung-count rule | (A) top rung = a face is ≤ 64 chunks (this report); (B) the face-covering rung (`m` a power of two) | **(A)**, with §7 item 12 measured before it shuts | (B) forbids realistic radii, which the owner asked for (§2.3) |
| D7 | The physics library's precision, which decides whether an anchor exists | (A) `rapier3d-f64`; (B) `rapier3d` f32 + per-island anchor; (C) another library | investigate together; no adoption here | the reuse rule: rapier3d was the plan and is not yet a dependency; this decides half two of the seam (§4.2) |
| D8 | HR4's own text names a stateful `FrameSpace` seam (`CLAUDE.md:110`, `PLAN.md:42`). May the hard rule's wording change to the stateless `GridMapping`? | (A) yes, rename the seam and move any anchor below it; (B) keep `FrameSpace` as written | **(A)** | no `f32` sits on the authoritative path (`pose.rs:547-551`), so the anchor's only reason is a library not yet chosen (§4.2, §5.4). D8's answer also decides the fate of the owed reanchor fixture (`DEFERRED.md:331`) — nothing there may be marked superseded until the owner rules |
| D9 | The corner build refusal, as an SL8 seam | (A) accept the refusal as it is; (B) the generator ALWAYS places a mountain, a crater or a sea at all eight corners, so no flat build site exists there; (C) a per-body landform, so the corners differ per planet; (D) design the blueprint walk to bend across a corner | **(B)** | it removes the refusal from the player's experience without changing the grid, and it costs eight seeded landforms per body. (D) is a research item with no known answer that keeps blocks square (§1.2, §6.5) |
| D10 | A rotation joint's geometry (V2.5) | (A) a MOUNT: a second identity grid inside the same realm, orientation authored by the hull; (B) the turret is a child REALM; (C) off-grid rigid geometry | **(A)** | it forks no code (the same arm, instantiated twice), so HR4 holds; (B) makes forty joints forty realms; (C) contradicts V2.3 and V2.5's own words (§3.6) |
| D11 | May a sub-metre block go in a PLANET cell? | (A) yes, and it inherits the cell's anisotropy (0.0884–0.1256 m per 1/8 m part, MEASURED); (B) identity grids only | **(A)** | (B) gives a surface base less detail than a hull, which is a "detail-by-box" seam under SL8 (§3.5) |

---

## 10. Stale claims in the investigation base, superseded

| Claim | Where | Superseded by |
|---|---|---|
| `LatticePos::map_offset` is the cell-preserving move | `DEFERRED.md` D-41 landed note | deleted; `translated` normalises internally (`crates/core/src/pose.rs:614-630`) |
| The FINE tier is `i64` at millimetres | D-41 text, `addendum_1`, the memory note "ONE mm grid" | `2⁻¹⁰ m` (`pose.rs:255`), with a written reason |
| The COARSE tier is dormant until P10 | D-41 | three rungs live: Fine, Galaxy `2 m`, Universe `32,768 m` (`pose.rs:475-486`; owner Q1 2026-08-24) |
| "Through P3 no production code calls `normalize`" | `pose.rs:635-638` (a doc comment) | `crates/physics/src/motion.rs:62` calls `translated`, which normalises inside itself (`pose.rs:623-630`) |
| `FrameSpace` / `feature::voxel::register` exist | `sealed_shards.md:200-234`, `capability.rs:11` | no such type or module in any crate (grep, §2.1) |
| The planet radius is per-body data on a 39.47 m ladder | `addendum_1` §C, `block_system_design.md` §3.4 | the code draws a continuous radius from the mass–radius law (`generate.rs:1232`, `taxonomy.rs:653`); the 39.47 m step is an artefact of whole tangential chunks (§2.2) |
| A full-depth body's `m` is a power of two | `block_system_design.md` §3.3.3 rider, §3.4 | refused: it forbids realistic radii; the top rung is budget-derived (§2.3) |
| The realm discriminator shares the chunk `u64` | `block_system_design.md` §2.7.1 | `RealmId` is an enum with `u64` and 128-bit payloads (`pose.rs:34-66`); it never shared a `u64` |
| A single planet cluster is a HARD ERROR asserted in code | `sealed_shards.md:236` | no `SurfaceAnchor` and no such assertion exist (grep); the constraint is a physics-island question, not a grid one (§5.4) |
| The galaxy is a lattice of cell-realms | `DEFERRED.md` D-SCALE-1 (2026-08-05) | dropped by owner Q7 2026-08-24; the galaxy is one realm at a `2 m` step, lazily generated |
| The client may not hold the generator | owner Q4 2026-08-24, S6 2026-08-27 | SL10 (2026-09-07): the client MAY derive the static shape with the same crate. **The link does not exist yet:** `vd-physics` is a DEV dependency of the client (`crates/client/Cargo.toml:20-22`), so SL10 requires a new, unmeasured link (§6.3, §7 item 10) |
| `addendum_2` "Fact one: the client has the generator and the seed" | `addendum_2` §B | was unlawful when written (S6 2026-08-27); lawful now by SL10, not by its own argument — and still not true of the tree |
| "The one-metre cell — answered by the owner" is a shut door | `decision_board.md` §3 "doors already passed" | qualified by V2.4: the 1 m cell stays the space-taking unit; a sub-lattice lives inside it (§3) |
| `coarsen_level` is a terrain ladder | `addendum_2` §A | a pose-precision hop counter; corrected by the board's S8/S11 and by this report's address (the tier is on the cell address, nowhere on the wire yet) |
| P4 delivers a "pinned `noise`" crate | `roadmap.json` P4, `Cargo.toml:78` | declared and unused; SL10 V1.4 forbids `libm` transcendentals inside the generator, which the generator report must weigh against O7's vendoring option |
| "A fixed three-step Newton inverse" is the frozen warp | `block_system_design.md` §3.3.2, and the previous revision of THIS report | the first guess was never stated and it decides the answer: three steps from `a₀ = t/k₁` leave 0.348 cells at the largest legal body. Four steps from `a₀ = t` (§1.3) |

---

## 11. Revision log

Two refuters reviewed the 2026-09-07 13:14 revision. Every finding is listed with what I did.

### From `verdicts/grid_law.md`

| Finding | Verdict | What I did |
|---|---|---|
| F1 — the hull's grid domain is `bound` (`built.rs:76`), not `look` (`built.rs:78`), and it does not reach the client | WRONG | FIXED and the mechanism changed. §6.1 row 2 rewritten with the right field and with the evidence that `RealmShape` no longer reaches the client (`channels.rs:350-353`) and that `TAG_LOOK` carries the LOOK (`look.rs:31`). **I also removed the need:** §1.4 makes the identity address SIGNED, so no bias and no domain datum is needed on the client, and §4.5 states the domain as a server-side admission rule. The conditional SL6 request is still written out in full in §6.1 in case the owner prefers a biased address. DISPUTED, narrowly: an SL6 request is not unavoidable — the lawful local formulation SL6 itself demands is the signed address. |
| F2 — the shipped client does not link `vd-physics` | WRONG | FIXED. §6.3 and §10 now state that `vd-physics` is under `[dev-dependencies]` (`crates/client/Cargo.toml:20-22`) and that SL10 requires a link that does not exist. Added §7 item 10, the binary-size and cold-build cost. |
| F3 — §4.4 claims "no store" while item 2 adds a field to a stored record | BREAKS_LAW | FIXED. §4.4 item 2 now DERIVES `GridParams` at each use from the seed and from `bound`; no record changes in the slice. The store facts (`built.rs:59-61,167-171`, `store_stamp.rs:256-267`) are cited. |
| F4 — the corner refusal is an unbooked SL8 seam | BREAKS_LAW | FIXED. Booked in §6.5 as an ACCEPTED SEAM pending a ruling, and added as D9 with four options; D1's row now says "signed with D9 in view". §1.2's example now shows the refusal. |
| F5 — `k₁+k₂+k₃ == 1.0` called MEASURED from another document; `±0.476 ly` called MEASURED | UNMEASURED_AS_FACT | FIXED and partly upgraded. I MEASURED the sum myself (python3 IEEE `f64`, both orders, exactly `0.0`) and said so with the target; the crate-side `const` assertion on both targets is added as §7 item 9. `±0.476 ly` relabelled DERIVED. |
| F6 — the three-arm cell list has no room for smooth terrain (V2.1) or a tree (V2.2) | MISSING | FIXED. §3.1 rewritten: the list is PLACEABLE state only, and a table names the `Terrain` and `Landscape` arms, who designs each, and that neither changes the address. V2.1 and V2.2 added to §1.1. §8 gains a row saying the payload is not this report's door. |
| F7 — `PLAN.md`'s stateful `FrameSpace` seam is dropped by argument, not escalated | MISSING | FIXED. §4.2 now carries a marked escalation, and D8 asks the owner whether HR4's wording may change. §5.4 is relabelled "the argument behind D8, not the decision". The owed reanchor fixture (`DEFERRED.md:331`) stays owed until the owner rules. |
| F8 — SL1 clause 4 untested for an area; single-writer ownership unresolved | MISSING | FIXED. §6.2 rewritten with clause 4 named, a lawful local formulation (area-local addresses), and D4 split into D4a (clause 4) and D4b (the single writer), each with a recommendation. |
| F9 — SL10 could let a client draw a dormant realm | MISSING | FIXED. Added the ruling sentence and its example to §6.5: the client derives only for a realm the gateway has already put in the window. |
| F10 — four citations drift | Citation drift | FIXED. `RealmId::Area` → `pose.rs:47`; `FrameRef::AreaLocal` → `pose.rs:105`; `StationLocal` → `pose.rs:102`; `VoxelGeometry` → `capability.rs:37-42`. I re-checked every other citation in the report against the tree and corrected `built.rs:180`, `pose.rs:623-630`, `pose.rs:709-715`, `pose.rs:816-818`, `taxonomy.rs:653`, `geometry.rs:338-347`, `channels.rs:378-379`. |
| §2.3's `N` column is the un-snapped `N` | (label) | FIXED. The table prints `N` ideal and `N` snapped, recomputed by my own script; the rows reproduce the refuter's hand check. |

### From `verdicts/grid_feasibility.md`

| Finding | Verdict | What I did |
|---|---|---|
| 1 — the Newton inverse's first guess is unstated and three steps are not bounded | UNMEASURED_AS_FACT | FIXED. I re-measured (200 001 samples, both guesses, 2–5 steps). §1.3 now carries the table, freezes `a₀ = t` AND four steps, and §8's warp door names both. DISPUTED on two numbers, both in the refuter's disfavour: the largest legal `N` is `2²⁶ = 6.711 × 10⁷` (from `N = πR/2` at a 42,723 km radius), not `4.3 × 10⁷`, so the three-step margin is 1.4, not 2.4; and I measure 0.052 cells at Earth where the refuter measured 0.104, a cell-width convention. The conclusion is unchanged and I adopted it. |
| 2 — the gate cannot see finding 1 | UNMEASURED_AS_FACT | FIXED. §4.3 splits the gate by `N`-DEPENDENCE into `G-MAPPING-TABLE` (`N = 62`, the cube net) and `G-MAPPING-ROUNDTRIP` (`N = 2²⁶`, sampled at the worst `a`). §1.5 states plainly what the `N = 62` run does and does not prove, with the 1.5-million margin measured. §7 items 4 and 4b follow. |
| 3 — the address cannot hold a hull's own cells | WRONG | FIXED. §1.4 makes `i, j, k` SIGNED `i32` and explains why a bias is worse; §5.2 and §5.3 follow; §8 adds the signedness as its own door; §7 item 11 adds the negative-half test. |
| 4 — `CornerDefect` is a tier-refusal seam | BREAKS_LAW | FIXED with F4, above (D9). |
| 5 — HR4 names the `FrameSpace` seam; the report renames it | BREAKS_LAW | FIXED with F7, above (D8). |
| 6 — the owner's tree (V2.2) appears nowhere | MISSING | FIXED with F6, above. §3.1 adds the `Landscape` arm, states that its volume MAY leave its anchor cell (the one exception, and why), gives the hull-through-canopy example, and hands the mined-anchor case to the trees report with §7 item 13. |
| 7 — no room for a smooth-terrain field (V2.1) | MISSING | FIXED with F6, above. §3.1 adds the `Terrain` arm and says the record width is the saved-record report's door. |
| 8 — sub-metre blocks chosen without re-reading the warp's anisotropy | MISSING | FIXED. New §3.5 states the rule, with my own measurement of the micro-cell size at three places on the face, and adds D11 (planet-legal, recommended, vs identity-grid only). |
| 9 — a rotating joint has no address | MISSING | FIXED. New §3.6 names three answers and adds D10, recommending the MOUNT (a second identity grid inside the same realm, hull-authored orientation). V2.5's joint clause added to §1.1. |
| 10 — the §2.3 `N` column | MISSING (label) | FIXED, above. |
| 11 — determinism argued about the function, never the inputs | UNMEASURED_AS_FACT | FIXED. §5.2 now says plainly that the gate proves the FUNCTION, that only aiming is exposed, and states the cure as a rule: the server names the cell and the client draws the highlight from the server's own answer. The old §5.2 example that claimed the gate covered it is replaced. |
| 12 — no frame budget for D6's rung rule | MISSING | FIXED as a recorded dependency. §2.3 states the top-rung chunk count and chunk size (ESTIMATED), names the render and terrain reports as the owners, and §7 item 12 owes the measurement before D6 shuts. |
| 13 — an edit in flight during a crossing | MISSING (small) | FIXED. §6.1's last row already refuses an occupant pose across a boundary; I added the rule to it: the owning realm refuses an edit whose author is no longer inside it, on the same fence the transfer already carries. |
| 14 — the §2.1 grep list is not exhaustive | MISSING (small) | FIXED. §2.1 now lists `crates/sim/src/stub/transient.rs:239` and `crates/node/src/saga_runtime/tests.rs:3513` as well, and says "among them". |
| §1 — "the second `Shell` fixture is `built.rs:180`, not 179" | (citation) | DISPUTED and kept. `built.rs:179` is the berth fixture's `bound` and `built.rs:180` is its `look`; §4.5 is about `bound`, so 179 is right. The dispute line is written in §4.5. |
