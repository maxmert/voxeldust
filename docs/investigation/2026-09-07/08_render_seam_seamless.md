# 08 — The render seam, and seamlessness from orbit to the ground

**Domain:** the seam between the engine-free client library and a render engine (V2.8), and the
seamless law (SL8, V2.9) for the landing sequence: from orbit to the ground, walking, and every realm
in the window at any distance.

**Date:** 2026-09-07. **Revision 2**, after the law refutation and the feasibility refutation. §12 is
the revision log. **Status:** an investigation report. It is an input to the owner's decisions. It
binds nothing. The rulings in `docs/design/owner_decisions_*.md` win over every sentence here.

**Method.** I read the binding law first (CLAUDE.md, the 2026-09-07 voxel ruling, the reach ruling, the
visibility ruling, the suit ruling, the seed ruling, the movement rulings, DEFERRED). I read the
investigation base second (block design §5, §6.7, addendum 2, the decision board, the stunning-look
plan, the snowflow analysis). I read the code last, and the code decides what exists today. Every claim
about the code carries a `file:line`. Every number is marked MEASURED (with how) or ESTIMATED.

---

## 0. What exists today, in the code

This section is the ground truth for the rest of the report. The base documents describe a past state.

### 0.1 The seam that exists

- The client library is engine-free. It talks to the gateway, decodes snapshots, interpolates, and
  assembles input. It touches no clock, no socket, no renderer
  (`crates/client/src/lib.rs:1-15`, `crates/client/Cargo.toml:6`).
- The render engine consumes ONE type: `MeshPrim { vertices: Vec<Vertex>, color_rgba, transform }`,
  with `Vertex { pos: [f32;3], normal: [f32;3] }` and
  `PrimTransform { translation, scale, rotation: [f32;4] }`
  (`crates/client/src/realm_scene.rs:770-796`). The rotation field EXISTS (D-MOVE-2,
  `realm_scene.rs:783-790`). The engine never learns the word "sphere" or "box"
  (`realm_scene.rs:792-794`).
- The Bevy renderer builds a `Mesh` from that vertex array and nothing else
  (`crates/client-render/src/lib.rs:1828-1846`). It is a NON-INDEXED triangle list, so the
  `ChunkGeometry` index buffer this report proposes is genuinely new, not a fit to the old type. The
  code comment says P4 greedy quads slot in there unchanged (`lib.rs:1824-1827`).
- Every realm the client draws is a `SceneRow`: a realm id, its parent id, a stamped pose in the
  origin frame, and a tagged skip-unknown TLV bag for its look (`crates/wire/src/channels.rs:378-391`).
  The bag carries three tags today: the outline (`TAG_LOOK`), the photometric datum (`TAG_LUMA`) and
  the extent (`TAG_EXTENT`) (`crates/core/src/look.rs:31-48`). A realm states its own look through
  `self_look_bag` (`crates/sim/src/stub/window.rs:485-496`).
- There is no chunk, no terrain, no voxel type anywhere in the client (grep for `chunk` in
  `crates/client/src/*.rs` returns only datagram chunking). `BulkKind::ChunkSnapshot` and
  `BulkKind::ChunkDelta` are reserved names with no producer (`crates/wire/src/channels.rs:283-296`);
  `BulkMsg` is the RELIABLE PER-SUBSCRIPTION lane (`channels.rs:283-289`).
- `FrameSpace`, `SphericalSpace` and `reanchor` appear only in COMMENTS and in one test NAME — six
  hits: `crates/core/src/fence.rs:18`, `crates/sim/src/capability.rs:11`, `:18`, `:35`,
  `crates/sim/src/stub/transient.rs:239`, and `crates/node/src/saga_runtime/tests.rs:3513`. No such
  TYPE and no such MODULE exists, and the code says so in its own words: "`FrameSpace` does not exist
  yet" (`crates/sim/src/capability.rs:18`), with the seam owed at P5.
- The workspace declares `noise = "=0.9.0"` (`Cargo.toml:78`), and no crate under `crates/` uses it
  (grep for `noise::` returns zero hits). The only noise code that ran is the standalone
  `scripts/noisebench` (its own workspace root). No crate names rapier (only a comment,
  `Cargo.toml:96`).

**Example.** Today a moon in the window is one `SceneRow`: its placement from the star system, and a
bag that says "a sphere of this extent". The engine gets a unit sphere scaled to that extent. It never
gets a surface.

### 0.2 The render frame that exists

- The camera sits at the render origin and carries only a rotation. Every drawn thing is placed at
  `world − eye`, subtracted in `f64` and narrowed to `f32` once
  (`crates/client-render/src/lib.rs:534-541`, `1264-1275`, `1519-1526`).
- The eye is held ON THE LATTICE with its unit (`RenderEye::eye_lattice`, `lib.rs:541-566`), and a drawn
  position is reduced against it before it is flattened (`lib.rs:1266-1274`).
- A row stated in a frame other than the session's own is a far row. It is placed from the sky anchor
  as the star cloud is (`lib.rs:1527-1548`).
- The near and far planes are derived every frame from what was drawn
  (`lib.rs:762-786`, `crates/client-harness/src/camera.rs:312`).
- The picture is drawn in the origin realm's own frame. The camera's up is that frame's `+Y`
  (`lib.rs:1136-1145`). There is no "local up" from a planet centre anywhere in the renderer.

### 0.3 The sky that exists

- ONE star cloud for the whole catalogue, one mesh, one entity, uploaded once, placed every frame by
  the sky anchor (`lib.rs:480-489`, `1407-1445`). The catalogue holds 233,220 stars, and the WATCHER at
  the spawn drew them in the owner's flight (MEASURED by that flight, reach ruling R9,
  `docs/design/owner_decisions_2026-09-02_reach.md:192-193`). The MEASURED drawn-IN-VIEW figure from
  the star gate is 41,345 (`docs/design/DEFERRED.md:1334`).
- The star law is Tier-A: flux `L/d²`, a power-law response, a pixel-size floor, a brightness cull,
  never a size cull (`crates/client/src/realm_scene.rs:651-681`). The shader transliterates it and every
  constant arrives as a uniform (`crates/client-render/src/star_sky.wgsl:1-45`).
- The `f32` underflow that culled a third of the faint stars is cured by folding the square root of the
  gain in before the square (`lib.rs:1350-1358`; MEASURED 2026-09-07: 0 misses of 41,345 stars in view,
  multisampling ON, `docs/design/DEFERRED.md:1325-1337`).
- **Multisampling at 4× is SUSPECTED of dimming a one-pixel star to a quarter. ESTIMATED from coverage
  arithmetic, never measured as a painted luminance.** What was OBSERVED is narrower: in ONE run, with
  multisampling OFF, twelve missing pixels came back while the underflow loss did not
  (`DEFERRED.md:1337-1340`, which says in its own words "not decided here"). The quarter is the
  coverage fraction of one sample in four, worked out on paper. §7 item 5 owes the painted-luminance
  reading, and §9 makes that reading a PRECONDITION of the anti-aliasing door.
- The renderer never names `Msaa` (grep over `crates/`: only two comments mention MSAA,
  `crates/bins/tests/render_boxes_smoke.rs:107`, `crates/client-harness/src/verdict.rs:149`). The
  client runs `Msaa::Sample4` by Bevy's default, never by choice.

### 0.4 The dev-control harness that exists (HR6)

- The protocol is engine-free JSON lines: `Move`, `Look`, `Action`, `Close`, `ResetInput`, `State`,
  `WaitUntil`, `Screenshot { at_tick, label }`, `Record { fps, secs, label }`, `WalkTo`, `LookAt`
  (`crates/devproto/src/dispatch.rs:15-73`).
- `DevState` carries the diagnosis surface: the drawn realm boxes, the origin, the held sky, the stars
  DRAWN, the camera mode, and the star probe — the renderer's own projection of its 48 brightest stars
  (`crates/devproto/src/state.rs:111-200`; the probe is filled at `lib.rs:1166-1192`).
- The screenshot is a wgpu readback node inside the Bevy renderer (`lib.rs:2303-2440`).
- The listener is a compile-time feature of the client binary (`crates/bins/Cargo.toml:60-66`).

### 0.5 The numbers that exist

- Tiers of the lattice: `Fine` = 2⁻¹⁰ m cells (about 1 mm), `Galaxy` = 2 m, `Universe` = 32,768 m
  (`crates/core/src/pose.rs:475-486`). A position is an `i64` cell plus an `f64` offset
  (`pose.rs:548-551`).
- **THERE ARE TWO ANGLES, and they do two different jobs.** The code separates them in its own words
  (`crates/core/src/geometry.rs:1163-1169`).
  - **The DOT angle**, `VISIBILITY_THETA_MIN_RAD = 0.026180` rad (1.5°), `geometry.rs:1140-1146`. It is
    the bar for WAKING a realm. **A realm states its own REACH with THIS angle** — `extent · cot(θ/2)`,
    `visibility_reach_m`, `geometry.rs:1148-1154`. Its factor is `cot(θ/2) = 76.4` (ESTIMATED;
    arithmetic on `visibility_factor`, `geometry.rs:1136`).
  - **The DRAWABLE angle**, `drawable_theta_min_rad() = 1.1506 × 10⁻³` rad — one pixel at the reference
    view of 45° vertical over 720 rows (`geometry.rs:1156-1173`, asserted at `geometry.rs:3000-3006`).
    It is the floor for SHIPPING an occupant to an observer, and it is the angle that sizes a CELL. Its
    factor is `cot(θ/2) = 1,738` (ESTIMATED, same arithmetic).
  - The two differ by 22.8× (ESTIMATED). Every sentence below names which one it uses.
- The tick rate is a PER-SHARD KNOB, not a constant. The code says so where it is read: "`tick_hz` is a
  per-shard knob passed in" (`crates/core/src/kinematics.rs:148-153`). The DEV CLUSTER runs it at
  50 Hz, and the constant that says so is named `DEV` with the doc "The standard dev-cluster
  parameters" (`crates/bins/src/lib.rs:412-414`). The client learns the rate from the wire
  (`ServerControlMsg::UniverseRate`, `crates/wire/src/channels.rs:130-137`); its own default tuning
  says 20 Hz until it learns (`crates/client/src/tuning.rs:22-27`). The interpolation buffer is 120 ms
  (`crates/wire/src/channels.rs:328`). No residency or bandwidth number below is quoted at a rate
  nobody ships.
- The datagram budget is 1,200 bytes (reach ruling R9; a 1,420-byte hull window frame was dropped
  forever by the transport, MEASURED, `owner_decisions_2026-09-02_reach.md:200-202`).
- Bevy 0.18, glam 0.30, `bevy_ecs` 0.18 on every shard (`Cargo.toml:43-52`). No rapier in the workspace
  manifest today. `crossbeam-channel = "0.5"` IS a workspace dependency (`Cargo.toml:66`).
- The world-generation tag exists on the store stamp (`crates/core/src/store_stamp.rs:118-146`) and on
  the inter-shard hello tag (`crates/wire/src/admin.rs:373-376`). SL10 §3 says it must also name the
  generator crate's version and the target's arithmetic profile. It does not yet.

---

## 1. Question 1 — The seam: what the client library hands an engine for a voxel realm

### 1.1 The three options, judged

| Criterion | (a) Cells + diffs; the engine meshes | (b) Meshes from a shared Rust mesher; the engine uploads | (c) Both; the engine chooses |
|---|---|---|---|
| **SL10 §2 one generator, no port** | The engine plugin would call the generator through C. Lawful. But an engine that MESHES writes a second surface extractor in C++. That is a port of the surface. **Refused.** | The generator AND the mesher are Rust, compiled once into the server and every client. **Lawful.** | Lawful ONLY if the cell lane never builds a surface the server answers a collision question about (§1.1 item 2, the collision line). |
| **SL10 §4 determinism rules** | The engine's C++ mesher obeys no §4 rule. **Refused.** | The mesher can sit inside the §4 fence: integer hashing, fixed evaluation order, no fast-math flag, no fused multiply-add contraction, no libm transcendental, only `+ − × ÷ √` (`owner_decisions_2026-09-07_voxels.md:36-41`). | Same as (b) for the shape lane. The cell lane is display-only and is §4-free. |
| **SL10 §5 collision on the same shape** | The server's collision surface comes from the Rust mesher. The engine's display surface comes from its own code. Two surfaces. **Drift by construction.** | One surface. | One surface, and the collision line is what keeps the cell lane out of it. |
| **No-drift as a measurement (SL10 §3)** | Cannot be measured: the engine's mesh is C++ output. | `ChunkGeometry` bytes compare server-to-client. **Measurable — and today UNMEASURED.** | Measurable for the shape lane. **UNMEASURED.** |
| **An Unreal plugin's needs** | Unreal gets cells and builds meshes with its own tools (Dynamic Mesh, Nanite). Natural for Unreal. | Unreal uploads an index/vertex buffer it did not build. Its own LOD is by tier request, not by mesh decimation. Nanite can take a runtime mesh, but a per-edit Nanite build is a cost (ESTIMATED; not measured on Unreal). | Unreal reads the shape lane for every colliding surface and the cell lane for material, decoration and non-colliding instancing. |
| **Bevy today** | Bevy has no mesher. New code in the coverage-exempt crate. | `mesh_from_vertices` already takes a vertex array (`lib.rs:1828-1846`), but it is non-indexed; the index buffer is new. One encoder function. | Same as (b) plus a cell reader. |
| **HR6** | Engine-side. | Engine-side. | Engine-side. The harness protocol is already engine-free. |
| **HR5** | The mesher is untested C++. | The mesher is Tier-A Rust at 100 %. | Same as (b). |

**Recommendation: (c), narrowed — and the recommendation is CONDITIONAL.** Option (c)-narrowed is
lawful **IF** the shape lane obeys SL10 §4, **IF** every colliding surface rides the shape lane, and
**IF** the byte-identity gate goes green on x86-64 and on aarch64. All three are UNMEASURED today. A
gate that compares bytes proves nothing about whether the bytes CAN agree; SL10 §4 is the ruling that
makes them able to.

The client library hands an engine TWO lanes. The SHAPE lane is mandatory and is the only lawful source
of a surface. The CELL lane is read-only and serves display attributes only.

1. **The shape lane.** The client library evaluates the shared generator for `(seed, address, tier)`,
   overlays the diffs it holds, runs the shared mesher, and hands the engine a `ChunkGeometry`:
   integer cell-space vertices at a STATED VERTEX QUANTUM (§1.4), indices, packed attributes, and a
   header with the tier, the quantum, the chunk origin in the realm's frame, the skirt depth, the
   part-transform table (§1.5) and a version counter. The engine encodes that into its own vertex
   layout and uploads it. The engine never changes a vertex.
2. **The cell lane, and THE COLLISION LINE.** The engine may read the cell field of a resident chunk:
   material ids, the provenance bit, the block parameters (V2.6 style), the attachment list (V2.5).
   **One test splits the two lanes, and it goes INTO the seam contract: ANYTHING THE SERVER ANSWERS A
   COLLISION QUESTION ABOUT COMES OUT OF THE SHAPE LANE; THE CELL LANE CARRIES ONLY WHAT NOBODY CAN
   WALK INTO.** So the cell lane carries textures, decoration density, bark detail, leaf cards, wind
   sway, grass, HUD widget art on faces, and ambient occlusion. It does NOT carry a tree's trunk or its
   main limbs, because a walker hits those and the moon's shard sweeps her against them. V2.2 is
   explicit: "the collisions are calculated on the server, so server somehow should know the shape"
   (`owner_decisions_2026-09-07_voxels.md:74-78`). A trunk goes through the SHARED MESHER with the
   ground.

**Why not (b) alone.** A block's style adds non-colliding details the server does not store (V2.6). An
attachment's WIDGET is a picture, not a wall (V2.5). Leaf cards and grass are decoration. Every one of
these is an engine-side instancing or material decision that reads the cell record, not the surface.
Option (b) alone forces those through the mesh lane and the shared mesher grows an engine's whole
decoration system.

**Why the derivation lives in the client library, not in the engine plugin.** SL10 §2 forbids a port.
An engine plugin in C++ cannot call Rust generics; it can call a C interface. The client library is
Rust and links the generator crate directly. The Unreal plugin links the client library through one C
interface (the voxel ruling V1.2 names this path). So the generator is called from exactly one place on
the client side: the client library. The Bevy renderer is a Rust crate and could call the generator
directly, and it must not: two call sites on the client are two places to disagree on tier, apron and
diff overlay.

**Example.** A pilot descends toward a moon. The client library asks the generator for the moon's
chunk at address `(sector 2, a 40, b 17, c 3)` at tier 5, overlays the two pyramid entries the moon's
shard sent for that region (a mine somebody dug), meshes it, and hands the engine about 1,900 quads
(ESTIMATED, block design §5.2.1 fixture) in cell space with `scale = 32 m` and the chunk's placement
inside the moon's row. A pine on that hillside comes out of the SAME mesh: its trunk is quads the
mesher wrote. The engine adds the needles as cards from the cell lane, because a walker cannot hit a
needle. The moon's shard evaluates the same crate at tier 0 when a character stands there, and the
character's boots meet the same slope and the same trunk.

### 1.2 The function set

The client library's engine-facing surface. Rust functions with a C mirror. Every function is
engine-free and Tier-A. Names are proposals.

**The set splits in two by SL10 §4, and an implementer must be able to tell the halves apart.**

- **§4-BOUND (the shape lane).** The generator, the mesher, the normal packing, and every attribute
  inside `ChunkGeometry`. These outputs are compared BYTE FOR BYTE between a server build and a client
  build, on x86-64 and on aarch64. They obey SL10 §4 in full: integer hashing for every random draw, a
  fixed evaluation order, no fast-math flag, no fused multiply-add contraction, no call into libm's
  `sin`/`exp`/`pow`, and only `+ − × ÷ √`. Where a curve is needed the crate carries its own
  deterministic implementation or an integer table.
- **§4-FREE (the lighting and sky lane).** `sun_set`, `exposure_ev100`, `atmosphere_params`. Their
  output is never compared byte for byte between two hosts. It is a per-observer display parameter that
  feeds a shader, and it computes `2·atan(R/d)`, an inverse-square illuminance and a colour temperature.
  These may use `atan` and `pow`. They must NOT feed anything the shape lane reads, and a control
  asserts it: the shape crate does not depend on the lighting module.

**The loop (exists today, unchanged).**
- `client_start(config) -> Client` — the core loop on its own thread (`crates/client/src/lib.rs:1-15`).
- `client_render_snapshot(&Client) -> Arc<RenderSnapshot>` — wait-free, once per frame
  (`crates/client/src/render_snapshot.rs:33-71`).
- `snapshot.rendered(now) -> [(entity, sub, RenderPose)]`, `snapshot.scene_now(now) -> RealmScene`,
  `snapshot.sky()`, `snapshot.sky_anchor_now(now)` (`render_snapshot.rs:386-459`).
- `client_input(&Client, InputAction)` — the input-resource seam the harness injects into
  (`crates/devproto/src/dispatch.rs:81-88`).

**The chunk lane (new, §4-bound).**
- `chunk_tier_for(&Client, realm, key, distance_m) -> u8` — **THE TIER RULE LIVES IN THE LIBRARY, not
  in the engine.** One rule, one `cell_angle_max` field, every realm kind. An engine may override it
  only under a debug flag, and `DevState` counts the overrides beside the refusal counter of §3.3
  item 9. Without this the detail-by-box tolerance (§3.3 item 4) cannot be gated on an Unreal client,
  and V2.8's engine-agnostic promise loses its only seam test.
- `chunk_request(&Client, realm, key: ChunkKey, tier: u8, priority)` — the engine asks. The library
  never refuses a legal tier (`0 ..= tier_depth(realm)`). The work runs on the library's own worker
  pool (§1.6); the render thread is never entered.
- `chunk_poll(&Client) -> [ChunkReady { realm, key, tier, version, geometry: ChunkGeometry }]` — the
  engine harvests finished chunks. A bounded number per call.
- `chunk_release(&Client, realm, key, tier)` — the engine drops residency. The library frees the cells
  and the geometry.
- `chunk_changed(&Client) -> [(realm, key)]` — a diff arrived for a resident chunk. The engine
  re-requests the tiers it holds. The library coalesces.
- `chunk_cells(&Client, realm, key, tier) -> &CellField` — the read-only cell lane, display only, under
  the collision line of §1.1 item 2.
- `realm_shape_params(&Client, realm) -> Option<ShapeParams>` — the seed-derived body parameters
  (radius on the ladder, the grid family, the tier depth, the atmosphere class) from the realm's look.
  `None` until the realm runs and states its look — the same presence gate the window already uses
  (`crates/sim/src/stub/window.rs:485-496`).

**The engine's placement of a chunk (exists today as `to_render_prims`).**
- `chunk_place(rbox: &RealmBox, key, tier, eye: &RenderEye) -> PrimTransform` — the chunk's origin in
  the render frame: the row's placement plus the chunk origin, reduced against the eye on the lattice,
  narrowed once. The row's rotation carries the planet's spin (`realm_scene.rs:783-790`).

**The lighting and sky parameters (new, Tier-A, engine-free, §4-FREE).**
- `sun_set(&snapshot) -> [Sun { direction, illuminance_lux, colour_temperature_k, angular_size }]` —
  **from the luminous rows IN THE WINDOW**, kind-blind. A row is a sun because it carries light
  (`TAG_LUMA`, `crates/core/src/look.rs:32-34`), never because it is a `RealmId::Star`. The chain is
  universe → galaxy → star system → planet, so for a pilot on a planet the STAR is the planet's
  SIBLING, never its ancestor: a walk of the ancestor chain finds no sun and the pilot lands in the
  dark. The window already ships the children in range (reach ruling R10 decision 3), and the star's
  row is one of them.
- `exposure_ev100(&snapshot) -> f64` — closed-form from the dominant sun's illuminance.
- `atmosphere_params(&snapshot, realm) -> Option<AtmosphereParams>` — bottom radius, top radius,
  scale heights, the scattering class, the ground albedo, derived from the body's seed record.

**The harness (exists; the engine owes two hooks).**
- The protocol stays `vd-devproto`. The engine implements `Screenshot`/`Record` as a readback of its
  own back buffer, and routes the injected `InputAction` into the same input resource the human uses.
  Bevy has both (`crates/client-render/src/lib.rs:2303-2440`, `894-1050`). An Unreal plugin owes both.

**Example.** The Bevy renderer calls `chunk_tier_for` and then `chunk_request` for about 505 tier-0
columns around the avatar (ESTIMATED, block design §5.9.2 column count; the range must be re-derived
from the one reference view, §7 item 12), and `chunk_poll` every frame. The Unreal plugin calls the
same three functions through the C mirror and feeds the geometry to its runtime mesh component.
Neither engine calls the generator, and neither engine picks its own ladder. The harness sends
`{"cmd":"screenshot"}` to both and reads the same manifest.

### 1.3 The C interface

The voxel ruling V1.2 says an Unreal client links the crate through a C interface. The tool that writes
a C header from Rust signatures is `cbindgen`. It is NOT a dependency today. Adopting it is the owner's
decision under the library rule. The alternative is a hand-written header, which drifts. This report
lists `cbindgen` as an option and adopts nothing.

### 1.4 The vertex quantum — the field that keeps V2.4's door open

**The problem.** §9 says the `ChunkGeometry` contract must shut BEFORE the Unreal plugin starts and
before the Bevy encoder is written. V2.4 is the owner's requirement for blocks smaller than one metre,
so a player can build little details INSIDE the one-metre cell. If the geometry's vertex integers count
whole metres, a half-metre handle on a hull door cannot be expressed, and the door §9 shuts first shuts
on V2.4.

**The fix.** `ChunkGeometry`'s header carries a named `vertex_quantum` field beside the tier: the
sub-cell step one vertex integer is worth. Its value is a DECISION for the owner, and the sub-metre
report should price the two candidates.

| Quantum | Steps across a 62-cell chunk | Bits per axis | What it can express |
|---|---|---|---|
| 1/16 m | 992 | 10 (fits a `u16`) | a 6.25 cm feature; a half-metre door handle easily |
| 1/32 m | 1,984 | 11 (fits a `u16`) | a 3.1 cm feature; twice the precision, the same `u16` |

Both fit one `u16` per axis at every legal tier (ESTIMATED; arithmetic on the 62-cell chunk of block
design §5.2.1). The collision CELL stays one metre (§5 item 3.1); the vertex QUANTUM is a different
statement about a different thing.

**Example.** A builder puts a half-metre handle on a hull's door. The handle is a sub-metre block
inside one 1 m cell. With the quantum at a sixteenth of a metre the mesher writes the handle's corners
as integers 8 and 16 inside that cell, and the same integers carry a moon's ground at tier 9. One seam,
one arithmetic.

### 1.5 A joint, a turret and a HUD widget that moves — the part-transform table

V2.5 names a rotation joint that turns what is built on it, a manipulator and a remote-controlled
turret (`owner_decisions_2026-09-07_voxels.md:88-92`). §3.3 item 8 rules that a chunk mesh rides the
row's placement and has NO pose lane of its own. A turret is part of a hull's grid, so it is not a
realm with a row of its own; and it turns every tick, so it cannot ride the hull's row unchanged.

**The lawful shape.** A joint's angle is LIVE STATE — a signal drives it — so it is NOT seed shape, and
SL10 §6 already rules where it goes: **it crosses as a DIFF from the owning realm.**

1. The hull's shard authors the joint angle, as it authors every other piece of its own state.
2. The angle rides the diff lane, stated ON CHANGE, one packed scalar per moving joint. It is not an
   occupant pose and not a realm placement, so SL2 and SL1 are untouched.
3. `ChunkGeometry`'s header carries a PART-TRANSFORM TABLE: a small array of part ids and their
   transforms, and every vertex names its part. The engine composes `row placement → part chain →
   chunk`, exactly as it composes the row's rotation today
   (`crates/client-render/src/lib.rs:1562-1570`).
4. A HUD widget on a turret's face is an attachment on a cell of a part, so it inherits the part's
   transform for free.

**Cost.** One packed angle per moving joint per change on a send-on-change lane, plus one `u8` part
index per vertex in the geometry (ESTIMATED). A hull with no joints has a one-entry table and pays
nothing. **This belongs to the `ChunkGeometry` door** (§9), because it is a header field, and a header
field added after two engine plugins consume the contract is two rewrites.

**Example.** A gunner turns a turret on a hull's spine while the pilot flies. The hull's shard states
the turret's angle when it changes. Every client composes the hull's row, then the turret's part
transform, then the turret's chunks and the HUD widget on its side plate. The pilot's client and the
gunner's client compose the identical chain.

### 1.6 The worker pool — the mechanism, named

§1.2 and §3.3 item 7 put generation and meshing on "the client library's own worker pool". `vd-client`
is Tier-A at 100 % region and branch under HR5, and it is fenced structurally: no clock, no sockets, no
sleep, no default-hasher maps (`crates/client/clippy.toml:6-18`). No thread-pool crate is a dependency
of it today (`crates/client/Cargo.toml:8-19`), and rayon is nowhere in the workspace.

- **Recommended, and it adopts nothing new:** `std::thread::spawn` plus `crossbeam-channel`, which IS
  already a workspace dependency (`Cargo.toml:66`). The clippy fence bans `std::thread::sleep`, not
  `std::thread::spawn`, so the shape fits the fence that exists.
- **The owner's alternative:** rayon, listed as an option and adopted by nobody here.
- **How it reaches 100 % branch.** The pool is a thin injected seam: a `ChunkWorkers` trait with one
  method that takes a job and returns a receiver. The Tier-A tests drive an INLINE implementation that
  runs the job on the calling thread, so every branch of the scheduling logic is covered without a
  thread; the threaded implementation lives in the Tier-B binary, exactly as the QUIC transport does
  today.

---

## 2. Question 2 — Who owns the detail tier, and what the server owes

### 2.1 The rule

- **The client LIBRARY picks the tier**, through `chunk_tier_for` (§1.2). The rule is the one angular
  rule the realm loop already uses: pick the rung whose cell subtends a fixed angle (addendum 2 §C.2;
  the stunning-look plan §6.1 makes the knob an ANGLE, `cell_angle_max`, not a pixel count).
- The engine asks for a tier and may override the library's answer only under a debug flag, counted in
  `DevState`.
- The library supplies any legal tier from the generator plus the diffs it holds.
- The server never learns the tier.

### 2.2 The laws, checked one by one

| Law | Question | Verdict |
|---|---|---|
| SL3 a realm draws itself | SL3 says the realm authors its look "at a detail level it chooses". Does the observer's client choosing the tier take that from the realm? | **No conflict.** Under SL10 the realm's look for terrain IS its statement `(seed, address, diffs)`. Every tier is a pure function of that one statement. The realm chose its look once. The tier is the observer's sampling of what the realm stated, exactly as the star cloud is one statement drawn at any distance. |
| SL1 / SL2 no pose crosses | Is a tier a pose? | **No.** A tier is a number on a chunk address in the realm's own frame. |
| SL6 ask before new data crosses | Does the tier cross? | **No data crosses for the tier.** Default NO respected. |
| SL9 unbounded child count | Does the tier scheme walk children? | **No.** Chunks are keyed by address inside one realm. Realms in the window are the reach ruling's business. |
| HR3 one tooling | Does anything branch on a realm kind? | **No.** A tier is data on the address (block design §5.1.4). The G-IDENTICAL fixture meshes one chunk at tier 0 and tier 3 on a planet profile and a hull profile. |
| V4 one machinery every realm | Does a hull get a different ladder from a moon? | **No.** A hull's chunks get the same ladder. A hull's coarse rung is pure pyramid (addendum 2 §C.4). |
| SL10 §6 diffs from the owning realm | Does the coarse tier need data the client cannot derive? | **Yes: the pyramid entries for edited regions.** That is a diff, and the owning realm ships it. See §2.3. |
| SL10 §4 determinism | Can the two hosts agree byte for byte? | **UNMEASURED.** The rules exist (`owner_decisions_2026-09-07_voxels.md:36-41`); the gate is §7 item 4 and has never run. |

### 2.3 What the server DOES need, and HOW THE DIFF REACHES A LOOKER WHO IS NOT STANDING THERE

The first revision said "the realm's shard derives the diff interest from the poses it authors" and
refused a client-stated interest, and left a hole: **a realm's shard does not know where a looker is
unless the looker is its own occupant.** A window carries a SCOPE and no eye — `WindowScope` is
`Occupants` or `Child(RealmId)` (`crates/wire/src/session_flow.rs:582-590`) — and the window registry
is keyed by the OPENER, never by an observer (`crates/sim/src/stub/window.rs:38-46`). SL2 forbids the
pilot's pose from entering the moon. Left as written, a tunnel a miner dug would be invisible to every
looker who is not on the ground, which is seam kind 6 manufactured by the design itself.

**SL6 says: find the local formulation first. There is one, and it has two halves.**

**Half 1 — every looker CLOSE ENOUGH to resolve a fine rung is already the realm's own occupant.** A
body's containment bound is its gravitational sphere of influence, not its surface: a moon's bound is
`planet_soi(...)` (`crates/physics/src/worldgen/generate.rs:1794-1798`), while its LOOK is its surface
radius (`generate.rs:1819`). Containment membership puts every occupant of that volume in the moon's
realm, so the moon AUTHORS its pose and derives interest from it with no crossing at all.

- The finest rung a looker at distance `d` can resolve has a cell edge `s ≥ d / 1,738` (ESTIMATED, from
  the drawable-angle factor of §0.5 with a cell's circumscribed extent).
- A hull ten kilometres above a moon resolves cells of about 6.6 m — rung 3 (ESTIMATED). It is deep
  inside the moon's sphere of influence, so the moon holds it and ships it what it needs.
- A looker OUTSIDE a moon-sized body's sphere of influence (about 6.6 × 10⁷ m for an Earth–Moon-like
  pair) resolves nothing finer than a 44 km cell — rung 16 (ESTIMATED). No fine diff could be drawn
  there even if it arrived.

**Half 2 — for a realm whose BOUND is close to its own size, the realm states its edits to every
watcher, down to the rung drawable AT ITS OWN BOUND.** That rung is a number the realm computes from
its own bound alone: `L_public = ceil(log2(bound_r / 1,738))`. Nothing about a looker crosses.

- A hull with a fifty-metre bound gets `L_public = 0`: it states its whole edit set to everyone
  watching it, and that set is bounded by the hull's own size, so it is small.
- A moon gets `L_public = 16`: it states only the coarsest pyramid entries to outsiders, which is
  exactly what an outsider can draw.
- The cost is bounded and it is the RIGHT bound: the volume of what a realm publishes scales with what
  a looker outside it can see, which is what the drawable angle measures.

**The residual, and the SL6 ask that covers it.** Half 2 makes a realm publish at one rung for all
outsiders, so a watcher hovering just outside a big station's bound gets a coarser picture than her
eyes could resolve. If the owner judges that visible, the lawful carrier is the CONNECTION PLANE, not
another realm: SL2's own clarification says the gateway is not a realm and alone holds both sides. §8
ask 3 states that ask in full. **Recommended: take the local formulation, and hold the ask in reserve.**

**The rest of what the server owes.**

1. **The diffs.** Player edits, placed blocks, sub-metre blocks, attachments, joint angles, growth
   stages, damage, live-state deposits — SL10 §6. Per chunk key, content-versioned, latest-wins per
   key, on the reliable per-subscription bulk lane (`BulkMsg`, `channels.rs:283-289`), never on the
   1,200-byte datagram.
2. **The edit pyramid.** Per coarse cell that holds an edit: the solid count and the dominant material,
   maintained by the owning realm on every edit (addendum 2 §C.4). The client cannot derive it from the
   seed, because an edit is not in the seed. The WHOLE pyramid above tier 0 costs about one seventh of
   the tier-0 edit volume (ESTIMATED; the geometric sum of `1/8^L` in three dimensions), so publishing
   it is cheap and the tier-0 diffs are the real volume.
3. **The reach of the realm** (exists: `crates/sim/src/stub/reach.rs:37-64`), stated on change and
   carried `Retained` — the shape every new send-on-change datum in this report copies.
4. **The tier-0 shape around every occupant, for collision** (SL10 §5). The server needs no other tier.
   The server never meshes for display.

**One asymmetry to record.** The server holds the pyramid and never draws a coarse rung. The client
draws coarse rungs and never writes a pyramid. That is correct: the pyramid is DATA about edits, and
edits belong to the realm.

**Example.** A miner digs a 300 m tunnel on a moon. The moon's shard writes the tier-0 diffs and about
one pyramid entry per rung above them. A pilot ten kilometres up is inside the moon's sphere of
influence, so the moon holds her hull, authors its pose, and ships her the entries her distance can
draw. Her client asked for rung 3; the moon never learned that. A second pilot a million kilometres
away, outside the moon entirely, gets only the moon's public coarse entries — and at that distance one
of those cells is 44 km wide, so she could not have drawn the tunnel anyway. The tunnel never vanishes
and never pops.

---

## 3. Question 3 — The landing sequence under the eleven seam kinds

### 3.1 The sequence

1. **In the star system.** The planet is a point of light, then a disc. It is in the window because the
   observer is inside its REACH — stated by the planet at the DOT angle (`geometry.rs:1140-1154`) — and
   it runs and draws itself (reach ruling R2, V2).
2. **Approach.** The disc grows. The realm's own look must become the top rung of the terrain ladder
   without a visible hand-over.
3. **Atmosphere entry.** The sky changes from black to blue. Exposure changes by more than ten stops
   (ESTIMATED: interplanetary night to a lit surface spans more than 20 EV, snowflow §3.12).
4. **Descent.** Rungs refine from the face-covering rung to tier 0. Chunks are generated, meshed and
   uploaded while the hull moves.
5. **Touchdown and walk.** Tier 0 everywhere within about 1,505 m — `0.866 m × 1,738`, a one-metre
   cell's circumscribed extent at the DRAWABLE angle (ESTIMATED; arithmetic on
   `visibility_reach_m`, `geometry.rs:1148-1154`, at `drawable_theta_min_rad()`,
   `geometry.rs:1171-1174`). The moon's shard computes collision on the same shape.
6. **Every other realm in the window.** The moons, the stations, the star, and the galaxy's cloud stay
   drawn throughout. The galaxy is never re-anchored (R1).

### 3.2 What the client must have generated BEFORE the hull descends

- The planet runs when the observer enters its REACH, and that reach uses the DOT angle, so it is
  22.8× larger than a range worked out at the drawable angle (§0.5). An Earth-radius body's reach by
  size is about `6.37 × 10⁶ m × 76.4 ≈ 4.9 × 10⁸ m` (ESTIMATED) — half an astronomical unit. The client
  has a very long warm-up, not a tight one.
- Its look reaches the client one hop up and through the gateway (R6 table). With the look come the
  body's shape parameters (§1.2 `realm_shape_params`). From that moment the client library can generate
  the coarse rungs.
- **Warming ahead, WITHOUT a predicted pose.** The client may not compute `p + v·t_warm`. The eye
  stands at the pilot's avatar, which is an OCCUPANT, and the client's render sample for an occupant is
  a `RenderPose` with NO velocity field at all (`crates/client/src/interp.rs:84-101`). The client crate
  states the fence twice as a STRUCTURAL rule — "the interpolation sample is a `RenderPose` with NO
  velocity field, and `StampedPose::advanced_ballistic`/`.vel` are never read on the render path"
  (`crates/client/src/lib.rs:11-15`, repeated at `crates/client/src/interp.rs:10-13`) — and CLAUDE.md
  and SL10 §7 both forbid a derived pose or velocity on the client. Two lawful shapes remain, and the
  owner picks one (§6 item 11):
  - **(A) The realm states the warm radius.** The realm already grows its reach by closing speed ×
    boot time (reach ruling R8), and `crates/sim/src/stub/aoi.rs:369` already reads a predictive
    horizon. The realm states one more scalar on the same send-on-change lane its reach rides.
  - **(B) The residency rule reads the SPREAD of the delivered track** — the distance the avatar
    actually moved between the last two DELIVERED stamps. That measures the past and predicts nothing;
    no pose is derived, only a working-set radius.
  - **Either way, keep the fence.** The residency reader lives in its own module, the draw path never
    names it, and a module dependency control FAILS when somebody deletes that rule — the same
    structural shape SL1 clause 5 and SL4 already demand, "never by care".
- **Coarse-first is a rule.** Never request a fine rung for a region with no coarser rung resident
  (block design §5.1.3 item 3). A late chunk is a blurrier chunk, never a hole.
- Rung ranges are NOT quoted here. §7 item 12 owes the one reference view first; the first revision
  quoted a rung-14 range from an angle of `1.27 × 10⁻³` rad while quoting a cell range from
  `1.1506 × 10⁻³` rad, which is the very trap §9's third door exists to shut.

**Example.** A hull leaves a station in orbit around a planet. The planet is already drawn at a coarse
rung from the seed, because the pilot was inside the planet's reach — at the DOT angle, half an
astronomical unit out — the whole time. The hull points down. The client library requests the next
finer rungs for the columns under a residency radius grown from the delivered track's own spread, not
from a predicted eye. When the hull is at 50 km the mid rungs are resident under it, and rung 0 has
not been asked for yet.

### 3.3 The eleven seam kinds, from orbit to the ground

Every tolerance is PHYSICAL (SL8). Every mechanism names the code or the design that provides it. Where
a tolerance is a number nobody has measured, it is marked ESTIMATED and §7 owes the bench.

**1. Jump** — a drawn thing moves by more than its motion.
- Where: the crossing from the star system into the planet realm (the origin swap); the hand-over from
  the point of light to the body; a chunk crossing a tier boundary.
- Mechanism: the origin swap waits for the hop so a pilot boards without a blank (commit `c2612e1`); the
  picture is re-expressed in the new origin frame and the view takes the delivered facing
  (`lib.rs:1110-1122`). A chunk's address is realm-local, so the crossing moves nothing inside a chunk:
  the chunk's placement is the row's placement plus a constant.
- Tolerance: a drawn chunk's screen position across the swap moves by less than one pixel. `f64`
  re-expression at 1 × 10⁷ m carries an error below 2 mm (ESTIMATED from the `f64` epsilon), far below a
  pixel at any distance the ladder draws.

**2. Black frame** — a frame with nothing where something was.
- Where: the hand-over between authors; a chunk mesh swap; a re-mesh after an edit.
- Mechanism: the author hand-over despawns and respawns IN THE SAME FRAME, never zero drawn, never both
  (`lib.rs:1572-1600`). **That code swaps a whole realm's body, not a chunk; chunks do not exist yet.
  The pattern is BORROWED, and the chunk swap must be written to it.** Plus coarse-first residency, and
  a mesh handle swap that holds both handles until the new one is resident (block design §5.1.3 item 6,
  with a named fallback of one frame that is LEDGERED, not accepted).
- Tolerance: zero frames in which a resident region has no rung drawn. The pop detector (snowflow §6
  item 10) asserts it over a recorded sequence; it does not exist yet.

**3. Flicker** — a thing appears and disappears on successive frames.
- Where: a star crossing a pixel boundary; a sub-pixel quad at a grazing angle; a dither band without a
  temporal resolve; z-fighting where a coarse rung meets a fine one.
- Mechanism: the star's pixel-size floor and brightness-only cull (`realm_scene.rs:651-681`,
  `min_crop_px`); skirts on the four lateral borders (block design §5.9.4); the dither crossfade needs a
  temporal resolve (block design §5.9.5) — which is the anti-aliasing decision of §5.
- Tolerance: frame-to-frame luminance variance in a fixed region below a threshold while the camera
  moves slowly (block design §5.10.3 slice 3's sparkle scenario). Threshold ESTIMATED; the bench owes it.

**4. Detail-by-box** — detail chosen by what KIND of realm a thing is, or by its bound.
- Where: a moon drawn as a grey ball because it is "a moon"; a hull drawn at full detail across the
  system because it is "a ship".
- Mechanism: the tier comes from the angle a cell subtends, one rule, in the LIBRARY, every realm kind
  (§2.1; visibility ruling V4). The client's kind-keyed helper `role_hsv` maps `Planet`, `System`,
  `Ship`, `Station`, `Area` and `Star` each to its own hue
  (`crates/client/src/realm_scene.rs:719-731`). It is a cosmetic FALLBACK for a realm that states no
  colour, its own note already says the drawn photometric colour rides `TAG_LUMA` "never this fallback
  family", and it must never reach the tier or the material path. **It RETIRES when a realm's material
  comes from its own seed record** — §10 item 12 ledgers that deletion, because until then a moon is
  grey partly because it is a moon, which is seam kind 4 in its mildest form.
- Tolerance: for every realm kind, the resident rung at distance `d` is
  `L = ceil(log2(d × cell_angle_max))` with one `cell_angle_max` field; a fixture asserts it for a moon,
  a hull and a station at the same distance. The fixture is only gateable because `chunk_tier_for`
  lives in the library (§1.2).

**5. Brightness pop** — the frame's brightness steps.
- Where: atmosphere entry; the sun rising over a limb; a warp arrival next to a bright star; the sky
  swap between two bodies' atmospheres.
- Mechanism: ONE physical exposure law, closed-form from the dominant star's illuminance `L/4πd²`, both
  seed- and tick-derived, with a bounded rate-limited adaptation as a pure Tier-A function (snowflow
  §3.12; the stunning-look plan O20 refuses automatic exposure because stars vanish on warp arrival).
  Every emissive becomes an absolute seed-derived radiance. The atmosphere's contribution ramps by
  physics (transmittance and in-scatter are continuous in altitude), never by a toggle.
- Tolerance: the frame's mean luminance changes by less than one eighth of a stop between successive
  frames outside a physical cause (ESTIMATED; a sunrise is a physical cause and is allowed).

**6. Arrival pop** — a realm appears with detail it did not grow into.
- Where: the point of light becoming a body; the body becoming terrain; a station appearing at its reach
  edge; **an edit that arrives late** (§2.3 answers that one).
- Mechanism: a running star system draws its body over its own point at the same brightness — no
  hand-over message exists, so no pop exists (reach ruling R2). For a planet: the realm's own look is
  the coarsest rung of the SAME generator (block design §5.9.8 option G1: blocks to the face-covering
  rung, the proxy only beyond it), so the disc and the terrain are one surface at two rungs, crossfaded.
  At the reach edge the grace latch and the hysteresis hold (`aoi.rs:181-186`, reach ruling R8).
- Tolerance: at the hand-over distance the coarse rung's silhouette differs from the sphere by less than
  one pixel. The relief bound `4·k_rough·2^L` (block design §3.5.6, ESTIMATED) divided by the distance
  must be below the DRAWABLE angle. This is the same angular rule again.

**7. Tick hitch** — a frame that takes much longer than its neighbours.
- Where: chunk generation on the render thread; a whole-mesh re-upload; a first-use shader compile at
  arrival; the atmosphere tables built when the planet's component is added.
- Mechanism: the client library's worker pool generates and meshes; the render thread only uploads
  (§1.6, block design §5.1.3). A per-frame upload budget — in Bevy that is `RenderAssetBytesPerFrame`,
  whose DEFAULT is `max_bytes: None`, meaning NO throttle at all
  (`bevy_render-0.18.1/src/render_asset.rs:466-471`), so the client must SET it; the mechanism does work
  because meshes report their size (`bevy_render-0.18.1/src/mesh/mod.rs:145`). The Unreal equivalent is
  the plugin's. The pipeline set is a function of the BUILD, never of the world (snowflow §3.5): a new
  planet adds uniforms and textures, never a shader permutation. The atmosphere is warmed at connect
  and its component is never added or removed at runtime (snowflow §4.9).
- **Tolerance, stated PHYSICALLY (SL8).** A frame is a hitch when THE DRAWN PICTURE JUMPS: the scene's
  angular motion in one frame exceeds the motion a constant-rate frame would have produced by more than
  the drawable angle, `1.1506 × 10⁻³` rad (`geometry.rs:1171-1174`). The permitted extra frame time is
  therefore `θ_pixel / ω`, where `ω` is the fastest drawn angular rate in that frame. At a 90°/s look
  turn that is 0.73 ms (ESTIMATED; arithmetic on the drawable angle). The instrument EXISTS: the star
  probe projects the 48 brightest stars through the renderer's own camera
  (`crates/devproto/src/state.rs:111-200`, filled at `lib.rs:1166-1192`), so a frame-to-frame pixel
  delta of those projections measures `ω` directly. The first revision's tolerance — "the 1 % low frame
  time stays within two times the median" — is a ratio between two engineering numbers, and on a 50 Hz
  dev cluster it permits a 40 ms frame beside a 20 ms one, which is the visible stutter the item exists
  to forbid.

**8. Re-state rate** — a thing moves in steps because its placement is restated too seldom.
- Where: the planet's row (its orbit and spin) under a chunk mesh; the sky anchor; a hull's row while a
  walker stands on its deck; **a turret turning on that hull** (§1.5).
- Mechanism: the chunk mesh rides the row's placement plus its part transform. It has NO pose lane of
  its own. The row is interpolated on the render cursor with the 120 ms buffer (`channels.rs:328`,
  `render_snapshot.rs:386-398`); the sky anchor is sampled at the SAME cursor (`lib.rs:1375-1378`).
- Tolerance: zero new per-tick data for terrain. A chunk drawn at the row's interpolated placement moves
  as the row moves, at the render rate. A joint's angle is send-on-change, not per tick.

**9. Tier refusal** — a request for a rung is refused, or a realm is refused because of what it is.
- Where: the client library refusing a rung above some cap; a hull refused a coarse rung because "a
  ship has no terrain"; a realm not drawn because the roster did not vouch for it (the 2026-09-01 black
  sky).
- Mechanism: the library never refuses a legal tier; a hull's coarse rungs are pure pyramid; the
  occupancy bit no longer decides what anybody can see (visibility ruling V5); a window ships every
  child in range (reach ruling R10 decision 3).
- Tolerance: a counter of refused tier requests in `DevState`, asserted zero in every flight, beside the
  engine-override counter of §1.2.

**10. Sprite cull** — a point of light removed because it became small, or dimmed by the raster.
- Where: the star cloud; a station's point at the edge of its reach; the far end of the marker floor.
- Mechanism: cull on brightness, never on size (`realm_scene.rs:658-664`); the pixel floor
  (`camera.rs:254-265`); the `f32` underflow cure (`lib.rs:1350-1358`). **Open:** 4× multisampling is
  SUSPECTED of dimming a one-pixel star to a quarter (ESTIMATED from coverage, §0.3; one run observed
  twelve pixels return with multisampling off, `DEFERRED.md:1337-1340`). The anti-aliasing decision
  (§5, item 3.8) closes it, and §9 forbids that door to shut before the luminance is read.
- Tolerance: every star the law draws paints at its predicted pixel (the gate exists and is green:
  0 misses of 41,345, MEASURED, `DEFERRED.md:1334`). Add: a one-pixel star's painted luminance is within
  a stated fraction of the law's amplitude under the chosen anti-aliasing.

**11. Lane flood** — a lane carries more than the link can deliver and everything behind it waits.
- Where: the diff lane after a stall; a client arriving at a heavily built city; a keep-alive that
  restates a whole roster (the galaxy's 22 MB per keep-alive, MEASURED, reach ruling R10,
  `owner_decisions_2026-09-02_reach.md:237-238`).
- Mechanism: diffs per chunk key, content-versioned, latest-wins per key, on the RELIABLE BULK lane; a
  client behind receives the pyramid summary for a region before its fine diffs (block design §5.2.3
  mitigation 2, coarse-first on the wire); the window ships children in range only; the datagram budget
  is 1,200 bytes and the reliable lane is paced by the transport tuning, never inline literals. **Every
  new look-bag tag is Retained and stated ON CHANGE** (§8 ask 1), never on a keep-alive, because the
  bag rides a per-row lane whose row count is unbounded under SL9 and that lane has already broken
  twice.
- Tolerance: the sustained diff lane below a stated fraction of the link, and a catch-up after a
  ten-second stall completes within the interpolation buffer plus one round trip for the tier-0 disc
  around the avatar (ESTIMATED; the bench owes it).

**Two cases the first revision did not cover.**

- **A diff arrives WHILE the origin swaps.** A diff is keyed by `(realm, chunk key, version)` and a
  chunk address is realm-local, so the swap changes the origin FRAME and never a chunk address. The
  diff therefore applies identically before and after. Tolerance: a fixture applies one diff on the very
  frame of the swap and asserts the drawn chunk is byte-identical to applying it one frame earlier.
- **A miner removes the cell UNDER a placed block.** Whether the block falls, floats or the removal is
  refused belongs to the block-record report. The SEAM consequence is stated here: one edit can
  invalidate a NEIGHBOUR chunk's pyramid entry, so a pyramid entry is keyed by the COARSE CELL, not by
  the edit, and the owning realm restates every entry an edit touches. Without that, latest-wins-per-key
  is not enough and the coarse rung keeps a hole that the fine rung has filled.

**Example, whole sequence.** A pilot in a hull leaves the home station. The planet below is one
`SceneRow` with its shape parameters; her client already holds its coarse rungs, because the planet's
reach at the dot angle covered the whole approach. She pushes the stick. The client requests the next
rung under a residency radius grown from her delivered track's own spread. The hull crosses into the
planet realm; the origin swaps after the hop; the planet's chunks do not move on screen. At 60 km the
sky brightens by physics, and the exposure law follows the star's illuminance, not a histogram. At 5 km
the fine rungs are resident and rung 0 is being generated for the landing field. She touches down. The
planet's shard evaluates the same chunk at tier 0 and her hull's skids sit on the drawn ground. The
stars are still there, dimmed by the daylight sky's transmittance, never culled. The galaxy's cloud was
never rebuilt.

---

## 4. Question 4 — Planet-scale facts the seam must survive

### 4.1 A planet radius on the ladder

- The lattice's fine step is 2⁻¹⁰ m and the cell index is `i64` (`pose.rs:475-478`, `548-551`). An
  Earth-radius body (6.37 × 10⁶ m) is 6.5 × 10⁹ fine cells across — inside `i64` by nine orders of
  magnitude.
- A chunk address at 1 m cells is exact integer arithmetic. The tier field on the address must hold the
  face-covering rung: `tier_depth = ceil(log2 m) + 1`, 12 for a 162 km body, 18 for Earth (block
  design §5.9.3, ESTIMATED arithmetic). Five bits hold it. This is the grid-format report's door, not
  this report's; this report only states that the render seam reads that field and never a second one.
- The planet's radius is per-body DATA from its mass (`crates/physics/src/taxonomy.rs:653-655`), never
  a constant. Its LOOK radius and its containment BOUND are different numbers: the look is the surface
  (`crates/physics/src/worldgen/generate.rs:1819`), the bound is the sphere of influence
  (`generate.rs:1794-1798`). §2.3 depends on that difference.

### 4.2 `f32` at the camera

- The camera is at the render origin; a chunk's transform is `f32` after one `f64` subtraction
  (`lib.rs:1264-1275`). A chunk's vertices are cell-local integers and exact in `f32` at either
  candidate quantum of §1.4 (992 or 1,984 steps per axis, far below `2^24`).
- The `f32` step at distance `d` is about `d × 6 × 10⁻⁸`. The cell size at distance `d` under the ladder
  is `d × θ_drawable = d × 1.1506 × 10⁻³`. The ratio is 1.9 × 10⁴ (ESTIMATED; arithmetic on the `f32`
  epsilon and the drawable angle): the placement error is always about twenty thousand times smaller
  than one drawn cell, at every distance, BECAUSE the cell grows with the distance. The ladder is what
  makes `f32` safe. A single-tier world at 10,000 km would have a 0.6 m placement error on a 1 m cell.
- The eye reduces on the lattice before the flatten (`lib.rs:541-566`). Two chunks of one column reduce
  against the same eye and never disagree on their shared edge.

### 4.3 The floating origin per realm row

- Every realm is drawn at its row: the row's placement in the origin frame, interpolated, plus the
  row's rotation (`realm_scene.rs:783-790`, `lib.rs:1547-1560`). A chunk is a child of its realm's row
  in the render frame. The planet's spin is on the row, and every chunk turns with it at no cost.
- The camera is inside a realm. The origin frame is that realm's frame. When the observer stands on the
  planet, the planet's own chunks are at their address plus nothing; the moons are far rows placed from
  the sky anchor as the star cloud is (`lib.rs:1527-1548`).
- **What is missing:** the camera's UP. Today it is the origin frame's `+Y` (`lib.rs:1136-1145`). On a
  planet, up is the direction from the planet's centre to the eye. The client can compute it from
  delivered data: the planet's row (its centre) and the eye. That is display arithmetic on delivered
  data, the same kind as the `f64` subtraction — it derives no pose and no velocity. The stunning-look
  plan §4.4 says the server must author the orientation. The two readings disagree; §6 puts it to the
  owner.

### 4.4 The atmosphere and lighting seam

The AAA lighting standard: a Hillaire-class sky, generic for thousands of planets, all from the seed.

- **The model is engine-agnostic and both engines have it.** Bevy 0.18 ships `bevy_pbr::atmosphere`
  (Hillaire 2020, four lookup tables and a composite; snowflow §4.4). Unreal's `SkyAtmosphere` is the
  same Hillaire 2020 model (its published reference). So the SEAM is the parameters, not the shader.
- **The parameters are a function of the seed and the delivered rows.** Bottom radius and top radius
  from the body's radius and scale height; the scattering class from the composition field the
  taxonomy already returns (`crates/physics/src/taxonomy.rs`, the classifier); the sun direction from
  `star row − observer` in the composed frame; illuminance `L/4πd²`; colour from the star's effective
  temperature; the sun's angular size `2·atan(R/d)`. Nothing is authored. The client library computes
  these as Tier-A, SL10 §4-FREE functions (§1.2 `sun_set`, `atmosphere_params`, `exposure_ev100`).
- **Scaling, and it is better supported than the first revision knew.** Bevy 0.18.1's `Atmosphere`
  component holds a `Handle<ScatteringMedium>`
  (`bevy_pbr-0.18.1/src/atmosphere/mod.rs:210-231`), so sharing a small set of archetypal media BY
  HANDLE, deduplicated by quantised parameters, is the engine's own shape and not a workaround.
  Per-planet distinctness rides the cheap per-camera fields (snowflow §4.4, ESTIMATED 8–16 classes).
- **The traps in Bevy's shipped sky, RE-VERIFIED against the vendored 0.18.1 source.** The first
  revision relayed three traps from an older Bevy. Two of them have moved:
  - *The Earth radius pair is NOT hardcoded on the component.* `bottom_radius` and `top_radius` are
    PUBLIC fields, and the Earth numbers live in a named constructor `Atmosphere::earthlike`
    (`bevy_pbr-0.18.1/src/atmosphere/mod.rs:210-244`). A methane world sets its own two numbers.
  - *The aerial-perspective range is a public settings field*, `aerial_view_lut_max_distance`, whose
    default is `3.2e4` — 32 km (`mod.rs:333`, `mod.rs:359`). The trap is the DEFAULT, not the code.
  - *The Mie transposition SURVIVES, inside the medium.* `ScatteringMedium::earthlike` writes the Mie
    term as `absorption: Vec3::splat(3.996e-6)` and `scattering: Vec3::splat(0.444e-6)`
    (`bevy_pbr-0.18.1/src/medium.rs:143-148`), while Hillaire 2020's reference values are Mie
    SCATTERING 3.996 × 10⁻⁶ and Mie ABSORPTION 4.40 × 10⁻⁶. The two are swapped and the absorption is a
    tenth of the reference. DISPUTED: the feasibility refuter (F6) wrote that "Mie survives only as
    comments inside the medium" — the numbers at `medium.rs:143-148` are live constructor arguments,
    not comments, and this project writes its own medium anyway, so the fix stays a data fix.
- **The structural trap is real, and the first revision's cure was too strong.** The shader assumes up
  is `+Y` and says so: `get_view_position` adds `vec3(0.0, atmosphere.bottom_radius, 0.0)`
  (`bevy_pbr-0.18.1/src/atmosphere/functions.wgsl:303-306`), and `get_local_up` carries the comment
  "We assume the `up` vector at the view position is the y axis, since the world is locally
  flat/level. … NOTE: this means that if your world is actually spherical, this will be wrong."
  (`functions.wgsl:308-313`). A planet-tangent frame fixes the EYE's up. It does NOT fix the flat
  approximation ALONG the ray at `functions.wgsl:311-312`, so the sky stays approximate toward the limb
  and at altitude. **The tangent frame is NECESSARY and NOT SUFFICIENT**, and §7 item 13 owes a
  measurement of the residual error at the horizon and from orbit. Unreal's sky takes a planet centre,
  so on Unreal the tangent frame is a convenience.
- **One atmosphere per camera, on both engines.** A warp fly-by past three worlds gets one sky. The
  hand-off: the sky you are in is the deepest ancestor whose atmosphere you are inside; every other
  body's sky is off; the swap point is where the departing sky's contribution is below the noise floor,
  invisible by physics (snowflow §4.4). Other bodies get an analytic limb-glow shell on their own look
  (the stunning-look plan §4.3, ESTIMATED 0.05 ms per body). The reach ruling makes this lawful: the
  body draws its own limb.
- **Cost.** ESTIMATED 0.70 ms of a 1.20 ms sky line on the reference machine (the stunning-look plan
  §4.3). UNMEASURED on this project.

**Example.** A pilot approaches a methane world. The client library reads the body's composition class
from its seed record and hands the engine "scattering class 7, bottom radius 2,410 km, top radius
2,470 km, ground albedo 0.31, sun at 4,200 K, 31,000 lux". Bevy fills its `Atmosphere` fields and picks
the shared medium handle for class 7; Unreal fills its `SkyAtmosphere` component. The same pilot on the
same approach sees the same orange horizon on both — near the zenith to within the residual §7 item 13
still owes at the limb.

### 4.5 Planet-scale depth

The camera's near and far planes are derived from what is drawn (`lib.rs:762-786`). With terrain the
subjects change: the far plane must reach the top resident rung and the near plane must sit at the
nearest skirt. The horizon is the natural far bound on the ground (snowflow §6 item 11). Infinite
reverse-Z with a 32-bit depth buffer is close to the best available and is not the bottleneck; the
`f32` vertex position before projection is (snowflow §6 item 6), and §4.2 closes it.

---

## 5. Question 5 — The open rendering decisions of snowflow §3.x

| § | Decision | Recommended answer | Engine-dependent? |
|---|---|---|---|
| 3.1 | One voxel in metres | **1 m** — answered by the owner (decision board §3 "The one-metre cell — Answered"). Sub-metre blocks (V2.4) live INSIDE the 1 m cell and the COLLISION CELL stays 1 m. The VERTEX QUANTUM is a separate field (§1.4) and it is what lets a half-metre handle exist at all. | Agnostic. Owner already answered the cell; the quantum is open. |
| 3.2 | Blocky quads or a smooth iso-surface | **Both, by material class** — V2.1 rules smooth realistic terrain built of voxels, and V2.3 rules square building blocks with ~20 shapes. So the shared mesher has a smooth lane for natural material and a shape lane for built blocks. The smooth lane's surface IS the collision surface (SL10 §5), so it must live in the shared crate. Tier seams on an iso-surface: skirts still close a downward crack; the transvoxel family is the reserve. **The join is an open question — see the row below.** | Agnostic — it is in the shared crate. This is the mesher report's domain; this report states only that the seam carries the smooth lane's output unchanged. |
| 3.2b | **A sub-metre block on a smooth slope: which surface do the boots meet?** *(new)* A 1 m collision cell and a sub-cell iso-surface cannot BOTH be what a boot meets at the join. | **ONE CELL HAS ONE SURFACE AUTHOR, chosen by the cell's provenance bit.** A built cell is meshed by the shape lane and subdivides only INSIDE itself at the vertex quantum; a natural cell is meshed by the smooth lane. The iso-surface is evaluated with built cells forced SOLID, so the smooth surface meets the built cell at the cell face with no gap and no overlap. The sweep reads whichever author owns the cell it enters. | Agnostic. This is the mesher report's door; the seam consequence is that `ChunkGeometry` must carry the provenance bit per face so the engine can pick the material without guessing. |
| 3.3 | The vertex format | **Split it.** The SEAM carries `ChunkGeometry` in integer cell space (positions at the stated quantum, packed attributes, indices, the header with the tier, the quantum, the skirt, the part table). Each engine's encoder packs it into its own layout (Bevy: 16 B position + `u32`, the V1 of block design §5.2.1; Unreal: its own). The rotation field already landed (`realm_scene.rs:783-790`). | The seam is agnostic. The GPU layout is engine-dependent, and it is NOT a one-way door as long as the encoder stays one function per engine. |
| 3.4 | The render frame: camera-anchored, `+Y` = local up | **Eye at the origin: landed** (`lib.rs:534`). **Local up: recommend the client computes the camera basis from the delivered planet row and the eye**, as display arithmetic that derives no pose. Continuous, never a discrete rebase (snowflow §6 item 1: a discrete rebase gives one frame of garbage motion vectors). | The NEED is agnostic. The COST is engine-dependent: on Bevy the tangent frame is NECESSARY for the shipped sky and NOT SUFFICIENT (§4.4); on Unreal it is a convenience. Owner answers §6 item 2. |
| 3.5 | The pipeline set is a function of the build | **Yes.** All biome, planet and material variety is seed-derived uniform and texture data into a fixed material set. Gate: no new pipeline after warm-up across a full flight. | The principle is agnostic. The gate is per engine (Bevy `PipelineCache`; Unreal's shader permutation and PSO cache). |
| 3.6 | The three-tier mark rule (V voxel edit / S surface state / C cosmetic) | **Take it, with one amendment from SL10 §6:** V and S are DIFFS from the owning realm; C is never on the wire. The assignment test stands (authoritative if it changes collision, containment, a resource fact, or evidence a player acts on). A joint's angle is S, by that test. | Agnostic. Owner answers now. |
| 3.7 | A cosmetic-local simulation tier | **Take it,** with its four rules: a pure function of delivered state + the render cursor + a shipped seed; never on the wire, never in a checkpoint, never read by the sim, never authoritative. It is not client prediction: no pose is predicted. | Agnostic. Owner answers now. |
| 3.8 | MSAA or the temporal family | **The temporal family, on every engine, decided atomically — BUT the door may not shut until §7 item 5 runs.** Two reasons stand on measured or structural ground: the dither crossfade between rungs needs a temporal resolve (block design §5.9.5), and alpha-to-coverage foliage is foreclosed anyway (the stunning-look plan §9). The third reason — 4× multisampling dims a one-pixel star to a quarter — is ESTIMATED from coverage, not measured, and a one-way door may not open on an unmeasured number. Consequence of the choice: the still-scene pixel gates change from "zero differing pixels" to a tolerance, and the star law's floor is re-checked under the resolve. | The DIRECTION is agnostic. The MECHANISM is per engine (Bevy TAA requires `Msaa::Off` and the prepasses; Unreal has TSR). |
| 3.9 | Terrain residency rides the demand loop | **Split it, and the split is new.** The SERVER's demand loop decides which REALMS run (reach ruling, at the dot angle). The CLIENT LIBRARY's chunk residency decides which CHUNKS are resident, by the same angular form at the DRAWABLE angle, with the velocity lead taken lawfully (§3.2 shape A or B). Not a second server scheduler: the server never learns a chunk's residency. | Agnostic. Owner confirms the split. |
| 3.10 | The window/capture asymmetry | **Close it.** Both Bevy apps now hold the star cloud (`lib.rs:624` and `lib.rs:2092`). UNMEASURED whether the two paths render the same image; the bench owes a pixel diff between a windowed frame and a capture at the same tick. For Unreal the same rule: the harness captures what the player sees. | The rule is agnostic. The readback is per engine. |
| 3.11 | The content-present predicate | **Replace it before a sky lands.** A corner pixel differing from the rest saturates under a gradient sky and the gate goes vacuous. | Agnostic; test-side. |
| 3.12 | One exposure law, physical | **Yes.** EV100 closed-form from the dominant star's illuminance; a bounded rate-limited adaptation as a Tier-A function; automatic exposure only as a debug A/B the harness forces off. Every emissive becomes an absolute seed-derived radiance. | Agnostic. Both engines expose manual exposure. Owner answers now. |

**Example for 3.8.** A walker on a moon's surface looks at a ridge where rung 0 fades into rung 1.
Under multisampling the dither band is a visible speckle. Under the temporal resolve it is a smooth
blend. The same walker looks up: whether the faint stars are dimmed by the raster is exactly what §7
item 5 must read before the owner shuts this door.

---

## 6. The open decisions for the owner

1. **The seam option.** (a), (b) or (c)-narrowed. Recommended: (c)-narrowed, CONDITIONAL on SL10 §4,
   on the collision line, and on the byte-identity gate (§1.1).
2. **Who computes the camera's local up.** The client from delivered rows (recommended, display
   arithmetic), or the server authors a camera orientation (the stunning-look plan §4.4).
3. **The anti-aliasing family.** Temporal (recommended) or multisampling. Atomic — and not before §7
   item 5 runs.
4. **The exposure law.** Physical closed-form (recommended) or adaptive.
5. **The mark rule and the cosmetic-local tier** (§5 items 3.6 and 3.7). Recommended: take both.
6. **The residency split** (§5 item 3.9). Recommended: server = realms at the dot angle, client library
   = chunks at the drawable angle.
7. **The seed to the client.** SL10 requires the client to evaluate `(seed, address)`. The world seed
   (or each body's shape record) must reach the client once. See §8 ask 1.
8. **The diff reach rule.** Recommended: the LOCAL formulation of §2.3 — the realm derives fine interest
   from the poses it authors, and publishes to outsiders down to the rung drawable at its own bound.
   The alternative is §8 ask 3, one interest radius per window from the gateway.
9. **The atmosphere hand-off.** A physics-invisible swap at the noise floor plus limb shells on every
   other body's own look (recommended), or a density ramp with a re-anchor.
10. **`cbindgen` as a tool** for the C interface. A new dependency; the owner's call. Alternative: a
    hand-written header.
11. **How the client warms ahead without a predicted pose** (§3.2). (A) the realm states a warm radius
    on its existing send-on-change lane, or (B) the residency rule reads the delivered track's spread.
    Recommended: (B), because it adds nothing to the wire; either way the module fence is mandatory.
12. **The vertex quantum** (§1.4). A sixteenth of a metre or a thirty-second. Recommended: a sixteenth,
    unless the sub-metre report prices a finer detail the owner wants.
13. **Who owns the tier rule** (§1.2). Recommended: the client LIBRARY, with a counted debug override.
14. **The worker-pool mechanism** (§1.6). Recommended: `std::thread` plus `crossbeam-channel`, which
    are already present. The alternative is rayon, a new dependency.
15. **What happens to a placed block when the cell under it is mined** (§3.3). The block-record report
    owns the answer; the seam needs it to key pyramid entries by the coarse cell.

---

## 7. UNMEASURED items and their benches

Every number in the investigation base is ESTIMATED except two: the noise evaluation (12.19 ns) and the
surface chunk with coarse-lattice caves (0.885 ms), both MEASURED on an Apple M4 Pro by
`scripts/noisebench` (README). The mesher figure of 65 µs per chunk is a published number from another
crate's benchmark, not ours.

**The decision board's three numbers (§5 of the board, "none exists").**

| # | Number | Bench | What it decides |
|---|---|---|---|
| 1 | **Per-tier mesher cost** — the cost curve over every legal rung of the fixture body | `bench-mesh`: one 62³ fixture at every tier 0..tier_depth, on a planet profile and a hull profile; assert a tier-L chunk costs strictly less than tier 0 (addendum 2 §D.3) | Whether octave dropping is real; the worker pool size; the coarse-first rule's cost |
| 2 | **Sustained columns per second at flight speed** through generate + mesh + UPLOAD | `bench-stream`: fly a straight line at 5, 30, 100, 240 and 528 m/s over generated terrain; measure columns/s ingested, bytes/s uploaded, frame-time 1 % low; assert upload ≤ 50 % of the per-frame budget at the declared ceiling | The flight-speed ceiling (the high-speed flight doc's binding order: bandwidth at ~240 m/s ESTIMATED); the vertex-format deadline |
| 3 | **Resident chunks of a 100 km view WHILE MOVING** | The residency gate: ≤ 7,500 chunks INCLUDING crossfade residency, measured on a 300 m/s flyover (block design §5.9.5) | The crossfade band fraction (0.15 passes, 0.20 fails, ESTIMATED); the memory cap |

**This report's own owed measurements.**

4. **The no-drift gate across targets** (SL10 §3): server-built and client-built `ChunkGeometry` bytes
   compared on x86-64 and aarch64 at every legal tier — the digest set indexed by (seed, key, tier).
   The gate must cover the MESHER output, not only the cell field, because the collision surface is the
   mesher's. **Until it is green, "no drift" is UNMEASURED and no sentence in this report may say
   otherwise.** The gate must run with the SL10 §4 rules enforced in the crate, or it measures nothing.
5. **The star dimming under the chosen anti-aliasing**: a one-pixel star's painted luminance against the
   law's amplitude, with the star probe (`DevState.star_probe`). **This is a PRECONDITION of the
   anti-aliasing door (§9), not a follow-up.**
6. **The window-versus-capture pixel diff** at one tick.
7. **The hitch detector**: the frame-to-frame star-probe pixel delta across a recorded descent, against
   the physical tolerance of §3.3 item 7.
8. **The hand-over pixel error** from the realm's own look to its top terrain rung, at the swap distance.
9. **The walk across a tier boundary** at eye height, frame-differenced (block design §5.10.3 L3 — the
   case the owner's cited game visibly fails).
10. **The atmosphere's cost** on the reference machine (ESTIMATED 0.70 ms).
11. **The diff-lane catch-up** after a ten-second stall at a built city.
12. **The reference view.** The code's drawable angle is 45°/720 rows (`geometry.rs:1156-1173`,
    1.1506 × 10⁻³ rad); the base's ladder uses 70° horizontal over 1920 px (6.36 × 10⁻⁴ rad) and the
    stunning-look plan's angle knob is 1.27 × 10⁻³ rad. Three references. One must survive, and every
    rung range must be re-derived from it before any residency number is quoted. This report quotes no
    rung range for that reason.
13. **The residual sky error under the `+Y` flat-ray approximation** (§4.4): the painted sky at the limb
    and from orbit against a reference integration, with the tangent frame in place.
14. **The shape-record tag's bytes on the look bag** (§8 ask 1), against the 1,200-byte datagram budget,
    with the tag stated on change and never on a keep-alive.

---

## 8. Law conflicts and SL6 asks

Default NO. Each entry states the data, from which realm to which, why the receiver cannot compute it,
and what doing without costs.

1. **The world seed and each body's shape record, to the client, once.**
   - Data: the seed the generator reads, and per body its address, radius, grid family and tier depth.
   - From: the owning realm → the gateway → the client. This is NOT realm-to-realm (SL2's clarification:
     the connection plane is not a realm). It is a new TAG on the existing skip-unknown look bag
     (`look.rs:31-48`), not a new wire arm.
   - Why the receiver cannot compute it: SL10 §1 makes the client evaluate `(seed, address)`. Without the
     seed there is nothing to evaluate.
   - Cost of doing without: every rung of every planet must be streamed. A 100 km view is about 6,800
     chunks (ESTIMATED); at the V1 format that is over a gigabyte per planet per client.
   - **THE LANE COST, which the first revision did not state.** The bag rides a PER-ROW lane whose row
     count is unbounded under SL9, and that lane has already broken twice: a 1,420-byte hull window
     frame was dropped forever against a 1,200-byte budget, and the galaxy's keep-alive reached 22 MB
     (both MEASURED, `owner_decisions_2026-09-02_reach.md:200-202`, `:237-238`). **Conditions on this
     ask:** the tag is stated ONCE PER REALM ON CHANGE and carried `Retained`, copying the reach datum's
     shape exactly (`crates/sim/src/stub/reach.rs:37-64`); it never rides a keep-alive; and §7 item 14
     gates its bytes against the 1,200-byte budget before it ships.
   - The 2026-08-27 seed ruling: the seed is safe to publish for geometry; SL10 supersedes S6 for the
     static shape. No secrecy is lost that was not already lost by design.

2. **The diff lane and the pyramid entries.**
   - Data: per chunk key, the cells the seed does not decide (SL10 §6), and per coarse cell the edit
     summary; plus a joint's angle (§1.5) by the same rule.
   - From: the owning realm → the gateway → the client, on the reliable per-subscription bulk lane.
   - Why: an edit is not in the seed, and a joint's angle is not in the seed.
   - Cost of doing without: a dug tunnel vanishes at one distance and reappears at another (addendum 2
     §C.4), and a turret cannot turn.
   - SL6 also asks before adding a wire arm. `BulkKind::ChunkDelta` is a reserved name with no producer
     (`channels.rs:293`). Giving it a schema is the arm this asks for.

3. **An interest radius per open window, from the gateway to a realm — HELD IN RESERVE, not asked for
   today.**
   - Data: one scalar radius per open window.
   - From: the GATEWAY → the realm. Not realm-to-realm: SL2's clarification says the gateway is not a
     realm and alone holds both sides.
   - Why the receiver cannot compute it: a realm's shard knows where a looker is only if the looker is
     its own occupant (`WindowScope` is `Occupants` or `Child(RealmId)`,
     `crates/wire/src/session_flow.rs:582-590`; the registry is keyed by the opener,
     `crates/sim/src/stub/window.rs:38-46`), and SL2 forbids the looker's pose from entering it.
   - Cost of doing without: §2.3's local formulation covers every case except a watcher hovering just
     outside a large realm's bound, who gets a coarser picture than her eyes could resolve.
   - **Recommendation: DO NOT ASK YET.** SL6 says find the local formulation first, and §2.3 finds one.
     Take the ask only if a flight measures the residual and the owner judges it visible.
   - *DISPUTED: the law refuter (B1) makes this ask mandatory and says "do not leave §8 with two asks
     and a refusal". The evidence for the local formulation is in the code: a moon's containment bound
     is its sphere of influence (`crates/physics/src/worldgen/generate.rs:1794-1798`) while its look is
     its surface radius (`generate.rs:1819`), so every looker close enough to resolve a fine rung is
     already inside the moon and the moon authors its pose. SL6 requires the local formulation to be
     tried first. The ask stands here, fully stated, so the owner can take it in one step.*

4. **The camera's local up.** No data crosses; the client derives it from rows it already holds. It
   derives no pose and no velocity. It is listed because the stunning-look plan reads it the other way
   and the owner should rule (§6 item 2).

5. **A tree's shape (V2.2).** A seed-placed tree is a static shape function of `(seed, address)`; a
   player-placed tree is a diff. **The trunk and the main limbs ride the SHAPE lane, out of the shared
   mesher, on both hosts** — that is the collision line of §1.1 item 2 and it is what keeps the drawn
   trunk and the swept trunk one surface. Only the non-colliding decoration is engine-side. No new
   crossing.

6. **The warm-ahead datum, IF the owner picks shape (A)** (§3.2, §6 item 11). Data: one warm radius per
   realm. From: the owning realm → the gateway → the client, on the same send-on-change lane the reach
   datum rides. Why: the client may not derive a velocity (`crates/client/src/lib.rs:11-15`, SL10 §7).
   Cost of doing without: shape (B) instead, which adds nothing to the wire and measures only the past.
   **Recommended: shape (B), so this ask is not made.**

---

## 9. One-way doors

| Door | Must shut | Cost if wrong |
|---|---|---|
| **The `ChunkGeometry` seam contract** — cell-space integers, **the vertex quantum (§1.4)**, packed attributes with the provenance bit, indices, **the part-transform table (§1.5)**, the header with the tier and the skirt, **and the COLLISION LINE that says the cell lane may carry nothing a walker can hit (§1.1)** | Before the Unreal plugin starts AND before the Bevy encoder is written; both consume it | Two engine plugins rewritten; the drift gate re-baselined; every golden fixture re-pinned; V2.4 shut out by whole-metre integers; V2.2 split into two surfaces; V2.5's turret with nowhere to live |
| **The SL10 §4 boundary** — which functions are byte-compared and which are display-only (§1.2) | With the `ChunkGeometry` contract, because §4 decides what the mesher may compute at all | A mesher that calls `atan2` is green on one target and red on the other, and no gate can cure it |
| **The mesher lives in the shared crate**, so the collision surface and the display surface are one function | Before the first collider (P5) | A second surface implementation on the server; drift forever; SL10 §5 broken by construction |
| **The client-linked generator's SURFACE — shape at an address, and nothing else** *(new)* | Before the Unreal plugin links anything, because the plugin links whatever the split leaves behind | SL1 clause 4 broken: a client holding the seed and a body's address could fold that body's absolute from the root. The crate the client links must expose SHAPE AT AN ADDRESS only — no placement function, no orbit, no absolute — split out if the placement solve lives there today, with a control that FAILS when the client-linked surface grows a placement symbol (SL1 clause 5 demands exactly that form, "never by care") |
| **The detail knob is an ANGLE with ONE reference view, and the RULE lives in the library** *(extended)* | Before the first residency gate, before any rung range is quoted, and before the Unreal plugin starts | Two ladders; a residency number quoted at one reference and spent at another (the 1.78× trap of the stunning-look plan §6.1); and, if an engine picks freely, no gateable detail-by-box tolerance on an Unreal client |
| **The anti-aliasing family** | Before the first terrain pixel-gate baseline — **and NOT before §7 item 5 measures a one-pixel star's painted luminance** | Every capture baseline moves twice; the star law re-tuned twice; the dither crossfade authored under the wrong resolve; a door opened on an unmeasured number |
| **The render frame's camera basis** (eye at origin — landed; local up — open) | Before the first atmosphere shader ships | On Bevy a fork of the sky shader, or a sky wrong everywhere except one tangent point — and even with the tangent frame the along-ray approximation stays, which §7 item 13 must size |
| **The tier field on the chunk address** (five bits) | Before the first pyramid is persisted (P6) — the grid-format report's door; this report only reads it | A world-format epoch over every saved planet |

---

## 10. Stale claims in the investigation base, and what supersedes them

1. **"`PrimTransform` has no rotation; a rotating planet cannot be drawn."** (snowflow §3.3, §6 item 2;
   block design §5.2.) Superseded: the rotation field landed (D-MOVE-2, `realm_scene.rs:783-790`) and the
   renderer applies it (`lib.rs:1569`).
2. **"The wire reserves a WHAT lane at the LOD the observed realm controls plus a `coarsen_level`
   ladder."** (addendum 2 §A; `docs/design/slice_3_renderer.md`.) The ONLY `coarsen_level` in the
   workspace is "the deleted lane's legs-travelled diagnostic" on a TOMBSTONE kept so a reserved
   discriminant keeps a decodable shape; the neighbouring field is labelled "the SL2 breach that
   condemned it" (`crates/wire/src/intershard.rs:1180-1189`). So the wire carries no terrain-rung field
   at all. The decision board already says so (§7 S8).
3. **"The starfield is 2,800 unit spheres on a follow-sphere, windowed only."** (snowflow §6 item 5.)
   Superseded: one point-cloud mesh of the whole catalogue, in both apps (`lib.rs:1407-1445`,
   `lib.rs:624`, `lib.rs:2092`); the catalogue holds 233,220 stars and the watcher at the spawn drew
   them (MEASURED, reach ruling R9); the star gate measures 41,345 in view (MEASURED,
   `DEFERRED.md:1334`).
4. **"P4's premise: no chunk streaming, no networked terrain — only the seed crosses the wire."**
   (snowflow §3.6.) Superseded by SL10 §6: everything the seed does not decide crosses as a diff.
5. **"The server has no meshes."** (decision board O26.) Under SL10 §5 the server evaluates the same
   shape crate for collision; it holds the surface where physics runs.
6. **"The realm proxy is the coarsest rung"** as a parent-authored marker. (addendum 2 §C.1.) The
   parent-authored marker is deleted LAST (reach ruling R2). The coarsest rung is the realm's OWN look
   (`window.rs:485-496`), the same generator at the face-covering rung (block design §5.9.8 G1).
7. **"The tick is 20 Hz."** (every design document; decision board §6 D3.) The tick is a PER-SHARD KNOB
   (`crates/core/src/kinematics.rs:148-153`); the dev cluster runs it at 50 Hz
   (`crates/bins/src/lib.rs:412-414`, a constant named `DEV`); the client learns the rate from the wire
   (`crates/wire/src/channels.rs:130-137`).
8. **"`boot_ticks_p99` is 0, so there is no predictive horizon."** (snowflow §3.9.) The horizon is read
   at `aoi.rs:369`; whether the shipped value is non-zero is UNMEASURED in this report. The reach ruling
   R8 adds closing speed × boot time to the reach, which is the predictive horizon in its ruled form.
9. **"MANUAL exposure from a server-authored per-realm illuminance."** (decision board O20.) No new
   datum is needed: the illuminance is `L/4πd²` from the star row the client already holds — a row in
   the WINDOW, not in the ancestor chain.
10. **The three reference views** (§7 item 12): the base's 786 m rung-0 range is computed at a reference
    the code does not use. At the code's own reference a 1 m cell is drawable to about 1,505 m
    (ESTIMATED, §3.1 item 5).
11. **"A moon's visibility sphere nests inside its parent's bound."** (the base's grandchild guard,
    `visibility.rs:66-116`.) Refused by the reach ruling R3: a reach is bigger than the bound and usually
    bigger than the parent's bound. The guard's arithmetic survives as a measurement; its verdict does
    not.
12. **"A realm's colour is a fixed table keyed by what KIND it is."** (`role_hsv`,
    `crates/client/src/realm_scene.rs:719-731`.) It is a FALLBACK for a realm that states no colour, and
    the code's own note already says the drawn photometric colour rides `TAG_LUMA` "never this fallback
    family". **It retires when a realm's material comes from its own seed record.** Ledger it, because a
    client choosing a colour by realm kind is the client authoring part of a realm's look, which SL3
    gives to the realm. *(New in revision 2.)*
13. **"`FrameSpace`, `SphericalSpace` and `reanchor` do not exist anywhere."** They exist as comments and
    one test name (§0.1). No TYPE and no MODULE exists, and `crates/sim/src/capability.rs:18` says the
    seam is still owed at P5. *(New in revision 2 — a correction to revision 1's own text.)*

---

## 11. Summary of the recommended design, in the game's words

A realm draws itself, and now it draws its ground. The realm states its look once: the seed, its
address, and every diff its players made. The client library, one Rust crate that every engine links,
evaluates that statement at whatever rung the LIBRARY's one angular rule asks for, and hands the engine
a surface in cell space. The engine uploads it and never changes a vertex. Anything a walker can hit —
a hillside, a hull plate, a tree's trunk — comes out of that one mesher; only what nobody can touch is
the engine's to decorate. The realm's shard evaluates the same crate at the finest rung where a
character stands, and the character stands on what the player sees — and that sameness is a
MEASUREMENT nobody has taken yet, gated by the determinism rules the owner wrote on 2026-09-07. A
planet in the window is already a surface at a coarse rung, drawn from the seed while the hull is still
in orbit, because a planet's reach at the dot angle covers half an astronomical unit. The descent
refines it rung by rung, coarse first, on worker threads, under one angular rule that never asks what
kind of realm it is, and the client never predicts where the pilot will be. A miner's tunnel reaches
every looker who can draw it, because a realm publishes to outsiders exactly down to the rung its own
bound makes drawable, and every looker closer than that is already inside it. The sky's brightness
follows the star's physics, never a histogram. The stars are culled by brightness only, and the
galaxy's cloud is never rebuilt. Every seam kind has a physical tolerance — including the hitch, which
is now measured as a jump the eye can see and not as a ratio between two frame times — and the numbers
nobody has measured are named, with the doors that may not shut before they are read.

---

## 12. Revision log

Revision 2, 2026-09-07, after `verdicts/render_law.md` (REFUTED) and `verdicts/render_feasibility.md`
(REFUTED). Each finding, and what I did.

**From the law refutation.**

| Finding | Kind | What I did |
|---|---|---|
| B1 the diff never reaches a looker who is not on the moon | MISSING | Rewrote §2.3 entirely. Added the CODE evidence that a shard has no eye (`session_flow.rs:582-590`, `window.rs:38-46`) and then the LOCAL formulation SL6 demands: a body's bound is its sphere of influence, not its surface (`generate.rs:1794-1798` vs `:1819`), so every looker who can resolve a fine rung is already the realm's occupant; and a realm publishes to outsiders down to the rung drawable at its own bound. Added the refuter's gateway-radius ask as §8 ask 3, fully stated, held in reserve, with a DISPUTED line. |
| B2 the predicted eye breaks the no-prediction fence | BREAKS_LAW | Deleted `p + v·t_warm` from §3.2. Cited the structural fence (`interp.rs:84-101`, `lib.rs:11-15`, `interp.rs:10-13`). Put two lawful shapes to the owner (§6 item 11) and made the module dependency control mandatory. Added §8 ask 6 for shape (A). |
| B3 the engine instances a tree, and the tree has collision | BREAKS_LAW | Wrote THE COLLISION LINE into §1.1 item 2 as part of the seam contract, removed "instancing of trees" from the cell lane, put trunk and main limbs through the shared mesher, restated §8 item 5, and added the line to the `ChunkGeometry` door in §9. |
| B4 the 870 m radius is a person's number | WRONG | Replaced with a cell's number: circumscribed extent 0.866 m × the drawable factor 1,738 ≈ 1,505 m, ESTIMATED from `visibility_reach_m`. Corrected §3.1 item 5 and §10 item 10. |
| B5 the sun is not an ancestor | WRONG | `sun_set` now reads "the luminous rows IN THE WINDOW", kind-blind, with the chain spelled out and the reason a star is a sibling. Also corrected §10 item 9. |
| B6 the seam states no vertex quantum, so it shuts V2.4's door | MISSING | Added §1.4 with the `vertex_quantum` header field, both candidates priced in bits, the note that the collision cell is a different statement, and §6 item 12 as an owner decision. Added it to the §9 door. |
| B7 no fence keeps the client's generator to shape only | MISSING | Added a NEW one-way door in §9: the client-linked crate exposes shape at an address and nothing else, with a control that fails when a placement symbol appears, in the form SL1 clause 5 demands. |
| B8 "the shipped tick is 50 Hz" | UNMEASURED_AS_FACT | §0.5 now says the tick is a per-shard knob, the dev cluster runs 50 Hz, and the client learns the rate from the wire. §10 item 7 restated. |
| B9 the multisampling quarter | UNMEASURED_AS_FACT | Marked ESTIMATED from coverage in §0.3 and §3.3 item 10; the one-run observation is quoted as what was seen; §5 item 3.8 now stands on its two other reasons; §9 makes §7 item 5 a precondition of the door. |
| B10 "zero hits" is not what the grep returns | WRONG | §0.1 now lists the six comment/test hits and says no type and no module exists, citing `capability.rs:18`. Also §10 item 13. |
| B11 a tick-hitch tolerance must be physical | BREAKS_LAW (SL8) | §3.3 item 7's tolerance is now a jump the eye can see: the picture's angular motion exceeding a constant-rate frame by more than the drawable angle, with the permitted extra frame time `θ/ω`, and the star probe named as the instrument that measures `ω`. |
| B12 a realm's colour still comes from its kind | MISSING | Ledgered as §10 item 12 and named in §3.3 item 4: `role_hsv` is a fallback and retires when a realm's material comes from its seed record. |
| B13 `coarsen_level` mis-read | WRONG (minor) | §10 item 2 now describes it as a tombstone's diagnostic on a deleted lane, with the neighbouring field's own label. |

**From the feasibility refutation.**

| Finding | Kind | What I did |
|---|---|---|
| F1 SL10 §4 never appears | WRONG (load-bearing) | Added a §4 row to the §1.1 table; made the recommendation CONDITIONAL and deleted "Holds"; split §1.2's function set into §4-BOUND and §4-FREE with the reason; added the §4 boundary as a one-way door; §7 item 4 now says the gate must run with §4 enforced. |
| F2 option (a) refused for a reason the cell lane commits | BREAKS_LAW | Same fix as B3, above. |
| F3 the multisampling number | UNMEASURED_AS_FACT | Same fix as B9, above, including the precondition on the door. |
| F4 the realm's reach is read off the wrong angle | WRONG | §0.5 now names the two angles apart with their jobs and factors, and every later sentence names which one it uses. §3.1 item 1 and §3.2 use the DOT angle for a realm's reach; §3.1 item 5, §3.3 items 6 and 7, and §4.2 use the DRAWABLE angle. |
| F5 a rotation joint and a turret have no seam | MISSING | Added §1.5: the angle is live state and rides the diff lane on change; `ChunkGeometry` carries a part-transform table; the engine composes row → part → chunk; the cost is stated; the door in §9 now carries it. §3.3 item 8 extended. |
| F6 the Bevy sky traps are relayed from an older Bevy | MISSING | Re-verified all three against the vendored 0.18.1 source and rewrote §4.4. The Earth radii are in a named constructor with public fields; the 32 km range is a public settings default. **DISPUTED on the Mie half:** the transposition is live at `medium.rs:143-148`, not a comment. |
| F7 the `+Y` cure is too strong | MISSING | §4.4 now says the tangent frame is NECESSARY and NOT SUFFICIENT, quotes the shader's own note and the along-ray approximation, and owes §7 item 13. §5 item 3.4 and the §9 door restated. |
| F8 the worker pool names no mechanism | MISSING | Added §1.6: `std::thread` plus `crossbeam-channel` (already a workspace dependency), rayon listed as the owner's alternative, and the injected-seam pattern that reaches 100 % branch. §6 item 14. |
| F9 the new look-bag tag is not costed | MISSING | §8 ask 1 now carries the lane cost with both measured breakages, and three conditions: stated once per realm on change, carried `Retained` copying `reach.rs:37-64`, never on a keep-alive. §7 item 14 gates the bytes. §3.3 item 11 restated. |
| F10 who owns the tier rule | MISSING | `chunk_tier_for` moved into the library (§1.2, §2.1), engine override only under a debug flag and counted in `DevState`; the detail-by-box tolerance in §3.3 item 4 now says it is gateable only because of that; a new door in §9. |
| F11 a sub-metre block on a smooth slope has two collision surfaces | MISSING | Added §5 row 3.2b: one cell has one surface author, chosen by the provenance bit; built cells force the iso-surface solid so the two meet at the cell face; `ChunkGeometry` carries the provenance bit per face. |
| F12 two seam cases not covered | MISSING | Added both at the end of §3.3: a diff arriving during a crossing (with a fixture tolerance), and a mined cell under a placed block (with the seam consequence that a pyramid entry is keyed by the coarse cell). |
| F13 small corrections | WRONG (minor) | 233,220 re-attributed to the catalogue and the watcher, with 41,345 as the measured drawn-in-view figure (§0.3, §10 item 3). `RenderAssetBytesPerFrame` default is `None`, so the client must set it (§3.3 item 7). No rung range is quoted anywhere until §7 item 12 settles the reference view (§3.2). The 505 columns and the 1,900 quads are marked ESTIMATED (§1.2, §1.1). §3.3 item 2 now says the hand-over pattern is BORROWED, because that code swaps a realm's body and chunks do not exist. |
