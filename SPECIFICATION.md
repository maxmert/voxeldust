# Voxeldust: Complete Game Specification

**"Star Citizen meets Minecraft"** — A multiplayer voxel game with procedurally generated spherical planets, player-built ships, Newtonian space physics, and a distributed shard architecture.

---

## 1. Vision

Players explore a procedurally generated universe of spherical voxel planets orbiting stars. They mine resources, build structures on planet surfaces, construct ships from blocks, fly between planets using Newtonian thrust, and engage in space combat. The entire universe is deterministic — the same seed always produces the same world — so clients and servers generate terrain independently, and only player-made changes (block edits) need to be networked.

### Core Pillars

1. **Spherical voxel planets** — Not flat worlds. Gravity pulls toward planet center. Players walk around the entire sphere. Planets are Earth-scale (~100k–200k block radius).
2. **Deterministic procedural generation** — Universe seed → star system seeds → planet seeds. All terrain, biomes, geology, and vegetation derived from seeds. No hand-placed content.
3. **Player-built ships** — Ships are voxel structures built block-by-block. Functional blocks (thrusters, cockpits, power generators) give ships behavior. Players walk inside ships while flying through space.
4. **Newtonian space physics** — Ships and players in space experience real gravity from celestial bodies. Brachistochrone trajectories for interplanetary travel. No magic teleportation.
5. **Distributed shard architecture** — The universe is divided across specialized server processes (shards) that communicate via QUIC. Players seamlessly hand off between shards as they cross boundaries.
6. **Server-authoritative** — The server owns all physics and game state. Clients predict locally for responsiveness but defer to server corrections.

---

## 2. World Model

### 2.1 Universe Hierarchy

```
Universe (universe_seed)
└── Star System (system_seed = hash(universe_seed, system_index))
    ├── Star (at origin)
    └── Planet (planet_seed = hash(system_seed, planet_index))
        ├── Terrain (from planet_seed)
        ├── Biomes (from planet_seed + climate model)
        └── Geology & vegetation (from planet_seed + depth/position)
```

Every parameter at every level is derived from seeds via deterministic hash functions. Nothing is random at runtime — given the same seed, the same world is produced on any machine.

### 2.2 Coordinate System: Cubic Sphere

Planets use a **quad-sphere** (cube-sphere) coordinate system:

- **6 cube faces** (sectors 0–5) are projected onto a sphere using **Nowell mapping** (a tangent-based projection that distributes distortion more evenly than simple normalization).
- Each face is subdivided into a grid of **chunks**.
- **Radial shells** stack chunks from the planet core outward.

#### Chunk Address
```
ChunkAddress {
    sector: u8,    // 0–5 (cube face)
    shell: u16,    // radial layer (0 = core, increasing outward)
    cx: u16,       // tangential grid X on face
    cy: u16,       // tangential grid Y on face
    cz: u16,       // Z within shell (usually 0 for surface)
}
```

#### Block Address
Within a chunk: `(bx: u8, by: u8, bz: u8)` — each 0–61.

#### World Position
All positions are `DVec3` (f64) in a planet-centered Cartesian frame. The planet center is at origin (0, 0, 0). A player standing on the surface at the "north pole" of sector 1 might be at position (0, 150000, 0).

### 2.3 Chunk Size: 62³

Chunks are **62×62×62 blocks** (not a power of 2). This is dictated by the `binary-greedy-meshing` crate, which operates on 64³ padded grids internally — 62 usable voxels + 1 padding on each side for neighbor lookups.

### 2.4 Planet Parameters

All derived from `PlanetParams::from_seed(planet_seed)`:

| Parameter | Range | Derivation |
|-----------|-------|------------|
| radius_in_blocks | 100,000 – 200,000 | Hash of seed |
| density | 2,000 – 6,000 kg/m³ | Hash of seed |
| planet_type | 8 types (see below) | Hash of seed |
| terrain_scale | 0.005 – 0.05 | From type + seed |
| terrain_amplitude | 3.0 – 8.0 | From type + seed |
| water_level_offset | -10 to +10 blocks | From seed |
| gravity | G × M / r² | Derived from density + radius |

**Planet Types**: Earthlike, Desert, Ice, Volcanic, Ocean, Barren, Exotic, Dusty — each has distinct terrain profiles, spline curves, biome distributions, and block palettes.

### 2.5 Block System

- **BlockId**: `u16` newtype. Supports up to 65,535 block types.
- **Block registry**: Static array of `BlockDef` structs, indexed by ID.
- **Block size**: 1 block = 1 meter.

#### Block Properties (BlockDef)
```
name, is_solid, is_transparent, color_hint
hardness, mining_tier, preferred_tool, blast_resistance
density, gravity_affected
flammability, fire_resistance, light_emission
liquid_properties (optional): gravity_multiplier, drag, swim_force
material_type: Stone | Soil | Wood | Metal | Glass | Organic | Fluid | Crystal | Ice
```

#### Current Block Types (~55 defined)
- **Basic** (0–5): Air, Stone, Dirt, Grass, Sand, Water
- **Stone variants** (6–13): Granite, Basalt, Limestone, Marble, Slate, Sandstone, Obsidian, Pumice
- **Soil/Surface** (20–25): Clay, Mud, Gravel, Peat, Red Sand, Dark Soil
- **Ice/Snow** (30–33): Snow, Packed Ice, Blue Ice, Permafrost
- **Volcanic** (40–42): Magma Rock, Ash, Sulfur
- **Ores** (100–111): Coal, Iron, Copper, Gold, Diamond, Emerald, Ruby, Sapphire, Uranium, Titanium, Mythril, Energy Crystal
- **Fluids** (200–202): Lava, Salt Water, Oil
- **Vegetation** (300–311): Oak/Pine/Jungle/Birch/Acacia logs and leaves, Tall Grass, Moss, Cactus, Dead Bush

#### Future: Functional Blocks
Thrusters, cockpits, power generators, gravity generators, shield emitters, weapons — these give player-built ships their behavior.

---

## 3. Terrain Generation

### 3.1 Architecture

Terrain generation lives in `core/src/terrain/` and is called identically by both server and client — determinism is essential.

#### Noise Stack
Multiple layers of OpenSimplex noise (pinned to exact crate version for cross-platform determinism):

| Noise | Purpose | Scale |
|-------|---------|-------|
| Continent | Large landmass vs. ocean | radius × √2 frequency |
| Erosion | Mountain/valley detail | Smaller scale |
| Peaks & Valleys (PV) | Fine terrain character | Smallest scale |
| Weirdness | Regional variation | Medium |
| Temperature | Climate zones | Latitude + altitude + noise |
| Humidity | Moisture | Noise + distance from water |
| Biome detail | Per-biome variation | Fine |
| River | Erosion features | Medium |
| Vegetation density | Clearings and groves | Fine |

**Critical**: All noise scales use a `REFERENCE_RADIUS = 256` constant so that angular frequency stays consistent regardless of planet size. This means terrain looks similar whether the planet has radius 100k or 200k.

### 3.2 Biome System

20 biome types selected by climate (temperature × humidity × weirdness):

Plains, Forest, Dense Forest, Jungle, Desert, Sandy Desert, Tundra, Frozen Tundra, Swamp, Deep Swamp, Mesa, Badlands, Volcanic, Volcanic Wasteland, Ocean, Deep Ocean, Mountain, Alpine, Exotic, Barren.

Each biome has:
- A **shaper** that controls terrain height profile
- Surface/subsurface/deep block palettes
- Vegetation rules
- Height bounds (for LOD culling)

### 3.3 Per-Biome Shapers

Each biome has a dedicated terrain shaper (not trait objects — zero-cost enum dispatch via `ShaperKind`). Each shaper implements:
- `shape_height(continent, erosion, pv, weirdness) → height` — the terrain elevation formula
- `select_block(depth, height, context) → Option<BlockId>` — what block goes where
- `height_bounds() → (min, max)` — for LOD early-exit

Per-planet-type **cubic Hermite splines** control how continent/erosion noise values map to terrain height. Volcanic planets have dramatic splines with sharp peaks; Ocean planets have mostly flat splines with deep basins.

### 3.4 Geology

Below the biome surface layer:
- **Shallow zone** (0–20 blocks deep): Biome-specific stone variants
- **Mid zone** (20–100 blocks): Mixed stone types
- **Deep zone** (100+ blocks): Dense stone, ore deposits

**Ore placement**: Noise-based probability per ore type, constrained by depth range and biome compatibility.

**Cave systems**: Cheese caves (large chambers) + spaghetti caves (narrow tunnels), generated via 3D noise thresholding.

### 3.5 Vegetation

- **Structure placement**: Deterministic grid (24-block cells) with hash-based probability per cell
- **Trees**: Procedural recursive branch graphs with species-specific configs (trunk height, branch angles, leaf density). Cross-chunk generation supported.
- **Ground cover**: Grass, moss, flowers placed on valid surface blocks

### 3.6 LOD Terrain Generation

5 active LOD tiers for distant terrain rendering:

| Tier | Stride | Render Radius (chunks) |
|------|--------|----------------------|
| LOD0 | 1 | 5 |
| LOD1 | 2 | 10 |
| LOD2 | 4 | 20 |
| LOD3 | 8 | 40 |
| LOD4 | 16 | 80 |

Each LOD chunk is still 62³ cells, but each cell represents `stride³` real blocks. The terrain generator samples the center of each LOD cell.

LOD chunks must have aligned coordinates: `cx/cy/shell` divisible by stride.

**Inner coverage**: Each LOD tier skips cells that are covered by finer tiers, avoiding overdraw.

---

## 4. Meshing

### 4.1 Binary Greedy Meshing

Uses the `binary-greedy-meshing` crate:
- Input: 64³ padded voxel buffer (ZXY layout: `index = (z+1) + (x+1)*64 + (y+1)*4096`)
- Output: Greedy-merged quads (adjacent same-type faces merged into larger rectangles)
- Performance: ~65μs per chunk
- The crate hardcodes ID 0 as always-passable; remap air/water to 0 for the transparent set

### 4.2 Vertex Format
```
position: [f32; 3]     // chunk-local [0..62]
normal: [f32; 3]       // face normal (one of 6 axis-aligned directions)
uv: [f32; 2]           // quad-local texture coordinates
block_type: u32         // block ID for color lookup
```

### 4.3 Sphere Projection

Meshing happens in **flat chunk-local space** [0, 62]. After meshing, vertices are **sphere-projected on CPU** before GPU upload:

1. Convert chunk-local position to cube-face UV coordinates
2. Apply Nowell mapping to get sphere direction
3. Scale by radial distance (shell height)

This is done on CPU (not in shader) because the projection is nonlinear and must match between client rendering and server collision.

### 4.4 Collision Meshes

Server builds Rapier3D `TriMesh` colliders from the same sphere-projected vertices. Only solid (non-transparent) blocks contribute to collision geometry. Shared shape handles minimize memory.

### 4.5 Water Border Patching

Water faces at chunk boundaries that would create artificial walls (where the adjacent chunk hasn't loaded yet) are suppressed. The mesh splits opaque and water geometry into separate index ranges for separate render passes.

---

## 5. Physics

### 5.1 Server-Side Only

All physics runs on the server via **Rapier3D**. The client does lightweight prediction (`physics_lite.rs`) for responsiveness but always defers to server state.

### 5.2 Character Controller

- **Rapier KinematicCharacterController** for player movement
- Player capsule: height 1.8 blocks, radius 0.3 blocks
- Per-player `up` vector (points away from planet center for spherical gravity)
- Ground detection via KCC sweep results

### 5.3 Movement Constants

| Parameter | Value |
|-----------|-------|
| Walk speed | 8.0 blocks/sec |
| Jump velocity | 7.0 blocks/sec |
| Fly speeds (4 tiers) | 20, 200, 2,000, 20,000 blocks/sec |
| Gravity | G × M / r² (per planet) |

### 5.4 Spherical Gravity

On planets: `g = G × M / r²` directed toward planet center, where M = density × (4/3)π r³. Each player has their own `up` vector = `normalize(position)`.

### 5.5 Space Physics

In system shards (space):
- **N-body gravity**: Brute-force summation of gravitational acceleration from all celestial bodies: `a = Σ GM_i / |r_i|² × dir_i`
- **Integration**: Velocity Verlet (symplectic, energy-conserving)
- **Thrust**: Player/ship input applies force in look direction

### 5.6 Liquid Physics

Water and other fluid blocks modify player physics:
- Reduced gravity (gravity_multiplier per fluid type)
- Drag force opposing velocity
- Swim force (upward when holding jump)

### 5.7 Client-Side Prediction

`core/src/physics_lite.rs` provides lightweight prediction without Rapier:
- `predict_fly()`: Pure math matching server fly physics
- `predict_ground()`: Simplified gravity + ground check
- `reconcile()`: Smooth correction (exponential lerp below 0.1 block threshold, hard snap above)
- Input history: Ring buffer of 60 ticks for replay-based reconciliation

---

## 6. Networking

### 6.1 Protocol: FlatBuffers

All messages serialized with FlatBuffers (`protocol/voxeldust.fbs`). Two root types:
- **ServerMessage**: Client ↔ server communication
- **ShardMessage**: Inter-shard communication

### 6.2 Client ↔ Server Transport

| Channel | Protocol | Port | Purpose |
|---------|----------|------|---------|
| Reliable | TCP | 7777 | Join, block edits, damage events, redirects |
| Fast | UDP | 7778 | Player input (client→server), world state (server→client) |

**Max message size**: 65,536 bytes.

### 6.3 Client → Server Messages

| Message | Channel | Content |
|---------|---------|---------|
| Connect | TCP | player_name |
| PlayerInput | UDP | movement, look, jump, action, fly_toggle, speed_tier, autopilot_target |
| BlockEditRequest | TCP | action (break/place), eye position, look direction, block_type |
| ChunkEditResyncRequest | TCP | chunk address + last_known_seq (for catching up missed edits) |

### 6.4 Server → Client Messages

| Message | Channel | Content |
|---------|---------|---------|
| JoinResponse | TCP | seed, planet_radius, player_id, spawn_pos/rot/forward |
| WorldState | UDP | tick, origin, player snapshots[], projectile snapshots[] |
| ChunkBlockMods | TCP | chunk address, seq number, block modifications[] |
| DamageEvent | TCP | target_id, source_id, damage, weapon_type |
| PlayerDestroyed | TCP | killer_id, position |
| AutopilotState | TCP | trajectory waypoints, phase |

### 6.5 No Chunk Streaming

Because terrain generation is deterministic, chunks are **never** sent over the network. Both client and server generate the same terrain from the same seed. Only block **edits** (player modifications) are networked, using monotonic per-chunk sequence numbers for causality.

### 6.6 Bandwidth Optimization

- **Bitpacked WorldState**: Custom binary format (not FlatBuffers) for UDP position updates. 30-byte header + 9–18 bytes per player. Smallest-three quaternion encoding for rotations.
- **Delta compression**: Per-client-per-player tracking. Skip players whose position/rotation hasn't changed beyond threshold.
- **Tiered Area of Interest (AOI)**: 3 distance tiers:
  - Full (0–100 blocks): 20 Hz updates
  - Reduced (100–500 blocks): 10 Hz
  - Coarse (500–2000 blocks): 4 Hz
  - 50-player cap per message, distance-sorted priority

---

## 7. Shard Architecture

### 7.1 Overview

The game world is divided across specialized server processes called **shards**. Each shard runs its own ECS world (bevy_ecs), physics simulation, and networking stack. Shards communicate via QUIC (quinn crate).

```
Orchestrator (1 per universe)
├── Gateway (1+ per cluster, client entry point)
├── System Shard (1 per star system)
│   ├── Orbital mechanics, space physics, combat
│   └── Transit detection (enter/leave planet SOI)
├── Planet Shard (1-N per planet, split by sector)
│   ├── Terrain, surface physics, block edits
│   └── Boundary detection (sector transitions)
└── Ship Shard (1 per active ship)
    ├── Ship interior physics (flat Cartesian)
    └── Exterior position sync with system shard
```

### 7.2 Shard Types

#### Planet Shard
- Manages a planet's surface (or a subset of its sectors)
- Runs Rapier3D with spherical gravity
- Loads/unloads terrain chunks based on player proximity
- Handles block edits with WAL persistence
- Detects when players cross sector boundaries for handoff

**Config flags**: `--seed`, `--planet-type`, `--sectors 0,1,2,3,4,5`, `--db`, `--shard-id`, `--orchestrator`, `--tcp-port`, `--udp-port`

#### System Shard
- Manages a star system's space (everything between planets)
- Full Keplerian orbital mechanics (Newton-Raphson eccentric anomaly solver)
- Newtonian space physics (Velocity Verlet integration)
- Deterministic system generation: star + planets with orbital elements from seed
- Transit detection: captures players entering planet sphere-of-influence
- Combat: weapons (Rocket, PDC, Railgun), projectile physics, damage, death/respawn
- Autopilot: brachistochrone trajectory solver for interplanetary travel

**Config flags**: `--system-seed`, `--shard-id`, `--orchestrator`, `--tcp-port 9777`, `--udp-port 9778`

#### Ship Shard
- Manages one ship's interior
- Flat Cartesian block grid (not spherical)
- Interior KCC physics with configurable gravity
- Syncs exterior position with system shard
- Boarding mechanics (player enters/exits ship)

**Config flags**: `--ship-id`, `--orchestrator`, `--host-shard`, `--db`

#### Orchestrator
- Central registry of all shards
- HTTP API for shard discovery and routing
- UDP heartbeat listener for health monitoring
- Scaling decisions (split/merge shards based on load)
- Shard provisioning (starts new shard processes)
- Persistence (redb) for crash recovery

**HTTP API**:
- `POST /register` — Shard heartbeat + registration
- `GET /shards` — List all shards
- `GET /shard/{id}` — Lookup specific shard
- `GET /planet/{seed}` — Find shards serving a planet
- `GET /system/{seed}` — Find shard for a system

**Heartbeat data**: tick_ms, p99_tick_ms, player_count, chunk_count

#### Gateway
- Client entry point
- Receives Connect message, queries orchestrator for routing
- Returns ShardRedirect (target shard TCP/UDP endpoints + session token)
- Session persistence (redb) for handoff resumption

### 7.3 Inter-Shard Communication (QUIC)

Shards communicate via **QUIC** (quinn crate) with self-signed certificates:

| Message | Purpose |
|---------|---------|
| PlayerHandoff | Full player state transfer between shards |
| HandoffAccepted | Confirmation + new shard ID |
| GhostUpdate | Replicate player position to adjacent shard (15 frames) |
| CrossShardBlockEdits | Block mods affecting adjacent chunk boundaries |
| ShipPositionUpdate | Ship orbital state sync |
| ShardHeartbeat | Health report to orchestrator |
| SplitDirective | Orchestrator tells shard to divide sectors |
| MergeDirective | Orchestrator tells shard to combine |

### 7.4 Player Handoff Protocol

When a player crosses a shard boundary:

1. **Boundary detection** — Source shard detects player near sector edge
2. **Peer lookup** — Query `PeerShardRegistry` for target shard (cached from orchestrator)
3. **Handoff send** — Serialize full `PlayerHandoff` (position, velocity, rotation, health, shield, fly mode, speed tier) via QUIC
4. **Pre-spawn** — Target shard creates entity with `HandoffPending` component, awaiting client
5. **HandoffAccepted** — Target confirms receipt back to source
6. **Ghost replication** — Source continues broadcasting player as ghost to target for smooth transition
7. **Client redirect** — Source sends `ShardRedirect` to client via TCP
8. **Client reconnects** — Client drops old connection, connects to target shard
9. **Resume** — Target shard matches reconnecting client to pre-spawned entity by player name
10. **Cleanup** — Source despawns player entity

### 7.5 Tick Rate

All shards run at **20 Hz** (50ms ticks). Each system runs in its own schedule for per-system profiling.

---

## 8. Persistence

### 8.1 Storage Engine: redb

Embedded key-value store (no external database dependency). Each shard has its own database file.

### 8.2 Tables

| Table | Key | Value | Purpose |
|-------|-----|-------|---------|
| chunk_edits | ChunkAddress bytes | Block edit list | Accumulated edits per chunk |
| chunk_snapshots | ChunkAddress bytes | Full 62³ × 2 bytes | Complete chunk state |
| edit_wal | Auto-increment u64 | Chunk address + edits | Write-ahead log for crash recovery |
| player_state | session_id u64 | Player checkpoint | Position, health, inventory |
| handoff_out | session_id u64 | Handoff state | In-progress outbound handoffs |
| handoff_in | session_id u64 | Handoff state | In-progress inbound handoffs |

### 8.3 Edit Format

5 bytes per block edit: `(bx: u8, by: u8, bz: u8, block_type_lo: u8, block_type_hi: u8)`

### 8.4 WAL + Snapshot Strategy

1. Block edits written synchronously to WAL (zero data loss)
2. WAL entries applied asynchronously to edit table
3. Periodic snapshot compaction (every 1200 ticks / 1 minute): full chunk state replaces accumulated edits
4. On crash recovery: load latest snapshot, replay WAL entries after it

### 8.5 Player Checkpoints

Every 20 ticks (1 second), all player state is checkpointed. On reconnect, the server restores from the latest checkpoint.

### 8.6 Graceful Shutdown

On SIGTERM: drain phase (200 tick timeout) → final checkpoint → persistence flush → orchestrator deregister.

---

## 9. Client (Voxygen)

### 9.1 Technology Stack

| Component | Technology |
|-----------|-----------|
| Rendering | wgpu 27 (GPU-agnostic, Vulkan/Metal/DX12) |
| Windowing | winit 0.30 |
| UI | egui 0.33 (immediate mode) |
| Math | glam 0.29 (SIMD) |
| Serialization | flatbuffers |
| Async I/O | tokio (for networking) |
| Worker threads | crossbeam-channel |
| Terrain/meshing | voxeldust-core (direct Rust dependency) |

### 9.2 Rendering Architecture

#### Floating Origin (Camera-Relative Rendering)

At planet scale (100k+ block radius), f32 loses precision. Solution:
- Camera stores position as `DVec3` (f64)
- View matrix is built with camera at origin: `look_to_rh(Vec3::ZERO, forward, up)`
- Each chunk stores its f64 world origin
- At render time: `offset = (chunk_origin - camera_pos) as Vec3` (f32, small values)
- Push constants pass per-draw `chunk_offset` to shaders
- All shaders compute positions relative to camera (camera is always at origin)

#### Render Pipelines

| Pipeline | Purpose | Topology | Blend |
|----------|---------|----------|-------|
| Terrain | Opaque voxel terrain | TriangleList | Opaque |
| Water | Transparent water surfaces | TriangleList | Dithered alpha |
| Far Terrain | Planet sphere at orbital distance | TriangleList | Opaque, near-discard |
| Highlight | Block selection wireframe | LineList | Opaque |
| Atmosphere | Sky dome (WIP) | TriangleList | Blend |

#### Terrain Shader
- Block colors: 312-entry lookup table indexed by `block_type`
- Flat face normals (6 axis-aligned directions)
- Directional lighting
- Distance fog (fully fogged beyond 20,000 blocks)
- LOD depth bias: coarser LODs pushed further in depth so LOD0 always wins z-test
- LOD fade: Bayer 4×4 dithering pattern for smooth transitions

#### Planet Sphere
- Shown only at orbital altitude (camera_altitude > 30% planet_radius)
- 256×256 heightmap per cube face, bilinearly interpolated
- Sphere-projected vertices with per-vertex block_type for coloring
- Near-distance discard in shader prevents z-fighting with terrain chunks

### 9.3 Chunk Manager

Manages async chunk loading/unloading with worker threads:

- **Load budget**: 32 dispatch / 16 mesh completions per frame
- **Per-LOD in-flight limits**: [24, 16, 12, 32, 32, 24, 24, 24]
- **Unload hysteresis**: 2.5× load distance (prevents oscillation at boundaries)
- **Max loaded chunks**: 20,000 (~6 GB RAM cap at ~300 KB/chunk)
- **Voxel cache**: LOD0 chunk data cached for block interaction raycasting
- **Edit chunks**: Modified chunks stored separately, bypass terrain gen on reload

### 9.4 Camera Modes

- **SphericalGravity**: Up = normalize(position). Yaw/pitch relative to tangent plane at current position. Used on planet surfaces.
- **FreeSpace**: Full 6DOF quaternion-based orientation. Used in space.

FOV: 70°. Near plane: 0.1. Far plane: 500,000 (dynamically extended for orbital view).

### 9.5 Block Interaction

- **Raycast**: Fixed-step marching (0.4-block steps, 8-block range) in `core/src/raycast.rs`
- **Break**: LMB removes targeted block
- **Place**: RMB places selected block type on adjacent face
- **Block selection**: Keys 1–5 select Stone/Dirt/Grass/Sand/Water
- **Highlight**: Wireframe cube rendered on targeted block

In standalone mode, edits apply locally. In multiplayer, client sends `BlockEditRequest` (eye + look + action); server raycasts authoritatively and broadcasts `ChunkBlockMods`.

### 9.6 Space Mode

When connected to a system shard:
- 6DOF camera (FreeSpace mode)
- 4 speed tiers (20 / 200 / 2,000 / 20,000 blocks/sec)
- Weapon input: LMB = Rocket, RMB = PDC, MMB = Railgun
- Autopilot toggle (T key): cycles through celestial body targets
- Projectile rendering on screen

### 9.7 UI (egui)

Debug overlay showing:
- World position (f64)
- Current biome
- Server address + shard type
- Health/shield bars
- FPS and frame times
- Kill feed
- Autopilot indicator
- Death screen with respawn timer

---

## 10. Combat System

### 10.1 Weapons

| Weapon | Type | Damage | Cooldown | Speed | Range |
|--------|------|--------|----------|-------|-------|
| Rocket | Guided missile | High | Long | Medium | Long (guided) |
| PDC | Point defense cannon | Low | Short | Fast | Medium (linear) |
| Railgun | Hitscan beam | Very high | Very long | Instant | Infinite |

### 10.2 Mechanics

- **Health**: 100 HP, no natural regen
- **Shield**: 100 SP, regenerates after delay (SHIELD_REGEN_DELAY)
- **Damage order**: Shield absorbs first, then health
- **Splash damage**: Rockets have splash radius
- **PDC intercept**: PDC rounds can shoot down incoming rockets
- **Death**: Player marked with DeathTimer, respawns at SOI edge after RESPAWN_DELAY
- **Kill attribution**: Killer ID tracked for kill feed

### 10.3 Autopilot

Brachistochrone trajectory solver (`core/src/trajectory.rs`):
- Iterative intercept computation accounting for target body's orbital motion
- Two phases: Accelerating (thrust toward intercept point) → Decelerating (flip and burn)
- Re-solves intercept every 100 ticks to track moving targets
- Phases: Accelerating → Decelerating → Arrived

---

## 11. Orbital Mechanics

### 11.1 Celestial Bodies

Each star system is generated deterministically from `system_seed`:
- **Star** at origin (stationary, `sma = 0`)
- **Planets** with full Keplerian orbital elements:
  - Semi-major axis, eccentricity, period
  - Inclination, longitude of ascending node, argument of periapsis
  - Gravitational parameter GM

### 11.2 Orbit Computation

1. Compute mean anomaly: `M = 2π × (t mod period) / period`
2. Solve Kepler's equation `M = E - e × sin(E)` via Newton-Raphson iteration
3. Convert eccentric anomaly E to true anomaly
4. Transform orbital plane coordinates to 3D via rotation matrices (Ω, i, ω)

### 11.3 Gravity

Brute-force N-body: `a = Σ GM_i / |r_i - r_player|² × normalize(r_i - r_player)`

Optimal for < 10 bodies per system. No Barnes-Hut needed.

### 11.4 Transit Detection

When a player's distance to a celestial body falls below its capture radius, the system shard initiates a handoff to the corresponding planet shard.

---

## 12. ECS Patterns (bevy_ecs)

The project uses `bevy_ecs` (not full Bevy engine) with specific conventions:

1. **Events over manual queues**: Use `bevy_ecs::event::Events<T>` for data flow between systems. For async→sync bridges (tokio → ECS), use a dedicated bridge system that drains mpsc into `EventWriter<T>`.

2. **Entity-scoped state**: State belonging to an entity is a Component on that entity, not an entry in a HashMap inside a Resource.

3. **Split resources**: If different systems touch disjoint fields, split into separate resources for finer-grained borrow checking.

4. **Automatic index sync**: Use `Added<T>` / `RemovedComponents<T>` change detection for index resources (like ClientMap) instead of manual insert/remove.

5. **Per-system schedules**: Each system runs in its own schedule for per-system timing at 20 Hz. Do not consolidate without replacing the profiling.

6. **apply_deferred placement**: Always place explicit `apply_deferred` between systems that spawn/despawn entities and systems that query those entities.

---

## 13. Resilience & Operations

### 13.1 Circuit Breaker

Per-peer circuit breaker for QUIC connections:
- 5 failure threshold
- Exponential backoff: 5s → 60s
- 5-second timeout on connect and send

### 13.2 Orchestrator Auto-Restart

Health check detects dead shards → looks up stored launch config → calls `start_shard()`. Max 3 restarts per 60 seconds per shard.

### 13.3 Monitoring

Shard heartbeats include:
- `tick_ms`: Last tick duration
- `p99_tick_ms`: 99th percentile tick duration
- `player_count`: Active players
- `chunk_count`: Loaded chunks (planet shards)

---

## 14. Technology Choices & Rationale

| Choice | Why |
|--------|-----|
| **Rust** | Memory safety without GC, predictable latency for game loops, shared core crate between server and client |
| **bevy_ecs (not full Bevy)** | Need ECS without the engine's renderer, asset system, etc. Minimal dependency footprint |
| **wgpu** | Cross-platform GPU API (Vulkan/Metal/DX12/WebGPU). No OpenGL legacy |
| **Rapier3D** | Mature Rust physics with KCC support. No C++ FFI needed |
| **FlatBuffers** | Zero-copy deserialization, schema evolution, compact binary. Faster than Protobuf for game networking |
| **QUIC (quinn)** | Multiplexed streams, built-in encryption, connection migration. Ideal for inter-shard |
| **redb** | Embedded, no external process, ACID transactions, pure Rust. Simple persistence for game state |
| **Noise crate (pinned)** | Deterministic cross-platform generation. Exact version pin prevents silent breakage |
| **binary-greedy-meshing** | Fastest known voxel meshing algorithm (~65μs/chunk). Drives the 62³ chunk size |
| **glam** | SIMD-accelerated math. Standard in Rust gamedev |
| **tokio** | Async runtime for TCP/UDP networking. De facto standard |
| **Nowell mapping** | Better distortion distribution than normalize() for cube→sphere. Less area distortion at face corners |
| **f64 world coordinates** | f32 loses sub-meter precision beyond ~16km. Planets are 100k+ blocks |

---

## 15. Build & Run

```bash
# Generate FlatBuffers code from schema
./build_protocol.sh

# Run tests
cargo test --workspace           # All tests (~192)
cargo test -p voxeldust-core     # Core only

# Run standalone client (no server)
cargo run -p voxygen

# Run multiplayer
cargo run -p voxeldust-server                              # Legacy monolith
cargo run -p planet-shard -- --seed 42 --sectors 0,1,2,3,4,5  # Planet shard
cargo run -p system-shard -- --system-seed 42              # System shard
cargo run -p orchestrator                                   # Orchestrator
cargo run -p gateway                                        # Gateway
cargo run -p voxygen -- --server 127.0.0.1:7777            # Client → server
```

### Dev Profile Optimization

The workspace `Cargo.toml` has `[profile.dev.package.*]` overrides for expensive crates:
- `noise`: opt-level=3 (40× speedup for terrain gen in debug)
- `voxeldust-core`: opt-level=2
- `rapier3d`, `parry3d`, `nalgebra`, `simba`: opt-level=2

---

## 16. Future Work

### Near-Term
- **Atmosphere rendering**: Rayleigh + Mie scattering (shader exists, needs integration). Scale heights must be Earth-proportional (~1.6% of planet_radius).
- **Dynamic lighting**: Light propagation from emissive blocks
- **Ship construction**: Functional block systems (power grid, thrust vectoring)

### Medium-Term
- **Weather**: Deterministic simulation from seed + time (owned by planet shard)
- **Fauna/Flora AI**: Creatures derived from planet seed
- **Multiplayer ships**: Dual physics — exterior rigid body + interior walkable KCC
- **Auto-scaling**: Wire up orchestrator split/merge to actual shard provisioning

### Long-Term
- **Earth-scale planets**: 6.4M block radius. Requires sparse shell storage, aggressive LOD
- **Orchestrator HA**: Multi-node via Raft consensus (openraft)
- **Rolling updates**: Zero-downtime shard upgrades
- **Anti-cheat**: Server-side validation of all client claims
