# Voxeldust-Mini

## Project Overview
Multiplayer voxel planet game — "Star Citizen meets Minecraft". Procedurally generated
universe with spherical voxel planets, player-built ships, and Newtonian space physics.

## Architecture

### Three-Crate Structure
- **Core** (`core/`): Rust library — terrain gen, meshing, coordinate math. Shared by server and client as a direct Rust dependency.
- **Server** (`server/`): Rust — Tokio async networking, Rapier3D physics, game loop at 20Hz.
- **Client** (`client/`): Rust + Bevy 0.18 — DefaultPlugins rendering pipeline (bevy_pbr + Bloom + AgX tonemap), bevy_egui UI. Shard-type plugins register via `ShardTypePlugin` trait so adding a new shard-type (DEBRIS, STATION, …) is one new file under `client/src/shard_types/`. `voxydust/` is the retired legacy client kept only as an implementation reference; `voxydust-next/` is the archived second attempt (removed from the workspace members).

### Key Principles
- **No magic numbers**: All world parameters derived deterministically from seeds (planet size, gravity, terrain, biomes).
- **Deterministic generation**: Same seed = same world on client and server. Only deltas (block edits) are networked.
- **Server-authoritative**: Server owns physics and game state. Client predicts and renders.
- **Seed hierarchy**: `universe_seed → system_seed → planet_seed` (each derived via hash).

### Coordinate System
- **Cubic sphere**: 6 cube faces projected to sphere via Nowell mapping.
- **Chunk**: 62³ blocks (binary-greedy-meshing crate requirement).
- **Address**: `(sector: u8, shell: u16, cx: u16, cy: u16, cz: u16)` + block `(bx: u8, by: u8, bz: u8)`.
- **Meshing**: Binary greedy meshing in flat chunk-local space [0,62]. Client sphere-projects vertices on CPU before upload. Server sphere-projects collision vertices for Rapier trimesh.

### Block System
- Block IDs are `u16` (supports up to 65535 block types).
- Current types: Air(0), Stone(1), Dirt(2), Grass(3), Sand(4), Water(5), plus 45+ more (ores, vegetation, ice, volcanic).
- Future: functional blocks (thrusters, cockpit, power generators, gravity generators, etc.).

### Physics (Rapier3D, server-only)
- `KinematicCharacterController` for player movement with per-player `up` vector.
- Terrain: static trimesh colliders per loaded chunk (sphere-projected vertices).
- Spherical gravity: computed per-player toward planet center, g = G*M/r².

### Networking
- TCP (port 7777): reliable — join response, block deltas.
- UDP (port 7778): fast — player input (client→server), world state (server→client).
- FlatBuffers serialization (`protocol/voxeldust.fbs`).
- No chunk streaming — both sides generate deterministically from seed.

### Shard Architecture (future)
```
System Shard (1 per star system)
├── Planet Shard (1-N per planet, by player density)
│   └── Rapier world, terrain chunks, surface players
├── Ship Shard (1 per active ship)
│   └── Ship's Rapier world, interior KCC, block systems simulation
└── Space players / ships in transit
```

### Future Architecture Notes
- **Ships**: Player-built from blocks, same 62³ chunk grid. Functional block systems (power grid, thrust vectoring). Dual physics: exterior rigid body (Newtonian thrust) + interior walkable space (KCC). Ship inertia configurable via gravity generator blocks.
- **Planets**: Type, biomes, weather, flora, fauna all derived from planet seed. Weather is deterministic simulation (seed + time).
- **Scale**: Earth-sized planets (~6.4M block radius). Requires floating origin (f32 precision), LOD, sparse shells.

## Build & Run
```bash
cargo test -p voxeldust-core          # Core tests (124 tests)
cargo run -p client                   # Run client (connects to localhost:7777 by default)
cargo run -p client -- --gateway 127.0.0.1:7777 --name Player  # Explicit options
./dev-cluster.sh up                   # Build images & deploy to k3d
./dev-cluster.sh rebuild              # Rebuild images & redeploy
./dev-cluster.sh down                 # Tear down cluster
./dev-cluster.sh status               # Check pod status
./dev-cluster.sh logs [component]     # Tail logs (default: orchestrator)
./build_protocol.sh                   # Regenerate FlatBuffers (Rust)
```

## Dependencies
- `binary-greedy-meshing`: 62³ binary greedy meshing (~65μs/chunk)
- `rapier3d`: Server-side physics (KCC, trimesh collision)
- `glam`: SIMD math (Vec3, Quat)
- `noise`: OpenSimplex procedural noise (pin exact version for determinism)
- `tokio`: Async TCP/UDP networking
- `flatbuffers`: Binary protocol serialization
- `wgpu`: GPU rendering (client)
- `winit`: Windowing and input (client)
- `egui`: Immediate-mode UI (client)

## Conventions
- Chunk size is always 62 (dictated by binary-greedy-meshing crate).
- All world parameters must be derivable from seeds — never hardcode planet-specific values.
- Pin noise crate to exact version (`noise = "=0.9.0"`) for cross-platform determinism.
- Block coordinates 0-61 stored as u8. Block type IDs as u16.
- When adding new player components or game state, always update the handoff system in lockstep:
  PlayerHandoff FlatBuffers table → Rust struct → serialize/deserialize → spawn system → boundary detection.
  Networking (WorldState broadcast, JoinResponse) must also be updated if the state is client-visible.

### ECS Patterns (bevy_ecs)
- **Events over manual queues**: Use `bevy_ecs::event::Events<T>` for producer-consumer data flow
  between systems. Never use `Vec<T>` drained with `mem::take`. For async→sync bridges (tokio → ECS),
  use a dedicated bridge system that drains mpsc into `EventWriter<T>` early in the schedule.
- **Entity-scoped state as components**: State belonging to a specific entity (handoff progress,
  cooldowns, pending actions) must be a Component on that entity, not an entry in a Vec/HashMap
  inside a Resource. This enables query-based lookup and automatic lifecycle management.
- **Split resources by access pattern**: If different systems touch disjoint subsets of a resource's
  fields, split into separate resources. This makes system signatures self-documenting and enables
  finer-grained borrow checking.
- **Automatic index sync**: For index resources (like ClientMap) that mirror component data, use a
  dedicated sync system with `Added<T>` / `RemovedComponents<T>` change detection instead of manual
  insert/remove scattered across systems.
- **Per-system schedules for profiling**: The one-schedule-per-system pattern in shard binaries is
  intentional for per-system timing at 20Hz. Do not consolidate without replacing the profiling.
- **apply_deferred placement**: Always place explicit `apply_deferred` between systems that spawn/despawn
  entities (Commands) and systems that query those entities.
