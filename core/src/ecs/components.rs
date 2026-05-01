//! Shared ECS component types used by both server shards and client.
//!
//! Components are split by access pattern: systems that touch disjoint subsets
//! of an entity's state can run with non-overlapping borrows.

use bevy_ecs::prelude::*;
use glam::{DQuat, DVec3};

use glam::IVec3;

use crate::autopilot::{AutopilotMode, FlightPhase, ShipPhysicalProperties};
use crate::block::{BlockId, FunctionalBlockKind};
use crate::shard_types::{SessionToken, ShardId};

// ---------------------------------------------------------------------------
// Transform (shared across all entity types)
// ---------------------------------------------------------------------------

/// World-space position (f64 for planetary-scale precision).
#[derive(Component)]
pub struct Position(pub DVec3);

/// Linear velocity in m/s.
#[derive(Component)]
pub struct Velocity(pub DVec3);

/// Orientation quaternion.
#[derive(Component)]
pub struct Rotation(pub DQuat);

/// Rotation rate in rad/s per axis.
#[derive(Component)]
pub struct AngularVelocity(pub DVec3);

/// Sub-grid identifier for mechanical systems (rotors, pistons).
/// SubGridId(0) is the root (parent ship body). Each mechanical mount
/// creates a new sub-grid with a unique ID.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct SubGridId(pub u32);

impl SubGridId {
    pub const ROOT: Self = Self(0);
}

/// Look/camera direction (players only).
#[derive(Component)]
pub struct Forward(pub DVec3);

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// Ship entity identifier (matches wire protocol ship_id).
#[derive(Component)]
pub struct ShipId(pub u64);

/// Player entity identifier (session token from gateway).
#[derive(Component)]
pub struct PlayerId(pub SessionToken);

/// Human-readable player name.
#[derive(Component)]
pub struct PlayerName(pub String);

// ---------------------------------------------------------------------------
// Ship control inputs (written by input/autopilot, read by physics)
// ---------------------------------------------------------------------------

/// Ship-local thrust vector (Newtons). Reset each tick after integration.
#[derive(Component)]
pub struct ThrustInput(pub DVec3);

/// Ship-local torque vector (N*m). Reset each tick after integration.
#[derive(Component)]
pub struct TorqueInput(pub DVec3);

// ---------------------------------------------------------------------------
// Ship physics configuration (read-only during simulation)
// ---------------------------------------------------------------------------

/// Ship physical properties — mass, cross-sections, drag, thrust limits.
/// Future: derived from block composition via aggregation system.
#[derive(Component)]
pub struct ShipPhysics(pub ShipPhysicalProperties);

// ---------------------------------------------------------------------------
// Autopilot (optional — presence on entity = autopilot engaged)
// ---------------------------------------------------------------------------

/// Autopilot state machine. Insert to engage, remove to disengage.
#[derive(Component)]
pub struct Autopilot {
    pub mode: AutopilotMode,
    pub phase: FlightPhase,
    pub target_body_id: u32,
    pub thrust_tier: u8,
    pub engage_time: f64,
    pub estimated_tof: f64,
    pub braking_committed: bool,
    pub target_orbit_altitude: f64,
}

/// Autopilot intercept trajectory data. Re-solved periodically.
#[derive(Component)]
pub struct AutopilotIntercept {
    pub intercept_pos: DVec3,
    pub target_arrival_vel: DVec3,
    pub last_solve_tick: u64,
}

/// Warp autopilot data (interstellar travel). Separate from planetary autopilot.
#[derive(Component)]
pub struct WarpAutopilot {
    pub target_star_index: u32,
    pub direction: DVec3,
}

// ---------------------------------------------------------------------------
// State markers (optional — presence = entity is in that state)
// ---------------------------------------------------------------------------

/// Ship is inside a planet's sphere of influence.
#[derive(Component)]
pub struct InSoi {
    pub planet_index: usize,
}

/// Ship is landed on a planet surface.
#[derive(Component)]
pub struct Landed {
    pub planet_index: usize,
    pub surface_radial: DVec3,
    pub celestial_time: f64,
}

/// Landing zone debounce counter.
#[derive(Component)]
pub struct LandingZoneDebounce {
    pub consecutive_ticks: u32,
}

/// Entity is mid-handoff to another shard. Suppresses input processing.
#[derive(Component)]
pub struct HandoffPending {
    pub target_shard: ShardId,
    pub initiated_tick: u64,
}

/// Player is on the ground (walking, not flying).
#[derive(Component)]
pub struct Grounded;

/// Player is in fly mode.
#[derive(Component)]
pub struct FlyMode;

/// Player is in pilot seat, controlling the ship.
#[derive(Component)]
pub struct Piloting;

// ---------------------------------------------------------------------------
// Player state
// ---------------------------------------------------------------------------

/// Player health (0-100).
#[derive(Component)]
pub struct Health(pub f32);

/// Player shield (0-100).
#[derive(Component)]
pub struct Shield(pub f32);

/// Movement speed tier (0-4, maps to engine tiers).
#[derive(Component)]
pub struct SpeedTier(pub u8);

// ---------------------------------------------------------------------------
// Rapier integration (per-entity handle into RapierContext resource)
// Gated behind "rapier" feature — only server shards need this.
// ---------------------------------------------------------------------------

/// Rapier rigid body handle. The actual RigidBody lives in the RapierContext resource.
/// Cleanup: watch `RemovedComponents<RapierBody>` to remove from RigidBodySet.
#[cfg(feature = "rapier")]
#[derive(Component)]
pub struct RapierBody(pub rapier3d::dynamics::RigidBodyHandle);

// ---------------------------------------------------------------------------
// Planet-specific
// ---------------------------------------------------------------------------

/// Orthonormal basis at a point on a sphere surface.
#[derive(Component)]
pub struct TangentFrame {
    pub up: DVec3,
    pub north: DVec3,
    pub east: DVec3,
}

/// Player yaw angle on planet surface or ship interior.
#[derive(Component)]
pub struct Yaw(pub f32);

/// Rapier origin offset for re-centering (planet shard).
#[cfg(feature = "rapier")]
#[derive(Component)]
pub struct RapierOrigin(pub DVec3);

/// Player input action state.
#[derive(Component)]
pub struct ActionInput {
    pub current: u8,
    pub previous: u8,
}

// ---------------------------------------------------------------------------
// Warp-specific (galaxy shard)
// ---------------------------------------------------------------------------

/// Warp travel state for a ship in interstellar space.
#[derive(Component)]
pub struct WarpState {
    pub origin_star_index: u32,
    pub target_star_index: u32,
    pub phase: FlightPhase,
    pub initial_distance_gu: f64,
    pub preconnect_sent: bool,
}

// ---------------------------------------------------------------------------
// Ship shard specifics
// ---------------------------------------------------------------------------

/// Marks an entity as the ship shard's reference to its host shard.
#[derive(Component)]
pub struct HostShardRef(pub Option<ShardId>);

/// Ship shard reference to the ship's shard identity.
#[derive(Component)]
pub struct ShipShardRef(pub Option<ShardId>);

/// Source system shard for warp ships.
#[derive(Component)]
pub struct SourceSystemShard(pub ShardId);

// ---------------------------------------------------------------------------
// Functional block entity
// ---------------------------------------------------------------------------

/// Marks an ECS entity as a functional block in the ship/planet grid.
///
/// This is the bidirectional link between the block grid and the ECS world:
/// - **Grid → Entity**: `BlockMeta::entity_index` stores the Entity index
/// - **Entity → Grid**: this component's `world_pos` stores the block position
///
/// Future phases add kind-specific components (ThrusterState, ReactorState, etc.)
/// on top of this via `Added<FunctionalBlockRef>` change detection.
#[derive(Component, Clone, Debug)]
pub struct FunctionalBlockRef {
    /// World-space block position in the ship grid.
    pub world_pos: IVec3,
    /// Block type ID.
    pub block_id: BlockId,
    /// Functional category — determines which subsystems interact with this block.
    pub kind: FunctionalBlockKind,
}

// ---------------------------------------------------------------------------
// Block identity (Phase 1 — signal pipeline capability foundation)
// ---------------------------------------------------------------------------

/// Per-functional-block identity stamped at placement time. Persisted to redb
/// alongside the block's voxel state.
///
/// `owner_id` is the player who placed (or was assigned ownership of) the
/// block. It anchors the channel namespace `<owner_id>.<path>` and is the
/// principal consulted by the channel's `publish_policy` / `subscribe_policy`
/// gates, plus by `RemoteAccessGrant` in Phase 3.
///
/// `block_uid` is a globally-unique 64-bit identifier minted by `OsRng` at
/// placement. It survives renames, ownership transfers, and shard migrations,
/// and is the stable handle used in default channel names like
/// `<owner_id>.seat-<block_uid>.thrust-forward`. Re-mining and replacing the
/// block at the same coordinates produces a fresh `block_uid` — that's
/// intentional: it's the block, not the position, that owns the identity.
///
/// `created_at_ms` is the wall-clock millisecond timestamp at placement.
/// Used by the Access Tokens panel to display "this grant created 2 days ago"
/// and to anchor the per-grant `created_at_ms` audit trail in Phase 3.
#[derive(Component, Clone, Copy, Debug)]
pub struct BlockOwnership {
    pub owner_id: u64,
    pub block_uid: u64,
    pub created_at_ms: u64,
}

impl BlockOwnership {
    /// Generate a fresh ownership stamp at placement time. Mints a
    /// cryptographically-strong `block_uid` from the OS RNG (not a
    /// predictable hash of position + tick — predictable uids would let a
    /// foreign shard pre-compute the channel names a fresh placement will
    /// use, defeating the namespacing collision-freedom property).
    pub fn new_at(owner_id: u64, now_ms: u64) -> Self {
        use rand::RngCore;
        let block_uid = rand::rngs::OsRng.next_u64();
        Self { owner_id, block_uid, created_at_ms: now_ms }
    }
}

// ---------------------------------------------------------------------------
// Player + seating (shard-agnostic — works on ships, planets, stations)
// ---------------------------------------------------------------------------

/// Marker component for a player entity. Distinguishes player-driven entities
/// from NPCs / drones / observers in queries that should only see real players
/// (e.g., the seat-input loop). Shard-agnostic: a player on a ship-shard,
/// planet-shard, or station-shard all carry this.
#[derive(Component)]
pub struct Player;

/// Per-player seated state. `seat_entity` points to the seat block entity the
/// player currently occupies. The seat itself owns the channel mapping
/// (`SeatChannelMapping`) — the seated player is just the trigger that
/// transports their input values into those channels each tick.
///
/// `seated_player_id` is captured on sit-down and consulted by
/// `ActivationAllowlist` (a future opt-in component on locked seats) to
/// gate publish; default seats with no allowlist don't read it.
#[derive(Component, Default)]
pub struct SeatedState {
    pub seated: bool,
    pub seat_entity: Option<Entity>,
}

/// Per-binding float values from the client's seat input evaluation.
/// Length matches the active seat's binding count. Written by the
/// shard's input drainer; read by the generic `signal_seat_publish`
/// system in `shard-common`.
///
/// Stored as a Component on the player entity (not a Resource) so the
/// generic publish system can iterate `(SeatedState, SeatInputValues)`
/// pairs without needing a per-shard player registry.
#[derive(Component, Default)]
pub struct SeatInputValues(pub Vec<f32>);

// ---------------------------------------------------------------------------
// Thermal state (ships in atmosphere)
// ---------------------------------------------------------------------------

/// Thermal energy state for atmospheric re-entry heating.
#[derive(Component)]
pub struct ThermalState {
    pub energy_j: f64,
}
