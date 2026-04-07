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
// Thermal state (ships in atmosphere)
// ---------------------------------------------------------------------------

/// Thermal energy state for atmospheric re-entry heating.
#[derive(Component)]
pub struct ThermalState {
    pub energy_j: f64,
}
