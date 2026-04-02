use glam::{DVec3, DQuat};
use serde::{Deserialize, Serialize};

use crate::shard_types::{SessionToken, ShardId};

/// Complete player state transferred between shards during handoff.
///
/// This struct must contain every field needed to fully reconstruct
/// a player entity on the target shard. Adding fields here requires
/// updating: FlatBuffers schema → serialize/deserialize → spawn system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerHandoff {
    /// Session token for matching the reconnecting client.
    pub session_token: SessionToken,
    /// Player display name.
    pub player_name: String,

    // -- Transform --
    /// World position (f64 for planetary precision).
    pub position: DVec3,
    /// Linear velocity.
    pub velocity: DVec3,
    /// Orientation quaternion.
    pub rotation: DQuat,
    /// Look direction (client camera forward).
    pub forward: DVec3,

    // -- Movement state --
    /// Whether the player is in fly mode.
    pub fly_mode: bool,
    /// Current fly speed tier (0-3).
    pub speed_tier: u8,
    /// Whether the player is grounded.
    pub grounded: bool,

    // -- Combat state --
    /// Current health (0-100).
    pub health: f32,
    /// Current shield (0-100).
    pub shield: f32,

    // -- Context --
    /// Shard this player is coming from.
    pub source_shard: ShardId,
    /// Tick number on the source shard when handoff was initiated.
    pub source_tick: u64,

    // -- Galaxy handoff context --
    /// For galaxy→system handoffs: the star index the player is entering.
    pub target_star_index: Option<u32>,
    /// For system→galaxy handoffs: galaxy context for coordinate transform.
    pub galaxy_context: Option<GalaxyHandoffContext>,

    // -- Ship/Planet handoff context --
    /// Target planet seed (system shard uses this to route to the correct planet shard).
    pub target_planet_seed: Option<u64>,
    /// Target planet index within the system.
    pub target_planet_index: Option<u32>,
    /// Target ship entity ID (for planet→ship re-entry handoffs).
    pub target_ship_id: Option<u64>,
    /// Target ship shard ID (for direct routing).
    pub target_ship_shard_id: Option<ShardId>,
    /// Ship's system-space position at handoff time (for coordinate transform).
    pub ship_system_position: Option<DVec3>,
    /// Ship's rotation at handoff time (for coordinate transform).
    pub ship_rotation: Option<DQuat>,
    /// System shard's authoritative celestial time for time synchronization.
    pub game_time: f64,
}

/// Context for handoffs between galaxy and system shards.
/// Carries the information needed to correctly position the player
/// after the coordinate system transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalaxyHandoffContext {
    pub galaxy_seed: u64,
    pub star_index: u32,
    pub star_position: DVec3,
}

/// Confirmation sent back from target shard to source shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffAccepted {
    pub session_token: SessionToken,
    pub target_shard: ShardId,
}

/// Position update sent from source shard to target shard for 15 frames
/// after handoff, so the target can show the player's ghost until the
/// real client reconnects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GhostUpdate {
    pub session_token: SessionToken,
    pub position: DVec3,
    pub rotation: DQuat,
    pub velocity: DVec3,
    pub tick: u64,
}

/// Redirect message sent to client, telling it to reconnect to a new shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRedirect {
    pub session_token: SessionToken,
    pub target_tcp_addr: String,
    pub target_udp_addr: String,
    pub shard_id: ShardId,
}

/// Ship position/state sent from system shard to planet shards within SOI.
/// Planet shards use this to render ships on the surface and detect re-entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipNearbyInfo {
    pub ship_id: u64,
    pub ship_shard_id: ShardId,
    /// Ship position in system-space (meters).
    pub position: DVec3,
    /// Ship rotation.
    pub rotation: DQuat,
    /// Ship velocity in system-space.
    pub velocity: DVec3,
}

/// Pre-connect notification sent from a shard to the client via TCP.
/// Tells the client to open a secondary connection for pre-loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardPreConnect {
    /// Shard type to pre-connect to (0=Planet, 2=Ship).
    pub shard_type: u8,
    /// TCP address for the secondary shard.
    pub tcp_addr: String,
    /// UDP address for the secondary shard.
    pub udp_addr: String,
    /// Seed for terrain/content generation.
    pub seed: u64,
    /// Planet index within the system (for planet shards).
    pub planet_index: u32,
    /// System-space position of the shard's origin.
    pub reference_position: DVec3,
    /// Rotation of the shard's local frame.
    pub reference_rotation: DQuat,
}
