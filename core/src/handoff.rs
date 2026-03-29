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
