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

    // -- Warp handoff context --
    /// For system→galaxy warp: target star index in the galaxy.
    pub warp_target_star_index: Option<u32>,
    /// For system→galaxy warp: current warp velocity in GU/s.
    pub warp_velocity_gu: Option<DVec3>,

    // -- EVA handoff context --
    /// Ship→system EVA exit: player leaves ship hull in space.
    /// When true, the system shard spawns an EVA player entity
    /// with the inherited ship velocity.
    pub target_system_eva: bool,

    // -- Character-system evolution --
    /// Schema version of [`Self::character_state`].
    /// `0` = pre-KCC (legacy), `1` = KCC migration (Phase A — blob reserved).
    /// Always-`Default` so deserializers from older wire format see `0`
    /// and reconstruct character defaults.
    #[serde(default)]
    pub schema_version: u16,
    /// Reserved zstd-compressed, tag-keyed blob for future character
    /// components ([`crate::character::hooks::CharacterComponentTag`]).
    /// Empty in the current version — the hooks aren't wired yet.
    #[serde(default)]
    pub character_state: Vec<u8>,
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
///
/// `spawn_pose` carries the target shard's **actual** spawn pose. This
/// may differ from the source's `PlayerHandoff.position` because the
/// target uses its CURRENT ship pose (not the pose at handoff-creation
/// time, which is ~40-100 ms stale due to inter-shard network latency)
/// when computing `spawn = ship.pos_current + ship.rot_current * player_local`.
///
/// When the source shard forwards this to the client via `ShardRedirect`,
/// the client renders its first post-transition frame exactly where the
/// target broadcasts the player — zero drift, zero client prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffAccepted {
    pub session_token: SessionToken,
    pub target_shard: ShardId,
    pub spawn_pose: Option<SpawnPose>,
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
/// Used by the gateway for initial routing (pre-game join). Also used by
/// shards for in-game transitions (exit/board/launch/land) where the
/// `spawn_pose` fields carry the authoritative first-frame camera pose the
/// client should render at.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRedirect {
    pub session_token: SessionToken,
    pub target_tcp_addr: String,
    pub target_udp_addr: String,
    pub shard_id: ShardId,
    /// Shard type of the destination (0=Planet, 1=System, 2=Ship, 3=Galaxy).
    /// `255` = unknown/legacy (falls back to last-connected secondary's type).
    pub target_shard_type: u8,
    /// Authoritative post-transition pose, in destination-shard-local
    /// coordinates (system-space for SYSTEM/GALAXY, planet-local for
    /// PLANET, ship-local for SHIP). `None` for gateway routing where
    /// no natural spawn pose exists (client uses the JoinResponse
    /// position from the destination shard instead).
    pub spawn_pose: Option<SpawnPose>,
}

/// Authoritative spawn pose carried in a `ShardRedirect` for in-game
/// transitions. The source shard computes these using the exact same
/// formula the target will use to spawn the player, so the client's
/// first rendered frame post-transition matches the server's authoritative
/// spawn position with zero client-side prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnPose {
    /// System-space position on the destination shard. For destination
    /// types that render in system-space (SYSTEM, GALAXY) this is used
    /// directly as `smooth.render_position + destination.ws.origin`.
    /// For local-frame destinations (PLANET, SHIP) it is converted to
    /// local coordinates by subtracting the destination `ws.origin`.
    pub position: DVec3,
    /// Initial body/look rotation. For EVA exit this is the ship's
    /// rotation at exit (so body-frame yaw/pitch remain visually
    /// continuous). For boarding, identity (ship-local frame).
    pub rotation: DQuat,
    /// Inherited velocity (ship's velocity on exit; zero on board).
    /// Used to prime the client's smoothing so the handoff blend does
    /// not visibly decelerate the camera during the first ~150 ms.
    pub velocity: DVec3,
}

/// Seamless handoff: client promotes an existing secondary connection to primary.
/// Replaces `ShardRedirect` for in-game transitions (launch, land, board, warp).
///
/// Preconditions:
/// - The destination shard is already streaming to the client as a secondary
///   (via `ShardPreConnect` + `ObserverConnect`).
/// - The destination shard holds a `PendingOwnership` observer entity keyed
///   on `session_token` (created when the secondary opened).
///
/// On receipt the client:
/// 1. Stops sending `PlayerInput` to the source shard and starts sending to
///    `target_tcp_addr` / `target_udp_addr`.
/// 2. Keeps the existing UDP socket for the destination shard open.
/// 3. Blends the owned player's pose linearly from the last source snapshot
///    toward `handoff_position/velocity/rotation` over 150 ms, then follows
///    destination WorldState normally.
///
/// The source shard keeps the player entity alive as a ghost for
/// `source_demote_after_ticks` ticks and broadcasts `GhostUpdate` to the
/// destination so other observers never see a gap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardHandoff {
    pub session_token: SessionToken,
    /// Shard type of the secondary to promote (0=Planet, 1=System, 2=Ship, 3=Galaxy).
    pub promote_shard_type: u8,
    /// Shard id of the secondary to promote.
    pub promote_shard_id: ShardId,
    /// TCP endpoint for the new primary (may match an already-open secondary).
    pub target_tcp_addr: String,
    /// UDP endpoint for the new primary.
    pub target_udp_addr: String,
    /// Ticks the source shard will keep ghosting before despawning (default 15).
    pub source_demote_after_ticks: u8,
    /// Anchor at handoff tick in destination-frame coordinates.
    pub handoff_position: DVec3,
    pub handoff_velocity: DVec3,
    pub handoff_rotation: DQuat,
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
    /// Shard ID for multi-secondary disambiguation.
    /// When shard_type is Ship, multiple secondaries may coexist;
    /// this field distinguishes them (together with shard_type).
    pub shard_id: u64,
}

/// Server → client: tear down a specific secondary connection.
/// The client matches on (shard_type, seed) to find and close
/// the secondary connection and remove its chunk source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardDisconnectNotify {
    pub shard_type: u8,
    pub seed: u64,
}
