use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use clap::Parser;
use glam::{DQuat, DVec3, Vec3};
use rapier3d::prelude::*;
use tracing::{info, warn};

use voxeldust_core::autopilot::{self, ShipPhysicalProperties};
use voxeldust_core::block::{
    self, BlockId, BlockRegistry, DamageResult, FunctionalBlockKind, ShipGrid,
    StarterShipLayout, build_starter_ship,
    edit_pipeline::{self, BlockEditQueue, BlockEdit, EditSource, EntityOp},
    raycast,
};
use voxeldust_core::ecs::components::FunctionalBlockRef;
use voxeldust_core::signal::{
    self, SignalChannelTable, SignalPublisher, SignalSubscriber, SignalConverterConfig,
    SeatChannelMapping, SeatInputBinding, PublishBinding, SubscribeBinding, SignalProperty,
    FlightComputerState, HoverModuleState, AutopilotBlockState, WarpComputerState, EngineControllerState,
    SeatInputSource, KeyMode, AxisDirection,
    seat_presets,
};
use voxeldust_core::shard_message::AutopilotSnapshotData;
use voxeldust_core::client_message::{
    BlockEditData, CelestialBodyData, ChunkDeltaData, JoinResponseData, LightingData,
    PlayerSnapshotData, SeatBindingsNotifyData, ServerMsg, SubGridAssignmentData,
    SubGridTransformData, WorldStateData,
};
use voxeldust_core::ecs;
use voxeldust_core::handoff;
use voxeldust_core::shard_message::{
    AutopilotCommandData, CelestialBodySnapshotData, LightingInfoData, ShardMsg, ShipControlInput,
    WarpAutopilotCommandData,
};
use voxeldust_core::shard_types::{SessionToken, ShardId, ShardType};
use voxeldust_core::system::{self, SystemParams};
use voxeldust_shard_common::authorized_peers::AuthorizedPeers;
use voxeldust_shard_common::client_listener;
use voxeldust_shard_common::harness::{NetworkBridge, ShardHarness, ShardHarnessConfig};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "ship-shard", about = "Voxeldust ship shard — ship interior physics")]
struct Args {
    #[arg(long)]
    shard_id: u64,
    #[arg(long, default_value = "0")]
    ship_id: u64,
    #[arg(long)]
    host_shard: Option<u64>,
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    orchestrator: String,
    #[arg(long, default_value = "127.0.0.1:9090")]
    orchestrator_heartbeat: String,
    #[arg(long, default_value = "7777")]
    tcp_port: u16,
    #[arg(long, default_value = "7778")]
    udp_port: u16,
    #[arg(long, default_value = "7779")]
    quic_port: u16,
    #[arg(long, default_value = "8081")]
    healthz_port: u16,
    #[arg(long)]
    advertise_host: Option<String>,
    #[arg(long, default_value = "0")]
    seed: u64,
    #[arg(long, default_value = "0")]
    system_seed: u64,
    #[arg(long, default_value = "0")]
    galaxy_seed: u64,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// WALK_SPEED, JUMP_IMPULSE, INTERACT_DIST moved to PlayerPhysics component.

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Rapier 3D physics context — all physics primitives in one resource.
#[derive(Resource)]
struct RapierContext {
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    integration_params: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
}

/// Configurable gravity source — future: one per gravity block entity.
/// For now, ships have a single default source (floor plates).
#[derive(Clone)]
struct GravitySource {
    direction: Vec3,
    strength: f32,
    shape: GravityShape,
}

#[derive(Clone)]
enum GravityShape {
    Uniform,
}

impl GravitySource {
    fn default_floor_plates() -> Self {
        Self {
            direction: Vec3::new(0.0, -1.0, 0.0),
            strength: 9.81,
            shape: GravityShape::Uniform,
        }
    }

    fn gravity_at(&self, _position: Vec3) -> Vector<f32> {
        match self.shape {
            GravityShape::Uniform => {
                let g = self.direction * self.strength;
                vector![g.x, g.y, g.z]
            }
        }
    }
}

/// Gravity sources in the ship. Future: populated from gravity block entities.
#[derive(Resource)]
struct GravitySources(Vec<GravitySource>);

/// Ship shard identity and configuration.
#[derive(Resource)]
struct ShipConfig {
    shard_id: ShardId,
    ship_id: u64,
    host_shard_id: Option<ShardId>,
    system_seed: u64,
    galaxy_seed: u64,
}

/// Ship exterior state (received from system/galaxy shard via QUIC).
#[derive(Resource)]
struct ShipExterior {
    position: DVec3,
    velocity: DVec3,
    rotation: DQuat,
    angular_velocity: DVec3,
    /// Authoritative atmosphere flag from system-shard.
    in_atmosphere: bool,
    /// Planet index if in atmosphere, -1 otherwise.
    atmosphere_planet_index: i32,
    /// Authoritative gravitational acceleration (m/s², world frame).
    gravity_acceleration: DVec3,
    /// Atmospheric density at ship altitude (kg/m³).
    atmosphere_density: f64,
}

/// Cached celestial scene from the host shard.
#[derive(Resource)]
struct SceneCache {
    bodies: Vec<CelestialBodySnapshotData>,
    lighting: Option<LightingInfoData>,
    game_time: f64,
    last_update_tick: u64,
    host_switch_tick: u64,
    authorized_peers: AuthorizedPeers,
}

/// Cached system parameters for Keplerian extrapolation and atmosphere detection.
#[derive(Resource)]
struct CachedSystemParams(Option<SystemParams>);

/// Cached galaxy map — generated once from seed, reused for warp target cycling.
/// Avoids regenerating 50-100K stars on every G-key press.
#[derive(Resource)]
struct CachedGalaxyMap(Option<voxeldust_core::galaxy::GalaxyMap>);

/// Ship physical properties (mass, thrust, drag).
#[derive(Resource)]
struct ShipProps(ShipPhysicalProperties);

/// Pilot thrust/torque accumulated from input, sent each tick, then kept for next.
#[derive(Resource, Default)]
struct PilotAccumulator {
    thrust: DVec3,
    torque: DVec3,
    /// Rotation intent from pilot + flight computer, in [-1, 1] per axis.
    /// Sent to system shard as rate command (not the physical torque).
    /// X = pitch, Y = yaw, Z = roll.
    rotation_command: DVec3,
}

// SeatInputValues and ActiveSeatEntity moved to per-entity Components above.

/// Runtime throttle state for a thruster, driven by signal channel subscription.
#[derive(Component)]
struct ThrusterState {
    throttle: f32,
    /// Boost multiplier from cruise drive (1.0 = normal, >1.0 = boosted).
    /// Set by Boost signal property. Multiplies both thrust output and power consumption.
    boost: f32,
}

impl Default for ThrusterState {
    fn default() -> Self {
        Self { throttle: 0.0, boost: 1.0 }
    }
}

/// Static thruster properties, copied from registry on spawn.
#[derive(Component)]
struct ThrusterBlock {
    thrust_n: f64,
    block_pos: glam::IVec3,
    direction: DVec3,
}

// ---------------------------------------------------------------------------
// Power network components
// ---------------------------------------------------------------------------

/// Reactor runtime state. Toggled via E key interaction.
#[derive(Component)]
struct ReactorState {
    active: bool,
    throttle: f32,
    max_generation_w: f64,
    broadcast_range: f32,
    access: PowerAccess,
    placed_by: u64,
    circuits: Vec<PowerCircuit>,
}

impl Default for ReactorState {
    fn default() -> Self {
        Self {
            active: true,
            throttle: 1.0,
            max_generation_w: 500_000.0,
            broadcast_range: 50.0,
            access: PowerAccess::OwnerOnly,
            placed_by: 0,
            circuits: Vec::new(),
        }
    }
}

/// A named power circuit on a reactor. Each circuit gets a fraction of the
/// reactor's output and tracks its own supply/demand budget.
#[derive(Clone, Debug)]
struct PowerCircuit {
    name: String,
    fraction: f32,
    supply_w: f64,
    demand_w: f64,
    power_ratio: f32,
}

/// Access control for wireless power broadcast.
#[derive(Clone, Debug, Default)]
enum PowerAccess {
    /// Only blocks placed by the same player (default).
    #[default]
    OwnerOnly,
    /// Owner + listed player session IDs.
    AllowList(Vec<u64>),
    /// Anyone in range.
    Open,
}

/// Links a consumer block to a reactor for wireless power.
/// The reactor is identified by its stable block position (survives chunk reload).
#[derive(Component)]
struct PoweredBy {
    reactor_pos: glam::IVec3,
    circuit: String,
    consumption_w: f64,
    placed_by: u64,
}

/// Runtime state for a cruise drive block. Reads throttle from "cruise" signal,
/// publishes boost multiplier to configured boost channels when active.
#[derive(Component)]
struct CruiseDriveState {
    /// On/off throttle from the "cruise" signal channel (0.0 or 1.0).
    throttle: f32,
    /// Boost multiplier from registry (e.g., 500.0 for small drive).
    boost_multiplier: f64,
}

/// Cached consumer entry for a reactor's consumer list.
#[derive(Clone, Debug)]
struct CachedConsumer {
    entity: Entity,
    circuit_idx: usize,
    consumption_w: f64,
}

/// Pre-computed consumer list for a reactor. Rebuilt when `dirty` is set.
#[derive(Component)]
struct ReactorConsumerCache {
    entries: Vec<CachedConsumer>,
    dirty: bool,
}

impl Default for ReactorConsumerCache {
    fn default() -> Self {
        Self { entries: Vec::new(), dirty: true }
    }
}

/// Battery runtime state (dormant — reserved for future use).
#[derive(Component)]
struct BatteryState {
    energy_j: f64,
    capacity_j: f64,
    max_rate_w: f64,
}

// ---------------------------------------------------------------------------
// Mechanical systems (rotors, pistons, hinges, sliders)
// ---------------------------------------------------------------------------

/// Operational status of a mechanical mount.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
enum MechanicalStatus {
    #[default]
    Idle = 0,
    Moving = 1,
    Blocked = 2,
    Error = 3,
}

/// Generic mechanical mount state — ONE component for all joint types (rotor, piston, hinge, slider).
/// Joint type and limits come from `MechanicalProps` in the registry. The system treats all
/// mechanical mounts identically; behavior differs only via `joint_type` field.
#[derive(Component)]
struct MechanicalState {
    /// Current value: angle (deg) for revolute, extension (m) for prismatic.
    current: f32,
    /// Target value from signal subscription.
    target: f32,
    /// Max speed from MechanicalProps (overridable via Speed signal).
    max_speed: f32,
    /// Max force/torque from MechanicalProps.
    max_force: f64,
    /// Max range from MechanicalProps.
    max_range: f64,
    /// Joint type (Revolute or Prismatic).
    joint_type: block::JointType,
    /// Sub-grid ID of the child grid this mount controls.
    child_grid_id: voxeldust_core::ecs::components::SubGridId,
    /// Operational status (published via signal).
    status: MechanicalStatus,
    /// Host block position (where the sub-block mount is placed).
    host_pos: glam::IVec3,
    /// Face on which the mount is placed (determines joint axis).
    face: u8,
    /// Ticks since last error decrease — for stuck detection.
    stuck_ticks: u16,
    /// Previous error magnitude for stuck detection.
    prev_error: f32,
    /// Direct velocity override from Throttle signal (deg/s). 0 = use position control.
    velocity_override: f32,
    /// Locked state: Active=false signal locks mechanism at current position.
    locked: bool,
}

/// Data for a single sub-grid (kinematic child rigid body).
struct SubGridData {
    /// Rapier rigid body handle for this sub-grid (KinematicPositionBased).
    body_handle: rapier3d::dynamics::RigidBodyHandle,
    /// Mechanism arm collider (piston rod or rotor axle). Physical collision object.
    arm_collider: Option<rapier3d::geometry::ColliderHandle>,
    /// Block positions that belong to this sub-grid.
    members: std::collections::HashSet<glam::IVec3>,
    /// Mechanical mount position that created this sub-grid.
    mount_pos: glam::IVec3,
    /// Face on which the mount sits.
    mount_face: u8,
    /// Parent sub-grid (for nested mounts: rotor on piston).
    parent_grid: voxeldust_core::ecs::components::SubGridId,
    /// Per-chunk collider handles for this sub-grid.
    collider_handles: std::collections::HashMap<glam::IVec3, rapier3d::geometry::ColliderHandle>,
}

/// Registry of all sub-grids on this ship. One per ship shard.
#[derive(Resource)]
struct SubGridRegistry {
    next_id: u32,
    grids: std::collections::HashMap<voxeldust_core::ecs::components::SubGridId, SubGridData>,
}

impl Default for SubGridRegistry {
    fn default() -> Self {
        Self { next_id: 1, grids: std::collections::HashMap::new() }
    }
}

/// World-space isometries for all mechanical sub-grids, computed by
/// `apply_mechanical_transforms`. Used by other systems (raycast, broadcast)
/// to know where sub-grid blocks are in world space.
#[derive(Resource, Default)]
struct MechanicalWorldIsometries(
    std::collections::HashMap<voxeldust_core::ecs::components::SubGridId, rapier3d::math::Isometry<f32>>,
);

// PlayerPlatform moved to per-entity PlayerPlatformState component above.

/// Handle for the root (parent ship body) Fixed rigid body.
/// All non-sub-grid chunk colliders are parented to this body.
#[derive(Resource)]
struct RootBodyHandle(rapier3d::dynamics::RigidBodyHandle);

/// Tracks which sub-grids need membership re-computation after block edits.
/// Set by `apply_block_edits` when a block is placed/removed adjacent to a
/// mechanical mount. Read and cleared by `process_mechanical_edits`.
#[derive(Resource, Default)]
struct MechanicalDirtyGrids {
    /// Sub-grid IDs that need re-BFS.
    dirty: std::collections::HashSet<voxeldust_core::ecs::components::SubGridId>,
}

/// Ship center of mass (block coordinates). Updated by aggregate_ship_properties.
#[derive(Resource, Default)]
struct ShipCenterOfMass(DVec3);

/// Cached autopilot snapshot from system-shard (for WorldState broadcast).
#[derive(Resource, Default)]
struct AutopilotSnapshotCache {
    snapshot: Option<AutopilotSnapshotData>,
}

// ---------------------------------------------------------------------------
// Player components (per-entity, multi-player)
// ---------------------------------------------------------------------------

/// Marker component for a player entity on this ship.
#[derive(Component)]
struct Player;

/// Player session identity — ties this entity to a network session.
#[derive(Component)]
struct SessionId(SessionToken);

/// Player display name.
#[derive(Component)]
struct PlayerName(String);

/// Per-player Rapier rigid body handle.
#[derive(Component)]
struct PlayerBody(RigidBodyHandle);

/// Per-player position in ship-local coordinates.
#[derive(Component)]
struct PlayerPosition(Vec3);

/// Per-player yaw angle (radians).
#[derive(Component)]
struct PlayerYaw(f32);

/// Per-player seated state. `seat_entity` points to the seat block entity.
/// Seat role (pilot/gunner/passenger) is determined by the seat's SeatChannelMapping.
#[derive(Component, Default)]
struct SeatedState {
    seated: bool,
    seat_entity: Option<Entity>,
}

/// Per-player input action state for edge detection.
#[derive(Component, Default)]
struct InputActions {
    current: u8,
    previous: u8,
}

/// Per-binding float values from the client's seat input evaluation.
/// Length matches the active seat's binding count. Written by process_input.
#[derive(Component, Default)]
struct SeatInputValues(Vec<f32>);

/// Per-player physics parameters — no magic numbers.
/// Default values are the standard humanoid parameters.
/// Future: can vary by character type, equipment, gravity, etc.
#[derive(Component, Clone)]
struct PlayerPhysics {
    walk_speed: f32,
    jump_impulse: f32,
    capsule_height: f32,
    capsule_radius: f32,
    interact_distance: f32,
}

impl Default for PlayerPhysics {
    fn default() -> Self {
        Self {
            walk_speed: 4.0,
            jump_impulse: 5.0,
            capsule_height: 0.6,
            capsule_radius: 0.3,
            interact_distance: 1.5,
        }
    }
}

/// Marker for a player with a pending shard handoff.
/// Prevents duplicate handoff attempts in hull_exit_check.
#[derive(Component)]
struct HandoffPending;

/// Tracks which sub-grid the player is currently standing on.
/// Used to apply platform motion delta to the player each tick.
#[derive(Component, Default)]
struct PlayerPlatformState {
    /// Sub-grid the player is currently riding (standing on).
    riding: Option<voxeldust_core::ecs::components::SubGridId>,
    /// World isometry of the sub-grid at the END of last tick.
    prev_iso: Option<rapier3d::math::Isometry<f32>>,
}

/// Component on seat entities — tracks who occupies this seat.
/// None = vacant, Some(entity) = occupied by that player.
#[derive(Component, Default)]
struct SeatOccupant(Option<Entity>);

/// Entity index: session_token → Entity for O(1) lookup.
/// Synced by spawn_player/despawn_player.
#[derive(Resource, Default)]
struct PlayerEntityIndex(std::collections::HashMap<SessionToken, Entity>);

/// Stores handoff data for players that are about to connect via TCP.
/// When a PlayerHandoff arrives via QUIC (re-entry), we store the spawn info here.
/// When the player's TCP Connect arrives, process_connects consumes it.
#[derive(Resource, Default)]
struct PendingPlayerHandoffs(std::collections::HashMap<String, handoff::PlayerHandoff>);

/// Ship voxel grid — the block-based ship structure.
#[derive(Resource)]
struct ShipGridResource(ShipGrid);

/// Block registry — static block property table.
#[derive(Resource)]
struct BlockRegistryResource(BlockRegistry);

/// Index mapping world position → Entity for O(1) lookup of functional blocks.
/// Synced automatically by the entity lifecycle system (process_entity_ops).
#[derive(Resource, Default)]
struct FunctionalBlockIndex(std::collections::HashMap<glam::IVec3, Entity>);

/// Pending entity lifecycle operations from the edit pipeline.
/// Set by apply_block_edits, consumed by process_entity_ops.
#[derive(Resource, Default)]
struct PendingEntityOps {
    ops: Vec<EntityOp>,
}

/// Whether ship physical properties need recomputation from block composition.
/// Set to `true` when blocks change. The aggregation system clears it after recomputing.
#[derive(Resource)]
struct AggregationDirty(bool);

/// Buffer for incoming cross-shard signals received via QUIC.
/// Drained by signal_publish at the start of the signal pipeline.
#[derive(Resource, Default)]
struct IncomingSignalBuffer {
    signals: Vec<(String, signal::SignalValue)>,
}

/// Last-known world positions of peer shards, learned from incoming signal
/// broadcasts.  Used for spatial filtering in `signal_broadcast_remote` so
/// ShortRange signals are only sent to peers within range.
///
/// Populated opportunistically: every incoming `SignalBroadcastBatch` carries
/// `source_shard_id` + `source_position`, so peers' positions are learned from
/// normal signal traffic without any additional messages.
///
/// Peers without a known position are included conservatively in broadcasts
/// (same as current behavior) — the receiver still does its own distance check.
#[derive(Resource, Default)]
struct PeerPositionCache {
    positions: std::collections::HashMap<voxeldust_core::shard_types::ShardId, glam::DVec3>,
}

/// Current inter-ship visibility set received from the system shard.
/// Tracks which other ships are visible (ship_id → connection info).
/// Used to detect enter/leave events and manage secondary connections.
#[derive(Resource, Default)]
struct VisibleShipRegistry {
    ships: std::collections::HashMap<u64, VisibleShipInfo>,
}

/// Connection info for a visible ship (from VisibilityDirective).
#[derive(Debug, Clone)]
struct VisibleShipInfo {
    shard_id: ShardId,
    tcp_addr: String,
    udp_addr: String,
}

/// Latest unified AOI snapshot received from this ship's host system shard.
/// The ship merges these entries into `WorldState.entities` (transformed from
/// system-space to ship-local origin) so all connected players see every
/// ship / EVA / surface player within the configured AOI range.
#[derive(Resource, Default)]
struct ExternalEntities {
    entities: Vec<voxeldust_core::client_message::ObservableEntityData>,
    observer_position: DVec3,
    tick: u64,
}

/// Handles of per-chunk compound colliders. Keyed by chunk key.
/// When blocks change in a chunk, we remove the old collider and insert a new one.
#[derive(Resource, Default)]
struct ChunkColliderHandles(std::collections::HashMap<glam::IVec3, ColliderHandle>);


/// Tight bounding box of all solid blocks in the ship.
/// Used for hull boundary exit detection — if the player leaves this volume
/// (with a small margin), they've exited the ship through a gap/hatch.
/// Recomputed when blocks change.
#[derive(Resource)]
struct ShipHullBounds {
    /// Min corner of the solid block AABB (world-space block coords).
    min: glam::IVec3,
    /// Max corner of the solid block AABB (world-space block coords).
    max: glam::IVec3,
    /// Margin in blocks beyond the hull before triggering exit (accounts for
    /// the player standing just outside a hatch without triggering handoff).
    margin: f32,
}

/// Default spawn position for new players connecting to this ship.
/// Computed at startup by scanning for the first COCKPIT block.
/// Players spawn 1 block above the cockpit to avoid clipping into the seat.
#[derive(Resource)]
struct DefaultSpawnPosition(Vec3);

/// Atmosphere tracking for ShardPreConnect.
#[derive(Resource, Default)]
struct AtmosphereState {
    in_atmosphere: bool,
    preconnect_sent: bool,
}

/// Landing state derived from ship velocity.
#[derive(Resource, Default)]
struct LandingState {
    landed: bool,
}

/// Pending QUIC messages to send to host shard (set by interaction/input systems).
#[derive(Resource, Default)]
struct PendingMessages {
    /// Multiple concurrent handoffs (one per player exiting).
    handoffs: Vec<ShardMsg>,
}

// PlayerBodyHandle, PlayerPosition, PlayerYaw, SeatedState, InputActions
// moved to per-entity Components above.

/// Whether gravity is enabled in the ship.
#[derive(Resource)]
struct GravityEnabled(bool);

// ---------------------------------------------------------------------------
// Messages (ship-shard-specific)
// ---------------------------------------------------------------------------

#[derive(Message)]
struct ClientConnectedMsg {
    session_token: SessionToken,
    player_name: String,
    tcp_write: Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>,
}

#[derive(Message)]
struct PlayerInputMsg {
    session: SessionToken,
    input: voxeldust_core::client_message::PlayerInputData,
}

#[derive(Message)]
struct BlockEditMsg {
    session: SessionToken,
    edit: BlockEditData,
}

#[derive(Message)]
struct SubBlockEditMsg {
    session: SessionToken,
    edit: voxeldust_core::client_message::SubBlockEditData,
}

#[derive(Message)]
struct ConfigUpdateMsg {
    session: SessionToken,
    update: voxeldust_core::signal::config::BlockConfigUpdateData,
}

/// Inbound handoff from another shard (player re-entering this ship).
#[derive(Message)]
struct InboundHandoffMsg {
    handoff: handoff::PlayerHandoff,
}

/// HandoffAccepted confirmation from target shard.
#[derive(Message)]
struct HandoffAcceptedMsg {
    accepted: handoff::HandoffAccepted,
}

/// HostSwitch directive from system/galaxy shard.
#[derive(Message)]
struct HostSwitchMsg {
    data: voxeldust_core::shard_message::HostSwitchData,
}

// ---------------------------------------------------------------------------
// SystemParam bundles
// ---------------------------------------------------------------------------

/// Signal + power queries bundled to stay within the 16-param system limit.
#[derive(SystemParam)]
struct SignalQueryCtx<'w, 's> {
    channels: Res<'w, SignalChannelTable>,
    pub_query: Query<'w, 's, &'static SignalPublisher>,
    sub_query: Query<'w, 's, &'static SignalSubscriber>,
    converter_query: Query<'w, 's, &'static SignalConverterConfig>,
    seat_query: Query<'w, 's, &'static SeatChannelMapping>,
    reactor_query: Query<'w, 's, (&'static mut ReactorState, Option<&'static FunctionalBlockRef>)>,
    powered_by_query: Query<'w, 's, &'static PoweredBy>,
    cache_query: Query<'w, 's, &'static mut ReactorConsumerCache>,
    block_ref_query: Query<'w, 's, &'static FunctionalBlockRef>,
    // System block queries for config snapshot.
    fc_query: Query<'w, 's, &'static FlightComputerState>,
    hm_query: Query<'w, 's, &'static HoverModuleState>,
    ap_query: Query<'w, 's, &'static AutopilotBlockState>,
    wc_query: Query<'w, 's, &'static WarpComputerState>,
    ec_query: Query<'w, 's, &'static EngineControllerState>,
    mech_query: Query<'w, 's, &'static MechanicalState>,
}

// ---------------------------------------------------------------------------
// System Sets
// ---------------------------------------------------------------------------

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum ShipSet {
    Bridge,
    Input,
    BlockEdit,
    Signal,
    Physics,
    Interaction,
    Send,
    Broadcast,
    Diagnostics,
}

// ---------------------------------------------------------------------------
// Bridge systems
// ---------------------------------------------------------------------------

fn drain_connects(
    mut bridge: ResMut<NetworkBridge>,
    mut events: MessageWriter<ClientConnectedMsg>,
) {
    for _ in 0..16 {
        let event = match bridge.connect_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        let conn = event.connection;
        if let Ok(mut reg) = bridge.client_registry.try_write() {
            reg.register(&conn);
        }
        info!(player = %conn.player_name, session = conn.session_token.0, "player entered ship");
        events.write(ClientConnectedMsg {
            session_token: conn.session_token,
            player_name: conn.player_name.clone(),
            tcp_write: conn.tcp_write.clone(),
        });
    }
}

fn drain_input(
    mut bridge: ResMut<NetworkBridge>,
    mut events: MessageWriter<PlayerInputMsg>,
) {
    // Snapshot current UDP→session mapping from ClientRegistry.
    let session_map: Vec<(std::net::SocketAddr, SessionToken)> =
        if let Ok(reg) = bridge.client_registry.try_read() {
            // Build a quick lookup from the registry.
            reg.udp_addrs()
                .iter()
                .filter_map(|addr| reg.session_for_udp(*addr).map(|s| (*addr, s)))
                .collect()
        } else {
            Vec::new()
        };

    for _ in 0..64 {
        let (src, input) = match bridge.input_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        // Resolve the UDP source address to a session token.
        let session = session_map
            .iter()
            .find(|(addr, _)| *addr == src)
            .map(|(_, token)| *token);
        if let Some(session) = session {
            events.write(PlayerInputMsg { session, input });
        }
    }
}

fn drain_quic(
    mut bridge: ResMut<NetworkBridge>,
    mut scene: ResMut<SceneCache>,
    mut exterior: ResMut<ShipExterior>,
    mut config: ResMut<ShipConfig>,
    mut autopilot_cache: ResMut<AutopilotSnapshotCache>,
    mut landing: ResMut<LandingState>,
    mut rapier: ResMut<RapierContext>,
    mut player_index: ResMut<PlayerEntityIndex>,
    players: Query<(&SessionId, &PlayerBody), With<Player>>,
    tick: Res<ecs::TickCounter>,
    mut incoming_signals: ResMut<IncomingSignalBuffer>,
    mut peer_positions: ResMut<PeerPositionCache>,
    mut pending_handoffs: ResMut<PendingPlayerHandoffs>,
    mut visible_ships: ResMut<VisibleShipRegistry>,
    mut external_entities: ResMut<ExternalEntities>,
    mut commands: Commands,
) {
    for _ in 0..32 {
        let queued = match bridge.quic_msg_rx.try_recv() {
            Ok(q) => q,
            Err(_) => break,
        };
        match queued.msg {
            ShardMsg::SystemSceneUpdate(data) => {
                if !scene.authorized_peers.is_authorized(queued.source_shard_id) {
                    continue;
                }
                scene.bodies = data.bodies;
                scene.lighting = Some(data.lighting);
                scene.game_time = data.game_time;
                scene.last_update_tick = tick.0;
            }
            ShardMsg::ShipPositionUpdate(data) => {
                if !scene.authorized_peers.is_authorized(queued.source_shard_id) {
                    continue;
                }
                if data.ship_id != config.ship_id {
                    continue;
                }
                exterior.position = data.position;
                exterior.velocity = data.velocity;
                exterior.rotation = data.rotation;
                exterior.angular_velocity = data.angular_velocity;
                exterior.in_atmosphere = data.in_atmosphere;
                exterior.atmosphere_planet_index = data.atmosphere_planet_index;
                exterior.gravity_acceleration = data.gravity_acceleration;
                exterior.atmosphere_density = data.atmosphere_density;
                autopilot_cache.snapshot = data.autopilot;

                // Detect landed state: system shard zeros velocity on landing.
                let was_landed = landing.landed;
                landing.landed = data.velocity.length() < 0.5;

            }
            ShardMsg::HandoffAccepted(accepted) => {
                // Target shard accepted the player. Send ShardRedirect, then despawn.
                let session = accepted.session_token;
                let target_shard = accepted.target_shard;
                info!(
                    session = session.0,
                    target = target_shard.0,
                    "received HandoffAccepted, sending ShardRedirect to client"
                );

                if let Ok(reg) = bridge.peer_registry.try_read() {
                    if let Some(peer_info) = reg.get(target_shard) {
                        let redirect = ServerMsg::ShardRedirect(handoff::ShardRedirect {
                            session_token: session,
                            target_tcp_addr: peer_info.endpoint.tcp_addr.to_string(),
                            target_udp_addr: peer_info.endpoint.udp_addr.to_string(),
                            shard_id: target_shard,
                        });
                        let cr = bridge.client_registry.clone();
                        tokio::spawn(async move {
                            if let Ok(reg) = cr.try_read() {
                                if let Err(e) = reg.send_tcp(session, &redirect).await {
                                    tracing::warn!(%e, "failed to send ShardRedirect");
                                }
                            }
                            if let Ok(mut reg) = cr.try_write() {
                                reg.unregister(&session);
                            }
                        });
                    }
                }

                // Despawn the player entity and remove Rapier body.
                if let Some(&entity) = player_index.0.get(&session) {
                    if let Ok((_, body_comp)) = players.get(entity) {
                        remove_player_body(&mut rapier, body_comp.0);
                    }
                    despawn_player(&mut commands, &mut rapier, &mut player_index, entity, session);
                }
            }
            ShardMsg::PlayerHandoff(h) => {
                // Player re-entering ship from planet. Store for process_connects.
                info!(
                    session = h.session_token.0,
                    player = %h.player_name,
                    "player re-entering ship via handoff — stored in pending"
                );

                // Store handoff for when the player's TCP connection arrives.
                pending_handoffs.0.insert(h.player_name.clone(), h.clone());

                // Send HandoffAccepted back to host shard.
                let accepted_msg = ShardMsg::HandoffAccepted(handoff::HandoffAccepted {
                    session_token: h.session_token,
                    target_shard: config.shard_id,
                });
                if let Some(host_id) = config.host_shard_id {
                    if let Ok(reg) = bridge.peer_registry.try_read() {
                        if let Some(addr) = reg.quic_addr(host_id) {
                            let _ = bridge.quic_send_tx.try_send((host_id, addr, accepted_msg));
                        }
                    }
                }
            }
            ShardMsg::HostSwitch(data) => {
                if data.ship_id != config.ship_id {
                    continue;
                }
                let new_host = data.new_host_shard_id;
                config.host_shard_id = Some(new_host);
                scene.authorized_peers.set_host(new_host);
                scene.bodies.clear();
                scene.lighting = Some(LightingInfoData {
                    sun_direction: DVec3::new(0.0, -1.0, 0.0),
                    sun_color: [0.5, 0.5, 0.6],
                    sun_intensity: 0.2,
                    ambient: 0.08,
                });
                scene.last_update_tick = tick.0;
                scene.host_switch_tick = tick.0;

                // Warp target clearing on host switch is now handled by
                // warp_computer_system via channel state.

                info!(
                    ship_id = config.ship_id,
                    new_host = new_host.0,
                    shard_type = data.new_host_shard_type,
                    quic_addr = %data.new_host_quic_addr,
                    "host switched — ship shard now reports to new host"
                );

                // Send ShardPreConnect to ALL connected players for secondary UDP.
                if !data.new_host_udp_addr.is_empty() {
                    let pc = ServerMsg::ShardPreConnect(handoff::ShardPreConnect {
                        shard_type: data.new_host_shard_type,
                        tcp_addr: data.new_host_tcp_addr.clone(),
                        udp_addr: data.new_host_udp_addr.clone(),
                        seed: data.seed,
                        planet_index: 0,
                        reference_position: DVec3::ZERO,
                        reference_rotation: DQuat::IDENTITY,
                        shard_id: data.new_host_shard_id.0,
                    });
                    let sessions: Vec<SessionToken> = players.iter().map(|(sid, _)| sid.0).collect();
                    let player_count = sessions.len();
                    if player_count > 0 {
                        let cr = bridge.client_registry.clone();
                        tokio::spawn(async move {
                            let reg = cr.read().await;
                            for session in &sessions {
                                let _ = reg.send_tcp(*session, &pc).await;
                            }
                        });
                        info!(players = player_count, "sent ShardPreConnect to all players for secondary UDP");
                    }
                } else {
                    warn!(
                        ship_id = config.ship_id,
                        shard_type = data.new_host_shard_type,
                        "HostSwitch has empty UDP address — ShardPreConnect NOT sent"
                    );
                }
            }
            ShardMsg::SignalBroadcast(data) => {
                // Legacy single-signal path (backward compat).
                if data.scope == 1 {
                    let dist = (exterior.position - data.source_position).length();
                    if dist > data.range_m {
                        continue;
                    }
                }
                let value = match data.value_type {
                    0 => signal::SignalValue::Bool(data.value_data > 0.5),
                    1 => signal::SignalValue::Float(data.value_data),
                    2 => signal::SignalValue::State(data.value_data as u8),
                    _ => signal::SignalValue::Float(data.value_data),
                };
                incoming_signals.signals.push((data.channel_name.clone(), value));
            }
            ShardMsg::SignalBroadcastBatch(batch) => {
                // Learn peer position from the batch header (zero-cost spatial data).
                let source_id = voxeldust_core::shard_types::ShardId(batch.source_shard_id);
                peer_positions.positions.insert(source_id, batch.source_position);

                for entry in &batch.entries {
                    // Distance check for ShortRange signals.
                    if entry.scope == 1 {
                        let dist = (exterior.position - batch.source_position).length();
                        if dist > entry.range_m {
                            continue;
                        }
                    }
                    let value = match entry.value_type {
                        0 => signal::SignalValue::Bool(entry.value_data > 0.5),
                        1 => signal::SignalValue::Float(entry.value_data),
                        2 => signal::SignalValue::State(entry.value_data as u8),
                        _ => signal::SignalValue::Float(entry.value_data),
                    };
                    incoming_signals.signals.push((entry.channel_name.clone(), value));
                }
            }
            ShardMsg::SystemEntitiesUpdate(data) => {
                // Only accept updates addressed to our ship.
                if let voxeldust_core::shard_message::AoiTarget::Ship(ship_id) = data.target {
                    if ship_id == config.ship_id && data.tick >= external_entities.tick {
                        external_entities.entities = data.entities;
                        external_entities.observer_position = data.observer_position;
                        external_entities.tick = data.tick;
                    }
                }
            }
            ShardMsg::VisibilityDirective(data) => {
                if data.ship_id != config.ship_id {
                    continue;
                }

                // Build new visibility map.
                let new_ships: std::collections::HashMap<u64, VisibleShipInfo> = data
                    .visible_ships
                    .iter()
                    .map(|e| (e.ship_id, VisibleShipInfo {
                        shard_id: e.shard_id,
                        tcp_addr: e.tcp_addr.clone(),
                        udp_addr: e.udp_addr.clone(),
                    }))
                    .collect();

                // Detect newly visible ships (entered).
                let sessions: Vec<SessionToken> = players.iter().map(|(sid, _)| sid.0).collect();
                for (ship_id, info) in &new_ships {
                    if !visible_ships.ships.contains_key(ship_id) {
                        // New ship entered visibility — send ShardPreConnect to all players.
                        let pc = ServerMsg::ShardPreConnect(handoff::ShardPreConnect {
                            shard_type: 2, // Ship
                            tcp_addr: info.tcp_addr.clone(),
                            udp_addr: info.udp_addr.clone(),
                            seed: *ship_id,
                            planet_index: 0,
                            reference_position: DVec3::ZERO,
                            reference_rotation: DQuat::IDENTITY,
                            shard_id: info.shard_id.0,
                        });
                        let cr = bridge.client_registry.clone();
                        let sess = sessions.clone();
                        tokio::spawn(async move {
                            if let Ok(reg) = cr.try_read() {
                                for session in &sess {
                                    let _ = reg.send_tcp(*session, &pc).await;
                                }
                            }
                        });
                        info!(ship_id, shard = info.shard_id.0, "ship entered visibility — sent ShardPreConnect");
                    }
                }

                // Detect ships that left visibility.
                for (ship_id, info) in &visible_ships.ships {
                    if !new_ships.contains_key(ship_id) {
                        // Ship left visibility — send ShardDisconnectNotify to all players.
                        let dn = ServerMsg::ShardDisconnectNotify(handoff::ShardDisconnectNotify {
                            shard_type: 2,
                            seed: *ship_id,
                        });
                        let cr = bridge.client_registry.clone();
                        let sess = sessions.clone();
                        tokio::spawn(async move {
                            if let Ok(reg) = cr.try_read() {
                                for session in &sess {
                                    let _ = reg.send_tcp(*session, &dn).await;
                                }
                            }
                        });
                        info!(ship_id, shard = info.shard_id.0, "ship left visibility — sent ShardDisconnectNotify");
                    }
                }

                visible_ships.ships = new_ships;
            }
            _ => {}
        }
    }
}


// ---------------------------------------------------------------------------
// Player spawn / despawn
// ---------------------------------------------------------------------------

/// Spawn a new player entity with Rapier body, components, and index entry.
fn spawn_player(
    commands: &mut Commands,
    rapier: &mut RapierContext,
    player_index: &mut PlayerEntityIndex,
    token: SessionToken,
    name: String,
    spawn_pos: Vec3,
    physics: PlayerPhysics,
) -> Entity {
    let player_rb = RigidBodyBuilder::dynamic()
        .translation(vector![spawn_pos.x, spawn_pos.y, spawn_pos.z])
        .lock_rotations()
        .build();
    let handle = rapier.rigid_body_set.insert(player_rb);
    let player_collider = ColliderBuilder::capsule_y(physics.capsule_height, physics.capsule_radius).build();
    rapier.collider_set.insert_with_parent(
        player_collider,
        handle,
        &mut rapier.rigid_body_set,
    );

    let entity = commands
        .spawn((
            Player,
            SessionId(token),
            PlayerName(name),
            PlayerBody(handle),
            PlayerPosition(spawn_pos),
            PlayerYaw(0.0),
            SeatedState::default(),
            InputActions::default(),
            SeatInputValues::default(),
            PlayerPlatformState::default(),
            physics,
        ))
        .id();

    player_index.0.insert(token, entity);
    entity
}

/// Despawn a player entity: remove Rapier body, deregister from index, despawn entity.
fn despawn_player(
    commands: &mut Commands,
    rapier: &mut RapierContext,
    player_index: &mut PlayerEntityIndex,
    entity: Entity,
    token: SessionToken,
) {
    // Look up the Rapier body handle from the entity before despawning.
    // (Caller must ensure the entity has PlayerBody component.)
    player_index.0.remove(&token);
    commands.entity(entity).despawn();
}

/// Remove a player's Rapier rigid body and all its attached colliders.
fn remove_player_body(rapier: &mut RapierContext, body_handle: RigidBodyHandle) {
    rapier.rigid_body_set.remove(
        body_handle,
        &mut rapier.island_manager,
        &mut rapier.collider_set,
        &mut rapier.impulse_joint_set,
        &mut rapier.multibody_joint_set,
        true,
    );
}

// ---------------------------------------------------------------------------
// Connect processing
// ---------------------------------------------------------------------------

fn process_connects(
    mut events: MessageReader<ClientConnectedMsg>,
    mut commands: Commands,
    mut rapier: ResMut<RapierContext>,
    mut player_index: ResMut<PlayerEntityIndex>,
    mut pending_handoffs: ResMut<PendingPlayerHandoffs>,
    default_spawn: Res<DefaultSpawnPosition>,
    exterior: Res<ShipExterior>,
    config: Res<ShipConfig>,
    scene: Res<SceneCache>,
    ship_grid: Res<ShipGridResource>,
    bridge: Res<NetworkBridge>,
) {
    for event in events.read() {
        let token = event.session_token;
        let player_name = event.player_name.clone();

        // Observer connections: send chunk snapshots but no player entity.
        if player_name.starts_with("__observer__") {
            let observer_name = player_name.strip_prefix("__observer__").unwrap_or(&player_name);
            info!(%observer_name, session = token.0, "observer connected — sending chunk snapshots");

            let tcp_write = event.tcp_write.clone();

            // Serialize all chunk snapshots for the observer.
            let mut chunk_snapshots: Vec<ServerMsg> = Vec::new();
            for (chunk_key, chunk) in ship_grid.0.iter_chunks() {
                if chunk.is_empty() {
                    continue;
                }
                let compressed = block::serialize_chunk(chunk);
                chunk_snapshots.push(ServerMsg::ChunkSnapshot(
                    voxeldust_core::client_message::ChunkSnapshotData {
                        chunk_x: chunk_key.x,
                        chunk_y: chunk_key.y,
                        chunk_z: chunk_key.z,
                        seq: chunk.edit_seq(),
                        data: compressed,
                    },
                ));
            }

            // Collect sub-grid assignments for observer.
            let sg_assignments: Vec<(glam::IVec3, u32)> = ship_grid.0.iter_sub_grid_assignments().collect();

            tokio::spawn(async move {
                let mut writer = tcp_write.lock().await;
                for snapshot_msg in &chunk_snapshots {
                    let _ = client_listener::send_tcp_msg(&mut *writer, snapshot_msg).await;
                }
                if !sg_assignments.is_empty() {
                    let sg_msg = ServerMsg::SubGridAssignmentUpdate(SubGridAssignmentData {
                        assignments: sg_assignments,
                    });
                    let _ = client_listener::send_tcp_msg(&mut *writer, &sg_msg).await;
                }
                info!(chunks = chunk_snapshots.len(), "sent chunk snapshots to observer");
            });

            continue; // Don't create a player entity for observers.
        }

        // Check if this player has a pending handoff (re-entering the ship).
        let spawn_pos = if let Some(handoff_info) = pending_handoffs.0.remove(&player_name) {
            // Re-entry: use the handoff position (ship-local).
            let sys_pos = handoff_info.position;
            let ship_pos = exterior.position;
            let ship_rot = exterior.rotation;
            let local = ship_rot.inverse() * (sys_pos - ship_pos);
            Vec3::new(local.x as f32, local.y as f32, local.z as f32)
        } else {
            // Fresh connect: spawn at cockpit position (computed at startup).
            default_spawn.0
        };

        // Spawn player entity with Rapier body.
        spawn_player(
            &mut commands,
            &mut rapier,
            &mut player_index,
            token,
            player_name,
            spawn_pos,
            PlayerPhysics::default(),
        );

        let tcp_write = event.tcp_write.clone();
        let ship_pos = exterior.position;
        let ship_rot = exterior.rotation;
        let game_time = scene.game_time;
        let ship_id = config.ship_id;
        let galaxy_seed = config.galaxy_seed;
        let system_seed = config.system_seed;

        // Serialize all chunk snapshots for the initial sync.
        let mut chunk_snapshots: Vec<ServerMsg> = Vec::new();
        for (chunk_key, chunk) in ship_grid.0.iter_chunks() {
            if chunk.is_empty() {
                continue;
            }
            let compressed = block::serialize_chunk(chunk);
            chunk_snapshots.push(ServerMsg::ChunkSnapshot(
                voxeldust_core::client_message::ChunkSnapshotData {
                    chunk_x: chunk_key.x,
                    chunk_y: chunk_key.y,
                    chunk_z: chunk_key.z,
                    seq: chunk.edit_seq(),
                    data: compressed,
                },
            ));
        }

        // Collect sub-grid block assignments for initial sync.
        let sg_assignments: Vec<(glam::IVec3, u32)> = ship_grid.0.iter_sub_grid_assignments().collect();

        // Look up the host system shard (for always-on scene secondary).
        let system_preconnect = if let Ok(reg) = bridge.peer_registry.try_read() {
            reg.find_by_type(ShardType::System).first().map(|info| {
                handoff::ShardPreConnect {
                    shard_type: 1, // System
                    tcp_addr: info.endpoint.tcp_addr.to_string(),
                    udp_addr: info.endpoint.udp_addr.to_string(),
                    seed: system_seed,
                    planet_index: 0,
                    reference_position: DVec3::ZERO,
                    reference_rotation: DQuat::IDENTITY,
                    shard_id: info.id.0,
                }
            })
        } else {
            None
        };

        tokio::spawn(async move {
            let jr = ServerMsg::JoinResponse(JoinResponseData {
                seed: ship_id,
                planet_radius: 0,
                player_id: token.0,
                spawn_position: DVec3::new(spawn_pos.x as f64, spawn_pos.y as f64, spawn_pos.z as f64),
                spawn_rotation: DQuat::IDENTITY,
                spawn_forward: DVec3::NEG_Z,
                session_token: token,
                shard_type: 2, // Ship
                galaxy_seed,
                system_seed,
                game_time,
                reference_position: ship_pos,
                reference_rotation: ship_rot,
            });
            let mut writer = tcp_write.lock().await;
            let _ = client_listener::send_tcp_msg(&mut *writer, &jr).await;

            // Always-on system scene secondary. Exempt from the client's
            // secondary cap; spans the session until a different system
            // shard takes over (e.g., after cross-system warp arrival).
            if let Some(pc) = system_preconnect {
                let _ = client_listener::send_tcp_msg(
                    &mut *writer,
                    &ServerMsg::ShardPreConnect(pc),
                )
                .await;
            }

            // Send all chunk snapshots immediately after JoinResponse.
            for snapshot_msg in &chunk_snapshots {
                let _ = client_listener::send_tcp_msg(&mut *writer, snapshot_msg).await;
            }

            // Send sub-grid block assignments so client can split meshes.
            if !sg_assignments.is_empty() {
                let sg_msg = ServerMsg::SubGridAssignmentUpdate(SubGridAssignmentData {
                    assignments: sg_assignments,
                });
                let _ = client_listener::send_tcp_msg(&mut *writer, &sg_msg).await;
                info!("sent sub-grid assignments to client");
            }

            info!(chunks = chunk_snapshots.len(), "sent initial chunk snapshots to client");
        });
    }
}

// ---------------------------------------------------------------------------
// Input processing
// ---------------------------------------------------------------------------

fn process_input(
    mut events: MessageReader<PlayerInputMsg>,
    player_index: Res<PlayerEntityIndex>,
    mut players: Query<(
        &PlayerBody,
        &mut PlayerYaw,
        &mut InputActions,
        &SeatedState,
        &mut SeatInputValues,
        &PlayerPhysics,
    ), With<Player>>,
    mut rapier: ResMut<RapierContext>,
) {
    for event in events.read() {
        let entity = match player_index.0.get(&event.session) {
            Some(&e) => e,
            None => continue,
        };
        let Ok((body, mut yaw, mut actions, seated, mut seat_input, physics)) =
            players.get_mut(entity)
        else {
            continue;
        };

        let input = &event.input;
        actions.previous = actions.current;
        actions.current = input.action;
        yaw.0 = input.look_yaw;

        if seated.seated {
            // Copy per-binding seat values from client input.
            if !input.seat_values.is_empty() {
                seat_input.0 = input.seat_values.clone();
            }
        } else {
            // Walking mode: apply velocity to Rapier body.
            let (sin_y, cos_y) = yaw.0.sin_cos();
            let fwd = Vec3::new(cos_y, 0.0, sin_y);
            let right = Vec3::new(-sin_y, 0.0, cos_y);

            let walk_speed = physics.walk_speed;
            let move_vel =
                fwd * input.movement[2] * walk_speed + right * input.movement[0] * walk_speed;

            if let Some(rb) = rapier.rigid_body_set.get_mut(body.0) {
                let current_vel = *rb.linvel();
                rb.set_linvel(vector![move_vel.x, current_vel.y, move_vel.z], true);

                if input.jump && current_vel.y.abs() < 0.1 {
                    rb.apply_impulse(vector![0.0, physics.jump_impulse, 0.0], true);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Preconnect check (atmosphere detection)
// ---------------------------------------------------------------------------

fn preconnect_check(
    mut atmo: ResMut<AtmosphereState>,
    players: Query<&SessionId, With<Player>>,
    config: Res<ShipConfig>,
    exterior: Res<ShipExterior>,
    scene: Res<SceneCache>,
    cached_sys: Res<CachedSystemParams>,
    bridge: Res<NetworkBridge>,
) {
    if atmo.preconnect_sent || players.is_empty() {
        return;
    }
    if config.system_seed == 0 {
        return;
    }

    let sys = match &cached_sys.0 {
        Some(sys) => sys,
        None => return,
    };

    // Fire preconnect as soon as we're CLOSE to atmosphere entry, not after.
    // Two triggers — whichever fires first:
    //  (a) already inside atmosphere (legacy; catches teleports / stationary spawns),
    //  (b) time-to-atmosphere < LEAD_TIME based on radial closing velocity.
    //
    // Trigger (b) is critical for fast approaches (warp-brake arrival, atmospheric
    // dives) where reaching atmosphere before the planet shard is ready would
    // cause a visible load pop. 2 s is enough to open TCP + UDP, observer-connect,
    // and pull the first chunk snapshots before the player enters atmosphere.
    const LEAD_TIME_S: f64 = 2.0;

    let candidate_planet = scene
        .bodies
        .iter()
        .filter(|b| b.body_id > 0)
        .find_map(|b| {
            let pi = (b.body_id - 1) as usize;
            if pi >= sys.planets.len() {
                return None;
            }
            let planet = &sys.planets[pi];
            if !planet.atmosphere.has_atmosphere {
                return None;
            }
            let to_ship = exterior.position - b.position;
            let dist = to_ship.length();
            let atmo_top_alt = planet.atmosphere.atmosphere_height;
            let atmo_top = planet.radius_m + atmo_top_alt;
            let alt = dist - planet.radius_m;
            // Already inside atmosphere → fire immediately.
            if alt < atmo_top_alt {
                return Some(pi);
            }
            // Closing velocity toward planet center (positive = approaching).
            let dir_in = (-to_ship).normalize_or_zero();
            let closing_speed = exterior.velocity.dot(dir_in);
            if closing_speed <= 0.0 {
                return None;
            }
            // Distance to atmosphere shell (not to planet center).
            let dist_to_atmo = (dist - atmo_top).max(0.0);
            let t = dist_to_atmo / closing_speed;
            if t <= LEAD_TIME_S {
                return Some(pi);
            }
            None
        });

    let was_in_atmo = atmo.in_atmosphere;
    atmo.in_atmosphere = candidate_planet.is_some();
    if was_in_atmo && !atmo.in_atmosphere {
        atmo.preconnect_sent = false;
    }

    let planet_index = match candidate_planet {
        Some(idx) => idx,
        None => return,
    };

    let planet_seed = match &cached_sys.0 {
        Some(sys) if planet_index < sys.planets.len() => sys.planets[planet_index].planet_seed,
        _ => return,
    };

    let planet_shard_info = if let Ok(reg) = bridge.peer_registry.try_read() {
        reg.find_by_type(ShardType::Planet)
            .iter()
            .find(|s| s.planet_seed == Some(planet_seed))
            .map(|s| {
                (
                    s.endpoint.tcp_addr.to_string(),
                    s.endpoint.udp_addr.to_string(),
                )
            })
    } else {
        None
    };

    let (tcp_addr, udp_addr) = match planet_shard_info {
        Some(info) => info,
        None => return,
    };

    let pc = handoff::ShardPreConnect {
        shard_type: 0,
        tcp_addr,
        udp_addr,
        seed: planet_seed,
        planet_index: planet_index as u32,
        reference_position: exterior.position,
        reference_rotation: DQuat::IDENTITY,
        shard_id: 0, // Planet shards use seed for identification
    };

    atmo.preconnect_sent = true;

    // Send to ALL connected players.
    let sessions: Vec<SessionToken> = players.iter().map(|sid| sid.0).collect();
    let player_count = sessions.len();
    let cr = bridge.client_registry.clone();
    let msg = ServerMsg::ShardPreConnect(pc);
    tokio::spawn(async move {
        if let Ok(reg) = cr.try_read() {
            for session in &sessions {
                if let Err(e) = reg.send_tcp(*session, &msg).await {
                    tracing::warn!(%e, session = session.0, "failed to send ShardPreConnect");
                }
            }
        }
    });

    info!(planet_seed, planet_index, players = player_count, "sent ShardPreConnect to all players");
}


/// Detect when any player has left the ship's hull volume and trigger
/// handoff. Per-player: each player checked independently.
/// - In atmosphere: hand off to planet shard (existing behavior).
/// - In space: hand off to system shard as EVA (future Phase 4).
fn hull_exit_check(
    players: Query<
        (Entity, &SessionId, &PlayerName, &PlayerPosition),
        (With<Player>, Without<HandoffPending>),
    >,
    hull_bounds: Res<ShipHullBounds>,
    exterior: Res<ShipExterior>,
    scene: Res<SceneCache>,
    cached_sys: Res<CachedSystemParams>,
    config: Res<ShipConfig>,
    tick: Res<ecs::TickCounter>,
    mut pending: ResMut<PendingMessages>,
    mut commands: Commands,
) {
    for (entity, session_id, player_name, player_pos) in &players {
        // Check if this player is outside the ship's solid block bounding box.
        let pos = player_pos.0;
        let margin = hull_bounds.margin;
        let inside = pos.x >= hull_bounds.min.x as f32 - margin
            && pos.x <= hull_bounds.max.x as f32 + 1.0 + margin
            && pos.y >= hull_bounds.min.y as f32 - margin
            && pos.y <= hull_bounds.max.y as f32 + 1.0 + margin
            && pos.z >= hull_bounds.min.z as f32 - margin
            && pos.z <= hull_bounds.max.z as f32 + 1.0 + margin;

        if inside {
            continue;
        }

        let player_local = DVec3::new(pos.x as f64, pos.y as f64, pos.z as f64);
        let player_system_pos = exterior.position + exterior.rotation * player_local;

        // Determine handoff target based on atmosphere state.
        if exterior.in_atmosphere {
            // In atmosphere: hand off to closest planet shard.
            let closest_planet = scene
                .bodies
                .iter()
                .filter(|b| b.body_id > 0)
                .min_by_key(|b| ((b.position - exterior.position).length() * 1000.0) as u64);

            if let Some(planet_body) = closest_planet {
                let planet_index = (planet_body.body_id - 1) as usize;
                if let Some(ref sys) = cached_sys.0 {
                    if planet_index < sys.planets.len() {
                        let planet_seed = sys.planets[planet_index].planet_seed;
                        let h = handoff::PlayerHandoff {
                            session_token: session_id.0,
                            player_name: player_name.0.clone(),
                            position: player_system_pos,
                            velocity: exterior.velocity,
                            rotation: DQuat::IDENTITY,
                            forward: exterior.rotation * DVec3::NEG_Z,
                            fly_mode: false,
                            speed_tier: 0,
                            grounded: false,
                            health: 100.0,
                            shield: 100.0,
                            source_shard: config.shard_id,
                            source_tick: tick.0,
                            target_star_index: None,
                            galaxy_context: None,
                            target_planet_seed: Some(planet_seed),
                            target_planet_index: Some(planet_index as u32),
                            target_ship_id: None,
                            target_ship_shard_id: None,
                            ship_system_position: Some(exterior.position),
                            ship_rotation: Some(exterior.rotation),
                            game_time: scene.game_time,
                            warp_target_star_index: None,
                            warp_velocity_gu: None,
                            target_system_eva: false,
                        };
                        commands.entity(entity).insert(HandoffPending);
                        pending.handoffs.push(ShardMsg::PlayerHandoff(h));
                        info!(
                            session = session_id.0.0,
                            planet_index,
                            planet_seed,
                            "hull exit in atmosphere — planet handoff initiated"
                        );
                    }
                }
            }
        } else {
            // In space: EVA handoff to system shard.
            // Player inherits ship velocity + local walking velocity rotated to system frame.
            let eva_velocity = exterior.velocity + exterior.rotation * player_local * 0.0; // local pos isn't velocity — use ship vel
            let h = handoff::PlayerHandoff {
                session_token: session_id.0,
                player_name: player_name.0.clone(),
                position: player_system_pos,
                velocity: exterior.velocity, // inherit ship velocity for momentum continuity
                rotation: exterior.rotation,
                forward: exterior.rotation * DVec3::NEG_Z,
                fly_mode: true, // EVA players start in fly mode
                speed_tier: 0,
                grounded: false,
                health: 100.0,
                shield: 100.0,
                source_shard: config.shard_id,
                source_tick: tick.0,
                target_star_index: None,
                galaxy_context: None,
                target_planet_seed: None,
                target_planet_index: None,
                target_ship_id: None,
                target_ship_shard_id: None,
                ship_system_position: Some(exterior.position),
                ship_rotation: Some(exterior.rotation),
                game_time: scene.game_time,
                warp_target_star_index: None,
                warp_velocity_gu: None,
                target_system_eva: true,
            };
            commands.entity(entity).insert(HandoffPending);
            pending.handoffs.push(ShardMsg::PlayerHandoff(h));
            info!(
                session = session_id.0.0,
                ship_vel = format!("{:.1} m/s", exterior.velocity.length()),
                "hull exit in space — EVA handoff to system shard"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// QUIC send
// ---------------------------------------------------------------------------

fn pilot_send(
    config: Res<ShipConfig>,
    pilot_acc: Res<PilotAccumulator>,
    mut autopilot_query: Query<&mut AutopilotBlockState>,
    mut warp_query: Query<&mut WarpComputerState>,
    mut pending: ResMut<PendingMessages>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
) {
    let Some(host_id) = config.host_shard_id else {
        if tick.0 % 100 == 0 {
            info!("no host_shard_id");
        }
        return;
    };

    let Ok(reg) = bridge.peer_registry.try_read() else {
        if tick.0 % 100 == 0 {
            info!("peer_reg lock failed");
        }
        return;
    };

    let Some(addr) = reg.quic_addr(host_id) else {
        if tick.0 % 100 == 0 {
            let all_peers: Vec<_> = reg
                .all()
                .iter()
                .map(|s| format!("{}({})", s.id, s.shard_type))
                .collect();
            info!(host = host_id.0, peers = ?all_peers, "host shard not found in peer registry");
        }
        return;
    };

    // Send all pending handoffs.
    for handoff_msg in pending.handoffs.drain(..) {
        let _ = bridge
            .quic_send_tx
            .try_send((host_id, addr, handoff_msg));
    }

    // Send pending autopilot commands from all autopilot entities.
    for mut ap in &mut autopilot_query {
        if let Some((target, tier)) = ap.pending_cmd.take() {
            let ap_msg = ShardMsg::AutopilotCommand(AutopilotCommandData {
                ship_id: config.ship_id,
                target_body_id: target,
                speed_tier: tier,
                autopilot_mode: 0,
            });
            let _ = bridge.quic_send_tx.try_send((host_id, addr, ap_msg));
        }
    }

    // Send pending warp commands from all warp computer entities.
    for mut wc in &mut warp_query {
        if let Some(target_star) = wc.pending_cmd.take() {
            let warp_msg = ShardMsg::WarpAutopilotCommand(WarpAutopilotCommandData {
                ship_id: config.ship_id,
                target_star_index: target_star,
                galaxy_seed: config.galaxy_seed,
            });
            let _ = bridge.quic_send_tx.try_send((host_id, addr, warp_msg));
        }
    }

    // Always send thrust/torque — automated systems (hover, flight computer)
    // need thrusters to work even without a pilot in the seat.
    let msg = ShardMsg::ShipControlInput(ShipControlInput {
        ship_id: config.ship_id,
        thrust: pilot_acc.thrust,
        torque: pilot_acc.rotation_command,
        braking: false,
        tick: tick.0,
    });
    let _ = bridge.quic_send_tx.try_send((host_id, addr, msg));
}

// ---------------------------------------------------------------------------
// Physics
// ---------------------------------------------------------------------------

/// Detect which sub-grid the player is standing on and apply its motion delta.
/// Runs BEFORE physics_step so the player moves with the platform before gravity/collision.
fn update_player_platform(
    mut rapier: ResMut<RapierContext>,
    mut players: Query<(&PlayerBody, &mut PlayerPlatformState), With<Player>>,
    sub_grids: Res<SubGridRegistry>,
    world_isos: Res<MechanicalWorldIsometries>,
) {
    use voxeldust_core::ecs::components::SubGridId;

    for (body_comp, mut platform) in &mut players {
        let player_pos = match rapier.rigid_body_set.get(body_comp.0) {
            Some(body) => *body.translation(),
            None => continue,
        };

        // Downward raycast from player feet to detect which collider is below.
        let ray_origin = rapier3d::math::Point::new(player_pos.x, player_pos.y - 0.1, player_pos.z);
        let ray_dir = rapier3d::math::Vector::new(0.0, -1.0, 0.0);
        let max_dist = 0.5; // half a block below feet

        // Update query pipeline and perform raycast.
        let ctx = &mut *rapier;
        ctx.query_pipeline.update(&ctx.collider_set);
        let hit = ctx.query_pipeline.cast_ray(
            &ctx.rigid_body_set,
            &ctx.collider_set,
            &rapier3d::geometry::Ray::new(ray_origin, ray_dir),
            max_dist,
            true,
            rapier3d::pipeline::QueryFilter::default().exclude_rigid_body(body_comp.0),
        );

        // Determine which sub-grid (if any) the hit collider belongs to.
        let mut riding_sg: Option<SubGridId> = None;
        if let Some((collider_handle, _dist)) = hit {
            if let Some(collider) = ctx.collider_set.get(collider_handle) {
                if let Some(parent_body) = collider.parent() {
                    for (sg_id, sg_data) in &sub_grids.grids {
                        if sg_data.body_handle == parent_body {
                            riding_sg = Some(*sg_id);
                            break;
                        }
                    }
                }
            }
        }

        // Apply platform delta if riding a sub-grid.
        if let Some(sg_id) = riding_sg {
            if let Some(&current_iso) = world_isos.0.get(&sg_id) {
                if platform.riding == Some(sg_id) {
                    if let Some(prev_iso) = platform.prev_iso {
                        let delta = current_iso * prev_iso.inverse();
                        if let Some(body) = ctx.rigid_body_set.get_mut(body_comp.0) {
                            let player_t = *body.translation();
                            let new_pos = delta * rapier3d::math::Point::new(player_t.x, player_t.y, player_t.z);
                            body.set_translation(rapier3d::math::Vector::new(new_pos.x, new_pos.y, new_pos.z), true);
                        }
                    }
                }
                platform.prev_iso = Some(current_iso);
            }
            platform.riding = Some(sg_id);
        } else {
            platform.riding = None;
            platform.prev_iso = None;
        }
    }
}

fn physics_step(
    mut rapier: ResMut<RapierContext>,
    gravity_sources: Res<GravitySources>,
    gravity_enabled: Res<GravityEnabled>,
    mut players: Query<(&PlayerBody, &mut PlayerPosition), With<Player>>,
) {
    // Compute gravity using the first player's position (ship-internal gravity
    // is uniform, so any player's position gives the same result).
    let sample_pos = players
        .iter()
        .next()
        .map(|(_, p)| p.0)
        .unwrap_or(Vec3::ZERO);

    let gravity = if gravity_enabled.0 {
        let mut total = vector![0.0, 0.0, 0.0];
        for source in &gravity_sources.0 {
            total += source.gravity_at(sample_pos);
        }
        total
    } else {
        vector![0.0, 0.0, 0.0]
    };

    // Destructure to satisfy the borrow checker — physics_pipeline.step() needs
    // mutable references to multiple fields simultaneously.
    let ctx = &mut *rapier;
    ctx.physics_pipeline.step(
        &gravity,
        &ctx.integration_params,
        &mut ctx.island_manager,
        &mut ctx.broad_phase,
        &mut ctx.narrow_phase,
        &mut ctx.rigid_body_set,
        &mut ctx.collider_set,
        &mut ctx.impulse_joint_set,
        &mut ctx.multibody_joint_set,
        &mut ctx.ccd_solver,
        Some(&mut ctx.query_pipeline),
        &(),
        &(),
    );

    // Sync positions from Rapier for all players.
    for (body_comp, mut pos) in &mut players {
        if let Some(body) = ctx.rigid_body_set.get(body_comp.0) {
            let t = body.translation();
            pos.0 = Vec3::new(t.x, t.y, t.z);
        }
    }
}

fn tick_counter(mut tick: ResMut<ecs::TickCounter>) {
    tick.0 += 1;
}

// ---------------------------------------------------------------------------
// Broadcast
// ---------------------------------------------------------------------------

fn broadcast_world_state(
    players: Query<(&SessionId, &PlayerName, &PlayerPosition, &SeatedState), With<Player>>,
    exterior: Res<ShipExterior>,
    scene: Res<SceneCache>,
    warp_query: Query<&WarpComputerState>,
    autopilot_cache: Res<AutopilotSnapshotCache>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
    rapier: Res<RapierContext>,
    sub_grids: Res<SubGridRegistry>,
    mechanicals: Query<&MechanicalState>,
    config: Res<ShipConfig>,
    external: Res<ExternalEntities>,
) {
    let scene_stale = tick.0 > scene.last_update_tick + 20;
    let bodies: Vec<CelestialBodyData> = if scene_stale {
        Vec::new()
    } else {
        scene
            .bodies
            .iter()
            .map(|b| CelestialBodyData {
                body_id: b.body_id,
                position: b.position,
                radius: b.radius,
                color: b.color,
            })
            .collect()
    };

    let lighting = if scene_stale {
        None
    } else {
        scene.lighting.as_ref().map(|l| LightingData {
            sun_direction: l.sun_direction,
            sun_color: l.sun_color,
            sun_intensity: l.sun_intensity,
            ambient: l.ambient,
        })
    };

    // Read warp target from the first warp computer entity (if any).
    let warp_target = warp_query.iter()
        .find_map(|wc| wc.target_star_index)
        .unwrap_or(0xFFFFFFFF);

    // Build player snapshots for ALL players.
    let player_snapshots: Vec<PlayerSnapshotData> = players
        .iter()
        .map(|(sid, _, pos, seated)| PlayerSnapshotData {
            player_id: sid.0.0,
            position: DVec3::new(pos.0.x as f64, pos.0.y as f64, pos.0.z as f64),
            rotation: exterior.rotation,
            velocity: exterior.velocity,
            grounded: !seated.seated,
            health: 100.0,
            shield: 100.0,
            seated: seated.seated,
        })
        .collect();

    // Build unified entities list:
    // - External entities from the host system shard (transformed ship-local).
    // - Every seated/onboard player on this ship as a Seated entity.
    // WorldState.origin == exterior.position (ship system-space). External
    // entities arrive in system-space — shift by -exterior.position so the
    // client can reconstruct world-space via `entity.position + origin`.
    use voxeldust_core::client_message::{EntityKind, LodTier, ObservableEntityData};
    let mut entities: Vec<ObservableEntityData> = Vec::with_capacity(
        external.entities.len() + players.iter().count(),
    );
    for e in &external.entities {
        let mut entity = e.clone();
        entity.position = e.position - exterior.position;
        // If this entry is our own ship, mark it — client uses this to bind camera.
        if matches!(entity.kind, EntityKind::Ship) && entity.entity_id == config.ship_id {
            entity.is_own = true;
        }
        entities.push(entity);
    }
    for (sid, name, pos, seated) in players.iter() {
        let kind = if seated.seated {
            EntityKind::Seated
        } else {
            EntityKind::Seated
        };
        entities.push(ObservableEntityData {
            entity_id: sid.0.0,
            kind,
            position: DVec3::new(pos.0.x as f64, pos.0.y as f64, pos.0.z as f64),
            rotation: exterior.rotation,
            velocity: exterior.velocity,
            bounding_radius: 1.0,
            lod_tier: LodTier::Full,
            shard_id: config.shard_id.0,
            shard_type: voxeldust_core::shard_types::ShardType::Ship as u8,
            is_own: false,
            name: name.0.clone(),
            health: 100.0,
            shield: 100.0,
        });
    }

    let mut ws = ServerMsg::WorldState(WorldStateData {
        tick: tick.0,
        origin: exterior.position,
        players: player_snapshots,
        bodies,
        ships: vec![],
        lighting,
        game_time: scene.game_time,
        warp_target_star_index: warp_target,
        autopilot: autopilot_cache.snapshot.clone(),
        sub_grids: Vec::new(), // Populated below after WorldStateData construction.
        entities,
    });
    // Build sub-grid transforms from MechanicalState (clean joint angle + axis)
    // instead of raw Rapier body quaternions (which have solver noise on off-axes).
    if let ServerMsg::WorldState(ref mut ws_data) = ws {
        ws_data.sub_grids = mechanicals.iter().filter_map(|state| {
            let sg_data = sub_grids.grids.get(&state.child_grid_id)?;
            // Read the body's world transform directly from Rapier.
            // Kinematic bodies have no solver noise — the transform was set exactly
            // by apply_mechanical_transforms (including parent chain composition).
            let body = rapier.rigid_body_set.get(sg_data.body_handle)?;
            let t = body.translation();
            let r = body.rotation();
            // Compute the original root-space anchor (fixed per sub-grid, never changes).
            let axis_offset = block::sub_block::face_to_offset(sg_data.mount_face);
            let anchor = glam::Vec3::new(
                sg_data.mount_pos.x as f32 + 0.5 + axis_offset.x as f32 * 0.5,
                sg_data.mount_pos.y as f32 + 0.5 + axis_offset.y as f32 * 0.5,
                sg_data.mount_pos.z as f32 + 0.5 + axis_offset.z as f32 * 0.5,
            );
            Some(SubGridTransformData {
                sub_grid_id: state.child_grid_id.0,
                translation: glam::Vec3::new(t.x, t.y, t.z),
                rotation: glam::Quat::from_xyzw(r.i, r.j, r.k, r.w),
                parent_grid: sg_data.parent_grid.0,
                anchor,
                mount_pos: sg_data.mount_pos,
                mount_face: sg_data.mount_face,
                joint_type: match state.joint_type {
                    block::JointType::Revolute => 0,
                    block::JointType::Prismatic => 1,
                },
                current_value: state.current,
            })
        }).collect();
    }
    if bridge.broadcast_tx.try_send(ws).is_err() {
        tracing::warn!("WorldState broadcast dropped — channel full");
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

fn log_state(
    players: Query<(&PlayerPosition, &SeatedState), With<Player>>,
    gravity_enabled: Res<GravityEnabled>,
    tick: Res<ecs::TickCounter>,
    scene: Res<SceneCache>,
    exterior: Res<ShipExterior>,
) {
    if tick.0 % 100 != 0 {
        return;
    }

    let player_count = players.iter().count();
    let first = players.iter().next();
    let (pos_str, piloting) = match first {
        Some((pos, seated)) => (
            format!("({:.1}, {:.1}, {:.1})", pos.0.x, pos.0.y, pos.0.z),
            seated.seated,
        ),
        None => ("(no players)".to_string(), false),
    };

    info!(
        pos = pos_str,
        piloting,
        gravity = gravity_enabled.0,
        player_count,
        tick = tick.0,
        "ship state"
    );

    // Planet distance diagnostic.
    if tick.0 % 20 == 0 && !scene.bodies.is_empty() {
        let nearest = scene
            .bodies
            .iter()
            .filter(|b| b.body_id > 0)
            .min_by_key(|b| ((exterior.position - b.position).length() * 1000.0) as u64);
        if let Some(body) = nearest {
            let dist = (exterior.position - body.position).length();
            let stale_ticks = tick.0.saturating_sub(scene.last_update_tick);
            tracing::warn!(
                dist = format!("{:.0}", dist),
                stale = stale_ticks,
                body_id = body.body_id,
                ship_pos = format!(
                    "({:.0},{:.0},{:.0})",
                    exterior.position.x, exterior.position.y, exterior.position.z
                ),
                body_pos = format!(
                    "({:.0},{:.0},{:.0})",
                    body.position.x, body.position.y, body.position.z
                ),
                "SHIP-SHARD planet distance diagnostic"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Block persistence (redb)
// ---------------------------------------------------------------------------

/// Ship block persistence — saves modified chunks to an embedded database.
///
/// Chunks are marked dirty by `apply_block_edits`. Every `SAVE_INTERVAL_TICKS`
/// ticks, all dirty chunks are batch-written to redb. On startup, if a database
/// file exists, chunks are loaded from it instead of using `build_starter_ship`.
#[derive(Resource)]
struct ShipPersistence {
    db: redb::Database,
    /// Chunks modified since the last save.
    pending_saves: std::collections::HashSet<glam::IVec3>,
    /// Tick counter at last save.
    last_save_tick: u64,
}

/// Save interval: every 60 ticks (3 seconds at 20Hz).
const SAVE_INTERVAL_TICKS: u64 = 60;

/// redb table definition for ship chunk data.
const SHIP_CHUNKS_TABLE: redb::TableDefinition<&[u8], &[u8]> = redb::TableDefinition::new("ship_chunks");

/// redb table definition for per-block power configs and channel overrides.
const BLOCK_CONFIGS_TABLE: redb::TableDefinition<&[u8], &[u8]> = redb::TableDefinition::new("block_configs");

/// Encode a chunk key as bytes for redb key.
fn chunk_key_bytes(key: glam::IVec3) -> [u8; 12] {
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&key.x.to_le_bytes());
    buf[4..8].copy_from_slice(&key.y.to_le_bytes());
    buf[8..12].copy_from_slice(&key.z.to_le_bytes());
    buf
}

/// Decode a chunk key from redb bytes.
fn chunk_key_from_bytes(buf: &[u8]) -> glam::IVec3 {
    glam::IVec3::new(
        i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
        i32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
        i32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
    )
}

/// Per-block config record for persistence. Stores channel override and power config.
struct BlockConfigRecord {
    /// Signal throttle channel name (empty = none). Used by starter ship for pre-configured blocks.
    channel_override: String,
    /// Boost channel name (empty = none).
    boost_channel: String,
    power_config: Option<block::PowerConfig>,
    /// Full subscribe bindings (channel_name, property).
    subscribe_bindings: Vec<(String, u8)>,
    /// Full publish bindings (channel_name, property).
    publish_bindings: Vec<(String, u8)>,
    /// Seat config (generic bindings + seated channel name).
    seat_config: block::SavedSeatConfig,
}

/// Serialize a BlockConfigRecord to bytes.
///
/// Format:
///   - channel_override: u32 LE length, then UTF-8 bytes
///   - power_config tag: 0=None, 1=Source, 2=Consumer
///   - Source: u32 LE circuit count, then per circuit: (u32 LE name_len, name bytes, f32 LE fraction), f32 LE broadcast_range
///   - Consumer: 3×i32 LE reactor_pos, u32 LE circuit_len, circuit bytes
fn serialize_block_config(record: &BlockConfigRecord) -> Vec<u8> {
    let mut buf = Vec::new();
    // channel_override
    let co_bytes = record.channel_override.as_bytes();
    buf.extend_from_slice(&(co_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(co_bytes);
    // boost_channel
    let bc_bytes = record.boost_channel.as_bytes();
    buf.extend_from_slice(&(bc_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bc_bytes);
    // power_config
    match &record.power_config {
        None => buf.push(0),
        Some(block::PowerConfig::Source { circuits, broadcast_range }) => {
            buf.push(1);
            buf.extend_from_slice(&(circuits.len() as u32).to_le_bytes());
            for (name, fraction) in circuits {
                let name_bytes = name.as_bytes();
                buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(name_bytes);
                buf.extend_from_slice(&fraction.to_le_bytes());
            }
            buf.extend_from_slice(&broadcast_range.to_le_bytes());
        }
        Some(block::PowerConfig::Consumer { reactor_pos, circuit }) => {
            buf.push(2);
            buf.extend_from_slice(&reactor_pos.x.to_le_bytes());
            buf.extend_from_slice(&reactor_pos.y.to_le_bytes());
            buf.extend_from_slice(&reactor_pos.z.to_le_bytes());
            let circuit_bytes = circuit.as_bytes();
            buf.extend_from_slice(&(circuit_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(circuit_bytes);
        }
    }
    // subscribe_bindings: u32 count, then per binding: (u32 name_len, name, u8 property)
    buf.extend_from_slice(&(record.subscribe_bindings.len() as u32).to_le_bytes());
    for (name, prop) in &record.subscribe_bindings {
        let nb = name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        buf.extend_from_slice(nb);
        buf.push(*prop);
    }
    // publish_bindings: same format
    buf.extend_from_slice(&(record.publish_bindings.len() as u32).to_le_bytes());
    for (name, prop) in &record.publish_bindings {
        let nb = name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        buf.extend_from_slice(nb);
        buf.push(*prop);
    }
    // seat_config: version_byte(2) + u32 binding_count, per binding:
    //   (u32 label_len, label, u8 source, u32 key_name_len, key_name, u8 key_mode, u8 axis_dir, u32 ch_len, ch_name, u8 property)
    // then: u32 seated_ch_len, seated_ch_name
    buf.push(2); // format version 2 (generic seat)
    buf.extend_from_slice(&(record.seat_config.bindings.len() as u32).to_le_bytes());
    for (label, source, key_name, key_mode, axis_dir, channel_name, property) in &record.seat_config.bindings {
        let lb = label.as_bytes();
        buf.extend_from_slice(&(lb.len() as u32).to_le_bytes());
        buf.extend_from_slice(lb);
        buf.push(*source);
        let kb = key_name.as_bytes();
        buf.extend_from_slice(&(kb.len() as u32).to_le_bytes());
        buf.extend_from_slice(kb);
        buf.push(*key_mode);
        buf.push(*axis_dir);
        let cb = channel_name.as_bytes();
        buf.extend_from_slice(&(cb.len() as u32).to_le_bytes());
        buf.extend_from_slice(cb);
        buf.push(*property);
    }
    let scb = record.seat_config.seated_channel_name.as_bytes();
    buf.extend_from_slice(&(scb.len() as u32).to_le_bytes());
    buf.extend_from_slice(scb);
    buf
}

/// Deserialize a BlockConfigRecord from bytes.
fn deserialize_block_config(data: &[u8]) -> Option<BlockConfigRecord> {
    let mut pos = 0usize;
    if data.len() < 4 { return None; }

    // channel_override
    let co_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
    pos += 4;
    if pos + co_len > data.len() { return None; }
    let channel_override = std::str::from_utf8(&data[pos..pos+co_len]).ok()?.to_string();
    pos += co_len;

    // boost_channel
    if pos + 4 > data.len() { return Some(BlockConfigRecord { channel_override, boost_channel: String::new(), power_config: None, subscribe_bindings: Vec::new(), publish_bindings: Vec::new(), seat_config: block::SavedSeatConfig::default() }); }
    let bc_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
    pos += 4;
    if pos + bc_len > data.len() { return None; }
    let boost_channel = std::str::from_utf8(&data[pos..pos+bc_len]).ok()?.to_string();
    pos += bc_len;

    if pos >= data.len() { return None; }
    let tag = data[pos];
    pos += 1;

    let power_config = match tag {
        0 => None,
        1 => {
            // Source
            if pos + 4 > data.len() { return None; }
            let circuit_count = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
            pos += 4;
            let mut circuits = Vec::with_capacity(circuit_count);
            for _ in 0..circuit_count {
                if pos + 4 > data.len() { return None; }
                let name_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
                pos += 4;
                if pos + name_len > data.len() { return None; }
                let name = std::str::from_utf8(&data[pos..pos+name_len]).ok()?.to_string();
                pos += name_len;
                if pos + 4 > data.len() { return None; }
                let fraction = f32::from_le_bytes(data[pos..pos+4].try_into().ok()?);
                pos += 4;
                circuits.push((name, fraction));
            }
            if pos + 4 > data.len() { return None; }
            let broadcast_range = f32::from_le_bytes(data[pos..pos+4].try_into().ok()?);
            pos += 4;
            Some(block::PowerConfig::Source { circuits, broadcast_range })
        }
        2 => {
            // Consumer
            if pos + 12 > data.len() { return None; }
            let rx = i32::from_le_bytes(data[pos..pos+4].try_into().ok()?);
            pos += 4;
            let ry = i32::from_le_bytes(data[pos..pos+4].try_into().ok()?);
            pos += 4;
            let rz = i32::from_le_bytes(data[pos..pos+4].try_into().ok()?);
            pos += 4;
            if pos + 4 > data.len() { return None; }
            let circuit_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
            pos += 4;
            if pos + circuit_len > data.len() { return None; }
            let circuit = std::str::from_utf8(&data[pos..pos+circuit_len]).ok()?.to_string();
            pos += circuit_len;
            Some(block::PowerConfig::Consumer {
                reactor_pos: glam::IVec3::new(rx, ry, rz),
                circuit,
            })
        }
        _ => return None,
    };

    // Subscribe bindings (optional — backward compatible with old records).
    let mut subscribe_bindings = Vec::new();
    if pos + 4 <= data.len() {
        let count = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
        pos += 4;
        for _ in 0..count {
            if pos + 4 > data.len() { break; }
            let name_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
            pos += 4;
            if pos + name_len + 1 > data.len() { break; }
            let name = std::str::from_utf8(&data[pos..pos+name_len]).unwrap_or("").to_string();
            pos += name_len;
            let prop = data[pos];
            pos += 1;
            subscribe_bindings.push((name, prop));
        }
    }

    // Publish bindings (optional).
    let mut publish_bindings = Vec::new();
    if pos + 4 <= data.len() {
        let count = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
        pos += 4;
        for _ in 0..count {
            if pos + 4 > data.len() { break; }
            let name_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
            pos += 4;
            if pos + name_len + 1 > data.len() { break; }
            let name = std::str::from_utf8(&data[pos..pos+name_len]).unwrap_or("").to_string();
            pos += name_len;
            let prop = data[pos];
            pos += 1;
            publish_bindings.push((name, prop));
        }
    }

    // Seat config (optional). Format version byte determines layout:
    //   v1 (old): no version byte, starts with u32 count — skip (use preset defaults)
    //   v2 (new): version_byte(2) + bindings + seated_channel_name
    let mut seat_config = block::SavedSeatConfig::default();
    if pos < data.len() {
        let version = data[pos];
        if version == 2 {
            // v2: generic seat format
            pos += 1;
            if pos + 4 <= data.len() {
                let count = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
                pos += 4;
                for _ in 0..count {
                    // label
                    if pos + 4 > data.len() { break; }
                    let label_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
                    pos += 4;
                    if pos + label_len > data.len() { break; }
                    let label = std::str::from_utf8(&data[pos..pos+label_len]).unwrap_or("").to_string();
                    pos += label_len;
                    // source
                    if pos + 1 > data.len() { break; }
                    let source = data[pos]; pos += 1;
                    // key_name
                    if pos + 4 > data.len() { break; }
                    let kn_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
                    pos += 4;
                    if pos + kn_len > data.len() { break; }
                    let key_name = std::str::from_utf8(&data[pos..pos+kn_len]).unwrap_or("").to_string();
                    pos += kn_len;
                    // key_mode, axis_dir
                    if pos + 2 > data.len() { break; }
                    let key_mode = data[pos]; pos += 1;
                    let axis_dir = data[pos]; pos += 1;
                    // channel_name
                    if pos + 4 > data.len() { break; }
                    let ch_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
                    pos += 4;
                    if pos + ch_len > data.len() { break; }
                    let channel_name = std::str::from_utf8(&data[pos..pos+ch_len]).unwrap_or("").to_string();
                    pos += ch_len;
                    // property
                    if pos + 1 > data.len() { break; }
                    let property = data[pos]; pos += 1;
                    seat_config.bindings.push((label, source, key_name, key_mode, axis_dir, channel_name, property));
                }
                // seated_channel_name
                if pos + 4 <= data.len() {
                    let sc_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
                    pos += 4;
                    if pos + sc_len <= data.len() {
                        seat_config.seated_channel_name = std::str::from_utf8(&data[pos..pos+sc_len]).unwrap_or("").to_string();
                        let _ = pos + sc_len; // consume
                    }
                }
            }
        }
        // v1 (old format) or unknown: seat_config stays empty → uses preset defaults
    }

    Some(BlockConfigRecord { channel_override, boost_channel, power_config, subscribe_bindings, publish_bindings, seat_config })
}

/// Load block configs from redb into a ShipGrid's channel_overrides and power_configs.
fn load_block_configs(db: &redb::Database, grid: &mut ShipGrid) {
    let txn = match db.begin_read() {
        Ok(t) => t,
        Err(_) => return,
    };
    let table = match txn.open_table(BLOCK_CONFIGS_TABLE) {
        Ok(t) => t,
        Err(_) => return, // table doesn't exist yet
    };

    let mut count = 0usize;
    let mut iter = match table.range::<&[u8]>(..) {
        Ok(i) => i,
        Err(_) => return,
    };
    while let Some(entry) = iter.next() {
        let (key, value) = match entry {
            Ok(kv) => kv,
            Err(_) => continue,
        };
        let key_bytes = key.value();
        if key_bytes.len() < 12 { continue; }
        let block_pos = chunk_key_from_bytes(key_bytes);
        if let Some(record) = deserialize_block_config(value.value()) {
            if !record.channel_override.is_empty() {
                grid.set_channel_override(block_pos.x, block_pos.y, block_pos.z, &record.channel_override);
            }
            if !record.boost_channel.is_empty() {
                grid.set_boost_channel(block_pos.x, block_pos.y, block_pos.z, &record.boost_channel);
            }
            if let Some(cfg) = record.power_config {
                grid.set_power_config(block_pos.x, block_pos.y, block_pos.z, cfg);
            }
            if !record.subscribe_bindings.is_empty() || !record.publish_bindings.is_empty() {
                grid.set_saved_signal_bindings(block_pos, record.subscribe_bindings, record.publish_bindings);
            }
            if !record.seat_config.bindings.is_empty() || !record.seat_config.seated_channel_name.is_empty() {
                grid.set_saved_seat_config(block_pos, record.seat_config);
            }
            count += 1;
        }
    }
    if count > 0 {
        info!(configs = count, "loaded block configs from persistence");
    }
}

/// Save ALL channel overrides and power configs from the grid to redb.
fn save_block_configs(db: &redb::Database, grid: &ShipGrid) {
    let txn = match db.begin_write() {
        Ok(t) => t,
        Err(e) => {
            warn!("redb write transaction failed for block configs: {e}");
            return;
        }
    };
    {
        let mut table = match txn.open_table(BLOCK_CONFIGS_TABLE) {
            Ok(t) => t,
            Err(e) => {
                warn!("redb open block_configs table failed: {e}");
                return;
            }
        };

        // Collect all positions that have any config.
        let mut positions = std::collections::HashSet::new();
        for (pos, _) in grid.iter_channel_overrides() { positions.insert(pos); }
        for (pos, _) in grid.iter_boost_channels() { positions.insert(pos); }
        for (pos, _) in grid.iter_power_configs() { positions.insert(pos); }

        for pos in &positions {
            let record = BlockConfigRecord {
                channel_override: grid.channel_override(*pos).unwrap_or("").to_string(),
                boost_channel: grid.boost_channel(*pos).unwrap_or("").to_string(),
                power_config: grid.power_config(*pos).cloned(),
                subscribe_bindings: Vec::new(),
                publish_bindings: Vec::new(),
                seat_config: block::SavedSeatConfig::default(),
            };
            let key_bytes = chunk_key_bytes(*pos);
            let data = serialize_block_config(&record);
            let _ = table.insert(key_bytes.as_slice(), data.as_slice());
        }
    }
    if let Err(e) = txn.commit() {
        warn!("redb commit failed for block configs: {e}");
    }
}

/// Save a single block's config to redb (called after individual config updates).
fn save_single_block_config(db: &redb::Database, pos: glam::IVec3, record: &BlockConfigRecord) {
    let txn = match db.begin_write() {
        Ok(t) => t,
        Err(e) => {
            warn!("redb write transaction failed for single block config: {e}");
            return;
        }
    };
    {
        let mut table = match txn.open_table(BLOCK_CONFIGS_TABLE) {
            Ok(t) => t,
            Err(e) => {
                warn!("redb open block_configs table failed: {e}");
                return;
            }
        };
        let key_bytes = chunk_key_bytes(pos);
        let data = serialize_block_config(record);
        let _ = table.insert(key_bytes.as_slice(), data.as_slice());
    }
    if let Err(e) = txn.commit() {
        warn!("redb commit failed for single block config: {e}");
    }
}

/// Load a ShipGrid from the redb database. Returns None if the database is empty.
fn load_grid_from_db(db: &redb::Database) -> Option<ShipGrid> {
    let txn = db.begin_read().ok()?;
    let table = txn.open_table(SHIP_CHUNKS_TABLE).ok()?;

    let mut grid = ShipGrid::new();
    let mut count = 0usize;

    let mut iter = table.range::<&[u8]>(..).ok()?;
    while let Some(entry) = iter.next() {
        let (key, value) = match entry {
            Ok(kv) => kv,
            Err(_) => continue,
        };
        let key_bytes = key.value();
        if key_bytes.len() < 12 {
            continue;
        }
        let chunk_key = chunk_key_from_bytes(key_bytes);
        match block::deserialize_chunk(value.value()) {
            Ok(chunk) => {
                grid.insert_chunk(chunk_key, chunk);
                count += 1;
            }
            Err(e) => {
                warn!("failed to load chunk {chunk_key}: {e}");
            }
        }
    }

    if count > 0 {
        info!(chunks = count, "loaded ship grid from persistence");
        Some(grid)
    } else {
        None
    }
}

/// Save dirty chunks to the database.
fn save_dirty_chunks(
    db: &redb::Database,
    grid: &ShipGrid,
    pending: &mut std::collections::HashSet<glam::IVec3>,
) {
    if pending.is_empty() {
        return;
    }

    let txn = match db.begin_write() {
        Ok(t) => t,
        Err(e) => {
            warn!("redb write transaction failed: {e}");
            return;
        }
    };

    {
        let mut table = match txn.open_table(SHIP_CHUNKS_TABLE) {
            Ok(t) => t,
            Err(e) => {
                warn!("redb open table failed: {e}");
                return;
            }
        };

        for &chunk_key in pending.iter() {
            let key_bytes = chunk_key_bytes(chunk_key);
            if let Some(chunk) = grid.get_chunk(chunk_key) {
                let data = block::serialize_chunk(chunk);
                let _ = table.insert(key_bytes.as_slice(), data.as_slice());
            }
        }
    }

    if let Err(e) = txn.commit() {
        warn!("redb commit failed: {e}");
        return;
    }

    let count = pending.len();
    pending.clear();
    info!(chunks = count, "persisted dirty chunks");
}

/// System: periodically save dirty chunks to redb.
fn persist_chunks(
    grid: Res<ShipGridResource>,
    mut persistence: ResMut<ShipPersistence>,
    tick: Res<ecs::TickCounter>,
) {
    if tick.0 < persistence.last_save_tick + SAVE_INTERVAL_TICKS {
        return;
    }
    persistence.last_save_tick = tick.0;

    // Destructure to satisfy the borrow checker — db is borrowed immutably,
    // pending_saves is borrowed mutably, from the same struct.
    let ShipPersistence { ref db, ref mut pending_saves, .. } = *persistence;
    save_dirty_chunks(db, &grid.0, pending_saves);
}

// ---------------------------------------------------------------------------
// Block edit systems
// ---------------------------------------------------------------------------

// ConfigUpdateMsg defined in Messages section above (with session field).

/// Bridge system: drain BlockConfigUpdate messages from the UDP channel.
fn drain_config_updates(
    mut bridge: ResMut<NetworkBridge>,
    mut events: MessageWriter<ConfigUpdateMsg>,
) {
    for _ in 0..16 {
        let (session, update) = match bridge.config_update_rx.try_recv() {
            Ok(u) => u,
            Err(_) => break,
        };
        events.write(ConfigUpdateMsg { session, update });
    }
}

/// Apply config updates received from clients to entity signal components.
/// Resolves string channel names → ChannelId at this boundary.
fn apply_config_updates(
    mut events: MessageReader<ConfigUpdateMsg>,
    block_index: Res<FunctionalBlockIndex>,
    mut commands: Commands,
    mut channels: ResMut<SignalChannelTable>,
    mut reactor_query: Query<(&mut ReactorState, Option<&mut ReactorConsumerCache>)>,
    block_ref_query: Query<&FunctionalBlockRef>,
    registry: Res<BlockRegistryResource>,
    grid: Res<ShipGridResource>,
    persistence: Option<Res<ShipPersistence>>,
    mut mech_state_query: Query<&mut MechanicalState>,
) {
    use voxeldust_core::signal::{ChannelMergeStrategy, SignalScope};
    use voxeldust_core::signal::config::*;
    let default_scope = SignalScope::Local;
    let default_merge = ChannelMergeStrategy::LastWrite;

    for event in events.read() {
        let update = &event.update;
        let entity = match block_index.0.get(&update.block_pos) {
            Some(&e) => e,
            None => {
                warn!(block = ?update.block_pos, "config update for nonexistent block");
                continue;
            }
        };

        // Resolve config types (string) → runtime types (ChannelId).
        if !update.publish_bindings.is_empty() {
            commands.entity(entity).insert(SignalPublisher {
                bindings: update.publish_bindings.iter().map(|b| PublishBinding {
                    channel_id: channels.resolve_or_create(&b.channel_name, default_scope, default_merge, 0),
                    property: b.property,
                }).collect(),
            });
        }
        if !update.subscribe_bindings.is_empty() {
            let bindings: Vec<SubscribeBinding> = update.subscribe_bindings.iter().map(|b| SubscribeBinding {
                channel_id: channels.resolve_or_create(&b.channel_name, default_scope, default_merge, 0),
                property: b.property,
            }).collect();
            commands.entity(entity).insert(SignalSubscriber { bindings });
        }
        if !update.converter_rules.is_empty() {
            commands.entity(entity).insert(SignalConverterConfig {
                rules: update.converter_rules.iter().map(|r| signal::SignalRule {
                    input_channel_id: channels.resolve_or_create(&r.input_channel, default_scope, default_merge, 0),
                    condition: r.condition.clone(),
                    output_channel_id: channels.resolve_or_create(&r.output_channel, default_scope, default_merge, 0),
                    expression: r.expression.clone(),
                }).collect(),
            });
        }
        if !update.seat_mappings.is_empty() || !update.seated_channel_name.is_empty() {
            let bindings = update.seat_mappings.iter().map(|s| SeatInputBinding {
                label: s.label.clone(),
                source: s.source,
                key_name: s.key_name.clone(),
                key_mode: s.key_mode,
                axis_direction: s.axis_direction,
                channel_id: channels.resolve_or_create(&s.channel_name, default_scope, default_merge, 0),
                property: s.property,
            }).collect();
            let seated_channel_id = if update.seated_channel_name.is_empty() {
                None
            } else {
                Some(channels.resolve_or_create(&update.seated_channel_name, default_scope, default_merge, 0))
            };
            commands.entity(entity).insert(SeatChannelMapping { bindings, seated_channel_id });
        }

        // Apply power source config updates (reactor circuit allocation + access).
        if let Some(ref ps) = update.power_source {
            if let Ok((mut rs, cache_opt)) = reactor_query.get_mut(entity) {
                rs.circuits = ps.circuits.iter().map(|c| PowerCircuit {
                    name: c.name.clone(),
                    fraction: c.fraction.clamp(0.0, 1.0),
                    supply_w: 0.0,
                    demand_w: 0.0,
                    power_ratio: 1.0,
                }).collect();
                rs.access = match &ps.access {
                    PowerAccessConfig::OwnerOnly => PowerAccess::OwnerOnly,
                    PowerAccessConfig::AllowList(names) => PowerAccess::AllowList(
                        names.iter().filter_map(|n| n.parse::<u64>().ok()).collect(),
                    ),
                    PowerAccessConfig::Open => PowerAccess::Open,
                };
                if let Some(mut cache) = cache_opt {
                    cache.dirty = true;
                }
            }
        }

        // Apply power consumer config updates (reactor selection + circuit).
        if let Some(ref pc) = update.power_consumer {
            if let Some(reactor_pos) = pc.reactor_pos {
                let consumption_w = block_ref_query.get(entity).ok()
                    .and_then(|fb| registry.0.power_props(fb.block_id))
                    .map(|p| p.consumption_w)
                    .unwrap_or(0.0);
                commands.entity(entity).insert(PoweredBy {
                    reactor_pos,
                    circuit: pc.circuit.clone(),
                    consumption_w,
                    placed_by: 0,
                });
                // Mark the target reactor's cache dirty so it picks up the new consumer.
                if let Some(&reactor_entity) = block_index.0.get(&reactor_pos) {
                    if let Ok((_, cache_opt)) = reactor_query.get_mut(reactor_entity) {
                        if let Some(mut cache) = cache_opt {
                            cache.dirty = true;
                        }
                    }
                }
            } else {
                commands.entity(entity).remove::<PoweredBy>();
                // Any reactor that had this consumer will pick up the removal
                // when the cache is rebuilt (via dirty flag or next rebuild cycle).
            }
        }

        // Persist ALL config changes to redb (signal bindings, power, everything).
        if let Some(ref persist) = persistence {
            let power_cfg = if let Some(ref ps) = update.power_source {
                Some(block::PowerConfig::Source {
                    circuits: ps.circuits.iter().map(|c| (c.name.clone(), c.fraction)).collect(),
                    broadcast_range: reactor_query.get(entity).ok()
                        .map(|(rs, _)| rs.broadcast_range)
                        .unwrap_or(50.0),
                })
            } else if let Some(ref pc) = update.power_consumer {
                pc.reactor_pos.map(|rp| block::PowerConfig::Consumer {
                    reactor_pos: rp,
                    circuit: pc.circuit.clone(),
                })
            } else {
                // Preserve existing power config from grid if no power update.
                grid.0.power_config(update.block_pos).cloned()
            };
            let record = BlockConfigRecord {
                channel_override: grid.0.channel_override(update.block_pos)
                    .unwrap_or("").to_string(),
                boost_channel: grid.0.boost_channel(update.block_pos)
                    .unwrap_or("").to_string(),
                power_config: power_cfg,
                subscribe_bindings: update.subscribe_bindings.iter()
                    .map(|b| (b.channel_name.clone(), b.property as u8))
                    .collect(),
                publish_bindings: update.publish_bindings.iter()
                    .map(|b| (b.channel_name.clone(), b.property as u8))
                    .collect(),
                seat_config: block::SavedSeatConfig {
                    bindings: update.seat_mappings.iter()
                        .map(|s| (s.label.clone(), s.source as u8, s.key_name.clone(), s.key_mode as u8, s.axis_direction as u8, s.channel_name.clone(), s.property as u8))
                        .collect(),
                    seated_channel_name: update.seated_channel_name.clone(),
                },
            };
            save_single_block_config(&persist.db, update.block_pos, &record);
        }

        // Apply mechanical config (speed override).
        if let Some(ref mc) = update.mechanical {
            if let Ok(mut ms) = mech_state_query.get_mut(entity) {
                if let Some(speed) = mc.speed_override {
                    // Cap by registry max_speed for the block type.
                    let max = block_ref_query.get(entity).ok()
                        .and_then(|fb| registry.0.mechanical_props(fb.block_id))
                        .map(|mp| mp.max_speed as f32)
                        .unwrap_or(360.0);
                    ms.max_speed = speed.clamp(0.1, max);
                }
            }
        }

        info!(block = ?update.block_pos, "applied signal config update from client");
    }
}

/// Bridge system: drain BlockEditRequest messages from the UDP channel.
fn drain_block_edits(
    mut bridge: ResMut<NetworkBridge>,
    mut events: MessageWriter<BlockEditMsg>,
) {
    for _ in 0..64 {
        let (session, edit) = match bridge.block_edit_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        events.write(BlockEditMsg { session, edit });
    }
}

// SubBlockEditMsg defined in Messages section above (with session field).

/// Bridge system: drain SubBlockEditRequest messages from the UDP channel.
fn drain_sub_block_edits(
    mut bridge: ResMut<NetworkBridge>,
    mut events: MessageWriter<SubBlockEditMsg>,
) {
    for _ in 0..64 {
        let (session, edit) = match bridge.sub_block_edit_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        events.write(SubBlockEditMsg { session, edit });
    }
}

/// Process sub-block edit requests: validate, apply to grid, broadcast delta.
fn process_sub_block_edits(
    mut events: MessageReader<SubBlockEditMsg>,
    mut grid: ResMut<ShipGridResource>,
    registry: Res<BlockRegistryResource>,
    bridge: Res<NetworkBridge>,
    mut persistence: Option<ResMut<ShipPersistence>>,
) {
    use voxeldust_core::block::sub_block::{SubBlockElement, SubBlockType};
    use voxeldust_core::client_message::{action, SubBlockModData, ChunkDeltaData, ServerMsg};

    for event in events.read() {
        let edit = &event.edit;
        let pos = edit.block_pos;

        // Validate: host block must be solid (can't place sub-blocks on air).
        let host_block = grid.0.get_block(pos.x, pos.y, pos.z);
        if host_block.is_air() && edit.action == action::PLACE_SUB {
            tracing::warn!(
                block = ?pos, face = edit.face,
                "sub-block edit rejected: host block is air"
            );
            continue;
        }

        // Validate: face index in range.
        if edit.face >= block::sub_block::FACE_COUNT {
            tracing::warn!(face = edit.face, "sub-block edit rejected: invalid face");
            continue;
        }

        // Validate: element type is known.
        let element_type = match SubBlockType::from_u8(edit.element_type) {
            Some(t) => t,
            None => {
                tracing::warn!(
                    element_type = edit.element_type,
                    "sub-block edit rejected: unknown element type"
                );
                continue;
            }
        };

        let chunk_key = block::ShipGrid::world_to_chunk(pos.x, pos.y, pos.z).0;
        let (_, lx, ly, lz) = block::ShipGrid::world_to_chunk(pos.x, pos.y, pos.z);

        match edit.action {
            action::PLACE_SUB => {
                let element = SubBlockElement {
                    face: edit.face,
                    element_type,
                    rotation: edit.rotation & 0x03,
                    flags: 0,
                };
                if !grid.0.add_sub_block(pos.x, pos.y, pos.z, element) {
                    tracing::warn!(
                        block = ?pos, face = edit.face,
                        "sub-block edit rejected: face already occupied"
                    );
                    continue;
                }
                tracing::info!(
                    block = ?pos, face = edit.face,
                    element = element_type.label(),
                    "placed sub-block"
                );
            }
            action::REMOVE_SUB => {
                if grid.0.remove_sub_block(pos.x, pos.y, pos.z, edit.face).is_none() {
                    continue; // Nothing to remove.
                }
                tracing::info!(
                    block = ?pos, face = edit.face,
                    "removed sub-block"
                );
            }
            _ => continue,
        }

        // Mark chunk dirty for persistence.
        if let Some(ref mut persist) = persistence {
            persist.pending_saves.insert(chunk_key);
        }

        // Broadcast sub-block delta to ALL connected clients.
        {
            let seq = grid.0.get_chunk_mut(chunk_key)
                .map(|c| c.next_edit_seq())
                .unwrap_or(1);
            let delta = ServerMsg::ChunkDelta(ChunkDeltaData {
                chunk_x: chunk_key.x,
                chunk_y: chunk_key.y,
                chunk_z: chunk_key.z,
                seq,
                mods: Vec::new(), // No block mods, only sub-block mods.
                sub_block_mods: vec![SubBlockModData {
                    bx: lx, by: ly, bz: lz,
                    face: edit.face,
                    element_type: edit.element_type,
                    rotation: edit.rotation,
                    action: edit.action,
                }],
            });
            let cr = bridge.client_registry.clone();
            tokio::spawn(async move {
                if let Ok(reg) = cr.try_read() {
                    for addr in reg.udp_addrs() {
                        if let Some(session) = reg.session_for_udp(addr) {
                            let _ = reg.send_tcp(session, &delta).await;
                        }
                    }
                }
            });
        }
    }
}

/// Damage amount per player hit. Higher-tier tools would increase this.
const PLAYER_BREAK_DAMAGE: u16 = 20;

/// Maximum raycast distance for block editing (in blocks).
const BLOCK_EDIT_RANGE: f32 = 8.0;

/// Producer system: processes player BlockEditRequests.
/// Performs server-authoritative raycast, applies progressive damage for breaking,
/// and pushes resulting edits into the generic BlockEditQueue.
fn produce_player_edits(
    mut events: MessageReader<BlockEditMsg>,
    mut queue: ResMut<BlockEditQueue>,
    mut grid: ResMut<ShipGridResource>,
    registry: Res<BlockRegistryResource>,
    player_index: Res<PlayerEntityIndex>,
    mut players: Query<(
        &PlayerPosition, &PlayerBody, &mut SeatedState, &PlayerPhysics,
    ), (With<Player>, Without<HandoffPending>)>,
    block_index: Res<FunctionalBlockIndex>,
    mut rapier: ResMut<RapierContext>,
    bridge: Res<NetworkBridge>,
    mut signal_ctx: SignalQueryCtx,
    world_isos: Res<MechanicalWorldIsometries>,
    sub_grids: Res<SubGridRegistry>,
) {

    for event in events.read() {
        let edit = &event.edit;

        // Resolve the editing player entity.
        let entity = match player_index.0.get(&event.session) {
            Some(&e) => e,
            None => continue,
        };
        let Ok((player_pos, _body, mut seated_state, _physics)) = players.get_mut(entity) else {
            continue;
        };

        // Action EXIT_SEAT: no raycast or position validation needed.
        if edit.action == voxeldust_core::client_message::action::EXIT_SEAT && seated_state.seated {
            seated_state.seated = false;
            seated_state.seat_entity = None;
            info!(session = event.session.0, "exited seat via F key");
            continue;
        }

        let eye = glam::Vec3::new(edit.eye.x as f32, edit.eye.y as f32, edit.eye.z as f32);
        let look = glam::Vec3::new(edit.look.x as f32, edit.look.y as f32, edit.look.z as f32);

        // Validate: player eye position should be close to their actual position.
        let eye_to_player = (eye - player_pos.0).length();
        if eye_to_player > 3.0 {
            warn!("block edit rejected: eye position too far from player ({:.1}m)", eye_to_player);
            continue;
        }

        // Server-authoritative raycast against the ShipGrid.
        // First: raycast against root-grid blocks (not in any sub-grid).
        let root_hit = raycast::raycast(eye, look, BLOCK_EDIT_RANGE, |x, y, z| {
            let bid = grid.0.get_block(x, y, z);
            if !registry.0.is_solid(bid) { return false; }
            grid.0.sub_grid_id(glam::IVec3::new(x, y, z)) == 0 // root only
        });

        // Then: raycast against each sub-grid in its local (unrotated) frame.
        // Transform the ray INTO the sub-grid's rest space using the inverse isometry.
        let mut best_hit: Option<raycast::BlockHit> = root_hit;
        for (sg_id, iso) in &world_isos.0 {
            let sg_data = match sub_grids.grids.get(sg_id) {
                Some(d) => d,
                None => continue,
            };
            if sg_data.members.is_empty() { continue; }

            // Compute the sub-grid's transform that maps root-space → world-space.
            // We need the INVERSE to map the ray from world-space → root-space.
            // The transform is: world = T(pos) * R * T(-anchor) * root_point
            // Inverse: root_point = T(anchor) * R_inv * T(-pos) * world_point
            let axis_offset = block::sub_block::face_to_offset(sg_data.mount_face);
            let anchor = glam::Vec3::new(
                sg_data.mount_pos.x as f32 + 0.5 + axis_offset.x as f32 * 0.5,
                sg_data.mount_pos.y as f32 + 0.5 + axis_offset.y as f32 * 0.5,
                sg_data.mount_pos.z as f32 + 0.5 + axis_offset.z as f32 * 0.5,
            );
            let iso_inv = iso.inverse();
            let local_eye_pt = iso_inv * rapier3d::math::Point::new(eye.x, eye.y, eye.z);
            let local_look_v = iso_inv.rotation * rapier3d::math::Vector::new(look.x, look.y, look.z);
            // Shift from body-local to root-space (add anchor back).
            let local_eye = glam::Vec3::new(
                local_eye_pt.x + anchor.x,
                local_eye_pt.y + anchor.y,
                local_eye_pt.z + anchor.z,
            );
            let local_look = glam::Vec3::new(local_look_v.x, local_look_v.y, local_look_v.z);

            let sg_id_u32 = sg_id.0;
            let sg_hit = raycast::raycast(local_eye, local_look, BLOCK_EDIT_RANGE, |x, y, z| {
                let pos = glam::IVec3::new(x, y, z);
                if grid.0.sub_grid_id(pos) != sg_id_u32 { return false; }
                registry.0.is_solid(grid.0.get_block(x, y, z))
            });

            if let Some(sh) = sg_hit {
                let closer = best_hit.as_ref().map_or(true, |bh| sh.distance < bh.distance);
                if closer {
                    best_hit = Some(sh);
                }
            }
        }

        let hit = match best_hit {
            Some(h) => h,
            None => continue, // ray didn't hit anything
        };

        use voxeldust_core::client_message::action;
        match edit.action {
            action::BREAK => {
                // Break: apply progressive damage.
                let result = grid.0.damage_block(
                    hit.world_pos.x,
                    hit.world_pos.y,
                    hit.world_pos.z,
                    PLAYER_BREAK_DAMAGE,
                    &registry.0,
                );
                match result {
                    DamageResult::Broken => {
                        // Block destroyed — push to edit queue for pipeline processing.
                        queue.push(BlockEdit {
                            pos: hit.world_pos,
                            new_block: BlockId::AIR,
                            source: EditSource::Player(event.session),
                        });
                        // Clear metadata (damage state) for the broken block.
                        grid.0.remove_meta(
                            hit.world_pos.x,
                            hit.world_pos.y,
                            hit.world_pos.z,
                        );
                    }
                    DamageResult::Damaged { stage } => {
                        // Block took damage but didn't break.
                        // TODO: send damage stage to client for crack overlay rendering.
                        info!(
                            block = ?hit.world_pos,
                            stage,
                            "block damaged"
                        );
                    }
                    DamageResult::NoEffect => {}
                }
            }
            action::PLACE => {
                // Place: put a new block on the face adjacent to the hit.
                let target = hit.world_pos + hit.face_normal;

                // Validate: target must be air.
                if !grid.0.get_block(target.x, target.y, target.z).is_air() {
                    continue;
                }

                // Validate: don't place inside the player's bounding box.
                // Player capsule is ~0.6 radius, ~1.8 tall. Check if the target
                // block overlaps the player position.
                let block_center = glam::Vec3::new(
                    target.x as f32 + 0.5,
                    target.y as f32 + 0.5,
                    target.z as f32 + 0.5,
                );
                let dx = (block_center.x - player_pos.0.x).abs();
                let dz = (block_center.z - player_pos.0.z).abs();
                let dy_low = block_center.y - player_pos.0.y; // relative Y
                // Player occupies roughly x±0.5, z±0.5, y-0.1 to y+1.7
                if dx < 1.0 && dz < 1.0 && dy_low > -0.6 && dy_low < 2.2 {
                    continue; // would place inside the player
                }

                // Validate: block type is a real block, not air.
                let block_type = BlockId::from_u16(edit.block_type);
                if block_type.is_air() {
                    continue;
                }

                // Set block orientation from placement face normal.
                // The block faces outward from the surface it's placed on.
                let orientation = block::BlockOrientation::from_face_normal(hit.face_normal);
                grid.0.set_orientation(target.x, target.y, target.z, orientation);

                queue.push(BlockEdit {
                    pos: target,
                    new_block: block_type,
                    source: EditSource::Player(event.session),
                });
            }
            action @ (action::INTERACT..=7 | 9) => {
                // Interaction: raycast to find targeted functional block,
                // look up its kind, check interaction schema, dispatch.
                if let Some(&entity) = block_index.0.get(&hit.world_pos) {
                    let block_id = grid.0.get_block(
                        hit.world_pos.x, hit.world_pos.y, hit.world_pos.z,
                    );
                    if let Some(kind) = registry.0.functional_kind(block_id) {
                        let schema = registry.0.interaction_schema(kind);
                        let matching_action = schema.actions.iter()
                            .find(|a| a.action_key == action);

                        if let Some(interaction_def) = matching_action {
                            match kind {
                                FunctionalBlockKind::Seat => {
                                    // Toggle seated state (per-player).
                                    seated_state.seated = !seated_state.seated;
                                    if seated_state.seated {
                                        seated_state.seat_entity = Some(entity);
                                        if let Some(body) = rapier.rigid_body_set.get_mut(_body.0) {
                                            body.set_linvel(vector![0.0, 0.0, 0.0], true);
                                        }
                                        // Send SeatBindingsNotify to client so it knows what inputs to evaluate.
                                        if let Ok(mapping) = signal_ctx.seat_query.get(entity) {
                                            let bindings = mapping.bindings.iter().map(|b| {
                                                voxeldust_core::signal::config::SeatInputBindingConfig {
                                                    label: b.label.clone(),
                                                    source: b.source,
                                                    key_name: b.key_name.clone(),
                                                    key_mode: b.key_mode,
                                                    axis_direction: b.axis_direction,
                                                    channel_name: signal_ctx.channels.name_for_id(b.channel_id).unwrap_or("").to_string(),
                                                    property: b.property,
                                                }
                                            }).collect();
                                            let seated_ch = mapping.seated_channel_id
                                                .and_then(|id| signal_ctx.channels.name_for_id(id))
                                                .unwrap_or("").to_string();
                                            let notify = ServerMsg::SeatBindingsNotify(SeatBindingsNotifyData {
                                                bindings,
                                                seated_channel_name: seated_ch,
                                            });
                                            let cr = bridge.client_registry.clone();
                                            let s = event.session;
                                            tokio::spawn(async move {
                                                if let Ok(reg) = cr.try_read() {
                                                    let _ = reg.send_tcp(s, &notify).await;
                                                }
                                            });
                                        }
                                        info!(session = event.session.0, "entered seat at {:?}", hit.world_pos);
                                    } else {
                                        seated_state.seat_entity = None;
                                        info!(session = event.session.0, "exited seat");
                                    }
                                }
                                FunctionalBlockKind::Reactor => {
                                    if let Ok((mut rs, _)) = signal_ctx.reactor_query.get_mut(entity) {
                                        rs.active = !rs.active;
                                        info!(active = rs.active, block = ?hit.world_pos, "reactor toggled");
                                    }
                                }
                                _ => {
                                    info!(
                                        kind = ?kind,
                                        interaction = interaction_def.label,
                                        block = ?hit.world_pos,
                                        "interaction not yet implemented"
                                    );
                                }
                            }
                        }
                    }
                }
            }
            action::OPEN_CONFIG => {
                // F key: universal config UI — open config for any functional block.
                // The entity may be a full functional block (reactor, thruster) OR a
                // mechanical sub-block mount (rotor, piston) registered at the host block position.
                if let Some(&entity) = block_index.0.get(&hit.world_pos) {
                    let block_id = grid.0.get_block(
                        hit.world_pos.x, hit.world_pos.y, hit.world_pos.z,
                    );
                    // Try block-level functional kind first, then fall back to the
                    // entity's FunctionalBlockRef (for sub-block mounts on non-functional host blocks).
                    let kind = registry.0.functional_kind(block_id)
                        .or_else(|| {
                            signal_ctx.block_ref_query.get(entity).ok().map(|r| r.kind)
                        });
                    if let Some(kind) = kind {
                        let config = build_config_snapshot(
                            entity, hit.world_pos, block_id, kind,
                            &signal_ctx.pub_query, &signal_ctx.sub_query,
                            &signal_ctx.converter_query, &signal_ctx.seat_query,
                            &signal_ctx.channels,
                            &signal_ctx.reactor_query,
                            &signal_ctx.powered_by_query,
                            &registry.0,
                            &signal_ctx.fc_query, &signal_ctx.hm_query,
                            &signal_ctx.ap_query, &signal_ctx.wc_query,
                            &signal_ctx.ec_query, &signal_ctx.mech_query,
                        );
                        {
                            let session = event.session;
                            let msg = ServerMsg::BlockConfigState(config);
                            let cr = bridge.client_registry.clone();
                            tokio::spawn(async move {
                                if let Ok(reg) = cr.try_read() {
                                    let _ = reg.send_tcp(session, &msg).await;
                                }
                            });
                        }
                        info!(kind = ?kind, block = ?hit.world_pos, "sent block config to client");
                    }
                }
            }
            _ => {} // unknown action
        }
    }
}

/// Build a config snapshot from entity components for the client UI.
/// Converts runtime types (ChannelId) → config types (String) for serialization.
fn build_config_snapshot(
    entity: Entity,
    pos: glam::IVec3,
    block_id: BlockId,
    kind: FunctionalBlockKind,
    pub_query: &Query<&SignalPublisher>,
    sub_query: &Query<&SignalSubscriber>,
    converter_query: &Query<&SignalConverterConfig>,
    seat_query: &Query<&SeatChannelMapping>,
    channels: &SignalChannelTable,
    reactor_query: &Query<(&mut ReactorState, Option<&FunctionalBlockRef>)>,
    powered_by_query: &Query<&PoweredBy>,
    _registry: &BlockRegistry,
    fc_query: &Query<&FlightComputerState>,
    hm_query: &Query<&HoverModuleState>,
    ap_query: &Query<&AutopilotBlockState>,
    wc_query: &Query<&WarpComputerState>,
    ec_query: &Query<&EngineControllerState>,
    mech_query: &Query<&MechanicalState>,
) -> voxeldust_core::signal::config::BlockSignalConfig {
    use voxeldust_core::signal::config::*;

    let id_to_name = |id: signal::ChannelId| -> String {
        channels.name_of(id).unwrap_or("?").to_string()
    };

    let publish_bindings = pub_query.get(entity)
        .map(|p| p.bindings.iter().map(|b| PublishBindingConfig {
            channel_name: id_to_name(b.channel_id),
            property: b.property,
        }).collect())
        .unwrap_or_default();

    let subscribe_bindings = sub_query.get(entity)
        .map(|s| s.bindings.iter().map(|b| SubscribeBindingConfig {
            channel_name: id_to_name(b.channel_id),
            property: b.property,
        }).collect())
        .unwrap_or_default();

    let converter_rules = converter_query.get(entity)
        .map(|c| c.rules.iter().map(|r| SignalRuleConfig {
            input_channel: id_to_name(r.input_channel_id),
            condition: r.condition.clone(),
            output_channel: id_to_name(r.output_channel_id),
            expression: r.expression.clone(),
        }).collect())
        .unwrap_or_default();

    let (seat_mappings, seated_channel_name) = seat_query.get(entity)
        .map(|s| {
            let bindings = s.bindings.iter().map(|b| SeatInputBindingConfig {
                label: b.label.clone(),
                source: b.source,
                key_name: b.key_name.clone(),
                key_mode: b.key_mode,
                axis_direction: b.axis_direction,
                channel_name: id_to_name(b.channel_id),
                property: b.property,
            }).collect();
            let seated_ch = s.seated_channel_id
                .map(|id| id_to_name(id))
                .unwrap_or_default();
            (bindings, seated_ch)
        })
        .unwrap_or_default();

    let mut available_channels: Vec<String> = channels.channel_names()
        .map(|s| s.to_string())
        .collect();
    available_channels.sort();

    // Power source config (reactors only).
    let power_source = if kind == FunctionalBlockKind::Reactor {
        reactor_query.get(entity).ok().map(|(rs, _)| {
            PowerSourceConfig {
                circuits: rs.circuits.iter().map(|c| PowerCircuitConfig {
                    name: c.name.clone(),
                    fraction: c.fraction,
                }).collect(),
                access: match &rs.access {
                    PowerAccess::OwnerOnly => PowerAccessConfig::OwnerOnly,
                    PowerAccess::AllowList(list) => PowerAccessConfig::AllowList(
                        list.iter().map(|id| format!("{id}")).collect(),
                    ),
                    PowerAccess::Open => PowerAccessConfig::Open,
                },
            }
        })
    } else {
        None
    };

    // Power consumer config.
    let power_consumer = powered_by_query.get(entity).ok().map(|pb| {
        PowerConsumerConfig {
            reactor_pos: Some(pb.reactor_pos),
            circuit: pb.circuit.clone(),
        }
    });

    // Nearby reactors in range (for consumer config dropdown).
    let mut nearby_reactors = Vec::new();
    for (rs, fb_ref) in reactor_query.iter() {
        let Some(reactor_ref) = fb_ref else { continue };
        let dist = (reactor_ref.world_pos - pos).as_vec3().length();
        if dist > rs.broadcast_range { continue; }
        let label = format!("Reactor ({})", reactor_ref.block_id.as_u16());
        nearby_reactors.push(NearbyReactorInfo {
            pos: reactor_ref.world_pos,
            label,
            distance: dist,
            circuits: rs.circuits.iter().map(|c| c.name.clone()).collect(),
        });
    }
    nearby_reactors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

    BlockSignalConfig {
        block_pos: pos,
        block_type: block_id.as_u16(),
        kind: kind as u8,
        publish_bindings,
        subscribe_bindings,
        converter_rules,
        seat_mappings,
        seated_channel_name,
        available_channels,
        power_source,
        power_consumer,
        nearby_reactors,
        flight_computer: fc_query.get(entity).ok().map(|fc| FlightComputerConfig {
            yaw_cw_channel: fc.yaw_cw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            yaw_ccw_channel: fc.yaw_ccw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            pitch_up_channel: fc.pitch_up_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            pitch_down_channel: fc.pitch_down_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            roll_cw_channel: fc.roll_cw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            roll_ccw_channel: fc.roll_ccw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            toggle_channel: fc.toggle_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            damping_gain: fc.damping_gain,
            dead_zone: fc.dead_zone,
            max_correction: fc.max_correction,
        }),
        hover_module: hm_query.get(entity).ok().map(|hm| HoverModuleConfig {
            thrust_forward_channel: hm.thrust_forward_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            thrust_reverse_channel: hm.thrust_reverse_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            thrust_right_channel: hm.thrust_right_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            thrust_left_channel: hm.thrust_left_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            thrust_up_channel: hm.thrust_up_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            thrust_down_channel: hm.thrust_down_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            yaw_cw_channel: hm.yaw_cw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            yaw_ccw_channel: hm.yaw_ccw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            pitch_up_channel: hm.pitch_up_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            pitch_down_channel: hm.pitch_down_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            roll_cw_channel: hm.roll_cw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            roll_ccw_channel: hm.roll_ccw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            activate_channel: hm.activate_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            cutoff_channel: hm.cutoff_channel.map(|id| id_to_name(id)).unwrap_or_default(),
        }),
        autopilot: ap_query.get(entity).ok().map(|ap| AutopilotBlockConfig {
            yaw_cw_channel: ap.yaw_cw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            yaw_ccw_channel: ap.yaw_ccw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            pitch_up_channel: ap.pitch_up_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            pitch_down_channel: ap.pitch_down_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            roll_cw_channel: ap.roll_cw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            roll_ccw_channel: ap.roll_ccw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            engage_channel: ap.engage_channel.map(|id| id_to_name(id)).unwrap_or_default(),
        }),
        warp_computer: wc_query.get(entity).ok().map(|wc| WarpComputerConfig {
            cycle_channel: wc.cycle_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            accept_channel: wc.accept_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            cancel_channel: wc.cancel_channel.map(|id| id_to_name(id)).unwrap_or_default(),
        }),
        engine_controller: ec_query.get(entity).ok().map(|ec| EngineControllerConfig {
            thrust_forward_channel: ec.thrust_forward_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            thrust_reverse_channel: ec.thrust_reverse_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            thrust_right_channel: ec.thrust_right_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            thrust_left_channel: ec.thrust_left_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            thrust_up_channel: ec.thrust_up_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            thrust_down_channel: ec.thrust_down_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            yaw_cw_channel: ec.yaw_cw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            yaw_ccw_channel: ec.yaw_ccw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            pitch_up_channel: ec.pitch_up_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            pitch_down_channel: ec.pitch_down_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            roll_cw_channel: ec.roll_cw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            roll_ccw_channel: ec.roll_ccw_channel.map(|id| id_to_name(id)).unwrap_or_default(),
            toggle_channel: ec.toggle_channel.map(|id| id_to_name(id)).unwrap_or_default(),
        }),
        mechanical: mech_query.get(entity).ok().map(|ms| {
            use voxeldust_core::signal::config::MechanicalConfig;
            MechanicalConfig {
                speed_override: Some(ms.max_speed),
            }
        }),
    }
}

/// Rebuild a single chunk's Rapier collider from the current ShipGrid state.
fn rebuild_chunk_collider(
    grid: &ShipGrid,
    chunk_key: glam::IVec3,
    registry: &BlockRegistry,
    rapier: &mut RapierContext,
    collider_handles: &mut ChunkColliderHandles,
    root_body: rapier3d::dynamics::RigidBodyHandle,
    sub_grids: &mut SubGridRegistry,
) {
    // Remove old root-grid collider for this chunk.
    if let Some(old_handle) = collider_handles.0.remove(&chunk_key) {
        rapier.collider_set.remove(
            old_handle,
            &mut rapier.island_manager,
            &mut rapier.rigid_body_set,
            true,
        );
    }

    // Build root-grid collider (only blocks NOT in any sub-grid).
    let root_shapes = grid.chunk_collider_shapes_filtered(chunk_key, glam::Vec3::ZERO, registry, None);
    if !root_shapes.is_empty() {
        let compound: Vec<_> = root_shapes.iter().map(|&(pos, he)| {
            (Isometry::translation(pos.x, pos.y, pos.z), SharedShape::cuboid(he.x, he.y, he.z))
        }).collect();
        let collider = ColliderBuilder::compound(compound).build();
        let handle = rapier.collider_set.insert_with_parent(
            collider, root_body, &mut rapier.rigid_body_set,
        );
        collider_handles.0.insert(chunk_key, handle);
    }

    // Rebuild colliders for each sub-grid that has blocks in this chunk.
    for (sg_id, sg_data) in &mut sub_grids.grids {
        // Remove old sub-grid collider for this chunk.
        if let Some(old_handle) = sg_data.collider_handles.remove(&chunk_key) {
            rapier.collider_set.remove(
                old_handle,
                &mut rapier.island_manager,
                &mut rapier.rigid_body_set,
                true,
            );
        }

        let sg_shapes = grid.chunk_collider_shapes_filtered(
            chunk_key, glam::Vec3::ZERO, registry, Some(sg_id.0),
        );
        if !sg_shapes.is_empty() {
            // Offset shapes relative to the sub-grid's FIXED anchor position
            // (mount_pos + face offset). NOT the body's current translation,
            // which changes for nested sub-grids as the parent rotates.
            // The anchor is the body's local origin — colliders rotate around it.
            let axis_offset = block::sub_block::face_to_offset(sg_data.mount_face);
            let body_pos = rapier3d::math::Vector::new(
                sg_data.mount_pos.x as f32 + 0.5 + axis_offset.x as f32 * 0.5,
                sg_data.mount_pos.y as f32 + 0.5 + axis_offset.y as f32 * 0.5,
                sg_data.mount_pos.z as f32 + 0.5 + axis_offset.z as f32 * 0.5,
            );
            let compound: Vec<_> = sg_shapes.iter().map(|&(pos, he)| {
                let local = Isometry::translation(
                    pos.x - body_pos.x,
                    pos.y - body_pos.y,
                    pos.z - body_pos.z,
                );
                (local, SharedShape::cuboid(he.x, he.y, he.z))
            }).collect();

            // No collision group filtering — sub-grid blocks should collide with
            // the ship hull (root body) and other sub-grids. Self-collision within
            // the compound shape is impossible (Rapier doesn't self-collide compounds).
            // Sub-grid blocks are already removed from the root collider via filtered
            // meshing, so there's no overlap at the mount interface.
            let collider = ColliderBuilder::compound(compound).build();
            let handle = rapier.collider_set.insert_with_parent(
                collider, sg_data.body_handle, &mut rapier.rigid_body_set,
            );
            sg_data.collider_handles.insert(chunk_key, handle);
        }
    }
}

/// Apply system: drains the BlockEditQueue, applies edits via the core pipeline,
/// then handles all shard-specific side effects.
fn apply_block_edits(
    mut queue: ResMut<BlockEditQueue>,
    mut grid: ResMut<ShipGridResource>,
    registry: Res<BlockRegistryResource>,
    mut rapier: ResMut<RapierContext>,
    mut collider_handles: ResMut<ChunkColliderHandles>,
    mut hull_bounds: ResMut<ShipHullBounds>,
    bridge: Res<NetworkBridge>,
    mut persistence: Option<ResMut<ShipPersistence>>,
    mut pending_entity_ops: Option<ResMut<PendingEntityOps>>,
    mut agg_dirty: ResMut<AggregationDirty>,
    mut collider_dirty: ResMut<ColliderSyncDirty>,
    root_body: Res<RootBodyHandle>,
    mut sub_grids: ResMut<SubGridRegistry>,
    mut mech_dirty: ResMut<MechanicalDirtyGrids>,
) {
    let edits = queue.drain_all();
    if edits.is_empty() {
        return;
    }

    // 1. Core: apply edits to grid (pure function, no side effects).
    let result = edit_pipeline::apply_edits_to_grid(&mut grid.0, &edits, &registry.0);

    if result.applied_count == 0 {
        return;
    }

    // Mark aggregation and collider sync as needing recomputation.
    agg_dirty.0 = true;
    collider_dirty.0 = true;

    // 2. Shard-specific: rebuild chunk colliders for all dirty chunks.
    for &chunk_key in result.dirty_chunks.keys() {
        rebuild_chunk_collider(&grid.0, chunk_key, &registry.0, &mut rapier, &mut collider_handles, root_body.0, &mut sub_grids);
    }

    // 2a. Check if any edited block is in or adjacent to an existing sub-grid.
    //     If so, mark that sub-grid for membership re-computation.
    for edit in &edits {
        let pos = edit.pos;
        // If the edited block itself belongs to a sub-grid, mark it dirty.
        let sg = grid.0.sub_grid_id(pos);
        if sg != 0 {
            mech_dirty.dirty.insert(voxeldust_core::ecs::components::SubGridId(sg));
        }
        // Also check if the edited position is the output neighbor of any mount.
        for (sg_id, sg_data) in &sub_grids.grids {
            let output_face = sg_data.mount_face;
            let output_pos = sg_data.mount_pos + block::sub_block::face_to_offset(output_face);
            // If the edit is at the output position or adjacent to any member,
            // this sub-grid's membership might have changed.
            if pos == output_pos || sg_data.members.contains(&pos) {
                mech_dirty.dirty.insert(*sg_id);
            }
            // Check 6-neighbors of the edited position for membership in this sub-grid.
            for face in 0..6u8 {
                let neighbor = pos + block::sub_block::face_to_offset(face);
                if sg_data.members.contains(&neighbor) {
                    mech_dirty.dirty.insert(*sg_id);
                    break;
                }
            }
        }
    }

    // 2b. Mark dirty chunks for persistence.
    if let Some(ref mut persist) = persistence {
        for &chunk_key in result.dirty_chunks.keys() {
            persist.pending_saves.insert(chunk_key);
        }
    }

    // 3. Shard-specific: update hull bounds if solidity changed.
    if result.solidity_changed {
        if let Some((min, max)) = grid.0.bounding_box() {
            hull_bounds.min = min;
            hull_bounds.max = max;
        }
    }

    // 4. Broadcast ChunkDeltas to ALL connected clients (one per dirty chunk).
    for (chunk_key, mods) in &result.dirty_chunks {
        let seq = grid.0.get_chunk_mut(*chunk_key)
            .map(|c| c.next_edit_seq())
            .unwrap_or(1);
        let delta = ServerMsg::ChunkDelta(ChunkDeltaData {
            chunk_x: chunk_key.x,
            chunk_y: chunk_key.y,
            chunk_z: chunk_key.z,
            seq,
            mods: mods.clone(),
            sub_block_mods: Vec::new(),
        });
        let cr = bridge.client_registry.clone();
        tokio::spawn(async move {
            if let Ok(reg) = cr.try_read() {
                for addr in reg.udp_addrs() {
                    if let Some(session) = reg.session_for_udp(addr) {
                        let _ = reg.send_tcp(session, &delta).await;
                    }
                }
            }
        });
    }

    // 6. Store entity lifecycle operations for process_entity_ops.
    if !result.entity_ops.is_empty() {
        if let Some(mut pending) = pending_entity_ops {
            pending.ops.extend(result.entity_ops);
        }
    }

    info!(
        applied = result.applied_count,
        dirty_chunks = result.dirty_chunks.len(),
        "block edits applied"
    );
}

/// Startup system: scan the ShipGrid for existing functional blocks and spawn entities.
/// Runs once on the first tick to initialize the FunctionalBlockIndex.
fn init_functional_blocks(
    mut commands: Commands,
    grid: Res<ShipGridResource>,
    registry: Res<BlockRegistryResource>,
    mut block_index: ResMut<FunctionalBlockIndex>,
    mut dirty: ResMut<AggregationDirty>,
    mut channels: ResMut<SignalChannelTable>,
) {
    let cs = block::CHUNK_SIZE as i32;
    for (chunk_key, chunk) in grid.0.iter_chunks() {
        if chunk.is_empty() {
            continue;
        }
        let chunk_origin = glam::IVec3::new(
            chunk_key.x * cs, chunk_key.y * cs, chunk_key.z * cs,
        );
        for x in 0..block::CHUNK_SIZE as u8 {
            for y in 0..block::CHUNK_SIZE as u8 {
                for z in 0..block::CHUNK_SIZE as u8 {
                    let id = chunk.get_block(x, y, z);
                    if let Some(kind) = registry.0.functional_kind(id) {
                        let wp = chunk_origin + glam::IVec3::new(x as i32, y as i32, z as i32);
                        let mut entity_cmds = commands.spawn(FunctionalBlockRef {
                            world_pos: wp,
                            block_id: id,
                            kind,
                        });
                        add_default_signal_bindings(&mut entity_cmds, kind, id, wp, &grid.0, &registry.0, &mut channels);
                        let entity = entity_cmds.id();
                        block_index.0.insert(wp, entity);
                    }
                }
            }
        }
    }
    if !block_index.0.is_empty() {
        info!(count = block_index.0.len(), "initialized functional block entities from grid");
    }

    // Trigger initial aggregation on first tick.
    dirty.0 = true;
}

/// Process entity lifecycle operations: spawn/despawn ECS entities for functional blocks.
/// Runs after apply_block_edits in the BlockEdit SystemSet.
fn process_entity_ops(
    mut commands: Commands,
    mut grid: ResMut<ShipGridResource>,
    registry: Res<BlockRegistryResource>,
    mut block_index: ResMut<FunctionalBlockIndex>,
    mut pending: ResMut<PendingEntityOps>,
    mut channels: ResMut<SignalChannelTable>,
) {
    for op in pending.ops.drain(..) {
        match op {
            EntityOp::Spawn { pos, block_id } => {
                if let Some(kind) = registry.0.functional_kind(block_id) {
                    let mut entity_cmds = commands.spawn(FunctionalBlockRef {
                        world_pos: pos,
                        block_id,
                        kind,
                    });

                    // Add default signal bindings based on block kind.
                    add_default_signal_bindings(&mut entity_cmds, kind, block_id, pos, &grid.0, &registry.0, &mut channels);

                    let entity = entity_cmds.id();

                    // Store entity index in block metadata (grid → entity link).
                    let meta = grid.0.get_or_create_meta(pos.x, pos.y, pos.z);
                    meta.entity_index = entity.index_u32();

                    // Store in position index (position → entity link).
                    block_index.0.insert(pos, entity);

                    info!(
                        kind = ?kind,
                        block = ?pos,
                        entity = entity.index_u32(),
                        "spawned functional block entity"
                    );
                }
            }
            EntityOp::Despawn { pos, entity_index: _ } => {
                // Find and despawn the entity via the position index.
                if let Some(entity) = block_index.0.remove(&pos) {
                    commands.entity(entity).despawn();
                    info!(
                        block = ?pos,
                        entity = entity.index_u32(),
                        "despawned functional block entity"
                    );
                }
                // Clear metadata at this position.
                grid.0.remove_meta(pos.x, pos.y, pos.z);
            }
        }
    }
}

/// Check for newly placed/removed mechanical sub-blocks (RotorMount, PistonMount, etc.)
/// and create/destroy sub-grids, Rapier joints, and functional entities accordingly.
fn process_mechanical_edits(
    mut commands: Commands,
    mut rapier: ResMut<RapierContext>,
    mut grid: ResMut<ShipGridResource>,
    registry: Res<BlockRegistryResource>,
    mut sub_grids: ResMut<SubGridRegistry>,
    root_body: Res<RootBodyHandle>,
    mut channels: ResMut<SignalChannelTable>,
    mut dirty: ResMut<AggregationDirty>,
    mut collider_handles: ResMut<ChunkColliderHandles>,
    mut mech_dirty: ResMut<MechanicalDirtyGrids>,
    mut block_index: ResMut<FunctionalBlockIndex>,
    bridge: Res<NetworkBridge>,
) {
    use voxeldust_core::block::sub_block::{SubBlockType, face_to_offset};
    use voxeldust_core::signal::{ChannelMergeStrategy, SignalScope};

    // Re-compute membership for dirty sub-grids (blocks placed/removed on child side).
    let dirty_ids: Vec<_> = mech_dirty.dirty.drain().collect();
    for &sg_id in &dirty_ids {
        let Some(sg_data) = sub_grids.grids.get(&sg_id) else { continue };
        let mount_pos = sg_data.mount_pos;
        let mount_face = sg_data.mount_face;
        let output_face = mount_face;

        // Re-BFS from the output face (respects sub-grid boundaries).
        let new_members = block::ship_grid::compute_sub_grid_members(
            &grid.0, &registry.0, mount_pos, output_face, sg_id.0,
        );

        // Determine which chunks are affected (old members + new members).
        let old_members = &sub_grids.grids[&sg_id].members;
        let mut affected_chunks: std::collections::HashSet<glam::IVec3> = std::collections::HashSet::new();
        for &pos in old_members.iter().chain(new_members.iter()) {
            affected_chunks.insert(block::ShipGrid::world_to_chunk(pos.x, pos.y, pos.z).0);
        }

        // Clear old assignments.
        for &pos in old_members {
            grid.0.set_sub_grid(pos, 0);
        }
        // Set new assignments.
        for &pos in &new_members {
            grid.0.set_sub_grid(pos, sg_id.0);
        }

        // Update the registry entry.
        if let Some(sg_data) = sub_grids.grids.get_mut(&sg_id) {
            sg_data.members = new_members;
        }

        // Update child body mass from new members.
        let mut child_mass = 0.0f64;
        if let Some(sg_data) = sub_grids.grids.get(&sg_id) {
            for &pos in &sg_data.members {
                let bid = grid.0.get_block(pos.x, pos.y, pos.z);
                child_mass += registry.0.get(bid).density as f64;
            }
        }
        child_mass = child_mass.max(1.0);
        if let Some(sg_data) = sub_grids.grids.get(&sg_id) {
            if let Some(body) = rapier.rigid_body_set.get_mut(sg_data.body_handle) {
                body.set_additional_mass(child_mass as f32, true);
            }
        }

        // Rebuild colliders for all affected chunks.
        for chunk_key in affected_chunks {
            rebuild_chunk_collider(
                &grid.0, chunk_key, &registry.0,
                &mut rapier, &mut collider_handles,
                root_body.0, &mut sub_grids,
            );
        }

        tracing::debug!(
            sg_id = sg_id.0,
            members = sub_grids.grids.get(&sg_id).map(|s| s.members.len()).unwrap_or(0),
            "re-computed sub-grid membership",
        );
    }

    // Broadcast assignment changes to client after dirty re-computation.
    if !dirty_ids.is_empty() {
        let assignments: Vec<(glam::IVec3, u32)> = dirty_ids.iter()
            .flat_map(|sg_id| {
                sub_grids.grids.get(sg_id).into_iter().flat_map(|sg_data| {
                    sg_data.members.iter().map(|&pos| (pos, sg_id.0))
                })
            })
            .collect();
        if !assignments.is_empty() {
            let msg = ServerMsg::SubGridAssignmentUpdate(SubGridAssignmentData {
                assignments,
            });
            let cr = bridge.client_registry.clone();
            tokio::spawn(async move {
                if let Ok(reg) = cr.try_read() {
                    for addr in reg.udp_addrs() {
                        if let Some(session) = reg.session_for_udp(addr) {
                            let _ = reg.send_tcp(session, &msg).await;
                        }
                    }
                }
            });
        }
    }

    // Check for mechanical sub-blocks that don't yet have a sub-grid.
    // Scan all chunks for RotorMount/PistonMount/HingeMount/SliderMount.
    // On startup, init_functional_blocks handles functional blocks but not sub-block mounts.
    // After placement, process_sub_block_edits stores the sub-block but doesn't create the joint.
    // This system bridges that gap.

    let mechanical_types = [
        SubBlockType::RotorMount,
        SubBlockType::PistonMount,
        SubBlockType::HingeMount,
        SubBlockType::SliderMount,
    ];

    // Collect all mechanical mount positions that DON'T have a sub-grid yet.
    let mut new_mounts: Vec<(glam::IVec3, u8, SubBlockType)> = Vec::new();
    let cs = block::CHUNK_SIZE as i32;

    for (chunk_key, chunk) in grid.0.iter_chunks() {
        for (flat_idx, elements) in chunk.iter_sub_blocks() {
            let (lx, ly, lz) = block::index_to_xyz(flat_idx as usize);
            let world_pos = glam::IVec3::new(
                chunk_key.x * cs + lx as i32,
                chunk_key.y * cs + ly as i32,
                chunk_key.z * cs + lz as i32,
            );
            for elem in elements {
                if !mechanical_types.contains(&elem.element_type) { continue; }
                // Check if this mount already has a sub-grid.
                let already_exists = sub_grids.grids.values()
                    .any(|sg| sg.mount_pos == world_pos && sg.mount_face == elem.face);
                if !already_exists {
                    new_mounts.push((world_pos, elem.face, elem.element_type));
                }
            }
        }
    }

    if new_mounts.is_empty() { return; }

    for (mount_pos, face, elem_type) in new_mounts {
        // Determine block ID for registry lookup.
        let block_id = match elem_type {
            SubBlockType::RotorMount | SubBlockType::HingeMount => block::BlockId::ROTOR,
            SubBlockType::PistonMount | SubBlockType::SliderMount => block::BlockId::PISTON,
            _ => continue,
        };

        let mech_props = match registry.0.mechanical_props(block_id) {
            Some(p) => p,
            None => continue,
        };

        // Compute child sub-grid membership via BFS.
        // The mount face IS the output direction (the face the sub-block element points toward).
        let output_face = face;
        // For new sub-grids, pass 0 as own_sg_id (no blocks belong to it yet).
        let members = block::ship_grid::compute_sub_grid_members(
            &grid.0, &registry.0, mount_pos, output_face, 0,
        );

        // Allocate sub-grid ID.
        let sg_id = voxeldust_core::ecs::components::SubGridId(sub_grids.next_id);
        sub_grids.next_id += 1;

        // Assign blocks to sub-grid.
        for &pos in &members {
            grid.0.set_sub_grid(pos, sg_id.0);
        }

        // Compute child mass properties.
        let mut child_mass = 0.0f64;
        let mut child_weighted_pos = glam::DVec3::ZERO;
        for &pos in &members {
            let bid = grid.0.get_block(pos.x, pos.y, pos.z);
            let def = registry.0.get(bid);
            let m = def.density as f64;
            child_mass += m;
            child_weighted_pos += glam::DVec3::new(
                pos.x as f64 + 0.5, pos.y as f64 + 0.5, pos.z as f64 + 0.5,
            ) * m;
        }
        child_mass = child_mass.max(1.0);
        let child_com = child_weighted_pos / child_mass;

        // Create child rigid body.
        let face_offset = face_to_offset(output_face);
        let anchor = rapier3d::math::Vector::new(
            mount_pos.x as f32 + 0.5 + face_offset.x as f32 * 0.5,
            mount_pos.y as f32 + 0.5 + face_offset.y as f32 * 0.5,
            mount_pos.z as f32 + 0.5 + face_offset.z as f32 * 0.5,
        );
        // Kinematic body: rotation set directly each tick, no motor/solver involvement.
        // Collisions work (kinematic pushes dynamic), zero jitter by design.
        let child_body = rapier3d::dynamics::RigidBodyBuilder::kinematic_position_based()
            .translation(anchor)
            .build();
        let child_handle = rapier.rigid_body_set.insert(child_body);

        // Detect parent sub-grid: if the mount block belongs to another sub-grid,
        // this is a nested mount (e.g., rotor on a piston, turret on a rotating base).
        let parent_sg_id = grid.0.sub_grid_id(mount_pos);
        let parent_grid = if parent_sg_id != 0 {
            voxeldust_core::ecs::components::SubGridId(parent_sg_id)
        } else {
            voxeldust_core::ecs::components::SubGridId::ROOT
        };

        // Create mechanism arm collider (physical rod/axle at the mount).
        let arm_radius = 0.08_f32;
        let face_normal = glam::Vec3::new(
            face_to_offset(output_face).x as f32,
            face_to_offset(output_face).y as f32,
            face_to_offset(output_face).z as f32,
        );
        // Arm collider starts at the face surface (block center + 0.5 along face normal).
        let mount_face_pos = glam::Vec3::new(
            mount_pos.x as f32 + 0.5,
            mount_pos.y as f32 + 0.5,
            mount_pos.z as f32 + 0.5,
        ) + face_normal * 0.5;

        let arm_collider_handle = match mech_props.joint_type {
            block::JointType::Revolute => {
                // Rotor axle: small fixed cuboid at the joint center (face surface).
                let collider = rapier3d::geometry::ColliderBuilder::cuboid(arm_radius, arm_radius, 0.06)
                    .translation(rapier3d::math::Vector::new(mount_face_pos.x, mount_face_pos.y, mount_face_pos.z))
                    .build();
                let RapierContext { ref mut collider_set, ref mut rigid_body_set, .. } = *rapier;
                Some(collider_set.insert_with_parent(collider, root_body.0, rigid_body_set))
            }
            block::JointType::Prismatic => {
                // Piston arm: starts at zero length at face surface, grows with extension.
                let collider = rapier3d::geometry::ColliderBuilder::cuboid(arm_radius, arm_radius, 0.001)
                    .translation(rapier3d::math::Vector::new(mount_face_pos.x, mount_face_pos.y, mount_face_pos.z))
                    .build();
                let RapierContext { ref mut collider_set, ref mut rigid_body_set, .. } = *rapier;
                Some(collider_set.insert_with_parent(collider, root_body.0, rigid_body_set))
            }
        };

        // Register sub-grid (colliders will be built by rebuild_chunk_collider below).
        let members_clone = members.clone();
        sub_grids.grids.insert(sg_id, SubGridData {
            body_handle: child_handle,
            arm_collider: arm_collider_handle,
            members,
            mount_pos,
            mount_face: face,
            parent_grid,
            collider_handles: std::collections::HashMap::new(),
        });

        // Spawn functional entity for signal integration.
        let kind = match elem_type {
            SubBlockType::RotorMount | SubBlockType::HingeMount => block::FunctionalBlockKind::Rotor,
            SubBlockType::PistonMount | SubBlockType::SliderMount => block::FunctionalBlockKind::Piston,
            _ => continue,
        };

        let mut entity_cmds = commands.spawn((
            voxeldust_core::ecs::components::FunctionalBlockRef {
                world_pos: mount_pos,
                block_id: block_id,
                kind,
            },
            MechanicalState {
                current: 0.0,
                target: 0.0,
                max_speed: mech_props.max_speed as f32,
                max_force: mech_props.max_force,
                max_range: mech_props.max_range,
                joint_type: mech_props.joint_type,
                child_grid_id: sg_id,
                status: MechanicalStatus::Idle,
                host_pos: mount_pos,
                face,
                stuck_ticks: 0,
                prev_error: 0.0,
                velocity_override: 0.0,
                locked: false,
            },
            SignalSubscriber::default(),
            SignalPublisher { bindings: Vec::new() },
        ));
        add_default_signal_bindings(&mut entity_cmds, kind, block_id, mount_pos, &grid.0, &registry.0, &mut channels);
        let entity = entity_cmds.id();

        block_index.0.insert(mount_pos, entity);
        dirty.0 = true;

        // Rebuild chunk colliders for all chunks containing sub-grid members.
        // This removes sub-grid blocks from root colliders and adds them to the
        // sub-grid's colliders with proper collision groups.
        let affected_chunks: std::collections::HashSet<glam::IVec3> = members_clone.iter()
            .map(|&pos| block::ShipGrid::world_to_chunk(pos.x, pos.y, pos.z).0)
            .collect();
        for chunk_key in affected_chunks {
            rebuild_chunk_collider(
                &grid.0, chunk_key, &registry.0,
                &mut rapier, &mut collider_handles,
                root_body.0, &mut sub_grids,
            );
        }

        // Broadcast new sub-grid assignments to all clients.
        {
            let assignments: Vec<(glam::IVec3, u32)> = members_clone.iter()
                .map(|&pos| (pos, sg_id.0))
                .collect();
            if !assignments.is_empty() {
                let msg = ServerMsg::SubGridAssignmentUpdate(SubGridAssignmentData {
                    assignments,
                });
                let cr = bridge.client_registry.clone();
                tokio::spawn(async move {
                    if let Ok(reg) = cr.try_read() {
                        for addr in reg.udp_addrs() {
                            if let Some(session) = reg.session_for_udp(addr) {
                                let _ = reg.send_tcp(session, &msg).await;
                            }
                        }
                    }
                });
            }
        }

        tracing::info!(
            mount = ?mount_pos, face = face,
            kind = ?elem_type,
            child_blocks = sub_grids.grids[&sg_id].members.len(),
            sg_id = sg_id.0,
            "created mechanical sub-grid",
        );
    }
}

/// Aggregation system: recomputes ship physical properties from block composition.
/// Runs after entity lifecycle ops, only when the dirty flag is set.
fn aggregate_ship_properties(
    mut dirty: ResMut<AggregationDirty>,
    grid: Res<ShipGridResource>,
    registry: Res<BlockRegistryResource>,
    mut ship_props: ResMut<ShipProps>,
    mut ship_com: ResMut<ShipCenterOfMass>,
    block_query: Query<&FunctionalBlockRef>,
    config: Res<ShipConfig>,
    bridge: Res<NetworkBridge>,
) {
    if !dirty.0 {
        return;
    }
    dirty.0 = false;

    // Compute mass properties from all blocks.
    let mass = block::aggregation::compute_mass_properties(&grid.0, &registry.0);

    // Collect thruster positions + orientations from ECS entities.
    let thrusters: Vec<_> = block_query
        .iter()
        .filter(|b| b.kind == FunctionalBlockKind::Thruster)
        .map(|b| {
            let orientation = grid
                .0
                .get_meta(b.world_pos.x, b.world_pos.y, b.world_pos.z)
                .map(|m| m.orientation)
                .unwrap_or_default();
            (b.world_pos, b.block_id, orientation)
        })
        .collect();

    // Compute thrust properties from thruster block positions.
    let thrust =
        block::aggregation::compute_thrust_properties(&thrusters, mass.center_of_mass, &registry.0);

    // Build the final ShipPhysicalProperties.
    let new_props = block::aggregation::build_ship_properties(&mass, &thrust);
    ship_props.0 = new_props.clone();
    ship_com.0 = mass.center_of_mass;

    info!(
        mass = format!("{:.0} kg", mass.total_mass_kg),
        blocks = mass.block_count,
        thrusters = thrust.thruster_count,
        forward_thrust = format!("{:.0} N", new_props.max_thrust_forward_n),
        torque = format!("{:.0} N·m", new_props.max_torque_nm),
        "ship properties aggregated from blocks"
    );

    // Send updated properties to the system shard via QUIC.
    if let Some(host_id) = config.host_shard_id {
        let msg = voxeldust_core::shard_message::ShardMsg::ShipPropertiesUpdate(
            voxeldust_core::shard_message::ShipPropertiesUpdateData {
                ship_id: config.ship_id,
                mass_kg: new_props.mass_kg,
                max_thrust_forward_n: new_props.max_thrust_forward_n,
                max_thrust_reverse_n: new_props.max_thrust_reverse_n,
                max_torque_nm: new_props.max_torque_nm,
                thrust_multiplier: new_props.thrust_multiplier,
                dimensions: new_props.dimensions,
            },
        );
        if let Ok(reg) = bridge.peer_registry.try_read() {
            if let Some(addr) = reg.quic_addr(host_id) {
                let _ = bridge.quic_send_tx.try_send((host_id, addr, msg));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Collider sync to host shard
// ---------------------------------------------------------------------------

/// Tracks whether collider shapes need re-syncing to the host shard.
/// Set true alongside AggregationDirty when blocks change.
/// Also true on first tick (initial sync).
#[derive(Resource)]
struct ColliderSyncDirty(bool);

/// Last-sent edit sequence per chunk, for delta tracking.
/// If a chunk's current seq differs from the stored value, it needs re-sync.
#[derive(Resource, Default)]
struct ColliderSyncSeqs(std::collections::HashMap<glam::IVec3, u64>);

/// Sync ship collider shapes to the host shard (planet or system).
/// Runs when blocks change (ColliderSyncDirty flag). Sends the full set of
/// collider shapes so the host can build Rapier compound colliders for
/// physical collision with the ship hull.
fn sync_colliders_to_host(
    mut dirty: ResMut<ColliderSyncDirty>,
    grid: Res<ShipGridResource>,
    registry: Res<BlockRegistryResource>,
    hull_bounds: Res<ShipHullBounds>,
    config: Res<ShipConfig>,
    bridge: Res<NetworkBridge>,
    mut seqs: ResMut<ColliderSyncSeqs>,
) {
    if !dirty.0 {
        return;
    }
    dirty.0 = false;

    // Build collider shapes for all non-empty chunks.
    let origin = glam::Vec3::ZERO; // ship-local, no offset
    let mut chunk_data: Vec<voxeldust_core::shard_message::ChunkColliderData> = Vec::new();
    let mut total_shapes = 0usize;

    for (chunk_key, chunk) in grid.0.iter_chunks() {
        if chunk.is_empty() {
            continue;
        }
        // Check if this chunk changed since last sync.
        let current_seq = chunk.edit_seq();
        let prev_seq = seqs.0.get(&chunk_key).copied().unwrap_or(0);
        // Always include all chunks in the sync message (host needs the full set).
        // The seq tracking is for future delta-only optimization.
        seqs.0.insert(chunk_key, current_seq);

        let shapes = grid.0.chunk_collider_shapes(chunk_key, origin, &registry.0);
        if shapes.is_empty() {
            continue;
        }
        total_shapes += shapes.len();
        chunk_data.push(voxeldust_core::shard_message::ChunkColliderData {
            chunk_key,
            shapes,
        });
    }

    if chunk_data.is_empty() {
        return;
    }

    let hull_min = glam::Vec3::new(
        hull_bounds.min.x as f32,
        hull_bounds.min.y as f32,
        hull_bounds.min.z as f32,
    );
    let hull_max = glam::Vec3::new(
        hull_bounds.max.x as f32 + 1.0,
        hull_bounds.max.y as f32 + 1.0,
        hull_bounds.max.z as f32 + 1.0,
    );

    let msg = ShardMsg::ShipColliderSync(voxeldust_core::shard_message::ShipColliderSyncData {
        ship_id: config.ship_id,
        chunks: chunk_data,
        hull_min,
        hull_max,
    });

    // Send to host shard (system or planet shard that manages this ship's exterior).
    if let Some(host_id) = config.host_shard_id {
        if let Ok(reg) = bridge.peer_registry.try_read() {
            if let Some(addr) = reg.quic_addr(host_id) {
                let _ = bridge.quic_send_tx.try_send((host_id, addr, msg));
                info!(
                    ship_id = config.ship_id,
                    chunks = seqs.0.len(),
                    shapes = total_shapes,
                    "synced collider shapes to host shard"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Signal default bindings
// ---------------------------------------------------------------------------

/// Convert u8 discriminant back to SignalProperty (for deserialized bindings).
fn signal_property_from_u8(v: u8) -> SignalProperty {
    match v {
        0 => SignalProperty::Active,
        1 => SignalProperty::Throttle,
        2 => SignalProperty::Angle,
        3 => SignalProperty::Extension,
        4 => SignalProperty::Pressure,
        5 => SignalProperty::Speed,
        6 => SignalProperty::Level,
        7 => SignalProperty::SwitchState,
        8 => SignalProperty::Boost,
        9 => SignalProperty::Status,
        _ => SignalProperty::Throttle,
    }
}

fn add_default_signal_bindings(
    entity_cmds: &mut bevy_ecs::system::EntityCommands,
    kind: FunctionalBlockKind,
    block_id: BlockId,
    pos: glam::IVec3,
    grid: &voxeldust_core::block::ShipGrid,
    registry: &voxeldust_core::block::BlockRegistry,
    channels: &mut SignalChannelTable,
) {
    use voxeldust_core::signal::{ChannelMergeStrategy, SignalScope};

    // Check for persisted signal bindings (from a previous config panel save).
    // If found, use them instead of the kind-specific defaults below.
    if let Some((sub_bindings, pub_bindings)) = grid.saved_signal_bindings(pos) {
        if !sub_bindings.is_empty() {
            entity_cmds.insert(SignalSubscriber {
                bindings: sub_bindings.iter().map(|(name, prop)| {
                    SubscribeBinding {
                        channel_id: channels.resolve_or_create(name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0),
                        property: signal_property_from_u8(*prop),
                    }
                }).collect(),
            });
        }
        if !pub_bindings.is_empty() {
            entity_cmds.insert(SignalPublisher {
                bindings: pub_bindings.iter().map(|(name, prop)| {
                    PublishBinding {
                        channel_id: channels.resolve_or_create(name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0),
                        property: signal_property_from_u8(*prop),
                    }
                }).collect(),
            });
        }
        // Still apply kind-specific non-signal components (ThrusterState, ReactorState, etc.)
        // by falling through the match below. The signal bindings are already set.
    }

    match kind {
        FunctionalBlockKind::Thruster => {
            let orientation = grid.get_meta(pos.x, pos.y, pos.z)
                .map(|m| m.orientation)
                .unwrap_or(block::BlockOrientation::DEFAULT);
            let dir = orientation.facing_direction();

            // Build signal subscriptions: throttle channel + optional boost channel.
            let mut bindings = Vec::new();
            if let Some(ch) = grid.channel_override(pos) {
                let id = channels.resolve_or_create(ch, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0);
                bindings.push(SubscribeBinding { channel_id: id, property: SignalProperty::Throttle });
            }
            if let Some(bc) = grid.boost_channel(pos) {
                let id = channels.resolve_or_create(bc, SignalScope::Local, ChannelMergeStrategy::Max, 0);
                bindings.push(SubscribeBinding { channel_id: id, property: SignalProperty::Boost });
                info!(pos = format!("({},{},{})", pos.x, pos.y, pos.z), boost = bc, "thruster boost binding created");
            }
            entity_cmds.insert(SignalSubscriber { bindings });

            entity_cmds.insert(ThrusterState::default());
            entity_cmds.insert(ThrusterBlock {
                thrust_n: registry.thruster_props(block_id).map(|p| p.thrust_n).unwrap_or(50_000.0),
                block_pos: pos,
                direction: dir,
            });
        }
        FunctionalBlockKind::Reactor => {
            // Reactor state with wireless power config from grid or defaults.
            let pp = registry.power_props(block_id);
            let mut rs = ReactorState {
                max_generation_w: pp.map(|p| p.generation_w).unwrap_or(500_000.0),
                broadcast_range: pp.map(|p| p.broadcast_range).unwrap_or(50.0),
                ..Default::default()
            };
            if let Some(block::PowerConfig::Source { circuits, broadcast_range }) = grid.power_config(pos) {
                rs.broadcast_range = *broadcast_range;
                rs.circuits = circuits.iter().map(|(name, frac)| PowerCircuit {
                    name: name.clone(),
                    fraction: *frac,
                    supply_w: 0.0,
                    demand_w: 0.0,
                    power_ratio: 1.0,
                }).collect();
            }
            entity_cmds.insert(rs);
            entity_cmds.insert(ReactorConsumerCache::default());
        }
        FunctionalBlockKind::Seat => {
            // Use saved seat config if available (player-configured via config panel).
            // Otherwise use preset defaults based on block ID.
            if let Some(saved) = grid.saved_seat_config(pos) {
                let bindings = saved.bindings.iter().map(|(label, source_u8, key_name, key_mode_u8, axis_dir_u8, channel_name, property_u8)| {
                    SeatInputBinding {
                        label: label.clone(),
                        source: SeatInputSource::from_u8(*source_u8).unwrap_or(SeatInputSource::Key),
                        key_name: key_name.clone(),
                        key_mode: KeyMode::from_u8(*key_mode_u8).unwrap_or(KeyMode::Momentary),
                        axis_direction: AxisDirection::from_u8(*axis_dir_u8).unwrap_or(AxisDirection::Positive),
                        channel_id: channels.resolve_or_create(channel_name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0),
                        property: signal_property_from_u8(*property_u8),
                    }
                }).collect();
                let seated_channel_id = if saved.seated_channel_name.is_empty() {
                    None
                } else {
                    Some(channels.resolve_or_create(&saved.seated_channel_name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0))
                };
                entity_cmds.insert(SeatChannelMapping { bindings, seated_channel_id });
            } else {
                // Use preset defaults based on block type.
                let preset = if block_id == BlockId::COCKPIT {
                    seat_presets::SeatPreset::Pilot
                } else {
                    seat_presets::SeatPreset::Generic
                };
                let binding_configs = preset.default_bindings();
                let bindings = binding_configs.iter().map(|cfg| {
                    SeatInputBinding {
                        label: cfg.label.clone(),
                        source: cfg.source,
                        key_name: cfg.key_name.clone(),
                        key_mode: cfg.key_mode,
                        axis_direction: cfg.axis_direction,
                        channel_id: channels.resolve_or_create(&cfg.channel_name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0),
                        property: cfg.property,
                    }
                }).collect();
                let seated_ch_name = preset.default_seated_channel();
                let seated_channel_id = Some(channels.resolve_or_create(seated_ch_name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0));
                entity_cmds.insert(SeatChannelMapping { bindings, seated_channel_id });
            }
        }
        FunctionalBlockKind::Sensor => {
            // Sensor publishes to its configured channel. Player sets via config panel.
            if let Some(ch) = grid.channel_override(pos) {
                let channel_id = channels.resolve_or_create(
                    ch, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0,
                );
                entity_cmds.insert(SignalPublisher {
                    bindings: vec![PublishBinding {
                        channel_id,
                        property: SignalProperty::Active,
                    }],
                });
            }
        }
        FunctionalBlockKind::SignalConverter => {
            // Signal converters start with an empty rule set (player configures).
            entity_cmds.insert(SignalConverterConfig::default());
        }
        FunctionalBlockKind::CruiseDrive => {
            // Cruise drive subscribes to "cruise" channel for on/off.
            // When active, publishes boost_multiplier to configured boost channels.
            let boost_multiplier = registry.cruise_drive_props(block_id)
                .map(|p| p.boost_multiplier)
                .unwrap_or(100.0);

            // Subscribe to "cruise" channel for on/off from pilot seat.
            if let Some(ch) = grid.channel_override(pos) {
                let id = channels.resolve_or_create(ch, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0);
                entity_cmds.insert(SignalSubscriber {
                    bindings: vec![SubscribeBinding { channel_id: id, property: SignalProperty::Throttle }],
                });
            }

            // Publish boost via standard SignalPublisher — visible and configurable in config panel.
            // Only if a boost channel is configured (starter ship sets it, player can add via UI).
            if let Some(boost_ch_name) = grid.boost_channel(pos) {
                let boost_channel_id = channels.resolve_or_create(
                    boost_ch_name, SignalScope::Local, ChannelMergeStrategy::Max, 0,
                );
                entity_cmds.insert(SignalPublisher {
                    bindings: vec![PublishBinding {
                        channel_id: boost_channel_id,
                        property: SignalProperty::Boost,
                    }],
                });
            }

            entity_cmds.insert(CruiseDriveState {
                throttle: 0.0,
                boost_multiplier,
            });
        }
        FunctionalBlockKind::FlightComputer => {
            let cfg = seat_presets::default_flight_computer_config();
            macro_rules! resolve_ch { ($name:expr) => { Some(channels.resolve_or_create($name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0)) }; }
            entity_cmds.insert(FlightComputerState {
                yaw_cw_channel: resolve_ch!(&cfg.yaw_cw_channel),
                yaw_ccw_channel: resolve_ch!(&cfg.yaw_ccw_channel),
                pitch_up_channel: resolve_ch!(&cfg.pitch_up_channel),
                pitch_down_channel: resolve_ch!(&cfg.pitch_down_channel),
                roll_cw_channel: resolve_ch!(&cfg.roll_cw_channel),
                roll_ccw_channel: resolve_ch!(&cfg.roll_ccw_channel),
                toggle_channel: resolve_ch!(&cfg.toggle_channel),
                damping_gain: cfg.damping_gain,
                dead_zone: cfg.dead_zone,
                max_correction: cfg.max_correction,
                ..Default::default()
            });
        }
        FunctionalBlockKind::HoverModule => {
            let cfg = seat_presets::default_hover_module_config();
            macro_rules! resolve_ch { ($name:expr) => { Some(channels.resolve_or_create($name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0)) }; }
            entity_cmds.insert(HoverModuleState {
                thrust_forward_channel: resolve_ch!(&cfg.thrust_forward_channel),
                thrust_reverse_channel: resolve_ch!(&cfg.thrust_reverse_channel),
                thrust_right_channel: resolve_ch!(&cfg.thrust_right_channel),
                thrust_left_channel: resolve_ch!(&cfg.thrust_left_channel),
                thrust_up_channel: resolve_ch!(&cfg.thrust_up_channel),
                thrust_down_channel: resolve_ch!(&cfg.thrust_down_channel),
                yaw_cw_channel: resolve_ch!(&cfg.yaw_cw_channel),
                yaw_ccw_channel: resolve_ch!(&cfg.yaw_ccw_channel),
                pitch_up_channel: resolve_ch!(&cfg.pitch_up_channel),
                pitch_down_channel: resolve_ch!(&cfg.pitch_down_channel),
                roll_cw_channel: resolve_ch!(&cfg.roll_cw_channel),
                roll_ccw_channel: resolve_ch!(&cfg.roll_ccw_channel),
                activate_channel: resolve_ch!(&cfg.activate_channel),
                cutoff_channel: resolve_ch!(&cfg.cutoff_channel),
                ..Default::default()
            });
        }
        FunctionalBlockKind::Autopilot => {
            let cfg = seat_presets::default_autopilot_config();
            macro_rules! resolve_ch { ($name:expr) => { Some(channels.resolve_or_create($name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0)) }; }
            entity_cmds.insert(AutopilotBlockState {
                yaw_cw_channel: resolve_ch!(&cfg.yaw_cw_channel),
                yaw_ccw_channel: resolve_ch!(&cfg.yaw_ccw_channel),
                pitch_up_channel: resolve_ch!(&cfg.pitch_up_channel),
                pitch_down_channel: resolve_ch!(&cfg.pitch_down_channel),
                roll_cw_channel: resolve_ch!(&cfg.roll_cw_channel),
                roll_ccw_channel: resolve_ch!(&cfg.roll_ccw_channel),
                engage_channel: resolve_ch!(&cfg.engage_channel),
                ..Default::default()
            });
        }
        FunctionalBlockKind::WarpComputer => {
            let cfg = seat_presets::default_warp_computer_config();
            macro_rules! resolve_ch { ($name:expr) => { Some(channels.resolve_or_create($name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0)) }; }
            entity_cmds.insert(WarpComputerState {
                cycle_channel: resolve_ch!(&cfg.cycle_channel),
                accept_channel: resolve_ch!(&cfg.accept_channel),
                cancel_channel: resolve_ch!(&cfg.cancel_channel),
                ..Default::default()
            });
        }
        FunctionalBlockKind::EngineController => {
            let cfg = seat_presets::default_engine_controller_config();
            macro_rules! resolve_ch { ($name:expr) => { Some(channels.resolve_or_create($name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0)) }; }
            entity_cmds.insert(EngineControllerState {
                thrust_forward_channel: resolve_ch!(&cfg.thrust_forward_channel),
                thrust_reverse_channel: resolve_ch!(&cfg.thrust_reverse_channel),
                thrust_right_channel: resolve_ch!(&cfg.thrust_right_channel),
                thrust_left_channel: resolve_ch!(&cfg.thrust_left_channel),
                thrust_up_channel: resolve_ch!(&cfg.thrust_up_channel),
                thrust_down_channel: resolve_ch!(&cfg.thrust_down_channel),
                yaw_cw_channel: resolve_ch!(&cfg.yaw_cw_channel),
                yaw_ccw_channel: resolve_ch!(&cfg.yaw_ccw_channel),
                pitch_up_channel: resolve_ch!(&cfg.pitch_up_channel),
                pitch_down_channel: resolve_ch!(&cfg.pitch_down_channel),
                roll_cw_channel: resolve_ch!(&cfg.roll_cw_channel),
                roll_ccw_channel: resolve_ch!(&cfg.roll_ccw_channel),
                toggle_channel: resolve_ch!(&cfg.toggle_channel),
                ..Default::default()
            });
        }
        FunctionalBlockKind::Rotor => {
            if let Some(ch) = grid.channel_override(pos) {
                let id = channels.resolve_or_create(ch, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0);
                entity_cmds.insert(SignalSubscriber {
                    bindings: vec![SubscribeBinding { channel_id: id, property: SignalProperty::Angle }],
                });
            }
        }
        FunctionalBlockKind::Piston => {
            if let Some(ch) = grid.channel_override(pos) {
                let id = channels.resolve_or_create(ch, SignalScope::Local, ChannelMergeStrategy::LastWrite, 0);
                entity_cmds.insert(SignalSubscriber {
                    bindings: vec![SubscribeBinding { channel_id: id, property: SignalProperty::Extension }],
                });
            }
        }
        _ => {
            // Other kinds: no default signal bindings.
        }
    }

    // Attach PoweredBy for blocks with power config (thrusters, etc.).
    if let Some(block::PowerConfig::Consumer { reactor_pos, circuit }) = grid.power_config(pos) {
        let consumption_w = registry.power_props(block_id)
            .map(|p| p.consumption_w)
            .unwrap_or(0.0);
        entity_cmds.insert(PoweredBy {
            reactor_pos: *reactor_pos,
            circuit: circuit.clone(),
            consumption_w,
            placed_by: 0, // starter ship — same owner as reactor
        });
    }
}

// ---------------------------------------------------------------------------
// Wireless power + Signal pipeline systems
// ---------------------------------------------------------------------------

/// Rebuild the per-reactor consumer cache when dirty.
/// Scans all PoweredBy consumers, applies range + auth checks, maps to circuit index.
fn rebuild_reactor_consumers(
    mut reactors: Query<(&ReactorState, &FunctionalBlockRef, &mut ReactorConsumerCache)>,
    consumers: Query<(Entity, &PoweredBy, &FunctionalBlockRef)>,
) {
    for (rs, reactor_ref, mut cache) in &mut reactors {
        if !cache.dirty { continue; }
        cache.dirty = false;
        cache.entries.clear();
        for (entity, powered_by, consumer_ref) in &consumers {
            if powered_by.reactor_pos != reactor_ref.world_pos { continue; }
            let dist = (consumer_ref.world_pos - reactor_ref.world_pos).as_dvec3().length();
            if dist > rs.broadcast_range as f64 { continue; }
            // Auth check.
            let authorized = match &rs.access {
                PowerAccess::OwnerOnly => powered_by.placed_by == rs.placed_by,
                PowerAccess::AllowList(list) => powered_by.placed_by == rs.placed_by || list.contains(&powered_by.placed_by),
                PowerAccess::Open => true,
            };
            if !authorized { continue; }
            if let Some(idx) = rs.circuits.iter().position(|c| c.name == powered_by.circuit) {
                cache.entries.push(CachedConsumer {
                    entity,
                    circuit_idx: idx,
                    consumption_w: powered_by.consumption_w,
                });
            }
        }
    }
}

/// Wireless power budget: each reactor broadcasts power to consumers in range.
/// Uses the pre-computed ReactorConsumerCache for O(consumers) instead of O(reactors * consumers).
fn compute_power_budget(
    mut reactors: Query<(&mut ReactorState, &ReactorConsumerCache)>,
    thruster_states: Query<&ThrusterState>,
) {
    for (mut rs, cache) in &mut reactors {
        let generation = rs.max_generation_w * if rs.active { rs.throttle as f64 } else { 0.0 };
        for c in &mut rs.circuits {
            c.supply_w = generation * c.fraction as f64;
            c.demand_w = 0.0;
        }
        for consumer in &cache.entries {
            let load = thruster_states.get(consumer.entity)
                .map(|ts| ts.throttle.abs())
                .unwrap_or(1.0);
            if consumer.circuit_idx < rs.circuits.len() {
                rs.circuits[consumer.circuit_idx].demand_w += consumer.consumption_w * load as f64;
            }
        }
        for c in &mut rs.circuits {
            c.power_ratio = if c.demand_w > 0.0 {
                (c.supply_w / c.demand_w).min(1.0) as f32
            } else {
                1.0
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Signal pipeline systems
// ---------------------------------------------------------------------------

/// Check if a block entity has power from its reactor circuit.
/// Returns the circuit's power_ratio (0.0 = no power, 1.0 = full power).
/// Returns 0.0 if the block has no PoweredBy, reactor is missing/inactive, or circuit not found.
fn block_power_ratio(
    entity: Entity,
    powered_by_query: &Query<&PoweredBy>,
    block_index: &FunctionalBlockIndex,
    reactors: &Query<&ReactorState>,
) -> f32 {
    let Some(powered_by) = powered_by_query.get(entity).ok() else { return 0.0 };
    let Some(&reactor_entity) = block_index.0.get(&powered_by.reactor_pos) else { return 0.0 };
    let Ok(rs) = reactors.get(reactor_entity) else { return 0.0 };
    if !rs.active { return 0.0; }
    rs.circuits.iter()
        .find(|c| c.name == powered_by.circuit)
        .map(|c| c.power_ratio)
        .unwrap_or(0.0)
}

/// Flight computer block system: reads angular velocity, injects counter-rotation
/// into rotation channels when pilot input is near zero.
///
/// Runs AFTER signal_publish + merge_pending. Reads merged channel values,
/// writes damped corrections via set_value_direct.
///
/// Uses capped proportional control: correction is proportional to angular
/// velocity but never exceeds max_correction throttle. Unconditionally stable
/// regardless of delay or ship mass.
fn flight_computer_system(
    mut channels: ResMut<SignalChannelTable>,
    mut query: Query<(Entity, &mut FlightComputerState)>,
    powered_by_query: Query<&PoweredBy>,
    block_index: Res<FunctionalBlockIndex>,
    reactors: Query<&ReactorState>,
    exterior: Res<ShipExterior>,
) {
    let ang_vel = exterior.rotation.inverse() * exterior.angular_velocity;

    for (entity, mut fc) in &mut query {
        // Power check: skip if no power available.
        if block_power_ratio(entity, &powered_by_query, &block_index, &reactors) <= 0.0 { continue; }

        // Toggle: rising edge flips active state.
        let toggle_val = fc.toggle_channel.map(|id| channels.read_value(id).as_f32()).unwrap_or(0.0);
        if toggle_val > 0.5 && fc.prev_toggle_value <= 0.5 {
            fc.active = !fc.active;
            info!(active = fc.active, "flight computer toggled");
        }
        fc.prev_toggle_value = toggle_val;
        if !fc.active { continue; }

        let gain = fc.damping_gain as f64;
        let dead_zone = fc.dead_zone as f64;
        let max_corr = fc.max_correction as f64;

        // Helper: for each axis pair, read current channel values (from seat publish).
        // If pilot input near zero on both CW and CCW, compute damping correction.
        let damp_axis = |channels: &mut SignalChannelTable, cw_ch: Option<signal::ChannelId>, ccw_ch: Option<signal::ChannelId>, angular_vel: f64| {
            let (Some(cw), Some(ccw)) = (cw_ch, ccw_ch) else { return };
            let cw_val = channels.read_value(cw).as_f32();
            let ccw_val = channels.read_value(ccw).as_f32();
            // Only apply damping when pilot is NOT actively commanding this axis.
            if cw_val.abs() > 0.01 || ccw_val.abs() > 0.01 { return; }
            if angular_vel.abs() <= dead_zone { return; }
            let correction = (angular_vel * gain).clamp(-max_corr, max_corr) as f32;
            if correction > 0.0 {
                channels.set_value_direct(cw, signal::SignalValue::Float(correction));
            } else {
                channels.set_value_direct(ccw, signal::SignalValue::Float(-correction));
            }
        };

        // Yaw (Y-axis angular velocity)
        damp_axis(&mut channels, fc.yaw_cw_channel, fc.yaw_ccw_channel, ang_vel.y);
        // Pitch (X-axis angular velocity)
        damp_axis(&mut channels, fc.pitch_up_channel, fc.pitch_down_channel, ang_vel.x);
        // Roll (Z-axis angular velocity)
        damp_axis(&mut channels, fc.roll_cw_channel, fc.roll_ccw_channel, ang_vel.z);
    }
}

/// Hover module block system: 6-DOF hover via signal channels.
///
/// Three layers:
/// 1. **Attitude hold**: PD controller → rotation channel corrections
/// 2. **Gravity compensation**: feedforward → thrust channel overrides
/// 3. **Velocity damping**: PD controller → thrust channel corrections
///
/// Runs AFTER flight_computer_system. Reads/writes channels via set_value_direct.
fn hover_module_system(
    mut channels: ResMut<SignalChannelTable>,
    mut query: Query<(Entity, &mut HoverModuleState)>,
    powered_by_query: Query<&PoweredBy>,
    block_index: Res<FunctionalBlockIndex>,
    reactors: Query<&ReactorState>,
    exterior: Res<ShipExterior>,
    ship_props: Res<ShipProps>,
    autopilot_query: Query<&AutopilotBlockState>,
) {
    for (entity, mut hm) in &mut query {
        // Power check.
        if block_power_ratio(entity, &powered_by_query, &block_index, &reactors) <= 0.0 { continue; }

        // Read activation channel.
        let activate_val = hm.activate_channel.map(|id| channels.read_value(id).as_f32()).unwrap_or(0.0);
        let cutoff_val = hm.cutoff_channel.map(|id| channels.read_value(id).as_f32()).unwrap_or(0.0);
        // Check if any autopilot is engaged (hover defers to autopilot).
        let autopilot_active = autopilot_query.iter().any(|ap| ap.target_body_id.is_some());
        let active = activate_val > 0.5 && exterior.in_atmosphere && cutoff_val < 0.5 && !autopilot_active;

        // Edge detect: capture heading on activation.
        if active && !hm.was_active {
            let g = exterior.gravity_acceleration.length();
            if g > 1e-6 {
                let up = -exterior.gravity_acceleration / g;
                let fwd_world = exterior.rotation * DVec3::NEG_Z;
                let fwd_horiz = fwd_world - up * fwd_world.dot(up);
                hm.captured_heading = if fwd_horiz.length() > 1e-6 {
                    fwd_horiz.normalize()
                } else {
                    let candidate = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
                    (candidate - up * candidate.dot(up)).normalize()
                };
            }
        }
        hm.was_active = active;
        if !active { continue; }

        let g = exterior.gravity_acceleration.length();
        if g < 1e-6 { continue; }

        // =====================================================================
        // Layer 1: Attitude hold
        // =====================================================================
        let up = -exterior.gravity_acceleration / g;
        let fwd_raw = hm.captured_heading;
        let right = fwd_raw.cross(up);
        let right = if right.length() > 1e-6 { right.normalize() } else {
            let c = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
            c.cross(up).normalize()
        };
        let fwd = up.cross(right);
        let target_rot = DQuat::from_mat3(&glam::DMat3::from_cols(right, up, -fwd)).normalize();
        let q_error = (target_rot * exterior.rotation.inverse()).normalize();
        let (axis, angle) = if q_error.w < 0.0 {
            let q = DQuat::from_xyzw(-q_error.x, -q_error.y, -q_error.z, -q_error.w);
            (DVec3::new(q.x, q.y, q.z), 2.0 * q.w.acos())
        } else {
            (DVec3::new(q_error.x, q_error.y, q_error.z), 2.0 * q_error.w.acos())
        };
        let error_world = if angle > 1e-6 { axis.normalize() * angle } else { DVec3::ZERO };
        let error_local = exterior.rotation.inverse() * error_world;
        let att_gain = 1.9_f64;

        // Write attitude corrections to rotation channels (adds to flight computer output).
        let write_rotation_pair = |channels: &mut SignalChannelTable, pos_ch: Option<signal::ChannelId>, neg_ch: Option<signal::ChannelId>, value: f64| {
            let clamped = value.clamp(-1.0, 1.0) as f32;
            if let Some(ch) = if clamped > 0.0 { pos_ch } else { neg_ch } {
                let current = channels.read_value(ch).as_f32();
                if current.abs() < 0.01 {
                    channels.set_value_direct(ch, signal::SignalValue::Float(clamped.abs()));
                }
            }
        };
        write_rotation_pair(&mut channels, hm.pitch_up_channel, hm.pitch_down_channel, error_local.x * att_gain);
        write_rotation_pair(&mut channels, hm.yaw_cw_channel, hm.yaw_ccw_channel, error_local.y * att_gain);
        // Roll: positive error_local.z → need CCW (roll_ccw)
        write_rotation_pair(&mut channels, hm.roll_ccw_channel, hm.roll_cw_channel, error_local.z * att_gain);

        // =====================================================================
        // Layer 2+3: Gravity compensation + velocity damping → thrust channels
        // =====================================================================
        let grav_local = exterior.rotation.inverse() * exterior.gravity_acceleration;
        let mass = ship_props.0.mass_kg;
        let tpa = &ship_props.0.thrust_per_axis;

        // Read thrust limiter from channel (if available).
        // tpa indices: 0=forward, 1=reverse, 2=up, 3=down, 4=right, 5=left
        let limiter = 0.75_f64.max(0.01); // TODO: read from thrust-limiter channel

        let effective = |axis: usize| -> f64 { tpa[axis] * limiter };

        // Feedforward: gravity compensation throttle.
        let grav_comp = |component: f64, pos_axis: usize, neg_axis: usize| -> f32 {
            if component < 0.0 && tpa[pos_axis] > 1.0 {
                ((-component) * mass / effective(pos_axis)) as f32
            } else if component > 0.0 && tpa[neg_axis] > 1.0 {
                -(component * mass / effective(neg_axis)) as f32
            } else { 0.0 }
        };

        let ff_y = grav_comp(grav_local.y, 3, 2);
        let ff_x = grav_comp(grav_local.x, 1, 0);
        let ff_z = grav_comp(grav_local.z, 5, 4);

        // PD velocity controller.
        let tau = 2.0_f64;
        let omega_n = std::f64::consts::TAU / (4.0 * tau);
        let kp = omega_n * omega_n;
        let kd = 2.0 * omega_n;
        let vel_local = exterior.rotation.inverse() * exterior.velocity;
        let dt = 0.05_f64;
        let accel_local = (vel_local - hm.prev_velocity_local) / dt;
        hm.prev_velocity_local = vel_local;

        let pd_force = |vel: f64, accel: f64| -> f64 { -(kp * vel + kd * accel) * mass };
        let to_throttle = |force: f64, pos_axis: usize, neg_axis: usize| -> f32 {
            if force > 0.0 && tpa[pos_axis] > 1.0 { (force / effective(pos_axis)) as f32 }
            else if force < 0.0 && tpa[neg_axis] > 1.0 { (force / effective(neg_axis)) as f32 }
            else { 0.0 }
        };

        let fb_y = to_throttle(pd_force(vel_local.y, accel_local.y), 3, 2);
        let fb_x = to_throttle(pd_force(vel_local.x, accel_local.x), 1, 0);
        let fb_z = to_throttle(pd_force(vel_local.z, accel_local.z), 5, 4);

        // Write combined thrust to channels (overrides seat's manual thrust).
        let write_thrust = |channels: &mut SignalChannelTable, pos_ch: Option<signal::ChannelId>, neg_ch: Option<signal::ChannelId>, total: f32| {
            let clamped = total.clamp(-1.0, 1.0);
            if clamped >= 0.0 {
                if let Some(ch) = pos_ch { channels.set_value_direct(ch, signal::SignalValue::Float(clamped)); }
                if let Some(ch) = neg_ch { channels.set_value_direct(ch, signal::SignalValue::Float(0.0)); }
            } else {
                if let Some(ch) = pos_ch { channels.set_value_direct(ch, signal::SignalValue::Float(0.0)); }
                if let Some(ch) = neg_ch { channels.set_value_direct(ch, signal::SignalValue::Float(-clamped)); }
            }
        };

        write_thrust(&mut channels, hm.thrust_up_channel, hm.thrust_down_channel, ff_y + fb_y);
        write_thrust(&mut channels, hm.thrust_right_channel, hm.thrust_left_channel, ff_x + fb_x);
        write_thrust(&mut channels, hm.thrust_forward_channel, hm.thrust_reverse_channel, ff_z + fb_z);
    }
}

/// Autopilot block system: target-tracking, publishes steering commands to rotation channels.
///
/// Reads engage channel for rising edge (0→1). On engage: finds best-aligned celestial body.
/// When target active: computes steering, writes rotation channels via set_value_direct.
fn autopilot_system(
    mut channels: ResMut<SignalChannelTable>,
    mut query: Query<(Entity, &mut AutopilotBlockState)>,
    powered_by_query: Query<&PoweredBy>,
    block_index: Res<FunctionalBlockIndex>,
    reactors: Query<&ReactorState>,
    exterior: Res<ShipExterior>,
    scene: Res<SceneCache>,
) {
    for (entity, mut ap) in &mut query {
        // Power check.
        if block_power_ratio(entity, &powered_by_query, &block_index, &reactors) <= 0.0 { continue; }
        // Read engage channel for rising-edge detection.
        let engage_val = ap.engage_channel.map(|id| channels.read_value(id).as_f32()).unwrap_or(0.0);
        let rising_edge = engage_val > 0.5 && ap.prev_engage_value <= 0.5;
        ap.prev_engage_value = engage_val;

        if rising_edge {
            if ap.target_body_id.is_some() {
                // Disengage.
                ap.pending_cmd = Some((0xFFFFFFFF, 0));
                ap.target_body_id = None;
                info!("autopilot disengage (channel edge)");
            } else {
                // Engage: find best-aligned celestial body.
                let ship_fwd = exterior.rotation * DVec3::NEG_Z;
                let mut best_body_id: Option<u32> = None;
                let mut best_dot = 0.9_f64;
                for body in &scene.bodies {
                    if body.body_id == 0 { continue; }
                    let to_body = (body.position - exterior.position).normalize_or_zero();
                    let d = ship_fwd.dot(to_body);
                    if d > best_dot {
                        best_dot = d;
                        best_body_id = Some(body.body_id);
                    }
                }
                if let Some(body_id) = best_body_id {
                    ap.target_body_id = Some(body_id);
                    ap.pending_cmd = Some((body_id, 0));
                    info!(body_id, "autopilot engage (channel edge)");
                }
            }
        }

        // Cancel autopilot on manual thrust input (any thrust channel > 0.01).
        if ap.target_body_id.is_some() {
            let has_manual = [
                ap.yaw_cw_channel, ap.yaw_ccw_channel,
                ap.pitch_up_channel, ap.pitch_down_channel,
                ap.roll_cw_channel, ap.roll_ccw_channel,
            ].iter().any(|ch| {
                ch.map(|id| channels.read_value(id).as_f32().abs() > 0.01).unwrap_or(false)
            });
            // Only cancel if pilot is actively commanding rotation (not from flight computer).
            // Flight computer values are typically < 0.3, while pilot input is ~1.0.
            let has_strong_manual = [
                ap.yaw_cw_channel, ap.yaw_ccw_channel,
                ap.pitch_up_channel, ap.pitch_down_channel,
            ].iter().any(|ch| {
                ch.map(|id| channels.read_value(id).as_f32().abs() > 0.5).unwrap_or(false)
            });
            if has_strong_manual {
                ap.target_body_id = None;
                ap.pending_cmd = Some((0xFFFFFFFF, 0));
                info!("autopilot cancelled by manual input");
            }
        }

        // When target is active, compute steering and write to rotation channels.
        // The actual orbital mechanics steering is handled by the system shard's autopilot.
        // Here we just maintain the target_body_id and send commands via pending_cmd.
        // The system shard sends back rotation commands via ShipPositionUpdate.
    }
}

/// Warp computer block system: target selection and warp initiation.
///
/// Reads target channel for rising edge → cycle star selection.
/// Reads confirm channel for rising edge → initiate warp via pending_cmd.
fn warp_computer_system(
    mut channels: ResMut<SignalChannelTable>,
    mut query: Query<(Entity, &mut WarpComputerState)>,
    powered_by_query: Query<&PoweredBy>,
    block_index: Res<FunctionalBlockIndex>,
    reactors: Query<&ReactorState>,
    exterior: Res<ShipExterior>,
    cached_sys: Res<CachedSystemParams>,
    cached_galaxy: Res<CachedGalaxyMap>,
) {
    for (entity, mut wc) in &mut query {
        // Power check.
        if block_power_ratio(entity, &powered_by_query, &block_index, &reactors) <= 0.0 { continue; }
        // Read cycle channel for rising edge (G key).
        let cycle_val = wc.cycle_channel.map(|id| channels.read_value(id).as_f32()).unwrap_or(0.0);
        let cycle_edge = cycle_val > 0.5 && wc.prev_cycle_value <= 0.5;
        wc.prev_cycle_value = cycle_val;

        // Read accept channel for rising edge (Enter key).
        let accept_val = wc.accept_channel.map(|id| channels.read_value(id).as_f32()).unwrap_or(0.0);
        let accept_edge = accept_val > 0.5 && wc.prev_accept_value <= 0.5;
        wc.prev_accept_value = accept_val;

        // Read cancel channel for rising edge (Backspace key).
        let cancel_val = wc.cancel_channel.map(|id| channels.read_value(id).as_f32()).unwrap_or(0.0);
        let cancel_edge = cancel_val > 0.5 && wc.prev_cancel_value <= 0.5;
        wc.prev_cancel_value = cancel_val;

        // Cancel: clear target selection.
        if cancel_edge && wc.target_star_index.is_some() {
            info!("warp target cancelled (channel edge)");
            wc.target_star_index = None;
        }

        if cycle_edge {
            // Cycle to next warp target star.
            let ship_fwd = exterior.rotation * DVec3::NEG_Z;
            if let Some(ref sys) = cached_sys.0 {
                if let Some(ref galaxy_map) = cached_galaxy.0 {
                    let current_star = galaxy_map.stars.iter().find(|s| s.system_seed == sys.system_seed);
                    if let Some(cur) = current_star {
                        let cur_pos = cur.position;
                        if let Some(current_target) = wc.target_star_index {
                            // Cycle to next-best aligned star.
                            let current_dot = galaxy_map.stars.iter()
                                .find(|s| s.index == current_target)
                                .map(|s| ship_fwd.dot((s.position - cur_pos).normalize()))
                                .unwrap_or(1.0);
                            let mut best: Option<(u32, f64)> = None;
                            for star in &galaxy_map.stars {
                                if star.index == cur.index || star.index == current_target { continue; }
                                let dot = ship_fwd.dot((star.position - cur_pos).normalize());
                                if dot < current_dot && dot > 0.3 {
                                    if dot > best.map(|b| b.1).unwrap_or(0.3) {
                                        best = Some((star.index, dot));
                                    }
                                }
                            }
                            if best.is_none() {
                                for star in &galaxy_map.stars {
                                    if star.index == cur.index { continue; }
                                    let dot = ship_fwd.dot((star.position - cur_pos).normalize());
                                    if dot > best.map(|b| b.1).unwrap_or(0.3) {
                                        best = Some((star.index, dot));
                                    }
                                }
                            }
                            if let Some((idx, _)) = best {
                                wc.target_star_index = Some(idx);
                                info!(target_star = idx, "warp target cycled (channel edge)");
                            }
                        } else {
                            // First target selection: pick best-aligned.
                            let mut best: Option<(u32, f64)> = None;
                            for star in &galaxy_map.stars {
                                if star.index == cur.index { continue; }
                                let alignment = ship_fwd.dot((star.position - cur_pos).normalize());
                                if alignment > best.map(|b| b.1).unwrap_or(0.3) {
                                    best = Some((star.index, alignment));
                                }
                            }
                            if let Some((target_index, _)) = best {
                                wc.target_star_index = Some(target_index);
                                info!(target_star = target_index, "warp target selected (channel edge)");
                            }
                        }
                    }
                }
            }
        }

        if accept_edge {
            if let Some(target) = wc.target_star_index {
                wc.pending_cmd = Some(target);
                info!(target_star = target, "warp accepted (channel edge)");
            }
        }
    }
}

/// Engine controller block system: master on/off toggle for all propulsion.
///
/// Rising edge on toggle channel flips engines_on state.
/// When engines are off, zeros ALL managed thrust and rotation channels.
fn engine_controller_system(
    mut channels: ResMut<SignalChannelTable>,
    mut query: Query<(Entity, &mut EngineControllerState)>,
    powered_by_query: Query<&PoweredBy>,
    block_index: Res<FunctionalBlockIndex>,
    reactors: Query<&ReactorState>,
) {
    for (entity, mut ec) in &mut query {
        // Power check — engine controller without power can't toggle anything.
        if block_power_ratio(entity, &powered_by_query, &block_index, &reactors) <= 0.0 { continue; }
        // Read toggle channel for rising edge.
        let toggle_val = ec.toggle_channel.map(|id| channels.read_value(id).as_f32()).unwrap_or(0.0);
        let rising_edge = toggle_val > 0.5 && ec.prev_toggle_value <= 0.5;
        ec.prev_toggle_value = toggle_val;

        if rising_edge {
            ec.engines_on = !ec.engines_on;
            info!(engines_on = ec.engines_on, "engine controller toggled");
        }

        // When engines are off, zero ALL managed channels.
        if !ec.engines_on {
            let zero = signal::SignalValue::Float(0.0);
            for ch in [
                ec.thrust_forward_channel, ec.thrust_reverse_channel,
                ec.thrust_right_channel, ec.thrust_left_channel,
                ec.thrust_up_channel, ec.thrust_down_channel,
                ec.yaw_cw_channel, ec.yaw_ccw_channel,
                ec.pitch_up_channel, ec.pitch_down_channel,
                ec.roll_cw_channel, ec.roll_ccw_channel,
            ] {
                if let Some(id) = ch {
                    channels.set_value_direct(id, zero);
                }
            }
        }
    }
}

/// Phase 1: All functional blocks with SignalPublisher write their state to channels.
/// Values accumulate in pending_values, merged after all publishers write.
fn signal_publish(
    mut channels: ResMut<SignalChannelTable>,
    publishers: Query<(&FunctionalBlockRef, &SignalPublisher, Option<&ReactorState>, Option<&CruiseDriveState>, Option<&MechanicalState>)>,
    mut incoming: ResMut<IncomingSignalBuffer>,
    seated_players: Query<(&SeatedState, &SeatInputValues), With<Player>>,
    seat_query: Query<&SeatChannelMapping>,
) {
    channels.clear_pending();

    // Cross-shard signals received via QUIC.
    for (name, value) in incoming.signals.drain(..) {
        channels.push_pending(&name, value);
    }

    // Per-player seat publishing: each seated player publishes to their seat's channels.
    for (seated_state, seat_input) in &seated_players {
        if !seated_state.seated {
            continue;
        }
        let Some(seat_entity) = seated_state.seat_entity else {
            continue;
        };
        let Ok(mapping) = seat_query.get(seat_entity) else {
            continue;
        };

        // Publish each binding's value from the client's seat input evaluation.
        let values = &seat_input.0;
        for (i, binding) in mapping.bindings.iter().enumerate() {
            let value = values.get(i).copied().unwrap_or(0.0);
            channels.push_pending_id(binding.channel_id, signal::SignalValue::Float(value));
        }
        // Publish seated occupancy signal.
        if let Some(ch) = mapping.seated_channel_id {
            channels.push_pending_id(ch, signal::SignalValue::Float(1.0));
        }
    }

    // Block publishers (reactors, sensors, cruise drives, mechanical mounts, etc.).
    for (_block_ref, publisher, reactor_state, cruise_state, mech_state) in &publishers {
        for binding in &publisher.bindings {
            let value = match binding.property {
                SignalProperty::Active => signal::SignalValue::Bool(true),
                SignalProperty::Throttle => signal::SignalValue::Float(0.0),
                SignalProperty::Level => {
                    // Reactors publish their actual output; others default to 1.0.
                    match reactor_state {
                        Some(rs) => signal::SignalValue::Float(if rs.active { rs.throttle } else { 0.0 }),
                        None => signal::SignalValue::Float(1.0),
                    }
                }
                SignalProperty::Boost => {
                    // Cruise drives publish boost multiplier when active, 1.0 when off.
                    match cruise_state {
                        Some(cs) => signal::SignalValue::Float(
                            if cs.throttle > 0.5 { cs.boost_multiplier as f32 } else { 1.0 }
                        ),
                        None => signal::SignalValue::Float(1.0),
                    }
                }
                SignalProperty::Angle | SignalProperty::Extension => {
                    // Mechanical mounts publish their current position as feedback.
                    match mech_state {
                        Some(ms) => signal::SignalValue::Float(ms.current),
                        None => signal::SignalValue::Float(0.0),
                    }
                }
                SignalProperty::Status => {
                    // Mechanical mounts publish operational status.
                    match mech_state {
                        Some(ms) => signal::SignalValue::Float(ms.status as u8 as f32),
                        None => signal::SignalValue::Float(0.0),
                    }
                }
                SignalProperty::Pressure => signal::SignalValue::Float(101.3),
                _ => signal::SignalValue::Float(0.0),
            };
            channels.push_pending_id(binding.channel_id, value);
        }
    }

    channels.merge_pending();
}

/// Phase 2: Signal Converters with dirty inputs evaluate their condition→action rules.
/// Lazy evaluation: only processes converters whose input channels changed this tick.
fn signal_evaluate(
    mut channels: ResMut<SignalChannelTable>,
    converters: Query<&SignalConverterConfig>,
) {
    for config in &converters {
        for rule in &config.rules {
            let (input_value, is_dirty) = match channels.get_by_id(rule.input_channel_id) {
                Some(ch) => (ch.value, ch.dirty),
                None => continue,
            };

            // Skip if input channel not dirty (lazy evaluation for 100K scale).
            if !is_dirty && !matches!(rule.condition, signal::SignalCondition::Always) {
                continue;
            }

            if rule.condition.evaluate(input_value, is_dirty) {
                let output = rule.expression.compute(input_value);
                channels.publish_direct_id(rule.output_channel_id, output);
            }
        }
    }
}

/// Threshold above which par_iter_mut outperforms sequential iteration.
/// Below this, thread pool dispatch overhead exceeds per-entity work.
/// Tuned for O(1) channel lookups (Vec index) + f32 assignment per entity.
const PAR_SUBSCRIBE_THRESHOLD: usize = 256;

/// Phase 3: Functional blocks with SignalSubscriber read channels and act.
/// Adaptive parallelism: uses par_iter_mut for large entity counts,
/// sequential iteration for small ships where dispatch overhead dominates.
fn signal_subscribe(
    channels: Res<SignalChannelTable>,
    mut subscribers: Query<(
        &SignalSubscriber,
        Option<&mut ThrusterState>,
        Option<&mut CruiseDriveState>,
        Option<&mut MechanicalState>,
    )>,
) {
    let apply = |(subscriber, mut thruster_state, mut cruise_state, mut mech_state): (
        &SignalSubscriber,
        Option<Mut<ThrusterState>>,
        Option<Mut<CruiseDriveState>>,
        Option<Mut<MechanicalState>>,
    )| {
        // Reset boost each tick — only active while boost signal is published.
        if let Some(ref mut ts) = thruster_state {
            ts.boost = 1.0;
        }
        // Reset velocity override each tick — only active while Throttle signal is published.
        if let Some(ref mut ms) = mech_state {
            ms.velocity_override = 0.0;
        }
        for binding in &subscriber.bindings {
            if let Some(ch) = channels.get_by_id(binding.channel_id) {
                if let Some(ref mut ts) = thruster_state {
                    match binding.property {
                        SignalProperty::Throttle => {
                            ts.throttle = ch.value.as_f32();
                        }
                        SignalProperty::Boost => {
                            ts.boost = ch.value.as_f32().max(1.0);
                        }
                        _ => {}
                    }
                }
                // CruiseDrive reads throttle from "cruise" channel.
                if binding.property == SignalProperty::Throttle {
                    if let Some(ref mut cs) = cruise_state {
                        cs.throttle = ch.value.as_f32();
                    }
                }
                // MechanicalState reads Angle, Extension, Throttle, Speed.
                // - Angle/Extension: absolute position target (degrees or meters).
                // - Throttle: velocity fraction [-1,1] — drives motor speed directly.
                if let Some(ref mut ms) = mech_state {
                    match binding.property {
                        SignalProperty::Angle | SignalProperty::Extension => {
                            ms.target = ch.value.as_f32();
                        }
                        SignalProperty::Throttle => {
                            let rate = ch.value.as_f32();
                            ms.target += rate * ms.max_speed * 0.05;
                            // Clamp target to stay within reachable distance of current.
                            // This prevents target from racing ahead when input is fast,
                            // keeping the motor responsive instead of chasing an unreachable goal.
                            let max_lead = ms.max_speed * 0.15; // ~1.5 ticks ahead max
                            ms.target = ms.target.clamp(
                                ms.current - max_lead,
                                ms.current + max_lead,
                            );
                            // Clamp to joint limits if not full rotation.
                            if ms.max_range < 360.0 {
                                let half = ms.max_range as f32 / 2.0;
                                ms.target = ms.target.clamp(-half, half);
                            }
                        }
                        SignalProperty::Speed => {
                            ms.max_speed = ch.value.as_f32();
                        }
                        SignalProperty::Active => {
                            // Active=false locks the mechanism at its current position.
                            ms.locked = !ch.value.as_bool();
                        }
                        _ => {}
                    }
                }
            }
        }
    };

    if subscribers.iter().len() >= PAR_SUBSCRIBE_THRESHOLD {
        subscribers.par_iter_mut().for_each(apply);
    } else {
        subscribers.iter_mut().for_each(apply);
    }
}

/// Compute ship thrust/torque from per-thruster signal channel values.
/// Each thruster reads its throttle from its subscribed channel and contributes
/// thrust_n × throttle × direction. Torque is computed from offset to center of mass.
fn compute_ship_thrust(
    mut pilot_acc: ResMut<PilotAccumulator>,
    thrusters: Query<(&ThrusterBlock, &ThrusterState, &PoweredBy)>,
    reactors: Query<&ReactorState>,
    block_index: Res<FunctionalBlockIndex>,
    channels: Res<SignalChannelTable>,
    ship_props: Res<ShipProps>,
    ship_com: Res<ShipCenterOfMass>,
    exterior: Res<ShipExterior>,
    scene: Res<SceneCache>,
    cached_sys: Res<CachedSystemParams>,
) {
    // Read thrust limiter from channel (seat publishes it).
    let limiter_id = channels.resolve(seat_presets::CH_THRUST_LIMITER);
    let limiter = limiter_id.map(|id| channels.read_value(id).as_f32() as f64).unwrap_or(0.75);

    let mut total_thrust = DVec3::ZERO;
    let mut total_torque = DVec3::ZERO;
    let com = ship_com.0;

    for (thruster, state, powered_by) in &thrusters {
        if state.throttle.abs() < 0.001 { continue; }

        // Wireless power: look up reactor by block position, find circuit power_ratio.
        let power_ratio = block_index.0.get(&powered_by.reactor_pos)
            .and_then(|&e| reactors.get(e).ok())
            .and_then(|rs| rs.circuits.iter().find(|c| c.name == powered_by.circuit))
            .map(|c| c.power_ratio as f64)
            .unwrap_or(0.0);

        let thrust_vec = -thruster.direction * thruster.thrust_n
            * state.throttle as f64 * limiter * power_ratio * state.boost as f64;
        total_thrust += thrust_vec;

        let block_center = DVec3::new(
            thruster.block_pos.x as f64 + 0.5,
            thruster.block_pos.y as f64 + 0.5,
            thruster.block_pos.z as f64 + 0.5,
        );
        total_torque += (block_center - com).cross(thrust_vec);
    }


    pilot_acc.thrust = total_thrust;
    pilot_acc.torque = total_torque;

    // Capture rotation intent from signal channels as [-1, 1] rate command.
    // After all system blocks have processed (FC, hover, autopilot), the channel values
    // represent the final desired rotation. Read them to build the rate command.
    // Rotation command requires at least one active reactor — no power = no rotation.
    let any_reactor_active = reactors.iter().any(|rs| rs.active);
    let read_ch = |name: &str| -> f32 {
        if !any_reactor_active { return 0.0; }
        channels.resolve(name).map(|id| channels.read_value(id).as_f32()).unwrap_or(0.0)
    };
    let pitch_up = read_ch(seat_presets::CH_TORQUE_PITCH_UP);
    let pitch_down = read_ch(seat_presets::CH_TORQUE_PITCH_DOWN);
    let yaw_cw = read_ch(seat_presets::CH_TORQUE_YAW_CW);
    let yaw_ccw = read_ch(seat_presets::CH_TORQUE_YAW_CCW);
    let roll_cw = read_ch(seat_presets::CH_TORQUE_ROLL_CW);
    let roll_ccw = read_ch(seat_presets::CH_TORQUE_ROLL_CCW);
    pilot_acc.rotation_command = DVec3::new(
        (pitch_up - pitch_down) as f64,   // X = pitch
        (yaw_cw - yaw_ccw) as f64,        // Y = yaw
        (roll_ccw - roll_cw) as f64,       // Z = roll
    );

    // Gravity compensation (hover + damping) is computed in the system-shard's
    // physics_integrate, where position/rotation/gravity are all current-tick
    // authoritative. This avoids rotation-mismatch and stale-data issues that
    // caused atmosphere catapulting when computed here.
}

/// Stub: with kinematic bodies, current = target (set in apply_mechanical_transforms).
/// Kept as a system slot for the schedule — does nothing.
fn read_mechanical_state(
    _mechanicals: Query<&MechanicalState>,
) {
}

/// Apply sub-grid transforms directly to kinematic bodies.
/// No motors, no forces, no solver — just set the body rotation/position each tick.
///
/// Computes all world transforms in a local HashMap (no Rapier reads), then writes
/// them to kinematic bodies in one pass. This ensures parent transforms are always
/// available when composing children, regardless of Rapier's internal update timing.
fn apply_mechanical_transforms(
    mut rapier: ResMut<RapierContext>,
    mut mechanicals: Query<(&mut MechanicalState, &voxeldust_core::ecs::components::FunctionalBlockRef)>,
    sub_grids: Res<SubGridRegistry>,
    mut world_isos: ResMut<MechanicalWorldIsometries>,
) {
    use voxeldust_core::ecs::components::SubGridId;

    struct SgWork {
        sg_id: SubGridId,
        target: f32,
        joint_type: block::JointType,
        mount_pos: glam::IVec3,
        mount_face: u8,
        parent_grid: SubGridId,
        body_handle: rapier3d::dynamics::RigidBodyHandle,
    }

    let mut work: Vec<SgWork> = Vec::new();
    for (mut state, _block_ref) in &mut mechanicals {
        let sg = match sub_grids.grids.get(&state.child_grid_id) {
            Some(sg) => sg,
            None => continue,
        };
        // When locked, hold current position (don't update target → current).
        if !state.locked {
            state.current = state.target;
        }
        work.push(SgWork {
            sg_id: state.child_grid_id,
            target: state.current, // Use current (which equals target unless locked)
            joint_type: state.joint_type,
            mount_pos: sg.mount_pos,
            mount_face: sg.mount_face,
            parent_grid: sg.parent_grid,
            body_handle: sg.body_handle,
        });
    }

    // Phase 1: Compute all world isometries in a local map (no Rapier dependency).
    // Topological order: multi-pass, parents before children.
    let mut world_isometries: std::collections::HashMap<SubGridId, rapier3d::math::Isometry<f32>> =
        std::collections::HashMap::new();

    for _depth in 0..8 {
        let mut any_progress = false;
        for w in &work {
            if world_isometries.contains_key(&w.sg_id) { continue; }
            if w.parent_grid != SubGridId::ROOT && !world_isometries.contains_key(&w.parent_grid) {
                continue; // parent not computed yet
            }

            let axis_offset = block::sub_block::face_to_offset(w.mount_face);
            let axis_f = rapier3d::math::Vector::new(
                axis_offset.x as f32, axis_offset.y as f32, axis_offset.z as f32,
            );

            // Anchor in root-space.
            let anchor_root = rapier3d::math::Vector::new(
                w.mount_pos.x as f32 + 0.5 + axis_f.x * 0.5,
                w.mount_pos.y as f32 + 0.5 + axis_f.y * 0.5,
                w.mount_pos.z as f32 + 0.5 + axis_f.z * 0.5,
            );

            // Own local rotation + position from target angle.
            let (own_rot, own_pos) = match w.joint_type {
                block::JointType::Revolute => {
                    let unit_axis = rapier3d::math::UnitVector::new_normalize(axis_f);
                    let rot = rapier3d::math::Rotation::new(*unit_axis * w.target.to_radians());
                    (rot, anchor_root)
                }
                block::JointType::Prismatic => {
                    let offset = axis_f * w.target;
                    (rapier3d::math::Rotation::identity(), anchor_root + offset)
                }
            };

            let world_iso = if w.parent_grid == SubGridId::ROOT {
                // Root-level: own transform is the world transform.
                rapier3d::math::Isometry::from_parts(own_pos.into(), own_rot)
            } else {
                // Nested: compose with parent's already-computed world transform.
                let parent_iso = world_isometries[&w.parent_grid];
                let parent_anchor_offset = block::sub_block::face_to_offset(
                    sub_grids.grids[&w.parent_grid].mount_face,
                );
                let parent_anchor = rapier3d::math::Vector::new(
                    sub_grids.grids[&w.parent_grid].mount_pos.x as f32 + 0.5 + parent_anchor_offset.x as f32 * 0.5,
                    sub_grids.grids[&w.parent_grid].mount_pos.y as f32 + 0.5 + parent_anchor_offset.y as f32 * 0.5,
                    sub_grids.grids[&w.parent_grid].mount_pos.z as f32 + 0.5 + parent_anchor_offset.z as f32 * 0.5,
                );

                // Transform child anchor through parent: rotate (anchor - parent_anchor)
                // by parent rotation, then add parent world position.
                let child_relative = anchor_root - parent_anchor;
                let child_world_pos = parent_iso.translation.vector + parent_iso.rotation * child_relative;
                let child_world_rot = parent_iso.rotation * own_rot;

                rapier3d::math::Isometry::from_parts(child_world_pos.into(), child_world_rot)
            };

            world_isometries.insert(w.sg_id, world_iso);
            any_progress = true;
        }
        if !any_progress { break; }
    }

    // Phase 2: Write all computed isometries to Rapier kinematic bodies in one pass.
    for w in &work {
        if let Some(&iso) = world_isometries.get(&w.sg_id) {
            if let Some(body) = rapier.rigid_body_set.get_mut(w.body_handle) {
                body.set_next_kinematic_position(iso);
            }
        }
    }

    // Phase 3: Update mechanism arm colliders (piston rods resize, rotor axles stay fixed).
    for w in &work {
        let Some(sg) = sub_grids.grids.get(&w.sg_id) else { continue };
        let Some(arm_handle) = sg.arm_collider else { continue };
        let Some(collider) = rapier.collider_set.get_mut(arm_handle) else { continue };

        if w.joint_type == block::JointType::Prismatic {
            // Piston arm: resize to current extension.
            let arm_radius = 0.08_f32;
            let half_len = (w.target / 2.0).max(0.001);
            collider.set_shape(rapier3d::geometry::SharedShape::cuboid(arm_radius, arm_radius, half_len));
            let axis_offset = block::sub_block::face_to_offset(w.mount_face);
            let axis_f = rapier3d::math::Vector::new(
                axis_offset.x as f32, axis_offset.y as f32, axis_offset.z as f32,
            );
            // Arm starts at the face surface (block center + 0.5 along normal).
            let mount_face_surface = rapier3d::math::Vector::new(
                w.mount_pos.x as f32 + 0.5 + axis_f.x * 0.5,
                w.mount_pos.y as f32 + 0.5 + axis_f.y * 0.5,
                w.mount_pos.z as f32 + 0.5 + axis_f.z * 0.5,
            );
            let arm_center = mount_face_surface + axis_f * half_len;
            collider.set_translation(arm_center);
        }
        // Revolute (rotor axle): fixed size, no update needed.
    }

    // Store computed isometries for use by other systems (raycast, broadcast).
    world_isos.0 = world_isometries;
}

/// Phase 4: Clear dirty flags after all processing.
fn signal_clear_dirty(mut channels: ResMut<SignalChannelTable>) {
    channels.clear_dirty();
}

/// Broadcast non-Local dirty signals to other shards via QUIC.
/// Batches all dirty signals into one message per destination per tick,
/// reducing QUIC sends from (signals × peers) to just (peers).
fn signal_broadcast_remote(
    channels: Res<SignalChannelTable>,
    exterior: Res<ShipExterior>,
    config: Res<ShipConfig>,
    bridge: Res<NetworkBridge>,
    peer_positions: Res<PeerPositionCache>,
) {
    use voxeldust_core::shard_message::{
        ShardMsg, SignalBroadcastBatchData, SignalBroadcastEntry,
    };

    let remote_signals = channels.drain_remote_dirty();
    if remote_signals.is_empty() {
        return;
    }

    // Encode each dirty signal into a broadcast entry and partition by destination.
    let mut peer_entries = Vec::new();   // ShortRange → all peers
    let mut host_entries = Vec::new();   // LongRange + Radio → system shard

    for (channel_name, value, scope) in remote_signals {
        let (scope_code, range_m, frequency) = match scope {
            signal::SignalScope::ShortRange { range_m } => (1u8, range_m, 0u32),
            signal::SignalScope::LongRange => (2u8, 0.0, 0u32),
            signal::SignalScope::Radio { frequency } => (3u8, 0.0, frequency),
            signal::SignalScope::Local => continue,
        };

        let (value_type, value_data) = match value {
            signal::SignalValue::Bool(b) => (0u8, if b { 1.0f32 } else { 0.0 }),
            signal::SignalValue::Float(f) => (1u8, f),
            signal::SignalValue::State(s) => (2u8, s as f32),
        };

        let entry = SignalBroadcastEntry {
            channel_name,
            value_type,
            value_data,
            scope: scope_code,
            range_m,
            frequency,
        };

        match scope_code {
            1 => peer_entries.push(entry),       // ShortRange
            _ => host_entries.push(entry),       // LongRange (2) + Radio (3)
        }
    }

    let source_shard_id = config.shard_id.0;
    let source_position = exterior.position;

    // Send ShortRange batch to peer shards within range (spatial filtering).
    // Peers without a known position are included conservatively — the receiver
    // still does its own distance check as a safety net.
    if !peer_entries.is_empty() {
        // Maximum range across all ShortRange entries in the batch.
        let max_range = peer_entries.iter()
            .map(|e| e.range_m)
            .fold(0.0f64, f64::max);

        let batch_msg = ShardMsg::SignalBroadcastBatch(SignalBroadcastBatchData {
            source_shard_id,
            source_position,
            entries: peer_entries,
        });
        if let Ok(reg) = bridge.peer_registry.try_read() {
            for peer in reg.all() {
                if peer.id == config.shard_id {
                    continue;
                }
                // Spatial filter: skip peers that are definitely out of range.
                if let Some(&peer_pos) = peer_positions.positions.get(&peer.id) {
                    if (peer_pos - source_position).length() > max_range {
                        continue;
                    }
                }
                // Peer without known position → include conservatively.
                if let Some(addr) = reg.quic_addr(peer.id) {
                    let _ = bridge.quic_send_tx.try_send((peer.id, addr, batch_msg.clone()));
                }
            }
        }
    }

    // Send LongRange + Radio batch to host (system) shard.
    if !host_entries.is_empty() {
        if let Some(host_id) = config.host_shard_id {
            let batch_msg = ShardMsg::SignalBroadcastBatch(SignalBroadcastBatchData {
                source_shard_id,
                source_position,
                entries: host_entries,
            });
            if let Ok(reg) = bridge.peer_registry.try_read() {
                if let Some(addr) = reg.quic_addr(host_id) {
                    let _ = bridge.quic_send_tx.try_send((host_id, addr, batch_msg));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// App construction
// ---------------------------------------------------------------------------

/// Build a Rapier compound collider from all solid blocks in a ShipGrid chunk.
///
/// Each solid block becomes a unit cube (half-extent 0.5) positioned at
/// the block's center in ship-local coordinates.
fn build_chunk_collider(
    grid: &ShipGrid,
    chunk_key: glam::IVec3,
    registry: &BlockRegistry,
) -> Option<Collider> {
    let shapes = grid.chunk_collider_shapes(chunk_key, glam::Vec3::ZERO, registry);
    if shapes.is_empty() {
        return None;
    }

    let compound_shapes: Vec<(Isometry<f32>, SharedShape)> = shapes
        .iter()
        .map(|&(pos, he)| {
            let iso = Isometry::translation(pos.x, pos.y, pos.z);
            let shape = SharedShape::cuboid(he.x, he.y, he.z);
            (iso, shape)
        })
        .collect();

    Some(ColliderBuilder::compound(compound_shapes).build())
}

/// Scan the ShipGrid for all interactable blocks (cockpits, doors, etc.).
///
/// Called at startup and whenever blocks change. Returns the full set of
/// interaction points — supporting any number of cockpits and doors.
fn build_ship_interior(
    shard_id: ShardId,
    ship_id: u64,
    host_shard_id: Option<ShardId>,
    system_seed: u64,
    galaxy_seed: u64,
) -> App {
    let mut rigid_body_set = RigidBodySet::new();
    let mut collider_set = ColliderSet::new();

    // Root body: Fixed rigid body for the ship's static structure.
    // All non-sub-grid chunk colliders are parented to this body.
    // Required for mechanical joints (rotors, pistons) to have an anchor.
    let root_body = rapier3d::dynamics::RigidBodyBuilder::fixed()
        .translation(rapier3d::math::Vector::zeros())
        .build();
    let root_body_handle = rigid_body_set.insert(root_body);

    // Initialize block persistence and load ship data.
    let db_path = format!("/tmp/voxeldust-ship-{ship_id}.redb");
    let db = redb::Database::create(&db_path)
        .unwrap_or_else(|e| panic!("failed to create redb at {db_path}: {e}"));
    let ship_persistence = ShipPersistence {
        db,
        pending_saves: std::collections::HashSet::new(),
        last_save_tick: 0,
    };

    // Try to load saved ship data. Fall back to starter ship if no saved data.
    let registry = BlockRegistry::new();
    let loaded = load_grid_from_db(&ship_persistence.db);
    let grid = match loaded {
        Some(mut saved_grid) => {
            // Load channel overrides and power configs from the block_configs table.
            load_block_configs(&ship_persistence.db, &mut saved_grid);
            info!(
                blocks = saved_grid.total_block_count(),
                chunks = saved_grid.chunk_count(),
                "loaded ship from persistence"
            );
            saved_grid
        }
        None => {
            let layout = StarterShipLayout::default_starter();
            let starter = build_starter_ship(&layout);
            // Save the starter ship to persistence so it's there on next restart.
            let chunk_keys: Vec<glam::IVec3> = starter.chunk_keys().collect();
            {
                let txn = ship_persistence.db.begin_write().expect("redb write");
                {
                    let mut table = txn.open_table(SHIP_CHUNKS_TABLE).expect("redb table");
                    for &chunk_key in &chunk_keys {
                        if let Some(chunk) = starter.get_chunk(chunk_key) {
                            let key_bytes = chunk_key_bytes(chunk_key);
                            let data = block::serialize_chunk(chunk);
                            let _ = table.insert(key_bytes.as_slice(), data.as_slice());
                        }
                    }
                }
                let _ = txn.commit();
            }
            // Persist the starter ship's initial power configs and channel overrides.
            save_block_configs(&ship_persistence.db, &starter);
            info!("built starter ship and saved to persistence");
            starter
        }
    };

    // Generate Rapier colliders from each chunk's solid blocks, parented to root body.
    let mut chunk_collider_handles = std::collections::HashMap::new();
    for chunk_key in grid.chunk_keys() {
        if let Some(collider) = build_chunk_collider(&grid, chunk_key, &registry) {
            let handle = collider_set.insert_with_parent(
                collider, root_body_handle, &mut rigid_body_set,
            );
            chunk_collider_handles.insert(chunk_key, handle);
        }
    }

    // Find the first cockpit/seat block for player spawn position.
    // Scan the grid for any COCKPIT block as the spawn anchor.
    let mut spawn_pos = Vec3::new(0.0, 1.5, 0.0); // fallback
    {
        let cs = block::CHUNK_SIZE as i32;
        'outer: for (chunk_key, chunk) in grid.iter_chunks() {
            if chunk.is_empty() { continue; }
            let co = glam::IVec3::new(chunk_key.x * cs, chunk_key.y * cs, chunk_key.z * cs);
            for x in 0..block::CHUNK_SIZE as u8 {
                for y in 0..block::CHUNK_SIZE as u8 {
                    for z in 0..block::CHUNK_SIZE as u8 {
                        if chunk.get_block(x, y, z) == BlockId::COCKPIT {
                            let wp = co + glam::IVec3::new(x as i32, y as i32, z as i32);
                            spawn_pos = Vec3::new(
                                wp.x as f32 + 0.5,
                                wp.y as f32 + 1.0,
                                wp.z as f32 + 1.5,
                            );
                            break 'outer;
                        }
                    }
                }
            }
        }
    }
    // Player bodies are now spawned on-demand in process_connects, not at startup.
    // spawn_pos is stored as DefaultSpawnPosition resource below (after App creation).

    // Initialize ship position and scene from system seed.
    let mut ship_position = DVec3::ZERO;
    let mut scene_bodies = Vec::new();
    let mut scene_lighting = None;
    let mut cached_system_params = None;

    if system_seed > 0 {
        let sys = SystemParams::from_seed(system_seed);
        let planet_pos = system::compute_planet_position(&sys.planets[0], 0.0);
        ship_position = planet_pos + DVec3::new(sys.scale.spawn_offset, 0.0, 0.0);

        scene_bodies.push(CelestialBodySnapshotData {
            body_id: 0,
            position: DVec3::ZERO,
            radius: sys.star.radius_m,
            color: sys.star.color,
        });
        for (i, planet) in sys.planets.iter().enumerate() {
            let pos = system::compute_planet_position(planet, 0.0);
            scene_bodies.push(CelestialBodySnapshotData {
                body_id: (i + 1) as u32,
                position: pos,
                radius: planet.radius_m,
                color: planet.color,
            });
        }
        let l = system::compute_lighting(ship_position, &sys.star);
        scene_lighting = Some(LightingInfoData {
            sun_direction: l.sun_direction,
            sun_color: l.sun_color,
            sun_intensity: l.sun_intensity,
            ambient: l.ambient,
        });
        cached_system_params = Some(sys);
    }

    let mut app = App::new();

    // Resources.
    app.insert_resource(RapierContext {
        rigid_body_set,
        collider_set,
        integration_params: {
            let mut p = IntegrationParameters::default();
            p.dt = 0.05;
            p
        },
        physics_pipeline: PhysicsPipeline::new(),
        island_manager: IslandManager::new(),
        broad_phase: DefaultBroadPhase::new(),
        narrow_phase: NarrowPhase::new(),
        impulse_joint_set: ImpulseJointSet::new(),
        multibody_joint_set: MultibodyJointSet::new(),
        ccd_solver: CCDSolver::new(),
        query_pipeline: QueryPipeline::new(),
    });
    app.insert_resource(GravitySources(vec![GravitySource::default_floor_plates()]));
    app.insert_resource(RootBodyHandle(root_body_handle));
    app.insert_resource(SubGridRegistry::default());
    app.insert_resource(MechanicalWorldIsometries::default());
    app.insert_resource(MechanicalDirtyGrids::default());
    // PlayerPlatform is now a per-entity PlayerPlatformState component.
    app.insert_resource(ShipConfig {
        shard_id,
        ship_id,
        host_shard_id,
        system_seed,
        galaxy_seed,
    });
    app.insert_resource(ShipExterior {
        position: ship_position,
        velocity: DVec3::ZERO,
        rotation: DQuat::IDENTITY,
        angular_velocity: DVec3::ZERO,
        in_atmosphere: false,
        atmosphere_planet_index: -1,
        gravity_acceleration: DVec3::ZERO,
        atmosphere_density: 0.0,
    });
    app.insert_resource(SceneCache {
        bodies: scene_bodies,
        lighting: scene_lighting,
        game_time: 0.0,
        last_update_tick: 0,
        host_switch_tick: 0,
        authorized_peers: AuthorizedPeers::default(),
    });
    app.insert_resource(CachedSystemParams(cached_system_params));
    app.insert_resource(CachedGalaxyMap(
        if galaxy_seed != 0 {
            Some(voxeldust_core::galaxy::GalaxyMap::generate(galaxy_seed))
        } else {
            None
        },
    ));
    // Compute tight hull bounding box for exit detection.
    let hull_bounds = match grid.bounding_box() {
        Some((min, max)) => ShipHullBounds { min, max, margin: 2.0 },
        None => ShipHullBounds {
            min: glam::IVec3::ZERO,
            max: glam::IVec3::ZERO,
            margin: 2.0,
        },
    };

    app.insert_resource(ShipGridResource(grid));
    app.insert_resource(BlockRegistryResource(registry));
    app.insert_resource(hull_bounds);
    app.insert_resource(ChunkColliderHandles(chunk_collider_handles));
    app.insert_resource(PendingEntityOps::default());
    app.insert_resource(FunctionalBlockIndex::default());
    app.insert_resource(ShipProps(ShipPhysicalProperties::starter_ship()));
    app.insert_resource(PilotAccumulator::default());
    // SeatInputValues, ActiveSeatEntity are now per-entity Components.
    app.insert_resource(ShipCenterOfMass::default());
    app.insert_resource(AutopilotSnapshotCache::default());
    // ConnectedPlayer removed — multi-player uses per-entity Player components.
    app.insert_resource(PlayerEntityIndex::default());
    app.insert_resource(PendingPlayerHandoffs::default());
    app.insert_resource(VisibleShipRegistry::default());
    app.insert_resource(ExternalEntities::default());
    app.insert_resource(AtmosphereState::default());
    app.insert_resource(LandingState::default());
    app.insert_resource(PendingMessages::default());
    app.insert_resource(DefaultSpawnPosition(spawn_pos));
    // PlayerBodyHandle, PlayerPosition, PlayerYaw, SeatedState, InputActions
    // are now per-entity Components — spawned on-demand in process_connects.
    app.insert_resource(GravityEnabled(true));
    app.insert_resource(ecs::TickCounter::default());
    app.insert_resource(BlockEditQueue::default());
    app.insert_resource(AggregationDirty(false));
    app.insert_resource(ColliderSyncDirty(true)); // true = initial sync on first tick
    app.insert_resource(ColliderSyncSeqs::default());
    app.insert_resource(SignalChannelTable::new());
    app.insert_resource(IncomingSignalBuffer::default());
    app.insert_resource(PeerPositionCache::default());
    app.insert_resource(ship_persistence);

    // Messages.
    app.add_message::<ClientConnectedMsg>();
    app.add_message::<PlayerInputMsg>();
    app.add_message::<BlockEditMsg>();
    app.add_message::<ConfigUpdateMsg>();
    app.add_message::<SubBlockEditMsg>();
    app.add_message::<InboundHandoffMsg>();
    app.add_message::<HandoffAcceptedMsg>();
    app.add_message::<HostSwitchMsg>();

    // System ordering.
    app.configure_sets(
        Update,
        (
            ShipSet::Bridge,
            ShipSet::Input,
            ShipSet::BlockEdit,
            ShipSet::Signal,
            ShipSet::Physics,
            ShipSet::Interaction,
            ShipSet::Send,
            ShipSet::Broadcast,
            ShipSet::Diagnostics,
        )
            .chain(),
    );

    // Bridge: drain async channels.
    app.add_systems(
        Update,
        (drain_connects, drain_input, drain_block_edits, drain_config_updates, drain_sub_block_edits, drain_quic).in_set(ShipSet::Bridge),
    );

    // Input: process connects, player input, preconnect, config updates.
    app.add_systems(
        Update,
        (process_connects, process_input, preconnect_check, apply_config_updates).in_set(ShipSet::Input),
    );

    // Startup: scan grid for existing functional blocks.
    app.add_systems(Startup, init_functional_blocks);

    // Block editing: produce edits, apply to grid, process entity lifecycle.
    app.add_systems(
        Update,
        (produce_player_edits, apply_block_edits, bevy_ecs::schedule::ApplyDeferred, process_entity_ops, process_sub_block_edits, process_mechanical_edits, aggregate_ship_properties, sync_colliders_to_host, rebuild_reactor_consumers)
            .chain()
            .in_set(ShipSet::BlockEdit),
    );

    // Signal pipeline: seat publish → custom block systems → evaluate → subscribe → thrust.
    // Order determines priority: later blocks override earlier ones on the same channel.
    app.add_systems(
        Update,
        (
            read_mechanical_state,
            signal_publish,             // Seat + block publishers → merge
            flight_computer_system,     // Damping on rotation channels
            hover_module_system,        // Attitude + gravity comp on all channels
            autopilot_system,           // Steering overrides on rotation channels
            warp_computer_system,       // Reads channels, manages warp state
            engine_controller_system,   // Cutoff: zeros ALL channels (runs last)
            signal_evaluate,            // Converter rules
            signal_subscribe,           // Thrusters read final channel values
            apply_mechanical_transforms,
            compute_power_budget,
            compute_ship_thrust,
            signal_clear_dirty,
        )
            .chain()
            .in_set(ShipSet::Signal),
    );

    // Physics.
    app.add_systems(
        Update,
        (tick_counter, update_player_platform, physics_step).chain().in_set(ShipSet::Physics),
    );

    // Interaction: hull exit detection (seat interaction moved to BlockEdit via raycast).
    app.add_systems(
        Update,
        hull_exit_check.in_set(ShipSet::Interaction),
    );

    // Send: QUIC messages to host shard.
    app.add_systems(Update, (pilot_send, signal_broadcast_remote).in_set(ShipSet::Send));

    // Broadcast: WorldState to client.
    app.add_systems(Update, broadcast_world_state.in_set(ShipSet::Broadcast));

    // Diagnostics.
    app.add_systems(Update, (persist_chunks, log_state).in_set(ShipSet::Diagnostics));

    app
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let ship_id = if args.ship_id > 0 {
        args.ship_id
    } else {
        args.seed
    };
    let host_shard_id = args.host_shard.map(ShardId);

    let bind = "0.0.0.0";
    let config = ShardHarnessConfig {
        shard_id: ShardId(args.shard_id),
        shard_type: ShardType::Ship,
        tcp_addr: format!("{bind}:{}", args.tcp_port).parse().unwrap(),
        udp_addr: format!("{bind}:{}", args.udp_port).parse().unwrap(),
        quic_addr: format!("{bind}:{}", args.quic_port).parse().unwrap(),
        orchestrator_url: args.orchestrator.clone(),
        orchestrator_heartbeat_addr: args.orchestrator_heartbeat,
        healthz_addr: format!("{bind}:{}", args.healthz_port).parse().unwrap(),
        planet_seed: None,
        system_seed: if args.system_seed > 0 {
            Some(args.system_seed)
        } else {
            None
        },
        ship_id: Some(ship_id),
        galaxy_seed: None,
        host_shard_id,
        advertise_host: args.advertise_host,
    };

    info!(
        shard_id = args.shard_id,
        ship_id,
        host_shard = ?host_shard_id,
        "ship shard starting"
    );

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let harness = ShardHarness::new(config);
        let app = build_ship_interior(
            ShardId(args.shard_id),
            ship_id,
            host_shard_id,
            args.system_seed,
            args.galaxy_seed,
        );

        info!("ship shard ECS app built, starting harness");
        harness.run_ecs(app).await;
    });
}
