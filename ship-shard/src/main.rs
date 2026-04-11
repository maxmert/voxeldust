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
    PlayerSnapshotData, SeatBindingsNotifyData, ServerMsg, WorldStateData,
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

const WALK_SPEED: f32 = 4.0;
const JUMP_IMPULSE: f32 = 5.0;
const INTERACT_DIST: f32 = 1.5;

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

/// Per-binding float values from the client's seat input evaluation.
/// Length matches the active seat's binding count. Written by process_input.
#[derive(Resource, Default)]
struct SeatInputValues(Vec<f32>);

/// Tracks which seat entity the pilot occupies (needed for SeatChannelMapping lookup).
#[derive(Resource, Default)]
struct ActiveSeatEntity(Option<Entity>);

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
}

/// Data for a single sub-grid (child rigid body connected via joint).
struct SubGridData {
    /// Rapier rigid body handle for this sub-grid.
    body_handle: rapier3d::dynamics::RigidBodyHandle,
    /// Rapier joint handle connecting this sub-grid to its parent.
    joint_handle: rapier3d::dynamics::ImpulseJointHandle,
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

/// Connected player state.
#[derive(Resource, Default)]
struct ConnectedPlayer {
    session: Option<SessionToken>,
    player_name: Option<String>,
    handoff_pending: bool,
}

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
    handoff: Option<ShardMsg>,
}

/// Player Rapier body handle.
#[derive(Resource)]
struct PlayerBodyHandle(RigidBodyHandle);

/// Player position in ship-local coordinates.
#[derive(Resource)]
struct PlayerPosition(Vec3);

/// Player yaw angle.
#[derive(Resource)]
struct PlayerYaw(f32);

/// Whether the player is seated and which seat they're in.
/// Seat role (pilot/gunner/passenger) is determined by the seat's SeatChannelMapping,
/// not by a flag here. A turret seat publishes turret channels, not thrust.
#[derive(Resource, Default)]
struct SeatedState {
    seated: bool,
    seat_entity: Option<Entity>,
}


/// Whether gravity is enabled in the ship.
#[derive(Resource)]
struct GravityEnabled(bool);

/// Input action state for edge detection.
#[derive(Resource, Default)]
struct InputActions {
    current: u8,
    previous: u8,
}

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
    input: voxeldust_core::client_message::PlayerInputData,
}

#[derive(Message)]
struct BlockEditMsg {
    edit: BlockEditData,
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
    for _ in 0..64 {
        let (_src, input) = match bridge.input_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        events.write(PlayerInputMsg { input });
    }
}

fn drain_quic(
    mut bridge: ResMut<NetworkBridge>,
    mut scene: ResMut<SceneCache>,
    mut exterior: ResMut<ShipExterior>,
    mut config: ResMut<ShipConfig>,
    mut autopilot_cache: ResMut<AutopilotSnapshotCache>,
    mut connected: ResMut<ConnectedPlayer>,
    mut landing: ResMut<LandingState>,
    mut rapier: ResMut<RapierContext>,
    body_handle: Res<PlayerBodyHandle>,
    mut player_pos: ResMut<PlayerPosition>,
    mut pilot_mode: ResMut<SeatedState>,
    tick: Res<ecs::TickCounter>,
    mut incoming_signals: ResMut<IncomingSignalBuffer>,
    mut peer_positions: ResMut<PeerPositionCache>,
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
                // Planet shard accepted the player. Send ShardRedirect to client.
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

                connected.session = None;
                connected.player_name = None;
                connected.handoff_pending = false;
                pilot_mode.seated = false;
            }
            ShardMsg::PlayerHandoff(h) => {
                // Player re-entering ship from planet.
                info!(
                    session = h.session_token.0,
                    player = %h.player_name,
                    "player re-entering ship via handoff"
                );
                connected.session = Some(h.session_token);
                connected.player_name = Some(h.player_name.clone());
                connected.handoff_pending = false;
                pilot_mode.seated = false;

                // Spawn player at a safe interior position.
                // TODO: find the nearest air block inside the hull boundary.
                let reentry_pos = Vec3::new(0.0, 1.5, 0.0);
                if let Some(body) = rapier.rigid_body_set.get_mut(body_handle.0) {
                    body.set_translation(
                        vector![reentry_pos.x, reentry_pos.y, reentry_pos.z],
                        true,
                    );
                    body.set_linvel(vector![0.0, 0.0, 0.0], true);
                }
                player_pos.0 = reentry_pos;

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

                // Send ShardPreConnect to client for secondary UDP.
                if !data.new_host_udp_addr.is_empty() {
                    let pc = ServerMsg::ShardPreConnect(handoff::ShardPreConnect {
                        shard_type: data.new_host_shard_type,
                        tcp_addr: data.new_host_tcp_addr.clone(),
                        udp_addr: data.new_host_udp_addr.clone(),
                        seed: data.seed,
                        planet_index: 0,
                        reference_position: DVec3::ZERO,
                        reference_rotation: DQuat::IDENTITY,
                    });
                    if let Some(session) = connected.session {
                        let cr = bridge.client_registry.clone();
                        tokio::spawn(async move {
                            let reg = cr.read().await;
                            let _ = reg.send_tcp(session, &pc).await;
                        });
                        info!("sent ShardPreConnect to client for secondary UDP");
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
            _ => {}
        }
    }
}


// ---------------------------------------------------------------------------
// Connect processing
// ---------------------------------------------------------------------------

fn process_connects(
    mut events: MessageReader<ClientConnectedMsg>,
    mut connected: ResMut<ConnectedPlayer>,
    exterior: Res<ShipExterior>,
    config: Res<ShipConfig>,
    scene: Res<SceneCache>,
    ship_grid: Res<ShipGridResource>,
) {
    for event in events.read() {
        connected.session = Some(event.session_token);
        connected.player_name = Some(event.player_name.clone());
        connected.handoff_pending = false;

        let token = event.session_token;
        let tcp_write = event.tcp_write.clone();
        let ship_pos = exterior.position;
        let ship_rot = exterior.rotation;
        let game_time = scene.game_time;
        let ship_id = config.ship_id;
        let galaxy_seed = config.galaxy_seed;
        let system_seed = config.system_seed;

        // Serialize all chunk snapshots for the initial sync.
        // Done synchronously on the ECS thread because ShipGrid is small
        // (starter ship is 1-2 chunks). For large ships/stations, this would
        // be moved to a background task with a read snapshot.
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

        tokio::spawn(async move {
            let jr = ServerMsg::JoinResponse(JoinResponseData {
                seed: ship_id,
                planet_radius: 0,
                player_id: token.0,
                spawn_position: DVec3::new(0.0, 1.0, 0.0),
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

            // Send all chunk snapshots immediately after JoinResponse.
            for snapshot_msg in &chunk_snapshots {
                let _ = client_listener::send_tcp_msg(&mut *writer, snapshot_msg).await;
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
    mut actions: ResMut<InputActions>,
    mut player_yaw: ResMut<PlayerYaw>,
    pilot_mode: Res<SeatedState>,
    mut seat_input: ResMut<SeatInputValues>,
    mut rapier: ResMut<RapierContext>,
    body_handle: Res<PlayerBodyHandle>,
) {
    for event in events.read() {
        let input = &event.input;
        actions.previous = actions.current;
        actions.current = input.action;
        player_yaw.0 = input.look_yaw;

        if pilot_mode.seated {
            // Copy per-binding seat values from client input.
            // The client evaluates its local SeatBindings and sends floats.
            // If seat_values is empty (legacy client), keep existing values.
            if !input.seat_values.is_empty() {
                seat_input.0 = input.seat_values.clone();
            }
        } else {
            // Walking mode: apply velocity to Rapier body.
            let (sin_y, cos_y) = player_yaw.0.sin_cos();
            let fwd = Vec3::new(cos_y, 0.0, sin_y);
            let right = Vec3::new(-sin_y, 0.0, cos_y);

            let move_vel =
                fwd * input.movement[2] * WALK_SPEED + right * input.movement[0] * WALK_SPEED;

            if let Some(body) = rapier.rigid_body_set.get_mut(body_handle.0) {
                let current_vel = *body.linvel();
                body.set_linvel(vector![move_vel.x, current_vel.y, move_vel.z], true);

                if input.jump && current_vel.y.abs() < 0.1 {
                    body.apply_impulse(vector![0.0, JUMP_IMPULSE, 0.0], true);
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
    connected: Res<ConnectedPlayer>,
    config: Res<ShipConfig>,
    exterior: Res<ShipExterior>,
    scene: Res<SceneCache>,
    cached_sys: Res<CachedSystemParams>,
    bridge: Res<NetworkBridge>,
) {
    if atmo.preconnect_sent || connected.session.is_none() {
        return;
    }
    if config.system_seed == 0 {
        return;
    }

    let sys = match &cached_sys.0 {
        Some(sys) => sys,
        None => return,
    };

    let in_atmo_planet = scene
        .bodies
        .iter()
        .filter(|b| b.body_id > 0)
        .find_map(|b| {
            let pi = (b.body_id - 1) as usize;
            if pi >= sys.planets.len() {
                return None;
            }
            let alt = (exterior.position - b.position).length() - sys.planets[pi].radius_m;
            if alt < sys.planets[pi].atmosphere.atmosphere_height
                && sys.planets[pi].atmosphere.has_atmosphere
            {
                Some(pi)
            } else {
                None
            }
        });

    let was_in_atmo = atmo.in_atmosphere;
    atmo.in_atmosphere = in_atmo_planet.is_some();
    if was_in_atmo && !atmo.in_atmosphere {
        atmo.preconnect_sent = false;
    }

    let planet_index = match in_atmo_planet {
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

    let session = connected.session.unwrap();
    let pc = handoff::ShardPreConnect {
        shard_type: 0,
        tcp_addr,
        udp_addr,
        seed: planet_seed,
        planet_index: planet_index as u32,
        reference_position: exterior.position,
        reference_rotation: DQuat::IDENTITY,
    };

    atmo.preconnect_sent = true;

    let cr = bridge.client_registry.clone();
    let msg = ServerMsg::ShardPreConnect(pc);
    tokio::spawn(async move {
        if let Ok(reg) = cr.try_read() {
            if let Err(e) = reg.send_tcp(session, &msg).await {
                tracing::warn!(%e, "failed to send ShardPreConnect");
            }
        }
    });

    info!(planet_seed, planet_index, "sent ShardPreConnect to client");
}


/// Detect when the player has left the ship's hull volume and trigger
/// handoff to the nearest planet shard. Works for any ship shape — checks
/// the player's position against the tight bounding box of all solid blocks.
fn hull_exit_check(
    player_pos: Res<PlayerPosition>,
    hull_bounds: Res<ShipHullBounds>,
    mut connected: ResMut<ConnectedPlayer>,
    exterior: Res<ShipExterior>,
    scene: Res<SceneCache>,
    cached_sys: Res<CachedSystemParams>,
    config: Res<ShipConfig>,
    tick: Res<ecs::TickCounter>,
    mut pending: ResMut<PendingMessages>,
) {
    if connected.handoff_pending {
        return;
    }

    // Check if the player is outside the ship's solid block bounding box.
    let pos = player_pos.0;
    let margin = hull_bounds.margin;
    let inside = pos.x >= hull_bounds.min.x as f32 - margin
        && pos.x <= hull_bounds.max.x as f32 + 1.0 + margin
        && pos.y >= hull_bounds.min.y as f32 - margin
        && pos.y <= hull_bounds.max.y as f32 + 1.0 + margin
        && pos.z >= hull_bounds.min.z as f32 - margin
        && pos.z <= hull_bounds.max.z as f32 + 1.0 + margin;

    if inside {
        return;
    }

    let (session, player_name) = match (connected.session, connected.player_name.clone()) {
        (Some(s), Some(n)) => (s, n),
        _ => return,
    };

    let player_local = DVec3::new(
        player_pos.0.x as f64,
        player_pos.0.y as f64,
        player_pos.0.z as f64,
    );
    let player_system_pos = exterior.position + exterior.rotation * player_local;

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
                    session_token: session,
                    player_name,
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
                };
                connected.handoff_pending = true;
                pending.handoff = Some(ShardMsg::PlayerHandoff(h));
                info!(
                    planet_index,
                    planet_seed, "threshold crossing — auto-handoff initiated"
                );
            }
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

    // Always send pending handoff.
    if let Some(handoff_msg) = pending.handoff.take() {
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

fn physics_step(
    mut rapier: ResMut<RapierContext>,
    gravity_sources: Res<GravitySources>,
    gravity_enabled: Res<GravityEnabled>,
    body_handle: Res<PlayerBodyHandle>,
    mut player_pos: ResMut<PlayerPosition>,
) {
    let gravity = if gravity_enabled.0 {
        let mut total = vector![0.0, 0.0, 0.0];
        for source in &gravity_sources.0 {
            total += source.gravity_at(player_pos.0);
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

    if let Some(body) = ctx.rigid_body_set.get(body_handle.0) {
        let t = body.translation();
        player_pos.0 = Vec3::new(t.x, t.y, t.z);
    }
}

fn tick_counter(mut tick: ResMut<ecs::TickCounter>) {
    tick.0 += 1;
}

// ---------------------------------------------------------------------------
// Broadcast
// ---------------------------------------------------------------------------

fn broadcast_world_state(
    player_pos: Res<PlayerPosition>,
    exterior: Res<ShipExterior>,
    scene: Res<SceneCache>,
    pilot_mode: Res<SeatedState>,
    warp_query: Query<&WarpComputerState>,
    autopilot_cache: Res<AutopilotSnapshotCache>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
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

    let ws = ServerMsg::WorldState(WorldStateData {
        tick: tick.0,
        origin: exterior.position,
        players: vec![PlayerSnapshotData {
            player_id: 0,
            position: DVec3::new(
                player_pos.0.x as f64,
                player_pos.0.y as f64,
                player_pos.0.z as f64,
            ),
            rotation: exterior.rotation,
            velocity: exterior.velocity,
            grounded: !pilot_mode.seated,  // TODO: use actual Rapier ground contact
            health: 100.0,
            shield: 100.0,
            seated: pilot_mode.seated,
        }],
        bodies,
        ships: vec![],
        lighting,
        game_time: scene.game_time,
        warp_target_star_index: warp_target,
        autopilot: autopilot_cache.snapshot.clone(),
    });
    let _ = bridge.broadcast_tx.try_send(ws);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

fn log_state(
    player_pos: Res<PlayerPosition>,
    pilot_mode: Res<SeatedState>,
    gravity_enabled: Res<GravityEnabled>,
    tick: Res<ecs::TickCounter>,
    scene: Res<SceneCache>,
    exterior: Res<ShipExterior>,
) {
    if tick.0 % 100 != 0 {
        return;
    }

    info!(
        pos = format!(
            "({:.1}, {:.1}, {:.1})",
            player_pos.0.x, player_pos.0.y, player_pos.0.z
        ),
        piloting = pilot_mode.seated,
        gravity = gravity_enabled.0,
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
    /// Full seat control mappings (control_u8, channel_name, property_u8).
    /// If non-empty, replaces the default 13-channel SeatChannelMapping on load.
    seat_mappings: Vec<(u8, String, u8)>,
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
    // seat_mappings: u32 count, then per mapping: (u8 control, u32 name_len, name, u8 property)
    buf.extend_from_slice(&(record.seat_mappings.len() as u32).to_le_bytes());
    for (control, name, prop) in &record.seat_mappings {
        buf.push(*control);
        let nb = name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        buf.extend_from_slice(nb);
        buf.push(*prop);
    }
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
    if pos + 4 > data.len() { return Some(BlockConfigRecord { channel_override, boost_channel: String::new(), power_config: None, subscribe_bindings: Vec::new(), publish_bindings: Vec::new(), seat_mappings: Vec::new() }); }
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

    // Seat mappings (optional — backward compatible).
    let mut seat_mappings = Vec::new();
    if pos + 4 <= data.len() {
        let count = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
        pos += 4;
        for _ in 0..count {
            if pos + 1 > data.len() { break; }
            let control = data[pos];
            pos += 1;
            if pos + 4 > data.len() { break; }
            let name_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok().unwrap_or([0;4])) as usize;
            pos += 4;
            if pos + name_len + 1 > data.len() { break; }
            let name = std::str::from_utf8(&data[pos..pos+name_len]).unwrap_or("").to_string();
            pos += name_len;
            let prop = data[pos];
            pos += 1;
            seat_mappings.push((control, name, prop));
        }
    }

    Some(BlockConfigRecord { channel_override, boost_channel, power_config, subscribe_bindings, publish_bindings, seat_mappings })
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
            if !record.seat_mappings.is_empty() {
                grid.set_saved_seat_mappings(block_pos, record.seat_mappings);
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
                seat_mappings: Vec::new(),
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

/// ECS Message for incoming config updates.
#[derive(Message)]
struct ConfigUpdateMsg {
    update: voxeldust_core::signal::config::BlockConfigUpdateData,
}

/// Bridge system: drain BlockConfigUpdate messages from the UDP channel.
fn drain_config_updates(
    mut bridge: ResMut<NetworkBridge>,
    mut events: MessageWriter<ConfigUpdateMsg>,
) {
    for _ in 0..16 {
        let update = match bridge.config_update_rx.try_recv() {
            Ok(u) => u,
            Err(_) => break,
        };
        events.write(ConfigUpdateMsg { update });
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
            commands.entity(entity).insert(SignalSubscriber {
                bindings: update.subscribe_bindings.iter().map(|b| SubscribeBinding {
                    channel_id: channels.resolve_or_create(&b.channel_name, default_scope, default_merge, 0),
                    property: b.property,
                }).collect(),
            });
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
        if !update.seat_mappings.is_empty() {
            commands.entity(entity).insert(SeatChannelMapping {
                bindings: update.seat_mappings.iter().map(|s| SeatInputBinding {
                    control: s.control,
                    channel_id: channels.resolve_or_create(&s.channel_name, default_scope, default_merge, 0),
                    property: s.property,
                }).collect(),
            });
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
                seat_mappings: update.seat_mappings.iter()
                    .map(|s| (s.control as u8, s.channel_name.clone(), s.property as u8))
                    .collect(),
            };
            save_single_block_config(&persist.db, update.block_pos, &record);
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
        let edit = match bridge.block_edit_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        events.write(BlockEditMsg { edit });
    }
}

/// ECS message for incoming sub-block edits.
#[derive(Message)]
struct SubBlockEditMsg {
    edit: voxeldust_core::client_message::SubBlockEditData,
}

/// Bridge system: drain SubBlockEditRequest messages from the UDP channel.
fn drain_sub_block_edits(
    mut bridge: ResMut<NetworkBridge>,
    mut events: MessageWriter<SubBlockEditMsg>,
) {
    for _ in 0..64 {
        let edit = match bridge.sub_block_edit_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        events.write(SubBlockEditMsg { edit });
    }
}

/// Process sub-block edit requests: validate, apply to grid, broadcast delta.
fn process_sub_block_edits(
    mut events: MessageReader<SubBlockEditMsg>,
    mut grid: ResMut<ShipGridResource>,
    registry: Res<BlockRegistryResource>,
    connected: Res<ConnectedPlayer>,
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

        // Broadcast sub-block delta to clients.
        if let Some(session) = connected.session {
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
                    let _ = reg.send_tcp(session, &delta).await;
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
    connected: Res<ConnectedPlayer>,
    player_pos: Res<PlayerPosition>,
    block_index: Res<FunctionalBlockIndex>,
    mut pilot_mode: ResMut<SeatedState>,
    mut active_seat: ResMut<ActiveSeatEntity>,
    mut rapier: ResMut<RapierContext>,
    body_handle: Res<PlayerBodyHandle>,
    bridge: Res<NetworkBridge>,
    mut signal_ctx: SignalQueryCtx,
) {
    if connected.session.is_none() || connected.handoff_pending {
        return;
    }
    let session = connected.session.unwrap();

    for event in events.read() {
        let edit = &event.edit;

        // Action EXIT_SEAT: no raycast or position validation needed.
        if edit.action == voxeldust_core::client_message::action::EXIT_SEAT && pilot_mode.seated {
            pilot_mode.seated = false;
            info!("exited pilot mode via F key");
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
        let hit = raycast::raycast(eye, look, BLOCK_EDIT_RANGE, |x, y, z| {
            registry.0.is_solid(grid.0.get_block(x, y, z))
        });

        let hit = match hit {
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
                            source: EditSource::Player(session),
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
                    source: EditSource::Player(session),
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
                                    // Toggle seated state.
                                    pilot_mode.seated = !pilot_mode.seated;
                                    if pilot_mode.seated {
                                        pilot_mode.seat_entity = Some(entity);
                                        active_seat.0 = Some(entity);
                                        if let Some(body) = rapier.rigid_body_set.get_mut(body_handle.0) {
                                            body.set_linvel(vector![0.0, 0.0, 0.0], true);
                                        }
                                        info!("entered seat at {:?}", hit.world_pos);
                                    } else {
                                        pilot_mode.seat_entity = None;
                                        // Keep active_seat for one-tick zero-publish.
                                        info!("exited seat");
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
                        );
                        if let Some(session) = connected.session {
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

    let seat_mappings = seat_query.get(entity)
        .map(|s| s.bindings.iter().map(|b| SeatInputBindingConfig {
            control: b.control,
            channel_name: id_to_name(b.channel_id),
            property: b.property,
        }).collect())
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
        available_channels,
        power_source,
        power_consumer,
        nearby_reactors,
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
            // Offset shapes relative to the sub-grid body position.
            let body_pos = rapier.rigid_body_set.get(sg_data.body_handle)
                .map(|b| *b.translation())
                .unwrap_or_default();
            let compound: Vec<_> = sg_shapes.iter().map(|&(pos, he)| {
                let local = Isometry::translation(
                    pos.x - body_pos.x,
                    pos.y - body_pos.y,
                    pos.z - body_pos.z,
                );
                (local, SharedShape::cuboid(he.x, he.y, he.z))
            }).collect();

            // Use collision groups to prevent parent-child Clang.
            let parent_group = 1u32 << (sg_data.parent_grid.0 % 32);
            let child_group = 1u32 << (sg_id.0 % 32);
            let collider = ColliderBuilder::compound(compound)
                .collision_groups(rapier3d::geometry::InteractionGroups::new(
                    rapier3d::geometry::Group::from(child_group),
                    rapier3d::geometry::Group::from(!parent_group),
                ))
                .build();
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
    connected: Res<ConnectedPlayer>,
    bridge: Res<NetworkBridge>,
    mut persistence: Option<ResMut<ShipPersistence>>,
    mut pending_entity_ops: Option<ResMut<PendingEntityOps>>,
    mut agg_dirty: ResMut<AggregationDirty>,
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

    // Mark aggregation as needing recomputation.
    agg_dirty.0 = true;

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
            let output_face = block::sub_block::opposite_face(sg_data.mount_face);
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

    // 4. Broadcast ChunkDeltas to all connected clients (one per dirty chunk).
    if let Some(session) = connected.session {
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
                    let _ = reg.send_tcp(session, &delta).await;
                }
            });
        }
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
) {
    use voxeldust_core::block::sub_block::{SubBlockType, face_to_offset};
    use voxeldust_core::signal::{ChannelMergeStrategy, SignalScope};

    // Re-compute membership for dirty sub-grids (blocks placed/removed on child side).
    let dirty_ids: Vec<_> = mech_dirty.dirty.drain().collect();
    for sg_id in dirty_ids {
        let Some(sg_data) = sub_grids.grids.get(&sg_id) else { continue };
        let mount_pos = sg_data.mount_pos;
        let mount_face = sg_data.mount_face;
        let output_face = block::sub_block::opposite_face(mount_face);

        // Re-BFS from the output face.
        let new_members = block::ship_grid::compute_sub_grid_members(
            &grid.0, &registry.0, mount_pos, output_face,
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
        let output_face = block::sub_block::opposite_face(face);
        let members = block::ship_grid::compute_sub_grid_members(
            &grid.0, &registry.0, mount_pos, output_face,
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
        let child_body = rapier3d::dynamics::RigidBodyBuilder::dynamic()
            .translation(anchor)
            .additional_mass(child_mass as f32)
            .build();
        let child_handle = rapier.rigid_body_set.insert(child_body);

        // Create joint.
        let axis_vec = face_to_offset(output_face);
        let axis = rapier3d::math::Vector::new(
            axis_vec.x as f32, axis_vec.y as f32, axis_vec.z as f32,
        );

        let joint_handle = match mech_props.joint_type {
            block::JointType::Revolute => {
                let unit_axis = rapier3d::math::UnitVector::new_normalize(axis);
                let mut builder = rapier3d::dynamics::RevoluteJointBuilder::new(unit_axis)
                    .local_anchor1(rapier3d::math::Point::from(anchor))
                    .local_anchor2(rapier3d::math::Point::origin());
                if mech_props.max_range < 360.0 {
                    let half = (mech_props.max_range / 2.0).to_radians() as f32;
                    builder = builder.limits([-half, half]);
                }
                rapier.impulse_joint_set.insert(
                    root_body.0, child_handle, builder.build(), true,
                )
            }
            block::JointType::Prismatic => {
                let unit_axis = rapier3d::math::UnitVector::new_normalize(axis);
                let builder = rapier3d::dynamics::PrismaticJointBuilder::new(unit_axis)
                    .local_anchor1(rapier3d::math::Point::from(anchor))
                    .local_anchor2(rapier3d::math::Point::origin())
                    .limits([0.0, mech_props.max_range as f32]);
                rapier.impulse_joint_set.insert(
                    root_body.0, child_handle, builder.build(), true,
                )
            }
        };

        // Register sub-grid (colliders will be built by rebuild_chunk_collider below).
        let members_clone = members.clone();
        sub_grids.grids.insert(sg_id, SubGridData {
            body_handle: child_handle,
            joint_handle,
            members,
            mount_pos,
            mount_face: face,
            parent_grid: voxeldust_core::ecs::components::SubGridId::ROOT,
            collider_handles: std::collections::HashMap::new(),
        });

        // Spawn functional entity for signal integration.
        let kind = match elem_type {
            SubBlockType::RotorMount | SubBlockType::HingeMount => block::FunctionalBlockKind::Rotor,
            SubBlockType::PistonMount | SubBlockType::SliderMount => block::FunctionalBlockKind::Piston,
            _ => continue,
        };

        let entity = commands.spawn((
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
            },
            SignalSubscriber::default(),
            SignalPublisher { bindings: Vec::new() },
        )).id();

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
        _ => {
            // Other kinds: no default signal bindings.
            // Players can add bindings via the config UI.
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

/// Flight computer: reads angular velocity and injects counter-rotation
/// signals into RawPilotInput on axes with no pilot input. These signals
/// flow through the normal signal pipeline → RCS thrusters → power consumption.
///
/// Uses capped proportional control: correction is proportional to angular
/// velocity but never exceeds MAX_CORRECTION throttle. This prevents
/// oscillation from network round-trip delay — the bounded correction
/// can't overshoot enough to reverse rotation, so the system is
/// unconditionally stable regardless of delay or ship mass.
fn flight_computer_damping(
    mut raw_input: ResMut<RawPilotInput>,
    exterior: Res<ShipExterior>,
) {
    if raw_input.engines_off { return; }

    // Angular velocity in ship-local frame.
    let ang_vel = exterior.rotation.inverse() * exterior.angular_velocity;

    // Dead zone: ignore angular velocity below this to prevent thruster jitter.
    const DEAD_ZONE: f64 = 0.005; // 0.005 rad/s ≈ 0.3 deg/s

    // Max correction throttle: never more than 30% RCS. This bounds the
    // angular deceleration so the round-trip delay (3-5 ticks) can't cause
    // enough velocity change to reverse rotation direction. The system is
    // unconditionally stable: even with infinite delay, the ship just
    // corrects slowly instead of oscillating.
    const MAX_CORRECTION: f64 = 0.3;

    // Proportional gain within the cap. At 0.5 rad/s → 30% throttle.
    // Below that, correction scales linearly for smooth convergence.
    const GAIN: f64 = 0.6; // MAX_CORRECTION / 0.5 rad/s

    // Signal convention: torque_yaw > 0 → fires CW thrusters → negative Y torque.
    // Same sign = counter-rotation (channel mapping provides the inversion).

    // --- Yaw (Y-axis) ---
    if raw_input.torque_yaw.abs() < 0.01 {
        let v = ang_vel.y;
        if v.abs() > DEAD_ZONE {
            raw_input.torque_yaw = (v * GAIN).clamp(-MAX_CORRECTION, MAX_CORRECTION) as f32;
        }
    }

    // --- Pitch (X-axis) ---
    if raw_input.torque_pitch.abs() < 0.01 {
        let v = ang_vel.x;
        if v.abs() > DEAD_ZONE {
            raw_input.torque_pitch = (v * GAIN).clamp(-MAX_CORRECTION, MAX_CORRECTION) as f32;
        }
    }

    // --- Roll (Z-axis) ---
    if raw_input.roll_cw < 0.01 && raw_input.roll_ccw < 0.01 {
        let v = ang_vel.z;
        if v.abs() > DEAD_ZONE {
            let c = (v * GAIN).clamp(-MAX_CORRECTION, MAX_CORRECTION) as f32;
            if c > 0.0 {
                raw_input.roll_cw = c;
            } else {
                raw_input.roll_ccw = -c;
            }
        }
    }
}

/// 6-DOF hover flight computer. Three layers, all through the signal pipeline:
///
/// 1. **Attitude hold**: maintains target orientation (belly-down + captured heading)
///    via RCS torque commands. Fights weathercock aero torque.
/// 2. **Gravity compensation**: fires thrusters to cancel gravity per ship-local axis.
/// 3. **Velocity damping**: brakes all velocity to zero for station-keeping.
///
/// All outputs write to `raw_input` → `signal_publish` → real thrusters.
/// Scales with terminal velocity: zero effect at re-entry speeds.
fn hover_computer(
    mut raw_input: ResMut<RawPilotInput>,
    mut hover: ResMut<HoverState>,
    exterior: Res<ShipExterior>,
    ship_props: Res<ShipProps>,
    pilot_mode: Res<SeatedState>,
    autopilot: Res<AutopilotState>,
) {
    let active = raw_input.atmo_comp_active && exterior.in_atmosphere
        && !raw_input.engines_off && autopilot.target_body_id.is_none();

    // Edge detect: capture heading on activation.
    if active && !hover.was_active {
        let g = exterior.gravity_acceleration.length();
        if g > 1e-6 {
            let up = -exterior.gravity_acceleration / g;
            let fwd_world = exterior.rotation * DVec3::NEG_Z;
            // Project forward onto horizontal plane (perpendicular to gravity).
            let fwd_horiz = fwd_world - up * fwd_world.dot(up);
            hover.captured_heading = if fwd_horiz.length() > 1e-6 {
                fwd_horiz.normalize()
            } else {
                // Ship pointing straight up/down — pick arbitrary horizontal heading.
                let candidate = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
                (candidate - up * candidate.dot(up)).normalize()
            };
        }
    }
    hover.was_active = active;
    if !active {
        // Log why hover is inactive (every 40 ticks).
        static DIAG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let t = DIAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if t % 40 == 0 && pilot_mode.seated {
            tracing::warn!(
                atmo_comp = raw_input.atmo_comp_active,
                in_atmo = exterior.in_atmosphere,
                engines_off = raw_input.engines_off,
                has_autopilot = autopilot.target_body_id.is_some(),
                "hover_computer INACTIVE"
            );
        }
        return;
    }

    let g = exterior.gravity_acceleration.length();
    if g < 1e-6 { return; }

    // No terminal velocity scaling: with real thrusters (limited capacity),
    // the controller can't produce enough force to catapult the ship.
    // The thrusters are the natural limiter — max ~40 m/s² vs drag at
    // hypersonic speeds which is thousands of m/s². Hover is harmless at
    // high speed and essential at low speed.
    let hf = 1.0_f64;

    // =========================================================================
    // Layer 1: Attitude hold — target orientation from gravity + captured heading
    // =========================================================================
    let up = -exterior.gravity_acceleration / g;
    let fwd_raw = hover.captured_heading;
    // Re-orthogonalize: right = fwd × up, then fwd = up × right.
    let right = fwd_raw.cross(up);
    let right = if right.length() > 1e-6 { right.normalize() } else {
        let c = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
        c.cross(up).normalize()
    };
    let fwd = up.cross(right); // orthogonal forward in horizontal plane

    // Target rotation: glam convention is -Z = forward, +Y = up.
    // Column matrix: [right, up, -forward] maps local axes to world.
    let target_rot = DQuat::from_mat3(&glam::DMat3::from_cols(right, up, -fwd)).normalize();

    // Error quaternion: rotation from current to target.
    let q_error = (target_rot * exterior.rotation.inverse()).normalize();

    // Convert to axis-angle. For small angles: axis×angle ≈ 2×(qx,qy,qz).
    // For large angles, use proper extraction.
    let (axis, angle) = if q_error.w < 0.0 {
        // Ensure shortest path (q and -q represent same rotation).
        let q = DQuat::from_xyzw(-q_error.x, -q_error.y, -q_error.z, -q_error.w);
        (DVec3::new(q.x, q.y, q.z), 2.0 * q.w.acos())
    } else {
        (DVec3::new(q_error.x, q_error.y, q_error.z), 2.0 * q_error.w.acos())
    };

    let error_world = if angle > 1e-6 { axis.normalize() * angle } else { DVec3::ZERO };
    let error_local = exterior.rotation.inverse() * error_world;

    // Proportional gain: at 30° error, command full torque (gain = 1/0.52 ≈ 1.9).
    let att_gain = 1.9 * hf;

    if raw_input.torque_pitch.abs() < 0.01 {
        raw_input.torque_pitch = (error_local.x * att_gain).clamp(-1.0, 1.0) as f32;
    }
    if raw_input.torque_yaw.abs() < 0.01 {
        raw_input.torque_yaw = (error_local.y * att_gain).clamp(-1.0, 1.0) as f32;
    }
    // Roll: error_local.z > 0 means need +Z rotation (CCW from pilot view = roll_ccw).
    // rotation_command.z = roll_ccw - roll_cw, so positive Z needs roll_ccw.
    if raw_input.roll_cw < 0.01 && raw_input.roll_ccw < 0.01 {
        let roll_cmd = (error_local.z * att_gain).clamp(-1.0, 1.0) as f32;
        if roll_cmd > 0.0 {
            raw_input.roll_ccw = roll_cmd;
        } else {
            raw_input.roll_cw = -roll_cmd;
        }
    }

    // =========================================================================
    // Layer 2: Gravity compensation — cancel gravity via directional thrusters
    // =========================================================================
    let grav_local = exterior.rotation.inverse() * exterior.gravity_acceleration;
    let mass = ship_props.0.mass_kg;
    let tpa = &ship_props.0.thrust_per_axis;

    // Effective thrust per axis: accounts for the thrust limiter that the signal
    // pipeline applies after throttle. We divide by it so our throttle commands
    // produce the intended force.
    let limiter = (raw_input.thrust_limiter as f64).max(0.01);
    let effective = |axis: usize| -> f64 { tpa[axis] * limiter };

    // =========================================================================
    // Layer 2: Gravity compensation (feedforward)
    // =========================================================================
    // Computes the throttle needed to cancel gravity. Accounts for thrust limiter.
    // This is the static baseline — gets ~95% right. Layer 3 corrects the rest.
    let grav_comp = |component: f64, pos_axis: usize, neg_axis: usize| -> f32 {
        if component < 0.0 && tpa[pos_axis] > 1.0 {
            ((-component) * mass / effective(pos_axis) * hf) as f32
        } else if component > 0.0 && tpa[neg_axis] > 1.0 {
            -(component * mass / effective(neg_axis) * hf) as f32
        } else { 0.0 }
    };

    let ff_y = grav_comp(grav_local.y, 3, 2);
    let ff_x = grav_comp(grav_local.x, 1, 0);
    let ff_z = grav_comp(grav_local.z, 5, 4);

    // =========================================================================
    // Layer 3: PD velocity controller (feedback)
    // =========================================================================
    // Measures actual drift and dynamically corrects. Handles everything the
    // feedforward can't: power_ratio changes, CoM offset, rotation lag, etc.
    //
    // PD gains derived from control theory (no magic numbers):
    //   τ = 2s settling time → ω_n = 2π/(4τ) natural frequency
    //   ζ = 1.0 (critical damping: fastest response without overshoot)
    //   Kp = ω_n²,  Kd = 2ζω_n
    let tau = 2.0_f64;
    let omega_n = std::f64::consts::TAU / (4.0 * tau);
    let kp = omega_n * omega_n;
    let kd = 2.0 * omega_n; // ζ = 1.0

    let vel_local = exterior.rotation.inverse() * exterior.velocity;
    let dt = 0.05_f64; // physics tick (20Hz)
    let accel_local = (vel_local - hover.prev_velocity_local) / dt;
    hover.prev_velocity_local = vel_local;

    // PD correction force per axis: F = -(Kp·v + Kd·a)·m
    let pd_force = |vel: f64, accel: f64| -> f64 {
        -(kp * vel + kd * accel) * mass
    };

    // Convert force to throttle, accounting for limiter and hover_fraction.
    let to_throttle = |force: f64, pos_axis: usize, neg_axis: usize| -> f32 {
        if force > 0.0 && tpa[pos_axis] > 1.0 {
            (force / effective(pos_axis) * hf) as f32
        } else if force < 0.0 && tpa[neg_axis] > 1.0 {
            (force / effective(neg_axis) * hf) as f32
        } else { 0.0 }
    };

    let fb_y = to_throttle(pd_force(vel_local.y, accel_local.y), 3, 2);
    let fb_x = to_throttle(pd_force(vel_local.x, accel_local.x), 1, 0);
    let fb_z = to_throttle(pd_force(vel_local.z, accel_local.z), 5, 4);

    // Combine feedforward + feedback.
    raw_input.thrust_vertical = (raw_input.thrust_vertical + ff_y + fb_y).clamp(-1.0, 1.0);
    raw_input.thrust_lateral  = (raw_input.thrust_lateral + ff_x + fb_x).clamp(-1.0, 1.0);
    raw_input.thrust_forward  = (raw_input.thrust_forward + ff_z + fb_z).clamp(-1.0, 1.0);

    // Diagnostic logging (every 20 ticks = 1 second).
    static HOVER_DIAG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tick = HOVER_DIAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if tick % 20 == 0 {
        let speed = exterior.velocity.length();
        tracing::warn!(
            g = format!("{:.2}", g),
            speed = format!("{:.1}", speed),
            grav_local = format!("({:.2},{:.2},{:.2})", grav_local.x, grav_local.y, grav_local.z),
            ff = format!("({:.3},{:.3},{:.3})", ff_x, ff_y, ff_z),
            fb = format!("({:.3},{:.3},{:.3})", fb_x, fb_y, fb_z),
            out = format!("({:.3},{:.3},{:.3})", raw_input.thrust_lateral, raw_input.thrust_vertical, raw_input.thrust_forward),
            limiter = format!("{:.2}", limiter),
            "hover_computer ACTIVE"
        );
    }
}

/// Phase 1: All functional blocks with SignalPublisher write their state to channels.
/// Values accumulate in pending_values, merged after all publishers write.
fn signal_publish(
    mut channels: ResMut<SignalChannelTable>,
    publishers: Query<(&FunctionalBlockRef, &SignalPublisher, Option<&ReactorState>, Option<&CruiseDriveState>, Option<&MechanicalState>)>,
    mut incoming: ResMut<IncomingSignalBuffer>,
    pilot_mode: Res<SeatedState>,
    mut active_seat: ResMut<ActiveSeatEntity>,
    raw_input: Res<RawPilotInput>,
    seat_query: Query<&SeatChannelMapping>,
) {
    channels.clear_pending();

    // Cross-shard signals received via QUIC.
    for (name, value) in incoming.signals.drain(..) {
        channels.push_pending(&name, value);
    }

    // Seat control publishing: map pilot input to signal channels via SeatChannelMapping.
    if let Some(seat_entity) = active_seat.0 {
        if let Ok(mapping) = seat_query.get(seat_entity) {
            // Diagnostic: log the active seat's channel bindings on every seat change.
            static LAST_SEAT_ENTITY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
            let seat_id_bits = seat_entity.to_bits() as u64;
            if LAST_SEAT_ENTITY.swap(seat_id_bits, std::sync::atomic::Ordering::Relaxed) != seat_id_bits {
                let names: Vec<_> = mapping.bindings.iter().map(|b| {
                    let ch_name = channels.name_for_id(b.channel_id).unwrap_or("?");
                    format!("{:?}→{}", b.control, ch_name)
                }).collect();
                info!(
                    seat_entity = ?seat_entity,
                    bindings = names.len(),
                    channels = ?names,
                    "active seat channel dump",
                );
            }
            for binding in &mapping.bindings {
                use voxeldust_core::signal::components::SeatControl;
                let value = if pilot_mode.seated {
                    match binding.control {
                        SeatControl::ThrustForward   => raw_input.thrust_forward.max(0.0),
                        SeatControl::ThrustReverse   => (-raw_input.thrust_forward).max(0.0),
                        SeatControl::ThrustRight     => raw_input.thrust_lateral.max(0.0),
                        SeatControl::ThrustLeft      => (-raw_input.thrust_lateral).max(0.0),
                        SeatControl::ThrustUp        => raw_input.thrust_vertical.max(0.0),
                        SeatControl::ThrustDown      => (-raw_input.thrust_vertical).max(0.0),
                        SeatControl::TorqueYawCW     => raw_input.torque_yaw.max(0.0),
                        SeatControl::TorqueYawCCW    => (-raw_input.torque_yaw).max(0.0),
                        SeatControl::TorquePitchUp   => raw_input.torque_pitch.max(0.0),
                        SeatControl::TorquePitchDown => (-raw_input.torque_pitch).max(0.0),
                        SeatControl::ThrustLimiter   => raw_input.thrust_limiter,
                        SeatControl::TorqueRollCW    => raw_input.roll_cw,
                        SeatControl::TorqueRollCCW   => raw_input.roll_ccw,
                        SeatControl::Cruise          => if raw_input.cruise_active && !raw_input.in_atmosphere { 1.0 } else { 0.0 },
                        SeatControl::AtmoComp        => if raw_input.atmo_comp_active { 1.0 } else { 0.0 },
                    }
                } else {
                    0.0 // Zero-publish when pilot exits seat.
                };
                channels.push_pending_id(binding.channel_id, signal::SignalValue::Float(value));
            }
        }
        // Clear seat tracking after zero-publish (pilot exited).
        if !pilot_mode.seated {
            active_seat.0 = None;
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
                // MechanicalState reads Angle, Extension, Speed, Active.
                if let Some(ref mut ms) = mech_state {
                    match binding.property {
                        SignalProperty::Angle | SignalProperty::Extension => {
                            ms.target = ch.value.as_f32();
                        }
                        SignalProperty::Speed => {
                            ms.max_speed = ch.value.as_f32();
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
    raw_input: Res<RawPilotInput>,
    ship_props: Res<ShipProps>,
    ship_com: Res<ShipCenterOfMass>,
    exterior: Res<ShipExterior>,
    scene: Res<SceneCache>,
    cached_sys: Res<CachedSystemParams>,
    autopilot: Res<AutopilotState>,
) {

    let limiter = raw_input.thrust_limiter as f64;

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

    // Capture pilot + flight computer rotation intent as [-1, 1] rate command.
    // This is what the system shard expects — not the physical torque in N·m.
    pilot_acc.rotation_command = DVec3::new(
        raw_input.torque_pitch as f64,                          // X = pitch
        raw_input.torque_yaw as f64,                            // Y = yaw
        (raw_input.roll_ccw - raw_input.roll_cw) as f64,        // Z = roll (ccw-cw: positive Z = CW from pilot's view)
    );

    // Gravity compensation (hover + damping) is computed in the system-shard's
    // physics_integrate, where position/rotation/gravity are all current-tick
    // authoritative. This avoids rotation-mismatch and stale-data issues that
    // caused atmosphere catapulting when computed here.
}

/// Read actual joint positions from Rapier into MechanicalState.
/// Runs BEFORE signal_publish so that current values are published as feedback.
fn read_mechanical_state(
    rapier: Res<RapierContext>,
    mut mechanicals: Query<&mut MechanicalState>,
    sub_grids: Res<SubGridRegistry>,
) {
    for mut state in &mut mechanicals {
        let sg = match sub_grids.grids.get(&state.child_grid_id) {
            Some(sg) => sg,
            None => continue,
        };
        let joint = match rapier.impulse_joint_set.get(sg.joint_handle) {
            Some(j) => j,
            None => continue,
        };

        // Read body rotations for joint angle computation.
        let rot1 = rapier.rigid_body_set.get(joint.body1)
            .map(|b| *b.rotation())
            .unwrap_or_default();
        let rot2 = rapier.rigid_body_set.get(joint.body2)
            .map(|b| *b.rotation())
            .unwrap_or_default();

        let current = match state.joint_type {
            block::JointType::Revolute => {
                joint.data.as_revolute()
                    .map(|r| r.angle(&rot1, &rot2).to_degrees())
                    .unwrap_or(0.0)
            }
            block::JointType::Prismatic => {
                // Compute translation along joint axis from body positions.
                let pos1 = rapier.rigid_body_set.get(joint.body1)
                    .map(|b| *b.translation()).unwrap_or_default();
                let pos2 = rapier.rigid_body_set.get(joint.body2)
                    .map(|b| *b.translation()).unwrap_or_default();
                let diff = pos2 - pos1;
                // Use the local anchor difference as a proxy for extension.
                let anchor1 = joint.data.local_anchor1();
                let anchor2 = joint.data.local_anchor2();
                let dir = (anchor1 - anchor2).normalize();
                diff.dot(&dir).abs()
            }
        };
        state.current = current;

        // Stuck detection: if error hasn't decreased in 10 ticks, set Blocked.
        let error = (state.target - state.current).abs();
        if error < 0.5 {
            state.status = MechanicalStatus::Idle;
            state.stuck_ticks = 0;
        } else if error < state.prev_error - 0.01 {
            state.status = MechanicalStatus::Moving;
            state.stuck_ticks = 0;
        } else {
            state.stuck_ticks = state.stuck_ticks.saturating_add(1);
            if state.stuck_ticks > 10 {
                state.status = MechanicalStatus::Blocked;
            } else {
                state.status = MechanicalStatus::Moving;
            }
        }
        state.prev_error = error;
    }
}

/// Apply Rapier joint motor targets from MechanicalState.
/// Uses ease-in-out speed profile: motor velocity is scaled by a smoothstep
/// function of the distance to target, giving natural acceleration/deceleration.
fn apply_mechanical_motors(
    mut rapier: ResMut<RapierContext>,
    mechanicals: Query<(&MechanicalState, Option<&PoweredBy>)>,
    reactors: Query<&ReactorState>,
    block_index: Res<FunctionalBlockIndex>,
    sub_grids: Res<SubGridRegistry>,
) {
    for (state, powered_by) in &mechanicals {
        // Power check: no power = no motion.
        let power_ratio = powered_by
            .and_then(|pb| block_index.0.get(&pb.reactor_pos))
            .and_then(|&e| reactors.get(e).ok())
            .and_then(|rs| rs.circuits.iter().find(|c| c.name == powered_by.unwrap().circuit))
            .map(|c| c.power_ratio as f64)
            .unwrap_or(1.0); // default to 1.0 if no PoweredBy (unpowered mounts still work for now)

        if power_ratio < 0.001 { continue; }

        let sg = match sub_grids.grids.get(&state.child_grid_id) {
            Some(sg) => sg,
            None => continue,
        };
        let joint = match rapier.impulse_joint_set.get_mut(sg.joint_handle) {
            Some(j) => j,
            None => continue,
        };

        let effective_force = state.max_force * power_ratio;

        // Ease-in-out speed profile using smoothstep:
        // smoothstep(t) = t² × (3 - 2t) maps [0,1] → [0,1] with zero derivative at both ends.
        // The ramp threshold defines the angular/linear distance over which speed ramps up/down.
        // This creates natural acceleration/deceleration: slow near target, full speed in the middle.
        let compute_eased_velocity = |error: f32, max_speed: f32, max_range: f64| -> f32 {
            let error_abs = error.abs();
            // Ramp distance: 10% of max range, minimum 0.5 (degrees or meters).
            let ramp_dist = (max_range as f32 * 0.1).max(0.5);
            let t = (error_abs / ramp_dist).min(1.0);
            let ease = t * t * (3.0 - 2.0 * t); // smoothstep
            error.signum() * max_speed * ease
        };

        match state.joint_type {
            block::JointType::Revolute => {
                if let Some(data) = joint.data.as_revolute_mut() {
                    let error_deg = state.target - state.current;
                    let vel_deg = compute_eased_velocity(error_deg, state.max_speed, state.max_range);
                    let vel_rad = vel_deg.to_radians();
                    data.set_motor_velocity(vel_rad, effective_force as f32);
                }
            }
            block::JointType::Prismatic => {
                if let Some(data) = joint.data.as_prismatic_mut() {
                    let target_clamped = state.target.clamp(0.0, state.max_range as f32);
                    let error = target_clamped - state.current;
                    let vel = compute_eased_velocity(error, state.max_speed, state.max_range);
                    data.set_motor_velocity(vel, effective_force as f32);
                }
            }
        }
    }
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
    let player_rb = RigidBodyBuilder::dynamic()
        .translation(vector![spawn_pos.x, spawn_pos.y, spawn_pos.z])
        .lock_rotations()
        .build();
    let player_handle = rigid_body_set.insert(player_rb);
    let player_collider = ColliderBuilder::capsule_y(0.6, 0.3).build();
    collider_set.insert_with_parent(player_collider, player_handle, &mut rigid_body_set);

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
    app.insert_resource(MechanicalDirtyGrids::default());
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
    app.insert_resource(RawPilotInput { thrust_limiter: 0.75, ..Default::default() });
    app.insert_resource(ShipCenterOfMass::default());
    // PowerNetworkTable removed — power is wireless via ReactorState circuits.
    app.insert_resource(ActiveSeatEntity::default());
    app.insert_resource(AutopilotState::default());
    app.insert_resource(HoverState::default());
    app.insert_resource(WarpTargetState::default());
    app.insert_resource(ConnectedPlayer::default());
    app.insert_resource(AtmosphereState::default());
    app.insert_resource(LandingState::default());
    app.insert_resource(PendingMessages::default());
    app.insert_resource(PlayerBodyHandle(player_handle));
    app.insert_resource(PlayerPosition(spawn_pos));
    app.insert_resource(PlayerYaw(0.0));
    app.insert_resource(SeatedState::default());
    app.insert_resource(GravityEnabled(true));
    app.insert_resource(InputActions::default());
    app.insert_resource(ecs::TickCounter::default());
    app.insert_resource(BlockEditQueue::default());
    app.insert_resource(AggregationDirty(false));
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
        (produce_player_edits, apply_block_edits, bevy_ecs::schedule::ApplyDeferred, process_entity_ops, process_sub_block_edits, process_mechanical_edits, aggregate_ship_properties, rebuild_reactor_consumers)
            .chain()
            .in_set(ShipSet::BlockEdit),
    );

    // Signal pipeline: publish → evaluate → subscribe → clear dirty.
    app.add_systems(
        Update,
        (read_mechanical_state, flight_computer_damping, hover_computer, signal_publish, signal_evaluate, signal_subscribe, apply_mechanical_motors, compute_power_budget, compute_ship_thrust, signal_clear_dirty)
            .chain()
            .in_set(ShipSet::Signal),
    );

    // Physics.
    app.add_systems(
        Update,
        (tick_counter, physics_step).chain().in_set(ShipSet::Physics),
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
