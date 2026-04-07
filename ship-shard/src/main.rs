use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
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
    SeatChannelMapping, PublishBinding, SubscribeBinding, SignalProperty,
};
use voxeldust_core::shard_message::AutopilotSnapshotData;
use voxeldust_core::client_message::{
    BlockEditData, CelestialBodyData, ChunkDeltaData, JoinResponseData, LightingData,
    PlayerSnapshotData, ServerMsg, WorldStateData,
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
}

/// Autopilot state tracked by the ship shard.
#[derive(Resource, Default)]
struct AutopilotState {
    target_body_id: Option<u32>,
    pending_cmd: Option<(u32, u8)>,
    snapshot: Option<AutopilotSnapshotData>,
}

/// Warp targeting state.
#[derive(Resource, Default)]
struct WarpTargetState {
    target_star_index: Option<u32>,
    pending_cmd: Option<u32>,
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

/// Whether the player is in pilot mode.
#[derive(Resource)]
struct PilotMode(bool);

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
    tcp_stream: Arc<tokio::sync::Mutex<tokio::net::TcpStream>>,
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
            tcp_stream: conn.tcp_stream.clone(),
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
    mut autopilot: ResMut<AutopilotState>,
    mut connected: ResMut<ConnectedPlayer>,
    mut landing: ResMut<LandingState>,
    mut rapier: ResMut<RapierContext>,
    mut warp: ResMut<WarpTargetState>,
    body_handle: Res<PlayerBodyHandle>,
    mut player_pos: ResMut<PlayerPosition>,
    mut pilot_mode: ResMut<PilotMode>,
    tick: Res<ecs::TickCounter>,
    mut incoming_signals: ResMut<IncomingSignalBuffer>,
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
                autopilot.snapshot = data.autopilot;

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
                pilot_mode.0 = false;
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
                pilot_mode.0 = false;

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

                if data.new_host_shard_type != 3 {
                    warp.target_star_index = None;
                }

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
                // Distance check for ShortRange signals.
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

                // Buffer for processing in signal_publish next tick.
                incoming_signals.signals.push((data.channel_name.clone(), value));
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
        let tcp_stream = event.tcp_stream.clone();
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
            let mut stream = tcp_stream.lock().await;
            let _ = client_listener::send_tcp_msg(&mut *stream, &jr).await;

            // Send all chunk snapshots immediately after JoinResponse.
            for snapshot_msg in &chunk_snapshots {
                let _ = client_listener::send_tcp_msg(&mut *stream, snapshot_msg).await;
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
    pilot_mode: Res<PilotMode>,
    mut pilot_acc: ResMut<PilotAccumulator>,
    mut rapier: ResMut<RapierContext>,
    body_handle: Res<PlayerBodyHandle>,
    exterior: Res<ShipExterior>,
    scene: Res<SceneCache>,
    cached_sys: Res<CachedSystemParams>,
    ship_props: Res<ShipProps>,
    mut autopilot: ResMut<AutopilotState>,
    mut warp: ResMut<WarpTargetState>,
    config: Res<ShipConfig>,
    cached_galaxy: Res<CachedGalaxyMap>,
) {
    for event in events.read() {
        let input = &event.input;
        actions.previous = actions.current;
        actions.current = input.action;
        player_yaw.0 = input.look_yaw;

        if pilot_mode.0 {
            // Autopilot toggle (T key = action 4, rising edge).
            let autopilot_pressed = input.action == 4 && actions.previous != 4;
            if autopilot_pressed {
                if autopilot.target_body_id.is_some() {
                    autopilot.target_body_id = None;
                    autopilot.pending_cmd = Some((0xFFFFFFFF, 0));
                    info!("autopilot disengage requested");
                } else {
                    let ship_fwd = exterior.rotation * DVec3::NEG_Z;
                    let mut best_body_id: Option<u32> = None;
                    let mut best_dot = 0.9;
                    for body in &scene.bodies {
                        if body.body_id == 0 {
                            continue;
                        }
                        let to_body = (body.position - exterior.position).normalize_or_zero();
                        let d = ship_fwd.dot(to_body);
                        if d > best_dot {
                            best_dot = d;
                            best_body_id = Some(body.body_id);
                        }
                    }
                    if let Some(body_id) = best_body_id {
                        autopilot.target_body_id = Some(body_id);
                        autopilot.pending_cmd = Some((body_id, input.speed_tier));
                        info!(body_id, "autopilot engage requested");
                    }
                }
            }

            // Warp targeting (G key = action 6).
            let warp_pressed = input.action == 6 && actions.previous != 6;
            if warp_pressed {
                let ship_fwd = exterior.rotation * DVec3::NEG_Z;
                if let Some(ref sys) = cached_sys.0 {
                    if let Some(ref galaxy_map) = cached_galaxy.0 {
                        let current_star = galaxy_map
                            .stars
                            .iter()
                            .find(|s| s.system_seed == sys.system_seed);
                        if let Some(cur) = current_star {
                            let cur_pos = cur.position;

                            if let Some(current_target) = warp.target_star_index {
                                // Cycle to next-best aligned star.
                                let current_dot = galaxy_map
                                    .stars
                                    .iter()
                                    .find(|s| s.index == current_target)
                                    .map(|s| {
                                        ship_fwd.dot((s.position - cur_pos).normalize())
                                    })
                                    .unwrap_or(1.0);

                                let mut best: Option<(u32, f64)> = None;
                                for star in &galaxy_map.stars {
                                    if star.index == cur.index
                                        || star.index == current_target
                                    {
                                        continue;
                                    }
                                    let dir = (star.position - cur_pos).normalize();
                                    let dot = ship_fwd.dot(dir);
                                    if dot < current_dot && dot > 0.3 {
                                        if dot > best.map(|b| b.1).unwrap_or(0.3) {
                                            best = Some((star.index, dot));
                                        }
                                    }
                                }
                                if best.is_none() {
                                    for star in &galaxy_map.stars {
                                        if star.index == cur.index {
                                            continue;
                                        }
                                        let dir = (star.position - cur_pos).normalize();
                                        let dot = ship_fwd.dot(dir);
                                        if dot > best.map(|b| b.1).unwrap_or(0.3) {
                                            best = Some((star.index, dot));
                                        }
                                    }
                                }
                                if let Some((idx, _)) = best {
                                    warp.target_star_index = Some(idx);
                                    info!(target_star = idx, "warp target cycled");
                                }
                            } else {
                                let mut best: Option<(u32, f64)> = None;
                                for star in &galaxy_map.stars {
                                    if star.index == cur.index {
                                        continue;
                                    }
                                    let dir = (star.position - cur_pos).normalize();
                                    let alignment = ship_fwd.dot(dir);
                                    if alignment > best.map(|b| b.1).unwrap_or(0.3) {
                                        best = Some((star.index, alignment));
                                    }
                                }
                                if let Some((target_index, _)) = best {
                                    warp.target_star_index = Some(target_index);
                                    info!(target_star = target_index, "warp target selected");
                                }
                            }
                        }
                    }
                }
            }

            // Warp confirm (Enter = action 7).
            let warp_confirm = input.action == 7 && actions.previous != 7;
            if warp_confirm {
                if let Some(target) = warp.target_star_index {
                    warp.pending_cmd = Some(target);
                    info!(target_star = target, "warp engage confirmed");
                }
            }

            // Cancel autopilot on manual WASD input.
            if autopilot.target_body_id.is_some() {
                let has_manual = input.movement[0].abs() > 0.01
                    || input.movement[1].abs() > 0.01
                    || input.movement[2].abs() > 0.01;
                if has_manual {
                    autopilot.target_body_id = None;
                    autopilot.pending_cmd = Some((0xFFFFFFFF, 0));
                    info!("autopilot cancelled by manual input");
                }
            }

            // Pilot thrust computation.
            {
                let in_atmosphere = if let Some(ref sys) = cached_sys.0 {
                    scene.bodies.iter().any(|b| {
                        if b.body_id == 0 {
                            return false;
                        }
                        let pi = (b.body_id - 1) as usize;
                        if pi >= sys.planets.len() {
                            return false;
                        }
                        let alt = (exterior.position - b.position).length()
                            - sys.planets[pi].radius_m;
                        alt < sys.planets[pi].atmosphere.atmosphere_height
                            && sys.planets[pi].atmosphere.has_atmosphere
                    })
                } else {
                    false
                };
                let effective_tier =
                    autopilot::effective_tier(input.speed_tier, in_atmosphere, false);
                let et = autopilot::engine_tier(effective_tier);
                let tier_thrust = et.thrust_force_n * ship_props.0.thrust_multiplier;
                pilot_acc.thrust = DVec3::new(
                    input.movement[0] as f64 * tier_thrust,
                    input.movement[1] as f64 * tier_thrust,
                    -input.movement[2] as f64 * tier_thrust,
                );
                pilot_acc.torque = DVec3::new(
                    input.look_pitch as f64,
                    input.look_yaw as f64,
                    0.0,
                );

                // Gravity compensation in atmosphere.
                let engines_off = input.action == 5;
                if in_atmosphere && autopilot.target_body_id.is_none() && !engines_off {
                    if let Some(ref sys) = cached_sys.0 {
                        for b in &scene.bodies {
                            if b.body_id == 0 {
                                continue;
                            }
                            let pi = (b.body_id - 1) as usize;
                            if pi >= sys.planets.len() {
                                continue;
                            }
                            let dist = (exterior.position - b.position).length();
                            let alt = dist - sys.planets[pi].radius_m;
                            if alt < sys.planets[pi].atmosphere.atmosphere_height {
                                let grav_mag = sys.planets[pi].gm / (dist * dist);
                                let grav_dir = (b.position - exterior.position).normalize();
                                let grav_world = grav_dir * grav_mag;

                                if input.movement[1].abs() < 0.01 {
                                    let grav_local =
                                        exterior.rotation.inverse() * (-grav_world);
                                    let hover = grav_local * ship_props.0.mass_kg;
                                    pilot_acc.thrust += DVec3::new(hover.x, hover.y, hover.z);
                                }

                                let has_input = input.movement[0].abs() > 0.01
                                    || input.movement[1].abs() > 0.01
                                    || input.movement[2].abs() > 0.01;
                                if !has_input {
                                    let max_accel =
                                        ship_props.0.engine_acceleration(effective_tier);
                                    let damping_rate = (max_accel * 0.1).min(5.0);
                                    let vel_local =
                                        exterior.rotation.inverse() * exterior.velocity;
                                    let damping =
                                        -vel_local * damping_rate * ship_props.0.mass_kg;
                                    let max_damp = tier_thrust * 0.5;
                                    let damping_clamped = damping.clamp_length_max(max_damp);
                                    pilot_acc.thrust += DVec3::new(
                                        damping_clamped.x,
                                        damping_clamped.y,
                                        damping_clamped.z,
                                    );
                                }
                                break;
                            }
                        }
                    }
                }
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
    pilot_mode: Res<PilotMode>,
    pilot_acc: Res<PilotAccumulator>,
    mut autopilot: ResMut<AutopilotState>,
    mut warp: ResMut<WarpTargetState>,
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

    // Always send pending autopilot command.
    if let Some((target, tier)) = autopilot.pending_cmd.take() {
        let ap_msg = ShardMsg::AutopilotCommand(AutopilotCommandData {
            ship_id: config.ship_id,
            target_body_id: target,
            speed_tier: tier,
            autopilot_mode: 0,
        });
        let _ = bridge.quic_send_tx.try_send((host_id, addr, ap_msg));
    }

    // Always send pending warp command.
    if let Some(target_star) = warp.pending_cmd.take() {
        let warp_msg = ShardMsg::WarpAutopilotCommand(WarpAutopilotCommandData {
            ship_id: config.ship_id,
            target_star_index: target_star,
            galaxy_seed: config.galaxy_seed,
        });
        let _ = bridge.quic_send_tx.try_send((host_id, addr, warp_msg));
    }

    // Thrust/torque only when piloting.
    if pilot_mode.0 {
        let msg = ShardMsg::ShipControlInput(ShipControlInput {
            ship_id: config.ship_id,
            thrust: pilot_acc.thrust,
            torque: pilot_acc.torque,
            braking: false,
            tick: tick.0,
        });
        let _ = bridge.quic_send_tx.try_send((host_id, addr, msg));
    }
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
    pilot_mode: Res<PilotMode>,
    warp: Res<WarpTargetState>,
    autopilot: Res<AutopilotState>,
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
            grounded: !pilot_mode.0,
            health: 100.0,
            shield: 100.0,
        }],
        bodies,
        ships: vec![],
        lighting,
        game_time: scene.game_time,
        warp_target_star_index: warp.target_star_index.unwrap_or(0xFFFFFFFF),
        autopilot: autopilot.snapshot.clone(),
    });
    let _ = bridge.broadcast_tx.try_send(ws);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

fn log_state(
    player_pos: Res<PlayerPosition>,
    pilot_mode: Res<PilotMode>,
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
        piloting = pilot_mode.0,
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
fn apply_config_updates(
    mut events: MessageReader<ConfigUpdateMsg>,
    block_index: Res<FunctionalBlockIndex>,
    mut commands: Commands,
) {
    for event in events.read() {
        let update = &event.update;
        let entity = match block_index.0.get(&update.block_pos) {
            Some(&e) => e,
            None => {
                warn!(block = ?update.block_pos, "config update for nonexistent block");
                continue;
            }
        };

        // Replace signal components on the entity.
        // Using insert() overwrites existing components.
        if !update.publish_bindings.is_empty() {
            commands.entity(entity).insert(SignalPublisher {
                bindings: update.publish_bindings.clone(),
            });
        }
        if !update.subscribe_bindings.is_empty() {
            commands.entity(entity).insert(SignalSubscriber {
                bindings: update.subscribe_bindings.clone(),
            });
        }
        if !update.converter_rules.is_empty() {
            commands.entity(entity).insert(SignalConverterConfig {
                rules: update.converter_rules.clone(),
            });
        }
        if !update.seat_mappings.is_empty() {
            commands.entity(entity).insert(SeatChannelMapping {
                bindings: update.seat_mappings.clone(),
            });
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
    mut pilot_mode: ResMut<PilotMode>,
    mut rapier: ResMut<RapierContext>,
    body_handle: Res<PlayerBodyHandle>,
    bridge: Res<NetworkBridge>,
    channels: Res<SignalChannelTable>,
    pub_query: Query<&SignalPublisher>,
    sub_query: Query<&SignalSubscriber>,
    converter_query: Query<&SignalConverterConfig>,
    seat_query: Query<&SeatChannelMapping>,
) {
    if connected.session.is_none() || connected.handoff_pending {
        return;
    }
    let session = connected.session.unwrap();

    for event in events.read() {
        let edit = &event.edit;
        let eye = glam::Vec3::new(edit.eye.x as f32, edit.eye.y as f32, edit.eye.z as f32);
        let look = glam::Vec3::new(edit.look.x as f32, edit.look.y as f32, edit.look.z as f32);

        // Validate: player eye position should be close to their actual position
        // (anti-cheat: don't allow editing from arbitrary positions).
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

        match edit.action {
            1 => {
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
            2 => {
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
            action @ 3..=9 => {
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
                                    // Toggle pilot mode.
                                    pilot_mode.0 = !pilot_mode.0;
                                    if pilot_mode.0 {
                                        if let Some(body) = rapier.rigid_body_set.get_mut(body_handle.0) {
                                            body.set_linvel(vector![0.0, 0.0, 0.0], true);
                                        }
                                        info!("entered pilot mode via seat at {:?}", hit.world_pos);
                                    } else {
                                        info!("exited pilot mode");
                                    }
                                }
                                _ => {
                                    // Future phases: dispatch to kind-specific handlers.
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
            8 => {
                // F key: universal config UI — open config for any functional block.
                if let Some(&entity) = block_index.0.get(&hit.world_pos) {
                    let block_id = grid.0.get_block(
                        hit.world_pos.x, hit.world_pos.y, hit.world_pos.z,
                    );
                    if let Some(kind) = registry.0.functional_kind(block_id) {
                        // Build config snapshot from entity components.
                        let config = build_config_snapshot(
                            entity, hit.world_pos, block_id, kind,
                            &pub_query, &sub_query, &converter_query, &seat_query,
                            &channels,
                        );
                        // Send to client via TCP.
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
) -> voxeldust_core::signal::config::BlockSignalConfig {
    let publish_bindings = pub_query.get(entity)
        .map(|p| p.bindings.clone())
        .unwrap_or_default();
    let subscribe_bindings = sub_query.get(entity)
        .map(|s| s.bindings.clone())
        .unwrap_or_default();
    let converter_rules = converter_query.get(entity)
        .map(|c| c.rules.clone())
        .unwrap_or_default();
    let seat_mappings = seat_query.get(entity)
        .map(|s| s.bindings.clone())
        .unwrap_or_default();

    let mut available_channels: Vec<String> = Vec::new();
    // Collect all channel names from the channel table for the dropdown.
    for name in channels.channel_names() {
        available_channels.push(name.to_string());
    }
    available_channels.sort();

    voxeldust_core::signal::config::BlockSignalConfig {
        block_pos: pos,
        block_type: block_id.as_u16(),
        kind: kind as u8,
        publish_bindings,
        subscribe_bindings,
        converter_rules,
        seat_mappings,
        available_channels,
    }
}

/// Rebuild a single chunk's Rapier collider from the current ShipGrid state.
fn rebuild_chunk_collider(
    grid: &ShipGrid,
    chunk_key: glam::IVec3,
    registry: &BlockRegistry,
    rapier: &mut RapierContext,
    collider_handles: &mut ChunkColliderHandles,
) {
    // Remove old collider for this chunk.
    if let Some(old_handle) = collider_handles.0.remove(&chunk_key) {
        rapier.collider_set.remove(
            old_handle,
            &mut rapier.island_manager,
            &mut rapier.rigid_body_set,
            true,
        );
    }
    // Build and insert new collider.
    if let Some(collider) = build_chunk_collider(grid, chunk_key, registry) {
        let handle = rapier.collider_set.insert(collider);
        collider_handles.0.insert(chunk_key, handle);
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
        rebuild_chunk_collider(&grid.0, chunk_key, &registry.0, &mut rapier, &mut collider_handles);
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
                        add_default_signal_bindings(&mut entity_cmds, kind);
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
                    add_default_signal_bindings(&mut entity_cmds, kind);

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

/// Aggregation system: recomputes ship physical properties from block composition.
/// Runs after entity lifecycle ops, only when the dirty flag is set.
fn aggregate_ship_properties(
    mut dirty: ResMut<AggregationDirty>,
    grid: Res<ShipGridResource>,
    registry: Res<BlockRegistryResource>,
    mut ship_props: ResMut<ShipProps>,
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

/// Add default signal publisher/subscriber components to a functional block entity
/// based on its kind. Players can reconfigure these via the block config UI.
fn add_default_signal_bindings(
    entity_cmds: &mut bevy_ecs::system::EntityCommands,
    kind: FunctionalBlockKind,
) {
    match kind {
        FunctionalBlockKind::Thruster => {
            // Thrusters subscribe to a throttle channel.
            entity_cmds.insert(SignalSubscriber {
                bindings: vec![SubscribeBinding {
                    channel_name: "thrust-forward".into(),
                    property: SignalProperty::Throttle,
                }],
            });
        }
        FunctionalBlockKind::Reactor => {
            // Reactors publish their power level.
            entity_cmds.insert(SignalPublisher {
                bindings: vec![PublishBinding {
                    channel_name: "power".into(),
                    property: SignalProperty::Level,
                }],
            });
        }
        FunctionalBlockKind::Seat => {
            // Seats get the full channel mapping for pilot controls.
            entity_cmds.insert(SeatChannelMapping::default());
        }
        FunctionalBlockKind::Sensor => {
            // Sensors publish their readings.
            entity_cmds.insert(SignalPublisher {
                bindings: vec![PublishBinding {
                    channel_name: "sensor".into(),
                    property: SignalProperty::Active,
                }],
            });
        }
        FunctionalBlockKind::SignalConverter => {
            // Signal converters start with an empty rule set (player configures).
            entity_cmds.insert(SignalConverterConfig::default());
        }
        _ => {
            // Other kinds: no default signal bindings.
            // Players can add bindings via the config UI.
        }
    }
}

// ---------------------------------------------------------------------------
// Signal pipeline systems
// ---------------------------------------------------------------------------

/// Phase 1: All functional blocks with SignalPublisher write their state to channels.
/// Values accumulate in pending_values, merged after all publishers write.
fn signal_publish(
    mut channels: ResMut<SignalChannelTable>,
    publishers: Query<(&FunctionalBlockRef, &SignalPublisher)>,
    mut incoming: ResMut<IncomingSignalBuffer>,
) {
    channels.clear_pending();

    // First: inject any cross-shard signals received via QUIC.
    for (name, value) in incoming.signals.drain(..) {
        channels.push_pending(&name, value);
    }

    for (_block_ref, publisher) in &publishers {
        for binding in &publisher.bindings {
            // For now, publish a default value based on the property type.
            // Future phases will read actual block state (reactor power level,
            // sensor readings, etc.) from kind-specific components.
            let value = match binding.property {
                SignalProperty::Active => signal::SignalValue::Bool(true),
                SignalProperty::Throttle => signal::SignalValue::Float(0.0),
                SignalProperty::Level => signal::SignalValue::Float(1.0),
                SignalProperty::Pressure => signal::SignalValue::Float(101.3),
                _ => signal::SignalValue::Float(0.0),
            };
            channels.push_pending(&binding.channel_name, value);
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
            let (input_value, is_dirty) = match channels.get(&rule.input_channel) {
                Some(ch) => (ch.value, ch.dirty),
                None => continue,
            };

            // Skip if input channel not dirty (lazy evaluation for 100K scale).
            if !is_dirty && !matches!(rule.condition, signal::SignalCondition::Always) {
                continue;
            }

            if rule.condition.evaluate(input_value, is_dirty) {
                let output = rule.expression.compute(input_value);
                channels.publish_direct(&rule.output_channel, output);
            }
        }
    }
}

/// Phase 3: Functional blocks with SignalSubscriber read channels and act.
/// For now, actions are logged. Future phases will apply to block-specific state
/// (thruster throttle, piston extension, rotor angle, etc.).
fn signal_subscribe(
    channels: Res<SignalChannelTable>,
    subscribers: Query<(&FunctionalBlockRef, &SignalSubscriber)>,
) {
    for (block_ref, subscriber) in &subscribers {
        for binding in &subscriber.bindings {
            if let Some(ch) = channels.get(&binding.channel_name) {
                if ch.dirty {
                    // Future: apply_signal_to_block(block_ref, binding.property, ch.value)
                    // For now, the signal value is available for future per-kind systems
                    // to read from the channel table.
                    let _ = (block_ref, &binding.property, ch.value);
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
/// Runs in the Send set (after signal processing, before broadcast).
fn signal_broadcast_remote(
    channels: Res<SignalChannelTable>,
    exterior: Res<ShipExterior>,
    config: Res<ShipConfig>,
    bridge: Res<NetworkBridge>,
) {
    let remote_signals = channels.drain_remote_dirty();
    if remote_signals.is_empty() {
        return;
    }

    for (channel_name, value, scope) in &remote_signals {
        let (scope_code, range_m) = match scope {
            signal::SignalScope::ShortRange { range_m } => (1u8, *range_m),
            signal::SignalScope::LongRange => (2u8, 0.0),
            signal::SignalScope::Local => continue, // shouldn't happen (filtered)
        };

        let (value_type, value_data) = match value {
            signal::SignalValue::Bool(b) => (0u8, if *b { 1.0f32 } else { 0.0 }),
            signal::SignalValue::Float(f) => (1u8, *f),
            signal::SignalValue::State(s) => (2u8, *s as f32),
        };

        let msg = voxeldust_core::shard_message::ShardMsg::SignalBroadcast(
            voxeldust_core::shard_message::SignalBroadcastData {
                source_shard_id: config.shard_id.0,
                channel_name: channel_name.clone(),
                value_type,
                value_data,
                scope: scope_code,
                range_m,
                source_position: exterior.position,
            },
        );

        // For ShortRange: send to all known shards (they distance-check on receive).
        // For LongRange: send to the host (system) shard for relay.
        if scope_code == 2 {
            // LongRange → system shard relay.
            if let Some(host_id) = config.host_shard_id {
                if let Ok(reg) = bridge.peer_registry.try_read() {
                    if let Some(addr) = reg.quic_addr(host_id) {
                        let _ = bridge.quic_send_tx.try_send((host_id, addr, msg));
                    }
                }
            }
        } else {
            // ShortRange → broadcast to all known peer shards.
            if let Ok(reg) = bridge.peer_registry.try_read() {
                for peer in reg.all() {
                    if peer.id == config.shard_id {
                        continue; // don't send to self
                    }
                    if let Some(addr) = reg.quic_addr(peer.id) {
                        let _ = bridge.quic_send_tx.try_send((peer.id, addr, msg.clone()));
                    }
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
        Some(saved_grid) => {
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
            info!("built starter ship and saved to persistence");
            starter
        }
    };

    // Generate Rapier colliders from each chunk's solid blocks.
    let mut chunk_collider_handles = std::collections::HashMap::new();
    for chunk_key in grid.chunk_keys() {
        if let Some(collider) = build_chunk_collider(&grid, chunk_key, &registry) {
            let handle = collider_set.insert(collider);
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
    app.insert_resource(AutopilotState::default());
    app.insert_resource(WarpTargetState::default());
    app.insert_resource(ConnectedPlayer::default());
    app.insert_resource(AtmosphereState::default());
    app.insert_resource(LandingState::default());
    app.insert_resource(PendingMessages::default());
    app.insert_resource(PlayerBodyHandle(player_handle));
    app.insert_resource(PlayerPosition(spawn_pos));
    app.insert_resource(PlayerYaw(0.0));
    app.insert_resource(PilotMode(false));
    app.insert_resource(GravityEnabled(true));
    app.insert_resource(InputActions::default());
    app.insert_resource(ecs::TickCounter::default());
    app.insert_resource(BlockEditQueue::default());
    app.insert_resource(AggregationDirty(false));
    app.insert_resource(SignalChannelTable::new());
    app.insert_resource(IncomingSignalBuffer::default());
    app.insert_resource(ship_persistence);

    // Messages.
    app.add_message::<ClientConnectedMsg>();
    app.add_message::<PlayerInputMsg>();
    app.add_message::<BlockEditMsg>();
    app.add_message::<ConfigUpdateMsg>();

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
        (drain_connects, drain_input, drain_block_edits, drain_config_updates, drain_quic).in_set(ShipSet::Bridge),
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
        (produce_player_edits, apply_block_edits, bevy_ecs::schedule::ApplyDeferred, process_entity_ops, aggregate_ship_properties)
            .chain()
            .in_set(ShipSet::BlockEdit),
    );

    // Signal pipeline: publish → evaluate → subscribe → clear dirty.
    app.add_systems(
        Update,
        (signal_publish, signal_evaluate, signal_subscribe, signal_clear_dirty)
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
