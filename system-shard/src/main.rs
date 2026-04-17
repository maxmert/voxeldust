use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use clap::Parser;
use glam::{DQuat, DVec3};
use tokio::sync::mpsc;
use tracing::{info, warn};

use voxeldust_core::autopilot::{self, FlightPhase, GuidanceCommand};
use voxeldust_core::client_message::{
    CelestialBodyData, JoinResponseData, LightingData, ServerMsg, WorldStateData,
};
use voxeldust_core::ecs::{
    self, AngularVelocity, Autopilot, AutopilotIntercept, InSoi, Landed, LandingZoneDebounce,
    Position, Rotation, ShipId, ShipPhysics, ThermalState, ThrustInput, TorqueInput, Velocity,
    WarpAutopilot,
};
use voxeldust_core::handoff;
use voxeldust_core::shard_message::{
    AoiTarget, AutopilotSnapshotData, CelestialBodySnapshotData, LightingInfoData,
    PlanetPlayerDigestData, ShardMsg, ShipNearbyInfoData, ShipPositionUpdate,
    SystemEntitiesUpdateData, SystemSceneUpdateData,
};
use voxeldust_core::client_message::{EntityKind, LodTier, ObservableEntityData};
use voxeldust_core::shard_types::{SessionToken, ShardId, ShardType};
use voxeldust_core::system::{
    check_atmosphere, compute_full_aerodynamics, compute_gravity_acceleration, compute_lighting,
    compute_planet_position, compute_planet_velocity, compute_soi_radius, SystemParams,
};
use voxeldust_shard_common::client_listener;
use voxeldust_shard_common::harness::{
    celestial_time_from_epoch, NetworkBridge, ShardHarness, ShardHarnessConfig,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "system-shard",
    about = "Voxeldust system shard — orbital mechanics"
)]
struct Args {
    #[arg(long)]
    shard_id: u64,
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    orchestrator: String,
    #[arg(long, default_value = "127.0.0.1:9090")]
    orchestrator_heartbeat: String,
    #[arg(long, default_value = "9777")]
    tcp_port: u16,
    #[arg(long, default_value = "9778")]
    udp_port: u16,
    #[arg(long, default_value = "9779")]
    quic_port: u16,
    #[arg(long, default_value = "9081")]
    healthz_port: u16,
    #[arg(long)]
    advertise_host: Option<String>,
    /// Galaxy seed for interstellar warp coordinate transforms.
    #[arg(long, default_value = "0")]
    galaxy_seed: u64,
    /// Star index within the galaxy.
    #[arg(long, default_value = "0")]
    star_index: u32,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Physics tick interval (seconds). Matches core::autopilot::PHYSICS_DT.
const DT: f64 = autopilot::PHYSICS_DT;

// ---------------------------------------------------------------------------
// Components (system-shard-specific)
// ---------------------------------------------------------------------------

/// Orbit stabilizer marker. When present, SOI entry zeros planet-relative
/// velocity instead of raw frame conversion. Default: present (on).
#[derive(Component)]
struct OrbitStabilizer;

// ---------------------------------------------------------------------------
// EVA player components
// ---------------------------------------------------------------------------

/// Marker component for an EVA player entity (player floating in space).
#[derive(Component)]
struct EvaPlayer;

/// EVA player session identity.
#[derive(Component)]
struct EvaSession(SessionToken);

/// EVA player display name.
#[derive(Component)]
struct EvaPlayerName(String);

/// EVA player position in system space (meters).
#[derive(Component)]
struct EvaPosition(DVec3);

/// EVA player velocity in system space (m/s).
#[derive(Component)]
struct EvaVelocity(DVec3);

/// EVA player rotation (orientation quaternion).
#[derive(Component)]
struct EvaRotation(DQuat);

/// EVA player physics parameters — jetpack thrust, rotation speed, drag, etc.
/// All parameters configurable per-entity, no magic numbers.
#[derive(Component, Clone)]
struct EvaPhysics {
    /// Jetpack thrust in Newtons.
    jetpack_thrust_n: f64,
    /// Rotation speed in rad/s.
    rotation_speed: f64,
    /// Player mass in kg (affects acceleration from thrust and gravity).
    mass_kg: f64,
    /// Projected cross-sectional area (m²) for atmospheric drag. Suited
    /// astronaut in spread posture ≈ 0.7 m²; standing vertical ≈ 0.5 m².
    drag_area_m2: f64,
    /// Drag coefficient (dimensionless). Human body in suit ≈ 1.0–1.2.
    drag_coefficient: f64,
}

impl Default for EvaPhysics {
    fn default() -> Self {
        Self {
            jetpack_thrust_n: 500.0,    // ~0.5 kN jetpack
            rotation_speed: 2.0,         // rad/s
            mass_kg: 80.0,              // typical suited astronaut
            drag_area_m2: 0.7,          // spread posture
            drag_coefficient: 1.1,      // human body in suit
        }
    }
}

/// Current EVA input state (from client UDP).
#[derive(Component, Default)]
struct EvaInputState {
    /// Movement input [-1, 1] per axis (forward, right, up).
    movement: [f32; 3],
    /// Look yaw (radians).
    look_yaw: f32,
    /// Look pitch (radians).
    look_pitch: f32,
    /// Action button state.
    action: u8,
    prev_action: u8,
}

/// EVA player registration: player_name → (Entity, session, spawn position).
/// Keyed by player name because the session token changes on every TCP
/// reconnect (each connect generates a fresh random token in client_listener).
/// The handoff carries the old session, but the new TCP connect has a new one.
#[derive(Resource, Default)]
struct EvaPlayerIndex(HashMap<String, (Entity, SessionToken, DVec3)>);

// ---------------------------------------------------------------------------
// Ship collider cache component
// ---------------------------------------------------------------------------

/// Cached ship collider shapes received from the ship shard.
/// Stored as component data for future EVA collision (Phase 4).
/// The system shard does not run Rapier physics — it stores the shapes
/// so they can be used when EVA players need hull collision detection.
#[derive(Component)]
struct ShipColliderCache {
    chunks: Vec<voxeldust_core::shard_message::ChunkColliderData>,
    hull_min: glam::Vec3,
    hull_max: glam::Vec3,
}

/// Session token for a directly connected TCP client (debug mode).
/// Inner field read when building WorldState broadcasts with per-client filtering.
#[derive(Component)]
struct ConnectedSession(#[allow(dead_code)] SessionToken);

/// Player name for a directly connected TCP client (debug mode).
/// Inner field read when building WorldState broadcasts with per-client filtering.
#[derive(Component)]
struct ConnectedPlayerName(#[allow(dead_code)] String);

/// Marker: ship arrived via warp handoff — exempt from stale-entity cleanup
/// because the peer registry still shows the old host.
#[derive(Component)]
struct WarpArrivedShip;

// ---------------------------------------------------------------------------
// Resources (system-shard-specific)
// ---------------------------------------------------------------------------

/// Immutable system parameters (derived from seed).
#[derive(Resource)]
struct SystemConfig(SystemParams);

/// This shard's identity.
#[derive(Resource)]
struct ShardIdentity(ShardId);

/// System seed for JoinResponse.
#[derive(Resource)]
struct SystemSeed(u64);

/// Current planet positions in system-space (updated each tick).
#[derive(Resource)]
struct PlanetPositions(Vec<DVec3>);

/// Previous tick's planet positions (for co-movement delta).
#[derive(Resource)]
struct OldPlanetPositions(Vec<DVec3>);

/// Orbital velocities of planets in system-space (m/s, real-time).
#[derive(Resource)]
struct PlanetVelocities(Vec<DVec3>);

/// Galaxy context for warp departure coordinate transforms.
#[derive(Resource)]
struct GalaxyContext {
    seed: u64,
    star_index: u32,
    star_position_gu: DVec3,
    provisioned_galaxy: Option<ShardId>,
    galaxy_provisioning_in_flight: bool,
}

/// Ships that have departed via warp (don't re-create in discover_ships).
#[derive(Resource, Default)]
struct DepartedShips(HashSet<u64>);

/// Per-ship visibility set: which other ships this ship can see.
/// Updated by `compute_visibility` and compared against previous state
/// to detect enter/leave events.
#[derive(Component, Default)]
struct ShipVisibilitySet(HashSet<u64>);

/// Configurable ship-to-ship visibility range (meters).
/// Hysteresis: open connections at `range`, close at `range * 1.5`.
/// Only used by the legacy `VisibilityDirective` path (retires with Phase 11).
#[derive(Resource)]
struct ShipVisibilityRange(f64);

impl Default for ShipVisibilityRange {
    fn default() -> Self {
        Self(50_000.0) // 50km default
    }
}

/// AOI tier thresholds for the unified entity visibility pass.
/// Star-Citizen-style defaults; tune per-deployment via environment in future.
#[derive(Resource, Clone)]
struct VisibilityConfig {
    ship_full_range: f64,
    ship_reduced_range: f64,
    ship_coarse_range: f64,
    player_full_range: f64,
    player_reduced_range: f64,
    player_coarse_range: f64,
    /// Hysteresis factor applied when *dropping* an entity out of any tier.
    /// Prevents tier flapping at exact threshold crossings.
    hysteresis: f64,
}

impl Default for VisibilityConfig {
    fn default() -> Self {
        Self {
            ship_full_range: 2_000.0,
            ship_reduced_range: 20_000.0,
            ship_coarse_range: 200_000.0,
            player_full_range: 200.0,
            player_reduced_range: 2_000.0,
            player_coarse_range: 20_000.0,
            hysteresis: 1.1,
        }
    }
}

/// Surface-player aggregate record from a planet shard's `PlanetPlayerDigest`.
/// Carries only what the system-shard AOI compute needs (position in system-space,
/// rotation, name). Detailed state stays local to the owning planet shard.
#[derive(Clone)]
struct SurfacePlayerRecord {
    player_name: String,
    planet_index: u32,
    planet_shard: ShardId,
    position: DVec3,
    rotation: DQuat,
    /// Last tick we saw a digest from this planet. Used for stale-entry eviction.
    last_tick: u64,
}

/// Registry of surface players known to this system shard (populated from
/// `PlanetPlayerDigest`). Key = session_token.0.
#[derive(Resource, Default)]
struct SurfacePlayerRegistry {
    players: HashMap<u64, SurfacePlayerRecord>,
}

/// Planet shards provisioned by this system shard: planet_seed → shard_id.
#[derive(Resource, Default)]
struct ProvisionedPlanets(HashMap<u64, ShardId>);

/// Planet seeds currently being provisioned (async in-flight).
#[derive(Resource, Default)]
struct ProvisioningInFlight(HashSet<u64>);

/// Pending handoffs being relayed: session_token → source shard id.
#[derive(Resource, Default)]
struct PendingHandoffs(HashMap<SessionToken, ShardId>);

/// Pending warp arrivals: ship_id → handoff data, stored until discover_ships
/// creates the ship entity.
#[derive(Resource, Default)]
struct PendingWarpArrivals(HashMap<u64, handoff::PlayerHandoff>);

/// Orchestrator URL for provisioning requests.
#[derive(Resource)]
struct OrchestratorUrl(String);

/// Shared HTTP client for provisioning (connection pooled).
#[derive(Resource)]
struct HttpClient(reqwest::Client);

/// Channel sender for planet provisioning results.
#[derive(Resource, Clone)]
struct PlanetProvisionSender(mpsc::Sender<(u64, ShardId)>);

/// Channel receiver for planet provisioning results.
#[derive(Resource)]
struct PlanetProvisionReceiver(mpsc::Receiver<(u64, ShardId)>);

/// Channel sender for galaxy shard provisioning results.
#[derive(Resource, Clone)]
struct GalaxyProvisionSender(mpsc::Sender<ShardId>);

/// Channel receiver for galaxy shard provisioning results.
#[derive(Resource)]
struct GalaxyProvisionReceiver(mpsc::Receiver<ShardId>);


// ---------------------------------------------------------------------------
// Messages (system-shard-specific)
// ---------------------------------------------------------------------------

/// A new TCP client connected (debug mode).
#[derive(Message)]
struct ClientConnectedMsg {
    session_token: SessionToken,
    player_name: String,
    tcp_write: Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>,
}

/// Ship control input from a ship shard via QUIC.
#[derive(Message)]
struct ShipControlMsg {
    ship_id: u64,
    thrust: DVec3,
    torque: DVec3,
}

/// Autopilot command: engage or disengage.
#[derive(Message)]
struct AutopilotMsg {
    ship_id: u64,
    target_body_id: u32,
    speed_tier: u8,
    autopilot_mode: u8,
}

/// Warp autopilot command: engage warp toward a star, or disengage.
#[derive(Message)]
struct WarpAutopilotMsg {
    ship_id: u64,
    target_star_index: u32,
}

/// Player handoff arrived via QUIC (ship→planet, planet→ship, galaxy→system).
#[derive(Message)]
struct HandoffMsg {
    handoff: handoff::PlayerHandoff,
    source_shard: ShardId,
}

/// Handoff accepted relay.
#[derive(Message)]
struct ShipPropsUpdateMsg {
    ship_id: u64,
    props: voxeldust_core::shard_message::ShipPropertiesUpdateData,
}

#[derive(Message)]
struct HandoffAcceptedMsg {
    session_token: SessionToken,
    target_shard: ShardId,
}

/// Planet provisioning completed.
#[derive(Message)]
struct PlanetProvisionedMsg {
    planet_seed: u64,
    shard_id: ShardId,
}

/// Galaxy shard provisioning completed.
#[derive(Message)]
struct GalaxyProvisionedMsg {
    shard_id: ShardId,
}

/// Ship collider shapes from ship shard.
#[derive(Message)]
struct ShipColliderSyncMsg {
    ship_id: u64,
    data: voxeldust_core::shard_message::ShipColliderSyncData,
}

// ---------------------------------------------------------------------------
// SystemSets
// ---------------------------------------------------------------------------

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum SystemShardSet {
    Bridge,
    Spawn,
    Physics,
    Detection,
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
        events.write(ClientConnectedMsg {
            session_token: conn.session_token,
            player_name: conn.player_name.clone(),
            tcp_write: conn.tcp_write.clone(),
        });
        info!(
            player = %conn.player_name,
            session = conn.session_token.0,
            "player joined system (direct TCP)"
        );
    }
}

fn drain_quic(
    mut bridge: ResMut<NetworkBridge>,
    mut control_events: MessageWriter<ShipControlMsg>,
    mut autopilot_events: MessageWriter<AutopilotMsg>,
    mut warp_events: MessageWriter<WarpAutopilotMsg>,
    mut handoff_events: MessageWriter<HandoffMsg>,
    mut accepted_events: MessageWriter<HandoffAcceptedMsg>,
    mut props_events: MessageWriter<ShipPropsUpdateMsg>,
    mut collider_sync_events: MessageWriter<ShipColliderSyncMsg>,
    mut digest_events: MessageWriter<PlanetPlayerDigestMsg>,
    ship_index: Res<ecs::ShipEntityIndex>,
    tick: Res<ecs::TickCounter>,
) {
    for _ in 0..32 {
        let queued = match bridge.quic_msg_rx.try_recv() {
            Ok(q) => q,
            Err(_) => break,
        };
        match queued.msg {
            ShardMsg::ShipControlInput(ctrl) => {
                control_events.write(ShipControlMsg {
                    ship_id: ctrl.ship_id,
                    thrust: ctrl.thrust,
                    torque: ctrl.torque,
                });
            }
            ShardMsg::AutopilotCommand(cmd) => {
                autopilot_events.write(AutopilotMsg {
                    ship_id: cmd.ship_id,
                    target_body_id: cmd.target_body_id,
                    speed_tier: cmd.speed_tier,
                    autopilot_mode: cmd.autopilot_mode,
                });
            }
            ShardMsg::WarpAutopilotCommand(cmd) => {
                warp_events.write(WarpAutopilotMsg {
                    ship_id: cmd.ship_id,
                    target_star_index: cmd.target_star_index,
                });
            }
            ShardMsg::PlayerHandoff(h) => {
                let source = h.source_shard;
                handoff_events.write(HandoffMsg {
                    handoff: h,
                    source_shard: source,
                });
            }
            ShardMsg::HandoffAccepted(a) => {
                accepted_events.write(HandoffAcceptedMsg {
                    session_token: a.session_token,
                    target_shard: a.target_shard,
                });
            }
            ShardMsg::ShipPropertiesUpdate(data) => {
                props_events.write(ShipPropsUpdateMsg {
                    ship_id: data.ship_id,
                    props: data,
                });
            }
            ShardMsg::SignalBroadcast(data) => {
                // Legacy single-signal relay (backward compat).
                if data.scope >= 2 {
                    if let Ok(reg) = bridge.peer_registry.try_read() {
                        for peer in reg.all() {
                            if peer.id.0 == data.source_shard_id {
                                continue;
                            }
                            if let Some(addr) = reg.quic_addr(peer.id) {
                                let relay_msg = ShardMsg::SignalBroadcast(data.clone());
                                let _ = bridge.quic_send_tx.try_send((peer.id, addr, relay_msg));
                            }
                        }
                    }
                }
            }
            ShardMsg::SignalBroadcastBatch(batch) => {
                // Relay LongRange (scope=2) and Radio (scope=3) entries to all peers.
                // ShortRange entries (scope=1) should not arrive here but are filtered
                // out defensively to avoid unnecessary relay.
                let relay_entries: Vec<_> = batch.entries.iter()
                    .filter(|e| e.scope >= 2)
                    .cloned()
                    .collect();
                if !relay_entries.is_empty() {
                    let relay_batch = ShardMsg::SignalBroadcastBatch(
                        voxeldust_core::shard_message::SignalBroadcastBatchData {
                            source_shard_id: batch.source_shard_id,
                            source_position: batch.source_position,
                            entries: relay_entries,
                        },
                    );
                    if let Ok(reg) = bridge.peer_registry.try_read() {
                        for peer in reg.all() {
                            if peer.id.0 == batch.source_shard_id {
                                continue;
                            }
                            if let Some(addr) = reg.quic_addr(peer.id) {
                                let _ = bridge.quic_send_tx.try_send((
                                    peer.id, addr, relay_batch.clone(),
                                ));
                            }
                        }
                    }
                }
            }
            ShardMsg::ShipColliderSync(data) => {
                // Store collider shapes on the ship entity for future EVA collision.
                if let Some(&entity) = ship_index.0.get(&data.ship_id) {
                    collider_sync_events.write(ShipColliderSyncMsg {
                        ship_id: data.ship_id,
                        data,
                    });
                } else {
                    warn!(
                        ship_id = data.ship_id,
                        "ShipColliderSync for unknown ship — ignoring"
                    );
                }
            }
            ShardMsg::PlanetPlayerDigest(data) => {
                digest_events.write(PlanetPlayerDigestMsg { data });
            }
            other => {
                if tick.0 % 100 == 0 {
                    info!(
                        "system shard received unexpected QUIC message: {:?}",
                        std::mem::discriminant(&other)
                    );
                }
            }
        }
    }
}

fn drain_provisions(
    mut planet_rx: ResMut<PlanetProvisionReceiver>,
    mut galaxy_rx: ResMut<GalaxyProvisionReceiver>,
    mut planet_events: MessageWriter<PlanetProvisionedMsg>,
    mut galaxy_events: MessageWriter<GalaxyProvisionedMsg>,
) {
    for _ in 0..16 {
        match planet_rx.0.try_recv() {
            Ok((planet_seed, shard_id)) => {
                planet_events.write(PlanetProvisionedMsg {
                    planet_seed,
                    shard_id,
                });
            }
            Err(_) => break,
        }
    }
    for _ in 0..4 {
        match galaxy_rx.0.try_recv() {
            Ok(shard_id) => {
                galaxy_events.write(GalaxyProvisionedMsg { shard_id });
            }
            Err(_) => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Spawn systems
// ---------------------------------------------------------------------------

/// Process direct TCP client connections (debug mode) — spawn a ship entity.
fn process_connects(
    mut commands: Commands,
    mut events: MessageReader<ClientConnectedMsg>,
    sys_config: Res<SystemConfig>,
    system_seed: Res<SystemSeed>,
    galaxy_ctx: Res<GalaxyContext>,
    planet_pos: Res<PlanetPositions>,
    celestial_time: Res<ecs::CelestialTime>,
    mut ship_index: ResMut<ecs::ShipEntityIndex>,
    eva_index: Res<EvaPlayerIndex>,
    bridge: Res<NetworkBridge>,
) {
    for event in events.read() {
        let token = event.session_token;
        let seed = system_seed.0;
        let galaxy_seed = galaxy_ctx.seed;
        let game_time = celestial_time.0;
        let tcp_write = event.tcp_write.clone();

        // Check if this connection belongs to an existing EVA player.
        // Keyed by player_name because session tokens change on every TCP
        // reconnect (client_listener generates a fresh random token).
        info!(
            player = %event.player_name,
            session = token.0,
            eva_index_keys = ?eva_index.0.keys().collect::<Vec<_>>(),
            "process_connects: checking EVA index"
        );
        if let Some(&(_eva_entity, _old_session, eva_pos)) = eva_index.0.get(&event.player_name) {
            info!(
                player = %event.player_name,
                eva_pos = format!("({:.0},{:.0},{:.0})", eva_pos.x, eva_pos.y, eva_pos.z),
                "process_connects: MATCHED EVA player — sending JoinResponse with EVA position"
            );

            // Collect all ship shards for observer connections.
            let ship_preconnects: Vec<(u64, String, String, u64)> =
                if let Ok(reg) = bridge.peer_registry.try_read() {
                    reg.find_by_type(ShardType::Ship)
                        .iter()
                        .filter_map(|info| {
                            let ship_id = info.ship_id?;
                            Some((
                                ship_id,
                                info.endpoint.tcp_addr.to_string(),
                                info.endpoint.udp_addr.to_string(),
                                info.id.0,
                            ))
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            // Send JoinResponse + ShardPreConnect for all ships.
            tokio::spawn(async move {
                let jr = ServerMsg::JoinResponse(JoinResponseData {
                    seed,
                    planet_radius: 0,
                    player_id: token.0,
                    spawn_position: eva_pos,
                    spawn_rotation: DQuat::IDENTITY,
                    spawn_forward: DVec3::NEG_Z,
                    session_token: token,
                    shard_type: 1, // System
                    galaxy_seed,
                    system_seed: seed,
                    game_time,
                    reference_position: DVec3::ZERO,
                    reference_rotation: DQuat::IDENTITY,
                });
                let mut writer = tcp_write.lock().await;
                let _ = client_listener::send_tcp_msg(&mut *writer, &jr).await;

                // Send ShardPreConnect for each ship shard so the EVA client
                // opens observer connections and receives ship block data.
                for (ship_id, tcp_addr, udp_addr, shard_id) in &ship_preconnects {
                    let pc = ServerMsg::ShardPreConnect(handoff::ShardPreConnect {
                        shard_type: 2, // Ship
                        tcp_addr: tcp_addr.clone(),
                        udp_addr: udp_addr.clone(),
                        seed: *ship_id,
                        planet_index: 0,
                        reference_position: DVec3::ZERO,
                        reference_rotation: DQuat::IDENTITY,
                        shard_id: *shard_id,
                    });
                    let _ = client_listener::send_tcp_msg(&mut *writer, &pc).await;
                }
            });

            info!(
                session = token.0,
                "EVA player TCP connected — JoinResponse + ship ShardPreConnects sent"
            );
            continue;
        }

        // Standard path: create a Ship entity for debug/direct connections.
        let ship_id = token.0;
        let start_pos = if !planet_pos.0.is_empty() {
            planet_pos.0[0] + DVec3::new(sys_config.0.scale.spawn_offset, 0.0, 0.0)
        } else {
            DVec3::new(sys_config.0.scale.fallback_spawn_distance, 0.0, 0.0)
        };

        let entity = commands
            .spawn((
                ShipId(ship_id),
                Position(start_pos),
                Velocity(DVec3::ZERO),
                Rotation(DQuat::IDENTITY),
                AngularVelocity(DVec3::ZERO),
                ThrustInput(DVec3::ZERO),
                TorqueInput(DVec3::ZERO),
                ShipPhysics(autopilot::ShipPhysicalProperties::starter_ship()),
                ThermalState { energy_j: 0.0 },
                LandingZoneDebounce {
                    consecutive_ticks: 0,
                },
                OrbitStabilizer,
                ConnectedSession(token),
                ConnectedPlayerName(event.player_name.clone()),
            ))
            .id();

        ship_index.0.insert(ship_id, entity);

        // Pre-register in SOI if spawning inside one.
        let mut in_soi_idx = None;
        for (i, planet) in sys_config.0.planets.iter().enumerate() {
            let dist = (start_pos - planet_pos.0[i]).length();
            let soi = compute_soi_radius(planet, &sys_config.0.star);
            if dist < soi {
                in_soi_idx = Some(i);
                info!(ship_id, planet_index = i, "spawned inside SOI — pre-registered");
                break;
            }
        }
        if let Some(pi) = in_soi_idx {
            commands.entity(entity).insert(InSoi { planet_index: pi });
        }

        // Send JoinResponse via TCP.
        tokio::spawn(async move {
            let jr = ServerMsg::JoinResponse(JoinResponseData {
                seed,
                planet_radius: 0,
                player_id: token.0,
                spawn_position: start_pos,
                spawn_rotation: DQuat::IDENTITY,
                spawn_forward: DVec3::NEG_Z,
                session_token: token,
                shard_type: 1,
                galaxy_seed,
                system_seed: seed,
                game_time,
                reference_position: DVec3::ZERO,
                reference_rotation: DQuat::IDENTITY,
            });
            let mut writer = tcp_write.lock().await;
            let _ = client_listener::send_tcp_msg(&mut *writer, &jr).await;
        });

        info!(ship_id, "spawned ship for direct TCP client");
    }
}

/// Process ship control input messages.
fn process_ship_control(
    mut events: MessageReader<ShipControlMsg>,
    mut ships: Query<(
        &ShipId,
        &mut ThrustInput,
        &mut TorqueInput,
        &ShipPhysics,
        Option<&Landed>,
        Option<&Autopilot>,
        Option<&WarpAutopilot>,
    )>,
    sys_config: Res<SystemConfig>,
    tick: Res<ecs::TickCounter>,
) {
    for event in events.read() {
        for (ship_id, mut thrust, mut torque, physics, landed, autopilot, warp_ap) in &mut ships {
            if ship_id.0 != event.ship_id {
                continue;
            }

            // Takeoff: if landed and receiving thrust that exceeds gravity, detach.
            // (Landed component removal happens in ground_contact system when altitude increases.)
            if landed.is_some() && event.thrust.length_squared() > 0.01 {
                let thrust_accel = event.thrust.length() / physics.0.mass_kg;
                let gravity = landed
                    .and_then(|l| sys_config.0.planets.get(l.planet_index))
                    .map(|p| p.surface_gravity)
                    .unwrap_or(10.0);
                if thrust_accel > gravity * 1.1 {
                    // Takeoff detected — will be handled by ground_contact system.
                    info!(
                        ship_id = event.ship_id,
                        "ship taking off — detach pending"
                    );
                }
            }

            thrust.0 = event.thrust;

            // During warp, the autopilot controls rotation exclusively.
            let is_warp = warp_ap.is_some()
                && autopilot
                    .map(|a| a.mode == autopilot::AutopilotMode::WarpTravel)
                    .unwrap_or(false);
            if !is_warp {
                torque.0 = event.torque;
            }

            if tick.0 % 100 == 0 {
                info!(
                    ship_id = event.ship_id,
                    thrust = format!(
                        "({:.0},{:.0},{:.0})",
                        event.thrust.x, event.thrust.y, event.thrust.z
                    ),
                    torque = format!(
                        "({:.2},{:.2},{:.2})",
                        event.torque.x, event.torque.y, event.torque.z
                    ),
                    "ShipControlInput applied"
                );
            }
            break;
        }
    }
}

/// Process ship physical properties updates from ship shards.
/// Updates the ShipPhysics component so the system shard uses correct mass/thrust.
fn process_ship_props_update(
    mut events: MessageReader<ShipPropsUpdateMsg>,
    mut ships: Query<(&ShipId, &mut ShipPhysics)>,
) {
    for event in events.read() {
        for (ship_id, mut physics) in &mut ships {
            if ship_id.0 != event.ship_id {
                continue;
            }

            let d = &event.props;
            physics.0.mass_kg = d.mass_kg;
            physics.0.max_thrust_forward_n = d.max_thrust_forward_n;
            physics.0.max_thrust_reverse_n = d.max_thrust_reverse_n;
            physics.0.max_torque_nm = d.max_torque_nm;
            physics.0.thrust_multiplier = d.thrust_multiplier;
            physics.0.dimensions = d.dimensions;
            // Recompute cross sections from new dimensions.
            physics.0.cross_section_front = d.dimensions.0 * d.dimensions.1;
            physics.0.cross_section_side = d.dimensions.2 * d.dimensions.1;
            physics.0.cross_section_top = d.dimensions.0 * d.dimensions.2;

            info!(
                ship_id = event.ship_id,
                mass = format!("{:.0} kg", d.mass_kg),
                thrust_fwd = format!("{:.0} N", d.max_thrust_forward_n),
                "ship properties updated from ship shard"
            );
            break;
        }
    }
}

/// Process autopilot commands (engage/disengage).
fn process_autopilot_commands(
    mut commands: Commands,
    mut events: MessageReader<AutopilotMsg>,
    mut ships: Query<(
        Entity,
        &ShipId,
        &Position,
        &Velocity,
        &ShipPhysics,
        Option<&Autopilot>,
        Option<&WarpAutopilot>,
    )>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    celestial_time: Res<ecs::CelestialTime>,
    physics_time: Res<ecs::PhysicsTime>,
    tick: Res<ecs::TickCounter>,
) {
    for event in events.read() {
        for (entity, ship_id, pos, vel, physics, autopilot, warp_ap) in &mut ships {
            if ship_id.0 != event.ship_id {
                continue;
            }

            // Disengage.
            if event.target_body_id == 0xFFFFFFFF {
                let is_warp = warp_ap.is_some()
                    && autopilot
                        .map(|a| a.mode == autopilot::AutopilotMode::WarpTravel)
                        .unwrap_or(false);
                if is_warp {
                    info!(
                        ship_id = event.ship_id,
                        "ignoring planet autopilot disengage during warp"
                    );
                } else {
                    commands
                        .entity(entity)
                        .remove::<Autopilot>()
                        .remove::<AutopilotIntercept>()
                        .remove::<WarpAutopilot>();
                    info!(ship_id = event.ship_id, "autopilot disengaged");
                }
                break;
            }

            let planet_index = (event.target_body_id - 1) as usize;
            if planet_index >= sys_config.0.planets.len() {
                break;
            }

            // Don't let planet autopilot override active warp.
            let is_warp = warp_ap.is_some()
                && autopilot
                    .map(|a| a.mode == autopilot::AutopilotMode::WarpTravel)
                    .unwrap_or(false);
            if is_warp {
                info!(
                    ship_id = event.ship_id,
                    "ignoring planet autopilot engage during warp"
                );
                break;
            }

            let solve_result = autopilot::solve_intercept(
                pos.0,
                vel.0,
                &sys_config.0.planets[planet_index],
                &sys_config.0.star,
                &sys_config.0,
                &planet_pos.0,
                celestial_time.0,
                sys_config.0.scale.time_scale,
                &physics.0,
                event.speed_tier,
            );

            if let Some(sol) = solve_result {
                let mode = autopilot::AutopilotMode::from_u8(event.autopilot_mode);
                let planet = &sys_config.0.planets[planet_index];
                let default_orbit_alt = (planet.radius_m * 0.01)
                    .clamp(planet.atmosphere.atmosphere_height * 1.5, planet.radius_m * 0.1);

                commands.entity(entity).insert((
                    Autopilot {
                        mode,
                        phase: FlightPhase::Accelerate,
                        target_body_id: event.target_body_id,
                        thrust_tier: event.speed_tier,
                        engage_time: physics_time.0,
                        estimated_tof: sol.tof_real_seconds,
                        braking_committed: false,
                        target_orbit_altitude: default_orbit_alt,
                    },
                    AutopilotIntercept {
                        intercept_pos: sol.intercept_pos,
                        target_arrival_vel: sol.arrival_planet_vel,
                        last_solve_tick: tick.0,
                    },
                ));

                info!(
                    ship_id = event.ship_id,
                    planet = planet_index,
                    tier = event.speed_tier,
                    mode = ?mode,
                    eta_s = sol.tof_real_seconds as u64,
                    "autopilot engaged"
                );
            }
            break;
        }
    }
}

/// Process warp autopilot commands.
fn process_warp_commands(
    mut commands: Commands,
    mut events: MessageReader<WarpAutopilotMsg>,
    mut ships: Query<(
        Entity,
        &ShipId,
        Option<&Autopilot>,
        Option<&WarpAutopilot>,
        &mut ThrustInput,
    )>,
    galaxy_ctx: Res<GalaxyContext>,
    physics_time: Res<ecs::PhysicsTime>,
    tick: Res<ecs::TickCounter>,
) {
    for event in events.read() {
        for (entity, ship_id, autopilot, warp_ap, mut thrust) in &mut ships {
            if ship_id.0 != event.ship_id {
                continue;
            }

            if event.target_star_index == 0xFFFFFFFF {
                // Disengage warp.
                let is_warp = warp_ap.is_some()
                    && autopilot
                        .map(|a| a.mode == autopilot::AutopilotMode::WarpTravel)
                        .unwrap_or(false);
                if is_warp {
                    commands
                        .entity(entity)
                        .remove::<Autopilot>()
                        .remove::<AutopilotIntercept>()
                        .remove::<WarpAutopilot>();
                    thrust.0 = DVec3::ZERO;
                    info!(ship_id = event.ship_id, "warp autopilot disengaged");
                }
            } else {
                // Engage warp toward target star.
                if galaxy_ctx.seed == 0 {
                    warn!(
                        ship_id = event.ship_id,
                        "warp command received but system shard has no galaxy context (galaxy_seed=0)"
                    );
                    break;
                }

                let galaxy_map =
                    voxeldust_core::galaxy::GalaxyMap::generate(galaxy_ctx.seed);
                if let Some(target_star) = galaxy_map.get_star(event.target_star_index) {
                    let dir_gu =
                        (target_star.position - galaxy_ctx.star_position_gu).normalize();

                    // Don't reset if already warping to the same target.
                    let already_warping = warp_ap
                        .map(|w| w.target_star_index == event.target_star_index)
                        .unwrap_or(false)
                        && autopilot
                            .map(|a| a.mode == autopilot::AutopilotMode::WarpTravel)
                            .unwrap_or(false);

                    if already_warping {
                        info!(
                            ship_id = event.ship_id,
                            target_star = event.target_star_index,
                            "warp already engaged for same target — ignoring duplicate"
                        );
                        break;
                    }

                    commands.entity(entity).insert((
                        Autopilot {
                            mode: autopilot::AutopilotMode::WarpTravel,
                            phase: FlightPhase::WarpAlign,
                            target_body_id: 0,
                            thrust_tier: 4,
                            engage_time: physics_time.0,
                            estimated_tof: 0.0,
                            braking_committed: false,
                            target_orbit_altitude: 0.0,
                        },
                        AutopilotIntercept {
                            intercept_pos: DVec3::ZERO,
                            target_arrival_vel: DVec3::ZERO,
                            last_solve_tick: tick.0,
                        },
                        WarpAutopilot {
                            target_star_index: event.target_star_index,
                            direction: dir_gu,
                        },
                    ));

                    info!(
                        ship_id = event.ship_id,
                        target_star = event.target_star_index,
                        dir = format!("({:.3},{:.3},{:.3})", dir_gu.x, dir_gu.y, dir_gu.z),
                        "warp autopilot engaged"
                    );
                }
            }
            break;
        }
    }
}

/// Process player handoff messages (ship→planet, planet→ship, galaxy→system warp arrival).
fn process_handoffs(
    mut commands: Commands,
    mut events: MessageReader<HandoffMsg>,
    mut ships: Query<(
        Entity,
        &ShipId,
        &mut Position,
        &mut Velocity,
        &mut Rotation,
    )>,
    mut pending_handoffs: ResMut<PendingHandoffs>,
    mut ship_index: ResMut<ecs::ShipEntityIndex>,
    mut eva_player_index: ResMut<EvaPlayerIndex>,
    provisioned_planets: Res<ProvisionedPlanets>,
    sys_config: Res<SystemConfig>,
    shard_identity: Res<ShardIdentity>,
    physics_time: Res<ecs::PhysicsTime>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
) {
    for event in events.read() {
        let h = &event.handoff;
        let source = event.source_shard;
        let session = h.session_token;
        pending_handoffs.0.insert(session, source);

        if let Some(planet_seed) = h.target_planet_seed {
            // Ship→Planet handoff: forward to planet shard.
            if let Some(&planet_shard_id) = provisioned_planets.0.get(&planet_seed) {
                if let Ok(reg) = bridge.peer_registry.try_read() {
                    if let Some(addr) = reg.quic_addr(planet_shard_id) {
                        let msg = ShardMsg::PlayerHandoff(h.clone());
                        match bridge.quic_send_tx.try_send((planet_shard_id, addr, msg)) {
                            Ok(()) => info!(
                                planet_seed,
                                target = planet_shard_id.0,
                                "forwarded player handoff to planet shard"
                            ),
                            Err(e) => tracing::error!(
                                planet_seed,
                                target = planet_shard_id.0,
                                %e,
                                "failed to queue player handoff to planet shard"
                            ),
                        }
                    } else {
                        tracing::warn!(
                            planet_seed,
                            target = planet_shard_id.0,
                            "no QUIC address for planet shard in peer registry"
                        );
                    }
                }
            } else {
                tracing::warn!(planet_seed, "no provisioned planet shard for handoff");
            }
        } else if let Some(target_ship_id) = h.target_ship_id {
            // Planet→Ship handoff: forward to ship shard.
            if let Ok(reg) = bridge.peer_registry.try_read() {
                let ship_shard = reg
                    .find_by_type(ShardType::Ship)
                    .iter()
                    .find(|s| s.ship_id == Some(target_ship_id))
                    .map(|s| (s.id, s.endpoint.quic_addr));
                if let Some((sid, addr)) = ship_shard {
                    let msg = ShardMsg::PlayerHandoff(h.clone());
                    match bridge.quic_send_tx.try_send((sid, addr, msg)) {
                        Ok(()) => info!(
                            ship_id = target_ship_id,
                            target = sid.0,
                            "forwarded player handoff to ship shard"
                        ),
                        Err(e) => tracing::error!(
                            ship_id = target_ship_id,
                            target = sid.0,
                            %e,
                            "failed to queue player handoff to ship shard"
                        ),
                    }
                }
            }
        } else if h.target_star_index.is_some() || h.galaxy_context.is_some() {
            // Galaxy → System warp arrival.
            let ship_id = h.session_token.0;
            let ship_forward = h.forward.normalize();

            let outermost_orbit = sys_config
                .0
                .planets
                .iter()
                .map(|p| p.orbital_elements.sma)
                .fold(sys_config.0.scale.base_sma, f64::max);
            let arrival_distance = outermost_orbit * 15.0;

            // Simulate departure exponential acceleration to find entry speed.
            let base_accel = 1_000_000.0_f64;
            let tau = 2.0_f64;
            let dt_sim = 0.05_f64;
            let accel_cap = 10_000_000_000.0_f64;
            let mut sim_v = 0.0_f64;
            let mut sim_d = 0.0_f64;
            let mut sim_t = 0.0_f64;
            while sim_d < arrival_distance && sim_t < 60.0 {
                let a = (base_accel * (2.0_f64).powf(sim_t / tau)).min(accel_cap);
                sim_v += a * dt_sim;
                sim_d += sim_v * dt_sim;
                sim_t += dt_sim;
            }
            let entry_speed = sim_v;

            // Cubic ease-out stopping distance.
            let mut orbit_smas: Vec<f64> = sys_config
                .0
                .planets
                .iter()
                .map(|p| p.orbital_elements.sma)
                .collect();
            orbit_smas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let stop_distance = if orbit_smas.is_empty() {
                outermost_orbit * 0.5
            } else {
                orbit_smas[orbit_smas.len() / 2]
            };
            let d_braking = (arrival_distance - stop_distance).max(arrival_distance * 0.5);
            let total_time = 4.0 * d_braking / entry_speed;

            let start_pos = -ship_forward * arrival_distance;
            let start_vel = ship_forward * entry_speed;
            let start_rot = h.rotation;

            // Try to update existing ship entity first.
            let mut found_existing = false;
            for (entity, sid, mut pos, mut vel, mut rot) in &mut ships {
                if sid.0 == ship_id {
                    pos.0 = start_pos;
                    vel.0 = start_vel;
                    rot.0 = start_rot;

                    commands.entity(entity).insert((
                        Autopilot {
                            mode: autopilot::AutopilotMode::WarpTravel,
                            phase: FlightPhase::WarpArrival,
                            target_body_id: 0,
                            thrust_tier: 4,
                            engage_time: physics_time.0,
                            estimated_tof: total_time,
                            braking_committed: false,
                            target_orbit_altitude: entry_speed,
                        },
                        AutopilotIntercept {
                            intercept_pos: DVec3::ZERO,
                            target_arrival_vel: DVec3::ZERO,
                            last_solve_tick: tick.0,
                        },
                        WarpAutopilot {
                            target_star_index: 0,
                            direction: ship_forward,
                        },
                    ));

                    info!(
                        ship_id,
                        pos = format!(
                            "({:.0},{:.0},{:.0})",
                            start_pos.x, start_pos.y, start_pos.z
                        ),
                        dist = format!("{:.0} m", arrival_distance),
                        speed = format!("{:.0} Gm/s", entry_speed / 1e9),
                        "warp arrival — updated existing ship"
                    );
                    found_existing = true;
                    break;
                }
            }

            if !found_existing {
                // Create ship entity directly from handoff data.
                let entity = commands
                    .spawn((
                        ShipId(ship_id),
                        Position(start_pos),
                        Velocity(start_vel),
                        Rotation(start_rot),
                        AngularVelocity(DVec3::ZERO),
                        ThrustInput(DVec3::ZERO),
                        TorqueInput(DVec3::ZERO),
                        ShipPhysics(autopilot::ShipPhysicalProperties::starter_ship()),
                        ThermalState { energy_j: 0.0 },
                        LandingZoneDebounce {
                            consecutive_ticks: 0,
                        },
                        OrbitStabilizer,
                        WarpArrivedShip,
                        Autopilot {
                            mode: autopilot::AutopilotMode::WarpTravel,
                            phase: FlightPhase::WarpArrival,
                            target_body_id: 0,
                            thrust_tier: 4,
                            engage_time: physics_time.0,
                            estimated_tof: total_time,
                            braking_committed: false,
                            target_orbit_altitude: entry_speed,
                        },
                        AutopilotIntercept {
                            intercept_pos: DVec3::ZERO,
                            target_arrival_vel: DVec3::ZERO,
                            last_solve_tick: tick.0,
                        },
                        WarpAutopilot {
                            target_star_index: 0,
                            direction: ship_forward,
                        },
                    ))
                    .id();

                ship_index.0.insert(ship_id, entity);

                info!(
                    ship_id,
                    pos = format!(
                        "({:.0},{:.0},{:.0})",
                        start_pos.x, start_pos.y, start_pos.z
                    ),
                    dist = format!("{:.0} m", arrival_distance),
                    vel = format!("{:.0} m/s", start_vel.length()),
                    "warp arrival — created ship from handoff"
                );
            }
        } else if h.target_system_eva {
            // Ship→System EVA: player exited ship hull in space.
            // Spawn an EVA player entity with inherited velocity.
            let eva_entity = commands
                .spawn((
                    EvaPlayer,
                    EvaSession(session),
                    EvaPlayerName(h.player_name.clone()),
                    EvaPosition(h.position),
                    EvaVelocity(h.velocity),
                    EvaRotation(h.rotation),
                    EvaPhysics::default(),
                    EvaInputState::default(),
                ))
                .id();

            eva_player_index.0.insert(h.player_name.clone(), (eva_entity, session, h.position));

            info!(
                session = session.0,
                player = %h.player_name,
                player_name_key = %h.player_name,
                pos = format!("({:.0},{:.0},{:.0})", h.position.x, h.position.y, h.position.z),
                pos = format!("({:.0},{:.0},{:.0})", h.position.x, h.position.y, h.position.z),
                vel = format!("{:.1} m/s", h.velocity.length()),
                "EVA player spawned from ship hull exit"
            );

            // Send HandoffAccepted back to the source ship shard.
            let accepted = ShardMsg::HandoffAccepted(handoff::HandoffAccepted {
                session_token: session,
                target_shard: shard_identity.0,
            });
            if let Ok(reg) = bridge.peer_registry.try_read() {
                if let Some(addr) = reg.quic_addr(source) {
                    let _ = bridge.quic_send_tx.try_send((source, addr, accepted));
                }
            }

            // ShardRedirect to client is handled by the ship shard upon receiving
            // HandoffAccepted — it looks up this system shard's TCP/UDP from the
            // peer registry and sends the redirect. No direct action needed here.
        } else {
            tracing::warn!("received PlayerHandoff with no target");
        }
    }
}

/// Process handoff accepted relay messages.
fn process_handoff_accepted(
    mut events: MessageReader<HandoffAcceptedMsg>,
    mut pending_handoffs: ResMut<PendingHandoffs>,
    bridge: Res<NetworkBridge>,
) {
    for event in events.read() {
        if let Some(source_shard) = pending_handoffs.0.remove(&event.session_token) {
            if let Ok(reg) = bridge.peer_registry.try_read() {
                if let Some(addr) = reg.quic_addr(source_shard) {
                    let msg = ShardMsg::HandoffAccepted(handoff::HandoffAccepted {
                        session_token: event.session_token,
                        target_shard: event.target_shard,
                    });
                    match bridge.quic_send_tx.try_send((source_shard, addr, msg)) {
                        Ok(()) => info!(
                            target = source_shard.0,
                            "relayed HandoffAccepted to source shard"
                        ),
                        Err(e) => tracing::error!(
                            source = source_shard.0,
                            %e,
                            "failed to queue HandoffAccepted relay"
                        ),
                    }
                }
            }
        }
    }
}

/// Process planet provisioning results.
fn process_planet_provisions(
    mut events: MessageReader<PlanetProvisionedMsg>,
    mut provisioned: ResMut<ProvisionedPlanets>,
    mut in_flight: ResMut<ProvisioningInFlight>,
) {
    for event in events.read() {
        in_flight.0.remove(&event.planet_seed);
        provisioned.0.insert(event.planet_seed, event.shard_id);
        info!(
            planet_seed = event.planet_seed,
            shard_id = event.shard_id.0,
            "planet shard provisioned"
        );
    }
}

/// Process galaxy provisioning results.
fn process_galaxy_provisions(
    mut events: MessageReader<GalaxyProvisionedMsg>,
    mut galaxy_ctx: ResMut<GalaxyContext>,
) {
    for event in events.read() {
        galaxy_ctx.provisioned_galaxy = Some(event.shard_id);
        galaxy_ctx.galaxy_provisioning_in_flight = false;
        info!(
            shard_id = event.shard_id.0,
            "galaxy shard provisioned and ready"
        );
    }
}

/// Discover ship shards from peer registry and create ship entities.
/// Also cleans up stale ship entities whose shard no longer exists.
fn discover_ships(
    mut commands: Commands,
    ships: Query<(Entity, &ShipId, Option<&WarpArrivedShip>)>,
    tick: Res<ecs::TickCounter>,
    shard_identity: Res<ShardIdentity>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    mut pending_warp_arrivals: ResMut<PendingWarpArrivals>,
    departed_ships: Res<DepartedShips>,
    mut ship_index: ResMut<ecs::ShipEntityIndex>,
    bridge: Res<NetworkBridge>,
) {
    if tick.0 % 40 != 0 {
        return;
    }

    let Ok(reg) = bridge.peer_registry.try_read() else {
        return;
    };

    // Collect existing ship IDs.
    let existing_ship_ids: HashSet<u64> = ships.iter().map(|(_, sid, _)| sid.0).collect();

    for shard_info in reg.find_by_type(ShardType::Ship) {
        if shard_info.host_shard_id != Some(shard_identity.0) {
            continue;
        }

        if let Some(ship_id) = shard_info.ship_id {
            if existing_ship_ids.contains(&ship_id) {
                continue;
            }
            if departed_ships.0.contains(&ship_id) {
                continue;
            }

            // Use pending warp arrival position if available.
            let (start_pos, start_vel, start_rot) =
                if let Some(arrival) = pending_warp_arrivals.0.remove(&ship_id) {
                    info!(
                        ship_id,
                        pos = format!(
                            "({:.0},{:.0},{:.0})",
                            arrival.position.x, arrival.position.y, arrival.position.z
                        ),
                        "creating ship from warp arrival handoff"
                    );
                    (arrival.position, arrival.velocity, arrival.rotation)
                } else {
                    let pos = if !planet_pos.0.is_empty() {
                        planet_pos.0[0]
                            + DVec3::new(sys_config.0.scale.spawn_offset, 0.0, 0.0)
                    } else {
                        DVec3::new(sys_config.0.scale.fallback_spawn_distance, 0.0, 0.0)
                    };
                    (pos, DVec3::ZERO, DQuat::IDENTITY)
                };

            let entity = commands
                .spawn((
                    ShipId(ship_id),
                    Position(start_pos),
                    Velocity(start_vel),
                    Rotation(start_rot),
                    AngularVelocity(DVec3::ZERO),
                    ThrustInput(DVec3::ZERO),
                    TorqueInput(DVec3::ZERO),
                    ShipPhysics(autopilot::ShipPhysicalProperties::starter_ship()),
                    ThermalState { energy_j: 0.0 },
                    LandingZoneDebounce {
                        consecutive_ticks: 0,
                    },
                    OrbitStabilizer,
                ))
                .id();

            ship_index.0.insert(ship_id, entity);

            // Pre-register in SOI if spawning inside one.
            for (i, planet) in sys_config.0.planets.iter().enumerate() {
                let dist = (start_pos - planet_pos.0[i]).length();
                let soi = compute_soi_radius(planet, &sys_config.0.star);
                if dist < soi {
                    commands.entity(entity).insert(InSoi { planet_index: i });
                    break;
                }
            }

            info!(
                ship_id,
                shard_id = shard_info.id.0,
                "discovered new ship shard — created ship entity"
            );
        }
    }

    // Clean up stale ship entities whose shard no longer exists in peer registry.
    let active_ship_ids: HashSet<u64> = reg
        .find_by_type(ShardType::Ship)
        .iter()
        .filter(|s| s.host_shard_id == Some(shard_identity.0))
        .filter_map(|s| s.ship_id)
        .collect();

    for (entity, ship_id, warp_arrived) in &ships {
        if !active_ship_ids.contains(&ship_id.0) && warp_arrived.is_none() {
            commands.entity(entity).despawn();
            ship_index.0.remove(&ship_id.0);
            info!(ship_id = ship_id.0, "removed stale ship entity");
        }
    }
}

// ---------------------------------------------------------------------------
// Physics systems
// ---------------------------------------------------------------------------

/// Update planet positions and velocities from Keplerian orbits.
fn update_orbits(
    sys_config: Res<SystemConfig>,
    mut planet_pos: ResMut<PlanetPositions>,
    mut old_planet_pos: ResMut<OldPlanetPositions>,
    mut planet_vel: ResMut<PlanetVelocities>,
    mut celestial_time: ResMut<ecs::CelestialTime>,
    mut physics_time: ResMut<ecs::PhysicsTime>,
    bridge: Res<NetworkBridge>,
) {
    physics_time.0 += DT;
    celestial_time.0 = celestial_time_from_epoch(
        &bridge.universe_epoch_ms,
        sys_config.0.scale.time_scale,
    );

    let ct = celestial_time.0;
    let time_scale = sys_config.0.scale.time_scale;
    let star_gm = sys_config.0.star.gm;

    old_planet_pos.0.clone_from(&planet_pos.0);

    for (i, planet) in sys_config.0.planets.iter().enumerate() {
        planet_pos.0[i] = compute_planet_position(planet, ct);
        planet_vel.0[i] = compute_planet_velocity(planet, star_gm, ct) * time_scale;
    }
}

/// Keep landed ships on planet surface (derived position).
fn update_landed_positions(
    mut ships: Query<
        (&ShipId, &mut Position, &mut Velocity, &mut AngularVelocity, &Landed, &InSoi, &ShipPhysics),
    >,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
) {
    for (_ship_id, mut pos, mut vel, mut ang_vel, landed, _in_soi, physics) in &mut ships {
        let planet = match sys_config.0.planets.get(landed.planet_index) {
            Some(p) => p,
            None => continue,
        };
        let pp = planet_pos.0[landed.planet_index];
        let contact_margin = physics.0.landing_gear_height + 0.5;
        pos.0 = pp + landed.surface_radial * (planet.radius_m + contact_margin);
        vel.0 = DVec3::ZERO;
        ang_vel.0 = DVec3::ZERO;
    }
}

/// Co-move flying (non-landed) ships inside SOI with their parent planet's orbital motion.
fn soi_co_movement(
    mut ships: Query<(&ShipId, &mut Position, &InSoi, Option<&Landed>)>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    old_planet_pos: Res<OldPlanetPositions>,
    tick: Res<ecs::TickCounter>,
) {
    for (ship_id, mut pos, in_soi, landed) in &mut ships {
        if landed.is_some() {
            continue;
        }
        let pi = in_soi.planet_index;
        let delta = planet_pos.0[pi] - old_planet_pos.0[pi];
        pos.0 += delta;

        // Diagnostic: log distance to planet for ships in SOI every second.
        if tick.0 % 20 == 0 {
            let dist = (pos.0 - planet_pos.0[pi]).length();
            let soi = compute_soi_radius(&sys_config.0.planets[pi], &sys_config.0.star);
            warn!(
                ship_id = ship_id.0,
                planet_idx = pi,
                dist = format!("{:.0}", dist),
                soi = format!("{:.0}", soi),
                delta = format!("{:.0}", delta.length()),
                "SOI co-move diagnostic"
            );
        }
    }
}

/// Diagnostic: log nearest planet distance for ships NOT in SOI.
fn soi_diagnostic(
    ships: Query<(&ShipId, &Position, &Velocity), Without<InSoi>>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    tick: Res<ecs::TickCounter>,
) {
    if tick.0 % 40 != 0 {
        return;
    }
    let ship_count = ships.iter().count();
    if ship_count == 0 {
        return;
    }
    for (ship_id, pos, vel) in &ships {
        let mut nearest_dist = f64::MAX;
        let mut nearest_soi = 0.0;
        let mut nearest_idx = 0usize;
        for (i, planet) in sys_config.0.planets.iter().enumerate() {
            let d = (pos.0 - planet_pos.0[i]).length();
            if d < nearest_dist {
                nearest_dist = d;
                nearest_soi = compute_soi_radius(planet, &sys_config.0.star);
                nearest_idx = i;
            }
        }
        warn!(
            ship_id = ship_id.0,
            planet = nearest_idx,
            dist = format!("{:.0}", nearest_dist),
            soi = format!("{:.0}", nearest_soi),
            ratio = format!("{:.2}", nearest_dist / nearest_soi),
            vel = format!("{:.0}", vel.0.length()),
            "NOT in SOI — nearest planet"
        );
    }
}

/// Re-solve autopilot intercepts every 20 ticks (~1 second).
fn autopilot_resolve(
    mut ships: Query<(
        &ShipId,
        &Position,
        &Velocity,
        &ShipPhysics,
        &Autopilot,
        &mut AutopilotIntercept,
    )>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    celestial_time: Res<ecs::CelestialTime>,
    tick: Res<ecs::TickCounter>,
) {
    for (_ship_id, pos, vel, physics, ap, mut intercept) in &mut ships {
        // Skip warp ships — they don't use planet intercepts.
        if ap.mode == autopilot::AutopilotMode::WarpTravel {
            continue;
        }
        // Don't re-solve while braking — ship is committed to its trajectory.
        if ap.braking_committed {
            continue;
        }
        if tick.0 - intercept.last_solve_tick < 20 {
            continue;
        }

        let planet_index = (ap.target_body_id - 1) as usize;
        if planet_index >= sys_config.0.planets.len() {
            continue;
        }

        if let Some(new_sol) = autopilot::solve_intercept(
            pos.0,
            vel.0,
            &sys_config.0.planets[planet_index],
            &sys_config.0.star,
            &sys_config.0,
            &planet_pos.0,
            celestial_time.0,
            sys_config.0.scale.time_scale,
            &physics.0,
            ap.thrust_tier,
        ) {
            let shift = (new_sol.intercept_pos - intercept.intercept_pos).length();
            let dist = (pos.0 - intercept.intercept_pos).length().max(1.0);
            if shift / dist > 0.005 {
                intercept.intercept_pos = new_sol.intercept_pos;
                intercept.target_arrival_vel = new_sol.arrival_planet_vel;
            }
            intercept.last_solve_tick = tick.0;
        }
    }
}

/// Compute autopilot guidance commands per flight phase.
fn autopilot_guidance(
    mut ships: Query<(
        &ShipId,
        &Position,
        &Velocity,
        &Rotation,
        &ShipPhysics,
        &mut Autopilot,
        &AutopilotIntercept,
        Option<&WarpAutopilot>,
        Option<&InSoi>,
    )>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    planet_vel: Res<PlanetVelocities>,
    celestial_time: Res<ecs::CelestialTime>,
    physics_time: Res<ecs::PhysicsTime>,
    tick: Res<ecs::TickCounter>,
    mut guidance_cache: Local<Vec<(u64, GuidanceCommand, FlightPhase)>>,
) {
    guidance_cache.clear();

    for (ship_id, pos, vel, rot, physics, ap, intercept, warp_ap, in_soi) in &ships {
        let planet_index = (ap.target_body_id.wrapping_sub(1)) as usize;

        // For warp phases, planet data is irrelevant.
        let is_warp = ap.mode == autopilot::AutopilotMode::WarpTravel;
        let planet = if !is_warp && planet_index < sys_config.0.planets.len() {
            Some(&sys_config.0.planets[planet_index])
        } else {
            None
        };
        let pp = if !is_warp && planet_index < planet_pos.0.len() {
            planet_pos.0[planet_index]
        } else {
            DVec3::ZERO
        };
        let pv = if !is_warp && planet_index < planet_vel.0.len() {
            planet_vel.0[planet_index]
        } else {
            DVec3::ZERO
        };

        let soi = planet
            .map(|p| compute_soi_radius(p, &sys_config.0.star))
            .unwrap_or(0.0);

        // Enforce tier restriction in atmosphere and SOI.
        let in_atmo = planet
            .map(|p| {
                let alt = (pos.0 - pp).length() - p.radius_m;
                alt < p.atmosphere.atmosphere_height && p.atmosphere.has_atmosphere
            })
            .unwrap_or(false);
        let is_in_soi = in_soi.is_some();
        let effective_tier = autopilot::effective_tier(ap.thrust_tier, in_atmo, is_in_soi);
        let engine_accel = physics.0.engine_acceleration(effective_tier);
        let thrust_force =
            autopilot::engine_tier(effective_tier).thrust_force_n * physics.0.thrust_multiplier;

        let guidance = match ap.phase {
            FlightPhase::Accelerate => {
                let g = autopilot::compute_guidance(
                    pos.0,
                    vel.0,
                    intercept.intercept_pos,
                    intercept.target_arrival_vel,
                    planet_index,
                    &sys_config.0,
                    &planet_pos.0,
                    celestial_time.0,
                    &physics.0,
                    effective_tier,
                    false,
                );
                if tick.0 % 20 == 0 {
                    let d = (intercept.intercept_pos - pos.0).length();
                    let vc = vel
                        .0
                        .dot((intercept.intercept_pos - pos.0).normalize_or_zero());
                    let rv = (vel.0 - intercept.target_arrival_vel).length();
                    warn!(
                        ship_id = ship_id.0,
                        d_int = format!("{:.0}", d),
                        v_close = format!("{:.0}", vc),
                        speed = format!("{:.0}", rv),
                        phase = ?g.phase,
                        flip = g.requires_flip,
                        "ACCEL_DIAG"
                    );
                }
                g
            }
            FlightPhase::Brake => {
                let rel_vel = vel.0 - pv;
                let rel_speed = rel_vel.length();
                let v_esc_soi = planet
                    .map(|p| autopilot::escape_velocity(p, soi - p.radius_m))
                    .unwrap_or(0.0);

                if rel_speed > v_esc_soi * 0.5 {
                    GuidanceCommand {
                        thrust_direction: -rel_vel.normalize(),
                        thrust_magnitude: thrust_force,
                        phase: FlightPhase::Brake,
                        completed: false,
                        eta_real_seconds: if engine_accel > 0.0 {
                            rel_speed / engine_accel
                        } else {
                            0.0
                        },
                        felt_g: autopilot::engine_tier(effective_tier).felt_g,
                        dampener_active: autopilot::engine_tier(effective_tier).dampened,
                        requires_flip: false,
                    }
                } else {
                    let to_planet = (pp - pos.0).normalize_or_zero();
                    let gentle_thrust =
                        autopilot::engine_tier(0).thrust_force_n * physics.0.thrust_multiplier;
                    GuidanceCommand {
                        thrust_direction: to_planet,
                        thrust_magnitude: gentle_thrust,
                        phase: FlightPhase::Brake,
                        completed: false,
                        eta_real_seconds: 0.0,
                        felt_g: autopilot::engine_tier(0).felt_g,
                        dampener_active: false,
                        requires_flip: false,
                    }
                }
            }
            FlightPhase::Flip => {
                let rel_vel = vel.0 - pv;
                let retrograde = if rel_vel.length() > 0.1 {
                    -rel_vel.normalize()
                } else {
                    -vel.0.normalize_or_zero()
                };
                GuidanceCommand {
                    thrust_direction: retrograde,
                    thrust_magnitude: 0.0,
                    phase: FlightPhase::Flip,
                    completed: false,
                    eta_real_seconds: 0.0,
                    felt_g: 0.0,
                    dampener_active: false,
                    requires_flip: false,
                }
            }
            FlightPhase::SoiApproach => {
                let rel_pos = pos.0 - pp;
                let altitude = rel_pos.length() - planet.map(|p| p.radius_m).unwrap_or(0.0);
                let speed = vel.0.length();
                let v_esc = planet
                    .map(|p| autopilot::escape_velocity(p, altitude))
                    .unwrap_or(0.0);

                if speed > v_esc * 0.7 {
                    let v_circ_target = planet
                        .map(|p| {
                            autopilot::circular_orbit_velocity(p, ap.target_orbit_altitude)
                        })
                        .unwrap_or(0.0);
                    GuidanceCommand {
                        thrust_direction: -vel.0.normalize_or_zero(),
                        thrust_magnitude: thrust_force,
                        phase: FlightPhase::SoiApproach,
                        completed: false,
                        eta_real_seconds: if engine_accel > 0.0 {
                            (speed - v_circ_target).max(0.0) / engine_accel
                        } else {
                            0.0
                        },
                        felt_g: autopilot::engine_tier(effective_tier).felt_g,
                        dampener_active: autopilot::engine_tier(effective_tier).dampened,
                        requires_flip: false,
                    }
                } else {
                    GuidanceCommand {
                        thrust_direction: DVec3::NEG_Z,
                        thrust_magnitude: 0.0,
                        phase: FlightPhase::SoiApproach,
                        completed: false,
                        eta_real_seconds: 0.0,
                        felt_g: 0.0,
                        dampener_active: false,
                        requires_flip: false,
                    }
                }
            }
            FlightPhase::CircularizeBurn | FlightPhase::AscentBurn => {
                let rel_pos = pos.0 - pp;
                let rel_vel = vel.0;
                let gm = planet.map(|p| p.gm).unwrap_or(0.0);
                let (dv, burn_dir) =
                    autopilot::circularization_delta_v(rel_pos, rel_vel, gm);
                GuidanceCommand {
                    thrust_direction: burn_dir,
                    thrust_magnitude: if dv > 1.0 { thrust_force } else { 0.0 },
                    phase: ap.phase,
                    completed: dv < 1.0,
                    eta_real_seconds: dv / engine_accel,
                    felt_g: autopilot::engine_tier(effective_tier).felt_g,
                    dampener_active: autopilot::engine_tier(effective_tier).dampened,
                    requires_flip: false,
                }
            }
            FlightPhase::StableOrbit => GuidanceCommand {
                thrust_direction: DVec3::NEG_Z,
                thrust_magnitude: 0.0,
                phase: FlightPhase::StableOrbit,
                completed: false,
                eta_real_seconds: 0.0,
                felt_g: 0.0,
                dampener_active: false,
                requires_flip: false,
            },
            FlightPhase::DeorbitBurn => {
                let rel_pos = pos.0 - pp;
                let rel_vel = vel.0;
                let h = rel_pos.cross(rel_vel);
                let prograde = if h.length_squared() > 1e-10 {
                    h.cross(rel_pos).normalize()
                } else {
                    DVec3::NEG_Z
                };
                GuidanceCommand {
                    thrust_direction: -prograde,
                    thrust_magnitude: thrust_force,
                    phase: FlightPhase::DeorbitBurn,
                    completed: false,
                    eta_real_seconds: 0.0,
                    felt_g: autopilot::engine_tier(effective_tier).felt_g,
                    dampener_active: autopilot::engine_tier(effective_tier).dampened,
                    requires_flip: false,
                }
            }
            FlightPhase::AtmosphericEntry | FlightPhase::TerminalDescent | FlightPhase::Landing => {
                if let Some(p) = planet {
                    autopilot::compute_landing_guidance(
                        pos.0,
                        vel.0,
                        pp,
                        p,
                        &physics.0,
                        engine_accel,
                    )
                } else {
                    GuidanceCommand {
                        thrust_direction: DVec3::NEG_Z,
                        thrust_magnitude: 0.0,
                        phase: ap.phase,
                        completed: false,
                        eta_real_seconds: 0.0,
                        felt_g: 0.0,
                        dampener_active: false,
                        requires_flip: false,
                    }
                }
            }
            FlightPhase::Landed => GuidanceCommand {
                thrust_direction: DVec3::Y,
                thrust_magnitude: 0.0,
                phase: FlightPhase::Landed,
                completed: true,
                eta_real_seconds: 0.0,
                felt_g: 0.0,
                dampener_active: false,
                requires_flip: false,
            },
            FlightPhase::Liftoff | FlightPhase::GravityTurn => {
                if let Some(p) = planet {
                    autopilot::compute_takeoff_guidance(
                        pos.0,
                        vel.0,
                        pp,
                        p,
                        &physics.0,
                        engine_accel,
                        ap.target_orbit_altitude,
                    )
                } else {
                    GuidanceCommand {
                        thrust_direction: DVec3::NEG_Z,
                        thrust_magnitude: 0.0,
                        phase: ap.phase,
                        completed: false,
                        eta_real_seconds: 0.0,
                        felt_g: 0.0,
                        dampener_active: false,
                        requires_flip: false,
                    }
                }
            }
            FlightPhase::EscapeBurn => {
                let rel_pos = pos.0 - pp;
                let rel_vel = vel.0;
                let h = rel_pos.cross(rel_vel);
                let prograde = if h.length_squared() > 1e-10 {
                    h.cross(rel_pos).normalize()
                } else {
                    DVec3::NEG_Z
                };
                GuidanceCommand {
                    thrust_direction: prograde,
                    thrust_magnitude: thrust_force,
                    phase: FlightPhase::EscapeBurn,
                    completed: (pos.0 - pp).length() > soi,
                    eta_real_seconds: 0.0,
                    felt_g: autopilot::engine_tier(effective_tier).felt_g,
                    dampener_active: autopilot::engine_tier(effective_tier).dampened,
                    requires_flip: false,
                }
            }
            FlightPhase::Arrived => GuidanceCommand {
                thrust_direction: DVec3::NEG_Z,
                thrust_magnitude: 0.0,
                phase: FlightPhase::Arrived,
                completed: true,
                eta_real_seconds: 0.0,
                felt_g: 0.0,
                dampener_active: false,
                requires_flip: false,
            },
            FlightPhase::WarpAlign => {
                let warp_dir = warp_ap.map(|w| w.direction).unwrap_or(DVec3::NEG_Z);
                let current_fwd = rot.0 * DVec3::NEG_Z;
                let dot = current_fwd.dot(warp_dir).clamp(-1.0, 1.0);
                let aligned = dot > 0.996; // cos(5°)

                GuidanceCommand {
                    thrust_direction: warp_dir,
                    thrust_magnitude: 0.0,
                    phase: if aligned {
                        FlightPhase::WarpAccelerate
                    } else {
                        FlightPhase::WarpAlign
                    },
                    completed: false,
                    eta_real_seconds: 0.0,
                    felt_g: 0.0,
                    dampener_active: true,
                    requires_flip: false,
                }
            }
            FlightPhase::WarpAccelerate => {
                let warp_dir = warp_ap.map(|w| w.direction).unwrap_or(DVec3::NEG_Z);
                let elapsed = physics_time.0 - ap.engage_time;
                let base_accel = 1_000_000.0_f64;
                let warp_accel =
                    (base_accel * (2.0_f64).powf(elapsed / 2.0)).min(10_000_000_000.0);

                GuidanceCommand {
                    thrust_direction: warp_dir,
                    thrust_magnitude: warp_accel * physics.0.mass_kg,
                    phase: FlightPhase::WarpAccelerate,
                    completed: false,
                    eta_real_seconds: 0.0,
                    felt_g: 1.0,
                    dampener_active: true,
                    requires_flip: false,
                }
            }
            FlightPhase::WarpArrival => {
                let warp_dir = warp_ap.map(|w| w.direction).unwrap_or(DVec3::NEG_Z);
                let elapsed = physics_time.0 - ap.engage_time;
                let total_time = ap.estimated_tof;
                let fraction = (elapsed / total_time).clamp(0.0, 1.0);
                let speed = vel.0.length();
                let completed = fraction >= 1.0 || speed < 1_000_000.0;

                GuidanceCommand {
                    thrust_direction: -warp_dir,
                    thrust_magnitude: if completed { 0.0 } else { 1.0 },
                    phase: if completed {
                        FlightPhase::Arrived
                    } else {
                        FlightPhase::WarpArrival
                    },
                    completed,
                    eta_real_seconds: (total_time - elapsed).max(0.0),
                    felt_g: 1.0,
                    dampener_active: true,
                    requires_flip: false,
                }
            }
            FlightPhase::WarpCruise | FlightPhase::WarpDecelerate => GuidanceCommand {
                thrust_direction: DVec3::NEG_Z,
                thrust_magnitude: 0.0,
                phase: ap.phase,
                completed: true,
                eta_real_seconds: 0.0,
                felt_g: 0.0,
                dampener_active: false,
                requires_flip: false,
            },
        };

        // Check phase transitions.
        let new_phase = if matches!(
            ap.phase,
            FlightPhase::WarpAlign
                | FlightPhase::WarpAccelerate
                | FlightPhase::WarpCruise
                | FlightPhase::WarpDecelerate
                | FlightPhase::WarpArrival
        ) {
            guidance.phase
        } else {
            let phase_vel = if in_soi.is_some() {
                DVec3::ZERO
            } else {
                pv
            };
            if let Some(p) = planet {
                autopilot::check_phase_transition(
                    ap.phase,
                    ap.mode,
                    pos.0,
                    vel.0,
                    pp,
                    phase_vel,
                    p,
                    &sys_config.0.star,
                    soi,
                    ap.target_orbit_altitude,
                    &physics.0,
                )
            } else {
                ap.phase
            }
        };

        guidance_cache.push((ship_id.0, guidance, new_phase));
    }

    // Apply phase transitions and guidance to ships.
    for (sid, guidance, new_phase) in guidance_cache.drain(..) {
        for (
            ship_id,
            _pos,
            _vel,
            _rot,
            _physics,
            mut ap,
            _intercept,
            _warp_ap,
            _in_soi,
        ) in &mut ships
        {
            if ship_id.0 != sid {
                continue;
            }

            if new_phase != ap.phase {
                if ap.phase == FlightPhase::WarpAlign
                    && new_phase == FlightPhase::WarpAccelerate
                {
                    // Reset engage_time for the exponential acceleration ramp.
                    ap.engage_time = physics_time.0;
                }
                info!(
                    ship_id = sid,
                    old = ?ap.phase,
                    new = ?new_phase,
                    "flight phase transition"
                );
                ap.phase = new_phase;
            }

            if guidance.completed {
                let should_disengage = matches!(
                    ap.phase,
                    FlightPhase::Arrived | FlightPhase::Landed
                ) || (ap.phase == FlightPhase::StableOrbit
                    && matches!(
                        ap.mode,
                        autopilot::AutopilotMode::OrbitInsertion
                            | autopilot::AutopilotMode::DirectApproach
                    ));
                if should_disengage {
                    let mode = ap.mode;
                    info!(ship_id = sid, mode = ?mode, "autopilot completed — disengaging");
                    ap.phase = FlightPhase::Arrived; // Mark for removal in next pass
                }
            }
            break;
        }
    }
}

/// Apply flip phase transitions for ships in Accelerate/Brake/Flip.
fn autopilot_phase_transition(
    mut commands: Commands,
    mut ships: Query<(
        Entity,
        &ShipId,
        &Rotation,
        &Velocity,
        &mut Autopilot,
        Option<&AutopilotIntercept>,
        Option<&WarpAutopilot>,
        &mut ThrustInput,
        &mut TorqueInput,
    )>,
    planet_vel: Res<PlanetVelocities>,
) {
    for (entity, ship_id, rot, vel, mut ap, _intercept, _warp_ap, mut thrust, mut torque) in
        &mut ships
    {
        // Handle completed autopilot — remove components.
        let should_disengage = matches!(ap.phase, FlightPhase::Arrived | FlightPhase::Landed)
            || (ap.phase == FlightPhase::StableOrbit
                && matches!(
                    ap.mode,
                    autopilot::AutopilotMode::OrbitInsertion
                        | autopilot::AutopilotMode::DirectApproach
                ));
        if should_disengage
            && (ap.phase == FlightPhase::Arrived || ap.phase == FlightPhase::Landed)
        {
            // Check if guidance just marked it — if yes, remove.
            commands
                .entity(entity)
                .remove::<Autopilot>()
                .remove::<AutopilotIntercept>()
                .remove::<WarpAutopilot>();
            thrust.0 = DVec3::ZERO;
            torque.0 = DVec3::ZERO;
            continue;
        }

        // Flip transitions for interplanetary phases.
        if !matches!(
            ap.phase,
            FlightPhase::Accelerate | FlightPhase::Brake | FlightPhase::Flip
        ) {
            continue;
        }

        let planet_index = (ap.target_body_id.wrapping_sub(1)) as usize;
        let pv = if planet_index < planet_vel.0.len() {
            planet_vel.0[planet_index]
        } else {
            DVec3::ZERO
        };

        // Flip → Brake: ship has rotated to face relative retrograde.
        if ap.phase == FlightPhase::Flip {
            let fwd = rot.0 * DVec3::NEG_Z;
            let rel_vel = vel.0 - pv;
            let retrograde = if rel_vel.length() > 0.1 {
                -rel_vel.normalize()
            } else {
                -vel.0.normalize_or_zero()
            };
            if fwd.dot(retrograde) > 0.966 {
                ap.phase = FlightPhase::Brake;
                info!(ship_id = ship_id.0, "autopilot: Flip → Brake (aligned)");
            }
        }
    }
}

/// Compute thrust from autopilot or manual input, then apply rotation control.
/// This system combines guidance → rotation → thrust to avoid passing guidance
/// between systems.
fn thrust_application(
    mut ships: Query<(
        &ShipId,
        &Position,
        &Velocity,
        &mut Rotation,
        &mut AngularVelocity,
        &mut ThrustInput,
        &mut TorqueInput,
        &ShipPhysics,
        &mut Autopilot,
        &AutopilotIntercept,
        Option<&WarpAutopilot>,
        Option<&InSoi>,
    )>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    planet_vel: Res<PlanetVelocities>,
    celestial_time: Res<ecs::CelestialTime>,
    physics_time: Res<ecs::PhysicsTime>,
) {
    for (
        ship_id,
        pos,
        vel,
        rot,
        ang_vel,
        mut thrust,
        mut torque,
        physics,
        mut ap,
        intercept,
        warp_ap,
        in_soi,
    ) in &mut ships
    {
        let planet_index = (ap.target_body_id.wrapping_sub(1)) as usize;
        let is_warp = ap.mode == autopilot::AutopilotMode::WarpTravel;

        let planet = if !is_warp && planet_index < sys_config.0.planets.len() {
            Some(&sys_config.0.planets[planet_index])
        } else {
            None
        };
        let pp = if !is_warp && planet_index < planet_pos.0.len() {
            planet_pos.0[planet_index]
        } else {
            DVec3::ZERO
        };
        let pv = if !is_warp && planet_index < planet_vel.0.len() {
            planet_vel.0[planet_index]
        } else {
            DVec3::ZERO
        };
        let soi = planet
            .map(|p| compute_soi_radius(p, &sys_config.0.star))
            .unwrap_or(0.0);

        let in_atmo = planet
            .map(|p| {
                let alt = (pos.0 - pp).length() - p.radius_m;
                alt < p.atmosphere.atmosphere_height && p.atmosphere.has_atmosphere
            })
            .unwrap_or(false);
        let is_in_soi = in_soi.is_some();
        let effective_tier = autopilot::effective_tier(ap.thrust_tier, in_atmo, is_in_soi);
        let engine_accel = physics.0.engine_acceleration(effective_tier);
        let thrust_force =
            autopilot::engine_tier(effective_tier).thrust_force_n * physics.0.thrust_multiplier;

        // Re-compute guidance (minimal cost — same pure functions).
        let guidance = match ap.phase {
            FlightPhase::Accelerate => autopilot::compute_guidance(
                pos.0,
                vel.0,
                intercept.intercept_pos,
                intercept.target_arrival_vel,
                planet_index,
                &sys_config.0,
                &planet_pos.0,
                celestial_time.0,
                &physics.0,
                effective_tier,
                false,
            ),
            FlightPhase::Brake => {
                let rel_vel = vel.0 - pv;
                let rel_speed = rel_vel.length();
                let v_esc_soi = planet
                    .map(|p| autopilot::escape_velocity(p, soi - p.radius_m))
                    .unwrap_or(0.0);
                if rel_speed > v_esc_soi * 0.5 {
                    GuidanceCommand {
                        thrust_direction: -rel_vel.normalize(),
                        thrust_magnitude: thrust_force,
                        phase: FlightPhase::Brake,
                        completed: false,
                        eta_real_seconds: if engine_accel > 0.0 {
                            rel_speed / engine_accel
                        } else {
                            0.0
                        },
                        felt_g: autopilot::engine_tier(effective_tier).felt_g,
                        dampener_active: autopilot::engine_tier(effective_tier).dampened,
                        requires_flip: false,
                    }
                } else {
                    let to_planet = (pp - pos.0).normalize_or_zero();
                    let gentle_thrust =
                        autopilot::engine_tier(0).thrust_force_n * physics.0.thrust_multiplier;
                    GuidanceCommand {
                        thrust_direction: to_planet,
                        thrust_magnitude: gentle_thrust,
                        phase: FlightPhase::Brake,
                        completed: false,
                        eta_real_seconds: 0.0,
                        felt_g: autopilot::engine_tier(0).felt_g,
                        dampener_active: false,
                        requires_flip: false,
                    }
                }
            }
            FlightPhase::Flip => {
                let rel_vel = vel.0 - pv;
                let retrograde = if rel_vel.length() > 0.1 {
                    -rel_vel.normalize()
                } else {
                    -vel.0.normalize_or_zero()
                };
                GuidanceCommand {
                    thrust_direction: retrograde,
                    thrust_magnitude: 0.0,
                    phase: FlightPhase::Flip,
                    completed: false,
                    eta_real_seconds: 0.0,
                    felt_g: 0.0,
                    dampener_active: false,
                    requires_flip: false,
                }
            }
            FlightPhase::WarpAlign => {
                let warp_dir = warp_ap.map(|w| w.direction).unwrap_or(DVec3::NEG_Z);
                GuidanceCommand {
                    thrust_direction: warp_dir,
                    thrust_magnitude: 0.0,
                    phase: FlightPhase::WarpAlign,
                    completed: false,
                    eta_real_seconds: 0.0,
                    felt_g: 0.0,
                    dampener_active: true,
                    requires_flip: false,
                }
            }
            FlightPhase::WarpAccelerate => {
                let warp_dir = warp_ap.map(|w| w.direction).unwrap_or(DVec3::NEG_Z);
                let elapsed = physics_time.0 - ap.engage_time;
                let base_accel = 1_000_000.0_f64;
                let warp_accel =
                    (base_accel * (2.0_f64).powf(elapsed / 2.0)).min(10_000_000_000.0);
                GuidanceCommand {
                    thrust_direction: warp_dir,
                    thrust_magnitude: warp_accel * physics.0.mass_kg,
                    phase: FlightPhase::WarpAccelerate,
                    completed: false,
                    eta_real_seconds: 0.0,
                    felt_g: 1.0,
                    dampener_active: true,
                    requires_flip: false,
                }
            }
            FlightPhase::WarpArrival => {
                let warp_dir = warp_ap.map(|w| w.direction).unwrap_or(DVec3::NEG_Z);
                let elapsed = physics_time.0 - ap.engage_time;
                let total_time = ap.estimated_tof;
                let fraction = (elapsed / total_time).clamp(0.0, 1.0);
                let speed = vel.0.length();
                let completed = fraction >= 1.0 || speed < 1_000_000.0;
                GuidanceCommand {
                    thrust_direction: -warp_dir,
                    thrust_magnitude: if completed { 0.0 } else { 1.0 },
                    phase: if completed {
                        FlightPhase::Arrived
                    } else {
                        FlightPhase::WarpArrival
                    },
                    completed,
                    eta_real_seconds: (total_time - elapsed).max(0.0),
                    felt_g: 1.0,
                    dampener_active: true,
                    requires_flip: false,
                }
            }
            _ => {
                // For all other phases, thrust was already set or is zero.
                // Skip rotation/thrust application.
                continue;
            }
        };

        // Flip detection: Accelerate → Flip when guidance requires flip.
        if guidance.requires_flip && ap.phase == FlightPhase::Accelerate {
            ap.phase = FlightPhase::Flip;
            ap.braking_committed = true;
            info!(
                ship_id = ship_id.0,
                "autopilot: Accelerate → Flip (braking committed)"
            );
        }

        let thrust_dir = guidance.thrust_direction;
        let elapsed = physics_time.0 - ap.engage_time;
        let tof = ap.estimated_tof;
        let ramp = autopilot::thrust_ramp_factor(elapsed, tof, guidance.phase);

        // Rotation: PD controller to align ship forward (-Z) with thrust direction.
        let needs_rotation = guidance.thrust_magnitude > 0.0
            || matches!(
                guidance.phase,
                FlightPhase::WarpAlign | FlightPhase::Flip
            );
        if needs_rotation {
            let desired_fwd = thrust_dir;
            let current_fwd = rot.0 * DVec3::NEG_Z;
            let mut cross = current_fwd.cross(desired_fwd);
            let dot_align = current_fwd.dot(desired_fwd).clamp(-1.0, 1.0);
            let angle = dot_align.acos();

            if angle > 0.1 && cross.length() < 1e-4 {
                let ship_up = rot.0 * DVec3::Y;
                cross = ship_up;
            }

            if angle > 0.001 && cross.length() > 1e-8 {
                let axis = cross.normalize();
                let p_gain = 3.0;
                let max_ang_vel = physics.0.max_angular_velocity();
                let target_omega = axis * (angle * p_gain).min(max_ang_vel);
                let world_omega = rot.0 * ang_vel.0;
                let error = target_omega - world_omega;
                let d_gain = 1.5;
                let local_error = rot.0.inverse() * (error * d_gain);
                torque.0 = local_error.clamp_length_max(2.0);
            } else {
                let world_omega = rot.0 * ang_vel.0;
                if world_omega.length() > 0.01 {
                    let brake = rot.0.inverse() * (-world_omega * 2.0);
                    torque.0 = brake.clamp_length_max(1.0);
                } else {
                    torque.0 = DVec3::ZERO;
                }
            }
        }

        // Thrust: main drive fires along ship's -Z axis.
        // WarpArrival deceleration is handled by the separate warp_arrival_deceleration system.
        let is_warp_arrival = ap.phase == FlightPhase::WarpArrival;
        if is_warp_arrival {
            thrust.0 = DVec3::ZERO;
            torque.0 = DVec3::ZERO;
        } else if guidance.thrust_magnitude > 0.0 {
            let dot_align = rot.0.mul_vec3(DVec3::NEG_Z).dot(thrust_dir).clamp(-1.0, 1.0);
            let thrust_scale = dot_align.max(0.0) * ramp;
            thrust.0 = DVec3::new(0.0, 0.0, -guidance.thrust_magnitude * thrust_scale);
        } else if needs_rotation {
            thrust.0 = DVec3::ZERO;
        } else {
            thrust.0 = DVec3::ZERO;
            torque.0 = DVec3::ZERO;
        }
    }
}

/// Handle warp arrival deceleration (direct velocity set — not through engine).
fn warp_arrival_deceleration(
    mut ships: Query<(
        &ShipId,
        &mut Velocity,
        &Autopilot,
    )>,
    physics_time: Res<ecs::PhysicsTime>,
) {
    for (_ship_id, mut vel, ap) in &mut ships {
        if ap.mode != autopilot::AutopilotMode::WarpTravel
            || ap.phase != FlightPhase::WarpArrival
        {
            continue;
        }

        let elapsed = physics_time.0 - ap.engage_time;
        let total_time = ap.estimated_tof;
        let v0 = ap.target_orbit_altitude; // entry speed stored here
        let fraction = (elapsed / total_time).clamp(0.0, 1.0);
        let target_speed = v0 * (1.0 - fraction).powi(3);
        if target_speed < 1_000_000.0 || fraction >= 1.0 {
            vel.0 = DVec3::ZERO;
        } else if vel.0.length() > 1.0 {
            vel.0 = vel.0.normalize() * target_speed;
        }
    }
}

/// Velocity Verlet (Störmer-Verlet) two-pass integration.
/// Pass A: compute a_old (gravity + thrust + aero), advance position.
/// Pass B: recompute gravity+aero at new position, average for velocity.
fn physics_integrate(
    mut ships: Query<(
        &ShipId,
        &mut Position,
        &mut Velocity,
        &mut Rotation,
        &mut AngularVelocity,
        &mut ThrustInput,
        &mut TorqueInput,
        &ShipPhysics,
        &mut ThermalState,
        Option<&Landed>,
        Option<&InSoi>,
        Option<&Autopilot>,
    )>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    celestial_time: Res<ecs::CelestialTime>,
) {
    // Collect Pass A data (avoids borrow conflicts).
    let pass_a: Vec<(u64, DVec3, DVec3, DVec3)> = ships
        .iter()
        .filter(|(_, _, _, _, _, _, _, _, _, landed, _, _)| landed.is_none())
        .map(
            |(ship_id, pos, vel, rot, ang_vel, thrust, _torque, physics, thermal, _, in_soi, _ap)| {
                let grav = if let Some(soi) = in_soi {
                    let r = pos.0 - planet_pos.0[soi.planet_index];
                    let dist_sq = r.length_squared();
                    if dist_sq > 1.0 {
                        -r.normalize() * sys_config.0.planets[soi.planet_index].gm / dist_sq
                    } else {
                        DVec3::ZERO
                    }
                } else {
                    compute_gravity_acceleration(
                        pos.0,
                        &sys_config.0.star,
                        &sys_config.0.planets,
                        &planet_pos.0,
                        celestial_time.0,
                    )
                };

                let thrust_accel = rot.0 * thrust.0 / physics.0.mass_kg;

                let mut aero_accel = DVec3::ZERO;
                let mut aero_torque = DVec3::ZERO;
                if let Some((planet_idx, _alt)) =
                    check_atmosphere(pos.0, &sys_config.0.planets, &planet_pos.0)
                {
                    let mut thermal_tmp = thermal.energy_j;
                    let aero = compute_full_aerodynamics(
                        pos.0,
                        vel.0,
                        rot.0,
                        ang_vel.0,
                        planet_pos.0[planet_idx],
                        &sys_config.0.planets[planet_idx],
                        &physics.0,
                        &mut thermal_tmp,
                        DT,
                    );
                    aero_accel = aero.drag_accel + aero.lift_accel;
                    aero_torque = aero.aero_torque;
                }

                let accel_old = grav + thrust_accel + aero_accel;
                (ship_id.0, accel_old, thrust_accel, aero_torque)
            },
        )
        .collect();

    // Advance positions.
    for &(sid, accel_old, _, _) in &pass_a {
        for (ship_id, mut pos, vel, _, _, _, _, _, _, _, _, _) in &mut ships {
            if ship_id.0 == sid {
                pos.0 += vel.0 * DT + 0.5 * accel_old * DT * DT;
                break;
            }
        }
    }

    // Pass B: recompute gravity+aero at new position, average with a_old.
    let pass_b: Vec<(u64, DVec3)> = pass_a
        .iter()
        .map(|&(sid, accel_old, thrust_accel, _)| {
            // Find the ship at its new position.
            let (pos_new, in_soi_idx) = ships
                .iter()
                .find(|(s, _, _, _, _, _, _, _, _, _, _, _)| s.0 == sid)
                .map(|(_, p, _, _, _, _, _, _, _, _, soi, _)| {
                    (p.0, soi.map(|s| s.planet_index))
                })
                .unwrap_or((DVec3::ZERO, None));

            let grav_new = if let Some(pi) = in_soi_idx {
                let r = pos_new - planet_pos.0[pi];
                let dist_sq = r.length_squared();
                if dist_sq > 1.0 {
                    -r.normalize() * sys_config.0.planets[pi].gm / dist_sq
                } else {
                    DVec3::ZERO
                }
            } else {
                compute_gravity_acceleration(
                    pos_new,
                    &sys_config.0.star,
                    &sys_config.0.planets,
                    &planet_pos.0,
                    celestial_time.0,
                )
            };

            let mut aero_accel_new = DVec3::ZERO;
            if let Some((planet_idx, _alt)) =
                check_atmosphere(pos_new, &sys_config.0.planets, &planet_pos.0)
            {
                // For pass B we approximate using current velocity (already advanced).
                // The thermal computation is done once per tick below.
                let ship_data = ships
                    .iter()
                    .find(|(s, _, _, _, _, _, _, _, _, _, _, _)| s.0 == sid);
                if let Some((_, _, v, r, av, _, _, phys, _, _, _, _)) = ship_data {
                    let mut thermal_tmp = 0.0;
                    let aero = compute_full_aerodynamics(
                        pos_new,
                        v.0,
                        r.0,
                        av.0,
                        planet_pos.0[planet_idx],
                        &sys_config.0.planets[planet_idx],
                        &phys.0,
                        &mut thermal_tmp,
                        DT,
                    );
                    aero_accel_new = aero.drag_accel + aero.lift_accel;
                }
            }

            let accel_new = grav_new + thrust_accel + aero_accel_new;
            let vel_delta = 0.5 * (accel_old + accel_new) * DT;
            (sid, vel_delta)
        })
        .collect();

    // Apply velocity, thermal energy, aero torque, angular velocity, and rotation.
    for (sid, vel_delta) in pass_b {
        let aero_torque = pass_a
            .iter()
            .find(|t| t.0 == sid)
            .map(|t| t.3)
            .unwrap_or(DVec3::ZERO);

        for (
            ship_id,
            _pos,
            mut vel,
            mut rot,
            mut ang_vel,
            mut thrust,
            mut torque_input,
            physics,
            mut thermal,
            _landed,
            _in_soi,
            autopilot,
        ) in &mut ships
        {
            if ship_id.0 != sid {
                continue;
            }

            vel.0 += vel_delta;
            thrust.0 = DVec3::ZERO;

            // Update thermal energy (authoritative, once per tick).
            if let Some((planet_idx, _alt)) =
                check_atmosphere(_pos.0, &sys_config.0.planets, &planet_pos.0)
            {
                compute_full_aerodynamics(
                    _pos.0,
                    vel.0,
                    rot.0,
                    ang_vel.0,
                    planet_pos.0[planet_idx],
                    &sys_config.0.planets[planet_idx],
                    &physics.0,
                    &mut thermal.energy_j,
                    DT,
                );
            }

            // Apply aerodynamic torque to angular velocity.
            if aero_torque.length_squared() > 1e-10 {
                let (ix, iy, iz) = physics.0.moment_of_inertia();
                let torque_local = rot.0.inverse() * aero_torque;
                let ang_accel_aero = DVec3::new(
                    torque_local.x / ix,
                    torque_local.y / iy,
                    torque_local.z / iz,
                );
                ang_vel.0 += ang_accel_aero * DT;
            }

            // Angular velocity control (manual + autopilot torque).
            let max_ang_vel_limit = physics.0.max_angular_velocity();
            let ang_accel_rate = physics.0.angular_acceleration();
            let target_angular_vel = DVec3::new(
                torque_input.0.x.clamp(-1.0, 1.0) * max_ang_vel_limit,
                torque_input.0.y.clamp(-1.0, 1.0) * max_ang_vel_limit,
                torque_input.0.z.clamp(-1.0, 1.0) * max_ang_vel_limit,
            );

            let diff = target_angular_vel - ang_vel.0;
            let ang_accel = diff.clamp_length_max(ang_accel_rate * DT);
            ang_vel.0 += ang_accel;

            let is_warp_rotation = autopilot
                .map(|a| a.mode == autopilot::AutopilotMode::WarpTravel)
                .unwrap_or(false);
            if torque_input.0.length_squared() < 0.001 && !is_warp_rotation {
                ang_vel.0 *= 0.95;
                if ang_vel.0.length_squared() < 1e-6 {
                    ang_vel.0 = DVec3::ZERO;
                }
            }

            let world_angular_vel = rot.0 * ang_vel.0;
            let ang_speed = world_angular_vel.length();
            if ang_speed > 1e-8 {
                let axis = world_angular_vel / ang_speed;
                let delta_rot = DQuat::from_axis_angle(axis, ang_speed * DT);
                rot.0 = (delta_rot * rot.0).normalize();
            }

            torque_input.0 = DVec3::ZERO;
            break;
        }
    }
}

/// Ground contact — spring-damper model. After integration so velocity includes gravity.
fn ground_contact(
    mut commands: Commands,
    mut ships: Query<(
        Entity,
        &ShipId,
        &mut Position,
        &mut Velocity,
        &mut AngularVelocity,
        &ShipPhysics,
        &mut LandingZoneDebounce,
        Option<&Landed>,
        Option<&mut ThrustInput>,
    )>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    celestial_time: Res<ecs::CelestialTime>,
) {
    for (
        entity,
        ship_id,
        mut pos,
        mut vel,
        mut ang_vel,
        physics,
        mut debounce,
        landed,
        _thrust,
    ) in &mut ships
    {
        // Skip ground contact for already-landed ships (position is derived).
        if landed.is_some() {
            continue;
        }

        let mut contact_found = false;

        for (i, planet) in sys_config.0.planets.iter().enumerate() {
            let planet_p = planet_pos.0[i];
            let to_ship = pos.0 - planet_p;
            let dist = to_ship.length();
            let contact_margin = physics.0.landing_gear_height + 0.5;
            let surface_dist = planet.radius_m + contact_margin;

            if dist < surface_dist && dist > 1.0 {
                contact_found = true;
                let radial = to_ship / dist;
                let penetration = surface_dist - dist;
                let v_radial = vel.0.dot(radial);

                // Spring-damper from ship mass and planet gravity.
                let weight = physics.0.mass_kg * planet.surface_gravity;
                let spring_k = weight * 5.0 / contact_margin;
                let damping_c = 2.0 * (spring_k * physics.0.mass_kg).sqrt();

                let spring_force = spring_k * penetration;
                let damping_force = -damping_c * v_radial;
                let ground_force = (spring_force + damping_force).max(0.0);
                let ground_accel = radial * ground_force / physics.0.mass_kg;

                // Friction: kill lateral velocity.
                let friction_rate = planet.surface_gravity / 2.0;

                // Impact damage on first contact.
                if v_radial < -0.5 {
                    let impact_speed = -v_radial;
                    let max_safe = physics.0.hull_strength * 0.08;
                    let lethal = physics.0.hull_strength * 0.3;
                    if impact_speed > lethal {
                        info!(
                            ship_id = ship_id.0,
                            impact_speed = format!("{:.1}", impact_speed),
                            "ship destroyed on impact"
                        );
                    } else if impact_speed > max_safe {
                        let damage = (impact_speed - max_safe).powi(2) * 0.5;
                        info!(
                            ship_id = ship_id.0,
                            impact_speed = format!("{:.1}", impact_speed),
                            damage = format!("{:.1}", damage),
                            "hard landing — damage taken"
                        );
                    }
                }

                // Apply ground reaction.
                vel.0 += ground_accel * DT;

                // Apply friction.
                let friction_decay = (-friction_rate * DT).exp();
                let v_rad_component = radial * vel.0.dot(radial);
                let v_lat_component = vel.0 - v_rad_component;
                vel.0 = v_rad_component + v_lat_component * friction_decay;

                // Prevent sinking below surface.
                let dist_now = (pos.0 - planet_p).length();
                if dist_now < planet.radius_m + 0.5 {
                    pos.0 = planet_p + radial * (planet.radius_m + contact_margin);
                }

                // Landing detection.
                let to_ship_check = pos.0 - planet_p;
                let dist_check = to_ship_check.length();
                let altitude_check = dist_check - planet.radius_m;
                let radial_check = to_ship_check / dist_check;
                let v_radial_check = vel.0.dot(radial_check);

                let landing_zone = physics.0.landing_gear_height + 1.0;
                let max_landing_speed = 2.0 * planet.surface_gravity.sqrt();

                if altitude_check < landing_zone && v_radial_check.abs() < max_landing_speed {
                    debounce.consecutive_ticks += 1;
                    if debounce.consecutive_ticks >= 10 {
                        commands.entity(entity).insert(Landed {
                            planet_index: i,
                            surface_radial: radial_check,
                            celestial_time: celestial_time.0,
                        });
                        vel.0 = DVec3::ZERO;
                        ang_vel.0 = DVec3::ZERO;
                        debounce.consecutive_ticks = 0;
                        info!(
                            ship_id = ship_id.0,
                            planet_index = i,
                            alt = format!("{:.1}", altitude_check),
                            "ship landed — surface attached"
                        );
                    }
                } else {
                    debounce.consecutive_ticks = 0;
                }

                break; // Only check nearest planet.
            }
        }

        if !contact_found {
            debounce.consecutive_ticks = 0;
        }
    }
}

/// Thermal update — atmospheric heating (already handled inline in physics_integrate).
/// This is a no-op placeholder for the system set ordering.
fn thermal_update() {
    // Thermal energy is updated in physics_integrate as part of the aerodynamics pass.
}

// ---------------------------------------------------------------------------
// Detection systems
// ---------------------------------------------------------------------------

/// SOI boundary detection with frame conversion (every tick).
fn soi_detection(
    mut commands: Commands,
    mut ships: Query<(
        Entity,
        &ShipId,
        &Position,
        &mut Velocity,
        Option<&InSoi>,
        Option<&OrbitStabilizer>,
    )>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    planet_vel: Res<PlanetVelocities>,
) {
    let mut soi_entries: Vec<(Entity, u64, usize)> = Vec::with_capacity(4);
    let mut soi_exits: Vec<(Entity, u64, usize)> = Vec::with_capacity(4);

    // Cache SOI radii per planet to avoid redundant compute_soi_radius calls.
    let soi_radii: Vec<f64> = sys_config.0.planets.iter()
        .map(|p| compute_soi_radius(p, &sys_config.0.star))
        .collect();

    for (entity, ship_id, pos, _vel, in_soi, _orbit_stab) in &ships {
        let mut found_soi = false;
        for (i, _planet) in sys_config.0.planets.iter().enumerate() {
            let dist = (pos.0 - planet_pos.0[i]).length();
            if dist < soi_radii[i] {
                found_soi = true;
                if in_soi.map(|s| s.planet_index) != Some(i) {
                    soi_entries.push((entity, ship_id.0, i));
                }
                break;
            }
        }
        if !found_soi {
            if let Some(soi_data) = in_soi {
                let pi = soi_data.planet_index;
                let dist_exit = (pos.0 - planet_pos.0[pi]).length();
                if dist_exit > soi_radii[pi] * 1.02 {
                    soi_exits.push((entity, ship_id.0, pi));
                }
            }
        }
    }

    // Process SOI entries — use direct entity lookup instead of scanning all ships.
    for (entity, sid, planet_index) in &soi_entries {
        let planet_real_vel = planet_vel.0[*planet_index];
        let planet_p = planet_pos.0[*planet_index];
        let planet_radius = sys_config.0.planets[*planet_index].radius_m;
        let planet_gm = sys_config.0.planets[*planet_index].gm;
        let planet_seed = sys_config.0.planets[*planet_index].planet_seed;

        if let Ok((_ent, _ship_id, pos, mut vel, _in_soi, orbit_stab)) = ships.get_mut(*entity) {
            let rel_speed_before = (vel.0 - planet_real_vel).length();
            vel.0 -= planet_real_vel;

            if orbit_stab.is_some() {
                let to_planet = (planet_p - pos.0).normalize_or_zero();
                let radial_speed = vel.0.dot(to_planet);
                let altitude = (pos.0 - planet_p).length() - planet_radius;
                let v_esc = (2.0 * planet_gm / (planet_radius + altitude)).sqrt();
                let approach_speed = radial_speed.max(v_esc * 0.3);
                vel.0 = to_planet * approach_speed;
                info!(
                    ship_id = *sid,
                    planet_index,
                    planet_seed,
                    planet_vel_mag = format!("{:.0}", planet_real_vel.length()),
                    rel_speed = format!("{:.0}", rel_speed_before),
                    approach = format!("{:.0}", approach_speed),
                    "ship entered planet SOI — orbit stabilizer: radial approach"
                );
            } else {
                info!(
                    ship_id = *sid,
                    planet_index,
                    planet_seed,
                    planet_vel_mag = format!("{:.0}", planet_real_vel.length()),
                    rel_speed = format!("{:.0}", rel_speed_before),
                    "ship entered planet SOI — patched conics frame conversion"
                );
            }

            commands.entity(*entity).insert(InSoi {
                planet_index: *planet_index,
            });
        }
    }

    // Process SOI exits — use direct entity lookup instead of scanning all ships.
    for (entity, sid, planet_index) in &soi_exits {
        let planet_real_vel = planet_vel.0[*planet_index];
        let planet_pos_exit = planet_pos.0[*planet_index];

        if let Ok((_ent, _ship_id, pos, mut vel, _in_soi, _orbit_stab)) = ships.get_mut(*entity) {
            let dist = (pos.0 - planet_pos_exit).length();
            vel.0 += planet_real_vel;
            warn!(
                ship_id = *sid,
                planet_index,
                dist = format!("{:.0}", dist),
                soi = format!("{:.0}", soi_radii[*planet_index]),
                vel = format!("{:.0}", vel.0.length()),
                "ship LEFT planet SOI — converted to system frame"
            );

            commands.entity(*entity).remove::<InSoi>();
        }
    }
}

/// Provision planet shards for ships in SOI (every 20 ticks).
fn planet_provisioning(
    ships: Query<(&ShipId, &InSoi)>,
    tick: Res<ecs::TickCounter>,
    sys_config: Res<SystemConfig>,
    provisioned: Res<ProvisionedPlanets>,
    mut in_flight: ResMut<ProvisioningInFlight>,
    orch_url: Res<OrchestratorUrl>,
    provision_tx: Res<PlanetProvisionSender>,
    http_client: Res<HttpClient>,
) {
    if tick.0 % 20 != 0 {
        return;
    }

    // Collect unique planet indices from ships in SOI.
    let mut planet_indices: HashSet<usize> = HashSet::new();
    for (_ship_id, in_soi) in &ships {
        planet_indices.insert(in_soi.planet_index);
    }

    for planet_index in planet_indices {
        let planet = &sys_config.0.planets[planet_index];
        let planet_seed = planet.planet_seed;
        if provisioned.0.contains_key(&planet_seed) || in_flight.0.contains(&planet_seed) {
            continue;
        }

        in_flight.0.insert(planet_seed);
        let url = orch_url.0.clone();
        let system_seed = sys_config.0.system_seed;
        let tx = provision_tx.0.clone();
        let client = http_client.0.clone();
        tokio::spawn(async move {
            let endpoint = format!(
                "{}/planet/{}?system_seed={}&planet_index={}",
                url, planet_seed, system_seed, planet_index
            );
            match client.get(&endpoint).send().await {
                Ok(resp) if resp.status().is_success() => {
                    if let Ok(body) = resp.text().await {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                            if let Some(id) = v
                                .get("shards")
                                .and_then(|s| s.get(0))
                                .and_then(|s| s.get("id"))
                                .and_then(|v| v.as_u64())
                            {
                                let _ = tx.try_send((planet_seed, ShardId(id)));
                            }
                        }
                    }
                }
                Ok(resp) => {
                    tracing::warn!(
                        status = %resp.status(),
                        planet_seed,
                        "failed to provision planet shard"
                    );
                }
                Err(e) => {
                    tracing::warn!(%e, planet_seed, "planet shard provision request failed");
                }
            }
        });
        info!(planet_seed, planet_index, "provisioning planet shard for ship in SOI");
    }
}

/// Warp boundary detection — handoff ships to galaxy shard (every 10 ticks).
fn warp_boundary(
    mut commands: Commands,
    ships: Query<(
        Entity,
        &ShipId,
        &Position,
        &Velocity,
        &Rotation,
        &Autopilot,
        Option<&WarpAutopilot>,
    )>,
    tick: Res<ecs::TickCounter>,
    sys_config: Res<SystemConfig>,
    mut galaxy_ctx: ResMut<GalaxyContext>,
    shard_identity: Res<ShardIdentity>,
    celestial_time: Res<ecs::CelestialTime>,
    mut departed_ships: ResMut<DepartedShips>,
    mut ship_index: ResMut<ecs::ShipEntityIndex>,
    orch_url: Res<OrchestratorUrl>,
    http_client: Res<HttpClient>,
    galaxy_provision_tx: Res<GalaxyProvisionSender>,
    bridge: Res<NetworkBridge>,
) {
    if tick.0 % 10 != 0 {
        return;
    }
    if galaxy_ctx.seed == 0 {
        return;
    }

    let outermost_orbit = sys_config
        .0
        .planets
        .iter()
        .map(|p| p.orbital_elements.sma)
        .fold(1.0e10_f64, f64::max);
    let departure_boundary = outermost_orbit * 20.0;
    let preconnect_boundary = departure_boundary * 0.5;

    let mut needs_provisioning = false;
    let mut ready_departures: Vec<Entity> = Vec::new();

    for (entity, _ship_id, pos, _vel, _rot, ap, warp_ap) in &ships {
        if ap.mode != autopilot::AutopilotMode::WarpTravel {
            continue;
        }
        if warp_ap.is_none() {
            continue;
        }

        let dist = pos.0.length();

        if dist > preconnect_boundary {
            needs_provisioning = true;
        }

        if dist > departure_boundary && galaxy_ctx.provisioned_galaxy.is_some() {
            ready_departures.push(entity);
        }
    }

    // Provision galaxy shard if needed.
    if needs_provisioning
        && galaxy_ctx.provisioned_galaxy.is_none()
        && !galaxy_ctx.galaxy_provisioning_in_flight
    {
        galaxy_ctx.galaxy_provisioning_in_flight = true;
        let url = format!("{}/galaxy/{}", orch_url.0, galaxy_ctx.seed);
        let client = http_client.0.clone();
        let tx = galaxy_provision_tx.0.clone();
        let galaxy_seed_log = galaxy_ctx.seed;
        tokio::spawn(async move {
            info!(galaxy_seed = galaxy_seed_log, %url, "sending galaxy provision request");
            match client.get(&url).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    info!(%status, "galaxy provision response received");
                    if let Ok(body) = resp.text().await {
                        info!(body_len = body.len(), "galaxy provision response body");
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                            if let Some(id) = v
                                .get("info")
                                .and_then(|i| i["id"].as_u64())
                                .or_else(|| v.get("id").and_then(|i| i.as_u64()))
                            {
                                info!(shard_id = id, "galaxy shard provisioned — sending result");
                                let _ = tx.send(ShardId(id)).await;
                            } else {
                                warn!(
                                    "galaxy provision response has no shard id: {}",
                                    &body[..body.len().min(200)]
                                );
                            }
                        } else {
                            warn!(
                                "galaxy provision response not valid JSON: {}",
                                &body[..body.len().min(200)]
                            );
                        }
                    }
                }
                Err(e) => warn!(%e, "failed to provision galaxy shard (HTTP error)"),
            }
        });
        info!("provisioning galaxy shard for warp departure");
    }

    // Velocity capping for ships past boundary without a ready galaxy shard
    // is handled by the separate warp_velocity_cap system.

    // Process departures.
    if ready_departures.is_empty() {
        return;
    }
    let galaxy_shard_id = match galaxy_ctx.provisioned_galaxy {
        Some(id) => id,
        None => return,
    };

    let galaxy_endpoint = if let Ok(reg) = bridge.peer_registry.try_read() {
        reg.get(galaxy_shard_id).map(|i| {
            (
                i.endpoint.quic_addr,
                i.endpoint.tcp_addr.to_string(),
                i.endpoint.udp_addr.to_string(),
            )
        })
    } else {
        None
    };
    let (galaxy_quic_addr, galaxy_tcp, galaxy_udp) = match galaxy_endpoint {
        Some(ep) => ep,
        None => {
            warn!("galaxy shard provisioned but not in peer registry yet");
            return;
        }
    };
    if galaxy_tcp.is_empty() || galaxy_udp.is_empty() {
        warn!(
            galaxy_shard = galaxy_shard_id.0,
            "galaxy shard has empty TCP/UDP address — deferring warp departure"
        );
        return;
    }

    let ship_shards: Vec<(ShardId, SocketAddr, Option<u64>)> =
        if let Ok(reg) = bridge.peer_registry.try_read() {
            reg.find_by_type(ShardType::Ship)
                .iter()
                .map(|s| (s.id, s.endpoint.quic_addr, s.ship_id))
                .collect()
        } else {
            Vec::new()
        };

    let shard_id = shard_identity.0;
    let galaxy_seed_local = galaxy_ctx.seed;
    let star_index = galaxy_ctx.star_index;
    let star_pos_gu = galaxy_ctx.star_position_gu;
    let ct = celestial_time.0;
    let tc = tick.0;

    for entity in ready_departures {
        let Ok((_, ship_id, pos, vel, rot, _ap, warp_ap)) = ships.get(entity) else {
            continue;
        };
        let Some(warp) = warp_ap else { continue };

        let vel_gu = vel.0 / voxeldust_core::galaxy::GALAXY_UNIT_IN_BLOCKS;
        let pos_gu = voxeldust_core::galaxy::system_to_galaxy(star_pos_gu, pos.0);

        let ship_shard_entry = ship_shards
            .iter()
            .find(|(_, _, sid)| *sid == Some(ship_id.0));

        let player_handoff = handoff::PlayerHandoff {
            session_token: SessionToken(ship_id.0),
            player_name: format!("ship-{}", ship_id.0),
            position: pos.0,
            velocity: vel.0,
            rotation: rot.0,
            forward: rot.0 * DVec3::NEG_Z,
            fly_mode: false,
            speed_tier: 0,
            grounded: false,
            health: 100.0,
            shield: 100.0,
            source_shard: shard_id,
            source_tick: tc,
            target_star_index: None,
            galaxy_context: Some(handoff::GalaxyHandoffContext {
                galaxy_seed: galaxy_seed_local,
                star_index,
                star_position: star_pos_gu,
            }),
            target_planet_seed: None,
            target_planet_index: None,
            target_ship_id: None,
            target_ship_shard_id: ship_shard_entry.map(|(id, _, _)| *id),
            ship_system_position: None,
            ship_rotation: None,
            game_time: ct,
            warp_target_star_index: Some(warp.target_star_index),
            warp_velocity_gu: Some(vel_gu),
            target_system_eva: false,
        };
        let _ = bridge.quic_send_tx.try_send((
            galaxy_shard_id,
            galaxy_quic_addr,
            ShardMsg::PlayerHandoff(player_handoff),
        ));

        // Send HostSwitch to ship shard.
        if let Some((ship_shard_id, ship_quic_addr, _)) = ship_shard_entry {
            let host_switch = ShardMsg::HostSwitch(voxeldust_core::shard_message::HostSwitchData {
                ship_id: ship_id.0,
                new_host_shard_id: galaxy_shard_id,
                new_host_quic_addr: galaxy_quic_addr.to_string(),
                new_host_tcp_addr: galaxy_tcp.clone(),
                new_host_udp_addr: galaxy_udp.clone(),
                new_host_shard_type: 3, // Galaxy
                seed: galaxy_seed_local,
            });
            let _ = bridge
                .quic_send_tx
                .try_send((*ship_shard_id, *ship_quic_addr, host_switch));
            info!(
                ship_id = ship_id.0,
                new_host = galaxy_shard_id.0,
                "sent HostSwitch to ship shard"
            );
        }

        departed_ships.0.insert(ship_id.0);
        ship_index.0.remove(&ship_id.0);
        commands.entity(entity).despawn();

        info!(
            ship_id = ship_id.0,
            pos_gu = format!("({:.1},{:.1},{:.1})", pos_gu.x, pos_gu.y, pos_gu.z),
            "warp departure — handoff to galaxy shard complete"
        );
    }
}

/// Cap velocity for warp ships past departure boundary when galaxy shard isn't ready.
fn warp_velocity_cap(
    mut ships: Query<(
        &ShipId,
        &Position,
        &mut Velocity,
        &Autopilot,
        Option<&WarpAutopilot>,
    )>,
    sys_config: Res<SystemConfig>,
    galaxy_ctx: Res<GalaxyContext>,
    tick: Res<ecs::TickCounter>,
) {
    if tick.0 % 10 != 0 {
        return;
    }
    if galaxy_ctx.provisioned_galaxy.is_some() {
        return;
    }

    let outermost_orbit = sys_config
        .0
        .planets
        .iter()
        .map(|p| p.orbital_elements.sma)
        .fold(1.0e10_f64, f64::max);
    let departure_boundary = outermost_orbit * 20.0;
    let max_boundary_speed =
        autopilot::WARP_MAX_SPEED_GU * voxeldust_core::galaxy::GALAXY_UNIT_IN_BLOCKS;

    for (_ship_id, pos, mut vel, ap, warp_ap) in &mut ships {
        if ap.mode != autopilot::AutopilotMode::WarpTravel {
            continue;
        }
        if warp_ap.is_none() {
            continue;
        }
        if pos.0.length() > departure_boundary {
            let speed = vel.0.length();
            if speed > max_boundary_speed {
                vel.0 = vel.0.normalize() * max_boundary_speed;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Visibility systems
// ---------------------------------------------------------------------------

/// Ensure newly spawned ships get a ShipVisibilitySet component.
/// Process incoming ShipColliderSync messages: attach collider cache to ship entities.
/// The actual Rapier physics world is not created here — it will be added in Phase 4 (EVA).
fn process_ship_collider_sync(
    mut events: MessageReader<ShipColliderSyncMsg>,
    ship_index: Res<ecs::ShipEntityIndex>,
    mut commands: Commands,
) {
    for event in events.read() {
        if let Some(&entity) = ship_index.0.get(&event.ship_id) {
            commands.entity(entity).insert(ShipColliderCache {
                chunks: event.data.chunks.clone(),
                hull_min: event.data.hull_min,
                hull_max: event.data.hull_max,
            });
            info!(
                ship_id = event.ship_id,
                chunks = event.data.chunks.len(),
                "cached ship collider shapes for EVA"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// EVA systems
// ---------------------------------------------------------------------------

/// Process player input from clients connected directly to the system shard.
/// Routes input to the correct EVA player entity via EvaPlayerIndex.
fn eva_input(
    mut bridge: ResMut<NetworkBridge>,
    eva_index: Res<EvaPlayerIndex>,
    mut eva_players: Query<&mut EvaInputState, With<EvaPlayer>>,
) {
    // EVA players connect directly to the system shard via TCP/UDP.
    // Input arrives via the same bridge.input_rx channel as ship inputs.
    // We look up session from UDP address, then map to EVA entity.
    for _ in 0..64 {
        let (src, input) = match bridge.input_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        // For EVA input routing: since there are few EVA players, iterate
        // all entities rather than index lookup (session tokens change on reconnect
        // and would require a secondary index to map new→old tokens).
        // Find the first EVA player — for multi-EVA, would need UDP→name mapping.
        let Some(mut input_state) = eva_players.iter_mut().next() else {
            continue;
        };

        input_state.prev_action = input_state.action;
        input_state.action = input.action;
        input_state.look_yaw = input.look_yaw;
        input_state.look_pitch = input.look_pitch;
        input_state.movement = input.movement;
    }
}

/// EVA physics: apply gravity and jetpack thrust, integrate velocity.
/// Uses the same Velocity Verlet integration as ships, but with jetpack
/// thrust instead of engine thrust.
fn eva_physics(
    mut eva_players: Query<(
        &mut EvaPosition, &mut EvaVelocity, &EvaRotation,
        &EvaPhysics, &EvaInputState,
    ), With<EvaPlayer>>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    old_planet_pos: Res<OldPlanetPositions>,
    celestial_time: Res<ecs::CelestialTime>,
) {
    let dt = DT;

    // Pre-compute SOI radii for the gravity check.
    let soi_radii: Vec<f64> = sys_config.0.planets.iter()
        .map(|p| compute_soi_radius(p, &sys_config.0.star))
        .collect();

    for (mut pos, mut vel, rot, physics, input) in &mut eva_players {
        // SOI co-move: ships inside a planet's SOI get `soi_co_movement` which
        // adds the per-tick planet-position delta to their `Position` so they
        // track the planet through its orbit around the star. EVA players must
        // receive the SAME co-move or they drift apart from the ship (and the
        // planet) at ~110 km/s (planet orbital velocity). Without this,
        // AOI distance from EVA → ship grows out of the 200 km coarse tier in
        // under two seconds, making ships vanish from EVA HUDs right after exit.
        for (i, planet) in sys_config.0.planets.iter().enumerate() {
            let r = pos.0 - planet_pos.0[i];
            let dist = r.length();
            if dist < soi_radii[i] {
                let delta = planet_pos.0[i] - old_planet_pos.0[i];
                pos.0 += delta;
                break;
            }
        }
        // SOI-aware gravity: same as ship physics_integrate.
        // When inside a planet's SOI, use only that planet's gravity.
        // This matches the ship's reference frame so the EVA player follows
        // the same orbital trajectory as an unpowered ship.
        let compute_grav = |p: DVec3| -> DVec3 {
            // Check if inside any planet's SOI.
            for (i, planet) in sys_config.0.planets.iter().enumerate() {
                let r = p - planet_pos.0[i];
                let dist = r.length();
                if dist < soi_radii[i] && dist > 1.0 {
                    return -r.normalize() * planet.gm / (dist * dist);
                }
            }
            // Not in any SOI — use full N-body gravity.
            compute_gravity_acceleration(
                p,
                &sys_config.0.star,
                &sys_config.0.planets,
                &planet_pos.0,
                celestial_time.0,
            )
        };

        let grav = compute_grav(pos.0);

        // Jetpack thrust: convert movement input to world-space thrust.
        // Movement is relative to player orientation.
        let (sin_y, cos_y) = (input.look_yaw as f64).sin_cos();
        let fwd = DVec3::new(cos_y, 0.0, sin_y);
        let right = DVec3::new(-sin_y, 0.0, cos_y);
        let up = DVec3::Y;

        let thrust_dir = fwd * input.movement[2] as f64
            + right * input.movement[0] as f64
            + up * input.movement[1] as f64;
        let thrust_accel = if thrust_dir.length_squared() > 0.0 {
            thrust_dir.normalize() * (physics.jetpack_thrust_n / physics.mass_kg)
        } else {
            DVec3::ZERO
        };

        // Atmospheric drag — same formula as ship drag but with the player's
        // scalar cross-section + Cd (no orientation-dependent areas).
        // Without this, free-fall in atmosphere is unrestrained and the player
        // reaches hypersonic speeds; ship fall times would differ by ~90×.
        let drag_accel = compute_eva_drag(pos.0, vel.0, physics, &sys_config.0, &planet_pos.0, dt);

        // Velocity Verlet integration (same as ship physics_integrate).
        // Pass A: advance position using current velocity + half-step accel.
        let accel_old = grav + thrust_accel + drag_accel;
        pos.0 += vel.0 * dt + 0.5 * accel_old * dt * dt;

        // Pass B: recompute gravity + drag at new position, average with old accel.
        let grav_new = compute_grav(pos.0);
        let drag_new = compute_eva_drag(pos.0, vel.0, physics, &sys_config.0, &planet_pos.0, dt);
        let accel_new = grav_new + thrust_accel + drag_new;
        vel.0 += 0.5 * (accel_old + accel_new) * dt;
    }
}

/// Compute atmospheric drag acceleration on an EVA player.
/// Mirrors the ship's aerodynamic drag model but with the player's scalar
/// cross-section + drag coefficient stored on `EvaPhysics`. Returns `DVec3::ZERO`
/// when outside any planet's atmosphere.
fn compute_eva_drag(
    pos: DVec3,
    vel: DVec3,
    physics: &EvaPhysics,
    sys: &voxeldust_core::system::SystemParams,
    planet_positions: &[DVec3],
    dt: f64,
) -> DVec3 {
    let Some((planet_idx, altitude)) =
        check_atmosphere(pos, &sys.planets, planet_positions)
    else {
        return DVec3::ZERO;
    };
    let density = sys.planets[planet_idx].atmosphere.density_at_altitude(altitude);
    if density < 1e-10 {
        return DVec3::ZERO;
    }
    let speed = vel.length();
    if speed < 0.01 {
        return DVec3::ZERO;
    }
    // a = -0.5 * Cd * (A/m) * rho * |v|² * v_hat
    let drag_force = 0.5
        * density
        * speed
        * speed
        * physics.drag_coefficient
        * physics.drag_area_m2;
    // Stability clamp — matches ship physics — prevents velocity reversal.
    let mag = (drag_force / physics.mass_kg).min(0.5 * speed / dt);
    -(vel / speed) * mag
}

/// Include EVA players in the WorldState broadcast to direct UDP clients.
fn eva_broadcast(
    // One EVA query, one ship query. Earlier versions had two of each with
    // overlapping component access; under bevy_ecs 0.18 multi_threaded this
    // caused the second query to come back empty at runtime so the AOI
    // pipeline produced zero entities for EVA clients. A single query per
    // kind eliminates the conflict.
    eva_players: Query<(
        &EvaSession,
        &EvaPlayerName,
        &EvaPosition,
        &EvaVelocity,
        &EvaRotation,
    ), With<EvaPlayer>>,
    ships: Query<(&ShipId, &Position, &Velocity, &Rotation, Option<&ShipPhysics>)>,
    surface: Res<SurfacePlayerRegistry>,
    vis_cfg: Res<VisibilityConfig>,
    sys_config: Res<SystemConfig>,
    shard_identity: Res<ShardIdentity>,
    planet_pos: Res<PlanetPositions>,
    celestial_time: Res<ecs::CelestialTime>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
) {
    // Only broadcast if there are EVA players connected.
    if eva_players.is_empty() {
        return;
    }
    if tick.0 % 20 == 0 {
        let count = eva_players.iter().count();
        let first_pos = eva_players.iter().next().map(|(_, _, p, _, _)| p.0);
        info!(
            eva_count = count,
            first_pos = ?first_pos,
            "eva_broadcast: sending WorldState with EVA players"
        );
    }

    let bodies: Vec<CelestialBodyData> = {
        let mut b = Vec::with_capacity(1 + sys_config.0.planets.len());
        b.push(CelestialBodyData {
            body_id: 0,
            position: DVec3::ZERO,
            radius: sys_config.0.star.radius_m,
            color: sys_config.0.star.color,
        });
        for (i, planet) in sys_config.0.planets.iter().enumerate() {
            b.push(CelestialBodyData {
                body_id: (i + 1) as u32,
                position: planet_pos.0[i],
                radius: planet.radius_m,
                color: planet.color,
            });
        }
        b
    };

    // Build player snapshots for all EVA players.
    let player_snapshots: Vec<voxeldust_core::client_message::PlayerSnapshotData> = eva_players
        .iter()
        .map(|(session, _name, pos, vel, rot)| voxeldust_core::client_message::PlayerSnapshotData {
            player_id: session.0.0,
            position: pos.0,
            rotation: rot.0,
            velocity: vel.0,
            grounded: false,
            health: 100.0,
            shield: 100.0,
            seated: false,
        })
        .collect();

    // Compute lighting from first EVA player's position.
    let observer_pos = eva_players
        .iter()
        .next()
        .map(|(_, _, p, _, _)| p.0)
        .unwrap_or(DVec3::new(1e11, 0.0, 0.0));
    let lighting = compute_lighting(observer_pos, &sys_config.0.star);

    // Include all ships in the WorldState so the EVA client can see them (legacy field).
    // Derived from the unified `ships` query — ignore velocity/physics for the snapshot.
    let ship_snapshots: Vec<voxeldust_core::client_message::ShipRenderData> = ships
        .iter()
        .map(|(id, pos, _, rot, _)| voxeldust_core::client_message::ShipRenderData {
            ship_id: id.0,
            position: pos.0,
            rotation: rot.0,
            is_own_ship: false,
        })
        .collect();

    // Build the unified `entities` field: union of every EVA observer's AOI,
    // deduplicated by entity_id with the finest observed LOD tier winning.
    // Since broadcast_tx fans out to all connected UDP clients, each EVA receives
    // the union — acceptable for typical player counts; can be split per-client later.
    let entities = {
        let ship_shards_by_id: HashMap<u64, ShardId> = {
            if let Ok(reg) = bridge.peer_registry.try_read() {
                reg.find_by_type(ShardType::Ship)
                    .iter()
                    .filter_map(|info| {
                        let sid = info.ship_id?;
                        if info.host_shard_id != Some(shard_identity.0) {
                            return None;
                        }
                        Some((sid, info.id))
                    })
                    .collect()
            } else {
                HashMap::new()
            }
        };
        let seed_to_index: HashMap<u64, u32> = sys_config
            .0
            .planets
            .iter()
            .enumerate()
            .map(|(i, p)| (p.planet_seed, i as u32))
            .collect();
        let planet_shards_by_index: HashMap<u32, ShardId> = {
            if let Ok(reg) = bridge.peer_registry.try_read() {
                reg.find_by_type(ShardType::Planet)
                    .iter()
                    .filter_map(|info| {
                        let seed = info.planet_seed?;
                        seed_to_index.get(&seed).map(|&idx| (idx, info.id))
                    })
                    .collect()
            } else {
                HashMap::new()
            }
        };
        let candidates = collect_aoi_candidates(
            &ships,
            &eva_players,
            &surface,
            &ship_shards_by_id,
            &planet_shards_by_index,
        );
        let mut merged: HashMap<u64, ObservableEntityData> = HashMap::new();
        for (session, _, pos, _, _) in eva_players.iter() {
            let per = compute_aoi(pos.0, &candidates, Some(session.0.0), &vis_cfg);
            for e in per {
                merged
                    .entry(e.entity_id)
                    .and_modify(|existing| {
                        // Keep the finer (lower-number) tier so EVA sees best-available detail.
                        if (e.lod_tier.as_u8()) < existing.lod_tier.as_u8() {
                            *existing = e.clone();
                        }
                    })
                    .or_insert(e);
            }
        }
        merged.into_values().collect::<Vec<_>>()
    };

    // Diagnostic: log entity count every second so we can verify the unified
    // AOI pipeline is populating the EVA broadcast with ships + players.
    if tick.0 % 20 == 0 {
        let ship_n = entities
            .iter()
            .filter(|e| matches!(e.kind, EntityKind::Ship))
            .count();
        let player_n = entities.len() - ship_n;
        info!(
            entity_count = entities.len(),
            ships = ship_n,
            players = player_n,
            legacy_ship_count = ship_snapshots.len(),
            "eva_broadcast: entities snapshot"
        );
    }

    let ws = ServerMsg::WorldState(WorldStateData {
        tick: tick.0,
        origin: DVec3::ZERO,
        players: player_snapshots,
        bodies,
        ships: ship_snapshots,
        lighting: Some(LightingData {
            sun_direction: lighting.sun_direction,
            sun_color: lighting.sun_color,
            sun_intensity: lighting.sun_intensity,
            ambient: lighting.ambient,
        }),
        game_time: celestial_time.0,
        warp_target_star_index: 0xFFFFFFFF,
        autopilot: None,
        sub_grids: vec![],
        entities,
    });
    if bridge.broadcast_tx.try_send(ws).is_err() {
        tracing::warn!("EVA WorldState broadcast dropped — channel full");
    }
}

/// EVA boarding detection: when an EVA player is near a ship, allow boarding via action key.
/// Uses distance check against ShipColliderCache hull bounds (when available)
/// or fallback distance threshold. Triggers handoff to ship shard.
fn eva_boarding_detection(
    eva_players: Query<(Entity, &EvaSession, &EvaPlayerName, &EvaPosition, &EvaInputState), With<EvaPlayer>>,
    ships: Query<(&ShipId, &Position, Option<&ShipColliderCache>)>,
    mut eva_index: ResMut<EvaPlayerIndex>,
    shard_identity: Res<ShardIdentity>,
    bridge: Res<NetworkBridge>,
    tick: Res<ecs::TickCounter>,
    mut commands: Commands,
) {
    if tick.0 % 10 != 0 {
        return;
    }

    for (eva_entity, session, name, eva_pos, input) in &eva_players {
        // Check for action key press (edge detection).
        let action_pressed = input.action == 3 && input.prev_action != 3;
        if !action_pressed {
            continue;
        }

        // Find the nearest ship.
        let mut nearest: Option<(u64, f64)> = None;
        for (ship_id, ship_pos, collider_cache) in &ships {
            let dist = (eva_pos.0 - ship_pos.0).length();

            // Boarding range: hull AABB diagonal / 2 + margin, or fallback 15m.
            let boarding_range = collider_cache
                .map(|c| {
                    let diag = (c.hull_max - c.hull_min).length() as f64;
                    diag / 2.0 + 5.0 // half diagonal + 5m margin
                })
                .unwrap_or(15.0);

            if dist < boarding_range {
                if nearest.map_or(true, |(_, d)| dist < d) {
                    nearest = Some((ship_id.0, dist));
                }
            }
        }

        let Some((target_ship_id, dist)) = nearest else { continue };

        // Find the ship shard for this ship.
        let ship_shard = if let Ok(reg) = bridge.peer_registry.try_read() {
            reg.find_by_type(ShardType::Ship)
                .iter()
                .find(|s| s.ship_id == Some(target_ship_id))
                .map(|s| (s.id, s.endpoint.quic_addr))
        } else {
            None
        };
        let Some((ship_shard_id, ship_quic_addr)) = ship_shard else { continue };

        // Create boarding handoff.
        let h = handoff::PlayerHandoff {
            session_token: session.0,
            player_name: name.0.clone(),
            position: eva_pos.0,
            velocity: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            forward: DVec3::NEG_Z,
            fly_mode: false,
            speed_tier: 0,
            grounded: false,
            health: 100.0,
            shield: 100.0,
            source_shard: shard_identity.0,
            source_tick: tick.0,
            target_star_index: None,
            galaxy_context: None,
            target_planet_seed: None,
            target_planet_index: None,
            target_ship_id: Some(target_ship_id),
            target_ship_shard_id: Some(ship_shard_id),
            ship_system_position: None,
            ship_rotation: None,
            game_time: 0.0,
            warp_target_star_index: None,
            warp_velocity_gu: None,
            target_system_eva: false,
        };
        let msg = ShardMsg::PlayerHandoff(h);
        let _ = bridge.quic_send_tx.try_send((ship_shard_id, ship_quic_addr, msg));

        // Despawn EVA entity.
        eva_index.0.remove(&name.0);
        commands.entity(eva_entity).despawn();

        info!(
            session = session.0.0,
            ship_id = target_ship_id,
            dist = format!("{:.1}m", dist),
            "EVA boarding — handoff to ship shard"
        );
    }
}

/// EVA atmosphere entry: when an EVA player descends into a planet's atmosphere,
/// hand them off to the planet shard. SOI alone is not sufficient — a player
/// in orbit (inside SOI but above atmosphere) stays as an EVA entity on the
/// system shard with gravity applied. Only atmosphere entry triggers the handoff.
fn eva_soi_detection(
    eva_players: Query<(Entity, &EvaSession, &EvaPlayerName, &EvaPosition, &EvaVelocity), With<EvaPlayer>>,
    mut eva_index: ResMut<EvaPlayerIndex>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    provisioned_planets: Res<ProvisionedPlanets>,
    shard_identity: Res<ShardIdentity>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
    mut commands: Commands,
) {
    for (eva_entity, session, name, eva_pos, eva_vel) in &eva_players {
        for (i, planet) in sys_config.0.planets.iter().enumerate() {
            if !planet.atmosphere.has_atmosphere {
                continue;
            }
            let dist = (eva_pos.0 - planet_pos.0[i]).length();
            let altitude = dist - planet.radius_m;
            if altitude < planet.atmosphere.atmosphere_height && altitude > 0.0 {
                // EVA player entered planet atmosphere — hand off to planet shard.
                let planet_seed = planet.planet_seed;
                let Some(&planet_shard_id) = provisioned_planets.0.get(&planet_seed) else {
                    warn!(planet_seed, "EVA player entered SOI but no planet shard provisioned");
                    continue;
                };

                let h = handoff::PlayerHandoff {
                    session_token: session.0,
                    player_name: name.0.clone(),
                    position: eva_pos.0,
                    velocity: eva_vel.0,
                    rotation: DQuat::IDENTITY,
                    forward: DVec3::NEG_Z,
                    fly_mode: false,
                    speed_tier: 0,
                    grounded: false,
                    health: 100.0,
                    shield: 100.0,
                    source_shard: shard_identity.0,
                    source_tick: tick.0,
                    target_star_index: None,
                    galaxy_context: None,
                    target_planet_seed: Some(planet_seed),
                    target_planet_index: Some(i as u32),
                    target_ship_id: None,
                    target_ship_shard_id: None,
                    ship_system_position: None,
                    ship_rotation: None,
                    game_time: 0.0,
                    warp_target_star_index: None,
                    warp_velocity_gu: None,
                    target_system_eva: false,
                };

                if let Ok(reg) = bridge.peer_registry.try_read() {
                    if let Some(addr) = reg.quic_addr(planet_shard_id) {
                        let msg = ShardMsg::PlayerHandoff(h);
                        let _ = bridge.quic_send_tx.try_send((planet_shard_id, addr, msg));
                    }
                }

                // Despawn EVA entity.
                eva_index.0.remove(&name.0);
                commands.entity(eva_entity).despawn();

                info!(
                    session = session.0.0,
                    planet_seed,
                    planet_index = i,
                    dist = format!("{:.0}m", dist),
                    "EVA player entered planet SOI — handoff to planet shard"
                );
                break; // only one planet transition per tick
            }
        }
    }
}

fn init_visibility_set(
    new_ships: Query<Entity, (With<ShipId>, Without<ShipVisibilitySet>)>,
    mut commands: Commands,
) {
    for entity in &new_ships {
        commands.entity(entity).insert(ShipVisibilitySet::default());
    }
}

/// Compute inter-ship visibility sets. Runs every 20 ticks (~1 second).
/// For each ship, determines which other ships are within visual range.
/// On change (enter/leave), sends VisibilityDirective to affected ship shards via QUIC.
/// Hysteresis: open at RANGE, close at RANGE * 1.5 to prevent connection flapping.
fn compute_visibility(
    tick: Res<ecs::TickCounter>,
    mut ships: Query<(Entity, &ShipId, &Position, &mut ShipVisibilitySet)>,
    vis_range: Res<ShipVisibilityRange>,
    bridge: Res<NetworkBridge>,
    shard_identity: Res<ShardIdentity>,
) {
    if tick.0 % 20 != 0 {
        return;
    }

    let open_range = vis_range.0;
    let close_range = vis_range.0 * 1.5;

    // Collect all ship positions for O(N²) distance checks.
    // For small ship counts (<100) this is fine. For larger counts,
    // a spatial index (e.g., grid hash) would be needed.
    let ship_data: Vec<(Entity, u64, DVec3)> = ships
        .iter()
        .map(|(e, id, pos, _)| (e, id.0, pos.0))
        .collect();

    // Look up peer shard info for building VisibilityDirective entries.
    let peer_info: Vec<(u64, ShardId, String, String)> =
        if let Ok(reg) = bridge.peer_registry.try_read() {
            reg.find_by_type(ShardType::Ship)
                .iter()
                .filter_map(|info| {
                    let ship_id = info.ship_id?;
                    // Only include ships hosted by this system shard.
                    if info.host_shard_id != Some(shard_identity.0) {
                        return None;
                    }
                    Some((
                        ship_id,
                        info.id,
                        info.endpoint.tcp_addr.to_string(),
                        info.endpoint.udp_addr.to_string(),
                    ))
                })
                .collect()
        } else {
            return;
        };

    // SAFETY: We need mutable access to ShipVisibilitySet while iterating.
    // Since we collected ship_data immutably above, we can now do a second pass
    // with mutable access. This requires unsafe or a different query pattern.
    // Use a collected approach: compute new sets, then apply.
    let mut updates: Vec<(Entity, u64, HashSet<u64>)> = Vec::new();

    for &(entity, ship_id, pos) in &ship_data {
        // Get current visibility set for hysteresis.
        let current_set: HashSet<u64> = ships
            .get(entity)
            .map(|(_, _, _, vis)| vis.0.clone())
            .unwrap_or_default();

        let mut new_set = HashSet::new();
        for &(_, other_id, other_pos) in &ship_data {
            if other_id == ship_id {
                continue;
            }
            let dist = (pos - other_pos).length();
            let was_visible = current_set.contains(&other_id);

            // Hysteresis: open at open_range, close at close_range.
            if was_visible {
                if dist <= close_range {
                    new_set.insert(other_id);
                }
                // else: left visibility (dist > close_range)
            } else if dist <= open_range {
                new_set.insert(other_id);
            }
        }

        updates.push((entity, ship_id, new_set));
    }

    // Apply updates and send directives for changed sets.
    for (entity, ship_id, new_set) in updates {
        let Ok((_, _, _, mut vis)) = ships.get_mut(entity) else {
            continue;
        };

        if vis.0 == new_set {
            continue; // No change.
        }

        // Build VisibilityDirective with connection info for each visible ship.
        let visible_entries: Vec<voxeldust_core::shard_message::VisibleShipEntryData> = new_set
            .iter()
            .filter_map(|&other_id| {
                let (_, shard_id, tcp, udp) = peer_info.iter().find(|(id, _, _, _)| *id == other_id)?;
                let dist = ship_data.iter().find(|(_, id, _)| *id == other_id).map(|(_, _, p)| {
                    let my_pos = ship_data.iter().find(|(_, id, _)| *id == ship_id).map(|(_, _, p)| *p).unwrap_or(DVec3::ZERO);
                    (*p - my_pos).length()
                }).unwrap_or(0.0);
                Some(voxeldust_core::shard_message::VisibleShipEntryData {
                    ship_id: other_id,
                    shard_id: *shard_id,
                    tcp_addr: tcp.clone(),
                    udp_addr: udp.clone(),
                    distance: dist,
                })
            })
            .collect();

        let directive = ShardMsg::VisibilityDirective(
            voxeldust_core::shard_message::VisibilityDirectiveData {
                ship_id,
                visible_ships: visible_entries,
            },
        );

        // Send to the ship shard that owns this ship.
        if let Some((_, shard_id, _, _)) = peer_info.iter().find(|(id, _, _, _)| *id == ship_id) {
            if let Ok(reg) = bridge.peer_registry.try_read() {
                if let Some(addr) = reg.quic_addr(*shard_id) {
                    let _ = bridge.quic_send_tx.try_send((*shard_id, addr, directive));
                }
            }
        }

        // Log visibility changes.
        let entered: Vec<u64> = new_set.difference(&vis.0).copied().collect();
        let left: Vec<u64> = vis.0.difference(&new_set).copied().collect();
        if !entered.is_empty() || !left.is_empty() {
            info!(
                ship_id,
                entered = ?entered,
                left = ?left,
                total = new_set.len(),
                "visibility set changed"
            );
        }

        vis.0 = new_set;
    }
}

// ---------------------------------------------------------------------------
// Unified AOI compute (ships + EVA + surface players)
// ---------------------------------------------------------------------------

/// A candidate observable entity gathered once per tick. Pure data — no ECS refs.
#[derive(Clone)]
struct AoiCandidate {
    entity_id: u64,
    kind: EntityKind,
    position: DVec3,
    velocity: DVec3,
    rotation: DQuat,
    bounding_radius: f32,
    name: String,
    /// Authoritative shard for this entity; 0 if no secondary upgrade is beneficial.
    shard_id: u64,
    /// Shard type for secondary routing (0=Planet, 1=System, 2=Ship, 3=Galaxy).
    shard_type: u8,
    health: f32,
    shield: f32,
}

/// Bounding-radius fallback for ships whose `ShipPhysics` is missing a dimensions field.
const DEFAULT_SHIP_RADIUS_M: f32 = 32.0;
/// Approximate player capsule bounding radius (meters).
const PLAYER_BOUNDING_RADIUS_M: f32 = 1.0;

fn ship_bounding_radius(phys: Option<&ShipPhysics>) -> f32 {
    match phys {
        Some(p) => {
            let (w, h, l) = p.0.dimensions;
            // Half the longest axis — a sphere that encloses the ship AABB.
            let longest = w.max(h).max(l);
            if longest > 0.0 {
                (longest as f32) * 0.5
            } else {
                DEFAULT_SHIP_RADIUS_M
            }
        }
        None => DEFAULT_SHIP_RADIUS_M,
    }
}

/// Tier threshold lookup — ships vs players use different ranges.
fn lod_for_distance(distance: f64, is_player: bool, cfg: &VisibilityConfig) -> Option<LodTier> {
    let (full, reduced, coarse) = if is_player {
        (
            cfg.player_full_range,
            cfg.player_reduced_range,
            cfg.player_coarse_range,
        )
    } else {
        (
            cfg.ship_full_range,
            cfg.ship_reduced_range,
            cfg.ship_coarse_range,
        )
    };
    if distance <= full {
        Some(LodTier::Full)
    } else if distance <= reduced {
        Some(LodTier::Reduced)
    } else if distance <= coarse {
        Some(LodTier::Coarse)
    } else {
        None
    }
}

/// Build an AOI snapshot around `observer_position`, skipping the observer itself
/// by matching `self_entity_id` (the observer's entity_id, if any).
fn compute_aoi(
    observer_position: DVec3,
    candidates: &[AoiCandidate],
    self_entity_id: Option<u64>,
    cfg: &VisibilityConfig,
) -> Vec<ObservableEntityData> {
    let mut out = Vec::with_capacity(candidates.len());
    for c in candidates {
        if Some(c.entity_id) == self_entity_id {
            continue;
        }
        let distance = (c.position - observer_position).length();
        let is_player = c.kind.is_player();
        let Some(tier) = lod_for_distance(distance, is_player, cfg) else {
            continue;
        };
        out.push(ObservableEntityData {
            entity_id: c.entity_id,
            kind: c.kind,
            position: c.position,
            rotation: c.rotation,
            velocity: c.velocity,
            bounding_radius: c.bounding_radius,
            lod_tier: tier,
            shard_id: c.shard_id,
            shard_type: c.shard_type,
            is_own: false,
            name: c.name.clone(),
            health: c.health,
            shield: c.shield,
        });
    }
    out
}

/// Collect every observable entity authoritative to this system shard:
/// ships (Position/Velocity/Rotation), EVA players, and surface-player aggregates.
fn collect_aoi_candidates<EvaFilter: bevy_ecs::query::QueryFilter>(
    ships: &Query<(&ShipId, &Position, &Velocity, &Rotation, Option<&ShipPhysics>)>,
    eva: &Query<
        (
            &EvaSession,
            &EvaPlayerName,
            &EvaPosition,
            &EvaVelocity,
            &EvaRotation,
        ),
        EvaFilter,
    >,
    surface: &SurfacePlayerRegistry,
    ship_shards_by_ship_id: &HashMap<u64, ShardId>,
    planet_shards_by_index: &HashMap<u32, ShardId>,
) -> Vec<AoiCandidate> {
    let mut out: Vec<AoiCandidate> = Vec::new();

    for (ship_id, pos, vel, rot, phys) in ships.iter() {
        let shard_id = ship_shards_by_ship_id
            .get(&ship_id.0)
            .map(|s| s.0)
            .unwrap_or(0);
        out.push(AoiCandidate {
            entity_id: ship_id.0,
            kind: EntityKind::Ship,
            position: pos.0,
            velocity: vel.0,
            rotation: rot.0,
            bounding_radius: ship_bounding_radius(phys),
            name: String::new(),
            shard_id,
            shard_type: ShardType::Ship as u8,
            health: 0.0,
            shield: 0.0,
        });
    }

    for (session, name, pos, vel, rot) in eva.iter() {
        out.push(AoiCandidate {
            entity_id: session.0.0,
            kind: EntityKind::EvaPlayer,
            position: pos.0,
            velocity: vel.0,
            rotation: rot.0,
            bounding_radius: PLAYER_BOUNDING_RADIUS_M,
            name: name.0.clone(),
            shard_id: 0, // EVA lives on system shard — no secondary needed.
            shard_type: ShardType::System as u8,
            health: 100.0,
            shield: 100.0,
        });
    }

    for (token, rec) in &surface.players {
        let shard_id = planet_shards_by_index
            .get(&rec.planet_index)
            .copied()
            .unwrap_or(rec.planet_shard)
            .0;
        out.push(AoiCandidate {
            entity_id: *token,
            kind: EntityKind::GroundedPlayer,
            position: rec.position,
            velocity: DVec3::ZERO,
            rotation: rec.rotation,
            bounding_radius: PLAYER_BOUNDING_RADIUS_M,
            name: rec.player_name.clone(),
            shard_id,
            shard_type: ShardType::Planet as u8,
            health: 100.0,
            shield: 100.0,
        });
    }

    out
}

/// Ingest PlanetPlayerDigest messages into SurfacePlayerRegistry.
/// Runs before broadcast_system_entities so AOI includes up-to-date surface players.
fn ingest_planet_player_digest(
    mut digest_reader: MessageReader<PlanetPlayerDigestMsg>,
    mut registry: ResMut<SurfacePlayerRegistry>,
) {
    for ev in digest_reader.read() {
        // Replace all entries for this planet (authoritative snapshot).
        // Gather tokens first to avoid iterator borrow on the map.
        let stale_tokens: Vec<u64> = registry
            .players
            .iter()
            .filter(|(_, r)| r.planet_index == ev.data.planet_index)
            .map(|(k, _)| *k)
            .collect();
        for token in stale_tokens {
            registry.players.remove(&token);
        }
        for entry in &ev.data.entries {
            registry.players.insert(
                entry.session_token.0,
                SurfacePlayerRecord {
                    player_name: entry.player_name.clone(),
                    planet_index: entry.planet_index,
                    planet_shard: ev.data.planet_shard,
                    position: entry.position,
                    rotation: entry.rotation,
                    last_tick: ev.data.tick,
                },
            );
        }
    }
}

/// Bridges an incoming `PlanetPlayerDigest` ShardMsg into an ECS Message
/// (consumed by `ingest_planet_player_digest`).
#[derive(Message)]
struct PlanetPlayerDigestMsg {
    data: PlanetPlayerDigestData,
}

/// Broadcast unified SystemEntitiesUpdate to every ship shard and every planet shard.
/// Produces per-observer AOI snapshots from the single `candidates` pass.
fn broadcast_system_entities(
    ships: Query<(&ShipId, &Position, &Velocity, &Rotation, Option<&ShipPhysics>)>,
    eva: Query<(
        &EvaSession,
        &EvaPlayerName,
        &EvaPosition,
        &EvaVelocity,
        &EvaRotation,
    )>,
    surface: Res<SurfacePlayerRegistry>,
    vis_cfg: Res<VisibilityConfig>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    shard_identity: Res<ShardIdentity>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
) {
    // Map planet_seed → index in this system (stable ordering from SystemConfig).
    let seed_to_index: HashMap<u64, u32> = sys_config
        .0
        .planets
        .iter()
        .enumerate()
        .map(|(i, p)| (p.planet_seed, i as u32))
        .collect();

    // Look up peer shards by type so we can send SystemEntitiesUpdate to each.
    let (ship_shards_by_id, planet_shards_by_index, ship_quic_addrs, planet_quic_addrs) = {
        let Ok(reg) = bridge.peer_registry.try_read() else {
            return;
        };
        let mut ship_by_id: HashMap<u64, ShardId> = HashMap::new();
        let mut ship_addrs: HashMap<ShardId, SocketAddr> = HashMap::new();
        for info in reg.find_by_type(ShardType::Ship) {
            if let Some(sid) = info.ship_id {
                // Only ships hosted by this system shard.
                if info.host_shard_id != Some(shard_identity.0) {
                    continue;
                }
                ship_by_id.insert(sid, info.id);
                ship_addrs.insert(info.id, info.endpoint.quic_addr);
            }
        }
        let mut planet_by_index: HashMap<u32, ShardId> = HashMap::new();
        let mut planet_addrs: HashMap<ShardId, SocketAddr> = HashMap::new();
        for info in reg.find_by_type(ShardType::Planet) {
            if let Some(seed) = info.planet_seed {
                if let Some(&idx) = seed_to_index.get(&seed) {
                    planet_by_index.insert(idx, info.id);
                    planet_addrs.insert(info.id, info.endpoint.quic_addr);
                }
            }
        }
        (ship_by_id, planet_by_index, ship_addrs, planet_addrs)
    };

    let candidates = collect_aoi_candidates(
        &ships,
        &eva,
        &surface,
        &ship_shards_by_id,
        &planet_shards_by_index,
    );

    // Ship observers → the ship shard hosting each ship.
    for (ship_id, pos, _, _, _) in ships.iter() {
        let Some(&shard_id) = ship_shards_by_id.get(&ship_id.0) else {
            continue;
        };
        let Some(&quic_addr) = ship_quic_addrs.get(&shard_id) else {
            continue;
        };
        let aoi = compute_aoi(pos.0, &candidates, Some(ship_id.0), &vis_cfg);
        let upd = ShardMsg::SystemEntitiesUpdate(SystemEntitiesUpdateData {
            target: AoiTarget::Ship(ship_id.0),
            observer_position: pos.0,
            entities: aoi,
            tick: tick.0,
        });
        let _ = bridge.quic_send_tx.try_send((shard_id, quic_addr, upd));
    }

    // Planet observers → each planet shard (observer = planet's system-space position).
    for (&idx, &shard_id) in &planet_shards_by_index {
        let idx_usize = idx as usize;
        if idx_usize >= planet_pos.0.len() {
            continue;
        }
        let Some(&quic_addr) = planet_quic_addrs.get(&shard_id) else {
            continue;
        };
        let observer = planet_pos.0[idx_usize];
        let aoi = compute_aoi(observer, &candidates, None, &vis_cfg);
        let upd = ShardMsg::SystemEntitiesUpdate(SystemEntitiesUpdateData {
            target: AoiTarget::Planet(idx),
            observer_position: observer,
            entities: aoi,
            tick: tick.0,
        });
        let _ = bridge.quic_send_tx.try_send((shard_id, quic_addr, upd));
    }
}

// ---------------------------------------------------------------------------
// Broadcast systems
// ---------------------------------------------------------------------------

/// Broadcast SystemSceneUpdate + ShipPositionUpdate to ship shards via QUIC.
fn broadcast_scene(
    ships: Query<(
        &ShipId,
        &Position,
        &Velocity,
        &Rotation,
        &AngularVelocity,
        Option<&Autopilot>,
        Option<&AutopilotIntercept>,
        Option<&InSoi>,
    )>,
    sys_config: Res<SystemConfig>,
    shard_identity: Res<ShardIdentity>,
    planet_pos: Res<PlanetPositions>,
    celestial_time: Res<ecs::CelestialTime>,
    bridge: Res<NetworkBridge>,
) {
    let ship_shards: Vec<(ShardId, SocketAddr, Option<u64>, Option<ShardId>)> =
        if let Ok(reg) = bridge.peer_registry.try_read() {
            reg.find_by_type(ShardType::Ship)
                .iter()
                .map(|info| {
                    (
                        info.id,
                        info.endpoint.quic_addr,
                        info.ship_id,
                        info.host_shard_id,
                    )
                })
                .collect()
        } else {
            return;
        };

    // Collect known ship IDs.
    let known_ship_ids: HashSet<u64> = ships.iter().map(|(sid, _, _, _, _, _, _, _)| sid.0).collect();

    let hosted_ships: Vec<(ShardId, SocketAddr, Option<u64>)> = ship_shards
        .iter()
        .filter(|(_, _, ship_id, host)| {
            *host == Some(shard_identity.0)
                || ship_id
                    .map(|id| known_ship_ids.contains(&id))
                    .unwrap_or(false)
        })
        .map(|(id, addr, ship_id, _)| (*id, *addr, *ship_id))
        .collect();

    if !hosted_ships.is_empty() {
        // Build body snapshots.
        let mut bodies = Vec::with_capacity(1 + sys_config.0.planets.len());
        bodies.push(CelestialBodySnapshotData {
            body_id: 0,
            position: DVec3::ZERO,
            radius: sys_config.0.star.radius_m,
            color: sys_config.0.star.color,
        });
        for (i, planet) in sys_config.0.planets.iter().enumerate() {
            bodies.push(CelestialBodySnapshotData {
                body_id: (i + 1) as u32,
                position: planet_pos.0[i],
                radius: planet.radius_m,
                color: planet.color,
            });
        }

        let observer_pos = ships
            .iter()
            .next()
            .map(|(_, p, _, _, _, _, _, _)| p.0)
            .unwrap_or(DVec3::new(1e11, 0.0, 0.0));
        let l = compute_lighting(observer_pos, &sys_config.0.star);
        let lighting = LightingInfoData {
            sun_direction: l.sun_direction,
            sun_color: l.sun_color,
            sun_intensity: l.sun_intensity,
            ambient: l.ambient,
        };

        let scene = SystemSceneUpdateData {
            game_time: celestial_time.0,
            bodies,
            ships: vec![],
            lighting,
        };
        let scene_msg = ShardMsg::SystemSceneUpdate(scene);

        for &(shard_id, quic_addr, _) in &hosted_ships {
            let _ = bridge
                .quic_send_tx
                .try_send((shard_id, quic_addr, scene_msg.clone()));
        }

        // Per-ship position updates.
        for (ship_id, pos, vel, rot, ang_vel, autopilot, intercept, in_soi) in &ships {
            let target = hosted_ships
                .iter()
                .find(|(_, _, sid)| *sid == Some(ship_id.0))
                .map(|&(sid, addr, _)| (sid, addr));
            if let Some((sid, addr)) = target {
                let ap_snapshot = autopilot.and_then(|ap| {
                    Some(AutopilotSnapshotData {
                        phase: ap.phase.to_u8(),
                        mode: ap.mode.to_u8(),
                        target_planet_index: (ap.target_body_id.wrapping_sub(1)),
                        thrust_tier: ap.thrust_tier,
                        intercept_pos: intercept.map(|i| i.intercept_pos).unwrap_or(DVec3::ZERO),
                        target_arrival_vel: intercept
                            .map(|i| i.target_arrival_vel)
                            .unwrap_or(DVec3::ZERO),
                        braking_committed: ap.braking_committed,
                        eta_real_seconds: ap.estimated_tof,
                        target_orbit_altitude: ap.target_orbit_altitude,
                    })
                });

                // Authoritative atmosphere + gravity for the ship-shard.
                let atmo = check_atmosphere(pos.0, &sys_config.0.planets, &planet_pos.0);
                let grav = if let Some(soi) = in_soi {
                    let r = pos.0 - planet_pos.0[soi.planet_index];
                    let dist_sq = r.length_squared();
                    if dist_sq > 1.0 {
                        -r.normalize() * sys_config.0.planets[soi.planet_index].gm / dist_sq
                    } else {
                        DVec3::ZERO
                    }
                } else {
                    compute_gravity_acceleration(
                        pos.0,
                        &sys_config.0.star,
                        &sys_config.0.planets,
                        &planet_pos.0,
                        celestial_time.0,
                    )
                };

                let pos_msg = ShardMsg::ShipPositionUpdate(ShipPositionUpdate {
                    ship_id: ship_id.0,
                    position: pos.0,
                    velocity: vel.0,
                    rotation: rot.0,
                    angular_velocity: ang_vel.0,
                    autopilot: ap_snapshot,
                    in_atmosphere: atmo.is_some(),
                    atmosphere_planet_index: atmo.map(|(i, _)| i as i32).unwrap_or(-1),
                    gravity_acceleration: grav,
                    atmosphere_density: atmo.map(|(pi, alt)| {
                        sys_config.0.planets[pi].atmosphere.density_at_altitude(alt)
                    }).unwrap_or(0.0),
                });
                let _ = bridge.quic_send_tx.try_send((sid, addr, pos_msg));
            }
        }
    }

    // ShipNearbyInfo to planet shards is handled by the separate broadcast_ship_nearby system.
}

/// Broadcast ShipNearbyInfo to planet shards for ships in their SOI.
fn broadcast_ship_nearby(
    ships: Query<(&ShipId, &Position, &Velocity, &Rotation, &InSoi)>,
    sys_config: Res<SystemConfig>,
    celestial_time: Res<ecs::CelestialTime>,
    bridge: Res<NetworkBridge>,
) {
    let planet_shards: Vec<(ShardId, SocketAddr, Option<u64>)> =
        if let Ok(reg) = bridge.peer_registry.try_read() {
            reg.find_by_type(ShardType::Planet)
                .iter()
                .map(|info| (info.id, info.endpoint.quic_addr, info.planet_seed))
                .collect()
        } else {
            return;
        };

    let ship_shards: Vec<(ShardId, Option<u64>)> =
        if let Ok(reg) = bridge.peer_registry.try_read() {
            reg.find_by_type(ShardType::Ship)
                .iter()
                .map(|info| (info.id, info.ship_id))
                .collect()
        } else {
            return;
        };

    for (ship_id, pos, vel, rot, in_soi) in &ships {
        let planet_seed = sys_config.0.planets[in_soi.planet_index].planet_seed;
        let planet_shard = planet_shards
            .iter()
            .find(|(_, _, ps)| *ps == Some(planet_seed))
            .map(|&(sid, addr, _)| (sid, addr));

        if let Some((psid, paddr)) = planet_shard {
            let ship_shard_id = ship_shards
                .iter()
                .find(|(_, sid)| *sid == Some(ship_id.0))
                .map(|(id, _)| *id)
                .unwrap_or(ShardId(0));

            let nearby_msg = ShardMsg::ShipNearbyInfo(ShipNearbyInfoData {
                ship_id: ship_id.0,
                ship_shard_id,
                position: pos.0,
                rotation: rot.0,
                velocity: vel.0,
                game_time: celestial_time.0,
            });
            let _ = bridge.quic_send_tx.try_send((psid, paddr, nearby_msg));
        }
    }
}

/// Broadcast WorldState to direct UDP clients (debug mode / non-EVA).
/// Skipped when EVA players exist — they receive WorldState from eva_broadcast
/// which includes player position data. Sending two WorldStates per tick would
/// race: this one has `players: vec![]` which prevents the client from ever
/// setting `synced = true`, locking render_position at origin.
fn broadcast_udp(
    ships: Query<(&ShipId, &Position)>,
    eva_players: Query<(), With<EvaPlayer>>,
    sys_config: Res<SystemConfig>,
    planet_pos: Res<PlanetPositions>,
    celestial_time: Res<ecs::CelestialTime>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
) {
    // When EVA players are connected, eva_broadcast is the single authoritative
    // WorldState source. Do not send a competing empty-players broadcast.
    if !eva_players.is_empty() {
        return;
    }
    let bodies: Vec<CelestialBodyData> = {
        let mut b = Vec::with_capacity(1 + sys_config.0.planets.len());
        b.push(CelestialBodyData {
            body_id: 0,
            position: DVec3::ZERO,
            radius: sys_config.0.star.radius_m,
            color: sys_config.0.star.color,
        });
        for (i, planet) in sys_config.0.planets.iter().enumerate() {
            b.push(CelestialBodyData {
                body_id: (i + 1) as u32,
                position: planet_pos.0[i],
                radius: planet.radius_m,
                color: planet.color,
            });
        }
        b
    };

    let observer_pos = ships
        .iter()
        .next()
        .map(|(_, p)| p.0)
        .unwrap_or(DVec3::new(1e11, 0.0, 0.0));
    let lighting = compute_lighting(observer_pos, &sys_config.0.star);

    let ws = ServerMsg::WorldState(WorldStateData {
        tick: tick.0,
        origin: DVec3::ZERO,
        players: vec![],
        bodies,
        ships: vec![],
        lighting: Some(LightingData {
            sun_direction: lighting.sun_direction,
            sun_color: lighting.sun_color,
            sun_intensity: lighting.sun_intensity,
            ambient: lighting.ambient,
        }),
        game_time: celestial_time.0,
        warp_target_star_index: 0xFFFFFFFF,
        autopilot: None,
        sub_grids: vec![],
        entities: vec![],
    });
    if bridge.broadcast_tx.try_send(ws).is_err() {
        tracing::warn!("WorldState broadcast dropped — channel full");
    }
}

// ---------------------------------------------------------------------------
// Diagnostics systems
// ---------------------------------------------------------------------------

fn tick_counter(mut tick: ResMut<ecs::TickCounter>) {
    tick.0 += 1;
}

fn log_state(
    ships: Query<(&ShipId, &Position, &Velocity)>,
    tick: Res<ecs::TickCounter>,
    physics_time: Res<ecs::PhysicsTime>,
    celestial_time: Res<ecs::CelestialTime>,
) {
    if tick.0 % 100 != 0 || tick.0 == 0 {
        return;
    }

    let ship_count = ships.iter().count();
    info!(
        physics_time = format!("{:.1}s", physics_time.0),
        celestial_time = format!("{:.1}s", celestial_time.0),
        ships = ship_count,
        tick = tick.0,
        "system state"
    );
}

// ---------------------------------------------------------------------------
// App construction
// ---------------------------------------------------------------------------

fn build_app(
    system_seed: u64,
    shard_id: ShardId,
    orchestrator_url: String,
    galaxy_seed: u64,
    star_index: u32,
) -> App {
    let system_params = SystemParams::from_seed(system_seed);

    // Compute initial planet positions at t=0.
    let planet_positions: Vec<DVec3> = system_params
        .planets
        .iter()
        .map(|p| compute_planet_position(p, 0.0))
        .collect();
    let star_gm = system_params.star.gm;
    let planet_velocities: Vec<DVec3> = system_params
        .planets
        .iter()
        .map(|p| compute_planet_velocity(p, star_gm, 0.0))
        .collect();

    let star_position_gu = if galaxy_seed != 0 {
        let galaxy_map = voxeldust_core::galaxy::GalaxyMap::generate(galaxy_seed);
        galaxy_map
            .get_star(star_index)
            .map(|s| s.position)
            .unwrap_or(DVec3::ZERO)
    } else {
        DVec3::ZERO
    };

    let (planet_prov_tx, planet_prov_rx) = mpsc::channel(32);
    let (galaxy_prov_tx, galaxy_prov_rx) = mpsc::channel(4);

    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("failed to build HTTP client");

    info!(
        system_seed,
        planets = system_params.planets.len(),
        galaxy_seed,
        star_index,
        "system params initialized"
    );

    let mut app = App::new();

    // Resources.
    app.insert_resource(SystemConfig(system_params));
    app.insert_resource(ShardIdentity(shard_id));
    app.insert_resource(SystemSeed(system_seed));
    app.insert_resource(PlanetPositions(planet_positions.clone()));
    app.insert_resource(OldPlanetPositions(planet_positions));
    app.insert_resource(PlanetVelocities(planet_velocities));
    app.insert_resource(ecs::CelestialTime::default());
    app.insert_resource(ecs::PhysicsTime::default());
    app.insert_resource(ecs::TickCounter::default());
    app.insert_resource(ecs::ShipEntityIndex::default());
    app.insert_resource(GalaxyContext {
        seed: galaxy_seed,
        star_index,
        star_position_gu,
        provisioned_galaxy: None,
        galaxy_provisioning_in_flight: false,
    });
    app.insert_resource(DepartedShips::default());
    app.insert_resource(ShipVisibilityRange::default());
    app.insert_resource(VisibilityConfig::default());
    app.insert_resource(SurfacePlayerRegistry::default());
    app.insert_resource(EvaPlayerIndex::default());
    app.insert_resource(ProvisionedPlanets::default());
    app.insert_resource(ProvisioningInFlight::default());
    app.insert_resource(PendingHandoffs::default());
    app.insert_resource(PendingWarpArrivals::default());
    app.insert_resource(OrchestratorUrl(orchestrator_url));
    app.insert_resource(HttpClient(http_client));
    app.insert_resource(PlanetProvisionSender(planet_prov_tx));
    app.insert_resource(PlanetProvisionReceiver(planet_prov_rx));
    app.insert_resource(GalaxyProvisionSender(galaxy_prov_tx));
    app.insert_resource(GalaxyProvisionReceiver(galaxy_prov_rx));

    // Messages.
    app.add_message::<ClientConnectedMsg>();
    app.add_message::<ShipControlMsg>();
    app.add_message::<AutopilotMsg>();
    app.add_message::<WarpAutopilotMsg>();
    app.add_message::<HandoffMsg>();
    app.add_message::<HandoffAcceptedMsg>();
    app.add_message::<PlanetProvisionedMsg>();
    app.add_message::<GalaxyProvisionedMsg>();
    app.add_message::<ShipPropsUpdateMsg>();
    app.add_message::<ShipColliderSyncMsg>();
    app.add_message::<PlanetPlayerDigestMsg>();

    // System ordering via SystemSets.
    app.configure_sets(
        Update,
        (
            SystemShardSet::Bridge,
            SystemShardSet::Spawn,
            SystemShardSet::Physics,
            SystemShardSet::Detection,
            SystemShardSet::Broadcast,
            SystemShardSet::Diagnostics,
        )
            .chain(),
    );

    // Bridge: drain async channels → ECS events.
    app.add_systems(
        Update,
        (drain_connects, drain_quic, drain_provisions, eva_input).in_set(SystemShardSet::Bridge),
    );

    // Spawn/process: convert events into entities and component updates.
    // Spawn/process: convert events into entities and component updates.
    // process_handoffs MUST run before process_connects so that EVA entities
    // are registered in EvaPlayerIndex before TCP connections are processed.
    app.add_systems(
        Update,
        (
            (
                process_handoffs,
                process_ship_control,
                process_ship_props_update,
                process_autopilot_commands,
                process_warp_commands,
                process_handoff_accepted,
                process_planet_provisions,
                process_galaxy_provisions,
                discover_ships,
                process_ship_collider_sync,
            ),
            process_connects, // runs after process_handoffs due to tuple nesting
        )
            .chain()
            .in_set(SystemShardSet::Spawn),
    );

    // apply_deferred between Spawn and Physics so newly spawned entities are queryable.
    app.add_systems(
        Update,
        bevy_ecs::schedule::ApplyDeferred
            .after(SystemShardSet::Spawn)
            .before(SystemShardSet::Physics),
    );

    // Physics: orbits → landed → co-movement → autopilot → thrust → integrate → ground contact.
    app.add_systems(
        Update,
        (
            update_orbits,
            update_landed_positions,
            soi_co_movement,
            soi_diagnostic,
            autopilot_resolve,
            autopilot_guidance,
            autopilot_phase_transition,
            // rotation_control is integrated into thrust_application
            thrust_application,
            warp_arrival_deceleration,
            physics_integrate,
            ground_contact,
            thermal_update,
            eva_physics,
        )
            .chain()
            .in_set(SystemShardSet::Physics),
    );

    // Detection: SOI → planet provisioning → warp boundary → EVA transitions → visibility.
    // ingest_planet_player_digest must run before compute_visibility/broadcast_system_entities
    // so AOI includes up-to-date surface-player aggregates.
    app.add_systems(
        Update,
        (soi_detection, planet_provisioning, warp_boundary, warp_velocity_cap,
         eva_boarding_detection, eva_soi_detection,
         init_visibility_set, compute_visibility,
         ingest_planet_player_digest)
            .chain()
            .in_set(SystemShardSet::Detection),
    );

    // apply_deferred between Detection and Broadcast so despawned entities are applied.
    app.add_systems(
        Update,
        bevy_ecs::schedule::ApplyDeferred
            .after(SystemShardSet::Detection)
            .before(SystemShardSet::Broadcast),
    );

    // Broadcast: send state to clients and ship shards.
    // broadcast_system_entities must run after the candidate producers (physics).
    app.add_systems(
        Update,
        (
            broadcast_scene,
            broadcast_ship_nearby,
            broadcast_udp,
            eva_broadcast,
            broadcast_system_entities,
        )
            .in_set(SystemShardSet::Broadcast),
    );

    // Diagnostics: tick counter + periodic logging.
    app.add_systems(
        Update,
        (tick_counter, log_state)
            .chain()
            .in_set(SystemShardSet::Diagnostics),
    );

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

    let shard_id = ShardId(args.shard_id);
    let bind = "0.0.0.0";
    let config = ShardHarnessConfig {
        shard_id,
        shard_type: ShardType::System,
        tcp_addr: format!("{bind}:{}", args.tcp_port).parse().unwrap(),
        udp_addr: format!("{bind}:{}", args.udp_port).parse().unwrap(),
        quic_addr: format!("{bind}:{}", args.quic_port).parse().unwrap(),
        orchestrator_url: args.orchestrator.clone(),
        orchestrator_heartbeat_addr: args.orchestrator_heartbeat,
        healthz_addr: format!("{bind}:{}", args.healthz_port).parse().unwrap(),
        planet_seed: None,
        system_seed: Some(args.seed),
        ship_id: None,
        galaxy_seed: None,
        host_shard_id: None,
        advertise_host: args.advertise_host,
    };

    info!(
        shard_id = args.shard_id,
        system_seed = args.seed,
        galaxy_seed = args.galaxy_seed,
        star_index = args.star_index,
        "system shard starting"
    );

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let harness = ShardHarness::new(config);
        let app = build_app(
            args.seed,
            shard_id,
            args.orchestrator,
            args.galaxy_seed,
            args.star_index,
        );

        info!("system shard ECS app built, starting harness");
        harness.run_ecs(app).await;
    });
}
