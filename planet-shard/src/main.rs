use std::collections::HashMap;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use clap::Parser;
use glam::{DQuat, DVec3, Vec3};
use rapier3d::prelude::*;
use tracing::{info, warn};

use voxeldust_core::client_message::{
    CelestialBodyData, EntityKind, JoinResponseData, LightingData, LodTier, ObservableEntityData,
    PlayerSnapshotData, ServerMsg, ShipRenderData, WorldStateData,
};
use voxeldust_core::ecs;
use voxeldust_core::handoff;
use voxeldust_core::shard_message::{
    AoiTarget, PlanetPlayerDigestData, PlanetPlayerDigestEntry, ShardMsg, ShipNearbyInfoData,
};
use voxeldust_core::shard_types::{SessionToken, ShardId, ShardType};
use voxeldust_core::system::{compute_lighting, compute_planet_position, SystemParams};
use voxeldust_shard_common::client_listener;
use voxeldust_shard_common::harness::{NetworkBridge, ShardHarness, ShardHarnessConfig};

use voxeldust_core::character::{
    self, build_character, move_one_character, CharacterBuildSpec, CharacterCapsule,
    CharacterCollisionEvent, CharacterController, CharacterMoveInput, CharacterVelocity,
    DesiredMovement, IsCharacter, LandedEvent, LocalUp, LocomotionState, MovementStats,
    PlatformDelta, PlatformSnapSuppressed, RapierWorld,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "planet-shard", about = "Voxeldust planet shard — surface physics")]
struct Args {
    #[arg(long)]
    shard_id: u64,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    system_seed: Option<u64>,
    #[arg(long, default_value = "0")]
    planet_index: u32,
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
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const G: f64 = 6.674e-11;

// Planet walking constants moved into the shared
// `voxeldust_core::character::MovementStats` defaults. Legacy
// `WALK_SPEED`/`JUMP_IMPULSE` values (4.0 / 5.0) were preserved as the
// new tunable defaults — speeds are unchanged; the physics is now
// kinematic-character-controller driven.

// ---------------------------------------------------------------------------
// Components (planet-shard-specific, on player entities)
// ---------------------------------------------------------------------------

/// Marker for a player entity on this planet.
#[derive(Component)]
struct PlanetPlayer;

// `PlayerBody` (RigidBodyHandle newtype) was removed in the KCC
// migration. The body handle now lives inside `CharacterController`
// from `voxeldust_core::character`.

/// Planet-local position (vector from planet center).
#[derive(Component)]
struct PlanetPosition(DVec3);

/// Orthonormal frame tangent to the sphere at the player's position.
#[derive(Component)]
struct TangentFrame {
    up: DVec3,
    north: DVec3,
    east: DVec3,
}

impl TangentFrame {
    fn from_up(up: DVec3) -> Self {
        let up = up.normalize();
        let pole = DVec3::Y;
        let east_raw = pole.cross(up);
        let east = if east_raw.length_squared() > 1e-10 {
            east_raw.normalize()
        } else {
            DVec3::Z.cross(up).normalize()
        };
        let north = up.cross(east).normalize();
        Self { up, north, east }
    }
}

/// Rapier flat-space position at last re-center.
#[derive(Component)]
struct RapierOrigin(DVec3);

/// Player yaw on the surface.
#[derive(Component)]
struct PlayerYaw(f32);

/// Player session token.
#[derive(Component)]
struct SessionId(SessionToken);

/// Player name.
#[derive(Component)]
struct Name(String);

/// Input action state for edge detection.
#[derive(Component)]
struct ActionState {
    current: u8,
    previous: u8,
}

/// Marks a player as having a pending handoff.
#[derive(Component)]
struct HandoffPending;

// ---------------------------------------------------------------------------
// Components for nearby ships
// ---------------------------------------------------------------------------

/// Marker for a nearby ship entity.
#[derive(Component)]
struct NearbyShip;

/// Ship identifier.
#[derive(Component)]
struct NearbyShipId(u64);

/// Ship's shard for handoff routing.
#[derive(Component)]
struct NearbyShipShard(ShardId);

/// Ship position in planet-local coordinates.
#[derive(Component)]
struct ShipPosition(DVec3);

/// Ship rotation.
#[derive(Component)]
struct ShipRotation(DQuat);

/// Rapier rigid body handle for the ship's exterior collider (KinematicPositionBased).
/// Created when ShipColliderSync is received from the ship shard.
#[derive(Component)]
struct ShipColliderBody(rapier3d::dynamics::RigidBodyHandle);

/// Rapier collider handle for the ship's compound hull shape.
#[derive(Component)]
struct ShipColliderAttached(rapier3d::geometry::ColliderHandle);

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Rapier 3D physics context.
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
    /// Set true after `refresh_query_pipeline`; cleared after
    /// `physics_pipeline.step()` invalidates the BVH snapshot.
    query_pipeline_fresh: bool,
}

impl RapierWorld for RapierContext {
    fn bodies(&self) -> &RigidBodySet { &self.rigid_body_set }
    fn colliders(&self) -> &ColliderSet { &self.collider_set }
    fn queries(&self) -> &QueryPipeline { &self.query_pipeline }
    fn bodies_mut(&mut self) -> &mut RigidBodySet { &mut self.rigid_body_set }
    fn refresh_query_pipeline(&mut self) {
        if !self.query_pipeline_fresh {
            self.query_pipeline.update(&self.collider_set);
            self.query_pipeline_fresh = true;
        }
    }
}

/// Planet physics dt. Shared between KCC + Rapier step so they integrate
/// against the same clock.
#[derive(Resource, Clone, Copy)]
struct PlanetIntegrationDt(pub f32);

/// Planet configuration (immutable after init).
#[derive(Resource)]
#[allow(dead_code)]
struct PlanetConfig {
    shard_id: ShardId,
    planet_seed: u64,
    planet_radius: f64,
    planet_mass: f64,
    surface_gravity: f64,
    system_seed: Option<u64>,
    planet_index: u32,
}

/// System parameters for orbital computation.
#[derive(Resource)]
struct SystemParamsRes(Option<SystemParams>);

/// Planet position in system-space (updated each tick from Keplerian orbit).
#[derive(Resource)]
struct PlanetPositionInSystem(DVec3);

/// Cached system-space positions for all planets (avoids recomputing Kepler per broadcast).
#[derive(Resource, Default)]
struct CachedAllPlanetPositions(Vec<DVec3>);

/// Celestial time derived from epoch (matches system shard).
#[derive(Resource, Default)]
struct CelestialTimeRes(f64);

/// Physics simulation time.
#[derive(Resource, Default)]
struct PhysicsTimeRes(f64);

/// Universe epoch for deterministic celestial time.
#[derive(Resource)]
struct UniverseEpoch(Arc<std::sync::atomic::AtomicU64>);

/// Entity index: session_token → Entity for O(1) lookup.
#[derive(Resource, Default)]
struct PlayerEntityIndex(HashMap<SessionToken, Entity>);

/// Entity index: ship_id → Entity for nearby ships.
#[derive(Resource, Default)]
struct ShipEntityIndex(HashMap<u64, Entity>);

/// Handoff spawn info stored by process_handoffs, consumed by process_connects.
/// Avoids deferred-Commands visibility issues (Commands not materialized in same tick).
struct HandoffSpawnInfo {
    surface_pos: DVec3,
}

/// Pending handoffs keyed by player name. process_handoffs inserts, process_connects consumes.
#[derive(Resource, Default)]
struct PendingHandoffs(HashMap<String, HandoffSpawnInfo>);

/// Latest AOI snapshot received from the system shard.
/// System shard sends entities in system-space coordinates with `observer_position`
/// set to this planet's system-space position. We transform to planet-local by
/// subtracting the planet's system-space position at broadcast time.
#[derive(Resource, Default)]
struct ExternalEntities {
    entities: Vec<ObservableEntityData>,
    observer_position: DVec3,
    tick: u64,
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// A new TCP client connected.
#[derive(Message)]
struct ClientConnectedMsg {
    session_token: SessionToken,
    player_name: String,
    tcp_write: Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>,
}

/// Player input from UDP.
#[derive(Message)]
struct PlayerInputMsg {
    session: SessionToken,
    input: voxeldust_core::client_message::PlayerInputData,
}

/// Player handoff from system shard.
#[derive(Message)]
struct InboundHandoffMsg {
    handoff: handoff::PlayerHandoff,
    relay_shard: ShardId,
}

/// Ship nearby info from system shard.
#[derive(Message)]
struct ShipNearbyMsg(ShipNearbyInfoData);

/// HandoffAccepted from ship shard (player re-entering ship).
#[derive(Message)]
struct HandoffAcceptedMsg {
    session: SessionToken,
    target_shard: ShardId,
}

/// Ship collider shapes from ship shard (for physical collision).
#[derive(Message)]
struct ShipColliderSyncMsg(voxeldust_core::shard_message::ShipColliderSyncData);


// ---------------------------------------------------------------------------
// System Sets
// ---------------------------------------------------------------------------

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum PlanetSet {
    Bridge,
    Spawn,
    Input,
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
        info!(player = %conn.player_name, session = conn.session_token.0, "player connected to planet");
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
    player_index: Res<PlayerEntityIndex>,
) {
    for _ in 0..64 {
        let (_src, input) = match bridge.input_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        // Route to first player (same as original — multi-player session routing
        // needs UDP addr→session mapping, which will come with proper multi-player).
        if let Some((&session, _)) = player_index.0.iter().next() {
            events.write(PlayerInputMsg { session, input });
        }
    }
}

fn drain_quic(
    mut bridge: ResMut<NetworkBridge>,
    mut handoff_events: MessageWriter<InboundHandoffMsg>,
    mut ship_nearby_events: MessageWriter<ShipNearbyMsg>,
    mut accepted_events: MessageWriter<HandoffAcceptedMsg>,
    mut collider_sync_events: MessageWriter<ShipColliderSyncMsg>,
    mut external_entities: ResMut<ExternalEntities>,
    mut celestial_time: ResMut<CelestialTimeRes>,
    mut planet_pos: ResMut<PlanetPositionInSystem>,
    sys_params: Res<SystemParamsRes>,
    config: Res<PlanetConfig>,
) {
    for _ in 0..32 {
        let queued = match bridge.quic_msg_rx.try_recv() {
            Ok(q) => q,
            Err(_) => break,
        };
        match queued.msg {
            ShardMsg::PlayerHandoff(h) => {
                // Sync celestial time from system shard authority.
                if h.game_time > celestial_time.0 {
                    celestial_time.0 = h.game_time;
                    update_planet_position(&sys_params.0, config.planet_index, celestial_time.0, &mut planet_pos);
                }
                handoff_events.write(InboundHandoffMsg {
                    handoff: h,
                    relay_shard: queued.source_shard_id,
                });
            }
            ShardMsg::ShipNearbyInfo(info) => {
                if info.game_time > celestial_time.0 {
                    celestial_time.0 = info.game_time;
                    update_planet_position(&sys_params.0, config.planet_index, celestial_time.0, &mut planet_pos);
                }
                ship_nearby_events.write(ShipNearbyMsg(info));
            }
            ShardMsg::SystemEntitiesUpdate(data) => {
                // Accept only updates addressed to this planet.
                if let AoiTarget::Planet(idx) = data.target {
                    if idx == config.planet_index && data.tick >= external_entities.tick {
                        external_entities.entities = data.entities;
                        external_entities.observer_position = data.observer_position;
                        external_entities.tick = data.tick;
                    }
                }
            }
            ShardMsg::HandoffAccepted(accepted) => {
                accepted_events.write(HandoffAcceptedMsg {
                    session: accepted.session_token,
                    target_shard: accepted.target_shard,
                });
            }
            ShardMsg::ShipColliderSync(data) => {
                collider_sync_events.write(ShipColliderSyncMsg(data));
            }
            _ => {}
        }
    }
}

fn update_planet_position(
    sys_params: &Option<SystemParams>,
    planet_index: u32,
    celestial_time: f64,
    planet_pos: &mut PlanetPositionInSystem,
) {
    if let Some(sys) = sys_params {
        if let Some(planet) = sys.planets.get(planet_index as usize) {
            planet_pos.0 = compute_planet_position(planet, celestial_time);
        }
    }
}

/// Refresh all-planet position cache once per tick (after physics_step updates celestial_time).
/// Separated from update_planet_position to avoid redundant Kepler solves in drain_quic.
fn refresh_planet_position_cache(
    sys_params: Res<SystemParamsRes>,
    celestial_time: Res<CelestialTimeRes>,
    mut cached: ResMut<CachedAllPlanetPositions>,
) {
    if let Some(ref sys) = sys_params.0 {
        cached.0.clear();
        cached.0.reserve(sys.planets.len());
        for planet in &sys.planets {
            cached.0.push(compute_planet_position(planet, celestial_time.0));
        }
    }
}

// ---------------------------------------------------------------------------
// Spawn / process events
// ---------------------------------------------------------------------------

fn process_connects(
    mut commands: Commands,
    mut events: MessageReader<ClientConnectedMsg>,
    mut rapier: ResMut<RapierContext>,
    config: Res<PlanetConfig>,
    planet_pos: Res<PlanetPositionInSystem>,
    celestial_time: Res<CelestialTimeRes>,
    mut player_index: ResMut<PlayerEntityIndex>,
    mut pending: ResMut<PendingHandoffs>,
    bridge: Res<NetworkBridge>,
) {
    for event in events.read() {
        let token = event.session_token;
        let tcp_write = event.tcp_write.clone();
        let sys_seed = config.system_seed.unwrap_or(0);
        let game_time = celestial_time.0;
        let planet_pos_val = planet_pos.0;

        // Check if a handoff was received for this player (by name).
        let spawn_pos = if let Some(handoff_info) = pending.0.remove(&event.player_name) {
            info!(player = %event.player_name, "spawning from handoff data");
            let entity = spawn_player(
                &mut commands,
                &mut rapier,
                config.planet_radius,
                token,
                event.player_name.clone(),
                handoff_info.surface_pos,
            );
            player_index.0.insert(token, entity);
            handoff_info.surface_pos
        } else {
            // New player — spawn at default position.
            warn!(player = %event.player_name, "no handoff data — spawning at default");
            let default_pos = DVec3::new(0.0, config.planet_radius + 2.0, 0.0);
            let entity = spawn_player(
                &mut commands,
                &mut rapier,
                config.planet_radius,
                token,
                event.player_name.clone(),
                default_pos,
            );
            player_index.0.insert(token, entity);
            default_pos
        };

        info!(player = %event.player_name, session = token.0, "player spawned on planet");

        // Look up the host system shard's endpoints (for always-on scene secondary).
        let system_preconnect = if let Ok(reg) = bridge.peer_registry.try_read() {
            reg.find_by_type(ShardType::System).first().map(|info| {
                handoff::ShardPreConnect {
                    shard_type: 1, // System
                    tcp_addr: info.endpoint.tcp_addr.to_string(),
                    udp_addr: info.endpoint.udp_addr.to_string(),
                    seed: sys_seed,
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
                seed: 0,
                planet_radius: 0,
                player_id: token.0,
                spawn_position: spawn_pos,
                spawn_rotation: DQuat::IDENTITY,
                spawn_forward: DVec3::NEG_Z,
                session_token: token,
                shard_type: 0, // Planet
                galaxy_seed: 0,
                system_seed: sys_seed,
                game_time,
                reference_position: planet_pos_val,
                reference_rotation: DQuat::IDENTITY,
            });
            let mut writer = tcp_write.lock().await;
            let _ = client_listener::send_tcp_msg(&mut *writer, &jr).await;
            // Follow immediately with a ShardPreConnect for the host system shard.
            // System and Galaxy secondaries are always-on and not counted against
            // the client's secondary cap.
            if let Some(pc) = system_preconnect {
                let _ = client_listener::send_tcp_msg(
                    &mut *writer,
                    &ServerMsg::ShardPreConnect(pc),
                )
                .await;
            }
        });
    }
}

fn spawn_player(
    commands: &mut Commands,
    rapier: &mut RapierContext,
    planet_radius: f64,
    session_token: SessionToken,
    name: String,
    planet_local_pos: DVec3,
) -> Entity {
    let radial = planet_local_pos.normalize();
    let height = (planet_local_pos.length() - planet_radius).max(0.5);
    let frame = TangentFrame::from_up(radial);

    // Build the KCC-backed kinematic body at the re-centered origin.
    // The planet keeps the same "character lives at (0, height, 0) in
    // flat Rapier space + re-center each tick" invariant — no change
    // to that pattern, only the body type.
    let stats = MovementStats::default();
    let capsule = CharacterCapsule::default();
    let ctrl = build_character(
        &mut rapier.rigid_body_set,
        &mut rapier.collider_set,
        CharacterBuildSpec {
            position: vector![0.0, height as f32, 0.0],
            capsule,
            stats,
            up_axis: rapier3d::na::Vector3::y_axis(),
        },
    );
    // Apply planet-specific collision groups so players don't get
    // filtered out by the existing planet-shard raycast filters.
    if let Some(col) = rapier.collider_set.get_mut(ctrl.collider) {
        col.set_collision_groups(InteractionGroups::new(
            Group::GROUP_1,
            Group::GROUP_1 | Group::GROUP_2,
        ));
    }

    let identity_bundle = (
        PlanetPlayer,
        PlanetPosition(planet_local_pos),
        frame,
        RapierOrigin(DVec3::new(0.0, height, 0.0)),
        PlayerYaw(0.0),
        SessionId(session_token),
        Name(name),
        ActionState {
            current: 0,
            previous: 0,
        },
    );
    let character_bundle = (
        IsCharacter,
        ctrl,
        capsule,
        stats,
        CharacterVelocity::zero(),
        LocomotionState::Airborne,
        LocalUp(radial),
        DesiredMovement::default(),
        PlatformDelta::default(),
        PlatformSnapSuppressed::default(),
    );

    commands.spawn((identity_bundle, character_bundle)).id()
}

fn process_handoffs(
    mut events: MessageReader<InboundHandoffMsg>,
    config: Res<PlanetConfig>,
    sys_params: Res<SystemParamsRes>,
    planet_pos: Res<PlanetPositionInSystem>,
    mut pending: ResMut<PendingHandoffs>,
    bridge: Res<NetworkBridge>,
) {
    for event in events.read() {
        let h = &event.handoff;

        // Compute planet position at handoff time for accurate system→planet-local conversion.
        let planet_pos_at_handoff = sys_params
            .0
            .as_ref()
            .and_then(|sys| sys.planets.get(config.planet_index as usize))
            .map(|p| compute_planet_position(p, h.game_time))
            .unwrap_or(planet_pos.0);
        let surface_pos = h.position - planet_pos_at_handoff;

        info!(
            player = %h.player_name,
            session = h.session_token.0,
            surface = format!("({:.1},{:.1},{:.1})", surface_pos.x, surface_pos.y, surface_pos.z),
            height = format!("{:.1}", surface_pos.length() - config.planet_radius),
            "handoff received"
        );

        // Store handoff data for process_connects to consume when the
        // player's TCP connection arrives. We don't spawn here because
        // deferred Commands aren't visible to process_connects in the
        // same tick (bevy_ecs ApplyDeferred timing).
        pending.0.insert(h.player_name.clone(), HandoffSpawnInfo {
            surface_pos,
        });

        // Send HandoffAccepted back to the relay system shard.
        let accepted = ShardMsg::HandoffAccepted(handoff::HandoffAccepted {
            session_token: h.session_token,
            target_shard: config.shard_id,
        });
        let relay_shard = event.relay_shard;
        if let Ok(reg) = bridge.peer_registry.try_read() {
            if let Some(addr) = reg.quic_addr(relay_shard) {
                let _ = bridge
                    .quic_send_tx
                    .try_send((relay_shard, addr, accepted));
                info!(
                    target = relay_shard.0,
                    "sent HandoffAccepted to relay system shard"
                );
            } else {
                warn!(
                    target = relay_shard.0,
                    "relay system shard not in peer registry"
                );
            }
        }
    }
}

fn process_ship_nearby(
    mut commands: Commands,
    mut events: MessageReader<ShipNearbyMsg>,
    sys_params: Res<SystemParamsRes>,
    config: Res<PlanetConfig>,
    mut ship_index: ResMut<ShipEntityIndex>,
    mut existing_ships: Query<(&NearbyShipId, &mut ShipPosition, &mut ShipRotation)>,
    bridge: Res<NetworkBridge>,
) {
    for event in events.read() {
        let info = &event.0;

        // Convert to planet-local using planet position at the ship's celestial time.
        let planet_pos_at_ship_time = sys_params
            .0
            .as_ref()
            .and_then(|sys| sys.planets.get(config.planet_index as usize))
            .map(|p| compute_planet_position(p, info.game_time))
            .unwrap_or(DVec3::ZERO);
        let planet_local_pos = info.position - planet_pos_at_ship_time;

        if let Some(&entity) = ship_index.0.get(&info.ship_id) {
            // Update existing ship entity.
            if let Ok((_, mut pos, mut rot)) = existing_ships.get_mut(entity) {
                pos.0 = planet_local_pos;
                rot.0 = info.rotation;
            }
        } else {
            // Spawn new nearby ship entity.
            info!(
                ship_id = info.ship_id,
                pos = format!(
                    "({:.0},{:.0},{:.0})",
                    info.position.x, info.position.y, info.position.z
                ),
                "new ship near planet"
            );
            let entity = commands
                .spawn((
                    NearbyShip,
                    NearbyShipId(info.ship_id),
                    NearbyShipShard(info.ship_shard_id),
                    ShipPosition(planet_local_pos),
                    ShipRotation(info.rotation),
                ))
                .id();
            ship_index.0.insert(info.ship_id, entity);

            // Send ShardPreConnect to all connected planet clients so they
            // open observer connections to this ship shard and receive chunk data.
            let ship_shard_id = info.ship_shard_id;
            let ship_id = info.ship_id;
            if let Ok(reg) = bridge.peer_registry.try_read() {
                if let Some(peer) = reg.get(ship_shard_id) {
                    let pc = ServerMsg::ShardPreConnect(handoff::ShardPreConnect {
                        shard_type: 2, // Ship
                        tcp_addr: peer.endpoint.tcp_addr.to_string(),
                        udp_addr: peer.endpoint.udp_addr.to_string(),
                        seed: ship_id,
                        planet_index: 0,
                        reference_position: DVec3::ZERO,
                        reference_rotation: DQuat::IDENTITY,
                        shard_id: ship_shard_id.0,
                    });
                    let cr = bridge.client_registry.clone();
                    tokio::spawn(async move {
                        if let Ok(reg) = cr.try_read() {
                            for addr in reg.udp_addrs() {
                                if let Some(session) = reg.session_for_udp(addr) {
                                    let _ = reg.send_tcp(session, &pc).await;
                                }
                            }
                        }
                    });
                    tracing::info!(ship_id, shard = ship_shard_id.0, "sent ShardPreConnect for ship to planet clients");
                }
            }
        }
    }
}

/// Build or update Rapier compound colliders for nearby ships from ShipColliderSync.
/// Uses a KinematicPositionBased rigid body so we can update position each tick.
fn process_ship_colliders(
    mut events: MessageReader<ShipColliderSyncMsg>,
    ship_index: Res<ShipEntityIndex>,
    mut rapier: ResMut<RapierContext>,
    mut commands: Commands,
    ships: Query<(
        &ShipPosition,
        &ShipRotation,
        Option<&ShipColliderBody>,
        Option<&ShipColliderAttached>,
    ), With<NearbyShip>>,
) {
    use rapier3d::prelude::*;

    for event in events.read() {
        let data = &event.0;
        let Some(&entity) = ship_index.0.get(&data.ship_id) else {
            warn!(ship_id = data.ship_id, "ShipColliderSync for unknown ship — ignoring");
            continue;
        };

        let Ok((pos, rot, existing_body, existing_collider)) = ships.get(entity) else {
            continue;
        };

        // Remove existing body (which also removes its attached colliders).
        if let Some(body_comp) = existing_body {
            let ctx = &mut *rapier;
            ctx.rigid_body_set.remove(
                body_comp.0,
                &mut ctx.island_manager,
                &mut ctx.collider_set,
                &mut ctx.impulse_joint_set,
                &mut ctx.multibody_joint_set,
                true,
            );
        }

        // Build compound shape from all chunks' collider shapes.
        let mut shapes: Vec<(Isometry<f32>, SharedShape)> = Vec::new();
        for chunk in &data.chunks {
            for &(center, half_extents) in &chunk.shapes {
                let iso = Isometry::translation(center.x, center.y, center.z);
                let shape = SharedShape::cuboid(half_extents.x, half_extents.y, half_extents.z);
                shapes.push((iso, shape));
            }
        }

        if shapes.is_empty() {
            // No solid blocks — remove collider components.
            commands.entity(entity).remove::<ShipColliderBody>();
            commands.entity(entity).remove::<ShipColliderAttached>();
            continue;
        }

        // Convert ship position to Rapier coordinates.
        // Planet shard positions are planet-local (DVec3); Rapier uses f32.
        // For ships near the surface, planet-local coords are small enough for f32.
        let ship_pos_f32 = glam::Vec3::new(
            pos.0.x as f32, pos.0.y as f32, pos.0.z as f32,
        );
        let ship_rot_f32 = glam::Quat::from_xyzw(
            rot.0.x as f32, rot.0.y as f32, rot.0.z as f32, rot.0.w as f32,
        );

        // Create KinematicPositionBased body (position updated each tick).
        let rb = RigidBodyBuilder::kinematic_position_based()
            .translation(vector![ship_pos_f32.x, ship_pos_f32.y, ship_pos_f32.z])
            .build();
        let ctx = &mut *rapier;
        let body_handle = ctx.rigid_body_set.insert(rb);

        // Attach compound collider.
        let compound = ColliderBuilder::compound(shapes).build();
        let collider_handle = ctx.collider_set.insert_with_parent(
            compound,
            body_handle,
            &mut ctx.rigid_body_set,
        );

        commands.entity(entity).insert(ShipColliderBody(body_handle));
        commands.entity(entity).insert(ShipColliderAttached(collider_handle));

        info!(
            ship_id = data.ship_id,
            chunks = data.chunks.len(),
            "built ship exterior compound collider"
        );
    }
}

/// Update Rapier body positions for ship exterior colliders when ShipNearbyInfo updates.
fn update_ship_collider_positions(
    ships: Query<(&ShipPosition, &ShipRotation, &ShipColliderBody), With<NearbyShip>>,
    mut rapier: ResMut<RapierContext>,
) {
    for (pos, rot, body_comp) in &ships {
        if let Some(body) = rapier.rigid_body_set.get_mut(body_comp.0) {
            let p = glam::Vec3::new(pos.0.x as f32, pos.0.y as f32, pos.0.z as f32);
            let r = glam::Quat::from_xyzw(
                rot.0.x as f32, rot.0.y as f32, rot.0.z as f32, rot.0.w as f32,
            );
            body.set_next_kinematic_position(rapier3d::na::Isometry3::from_parts(
                rapier3d::na::Translation3::new(p.x, p.y, p.z),
                rapier3d::na::UnitQuaternion::new_normalize(
                    rapier3d::na::Quaternion::new(r.w, r.x, r.y, r.z),
                ),
            ));
        }
    }
}

fn process_handoff_accepted(
    mut commands: Commands,
    mut events: MessageReader<HandoffAcceptedMsg>,
    mut rapier: ResMut<RapierContext>,
    bridge: Res<NetworkBridge>,
    player_index: Res<PlayerEntityIndex>,
    players: Query<&CharacterController, With<PlanetPlayer>>,
) {
    for event in events.read() {
        let session = event.session;
        let target_shard = event.target_shard;
        info!(
            session = session.0,
            target = target_shard.0,
            "received HandoffAccepted for ship re-entry"
        );

        // Send ShardRedirect to client.
        if let Ok(reg) = bridge.peer_registry.try_read() {
            if let Some(peer_info) = reg.get(target_shard) {
                let redirect = ServerMsg::ShardRedirect(handoff::ShardRedirect {
                    session_token: session,
                    target_tcp_addr: peer_info.endpoint.tcp_addr.to_string(),
                    target_udp_addr: peer_info.endpoint.udp_addr.to_string(),
                    shard_id: target_shard,
                    target_shard_type: peer_info.shard_type as u8,
                });
                let cr = bridge.client_registry.clone();
                tokio::spawn(async move {
                    if let Ok(reg) = cr.try_read() {
                        if let Err(e) = reg.send_tcp(session, &redirect).await {
                            tracing::warn!(%e, "failed to send ShardRedirect for re-entry");
                        }
                    }
                    if let Ok(mut reg) = cr.try_write() {
                        reg.unregister(&session);
                    }
                });
            }
        }

        // Remove player entity and KCC body.
        if let Some(&entity) = player_index.0.get(&session) {
            if let Ok(ctrl) = players.get(entity) {
                let ctx = &mut *rapier;
                ctx.rigid_body_set.remove(
                    ctrl.body,
                    &mut ctx.island_manager,
                    &mut ctx.collider_set,
                    &mut ctx.impulse_joint_set,
                    &mut ctx.multibody_joint_set,
                    true,
                );
            }
            commands.entity(entity).despawn();
            info!(session = session.0, "player removed from planet (re-entry)");
        }
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// Translate client input into `DesiredMovement`; KCC system in Physics
/// set consumes it. Same pattern as ship-shard.
fn process_input(
    mut events: MessageReader<PlayerInputMsg>,
    player_index: Res<PlayerEntityIndex>,
    mut players: Query<
        (&mut PlayerYaw, &mut ActionState, &mut DesiredMovement),
        With<PlanetPlayer>,
    >,
) {
    use voxeldust_core::client_message::input_action_bits as bits;
    for event in events.read() {
        let entity = match player_index.0.get(&event.session) {
            Some(&e) => e,
            None => continue,
        };
        let Ok((mut yaw, mut actions, mut desired)) = players.get_mut(entity) else {
            continue;
        };

        actions.previous = actions.current;
        actions.current = event.input.action;
        yaw.0 = event.input.look_yaw;

        // Planet-local horizontal input: [strafe, forward] — same
        // semantics as ship-shard.
        desired.horizontal = glam::Vec2::new(event.input.movement[0], event.input.movement[2]);
        // Jump is edge-triggered; OR so packets arriving same-tick don't lose it.
        desired.jump |= event.input.jump;
        desired.sprint = (event.input.actions_bits & bits::SPRINT) != 0;
        desired.crouch = (event.input.actions_bits & bits::CROUCH) != 0;
        let new_stance = if (event.input.actions_bits & bits::STANCE_CYCLE_UP) != 0 {
            Some(character::StanceAction::CycleUp)
        } else if (event.input.actions_bits & bits::STANCE_CYCLE_DOWN) != 0 {
            Some(character::StanceAction::CycleDown)
        } else {
            None
        };
        if new_stance.is_some() {
            desired.stance_action = new_stance;
        }
    }
}

/// Drive the kinematic character controllers for planet-surface walkers.
///
/// The planet-shard uses the `TangentFrame` + `RapierOrigin` re-center
/// trick: the character body lives near (0, height, 0) in flat Rapier
/// space and its horizontal displacement is remapped to sphere-surface
/// rotation in `tangent_frame_sync`. The KCC operates IN that flat
/// frame — no special-case math here — because:
/// 1. Gravity is `-surface_g * flat_Y` (the tangent frame's local Y).
/// 2. Displacement is tangential; curvature is applied post-KCC.
fn kcc_move_characters_system(
    mut rapier: ResMut<RapierContext>,
    config: Res<PlanetConfig>,
    integration: Res<PlanetIntegrationDt>,
    mut characters: Query<
        (
            Entity,
            &CharacterController,
            &mut DesiredMovement,
            &PlatformDelta,
            &MovementStats,
            &mut CharacterVelocity,
            &mut LocomotionState,
            &PlayerYaw,
            &PlatformSnapSuppressed,
        ),
        With<IsCharacter>,
    >,
    mut landed_writer: MessageWriter<LandedEvent>,
    mut collision_writer: MessageWriter<CharacterCollisionEvent>,
) {
    let dt = integration.0;
    let gravity = Vec3::new(0.0, -(config.surface_gravity as f32), 0.0);

    rapier.refresh_query_pipeline();

    // Planet yaw convention: yaw=0 → facing +Z. Shared KCC convention:
    // yaw=0 → facing +X. Offset by +π/2 so pressing W at planet-yaw=0
    // walks toward +Z as before.
    const PLANET_YAW_OFFSET: f32 = std::f32::consts::FRAC_PI_2;

    for (entity, ctrl, mut desired, platform, stats, mut vel, mut state, yaw, snap) in
        characters.iter_mut()
    {
        if state.skips_kcc() {
            desired.clear_edges();
            continue;
        }
        let input = CharacterMoveInput {
            dt,
            prev_state: *state,
            velocity: vel.0,
            desired: *desired,
            platform_delta: *platform,
            gravity,
            yaw: yaw.0 + PLANET_YAW_OFFSET,
            stats,
            crouching: desired.crouch,
            jump_grace_remaining: 0.0,
            snap_suppressed: snap.0,
        };
        let result = move_one_character(
            &rapier.rigid_body_set,
            &rapier.collider_set,
            &rapier.query_pipeline,
            ctrl,
            input,
            |hit| {
                collision_writer.write(CharacterCollisionEvent { entity, hit });
            },
        );
        if let Some(body) = rapier.rigid_body_set.get_mut(ctrl.body) {
            let cur = body.position().translation.vector;
            body.set_next_kinematic_translation(cur + result.translation);
        }
        vel.0 = result.new_velocity;
        *state = result.new_state;
        if let Some(impact) = result.landed_with_impact_speed {
            landed_writer.write(LandedEvent {
                entity,
                impact_speed: impact,
            });
        }
        desired.clear_edges();
    }
}

// ---------------------------------------------------------------------------
// Physics
// ---------------------------------------------------------------------------

fn physics_step(
    mut rapier: ResMut<RapierContext>,
    config: Res<PlanetConfig>,
    mut physics_time: ResMut<PhysicsTimeRes>,
    mut celestial_time: ResMut<CelestialTimeRes>,
    mut tick: ResMut<ecs::TickCounter>,
    sys_params: Res<SystemParamsRes>,
    mut planet_pos: ResMut<PlanetPositionInSystem>,
    epoch: Res<UniverseEpoch>,
) {
    physics_time.0 += 0.05;
    celestial_time.0 = voxeldust_shard_common::harness::celestial_time_from_epoch(
        &epoch.0,
        sys_params.0.as_ref().map(|s| s.scale.time_scale).unwrap_or(1.0),
    );
    tick.0 += 1;

    // Update planet position from Keplerian orbit.
    update_planet_position(
        &sys_params.0,
        config.planet_index,
        celestial_time.0,
        &mut planet_pos,
    );

    // Step Rapier with surface gravity.
    let gravity = vector![0.0, -(config.surface_gravity as f32), 0.0];
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
    // Step invalidates the BVH snapshot; next Physics set must refresh.
    ctx.query_pipeline_fresh = false;
}

/// Tangent frame sync: maps Rapier flat-space deltas to sphere surface movement.
/// Re-centers bodies to prevent f32 drift, recomputes tangent frames.
/// Tangent-frame sync: map the character's flat-space horizontal
/// displacement to a rotation of the sphere-tangent basis, re-center the
/// kinematic body to the origin, and transform the character's
/// persisted velocity from the old frame to the new one.
///
/// Runs AFTER `physics_step` so the body's post-step translation
/// reflects the KCC move + any external dynamic interactions.
fn tangent_frame_sync(
    mut rapier: ResMut<RapierContext>,
    config: Res<PlanetConfig>,
    mut players: Query<
        (
            &CharacterController,
            &mut PlanetPosition,
            &mut TangentFrame,
            &mut RapierOrigin,
            &mut CharacterVelocity,
            &mut LocalUp,
        ),
        With<PlanetPlayer>,
    >,
) {
    let planet_radius = config.planet_radius;

    for (ctrl, mut position, mut frame, mut rapier_origin, mut char_vel, mut local_up) in
        &mut players
    {
        let body = match rapier.rigid_body_set.get_mut(ctrl.body) {
            Some(b) => b,
            None => continue,
        };
        let t = body.translation();
        let rapier_pos = DVec3::new(t.x as f64, t.y as f64, t.z as f64);

        let delta = rapier_pos - rapier_origin.0;
        if delta.length_squared() < 1e-8 {
            continue;
        }

        // Map horizontal displacement to sphere surface movement.
        let tangent_disp = frame.east * delta.x + frame.north * delta.z;
        let horiz_dist = tangent_disp.length();

        let new_up = if horiz_dist > 1e-12 {
            let tangent_dir = tangent_disp / horiz_dist;
            let angle = horiz_dist / planet_radius;
            (frame.up * angle.cos() + tangent_dir * angle.sin()).normalize()
        } else {
            frame.up
        };

        let height = rapier_pos.y as f64;

        // Transform the *persisted* character velocity from the old
        // frame to world space, then back into the new frame. This is
        // the piece that used to read the Rapier body's linvel — now
        // we own it via `CharacterVelocity`.
        let vel_local = char_vel.0;
        let vel_world = frame.east * vel_local.x as f64
            + frame.up * vel_local.y as f64
            + frame.north * vel_local.z as f64;

        // Update to new frame.
        position.0 = new_up * (planet_radius + height);
        *frame = TangentFrame::from_up(new_up);
        local_up.0 = new_up;

        let new_vx = vel_world.dot(frame.east) as f32;
        let new_vy = vel_world.dot(frame.up) as f32;
        let new_vz = vel_world.dot(frame.north) as f32;
        char_vel.0 = Vec3::new(new_vx, new_vy, new_vz);

        // Re-center the body to (0, height, 0) in flat space.
        body.set_translation(vector![0.0, t.y, 0.0], true);

        rapier_origin.0 = DVec3::new(0.0, height, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Detection: ship proximity, disconnect cleanup
// ---------------------------------------------------------------------------

fn ship_proximity(
    tick: Res<ecs::TickCounter>,
    config: Res<PlanetConfig>,
    planet_pos: Res<PlanetPositionInSystem>,
    celestial_time: Res<CelestialTimeRes>,
    players: Query<
        (Entity, &SessionId, &Name, &PlanetPosition, &ActionState, Has<HandoffPending>),
        With<PlanetPlayer>,
    >,
    ships: Query<(&NearbyShipId, &NearbyShipShard, &ShipPosition), With<NearbyShip>>,
    bridge: Res<NetworkBridge>,
    mut commands: Commands,
) {
    if tick.0 % 10 != 0 {
        return;
    }

    for (entity, session_id, name, pos, actions, has_handoff) in &players {
        if has_handoff {
            continue;
        }
        let action_pressed = actions.current == 3 && actions.previous != 3;
        if !action_pressed {
            continue;
        }

        for (ship_id, ship_shard, ship_pos) in &ships {
            let dist = (pos.0 - ship_pos.0).length();
            if dist >= 10.0 {
                continue;
            }

            let player_system = pos.0 + planet_pos.0;

            let h = handoff::PlayerHandoff {
                session_token: session_id.0,
                player_name: name.0.clone(),
                position: player_system,
                velocity: DVec3::ZERO,
                rotation: DQuat::IDENTITY,
                forward: DVec3::NEG_Z,
                fly_mode: false,
                speed_tier: 0,
                grounded: true,
                health: 100.0,
                shield: 100.0,
                source_shard: config.shard_id,
                source_tick: tick.0,
                target_star_index: None,
                galaxy_context: None,
                target_planet_seed: None,
                target_planet_index: None,
                target_ship_id: Some(ship_id.0),
                target_ship_shard_id: Some(ship_shard.0),
                ship_system_position: None,
                ship_rotation: None,
                game_time: celestial_time.0,
                warp_target_star_index: None,
                warp_velocity_gu: None,
                target_system_eva: false,
                schema_version: 1,
                character_state: Vec::new(),
            };

            // Mark player as pending handoff.
            commands.entity(entity).insert(HandoffPending);

            // Send to system shard for routing to ship shard.
            if let Ok(reg) = bridge.peer_registry.try_read() {
                let system_shard = reg
                    .find_by_type(ShardType::System)
                    .first()
                    .map(|s| (s.id, s.endpoint.quic_addr));
                if let Some((sid, addr)) = system_shard {
                    let _ = bridge
                        .quic_send_tx
                        .try_send((sid, addr, ShardMsg::PlayerHandoff(h)));
                    info!(
                        session = session_id.0.0,
                        ship_id = ship_id.0,
                        "ship re-entry handoff initiated"
                    );
                }
            }
            break;
        }
    }
}

fn disconnect_cleanup(
    mut commands: Commands,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
    mut rapier: ResMut<RapierContext>,
    mut player_index: ResMut<PlayerEntityIndex>,
    players: Query<(Entity, &SessionId, &CharacterController, &Name, Has<HandoffPending>), With<PlanetPlayer>>,
) {
    if tick.0 % 40 != 0 {
        return;
    }

    let connected_tokens: std::collections::HashSet<SessionToken> =
        if let Ok(reg) = bridge.client_registry.try_read() {
            players
                .iter()
                .filter(|(_, sid, _, _, _)| reg.has_client(&sid.0))
                .map(|(_, sid, _, _, _)| sid.0)
                .collect()
        } else {
            return;
        };

    let orphaned: Vec<(Entity, SessionToken, RigidBodyHandle, String)> = players
        .iter()
        .filter(|(_, sid, _, _, has_handoff)| {
            !connected_tokens.contains(&sid.0) && !has_handoff
        })
        .map(|(entity, sid, ctrl, name, _)| (entity, sid.0, ctrl.body, name.0.clone()))
        .collect();

    for (entity, token, body_handle, player_name) in orphaned {
        let ctx = &mut *rapier;
        ctx.rigid_body_set.remove(
            body_handle,
            &mut ctx.island_manager,
            &mut ctx.collider_set,
            &mut ctx.impulse_joint_set,
            &mut ctx.multibody_joint_set,
            true,
        );
        player_index.0.remove(&token);
        commands.entity(entity).despawn();
        info!(
            session = token.0,
            player = %player_name,
            "cleaned up disconnected player"
        );
    }
}

// ---------------------------------------------------------------------------
// Broadcast
// ---------------------------------------------------------------------------

fn broadcast_world_state(
    players: Query<(&SessionId, &Name, &PlanetPosition), With<PlanetPlayer>>,
    ships: Query<(&NearbyShipId, &ShipPosition, &ShipRotation), With<NearbyShip>>,
    config: Res<PlanetConfig>,
    sys_params: Res<SystemParamsRes>,
    planet_pos: Res<PlanetPositionInSystem>,
    cached_all_planets: Res<CachedAllPlanetPositions>,
    celestial_time: Res<CelestialTimeRes>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
    external: Res<ExternalEntities>,
) {
    let player_snapshots: Vec<PlayerSnapshotData> = players
        .iter()
        .map(|(sid, _, pos)| PlayerSnapshotData {
            player_id: sid.0.0,
            position: pos.0,
            rotation: DQuat::IDENTITY,
            velocity: DVec3::ZERO,
            grounded: true,
            health: 100.0,
            shield: 100.0,
            seated: false,
        })
        .collect();

    // Sky bodies from system params (uses cached planet positions to avoid
    // redundant Kepler equation solving — positions are computed in physics_step).
    let mut bodies = Vec::new();
    if let Some(ref sys) = sys_params.0 {
        bodies.reserve(sys.planets.len() + 1);
        bodies.push(CelestialBodyData {
            body_id: 0,
            position: -planet_pos.0,
            radius: sys.star.radius_m,
            color: sys.star.color,
        });
        for (i, planet) in sys.planets.iter().enumerate() {
            let planet_sys_pos = cached_all_planets.0.get(i)
                .copied()
                .unwrap_or_else(|| compute_planet_position(planet, celestial_time.0));
            bodies.push(CelestialBodyData {
                body_id: (i + 1) as u32,
                position: planet_sys_pos - planet_pos.0,
                radius: planet.radius_m,
                color: planet.color,
            });
        }
    }

    // Lighting from star.
    let first_player_pos = players
        .iter()
        .next()
        .map(|(_, _, p)| p.0)
        .unwrap_or(DVec3::new(0.0, config.planet_radius + 2.0, 0.0));

    let lighting = if let Some(ref sys) = sys_params.0 {
        let star_dir = (-planet_pos.0 - first_player_pos).normalize();
        let l = compute_lighting(planet_pos.0 + first_player_pos, &sys.star);
        Some(LightingData {
            sun_direction: star_dir,
            sun_color: l.sun_color,
            sun_intensity: l.sun_intensity,
            ambient: l.ambient,
        })
    } else {
        None
    };

    let ship_renders: Vec<ShipRenderData> = ships
        .iter()
        .map(|(id, pos, rot)| ShipRenderData {
            ship_id: id.0,
            position: pos.0,
            rotation: rot.0,
            is_own_ship: false,
        })
        .collect();

    // Build unified entities list: external (system-space) entities transformed
    // into planet-local + every local grounded player as a GroundedPlayer.
    // Origin == planet_pos.0, so the client adds origin to reconstruct world-space.
    let mut entities: Vec<ObservableEntityData> = Vec::with_capacity(
        external.entities.len() + players.iter().count(),
    );
    for e in &external.entities {
        let mut entity = e.clone();
        // Convert system-space position to planet-local frame (origin = planet_pos.0).
        entity.position = e.position - planet_pos.0;
        entities.push(entity);
    }
    for (sid, name, pos) in players.iter() {
        entities.push(ObservableEntityData {
            entity_id: sid.0.0,
            kind: EntityKind::GroundedPlayer,
            position: pos.0,
            rotation: DQuat::IDENTITY,
            velocity: DVec3::ZERO,
            bounding_radius: 1.0,
            lod_tier: LodTier::Full,
            shard_id: config.shard_id.0,
            shard_type: ShardType::Planet as u8,
            is_own: false,
            name: name.0.clone(),
            health: 100.0,
            shield: 100.0,
        });
    }

    let ws = ServerMsg::WorldState(WorldStateData {
        tick: tick.0,
        origin: planet_pos.0,
        players: player_snapshots,
        bodies,
        ships: ship_renders,
        lighting,
        game_time: celestial_time.0,
        warp_target_star_index: 0xFFFFFFFF,
        autopilot: None,
        sub_grids: vec![],
        entities,
    });
    if bridge.broadcast_tx.try_send(ws).is_err() {
        tracing::warn!("WorldState broadcast dropped — channel full");
    }
}

/// Emit a PlanetPlayerDigest at 1 Hz so the system shard can include surface
/// players in its AOI for distant observers (ships, EVA).
fn send_player_digest(
    players: Query<(&SessionId, &Name, &PlanetPosition), With<PlanetPlayer>>,
    config: Res<PlanetConfig>,
    planet_pos: Res<PlanetPositionInSystem>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
) {
    // ~1 Hz at 20 Hz tick.
    if tick.0 % 20 != 0 {
        return;
    }
    if players.is_empty() {
        return;
    }

    let entries: Vec<PlanetPlayerDigestEntry> = players
        .iter()
        .map(|(sid, name, pos)| PlanetPlayerDigestEntry {
            session_token: sid.0,
            player_name: name.0.clone(),
            // Transform planet-local position to system-space (what system shard expects).
            position: planet_pos.0 + pos.0,
            rotation: DQuat::IDENTITY,
            planet_index: config.planet_index,
        })
        .collect();

    let digest = ShardMsg::PlanetPlayerDigest(PlanetPlayerDigestData {
        planet_shard: config.shard_id,
        planet_seed: config.planet_seed,
        planet_index: config.planet_index,
        entries,
        tick: tick.0,
    });

    // Find the system shard and send.
    if let Ok(reg) = bridge.peer_registry.try_read() {
        if let Some(info) = reg.find_by_type(ShardType::System).first() {
            let _ = bridge.quic_send_tx.try_send((info.id, info.endpoint.quic_addr, digest));
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

fn log_state(
    players: Query<&PlanetPlayer>,
    config: Res<PlanetConfig>,
    physics_time: Res<PhysicsTimeRes>,
    celestial_time: Res<CelestialTimeRes>,
    tick: Res<ecs::TickCounter>,
) {
    if tick.0 % 100 != 0 || tick.0 == 0 {
        return;
    }
    info!(
        physics_time = format!("{:.1}s", physics_time.0),
        celestial_time = format!("{:.1}s", celestial_time.0),
        players = players.iter().count(),
        surface_g = format!("{:.2} m/s²", config.surface_gravity),
        "planet state"
    );
}

// ---------------------------------------------------------------------------
// App construction
// ---------------------------------------------------------------------------

fn build_app(
    shard_id: ShardId,
    planet_seed: u64,
    planet_radius: f64,
    planet_mass: f64,
    system_seed: Option<u64>,
    planet_index: u32,
    universe_epoch: Arc<std::sync::atomic::AtomicU64>,
) -> App {
    let surface_gravity = G * planet_mass / (planet_radius * planet_radius);

    let rigid_body_set = RigidBodySet::new();
    let mut collider_set = ColliderSet::new();

    // Ground: halfspace at Y=0.
    let ground = ColliderBuilder::halfspace(nalgebra::Unit::new_normalize(vector![0.0, 1.0, 0.0]))
        .translation(vector![0.0, 0.0, 0.0])
        .collision_groups(InteractionGroups::new(Group::GROUP_2, Group::GROUP_1))
        .build();
    collider_set.insert(ground);

    let system_params = system_seed.map(SystemParams::from_seed);
    let planet_position_in_system = system_params
        .as_ref()
        .and_then(|sys| sys.planets.get(planet_index as usize))
        .map(|p| compute_planet_position(p, 0.0))
        .unwrap_or(DVec3::ZERO);

    let mut app = App::new();

    const PLANET_TICK_DT: f32 = 0.05;
    app.insert_resource(PlanetIntegrationDt(PLANET_TICK_DT));
    app.insert_resource(RapierContext {
        rigid_body_set,
        collider_set,
        integration_params: {
            let mut p = IntegrationParameters::default();
            p.dt = PLANET_TICK_DT;
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
        query_pipeline_fresh: false,
    });
    app.insert_resource(PlanetConfig {
        shard_id,
        planet_seed,
        planet_radius,
        planet_mass,
        surface_gravity,
        system_seed,
        planet_index,
    });
    app.insert_resource(SystemParamsRes(system_params));
    app.insert_resource(PlanetPositionInSystem(planet_position_in_system));
    app.insert_resource(CachedAllPlanetPositions::default());
    app.insert_resource(CelestialTimeRes::default());
    app.insert_resource(PhysicsTimeRes::default());
    app.insert_resource(ecs::TickCounter::default());
    app.insert_resource(UniverseEpoch(universe_epoch));
    app.insert_resource(PlayerEntityIndex::default());
    app.insert_resource(ShipEntityIndex::default());
    app.insert_resource(PendingHandoffs::default());
    app.insert_resource(ExternalEntities::default());

    // Messages.
    app.add_message::<ClientConnectedMsg>();
    app.add_message::<PlayerInputMsg>();
    app.add_message::<InboundHandoffMsg>();
    app.add_message::<ShipNearbyMsg>();
    app.add_message::<HandoffAcceptedMsg>();
    app.add_message::<ShipColliderSyncMsg>();
    // Character-layer events (reserved — consumed by future gameplay).
    app.add_message::<LandedEvent>();
    app.add_message::<CharacterCollisionEvent>();

    // System ordering.
    app.configure_sets(
        Update,
        (
            PlanetSet::Bridge,
            PlanetSet::Spawn,
            PlanetSet::Input,
            PlanetSet::Physics,
            PlanetSet::Detection,
            PlanetSet::Broadcast,
            PlanetSet::Diagnostics,
        )
            .chain(),
    );

    // Bridge.
    app.add_systems(
        Update,
        (drain_connects, drain_input, drain_quic).in_set(PlanetSet::Bridge),
    );

    // Spawn: process_handoffs stores handoff data in PendingHandoffs resource,
    // process_connects consumes it to spawn at the correct position.
    app.add_systems(
        Update,
        (
            process_connects,
            process_handoffs,
            process_ship_nearby,
            process_ship_colliders,
            process_handoff_accepted,
        )
            .in_set(PlanetSet::Spawn),
    );

    // apply_deferred so newly spawned entities are visible to Input/Physics.
    app.add_systems(
        Update,
        bevy_ecs::schedule::ApplyDeferred
            .after(PlanetSet::Spawn)
            .before(PlanetSet::Input),
    );

    // Input.
    app.add_systems(Update, process_input.in_set(PlanetSet::Input));

    // Physics.
    //
    // Order: `kcc_move_characters_system` drives character translation
    // from `DesiredMovement` → `set_next_kinematic_translation`. Then
    // `physics_step` runs Rapier's full step (kinematic bodies pick up
    // their new positions; any dynamic bodies integrate normally).
    // `tangent_frame_sync` re-centers the character in the flat Rapier
    // frame and updates the sphere-surface tangent basis.
    app.add_systems(
        Update,
        (
            kcc_move_characters_system,
            physics_step,
            refresh_planet_position_cache,
            tangent_frame_sync,
            update_ship_collider_positions,
        )
            .chain()
            .in_set(PlanetSet::Physics),
    );

    // Detection.
    app.add_systems(
        Update,
        (ship_proximity, disconnect_cleanup).in_set(PlanetSet::Detection),
    );

    // apply_deferred so despawned entities don't appear in broadcast.
    app.add_systems(
        Update,
        bevy_ecs::schedule::ApplyDeferred
            .after(PlanetSet::Detection)
            .before(PlanetSet::Broadcast),
    );

    // Broadcast.
    app.add_systems(
        Update,
        (broadcast_world_state, send_player_digest).in_set(PlanetSet::Broadcast),
    );

    // Diagnostics.
    app.add_systems(Update, log_state.in_set(PlanetSet::Diagnostics));

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

    let (planet_radius, planet_mass) = if let Some(sys_seed) = args.system_seed {
        let sys = SystemParams::from_seed(sys_seed);
        if let Some(planet) = sys.planets.get(args.planet_index as usize) {
            (planet.radius_m, planet.mass_kg)
        } else {
            (6.371e6, 5.972e24)
        }
    } else {
        (6.371e6, 5.972e24)
    };

    let bind = "0.0.0.0";
    let config = ShardHarnessConfig {
        shard_id: ShardId(args.shard_id),
        shard_type: ShardType::Planet,
        tcp_addr: format!("{bind}:{}", args.tcp_port).parse().unwrap(),
        udp_addr: format!("{bind}:{}", args.udp_port).parse().unwrap(),
        quic_addr: format!("{bind}:{}", args.quic_port).parse().unwrap(),
        orchestrator_url: args.orchestrator,
        orchestrator_heartbeat_addr: args.orchestrator_heartbeat,
        healthz_addr: format!("{bind}:{}", args.healthz_port).parse().unwrap(),
        planet_seed: Some(args.seed),
        system_seed: args.system_seed,
        ship_id: None,
        galaxy_seed: None,
        host_shard_id: None,
        advertise_host: args.advertise_host,
    };

    info!(
        shard_id = args.shard_id,
        planet_seed = args.seed,
        radius_km = planet_radius / 1000.0,
        surface_g = format!("{:.2}", G * planet_mass / (planet_radius * planet_radius)),
        "planet shard starting"
    );

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let harness = ShardHarness::new(config);
        let universe_epoch = harness.epoch_arc();
        let app = build_app(
            ShardId(args.shard_id),
            args.seed,
            planet_radius,
            planet_mass,
            args.system_seed,
            args.planet_index,
            universe_epoch,
        );

        info!("planet shard ECS app built, starting harness");
        harness.run_ecs(app).await;
    });
}
