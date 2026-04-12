use std::collections::HashMap;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use clap::Parser;
use glam::{DQuat, DVec3, Vec3};
use rapier3d::prelude::*;
use tracing::{info, warn};

use voxeldust_core::client_message::{
    CelestialBodyData, JoinResponseData, LightingData, PlayerSnapshotData, ServerMsg,
    ShipRenderData, WorldStateData,
};
use voxeldust_core::ecs;
use voxeldust_core::handoff;
use voxeldust_core::shard_message::{ShardMsg, ShipNearbyInfoData};
use voxeldust_core::shard_types::{SessionToken, ShardId, ShardType};
use voxeldust_core::system::{compute_lighting, compute_planet_position, SystemParams};
use voxeldust_shard_common::client_listener;
use voxeldust_shard_common::harness::{NetworkBridge, ShardHarness, ShardHarnessConfig};

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
const WALK_SPEED: f32 = 4.0;
const JUMP_IMPULSE: f32 = 5.0;

// ---------------------------------------------------------------------------
// Components (planet-shard-specific, on player entities)
// ---------------------------------------------------------------------------

/// Marker for a player entity on this planet.
#[derive(Component)]
struct PlanetPlayer;

/// Rapier rigid body handle for this player.
#[derive(Component)]
struct PlayerBody(RigidBodyHandle);

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
}

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
    mut celestial_time: ResMut<CelestialTimeRes>,
    mut planet_pos: ResMut<PlanetPositionInSystem>,
    mut cached_all_planets: ResMut<CachedAllPlanetPositions>,
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
                    update_planet_position(&sys_params.0, config.planet_index, celestial_time.0, &mut planet_pos, &mut cached_all_planets);
                }
                handoff_events.write(InboundHandoffMsg {
                    handoff: h,
                    relay_shard: queued.source_shard_id,
                });
            }
            ShardMsg::ShipNearbyInfo(info) => {
                if info.game_time > celestial_time.0 {
                    celestial_time.0 = info.game_time;
                    update_planet_position(&sys_params.0, config.planet_index, celestial_time.0, &mut planet_pos, &mut cached_all_planets);
                }
                ship_nearby_events.write(ShipNearbyMsg(info));
            }
            ShardMsg::HandoffAccepted(accepted) => {
                accepted_events.write(HandoffAcceptedMsg {
                    session: accepted.session_token,
                    target_shard: accepted.target_shard,
                });
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
    cached_all: &mut CachedAllPlanetPositions,
) {
    if let Some(sys) = sys_params {
        if let Some(planet) = sys.planets.get(planet_index as usize) {
            planet_pos.0 = compute_planet_position(planet, celestial_time);
        }
        // Refresh all planet positions while we're computing orbits.
        cached_all.0.clear();
        for planet in &sys.planets {
            cached_all.0.push(compute_planet_position(planet, celestial_time));
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

    let player_rb = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, height as f32, 0.0])
        .lock_rotations()
        .build();
    let handle = rapier.rigid_body_set.insert(player_rb);
    let player_collider = ColliderBuilder::capsule_y(0.6, 0.3)
        .collision_groups(InteractionGroups::new(
            Group::GROUP_1,
            Group::GROUP_1 | Group::GROUP_2,
        ))
        .build();
    rapier.collider_set.insert_with_parent(
        player_collider,
        handle,
        &mut rapier.rigid_body_set,
    );

    commands
        .spawn((
            PlanetPlayer,
            PlayerBody(handle),
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
        ))
        .id()
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
        }
    }
}

fn process_handoff_accepted(
    mut commands: Commands,
    mut events: MessageReader<HandoffAcceptedMsg>,
    mut rapier: ResMut<RapierContext>,
    bridge: Res<NetworkBridge>,
    player_index: Res<PlayerEntityIndex>,
    players: Query<&PlayerBody, With<PlanetPlayer>>,
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

        // Remove player entity and Rapier body.
        if let Some(&entity) = player_index.0.get(&session) {
            if let Ok(body) = players.get(entity) {
                let ctx = &mut *rapier;
                ctx.rigid_body_set.remove(
                    body.0,
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

fn process_input(
    mut events: MessageReader<PlayerInputMsg>,
    player_index: Res<PlayerEntityIndex>,
    mut players: Query<(&PlayerBody, &mut PlayerYaw, &mut ActionState), With<PlanetPlayer>>,
    mut rapier: ResMut<RapierContext>,
) {
    for event in events.read() {
        let entity = match player_index.0.get(&event.session) {
            Some(&e) => e,
            None => continue,
        };
        let Ok((body, mut yaw, mut actions)) = players.get_mut(entity) else {
            continue;
        };

        actions.previous = actions.current;
        actions.current = event.input.action;
        yaw.0 = event.input.look_yaw;

        let has_movement = event.input.movement[0].abs() > 0.001
            || event.input.movement[2].abs() > 0.001;

        if let Some(rb) = rapier.rigid_body_set.get_mut(body.0) {
            if has_movement || event.input.jump {
                let (sin_y, cos_y) = yaw.0.sin_cos();
                let fwd = Vec3::new(sin_y, 0.0, cos_y);
                let right = Vec3::new(cos_y, 0.0, -sin_y);

                let move_vel = fwd * event.input.movement[2] * WALK_SPEED
                    + right * event.input.movement[0] * WALK_SPEED;

                let current_vel = *rb.linvel();
                rb.set_linvel(vector![move_vel.x, current_vel.y, move_vel.z], true);

                if event.input.jump && current_vel.y.abs() < 0.5 {
                    rb.apply_impulse(vector![0.0, JUMP_IMPULSE, 0.0], true);
                }
            }
        }
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
    mut cached_all_planets: ResMut<CachedAllPlanetPositions>,
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
        &mut cached_all_planets,
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
}

/// Tangent frame sync: maps Rapier flat-space deltas to sphere surface movement.
/// Re-centers bodies to prevent f32 drift, recomputes tangent frames.
fn tangent_frame_sync(
    mut rapier: ResMut<RapierContext>,
    config: Res<PlanetConfig>,
    mut players: Query<
        (
            &PlayerBody,
            &mut PlanetPosition,
            &mut TangentFrame,
            &mut RapierOrigin,
        ),
        With<PlanetPlayer>,
    >,
) {
    let planet_radius = config.planet_radius;

    for (body_comp, mut position, mut frame, mut rapier_origin) in &mut players {
        let body = match rapier.rigid_body_set.get_mut(body_comp.0) {
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

        // Transform velocity from old frame to world, then into new frame.
        let vel = body.linvel();
        let vel_world = frame.east * vel.x as f64
            + frame.up * vel.y as f64
            + frame.north * vel.z as f64;

        // Update to new frame.
        position.0 = new_up * (planet_radius + height);
        *frame = TangentFrame::from_up(new_up);

        let new_vx = vel_world.dot(frame.east) as f32;
        let new_vy = vel_world.dot(frame.up) as f32;
        let new_vz = vel_world.dot(frame.north) as f32;
        body.set_translation(vector![0.0, t.y, 0.0], true);
        body.set_linvel(vector![new_vx, new_vy, new_vz], true);

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
    players: Query<(Entity, &SessionId, &PlayerBody, &Name, Has<HandoffPending>), With<PlanetPlayer>>,
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
        .map(|(entity, sid, body, name, _)| (entity, sid.0, body.0, name.0.clone()))
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
    players: Query<(&SessionId, &PlanetPosition), With<PlanetPlayer>>,
    ships: Query<(&NearbyShipId, &ShipPosition, &ShipRotation), With<NearbyShip>>,
    config: Res<PlanetConfig>,
    sys_params: Res<SystemParamsRes>,
    planet_pos: Res<PlanetPositionInSystem>,
    cached_all_planets: Res<CachedAllPlanetPositions>,
    celestial_time: Res<CelestialTimeRes>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
) {
    let player_snapshots: Vec<PlayerSnapshotData> = players
        .iter()
        .map(|(sid, pos)| PlayerSnapshotData {
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
        .map(|(_, p)| p.0)
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
    });
    let _ = bridge.broadcast_tx.try_send(ws);
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

    // Messages.
    app.add_message::<ClientConnectedMsg>();
    app.add_message::<PlayerInputMsg>();
    app.add_message::<InboundHandoffMsg>();
    app.add_message::<ShipNearbyMsg>();
    app.add_message::<HandoffAcceptedMsg>();

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
    app.add_systems(
        Update,
        (physics_step, tangent_frame_sync)
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
    app.add_systems(Update, broadcast_world_state.in_set(PlanetSet::Broadcast));

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
