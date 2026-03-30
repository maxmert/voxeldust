use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use clap::Parser;
use glam::{DQuat, DVec3, Vec3};
use rapier3d::prelude::*;
use tracing::info;

use voxeldust_core::client_message::{
    CelestialBodyData, JoinResponseData, LightingData, PlayerSnapshotData,
    ServerMsg, WorldStateData,
};
use voxeldust_core::handoff;
use voxeldust_core::shard_message::{ShardMsg, ShipNearbyInfoData};
use voxeldust_core::shard_types::{ShardId, ShardType, SessionToken};
use voxeldust_core::system::{
    compute_lighting, compute_planet_position, SystemParams,
};
use voxeldust_shard_common::client_listener;
use voxeldust_shard_common::harness::{ShardHarness, ShardHarnessConfig};

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

const G: f64 = 6.674e-11;
const WALK_SPEED: f32 = 4.0;
const JUMP_IMPULSE: f32 = 5.0;

struct PlayerOnPlanet {
    session_token: SessionToken,
    player_name: String,
    body_handle: RigidBodyHandle,
    position: DVec3,
    up: DVec3,
    yaw: f32,
    last_action: u8,
    prev_action: u8,
    handoff_pending: bool,
}

/// Ship near the planet surface, tracked for rendering and re-entry detection.
struct NearbyShip {
    ship_id: u64,
    ship_shard_id: ShardId,
    position: DVec3,    // system-space
    rotation: DQuat,
    velocity: DVec3,
}

struct PlanetState {
    shard_id: ShardId,
    planet_seed: u64,
    planet_radius: f64,
    planet_mass: f64,
    surface_gravity: f64,
    system_seed: Option<u64>,
    planet_index: u32,

    // Rapier physics
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
    ground_collider: Option<ColliderHandle>,

    players: HashMap<SessionToken, PlayerOnPlanet>,
    /// Ships near this planet's surface (from system shard ShipNearbyInfo).
    nearby_ships: HashMap<u64, NearbyShip>,

    // System context
    system_params: Option<SystemParams>,
    physics_time: f64,
    celestial_time: f64,
    tick_count: u64,
    planet_position_in_system: DVec3,
}

impl PlanetState {
    fn new(shard_id: ShardId, planet_seed: u64, planet_radius: f64, planet_mass: f64, system_seed: Option<u64>, planet_index: u32) -> Self {
        let surface_gravity = G * planet_mass / (planet_radius * planet_radius);

        let rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();

        let ground = ColliderBuilder::halfspace(nalgebra::Unit::new_normalize(vector![0.0, 1.0, 0.0]))
            .translation(vector![0.0, 0.0, 0.0])
            .build();
        let ground_handle = collider_set.insert(ground);

        let system_params = system_seed.map(SystemParams::from_seed);

        Self {
            shard_id, planet_seed, planet_radius, planet_mass, surface_gravity,
            system_seed, planet_index,
            rigid_body_set, collider_set,
            integration_params: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            ground_collider: Some(ground_handle),
            players: HashMap::new(),
            nearby_ships: HashMap::new(),
            system_params,
            physics_time: 0.0,
            celestial_time: 0.0,
            tick_count: 0,
            planet_position_in_system: DVec3::ZERO,
        }
    }

    fn spawn_player(&mut self, session_token: SessionToken, name: String) {
        self.spawn_player_at(session_token, name, DVec3::new(0.0, self.planet_radius + 2.0, 0.0));
    }

    /// Spawn a player at a specific planet-local position.
    fn spawn_player_at(&mut self, session_token: SessionToken, name: String, planet_local_pos: DVec3) {
        // Rapier works in flat space near the surface. Convert planet-local
        // position to flat-space: y is height above surface, x/z are tangent.
        let height_above_surface = planet_local_pos.length() - self.planet_radius;
        // For simplicity, project to flat ground coords at the pole.
        // Full spherical → flat mapping will come with proper chunk-based terrain.
        let flat_y = height_above_surface.max(0.5) as f32;

        let player_rb = RigidBodyBuilder::dynamic()
            .translation(vector![0.0, flat_y, 0.0])
            .lock_rotations()
            .build();
        let handle = self.rigid_body_set.insert(player_rb);
        let player_collider = ColliderBuilder::capsule_y(0.6, 0.3).build();
        self.collider_set.insert_with_parent(player_collider, handle, &mut self.rigid_body_set);

        self.players.insert(session_token, PlayerOnPlanet {
            session_token, player_name: name,
            body_handle: handle,
            position: planet_local_pos,
            up: planet_local_pos.normalize(),
            yaw: 0.0,
            last_action: 0,
            prev_action: 0,
            handoff_pending: false,
        });
    }

    fn step(&mut self) {
        let gravity = vector![0.0, -(self.surface_gravity as f32), 0.0];

        self.physics_pipeline.step(
            &gravity, &self.integration_params,
            &mut self.island_manager, &mut self.broad_phase, &mut self.narrow_phase,
            &mut self.rigid_body_set, &mut self.collider_set,
            &mut self.impulse_joint_set, &mut self.multibody_joint_set,
            &mut self.ccd_solver, Some(&mut self.query_pipeline), &(), &(),
        );

        for player in self.players.values_mut() {
            if let Some(body) = self.rigid_body_set.get(player.body_handle) {
                let t = body.translation();
                player.position = DVec3::new(t.x as f64, self.planet_radius + t.y as f64, t.z as f64);
                player.up = player.position.normalize();
            }
        }
    }

    fn compute_planet_system_position(&mut self) {
        if let Some(ref sys) = self.system_params {
            if let Some(planet) = sys.planets.get(self.planet_index as usize) {
                self.planet_position_in_system = compute_planet_position(planet, self.celestial_time);
            }
        }
    }

    fn build_world_state(&self) -> WorldStateData {
        // Player positions.
        let players: Vec<PlayerSnapshotData> = self.players.values().map(|p| {
            PlayerSnapshotData {
                player_id: p.session_token.0,
                position: p.position,
                rotation: DQuat::IDENTITY,
                velocity: DVec3::ZERO,
                grounded: true,
                health: 100.0,
                shield: 100.0,
            }
        }).collect();

        // Sky bodies — compute from system params, transform to planet-local frame.
        let mut bodies = Vec::new();
        if let Some(ref sys) = self.system_params {
            // Star: in planet-local frame, star is at -planet_position.
            bodies.push(CelestialBodyData {
                body_id: 0,
                position: -self.planet_position_in_system,
                radius: sys.star.radius_m,
                color: sys.star.color,
            });

            // Other planets.
            for (i, planet) in sys.planets.iter().enumerate() {
                let planet_sys_pos = compute_planet_position(planet, self.celestial_time);
                bodies.push(CelestialBodyData {
                    body_id: (i + 1) as u32,
                    position: planet_sys_pos - self.planet_position_in_system,
                    radius: planet.radius_m,
                    color: planet.color,
                });
            }
        }

        // Lighting from star.
        let first_player_pos = self.players.values().next()
            .map(|p| p.position)
            .unwrap_or(DVec3::new(0.0, self.planet_radius + 2.0, 0.0));

        let lighting = if let Some(ref sys) = self.system_params {
            let star_dir_from_surface = (-self.planet_position_in_system - first_player_pos).normalize();
            let l = compute_lighting(self.planet_position_in_system + first_player_pos, &sys.star);
            Some(LightingData {
                sun_direction: star_dir_from_surface,
                sun_color: l.sun_color,
                sun_intensity: l.sun_intensity,
                ambient: l.ambient,
            })
        } else {
            None
        };

        WorldStateData {
            tick: self.tick_count,
            origin: self.planet_position_in_system,
            players, bodies,
            ships: vec![],
            lighting,
            game_time: self.celestial_time,
        }
    }
}

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
        shard_id = args.shard_id, planet_seed = args.seed,
        radius_km = planet_radius / 1000.0,
        surface_g = format!("{:.2}", G * planet_mass / (planet_radius * planet_radius)),
        "planet shard starting"
    );

    let system_seed = args.system_seed;
    let planet_index = args.planet_index;

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let mut harness = ShardHarness::new(config);

        let state = Arc::new(Mutex::new(
            PlanetState::new(ShardId(args.shard_id), args.seed, planet_radius, planet_mass, system_seed, planet_index)
        ));
        let client_registry = harness.client_registry.clone();

        // Take channels.
        let mut connect_rx = std::mem::replace(
            &mut harness.connect_rx, tokio::sync::mpsc::unbounded_channel().1,
        );
        let mut input_rx = std::mem::replace(
            &mut harness.input_rx, tokio::sync::mpsc::unbounded_channel().1,
        );
        let mut quic_msg_rx = std::mem::replace(
            &mut harness.quic_msg_rx, tokio::sync::mpsc::unbounded_channel().1,
        );
        let broadcast_tx = harness.broadcast_tx.clone();
        let quic_send_tx = harness.quic_send_tx.clone();
        let peer_registry = harness.peer_registry.clone();

        // System 1: Drain connects — register + send JoinResponse.
        let state_connect = state.clone();
        let registry_connect = client_registry.clone();
        harness.add_system("drain_connects", move || {
            while let Ok(event) = connect_rx.try_recv() {
                let conn = event.connection;
                let token = conn.session_token;

                if let Ok(mut reg) = registry_connect.try_write() {
                    reg.register(&conn);
                }

                let st = state_connect.lock().unwrap();
                let tcp_stream = conn.tcp_stream.clone();
                let game_time = st.celestial_time;
                let planet_pos = st.planet_position_in_system;
                let sys_seed = st.system_seed.unwrap_or(0);

                // Spawn player.
                drop(st);
                {
                    let mut st = state_connect.lock().unwrap();
                    st.spawn_player(token, conn.player_name.clone());
                }

                tokio::spawn(async move {
                    let jr = ServerMsg::JoinResponse(JoinResponseData {
                        seed: 0, planet_radius: 0, player_id: token.0,
                        spawn_position: DVec3::new(0.0, 2.0, 0.0),
                        spawn_rotation: DQuat::IDENTITY,
                        spawn_forward: DVec3::NEG_Z,
                        session_token: token,
                        shard_type: 0, // Planet
                        galaxy_seed: 0, system_seed: sys_seed,
                        game_time,
                        reference_position: planet_pos,
                        reference_rotation: DQuat::IDENTITY,
                    });
                    let mut stream = tcp_stream.lock().await;
                    let _ = client_listener::send_tcp_msg(&mut stream, &jr).await;
                });

                info!(player = %conn.player_name, session = token.0, "player spawned on planet");
            }
        });

        // System 2: Drain PlayerInput — walk on surface.
        let state_input = state.clone();
        harness.add_system("drain_input", move || {
            while let Ok((_src, input)) = input_rx.try_recv() {
                let mut st = state_input.lock().unwrap();

                // Apply to first player (simplified — multi-player needs routing by session).
                let player_info: Option<(RigidBodyHandle, f32)> = st.players.values_mut().next()
                    .map(|p| {
                        p.prev_action = p.last_action;
                        p.last_action = input.action;
                        p.yaw = input.look_yaw;
                        (p.body_handle, p.yaw)
                    });

                if let Some((handle, yaw)) = player_info {
                    if let Some(body) = st.rigid_body_set.get_mut(handle) {
                        let (sin_y, cos_y) = yaw.sin_cos();
                        let fwd = Vec3::new(cos_y, 0.0, sin_y);
                        let right = Vec3::new(-sin_y, 0.0, cos_y);

                        let move_vel = fwd * input.movement[2] * WALK_SPEED
                            + right * input.movement[0] * WALK_SPEED;

                        let current_vel = *body.linvel();
                        body.set_linvel(vector![move_vel.x, current_vel.y, move_vel.z], true);

                        if input.jump && current_vel.y.abs() < 0.5 {
                            body.apply_impulse(vector![0.0, JUMP_IMPULSE, 0.0], true);
                        }
                    }
                }
            }
        });

        // System 3: Drain QUIC — receive PlayerHandoff, ShipNearbyInfo, HandoffAccepted.
        let state_quic = state.clone();
        let peer_reg_quic = peer_registry.clone();
        let quic_send_quic = quic_send_tx.clone();
        let client_reg_quic = client_registry.clone();
        harness.add_system("drain_quic", move || {
            while let Ok(msg) = quic_msg_rx.try_recv() {
                match msg {
                    ShardMsg::PlayerHandoff(h) => {
                        let mut st = state_quic.lock().unwrap();
                        let planet_local = h.position - st.planet_position_in_system;
                        let surface_pos = planet_local.normalize() * (st.planet_radius + 2.0);

                        info!(
                            player = %h.player_name,
                            session = h.session_token.0,
                            "received player handoff, spawning at surface"
                        );

                        st.spawn_player_at(h.session_token, h.player_name.clone(), surface_pos);

                        // Send HandoffAccepted back to system shard (which relays to source).
                        let shard_id = st.shard_id;
                        let accepted = ShardMsg::HandoffAccepted(handoff::HandoffAccepted {
                            session_token: h.session_token,
                            target_shard: shard_id,
                        });
                        // Send to source shard via the system shard (host).
                        // The system shard relays based on pending_handoffs.
                        let source = h.source_shard;
                        drop(st);
                        if let Ok(reg) = peer_reg_quic.try_read() {
                            // Find the system shard to relay through.
                            let system_shard = reg.find_by_type(ShardType::System).first()
                                .map(|s| (s.id, s.endpoint.quic_addr));
                            if let Some((sid, addr)) = system_shard {
                                let _ = quic_send_quic.send((sid, addr, accepted));
                                info!(target = sid.0, "sent HandoffAccepted to system shard");
                            }
                        }
                    }
                    ShardMsg::ShipNearbyInfo(info) => {
                        let mut st = state_quic.lock().unwrap();
                        st.nearby_ships.insert(info.ship_id, NearbyShip {
                            ship_id: info.ship_id,
                            ship_shard_id: info.ship_shard_id,
                            position: info.position,
                            rotation: info.rotation,
                            velocity: info.velocity,
                        });
                    }
                    ShardMsg::HandoffAccepted(accepted) => {
                        // Ship shard accepted re-entry. Send ShardRedirect to client.
                        let st = state_quic.lock().unwrap();
                        let session = accepted.session_token;
                        let target_shard = accepted.target_shard;
                        info!(session = session.0, target = target_shard.0,
                            "received HandoffAccepted for ship re-entry");

                        drop(st);
                        if let Ok(reg) = peer_reg_quic.try_read() {
                            if let Some(info) = reg.get(target_shard) {
                                let redirect = ServerMsg::ShardRedirect(handoff::ShardRedirect {
                                    session_token: session,
                                    target_tcp_addr: info.endpoint.tcp_addr.to_string(),
                                    target_udp_addr: info.endpoint.udp_addr.to_string(),
                                    shard_id: target_shard,
                                });
                                let cr = client_reg_quic.clone();
                                tokio::spawn(async move {
                                    if let Ok(reg) = cr.try_read() {
                                        if let Err(e) = reg.send_tcp(session, &redirect).await {
                                            tracing::warn!(%e, "failed to send ShardRedirect for re-entry");
                                        }
                                    }
                                });
                            }
                        }

                        // Remove player from planet state.
                        let mut st = state_quic.lock().unwrap();
                        if let Some(player) = st.players.remove(&session) {
                            // Remove physics body via destructured borrows.
                            let PlanetState {
                                ref mut rigid_body_set, ref mut island_manager,
                                ref mut collider_set, ref mut impulse_joint_set,
                                ref mut multibody_joint_set, ..
                            } = *st;
                            rigid_body_set.remove(
                                player.body_handle,
                                island_manager, collider_set,
                                impulse_joint_set, multibody_joint_set,
                                true,
                            );
                            info!(session = session.0, "player removed from planet (re-entry)");
                        }
                    }
                    _ => {}
                }
            }
        });

        // System 4: Physics step.
        let state_physics = state.clone();
        harness.add_system("physics_step", move || {
            let mut st = state_physics.lock().unwrap();
            st.physics_time += 0.05;
            st.celestial_time += 0.05 * st.system_params.as_ref()
                .map(|s| s.scale.time_scale).unwrap_or(1.0);
            st.tick_count += 1;
            st.compute_planet_system_position();
            st.step();
        });

        // System 5: Ship proximity detection — player E-press near ship triggers re-entry.
        let state_proximity = state.clone();
        let quic_send_proximity = quic_send_tx.clone();
        let peer_reg_proximity = peer_registry.clone();
        harness.add_system("ship_proximity", move || {
            let mut st = state_proximity.lock().unwrap();
            if st.tick_count % 10 != 0 { return; } // check every 0.5s

            // Collect re-entry handoff requests (can't mutate while iterating).
            let mut handoff_requests: Vec<(SessionToken, String, DVec3, u64, ShardId)> = Vec::new();

            for player in st.players.values() {
                if player.handoff_pending { continue; }
                // E key rising edge (action 3).
                let action_pressed = player.last_action == 3 && player.prev_action != 3;
                if !action_pressed { continue; }

                // Check proximity to each nearby ship (planet-local coords).
                for ship in st.nearby_ships.values() {
                    let ship_planet_local = ship.position - st.planet_position_in_system;
                    let dist = (player.position - ship_planet_local).length();
                    let interact_range = 10.0; // meters

                    if dist < interact_range {
                        let player_system = player.position + st.planet_position_in_system;
                        handoff_requests.push((
                            player.session_token,
                            player.player_name.clone(),
                            player_system,
                            ship.ship_id,
                            ship.ship_shard_id,
                        ));
                        break;
                    }
                }
            }

            // Process handoff requests.
            for (session, name, system_pos, ship_id, ship_shard_id) in handoff_requests {
                if let Some(player) = st.players.get_mut(&session) {
                    player.handoff_pending = true;
                }

                let h = handoff::PlayerHandoff {
                    session_token: session,
                    player_name: name,
                    position: system_pos,
                    velocity: DVec3::ZERO,
                    rotation: DQuat::IDENTITY,
                    forward: DVec3::NEG_Z,
                    fly_mode: false,
                    speed_tier: 0,
                    grounded: true,
                    health: 100.0,
                    shield: 100.0,
                    source_shard: st.shard_id,
                    source_tick: st.tick_count,
                    target_star_index: None,
                    galaxy_context: None,
                    target_planet_seed: None,
                    target_planet_index: None,
                    target_ship_id: Some(ship_id),
                    target_ship_shard_id: Some(ship_shard_id),
                    ship_system_position: None,
                    ship_rotation: None,
                };

                // Send to system shard for routing to ship shard.
                if let Ok(reg) = peer_reg_proximity.try_read() {
                    let system_shard = reg.find_by_type(ShardType::System).first()
                        .map(|s| (s.id, s.endpoint.quic_addr));
                    if let Some((sid, addr)) = system_shard {
                        let _ = quic_send_proximity.send((sid, addr, ShardMsg::PlayerHandoff(h)));
                        info!(session = session.0, ship_id, "ship re-entry handoff initiated");
                    }
                }
            }
        });

        // System 6: Broadcast WorldState.
        let state_broadcast = state.clone();
        harness.add_system("broadcast", move || {
            let st = state_broadcast.lock().unwrap();
            let ws = ServerMsg::WorldState(st.build_world_state());
            let _ = broadcast_tx.send(ws);
        });

        // System 6: Periodic log.
        let state_log = state.clone();
        harness.add_system("log_state", move || {
            let st = state_log.lock().unwrap();
            if st.tick_count % 100 == 0 && st.tick_count > 0 {
                info!(
                    physics_time = format!("{:.1}s", st.physics_time),
                    celestial_time = format!("{:.1}s", st.celestial_time),
                    players = st.players.len(),
                    surface_g = format!("{:.2} m/s²", st.surface_gravity),
                    "planet state"
                );
            }
        });

        harness.run().await;
    });
}
