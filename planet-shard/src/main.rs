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
use voxeldust_core::handoff::HandoffAccepted;
use voxeldust_core::shard_message::ShardMsg;
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
}

struct PlanetState {
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

    // System context
    system_params: Option<SystemParams>,
    game_time: f64,
    tick_count: u64,
    planet_position_in_system: DVec3,
}

impl PlanetState {
    fn new(planet_seed: u64, planet_radius: f64, planet_mass: f64, system_seed: Option<u64>, planet_index: u32) -> Self {
        let surface_gravity = G * planet_mass / (planet_radius * planet_radius);

        let rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();

        let ground = ColliderBuilder::halfspace(nalgebra::Unit::new_normalize(vector![0.0, 1.0, 0.0]))
            .translation(vector![0.0, 0.0, 0.0])
            .build();
        let ground_handle = collider_set.insert(ground);

        let system_params = system_seed.map(SystemParams::from_seed);

        Self {
            planet_seed, planet_radius, planet_mass, surface_gravity,
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
            system_params,
            game_time: 0.0,
            tick_count: 0,
            planet_position_in_system: DVec3::ZERO,
        }
    }

    fn spawn_player(&mut self, session_token: SessionToken, name: String) {
        let player_rb = RigidBodyBuilder::dynamic()
            .translation(vector![0.0, 2.0, 0.0])
            .lock_rotations()
            .build();
        let handle = self.rigid_body_set.insert(player_rb);
        let player_collider = ColliderBuilder::capsule_y(0.6, 0.3).build();
        self.collider_set.insert_with_parent(player_collider, handle, &mut self.rigid_body_set);

        let spawn_pos = DVec3::new(0.0, self.planet_radius + 2.0, 0.0);
        self.players.insert(session_token, PlayerOnPlanet {
            session_token, player_name: name,
            body_handle: handle,
            position: spawn_pos,
            up: spawn_pos.normalize(),
            yaw: 0.0,
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
                self.planet_position_in_system = compute_planet_position(planet, self.game_time);
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
                let planet_sys_pos = compute_planet_position(planet, self.game_time);
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
            game_time: self.game_time,
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
            PlanetState::new(args.seed, planet_radius, planet_mass, system_seed, planet_index)
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
                let game_time = st.game_time;
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
                    .map(|p| { p.yaw = input.look_yaw; (p.body_handle, p.yaw) });

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

        // System 3: Drain QUIC — receive PlayerHandoff.
        let state_quic = state.clone();
        harness.add_system("drain_quic", move || {
            while let Ok(msg) = quic_msg_rx.try_recv() {
                match msg {
                    ShardMsg::PlayerHandoff(handoff) => {
                        let mut st = state_quic.lock().unwrap();
                        let planet_local = handoff.position - st.planet_position_in_system;
                        let surface_pos = planet_local.normalize() * (st.planet_radius + 2.0);

                        info!(
                            player = %handoff.player_name,
                            session = handoff.session_token.0,
                            "received player handoff from system shard"
                        );

                        st.spawn_player(handoff.session_token, handoff.player_name.clone());

                        // Send HandoffAccepted back.
                        let accepted = ShardMsg::HandoffAccepted(HandoffAccepted {
                            session_token: handoff.session_token,
                            target_shard: ShardId(0), // TODO: use actual shard ID
                        });
                        let source = handoff.source_shard;
                        // Would send via quic_send_tx but need source shard QUIC addr.
                        // For now, just log.
                        info!(source_shard = source.0, "handoff accepted (QUIC ack TODO)");
                    }
                    _ => {}
                }
            }
        });

        // System 4: Physics step.
        let state_physics = state.clone();
        harness.add_system("physics_step", move || {
            let mut st = state_physics.lock().unwrap();
            st.game_time += 0.05;
            st.tick_count += 1;
            st.compute_planet_system_position();
            st.step();
        });

        // System 5: Broadcast WorldState.
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
                    game_time = format!("{:.1}s", st.game_time),
                    players = st.players.len(),
                    surface_g = format!("{:.2} m/s²", st.surface_gravity),
                    "planet state"
                );
            }
        });

        harness.run().await;
    });
}
