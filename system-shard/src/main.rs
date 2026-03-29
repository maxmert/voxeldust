use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use clap::Parser;
use glam::{DQuat, DVec3};
use tracing::info;

use voxeldust_core::client_message::{
    CelestialBodyData, JoinResponseData, LightingData, ServerMsg, WorldStateData,
};
use voxeldust_core::shard_message::{
    CelestialBodySnapshotData, LightingInfoData, ShardMsg, ShipPositionUpdate,
    ShipSnapshotEntryData, SystemSceneUpdateData,
};
use voxeldust_core::shard_types::{ShardId, ShardType};
use voxeldust_core::system::{
    compute_gravity_acceleration, compute_lighting, compute_planet_position, compute_soi_radius,
    SystemParams,
};
use voxeldust_shard_common::client_listener;
use voxeldust_shard_common::harness::{ShardHarness, ShardHarnessConfig};

#[derive(Parser, Debug)]
#[command(name = "system-shard", about = "Voxeldust system shard — orbital mechanics")]
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
}

struct ShipState {
    ship_id: u64,
    position: DVec3,
    velocity: DVec3,
    rotation: DQuat,
    angular_velocity: DVec3,
    thrust: DVec3,
    torque: DVec3,
}

/// Maximum angular velocity (rad/s). ~30 deg/s — gameplay-appropriate for a small ship.
/// Star Citizen uses similar rates for medium fighters.
const MAX_ANGULAR_VELOCITY: f64 = 0.52;
/// Angular acceleration rate (rad/s²). Reaches max angular velocity in ~0.5 seconds.
const ANGULAR_ACCEL_RATE: f64 = 1.0;

struct SystemState {
    system_params: SystemParams,
    shard_id: ShardId,
    game_time: f64,
    planet_positions: Vec<DVec3>,
    ships: HashMap<u64, ShipState>,
    next_ship_id: u64,
    tick_count: u64,
}

impl SystemState {
    fn new(system_seed: u64, shard_id: ShardId) -> Self {
        let system_params = SystemParams::from_seed(system_seed);
        let planet_count = system_params.planets.len();
        Self {
            system_params,
            shard_id,
            game_time: 0.0,
            planet_positions: vec![DVec3::ZERO; planet_count],
            ships: HashMap::new(),
            next_ship_id: 1,
            tick_count: 0,
        }
    }

    fn spawn_ship(&mut self) -> u64 {
        let id = self.next_ship_id;
        self.next_ship_id += 1;
        let start_pos = if !self.planet_positions.is_empty() {
            self.planet_positions[0] + DVec3::new(1e8, 0.0, 0.0)
        } else {
            DVec3::new(1e11, 0.0, 0.0)
        };
        self.ships.insert(id, ShipState {
            ship_id: id,
            position: start_pos,
            velocity: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
            thrust: DVec3::ZERO,
            torque: DVec3::ZERO,
        });
        info!(ship_id = id, "spawned ship");
        id
    }

    fn build_body_snapshots(&self) -> Vec<CelestialBodySnapshotData> {
        let mut bodies = Vec::with_capacity(1 + self.system_params.planets.len());
        bodies.push(CelestialBodySnapshotData {
            body_id: 0,
            position: DVec3::ZERO,
            radius: self.system_params.star.radius_m,
            color: self.system_params.star.color,
        });
        for (i, planet) in self.system_params.planets.iter().enumerate() {
            bodies.push(CelestialBodySnapshotData {
                body_id: (i + 1) as u32,
                position: self.planet_positions[i],
                radius: planet.radius_m,
                color: planet.color,
            });
        }
        bodies
    }

    fn build_lighting(&self, observer_pos: DVec3) -> LightingInfoData {
        let l = compute_lighting(observer_pos, &self.system_params.star);
        LightingInfoData {
            sun_direction: l.sun_direction,
            sun_color: l.sun_color,
            sun_intensity: l.sun_intensity,
            ambient: l.ambient,
        }
    }

    fn build_world_state(&self) -> WorldStateData {
        let bodies: Vec<CelestialBodyData> = self.build_body_snapshots().iter().map(|b| {
            CelestialBodyData { body_id: b.body_id, position: b.position, radius: b.radius, color: b.color }
        }).collect();

        let observer_pos = self.ships.values().next()
            .map(|s| s.position).unwrap_or(DVec3::new(1e11, 0.0, 0.0));
        let lighting = compute_lighting(observer_pos, &self.system_params.star);

        WorldStateData {
            tick: self.tick_count, origin: DVec3::ZERO, players: vec![],
            bodies, ships: vec![],
            lighting: Some(LightingData {
                sun_direction: lighting.sun_direction, sun_color: lighting.sun_color,
                sun_intensity: lighting.sun_intensity, ambient: lighting.ambient,
            }),
            game_time: self.game_time,
        }
    }
}

const DT: f64 = 0.05;
const SHIP_MASS: f64 = 10_000.0;

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
        orchestrator_url: args.orchestrator,
        orchestrator_heartbeat_addr: args.orchestrator_heartbeat,
        healthz_addr: format!("{bind}:{}", args.healthz_port).parse().unwrap(),
        planet_seed: None,
        system_seed: Some(args.seed),
        ship_id: None,
        galaxy_seed: None,
        host_shard_id: None,
        advertise_host: args.advertise_host,
    };

    info!(shard_id = args.shard_id, system_seed = args.seed, "system shard starting");

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let mut harness = ShardHarness::new(config);

        let state = Arc::new(Mutex::new(SystemState::new(args.seed, shard_id)));
        let client_registry = harness.client_registry.clone();
        let system_seed = args.seed;

        // Take channels.
        let mut connect_rx = std::mem::replace(
            &mut harness.connect_rx, tokio::sync::mpsc::unbounded_channel().1,
        );
        let mut quic_msg_rx = std::mem::replace(
            &mut harness.quic_msg_rx, tokio::sync::mpsc::unbounded_channel().1,
        );
        let broadcast_tx = harness.broadcast_tx.clone();
        let quic_send_tx = harness.quic_send_tx.clone();
        let peer_registry = harness.peer_registry.clone();

        // System 1: Drain client connects (direct connections — debug mode).
        let state_connect = state.clone();
        let registry_connect = client_registry.clone();
        harness.add_system("drain_connects", move || {
            while let Ok(event) = connect_rx.try_recv() {
                let conn = event.connection;
                let mut st = state_connect.lock().unwrap();
                let ship_id = st.spawn_ship();
                let spawn_pos = st.ships[&ship_id].position;
                let token = conn.session_token;

                if let Ok(mut reg) = registry_connect.try_write() {
                    reg.register(&conn);
                }

                let tcp_stream = conn.tcp_stream.clone();
                let game_time = st.game_time;
                tokio::spawn(async move {
                    let jr = ServerMsg::JoinResponse(JoinResponseData {
                        seed: system_seed, planet_radius: 0, player_id: token.0,
                        spawn_position: spawn_pos, spawn_rotation: DQuat::IDENTITY,
                        spawn_forward: DVec3::NEG_Z, session_token: token,
                        shard_type: 1, galaxy_seed: 0, system_seed,
                        game_time, reference_position: DVec3::ZERO, reference_rotation: DQuat::IDENTITY,
                    });
                    let mut stream = tcp_stream.lock().await;
                    let _ = client_listener::send_tcp_msg(&mut stream, &jr).await;
                });

                info!(player = %conn.player_name, session = token.0, ship_id, "player joined system");
            }
        });

        // System 2: Drain QUIC messages — ShipControlInput from ship shards.
        let state_quic = state.clone();
        harness.add_system("drain_quic", move || {
            while let Ok(msg) = quic_msg_rx.try_recv() {
                match msg {
                    ShardMsg::ShipControlInput(ctrl) => {
                        let mut st = state_quic.lock().unwrap();
                        if let Some(ship) = st.ships.get_mut(&ctrl.ship_id) {
                            ship.thrust = ctrl.thrust;
                            ship.torque = ctrl.torque;
                        } else {
                            info!(ship_id = ctrl.ship_id, "received ShipControlInput for unknown ship");
                        }
                    }
                    other => {
                        info!("received QUIC message: {:?}", std::mem::discriminant(&other));
                    }
                }
            }
        });

        // System 2b: Discover ship shards from peer registry and create ship entities.
        // The peer registry is refreshed every 10s from the orchestrator.
        // When a new ship shard appears with host_shard_id matching this system shard,
        // we create a ship entity in the physics world so it can be simulated.
        let state_discover = state.clone();
        let peer_reg_discover = peer_registry.clone();
        harness.add_system("discover_ships", move || {
            let mut st = state_discover.lock().unwrap();
            // Only run every 2 seconds (40 ticks).
            if st.tick_count % 40 != 0 { return; }

            if let Ok(reg) = peer_reg_discover.try_read() {
                for shard_info in reg.find_by_type(ShardType::Ship) {
                    // Only ships hosted by this system shard.
                    if shard_info.host_shard_id != Some(st.shard_id) { continue; }

                    // Check if we already track this ship.
                    if let Some(ship_id) = shard_info.ship_id {
                        if st.ships.contains_key(&ship_id) { continue; }

                        // New ship — create entity at default position.
                        let start_pos = if !st.planet_positions.is_empty() {
                            st.planet_positions[0] + DVec3::new(1e8, 0.0, 0.0)
                        } else {
                            DVec3::new(1e11, 0.0, 0.0)
                        };

                        st.ships.insert(ship_id, ShipState {
                            ship_id,
                            position: start_pos,
                            velocity: DVec3::ZERO,
                            rotation: DQuat::IDENTITY,
                            angular_velocity: DVec3::ZERO,
                            thrust: DVec3::ZERO,
                            torque: DVec3::ZERO,
                        });

                        info!(
                            ship_id,
                            shard_id = shard_info.id.0,
                            "discovered new ship shard — created ship entity"
                        );
                    }
                }
            }
        });

        // System 3: Physics — orbits + ship gravity + integration.
        let state_physics = state.clone();
        harness.add_system("physics", move || {
            let mut st = state_physics.lock().unwrap();
            st.game_time += DT;
            st.tick_count += 1;

            let game_time = st.game_time;
            let positions: Vec<DVec3> = st.system_params.planets.iter()
                .map(|p| compute_planet_position(p, game_time))
                .collect();
            st.planet_positions = positions;

            let accels: Vec<(u64, DVec3)> = st.ships.iter().map(|(&id, ship)| {
                let grav = compute_gravity_acceleration(
                    ship.position, &st.system_params.star,
                    &st.system_params.planets, &st.planet_positions, st.game_time,
                );
                let thrust_world = ship.rotation * ship.thrust / SHIP_MASS;
                (id, grav + thrust_world)
            }).collect();

            for (ship_id, accel) in accels {
                let ship = st.ships.get_mut(&ship_id).unwrap();

                // Linear integration (Velocity Verlet).
                ship.position += ship.velocity * DT + 0.5 * accel * DT * DT;
                ship.velocity += accel * DT;
                ship.thrust = DVec3::ZERO;

                // Angular control: torque field carries the desired yaw/pitch rate (-1 to 1).
                // Convert to target angular velocity, accelerate toward it.
                let target_angular_vel = DVec3::new(
                    ship.torque.x.clamp(-1.0, 1.0) * MAX_ANGULAR_VELOCITY,
                    ship.torque.y.clamp(-1.0, 1.0) * MAX_ANGULAR_VELOCITY,
                    ship.torque.z.clamp(-1.0, 1.0) * MAX_ANGULAR_VELOCITY,
                );

                // Accelerate toward target (proportional control).
                let diff = target_angular_vel - ship.angular_velocity;
                let accel = diff.clamp_length_max(ANGULAR_ACCEL_RATE * DT);
                ship.angular_velocity += accel;

                // When no input, decelerate (reaction wheel damping).
                if ship.torque.length_squared() < 0.001 {
                    ship.angular_velocity *= 0.95;
                    if ship.angular_velocity.length_squared() < 1e-6 {
                        ship.angular_velocity = DVec3::ZERO;
                    }
                }

                // Apply angular velocity to rotation quaternion.
                let ang_speed = ship.angular_velocity.length();
                if ang_speed > 1e-8 {
                    let axis = ship.angular_velocity / ang_speed;
                    let delta_rot = DQuat::from_axis_angle(axis, ang_speed * DT);
                    ship.rotation = (delta_rot * ship.rotation).normalize();
                }

                ship.torque = DVec3::ZERO;
            }
        });

        // System 4: SOI detection.
        let state_soi = state.clone();
        harness.add_system("soi_detection", move || {
            let st = state_soi.lock().unwrap();
            if st.tick_count % 20 != 0 { return; } // check every second

            for ship in st.ships.values() {
                for (i, planet) in st.system_params.planets.iter().enumerate() {
                    let dist = (ship.position - st.planet_positions[i]).length();
                    let soi = compute_soi_radius(planet, &st.system_params.star);
                    if dist < soi {
                        info!(
                            ship_id = ship.ship_id,
                            planet_index = i,
                            distance = format!("{:.2e}", dist),
                            soi_radius = format!("{:.2e}", soi),
                            "ship entered planet SOI — handoff needed"
                        );
                        // TODO: provision planet shard, send PlayerHandoff
                    }
                }
            }
        });

        // System 5: Broadcast SystemSceneUpdate to ship shards via QUIC.
        let state_scene = state.clone();
        let peer_reg_scene = peer_registry.clone();
        let quic_send_scene = quic_send_tx.clone();
        harness.add_system("broadcast_scene", move || {
            let st = state_scene.lock().unwrap();

            // Find ship shards from peer registry.
            let ship_shards: Vec<(ShardId, SocketAddr)> = if let Ok(reg) = peer_reg_scene.try_read() {
                reg.find_by_type(ShardType::Ship).iter().filter_map(|info| {
                    // Only send to ships hosted by this system shard.
                    if info.host_shard_id == Some(st.shard_id) {
                        Some((info.id, info.endpoint.quic_addr))
                    } else {
                        None
                    }
                }).collect()
            } else {
                vec![]
            };

            if ship_shards.is_empty() { return; }

            // Build scene update.
            let bodies = st.build_body_snapshots();
            let observer_pos = st.ships.values().next()
                .map(|s| s.position).unwrap_or(DVec3::new(1e11, 0.0, 0.0));
            let lighting = st.build_lighting(observer_pos);

            let scene = SystemSceneUpdateData {
                game_time: st.game_time,
                bodies,
                ships: vec![], // TODO: include other ships
                lighting,
            };

            let scene_msg = ShardMsg::SystemSceneUpdate(scene);

            for (shard_id, quic_addr) in &ship_shards {
                let _ = quic_send_scene.send((*shard_id, *quic_addr, scene_msg.clone()));
            }

            // Also send per-ship position updates.
            for ship in st.ships.values() {
                // Find which ship shard manages this ship (by ship_id match in ShardInfo).
                for (shard_id, quic_addr) in &ship_shards {
                    let pos_msg = ShardMsg::ShipPositionUpdate(ShipPositionUpdate {
                        ship_id: ship.ship_id,
                        position: ship.position,
                        velocity: ship.velocity,
                        rotation: ship.rotation,
                        angular_velocity: DVec3::ZERO,
                    });
                    let _ = quic_send_scene.send((*shard_id, *quic_addr, pos_msg));
                }
            }
        });

        // System 6: Broadcast WorldState to direct UDP clients (debug mode).
        let state_broadcast = state.clone();
        harness.add_system("broadcast_udp", move || {
            let st = state_broadcast.lock().unwrap();
            let ws = ServerMsg::WorldState(st.build_world_state());
            let _ = broadcast_tx.send(ws);
        });

        // System 7: Periodic log.
        let state_log = state.clone();
        harness.add_system("log_state", move || {
            let st = state_log.lock().unwrap();
            if st.tick_count % 100 == 0 && st.tick_count > 0 {
                info!(
                    game_time = format!("{:.1}s", st.game_time),
                    ships = st.ships.len(),
                    tick = st.tick_count,
                    "system state"
                );
            }
        });

        harness.run().await;
    });
}
