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
use voxeldust_core::autopilot::{self, FlightPhase, GuidanceCommand};
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

struct AutopilotState {
    target_planet_index: usize,
    thrust_tier: u8,
    intercept_pos: DVec3,
    last_solve_tick: u64,
}

struct ShipState {
    ship_id: u64,
    position: DVec3,
    velocity: DVec3,
    rotation: DQuat,
    angular_velocity: DVec3,
    thrust: DVec3,
    torque: DVec3,
    autopilot: Option<AutopilotState>,
}

/// Maximum angular velocity (rad/s). ~30 deg/s — gameplay-appropriate for a small ship.
/// Star Citizen uses similar rates for medium fighters.
const MAX_ANGULAR_VELOCITY: f64 = 0.52;
/// Angular acceleration rate (rad/s²). Reaches max angular velocity in ~0.5 seconds.
const ANGULAR_ACCEL_RATE: f64 = 1.0;

struct SystemState {
    system_params: SystemParams,
    shard_id: ShardId,
    physics_time: f64,
    celestial_time: f64,
    planet_positions: Vec<DVec3>,
    ships: HashMap<u64, ShipState>,
    next_ship_id: u64,
    tick_count: u64,
}

impl SystemState {
    fn new(system_seed: u64, shard_id: ShardId) -> Self {
        let system_params = SystemParams::from_seed(system_seed);
        // Compute initial planet positions at t=0 so ships spawned before
        // the first physics tick get placed near actual planets, not the star.
        let planet_positions: Vec<DVec3> = system_params.planets.iter()
            .map(|p| compute_planet_position(p, 0.0))
            .collect();
        Self {
            system_params,
            shard_id,
            physics_time: 0.0,
            celestial_time: 0.0,
            planet_positions,
            ships: HashMap::new(),
            next_ship_id: 1,
            tick_count: 0,
        }
    }

    fn spawn_ship(&mut self) -> u64 {
        let id = self.next_ship_id;
        self.next_ship_id += 1;
        let start_pos = if !self.planet_positions.is_empty() {
            self.planet_positions[0] + DVec3::new(self.system_params.scale.spawn_offset, 0.0, 0.0)
        } else {
            DVec3::new(self.system_params.scale.fallback_spawn_distance, 0.0, 0.0)
        };
        self.ships.insert(id, ShipState {
            ship_id: id,
            position: start_pos,
            velocity: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
            thrust: DVec3::ZERO,
            torque: DVec3::ZERO,
            autopilot: None,
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
            game_time: self.celestial_time,
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
                let game_time = st.celestial_time;
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
                    ShardMsg::AutopilotCommand(cmd) => {
                        let mut st = state_quic.lock().unwrap();
                        if cmd.target_body_id == 0xFFFFFFFF {
                            if let Some(ship) = st.ships.get_mut(&cmd.ship_id) {
                                ship.autopilot = None;
                                info!(ship_id = cmd.ship_id, "autopilot disengaged");
                            }
                        } else {
                            let planet_index = (cmd.target_body_id - 1) as usize;
                            // Read ship state and system params before mutating.
                            let solve_result = st.ships.get(&cmd.ship_id).and_then(|ship| {
                                if planet_index >= st.system_params.planets.len() { return None; }
                                let intercept = autopilot::solve_intercept(
                                    ship.position, ship.velocity,
                                    &st.system_params.planets[planet_index],
                                    &st.system_params.star,
                                    st.celestial_time,
                                    st.system_params.scale.time_scale,
                                    cmd.speed_tier,
                                ).map(|(pos, _)| pos)
                                    .unwrap_or(st.planet_positions[planet_index]);
                                Some(intercept)
                            });
                            if let Some(intercept_pos) = solve_result {
                                let tick = st.tick_count;
                                if let Some(ship) = st.ships.get_mut(&cmd.ship_id) {
                                    ship.autopilot = Some(AutopilotState {
                                        target_planet_index: planet_index,
                                        thrust_tier: cmd.speed_tier,
                                        intercept_pos,
                                        last_solve_tick: tick,
                                    });
                                    info!(ship_id = cmd.ship_id, planet = planet_index,
                                        tier = cmd.speed_tier, "autopilot engaged");
                                }
                            }
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
                            st.planet_positions[0] + DVec3::new(st.system_params.scale.spawn_offset, 0.0, 0.0)
                        } else {
                            DVec3::new(st.system_params.scale.fallback_spawn_distance, 0.0, 0.0)
                        };

                        st.ships.insert(ship_id, ShipState {
                            ship_id,
                            position: start_pos,
                            velocity: DVec3::ZERO,
                            rotation: DQuat::IDENTITY,
                            angular_velocity: DVec3::ZERO,
                            thrust: DVec3::ZERO,
                            torque: DVec3::ZERO,
                            autopilot: None,
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
            st.physics_time += DT;
            st.celestial_time += DT * st.system_params.scale.time_scale;
            st.tick_count += 1;

            let celestial_time = st.celestial_time;
            let positions: Vec<DVec3> = st.system_params.planets.iter()
                .map(|p| compute_planet_position(p, celestial_time))
                .collect();
            st.planet_positions = positions;

            // Destructure to satisfy borrow checker — ships, system_params,
            // planet_positions, and celestial_time are disjoint fields.
            let sys = &st.system_params;
            let pp = &st.planet_positions;
            let ct = st.celestial_time;
            let tick = st.tick_count;

            // Re-solve autopilot intercepts every 20 ticks (~1 second).
            let ap_updates: Vec<(u64, DVec3)> = st.ships.iter().filter_map(|(&id, ship)| {
                if let Some(ref ap) = ship.autopilot {
                    if tick - ap.last_solve_tick >= 20 {
                        if let Some((new_intercept, _)) = autopilot::solve_intercept(
                            ship.position, ship.velocity,
                            &sys.planets[ap.target_planet_index],
                            &sys.star, ct, sys.scale.time_scale, ap.thrust_tier,
                        ) {
                            return Some((id, new_intercept));
                        }
                    }
                }
                None
            }).collect();

            for (sid, new_intercept) in ap_updates {
                if let Some(ship) = st.ships.get_mut(&sid) {
                    if let Some(ref mut ap) = ship.autopilot {
                        ap.intercept_pos = new_intercept;
                        ap.last_solve_tick = tick;
                    }
                }
            }

            // Phase 1: Autopilot controller — sets ship.thrust and ship.torque
            // through the same thruster system as manual piloting.
            // Guidance computes desired direction; autopilot rotates ship to face it,
            // then fires the main drive along the ship's forward axis (-Z).
            {
            let sys = &st.system_params;
            let pp = &st.planet_positions;
            let ct = st.celestial_time;
            let ap_commands: Vec<(u64, GuidanceCommand)> = st.ships.iter()
                .filter_map(|(&id, ship)| {
                    let ap = ship.autopilot.as_ref()?;
                    let guidance = autopilot::compute_guidance(
                        ship.position, ship.velocity,
                        ap.intercept_pos, ap.target_planet_index,
                        sys, pp, ct, ap.thrust_tier,
                    );
                    Some((id, guidance))
                }).collect();

            for (ship_id, guidance) in &ap_commands {
                let ship = st.ships.get_mut(ship_id).unwrap();

                if guidance.completed {
                    info!(ship_id, "autopilot arrived at SOI — disengaging");
                    ship.autopilot = None;
                    ship.thrust = DVec3::ZERO;
                    ship.torque = DVec3::ZERO;
                    continue;
                }

                // Rotation: PD controller to align ship forward (-Z) with thrust direction.
                let desired_fwd = guidance.thrust_direction;
                let current_fwd = ship.rotation * DVec3::NEG_Z;
                let cross = current_fwd.cross(desired_fwd);
                let dot_align = current_fwd.dot(desired_fwd).clamp(-1.0, 1.0);
                let angle = dot_align.acos();

                if angle > 0.001 && cross.length() > 1e-8 {
                    let axis = cross.normalize();
                    let p_gain = 2.0;
                    let target_omega = axis * (angle * p_gain).min(MAX_ANGULAR_VELOCITY);
                    let world_omega = ship.rotation * ship.angular_velocity;
                    let error = target_omega - world_omega;
                    let local_error = ship.rotation.inverse() * error;
                    ship.torque = local_error.clamp_length_max(1.0);
                } else {
                    ship.torque = DVec3::ZERO;
                }

                // Thrust: main drive fires along ship's -Z axis.
                // Scale by alignment — only full thrust when facing the right direction.
                // cos(misalignment) gives the useful thrust component along desired direction.
                // Below 0 (>90° off) = no thrust, ship is still rotating.
                let thrust_scale = dot_align.max(0.0);
                ship.thrust = DVec3::new(0.0, 0.0, -guidance.thrust_magnitude * thrust_scale);
            }

            } // end autopilot controller block

            // Phase 2: True Velocity Verlet (Störmer-Verlet) integration.
            // Two-pass for symplectic energy conservation — critical for orbital stability.
            //   Pass A: a_old at current position → advance position
            //   Pass B: a_new at new position → advance velocity with average
            let sys = &st.system_params;
            let pp = &st.planet_positions;
            let ct = st.celestial_time;

            // Pass A: compute a_old, store thrust_accel, advance position.
            let pass_a: Vec<(u64, DVec3, DVec3)> = {
                let sys = &st.system_params;
                let pp = &st.planet_positions;
                let ct = st.celestial_time;
                st.ships.iter().map(|(&id, ship)| {
                    let grav = compute_gravity_acceleration(
                        ship.position, &sys.star, &sys.planets, pp, ct,
                    );
                    let thrust_accel = ship.rotation * ship.thrust / SHIP_MASS;
                    let accel_old = grav + thrust_accel;
                    (id, accel_old, thrust_accel)
                }).collect()
            };

            for &(ship_id, accel_old, _) in &pass_a {
                let ship = st.ships.get_mut(&ship_id).unwrap();
                ship.position += ship.velocity * DT + 0.5 * accel_old * DT * DT;
            }

            // Pass B: recompute gravity at new position, average with a_old for velocity.
            // Collect new positions first (immutable borrow), then mutate.
            let pass_b: Vec<(u64, DVec3)> = {
                let sys = &st.system_params;
                let pp = &st.planet_positions;
                let ct = st.celestial_time;
                pass_a.iter().map(|&(ship_id, accel_old, thrust_accel)| {
                    let ship = &st.ships[&ship_id];
                    let grav_new = compute_gravity_acceleration(
                        ship.position, &sys.star, &sys.planets, pp, ct,
                    );
                    let accel_new = grav_new + thrust_accel;
                    let vel_delta = 0.5 * (accel_old + accel_new) * DT;
                    (ship_id, vel_delta)
                }).collect()
            };
            for (ship_id, vel_delta) in pass_b {
                let ship = st.ships.get_mut(&ship_id).unwrap();
                ship.velocity += vel_delta;
                ship.thrust = DVec3::ZERO;

                // Standard angular velocity control (works for both manual and autopilot torque).
                let target_angular_vel = DVec3::new(
                    ship.torque.x.clamp(-1.0, 1.0) * MAX_ANGULAR_VELOCITY,
                    ship.torque.y.clamp(-1.0, 1.0) * MAX_ANGULAR_VELOCITY,
                    ship.torque.z.clamp(-1.0, 1.0) * MAX_ANGULAR_VELOCITY,
                );

                let diff = target_angular_vel - ship.angular_velocity;
                let ang_accel = diff.clamp_length_max(ANGULAR_ACCEL_RATE * DT);
                ship.angular_velocity += ang_accel;

                if ship.torque.length_squared() < 0.001 {
                    ship.angular_velocity *= 0.95;
                    if ship.angular_velocity.length_squared() < 1e-6 {
                        ship.angular_velocity = DVec3::ZERO;
                    }
                }

                let world_angular_vel = ship.rotation * ship.angular_velocity;
                let ang_speed = world_angular_vel.length();
                if ang_speed > 1e-8 {
                    let axis = world_angular_vel / ang_speed;
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
                game_time: st.celestial_time,
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
                    physics_time = format!("{:.1}s", st.physics_time),
                    celestial_time = format!("{:.1}s", st.celestial_time),
                    ships = st.ships.len(),
                    tick = st.tick_count,
                    "system state"
                );
            }
        });

        harness.run().await;
    });
}
