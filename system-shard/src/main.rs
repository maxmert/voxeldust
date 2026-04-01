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
    CelestialBodySnapshotData, LightingInfoData, ShardMsg, ShipNearbyInfoData,
    ShipPositionUpdate, ShipSnapshotEntryData, SystemSceneUpdateData,
};
use voxeldust_core::shard_types::{SessionToken, ShardId, ShardType};
use voxeldust_core::system::{
    compute_atmospheric_drag, compute_full_aerodynamics, compute_gravity_acceleration,
    compute_lighting, compute_planet_position, compute_planet_rotation, compute_planet_velocity,
    compute_soi_radius, check_atmosphere, SystemParams,
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

#[derive(Clone)]
struct AutopilotState {
    mode: autopilot::AutopilotMode,
    phase: autopilot::FlightPhase,
    target_planet_index: usize,
    thrust_tier: u8,
    intercept_pos: DVec3,
    last_solve_tick: u64,
    /// Physics time when autopilot was engaged (for thrust ramp).
    engage_time: f64,
    /// Estimated total time of flight (real seconds, updated on re-solve).
    estimated_tof: f64,
    /// Once true, ship stays in Brake phase until velocity reverses or autopilot disengages.
    braking_committed: bool,
    /// Target orbit altitude (meters above surface). 0 = default.
    target_orbit_altitude: f64,
}

#[derive(Clone)]
struct ShipState {
    ship_id: u64,
    position: DVec3,
    velocity: DVec3,
    rotation: DQuat,
    angular_velocity: DVec3,
    thrust: DVec3,
    torque: DVec3,
    autopilot: Option<AutopilotState>,
    /// Per-ship physical properties (mass, drag, cross-section).
    physical_properties: autopilot::ShipPhysicalProperties,
    /// Accumulated thermal energy from re-entry heating (Joules).
    thermal_energy: f64,
    /// Whether the ship is on a planet surface.
    landed: bool,
    /// Planet index the ship is landed on (valid only when landed=true).
    landed_planet_index: Option<usize>,
    /// Radial direction from planet center at landing time (normalized, planet-local).
    /// Used with planet rotation to derive position each tick. Only valid when landed.
    landed_surface_radial: DVec3,
    /// Celestial time when landing occurred. Used to compute rotation delta.
    landed_celestial_time: f64,
    /// Consecutive ticks the ship has been in the landing zone (debounce for landing detection).
    landing_zone_ticks: u32,
}

/// Local workspace for the physics system. Holds mutable data extracted from
/// `SystemState` so the physics computation runs without holding the Mutex.
/// This eliminates lock contention with async tasks and keeps the tick responsive.
struct PhysicsWorkspace<'a> {
    system_params: &'a SystemParams,
    ships: HashMap<u64, ShipState>,
    planet_positions: Vec<DVec3>,
    planet_velocities: Vec<DVec3>,
    in_soi: HashMap<u64, usize>,
    celestial_time: f64,
    physics_time: f64,
    tick_count: u64,
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
    /// Orbital velocities of planets in system-space (m/s). Used for patched conics SOI transitions.
    planet_velocities: Vec<DVec3>,
    ships: HashMap<u64, ShipState>,
    next_ship_id: u64,
    tick_count: u64,
    /// Ships currently inside a planet's SOI: ship_id → planet_index.
    in_soi: HashMap<u64, usize>,
    /// Planet shards provisioned by this system shard: planet_seed → shard_id.
    provisioned_planets: HashMap<u64, ShardId>,
    /// Planet seeds currently being provisioned (async in-flight, not yet ready).
    provisioning_in_flight: std::collections::HashSet<u64>,
    /// Pending handoffs being relayed: session_token → source shard id.
    pending_handoffs: HashMap<SessionToken, ShardId>,
}

impl SystemState {
    fn new(system_seed: u64, shard_id: ShardId) -> Self {
        let system_params = SystemParams::from_seed(system_seed);
        // Compute initial planet positions at t=0 so ships spawned before
        // the first physics tick get placed near actual planets, not the star.
        let planet_positions: Vec<DVec3> = system_params.planets.iter()
            .map(|p| compute_planet_position(p, 0.0))
            .collect();
        let star_gm = system_params.star.gm;
        let planet_velocities: Vec<DVec3> = system_params.planets.iter()
            .map(|p| compute_planet_velocity(p, star_gm, 0.0))
            .collect();
        Self {
            system_params,
            shard_id,
            physics_time: 0.0,
            celestial_time: 0.0,
            planet_positions,
            planet_velocities,
            ships: HashMap::new(),
            next_ship_id: 1,
            tick_count: 0,
            in_soi: HashMap::new(),
            provisioned_planets: HashMap::new(),
            provisioning_in_flight: std::collections::HashSet::new(),
            pending_handoffs: HashMap::new(),
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
            physical_properties: autopilot::ShipPhysicalProperties::starter_ship(),
            thermal_energy: 0.0,
            landed: false,
            landed_planet_index: None,
            landed_surface_radial: DVec3::ZERO,
            landed_celestial_time: 0.0,
            landing_zone_ticks: 0,
        });
        // Pre-register in SOI if spawning inside one.
        // Velocity is already zero (planet-relative) — skip the patched conics entry impulse.
        for (i, planet) in self.system_params.planets.iter().enumerate() {
            let dist = (start_pos - self.planet_positions[i]).length();
            let soi = compute_soi_radius(planet, &self.system_params.star);
            if dist < soi {
                self.in_soi.insert(id, i);
                info!(ship_id = id, planet_index = i, "spawned inside SOI — pre-registered");
                break;
            }
        }
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

fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let shard_id = ShardId(args.shard_id);
    let orchestrator_url = args.orchestrator.clone();
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
        // Immutable system params shared with physics (avoids cloning every tick).
        let sys_params_arc = Arc::new(SystemParams::from_seed(args.seed));
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

        // Channel for async planet provisioning results (avoids locking the main
        // state mutex from spawned tokio tasks, which caused tick-loop stalls).
        let (provision_tx, mut provision_rx) =
            tokio::sync::mpsc::channel::<(u64, ShardId)>(32);

        // Shared HTTP client with a timeout so slow/unreachable orchestrator
        // responses don't block tokio worker threads indefinitely.
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("failed to build HTTP client");

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

        // System 2: Drain QUIC messages — ShipControlInput, AutopilotCommand, PlayerHandoff, HandoffAccepted.
        let state_quic = state.clone();
        let peer_reg_quic = peer_registry.clone();
        let quic_send_quic = quic_send_tx.clone();
        harness.add_system("drain_quic", move || {
            while let Ok(msg) = quic_msg_rx.try_recv() {
                match msg {
                    ShardMsg::ShipControlInput(ctrl) => {
                        let mut st = state_quic.lock().unwrap();
                        // Pre-extract gravity for takeoff check (avoids borrow conflict).
                        let takeoff_gravity = st.ships.get(&ctrl.ship_id)
                            .and_then(|s| s.landed_planet_index)
                            .and_then(|pi| st.system_params.planets.get(pi).map(|p| p.surface_gravity));
                        if let Some(ship) = st.ships.get_mut(&ctrl.ship_id) {
                            // Takeoff: if landed and receiving thrust that exceeds gravity, detach.
                            if ship.landed && ctrl.thrust.length_squared() > 0.01 {
                                let thrust_accel = ctrl.thrust.length() / ship.physical_properties.mass_kg;
                                let gravity = takeoff_gravity.unwrap_or(10.0);
                                if thrust_accel > gravity * 1.1 {
                                    ship.landed = false;
                                    ship.landed_planet_index = None;
                                    info!(ship_id = ctrl.ship_id, "ship taking off — detached from surface");
                                }
                            }
                            ship.thrust = ctrl.thrust;
                            ship.torque = ctrl.torque;
                            if st.tick_count % 100 == 0 {
                                info!(ship_id = ctrl.ship_id,
                                    thrust = format!("({:.0},{:.0},{:.0})", ctrl.thrust.x, ctrl.thrust.y, ctrl.thrust.z),
                                    torque = format!("({:.2},{:.2},{:.2})", ctrl.torque.x, ctrl.torque.y, ctrl.torque.z),
                                    "ShipControlInput applied");
                            }
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
                                autopilot::solve_intercept(
                                    ship.position, ship.velocity,
                                    &st.system_params.planets[planet_index],
                                    &st.system_params.star,
                                    st.celestial_time,
                                    st.system_params.scale.time_scale,
                                    cmd.speed_tier,
                                )
                            });
                            if let Some((intercept_pos, estimated_tof)) = solve_result {
                                let tick = st.tick_count;
                                let physics_time = st.physics_time;
                                let mode = autopilot::AutopilotMode::from_u8(cmd.autopilot_mode);
                                let planet = &st.system_params.planets[planet_index];
                                // Default orbit altitude: 1% of planet radius, clamped.
                                let default_orbit_alt = (planet.radius_m * 0.01)
                                    .clamp(planet.atmosphere.atmosphere_height * 1.5, planet.radius_m * 0.1);
                                if let Some(ship) = st.ships.get_mut(&cmd.ship_id) {
                                    ship.autopilot = Some(AutopilotState {
                                        mode,
                                        phase: FlightPhase::Accelerate,
                                        target_planet_index: planet_index,
                                        thrust_tier: cmd.speed_tier,
                                        intercept_pos,
                                        last_solve_tick: tick,
                                        engage_time: physics_time,
                                        estimated_tof,
                                        braking_committed: false,
                                        target_orbit_altitude: default_orbit_alt,
                                    });
                                    info!(ship_id = cmd.ship_id, planet = planet_index,
                                        tier = cmd.speed_tier, mode = ?mode,
                                        eta_s = estimated_tof as u64,
                                        "autopilot engaged");
                                }
                            }
                        }
                    }
                    ShardMsg::PlayerHandoff(handoff) => {
                        // Forward handoff to target shard (ship→planet or planet→ship).
                        let mut st = state_quic.lock().unwrap();
                        let source = handoff.source_shard;
                        let session = handoff.session_token;
                        st.pending_handoffs.insert(session, source);

                        if let Some(planet_seed) = handoff.target_planet_seed {
                            // Ship→Planet handoff: forward to planet shard.
                            if let Some(&planet_shard_id) = st.provisioned_planets.get(&planet_seed) {
                                drop(st);
                                if let Ok(reg) = peer_reg_quic.try_read() {
                                    if let Some(addr) = reg.quic_addr(planet_shard_id) {
                                        let msg = ShardMsg::PlayerHandoff(handoff);
                                        let _ = quic_send_quic.try_send((planet_shard_id, addr, msg));
                                        info!(planet_seed, target = planet_shard_id.0,
                                            "forwarded player handoff to planet shard");
                                    }
                                }
                            } else {
                                tracing::warn!(planet_seed, "no provisioned planet shard for handoff");
                            }
                        } else if let Some(ship_id) = handoff.target_ship_id {
                            // Planet→Ship handoff: forward to ship shard.
                            drop(st);
                            if let Ok(reg) = peer_reg_quic.try_read() {
                                let ship_shard = reg.find_by_type(ShardType::Ship).iter()
                                    .find(|s| s.ship_id == Some(ship_id))
                                    .map(|s| (s.id, s.endpoint.quic_addr));
                                if let Some((sid, addr)) = ship_shard {
                                    let msg = ShardMsg::PlayerHandoff(handoff);
                                    let _ = quic_send_quic.try_send((sid, addr, msg));
                                    info!(ship_id, target = sid.0,
                                        "forwarded player handoff to ship shard");
                                }
                            }
                        } else {
                            tracing::warn!("received PlayerHandoff with no target");
                        }
                    }
                    ShardMsg::HandoffAccepted(accepted) => {
                        // Relay HandoffAccepted back to the source shard.
                        let mut st = state_quic.lock().unwrap();
                        if let Some(source_shard) = st.pending_handoffs.remove(&accepted.session_token) {
                            drop(st);
                            if let Ok(reg) = peer_reg_quic.try_read() {
                                if let Some(addr) = reg.quic_addr(source_shard) {
                                    let msg = ShardMsg::HandoffAccepted(accepted);
                                    let _ = quic_send_quic.try_send((source_shard, addr, msg));
                                    info!(target = source_shard.0,
                                        "relayed HandoffAccepted to source shard");
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
                            physical_properties: autopilot::ShipPhysicalProperties::starter_ship(),
                            thermal_energy: 0.0,
                            landed: false,
                            landed_planet_index: None,
                            landed_surface_radial: DVec3::ZERO,
                            landed_celestial_time: 0.0,
                            landing_zone_ticks: 0,
                        });

                        // Pre-register in SOI if spawning inside one.
                        for (i, planet) in st.system_params.planets.iter().enumerate() {
                            let dist = (start_pos - st.planet_positions[i]).length();
                            let soi = compute_soi_radius(planet, &st.system_params.star);
                            if dist < soi {
                                st.in_soi.insert(ship_id, i);
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
                let active_ship_ids: std::collections::HashSet<u64> = reg.find_by_type(ShardType::Ship)
                    .iter()
                    .filter(|s| s.host_shard_id == Some(st.shard_id))
                    .filter_map(|s| s.ship_id)
                    .collect();
                let stale: Vec<u64> = st.ships.keys()
                    .filter(|id| !active_ship_ids.contains(id))
                    .copied()
                    .collect();
                for id in stale {
                    st.ships.remove(&id);
                    st.in_soi.remove(&id);
                    info!(ship_id = id, "removed stale ship entity");
                }
            }
        });

        // System 3: Physics — orbits + ship gravity + integration.
        // Uses extract → compute → writeback to minimize mutex hold time.
        let state_physics = state.clone();
        let sys_params_physics = sys_params_arc.clone();
        harness.add_system("physics", move || {
            // Phase A: Extract mutable state (lock held for microseconds).
            let mut st = {
                let mut locked = state_physics.lock().unwrap();
                locked.physics_time += DT;
                locked.celestial_time += DT * locked.system_params.scale.time_scale;
                locked.tick_count += 1;
                PhysicsWorkspace {
                    system_params: &*sys_params_physics,
                    ships: std::mem::take(&mut locked.ships),
                    planet_positions: std::mem::take(&mut locked.planet_positions),
                    planet_velocities: std::mem::take(&mut locked.planet_velocities),
                    in_soi: locked.in_soi.clone(),
                    celestial_time: locked.celestial_time,
                    physics_time: locked.physics_time,
                    tick_count: locked.tick_count,
                }
            }; // Lock released — all physics runs without holding the mutex.

            let celestial_time = st.celestial_time;
            let time_scale = st.system_params.scale.time_scale;
            let old_planet_positions = st.planet_positions.clone();
            let positions: Vec<DVec3> = st.system_params.planets.iter()
                .map(|p| compute_planet_position(p, celestial_time))
                .collect();
            let star_gm = st.system_params.star.gm;
            // Scale to real-time velocity (m/s per real second).
            let velocities: Vec<DVec3> = st.system_params.planets.iter()
                .map(|p| compute_planet_velocity(p, star_gm, celestial_time) * time_scale)
                .collect();
            st.planet_positions = positions;
            st.planet_velocities = velocities;

            // Pre-compute landed ship position updates (avoids borrow conflicts).
            let soi_ships: Vec<(u64, usize)> = st.in_soi.iter()
                .map(|(&sid, &pi)| (sid, pi)).collect();
            let celestial_time = st.celestial_time;

            // Compute landed ship derived positions.
            let landed_updates: Vec<(u64, DVec3)> = soi_ships.iter().filter_map(|(ship_id, planet_index)| {
                let ship = st.ships.get(ship_id)?;
                if !ship.landed { return None; }
                let planet = st.system_params.planets.get(*planet_index)?;
                let planet_pos = st.planet_positions[*planet_index];
                let contact_margin = ship.physical_properties.landing_gear_height + 0.5;
                let landing_rot = compute_planet_rotation(planet, ship.landed_celestial_time);
                let current_rot = compute_planet_rotation(planet, celestial_time);
                let delta_rot = current_rot - landing_rot;
                let spin_quat = DQuat::from_axis_angle(DVec3::Y, delta_rot);
                let current_radial = (spin_quat * ship.landed_surface_radial).normalize();
                let new_pos = planet_pos + current_radial * (planet.radius_m + contact_margin);
                Some((*ship_id, new_pos))
            }).collect();

            // Apply landed ship positions.
            for (ship_id, new_pos) in landed_updates {
                if let Some(ship) = st.ships.get_mut(&ship_id) {
                    ship.position = new_pos;
                    ship.velocity = DVec3::ZERO;
                    ship.angular_velocity = DVec3::ZERO;
                }
            }

            // Co-move FLYING (non-landed) ships inside SOI with their parent planet.
            // Pre-compute deltas to avoid borrow conflicts.
            let co_move_deltas: Vec<(u64, DVec3)> = soi_ships.iter().filter_map(|(ship_id, planet_index)| {
                let ship = st.ships.get(ship_id)?;
                if ship.landed { return None; }
                let delta = st.planet_positions[*planet_index] - old_planet_positions[*planet_index];
                Some((*ship_id, delta))
            }).collect();
            for (ship_id, delta_pos) in co_move_deltas {
                if let Some(ship) = st.ships.get_mut(&ship_id) {
                    ship.position += delta_pos;
                }
            }

            // Destructure to satisfy borrow checker — ships, system_params,
            // planet_positions, and celestial_time are disjoint fields.
            let sys = &st.system_params;
            let pp = &st.planet_positions;
            let ct = st.celestial_time;
            let tick = st.tick_count;

            // Re-solve autopilot intercepts every 20 ticks (~1 second).
            // Smoothed update: blend toward new intercept to prevent oscillation
            // when planet orbital velocity exceeds ship velocity.
            let ap_updates: Vec<(u64, DVec3, f64)> = st.ships.iter().filter_map(|(&id, ship)| {
                if let Some(ref ap) = ship.autopilot {
                    // Don't re-solve while braking — ship is committed to its trajectory.
                    if tick - ap.last_solve_tick >= 20 && !ap.braking_committed {
                        if let Some((new_intercept, new_tof)) = autopilot::solve_intercept(
                            ship.position, ship.velocity,
                            &sys.planets[ap.target_planet_index],
                            &sys.star, ct, sys.scale.time_scale, ap.thrust_tier,
                        ) {
                            return Some((id, new_intercept, new_tof));
                        }
                    }
                }
                None
            }).collect();

            for (sid, new_intercept, new_tof) in ap_updates {
                if let Some(ship) = st.ships.get_mut(&sid) {
                    if let Some(ref mut ap) = ship.autopilot {
                        let old_dist = (ship.position - ap.intercept_pos).length();
                        let shift = (new_intercept - ap.intercept_pos).length();
                        let relative_shift = shift / old_dist.max(1.0);
                        if relative_shift > 0.1 {
                            // Large correction: blend at 30%
                            ap.intercept_pos = ap.intercept_pos.lerp(new_intercept, 0.3);
                        } else if relative_shift > 0.02 {
                            // Small correction: blend at 10%
                            ap.intercept_pos = ap.intercept_pos.lerp(new_intercept, 0.1);
                        }
                        // <2% shift: keep current intercept (stable enough)
                        ap.estimated_tof = new_tof;
                        ap.last_solve_tick = tick;
                    }
                }
            }

            // Phase 1: Autopilot controller — phase-aware dispatch.
            // Interplanetary phases use brachistochrone guidance.
            // Orbital/descent/ascent phases use specialized guidance functions.
            // All phases produce thrust/torque commands through the same thruster system.
            {
            let sys = &st.system_params;
            let pp = &st.planet_positions;
            let pv = &st.planet_velocities;
            let ct = st.celestial_time;
            let physics_time = st.physics_time;

            // Compute guidance for each ship with active autopilot.
            let ap_commands: Vec<(u64, GuidanceCommand, FlightPhase)> = st.ships.iter()
                .filter_map(|(&id, ship)| {
                    let ap = ship.autopilot.as_ref()?;
                    let planet_idx = ap.target_planet_index;
                    let planet = &sys.planets[planet_idx];
                    let planet_pos = pp[planet_idx];
                    let planet_vel = pv[planet_idx];
                    let soi = compute_soi_radius(planet, &sys.star);

                    // Enforce tier restriction in atmosphere.
                    let in_atmo = {
                        let alt = (ship.position - planet_pos).length() - planet.radius_m;
                        alt < planet.atmosphere.atmosphere_height && planet.atmosphere.has_atmosphere
                    };
                    let effective_tier = autopilot::effective_tier(ap.thrust_tier, in_atmo);
                    let engine_accel = autopilot::engine_tier(effective_tier).acceleration;

                    // Phase-aware guidance dispatch.
                    let guidance = match ap.phase {
                        // Interplanetary phases: existing brachistochrone guidance.
                        FlightPhase::Accelerate | FlightPhase::Brake | FlightPhase::Flip => {
                            autopilot::compute_guidance(
                                ship.position, ship.velocity,
                                ap.intercept_pos, ap.target_planet_index,
                                sys, pp, ct, effective_tier,
                            )
                        }
                        // SOI approach: continue braking toward planet.
                        FlightPhase::SoiApproach => {
                            autopilot::compute_guidance(
                                ship.position, ship.velocity,
                                planet_pos, ap.target_planet_index,
                                sys, pp, ct, effective_tier,
                            )
                        }
                        // Circularize / ascent burn: prograde/retrograde burn.
                        FlightPhase::CircularizeBurn | FlightPhase::AscentBurn => {
                            let rel_pos = ship.position - planet_pos;
                            let rel_vel = ship.velocity; // already planet-relative (patched conics)
                            let (dv, burn_dir) = autopilot::circularization_delta_v(
                                rel_pos, rel_vel, planet.gm,
                            );
                            GuidanceCommand {
                                thrust_direction: burn_dir,
                                thrust_magnitude: if dv > 1.0 { engine_accel * ship.physical_properties.mass_kg } else { 0.0 },
                                phase: ap.phase,
                                completed: dv < 1.0,
                                eta_real_seconds: dv / engine_accel,
                                felt_g: autopilot::engine_tier(effective_tier).felt_g,
                                dampener_active: autopilot::engine_tier(effective_tier).dampened,
                            }
                        }
                        // Stable orbit: no thrust.
                        FlightPhase::StableOrbit => {
                            GuidanceCommand {
                                thrust_direction: DVec3::NEG_Z,
                                thrust_magnitude: 0.0,
                                phase: FlightPhase::StableOrbit,
                                completed: false,
                                eta_real_seconds: 0.0,
                                felt_g: 0.0,
                                dampener_active: false,
                            }
                        }
                        // Deorbit burn: retrograde.
                        FlightPhase::DeorbitBurn => {
                            let rel_pos = ship.position - planet_pos;
                            let rel_vel = ship.velocity; // already planet-relative (patched conics)
                            let h = rel_pos.cross(rel_vel);
                            let prograde = if h.length_squared() > 1e-10 {
                                h.cross(rel_pos).normalize()
                            } else {
                                DVec3::NEG_Z
                            };
                            GuidanceCommand {
                                thrust_direction: -prograde,
                                thrust_magnitude: engine_accel * ship.physical_properties.mass_kg,
                                phase: FlightPhase::DeorbitBurn,
                                completed: false,
                                eta_real_seconds: 0.0,
                                felt_g: autopilot::engine_tier(effective_tier).felt_g,
                                dampener_active: autopilot::engine_tier(effective_tier).dampened,
                            }
                        }
                        // Atmospheric entry / terminal descent / landing: use landing guidance.
                        FlightPhase::AtmosphericEntry | FlightPhase::TerminalDescent | FlightPhase::Landing => {
                            autopilot::compute_landing_guidance(
                                ship.position, ship.velocity, planet_pos, planet,
                                &ship.physical_properties, engine_accel,
                            )
                        }
                        // Landed: zero thrust.
                        FlightPhase::Landed => {
                            GuidanceCommand {
                                thrust_direction: DVec3::Y,
                                thrust_magnitude: 0.0,
                                phase: FlightPhase::Landed,
                                completed: true,
                                eta_real_seconds: 0.0,
                                felt_g: 0.0,
                                dampener_active: false,
                            }
                        }
                        // Liftoff / gravity turn / ascent: use takeoff guidance.
                        FlightPhase::Liftoff | FlightPhase::GravityTurn => {
                            autopilot::compute_takeoff_guidance(
                                ship.position, ship.velocity, planet_pos, planet,
                                &ship.physical_properties, engine_accel, ap.target_orbit_altitude,
                            )
                        }
                        // Escape burn: prograde until outside SOI.
                        FlightPhase::EscapeBurn => {
                            let rel_pos = ship.position - planet_pos;
                            let rel_vel = ship.velocity; // already planet-relative (patched conics)
                            let h = rel_pos.cross(rel_vel);
                            let prograde = if h.length_squared() > 1e-10 {
                                h.cross(rel_pos).normalize()
                            } else {
                                DVec3::NEG_Z
                            };
                            GuidanceCommand {
                                thrust_direction: prograde,
                                thrust_magnitude: engine_accel * ship.physical_properties.mass_kg,
                                phase: FlightPhase::EscapeBurn,
                                completed: (ship.position - planet_pos).length() > soi,
                                eta_real_seconds: 0.0,
                                felt_g: autopilot::engine_tier(effective_tier).felt_g,
                                dampener_active: autopilot::engine_tier(effective_tier).dampened,
                            }
                        }
                        FlightPhase::Arrived => {
                            GuidanceCommand {
                                thrust_direction: DVec3::NEG_Z,
                                thrust_magnitude: 0.0,
                                phase: FlightPhase::Arrived,
                                completed: true,
                                eta_real_seconds: 0.0,
                                felt_g: 0.0,
                                dampener_active: false,
                            }
                        }
                    };

                    // Check phase transitions (deterministic, shared with client).
                    let new_phase = autopilot::check_phase_transition(
                        ap.phase, ap.mode,
                        ship.position, ship.velocity,
                        planet_pos, planet_vel, planet, &sys.star,
                        soi, ap.target_orbit_altitude, &ship.physical_properties,
                    );

                    Some((id, guidance, new_phase))
                }).collect();

            for (ship_id, guidance, new_phase) in &ap_commands {
                let ship = st.ships.get_mut(ship_id).unwrap();

                // Apply phase transition.
                if let Some(ref mut ap) = ship.autopilot {
                    if *new_phase != ap.phase {
                        info!(ship_id, old = ?ap.phase, new = ?new_phase, "flight phase transition");
                        ap.phase = *new_phase;
                    }
                }

                // Handle completion.
                if guidance.completed {
                    let should_disengage = if let Some(ref ap) = ship.autopilot {
                        matches!(ap.phase, FlightPhase::Arrived | FlightPhase::Landed)
                            || (ap.phase == FlightPhase::StableOrbit && ap.mode == autopilot::AutopilotMode::OrbitInsertion)
                    } else {
                        true
                    };
                    if should_disengage {
                        let mode = ship.autopilot.as_ref().map(|a| a.mode);
                        info!(ship_id, mode = ?mode, "autopilot completed — disengaging");
                        ship.autopilot = None;
                        ship.thrust = DVec3::ZERO;
                        ship.torque = DVec3::ZERO;
                        continue;
                    }
                }

                // Sticky brake for interplanetary phases.
                if let Some(ref mut ap) = ship.autopilot {
                    if matches!(ap.phase, FlightPhase::Accelerate | FlightPhase::Brake | FlightPhase::Flip) {
                        if guidance.phase == FlightPhase::Brake && !ap.braking_committed {
                            ap.braking_committed = true;
                        }
                    }
                }

                let thrust_dir = guidance.thrust_direction;
                let elapsed = ship.autopilot.as_ref()
                    .map(|ap| physics_time - ap.engage_time).unwrap_or(0.0);
                let tof = ship.autopilot.as_ref()
                    .map(|ap| ap.estimated_tof).unwrap_or(1.0);
                let ramp = autopilot::thrust_ramp_factor(elapsed, tof, guidance.phase);

                // Rotation: PD controller to align ship forward (-Z) with thrust direction.
                if guidance.thrust_magnitude > 0.0 {
                    let desired_fwd = thrust_dir;
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
                    let dot_align = ship.rotation.mul_vec3(DVec3::NEG_Z).dot(thrust_dir).clamp(-1.0, 1.0);
                    let thrust_scale = dot_align.max(0.0) * ramp;
                    ship.thrust = DVec3::new(0.0, 0.0, -guidance.thrust_magnitude * thrust_scale);
                } else {
                    ship.thrust = DVec3::ZERO;
                    ship.torque = DVec3::ZERO;
                }
            }

            } // end autopilot controller block

            // Phase 2: True Velocity Verlet (Störmer-Verlet) integration.
            // Two-pass for symplectic energy conservation — critical for orbital stability.
            // Includes: N-body gravity, ship thrust, orientation-dependent aero forces.

            // Find which planet (if any) each ship is in atmosphere of.
            // Checked every tick against all planets (not gated by in_soi poll).
            let ship_atmo: HashMap<u64, usize> = {
                let sys = &st.system_params;
                let pp = &st.planet_positions;
                st.ships.iter()
                    .filter_map(|(&id, ship)| {
                        check_atmosphere(ship.position, &sys.planets, pp)
                            .map(|(planet_idx, _alt)| (id, planet_idx))
                    }).collect()
            };

            // Snapshot in_soi for gravity computation (avoids borrow conflicts).
            let in_soi_snapshot: HashMap<u64, usize> = st.in_soi.clone();

            // Pass A: compute a_old (gravity + thrust + aero), advance position.
            // Landed ships are excluded — their position is derived from planet state.
            let pass_a: Vec<(u64, DVec3, DVec3, DVec3)> = {
                let sys = &st.system_params;
                let pp = &st.planet_positions;
                let ct = st.celestial_time;
                st.ships.iter().filter(|(_, ship)| !ship.landed).map(|(&id, ship)| {
                    // Inside SOI: only parent planet gravity (star gravity handled by co-movement).
                    // Outside SOI: full N-body gravity.
                    let grav = if let Some(&planet_idx) = in_soi_snapshot.get(&id) {
                        let r = ship.position - pp[planet_idx];
                        let dist_sq = r.length_squared();
                        if dist_sq > 1.0 {
                            -r.normalize() * sys.planets[planet_idx].gm / dist_sq
                        } else {
                            DVec3::ZERO
                        }
                    } else {
                        compute_gravity_acceleration(
                            ship.position, &sys.star, &sys.planets, pp, ct,
                        )
                    };
                    let thrust_accel = ship.rotation * ship.thrust / ship.physical_properties.mass_kg;

                    // Full orientation-dependent aerodynamics.
                    let mut aero_accel = DVec3::ZERO;
                    let mut aero_torque = DVec3::ZERO;
                    if let Some(&planet_idx) = ship_atmo.get(&id) {
                        let mut thermal_tmp = ship.thermal_energy;
                        let aero = compute_full_aerodynamics(
                            ship.position, ship.velocity, ship.rotation, ship.angular_velocity,
                            pp[planet_idx], &sys.planets[planet_idx],
                            &ship.physical_properties, &mut thermal_tmp, DT,
                        );
                        aero_accel = aero.drag_accel + aero.lift_accel;
                        aero_torque = aero.aero_torque;
                    }

                    let accel_old = grav + thrust_accel + aero_accel;
                    (id, accel_old, thrust_accel, aero_torque)
                }).collect()
            };

            for &(ship_id, accel_old, _, _) in &pass_a {
                let ship = st.ships.get_mut(&ship_id).unwrap();
                ship.position += ship.velocity * DT + 0.5 * accel_old * DT * DT;
            }

            // Pass B: recompute gravity+aero at new position, average with a_old for velocity.
            let pass_b: Vec<(u64, DVec3)> = {
                let sys = &st.system_params;
                let pp = &st.planet_positions;
                let ct = st.celestial_time;
                pass_a.iter().map(|&(ship_id, accel_old, thrust_accel, _)| {
                    let ship = &st.ships[&ship_id];
                    // Same SOI-aware gravity as Pass A.
                    let grav_new = if let Some(&planet_idx) = in_soi_snapshot.get(&ship_id) {
                        let r = ship.position - pp[planet_idx];
                        let dist_sq = r.length_squared();
                        if dist_sq > 1.0 {
                            -r.normalize() * sys.planets[planet_idx].gm / dist_sq
                        } else {
                            DVec3::ZERO
                        }
                    } else {
                        compute_gravity_acceleration(
                            ship.position, &sys.star, &sys.planets, pp, ct,
                        )
                    };
                    let mut aero_accel_new = DVec3::ZERO;
                    if let Some(&planet_idx) = ship_atmo.get(&ship_id) {
                        let mut thermal_tmp = ship.thermal_energy;
                        let aero = compute_full_aerodynamics(
                            ship.position, ship.velocity, ship.rotation, ship.angular_velocity,
                            pp[planet_idx], &sys.planets[planet_idx],
                            &ship.physical_properties, &mut thermal_tmp, DT,
                        );
                        aero_accel_new = aero.drag_accel + aero.lift_accel;
                    }
                    let accel_new = grav_new + thrust_accel + aero_accel_new;
                    let vel_delta = 0.5 * (accel_old + accel_new) * DT;
                    (ship_id, vel_delta)
                }).collect()
            };

            // Apply velocity, thermal energy, and aero torque.
            for (ship_id, vel_delta) in pass_b {
                let ship = st.ships.get_mut(&ship_id).unwrap();
                ship.velocity += vel_delta;
                ship.thrust = DVec3::ZERO;

                // Update thermal energy (authoritative, once per tick).
                if let Some(&planet_idx) = ship_atmo.get(&ship_id) {
                    compute_full_aerodynamics(
                        ship.position, ship.velocity, ship.rotation, ship.angular_velocity,
                        st.planet_positions[planet_idx], &st.system_params.planets[planet_idx],
                        &ship.physical_properties, &mut ship.thermal_energy, DT,
                    );
                }

                // Apply aerodynamic torque to angular velocity (from pass A).
                if let Some(&(_, _, _, aero_torque)) = pass_a.iter().find(|t| t.0 == ship_id) {
                    if aero_torque.length_squared() > 1e-10 {
                        let (ix, iy, iz) = ship.physical_properties.moment_of_inertia();
                        let torque_local = ship.rotation.inverse() * aero_torque;
                        let ang_accel_aero = DVec3::new(
                            torque_local.x / ix,
                            torque_local.y / iy,
                            torque_local.z / iz,
                        );
                        ship.angular_velocity += ang_accel_aero * DT;
                    }
                }

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

            // Ground contact — spring-damper model. AFTER Pass B so velocity includes gravity.
            // Checks ALL planets. Critically damped to prevent bounce.
            // Impact damage based on ship hull strength, scaled by planet gravity.
            let ship_ids: Vec<u64> = st.ships.keys().copied().collect();
            for ship_id in ship_ids {
                // Skip ground contact for already-landed ships (position is derived).
                if st.ships.get(&ship_id).map_or(false, |s| s.landed) { continue; }
                let (contact_planet, contact_data) = {
                    let ship = &st.ships[&ship_id];
                    let mut result = None;
                    for (i, planet) in st.system_params.planets.iter().enumerate() {
                        let planet_pos = st.planet_positions[i];
                        let to_ship = ship.position - planet_pos;
                        let dist = to_ship.length();
                        let contact_margin = ship.physical_properties.landing_gear_height + 0.5;
                        let surface_dist = planet.radius_m + contact_margin;

                        if dist < surface_dist && dist > 1.0 {
                            let radial = to_ship / dist;
                            let penetration = surface_dist - dist;
                            let v_radial = ship.velocity.dot(radial);

                            // Derive spring-damper from ship mass and planet gravity (no magic numbers).
                            let weight = ship.physical_properties.mass_kg * planet.surface_gravity;
                            let spring_k = weight * 5.0 / contact_margin;
                            let damping_c = 2.0 * (spring_k * ship.physical_properties.mass_kg).sqrt();

                            // Spring-damper ground reaction
                            let spring_force = spring_k * penetration;
                            let damping_force = -damping_c * v_radial;
                            let ground_force = (spring_force + damping_force).max(0.0);
                            let ground_accel = radial * ground_force / ship.physical_properties.mass_kg;

                            // Friction: kill lateral velocity (rate proportional to gravity)
                            let v_lateral = ship.velocity - radial * v_radial;
                            let friction_rate = planet.surface_gravity / 2.0;

                            result = Some((i, radial, ground_accel, v_radial, v_lateral,
                                           friction_rate, penetration, contact_margin, planet_pos,
                                           planet.radius_m));
                            break;
                        }
                    }
                    (result.map(|r| r.0), result)
                };

                if let Some((planet_idx, radial, ground_accel, v_radial, v_lateral,
                             friction_rate, penetration, contact_margin, planet_pos, planet_radius)) = contact_data
                {
                    let surface_gravity = st.system_params.planets.get(planet_idx)
                        .map(|p| p.surface_gravity).unwrap_or(9.81);
                    let ship = st.ships.get_mut(&ship_id).unwrap();

                    // Impact damage on first contact
                    if !ship.landed && v_radial < -0.5 {
                        let impact_speed = -v_radial;
                        let max_safe = ship.physical_properties.hull_strength * 0.08;
                        let lethal = ship.physical_properties.hull_strength * 0.3;
                        if impact_speed > lethal {
                            info!(ship_id, impact_speed = format!("{:.1}", impact_speed), "ship destroyed on impact");
                            // TODO: emit destruction event
                        } else if impact_speed > max_safe {
                            let damage = (impact_speed - max_safe).powi(2) * 0.5;
                            info!(ship_id, impact_speed = format!("{:.1}", impact_speed),
                                damage = format!("{:.1}", damage), "hard landing — damage taken");
                        }
                    }

                    // Apply ground reaction
                    ship.velocity += ground_accel * DT;

                    // Apply friction
                    let friction_decay = (-friction_rate * DT).exp();
                    let v_rad_component = radial * ship.velocity.dot(radial);
                    let v_lat_component = ship.velocity - v_rad_component;
                    ship.velocity = v_rad_component + v_lat_component * friction_decay;

                    // Prevent sinking below surface
                    let dist_now = (ship.position - planet_pos).length();
                    if dist_now < planet_radius + 0.5 {
                        ship.position = planet_pos + radial * (planet_radius + contact_margin);
                    }

                    // Landing detection: proximity + low radial velocity + debounce.
                    // Uses altitude and radial velocity (not penetration, which conflicts with position snap).
                    let to_ship = ship.position - planet_pos;
                    let dist_check = to_ship.length();
                    let altitude_check = dist_check - planet_radius;
                    let radial_check = to_ship / dist_check;
                    let v_radial_check = ship.velocity.dot(radial_check);

                    let landing_zone = ship.physical_properties.landing_gear_height + 1.0;
                    let max_landing_speed = 2.0 * surface_gravity.sqrt();

                    if altitude_check < landing_zone && v_radial_check.abs() < max_landing_speed {
                        ship.landing_zone_ticks += 1;
                        // 10 consecutive ticks (~0.5s) in landing zone confirms landing.
                        if ship.landing_zone_ticks >= 10 {
                            ship.landed = true;
                            ship.landed_planet_index = Some(planet_idx);
                            ship.landed_surface_radial = radial_check;
                            ship.landed_celestial_time = celestial_time;
                            ship.velocity = DVec3::ZERO;
                            ship.angular_velocity = DVec3::ZERO;
                            ship.landing_zone_ticks = 0;
                            info!(ship_id, planet_index = planet_idx,
                                alt = format!("{:.1}", altitude_check),
                                "ship landed — surface attached");
                        }
                    } else {
                        ship.landing_zone_ticks = 0;
                    }
                } else if let Some(ship) = st.ships.get_mut(&ship_id) {
                    // No contact — check for takeoff
                    if ship.landed {
                        ship.landed = false;
                        ship.landed_planet_index = None;
                    }
                }
            }

            // Phase C: Write back results (lock held for microseconds).
            {
                let mut locked = state_physics.lock().unwrap();
                locked.ships = st.ships;
                locked.planet_positions = st.planet_positions;
                locked.planet_velocities = st.planet_velocities;
            }
        });

        // System 4: SOI detection + planet shard provisioning.
        // Tracks ships entering/leaving planet SOIs and provisions planet shards.
        let state_soi = state.clone();
        let orchestrator_url_soi = orchestrator_url.clone();
        let provision_tx_soi = provision_tx.clone();
        let http_client_soi = http_client.clone();
        harness.add_system("soi_detection", move || {
            let mut st = state_soi.lock().unwrap();
            if st.tick_count % 20 != 0 { return; } // check every second

            // Drain async planet provision results from the channel.
            while let Ok((planet_seed, shard_id)) = provision_rx.try_recv() {
                st.provisioned_planets.insert(planet_seed, shard_id);
                st.provisioning_in_flight.remove(&planet_seed);
                info!(planet_seed, shard_id = shard_id.0, "planet shard provisioned");
            }

            // Check each ship against each planet SOI.
            let mut soi_entries: Vec<(u64, usize)> = Vec::new();
            let mut soi_exits: Vec<u64> = Vec::new();

            for ship in st.ships.values() {
                let mut found_soi = false;
                for (i, planet) in st.system_params.planets.iter().enumerate() {
                    let dist = (ship.position - st.planet_positions[i]).length();
                    let soi = compute_soi_radius(planet, &st.system_params.star);
                    if dist < soi {
                        found_soi = true;
                        if st.in_soi.get(&ship.ship_id) != Some(&i) {
                            soi_entries.push((ship.ship_id, i));
                        }
                        break;
                    }
                }
                if !found_soi && st.in_soi.contains_key(&ship.ship_id) {
                    soi_exits.push(ship.ship_id);
                }
            }

            // Process SOI entries — patched conics: convert to planet-relative frame.
            // Subtract planet's real-time velocity so the ship has planet-relative velocity.
            // The co-movement system (above) keeps the ship's position tracking the planet.
            for (ship_id, planet_index) in soi_entries {
                st.in_soi.insert(ship_id, planet_index);
                let planet = &st.system_params.planets[planet_index];
                let planet_seed = planet.planet_seed;

                let planet_real_vel = st.planet_velocities[planet_index];
                if let Some(ship) = st.ships.get_mut(&ship_id) {
                    let rel_speed_before = (ship.velocity - planet_real_vel).length();
                    ship.velocity -= planet_real_vel; // convert to planet-relative frame
                    info!(ship_id, planet_index, planet_seed,
                        planet_vel_mag = format!("{:.0}", planet_real_vel.length()),
                        rel_speed = format!("{:.0}", rel_speed_before),
                        "ship entered planet SOI — converted to planet-relative frame");
                }

                // Provision planet shard if not already provisioned or in-flight.
                if !st.provisioned_planets.contains_key(&planet_seed)
                    && !st.provisioning_in_flight.contains(&planet_seed) {
                    st.provisioning_in_flight.insert(planet_seed);
                    let url = orchestrator_url_soi.clone();
                    let system_seed = st.system_params.system_seed;
                    let tx = provision_tx_soi.clone();
                    let client = http_client_soi.clone();
                    tokio::spawn(async move {
                        let endpoint = format!(
                            "{}/planet/{}?system_seed={}&planet_index={}",
                            url, planet_seed, system_seed, planet_index
                        );
                        match client.get(&endpoint).send().await {
                            Ok(resp) if resp.status().is_success() => {
                                if let Ok(body) = resp.text().await {
                                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                                        if let Some(id) = v.get("shards").and_then(|s| s.get(0)).and_then(|s| s.get("id")).and_then(|v| v.as_u64()) {
                                            let _ = tx.try_send((planet_seed, ShardId(id)));
                                        }
                                    }
                                }
                            }
                            Ok(resp) => {
                                tracing::warn!(status = %resp.status(), planet_seed,
                                    "failed to provision planet shard");
                            }
                            Err(e) => {
                                tracing::warn!(%e, planet_seed, "planet shard provision request failed");
                            }
                        }
                    });
                }
            }

            // Provision planet shards for all ships in SOI (catches pre-registered spawns).
            // This runs every SOI check tick and is idempotent — only provisions if not yet in map.
            let soi_planet_indices: Vec<usize> = st.in_soi.values().copied().collect();
            for planet_index in soi_planet_indices {
                let planet = &st.system_params.planets[planet_index];
                let planet_seed = planet.planet_seed;
                if !st.provisioned_planets.contains_key(&planet_seed)
                    && !st.provisioning_in_flight.contains(&planet_seed) {
                    st.provisioning_in_flight.insert(planet_seed);
                    let url = orchestrator_url_soi.clone();
                    let system_seed = st.system_params.system_seed;
                    let tx = provision_tx_soi.clone();
                    let client = http_client_soi.clone();
                    tokio::spawn(async move {
                        let endpoint = format!(
                            "{}/planet/{}?system_seed={}&planet_index={}",
                            url, planet_seed, system_seed, planet_index
                        );
                        match client.get(&endpoint).send().await {
                            Ok(resp) if resp.status().is_success() => {
                                if let Ok(body) = resp.text().await {
                                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                                        if let Some(id) = v.get("shards").and_then(|s| s.get(0)).and_then(|s| s.get("id")).and_then(|v| v.as_u64()) {
                                            let _ = tx.try_send((planet_seed, ShardId(id)));
                                        }
                                    }
                                }
                            }
                            Ok(resp) => {
                                tracing::warn!(status = %resp.status(), planet_seed,
                                    "failed to provision planet shard (from SOI check)");
                            }
                            Err(e) => {
                                tracing::warn!(%e, planet_seed,
                                    "planet shard provision request failed (from SOI check)");
                            }
                        }
                    });
                    info!(planet_seed, planet_index, "provisioning planet shard for ship in SOI");
                }
            }

            // Process SOI exits — patched conics: convert back to system-absolute frame.
            for ship_id in soi_exits {
                if let Some(&planet_index) = st.in_soi.get(&ship_id) {
                    let planet_real_vel = st.planet_velocities[planet_index];
                    if let Some(ship) = st.ships.get_mut(&ship_id) {
                        ship.velocity += planet_real_vel; // convert back to system frame
                        info!(ship_id, "ship left planet SOI — converted to system frame");
                    }
                }
                st.in_soi.remove(&ship_id);
            }
        });

        // System 5: Broadcast SystemSceneUpdate to ship shards via QUIC.
        let state_scene = state.clone();
        let peer_reg_scene = peer_registry.clone();
        let quic_send_scene = quic_send_tx.clone();
        harness.add_system("broadcast_scene", move || {
            // Snapshot peer registry ONCE, before locking state.
            // Eliminates 3 separate try_read() calls while holding the state lock,
            // and prevents RwLock contention with the periodic registry refresh.
            let (ship_shards, planet_shards): (
                Vec<(ShardId, SocketAddr, Option<u64>, Option<ShardId>)>,
                Vec<(ShardId, SocketAddr, Option<u64>)>,
            ) = if let Ok(reg) = peer_reg_scene.try_read() {
                let ships = reg.find_by_type(ShardType::Ship).iter()
                    .map(|info| (info.id, info.endpoint.quic_addr, info.ship_id, info.host_shard_id))
                    .collect();
                let planets = reg.find_by_type(ShardType::Planet).iter()
                    .map(|info| (info.id, info.endpoint.quic_addr, info.planet_seed))
                    .collect();
                (ships, planets)
            } else {
                return; // Registry write-locked (refresh) — skip this tick
            };

            let st = state_scene.lock().unwrap();

            // Filter to ship shards hosted by this system shard.
            let hosted_ships: Vec<(ShardId, SocketAddr, Option<u64>)> = ship_shards.iter()
                .filter(|(_, _, _, host)| *host == Some(st.shard_id))
                .map(|(id, addr, ship_id, _)| (*id, *addr, *ship_id))
                .collect();

            if hosted_ships.is_empty() { return; }

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

            for &(shard_id, quic_addr, _) in &hosted_ships {
                let _ = quic_send_scene.try_send((shard_id, quic_addr, scene_msg.clone()));
            }

            // Send per-ship position updates to the CORRECT ship shard only.
            for ship in st.ships.values() {
                let target = hosted_ships.iter()
                    .find(|(_, _, sid)| *sid == Some(ship.ship_id))
                    .map(|&(sid, addr, _)| (sid, addr));
                if let Some((sid, addr)) = target {
                    let pos_msg = ShardMsg::ShipPositionUpdate(ShipPositionUpdate {
                        ship_id: ship.ship_id,
                        position: ship.position,
                        velocity: ship.velocity,
                        rotation: ship.rotation,
                        angular_velocity: DVec3::ZERO,
                    });
                    let _ = quic_send_scene.try_send((sid, addr, pos_msg));
                }
            }

            // Send ShipNearbyInfo to planet shards for ships within their SOI.
            for (&ship_id, &planet_index) in &st.in_soi {
                let planet_seed = st.system_params.planets[planet_index].planet_seed;
                let planet_shard = planet_shards.iter()
                    .find(|(_, _, ps)| *ps == Some(planet_seed))
                    .map(|&(sid, addr, _)| (sid, addr));

                if let (Some((psid, paddr)), Some(ship)) = (planet_shard, st.ships.get(&ship_id)) {
                    let ship_shard_id = hosted_ships.first()
                        .map(|&(sid, _, _)| sid)
                        .unwrap_or(ShardId(0));

                    let nearby_msg = ShardMsg::ShipNearbyInfo(ShipNearbyInfoData {
                        ship_id,
                        ship_shard_id,
                        position: ship.position,
                        rotation: ship.rotation,
                        velocity: ship.velocity,
                    });
                    let _ = quic_send_scene.try_send((psid, paddr, nearby_msg));
                }
            }
        });

        // System 6: Broadcast WorldState to direct UDP clients (debug mode).
        let state_broadcast = state.clone();
        harness.add_system("broadcast_udp", move || {
            let st = state_broadcast.lock().unwrap();
            let ws = ServerMsg::WorldState(st.build_world_state());
            let _ = broadcast_tx.try_send(ws);
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
