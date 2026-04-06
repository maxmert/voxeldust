use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use clap::Parser;
use glam::{DQuat, DVec3, Vec3};
use rapier3d::prelude::*;
use tracing::{info, warn};

use voxeldust_core::client_message::{
    CelestialBodyData, JoinResponseData, LightingData, PlayerSnapshotData,
    ServerMsg, ShipRenderData, WorldStateData,
};
use voxeldust_core::shard_message::{
    AutopilotCommandData, AutopilotSnapshotData, CelestialBodySnapshotData,
    LightingInfoData, ShardMsg, ShipControlInput, ShipSnapshotEntryData,
};
use voxeldust_core::handoff;
use voxeldust_core::shard_types::{ShardId, ShardType, SessionToken};
use voxeldust_shard_common::client_listener;
use voxeldust_shard_common::harness::{ShardHarness, ShardHarnessConfig};

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

const SHIP_WIDTH: f32 = 4.0;
const SHIP_LENGTH: f32 = 8.0;
const SHIP_HEIGHT: f32 = 3.0;
const WALL_THICKNESS: f32 = 0.1;
const WALK_SPEED: f32 = 4.0;
const JUMP_IMPULSE: f32 = 5.0;
const THRUST_FORCE: f64 = 49_050.0; // Newtons (maneuver tier: 0.5g at 10t)
const TORQUE_FORCE: f64 = 5_000.0;

// Interaction points (ship-local coordinates).
const PILOT_SEAT: Vec3 = Vec3::new(0.0, 0.5, -3.0);
const EXIT_DOOR: Vec3 = Vec3::new(2.0, 0.5, 0.0);
const INTERACT_DIST: f32 = 1.5;

/// Configurable gravity source — future: one per gravity block.
/// For now, ships have a single default source (floor plates).
#[derive(Clone)]
struct GravitySource {
    /// Direction of gravity in ship-local space.
    direction: Vec3,
    /// Strength in m/s².
    strength: f32,
    /// Shape of the gravity field.
    shape: GravityShape,
}

#[derive(Clone)]
enum GravityShape {
    /// Applies uniformly everywhere in the ship.
    Uniform,
    // Future: Sphere { radius: f32 }, Zone { min: Vec3, max: Vec3 }
}

impl GravitySource {
    fn default_floor_plates() -> Self {
        Self {
            direction: Vec3::new(0.0, -1.0, 0.0),
            strength: 9.81,
            shape: GravityShape::Uniform,
        }
    }

    /// Compute gravity vector at a given ship-local position.
    fn gravity_at(&self, _position: Vec3) -> Vector<f32> {
        match self.shape {
            GravityShape::Uniform => {
                let g = self.direction * self.strength;
                vector![g.x, g.y, g.z]
            }
            // Future: Sphere/Zone would check distance/containment
        }
    }
}

struct ShipInteriorState {
    // Rapier physics
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    /// Gravity sources in the ship. Future: populated from gravity blocks.
    gravity_sources: Vec<GravitySource>,
    integration_params: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,

    // Player
    player_body: Option<RigidBodyHandle>,
    player_position: Vec3,
    player_yaw: f32,
    is_piloting: bool,
    gravity_enabled: bool,

    // Ship exterior state (from system shard)
    ship_position: DVec3,
    ship_velocity: DVec3,
    ship_rotation: DQuat,

    // Cached scene from system shard
    scene_bodies: Vec<CelestialBodySnapshotData>,
    scene_lighting: Option<LightingInfoData>,
    game_time: f64,
    /// Tick when last SystemSceneUpdate arrived via QUIC.
    last_scene_update_tick: u64,
    /// Tick when last HostSwitch was processed. SystemSceneUpdate messages
    /// arriving within 40 ticks of a HostSwitch are ignored (stale QUIC drain).
    host_switch_tick: u64,
    /// Authorized QUIC peers for scene data. Updated on HostSwitch.
    authorized_peers: voxeldust_shard_common::authorized_peers::AuthorizedPeers,
    /// Cached system params for Keplerian planet extrapolation during QUIC stalls.
    cached_system_params: Option<voxeldust_core::system::SystemParams>,
    galaxy_seed: u64,

    /// Ship physical properties (mass, thrust, drag). Per-ship, derived from block composition.
    ship_props: voxeldust_core::autopilot::ShipPhysicalProperties,

    // Pilot thrust (accumulated from input, sent each tick, then reset)
    pilot_thrust: DVec3,
    pilot_torque: DVec3,

    // Identity
    shard_id: ShardId,
    ship_id: u64,
    host_shard_id: Option<ShardId>,
    tick_count: u64,

    // Input state
    last_action: u8,
    prev_action: u8,

    // Autopilot
    autopilot_target_body_id: Option<u32>,
    pending_autopilot_cmd: Option<(u32, u8)>, // (target_body_id, speed_tier)
    /// Server-authoritative autopilot state from system shard.
    autopilot_snapshot: Option<AutopilotSnapshotData>,

    // Warp
    warp_target_star_index: Option<u32>,
    pending_warp_cmd: Option<u32>, // target_star_index (u32::MAX = disengage)

    // Connected player (for handoff)
    connected_session: Option<SessionToken>,
    connected_player_name: Option<String>,
    /// True while a handoff is in-flight (block input processing).
    handoff_pending: bool,
    /// Whether the exit door is open (blocker collider removed).
    door_open: bool,
    /// Handle to the door blocker collider (None when door is open).
    door_blocker_handle: Option<ColliderHandle>,
    /// True when the ship is on a planet surface (velocity near zero from system shard).
    ship_landed: bool,
    /// True when the ship is inside a planet's atmosphere (for ShardPreConnect trigger).
    ship_in_atmosphere: bool,
    /// True after ShardPreConnect has been sent for this atmosphere entry (prevent spam).
    preconnect_sent: bool,
    /// System seed for planet seed derivation during handoff.
    system_seed: u64,
    /// Pending handoff message to send via QUIC (set by interaction, sent by pilot_send).
    pending_handoff_msg: Option<ShardMsg>,
}

impl ShipInteriorState {
    fn new(shard_id: ShardId, ship_id: u64, host_shard_id: Option<ShardId>, system_seed: u64, galaxy_seed: u64) -> Self {
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();

        // Ship interior colliders.
        let floor = ColliderBuilder::cuboid(SHIP_WIDTH / 2.0, WALL_THICKNESS, SHIP_LENGTH / 2.0)
            .translation(vector![0.0, -WALL_THICKNESS, 0.0]).build();
        collider_set.insert(floor);

        let ceiling = ColliderBuilder::cuboid(SHIP_WIDTH / 2.0, WALL_THICKNESS, SHIP_LENGTH / 2.0)
            .translation(vector![0.0, SHIP_HEIGHT + WALL_THICKNESS, 0.0]).build();
        collider_set.insert(ceiling);

        let left = ColliderBuilder::cuboid(WALL_THICKNESS, SHIP_HEIGHT / 2.0, SHIP_LENGTH / 2.0)
            .translation(vector![-SHIP_WIDTH / 2.0 - WALL_THICKNESS, SHIP_HEIGHT / 2.0, 0.0]).build();
        collider_set.insert(left);

        // Right wall: split around door opening (z: -0.6 to +0.6, y: 0 to 2.1).
        // Matches the visual mesh gap in gpu.rs generate_box_mesh().
        let door_half_z = 0.6_f32;
        let door_height = 2.1_f32;
        let right_x = SHIP_WIDTH / 2.0 + WALL_THICKNESS;

        // Segment left of door (z: -SHIP_LENGTH/2 to -door_half_z)
        let seg_len = (SHIP_LENGTH / 2.0 - door_half_z) / 2.0;
        let seg_z = -(SHIP_LENGTH / 2.0 + door_half_z) / 2.0;
        let right_left = ColliderBuilder::cuboid(WALL_THICKNESS, SHIP_HEIGHT / 2.0, seg_len)
            .translation(vector![right_x, SHIP_HEIGHT / 2.0, seg_z]).build();
        collider_set.insert(right_left);

        // Segment right of door (z: +door_half_z to +SHIP_LENGTH/2)
        let right_right = ColliderBuilder::cuboid(WALL_THICKNESS, SHIP_HEIGHT / 2.0, seg_len)
            .translation(vector![right_x, SHIP_HEIGHT / 2.0, -seg_z]).build();
        collider_set.insert(right_right);

        // Segment above door (z: -door_half_z to +door_half_z, y: door_height to SHIP_HEIGHT)
        let above_half_h = (SHIP_HEIGHT - door_height) / 2.0;
        let right_above = ColliderBuilder::cuboid(WALL_THICKNESS, above_half_h, door_half_z)
            .translation(vector![right_x, door_height + above_half_h, 0.0]).build();
        collider_set.insert(right_above);

        // Door blocker: fills the door gap. Removed when door opens, re-inserted when closed.
        let door_blocker = ColliderBuilder::cuboid(WALL_THICKNESS, door_height / 2.0, door_half_z)
            .translation(vector![right_x, door_height / 2.0, 0.0]).build();
        let door_blocker_handle = collider_set.insert(door_blocker);

        let back = ColliderBuilder::cuboid(SHIP_WIDTH / 2.0, SHIP_HEIGHT / 2.0, WALL_THICKNESS)
            .translation(vector![0.0, SHIP_HEIGHT / 2.0, SHIP_LENGTH / 2.0 + WALL_THICKNESS]).build();
        collider_set.insert(back);

        let front = ColliderBuilder::cuboid(SHIP_WIDTH / 2.0, SHIP_HEIGHT / 2.0, WALL_THICKNESS)
            .translation(vector![0.0, SHIP_HEIGHT / 2.0, -SHIP_LENGTH / 2.0 - WALL_THICKNESS]).build();
        collider_set.insert(front);

        // Player capsule.
        let player_rb = RigidBodyBuilder::dynamic()
            .translation(vector![0.0, 1.0, 0.0])
            .lock_rotations()
            .build();
        let player_handle = rigid_body_set.insert(player_rb);
        let player_collider = ColliderBuilder::capsule_y(0.6, 0.3).build();
        collider_set.insert_with_parent(player_collider, player_handle, &mut rigid_body_set);

        let mut s = Self {
            rigid_body_set, collider_set,
            gravity_sources: vec![GravitySource::default_floor_plates()],
            integration_params: {
                let mut p = IntegrationParameters::default();
                p.dt = 0.05; // Match 20Hz tick rate
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
            player_body: Some(player_handle),
            player_position: Vec3::new(0.0, 1.0, 0.0),
            player_yaw: 0.0,
            is_piloting: false,
            gravity_enabled: true,
            ship_position: DVec3::ZERO, // overwritten below
            ship_velocity: DVec3::ZERO,
            ship_rotation: DQuat::IDENTITY,
            scene_bodies: vec![],
            scene_lighting: None,
            game_time: 0.0,
            last_scene_update_tick: 0,
            host_switch_tick: 0,
            authorized_peers: Default::default(),
            cached_system_params: None,
            galaxy_seed,
            ship_props: voxeldust_core::autopilot::ShipPhysicalProperties::starter_ship(),
            pilot_thrust: DVec3::ZERO,
            pilot_torque: DVec3::ZERO,
            shard_id, ship_id, host_shard_id,
            tick_count: 0,
            last_action: 0,
            prev_action: 0,
            autopilot_target_body_id: None,
            pending_autopilot_cmd: None,
            autopilot_snapshot: None,
            warp_target_star_index: None,
            pending_warp_cmd: None,
            connected_session: None,
            connected_player_name: None,
            handoff_pending: false,
            door_open: false,
            door_blocker_handle: Some(door_blocker_handle),
            ship_landed: false,
            ship_in_atmosphere: false,
            preconnect_sent: false,
            system_seed,
            pending_handoff_msg: None,
        };

        // Initialize ship position and scene from system seed so the ship
        // has correct state before the system shard's first QUIC update.
        if system_seed > 0 {
            use voxeldust_core::system::{SystemParams, compute_planet_position, compute_lighting};
            let sys = SystemParams::from_seed(system_seed);
            let planet_pos = compute_planet_position(&sys.planets[0], 0.0);
            s.ship_position = planet_pos + DVec3::new(sys.scale.spawn_offset, 0.0, 0.0);

            s.scene_bodies.push(CelestialBodySnapshotData {
                body_id: 0,
                position: DVec3::ZERO,
                radius: sys.star.radius_m,
                color: sys.star.color,
            });
            for (i, planet) in sys.planets.iter().enumerate() {
                let pos = compute_planet_position(planet, 0.0);
                s.scene_bodies.push(CelestialBodySnapshotData {
                    body_id: (i + 1) as u32,
                    position: pos,
                    radius: planet.radius_m,
                    color: planet.color,
                });
            }
            let l = compute_lighting(s.ship_position, &sys.star);
            s.scene_lighting = Some(LightingInfoData {
                sun_direction: l.sun_direction,
                sun_color: l.sun_color,
                sun_intensity: l.sun_intensity,
                ambient: l.ambient,
            });
            s.cached_system_params = Some(sys);
        }

        s
    }

    fn step(&mut self) {
        // Compute gravity from all sources at the player's position.
        let gravity = if self.gravity_enabled {
            let mut total = vector![0.0, 0.0, 0.0];
            for source in &self.gravity_sources {
                total += source.gravity_at(self.player_position);
            }
            total
        } else {
            vector![0.0, 0.0, 0.0]
        };

        self.physics_pipeline.step(
            &gravity, &self.integration_params,
            &mut self.island_manager, &mut self.broad_phase, &mut self.narrow_phase,
            &mut self.rigid_body_set, &mut self.collider_set,
            &mut self.impulse_joint_set, &mut self.multibody_joint_set,
            &mut self.ccd_solver, Some(&mut self.query_pipeline), &(), &(),
        );
        if let Some(handle) = self.player_body {
            if let Some(body) = self.rigid_body_set.get(handle) {
                let t = body.translation();
                self.player_position = Vec3::new(t.x, t.y, t.z);
            }
        }
    }

    fn build_world_state(&self) -> WorldStateData {
        // If scene data is stale (>1 second without update from host), clear bodies.
        // This prevents rendering planets from a departed system after HostSwitch
        // while waiting for the new host's scene updates to arrive.
        let scene_stale = self.tick_count > self.last_scene_update_tick + 20;
        let bodies: Vec<CelestialBodyData> = if scene_stale {
            Vec::new()
        } else {
            self.scene_bodies.iter().map(|b| {
                CelestialBodyData {
                    body_id: b.body_id,
                    position: b.position,
                    radius: b.radius,
                    color: b.color,
                }
            }).collect()
        };

        let lighting = if scene_stale {
            None
        } else {
            self.scene_lighting.as_ref().map(|l| LightingData {
                sun_direction: l.sun_direction,
                sun_color: l.sun_color,
                sun_intensity: l.sun_intensity,
                ambient: l.ambient,
            })
        };

        // Diagnostic: log ship-to-planet distance as seen by the ship shard.
        if self.tick_count % 20 == 0 && !self.scene_bodies.is_empty() {
            let nearest = self.scene_bodies.iter().filter(|b| b.body_id > 0).min_by_key(|b| {
                ((self.ship_position - b.position).length() * 1000.0) as u64
            });
            if let Some(body) = nearest {
                let dist = (self.ship_position - body.position).length();
                let stale_ticks = self.tick_count.saturating_sub(self.last_scene_update_tick);
                tracing::warn!(
                    dist = format!("{:.0}", dist),
                    stale = stale_ticks,
                    body_id = body.body_id,
                    ship_pos = format!("({:.0},{:.0},{:.0})", self.ship_position.x, self.ship_position.y, self.ship_position.z),
                    body_pos = format!("({:.0},{:.0},{:.0})", body.position.x, body.position.y, body.position.z),
                    "SHIP-SHARD planet distance diagnostic"
                );
            }
        }

        WorldStateData {
            tick: self.tick_count,
            origin: self.ship_position,
            players: vec![PlayerSnapshotData {
                player_id: 0,
                position: DVec3::new(
                    self.player_position.x as f64,
                    self.player_position.y as f64,
                    self.player_position.z as f64,
                ),
                rotation: self.ship_rotation,
                velocity: self.ship_velocity,
                grounded: !self.is_piloting,
                health: 100.0,
                shield: 100.0,
            }],
            bodies,
            ships: vec![],
            lighting,
            game_time: self.game_time,
            warp_target_star_index: self.warp_target_star_index.unwrap_or(0xFFFFFFFF),
            autopilot: self.autopilot_snapshot.clone(),
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

    let ship_id = if args.ship_id > 0 { args.ship_id } else { args.seed };
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
        system_seed: if args.system_seed > 0 { Some(args.system_seed) } else { None },
        ship_id: Some(ship_id),
        galaxy_seed: None,
        host_shard_id,
        advertise_host: args.advertise_host,
    };

    info!(shard_id = args.shard_id, ship_id, host_shard = ?host_shard_id, "ship shard starting");

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let mut harness = ShardHarness::new(config);

        let state = Arc::new(Mutex::new(ShipInteriorState::new(
            ShardId(args.shard_id), ship_id, host_shard_id, args.system_seed, args.galaxy_seed,
        )));
        let client_registry = harness.client_registry.clone();

        // Take channels from harness.
        let mut connect_rx = std::mem::replace(
            &mut harness.connect_rx,
            tokio::sync::mpsc::unbounded_channel().1,
        );
        let mut input_rx = std::mem::replace(
            &mut harness.input_rx,
            tokio::sync::mpsc::unbounded_channel().1,
        );
        let mut quic_msg_rx = std::mem::replace(
            &mut harness.quic_msg_rx,
            tokio::sync::mpsc::unbounded_channel().1,
        );
        let broadcast_tx = harness.broadcast_tx.clone();
        let quic_send_tx = harness.quic_send_tx.clone();
        let peer_registry = harness.peer_registry.clone();

        // System 1: Drain client connections + send JoinResponse.
        let state_connect = state.clone();
        let registry_connect = client_registry.clone();
        let orchestrator_url = args.orchestrator.clone();
        let system_seed_for_jr = args.system_seed;
        harness.add_system("drain_connects", move || {
            for _ in 0..16 { let event = match connect_rx.try_recv() { Ok(e) => e, Err(_) => break };
                let conn = event.connection;
                let token = conn.session_token;

                if let Ok(mut reg) = registry_connect.try_write() {
                    reg.register(&conn);
                }

                let mut st = state_connect.lock().unwrap();
                st.connected_session = Some(token);
                st.connected_player_name = Some(conn.player_name.clone());
                st.handoff_pending = false;
                let tcp_stream = conn.tcp_stream.clone();
                let ship_pos = st.ship_position;
                let ship_rot = st.ship_rotation;
                let game_time = st.game_time;
                let ship_id = st.ship_id;
                let galaxy_seed_jr = st.galaxy_seed;

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
                        galaxy_seed: galaxy_seed_jr,
                        system_seed: system_seed_for_jr,
                        game_time,
                        reference_position: ship_pos,
                        reference_rotation: ship_rot,
                    });
                    let mut stream = tcp_stream.lock().await;
                    let _ = client_listener::send_tcp_msg(&mut stream, &jr).await;
                });

                info!(player = %conn.player_name, session = token.0, "player entered ship");
            }
        });

        // System 2: Drain PlayerInput from UDP.
        let state_input = state.clone();
        harness.add_system("drain_input", move || {
            for _ in 0..64 { let (_src, input) = match input_rx.try_recv() { Ok(e) => e, Err(_) => break };
                let mut st = state_input.lock().unwrap();
                st.prev_action = st.last_action;
                st.last_action = input.action;
                st.player_yaw = input.look_yaw;

                if st.is_piloting {
                    // Autopilot toggle (T key = action 4, rising edge).
                    let autopilot_pressed = input.action == 4 && st.prev_action != 4;
                    if autopilot_pressed {
                        if st.autopilot_target_body_id.is_some() {
                            // Disengage.
                            st.autopilot_target_body_id = None;
                            st.pending_autopilot_cmd = Some((0xFFFFFFFF, 0));
                            info!("autopilot disengage requested");
                        } else {
                            // Engage: find planet most aligned with ship forward.
                            let ship_fwd = st.ship_rotation * DVec3::NEG_Z;
                            let mut best_body_id: Option<u32> = None;
                            let mut best_dot = 0.9; // ~25° alignment threshold
                            for body in &st.scene_bodies {
                                if body.body_id == 0 { continue; } // skip star
                                let to_body = (body.position - st.ship_position).normalize_or_zero();
                                let d = ship_fwd.dot(to_body);
                                if d > best_dot {
                                    best_dot = d;
                                    best_body_id = Some(body.body_id);
                                }
                            }
                            if let Some(body_id) = best_body_id {
                                st.autopilot_target_body_id = Some(body_id);
                                st.pending_autopilot_cmd = Some((body_id, input.speed_tier));
                                info!(body_id, "autopilot engage requested");
                            }
                        }
                    }

                    // Warp targeting (G key = action 6): select/cycle target star.
                    // The client handles visual targeting; the ship shard mirrors
                    // the selection so it knows which star to target on confirmation.
                    // Warp target cycling (G = action 6): server-authoritative.
                    // First press: best-aligned star. Subsequent: cycle to next.
                    let warp_pressed = input.action == 6 && st.prev_action != 6;
                    if warp_pressed {
                        let ship_fwd = st.ship_rotation * DVec3::NEG_Z;
                        if let Some(ref sys) = st.cached_system_params {
                            let galaxy_seed = st.galaxy_seed;
                            if galaxy_seed != 0 {
                                let galaxy_map = voxeldust_core::galaxy::GalaxyMap::generate(galaxy_seed);
                                let current_star = galaxy_map.stars.iter()
                                    .find(|s| s.system_seed == sys.system_seed);
                                if let Some(cur) = current_star {
                                    let cur_pos = cur.position;

                                    if let Some(current_target) = st.warp_target_star_index {
                                        // Cycle: find next-best aligned star after current target.
                                        let current_dot = galaxy_map.stars.iter()
                                            .find(|s| s.index == current_target)
                                            .map(|s| ship_fwd.dot((s.position - cur_pos).normalize()))
                                            .unwrap_or(1.0);

                                        let mut best: Option<(u32, f64)> = None;
                                        for star in &galaxy_map.stars {
                                            if star.index == cur.index || star.index == current_target { continue; }
                                            let dir = (star.position - cur_pos).normalize();
                                            let dot = ship_fwd.dot(dir);
                                            if dot < current_dot && dot > 0.3 {
                                                if dot > best.map(|b| b.1).unwrap_or(0.3) {
                                                    best = Some((star.index, dot));
                                                }
                                            }
                                        }
                                        // Wrap around if no next candidate.
                                        if best.is_none() {
                                            for star in &galaxy_map.stars {
                                                if star.index == cur.index { continue; }
                                                let dir = (star.position - cur_pos).normalize();
                                                let dot = ship_fwd.dot(dir);
                                                if dot > best.map(|b| b.1).unwrap_or(0.3) {
                                                    best = Some((star.index, dot));
                                                }
                                            }
                                        }
                                        if let Some((idx, _)) = best {
                                            st.warp_target_star_index = Some(idx);
                                            info!(target_star = idx, "warp target cycled");
                                        }
                                    } else {
                                        // First press: find best-aligned star.
                                        let mut best: Option<(u32, f64)> = None;
                                        for star in &galaxy_map.stars {
                                            if star.index == cur.index { continue; }
                                            let dir = (star.position - cur_pos).normalize();
                                            let alignment = ship_fwd.dot(dir);
                                            if alignment > best.map(|b| b.1).unwrap_or(0.3) {
                                                best = Some((star.index, alignment));
                                            }
                                        }
                                        if let Some((target_index, _)) = best {
                                            st.warp_target_star_index = Some(target_index);
                                            info!(target_star = target_index, "warp target selected");
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Warp confirm (Enter = action 7): engage warp to targeted star.
                    let warp_confirm = input.action == 7 && st.prev_action != 7;
                    if warp_confirm {
                        if let Some(target) = st.warp_target_star_index {
                            st.pending_warp_cmd = Some(target);
                            info!(target_star = target, "warp engage confirmed");
                        }
                    }

                    // Cancel autopilot on manual WASD input.
                    if st.autopilot_target_body_id.is_some() {
                        let has_manual = input.movement[0].abs() > 0.01
                            || input.movement[1].abs() > 0.01
                            || input.movement[2].abs() > 0.01;
                        if has_manual {
                            st.autopilot_target_body_id = None;
                            st.pending_autopilot_cmd = Some((0xFFFFFFFF, 0));
                            info!("autopilot cancelled by manual input");
                        }
                    }

                    // Pilot mode: WASD → thrust.
                    // WASD cancels autopilot above, so this always applies.
                    {
                        // Enforce tier restriction inside atmosphere.
                        let in_atmosphere = if let Some(ref sys) = st.cached_system_params {
                            // Check altitude vs closest planet from scene_bodies.
                            st.scene_bodies.iter().any(|b| {
                                if b.body_id == 0 { return false; } // skip star
                                let pi = (b.body_id - 1) as usize;
                                if pi >= sys.planets.len() { return false; }
                                let alt = (st.ship_position - b.position).length() - sys.planets[pi].radius_m;
                                alt < sys.planets[pi].atmosphere.atmosphere_height
                                    && sys.planets[pi].atmosphere.has_atmosphere
                            })
                        } else { false };
                        let effective_tier = voxeldust_core::autopilot::effective_tier(
                            input.speed_tier, in_atmosphere, false);
                        let et = voxeldust_core::autopilot::engine_tier(effective_tier);
                        let tier_thrust = et.thrust_force_n * st.ship_props.thrust_multiplier;
                        st.pilot_thrust = DVec3::new(
                            input.movement[0] as f64 * tier_thrust,
                            input.movement[1] as f64 * tier_thrust,
                            -input.movement[2] as f64 * tier_thrust, // negated: W = forward = -Z
                        );
                        st.pilot_torque = DVec3::new(
                            input.look_pitch as f64,
                            input.look_yaw as f64,
                            0.0,
                        );

                        // Gravity compensation (coupled mode / hover-assist) in atmosphere.
                        // When no vertical input, auto-counters gravity so ship hovers.
                        // When no input at all, dampens velocity to bring ship to a stop.
                        // Skipped when engines are off (action == 5) — ship goes ballistic.
                        let engines_off = input.action == 5;
                        if in_atmosphere && st.autopilot_target_body_id.is_none() && !engines_off {
                            // Compute gravity from nearest planet.
                            if let Some(ref sys) = st.cached_system_params {
                                for b in &st.scene_bodies {
                                    if b.body_id == 0 { continue; }
                                    let pi = (b.body_id - 1) as usize;
                                    if pi >= sys.planets.len() { continue; }
                                    let dist = (st.ship_position - b.position).length();
                                    let alt = dist - sys.planets[pi].radius_m;
                                    if alt < sys.planets[pi].atmosphere.atmosphere_height {
                                        // Compute gravity acceleration toward planet.
                                        let grav_mag = sys.planets[pi].gm / (dist * dist);
                                        let grav_dir = (b.position - st.ship_position).normalize();
                                        let grav_world = grav_dir * grav_mag;

                                        // Hover thrust: negate gravity in ship-local frame.
                                        if input.movement[1].abs() < 0.01 {
                                            let grav_local = st.ship_rotation.inverse() * (-grav_world);
                                            let hover = grav_local * st.ship_props.mass_kg;
                                            st.pilot_thrust += DVec3::new(hover.x, hover.y, hover.z);
                                        }

                                        // Velocity damping when no movement input.
                                        let has_input = input.movement[0].abs() > 0.01
                                            || input.movement[1].abs() > 0.01
                                            || input.movement[2].abs() > 0.01;
                                        if !has_input {
                                            let max_accel = st.ship_props.engine_acceleration(effective_tier);
                                            let damping_rate = (max_accel * 0.1).min(5.0);
                                            let vel_local = st.ship_rotation.inverse() * st.ship_velocity;
                                            let damping = -vel_local * damping_rate * st.ship_props.mass_kg;
                                            let max_damp = tier_thrust * 0.5;
                                            let damping_clamped = damping.clamp_length_max(max_damp);
                                            st.pilot_thrust += DVec3::new(damping_clamped.x, damping_clamped.y, damping_clamped.z);
                                        }
                                        break; // only nearest planet
                                    }
                                }
                            }
                        }
                    }
                } else if let Some(handle) = st.player_body {
                    // Walking mode: apply velocity to Rapier body.
                    let (sin_y, cos_y) = st.player_yaw.sin_cos();
                    let fwd = Vec3::new(cos_y, 0.0, sin_y);
                    let right = Vec3::new(-sin_y, 0.0, cos_y);

                    let move_vel = fwd * input.movement[2] * WALK_SPEED
                        + right * input.movement[0] * WALK_SPEED;

                    if let Some(body) = st.rigid_body_set.get_mut(handle) {
                        let current_vel = *body.linvel();
                        body.set_linvel(vector![move_vel.x, current_vel.y, move_vel.z], true);

                        if input.jump && current_vel.y.abs() < 0.1 {
                            body.apply_impulse(vector![0.0, JUMP_IMPULSE, 0.0], true);
                        }
                    }
                }
            }
        });

        // System 3: Drain QUIC messages from system shard.
        let state_quic = state.clone();
        let client_reg_quic = client_registry.clone();
        let peer_reg_quic = peer_registry.clone();
        let quic_send_quic = quic_send_tx.clone();
        harness.add_system("drain_quic", move || {
            for _ in 0..32 { let queued = match quic_msg_rx.try_recv() { Ok(q) => q, Err(_) => break };
                let mut st = state_quic.lock().unwrap();
                match queued.msg {
                    ShardMsg::SystemSceneUpdate(data) => {
                        // Only accept scene updates from the authorized host shard.
                        if !st.authorized_peers.is_authorized(queued.source_shard_id) {
                            continue;
                        }
                        st.scene_bodies = data.bodies;
                        st.scene_lighting = Some(data.lighting);
                        st.game_time = data.game_time;
                        st.last_scene_update_tick = st.tick_count;
                    }
                    ShardMsg::ShipPositionUpdate(data) => {
                        // Only accept from the authorized host shard + for THIS ship.
                        if !st.authorized_peers.is_authorized(queued.source_shard_id) {
                            continue;
                        }
                        if data.ship_id == st.ship_id {
                            st.ship_position = data.position;
                            st.ship_velocity = data.velocity;
                            st.ship_rotation = data.rotation;
                            st.autopilot_snapshot = data.autopilot;

                            // Detect landed state: system shard zeros velocity on landing.
                            let was_landed = st.ship_landed;
                            st.ship_landed = data.velocity.length() < 0.5;

                            // Auto-close door on takeoff.
                            if was_landed && !st.ship_landed && st.door_open {
                                st.door_open = false;
                                let ShipInteriorState {
                                    ref mut collider_set, ref mut island_manager,
                                    ref mut rigid_body_set, ref mut door_blocker_handle, ..
                                } = *st;
                                if let Some(handle) = door_blocker_handle.take() {
                                    collider_set.remove(handle, island_manager, rigid_body_set, true);
                                }
                                let right_x = SHIP_WIDTH / 2.0 + WALL_THICKNESS;
                                let blocker = ColliderBuilder::cuboid(
                                    WALL_THICKNESS, 2.1 / 2.0, 0.6,
                                ).translation(vector![right_x, 2.1 / 2.0, 0.0]).build();
                                *door_blocker_handle = Some(collider_set.insert(blocker));
                                info!("door auto-closed on takeoff");
                            }
                        }
                    }
                    ShardMsg::HandoffAccepted(accepted) => {
                        // Planet shard accepted the player. Send ShardRedirect to client.
                        let session = accepted.session_token;
                        let target_shard = accepted.target_shard;
                        info!(session = session.0, target = target_shard.0,
                            "received HandoffAccepted, sending ShardRedirect to client");

                        // Look up target shard endpoint from peer registry.
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
                                            tracing::warn!(%e, "failed to send ShardRedirect");
                                        }
                                    }
                                    // Unregister AFTER the send completes so the TCP stream
                                    // isn't dropped mid-flight.
                                    if let Ok(mut reg) = cr.try_write() {
                                        reg.unregister(&session);
                                    }
                                });
                            }
                        }

                        // Remove player from interior.
                        st.connected_session = None;
                        st.connected_player_name = None;
                        st.handoff_pending = false;
                        st.is_piloting = false;
                    }
                    ShardMsg::PlayerHandoff(h) => {
                        // Player re-entering ship from planet.
                        info!(session = h.session_token.0, player = %h.player_name,
                            "player re-entering ship via handoff");
                        st.connected_session = Some(h.session_token);
                        st.connected_player_name = Some(h.player_name.clone());
                        st.handoff_pending = false;
                        st.is_piloting = false;

                        // Spawn player at EXIT_DOOR interior position.
                        if let Some(handle) = st.player_body {
                            if let Some(body) = st.rigid_body_set.get_mut(handle) {
                                body.set_translation(
                                    vector![EXIT_DOOR.x, EXIT_DOOR.y, EXIT_DOOR.z], true);
                                body.set_linvel(vector![0.0, 0.0, 0.0], true);
                            }
                        }
                        st.player_position = EXIT_DOOR;

                        // Send HandoffAccepted back to system shard.
                        let accepted_msg = ShardMsg::HandoffAccepted(handoff::HandoffAccepted {
                            session_token: h.session_token,
                            target_shard: st.shard_id,
                        });
                        if let Some(host_id) = st.host_shard_id {
                            if let Ok(reg) = peer_reg_quic.try_read() {
                                if let Some(addr) = reg.quic_addr(host_id) {
                                    let _ = quic_send_quic.send((host_id, addr, accepted_msg));
                                }
                            }
                        }
                    }
                    ShardMsg::HostSwitch(data) => {
                        if data.ship_id == st.ship_id {
                            let new_host = data.new_host_shard_id;
                            st.host_shard_id = Some(new_host);

                            // Authorize the new host shard by ID. Only scene data
                            // from this shard will be accepted — stale messages
                            // from the old host are rejected immediately.
                            st.authorized_peers.set_host(new_host);

                            // Clear stale scene data and set interstellar lighting.
                            st.scene_bodies.clear();
                            st.scene_lighting = Some(LightingInfoData {
                                sun_direction: DVec3::new(0.0, -1.0, 0.0),
                                sun_color: [0.5, 0.5, 0.6],
                                sun_intensity: 0.2,
                                ambient: 0.08,
                            });
                            st.last_scene_update_tick = st.tick_count;
                            st.host_switch_tick = st.tick_count;

                            // Warp complete — clear warp target when switching
                            // to a non-galaxy host (arrival at destination system).
                            if data.new_host_shard_type != 3 {
                                st.warp_target_star_index = None;
                            }

                            info!(
                                ship_id = st.ship_id,
                                new_host = new_host.0,
                                shard_type = data.new_host_shard_type,
                                quic_addr = %data.new_host_quic_addr,
                                "host switched — ship shard now reports to new host"
                            );

                            // Send ShardPreConnect to client so it opens a secondary
                            // UDP connection to the new host (dual-shard compositing).
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
                                if let Some(session) = st.connected_session {
                                    let cr = client_reg_quic.clone();
                                    tokio::spawn(async move {
                                        let reg = cr.read().await;
                                        let _ = reg.send_tcp(session, &pc).await;
                                    });
                                    info!("sent ShardPreConnect to client for secondary UDP");
                                }
                            } else {
                                warn!(
                                    ship_id = st.ship_id,
                                    shard_type = data.new_host_shard_type,
                                    "HostSwitch has empty UDP address — ShardPreConnect NOT sent"
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        });

        // System 3b: Send ShardPreConnect when ship enters a planet's atmosphere.
        // Triggers early so the client can pre-connect to the planet shard and start
        // receiving WorldState (surface data, lighting) before landing. This gives
        // maximum pre-loading time for future terrain LOD systems.
        let state_preconnect = state.clone();
        let peer_reg_preconnect = peer_registry.clone();
        let client_reg_preconnect = client_registry.clone();
        harness.add_system("preconnect_check", move || {
            let mut st = state_preconnect.lock().unwrap();
            if st.preconnect_sent || st.connected_session.is_none() { return; }
            if st.system_seed == 0 { return; }

            // Check if ship is inside any planet's atmosphere.
            let sys = match &st.cached_system_params {
                Some(sys) => sys,
                None => return,
            };
            let in_atmo_planet = st.scene_bodies.iter()
                .filter(|b| b.body_id > 0)
                .find_map(|b| {
                    let pi = (b.body_id - 1) as usize;
                    if pi >= sys.planets.len() { return None; }
                    let alt = (st.ship_position - b.position).length() - sys.planets[pi].radius_m;
                    if alt < sys.planets[pi].atmosphere.atmosphere_height
                        && sys.planets[pi].atmosphere.has_atmosphere {
                        Some(pi)
                    } else {
                        None
                    }
                });

            // Track atmosphere transitions to reset preconnect on exit.
            let was_in_atmo = st.ship_in_atmosphere;
            st.ship_in_atmosphere = in_atmo_planet.is_some();
            if was_in_atmo && !st.ship_in_atmosphere {
                st.preconnect_sent = false; // Reset on atmosphere exit
            }

            let planet_index = match in_atmo_planet {
                Some(idx) => idx,
                None => return,
            };

            let planet_seed = match &st.cached_system_params {
                Some(sys) if planet_index < sys.planets.len() => sys.planets[planet_index].planet_seed,
                _ => return,
            };

            // Look up planet shard TCP/UDP from peer registry.
            let planet_shard_info = if let Ok(reg) = peer_reg_preconnect.try_read() {
                reg.find_by_type(ShardType::Planet).iter()
                    .find(|s| s.planet_seed == Some(planet_seed))
                    .map(|s| (s.endpoint.tcp_addr.to_string(), s.endpoint.udp_addr.to_string()))
            } else {
                None
            };

            let (tcp_addr, udp_addr) = match planet_shard_info {
                Some(info) => info,
                None => return, // Planet shard not yet provisioned
            };

            let session = st.connected_session.unwrap();
            let pc = voxeldust_core::handoff::ShardPreConnect {
                shard_type: 0, // Planet
                tcp_addr: tcp_addr.clone(),
                udp_addr: udp_addr.clone(),
                seed: planet_seed,
                planet_index: planet_index as u32,
                reference_position: st.ship_position,
                reference_rotation: DQuat::IDENTITY,
            };

            st.preconnect_sent = true;

            let cr = client_reg_preconnect.clone();
            let msg = ServerMsg::ShardPreConnect(pc);
            tokio::spawn(async move {
                if let Ok(reg) = cr.try_read() {
                    if let Err(e) = reg.send_tcp(session, &msg).await {
                        tracing::warn!(%e, "failed to send ShardPreConnect");
                    }
                }
            });

            info!(planet_seed, planet_index, "sent ShardPreConnect to client");
        });

        // System 4: Interaction — pilot seat + exit door.
        let state_interact = state.clone();
        harness.add_system("interaction", move || {
            let mut st = state_interact.lock().unwrap();
            // Action key pressed this tick (rising edge: action == 3, prev != 3).
            let action_pressed = st.last_action == 3 && st.prev_action != 3;

            if action_pressed {
                let pos = st.player_position;
                let dist_to_seat = (pos - PILOT_SEAT).length();
                let dist_to_exit = (pos - EXIT_DOOR).length();

                if dist_to_seat < INTERACT_DIST {
                    st.is_piloting = !st.is_piloting;
                    if st.is_piloting {
                        // Lock player in seat.
                        if let Some(handle) = st.player_body {
                            if let Some(body) = st.rigid_body_set.get_mut(handle) {
                                body.set_linvel(vector![0.0, 0.0, 0.0], true);
                            }
                        }
                        info!("entered pilot mode");
                    } else {
                        info!("exited pilot mode");
                    }
                } else if dist_to_exit < INTERACT_DIST {
                    // Exit door: toggle open/closed (only when landed).
                    if st.ship_landed {
                        st.door_open = !st.door_open;
                        let opening = st.door_open;
                        let ShipInteriorState {
                            ref mut collider_set, ref mut island_manager,
                            ref mut rigid_body_set, ref mut door_blocker_handle, ..
                        } = *st;
                        if opening {
                            if let Some(handle) = door_blocker_handle.take() {
                                collider_set.remove(handle, island_manager, rigid_body_set, true);
                            }
                            info!("exit door opened");
                        } else {
                            let right_x = SHIP_WIDTH / 2.0 + WALL_THICKNESS;
                            let blocker = ColliderBuilder::cuboid(
                                WALL_THICKNESS, 2.1 / 2.0, 0.6,
                            ).translation(vector![right_x, 2.1 / 2.0, 0.0]).build();
                            *door_blocker_handle = Some(collider_set.insert(blocker));
                            info!("exit door closed");
                        }
                    }
                }
            }
        });

        // System 4b: Door threshold — auto-handoff when player walks outside.
        let state_threshold = state.clone();
        harness.add_system("door_threshold", move || {
            let mut st = state_threshold.lock().unwrap();
            if !st.door_open || st.handoff_pending { return; }

            // Player crossed outside: ship-local X > SHIP_WIDTH/2 + margin.
            let threshold_x = SHIP_WIDTH / 2.0 + 0.5;
            if st.player_position.x <= threshold_x { return; }

            // Auto-trigger handoff to planet shard (same logic as old E-key exit).
            let (session, player_name) = match (st.connected_session, st.connected_player_name.clone()) {
                (Some(s), Some(n)) => (s, n),
                _ => return,
            };

            let player_local = DVec3::new(
                st.player_position.x as f64, st.player_position.y as f64, st.player_position.z as f64,
            );
            let player_system_pos = st.ship_position + st.ship_rotation * player_local;

            let closest_planet = st.scene_bodies.iter()
                .filter(|b| b.body_id > 0)
                .min_by_key(|b| ((b.position - st.ship_position).length() * 1000.0) as u64);

            if let Some(planet_body) = closest_planet {
                let planet_index = (planet_body.body_id - 1) as usize;
                if let Some(ref sys) = st.cached_system_params {
                    if planet_index < sys.planets.len() {
                        let planet_seed = sys.planets[planet_index].planet_seed;
                        let h = handoff::PlayerHandoff {
                            session_token: session,
                            player_name,
                            position: player_system_pos,
                            velocity: st.ship_velocity,
                            rotation: DQuat::IDENTITY,
                            forward: st.ship_rotation * DVec3::NEG_Z,
                            fly_mode: false,
                            speed_tier: 0,
                            grounded: false,
                            health: 100.0,
                            shield: 100.0,
                            source_shard: st.shard_id,
                            source_tick: st.tick_count,
                            target_star_index: None,
                            galaxy_context: None,
                            target_planet_seed: Some(planet_seed),
                            target_planet_index: Some(planet_index as u32),
                            target_ship_id: None,
                            target_ship_shard_id: None,
                            ship_system_position: Some(st.ship_position),
                            ship_rotation: Some(st.ship_rotation),
                            game_time: st.game_time,
                            warp_target_star_index: None,
                            warp_velocity_gu: None,
                        };
                        st.handoff_pending = true;
                        st.pending_handoff_msg = Some(ShardMsg::PlayerHandoff(h));
                        info!(planet_index, planet_seed, "threshold crossing — auto-handoff initiated");
                    }
                }
            }
        });

        // System 5: Send messages to system shard via QUIC.
        // Handoff and autopilot messages are sent regardless of piloting state.
        // Thrust/torque is only sent when piloting.
        let state_pilot = state.clone();
        let peer_reg = peer_registry.clone();
        harness.add_system("pilot_send", move || {
            let mut st = state_pilot.lock().unwrap();

            if let Some(host_id) = st.host_shard_id {
                if let Ok(reg) = peer_reg.try_read() {
                    if let Some(addr) = reg.quic_addr(host_id) {
                        // Always send pending handoff (works when walking to exit door).
                        if let Some(handoff_msg) = st.pending_handoff_msg.take() {
                            let _ = quic_send_tx.try_send((host_id, addr, handoff_msg));
                        }

                        // Always send pending autopilot command.
                        if let Some((target, tier)) = st.pending_autopilot_cmd.take() {
                            let ap_msg = ShardMsg::AutopilotCommand(AutopilotCommandData {
                                ship_id: st.ship_id,
                                target_body_id: target,
                                speed_tier: tier,
                                autopilot_mode: 0,
                            });
                            let _ = quic_send_tx.try_send((host_id, addr, ap_msg));
                        }

                        // Always send pending warp command.
                        if let Some(target_star) = st.pending_warp_cmd.take() {
                            let warp_msg = ShardMsg::WarpAutopilotCommand(
                                voxeldust_core::shard_message::WarpAutopilotCommandData {
                                    ship_id: st.ship_id,
                                    target_star_index: target_star,
                                    galaxy_seed: st.galaxy_seed,
                                },
                            );
                            let _ = quic_send_tx.try_send((host_id, addr, warp_msg));
                        }

                        // Thrust/torque only when piloting.
                        if st.is_piloting {
                            let msg = ShardMsg::ShipControlInput(ShipControlInput {
                                ship_id: st.ship_id,
                                thrust: st.pilot_thrust,
                                torque: st.pilot_torque,
                                braking: false,
                                tick: st.tick_count,
                            });
                            let _ = quic_send_tx.try_send((host_id, addr, msg));
                        }
                    } else if st.tick_count % 100 == 0 {
                        let all_peers: Vec<_> = reg.all().iter().map(|s| format!("{}({})", s.id, s.shard_type)).collect();
                        info!(host = host_id.0, peers = ?all_peers, "host shard not found in peer registry");
                    }
                } else if st.tick_count % 100 == 0 {
                    info!("peer_reg lock failed");
                }
            } else if st.tick_count % 100 == 0 {
                info!("no host_shard_id");
            }

            // Don't zero thrust/torque — keep last input alive until new input arrives.
            // drain_input sets new values each time a UDP packet arrives from the client.
        });

        // System 6: Physics step.
        let state_physics = state.clone();
        harness.add_system("physics_step", move || {
            let mut st = state_physics.lock().unwrap();
            st.tick_count += 1;
            st.step();
        });

        // System 7: Broadcast WorldState to client.
        let state_broadcast = state.clone();
        harness.add_system("broadcast", move || {
            let st = state_broadcast.lock().unwrap();
            let ws = ServerMsg::WorldState(st.build_world_state());
            let _ = broadcast_tx.try_send(ws);
        });

        // System 8: Periodic log.
        let state_log = state.clone();
        harness.add_system("log_state", move || {
            let st = state_log.lock().unwrap();
            if st.tick_count % 100 == 0 {
                info!(
                    pos = format!("({:.1}, {:.1}, {:.1})", st.player_position.x, st.player_position.y, st.player_position.z),
                    piloting = st.is_piloting,
                    gravity = st.gravity_enabled,
                    tick = st.tick_count,
                    "ship state"
                );
            }
        });

        harness.run().await;
    });
}
