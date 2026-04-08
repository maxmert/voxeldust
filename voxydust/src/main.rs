//! Voxydust — first-person game client with server-authoritative movement.
//!
//! Game state lives in bevy_ecs Resources, updated by ECS systems each frame.
//! GPU state (GpuState, Window, uniform_data) stays outside the World because
//! egui_winit::State is !Send.

mod block_render;
mod camera;
mod events;
mod gpu;
mod hud;
mod input;
mod mesh;
mod network;
mod render;
mod stars;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use clap::Parser;
use glam::{DQuat, DVec3};
use tokio::sync::mpsc;
use tracing::info;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use voxeldust_core::autopilot::{self, AutopilotMode, TrajectoryPlan};
use voxeldust_core::client_message::{PlayerInputData, WorldStateData};
use voxeldust_core::shard_message::AutopilotSnapshotData;
use voxeldust_core::system::SystemParams;

use crate::gpu::{GpuState, ObjectUniforms, MAX_OBJECTS};
use crate::network::NetEvent;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "voxydust", about = "Voxydust game client")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:7777")]
    gateway: SocketAddr,
    #[arg(long, default_value = "Player")]
    name: String,
    #[arg(long)]
    direct: Option<String>,
}

// ---------------------------------------------------------------------------
// ECS Resources
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct NetworkChannels {
    event_rx: Option<mpsc::UnboundedReceiver<NetEvent>>,
    input_tx: Option<mpsc::UnboundedSender<PlayerInputData>>,
    block_edit_tx: Option<mpsc::UnboundedSender<voxeldust_core::client_message::BlockEditData>>,
    /// TCP outbound channel for reliable messages (block edits, sub-block edits, config updates).
    /// Packets sent here are written to the TCP stream by the network task.
    tcp_out_tx: Option<mpsc::UnboundedSender<Vec<u8>>>,
}

/// Block type selection hotbar. Keys 1-9 select the active slot (when not piloting).
#[derive(Resource)]
struct BlockHotbar {
    selected_slot: usize,
    slots: [voxeldust_core::block::BlockId; 9],
}

impl Default for BlockHotbar {
    fn default() -> Self {
        use voxeldust_core::block::BlockId;
        Self {
            selected_slot: 0,
            slots: [
                BlockId::HULL_STANDARD,
                BlockId::HULL_LIGHT,
                BlockId::HULL_HEAVY,
                BlockId::HULL_ARMORED,
                BlockId::WINDOW,
                BlockId::STONE,
                BlockId::DIRT,
                BlockId::SAND,
                BlockId::GRASS,
            ],
        }
    }
}

impl BlockHotbar {
    fn selected_block(&self) -> voxeldust_core::block::BlockId {
        self.slots[self.selected_slot]
    }
}

/// Cached block registry — built once at startup, not per frame.
#[derive(Resource)]
struct ClientBlockRegistry(voxeldust_core::block::BlockRegistry);

/// Currently targeted block (client-side prediction raycast, updated each frame).
#[derive(Resource, Default)]
struct BlockTarget {
    hit: Option<voxeldust_core::block::raycast::BlockHit>,
}

/// Sub-block placement tool state. When active, clicks place/remove sub-block elements
/// instead of full blocks. Toggled with Tab key.
#[derive(Resource)]
struct SubBlockTool {
    /// Whether sub-block placement mode is active (vs. normal block mode).
    active: bool,
    /// Currently selected sub-block type.
    selected_type: voxeldust_core::block::sub_block::SubBlockType,
    /// Rotation on face (0-3).
    rotation: u8,
}

impl Default for SubBlockTool {
    fn default() -> Self {
        Self {
            active: false,
            selected_type: voxeldust_core::block::sub_block::SubBlockType::PowerWire,
            rotation: 0,
        }
    }
}

/// Client-side index of functional block positions. Updated only when chunks change.
/// Maps world position → (kind, display letter). Typically <100 entries on a ship.
#[derive(Resource, Default)]
struct ClientBlockIndex {
    entries: Vec<(glam::Vec3, u8, char)>,
}

impl ClientBlockIndex {
    /// Rebuild the index by scanning all loaded chunks for functional blocks.
    /// Called only when chunks are dirty (new snapshot or delta applied).
    fn rebuild(
        &mut self,
        cache: &voxeldust_core::block::client_chunks::ClientChunkCache,
        source: voxeldust_core::block::client_chunks::ChunkSourceId,
        registry: &voxeldust_core::block::BlockRegistry,
    ) {
        self.entries.clear();
        let chunk_keys: Vec<_> = cache.source_chunk_keys(source).collect();
        for chunk_key in chunk_keys {
            let chunk = match cache.get_chunk(source, chunk_key) {
                Some(c) => c,
                None => continue,
            };
            for bx in 0u8..62 {
                for by in 0u8..62 {
                    for bz in 0u8..62 {
                        let block_id = chunk.get_block(bx, by, bz);
                        if block_id.is_air() { continue; }
                        if let Some(kind) = registry.functional_kind(block_id) {
                            let wx = chunk_key.x * 62 + bx as i32;
                            let wy = chunk_key.y * 62 + by as i32;
                            let wz = chunk_key.z * 62 + bz as i32;
                            let pos = glam::Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
                            let letter = match kind {
                                voxeldust_core::block::FunctionalBlockKind::Thruster => 'T',
                                voxeldust_core::block::FunctionalBlockKind::Reactor => 'R',
                                voxeldust_core::block::FunctionalBlockKind::Seat => 'S',
                                voxeldust_core::block::FunctionalBlockKind::Battery => 'B',
                                voxeldust_core::block::FunctionalBlockKind::SignalConverter => 'C',
                                voxeldust_core::block::FunctionalBlockKind::Antenna => 'A',
                                _ => 'F',
                            };
                            self.entries.push((pos, kind as u8, letter));
                        }
                    }
                }
            }
        }
    }
}

/// AR interaction phase: animating in, interactive, animating out.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
enum ConfigPanelPhase {
    #[default]
    Closed,
    /// Camera is lerping toward the block face.
    AnimatingIn { progress: f32 },
    /// Camera is at the block, cursor is free, UI is interactive.
    Interactive,
    /// Camera is lerping back to where the player was looking.
    AnimatingOut { progress: f32 },
}

/// State for the block signal config UI panel (SC-style AR interaction).
#[derive(Resource, Default)]
struct BlockConfigUIState {
    /// Currently open config (None = UI closed).
    open_config: Option<voxeldust_core::signal::config::BlockSignalConfig>,
    /// Whether the config panel is open (suppresses game input).
    is_open: bool,
    /// Current AR interaction phase.
    phase: ConfigPanelPhase,
    /// Saved camera yaw/pitch before entering config mode (ship-local).
    saved_yaw: f64,
    saved_pitch: f64,
    /// Target yaw/pitch to face the block.
    target_yaw: f64,
    target_pitch: f64,
    /// Block world position (ship-local) being configured.
    block_world_pos: glam::IVec3,
    /// Direction from eye to block (ship-local, normalized). Used for camera zoom offset.
    zoom_direction: DVec3,
    /// Distance from eye to block center.
    zoom_distance: f64,
}

/// Camera position offset applied during config panel zoom animation.
/// Ship-local offset added to the render position before camera computation.
#[derive(Resource, Default)]
struct CameraZoomOffset {
    offset: DVec3,
}

const CONFIG_ANIM_DURATION: f32 = 0.35; // seconds for camera transition

#[derive(Resource)]
struct ConnectionInfo {
    connected: bool,
    current_shard_type: u8,
    reference_position: DVec3,
    reference_rotation: DQuat,
    secondary_shard_type: Option<u8>,
    system_seed: u64,
    galaxy_seed: u64,
}

#[derive(Resource)]
struct ServerWorldState {
    latest: Option<WorldStateData>,
    secondary: Option<WorldStateData>,
}

/// Client-side chunk cache for block data received from the server.
/// Multi-source: supports simultaneous ship + planet chunks during transitions.
#[derive(Resource)]
struct ClientChunkCacheRes {
    cache: voxeldust_core::block::client_chunks::ClientChunkCache,
    /// Source ID for the current primary shard connection.
    /// Set when a new shard connection provides chunk data.
    primary_source: Option<voxeldust_core::block::client_chunks::ChunkSourceId>,
}

impl std::ops::Deref for ClientChunkCacheRes {
    type Target = voxeldust_core::block::client_chunks::ClientChunkCache;
    fn deref(&self) -> &Self::Target { &self.cache }
}

impl std::ops::DerefMut for ClientChunkCacheRes {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.cache }
}

#[derive(Resource)]
struct LocalPlayer {
    position: DVec3,
    velocity: DVec3,
    is_piloting: bool,
    ship_rotation: DQuat,
}

#[derive(Resource)]
struct RenderSmoothing {
    // --- Walking ---
    render_position: DVec3,
    prev_server_position: DVec3,
    walk_velocity: DVec3,
    last_server_update_time: Instant,
    has_prev_server_pos: bool,

    // --- Piloting ---
    render_rotation: DQuat,
    ship_velocity: DVec3,

    // --- Galaxy warp ---
    galaxy_render_position: DVec3,
    galaxy_render_rotation: DQuat,
    galaxy_velocity: DVec3,
    prev_galaxy_position: DVec3,
    prev_galaxy_rotation: DQuat,
    last_galaxy_update_time: Instant,
    has_prev_galaxy_pos: bool,
}

#[derive(Resource)]
struct CameraControl {
    yaw: f64,
    pitch: f64,
    pilot_yaw_rate: f64,
    pilot_pitch_rate: f64,
}

#[derive(Resource)]
struct KeyboardState {
    keys_held: HashSet<KeyCode>,
    mouse_grabbed: bool,
}

#[derive(Resource)]
struct FlightControl {
    selected_thrust_tier: u8,
    engines_off: bool,
    /// Thrust limiter (0.0–1.0), adjusted by mouse wheel.
    thrust_limiter: f32,
}

#[derive(Resource)]
struct ClientAutopilot {
    target: Option<usize>,
    trajectory_plan: Option<TrajectoryPlan>,
    server_autopilot: Option<AutopilotSnapshotData>,
    mode: AutopilotMode,
    last_t_press: Option<Instant>,
}

/// Background trajectory planner — offloads expensive plan_trajectory() to a
/// dedicated thread so it never blocks the render loop. Up to 10,000 physics
/// sim steps with Lambert solver iterations run off-thread; the main thread
/// just polls for the latest result each frame.
#[derive(Resource)]
struct TrajectoryWorker {
    /// Send a computation request to the background thread.
    request_tx: std::sync::mpsc::Sender<TrajectoryRequest>,
    /// Receive completed trajectory plans.
    /// Wrapped in Mutex because std::sync::mpsc::Receiver is !Sync (required by Resource).
    /// Only the main thread ever locks this, so contention is zero.
    result_rx: std::sync::Mutex<std::sync::mpsc::Receiver<Option<TrajectoryPlan>>>,
    /// Sequence number of the last submitted request (prevents stale results).
    last_submitted: u64,
    /// Sequence number of the last received result.
    last_received: u64,
}

/// Data needed by the background thread to compute a trajectory.
struct TrajectoryRequest {
    seq: u64,
    ship_pos: DVec3,
    ship_vel: DVec3,
    target_planet_index: usize,
    system: SystemParams,
    game_time: f64,
    ship_props: autopilot::ShipPhysicalProperties,
    thrust_tier: u8,
    sample_count: usize,
    seed: Option<autopilot::AutopilotSeed>,
}

impl TrajectoryWorker {
    fn spawn() -> Self {
        let (request_tx, request_rx) = std::sync::mpsc::channel::<TrajectoryRequest>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<Option<TrajectoryPlan>>();

        std::thread::Builder::new()
            .name("trajectory-planner".into())
            .spawn(move || {
                while let Ok(req) = request_rx.recv() {
                    let plan = if let Some(seed) = req.seed {
                        autopilot::plan_trajectory_seeded(
                            req.ship_pos,
                            req.ship_vel,
                            req.target_planet_index,
                            &req.system,
                            req.game_time,
                            &req.ship_props,
                            req.thrust_tier,
                            req.sample_count,
                            &seed,
                        )
                    } else {
                        autopilot::plan_trajectory(
                            req.ship_pos,
                            req.ship_vel,
                            req.target_planet_index,
                            &req.system,
                            req.game_time,
                            &req.ship_props,
                            req.thrust_tier,
                            req.sample_count,
                        )
                    };
                    // If the main thread dropped its receiver, exit gracefully.
                    if result_tx.send(plan).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn trajectory planner thread");

        Self {
            request_tx,
            result_rx: std::sync::Mutex::new(result_rx),
            last_submitted: 0,
            last_received: 0,
        }
    }
}

#[derive(Resource)]
struct ClientWarp {
    target_star_index: Option<u32>,
    galaxy_position: Option<DVec3>,
    galaxy_rotation: Option<DQuat>,
    prev_g_pressed: bool,
}

#[derive(Resource)]
struct StarFieldRes(Option<stars::StarField>);

#[derive(Resource)]
struct SystemParamsCache(Option<SystemParams>);

#[derive(Resource)]
struct FrameTime {
    count: u64,
    last_time: Instant,
    dt: f64,
}

// ---------------------------------------------------------------------------
// System Sets
// ---------------------------------------------------------------------------

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum ClientSet {
    FrameTime,
    Network,
    Input,
    Smooth,
    AutopilotTimeout,
    Trajectory,
}

// ---------------------------------------------------------------------------
// ECS Systems
// ---------------------------------------------------------------------------

fn update_frame_time(mut ft: ResMut<FrameTime>) {
    let now = Instant::now();
    ft.dt = (now - ft.last_time).as_secs_f64();
    ft.last_time = now;
    ft.count += 1;
}

fn poll_network(
    mut net: ResMut<NetworkChannels>,
    mut conn: ResMut<ConnectionInfo>,
    mut ws: ResMut<ServerWorldState>,
    mut player: ResMut<LocalPlayer>,
    mut smooth: ResMut<RenderSmoothing>,
    mut cam: ResMut<CameraControl>,
    mut ap: ResMut<ClientAutopilot>,
    mut warp: ResMut<ClientWarp>,
    mut sf_res: ResMut<StarFieldRes>,
    mut sys_params: ResMut<SystemParamsCache>,
    mut chunk_cache: ResMut<ClientChunkCacheRes>,
    ft: Res<FrameTime>,
    mut config_open_msgs: MessageWriter<events::ConfigPanelOpenMsg>,
) {
    let rx = match net.event_rx.as_mut() {
        Some(rx) => rx,
        None => return,
    };
    while let Ok(event) = rx.try_recv() {
        match event {
            NetEvent::WorldState(mut world_state) => {
                // During warp the galaxy shard is the authority for the
                // scene — the ship shard should send empty bodies.  Stale
                // SystemSceneUpdate messages from the old system shard can
                // leak through QUIC queues and briefly restore celestial
                // bodies. Drop them on the client so they never render.
                if warp.galaxy_position.is_some() {
                    world_state.bodies.clear();
                }

                // Server-authoritative warp target: ship shard tells client
                // which star is selected (0xFFFFFFFF = none).
                if world_state.warp_target_star_index != 0xFFFFFFFF {
                    warp.target_star_index = Some(world_state.warp_target_star_index);
                } else if warp.galaxy_position.is_none() {
                    // Only clear target when not in warp (during warp,
                    // the galaxy shard doesn't send this field).
                    warp.target_star_index = None;
                }

                // Update player position from server.
                if let Some(p) = world_state.players.first() {
                    // Derive walking velocity from position deltas for
                    // smooth inter-tick extrapolation on the client.
                    let new_pos = p.position;
                    let now_ws = Instant::now();
                    if smooth.has_prev_server_pos {
                        let elapsed = (now_ws - smooth.last_server_update_time).as_secs_f64();
                        if elapsed > 0.01 && elapsed < 0.15 {
                            smooth.walk_velocity = (new_pos - smooth.prev_server_position) / elapsed;
                        } else {
                            smooth.walk_velocity = DVec3::ZERO;
                        }
                    } else {
                        smooth.walk_velocity = DVec3::ZERO;
                        smooth.has_prev_server_pos = true;
                    }
                    smooth.prev_server_position = new_pos;
                    smooth.last_server_update_time = now_ws;
                    // Only snap render_position on large corrections (teleport/shard change).
                    // Small corrections are handled by per-frame lerp for smoothness.
                    if (new_pos - smooth.render_position).length() > 2.0 {
                        smooth.render_position = new_pos;
                    }

                    player.position = new_pos;
                    player.velocity = p.velocity;
                    let was_piloting = player.is_piloting;
                    player.is_piloting = !p.grounded;

                    if was_piloting != player.is_piloting {
                        info!(piloting = player.is_piloting, grounded = p.grounded, frame = ft.count, "pilot mode changed");
                    }

                    // Always update ship_rotation on a ship shard — needed for
                    // correct floating-origin rendering of celestial bodies
                    // even while walking inside the ship.
                    if conn.current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
                        player.ship_rotation = p.rotation;
                    }

                    // Piloting smoothing: use server velocity for extrapolation,
                    // snap on large corrections (shard change, warp arrival).
                    if player.is_piloting {
                        smooth.ship_velocity = p.velocity;
                        if (new_pos - smooth.render_position).length() > 50.0 {
                            smooth.render_position = new_pos;
                            smooth.render_rotation = p.rotation;
                        }
                    }

                    if player.is_piloting && !was_piloting {
                        // Walk → Pilot: snap rotation and velocity to prevent
                        // stale walking state from bleeding into piloting.
                        smooth.render_rotation = p.rotation;
                        smooth.ship_velocity = p.velocity;
                        // Sync camera yaw to ship heading so there's no visual snap.
                        let fwd = p.rotation * DVec3::NEG_Z;
                        cam.yaw = fwd.z.atan2(fwd.x) as f64;
                        cam.pitch = fwd.y.asin() as f64;
                    }

                    // Pilot → walk transition: reset camera to ship-local forward.
                    // Autopilot continues running on the system shard — preserve
                    // local tracking so the HUD can display it when re-entering.
                    if was_piloting && !player.is_piloting {
                        cam.yaw = -std::f64::consts::FRAC_PI_2;
                        cam.pitch = 0.0;
                        smooth.has_prev_server_pos = false;
                        smooth.walk_velocity = DVec3::ZERO;
                    }
                }
                // Extract server-authoritative autopilot state.
                ap.server_autopilot = world_state.autopilot.clone();
                if let Some(ref autopilot_snap) = ap.server_autopilot {
                    if autopilot_snap.target_planet_index != 0xFFFFFFFF {
                        ap.target = Some(autopilot_snap.target_planet_index as usize);
                    }
                }
                if ws.latest.is_none() {
                    info!(bodies = world_state.bodies.len(), tick = world_state.tick, "first WorldState");
                }
                ws.latest = Some(world_state);
            }
            NetEvent::Connected { shard_type, reference_position, reference_rotation, system_seed, galaxy_seed, .. } => {
                conn.current_shard_type = shard_type;
                conn.reference_position = reference_position;
                conn.reference_rotation = reference_rotation;
                if shard_type == voxeldust_core::client_message::shard_type::SHIP {
                    player.ship_rotation = reference_rotation;
                }
                conn.connected = true;
                conn.system_seed = system_seed;
                conn.galaxy_seed = galaxy_seed;
                // Reset all smoothing — coordinate systems differ between shards.
                smooth.has_prev_server_pos = false;
                smooth.walk_velocity = DVec3::ZERO;
                smooth.ship_velocity = DVec3::ZERO;
                smooth.render_rotation = reference_rotation;
                smooth.has_prev_galaxy_pos = false;
                smooth.galaxy_velocity = DVec3::ZERO;

                // Clear warp state on any new shard connection.
                warp.galaxy_position = None;
                warp.galaxy_rotation = None;
                warp.target_star_index = None;
                if system_seed > 0 {
                    sys_params.0 = Some(SystemParams::from_seed(system_seed));
                }
                // Initialize star field from galaxy seed (deterministic, no network needed).
                if galaxy_seed > 0 && sf_res.0.is_none() {
                    info!(galaxy_seed, "generating star catalog");
                    sf_res.0 = Some(stars::StarField::from_galaxy_seed(galaxy_seed, system_seed));
                    info!(stars = sf_res.0.as_ref().unwrap().catalog.len(), "star field ready");
                }
                // Restore star field's current star exclusion after warp arrival.
                if system_seed > 0 {
                    if let Some(ref mut sf) = sf_res.0 {
                        sf.set_current_system_seed(system_seed);
                    }
                }
                let shard_name = match shard_type { 0 => "Planet", 1 => "System", 2 => "Ship", 3 => "Galaxy", _ => "?" };
                info!(shard_name, "connected to shard");
            }
            NetEvent::Disconnected(reason) => {
                info!(%reason, "disconnected");
                conn.connected = false;
            }
            NetEvent::SecondaryConnected { shard_type, seed, .. } => {
                let shard_name = match shard_type { 0 => "Planet", 2 => "Ship", 3 => "Galaxy", _ => "?" };
                info!(shard_name, seed, shard_type, "secondary shard connected for dual compositing");
                conn.secondary_shard_type = Some(shard_type);

                if shard_type == 3 {
                    // Galaxy secondary: entering warp. Keep current_star_index
                    // until the first GalaxyWorldState arrives so the star field
                    // reference position stays correct during the gap.
                } else if warp.galaxy_position.is_some() {
                    // Non-galaxy secondary while warp is active: warp has ended,
                    // ship arrived at destination system. Clear warp state so the
                    // star field returns to skybox mode and bodies render normally.
                    info!("warp ended — arrived at destination system");
                    warp.galaxy_position = None;
                    warp.galaxy_rotation = None;
                    warp.target_star_index = None;
                    if seed > 0 {
                        if let Some(ref mut sf) = sf_res.0 {
                            sf.set_current_system_seed(seed);
                        }
                    }
                }
            }
            NetEvent::SecondaryWorldState(secondary_ws) => {
                ws.secondary = Some(secondary_ws);
            }
            NetEvent::GalaxyWorldState(gws) => {
                // Only process while a galaxy secondary is the active type.
                // After arrival, SecondaryConnected(type=1) changes the type
                // to System, but stale GalaxyWorldState events may still be
                // in the channel from before the galaxy secondary was cancelled.
                if conn.secondary_shard_type != Some(3) {
                    continue;
                }

                // Galaxy smoothing: derive velocity from position deltas.
                let now_gws = Instant::now();
                if smooth.has_prev_galaxy_pos {
                    let elapsed = (now_gws - smooth.last_galaxy_update_time).as_secs_f64();
                    if elapsed > 0.01 && elapsed < 0.15 {
                        smooth.galaxy_velocity = (gws.ship_position - smooth.prev_galaxy_position) / elapsed;
                    }
                    // Snap on large corrections (warp phase transitions).
                    if (gws.ship_position - smooth.galaxy_render_position).length() > 1e6 {
                        smooth.galaxy_render_position = gws.ship_position;
                        smooth.galaxy_render_rotation = gws.ship_rotation;
                    }
                } else {
                    // First galaxy update: snap render state.
                    smooth.galaxy_render_position = gws.ship_position;
                    smooth.galaxy_render_rotation = gws.ship_rotation;
                    smooth.galaxy_velocity = DVec3::ZERO;
                    smooth.has_prev_galaxy_pos = true;
                }
                smooth.prev_galaxy_position = gws.ship_position;
                smooth.prev_galaxy_rotation = gws.ship_rotation;
                smooth.last_galaxy_update_time = now_gws;

                let was_none = warp.galaxy_position.is_none();
                warp.galaxy_position = Some(gws.ship_position);
                warp.galaxy_rotation = Some(gws.ship_rotation);
                if was_none {
                    info!(
                        pos = format!("({:.1},{:.1},{:.1})", gws.ship_position.x, gws.ship_position.y, gws.ship_position.z),
                        "first GalaxyWorldState received — switching to galaxy star mode"
                    );
                    // Now that galaxy mode is active, stop excluding the departure
                    // star so it appears as a dot in the galaxy-scale star field.
                    if let Some(ref mut sf) = sf_res.0 {
                        sf.current_star_index = None;
                    }
                }
            }
            NetEvent::BlockConfigState(config) => {
                info!(block = ?config.block_pos, "received BlockConfigState — writing ConfigPanelOpenMsg");
                config_open_msgs.write(events::ConfigPanelOpenMsg { config });
            }
            NetEvent::ChunkSnapshot(cs) => {
                // Ensure we have a source for the current primary shard.
                if chunk_cache.primary_source.is_none() {
                    let new_source = chunk_cache.cache.add_source();
                    chunk_cache.primary_source = Some(new_source);
                    info!(source = new_source.0, "created chunk source for primary shard");
                }
                let source = chunk_cache.primary_source.unwrap();
                let chunk_pos = glam::IVec3::new(cs.chunk_x, cs.chunk_y, cs.chunk_z);
                match chunk_cache.insert_snapshot(source, chunk_pos, cs.seq, &cs.data) {
                    Ok(()) => {
                        info!(
                            chunk = %chunk_pos,
                            data_len = cs.data.len(),
                            total_chunks = chunk_cache.total_chunk_count(),
                            "received chunk snapshot"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(%e, "failed to insert chunk snapshot");
                    }
                }
            }
            NetEvent::ChunkDelta(cd) => {
                if let Some(source) = chunk_cache.primary_source {
                    let chunk_pos = glam::IVec3::new(cd.chunk_x, cd.chunk_y, cd.chunk_z);
                    // Apply block mods.
                    if !cd.mods.is_empty() {
                        let edits: Vec<_> = cd.mods.iter()
                            .map(|m| (m.bx, m.by, m.bz, voxeldust_core::block::BlockId::from_u16(m.block_type)))
                            .collect();
                        chunk_cache.apply_delta(source, chunk_pos, cd.seq, &edits);
                    }
                    // Apply sub-block mods.
                    if !cd.sub_block_mods.is_empty() {
                        chunk_cache.apply_sub_block_delta(source, chunk_pos, &cd.sub_block_mods);
                    }
                }
            }
            NetEvent::Transitioning => {
                info!("transitioning to new shard...");
                smooth.has_prev_server_pos = false;
                smooth.walk_velocity = DVec3::ZERO;

                // Transition to new shard: the OLD source's chunks remain in the
                // cache (e.g., ship chunks stay visible as exterior after exiting).
                // A new source will be created when the new shard starts sending data.
                // The old source is removed later when its chunks are no longer needed
                // (out of render range, or explicitly cleaned up).
                //
                // For now, we keep the old source and create a fresh one for the new shard.
                // TODO: implement distance-based cleanup of old sources.
                chunk_cache.primary_source = None;

                // Clear warp state — warp travel is over.
                warp.galaxy_position = None;
                warp.galaxy_rotation = None;
                warp.target_star_index = None;

                if ws.secondary.is_some() {
                    // Seamless: promote secondary to primary.
                    info!("seamless transition — secondary data available");

                    // Convert camera yaw/pitch from ship frame to planet tangent
                    // frame so the view direction is preserved across the transition.
                    if conn.current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
                        // Step 1: world-space forward from ship-local yaw/pitch.
                        let sp = (cam.pitch as f32).sin() as f64;
                        let cp = (cam.pitch as f32).cos() as f64;
                        let sy = (cam.yaw as f32).sin() as f64;
                        let cy = (cam.yaw as f32).cos() as f64;
                        let local_fwd = DVec3::new(cy * cp, sp, sy * cp);
                        let cam_fwd_world = (player.ship_rotation * local_fwd).normalize();

                        // Step 2: planet tangent frame at player's position.
                        // Use promoted WorldState's first player position (planet-local).
                        let planet_pos = ws.secondary.as_ref()
                            .and_then(|secondary| secondary.players.first())
                            .map(|p| p.position)
                            .unwrap_or(DVec3::Y);
                        let up = planet_pos.normalize();
                        let pole = DVec3::Y;
                        let east_raw = pole.cross(up);
                        let east = if east_raw.length_squared() > 1e-10 {
                            east_raw.normalize()
                        } else {
                            DVec3::Z.cross(up).normalize()
                        };
                        let north = up.cross(east).normalize();

                        // Step 3: project world forward onto tangent frame.
                        let fwd_north = cam_fwd_world.dot(north);
                        let fwd_up = cam_fwd_world.dot(up);
                        let fwd_east = cam_fwd_world.dot(east);

                        // Step 4: extract planet-frame yaw/pitch.
                        cam.pitch = fwd_up.asin();
                        cam.yaw = fwd_east.atan2(fwd_north);
                        cam.pitch = cam.pitch.clamp(
                            -std::f64::consts::FRAC_PI_2 + 0.01,
                            std::f64::consts::FRAC_PI_2 - 0.01,
                        );
                    }

                    ws.latest = ws.secondary.take();
                    if let Some(st) = conn.secondary_shard_type.take() {
                        conn.current_shard_type = st;
                    }
                    player.is_piloting = false;
                } else {
                    // Hard transition: clear and reconnect.
                    ws.latest = None;
                }
                ws.secondary = None;
                conn.secondary_shard_type = None;
            }
        }
    }
}

fn smooth_render_position(
    mut smooth: ResMut<RenderSmoothing>,
    player: Res<LocalPlayer>,
    warp: Res<ClientWarp>,
    ft: Res<FrameTime>,
) {
    // Exponential smoothing: converges ~90% within 2 server ticks (100ms).
    // Frame-rate independent via dt-scaled blend factor.
    let blend = 1.0 - (-20.0 * ft.dt).exp();

    if warp.galaxy_position.is_some() {
        // Galaxy warp mode: smooth galaxy-scale position + rotation.
        let elapsed = (ft.last_time - smooth.last_galaxy_update_time).as_secs_f64().min(0.06);
        let target = smooth.prev_galaxy_position + smooth.galaxy_velocity * elapsed;
        let delta = (target - smooth.galaxy_render_position) * blend;
        smooth.galaxy_render_position = smooth.galaxy_render_position + delta;
        let rot = smooth.galaxy_render_rotation.slerp(smooth.prev_galaxy_rotation, blend);
        smooth.galaxy_render_rotation = rot;
    } else if player.is_piloting {
        // Piloting mode: player's ship-local position is static while seated —
        // no velocity extrapolation needed.  Only rotation is smoothed via slerp.
        smooth.render_position = player.position;
        let rot = smooth.render_rotation.slerp(player.ship_rotation, blend);
        smooth.render_rotation = rot;
    } else {
        // Walking mode: smooth player position between server ticks.
        let elapsed = (ft.last_time - smooth.last_server_update_time).as_secs_f64().min(0.06);
        let target = player.position + smooth.walk_velocity * elapsed;
        let delta = (target - smooth.render_position) * blend;
        smooth.render_position = smooth.render_position + delta;
    }
}

fn check_autopilot_timeout(
    mut ap: ResMut<ClientAutopilot>,
    warp: Res<ClientWarp>,
    ws: Res<ServerWorldState>,
    player: Res<LocalPlayer>,
) {
    // Check for double-tap T timeout — if 400ms elapsed since first tap, engage DirectApproach.
    // Skipped when warp target is active (warp takes priority over planet autopilot).
    if warp.target_star_index.is_some() {
        ap.last_t_press = None; // Clear stale T-press when warp is active.
        return;
    }
    if let Some(press_time) = ap.last_t_press {
        if press_time.elapsed() >= std::time::Duration::from_millis(400) {
            ap.last_t_press = None;
            // Engage autopilot to nearest planet in ship's forward direction.
            engage_autopilot_to_nearest(&mut ap, &ws, &player, AutopilotMode::DirectApproach);
        }
    }
}

fn compute_trajectory(
    ft: Res<FrameTime>,
    mut ap: ResMut<ClientAutopilot>,
    ws: Res<ServerWorldState>,
    player: Res<LocalPlayer>,
    sys_params: Res<SystemParamsCache>,
    flight: Res<FlightControl>,
    mut worker: ResMut<TrajectoryWorker>,
) {
    // Poll for completed trajectory from the background thread (non-blocking).
    // Collect results first, then update state (separates rx borrow from worker mutation).
    let mut latest_plan: Option<Option<TrajectoryPlan>> = None;
    let mut received_count = 0u64;
    {
        let rx = worker.result_rx.lock().unwrap();
        while let Ok(plan) = rx.try_recv() {
            received_count += 1;
            latest_plan = Some(plan);
        }
    }
    if received_count > 0 {
        worker.last_received += received_count;
        // Use only the latest result (discard intermediate ones).
        if let Some(plan) = latest_plan {
            ap.trajectory_plan = plan;
        }
    }

    // Clear trajectory when autopilot is disengaged.
    if ap.target.is_none() {
        ap.trajectory_plan = None;
        return;
    }

    // Submit a new computation every 10 frames (~6Hz), but only if the
    // background thread has finished the previous one (don't queue up stale jobs).
    let ready_for_new = worker.last_submitted == worker.last_received;
    if !ready_for_new {
        return; // Background thread still working — wait for it.
    }
    if ft.count % 10 != 0 && ap.trajectory_plan.is_some() {
        return; // Throttled — keep current plan.
    }

    if let (Some(system), Some(world_state), Some(target_idx)) =
        (&sys_params.0, &ws.latest, ap.target)
    {
        let seed = ap.server_autopilot.as_ref().map(|snap| autopilot::AutopilotSeed {
            intercept_pos: snap.intercept_pos,
            target_arrival_vel: snap.target_arrival_vel,
            phase: autopilot::FlightPhase::from_u8(snap.phase),
            braking_committed: snap.braking_committed,
        });
        let thrust_tier = ap
            .server_autopilot
            .as_ref()
            .map(|s| s.thrust_tier)
            .unwrap_or(flight.selected_thrust_tier);

        worker.last_submitted += 1;
        let _ = worker.request_tx.send(TrajectoryRequest {
            seq: worker.last_submitted,
            ship_pos: world_state.origin,
            ship_vel: player.velocity,
            target_planet_index: target_idx,
            system: system.clone(),
            game_time: world_state.game_time,
            ship_props: autopilot::ShipPhysicalProperties::starter_ship(),
            thrust_tier,
            sample_count: 200,
            seed,
        });
    }
}

// ---------------------------------------------------------------------------
// Helper: engage autopilot to nearest planet in ship's forward direction
// ---------------------------------------------------------------------------

fn engage_autopilot_to_nearest(
    ap: &mut ClientAutopilot,
    ws: &ServerWorldState,
    player: &LocalPlayer,
    mode: AutopilotMode,
) {
    if let Some(ref world_state) = ws.latest {
        let ship_fwd = player.ship_rotation * DVec3::NEG_Z;
        let ship_pos = world_state.origin;
        let mut best: Option<(usize, f64)> = None;
        for body in &world_state.bodies {
            if body.body_id == 0 { continue; }
            let to_body = (body.position - ship_pos).normalize_or_zero();
            let d = ship_fwd.dot(to_body);
            if d > best.map(|(_, bd)| bd).unwrap_or(0.7) {
                best = Some(((body.body_id - 1) as usize, d));
            }
        }
        if let Some((idx, _)) = best {
            ap.target = Some(idx);
            ap.mode = mode;
            info!(planet = idx, mode = ?mode, "autopilot engaged");
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build warp target info for HUD
// ---------------------------------------------------------------------------

fn build_warp_target_info(
    warp: &ClientWarp,
    sf_res: &StarFieldRes,
    smooth_galaxy_pos: Option<DVec3>,
) -> Option<hud::WarpTargetInfo> {
    let target_idx = warp.target_star_index?;
    let sf = sf_res.0.as_ref()?;
    let star = sf.get_star(target_idx)?;

    // Reference position: warp camera position during warp, current star otherwise.
    let ref_pos = smooth_galaxy_pos.or(warp.galaxy_position).unwrap_or_else(|| {
        sf.current_star_index
            .and_then(|idx| sf.catalog.iter().find(|s| s.index == idx))
            .map(|s| s.galaxy_position)
            .unwrap_or(DVec3::ZERO)
    });
    let distance = (star.galaxy_position - ref_pos).length();
    let galaxy_dir = (star.galaxy_position - ref_pos).normalize();

    // galaxy_dir is in galaxy/world space. The main camera VP already
    // transforms from world space to screen space (including ship_rotation
    // which equals warp_galaxy_rotation during warp). No extra transform
    // needed — projecting galaxy_dir through ctx.vp gives the correct
    // screen position in both warp and normal flight.
    let direction = galaxy_dir;

    Some(hud::WarpTargetInfo {
        star_index: target_idx,
        star_class_name: stars::star_class_name(star.star_class),
        distance_gu: distance,
        luminosity: star.luminosity,
        direction,
    })
}

// ---------------------------------------------------------------------------
// ClientApp — owns ECS App + GPU state
// ---------------------------------------------------------------------------

struct ClientApp {
    ecs_app: bevy_app::App,
    // GPU state kept outside World (egui_winit::State is !Send).
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    block_renderer: Option<block_render::BlockRenderer>,
    uniform_data: Vec<ObjectUniforms>,
    /// Cached block registry — built once, used for meshing and raycast.
    /// Stored outside the ECS world to avoid borrow conflicts with chunk cache.
    registry: voxeldust_core::block::BlockRegistry,
    args: Args,
}

impl ClientApp {
    fn new(args: Args) -> Self {
        // Initialize compute task pool for bevy's MultiThreadedExecutor.
        bevy_tasks::ComputeTaskPool::get_or_init(bevy_tasks::TaskPool::default);

        let mut ecs_app = bevy_app::App::new();

        // Insert resources with initial values.
        ecs_app.insert_resource(NetworkChannels {
            event_rx: None,
            input_tx: None,
            block_edit_tx: None,
            tcp_out_tx: None,
        });
        ecs_app.insert_resource(BlockHotbar::default());
        ecs_app.insert_resource(BlockTarget::default());
        ecs_app.insert_resource(SubBlockTool::default());
        ecs_app.insert_resource(ClientBlockIndex::default());
        ecs_app.insert_resource(BlockConfigUIState::default());
        ecs_app.insert_resource(events::InputContext::default());
        ecs_app.insert_resource(events::RawInputBuffer::default());
        ecs_app.insert_resource(CameraZoomOffset::default());
        ecs_app.insert_resource(ConnectionInfo {
            connected: false,
            current_shard_type: 255,
            reference_position: DVec3::ZERO,
            reference_rotation: DQuat::IDENTITY,
            secondary_shard_type: None,
            system_seed: 0,
            galaxy_seed: 0,
        });
        ecs_app.insert_resource(ServerWorldState {
            latest: None,
            secondary: None,
        });
        ecs_app.insert_resource(LocalPlayer {
            position: DVec3::new(0.0, 1.0, 0.0),
            velocity: DVec3::ZERO,
            is_piloting: false,
            ship_rotation: DQuat::IDENTITY,
        });
        ecs_app.insert_resource(RenderSmoothing {
            render_position: DVec3::new(0.0, 1.0, 0.0),
            prev_server_position: DVec3::ZERO,
            walk_velocity: DVec3::ZERO,
            last_server_update_time: Instant::now(),
            has_prev_server_pos: false,
            render_rotation: DQuat::IDENTITY,
            ship_velocity: DVec3::ZERO,
            galaxy_render_position: DVec3::ZERO,
            galaxy_render_rotation: DQuat::IDENTITY,
            galaxy_velocity: DVec3::ZERO,
            prev_galaxy_position: DVec3::ZERO,
            prev_galaxy_rotation: DQuat::IDENTITY,
            last_galaxy_update_time: Instant::now(),
            has_prev_galaxy_pos: false,
        });
        ecs_app.insert_resource(CameraControl {
            yaw: 0.0,
            pitch: 0.0,
            pilot_yaw_rate: 0.0,
            pilot_pitch_rate: 0.0,
        });
        ecs_app.insert_resource(KeyboardState {
            keys_held: HashSet::new(),
            mouse_grabbed: false,
        });
        ecs_app.insert_resource(FlightControl {
            selected_thrust_tier: 3,
            engines_off: false,
            thrust_limiter: 0.75,
        });
        ecs_app.insert_resource(ClientAutopilot {
            target: None,
            trajectory_plan: None,
            server_autopilot: None,
            mode: AutopilotMode::DirectApproach,
            last_t_press: None,
        });
        ecs_app.insert_resource(ClientWarp {
            target_star_index: None,
            galaxy_position: None,
            galaxy_rotation: None,
            prev_g_pressed: false,
        });
        ecs_app.insert_resource(TrajectoryWorker::spawn());
        ecs_app.insert_resource(StarFieldRes(None));
        ecs_app.insert_resource(SystemParamsCache(None));
        ecs_app.insert_resource(ClientChunkCacheRes {
            cache: voxeldust_core::block::client_chunks::ClientChunkCache::new(),
            primary_source: None,
        });
        ecs_app.insert_resource(FrameTime {
            count: 0,
            last_time: Instant::now(),
            dt: 0.0,
        });

        // Configure system sets with ordering.
        ecs_app.configure_sets(
            Update,
            (
                ClientSet::FrameTime,
                ClientSet::Network,
                ClientSet::Input,
                ClientSet::Smooth,
                ClientSet::AutopilotTimeout,
                ClientSet::Trajectory,
            )
                .chain(),
        );

        // Register messages.
        ecs_app.add_message::<events::KeyPressedMsg>();
        ecs_app.add_message::<events::KeyReleasedMsg>();
        ecs_app.add_message::<events::MouseButtonMsg>();
        ecs_app.add_message::<events::MouseMotionMsg>();
        ecs_app.add_message::<events::CursorGrabRequest>();
        ecs_app.add_message::<events::ConfigPanelCloseMsg>();
        ecs_app.add_message::<events::MouseScrollMsg>();
        ecs_app.add_message::<events::ConfigPanelOpenMsg>();

        // Register systems.
        ecs_app.add_systems(Update, update_frame_time.in_set(ClientSet::FrameTime));
        ecs_app.add_systems(Update, poll_network.in_set(ClientSet::Network));
        ecs_app.add_systems(Update, (
            bridge_raw_input_system,
            interpret_camera_system,
            interpret_key_system,
            interpret_mouse_system,
            tick_config_panel_system,
        ).chain().in_set(ClientSet::Input));
        ecs_app.add_systems(Update, send_input_system.after(ClientSet::Input).in_set(ClientSet::Smooth));
        ecs_app.add_systems(Update, smooth_render_position.in_set(ClientSet::Smooth));
        ecs_app.add_systems(Update, check_autopilot_timeout.in_set(ClientSet::AutopilotTimeout));
        ecs_app.add_systems(Update, compute_trajectory.in_set(ClientSet::Trajectory));

        Self {
            ecs_app,
            window: None,
            gpu: None,
            block_renderer: None,
            uniform_data: vec![bytemuck::Zeroable::zeroed(); MAX_OBJECTS],
            registry: voxeldust_core::block::BlockRegistry::new(),
            args,
        }
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let gpu_state = gpu::init_gpu(window.clone());
        // Create the block renderer now that we have GPU state.
        let br = block_render::BlockRenderer::new(
            &gpu_state.device,
            gpu_state.config.format,
            &gpu_state.bind_group_layout,
            &gpu_state.scene_bind_group_layout,
            &gpu_state.shadow_bind_group_layout,
        );
        self.block_renderer = Some(br);
        self.window = Some(window);
        self.gpu = Some(gpu_state);
    }

    fn start_networking(&mut self) {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (block_edit_tx, block_edit_rx) = mpsc::unbounded_channel();
        let (tcp_out_tx, tcp_out_rx) = mpsc::unbounded_channel();
        let input_rx = Arc::new(tokio::sync::Mutex::new(input_rx));
        let block_edit_rx = Arc::new(tokio::sync::Mutex::new(block_edit_rx));
        let tcp_out_rx = Arc::new(tokio::sync::Mutex::new(tcp_out_rx));

        let gateway = self.args.gateway;
        let name = self.args.name.clone();
        let direct = self.args.direct.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(network::run_network(gateway, name, event_tx, input_rx, block_edit_rx, tcp_out_rx, direct));
        });

        let world = self.ecs_app.world_mut();
        let mut net = world.resource_mut::<NetworkChannels>();
        net.event_rx = Some(event_rx);
        net.input_tx = Some(input_tx);
        net.block_edit_tx = Some(block_edit_tx);
        net.tcp_out_tx = Some(tcp_out_tx);
    }

    fn render(&mut self) {
        // Run all ECS systems (frame time, network, input, smoothing, autopilot, trajectory).
        self.ecs_app.update();

        // Apply cursor grab/ungrab requests from ECS messages (needs !Send Window).
        self.apply_cursor_changes();

        let gpu = match &mut self.gpu { Some(g) => g, None => return };

        // Extract all scalar/Copy values from resources in a scoped block so
        // the immutable borrow on `self.ecs_app` ends before we take `world_mut`.
        let (
            warp_target_info,
            cam,
            frame_count,
            secondary_shard_type,
            render_position,
            player_velocity,
            ship_rotation,
            is_piloting,
            connected,
            current_shard_type,
            selected_thrust_tier,
            engines_off,
            autopilot_target,
            warp_galaxy_position,
            warp_galaxy_rotation,
            warp_target_star_index,
            cam_yaw,
            cam_pitch,
        ) = {
            let world = self.ecs_app.world();
            let smooth = world.resource::<RenderSmoothing>();
            let cam_ctrl = world.resource::<CameraControl>();
            let player = world.resource::<LocalPlayer>();
            let conn = world.resource::<ConnectionInfo>();
            let ws = world.resource::<ServerWorldState>();
            let kb = world.resource::<KeyboardState>();
            let flight = world.resource::<FlightControl>();
            let ap = world.resource::<ClientAutopilot>();
            let warp = world.resource::<ClientWarp>();
            let sf_res = world.resource::<StarFieldRes>();
            let ft = world.resource::<FrameTime>();

            let smooth_gal = if warp.galaxy_position.is_some() { Some(smooth.galaxy_render_position) } else { None };
            let warp_target_info = build_warp_target_info(warp, sf_res, smooth_gal);

            // Apply config panel zoom offset to camera position.
            let zoom = world.resource::<CameraZoomOffset>();
            let cam_position = smooth.render_position + zoom.offset;
            if zoom.offset.length_squared() > 0.001 {
                info!(
                    ox = zoom.offset.x, oy = zoom.offset.y, oz = zoom.offset.z,
                    len = zoom.offset.length(),
                    "applying camera zoom offset",
                );
            }

            // Compute camera.
            let cam = camera::compute_camera(
                cam_position,
                cam_ctrl.yaw,
                cam_ctrl.pitch,
                player.is_piloting,
                conn.current_shard_type,
                smooth.render_rotation,
                &kb.keys_held,
                gpu.config.width,
                gpu.config.height,
                ws.latest.as_ref(),
            );

            (
                warp_target_info,
                cam,
                ft.count,
                conn.secondary_shard_type,
                cam_position,
                player.velocity,
                player.ship_rotation,
                player.is_piloting,
                conn.connected,
                conn.current_shard_type,
                flight.selected_thrust_tier,
                flight.engines_off,
                ap.target,
                if warp.galaxy_position.is_some() { Some(smooth.galaxy_render_position) } else { None },
                if warp.galaxy_rotation.is_some() { Some(smooth.galaxy_render_rotation) } else { None },
                warp.target_star_index,
                cam_ctrl.yaw,
                cam_ctrl.pitch,
            )
        };

        let window = self.window.as_ref().unwrap();

        // Update star field and upload instances to GPU.
        // Star field requires &mut for update_instances, so we need mutable world access.
        let star_instance_count = {
            let world_mut = self.ecs_app.world_mut();
            let mut sf_res_mut = world_mut.resource_mut::<StarFieldRes>();
            if let Some(ref mut sf) = sf_res_mut.0 {
                // During warp: galaxy mode with real ship position (parallax).
                // Otherwise: skybox mode (directions at infinity, no parallax).
                let skybox_mode = warp_galaxy_position.is_none();
                let current_star_pos = sf.current_star_index
                    .and_then(|idx| sf.catalog.iter().find(|s| s.index == idx))
                    .map(|s| s.galaxy_position)
                    .unwrap_or(DVec3::ZERO);
                let cam_galaxy_pos = warp_galaxy_position.unwrap_or(current_star_pos);
                sf.update_instances(current_star_pos, cam_galaxy_pos, skybox_mode, warp_target_star_index);
                let count = sf.instances.len() as u32;
                // DEBUG: log star mode + position during warp
                if frame_count % 120 == 0 && secondary_shard_type == Some(3) {
                    if let Some(pos) = warp_galaxy_position {
                        tracing::info!(
                            star_count = count, skybox_mode,
                            pos = format!("({:.2},{:.2},{:.2})", pos.x, pos.y, pos.z),
                            "warp star field status"
                        );
                    }
                }
                if count > 0 {
                    gpu.queue.write_buffer(
                        &gpu.star_instance_buf,
                        0,
                        bytemuck::cast_slice(&sf.instances),
                    );

                    // Compute star scene uniforms.
                    // During warp: dedicated galaxy-frame view-projection so stars
                    // appear at correct positions relative to the cockpit.
                    // Normal flight: use the main camera's view-projection (skybox).
                    let star_uniforms = if let Some(warp_rot) = warp_galaxy_rotation {
                        // Galaxy-frame view: ship faces target star via warp_rotation.
                        // When piloting, camera is locked to ship heading (NEG_Z) —
                        // the star VP must match so the target star appears centered
                        // through the cockpit. When walking, use free yaw/pitch.
                        let local_look = if is_piloting {
                            glam::DVec3::NEG_Z
                        } else {
                            let (sy, cy) = (cam_yaw as f32).sin_cos();
                            let (sp, cp) = (cam_pitch as f32).sin_cos();
                            glam::DVec3::new((cy * cp) as f64, sp as f64, (sy * cp) as f64)
                        };
                        let galaxy_fwd = (warp_rot * local_look).normalize().as_vec3();
                        let galaxy_up = (warp_rot * DVec3::Y).normalize().as_vec3();
                        let galaxy_right = galaxy_fwd.cross(galaxy_up).normalize();
                        let corrected_up = galaxy_right.cross(galaxy_fwd).normalize();

                        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
                        let star_view = glam::Mat4::look_to_rh(
                            glam::Vec3::ZERO, galaxy_fwd, corrected_up,
                        );
                        let star_proj = glam::Mat4::perspective_infinite_reverse_rh(
                            70.0_f32.to_radians(), aspect, 0.1,
                        );
                        let star_vp = star_proj * star_view;

                        stars::StarSceneUniforms {
                            view_proj: star_vp.to_cols_array_2d(),
                            camera_right: [galaxy_right.x, galaxy_right.y, galaxy_right.z, 0.0],
                            camera_up: [corrected_up.x, corrected_up.y, corrected_up.z, 0.0],
                            warp_velocity: [0.0, 0.0, 0.0, 0.0],
                            render_mode: [0.0, 0.0, 0.0, 0.0], // skybox mode (directions at 500 units)
                        }
                    } else {
                        // Normal flight: use main camera's view-projection.
                        let cam_right = cam.cam_fwd.cross(cam.cam_up).normalize();
                        stars::StarSceneUniforms {
                            view_proj: cam.vp.to_cols_array_2d(),
                            camera_right: [cam_right.x, cam_right.y, cam_right.z, 0.0],
                            camera_up: [cam.cam_up.x, cam.cam_up.y, cam.cam_up.z, 0.0],
                            warp_velocity: [0.0, 0.0, 0.0, 0.0],
                            render_mode: [0.0, 0.0, 0.0, 0.0], // skybox mode
                        }
                    };

                    gpu.queue.write_buffer(
                        &gpu.star_scene_uniform_buf,
                        0,
                        bytemuck::bytes_of(&star_uniforms),
                    );
                }
                count
            } else {
                0
            }
        };

        // Process dirty chunks: generate meshes and upload to GPU.
        // This bridges the ECS chunk cache (populated by poll_network) with
        // the GPU block renderer (consumed by render_frame).
        if let Some(ref mut br) = self.block_renderer {
            let registry = &self.registry;
            let world_mut = self.ecs_app.world_mut();
            let mut chunk_cache = world_mut.resource_mut::<ClientChunkCacheRes>();

            // Clean up GPU buffers for removed sources.
            for source in chunk_cache.drain_removed_sources() {
                br.remove_source(source);
            }

            // Process newly dirty chunks (source-aware keys).
            if chunk_cache.has_dirty() {
                let dirty_keys = chunk_cache.drain_dirty();
                for dk in dirty_keys {
                    if let Some(chunk) = chunk_cache.get_chunk(dk.source, dk.chunk) {
                        let neighbors = chunk_cache.get_neighbors(dk.source, dk.chunk);
                        let mesh = block_render::generate_chunk_gpu_mesh(
                            chunk, &neighbors, &registry, false,
                        );
                        tracing::info!(
                            chunk = %dk.chunk,
                            vertices = mesh.vertices.len(),
                            indices = mesh.indices.len(),
                            empty = mesh.is_empty(),
                            "meshed chunk for GPU"
                        );
                        br.upload_chunk_mesh(&gpu.device, dk, &mesh);

                        // Generate and upload sub-block mesh for this chunk.
                        let sub_mesh = voxeldust_core::block::sub_block_mesher::mesh_sub_blocks(chunk);
                        br.upload_sub_block_mesh(&gpu.device, dk, &sub_mesh);
                    } else {
                        br.remove_chunk_mesh(dk);
                        br.remove_sub_block_mesh(dk);
                    }
                }
                tracing::info!(
                    total_gpu_chunks = br.total_chunk_count(),
                    "block renderer chunk update complete"
                );

                // Rebuild functional block index (only when chunks changed).
                // Build the entries in a local Vec, then swap into the resource.
                let primary_src = chunk_cache.primary_source;
                if let Some(src) = primary_src {
                    let mut new_entries = Vec::new();
                    let chunk_keys: Vec<_> = chunk_cache.cache.source_chunk_keys(src).collect();
                    for ck in chunk_keys {
                        let chunk = match chunk_cache.cache.get_chunk(src, ck) {
                            Some(c) => c,
                            None => continue,
                        };
                        for bx in 0u8..62 {
                            for by in 0u8..62 {
                                for bz in 0u8..62 {
                                    let bid = chunk.get_block(bx, by, bz);
                                    if bid.is_air() { continue; }
                                    if let Some(kind) = registry.functional_kind(bid) {
                                        let wx = ck.x * 62 + bx as i32;
                                        let wy = ck.y * 62 + by as i32;
                                        let wz = ck.z * 62 + bz as i32;
                                        let pos = glam::Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
                                        let letter = match kind {
                                            voxeldust_core::block::FunctionalBlockKind::Thruster => 'T',
                                            voxeldust_core::block::FunctionalBlockKind::Reactor => 'R',
                                            voxeldust_core::block::FunctionalBlockKind::Seat => 'S',
                                            voxeldust_core::block::FunctionalBlockKind::Battery => 'B',
                                            voxeldust_core::block::FunctionalBlockKind::SignalConverter => 'C',
                                            voxeldust_core::block::FunctionalBlockKind::Antenna => 'A',
                                            _ => 'F',
                                        };
                                        new_entries.push((pos, kind as u8, letter));
                                    }
                                }
                            }
                        }
                    }
                    drop(chunk_cache);
                    world_mut.resource_mut::<ClientBlockIndex>().entries = new_entries;
                }
            }
        }

        // Client-side block targeting raycast (for highlight).
        // Only when walking inside a ship shard (not piloting).
        if current_shard_type == voxeldust_core::client_message::shard_type::SHIP && !is_piloting {
            let world_mut = self.ecs_app.world_mut();
            let chunk_cache = world_mut.resource::<ClientChunkCacheRes>();
            let registry = &self.registry;
            let source = chunk_cache.primary_source;

            let eye = glam::Vec3::new(
                render_position.x as f32 + 0.0,
                render_position.y as f32 + gpu::EYE_HEIGHT as f32,
                render_position.z as f32 + 0.0,
            );
            let (sy, cy) = (cam_yaw as f32).sin_cos();
            let (sp, cp) = (cam_pitch as f32).sin_cos();
            let look = glam::Vec3::new(cy * cp, sp, sy * cp);

            let hit = if let Some(src) = source {
                voxeldust_core::block::raycast::raycast(eye, look, 8.0, |x, y, z| {
                    let (chunk_key, lx, ly, lz) =
                        voxeldust_core::block::ShipGrid::world_to_chunk(x, y, z);
                    chunk_cache
                        .get_chunk(src, chunk_key)
                        .map(|c| registry.is_solid(c.get_block(lx, ly, lz)))
                        .unwrap_or(false)
                })
            } else {
                None
            };

            world_mut.resource_mut::<BlockTarget>().hit = hit;
        } else {
            let world_mut = self.ecs_app.world_mut();
            world_mut.resource_mut::<BlockTarget>().hit = None;
        }

        // Extract config state for the egui panel — only during Interactive phase.
        let mut config_for_panel = {
            let world_mut = self.ecs_app.world_mut();
            let ui_state = world_mut.resource::<BlockConfigUIState>();
            if matches!(ui_state.phase, ConfigPanelPhase::Interactive) {
                ui_state.open_config.clone()
            } else {
                None
            }
        };

        // Re-read resources for render_frame call.
        let world = self.ecs_app.world();
        let ws = world.resource::<ServerWorldState>();
        let ap = world.resource::<ClientAutopilot>();
        let sys_params = world.resource::<SystemParamsCache>();

        let panel_action = render::render_frame(
            gpu,
            window,
            &mut self.uniform_data,
            &cam,
            ws.latest.as_ref(),
            ws.secondary.as_ref(),
            current_shard_type,
            render_position,
            player_velocity,
            ship_rotation,
            is_piloting,
            connected,
            selected_thrust_tier,
            engines_off,
            autopilot_target,
            ap.trajectory_plan.as_ref(),
            ap.server_autopilot.as_ref(),
            sys_params.0.as_ref(),
            frame_count,
            star_instance_count,
            warp_target_info,
            self.block_renderer.as_ref(),
            world.resource::<BlockTarget>().hit.as_ref(),
            &world.resource::<ClientBlockIndex>().entries,
            config_for_panel.as_mut(),
        );

        // Handle config panel actions.
        match panel_action {
            hud::ConfigPanelAction::Save => {
                if let Some(config) = &config_for_panel {
                    let update = voxeldust_core::signal::config::BlockConfigUpdateData {
                        block_pos: config.block_pos,
                        publish_bindings: config.publish_bindings.clone(),
                        subscribe_bindings: config.subscribe_bindings.clone(),
                        converter_rules: config.converter_rules.clone(),
                        seat_mappings: config.seat_mappings.clone(),
                    };
                    let world = self.ecs_app.world();
                    let net = world.resource::<NetworkChannels>();
                    if let Some(ref tx) = net.block_edit_tx {
                        let msg = voxeldust_core::client_message::ClientMsg::BlockConfigUpdate(update);
                        let data = msg.serialize();
                        let mut pkt = Vec::new();
                        voxeldust_core::wire_codec::encode(&data, &mut pkt);
                        // TODO: add a separate config_update_tx channel for proper typing.
                    }
                    info!(block = ?config.block_pos, "config saved (sending to server)");
                }
                // Fire close message → tick_config_panel_system handles animate-out.
                let world = self.ecs_app.world_mut();
                world.resource_mut::<events::RawInputBuffer>().key_presses.push(
                    events::KeyPressedMsg { key: KeyCode::Escape },
                );
                // Direct: close immediately via config_ui since we're outside the ECS schedule.
                let mut ui = world.resource_mut::<BlockConfigUIState>();
                ui.open_config = None;
                ui.phase = ConfigPanelPhase::AnimatingOut { progress: 0.0 };
                world.resource_mut::<events::InputContext>().mode = events::InputMode::Animating;
                // Cursor re-grab will happen via apply_cursor_changes next frame.
            }
            hud::ConfigPanelAction::Close => {
                let world = self.ecs_app.world_mut();
                let mut ui = world.resource_mut::<BlockConfigUIState>();
                ui.open_config = None;
                ui.phase = ConfigPanelPhase::AnimatingOut { progress: 0.0 };
                world.resource_mut::<events::InputContext>().mode = events::InputMode::Animating;
            }
            hud::ConfigPanelAction::None => {
                // Panel still open — write back edited config.
                if let Some(config) = config_for_panel {
                    let world_mut = self.ecs_app.world_mut();
                    let mut ui_state = world_mut.resource_mut::<BlockConfigUIState>();
                    ui_state.open_config = Some(config);
                }
            }
        }
    }

    /// Drain CursorGrabRequest messages and apply to the !Send Window.
    fn apply_cursor_changes(&mut self) {
        let world = self.ecs_app.world_mut();
        // Read pending cursor requests from the InputContext (set by tick_config_panel_system).
        // We use a simple approach: check if InputContext.cursor_grabbed disagrees with
        // KeyboardState.mouse_grabbed (the latter is the actual OS state).
        let desired_grabbed = {
            let input_ctx = world.resource::<events::InputContext>();
            match input_ctx.mode {
                events::InputMode::Game => true,     // Game mode = cursor locked
                events::InputMode::Animating => true, // Animating = cursor locked
                events::InputMode::UiPanel => false,  // UI panel = cursor free
                events::InputMode::MenuFocus => false, // Menu = cursor free
            }
        };
        let currently_grabbed = world.resource::<KeyboardState>().mouse_grabbed;

        if desired_grabbed != currently_grabbed {
            if let Some(ref w) = self.window {
                if desired_grabbed {
                    if w.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                        let _ = w.set_cursor_grab(CursorGrabMode::Confined);
                    }
                    w.set_cursor_visible(false);
                } else {
                    let _ = w.set_cursor_grab(CursorGrabMode::None);
                    w.set_cursor_visible(true);
                }
                world.resource_mut::<KeyboardState>().mouse_grabbed = desired_grabbed;
                world.resource_mut::<events::InputContext>().cursor_grabbed = desired_grabbed;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Config animation helpers
// ---------------------------------------------------------------------------

/// Serialize a ClientMsg and send via the TCP outbound channel (reliable delivery).
fn send_tcp_msg(net: &NetworkChannels, msg: voxeldust_core::client_message::ClientMsg) {
    if let Some(ref tx) = net.tcp_out_tx {
        let data = msg.serialize();
        let mut pkt = Vec::new();
        voxeldust_core::wire_codec::encode(&data, &mut pkt);
        let _ = tx.send(pkt);
    }
}

/// Smoothstep for [0,1] — ease in/out.
fn smooth_step(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Lerp between two angles, taking the shortest path.
fn lerp_angle(a: f64, b: f64, t: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut diff = (b - a) % tau;
    if diff > std::f64::consts::PI { diff -= tau; }
    if diff < -std::f64::consts::PI { diff += tau; }
    a + diff * t
}

fn lerp_f64(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// ECS input pipeline: bridge → interpret → config panel
// ---------------------------------------------------------------------------

/// Drain raw input buffer (populated by window_event) into ECS messages.
fn bridge_raw_input_system(
    mut buf: ResMut<events::RawInputBuffer>,
    mut key_presses: MessageWriter<events::KeyPressedMsg>,
    mut key_releases: MessageWriter<events::KeyReleasedMsg>,
    mut mouse_buttons: MessageWriter<events::MouseButtonMsg>,
    mut mouse_motions: MessageWriter<events::MouseMotionMsg>,
    mut mouse_scrolls: MessageWriter<events::MouseScrollMsg>,
) {
    for ev in buf.key_presses.drain(..) { key_presses.write(ev); }
    for ev in buf.key_releases.drain(..) { key_releases.write(ev); }
    for ev in buf.mouse_buttons.drain(..) { mouse_buttons.write(ev); }
    for ev in buf.mouse_motions.drain(..) { mouse_motions.write(ev); }
    for ev in buf.mouse_scrolls.drain(..) { mouse_scrolls.write(ev); }
}

// ---------------------------------------------------------------------------
// SystemParam bundles — group related params by domain
// ---------------------------------------------------------------------------

/// Player state and camera — read by all input interpretation systems.
#[derive(SystemParam)]
struct PlayerCtx<'w> {
    player: Res<'w, LocalPlayer>,
    conn: Res<'w, ConnectionInfo>,
    smooth: Res<'w, RenderSmoothing>,
    cam: ResMut<'w, CameraControl>,
    kb: Res<'w, KeyboardState>,
}

/// Game action targets — resources mutated by key/mouse input interpretation.
#[derive(SystemParam)]
struct GameActionCtx<'w> {
    net: Res<'w, NetworkChannels>,
    flight: ResMut<'w, FlightControl>,
    hotbar: ResMut<'w, BlockHotbar>,
    sub_block_tool: ResMut<'w, SubBlockTool>,
    ap: ResMut<'w, ClientAutopilot>,
    warp: ResMut<'w, ClientWarp>,
    ws: Res<'w, ServerWorldState>,
}

/// Interpret mouse motion → camera rotation.
fn interpret_camera_system(
    mut ctx: PlayerCtx,
    input_ctx: ResMut<events::InputContext>,
    mut motion_events: MessageReader<events::MouseMotionMsg>,
) {
    if !input_ctx.cursor_grabbed || input_ctx.mode != events::InputMode::Game {
        return;
    }
    for motion in motion_events.read() {
        let sensitivity = 0.003;
        let free_look = ctx.kb.keys_held.contains(&KeyCode::AltLeft);
        if ctx.player.is_piloting && !free_look {
            ctx.cam.pilot_yaw_rate = (ctx.cam.pilot_yaw_rate - motion.delta_x * sensitivity * 5.0).clamp(-1.0, 1.0);
            ctx.cam.pilot_pitch_rate = (ctx.cam.pilot_pitch_rate - motion.delta_y * sensitivity * 5.0).clamp(-1.0, 1.0);
        } else {
            ctx.cam.yaw += motion.delta_x * sensitivity;
            ctx.cam.pitch -= motion.delta_y * sensitivity;
            ctx.cam.pitch = ctx.cam.pitch.clamp(
                -std::f64::consts::FRAC_PI_2 + 0.01,
                std::f64::consts::FRAC_PI_2 - 0.01,
            );
        }
    }
}

/// Interpret key presses → game actions (autopilot, interact, hotbar, warp, etc.).
fn interpret_key_system(
    ctx: PlayerCtx,
    mut input_ctx: ResMut<events::InputContext>,
    mut actions: GameActionCtx,
    mut key_events: MessageReader<events::KeyPressedMsg>,
    mut config_closes: MessageWriter<events::ConfigPanelCloseMsg>,
) {
    match input_ctx.mode {
        events::InputMode::Game => {
            for ev in key_events.read() {
                let key = ev.key;
                let is_piloting = ctx.player.is_piloting;
                let warp_target = actions.warp.target_star_index;
                let shard_type = ctx.conn.current_shard_type;

                // Autopilot: double-tap T = orbit, single-tap T = direct.
                if key == KeyCode::KeyT && is_piloting && warp_target.is_none() {
                    if actions.ap.target.is_some() {
                        actions.ap.target = None;
                        actions.ap.trajectory_plan = None;
                        actions.ap.mode = AutopilotMode::DirectApproach;
                    } else {
                        let now = Instant::now();
                        let is_double = actions.ap.last_t_press
                            .map(|prev| now.duration_since(prev) < std::time::Duration::from_millis(400))
                            .unwrap_or(false);
                        if is_double {
                            actions.ap.last_t_press = None;
                            let best = if let Some(ref world_state) = actions.ws.latest {
                                let ship_fwd = ctx.player.ship_rotation * DVec3::NEG_Z;
                                let ship_pos = world_state.origin;
                                let mut best_result: Option<(usize, f64)> = None;
                                for body in &world_state.bodies {
                                    if body.body_id == 0 { continue; }
                                    let to_body = (body.position - ship_pos).normalize_or_zero();
                                    let d = ship_fwd.dot(to_body);
                                    if d > best_result.map(|(_, bd)| bd).unwrap_or(0.7) {
                                        best_result = Some(((body.body_id - 1) as usize, d));
                                    }
                                }
                                best_result
                            } else { None };
                            if let Some((idx, _)) = best {
                                actions.ap.target = Some(idx);
                                actions.ap.mode = AutopilotMode::OrbitInsertion;
                                info!(planet = idx, mode = ?AutopilotMode::OrbitInsertion, "autopilot engaged");
                            }
                        } else {
                            actions.ap.last_t_press = Some(now);
                        }
                    }
                }

                // WASD cancels autopilot.
                if actions.ap.target.is_some() && matches!(key,
                    KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD) {
                    actions.ap.target = None;
                    actions.ap.trajectory_plan = None;
                    actions.ap.mode = AutopilotMode::DirectApproach;
                }

                // X key: toggle engine cutoff.
                if key == KeyCode::KeyX && is_piloting {
                    actions.flight.engines_off = !actions.flight.engines_off;
                    info!(engines_off = actions.flight.engines_off, "engine cutoff toggled");
                }

                // Tab: toggle sub-block placement mode.
                if key == KeyCode::Tab && !is_piloting {
                    actions.sub_block_tool.active = !actions.sub_block_tool.active;
                    info!(
                        active = actions.sub_block_tool.active,
                        element = actions.sub_block_tool.selected_type.label(),
                        "sub-block tool toggled",
                    );
                }

                // R: rotate sub-block element on face (0→1→2→3→0).
                if key == KeyCode::KeyR && !is_piloting && actions.sub_block_tool.active {
                    actions.sub_block_tool.rotation = (actions.sub_block_tool.rotation + 1) & 0x03;
                }

                // Digit keys: select block hotbar (normal mode) or sub-block type (sub-block mode).
                if !is_piloting {
                    use voxeldust_core::block::sub_block::SubBlockType;
                    let slot = match key {
                        KeyCode::Digit1 => Some(0usize),
                        KeyCode::Digit2 => Some(1),
                        KeyCode::Digit3 => Some(2),
                        KeyCode::Digit4 => Some(3),
                        KeyCode::Digit5 => Some(4),
                        KeyCode::Digit6 => Some(5),
                        KeyCode::Digit7 => Some(6),
                        KeyCode::Digit8 => Some(7),
                        KeyCode::Digit9 => Some(8),
                        _ => None,
                    };
                    if let Some(s) = slot {
                        if actions.sub_block_tool.active {
                            // Sub-block type selection.
                            const SUB_BLOCK_PALETTE: [SubBlockType; 9] = [
                                SubBlockType::PowerWire,
                                SubBlockType::Rail,
                                SubBlockType::Pipe,
                                SubBlockType::Ladder,
                                SubBlockType::RotorMount,
                                SubBlockType::PistonMount,
                                SubBlockType::HingeMount,
                                SubBlockType::SurfaceLight,
                                SubBlockType::Cable,
                            ];
                            if s < SUB_BLOCK_PALETTE.len() {
                                actions.sub_block_tool.selected_type = SUB_BLOCK_PALETTE[s];
                                info!(element = SUB_BLOCK_PALETTE[s].label(), "sub-block type selected");
                            }
                        } else {
                            actions.hotbar.selected_slot = s;
                        }
                    }
                }

                // E key: primary interaction on ship shard.
                if key == KeyCode::KeyE && shard_type == voxeldust_core::client_message::shard_type::SHIP && input_ctx.cursor_grabbed {
                    let eye = ctx.smooth.render_position + DVec3::new(0.0, gpu::EYE_HEIGHT, 0.0);
                    let (sy, cy) = (ctx.cam.yaw as f32).sin_cos();
                    let (sp, cp) = (ctx.cam.pitch as f32).sin_cos();
                    let look = DVec3::new((cy * cp) as f64, sp as f64, (sy * cp) as f64);
                    let edit = voxeldust_core::client_message::BlockEditData {
                        action: voxeldust_core::client_message::action::INTERACT, eye, look, block_type: 0,
                    };
                    send_tcp_msg(&actions.net, voxeldust_core::client_message::ClientMsg::BlockEditRequest(edit));
                }

                // F key: exit seat (when piloting) or open config UI (when walking).
                if key == KeyCode::KeyF && shard_type == voxeldust_core::client_message::shard_type::SHIP && input_ctx.cursor_grabbed && is_piloting {
                    // Send exit-seat action (no raycast needed on server).
                    let edit = voxeldust_core::client_message::BlockEditData {
                        action: voxeldust_core::client_message::action::EXIT_SEAT, eye: DVec3::ZERO, look: DVec3::ZERO, block_type: 0,
                    };
                    send_tcp_msg(&actions.net, voxeldust_core::client_message::ClientMsg::BlockEditRequest(edit));
                }
                if key == KeyCode::KeyF && shard_type == voxeldust_core::client_message::shard_type::SHIP && input_ctx.cursor_grabbed && !is_piloting {
                    let eye = ctx.smooth.render_position + DVec3::new(0.0, gpu::EYE_HEIGHT, 0.0);
                    let (sy, cy) = (ctx.cam.yaw as f32).sin_cos();
                    let (sp, cp) = (ctx.cam.pitch as f32).sin_cos();
                    let look = DVec3::new((cy * cp) as f64, sp as f64, (sy * cp) as f64);
                    let edit = voxeldust_core::client_message::BlockEditData {
                        action: voxeldust_core::client_message::action::OPEN_CONFIG, eye, look, block_type: 0,
                    };
                    send_tcp_msg(&actions.net, voxeldust_core::client_message::ClientMsg::BlockEditRequest(edit));
                }

                // Enter key: confirm warp.
                if key == KeyCode::Enter && is_piloting && warp_target.is_some() {
                    info!(target = ?warp_target, "warp confirmed via Enter");
                }

                // Escape: cancel warp target.
                if key == KeyCode::Escape && warp_target.is_some() {
                    actions.warp.target_star_index = None;
                    info!("warp target cancelled");
                }

                // Escape: ungrab cursor → MenuFocus mode.
                if key == KeyCode::Escape && input_ctx.cursor_grabbed {
                    input_ctx.mode = events::InputMode::MenuFocus;
                    input_ctx.cursor_grabbed = false;
                }
            }
        }
        events::InputMode::UiPanel => {
            for ev in key_events.read() {
                if ev.key == KeyCode::Escape {
                    config_closes.write(events::ConfigPanelCloseMsg);
                }
            }
        }
        events::InputMode::Animating | events::InputMode::MenuFocus => {}
    }
}

/// Interpret mouse buttons/scroll → cursor grab / block editing / thrust limiter.
fn interpret_mouse_system(
    ctx: PlayerCtx,
    mut actions: GameActionCtx,
    block_target: Res<BlockTarget>,
    mut mouse_events: MessageReader<events::MouseButtonMsg>,
    mut scroll_events: MessageReader<events::MouseScrollMsg>,
    mut input_ctx: ResMut<events::InputContext>,
) {
    // Mouse wheel → thrust limiter (only when piloting).
    if ctx.player.is_piloting && input_ctx.mode == events::InputMode::Game {
        for ev in scroll_events.read() {
            actions.flight.thrust_limiter = (actions.flight.thrust_limiter + ev.delta_y * 0.05).clamp(0.0, 1.0);
        }
    }
    match input_ctx.mode {
        events::InputMode::Game => {
            for ev in mouse_events.read() {
                if !input_ctx.cursor_grabbed {
                    if ev.pressed && ev.button == winit::event::MouseButton::Left {
                        input_ctx.mode = events::InputMode::Game;
                        input_ctx.cursor_grabbed = true; // Will be applied by apply_cursor_changes.
                    }
                } else if ev.pressed {
                    let is_piloting = ctx.player.is_piloting;
                    let shard_type = ctx.conn.current_shard_type;
                    if !is_piloting && shard_type == voxeldust_core::client_message::shard_type::SHIP {
                        if actions.sub_block_tool.active {
                            // Sub-block mode: LMB places, RMB removes.
                            if let Some(ref hit) = block_target.hit {
                                use voxeldust_core::client_message::{action, SubBlockEditData};
                                let sub_action = match ev.button {
                                    winit::event::MouseButton::Left => action::PLACE_SUB,
                                    winit::event::MouseButton::Right => action::REMOVE_SUB,
                                    _ => 0u8,
                                };
                                if sub_action != 0 {
                                    let face = voxeldust_core::block::sub_block::face_from_normal(hit.face_normal);
                                    let edit = SubBlockEditData {
                                        block_pos: hit.world_pos,
                                        face,
                                        element_type: actions.sub_block_tool.selected_type as u8,
                                        rotation: actions.sub_block_tool.rotation,
                                        action: sub_action,
                                    };
                                    send_tcp_msg(&actions.net, voxeldust_core::client_message::ClientMsg::SubBlockEdit(edit));
                                }
                            }
                        } else {
                            // Normal block mode: LMB breaks, RMB places.
                            let action = match ev.button {
                                winit::event::MouseButton::Left => voxeldust_core::client_message::action::BREAK,
                                winit::event::MouseButton::Right => voxeldust_core::client_message::action::PLACE,
                                _ => 0u8,
                            };
                            if action > 0 {
                                let eye = ctx.smooth.render_position + DVec3::new(0.0, gpu::EYE_HEIGHT, 0.0);
                                let (sy, cy) = (ctx.cam.yaw as f32).sin_cos();
                                let (sp, cp) = (ctx.cam.pitch as f32).sin_cos();
                                let look = DVec3::new((cy * cp) as f64, sp as f64, (sy * cp) as f64);
                                let block_type = if action == voxeldust_core::client_message::action::PLACE {
                                    actions.hotbar.selected_block().as_u16()
                                } else { 0 };
                                let edit = voxeldust_core::client_message::BlockEditData {
                                    action, eye, look, block_type,
                                };
                                send_tcp_msg(&actions.net, voxeldust_core::client_message::ClientMsg::BlockEditRequest(edit));
                            }
                        }
                    }
                }
            }
        }
        events::InputMode::MenuFocus => {
            for ev in mouse_events.read() {
                if ev.pressed && ev.button == winit::event::MouseButton::Left {
                    input_ctx.mode = events::InputMode::Game;
                    input_ctx.cursor_grabbed = true;
                }
            }
        }
        events::InputMode::UiPanel | events::InputMode::Animating => {}
    }
}

/// Config panel state machine: animation, input mode transitions, cursor management.
fn tick_config_panel_system(
    mut config_ui: ResMut<BlockConfigUIState>,
    mut input_ctx: ResMut<events::InputContext>,
    mut cam: ResMut<CameraControl>,
    mut zoom_offset: ResMut<CameraZoomOffset>,
    smooth: Res<RenderSmoothing>,
    block_target: Res<BlockTarget>,
    ft: Res<FrameTime>,
    mut open_msgs: MessageReader<events::ConfigPanelOpenMsg>,
    mut close_msgs: MessageReader<events::ConfigPanelCloseMsg>,
    mut cursor_grabs: MessageWriter<events::CursorGrabRequest>,
) {
    // Handle open requests (from network).
    for msg in open_msgs.read() {
        let config = &msg.config;
        let block_center = DVec3::new(
            config.block_pos.x as f64 + 0.5,
            config.block_pos.y as f64 + 0.5,
            config.block_pos.z as f64 + 0.5,
        );
        let eye = smooth.render_position + DVec3::new(0.0, gpu::EYE_HEIGHT, 0.0);

        // Use the block face normal from client raycast to align camera perpendicular
        // to the face — looking straight into the block surface, not at an angle.
        let face_normal = block_target.hit.as_ref()
            .filter(|h| h.face_normal != glam::IVec3::ZERO)
            .map(|h| DVec3::new(h.face_normal.x as f64, h.face_normal.y as f64, h.face_normal.z as f64))
            .unwrap_or_else(|| (block_center - eye).normalize());

        // Camera looks in the direction opposite to the face normal (into the face).
        let look_dir = -face_normal;
        let target_yaw = look_dir.z.atan2(look_dir.x);
        let target_pitch = look_dir.y.asin();

        // Zoom target: position camera directly in front of the face, 1.0m out.
        let face_center = block_center + face_normal * 0.5; // center of the block face
        let target_cam_pos = face_center + face_normal * 1.0; // 1m from the face
        let zoom_vec = target_cam_pos - eye;

        config_ui.block_world_pos = config.block_pos;
        config_ui.saved_yaw = cam.yaw;
        config_ui.saved_pitch = cam.pitch;
        config_ui.target_yaw = target_yaw;
        config_ui.target_pitch = target_pitch;
        config_ui.zoom_direction = zoom_vec.normalize();
        config_ui.zoom_distance = zoom_vec.length();
        config_ui.phase = ConfigPanelPhase::AnimatingIn { progress: 0.0 };
        config_ui.open_config = Some(config.clone());
        config_ui.is_open = true;
        zoom_offset.offset = DVec3::ZERO;
        input_ctx.mode = events::InputMode::Animating;
    }

    // Handle close requests (ESC, Cancel, Save).
    for _msg in close_msgs.read() {
        if matches!(config_ui.phase, ConfigPanelPhase::Interactive) {
            config_ui.open_config = None;
            config_ui.phase = ConfigPanelPhase::AnimatingOut { progress: 0.0 };
            input_ctx.mode = events::InputMode::Animating;
            cursor_grabs.write(events::CursorGrabRequest { grabbed: true });
        }
    }

    // Tick animation.
    let dt = ft.dt as f32;
    let (saved_yaw, saved_pitch, target_yaw, target_pitch) = (
        config_ui.saved_yaw, config_ui.saved_pitch,
        config_ui.target_yaw, config_ui.target_pitch,
    );

    let zoom_dir = config_ui.zoom_direction;
    let zoom_dist = config_ui.zoom_distance;

    match config_ui.phase {
        ConfigPanelPhase::AnimatingIn { progress } => {
            let new_progress = (progress + dt / CONFIG_ANIM_DURATION).min(1.0);
            if new_progress >= 1.0 {
                config_ui.phase = ConfigPanelPhase::Interactive;
                input_ctx.mode = events::InputMode::UiPanel;
                cursor_grabs.write(events::CursorGrabRequest { grabbed: false });
            } else {
                config_ui.phase = ConfigPanelPhase::AnimatingIn { progress: new_progress };
            }
            let t = smooth_step(new_progress.min(1.0));
            cam.yaw = lerp_angle(saved_yaw, target_yaw, t as f64);
            cam.pitch = lerp_f64(saved_pitch, target_pitch, t as f64);
            zoom_offset.offset = zoom_dir * zoom_dist * t as f64;
        }
        ConfigPanelPhase::AnimatingOut { progress } => {
            let new_progress = (progress + dt / CONFIG_ANIM_DURATION).min(1.0);
            if new_progress >= 1.0 {
                config_ui.phase = ConfigPanelPhase::Closed;
                config_ui.is_open = false;
                input_ctx.mode = events::InputMode::Game;
                cam.yaw = saved_yaw;
                cam.pitch = saved_pitch;
                zoom_offset.offset = DVec3::ZERO;
            } else {
                config_ui.phase = ConfigPanelPhase::AnimatingOut { progress: new_progress };
                let t = smooth_step(new_progress);
                cam.yaw = lerp_angle(target_yaw, saved_yaw, t as f64);
                cam.pitch = lerp_f64(target_pitch, saved_pitch, t as f64);
                // Reverse zoom: from full offset back to zero.
                zoom_offset.offset = zoom_dir * zoom_dist * (1.0 - t as f64);
            }
        }
        ConfigPanelPhase::Interactive => {
            // Maintain full zoom offset while panel is open.
            zoom_offset.offset = zoom_dir * zoom_dist;
        }
        ConfigPanelPhase::Closed => {}
    }
}

// ---------------------------------------------------------------------------
// send_input ECS system wrapper — avoids aliased ResMut<FlightControl>
// ---------------------------------------------------------------------------

fn send_input_system(
    net: Res<NetworkChannels>,
    kb: Res<KeyboardState>,
    mut flight: ResMut<FlightControl>,
    mut cam: ResMut<CameraControl>,
    player: Res<LocalPlayer>,
    ft: Res<FrameTime>,
    input_ctx: Res<events::InputContext>,
) {
    // Suppress movement input when not in Game mode.
    if input_ctx.mode != events::InputMode::Game {
        return;
    }
    let cam = &mut *cam;
    let limiter = flight.thrust_limiter;
    input::send_input_with_dt(
        &net.input_tx,
        &kb.keys_held,
        flight.engines_off,
        player.is_piloting,
        cam.yaw,
        cam.pitch,
        &mut cam.pilot_yaw_rate,
        &mut cam.pilot_pitch_rate,
        &mut flight.selected_thrust_tier,
        limiter,
        ft.count,
        ft.dt,
    );
}

// ---------------------------------------------------------------------------
// Winit ApplicationHandler
// ---------------------------------------------------------------------------

impl ApplicationHandler for ClientApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let attrs = WindowAttributes::default()
                .with_title("Voxydust")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            let window = Arc::new(event_loop.create_window(attrs).expect("window"));
            self.init_gpu(window);
            self.start_networking();
            info!("Voxydust ready");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Forward events to egui for UI panel interaction.
        if let (Some(gpu), Some(window)) = (&mut self.gpu, &self.window) {
            let _ = gpu.egui_winit.on_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    gpu.depth_view = gpu::create_depth_texture(&gpu.device, gpu.config.width, gpu.config.height, wgpu::TextureFormat::Depth32Float);
                }
            }
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if let PhysicalKey::Code(key) = key_event.physical_key {
                    let world = self.ecs_app.world_mut();
                    if key_event.state.is_pressed() {
                        world.resource_mut::<KeyboardState>().keys_held.insert(key);
                        world.resource_mut::<events::RawInputBuffer>().key_presses.push(
                            events::KeyPressedMsg { key },
                        );
                        // ESC with no cursor grab → quit.
                        let mode = world.resource::<events::InputContext>().mode;
                        let grabbed = world.resource::<events::InputContext>().cursor_grabbed;
                        if key == KeyCode::Escape && mode == events::InputMode::Game && !grabbed {
                            event_loop.exit();
                        }
                    } else {
                        world.resource_mut::<KeyboardState>().keys_held.remove(&key);
                        world.resource_mut::<events::RawInputBuffer>().key_releases.push(
                            events::KeyReleasedMsg { key },
                        );
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                };
                let world = self.ecs_app.world_mut();
                world.resource_mut::<events::RawInputBuffer>().mouse_scrolls.push(
                    events::MouseScrollMsg { delta_y: dy },
                );
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let world = self.ecs_app.world_mut();
                world.resource_mut::<events::RawInputBuffer>().mouse_buttons.push(
                    events::MouseButtonMsg { button, pressed: state.is_pressed() },
                );
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(ref window) = self.window { window.request_redraw(); }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _did: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            let world = self.ecs_app.world_mut();
            world.resource_mut::<events::RawInputBuffer>().mouse_motions.push(
                events::MouseMotionMsg { delta_x: delta.0, delta_y: delta.1 },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    info!("Voxydust starting");
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = ClientApp::new(args);
    event_loop.run_app(&mut app).expect("event loop error");
}
