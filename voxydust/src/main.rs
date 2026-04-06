//! Voxydust — first-person game client with server-authoritative movement.
//!
//! Game state lives in bevy_ecs Resources, updated by ECS systems each frame.
//! GPU state (GpuState, Window, uniform_data) stays outside the World because
//! egui_winit::State is !Send.

mod block_render;
mod camera;
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
}

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
    render_position: DVec3,
    prev_server_position: DVec3,
    walk_velocity: DVec3,
    last_server_update_time: Instant,
    has_prev_server_pos: bool,
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
                    if conn.current_shard_type == 2 {
                        player.ship_rotation = p.rotation;
                    }

                    if player.is_piloting && !was_piloting {
                        // First frame of piloting: sync camera yaw to ship heading
                        // so there's no visual snap.
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
                if shard_type == 2 {
                    player.ship_rotation = reference_rotation;
                }
                conn.connected = true;
                conn.system_seed = system_seed;
                conn.galaxy_seed = galaxy_seed;
                smooth.has_prev_server_pos = false;
                smooth.walk_velocity = DVec3::ZERO;

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
                    let edits: Vec<_> = cd.mods.iter()
                        .map(|m| (m.bx, m.by, m.bz, voxeldust_core::block::BlockId::from_u16(m.block_type)))
                        .collect();
                    chunk_cache.apply_delta(source, chunk_pos, cd.seq, &edits);
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
                    if conn.current_shard_type == 2 {
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
    ft: Res<FrameTime>,
) {
    // Smooth render position: lerp toward extrapolated target each frame.
    // This handles start/stop transitions gracefully — no snapping.
    if !player.is_piloting {
        let elapsed = (ft.last_time - smooth.last_server_update_time).as_secs_f64();
        let clamped = elapsed.min(0.06);
        let target = player.position + smooth.walk_velocity * clamped;
        // Exponential smoothing: converges ~90% within 2 server ticks (100ms).
        let blend = 1.0 - (-20.0 * ft.dt).exp();
        smooth.render_position = smooth.render_position + (target - smooth.render_position) * blend;
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
) -> Option<hud::WarpTargetInfo> {
    let target_idx = warp.target_star_index?;
    let sf = sf_res.0.as_ref()?;
    let star = sf.get_star(target_idx)?;

    // Reference position: warp camera position during warp, current star otherwise.
    let ref_pos = warp.galaxy_position.unwrap_or_else(|| {
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
    args: Args,
}

impl ClientApp {
    fn new(args: Args) -> Self {
        let mut ecs_app = bevy_app::App::new();

        // Insert resources with initial values.
        ecs_app.insert_resource(NetworkChannels {
            event_rx: None,
            input_tx: None,
        });
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
            selected_thrust_tier: 3, // default Long Range
            engines_off: false,
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

        // Register systems.
        ecs_app.add_systems(Update, update_frame_time.in_set(ClientSet::FrameTime));
        ecs_app.add_systems(Update, poll_network.in_set(ClientSet::Network));
        ecs_app.add_systems(Update, send_input_system.in_set(ClientSet::Input));
        ecs_app.add_systems(Update, smooth_render_position.in_set(ClientSet::Smooth));
        ecs_app.add_systems(Update, check_autopilot_timeout.in_set(ClientSet::AutopilotTimeout));
        ecs_app.add_systems(Update, compute_trajectory.in_set(ClientSet::Trajectory));

        Self {
            ecs_app,
            window: None,
            gpu: None,
            block_renderer: None,
            uniform_data: vec![bytemuck::Zeroable::zeroed(); MAX_OBJECTS],
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
        let input_rx = Arc::new(tokio::sync::Mutex::new(input_rx));

        let gateway = self.args.gateway;
        let name = self.args.name.clone();
        let direct = self.args.direct.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(network::run_network(gateway, name, event_tx, input_rx, direct));
        });

        let world = self.ecs_app.world_mut();
        let mut net = world.resource_mut::<NetworkChannels>();
        net.event_rx = Some(event_rx);
        net.input_tx = Some(input_tx);
    }

    fn render(&mut self) {
        // Run all ECS systems (frame time, network, input, smoothing, autopilot, trajectory).
        self.ecs_app.update();

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

            let warp_target_info = build_warp_target_info(warp, sf_res);

            // Compute camera.
            let cam = camera::compute_camera(
                smooth.render_position,
                cam_ctrl.yaw,
                cam_ctrl.pitch,
                player.is_piloting,
                conn.current_shard_type,
                player.ship_rotation,
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
                smooth.render_position,
                player.velocity,
                player.ship_rotation,
                player.is_piloting,
                conn.connected,
                conn.current_shard_type,
                flight.selected_thrust_tier,
                flight.engines_off,
                ap.target,
                warp.galaxy_position,
                warp.galaxy_rotation,
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
            let world_mut = self.ecs_app.world_mut();
            let mut chunk_cache = world_mut.resource_mut::<ClientChunkCacheRes>();
            let registry = voxeldust_core::block::BlockRegistry::new();

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
                    } else {
                        br.remove_chunk_mesh(dk);
                    }
                }
                tracing::info!(
                    total_gpu_chunks = br.total_chunk_count(),
                    "block renderer chunk update complete"
                );
            }
        }

        // Re-read resources for render_frame call.
        let world = self.ecs_app.world();
        let ws = world.resource::<ServerWorldState>();
        let ap = world.resource::<ClientAutopilot>();
        let sys_params = world.resource::<SystemParamsCache>();

        render::render_frame(
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
        );
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
) {
    let cam = &mut *cam;
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
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    let world = self.ecs_app.world_mut();
                    if event.state.is_pressed() {
                        {
                            let mut kb = world.resource_mut::<KeyboardState>();
                            kb.keys_held.insert(key);
                        }

                        // Read is_piloting for input handling.
                        let is_piloting = world.resource::<LocalPlayer>().is_piloting;
                        let warp_target = world.resource::<ClientWarp>().target_star_index;

                        // Autopilot: double-tap T = orbit, single-tap T = direct approach.
                        // Skip planet autopilot if a warp target is selected.
                        if key == KeyCode::KeyT && is_piloting && warp_target.is_none() {
                            let autopilot_target = world.resource::<ClientAutopilot>().target;
                            if autopilot_target.is_some() {
                                // Disengage.
                                let mut ap = world.resource_mut::<ClientAutopilot>();
                                ap.target = None;
                                ap.trajectory_plan = None;
                                ap.mode = AutopilotMode::DirectApproach;
                            } else {
                                let last_t = world.resource::<ClientAutopilot>().last_t_press;
                                let now = Instant::now();
                                let is_double = last_t
                                    .map(|prev| now.duration_since(prev) < std::time::Duration::from_millis(400))
                                    .unwrap_or(false);

                                if is_double {
                                    // Double-tap: orbit insertion mode — engage immediately.
                                    world.resource_mut::<ClientAutopilot>().last_t_press = None;
                                    // Need to call engage_autopilot_to_nearest with world access.
                                    let ws = world.resource::<ServerWorldState>();
                                    let player = world.resource::<LocalPlayer>();
                                    let ship_rotation = player.ship_rotation;
                                    let _is_piloting_now = player.is_piloting;
                                    // Find best target.
                                    let best = if let Some(ref world_state) = ws.latest {
                                        let ship_fwd = ship_rotation * DVec3::NEG_Z;
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
                                    } else {
                                        None
                                    };
                                    if let Some((idx, _)) = best {
                                        let mut ap = world.resource_mut::<ClientAutopilot>();
                                        ap.target = Some(idx);
                                        ap.mode = AutopilotMode::OrbitInsertion;
                                        info!(planet = idx, mode = ?AutopilotMode::OrbitInsertion, "autopilot engaged");
                                    }
                                } else {
                                    // First tap: record time, wait for possible second tap.
                                    world.resource_mut::<ClientAutopilot>().last_t_press = Some(now);
                                }
                            }
                        }

                        // WASD cancels autopilot.
                        if world.resource::<ClientAutopilot>().target.is_some() && matches!(key,
                            KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD) {
                            let mut ap = world.resource_mut::<ClientAutopilot>();
                            ap.target = None;
                            ap.trajectory_plan = None;
                            ap.mode = AutopilotMode::DirectApproach;
                        }

                        // X key: toggle engine cutoff (SC-style decoupled mode).
                        if key == KeyCode::KeyX && is_piloting {
                            let mut flight = world.resource_mut::<FlightControl>();
                            flight.engines_off = !flight.engines_off;
                            info!(engines_off = flight.engines_off, "engine cutoff toggled");
                        }

                        // Enter key: confirm warp to targeted star.
                        if key == KeyCode::Enter && is_piloting && warp_target.is_some() {
                            info!(target = ?warp_target, "warp confirmed via Enter");
                        }

                        // Escape key: cancel warp target.
                        if key == KeyCode::Escape && warp_target.is_some() {
                            world.resource_mut::<ClientWarp>().target_star_index = None;
                            info!("warp target cancelled");
                        }

                        // G key: warp target cycling is server-authoritative.
                        // The ship shard handles action=6 and sends the selected
                        // star index via WorldState.warp_target_star_index.

                        if key == KeyCode::Escape {
                            let mouse_grabbed = world.resource::<KeyboardState>().mouse_grabbed;
                            if mouse_grabbed {
                                if let Some(ref w) = self.window {
                                    let _ = w.set_cursor_grab(CursorGrabMode::None);
                                    w.set_cursor_visible(true);
                                    world.resource_mut::<KeyboardState>().mouse_grabbed = false;
                                }
                            } else {
                                event_loop.exit();
                            }
                        }
                    } else {
                        let mut kb = world.resource_mut::<KeyboardState>();
                        kb.keys_held.remove(&key);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let world = self.ecs_app.world_mut();
                let mouse_grabbed = world.resource::<KeyboardState>().mouse_grabbed;
                if state.is_pressed() && button == winit::event::MouseButton::Left && !mouse_grabbed {
                    if let Some(ref w) = self.window {
                        if w.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                            let _ = w.set_cursor_grab(CursorGrabMode::Confined);
                        }
                        w.set_cursor_visible(false);
                        world.resource_mut::<KeyboardState>().mouse_grabbed = true;
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(ref window) = self.window { window.request_redraw(); }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _did: DeviceId, event: DeviceEvent) {
        let world = self.ecs_app.world_mut();
        let mouse_grabbed = world.resource::<KeyboardState>().mouse_grabbed;
        if !mouse_grabbed { return; }
        if let DeviceEvent::MouseMotion { delta } = event {
            let sensitivity = 0.003;
            let is_piloting = world.resource::<LocalPlayer>().is_piloting;
            let free_look = world.resource::<KeyboardState>().keys_held.contains(&KeyCode::AltLeft);

            if is_piloting && !free_look {
                // Piloting: set yaw/pitch rate from mouse movement.
                // Rate is proportional to mouse velocity, clamped to [-1, 1].
                let mut cam = world.resource_mut::<CameraControl>();
                cam.pilot_yaw_rate = (cam.pilot_yaw_rate - delta.0 * sensitivity * 5.0).clamp(-1.0, 1.0);
                cam.pilot_pitch_rate = (cam.pilot_pitch_rate - delta.1 * sensitivity * 5.0).clamp(-1.0, 1.0);
            } else {
                // Walking or free-look: move camera yaw/pitch directly.
                let mut cam = world.resource_mut::<CameraControl>();
                cam.yaw += delta.0 * sensitivity;
                cam.pitch -= delta.1 * sensitivity;
                cam.pitch = cam.pitch.clamp(
                    -std::f64::consts::FRAC_PI_2 + 0.01,
                    std::f64::consts::FRAC_PI_2 - 0.01,
                );
            }
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
