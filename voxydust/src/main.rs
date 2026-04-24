//! Voxydust — first-person game client with server-authoritative movement.
//!
//! Game state lives in bevy_ecs Resources, updated by ECS systems each frame.
//! GPU state (GpuState, Window, uniform_data) stays outside the World because
//! egui_winit::State is !Send.

mod block_render;
mod camera;
mod cloud_system;
mod events;
mod gpu;
pub mod graphics_settings;
mod hud;
mod input;
mod mesh;
mod network;
mod render;
mod stars;
mod voxel_volume;

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
use voxeldust_core::client_message::{CelestialBodyData, PlayerInputData, WorldStateData};
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
    /// Graphics quality preset: low, medium, high, ultra
    #[arg(long, default_value = "medium")]
    graphics: String,
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
                BlockId::HULL_ARMORED,
                BlockId::WINDOW,
                BlockId::COCKPIT,             // seat (functional)
                BlockId::REACTOR_SMALL,       // power source
                BlockId::ROTOR,               // mechanical (also placeable as full block)
                BlockId::PISTON,              // mechanical
                BlockId::SEAT,                // generic seat
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
    /// Shard seed — for ship shards this is the ship_id, used to filter
    /// the player's own ship from secondary WorldState rendering.
    shard_seed: u64,
    reference_position: DVec3,
    reference_rotation: DQuat,
    secondary_shard_type: Option<u8>,
    /// Target shard type during a seamless promotion. The render/logic code
    /// keeps using `current_shard_type` (still the OLD primary's type) until
    /// the new primary's first WorldState that contains our player arrives —
    /// at which point we flip `current_shard_type = pending_shard_type.take()`.
    /// This prevents the interior-ship render branch from activating with a
    /// stale player_position during the ~100-300 ms reconnect window.
    pending_shard_type: Option<u8>,
    system_seed: u64,
    galaxy_seed: u64,
    /// Our player's id (session token) on the current primary shard.
    /// Set from `JoinResponse.player_id` on every `Connected`. The
    /// WorldState handler MUST filter `players[]` by this — system-
    /// shard EVA broadcasts include every EVA player in the shard,
    /// not just the recipient, so `players.first()` would pick an
    /// arbitrary other player's position and snap the camera there.
    own_player_id: u64,
}

#[derive(Resource)]
struct ServerWorldState {
    latest: Option<WorldStateData>,
    /// Last-received secondary WorldState, ANY shard type. Kept for
    /// backward-compat with code paths that expect a single slot.
    /// For Transitioning promotion, prefer `secondary_by_type`.
    secondary: Option<WorldStateData>,
    /// WorldState per secondary shard type. Multiple secondaries can
    /// be connected at once (SYSTEM + PLANET while flying in
    /// atmosphere), each broadcasting at ~20 Hz; without per-type
    /// storage, `secondary` gets overwritten non-deterministically and
    /// Transitioning promotes the wrong shard — e.g. on Ship→System
    /// exit, `ws.secondary` might be the PLANET WS (origin ≈ 3.58 Gm)
    /// instead of SYSTEM (origin = 0), and `planet_local = live_ship_origin
    /// - planet_pos = 0` → camera rendered at the star.
    secondary_by_type: std::collections::HashMap<u8, WorldStateData>,
    /// Last-known primary WorldState, stashed on a hard transition so the
    /// renderer can keep drawing ships/players/entities from it for a short
    /// grace period instead of showing a blank world until the new primary's
    /// first WorldState arrives (previously a 200-500 ms blackout).
    /// Cleared as soon as `latest` is populated or after the grace expires.
    last_primary: Option<WorldStateData>,
    last_primary_until: Option<Instant>,
    /// Ship the client just left. Populated on SHIP→!SHIP Transitioning,
    /// cleared when expiry passes. Co-located with `last_primary` because
    /// they share the same grace window and lifecycle.
    departed_ship: Option<DepartedShipData>,
    /// If set, force per-frame diagnostic logging until this instant.
    /// Set by `Transitioning` for a short window so we capture exactly
    /// what the renderer saw during the blink.
    trace_render_until: Option<Instant>,
}

/// Grace period during which `last_primary` is rendered after a hard transition.
const LAST_PRIMARY_GRACE_SECS: f64 = 1.5;

/// Ship the client just left (SHIP → !SHIP transition), captured so the
/// renderer can draw it as an exterior mesh in system-space during the
/// grace window. See `ServerWorldState::departed_ship`.
struct DepartedShipData {
    /// Chunk source on GPU; holds the ship's block meshes. Sourced from the
    /// client's old primary or secondary source. Freed when the entry in
    /// `ClientChunkCacheRes::grace_sources` expires, NOT here.
    source: voxeldust_core::block::client_chunks::ChunkSourceId,
    /// Ship center in absolute system-space at the moment of exit.
    world_position: glam::DVec3,
    /// Ship orientation in world-space at the moment of exit.
    rotation: glam::DQuat,
    /// Conservative bounding sphere radius (m) of the ship's block mesh,
    /// used for frustum culling. Taken from the own-ship entity's
    /// `bounding_radius` (derived from hull AABB on ship-shard), so
    /// complex / fractional hulls are sized correctly — not a guess.
    bounding_radius: f32,
    /// When to stop rendering this ship. Matches `LAST_PRIMARY_GRACE_SECS`.
    expiry: Instant,
}

impl ServerWorldState {
    /// Returns the WorldState to show in the 3D/HUD scene this frame. Prefers
    /// the live primary (`latest`); if the primary was just torn down on a
    /// hard transition, falls back to the stashed `last_primary` until the
    /// grace window expires. Used for ship/entity rendering so the world
    /// doesn't go blank during a reconnect. Camera logic intentionally keeps
    /// using `latest` directly (to avoid animating the own avatar from stale
    /// data) — only the scene-render path is allowed to fall back here.
    fn effective(&self) -> Option<&WorldStateData> {
        if self.latest.is_some() {
            return self.latest.as_ref();
        }
        if let Some(deadline) = self.last_primary_until {
            if Instant::now() <= deadline {
                return self.last_primary.as_ref();
            }
        }
        None
    }

    /// Returns `last_primary` iff still within the grace window, regardless of
    /// whether `latest` is populated. Used by the renderer to keep drawing
    /// cross-shard entities (e.g., the ship's LOD proxy from the old system WS)
    /// for a brief overlap during a seamless promotion so the visual doesn't
    /// pop when `latest` is swapped to the promoted secondary.
    fn grace_fallback(&self) -> Option<&WorldStateData> {
        if let Some(deadline) = self.last_primary_until {
            if Instant::now() <= deadline {
                return self.last_primary.as_ref();
            }
        }
        None
    }
}

/// Client-side chunk cache for block data received from the server.
/// Multi-source: supports simultaneous ship + planet chunks during transitions.
#[derive(Resource)]
struct ClientChunkCacheRes {
    cache: voxeldust_core::block::client_chunks::ClientChunkCache,
    /// Source ID for the current primary shard connection.
    /// Set when a new shard connection provides chunk data.
    primary_source: Option<voxeldust_core::block::client_chunks::ChunkSourceId>,
    /// Shard seed that `primary_source` holds chunks for. Stored so that
    /// on a new primary `Connected` event we can promote a matching
    /// pre-observed secondary source (zero re-upload, zero visual gap)
    /// and — critically — tell when the old primary's chunks are
    /// redundant with a new secondary covering the same seed.
    primary_seed: Option<u64>,
    /// Source IDs for secondary (observer) connections, keyed by seed (ship_id).
    /// Created on SecondaryConnected, populated by SecondaryChunkSnapshot,
    /// removed on SecondaryDisconnected (or when promoted to primary).
    secondary_sources: std::collections::HashMap<u64, voxeldust_core::block::client_chunks::ChunkSourceId>,
    /// Sources kept alive across the shard-switch seam so the renderer can
    /// still draw a ship whose source just changed role (old primary freed,
    /// or secondary promoted to primary). Keyed by the source's original
    /// seed. Split from `grace_expiries` so the renderer can look up by
    /// seed without allocating. Both maps stay in sync: every entry in
    /// `grace_sources` has a matching entry in `grace_expiries`.
    /// `LAST_PRIMARY_GRACE_SECS` — mirrors the `last_primary` WorldState
    /// grace. Drained by `drain_expired_grace_sources`.
    grace_sources: std::collections::HashMap<u64, voxeldust_core::block::client_chunks::ChunkSourceId>,
    grace_expiries: std::collections::HashMap<u64, Instant>,
}

impl std::ops::Deref for ClientChunkCacheRes {
    type Target = voxeldust_core::block::client_chunks::ClientChunkCache;
    fn deref(&self) -> &Self::Target { &self.cache }
}

impl std::ops::DerefMut for ClientChunkCacheRes {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.cache }
}

/// Sub-grid state: block assignments and body transforms for mechanical mounts.
/// Sub-grid state: block→sub-grid assignments for meshing.
/// Transform interpolation is handled by the tick-based SnapshotBuffer/RenderSmoothing.
#[derive(Resource, Default)]
struct SubGridState {
    /// Per-block sub-grid assignment: world_pos → sub_grid_id (absent = root).
    assignments: std::collections::HashMap<glam::IVec3, u32>,
}

#[derive(Resource)]
struct LocalPlayer {
    position: DVec3,
    velocity: DVec3,
    is_piloting: bool,
    ship_rotation: DQuat,
    /// Body orientation in world (system) frame when on SYSTEM shard
    /// as an EVA player. Persists across ticks — EVA in vacuum has no
    /// gravity "up", so the player keeps whatever orientation they
    /// inherited from the ship they left (or from thruster input).
    /// `cam.yaw/pitch` are head rotation RELATIVE to this body frame,
    /// so inside or outside the ship, the same scalar yaw/pitch
    /// produce a continuous world-space view (no snap at handoff).
    body_rotation: DQuat,
}

/// Interpolated render state — output of snapshot interpolation, consumed by camera + renderer.
#[derive(Resource)]
struct RenderSmoothing {
    render_position: DVec3,
    render_rotation: DQuat,
    /// Interpolated ship system-space origin (for exterior rendering / cam_system_pos).
    ship_origin: DVec3,
    /// Interpolated celestial body positions — lerped with the same `t` as ship_origin
    /// so body offsets and camera share the same virtual time instant.
    bodies: Vec<CelestialBodyData>,
    /// Current interpolation factor (0=prev snapshot, 1=current snapshot).
    /// Used by sub-grid rendering to match ship transform interpolation.
    interpolation_t: f64,
    /// Interpolated sub-grid transforms (mechanical mounts), produced by the same
    /// tick-based interpolation as position/rotation.
    sub_grid_transforms: std::collections::HashMap<u32, voxeldust_core::client_message::SubGridTransformData>,

    // --- Handoff blend (seamless primary-switch smoothing) ---
    /// When a primary switch occurs (secondary promoted to primary), the camera
    /// world-space position is held at `handoff_start_world` and lerped toward
    /// the new primary's authoritative position over `HANDOFF_BLEND_SECS`.
    /// `None` when no blend is active.
    handoff_start_world: Option<DVec3>,
    /// Instant at which the handoff blend started.
    handoff_started_at: Instant,
    /// Ship/player velocity (m/s, system-space) at the moment of the primary
    /// switch. `handoff_start_world` is extrapolated by this velocity over
    /// the blend duration so the reference point moves WITH the ship's
    /// orbital motion (110 km/s at planet orbital velocity). Without this
    /// the camera would appear to lag behind the ship by `velocity *
    /// blend_time` (~16.5 km over 150 ms) as the authoritative target
    /// position in the new shard advances each tick. Purely dead-reckoning
    /// of an authoritative server-provided velocity — no client prediction
    /// of position is involved.
    handoff_velocity: DVec3,

    /// Post-transition anchor (legacy). Not used for re-anchoring
    /// anymore — server-authoritative spawn positions in the
    /// ShardRedirect/handoff make client-side prediction unnecessary.
    /// Kept only to track whether a Transitioning snap is still the
    /// sole snapshot in the buffer.
    exit_ship_anchor: Option<DVec3>,

    // --- Galaxy warp (separate interpolation, kept for warp travel) ---
    galaxy_render_position: DVec3,
    galaxy_render_rotation: DQuat,
    galaxy_velocity: DVec3,
    prev_galaxy_position: DVec3,
    prev_galaxy_rotation: DQuat,
    last_galaxy_update_time: Instant,
    has_prev_galaxy_pos: bool,
}

/// Handoff blend window — matches the server's 15-tick ghost-update tail
/// (150 ms at 20 Hz) so the source-shard ghost covers the same window the
/// client is interpolating across. Interpolation only — no extrapolation.
const HANDOFF_BLEND_SECS: f64 = 0.15;

/// Server tick duration (20Hz = 50ms).
const TICK_DURATION_SECS: f64 = 1.0 / 20.0;
/// Interpolation delay in server ticks. 3 ticks = 150ms — survives 2 consecutive
/// packet losses and typical internet jitter at 20Hz.
const INTERP_DELAY_TICKS: f64 = 3.0;
/// Ring buffer capacity in snapshots (~1.6s of history at 20Hz).
const SNAPSHOT_BUFFER_CAP: usize = 32;

/// A server snapshot stored by tick for interpolation.
#[derive(Clone)]
struct TimedSnapshot {
    tick: u64,
    player_position: DVec3,
    ship_rotation: DQuat,
    ship_origin: DVec3,
    velocity: DVec3,
    bodies: Vec<CelestialBodyData>,
    sub_grids: std::collections::HashMap<u32, voxeldust_core::client_message::SubGridTransformData>,
}

/// Ring buffer of recent server snapshots, sorted by tick.
/// Interpolation uses server tick numbers (not arrival timestamps) to compute
/// the interpolation fraction, making it immune to client-side message batching.
#[derive(Resource)]
struct SnapshotBuffer {
    /// Sorted by tick ascending. Oldest at front, newest at back.
    snapshots: std::collections::VecDeque<TimedSnapshot>,
    /// Highest tick ever received (for rejecting stale packets).
    highest_tick: u64,
    /// Client's estimate of the current server tick (sub-tick precision).
    estimated_server_tick: f64,
    /// Whether we've received the first snapshot and synced the clock.
    synced: bool,
    /// Wall-clock anchor: the `Instant` that corresponds to
    /// `anchor_server_tick`. Updated when a WorldState arrives so
    /// `estimated_server_tick = anchor_server_tick + (Instant::now() -
    /// anchor_time) / TICK_DURATION_SECS`. This makes the tick estimate
    /// advance at a **constant real-time rate** instead of accumulating
    /// per-frame `ft.dt`, which amplifies frame-rate jitter into visible
    /// per-frame position steps (the "patchy walking" feel). Frame
    /// stutters up to `~1 tick` now produce a larger step once and then
    /// self-correct, rather than shrinking/growing every step with fps.
    anchor_time: Option<std::time::Instant>,
    anchor_server_tick: f64,
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
    mouse_buttons_held: HashSet<winit::event::MouseButton>,
    mouse_grabbed: bool,
    /// Accumulated scroll delta this frame (for seat scroll wheel bindings).
    scroll_delta: f32,
}

#[derive(Resource)]
struct FlightControl {
    selected_thrust_tier: u8,
    engines_off: bool,
    /// Thrust limiter (0.0–1.0), adjusted by mouse wheel.
    thrust_limiter: f32,
    /// Supercruise mode (C key toggle). 100-1000 km/s in-system travel.
    cruise_active: bool,
    /// Atmosphere compensation (hover) mode (H key toggle).
    atmo_comp_active: bool,
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
    mut snapshots: ResMut<SnapshotBuffer>,
    mut cam: ResMut<CameraControl>,
    mut ap: ResMut<ClientAutopilot>,
    mut warp: ResMut<ClientWarp>,
    mut sf_res: ResMut<StarFieldRes>,
    mut sys_params: ResMut<SystemParamsCache>,
    mut chunk_cache: ResMut<ClientChunkCacheRes>,
    ft: Res<FrameTime>,
    mut config_open_msgs: MessageWriter<events::ConfigPanelOpenMsg>,
    mut active_seat: ResMut<input::ActiveSeatBindings>,
    mut sub_grid_state: ResMut<SubGridState>,
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

                // Insert snapshot into ring buffer for tick-based
                // interpolation. MUST filter by `own_player_id`, not
                // `first()`: system-shard EVA broadcasts include every
                // EVA player in the shard, so `first()` can return a
                // totally different player and snap our camera to them.
                // When our player hasn't been spawned server-side yet
                // (the handoff is still in flight), `find()` returns
                // None and we intentionally insert no snapshot,
                // leaving the Transitioning fallback snap intact so
                // the post-transition anchor code can keep the camera
                // on the ship.
                if let Some(p) = world_state.players.iter()
                    .find(|p| p.player_id == conn.own_player_id)
                {
                    let tick = world_state.tick;
                    let new_pos = p.position;

                    let snapshot = TimedSnapshot {
                        tick,
                        player_position: new_pos,
                        ship_rotation: p.rotation,
                        ship_origin: world_state.origin,
                        velocity: p.velocity,
                        bodies: world_state.bodies.clone(),
                        sub_grids: world_state.sub_grids.iter()
                            .map(|sg| (sg.sub_grid_id, sg.clone())).collect(),
                    };

                    // Teleport detection: clear buffer on large, discontinuous
                    // jumps. The threshold is velocity-aware — the previous
                    // snapshot's velocity × dt predicts the expected per-tick
                    // delta. Anything substantially larger than that is a real
                    // teleport (warp arrival, respawn, handoff). A fixed 10 m
                    // threshold (the old version) was broken for system-shard
                    // EVA, where the player legitimately moves at ship velocity
                    // (km/s) — every tick was (incorrectly) classified as a
                    // teleport and the interpolation buffer never filled.
                    //
                    // `unwrap_or(true)`: on an empty buffer (first WS after a
                    // shard transition) we always snap immediately.
                    // Threshold strategy: use the PREVIOUS per-tick
                    // position delta as the reference, with a margin. This
                    // is independent of the server's reported `velocity`
                    // field — which on EVA reports velocity relative to
                    // a local frame (e.g. ship- or planet-relative) and
                    // does not capture the large orbital motion that
                    // manifests as ~5000 m per-tick deltas in system-
                    // space coordinates. Position deltas always reflect
                    // the true per-tick displacement.
                    //
                    // Requires at least 2 prior snapshots to establish
                    // a delta; for a 1-element buffer we fall back to a
                    // generous 100 km floor (catches warp arrivals,
                    // not normal motion at any velocity < 2000 km/s).
                    const TELEPORT_DELTA_MARGIN: f64 = 5.0;
                    const TELEPORT_MIN_FLOOR_M: f64 = 100.0;
                    const TELEPORT_HARD_FLOOR_M: f64 = 100_000.0;
                    let is_teleport = {
                        let snaps = &snapshots.snapshots;
                        let n = snaps.len();
                        if n == 0 {
                            true
                        } else if n == 1 {
                            let last = &snaps[0];
                            let delta = (new_pos - last.player_position).length();
                            delta > TELEPORT_HARD_FLOOR_M
                        } else {
                            let last = &snaps[n - 1];
                            let prev = &snaps[n - 2];
                            let prev_delta =
                                (last.player_position - prev.player_position).length();
                            let delta = (new_pos - last.player_position).length();
                            let threshold = (prev_delta * TELEPORT_DELTA_MARGIN)
                                .max(TELEPORT_MIN_FLOOR_M);
                            delta > threshold
                        }
                    };
                    if is_teleport {
                        snapshots.snapshots.clear();
                        snapshots.estimated_server_tick = tick as f64;
                        snapshots.anchor_time = Some(std::time::Instant::now());
                        snapshots.anchor_server_tick = tick as f64;
                        smooth.render_position = new_pos;
                        smooth.render_rotation = p.rotation;
                        smooth.ship_origin = world_state.origin;
                        smooth.bodies = world_state.bodies.clone();
                        info!(
                            tick,
                            new_pos = format!("({:.0},{:.0},{:.0})", new_pos.x, new_pos.y, new_pos.z),
                            origin = format!("({:.0},{:.0},{:.0})", world_state.origin.x, world_state.origin.y, world_state.origin.z),
                            players = world_state.players.len(),
                            "teleport snap — render_position updated"
                        );
                    }

                    // Sorted insert (handles out-of-order UDP).
                    if tick > snapshots.highest_tick.saturating_sub(SNAPSHOT_BUFFER_CAP as u64) {
                        let pos = snapshots.snapshots.partition_point(|s| s.tick < tick);
                        if snapshots.snapshots.get(pos).map_or(true, |s| s.tick != tick) {
                            snapshots.snapshots.insert(pos, snapshot);
                        }
                    }
                    snapshots.highest_tick = snapshots.highest_tick.max(tick);

                    // Evict old snapshots beyond buffer capacity.
                    let cutoff = snapshots.highest_tick.saturating_sub(SNAPSHOT_BUFFER_CAP as u64);
                    while snapshots.snapshots.front().map_or(false, |s| s.tick < cutoff) {
                        snapshots.snapshots.pop_front();
                    }

                    // Initialize clock on first snapshot.
                    if !snapshots.synced {
                        snapshots.estimated_server_tick = tick as f64;
                        snapshots.anchor_time = Some(std::time::Instant::now());
                        snapshots.anchor_server_tick = tick as f64;
                        snapshots.synced = true;
                        info!(
                            tick,
                            pos = format!("({:.0},{:.0},{:.0})", new_pos.x, new_pos.y, new_pos.z),
                            render_pos = format!("({:.0},{:.0},{:.0})", smooth.render_position.x, smooth.render_position.y, smooth.render_position.z),
                            "synced on first snapshot"
                        );
                    }

                    // Gentle clock drift correction: blend anchor toward
                    // a new anchor pinned to the just-arrived server
                    // tick. We rebuild `anchor_time` + `anchor_server_tick`
                    // so that the wall-clock interpolation sees a smooth
                    // catch-up (10 % per packet = 0.5 s half-life) without
                    // any per-frame-dt accumulation.
                    let now = std::time::Instant::now();
                    let prev_anchor_time =
                        snapshots.anchor_time.unwrap_or(now);
                    let prev_anchor_tick = snapshots.anchor_server_tick;
                    // What the current anchor would predict for "right now":
                    let predicted_now =
                        prev_anchor_tick + (now - prev_anchor_time).as_secs_f64() / TICK_DURATION_SECS;
                    let server_now = tick as f64;
                    // Blend the predicted tick toward the server tick.
                    let blended = predicted_now + (server_now - predicted_now) * 0.1;
                    snapshots.anchor_time = Some(now);
                    snapshots.anchor_server_tick = blended;
                    snapshots.estimated_server_tick = blended;

                    // Game logic (non-rendering).
                    player.position = new_pos;
                    player.velocity = p.velocity;
                    let was_piloting = player.is_piloting;
                    player.is_piloting = p.seated;

                    if was_piloting != player.is_piloting {
                        info!(piloting = player.is_piloting, grounded = p.grounded, frame = ft.count, "pilot mode changed");
                    }

                    if conn.current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
                        player.ship_rotation = p.rotation;
                    }
                    // On SYSTEM shard (EVA), the player's WS rotation
                    // field IS their body orientation in world/system
                    // space — server persists it across ticks as there
                    // is no gravity "up" to re-normalize to. Sync to the
                    // client so camera.rs can render head rotation as
                    // body-relative, matching the pre-exit view 1:1.
                    if conn.current_shard_type
                        == voxeldust_core::client_message::shard_type::SYSTEM
                    {
                        player.body_rotation = p.rotation;
                    }

                    if player.is_piloting && !was_piloting {
                        let fwd = p.rotation * DVec3::NEG_Z;
                        cam.yaw = fwd.z.atan2(fwd.x) as f64;
                        cam.pitch = fwd.y.asin() as f64;
                        cam.pilot_yaw_rate = 0.0;
                        cam.pilot_pitch_rate = 0.0;
                    }

                    if was_piloting && !player.is_piloting {
                        cam.yaw = -std::f64::consts::FRAC_PI_2;
                        cam.pitch = 0.0;
                        active_seat.clear();
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
                // New primary data — the stash is no longer needed.
                ws.last_primary = None;
                ws.last_primary_until = None;
            }
            NetEvent::Connected { shard_type, seed, reference_position, reference_rotation, system_seed, galaxy_seed, player_id, .. } => {
                conn.own_player_id = player_id;
                conn.current_shard_type = shard_type;
                // New primary is live — pending flip is satisfied. Clear so
                // the WorldState-level fallback flip doesn't fire redundantly.
                conn.pending_shard_type = None;
                conn.shard_seed = seed;
                conn.reference_position = reference_position;
                conn.reference_rotation = reference_rotation;

                // If we were already observing this shard as a secondary
                // (typical on system → ship boarding: the ship was a
                // secondary for nearby rendering), promote its chunk source
                // to primary instead of allocating a fresh one. This both
                // avoids re-downloading + re-uploading ~8 MB of chunk data
                // AND prevents a triple-allocation leak where the old
                // primary, the now-redundant secondary, and the freshly
                // allocated primary all sit in GPU memory together.
                // If Transitioning's eager-promotion already set
                // `primary_source` + `primary_seed` for THIS shard
                // (common boarding path), the work is already done.
                // Do NOT `take()` + free — the "old primary" IS the
                // very source we just promoted; freeing it removes
                // the GPU buffers we need.
                let already_promoted = chunk_cache.primary_seed == Some(seed)
                    && chunk_cache.primary_source.is_some();
                if !already_promoted {
                    if let Some(old_primary) = chunk_cache.primary_source.take() {
                        // New `Connected` supersedes any previous primary
                        // (unrelated shard seed). Free it.
                        chunk_cache.cache.remove_source(old_primary);
                        info!(source = old_primary.0, "freed stale primary source on new Connected");
                    }
                    let grace_expiry = Instant::now()
                        + std::time::Duration::from_secs_f64(LAST_PRIMARY_GRACE_SECS);
                    // Prefer secondary_sources (normal promote), then
                    // grace_sources (either from a prior eager-promote at
                    // Transitioning, or from SecondaryDisconnected rescuing
                    // the source before this Connected was dispatched). In
                    // all cases the source already has chunks uploaded — we
                    // just rebind it without re-downloading.
                    let sec_src = chunk_cache.secondary_sources.remove(&seed)
                        .or_else(|| chunk_cache.grace_sources.get(&seed).copied());
                    if let Some(sec_src) = sec_src {
                        chunk_cache.primary_source = Some(sec_src);
                        chunk_cache.grace_sources.insert(seed, sec_src);
                        chunk_cache.grace_expiries.insert(seed, grace_expiry);
                        info!(seed, source = sec_src.0, "promoted pre-observed secondary to primary (grace-pinned)");
                    }
                    chunk_cache.primary_seed = Some(seed);
                } else {
                    info!(seed, source = ?chunk_cache.primary_source.map(|s| s.0),
                        "Connected: primary already eagerly promoted at Transitioning, keeping");
                }
                if shard_type == voxeldust_core::client_message::shard_type::SHIP {
                    player.ship_rotation = reference_rotation;
                }
                conn.connected = true;
                conn.system_seed = system_seed;
                conn.galaxy_seed = galaxy_seed;
                // Reset interpolation — but only if snapshots are empty
                // (fresh connect). After a seamless transition, snapshots
                // were already initialized from the promoted secondary WS.
                // Clearing them would cause a blink until 2 new WorldStates arrive.
                if snapshots.snapshots.is_empty() {
                    // Reset render state for the new shard's coordinate frame.
                    // Position set to zero — the first WorldState will snap it
                    // to the correct value via teleport detection (unwrap_or(true)).
                    // We do NOT use reference_position here because for ship shards
                    // it's the ship's system-space position (~1e10), not a player
                    // position — using it would overflow the i32 raycast.
                    smooth.render_position = DVec3::ZERO;
                    smooth.render_rotation = reference_rotation;
                    smooth.bodies.clear();
                }
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

                // Create a chunk source for this secondary connection's block data.
                // Ship secondaries provide their own chunk data via observer TCP.
                // If a secondary with the same seed is already tracked (reconnect
                // after network hiccup, or pre-connect echo), reuse its source —
                // chunks are already cached, no re-upload needed.
                if shard_type == voxeldust_core::client_message::shard_type::SHIP {
                    if !chunk_cache.secondary_sources.contains_key(&seed) {
                        let source = chunk_cache.cache.add_source();
                        chunk_cache.secondary_sources.insert(seed, source);
                        info!(seed, source = source.0, "created secondary chunk source for ship observer");
                    }
                }

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
            NetEvent::SecondaryDisconnected { seed } => {
                // Secondary connection ended. DO NOT free the chunk source
                // immediately — the common trigger is `ShardRedirect`
                // cancelling the current-shard observer a moment BEFORE
                // the Transitioning event that would promote this same
                // source to primary (observed ordering: network emits
                // SecondaryDisconnected then Transitioning, main loop
                // processes them in that order, so a free here destroys
                // the exact source the next Transitioning will try to
                // promote — boarding blink).
                //
                // Instead move the source into `grace_sources` for the
                // standard grace window. If Transitioning + Connected
                // promote it the entry will be overwritten there;
                // otherwise the periodic cleanup releases it after the
                // window expires, same as any other grace entry.
                if let Some(src) = chunk_cache.secondary_sources.remove(&seed) {
                    let expiry = Instant::now()
                        + std::time::Duration::from_secs_f64(LAST_PRIMARY_GRACE_SECS);
                    chunk_cache.grace_sources.insert(seed, src);
                    chunk_cache.grace_expiries.insert(seed, expiry);
                    info!(seed, source = src.0, "grace-pinned secondary source on disconnect");
                }
            }
            NetEvent::SecondaryWorldState { shard_type, ws: secondary_ws } => {
                // Store by shard_type so Transitioning can promote the
                // correct secondary for the redirect target. Also keep
                // the single-slot `ws.secondary` updated for rendering
                // code that uses whichever is most recent.
                ws.secondary = Some(secondary_ws.clone());
                ws.secondary_by_type.insert(shard_type, secondary_ws);
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
            NetEvent::SeatBindingsNotify(data) => {
                info!(bindings = data.bindings.len(), "received SeatBindingsNotify — activating seat bindings");
                *active_seat = input::ActiveSeatBindings::from_config(&data.bindings);
            }
            NetEvent::SubGridAssignmentUpdate(data) => {
                // Update assignments and mark affected chunks dirty for remesh.
                for (pos, sg_id) in &data.assignments {
                    if *sg_id == 0 {
                        sub_grid_state.assignments.remove(pos);
                    } else {
                        sub_grid_state.assignments.insert(*pos, *sg_id);
                    }
                    // Mark the chunk containing this block as dirty.
                    if let Some(source) = chunk_cache.primary_source {
                        let (chunk_key, _, _, _) = voxeldust_core::block::ShipGrid::world_to_chunk(pos.x, pos.y, pos.z);
                        chunk_cache.mark_dirty(source, chunk_key);
                    }
                }
                info!(assignments = data.assignments.len(), "received SubGridAssignmentUpdate");
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
            NetEvent::SecondaryChunkSnapshot { seed, data: cs } => {
                if let Some(&source) = chunk_cache.secondary_sources.get(&seed) {
                    let chunk_pos = glam::IVec3::new(cs.chunk_x, cs.chunk_y, cs.chunk_z);
                    match chunk_cache.insert_snapshot(source, chunk_pos, cs.seq, &cs.data) {
                        Ok(()) => {
                            info!(
                                seed,
                                chunk = %chunk_pos,
                                "secondary chunk snapshot received"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(%e, seed, "failed to insert secondary chunk snapshot");
                        }
                    }
                }
            }
            NetEvent::SecondaryChunkDelta { seed, data: cd } => {
                if let Some(&source) = chunk_cache.secondary_sources.get(&seed) {
                    let chunk_pos = glam::IVec3::new(cd.chunk_x, cd.chunk_y, cd.chunk_z);
                    if !cd.mods.is_empty() {
                        let edits: Vec<_> = cd.mods.iter()
                            .map(|m| (m.bx, m.by, m.bz, voxeldust_core::block::BlockId::from_u16(m.block_type)))
                            .collect();
                        chunk_cache.apply_delta(source, chunk_pos, cd.seq, &edits);
                    }
                    if !cd.sub_block_mods.is_empty() {
                        chunk_cache.apply_sub_block_delta(source, chunk_pos, &cd.sub_block_mods);
                    }
                }
            }
            NetEvent::SecondarySubGridAssignment { seed, data } => {
                // Sub-grid assignments for secondary ships — store for rendering.
                let _ = (seed, data); // TODO: apply to secondary source sub-grid state
            }
            NetEvent::Transitioning { target_shard_type, spawn_pose } => {
                info!(
                    target_shard_type,
                    has_spawn_pose = spawn_pose.is_some(),
                    spawn_pos = ?spawn_pose.as_ref().map(|sp| (sp.position.x as i64, sp.position.y as i64, sp.position.z as i64)),
                    "transitioning to new shard..."
                );
                // Force per-frame render diagnostics for 500 ms so we
                // capture the transition window (the 40-frame sampled
                // diagnostic misses the first ~200 ms).
                {
                    use std::sync::atomic::Ordering;
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);
                    render::TRACE_RENDER_UNTIL_MS.store(now_ms + 500, Ordering::Relaxed);
                }
                // Defer the `current_shard_type` flip. Interior-ship render,
                // block raycast, and other gates keep seeing the OLD shard
                // type until the new primary's first WorldState with our
                // player arrives — at which point we flip to the destination
                // type. Prevents a ~100-300 ms window where `current_shard_type`
                // is SHIP but `ws.latest` / `smooth.*` still hold stale EVA
                // coordinates, which would paint the interior mesh off-frame.
                if target_shard_type != 255 {
                    conn.pending_shard_type = Some(target_shard_type);
                } else {
                    // Legacy `ShardRedirect` with no type hint: fall back to
                    // whichever secondary shard type most recently connected.
                    conn.pending_shard_type = conn.secondary_shard_type;
                }

                // Transition to new shard: release the OLD primary source so
                // its chunks + GPU buffers are freed. If the player needs to
                // see the shard we just left (e.g. the ship they exited,
                // rendered as exterior from system shard), the NEW primary
                // will pre-connect it back as a secondary and re-stream its
                // chunks into a fresh secondary source. The previous code
                // only nulled the pointer, which orphaned the source and
                // accumulated ~8 chunks × ~100 KB per transition — the root
                // cause of the post-transition frame-rate drop (120 → 70 →
                // 50 fps across two round trips).
                // Move the old primary source into `grace_sources` under its
                // original seed so the renderer's grace-fallback pass (which
                // iterates the stashed last-primary WorldState for the next
                // `LAST_PRIMARY_GRACE_SECS`) can still resolve the ship's
                // chunks by entity_id. Free happens lazily in
                // `drain_expired_grace_sources`, not here. Without this, an
                // exit ship→planet transition would blink the ship off the
                // moment the old primary WS is stashed — chunks gone,
                // grace WS entity orphaned.
                let grace_expiry = Instant::now()
                    + std::time::Duration::from_secs_f64(LAST_PRIMARY_GRACE_SECS);
                if let Some(old_primary) = chunk_cache.primary_source.take() {
                    if let Some(seed) = chunk_cache.primary_seed {
                        chunk_cache.grace_sources.insert(seed, old_primary);
                        chunk_cache.grace_expiries.insert(seed, grace_expiry);
                        info!(source = old_primary.0, seed,
                            "grace-pinned old primary chunk source on transition");
                    } else {
                        // No seed recorded (legacy / first connect). Free
                        // immediately — nothing in `grace_sources` can key
                        // to a None seed, so it would never render anyway.
                        chunk_cache.cache.remove_source(old_primary);
                        info!(source = old_primary.0,
                            "released old primary chunk source on transition (no seed)");
                    }
                    // Leaving a SHIP primary: capture the ship's system-space
                    // pose + source for the grace-window exterior render.
                    // The render path uses absolute system-space coords so
                    // the ship draws correctly regardless of what frame the
                    // new primary's camera is in (system, planet-local,
                    // etc). Only populate when the source is actually
                    // available — if primary_source was already None we
                    // have nothing to render anyway.
                    if conn.current_shard_type
                        == voxeldust_core::client_message::shard_type::SHIP
                    {
                        // Ship's system-space pose: ws.latest.origin is the
                        // ship's exterior.position (per ship-shard broadcast
                        // convention). Rotation comes from the own-ship
                        // entity we inject on ship shard, or falls back to
                        // the cached player.ship_rotation.
                        let (ship_sys_pos, ship_rot, ship_br) = {
                            let origin = ws
                                .latest
                                .as_ref()
                                .map(|w| w.origin)
                                .unwrap_or(smooth.ship_origin);
                            let own_entity = ws
                                .latest
                                .as_ref()
                                .and_then(|w| w.entities.iter().find(|e| e.is_own
                                    && matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship)));
                            let rot = own_entity.map(|e| e.rotation)
                                .unwrap_or(player.ship_rotation);
                            // Bounding radius authoritative from ship-shard
                            // (hull AABB half-diagonal). Ships without a
                            // populated entity fall back to a large value so
                            // we err on the side of rendering — a frustum
                            // miss on exit is the bug we're fixing.
                            let br = own_entity
                                .map(|e| e.bounding_radius as f32)
                                .unwrap_or(2048.0);
                            (origin, rot, br)
                        };
                        ws.departed_ship = Some(DepartedShipData {
                            source: old_primary,
                            world_position: ship_sys_pos,
                            rotation: ship_rot,
                            bounding_radius: ship_br,
                            expiry: grace_expiry,
                        });
                        info!(
                            source = old_primary.0,
                            pos = ?(ship_sys_pos.x, ship_sys_pos.y, ship_sys_pos.z),
                            rot = ?(ship_rot.x, ship_rot.y, ship_rot.z, ship_rot.w),
                            to_shard = target_shard_type,
                            "captured departed ship for grace-window render"
                        );
                    }
                }
                chunk_cache.primary_seed = None;

                // Clear warp state — warp travel is over.
                warp.galaxy_position = None;
                warp.galaxy_rotation = None;
                warp.target_star_index = None;

                // Pick the secondary WS that matches the redirect target.
                // Fall back to any available `ws.secondary` if no per-type
                // match (shouldn't happen in practice but keeps the code
                // robust for legacy/unknown target types).
                let target_secondary = if target_shard_type != 255 {
                    ws.secondary_by_type.remove(&target_shard_type)
                        .or_else(|| ws.secondary.clone())
                } else {
                    ws.secondary.clone()
                };
                if let Some(matched_sec) = target_secondary.as_ref() {
                    info!(target_shard_type, ws_origin = ?(matched_sec.origin.x as i64, matched_sec.origin.y as i64, matched_sec.origin.z as i64),
                        "Transitioning: picked matching secondary");
                }
                if target_secondary.is_some() {
                    // Seamless: promote secondary to primary.
                    info!("seamless transition — secondary data available");

                    // Convert camera yaw/pitch so the view direction is
                    // preserved across the transition. The frame of
                    // interpretation depends on where we're going:
                    //   * Ship (walking inside): yaw/pitch are ship-local —
                    //     rotated by ship_rotation on display.
                    //   * System (EVA): yaw/pitch are world Y-up.
                    //   * Planet (on surface): yaw/pitch are local tangent.
                    // When transitioning AWAY FROM a ship we first compute
                    // the world-space forward vector, then convert it into
                    // the target shard's coordinate conventions so the
                    // camera points in the exact same world direction
                    // before and after the transition.
                    // Compute world-space forward for the CURRENT camera so
                    // we can re-express it in the new shard's frame.
                    let sp = (cam.pitch as f32).sin() as f64;
                    let cp = (cam.pitch as f32).cos() as f64;
                    let sy = (cam.yaw as f32).sin() as f64;
                    let cy = (cam.yaw as f32).cos() as f64;
                    let basis_fwd = DVec3::new(cy * cp, sp, sy * cp);
                    let (cam_fwd_world, ship_rot_used, ship_rot_source) = match conn.current_shard_type {
                        // Ship frame: rotate ship-local forward into world
                        // via the ship's authoritative quaternion.
                        t if t == voxeldust_core::client_message::shard_type::SHIP => {
                            let own_entity_rot = ws.latest.as_ref()
                                .and_then(|latest| latest.entities.iter().find(|e| e.is_own
                                    && matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship)))
                                .map(|e| e.rotation);
                            let (ship_rot_now, src) = match own_entity_rot {
                                Some(r) => (r, "ws.entities.own_ship"),
                                None => (player.ship_rotation, "player.ship_rotation (FALLBACK)"),
                            };
                            ((ship_rot_now * basis_fwd).normalize(), ship_rot_now, src)
                        }
                        // EVA (SYSTEM): camera_yaw/pitch are body-local,
                        // world forward is `body_rotation * basis_fwd`.
                        // Before the body-frame refactor this arm fell
                        // through to world-Y-up, which caused the
                        // wrong-angle boarding bug when returning to a
                        // ship from EVA because the target shard's
                        // projection was reading an unrotated world
                        // vector.
                        t if t == voxeldust_core::client_message::shard_type::SYSTEM => {
                            let br = player.body_rotation;
                            ((br * basis_fwd).normalize(), br, "player.body_rotation (EVA)")
                        }
                        // Planet / unknown: world Y-up (scalar yaw/pitch
                        // are tangent-frame already for planet,
                        // world-Y-up for unknown).
                        _ => (basis_fwd.normalize(), DQuat::IDENTITY, "world Y-up"),
                    };
                    info!(
                        source = ship_rot_source,
                        ship_rot = ?(ship_rot_used.x, ship_rot_used.y, ship_rot_used.z, ship_rot_used.w),
                        cam_yaw = cam.yaw,
                        cam_pitch = cam.pitch,
                        basis_fwd = ?(basis_fwd.x, basis_fwd.y, basis_fwd.z),
                        cam_fwd_world = ?(cam_fwd_world.x, cam_fwd_world.y, cam_fwd_world.z),
                        from_shard = conn.current_shard_type,
                        to_shard = target_shard_type,
                        "transition rotation math"
                    );

                    // Re-express cam_fwd_world in the new shard's frame.
                    if target_shard_type == voxeldust_core::client_message::shard_type::SHIP {
                        // System/Planet → Ship (boarding): convert world
                        // forward into ship-local frame using the
                        // promoted ship's rotation. After boarding the
                        // camera reads yaw/pitch as ship-local and
                        // `camera.rs` multiplies by `ship_rotation` to
                        // display — we need `ship_rotation * local_fwd ==
                        // cam_fwd_world`.
                        let ship_rot = ws.secondary.as_ref()
                            .and_then(|secondary| secondary.entities.iter()
                                .find(|e| e.is_own
                                    && matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship)))
                            .map(|e| e.rotation)
                            .unwrap_or(player.ship_rotation);
                        let local_fwd = ship_rot.inverse() * cam_fwd_world;
                        cam.pitch = local_fwd.y.asin();
                        cam.yaw = local_fwd.z.atan2(local_fwd.x);
                        // Also update the cached ship_rotation used by the
                        // camera so the first render frame uses the correct
                        // orientation.
                        player.ship_rotation = ship_rot;
                    } else if target_shard_type == voxeldust_core::client_message::shard_type::PLANET {
                        // Ship/System → Planet: project world forward onto the
                        // player's planet tangent frame.
                        //
                        // The tangent frame is defined by the player's
                        // planet-local position (the radial up). Previously
                        // this was read from `ws.secondary.players.first()`
                        // which is wrong: during the handoff the exiting
                        // player isn't in the planet's player list yet, so
                        // `.first()` returned *another* player's position
                        // (or the `DVec3::Y` fallback), giving a tangent
                        // frame unrelated to where the player actually lands
                        // → camera yaw/pitch that drift 10s-to-180° off.
                        //
                        // Derive the actual landing position from state the
                        // client already has: the player's current world
                        // position (ship-local rotated into system-space,
                        // plus ship origin) minus the planet's system-space
                        // position (`ws.secondary.origin`, which the planet
                        // shard publishes as its WorldState origin).
                        let player_system_pos = if conn.current_shard_type
                            == voxeldust_core::client_message::shard_type::SHIP
                        {
                            let ship_rot = ws.latest.as_ref()
                                .and_then(|latest| latest.entities.iter().find(|e| e.is_own
                                    && matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship)))
                                .map(|e| e.rotation)
                                .unwrap_or(player.ship_rotation);
                            smooth.ship_origin + ship_rot * smooth.render_position
                        } else {
                            // EVA → Planet: smooth.render_position is already
                            // system-space.
                            smooth.render_position
                        };
                        let planet_origin = ws.secondary.as_ref()
                            .map(|secondary| secondary.origin)
                            .unwrap_or(DVec3::ZERO);
                        let planet_pos = player_system_pos - planet_origin;
                        let up = if planet_pos.length_squared() > 1e-10 {
                            planet_pos.normalize()
                        } else {
                            // Degenerate (player at planet center). Fall back
                            // to world-Y so the math is defined; this case
                            // shouldn't happen in practice (handoff only
                            // fires in atmosphere or surface).
                            DVec3::Y
                        };
                        let pole = DVec3::Y;
                        let east_raw = pole.cross(up);
                        let east = if east_raw.length_squared() > 1e-10 {
                            east_raw.normalize()
                        } else {
                            DVec3::Z.cross(up).normalize()
                        };
                        let north = up.cross(east).normalize();
                        let fwd_north = cam_fwd_world.dot(north);
                        let fwd_up = cam_fwd_world.dot(up);
                        let fwd_east = cam_fwd_world.dot(east);
                        cam.pitch = fwd_up.asin();
                        cam.yaw = fwd_east.atan2(fwd_north);
                        info!(
                            player_sys_pos = ?(player_system_pos.x, player_system_pos.y, player_system_pos.z),
                            planet_origin = ?(planet_origin.x, planet_origin.y, planet_origin.z),
                            planet_local_pos = ?(planet_pos.x, planet_pos.y, planet_pos.z),
                            up = ?(up.x, up.y, up.z),
                            east = ?(east.x, east.y, east.z),
                            north = ?(north.x, north.y, north.z),
                            fwd_north, fwd_up, fwd_east,
                            new_yaw = cam.yaw,
                            new_pitch = cam.pitch,
                            "ship→planet tangent projection"
                        );
                    } else {
                        // → System (EVA in vacuum) or fallback. No
                        // gravity "up" to align to — inherit the ship's
                        // body orientation so the view doesn't snap at
                        // the hatch. `cam.yaw/pitch` stay BODY-LOCAL
                        // (same values as inside the ship), and
                        // `player.body_rotation` now carries the ship's
                        // world orientation. camera.rs's SYSTEM branch
                        // renders `body_rotation * head_fwd`, which
                        // equals the pre-exit `ship_rotation * head_fwd`
                        // by construction.
                        if target_shard_type == voxeldust_core::client_message::shard_type::SYSTEM {
                            player.body_rotation = ship_rot_used;
                        } else {
                            // Unknown/default — project to world-Y-up as
                            // before; no body-frame to inherit.
                            cam.pitch = cam_fwd_world.y.asin();
                            cam.yaw = cam_fwd_world.z.atan2(cam_fwd_world.x);
                            player.body_rotation = DQuat::IDENTITY;
                        }
                    }
                    cam.pitch = cam.pitch.clamp(
                        -std::f64::consts::FRAC_PI_2 + 0.01,
                        std::f64::consts::FRAC_PI_2 - 0.01,
                    );

                    // Capture the SOURCE shard type before the flip at line
                    // below overwrites `conn.current_shard_type`. Used to
                    // decide whether `smooth.render_position` is in
                    // ship-local (needs rotation) or system-space (already
                    // world-aligned) coordinates.
                    let source_shard_type = conn.current_shard_type;

                    // Flip `current_shard_type` immediately for seamless
                    // transitions: the camera + rendering code branches on
                    // it, and having it stale during the promotion window
                    // causes visual chaos (e.g. SHIP-frame data rendered
                    // through the SYSTEM-frame camera path, where cam_pos
                    // is interpreted without adding the ship's origin).
                    // The `pending_shard_type` safety valve kept for the
                    // hard-transition path below.
                    if target_shard_type != 255 {
                        conn.current_shard_type = target_shard_type;
                        conn.pending_shard_type = None;
                    }

                    // Save ship-local state before overwriting — needed as
                    // fallback if the secondary WS doesn't include the player.
                    // With `smooth.ship_origin` now tracking `ws.latest.origin`
                    // directly (no interp lag on SHIP shard — see
                    // `smooth_render_position`), this is the authoritative
                    // anchor camera.rs was using to render `cam_system_pos`
                    // on the last frame, so `start_world` matches the pixel
                    // position the user last saw.
                    let prev_ship_origin = smooth.ship_origin;
                    let prev_render_pos = smooth.render_position;
                    // On SHIP source shard, `prev_render_pos` is the player's
                    // SHIP-LOCAL position (un-rotated). The actual rendered
                    // camera applies the ship's rotation: `cam_sys = ws.origin
                    // + ship_rot * player_local`. If we compute `start_world`
                    // without that rotation, we're 10+ km off from the camera
                    // that was actually on screen last frame.
                    //
                    // For SYSTEM/PLANET source shards `prev_render_pos` is
                    // already the player's system-space (or planet-local)
                    // world position, so no rotation is applied — multiplying
                    // a gigameter-scale vector by a non-identity quaternion
                    // would teleport the camera across the star system.
                    let prev_render_pos_world = if source_shard_type
                        == voxeldust_core::client_message::shard_type::SHIP
                    {
                        smooth.render_rotation * prev_render_pos
                    } else {
                        prev_render_pos
                    };
                    // Capture pre-switch camera world position for the 150 ms
                    // handoff blend. smooth_render_position will lerp the
                    // rendered camera from this world-space anchor toward the
                    // new primary's authoritative position so the view doesn't
                    // snap on primary switch.
                    //
                    // When the redirect carries a server-authoritative
                    // `spawn_pose`, the blend's TARGET is the authoritative
                    // spawn position. Starting from the last-rendered camera
                    // position (`prev_ship_origin + prev_render_pos_world`)
                    // and lerping toward the server's spawn over 150 ms
                    // masks the ~1-tick ship-pose drift between client and
                    // server without any client-side prediction.
                    // When the redirect carries a server-authoritative
                    // `spawn_pose` (computed by the target shard using its
                    // CURRENT pose), blending from the source's stale
                    // position to the target's authoritative position
                    // introduces a visible ~5 km "catch-up" sweep over
                    // 150 ms — the gap is the inter-shard latency drift
                    // (source shard sees itself 50-100 ms stale because it
                    // receives position updates from the target at 20 Hz).
                    // The gap is inherent at the protocol level; blending
                    // over it just makes the lag into visible motion.
                    //
                    // Snap the camera directly to the authoritative target
                    // instead. The snapped path already wrote the target's
                    // position into `smooth.render_position`, so not
                    // activating the blend means the first post-transition
                    // frame renders exactly there. The one-frame position
                    // discontinuity (~5 km) is smaller and less obvious
                    // than a 150 ms lerp that visibly traverses 18+ km.
                    if spawn_pose.is_some() {
                        smooth.handoff_start_world = None;
                        smooth.handoff_velocity = DVec3::ZERO;
                    } else {
                        smooth.handoff_start_world = Some(prev_ship_origin + prev_render_pos_world);
                        smooth.handoff_started_at = Instant::now();
                        // Capture the ship's orbital velocity for the legacy
                        // (non-authoritative) blend path. The target moves at
                        // ship velocity each tick; advancing start_world by
                        // the same rate keeps the lerp delta bounded.
                        smooth.handoff_velocity = ws.latest.as_ref()
                            .and_then(|w| w.entities.iter().find(|e| e.is_own
                                && matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship)))
                            .map(|e| e.velocity)
                            .unwrap_or(DVec3::ZERO);
                    }

                    // Stash the old primary WS for a grace window so the renderer
                    // can keep drawing its LOD-proxy ship entity during the
                    // promotion overlap. Without this, the moment we swap
                    // `ws.latest` to the promoted secondary (whose own ship is
                    // marked `is_own=true` and skipped by draw_entities), the
                    // orange LOD sphere for the ship vanishes from screen.
                    if let Some(prev) = ws.latest.take() {
                        ws.last_primary = Some(prev);
                        ws.last_primary_until =
                            Some(Instant::now() + std::time::Duration::from_secs_f64(LAST_PRIMARY_GRACE_SECS));
                    }
                    // Use the target-matched secondary (SYSTEM for
                    // Ship→System exit, not PLANET even if the PLANET
                    // secondary was the most recently received).
                    ws.latest = target_secondary;
                    // Clear the single-slot alias too so stale data
                    // from other shards doesn't linger.
                    ws.secondary = None;
                    ws.secondary_by_type.clear();
                    // Clear `secondary_shard_type` to avoid stale last-writer
                    // values leaking into later decisions. The authoritative
                    // destination type is `pending_shard_type`, set above.
                    conn.secondary_shard_type = None;
                    player.is_piloting = false;

                    // Eagerly promote the ship secondary's chunk source to
                    // primary when boarding (target=SHIP). Without this the
                    // interior render branch looks up the source via
                    // `shard_seed`/own-ship-entity during the 18-40 ms
                    // window between `Transitioning` and `Connected`, but:
                    //   * `shard_seed` still holds the PREVIOUS primary's
                    //     seed (updated only in `Connected`), so the
                    //     HashMap lookup misses.
                    //   * `ws.latest` may not yet contain the own-ship
                    //     Ship entity (depends on arrival timing of the
                    //     first ship WS with the injection), so the
                    //     fallback via entities doesn't resolve either.
                    // Consequence: the interior mesh renders nothing for
                    // that window = boarding blink. Promote here and also
                    // update `shard_seed` so every lookup during the gap
                    // succeeds. `Connected`'s promotion code is idempotent
                    // (it removes from `secondary_sources` if present) so
                    // re-running it does no harm.
                    if target_shard_type
                        == voxeldust_core::client_message::shard_type::SHIP
                    {
                        let ship_seed = ws.latest.as_ref()
                            .and_then(|w| w.entities.iter().find(|e| e.is_own
                                && matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship))
                                .map(|e| e.entity_id));
                        if let Some(seed) = ship_seed {
                            // Check secondary_sources first (normal
                            // observer case) then grace_sources (case
                            // where SecondaryDisconnected just
                            // grace-pinned the source a moment before
                            // this Transitioning was processed — common
                            // for ShardRedirect replacing the current
                            // observer with the new primary).
                            let sec_src = chunk_cache.secondary_sources.remove(&seed)
                                .or_else(|| chunk_cache.grace_sources.get(&seed).copied());
                            if let Some(sec_src) = sec_src {
                                // Free any previous primary first (we
                                // already grace-pinned the old primary
                                // source above, so this only fires for
                                // rare paths where primary_source was
                                // non-None here).
                                if let Some(old) = chunk_cache.primary_source.take() {
                                    chunk_cache.cache.remove_source(old);
                                }
                                chunk_cache.primary_source = Some(sec_src);
                                // Also grace-pin under the same key so
                                // the external-ship entity path can find
                                // this source while `is_own` filtering
                                // skips the interior duplicate. Extends
                                // the expiry (overwrite ok).
                                chunk_cache.grace_sources.insert(seed, sec_src);
                                chunk_cache.grace_expiries.insert(seed, grace_expiry);
                                chunk_cache.primary_seed = Some(seed);
                                conn.shard_seed = seed;
                                info!(seed, source = sec_src.0,
                                    "eagerly promoted ship secondary to primary at Transitioning");
                            }
                        }
                    }

                    // Snap interpolation state from promoted WorldState so
                    // smooth_render_position has data on the very next frame.
                    // Without this, snapshots are empty for ~100ms causing a
                    // blink where stale ship-local coords are interpreted as
                    // planet-local (camera inside the planet core).
                    let mut snapped = false;
                    if let Some(ref promoted) = ws.latest {
                        // Filter by own_player_id, NOT `first()`. System
                        // shard EVA broadcasts include every EVA player
                        // in the shard, so `first()` often returns the
                        // wrong one and snaps the camera to someone
                        // else's position — the "star fly-through"
                        // symptom seen in video captures. If our own
                        // player isn't present yet (handoff still in
                        // flight on the server), we fall through to the
                        // `snapped=false` path below, which sets up the
                        // re-anchor to ship that `smooth_render_position`
                        // uses during the blink window.
                        if let Some(p) = promoted.players.iter()
                            .find(|p| p.player_id == conn.own_player_id)
                        {
                            let snap = TimedSnapshot {
                                tick: promoted.tick,
                                player_position: p.position,
                                ship_rotation: p.rotation,
                                ship_origin: promoted.origin,
                                velocity: p.velocity,
                                bodies: promoted.bodies.clone(),
                                sub_grids: promoted.sub_grids.iter()
                                    .map(|sg| (sg.sub_grid_id, sg.clone())).collect(),
                            };
                            snapshots.snapshots.clear();
                            snapshots.snapshots.push_back(snap);
                            snapshots.highest_tick = promoted.tick;
                            snapshots.estimated_server_tick = promoted.tick as f64;
                            snapshots.anchor_time = Some(std::time::Instant::now());
                            snapshots.anchor_server_tick = promoted.tick as f64;
                            snapshots.synced = true;
                            smooth.render_position = p.position;
                            smooth.render_rotation = p.rotation;
                            smooth.ship_origin = promoted.origin;
                            smooth.bodies = promoted.bodies.clone();
                            snapped = true;
                        }
                    }
                    if !snapped {
                        // Secondary WS has no player yet (handoff entity not
                        // created when the last secondary broadcast went out).
                        // Compute ship-local from ship's last known state.
                        // Carry the ship's rotation forward from the promoted
                        // secondary's own ship entity if present — otherwise
                        // keep whatever the camera was already using, so the
                        // hull mesh doesn't snap to IDENTITY orientation for
                        // the first render frame after boarding.
                        if let Some(ref promoted) = ws.latest {
                            // Use the ship's LIVE system-space origin (ws.origin
                            // from the just-stashed ship WS) instead of the
                            // interpolated `smooth.ship_origin`, which lags
                            // ~3 ticks behind at 20 Hz (~15 km at 110 km/s
                            // orbital velocity). That lag is invisible on
                            // SHIP shard because interior rendering is
                            // camera-relative, but at Transitioning it
                            // baked 15 km of staleness into `render_position`
                            // — putting the new primary's ship AOI entity
                            // (at the live position) 15 km outside the
                            // camera frustum. The live origin is what the
                            // new primary sees the ship at right now, so
                            // starting the fallback from there keeps the
                            // camera near the ship even before velocity
                            // extrapolation kicks in.
                            let live_ship_origin = ws.last_primary.as_ref()
                                .map(|last| last.origin)
                                .unwrap_or(prev_ship_origin);
                            // Server-authoritative spawn position (preferred):
                            // the source shard computed this using the exact
                            // same formula the target shard will use to spawn
                            // the player. Using it directly eliminates all
                            // client-side fallback math — no interp-lag
                            // compensation, no rotation reconstruction, no
                            // guessing. When absent (gateway routing, legacy
                            // paths), we fall back to the client-side
                            // reconstruction using the last ship pose.
                            let (system_pos, authoritative_rot) = match spawn_pose.as_ref() {
                                Some(sp) => (sp.position, Some(sp.rotation)),
                                None => (live_ship_origin + prev_render_pos_world, None),
                            };
                            let planet_local = system_pos - promoted.origin;
                            info!(
                                has_spawn_pose = spawn_pose.is_some(),
                                spawn_pos = ?spawn_pose.as_ref().map(|sp| (sp.position.x as i64, sp.position.y as i64, sp.position.z as i64)),
                                spawn_vel = ?spawn_pose.as_ref().map(|sp| (sp.velocity.x as i64, sp.velocity.y as i64, sp.velocity.z as i64)),
                                live_ship_origin = ?(live_ship_origin.x as i64, live_ship_origin.y as i64, live_ship_origin.z as i64),
                                fallback_system_pos = ?((live_ship_origin + prev_render_pos_world).x as i64,
                                    (live_ship_origin + prev_render_pos_world).y as i64,
                                    (live_ship_origin + prev_render_pos_world).z as i64),
                                actual_system_pos = ?(system_pos.x as i64, system_pos.y as i64, system_pos.z as i64),
                                promoted_origin = ?(promoted.origin.x as i64, promoted.origin.y as i64, promoted.origin.z as i64),
                                "Transitioning fallback: choosing spawn position"
                            );
                            // Prefer the server-authoritative rotation from
                            // the redirect's spawn_pose. Fall back to the
                            // target shard's own-ship entity rotation, and
                            // finally to the source's last rendered rotation.
                            let carried_rot = authoritative_rot.unwrap_or_else(|| {
                                promoted
                                    .entities
                                    .iter()
                                    .find(|e| e.is_own
                                        && matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship))
                                    .map(|e| e.rotation)
                                    .unwrap_or(smooth.render_rotation)
                            });
                            // Capture the ship's system-space velocity at
                            // transition time so the extrapolation below
                            // advances `render_position` with the ship's
                            // orbit until the new primary's first WS with
                            // our player arrives (~50-200 ms). Without
                            // this, `render_position` stays frozen while
                            // the ship entity in the new primary's AOI
                            // moves ~10 km in orbital velocity, putting
                            // the ship outside the frustum — the blink.
                            // Prefer server-authoritative velocity from the
                            // redirect's spawn_pose (inherited ship velocity
                            // for EVA exit); fall back to last-seen ship
                            // velocity from the source WS.
                            let carried_velocity = spawn_pose.as_ref()
                                .map(|sp| sp.velocity)
                                .unwrap_or_else(|| {
                                    ws.last_primary.as_ref()
                                        .and_then(|last| last.entities.iter().find(|e| e.is_own
                                            && matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship)))
                                        .map(|e| e.velocity)
                                        .unwrap_or(DVec3::ZERO)
                                });
                            let snap = TimedSnapshot {
                                tick: promoted.tick,
                                player_position: planet_local,
                                ship_rotation: carried_rot,
                                ship_origin: promoted.origin,
                                velocity: carried_velocity,
                                bodies: promoted.bodies.clone(),
                                sub_grids: std::collections::HashMap::new(),
                            };
                            // Record the live ship anchor separately so
                            // `smooth_render_position` can re-target the
                            // camera to the ship's CURRENT AOI position
                            // each frame. Keeping this out of the
                            // TimedSnapshot (and off `smooth.ship_origin`)
                            // preserves the `handoff_blend` invariant
                            // `smooth.ship_origin + render_position ==
                            // world_pos`, which broke catastrophically
                            // when ship_origin was set to a non-zero
                            // value (the camera was briefly rendered at
                            // the star then lerped out to the ship).
                            smooth.exit_ship_anchor = Some(live_ship_origin);
                            info!(
                                live_ship_origin = ?(live_ship_origin.x as i64, live_ship_origin.y as i64, live_ship_origin.z as i64),
                                planet_local = ?(planet_local.x as i64, planet_local.y as i64, planet_local.z as i64),
                                prev_render_pos = ?(prev_render_pos.x as i64, prev_render_pos.y as i64, prev_render_pos.z as i64),
                                prev_ship_origin = ?(prev_ship_origin.x as i64, prev_ship_origin.y as i64, prev_ship_origin.z as i64),
                                promoted_origin = ?(promoted.origin.x as i64, promoted.origin.y as i64, promoted.origin.z as i64),
                                ws_last_primary_origin = ?ws.last_primary.as_ref().map(|l| (l.origin.x as i64, l.origin.y as i64, l.origin.z as i64)),
                                "Transitioning fallback: exit_ship_anchor set"
                            );
                            snapshots.snapshots.clear();
                            snapshots.snapshots.push_back(snap);
                            snapshots.highest_tick = promoted.tick;
                            snapshots.estimated_server_tick = promoted.tick as f64;
                            snapshots.anchor_time = Some(std::time::Instant::now());
                            snapshots.anchor_server_tick = promoted.tick as f64;
                            snapshots.synced = true;
                            smooth.render_position = planet_local;
                            smooth.render_rotation = carried_rot;
                            smooth.ship_origin = promoted.origin;
                            smooth.bodies = promoted.bodies.clone();
                        } else {
                            snapshots.snapshots.clear();
                            snapshots.synced = false;
                            snapshots.anchor_time = None;
                        }
                    }
                } else {
                    // Hard transition: clear and reconnect.
                    // Stash the last-known primary WorldState so the renderer
                    // keeps showing entities (ships, players, bodies) for up to
                    // LAST_PRIMARY_GRACE_SECS instead of going blank for the
                    // full reconnect latency. Cleared when the new primary's
                    // first WorldState arrives (see NetEvent::WorldState handler).
                    info!("hard transition — no secondary data");
                    snapshots.snapshots.clear();
                    snapshots.synced = false;
                    snapshots.anchor_time = None;
                    if let Some(prev) = ws.latest.take() {
                        ws.last_primary = Some(prev);
                        ws.last_primary_until =
                            Some(Instant::now() + std::time::Duration::from_secs_f64(LAST_PRIMARY_GRACE_SECS));
                    }
                }
                ws.secondary = None;
                conn.secondary_shard_type = None;
            }
        }
    }
}

/// Tick-based snapshot interpolation (Valve Source / Overwatch style).
///
/// Renders at a fixed delay behind the estimated server tick (`INTERP_DELAY_TICKS`).
/// Uses server tick numbers for timing (immune to client-side message batching).
/// The ring buffer of `TimedSnapshot`s is searched for the two snapshots bracketing
/// the render tick, then all state is lerped between them.
///
/// Galaxy warp uses its own velocity-based smoothing (separate from WorldState snapshots).
/// Planet shards snap to the latest snapshot (no interpolation) — body positions update
/// discretely from orbital mechanics, so interpolating the camera would desync it.
fn smooth_render_position(
    mut smooth: ResMut<RenderSmoothing>,
    mut snapshots: ResMut<SnapshotBuffer>,
    warp: Res<ClientWarp>,
    conn: Res<ConnectionInfo>,
    ft: Res<FrameTime>,
    ws: Res<ServerWorldState>,
) {
    // Galaxy warp: separate smoothing path (GalaxyWorldState, not WorldState snapshots).
    if warp.galaxy_position.is_some() {
        let blend = 1.0 - (-20.0 * ft.dt).exp();
        let elapsed = (ft.last_time - smooth.last_galaxy_update_time).as_secs_f64().min(0.06);
        let target = smooth.prev_galaxy_position + smooth.galaxy_velocity * elapsed;
        let delta = (target - smooth.galaxy_render_position) * blend;
        smooth.galaxy_render_position = smooth.galaxy_render_position + delta;
        let rot = smooth.galaxy_render_rotation.slerp(smooth.prev_galaxy_rotation, blend);
        smooth.galaxy_render_rotation = rot;
        smooth.bodies.clear();
        return;
    }

    if !snapshots.synced || snapshots.snapshots.is_empty() {
        return;
    }

    // Tick estimate is anchored to wall-clock, not accumulated frame
    // dt. This makes per-frame position step a constant function of
    // real elapsed time → per-frame step at 120 fps is exactly half
    // of the step at 60 fps, and a frame-rate oscillation no longer
    // produces visible per-frame jumps. Previously
    // `estimated_server_tick += ft.dt / TICK_DURATION_SECS` pushed
    // every frame's dt jitter straight into the interpolation — a
    // 25 ms frame followed by a 8 ms frame looked like a teleport
    // even when the server data was perfectly smooth.
    //
    // The clock-drift correction in the WorldState handler still
    // pulls the anchor toward the server tick gently (0.1× blend).
    let _ = ft; // kept in the signature for compat; no longer consumed.
    if let (Some(anchor_time), anchor_tick) =
        (snapshots.anchor_time, snapshots.anchor_server_tick)
    {
        let elapsed = anchor_time.elapsed().as_secs_f64();
        snapshots.estimated_server_tick = anchor_tick + elapsed / TICK_DURATION_SECS;
    }

    // Render at a fixed delay behind the estimated server tick.
    let render_tick = snapshots.estimated_server_tick - INTERP_DELAY_TICKS;

    // Find bracketing snapshots via binary search on tick.
    let idx = snapshots.snapshots.partition_point(|s| (s.tick as f64) <= render_tick);

    let (snap_a, snap_b) = if idx == 0 {
        // render_tick is before all snapshots — hold at oldest.
        let a = &snapshots.snapshots[0];
        (a, a)
    } else if idx >= snapshots.snapshots.len() {
        // render_tick is past all snapshots — hold at newest (no extrapolation).
        let a = snapshots.snapshots.back().unwrap();
        (a, a)
    } else {
        (&snapshots.snapshots[idx - 1], &snapshots.snapshots[idx])
    };

    let t = if snap_a.tick == snap_b.tick {
        1.0
    } else {
        ((render_tick - snap_a.tick as f64) / (snap_b.tick - snap_a.tick) as f64).clamp(0.0, 1.0)
    };

    // Shards where the client is positioned in a FAST-MOVING reference
    // frame (system-space positions change by ~5000 m/tick at orbital
    // velocity; planet-shard origin rotates / orbits) snap directly to
    // the latest snapshot instead of interpolating. Reason: ship
    // entities in the WorldState are rendered from `ws.entities[].position`
    // at the *latest* server tick — if the camera is interp-delayed by
    // 3 ticks (150 ms), the other ship appears thousands of meters
    // ahead of the camera even though on the server they're co-located.
    // Snapping keeps the camera aligned with peer-ship positions. The
    // ship-interior shard (stable local frame, sub-meter motion) is
    // the only place standard snapshot interpolation is worthwhile,
    // and it benefits from the extra smoothness.
    let snap_to_latest = matches!(
        conn.current_shard_type,
        voxeldust_core::client_message::shard_type::PLANET
            | voxeldust_core::client_message::shard_type::SYSTEM
            | voxeldust_core::client_message::shard_type::GALAXY
    );
    if snap_to_latest {
        if let Some(latest) = snapshots.snapshots.back() {
            // Post-transition anchor: during the 50-200 ms window
            // between Transitioning and the first new-primary WS that
            // contains our (spawned) player, the snapshot buffer holds
            // only the synthetic snap inserted at Transitioning.
            // Velocity extrapolation was tried and rejected — it
            // unboundedly advances past the ship's AOI entity, which
            // is frozen in the WS between server ticks. The correct
            // behavior is to re-anchor `render_position` each frame
            // to the ship entity's CURRENT position in the new
            // primary's WS, offset by the player's ship-local
            // position at exit time. That keeps the client camera
            // within frustum range of the ship regardless of how
            // many WS ticks have advanced on the server.
            //
            // `latest.player_position - latest.ship_origin_at_exit`
            // encodes the world-space player offset captured at the
            // Transitioning snap fallback. Here `latest.ship_origin`
            // stores that anchor (live ship origin at stash time, per
            // the Transitioning handler). Subtracting gives the
            // offset from ship, which we apply to whatever position
            // the new primary's ship entity is at RIGHT NOW.
            // Real player WS arrived — snapshot buffer has >1 entries
            // now, clear the post-transition anchor so normal
            // snap_to_latest takes over.
            if snapshots.snapshots.len() > 1 {
                smooth.exit_ship_anchor = None;
            }
            smooth.render_position = latest.player_position;
            smooth.render_rotation = latest.ship_rotation;
            smooth.ship_origin = latest.ship_origin;
            smooth.bodies = latest.bodies.clone();
        }
        smooth.interpolation_t = 1.0;
    } else {
        // Ship/system shard: interpolate between bracketing snapshots.
        smooth.interpolation_t = t;
        let prev_render = smooth.render_position;
        smooth.render_position = snap_a.player_position.lerp(snap_b.player_position, t);
        smooth.render_rotation = snap_a.ship_rotation.slerp(snap_b.ship_rotation, t);
        // DO NOT interpolate `ship_origin`: at orbital velocity (~110 km/s)
        // interp lag of ~3 ticks puts `smooth.ship_origin` 15+ km behind the
        // authoritative `ws.latest.origin`. On SHIP shard the interior scene
        // is camera-relative so that lag is invisible — but it becomes a
        // 15 km CAMERA JUMP at the moment of Ship→System transition, because
        // the new shard's ship entity is at the authoritative position while
        // `handoff_start_world` and the last-rendered `cam_system_pos` were
        // anchored at the lagged interp origin. Pin `ship_origin` to the
        // live WS origin so interior rendering and system-space anchoring
        // agree. The 20 Hz step change is sub-arc-second for celestial
        // bodies and cancels out in nearby-entity delta math (all ship
        // shard entities are ship-local so `entity_world - cam_system_pos`
        // is independent of `ship_origin`).
        smooth.ship_origin = ws.latest.as_ref()
            .map(|w| w.origin)
            .unwrap_or_else(|| snap_b.ship_origin);

        // 1 Hz diagnostic: verify interpolation is actually producing
        // smooth per-frame deltas. `snap_delta_mag` (distance between
        // the two bracketing server snapshots) and `frame_delta_mag`
        // (distance render_position moved this frame) tell us whether
        // the interpolation is advancing smoothly.
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static LAST_LOG: AtomicU64 = AtomicU64::new(0);
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let last = LAST_LOG.load(Ordering::Relaxed);
            if now_ms >= last + 1000 {
                LAST_LOG.store(now_ms, Ordering::Relaxed);
                let snap_delta_mag = (snap_b.player_position - snap_a.player_position).length();
                let frame_delta_mag = (smooth.render_position - prev_render).length();
                tracing::info!(
                    snap_a_tick = snap_a.tick,
                    snap_b_tick = snap_b.tick,
                    t,
                    snap_delta_mag = snap_delta_mag as f32,
                    frame_delta_mag = frame_delta_mag as f32,
                    buffer_len = snapshots.snapshots.len(),
                    est_server_tick = snapshots.estimated_server_tick as f32,
                    "interp diag (1Hz)"
                );
            }
        }

        // Interpolate celestial body positions at same t.
        smooth.bodies.clear();
        for (i, curr_body) in snap_b.bodies.iter().enumerate() {
            let pos = snap_a.bodies.get(i)
                .map(|prev_body| prev_body.position.lerp(curr_body.position, t))
                .unwrap_or(curr_body.position);
            smooth.bodies.push(CelestialBodyData {
                body_id: curr_body.body_id,
                position: pos,
                radius: curr_body.radius,
                color: curr_body.color,
            });
        }

        // Interpolate sub-grid transforms at same t.
        smooth.sub_grid_transforms.clear();
        for (&sg_id, sg_b) in &snap_b.sub_grids {
            if let Some(sg_a) = snap_a.sub_grids.get(&sg_id) {
                smooth.sub_grid_transforms.insert(sg_id, voxeldust_core::client_message::SubGridTransformData {
                    sub_grid_id: sg_id,
                    translation: sg_a.translation.lerp(sg_b.translation, t as f32),
                    rotation: sg_a.rotation.slerp(sg_b.rotation, t as f32),
                    parent_grid: sg_b.parent_grid,
                    anchor: sg_b.anchor,
                    mount_pos: sg_b.mount_pos,
                    mount_face: sg_b.mount_face,
                    joint_type: sg_b.joint_type,
                    current_value: sg_a.current_value + (sg_b.current_value - sg_a.current_value) * t as f32,
                });
            } else {
                smooth.sub_grid_transforms.insert(sg_id, sg_b.clone());
            }
        }
    }

    // Handoff blend: if a primary switch just occurred, lerp the rendered
    // camera's world-space position from the pre-switch anchor to the new
    // primary's authoritative position over HANDOFF_BLEND_SECS. Interpolation
    // only — we never extrapolate past either endpoint.
    if let Some(start_world) = smooth.handoff_start_world {
        let elapsed = smooth.handoff_started_at.elapsed().as_secs_f64();
        if elapsed >= HANDOFF_BLEND_SECS {
            smooth.handoff_start_world = None;
            smooth.handoff_velocity = DVec3::ZERO;
            tracing::info!("handoff_blend completed");
        } else {
            let t = (elapsed / HANDOFF_BLEND_SECS).clamp(0.0, 1.0);
            let s = t * t * (3.0 - 2.0 * t);
            // Advance start_world by the inherited ship velocity so the
            // reference point moves WITH the ship. Target (authoritative
            // EVA position in the new shard) also advances by the same
            // velocity each tick. With both moving at the same rate,
            // `target - start` stays at the constant inter-shard latency
            // drift (~5 km) instead of growing to ~22 km over 150 ms.
            let start_extrapolated = start_world + smooth.handoff_velocity * elapsed;
            let target_world = smooth.ship_origin + smooth.render_position;
            let pre_rp = smooth.render_position;
            let blended_world = start_extrapolated.lerp(target_world, s);
            smooth.render_position = blended_world - smooth.ship_origin;
            tracing::info!(
                t, s,
                start_world = ?(start_world.x as i64, start_world.y as i64, start_world.z as i64),
                target_world = ?(target_world.x as i64, target_world.y as i64, target_world.z as i64),
                ship_origin = ?(smooth.ship_origin.x as i64, smooth.ship_origin.y as i64, smooth.ship_origin.z as i64),
                pre_render_pos = ?(pre_rp.x as i64, pre_rp.y as i64, pre_rp.z as i64),
                blended = ?(blended_world.x as i64, blended_world.y as i64, blended_world.z as i64),
                post_render_pos = ?(smooth.render_position.x as i64, smooth.render_position.y as i64, smooth.render_position.z as i64),
                "handoff_blend step"
            );
        }
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
    voxel_volume: Option<voxel_volume::VoxelVolume>,
    cloud_system: Option<cloud_system::CloudSystem>,
    uniform_data: Vec<ObjectUniforms>,
    /// Reused across frames to avoid per-frame HashMap allocation.
    chunk_uniform_map: std::collections::HashMap<voxeldust_core::block::client_chunks::ChunkKey, usize>,
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
        let gfx_preset = match args.graphics.to_lowercase().as_str() {
            "low" => graphics_settings::GraphicsSettings::low(),
            "medium" => graphics_settings::GraphicsSettings::medium(),
            "high" => graphics_settings::GraphicsSettings::high(),
            "ultra" => graphics_settings::GraphicsSettings::ultra(),
            other => {
                tracing::warn!(preset = other, "unknown graphics preset, using medium");
                graphics_settings::GraphicsSettings::medium()
            }
        };
        tracing::info!(
            preset = args.graphics.as_str(),
            atmosphere = gfx_preset.atmosphere_enabled,
            clouds = gfx_preset.cloud_enabled,
            hdr = gfx_preset.hdr_enabled,
            "graphics settings"
        );
        ecs_app.insert_resource(gfx_preset);
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
            pending_shard_type: None,
            shard_seed: 0,
            reference_position: DVec3::ZERO,
            reference_rotation: DQuat::IDENTITY,
            secondary_shard_type: None,
            system_seed: 0,
            galaxy_seed: 0,
            own_player_id: 0,
        });
        ecs_app.insert_resource(ServerWorldState {
            latest: None,
            secondary: None,
            secondary_by_type: std::collections::HashMap::new(),
            last_primary: None,
            last_primary_until: None,
            departed_ship: None,
            trace_render_until: None,
        });
        ecs_app.insert_resource(LocalPlayer {
            position: DVec3::new(0.0, 1.0, 0.0),
            velocity: DVec3::ZERO,
            is_piloting: false,
            ship_rotation: DQuat::IDENTITY,
            body_rotation: DQuat::IDENTITY,
        });
        ecs_app.insert_resource(RenderSmoothing {
            render_position: DVec3::new(0.0, 1.0, 0.0),
            render_rotation: DQuat::IDENTITY,
            ship_origin: DVec3::ZERO,
            bodies: Vec::new(),
            interpolation_t: 1.0,
            sub_grid_transforms: std::collections::HashMap::new(),
            handoff_start_world: None,
            handoff_started_at: Instant::now(),
            handoff_velocity: DVec3::ZERO,
            exit_ship_anchor: None,
            galaxy_render_position: DVec3::ZERO,
            galaxy_render_rotation: DQuat::IDENTITY,
            galaxy_velocity: DVec3::ZERO,
            prev_galaxy_position: DVec3::ZERO,
            prev_galaxy_rotation: DQuat::IDENTITY,
            last_galaxy_update_time: Instant::now(),
            has_prev_galaxy_pos: false,
        });
        ecs_app.insert_resource(SnapshotBuffer {
            snapshots: std::collections::VecDeque::with_capacity(SNAPSHOT_BUFFER_CAP),
            highest_tick: 0,
            estimated_server_tick: 0.0,
            synced: false,
            anchor_time: None,
            anchor_server_tick: 0.0,
        });
        ecs_app.insert_resource(CameraControl {
            yaw: 0.0,
            pitch: 0.0,
            pilot_yaw_rate: 0.0,
            pilot_pitch_rate: 0.0,
        });
        ecs_app.insert_resource(KeyboardState {
            keys_held: HashSet::new(),
            mouse_buttons_held: HashSet::new(),
            mouse_grabbed: false,
            scroll_delta: 0.0,
        });
        ecs_app.insert_resource(input::ActiveSeatBindings::default());
        ecs_app.insert_resource(SubGridState::default());
        ecs_app.insert_resource(FlightControl {
            selected_thrust_tier: 3,
            engines_off: false,
            thrust_limiter: 0.75,
            cruise_active: false,
            atmo_comp_active: false,
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
            primary_seed: None,
            secondary_sources: std::collections::HashMap::new(),
            grace_sources: std::collections::HashMap::new(),
            grace_expiries: std::collections::HashMap::new(),
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
            voxel_volume: None,
            cloud_system: None,
            uniform_data: vec![bytemuck::Zeroable::zeroed(); MAX_OBJECTS],
            chunk_uniform_map: std::collections::HashMap::new(),
            registry: voxeldust_core::block::BlockRegistry::new(),
            args,
        }
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let gfx = self.ecs_app.world().resource::<graphics_settings::GraphicsSettings>().clone();
        let mut gpu_state = gpu::init_gpu(window.clone(), gfx.hdr_enabled);
        // Create the block renderer now that we have GPU state.
        let br = block_render::BlockRenderer::new(
            &gpu_state.device,
            gpu_state.render_format,
            &gpu_state.bind_group_layout,
            &gpu_state.scene_bind_group_layout,
            &gpu_state.voxel_bind_group_layout,
        );
        self.block_renderer = Some(br);
        let gfx = self.ecs_app.world().resource::<graphics_settings::GraphicsSettings>().clone();
        let vol = voxel_volume::VoxelVolume::new(&gpu_state.device, &gfx);
        gpu_state.voxel_bind_group = Some(gpu::create_voxel_bind_group(
            &gpu_state.device,
            &gpu_state.voxel_bind_group_layout,
            &vol,
            gpu_state.shadow_texture_view.as_ref().unwrap(),
            gpu_state.shadow_sampler.as_ref().unwrap(),
            gpu_state.shadow_cascade_buf.as_ref().unwrap(),
        ));
        self.voxel_volume = Some(vol);
        self.cloud_system = Some(cloud_system::CloudSystem::new(&gpu_state.device));
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
            shard_seed,
            selected_thrust_tier,
            engines_off,
            cruise_active,
            atmo_comp_active,
            autopilot_target,
            warp_galaxy_position,
            warp_galaxy_rotation,
            warp_target_star_index,
            cam_yaw,
            cam_pitch,
            interpolated_bodies,
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
                player.body_rotation,
                &kb.keys_held,
                gpu.config.width,
                gpu.config.height,
                ws.latest.as_ref(),
                Some(smooth.ship_origin),
            );

            (
                warp_target_info,
                cam,
                ft.count,
                conn.secondary_shard_type,
                cam_position,
                player.velocity,
                smooth.render_rotation,
                player.is_piloting,
                conn.connected,
                conn.current_shard_type,
                conn.shard_seed,
                flight.selected_thrust_tier,
                flight.engines_off,
                flight.cruise_active,
                flight.atmo_comp_active,
                ap.target,
                if warp.galaxy_position.is_some() { Some(smooth.galaxy_render_position) } else { None },
                if warp.galaxy_rotation.is_some() { Some(smooth.galaxy_render_rotation) } else { None },
                warp.target_star_index,
                cam_ctrl.yaw,
                cam_ctrl.pitch,
                smooth.bodies.clone(),
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
            let sub_grid_assignments = {
                let sg_state = world_mut.resource::<SubGridState>();
                sg_state.assignments.clone()
            };
            let mut chunk_cache = world_mut.resource_mut::<ClientChunkCacheRes>();

            // Expire grace-pinned sources whose window has elapsed. Freed
            // sources flow into `removed_sources` via `remove_source` and
            // are picked up by the `drain_removed_sources` call below so
            // GPU buffers are released in the same frame.
            let now = Instant::now();
            let expired_seeds: Vec<u64> = chunk_cache
                .grace_expiries
                .iter()
                .filter_map(|(seed, deadline)| (now >= *deadline).then_some(*seed))
                .collect();
            for seed in expired_seeds {
                chunk_cache.grace_expiries.remove(&seed);
                if let Some(src) = chunk_cache.grace_sources.remove(&seed) {
                    // Only free if the source isn't still being used as the
                    // current primary (the promotion path keeps it in both
                    // `primary_source` and `grace_sources`; only the grace
                    // alias expires, the primary keeps ownership).
                    let still_primary = chunk_cache.primary_source == Some(src);
                    let still_secondary = chunk_cache
                        .secondary_sources
                        .values()
                        .any(|&s| s == src);
                    if !still_primary && !still_secondary {
                        chunk_cache.cache.remove_source(src);
                        info!(seed, source = src.0, "released grace-pinned source");
                    } else {
                        info!(seed, source = src.0, still_primary, still_secondary,
                            "grace window expired, source retained by active role");
                    }
                }
            }

            // Clean up GPU buffers for removed sources.
            for source in chunk_cache.drain_removed_sources() {
                br.remove_source(source);
            }

            // Process newly dirty chunks (source-aware keys).
            if chunk_cache.has_dirty() {
                let dirty_keys = chunk_cache.drain_dirty();
                let has_sub_grids = !sub_grid_assignments.is_empty();
                // Reusable 512KB padded voxel buffer — avoids repeated allocation
                // when meshing many chunks in sequence.
                let mut voxel_buf: Vec<u16> = Vec::new();
                for dk in dirty_keys {
                    if let Some(chunk) = chunk_cache.get_chunk(dk.source, dk.chunk) {
                        let neighbors = chunk_cache.get_neighbors(dk.source, dk.chunk);

                        if has_sub_grids {
                            // Sub-grid-aware meshing: generate root mesh + per-sub-grid meshes.
                            let root_mesh = block_render::generate_chunk_gpu_mesh_filtered_with_buf(
                                chunk, dk.chunk, &neighbors, &registry, false,
                                &sub_grid_assignments, None, &mut voxel_buf,
                            );
                            br.upload_chunk_mesh(&gpu.device, dk, &root_mesh);

                            // Collect which sub-grids have blocks in this chunk.
                            let cs = voxeldust_core::block::CHUNK_SIZE as i32;
                            let mut sg_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
                            for (&pos, &sg_id) in &sub_grid_assignments {
                                let (ck, _, _, _) = voxeldust_core::block::ShipGrid::world_to_chunk(pos.x, pos.y, pos.z);
                                if ck == dk.chunk {
                                    sg_ids.insert(sg_id);
                                }
                            }

                            // Generate a separate mesh for each sub-grid in this chunk.
                            for sg_id in sg_ids {
                                let sg_mesh = block_render::generate_chunk_gpu_mesh_filtered_with_buf(
                                    chunk, dk.chunk, &neighbors, &registry, false,
                                    &sub_grid_assignments, Some(sg_id), &mut voxel_buf,
                                );
                                let sg_key = block_render::SubGridMeshKey {
                                    source: dk.source,
                                    chunk: dk.chunk,
                                    sub_grid_id: sg_id,
                                };
                                br.upload_sub_grid_mesh(&gpu.device, sg_key, &sg_mesh);
                            }
                        } else {
                            // No sub-grids: standard meshing (all blocks in one mesh).
                            let mesh = block_render::generate_chunk_gpu_mesh_with_buf(
                                chunk, &neighbors, &registry, false, &mut voxel_buf,
                            );
                            br.upload_chunk_mesh(&gpu.device, dk, &mesh);
                        }

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

            let sg_state = world_mut.resource::<SubGridState>();
            let sg_assignments = &sg_state.assignments;
            let smooth_sg = world_mut.resource::<RenderSmoothing>();
            let sg_transforms = &smooth_sg.sub_grid_transforms;

            // Root-grid raycast (blocks NOT in any sub-grid).
            let root_hit = if let Some(src) = source {
                voxeldust_core::block::raycast::raycast(eye, look, 8.0, |x, y, z| {
                    let pos = glam::IVec3::new(x, y, z);
                    if sg_assignments.contains_key(&pos) { return false; } // skip sub-grid blocks
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

            // Sub-grid raycasts: transform ray into each sub-grid's local frame.
            let mut best_hit = root_hit;
            // Collect unique sub-grid IDs that have blocks.
            let mut active_sg_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
            for &sg_id in sg_assignments.values() {
                active_sg_ids.insert(sg_id);
            }
            for sg_id in &active_sg_ids {
                let sgt = match sg_transforms.get(sg_id) {
                    Some(t) => t,
                    None => continue,
                };
                // Inverse transform: world → local root-space.
                // Forward: world = T(translation) * R(rotation) * T(-anchor) * root_point
                // Inverse: root_point = T(anchor) * R_inv * T(-translation) * world_point
                let rot_inv = sgt.rotation.inverse();
                let local_eye = sgt.anchor + rot_inv * (eye - sgt.translation);
                let local_look = rot_inv * look;

                let sg_id_copy = *sg_id;
                let sg_hit = if let Some(src) = source {
                    voxeldust_core::block::raycast::raycast(local_eye, local_look, 8.0, |x, y, z| {
                        let pos = glam::IVec3::new(x, y, z);
                        if sg_assignments.get(&pos).copied() != Some(sg_id_copy) { return false; }
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
                if let Some(sh) = sg_hit {
                    let closer = best_hit.as_ref().map_or(true, |bh| sh.distance < bh.distance);
                    if closer {
                        best_hit = Some(sh);
                    }
                }
            }

            world_mut.resource_mut::<BlockTarget>().hit = best_hit;
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

        // Update voxel volume from nearby chunk data.
        if let Some(vol) = &mut self.voxel_volume {
            let world = self.ecs_app.world();
            let gfx = world.resource::<graphics_settings::GraphicsSettings>();
            let resized = vol.resize_if_needed(&gpu.device, gfx);

            // Recreate the voxel bind group if the volume was resized.
            if resized {
                gpu.voxel_bind_group = Some(gpu::create_voxel_bind_group(
                    &gpu.device,
                    &gpu.voxel_bind_group_layout,
                    vol,
                    gpu.shadow_texture_view.as_ref().unwrap(),
                    gpu.shadow_sampler.as_ref().unwrap(),
                    gpu.shadow_cascade_buf.as_ref().unwrap(),
                ));
            }

            // Voxel volume is only meaningful on planet/ship shards where block
            // coordinates fit i32. On system shards, render_position is in system-space
            // (~1e10 meters) which overflows i32 on conversion to block coords.
            let safe_for_voxel_volume = current_shard_type != voxeldust_core::client_message::shard_type::SYSTEM;
            if safe_for_voxel_volume {
                let chunk_cache_res = world.resource::<ClientChunkCacheRes>();
                let active: Vec<_> = chunk_cache_res.active_sources_iter().collect();
                let player_block = glam::IVec3::new(
                    render_position.x.floor() as i32,
                    render_position.y.floor() as i32,
                    render_position.z.floor() as i32,
                );
                vol.populate(player_block, &*chunk_cache_res, &self.registry, &active);
                vol.upload(&gpu.queue);
            }

            if !safe_for_voxel_volume {
                // System shard: no voxel volume, skip transform/propagation.
            } else {
            // Compute the base_transform matching the one used for chunk rendering,
            // so the world_to_volume matrix correctly maps fragment world_pos back
            // to volume-index coordinates.
            let base_transform = if current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
                let ship_rot_mat = glam::Mat4::from_quat(ship_rotation.as_quat());
                let origin_local = -(render_position + DVec3::new(0.0, gpu::EYE_HEIGHT, 0.0));
                let ship_origin_offset = (ship_rotation * origin_local).as_vec3();
                glam::Mat4::from_translation(ship_origin_offset) * ship_rot_mat
            } else {
                // Planet/system shard: camera-relative translation only.
                glam::Mat4::from_translation(-render_position.as_vec3())
            };

            // Extract sun direction from world state (same space as fragment world_pos).
            let ws_ref = world.resource::<ServerWorldState>();
            let sun_dir_world = ws_ref.latest.as_ref()
                .and_then(|ws| ws.lighting.as_ref())
                .map(|l| l.sun_direction.normalize().as_vec3())
                .unwrap_or(glam::Vec3::new(0.0, -1.0, 0.0));

            let params = vol.compute_params(base_transform, sun_dir_world);
            gpu.queue.write_buffer(&vol.params_buf, 0, bytemuck::bytes_of(&params));

            // Dispatch block light propagation compute shader if needed.
            if vol.needs_propagation {
                // Clear light texture A to prevent uninitialized data from flooding.
                vol.clear_light_texture_a(&gpu.queue);

                let world = self.ecs_app.world();
                let gfx = world.resource::<graphics_settings::GraphicsSettings>();
                let iterations = gfx.block_light_iterations;
                // Round up to even so the result always lands in texture A
                // (which is what the fragment shader reads via the voxel bind group).
                let iterations = if iterations % 2 != 0 { iterations + 1 } else { iterations };
                let mut encoder = gpu.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("light_propagate") },
                );
                vol.propagate_light(&mut encoder, iterations);
                gpu.queue.submit(std::iter::once(encoder.finish()));
            }
            } // end safe_for_voxel_volume
        }

        // Generate cloud noise textures if near a cloudy planet (once per planet).
        if let Some(cloud_sys) = &mut self.cloud_system {
            let world = self.ecs_app.world();
            let gfx = world.resource::<graphics_settings::GraphicsSettings>();
            if gfx.cloud_enabled {
                let sys_cache = world.resource::<SystemParamsCache>();
                let ws_res = world.resource::<ServerWorldState>();
                if let (Some(sys), Some(ws)) = (&sys_cache.0, ws_res.latest.as_ref()) {
                    let cam_sys_pos = {
                        let smooth = world.resource::<RenderSmoothing>();
                        let conn = world.resource::<ConnectionInfo>();
                        // Approximate cam_system_pos for cloud planet detection.
                        if conn.current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
                            ws.origin + smooth.render_position
                        } else {
                            smooth.render_position
                        }
                    };
                    // Check for nearby cloudy planet.
                    let mut found_any_cloud_planet = false;
                    for body in &ws.bodies {
                        if body.body_id == 0 { continue; }
                        let pi = (body.body_id - 1) as usize;
                        if pi >= sys.planets.len() { continue; }
                        let planet = &sys.planets[pi];
                        if !planet.clouds.has_clouds { continue; }
                        found_any_cloud_planet = true;
                        let dist = (cam_sys_pos - body.position).length();
                        // Clouds are visible from far away (white patches on planet).
                        // Use a generous range: render clouds if planet subtends > 0.5° on screen.
                        let planet_angular = (planet.radius_m / dist).atan();
                        let min_angular = 0.005; // ~0.3 degrees — planet clearly visible
                        if planet_angular > min_angular {
                            let prev_has = cloud_sys.has_noise();
                            cloud_sys.ensure_noise_for_planet(&gpu.device, &gpu.queue, planet.planet_seed);
                            // Rebuild composite bind group if noise textures were just created.
                            if !prev_has && cloud_sys.has_noise() {
                                if let (Some(layout), Some(params_buf), Some(atmo_buf),
                                       Some(depth_tex), Some(cloud_buf),
                                       Some(hdr_v), Some(shape_v), Some(detail_v)) = (
                                    &gpu.composite_bind_group_layout,
                                    &gpu.composite_params_buf,
                                    &gpu.atmosphere_buf,
                                    &gpu.depth_texture,
                                    &gpu.cloud_uniform_buf,
                                    &gpu.hdr_view,
                                    &cloud_sys.shape_view,
                                    &cloud_sys.detail_view,
                                ) {
                                    let screen_s = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                                        label: Some("screen_sampler"),
                                        mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
                                        ..Default::default()
                                    });
                                    let depth_sv = depth_tex.create_view(&Default::default());
                                    gpu.composite_bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: Some("composite_bind_group_clouds"),
                                        layout,
                                        entries: &[
                                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr_v) },
                                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&screen_s) },
                                            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&depth_sv) },
                                            wgpu::BindGroupEntry { binding: 4, resource: atmo_buf.as_entire_binding() },
                                            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(shape_v) },
                                            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(detail_v) },
                                            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&cloud_sys.noise_sampler) },
                                            wgpu::BindGroupEntry { binding: 8, resource: cloud_buf.as_entire_binding() },
                                            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(
                                                cloud_sys.weather_map_view_or_dummy(&gpu.device)
                                            ) },
                                        ],
                                    }));
                                    tracing::info!("rebuilt composite bind group with cloud noise textures");
                                }
                            }
                            // Request async weather map generation for this planet.
                            cloud_sys.request_weather_update(planet, ws.game_time);

                            break;
                        }
                    }

                    // Poll for completed weather map generation (non-blocking).
                    let weather_rebuilt = cloud_sys.poll_weather_map(&gpu.device, &gpu.queue);
                    if weather_rebuilt {
                        // Rebuild composite bind group with the new weather map texture.
                        if let (Some(layout), Some(params_buf), Some(atmo_buf),
                               Some(depth_tex), Some(cloud_buf), Some(hdr_v)) = (
                            &gpu.composite_bind_group_layout,
                            &gpu.composite_params_buf,
                            &gpu.atmosphere_buf,
                            &gpu.depth_texture,
                            &gpu.cloud_uniform_buf,
                            &gpu.hdr_view,
                        ) {
                            let screen_s = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                                label: Some("screen_sampler"),
                                mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
                                ..Default::default()
                            });
                            let depth_sv = depth_tex.create_view(&Default::default());
                            let shape_v = cloud_sys.shape_view.as_ref().unwrap_or(&cloud_sys.dummy_2d_view);
                            let detail_v = cloud_sys.detail_view.as_ref().unwrap_or(&cloud_sys.dummy_2d_view);
                            gpu.composite_bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("composite_bind_group_weather"),
                                layout,
                                entries: &[
                                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr_v) },
                                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&screen_s) },
                                    wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&depth_sv) },
                                    wgpu::BindGroupEntry { binding: 4, resource: atmo_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(shape_v) },
                                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(detail_v) },
                                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&cloud_sys.noise_sampler) },
                                    wgpu::BindGroupEntry { binding: 8, resource: cloud_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(
                                        cloud_sys.weather_map_view_or_dummy(&gpu.device)
                                    ) },
                                ],
                            }));
                            tracing::info!("rebuilt composite bind group with weather map");
                        }
                    }

                }
            }
        }

        // Re-read resources for render_frame call.
        let world = self.ecs_app.world();
        let ws = world.resource::<ServerWorldState>();
        let ap = world.resource::<ClientAutopilot>();
        let sys_params = world.resource::<SystemParamsCache>();
        let gfx_settings = world.resource::<graphics_settings::GraphicsSettings>().clone();
        let sg_state = world.resource::<SubGridState>();

        let smooth = world.resource::<RenderSmoothing>();

        let panel_action = render::render_frame(
            gpu,
            window,
            &mut self.uniform_data,
            &mut self.chunk_uniform_map,
            &cam,
            // Use `effective()` so the scene (ships, entities, bodies) keeps
            // drawing from the stashed `last_primary` during the brief window
            // of a hard primary transition — prevents the 200-500 ms blackout
            // users would otherwise see while the new primary's first
            // WorldState is in flight.
            ws.effective(),
            ws.secondary.as_ref(),
            // Grace-window fallback: during a seamless promotion, the old
            // primary WS is stashed in `last_primary` for
            // `LAST_PRIMARY_GRACE_SECS` so the renderer can keep drawing its
            // LOD-proxy ship entity (the new primary marks its own ship
            // `is_own`, which draw_entities would otherwise skip).
            ws.grace_fallback(),
            &interpolated_bodies,
            current_shard_type,
            shard_seed,
            render_position,
            player_velocity,
            ship_rotation,
            is_piloting,
            connected,
            selected_thrust_tier,
            engines_off,
            cruise_active,
            atmo_comp_active,
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
            &gfx_settings,
            &smooth.sub_grid_transforms,
            &sg_state.assignments,
            // A4b: route block-mesh rendering through ship_id → source lookup.
            // Primary source = own ship's chunks on SHIP shard. Secondary ship
            // sources map nearby-ship entity_id → their observer chunk source.
            world.resource::<ClientChunkCacheRes>().primary_source,
            &world.resource::<ClientChunkCacheRes>().secondary_sources,
            // Grace-pinned sources keep ships renderable across the
            // handoff seam (see ClientChunkCacheRes::grace_sources).
            &world.resource::<ClientChunkCacheRes>().grace_sources,
            // Departed ship: disabled — rendering the old ship as a
            // system-space frozen mesh produced a "ghost ship" that
            // drifted off at orbital velocity (~110 km/s) because the
            // stashed world_position doesn't advance with the ship's
            // physical motion. The entity-based path (push_ship_entity
            // reading grace_ship_sources first) correctly renders the
            // ship at its CURRENT authoritative position every frame,
            // which is what we want. DepartedShip captured data is
            // still kept in ServerWorldState in case we later want it
            // for a specific edge case (e.g. the target shard doesn't
            // yet carry the ship as an entity), but the render path is
            // off.
            None,
            self.cloud_system.as_ref(),
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
                        seated_channel_name: config.seated_channel_name.clone(),
                        power_source: config.power_source.clone(),
                        power_consumer: config.power_consumer.clone(),
                        flight_computer: config.flight_computer.clone(),
                        hover_module: config.hover_module.clone(),
                        autopilot: config.autopilot.clone(),
                        warp_computer: config.warp_computer.clone(),
                        engine_controller: config.engine_controller.clone(),
                        mechanical: config.mechanical.clone(),
                    };
                    let msg = voxeldust_core::client_message::ClientMsg::BlockConfigUpdate(update);
                    let world = self.ecs_app.world();
                    let net = world.resource::<NetworkChannels>();
                    send_tcp_msg(net, msg);
                    info!(block = ?config.block_pos, "config saved (sent to server)");
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
            ctx.cam.pilot_yaw_rate = (ctx.cam.pilot_yaw_rate - motion.delta_x * sensitivity * 1.5).clamp(-1.0, 1.0);
            ctx.cam.pilot_pitch_rate = (ctx.cam.pilot_pitch_rate - motion.delta_y * sensitivity * 1.5).clamp(-1.0, 1.0);
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

                // WASD cancels autopilot and cruise.
                if matches!(key,
                    KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD) {
                    if actions.ap.target.is_some() {
                        actions.ap.target = None;
                        actions.ap.trajectory_plan = None;
                        actions.ap.mode = AutopilotMode::DirectApproach;
                    }
                    if actions.flight.cruise_active {
                        actions.flight.cruise_active = false;
                        info!("cruise cancelled by manual input");
                    }
                }

                // C key: toggle supercruise.
                if key == KeyCode::KeyC && is_piloting && !actions.flight.engines_off {
                    actions.flight.cruise_active = !actions.flight.cruise_active;
                    info!(cruise = actions.flight.cruise_active, "cruise mode toggled");
                }

                // H key: toggle atmosphere compensation (hover).
                if key == KeyCode::KeyH && is_piloting && !actions.flight.engines_off {
                    actions.flight.atmo_comp_active = !actions.flight.atmo_comp_active;
                    info!(atmo_comp = actions.flight.atmo_comp_active, "atmosphere compensation toggled");
                }

                // X key: toggle engine cutoff.
                if key == KeyCode::KeyX && is_piloting {
                    actions.flight.engines_off = !actions.flight.engines_off;
                    if actions.flight.engines_off {
                        actions.flight.cruise_active = false;
                    }
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
                // Always allowed — even while piloting (to toggle seat on/off).
                // Roll input goes through send_input_with_dt, not through INTERACT.
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
                                    winit::event::MouseButton::Right => action::PLACE_SUB,
                                    winit::event::MouseButton::Left => action::REMOVE_SUB,
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
    mut kb: ResMut<KeyboardState>,
    mut flight: ResMut<FlightControl>,
    mut cam: ResMut<CameraControl>,
    player: Res<LocalPlayer>,
    ft: Res<FrameTime>,
    input_ctx: Res<events::InputContext>,
    mut active_seat: ResMut<input::ActiveSeatBindings>,
) {
    // Suppress movement input when not in Game mode.
    if input_ctx.mode != events::InputMode::Game {
        return;
    }
    let cam = &mut *cam;
    let limiter = flight.thrust_limiter;
    let scroll_delta = kb.scroll_delta;
    kb.scroll_delta = 0.0; // Consume scroll delta this frame.
    input::send_input_with_dt(
        &net.input_tx,
        &kb.keys_held,
        &kb.mouse_buttons_held,
        player.is_piloting,
        cam.yaw,
        cam.pitch,
        &mut cam.pilot_yaw_rate,
        &mut cam.pilot_pitch_rate,
        &mut flight.selected_thrust_tier,
        limiter,
        ft.count,
        ft.dt,
        &mut active_seat,
        scroll_delta,
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
                    let (depth_tex, depth_v) = gpu::create_depth_texture(&gpu.device, gpu.config.width, gpu.config.height, wgpu::TextureFormat::Depth32Float);
                    gpu.depth_view = depth_v;
                    gpu.depth_texture = Some(depth_tex);
                    // Recreate HDR texture and composite bind group on resize.
                    if gpu.hdr_enabled {
                        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("hdr_color"),
                            size: wgpu::Extent3d { width: gpu.config.width, height: gpu.config.height, depth_or_array_layers: 1 },
                            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba16Float,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        });
                        let view = tex.create_view(&Default::default());
                        // Rebuild composite bind group with new HDR + depth views.
                        if let (Some(layout), Some(params_buf), Some(atmo_buf), Some(depth_tex)) =
                            (&gpu.composite_bind_group_layout, &gpu.composite_params_buf,
                             &gpu.atmosphere_buf, &gpu.depth_texture)
                        {
                            let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                                label: Some("screen_sampler"),
                                mag_filter: wgpu::FilterMode::Linear,
                                min_filter: wgpu::FilterMode::Linear,
                                ..Default::default()
                            });
                            let depth_sample_view = depth_tex.create_view(&Default::default());
                            // Dummy 3D texture for cloud bindings (will be replaced when cloud noise loads).
                            let dummy_3d = gpu.device.create_texture(&wgpu::TextureDescriptor {
                                label: Some("dummy_3d"),
                                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                                view_formats: &[],
                            });
                            let dummy_3d_view = dummy_3d.create_view(&Default::default());
                            let cloud_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                                label: Some("cloud_noise_sampler"),
                                address_mode_u: wgpu::AddressMode::Repeat,
                                address_mode_v: wgpu::AddressMode::Repeat,
                                address_mode_w: wgpu::AddressMode::Repeat,
                                mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
                                ..Default::default()
                            });
                            let cloud_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("cloud_uniforms_resize"),
                                size: std::mem::size_of::<cloud_system::CloudUniforms>() as u64,
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            gpu.composite_bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("composite_bind_group"),
                                layout,
                                entries: &[
                                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                                    wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&depth_sample_view) },
                                    wgpu::BindGroupEntry { binding: 4, resource: atmo_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&dummy_3d_view) },
                                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&dummy_3d_view) },
                                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&cloud_sampler) },
                                    wgpu::BindGroupEntry { binding: 8, resource: cloud_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView({
                                        // Use weather map view from cloud system if available, else dummy
                                        if let Some(cs) = &self.cloud_system {
                                            cs.weather_map_view_or_dummy(&gpu.device)
                                        } else {
                                            // No cloud system — create inline dummy 2D
                                            let d = gpu.device.create_texture(&wgpu::TextureDescriptor {
                                                label: Some("dummy_weather"), size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                                                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                                                format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
                                            });
                                            // This leaks the texture, but resize is rare. Acceptable.
                                            Box::leak(Box::new(d.create_view(&Default::default())))
                                        }
                                    }) },
                                ],
                            }));
                        }
                        gpu.hdr_texture = Some(tex);
                        gpu.hdr_view = Some(view);
                    }
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
