//! `vd-client-render` (Tier-B, P1.5 Slice 3 T4) — the windowed Bevy client.
//!
//! This is the ONLY place wgpu/winit/Bevy live. It is a pure CONSUMER of the
//! renderer-free core: it reads an immutable [`RenderSnapshot`] the core thread
//! publishes lock-free (via `ArcSwap`) and writes injected [`InputAction`]s into the
//! SAME bounded mailbox `vdctl` feeds — so a windowed keypress is byte-identical to a
//! dev-control command, and the sim/netcode runs at a deterministic 20 Hz on its own
//! thread while this window interpolates smoothly at the display refresh rate.
//!
//! No prediction: poses come from [`RenderSnapshot::rendered`] (delivered + interpolated,
//! never extrapolated). Mouse-look turns the LOCAL camera immediately (a view-only
//! response); the rendered ENTITY orientation stays server-delivered. ALL branching
//! logic lives in `vd-client-harness` (Tier-A); this crate is the glue, proven by the
//! human-in-the-loop window + (T6) `G-RENDER-SMOKE`, not by `llvm-cov`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::SyncSender;
use std::time::Instant;

use arc_swap::ArcSwap;
use bevy::app::{AppExit, ScheduleRunnerPlugin};
use bevy::camera::RenderTarget;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::image::TextureFormatPixelInfo;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    self, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel,
};
use bevy::render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode, PollType,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TextureFormat, TextureUsages,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy::window::{CursorGrabMode, CursorOptions, ExitCondition, PrimaryWindow};
use bevy::winit::WinitPlugin;
use bevy_egui::{
    EguiContext, EguiContexts, EguiGlobalSettings, EguiMultipassSchedule, EguiPlugin,
    EguiPrimaryContextPass, PrimaryEguiContext, egui,
};
use crossbeam_channel::{Receiver, Sender};
use vd_client::net::ClientPhase;
use vd_client::realm_scene::{MeshPrim, to_render_prims};
use vd_client::render_snapshot::RenderSnapshot;
use vd_client_harness::camera::FollowCamera;
use vd_client_harness::capture::capture_rel_path;
use vd_client_harness::input_map::{MovementKeys, mouse_look};
use vd_client_harness::manifest::CaptureKind;
use vd_core::EntityId;
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_devproto::InputAction;

// ---- render tuning (named consts; no inline magic numbers) ----------------------
const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;
/// The scene clear color (sRGB) — the cleared-background of BOTH the windowed and the
/// headless-capture cameras (single-sourced so the two can never drift, and so the
/// G-RENDER-SMOKE content check measures content against the one true background). Deep-space
/// BLACK: the starfield is the only sky, so the gaps between stars must read as empty space.
const CLEAR_SRGB: [f32; 3] = [0.0, 0.0, 0.0];
const DOT_RADIUS: f32 = 0.5;
/// Reference-scene extents so motion is VISIBLE in the empty stub world (P1.5 has no
/// terrain): a ground plate + a ring of distinct landmark pillars for parallax. Pure
/// render scaffolding — replaced wholesale by real terrain at P4, never extended.
const GROUND_HALF: f32 = 250.0;
const LANDMARK_RING_RADIUS: f32 = 25.0;
const LANDMARK_HEIGHT: f32 = 6.0;
const LANDMARK_COLORS: [Color; 8] = [
    Color::srgb(0.9, 0.3, 0.3),
    Color::srgb(0.9, 0.6, 0.2),
    Color::srgb(0.9, 0.9, 0.2),
    Color::srgb(0.3, 0.9, 0.3),
    Color::srgb(0.2, 0.8, 0.8),
    Color::srgb(0.3, 0.5, 0.95),
    Color::srgb(0.6, 0.3, 0.9),
    Color::srgb(0.9, 0.3, 0.7),
];
// ---- starfield (VU-0: the always-visible night sky) ------------------------------------
/// Fixed galaxy seed for the ambient background starfield. The far, UNREACHABLE stars are a
/// permanent night-sky backdrop (a legitimate distant-point-of-light representation — NOT a
/// mesh proxy). SEAM: at the real-astronomy / galaxy-catalog slice (warp) this seed becomes
/// SERVER-provided (the agnostic client discovers it), the NEAR reachable systems promote to
/// real realm content you fly to, and the far field stays a backdrop. Deterministic (the same
/// sky every session, on every machine) via `SplitMix64`.
const STARFIELD_SEED: u64 = 0x5644_5354_4152_5300; // "VDSTARS\0"
/// Star-sphere radius (render-m). 50x beyond the 40 render-m visual system, so the field reads
/// as at infinity (parallax-free) yet sits inside [`STAR_FAR_PLANE`]. Revisited at the
/// real-astronomy switch (a floating-origin / skybox pass supersedes this follow-sphere).
const STAR_SPHERE_RADIUS: f32 = 2000.0;
/// Windowed camera far plane — bumped from Bevy's default 1000 so the star sphere is drawn.
const STAR_FAR_PLANE: f32 = 6000.0;
/// Uniform all-sky star count + the extra Milky-Way band over-density.
const STAR_COUNT_UNIFORM: usize = 1600;
const STAR_COUNT_BAND: usize = 1200;
/// Milky-Way band half-thickness (rad) — band stars cluster within this of the galactic plane.
const STAR_BAND_HALF_ANGLE: f32 = 0.32;
/// Apparent star size (render-m at [`STAR_SPHERE_RADIUS`]) — MIN≈1px, MAX≈3px in the 1280-wide view.
const STAR_SIZE_MIN: f32 = 0.7;
const STAR_SIZE_MAX: f32 = 2.4;
/// Tilted galactic-plane normal (unnormalised; normalised at build) so the band is not axis-aligned.
const GALACTIC_NORMAL: Vec3 = Vec3::new(0.30, 1.0, -0.20);
/// Spectral-class palette `(base_color, emissive)` O/B..M — real-ish star colors. A star's tier
/// indexes this; shared as N materials so the whole field batches by material.
const STAR_TIERS: [([f32; 3], [f32; 3]); 5] = [
    ([0.75, 0.83, 1.00], [0.55, 0.62, 0.90]), // O/B blue-white
    ([0.95, 0.97, 1.00], [0.75, 0.77, 0.82]), // A  white
    ([1.00, 0.97, 0.86], [0.85, 0.80, 0.62]), // G  yellow-white
    ([1.00, 0.86, 0.62], [0.90, 0.60, 0.35]), // K  orange
    ([1.00, 0.72, 0.55], [0.85, 0.42, 0.28]), // M  red
];
/// Cumulative spectral weights (M/K dwarfs dominate the real sky) — a uniform draw picks a tier.
const STAR_TIER_CUM: [f32; 5] = [0.07, 0.20, 0.40, 0.65, 1.00]; // O/B, A, G, K, M
/// Offscreen capture resolution. Width chosen so `width*4` is NOT a multiple of 256
/// (1280*4 = 5120 IS a multiple → no padding; use 1284 to force the 256-byte row-pad
/// strip path that the readback must handle). Height even. PUBLIC so a capture-gate test
/// (`render_boxes_smoke`) can reconstruct the exact `fit_camera_to_scene` camera the offscreen
/// render used, and project a box's screen AABB against the very pixels it produced.
pub const CAPTURE_W: u32 = 1284;
pub const CAPTURE_H: u32 = 720;
/// Frames to render before the first capture can serve (let the render world warm up +
/// the egui pass + the readback pipeline fill — the spike used 8).
const CAPTURE_PRE_ROLL: u32 = 8;

/// Which app to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    /// A real window for a human (winit + egui primary-context HUD).
    Windowed,
    /// Headless offscreen render → wgpu readback → PNG on request (the AGENT's eyes; no
    /// display needed). egui composites into the captured image via the multipass schedule.
    Capture,
}

/// A capture request from the dev-control listener to the render thread. The listener has
/// ALREADY done any `--at-tick` wait (reusing wait-until on the delivered universe tick),
/// so this is just "capture the current frame now → reply with the path".
pub struct CaptureJob {
    /// Screenshot (→ `shots/`) or one record Frame (→ `frames/`); also the manifest kind.
    pub kind: CaptureKind,
    /// Optional agent name for the file (else a frame counter).
    pub label: Option<String>,
    /// The render thread sends the result here (or an Err string).
    pub reply: Sender<Result<CaptureResult, String>>,
}

/// The result of a served capture.
pub struct CaptureResult {
    /// The written PNG path (cwd-relative — `runs/<run>/shots/foo.png`): what `vdctl`
    /// reports so an agent can open it directly.
    pub path: String,
    /// The SAME path relative to the run dir (`shots/foo.png`) — what the manifest stores.
    pub rel_path: String,
    /// The freshest delivered universe tick sampled from the render snapshot AT capture time
    /// (NOT a post-roundtrip poll on the dev-control side, which drifts forward by the whole
    /// render round-trip) — so the manifest tick identifies the captured world. Residual: the
    /// served readback is up to `CAPTURE_PRE_ROLL` frames old, so this can lead the pixels by
    /// that bounded margin; far tighter than the unbounded post-roundtrip skew it replaces.
    pub freshest_tick: Option<u64>,
    /// The render cursor at capture time (the same snapshot), for the manifest.
    pub cursor: Option<f64>,
}

/// The handles the bin wires into the window (constructed on the main thread before
/// `run_window`). Every field is `Send + Sync`, so the Bevy resource is plain.
pub struct RenderHandles {
    /// The lock-free render snapshot, published every step by the core thread.
    pub snapshot: Arc<ArcSwap<RenderSnapshot>>,
    /// The input mailbox — the SAME `SyncSender` the dev-control listener feeds.
    pub input: SyncSender<InputAction>,
    /// The SHARED shed-command counter (`DevState::dev_commands_dropped`): the window
    /// bumps it on a dropped input, so BOTH producers feed one observable back-pressure
    /// metric (the "back-pressure is never silent" rule).
    pub dropped: Arc<AtomicU64>,
    /// Liveness flag the core thread flips false when its loop EXITS (gateway Close /
    /// panic). The window emits `AppExit` when it sees false — bidirectional shutdown.
    pub core_alive: Arc<AtomicBool>,
    /// The shared monotonic epoch the core anchored its render clock on (so this
    /// thread computes the display cursor in the same timeline).
    pub started_at: Instant,
    /// Windowed (human) or headless Capture (agent).
    pub mode: RenderMode,
    /// Capture requests (Capture mode only — `None` for Windowed). A `crossbeam`
    /// receiver so it can be a Bevy Resource (Send+Sync).
    pub captures: Option<Receiver<CaptureJob>>,
    /// Where capture PNGs land (Capture mode); the bin creates the run dir.
    pub runs_dir: PathBuf,
}

/// The bin's handles, held as a Bevy resource (read by every system).
#[derive(Resource)]
struct Net {
    snapshot: Arc<ArcSwap<RenderSnapshot>>,
    input: SyncSender<InputAction>,
    dropped: Arc<AtomicU64>,
    core_alive: Arc<AtomicBool>,
    started_at: Instant,
}

/// The LOCAL first-person camera state (turns immediately on mouse-look; the entity
/// orient stays server-delivered) + the last movement keys we sent (latest-wins on the
/// wire, so we only resend on change).
#[derive(Resource)]
struct CameraState {
    cam: FollowCamera,
    last_movement: MovementKeys,
}

/// The map from a delivered entity to its spawned Bevy dot entity.
#[derive(Resource, Default)]
struct DotEntities(BTreeMap<EntityId, Entity>);

/// The map from a realm to its spawned translucent box entity — the SIBLING of [`DotEntities`]
/// (the binary-render rule: the realm-box render is a DISTINCT path, never bolted onto the dot
/// path). Keyed by [`RealmId`] for a deterministic spawn/despawn order.
#[derive(Resource, Default)]
struct RealmBoxEntities(BTreeMap<RealmId, Entity>);

/// Marker: a rendered realm box (a translucent colored volume).
#[derive(Component)]
struct RealmBoxMarker;

/// Shared dot render assets (one sphere mesh; own/other materials) built once at setup.
#[derive(Resource)]
struct DotAssets {
    mesh: Handle<Mesh>,
    own: Handle<StandardMaterial>,
    other: Handle<StandardMaterial>,
}

/// Marker: a rendered dot (an authoritative delivered entity).
#[derive(Component)]
struct Dot;

/// Marker: the follow camera.
#[derive(Component)]
struct FollowCam;

/// Run the client renderer. BLOCKS until exit; the bin MUST call this on the MAIN thread
/// (winit/the runner need it) with the core loop on a separate thread. Dispatches on the
/// mode: a real window (human) or headless offscreen capture (the agent's eyes).
pub fn run(handles: RenderHandles) {
    match handles.mode {
        RenderMode::Windowed => run_windowed(handles),
        RenderMode::Capture => run_capture(handles),
    }
}

/// The windowed app (winit window + egui primary-context HUD + winit input).
fn run_windowed(handles: RenderHandles) {
    tracing::info!("windowed client starting (Bevy {}x{})", WINDOW_W, WINDOW_H);
    App::new()
        .insert_resource(ClearColor(Color::srgb(
            CLEAR_SRGB[0],
            CLEAR_SRGB[1],
            CLEAR_SRGB[2],
        )))
        .insert_resource(Net {
            snapshot: handles.snapshot,
            input: handles.input,
            dropped: handles.dropped,
            core_alive: handles.core_alive,
            started_at: handles.started_at,
        })
        .insert_resource(CameraState {
            cam: FollowCamera::new(DVec3::Y),
            last_movement: MovementKeys::default(),
        })
        .init_resource::<DotEntities>()
        .init_resource::<RealmBoxEntities>()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Voxeldust — dev client".into(),
                resolution: (WINDOW_W, WINDOW_H).into(),
                ..default()
            }),
            // First-person: start the pointer LOCKED + hidden so mouse-look gets unbounded
            // delta and never sticks at the window edge (the AAA FPS baseline). Esc releases
            // it (click away / close); a click re-grabs — see `cursor_grab`.
            primary_cursor_options: Some(CursorOptions {
                grab_mode: CursorGrabMode::Locked,
                visible: false,
                ..default()
            }),
            ..default()
        }))
        // egui (multipass primary context — the default; auto-creates the window's
        // PrimaryEguiContext + its EguiPrimaryContextPass schedule).
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                input_system,
                sync_world,
                // Sync the realm spheres, then despawn the stub-world scaffolding once they exist (chained
                // so the despawn sees the just-spawned boxes the same frame — no scaffold flash in space).
                (sync_realm_boxes, despawn_reference_scaffold).chain(),
                // Keep the ambient starfield centered on the camera (an inertial sky at infinity).
                follow_starfield,
                cursor_grab,
                exit_when_core_stops,
            ),
        )
        // The HUD draws in the egui pass (NOT Update — the 0.39 multipass idiom).
        .add_systems(EguiPrimaryContextPass, hud_primary)
        .run();
}

/// Windowed setup: the window follow-camera + the shared world.
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // First-person follow camera (positioned each frame by `sync_world`). Far plane bumped so the
    // ambient star sphere (at `STAR_SPHERE_RADIUS`) is not clipped by Bevy's default 1000 m far.
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            far: STAR_FAR_PLANE,
            ..default()
        }),
        Transform::from_xyz(0.0, 1.6, 0.0).looking_at(Vec3::NEG_Z, Vec3::Y),
        FollowCam,
    ));
    setup_world(&mut commands, &mut meshes, &mut materials);
    setup_starfield(&mut commands, &mut meshes, &mut materials);
}

// (the windowed camera above; the shared world below — capture's offscreen camera lives
// in `run_capture` and reuses `setup_world`.)

/// Marker for the empty-stub-world reference scaffolding (the ground plate + the landmark pillars) — a
/// P1.5 MOTION reference, despawned by [`despawn_reference_scaffold`] the moment real realm content loads.
#[derive(Component)]
struct ReferenceScaffold;

/// Despawn the reference scaffolding ONCE real realm content (any [`RealmBox`]) is present. The visual
/// universe — and the eventual game — ALWAYS has realm spheres, so the ground/pillars belong ONLY to the
/// empty P1.5 stub world (`render_smoke`, launched with NO `--realm-boxes`), never the SPACE view. Seamless:
/// they vanish as the boot realm scene loads. Windowed-only — the capture path deliberately keeps them so
/// `render_smoke`'s empty-world content floor still holds (`render_boxes_smoke` checks the box, not these).
fn despawn_reference_scaffold(
    boxes: Res<RealmBoxEntities>,
    scaffold: Query<Entity, With<ReferenceScaffold>>,
    mut commands: Commands,
) {
    if boxes.0.is_empty() {
        return;
    }
    for entity in &scaffold {
        commands.entity(entity).despawn();
    }
}

/// Spawn the key light, reference scene (ground + landmark pillars), and the shared dot
/// assets — identical for the windowed and capture cameras (DRY: one world, two views).
fn setup_world(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    // A single key light (dots are also emissive, so they read even unlit).
    commands.spawn((
        DirectionalLight {
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(30.0, 60.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Reference ground plate (a thin slab — motion reference for the empty stub world). Tagged
    // `ReferenceScaffold`: despawned the moment real realm content loads (never in the space view).
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(GROUND_HALF * 2.0, 0.2, GROUND_HALF * 2.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.06, 0.08, 0.11),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.1, 0.0),
        ReferenceScaffold,
    ));
    // Landmark pillars in a ring — distinct colors give parallax as the player walks (stub-world only).
    let pillar = meshes.add(Cuboid::new(1.0, LANDMARK_HEIGHT, 1.0));
    for (i, color) in LANDMARK_COLORS.iter().enumerate() {
        let angle = i as f32 / LANDMARK_COLORS.len() as f32 * std::f32::consts::TAU;
        commands.spawn((
            Mesh3d(pillar.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: *color,
                ..default()
            })),
            Transform::from_xyz(
                LANDMARK_RING_RADIUS * angle.cos(),
                LANDMARK_HEIGHT * 0.5,
                LANDMARK_RING_RADIUS * angle.sin(),
            ),
            ReferenceScaffold,
        ));
    }

    // Shared dot assets (built once, reused per spawned dot).
    commands.insert_resource(DotAssets {
        mesh: meshes.add(Sphere::new(DOT_RADIUS)),
        own: materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.9, 1.0),
            emissive: LinearRgba::rgb(0.0, 0.6, 0.8),
            ..default()
        }),
        other: materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.8, 0.35),
            emissive: LinearRgba::rgb(0.12, 0.10, 0.0),
            ..default()
        }),
    });
    // The stats HUD is drawn each frame by `hud_primary` via egui (no entity to spawn).
}

/// Marker: the root of the ambient starfield. It follows the camera POSITION each frame (so the
/// sky is at infinity — parallax-free) but keeps identity ROTATION (an inertial night sky: turn
/// or fly and the stars hold their world directions). Windowed-only; not `ReferenceScaffold`, so
/// it survives when the realm content loads (the sky is always there).
#[derive(Component)]
struct StarfieldRoot;

/// One background star: a world DIRECTION on the celestial sphere, an apparent SIZE, and a
/// spectral TIER (index into [`STAR_TIERS`]).
struct Star {
    dir: Vec3,
    size: f32,
    tier: usize,
}

/// Seed-stable ambient starfield: a uniform all-sky population plus a denser, tilted Milky-Way
/// band. Pure (`SplitMix64`) so the sky is byte-identical every session and on every machine.
fn generate_starfield(seed: u64) -> Vec<Star> {
    let mut rng = vd_core::rng::SplitMix64::new(seed);
    let mut stars = Vec::with_capacity(STAR_COUNT_UNIFORM + STAR_COUNT_BAND);

    // Uniform all-sky field (z uniform in [-1,1], azimuth uniform → uniform on the sphere).
    for _ in 0..STAR_COUNT_UNIFORM {
        let y = 2.0 * rng.next_f64() as f32 - 1.0;
        let phi = std::f32::consts::TAU * rng.next_f64() as f32;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let dir = Vec3::new(r * phi.cos(), y, r * phi.sin());
        stars.push(make_star(dir, &mut rng, 1.0));
    }

    // Denser, tilted Milky-Way band: longitude uniform around a tilted plane, latitude a cubic
    // draw (concentrates near 0 → a thin band). Band stars run a touch larger/brighter.
    let n = GALACTIC_NORMAL.normalize();
    let u_axis = n.any_orthonormal_vector();
    let v_axis = n.cross(u_axis);
    for _ in 0..STAR_COUNT_BAND {
        let lon = std::f32::consts::TAU * rng.next_f64() as f32;
        let t = 2.0 * rng.next_f64() as f32 - 1.0;
        let lat = STAR_BAND_HALF_ANGLE * t * t * t;
        let in_plane = lon.cos() * u_axis + lon.sin() * v_axis;
        let dir = (lat.cos() * in_plane + lat.sin() * n).normalize();
        stars.push(make_star(dir, &mut rng, 1.15));
    }
    stars
}

/// Assign a star its spectral tier (one weighted draw) and apparent size (a squared draw → most
/// stars faint, a few bright), scaled by `size_boost` (the band population runs slightly brighter).
fn make_star(dir: Vec3, rng: &mut vd_core::rng::SplitMix64, size_boost: f32) -> Star {
    let x = rng.next_f64() as f32;
    let tier = STAR_TIER_CUM
        .iter()
        .position(|&c| x <= c)
        .unwrap_or(STAR_TIERS.len() - 1);
    let m = rng.next_f64() as f32;
    let size = (STAR_SIZE_MIN + (STAR_SIZE_MAX - STAR_SIZE_MIN) * m * m) * size_boost;
    Star { dir, size, tier }
}

/// Spawn the ambient starfield: one shared unit sphere (scaled per star), one shared unlit
/// emissive material per spectral tier, all parented to a [`StarfieldRoot`] so the whole sky
/// follows the camera as one entity. Windowed-only (the capture path keeps its deterministic
/// box-framing scene untouched).
fn setup_starfield(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let unit = meshes.add(Sphere::new(1.0));
    let tier_mats: Vec<Handle<StandardMaterial>> = STAR_TIERS
        .iter()
        .map(|(base, emis)| {
            materials.add(StandardMaterial {
                base_color: Color::srgb(base[0], base[1], base[2]),
                emissive: LinearRgba::rgb(emis[0], emis[1], emis[2]),
                unlit: true,
                ..default()
            })
        })
        .collect();

    let stars = generate_starfield(STARFIELD_SEED);
    commands
        .spawn((StarfieldRoot, Transform::default(), Visibility::default()))
        .with_children(|parent| {
            for star in &stars {
                parent.spawn((
                    Mesh3d(unit.clone()),
                    MeshMaterial3d(tier_mats[star.tier].clone()),
                    Transform::from_translation(star.dir * STAR_SPHERE_RADIUS)
                        .with_scale(Vec3::splat(star.size)),
                ));
            }
        });
}

/// Keep the starfield centered on the camera (position only; identity rotation) so it reads as an
/// inertial sky at infinity. A one-frame lag is invisible at the star radius, so no ordering is
/// needed against the camera update.
fn follow_starfield(
    cam: Query<&Transform, (With<FollowCam>, Without<StarfieldRoot>)>,
    mut field: Query<&mut Transform, (With<StarfieldRoot>, Without<FollowCam>)>,
) {
    let Some(cam_tf) = cam.iter().next() else {
        return;
    };
    let pos = cam_tf.translation;
    for mut tf in &mut field {
        tf.translation = pos;
    }
}

/// Keyboard → held movement (resent only on change; latest-wins on the wire); mouse →
/// the LOCAL camera turn (immediate) AND a `Look` delta to the server. Both ride the
/// shared mailbox — byte-identical to a `vdctl` injection.
fn input_system(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<AccumulatedMouseMotion>,
    net: Res<Net>,
    mut camera: ResMut<CameraState>,
) {
    let movement = MovementKeys {
        forward: keys.pressed(KeyCode::KeyW),
        back: keys.pressed(KeyCode::KeyS),
        left: keys.pressed(KeyCode::KeyA),
        right: keys.pressed(KeyCode::KeyD),
        up: keys.pressed(KeyCode::Space),
        down: keys.pressed(KeyCode::ShiftLeft),
    };
    if movement != camera.last_movement {
        if net.input.try_send(movement.move_action()).is_err() {
            net.dropped.fetch_add(1, Ordering::Relaxed);
        }
        camera.last_movement = movement;
    }

    let delta = mouse.delta;
    if delta != Vec2::ZERO {
        // The ONE shared mapping (pixels → look delta); the camera and the server apply
        // the SAME values, so the local view and the delivered orient agree.
        if let InputAction::Look(look) = mouse_look(delta.x, delta.y) {
            camera
                .cam
                .apply_look(f64::from(look[0]), f64::from(look[1]));
            if net.input.try_send(InputAction::Look(look)).is_err() {
                net.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// First-person pointer discipline (windowed only): Esc RELEASES the cursor (so the human
/// can click away or close the window) and a left-click RE-GRABS it (locked + hidden). The
/// window starts locked (set in `WindowPlugin`), so mouse-look has unbounded delta from the
/// first frame and never sticks at the window edge. `Single` silently skips the system if the
/// window is gone (shutdown) — no panic.
fn cursor_grab(
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let mut cursor = cursor.into_inner();
    if keys.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    } else if buttons.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
}

/// Sync the dot entities to the delivered+interpolated snapshot (spawn/move/despawn) and
/// place the first-person camera at the own entity's eye.
#[allow(clippy::too_many_arguments)]
fn sync_world(
    net: Res<Net>,
    camera: Res<CameraState>,
    assets: Res<DotAssets>,
    mut dots: ResMut<DotEntities>,
    mut commands: Commands,
    mut dot_tf: Query<&mut Transform, With<Dot>>,
    mut cam_tf: Query<&mut Transform, (With<FollowCam>, Without<Dot>)>,
) {
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let own = snap.own_entity();
    let rendered = snap.rendered(now_s);

    let mut seen: BTreeSet<EntityId> = BTreeSet::new();
    let mut own_world: Option<DVec3> = None;
    for (id, _sub, pose) in &rendered {
        seen.insert(*id);
        let world = snap.world_pos(pose); // A5 — pure range-reduction against the server-told render origin
        if Some(*id) == own {
            own_world = Some(world);
        }
        match dots.0.get(id) {
            // Existing dot: move it (available from the frame after it was spawned).
            Some(&entity) => {
                if let Ok(mut transform) = dot_tf.get_mut(entity) {
                    transform.translation = world.as_vec3();
                }
            }
            // New dot: spawn it with the right material (own highlighted).
            None => {
                let material = if Some(*id) == own {
                    assets.own.clone()
                } else {
                    assets.other.clone()
                };
                let entity = commands
                    .spawn((
                        Mesh3d(assets.mesh.clone()),
                        MeshMaterial3d(material),
                        Transform::from_translation(world.as_vec3()),
                        Dot,
                    ))
                    .id();
                dots.0.insert(*id, entity);
            }
        }
    }
    // Despawn dots that are no longer delivered.
    dots.0.retain(|id, entity| {
        if seen.contains(id) {
            true
        } else {
            commands.entity(*entity).despawn();
            false
        }
    });

    // First-person camera at the own entity's eye, looking along the LOCAL camera basis.
    if let Some(mut transform) = cam_tf.iter_mut().next()
        && let Some(own_pos) = own_world
    {
        let eye = camera.cam.eye(own_pos);
        let target = eye + camera.cam.forward();
        *transform = Transform::from_translation(eye.as_vec3())
            .looking_at(target.as_vec3(), camera.cam.up.as_vec3());
    }
}

/// Sync the translucent realm-box meshes to the boot-loaded [`RenderSnapshot`] scene (Visual
/// Crossing Playground V3) — the SIBLING path to `sync_world`'s dots. For each [`RealmBox`] the
/// scene carries, place it at the world position of its realm frame's ORIGIN composed through the
/// ONE `DeliveredView::world_pos` chokepoint (so a nested/hull-borne box lands correctly, without a
/// second composition path), lower it to [`MeshPrim`] VERTICES (H4: the renderer consumes vertices,
/// NEVER a shape variant), and spawn a translucent [`StandardMaterial`] volume. Straight-line glue:
/// every geometry/color/placement decision is a Tier-A call (`to_render_prims`, `world_pos`); this
/// only builds Bevy `Mesh`/`Transform`/material handles and spawns/despawns to match the scene.
fn sync_realm_boxes(
    net: Res<Net>,
    mut boxes: ResMut<RealmBoxEntities>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
    mut box_tf: Query<&mut Transform, With<RealmBoxMarker>>,
) {
    let snap = net.snapshot.load();
    let mut seen: BTreeSet<RealmId> = BTreeSet::new();
    for (realm, rbox) in snap.scene().iter() {
        seen.insert(realm);
        // The box's OWN position, reduced ONCE against the server-told render origin through the ONE
        // chokepoint. Slice 5: this used to FABRICATE a zero pose purely to extract `-pin` from
        // `world_pos`, then add a centre whose coarse half had already been discarded — the client
        // inventing an input to a server-authoritative seam in order to do arithmetic it should not
        // be doing. The box carries its full position now, so there is one reduction and no compose.
        let draw_center = rbox.draw_center(snap.origin());
        // Lower to render primitives (VERTICES) at that drawn centre — no shape branch here.
        let prims = to_render_prims(rbox, draw_center);
        match boxes.0.get(&realm) {
            // Existing box: the geometry is fixed (config, not delivered state) through P3, so only
            // the transform can move (a hull-borne box at P8). Update its translation.
            Some(&entity) => {
                if let Ok(mut transform) = box_tf.get_mut(entity)
                    && let Some(prim) = prims.first()
                {
                    transform.translation = Vec3::from_array(prim.transform.translation);
                }
            }
            // New box: build the translucent mesh + material once and spawn it.
            None => {
                if let Some(entity) =
                    spawn_realm_box(&prims, &mut meshes, &mut materials, &mut commands)
                {
                    boxes.0.insert(realm, entity);
                }
            }
        }
    }
    // Despawn boxes no longer in the scene (empty through P3, but the seam supports dynamic scenes).
    boxes.0.retain(|realm, entity| {
        if seen.contains(realm) {
            true
        } else {
            commands.entity(*entity).despawn();
            false
        }
    });
}

/// Frame the OFFSCREEN capture camera on the whole realm-box scene (capture-only, V3): when a
/// `boxes.json` scene is loaded, point the camera at `fit_camera_to_scene` so EVERY box lands in the
/// readback — the deterministic capture camera the pixel proof (`render_boxes_smoke`) reconstructs
/// to project the box's screen AABB. Runs AFTER `sync_world` so, when a scene is present, the
/// box-framing view WINS over the follow-the-dot camera (a loaded scene means "show the boxes"). An
/// empty scene leaves the follow camera untouched (the existing behaviour). Tier-A math
/// (`fit_camera_to_scene`); this only applies the returned pose to the Bevy transform.
fn frame_scene_camera(net: Res<Net>, mut cam_tf: Query<&mut Transform, With<FollowCam>>) {
    let snap = net.snapshot.load();
    // Framed in the SAME render space the boxes are drawn in — reduced against the server-told
    // render origin (slice 5). The pixel proof reconstructs this camera with the origin it reads
    // back from the client, so the two agree by construction instead of by both assuming zero.
    let Some(cam) = vd_client_harness::camera::fit_camera_to_scene(
        snap.scene(),
        snap.origin(),
        CAPTURE_W as usize,
        CAPTURE_H as usize,
    ) else {
        return; // empty scene (or degenerate viewport) → keep the follow camera
    };
    if let Some(mut transform) = cam_tf.iter_mut().next() {
        *transform = Transform::from_translation(cam.eye.as_vec3())
            .looking_at(cam.target.as_vec3(), cam.up.as_vec3());
    }
}

/// Spawn one realm box from its lowered [`MeshPrim`]s (today exactly one per box): a Bevy `Mesh`
/// built from the prim VERTICES + a TRANSLUCENT [`StandardMaterial`] (`AlphaMode::Blend`,
/// DOUBLE-SIDED (`cull_mode: None`) so the shell renders from BOTH sides, `unlit` so the color is
/// legible regardless of lighting). Double-sided is the seamless mandate ("the moon does NOT
/// disappear when you enter the building"): the CONTAINER realm you are inside (e.g. the System, the
/// Galaxy) stays visible as the faint shell AROUND you, never culled to black — you see its far wall
/// through the near wall (honest translucency) plus every child box + the starfield beyond. An
/// earlier build back-face-culled these so a container you entered vanished from inside; that culling
/// was the bug the user hit, not a feature. Returns `None` if the box lowered to no prim (defensive).
fn spawn_realm_box(
    prims: &[MeshPrim],
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    commands: &mut Commands,
) -> Option<Entity> {
    let prim = prims.first()?;
    let mesh = meshes.add(mesh_from_prim(prim));
    let [r, g, b, a] = prim.color_rgba;
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba(r, g, b, a),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let entity = commands
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_translation(Vec3::from_array(prim.transform.translation))
                .with_scale(Vec3::from_array(prim.transform.scale)),
            RealmBoxMarker,
        ))
        .id();
    Some(entity)
}

/// Build a Bevy `Mesh` from a [`MeshPrim`]'s vertex buffer — a non-indexed triangle list of
/// positions + normals (the LOCAL unit geometry; the entity `Transform` scale/translation places
/// it). PURE glue: no shape branch, no logic — exactly the H4 seam (P4 greedy quads slot in here
/// unchanged, more vertices through the same path).
fn mesh_from_prim(prim: &MeshPrim) -> Mesh {
    let positions: Vec<[f32; 3]> = prim.vertices.iter().map(|v| v.pos).collect();
    let normals: Vec<[f32; 3]> = prim.vertices.iter().map(|v| v.normal).collect();
    Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
}

/// The windowed HUD: draw into the PRIMARY egui context (the 0.39 multipass idiom —
/// registered in `EguiPrimaryContextPass`, context via `EguiContexts::ctx_mut()` which
/// returns a `Result`, so this system returns `Result`).
fn hud_primary(mut contexts: EguiContexts, net: Res<Net>) -> Result {
    draw_hud(contexts.ctx_mut()?, &net);
    Ok(())
}

/// Draw the player-stats HUD from the delivered snapshot (wire truth) into an egui
/// context — a floating top-left overlay. SHARED by the windowed primary-context pass and
/// the headless capture multipass (so the captured overlay is byte-for-byte the same HUD).
fn draw_hud(ctx: &egui::Context, net: &Net) {
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let rendered = snap.rendered(now_s);
    let own = snap.own_entity();
    let pos = rendered
        .iter()
        .find(|(id, _, _)| Some(*id) == own)
        .map_or_else(
            || "—".to_owned(),
            |(_, _, pose)| format!("{:.1}, {:.1}, {:.1}", pose.pos.x, pose.pos.y, pose.pos.z),
        );
    let location = snap.location().unwrap_or_else(|| "—".to_owned());
    let entity = own.map(|e| e.to_string()).unwrap_or_else(|| "—".to_owned());
    egui::Area::new(egui::Id::new("vd_hud"))
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
        .show(ctx, |ui| {
            ui.label("VOXELDUST — dev client");
            ui.label(format!("status:   {}", phase_label(snap.phase())));
            ui.label(format!("location: {location}"));
            ui.label(format!("entity:   {entity}"));
            ui.label(format!("position: {pos}"));
            ui.label(format!("visible:  {}", rendered.len()));
        });
}

/// Close the window when the core loop has stopped — a gateway `Close` drove the session
/// to `Closed`, or the worker thread died/panicked (its drop-guard flips `core_alive`).
/// The worker->window half of the bidirectional shutdown, so a server-initiated
/// disconnect tears the window down instead of leaving a frozen zombie window.
fn exit_when_core_stops(net: Res<Net>, mut exit: MessageWriter<AppExit>) {
    if !net.core_alive.load(Ordering::Relaxed) {
        exit.write(AppExit::Success);
    }
}

/// A player-facing label for the client lifecycle phase.
fn phase_label(phase: ClientPhase) -> &'static str {
    match phase {
        ClientPhase::Connecting => "connecting",
        ClientPhase::AwaitingWelcome => "authenticating",
        ClientPhase::AwaitingSubscription => "subscribing",
        ClientPhase::Active => "live",
        ClientPhase::Closed => "closed",
    }
}

// ===================== headless offscreen capture (Slice-3 T5) =====================
// The recipe is the validated spike (docs/design/spikes/bevy-readback/): render the scene
// + the egui HUD into a RenderTarget::Image, copy that texture to a CPU buffer via a
// render-graph node each frame, and write a PNG when the dev-control listener asks. NO
// window/display — this is the AGENT's eyes (HR6). egui composites into the captured image
// via the multipass schedule (NOT the Screenshot component, which drops egui — issue #16689).

/// The custom egui pass for the offscreen camera's (non-primary) context.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct OffscreenEguiPass;

/// The capture-request channel from the dev-control listener (crossbeam → Send+Sync Resource).
#[derive(Resource)]
struct CaptureChannel(Receiver<CaptureJob>);

/// Capture run state: the output dir + frame/shot counters.
#[derive(Resource)]
struct CaptureCfg {
    runs_dir: PathBuf,
    frame: u32,
    shot: u64,
}

/// The offscreen render-target image handle (main world), read by the PNG writer.
#[derive(Resource)]
struct RenderTargetImage(Handle<Image>);

/// The render-world → main-world readback channel (crossbeam: a Bevy Resource needs Sync,
/// which `std::sync::mpsc::Receiver` is not).
#[derive(Resource, Deref)]
struct MainWorldReceiver(crossbeam_channel::Receiver<Vec<u8>>);
#[derive(Resource, Deref)]
struct RenderWorldSender(crossbeam_channel::Sender<Vec<u8>>);

/// The headless capture app (no window): renders the scene + egui HUD into an Image, reads
/// it back each frame, writes a PNG on a dev-control request.
fn run_capture(handles: RenderHandles) {
    tracing::info!("headless capture client starting ({CAPTURE_W}x{CAPTURE_H})");
    let captures = handles
        .captures
        .expect("Capture mode requires a capture channel");
    App::new()
        .insert_resource(ClearColor(Color::srgb(
            CLEAR_SRGB[0],
            CLEAR_SRGB[1],
            CLEAR_SRGB[2],
        )))
        .insert_resource(Net {
            snapshot: handles.snapshot,
            input: handles.input,
            dropped: handles.dropped,
            core_alive: handles.core_alive,
            started_at: handles.started_at,
        })
        .insert_resource(CameraState {
            cam: FollowCamera::new(DVec3::Y),
            last_movement: MovementKeys::default(),
        })
        .insert_resource(CaptureChannel(captures))
        .insert_resource(CaptureCfg {
            runs_dir: handles.runs_dir,
            frame: 0,
            shot: 0,
        })
        .init_resource::<DotEntities>()
        .init_resource::<RealmBoxEntities>()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                // No window/display server (headless): the runner drives frames instead.
                .disable::<WinitPlugin>(),
        )
        .add_plugins(EguiPlugin::default())
        .add_plugins(ImageCopyPlugin)
        .add_plugins(ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_secs_f64(1.0 / 60.0),
        ))
        // Headless: no window ⇒ no primary egui context; the offscreen camera owns its own.
        .add_systems(PreStartup, disable_primary_egui_context)
        .add_systems(Startup, setup_capture)
        .add_systems(OffscreenEguiPass, hud_offscreen)
        .add_systems(
            Update,
            (
                sync_world,
                sync_realm_boxes,
                // AFTER sync_world so, with a scene loaded, the box-framing view overrides the
                // follow-the-dot camera (chain: the box camera is the last word on the transform).
                frame_scene_camera.after(sync_world),
                serve_captures,
                exit_when_core_stops,
            ),
        )
        .run();
}

/// Headless: stop bevy_egui auto-creating a primary (window) context — the offscreen
/// camera's `EguiMultipassSchedule` context is the only one.
fn disable_primary_egui_context(mut settings: ResMut<EguiGlobalSettings>) {
    settings.auto_create_primary_context = false;
}

/// Capture setup: the offscreen camera (renders to an Image, egui composited via the
/// multipass schedule) + the readback copier + the SHARED world.
fn setup_capture(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
) {
    let size = Extent3d {
        width: CAPTURE_W,
        height: CAPTURE_H,
        ..default()
    };
    let mut image =
        Image::new_target_texture(CAPTURE_W, CAPTURE_H, TextureFormat::bevy_default(), None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let handle = images.add(image);
    commands.insert_resource(RenderTargetImage(handle.clone()));
    commands.spawn(ImageCopier::new(handle.clone(), size, &render_device));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.6, 0.0).looking_at(Vec3::NEG_Z, Vec3::Y),
        // In Bevy 0.18 RenderTarget is a SEPARATE component (not a Camera field).
        RenderTarget::Image(handle.into()),
        // bevy_egui creates+manages a (non-primary) EguiContext on this entity and renders
        // its passes INTO this camera's image, so the readback composites scene + HUD.
        EguiMultipassSchedule::new(OffscreenEguiPass),
        FollowCam,
    ));
    setup_world(&mut commands, &mut meshes, &mut materials);
}

/// The offscreen HUD: draw into the (non-primary) egui context bound to the Image camera.
fn hud_offscreen(mut ctx: Single<&mut EguiContext, Without<PrimaryEguiContext>>, net: Res<Net>) {
    draw_hud(ctx.get_mut(), &net);
}

/// Serve at most one capture per frame: keep the freshest readback, and once a job is
/// pending AND a warm frame exists, strip the row-padding, write the PNG, reply.
#[allow(clippy::too_many_arguments)] // a Bevy system: all params are injected resources
fn serve_captures(
    mut cfg: ResMut<CaptureCfg>,
    chan: Res<CaptureChannel>,
    receiver: Res<MainWorldReceiver>,
    target: Res<RenderTargetImage>,
    images: Res<Assets<Image>>,
    net: Res<Net>,
    mut latest: Local<Option<Vec<u8>>>,
    mut pending: Local<Option<CaptureJob>>,
) {
    cfg.frame += 1;
    while let Ok(data) = receiver.try_recv() {
        *latest = Some(data);
    }
    if pending.is_none()
        && let Ok(job) = chan.0.try_recv()
    {
        *pending = Some(job);
    }
    if cfg.frame < CAPTURE_PRE_ROLL || pending.is_none() {
        return;
    }
    let Some(bytes) = latest.clone() else {
        return; // no readback frame yet — keep the job pending for a later frame
    };
    let Some(job) = pending.take() else {
        return;
    };
    let shot = cfg.shot;
    // Organize captures by kind: screenshots under `shots/`, record frames under `frames/`
    // (the manifest stores these run-relative paths). The requester usually supplies the
    // stem (a screenshot label, or `<base>-NNNN` per record frame); the shot counter is the
    // fallback.
    let fallback = match job.kind {
        CaptureKind::Screenshot => format!("shot-{shot:04}"),
        CaptureKind::Frame => format!("frame-{shot:04}"),
    };
    let stem = job.label.clone().unwrap_or(fallback);
    // The ONE Tier-A derivation of a capture's on-disk identity: kind → shots/|frames/, the
    // agent-supplied stem SANITIZED (a slashed/`..` label is contained to one component, so
    // it can never escape the run dir or desync the PNG↔state pairing).
    let rel = capture_rel_path(job.kind, &stem);
    let path = cfg.runs_dir.join(&rel);
    // Sample the alignment from the render snapshot AT serve time (not a dev-side
    // post-roundtrip poll), so the manifest tick identifies the captured world.
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let result = write_capture_png(&path, &target, &images, &bytes).map(|()| CaptureResult {
        path: path.display().to_string(),
        rel_path: rel,
        freshest_tick: snap.freshest_tick(),
        cursor: snap.cursor(now_s),
    });
    match &result {
        Ok(r) => {
            cfg.shot += 1;
            tracing::info!(path = %r.path, "capture written");
        }
        Err(e) => tracing::warn!(error = %e, "capture failed"),
    }
    let _ = job.reply.send(result);
}

/// Strip the 256-byte row padding from the raw readback bytes and encode a PNG to `path`.
fn write_capture_png(
    path: &std::path::Path,
    target: &RenderTargetImage,
    images: &Assets<Image>,
    raw: &[u8],
) -> Result<(), String> {
    let img = images.get(&target.0).ok_or("offscreen image missing")?;
    let pixel_size = img
        .texture_descriptor
        .format
        .pixel_size()
        .map_err(|e| e.to_string())?;
    let row_bytes = CAPTURE_W as usize * pixel_size;
    let aligned = RenderDevice::align_copy_bytes_per_row(row_bytes);
    let unpadded: Vec<u8> = if row_bytes == aligned {
        raw.to_vec()
    } else {
        raw.chunks(aligned)
            .take(CAPTURE_H as usize)
            .flat_map(|row| &row[..row_bytes.min(row.len())])
            .copied()
            .collect()
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut copy = img.clone();
    copy.data = Some(unpadded);
    let dynamic = copy.try_into_dynamic().map_err(|e| e.to_string())?;
    dynamic.to_rgba8().save(path).map_err(|e| e.to_string())?;
    Ok(())
}

// ---- render-world image copy (ported verbatim-ish from the bevy headless_renderer
// ---- example via the spike; .unwrap()/.expect()/panic! replaced by guards for -D warnings).
struct ImageCopyPlugin;
impl Plugin for ImageCopyPlugin {
    fn build(&self, app: &mut App) {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let render_app = app
            .insert_resource(MainWorldReceiver(receiver))
            .sub_app_mut(RenderApp);
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(ImageCopyLabel, ImageCopyDriver);
        graph.add_node_edge(bevy::render::graph::CameraDriverLabel, ImageCopyLabel);
        render_app
            .insert_resource(RenderWorldSender(sender))
            .add_systems(ExtractSchedule, image_copy_extract)
            .add_systems(
                Render,
                receive_image_from_buffer.after(RenderSystems::Render),
            );
    }
}

#[derive(Clone, Default, Resource, Deref, DerefMut)]
struct ImageCopiers(Vec<ImageCopier>);

#[derive(Clone, Component)]
struct ImageCopier {
    buffer: Buffer,
    enabled: Arc<AtomicBool>,
    src_image: Handle<Image>,
}

impl ImageCopier {
    fn new(src_image: Handle<Image>, size: Extent3d, render_device: &RenderDevice) -> ImageCopier {
        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(size.width as usize) * 4;
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: padded_bytes_per_row as u64 * size.height as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ImageCopier {
            buffer,
            src_image,
            enabled: Arc::new(AtomicBool::new(true)),
        }
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

fn image_copy_extract(mut commands: Commands, image_copiers: Extract<Query<&ImageCopier>>) {
    commands.insert_resource(ImageCopiers(image_copiers.iter().cloned().collect()));
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, RenderLabel)]
struct ImageCopyLabel;

#[derive(Default)]
struct ImageCopyDriver;

impl render_graph::Node for ImageCopyDriver {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(image_copiers) = world.get_resource::<ImageCopiers>() else {
            return Ok(());
        };
        let Some(gpu_images) =
            world.get_resource::<RenderAssets<bevy::render::texture::GpuImage>>()
        else {
            return Ok(());
        };
        for image_copier in image_copiers.iter() {
            if !image_copier.enabled() {
                continue;
            }
            let Some(src_image) = gpu_images.get(&image_copier.src_image) else {
                continue;
            };
            let mut encoder = render_context
                .render_device()
                .create_command_encoder(&CommandEncoderDescriptor::default());
            let block_dimensions = src_image.texture_format.block_dimensions();
            let Some(block_size) = src_image.texture_format.block_copy_size(None) else {
                continue;
            };
            let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
                (src_image.size.width as usize / block_dimensions.0 as usize) * block_size as usize,
            );
            let Some(bytes_per_row) = std::num::NonZero::<u32>::new(padded_bytes_per_row as u32)
            else {
                continue;
            };
            encoder.copy_texture_to_buffer(
                src_image.texture.as_image_copy(),
                TexelCopyBufferInfo {
                    buffer: &image_copier.buffer,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row.into()),
                        rows_per_image: None,
                    },
                },
                src_image.size,
            );
            let Some(render_queue) = world.get_resource::<RenderQueue>() else {
                continue;
            };
            render_queue.submit(std::iter::once(encoder.finish()));
        }
        Ok(())
    }
}

fn receive_image_from_buffer(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    sender: Res<RenderWorldSender>,
) {
    for image_copier in image_copiers.0.iter() {
        if !image_copier.enabled() {
            continue;
        }
        let buffer_slice = image_copier.buffer.slice(..);
        let (tx, rx) = crossbeam_channel::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        if render_device.poll(PollType::wait_indefinitely()).is_err() {
            continue;
        }
        if !matches!(rx.recv(), Ok(Ok(()))) {
            continue;
        }
        let _ = sender.send(buffer_slice.get_mapped_range().to_vec());
        image_copier.buffer.unmap();
    }
}
