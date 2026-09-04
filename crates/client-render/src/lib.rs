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
use vd_client::realm_scene::{
    BodyKind, MARKER_CLASS_SRGB, MeshPrim, PrimTransform, RealmBox, Vertex, point_sprite_vertices,
    to_render_prims,
};
use vd_client::render_snapshot::RenderSnapshot;
use vd_client_harness::camera::FollowCamera;
use vd_client_harness::capture::capture_rel_path;
use vd_client_harness::input_map::{self, MovementKeys, mouse_look};
use vd_client_harness::manifest::CaptureKind;
use vd_core::EntityId;
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_devproto::InputAction;
// S11's star sprite: the custom material's surface. Spelled out rather than glob-imported so a reader
// can see which crate each name comes from — these span four of Bevy's sub-crates.
use bevy::mesh::{MeshVertexAttribute, MeshVertexBufferLayoutRef, VertexFormat};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

// ---- render tuning (named consts; no inline magic numbers) ----------------------
const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;
/// The scene clear color (sRGB) — the cleared-background of BOTH the windowed and the
/// headless-capture cameras (single-sourced so the two can never drift, and so the
/// G-RENDER-SMOKE content check measures content against the one true background). Deep-space
/// BLACK: every point of light is a REAL streamed realm, so the gaps between them are empty space
/// and must read as such (owner ruling 2026-08-20 — "No unreachable star sky spheres please!").
const CLEAR_SRGB: [f32; 3] = [0.0, 0.0, 0.0];
/// The dot marker's BASE world radius (m). `pub` so the pixel gates size the dot's projected
/// rectangle from the SAME base the renderer draws — through the one shared
/// `vd_client_harness::camera::marker_world_radius` floor (see `sync_world`'s marker scale).
///
/// TAKEN FROM Tier-A (`vd_client::realm_scene::POINT_SOURCE_BASE_RADIUS_M`), never restated: the
/// avatar dot and a unit-luminosity star marker are the SAME point-source ladder (window lane
/// Slice D), and two spellings of one convention is how they would drift.
pub const DOT_RADIUS: f32 = vd_client::realm_scene::POINT_SOURCE_BASE_RADIUS_M as f32;
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
    /// THE STARS ON SCREEN (`DevState::stars_drawn`, owner ruling 2026-09-02 R9 step 1): the
    /// window writes how many points of light its star cloud holds, the core thread reads it
    /// into the diagnosis surface. Zero while no cloud is drawn. The one instrument that can
    /// say "the sky is black" while the catalogue is held in full.
    pub stars_drawn: Arc<AtomicU64>,
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
    /// PILOT VIEW (Capture mode only; window lane Slice D): render the capture from the AVATAR'S
    /// EYE along its DELIVERED FACING instead of the scene-fitting diagnostic framing.
    ///
    /// `false` (the default) keeps every existing box gate byte-identical: the offscreen camera
    /// frames the union of the whole drawn scene so every box lands in the readback. `true` is
    /// what the warp acceptance needs — the union framing barely moves on a flight between two
    /// fixed ring positions, so a system you fly toward cannot grow on screen, and the warp is a
    /// statement about what the PILOT sees. The camera itself is Tier-A
    /// (`vd_client_harness::camera::pilot_capture_camera`), so the pixel gate reconstructs exactly
    /// the camera the renderer used from the delivered pose it already reads.
    pub pilot_view: bool,
}

/// The capture framing choice, held as a Bevy resource so both camera systems read one value.
#[derive(Resource)]
struct CaptureView {
    pilot: bool,
}

/// The bin's handles, held as a Bevy resource (read by every system).
#[derive(Resource)]
struct Net {
    snapshot: Arc<ArcSwap<RenderSnapshot>>,
    input: SyncSender<InputAction>,
    dropped: Arc<AtomicU64>,
    stars_drawn: Arc<AtomicU64>,
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
    /// THROWAWAY (test instrument): the selected throttle tier, `0..=THROTTLE_TIERS`. `]` raises it,
    /// `[` lowers it. It starts at the top because full throttle is the DESIGNED travel speed — the
    /// realm's own width in three minutes — and the tiers below it exist for close manoeuvring.
    throttle_tier: u8,
    /// THROWAWAY: the eased commanded magnitude the wire actually carries (see `THROTTLE_EASE_TAU_S`).
    throttle_now: f32,
    /// THROWAWAY: which end of the rating the ten tiers cover (owner, 2026-09-02). `X` swaps it.
    /// See [`input_map::ThrottleMode`].
    throttle_mode: input_map::ThrottleMode,
    /// ★ THE REALM THE PICTURE LAST STOOD IN (owner, 2026-09-02: the facing crosses with the body and
    /// the view follows it). When it changes — a scene swap — the camera takes its yaw and pitch from
    /// the DELIVERED facing instead of keeping a local number that now lives in another realm's frame.
    /// Without this a hull that has turned makes the view jump by the hull's rotation at boarding.
    /// `None` until the first avatar is delivered, which is also a swap: the server's facing is the truth.
    last_origin: Option<RealmId>,
    /// ★ WHICH VIEW THE PLAYER IS IN (D-MOVE-2; owner 2026-09-01). See [`CameraMode`].
    mode: CameraMode,
    /// ★ THE CAMERA LOCKED TO THE REALM'S OWN AXES (owner, 2026-09-02, at the controls of a hull):
    /// `L` toggles it. Locked, the eye looks along the realm's own forward (its −Z) and the mouse
    /// turns the REALM alone — a pilot's look is already the stick's turn on the wire, so applying
    /// it to the camera as well turned the eye twice for one motion and the camera flew around the
    /// hull. Unlocked, the mouse turns the camera as it always did (a walker looks around). THE
    /// PLAYER CHOOSES, with a key: nothing here asks what kind of realm holds them, and when a
    /// SEAT exists the seat will say who is at the controls.
    locked: bool,
}

/// ★ THE TWO VIEWS, AND WHY THIS IS A MODE RATHER THAN A DISTANCE (owner ruling 2026-09-01).
///
/// A distance alone cannot express what changes between them, because more than the eye moves:
///
/// | | first person | third person |
/// |---|---|---|
/// | the eye | in the chair | behind and above the hull |
/// | the pilot | you ARE them | not drawn — you see the hull instead |
/// | the instruments | a cockpit panel | drawn AROUND the ship, as a strategy game draws them |
///
/// The owner's words: *"if we are flying the ship now and we switch to the third-person view, we
/// don't see the pilot in the chair, but we see the ship from outside, and if we distance far enough
/// we would see the HUD around the ship."* None of that is a number.
///
/// **AND THE WHEEL IS NOT FREE.** In the game the wheel belongs to some other mechanic, and camera
/// distance is one MODIFIED use of it. So the mode must not depend on the wheel existing — the wheel
/// adjusts a distance the mode already has.
#[derive(Clone, Copy, Debug, PartialEq)]
enum CameraMode {
    /// The eye sits at the avatar. The view every existing pixel gate was written against.
    FirstPerson,
    /// The eye sits behind and above what it follows, so the hull and its facing are visible.
    ///
    /// It carries its own distance because the distance means nothing in the other mode — a mode that
    /// stores a field it never reads is the shape somebody later reads by mistake.
    ThirdPerson {
        /// How far back the eye sits, in metres. It must clear the hull, or the camera sits inside the
        /// box and sees its inner faces.
        distance_m: f64,
    },
}

impl CameraMode {
    /// How far back and how far up this mode puts the eye. First person is `(0, 0)`, which returns
    /// EXACTLY the first-person eye — so the walking view and every gate on it stay untouched.
    fn chase(self) -> (f64, f64) {
        match self {
            CameraMode::FirstPerson => (0.0, 0.0),
            // The lift is a share of the distance, so the tilt stays consistent however far out the
            // eye goes. Pulling back without lifting would put the hull's own tail between the eye and
            // everything ahead of it.
            CameraMode::ThirdPerson { distance_m } => (distance_m, distance_m * CHASE_LIFT_SHARE),
        }
    }
}

/// How much the eye lifts, as a share of how far back it sits — the "back-top" view the owner asked
/// for. A quarter is a gentle downward tilt that keeps the hull and what is ahead of it both in frame.
const CHASE_LIFT_SHARE: f64 = 0.25;

#[cfg(test)]
mod camera_mode_tests {
    use super::{CHASE_LIFT_SHARE, CHASE_START_M, CameraMode};

    #[test]
    fn first_person_puts_the_eye_exactly_where_it_always_was() {
        // The walking view and every pixel gate written against it must be untouched by the addition
        // of a second mode.
        assert_eq!(CameraMode::FirstPerson.chase(), (0.0, 0.0));
    }

    #[test]
    fn third_person_steps_back_and_lifts_a_share_of_the_step() {
        let (back, lift) = CameraMode::ThirdPerson { distance_m: 60.0 }.chase();
        assert!((back - 60.0).abs() < 1e-12);
        assert!((lift - 60.0 * CHASE_LIFT_SHARE).abs() < 1e-12);
    }

    #[test]
    fn the_tilt_stays_the_same_however_far_out_the_eye_goes() {
        // The lift is a SHARE, not a fixed height. A fixed height would flatten the view to nothing at
        // long range, and the hull's own tail would sit between the eye and everything ahead of it.
        let near = CameraMode::ThirdPerson { distance_m: 20.0 }.chase();
        let far = CameraMode::ThirdPerson { distance_m: 400.0 }.chase();
        assert!(((near.1 / near.0) - (far.1 / far.0)).abs() < 1e-12);
    }

    #[test]
    fn stepping_out_starts_clear_of_a_small_hull() {
        // The starting distance must put the eye OUTSIDE a hull, or switching the view shows the
        // inside faces of a box and reads as a rendering fault.
        let (back, _) = CameraMode::ThirdPerson {
            distance_m: CHASE_START_M,
        }
        .chase();
        assert!(back > 20.0, "clear of a twenty-metre ship: {back}");
    }
}

/// Where the eye starts when a player first steps out to third person: clear of a small hull, near
/// enough to read its facing at a glance.
const CHASE_START_M: f64 = 60.0;

/// The map from a delivered entity to its spawned Bevy dot entity.
#[derive(Resource, Default)]
struct DotEntities(BTreeMap<EntityId, Entity>);

/// The map from a realm to its spawned body entity AND the lawful author that drew it — the
/// SIBLING of [`DotEntities`] (the binary-render rule: the realm-box render is a DISTINCT path,
/// never bolted onto the dot path). Keyed by [`RealmId`] for a deterministic spawn/despawn order.
///
/// The stored [`BodyKind`] is what makes THE HANDOVER (window lane §2.8) mechanical: a realm whose
/// bag flips marker⇒self-look (it woke and states its own outline) or self-look⇒marker (it tore
/// down and its parent's ever-present point of light resumes) despawns and respawns IN THE SAME
/// FRAME, so exactly one of {marker, body} is ever on screen — never zero, never both.
#[derive(Resource, Default)]
struct RealmBoxEntities(BTreeMap<RealmId, (Entity, BodyKind)>);

/// Marker: a rendered realm box (a translucent colored volume).
#[derive(Component)]
struct RealmBoxMarker;

/// How far inside the far plane the sky sits. A star ON the plane would flicker as the local scene's
/// own extent moves the plane frame by frame.
const SKY_DEPTH_MARGIN: f64 = 0.9;
/// The sky depth used when the camera is not perspective — a defensive value, never the live path.
const DEFAULT_SKY_FAR_M: f32 = 1.0e9;

/// Marker: THE star cloud (S11) — one entity for the whole galaxy.
#[derive(Component)]
struct StarPointMarker;

/// The custom vertex attributes the star sprite needs, beside `Mesh::ATTRIBUTE_POSITION`.
///
/// The position holds the star's CENTRE, repeated for all four corners, so Bevy still computes a
/// correct bounding box for the cloud. The corner sign is what the vertex shader expands.
const ATTRIBUTE_STAR_CORNER: MeshVertexAttribute =
    MeshVertexAttribute::new("StarCorner", 0x5741_0001, VertexFormat::Float32x2);
const ATTRIBUTE_STAR_COLOR: MeshVertexAttribute =
    MeshVertexAttribute::new("StarColor", 0x5741_0002, VertexFormat::Float32x4);
const ATTRIBUTE_STAR_BASE_R: MeshVertexAttribute =
    MeshVertexAttribute::new("StarBaseRadius", 0x5741_0003, VertexFormat::Float32);

/// THE STAR SPRITE MATERIAL (S11): the apparent-size floor's constants, and nothing else.
///
/// ★ EVERY VALUE IS READ FROM TIER-A AT BIND TIME. The floor is `marker_world_radius`, which the pixel
/// gates also derive their asserted rectangle from. A copy of those numbers typed into the WGSL would
/// let the GPU draw one size while the gate asserts another — and each would look correct alone.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct StarSkyMaterial {
    #[uniform(0)]
    params: StarSkyParams,
}

#[derive(ShaderType, Debug, Clone)]
struct StarSkyParams {
    /// (Kept for the uniform's shape; the STAR path no longer reads it — a star is sized by its own
    /// flux now, not by the shared presence floor. The floor still governs realm markers.)
    min_apparent_radius_px: f32,
    tan_half_fov: f32,
    viewport_h_px: f32,
    /// Where the sky sits in depth (m) — just inside the camera's own far plane. The star's true
    /// direction is kept; only this depth slot is fixed. See `star_sky.wgsl`.
    sky_radius_m: f32,
    /// ★ THE POINT-SOURCE LAW'S NUMBERS, from Tier-A's `StarTuning` — never typed here. The law lives
    /// in `vd_client::realm_scene::star_draw`, where it is unit-tested; the shader transliterates it.
    flux_gain: f32,
    response_exponent: f32,
    halo_sigma_px: f32,
    halo_weight: f32,
    core_sigma_px: f32,
    min_crop_px: f32,
    cull_level: f32,
}

impl Material for StarSkyMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://vd_client_render/star_sky.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://vd_client_render/star_sky.wgsl".into()
    }
    /// ★ ADDITIVE, BECAUSE LIGHT ADDS (2026-08-29). This was `Blend`, where a nearer sprite REPLACES
    /// what is behind it — so two stars along one line of sight showed only the nearer, and a dense
    /// arm looked no brighter than a sparse one.
    ///
    /// Bevy maps `Add` to premultiplied-alpha blending, whose colour term is `src + dst·(1 - src.a)`.
    /// The fragment stage emits premultiplied colour with alpha ZERO, so that reduces exactly to
    /// `src + dst` — true addition. The Milky Way's glow is the sum of stars too faint to separate,
    /// and this is the mechanism that produces it rather than painting it.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<StarSkyMaterial>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            ATTRIBUTE_STAR_CORNER.at_shader_location(1),
            ATTRIBUTE_STAR_COLOR.at_shader_location(2),
            ATTRIBUTE_STAR_BASE_R.at_shader_location(3),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

/// THE STAR CLOUD ON SCREEN — built ONCE per catalogue and never rebuilt (owner ruling 2026-09-02
/// R1: *"Galaxy is always visible, we just don't repopulate it, as it doesn't move"*).
///
/// Its vertices are metres from ONE reference cell of the galaxy's lattice, chosen when the cloud is
/// first built. Where the observer stands enters only as the cloud entity's TRANSFORM, recomputed
/// every frame from the SKY ANCHOR the gateway states on the realm lane — the origin realm placed
/// in the galaxy's frame — and the eye's own offset inside that realm. Parallax is that transform
/// moving. A crossing changes nothing here: the anchor names the new origin, the transform follows.
///
/// ★ WHAT THIS REPLACED. The cloud used to be keyed on the observer's OWN STAR SYSTEM, deleted on
/// every crossing and rebuilt by searching the catalogue for the realm the observer stood in. The
/// catalogue names star systems only, so a hull, a planet, a station, an area and the galaxy itself
/// all failed the search, and the sky went black for good — MEASURED 2026-09-01 inside a hull.
#[derive(Resource, Default)]
struct DrawnSky {
    /// `(generation, reference cell)` of the cloud on screen, or `None` while nothing is drawn.
    shown: Option<(u64, vd_core::glam::I64Vec3)>,
    /// THE one cloud entity, held so a new catalogue (or a rebase) can replace it.
    cloud: Option<Entity>,
    /// The material handle, held so the per-frame camera uniforms can be refreshed without touching
    /// the star data.
    material: Option<Handle<StarSkyMaterial>>,
}

/// The SHARED marker point-sprite assets: ONE unit vertex buffer for every point of light plus one
/// unlit emissive material per Morgan-Keenan spectral class (`vd_client::realm_scene::
/// MARKER_CLASS_SRGB`), built once at setup. This is what "a dot never costs a mesh" means in the
/// renderer (window lane §2.14's tier-0 rung): a sleeping star adds an entity and a transform, no
/// mesh build and no material allocation, however many of them the sky holds.
#[derive(Resource)]
struct MarkerAssets {
    mesh: Handle<Mesh>,
    class: Vec<Handle<StandardMaterial>>,
}

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

/// ★ THE RENDER FRAME (S5 — D-LOOK-3): the eye's position in the composed picture's own frame,
/// decided ONCE per frame by [`place_camera`] and read by everything that places geometry.
///
/// The camera itself sits at the RENDER ORIGIN (`Transform.translation == Vec3::ZERO`) carrying
/// only a rotation; every drawn thing is placed at `world - eye`, subtracted in `f64` and narrowed
/// to `f32` only at the end. That is the whole cure for D-LOOK-3: on THE world an absolute render
/// position is quantized to 16 m at a planet and 1.3e8 m at the star gap, and — worse — Bevy's
/// `Transform::looking_at` subtracts in `f32`, so a target one metre ahead of a `1e11` m eye
/// rounded to the eye itself and the camera silently fell back to facing world `-Z`.
///
/// `Default` (a zero eye) is the honest pre-login state: nothing is drawn yet.
#[derive(Resource, Default)]
struct RenderEye {
    eye: DVec3,
    /// ★ THE EYE ON THE LATTICE, WITH ITS UNIT (slice S4), so a drawn position is reduced against it
    /// BEFORE anything is flattened.
    ///
    /// The flattened `eye` above is kept for the paths that legitimately need a metre value (the camera
    /// transform itself), but every position drawn RELATIVE to the eye now reduces against this one.
    /// The difference is not cosmetic: at the star placement radius each independent flatten rounds by
    /// a quarter of a metre, so two ships flying in convoy drew a separation that was wrong and that
    /// flickered as they moved.
    ///
    /// `None` until the first frame that has a delivered avatar to stand at. Deliberately an option
    /// rather than a default: there is no default unit, and a position reduced against a guessed one is
    /// wrong by the ratio between the guess and the truth — which is the whole defect this slice exists
    /// to remove.
    eye_lattice: Option<(vd_core::pose::LatticePos, vd_core::pose::Tier)>,
    /// The camera's vertical FOV (rad) and viewport rows, sampled from the live camera — the two
    /// facts every apparent-size and near-plane derivation needs, read once per frame.
    view: Option<(f64, f64)>,
}

/// Run the client renderer. BLOCKS until exit; the bin MUST call this on the MAIN thread
/// (winit/the runner need it) with the core loop on a separate thread. Dispatches on the
/// mode: a real window (human) or headless offscreen capture (the agent's eyes).
/// EMBED THE STAR SHADER IN THE BINARY (S11).
///
/// This repo has no `assets/` directory and no asset path of any kind. Loading the WGSL from disk
/// would make a headless capture — and a deployed client — depend on finding a file beside the
/// binary. Embedding it removes that failure mode entirely: the shader ships inside the executable.
struct StarSkyShaderPlugin;

impl Plugin for StarSkyShaderPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "star_sky.wgsl");
    }
}

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
            stars_drawn: handles.stars_drawn,
            core_alive: handles.core_alive,
            started_at: handles.started_at,
        })
        .insert_resource(CameraState {
            cam: FollowCamera::new(DVec3::Y),
            last_movement: MovementKeys::default(),
            throttle_tier: input_map::THROTTLE_TIERS,
            throttle_now: 0.0,
            throttle_mode: input_map::ThrottleMode::default(),
            last_origin: None,
            // A player starts in the chair; the mode key steps them out. See `CameraMode`.
            mode: CameraMode::FirstPerson,
            locked: false,
        })
        // A window is always the human's own first-person view; the pilot-view switch exists only
        // for the HEADLESS capture path (there is no scene-fitting framing here to decline).
        .insert_resource(CaptureView { pilot: false })
        .init_resource::<DotEntities>()
        .init_resource::<RealmBoxEntities>()
        .init_resource::<DrawnSky>() // S11: which sky is on screen, and around which system
        .init_resource::<RenderEye>()
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
        // ★ AFTER `DefaultPlugins`, and that ORDER IS LOAD-BEARING (S11). `embedded_asset!` writes into
        // `EmbeddedAssetRegistry`, which `AssetPlugin` creates — registering earlier panics at startup
        // with "resource does not exist", which is a runtime failure no compile can catch. Measured:
        // the capture client exited 101 before its listener came up.
        .add_plugins((
            StarSkyShaderPlugin,
            MaterialPlugin::<StarSkyMaterial>::default(),
        ))
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                // ★ S5 — THE RENDER FRAME IS DECIDED FIRST, then everything is placed relative to
                // it, then the depth planes are read off what was drawn. The chain IS the law: a
                // body placed against last frame's eye is a body drawn in the wrong place.
                place_camera,
                (
                    sync_world,
                    // Sync the realm spheres, then despawn the stub-world scaffolding once they exist
                    // (chained so the despawn sees the just-spawned boxes the same frame — no scaffold
                    // flash in space).
                    (sync_realm_boxes, despawn_reference_scaffold).chain(),
                    // S11: the galaxy. Independent of the box/dot lanes — it re-spawns only on a
                    // crossing or a new catalogue, so it is not chained into their per-frame work.
                    sync_star_sky,
                    place_reference_scaffold,
                ),
                derive_camera_planes,
            )
                .chain(),
        )
        .add_systems(Update, (input_system, cursor_grab, exit_when_core_stops))
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
    // First-person follow camera — ORIENTED each frame by `place_camera` and always AT THE RENDER
    // ORIGIN (S5's camera-relative flatten). Its near/far are rewritten every frame by
    // `derive_camera_planes` from what is actually drawn; the spawned pair is only the first
    // frame's placeholder, never a declared reach.
    let mut cam = commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection::default()),
        Transform::from_translation(Vec3::ZERO).looking_at(Vec3::NEG_Z, Vec3::Y),
        FollowCam,
    ));
    // ★ BLOOM IS OFF, AND ONE ENV VAR AWAY (owner, 2026-08-29: "skip bloom, but keep the possibility
    // to quickly enable for a test").
    //
    // The glow belongs in the SPRITE, not in a post-process: the two most astronomy-accurate
    // renderers ship without bloom, and a star's halo is a property of the eye rather than of the
    // camera. Bloom also spreads light from everything else in frame, which at a quarter of a million
    // point sources is a picture-wide wash rather than a per-star glow.
    //
    // But it is a LOOK decision, and a look decision is judged by looking. `VD_STAR_BLOOM=1` turns it
    // on for a run, alongside the HDR pass it needs — no rebuild, no code change, and off again by
    // unsetting it.
    if std::env::var("VD_STAR_BLOOM").is_ok_and(|v| v == "1") {
        cam.insert((
            bevy::post_process::bloom::Bloom::NATURAL,
            bevy::render::view::Hdr,
        ));
        tracing::info!("VD_STAR_BLOOM=1 — bloom and HDR enabled for this run");
    }
    setup_world(&mut commands, &mut meshes, &mut materials);
}

// (the windowed camera above; the shared world below — capture's offscreen camera lives
// in `run_capture` and reuses `setup_world`.)

/// Marker for the empty-stub-world reference scaffolding (the ground plate + the landmark pillars) — a
/// P1.5 MOTION reference, despawned by [`despawn_reference_scaffold`] the moment real realm content loads.
/// Carries its WORLD ANCHOR so [`place_reference_scaffold`] can re-express it in the render frame each
/// frame (S5): the camera is the origin now, so a fixture pinned at the composed origin has to move.
#[derive(Component)]
struct ReferenceScaffold(DVec3);

/// Despawn the reference scaffolding ONCE real realm content (any [`RealmBox`]) is present. The visual
/// universe — and the eventual game — ALWAYS has realm spheres, so the ground/pillars belong ONLY to the
/// empty P1.5 stub world (`render_smoke`, launched with NO `--realm-boxes`), never the SPACE view. Seamless:
/// they vanish as the boot realm scene loads. Registered in BOTH schedules: the capture path used to keep
/// them under a stale justification ("render_boxes_smoke checks the box, not these") — in fact the slab sat
/// at the origin of every realm-scene capture, inside every projected rectangle the pixel gates counted, so
/// the gates could pass with no realm box drawn at all (batch review). `render_smoke`'s empty-world content
/// floor is untouched: with no realm content this is a no-op.
fn place_reference_scaffold(
    render_eye: Res<RenderEye>,
    mut scaffold: Query<(&ReferenceScaffold, &mut Transform)>,
) {
    for (anchor, mut transform) in &mut scaffold {
        transform.translation =
            vd_client_harness::camera::eye_relative(anchor.0, render_eye.eye).as_vec3();
    }
}

/// Everything the picture actually draws, in the render frame — the dots, the realm bodies and the
/// stub scaffolding. Named once so the system signature and the reader agree on it.
type DrawnQuery<'w, 's> = Query<
    'w,
    's,
    &'static Transform,
    Or<(With<Dot>, With<RealmBoxMarker>, With<ReferenceScaffold>)>,
>;

/// ★ S5 — THE DERIVED DEPTH PLANES (D-LOOK-3). Read the near/far pair off WHAT WAS JUST DRAWN and
/// write it onto the camera, every frame. Runs last: every drawn transform is already in the render
/// frame, so a translation's own LENGTH is the view distance and its scale is the world radius (all
/// drawn meshes are UNIT geometry placed by scale — `to_render_prims`, the point sprite, the dot).
///
/// The law and its derivation live in `vd_client_harness::camera::depth_planes`; this is glue.
fn derive_camera_planes(
    render_eye: Res<RenderEye>,
    drawn: DrawnQuery,
    mut cam: Query<&mut Projection, With<FollowCam>>,
) {
    let Some((fov_y, viewport_h)) = render_eye.view else {
        return;
    };
    let subjects: Vec<(f64, f64)> = drawn
        .iter()
        .map(|t| {
            (
                f64::from(t.translation.length()),
                f64::from(t.scale.max_element()),
            )
        })
        .collect();
    let planes = vd_client_harness::camera::depth_planes(&subjects, fov_y, viewport_h);
    if let Some(mut projection) = cam.iter_mut().next()
        && let Projection::Perspective(p) = &mut *projection
    {
        p.near = planes.near as f32;
        p.far = planes.far as f32;
    }
}

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
        ReferenceScaffold(DVec3::new(0.0, -0.1, 0.0)),
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
            ReferenceScaffold(DVec3::new(
                f64::from(LANDMARK_RING_RADIUS * angle.cos()),
                f64::from(LANDMARK_HEIGHT * 0.5),
                f64::from(LANDMARK_RING_RADIUS * angle.sin()),
            )),
        ));
    }

    // Shared dot assets (built once, reused per spawned dot).
    commands.insert_resource(DotAssets {
        // A UNIT sphere, scaled to the dot's WORLD RADIUS by `sync_world` — the same shape every
        // other drawn mesh has (`to_render_prims`, the point sprite). One convention: a drawn
        // entity's transform SCALE is its world radius, which is what lets `derive_camera_planes`
        // read the picture's own extents straight off the transforms it just wrote.
        mesh: meshes.add(Sphere::new(1.0)),
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
    // The SHARED marker point-sprite assets (window lane Slice D): one unit vertex buffer + one
    // unlit emissive material per spectral class. `unlit` so a point of light reads at its stated
    // colour regardless of where the key light is — a star is not lit, it emits — which also makes
    // the pixel gates' local probes deterministic.
    commands.insert_resource(MarkerAssets {
        mesh: meshes.add(mesh_from_vertices(&point_sprite_vertices())),
        class: MARKER_CLASS_SRGB
            .iter()
            .map(|&[r, g, b]| {
                materials.add(StandardMaterial {
                    base_color: Color::srgb(r, g, b),
                    emissive: LinearRgba::rgb(r, g, b),
                    unlit: true,
                    ..default()
                })
            })
            .collect(),
    });
    // The stats HUD is drawn each frame by `hud_primary` via egui (no entity to spawn).
}

/// Keyboard → held movement (resent only on change; latest-wins on the wire); mouse →
/// the LOCAL camera turn (immediate) AND a `Look` delta to the server. Both ride the
/// shared mailbox — byte-identical to a `vdctl` injection.
fn input_system(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mouse: Res<AccumulatedMouseMotion>,
    net: Res<Net>,
    mut camera: ResMut<CameraState>,
) {
    // ★ THE VIEW KEY SWITCHES THE MODE (D-MOVE-2, owner 2026-09-01). It is a MODE and not a distance
    // because more than the eye changes between them: in the chair you see a cockpit and you ARE the
    // pilot; outside you see the hull, the pilot is not drawn, and the instruments belong around the
    // ship rather than in front of your face.
    //
    // **THE PLAYER CHOOSES, so nothing here asks what kind of realm holds them.** Switching
    // automatically when the player's frame was a ship's would be a KIND TEST — wrong here for the
    // same reason it is wrong everywhere else: a station with engines, or anything else somebody
    // builds later, would need adding to a list. A key needs no list.
    if keys.just_pressed(KeyCode::KeyV) {
        camera.mode = match camera.mode {
            CameraMode::FirstPerson => CameraMode::ThirdPerson {
                distance_m: CHASE_START_M,
            },
            CameraMode::ThirdPerson { .. } => CameraMode::FirstPerson,
        };
        tracing::info!(mode = ?camera.mode, "view switched");
    }
    // THROWAWAY: `X` swaps the throttle between its two ladders — normal for a star system, warp for
    // the run between stars (owner, 2026-09-02). The tier is kept; only what it means changes.
    if keys.just_pressed(KeyCode::KeyX) {
        camera.throttle_mode = camera.throttle_mode.toggled();
        tracing::info!(mode = ?camera.throttle_mode, tier = camera.throttle_tier, "throttle mode");
    }
    if keys.just_pressed(KeyCode::KeyL) {
        camera.locked = !camera.locked;
        if camera.locked {
            // ★ LOCK = FACE THE NOSE (owner, 2026-09-02). The push follows the pilot's facing, so a
            // locked view along the nose must be a FACING along the nose, or the hull would fly where
            // the pilot last looked rather than where the locked eye looks. The one look that turns
            // the delivered facing onto the realm's own forward is sent — input, like any look — and
            // the local camera goes there at once. While locked the mouse sends nothing, so the
            // facing and the eye stay together.
            let snap = net.snapshot.load();
            let own = snap.own_entity();
            let facing = snap
                .rendered(net.started_at.elapsed().as_secs_f64())
                .into_iter()
                .find(|(id, _, _)| Some(*id) == own)
                .map(|(_, _, pose)| pose.orient);
            if let Some(facing) = facing {
                let (yaw, pitch) = vd_core::kinematics::yaw_pitch_from_orient(facing);
                #[allow(clippy::cast_possible_truncation)] // a look delta is f32 on the wire
                let to_nose = InputAction::Look([-yaw as f32, -pitch as f32]);
                if net.input.try_send(to_nose).is_err() {
                    net.dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
            camera.cam.yaw = 0.0;
            camera.cam.pitch = 0.0;
        }
        tracing::info!(
            locked = camera.locked,
            "camera lock: locked = the pilot faces the nose and the eye holds the realm's own axes"
        );
    }
    // THROWAWAY (test instrument): `]` raises the throttle tier, `[` lowers it. The tier is announced
    // in the log because there is no HUD yet to show it.
    let raised = u8::from(keys.just_pressed(KeyCode::BracketRight));
    let lowered = u8::from(keys.just_pressed(KeyCode::BracketLeft));
    let tier = (camera.throttle_tier + raised)
        .min(input_map::THROTTLE_TIERS)
        .saturating_sub(lowered);
    if tier != camera.throttle_tier {
        camera.throttle_tier = tier;
        tracing::info!(tier, of = input_map::THROTTLE_TIERS, "throttle");
    }

    let mut movement = MovementKeys {
        forward: keys.pressed(KeyCode::KeyW),
        back: keys.pressed(KeyCode::KeyS),
        left: keys.pressed(KeyCode::KeyA),
        right: keys.pressed(KeyCode::KeyD),
        up: keys.pressed(KeyCode::Space),
        down: keys.pressed(KeyCode::ControlLeft),
        throttle: 0.0,
    };
    // Ease the COMMANDED magnitude toward the tier while a key is held, and toward zero when none is —
    // so the ship gathers way and loses it instead of starting and stopping dead. `held` is derived from
    // the keys we just read, so the ramp cannot disagree with what is pressed.
    let held = movement.axes().iter().any(|a| *a != 0.0)
        || movement.forward
        || movement.back
        || movement.left
        || movement.right
        || movement.up
        || movement.down;
    let target =
        f32::from(held) * input_map::throttle_magnitude(camera.throttle_mode, camera.throttle_tier);
    let alpha = 1.0 - (-time.delta_secs() / input_map::THROTTLE_EASE_TAU_S).exp();
    camera.throttle_now += (target - camera.throttle_now) * alpha;
    // Snap the last sliver so a release reaches a true standstill rather than creeping forever.
    let settled = camera.throttle_now < 1.0e-3 && target == 0.0;
    camera.throttle_now *= f32::from(!settled);
    movement.throttle = camera.throttle_now;

    // The eased magnitude changes every frame, so resend whenever the wire value moved at all.
    if movement != camera.last_movement {
        if net.input.try_send(movement.move_action()).is_err() {
            net.dropped.fetch_add(1, Ordering::Relaxed);
        }
        camera.last_movement = movement;
    }

    // THROWAWAY (the temporary control seam, owner 2026-09-02): `E` raises the nose of a realm that
    // flies on a stick and `Q` lowers it. They ride the wire as two action bits; the shard binds the
    // bits (see `vd_core::controls`). A release is sent after a press so a key that went down and up
    // inside one frame ends CLEAR — a bit left set would hold the nose up until the next press.
    for (key, index) in [
        (KeyCode::KeyQ, vd_core::controls::PILOT_PITCH_DOWN_INDEX),
        (KeyCode::KeyE, vd_core::controls::PILOT_PITCH_UP_INDEX),
    ] {
        for pressed in [true, false] {
            let edge = if pressed {
                keys.just_pressed(key)
            } else {
                keys.just_released(key)
            };
            if edge
                && let Some(action) = input_map::action(index, pressed)
                && net.input.try_send(action).is_err()
            {
                net.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    let delta = mouse.delta;
    if delta != Vec2::ZERO {
        // The ONE shared mapping (pixels → look delta); the camera and the server apply
        // the SAME values, so the local view and the delivered orient agree. On a hull the shard
        // reads NO turn from the look (owner, 2026-09-02): the mouse is the pilot's own eyes.
        // Locked: the mouse sends NOTHING and turns nothing — the facing stays on the nose and the
        // eye with it (see the `L` key above). Unlocked: the look turns the body on the server and
        // the local camera at once, the same values, so the view and the delivered facing agree.
        if !camera.locked
            && let InputAction::Look(look) = mouse_look(delta.x, delta.y)
        {
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

/// ★ S5 — THE ONE CAMERA DECISION (D-LOOK-3), taken BEFORE anything is placed.
///
/// Places the follow camera and publishes the frame's [`RenderEye`]. Two things changed here and
/// both were measured defects:
///
/// 1. The camera now sits at the RENDER ORIGIN carrying only a rotation, and that rotation is
///    built in `f64` by `vd_client_harness::camera::look_rotation`. Bevy's `looking_at` subtracts
///    `target - translation` in `f32`; at a `1e8`–`1e15` m eye a one-metre-ahead target rounds to
///    the eye itself, the direction comes out ZERO and `look_to` falls back to `Dir3::NEG_Z` — the
///    camera faced world `-Z` regardless of the pilot, which is exactly why correctly-composed
///    subjects came back as EMPTY readback rectangles.
/// 2. The eye is decided BEFORE the dots and bodies are placed, so the apparent-size floor and the
///    eye-relative flatten both read THIS frame's eye instead of last frame's (the old O(0.1%)
///    scale skew is gone with the ordering, not absorbed by a tolerance).
fn place_camera(
    net: Res<Net>,
    mut camera: ResMut<CameraState>,
    view: Res<CaptureView>,
    mut eye: ResMut<RenderEye>,
    mut cam: Query<(&mut Transform, &Camera, &Projection), With<FollowCam>>,
) {
    let Some((mut transform, cam_props, projection)) = cam.iter_mut().next() else {
        return; // no camera yet (the first frame)
    };
    eye.view = Some(camera_view(cam_props, projection));

    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let own = snap.own_entity();
    let rendered = snap.rendered(now_s);
    let Some((_, _, own_pose)) = rendered.iter().find(|(id, _, _)| Some(*id) == own) else {
        return; // no delivered avatar yet — nothing to stand at
    };
    let own_world = snap.world_pos(own_pose);
    // ★ A SCENE SWAP RE-EXPRESSES THE VIEW (owner, 2026-09-02). The local yaw and pitch are numbers
    // in the frame the picture is drawn in; when that frame changes they mean another direction. The
    // delivered facing is the server's own answer in the NEW frame, so the camera takes it — once,
    // at the swap — and the mouse continues from there. A walker who steps into a hull that has
    // turned keeps looking where they looked.
    let origin = snap.origin();
    if origin != camera.last_origin {
        let (yaw, pitch) = vd_core::kinematics::yaw_pitch_from_orient(own_pose.orient);
        camera.cam.yaw = yaw;
        camera.cam.pitch = pitch;
        camera.last_origin = origin;
        tracing::info!(
            ?origin,
            yaw,
            pitch,
            "scene swap: the view takes the delivered facing"
        );
    }
    // PILOT VIEW (the headless acceptance): the avatar's eye along its SERVER-DELIVERED facing,
    // through the ONE Tier-A expression the pixel gates reconstruct the camera from — so an
    // injected `LookAt` turns the avatar and the agent's eyes together. Otherwise: the LOCAL
    // first-person basis, so mouse-look turns the view immediately.
    let (eye_pos, direction, up) = if view.pilot {
        let cam = vd_client_harness::camera::pilot_capture_camera(
            own_world,
            own_pose.orient,
            CAPTURE_W as usize,
            CAPTURE_H as usize,
        );
        (cam.eye, cam.target - cam.eye, cam.up)
    } else if camera.locked {
        // ★ LOCKED TO THE REALM (owner, 2026-09-02): the picture is drawn in the origin realm's own
        // frame, so its forward is a constant −Z and its up +Y here. First person: the eye at the
        // avatar, looking along the nose. Third person: the eye behind the tail and lifted, looking
        // the same way, so the hull sits ahead of the eye and the sky turns around both.
        let (back_m, lift_m) = camera.mode.chase();
        (
            own_world + DVec3::new(0.0, lift_m, back_m),
            DVec3::NEG_Z,
            DVec3::Y,
        )
    } else {
        // ★ THIRD PERSON WHEN A CHASE DISTANCE IS SET (D-MOVE-2). Zero — the default — returns
        // exactly the first-person eye, so the walking view and every pixel gate on it are unchanged.
        //
        // A pilot needs this and a walker does not: from inside a hull you cannot see the hull, and a
        // pilot must see which way the nose points to fly at all.
        let (back_m, lift_m) = camera.mode.chase();
        let e = camera.cam.chase_eye(own_world, back_m, lift_m);
        (e, camera.cam.forward(), camera.cam.up)
    };
    eye.eye = eye_pos;
    // ★ THE EYE ON THE LATTICE (slice S4). The camera's own offset from the avatar is a SMALL local
    // displacement — a few metres — so carrying it as a lattice step is exact at any magnitude, unlike
    // the flattened `eye` above, which is a distance from the frame origin and rounds with it.
    eye.eye_lattice = Some((
        vd_core::pose::LatticePos::at(own_pose.cell, own_pose.pos)
            .translated(eye_pos - own_world, own_pose.tier),
        own_pose.tier,
    ));
    *transform = camera_transform(direction, up);
}

/// The camera's Bevy transform in the RENDER FRAME: at the origin, carrying the `f64`-built
/// rotation. The one place the S5 flatten's camera half is expressed.
fn camera_transform(direction: DVec3, up: DVec3) -> Transform {
    Transform::from_translation(Vec3::ZERO)
        .with_rotation(vd_client_harness::camera::look_rotation(direction, up).as_quat())
}

/// The camera facts every apparent-size and near-plane derivation needs: `(vertical fov, viewport
/// rows)`. Read once per frame by [`place_camera`] into [`RenderEye`].
fn camera_view(camera: &Camera, projection: &Projection) -> (f64, f64) {
    let fov_y = match projection {
        Projection::Perspective(p) => f64::from(p.fov),
        // Non-perspective projections do not occur here (both cameras declare Perspective);
        // fall back to the declared default rather than a magic number.
        _ => f64::from(PerspectiveProjection::default().fov),
    };
    let viewport_h = camera
        .physical_viewport_size()
        .map_or(f64::from(CAPTURE_H), |s| f64::from(s.y));
    (fov_y, viewport_h)
}

/// Sync the dot entities to the delivered+interpolated snapshot (spawn/move/despawn), placed in the
/// RENDER FRAME (eye-relative — see [`RenderEye`]).
#[allow(clippy::too_many_arguments)]
fn sync_world(
    net: Res<Net>,
    render_eye: Res<RenderEye>,
    assets: Res<DotAssets>,
    mut dots: ResMut<DotEntities>,
    mut commands: Commands,
    mut dot_tf: Query<&mut Transform, With<Dot>>,
) {
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let own = snap.own_entity();
    let rendered = snap.rendered(now_s);

    // THE MARKER FLOOR context (the VU marker phase; batch review): a 0.5 m dot at a scene-fitted
    // camera's ~780 m eye subtends well under a pixel, so whether it rasterized AT ALL was sampling
    // luck — no pixel gate could be dot-sensitive, and a distant player was invisible in the window
    // too. Each dot is scaled so its apparent radius never falls below the shared
    // `DOT_MIN_APPARENT_RADIUS_PX` floor (`marker_world_radius` — the SAME derivation the gates
    // size their rectangles from). S5: the eye is THIS frame's (`place_camera` ran first), so the
    // old last-frame scale skew is gone. First-person (windowed) the own dot sits at the eye ⇒ the
    // base radius stands, byte-identical to the unscaled marker.
    let marker_scale = |rel: DVec3| -> Vec3 {
        let radius_m = match render_eye.view {
            Some((fov_y, viewport_h)) => vd_client_harness::camera::marker_world_radius(
                f64::from(DOT_RADIUS),
                rel.length(),
                fov_y,
                viewport_h,
            ),
            // No camera yet (the first frame): the base radius.
            None => f64::from(DOT_RADIUS),
        };
        Vec3::splat(radius_m as f32)
    };

    let mut seen: BTreeSet<EntityId> = BTreeSet::new();
    for (id, _sub, pose) in &rendered {
        seen.insert(*id);
        let world = snap.world_pos(pose); // a passthrough: the server ships pin-space positions
        // ★ SLICE S4 — REDUCE AGAINST THE EYE ON THE LATTICE, then flatten once. Flattening both
        // halves first made each round independently, and at the star placement radius that is a
        // quarter of a metre EACH — so two ships flying in convoy drew a wrong separation that
        // flickered as they moved. `None` only before the first delivered avatar, where the flattened
        // path is all there is and nothing is being compared to anything yet.
        // ★ THE CAMERA-RELATIVE FLATTEN: subtract the eye in f64, narrow to f32 once.
        let rel = match render_eye.eye_lattice {
            Some((eye, tier)) => vd_client_harness::camera::eye_relative_lattice(
                vd_core::pose::LatticePos::at(pose.cell, pose.pos),
                eye,
                tier,
            ),
            None => vd_client_harness::camera::eye_relative(world, render_eye.eye),
        };
        match dots.0.get(id) {
            // Existing dot: move it (available from the frame after it was spawned).
            Some(&entity) => {
                if let Ok(mut transform) = dot_tf.get_mut(entity) {
                    transform.translation = rel.as_vec3();
                    transform.scale = marker_scale(rel);
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
                        Transform::from_translation(rel.as_vec3()).with_scale(marker_scale(rel)),
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
}

/// ★ THE STAR SKY (S11; owner ruling 2026-09-02 R1): ONE point cloud for the whole catalogue,
/// uploaded once and PLACED every frame by the sky anchor — never rebuilt for a crossing.
///
/// Per frame, in order: the material's camera uniforms are refreshed (the size floor is in pixels
/// and the camera can change); with no whole sky held or no anchor stated yet, nothing is drawn and
/// the drawn count says zero; otherwise the cloud is built if the catalogue changed (or, once per
/// galaxy-to-galaxy journey, if the placement has drifted past the single-precision bound — see
/// `sky_rebase_due`), and its transform is set from the anchor and the eye.
#[allow(clippy::too_many_arguments)] // a Bevy system: all params are injected resources/queries
fn sync_star_sky(
    net: Res<Net>,
    eye: Res<RenderEye>,
    mut drawn: ResMut<DrawnSky>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StarSkyMaterial>>,
    camera: Single<(&Camera, &Projection)>,
    mut clouds: Query<&mut Transform, With<StarPointMarker>>,
    mut commands: Commands,
) {
    // THE UNIFORMS ARE REFRESHED EVERY FRAME, because the size floor is measured in PIXELS and the
    // camera can change: a window resize moves `viewport_h_px`, and the projection owns the field of
    // view. The star DATA is not touched here — only these four numbers.
    let (cam, projection) = *camera;
    let (fov_y, viewport_h_px) = camera_view(cam, projection);
    // ★ THE SKY'S DEPTH COMES FROM THE CAMERA'S OWN FAR PLANE, never a constant. `derive_camera_planes`
    // sets that plane from the LOCAL scene every frame, so a literal here would be beyond it on one
    // scene and inside it on another — and a star beyond the far plane is silently clipped, which is
    // the black sky nobody can explain. The margin keeps the cloud strictly inside.
    let far_m = match projection {
        Projection::Perspective(p) => f64::from(p.far),
        _ => f64::from(DEFAULT_SKY_FAR_M),
    };
    let tuning = vd_client::realm_scene::StarTuning::default();
    let params = StarSkyParams {
        min_apparent_radius_px: vd_client_harness::camera::DOT_MIN_APPARENT_RADIUS_PX as f32,
        tan_half_fov: (fov_y * 0.5).tan() as f32,
        viewport_h_px: viewport_h_px as f32,
        sky_radius_m: (far_m * SKY_DEPTH_MARGIN) as f32,
        // THE LAW'S NUMBERS, read from Tier-A. `StarTuning::default()` is the starting point the
        // owner judges by looking; nothing here is a literal.
        flux_gain: tuning.flux_gain as f32,
        response_exponent: tuning.response_exponent as f32,
        halo_sigma_px: tuning.halo_sigma_px as f32,
        halo_weight: tuning.halo_weight as f32,
        core_sigma_px: tuning.core_sigma_px as f32,
        min_crop_px: tuning.min_crop_px as f32,
        cull_level: tuning.cull_level as f32,
    };
    if let Some(material) = drawn
        .material
        .as_ref()
        .and_then(|handle| materials.get_mut(handle))
    {
        material.params = params.clone();
    }

    let snap = net.snapshot.load();
    // ★ The anchor at the SAME render cursor the bodies are drawn from (owner 2026-09-04): the sky
    // and the bodies turn together when the hull turns.
    let now_s = net.started_at.elapsed().as_secs_f64();
    let Some((sky, anchor)) = snap.sky().zip(snap.sky_anchor_now(now_s)) else {
        // No whole sky held, or the observer chain has not reached the galaxy yet: nothing is
        // drawn — never a wrong sky — and the instrument says so.
        if let Some(entity) = drawn.cloud.take() {
            commands.entity(entity).despawn();
        }
        drawn.shown = None;
        net.stars_drawn.store(0, Ordering::Relaxed);
        return;
    };
    // The reference the cloud on screen was built from, if it is still the right cloud: the same
    // catalogue, and a placement still inside the single-precision bound.
    let kept = drawn.shown.filter(|(generation, reference)| {
        let (translation, _) =
            vd_client::render_snapshot::sky_cloud_transform(*reference, &anchor, eye.eye);
        (*generation == sky.generation) && !vd_client::render_snapshot::sky_rebase_due(translation)
    });
    let reference = match kept {
        Some((_, reference)) => reference,
        None => {
            // A NEW CATALOGUE (or a rebase): replace the cloud. Leaving the old one would draw two
            // galaxies at once.
            if let Some(entity) = drawn.cloud.take() {
                commands.entity(entity).despawn();
            }
            drawn.shown = None;
            net.stars_drawn.store(0, Ordering::Relaxed);
            let reference = anchor.pos.cell();
            let points = sky.points_from(reference);
            let cloud = vd_client::render_snapshot::StarCloud::build(&points);
            if cloud.is_empty() {
                return;
            }
            // ★ ONE MESH, ONE ENTITY, WHATEVER THE CENSUS. At 150,000 stars this is 14.4 MB of
            // vertices plus 3.6 MB of U32 indices, uploaded ONCE — against 150,000 `MeshUniform`
            // records rebuilt every frame if each star were its own entity. `U32` is forced:
            // 600,000 vertices overflow a `u16`.
            let mut mesh = Mesh::new(
                bevy::mesh::PrimitiveTopology::TriangleList,
                bevy::asset::RenderAssetUsages::default(),
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, cloud.positions.clone());
            mesh.insert_attribute(ATTRIBUTE_STAR_CORNER, cloud.corners.clone());
            mesh.insert_attribute(ATTRIBUTE_STAR_COLOR, cloud.colors.clone());
            mesh.insert_attribute(ATTRIBUTE_STAR_BASE_R, cloud.base_radius_m.clone());
            mesh.insert_indices(bevy::mesh::Indices::U32(cloud.indices.clone()));
            let material = materials.add(StarSkyMaterial { params });
            let (translation, rotation) =
                vd_client::render_snapshot::sky_cloud_transform(reference, &anchor, eye.eye);
            drawn.cloud = Some(
                commands
                    .spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(material.clone()),
                        // THE PLACEMENT: the anchor and the eye, folded in f64 and narrowed once.
                        // The shader applies this model transform; nothing else moves the sky.
                        Transform {
                            translation: translation.as_vec3(),
                            rotation: rotation.as_quat(),
                            scale: Vec3::ONE,
                        },
                        // ★ NO FRUSTUM CULLING (S11). The cloud's bounding box spans the whole
                        // galaxy — half-extents of about 4.6e18 m — and Bevy's visibility test
                        // against a box that large, in f32, is not a test worth trusting. The camera
                        // stands INSIDE it always, so culling could only ever remove the sky by
                        // mistake, never save work.
                        bevy::camera::visibility::NoFrustumCulling,
                        StarPointMarker,
                    ))
                    .id(),
            );
            drawn.material = Some(material);
            drawn.shown = Some((sky.generation, reference));
            net.stars_drawn
                .store(points.len() as u64, Ordering::Relaxed);
            return;
        }
    };
    // THE PLACEMENT, EVERY FRAME: the anchor moved (the origin realm flew), or the eye moved inside
    // the origin. Either way the one cloud slides; near stars slide more than far ones, which is the
    // parallax — nothing draws it on purpose.
    let (translation, rotation) =
        vd_client::render_snapshot::sky_cloud_transform(reference, &anchor, eye.eye);
    for mut transform in &mut clouds {
        transform.translation = translation.as_vec3();
        transform.rotation = rotation.as_quat();
    }
}

#[allow(clippy::too_many_arguments)] // a Bevy system: all params are injected resources/queries
fn sync_realm_boxes(
    net: Res<Net>,
    render_eye: Res<RenderEye>,
    mut boxes: ResMut<RealmBoxEntities>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    markers: Res<MarkerAssets>,
    mut commands: Commands,
    mut box_tf: Query<&mut Transform, (With<RealmBoxMarker>, Without<FollowCam>)>,
) {
    // SLICE 6 S4 — ONE MOMENT. Resolve the realm boxes at the SAME display cursor `sync_world` samples
    // the entities at, so the ground and the player standing on it are drawn from one instant. This
    // used to read a scene the core thread had baked from the newest arrival, on a different schedule
    // entirely — which is what made their relative geometry jump.
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let scene = snap.scene_now(now_s);
    // The camera facts the apparent-size floor needs — THIS frame's (`place_camera` ran first),
    // the same pair `sync_world` scales the avatar dot by.
    let view = render_eye.view;
    let mut seen: BTreeSet<RealmId> = BTreeSet::new();
    for (realm, rbox) in scene.iter() {
        seen.insert(realm);
        // The box's OWN position, flattened ONCE through the ONE chokepoint. Slice 5: this used to
        // FABRICATE a zero pose purely to extract `-pin` from `world_pos`, then add a centre whose
        // coarse half had already been discarded — the client inventing an input to a
        // server-authoritative seam in order to do arithmetic it should not be doing. The box carries
        // its full position now, already measured from the realm this session stands in, and in the unit
        // the shipper stated for it — so there is nothing to subtract and no unit to pick.
        //
        // ★ S5 — THE CAMERA-RELATIVE FLATTEN: the composed centre is expressed in the RENDER FRAME
        // (eye at the origin) by an f64 subtraction, and only the RESULT is narrowed to f32. This
        // is what lets a body 2.2e15 m away be drawn at all: as an absolute f32 its position was
        // quantized to 1.3e8 m — the whole marker — and every f32 view matrix built from it lost
        // the camera's own facing (see `place_camera`).
        // ★ SLICE S4 — the box reduces against the eye ON THE LATTICE too, for the same reason: a
        // realm's drawn centre and the eye are both distances from the frame origin, and flattening
        // each before subtracting rounds them independently.
        // ★ A FAR ROW (2026-09-04): a box stated in a frame other than the one this session stands
        // in is a far realm the composer could not turn exactly — it rides in the sky's frame, and
        // it is placed from the sky anchor exactly as the star cloud is (one transform, f64,
        // narrowed once). Without an anchor it cannot be placed and is left where it was.
        let far = |anchor: &vd_core::pose::StampedPose| {
            let (translation, rotation) = vd_client::render_snapshot::sky_cloud_transform(
                rbox.center.cell(),
                anchor,
                render_eye.eye,
            );
            translation + rotation * rbox.center.offset()
        };
        let draw_center = match (render_eye.eye_lattice, snap.sky_anchor_now(now_s)) {
            (Some((_, tier)), Some(anchor)) if rbox.tier != tier => far(&anchor),
            (Some((eye, tier)), _) => {
                vd_client_harness::camera::eye_relative_lattice(rbox.center, eye, tier)
            }
            (None, _) => {
                vd_client_harness::camera::eye_relative(rbox.draw_center(), render_eye.eye)
            }
        };
        // Lower to render primitives (VERTICES) at that drawn centre — no shape branch here. THE
        // DRAW LAW's two arms are the two lawful AUTHORS, decided by the row's bag upstream in
        // Tier-A: a self-authored outline lowers through the shape tessellation; a parent-authored
        // point of light lowers to the ONE shared point sprite, sized by the apparent-size floor.
        let prims = match rbox.body {
            BodyKind::Look => to_render_prims(rbox, draw_center),
            BodyKind::Marker => marker_prims(rbox, draw_center, view),
        };
        match boxes.0.get(&realm) {
            // Existing body drawn by the SAME author: the outline geometry is fixed (config, not
            // delivered state) through P3, so only the transform can move (a hull-borne box at P8);
            // a marker's scale also breathes with its distance, so both ride the transform.
            Some(&(entity, kind)) if kind == rbox.body => {
                if let Ok(mut transform) = box_tf.get_mut(entity)
                    && let Some(prim) = prims.first()
                {
                    transform.translation = Vec3::from_array(prim.transform.translation);
                    transform.scale = Vec3::from_array(prim.transform.scale);
                    // ★ THE FACING ITS PARENT AUTHORED (D-MOVE-2). Without this a hull that turns
                    // shows the same face for ever, and a pilot has no way to see which way the nose
                    // points — which is the whole reason a ship is drawn as a box rather than a dot.
                    transform.rotation = Quat::from_array(prim.transform.rotation);
                }
            }
            // THE HANDOVER, both ways (window lane §2.8): the author changed — a sleeping child
            // woke and now states its own outline, or a departed one's look was pruned and its
            // parent's ever-present point of light resumed. Despawn and respawn IN THIS FRAME, so
            // the swap is atomic on screen: never zero drawn, never both.
            Some(&(entity, _)) => {
                commands.entity(entity).despawn();
                spawn_body(
                    realm,
                    rbox,
                    &prims,
                    &markers,
                    &mut meshes,
                    &mut materials,
                    &mut commands,
                    &mut boxes,
                );
            }
            // New body: build it once and spawn it.
            None => spawn_body(
                realm,
                rbox,
                &prims,
                &markers,
                &mut meshes,
                &mut materials,
                &mut commands,
                &mut boxes,
            ),
        }
    }
    // Despawn bodies no longer in the scene (a realm that left the drawn set entirely — neither a
    // look nor a marker statement reached the composer, so nothing may be drawn for it).
    boxes.0.retain(|realm, (entity, _)| {
        if seen.contains(realm) {
            true
        } else {
            commands.entity(*entity).despawn();
            false
        }
    });
}

/// One MARKER body's render primitive: the SHARED unit point sprite at the drawn centre, scaled
/// to its base radius — the LARGER of the luma-derived √L radius and the parent's one stated
/// extent (look_horizon.md slice 1's presence floor, `marker_base_radius_m`) — after the ONE
/// apparent-size floor (`vd_client_harness::camera::marker_world_radius` — the same expression
/// the pixel gates size their rectangles from, so the drawn footprint and the asserted rectangle
/// cannot disagree). A luma-less marker (a non-glowing subject) draws in the box's own role
/// colour at its stated extent — never nothing.
fn marker_prims(rbox: &RealmBox, draw_center: DVec3, view: Option<(f64, f64)>) -> Vec<MeshPrim> {
    let base = vd_client::realm_scene::marker_base_radius_m(
        rbox.luma,
        vd_client::realm_scene::shape_extent_m(rbox.shape),
    );
    let color_rgba = vd_client::realm_scene::marker_color_rgba(rbox.color_rgba, rbox.luma);
    // `draw_center` is ALREADY in the render frame (eye at the origin), so its own length IS the
    // view distance the apparent-size floor needs — no second eye subtraction, no second unit.
    let radius = match view {
        Some((fov_y, viewport_h)) => vd_client_harness::camera::marker_world_radius(
            base,
            draw_center.length(),
            fov_y,
            viewport_h,
        ),
        None => base,
    };
    let r = radius as f32;
    vec![MeshPrim {
        vertices: point_sprite_vertices(),
        color_rgba,
        transform: PrimTransform {
            translation: [
                draw_center.x as f32,
                draw_center.y as f32,
                draw_center.z as f32,
            ],
            scale: [r, r, r],
            // A POINT SPRITE always faces the camera, so a facing means nothing to it. Stated
            // as identity rather than left out, so the field cannot be forgotten the day a
            // sprite grows a shape.
            rotation: [0.0, 0.0, 0.0, 1.0],
        },
    }]
}

/// Spawn one realm's body and record WHICH lawful author drew it. A marker reuses the SHARED point
/// sprite mesh + its spectral-class material (no mesh build, no material allocation — §2.14's
/// tier-0 rung); an outline builds its own translucent volume.
#[allow(clippy::too_many_arguments)]
fn spawn_body(
    realm: RealmId,
    rbox: &RealmBox,
    prims: &[MeshPrim],
    markers: &MarkerAssets,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    commands: &mut Commands,
    boxes: &mut RealmBoxEntities,
) {
    let spawned = match rbox.body {
        BodyKind::Look => spawn_realm_box(prims, rbox.luma.is_some(), meshes, materials, commands),
        BodyKind::Marker => spawn_marker(prims, rbox, markers, commands),
    };
    if let Some(entity) = spawned {
        boxes.0.insert(realm, (entity, rbox.body));
    }
}

/// Spawn one MARKER point sprite from the shared assets. Returns `None` if the marker lowered to no
/// prim (defensive — `marker_prims` always emits exactly one).
fn spawn_marker(
    prims: &[MeshPrim],
    rbox: &RealmBox,
    markers: &MarkerAssets,
    commands: &mut Commands,
) -> Option<Entity> {
    let prim = prims.first()?;
    let (class_code, _) = rbox.luma.unwrap_or_default();
    let material = markers
        .class
        .get(usize::from(class_code))
        .or_else(|| markers.class.last())?;
    let entity = commands
        .spawn((
            Mesh3d(markers.mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(Vec3::from_array(prim.transform.translation))
                .with_scale(Vec3::from_array(prim.transform.scale)),
            RealmBoxMarker,
        ))
        .id();
    Some(entity)
}

/// Frame the OFFSCREEN capture camera on the whole realm-box scene (capture-only, V3): when a
/// `boxes.json` scene is loaded, point the camera at `fit_camera_to_scene` so EVERY box lands in the
/// readback — the deterministic capture camera the pixel proof (`render_boxes_smoke`) reconstructs
/// to project the box's screen AABB. Runs AFTER `sync_world` so, when a scene is present, the
/// box-framing view WINS over the follow-the-dot camera (a loaded scene means "show the boxes"). An
/// empty scene leaves the follow camera untouched (the existing behaviour). Tier-A math
/// (`fit_camera_to_scene`); this only applies the returned pose to the Bevy transform.
fn frame_scene_camera(
    net: Res<Net>,
    view: Res<CaptureView>,
    mut eye: ResMut<RenderEye>,
    mut cam_tf: Query<&mut Transform, With<FollowCam>>,
) {
    // PILOT VIEW: the scene-fitting framing is DECLINED whole — `place_camera` already placed the
    // camera at the avatar's eye along its delivered facing, and this system is the only thing
    // that would override it (it is registered `.after(place_camera)` precisely to be the last word).
    if view.pilot {
        return;
    }
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    // Framed on the SAME resolved-at-cursor scene the boxes are drawn from (slice 6 S4) — otherwise the
    // fitted frustum would silently track a different instant than the pixels inside it.
    let scene = snap.scene_now(now_s);
    // Framed in the SAME render space the boxes are drawn in. There is only one such space now — the
    // session's pin — because the server converts into it before shipping, so the pixel proof and this
    // camera agree by construction rather than by both assuming a zero origin.
    let Some(cam) = vd_client_harness::camera::fit_camera_to_scene(
        &scene,
        CAPTURE_W as usize,
        CAPTURE_H as usize,
    ) else {
        return; // empty scene (or degenerate viewport) → keep the follow camera
    };
    // ★ S5: the fitted eye is published as the RENDER FRAME's origin and the rotation is built in
    // f64 — a fitted standoff can be 1e11 m out, where `looking_at`'s f32 subtraction of two
    // like-sized coordinates loses whole kilometres of the view direction.
    if let Some(mut transform) = cam_tf.iter_mut().next() {
        // ★ MOVE BOTH HALVES OF THE EYE, OR THE PICTURE IS DRAWN FROM TWO PLACES AT ONCE.
        //
        // This used to write `eye.eye` alone. `place_camera` had already published the AVATAR's eye on
        // the lattice (`eye_lattice`), and BOTH placement systems — `sync_world` for dots and
        // `sync_realm_boxes` for bodies — read the lattice eye FIRST. So the camera turned to the
        // fitted three-quarter view while every body stayed placed relative to the avatar's standoff:
        // rotation and placement disagreed, and the scene swung off the edge of the frame.
        //
        // MEASURED: G-RENDER-SMOKE captured content_fraction 0.0177 against its 0.05 floor — five
        // times the HUD-only 0.0036 (so the scene WAS drawing) and five times below the 0.0983 this
        // exact scene recorded when the gate last passed. The reconstructed geometry put the home star
        // 33 degrees off axis, 40 px of it past the top edge.
        //
        // ★ AND THE FIX IS NOT TO LOWER THE FLOOR. 0.05 is calibrated and 0.0983 is the recorded pass;
        // moving it would bless a broken camera and retire the gate's ability to catch this ever again.
        //
        // The lattice eye is TRANSLATED rather than cleared. Clearing it would drop both readers onto
        // the flattened fallback, which is the exact path S4 replaced: two positions flattened
        // independently before subtraction round apart, which is what made a convoy's separation
        // flicker. Both eyes live in the same render frame, so this displacement is exact.
        let delta = cam.eye - eye.eye;
        if let Some((pos, tier)) = eye.eye_lattice {
            eye.eye_lattice = Some((pos.translated(delta, tier), tier));
        }
        eye.eye = cam.eye;
        *transform = camera_transform(cam.target - cam.eye, cam.up);
    }
}

/// Spawn one realm box from its lowered [`MeshPrim`]s (today exactly one per box): a Bevy `Mesh`
/// built from the prim VERTICES + a TRANSLUCENT [`StandardMaterial`] (`AlphaMode::Blend`,
/// DOUBLE-SIDED (`cull_mode: None`) so the shell renders from BOTH sides, `unlit` so the color is
/// legible regardless of lighting). Double-sided is the seamless mandate ("the moon does NOT
/// disappear when you enter the building"): the CONTAINER realm you are inside (e.g. the System, the
/// Galaxy) stays visible as the faint shell AROUND you, never culled to black — you see its far wall
/// through the near wall (honest translucency) plus every child box beyond it. An
/// earlier build back-face-culled these so a container you entered vanished from inside; that culling
/// was the bug the user hit, not a feature. Returns `None` if the box lowered to no prim (defensive).
///
/// ★ A BODY THAT STATES ITS OWN LIGHT IS OPAQUE (2026-09-04, the tenth flight: the owner saw the
/// star field's point through the star's own body). A container is a translucent volume because
/// you stand inside it and must see out; a glowing body has no interior to see through, and its
/// depth must hide the point of light the sky draws for it (owner ruling 2026-09-02 R1: the body
/// draws on top of its own point). Decided by data presence — the self-look carries `TAG_LUMA` —
/// never by kind. Example: the star of System 7 draws as a solid disc and its sky point is behind
/// it; System 7's own shell around the pilot stays translucent.
fn spawn_realm_box(
    prims: &[MeshPrim],
    lit: bool,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    commands: &mut Commands,
) -> Option<Entity> {
    let prim = prims.first()?;
    let mesh = meshes.add(mesh_from_prim(prim));
    let [r, g, b, a] = prim.color_rgba;
    let (alpha, alpha_mode) = if lit {
        (1.0, AlphaMode::Opaque)
    } else {
        (a, AlphaMode::Blend)
    };
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba(r, g, b, alpha),
        alpha_mode,
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
    mesh_from_vertices(&prim.vertices)
}

/// Build a Bevy `Mesh` from a Tier-A vertex buffer — the shared body of [`mesh_from_prim`] and the
/// one-time marker point-sprite build.
fn mesh_from_vertices(vertices: &[Vertex]) -> Mesh {
    let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.pos).collect();
    let normals: Vec<[f32; 3]> = vertices.iter().map(|v| v.normal).collect();
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
fn hud_primary(
    mut contexts: EguiContexts,
    net: Res<Net>,
    camera: Res<CameraState>,
    keys: Res<ButtonInput<KeyCode>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    draw_hud(ctx, &net);
    draw_pilot_panel(ctx, &net, &camera, &keys);
    Ok(())
}

/// THROWAWAY — THE PILOT'S READOUT (owner, 2026-09-02): what the client already holds about the realm
/// the player stands in, on the right of the window. It reads NOTHING new off the wire: the realm's
/// placement in the galaxy is the sky anchor the gateway ships for the star cloud (its velocity is
/// the same fold, so the speed is the star system's own number for the hull), and the stick is what
/// this window last sent. The hull's RATED push is the hull's own row and never crosses to a screen,
/// so the panel shows the fraction of it the throttle commands; the berth tool prints the rating.
///
/// Windowed only. The capture HUD stays byte-for-byte what the pixel gates were measured against.
///
/// Example: the pilot holds `W` at tier 10. The panel reads the speed climbing tick by tick, the
/// throttle at the whole rating, and "yaw —" until they touch `A` or `D`.
fn draw_pilot_panel(
    ctx: &egui::Context,
    net: &Net,
    camera: &CameraState,
    keys: &ButtonInput<KeyCode>,
) {
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let own = snap.own_entity();
    let in_realm = snap
        .rendered(now_s)
        .into_iter()
        .find(|(id, _, _)| Some(*id) == own)
        .map(|(_, _, pose)| snap.world_pos(&pose));
    let location = snap.location().unwrap_or_else(|| "—".to_owned());
    let axes = camera.last_movement.axes();
    let sign = |x: f32| match x.partial_cmp(&0.0) {
        Some(std::cmp::Ordering::Greater) => "+",
        Some(std::cmp::Ordering::Less) => "−",
        _ => "—",
    };
    let pitch = f32::from(keys.pressed(KeyCode::KeyE)) - f32::from(keys.pressed(KeyCode::KeyQ));
    egui::Area::new(egui::Id::new("vd_pilot"))
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 10.0))
        .show(ctx, |ui| {
            ui.label("PILOT — temporary controls");
            ui.label(format!("realm:             {location}"));
            ui.label(match in_realm {
                Some(p) => format!("in realm (m):      {:.1}, {:.1}, {:.1}", p.x, p.y, p.z),
                None => "in realm (m):      —".to_owned(),
            });
            match snap.sky_anchor() {
                Some(a) => {
                    let p = a
                        .pos
                        .delta_m(vd_core::pose::LatticePos::ORIGIN, a.frame.tier());
                    ui.label(format!(
                        "realm in galaxy:   {:.4e}, {:.4e}, {:.4e} m",
                        p.x, p.y, p.z
                    ));
                    let speed = a.vel.length();
                    ui.label(format!(
                        "speed:             {speed:.1} m/s  ({:.3e} km/s)",
                        speed / 1000.0
                    ));
                    ui.label(format!(
                        "velocity (m/s):    {:.3e}, {:.3e}, {:.3e}",
                        a.vel.x, a.vel.y, a.vel.z
                    ));
                }
                None => {
                    ui.label("realm in galaxy:   — (no placement stated yet)");
                }
            }
            ui.label(format!(
                "throttle:          {:?} (X swaps)  tier {}/{}  →  {:.2e} of the rated push",
                camera.throttle_mode,
                camera.throttle_tier,
                input_map::THROTTLE_TIERS,
                camera.throttle_now
            ));
            ui.label(format!(
                "stick:             push fwd {} {:.2e}  up {} {:.2e}  |  yaw {}  pitch {}",
                sign(axes[0]),
                axes[0].abs(),
                sign(axes[2]),
                axes[2].abs(),
                sign(-axes[1]),
                sign(pitch)
            ));
            ui.label(format!(
                "camera:            {}  ({:?})",
                if camera.locked {
                    "facing the nose, eye locked (L frees)"
                } else {
                    "free look, you fly where you look (L faces the nose)"
                },
                camera.mode
            ));
            ui.label(
                "keys: W/S push  A/D yaw  E/Q nose up/down  Space/Ctrl up/down  ]/[ tier  X warp  V view",
            );
        });
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
            // THROUGH THE ONE CHOKEPOINT, like everything that is drawn. `pose.pos` is only the sub-cell
            // remainder of a position; since the server began folding the whole-number part out, reading it
            // alone made this readout sit still while the player moved. It is the same call the geometry
            // above already goes through — the HUD was the one consumer hand-rolling its own arithmetic.
            |(_, _, pose)| {
                let w = snap.world_pos(pose);
                format!("{:.1}, {:.1}, {:.1}", w.x, w.y, w.z)
            },
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
            stars_drawn: handles.stars_drawn,
            core_alive: handles.core_alive,
            started_at: handles.started_at,
        })
        .insert_resource(CameraState {
            cam: FollowCamera::new(DVec3::Y),
            last_movement: MovementKeys::default(),
            throttle_tier: input_map::THROTTLE_TIERS,
            throttle_now: 0.0,
            throttle_mode: input_map::ThrottleMode::default(),
            last_origin: None,
            // A player starts in the chair; the mode key steps them out. See `CameraMode`.
            mode: CameraMode::FirstPerson,
            locked: false,
        })
        .insert_resource(CaptureView {
            pilot: handles.pilot_view,
        })
        .insert_resource(CaptureChannel(captures))
        .insert_resource(CaptureCfg {
            runs_dir: handles.runs_dir,
            frame: 0,
            shot: 0,
        })
        .init_resource::<DotEntities>()
        .init_resource::<RealmBoxEntities>()
        .init_resource::<DrawnSky>() // S11: which sky is on screen, and around which system
        .init_resource::<RenderEye>()
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
        // ★ AFTER `DefaultPlugins`, and that ORDER IS LOAD-BEARING (S11). `embedded_asset!` writes into
        // `EmbeddedAssetRegistry`, which `AssetPlugin` creates — registering earlier panics at startup
        // with "resource does not exist", which is a runtime failure no compile can catch. Measured:
        // the capture client exited 101 before its listener came up.
        .add_plugins((
            StarSkyShaderPlugin,
            MaterialPlugin::<StarSkyMaterial>::default(),
        ))
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
                // ★ S5 — the render frame first (`place_camera`), then the box-framing view gets the
                // last word on it (`frame_scene_camera`), then everything is placed relative to the
                // decided eye, then the depth planes are read off what was drawn.
                place_camera,
                frame_scene_camera,
                (
                    sync_world,
                    // Sync the realm boxes, then despawn the stub-world scaffolding once they exist —
                    // the SAME pair the windowed schedule runs. The capture path used to keep the
                    // ground plate + pillars forever, which parked a 500 m slab at the origin of every
                    // realm-scene capture and let the pixel gates pass on scaffold paint alone (batch
                    // review: the H2 confound). render_smoke (NO --realm-boxes) still keeps them: with
                    // no realm content the despawn is a no-op, so its content floor stands.
                    (sync_realm_boxes, despawn_reference_scaffold).chain(),
                    // S11: the galaxy. Independent of the box/dot lanes — it re-spawns only on a
                    // crossing or a new catalogue, so it is not chained into their per-frame work.
                    sync_star_sky,
                    place_reference_scaffold,
                ),
                derive_camera_planes,
            )
                .chain(),
        )
        .add_systems(Update, (serve_captures, exit_when_core_stops))
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
        // The SAME projection the windowed camera declares — parity, so the two cameras cannot
        // diverge on a projection field. Both are rewritten every frame by `derive_camera_planes`
        // (S5): Bevy builds a reverse-Z INFINITE perspective, whose depth code is `near/z` and whose
        // relative resolution is therefore constant at every range, so the planes are a statement
        // about the picture rather than a clip distance anyone has to guess.
        Projection::Perspective(PerspectiveProjection::default()),
        Transform::from_translation(Vec3::ZERO).looking_at(Vec3::NEG_Z, Vec3::Y),
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
