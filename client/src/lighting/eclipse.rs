//! Phase 6 / 6.1 — Eclipse occlusion of the directional sun.
//!
//! Two render paths, one source of truth:
//!
//!   * **Forward** (Low / Medium presets): `ChunkMaterial =
//!     ExtendedMaterial<StandardMaterial, EclipseExt>`. The extension's
//!     fragment shader (`eclipse.wgsl`) replaces Bevy's standard PBR
//!     fragment, runs `apply_pbr_lighting`, then SUBTRACTS the would-be
//!     directional-sun contribution scaled by `1 - eclipse_factor`.
//!
//!   * **Deferred** (High / Ultra: SSR enabled → `DeferredPrepass`):
//!     a fullscreen post-process pass (`eclipse_deferred`) runs after
//!     Bevy's `DeferredLightingPass` and reads the same eclipse state
//!     to apply the same surgical sun-only attenuation against the
//!     radiance buffer.
//!
//! Both paths consume the same [`EclipseState`] resource (sun position
//! + radius, plus up to [`MAX_ECLIPSE_OCCLUDERS`] occluders) — built
//! from `WorldState.bodies` once per frame in
//! [`update_eclipse_uniform`]. A small mirror system copies the
//! resource onto the chunk-material asset so the forward path keeps
//! working through Bevy's standard material pipeline; the deferred
//! pass reads the resource directly via `ExtractResource`.
//!
//! The eclipse math itself (lens-area angular-disk intersection +
//! `sun_direct_contribution` reconstruction) lives in the shared WGSL
//! module `voxeldust::eclipse_lib`, imported by both shaders. One
//! source of truth for the formula; one for the data.
//!
//! No magic numbers in lighting values: every input is either a
//! physical constant ([`R_SUN_M`]) or a server-broadcast field
//! (`body.position`, `body.radius`, `body.stellar.radius_solar`).

use bevy::asset::Asset;
use bevy::math::{UVec4, Vec4};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::{AsBindGroup, ShaderType},
};
use bevy::shader::ShaderRef;
use glam::DVec3;
use std::collections::BTreeMap;

use voxeldust_core::client_message::CelestialBodyData;
use voxeldust_core::physics_constants::R_SUN_M;

use crate::chunk::material::ChunkMaterialCache;
use crate::shard::{
    CameraWorldPos, PrimaryWorldState, SecondaryWorldStates, ShardOriginSet,
};

/// Maximum simultaneous occluders the shader iterates each frame.
///
/// Physically: a single observer in a star system has at most a
/// handful of bodies large enough + close enough to subtend any
/// meaningful angular size — the host planet, its moons, sister
/// planets in inferior orbits. Eight comfortably exceeds that for any
/// configuration we'd plausibly broadcast and keeps the uniform
/// compact (sun + 8 vec4s + count = 160 bytes).
pub const MAX_ECLIPSE_OCCLUDERS: usize = 8;

/// The `body_id` reserved for the system's primary star. Mirrors
/// `lighting::solar`'s convention (see `solar.rs:178`).
const STAR_BODY_ID: u32 = 0;

// ───── Shared uniform layout ──────────────────────────────────────────────

/// Layout matched by `voxeldust::eclipse_lib::EclipseUniform` in WGSL.
/// `ShaderType` derives the std140-aligned GPU layout; the std140
/// padding lives in the unused lanes of `count` (`.y`, `.z`, `.w`).
///
/// **Both** the forward material extension's `#[uniform(100)]` field
/// AND the deferred pass's per-frame uniform buffer are filled from
/// the same source ([`EclipseState`]) and use this exact same layout
/// to match the shared shader-side struct.
#[derive(ShaderType, Clone, Copy, Debug)]
pub struct EclipseUniform {
    /// `.xyz` = sun position camera-relative (m); `.w` = sun radius (m).
    /// `w == 0` is the sentinel for "no sun configured" — the shader
    /// short-circuits and returns `1.0` (no occlusion).
    pub sun: Vec4,
    /// `.xyz` = occluder position camera-relative (m); `.w` = radius (m).
    /// Inactive slots have `w == 0` and are skipped in the shader loop.
    pub occluders: [Vec4; MAX_ECLIPSE_OCCLUDERS],
    /// `.x` = active occluder count (≤ `MAX_ECLIPSE_OCCLUDERS`); the
    /// remaining lanes are std140 padding.
    pub count: UVec4,
}

impl EclipseUniform {
    /// All-zero state — both shaders interpret this as "no sun
    /// configured" and return `1.0` unconditionally.
    pub const INACTIVE: Self = Self {
        sun: Vec4::ZERO,
        occluders: [Vec4::ZERO; MAX_ECLIPSE_OCCLUDERS],
        count: UVec4::ZERO,
    };
}

impl Default for EclipseUniform {
    /// `Default` returns the [`INACTIVE`](Self::INACTIVE) sentinel —
    /// safe to bind on the very first frame before
    /// `update_eclipse_uniform` has run.
    fn default() -> Self {
        Self::INACTIVE
    }
}

// ───── Authoritative resource ─────────────────────────────────────────────

/// Single source of truth for the per-frame eclipse state. Built each
/// frame in main world by [`update_eclipse_uniform`]; mirrored to the
/// chunk material asset (forward path) and extracted to the render
/// world (deferred post-process pass).
#[derive(Resource, ExtractResource, Clone, Copy, Debug)]
pub struct EclipseState {
    pub uniform: EclipseUniform,
}

impl Default for EclipseState {
    fn default() -> Self {
        Self {
            uniform: EclipseUniform::INACTIVE,
        }
    }
}

// ───── Forward path: material extension ───────────────────────────────────

/// `EclipseExt` — shader-side bindings for the forward path's
/// per-fragment eclipse subtraction. The base material's
/// `StandardMaterial` bindings live in the material bind group at
/// bindings 0..N; we bind at 100 to leave headroom for
/// `StandardMaterial`'s growing surface (Bevy adds bindings between
/// minor versions).
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct EclipseExt {
    #[uniform(100)]
    pub uniform: EclipseUniform,
}

impl Default for EclipseExt {
    fn default() -> Self {
        Self {
            uniform: EclipseUniform::INACTIVE,
        }
    }
}

impl MaterialExtension for EclipseExt {
    fn fragment_shader() -> ShaderRef {
        "shaders/eclipse.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        // In deferred mode the extension fragment runs as the prepass
        // (G-buffer write); the WGSL `PREPASS_PIPELINE` branch handles
        // it unchanged. Eclipse for deferred is applied by the
        // separate `eclipse_deferred` post-process pass.
        "shaders/eclipse.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }
}

// ───── Plugin wiring ──────────────────────────────────────────────────────

/// Strong handle to `eclipse_lib.wgsl`. The lib defines
/// `#define_import_path voxeldust::eclipse_lib`; both `eclipse.wgsl`
/// (forward) and `eclipse_deferred.wgsl` (deferred) `#import` from
/// that path. Bevy's shader cache parses the `#define_import_path`
/// directive only after the file is loaded as a `Shader` asset, so
/// we hold a strong handle to keep the asset alive (otherwise the
/// dependent shaders would fail to resolve the import on first
/// pipeline build).
#[derive(Resource)]
pub struct EclipseLibShader(#[allow(dead_code)] pub Handle<bevy::shader::Shader>);

pub struct EclipsePlugin;

impl Plugin for EclipsePlugin {
    fn build(&self, app: &mut App) {
        // Eagerly load the shared eclipse-math library so its
        // `#define_import_path voxeldust::eclipse_lib` directive
        // registers before any importer compiles.
        let lib_handle: Handle<bevy::shader::Shader> = app
            .world()
            .resource::<AssetServer>()
            .load("shaders/eclipse_lib.wgsl");
        app.insert_resource(EclipseLibShader(lib_handle));

        app.init_resource::<EclipseState>()
            .add_plugins(ExtractResourcePlugin::<EclipseState>::default())
            .add_plugins(
                MaterialPlugin::<ExtendedMaterial<StandardMaterial, EclipseExt>>::default(),
            )
            // `update_eclipse_uniform` runs after shard-origin
            // rebasing so the camera-relative positions we write are
            // consistent with the chunk fragments the forward shader
            // will read on the same frame. Same ordering rule as
            // `solar.rs::update_solar_light`.
            //
            // `mirror_state_to_chunk_material` runs after the update
            // so the freshly-computed state is what gets copied to
            // the material asset.
            .add_systems(
                Update,
                (
                    update_eclipse_uniform.after(ShardOriginSet),
                    mirror_state_to_chunk_material.after(update_eclipse_uniform),
                ),
            );

        // Phase 6.1's deferred post-process pass is intentionally
        // NOT registered. Bevy 0.18's `DefaultOpaqueRendererMethod`
        // is `Forward`, and `StandardMaterial::opaque_render_method`
        // defaults to `Auto` which resolves to Forward — so even
        // when `ScreenSpaceReflections` forces `DeferredPrepass` on
        // the camera (for SSR's G-buffer ray-marching), all our
        // opaque draws still run through the forward path. The
        // forward `EclipseExt` extension above already attenuates the
        // sun term per-fragment in that path, leaving nothing for a
        // deferred post-process to do.
        //
        // Worse, registering it anyway introduces a subtle break: an
        // unconditional `post_process_write` ping-pongs `ViewTarget`
        // and runs `pbr_input_from_deferred_gbuffer` against
        // forward-cleared (zero) G-buffer pixels — the result is
        // pixels off the forward main pass keeping our garbage
        // output, which on Ultra (SSR-on) presents as "no sun
        // lighting" on the dimmer parts of the planet / hull.
        //
        // The shared `eclipse_lib` WGSL module + `EclipseState`
        // resource + the `EclipseDeferredPlugin` itself stay in the
        // tree for the day a material in this codebase opts into
        // `OpaqueRendererMethod::Deferred` (e.g. when we add a true
        // deferred-only effect path). Until then it stays unwired.
    }
}

// ───── Per-frame update ───────────────────────────────────────────────────

/// Drains `WorldState.bodies` from the primary + every secondary,
/// dedupes by `body_id`, camera-rebases positions in double precision,
/// and writes the resulting [`EclipseUniform`] to the [`EclipseState`]
/// resource. The forward path (via [`mirror_state_to_chunk_material`])
/// and the deferred path (via `ExtractResource<EclipseState>`) both
/// read from this single source.
fn update_eclipse_uniform(
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    camera_world: Res<CameraWorldPos>,
    mut state: ResMut<EclipseState>,
) {
    // Collect bodies. Primary wins on collision (it's the
    // authoritative source for the player's current shard); SYSTEM
    // secondary fills gaps during the SHIP / PLANET / SYSTEM
    // dual-shard transition window.
    //
    // BTreeMap (instead of HashMap) keeps iteration order stable from
    // frame to frame — same body_id always lands in the same occluder
    // slot, so the GPU-side per-fragment loop reads consistent slots
    // across frames (avoids a one-frame "occluder shuffle" when a
    // secondary is added or dropped).
    let mut bodies: BTreeMap<u32, &CelestialBodyData> = BTreeMap::new();
    if let Some(ws) = primary_ws.latest.as_ref() {
        for b in &ws.bodies {
            bodies.insert(b.body_id, b);
        }
    }
    for (_shard_type, (ws, _connected)) in secondary_ws.by_shard_type.iter() {
        for b in &ws.bodies {
            // Don't override the primary's body — primary wins.
            bodies.entry(b.body_id).or_insert(b);
        }
    }

    let mut uniform = EclipseUniform::INACTIVE;

    // Sun: body_id == 0. Radius preference: spectroscopically-derived
    // `stellar.radius_solar * R_SUN_M` (server-pre-derived from the
    // mass-radius relation in `core::stellar`); fall back to the
    // generic `body.radius` field if no `StellarState` was broadcast
    // (legacy server, transient pre-Phase-1 system, etc.). Both are
    // physical metres — no magic numbers.
    let Some(star) = bodies.get(&STAR_BODY_ID) else {
        // No sun broadcast → leave uniform at INACTIVE; shaders will
        // return 1.0 (no occlusion). This is the legitimate steady
        // state on the GALAXY shard (no system-shard yet) and during
        // the first-tick race before SystemSceneUpdate arrives.
        state.uniform = uniform;
        return;
    };
    let star_radius_m = star
        .stellar
        .map(|s| (s.radius_solar as f64) * R_SUN_M)
        .unwrap_or(star.radius);
    let star_pos_rel = star.position - camera_world.pos;
    uniform.sun = camera_relative_with_radius(star_pos_rel, star_radius_m as f32);

    // Occluders: every other body, sorted by body_id (BTreeMap
    // iteration already does this). Cap at MAX_ECLIPSE_OCCLUDERS;
    // the bound is hit very rarely.
    let mut count: u32 = 0;
    for (body_id, body) in bodies.iter() {
        if *body_id == STAR_BODY_ID {
            continue;
        }
        if (count as usize) >= MAX_ECLIPSE_OCCLUDERS {
            break;
        }
        let pos_rel = body.position - camera_world.pos;
        uniform.occluders[count as usize] =
            camera_relative_with_radius(pos_rel, body.radius as f32);
        count += 1;
    }
    uniform.count = UVec4::new(count, 0, 0, 0);

    state.uniform = uniform;
}

/// Mirror the authoritative [`EclipseState`] onto the shared chunk
/// material asset so the forward path's material-extension fragment
/// shader sees the latest values through Bevy's standard material
/// pipeline. The deferred path doesn't need this step — it reads
/// `EclipseState` directly via `ExtractResource`.
fn mirror_state_to_chunk_material(
    state: Res<EclipseState>,
    cache: Res<ChunkMaterialCache>,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, EclipseExt>>>,
) {
    let Some(handle) = cache.opaque.as_ref() else {
        return;
    };
    let Some(material) = materials.get_mut(handle) else {
        return;
    };
    material.extension.uniform = state.uniform;
}

/// Pack a camera-relative position (DVec3) + radius (f32) into a
/// shader-side Vec4. The DVec3 → f32 cast is safe: the camera-rebased
/// magnitudes are bounded by orbital distances (≤ 10¹³ m intra-system,
/// ≤ 10¹⁶ intra-galaxy) and f32 retains millimetre precision well
/// below those scales relative to the camera origin.
#[inline]
fn camera_relative_with_radius(pos_rel: DVec3, radius_m: f32) -> Vec4 {
    Vec4::new(pos_rel.x as f32, pos_rel.y as f32, pos_rel.z as f32, radius_m)
}
