//! Phase 6 — Eclipse occlusion of the directional sun.
//!
//! A single shared `EclipseUniform` (sun position + radius, plus up to
//! `MAX_ECLIPSE_OCCLUDERS` celestial occluders) is built each frame
//! from `WorldState.bodies` (camera-rebased), then written to the chunk
//! material asset. Every chunk + ship-hull mesh inherits the eclipse
//! computation through `ChunkMaterial = ExtendedMaterial<StandardMaterial,
//! EclipseExt>` — one buffer update, all sun-lit surfaces affected.
//!
//! The shader (`assets/shaders/eclipse.wgsl`) computes the visible-sun
//! fraction at each fragment via lens-area angular-disk intersection
//! (soft penumbra), then **surgically** subtracts only the directional
//! sun's contribution. IBL, ambient, point + spot lights, and the
//! cascade shadow map are untouched — the sky stays lit while the
//! solar disk loses energy, matching real eclipses where atmospheric
//! scattering and rim-light persist through totality.
//!
//! No magic numbers in lighting values: every input is either a
//! physical constant (`R_SUN_M`) or a server-broadcast field
//! (`body.position`, `body.radius`, `body.stellar.radius_solar`).
//!
//! Forward-pipeline support today; deferred (SSR-on, High/Ultra)
//! follow-up is documented in the shader header.

use bevy::asset::Asset;
use bevy::math::{UVec4, Vec4};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
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

/// `EclipseExt` — shader-side bindings for the per-frame eclipse
/// uniform. The base material's `StandardMaterial` bindings live at
/// group 2 bindings 0..N; we bind at 100 to leave headroom for
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

/// Layout matched by `eclipse.wgsl`'s `EclipseUniform`. `ShaderType`
/// derives the std140-aligned GPU layout; the std140 padding lives in
/// the unused lanes of `count` (`.y`, `.z`, `.w`).
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
    /// All-zero state — the shader interprets this as "no sun
    /// configured" and returns 1.0 unconditionally.
    pub const INACTIVE: Self = Self {
        sun: Vec4::ZERO,
        occluders: [Vec4::ZERO; MAX_ECLIPSE_OCCLUDERS],
        count: UVec4::ZERO,
    };
}

impl MaterialExtension for EclipseExt {
    fn fragment_shader() -> ShaderRef {
        "shaders/eclipse.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        // Same shader; the WGSL `PREPASS_PIPELINE` branch writes the
        // G-buffer unchanged (per-fragment eclipse is a deferred-pass
        // follow-up — see shader header).
        "shaders/eclipse.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }
}

pub struct EclipsePlugin;

impl Plugin for EclipsePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, EclipseExt>>::default(),
        )
        // Run after the shard-origin rebasing so the camera-relative
        // positions we write are consistent with the chunk fragments
        // the shader will read on the same frame. Same ordering rule
        // as `solar.rs::update_solar_light`.
        .add_systems(Update, update_eclipse_uniform.after(ShardOriginSet));
    }
}

/// Drains `WorldState.bodies` from the primary + every secondary,
/// dedupes by `body_id`, camera-rebases positions in double precision,
/// and writes the resulting `EclipseUniform` to the shared chunk
/// material so every chunk + ship-hull fragment shares the same
/// occlusion state for the frame.
fn update_eclipse_uniform(
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    camera_world: Res<CameraWorldPos>,
    cache: Res<ChunkMaterialCache>,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, EclipseExt>>>,
) {
    let Some(handle) = cache.opaque.as_ref() else {
        return;
    };
    let Some(material) = materials.get_mut(handle) else {
        return;
    };

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

    // Sun: body_id == 0. Radius preference: spectroscopically-derived
    // `stellar.radius_solar * R_SUN_M` (server-pre-derived from the
    // mass-radius relation in `core::stellar`); fall back to the
    // generic `body.radius` field if no `StellarState` was broadcast
    // (legacy server, transient pre-Phase-1 system, etc.). Both are
    // physical metres — no magic numbers.
    let mut uniform = EclipseUniform::INACTIVE;
    if let Some(star) = bodies.get(&STAR_BODY_ID) {
        let star_radius_m = star
            .stellar
            .map(|s| (s.radius_solar as f64) * R_SUN_M)
            .unwrap_or(star.radius);
        let star_pos_rel = star.position - camera_world.pos;
        uniform.sun = camera_relative_with_radius(star_pos_rel, star_radius_m as f32);
    } else {
        // No sun broadcast → leave uniform at INACTIVE; shader will
        // return 1.0 (no occlusion). This is the legitimate steady
        // state on the GALAXY shard (no system-shard yet) and during
        // the first-tick race before SystemSceneUpdate arrives.
        material.extension.uniform = uniform;
        return;
    }

    // Occluders: every other body. Sort by body_id (BTreeMap iteration
    // already does this) so the slot assignment is stable across
    // frames; cap at `MAX_ECLIPSE_OCCLUDERS` (the loop bound is hit
    // very rarely — 8 is well above the visible occluder count from
    // any realistic vantage in any realistic system).
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

    material.extension.uniform = uniform;
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
