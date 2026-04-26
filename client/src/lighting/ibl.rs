//! Image-based lighting (IBL) — Phase 2 (runtime-baked starfield cubemap).
//!
//! Bakes an HDR cubemap from the server-broadcast `StarCatalog` and
//! attaches it to the main camera as Bevy 0.18's
//! [`bevy::light::GeneratedEnvironmentMapLight`]. Bevy's compute-shader
//! prefilter pipeline (`bevy_pbr::light_probe::generate`) runtime-filters
//! the source cubemap into the diffuse (Lambertian) and specular (GGX)
//! maps that the PBR shader uses for ambient + reflection contribution.
//!
//! What this gives the player:
//!
//!   * Anti-sun-side hulls fill in with **directional** starfield-tinted
//!     ambient — the subtle blue-violet cast from a galactic plane on
//!     the shadowed side of a ship that you see in any well-shot space
//!     photograph. Replaces the prior scalar `STARFIELD_AMBIENT_FLOOR`
//!     contribution, which was uniform across all directions.
//!   * Sharp specular reflections of the bright stars on glossy surfaces
//!     (shows up clearly on Phase 8's reflective ship-block materials).
//!
//! BE-authoritative determinism:
//!
//!   * Star directions, classes, luminosities, and physics-derived
//!     spectral colours come exclusively from the broadcast
//!     `StarCatalog` (server-authoritative; computed once per galaxy by
//!     `core::stellar::StellarState::from_class_and_seed` on the
//!     galaxy-shard, then echoed via FlatBuffers).
//!   * The bake itself is a pure deterministic function of
//!     `(galaxy_seed, &[StarCatalogEntryData])`. Two clients on the same
//!     galaxy produce byte-identical cubemaps. The
//!     `tests/no_client_seed_derivation.rs` harness still passes — we
//!     consume broadcast data, never call `from_seed*`.
//!   * Per-star colour preferentially uses
//!     `StarCatalogEntryData.stellar.color_linear_rgb` (Phase-1 blackbody
//!     output). The class-based fallback table is consulted only when
//!     `stellar` is `None` — a transient state for pre-Phase-1 catalogs
//!     or local-from-seed bootstrap before the server's catalog arrives.
//!
//! AAA-quality choices:
//!
//!   * **HDR `Rgba16Float`** source format. Star peak luminance can sit
//!     well above 1.0 linear, so the prefilter sees true HDR values
//!     instead of a flat 8-bit clip. Specular reflections of bright
//!     stars are visibly crisp and bright on glossy hulls.
//!   * **Async bake** on `AsyncComputeTaskPool`. The CPU bake of ~50k
//!     stars over 256² × 6 faces with a per-star Gaussian splat takes
//!     tens of milliseconds — running it on a worker thread keeps the
//!     game-thread frametime flat through warp transitions.
//!   * **Gaussian splat** per star, truncated at ~4σ, with HDR linear
//!     accumulation. Adjacent / overlapping stars combine additively in
//!     full HDR; the prefilter's downsampling produces a smooth diffuse
//!     output instead of the aliasing you get from hard-edged 1×1 stars.
//!   * **Procedural galactic-plane nebula floor** with a galaxy-seed-
//!     derived tint. Different galaxies get visibly different ambient
//!     colour casts (purple, teal, peach …) deterministically.

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::light::GeneratedEnvironmentMapLight;
use bevy::pbr::Atmosphere;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use glam::DVec3;
use half::f16;

use voxeldust_core::client_message::StarCatalogEntryData;
use voxeldust_core::seed::{derive_seed, seed_to_f64};

use crate::config::GameConfig;
use crate::lighting::camera::LightingCamera;
use crate::shard_types::starfield::StarCatalog;

// ──────────────────────────────────────────────────────────────────────────
// Cubemap geometry
// ──────────────────────────────────────────────────────────────────────────

/// Per-face cubemap resolution (texels). Power of two as required by Bevy
/// 0.18's `GeneratedEnvironmentMapLight` prefilter (cubemap size must be
/// a power of two ≤ 8192). 256² gives ~0.35° per texel — finer than the
/// angular size of a single bright star — and bakes in tens of ms on a
/// background thread for a 50 k-star catalog.
const CUBEMAP_FACE_SIZE: u32 = 256;

/// Number of cubemap faces; constant of cube geometry.
const CUBEMAP_FACES: u32 = 6;

/// Bytes per `Rgba16Float` pixel: 4 channels × 2 bytes.
const PIXEL_BYTES: usize = 8;

/// Cube face indices.
const FACE_POS_X: u8 = 0;
const FACE_NEG_X: u8 = 1;
const FACE_POS_Y: u8 = 2;
const FACE_NEG_Y: u8 = 3;
const FACE_POS_Z: u8 = 4;
const FACE_NEG_Z: u8 = 5;

// ──────────────────────────────────────────────────────────────────────────
// Star projection
// ──────────────────────────────────────────────────────────────────────────

/// Stars dimmer than this (luminosity in solar units) are skipped by the
/// IBL bake. Same threshold the starfield mesh uses
/// (`shard_types/starfield.rs::MIN_VISIBLE_LUMINOSITY`). Drops the long
/// tail of M-dwarfs whose individual contribution is below the
/// prefilter's diffuse-map quantisation floor.
const MIN_VISIBLE_LUMINOSITY: f32 = 0.05;

/// Gaussian-splat sigma (texels) used to render each star into the
/// cubemap. 0.8 texels gives a soft 3 × 3 footprint at the centre with
/// energy concentrated in the central pixel — wide enough to alias
/// gracefully across the prefilter, tight enough that adjacent stars
/// stay distinguishable on the source.
const STAR_KERNEL_SIGMA_TEXELS: f32 = 0.8;

/// Gaussian-splat truncation radius (texels). 3 σ contains > 99.7 % of
/// the kernel energy; clipping there saves the per-star cost of writing
/// to texels that contribute essentially zero.
const STAR_KERNEL_RADIUS: i32 = 3;

/// Stevens'-power perceptual exponent for star brightness. Real stellar
/// magnitudes follow a logarithmic ratio, but humans perceive luminance
/// closer to a cube-root law (Stevens 1957). A cube root maps the
/// catalog's wide L⊙ range (~0.05 → 10⁵) onto a visually distinguishable
/// brightness range without crushing the dim end or saturating the bright
/// end on the source cubemap.
const BRIGHTNESS_PERCEPTUAL_EXPONENT: f32 = 1.0 / 3.0;

/// HDR linear-luminance scale applied to a 1 L⊙ star's perceptual
/// brightness on the source cubemap. Chosen so the brightest stars in a
/// typical galaxy (~100 L⊙ ⇒ cube root 4.6 ⇒ ~37) saturate the brightest
/// f16 storage range without exceeding the prefilter's stable input
/// envelope. Visibly bright HDR stars produce visibly bright specular
/// reflections on glossy hulls.
const STAR_HDR_LUMINANCE_PEAK: f32 = 8.0;

// ──────────────────────────────────────────────────────────────────────────
// Nebula floor
// ──────────────────────────────────────────────────────────────────────────

/// Deepest sky-pixel value before stars are added — the "interstellar
/// dust + diffuse galactic-plane glow" background, in linear sRGB. Bevy's
/// prefilter convolves this into the diffuse map's per-direction
/// average, providing a continuous low-luminance ambient floor that
/// keeps the anti-sun side of hulls from reading as pitch black even
/// before any star is integrated.
const NEBULA_FLOOR_LINEAR_R: f32 = 0.005;
const NEBULA_FLOOR_LINEAR_G: f32 = 0.006;
const NEBULA_FLOOR_LINEAR_B: f32 = 0.012;

/// Galaxy-seed-derived nebula-tint scaling for the red and green
/// channels. The `seed_to_f64` output is a uniform [0, 1); multiplying
/// by this gain caps the per-channel tint at this magnitude on the
/// galactic-plane equator.
const NEBULA_PLANE_TINT_GAIN_RG: f32 = 0.04;

/// As above for the blue channel — a hair higher because galaxies in
/// space-art tend to read as bluer than their warm-tone equivalents.
const NEBULA_PLANE_TINT_GAIN_B: f32 = 0.06;

/// Sub-seed indices (paired with `galaxy_seed` via `derive_seed`) used to
/// produce the three nebula-tint channels. The exact constants don't
/// matter — only that they're distinct so the three channels vary
/// independently across galaxies.
const NEBULA_TINT_INDEX_R: u32 = 0xCAFE_BABE;
const NEBULA_TINT_INDEX_G: u32 = 0xDEAD_BEEF;
const NEBULA_TINT_INDEX_B: u32 = 0xFEED_FACE;

/// Galactic-plane gradient exponent. The plane brightness falls off as
/// `(1 − y²)^N` where y is the cube-direction's altitude relative to the
/// galaxy plane. N = 2 gives a soft bell that mirrors a typical
/// edge-on galaxy's apparent thickness.
const NEBULA_PLANE_FALLOFF_EXPONENT: f32 = 2.0;

// ──────────────────────────────────────────────────────────────────────────
// Class-colour fallback (used only when broadcast `stellar` is None)
// ──────────────────────────────────────────────────────────────────────────
//
// Phase 1 populates `StarCatalogEntryData.stellar.color_linear_rgb` from
// the blackbody temperature on the galaxy-shard. These constants are the
// transient-state fallback for legacy / pre-Phase-1 catalogs and for the
// local-from-seed bootstrap path (`shard_types/starfield.rs:285`) where
// `stellar = None` is set deliberately to avoid client-side seed
// derivation. Linear sRGB approximations of the published star-class
// chromaticities — same table the starfield mesh's per-vertex tinting
// uses.

const FALLBACK_STAR_COLOR_O: [f32; 3] = [0.62, 0.69, 1.0];
const FALLBACK_STAR_COLOR_B: [f32; 3] = [0.70, 0.78, 1.0];
const FALLBACK_STAR_COLOR_A: [f32; 3] = [0.84, 0.88, 1.0];
const FALLBACK_STAR_COLOR_F: [f32; 3] = [0.97, 0.96, 0.95];
const FALLBACK_STAR_COLOR_G: [f32; 3] = [1.0, 0.94, 0.80];
const FALLBACK_STAR_COLOR_K: [f32; 3] = [1.0, 0.78, 0.50];
const FALLBACK_STAR_COLOR_M: [f32; 3] = [1.0, 0.55, 0.35];

/// Wire-protocol byte values for `StarClass`. Matches the enum's `as u8`
/// cast order in `core::galaxy::StarClass` (O = 0 … M = 6) and the server-
/// side conversion in `core::client_message::StarCatalogEntryData`. Named
/// here so the fallback-color match arms reference identifiers rather
/// than bare integer literals.
const STAR_CLASS_O: u8 = 0;
const STAR_CLASS_B: u8 = 1;
const STAR_CLASS_A: u8 = 2;
const STAR_CLASS_F: u8 = 3;
const STAR_CLASS_G: u8 = 4;
const STAR_CLASS_K: u8 = 5;
const STAR_CLASS_M: u8 = 6;

// ──────────────────────────────────────────────────────────────────────────
// Resource + plugin
// ──────────────────────────────────────────────────────────────────────────

/// State for the runtime-baked starfield IBL. The `image` handle is
/// fed to `GeneratedEnvironmentMapLight` on the camera; the `pending`
/// task tracks an in-flight async bake so we don't kick off duplicates.
#[derive(Resource, Default)]
pub struct StarfieldIbl {
    /// Galaxy seed the live `image` was baked for. `None` until the
    /// first bake completes.
    pub current_galaxy_seed: Option<u64>,
    /// Handle to the source cubemap. Bevy's prefilter consumes this
    /// and produces the diffuse + specular maps automatically.
    pub image: Option<Handle<Image>>,
    /// In-flight async bake. `None` when nothing is baking.
    pending: Option<PendingBake>,
}

struct PendingBake {
    galaxy_seed: u64,
    task: Task<Vec<u8>>,
}

pub struct IblPlugin;

impl Plugin for IblPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StarfieldIbl>().add_systems(
            Update,
            (
                kick_off_starfield_bake,
                poll_pending_bake,
                attach_starfield_ibl,
            )
                .chain(),
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Bake lifecycle
// ──────────────────────────────────────────────────────────────────────────

/// Spawn an async bake task on the compute pool when the catalog's
/// `galaxy_seed` differs from the one the live cubemap was built for and
/// no bake is currently in flight. Cloning the catalog (~1 KB / 50k
/// stars) is cheap; the bake itself runs on a worker thread.
fn kick_off_starfield_bake(catalog: Res<StarCatalog>, mut state: ResMut<StarfieldIbl>) {
    if state.pending.is_some() {
        return;
    }
    if state.current_galaxy_seed == Some(catalog.galaxy_seed) {
        return;
    }
    if catalog.stars.is_empty() {
        return;
    }
    let galaxy_seed = catalog.galaxy_seed;
    let stars = catalog.stars.clone();
    let task = AsyncComputeTaskPool::get()
        .spawn(async move { bake_starfield_cubemap_bytes(galaxy_seed, &stars) });
    state.pending = Some(PendingBake { galaxy_seed, task });
    tracing::info!(
        galaxy_seed,
        face_size = CUBEMAP_FACE_SIZE,
        "starfield IBL: bake kicked off (async)"
    );
}

/// Poll the pending bake task each frame. When complete, wrap the bytes
/// in a Bevy `Image` and stash the handle for `attach_starfield_ibl` to
/// pick up. Image asset insertion happens on the main thread because
/// `Assets<Image>` is `!Send`.
fn poll_pending_bake(mut state: ResMut<StarfieldIbl>, mut images: ResMut<Assets<Image>>) {
    let Some(pending) = state.pending.as_mut() else {
        return;
    };
    let Some(bytes) = block_on(poll_once(&mut pending.task)) else {
        return;
    };
    let galaxy_seed = pending.galaxy_seed;
    state.pending = None;
    let img = build_cubemap_image(bytes);
    let handle = images.add(img);
    state.image = Some(handle);
    state.current_galaxy_seed = Some(galaxy_seed);
    tracing::info!(
        galaxy_seed,
        face_size = CUBEMAP_FACE_SIZE,
        "starfield IBL: bake completed; cubemap handle published"
    );
}

/// Attach (or re-attach) `GeneratedEnvironmentMapLight` to the lighting
/// camera once the cubemap handle is available, *but only when no
/// `Atmosphere` component is present*. The atmosphere brings its own
/// `AtmosphereEnvironmentMapLight` (per Bevy 0.18's atmosphere stack) —
/// stacking the starfield IBL on top would double-light the scene. The
/// atmosphere module owns the camera's IBL while a planet shard is
/// connected; when that module removes `Atmosphere`, this system
/// re-attaches the starfield IBL automatically on the next frame.
///
/// The `intensity` mirrors the active `LightingFidelity.ibl_intensity_space` —
/// quality preset changes apply on the next frame; galaxy warps re-bake
/// the cubemap asynchronously and the new handle is swapped in here when
/// ready.
fn attach_starfield_ibl(
    state: Res<StarfieldIbl>,
    config: Res<GameConfig>,
    mut commands: Commands,
    cameras: Query<(Entity, Option<&Atmosphere>), With<LightingCamera>>,
    existing: Query<&GeneratedEnvironmentMapLight>,
) {
    let Some(handle) = state.image.as_ref() else {
        return;
    };
    let target_intensity = config.graphics.lighting.ibl_intensity_space;
    for (entity, atmosphere) in &cameras {
        if atmosphere.is_some() {
            // Atmosphere is active — suppress the starfield IBL so the
            // atmosphere's own environment-map light is the sole IBL.
            if existing.get(entity).is_ok() {
                commands.entity(entity).remove::<GeneratedEnvironmentMapLight>();
                tracing::info!(
                    "starfield IBL: removed (atmosphere is active on camera)"
                );
            }
            continue;
        }
        if let Ok(existing) = existing.get(entity) {
            if existing.environment_map == *handle
                && (existing.intensity - target_intensity).abs() < f32::EPSILON
            {
                continue;
            }
        }
        commands.entity(entity).insert(GeneratedEnvironmentMapLight {
            environment_map: handle.clone(),
            intensity: target_intensity,
            rotation: Quat::IDENTITY,
            affects_lightmapped_mesh_diffuse: true,
        });
        tracing::info!(
            intensity = target_intensity,
            "starfield IBL: attached to camera"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Image construction
// ──────────────────────────────────────────────────────────────────────────

/// Wrap the baked HDR f16 byte buffer in a Bevy `Image` ready for the
/// `GeneratedEnvironmentMapLight` prefilter pipeline.
///
/// The prefilter expects:
///
///   * `TextureFormat::Rgba16Float` (HDR-linear, filterable).
///   * Power-of-two square per-face size, ≤ 8192 — `CUBEMAP_FACE_SIZE`
///     satisfies this.
///   * `depth_or_array_layers == 6` — Bevy creates a `D2Array` view over
///     the layers internally and re-binds it as a cube for sampling.
///   * `TextureUsages::TEXTURE_BINDING | COPY_DST` — the prefilter copies
///     the source into a `Rgba16Float` storage texture before downsampling.
fn build_cubemap_image(bytes: Vec<u8>) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: CUBEMAP_FACE_SIZE,
            height: CUBEMAP_FACE_SIZE,
            depth_or_array_layers: CUBEMAP_FACES,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image
}

// ──────────────────────────────────────────────────────────────────────────
// CPU bake (runs on `AsyncComputeTaskPool` worker thread)
// ──────────────────────────────────────────────────────────────────────────

/// Pure-functional CPU bake. Returns the cubemap bytes laid out for
/// `Rgba16Float` consumption: 6 face layers, each `face_size × face_size`
/// pixels, each pixel 4 × f16 little-endian.
///
/// Determinism: the result is a function of `(galaxy_seed, stars)`. Two
/// invocations with byte-equal inputs produce byte-equal outputs.
fn bake_starfield_cubemap_bytes(galaxy_seed: u64, stars: &[StarCatalogEntryData]) -> Vec<u8> {
    let face_size = CUBEMAP_FACE_SIZE as usize;
    let pixel_count = face_size * face_size * (CUBEMAP_FACES as usize);
    // Working buffer in linear HDR f32 (with α ≡ 1.0). f32 lets the per-
    // star Gaussian splats accumulate without precision loss; we encode
    // to f16 only at the end.
    let mut linear: Vec<[f32; 4]> = vec![[0.0; 4]; pixel_count];

    paint_nebula_floor(&mut linear, face_size, galaxy_seed);
    splat_stars(&mut linear, face_size, stars);
    encode_to_f16_le(&linear)
}

/// Lay down the per-direction nebula floor across all 6 faces.
fn paint_nebula_floor(linear: &mut [[f32; 4]], face_size: usize, galaxy_seed: u64) {
    let nebula_tint = nebula_tint_from_seed(galaxy_seed);
    for face in 0..(CUBEMAP_FACES as u8) {
        for py in 0..face_size {
            for px in 0..face_size {
                let u = (px as f64 + 0.5) / face_size as f64;
                let v = (py as f64 + 0.5) / face_size as f64;
                let dir = cube_face_uv_to_dir(face, u, v);
                let plane = directional_plane_falloff(dir);
                let r = NEBULA_FLOOR_LINEAR_R + nebula_tint[0] * plane;
                let g = NEBULA_FLOOR_LINEAR_G + nebula_tint[1] * plane;
                let b = NEBULA_FLOOR_LINEAR_B + nebula_tint[2] * plane;
                let idx = pixel_index(face_size, face, px, py);
                linear[idx] = [r, g, b, 1.0];
            }
        }
    }
}

/// Splat each visible star onto the cubemap with a truncated Gaussian
/// kernel, accumulating in linear HDR. Per-star colour comes from the
/// broadcast `stellar.color_linear_rgb` when present (Phase-1
/// physics-derived blackbody output) and falls back to the class table
/// only for transient pre-Phase-1 catalogs.
fn splat_stars(linear: &mut [[f32; 4]], face_size: usize, stars: &[StarCatalogEntryData]) {
    let two_sigma_sq = 2.0 * STAR_KERNEL_SIGMA_TEXELS * STAR_KERNEL_SIGMA_TEXELS;
    for star in stars {
        let lum = star.luminosity;
        if !lum.is_finite() || lum < MIN_VISIBLE_LUMINOSITY {
            continue;
        }
        let Some(dir) = star.position.try_normalize() else {
            continue;
        };
        let (face, u, v) = dir_to_cube_face_uv(dir);
        let cx = u * face_size as f64 - 0.5;
        let cy = v * face_size as f64 - 0.5;

        let brightness = STAR_HDR_LUMINANCE_PEAK * lum.powf(BRIGHTNESS_PERCEPTUAL_EXPONENT);
        let color = star_color_for_ibl(star);

        let cx_i = cx.round() as i32;
        let cy_i = cy.round() as i32;
        for dy in -STAR_KERNEL_RADIUS..=STAR_KERNEL_RADIUS {
            for dx in -STAR_KERNEL_RADIUS..=STAR_KERNEL_RADIUS {
                let px = cx_i + dx;
                let py = cy_i + dy;
                if px < 0 || py < 0 || px >= face_size as i32 || py >= face_size as i32 {
                    continue;
                }
                let dxf = (px as f64) - cx;
                let dyf = (py as f64) - cy;
                let weight = (-((dxf * dxf + dyf * dyf) as f32) / two_sigma_sq).exp();
                let idx = pixel_index(face_size, face, px as usize, py as usize);
                linear[idx][0] += brightness * color[0] * weight;
                linear[idx][1] += brightness * color[1] * weight;
                linear[idx][2] += brightness * color[2] * weight;
            }
        }
    }
}

/// Encode the linear HDR `[f32; 4]` buffer to little-endian `Rgba16Float`
/// bytes — the layout Bevy expects for the source cubemap.
fn encode_to_f16_le(linear: &[[f32; 4]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(linear.len() * PIXEL_BYTES);
    for px in linear {
        for chan in px {
            bytes.extend_from_slice(&f16::from_f32(*chan).to_le_bytes());
        }
    }
    bytes
}

/// Per-star colour for the IBL bake. Prefer the server's physics-derived
/// `color_linear_rgb` (blackbody from spectral temperature; broadcast in
/// `StarCatalogEntryData.stellar`); fall back to the class table only
/// when `stellar` is `None`.
fn star_color_for_ibl(star: &StarCatalogEntryData) -> [f32; 3] {
    if let Some(stellar) = &star.stellar {
        return stellar.color_linear_rgb;
    }
    fallback_star_class_color(star.star_class)
}

/// Class → linear-sRGB fallback. Only consulted when the broadcast
/// `stellar` field is absent (legacy / bootstrap state).
fn fallback_star_class_color(star_class: u8) -> [f32; 3] {
    match star_class {
        STAR_CLASS_O => FALLBACK_STAR_COLOR_O,
        STAR_CLASS_B => FALLBACK_STAR_COLOR_B,
        STAR_CLASS_A => FALLBACK_STAR_COLOR_A,
        STAR_CLASS_F => FALLBACK_STAR_COLOR_F,
        STAR_CLASS_G => FALLBACK_STAR_COLOR_G,
        STAR_CLASS_K => FALLBACK_STAR_COLOR_K,
        _ => FALLBACK_STAR_COLOR_M,
    }
}

/// Galaxy-seed-derived nebula-tint scale per channel. Each channel
/// independently samples a `derive_seed`-mixed value and maps it to a
/// `[0, max_gain)` interval via the named `NEBULA_PLANE_TINT_GAIN_*`
/// constants. Output is added on top of `NEBULA_FLOOR_LINEAR_*`,
/// modulated by the galactic-plane falloff.
fn nebula_tint_from_seed(galaxy_seed: u64) -> [f32; 3] {
    let r = seed_to_f64(derive_seed(galaxy_seed, NEBULA_TINT_INDEX_R)) as f32
        * NEBULA_PLANE_TINT_GAIN_RG;
    let g = seed_to_f64(derive_seed(galaxy_seed, NEBULA_TINT_INDEX_G)) as f32
        * NEBULA_PLANE_TINT_GAIN_RG;
    let b = seed_to_f64(derive_seed(galaxy_seed, NEBULA_TINT_INDEX_B)) as f32
        * NEBULA_PLANE_TINT_GAIN_B;
    [r, g, b]
}

/// Soft galactic-plane gradient — a function of cube-direction's
/// y-component. Directions parallel to the equator (y ≈ 0) get the full
/// nebula tint; directions toward the poles get only the floor.
/// `NEBULA_PLANE_FALLOFF_EXPONENT` controls how thin the band is.
fn directional_plane_falloff(dir: DVec3) -> f32 {
    let y = dir.y as f32;
    (1.0 - y * y).max(0.0).powf(NEBULA_PLANE_FALLOFF_EXPONENT)
}

// ──────────────────────────────────────────────────────────────────────────
// Cubemap projection helpers
// ──────────────────────────────────────────────────────────────────────────

/// Direction → `(face_index, u, v)` with `u, v ∈ [0, 1]`. Face indices
/// match the standard cubemap order (`+X, −X, +Y, −Y, +Z, −Z`).
fn dir_to_cube_face_uv(dir: DVec3) -> (u8, f64, f64) {
    let abs = dir.abs();
    if abs.x >= abs.y && abs.x >= abs.z {
        if dir.x > 0.0 {
            (FACE_POS_X, 0.5 * (-dir.z / abs.x + 1.0), 0.5 * (-dir.y / abs.x + 1.0))
        } else {
            (FACE_NEG_X, 0.5 * (dir.z / abs.x + 1.0), 0.5 * (-dir.y / abs.x + 1.0))
        }
    } else if abs.y >= abs.z {
        if dir.y > 0.0 {
            (FACE_POS_Y, 0.5 * (dir.x / abs.y + 1.0), 0.5 * (dir.z / abs.y + 1.0))
        } else {
            (FACE_NEG_Y, 0.5 * (dir.x / abs.y + 1.0), 0.5 * (-dir.z / abs.y + 1.0))
        }
    } else if dir.z > 0.0 {
        (FACE_POS_Z, 0.5 * (dir.x / abs.z + 1.0), 0.5 * (-dir.y / abs.z + 1.0))
    } else {
        (FACE_NEG_Z, 0.5 * (-dir.x / abs.z + 1.0), 0.5 * (-dir.y / abs.z + 1.0))
    }
}

/// `(face_index, u, v)` → unit direction. Inverse of `dir_to_cube_face_uv`.
fn cube_face_uv_to_dir(face: u8, u: f64, v: f64) -> DVec3 {
    let s = 2.0 * u - 1.0;
    let t = 2.0 * v - 1.0;
    let dir = match face {
        FACE_POS_X => DVec3::new(1.0, -t, -s),
        FACE_NEG_X => DVec3::new(-1.0, -t, s),
        FACE_POS_Y => DVec3::new(s, 1.0, t),
        FACE_NEG_Y => DVec3::new(s, -1.0, -t),
        FACE_POS_Z => DVec3::new(s, -t, 1.0),
        _ => DVec3::new(-s, -t, -1.0),
    };
    dir.normalize()
}

/// Index into the linear `[f32; 4]` working buffer for `(face, px, py)`.
/// Layout matches Bevy's expected packing: faces stack as 2D-array
/// layers, each layer's pixels in row-major order.
#[inline]
fn pixel_index(face_size: usize, face: u8, px: usize, py: usize) -> usize {
    let face_pixels = face_size * face_size;
    (face as usize) * face_pixels + py * face_size + px
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: pick a direction, project to face/UV, project back —
    /// must recover the (normalised) original.
    #[test]
    fn cube_projection_round_trip() {
        let directions = [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, -1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(0.0, 0.0, -1.0),
            DVec3::new(0.5, 0.5, 0.5).normalize(),
            DVec3::new(-0.3, 0.7, -0.2).normalize(),
        ];
        for dir in directions {
            let (face, u, v) = dir_to_cube_face_uv(dir);
            let recovered = cube_face_uv_to_dir(face, u, v);
            let dot = dir.dot(recovered);
            assert!(
                dot > 0.9999,
                "round-trip lost precision: dir={:?}, face={}, u={}, v={}, recovered={:?}, dot={}",
                dir, face, u, v, recovered, dot
            );
        }
    }

    /// Two bakes with identical inputs must produce byte-identical
    /// outputs — the determinism guarantee.
    #[test]
    fn bake_is_deterministic() {
        let stars = vec![StarCatalogEntryData {
            index: 0,
            position: DVec3::new(1.0, 0.0, 0.0),
            system_seed: 42,
            star_class: 4,
            luminosity: 1.0,
            stellar: None,
        }];
        let a = bake_starfield_cubemap_bytes(0xC0FFEE, &stars);
        let b = bake_starfield_cubemap_bytes(0xC0FFEE, &stars);
        assert_eq!(a, b, "bake output is not deterministic");
    }

    /// Different galaxy seeds must produce different cubemaps (the
    /// nebula tint depends on the seed).
    #[test]
    fn different_seeds_differ() {
        let stars: Vec<StarCatalogEntryData> = vec![];
        let a = bake_starfield_cubemap_bytes(0x1234_5678, &stars);
        let b = bake_starfield_cubemap_bytes(0xABCD_EF01, &stars);
        assert_ne!(a, b, "different seeds produced identical nebula floors");
    }
}
