//! Graphics / rendering-fidelity configuration.
//!
//! `LightingFidelity` is the user-selectable rendering-quality knob. It
//! configures how shadows are rasterized, which AA mode runs, whether
//! post-FX passes execute, and how detailed the atmosphere LUT is.
//!
//! It does NOT configure any *lighting value* — sun colour, illuminance,
//! atmospheric scattering parameters, planet rotation phase, eclipse
//! occlusion are all derived from physics + server-broadcast state and are
//! identical across all quality presets. Ultra and Low render the *same*
//! scene; Ultra just resolves it more accurately.
//!
//! The `tests/no_magic_numbers.rs` integration test allows numeric literals
//! in this file (rendering-fidelity knobs have no physical meaning) but bans
//! them in lighting modules — lighting code must reference these fields by
//! name, not inline literals.
//!
//! Presets (`low / medium / high / ultra`) are constructors that fill every
//! field. The "Custom" preset is implicit: any user override of a single
//! field via the in-game settings UI keeps the original preset name as a
//! debug hint while individual fields take their override values.

/// Interior-spec ambient floor: ~bright-office equivalent (cd/m²).
///
/// Used when the camera is *inside* a sealed ship hull (Phase 11+ adds the
/// inside-vs-outside detection). For now, presets default to the
/// vacuum floor `voxeldust_core::physics_constants::STARFIELD_AMBIENT_FLOOR_CD_PER_M2`
/// because the most common camera state is "in space looking at hulls" —
/// using the interior value there washes out the lit-vs-shaded contrast.
#[allow(dead_code)]
const INTERIOR_AMBIENT_FLOOR_CD_PER_M2: f32 = 800.0;

/// Vacuum ambient floor (cd/m²) — pulled from the physics constants so a
/// single source of truth governs the "near-black ambient that still leaves
/// the anti-sun hull readable" target. See
/// `core/src/physics_constants.rs::STARFIELD_AMBIENT_FLOOR_CD_PER_M2` for
/// the photometric derivation.
const VACUUM_AMBIENT_FLOOR_CD_PER_M2: f32 =
    voxeldust_core::physics_constants::STARFIELD_AMBIENT_FLOOR_CD_PER_M2 as f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightingPreset {
    Low,
    Medium,
    High,
    Ultra,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AaMode {
    /// No anti-aliasing.
    None,
    /// Sub-pixel morphological anti-aliasing (post-process; no prepass cost).
    Smaa,
    /// Temporal anti-aliasing (requires depth + normal + motion-vector prepass).
    Taa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtmosphereMethod {
    /// Cheap, mobile-friendly: precomputed transmittance + multiscatter LUTs.
    LookupTexture,
    /// Full ray-marched scattering — highest quality, expensive.
    Raymarched,
}

/// Rendering-fidelity knobs for the lighting / post-processing pipeline.
/// **Never affects lighting values** — only how they're rasterized / approximated.
#[derive(Debug, Clone)]
pub struct LightingFidelity {
    /// Originating preset (debug / UI). Individual fields may have been
    /// overridden in-place; this remains the "starting point" indicator.
    pub preset: LightingPreset,

    // ─── Shadows (CSM for the directional sun) ─────────────────────────────
    /// Number of cascaded-shadow-map cascades (Bevy: 1..=8). More cascades =
    /// crisper shadows at distance, more depth-pass cost.
    pub cascade_count: u8,
    /// Square shadow-map texture size per cascade. Higher = sharper edges,
    /// more VRAM (each cascade is its own depth texture of this size).
    pub shadow_map_size: u32,
    /// Far cap of the last CSM cascade, in metres. Beyond this distance no
    /// shadows are computed (atmosphere occlusion still applies). Should be
    /// the player's desired shadow-render distance, not the planet radius.
    pub cascade_max_distance_m: f32,
    /// Near cap of the first cascade — sets the closest crisp-shadow scale.
    pub cascade_minimum_distance_m: f32,
    /// Boundary of the first cascade's far edge, in metres. Tight values
    /// give crisp character / cockpit shadows.
    pub first_cascade_far_bound_m: f32,
    /// Cascade-blend overlap proportion (0..=1). Hides cascade-seam pop.
    pub cascade_overlap_proportion: f32,

    // ─── Anti-aliasing ─────────────────────────────────────────────────────
    pub aa_mode: AaMode,

    // ─── Atmosphere ────────────────────────────────────────────────────────
    pub atmosphere_method: AtmosphereMethod,

    // ─── Post-FX gates ─────────────────────────────────────────────────────
    pub ssao_enabled: bool,
    pub ssr_enabled: bool,
    pub volumetric_fog_enabled: bool,

    // ─── Bloom ─────────────────────────────────────────────────────────────
    /// Bloom strength (Bevy: 0..=1). Visual taste, not a physical value.
    pub bloom_intensity: f32,

    // ─── Exposure ──────────────────────────────────────────────────────────
    /// EV100 to apply when the camera is in deep space (no atmosphere).
    pub exposure_ev100_space: f32,
    /// EV100 to apply on a sunlit planet surface.
    pub exposure_ev100_surface: f32,
    /// Time (seconds) over which to lerp exposure between regimes during
    /// shard transitions. Altitude-driven, not boolean — see
    /// `client/src/lighting/camera.rs`. Configurable so users can tune
    /// transition smoothness.
    pub exposure_transition_time_s: f32,
    /// If true, `bevy::post_process::auto_exposure::AutoExposure` runs
    /// instead of fixed lerping. Recommended on High+ only (#13446).
    pub auto_exposure_enabled: bool,

    // ─── Local-light budget ────────────────────────────────────────────────
    /// Maximum number of point/spot lights allowed to cast shadows
    /// simultaneously. Non-shadow-casting lights are nearly free in Bevy's
    /// clustered forward+ pipeline; shadow-casting lights cost a depth pass
    /// each. The lighting module enforces this budget — overflow lights run
    /// shadowless.
    pub max_shadow_casting_local_lights: u32,

    // ─── Image-based lighting (IBL) ────────────────────────────────────────
    /// Intensity multiplier for the runtime-baked starfield environment-map
    /// in space (cd/m² scale). Higher values give brighter rim lighting on
    /// non-sun-side hulls. The starfield map content itself is physical
    /// (baked from the StarCatalog); this scalar is a visual taste knob.
    pub ibl_intensity_space: f32,

    // ─── Atmosphere-driven IBL ─────────────────────────────────────────────
    /// Intensity multiplier for `AtmosphereEnvironmentMapLight` when on a
    /// planet surface. Same role as `ibl_intensity_space` but for the
    /// auto-baked atmosphere cubemap.
    pub ibl_intensity_atmosphere: f32,

    // ─── Starfield ambient floor ───────────────────────────────────────────
    /// Ambient luminance from the integrated starfield + scattered
    /// host-star light + planetary albedo, in candela per square metre.
    /// Provides the physical "darkness floor" inside a star system so the
    /// anti-sun side of hulls and unlit ship interiors are not pitch black.
    /// Default is sourced from `core::physics_constants::STARFIELD_AMBIENT_FLOOR_CD_PER_M2`
    /// (~200 cd/m², scaled by quality preset for visual taste). Phase 2's
    /// runtime-baked starfield IBL replaces this with directional ambient.
    pub starfield_ambient_floor_cd_per_m2: f32,

    // ─── Directional-light safety knobs ────────────────────────────────────
    /// Whether the directional sun casts shadows (CSM). When `false`, the
    /// star light reaches every visible fragment regardless of cascade
    /// frusta — useful as a diagnostic to isolate shadow-rendering bugs
    /// from lighting bugs, and as a Low-tier fallback on platforms where
    /// CSM has issues (e.g., specific Metal driver versions).
    pub directional_shadows_enabled: bool,
    /// Upper bound on the directional light's illuminance, in lux. Bevy
    /// 0.18's PBR shader has been observed to behave unstably at very high
    /// illuminance values (~10⁶ lux) on some backends. Real Sol-at-Earth is
    /// ~127 000 lux; clamping to ~200 000 lux keeps the lit-face appearance
    /// at "AAA bright outdoor sunlit" without crossing into edge-case
    /// territory. The clamp is a *rendering* safety knob — if the player is
    /// physically very close to a star and would receive 10⁷ lux, the
    /// scene has already exited any reasonable exposure regime; auto-
    /// exposure (Phase 7) will handle that case correctly.
    pub max_directional_illuminance_lux: f32,

    // ─── Shadow biases (voxel-tuned) ───────────────────────────────────────
    /// Depth bias added to fragment-depth before shadow-map comparison.
    /// Tuned for axis-aligned voxel cube faces — Bevy's default 0.02 is
    /// calibrated for organic mesh geometry where adjacent face normals
    /// vary smoothly; voxel cubes meet at sharp 90° corners that need
    /// much more aggressive biases (~0.5) to prevent self-shadowing on
    /// the corner. Without this, every voxel face samples its own
    /// adjacent face's depth as an occluder → every fragment reads as
    /// in-shadow → fully dark hull.
    pub shadow_depth_bias: f32,
    /// Distance to push the shadow-comparison sample point along the
    /// surface normal. Same calibration logic as `shadow_depth_bias`:
    /// voxel-cube corners need ~5.0 vs Bevy's default 1.8.
    pub shadow_normal_bias: f32,

    // ─── Camera HDR pipeline ───────────────────────────────────────────────
    /// Enable the HDR rendering pipeline (Hdr render target → Tonemapping →
    /// Bloom → LDR display). Without HDR, the camera writes 8-bit-per-channel
    /// LDR values directly, the directional sun appears clipped to white on
    /// lit faces, and Bloom is unavailable. With HDR enabled, the full
    /// dynamic range is preserved through the post-FX chain.
    ///
    /// Defaults to `false` because **Bevy 0.18 + Metal currently produces a
    /// degenerate cascade shadow map when HDR is attached to the camera** —
    /// every fragment samples as in-shadow, making the whole hull pitch
    /// black. Disabling HDR matches the Phase-0 known-working camera
    /// configuration. Toggle to `true` on backends where HDR + CSM work
    /// (Vulkan, DX12) or once the Metal interaction is root-caused.
    pub hdr_enabled: bool,
    /// Enable Bloom (HDR-only — automatically gated on `hdr_enabled`).
    pub bloom_enabled: bool,
    /// Enable filmic tonemapping. When `false`, the camera uses Bevy's
    /// default no-op tonemap; when `true`, applies AgX (filmic, wide-DR).
    /// Visually meaningful only when `hdr_enabled` is also `true`.
    pub tonemap_enabled: bool,
}

impl LightingFidelity {
    /// Lowest fidelity — playable on integrated GPUs / older hardware.
    /// Uses LDR rendering with an explicit `Tonemapping::None` (set by
    /// `lighting::camera`) so Bevy's auto-required `TonyMcMapface` doesn't
    /// silently apply on top of LDR. The directional sun is clamped to a
    /// value that lands at the LDR clipping threshold under
    /// `exposure_ev100_space = 11`.
    pub fn low() -> Self {
        Self {
            preset: LightingPreset::Low,
            cascade_count: 2,
            shadow_map_size: 1024,
            cascade_max_distance_m: 250.0,
            cascade_minimum_distance_m: 0.1,
            first_cascade_far_bound_m: 16.0,
            cascade_overlap_proportion: 0.2,
            aa_mode: AaMode::Smaa,
            atmosphere_method: AtmosphereMethod::LookupTexture,
            ssao_enabled: false,
            ssr_enabled: false,
            volumetric_fog_enabled: false,
            bloom_intensity: 0.20,
            exposure_ev100_space: 11.0,
            exposure_ev100_surface: 13.0,
            exposure_transition_time_s: 1.5,
            auto_exposure_enabled: false,
            max_shadow_casting_local_lights: 4,
            ibl_intensity_space: 50.0,
            ibl_intensity_atmosphere: 200.0,
            starfield_ambient_floor_cd_per_m2: VACUUM_AMBIENT_FLOOR_CD_PER_M2,
            directional_shadows_enabled: true,
            shadow_depth_bias: shadow_depth_bias_voxel(),
            shadow_normal_bias: shadow_normal_bias_voxel(),
            // LDR clamp lands a vertex-colored hull face at `NdotL = 1` near
            // the 1.0-linear LDR clip threshold under `ev100 = 11`, leaving
            // the full Lambertian ramp visible. The HDR-on presets lift this
            // cap so AgX tonemaps the actual physical sun.
            max_directional_illuminance_lux: 30_000.0,
            hdr_enabled: false,
            bloom_enabled: false,
            tonemap_enabled: false,
        }
    }

    /// Default for typical mid-range hardware. Ships the AAA HDR + AgX +
    /// Bloom pipeline: physical sun illuminance flows through the shader
    /// uncapped, AgX rolloff handles the wide dynamic range, Bloom adds the
    /// energy-conserving glow on the bright end. SSR and volumetric fog are
    /// gated off (those are High+ only).
    ///
    /// Cascade ranges and shadow biases match the voxel-tuned values from
    /// `low()` and `high()` (smooth ordering Low→Medium→High→Ultra by
    /// `max_distance` and `first_cascade_far_bound`). Same biases as the
    /// other AAA presets — deviating from those defaults caused visible
    /// shadow acne in earlier iterations.
    pub fn medium() -> Self {
        Self {
            preset: LightingPreset::Medium,
            cascade_count: 4,
            shadow_map_size: 4096,
            cascade_max_distance_m: 500.0,
            cascade_minimum_distance_m: 0.1,
            first_cascade_far_bound_m: 8.0,
            cascade_overlap_proportion: 0.2,
            aa_mode: AaMode::Smaa,
            atmosphere_method: AtmosphereMethod::LookupTexture,
            ssao_enabled: true,
            ssr_enabled: false,
            volumetric_fog_enabled: false,
            bloom_intensity: 0.25,
            exposure_ev100_space: hdr_ev100_space(),
            exposure_ev100_surface: 14.0,
            exposure_transition_time_s: 1.5,
            auto_exposure_enabled: false,
            max_shadow_casting_local_lights: 32,
            ibl_intensity_space: 100.0,
            ibl_intensity_atmosphere: 500.0,
            starfield_ambient_floor_cd_per_m2: VACUUM_AMBIENT_FLOOR_CD_PER_M2,
            directional_shadows_enabled: true,
            shadow_depth_bias: shadow_depth_bias_voxel(),
            shadow_normal_bias: shadow_normal_bias_voxel(),
            max_directional_illuminance_lux: HDR_ILLUMINANCE_NO_CLAMP,
            hdr_enabled: true,
            bloom_enabled: true,
            tonemap_enabled: true,
        }
    }

    /// Recommended for modern GPUs (RTX 30xx-class and above).
    pub fn high() -> Self {
        Self {
            preset: LightingPreset::High,
            cascade_count: 4,
            shadow_map_size: 4096,
            cascade_max_distance_m: 1000.0,
            cascade_minimum_distance_m: 0.1,
            first_cascade_far_bound_m: 4.0,
            cascade_overlap_proportion: 0.2,
            aa_mode: AaMode::Taa,
            atmosphere_method: AtmosphereMethod::Raymarched,
            ssao_enabled: true,
            ssr_enabled: true,
            volumetric_fog_enabled: true,
            bloom_intensity: 0.30,
            exposure_ev100_space: hdr_ev100_space(),
            exposure_ev100_surface: 14.0,
            exposure_transition_time_s: 1.0,
            auto_exposure_enabled: true,
            max_shadow_casting_local_lights: 256,
            ibl_intensity_space: 150.0,
            ibl_intensity_atmosphere: 1000.0,
            starfield_ambient_floor_cd_per_m2: VACUUM_AMBIENT_FLOOR_CD_PER_M2,
            directional_shadows_enabled: true,
            shadow_depth_bias: shadow_depth_bias_voxel(),
            shadow_normal_bias: shadow_normal_bias_voxel(),
            max_directional_illuminance_lux: HDR_ILLUMINANCE_NO_CLAMP,
            hdr_enabled: true,
            bloom_enabled: true,
            tonemap_enabled: true,
        }
    }

    /// Maximum visual fidelity — high-end GPUs only.
    pub fn ultra() -> Self {
        Self {
            preset: LightingPreset::Ultra,
            cascade_count: 6,
            shadow_map_size: 4096,
            cascade_max_distance_m: 2000.0,
            cascade_minimum_distance_m: 0.05,
            first_cascade_far_bound_m: 2.0,
            cascade_overlap_proportion: 0.2,
            aa_mode: AaMode::Taa,
            atmosphere_method: AtmosphereMethod::Raymarched,
            ssao_enabled: true,
            ssr_enabled: true,
            volumetric_fog_enabled: true,
            bloom_intensity: 0.35,
            exposure_ev100_space: hdr_ev100_space(),
            exposure_ev100_surface: 14.0,
            exposure_transition_time_s: 0.8,
            auto_exposure_enabled: true,
            max_shadow_casting_local_lights: 1024,
            ibl_intensity_space: 200.0,
            ibl_intensity_atmosphere: 2000.0,
            starfield_ambient_floor_cd_per_m2: VACUUM_AMBIENT_FLOOR_CD_PER_M2,
            directional_shadows_enabled: true,
            shadow_depth_bias: shadow_depth_bias_voxel(),
            shadow_normal_bias: shadow_normal_bias_voxel(),
            max_directional_illuminance_lux: HDR_ILLUMINANCE_NO_CLAMP,
            hdr_enabled: true,
            bloom_enabled: true,
            tonemap_enabled: true,
        }
    }
}

/// Effective "no clamp" — orders of magnitude above any physical stellar
/// illuminance the player can be exposed to (Sol-at-Earth = ~1.27 × 10⁵ lux).
/// AgX tonemap handles the dynamic range; the clamp exists only as a render-
/// stability guard against pathological values (e.g. camera coincident with
/// the star at game start).
const HDR_ILLUMINANCE_NO_CLAMP: f32 = 1.0e9;

/// Shadow-bias defaults for voxel scenes under a rotating ChunkSource
/// parent.
///
/// Bevy defaults (`0.02 / 1.8`) leave the cascade depth-comparison
/// sensitive to per-pixel precision noise — fragments at the cascade
/// shadow-map texel boundary flicker between in-shadow and lit each
/// frame as the camera rotates the ship, reading as wandering bright
/// dots/spots on otherwise dark hull faces. The fix raises the normal
/// bias enough to push the comparison sample reliably off the receiver
/// surface, while keeping depth bias modest so soft contact shadows
/// (under crates, in corners) stay visually grounded.
///
/// Calibrated on Bevy 0.18 + Metal at 4 cascades, 0.1–500 m,
/// 4096² shadow map, with ChunkSource rotation driven from
/// authoritative ship pose.
const fn shadow_depth_bias_voxel() -> f32 {
    0.02
}
const fn shadow_normal_bias_voxel() -> f32 {
    1.8
}

/// `ev100` for sunlit space scenes with HDR + AgX. Sunny-16 (`ev100 = 15`)
/// over-darkens the 200 cd/m² ambient floor (interior/anti-sun faces go
/// near-black); `11` over-exposes the lit hull (clipped to a blown-out
/// white). `13` strikes the bright/dark balance until Phase 7's
/// `AutoExposure` plugin lands.
const fn hdr_ev100_space() -> f32 {
    13.0
}

// Calibration notes (visible to anyone reading this file):
//   ev100 ≈ 15 corresponds to "bright outdoor sunlight"
//   (middle-gray luminance ≈ 7100 cd/m², matching real-world Sunny-16 photography).
//   In space close to a Sol-class star (E ≈ 1.27e5 lux), a Lambertian
//   surface with albedo 0.3 luminates at ≈ 0.3 · E / π ≈ 12 100 cd/m² —
//   1.7× middle gray, properly exposed by AgX.
//   ev100 ≈ 14 is one stop more open, used on planet surfaces where the
//   atmosphere transmits ~70 % of incoming sunlight and the surface
//   appears slightly dimmer than equivalent space exposure.

impl Default for LightingFidelity {
    fn default() -> Self {
        Self::medium()
    }
}

/// Top-level graphics config — sub-section of `GameConfig`.
#[derive(Debug, Clone, Default)]
pub struct GraphicsConfig {
    pub lighting: LightingFidelity,
}
