//! Camera HDR + tonemapping + bloom + exposure pipeline.
//!
//! Components attached to the main camera here:
//!   * `Hdr` — render target keeps wide dynamic range until tonemap.
//!   * `Tonemapping::AgX` — film-like rolloff that handles >>1.0 emissive
//!     suns gracefully.
//!   * `Bloom::NATURAL` — energy-conserving, multi-mip Gaussian; intensity
//!     from `LightingFidelity.bloom_intensity`.
//!   * `Exposure` — fixed `ev100` from the per-shard-context lerp target
//!     (see `exposure_ev100_space` / `exposure_ev100_surface`); Phase 3
//!     introduces altitude-driven smooth interpolation between regimes.
//!
//! Apply path: a Bevy `Update` system attaches the components when the
//! camera entity first exists. The choice of HDR/Bloom/Tonemap is read once
//! from the active `LightingFidelity` preset and is not reconfigured live.
//! Cascade-shadow config flows through `solar.rs`.
//!
//! Future phases extend this stack:
//!   * Phase 2: `EnvironmentMapLight` (runtime-baked starfield) + atmosphere
//!     cubemap.
//!   * Phase 3: `Atmosphere` + `AtmosphereSettings` + `AtmosphereEnvironmentMapLight`
//!     when a planet shard is connected.
//!   * Phase 5: `VolumetricFog` (quality-gated).
//!   * Phase 7: `DepthPrepass` / `NormalPrepass` / `MotionVectorPrepass` +
//!     `TemporalAntiAliasing` / `Smaa` / `Ssao` / `Ssr` / `AutoExposurePlugin`.

use bevy::anti_alias::smaa::{Smaa, SmaaPreset};
use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::camera::Exposure;
use bevy::core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{GlobalAmbientLight, VolumetricFog};
use bevy::pbr::{
    ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel,
    ScreenSpaceReflections,
};
use bevy::post_process::auto_exposure::AutoExposure;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::{Hdr, Msaa};

use crate::MainCamera;
use crate::config::{AaMode, GameConfig};

/// Lower bound of the AutoExposure histogram range, in ev100 stops.
/// The default range is `[-8, +8]` (16 stops), which on Bevy 0.18 can
/// run away into the over-exposed regime when scenes go very dark
/// (issue #13446). Clamping the dark side to −2 keeps the metering
/// inside a physically plausible envelope: −2 ev100 corresponds to a
/// scene luminance of ~3 cd/m² — well below dim interior lighting but
/// above the integrated-starlight floor we never want to meter against.
const AUTO_EXPOSURE_EV_RANGE_LOW: f32 = -2.0;

/// Upper bound of the AutoExposure histogram range, in ev100 stops.
/// 14 stops corresponds to ~16 000 cd/m² — between an overcast sky and
/// direct daylight. The bright tail covers our peak photometric sun
/// without being so wide that one over-saturated specular highlight
/// pulls the whole-scene average upward.
const AUTO_EXPOSURE_EV_RANGE_HIGH: f32 = 14.0;

/// Marker so the apply system can find "the camera that lighting owns" —
/// kept narrower than `MainCamera` in case future cameras (e.g., a separate
/// in-world tablet camera) shouldn't get the lighting stack.
#[derive(Component, Debug, Clone, Copy)]
pub struct LightingCamera;

pub struct LightingCameraPlugin;

impl Plugin for LightingCameraPlugin {
    fn build(&self, app: &mut App) {
        // Ambient light is owned here (not in `main.rs`). Initial brightness
        // is overwritten by `apply_starfield_ambient_floor` on the first
        // `GameConfig::is_changed()` tick (which fires when the resource is
        // first inserted), so the brightness reflects the active preset's
        // `starfield_ambient_floor_cd_per_m2`.
        app.insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 0.0,
            affects_lightmapped_meshes: true,
        });
        app.add_systems(
            Update,
            (attach_camera_lighting_stack, apply_starfield_ambient_floor),
        );
    }
}

/// Attach lighting components to the main camera as soon as it exists. The
/// `LightingCamera` marker is added unconditionally so the query becomes a
/// no-op after first attach.
///
/// Runs in `Update` (not `Startup`) because `setup_camera` and this system
/// are independent and `Startup`-ordering is not guaranteed — a `Startup`
/// attach would race the camera spawn and miss it. The `Without<LightingCamera>`
/// filter makes subsequent frames cheap no-ops once the stack is in place.
///
/// **Why `Tonemapping` is always inserted explicitly**: Bevy 0.18's
/// `Core3dPlugin` registers `Tonemapping` as a required component of
/// `Camera3d` (`bevy_core_pipeline-0.18.1/src/core_3d/mod.rs:149`), and the
/// enum's `#[default]` is `TonyMcMapface`
/// (`tonemapping/mod.rs:157`). If we *skip* inserting Tonemapping on
/// "tonemap disabled" presets, Bevy silently runs TonyMcMapface anyway —
/// which compresses the directional sun against the ambient floor and
/// produces the "no light, hull not highlighted" symptom. Inserting an
/// explicit `Tonemapping::None` is the only way to actually disable
/// tonemapping in this Bevy version.
fn attach_camera_lighting_stack(
    mut commands: Commands,
    config: Res<GameConfig>,
    cameras: Query<Entity, (With<MainCamera>, Without<LightingCamera>)>,
) {
    let fidelity = &config.graphics.lighting;
    for entity in &cameras {
        let mut entity_commands = commands.entity(entity);
        entity_commands.insert(LightingCamera);

        if fidelity.hdr_enabled {
            entity_commands.insert(Hdr);
            // Disable hardware MSAA when HDR is on — matches the reference
            // setup in `bevy/examples/3d/atmosphere.rs`. The HDR + post-FX
            // chain (tonemap, bloom, SMAA/TAA) does its own AA in screen
            // space; layering hardware MSAA on top wastes shading cost and
            // can interact badly with cascade-shadow filtering on Metal.
            entity_commands.insert(Msaa::Off);
        }

        let tonemap = pick_tonemap(fidelity.tonemap_enabled, fidelity.hdr_enabled);
        entity_commands.insert(tonemap);

        if fidelity.bloom_enabled && fidelity.hdr_enabled {
            entity_commands.insert(Bloom {
                intensity: fidelity.bloom_intensity,
                ..Bloom::NATURAL
            });
        }

        // ─── Prepasses (Phase 7) ──────────────────────────────────────────
        //
        // Most post-FX (TAA, SSAO, motion-blur in future) need extra
        // per-fragment data that the main forward pass doesn't write.
        // We attach the prepass components here so the shader pipeline
        // emits them only when something actually consumes them — keeps
        // Low-tier rendering free of any prepass cost.
        //
        //   * DepthPrepass — needed by SSAO, TAA, SSR, and the volumetric
        //     fog pass (Phase 5). One depth texture, ~2 ms / frame at 4K.
        //   * NormalPrepass — needed by SSAO + TAA's history-validation
        //     reprojection. Adds the normal G-buffer.
        //   * MotionVectorPrepass — required by TAA so the shader can
        //     reproject the previous frame; also useful for motion blur.
        let needs_depth_prepass = fidelity.ssao_enabled
            || matches!(fidelity.aa_mode, AaMode::Taa)
            || fidelity.ssr_enabled
            || fidelity.volumetric_fog_enabled;
        let needs_normal_prepass = fidelity.ssao_enabled
            || matches!(fidelity.aa_mode, AaMode::Taa);
        let needs_motion_prepass = matches!(fidelity.aa_mode, AaMode::Taa);
        if needs_depth_prepass {
            entity_commands.insert(DepthPrepass);
        }
        if needs_normal_prepass {
            entity_commands.insert(NormalPrepass);
        }
        if needs_motion_prepass {
            entity_commands.insert(MotionVectorPrepass);
        }

        // ─── Anti-aliasing ────────────────────────────────────────────────
        //
        // SMAA is post-process and works without any prepass — the safe
        // pick at Low and as a fallback when the user explicitly asks for
        // it. TAA is the AAA-tier choice on Medium+: temporally
        // accumulated samples eliminate sub-pixel shimmer on voxel-block
        // edges, sun discs, and the atmospheric ring. TAA's history
        // reprojection is exactly why we attached the motion-vector
        // prepass above.
        match fidelity.aa_mode {
            AaMode::Smaa => {
                entity_commands.insert(Smaa {
                    preset: SmaaPreset::Ultra,
                });
            }
            AaMode::Taa => {
                entity_commands.insert(TemporalAntiAliasing::default());
            }
            AaMode::None => {}
        }

        // ─── SSAO ─────────────────────────────────────────────────────────
        //
        // Adds contact shadows in concave geometry — corners of voxel
        // blocks, ship-interior crevices. Bevy's SSAO requires
        // `DepthPrepass` + `NormalPrepass` (attached above when enabled).
        // Quality level scales sample count + cost; `Medium` is the
        // visible-quality / GPU-cost sweet spot on consumer hardware.
        if fidelity.ssao_enabled && needs_depth_prepass && needs_normal_prepass {
            entity_commands.insert(ScreenSpaceAmbientOcclusion {
                quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Medium,
                ..ScreenSpaceAmbientOcclusion::default()
            });
        }

        // ─── Volumetric fog (Phase 5) ─────────────────────────────────────
        //
        // Adds light shafts ("god rays") through ship windows / planet
        // atmospheres. Bevy raymarches the participating-medium volume
        // along view rays, scattering the directional sun's contribution
        // toward the camera. The DirectionalLight that contributes is
        // marked with `VolumetricLight` (see `solar.rs::setup_solar_light`).
        // Default `VolumetricFog::default()` parameters give a subtle,
        // tasteful effect; quality presets toggle this on at High+.
        if fidelity.volumetric_fog_enabled {
            entity_commands.insert(VolumetricFog::default());
        }

        // ─── SSR ──────────────────────────────────────────────────────────
        //
        // Screen-space reflections capture glossy mirror-like reflections
        // of nearby geometry — strongest on low-roughness surfaces. The
        // `ScreenSpaceReflections` component carries
        // `#[require(DepthPrepass, DeferredPrepass)]`; Bevy auto-inserts
        // both, switching the camera's opaque pipeline to the deferred
        // renderer for the duration. `StandardMaterial` (which
        // `ChunkMaterial` aliases) natively supports both forward and
        // deferred, so this is a per-camera switch rather than a global
        // pipeline change. Using Bevy's defaults (intersection threshold,
        // step counts) — they're tuned for consumer GPUs.
        if fidelity.ssr_enabled && fidelity.hdr_enabled {
            entity_commands.insert(ScreenSpaceReflections::default());
        }

        // ─── Exposure ─────────────────────────────────────────────────────
        //
        // AutoExposure (HDR-only, compute-shader path) auto-adapts to
        // scene luminance — the ship-interior-to-space-to-surface
        // exposure transitions become invisible. The histogram range is
        // intentionally tightened to `[-2, +14] ev100` to avoid the Bevy
        // 0.18 dark-scene over-expose bug (#13446); without the clamp,
        // very dark scenes (e.g. ship interior in a planet's umbra)
        // can run away to maximum sensitivity and over-expose on the
        // next bright frame. The clamp keeps the metering inside a
        // reasonable physical envelope.
        //
        // When AutoExposure is OFF, Bevy's PBR shader still requires an
        // `Exposure` component for both HDR and LDR pipelines, so we
        // insert the static `ev100` from the active fidelity preset.
        if fidelity.auto_exposure_enabled && fidelity.hdr_enabled {
            entity_commands.insert(AutoExposure {
                range: AUTO_EXPOSURE_EV_RANGE_LOW..=AUTO_EXPOSURE_EV_RANGE_HIGH,
                ..AutoExposure::default()
            });
        } else {
            entity_commands.insert(Exposure {
                ev100: fidelity.exposure_ev100_space,
            });
        }

        tracing::info!(
            preset = ?fidelity.preset,
            hdr = fidelity.hdr_enabled,
            tonemap = ?tonemap,
            bloom = fidelity.bloom_enabled,
            aa_mode = ?fidelity.aa_mode,
            shadows = fidelity.directional_shadows_enabled,
            ev100 = fidelity.exposure_ev100_space,
            auto_exposure = fidelity.auto_exposure_enabled,
            ssao = fidelity.ssao_enabled,
            ssr = fidelity.ssr_enabled,
            depth_prepass = needs_depth_prepass,
            normal_prepass = needs_normal_prepass,
            motion_prepass = needs_motion_prepass,
            ambient_cd_per_m2 = fidelity.starfield_ambient_floor_cd_per_m2,
            illum_clamp_lux = fidelity.max_directional_illuminance_lux,
            "lighting/camera: attached lighting stack to main camera"
        );
    }
}

/// Translate the `tonemap_enabled` + `hdr_enabled` config pair into a
/// concrete `Tonemapping` variant.
fn pick_tonemap(tonemap_enabled: bool, hdr_enabled: bool) -> Tonemapping {
    if tonemap_enabled && hdr_enabled {
        Tonemapping::AgX
    } else {
        // Both LDR mode and "HDR but tonemap disabled" want a straight
        // pass-through. `Tonemapping::None` is the only variant Bevy 0.18
        // treats as a true no-op (`is_enabled()` returns false).
        Tonemapping::None
    }
}

/// Mirror the user's chosen `starfield_ambient_floor_cd_per_m2` onto Bevy's
/// `GlobalAmbientLight` resource each frame `GameConfig` changes. This keeps
/// the ambient floor in lockstep with the rest of the photometric pipeline
/// (HDR, tonemap, exposure) — Bevy's GlobalAmbientLight is interpreted in
/// cd/m², so the value flows through unchanged.
fn apply_starfield_ambient_floor(
    config: Res<GameConfig>,
    mut ambient: ResMut<GlobalAmbientLight>,
) {
    if !config.is_changed() {
        return;
    }
    let target = config.graphics.lighting.starfield_ambient_floor_cd_per_m2;
    if (ambient.brightness - target).abs() > f32::EPSILON {
        ambient.brightness = target;
        tracing::info!(
            cd_per_m2 = target,
            "lighting/camera: applied starfield ambient floor"
        );
    }
}
