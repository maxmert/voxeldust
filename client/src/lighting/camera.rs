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

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::GlobalAmbientLight;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::Hdr;

use crate::MainCamera;
use crate::config::GameConfig;

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
        }

        let tonemap = pick_tonemap(fidelity.tonemap_enabled, fidelity.hdr_enabled);
        entity_commands.insert(tonemap);

        if fidelity.bloom_enabled && fidelity.hdr_enabled {
            entity_commands.insert(Bloom {
                intensity: fidelity.bloom_intensity,
                ..Bloom::NATURAL
            });
        }

        // Bevy's PBR shader reads `Exposure` unconditionally for both HDR
        // and LDR render targets.
        entity_commands.insert(Exposure {
            ev100: fidelity.exposure_ev100_space,
        });

        tracing::info!(
            preset = ?fidelity.preset,
            hdr = fidelity.hdr_enabled,
            tonemap = ?tonemap,
            bloom = fidelity.bloom_enabled,
            shadows = fidelity.directional_shadows_enabled,
            ev100 = fidelity.exposure_ev100_space,
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
