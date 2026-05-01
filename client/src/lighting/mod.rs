//! Voxeldust lighting pipeline.
//!
//! This module is the *consumer* of celestial-body physics state broadcast
//! by the server (system-shard's authoritative computation, propagated to
//! planet-shard / ship-shard, broadcast to clients via `WorldState.bodies`).
//! The client never re-derives stellar / geophysical / rotation state from
//! seeds — `tests/no_client_seed_derivation.rs` enforces this statically.
//!
//! Phase layout (per `/Users/maxim/.claude/plans/i-need-you-to-precious-bumblebee.md`):
//!
//! - `quality.rs`   — LightingFidelity preset apply system (rendering knobs).
//! - `solar.rs`     — Reads broadcast stellar state, configures DirectionalLight
//!                    + per-camera sun direction (preserved from the original
//!                    diagnostic implementation through Phase 0; replaced with
//!                    physics-derived colour & illuminance in Phase 1).
//! - `camera.rs`    — (Phase 1) HDR + Tonemap + Bloom + Exposure component stack.
//! - `atmosphere.rs`— (Phase 3) Per-planet ScatteringMedium swap.
//! - `ibl.rs`       — (Phase 2/3) IBL state machine.
//! - `eclipse.rs`   — (Phase 6) Eclipse occlusion material extension.
//! - `local.rs`     — (Phase 8) Block-driven PointLight / SpotLight.
//! - `volumetric.rs`— (Phase 5) VolumetricFog + VolumetricLight.
//! - `rotation.rs`  — (Phase 4) Reads broadcast rotation_params, animates
//!                    ShardOrigin.rotation per frame from `rotation_at(params, t)`.

use bevy::prelude::*;

pub mod atmosphere;
pub mod camera;
pub mod eclipse;
pub mod eclipse_deferred;
pub mod emitters;
pub mod ibl;
pub mod quality;
pub mod rotation;
pub mod solar;

pub use solar::{SolarLight, SolarLightPlugin};

/// Parent lighting plugin. Wires every lighting sub-plugin so callers add a
/// single plugin in `main.rs`.
///
/// **Eclipse plugin must be added before any system that touches the
/// chunk material handle**, because it registers
/// `MaterialPlugin::<ExtendedMaterial<StandardMaterial, EclipseExt>>`
/// (the asset type for `ChunkMaterial`). Without that registration,
/// `ResMut<Assets<ChunkMaterial>>` queries fail and the chunk-stream
/// systems can't allocate the shared material handle. We add it
/// first in this plugin chain — the chunk plugin (`ChunkStreamPlugin`)
/// is added by `main.rs` after `LightingPlugin`.
pub struct LightingPlugin;

impl Plugin for LightingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(eclipse::EclipsePlugin)
            .add_plugins(quality::LightingQualityPlugin)
            .add_plugins(camera::LightingCameraPlugin)
            .add_plugins(solar::SolarLightPlugin)
            .add_plugins(ibl::IblPlugin)
            .add_plugins(atmosphere::AtmospherePlugin)
            .add_plugins(rotation::PlanetRotationPlugin)
            .add_plugins(emitters::LocalLightingPlugin);
    }
}
