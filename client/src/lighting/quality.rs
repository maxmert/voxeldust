//! Lighting fidelity apply system.
//!
//! Reads `Res<GameConfig>` (specifically `graphics.lighting`) and
//! reconfigures every camera / light component on `Changed<GameConfig>`.
//! This is the only place rendering-fidelity knobs (cascade count, AA
//! mode, post-FX gates, etc.) are translated into Bevy components.
//!
//! Phase 0 wires the plugin with a no-op apply (just structure / change
//! detection). Phase 1 fleshes out the apply path for CSM, AA, post-FX,
//! atmosphere method, and IBL. Subsequent phases extend the apply system
//! as new components join the camera stack.
//!
//! Determinism note: this module *intentionally* contains numeric
//! literals describing rendering knobs (e.g., the no-op apply doesn't
//! contain literals yet, but Phase 1+ will reference values from
//! `config::graphics::LightingFidelity`). Per the plan, lighting
//! *values* (sun colour, atmosphere parameters, …) never live here —
//! those are physics, broadcast by the server.

use bevy::prelude::*;

use crate::config::GameConfig;

pub struct LightingQualityPlugin;

impl Plugin for LightingQualityPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, log_lighting_fidelity_changes);
    }
}

/// Phase 0 stub: detects `GameConfig` changes and logs the new fidelity
/// preset. Phase 1+ replaces the body with the real apply path.
fn log_lighting_fidelity_changes(config: Res<GameConfig>) {
    if !config.is_changed() {
        return;
    }
    let f = &config.graphics.lighting;
    tracing::info!(
        preset = ?f.preset,
        cascade_count = f.cascade_count,
        shadow_map_size = f.shadow_map_size,
        cascade_max_distance_m = f.cascade_max_distance_m,
        aa_mode = ?f.aa_mode,
        atmosphere_method = ?f.atmosphere_method,
        ssao = f.ssao_enabled,
        ssr = f.ssr_enabled,
        volumetric_fog = f.volumetric_fog_enabled,
        "lighting fidelity changed"
    );
}
