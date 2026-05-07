//! Cross-cutting throttle modulation for signal-driven emitters.
//!
//! Signal-modulated visual effects (thruster exhaust glow today;
//! future lamps with subscribed channels) all share the same data
//! flow:
//!
//! 1. Pilot's seat key publishes (W → `"thrust-forward" = 1.0`).
//! 2. Server propagates through the signal graph and emits a
//!    delta-encoded `ServerMsg::HudSignalDelta` over TCP.
//! 3. Client `signal_registry::drain_hud_signal_deltas` applies the
//!    delta to [`SignalRegistry`].
//! 4. Per-frame [`apply_throttle_modulation`] reads
//!    [`SignalRegistry`] directly and scales every
//!    [`ThrottleModulatedLight`] / [`ThrottleModulatedProxy`] by the
//!    current channel value.
//!
//! Phase 4.4.5: this module previously kept its own `ThrottleSignals`
//! mirror populated from the legacy `WorldState.hud_signals` UDP
//! field. The mirror went away — `SignalRegistry` is already the
//! single source of truth for HUD-channel state, so reading from it
//! directly is more DRY and removes one stale-update path.
//!
//! Per-emitter channel resolution lives in the calling block-kind
//! module (e.g. `full_block::thruster_channel_for_facing` for
//! thrusters). This module is purely the cross-cutting plumbing that
//! wires a channel value to a visual property.

use bevy::color::LinearRgba;
use bevy::light::{PointLight, SpotLight};
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;

use crate::hud::signal_registry::SignalRegistry;

/// Component on a `PointLight` / `SpotLight` whose intensity should
/// scale with a HUD-tracked throttle channel.
#[derive(Component)]
pub struct ThrottleModulatedLight {
    pub channel: String,
    pub base_intensity: f32,
}

/// Component on an emissive proxy whose `StandardMaterial.emissive`
/// should scale with a HUD-tracked throttle channel. Material handle
/// is looked up via the entity's `MeshMaterial3d<StandardMaterial>`.
#[derive(Component)]
pub struct ThrottleModulatedProxy {
    pub channel: String,
    pub base_emissive: LinearRgba,
}

/// Project a registry value onto the `[0, 1]` modulation range.
/// Bool/State values cast to f32 then clamp; Text values fall to 0.0
/// (text channels can't drive a brightness multiplier).
fn channel_modulation(registry: &SignalRegistry, channel: &str) -> f32 {
    registry
        .get(channel)
        .map(|s| s.value.as_f32())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
}

/// Per-frame: scale every `ThrottleModulatedLight`'s `intensity` and
/// every `ThrottleModulatedProxy`'s emissive by the current channel
/// value. `Assets<StandardMaterial>::get_mut` triggers the engine's
/// per-frame asset diff and re-uploads only the changed materials —
/// for the typical ~60 thrusters of a starter ship the cost is
/// negligible.
pub(super) fn apply_throttle_modulation(
    registry: Res<SignalRegistry>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut lights: Query<(&ThrottleModulatedLight, &mut PointLight)>,
    mut spotlights: Query<(&ThrottleModulatedLight, &mut SpotLight)>,
    proxies: Query<(&ThrottleModulatedProxy, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (modulator, mut light) in &mut lights {
        let value = channel_modulation(&registry, &modulator.channel);
        light.intensity = modulator.base_intensity * value;
    }
    for (modulator, mut light) in &mut spotlights {
        let value = channel_modulation(&registry, &modulator.channel);
        light.intensity = modulator.base_intensity * value;
    }
    for (modulator, mat_handle) in &proxies {
        let value = channel_modulation(&registry, &modulator.channel);
        let Some(material) = materials.get_mut(&mat_handle.0) else {
            continue;
        };
        material.emissive = LinearRgba::new(
            modulator.base_emissive.red * value,
            modulator.base_emissive.green * value,
            modulator.base_emissive.blue * value,
            modulator.base_emissive.alpha,
        );
    }
}
