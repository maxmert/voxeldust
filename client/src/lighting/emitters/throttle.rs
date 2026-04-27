//! Cross-cutting throttle modulation for signal-driven emitters.
//!
//! Signal-modulated visual effects (thruster exhaust glow today; future
//! lamps with subscribed channels in Slice C) all share the same data
//! flow:
//!
//! 1. Pilot's seat key publishes (W → `"thrust-forward" = 1.0`).
//! 2. Server propagates through the signal graph and broadcasts the
//!    per-channel value in `WorldStateData.hud_signals`.
//! 3. Client [`update_throttle_signals`] ingests this snapshot into the
//!    [`ThrottleSignals`] resource each tick.
//! 4. Per-frame [`apply_throttle_modulation`] queries every
//!    [`ThrottleModulatedLight`] / [`ThrottleModulatedProxy`] and scales
//!    its intensity / emissive by the channel's current value.
//!
//! Per-emitter channel resolution lives in the calling block-kind
//! module (e.g. `full_block::thruster_channel_for_facing` for thrusters
//! today). This module is purely the cross-cutting plumbing that wires
//! a channel value to a visual property.

use std::collections::HashMap;

use bevy::color::LinearRgba;
use bevy::light::{PointLight, SpotLight};
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;

use voxeldust_core::client_message::HudSignalValue;

use crate::net::{GameEvent, NetEvent};

/// Mutable per-frame intensity / emissive multiplier sourced from the
/// broadcast `WorldStateData.hud_signals`. Empty until the first
/// WorldState arrives; values are clamped to `[0.0, 1.0]` on read so
/// downstream multiplications can't blow up the lumens scale.
#[derive(Resource, Default)]
pub struct ThrottleSignals {
    by_channel: HashMap<String, f32>,
}

impl ThrottleSignals {
    /// Look up the current throttle value for a channel. Returns `0.0`
    /// for absent channels — meaning lights without a signal default
    /// to "off" rather than "full bright".
    pub fn get(&self, channel: &str) -> f32 {
        self.by_channel
            .get(channel)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0)
    }
}

/// Component on a `PointLight` / `SpotLight` whose intensity should
/// scale with a broadcast throttle channel.
#[derive(Component)]
pub struct ThrottleModulatedLight {
    pub channel: String,
    pub base_intensity: f32,
}

/// Component on an emissive proxy whose `StandardMaterial.emissive`
/// should scale with a broadcast throttle channel. Material handle is
/// looked up via the entity's `MeshMaterial3d<StandardMaterial>`.
#[derive(Component)]
pub struct ThrottleModulatedProxy {
    pub channel: String,
    pub base_emissive: LinearRgba,
}

/// Drain `WorldStateData.hud_signals` from the network event stream
/// each tick into the `ThrottleSignals` resource. Bool / state values
/// project onto the `[0, 1]` range so emergency strobes (Bool) and
/// step-state inputs participate cleanly. Text values are skipped.
pub(super) fn update_throttle_signals(
    mut events: MessageReader<GameEvent>,
    mut signals: ResMut<ThrottleSignals>,
) {
    for GameEvent(ev) in events.read() {
        let NetEvent::WorldState(ws) = ev else { continue };
        for entry in &ws.hud_signals {
            let value = match &entry.value {
                HudSignalValue::Bool(b) => {
                    if *b {
                        SIGNAL_BOOL_TRUE
                    } else {
                        SIGNAL_BOOL_FALSE
                    }
                }
                HudSignalValue::Float(f) => *f,
                HudSignalValue::State(s) => *s as f32,
                HudSignalValue::Text(_) => continue,
            };
            signals
                .by_channel
                .insert(entry.channel_name.clone(), value);
        }
    }
}

/// Per-frame: scale every `ThrottleModulatedLight`'s `intensity` and
/// every `ThrottleModulatedProxy`'s emissive by the current channel
/// value. `Assets<StandardMaterial>::get_mut` triggers the engine's
/// per-frame asset diff and re-uploads only the changed materials —
/// for the typical ~60 thrusters of a starter ship the cost is
/// negligible.
pub(super) fn apply_throttle_modulation(
    signals: Res<ThrottleSignals>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut lights: Query<(&ThrottleModulatedLight, &mut PointLight)>,
    mut spotlights: Query<(&ThrottleModulatedLight, &mut SpotLight)>,
    proxies: Query<(&ThrottleModulatedProxy, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (modulator, mut light) in &mut lights {
        let value = signals.get(&modulator.channel);
        light.intensity = modulator.base_intensity * value;
    }
    for (modulator, mut light) in &mut spotlights {
        let value = signals.get(&modulator.channel);
        light.intensity = modulator.base_intensity * value;
    }
    for (modulator, mat_handle) in &proxies {
        let value = signals.get(&modulator.channel);
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

/// Numeric values used to project boolean throttle signals onto the
/// `[0, 1]` modulation range. A `true` boolean lights the modulated
/// emitter at full brightness; `false` darkens it to zero.
const SIGNAL_BOOL_TRUE: f32 = 1.0;
const SIGNAL_BOOL_FALSE: f32 = 0.0;
