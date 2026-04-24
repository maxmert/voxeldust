//! Central game config — the single source of truth for all tunable
//! settings the user can change (eventually via an in-game settings UI
//! or a TOML file at startup).
//!
//! Design: one `GameConfig` Resource with typed sub-structs per
//! subsystem (`control`, future `graphics`, `audio`, `keybinds`, …).
//! Subsystems read from `Res<GameConfig>` — they never own their own
//! settings Resource. Adding a new setting = adding a field to the
//! matching sub-struct + reading it where it applies.
//!
//! Defaults live in each sub-struct's `Default` impl. A future
//! `load_from_toml` pass can override them from
//! `~/.voxeldust/client.toml`; this is forward-compatible — no
//! consumer changes required.

use bevy::prelude::*;

/// Root config resource. Register once via `GameConfigPlugin`; every
/// other plugin reads its own sub-struct from here.
#[derive(Resource, Debug, Clone, Default)]
pub struct GameConfig {
    pub control: ControlConfig,
    // Future:
    //   pub graphics: GraphicsConfig,
    //   pub audio: AudioConfig,
    //   pub keybinds: KeybindConfig,
}

/// Input / control settings — mouse, scroll, seat gains, etc.
/// All units / rationale in the field-level comments.
#[derive(Debug, Clone)]
pub struct ControlConfig {
    /// Radians of yaw/pitch per pixel of mouse motion at default
    /// sensitivity. Legacy voxydust baseline: `0.003` rad/px.
    pub mouse_sensitivity: f32,
    /// Invert horizontal mouse: flips the sign applied to yaw + ship
    /// yaw-rate seat signals. Future in-game settings UI toggles
    /// this; for now drives a config-file / CLI override.
    pub mouse_invert_x: bool,
    /// Invert vertical mouse (same semantics as X, for pitch).
    pub mouse_invert_y: bool,
    /// Per-notch scroll-wheel change for thrust limiter + scroll-
    /// bound seat bindings. Five notches cover the 0–1 range.
    pub scroll_thrust_step: f32,
    /// Gain factor for `PilotRates` → `PlayerInput.look_yaw/pitch`.
    /// The server's flight-computer block applies additional scaling
    /// on top of this, so the number is small.
    pub pilot_rate_gain: f32,
    /// Gain factor for seat-bound MouseMoveX/Y → `seat_values[i]`.
    /// These values feed the signal graph DIRECTLY (no extra server-
    /// side multiplier), so this needs to be about an order of
    /// magnitude larger than `pilot_rate_gain` to match the effort
    /// feel of held-key bindings that saturate at 1.0.
    pub seat_mouse_gain: f32,
}

impl Default for ControlConfig {
    fn default() -> Self {
        Self {
            mouse_sensitivity: 0.003,
            // User default: both axes inverted. Future settings UI
            // lets the player toggle either. Change the defaults here
            // to flip the out-of-box feel; toggle at runtime via the
            // Resource to preview.
            mouse_invert_x: true,
            mouse_invert_y: true,
            scroll_thrust_step: 0.2,
            pilot_rate_gain: 0.3,
            seat_mouse_gain: 3.0,
        }
    }
}

pub struct GameConfigPlugin;

impl Plugin for GameConfigPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GameConfig>();
    }
}
