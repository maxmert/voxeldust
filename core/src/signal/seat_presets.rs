//! Pre-made seat type templates and default custom block configurations.
//!
//! Presets provide starting-point bindings for different seat roles.
//! All bindings are editable by the player after placement.

use super::components::{AxisDirection, KeyMode, SeatInputSource};
use super::config::{
    AutopilotBlockConfig, EngineControllerConfig, FlightComputerConfig, HoverModuleConfig,
    SeatInputBindingConfig, WarpComputerConfig,
};
use super::types::SignalProperty;

/// Pre-made seat type templates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeatPreset {
    /// Full ship control: WASD, mouse, scroll, toggles for cruise/hover/autopilot/engine/warp.
    Pilot,
    /// No default bindings — player configures everything.
    Generic,
}

// ---------------------------------------------------------------------------
// Standard channel names — shared contract between seat presets and block defaults
// ---------------------------------------------------------------------------

pub const CH_THRUST_FORWARD: &str = "thrust-forward";
pub const CH_THRUST_REVERSE: &str = "thrust-reverse";
pub const CH_THRUST_RIGHT: &str = "thrust-right";
pub const CH_THRUST_LEFT: &str = "thrust-left";
pub const CH_THRUST_UP: &str = "thrust-up";
pub const CH_THRUST_DOWN: &str = "thrust-down";
pub const CH_TORQUE_YAW_CW: &str = "torque-yaw-cw";
pub const CH_TORQUE_YAW_CCW: &str = "torque-yaw-ccw";
pub const CH_TORQUE_PITCH_UP: &str = "torque-pitch-up";
pub const CH_TORQUE_PITCH_DOWN: &str = "torque-pitch-down";
pub const CH_TORQUE_ROLL_CW: &str = "torque-roll-cw";
pub const CH_TORQUE_ROLL_CCW: &str = "torque-roll-ccw";
pub const CH_THRUST_LIMITER: &str = "thrust-limiter";
pub const CH_CRUISE: &str = "cruise";
pub const CH_ATMO_COMP: &str = "atmo-comp";
pub const CH_AUTOPILOT_ENGAGE: &str = "autopilot-engage";
pub const CH_ENGINE_TOGGLE: &str = "engine-toggle";
pub const CH_FLIGHT_COMPUTER_TOGGLE: &str = "flight-computer-toggle";
pub const CH_WARP_CYCLE: &str = "warp-cycle";
pub const CH_WARP_ACCEPT: &str = "warp-accept";
pub const CH_WARP_CANCEL: &str = "warp-cancel";
pub const CH_PILOT_SEATED: &str = "pilot-seated";
pub const CH_SEATED: &str = "seated";

// ---------------------------------------------------------------------------
// Seat presets
// ---------------------------------------------------------------------------

impl SeatPreset {
    pub fn default_bindings(&self) -> Vec<SeatInputBindingConfig> {
        match self {
            Self::Pilot => pilot_bindings(),
            Self::Generic => Vec::new(),
        }
    }

    pub fn default_seated_channel(&self) -> &'static str {
        match self {
            Self::Pilot => CH_PILOT_SEATED,
            Self::Generic => CH_SEATED,
        }
    }

    /// Determine the seat preset for a given block ID.
    pub fn for_block(block_id: crate::block::BlockId) -> Self {
        use crate::block::BlockId;
        match block_id {
            BlockId::COCKPIT => Self::Pilot,
            BlockId::SEAT => Self::Generic,
            _ => Self::Generic,
        }
    }
}

fn key_binding(label: &str, key: &str, channel: &str) -> SeatInputBindingConfig {
    SeatInputBindingConfig {
        label: label.into(),
        source: SeatInputSource::Key,
        key_name: key.into(),
        key_mode: KeyMode::Momentary,
        axis_direction: AxisDirection::Positive,
        channel_name: channel.into(),
        property: SignalProperty::Throttle,
    }
}

fn toggle_binding(label: &str, key: &str, channel: &str) -> SeatInputBindingConfig {
    SeatInputBindingConfig {
        label: label.into(),
        source: SeatInputSource::Key,
        key_name: key.into(),
        key_mode: KeyMode::Toggle,
        axis_direction: AxisDirection::Positive,
        channel_name: channel.into(),
        property: SignalProperty::Throttle,
    }
}

fn mouse_x_binding(label: &str, dir: AxisDirection, channel: &str) -> SeatInputBindingConfig {
    SeatInputBindingConfig {
        label: label.into(),
        source: SeatInputSource::MouseMoveX,
        key_name: String::new(),
        key_mode: KeyMode::Momentary,
        axis_direction: dir,
        channel_name: channel.into(),
        property: SignalProperty::Throttle,
    }
}

fn mouse_y_binding(label: &str, dir: AxisDirection, channel: &str) -> SeatInputBindingConfig {
    SeatInputBindingConfig {
        label: label.into(),
        source: SeatInputSource::MouseMoveY,
        key_name: String::new(),
        key_mode: KeyMode::Momentary,
        axis_direction: dir,
        channel_name: channel.into(),
        property: SignalProperty::Throttle,
    }
}

fn pilot_bindings() -> Vec<SeatInputBindingConfig> {
    vec![
        // Linear thrust (WASD + Space/Ctrl)
        key_binding("Thrust Forward", "KeyW", CH_THRUST_FORWARD),
        key_binding("Thrust Reverse", "KeyS", CH_THRUST_REVERSE),
        key_binding("Thrust Right", "KeyD", CH_THRUST_RIGHT),
        key_binding("Thrust Left", "KeyA", CH_THRUST_LEFT),
        key_binding("Thrust Up", "Space", CH_THRUST_UP),
        key_binding("Thrust Down", "ControlLeft", CH_THRUST_DOWN),
        // Rotation (mouse axes, split into positive/negative)
        mouse_x_binding("Yaw CW", AxisDirection::Positive, CH_TORQUE_YAW_CW),
        mouse_x_binding("Yaw CCW", AxisDirection::Negative, CH_TORQUE_YAW_CCW),
        mouse_y_binding("Pitch Up", AxisDirection::Positive, CH_TORQUE_PITCH_UP),
        mouse_y_binding("Pitch Down", AxisDirection::Negative, CH_TORQUE_PITCH_DOWN),
        // Roll (Q/E)
        key_binding("Roll CW", "KeyE", CH_TORQUE_ROLL_CW),
        key_binding("Roll CCW", "KeyQ", CH_TORQUE_ROLL_CCW),
        // Thrust limiter (scroll wheel)
        SeatInputBindingConfig {
            label: "Thrust Limiter".into(),
            source: SeatInputSource::ScrollWheel,
            key_name: String::new(),
            key_mode: KeyMode::Momentary,
            axis_direction: AxisDirection::Both,
            channel_name: CH_THRUST_LIMITER.into(),
            property: SignalProperty::Throttle,
        },
        // System block toggles
        toggle_binding("Cruise", "KeyC", CH_CRUISE),
        toggle_binding("Atmo Comp", "KeyH", CH_ATMO_COMP),
        toggle_binding("Autopilot", "KeyT", CH_AUTOPILOT_ENGAGE),
        toggle_binding("Flight Computer", "F1", CH_FLIGHT_COMPUTER_TOGGLE),
        // Engine toggle (momentary — engine controller detects rising edge internally)
        key_binding("Engine Toggle", "KeyX", CH_ENGINE_TOGGLE),
        // Warp computer (momentary — warp computer detects rising edges)
        key_binding("Warp Cycle", "KeyG", CH_WARP_CYCLE),
        key_binding("Warp Accept", "Enter", CH_WARP_ACCEPT),
        key_binding("Warp Cancel", "Backspace", CH_WARP_CANCEL),
    ]
}

// ---------------------------------------------------------------------------
// Default configs for each custom ship system block
// ---------------------------------------------------------------------------

pub fn default_flight_computer_config() -> FlightComputerConfig {
    FlightComputerConfig {
        yaw_cw_channel: CH_TORQUE_YAW_CW.into(),
        yaw_ccw_channel: CH_TORQUE_YAW_CCW.into(),
        pitch_up_channel: CH_TORQUE_PITCH_UP.into(),
        pitch_down_channel: CH_TORQUE_PITCH_DOWN.into(),
        roll_cw_channel: CH_TORQUE_ROLL_CW.into(),
        roll_ccw_channel: CH_TORQUE_ROLL_CCW.into(),
        toggle_channel: CH_FLIGHT_COMPUTER_TOGGLE.into(),
        damping_gain: 0.6,
        dead_zone: 0.005,
        max_correction: 0.3,
    }
}

pub fn default_hover_module_config() -> HoverModuleConfig {
    HoverModuleConfig {
        thrust_forward_channel: CH_THRUST_FORWARD.into(),
        thrust_reverse_channel: CH_THRUST_REVERSE.into(),
        thrust_right_channel: CH_THRUST_RIGHT.into(),
        thrust_left_channel: CH_THRUST_LEFT.into(),
        thrust_up_channel: CH_THRUST_UP.into(),
        thrust_down_channel: CH_THRUST_DOWN.into(),
        yaw_cw_channel: CH_TORQUE_YAW_CW.into(),
        yaw_ccw_channel: CH_TORQUE_YAW_CCW.into(),
        pitch_up_channel: CH_TORQUE_PITCH_UP.into(),
        pitch_down_channel: CH_TORQUE_PITCH_DOWN.into(),
        roll_cw_channel: CH_TORQUE_ROLL_CW.into(),
        roll_ccw_channel: CH_TORQUE_ROLL_CCW.into(),
        activate_channel: CH_ATMO_COMP.into(),
        cutoff_channel: CH_ENGINE_TOGGLE.into(),
    }
}

pub fn default_autopilot_config() -> AutopilotBlockConfig {
    AutopilotBlockConfig {
        yaw_cw_channel: CH_TORQUE_YAW_CW.into(),
        yaw_ccw_channel: CH_TORQUE_YAW_CCW.into(),
        pitch_up_channel: CH_TORQUE_PITCH_UP.into(),
        pitch_down_channel: CH_TORQUE_PITCH_DOWN.into(),
        roll_cw_channel: CH_TORQUE_ROLL_CW.into(),
        roll_ccw_channel: CH_TORQUE_ROLL_CCW.into(),
        engage_channel: CH_AUTOPILOT_ENGAGE.into(),
    }
}

pub fn default_warp_computer_config() -> WarpComputerConfig {
    WarpComputerConfig {
        cycle_channel: CH_WARP_CYCLE.into(),
        accept_channel: CH_WARP_ACCEPT.into(),
        cancel_channel: CH_WARP_CANCEL.into(),
    }
}

pub fn default_engine_controller_config() -> EngineControllerConfig {
    EngineControllerConfig {
        thrust_forward_channel: CH_THRUST_FORWARD.into(),
        thrust_reverse_channel: CH_THRUST_REVERSE.into(),
        thrust_right_channel: CH_THRUST_RIGHT.into(),
        thrust_left_channel: CH_THRUST_LEFT.into(),
        thrust_up_channel: CH_THRUST_UP.into(),
        thrust_down_channel: CH_THRUST_DOWN.into(),
        yaw_cw_channel: CH_TORQUE_YAW_CW.into(),
        yaw_ccw_channel: CH_TORQUE_YAW_CCW.into(),
        pitch_up_channel: CH_TORQUE_PITCH_UP.into(),
        pitch_down_channel: CH_TORQUE_PITCH_DOWN.into(),
        roll_cw_channel: CH_TORQUE_ROLL_CW.into(),
        roll_ccw_channel: CH_TORQUE_ROLL_CCW.into(),
        toggle_channel: CH_ENGINE_TOGGLE.into(),
    }
}
