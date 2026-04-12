//! Signal-related ECS components for functional block entities.
//!
//! Runtime binding structs use `ChannelId` for O(1) lookups.  String-based
//! channel names are resolved to IDs at the config/spawn boundary and live
//! only in `SignalChannelTable::name_to_id`.

use bevy_ecs::prelude::*;
use glam::DVec3;

use super::channel::ChannelId;
use super::converter::SignalRule;
use super::types::SignalProperty;

// ---------------------------------------------------------------------------
// Generic signal bindings (publish / subscribe)
// ---------------------------------------------------------------------------

/// A single publish binding: this block writes a property to a channel.
#[derive(Clone, Debug)]
pub struct PublishBinding {
    /// Resolved channel ID (O(1) table lookup).
    pub channel_id: ChannelId,
    /// Which property of this block to read and publish.
    pub property: SignalProperty,
}

/// A single subscribe binding: this block reads a channel and applies to a property.
#[derive(Clone, Debug)]
pub struct SubscribeBinding {
    /// Resolved channel ID (O(1) table lookup).
    pub channel_id: ChannelId,
    /// Which property of this block to drive from the channel value.
    pub property: SignalProperty,
}

/// Channels this functional block publishes to.
/// Attached to any functional block entity that produces signal data.
#[derive(Component, Default, Clone, Debug)]
pub struct SignalPublisher {
    pub bindings: Vec<PublishBinding>,
}

/// Channels this functional block subscribes to.
/// Attached to any functional block entity that consumes signal data.
#[derive(Component, Default, Clone, Debug)]
pub struct SignalSubscriber {
    pub bindings: Vec<SubscribeBinding>,
}

/// Configuration for a Signal Converter block — condition → action rules.
/// Only attached to entities whose FunctionalBlockKind == SignalConverter.
#[derive(Component, Default, Clone, Debug)]
pub struct SignalConverterConfig {
    pub rules: Vec<SignalRule>,
}

// ---------------------------------------------------------------------------
// Generic seat system — shard-agnostic, works anywhere blocks are supported
// ---------------------------------------------------------------------------

/// Physical input source type for a seat binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SeatInputSource {
    /// Keyboard key or mouse button (binary: 0.0 released, 1.0 held).
    Key = 0,
    /// Horizontal mouse movement (spring-centered continuous axis).
    MouseMoveX = 1,
    /// Vertical mouse movement (spring-centered continuous axis).
    MouseMoveY = 2,
    /// Mouse scroll wheel (accumulative, persists between frames).
    ScrollWheel = 3,
}

impl SeatInputSource {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Key),
            1 => Some(Self::MouseMoveX),
            2 => Some(Self::MouseMoveY),
            3 => Some(Self::ScrollWheel),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Key => "Key",
            Self::MouseMoveX => "Mouse X",
            Self::MouseMoveY => "Mouse Y",
            Self::ScrollWheel => "Scroll",
        }
    }
}

/// Key activation mode (only meaningful for `SeatInputSource::Key`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KeyMode {
    /// Value = 1.0 while held, 0.0 when released.
    Momentary = 0,
    /// Each press toggles between 0.0 and 1.0.
    Toggle = 1,
}

impl KeyMode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Momentary),
            1 => Some(Self::Toggle),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Momentary => "Hold",
            Self::Toggle => "Toggle",
        }
    }
}

/// Direction filter for mouse axis bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AxisDirection {
    /// Only positive values (right / up). Negative clamped to 0.
    Positive = 0,
    /// Only negative values, output made positive (left / down).
    Negative = 1,
    /// Full bipolar [-1, 1].
    Both = 2,
}

impl AxisDirection {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Positive),
            1 => Some(Self::Negative),
            2 => Some(Self::Both),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Positive => "\u{2192} Positive",
            Self::Negative => "\u{2190} Negative",
            Self::Both => "\u{2194} Both",
        }
    }

    /// Apply direction filter to a raw bipolar value.
    pub fn apply(self, raw: f32) -> f32 {
        match self {
            Self::Positive => raw.max(0.0),
            Self::Negative => (-raw).max(0.0),
            Self::Both => raw,
        }
    }
}

/// A single input-to-channel binding in a seat.
#[derive(Clone, Debug)]
pub struct SeatInputBinding {
    /// Human-readable label (e.g., "Thrust Forward").
    pub label: String,
    /// What physical input triggers this binding.
    pub source: SeatInputSource,
    /// Key name for Key source (e.g., "KeyW", "Space", "MouseLeft"). Empty for axes/scroll.
    pub key_name: String,
    /// Key activation mode (only for Key source).
    pub key_mode: KeyMode,
    /// Direction filter (only for MouseMoveX/Y).
    pub axis_direction: AxisDirection,
    /// Target signal channel (resolved ID).
    pub channel_id: ChannelId,
    /// What property to publish.
    pub property: SignalProperty,
}

/// Seat channel mapping component. Maps physical inputs to signal channels.
/// Shard-agnostic — works in any shard where blocks are supported.
/// Attached to entities whose FunctionalBlockKind == Seat.
#[derive(Component, Clone, Debug)]
pub struct SeatChannelMapping {
    pub bindings: Vec<SeatInputBinding>,
    /// Channel to emit 1.0 when player is seated, 0.0 when not.
    pub seated_channel_id: Option<ChannelId>,
}

// ---------------------------------------------------------------------------
// Custom ship system block components — each fully independent
// ---------------------------------------------------------------------------

/// Flight computer — angular velocity damping when pilot input is near zero.
/// Each channel field is explicit — no implicit ordering.
#[derive(Component, Clone, Debug)]
pub struct FlightComputerState {
    pub yaw_cw_channel: Option<ChannelId>,
    pub yaw_ccw_channel: Option<ChannelId>,
    pub pitch_up_channel: Option<ChannelId>,
    pub pitch_down_channel: Option<ChannelId>,
    pub roll_cw_channel: Option<ChannelId>,
    pub roll_ccw_channel: Option<ChannelId>,
    pub damping_gain: f32,
    pub dead_zone: f32,
    pub max_correction: f32,
}

impl Default for FlightComputerState {
    fn default() -> Self {
        Self {
            yaw_cw_channel: None, yaw_ccw_channel: None,
            pitch_up_channel: None, pitch_down_channel: None,
            roll_cw_channel: None, roll_ccw_channel: None,
            damping_gain: 0.6,
            dead_zone: 0.005,
            max_correction: 0.3,
        }
    }
}

/// Hover module — 6-DOF hover: attitude hold + gravity compensation + velocity damping.
#[derive(Component, Clone, Debug)]
pub struct HoverModuleState {
    // Thrust channels:
    pub thrust_forward_channel: Option<ChannelId>,
    pub thrust_reverse_channel: Option<ChannelId>,
    pub thrust_right_channel: Option<ChannelId>,
    pub thrust_left_channel: Option<ChannelId>,
    pub thrust_up_channel: Option<ChannelId>,
    pub thrust_down_channel: Option<ChannelId>,
    // Rotation channels:
    pub yaw_cw_channel: Option<ChannelId>,
    pub yaw_ccw_channel: Option<ChannelId>,
    pub pitch_up_channel: Option<ChannelId>,
    pub pitch_down_channel: Option<ChannelId>,
    pub roll_cw_channel: Option<ChannelId>,
    pub roll_ccw_channel: Option<ChannelId>,
    // Activation:
    pub activate_channel: Option<ChannelId>,
    pub cutoff_channel: Option<ChannelId>,
    // Runtime state:
    pub was_active: bool,
    pub captured_heading: DVec3,
    pub prev_velocity_local: DVec3,
}

impl Default for HoverModuleState {
    fn default() -> Self {
        Self {
            thrust_forward_channel: None, thrust_reverse_channel: None,
            thrust_right_channel: None, thrust_left_channel: None,
            thrust_up_channel: None, thrust_down_channel: None,
            yaw_cw_channel: None, yaw_ccw_channel: None,
            pitch_up_channel: None, pitch_down_channel: None,
            roll_cw_channel: None, roll_ccw_channel: None,
            activate_channel: None,
            cutoff_channel: None,
            was_active: false,
            captured_heading: DVec3::NEG_Z,
            prev_velocity_local: DVec3::ZERO,
        }
    }
}

/// Autopilot — target-tracking, publishes steering commands to rotation channels.
#[derive(Component, Clone, Debug)]
pub struct AutopilotBlockState {
    // Rotation channels (autopilot writes steering commands here):
    pub yaw_cw_channel: Option<ChannelId>,
    pub yaw_ccw_channel: Option<ChannelId>,
    pub pitch_up_channel: Option<ChannelId>,
    pub pitch_down_channel: Option<ChannelId>,
    pub roll_cw_channel: Option<ChannelId>,
    pub roll_ccw_channel: Option<ChannelId>,
    // Activation:
    pub engage_channel: Option<ChannelId>,
    // Runtime state:
    pub target_body_id: Option<u32>,
    pub pending_cmd: Option<(u32, u8)>,
    pub prev_engage_value: f32,
}

impl Default for AutopilotBlockState {
    fn default() -> Self {
        Self {
            yaw_cw_channel: None, yaw_ccw_channel: None,
            pitch_up_channel: None, pitch_down_channel: None,
            roll_cw_channel: None, roll_ccw_channel: None,
            engage_channel: None,
            target_body_id: None,
            pending_cmd: None,
            prev_engage_value: 0.0,
        }
    }
}

/// Warp computer — target selection and warp initiation.
#[derive(Component, Clone, Debug, Default)]
pub struct WarpComputerState {
    pub target_channel: Option<ChannelId>,
    pub confirm_channel: Option<ChannelId>,
    pub target_star_index: Option<u32>,
    pub pending_cmd: Option<u32>,
    pub prev_target_value: f32,
    pub prev_confirm_value: f32,
}

/// Engine controller — master on/off toggle for all propulsion.
/// Rising edge on toggle channel flips engines_on. When off, zeros all managed channels.
#[derive(Component, Clone, Debug)]
pub struct EngineControllerState {
    // All channels zeroed when engines are off:
    pub thrust_forward_channel: Option<ChannelId>,
    pub thrust_reverse_channel: Option<ChannelId>,
    pub thrust_right_channel: Option<ChannelId>,
    pub thrust_left_channel: Option<ChannelId>,
    pub thrust_up_channel: Option<ChannelId>,
    pub thrust_down_channel: Option<ChannelId>,
    pub yaw_cw_channel: Option<ChannelId>,
    pub yaw_ccw_channel: Option<ChannelId>,
    pub pitch_up_channel: Option<ChannelId>,
    pub pitch_down_channel: Option<ChannelId>,
    pub roll_cw_channel: Option<ChannelId>,
    pub roll_ccw_channel: Option<ChannelId>,
    // Toggle:
    pub toggle_channel: Option<ChannelId>,
    pub engines_on: bool,
    pub prev_toggle_value: f32,
}

impl Default for EngineControllerState {
    fn default() -> Self {
        Self {
            thrust_forward_channel: None, thrust_reverse_channel: None,
            thrust_right_channel: None, thrust_left_channel: None,
            thrust_up_channel: None, thrust_down_channel: None,
            yaw_cw_channel: None, yaw_ccw_channel: None,
            pitch_up_channel: None, pitch_down_channel: None,
            roll_cw_channel: None, roll_ccw_channel: None,
            toggle_channel: None,
            engines_on: true,
            prev_toggle_value: 0.0,
        }
    }
}
