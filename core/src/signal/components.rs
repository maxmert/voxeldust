//! Signal-related ECS components for functional block entities.

use bevy_ecs::prelude::*;

use super::converter::SignalRule;
use super::types::SignalProperty;

/// A single publish binding: this block writes a property to a named channel.
#[derive(Clone, Debug)]
pub struct PublishBinding {
    /// Channel name to publish to.
    pub channel_name: String,
    /// Which property of this block to read and publish.
    pub property: SignalProperty,
}

/// A single subscribe binding: this block reads a named channel and applies to a property.
#[derive(Clone, Debug)]
pub struct SubscribeBinding {
    /// Channel name to subscribe from.
    pub channel_name: String,
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

/// Pilot seat channel mapping — maps control inputs to signal channels.
/// Only attached to entities whose FunctionalBlockKind == Seat.
#[derive(Component, Clone, Debug)]
pub struct SeatChannelMapping {
    pub bindings: Vec<SeatInputBinding>,
}

/// Typed controls available to seat occupants.
/// Each seat type uses a subset. New seat types add new variants.
/// The server maps each control to a concrete input source (key, mouse axis, slider).
/// All values are unipolar 0.0–1.0 (thrusters fire in one direction only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SeatControl {
    // Linear thrust (6 directions, each 0.0–1.0)
    ThrustForward   = 0,   // W key
    ThrustReverse   = 1,   // S key
    ThrustRight     = 2,   // D key
    ThrustLeft      = 3,   // A key
    ThrustUp        = 4,   // Space
    ThrustDown      = 5,   // Ctrl
    // Rotation (unipolar — separate CW/CCW channels)
    TorqueYawCW     = 6,   // Mouse right
    TorqueYawCCW    = 7,   // Mouse left
    TorquePitchUp   = 8,   // Mouse up
    TorquePitchDown = 9,   // Mouse down
    // Thrust limiter (mouse wheel, 0.0–1.0)
    ThrustLimiter   = 10,
    // Roll (Q/E keys when piloting)
    TorqueRollCW    = 11,  // E key (piloting)
    TorqueRollCCW   = 12,  // Q key (piloting)
    // Future: Gunner controls (20-29)
    // AimX           = 20,
    // AimY           = 21,
    // FirePrimary    = 22,
    // FireSecondary  = 23,
    // Future: Engineer controls (30-39)
    // PowerWeapons   = 30,
    // PowerShields   = 31,
    // PowerEngines   = 32,
}

impl SeatControl {
    /// Human-readable label for the UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::ThrustForward   => "Thrust Forward",
            Self::ThrustReverse   => "Thrust Reverse",
            Self::ThrustRight     => "Thrust Right",
            Self::ThrustLeft      => "Thrust Left",
            Self::ThrustUp        => "Thrust Up",
            Self::ThrustDown      => "Thrust Down",
            Self::TorqueYawCW     => "Yaw CW",
            Self::TorqueYawCCW    => "Yaw CCW",
            Self::TorquePitchUp   => "Pitch Up",
            Self::TorquePitchDown => "Pitch Down",
            Self::ThrustLimiter   => "Thrust Limiter",
            Self::TorqueRollCW    => "Roll CW",
            Self::TorqueRollCCW   => "Roll CCW",
        }
    }

    /// All controls valid for a given functional block kind.
    pub fn available_for_kind(_kind: crate::block::FunctionalBlockKind) -> &'static [SeatControl] {
        // All pilot controls — future seat types will return different subsets.
        &[
            SeatControl::ThrustForward, SeatControl::ThrustReverse,
            SeatControl::ThrustRight, SeatControl::ThrustLeft,
            SeatControl::ThrustUp, SeatControl::ThrustDown,
            SeatControl::TorqueYawCW, SeatControl::TorqueYawCCW,
            SeatControl::TorquePitchUp, SeatControl::TorquePitchDown,
            SeatControl::ThrustLimiter,
            SeatControl::TorqueRollCW, SeatControl::TorqueRollCCW,
        ]
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0  => Some(Self::ThrustForward),
            1  => Some(Self::ThrustReverse),
            2  => Some(Self::ThrustRight),
            3  => Some(Self::ThrustLeft),
            4  => Some(Self::ThrustUp),
            5  => Some(Self::ThrustDown),
            6  => Some(Self::TorqueYawCW),
            7  => Some(Self::TorqueYawCCW),
            8  => Some(Self::TorquePitchUp),
            9  => Some(Self::TorquePitchDown),
            10 => Some(Self::ThrustLimiter),
            11 => Some(Self::TorqueRollCW),
            12 => Some(Self::TorqueRollCCW),
            _  => None,
        }
    }
}

/// Maps a seat control to a signal channel.
#[derive(Clone, Debug)]
pub struct SeatInputBinding {
    /// Which typed control this binding reads.
    pub control: SeatControl,
    /// Channel to publish to (free-text, user-defined).
    pub channel_name: String,
    /// What type of value to publish.
    pub property: SignalProperty,
}

impl Default for SeatChannelMapping {
    fn default() -> Self {
        use SeatControl::*;
        Self {
            bindings: vec![
                SeatInputBinding { control: ThrustForward,   channel_name: "thrust-forward".into(),      property: SignalProperty::Throttle },
                SeatInputBinding { control: ThrustReverse,    channel_name: "thrust-reverse".into(),      property: SignalProperty::Throttle },
                SeatInputBinding { control: ThrustRight,      channel_name: "thrust-right".into(),        property: SignalProperty::Throttle },
                SeatInputBinding { control: ThrustLeft,       channel_name: "thrust-left".into(),         property: SignalProperty::Throttle },
                SeatInputBinding { control: ThrustUp,         channel_name: "thrust-up".into(),           property: SignalProperty::Throttle },
                SeatInputBinding { control: ThrustDown,       channel_name: "thrust-down".into(),         property: SignalProperty::Throttle },
                SeatInputBinding { control: TorqueYawCW,      channel_name: "torque-yaw-cw".into(),      property: SignalProperty::Throttle },
                SeatInputBinding { control: TorqueYawCCW,     channel_name: "torque-yaw-ccw".into(),     property: SignalProperty::Throttle },
                SeatInputBinding { control: TorquePitchUp,    channel_name: "torque-pitch-up".into(),    property: SignalProperty::Throttle },
                SeatInputBinding { control: TorquePitchDown,  channel_name: "torque-pitch-down".into(),  property: SignalProperty::Throttle },
                SeatInputBinding { control: ThrustLimiter,    channel_name: "thrust-limiter".into(),     property: SignalProperty::Throttle },
                SeatInputBinding { control: TorqueRollCW,     channel_name: "torque-roll-cw".into(),     property: SignalProperty::Throttle },
                SeatInputBinding { control: TorqueRollCCW,    channel_name: "torque-roll-ccw".into(),    property: SignalProperty::Throttle },
            ],
        }
    }
}
