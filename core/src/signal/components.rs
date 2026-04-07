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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SeatControl {
    // Pilot controls (WASD, mouse)
    ThrustForward  = 0,
    ThrustLateral  = 1,
    ThrustVertical = 2,
    TorqueYaw      = 3,
    TorquePitch    = 4,
    // Future: Gunner controls (10-19)
    // AimX           = 10,
    // AimY           = 11,
    // FirePrimary    = 12,
    // FireSecondary  = 13,
    // Future: Engineer controls (20-29)
    // PowerWeapons   = 20,
    // PowerShields   = 21,
    // PowerEngines   = 22,
    // CoolantLevel   = 23,
}

impl SeatControl {
    /// Human-readable label for the UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::ThrustForward  => "Thrust Forward",
            Self::ThrustLateral  => "Thrust Lateral",
            Self::ThrustVertical => "Thrust Vertical",
            Self::TorqueYaw      => "Torque Yaw",
            Self::TorquePitch    => "Torque Pitch",
        }
    }

    /// All controls valid for a given functional block kind.
    /// For now all seats are pilot seats.
    pub fn available_for_kind(_kind: crate::block::FunctionalBlockKind) -> &'static [SeatControl] {
        &[
            SeatControl::ThrustForward,
            SeatControl::ThrustLateral,
            SeatControl::ThrustVertical,
            SeatControl::TorqueYaw,
            SeatControl::TorquePitch,
        ]
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::ThrustForward),
            1 => Some(Self::ThrustLateral),
            2 => Some(Self::ThrustVertical),
            3 => Some(Self::TorqueYaw),
            4 => Some(Self::TorquePitch),
            _ => None,
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
        Self {
            bindings: vec![
                SeatInputBinding { control: SeatControl::ThrustForward,  channel_name: "thrust-forward".into(),  property: SignalProperty::Throttle },
                SeatInputBinding { control: SeatControl::ThrustLateral,  channel_name: "thrust-lateral".into(),  property: SignalProperty::Throttle },
                SeatInputBinding { control: SeatControl::ThrustVertical, channel_name: "thrust-vertical".into(), property: SignalProperty::Throttle },
                SeatInputBinding { control: SeatControl::TorqueYaw,      channel_name: "torque-yaw".into(),      property: SignalProperty::Throttle },
                SeatInputBinding { control: SeatControl::TorquePitch,    channel_name: "torque-pitch".into(),    property: SignalProperty::Throttle },
            ],
        }
    }
}
