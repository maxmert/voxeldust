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

/// Maps a pilot input (key/axis) to a signal channel publish.
#[derive(Clone, Debug)]
pub struct SeatInputBinding {
    /// Human-readable input name (e.g., "W", "thrust_forward", "button_1").
    pub input_name: String,
    /// Channel to publish to when this input is active.
    pub channel_name: String,
    /// What type of value to publish.
    pub property: SignalProperty,
}

impl Default for SeatChannelMapping {
    fn default() -> Self {
        Self {
            bindings: vec![
                SeatInputBinding {
                    input_name: "thrust_forward".into(),
                    channel_name: "thrust-forward".into(),
                    property: SignalProperty::Throttle,
                },
                SeatInputBinding {
                    input_name: "thrust_lateral".into(),
                    channel_name: "thrust-lateral".into(),
                    property: SignalProperty::Throttle,
                },
                SeatInputBinding {
                    input_name: "thrust_vertical".into(),
                    channel_name: "thrust-vertical".into(),
                    property: SignalProperty::Throttle,
                },
                SeatInputBinding {
                    input_name: "torque_yaw".into(),
                    channel_name: "torque-yaw".into(),
                    property: SignalProperty::Throttle,
                },
                SeatInputBinding {
                    input_name: "torque_pitch".into(),
                    channel_name: "torque-pitch".into(),
                    property: SignalProperty::Throttle,
                },
            ],
        }
    }
}
