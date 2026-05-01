//! Block signal configuration — the data that flows between server and client
//! for configuring functional block signal bindings, converter rules, and seat mappings.
//!
//! These types use string-based channel names (not `ChannelId`) because they
//! cross the network boundary where IDs are shard-local.  The server resolves
//! names → IDs when applying config updates.

use glam::IVec3;

use super::components::{AxisDirection, KeyMode, SeatInputSource};
use super::types::SignalProperty;

// ---------------------------------------------------------------------------
// Config-layer binding types (string-based, for serialization / UI)
// ---------------------------------------------------------------------------

/// Publish binding in config form (string channel name).
#[derive(Clone, Debug)]
pub struct PublishBindingConfig {
    pub channel_name: String,
    pub property: SignalProperty,
}

/// Subscribe binding in config form (string channel name).
#[derive(Clone, Debug)]
pub struct SubscribeBindingConfig {
    pub channel_name: String,
    pub property: SignalProperty,
}

/// Generic seat input binding in config form (string channel name).
#[derive(Clone, Debug)]
pub struct SeatInputBindingConfig {
    pub label: String,
    pub source: SeatInputSource,
    pub key_name: String,
    pub key_mode: KeyMode,
    pub axis_direction: AxisDirection,
    pub channel_name: String,
    pub property: SignalProperty,
}

/// Signal converter rule in config form (string channel names).
#[derive(Clone, Debug)]
pub struct SignalRuleConfig {
    pub input_channel: String,
    pub condition: super::converter::SignalCondition,
    pub output_channel: String,
    pub expression: super::converter::SignalExpression,
}

// ---------------------------------------------------------------------------
// Custom block config types — each block has its own typed config
// ---------------------------------------------------------------------------

/// Flight computer configuration. Each channel is explicit.
#[derive(Clone, Debug, Default)]
pub struct FlightComputerConfig {
    pub yaw_cw_channel: String,
    pub yaw_ccw_channel: String,
    pub pitch_up_channel: String,
    pub pitch_down_channel: String,
    pub roll_cw_channel: String,
    pub roll_ccw_channel: String,
    pub toggle_channel: String,
    pub damping_gain: f32,
    pub dead_zone: f32,
    pub max_correction: f32,
}

/// Hover module configuration. Each channel is explicit.
#[derive(Clone, Debug, Default)]
pub struct HoverModuleConfig {
    pub thrust_forward_channel: String,
    pub thrust_reverse_channel: String,
    pub thrust_right_channel: String,
    pub thrust_left_channel: String,
    pub thrust_up_channel: String,
    pub thrust_down_channel: String,
    pub yaw_cw_channel: String,
    pub yaw_ccw_channel: String,
    pub pitch_up_channel: String,
    pub pitch_down_channel: String,
    pub roll_cw_channel: String,
    pub roll_ccw_channel: String,
    pub activate_channel: String,
    pub cutoff_channel: String,
}

/// Autopilot configuration. Each channel is explicit.
#[derive(Clone, Debug, Default)]
pub struct AutopilotBlockConfig {
    pub yaw_cw_channel: String,
    pub yaw_ccw_channel: String,
    pub pitch_up_channel: String,
    pub pitch_down_channel: String,
    pub roll_cw_channel: String,
    pub roll_ccw_channel: String,
    pub engage_channel: String,
}

/// Warp computer configuration.
#[derive(Clone, Debug, Default)]
pub struct WarpComputerConfig {
    pub cycle_channel: String,
    pub accept_channel: String,
    pub cancel_channel: String,
}

/// Engine controller configuration. Each channel is explicit.
#[derive(Clone, Debug, Default)]
pub struct EngineControllerConfig {
    pub thrust_forward_channel: String,
    pub thrust_reverse_channel: String,
    pub thrust_right_channel: String,
    pub thrust_left_channel: String,
    pub thrust_up_channel: String,
    pub thrust_down_channel: String,
    pub yaw_cw_channel: String,
    pub yaw_ccw_channel: String,
    pub pitch_up_channel: String,
    pub pitch_down_channel: String,
    pub roll_cw_channel: String,
    pub roll_ccw_channel: String,
    pub toggle_channel: String,
}

/// Mechanical mount configuration (rotor/piston speed override).
#[derive(Clone, Debug, Default)]
pub struct MechanicalConfig {
    /// Speed override (deg/s for revolute, m/s for prismatic).
    /// None = use registry default. Capped by MechanicalProps.max_speed.
    pub speed_override: Option<f32>,
}

/// Phase 3E: Antenna block config. Wire form for the F-key UI; resolved
/// to `AntennaState` at `apply_config_updates` time. The grant key
/// itself is NOT in this config — antennas reference a grant by id that
/// must already be in the placing player's `HeldGrants` (registered via
/// `ClientMsg::AddHeldGrant`).
#[derive(Clone, Debug, Default)]
pub struct AntennaConfig {
    /// Local channel the antenna READS from each tick.
    pub source_channel_name: String,
    /// Remote channel name on the target shard (the publish target).
    pub remote_channel_name: String,
    pub frequency: u32,
    pub grant_id: u64,
    pub target_shard_id: u64,
}

/// Phase 3E: Listener block config. Symmetric to `AntennaConfig`.
#[derive(Clone, Debug, Default)]
pub struct ListenerConfig {
    /// Local Local-scope channel where the bridged value gets mirrored.
    /// Internal subscribers wire to this.
    pub destination_channel_name: String,
    /// Local Radio-scope channel that receives forwarded entries.
    /// Created automatically by the listener's apply step at the
    /// publisher's listed `remote_channel_name` so try_push_remote
    /// has somewhere to push to.
    pub bridged_channel_name: String,
    pub frequency: u32,
    pub grant_id: u64,
    pub source_shard_id: u64,
}

// ---------------------------------------------------------------------------
// Power configuration types
// ---------------------------------------------------------------------------

/// A named power circuit on a reactor (config form, for serialization / UI).
#[derive(Clone, Debug)]
pub struct PowerCircuitConfig {
    pub name: String,
    pub fraction: f32,
}

/// Power access mode for a reactor (config form).
#[derive(Clone, Debug, Default)]
pub enum PowerAccessConfig {
    #[default]
    OwnerOnly,
    AllowList(Vec<String>),
    Open,
}

/// Reactor power source configuration (sent in config snapshot / update).
#[derive(Clone, Debug, Default)]
pub struct PowerSourceConfig {
    pub circuits: Vec<PowerCircuitConfig>,
    pub access: PowerAccessConfig,
}

/// Power consumer configuration — which reactor + circuit to draw from.
#[derive(Clone, Debug, Default)]
pub struct PowerConsumerConfig {
    pub reactor_pos: Option<IVec3>,
    pub circuit: String,
}

/// Info about a nearby reactor (for consumer dropdown in config UI).
#[derive(Clone, Debug)]
pub struct NearbyReactorInfo {
    pub pos: IVec3,
    pub label: String,
    pub distance: f32,
    pub circuits: Vec<String>,
}

// ---------------------------------------------------------------------------
// Config snapshots
// ---------------------------------------------------------------------------

/// Complete signal configuration snapshot for a functional block.
/// Sent from server → client when a player opens the config UI.
#[derive(Clone, Debug, Default)]
pub struct BlockSignalConfig {
    pub block_pos: IVec3,
    pub block_type: u16,
    pub kind: u8,
    pub publish_bindings: Vec<PublishBindingConfig>,
    pub subscribe_bindings: Vec<SubscribeBindingConfig>,
    pub converter_rules: Vec<SignalRuleConfig>,
    pub seat_mappings: Vec<SeatInputBindingConfig>,
    pub seated_channel_name: String,
    pub available_channels: Vec<String>,
    /// Property options the configurator UI should offer for *publish* bindings
    /// on this block kind. Each entry is `(SignalProperty as_ordinal, hint_text)`.
    /// Empty if the block doesn't publish at all (e.g., Thruster).
    pub publish_property_options: Vec<(u8, String)>,
    /// Property options the configurator UI should offer for *subscribe* bindings.
    /// Empty if the block doesn't subscribe at all.
    pub subscribe_property_options: Vec<(u8, String)>,
    pub power_source: Option<PowerSourceConfig>,
    pub power_consumer: Option<PowerConsumerConfig>,
    pub nearby_reactors: Vec<NearbyReactorInfo>,
    // Custom block configs (at most one populated per block type):
    pub flight_computer: Option<FlightComputerConfig>,
    pub hover_module: Option<HoverModuleConfig>,
    pub autopilot: Option<AutopilotBlockConfig>,
    pub warp_computer: Option<WarpComputerConfig>,
    pub engine_controller: Option<EngineControllerConfig>,
    pub mechanical: Option<MechanicalConfig>,
    pub antenna: Option<AntennaConfig>,
    pub listener: Option<ListenerConfig>,
}

impl BlockSignalConfig {
    /// Populate `publish_property_options` and `subscribe_property_options`
    /// from a `FunctionalBlockKind`'s static schema. Single source of truth
    /// shared with `apply_config_updates` validation.
    pub fn set_property_options_from_kind(
        &mut self,
        kind: crate::block::registry::FunctionalBlockKind,
    ) {
        let schema = kind.signal_schema();
        let hint_for = |prop: SignalProperty| -> String {
            schema
                .property_hints
                .iter()
                .find(|(p, _)| *p == prop)
                .map(|(_, h)| (*h).to_string())
                .unwrap_or_default()
        };
        self.publish_property_options = schema
            .publish_properties
            .iter()
            .map(|p| (p.as_ordinal(), hint_for(*p)))
            .collect();
        self.subscribe_property_options = schema
            .subscribe_properties
            .iter()
            .map(|p| (p.as_ordinal(), hint_for(*p)))
            .collect();
    }
}

/// Reasons a `BlockConfigUpdateData` can be rejected at the server boundary.
/// Surfaces to the client as a HUD error toast so the player understands why
/// their config didn't apply (e.g., picked a property the block doesn't support).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigInvalid {
    /// The chosen `SignalProperty` for a publish binding isn't in the block
    /// kind's `publish_properties` schema. Carries the offending property
    /// and the kind's allowed set for human-readable error rendering.
    PublishPropertyNotSupported {
        kind: crate::block::registry::FunctionalBlockKind,
        property: SignalProperty,
        allowed: Vec<SignalProperty>,
    },
    /// Symmetric for subscribe bindings.
    SubscribePropertyNotSupported {
        kind: crate::block::registry::FunctionalBlockKind,
        property: SignalProperty,
        allowed: Vec<SignalProperty>,
    },
}

impl ConfigInvalid {
    /// Check every binding in `update` against the kind's schema. Returns
    /// `Ok(())` if all bindings are valid, `Err(ConfigInvalid)` on the first
    /// violation. Caller is responsible for surfacing the error.
    pub fn validate_against_kind(
        update: &BlockConfigUpdateData,
        kind: crate::block::registry::FunctionalBlockKind,
    ) -> Result<(), Self> {
        let schema = kind.signal_schema();
        for b in &update.publish_bindings {
            if !schema.publish_properties.contains(&b.property) {
                return Err(Self::PublishPropertyNotSupported {
                    kind,
                    property: b.property,
                    allowed: schema.publish_properties.to_vec(),
                });
            }
        }
        for b in &update.subscribe_bindings {
            if !schema.subscribe_properties.contains(&b.property) {
                return Err(Self::SubscribePropertyNotSupported {
                    kind,
                    property: b.property,
                    allowed: schema.subscribe_properties.to_vec(),
                });
            }
        }
        Ok(())
    }

    /// Render a one-line user-facing message for the HUD toast.
    pub fn to_user_message(&self) -> String {
        match self {
            Self::PublishPropertyNotSupported { kind, property, allowed } => {
                format!(
                    "A {:?} cannot publish '{:?}' — supported: {}",
                    kind,
                    property,
                    format_property_list(allowed),
                )
            }
            Self::SubscribePropertyNotSupported { kind, property, allowed } => {
                format!(
                    "A {:?} cannot subscribe to '{:?}' — supported: {}",
                    kind,
                    property,
                    format_property_list(allowed),
                )
            }
        }
    }
}

fn format_property_list(props: &[SignalProperty]) -> String {
    if props.is_empty() {
        "(none)".to_string()
    } else {
        props
            .iter()
            .map(|p| format!("{:?}", p))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::FunctionalBlockKind;

    #[test]
    fn validate_rejects_pressure_on_thruster() {
        let mut update = BlockConfigUpdateData::default();
        update.subscribe_bindings.push(SubscribeBindingConfig {
            channel_name: "test".into(),
            property: SignalProperty::Pressure,
        });
        let err = ConfigInvalid::validate_against_kind(&update, FunctionalBlockKind::Thruster)
            .expect_err("Pressure must not be allowed on a Thruster");
        match err {
            ConfigInvalid::SubscribePropertyNotSupported { kind, property, .. } => {
                assert_eq!(kind, FunctionalBlockKind::Thruster);
                assert_eq!(property, SignalProperty::Pressure);
            }
            other => panic!("wrong error variant: {:?}", other),
        }
    }

    #[test]
    fn validate_accepts_throttle_on_thruster() {
        let mut update = BlockConfigUpdateData::default();
        update.subscribe_bindings.push(SubscribeBindingConfig {
            channel_name: "test".into(),
            property: SignalProperty::Throttle,
        });
        ConfigInvalid::validate_against_kind(&update, FunctionalBlockKind::Thruster)
            .expect("Throttle is in the Thruster's subscribe schema");
    }

    #[test]
    fn validate_rejects_publish_on_thruster() {
        // Thrusters' publish_properties is empty — *any* publish binding is invalid.
        let mut update = BlockConfigUpdateData::default();
        update.publish_bindings.push(PublishBindingConfig {
            channel_name: "test".into(),
            property: SignalProperty::Throttle,
        });
        let err = ConfigInvalid::validate_against_kind(&update, FunctionalBlockKind::Thruster)
            .expect_err("Thrusters must not publish anything");
        assert!(matches!(
            err,
            ConfigInvalid::PublishPropertyNotSupported { .. }
        ));
    }

    #[test]
    fn validate_user_message_lists_allowed_properties() {
        let mut update = BlockConfigUpdateData::default();
        update.subscribe_bindings.push(SubscribeBindingConfig {
            channel_name: "test".into(),
            property: SignalProperty::Speed,
        });
        let err =
            ConfigInvalid::validate_against_kind(&update, FunctionalBlockKind::Thruster).unwrap_err();
        let msg = err.to_user_message();
        // The message should mention what's allowed so the player learns.
        assert!(msg.contains("Throttle"), "msg: {}", msg);
        assert!(msg.contains("Speed"), "msg: {}", msg);
    }

    #[test]
    fn set_property_options_from_kind_thruster() {
        let mut cfg = BlockSignalConfig::default();
        cfg.set_property_options_from_kind(FunctionalBlockKind::Thruster);
        // Thrusters don't publish.
        assert!(cfg.publish_property_options.is_empty());
        // Subscribe options match the schema (Throttle, Boost, Active).
        let ords: Vec<u8> = cfg
            .subscribe_property_options
            .iter()
            .map(|(o, _)| *o)
            .collect();
        assert_eq!(ords, vec![
            SignalProperty::Throttle.as_ordinal(),
            SignalProperty::Boost.as_ordinal(),
            SignalProperty::Active.as_ordinal(),
        ]);
        // Hints come through.
        let throttle_hint = &cfg.subscribe_property_options[0].1;
        assert!(throttle_hint.contains("0.0"), "hint: {}", throttle_hint);
    }
}

/// Config update sent from client → server after the player edits bindings.
/// Server validates and applies to the entity's signal components.
#[derive(Clone, Debug, Default)]
pub struct BlockConfigUpdateData {
    pub block_pos: IVec3,
    pub publish_bindings: Vec<PublishBindingConfig>,
    pub subscribe_bindings: Vec<SubscribeBindingConfig>,
    pub converter_rules: Vec<SignalRuleConfig>,
    pub seat_mappings: Vec<SeatInputBindingConfig>,
    pub seated_channel_name: String,
    pub power_source: Option<PowerSourceConfig>,
    pub power_consumer: Option<PowerConsumerConfig>,
    // Custom block configs:
    pub flight_computer: Option<FlightComputerConfig>,
    pub hover_module: Option<HoverModuleConfig>,
    pub autopilot: Option<AutopilotBlockConfig>,
    pub warp_computer: Option<WarpComputerConfig>,
    pub engine_controller: Option<EngineControllerConfig>,
    pub mechanical: Option<MechanicalConfig>,
    pub antenna: Option<AntennaConfig>,
    pub listener: Option<ListenerConfig>,
}
