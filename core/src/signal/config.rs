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
}
