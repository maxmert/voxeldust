//! Block signal configuration — the data that flows between server and client
//! for configuring functional block signal bindings, converter rules, and seat mappings.
//!
//! These types use string-based channel names (not `ChannelId`) because they
//! cross the network boundary where IDs are shard-local.  The server resolves
//! names → IDs when applying config updates.

use glam::IVec3;

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

/// Seat input binding in config form (string channel name).
#[derive(Clone, Debug)]
pub struct SeatInputBindingConfig {
    pub channel_name: String,
    pub control: super::components::SeatControl,
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
// Power configuration types
// ---------------------------------------------------------------------------

/// A named power circuit on a reactor (config form, for serialization / UI).
#[derive(Clone, Debug)]
pub struct PowerCircuitConfig {
    /// Circuit name (e.g., "main", "rcs", "lights").
    pub name: String,
    /// Fraction of reactor output allocated to this circuit (0.0–1.0).
    pub fraction: f32,
}

/// Power access mode for a reactor (config form).
#[derive(Clone, Debug, Default)]
pub enum PowerAccessConfig {
    /// Only blocks placed by the same player.
    #[default]
    OwnerOnly,
    /// Owner + listed player names.
    AllowList(Vec<String>),
    /// Anyone in range.
    Open,
}

/// Reactor power source configuration (sent in config snapshot / update).
#[derive(Clone, Debug, Default)]
pub struct PowerSourceConfig {
    /// Named circuits with allocated fractions.
    pub circuits: Vec<PowerCircuitConfig>,
    /// Access control mode.
    pub access: PowerAccessConfig,
}

/// Power consumer configuration — which reactor + circuit to draw from.
#[derive(Clone, Debug, Default)]
pub struct PowerConsumerConfig {
    /// Block position of the reactor (None = not connected).
    pub reactor_pos: Option<IVec3>,
    /// Which circuit on that reactor.
    pub circuit: String,
}

/// Info about a nearby reactor (for consumer dropdown in config UI).
#[derive(Clone, Debug)]
pub struct NearbyReactorInfo {
    /// Block position of the reactor.
    pub pos: IVec3,
    /// Human-readable label (e.g., "Small Reactor").
    pub label: String,
    /// Distance from the consumer block in blocks.
    pub distance: f32,
    /// Circuit names available on this reactor.
    pub circuits: Vec<String>,
}

// ---------------------------------------------------------------------------
// Config snapshots
// ---------------------------------------------------------------------------

/// Complete signal configuration snapshot for a functional block.
/// Sent from server → client when a player opens the config UI.
#[derive(Clone, Debug, Default)]
pub struct BlockSignalConfig {
    /// Block world position.
    pub block_pos: IVec3,
    /// Block type ID.
    pub block_type: u16,
    /// Functional block kind.
    pub kind: u8,
    /// Publish bindings (block → channels).
    pub publish_bindings: Vec<PublishBindingConfig>,
    /// Subscribe bindings (channels → block).
    pub subscribe_bindings: Vec<SubscribeBindingConfig>,
    /// Converter rules (only for SignalConverter blocks).
    pub converter_rules: Vec<SignalRuleConfig>,
    /// Seat input mappings (only for Seat blocks).
    pub seat_mappings: Vec<SeatInputBindingConfig>,
    /// All channel names on this structure (for dropdown selection in UI).
    pub available_channels: Vec<String>,
    /// Reactor power source config (only for Reactor blocks).
    pub power_source: Option<PowerSourceConfig>,
    /// Consumer power subscription (only for power-consuming blocks).
    pub power_consumer: Option<PowerConsumerConfig>,
    /// Nearby reactors in range (for consumer dropdown in config UI).
    pub nearby_reactors: Vec<NearbyReactorInfo>,
}

/// Config update sent from client → server after the player edits bindings.
/// Server validates and applies to the entity's signal components.
#[derive(Clone, Debug, Default)]
pub struct BlockConfigUpdateData {
    /// Block world position (identifies which block to update).
    pub block_pos: IVec3,
    /// Updated publish bindings.
    pub publish_bindings: Vec<PublishBindingConfig>,
    /// Updated subscribe bindings.
    pub subscribe_bindings: Vec<SubscribeBindingConfig>,
    /// Updated converter rules.
    pub converter_rules: Vec<SignalRuleConfig>,
    /// Updated seat mappings.
    pub seat_mappings: Vec<SeatInputBindingConfig>,
    /// Updated reactor power source config (only for Reactor blocks).
    pub power_source: Option<PowerSourceConfig>,
    /// Updated consumer power subscription.
    pub power_consumer: Option<PowerConsumerConfig>,
}
