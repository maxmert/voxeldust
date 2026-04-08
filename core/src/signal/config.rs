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
}
