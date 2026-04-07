//! Block signal configuration — the data that flows between server and client
//! for configuring functional block signal bindings, converter rules, and seat mappings.

use glam::IVec3;

use crate::block::{BlockId, FunctionalBlockKind};

use super::components::{PublishBinding, SeatInputBinding, SubscribeBinding};
use super::converter::SignalRule;

/// Complete signal configuration snapshot for a functional block.
/// Sent from server → client when a player opens the config UI.
/// Contains all editable signal properties for the block.
#[derive(Clone, Debug, Default)]
pub struct BlockSignalConfig {
    /// Block world position.
    pub block_pos: IVec3,
    /// Block type ID.
    pub block_type: u16,
    /// Functional block kind.
    pub kind: u8,
    /// Publish bindings (block → channels).
    pub publish_bindings: Vec<PublishBinding>,
    /// Subscribe bindings (channels → block).
    pub subscribe_bindings: Vec<SubscribeBinding>,
    /// Converter rules (only for SignalConverter blocks).
    pub converter_rules: Vec<SignalRule>,
    /// Seat input mappings (only for Seat blocks).
    pub seat_mappings: Vec<SeatInputBinding>,
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
    pub publish_bindings: Vec<PublishBinding>,
    /// Updated subscribe bindings.
    pub subscribe_bindings: Vec<SubscribeBinding>,
    /// Updated converter rules.
    pub converter_rules: Vec<SignalRule>,
    /// Updated seat mappings.
    pub seat_mappings: Vec<SeatInputBinding>,
}
