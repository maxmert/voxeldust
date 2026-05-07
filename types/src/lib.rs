//! Foundational primitive types — see Cargo.toml header.
//!
//! Module layout follows the conceptual layers:
//!  * [`shard`]                — orchestration identifiers (ShardId, SessionToken, …)
//!  * [`block_id`]             — voxel block identity
//!  * [`signal_property`]      — per-signal property enumeration
//!  * [`functional_block_kind`] — block→signal schema bridge
//!  * [`hud_delta_flags`]      — HUD-delta wire bit flags

pub mod block_id;
pub mod functional_block_kind;
pub mod hud_delta_flags;
pub mod shard;
pub mod signal_property;

// Convenience re-exports for the most-imported names.
pub use block_id::BlockId;
pub use functional_block_kind::{BlockKindSignalSchema, FunctionalBlockKind};
pub use shard::{
    SessionToken, ShardEndpoint, ShardHeartbeat, ShardId, ShardInfo, ShardState, ShardType,
};
pub use signal_property::SignalProperty;
