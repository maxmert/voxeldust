//! Voxeldust signal subsystem — channels, grants, auth, ingress, rate
//! limiting, HUD-session bookkeeping, wire-dict interning, radio-frequency
//! and ship-subscription tables, plus the per-block config and seat-input
//! preset tables.
//!
//! Pulled out of `voxeldust-core` so that:
//!  * 8.4 KL of signal code no longer cascades a rebuild of unrelated
//!    `core/` modules (block, character, handoff, autopilot, weather, …);
//!  * the block ↔ signal cycle that previously existed in core is broken
//!    by hoisting `BlockId`, `FunctionalBlockKind`, and `SignalProperty`
//!    into the foundational `voxeldust-types` crate.
//!
//! `voxeldust-core` re-exports this crate as `voxeldust_core::signal`
//! (umbrella shim), so existing import paths in shards/client continue to
//! resolve.

pub mod auth;
pub mod channel;
pub mod components;
pub mod config;
pub mod converter;
pub mod grants;
pub mod hud_session;
pub mod ingress;
pub mod key_names;
pub mod metrics;
pub mod radio_subscribers;
pub mod rate_limit;
pub mod seat_presets;
pub mod ship_frequency_interests;
pub mod types;
pub mod wire;
pub mod wire_dict;

pub use channel::{
    current_unix_millis, ChannelId, RemoteDirtyEntry, RemoteIngressDenied, ReplayReject,
    ReplayWindow, SignalChannelTable, SubscriberRef, REPLAY_TIMESTAMP_WINDOW_MS,
};
pub use grants::{
    decode_grant_key, encode_grant_key, generate_grant_id, generate_grant_key, GrantOps,
    GrantsRegistry, RemoteAccessGrant,
};
pub use ingress::{IncomingSignalBuffer, IncomingSignalEntry, IncomingSubscribeBuffer};
pub use rate_limit::{ClientBuckets, ClientRateLimits, TokenBucket};
#[allow(deprecated)]
pub use components::ListenerState;
pub use components::{
    AntennaRxSide, AntennaState, AntennaTxSide, AutopilotBlockState, AxisDirection,
    EngineControllerState, FlightComputerState, HoverModuleState, KeyMode,
    PublishBinding, SeatChannelMapping, SeatInputBinding, SeatInputSource,
    SignalConverterConfig, SignalPublisher, SignalSubscriber, SubscribeBinding,
    TerminalState, WarpComputerState,
};
pub use config::{
    AccessStatusForChannel, AntennaConfig, AntennaSide, AutopilotBlockConfig,
    BlockConfigUpdateData, BlockSignalConfig, EngineControllerConfig,
    FlightComputerConfig, HeldGrantSummary, HoverModuleConfig,
    PublishBindingConfig, SeatInputBindingConfig, SignalRuleConfig,
    SubscribeBindingConfig, TerminalConfig, WarpComputerConfig,
};
pub use converter::{SignalCondition, SignalExpression, SignalRule};
pub use seat_presets::SeatPreset;
pub use types::{AccessPolicy, ChannelMergeStrategy, SignalScope, SignalValue};
// SignalProperty lives in `voxeldust-types` (see `crate::types::SignalProperty`
// re-export); surface it here so `voxeldust_signal::SignalProperty` resolves.
pub use voxeldust_types::SignalProperty;
pub use wire::{GrantPublicView, GrantsSnapshotData, SignalSubscribeData, SignalUnsubscribeData};
