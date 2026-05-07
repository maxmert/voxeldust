//! Bit flags for `HudSignalEntryV2Data::flags` on the wire.
//!
//! Lifted from the prior `voxeldust_core::client_message::hud_delta_flags`
//! module so that `voxeldust-signal::hud_session` can read them without
//! pulling all of `client_message.rs` into its dependency closure.

/// The entry carries a fresh `wire_id → name` binding. Receiver
/// inserts/overwrites in its inbound dict before resolving.
pub const REGISTER: u8 = 0x01;
/// The wire_id is being dropped from the session's dictionary.
/// `value_type`, `value_num`, `value_text`, `property` are all
/// ignored. `channel_name` is empty.
pub const REMOVE: u8 = 0x02;
