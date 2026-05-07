//! Signal-domain wire-format types.
//!
//! Server↔client snapshot/notification shapes used by the signal subsystem
//! when crossing the network boundary. Kept separate from the in-memory
//! stores in [`crate::grants`] / [`crate::ingress`] so:
//!  * code that only needs the wire shape (configurator UI, FlatBuffers
//!    serialize/deserialize in `voxeldust_core::client_message`) does not
//!    have to import the full `RemoteAccessGrant` / channel-table machinery;
//!  * the wire shape is a stable contract independent of registry layout
//!    refactors.
//!
//! Lives in `voxeldust-signal` (rather than `voxeldust-core::client_message`,
//! its pre-extraction home) because every wire type here is signal-domain
//! data — keeping it next to the signal logic is what enables the
//! `signal → core` inversion the workspace split required.

// ---------------------------------------------------------------------------
// Grant snapshot — server → owning client tablet UI
// ---------------------------------------------------------------------------

/// Public view of one `RemoteAccessGrant` for the owner's tablet UI.
/// Wire-format shape — separate from the in-memory `RemoteAccessGrant`
/// store in [`crate::grants`] so that the FlatBuffers serializer in
/// `voxeldust_core::client_message` can construct/parse these without
/// exposing the registry's private storage.
#[derive(Debug, Clone, Default)]
pub struct GrantPublicView {
    pub grant_id: u64,
    /// Base64-encoded 32-byte HMAC key. **Empty for non-owner recipients
    /// of the snapshot.** The server populates this only when the recipient
    /// owns the grant — defending against accidental key disclosure to
    /// other players in the same shard.
    pub key_b64: String,
    pub channel_names: Vec<String>,
    pub ops: u8,
    pub label: String,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
    pub created_by: u64,
    pub revoked: bool,
    pub namespace_glob: String,
}

/// Server → client snapshot of one player's owned grants. Sent in response
/// to any `GrantCreate` / `GrantRevoke` that mutates state for them.
/// Replaces, not merges — the snapshot is authoritative.
#[derive(Debug, Clone, Default)]
pub struct GrantsSnapshotData {
    pub grants: Vec<GrantPublicView>,
}

// ---------------------------------------------------------------------------
// Cross-shard subscribe / unsubscribe — used by `voxeldust_core::shard_message`
// to parse the FlatBuffers `SignalSubscribe` / `SignalUnsubscribe` wire
// payloads, and by [`crate::ingress::IncomingSubscribeBuffer`] to queue
// them for the bevy-side signal pipeline.
// ---------------------------------------------------------------------------

/// Payload for `ShardMsg::SignalSubscribe`. See `protocol/voxeldust.fbs::SignalSubscribe`
/// for the canonical wire shape.
#[derive(Debug, Clone, Default)]
pub struct SignalSubscribeData {
    pub subscriber_shard_id: u64,
    pub channel_name: String,
    pub grant_id: u64,
    pub nonce: u64,
    pub timestamp_ms: u64,
    pub valid_until_tick: u64,
    /// HMAC-SHA256-truncated-to-16 over the canonical request bytes.
    pub auth_tag: Vec<u8>,
}

/// Payload for `ShardMsg::SignalUnsubscribe`.
#[derive(Debug, Clone, Default)]
pub struct SignalUnsubscribeData {
    pub subscriber_shard_id: u64,
    pub channel_name: String,
    pub grant_id: u64,
}
