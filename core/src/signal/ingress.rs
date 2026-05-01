//! Cross-shard inbound signal staging buffer — drained at the start of the
//! signal pipeline and pushed into `SignalChannelTable` after all access
//! checks pass (Phase 1+).
//!
//! Lives in `core` so any shard (ship, planet, station, galaxy) can host the
//! same generic ingress path. The fields beyond `name`/`value`/`scope_code`
//! are placeholders for the Phase 2/3 wire-protocol additions: they default
//! to zero so today's wire format keeps decoding cleanly.

use bevy_ecs::prelude::*;

use super::types::SignalValue;

/// One inbound entry queued from a peer shard's `SignalBroadcastBatch`.
///
/// Phase 1 only consumes `name`, `value`, and `scope_code` (for scope-class
/// matching against the local channel). The remaining fields land Phase 2+:
/// `sender_shard_id` + `seq` + `timestamp_ms` drive replay defense; `grant_id`
/// + `auth_tag` carry the HMAC capability proof. Defaults are zero/empty so a
/// pre-Phase-2 decoder produces a fully valid `IncomingSignalEntry`.
#[derive(Clone, Debug)]
pub struct IncomingSignalEntry {
    /// Channel name as sent over the wire (will be `<owner_id>.<path>` once
    /// auto-namespacing lands in Phase 1's bullet 8).
    pub name: String,
    pub value: SignalValue,
    /// Wire scope code: 0=Local, 1=ShortRange, 2=LongRange, 3=Radio.
    /// Receiver enforces scope-class match against the local channel.
    pub scope_code: u8,
    pub freq: u32,
    pub range_m: f64,
    /// Phase 2+ replay defense — sender's shard id; receiver tracks per
    /// `(channel, sender_shard_id)` sequence windows.
    pub sender_shard_id: u64,
    pub timestamp_ms: u64,
    pub seq: u64,
    /// Phase 3 capability — grant_id 0 means "no grant" (open broadcast).
    pub grant_id: u64,
    /// Phase 3 capability — 16-byte HMAC tag, empty if unauthenticated.
    /// Stored as `Vec<u8>` rather than `[u8; 16]` so the resource has a
    /// stable shape across migration phases.
    pub auth_tag: Vec<u8>,
}

impl IncomingSignalEntry {
    /// Convenience constructor for legacy callers that only have name+value.
    /// Equivalent to "open Local-scoped publish from an unidentified peer" —
    /// will be rejected by Phase 1's `try_push_remote` once that lands.
    pub fn legacy(name: String, value: SignalValue) -> Self {
        Self {
            name,
            value,
            scope_code: 0,
            freq: 0,
            range_m: 0.0,
            sender_shard_id: 0,
            timestamp_ms: 0,
            seq: 0,
            grant_id: 0,
            auth_tag: Vec::new(),
        }
    }
}

/// Resource: pending inbound signals queued by the shard's QUIC drain step,
/// drained by `signal_remote_ingress` at the top of the signal pipeline.
///
/// Stored as a `Resource` (not an event) so a tick that ingests but doesn't
/// publish (e.g., a paused shard) retains entries for the next tick — events
/// would auto-clear at frame boundaries.
#[derive(Resource, Default, Debug)]
pub struct IncomingSignalBuffer {
    pub entries: Vec<IncomingSignalEntry>,
}

/// Resource: pending cross-shard `ShardMsg::SignalSubscribe` /
/// `SignalUnsubscribe` requests queued by the shard's QUIC drain step.
/// Drained by `apply_signal_subscribe` / `apply_signal_unsubscribe` in
/// `shard-common::signal_pipeline` at the top of the signal pipeline so
/// freshly-registered subscriptions take effect this tick.
///
/// Same Resource-vs-event rationale as [`IncomingSignalBuffer`]: messages
/// arriving between ticks need to persist until the system runs.
#[derive(Resource, Default, Debug)]
pub struct IncomingSubscribeBuffer {
    pub subscribes: Vec<crate::shard_message::SignalSubscribeData>,
    pub unsubscribes: Vec<crate::shard_message::SignalUnsubscribeData>,
}

impl IncomingSubscribeBuffer {
    pub fn push_subscribe(&mut self, data: crate::shard_message::SignalSubscribeData) {
        self.subscribes.push(data);
    }

    pub fn push_unsubscribe(&mut self, data: crate::shard_message::SignalUnsubscribeData) {
        self.unsubscribes.push(data);
    }
}

impl IncomingSignalBuffer {
    /// Append one inbound entry. Used by every shard's `drain_quic` step.
    pub fn push(&mut self, entry: IncomingSignalEntry) {
        self.entries.push(entry);
    }

    /// Convenience for the most common call site that just has name+value.
    pub fn push_legacy(&mut self, name: String, value: SignalValue) {
        self.entries.push(IncomingSignalEntry::legacy(name, value));
    }

    pub fn drain(&mut self) -> std::vec::Drain<'_, IncomingSignalEntry> {
        self.entries.drain(..)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
