//! Per-peer wire-dictionary registry — the runtime home of
//! [`OutboundDict`] and [`InboundDict`] keyed by peer `ShardId`.
//!
//! This registry is the bridge between the per-connection wire-dict
//! protocol (defined in `core::signal::wire_dict`) and the runtime
//! topology of QUIC peers maintained in `peer_registry`. It owns the
//! mutable state both ends of the protocol need:
//!
//! - **Send path**: `signal_broadcast_remote` (and any other system
//!   building V2 batches) interns channel names against
//!   `outbound_for(peer)` to assign wire ids.
//! - **Receive path**: the QUIC drain loop calls `inbound_for(peer)`
//!   to verify `dict_seq` monotonicity, register `FLAG_REGISTER`
//!   entries, and resolve bare entries' wire ids back to channel
//!   names.
//!
//! # Lifecycle
//!
//! Dicts are inserted lazily on first send/receive — zero state for
//! peers we never talk to. They persist across orchestrator-level
//! peer-list refreshes (which fire frequently and have no bearing on
//! actual QUIC connection state). They are explicitly cleared via
//! [`reset_peer`] when a QUIC connection is re-established (the
//! receiver's inbound state is wiped on the wire, so our outbound
//! cache is stale and we must re-register all bindings).
//!
//! # Concurrency
//!
//! The registry is shared between the tokio QUIC drain task and
//! Bevy ECS systems. The standard pattern in this crate (see
//! `peer_registry`) wraps it in `Arc<RwLock<...>>` and stores the
//! handle in `NetworkBridge`. Read locks are held for short windows
//! (lookup by peer + a single intern/register call) so the lock is
//! not a hot-path bottleneck even at 100K-peer scale.

use std::collections::HashMap;

use glam::DVec3;
use thiserror::Error;
use voxeldust_core::shard_message::{
    SignalBroadcastBatchV2Data, SignalBroadcastEntry, SignalBroadcastEntryV2,
};
use voxeldust_core::shard_types::ShardId;
use voxeldust_core::signal::metrics as signal_metrics;
use voxeldust_core::signal::wire_dict::{
    InboundDict, InboundDictError, OutboundDict, FLAG_REGISTER,
};

/// One outbound + one inbound dictionary per peer shard.
#[derive(Debug, Default)]
pub struct WireDictRegistry {
    outbound: HashMap<ShardId, OutboundDict>,
    inbound: HashMap<ShardId, InboundDict>,
}

impl WireDictRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mutable handle to the outbound dict for `peer`. Created on
    /// first call — subsequent calls reuse the same instance.
    /// Sender code path.
    pub fn outbound_for(&mut self, peer: ShardId) -> &mut OutboundDict {
        self.outbound.entry(peer).or_default()
    }

    /// Mutable handle to the inbound dict for `peer`. Created on
    /// first call. Receiver code path.
    pub fn inbound_for(&mut self, peer: ShardId) -> &mut InboundDict {
        self.inbound.entry(peer).or_default()
    }

    /// Drop both dicts for `peer`. Called on QUIC reconnect — the
    /// peer's inbound state on the wire is gone, so our outbound
    /// cache is stale; the peer's outbound state on the wire is gone
    /// too, so our inbound resolution would fail anyway.
    pub fn reset_peer(&mut self, peer: ShardId) {
        self.outbound.remove(&peer);
        self.inbound.remove(&peer);
    }

    /// Drop all peers' state. Used when the local shard is itself
    /// resetting (e.g., after an orchestrator-driven shard restart).
    pub fn reset_all(&mut self) {
        self.outbound.clear();
        self.inbound.clear();
    }

    /// Sum of bindings across every peer's outbound dict. Drives the
    /// `signal_dict_size{direction="outbound"}` gauge.
    pub fn outbound_total_len(&self) -> usize {
        self.outbound.values().map(|d| d.len()).sum()
    }

    /// Sum of bindings across every peer's inbound dict.
    pub fn inbound_total_len(&self) -> usize {
        self.inbound.values().map(|d| d.len()).sum()
    }

    /// Peers we currently have outbound state for.
    pub fn outbound_peer_count(&self) -> usize {
        self.outbound.len()
    }

    /// Peers we currently have inbound state for.
    pub fn inbound_peer_count(&self) -> usize {
        self.inbound.len()
    }
}

/// Reasons [`decode_v2_batch`] rejects a batch. Each is a hard
/// resync trigger — the caller drops the batch, increments the
/// `signal_dict_resync_total` counter, and (ideally) clears its
/// inbound dict for the peer so the sender's next batch's
/// `FLAG_REGISTER` entries rebuild from scratch.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum V2DecodeError {
    /// Sender's `dict_seq` went backwards relative to the highest seq
    /// we already accepted. Either packet reorder (rare with QUIC's
    /// in-order streams) or sender restarted without our knowledge.
    #[error("dict_seq drift on inbound batch: {0}")]
    SeqDrift(InboundDictError),

    /// A non-FLAG_REGISTER entry referenced a `wire_id` we don't have
    /// a binding for. Either the sender's dict has bindings ours
    /// doesn't (we evicted differently, or never received the
    /// FLAG_REGISTER) — we drop the batch and trust the sender to
    /// re-register on the next pass.
    #[error("unresolved wire_id {wire_id} in inbound batch")]
    UnresolvedWireId { wire_id: u32 },
}

impl From<InboundDictError> for V2DecodeError {
    fn from(e: InboundDictError) -> Self {
        Self::SeqDrift(e)
    }
}

/// Encode a batch of V1-shaped entries into a V2 batch by interning
/// each channel name against the per-peer outbound dict.
///
/// This is the canonical sender-side path: a Bevy system collects
/// dirty entries (already shaped as `SignalBroadcastEntry`) and hands
/// them to this function, which:
///
/// 1. Acquires the outbound dict for `peer` (lazily inserted on first
///    call).
/// 2. Calls `intern` per entry — first sighting of a name returns
///    `needs_register=true` and the name goes on the wire with
///    `FLAG_REGISTER`; subsequent sightings carry only the wire_id.
/// 3. Stamps the batch with `dict_seq` so the receiver can verify
///    monotonicity.
///
/// All input fields except `channel_name` pass through unchanged
/// (auth_tag, sequence, timestamp_ms, grant_id, etc. are still
/// computed by the caller — this function is purely a name-interning
/// transformation).
pub fn encode_v2_batch(
    registry: &mut WireDictRegistry,
    peer: ShardId,
    source_shard_id: u64,
    source_position: DVec3,
    entries: Vec<SignalBroadcastEntry>,
) -> SignalBroadcastBatchV2Data {
    let dict = registry.outbound_for(peer);
    let v2_entries = entries
        .into_iter()
        .map(|e| {
            let r = dict.intern(&e.channel_name);
            // FLAG_REGISTER carries the name; bare entries ship empty
            // string for FB schema uniformity (receivers MUST gate on
            // the flag, not on string emptiness).
            let flags = if r.needs_register { FLAG_REGISTER } else { 0 };
            if r.needs_register {
                signal_metrics::record_dict_register();
            } else {
                signal_metrics::record_dict_hit();
            }
            let channel_name = if r.needs_register {
                e.channel_name
            } else {
                String::new()
            };
            SignalBroadcastEntryV2 {
                flags,
                wire_id: r.wire_id,
                channel_name,
                value_type: e.value_type,
                value_data: e.value_data,
                scope: e.scope,
                range_m: e.range_m,
                frequency: e.frequency,
                sequence: e.sequence,
                timestamp_ms: e.timestamp_ms,
                grant_id: e.grant_id,
                auth_tag: e.auth_tag,
            }
        })
        .collect();
    let dict_seq = dict.dict_seq();
    SignalBroadcastBatchV2Data {
        source_shard_id,
        source_position,
        dict_seq,
        entries: v2_entries,
    }
}

/// Decode a V2 batch back to V1-shaped entries by resolving each
/// entry's `wire_id` against the per-peer inbound dict.
///
/// The receive path:
///
/// 1. Verifies `batch.dict_seq` against `last_dict_seq` (rejects
///    backward drift).
/// 2. For every `FLAG_REGISTER` entry, binds `wire_id → channel_name`
///    in the inbound dict, then yields the entry with the carried
///    name.
/// 3. For every bare entry, looks up `wire_id` in the dict; if
///    present, yields the entry with the resolved name; if absent,
///    aborts with `UnresolvedWireId` so the caller can resync.
///
/// On `Err`, the inbound dict's `last_dict_seq` may have already
/// advanced (from the seq verify in step 1) — that's fine, since the
/// next batch's seq will be >= ours.
pub fn decode_v2_batch(
    registry: &mut WireDictRegistry,
    peer: ShardId,
    batch: SignalBroadcastBatchV2Data,
) -> Result<Vec<SignalBroadcastEntry>, V2DecodeError> {
    let dict = registry.inbound_for(peer);
    if let Err(e) = dict.verify_seq(batch.dict_seq) {
        signal_metrics::record_dict_resync(signal_metrics::DictResyncReason::SeqDrift);
        return Err(V2DecodeError::SeqDrift(e));
    }
    let mut out = Vec::with_capacity(batch.entries.len());
    for e in batch.entries {
        let needs_register = (e.flags & FLAG_REGISTER) != 0;
        let resolved_name = if needs_register {
            // Carrying a fresh binding — overwrite our dict.
            dict.register(e.wire_id, e.channel_name.clone());
            e.channel_name
        } else {
            // Bare entry — must be resolvable from the dict.
            match dict.resolve(e.wire_id) {
                Some(name) => name.to_owned(),
                None => {
                    signal_metrics::record_dict_resync(
                        signal_metrics::DictResyncReason::UnresolvedWireId,
                    );
                    return Err(V2DecodeError::UnresolvedWireId {
                        wire_id: e.wire_id,
                    });
                }
            }
        };
        out.push(SignalBroadcastEntry {
            channel_name: resolved_name,
            value_type: e.value_type,
            value_data: e.value_data,
            scope: e.scope,
            range_m: e.range_m,
            frequency: e.frequency,
            sequence: e.sequence,
            timestamp_ms: e.timestamp_ms,
            grant_id: e.grant_id,
            auth_tag: e.auth_tag,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(id: u64) -> ShardId {
        ShardId(id)
    }

    #[test]
    fn lazy_insert_creates_dict_on_first_access() {
        let mut r = WireDictRegistry::new();
        assert_eq!(r.outbound_peer_count(), 0);
        let _ = r.outbound_for(p(1));
        assert_eq!(r.outbound_peer_count(), 1);
        // Second call to the same peer reuses the instance.
        let _ = r.outbound_for(p(1));
        assert_eq!(r.outbound_peer_count(), 1);
    }

    #[test]
    fn outbound_and_inbound_are_independent() {
        let mut r = WireDictRegistry::new();
        let _ = r.outbound_for(p(1));
        assert_eq!(r.outbound_peer_count(), 1);
        assert_eq!(r.inbound_peer_count(), 0);
        let _ = r.inbound_for(p(2));
        assert_eq!(r.inbound_peer_count(), 1);
        assert_eq!(r.outbound_peer_count(), 1);
    }

    #[test]
    fn reset_peer_drops_only_that_peer() {
        let mut r = WireDictRegistry::new();
        r.outbound_for(p(1)).intern("a");
        r.outbound_for(p(2)).intern("b");
        r.inbound_for(p(1)).register(0, "x".into());
        r.reset_peer(p(1));
        assert_eq!(r.outbound_peer_count(), 1);
        assert_eq!(r.inbound_peer_count(), 0);
        // Peer 2's outbound state survives.
        assert_eq!(r.outbound_for(p(2)).len(), 1);
    }

    #[test]
    fn totals_aggregate_across_peers() {
        let mut r = WireDictRegistry::new();
        r.outbound_for(p(1)).intern("a");
        r.outbound_for(p(1)).intern("b");
        r.outbound_for(p(2)).intern("c");
        assert_eq!(r.outbound_total_len(), 3);
        r.inbound_for(p(1)).register(7, "x".into());
        assert_eq!(r.inbound_total_len(), 1);
    }

    // ---------------- encode/decode helpers ------------------

    fn entry(name: &str, val: f32) -> SignalBroadcastEntry {
        SignalBroadcastEntry {
            channel_name: name.into(),
            value_type: 1,
            value_data: val,
            scope: 1,
            range_m: 100.0,
            frequency: 0,
            sequence: 1,
            timestamp_ms: 1_700_000_000_000,
            grant_id: 0,
            auth_tag: vec![],
        }
    }

    #[test]
    fn encode_first_send_marks_register_for_each_distinct_name() {
        let mut r = WireDictRegistry::new();
        let batch = encode_v2_batch(
            &mut r,
            p(1),
            42,
            DVec3::ZERO,
            vec![entry("a", 1.0), entry("b", 2.0), entry("a", 3.0)],
        );
        // Three entries: a + b are first sightings (FLAG_REGISTER set);
        // the second `a` reuses the wire_id without FLAG_REGISTER.
        assert_eq!(batch.entries.len(), 3);
        assert_eq!(batch.entries[0].flags & FLAG_REGISTER, FLAG_REGISTER);
        assert_eq!(batch.entries[0].channel_name, "a");
        assert_eq!(batch.entries[1].flags & FLAG_REGISTER, FLAG_REGISTER);
        assert_eq!(batch.entries[1].channel_name, "b");
        assert_eq!(batch.entries[2].flags, 0);
        assert!(batch.entries[2].channel_name.is_empty());
        assert_eq!(batch.entries[2].wire_id, batch.entries[0].wire_id);
        // dict_seq covers the two register events, not the cache hit.
        assert_eq!(batch.dict_seq, 2);
    }

    #[test]
    fn encode_subsequent_batch_to_same_peer_has_no_register_for_known_names() {
        let mut r = WireDictRegistry::new();
        let _ = encode_v2_batch(&mut r, p(1), 42, DVec3::ZERO, vec![entry("a", 1.0)]);
        let batch = encode_v2_batch(&mut r, p(1), 42, DVec3::ZERO, vec![entry("a", 2.0)]);
        // Second batch: `a` is now a hit — bare entry.
        assert_eq!(batch.entries[0].flags, 0);
        assert!(batch.entries[0].channel_name.is_empty());
    }

    #[test]
    fn decode_resolves_register_then_bare_entry() {
        let mut sender = WireDictRegistry::new();
        let batch = encode_v2_batch(
            &mut sender,
            p(1),
            42,
            DVec3::ZERO,
            vec![entry("a", 1.0), entry("a", 2.0)],
        );

        let mut receiver = WireDictRegistry::new();
        let resolved = decode_v2_batch(&mut receiver, p(1), batch).unwrap();
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].channel_name, "a");
        assert_eq!(resolved[1].channel_name, "a", "bare entry resolves via dict");
        assert!(
            (resolved[1].value_data - 2.0).abs() < f32::EPSILON,
            "value_data passes through"
        );
    }

    #[test]
    fn decode_rejects_backward_dict_seq() {
        let mut receiver = WireDictRegistry::new();
        let high = SignalBroadcastBatchV2Data {
            source_shard_id: 42,
            source_position: DVec3::ZERO,
            dict_seq: 10,
            entries: vec![],
        };
        decode_v2_batch(&mut receiver, p(1), high).unwrap();
        // A second batch with a smaller dict_seq is rejected.
        let stale = SignalBroadcastBatchV2Data {
            source_shard_id: 42,
            source_position: DVec3::ZERO,
            dict_seq: 5,
            entries: vec![],
        };
        let err = decode_v2_batch(&mut receiver, p(1), stale).unwrap_err();
        assert!(matches!(err, V2DecodeError::SeqDrift(_)));
    }

    #[test]
    fn decode_rejects_unresolved_wire_id() {
        // Hand-craft a batch that references wire_id 99 without a
        // matching FLAG_REGISTER. The receiver must reject — this is
        // the resync trigger.
        let bad_batch = SignalBroadcastBatchV2Data {
            source_shard_id: 42,
            source_position: DVec3::ZERO,
            dict_seq: 1,
            entries: vec![SignalBroadcastEntryV2 {
                flags: 0,
                wire_id: 99,
                channel_name: String::new(),
                value_type: 1,
                value_data: 0.0,
                scope: 1,
                range_m: 0.0,
                frequency: 0,
                sequence: 0,
                timestamp_ms: 0,
                grant_id: 0,
                auth_tag: vec![],
            }],
        };
        let mut receiver = WireDictRegistry::new();
        let err = decode_v2_batch(&mut receiver, p(1), bad_batch).unwrap_err();
        assert_eq!(err, V2DecodeError::UnresolvedWireId { wire_id: 99 });
    }

    #[test]
    fn send_policy_defaults_to_v2_post_cutover() {
        // After the V1→V2 cutover the codebase-wide receiver path
        // accepts both formats (Phase 4.3.8/4.3.9), so V2 emission is
        // the default. The kill switch (`v2_enabled = false`) remains
        // available for a regression-driven rollback.
        use crate::harness::WireDictSendPolicy;
        let p = WireDictSendPolicy::default();
        assert!(
            p.v2_enabled,
            "post-cutover default must emit V2 to realize the bandwidth gains"
        );
    }

    #[test]
    fn round_trip_through_flatbuffers_full_pipeline() {
        // The complete production path: encode → ShardMsg → FB
        // serialize → bytes → FB deserialize → ShardMsg → decode.
        // Proves the helpers + the FB schema + the serde all line up.
        use voxeldust_core::shard_message::ShardMsg;

        let mut sender = WireDictRegistry::new();
        let mut receiver = WireDictRegistry::new();

        // Two batches to exercise both the register path (first
        // batch's first sighting of `a`, `b`) and the cache-hit path
        // (second batch reuses `a`).
        let b1_entries = vec![entry("alice.thrust", 0.5), entry("alice.lights", 1.0)];
        let b1 = encode_v2_batch(&mut sender, p(7), 42, DVec3::new(1.0, 2.0, 3.0), b1_entries);
        // Round-trip through FB.
        let bytes = ShardMsg::SignalBroadcastBatchV2(b1.clone()).serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::SignalBroadcastBatchV2(b1_decoded) = decoded else {
            panic!("wrong variant");
        };
        let resolved1 = decode_v2_batch(&mut receiver, p(7), b1_decoded).unwrap();
        assert_eq!(resolved1.len(), 2);
        assert_eq!(resolved1[0].channel_name, "alice.thrust");
        assert_eq!(resolved1[1].channel_name, "alice.lights");

        // Second batch: alice.thrust is now a hit on the sender side
        // → bare entry. Receiver resolves via its inbound dict.
        let b2_entries = vec![entry("alice.thrust", 0.75)];
        let b2 = encode_v2_batch(&mut sender, p(7), 42, DVec3::ZERO, b2_entries);
        // Confirm the bare-entry property is preserved across FB.
        assert_eq!(b2.entries[0].flags, 0);
        assert!(b2.entries[0].channel_name.is_empty());
        let bytes2 = ShardMsg::SignalBroadcastBatchV2(b2).serialize();
        let decoded2 = ShardMsg::deserialize(&bytes2).unwrap();
        let ShardMsg::SignalBroadcastBatchV2(b2_decoded) = decoded2 else {
            panic!("wrong variant");
        };
        let resolved2 = decode_v2_batch(&mut receiver, p(7), b2_decoded).unwrap();
        assert_eq!(resolved2.len(), 1);
        assert_eq!(resolved2[0].channel_name, "alice.thrust");
        assert!((resolved2[0].value_data - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn round_trip_capacity_overflow_recovers_via_eviction_and_re_register() {
        // Sender's outbound dict tied to a 2-slot capacity. A batch
        // referencing 3 distinct names forces an eviction; receiver's
        // FLAG_REGISTER overwrite keeps the dicts coherent without
        // any explicit deregister message.
        let mut sender = WireDictRegistry::new();
        sender.outbound_for(p(1)); // create
        // Replace with a tight-capacity dict to drive the test.
        sender
            .outbound
            .insert(p(1), OutboundDict::with_capacity(2));

        let mut receiver = WireDictRegistry::new();
        receiver.inbound_for(p(1));
        receiver
            .inbound
            .insert(p(1), InboundDict::with_capacity(2));

        let batch = encode_v2_batch(
            &mut sender,
            p(1),
            42,
            DVec3::ZERO,
            vec![entry("a", 1.0), entry("b", 2.0), entry("c", 3.0)],
        );
        // All three are new bindings → all three carry FLAG_REGISTER.
        // The sender evicted `a` to make room for `c`; the receiver
        // overwrites its slot for that wire_id.
        assert!(batch.entries.iter().all(|e| (e.flags & FLAG_REGISTER) != 0));
        let resolved = decode_v2_batch(&mut receiver, p(1), batch).unwrap();
        assert_eq!(resolved.len(), 3);
        assert_eq!(resolved[0].channel_name, "a");
        assert_eq!(resolved[1].channel_name, "b");
        assert_eq!(resolved[2].channel_name, "c");
    }
}
