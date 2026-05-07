//! Per-connection wire dictionary — name interning for `SignalBroadcastEntry`.
//!
//! # The bandwidth problem
//!
//! `SignalBroadcastEntry` carries a `String` channel name on every wire
//! send. At 100K simulated load with broadcasts at 20 Hz, those names
//! repeat constantly: the same `alice.thrust-forward` ships in batch
//! after batch. Average channel name in the wild is ~30 bytes, vs ~14
//! bytes of fixed structural overhead per entry. The string is most of
//! the weight.
//!
//! # The interning solution
//!
//! Each connected peer pair maintains a shared dictionary that maps
//! channel names to a small `wire_id: u32`. Once a name has been
//! announced once (via `FLAG_REGISTER`), subsequent entries carry only
//! the `wire_id` and the receiver looks up the name locally.
//!
//! Steady-state savings: ~30 bytes name → 4 bytes id + 1 byte flags ≈
//! **3-5× wire-size reduction** on the per-entry hot path.
//!
//! # Why it's per-connection, not per-shard or global
//!
//! - **Per-connection**: each (sender, receiver) pair has independent
//!   dict state. When the connection drops, both dicts vanish; on
//!   reconnect they rebuild from scratch. This makes lifetime
//!   management trivial (no cross-connection invalidation, no
//!   distributed cache coherence) and cleanly bounds memory.
//! - **NOT global**: a global dict would force coordination of
//!   `wire_id` allocation across the cluster. Worth it only at
//!   astronomical channel counts; pointless at our scale.
//! - **NOT per-shard**: one dict per peer means each pair pays exactly
//!   for the channels actually flowing between them, not all channels
//!   the shard hosts.
//!
//! # Eviction protocol — LRU + dict_seq monotonicity
//!
//! Capacity is bounded (default 4096 — large enough that real
//! workloads fit, small enough that worst-case memory is ~2 MB). When
//! the sender hits capacity:
//!
//! 1. Evict the LRU entry locally.
//! 2. Assign the freed `wire_id` to the new channel.
//! 3. Increment `dict_seq` and emit a `FLAG_REGISTER` for the new
//!    binding.
//!
//! The receiver overwrites its own binding when it sees a
//! `FLAG_REGISTER` for a `wire_id` already in its dict — so the dicts
//! stay in sync without an explicit deregister message. The receiver
//! also has an internal capacity limit; if it diverges from the sender
//! (e.g., different defaults across shard versions), the *receiver*
//! evicts on its own and the next non-register reference to that
//! `wire_id` becomes a resync trigger.
//!
//! `dict_seq` is a strictly monotonic counter. Each send batch carries
//! the sender's current `dict_seq`. The receiver verifies: every batch
//! it sees has `dict_seq >= last_dict_seq`. If a gap is detected
//! (sender restarted, packet loss across QUIC reconnection), the
//! receiver triggers a full resync (clears its dict, requests
//! re-registration). The sender's dict survives a resync request — we
//! just re-emit `FLAG_REGISTER` for the next-touched channels until
//! the dicts converge again.
//!
//! # Why a u32 wire_id, not u16
//!
//! u16 caps at 65,535 — too tight if a future block-spam scenario
//! (player-built signal exchanges, aggregator nodes) creates tens of
//! thousands of distinct channels per connection. u32 gives a
//! 4-billion ceiling that we'll never realistically hit, costs 2
//! extra bytes per entry, and removes one operational concern from
//! the table.
//!
//! # Why not the `lru` crate
//!
//! The `lru` crate's `LruCache<K, V>` uses an internal doubly-linked
//! list; every read mutates the list, requiring a mutable borrow.
//! Our hot path is "look up by name on send, look up by id on
//! receive" — both want shared (immutable) read access most of the
//! time, mutating only when a fresh registration happens. We use a
//! sequence counter for ordering: each access bumps a u64 counter
//! that's stored in the entry; eviction picks the entry with the
//! smallest counter. Eviction is rare (only at capacity), so the
//! O(n) scan is amortized and cheap (4096 u64 comparisons ≈ 10 µs).
//! This trades a doubly-linked list for one extra u64 per entry and
//! avoids the borrow contention.

use std::collections::HashMap;

use thiserror::Error;

/// Default capacity for both outbound and inbound dictionaries. 4096
/// covers realistic per-pair channel counts (a typical ship has <100
/// channels; even a dense mining operation rarely exceeds a few
/// thousand). Memory at full: ~30B name × 2 maps × 4096 ≈ 240 KB per
/// peer pair, which is well under the per-connection budget.
pub const DEFAULT_DICT_CAPACITY: u32 = 4096;

/// Wire flags packed into the entry's `flags: u8`.
///
/// Bit 0 (`FLAG_REGISTER`): when set, the entry carries a fresh
/// (wire_id, channel_name) binding. Receiver inserts/overwrites in
/// its inbound dict before resolving the entry.
///
/// Bits 1-7: reserved for future use (e.g., compression flags, value
/// type extensions). Allocate from low bits to keep `flags == 0` the
/// "plain entry" common case.
pub const FLAG_REGISTER: u8 = 0x01;

/// Outcome of [`OutboundDict::intern`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternResult {
    /// The wire_id to put on the entry (always present).
    pub wire_id: u32,
    /// True when the caller must include `FLAG_REGISTER` and the
    /// channel_name in the wire entry. False means the receiver
    /// already knows this binding from a prior batch and the entry
    /// can ship just the wire_id.
    pub needs_register: bool,
    /// When non-None, an existing entry was evicted to make room for
    /// this registration. The receiver does NOT need to know about
    /// the eviction — it overwrites its own binding when it sees the
    /// new FLAG_REGISTER. This field is exposed for telemetry and
    /// debug only.
    pub evicted_wire_id: Option<u32>,
}

/// Sender-side dictionary. Tracks which channel names have been
/// announced to a specific peer and what wire_id they map to.
///
/// One instance per peer connection. Created on connect, dropped on
/// disconnect.
#[derive(Debug)]
pub struct OutboundDict {
    /// Forward index: channel name → wire_id.
    name_to_wire: HashMap<String, u32>,
    /// Reverse index + LRU metadata: wire_id → (name, last_used_seq).
    /// We carry the name here too so eviction doesn't need a separate
    /// scan and so we can report the evicted name in telemetry.
    wire_to_meta: HashMap<u32, OutboundEntry>,
    /// Next wire_id to assign while still below capacity. Once we hit
    /// capacity we eviction-recycle ids instead of advancing this.
    next_id: u32,
    /// Maximum entries this dict will hold. See `DEFAULT_DICT_CAPACITY`.
    capacity: u32,
    /// Strictly monotonic. Bumped on every register and every evict.
    /// Receiver verifies against its `last_dict_seq` to detect drift.
    dict_seq: u64,
    /// LRU access counter. Each `intern` bumps it; entries store
    /// their last value to make eviction O(n) over the dict (cheap
    /// at our capacity).
    access_counter: u64,
}

#[derive(Debug, Clone)]
struct OutboundEntry {
    name: String,
    last_used_seq: u64,
}

impl Default for OutboundDict {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_DICT_CAPACITY)
    }
}

impl OutboundDict {
    /// Construct with an explicit capacity. `capacity` of 0 is illegal
    /// at the data-structure level (would deadlock the eviction path);
    /// we clamp to 1 with a warning rather than panic — keeps test
    /// fixtures simple.
    pub fn with_capacity(capacity: u32) -> Self {
        let capacity = capacity.max(1);
        Self {
            name_to_wire: HashMap::with_capacity(capacity as usize),
            wire_to_meta: HashMap::with_capacity(capacity as usize),
            next_id: 0,
            capacity,
            dict_seq: 0,
            access_counter: 0,
        }
    }

    /// Look up or assign a wire_id for `name`.
    ///
    /// On a hit (name already known to this peer), returns
    /// `needs_register=false` and the existing wire_id. On a miss,
    /// allocates a fresh wire_id (evicting LRU if at capacity), bumps
    /// `dict_seq`, and returns `needs_register=true` so the caller
    /// knows to include the name + FLAG_REGISTER on the wire.
    pub fn intern(&mut self, name: &str) -> InternResult {
        // Hot path: existing binding. Two HashMap reads but no
        // allocation; the `name` borrow is forwarded into the lookup.
        self.access_counter = self.access_counter.wrapping_add(1);
        if let Some(&wire_id) = self.name_to_wire.get(name) {
            // Update LRU metadata. The Some-branch is hit-rate-driven
            // common case at scale; we accept the extra HashMap touch
            // because it's cheap and keeps eviction fair.
            if let Some(meta) = self.wire_to_meta.get_mut(&wire_id) {
                meta.last_used_seq = self.access_counter;
            }
            return InternResult {
                wire_id,
                needs_register: false,
                evicted_wire_id: None,
            };
        }

        // Miss path: allocate a wire_id. Either there's room (advance
        // `next_id`) or we're at capacity and must evict LRU.
        let (wire_id, evicted_wire_id) = if self.wire_to_meta.len() < self.capacity as usize {
            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1);
            (id, None)
        } else {
            // Capacity hit — find the least-recently-used entry. Linear
            // scan over `wire_to_meta`; at 4096 entries this is ~10µs
            // and only happens after the dict is full.
            let evict_id = self
                .wire_to_meta
                .iter()
                .min_by_key(|(_, meta)| meta.last_used_seq)
                .map(|(id, _)| *id)
                .expect("non-empty map at capacity");
            // Drop both indices.
            let evicted = self
                .wire_to_meta
                .remove(&evict_id)
                .expect("just confirmed via min_by_key");
            self.name_to_wire.remove(&evicted.name);
            (evict_id, Some(evict_id))
        };

        // Insert the new binding.
        self.name_to_wire.insert(name.to_owned(), wire_id);
        self.wire_to_meta.insert(
            wire_id,
            OutboundEntry {
                name: name.to_owned(),
                last_used_seq: self.access_counter,
            },
        );
        // dict_seq bumps on every state change so the receiver can
        // detect drift via simple monotonicity check.
        self.dict_seq = self.dict_seq.wrapping_add(1);

        InternResult {
            wire_id,
            needs_register: true,
            evicted_wire_id,
        }
    }

    /// Current dict_seq. Each batch send embeds this so the receiver
    /// can verify monotonicity.
    pub fn dict_seq(&self) -> u64 {
        self.dict_seq
    }

    /// Number of live bindings.
    pub fn len(&self) -> usize {
        self.wire_to_meta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.wire_to_meta.is_empty()
    }

    /// Capacity (immutable after construction).
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Reset all state. Called on connection re-establishment.
    pub fn reset(&mut self) {
        self.name_to_wire.clear();
        self.wire_to_meta.clear();
        self.next_id = 0;
        self.dict_seq = 0;
        self.access_counter = 0;
    }
}

/// Receiver-side dictionary. Resolves wire_ids back to channel names.
///
/// One instance per peer connection. Reset on disconnect.
#[derive(Debug)]
pub struct InboundDict {
    /// Resolution table: wire_id → channel name.
    wire_to_name: HashMap<u32, String>,
    /// Highest dict_seq we've accepted so far. Each incoming batch's
    /// `dict_seq` must be >= this.
    last_dict_seq: u64,
    /// Capacity. Must match (or exceed) the sender's capacity for the
    /// dicts to stay coherent without protocol-level re-registration
    /// hints. We over-provision the receiver by default.
    capacity: u32,
}

/// Reasons a `register` call rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InboundDictError {
    /// The batch's `dict_seq` is older than the highest seq we've
    /// already accepted. Either packet reordering (rare with QUIC's
    /// in-order streams) or a sender restart that didn't reset the
    /// connection. Receiver should drop the batch and rely on the
    /// sender's next batch to bump seq forward.
    #[error("dict_seq drift: incoming {incoming} < last_accepted {last_accepted}")]
    SeqDrift { incoming: u64, last_accepted: u64 },
}

impl Default for InboundDict {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_DICT_CAPACITY)
    }
}

impl InboundDict {
    pub fn with_capacity(capacity: u32) -> Self {
        let capacity = capacity.max(1);
        Self {
            wire_to_name: HashMap::with_capacity(capacity as usize),
            last_dict_seq: 0,
            capacity,
        }
    }

    /// Verify a batch's `dict_seq` against the highest we've accepted.
    /// Call this once per batch BEFORE iterating entries. Equality is
    /// allowed: a sender that doesn't change its dict between batches
    /// keeps the same `dict_seq`, which is the steady-state hot path.
    pub fn verify_seq(&mut self, dict_seq: u64) -> Result<(), InboundDictError> {
        if dict_seq < self.last_dict_seq {
            return Err(InboundDictError::SeqDrift {
                incoming: dict_seq,
                last_accepted: self.last_dict_seq,
            });
        }
        self.last_dict_seq = dict_seq;
        Ok(())
    }

    /// Process a `FLAG_REGISTER` entry: bind `wire_id → name`.
    /// Overwrites any existing binding for that wire_id (the sender
    /// has evicted-and-reused it). If the dict is full, drop the
    /// oldest binding by removing one arbitrary entry — the receiver's
    /// LRU eviction is naive on purpose, since the sender drives the
    /// global ordering and the receiver just keeps up.
    pub fn register(&mut self, wire_id: u32, name: String) {
        if !self.wire_to_name.contains_key(&wire_id)
            && self.wire_to_name.len() >= self.capacity as usize
        {
            // Receiver eviction: drop one entry to make room. We don't
            // track LRU here because the sender's order is the only
            // ground truth. Picking *any* entry is sound — the sender
            // will re-register if it later references that wire_id.
            // Picking deterministically (smallest wire_id) makes
            // tests reproducible without affecting correctness.
            if let Some(&victim) = self.wire_to_name.keys().min() {
                self.wire_to_name.remove(&victim);
            }
        }
        self.wire_to_name.insert(wire_id, name);
    }

    /// Resolve `wire_id` to a channel name. Returns `None` if the
    /// sender referenced an id we don't have — caller must trigger a
    /// resync (drop batch + log + bump metric).
    pub fn resolve(&self, wire_id: u32) -> Option<&str> {
        self.wire_to_name.get(&wire_id).map(String::as_str)
    }

    /// Live binding count.
    pub fn len(&self) -> usize {
        self.wire_to_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.wire_to_name.is_empty()
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn last_dict_seq(&self) -> u64 {
        self.last_dict_seq
    }

    /// Full reset. Called after a resync handshake or on connection
    /// re-establishment.
    pub fn reset(&mut self) {
        self.wire_to_name.clear();
        self.last_dict_seq = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------- OutboundDict ------------------

    #[test]
    fn outbound_first_intern_assigns_id_and_marks_register() {
        let mut d = OutboundDict::default();
        let r = d.intern("alice.thrust");
        assert_eq!(r.wire_id, 0, "first id starts at 0");
        assert!(r.needs_register, "first sighting must register");
        assert_eq!(r.evicted_wire_id, None);
        assert_eq!(d.len(), 1);
        assert_eq!(d.dict_seq(), 1, "register bumps seq");
    }

    #[test]
    fn outbound_repeat_intern_hits_cache_does_not_register() {
        let mut d = OutboundDict::default();
        let first = d.intern("alice.thrust");
        let second = d.intern("alice.thrust");
        assert_eq!(first.wire_id, second.wire_id);
        assert!(!second.needs_register);
        assert_eq!(d.dict_seq(), 1, "repeat lookup does not bump seq");
    }

    #[test]
    fn outbound_distinct_names_get_distinct_ids() {
        let mut d = OutboundDict::default();
        let a = d.intern("a").wire_id;
        let b = d.intern("b").wire_id;
        let c = d.intern("c").wire_id;
        // All distinct.
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn outbound_eviction_at_capacity_recycles_lru_id() {
        let mut d = OutboundDict::with_capacity(2);
        let a = d.intern("a").wire_id; // last_used 1
        let b = d.intern("b").wire_id; // last_used 2
        // Promote `a`: now b is LRU.
        let _ = d.intern("a"); // last_used 3 (a)
        let r = d.intern("c");
        assert!(r.needs_register);
        assert_eq!(
            r.evicted_wire_id,
            Some(b),
            "LRU `b` must be evicted, not `a`"
        );
        assert_eq!(r.wire_id, b, "freed wire_id is reused");
        // After eviction `b` is no longer resolvable.
        let r2 = d.intern("b");
        assert!(r2.needs_register, "re-registering `b` must register fresh");
        assert_ne!(r2.wire_id, r.wire_id, "must not reuse `c`'s slot");
    }

    #[test]
    fn outbound_dict_seq_bumps_on_register_and_evict() {
        let mut d = OutboundDict::with_capacity(2);
        d.intern("a"); // seq 1
        d.intern("b"); // seq 2
        let _ = d.intern("a"); // seq unchanged (cache hit)
        assert_eq!(d.dict_seq(), 2);
        d.intern("c"); // evict + register → seq 3
        assert_eq!(d.dict_seq(), 3);
    }

    #[test]
    fn outbound_reset_clears_state() {
        let mut d = OutboundDict::default();
        d.intern("a");
        d.intern("b");
        d.reset();
        assert_eq!(d.len(), 0);
        assert_eq!(d.dict_seq(), 0);
        let r = d.intern("a");
        assert_eq!(r.wire_id, 0, "post-reset id sequence restarts");
        assert!(r.needs_register);
    }

    #[test]
    fn outbound_capacity_zero_clamps_to_one() {
        // We refuse to construct a 0-capacity dict (would deadlock the
        // eviction path). The clamp must not panic.
        let mut d = OutboundDict::with_capacity(0);
        assert_eq!(d.capacity(), 1);
        let r1 = d.intern("a");
        assert!(r1.needs_register);
        // Second intern evicts the first.
        let r2 = d.intern("b");
        assert!(r2.needs_register);
        assert_eq!(r2.evicted_wire_id, Some(r1.wire_id));
    }

    // ---------------- InboundDict ------------------

    #[test]
    fn inbound_register_and_resolve() {
        let mut d = InboundDict::default();
        d.register(7, "alice.thrust".into());
        assert_eq!(d.resolve(7), Some("alice.thrust"));
        assert_eq!(d.resolve(8), None, "unregistered id returns None");
    }

    #[test]
    fn inbound_register_overwrites_existing_binding() {
        // Sender evicted wire_id 7 and reused it for a new channel.
        let mut d = InboundDict::default();
        d.register(7, "old.channel".into());
        d.register(7, "new.channel".into());
        assert_eq!(d.resolve(7), Some("new.channel"));
    }

    #[test]
    fn inbound_verify_seq_accepts_monotonic_advance_and_equality() {
        let mut d = InboundDict::default();
        assert!(d.verify_seq(0).is_ok());
        assert!(d.verify_seq(1).is_ok());
        assert!(d.verify_seq(1).is_ok(), "equality is allowed (steady state)");
        assert!(d.verify_seq(5).is_ok(), "skip-ahead is allowed");
    }

    #[test]
    fn inbound_verify_seq_rejects_backward_drift() {
        let mut d = InboundDict::default();
        d.verify_seq(10).unwrap();
        let err = d.verify_seq(9).unwrap_err();
        assert_eq!(
            err,
            InboundDictError::SeqDrift {
                incoming: 9,
                last_accepted: 10,
            }
        );
        // After drift, last_accepted is unchanged so future inputs
        // are gated by the same threshold.
        assert!(d.verify_seq(10).is_ok());
    }

    #[test]
    fn inbound_eviction_drops_some_entry_when_full() {
        let mut d = InboundDict::with_capacity(2);
        d.register(1, "a".into());
        d.register(2, "b".into());
        d.register(3, "c".into()); // forces eviction
        assert_eq!(d.len(), 2);
        // wire_id=3 must be present (just registered).
        assert_eq!(d.resolve(3), Some("c"));
    }

    #[test]
    fn inbound_reset_clears_state() {
        let mut d = InboundDict::default();
        d.register(1, "a".into());
        d.verify_seq(42).unwrap();
        d.reset();
        assert_eq!(d.len(), 0);
        assert_eq!(d.last_dict_seq(), 0);
        assert!(d.verify_seq(0).is_ok(), "post-reset accepts seq 0");
    }

    // ---------------- Round-trip ------------------

    /// Drive a sender's intern stream through a receiver's
    /// register/resolve to confirm the dicts converge in lockstep.
    /// This is the core happy-path invariant.
    #[test]
    fn round_trip_sender_receiver_converge_under_steady_state() {
        let mut out = OutboundDict::default();
        let mut inn = InboundDict::default();

        let names = ["a", "b", "c", "a", "b", "d", "a"];
        for name in names {
            let r = out.intern(name);
            // Receiver tracks dict_seq monotonicity per batch.
            inn.verify_seq(out.dict_seq()).unwrap();
            if r.needs_register {
                inn.register(r.wire_id, name.to_string());
            }
            assert_eq!(
                inn.resolve(r.wire_id),
                Some(name),
                "receiver must resolve every wire_id the sender intended"
            );
        }
        // After 4 distinct names: dict has 4 entries on both sides.
        assert_eq!(out.len(), 4);
        assert_eq!(inn.len(), 4);
    }

    /// Capacity overflow on the sender forces an eviction; the
    /// receiver overwrites its binding cleanly without state-tracking
    /// on the eviction path.
    #[test]
    fn round_trip_with_capacity_overflow_recovers_cleanly() {
        let mut out = OutboundDict::with_capacity(2);
        let mut inn = InboundDict::with_capacity(2);

        // Fill: a, b → both registered.
        let ra = out.intern("a");
        inn.verify_seq(out.dict_seq()).unwrap();
        if ra.needs_register {
            inn.register(ra.wire_id, "a".into());
        }
        let rb = out.intern("b");
        inn.verify_seq(out.dict_seq()).unwrap();
        if rb.needs_register {
            inn.register(rb.wire_id, "b".into());
        }

        // Overflow with `c` — `a` is LRU and gets evicted.
        let rc = out.intern("c");
        inn.verify_seq(out.dict_seq()).unwrap();
        assert!(rc.needs_register);
        assert_eq!(rc.evicted_wire_id, Some(ra.wire_id));
        if rc.needs_register {
            // Receiver overwrites its old binding for the recycled id.
            inn.register(rc.wire_id, "c".into());
        }
        // After overwrite, the recycled id resolves to `c`, and the
        // old `a` binding is gone.
        assert_eq!(inn.resolve(rc.wire_id), Some("c"));
        // `b` survives.
        assert_eq!(inn.resolve(rb.wire_id), Some("b"));
    }
}
