//! Phase 3F.9 — system-shard's per-ship Radio frequency interest tracker.
//!
//! Ships periodically send `ShardMsg::ShipFrequencyInterest` to their
//! host system-shard with the COMPLETE current set of Radio
//! frequencies they have Listener blocks for. The system-shard
//! aggregates across all hosted ships and propagates a precise
//! per-frequency `RadioSubscribe` to galaxy. This eliminates the
//! wildcard-everywhere subscription pattern and saves bandwidth at
//! 100K-user scale.
//!
//! # Why full-set semantics
//!
//! Each `ShipFrequencyInterest` carries the ship's COMPLETE current
//! freq set, not a delta. Trade-offs:
//!
//! - **Pro**: self-healing. A dropped message just delays
//!   convergence by one renewal cycle (90s); no diff state to lose
//!   on the wire.
//! - **Pro**: simple lease model. Ship-shard renews periodically;
//!   system-shard expires stale entries automatically. No explicit
//!   teardown handshake needed.
//! - **Con**: each message is `O(freq_count)` bytes. At realistic
//!   ship-listener counts (<10 per ship) this is trivial.
//!
//! # Aggregation semantics
//!
//! `aggregated_frequencies()` returns the UNION across every active
//! ship's set. A frequency is "active" if at least one hosted ship
//! has it listed AND the ship's lease hasn't expired. This is what
//! the system-shard's reconciler ships to galaxy as a `RadioSubscribe`
//! payload.
//!
//! # Why not store the union eagerly
//!
//! Computing the union on demand is `O(ships × freqs_per_ship)` —
//! cheap at MVP scale (hundreds of ships per system × <10 freqs).
//! Eagerly maintaining the union would add tracking complexity
//! (refcount per freq) that buys nothing measurable. Recompute on
//! reconcile is the AAA-quality call.

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::Resource;

use crate::shard_types::ShardId;

/// One ship's current freq interest set + its lease deadline.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ShipInterest {
    frequencies: HashSet<u32>,
    lease_until_ms: u64,
}

/// System-shard tracker for `ShipFrequencyInterest` messages from
/// its hosted ships.
#[derive(Resource, Debug, Default)]
pub struct ShipFrequencyInterests {
    by_ship: HashMap<ShardId, ShipInterest>,
}

impl ShipFrequencyInterests {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest a `ShipFrequencyInterest` payload. Replaces this ship's
    /// entire freq set. An empty `frequencies` set DROPS the ship's
    /// entry entirely — the ship is signalling "I no longer have any
    /// Radio listeners," and we should stop counting it toward the
    /// union.
    ///
    /// `lease_until_ms` should never decrease for an existing ship —
    /// we accept the max of (existing, new) so a misbehaving sender
    /// can't accidentally evict its own interest by sending a
    /// shorter lease.
    pub fn upsert(
        &mut self,
        ship_shard: ShardId,
        frequencies: impl IntoIterator<Item = u32>,
        lease_until_ms: u64,
    ) {
        let freq_set: HashSet<u32> = frequencies.into_iter().collect();
        if freq_set.is_empty() {
            self.by_ship.remove(&ship_shard);
            return;
        }
        self.by_ship
            .entry(ship_shard)
            .and_modify(|existing| {
                existing.frequencies = freq_set.clone();
                existing.lease_until_ms = existing.lease_until_ms.max(lease_until_ms);
            })
            .or_insert_with(|| ShipInterest {
                frequencies: freq_set,
                lease_until_ms,
            });
    }

    /// Drop a ship's entry — used when the ship-shard goes away
    /// (handoff, crash, orchestrator removal).
    pub fn drop_ship(&mut self, ship_shard: ShardId) {
        self.by_ship.remove(&ship_shard);
    }

    /// Sweep ships whose lease has expired. Call at low cadence
    /// (1Hz is plenty — leases are minute-scale). Returns count
    /// dropped for the gauge.
    pub fn cleanup_expired(&mut self, now_ms: u64) -> usize {
        let before = self.by_ship.len();
        self.by_ship.retain(|_ship, entry| entry.lease_until_ms > now_ms);
        before - self.by_ship.len()
    }

    /// Union of frequencies across every active (non-expired) ship.
    /// This is what the system-shard's reconciler ships upward.
    pub fn aggregated_frequencies(&self, now_ms: u64) -> HashSet<u32> {
        let mut out = HashSet::new();
        for entry in self.by_ship.values() {
            if entry.lease_until_ms > now_ms {
                out.extend(entry.frequencies.iter().copied());
            }
        }
        out
    }

    /// Live ship count (entries not yet expired).
    pub fn active_ship_count(&self, now_ms: u64) -> usize {
        self.by_ship
            .values()
            .filter(|e| e.lease_until_ms > now_ms)
            .count()
    }

    /// Total entries (including any whose lease has just lapsed but
    /// not been swept). Drives the `signal_ship_freq_interests` gauge.
    pub fn total_count(&self) -> usize {
        self.by_ship.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(id: u64) -> ShardId {
        ShardId(id)
    }

    #[test]
    fn upsert_replaces_ship_full_set() {
        let mut t = ShipFrequencyInterests::new();
        t.upsert(s(1), [100, 200], 1_000);
        let agg: Vec<u32> = {
            let mut v: Vec<u32> = t.aggregated_frequencies(0).into_iter().collect();
            v.sort();
            v
        };
        assert_eq!(agg, vec![100, 200]);
        // Replace with a different set — old freqs disappear.
        t.upsert(s(1), [300], 1_000);
        let agg: Vec<u32> = {
            let mut v: Vec<u32> = t.aggregated_frequencies(0).into_iter().collect();
            v.sort();
            v
        };
        assert_eq!(agg, vec![300]);
    }

    #[test]
    fn upsert_empty_set_drops_ship() {
        let mut t = ShipFrequencyInterests::new();
        t.upsert(s(1), [100], 1_000);
        assert_eq!(t.total_count(), 1);
        // Empty interest = "remove me from tracking."
        t.upsert(s(1), std::iter::empty(), 1_000);
        assert_eq!(t.total_count(), 0);
        assert!(t.aggregated_frequencies(0).is_empty());
    }

    #[test]
    fn upsert_extends_lease_takes_max() {
        let mut t = ShipFrequencyInterests::new();
        t.upsert(s(1), [100], 5_000);
        // A shorter lease must NOT shrink the existing entry.
        t.upsert(s(1), [100], 1_000);
        // At now=4_999 the entry is still alive (max kept).
        let dropped = t.cleanup_expired(4_999);
        assert_eq!(dropped, 0);
    }

    #[test]
    fn aggregated_frequencies_unions_across_ships() {
        let mut t = ShipFrequencyInterests::new();
        t.upsert(s(1), [100, 200], 1_000);
        t.upsert(s(2), [200, 300], 1_000);
        let mut agg: Vec<u32> = t.aggregated_frequencies(0).into_iter().collect();
        agg.sort();
        assert_eq!(agg, vec![100, 200, 300], "union dedupes the shared 200");
    }

    #[test]
    fn aggregated_frequencies_skips_expired_ships() {
        let mut t = ShipFrequencyInterests::new();
        t.upsert(s(1), [100], 500);
        t.upsert(s(2), [200], 5_000);
        // At now=2_000, ship 1 is expired but not yet swept.
        let mut agg: Vec<u32> = t.aggregated_frequencies(2_000).into_iter().collect();
        agg.sort();
        assert_eq!(agg, vec![200], "expired ship contributes nothing");
    }

    #[test]
    fn cleanup_expired_drops_only_stale_ships() {
        let mut t = ShipFrequencyInterests::new();
        t.upsert(s(1), [100], 500);
        t.upsert(s(2), [200], 5_000);
        let dropped = t.cleanup_expired(1_000);
        assert_eq!(dropped, 1);
        assert_eq!(t.total_count(), 1);
        // Ship 2's entry survives.
        let agg: HashSet<u32> = t.aggregated_frequencies(1_000);
        assert!(agg.contains(&200));
    }

    #[test]
    fn drop_ship_removes_entry_immediately() {
        let mut t = ShipFrequencyInterests::new();
        t.upsert(s(1), [100], 1_000);
        assert_eq!(t.total_count(), 1);
        t.drop_ship(s(1));
        assert_eq!(t.total_count(), 0);
        // Idempotent — re-drop is a no-op.
        t.drop_ship(s(1));
    }

    #[test]
    fn active_ship_count_filters_by_now_ms() {
        let mut t = ShipFrequencyInterests::new();
        t.upsert(s(1), [100], 500);
        t.upsert(s(2), [200], 5_000);
        assert_eq!(t.active_ship_count(0), 2);
        assert_eq!(t.active_ship_count(1_000), 1);
        assert_eq!(t.active_ship_count(10_000), 0);
        // total_count still 2 — cleanup_expired hasn't run yet.
        assert_eq!(t.total_count(), 2);
    }
}
