//! Phase 3F — galaxy-shard subscriber registry for Radio-scope signal
//! relay.
//!
//! The galaxy-shard hosts this resource. A system-shard sends
//! `RadioSubscribe { subscriber_shard_id, frequencies | wildcard,
//! lease_until_ms }` to register its interest in Radio traffic; the
//! galaxy stores the subscription here and consults it when fanning
//! out incoming `SignalBroadcastBatch`/`V2` messages of scope=Radio.
//!
//! # Two subscription kinds
//!
//! - **Per-frequency**: a system-shard knows exactly which frequencies
//!   it has Listener blocks for, and registers them. Bandwidth-optimal
//!   — the galaxy fans out only matching batches.
//! - **Wildcard**: the system-shard wants every Radio batch regardless
//!   of frequency. Used as the MVP default while ship-shard listener
//!   tracking → system-shard frequency-set propagation is being built.
//!
//! Both kinds carry independent leases per-subscriber. A shard can
//! hold a wildcard plus per-frequency entries simultaneously without
//! conflict; the fan-out path checks both and dedupes by ShardId.
//!
//! # Lease lifecycle
//!
//! `lease_until_ms` is UNIX wall-clock millis. Galaxy compares against
//! its own clock during lookup and during the periodic
//! `cleanup_expired` sweep. Senders are expected to renew before
//! expiry; a missed renewal silently drops the subscription so a
//! crashed/disconnected shard cannot keep eating relay bandwidth.
//!
//! # Why HashMaps over `SmallVec<[ListenerEntry; 8]>` (per the plan)
//!
//! The plan sketches `HashMap<u32, SmallVec<[ListenerEntry; 8]>>`
//! optimizing the common-case "few subscribers per frequency". At
//! 100K-user scale the worst case is hundreds of system-shards
//! subscribing to popular frequencies (faction comms, emergency
//! channels). HashMap-of-HashMap keeps insertion/lookup O(1) without
//! the linear-scan-on-add cost of SmallVec at high cardinality. We
//! pay an extra HashMap allocation per frequency, which is
//! negligible compared to the resolution speed-up.

use std::collections::HashMap;

use bevy_ecs::prelude::Resource;

use crate::shard_types::ShardId;

/// Galaxy-shard's subscriber registry for Radio relay.
#[derive(Resource, Debug, Default)]
pub struct RadioSubscribers {
    /// Per-frequency entries: frequency → (subscriber → lease deadline).
    /// Inserted on `RadioSubscribe`, removed on `RadioUnsubscribe` /
    /// expiry / explicit drop.
    by_frequency: HashMap<u32, HashMap<ShardId, u64>>,
    /// Wildcard entries: subscriber → lease deadline. Wildcards
    /// receive every Radio batch regardless of frequency.
    wildcards: HashMap<ShardId, u64>,
}

impl RadioSubscribers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a subscription. `lease_until_ms` is UNIX millis; galaxy
    /// drops the entry on expiry. Idempotent — re-subscribing the same
    /// `(shard, frequency)` extends the lease.
    pub fn add(
        &mut self,
        shard_id: ShardId,
        frequencies: &[u32],
        wildcard: bool,
        lease_until_ms: u64,
    ) {
        if wildcard {
            // Wildcard subscription: extend or create the entry. We
            // treat lease_until_ms as a max — re-subscribing with an
            // earlier deadline does not shorten an existing lease.
            // Stops a misbehaving sender from accidentally evicting
            // its own active wildcard.
            self.wildcards
                .entry(shard_id)
                .and_modify(|d| *d = (*d).max(lease_until_ms))
                .or_insert(lease_until_ms);
        }
        for &freq in frequencies {
            let entry = self.by_frequency.entry(freq).or_default();
            entry
                .entry(shard_id)
                .and_modify(|d| *d = (*d).max(lease_until_ms))
                .or_insert(lease_until_ms);
        }
    }

    /// Drop a subscription. Idempotent — non-existent entries are a
    /// no-op. `wildcard=true` removes the wildcard entry; per-freq
    /// values are removed independently.
    pub fn remove(&mut self, shard_id: ShardId, frequencies: &[u32], wildcard: bool) {
        if wildcard {
            self.wildcards.remove(&shard_id);
        }
        for &freq in frequencies {
            if let Some(entry) = self.by_frequency.get_mut(&freq) {
                entry.remove(&shard_id);
                if entry.is_empty() {
                    self.by_frequency.remove(&freq);
                }
            }
        }
    }

    /// Drop EVERY subscription for a shard — wildcard + every
    /// per-frequency entry. Called when the galaxy detects a peer
    /// has disappeared (orchestrator removed it from the peer list)
    /// without sending a clean unsubscribe.
    pub fn drop_shard(&mut self, shard_id: ShardId) {
        self.wildcards.remove(&shard_id);
        self.by_frequency.retain(|_freq, entry| {
            entry.remove(&shard_id);
            !entry.is_empty()
        });
    }

    /// Iterator of every shard subscribed to `freq` whose lease is
    /// still active at `now_ms`. Wildcard subscribers + per-freq
    /// subscribers both qualify; deduped within the call so a shard
    /// holding both kinds shows up exactly once.
    ///
    /// Non-allocating in the steady state when there are no
    /// per-frequency subscribers AND no wildcards (the iterator just
    /// closes); allocates a small dedup set otherwise.
    pub fn subscribers_for(
        &self,
        freq: u32,
        now_ms: u64,
    ) -> impl Iterator<Item = ShardId> + '_ {
        // Materialize into a Vec for dedup. At MVP scale the per-freq
        // count is small (tens at most); the wildcard count is bounded
        // by system-shard count. Worst-case allocation is tiny.
        let mut shards: Vec<ShardId> = Vec::new();
        if let Some(entry) = self.by_frequency.get(&freq) {
            for (shard, deadline) in entry {
                if *deadline > now_ms {
                    shards.push(*shard);
                }
            }
        }
        for (shard, deadline) in &self.wildcards {
            if *deadline > now_ms && !shards.contains(shard) {
                shards.push(*shard);
            }
        }
        shards.into_iter()
    }

    /// Sweep expired subscriptions. Call at low cadence (1 Hz is
    /// plenty — leases are minute-scale). Returns the number of
    /// entries dropped, useful for the `signal_radio_subscriber_*`
    /// metrics.
    pub fn cleanup_expired(&mut self, now_ms: u64) -> usize {
        let mut dropped = 0usize;
        // Wildcard sweep.
        self.wildcards.retain(|_shard, deadline| {
            let alive = *deadline > now_ms;
            if !alive {
                dropped += 1;
            }
            alive
        });
        // Per-frequency sweep — drop expired entries, then drop now-
        // empty frequency buckets to keep the outer map lean.
        self.by_frequency.retain(|_freq, entry| {
            entry.retain(|_shard, deadline| {
                let alive = *deadline > now_ms;
                if !alive {
                    dropped += 1;
                }
                alive
            });
            !entry.is_empty()
        });
        dropped
    }

    /// Total subscriptions including wildcard + per-freq entries.
    /// Drives the `signal_radio_subscribers` gauge.
    pub fn total_count(&self) -> usize {
        self.wildcards.len()
            + self.by_frequency.values().map(|m| m.len()).sum::<usize>()
    }

    /// Distinct frequencies tracked.
    pub fn frequency_count(&self) -> usize {
        self.by_frequency.len()
    }

    /// Distinct wildcard subscribers.
    pub fn wildcard_count(&self) -> usize {
        self.wildcards.len()
    }

    /// Group `entries` into per-subscriber buckets for a Radio fan-out.
    ///
    /// For each entry, `frequency_of(&entry)` selects the subscriber
    /// set; the entry is `Clone`d into every matching subscriber's
    /// bucket. Wildcard subscribers receive every entry regardless of
    /// frequency. The source shard is excluded — a sender's own batch
    /// must never bounce back to it.
    ///
    /// Generic over the entry type via the `frequency_of` projection
    /// so this works uniformly for `SignalBroadcastEntry` (V1) and
    /// for resolved V2 batches (which become V1-shaped after
    /// `decode_v2_batch`). One implementation, two callers.
    pub fn group_for_relay<E, F>(
        &self,
        source_shard: ShardId,
        entries: &[E],
        now_ms: u64,
        frequency_of: F,
    ) -> HashMap<ShardId, Vec<E>>
    where
        E: Clone,
        F: Fn(&E) -> u32,
    {
        let mut buckets: HashMap<ShardId, Vec<E>> = HashMap::new();
        for entry in entries {
            for sub in self.subscribers_for(frequency_of(entry), now_ms) {
                if sub == source_shard {
                    continue;
                }
                buckets.entry(sub).or_default().push(entry.clone());
            }
        }
        buckets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(id: u64) -> ShardId {
        ShardId(id)
    }

    #[test]
    fn add_per_frequency_resolves_via_subscribers_for() {
        let mut r = RadioSubscribers::new();
        r.add(s(1), &[100, 200], false, 1_000);
        let mut subs: Vec<_> = r.subscribers_for(100, 0).collect();
        subs.sort_by_key(|s| s.0);
        assert_eq!(subs, vec![s(1)]);
        let other: Vec<_> = r.subscribers_for(999, 0).collect();
        assert!(other.is_empty(), "non-subscribed freq returns empty");
    }

    #[test]
    fn add_wildcard_resolves_for_any_frequency() {
        let mut r = RadioSubscribers::new();
        r.add(s(1), &[], true, 1_000);
        let subs1: Vec<_> = r.subscribers_for(100, 0).collect();
        let subs2: Vec<_> = r.subscribers_for(9999, 0).collect();
        assert_eq!(subs1, vec![s(1)]);
        assert_eq!(subs2, vec![s(1)]);
    }

    #[test]
    fn wildcard_and_per_freq_subscribers_dedupe() {
        let mut r = RadioSubscribers::new();
        // Same shard holds both. Should appear exactly once in the
        // resolution.
        r.add(s(1), &[100], false, 1_000);
        r.add(s(1), &[], true, 1_000);
        let subs: Vec<_> = r.subscribers_for(100, 0).collect();
        assert_eq!(subs, vec![s(1)], "dedup across kinds");
    }

    #[test]
    fn lease_extension_picks_max_deadline() {
        let mut r = RadioSubscribers::new();
        r.add(s(1), &[100], false, 5_000);
        // A shorter lease must NOT shorten the existing entry.
        r.add(s(1), &[100], false, 1_000);
        // Sweeping at 4_999 keeps it alive (max was 5_000).
        let dropped = r.cleanup_expired(4_999);
        assert_eq!(dropped, 0);
    }

    #[test]
    fn cleanup_expired_drops_per_freq_and_wildcards() {
        let mut r = RadioSubscribers::new();
        r.add(s(1), &[100], false, 1_000);
        r.add(s(2), &[], true, 2_000);
        // Sweep at 1_500: drops shard 1 only.
        assert_eq!(r.cleanup_expired(1_500), 1);
        let subs: Vec<_> = r.subscribers_for(100, 1_500).collect();
        assert!(!subs.contains(&s(1)));
        // Wildcard shard 2 still wins for any frequency.
        let subs2: Vec<_> = r.subscribers_for(100, 1_500).collect();
        assert_eq!(subs2, vec![s(2)]);
        // Cleanup at 2_500 drops shard 2.
        assert_eq!(r.cleanup_expired(2_500), 1);
        assert_eq!(r.total_count(), 0);
    }

    #[test]
    fn cleanup_drops_empty_frequency_buckets() {
        let mut r = RadioSubscribers::new();
        r.add(s(1), &[100], false, 1_000);
        assert_eq!(r.frequency_count(), 1);
        r.cleanup_expired(2_000);
        assert_eq!(r.frequency_count(), 0, "empty freq bucket must be removed");
    }

    #[test]
    fn remove_idempotent_on_missing_entries() {
        let mut r = RadioSubscribers::new();
        // Removing what's never been added is a no-op.
        r.remove(s(1), &[100], false);
        r.remove(s(1), &[], true);
        assert_eq!(r.total_count(), 0);
    }

    #[test]
    fn drop_shard_removes_every_entry() {
        let mut r = RadioSubscribers::new();
        r.add(s(1), &[100, 200], false, 1_000);
        r.add(s(1), &[], true, 1_000);
        r.add(s(2), &[100], false, 1_000);
        // Pre-drop: 4 total (1's wildcard, 1's two freqs, 2's freq).
        assert_eq!(r.total_count(), 4);
        r.drop_shard(s(1));
        // Only shard 2's freq=100 remains.
        assert_eq!(r.total_count(), 1);
        let subs: Vec<_> = r.subscribers_for(100, 0).collect();
        assert_eq!(subs, vec![s(2)]);
    }

    // ---------------- group_for_relay ------------------

    /// Test entry: just a frequency. The grouping helper is generic
    /// over entry shape; this proves the abstraction works without
    /// pulling in the full `SignalBroadcastEntry` type.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestEntry {
        freq: u32,
        tag: u8,
    }

    fn freq_of(e: &TestEntry) -> u32 {
        e.freq
    }

    #[test]
    fn group_for_relay_picks_per_frequency_subscribers() {
        let mut r = RadioSubscribers::new();
        r.add(s(10), &[100], false, 1_000); // sub10 → freq 100
        r.add(s(11), &[200], false, 1_000); // sub11 → freq 200
        let entries = vec![
            TestEntry { freq: 100, tag: 1 },
            TestEntry { freq: 200, tag: 2 },
            TestEntry { freq: 300, tag: 3 }, // no subscriber
        ];
        let buckets = r.group_for_relay(s(99), &entries, 0, freq_of);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[&s(10)], vec![entries[0].clone()]);
        assert_eq!(buckets[&s(11)], vec![entries[1].clone()]);
    }

    #[test]
    fn group_for_relay_wildcard_gets_all_entries() {
        let mut r = RadioSubscribers::new();
        r.add(s(10), &[], true, 1_000); // wildcard
        let entries = vec![
            TestEntry { freq: 100, tag: 1 },
            TestEntry { freq: 200, tag: 2 },
            TestEntry { freq: 300, tag: 3 },
        ];
        let buckets = r.group_for_relay(s(99), &entries, 0, freq_of);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[&s(10)], entries, "wildcard receives all 3");
    }

    #[test]
    fn group_for_relay_source_never_self_echoes() {
        // A subscriber that is ALSO the source must not appear in
        // any bucket — otherwise a sender would receive its own
        // signals on a bandwidth-amplifying loop.
        let mut r = RadioSubscribers::new();
        r.add(s(7), &[], true, 1_000); // sub7 holds wildcard
        let entries = vec![TestEntry { freq: 100, tag: 1 }];
        let buckets = r.group_for_relay(s(7), &entries, 0, freq_of);
        assert!(
            buckets.is_empty(),
            "source shard must not appear among buckets"
        );
    }

    #[test]
    fn group_for_relay_skips_expired_subscribers() {
        let mut r = RadioSubscribers::new();
        r.add(s(10), &[100], false, 500);
        let entries = vec![TestEntry { freq: 100, tag: 1 }];
        let buckets = r.group_for_relay(s(99), &entries, 1_000, freq_of);
        assert!(buckets.is_empty(), "expired sub must drop out");
    }

    #[test]
    fn group_for_relay_dedupes_across_kinds() {
        // sub10 holds both wildcard AND per-freq for 100. A single
        // entry on freq 100 must appear exactly once in their
        // bucket, not twice.
        let mut r = RadioSubscribers::new();
        r.add(s(10), &[100], false, 1_000);
        r.add(s(10), &[], true, 1_000);
        let entries = vec![TestEntry { freq: 100, tag: 1 }];
        let buckets = r.group_for_relay(s(99), &entries, 0, freq_of);
        assert_eq!(buckets.len(), 1);
        assert_eq!(
            buckets[&s(10)].len(),
            1,
            "wildcard + per-freq for same shard must dedupe"
        );
    }

    /// Cross-system Radio fan-out, end-to-end at the routing layer.
    /// Setup mirrors what production looks like:
    ///   - source: ship-shard 1 (in system A) emits a Radio batch
    ///   - system-shard A forwards to galaxy
    ///   - galaxy fans out to the system-shards subscribed wildcardly
    ///   - destination: ship-shard 2 (in system B) eventually receives
    /// This test exercises the galaxy fan-out step in isolation. The
    /// other steps are exercised by the generic round-trip tests in
    /// the wire-dict layer + the per-shard build verification.
    #[test]
    fn cross_system_radio_routes_to_subscribed_systems_only() {
        let mut r = RadioSubscribers::new();
        // System B and C are wildcardly subscribed; A is the source.
        let system_a = s(101);
        let system_b = s(201);
        let system_c = s(301);
        r.add(system_b, &[], true, 1_000);
        r.add(system_c, &[], true, 1_000);
        let entries = vec![
            TestEntry { freq: 1234, tag: 1 },
            TestEntry { freq: 5678, tag: 2 },
        ];
        let buckets = r.group_for_relay(system_a, &entries, 0, freq_of);
        assert_eq!(buckets.len(), 2, "exactly two subscribed systems");
        assert!(buckets.contains_key(&system_b));
        assert!(buckets.contains_key(&system_c));
        assert!(
            !buckets.contains_key(&system_a),
            "source system never receives its own emission"
        );
        // Each subscribed system gets BOTH entries (wildcard).
        assert_eq!(buckets[&system_b].len(), 2);
        assert_eq!(buckets[&system_c].len(), 2);
    }

    #[test]
    fn subscribers_for_filters_by_now_ms() {
        let mut r = RadioSubscribers::new();
        r.add(s(1), &[100], false, 500);
        // At now_ms=500, the lease is exactly equal — strictly-greater
        // semantics means it's expired (treat the deadline as a lower
        // bound that the current time must not yet have reached).
        let subs: Vec<_> = r.subscribers_for(100, 500).collect();
        assert!(
            subs.is_empty(),
            "lease at now_ms is expired (strict greater-than)"
        );
        // At now_ms=499, still alive.
        let subs: Vec<_> = r.subscribers_for(100, 499).collect();
        assert_eq!(subs, vec![s(1)]);
    }
}
