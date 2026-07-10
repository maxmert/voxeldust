//! Latency-percentile helper shared by the hard latency GATES (HR3 one-tooling): the
//! SPIKE-2a route/forward-decision p99 (in `vd-connection-plane`'s tests) and the SPIKE-3a
//! snapshot-under-bulk-burst p99 (in `vd-io-prod`'s tests). ONE implementation so the two
//! gates can never drift.
//!
//! Pure [`Duration`] math — reads NO clock, so the Tier-A `Instant::now`/`SystemTime::now`
//! bans (see `clippy.toml`) do not apply here: the CALLER samples time via its own source
//! (the injected `sim::io::Clock` in Tier-A, or `Instant` in an io-prod integration test)
//! and hands us the resulting `Duration`s. Consumed as a `[dev-dependencies]` edge only, so
//! it never inverts the `bins → node → sim → wire → core` build graph (harness sits above
//! node; a dev-scope edge is test-only and acyclic).

use std::time::Duration;

/// The p99-style tail of a latency sample set (sort + nearest-rank index). TOTAL — an empty
/// set is [`Duration::ZERO`] (no panic), so a caller that may legitimately measure zero
/// deliveries (e.g. a fully-shed latest-wins unreliable stream) is well-defined. Pair it with
/// a delivery-count/ratio floor to tell "fast" apart from "nothing arrived" (an empty set's
/// `ZERO` would otherwise pass a `< budget` gate deceptively). `pct` is a percentile in
/// `0..=100`; `100` clamps to the max (never out-of-bounds).
#[must_use]
pub fn percentile_unstable(mut samples: Vec<Duration>, pct: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    let rank = samples.len().saturating_mul(pct) / 100;
    samples[rank.min(samples.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::percentile_unstable;
    use std::time::Duration;

    #[test]
    fn percentile_unstable_total_over_empty_single_and_edges() {
        let d = Duration::from_nanos;
        // Empty → ZERO (the total-ness the gate callers rely on; no panic).
        assert_eq!(percentile_unstable(Vec::new(), 99), Duration::ZERO);
        // Single element → itself at any percentile.
        assert_eq!(percentile_unstable(vec![d(5)], 99), d(5));
        assert_eq!(percentile_unstable(vec![d(5)], 0), d(5));
        // Nearest-rank over a known set; p100 clamps to the max (no out-of-bounds).
        let s = vec![d(10), d(40), d(20), d(30), d(50)]; // sorts to 10,20,30,40,50
        assert_eq!(percentile_unstable(s.clone(), 99), d(50)); // rank 4
        assert_eq!(percentile_unstable(s.clone(), 100), d(50)); // rank 5 → clamp to 4
        assert_eq!(percentile_unstable(s, 50), d(30)); // rank 2
    }
}
