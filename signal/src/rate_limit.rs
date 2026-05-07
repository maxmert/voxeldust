//! Token-bucket rate limiting for client-driven signal operations.
//!
//! # Threat model
//!
//! A connected client can in principle send arbitrary `ClientMsg` traffic
//! over its TCP socket — `BlockConfigUpdate`, `SignalPublish`,
//! `RemoteSignalPublish`, `GrantCreate`/`Revoke`, `AddHeldGrant`/`ForgetHeldGrant`.
//! Without per-session caps, one bad actor (compromised client, hostile
//! tester, accidentally-misbehaving game logic) can saturate a shard's
//! tick budget by flooding any of these.
//!
//! Rate limiting addresses this with token buckets sized to the
//! legitimate-use ceiling for each operation class:
//!
//! | Bucket | Capacity (burst) | Refill (steady state) |
//! |---|---|---|
//! | `signal_publish` | 200 | 100/sec |
//! | `remote_publish` | 100 | 50/sec |
//! | `grant_ops`      | 50  | 10/sec |
//!
//! These ceilings are defaults; the [`ClientRateLimits`] resource
//! exposes them as tunable fields so deployments can adjust per-cluster
//! based on observed traffic patterns.
//!
//! # Why per-session, not per-IP
//!
//! Per-IP bucketing fails behind NAT (a friend group sharing one IP
//! gets one bucket). Per-session matches the ECS authentication boundary
//! — if a session is misbehaving, deny just that session.
//!
//! # Algorithm
//!
//! Standard leaky-bucket-as-counter with monotonic UNIX-millis clock.
//! `try_consume(now_ms)` refills based on elapsed time since `last_refill_ms`,
//! checks if a token is available, and atomically consumes one if so.
//! Returns `true` (proceed) or `false` (rate-limited).
//!
//! Refill uses `f32` accumulator so sub-token-per-tick refill rates work
//! (e.g., 0.5 tokens/tick = 10/sec at 20Hz). Saturating at `capacity` so
//! idle sessions don't accumulate unbounded budget.

use bevy_ecs::prelude::*;
use std::collections::HashMap;

use voxeldust_types::SessionToken;

/// One bucket. Holds `capacity` tokens at most; refills at `refill_per_sec`
/// while non-full. `try_consume` deducts one token if any remain.
#[derive(Clone, Debug)]
pub struct TokenBucket {
    pub capacity: f32,
    pub refill_per_sec: f32,
    pub tokens: f32,
    /// Last UNIX millis when we computed the refill. Updated on every
    /// `try_consume` call so refill is amortized across queries.
    pub last_refill_ms: u64,
}

impl TokenBucket {
    /// Create a fresh bucket starting at full capacity. Bursts of
    /// `capacity` size are allowed immediately on first use.
    pub fn full(capacity: f32, refill_per_sec: f32, now_ms: u64) -> Self {
        Self {
            capacity,
            refill_per_sec,
            tokens: capacity,
            last_refill_ms: now_ms,
        }
    }

    /// Refill the bucket based on elapsed time and try to consume one
    /// token. Returns `true` if a token was available (and consumed),
    /// `false` if the bucket was empty after refill.
    pub fn try_consume(&mut self, now_ms: u64) -> bool {
        // Refill: tokens += elapsed_seconds * refill_rate, clamped to capacity.
        let elapsed_ms = now_ms.saturating_sub(self.last_refill_ms);
        if elapsed_ms > 0 {
            let elapsed_s = elapsed_ms as f32 / 1000.0;
            self.tokens = (self.tokens + elapsed_s * self.refill_per_sec).min(self.capacity);
            self.last_refill_ms = now_ms;
        }
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Inspect remaining tokens without modifying state. Used by metrics
    /// + tests; never called on the hot path.
    pub fn peek(&self) -> f32 {
        self.tokens
    }
}

/// Per-operation buckets for one client session. Each operation class
/// has its own bucket so a flood on one path doesn't starve another
/// (e.g., a SignalPublish flood doesn't block GrantRevoke from going
/// through — important if GrantRevoke is the only way to stop the flood).
#[derive(Clone, Debug)]
pub struct ClientBuckets {
    /// `ClientMsg::SignalPublish` — HUD button presses, slider drags.
    /// High cap because publisher widgets fire continuously while held.
    pub signal_publish: TokenBucket,
    /// `ClientMsg::RemoteSignalPublish` — tablet → cross-shard publish
    /// via held grant. Lower cap (cross-shard work is more expensive).
    pub remote_publish: TokenBucket,
    /// All grant lifecycle ops: `GrantCreate`, `GrantRevoke`,
    /// `AddHeldGrant`, `ForgetHeldGrant`. Aggressive cap — these are
    /// human-initiated discrete actions, never automated streams.
    pub grant_ops: TokenBucket,
    /// Counter — number of times this session was rate-limited (across
    /// all buckets). Useful for telemetry + flagging suspicious sessions.
    pub denied_total: u64,
}

impl ClientBuckets {
    /// Default tuning. See module docs for rationale.
    pub fn default_at(now_ms: u64) -> Self {
        Self {
            signal_publish: TokenBucket::full(200.0, 100.0, now_ms),
            remote_publish: TokenBucket::full(100.0, 50.0, now_ms),
            grant_ops: TokenBucket::full(50.0, 10.0, now_ms),
            denied_total: 0,
        }
    }
}

/// Per-shard rate-limit registry. Indexed by `SessionToken` (= player_id
/// in the unified component model). Inserted lazily on first message
/// from a session; evicted by the disconnect-cleanup system to bound
/// the table.
#[derive(Resource, Default)]
pub struct ClientRateLimits {
    pub by_session: HashMap<SessionToken, ClientBuckets>,
}

impl ClientRateLimits {
    /// Look up (or lazily create) the buckets for a session, then run
    /// `f` against them. Used by the apply systems to combine refill +
    /// consume + counter-bump in one place.
    pub fn with_session<R>(
        &mut self,
        session: SessionToken,
        now_ms: u64,
        f: impl FnOnce(&mut ClientBuckets) -> R,
    ) -> R {
        let buckets = self
            .by_session
            .entry(session)
            .or_insert_with(|| ClientBuckets::default_at(now_ms));
        f(buckets)
    }

    /// Forget a session's buckets. Call from the disconnect-cleanup
    /// path so the table doesn't grow unbounded across long-running
    /// shard lifetimes.
    pub fn forget(&mut self, session: SessionToken) {
        self.by_session.remove(&session);
    }

    /// Sweep stale sessions whose buckets haven't been touched recently.
    /// Used by the periodic GC system in `shard-common::signal_pipeline`
    /// as a fallback for disconnects we didn't see (TCP RST, crash,
    /// etc.) — bounds memory across long-running shard lifetimes.
    ///
    /// "Stale" = `now_ms - max(last_refill_ms_across_buckets) > stale_threshold_ms`.
    /// Reasonable threshold: ~10 minutes, well past the longest
    /// legitimate idle gap a player might have between actions.
    ///
    /// Returns the number of sessions evicted.
    pub fn evict_stale(&mut self, now_ms: u64, stale_threshold_ms: u64) -> usize {
        let before = self.by_session.len();
        self.by_session.retain(|_, b| {
            let most_recent = b.signal_publish.last_refill_ms
                .max(b.remote_publish.last_refill_ms)
                .max(b.grant_ops.last_refill_ms);
            now_ms.saturating_sub(most_recent) < stale_threshold_ms
        });
        before - self.by_session.len()
    }

    /// Number of tracked sessions — for `/metrics` exposition.
    pub fn tracked_sessions(&self) -> usize {
        self.by_session.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_bucket_starts_full_and_consumes_one() {
        let mut b = TokenBucket::full(10.0, 5.0, /*now=*/ 0);
        assert_eq!(b.peek(), 10.0);
        for _ in 0..10 {
            assert!(b.try_consume(0));
        }
        // 11th attempt: empty.
        assert!(!b.try_consume(0));
    }

    #[test]
    fn refill_brings_tokens_back_at_rate() {
        let mut b = TokenBucket::full(10.0, 5.0, /*now=*/ 0);
        // Drain the bucket.
        for _ in 0..10 {
            assert!(b.try_consume(0));
        }
        assert!(!b.try_consume(0));
        // Wait 1 second @ 5/sec → 5 tokens.
        assert!(b.try_consume(1000));
        // 4 more available now.
        for _ in 0..4 {
            assert!(b.try_consume(1000));
        }
        // 6th attempt at the same time: empty.
        assert!(!b.try_consume(1000));
    }

    #[test]
    fn refill_clamps_to_capacity() {
        let mut b = TokenBucket::full(10.0, 5.0, /*now=*/ 0);
        // Idle for an hour.
        let later = 60 * 60 * 1000;
        assert!(b.try_consume(later));
        // Should still have at most capacity-1 tokens (not capacity*60*60).
        assert!(b.peek() <= 9.0 + 0.001);
        assert!(b.peek() >= 9.0 - 0.001);
    }

    #[test]
    fn fractional_refill_accumulates() {
        // 1 token per second — at 200ms intervals.
        let mut b = TokenBucket::full(2.0, 1.0, /*now=*/ 0);
        assert!(b.try_consume(0));
        assert!(b.try_consume(0));
        // Empty now.
        assert!(!b.try_consume(0));
        // 200ms later: 0.2 tokens. Still empty.
        assert!(!b.try_consume(200));
        // 1000ms total: should have 1.0 token now (started at 0 after
        // 200ms, accrued 0.8 more in the next 800ms).
        assert!(b.try_consume(1000));
    }

    #[test]
    fn session_isolation() {
        let mut limits = ClientRateLimits::default();
        let s1 = SessionToken(1);
        let s2 = SessionToken(2);
        // Drain s1.
        for _ in 0..200 {
            limits.with_session(s1, 0, |b| {
                assert!(b.signal_publish.try_consume(0));
            });
        }
        // s1 is now at zero — next consume fails.
        let s1_denied = limits.with_session(s1, 0, |b| !b.signal_publish.try_consume(0));
        assert!(s1_denied);
        // s2 starts fresh — its bucket is full.
        let s2_ok = limits.with_session(s2, 0, |b| b.signal_publish.try_consume(0));
        assert!(s2_ok, "different sessions must have independent buckets");
    }

    #[test]
    fn forget_removes_session_buckets() {
        let mut limits = ClientRateLimits::default();
        let s = SessionToken(42);
        limits.with_session(s, 0, |b| b.signal_publish.try_consume(0));
        assert_eq!(limits.tracked_sessions(), 1);
        limits.forget(s);
        assert_eq!(limits.tracked_sessions(), 0);
    }

    #[test]
    fn evict_stale_removes_idle_sessions_only() {
        let mut limits = ClientRateLimits::default();
        let active_session = SessionToken(1);
        let stale_session = SessionToken(2);
        // Active: last touched at now_ms = 60_000.
        limits.with_session(active_session, 60_000, |b| {
            b.signal_publish.try_consume(60_000);
        });
        // Stale: last touched at now_ms = 0.
        limits.with_session(stale_session, 0, |b| {
            b.signal_publish.try_consume(0);
        });
        assert_eq!(limits.tracked_sessions(), 2);

        // Sweep at now=60_000 with 30s threshold — stale session evicted.
        let evicted = limits.evict_stale(60_000, 30_000);
        assert_eq!(evicted, 1);
        assert_eq!(limits.tracked_sessions(), 1);
        assert!(limits.by_session.contains_key(&active_session));
        assert!(!limits.by_session.contains_key(&stale_session));
    }

    #[test]
    fn evict_stale_keeps_recently_touched_sessions() {
        // A session that consumed against ANY bucket recently isn't stale.
        let mut limits = ClientRateLimits::default();
        let session = SessionToken(7);
        // signal_publish touched at 0, grant_ops touched at 100_000.
        limits.with_session(session, 0, |b| {
            b.signal_publish.try_consume(0);
        });
        limits.with_session(session, 100_000, |b| {
            b.grant_ops.try_consume(100_000);
        });
        // At now=110_000 with 30s threshold: most-recent is 100_000,
        // diff is 10_000 < 30_000 → keep.
        let evicted = limits.evict_stale(110_000, 30_000);
        assert_eq!(evicted, 0);
        assert_eq!(limits.tracked_sessions(), 1);
    }

    #[test]
    fn denied_total_increments_on_exhaustion() {
        let mut buckets = ClientBuckets::default_at(0);
        // Drain.
        while buckets.signal_publish.try_consume(0) {}
        // Caller pattern: when try_consume returns false, bump counter.
        // Verifies the structural intent — the counter belongs to the
        // bucket struct, not the bucket itself.
        if !buckets.signal_publish.try_consume(0) {
            buckets.denied_total += 1;
        }
        assert_eq!(buckets.denied_total, 1);
    }
}
