//! Centralized metrics for the signal subsystem.
//!
//! Single source of truth for every metric name + label this subsystem
//! emits. Putting them in one module keeps Prometheus dashboards stable
//! across refactors (renaming a metric in one place changes it
//! everywhere) and lets operators discover the full surface by reading
//! one file.
//!
//! # Naming convention
//!
//! Prometheus best practice: `<subsystem>_<noun>_<unit>` for cumulative
//! counters, `<subsystem>_<noun>` for gauges. All metrics in this module
//! are prefixed `signal_` to namespace cleanly under
//! `voxeldust_*` (other subsystems pick their own prefixes — `physics_`,
//! `network_`, etc.).
//!
//! # Cardinality discipline
//!
//! Labels are bounded enums — no per-channel-name labels (channel names
//! are unbounded user input; high cardinality kills Prometheus). Per-
//! reason / per-bucket labels are fine because the value space is
//! enumerated.
//!
//! # Why centralize via consts + helpers
//!
//! Inline `metrics::counter!("signal_xyz")` calls scatter the metric
//! schema across files. A compile-time constant + a helper function
//! gives:
//!   * Single rename point if a metric needs renaming.
//!   * Type-checked label values (the helper only accepts the enum,
//!     not arbitrary strings).
//!   * Trivial test fixtures via `metrics::with_local_recorder`.

use metrics::{counter, gauge};

// ---------------------------------------------------------------------------
// Counter names
// ---------------------------------------------------------------------------

/// Cross-shard ingress entries that passed `try_push_remote` end-to-end.
/// Labels: `scope` ∈ {"local", "shortrange", "longrange", "radio"}.
/// Increment site: `signal_ingest_remote` after a successful
/// `try_push_remote` call.
pub const SIGNAL_INGEST_ACCEPTED_TOTAL: &str = "signal_ingest_accepted_total";

/// Cross-shard ingress entries rejected by `try_push_remote`. Labels:
/// `reason` ∈ {"unknown_channel", "local_immutable", "scope_mismatch",
/// "auth_failed", "stale_or_future_timestamp", "replay"}.
pub const SIGNAL_INGEST_REJECTED_TOTAL: &str = "signal_ingest_rejected_total";

/// Successful `RemoteAccessGrant` creations (issuer-side).
pub const SIGNAL_GRANTS_CREATED_TOTAL: &str = "signal_grants_created_total";
/// Successful grant revocations (issuer-side).
pub const SIGNAL_GRANTS_REVOKED_TOTAL: &str = "signal_grants_revoked_total";
/// Held-grant registrations on recipient side (`AddHeldGrant`).
pub const SIGNAL_HELD_GRANTS_REGISTERED_TOTAL: &str = "signal_held_grants_registered_total";
/// Held-grant deletions (`ForgetHeldGrant`).
pub const SIGNAL_HELD_GRANTS_FORGOTTEN_TOTAL: &str = "signal_held_grants_forgotten_total";

/// Cross-shard publishes via held grant successfully dispatched.
pub const SIGNAL_REMOTE_PUBLISH_DISPATCHED_TOTAL: &str =
    "signal_remote_publish_dispatched_total";

/// Listener-block lease re-issuances.
pub const SIGNAL_LISTENER_LEASE_RENEWED_TOTAL: &str = "signal_listener_lease_renewed_total";

/// Subscriber-side rate-limit GC evictions (stale sessions swept from
/// `ClientRateLimits`).
pub const SIGNAL_RATE_LIMIT_SESSIONS_EVICTED_TOTAL: &str =
    "signal_rate_limit_sessions_evicted_total";

/// Client-driven message dropped by the rate limiter. Labels: `bucket`
/// ∈ {"signal_publish", "remote_publish", "grant_ops"}.
pub const SIGNAL_RATE_LIMITED_TOTAL: &str = "signal_rate_limited_total";

/// SignalSubscribe requests that successfully registered (or renewed)
/// a remote subscription.
pub const SIGNAL_SUBSCRIBES_REGISTERED_TOTAL: &str = "signal_subscribes_registered_total";

/// SignalSubscribe requests rejected at validation (HMAC/grant/etc.).
/// Labels: `reason` ∈ {"unknown_channel", "stale_timestamp",
/// "grant_check_failed", "hmac_failed"}.
pub const SIGNAL_SUBSCRIBES_REJECTED_TOTAL: &str = "signal_subscribes_rejected_total";

/// Antenna-block publishes dispatched (per-tick block-driven forwards).
pub const SIGNAL_ANTENNA_PUBLISHES_TOTAL: &str = "signal_antenna_publishes_total";

// -- Phase 4.3 wire-dict counters ---------------------------------------

/// Outbound `FLAG_REGISTER` events — first-sighting of a channel name
/// for a given peer, or a re-register after LRU eviction. Pair this
/// with `signal_dict_hits_total` to compute hit-rate
/// `(hits / (hits + registers))` — operators chart this to confirm
/// the dict is actually saving bandwidth at scale.
pub const SIGNAL_DICT_REGISTERS_TOTAL: &str = "signal_dict_registers_total";

/// Outbound dict cache hits — the channel name was already known to
/// the peer, so the wire entry shipped only the wire_id.
pub const SIGNAL_DICT_HITS_TOTAL: &str = "signal_dict_hits_total";

/// Inbound resync events — the receiver dropped a batch because of
/// `dict_seq` drift or unresolved wire_id. Drives the alert on
/// "we're losing data because dicts are out of sync."
pub const SIGNAL_DICT_RESYNCS_TOTAL: &str = "signal_dict_resyncs_total";

// ---------------------------------------------------------------------------
// Gauge names (sampled values, not monotonic)
// ---------------------------------------------------------------------------

/// Live channel count in `SignalChannelTable`.
pub const SIGNAL_CHANNELS: &str = "signal_channels";
/// Live grant count in `GrantsRegistry` (excludes mirror grants).
pub const SIGNAL_GRANTS: &str = "signal_grants";
/// Held-grant count across all sessions on this shard.
pub const SIGNAL_HELD_GRANTS: &str = "signal_held_grants";
/// Sessions currently tracked by `ClientRateLimits`.
pub const SIGNAL_RATE_LIMITED_SESSIONS_TRACKED: &str =
    "signal_rate_limited_sessions_tracked";
/// Active remote subscribers across all channels.
pub const SIGNAL_REMOTE_SUBSCRIBERS: &str = "signal_remote_subscribers";

// -- Phase 4.3 wire-dict gauges -----------------------------------------

/// Total outbound dict bindings across every peer.
pub const SIGNAL_DICT_OUTBOUND_BINDINGS: &str = "signal_dict_outbound_bindings";
/// Total inbound dict bindings across every peer.
pub const SIGNAL_DICT_INBOUND_BINDINGS: &str = "signal_dict_inbound_bindings";
/// Peers we currently have outbound dict state for.
pub const SIGNAL_DICT_OUTBOUND_PEERS: &str = "signal_dict_outbound_peers";
/// Peers we currently have inbound dict state for.
pub const SIGNAL_DICT_INBOUND_PEERS: &str = "signal_dict_inbound_peers";

// ---------------------------------------------------------------------------
// Label-value enums (cardinality-bounded)
// ---------------------------------------------------------------------------

/// Reasons `try_push_remote` rejected an inbound entry. The string form
/// stays stable so dashboard queries don't break across refactors.
#[derive(Clone, Copy, Debug)]
pub enum IngressRejectReason {
    UnknownChannel,
    LocalChannelImmutable,
    ScopeMismatch,
    AuthFailed,
    StaleOrFutureTimestamp,
    Replay,
}

impl IngressRejectReason {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::UnknownChannel => "unknown_channel",
            Self::LocalChannelImmutable => "local_immutable",
            Self::ScopeMismatch => "scope_mismatch",
            Self::AuthFailed => "auth_failed",
            Self::StaleOrFutureTimestamp => "stale_or_future_timestamp",
            Self::Replay => "replay",
        }
    }
}

/// Conversion from the channel-layer error to the metrics label so the
/// `signal_ingest_remote` system can pass through the result without
/// knowing the label string.
impl From<crate::signal::channel::RemoteIngressDenied> for IngressRejectReason {
    fn from(d: crate::signal::channel::RemoteIngressDenied) -> Self {
        use crate::signal::channel::RemoteIngressDenied as R;
        match d {
            R::UnknownChannel => Self::UnknownChannel,
            R::LocalChannelImmutable => Self::LocalChannelImmutable,
            R::ScopeMismatch => Self::ScopeMismatch,
            R::AuthFailed => Self::AuthFailed,
            R::StaleOrFutureTimestamp => Self::StaleOrFutureTimestamp,
            R::Replay => Self::Replay,
        }
    }
}

/// Scope code from a `SignalBroadcastEntry` mapped to a stable label.
pub fn scope_label(scope_code: u8) -> &'static str {
    match scope_code {
        0 => "local",
        1 => "shortrange",
        2 => "longrange",
        3 => "radio",
        _ => "unknown",
    }
}

/// Rate-limit bucket name for the `signal_rate_limited_total{bucket}` label.
#[derive(Clone, Copy, Debug)]
pub enum RateLimitBucket {
    SignalPublish,
    RemotePublish,
    GrantOps,
}

impl RateLimitBucket {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::SignalPublish => "signal_publish",
            Self::RemotePublish => "remote_publish",
            Self::GrantOps => "grant_ops",
        }
    }
}

// ---------------------------------------------------------------------------
// Emit helpers — type-checked wrappers around the macro calls
// ---------------------------------------------------------------------------

#[inline]
pub fn record_ingest_accepted(scope_code: u8) {
    counter!(SIGNAL_INGEST_ACCEPTED_TOTAL, "scope" => scope_label(scope_code)).increment(1);
}

#[inline]
pub fn record_ingest_rejected(reason: IngressRejectReason) {
    counter!(SIGNAL_INGEST_REJECTED_TOTAL, "reason" => reason.as_label()).increment(1);
}

#[inline]
pub fn record_grant_created() {
    counter!(SIGNAL_GRANTS_CREATED_TOTAL).increment(1);
}

#[inline]
pub fn record_grant_revoked() {
    counter!(SIGNAL_GRANTS_REVOKED_TOTAL).increment(1);
}

#[inline]
pub fn record_held_grant_registered() {
    counter!(SIGNAL_HELD_GRANTS_REGISTERED_TOTAL).increment(1);
}

#[inline]
pub fn record_held_grant_forgotten() {
    counter!(SIGNAL_HELD_GRANTS_FORGOTTEN_TOTAL).increment(1);
}

#[inline]
pub fn record_remote_publish_dispatched() {
    counter!(SIGNAL_REMOTE_PUBLISH_DISPATCHED_TOTAL).increment(1);
}

#[inline]
pub fn record_listener_lease_renewed() {
    counter!(SIGNAL_LISTENER_LEASE_RENEWED_TOTAL).increment(1);
}

#[inline]
pub fn record_rate_limit_sessions_evicted(n: usize) {
    counter!(SIGNAL_RATE_LIMIT_SESSIONS_EVICTED_TOTAL).increment(n as u64);
}

#[inline]
pub fn record_rate_limited(bucket: RateLimitBucket) {
    counter!(SIGNAL_RATE_LIMITED_TOTAL, "bucket" => bucket.as_label()).increment(1);
}

#[inline]
pub fn record_subscribe_registered() {
    counter!(SIGNAL_SUBSCRIBES_REGISTERED_TOTAL).increment(1);
}

#[inline]
pub fn record_subscribe_rejected(reason_label: &'static str) {
    counter!(SIGNAL_SUBSCRIBES_REJECTED_TOTAL, "reason" => reason_label).increment(1);
}

#[inline]
pub fn record_antenna_publish() {
    counter!(SIGNAL_ANTENNA_PUBLISHES_TOTAL).increment(1);
}

#[inline]
pub fn record_dict_register() {
    counter!(SIGNAL_DICT_REGISTERS_TOTAL).increment(1);
}

#[inline]
pub fn record_dict_hit() {
    counter!(SIGNAL_DICT_HITS_TOTAL).increment(1);
}

/// Reasons for an inbound resync. Aggregated with a single counter
/// label so dashboards can split by cause.
#[derive(Clone, Copy, Debug)]
pub enum DictResyncReason {
    SeqDrift,
    UnresolvedWireId,
}

impl DictResyncReason {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::SeqDrift => "seq_drift",
            Self::UnresolvedWireId => "unresolved_wire_id",
        }
    }
}

#[inline]
pub fn record_dict_resync(reason: DictResyncReason) {
    counter!(SIGNAL_DICT_RESYNCS_TOTAL, "reason" => reason.as_label()).increment(1);
}

// Gauge updaters (called periodically from the GC / sweep systems)

#[inline]
pub fn set_channel_count(n: usize) {
    gauge!(SIGNAL_CHANNELS).set(n as f64);
}

#[inline]
pub fn set_grants_count(n: usize) {
    gauge!(SIGNAL_GRANTS).set(n as f64);
}

#[inline]
pub fn set_held_grants_count(n: usize) {
    gauge!(SIGNAL_HELD_GRANTS).set(n as f64);
}

#[inline]
pub fn set_rate_limited_sessions_tracked(n: usize) {
    gauge!(SIGNAL_RATE_LIMITED_SESSIONS_TRACKED).set(n as f64);
}

#[inline]
pub fn set_remote_subscribers(n: usize) {
    gauge!(SIGNAL_REMOTE_SUBSCRIBERS).set(n as f64);
}

#[inline]
pub fn set_dict_outbound_bindings(n: usize) {
    gauge!(SIGNAL_DICT_OUTBOUND_BINDINGS).set(n as f64);
}

#[inline]
pub fn set_dict_inbound_bindings(n: usize) {
    gauge!(SIGNAL_DICT_INBOUND_BINDINGS).set(n as f64);
}

#[inline]
pub fn set_dict_outbound_peers(n: usize) {
    gauge!(SIGNAL_DICT_OUTBOUND_PEERS).set(n as f64);
}

#[inline]
pub fn set_dict_inbound_peers(n: usize) {
    gauge!(SIGNAL_DICT_INBOUND_PEERS).set(n as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingress_reject_reason_label_round_trip() {
        // Each variant produces a stable, distinct label.
        let labels: Vec<_> = [
            IngressRejectReason::UnknownChannel,
            IngressRejectReason::LocalChannelImmutable,
            IngressRejectReason::ScopeMismatch,
            IngressRejectReason::AuthFailed,
            IngressRejectReason::StaleOrFutureTimestamp,
            IngressRejectReason::Replay,
        ]
        .iter()
        .map(|r| r.as_label())
        .collect();
        // No duplicates.
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "labels must be distinct");
        // No accidental "" labels.
        assert!(labels.iter().all(|l| !l.is_empty()));
    }

    #[test]
    fn scope_label_covers_all_wire_codes() {
        // Every documented wire scope code maps to a non-"unknown" label.
        for scope in 0..=3 {
            assert_ne!(scope_label(scope), "unknown");
        }
        // Out-of-range falls into the catch-all.
        assert_eq!(scope_label(99), "unknown");
    }

    #[test]
    fn rate_limit_bucket_labels_match_struct_fields() {
        // Smoke: the labels match the `ClientBuckets` field naming
        // convention so a Prometheus dashboard query can be derived
        // mechanically from the struct.
        assert_eq!(RateLimitBucket::SignalPublish.as_label(), "signal_publish");
        assert_eq!(RateLimitBucket::RemotePublish.as_label(), "remote_publish");
        assert_eq!(RateLimitBucket::GrantOps.as_label(), "grant_ops");
    }

    #[test]
    fn dict_resync_reason_labels_distinct_and_nonempty() {
        let labels = [
            DictResyncReason::SeqDrift.as_label(),
            DictResyncReason::UnresolvedWireId.as_label(),
        ];
        let mut sorted = labels.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len());
        assert!(labels.iter().all(|l| !l.is_empty()));
    }

    #[test]
    fn from_remote_ingress_denied_covers_every_variant() {
        // Compile-time check that every channel-layer rejection variant
        // has a metric mapping. Catches drift if a new RemoteIngressDenied
        // variant is added without a corresponding label.
        use crate::signal::channel::RemoteIngressDenied as R;
        for d in [
            R::UnknownChannel,
            R::LocalChannelImmutable,
            R::ScopeMismatch,
            R::AuthFailed,
            R::StaleOrFutureTimestamp,
            R::Replay,
        ] {
            // If a variant is missing from the From impl this panics;
            // here it just exercises the conversion.
            let _: IngressRejectReason = d.into();
        }
    }
}
