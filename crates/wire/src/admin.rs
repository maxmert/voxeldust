//! The read-only orchestrator admin contract: the SHAPES a 2am `curl` gets back
//! (`docs/design/test_harness.md` observability; PLAN.md P0 row — "curl a stuck saga
//! instead of grepping logs"). The HTTP shell that serves these lives in `io-prod`;
//! this module is the pure, frozen contract both sides build against.
//!
//! Views are OPERATOR-FACING: ids are pre-rendered through their canonical `Display`
//! forms (hex-stable, grep-friendly) so a dump is readable without tooling, and the
//! u128 ids never hit a JSON number (which cannot carry them).

use serde::{Deserialize, Serialize};
use vd_core::{EpochId, Fence, NodeId, UniverseTick};

use crate::seams::directory::{AuthorityRef, DirectoryKey, OwnerRecord};

/// One directory row, rendered for operators.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryEntryView {
    /// Canonical `Display` of the [`DirectoryKey`] (e.g. `sess-…`, `planet-…`).
    pub key: String,
    /// Canonical `Display` of the owning node (e.g. `shard:node-3`).
    pub authority: String,
    pub fence: Fence,
    pub lease_expires: UniverseTick,
    /// Canonical `Display` of the in-flight transfer, if the key is locked.
    pub in_transfer: Option<String>,
}

/// Build the operator view of one directory row.
#[must_use]
pub fn directory_entry_view(key: &DirectoryKey, record: &OwnerRecord) -> DirectoryEntryView {
    DirectoryEntryView {
        key: render_directory_key(key),
        authority: render_authority(&record.authority),
        fence: record.fence,
        lease_expires: record.lease_expires,
        in_transfer: record.in_transfer.map(|t| t.to_string()),
    }
}

fn render_directory_key(key: &DirectoryKey) -> String {
    match key {
        DirectoryKey::Session(s) => s.to_string(),
        DirectoryKey::Entity(e) => e.to_string(),
        DirectoryKey::Realm(r) => r.to_string(),
        DirectoryKey::Ship(e) => format!("ship-of-{e}"),
    }
}

fn render_authority(authority: &AuthorityRef) -> String {
    match authority {
        AuthorityRef::Shard(n) => format!("shard:{n}"),
        AuthorityRef::Gateway(n) => format!("gateway:{n}"),
    }
}

/// One saga, rendered for operators. The state string is the orchestrator's own
/// `Debug`/`Display` of its saga FSM state (wire cannot depend on `sim`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SagaView {
    /// Canonical `Display` of the `TransferId`.
    pub transfer: String,
    pub state: String,
    /// When the saga entered its current state (staleness is `now - since`).
    pub since: UniverseTick,
}

/// Lease liveness for one node, rendered for operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseHealthView {
    pub node: NodeId,
    pub lease_expires: UniverseTick,
    /// Negative means LAPSED (recovery pending orchestrator confirmation).
    pub ticks_remaining: i64,
}

/// The whole read-only snapshot one `GET /admin/snapshot` returns. Empty-but-shaped
/// from day one (the P0 demo); the orchestrator fills it as subsystems land.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdminSnapshot {
    pub universe_tick: u64,
    pub epoch: u64,
    pub directory: Vec<DirectoryEntryView>,
    pub sagas: Vec<SagaView>,
    pub leases: Vec<LeaseHealthView>,
}

impl AdminSnapshot {
    /// The shaped-but-empty snapshot (clock fields only) — what a fresh orchestrator
    /// serves before any state exists.
    #[must_use]
    pub fn shaped_empty(universe_tick: UniverseTick, epoch: EpochId) -> AdminSnapshot {
        AdminSnapshot {
            universe_tick: universe_tick.0,
            epoch: epoch.0,
            directory: Vec::new(),
            sagas: Vec::new(),
            leases: Vec::new(),
        }
    }

    /// True once the cluster has BOOTSTRAPPED: some shard has granted a realm into
    /// the directory, which only happens after the orchestrator↔shard mTLS/QUIC
    /// handshake. The dev-cluster launcher polls this (over the parsed admin
    /// snapshot) as its readiness signal — replacing a brittle substring match on
    /// raw JSON. (A shard's realm authority renders as `shard:…`; the gateway
    /// registers no directory row in P1, so the launcher confirms gateway liveness
    /// separately, by process.)
    #[must_use]
    pub fn cluster_bootstrapped(&self) -> bool {
        self.directory
            .iter()
            .any(|entry| entry.authority.starts_with("shard:"))
    }
}

/// The reviewed Prometheus metric-name registry (PLAN.md observability list). ONE
/// place; never inline string literals at instrumentation sites. Names follow the
/// Prometheus conventions: `vd_` namespace, `_total` for counters, unit suffixes.
pub mod metric_names {
    /// Gauge: sagas past their per-phase deadline — any nonzero value is an alert,
    /// never a silent wedge.
    pub const SAGA_STUCK: &str = "vd_saga_stuck";
    /// Counter: FENCE-THEN-FORCE bounded resolutions taken.
    pub const FORCED_COMMIT_TOTAL: &str = "vd_forced_commit_total";
    /// Counter: entities re-transferring within their dwell cooldown.
    pub const TRANSFER_PINGPONG_TOTAL: &str = "vd_transfer_pingpong_total";
    /// Gauge: per-node analytic-clock skew against the orchestrator.
    pub const CLOCK_SKEW_MS: &str = "vd_clock_skew_ms";
    /// Histogram: durable-store fsync latency (p99 is derived at query time).
    pub const FSYNC_DURATION_MS: &str = "vd_fsync_duration_ms";
    /// Gauge: ticks since the freshest GhostDelta per ghost band.
    pub const GHOST_STALENESS_TICKS: &str = "vd_ghost_staleness_ticks";
    /// Gauge: sagas currently in flight.
    pub const TRANSFERS_IN_FLIGHT: &str = "vd_transfers_in_flight";
    /// Counter: snapshot datagrams dropped because they exceeded the path MTU — a
    /// partitioner-budget misconfiguration ALERT (must stay 0; audit GW-1).
    pub const DATAGRAMS_DROPPED_TOO_LARGE: &str = "vd_datagrams_dropped_too_large";

    /// Every registered name (the conformance test iterates this; adding a metric
    /// without listing it here is a review-rejectable defect).
    pub const ALL: &[&str] = &[
        SAGA_STUCK,
        FORCED_COMMIT_TOTAL,
        TRANSFER_PINGPONG_TOTAL,
        CLOCK_SKEW_MS,
        FSYNC_DURATION_MS,
        GHOST_STALENESS_TICKS,
        TRANSFERS_IN_FLIGHT,
        DATAGRAMS_DROPPED_TOO_LARGE,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::pose::RealmId;
    use vd_core::{EntityId, SessionId, TransferId};

    fn record(in_transfer: Option<TransferId>) -> OwnerRecord {
        OwnerRecord {
            authority: AuthorityRef::Shard(NodeId(3)),
            fence: Fence(7),
            lease_expires: UniverseTick(900),
            in_transfer,
        }
    }

    #[test]
    fn directory_views_render_every_key_kind() {
        let session = directory_entry_view(&DirectoryKey::Session(SessionId(0xAB)), &record(None));
        assert_eq!(
            session.key,
            "sess-000000000000000000000000000000ab".to_owned()
        );
        assert_eq!(session.authority, "shard:node-3");
        assert_eq!(session.in_transfer, None);

        let entity = directory_entry_view(&DirectoryKey::Entity(EntityId(1)), &record(None));
        assert_eq!(entity.key, EntityId(1).to_string());

        let realm =
            directory_entry_view(&DirectoryKey::Realm(RealmId::Planet(0xDEAD)), &record(None));
        assert_eq!(realm.key, "planet-000000000000dead".to_owned());

        let ship = directory_entry_view(
            &DirectoryKey::Ship(EntityId(2)),
            &record(Some(TransferId(5))),
        );
        assert_eq!(ship.key, format!("ship-of-{}", EntityId(2)));
        assert_eq!(ship.in_transfer, Some(TransferId(5).to_string()));
    }

    #[test]
    fn gateway_authority_renders_distinctly() {
        let view = directory_entry_view(
            &DirectoryKey::Session(SessionId(1)),
            &OwnerRecord {
                authority: AuthorityRef::Gateway(NodeId(9)),
                ..record(None)
            },
        );
        assert_eq!(view.authority, "gateway:node-9");
    }

    #[test]
    fn shaped_empty_snapshot_roundtrips() {
        let snap = AdminSnapshot::shaped_empty(UniverseTick(42), EpochId(2));
        assert_eq!(snap.universe_tick, 42);
        assert_eq!(snap.epoch, 2);
        assert_eq!(snap.directory, Vec::new());
        assert_eq!(snap.sagas, Vec::new());
        assert_eq!(snap.leases, Vec::new());
        let bytes = postcard::to_allocvec(&snap).expect("encode");
        let back: AdminSnapshot = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, snap);
    }

    #[test]
    fn populated_snapshot_roundtrips() {
        let snap = AdminSnapshot {
            universe_tick: 100,
            epoch: 1,
            directory: vec![directory_entry_view(
                &DirectoryKey::Entity(EntityId(7)),
                &record(Some(TransferId(9))),
            )],
            sagas: vec![SagaView {
                transfer: TransferId(9).to_string(),
                state: "Cutting".to_owned(),
                since: UniverseTick(95),
            }],
            leases: vec![LeaseHealthView {
                node: NodeId(3),
                lease_expires: UniverseTick(90),
                ticks_remaining: -10,
            }],
        };
        let bytes = postcard::to_allocvec(&snap).expect("encode");
        let back: AdminSnapshot = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, snap);
    }

    #[test]
    fn cluster_bootstrapped_iff_a_shard_holds_a_realm() {
        // Empty directory: not bootstrapped.
        let empty = AdminSnapshot::shaped_empty(UniverseTick(1), EpochId(1));
        assert!(!empty.cluster_bootstrapped());

        // A realm granted to a SHARD: bootstrapped.
        let with_shard = AdminSnapshot {
            directory: vec![directory_entry_view(
                &DirectoryKey::Realm(RealmId::System(7)),
                &record(None),
            )],
            ..empty.clone()
        };
        assert!(with_shard.cluster_bootstrapped());

        // A row owned only by a GATEWAY does NOT signal cluster bootstrap.
        let gateway_only = AdminSnapshot {
            directory: vec![directory_entry_view(
                &DirectoryKey::Session(SessionId(1)),
                &OwnerRecord {
                    authority: AuthorityRef::Gateway(NodeId(2)),
                    ..record(None)
                },
            )],
            ..empty
        };
        assert!(!gateway_only.cluster_bootstrapped());
    }

    #[test]
    fn metric_names_are_unique_and_prometheus_valid() {
        let mut seen = std::collections::BTreeSet::new();
        for name in metric_names::ALL {
            assert!(seen.insert(*name), "duplicate metric name: {name}");
            assert!(name.starts_with("vd_"), "missing namespace: {name}");
            let valid = name
                .bytes()
                .all(|b| b.is_ascii_lowercase() | b.is_ascii_digit() | (b == b'_'));
            assert!(valid, "invalid prometheus identifier: {name}");
        }
        assert_eq!(seen.len(), metric_names::ALL.len());
    }
}
