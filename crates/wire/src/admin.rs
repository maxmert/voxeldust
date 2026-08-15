//! The read-only orchestrator admin contract: the SHAPES a 2am `curl` gets back
//! (`docs/design/test_harness.md` observability; PLAN.md P0 row — "curl a stuck saga
//! instead of grepping logs"). The HTTP shell that serves these lives in `io-prod`;
//! this module is the pure, frozen contract both sides build against.
//!
//! Views are OPERATOR-FACING: ids are pre-rendered through their canonical `Display`
//! forms (hex-stable, grep-friendly) so a dump is readable without tooling, and the
//! u128 ids never hit a JSON number (which cannot carry them).
//!
//! SECURITY CONTRACT: read-only by construction AND loopback/internal-only — the
//! snapshot serves internal topology (node ids, fences, leases), so the serving
//! endpoint MUST gain authentication (bearer/mTLS on the route) before any routable
//! bind. The dev cluster binds loopback; real auth is a deploy-readiness item.
//! (Audit CAF-2.)

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

/// The realm-lifecycle reconciler's observability counters + gauges (RLM 5f-4), rendered for operators —
/// the "curl the demand loop at 2am" view: how many realms were spun up / failed / reaped, and the MEASURED
/// pod-boot latency the launch-TTL can be tuned from. All zero on a quiescent (or inert) orchestrator.
/// JSON-ONLY: this rides the `/admin/snapshot` HTTP body, NEVER the frozen postcard `InterShardFlow` wire,
/// so appending it here does not touch the wire contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RlmView {
    /// `SpinUp` intents executed (a demanded realm's shard was launched).
    pub spins_requested: u64,
    /// `spawn_realm` refusals (drives the backoff + the stuck alarm).
    pub spins_failed: u64,
    /// `Kill`s executed (a reclaimed realm).
    pub teardowns_reaped: u64,
    /// `ForceReap`s executed (a zombie head cleaned).
    pub force_reaps: u64,
    /// Saga frames that did not decode to a `RealmDemand` (honesty; 0 in a healthy run).
    pub undecodable_demands: u64,
    /// Demands whose sender the directory head shows holding neither the demanded realm's parent nor
    /// the realm itself (or whose heads are unresolvable). MEASURE-ONLY — the demand is honored
    /// unchanged (owner ruling 2026-08-15; enforcement waits for cloud mTLS).
    pub demand_sender_mismatch: u64,
    /// Last sweep's desired-realm count (gauge).
    pub desired_gauge: u64,
    /// Last sweep's running-realm count (gauge).
    pub running_gauge: u64,
    /// The MAX observed launch→head-up latency in universe ticks (monotone). The measured pod boot the
    /// launch-TTL is tuned from (`VD_BOOT_TICKS_P99`, 5f-4j); 0 until a demand-spawned head first appears.
    pub boot_ticks_observed_max: u64,
    /// MONOTONE count of realm-sweeps where a pending hand-off was the SOLE reason a realm stayed alive
    /// — the arrival shield, counted by name. ZERO on every healthy run; a climbing value is the visible
    /// signature of a hand-off that is wedged rather than merely slow.
    pub arrival_shield_vetoes: u64,
    /// How many realms the arrival shield is holding up right now (last sweep's gauge).
    pub arrival_shield_gauge: u64,
}

/// The gateway's session/routing honesty counters + live gauges (RLM RG-4; the gateway half of the ledgered
/// D-10 per-node observability). The gateway analogue of [`RlmView`]: the `GatewayStats` counters (defined in
/// `vd-connection-plane`, mirrored here so the admin contract never depends on the sim crate) plus two live
/// gauges. All zero on a fresh gateway; a nonzero `presence_announces` proves demand-spawned shards are
/// reactively greeting this gateway. JSON-ONLY: this rides the `/admin/snapshot` HTTP body, NEVER the frozen
/// postcard `InterShardFlow` wire, so appending it here does not touch the wire contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayView {
    pub logins_rejected: u64,
    pub version_rejected: u64,
    pub sessions_refused_capacity: u64,
    pub session_mints_refused: u64,
    pub resumes_refused: u64,
    pub inputs_deduped: u64,
    pub inputs_unroutable: u64,
    pub inputs_malformed: u64,
    pub stale_frames_dropped: u64,
    /// A shard's `EntityRemoved` refused for one subscriber on a stale fence (the remove message,
    /// proto_minor 14) — counted apart from `stale_frames_dropped` because a wrongly-dropped
    /// removal strands a frozen figure, not a tick of motion.
    pub stale_removals_dropped: u64,
    /// A frame the gateway could not decode/dispatch — the honesty floor; MUST stay 0 for a healthy demand
    /// login (a miscounted greeting would show up here instead of `presence_announces`).
    pub undecodable: u64,
    /// Frames DISCARDED because the router knows their sender as neither a shard nor a client. A fact
    /// about the SENDER, split out of `undecodable` (which is about the BYTES) — the two shared a bucket,
    /// and that is how a running shard's whole output was thrown away in silence while a player it owned
    /// went on being drawn by the realm they had already left. MUST be 0: a shard that is up and speaking
    /// is either known or it is being ignored, and there is no third healthy state.
    pub refused_unknown_sender: u64,
    /// Rosters applied from the ownership record — the router being TOLD whose frames it may read.
    pub shard_rosters_applied: u64,
    /// Rosters refused as older than the one held (a redelivery on a re-driven lane is expected).
    pub shard_roster_stale: u64,
    pub frame_sub_desync: u64,
    pub transfer_unroutable: u64,
    pub transfer_control_parked: u64,
    pub commit_without_cut: u64,
    pub inputs_buffered_for_dest: u64,
    pub dest_inputs_dropped: u64,
    pub sessions_self_fenced_lapsed: u64,
    /// A demand login whose home realm did not become routable inside its bootstrap window — the ops signal
    /// that the spawn/greet path is broken. 0 on a healthy demand login.
    pub home_bootstrap_timeouts: u64,
    pub home_wait_desync: u64,
    pub logins_held_pre_sync: u64,
    pub sessions_self_fenced_revoked: u64,
    /// Reactive-greeting `ShardPresence` frames received from demand-spawned shards (RG-3). Nonzero = a real
    /// forked shard greeted this gateway; the direct process-tier proof the reactive path fired.
    pub presence_announces: u64,
    /// Sub closes REFUSED by the authority-sub invariant: the shard named for closing is the session's
    /// CURRENT authority, so closing it would strand the client on its own owner (the total-freeze
    /// class). Its reachable producer is a SAME-NODE re-home (`source == dest`) — expected whenever an
    /// occupant re-enters the realm it just left — so nonzero is a health signal, not an error.
    pub sub_close_refused_authority: u64,
    /// Gauge: sessions currently open on this gateway.
    pub sessions_open: u64,
    /// Gauge: demand-spawned home shards on the runtime routable roster — nonzero iff the dynamic-home
    /// resolve fired (the single call site of `claim_dynamic_shard`), i.e. a login routed to a spawned node.
    pub dynamic_shards: u64,
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
    /// RLM 5f-4 — the demand-reconciler view. A JSON-only field on the `/admin/snapshot` body (never the
    /// frozen postcard wire); a JSON reader ignores fields it does not know, so appending it is safe for an
    /// older operator tool reading a newer snapshot.
    pub rlm: RlmView,
    /// RLM RG-4 — the GATEWAY's counters, present ONLY on a gateway's snapshot (`None` on the orchestrator's,
    /// which owns the directory but no gateway session state — so the field is self-identifying rather than a
    /// misleading all-zero struct). Always serialized (`null` when `None`), so no `#[serde(default)]` is needed
    /// on this same-build internal admin surface — the [`RlmView`] precedent (which also avoids `serde_json`).
    /// JSON-only, never the frozen postcard wire.
    pub gateway: Option<GatewayView>,
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
            rlm: RlmView::default(),
            gateway: None,
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

    /// True once EVERY realm in `realms` has been granted to a SHARD in the directory — the
    /// STRICT both-realms readiness gate a MULTI-shard cluster's `await_ready` needs (Track R / 1d.2,
    /// C1). [`cluster_bootstrapped`](Self::cluster_bootstrapped) returns `true` the instant ANY one
    /// shard grants ANY realm; a dual cluster that drove a crossing on that signal would proceed
    /// BEFORE the DEST shard granted its realm, so `handle_crossing_request` would resolve no dest
    /// head and only COUNT `crossing_unresolved` — the crossing would never fire. This predicate
    /// requires a `shard:`-authority row keyed on EACH realm's canonical `Display`, so `[System(7),
    /// System(8)]` is satisfied only once BOTH the source and dest shards have granted. An EMPTY
    /// `realms` slice is vacuously `true` (no realm is owed) — the single-shard launcher keeps
    /// calling [`cluster_bootstrapped`](Self::cluster_bootstrapped) and stays byte-identical.
    ///
    /// A realm's directory `key` is rendered through its canonical [`RealmId`](vd_core::pose::RealmId)
    /// `Display` (see `render_directory_key`), so the caller passes the realm ids and this matches
    /// against the same rendered form — never a hand-built string.
    #[must_use]
    pub fn realms_present(&self, realms: &[vd_core::pose::RealmId]) -> bool {
        realms.iter().all(|realm| self.realm_granted(*realm))
    }

    /// True iff a `shard:`-authority directory row is keyed on `realm`'s canonical `Display`. A
    /// MONOMORPHIC helper so the two branch arms (matching row / no matching row) are covered off
    /// the generic `all` closure in [`realms_present`](Self::realms_present) (HR5 generic-shim rule).
    fn realm_granted(&self, realm: vd_core::pose::RealmId) -> bool {
        let key = realm.to_string();
        self.directory
            .iter()
            .any(|entry| entry.key == key && entry.authority.starts_with("shard:"))
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
    /// partitioner-budget misconfiguration ALERT (must stay 0; audit GW-1). LEGACY
    /// EXCEPTION: this name predates the `_total` counter convention and is already a
    /// frozen scraped name, so it deliberately keeps its un-suffixed form (renaming a
    /// live scraped metric breaks dashboards) — the R-4d M4 mesh counters below DO
    /// carry `_total`.
    pub const DATAGRAMS_DROPPED_TOO_LARGE: &str = "vd_datagrams_dropped_too_large";

    // --- R-4d M4: the io-prod mesh reliability counters (MeshStatsSnapshot). Surfaced so a
    // soak/deploy can alert on a dead ack path (reliable_acked stuck at 0), a contiguity gap
    // (gap_drop MUST be 0), or a saturated retry buffer / oversize frame (reliable_shed). ---

    /// Counter: snapshot datagrams dropped on a failed transport send (the mesh datagram
    /// hot path could not hand the frame to quinn) — distinct from the MTU-too-large drop.
    pub const DATAGRAMS_DROPPED_SEND_TOTAL: &str = "vd_datagrams_dropped_send_total";
    /// Counter: RELIABLE inbound messages dropped because the bounded inbox was saturated
    /// with reliable traffic — a genuine overload ALERT (must stay 0 on a healthy run).
    pub const INBOUND_DROPPED_RELIABLE_TOTAL: &str = "vd_inbound_dropped_reliable_total";
    /// Counter: UNRELIABLE inbound messages evicted to bound the inbox — by-design
    /// latest-wins back-pressure, not an alert.
    pub const INBOUND_DROPPED_UNRELIABLE_TOTAL: &str = "vd_inbound_dropped_unreliable_total";
    /// Counter: reliable frames dropped by the receiver as belonging to a STALE sender
    /// incarnation (a restarted sender's ledger reset — R-3').
    pub const STALE_INCARNATION_DROP_TOTAL: &str = "vd_stale_incarnation_drop_total";
    /// Counter: reliable frames dropped by the receiver as belonging to a STALE lane epoch
    /// (a pre-blip epoch after a redial — R-3').
    pub const STALE_EPOCH_DROP_TOTAL: &str = "vd_stale_epoch_drop_total";
    /// Counter: reliable frames the receiver DEDUPED (already delivered — a replay after a
    /// blip re-covered an already-acked tail; benign, R-3').
    pub const DEDUP_DROP_TOTAL: &str = "vd_dedup_drop_total";
    /// Counter: reliable frames dropped leaving a contiguity GAP (seq > hw+1) — a
    /// MUST-BE-0 alert (the wire epoch keeps it genuinely 0; R-3').
    pub const GAP_DROP_TOTAL: &str = "vd_gap_drop_total";
    /// Counter: reliable sends SHED at the sender — the retry buffer hit its byte cap
    /// (dead ack path) or the frame was un-framable (oversize). Nonzero = an ALERT (R-4b/R-4d).
    pub const RELIABLE_SHED_TOTAL: &str = "vd_reliable_shed_total";
    /// Counter: reliable frames RETIRED by an incoming cumulative ack (the retry buffer
    /// draining). Stuck at 0 while sends flow = a DEAD ACK PATH (R-3'/R-4a).
    pub const RELIABLE_ACKED_TOTAL: &str = "vd_reliable_acked_total";
    /// Counter: dial-in peers REJECTED because the learned-peer table hit its cap (RLM RG-4). A nonzero value
    /// is a cap-too-low / peer-churn ALERT — on a demand cluster it would mean a spawned shard's reactive
    /// greeting was refused and the shard is unreachable. Landed-but-blind on every node until now; the VALUE
    /// mapping lands in the io-prod mesh-metrics wiring (RG-4a3).
    pub const LEARNED_PEERS_REJECTED_TOTAL: &str = "vd_learned_peers_rejected_total";

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
        DATAGRAMS_DROPPED_SEND_TOTAL,
        INBOUND_DROPPED_RELIABLE_TOTAL,
        INBOUND_DROPPED_UNRELIABLE_TOTAL,
        STALE_INCARNATION_DROP_TOTAL,
        STALE_EPOCH_DROP_TOTAL,
        DEDUP_DROP_TOTAL,
        GAP_DROP_TOTAL,
        RELIABLE_SHED_TOTAL,
        RELIABLE_ACKED_TOTAL,
        LEARNED_PEERS_REJECTED_TOTAL,
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
        assert_eq!(
            snap.rlm,
            RlmView::default(),
            "a fresh snapshot has the zero RLM view"
        );
        assert_eq!(snap.gateway, None, "a fresh snapshot names no gateway view");
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
            rlm: RlmView {
                spins_requested: 4,
                spins_failed: 1,
                teardowns_reaped: 2,
                force_reaps: 1,
                undecodable_demands: 0,
                demand_sender_mismatch: 7,
                desired_gauge: 3,
                running_gauge: 2,
                boot_ticks_observed_max: 37,
                arrival_shield_vetoes: 5,
                arrival_shield_gauge: 6,
            },
            // Every field a DISTINCT value so the round-trip catches a transposition.
            gateway: Some(GatewayView {
                logins_rejected: 1,
                version_rejected: 2,
                sessions_refused_capacity: 3,
                session_mints_refused: 4,
                resumes_refused: 5,
                inputs_deduped: 6,
                inputs_unroutable: 7,
                inputs_malformed: 8,
                stale_frames_dropped: 9,
                stale_removals_dropped: 29,
                undecodable: 10,
                // 26, not 11: the sequence above is positional and renumbering it to slot this in would
                // have rewritten every value below — which is exactly the transposition this fixture
                // exists to catch, done by hand.
                refused_unknown_sender: 26,
                shard_rosters_applied: 27,
                shard_roster_stale: 28,
                frame_sub_desync: 11,
                transfer_unroutable: 12,
                transfer_control_parked: 13,
                commit_without_cut: 14,
                inputs_buffered_for_dest: 15,
                dest_inputs_dropped: 16,
                sessions_self_fenced_lapsed: 17,
                home_bootstrap_timeouts: 18,
                home_wait_desync: 19,
                logins_held_pre_sync: 20,
                sessions_self_fenced_revoked: 21,
                presence_announces: 22,
                sub_close_refused_authority: 23,
                sessions_open: 24,
                dynamic_shards: 25,
            }),
        };
        let bytes = postcard::to_allocvec(&snap).expect("encode");
        let back: AdminSnapshot = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(
            back, snap,
            "the RLM + gateway views round-trip with the rest of the snapshot"
        );
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

    /// One `shard:`-authority directory row keyed on `realm`'s canonical `Display`.
    fn realm_row(realm: RealmId) -> DirectoryEntryView {
        directory_entry_view(&DirectoryKey::Realm(realm), &record(None))
    }

    #[test]
    fn realms_present_is_the_strict_both_realms_gate() {
        // C1 (Track R / 1d.2): the dual-cluster readiness gate must wait for BOTH realms, never an "OR".
        let empty = AdminSnapshot::shaped_empty(UniverseTick(1), EpochId(1));

        // 0 realms owed: vacuously true (the single-shard path passes this way; no realm is required).
        assert!(empty.realms_present(&[]));

        // 1 realm owed, but the directory is empty ⇒ NOT present.
        assert!(!empty.realms_present(&[RealmId::System(7)]));

        // Only realm 7 granted: [7] present, [7,8] NOT present (the both-realms gate holds the launcher).
        let only_source = AdminSnapshot {
            directory: vec![realm_row(RealmId::System(7))],
            ..empty.clone()
        };
        assert!(only_source.realms_present(&[RealmId::System(7)]));
        assert!(
            !only_source.realms_present(&[RealmId::System(7), RealmId::System(8)]),
            "the crossing must NOT be driven until the DEST realm is granted too"
        );

        // BOTH realms granted to shards: the both-realms gate is satisfied.
        let both = AdminSnapshot {
            directory: vec![realm_row(RealmId::System(7)), realm_row(RealmId::System(8))],
            ..empty.clone()
        };
        assert!(both.realms_present(&[RealmId::System(7), RealmId::System(8)]));

        // A realm granted only to a GATEWAY (not a shard) does NOT count as present.
        let gateway_realm = AdminSnapshot {
            directory: vec![directory_entry_view(
                &DirectoryKey::Realm(RealmId::System(8)),
                &OwnerRecord {
                    authority: AuthorityRef::Gateway(NodeId(2)),
                    ..record(None)
                },
            )],
            ..empty
        };
        assert!(
            !gateway_realm.realms_present(&[RealmId::System(8)]),
            "a gateway-owned realm row is not a shard grant"
        );
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
