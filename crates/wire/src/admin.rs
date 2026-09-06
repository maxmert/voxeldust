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
    /// `ExteriorMoved` statements applied (the ruler switch, slice 5): a realm on some chain moved house.
    pub exterior_moves_applied: u64,
    /// `ExteriorMoved` statements refused as older than the one held for that child.
    pub exterior_moves_stale: u64,
    /// Session chains spliced by an applied statement — the sessions aboard the moved realm.
    pub exterior_moves_sessions_spliced: u64,
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
    /// THE WINDOW LANE's admitted + INGESTED rows (mesh minor 16): a `WindowFrame`/`WindowBody`/
    /// `WindowMembership` row that PASSED attestation and fed the Slice-B composer (the Slice-A
    /// `window_rows_unconsumed` retired when the engine started consuming). Counted apart from
    /// `undecodable`. JSON-only appended field — a JSON reader ignores fields it does not know,
    /// so an older operator tool keeps reading a newer snapshot.
    pub window_rows_ingested: u64,
    /// THE WINDOW LANE, fail-closed (Slice A): a window row whose sender is not the head this
    /// gateway resolved for the stating realm — forged or deposed, dropped + counted. 0 healthy.
    pub window_sender_mismatch: u64,
    /// THE WINDOW LANE, fail-closed (Slice A): a `WindowBody` refused by the admission rule (a
    /// look not about its author; a marker not about a direct child). 0 healthy.
    pub window_misauthored_body: u64,
    /// THE WINDOW LANE, fail-closed (Slice A): a window row naming an id this gateway holds no
    /// window for — a straggler around a close (benign in small numbers) or a forged id.
    pub window_unknown_row: u64,
    /// THE WINDOW LANE's control egress (Slice A): `WindowOpen`s for newly derived windows.
    pub window_open_sent: u64,
    /// THE WINDOW LANE's control egress (Slice A): `WindowClose`s for windows that stopped being
    /// derivable (a session ended or crossed away).
    pub window_close_sent: u64,
    /// THE WINDOW LANE's keep-alive egress (Slice A): `WindowOpen` re-asserts on the derived
    /// cadence (the shard's TTL is 2 beats + 1 of the same derivation).
    pub window_keepalives_sent: u64,
    /// Slice B — a `WindowFrame` stamped behind its window's derived ring span: refused.
    pub window_level_refused: u64,
    /// Slice B — a `WindowBody` older than the held statement for its subject: refused.
    pub window_body_stale: u64,
    /// Slice B — a `Marker` before its author's first level (no attested roster to vouch): refused.
    pub window_body_preroster: u64,
    /// Slice B — shared folds computed (one per (origin, tick) per gateway tick — §2.14).
    pub window_folds: u64,
    /// Slice B — sessions served from an already-computed shared fold (§2.14 memoization).
    pub window_fold_hits: u64,
    /// Slice B — shared-vs-per-session fold bit-level divergences (MUST stay 0 — parity-gated).
    pub window_fold_divergence: u64,
    /// Slice B — folds whose fresh prefix covered a whole ≥2-level chain: the exact-cadence
    /// boot pin's direct proof (two shards stamped identical universe ticks).
    pub window_full_chain_folds: u64,
    /// Slice B — GAUGE: sessions holding a non-empty derived chain this tick.
    pub window_chains_held: u64,
    /// GAUGE (2026-09-02): sessions whose newest fold carries a sky anchor this tick — the chain
    /// reached the galaxy's frame, so the star cloud can be placed.
    pub window_sky_anchored: u64,
    /// GAUGE (2026-09-02): the widest stamp gap, in ticks, between the origin level and any hop
    /// level of a held chain (`u64::MAX` for an empty hop ring). Wider than the ring span ⇒ no
    /// common tick ⇒ the fold never covers that hop.
    pub window_chain_stamp_gap_max: u64,
    /// A static attach onto a realm the seeded forest does not name: a one-realm chain, counted.
    pub attach_lineage_unresolved: u64,
    /// Slice B — strata held at last composed poses (per stratum per tick — §2.6.4).
    pub window_compose_hold_ticks: u64,
    /// Slice B — held strata removed by the dead-hop exit (§2.6.4).
    pub window_hop_dead: u64,
    /// Slice B — would-be common-tick rewinds frozen (§2.6.3 monotone-T). 0 healthy.
    pub window_t_monotone_stalled: u64,
    /// Slice B — chain derivations truncated on a cycle, fail-closed. 0 healthy.
    pub window_chain_cycle: u64,
    /// Slice B — fold rows refused by `InstantMismatch` (the shear law). 0 healthy — parity-gated.
    pub window_instant_mismatch: u64,
    /// Slice B — fold rows refused by `RotationBeyondExactReach` (a rotated frame past the
    /// millimetre rotation reach — R2, the P10 trigger).
    pub window_rotated_refused: u64,
    /// Far rows shipped in the sky's frame (2026-09-04).
    pub window_far_rows: u64,
    /// Looks carried from the session's shelf through a window churn (2026-09-04).
    pub window_looks_carried: u64,
    /// Origin swaps deferred until the new chain covers the lineage (2026-09-04).
    pub window_origin_swap_deferred: u64,
    /// Origin swaps forced after one hold of deferral (2026-09-04).
    pub window_origin_swap_forced: u64,
    /// Full levels restated on the keep-alive beat (2026-09-04).
    pub scene_levels_restated: u64,
    /// Rows logged by the per-tick trace (2026-09-05).
    pub window_trace_rows: u64,
    /// Sessions closed because the client's connection died or a frame toward it was undeliverable
    /// (2026-09-05, the vanished-client wedge).
    pub sessions_closed_peer_lost: u64,
    /// Sessions closed because a new process now speaks for the client's node id (2026-09-05).
    pub sessions_closed_peer_reincarnated: u64,
    /// Sessions ended by a fresh `Hello` from the same node (2026-09-05).
    pub sessions_replaced_by_relogin: u64,
    /// Interest bodies naming a session that does not subscribe to the sender (2026-09-05).
    pub interest_recipient_unsubscribed: u64,
    /// Out-of-interest notices delivered to one session (2026-09-05).
    pub interest_removals_fanned: u64,
    /// Slice B — rows whose stated frame was not their level's own: alien, dropped.
    pub window_alien_rows: u64,
    /// Slice B — chain levels whose hop was absent/mismatched/rosterless: prefix capped there.
    pub window_hop_invalid: u64,
    /// Slice B — Active sessions whose composed feed was withheld (no standing realm/window yet).
    pub window_unresolved_standing: u64,
    /// Slice B — §2.12 hop-vs-child-row agreement: bit-level disagreements. 0 — parity-gated.
    pub window_dedup_disagree: u64,
    /// Slice B — GAUGE (max): the measured hop-vs-child-row deviation bound, in integer CELLS
    /// (per-axis Chebyshev; exact at every magnitude — a nanometre gauge saturates u64 at
    /// star-gap magnitudes and goes vacuous; real-scale addendum §A4.8 row 14). Postcard is
    /// positional, so the rename moves zero bytes.
    pub window_dedup_max_dev_cells: u64,
    /// Slice B — lineage-ancestor `HeadRead{Realm}` polls sent on the keep-alive cadence.
    pub window_head_reads_sent: u64,
    /// Slice B — composed rows produced across all folds.
    pub window_composed_rows: u64,
    /// ★TOMBSTONED LANE (Slice C2, minor 19) — old opaque realm datagrams received from a shard
    /// and dropped. `0` healthy; non-zero means a shard speaks a lane nobody serves.
    pub old_realm_frames_dropped: u64,
    /// ★TOMBSTONED LANE (Slice C2, minor 19) — old-lane per-observer scene deltas dropped at the
    /// gateway. `0` healthy; non-zero means a shard speaks a lane nobody serves.
    pub old_scene_deltas_dropped: u64,
    /// Composed-scene egress — full levels shipped (one per epoch bump: login, crossing).
    pub scene_levels_sent: u64,
    /// Composed-scene egress — reliable deltas shipped on membership/body change.
    pub scene_deltas_sent: u64,
    /// Composed-scene egress — per-tick composed realm datagram chunks shipped.
    pub scene_datagrams_sent: u64,
    /// Q2 relay (mesh minor 17) — relayed statements admitted into a window's ingest.
    pub window_relays_ingested: u64,
    /// Q2 relay — sealed blobs that did not decode (dropped, fail-closed).
    pub window_relay_undecodable: u64,
    /// Q2 relay — relays for a child the author's own attested roster does not vouch.
    pub window_relay_unvouched: u64,
    /// Q2 relay — relays refused by the child's fence order / an older relayed level.
    pub window_relay_stale: u64,
    /// Q2 relay (§2.6.5 step 4) — relayed live-child interior rows composed into folds.
    pub window_relay_rows_composed: u64,
    /// C5 split (look_horizon.md slice 0) — the chain descent below a relaying stratum could
    /// not be rebuilt at the fold's tick.
    pub window_relay_descent_refused: u64,
    /// C5 split — a relayed child with no ring level at-or-before the fold's tick
    /// (G-RELAY-STAMP asserts 0 over a process flight — the D-WINDOW-6(2) discharge).
    pub window_relay_stamp_missing: u64,
    /// C5 split — a member child missing from the stratum author's own level at the fold's tick.
    pub window_relay_unrostered: u64,
    /// C5 GAUGE (max) — the at-or-before fallback's declared skew, ticks (bounded by one beat).
    pub window_relay_stamp_skew_ticks: u64,
    /// The widest skew a moving relayed row was advanced over (2026-09-05).
    pub window_relay_moving_skew_ticks: u64,
    /// C5 GAUGE (max) — deepest relayed subject, levels below its forwarding author (2 = the
    /// carrier's whole arity; more is an implementation climb bug).
    pub window_relay_depth_max: u64,
    /// Slice D (§2.8 departure mirror): self-looks dropped by the derived roster-loss window.
    pub window_looks_pruned: u64,
    /// Slice D: relayed interior levels dropped by the same window.
    pub window_relay_levels_pruned: u64,
    /// Look horizon slice 3 — interior forwards naming an unrostered grandchild (VIOLATION:
    /// asserted 0 on every lawful flight).
    pub window_relay_interior_unvouched: u64,
    /// Look horizon slice 3 — depth-3 statements lawfully filtered out of admitted interior
    /// batches (EXPECTED non-zero wherever an interior forwards; never asserted zero).
    pub window_relay_interior_filtered: u64,
    /// S11 sky lane — star-catalogue parts forwarded to clients. On a static world this reaches a
    /// small number and STOPS. A counter that keeps climbing means the catalogue is re-issued for
    /// nothing, which is the 7.0 MB-per-client cost S11 exists to remove.
    pub star_catalogue_parts_sent: u64,
    /// S11 sky lane — liveness beats forwarded to clients. ★ THE ONE COUNTER HERE WHOSE FLAT LINE IS
    /// THE DEFECT, not its growth: it is EXPECTED to climb for ever on a perfectly static world,
    /// because "the sky did not change" is exactly what it exists to keep saying.
    pub sky_alive_beats_sent: u64,
    /// S11 sky lane — requests for the sky sent to shards. Settles once every client is served. A
    /// counter that keeps climbing means a client is never confirming what it holds.
    pub sky_requests_sent: u64,
    /// S11 sky lane — clients stating the sky they hold. Each statement is what lets the gateway skip
    /// sending a catalogue to a client that already has it.
    pub sky_held_stated: u64,
    /// S11 sky lane — catalogue parts NOT sent because the client already held that sky. This is the
    /// saving, counted: every skipped part is bytes that did not cross for a galaxy that did not move.
    pub sky_parts_skipped: u64,
    /// Gauge: sessions currently open on this gateway.
    pub sessions_open: u64,
    /// Gauge: demand-spawned home shards on the runtime routable roster — nonzero iff the dynamic-home
    /// resolve fired (the single call site of `claim_dynamic_shard`), i.e. a login routed to a spawned node.
    pub dynamic_shards: u64,
    /// Gauge: windows this gateway currently holds open (THE WINDOW LANE, Slice A) — zero at zero
    /// sessions, structurally (the design's teardown test).
    pub windows_open: u64,
}

/// THE TAG THIS BUILD OFFERS ON EVERY INTER-NODE CONNECTION, as text (slice S3; the world half added at
/// S9).
///
/// The same string the transport puts on the wire, so an operator comparing two nodes' admin views is
/// comparing exactly what the handshake compared — not a second rendering of it that could differ.
///
/// ★ IT CARRIES TWO THINGS, AND THE SECOND WAS MISSING. The unit says how a position is COUNTED. The
/// world generation says how big the world IS. Two builds can agree on the first and disagree on the
/// second — and then they connect, decode every message cleanly, and place the same player somewhere
/// else. Nothing crashes and nothing is logged, which is why it had to move into the handshake rather
/// than into a check somebody remembers to run.
///
/// The world half is an ARGUMENT because the world's shape lives in the generator and this crate cannot
/// reach it — the same reason the saved-data label takes it as one.
#[must_use]
pub fn coordinate_unit_tag(world_generation: u64) -> String {
    format!(
        "vd-intershard/1+unit-{:016x}+world-{world_generation:016x}",
        vd_core::store_stamp::coordinate_generation()
    )
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
    /// ★ THE UNIT THIS NODE COUNTS POSITIONS IN, and the exact transport tag it offers (slice S3;
    /// owner-approved 2026-08-24, Q1 condition 3).
    ///
    /// THIS IS THE ONLY DIAGNOSIS CHANNEL FOR A FLEET REFUSAL, and it exists because the refusal
    /// itself cannot carry one. Two nodes that count positions in different units are refused by the
    /// transport handshake, which reports `no_application_protocol` — a connection error with no field,
    /// no value and no unit. An operator facing that has nothing to compare.
    ///
    /// So each node states its own tag here and in its start-up log, and comparing two nodes is then a
    /// matter of reading two lines. JSON-only, never the frozen wire.
    pub coordinate_unit_tag: String,
}

impl AdminSnapshot {
    /// The shaped-but-empty snapshot (clock fields only) — what a fresh orchestrator
    /// serves before any state exists.
    #[must_use]
    pub fn shaped_empty(
        universe_tick: UniverseTick,
        epoch: EpochId,
        world_generation: u64,
    ) -> AdminSnapshot {
        AdminSnapshot {
            universe_tick: universe_tick.0,
            epoch: epoch.0,
            directory: Vec::new(),
            sagas: Vec::new(),
            leases: Vec::new(),
            rlm: RlmView::default(),
            gateway: None,
            // Stated by every snapshot, including the empty one — a node that has done nothing yet is
            // exactly the node an operator is comparing against a node that refused it.
            coordinate_unit_tag: coordinate_unit_tag(world_generation),
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
mod unit_tag {
    //! THE ONLY DIAGNOSIS CHANNEL A FLEET REFUSAL HAS (slice S3).
    use super::*;

    #[test]
    fn the_admin_tag_and_the_transport_tag_are_the_same_string() {
        // TWO RENDERINGS OF ONE FACT is how an operator gets sent to the wrong cause. The transport
        // compares one string; the admin view shows another. If they could differ, an operator could
        // read two nodes whose views MATCH and still be refused — the worst possible state, because it
        // rules out the true cause.
        //
        // Kept honest by comparing the bytes the transport actually offers.
        // A world generation with no relationship to the unit, so the two halves cannot pass for each
        // other: a tag that printed the unit twice would fail this.
        const WORLD: u64 = 0xfeed_face_dead_beef;
        assert_eq!(
            coordinate_unit_tag(WORLD).into_bytes(),
            vd_io_prod_tag(WORLD),
            "the tag an operator reads must be the tag the handshake compares"
        );
        // AND IT MOVES WITH THE WORLD, not only with the unit — the half that was missing.
        assert_ne!(coordinate_unit_tag(WORLD), coordinate_unit_tag(WORLD + 1));
    }

    /// The transport's own tag, reproduced here from its own inputs rather than imported: `vd-wire`
    /// sits BELOW `vd-io-prod` and may not depend on it. Reproducing the format is exactly the drift
    /// this test exists to catch, so the test above is the thing that keeps the two in step.
    fn vd_io_prod_tag(world_generation: u64) -> Vec<u8> {
        format!(
            "vd-intershard/1+unit-{:016x}+world-{world_generation:016x}",
            vd_core::store_stamp::coordinate_generation()
        )
        .into_bytes()
    }

    #[test]
    fn a_different_unit_produces_a_different_tag() {
        // NON-VACUITY: a tag that ignored the unit would satisfy the test above and refuse nobody.
        let ours = coordinate_unit_tag(0);
        let theirs = format!(
            "vd-intershard/1+unit-{:016x}",
            vd_core::store_stamp::coordinate_generation() ^ 1
        );
        assert_ne!(ours, theirs);
        // And the unit is READABLE in it — the whole point, since the refusal carries no message.
        assert!(
            ours.contains(&format!(
                "{:016x}",
                vd_core::store_stamp::coordinate_generation()
            )),
            "the unit must be readable in the tag: {ours}"
        );
    }
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
        let snap = AdminSnapshot::shaped_empty(UniverseTick(42), EpochId(2), 0);
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
            coordinate_unit_tag: coordinate_unit_tag(0),
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
                exterior_moves_applied: 401,
                exterior_moves_stale: 402,
                exterior_moves_sessions_spliced: 403,
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
                // 30, not 24: appended after the fixture reached 29 (same positional-sequence
                // reasoning as refused_unknown_sender's 26 above).
                window_rows_ingested: 30,
                // 31..37: the Slice-A window-lane counters + gauge, appended in declaration order.
                window_sender_mismatch: 31,
                window_misauthored_body: 32,
                window_unknown_row: 33,
                window_open_sent: 34,
                window_close_sent: 35,
                window_keepalives_sent: 36,
                // 38..69: the Slice-B composer + the composed-egress/tombstone counters, in
                // declaration order.
                window_level_refused: 38,
                window_body_stale: 39,
                window_body_preroster: 40,
                window_folds: 41,
                window_fold_hits: 42,
                window_fold_divergence: 43,
                window_full_chain_folds: 44,
                window_chains_held: 45,
                window_sky_anchored: 145,
                window_chain_stamp_gap_max: 245,
                attach_lineage_unresolved: 345,
                window_compose_hold_ticks: 46,
                window_hop_dead: 47,
                window_t_monotone_stalled: 48,
                window_chain_cycle: 49,
                window_instant_mismatch: 50,
                window_rotated_refused: 51,
                window_far_rows: 52,
                window_looks_carried: 54,
                window_origin_swap_deferred: 55,
                window_origin_swap_forced: 56,
                scene_levels_restated: 57,
                window_trace_rows: 59,
                sessions_closed_peer_lost: 60,
                sessions_closed_peer_reincarnated: 61,
                sessions_replaced_by_relogin: 62,
                interest_recipient_unsubscribed: 63,
                interest_removals_fanned: 64,
                window_alien_rows: 52,
                window_hop_invalid: 53,
                window_unresolved_standing: 54,
                window_dedup_disagree: 55,
                window_dedup_max_dev_cells: 56,
                window_head_reads_sent: 57,
                window_composed_rows: 58,
                old_realm_frames_dropped: 59,
                old_scene_deltas_dropped: 60,
                scene_levels_sent: 61,
                scene_deltas_sent: 62,
                scene_datagrams_sent: 63,
                window_relays_ingested: 64,
                window_relay_undecodable: 65,
                window_relay_unvouched: 66,
                window_relay_stale: 67,
                window_relay_rows_composed: 68,
                window_relay_descent_refused: 82,
                window_relay_stamp_missing: 83,
                window_relay_unrostered: 84,
                window_relay_stamp_skew_ticks: 85,
                window_relay_moving_skew_ticks: 58,
                window_relay_depth_max: 86,
                window_looks_pruned: 80,
                window_relay_levels_pruned: 81,
                // 87..88: the look-horizon slice-3 interior-forward pair, appended in
                // declaration order.
                window_relay_interior_unvouched: 87,
                window_relay_interior_filtered: 88,
                star_catalogue_parts_sent: 89,
                sky_alive_beats_sent: 90,
                sky_requests_sent: 91,
                sky_held_stated: 92,
                sky_parts_skipped: 93,
                sessions_open: 24,
                dynamic_shards: 25,
                windows_open: 37,
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
        let empty = AdminSnapshot::shaped_empty(UniverseTick(1), EpochId(1), 0);
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
        let empty = AdminSnapshot::shaped_empty(UniverseTick(1), EpochId(1), 0);

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
