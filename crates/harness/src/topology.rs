//! The in-process `Topology`: N nodes + the fabric, advanced tick-by-tick with
//! deterministic ordering, scheduled crashes (three `CrashWhen` phases), per-node
//! stagger, and a serializable logical trace (`docs/design/test_harness.md` §2).
//!
//! One `step()` = one topology tick: 12,000 of them run in milliseconds with zero
//! real sleeping. The trace is the byte-comparable artifact behind the chaos
//! reproducibility gate.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use vd_core::pose::{RealmId, StampedPose};
use vd_core::{EntityId, Fence, NodeId, SessionId, TickId};
use vd_node::TickReport;
use vd_sim::stub::DiscardReason;
use vd_wire::seams::directory::{DirectoryKey, OwnerRecord};

use crate::fabric::{CrashWhen, FaultFabric};

/// What one node EXPOSES to the oracles, on request: the orchestrator's directory
/// view, a shard's held-set + input logs, a client's sent log. Ground truth for
/// AUTHORITY-UNIQUE / INPUT-CONSERVATION — nodes report, oracles audit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InspectReport {
    /// Directory records (only the orchestrator fills this).
    pub directory: Vec<(DirectoryKey, OwnerRecord)>,
    /// Entities this node holds authoritatively, with the directory-recorded fence
    /// it holds them at (shards fill this; FENCE-9 asserts the fence matches).
    pub held_entities: Vec<(EntityId, Fence)>,
    /// The held entities' POSES (shards fill this; 1d.1). The cross-shard crossing gate asserts a
    /// transferred entity's dest pose equals its sanitized source pose (and is non-origin) — proof
    /// the entity STATE crossed, not just authority. Aligned 1:1 with `held_entities` by entity.
    pub held_poses: Vec<(EntityId, StampedPose)>,
    /// Realms this node holds authoritatively, with their fence (shards fill this;
    /// extends AUTHORITY-UNIQUE to realm keys — FENCE-3).
    pub held_realms: Vec<(RealmId, Fence)>,
    /// Realms this node has REQUESTED but not yet confirmed (the legal
    /// commit-to-knowledge window — excuses a recorded realm not yet held).
    pub pending_realms: Vec<RealmId>,
    /// Entities this node has REQUESTED authority for but not yet seen the
    /// directory confirm (the legal commit-to-knowledge window; a pending entry
    /// excuses a directory record with no holder — an orphan does not).
    pub pending_entities: Vec<EntityId>,
    /// Entities this node is RELEASING (revoke in flight): still held, possibly
    /// already cleared from the directory — the legal release window.
    pub departing_entities: Vec<EntityId>,
    /// Entities this node hosts a NON-SIMULATING dot for — a retained cross-shard collider GHOST
    /// (`Authority::Ghost`, a source ghost fed by the new owner; or a transient Frozen). NOT held
    /// authoritatively (excluded from `held_entities`). Band-exit teardown (1d.5b.3c) REMOVES the
    /// dot, so a torn-down ghost drops out of this list — the oracle ground truth for the ghost
    /// lifecycle END.
    pub ghost_dots: Vec<EntityId>,
    /// LIVE (in-flight) transfers this node knows of — ORCHESTRATOR-ONLY (from `SagaRuntimeRes`);
    /// empty on shards/clients and once every saga tombstones. The mid-flight AUTHORITY-UNIQUE oracle
    /// (1d.5b.3d) keys its W1 transfer-window excuse on this: it excuses the post-CAS, pre-demote
    /// window (source still `Owned` while the directory records the dest) ONLY for a listed subject of
    /// the matching (source→dest) shape — so the per-tick gate never masks a real split-brain.
    pub active_transfers: Vec<vd_node::saga_runtime::ActiveTransfer>,
    /// TRANSIENTS this shard holds AUTHORITATIVELY (D-7), with the realm-lease fence each is anchored
    /// to — the `is_held()` subset of `OwnedTransients` (the uncounted `Arriving` mid-flight tier is
    /// EXCLUDED, the transient twin of `ghost_dots` exclusion). The `TRANSIENT-AUTHORITY-HELD` oracle
    /// ground truth: every live transient in exactly ONE shard's set, anchored at that shard's realm
    /// fence, backed by a committed go-token (`batch_goes`). NEVER a directory `OwnerRecord`.
    pub owned_transients: Vec<(EntityId, Fence)>,
    /// The held transients' POSES (D-7b — the transient twin of `held_poses`): the `is_held()` subset's
    /// re-advanced poses, the render-continuity ground truth for a MOVING debris. The uncounted
    /// `Arriving`/`Departing` tiers are EXCLUDED (not rendered). Feeds the moving-Debris POSE-CONTINUITY
    /// gate (no teleport across the cut). Aligned by entity with `owned_transients`.
    pub held_transient_poses: Vec<(EntityId, StampedPose)>,
    /// Committed batched-transient go-tokens (D-7) — ORCHESTRATOR-ONLY (from `SagaRuntimeRes`); empty
    /// on shards/clients. `(BatchId, commit_fence)`; the `TRANSIENT-AUTHORITY-HELD` oracle cross-checks
    /// each held transient's anchor against a go-token here (a transient has no directory row to check).
    pub batch_goes: Vec<(vd_core::BatchId, Fence)>,
    /// The MONOTONIC go-token WRITE count (D-7c G-TIER) — ORCHESTRATOR-ONLY. `batch_goes.len()` is
    /// false-green for the write-rate claim (idempotent `or_insert` collapses same-key writes); the
    /// gate sums THIS and asserts it equals the distinct-batch count, NOT the burst item count.
    pub batch_go_writes: u64,
    /// HANDOVER-attributable transient LOSS per kind (D-7b.3), from `StubStats.transients_lost_in_handover`
    /// — `(EntityKind, lost)` in kind order. The `verify_transient_loss_budget` gate sums this per kind
    /// across shards and compares to the kind's `LossBudget` (NOT the gross `transients_dropped`).
    pub transient_loss: Vec<(vd_core::entity_kind::EntityKind, u64)>,
    /// Inputs this node APPLIED, in order (shards fill this).
    pub applied_inputs: Vec<(SessionId, u64)>,
    /// Inputs this node DISCARDED, with the typed reason (shards fill this).
    pub discarded_inputs: Vec<(SessionId, Option<u64>, DiscardReason)>,
    /// Inputs this node SENT (scripted clients fill this).
    pub sent_inputs: Vec<(SessionId, u64)>,
    /// The `resume_from_seq` the gateway EMITTED in the most recent `OpenInputSlot` this shard
    /// honored — the transfer-dest's resume watermark (`None` on nodes/runs with no cut). The
    /// cross-cut conservation gate asserts it equals the client's own CUT_MARKER seq, threaded
    /// through the real saga (D-28). Read from `StubStats.last_input_slot_resume`, NOT the dot,
    /// so it is immune to the provisional dot's watermark advancing or its grant flipping.
    pub dest_resume_seq: Option<u64>,
    /// `InputLog.window_evictions` — inputs the bounded log silently aged out of its window.
    /// MUST be 0 for a conservation assertion to be trustworthy (a non-zero value means the
    /// oracle did not see the whole run).
    pub input_window_evictions: u64,
    /// Slice 3g — DURABLE geometric crossings REQUESTED by this shard's `evaluate_realm_boundaries`
    /// (from `StubStats.crossings_requested`; `0`/None-arm on the orchestrator/client). The Leg-1
    /// observable of the crossing-e2e: a triggered durable crossing emitted a `CrossingRequest`.
    pub crossings_requested: u64,
    /// Slice 3g — DURABLE crossing SAGAS STARTED on the orchestrator (from `SagaRuntimeRes.crossings_started`;
    /// `0`/None-arm on shards/clients). THE composition proof of the crossing-e2e: a source's
    /// `CrossingRequest` resolved its three heads and BECAME a saga — the geometric trigger drove the
    /// transfer saga end-to-end (never hand-fed via `trigger_transfer`).
    pub crossings_started: u64,
    /// Slice 3g — TRANSIENT geometric crossings REQUESTED by this shard (from
    /// `StubStats.transient_crossings_requested`). The transient (HR2 second-class) Leg-1 observable.
    pub transient_crossings_requested: u64,
    /// Slice 3g — TRANSIENT crossings GRANTED on the orchestrator (from
    /// `SagaRuntimeRes.transient_crossings_granted`). The transient auto-grant composition proof.
    pub transient_crossings_granted: u64,
    /// Slice 3g — `RequestInFlight` durable-crossing latches CLEARED by a saga terminal at the SOURCE
    /// (from `StubStats.crossing_latches_cleared`). Proves the POSITIVE latch clear on a committed
    /// crossing — the latch lifecycle END.
    pub crossing_latches_cleared: u64,
    /// Slice 3g — the entities currently latched in this shard's `RequestInFlight` (the standing
    /// durable-crossing latches; `RequestInFlight.0.keys()`, sorted). Empty once a crossing's terminal
    /// clears the latch — the abort/commit-leg "latch empty" ground truth.
    pub in_flight_latches: Vec<EntityId>,
    /// Slice 3g abort/crash-leg — durable crossing-abort replies STAGED on the orchestrator awaiting a source
    /// `CrossingAbortedAck` (from `SagaRuntimeRes.pending_abort_replies_len`; `0`/None-arm on shards/clients).
    /// The Mechanism-Y observable: `>= 1` mid-flight proves the crossing-origin pre-CAS abort staged its
    /// persisted latch-clear obligation (a non-crossing abort stages NOTHING); `== 0` after proves the source
    /// ack reaped it; `>= 1` immediately after an orchestrator rebuild proves the entry rode the WAL.
    pub pending_abort_replies: usize,
}

/// Anything the topology can drive. `ShardNode<FabricTransport>` is the canonical
/// implementation; scripted clients implement it too.
pub trait SteppableNode {
    fn node_id(&self) -> NodeId;
    fn step(&mut self) -> TickReport;
    /// Expose oracle ground truth (default: nothing to report).
    fn inspect(&mut self) -> InspectReport {
        InspectReport::default()
    }
    /// Opt-in downcast hook for scenario code that drives a concrete node type
    /// (scripted clients override this; infrastructure nodes need not).
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

impl<T: vd_sim::io::Transport + 'static> SteppableNode for vd_node::ShardNode<T> {
    fn node_id(&self) -> NodeId {
        vd_node::ShardNode::node_id(self)
    }
    fn step(&mut self) -> TickReport {
        self.step_tick()
    }
    fn inspect(&mut self) -> InspectReport {
        inspect_world(self.world_mut())
    }
    /// Scenario code that DRIVES a transfer (e.g. the orchestrator's `SagaRuntimeRes`) needs
    /// the concrete node; opt in (the default returns `None`).
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

/// Monomorphic resource scrape: every node kind reports whatever oracle-relevant
/// resources its registered systems maintain.
fn inspect_world(world: &mut bevy_ecs::prelude::World) -> InspectReport {
    let mut report = InspectReport::default();
    if let Some(dir) = world.get_resource::<vd_node::orchestrator::DirectoryRes>() {
        report.directory = dir.0.entries().map(|(k, r)| (*k, *r)).collect();
    }
    if let Some(rt) = world.get_resource::<vd_node::saga_runtime::SagaRuntimeRes>() {
        // Orchestrator-only: the live-saga set the mid-flight AUTHORITY-UNIQUE oracle excuses against,
        // and the committed go-token ledger the TRANSIENT-AUTHORITY-HELD oracle cross-checks (D-7).
        report.active_transfers = rt.active_transfers();
        report.batch_goes = rt.batch_goes();
        report.batch_go_writes = rt.batch_go_writes(); // D-7c G-TIER write-rate observable
        // Slice 3g: the crossing-e2e composition proofs — a source `CrossingRequest` that BECAME a saga
        // (durable) and a `TransientCrossingRequest` that was auto-GRANTED (transient). Orchestrator-only.
        report.crossings_started = rt.crossings_started();
        report.transient_crossings_granted = rt.transient_crossings_granted();
        // Slice 3g abort/crash-leg: the staged durable crossing-abort replies (Mechanism-Y), the
        // pre-CAS abort's persisted latch-clear obligation. Orchestrator-only (shards/clients: 0).
        report.pending_abort_replies = rt.pending_abort_replies_len();
    }
    if let Some(dots) = world.get_resource::<vd_sim::stub::Dots>() {
        // A dot is HELD only while it SIMULATES (`Authority::Owned`, 1d.4b): a Ghost (a retained
        // self-fenced source, or a pre-promote dest) is EXCLUDED so a mid-flight retained Ghost
        // cannot false-trip `verify_authority_unique` (WrongHolderCount). `granted` (the directory
        // predicate) still drives pending/departing below — they are NOT the authority truth.
        report.held_entities = dots
            .0
            .values()
            .filter(|d| d.authority.simulates())
            .map(|d| (d.entity, d.entity_fence))
            .collect();
        report.held_poses = dots
            .0
            .values()
            .filter(|d| d.authority.simulates())
            .map(|d| (d.entity, d.pose))
            .collect();
        report.pending_entities = dots
            .0
            .values()
            .filter(|d| !d.granted)
            .map(|d| d.entity)
            .collect();
        report.departing_entities = dots
            .0
            .values()
            .filter(|d| d.granted & d.departing)
            .map(|d| d.entity)
            .collect();
        // Retained cross-shard collider ghosts (1d.5b.3c): a dot present but NOT simulating. Band-exit
        // teardown removes the dot, so it drops out here — the ghost-lifecycle-END ground truth.
        report.ghost_dots = dots
            .0
            .values()
            .filter(|d| !d.authority.simulates())
            .map(|d| d.entity)
            .collect();
    }
    if let Some(owned) = world.get_resource::<vd_sim::stub::OwnedTransients>() {
        // D-7: the held-set-anchored transients. Only the `is_held()` subset is authoritative — the
        // uncounted mid-flight `Arriving` tier is EXCLUDED (the transient twin of the ghost exclusion
        // above) so a mid-flight adopt can never false-trip TRANSIENT-AUTHORITY-HELD (DoubleHeld).
        report.owned_transients = owned
            .0
            .iter()
            .filter(|(_, t)| t.status.is_held())
            .map(|(entity, t)| (*entity, t.anchor_fence))
            .collect();
        // D-7b: the same is_held() subset's POSES — the moving-Debris render-continuity ground truth.
        report.held_transient_poses = owned
            .0
            .iter()
            .filter(|(_, t)| t.status.is_held())
            .map(|(entity, t)| (*entity, t.pose))
            .collect();
    }
    if let (Some(config), Some(authority)) = (
        world.get_resource::<vd_sim::stub::StubConfig>(),
        world.get_resource::<vd_sim::stub::RealmAuthority>(),
    ) {
        // A shard reports its realm authority (and fence) only while it holds the
        // lease — a self-fenced shard reports nothing (FENCE-3).
        match authority.0 {
            Some(fence) => report.held_realms = vec![(config.realm, fence)],
            None => report.pending_realms = vec![config.realm],
        }
    }
    if let Some(log) = world.get_resource::<vd_sim::stub::InputLog>() {
        report.applied_inputs = log.applied();
        report.discarded_inputs = log.discarded();
        report.input_window_evictions = log.window_evictions;
    }
    if let Some(stats) = world.get_resource::<vd_sim::stub::StubStats>() {
        report.dest_resume_seq = stats.last_input_slot_resume;
        // D-7b.3: the handover-attributable per-kind loss, in deterministic kind order (the
        // DetHashMap iteration is fixed-seed-deterministic, but sort so the Vec is replay-stable
        // regardless of insertion order — the byte-identical-trace gate).
        let mut loss: Vec<(vd_core::entity_kind::EntityKind, u64)> = stats
            .transients_lost_in_handover
            .iter()
            .map(|(k, n)| (*k, *n))
            .collect();
        loss.sort_by_key(|(k, _)| *k as u8);
        report.transient_loss = loss;
        // Slice 3g: the shard-side crossing counters — Leg-1 (a `CrossingRequest`/`TransientCrossingRequest`
        // emitted) and the POSITIVE latch clear on a committed durable crossing.
        report.crossings_requested = stats.crossings_requested;
        report.transient_crossings_requested = stats.transient_crossings_requested;
        report.crossing_latches_cleared = stats.crossing_latches_cleared;
    }
    if let Some(in_flight) = world.get_resource::<vd_sim::stub::RequestInFlight>() {
        // Slice 3g: the standing durable-crossing latches (the abort/commit-leg "latch empty" ground
        // truth). BTreeMap keys are already sorted — the trace is replay-stable.
        report.in_flight_latches = in_flight.0.keys().copied().collect();
    }
    report
}

/// Per-node tick offset: node N fires only on topology ticks > its offset, so its
/// local tick permanently lags — deliberate tick-skew in the fast tier (there is NO
/// globally synchronized sim tick in production; the harness must exercise that).
#[derive(Clone, Debug, Default)]
pub struct StaggerPlan {
    offsets: BTreeMap<NodeId, u64>,
}

impl StaggerPlan {
    #[must_use]
    pub fn lockstep() -> StaggerPlan {
        StaggerPlan::default()
    }

    #[must_use]
    pub fn with_offset(mut self, node: NodeId, offset: u64) -> StaggerPlan {
        self.offsets.insert(node, offset);
        self
    }

    /// Does `node` fire its local tick on topology tick `t`?
    #[must_use]
    pub fn fires(&self, node: NodeId, t: TickId) -> bool {
        t.0 > self.offsets.get(&node).copied().unwrap_or(0)
    }
}

/// One serializable trace entry. The chaos gate compares postcard bytes of the whole
/// log across two separate processes — byte-identical or the run is nondeterministic.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceEvent {
    Stepped {
        tick: TickId,
        node: NodeId,
        drained: u64,
        sent: u64,
        backpressured: u64,
        /// The RELIABLE subset of this node's staging-cap shed this tick — a lost
        /// transfer/control/membership frame (vs benign latest-wins snapshot shedding).
        /// MUST stay 0 on a healthy run; [`Topology::verify_no_reliable_shed`] gates it.
        reliable_shed: u64,
        unreachable: u64,
    },
    Skipped {
        tick: TickId,
        node: NodeId,
    },
    Crashed {
        tick: TickId,
        node: NodeId,
        when: u8,
    },
    Resurrected {
        tick: TickId,
        node: NodeId,
    },
    Released {
        tick: TickId,
        deliveries: u64,
    },
}

/// A node's claims diverged from what the fabric actually delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "wire-truth violation at {node}: delivered {delivered} != claimed {claimed} \
     + crash-cleared {cleared_by_crash} + pending {pending}"
)]
pub struct WireTruthViolation {
    pub node: NodeId,
    pub delivered: u64,
    pub claimed: u64,
    pub cleared_by_crash: u64,
    pub pending: u64,
}

/// A node shed a RELIABLE outbound frame at its staging cap — a lost transfer/control/
/// membership frame (e.g. a post-commit `OpenInputSlot` or resume `SessionInput` that
/// strands a transferred player's input), never a benign latest-wins snapshot drop.
/// Must be 0 on any healthy/zero-fault run; nonzero is a distinct overload ALERT, not
/// indistinguishable from a dropped snapshot (the observability hole the audit caught).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "reliable-shed violation at {node}: {reliable_shed} reliable frame(s) shed at the staging cap"
)]
pub struct ReliableShedViolation {
    pub node: NodeId,
    pub reliable_shed: u64,
}

/// The deterministic multi-node driver.
pub struct Topology {
    nodes: BTreeMap<NodeId, Box<dyn SteppableNode>>,
    fabric: FaultFabric,
    stagger: StaggerPlan,
    /// Crashes keyed by the exact phase they fire in: a phase scan ranges precisely
    /// its own `(tick, phase, *)` slice — every entry found fires, no guards.
    crashes: BTreeSet<(TickId, CrashWhen, NodeId)>,
    /// Resurrections fire once, at the start of a tick (before any phase).
    resurrects: BTreeSet<(TickId, NodeId)>,
    tick: TickId,
    trace: Vec<TraceEvent>,
}

impl Topology {
    #[must_use]
    pub fn new(fabric: FaultFabric, stagger: StaggerPlan) -> Topology {
        Topology {
            nodes: BTreeMap::new(),
            fabric,
            stagger,
            crashes: BTreeSet::new(),
            resurrects: BTreeSet::new(),
            tick: TickId(0),
            trace: Vec::new(),
        }
    }

    /// Add a node (its transport must already be registered on this fabric).
    pub fn add_node(&mut self, node: Box<dyn SteppableNode>) {
        let id = node.node_id();
        let previous = self.nodes.insert(id, node);
        assert!(previous.is_none(), "duplicate topology node: {id}");
    }

    /// REPLACE an existing node with a freshly-rebuilt one — the D-6 kill-9 REBUILD (the durable-recovery
    /// crash cells): the old node's World is DROPPED here (its in-memory state is gone, modeling the lost
    /// process RAM), and the caller has already built `node` afresh — re-attached via
    /// `FaultFabric::reregister` and RE-HYDRATED from the node's retained `Store`. The id MUST be present
    /// (a rebuild reclaims a live identity, unlike `add_node`).
    ///
    /// # Panics
    /// If `id` is not already present — a rebuild reclaims an existing node.
    pub fn replace_node(&mut self, node: Box<dyn SteppableNode>) {
        let id = node.node_id();
        let previous = self.nodes.insert(id, node);
        assert!(
            previous.is_some(),
            "replace_node reclaims an existing node: {id}"
        );
    }

    /// Schedule a crash at `tick` in phase `when`.
    pub fn schedule_crash(&mut self, node: NodeId, tick: TickId, when: CrashWhen) {
        self.crashes.insert((tick, when, node));
    }

    /// Schedule a resurrection at `tick` (fires at tick start, before injection).
    pub fn schedule_resurrect(&mut self, node: NodeId, tick: TickId) {
        self.resurrects.insert((tick, node));
    }

    fn apply_crashes(&mut self, phase: CrashWhen) {
        let tick = self.tick;
        let due: Vec<NodeId> = self
            .crashes
            .range((tick, phase, NodeId(0))..=(tick, phase, NodeId(u64::MAX)))
            .map(|(_, _, node)| *node)
            .collect();
        for node in due {
            self.crashes.remove(&(tick, phase, node));
            self.fabric.crash(node);
            self.trace.push(TraceEvent::Crashed {
                tick,
                node,
                when: phase as u8,
            });
        }
    }

    fn apply_resurrects(&mut self) {
        let tick = self.tick;
        let due: Vec<NodeId> = self
            .resurrects
            .range((tick, NodeId(0))..=(tick, NodeId(u64::MAX)))
            .map(|(_, node)| *node)
            .collect();
        for node in due {
            self.resurrects.remove(&(tick, node));
            self.fabric.resurrect(node);
            self.trace.push(TraceEvent::Resurrected { tick, node });
        }
    }

    /// One topology tick: lifecycle phases interleave with injection and stepping
    /// exactly as the design's three-phase crash model prescribes.
    pub fn step(&mut self) {
        self.tick = self.tick.next();
        let tick = self.tick;

        // PHASE A: resurrections first (a node returning this tick is alive for its
        // deliveries), then crashes-before-injection (the node misses this tick).
        self.apply_resurrects();
        self.apply_crashes(CrashWhen::PreInject);

        // 1. Release due deliveries into inbound queues.
        let deliveries = self.fabric.pump(tick) as u64;
        self.trace.push(TraceEvent::Released { tick, deliveries });

        // PHASE B: crashes after injection (messages sit unacked -> must redeliver).
        self.apply_crashes(CrashWhen::PostInject);

        // 2. Step every live node in NodeId order (deterministic), honoring stagger.
        let ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        for id in ids {
            if self.fabric.is_crashed(id) {
                self.trace.push(TraceEvent::Skipped { tick, node: id });
                continue;
            }
            if !self.stagger.fires(id, tick) {
                self.trace.push(TraceEvent::Skipped { tick, node: id });
                continue;
            }
            let report = self
                .nodes
                .get_mut(&id)
                .expect("node present: ids snapshot from the map")
                .step();
            // The node SURVIVED its step: ack everything it drained (at-least-once
            // commit point).
            self.fabric.ack_survivor(id);
            self.trace.push(TraceEvent::Stepped {
                tick,
                node: id,
                drained: report.drained as u64,
                sent: report.sent as u64,
                backpressured: report.backpressured as u64,
                reliable_shed: report.reliable_shed as u64,
                unreachable: report.unreachable as u64,
            });
        }

        // PHASE C: crashes after stepping (the tick's effects survive in queues).
        self.apply_crashes(CrashWhen::PostStep);

        // Conservation is checked EVERY tick: a leak is a harness bug, immediately.
        assert!(
            self.fabric.conservation_holds(),
            "fabric conservation violated at {tick}"
        );
    }

    #[must_use]
    pub fn tick(&self) -> TickId {
        self.tick
    }

    #[must_use]
    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }

    /// Borrow one node by id (scenario code drives concrete types through
    /// `SteppableNode::as_any_mut`).
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut (dyn SteppableNode + 'static)> {
        self.nodes.get_mut(&id).map(|n| n.as_mut())
    }

    /// Collect every node's oracle report (deterministic NodeId order).
    pub fn inspect_all(&mut self) -> Vec<(NodeId, InspectReport)> {
        self.nodes
            .iter_mut()
            .map(|(id, n)| (*id, n.inspect()))
            .collect()
    }

    /// Like [`inspect_all`](Self::inspect_all) but EXCLUDES nodes the fabric reports DEAD
    /// (killed/crashed-not-resurrected) — the dead-node-aware view for the P3 crash matrix. A killed
    /// participant's stale `Dot`/`held_entities` are a CORPSE that must not be read as live authority;
    /// dropping them turns a "held@dead-node" false-pass / a legit-park false-RED into the honest
    /// directory-vs-liveness verdict (`oracle::verify_authority_unique_excluding`). `inspect_all`
    /// stays for crash+resurrect cells, where the resurrected node IS live and must be inspected.
    pub fn inspect_live(&mut self) -> Vec<(NodeId, InspectReport)> {
        let dead: std::collections::BTreeSet<NodeId> = self
            .nodes
            .keys()
            .copied()
            .filter(|id| self.fabric.is_dead(*id))
            .collect();
        self.nodes
            .iter_mut()
            .filter(|(id, _)| !dead.contains(id))
            .map(|(id, n)| (*id, n.inspect()))
            .collect()
    }

    /// The set of nodes the fabric currently reports DEAD — the `dead` argument for
    /// [`oracle::verify_authority_unique_excluding`](crate::oracle::verify_authority_unique_excluding)
    /// when a scenario inspects with `inspect_all` (a kill cell keeps the dead node's report so the
    /// oracle can attribute the orphan, but excludes it from the live-holder logic).
    #[must_use]
    pub fn dead_nodes(&self) -> std::collections::BTreeSet<NodeId> {
        self.nodes
            .keys()
            .copied()
            .filter(|id| self.fabric.is_dead(*id))
            .collect()
    }

    /// THE WIRE-TRUTH CHECK (the P0 WireMonitor): a node's CLAIMED drain counts (its
    /// own TickReports, summed from the trace) must exactly account for what the
    /// fabric ACTUALLY delivered: `delivered == claimed + cleared_by_crash + pending`.
    /// A node that lies about what it received — the old system's connect_tx bug
    /// class, where internal state diverged from the wire — fails this loudly.
    pub fn verify_wire_truth(&self) -> Result<(), WireTruthViolation> {
        for id in self.nodes.keys() {
            let claimed: u64 = self
                .trace
                .iter()
                .filter_map(|e| match e {
                    TraceEvent::Stepped { node, drained, .. } if node == id => Some(*drained),
                    _ => None,
                })
                .sum();
            let truth = self.fabric.delivery_accounting(*id);
            if truth.delivered != claimed + truth.cleared_by_crash + truth.pending {
                return Err(WireTruthViolation {
                    node: *id,
                    delivered: truth.delivered,
                    claimed,
                    cleared_by_crash: truth.cleared_by_crash,
                    pending: truth.pending,
                });
            }
        }
        Ok(())
    }

    /// NO RELIABLE SHED: no node shed a reliable (transfer/control/membership) frame at its
    /// outbound staging cap across the whole run. A reliable shed silently strands a transfer
    /// command or a post-commit input slot — and (pre the D-8 re-drive) there is no recovery —
    /// so on any healthy/zero-fault scenario this MUST hold. Returns the FIRST offending node
    /// (deterministic trace order). The metric is the node's own `TickReport.reliable_shed`,
    /// summed from the trace — wire-truth, not internal hope (same discipline as
    /// [`Self::verify_wire_truth`]).
    pub fn verify_no_reliable_shed(&self) -> Result<(), ReliableShedViolation> {
        for id in self.nodes.keys() {
            let reliable_shed: u64 = self
                .trace
                .iter()
                .filter_map(|e| match e {
                    TraceEvent::Stepped {
                        node,
                        reliable_shed,
                        ..
                    } if node == id => Some(*reliable_shed),
                    _ => None,
                })
                .sum();
            if reliable_shed > 0 {
                return Err(ReliableShedViolation {
                    node: *id,
                    reliable_shed,
                });
            }
        }
        Ok(())
    }

    /// The byte-comparable artifact for the reproducibility gate.
    #[must_use]
    pub fn trace_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(&self.trace).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::{Res, ResMut};
    use vd_node::app::{InboundBox, NodeConfig, OutboundBox, OutboundStagingCap};
    use vd_sim::capability::NodeKind;
    use vd_sim::io::{Inbound, MsgClass, Transport};

    const A: NodeId = NodeId(1);
    const B: NodeId = NodeId(2);

    /// Infrastructure relay (not game logic): forward every Wire payload to a fixed
    /// peer, so a pair of relays ping-pongs traffic across their links forever.
    fn build_relay_node(fabric: &FaultFabric, id: NodeId, peer: NodeId) -> Box<dyn SteppableNode> {
        let mut node = vd_node::build_app(
            NodeConfig {
                node_id: id,
                kind: NodeKind::StubShard,
            },
            fabric.register(id),
        );
        node.schedule_mut().add_systems(
            move |inbox: Res<InboundBox>, mut outbox: ResMut<OutboundBox>| {
                for msg in &inbox.0 {
                    if let Inbound::Wire { class, bytes, .. } = msg {
                        outbox.0.push((
                            peer,
                            *class,
                            bytes.clone(),
                            vd_sim::io::Durability::Ephemeral,
                        ));
                    }
                }
            },
        );
        Box::new(node)
    }

    /// A↔B relay pair with one seed message injected toward B (the seeder endpoint
    /// only sends; B's relay forwards to A, so traffic crosses BOTH links forever).
    fn seeded_topology(fabric: &FaultFabric) -> Topology {
        let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());
        topo.add_node(build_relay_node(fabric, A, B));
        topo.add_node(build_relay_node(fabric, B, A));
        let mut seeder = fabric.register(NodeId(99));
        seeder
            .send(B, MsgClass::Control, vec![1].into())
            .expect("seed message accepted");
        topo
    }

    #[test]
    fn replace_node_swaps_a_rebuilt_node_in_place() {
        // D-6: replace_node reclaims an EXISTING identity (the kill-9 rebuild) — exactly one node per id
        // (the old World dropped, the rebuilt one swapped in), never a duplicate / loss. The rebuilt node
        // re-attaches via `reregister` (plain `register` would panic on the still-live identity).
        let fabric = FaultFabric::new(7, 2);
        let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());
        let original = vd_node::build_app(
            NodeConfig {
                node_id: A,
                kind: NodeKind::StubShard,
            },
            fabric.register(A),
        );
        topo.add_node(Box::new(original));
        assert_eq!(topo.inspect_all().len(), 1);
        let rebuilt = vd_node::build_app(
            NodeConfig {
                node_id: A,
                kind: NodeKind::StubShard,
            },
            fabric.reregister(A),
        );
        topo.replace_node(Box::new(rebuilt));
        assert_eq!(
            topo.inspect_all().len(),
            1,
            "replace swapped A in place — no duplicate, no loss"
        );
        topo.step(); // the rebuilt node steps cleanly
    }

    #[test]
    #[should_panic(expected = "replace_node reclaims an existing node")]
    fn replace_node_panics_on_an_absent_id() {
        // The mirror of `add_node`'s duplicate guard: `replace_node` RECLAIMS a live identity, so an
        // absent id is a caller bug (a rebuild of a node never added). Covers the assert's panic arm +
        // its lazily-evaluated `{id}` message (HR5: a format arg is uncoverable until the panic fires).
        let fabric = FaultFabric::new(8, 2);
        let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());
        let orphan = vd_node::build_app(
            NodeConfig {
                node_id: A,
                kind: NodeKind::StubShard,
            },
            fabric.register(A),
        );
        topo.replace_node(Box::new(orphan)); // A was never `add_node`d — reclaims nothing
    }

    #[test]
    fn ping_pong_runs_deterministically_for_many_ticks() {
        let fabric = FaultFabric::new(5, 2);
        let mut topo = seeded_topology(&fabric);
        for _ in 0..100 {
            topo.step();
        }
        assert_eq!(topo.tick(), TickId(100));
        // The echo pair keeps the message bouncing: every tick after warmup steps
        // both nodes and delivers exactly one message.
        let stepped = topo
            .trace()
            .iter()
            .filter(|e| matches!(e, TraceEvent::Stepped { .. }))
            .count();
        assert_eq!(stepped, 200, "both nodes stepped every tick");
    }

    #[test]
    fn identical_seeds_produce_byte_identical_traces() {
        let run = |seed: u64, with_faults: bool| {
            let fabric = FaultFabric::new(seed, 2);
            if with_faults {
                fabric.set_policy(
                    A,
                    B,
                    crate::fabric::LinkPolicy {
                        drop_p: 0.4,
                        max_extra_delay_ticks: 3,
                        ..crate::fabric::LinkPolicy::default()
                    },
                );
            }
            let mut topo = seeded_topology(&fabric);
            for _ in 0..64 {
                topo.step();
            }
            topo.trace_bytes()
        };
        // Same seed: byte-identical, with and without faults.
        assert_eq!(run(7, false), run(7, false));
        assert_eq!(run(7, true), run(7, true));
        // Fault-free runs are seed-INDEPENDENT (no draws affect anything) — that is
        // itself a determinism property worth pinning.
        assert_eq!(run(7, false), run(8, false));
        // With faults, the seed IS the fault tape: different seeds diverge.
        assert_ne!(run(7, true), run(8, true));
    }

    #[test]
    fn post_inject_crash_redelivers_after_resurrection() {
        let fabric = FaultFabric::new(11, 2);
        let mut topo = seeded_topology(&fabric);
        // Crash B WITH the seed message sitting in its inbound; resurrect later.
        topo.schedule_crash(B, TickId(1), CrashWhen::PostInject);
        topo.schedule_resurrect(B, TickId(4));
        for _ in 0..12 {
            topo.step();
        }
        // B was skipped while crashed, then stepped with a drained message again
        // (redelivery can land exactly on the resurrection tick). Non-short-circuit
        // `&` keeps the predicate fully evaluated (no uncoverable && arms).
        let b_drained_after = topo.trace().iter().any(|e| match e {
            TraceEvent::Stepped {
                node,
                drained,
                tick,
                ..
            } => (*node == B) & (*drained > 0) & (tick.0 >= 4),
            _ => false,
        });
        assert!(
            b_drained_after,
            "the unacked message redelivered after resurrection"
        );
        let skipped_nodes: Vec<NodeId> = topo
            .trace()
            .iter()
            .filter_map(|e| match e {
                TraceEvent::Skipped { node, .. } => Some(*node),
                _ => None,
            })
            .collect();
        let b_skips = skipped_nodes.iter().filter(|n| **n == B).count();
        assert!(b_skips >= 2, "B skipped while crashed");
    }

    #[test]
    fn stagger_delays_a_nodes_first_tick() {
        let fabric = FaultFabric::new(3, 2);
        let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep().with_offset(B, 3));
        topo.add_node(build_relay_node(&fabric, A, B));
        topo.add_node(build_relay_node(&fabric, B, A));
        for _ in 0..5 {
            topo.step();
        }
        let stepped_nodes: Vec<NodeId> = topo
            .trace()
            .iter()
            .filter_map(|e| match e {
                TraceEvent::Stepped { node, .. } => Some(*node),
                _ => None,
            })
            .collect();
        let b_steps = stepped_nodes.iter().filter(|n| **n == B).count();
        assert_eq!(b_steps, 2, "B fired only on ticks 4 and 5");
    }

    #[test]
    fn all_three_crash_phases_fire_and_trace() {
        let fabric = FaultFabric::new(13, 2);
        let mut topo = seeded_topology(&fabric);
        topo.schedule_crash(A, TickId(2), CrashWhen::PreInject);
        topo.schedule_resurrect(A, TickId(3));
        topo.schedule_crash(B, TickId(5), CrashWhen::PostStep);
        topo.schedule_resurrect(B, TickId(7));
        for _ in 0..10 {
            topo.step();
        }
        let crash_phases: Vec<u8> = topo
            .trace()
            .iter()
            .filter_map(|e| match e {
                TraceEvent::Crashed { when, .. } => Some(*when),
                _ => None,
            })
            .collect();
        assert_eq!(
            crash_phases,
            vec![CrashWhen::PreInject as u8, CrashWhen::PostStep as u8]
        );
        let resurrections = topo
            .trace()
            .iter()
            .filter(|e| matches!(e, TraceEvent::Resurrected { .. }))
            .count();
        assert_eq!(resurrections, 2);
    }

    #[test]
    fn inspect_live_and_dead_nodes_track_a_killed_node() {
        // P3 crash matrix: a killed node is EXCLUDED from inspect_live (its corpse report is dropped)
        // but RETAINED by inspect_all (so a kill cell can still attribute the orphan); dead_nodes names it.
        let fabric = FaultFabric::new(19, 2);
        let mut topo = seeded_topology(&fabric);
        assert!(topo.dead_nodes().is_empty(), "no node dead at the start");
        assert_eq!(
            topo.inspect_live().len(),
            topo.inspect_all().len(),
            "all live initially"
        );
        fabric.kill(B);
        assert_eq!(
            topo.dead_nodes(),
            [B].into_iter().collect(),
            "the killed node is dead"
        );
        let live: Vec<NodeId> = topo.inspect_live().into_iter().map(|(n, _)| n).collect();
        assert!(
            !live.contains(&B),
            "the killed node is excluded from inspect_live"
        );
        assert!(live.contains(&A), "the live node remains");
        let all: Vec<NodeId> = topo.inspect_all().into_iter().map(|(n, _)| n).collect();
        assert!(
            all.contains(&B),
            "inspect_all RETAINS the killed node (orphan attribution)"
        );
    }

    #[test]
    fn killed_peer_notices_reach_the_relay_and_wire_truth_holds() {
        let fabric = FaultFabric::new(17, 2);
        let mut topo = seeded_topology(&fabric);
        for _ in 0..3 {
            topo.step();
        }
        // Permanent death mid-run: A's next forward toward B bounces back as a
        // NodeUnreachable notice, which A's relay drains (and does NOT forward —
        // the non-Wire inbound arm).
        fabric.kill(B);
        for _ in 0..10 {
            topo.step();
        }
        let a_unreachable: u64 = topo
            .trace()
            .iter()
            .filter_map(|e| match e {
                TraceEvent::Stepped {
                    node, unreachable, ..
                } => Some((*node, *unreachable)),
                _ => None,
            })
            .filter_map(|(node, unreachable)| (node == A).then_some(unreachable))
            .sum();
        assert!(a_unreachable > 0, "A observed the unreachable notice");
        topo.verify_wire_truth()
            .expect("wire truth holds with a killed node");
    }

    #[test]
    fn wire_truth_holds_for_honest_nodes_even_under_crashes() {
        let fabric = FaultFabric::new(21, 2);
        let mut topo = seeded_topology(&fabric);
        topo.schedule_crash(B, TickId(2), CrashWhen::PostInject);
        topo.schedule_resurrect(B, TickId(5));
        for _ in 0..20 {
            topo.step();
        }
        topo.verify_wire_truth()
            .expect("honest nodes account exactly");
    }

    /// THE META-DIVERGENCE TEST (P0 DoD): a node that LIES about what it received —
    /// internal state diverging from the wire, the old system's connect_tx bug class
    /// — must be caught by the wire-truth check. This proves the monitor checks wire
    /// reality, not internal hope.
    #[test]
    fn a_lying_node_is_caught_by_the_wire_truth_check() {
        struct Liar {
            inner: Box<dyn SteppableNode>,
        }
        impl SteppableNode for Liar {
            fn node_id(&self) -> NodeId {
                self.inner.node_id()
            }
            fn step(&mut self) -> TickReport {
                let mut report = self.inner.step();
                report.drained += 1; // claim a message that never arrived
                report
            }
        }

        let fabric = FaultFabric::new(31, 2);
        let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());
        topo.add_node(Box::new(Liar {
            inner: build_relay_node(&fabric, A, B),
        }));
        topo.add_node(build_relay_node(&fabric, B, A));
        let mut seeder = fabric.register(NodeId(99));
        seeder
            .send(B, MsgClass::Control, vec![1].into())
            .expect("seed message accepted");
        for _ in 0..5 {
            topo.step();
        }
        let violation = topo
            .verify_wire_truth()
            .expect_err("the liar must be caught");
        assert_eq!(violation.node, A);
        assert!(violation.claimed > violation.delivered);
        // The Liar uses the trait DEFAULTS: nothing to report, no downcast hook.
        let liar = topo.node_mut(A).expect("liar present");
        assert!(liar.as_any_mut().is_none(), "no downcast for infra nodes");
        assert_eq!(liar.inspect(), InspectReport::default());
        assert_eq!(topo.node_mut(NodeId(77)).map(|_| ()), None, "unknown id");
    }

    /// `inspect_world` scrapes whatever oracle-relevant resources a node's
    /// registered systems maintain — proven here on a mini orchestrator + stub
    /// cluster composed entirely from harness-visible crates.
    #[test]
    fn inspect_reports_directory_and_shard_ground_truth() {
        use vd_node::orchestrator::{OrchestratorConfig, register_orchestrator};
        use vd_sim::directory::DirectoryTuning;
        use vd_sim::stub::{StubConfig, register_stub_shard};

        let fabric = FaultFabric::new(33, 2);
        let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());

        let mut orch = vd_node::build_app(
            NodeConfig {
                node_id: A,
                kind: NodeKind::Orchestrator,
            },
            fabric.register(A),
        );
        let (world, schedule) = orch.parts_mut();
        register_orchestrator(
            world,
            schedule,
            &OrchestratorConfig {
                epoch: vd_core::EpochId(1),
                reserve_chunk: 64,
                clock_peers: vec![B],
                directory: DirectoryTuning {
                    lease_ttl_ticks: 100,
                    ..DirectoryTuning::default()
                },
                saga: vd_sim::saga::SagaTuning::default(),
                liveness: vd_sim::saga::LivenessTuning::default(),
                roster: std::collections::BTreeMap::new(),
            },
        );
        topo.add_node(Box::new(orch));

        let mut shard = vd_node::build_app(
            NodeConfig {
                node_id: B,
                kind: NodeKind::StubShard,
            },
            fabric.register(B),
        );
        let (world, schedule) = shard.parts_mut();
        vd_node::follower::register_clock_follower(world, schedule);
        register_stub_shard(
            world,
            schedule,
            StubConfig {
                realm: vd_core::pose::RealmId::System(5),
                frame: vd_core::pose::FrameRef::SystemSpace { system_seed: 5 },
                move_speed_mps: 1.0,
                tick_dt_s: 0.05,
                orchestrator: A,
                mint_seed: 3,
                input_log_capacity: 1_000_000,
                realm_recheck_interval: 0,
                lease_renew_interval_ticks: 0,
                self_fence_grace_ticks: 0,
                snapshot_datagram_budget: 1100,
                boundary: vd_core::geometry::BoundaryTuning::DEFAULT,
                request_ttl_ticks: 0,
            },
        );
        topo.add_node(Box::new(shard));

        // Before any step the shard has no realm authority yet: it reports the realm
        // as PENDING (the commit-to-knowledge window), not held.
        let early = topo.inspect_all();
        assert_eq!(early[1].1.held_realms, Vec::new());
        assert_eq!(
            early[1].1.pending_realms,
            vec![vd_core::pose::RealmId::System(5)],
            "an ungranted shard reports its realm pending"
        );

        // A few ticks: the shard wins its realm lease through the directory.
        for _ in 0..6 {
            topo.step();
        }
        let reports = topo.inspect_all();
        assert_eq!(reports.len(), 2);
        let (orch_id, orch_report) = &reports[0];
        assert_eq!(*orch_id, A);
        assert_eq!(
            orch_report.directory.len(),
            1,
            "the realm lease is recorded"
        );
        assert_eq!(orch_report.held_entities, Vec::new());
        let (shard_id, shard_report) = &reports[1];
        assert_eq!(*shard_id, B);
        assert_eq!(shard_report.directory, Vec::new());
        assert_eq!(shard_report.held_entities, Vec::new(), "no dots attached");
        assert_eq!(shard_report.applied_inputs, Vec::new());
        assert_eq!(shard_report.pending_entities, Vec::new());
        assert_eq!(
            shard_report.dest_resume_seq, None,
            "no OpenInputSlot honored yet"
        );
        assert_eq!(
            shard_report.input_window_evictions, 0,
            "nothing aged out of the input log"
        );
        // The orchestrator has no StubStats/InputLog — the scrape's None arms.
        assert_eq!(orch_report.dest_resume_seq, None);
        assert_eq!(orch_report.input_window_evictions, 0);

        // Attach one session through a bare gateway endpoint: the dot transits
        // provisional → granted through the REAL directory, and the inspect
        // filters report each state.
        let mut gateway_endpoint = fabric.register(NodeId(50));
        let attach = vd_wire::session_flow::GatewayToShard::AttachSession {
            session: vd_core::SessionId(7),
            fence: vd_core::Fence(1),
            account: vd_core::AccountId(1),
        };
        gateway_endpoint
            .send(
                B,
                vd_sim::io::MsgClass::Control,
                vd_sim::io::bytes(postcard::to_allocvec(&attach).expect("encode")),
            )
            .expect("sent");
        for _ in 0..6 {
            topo.step();
            fabric.ack_survivor(NodeId(50));
        }
        let reports = topo.inspect_all();
        let (_, shard_report) = &reports[1];
        assert_eq!(shard_report.held_entities.len(), 1, "granted and held");
        assert_eq!(shard_report.pending_entities, Vec::new());
        assert_eq!(shard_report.departing_entities, Vec::new());

        // ShardNodes OPT INTO downcasting (the saga-driving scenario hook): the concrete node
        // is reachable via `as_any_mut`, unlike the infra `Liar` that uses the None default.
        let shard = topo.node_mut(B).expect("shard present");
        assert!(
            shard
                .as_any_mut()
                .expect("ShardNode opts into downcasting")
                .downcast_mut::<vd_node::ShardNode<crate::fabric::FabricTransport>>()
                .is_some(),
            "the stub node downcasts to its concrete ShardNode",
        );

        // D-7: a HELD transient appears in `owned_transients` (the `is_held` filter + the map
        // closure); an `Arriving` one does NOT (the uncounted mid-flight tier — the transient twin of
        // the ghost-dot exclusion). Seed both on the shard, then inspect.
        let debris = vd_core::EntityId::pack(vd_core::entity_kind::EntityKind::Debris, 1, 1, 0);
        let arriving = vd_core::EntityId::pack(vd_core::entity_kind::EntityKind::Debris, 1, 2, 0);
        let pose = vd_core::pose::StampedPose::at_rest(
            vd_core::pose::FrameRef::SystemSpace { system_seed: 5 },
            vd_core::glam::DVec3::ZERO,
            vd_core::UniverseTick(1),
        );
        {
            let concrete = topo
                .node_mut(B)
                .expect("shard present")
                .as_any_mut()
                .expect("downcast")
                .downcast_mut::<vd_node::ShardNode<crate::fabric::FabricTransport>>()
                .expect("ShardNode");
            let owned = &mut concrete
                .world_mut()
                .resource_mut::<vd_sim::stub::OwnedTransients>()
                .0;
            owned.insert(
                debris,
                vd_sim::stub::Transient {
                    pose,
                    anchor_fence: vd_core::Fence(7),
                    status: vd_sim::stub::TransientStatus::Held { outbound: None },
                    // Slice 3d: the geometric-trigger prev-offset seed (no boundaries planted here, so
                    // it is never read — seed to the pose offset for the degenerate first segment).
                    prev_offset: pose.pos.offset(),
                },
            );
            owned.insert(
                arriving,
                vd_sim::stub::Transient {
                    pose,
                    anchor_fence: vd_core::Fence(7),
                    status: vd_sim::stub::TransientStatus::Arriving {
                        batch: vd_core::TransferId(1),
                    },
                    prev_offset: pose.pos.offset(),
                },
            );
        }
        // D-7b.3: seed the per-kind handover-loss counter so the inspect scrape's map+sort closures
        // run over a NON-EMPTY map (kind order: Debris=10 before Projectile=12).
        {
            let concrete = topo
                .node_mut(B)
                .expect("shard present")
                .as_any_mut()
                .expect("downcast")
                .downcast_mut::<vd_node::ShardNode<crate::fabric::FabricTransport>>()
                .expect("ShardNode");
            let mut stats = concrete
                .world_mut()
                .resource_mut::<vd_sim::stub::StubStats>();
            stats
                .transients_lost_in_handover
                .insert(vd_core::entity_kind::EntityKind::Projectile, 1);
            stats
                .transients_lost_in_handover
                .insert(vd_core::entity_kind::EntityKind::Debris, 2);
        }
        let reports = topo.inspect_all();
        assert_eq!(
            reports[1].1.owned_transients,
            vec![(debris, vd_core::Fence(7))],
            "only the HELD transient is reported (Arriving is the uncounted mid-flight tier)"
        );
        // D-7b: the same is_held() subset's POSES feed the render-continuity trace (Arriving excluded).
        assert_eq!(
            reports[1].1.held_transient_poses,
            vec![(debris, pose)],
            "only the HELD transient's pose is reported for the render trace"
        );
        // D-7b.3: the per-kind handover loss is reported in kind order (Debris before Projectile).
        assert_eq!(
            reports[1].1.transient_loss,
            vec![
                (vd_core::entity_kind::EntityKind::Debris, 2),
                (vd_core::entity_kind::EntityKind::Projectile, 1),
            ],
            "the handover-loss counter is scraped + sorted by kind"
        );
    }

    /// D-43 #9 (the REQUIRED composition regression): a crossing-triggered TRANSIENT completes its
    /// handoff end-to-end across a REAL 3-node cluster (orch A + source shard B realm FROM + dest shard C
    /// realm TO). Before the fix, `handle_transient_crossing_request` granted the dest but never STARTED a
    /// `BatchHandoff` saga, so the dest's `BatchAdopted` hit `deliver`'s silent None early-return — the
    /// item was stuck `Arriving`, no go-token was written (`orch.batch_goes == []`,
    /// `dest.owned_transients == []`). This proves the silent-early-return is CURED with a REAL dest
    /// reacting to a REAL envelope: the go-token lands, the dest promotes `Arriving→Held`, and the
    /// single-transient-holder oracle passes. The `saga_runtime` rig SIMULATES the dest ack; only a
    /// topology-level run proves the whole choreography self-drives.
    #[test]
    fn a_transient_crossing_composes_end_to_end_dest_owns_and_go_token_recorded() {
        use vd_core::entity_kind::EntityKind;
        use vd_core::geometry::{BoundaryTuning, CrossEffect, RealmBoundary};
        use vd_core::glam::DVec3;
        use vd_core::pose::{FrameRef, LatticePos};
        use vd_node::orchestrator::{OrchestratorConfig, register_orchestrator};
        use vd_sim::directory::DirectoryTuning;
        use vd_sim::stub::{
            OwnedTransients, RealmBoundaries, StubConfig, Transient, TransientStatus,
            register_stub_shard,
        };

        // The two realms the transient crosses: FROM (source shard B's realm) → TO (dest shard C's realm).
        const C: NodeId = NodeId(3);
        let from_realm = RealmId::System(5);
        let to_realm = RealmId::Planet(42);

        let fabric = FaultFabric::new(4343, 2);
        let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());

        // Orchestrator (A) — owns the directory (both shards' realm leases) + the saga runtime.
        let mut orch = vd_node::build_app(
            NodeConfig {
                node_id: A,
                kind: NodeKind::Orchestrator,
            },
            fabric.register(A),
        );
        let (world, schedule) = orch.parts_mut();
        register_orchestrator(
            world,
            schedule,
            &OrchestratorConfig {
                epoch: vd_core::EpochId(1),
                reserve_chunk: 64,
                clock_peers: vec![B, C],
                directory: DirectoryTuning {
                    lease_ttl_ticks: 100,
                    ..DirectoryTuning::default()
                },
                saga: vd_sim::saga::SagaTuning::default(),
                liveness: vd_sim::saga::LivenessTuning::default(),
                roster: std::collections::BTreeMap::new(),
            },
        );
        topo.add_node(Box::new(orch));

        // A stub shard builder (both B and C are the ONE `shard` binary — HR3 — differing only by realm).
        let build_shard = |id: NodeId, realm: RealmId, system_seed: u64| {
            let mut shard = vd_node::build_app(
                NodeConfig {
                    node_id: id,
                    kind: NodeKind::StubShard,
                },
                fabric.register(id),
            );
            let (world, schedule) = shard.parts_mut();
            vd_node::follower::register_clock_follower(world, schedule);
            register_stub_shard(
                world,
                schedule,
                StubConfig {
                    realm,
                    frame: FrameRef::SystemSpace { system_seed },
                    move_speed_mps: 1.0,
                    tick_dt_s: 0.05,
                    orchestrator: A,
                    mint_seed: system_seed,
                    input_log_capacity: 1_000_000,
                    realm_recheck_interval: 0,
                    lease_renew_interval_ticks: 0,
                    self_fence_grace_ticks: 0,
                    snapshot_datagram_budget: 1100,
                    boundary: BoundaryTuning::DEFAULT,
                    request_ttl_ticks: 0,
                },
            );
            shard
        };
        topo.add_node(Box::new(build_shard(B, from_realm, 5)));
        topo.add_node(Box::new(build_shard(C, to_realm, 42)));

        // Let both shards win their realm leases through the REAL directory (the authority gate on the
        // crossing trigger + `emit_transient_batch` requires B to hold FROM and C to hold TO).
        for _ in 0..8 {
            topo.step();
        }
        {
            let reports = topo.inspect_all();
            let b = &reports.iter().find(|(id, _)| *id == B).expect("B").1;
            let c = &reports.iter().find(|(id, _)| *id == C).expect("C").1;
            assert_eq!(
                b.held_realms,
                vec![(from_realm, Fence(1))],
                "source shard B holds its FROM realm lease"
            );
            assert_eq!(
                c.held_realms,
                vec![(to_realm, Fence(1))],
                "dest shard C holds its TO realm lease"
            );
        }

        // Seed a Debris transient dwelling over the B→C boundary on the SOURCE shard: plant an Authority
        // shell boundary on B whose exterior realm is B's (FROM) and whose `to_realm` is C's (TO), and a
        // held Debris a few hundred metres out (comfortably in the create band, < the 1150 m edge). The
        // rising edge fires on the 3rd consecutive in-band tick (`n_entry` DEFAULT). Mirror the shard-level
        // `a_transient_crossing_emits_a_transient_crossing_request` seed.
        let debris = EntityId::pack(EntityKind::Debris, 5, 1, 3);
        {
            let shard = topo
                .node_mut(B)
                .expect("B present")
                .as_any_mut()
                .expect("downcast")
                .downcast_mut::<vd_node::ShardNode<crate::fabric::FabricTransport>>()
                .expect("ShardNode");
            let world = shard.world_mut();
            world.resource_mut::<RealmBoundaries>().0.push(RealmBoundary::shell(
                from_realm,
                LatticePos::local(DVec3::ZERO),
                1000.0, // r_soi
                1.15,   // create_factor → create edge 1150 m
                1.30,   // destroy_factor
                1.0,    // v_rel (slow)
                0.05,   // dt
                0.5,    // pad_floor
                1.0,    // k_safety_extra
                None,   // top-level
                to_realm,
                CrossEffect::Authority,
            ));
            let pose = StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 5 },
                DVec3::new(100.0, 0.0, 0.0),
                vd_core::UniverseTick(1),
            );
            world.resource_mut::<OwnedTransients>().0.insert(
                debris,
                Transient {
                    pose,
                    anchor_fence: Fence(1),
                    status: TransientStatus::Held { outbound: None },
                    prev_offset: pose.pos.offset(),
                },
            );
        }

        // Step long enough for the FULL choreography: rising-edge → TransientCrossingRequest → grant +
        // saga-start (THE fix) → source flips + ships the batch envelope → dest adopts (BatchAdopted) →
        // AwaitRelease/Promote/Complete round-trips → dest promotes Arriving→Held → source retires.
        for _ in 0..40 {
            topo.step();
        }

        let reports = topo.inspect_all();
        let orch = &reports.iter().find(|(id, _)| *id == A).expect("A").1;
        let dest = &reports.iter().find(|(id, _)| *id == C).expect("C").1;

        // THE D-43 #9 symptom, before-vs-after: the go-token was written (was `[]` before the fix).
        assert!(
            !orch.batch_goes.is_empty(),
            "orch.batch_goes is non-empty (was [] before the fix — the silent early-return)"
        );
        // The dest OWNS the transient authoritatively (was stuck Arriving, so `owned_transients == []`).
        assert_eq!(
            dest.owned_transients,
            vec![(debris, Fence(1))],
            "dest C promoted the transient Arriving→Held at its realm-lease fence (was [] before the fix)"
        );
        // The single-transient-holder oracle passes: held by exactly ONE shard, anchored to a live realm
        // fence AND a committed go-token fence (no {source,dest} both-held window).
        crate::oracle::verify_transient_authority_held(&reports)
            .expect("TRANSIENT-AUTHORITY-HELD holds after the composed crossing");
    }

    #[test]
    #[should_panic(expected = "duplicate topology node")]
    fn duplicate_nodes_panic() {
        let fabric = FaultFabric::new(1, 2);
        let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());
        topo.add_node(build_relay_node(&fabric, A, B));
        let mut second = vd_node::build_app(
            NodeConfig {
                node_id: A,
                kind: NodeKind::StubShard,
            },
            fabric.register(NodeId(50)),
        );
        // Force the same reported id by wrapping (node_id comes from the transport,
        // so build a node whose transport id collides).
        let _ = &mut second;
        let fabric2 = FaultFabric::new(2, 2);
        let dup = vd_node::build_app(
            NodeConfig {
                node_id: A,
                kind: NodeKind::StubShard,
            },
            fabric2.register(A),
        );
        topo.add_node(Box::new(dup));
    }

    #[test]
    fn verify_no_reliable_shed_passes_clean_and_catches_a_reliable_staging_shed() {
        // OK arm: a healthy relay pair sends everything — nothing is ever shed.
        let fabric = FaultFabric::new(5, 2);
        let mut clean = seeded_topology(&fabric);
        for _ in 0..20 {
            clean.step();
        }
        clean
            .verify_no_reliable_shed()
            .expect("a healthy run sheds no reliable frame");

        // ERR arm: a node flooding RELIABLE (Saga) frames toward a send-REJECTING peer over a
        // tight staging cap sheds reliable frames every tick — the catastrophic class the audit
        // flagged. The oracle reads the node's own `TickReport.reliable_shed` from the trace
        // (wire-truth) and catches it, naming the offending node.
        let fabric = FaultFabric::new(9, 2);
        fabric.set_policy(
            A,
            B,
            crate::fabric::LinkPolicy {
                send_reject_p: 1.0, // every send to B refuses synchronously ⇒ frames stage
                ..crate::fabric::LinkPolicy::default()
            },
        );
        let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());
        let mut flood = vd_node::build_app(
            NodeConfig {
                node_id: A,
                kind: NodeKind::StubShard,
            },
            fabric.register(A),
        );
        flood.world_mut().insert_resource(OutboundStagingCap(1)); // tiny cap ⇒ shed over 1
        flood
            .schedule_mut()
            .add_systems(move |mut outbox: ResMut<OutboundBox>| {
                for n in 0..4u8 {
                    outbox.0.push((
                        B,
                        MsgClass::Saga,
                        vec![n].into(),
                        vd_sim::io::Durability::Ephemeral,
                    ));
                }
            });
        topo.add_node(Box::new(flood));
        topo.add_node(build_relay_node(&fabric, B, A)); // a sink (receives nothing under the reject)
        for _ in 0..3 {
            topo.step();
        }
        let violation = topo
            .verify_no_reliable_shed()
            .expect_err("the reliable staging shed must be caught");
        assert_eq!(violation.node, A);
        assert!(
            violation.reliable_shed > 0,
            "a reliable frame was shed and surfaced distinctly: {violation:?}",
        );
    }
}
