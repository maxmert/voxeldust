//! THE LIVE-SAGA SET: every transfer in flight, and the queues feeding it.
//!
//! Owns: the resource the schedule single-writes — the live sagas by id, the create-on-trigger
//! queue, the standing re-home queue, the rejection ledger with its hard ceiling, and the batched
//! go-tokens one tick produced. Plus the arrival classifier that answers which realm a saga is
//! currently delivering INTO, which is what lets a destination realm be held alive while somebody is
//! on the way to it.
//!
//! Does NOT own: the FSM. The saga itself is a pure `fn step(State, Event) -> (State, Vec<Action>)`
//! in `vd_sim::saga`; everything here is the wrapper that routes its actions and remembers its
//! place.

use super::LivenessTracker;
use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use vd_core::pose::{RealmId, StampedPose};
use vd_core::{BatchId, Fence, NodeId, TransferId, UniverseTick};
use vd_sim::capability::ShardProfile;
use vd_sim::io::Bytes;
use vd_sim::saga::{AbortReason, LivenessTuning, SagaCtx, SagaState, SagaTuning};
use vd_wire::seams::directory::DirectoryKey;

/// One live (in-flight) transfer's identity for the mid-flight AUTHORITY-UNIQUE oracle (1d.5b.3d):
/// the subject key + the source/dest shards. The oracle EXCUSES the post-CAS, pre-demote window (the
/// source still holds the subject `Owned` at the old fence while the directory already records the
/// dest) ONLY for a subject with a live saga of this exact (source→dest) shape — so a real split-brain
/// is never masked. Typed `DirectoryKey` (NOT the `String` `SagaView`) so the oracle matches Entity keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveTransfer {
    pub subject: DirectoryKey,
    pub source: NodeId,
    pub dest: NodeId,
}

/// One live saga: the pure FSM (ctx + state) plus the wrapper-only routing — the gateway
/// `NodeId` to send `TransferControl` to (resolved at creation from the session's
/// directory record; not the FSM's concern).
pub(crate) struct LiveSaga {
    pub(crate) ctx: SagaCtx,
    pub(crate) state: SagaState,
    pub(crate) gateway: NodeId,
    /// The universe tick of the most recent transition that advanced this saga (set
    /// at creation, refreshed on every write-back). A PARKED saga receives no events,
    /// so `commit_result` is never re-entered for it and `since` stays put — which is
    /// exactly when it entered its current (stuck) state. The admin view reports
    /// `now - since` as staleness so a 2am operator sees how long a saga has been wedged.
    pub(crate) since: UniverseTick,
    /// The source's authoritative pose, stashed when its `SourceFlushed` reply arrives (1d.1) and
    /// read by the `EmitCrossing` executor at/after commit. The pose-before-promote gate guarantees
    /// it is `Some` before the CAS for an Entity subject, so `EmitCrossing` never ships a None pose.
    /// (The TLV state blob joins this at 1d.6; pose-only now.)
    pub(crate) flushed_pose: Option<StampedPose>,
    /// D-3 — WHICH participant was first observed CONFIRMED-DEAD and WHEN, while this saga awaits a
    /// destructive resolution (`None` otherwise). A DESTRUCTIVE resolution (dest-abandon, source
    /// discard/self-promote, dest re-home) waits the LARGE `abort_deadline_ticks` measured FROM the tick
    /// here — NOT from `since` (which `scan_deadlines` re-arms to `now` on every fire, making a `now -
    /// since >= abort` gate structurally unsatisfiable). R-6d3c keys it by `NodeId`: a CAUSE-SWITCH (dest
    /// confirmed dead → the dest RECOVERS via `record_ack` → the SOURCE is confirmed dead) RE-ANCHORS to
    /// `now` (`dead_budget_elapsed`), so the new participant's budget is never measured from the OTHER
    /// participant's stale first-dead observation. RAM-ONLY (NOT persisted in `SagaSnapshot`): on rehydrate
    /// it re-derives `None`, so a restarted orchestrator re-accrues the abort budget from scratch rather
    /// than firing an irreversible resolution immediately.
    pub(crate) dead_observed_since: Option<(NodeId, UniverseTick)>,
    /// CA-1 S3/S4 — the OVER-DISCARD SAFETY latch. Monotone `true` once the orchestrator has ANY evidence
    /// the dest adopted this batch (a `BatchAdopted` seen in the inbox). `rehome_event_for` fires the
    /// destructive `SourceUnreachablePreAdopt` discard ONLY when this is `false` — so the discard is a
    /// genuine "the dest never got the batch" loss, never a "the ack is lost/in-flight" false loss. Set in
    /// the PRE-SCAN inbox pass (`latch_adopted_from_inbox`) BEFORE `scan_deadlines`, so a `BatchAdopted`
    /// arriving on the exact budget-maturity tick suppresses the discard THAT tick (latching in `deliver`,
    /// which runs in the post-scan drain, would be one tick too late). RAM-ONLY (NOT in `SagaSnapshot`): re-
    /// derives `false` on rehydrate — safe, because the liveness tracker is EMPTY on rehydrate (the D-6
    /// freeze), so no discard can fire until fresh notices re-accrue, and the dest's per-tick `BatchAdopted`
    /// re-drive re-establishes the latch long before then.
    pub(crate) dest_adopted: bool,
    /// The universe tick this saga was INSERTED — the age anchor for the arrival shield's leak bound.
    /// Deliberately NOT [`LiveSaga::since`], which `scan_deadlines` re-arms to `now` on every re-drive:
    /// anchoring the cap there would refresh the shield forever on exactly the wedged saga the cap exists
    /// to bound. RAM-ONLY (not in [`SagaSnapshot`]) — a rehydrated saga re-derives it to the recovered
    /// clock, the established precedent of `dead_observed_since`/`dest_adopted`, so a restart grants a
    /// fresh budget rather than instantly expiring a recovered arrival.
    pub(crate) opened: UniverseTick,
}

/// One hand-off whose arrival shield ran out of budget — reported so the lift is LOUD. Carries what a
/// 2am operator needs to find it: which transfer, which realm it was holding, what phase it is stuck in,
/// and for how long.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpiredArrival {
    pub transfer: TransferId,
    pub realm: RealmId,
    pub state: String,
    pub age_ticks: u64,
}

/// THE ARRIVAL CLASSIFIER: the realm this saga is currently delivering its subject INTO, or `None` if
/// nobody is on their way anywhere because of it. EXHAUSTIVE with no wildcard — a new saga shape does
/// not compile until someone decides which side of this line it falls on.
///
/// `None` for the three terminal/compensating shapes (a compensating saga sends the subject BACK to the
/// source, so the shield must drop immediately rather than after a timeout), and `None` for the two
/// RE-HOME shapes. A re-home is a recovery hand-off between MACHINES, not a move between PLACES: the
/// standing re-home fabricates its realm fields ([`rehome_ctx`]'s `PLACEHOLDERS` — an entity-derived
/// `System(entity)` that names no real realm) and parks indefinitely by design, so including it would
/// pin one garbage realm alive forever per orphaned entity on the first machine failure; and a crossing
/// that DEGRADED into a re-home is re-targeting away from a destination whose node is already confirmed
/// dead, where the zombie force-reap must be allowed to run.
pub(crate) fn arrival_dest(live: &LiveSaga) -> Option<RealmId> {
    match live.state {
        SagaState::Aborting { .. } | SagaState::Aborted { .. } | SagaState::Done { .. } => None,
        SagaState::ReHoming { .. }
        | SagaState::Promoting {
            rehome_target: Some(_),
            ..
        } => None,
        SagaState::AwaitProvision
        | SagaState::Preparing
        | SagaState::Cutting
        | SagaState::Freezing { .. }
        | SagaState::CommittingCas { .. }
        | SagaState::BatchCommitting { .. }
        | SagaState::BatchHandoff { .. }
        | SagaState::Swapping { .. }
        | SagaState::Demoting { .. }
        | SagaState::Promoting {
            rehome_target: None,
            ..
        }
        | SagaState::Releasing { .. } => Some(live.ctx.to_realm),
    }
}

/// A pending create-on-trigger. Enqueued by [`SagaRuntimeRes::start_transfer`] and processed
/// next tick (a stub-shard boundary-crossing request wires the real trigger at Slice 1d).
pub(crate) struct PendingStart {
    pub(crate) ctx: SagaCtx,
    pub(crate) gateway: NodeId,
}

/// D-37 Slice 3: a STANDING re-home detected by the expiry reaper — an `Entity` key whose committed
/// owner is confirmed-dead, lease lapsed, and key UNLOCKED (no live saga). Enqueued by
/// [`reap_lapsed_leases`] and drained the SAME tick by [`process_rehome_starts`] (a within-barrier
/// hand-off, RAM-only). It is NOT separately WAL'd: the durable artifacts are the locked directory
/// record + the armed re-home saga (both persisted in the same group-commit barrier); a kill-9 in the
/// enqueue→drain window loses only this RAM list, and on reboot the reaper re-detects the still-dead,
/// still-UNLOCKED record (the directory IS the durable trigger) and re-enqueues — idempotent recovery
/// with no new WAL family. Carries only REAL recoverable values (no fabricated pose — HR1: the dead
/// owner's store is sealed; the dot reconstruction is owed Slice 4/D-6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PendingReHome {
    /// The orphaned `Entity` directory key to recover.
    pub(crate) subject: DirectoryKey,
    /// The confirmed-dead committed owner (the re-home `source`/provenance; the CAS expectation owner).
    pub(crate) dead_owner: NodeId,
    /// The fence the dead owner was committed at — Slice 4's `ReHomeCommit` CAS expectation (`fence+1`).
    pub(crate) prev_fence: Fence,
}

/// Slice 3f-D (Mechanism Y): a durable orchestrator record that a CROSSING-ORIGIN durable saga tombstoned
/// ABORTED (a pre-CAS failure), so the SOURCE's `RequestInFlight` crossing latch must be POSITIVELY cleared
/// — a lost RAM enqueue would otherwise strand the entity's latch forever. Held in
/// [`SagaRuntimeRes::pending_abort_replies`] (keyed by the aborted `TransferId`) AND persisted
/// (`StoreKey::AbortReply`), so an orchestrator restart BETWEEN the abort and the source's ack re-emits
/// `CrossingAborted` via rehydrate → the latch still clears (a RAM-only map would have lost it on the kill).
/// Dropped by the source's `CrossingAbortedAck`, or SELF-REAPED when `source` is confirmed dead (D-37
/// already re-homed the entity → the abort is moot). Self-describing (carries `transfer`), so rehydrate
/// never parses the store key. NO generation/attempt field is needed: the 3f-D1 attempt-stamp already
/// makes each crossing ATTEMPT's id UNIQUE (`crossing_transfer_id(subject, subject_fence, attempt)`), so
/// there is no ABA on the `TransferId` key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PendingAbortReply {
    /// The aborted crossing's `TransferId` — the map key AND the id the source latch keys on.
    pub transfer: TransferId,
    /// The SOURCE shard to re-emit `CrossingAborted` to (and whose death triggers the moot-abort reap).
    pub source: NodeId,
    /// The subject key echoed back in `CrossingAborted` (the source's latch is entity-keyed).
    pub subject: DirectoryKey,
}

/// The live-saga set + the trigger queue, resource-wrapped (single writer: this schedule).
#[derive(Resource, Default)]
pub struct SagaRuntimeRes {
    pub(crate) sagas: BTreeMap<TransferId, LiveSaga>,
    pub(crate) pending: Vec<PendingStart>,
    /// D-37 Slice 3: standing re-homes the reaper detected THIS sweep, drained the same tick by
    /// [`process_rehome_starts`]. RAM-only (a within-barrier hand-off): see [`PendingReHome`] for why
    /// no WAL family is needed (the locked record + armed saga are the durable artifacts; the reaper
    /// re-detects on reboot). Always empty BETWEEN ticks (drained at the end of every barrier).
    pub(crate) pending_rehome: Vec<PendingReHome>,
    /// Typed client-facing rejections awaiting the surfacing system. BOUNDED ring
    /// (`REJECTION_LEDGER_CAP`): the PROPER consumer — the typed rejection→client channel —
    /// lands at Slice 1c.9 (CPO-4) and drains via `std::mem::take` (lifetime ONE cycle,
    /// matching `pending`). Until then this is capped so it can NEVER grow without limit on
    /// the orchestrator at MMO scale (audit ROB-1): on overflow the OLDEST rejections are
    /// shed with a counted + warned drop (`rejections_dropped`) — never an unbounded leak,
    /// never silent.
    pub rejected: Vec<(TransferId, AbortReason)>,
    /// Rejections shed because `rejected` reached `REJECTION_LEDGER_CAP` before the 1c.9
    /// consumer drained it — the loud overflow ALERT (0 in any healthy run).
    pub rejections_dropped: u64,
    /// The deadline budget the `scan_deadlines` producer reads (Slice 2a). Set from
    /// `OrchestratorConfig.saga` in `register_orchestrator`; `Default` is the dev/test value
    /// ([`SagaTuning::default`]) for the in-process rigs.
    pub(crate) tuning: SagaTuning,
    /// THE batched `TransientGo` go-token ledger (HR2/D-7): one record per `BatchId` =
    /// `(commit_fence, source, dest)`. Written by the `IssueTransientGo` executor at the transient
    /// SHORT-PATH commit point — ONE write per batch regardless of item count (G-TIER); the
    /// `BatchAdopted`/`DropApplied` handlers read it to drive the D-7b structural drop-before-promote
    /// (release the source, gate the dest promote, then ReleaseComplete the source). The
    /// `TRANSIENT-AUTHORITY-HELD` oracle cross-checks every Held transient's set
    /// anchor against a committed go-token here — a transient is NEVER a directory `OwnerRecord`
    /// (burst-isolation: a 1000-debris burst writes ZERO directory rows). ⚠️ IN-MEMORY (durability
    /// owed D-6). ⚠️ SAME UNBOUNDED-LEDGER CLASS as `rejected` (audit ROB-1) but NOT yet bounded:
    /// one `(Fence, NodeId, NodeId)` entry per batch EVER committed accumulates for orchestrator
    /// uptime (worst case = total batches committed since start) — UNLIKE `rejected`, this cannot be
    /// a cap-and-shed ring because the `TRANSIENT-AUTHORITY-HELD` oracle needs the live record while
    /// any item is Held. The proper GC fires at the shard's terminal retire (`on_release_complete`),
    /// gated on the drop-completion signal D-7d co-designs (owed D-7d, NOT a timeout). Until then do
    /// not run a long-lived burst soak without monitoring orchestrator RSS. `or_insert` idempotent:
    /// a Slice-2a re-driven go-token re-records the SAME (batch → fence) without duplication.
    pub(crate) batch_goes: BTreeMap<BatchId, (Fence, NodeId, NodeId)>,
    /// THE G-TIER observable (D-7c): a MONOTONIC count of go-token WRITES (incremented once per
    /// emitted `BatchGo`, BEFORE the idempotent `or_insert`). `batch_goes.len()` alone is FALSE-GREEN
    /// for the "write rate scales with BATCH count, not ITEM count" claim — the `or_insert` collapses N
    /// same-key writes to one entry, so a per-item-write regression would leave `len` at 1 yet inflate
    /// THIS counter to N. The G-TIER gate asserts `batch_go_writes == distinct-batch-count` (== 1 for a
    /// 1000-item single batch). The metric is forward-compatible with [[D-6]]'s durable WAL: one
    /// in-memory write per batch is exactly one fsync per batch.
    pub(crate) batch_go_writes: u64,
    /// D-3 the dead-vs-slow discriminator (CSCALE-1 cure): evidence-gated + clearable (was the insert-only
    /// `dead_participants` set). Fed by the `Inbound::NodeUnreachable` arm (`record_unreachable`) + cleared
    /// by ANY successful inbound (`record_ack`); `scan_deadlines` reads `is_confirmed_dead` to resolve a
    /// `BatchHandoff` saga whose source/dest is dead. A single recoverable blip no longer confirms a HEALTHY
    /// peer dead (needs `n_consecutive_unreachable` within the window), and the DESTRUCTIVE dest-abandon is
    /// further gated behind the LARGE `abort_deadline_ticks` from the per-saga `dead_observed_since`.
    pub(crate) liveness: LivenessTracker,
    /// D-7d anti-vacuity observables (orchestrator-side: the dest cannot tell a self-promote from a
    /// normal promote, so the resolution is counted HERE where it is decided). `source_unreachable`
    /// counts dead-SOURCE self-promote resolutions (BatchCommittedAt, zero loss); `dest_unreachable`
    /// counts dead-DEST abandon resolutions (BatchDroppedWithinBudget). Both 0 on every healthy run —
    /// `> 0` is the proof a crash cell's resolution actually fired (not that the happy path completed).
    pub(crate) source_unreachable_resolutions: u64,
    pub(crate) dest_unreachable_resolutions: u64,
    /// R-6d3c anti-vacuity observable: dead-SOURCE PRE-adopt resolutions (`AwaitAdopt` +
    /// `SourceUnreachablePreAdopt`) — the batch is counted lost (discard-to-dest, NOT self-promote).
    /// Distinct from `source_unreachable_resolutions` (post-adopt, zero-loss self-promote) so the two
    /// dead-source causes are never conflated in the ledger. 0 on every healthy run and on a RESTART
    /// recovery; `> 0` is the never-restart cell's proof its resolution fired.
    pub(crate) batch_lost_source_crash: u64,
    /// D-6 — durable writes STAGED this tick by `commit_result` (saga snapshots + go-tokens, encoded at
    /// the monomorphic call site), drained to the [`StoreRes`] + group-committed ONCE at the end of
    /// `drive_sagas` (the ~1-fsync/tick barrier). `Some(bytes)` = put, `None` = delete (a tombstoned
    /// saga). Held on the runtime (not threaded through `commit_result`) so the persist stays a
    /// straight-line push; the single drain+commit is the only place that touches the store for sagas.
    pub(crate) pending_writes: Vec<(Vec<u8>, Option<Bytes>)>,
    /// D-3 anti-vacuity observable: total `NodeUnreachable` notices fed to the liveness tracker. A
    /// CSCALE-1 flap cell asserts this is `> 0` (the blip was genuinely observed — the path the OLD
    /// insert-only code would have abandoned on) AND that `dest_unreachable_resolutions == 0` (the cure:
    /// a recoverable blip toward a healthy dest no longer abandons the batch). 0 on a no-fault run.
    pub(crate) liveness_notices: u64,
    /// R-4d M3 orchestrator-side shed observable: total `Inbound::SendShed` notices seen. A shed is a
    /// LOCAL transport refusal (a lane hit its retry-buffer byte cap, or an oversize frame) — it says
    /// NOTHING about the peer's liveness, so it is counted here and NEVER fed to `record_unreachable`
    /// (the false-confirm cure). 0 on a healthy run; `> 0` cross-references the mesh `reliable_shed`
    /// counter (an ALERT: a saturated retry buffer / a mis-sized frame).
    pub(crate) sends_shed: u64,
    /// D-3 Slice 4: the universe tick the expiry REAPER last swept, re-armed on fire so the O(directory)
    /// sweep runs at most once per `reaper_interval_ticks` (never per tick — like `scan_deadlines`' `since`).
    pub(crate) last_reap_tick: UniverseTick,
    /// D-3 Slice 4 CAP freeze: the reaper does NOT act before this tick. Set on rehydrate to
    /// `now + recovery_grace_ticks` — a FIXED post-restart freeze (belt-and-suspenders atop the RAM-empty
    /// tracker, which is the PRIMARY freeze: an empty tracker confirms nobody dead until fresh notices
    /// re-accrue). 0 at genesis (a fresh orchestrator has no stale leases to reap, so no freeze needed).
    pub(crate) liveness_quiesced_until: UniverseTick,
    /// D-37 forward re-home target roster: `NodeId → ShardProfile` for the shards a re-home may land on,
    /// set from `OrchestratorConfig.roster` (the cluster builder maps each stub shard to the empty
    /// profile). RAM-only operational config (rebuilt on rehydrate, like `liveness`/`clock_peers`).
    /// `select_rehome_target` reads it for the lowest LIVE capability-matched shard; an EMPTY roster
    /// (`Default`) ⇒ no target ⇒ the saga stays parked (honest), which every legacy rig expects.
    pub(crate) roster: BTreeMap<NodeId, ShardProfile>,
    /// Slice 3f-B: durable `CrossingRequest`s that RESOLVED (subject + dest-realm + session all found in
    /// the directory) and STARTED a crossing saga. A monotonic count; a redelivered request for an
    /// already-live crossing is absorbed by `process_starts`' `contains_key`/`lock_transfer` guard and
    /// does NOT re-increment (the count reflects distinct started sagas, not requests seen). 0 until the
    /// first durable boundary crossing lands.
    pub(crate) crossings_started: u64,
    /// Slice 3f-B: durable `CrossingRequest`s whose SUBJECT owner was found but whose dest-`Realm` OR
    /// `Session` head was UNRESOLVED, so no saga could start this tick. Counted-only for now — the
    /// abort-reply egress that clears the source latch is a LATER sub-slice (3f-D); until then a lost
    /// crossing here is re-driven by the source's ongoing boundary detection (the `ReDriven` class).
    pub(crate) crossing_unresolved: u64,
    /// Slice 3f-B: durable `CrossingRequest`s whose SUBJECT had no directory `OwnerRecord` at all (the
    /// entity's authority already moved / was revoked between the source's latch and this resolve). No
    /// owner to reply to ⇒ a counted drop.
    pub(crate) crossing_subject_gone: u64,
    /// Slice 3f-C: transient `TransientCrossingRequest`s whose dest `Realm` head RESOLVED, so a
    /// `TransientCrossingGrant` was emitted back to the transport-origin. A redelivered request re-grants
    /// the SAME deterministic `batch` id (idempotent at the source), so this counts grants EMITTED.
    pub(crate) transient_crossings_granted: u64,
    /// Slice 3f-C: transient `TransientCrossingRequest`s whose dest `Realm` head was UNRESOLVED — no grant
    /// emitted (the source re-requests; the grant is idempotent, so a later-resolving realm still grants).
    pub(crate) transient_dest_unresolved: u64,
    /// Slice 3f-D (Mechanism Y): crossing-origin durable-abort replies pending a source ack, keyed by the
    /// aborted `TransferId`. DURABLE (persisted via `StoreKey::AbortReply` + rehydrated) — the whole point is
    /// crash-durability of the latch-clear obligation, so an orchestrator restart between the abort and the
    /// source's ack re-emits `CrossingAborted` and the source latch still clears. `scan_deadlines` re-emits
    /// per entry (throttled to `redrive_deadline_ticks`) until a `CrossingAbortedAck` drops it (or a
    /// dead-source reap). Empty on `Default`. See [`PendingAbortReply`].
    pub(crate) pending_abort_replies: BTreeMap<TransferId, PendingAbortReply>,
    /// Slice 3f-D: the universe tick the abort-reply RE-EMIT pass last fired (like `scan_deadlines`' per-saga
    /// `since`, but ONE runtime-level cadence gate for the whole pending set) — so the re-emit runs at most
    /// once per `redrive_deadline_ticks`, never a per-tick storm against an alive-but-ack-stalled source. RAM
    /// throttle only (NOT persisted): a rebuilt orchestrator re-emits once immediately on its first post-restart
    /// scan, which is exactly the harmless prompt behaviour rehydrate wants (the source re-acks). 0 at genesis.
    pub(crate) last_abort_reply_emit: UniverseTick,
}

/// One batched `TransientGo` go-token emitted by the `IssueTransientGo` executor, COLLECTED by
/// `run_to_quiescence` (which is deliberately denied a `runtime` handle — `commit_result`'s
/// write-back soundness rests on it) and recorded into [`SagaRuntimeRes::batch_goes`] by
/// `commit_result`. `batch` is the `BatchId` (the transient saga's transfer); `fence` is the dest
/// realm-lease fence the batch committed at; `source`/`dest` are the shards the D-7b handoff commands
/// (`TransientRelease`/`TransientDrop`/`ReleaseComplete`) target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BatchGo {
    pub(crate) batch: BatchId,
    pub(crate) fence: Fence,
    pub(crate) source: NodeId,
    pub(crate) dest: NodeId,
}

/// Hard ceiling on the un-drained rejection ledger (operational param, ONE home — no inline
/// literal). Sized to absorb a large abort burst between 1c.9 drain cycles; beyond it the
/// oldest rejections are shed (counted) rather than grow the orchestrator without bound.
pub(crate) const REJECTION_LEDGER_CAP: usize = 1024;

impl SagaRuntimeRes {
    /// Whether `node` has been LATCHED dead by the D-3 liveness tracker (RLM Step 3e: the reconciler's
    /// BUG-A zombie check reads this to tell a live head from a dead-but-recorded one). Delegates to the
    /// private liveness latch — the ONE authority on confirmed-dead nodes (`should_reap` reads the same).
    #[must_use]
    pub fn is_node_latched_dead(&self, node: NodeId) -> bool {
        self.liveness.is_latched_dead(node)
    }

    /// Construct with the deadline budget from config (Slice 2a). The production path
    /// (`register_orchestrator`) calls this with `cfg.saga`; `Default` is the dev/test value.
    #[must_use]
    pub fn with_tuning(tuning: SagaTuning) -> SagaRuntimeRes {
        // The dev/test path: the kill-equivalent liveness default (n == 1) — a permanent kill's first
        // NodeUnreachable confirms, so the existing crash cells keep their behavior.
        Self::with_tunings(tuning, LivenessTuning::default())
    }

    /// The production constructor (`register_orchestrator`): the saga deadline budget + the D-3 dead-vs-slow
    /// confirmation tuning, both from `OrchestratorConfig`. Prod sets `n_consecutive_unreachable >= 3` (the
    /// CSCALE-1 margin) so a single recoverable blip never confirms a healthy peer dead.
    #[must_use]
    pub fn with_tunings(tuning: SagaTuning, liveness: LivenessTuning) -> SagaRuntimeRes {
        SagaRuntimeRes {
            tuning,
            liveness: LivenessTracker::new(liveness),
            ..SagaRuntimeRes::default()
        }
    }

    /// Re-tune the dead-vs-slow discriminator after construction — for a test scenario that needs a
    /// specific confirmation margin (the CSCALE-1 flap cells run `n_consecutive_unreachable == 3` so a
    /// short blip stays below the threshold). Prod configures it via [`with_tunings`] / `OrchestratorConfig`.
    pub fn set_liveness_tuning(&mut self, liveness: LivenessTuning) {
        self.liveness.set_tuning(liveness);
    }

    /// Seed the D-37 re-home target roster (set in `register_orchestrator` from `OrchestratorConfig.roster`,
    /// re-seeded on rehydrate). A test rig calls this to make a re-home target available; the default
    /// (empty) roster makes `select_rehome_target` return `None` (the saga parks — honest).
    pub fn set_roster(&mut self, roster: BTreeMap<NodeId, ShardProfile>) {
        self.roster = roster;
    }

    /// Trigger a new transfer (create-on-trigger). `ctx.subject` MUST already be a directory
    /// record — [`drive_sagas`] asserts `lock_transfer` succeeds (a missing/already-locked
    /// subject refuses the start). `gateway` is the session's gateway (the caller resolves it
    /// from the `Session` directory record).
    ///
    /// 1c CONTRACT (audit CPO-4): a lock refusal is currently a silent drop (acceptable while
    /// the only caller is a test rig). When the 1c rejection-surfacing channel lands, refusals
    /// MUST route through that SAME channel with a typed reason (SubjectLocked /
    /// SubjectUnrecorded) so a shard-initiated trigger (1d) always learns its fate.
    pub fn start_transfer(&mut self, ctx: SagaCtx, gateway: NodeId) {
        self.pending.push(PendingStart { ctx, gateway });
    }

    /// Live saga count (the oracle/admin surface; bounded — terminal sagas are tombstoned).
    #[must_use]
    pub fn live(&self) -> usize {
        self.sagas.len()
    }

    /// THE ARRIVAL SET — every realm a live hand-off is currently delivering someone INTO, so the
    /// lifecycle reconciler can refuse to shut down a place somebody is on their way to.
    ///
    /// DERIVED fresh from the live sagas on every call, never stored. That is the whole safety argument:
    /// there is no clearing path to forget, no eviction rule, and no second source of truth that could
    /// disagree. When `commit_result` tombstones a saga the realm leaves this set on the same sweep, for
    /// free; when a crossing aborts, [`arrival_dest`] classifies it out immediately rather than after a
    /// timeout; and because the saga snapshot persists its whole context, the set re-forms on the first
    /// sweep after an orchestrator restart. The fact and the thing that needs the fact are the same
    /// object, in the same process, on the same tick — there is no message to lose and no race to win.
    /// `max_age` is the leak bound: a single hand-off may hold its destination for that many ticks and
    /// no longer. `0` DISARMS the shield entirely (the inert default). Expiry is evaluated PER HAND-OFF,
    /// not per realm, so one wedged crossing cannot drag down every other arrival at a busy destination.
    /// Returns the shielded realms plus every hand-off whose budget ran out, so the caller can say so
    /// loudly rather than dropping a player's protection in silence.
    #[must_use]
    pub fn arriving_dest_realms(
        &self,
        now: UniverseTick,
        max_age: u64,
    ) -> (BTreeSet<RealmId>, Vec<ExpiredArrival>) {
        let mut shielded = BTreeSet::new();
        let mut expired = Vec::new();
        if max_age == 0 {
            return (shielded, expired); // disarmed
        }
        for (transfer, live) in &self.sagas {
            let Some(realm) = arrival_dest(live) else {
                continue;
            };
            let age = now.0.saturating_sub(live.opened.0);
            if age > max_age {
                expired.push(ExpiredArrival {
                    transfer: *transfer,
                    realm,
                    state: format!("{:?}", live.state),
                    age_ticks: age,
                });
            } else {
                shielded.insert(realm);
            }
        }
        (shielded, expired)
    }

    /// Operator-facing view of every live (in-flight) saga — the admin snapshot's
    /// `sagas` list, the 2am `curl` that AAA-1 promised. Bounded: terminal sagas are
    /// tombstoned, so this is the in-flight set only, in deterministic `TransferId`
    /// order (BTreeMap). `SagaState` is `Debug`-only and lives in `vd-sim`, so the
    /// operator string is its `{:?}` (wire cannot depend on sim — admin.rs §SagaView).
    #[must_use]
    pub fn views(&self) -> Vec<vd_wire::admin::SagaView> {
        self.sagas
            .values()
            .map(|live| vd_wire::admin::SagaView {
                transfer: live.ctx.transfer.to_string(),
                state: format!("{:?}", live.state),
                since: live.since,
            })
            .collect()
    }

    /// The subject + source/dest of every LIVE (in-flight) saga — the typed ground truth the
    /// mid-flight AUTHORITY-UNIQUE oracle (1d.5b.3d) keys its W1 transfer-window excuse on. Empty once
    /// every saga is tombstoned, so the POST-QUIESCE oracle is strict (no transfer ⇒ no excuse). In
    /// deterministic `TransferId` order (BTreeMap). Orchestrator-only state — a shard never holds it.
    #[must_use]
    pub fn active_transfers(&self) -> Vec<ActiveTransfer> {
        self.sagas
            .values()
            .map(|live| ActiveTransfer {
                subject: live.ctx.subject,
                source: live.ctx.source,
                dest: live.ctx.dest,
            })
            .collect()
    }

    /// The committed batched-transient go-tokens — the `(BatchId, commit_fence)` ground truth the
    /// `TRANSIENT-AUTHORITY-HELD` oracle cross-checks every Held transient against (D-7). In
    /// deterministic `BatchId` order (BTreeMap). Orchestrator-only state (a shard never holds it).
    #[must_use]
    pub fn batch_goes(&self) -> Vec<(BatchId, Fence)> {
        self.batch_goes
            .iter()
            .map(|(batch, (fence, _src, _dst))| (*batch, *fence))
            .collect()
    }

    /// The MONOTONIC go-token write count (D-7c G-TIER observable): how many `BatchGo`s have been
    /// recorded, regardless of how many collapsed onto the same `BatchId` via `or_insert`. The gate
    /// asserts this equals the distinct-batch count (NOT the item count) — the proof the orchestrator
    /// write rate scales with batch count, never burst size.
    #[must_use]
    pub fn batch_go_writes(&self) -> u64 {
        self.batch_go_writes
    }

    /// D-7d — the count of dead-SOURCE self-promote resolutions (the dest was self-promoted from the
    /// go-token after the source died mid-handoff). 0 on every healthy run; `> 0` proves a SOURCE-kill
    /// cell's resolution fired (anti-vacuity for `BatchCommittedAt`).
    #[must_use]
    pub fn source_unreachable_resolutions(&self) -> u64 {
        self.source_unreachable_resolutions
    }

    /// D-7d — the count of dead-DEST abandon resolutions (the source's retained copy was dropped as an
    /// accounted loss after the dest died mid-handoff). 0 on every healthy run; `> 0` proves a DEST-kill
    /// cell's resolution fired (anti-vacuity for `BatchDroppedWithinBudget`).
    #[must_use]
    pub fn dest_unreachable_resolutions(&self) -> u64 {
        self.dest_unreachable_resolutions
    }

    /// R-6d3c — the count of dead-SOURCE PRE-adopt resolutions (`AwaitAdopt` +
    /// `SourceUnreachablePreAdopt`): the batch was counted lost + discarded-to-dest (NOT self-promoted).
    /// 0 on every healthy run and on a RESTART recovery; `> 0` proves the never-restart cell's resolution
    /// fired (anti-vacuity for the accounted-loss-on-detection).
    #[must_use]
    pub fn batch_lost_source_crash(&self) -> u64 {
        self.batch_lost_source_crash
    }

    /// D-3 anti-vacuity: total `NodeUnreachable` notices observed (the CSCALE-1 flap cell asserts `> 0`,
    /// so the blip genuinely exercised the path the old insert-only code would have abandoned on).
    #[must_use]
    pub fn liveness_notices(&self) -> u64 {
        self.liveness_notices
    }

    /// R-4d M3 orchestrator-side shed observable: total `Inbound::SendShed` notices seen. 0 on a healthy
    /// run; `> 0` = a lane hit its retry cap or an oversize frame was refused (cross-reference the mesh
    /// `reliable_shed` counter). A shed NEVER accrues to the liveness tracker — the regression proof is
    /// that a burst of sheds toward a live peer leaves it un-confirmed-dead.
    #[must_use]
    pub fn sends_shed(&self) -> u64 {
        self.sends_shed
    }

    /// Slice 3f-B — durable crossing sagas STARTED from a resolved `CrossingRequest` (subject, dest-realm,
    /// and session all in the directory). Monotonic; a redelivered request for a live crossing does not
    /// re-count (the `contains_key`/`lock_transfer` guard absorbs it).
    #[must_use]
    pub fn crossings_started(&self) -> u64 {
        self.crossings_started
    }

    /// Slice 3f-D (Mechanism Y) — the number of durable crossing-abort replies STAGED and awaiting a source
    /// `CrossingAbortedAck` (the persisted latch-clear obligations). Bumps at the pre-CAS abort tombstone
    /// (`commit_result`), drops when the source acks (or a dead-source ownership reap in `scan_deadlines`).
    /// The 3g abort/crash-leg observable: `>= 1` mid-flight proves the Mechanism-Y path ran (a non-crossing
    /// abort stages NOTHING); `== 0` after the ack proves the round-trip reaped it; restored from the WAL by
    /// `rehydrate` so a `>= 1` immediately post-restart proves the entry rode the store, not RAM survival.
    #[must_use]
    pub fn pending_abort_replies_len(&self) -> usize {
        self.pending_abort_replies.len()
    }

    /// Slice 3f-B — durable `CrossingRequest`s the subject was known for but the dest-realm OR session head
    /// was unresolved (no saga started this tick). The drop is source-healed (D-WORLD-2): the source's
    /// armed ttl re-drive re-presents the SAME request a budgeted number of times, then aborts LOCALLY
    /// (its latch clears; the entity's next crossing fires) — so a persistent count here is an
    /// unhosted/unresolvable realm being retried, never a permanent strand.
    #[must_use]
    pub fn crossing_unresolved(&self) -> u64 {
        self.crossing_unresolved
    }

    /// Slice 3f-B — durable `CrossingRequest`s whose subject had no directory owner (authority already
    /// moved/revoked); a counted drop with no reply target.
    #[must_use]
    pub fn crossing_subject_gone(&self) -> u64 {
        self.crossing_subject_gone
    }

    /// Slice 3f-C — transient `TransientCrossingGrant`s emitted for a resolved dest realm.
    #[must_use]
    pub fn transient_crossings_granted(&self) -> u64 {
        self.transient_crossings_granted
    }

    /// Slice 3f-C — transient `TransientCrossingRequest`s dropped because the dest realm was unresolved.
    #[must_use]
    pub fn transient_dest_unresolved(&self) -> u64 {
        self.transient_dest_unresolved
    }
}
