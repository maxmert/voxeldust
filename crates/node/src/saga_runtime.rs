//! The Transfer Saga RUNTIME (P2 Slice 1b) — the orchestrator-side wrapper around the
//! pure [`vd_sim::saga`] FSM. It is the ONE owner of a transfer's lifecycle at runtime:
//!
//! - **Per-key serialization**: at most one saga per `DirectoryKey`, enforced by
//!   `DirectoryCore::lock_transfer` (the directory's `in_transfer` field) — a duplicate
//!   trigger is refused, not double-started.
//! - **Drives the gateway**: every `SagaAction::Send(TransferControl)` is serialized onto
//!   the `InterShardFlow::Saga` arm to the session's gateway (the gateway never decides a
//!   transfer; HR1 typed seam).
//! - **The single commit point, DIRECT**: `IssueCommitCas` calls `DirectoryCore::commit_cas`
//!   IN PROCESS — the orchestrator owns `DirectoryRes` on this same single-threaded schedule,
//!   so there is no self-addressed wire round-trip (no tick split, no `ResMut` race). The CAS
//!   outcome (`Won`/`Lost`) is fed straight back into the FSM as the next event.
//! - **Class-aware from line one (HR2)**: a `Transient` subject reaches the same commit point
//!   but issues the batched `TransientGo` go-token (a no-op stub until P3), never a per-entity
//!   CAS — the FSM's `commit_action` fan-out, executed here without a shard-kind branch.
//!
//! The sim thread NEVER awaits: all egress is enqueue-only (`OutboundBox`). At-least-once
//! delivery + adaptive timeouts + the CAS re-read loop land at Slice 2; the client-facing cut
//! cycle landed at Slice 1c/1e. The shard-bound `StubCrossing` state transfer is synthesized at
//! Slice 1d. The ORDERED demote-before-promote tail (1d.5b.1, D-2) pushes `Demote`→source then
//! (on `DemoteAcked`) `Promote`→dest, and releases the source sub only once the dest both acked
//! the promote AND delivered to every observer — the seamless no-vanish gate (`saga.rs` Promoting).
//!
//! ## Slice 2a — the RECOVERY MACHINERY (the R1 cure; closes D-1 + the post-commit park)
//! [`scan_deadlines`] is the production TIMEOUT PRODUCER: a `now - since >= deadline_for(phase)`
//! scan (two thresholds — cheap `redrive` for post-commit, large `abort` for the destructive
//! pre-freeze/compensation phases) that injects `SagaEvent::Timeout`, so a lost saga-ack RE-DRIVES
//! (post-commit) or ABORTS (pre-freeze) instead of PARKING forever. The terminal `Aborted` edge now
//! emits `SagaAction::ClearTransferLock` → [`DirectoryCore::abort_clear`] (the stale-fence head
//! re-read — a CAS-loser's `expected_fence` is stale by definition), so the directory lock CLEARS
//! and an aborted subject can immediately re-transfer (D-1 CLOSED; the two pinned asserts FLIPPED
//! to `None`). ⚠️ HONEST SCOPE: a fixed tick deadline is NOT a crash-vs-slow discriminator (only a
//! permanent `kill` emits `NodeUnreachable`) — a mis-tuned `abort_deadline_ticks` below a deployment's
//! worst-case healthy pre-freeze round-trip WILL false-abort a slow-but-alive saga (the real
//! discriminator is lease-lapse liveness, D-3, owed). The producer cures lost SAGA-ACKS, not a starved
//! standing delivery watermark (a `Promoting` saga whose dest frame delivery is blocked stays
//! half-open — owed, P3) nor an orchestrator CRASH (no durable WAL, D-6). (Caught by Slice-1b audit
//! CPO-1/CPO-2 + Slice-2a design `wf_9f22c70d`.)

use std::collections::{BTreeMap, VecDeque};

use bevy_ecs::prelude::{Res, ResMut, Resource};
use vd_core::pose::StampedPose;
use vd_core::{BatchId, EpochId, Fence, NodeId, TransferId, UniverseTick};
use vd_sim::directory::DirectoryCore;
use vd_sim::io::{Inbound, MsgClass};
use vd_sim::runtime::{ClockSample, InboundBox, OutboundBox};
use vd_sim::saga::{self, AbortReason, SagaAction, SagaCtx, SagaEvent, SagaState, SagaTuning};
use vd_wire::intershard::{
    DEMOTE_STEP, DemoteCmd, FLUSH_SOURCE_STEP, FlushSource, InterShardFlow, PROMOTE_STEP,
    PromoteCmd, STUB_CROSSING_STEP, TRANSFER_SCHEMA_VERSION, TRANSIENT_DROP_STEP,
    TRANSIENT_RELEASE_STEP, TransferAck, TransferEnvelope, TransientHandoff, TransitionPayload,
};
use vd_wire::seams::directory::{AuthorityRef, CasOutcome, DirectoryKey};
use vd_wire::seams::transfer_control::TransferControlAck;

use crate::orchestrator::DirectoryRes;

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
struct LiveSaga {
    ctx: SagaCtx,
    state: SagaState,
    gateway: NodeId,
    /// The universe tick of the most recent transition that advanced this saga (set
    /// at creation, refreshed on every write-back). A PARKED saga receives no events,
    /// so `commit_result` is never re-entered for it and `since` stays put — which is
    /// exactly when it entered its current (stuck) state. The admin view reports
    /// `now - since` as staleness so a 2am operator sees how long a saga has been wedged.
    since: UniverseTick,
    /// The source's authoritative pose, stashed when its `SourceFlushed` reply arrives (1d.1) and
    /// read by the `EmitCrossing` executor at/after commit. The pose-before-promote gate guarantees
    /// it is `Some` before the CAS for an Entity subject, so `EmitCrossing` never ships a None pose.
    /// (The TLV state blob joins this at 1d.6; pose-only now.)
    flushed_pose: Option<StampedPose>,
}

/// A pending create-on-trigger. Enqueued by [`SagaRuntimeRes::start_transfer`] and processed
/// next tick (a stub-shard boundary-crossing request wires the real trigger at Slice 1d).
struct PendingStart {
    ctx: SagaCtx,
    gateway: NodeId,
}

/// The live-saga set + the trigger queue, resource-wrapped (single writer: this schedule).
#[derive(Resource, Default)]
pub struct SagaRuntimeRes {
    sagas: BTreeMap<TransferId, LiveSaga>,
    pending: Vec<PendingStart>,
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
    tuning: SagaTuning,
    /// THE batched `TransientGo` go-token ledger (HR2/D-7): one record per `BatchId` =
    /// `(commit_fence, source, dest)`. Written by the `IssueTransientGo` executor at the transient
    /// SHORT-PATH commit point — ONE write per batch regardless of item count (G-TIER); the
    /// `BatchAdopted`/`DropApplied` handlers read it to drive the D-7b structural drop-before-promote
    /// (release the source, gate the dest promote, then ReleaseComplete the source). The
    /// `TRANSIENT-AUTHORITY-HELD` oracle cross-checks every Held transient's set
    /// anchor against a committed go-token here — a transient is NEVER a directory `OwnerRecord`
    /// (burst-isolation: a 1000-debris burst writes ZERO directory rows). ⚠️ IN-MEMORY (durability
    /// owed D-6) + UNBOUNDED in D-7a (bounded GC of completed go-tokens is owed D-7c — it needs the
    /// drop-completion signal; the oracle needs the live record until then). `or_insert` idempotent:
    /// a Slice-2a re-driven go-token re-records the SAME (batch → fence) without duplication.
    batch_goes: BTreeMap<BatchId, (Fence, NodeId, NodeId)>,
}

/// One batched `TransientGo` go-token emitted by the `IssueTransientGo` executor, COLLECTED by
/// `run_to_quiescence` (which is deliberately denied a `runtime` handle — `commit_result`'s
/// write-back soundness rests on it) and recorded into [`SagaRuntimeRes::batch_goes`] by
/// `commit_result`. `batch` is the `BatchId` (the transient saga's transfer); `fence` is the dest
/// realm-lease fence the batch committed at; `source`/`dest` are the shards the D-7b handoff commands
/// (`TransientRelease`/`TransientDrop`/`ReleaseComplete`) target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchGo {
    batch: BatchId,
    fence: Fence,
    source: NodeId,
    dest: NodeId,
}

/// Hard ceiling on the un-drained rejection ledger (operational param, ONE home — no inline
/// literal). Sized to absorb a large abort burst between 1c.9 drain cycles; beyond it the
/// oldest rejections are shed (counted) rather than grow the orchestrator without bound.
const REJECTION_LEDGER_CAP: usize = 1024;

impl SagaRuntimeRes {
    /// Construct with the deadline budget from config (Slice 2a). The production path
    /// (`register_orchestrator`) calls this with `cfg.saga`; `Default` is the dev/test value.
    #[must_use]
    pub fn with_tuning(tuning: SagaTuning) -> SagaRuntimeRes {
        SagaRuntimeRes {
            tuning,
            ..SagaRuntimeRes::default()
        }
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
}

/// Map a gateway→saga ack to the FSM event it drives. `CasWon`/`CasLost` are NOT here — they
/// come from the DIRECT `commit_cas`, never the wire.
fn ack_to_event(ack: TransferControlAck) -> SagaEvent {
    match ack {
        TransferControlAck::Prepared { result, .. } => SagaEvent::Prepared(result),
        TransferControlAck::CutConfirmed { marker_seq, .. } => {
            SagaEvent::CutConfirmed { marker_seq }
        }
        TransferControlAck::SourceFrozen { drained_seq, .. } => {
            SagaEvent::SourceFrozen { drained_seq }
        }
        // The gateway's `CommitAuthority` ack = the route swapped (post-commit, forward-only).
        TransferControlAck::Committed { .. } => SagaEvent::RouteSwapped,
        TransferControlAck::SourceThawed { .. } => SagaEvent::SourceThawed,
        TransferControlAck::Aborted { .. } => SagaEvent::DestAborted,
        TransferControlAck::Released { .. } => SagaEvent::Released,
        // 1d.5b.1 (D-2): the ordered demote/promote acks + the standing delivery watermark drive the
        // FSM tail directly — DemoteAcked: Demoting→Promoting; PromoteAcked + DestDelivered: the
        // Promoting→Releasing seamless gate (DestDelivered is also latched early in Demoting).
        TransferControlAck::DemoteAck { .. } => SagaEvent::DemoteAcked,
        TransferControlAck::PromoteAck { .. } => SagaEvent::PromoteAcked,
        TransferControlAck::DeliveredToObservers { .. } => SagaEvent::DestDelivered,
    }
}

/// Build the `StubCrossing` envelope the saga ships to the dest at/after commit (1d.1). Returns
/// `None` (a no-op) for a non-Entity subject or a missing flushed pose — the pose-before-promote
/// gate makes the latter unreachable for an Entity subject. The `fence` is the post-CAS authority
/// fence (fence rule 1: the receiver rejects a stale crossing); `state` is empty (the TLV blob
/// lands at 1d.6). Monomorphic helper so the executor arm stays a branchless dispatch (HR5).
fn build_crossing(
    ctx: &SagaCtx,
    fence: Fence,
    flush_pose: Option<StampedPose>,
    epoch: EpochId,
) -> Option<InterShardFlow> {
    let entity = ctx.subject.transfer_subject_entity()?;
    let pose = flush_pose?;
    Some(InterShardFlow::Transfer(TransferEnvelope {
        transfer_id: ctx.transfer,
        universe_epoch: epoch,
        schema_version: TRANSFER_SCHEMA_VERSION,
        fence,
        step_id: STUB_CROSSING_STEP,
        class: ctx.class,
        payload: TransitionPayload::StubCrossing {
            entity,
            from_realm: ctx.from_realm,
            to_realm: ctx.to_realm,
            pose,
            state: vec![],
        },
    }))
}

/// Stash the source's flushed pose onto its live saga, BEFORE the `SourceFlushed` event is
/// stepped — so an `EmitCrossing` reachable on the same tick reads it. A flush for an unknown /
/// GC'd saga is a stale reply: dropped (the event deliver is also a no-op).
fn stash_flush(runtime: &mut SagaRuntimeRes, transfer: TransferId, pose: StampedPose) {
    if let Some(live) = runtime.sagas.get_mut(&transfer) {
        live.flushed_pose = Some(pose);
    }
}

/// Push the crossing to the dest if one can be built (Entity subject + a stashed pose), else a
/// LOUD no-op. Holds the Some/None branch (covered both ways by unit tests) so the executor arm
/// stays branchless (HR5). For an Entity subject the pose-before-promote gate guarantees a pose,
/// so the None arm is reached only by the non-Entity subjects the FSM proptests drive.
fn emit_crossing(
    ctx: &SagaCtx,
    fence: Fence,
    flush_pose: Option<StampedPose>,
    epoch: EpochId,
    outbox: &mut OutboundBox,
) {
    match build_crossing(ctx, fence, flush_pose, epoch) {
        Some(crossing) => outbox.push_flow(ctx.dest, MsgClass::Saga, &crossing),
        None => tracing::warn!(
            transfer = ctx.transfer.0,
            "EmitCrossing skipped: non-Entity subject or no flushed pose"
        ),
    }
}

/// The action executor: run a saga from `(state, actions)` to quiescence, executing each
/// action and feeding any SYNCHRONOUS follow-up event (the direct CAS outcome) back into the
/// FSM. Returns the final state, whether it tombstoned (terminal), and any client rejections.
///
/// ORDERING CONTRACT (audit FF-1): a batch's actions execute IN EMITTED ORDER and the WHOLE
/// batch completes before the next queued event is stepped — so when `CasWon` produces
/// `[PersistCheckpoint, Send(CommitAuthority), EmitCrossing]`, the sends happen before any later
/// event is processed. The FSM still emits at most ONE synchronous-feedback action (the commit)
/// per batch — `FlushSource`/`EmitCrossing` are pure egress (no event fed back) — so the loop
/// depth stays bounded (1d.1 preserves the FF-1 second-feedback-source bound).
///
/// `flush_pose` is the saga's stashed source pose (`Some` once `SourceFlushed` arrived); the
/// `EmitCrossing` arm reads it. `epoch` stamps the emitted crossing envelope.
#[allow(clippy::too_many_arguments)]
fn run_to_quiescence(
    ctx: &SagaCtx,
    gateway: NodeId,
    mut state: SagaState,
    mut actions: Vec<SagaAction>,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
    flush_pose: Option<StampedPose>,
) -> (
    SagaState,
    bool,
    Vec<(TransferId, AbortReason)>,
    Vec<BatchGo>,
) {
    let mut events: VecDeque<SagaEvent> = VecDeque::new();
    let mut tombstone = false;
    let mut rejected = Vec::new();
    let mut batch_gos = Vec::new();
    loop {
        for action in actions {
            match action {
                SagaAction::Send(cmd) => {
                    outbox.push_flow(gateway, MsgClass::Saga, &InterShardFlow::Saga(cmd));
                }
                // Tell the SOURCE to ship the subject's pose (1d.1). Pure egress to ctx.source —
                // no synchronous feedback (FF-1 preserved).
                SagaAction::FlushSource => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::FlushSource(FlushSource {
                            transfer: ctx.transfer,
                            subject: ctx.subject,
                            step_id: FLUSH_SOURCE_STEP,
                        }),
                    );
                }
                // Emit the entity-STATE crossing to the DEST (1d.1) from the stashed pose. The
                // Some/None branch lives in `emit_crossing` (a monomorphic helper, unit-tested both
                // ways), so this arm stays a branchless dispatch (HR5).
                SagaAction::EmitCrossing { fence } => {
                    emit_crossing(ctx, fence, flush_pose, epoch, outbox);
                }
                // Push the ORDERED Demote to the SOURCE shard (1d.5b.1, D-2) — the fence-enforced
                // Owned→Frozen→Ghost that REPLACES the 1c.8 poll. Pure egress to ctx.source (its
                // DemoteAck rides InterShardFlow::SagaAck back); no synchronous feedback (FF-1).
                SagaAction::Demote { new_owner_fence } => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::Demote(DemoteCmd {
                            transfer: ctx.transfer,
                            subject: ctx.subject,
                            new_owner_fence,
                            step_id: DEMOTE_STEP,
                        }),
                    );
                }
                // Push the ORDERED Promote to the DEST shard (1d.5b.1, D-2), emitted only after
                // DemoteAcked (demote-before-promote). Pure egress to ctx.dest (its PromoteAck rides
                // SagaAck back); no synchronous feedback (FF-1).
                SagaAction::Promote { new_fence } => {
                    outbox.push_flow(
                        ctx.dest,
                        MsgClass::Saga,
                        &InterShardFlow::Promote(PromoteCmd {
                            transfer: ctx.transfer,
                            subject: ctx.subject,
                            new_fence,
                            step_id: PROMOTE_STEP,
                            // The dest registers this SOURCE as a ghost-neighbor + drives the
                            // GhostFlow collider feed to it after promoting (1d.5b.3b).
                            source: ctx.source,
                        }),
                    );
                }
                // THE single commit point, called DIRECTLY (this thread owns the directory).
                // The outcome re-enters the FSM immediately — no wire round-trip, no tick split.
                // ⚠️ SCALE (DEFERRED D-32): this in-process DIRECT CAS is correct ONLY while ONE
                // orchestrator owns the WHOLE keyspace. When the directory partitions by region
                // (N-orchestrator scaling), this must ROUTE a CommitCas op to `ctx.subject`'s
                // coordinator (`coordinator_of(key)`), not call `commit_cas` locally.
                SagaAction::IssueCommitCas { expected } => {
                    let outcome =
                        dir.commit_cas(ctx.subject, expected, AuthorityRef::Shard(ctx.dest), now);
                    events.push_back(match outcome {
                        CasOutcome::Won { new_fence } => SagaEvent::CasWon { new_fence },
                        CasOutcome::Lost { current } => SagaEvent::CasLost { current },
                    });
                }
                // D-7: the batched `TransientGo` go-token IS the transient commit point (HR2 — the
                // SAME `CasWon` feedback as the durable CAS, fanned out by `commit_action`, NEVER a
                // per-entity directory CAS: transients aren't `OwnerRecord`s, so a 1000-debris burst
                // writes ZERO directory rows — burst isolation, G-TIER). Collect the go-token (it is
                // recorded into `runtime.batch_goes` by `commit_result` — `run_to_quiescence` is
                // intentionally denied the runtime handle) and feed `CasWon` back so the SHORT-PATH
                // saga reaches `Done` (FF-1: exactly ONE feedback event, like `IssueCommitCas`). The
                // commit fence is `expected` — the dest realm-lease fence the batch set anchors to.
                SagaAction::IssueTransientGo { expected } => {
                    batch_gos.push(BatchGo {
                        batch: BatchId(ctx.transfer),
                        fence: expected,
                        source: ctx.source,
                        dest: ctx.dest,
                    });
                    events.push_back(SagaEvent::CasWon {
                        new_fence: expected,
                    });
                }
                // P3: group-committed redb WAL. LOUD deferral: an orchestrator restart loses
                // every in-flight saga until then — never let a log read as if durability
                // exists (audit ROB-2).
                SagaAction::PersistCheckpoint => {
                    tracing::warn!(
                        transfer = ctx.transfer.0,
                        "PersistCheckpoint is a P3 stub: saga state is IN-MEMORY ONLY \
                         (a restart loses in-flight sagas)"
                    );
                }
                SagaAction::NotifyRejected(reason) => rejected.push((ctx.transfer, reason)),
                // Clear the directory lock at TERMINAL abort (Slice 2a, closes D-1), called DIRECTLY
                // (this thread owns the directory — same as IssueCommitCas). The stale-fence re-read
                // (`abort_clear`); the saga is already terminal so the outcome feeds nothing back into
                // the FSM (Won on the first terminal, Lost on an idempotent re-driven terminal). FF-1.
                SagaAction::ClearTransferLock => {
                    let _ = dir.abort_clear(ctx.subject, ctx.transfer);
                }
                SagaAction::Tombstone => tombstone = true,
            }
        }
        match events.pop_front() {
            Some(event) => {
                let (next, acts) = saga::step(ctx, state, event);
                state = next;
                actions = acts;
            }
            None => break,
        }
    }
    (state, tombstone, rejected, batch_gos)
}

/// Apply one event to an existing saga, then run it to quiescence + persist the result. A
/// stale event for an unknown/GC'd saga is an idempotent no-op (at-least-once delivery).
fn deliver(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
    transfer: TransferId,
    event: SagaEvent,
) {
    let Some(live) = runtime.sagas.get_mut(&transfer) else {
        return;
    };
    let ctx = live.ctx;
    let gateway = live.gateway;
    let flush_pose = live.flushed_pose; // Copy; the EmitCrossing executor reads it
    let (state, actions) = saga::step(&ctx, live.state, event);
    let (final_state, tombstone, rejected, batch_gos) = run_to_quiescence(
        &ctx, gateway, state, actions, dir, outbox, epoch, now, flush_pose,
    );
    commit_result(
        runtime,
        transfer,
        final_state,
        tombstone,
        rejected,
        batch_gos,
        now,
    );
}

/// ROB-1: bound the un-drained rejection ledger to `REJECTION_LEDGER_CAP`, shedding the
/// OLDEST beyond it with a counted, warned drop — never an unbounded leak, never silent.
/// Monomorphic helper (the branch is covered once; `commit_result` stays a straight shim).
fn bound_rejection_ledger(runtime: &mut SagaRuntimeRes) {
    let shed = runtime.rejected.len().saturating_sub(REJECTION_LEDGER_CAP);
    if shed > 0 {
        runtime.rejected.drain(0..shed);
        runtime.rejections_dropped += shed as u64;
        tracing::warn!(
            shed,
            cap = REJECTION_LEDGER_CAP,
            "saga rejection ledger over cap: shed oldest (un-surfaced until Slice 1c.9 CPO-4); \
             never silent"
        );
    }
}

/// Persist a quiescent saga: record rejections + the batched go-tokens, then GC (terminal) or write
/// back the state.
fn commit_result(
    runtime: &mut SagaRuntimeRes,
    transfer: TransferId,
    final_state: SagaState,
    tombstone: bool,
    rejected: Vec<(TransferId, AbortReason)>,
    batch_gos: Vec<BatchGo>,
    now: UniverseTick,
) {
    runtime.rejected.extend(rejected);
    bound_rejection_ledger(runtime); // ROB-1: never let the un-drained ledger grow unbounded
    // D-7: record the batched go-tokens emitted this run (here, NOT in `run_to_quiescence` — which
    // is denied the runtime handle so this write-back stays sound). `or_insert` is idempotent: a
    // Slice-2a re-driven go-token re-records the SAME (batch → fence/source/dest), never a dup.
    for bg in batch_gos {
        runtime
            .batch_goes
            .entry(bg.batch)
            .or_insert((bg.fence, bg.source, bg.dest));
    }
    if tombstone {
        runtime.sagas.remove(&transfer);
    } else {
        // Sound because nothing between the lookup and this write-back can touch
        // `runtime.sagas`: `run_to_quiescence` takes only `dir`/`outbox` (the borrow
        // checker enforces it cannot reach the runtime), and this schedule is
        // single-threaded. If a future refactor hands it the runtime, re-prove this.
        let live = runtime
            .sagas
            .get_mut(&transfer)
            .expect("the saga was present at the start of this run");
        // AAA-1 staleness: refresh `since` ONLY on an ACTUAL state change. `deliver` runs
        // for EVERY matching ack — including at-least-once duplicates / stray acks the FSM
        // absorbs as same-state (`saga.rs`'s `(state, _) => (state, vec![])` no-op arm) — so
        // an unconditional bump would RESET a parked saga's staleness on a redelivery,
        // defeating the very stuck-saga metric this field exists for. The stored `live.state`
        // is still the prior state here (single-threaded schedule; nothing touched it between
        // the lookup and now), so the compare is exact.
        if live.state != final_state {
            live.since = now;
        }
        live.state = final_state;
    }
}

/// Process the create-on-trigger queue: per-key-serialize via `lock_transfer`, `start` the
/// FSM, and run its initial actions to quiescence.
fn process_starts(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
) {
    for PendingStart { ctx, gateway } in std::mem::take(&mut runtime.pending) {
        // Per-key serialization (DURABLE only): `lock_transfer` sets the directory `in_transfer`.
        // FALSE means the subject is absent OR already transferring — refuse to start (a
        // spurious/duplicate trigger is a no-op, never a second concurrent saga on the same key). A
        // TRANSIENT batch is NOT in the directory (`locks_directory_key(Transient)=false`) — burst
        // isolation is STRUCTURAL: a debris burst takes ZERO directory locks, so it can never
        // serialize against / wedge a concurrent Durable saga (HR2). The short-circuit `&&` is
        // HR5-coverable here: `locks_directory_key` is exercised true (Durable) AND false (Transient),
        // and its true arm's `lock_transfer` is exercised both ways (the durable happy path = ok; the
        // unrecorded-subject refusal = fail) — so every branch of both operands is hit.
        if locks_directory_key(ctx.class) && !dir.lock_transfer(ctx.subject, ctx.transfer) {
            continue;
        }
        let (state, actions) = saga::start(&ctx);
        runtime.sagas.insert(
            ctx.transfer,
            LiveSaga {
                ctx,
                state,
                gateway,
                since: now,
                flushed_pose: None, // filled when the source flushes (after Freezing)
            },
        );
        // Durable start = PrepareSubscribe (no EmitCrossing yet); transient start = the go-token
        // (collected by run_to_quiescence). `None` flush_pose is correct for both at start.
        let (final_state, tombstone, rejected, batch_gos) =
            run_to_quiescence(&ctx, gateway, state, actions, dir, outbox, epoch, now, None);
        commit_result(
            runtime,
            ctx.transfer,
            final_state,
            tombstone,
            rejected,
            batch_gos,
            now,
        );
    }
}

/// Whether a saga of this class takes a directory `in_transfer` LOCK at start (HR2 fan-out, the
/// burst-isolation seam): `Durable` subjects are per-entity `OwnerRecord`s serialized by the
/// directory; `Transient` batches are NOT in the directory at all (held-set anchored, committed by
/// the batched go-token), so a debris burst takes zero locks and can never wedge a Durable saga. An
/// explicit `match` (not `matches!`) so both arms are covered regions (HR5).
#[must_use]
fn locks_directory_key(class: vd_core::entity_kind::DurabilityClass) -> bool {
    match class {
        vd_core::entity_kind::DurabilityClass::Durable => true,
        vd_core::entity_kind::DurabilityClass::Transient => false,
    }
}

/// The deadline a saga's CURRENT phase fires its `Timeout` at (Slice 2a). Pre-freeze/freeze/aborting
/// phases use the LARGE `abort_deadline_ticks` (their Timeout is DESTRUCTIVE — an abort — or re-drives
/// a compensator on the same slow latency profile); committing/post-commit phases use the small
/// `redrive_deadline_ticks` (a cheap idempotent re-drive that may fire "early" at zero correctness
/// cost); terminal phases return `u64::MAX` (never fire). A straight monomorphic match — the ONLY
/// per-phase logic, hoisted out of the phase-agnostic producer scan (HR3).
#[must_use]
fn deadline_for(state: &SagaState, tuning: &SagaTuning) -> u64 {
    match state {
        SagaState::AwaitProvision
        | SagaState::Preparing
        | SagaState::Cutting
        | SagaState::Freezing { .. }
        | SagaState::Aborting { .. } => tuning.abort_deadline_ticks,
        SagaState::CommittingCas { .. }
        | SagaState::BatchCommitting { .. }
        | SagaState::Swapping { .. }
        | SagaState::Demoting { .. }
        | SagaState::Promoting { .. }
        | SagaState::Releasing { .. } => tuning.redrive_deadline_ticks,
        SagaState::Done { .. } | SagaState::Aborted { .. } => u64::MAX,
    }
}

/// THE Slice 2a TIMEOUT PRODUCER (the R1 cure): inject `SagaEvent::Timeout` into every live saga whose
/// `now - since` reached its phase deadline, so a lost saga-ack RE-DRIVES (post-commit) or ABORTS
/// (pre-freeze) instead of PARKING forever. Phase-agnostic — ONE scan, ONE injected event; the FSM's
/// existing per-phase Timeout arms do all the work. `since` is REFRESHED on fire (BEFORE delivery), so
/// a still-stale saga re-fires at most once per its deadline window, never every tick (no storm).
/// O(live sagas)/tick — matches the existing `views()`/`active_transfers()` scans; a deadline-ordered
/// structure (pop only the due) is a P3/MMO-scale optimization, not built now.
fn scan_deadlines(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
) {
    let tuning = runtime.tuning;
    let mut due: Vec<TransferId> = Vec::new();
    for (transfer, live) in runtime.sagas.iter_mut() {
        if now.0.saturating_sub(live.since.0) >= deadline_for(&live.state, &tuning) {
            live.since = now; // re-arm BEFORE delivery — one fire per window, never a per-tick storm
            due.push(*transfer);
        }
    }
    for transfer in due {
        deliver(
            runtime,
            dir,
            outbox,
            epoch,
            now,
            transfer,
            SagaEvent::Timeout,
        );
    }
}

/// Look up a batch's committed go-token (commit-fence, source, dest) and run `f` with it; a missing
/// go-token (a stray/duplicate ack, or a batch that never started/already retired) is a LOUD no-op —
/// never a silent drop. The shared lookup for both transient-handoff ack handlers (DRY); read-only on
/// the runtime (the egress is pure; FF-1).
fn with_batch_token(
    runtime: &SagaRuntimeRes,
    transfer: TransferId,
    f: impl FnOnce(Fence, NodeId, NodeId),
) {
    match runtime.batch_goes.get(&BatchId(transfer)) {
        Some(&(fence, source, dest)) => f(fence, source, dest),
        None => tracing::warn!(
            transfer = transfer.0,
            "transient handoff ack with no committed go-token: ignored (stray/duplicate)"
        ),
    }
}

/// Handle the dest's `BatchAdopted` ack (D-7b adopt-before-drop, PHASE 1): the dest now holds the
/// batch as the uncounted `Arriving` tier, so tell the SOURCE ONLY to RELEASE (flip `Held→Departing`,
/// go uncounted) — the transient twin of the durable ordered `Demote`. The dest promote does NOT fire
/// yet; it is gated on the source's `DropApplied` (`handle_drop_applied`), so the source is uncounted
/// BEFORE the dest counts — the holder set is never `{source, dest}`.
fn handle_batch_adopted(runtime: &SagaRuntimeRes, outbox: &mut OutboundBox, transfer: TransferId) {
    with_batch_token(runtime, transfer, |fence, source, _dest| {
        outbox.push_flow(
            source,
            MsgClass::Saga,
            &InterShardFlow::TransientRelease(TransientHandoff {
                transfer,
                step_id: TRANSIENT_RELEASE_STEP,
                fence,
            }),
        );
    });
}

/// Handle a `DropApplied` ack (D-7b, the phased proof-of-apply gate): `TRANSIENT_RELEASE_STEP` means
/// the SOURCE released (now uncounted) → PROMOTE the dest (`TransientDrop` to DEST only); any other
/// phase is the DEST's promote-confirm (`TRANSIENT_DROP_STEP`) → retire the source's retained
/// `Departing` copy (`ReleaseComplete` to SOURCE only). The binary if/else is total over the only two
/// phases `DropApplied` ever carries (enforced by its sole producers — the source release-ack + the
/// dest promote-confirm), so both arms are reachable and there is no uncoverable third branch (HR5).
fn handle_drop_applied(
    runtime: &SagaRuntimeRes,
    outbox: &mut OutboundBox,
    transfer: TransferId,
    step_id: u32,
) {
    with_batch_token(runtime, transfer, |fence, source, dest| {
        if step_id == TRANSIENT_RELEASE_STEP {
            outbox.push_flow(
                dest,
                MsgClass::Saga,
                &InterShardFlow::TransientDrop(TransientHandoff {
                    transfer,
                    step_id: TRANSIENT_DROP_STEP,
                    fence,
                }),
            );
        } else {
            outbox.push_flow(
                source,
                MsgClass::Saga,
                &InterShardFlow::ReleaseComplete(TransientHandoff {
                    transfer,
                    step_id: TRANSIENT_RELEASE_STEP,
                    fence,
                }),
            );
        }
    });
}

/// The orchestrator saga-runtime system: process new triggers, FIRE due deadlines (Slice 2a), then
/// drive every live saga forward on the gateway acks delivered this tick. Runs on the orchestrator's
/// single-threaded schedule; the directory CAS is a direct in-process call (no await, no lock across a send).
pub fn drive_sagas(
    inbox: Res<InboundBox>,
    clock: Res<ClockSample>,
    mut dir: ResMut<DirectoryRes>,
    mut runtime: ResMut<SagaRuntimeRes>,
    mut outbox: ResMut<OutboundBox>,
) {
    let now = clock.universe_tick;
    let epoch = clock.epoch;
    process_starts(&mut runtime, &mut dir.0, &mut outbox, epoch, now);
    // Slice 2a: fire due deadlines BEFORE the ack loop — a saga that loses its ack this tick still
    // gets its Timeout re-drive/abort next tick (the producer is the R1 backstop, never a wedge).
    scan_deadlines(&mut runtime, &mut dir.0, &mut outbox, epoch, now);
    for msg in &inbox.0 {
        let Inbound::Wire { class, bytes, .. } = msg else {
            continue;
        };
        if *class != MsgClass::Saga {
            continue;
        }
        // Two saga-driving inbound arms ride MsgClass::Saga to the orchestrator: the gateway's
        // SagaAck (route-swap phases) and the SOURCE's TransferAck::SourceFlushed (the pose). The
        // DEST's crossing ack is decoded-and-dropped (1d.1 — see below). Directory ops are
        // `serve_directory`'s; Saga commands / FlushSource flow OUT, never in.
        match postcard::from_bytes::<InterShardFlow>(bytes) {
            Ok(InterShardFlow::SagaAck(ack)) => {
                deliver(
                    &mut runtime,
                    &mut dir.0,
                    &mut outbox,
                    epoch,
                    now,
                    ack.transfer(),
                    ack_to_event(ack),
                );
            }
            Ok(InterShardFlow::TransferAck(TransferAck::SourceFlushed {
                transfer_id,
                drained_seq,
                pose,
                ..
            })) => {
                // STASH the pose BEFORE stepping the event, so an EmitCrossing reachable on this
                // same tick (once both freeze + flush have landed) reads it.
                stash_flush(&mut runtime, transfer_id, pose);
                deliver(
                    &mut runtime,
                    &mut dir.0,
                    &mut outbox,
                    epoch,
                    now,
                    transfer_id,
                    SagaEvent::SourceFlushed { drained_seq },
                );
            }
            // D-7b adopt-before-drop PHASE 1: the dest ADOPTED the batch (uncounted `Arriving`) → tell
            // the SOURCE ONLY to release (`Held→Departing`), gating the dest promote on the source's
            // `DropApplied`. The transient twin of the demote-before-promote tail.
            Ok(InterShardFlow::TransferAck(TransferAck::BatchAdopted { transfer_id, .. })) => {
                handle_batch_adopted(&runtime, &mut outbox, transfer_id);
            }
            // D-7b PHASES 2+3: a `DropApplied` proof-of-apply — RELEASE_STEP (source released) gates
            // the dest PROMOTE; DROP_STEP (dest promote-confirmed) drives the source `ReleaseComplete`.
            Ok(InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id,
                step_id,
            })) => {
                handle_drop_applied(&runtime, &mut outbox, transfer_id, step_id);
            }
            // The DEST's crossing ack: the dest journal is the exactly-once dedup. Release is NOT
            // gated on this ack — it rides the ordered `PromoteAck` (1d.5b.1) instead. The crossing
            // ack is decoded + dropped here — never a phase transition.
            Ok(InterShardFlow::TransferAck(
                TransferAck::Accepted { .. } | TransferAck::Rejected { .. },
            )) => {}
            // Everything else (Ghost / Directory / Saga commands / DirectoryReply / FlushSource)
            // and any decode failure: not a saga-driving inbound here.
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{NodeConfig, build_app};
    use crate::orchestrator::{OrchestratorConfig, register_orchestrator};
    use vd_core::entity_kind::{DurabilityClass, EntityKind};
    use vd_core::glam::DVec3;
    use vd_core::pose::{FrameRef, RealmId};
    use vd_core::{EntityId, EpochId, Fence, SessionId, UniverseTick};
    use vd_sim::capability::NodeKind;
    use vd_sim::directory::DirectoryTuning;
    use vd_sim::io::Transport;
    use vd_sim::io::mem::MemHub;
    use vd_wire::seams::directory::{DirectoryKey, DirectoryOp};
    use vd_wire::seams::transfer_control::{
        PrepareReject, PrepareResult, SpatialReject, TransferControl,
    };

    const FROM_REALM: RealmId = RealmId::System(7);
    const TO_REALM: RealmId = RealmId::System(8);

    /// A non-origin pose the source "flushes" — distinct components so a test can confirm the
    /// EmitCrossing carried THIS pose (not a default).
    fn flushed_pose() -> StampedPose {
        StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 7 },
            DVec3::new(1.0, 2.0, 3.0),
            UniverseTick(5),
        )
    }

    const ORCH: NodeId = NodeId(1);
    const SOURCE: NodeId = NodeId(2);
    const DEST: NodeId = NodeId(3);
    const GATEWAY: NodeId = NodeId(4);
    const XFER: TransferId = TransferId(77);
    const SESSION: SessionId = SessionId(5);

    fn subject_eid() -> EntityId {
        EntityId::pack(EntityKind::Player, 1, 7, 3)
    }

    fn subject() -> DirectoryKey {
        DirectoryKey::Entity(subject_eid())
    }

    /// The wire form of one `TransferControl` command as the gateway receives it from the
    /// orchestrator (full-value equality target).
    fn saga_wire(cmd: TransferControl) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(&InterShardFlow::Saga(cmd))
                .expect("encode")
                .into(),
        }
    }

    /// The wire form of the ordered `Demote` as the SOURCE receives it (1d.5b.1).
    fn demote_wire(new_owner_fence: Fence) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(&InterShardFlow::Demote(vd_wire::intershard::DemoteCmd {
                transfer: XFER,
                subject: subject(),
                new_owner_fence,
                step_id: vd_wire::intershard::DEMOTE_STEP,
            }))
            .expect("encode")
            .into(),
        }
    }

    /// The wire form of the ordered `Promote` as the DEST receives it (1d.5b.1).
    fn promote_wire(new_fence: Fence) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(&InterShardFlow::Promote(
                vd_wire::intershard::PromoteCmd {
                    transfer: XFER,
                    subject: subject(),
                    new_fence,
                    step_id: vd_wire::intershard::PROMOTE_STEP,
                    source: SOURCE,
                },
            ))
            .expect("encode")
            .into(),
        }
    }

    fn ctx(class: vd_core::entity_kind::DurabilityClass, expected_fence: Fence) -> SagaCtx {
        SagaCtx {
            transfer: XFER,
            session: SESSION,
            subject: subject(),
            expected_fence,
            source: SOURCE,
            dest: DEST,
            class,
            needs_provision: false,
            from_realm: FROM_REALM,
            to_realm: TO_REALM,
        }
    }

    /// A TRANSIENT batch saga ctx (D-7): the subject is the dest realm (inert provenance — never
    /// enters the directory, since `locks_directory_key(Transient)=false`); `expected_fence` is the
    /// dest realm-lease fence the batched go-token commits at.
    fn transient_ctx(expected_fence: Fence) -> SagaCtx {
        SagaCtx {
            subject: DirectoryKey::Realm(TO_REALM),
            ..ctx(DurabilityClass::Transient, expected_fence)
        }
    }

    /// The wire form of an orchestrator→shard `InterShardFlow` as the source/dest receive it (D-7b:
    /// the structural drop-before-promote `TransientRelease`/`TransientDrop`/`ReleaseComplete`).
    fn flow_inbound(flow: &InterShardFlow) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(flow).expect("encode").into(),
        }
    }

    /// A `TransientHandoff` egress the orchestrator emits (the target asserts it against `flow_inbound`).
    fn handoff(
        arm: fn(TransientHandoff) -> InterShardFlow,
        step_id: u32,
        fence: Fence,
    ) -> InterShardFlow {
        arm(TransientHandoff {
            transfer: XFER,
            step_id,
            fence,
        })
    }

    /// A stepped orchestrator + a hub the test drives the saga through. The `source`/`dest`
    /// endpoints receive the 1d.1 `FlushSource`/`StubCrossing` egress and let the source inject
    /// its `SourceFlushed` pose reply.
    struct Rig {
        hub: MemHub,
        orch: crate::app::ShardNode<vd_sim::io::mem::MemTransport>,
        gateway: vd_sim::io::mem::MemTransport,
        source: vd_sim::io::mem::MemTransport,
        dest: vd_sim::io::mem::MemTransport,
    }

    impl Rig {
        fn new() -> Rig {
            let hub = MemHub::new();
            let mut orch = build_app(
                NodeConfig {
                    node_id: ORCH,
                    kind: NodeKind::Orchestrator,
                },
                hub.register(ORCH, 64),
            );
            let (world, schedule) = orch.parts_mut();
            register_orchestrator(
                world,
                schedule,
                &OrchestratorConfig {
                    epoch: EpochId(1),
                    reserve_chunk: 1024,
                    clock_peers: vec![],
                    directory: DirectoryTuning {
                        lease_ttl_ticks: 10_000,
                    },
                    saga: SagaTuning::default(),
                },
            );
            let gateway = hub.register(GATEWAY, 64);
            let source = hub.register(SOURCE, 64);
            let dest = hub.register(DEST, 64);
            Rig {
                hub,
                orch,
                gateway,
                source,
                dest,
            }
        }

        /// Deliver pending sends to the orchestrator, run one tick, then deliver its outbound
        /// replies/commands to the peer endpoints (pump → step → pump).
        fn settle(&mut self) {
            self.hub.pump();
            let _ = self.orch.step_tick();
            self.hub.pump();
        }

        /// Grant a directory record for the subject at `fence` (so `lock_transfer` + the CAS
        /// have something to act on), owned by the source shard.
        fn grant_subject(&mut self, fence: Fence) {
            self.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::Directory(
                            DirectoryOp::LeaseGrant {
                                key: subject(),
                                owner: AuthorityRef::Shard(SOURCE),
                                fence,
                            },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// The SOURCE ships its pose (the reply to `FlushSource`) — the second half of the
        /// pose-before-promote gate.
        fn flush(&mut self) {
            self.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::TransferAck(
                            TransferAck::SourceFlushed {
                                transfer_id: XFER,
                                step_id: FLUSH_SOURCE_STEP,
                                pose: flushed_pose(),
                                drained_seq: 0,
                            },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// Everything the orchestrator has sent the DEST since the last drain (the `StubCrossing`
        /// crossing egress + the ordered `Promote`).
        fn drain_dest(&mut self) -> Vec<Inbound> {
            self.dest.drain_inbound()
        }

        /// Everything the orchestrator has sent the SOURCE since the last drain (the `FlushSource`
        /// request + the ordered `Demote`).
        fn drain_source(&mut self) -> Vec<Inbound> {
            self.source.drain_inbound()
        }

        fn trigger(&mut self, ctx: SagaCtx) {
            self.orch
                .world_mut()
                .resource_mut::<SagaRuntimeRes>()
                .start_transfer(ctx, GATEWAY);
        }

        /// Deliver one gateway ack to the orchestrator and step it.
        fn ack(&mut self, ack: TransferControlAck) {
            self.gateway
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::SagaAck(ack)).expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// The DEST acks `BatchAdopted` (D-7) — it now holds the batch as `Arriving`. Drives the
        /// orchestrator's D-7b adopt-before-drop `TransientRelease` egress to the source.
        fn batch_adopted(&mut self, transfer: TransferId) {
            self.dest
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::TransferAck(
                            TransferAck::BatchAdopted {
                                transfer_id: transfer,
                                step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
                            },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// A shard's `DropApplied` proof-of-apply (D-7b): `TRANSIENT_RELEASE_STEP` = the source
        /// released (gates the dest promote); `TRANSIENT_DROP_STEP` = the dest promote-confirmed
        /// (drives the source `ReleaseComplete`). The orchestrator ignores the sender, so it rides
        /// `self.source` regardless of which shard it models.
        fn drop_applied(&mut self, step_id: u32) {
            self.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::TransferAck(
                            TransferAck::DropApplied {
                                transfer_id: XFER,
                                step_id,
                            },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// Everything the orchestrator has sent the gateway since the last drain, as raw
        /// `Inbound` (full-value equality, no destructuring — so there are no never-taken
        /// arms to leave uncovered; the codebase convention).
        fn drain_gateway(&mut self) -> Vec<Inbound> {
            self.gateway.drain_inbound()
        }

        fn subject_head(&mut self) -> vd_wire::seams::directory::OwnerRecord {
            self.orch
                .world_mut()
                .resource::<DirectoryRes>()
                .0
                .head(subject())
                .expect("subject recorded")
        }

        fn live(&mut self) -> usize {
            self.orch.world_mut().resource::<SagaRuntimeRes>().live()
        }
    }

    #[test]
    fn durable_happy_path_drives_the_gateway_through_the_direct_commit() {
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle(); // process the start → PrepareSubscribe
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::PrepareSubscribe {
                transfer: XFER,
                session: SESSION,
                dest: DEST,
            })]
        );

        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::RequestCut {
                transfer: XFER,
                session: SESSION,
            })]
        );

        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 42,
        });
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::FreezeSource {
                transfer: XFER,
                session: SESSION,
                marker_seq: 42,
                dest: DEST,
            })]
        );

        // The pose-before-promote gate: SourceFrozen alone leaves the saga in Freezing (no CAS).
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 42,
        });
        assert!(
            rig.drain_gateway().is_empty(),
            "SourceFrozen alone does not commit — the gate awaits the source flush"
        );
        // The SOURCE flushes its pose → BOTH gate conditions met → the DIRECT commit_cas wins
        // in-process → CommitAuthority is sent with the NEW fence, all in ONE tick.
        rig.flush();
        let new_fence = Fence(1).next();
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::CommitAuthority {
                transfer: XFER,
                session: SESSION,
                new_fence,
                subject: subject(),
            })]
        );
        // The entity-STATE crossing rode the SAME commit batch to the DEST, carrying the flushed
        // pose at the new authority fence (the 1d.1 deliverable, asserted end-to-end).
        assert_eq!(
            rig.drain_dest(),
            vec![Inbound::Wire {
                from: ORCH,
                class: MsgClass::Saga,
                bytes: postcard::to_allocvec(&InterShardFlow::Transfer(TransferEnvelope {
                    transfer_id: XFER,
                    universe_epoch: EpochId(1),
                    schema_version: TRANSFER_SCHEMA_VERSION,
                    fence: new_fence,
                    step_id: STUB_CROSSING_STEP,
                    class: DurabilityClass::Durable,
                    payload: TransitionPayload::StubCrossing {
                        entity: subject_eid(),
                        from_realm: FROM_REALM,
                        to_realm: TO_REALM,
                        pose: flushed_pose(),
                        state: vec![],
                    },
                }))
                .expect("encode")
                .into(),
            }],
            "the crossing reached the dest with the flushed pose at the new fence"
        );
        // The directory CAS committed: the dest now owns the subject at the new fence.
        let head = rig.subject_head();
        assert_eq!(head.authority, AuthorityRef::Shard(DEST));
        assert_eq!(head.fence, new_fence);

        // Clear the source buffer of the earlier egress (the grant DirectoryReply + the Freezing
        // FlushSource) so the next drain isolates the ordered Demote.
        let _ = rig.drain_source();

        // Committed (the route swapped) advances to Demoting and PUSHES the ordered Demote to the
        // SOURCE (the fence-enforced Owned→Frozen→Ghost, at the new owner fence) — NOT yet a release.
        rig.ack(TransferControlAck::Committed { transfer: XFER });
        assert_eq!(
            rig.drain_source(),
            vec![demote_wire(new_fence)],
            "Demoting pushes the ordered Demote to the source at the new owner fence"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "no ReleaseSubscribe before the dest is promoted + delivered (demote-before-promote)"
        );
        assert_eq!(
            rig.live(),
            1,
            "still live in Demoting, awaiting the demote-ack"
        );

        // The source acks DemoteAck (proof-of-freeze) → Promoting + the ordered Promote pushed to
        // the DEST. The demote-ack ALONE advances (breaking R1) — release is NOT yet due.
        rig.ack(TransferControlAck::DemoteAck { transfer: XFER });
        assert_eq!(
            rig.drain_dest(),
            vec![promote_wire(new_fence)],
            "the demote-ack advances to Promoting and pushes the Promote to the dest"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "promote pushed; the source sub is still held (seamless overlap)"
        );
        assert_eq!(rig.live(), 1);

        // PromoteAck alone holds; DeliveredToObservers completes the seamless gate → ReleaseSubscribe.
        rig.ack(TransferControlAck::PromoteAck { transfer: XFER });
        assert!(
            rig.drain_gateway().is_empty(),
            "promote-ack alone does not release — the delivery half is still pending"
        );
        rig.ack(TransferControlAck::DeliveredToObservers { transfer: XFER });
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::ReleaseSubscribe {
                transfer: XFER,
                session: SESSION,
                src: SOURCE,
            })],
            "promote-ack AND delivery release the source sub (the seamless gate)"
        );
        assert_eq!(rig.live(), 1);

        // Released → Done → Tombstone: the tail closes, live()→0 with the subject settled at DEST.
        rig.ack(TransferControlAck::Released { transfer: XFER });
        assert_eq!(
            rig.live(),
            0,
            "the ordered demote/promote/release tail reaches Done"
        );
    }

    #[test]
    fn early_delivery_in_demoting_is_carried_into_promoting_and_never_lost() {
        // 1d.5b.1 race fix: the gateway's standing watermark (`DeliveredToObservers`) can arrive
        // while the saga is STILL Demoting — the dest is already delivering from its autonomous
        // adopt-promote, before the source's Demote round-trip completes. The FSM LATCHES it in
        // `Demoting.dest_delivered` and carries it into `Promoting`, so the release gate can never
        // lose it (no park). This drives the full Demoting→Promoting→Releasing→Done tail end-to-end.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        let new_fence = Fence(1).next();
        // Inject a live saga already in Demoting (the post-RouteSwapped entry state).
        {
            let c = ctx(DurabilityClass::Durable, Fence(1));
            let mut runtime = rig.orch.world_mut().resource_mut::<super::SagaRuntimeRes>();
            runtime.sagas.insert(
                XFER,
                LiveSaga {
                    ctx: c,
                    state: SagaState::Demoting {
                        new_fence,
                        dest_delivered: false,
                    },
                    gateway: GATEWAY,
                    since: UniverseTick(0),
                    flushed_pose: None,
                },
            );
        }
        // Delivery arrives FIRST, while still Demoting: latched, NO Promote yet (awaiting the
        // demote-ack — demote-before-promote), NO release.
        rig.ack(TransferControlAck::DeliveredToObservers { transfer: XFER });
        assert!(
            rig.drain_dest().is_empty(),
            "no Promote while still Demoting — the demote-ack has not landed"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "no release while still Demoting"
        );
        assert_eq!(rig.live(), 1);

        // DemoteAck → Promoting (carrying the latched delivery) + the ordered Promote to the dest.
        rig.ack(TransferControlAck::DemoteAck { transfer: XFER });
        assert_eq!(
            rig.drain_dest(),
            vec![promote_wire(new_fence)],
            "the demote-ack advances to Promoting and pushes the Promote"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "delivery already counted, but the promote-ack is still pending"
        );

        // PromoteAck ALONE now completes the gate (the early delivery was carried forward) →
        // ReleaseSubscribe — proving the watermark was never lost across the Demoting→Promoting edge.
        rig.ack(TransferControlAck::PromoteAck { transfer: XFER });
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::ReleaseSubscribe {
                transfer: XFER,
                session: SESSION,
                src: SOURCE,
            })],
            "the carried delivery + the promote-ack release the source sub"
        );
        rig.ack(TransferControlAck::Released { transfer: XFER });
        assert_eq!(rig.live(), 0, "the ordered tail reached Done");
    }

    #[test]
    fn prepare_rejection_aborts_and_tombstones_with_typed_feedback() {
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle();
        let _ = rig.drain_gateway();

        let reject = PrepareReject::Spatial(SpatialReject::Obstructed);
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Rejected(reject),
        });
        // The dest is torn down; the typed rejection is recorded for the client (Slice 1c).
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::AbortTransfer {
                transfer: XFER,
                session: SESSION,
            })]
        );
        assert_eq!(
            rig.orch.world_mut().resource::<SagaRuntimeRes>().rejected,
            vec![(XFER, AbortReason::PrepareRejected(reject))]
        );

        // DestAborted → terminal Aborted → tombstoned (GC'd). The player stayed on the source.
        rig.ack(TransferControlAck::Aborted { transfer: XFER });
        assert_eq!(rig.live(), 0, "terminal saga is tombstoned");
        // The subject's authority never moved (the CAS never ran): source still owns it.
        let head = rig.subject_head();
        assert_eq!(head.authority, AuthorityRef::Shard(SOURCE));
        // ✅ FLIPPED (Slice 2a, D-1): the terminal Aborted edge emits ClearTransferLock →
        // abort_clear, so the lock CLEARS — the subject can immediately re-transfer (no wedge).
        assert_eq!(
            head.in_transfer, None,
            "Slice 2a: terminal abort clears the directory lock (D-1 closed)"
        );
    }

    #[test]
    fn cas_loss_unwinds_with_a_thaw() {
        // The saga expects Fence(1) but the directory moved to Fence(5) (someone else won) →
        // the DIRECT commit_cas LOSES → abort-with-thaw fires (the frozen source must thaw).
        let mut rig = Rig::new();
        rig.grant_subject(Fence(5));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1))); // stale expectation
        rig.settle();
        let _ = rig.drain_gateway();
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 9,
        });
        let _ = rig.drain_gateway();

        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 9,
        });
        rig.flush(); // gate: both freeze + flush, so the CAS actually runs (and loses)
        // A lost CAS unwinds with the compensator chain: ThawSource then AbortTransfer.
        assert_eq!(
            rig.drain_gateway(),
            vec![
                saga_wire(TransferControl::ThawSource {
                    transfer: XFER,
                    session: SESSION,
                }),
                saga_wire(TransferControl::AbortTransfer {
                    transfer: XFER,
                    session: SESSION,
                }),
            ]
        );
        // The authority never moved (CAS lost); the dest is being torn down + source thawed.
        assert_eq!(rig.subject_head().fence, Fence(5));
        rig.ack(TransferControlAck::SourceThawed { transfer: XFER });
        rig.ack(TransferControlAck::Aborted { transfer: XFER });
        assert_eq!(rig.live(), 0);
        // ✅ FLIPPED (Slice 2a, D-1, audit CPO-2): even the CAS-LOSER's terminal abort clears the
        // lock — `abort_clear` RE-READS the head (this saga's expected fence is stale by definition,
        // so a naive `abort_cas(expected)` would lose too), bumping the fence + clearing the lock.
        let head = rig.subject_head();
        assert_eq!(
            head.in_transfer, None,
            "Slice 2a: the CAS-loser's terminal abort clears the lock via the head re-read (D-1)"
        );
        assert_eq!(
            head.fence,
            Fence(6),
            "abort_clear bumped the re-read head fence (5 → 6) to fence-out any stale crossing"
        );
    }

    #[test]
    fn transient_subject_commits_via_the_go_token_not_a_cas() {
        // HR2 SHORT PATH (D-7): a Transient subject is NOT in the directory (no `grant_subject` —
        // burst isolation), takes NO lock, SKIPS Prepare/Cut/Freeze, and commits the batched
        // go-token at START → reaches Done + tombstones the SAME tick (`live()==0`). It records
        // EXACTLY ONE go-token (G-TIER: one write per batch, never per item), NEVER a per-entity CAS,
        // NEVER a `CommitAuthority`. The dest's `BatchAdopted` then drives the D-7b structural
        // drop-before-promote (release source → promote dest → ReleaseComplete). (The OLD
        // `IssueTransientGo` stub PARKED here in `CommittingCas` with the key locked — D-7 closes that.)
        let mut rig = Rig::new();
        rig.trigger(transient_ctx(Fence(9)));
        rig.settle(); // process_starts: no lock (Transient) → start → BatchCommitting → go-token → CasWon → Done

        assert_eq!(
            rig.live(),
            0,
            "the short-path transient saga reaches Done at start — no park"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "a transient drives NO gateway route-swap (no PrepareSubscribe, no CommitAuthority)"
        );
        // Exactly ONE go-token, at the dest realm-lease fence (G-TIER: one orchestrator write per batch).
        assert_eq!(
            rig.orch
                .world_mut()
                .resource::<SagaRuntimeRes>()
                .batch_goes(),
            vec![(BatchId(XFER), Fence(9))],
            "one batched go-token recorded at the commit fence"
        );

        // The D-7b structural drop-before-promote, driven through the orchestrator (two ordered
        // round-trips). PHASE 1 — the DEST's BatchAdopted → the orchestrator tells the SOURCE ONLY to
        // RELEASE (not a broadcast); the dest gets NOTHING yet (gated on the source's DropApplied).
        let _ = (rig.drain_source(), rig.drain_dest());
        rig.batch_adopted(XFER);
        assert_eq!(
            rig.drain_source(),
            vec![flow_inbound(&handoff(
                InterShardFlow::TransientRelease,
                TRANSIENT_RELEASE_STEP,
                Fence(9)
            ))],
            "the source is told to RELEASE (Held→Departing) — adopt-before-drop phase 1"
        );
        assert!(
            rig.drain_dest().is_empty(),
            "the dest is NOT promoted until the source releases (no both-held window)"
        );

        // PHASE 2 — the source's DropApplied(RELEASE) → the orchestrator PROMOTES the DEST only.
        rig.drop_applied(TRANSIENT_RELEASE_STEP);
        assert_eq!(
            rig.drain_dest(),
            vec![flow_inbound(&handoff(
                InterShardFlow::TransientDrop,
                TRANSIENT_DROP_STEP,
                Fence(9)
            ))],
            "the dest is told to PROMOTE (Arriving→Held) only after the source released"
        );
        assert!(
            rig.drain_source().is_empty(),
            "no source egress on the promote step"
        );

        // PHASE 3 — the dest's DropApplied(DROP, promote-confirm) → the orchestrator tells the SOURCE
        // to retire the retained Departing copy (ReleaseComplete).
        rig.drop_applied(TRANSIENT_DROP_STEP);
        assert_eq!(
            rig.drain_source(),
            vec![flow_inbound(&handoff(
                InterShardFlow::ReleaseComplete,
                TRANSIENT_RELEASE_STEP,
                Fence(9)
            ))],
            "the dest's promote-confirm retires the source's Departing copy"
        );
    }

    #[test]
    fn a_transient_handoff_ack_with_no_go_token_is_a_loud_noop() {
        // D-7 defensive: a `BatchAdopted` OR a `DropApplied` for a batch with NO committed go-token (a
        // stray/duplicate, or a batch that never started/already retired) emits NOTHING — a LOUD
        // no-op, never a silent drop and never a panic (covers the `None` arm of `with_batch_token`,
        // reached via both `handle_batch_adopted` and `handle_drop_applied`).
        let mut rig = Rig::new();
        let _ = (rig.drain_source(), rig.drain_dest());
        rig.batch_adopted(XFER); // no prior trigger → batch_goes is empty
        assert!(rig.drain_source().is_empty(), "no go-token ⇒ no release");
        assert!(rig.drain_dest().is_empty(), "no go-token ⇒ no promote");
        rig.drop_applied(TRANSIENT_RELEASE_STEP); // also no go-token → handle_drop_applied None arm
        assert!(
            rig.drain_source().is_empty(),
            "a DropApplied with no go-token is a loud no-op (no source egress)"
        );
        assert!(
            rig.drain_dest().is_empty(),
            "a DropApplied with no go-token is a loud no-op (no dest egress)"
        );
    }

    #[test]
    fn durable_saga_parked_in_committing_is_curl_visible() {
        // AAA-1: a parked saga is CURL-VISIBLE through the real admin snapshot — its transfer id, its
        // actual stuck phase, and a staleness anchor, never silently wedged. (Drives both
        // `SagaRuntimeRes::views` and `admin_snapshot`'s saga-population end to end.) D-7 moved this
        // OFF the old transient-park vehicle: a DURABLE saga whose direct CAS is starved (the granted
        // fence races ahead so the CAS would lose) is the wrong vehicle; instead we hold a durable
        // saga in `Demoting` awaiting its `DemoteAck` — a genuine post-commit park.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle();
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 3,
        });
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 3,
        });
        rig.flush(); // both gate conditions → the DIRECT CAS wins → Swapping
        rig.ack(TransferControlAck::Committed { transfer: XFER }); // → Demoting (parked, awaiting DemoteAck)
        let _ = (rig.drain_gateway(), rig.drain_source());
        assert_eq!(rig.live(), 1);

        let snap = crate::orchestrator::admin_snapshot(rig.orch.world_mut());
        assert_eq!(snap.sagas.len(), 1);
        assert_eq!(snap.sagas[0].transfer, XFER.to_string());
        // The operator sees the REAL parked phase with its fields (the post-CAS authority fence),
        // not a placeholder. Exact equality (no `assert!`-message format arm — HR5 discipline).
        assert_eq!(
            snap.sagas[0].state,
            "Demoting { new_fence: Fence(2), dest_delivered: false }"
        );
        // `since` was set when the saga last advanced and is bounded by the clock.
        assert!(snap.sagas[0].since.0 <= snap.universe_tick);
    }

    #[test]
    fn a_stray_ack_to_a_parked_saga_preserves_its_staleness_anchor() {
        // AAA-1-SINCE regression guard: a PARKED saga that receives a duplicate / stray ack the FSM
        // absorbs as same-state must KEEP its `since` — an unconditional bump would reset stuck-saga
        // staleness on every at-least-once redelivery. (Exercises the same-state arm of the
        // `final_state != prior` gate in `commit_result`.) D-7 moved this OFF the old transient
        // CommittingCas park (transients no longer park) onto a DURABLE saga held in `Demoting`
        // awaiting its `DemoteAck`: a stray duplicate `Committed` (route-swap) ack is absorbed there
        // as same-state (`RouteSwapped` is not handled in `Demoting`), the genuine post-commit park.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle();
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 3,
        });
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 3,
        });
        rig.flush(); // both gate conditions → the DIRECT CAS wins → Swapping
        rig.ack(TransferControlAck::Committed { transfer: XFER }); // → Demoting (parked, awaiting DemoteAck)
        let _ = (rig.drain_gateway(), rig.drain_source());
        let since_parked = crate::orchestrator::admin_snapshot(rig.orch.world_mut()).sagas[0].since;

        // Two duplicate Committed acks (at-least-once redelivery of the route-swap ack). The FSM
        // absorbs each as same-state in Demoting (RouteSwapped is not handled there); the clock
        // advances on every step. (Well within the redrive deadline, so the producer does not fire.)
        rig.ack(TransferControlAck::Committed { transfer: XFER });
        rig.ack(TransferControlAck::Committed { transfer: XFER });

        let snap = crate::orchestrator::admin_snapshot(rig.orch.world_mut());
        assert_eq!(
            snap.sagas.len(),
            1,
            "still parked in Demoting, not advanced by stray acks"
        );
        assert_eq!(
            snap.sagas[0].state, "Demoting { new_fence: Fence(2), dest_delivered: false }",
            "state unchanged by the stray acks"
        );
        assert_eq!(
            snap.sagas[0].since.0, since_parked.0,
            "the staleness anchor is NOT reset by a stray same-state ack"
        );
        assert!(
            snap.universe_tick > since_parked.0,
            "the clock DID advance — proving `since` was held, not re-stamped to `now`"
        );
    }

    #[test]
    fn active_transfers_reports_the_live_triple_and_is_empty_when_idle() {
        // 1d.5b.3d: the typed mid-flight ground truth the AUTHORITY-UNIQUE oracle excuses against —
        // each live saga's (subject, source, dest); empty once tombstoned (post-quiesce ⇒ strict).
        let mut rt = SagaRuntimeRes::default();
        assert!(rt.active_transfers().is_empty());
        rt.sagas.insert(
            XFER,
            LiveSaga {
                ctx: ctx(vd_core::entity_kind::DurabilityClass::Durable, Fence(1)),
                state: SagaState::Demoting {
                    new_fence: Fence(2),
                    dest_delivered: false,
                },
                gateway: GATEWAY,
                since: UniverseTick(0),
                flushed_pose: None,
            },
        );
        assert_eq!(
            rt.active_transfers(),
            vec![ActiveTransfer {
                subject: subject(),
                source: SOURCE,
                dest: DEST,
            }],
        );
    }

    #[test]
    fn deadline_for_maps_every_phase_to_its_risk_class() {
        // Slice 2a: pre-freeze/freeze/aborting → the LARGE destructive abort deadline; committing/
        // post-commit → the cheap redrive deadline; terminal → u64::MAX (never fires).
        let t = SagaTuning {
            redrive_deadline_ticks: 8,
            abort_deadline_ticks: 24,
        };
        for s in [
            SagaState::AwaitProvision,
            SagaState::Preparing,
            SagaState::Cutting,
            SagaState::Freezing {
                marker_seq: 0,
                frozen_drained: None,
                flushed: false,
            },
            SagaState::Aborting {
                reason: AbortReason::CutTimeout,
                awaiting_thaw: true,
                awaiting_abort_ack: true,
            },
        ] {
            assert_eq!(deadline_for(&s, &t), 24, "{s:?} uses the abort deadline");
        }
        for s in [
            SagaState::CommittingCas {
                marker_seq: 0,
                drained_seq: 0,
            },
            SagaState::Swapping {
                new_fence: Fence(2),
            },
            SagaState::Demoting {
                new_fence: Fence(2),
                dest_delivered: false,
            },
            SagaState::Promoting {
                new_fence: Fence(2),
                promote_acked: false,
                dest_delivered: false,
            },
            SagaState::Releasing {
                new_fence: Fence(2),
            },
        ] {
            assert_eq!(deadline_for(&s, &t), 8, "{s:?} uses the redrive deadline");
        }
        assert_eq!(
            deadline_for(
                &SagaState::Done {
                    new_fence: Fence(2)
                },
                &t
            ),
            u64::MAX
        );
        assert_eq!(
            deadline_for(
                &SagaState::Aborted {
                    reason: AbortReason::CutTimeout
                },
                &t
            ),
            u64::MAX
        );
    }

    /// Inject a live saga directly (controlled state + `since`) for the producer tests.
    fn inject_saga(runtime: &mut SagaRuntimeRes, state: SagaState, since: UniverseTick) {
        runtime.sagas.insert(
            XFER,
            LiveSaga {
                ctx: ctx(DurabilityClass::Durable, Fence(1)),
                state,
                gateway: GATEWAY,
                since,
                flushed_pose: None,
            },
        );
    }

    fn flows_to_node(outbox: &OutboundBox, node: NodeId) -> Vec<InterShardFlow> {
        outbox
            .0
            .iter()
            .filter(|(to, _, _)| *to == node)
            .filter_map(|(_, _, b)| postcard::from_bytes(b).ok())
            .collect()
    }

    #[test]
    fn scan_deadlines_re_drives_a_due_saga_re_arms_it_and_skips_a_fresh_one() {
        // THE R1 cure: a post-commit (Demoting) saga whose ack was lost re-drives at the redrive
        // deadline (8); the re-arm (since←now) means it fires at most ONCE per window, never every tick.
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive=8, abort=24
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
        });
        inject_saga(
            &mut runtime,
            SagaState::Demoting {
                new_fence: Fence(2),
                dest_delivered: false,
            },
            UniverseTick(0),
        );

        // BELOW the deadline (now=7 < 8): nothing re-driven.
        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(7),
        );
        assert!(outbox.0.is_empty(), "below the deadline: no re-drive");

        // AT the deadline (now=8): Timeout → Demoting re-emits the Demote to the SOURCE.
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8),
        );
        let expected = InterShardFlow::Demote(DemoteCmd {
            transfer: XFER,
            subject: subject(),
            new_owner_fence: Fence(2),
            step_id: DEMOTE_STEP,
        });
        assert!(
            flows_to_node(&outbox, SOURCE).contains(&expected),
            "the deadline producer re-drove the Demote: {:?}",
            outbox.0
        );

        // RE-ARM: `since` refreshed to 8, so the next scan within the window (now=9, 9-8=1 < 8) is silent.
        let mut outbox2 = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox2,
            EpochId(1),
            UniverseTick(9),
        );
        assert!(
            outbox2.0.is_empty(),
            "within the re-armed window: no second re-drive (no per-tick storm)"
        );
    }

    #[test]
    fn scan_deadlines_re_drives_a_parked_aborting_saga() {
        // finding #6: a saga PARKED in Aborting (a dropped compensator ack) is re-driven by the
        // producer — at the LARGE abort deadline (24) — re-emitting BOTH outstanding compensators.
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
        });
        inject_saga(
            &mut runtime,
            SagaState::Aborting {
                reason: AbortReason::FreezeTimeout,
                awaiting_thaw: true,
                awaiting_abort_ack: true,
            },
            UniverseTick(0),
        );

        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(23),
        );
        assert!(outbox.0.is_empty(), "below the abort deadline: no re-drive");

        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(24),
        );
        let to_gateway = flows_to_node(&outbox, GATEWAY);
        assert!(
            to_gateway.contains(&InterShardFlow::Saga(TransferControl::ThawSource {
                transfer: XFER,
                session: SESSION,
            })),
            "the producer re-drove ThawSource: {to_gateway:?}"
        );
        assert!(
            to_gateway.contains(&InterShardFlow::Saga(TransferControl::AbortTransfer {
                transfer: XFER,
                session: SESSION,
            })),
            "the producer re-drove AbortTransfer: {to_gateway:?}"
        );
    }

    #[test]
    fn trigger_on_an_unrecorded_subject_refuses_to_start() {
        // Per-key serialization: no directory record → lock_transfer FALSE → no saga created.
        let mut rig = Rig::new();
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle();
        assert_eq!(rig.live(), 0, "an unrecorded subject cannot start a saga");
        assert!(rig.drain_gateway().is_empty());
    }

    #[test]
    fn an_ack_for_an_unknown_saga_is_an_idempotent_noop() {
        // A duplicate/stale ack after GC (or for a never-started transfer) is a no-op.
        let mut rig = Rig::new();
        rig.ack(TransferControlAck::Committed {
            transfer: TransferId(999),
        });
        assert_eq!(rig.live(), 0);
        assert!(rig.drain_gateway().is_empty());
    }

    #[test]
    fn ack_to_event_maps_every_ack_phase() {
        let cases: [(TransferControlAck, SagaEvent); 10] = [
            (
                TransferControlAck::Prepared {
                    transfer: XFER,
                    result: PrepareResult::Ready,
                },
                SagaEvent::Prepared(PrepareResult::Ready),
            ),
            (
                TransferControlAck::CutConfirmed {
                    transfer: XFER,
                    marker_seq: 1,
                },
                SagaEvent::CutConfirmed { marker_seq: 1 },
            ),
            (
                TransferControlAck::SourceFrozen {
                    transfer: XFER,
                    drained_seq: 2,
                },
                SagaEvent::SourceFrozen { drained_seq: 2 },
            ),
            (
                TransferControlAck::Committed { transfer: XFER },
                SagaEvent::RouteSwapped,
            ),
            (
                TransferControlAck::SourceThawed { transfer: XFER },
                SagaEvent::SourceThawed,
            ),
            (
                TransferControlAck::Aborted { transfer: XFER },
                SagaEvent::DestAborted,
            ),
            (
                TransferControlAck::Released { transfer: XFER },
                SagaEvent::Released,
            ),
            (
                TransferControlAck::DemoteAck { transfer: XFER },
                SagaEvent::DemoteAcked,
            ),
            (
                TransferControlAck::PromoteAck { transfer: XFER },
                SagaEvent::PromoteAcked,
            ),
            (
                TransferControlAck::DeliveredToObservers { transfer: XFER },
                SagaEvent::DestDelivered,
            ),
        ];
        for (ack, event) in cases {
            assert_eq!(ack_to_event(ack), event);
        }
    }

    #[test]
    fn the_rejection_ledger_is_bounded_with_a_counted_drop() {
        // ROB-1: the un-drained ledger can NEVER grow without limit — beyond the cap the
        // oldest rejections are shed, counted (the loud overflow ALERT), oldest-first.
        let mut runtime = SagaRuntimeRes::default();
        for i in 0..(REJECTION_LEDGER_CAP as u128 + 5) {
            runtime
                .rejected
                .push((TransferId(i), AbortReason::Cancelled));
        }
        bound_rejection_ledger(&mut runtime);
        assert_eq!(runtime.rejected.len(), REJECTION_LEDGER_CAP);
        assert_eq!(runtime.rejections_dropped, 5);
        assert_eq!(
            runtime.rejected.first().expect("non-empty").0,
            TransferId(5),
            "the 5 OLDEST were shed; the newest survive"
        );
    }

    #[test]
    fn emit_crossing_builds_for_an_entity_subject_and_skips_otherwise() {
        let c = ctx(DurabilityClass::Durable, Fence(1));

        // Entity subject + a flushed pose → the crossing is pushed to the DEST, stamped with the
        // given fence, STUB_CROSSING_STEP, and an empty (1d.1) state blob.
        let mut outbox = OutboundBox::default();
        emit_crossing(&c, Fence(2), Some(flushed_pose()), EpochId(1), &mut outbox);
        assert_eq!(
            outbox.0.len(),
            1,
            "an Entity subject with a pose emits a crossing"
        );
        let (to, class, bytes) = &outbox.0[0];
        assert_eq!(*to, DEST);
        assert_eq!(*class, MsgClass::Saga);
        assert_eq!(
            postcard::from_bytes::<InterShardFlow>(bytes).expect("decode"),
            InterShardFlow::Transfer(TransferEnvelope {
                transfer_id: XFER,
                universe_epoch: EpochId(1),
                schema_version: TRANSFER_SCHEMA_VERSION,
                fence: Fence(2),
                step_id: STUB_CROSSING_STEP,
                class: DurabilityClass::Durable,
                payload: TransitionPayload::StubCrossing {
                    entity: subject_eid(),
                    from_realm: FROM_REALM,
                    to_realm: TO_REALM,
                    pose: flushed_pose(),
                    state: vec![],
                },
            })
        );

        // No stashed pose → no-op (the gate makes this unreachable for an Entity subject, but the
        // helper is total — this covers the missing-pose None arm).
        let mut no_pose = OutboundBox::default();
        emit_crossing(&c, Fence(2), None, EpochId(1), &mut no_pose);
        assert!(no_pose.0.is_empty(), "no pose → no crossing");

        // Non-Entity subject (the Realm-subject FSM proptests) → no-op.
        let realm_ctx = SagaCtx {
            subject: DirectoryKey::Realm(RealmId::System(9)),
            ..c
        };
        let mut realm_out = OutboundBox::default();
        emit_crossing(
            &realm_ctx,
            Fence(2),
            Some(flushed_pose()),
            EpochId(1),
            &mut realm_out,
        );
        assert!(
            realm_out.0.is_empty(),
            "a non-Entity subject emits no crossing"
        );
    }

    #[test]
    fn a_flush_for_an_unknown_saga_is_a_noop() {
        // stash_flush's None arm: a SourceFlushed for an unknown / GC'd transfer mutates nothing.
        let mut runtime = SagaRuntimeRes::default();
        stash_flush(&mut runtime, TransferId(999), flushed_pose());
        assert_eq!(runtime.live(), 0, "a stray flush creates/mutates no saga");
    }

    #[test]
    fn a_dest_crossing_ack_and_a_non_saga_arm_are_dropped() {
        // drive_sagas decodes but DROPS the DEST's crossing ack (Accepted) and any non-saga-driving
        // arm (here a Saga command, which the orchestrator only ever SENDS): no saga starts,
        // nothing is emitted.
        let mut rig = Rig::new();
        for flow in [
            InterShardFlow::TransferAck(TransferAck::Accepted {
                transfer_id: XFER,
                step_id: STUB_CROSSING_STEP,
            }),
            InterShardFlow::Saga(TransferControl::RequestCut {
                transfer: XFER,
                session: SESSION,
            }),
        ] {
            rig.gateway
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(postcard::to_allocvec(&flow).expect("encode")),
                )
                .expect("sent");
        }
        rig.settle();
        assert_eq!(
            rig.live(),
            0,
            "neither a crossing ack nor a stray arm starts a saga"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "nothing is emitted in response"
        );
    }
}
