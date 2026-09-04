//! THE ACTION EXECUTOR: run one saga to quiescence, then commit what it became.
//!
//! Owns: the loop that takes `(state, actions)` from the pure FSM, performs each action against the
//! world, feeds any synchronous result back in, and stops when nothing more is producible — plus the
//! persistence of the quiescent result and the garbage collection of a terminal one.
//!
//! Does NOT own: any decision. Every branch about WHAT to do next belongs to the pure FSM; if a
//! choice appears in this file, the FSM has been bypassed and the crash-replay proof no longer
//! covers it.

use super::{
    BatchGo, GoTokenSnapshot, PendingAbortReply, REJECTION_LEDGER_CAP, SagaRuntimeRes,
    SagaSnapshot, StoreKey, emit_crossing, emit_rehome, encode,
};
use std::collections::VecDeque;
use vd_core::pose::StampedPose;
use vd_core::{BatchId, EpochId, NodeId, TransferId, UniverseTick};
use vd_sim::directory::DirectoryCore;
use vd_sim::io::MsgClass;
use vd_sim::runtime::OutboundBox;
use vd_sim::saga::{self, AbortReason, SagaAction, SagaCtx, SagaEvent, SagaState};
use vd_wire::intershard::{
    DEMOTE_STEP, DemoteCmd, FLUSH_SOURCE_STEP, FlushSource, InterShardFlow, PROMOTE_STEP,
    PromoteCmd, RE_SOLICIT_STEP, TRANSIENT_ABANDON_STEP, TRANSIENT_DISCARD_STEP,
    TRANSIENT_DROP_STEP, TRANSIENT_RELEASE_STEP, TransientHandoff,
};
use vd_wire::seams::directory::{AuthorityRef, CasOutcome, DirectoryKey};
use vd_wire::seams::transfer_control::TransferControlAck;

/// Map a gateway→saga ack to the FSM event it drives. `CasWon`/`CasLost` are NOT here — they
/// come from the DIRECT `commit_cas`, never the wire.
pub(crate) fn ack_to_event(ack: TransferControlAck) -> SagaEvent {
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
pub(crate) fn run_to_quiescence(
    ctx: &SagaCtx,
    gateway: NodeId,
    mut state: SagaState,
    mut actions: Vec<SagaAction>,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
    flush_pose: Option<StampedPose>,
    flush_state: &[u8],
) -> (
    SagaState,
    bool,
    Vec<(TransferId, AbortReason)>,
    Vec<BatchGo>,
    bool,
) {
    let mut events: VecDeque<SagaEvent> = VecDeque::new();
    let mut tombstone = false;
    let mut rejected = Vec::new();
    let mut batch_gos = Vec::new();
    // ★ WHETHER THE DIRECTORY COMMIT WON INSIDE THIS STEP. The win is fed back as an event right
    // here, never through the caller, so a caller that wants to act on it (the exterior's reparent
    // note) must be told. MEASURED on the fifth flight (2026-09-03): the note was taken only from the
    // caller's own event, the hull's hand-down to its star committed without one, the reconciler
    // never re-keyed it, the gateway was never told, and the pilot's picture froze.
    let mut cas_won = false;
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
                            // The SOURCE rebases the flushed pose into this dest realm's live frame before
                            // shipping it (the moving-realm crossing fix); the saga already holds the dest.
                            to_realm: ctx.to_realm,
                            to_parent: ctx.to_parent,
                        }),
                    );
                }
                // Emit the entity-STATE crossing to the DEST (1d.1) from the stashed pose. The
                // Some/None branch lives in `emit_crossing` (a monomorphic helper, unit-tested both
                // ways), so this arm stays a branchless dispatch (HR5).
                SagaAction::EmitCrossing { fence } => {
                    emit_crossing(ctx, fence, flush_pose, flush_state, epoch, outbox);
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
                        CasOutcome::Won { new_fence } => {
                            cas_won = true;
                            SagaEvent::CasWon { new_fence }
                        }
                        CasOutcome::Lost { current } => SagaEvent::CasLost { current },
                    });
                }
                // D-37 forward re-home commit — the SAME single commit point (D-32 routing caveat applies
                // identically), re-pointed at the live `target` rather than the hardcoded `ctx.dest`. The
                // `cas_next` bump (`expected` → `expected+1`) atomically names `target`, clears the lock,
                // and strictly stales the dead owner's claim (the fence-monotone no-double-owner property).
                // Feeds `CasWon`/`CasLost` back exactly like `IssueCommitCas` (FF-1: one feedback event).
                SagaAction::ReHomeCommit { expected, target } => {
                    let outcome =
                        dir.commit_cas(ctx.subject, expected, AuthorityRef::Shard(target), now);
                    events.push_back(match outcome {
                        CasOutcome::Won { new_fence } => {
                            cas_won = true;
                            SagaEvent::CasWon { new_fence }
                        }
                        CasOutcome::Lost { current } => SagaEvent::CasLost { current },
                    });
                }
                // D-37 forward re-home adopt — emit the dedicated `ReHome` to `target` from the stashed
                // pose. The Some/None branch lives in `emit_rehome` (unit-tested both ways), so this arm
                // stays a branchless dispatch (HR5). Pure egress (the target's `PromoteAck` rides back).
                SagaAction::ReHomeAdopt { new_fence, target } => {
                    emit_rehome(ctx, new_fence, target, flush_pose, epoch, outbox);
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
                // D-7d transient post-commit tail egress (replacing the deleted ledger-driven
                // handle_batch_adopted/handle_drop_applied): pure egress to ctx.source/ctx.dest carrying
                // the committed go-token `fence`. The shard journals each by (transfer, step), so a
                // Timeout re-emit (the producer re-drive) is idempotent. FF-1: no synchronous feedback —
                // the shard's ack rides TransferAck back, routed to the FSM in `drive_sagas`.
                SagaAction::EmitTransientRelease { fence } => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::TransientRelease(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_RELEASE_STEP,
                            fence,
                        }),
                    );
                }
                SagaAction::EmitTransientPromote { fence } => {
                    outbox.push_flow(
                        ctx.dest,
                        MsgClass::Saga,
                        &InterShardFlow::TransientDrop(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_DROP_STEP,
                            fence,
                        }),
                    );
                }
                SagaAction::EmitReleaseComplete { fence } => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::ReleaseComplete(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_RELEASE_STEP,
                            fence,
                        }),
                    );
                }
                // D-7d dead-DEST resolution: the source ABANDONS the batch as an accounted loss (the
                // dest died, so the promote target is gone). Pure egress to ctx.source carrying the
                // go-token fence; the source journals `(transfer, TRANSIENT_ABANDON_STEP)`. FF-1.
                SagaAction::EmitTransientAbandon { fence } => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::TransientAbandon(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_ABANDON_STEP,
                            fence,
                        }),
                    );
                }
                // R-6d3c NEVER-restart resolution: the source died in AwaitAdopt (pre-adopt), so tell the
                // DEST to DISCARD any late-replayed Arriving copy + poison its adopt. Pure egress to
                // ctx.dest carrying the go-token fence; the dest journals `(transfer, TRANSIENT_DISCARD_
                // STEP)`. Ack-FREE — the resolving saga is terminal (mirroring the abandon egress).
                SagaAction::EmitTransientDiscard { fence } => {
                    outbox.push_flow(
                        ctx.dest,
                        MsgClass::Saga,
                        &InterShardFlow::TransientDiscard(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_DISCARD_STEP,
                            fence,
                        }),
                    );
                }
                // R-6d3c — the batch-lost-source-crash count is applied in `scan_deadlines` PRE-delivery
                // (beside `source_unreachable_resolutions`), so this executor arm is a straight-line no-op.
                SagaAction::CountBatchLostSourceCrash => {}
                // CA-1 S3 — the AwaitAdopt liveness PROBE to the SOURCE. Pure egress to ctx.source carrying
                // the go-token fence, keyed `(transfer, RE_SOLICIT_STEP)`. Its SEND outcome is the signal: a
                // dead source's send fails → `NodeUnreachable` → `is_confirmed_dead(source)` → the pre-adopt
                // discard becomes reachable. A live source no-ops it. Re-driven every AwaitAdopt Timeout.
                SagaAction::EmitReSolicit { fence } => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::ReSolicitBatch(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: RE_SOLICIT_STEP,
                            fence,
                        }),
                    );
                }
                // D-6: durability is REAL now — `commit_result` stages the QUIESCENT saga snapshot at
                // EVERY transition (this checkpoint's `Send`/`EmitCrossing` effect included), and the
                // end-of-tick `Store::commit()` barrier flushes it BEFORE the node's flush phase sends
                // the tick's effects (persist-before-effect). So the two emit sites (dest-ghost-exists,
                // authority-flipped) need no per-action work — the write-back + barrier subsume them.
                SagaAction::PersistCheckpoint => {}
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
    (state, tombstone, rejected, batch_gos, cas_won)
}

/// Apply one event to an existing saga, then run it to quiescence + persist the result. A
/// stale event for an unknown/GC'd saga is an idempotent no-op (at-least-once delivery).
pub(crate) fn deliver(
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
    let flush_state = live.flushed_state.clone();
    // The ruler switch, slice 4: an EXTERIOR's CAS win is the moment the child moved house — the
    // reconciler re-keys its cell and its launch record from this note. The win arrives either as
    // this step's own event (a wire CAS) or from inside the quiescence run (the in-process CAS —
    // the shipped path, missed until the fifth flight measured it).
    let won_outside = matches!(event, SagaEvent::CasWon { .. });
    let (state, actions) = saga::step(&ctx, live.state, event);
    let (final_state, tombstone, rejected, batch_gos, won_inside) = run_to_quiescence(
        &ctx,
        gateway,
        state,
        actions,
        dir,
        outbox,
        epoch,
        now,
        flush_pose,
        &flush_state,
    );
    let exterior_committed = ctx.exterior & (won_outside | won_inside);
    if exterior_committed && let DirectoryKey::Ship(entity) = ctx.subject {
        runtime.pending_reparents.push((
            vd_core::pose::RealmId::Ship(entity),
            ctx.to_realm,
            ctx.dest,
        ));
    }
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
pub(crate) fn bound_rejection_ledger(runtime: &mut SagaRuntimeRes) {
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

/// Slice 3f-D: is this saga a CROSSING-ORIGIN transfer (a geometric boundary crossing, id-namespaced
/// `0x39` in the `TransferId`'s high byte by [`crossing_transfer_id`])? A STATELESS TAG-CHECK on the id —
/// `0x39` is EXCLUSIVE of the re-home namespace (`0x37`) and the connection-plane's small high-byte-`0x00`
/// ids (see `namespaced_transfer_id`), so the discriminator needs no `SagaSnapshot` field, no version bump,
/// and no id recompute. A branchless monomorphic expression (HR5). Only a crossing-origin durable saga that
/// tombstones ABORTED owes a `PendingAbortReply` (Mechanism Y); every other tombstone is inert here.
#[must_use]
pub(crate) fn is_crossing_origin(ctx: &SagaCtx) -> bool {
    (ctx.transfer.0 >> 120) == 0x39
}

/// Persist a quiescent saga: record rejections + the batched go-tokens, then GC (terminal) or write
/// back the state.
pub(crate) fn commit_result(
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
        // D-7c: count the WRITE (the G-TIER observable) BEFORE the idempotent `or_insert` — so a
        // per-item-write regression inflates this to N even though `or_insert` keeps `len` at 1.
        runtime.batch_go_writes += 1;
        runtime
            .batch_goes
            .entry(bg.batch)
            .or_insert((bg.fence, bg.source, bg.dest));
        // D-6: STAGE the go-token durably (self-describing; one record per batch — preserves the
        // G-TIER decouple, since rehydrate restores `batch_go_writes` from the map len, never re-driving
        // the `+= 1`). Committed at the end-of-tick barrier with the saga snapshots.
        let token = GoTokenSnapshot {
            batch: bg.batch,
            fence: bg.fence,
            source: bg.source,
            dest: bg.dest,
        };
        runtime
            .pending_writes
            .push((StoreKey::BatchGo(bg.batch).bytes(), Some(encode(&token))));
    }
    if tombstone {
        // Slice 3f-D (Mechanism Y): a CROSSING-ORIGIN (`0x39`) DURABLE saga that tombstoned ABORTED (a
        // pre-CAS failure — source authority never handed off) owes the SOURCE a positive latch-clear, else
        // a lost RAM enqueue strands the entity's `RequestInFlight` latch. Read the live saga's ctx BEFORE
        // the `remove` (the source/subject to reply to), gated on the stateless id tag-check
        // (`is_crossing_origin`) AND the terminal being `Aborted` AND `Durable` class. `.get(..).expect(..)`
        // (NOT `if let Some`) — the saga is GUARANTEED present here (it was live when `run_to_quiescence`
        // produced this terminal, and this single-threaded schedule touched nothing in between), matching the
        // else-branch's `.expect` at the write-back below — so there is no uncoverable `None` region (HR5).
        let aborted = matches!(final_state, SagaState::Aborted { .. });
        let live = runtime
            .sagas
            .get(&transfer)
            .expect("the saga was present at the start of this run");
        let crossing_origin = is_crossing_origin(&live.ctx);
        let durable = live.ctx.class == vd_core::entity_kind::DurabilityClass::Durable;
        if aborted && crossing_origin && durable {
            let reply = PendingAbortReply {
                transfer,
                source: live.ctx.source,
                subject: live.ctx.subject,
            };
            runtime.pending_abort_replies.insert(transfer, reply);
            // Persist (self-describing). Committed at the SAME end-of-tick group-commit barrier as the
            // saga-snapshot DELETE below and the directory `abort_clear` (all ride `pending_writes` /
            // `dirty`), so the abort reply + the tombstone are ATOMIC post-crash — a restart never sees one
            // without the other. The FIRST `CrossingAborted` emit is left to `scan_deadlines` (which owns the
            // outbox + the throttle); Mechanism Y IS "scan re-emits per entry until acked", so a scan-driven
            // first emit is the design (and the first scan sees `last_abort_reply_emit == 0` → emits promptly).
            runtime
                .pending_writes
                .push((StoreKey::AbortReply(transfer).bytes(), Some(encode(&reply))));
        }
        runtime.sagas.remove(&transfer);
        // D-6: a tombstoned saga's durable snapshot is DELETED — rehydrate's Saga-scan must not
        // resurrect it (the matching directory mutation commits in the SAME barrier, so the durable
        // saga set + directory are always consistent post-crash). UNCONDITIONAL (never gated on the
        // crossing check) — every tombstone deletes its snapshot.
        runtime
            .pending_writes
            .push((StoreKey::Saga(transfer).bytes(), None));
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
        // D-6: STAGE the QUIESCENT snapshot (built by copying the just-updated live saga, so `live`'s
        // borrow ends before the disjoint `pending_writes` push — NLL). The end-of-tick barrier commits
        // it before the node's flush phase sends this tick's effects (persist-before-effect).
        let snapshot = SagaSnapshot {
            ctx: live.ctx,
            state: live.state,
            gateway: live.gateway,
            since: live.since,
            flushed_pose: live.flushed_pose,
            flushed_state: live.flushed_state.clone(),
        };
        runtime
            .pending_writes
            .push((StoreKey::Saga(transfer).bytes(), Some(encode(&snapshot))));
    }
}
