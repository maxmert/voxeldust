//! THE TIMEOUT PRODUCER: what a stalled phase turns into.
//!
//! Owns: the per-phase deadline, the dead-aware variant that shortens a wait once a participant is
//! confirmed gone rather than merely slow, and the per-tick scan that injects the timeout event into
//! every saga past its budget.
//!
//! Does NOT own: what a timeout MEANS. The event goes into the pure FSM and the FSM decides whether
//! this phase compensates, retries or aborts — which is why a stalled transfer can never be stranded
//! by a missing arm here.

use super::{PendingAbortReply, SagaRuntimeRes, StoreKey, deliver, rehome_event_for};
use vd_core::{EpochId, NodeId, TransferId, UniverseTick};
use vd_sim::capability::CapRequest;
use vd_sim::directory::DirectoryCore;
use vd_sim::io::MsgClass;
use vd_sim::runtime::OutboundBox;
use vd_sim::saga::{SagaEvent, SagaState, SagaTuning};
use vd_wire::intershard::{InterShardFlow, TRANSIENT_DROP_STEP, TRANSIENT_RELEASE_STEP};
use vd_wire::seams::directory::DirectoryKey;

/// The deadline a saga's CURRENT phase fires its `Timeout` at (Slice 2a). Pre-freeze/freeze/aborting
/// phases use the LARGE `abort_deadline_ticks` (their Timeout is DESTRUCTIVE — an abort — or re-drives
/// a compensator on the same slow latency profile); committing/post-commit phases use the small
/// `redrive_deadline_ticks` (a cheap idempotent re-drive that may fire "early" at zero correctness
/// cost); terminal phases return `u64::MAX` (never fire). A straight monomorphic match — the ONLY
/// per-phase logic, hoisted out of the phase-agnostic producer scan (HR3).
#[must_use]
pub(crate) fn deadline_for(state: &SagaState, tuning: &SagaTuning) -> u64 {
    match state {
        SagaState::AwaitProvision
        | SagaState::Preparing
        | SagaState::Cutting
        | SagaState::Freezing { .. }
        | SagaState::Aborting { .. } => tuning.abort_deadline_ticks,
        SagaState::CommittingCas { .. }
        | SagaState::BatchCommitting { .. }
        | SagaState::BatchHandoff { .. }
        | SagaState::Swapping { .. }
        | SagaState::Demoting { .. }
        | SagaState::Promoting { .. }
        // D-37: a CAS-in-flight state like CommittingCas. In-process the re-home CAS feedback is
        // synchronous (run_to_quiescence resolves CasWon/CasLost before the tick ends), so a saga never
        // PERSISTS in ReHoming and this deadline is not consulted at runtime; the redrive value is the
        // consistent CAS-in-flight choice (and the D-32 async-routed CAS re-drive home, if it ever persists).
        | SagaState::ReHoming { .. }
        | SagaState::Releasing { .. } => tuning.redrive_deadline_ticks,
        SagaState::Done { .. } | SagaState::Aborted { .. } => u64::MAX,
    }
}

/// The dead-aware deadline event for a due saga (D-7d transient hand-off + D-3 liveness discriminator +
/// D-37 durable re-home), hoisted out of [`scan_deadlines`] so the scan stays a branchless dispatch and
/// every branch is covered in ONE monomorphic place (HR5). A re-drive toward a CONFIRMED-DEAD participant
/// (`is_confirmed_dead` — the evidence-gated run, NOT a single blip) becomes a RESOLUTION instead of
/// looping at a corpse forever; a healthy slow peer (no kill ⇒ no NodeUnreachable run ⇒ never confirmed)
/// only ever re-drives via plain `Timeout`. SOURCE-dead is the cheap NON-destructive path (self-promote
/// the already-committed dest/batch — the demote toward the corpse is moot, its claim excluded by the
/// dead-aware oracle). DEST-dead is the DESTRUCTIVE budget gate: resolve only after `abort_deadline_ticks`
/// measured from `dead_observed_since` (the CSCALE-1 cure — a recoverable blip that clears in time never
/// resolves a healthy dest); `dead_observed_since` is the per-saga budget anchor, NOT `since` (which
/// re-armed to `now`), so the budget accrues across re-drives rather than resetting every fire.
/// R-6d3c — the DESTRUCTIVE-resolution budget elapsed for `node`, keyed by the confirmed-dead
/// PARTICIPANT so a CAUSE-SWITCH re-anchors. A single per-saga `Option<(NodeId, UniverseTick)>` times
/// "how long has THIS participant been observed dead"; if the anchor is unset OR held by a DIFFERENT node
/// (the dest was confirmed dead, then RECOVERED, then the source was confirmed dead) it RE-ANCHORS to
/// `now` — so the new participant's `abort_deadline_ticks` budget is never measured from the OTHER
/// participant's stale first-dead observation (the shared-anchor regression the R-6d3c review caught: a
/// destructive source resolution would otherwise fire before its own budget elapsed). Monomorphic — the
/// re-anchor branch is covered ONCE here so both call sites stay branchless (HR5).
pub(crate) fn dead_budget_elapsed(
    anchor: &mut Option<(NodeId, UniverseTick)>,
    node: NodeId,
    now: UniverseTick,
) -> u64 {
    let observed = match *anchor {
        Some((n, t)) if n == node => t,
        _ => {
            *anchor = Some((node, now));
            now
        }
    };
    now.0.saturating_sub(observed.0)
}

/// THE Slice 2a TIMEOUT PRODUCER (the R1 cure): inject `SagaEvent::Timeout` into every live saga whose
/// `now - since` reached its phase deadline, so a lost saga-ack RE-DRIVES (post-commit) or ABORTS
/// (pre-freeze) instead of PARKING forever. Phase-agnostic — ONE scan, ONE injected event; the FSM's
/// existing per-phase Timeout arms do all the work. `since` is REFRESHED on fire (BEFORE delivery), so
/// a still-stale saga re-fires at most once per its deadline window, never every tick (no storm).
/// O(live sagas)/tick — matches the existing `views()`/`active_transfers()` scans; a deadline-ordered
/// structure (pop only the due) is a P3/MMO-scale optimization, not built now.
pub(crate) fn scan_deadlines(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
) {
    let tuning = runtime.tuning;
    // Bound the re-home inputs ONCE outside the loop (BTreeMap is not Copy → shared ref; `req` is the
    // subject's required caps — empty for a P3 bare-point Entity, the D-31 KindDef seam, one home no
    // inline). `&runtime.roster` is a sound disjoint partial borrow alongside `runtime.sagas.iter_mut()`,
    // exactly like `&runtime.liveness` below.
    let roster = &runtime.roster;
    let req = CapRequest::default();
    let mut due: Vec<(TransferId, SagaEvent)> = Vec::new();
    for (transfer, live) in runtime.sagas.iter_mut() {
        if now.0.saturating_sub(live.since.0) >= deadline_for(&live.state, &tuning) {
            live.since = now; // re-arm BEFORE delivery — one fire per window, never a per-tick storm
            // The CURRENT directory owner of the subject (the re-home liveness check keys on THIS, not the
            // stale `ctx.dest`, so a re-home fires AT MOST once — the next fire sees the live target). The
            // `ctx.dest` fallback covers a (transient) subject with no directory record. `dir.head` is a
            // shared reborrow disjoint from `runtime.sagas.iter_mut()` (dir is a separate param).
            let subject_owner = dir
                .head(live.ctx.subject)
                .map_or(live.ctx.dest, |r| r.authority.node());
            // The forward-re-home target's ONLY lawful value (see `select_rehome_target`): the live
            // directory owner of the stashed pose's OWN realm — the node whose `place_arriving_pose`
            // accepts that frame. Resolved from the SAME fenced directory as every authority answer.
            let pose_realm_owner = live
                .flushed_pose
                .and_then(|p| p.frame.realm())
                .and_then(|realm| dir.head(DirectoryKey::Realm(realm)))
                .map(|r| r.authority.node());
            // The dead-aware deadline event: a re-drive toward a CONFIRMED-DEAD participant becomes a
            // RESOLUTION instead of looping at a corpse forever. ALL branching lives in the monomorphic
            // `rehome_event_for` helper (see its doc) so this scan stays a branchless dispatch (HR5). The
            // borrows are disjoint: `liveness`/`roster` are fields of `runtime` disjoint from `sagas` (a
            // sound partial borrow), and `state`/`ctx`/`dead_observed_since` are disjoint fields of `live`.
            let event = rehome_event_for(
                &live.state,
                &live.ctx,
                &runtime.liveness,
                &mut live.dead_observed_since,
                &tuning,
                now,
                roster,
                &req,
                subject_owner,
                pose_realm_owner,
                live.dest_adopted,
            );
            due.push((*transfer, event));
        }
    }
    for (transfer, event) in due {
        // Count the resolution (orchestrator-side anti-vacuity) BEFORE delivery; the resolution arm is
        // terminal-on-first-fire (the saga tombstones), so each resolved batch counts exactly once.
        match event {
            SagaEvent::SourceUnreachable => runtime.source_unreachable_resolutions += 1,
            SagaEvent::DestUnreachable => runtime.dest_unreachable_resolutions += 1,
            // R-6d3c: the never-restart PRE-adopt resolution (accounted loss, discard-to-dest). Counted
            // HERE — this `_ => {}` wildcard is the ONE place a missing counter arm would compile green
            // (unlike every other new SagaEvent/SagaAction, which the total FSM/executor match rejects).
            SagaEvent::SourceUnreachablePreAdopt => runtime.batch_lost_source_crash += 1,
            _ => {}
        }
        deliver(runtime, dir, outbox, epoch, now, transfer, event);
    }

    // Slice 3f-D (Mechanism Y): drive the durable crossing-abort replies — a SEPARATE pass (disjoint from
    // `sagas`), run AFTER the `due` drain. Per entry: (1) REAP when the subject NO LONGER belongs to
    // `source` (`dir.head(subject) != source`) — the abort is then MOOT: D-37 has re-homed the dead
    // source's entity, OR it transferred away, so the source's stale crossing latch is irrelevant. This is
    // the SAME "entity left the owner" fact D-37's re-home discharges, NOT the `is_confirmed_dead` PULSE
    // (which flips true on a transient blip AND EXPIRES on recovery — so it would reap a briefly-blipped-
    // then-RECOVERED source's live latch-clear obligation BEFORE re-home ever completes, stranding the
    // entity at the boundary; adversary-review HIGH). A recovered source still owns the subject → head
    // stays `source` → the reply keeps re-emitting until that source acks. (2) else RE-EMIT
    // `CrossingAborted` to the source, THROTTLED to `redrive_deadline_ticks` (cheap-idempotent cadence — an
    // alive-but-ack-stalled source must not draw a per-tick egress). The throttle is ONE runtime-level gate
    // (`last_abort_reply_emit`), so `PendingAbortReply` stays a 3-field record (no per-entry stamp, no ABA).
    // COLLECT-then-APPLY into Vecs so mutation happens after the shared borrow ends (determinism +
    // borrow-safety); `dir` is a disjoint param, so `dir.head` inside the `runtime`-values loop is sound.
    let reemit_due =
        now.0.saturating_sub(runtime.last_abort_reply_emit.0) >= tuning.redrive_deadline_ticks;
    let mut reaped: Vec<TransferId> = Vec::new();
    let mut to_emit: Vec<PendingAbortReply> = Vec::new();
    for reply in runtime.pending_abort_replies.values() {
        let owner = dir.head(reply.subject).map(|r| r.authority.node());
        if owner != Some(reply.source) {
            reaped.push(reply.transfer);
        } else if reemit_due {
            to_emit.push(*reply);
        }
    }
    if !to_emit.is_empty() {
        runtime.last_abort_reply_emit = now;
    }
    for reply in to_emit {
        outbox.push_flow(
            reply.source,
            MsgClass::Saga,
            &InterShardFlow::CrossingAborted(vd_wire::intershard::CrossingAborted {
                subject: reply.subject,
                transfer: reply.transfer,
            }),
        );
    }
    for transfer in reaped {
        runtime.pending_abort_replies.remove(&transfer);
        runtime
            .pending_writes
            .push((StoreKey::AbortReply(transfer).bytes(), None));
    }
}

/// Map a `DropApplied` proof-of-apply ack to the [`SagaState::BatchHandoff`] event its `step_id` proves
/// (D-7d). Total over the only three steps `DropApplied` carries — RELEASE (the SOURCE went
/// `Held→Departing`), DROP (the DEST promoted `Arriving→Held`), else COMPLETE (the SOURCE retired its
/// retained `Departing` copy) — each emitted by exactly ONE producer, so all three arms are reachable
/// and the `else` is the COMPLETE case, NOT an unexpected-step catch-all (HR5). The egress that used to
/// live in the old `handle_drop_applied`/`handle_batch_adopted` is now in the saga's `Emit*` executors
/// (the saga owns the choreography — so a stranded handoff is visible to `scan_deadlines`).
#[must_use]
pub(crate) fn drop_applied_event(step_id: u32) -> SagaEvent {
    if step_id == TRANSIENT_RELEASE_STEP {
        SagaEvent::SourceDropApplied
    } else if step_id == TRANSIENT_DROP_STEP {
        SagaEvent::DestDropApplied
    } else {
        SagaEvent::SourceRetired
    }
}
