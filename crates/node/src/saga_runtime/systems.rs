//! THE SCHEDULED SYSTEMS: the orchestrator's tick, and the one barrier that ends it.
//!
//! Owns: the per-tick drive (new triggers, due deadlines, then every live saga), the pre-scan that
//! latches destination adoption out of the inbox, and the GROUP-COMMIT BARRIER that runs strictly
//! last and flushes everything this tick staged.
//!
//! Does NOT own: any effect of its own. Every system here delegates — it decides only ORDER, and the
//! order is the point: nothing is durable until the barrier says so, and the barrier is one place.

use super::{
    DirSnapshot, LiveSaga, PendingStart, SagaRuntimeRes, StoreKey, StoreRes, ack_to_event,
    commit_result, deliver, drop_applied_event, encode, handle_crossing_request,
    handle_exterior_crossing_request, handle_transient_crossing_request, process_rehome_starts,
    reap_lapsed_leases, run_to_quiescence, scan_deadlines, stash_flush,
};
use crate::orchestrator::{DirectoryRes, UniverseClockRes};
use bevy_ecs::prelude::{Res, ResMut};
use vd_core::{EpochId, UniverseTick};
use vd_sim::directory::DirectoryCore;
use vd_sim::io::{Inbound, MsgClass, Store};
use vd_sim::runtime::{ClockSample, InboundBox, OutboundBox};
use vd_sim::saga::{self, SagaEvent};
use vd_wire::intershard::{InterShardFlow, TransferAck};

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
        // GUARD (audit D6-1): a re-trigger for an ALREADY-LIVE transfer is a no-op — never CLOBBER an
        // in-flight saga. The durable path's `lock_transfer` already refuses a second saga (the subject
        // is `in_transfer`-locked), but a TRANSIENT batch takes NO lock, so an unconditional re-insert
        // would overwrite its live `BatchHandoff` saga (resetting the choreography). One key → one saga.
        if runtime.sagas.contains_key(&ctx.transfer) {
            continue;
        }
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
                flushed_state: Vec::new(),
                dead_observed_since: None,
                dest_adopted: false,
                opened: now,
            },
        );
        // Durable start = PrepareSubscribe (no EmitCrossing yet); transient start = the go-token
        // (collected by run_to_quiescence). `None` flush_pose is correct for both at start.
        let (final_state, tombstone, rejected, batch_gos, _won_inside) = run_to_quiescence(
            &ctx,
            gateway,
            state,
            actions,
            dir,
            outbox,
            epoch,
            now,
            None,
            &[],
        );
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

/// CA-1 S3/S4 PRE-SCAN latch pass. Sets `dest_adopted = true` for every live saga whose dest's
/// `BatchAdopted` is present in THIS tick's inbox, BEFORE `scan_deadlines` runs. This is the intra-tick
/// ordering that makes the AwaitAdopt over-discard guarantee HARD: `scan_deadlines` runs before the inbox
/// drain, so latching in `deliver` (inside the drain) is one tick too late for a `BatchAdopted` that lands
/// on the exact budget-maturity tick — the discard would already have fired + tombstoned. Decodes only
/// enough to spot `BatchAdopted` (the drain re-decodes and acts on the same message); a non-Saga message, a
/// decode failure, a non-adopt arm, or an adopt for an absent/tombstoned saga is a skip (no side effect).
/// The latch is MONOTONE (never cleared) and RAM-only — the dest re-drives `BatchAdopted` every tick it
/// holds `Arriving`, so the latch is re-established every tick the ack lane delivers.
pub(crate) fn latch_adopted_from_inbox(runtime: &mut SagaRuntimeRes, inbox: &InboundBox) {
    for msg in &inbox.0 {
        if let Inbound::Wire {
            class: MsgClass::Saga,
            bytes,
            ..
        } = msg
            && let Ok(InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id, ..
            })) = postcard::from_bytes::<InterShardFlow>(bytes)
            && let Some(live) = runtime.sagas.get_mut(&transfer_id)
        {
            live.dest_adopted = true;
        }
    }
}

/// The orchestrator saga-runtime system: process new triggers, FIRE due deadlines (Slice 2a), then
/// drive every live saga forward on the gateway acks delivered this tick. Runs on the orchestrator's
/// single-threaded schedule; the directory CAS is a direct in-process call (no await, no lock across a send).
pub fn drive_sagas_core(
    inbox: Res<InboundBox>,
    clock: Res<ClockSample>,
    mut dir: ResMut<DirectoryRes>,
    mut runtime: ResMut<SagaRuntimeRes>,
    mut outbox: ResMut<OutboundBox>,
) {
    let now = clock.universe_tick;
    let epoch = clock.epoch;
    process_starts(&mut runtime, &mut dir.0, &mut outbox, epoch, now);
    // CA-1 S3/S4 — latch `dest_adopted` from THIS tick's inbox BEFORE `scan_deadlines` decides any
    // destructive resolution. The scan runs before the ack drain below, so a `BatchAdopted` arriving on the
    // exact budget-maturity tick would otherwise be seen too late (the discard would already have fired +
    // tombstoned). This pre-scan latch makes the AwaitAdopt over-discard guarantee HARD (see the latch doc).
    latch_adopted_from_inbox(&mut runtime, &inbox);
    // Slice 2a: fire due deadlines BEFORE the ack loop — a saga that loses its ack this tick still
    // gets its Timeout re-drive/abort next tick (the producer is the R1 backstop, never a wedge).
    scan_deadlines(&mut runtime, &mut dir.0, &mut outbox, epoch, now);
    for msg in &inbox.0 {
        // `from` is the transport ORIGIN — carried past the class filter because the transient crossing
        // consumer (3f-C) replies to it (a transient has no directory `OwnerRecord`, so the connection it
        // arrived on is the only authoritative reply address). The other saga-driving arms ignore it.
        let (from, class, bytes) = match msg {
            // D-3 CLEAR-ON-ACK: ANY successful inbound from a peer is proof it is alive — clear its
            // unreachable evidence (done at the TOP, BEFORE the class filter, so a peer that only sends
            // `LeaseRenew`/Directory ops — not saga acks — still un-marks itself; this is why no separate
            // `serve_directory` clear is needed: both systems read the same inbox, this one sees it all).
            Inbound::Wire { from, class, bytes } => {
                runtime.liveness.record_ack(*from);
                (*from, class, bytes)
            }
            // D-3: a delivery failure toward `to` — record one unreachable notice. The evidence-gated
            // tracker confirms `to` dead only after `n_consecutive_unreachable` within the window with no
            // intervening ack (so a recoverable blip never confirms a healthy peer); `scan_deadlines`
            // resolves a `BatchHandoff` whose source/dest is CONFIRMED dead.
            Inbound::NodeUnreachable { to, .. } => {
                runtime.liveness.record_unreachable(*to, now);
                runtime.liveness_notices += 1;
                continue;
            }
            // R-4d M3: a LOCAL send shed says NOTHING about `to`'s liveness — routing it to
            // `record_unreachable` would false-confirm a live-but-ack-stalled peer dead and trip a
            // destructive re-home. Count it and CONTINUE; NEVER touch the liveness tracker, and NEVER
            // clear liveness evidence (a shed is orthogonal to the peer's inbound stream — `record_ack`
            // still fires only on `Inbound::Wire`).
            Inbound::SendShed { .. } => {
                runtime.sends_shed += 1;
                continue;
            }
            // A peer reset is a SESSION fact. The saga runtime binds durable flows to a NODE: a
            // restarted shard is the same node and its at-least-once flows replay to it by design,
            // so nothing here changes. Counted, never fed to the liveness tracker.
            Inbound::PeerReset { .. } => {
                runtime.peer_resets += 1;
                continue;
            }
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
                state,
                ..
            })) => {
                // STASH the pose BEFORE stepping the event, so an EmitCrossing reachable on this
                // same tick (once both freeze + flush have landed) reads it.
                stash_flush(&mut runtime, transfer_id, pose, state);
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
            // D-7d adopt-before-drop PHASE 1: the dest ADOPTED the batch (uncounted `Arriving`) → drive
            // the LIVE saga's `BatchHandoff` tail (`AwaitAdopt→AwaitRelease`, emitting `TransientRelease`
            // to the source). Routing through `deliver` (not the old read-only handler) is what keeps the
            // saga alive across the handoff so `scan_deadlines` can resolve a stranded item (D-7d kills).
            Ok(InterShardFlow::TransferAck(TransferAck::BatchAdopted { transfer_id, .. })) => {
                deliver(
                    &mut runtime,
                    &mut dir.0,
                    &mut outbox,
                    epoch,
                    now,
                    transfer_id,
                    SagaEvent::BatchAdopted,
                );
            }
            // D-7d PHASES 2-4: a `DropApplied` proof-of-apply advances the tail by the phase its step
            // proves (`drop_applied_event`): RELEASE → `AwaitRelease→AwaitPromote` (promote the dest),
            // DROP → `AwaitPromote→AwaitComplete` (release-complete the source), COMPLETE → `Done`.
            Ok(InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id,
                step_id,
            })) => {
                deliver(
                    &mut runtime,
                    &mut dir.0,
                    &mut outbox,
                    epoch,
                    now,
                    transfer_id,
                    drop_applied_event(step_id),
                );
            }
            // The DEST's crossing ack: the dest journal is the exactly-once dedup. Release is NOT
            // gated on this ack — it rides the ordered `PromoteAck` (1d.5b.1) instead. The crossing
            // ack is decoded + dropped here — never a phase transition.
            Ok(InterShardFlow::TransferAck(
                TransferAck::Accepted { .. } | TransferAck::Rejected { .. },
            )) => {}
            // Slice 3f-B: a DURABLE entity crossed a realm boundary — resolve the three heads and START a
            // crossing saga (or count the unresolved/subject-gone outcome). The subject/source/dest/session
            // all come from the directory; the id rides the WIRE fence so it matches the source latch.
            Ok(InterShardFlow::CrossingRequest(req)) => {
                handle_crossing_request(&mut runtime, &dir.0, &mut outbox, req);
            }
            // Slice 3f-C: a TRANSIENT crossed a realm boundary — resolve the dest realm and GRANT it back to
            // the transport-origin `from` (a transient has no directory owner to look up).
            Ok(InterShardFlow::TransientCrossingRequest(req)) => {
                handle_transient_crossing_request(&dir.0, &mut outbox, &mut runtime, req, from);
            }
            // The ruler switch, slice 1: a parent asks to move its hull's EXTERIOR.
            Ok(InterShardFlow::ExteriorCrossingRequest(req)) => {
                handle_exterior_crossing_request(&mut runtime, &dir.0, req, from);
            }
            // Slice 3f-D (Mechanism Y): the SOURCE acked a crossing-abort reply — drop the pending entry +
            // stage its persist-DELETE (both ride this tick's group-commit barrier, so a kill after the ack
            // never re-emits). `is_some()`-gated so a REDELIVERED ack (the `ReDriven` class re-sends it on
            // every re-delivered `CrossingAborted`) is an idempotent no-op — never a double-DELETE stage.
            Ok(InterShardFlow::CrossingAbortedAck(ack)) => {
                if runtime
                    .pending_abort_replies
                    .remove(&ack.transfer)
                    .is_some()
                {
                    runtime
                        .pending_writes
                        .push((StoreKey::AbortReply(ack.transfer).bytes(), None));
                }
            }
            // Everything else (Ghost / Directory / Saga commands / DirectoryReply / FlushSource /
            // CrossingAborted — the orchestrator EMITS the abort reply, never consumes it) and any decode
            // failure: not a saga-driving inbound here.
            _ => {}
        }
    }
    // D-3 Slice 4: the expiry REAPER runs at the HEAD of the barrier — BEFORE the directory reconcile below
    // — so a revoke it makes lands in the SAME tick's `dirty` delta set and is captured by the incremental
    // reconcile + the single commit (the revoke is durable this tick as a DELETE; a reaped record never
    // resurrects on a kill-9 — the COMP-2 guarantee). It mutates the directory in RAM; the reconcile then
    // drains the delta set the mutation recorded.
    reap_lapsed_leases(&mut runtime, &mut dir.0, now);
    // D-37 Slice 3: ARM the standing re-homes the reaper just enqueued — SAME tick, still inside the
    // barrier, so the `lock_transfer` + the armed parked saga are captured by the reconcile + commit below
    // (durable this tick; on a kill-9 mid-window the RAM queue is lost but the still-dead UNLOCKED record
    // makes the reaper re-detect on reboot — see `PendingReHome`). Must run AFTER the reaper (it drains what
    // the reaper enqueued) and BEFORE the reconcile (so the lock/saga persist this tick).
    process_rehome_starts(&mut runtime, &mut dir.0, &mut outbox, epoch, now);
    // RLM Step 3e: the D-6 GROUP-COMMIT BARRIER was extracted to the [`commit_barrier`] system (below) so
    // the realm-lifecycle reconciler (`reconcile_realm_lifecycle`) can run BETWEEN this core and the ONE
    // fsync, staging its grant/revoke into the SAME `dir.0.dirty` set. The reaper + rehome-arm above STAY
    // here (they run before the reconcile); ONLY the drain/commit moved — the mutation phases are unchanged,
    // so persist-before-effect + the COMP-2 anti-zombie guarantees hold exactly as before.
}

/// The D-6 GROUP-COMMIT BARRIER body (RLM Step 3e extraction). Drains this tick's staged saga/go-token
/// writes (`pending_writes`), then this tick's incremental DIRECTORY deltas (`dir.dirty` — `Some` ⇒ PUT the
/// new snapshot, `None`/DELETE ⇒ the COMP-2 anti-zombie: a revoked/reaped record is deleted durably so
/// `rehydrate` can't resurrect it), then the durable clock ceiling, then ONE `commit()`. The `dirty` set
/// accumulates across `serve_directory` + `drive_sagas_core` + `reconcile_realm_lifecycle`, so this single
/// drain — the one place with both the store and the directory — captures every change this tick.
/// `O(changes)`, not `O(directory)`: a quiescent tick stages nothing.
fn group_commit(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    store: &mut (dyn Store + Send + Sync),
    clock_res: &UniverseClockRes,
) {
    for (key, value) in std::mem::take(&mut runtime.pending_writes) {
        match value {
            Some(bytes) => store.put(&key, &bytes),
            None => store.delete(&key),
        }
    }
    for (key, change) in dir.take_dirty() {
        match change {
            Some(record) => {
                let snapshot = DirSnapshot { key, record };
                store.put(&StoreKey::Directory(key).bytes(), &encode(&snapshot));
            }
            None => store.delete(&StoreKey::Directory(key).bytes()),
        }
    }
    store.put(
        &StoreKey::Clock.bytes(),
        &encode(&(clock_res.0.epoch(), clock_res.0.confirmed_ceiling())),
    );
    store.commit();
}

/// The orchestrator's FINAL chained system (RLM Step 3e): the D-6 group-commit barrier. Runs strictly AFTER
/// `drive_sagas_core` (the reaper + rehome-arm) and `reconcile_realm_lifecycle` (the RLM grant/revoke), so
/// ONE fsync captures every state change this tick and no effect leaves the orchestrator before the state
/// authorizing it is durable (persist-before-effect). Splitting the barrier out is a MOVE of the drain/
/// commit only — the mutation order is unchanged, so the D-6 + COMP-2 guarantees are preserved exactly.
pub fn commit_barrier(
    clock_res: Res<UniverseClockRes>,
    mut dir: ResMut<DirectoryRes>,
    mut runtime: ResMut<SagaRuntimeRes>,
    mut store: ResMut<StoreRes>,
) {
    group_commit(&mut runtime, &mut dir.0, &mut *store.0, &clock_res);
}
