//! THE TRANSFER-CONTROL CONSUMER: seven phases, each separately acked and separately compensatable.
//!
//! Owns: the gateway's counterpart to the saga runtime — prepare, request-cut, freeze, commit, thaw,
//! abort, release — the seq cut that partitions a session's input across the swap, and the standing
//! per-observer delivery watermark that tells the saga when the destination is actually being seen.
//!
//! Does NOT own: the commit decision. The directory CAS is the only commit point and it is somebody
//! else's; every phase here applies an ordered command, acks it, and is safe to be handed the same
//! command again — which it will be.

use super::{
    GatewayConfig, GatewaySessions, GatewayStats, RosterEdits, SeqCut, Session, SessionPhase,
    TransferProgress, push_control, push_to_shard, reply_ack, store_commit, store_cut,
};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::Ordering;
use vd_core::{Fence, NodeId, SessionId, TransferId};
use vd_sim::io::MsgClass;
use vd_sim::runtime::OutboundBox;
use vd_wire::channels::ServerControlMsg;
use vd_wire::seams::directory::DirectoryKey;
use vd_wire::seams::transfer_control::{
    PrepareReject, PrepareResult, TransferControl, TransferControlAck,
};
use vd_wire::session_flow::GatewayToShard;

/// 1d.5a — the STANDING per-observer delivery watermark pass. For each in-flight transfer, emit
/// `DeliveredToObservers` iff the dest observer set is NON-EMPTY (an empty set is NEVER vacuously
/// satisfied — in the window before the dest sub opens the saga must NOT be told delivery is done)
/// AND every current dest observer has received >=1 dest frame. RECOMPUTED each tick (never latched
/// here): an observer opening mid-demote inherits watermark 0 and RE-BLOCKS. Idempotent re-emit
/// while true (the saga latches `dest_delivered` and the FSM absorbs the repeat) — a STANDING
/// per-TICK (not per-frame; no 20Hz regression — D-24) reliable ack for the demote-tail duration;
/// bounded by the tail, rising-edge gating deferred. HR1: the watermark is gateway-internal; only
/// the boolean `DeliveredToObservers` crosses, on the existing SagaAck arm.
pub(crate) fn recompute_delivery_watermarks(
    sessions: &GatewaySessions,
    config: &GatewayConfig,
    outbox: &mut OutboundBox,
) {
    for session in sessions.by_session.values() {
        let Some(progress) = session.transfer.as_ref() else {
            continue;
        };
        if every_observer_delivered(sessions, progress.dest) {
            reply_ack(
                outbox,
                config.orchestrator,
                TransferControlAck::DeliveredToObservers {
                    transfer: progress.transfer,
                },
            );
        }
    }
}

/// Whether the dest observer set is NON-EMPTY AND every observer has delivered >=1 frame on its
/// dest sub. Iterates sessions directly (the `subs.get(&dest)` Some/None is the observer /
/// non-observer split — a domain case, never a cross-lookup desync). Bitwise `&` so neither the
/// found-nor-all arm is a short-circuit-uncoverable region (HR5). Empty observer set ⇒ false.
#[must_use]
pub(crate) fn every_observer_delivered(sessions: &GatewaySessions, dest: NodeId) -> bool {
    let mut found = false;
    let mut all = true;
    for session in sessions.by_session.values() {
        let Some(rec) = session.subs.get(&dest) else {
            continue;
        };
        found = true;
        all &= session.delivered.get(&rec.sub).copied().unwrap_or(0) >= 1;
    }
    found & all
}

/// THE gateway counterpart to the saga runtime: consume one `TransferControl` command.
/// EVERY phase is now LIVE: Prepare/RequestCut/FreezeSource (the cut install, 1c.3)/
/// CommitAuthority (the route swap, 1c.4)/ThawSource/AbortTransfer/ReleaseSubscribe (the demote
/// tail's SUCCESS teardown, 1c.8 — acks `Released` so the saga reaches `Done`). The applied-steps
/// journal gates every command BEFORE any effect (consult-before-effect) and records AFTER
/// (record-after-effect) so an at-least-once redelivery re-sends the recorded ack verbatim and
/// never re-applies.
pub(crate) fn on_transfer_control(
    cmd: TransferControl,
    // `&mut` (was `&`): the one-shot `reject_next_prepare` lever is `.take()`n by `apply_prepare`
    // (3g abort-leg). The borrow is split at the call site — `config.orchestrator` reads and
    // `&mut config.reject_next_prepare` never overlap (distinct statements / a disjoint field).
    config: &mut GatewayConfig,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let transfer = cmd.transfer();
    let step = cmd.step_id();
    let session_id = cmd.session(); // the get_mut key; reused for the dest-bound OpenInputSlot/SessionInput
    let Some(session) = sessions.by_session.get_mut(&session_id) else {
        stats.transfer_unroutable += 1; // unknown/absent session: counted, never panic
        return;
    };
    // REDELIVERY GATE (consult-before-effect): a recorded step for THIS transfer re-sends
    // the recorded ack verbatim, no effect — via the ONE dedup accessor `TransferProgress::
    // recorded`. A let-chain (each link's true/false arm is separately exercised: no-transfer
    // / wrong-transfer / unrecorded-step / recorded-step).
    //
    // S3: `RequestCut` is now SELF-ACKING (`apply_request_cut` returns `CutConfirmed`, journaled
    // at step 1 by the standard record-then-send path below), so it is NO LONGER excluded from
    // this gate — a redelivered `RequestCut` re-serves the recorded `CutConfirmed` verbatim, never
    // re-pushing the client `RequestCut` or re-generating the ack. (The old exclusion existed only
    // because the now-inert cut-marker observer owned the step-1 journal; the server owns it now.)
    if let Some(tp) = session.transfer.as_ref()
        && tp.transfer == transfer
        && let Some(prior) = tp.recorded(step)
    {
        reply_ack(outbox, config.orchestrator, prior);
        return;
    }
    // The READ-plane subs to close AFTER the ack/journal (the `&mut Session` borrow must end
    // before `GatewaySessions::close_sub` — the sole sub-close primitive — can run).
    // ReleaseSubscribe closes the SOURCE sub (`src` from the command); AbortTransfer closes
    // EXACTLY this transfer's DEST sub (`tp.dest`, captured at PrepareSubscribe) — never an "any
    // sub != config.shard" heuristic (which would close the player's CURRENT live sub on a chained
    // transfer and ALL composited subs in the N-shard end goal). Inert in the 1d.2 happy path
    // (abort runs pre-CAS, before the dest sub exists; `close_sub` no-ops then) but PRECISE for the
    // commit/flip-window-open ordering and the N-shard future.
    let mut subs_to_close: Vec<NodeId> = Vec::new();
    // RLM 5f-4: the runtime routable-roster edits this command produced, applied after the borrow ends
    // (the same deferred-mutation discipline as `subs_to_close` — see [`RosterEdits`]).
    let mut roster = RosterEdits::default();
    // Compute the ack (or None for deferred/parked phases), then record-then-send below.
    let ack: Option<TransferControlAck> = match cmd {
        TransferControl::PrepareSubscribe { dest, .. } => {
            // RLM 5f-4: the dest a DEFENSIVE replace is about to displace — captured BEFORE
            // `apply_prepare` overwrites the progress, because that displaced dest's roster claim must be
            // released WITH it or a second Prepare on this session leaks a refcount that pins the old dest
            // on the roster forever.
            let displaced = session.transfer.as_ref().map(|tp| tp.dest);
            // RLM 5f-4e: and the displaced progress's DEMOTING SOURCE home. A second Prepare on a session
            // that ALREADY committed a crossing (its demote tail still open) would otherwise strand that
            // source claim: the replacement progress starts `demoting_home: None`, so after the overwrite NO
            // holder can name the old source and no terminal will ever release it.
            let displaced_home = session.transfer.as_ref().and_then(|tp| tp.demoting_home);
            // Split borrow: `session` from `sessions`, `&mut config.reject_next_prepare` from
            // `config` (disjoint resources / a disjoint field) — the one-shot 3g reject lever.
            let ack = apply_prepare(
                session,
                transfer,
                dest,
                stats,
                &mut config.reject_next_prepare,
            );
            // RLM 5f-4 — THE CROSSING-DEST ADMISSION. The roster edits ride the SAME condition as the
            // progress write: `apply_prepare` returns `None` from exactly ONE guard (a not-Active session)
            // and that guard opens NO progress, so a refused prepare must claim nothing (a claim with no
            // terminal to release it IS the leak) and displace nothing. A `Rejected` verdict still opened
            // the progress and still gets the claim — its `AbortTransfer` terminal releases it.
            // BOTH displaced roles ride the SAME `ack.is_some()` gate as the claim: a refused prepare
            // overwrote no progress, so it must hand nothing back either.
            if ack.is_some() {
                roster.claim.push(dest);
                roster.release.push(displaced);
                roster.release.push(displaced_home);
            }
            ack
        }
        TransferControl::RequestCut { .. } => {
            // S3: self-acking `CutConfirmed` (server-timed cut) — journaled at step 1 below.
            apply_request_cut(session, outbox, transfer, stats)
        }
        TransferControl::FreezeSource {
            marker_seq, dest, ..
        } => apply_freeze(session, transfer, marker_seq, dest, stats),
        TransferControl::ThawSource { .. } => apply_thaw(session, transfer, stats),
        TransferControl::AbortTransfer { .. } => {
            // Close EXACTLY this transfer's dest sub (`tp.dest`), mirroring ReleaseSubscribe's
            // precise `src`; a foreign/absent abort matches no `tp` and closes nothing. `close_sub`
            // no-ops if the dest sub is not (yet) open — the normal pre-CAS abort case.
            if let Some(tp) = session
                .transfer
                .as_ref()
                .filter(|tp| tp.transfer == transfer)
            {
                subs_to_close.push(tp.dest);
                // RLM 5f-4e: a POST-CAS abort ALSO closes the DEMOTING SOURCE sub, so the un-route below
                // never outlives its subscription (the invariant the whole `demoting_home` stash exists to
                // hold). `extend` of an `Option` is branchless — `None` (every PRE-CAS abort) is a no-op, so
                // no new region and every existing abort cell is untouched. This arm is not reachable from
                // the shipped sender (`saga.rs post_commit_is_forward_only` proptests no AbortTransfer after
                // CasWon), but it is the one place the rework's own invariant could be violated.
                subs_to_close.extend(tp.demoting_home);
                // RLM 5f-4: the abort terminal RELEASES this transfer's crossing-dest claim — the
                // compensating half of the Prepare-time admission. A pre-CAS abort never re-pointed
                // `home_shard`, so this is the dest's ONLY claim and it leaves the roster here.
                roster.release.push(Some(tp.dest));
                // RLM 5f-4e: …AND the DEMOTING SOURCE home the commit stashed (released in lock-step with the
                // sub close above). For a PRE-CAS abort this is always `None` (no commit ran ⇒ nothing
                // displaced) — the no-op arm of `release_dynamic_shard`, so no new branch. Load-bearing for a
                // POST-CAS abort: `apply_abort` PRUNES `session.transfer`, so a terminal that skipped the
                // stash would strand the source's claim forever with no later holder able to name it.
                roster.release.push(tp.demoting_home);
            }
            apply_abort(session, transfer)
        }
        TransferControl::CommitAuthority {
            new_fence, subject, ..
        } => apply_commit(
            session,
            transfer,
            new_fence,
            subject,
            session_id,
            stats,
            outbox,
            &mut roster,
        ),
        TransferControl::ReleaseSubscribe { src, .. } => {
            // Close the SOURCE sub ONLY for the matching in-flight transfer (mirroring
            // `apply_release`'s prune); a foreign/absent release closes nothing.
            if let Some(tp) = session
                .transfer
                .as_ref()
                .filter(|tp| tp.transfer == transfer)
            {
                subs_to_close.push(src);
                // RLM 5f-4: the demote tail releases this transfer's crossing-dest claim. The dest stays
                // ROUTABLE because `CommitAuthority` claimed it a SECOND time for the session's new home
                // role — that claim lives until the session exits, so the post-crossing player keeps
                // receiving its dest frames long after the saga reaches `Done`.
                roster.release.push(Some(tp.dest));
                // RLM 5f-4e: the DEMOTING SOURCE's claim is due at the demote tail — the `close_sub` above
                // (same command) stops the client reading the source. Holding the claim until here is what
                // closes the blind window (the source stayed routable across the whole CommitAuthority→here
                // grace). NOTE the routable window ends at the sub's CLOSE, one inbound-batch short of its
                // REMOVAL: `close_sub` deliberately keeps the sub drainable for the rest of this batch + until
                // the next `sweep_draining`, whereas this release lands at the end of THIS command — so a
                // source straggler later in the same batch is dropped/`undecodable` for a demand-spawned
                // source (a `known_shards` source is still served). Tying the release to the sweep instead is
                // OWED (DEFERRED). `None` for a static/never-dynamically-homed session (the no-op arm).
                roster.release.push(tp.demoting_home);
            }
            apply_release(session, transfer)
        }
    };
    // RECORD-then-SEND for the LIVE acking phases. The journal write is GUARDED to the
    // IN-FLIGHT transfer (`tp.transfer == transfer`): an idempotent re-ack of a phase for a
    // NON-in-flight transfer (e.g. a stray Thaw/Abort while a different transfer is live)
    // must never pollute the live transfer's journal — bounding it to its own steps. (A
    // terminal AbortTransfer prunes `session.transfer` inside `apply_abort`, so `as_mut()`
    // is already `None` there — no record, the abort being idempotently re-ackable anyway.)
    if let Some(ack) = ack {
        if let Some(tp) = session.transfer.as_mut()
            && tp.transfer == transfer
        {
            tp.journal(step, ack);
        }
        reply_ack(outbox, config.orchestrator, ack);
    }
    // The `&mut Session` borrow has ended. RLM 5f-4: apply the runtime-roster edits FIRST — claims before
    // releases (see [`RosterEdits`]) — so a just-admitted dest is node-class-dispatchable for the REST of
    // this very tick's inbound batch, not only from the next tick.
    for dest in roster.claim {
        sessions.claim_crossing_dest(config, dest);
    }
    for node in roster.release {
        sessions.release_dynamic_shard(node);
    }
    // Then close the collected subs through the sole close primitive (Draining grace; the next-tick sweep
    // removes them). `close_sub` is idempotent and a no-op for a shard this session does not subscribe to.
    for shard in subs_to_close {
        sessions.close_sub(session_id, shard, config, outbox, stats);
    }
}

/// `PrepareSubscribe` (step 0): open the per-session transfer progress + reply `Prepared`.
/// 1c STUB readiness — there is no dest-subscription/ghost machinery yet (one stub shard),
/// so the verdict is `Ready` (the typed `Rejected` path stays WIRED for later bands, not
/// dead live code). DEFENSIVE replace: overwrite any stale progress.
///
/// REQUIRES the session be ACTIVE (WB-1): a transfer can only run for an attached avatar.
/// This single guard ENFORCES "attach strictly precedes transfer" LOCALLY (rather than
/// borrowing it from saga ordering) and is what makes every downstream route writer safe:
/// `session.transfer` is set Some ONLY here, ONLY when Active; Active is monotonic-until-
/// removal; and the attach path early-returns once Active (never re-storing the route). So a
/// cut installed by `apply_freeze` — or a route swapped by 1c.4 `CommitAuthority` — can never
/// be clobbered by a late `SessionAttached`. A not-Active prepare is counted + un-acked (pins
/// the saga, per WEDGE-1), never a route-touching transfer on a half-attached session.
pub(crate) fn apply_prepare(
    session: &mut Session,
    transfer: TransferId,
    dest: NodeId,
    stats: &mut GatewayStats,
    // 3g abort-leg (INERT test lever): consumed ONCE when this prepare would otherwise reply `Ready`.
    // The not-Active guard below stays ABOVE this and MUST NOT consume it — the lever fires only for a
    // would-be-`Ready` prepare, so a not-Active prepare leaves it armed for the retried (Active) one.
    reject_next_prepare: &mut Option<PrepareReject>,
) -> Option<TransferControlAck> {
    if !matches!(session.phase, SessionPhase::Active { .. }) {
        stats.transfer_unroutable += 1;
        return None;
    }
    // RACE-1 (pinned to Slice 2 — DEFERRED D-23): replacing a LIVE, different transfer's
    // progress discards its journal. Impossible today — the orchestrator serializes one
    // saga per session-subject (`DirectoryCore::lock_transfer`), and a redelivered SAME
    // transfer is caught by the redelivery gate before reaching here — so any existing
    // progress here is necessarily a stale DIFFERENT transfer. Becomes reachable only once
    // Slice-2 closes the abort-lock leak (D-1); pinned loud so it is never silently relied on.
    if let Some(displaced) = session.transfer.as_ref() {
        tracing::warn!(
            displaced = displaced.transfer.0,
            opening = transfer.0,
            "PrepareSubscribe replaced a stale in-flight transfer's progress — revisit at \
             Slice 2 (the one-saga-per-subject lock makes this benign today; D-23)"
        );
    }
    session.transfer = Some(TransferProgress {
        transfer,
        cut_requested: false,
        dest, // captured here for the precise abort-time dest-sub close (1d.2)
        // RLM 5f-4e: a FRESH progress is demoting nothing — the stash is written ONLY by
        // `apply_commit` (the one place a home is displaced). A defensive replace therefore starts
        // `None`, which is why the DISPLACED progress's stash must be handed back at the replace
        // (`on_transfer_control`'s PrepareSubscribe arm) or its source claim would be stranded.
        demoting_home: None,
        applied: BTreeMap::new(),
        dest_buffer: VecDeque::new(),
    });
    // The dest input slot is opened at CommitAuthority (apply_commit) — the gateway BUFFERS
    // seq>marker locally during the cut, so the dest needs nothing until the commit drain.
    // (An early prepare-time open is a P3 gateway-adoption resilience concern, not 1c.5.)
    //
    // 3g abort-leg: the ONE-SHOT reject lever. `None` (every real cluster) = the 1c `Ready` stub
    // (behaviour-identical). `Some(reject)` = reply `Rejected(reject)` and SELF-CLEAR (`.take()`),
    // so the very next prepare is `Ready` again — the sole way to drive a crossing-origin durable
    // saga into its pre-CAS abort in a cluster (the gateway is the durable Prepare decider).
    let result = match reject_next_prepare.take() {
        Some(reject) => {
            tracing::debug!(
                transfer = transfer.0,
                ?reject,
                "PrepareSubscribe REJECTED by the one-shot reject_next_prepare lever (3g abort-leg)"
            );
            PrepareResult::Rejected(reject)
        }
        None => {
            tracing::debug!(
                transfer = transfer.0,
                "PrepareSubscribe readiness is a 1c stub (Ready)"
            );
            PrepareResult::Ready
        }
    };
    Some(TransferControlAck::Prepared { transfer, result })
}

/// `RequestCut` (step 1): mark the cut as requested + SELF-ACK `CutConfirmed` (the SERVER-TIMED
/// cut, S3). The cut is now driven ENTIRELY server-side — the saga no longer waits on a client
/// `CUT_MARKER` (which multi-hop breaks: on hop 2+ the client's session is bound to the FIRST
/// shard's port, so the marker never reaches the current authority, and the saga stalls
/// `Cutting`→`CutTimeout`). The gateway still pushes `ServerControlMsg::RequestCut` to the
/// client (harmless/cosmetic — the old client's marker is now an inert ordinary input, S3),
/// but the `CutConfirmed` seq here is a PLACEHOLDER: the REAL input-cut seq is derived ATOMICALLY
/// at cut-install time in [`apply_freeze`] from `last_input_seq` (the leak-free partition point),
/// and rides `SourceFrozen.drained_seq` + the installed `SeqCut.marker_seq` — never this value.
/// So the FSM carries this `marker_seq` unread (`CommittingCas` reads `drained_seq`, not it).
///
/// Journaled at step 1 (via the standard record-then-send path in `on_transfer_control`), so a
/// redelivered `RequestCut` re-serves the SAME `CutConfirmed` verbatim (idempotent). Requires the
/// matching progress (`PrepareSubscribe` precedes it); setting `cut_requested` keeps the F1 guard
/// intact for the now-inert marker observer. Returns the ack for the caller to journal + reply.
fn apply_request_cut(
    session: &mut Session,
    outbox: &mut OutboundBox,
    transfer: TransferId,
    stats: &mut GatewayStats,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if !bound {
        stats.transfer_unroutable += 1; // RequestCut without a matching Prepared: drop+count
        return None;
    }
    session
        .transfer
        .as_mut()
        .expect("bound implies the in-flight transfer is present")
        .cut_requested = true;
    push_control(
        outbox,
        session.client,
        &ServerControlMsg::RequestCut { transfer },
    );
    // SELF-ACK the cut server-side (S3). `marker_seq: 0` is a PLACEHOLDER — the FSM carries it
    // unread; the real input-cut seq is `apply_freeze`'s install-time `last_input_seq`.
    Some(TransferControlAck::CutConfirmed {
        transfer,
        marker_seq: 0,
    })
}

/// `FreezeSource` (step 2): INSTALL the live cut — `seq <= marker_seq` stays bound to the
/// source, `seq > marker_seq` will buffer toward `dest`. The SOLE `cut: Some(..)` writer; its
/// mirror image is `apply_thaw`'s `cut: None`, both via `store_cut`. The `seq > marker`
/// partition READ lands in 1c.5; 1c.3 installs the latent field only (this hot path reads
/// only `.authority`, so the install is inert to routing until then).
///
/// UNBOUND DIVERGES from `apply_thaw`: freeze is a FORWARD phase (its `SourceFrozen` drives
/// the directory CAS), so an unbound freeze must NOT fabricate an ack — acking a
/// never-installed cut would advance the saga to a CAS / route-swap against a session this
/// gateway no longer holds (the Bye-mid-transfer wedge, WEDGE-1). Returning `None` correctly
/// PINS the saga until the Slice-2 abort producer (D-23). Bound-check FIRST, mirroring
/// `apply_request_cut` (the forward sibling), NOT `apply_thaw` (a compensator that no-ops
/// truthfully when unbound). Self-acking, so a redelivery is re-served by the gate, never
/// re-entered here (the route is never re-touched).
pub(crate) fn apply_freeze(
    session: &mut Session,
    transfer: TransferId,
    _cmd_marker_seq: u64, // S3: the command's marker is a PLACEHOLDER — the gateway derives its own below.
    dest: NodeId,
    stats: &mut GatewayStats,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if !bound {
        stats.transfer_unroutable += 1; // no source to freeze: pin the saga (no ack)
        return None;
    }
    // S3 SERVER-TIMED CUT — derive the input-cut seq at INSTALL time from `last_input_seq`, the
    // gateway's per-session high-water. `route_input`'s `fetch_max` advanced `last_input_seq` and
    // then Forwarded EVERY pre-cut input to the current authority (the source), so at the instant
    // this `store_cut` installs the `SeqCut`, `last_input_seq` == "the highest seq forwarded to the
    // source". Everything `<= marker_seq` went to the source and WILL apply (in-order,
    // at-least-once); everything `> marker_seq` buffers for the dest, which resumes at
    // `marker_seq + 1`. So inputs partition IDENTICALLY across the cut — NONE lost, NONE doubled —
    // which is exactly why the client `CUT_MARKER` is no longer needed (S3). This RETIRES the OLD
    // hazard: the OLD marker came from the client's chosen `CUT_MARKER` seq, which could sit BELOW
    // inputs already Forwarded to the source, so the dest re-applied them (double).
    //
    // PARTITION SAFETY IS BY SERIALIZATION, NOT BY THESE ATOMICS (verify wf review a9946a1c). Today
    // `route_input` (from `on_client_input`) and this `apply_freeze` (from `on_transfer_control`)
    // both run inside the SINGLE `process_gateway_inbound` system on the ONE sim thread, one inbound
    // msg at a time — they never overlap, so the `load(marker)`→`store_cut` pair is effectively
    // atomic w.r.t. the router. **TRIPWIRE (DEFERRED.md D-46):** the "(future) threaded 20 Hz
    // forwarder" would make this a live TOCTOU — a concurrent `route_input` could `fetch_max(X+1)`
    // and Forward X+1 to the source in the window between the load and the store (marker=X), then
    // the dest re-applies X+1 (the dest seeds `last_applied` to `marker_seq`, NOT the source's true
    // last-applied, so it does NOT dedup this cross-shard double). Before threading the forwarder,
    // install the cut FIRST with a sentinel marker then settle it (so a racing input Buffers), or
    // seed the dest watermark from the source's acked last-applied — NOT this gateway high-water.
    let marker_seq = session.hot.last_input_seq.load(Ordering::Relaxed);
    store_cut(&session.hot, Some(SeqCut { marker_seq, dest }));
    // drained_seq = marker_seq (the install-time high-water): "the source applied input through
    // exactly this seq" holds because the source applies everything the gateway forwarded (all
    // `<= marker_seq`). The FSM threads THIS `SourceFrozen.drained_seq` into `CommittingCas` as the
    // CAS watermark + it seeds the dest's `OpenInputSlot.resume_from_seq` at `apply_commit`.
    Some(TransferControlAck::SourceFrozen {
        transfer,
        drained_seq: marker_seq,
    })
}

/// `CommitAuthority` (step 3): THE route swap. MOVE authority → `dest` (the `SeqCut.dest` the
/// preceding `FreezeSource` installed), CLEAR the cut, in ONE atomic `store_commit`.
///
/// `dest` HAS NO WIRE CARRIER by construction — `CommitAuthority { transfer, session, new_fence }`
/// carries no dest — so the installed `route.cut.dest` is authoritative: the gateway-local
/// crystallization of the saga's durable `SagaCtx.dest` (the SAME value `FreezeSource.dest`
/// rode). FreezeSource strictly precedes commit on reliable+ordered CONTROL (the saga emits
/// CommitAuthority only after `CommittingCas`), so the cut is present at first delivery.
///
/// Bound-check FIRST (mirror `apply_freeze`): a post-CAS forward phase must NOT fabricate
/// `Committed` for a session this gateway no longer holds (WEDGE-1) — pin instead.
///
/// **R-FENCE (DEFERRED D-25): `new_fence` is CARRIED, NOT installed as `route.fence`.** Frames
/// stamp the REALM fence (each sub accepts at its per-shard `SubEntry::accepted` realm fence,
/// checked in `on_shard_frame`), whereas `new_fence` is the per-ENTITY CAS fence
/// (`DirectoryKey::Entity`). Installing the Entity fence here would make the dest's OWN
/// realm-stamped frames stale
/// (`realm_fence.is_stale_against(new_fence) == true`) — a black screen. Fence rule 5 (fence
/// out a demoted REMOTE owner) activates by REALM-fence movement at 1d/mesh when source/dest
/// are distinct realm leases; intra-shard (one realm lease) it correctly does NOT fire.
/// `new_fence` stays threaded through wire+saga (the CAS linearization point); the gateway
/// consumes it without installing it until the dest re-stamps its realm lease (1d/mesh).
///
/// **RLM 5f-4e — closes D-34's A1 (authority-following detach); the composited-subs clause remains OWED.**
/// Commit is where the session's ROUTING TARGET follows authority: `home_shard` is re-pointed to `dest`.
/// The displaced OLD home's roster claim is **STASHED** in `TransferProgress::demoting_home`, NOT released
/// here — the source's read sub outlives the commit by the whole demote grace (RLM 5f-4e). This re-points
/// only the CURRENT authority; the SOURCE keeps its own sub/ghost until `ReleaseSubscribe`, so D-34's
/// composited-subs half is not yet closed. See step (b2).
#[allow(clippy::too_many_arguments)] // the 8th is the deferred roster carrier (the borrow-discipline seam)
fn apply_commit(
    session: &mut Session,
    transfer: TransferId,
    new_fence: Fence,
    subject: DirectoryKey,
    session_id: SessionId,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
    // RLM 5f-4: the deferred roster edits — `dest` is discovered HERE (from the installed cut), but
    // `dynamic_shards` lives on `GatewaySessions`, which is borrow-locked while `session` is held.
    roster: &mut RosterEdits,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if !bound {
        stats.transfer_unroutable += 1; // session/transfer absent: pin (WEDGE-1)
        return None;
    }
    // Capture BOTH `dest` and `marker_seq` from the installed cut BEFORE `store_commit`
    // clears it: `marker_seq` seeds the dest's resume watermark; `dest` is the new authority.
    let Some(SeqCut { dest, marker_seq }) = session.hot.route.load().cut else {
        // BOUND but no cut: FreezeSource must precede commit (ordered CONTROL). Loud,
        // counted, route UNTOUCHED — never a swap to a garbage dest; the saga pins.
        stats.commit_without_cut += 1;
        tracing::error!(
            transfer = transfer.0,
            "CommitAuthority with no installed cut — FreezeSource must precede commit; \
             route untouched, saga pins"
        );
        return None;
    };
    let _ = new_fence; // R-FENCE: carried-not-installed in 1c.4 (D-25); 1d/mesh installs on dest re-stamp
    // (a) AUTHORITATIVE OpenInputSlot: the dest seeds last_applied_seq = marker_seq, so the
    // drained resume batch (marker+1..) applies in order and a `seq <= marker` replay is
    // rejected — closing the UnknownSession input-loss hole for the genuinely-distinct dest.
    // 1c.8: it ALSO carries the transfer `subject` (forwarded verbatim from CommitAuthority) so
    // the dest ADOPTS the transferred avatar (its Entity becomes the dot's id). `new_fence` is
    // NOT carried — the dest learns it from its own directory HeadRead (pull-through).
    push_to_shard(
        outbox,
        dest,
        MsgClass::Control,
        &GatewayToShard::OpenInputSlot {
            session: session_id,
            fence: session.fence,
            account: session.account,
            resume_from_seq: marker_seq,
            subject,
        },
    );
    // (b) THE swap: authority:=dest, fence carried, cut:=None (ONE atomic publish).
    store_commit(&session.hot, dest);
    // (b2) RLM 5f-4e — closes D-34's A1 (composited-subs clause OWED): the player is now HOMED on the dest,
    // so the session's ROUTING TARGET
    // ([`session_target`]) must follow authority. `Option::replace` does both halves in ONE branchless
    // expression: point `home_shard` at `dest` and hand back the OLD home. Roster consequences:
    // - CLAIM `dest` for the new HOME role. This is a SECOND claim on top of the Prepare-time crossing
    //   claim (the refcount makes it idempotent-by-construction), and it is what keeps the dest routable
    //   after `ReleaseSubscribe` releases the crossing one — without it a demand-spawned dest left the
    //   roster the instant the demote tail landed and the player went blind again.
    // - RLM 5f-4e: the OLD home's claim is **STASHED, NOT RELEASED HERE** (`TransferProgress::
    //   demoting_home`). Commit moves the WRITE authority, but the client's READ subscription on the
    //   SOURCE stays open for the whole demote grace (closed at `ReleaseSubscribe`, a later tick) and the
    //   source keeps shipping frames. Releasing at commit un-routed a DEMAND-SPAWNED source, so its
    //   `Snapshot`/`RealmSnapshot` frames were counted `undecodable` for the entire tail — a BLIND player
    //   right after every crossing. The stash is handed back where the sub actually closes
    //   (`ReleaseSubscribe` / `AbortTransfer`), or at the session's exit if the client quits in the window,
    //   so a chain of crossings still accumulates no refcount per hop.
    // Before this, `home_shard` was never re-pointed at commit (the open D-34 owe), so a post-crossing
    // `Bye` detached at the SOURCE and leaked the dest's `SessionTable` entry.
    roster.claim.push(dest);
    // Two statements, not one expression: `home_shard` and `transfer` are disjoint fields but a single
    // `progress.demoting_home = session.home_shard.replace(dest)` would borrow `session` twice.
    let demoting_home = session.home_shard.replace(dest);
    // ONE `&mut` reach into the in-flight progress serves BOTH remaining steps (the stash write and the
    // buffer take), so the bound-check's unreachable-`None` shape is asserted exactly once.
    // The bound-check above guarantees `session.transfer` is Some (the in-flight transfer);
    // `expect` is the unreachable-arm shape.
    let progress = session
        .transfer
        .as_mut()
        .expect("bound-check above guarantees the in-flight transfer is present");
    progress.demoting_home = demoting_home;
    // (c) DRAIN the cut buffer to the now-authoritative dest as ordinary SessionInput, in
    // push order (== seq order: route_input's fetch_max dropped seq<=prev BEFORE buffering,
    // so the buffer is strictly increasing). `std::mem::take` empties it — the STRUCTURAL
    // drain-once guarantee: a redelivered CommitAuthority re-serves Committed via the gate
    // (above) AND finds the buffer empty, so it never re-drains. That SAME gate is what keeps the
    // (b2) stash exact under at-least-once delivery: a redelivered commit never re-enters here, so
    // `demoting_home` can never be overwritten with `Some(dest)` (which would strand the real source
    // claim) and `dest` is never claimed a third time.
    // Push order == seq order: `route_input`'s `fetch_max` returns `Deduped` for any
    // `seq <= prev` BEFORE the partition, so only a strictly-increasing subsequence is ever
    // buffered. The dest's own `last_applied_seq` dedup is the backstop regardless of order.
    let buffered = std::mem::take(&mut progress.dest_buffer);
    for input_bytes in buffered {
        push_to_shard(
            outbox,
            dest,
            MsgClass::Input,
            &GatewayToShard::SessionInput {
                session: session_id,
                fence: session.fence, // session-grant fence (NOT new_fence; R-FENCE/D-25)
                input_bytes,
            },
        );
    }
    Some(TransferControlAck::Committed { transfer })
}

/// `ThawSource` (step 4): the `FreezeSource` compensator — input resumes to the source.
/// Clears the cut (via `store_cut`) ONLY for the matching in-flight transfer
/// (ROB-THAW-UNBOUND-STORE): a thaw naming a different/absent transfer must NOT touch the
/// route, else it would clobber the in-flight transfer's LIVE cut (live since 1c.3). A thaw
/// against a never-frozen source is a correct no-op (the source never stopped), so it ALWAYS
/// acks `SourceThawed` (even unbound — counted — so the saga's thaw compensator can always
/// complete), but only the BOUND case stores the route. The live mirror image of `apply_freeze`.
fn apply_thaw(
    session: &mut Session,
    transfer: TransferId,
    stats: &mut GatewayStats,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if bound {
        store_cut(&session.hot, None);
    } else {
        stats.transfer_unroutable += 1;
    }
    Some(TransferControlAck::SourceThawed { transfer })
}

/// `AbortTransfer` (step 5, terminal): `PrepareSubscribe`'s compensator — tear down the dest
/// ghost. In 1c.2 `PrepareSubscribe` built no ghost (`Ready` stub), so teardown is inert on
/// session state and the route is untouched (the source stayed authoritative). NOT a
/// disconnect: the `Session`/`phase` are untouched. Prunes ONLY the matching in-flight
/// transfer (CP-1: an abort for transfer B must never clobber a live transfer A). An abort
/// is an IDEMPOTENT terminal: a redelivered abort, or an abort for a transfer this gateway
/// never held, is a benign no-op re-ack — NOT a routing failure (F2: it is uncounted, so
/// `transfer_unroutable` stays about genuine failures). Always acks `Aborted`.
fn apply_abort(session: &mut Session, transfer: TransferId) -> Option<TransferControlAck> {
    if session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer)
    {
        session.transfer = None; // prune ONLY the matching in-flight transfer
        // ROB-1c3: clear any installed cut LOCALLY. The saga emits ThawSource before
        // AbortTransfer over reliable CONTROL, so today the cut is usually already clear —
        // but abort must be LOCALLY fail-safe, not borrow that ordering: once the cut READ
        // goes live (1c.5) a survived cut becomes a live mis-route, and Slice-2/warp abort
        // producers need not preserve Thaw-before-Abort. `store_cut(None)` is idempotent
        // (a no-op when the cut is already clear), so it is safe on every abort path.
        store_cut(&session.hot, None);
    }
    tracing::debug!(
        transfer = transfer.0,
        "AbortTransfer tears down the dest ghost (inert in 1c.2 — no ghost) + clears any cut"
    );
    Some(TransferControlAck::Aborted { transfer })
}

/// `ReleaseSubscribe` (step 6, the SUCCESS teardown of the demote tail): close the source
/// subscription after demote grace and ack `Released` so the saga reaches `Done`. The cut is
/// already `None` (cleared at `CommitAuthority`'s `store_commit`) and the dest buffer was
/// take-drained there, so the ONLY post-commit remnant to clear is `session.transfer` — pruned
/// here ONLY when bound to THIS transfer (mirroring `apply_abort`'s prune). A stray re-ack of an
/// absent/torn-down transfer is a clean no-op-and-ack: it touches no state and still acks
/// `Released`, so an at-least-once redelivery is idempotent without polluting the journal (the
/// record-then-send guard sees `session.transfer == None` and skips the write, exactly as on
/// `apply_abort`). NOT a routing failure (uncounted) — `transfer_unroutable` stays about genuine
/// failures. Always acks `Released`.
fn apply_release(session: &mut Session, transfer: TransferId) -> Option<TransferControlAck> {
    if session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer)
    {
        session.transfer = None; // prune ONLY the matching in-flight transfer (the last remnant)
    }
    tracing::debug!(
        transfer = transfer.0,
        "ReleaseSubscribe closes the source subscription (demote tail) + clears session.transfer"
    );
    Some(TransferControlAck::Released { transfer })
}

// S3 — THE CUT-MARKER OBSERVER IS RETIRED. The cut is now driven ENTIRELY server-side:
// `apply_request_cut` self-acks `CutConfirmed` (server-timed advance) and `apply_freeze` derives
// the real input-cut seq from `last_input_seq` at install time. A client's stamped `CUT_MARKER`
// (which the OLD client still emits on a `ServerControlMsg::RequestCut`) no longer drives anything
// — it is an ordinary input routed by `route_input` like any other, so the former `on_cut_marker`
// per-input observer (a cold `peek_is_cut_marker` decode for every mid-transfer input) is DELETED.
//
// This is what makes MULTI-HOP durable crossings work: on hop 2+ the client's session stays bound
// to the FIRST shard's port, so its marker never reaches the current authority's saga — under the
// old marker-driven cut that starved the `Cutting` step into `CutTimeout`→abort. Server-timing the
// cut removes that client dependency entirely. The `TransferProgress::cut_requested` field is kept
// (set by `apply_request_cut`) as the F1 phase-order record, harmless now that no observer reads it.
