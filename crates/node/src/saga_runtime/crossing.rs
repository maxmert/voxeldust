//! THE CROSSING: a subject's state on the wire, and the two requests that start one.
//!
//! Owns: the envelope the saga ships to the destination at or after commit, the stash that holds the
//! source's flushed pose until the FSM asks for it, and the consumers of a shard's durable and
//! transient crossing requests — each of which resolves the destination realm's owner and hands the
//! answer back.
//!
//! Does NOT own: the geometry. A shard decided that its subject left; this module never re-judges
//! that, and never asks HOW the subject moves — durable and transient differ in their commit point
//! alone (HR2).

use super::SagaRuntimeRes;
use vd_core::pose::StampedPose;
use vd_core::{EpochId, Fence, NodeId, SessionId, TransferId};
use vd_sim::directory::DirectoryCore;
use vd_sim::io::MsgClass;
use vd_sim::runtime::OutboundBox;
use vd_sim::saga::SagaCtx;
use vd_wire::intershard::{
    CrossingRequest, ExteriorCrossingRequest, InterShardFlow, STUB_CROSSING_STEP,
    TRANSFER_SCHEMA_VERSION, TransferEnvelope, TransientCrossingGrant, TransientCrossingRequest,
    TransitionPayload, crossing_transfer_id,
};
use vd_wire::seams::directory::DirectoryKey;

/// Build the `StubCrossing` envelope the saga ships to the dest at/after commit (1d.1). Returns
/// `None` (a no-op) for a non-Entity subject or a missing flushed pose — the pose-before-promote
/// gate makes the latter unreachable for an Entity subject. The `fence` is the post-CAS authority
/// fence (fence rule 1: the receiver rejects a stale crossing); `state` is empty (the TLV blob
/// lands at 1d.6). Monomorphic helper so the executor arm stays a branchless dispatch (HR5).
fn build_crossing(
    ctx: &SagaCtx,
    fence: Fence,
    flush_pose: Option<StampedPose>,
    flush_state: &[u8],
    epoch: EpochId,
) -> Option<InterShardFlow> {
    // An exterior's envelope names the child by the entity its `Ship` key is named by; every other
    // subject keeps the per-entity extraction (a hull can never be mis-adopted by an occupant path).
    let entity = match ctx.subject {
        DirectoryKey::Ship(entity) if ctx.exterior => entity,
        other => other.transfer_subject_entity()?,
    };
    // THE POSE TRAVELS VERBATIM, carrying the SOURCE's own frame tag. The saga is a courier: it holds no
    // ephemeris and knows where no realm sits, so there is nothing here it could correctly convert.
    //
    // This line used to call `rebind_pose_to_dest(.., &IdentityFrames)`, which RELABELS: it stamped the
    // destination's frame onto the pose without moving the number. Measured on the worked example, that
    // turned "3 m from the planet's centre" into "3 m from the star's centre" purely by renaming it — and
    // because the receiver's own conversion then saw a pose already wearing its frame, it took the
    // same-frame shortcut and added nothing. The player arrived at the star. Shipping the source's tag is
    // what makes the receiver's conversion actually fire, so the parent adds its child's placement.
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
            // The exterior blob the source flushed, verbatim; empty for an occupant (D-31 owes theirs).
            state: flush_state.to_vec(),
        },
    }))
}

/// Stash the source's flushed pose onto its live saga, BEFORE the `SourceFlushed` event is
/// stepped — so an `EmitCrossing` reachable on the same tick reads it. A flush for an unknown /
/// GC'd saga is a stale reply: dropped (the event deliver is also a no-op).
pub(crate) fn stash_flush(
    runtime: &mut SagaRuntimeRes,
    transfer: TransferId,
    pose: StampedPose,
    state: Vec<u8>,
) {
    if let Some(live) = runtime.sagas.get_mut(&transfer) {
        live.flushed_pose = Some(pose);
        live.flushed_state = state;
    }
}

/// Push the crossing to the dest if one can be built (Entity subject + a stashed pose), else a
/// LOUD no-op. Holds the Some/None branch (covered both ways by unit tests) so the executor arm
/// stays branchless (HR5). For an Entity subject the pose-before-promote gate guarantees a pose,
/// so the None arm is reached only by the non-Entity subjects the FSM proptests drive.
pub(crate) fn emit_crossing(
    ctx: &SagaCtx,
    fence: Fence,
    flush_pose: Option<StampedPose>,
    flush_state: &[u8],
    epoch: EpochId,
    outbox: &mut OutboundBox,
) {
    match build_crossing(ctx, fence, flush_pose, flush_state, epoch) {
        Some(crossing) => outbox.push_flow(ctx.dest, MsgClass::Saga, &crossing),
        None => tracing::warn!(
            transfer = ctx.transfer.0,
            "EmitCrossing skipped: non-Entity subject or no flushed pose"
        ),
    }
}

/// Slice 3f-B — consume a DURABLE `CrossingRequest` from a source shard's boundary detector: resolve the
/// subject's current owner, the dest realm's owner, and the client's gateway (from the `Session` head), then
/// START a durable crossing saga on the EXISTING `start_transfer` entry. The saga's `ctx.transfer` is the id
/// the SOURCE latched — `crossing_transfer_id(req.subject, req.subject_fence)` over the WIRE fence — so the
/// eventual `Demote`/abort terminal carries the exact id the source's `RequestInFlight` latch keys on, even
/// if the head fence advanced between latch and resolve. `ctx.expected_fence` is the CURRENT head fence (the
/// CAS expectation), which is distinct from the id fence by design.
///
/// MONOMORPHIC (no generic body): ALL branching is the single 3-arm `match` on the three head reads (NOT a
/// let-else), so each outcome — start / unresolved / subject-gone — is a covered region (HR5). An
/// unresolved crossing is COUNTED + warned, never replied to (no saga ⇒ no abort machinery to reply
/// from); the SOURCE heals it (D-WORLD-2): its armed ttl re-drive re-presents the SAME id (absorbed
/// by the `contains_key` guard if a dup DID start), and on budget exhaustion it aborts locally,
/// clearing its own latch — so this arm is a bounded retry window, not a strand.
pub(crate) fn handle_crossing_request(
    runtime: &mut SagaRuntimeRes,
    dir: &DirectoryCore,
    _outbox: &mut OutboundBox,
    req: CrossingRequest,
) {
    // The id the source latched — derived from the WIRE fence (`req.subject_fence`) + the source's
    // per-attempt counter (`req.attempt`), NOT the head fence, so a terminal carrying it matches the source
    // latch even if the head advanced, and a post-abort re-cross's fresh attempt gets a distinct id (H2).
    let transfer = crossing_transfer_id(req.subject, req.subject_fence, req.attempt);
    match (
        dir.head(req.subject),
        dir.head(DirectoryKey::Realm(req.to_realm)),
        dir.head(DirectoryKey::Session(req.session)),
    ) {
        // REDELIVERY GUARD (mirrors `process_starts`' `contains_key`): a re-delivered request for a crossing
        // whose saga is ALREADY live is absorbed — no second enqueue, no re-count — so `crossings_started`
        // reflects DISTINCT started sagas, not requests seen. The source re-drives the request every tick the
        // entity stays over the boundary (`ReDriven`), so this arm is the common steady-state case.
        (Some(_subj), Some(_dest_rec), Some(_sess_rec))
            if runtime.sagas.contains_key(&transfer) => {}
        (Some(subj), Some(dest_rec), Some(sess_rec)) => {
            let ctx = SagaCtx {
                transfer,
                session: req.session,
                subject: req.subject,
                // The CURRENT head fence is the CAS expectation (distinct from the WIRE-fence-derived id).
                expected_fence: subj.fence,
                source: subj.authority.node(),
                dest: dest_rec.authority.node(),
                class: vd_core::entity_kind::DurabilityClass::Durable,
                needs_provision: false,
                from_realm: req.from_realm,
                to_realm: req.to_realm,
                // Thread the request's `to_parent` VERBATIM — a ★DEAD wire field kept for shape only
                // (its consumer `rebind_pose_to_dest` is deleted, D-PLACE-1; the dest forms an Area
                // frame from its own roster at adopt). Flag-day removal ledgered D-WIRE-1.
                to_parent: req.to_parent,
                exterior: false,
            };
            let gateway = sess_rec.authority.node();
            // Stage A (rehome_one_mechanism §4u): the request→saga binding line. The FLUSH runs on
            // `flush_source_node` — the subject's CURRENT directory head — not on the shard whose scan
            // fired the request (`from_realm`). When those part company, the flush ships a different
            // realm's frame; this line beside the per-node HAND-OFF lines makes that visible per saga.
            tracing::info!(
                transfer = ?ctx.transfer,
                subject = ?req.subject,
                from_realm = ?req.from_realm,
                to_realm = ?req.to_realm,
                attempt = req.attempt,
                flush_source_node = ?ctx.source,
                dest_node = ?ctx.dest,
                expected_fence = ?ctx.expected_fence,
                "CROSSING SAGA STARTED",
            );
            runtime.start_transfer(ctx, gateway);
            runtime.crossings_started += 1;
        }
        // The subject owner is known but the dest realm OR the session route is unresolved: no saga can start
        // this tick. COUNTED only — no saga exists, so there is no abort machinery to reply from; the
        // SOURCE heals the drop (D-WORLD-2): its armed ttl re-drive re-presents this same request a
        // budgeted number of times, then aborts locally and clears its own latch.
        (Some(subj), _, _) => {
            runtime.crossing_unresolved += 1;
            // Stage A: the unresolved drop (§4u refutation 6 — a permanent strand before the D-WORLD-2
            // cure armed the source re-drive/exhaustion). Still WARN, never silent: a run of these is
            // an unhosted/unresolvable realm being retried, and the source's exhaustion abort is the
            // matching "CROSSING EXHAUSTED" line in its shard log.
            tracing::warn!(
                transfer = ?transfer,
                subject = ?req.subject,
                from_realm = ?req.from_realm,
                to_realm = ?req.to_realm,
                dest_head_missing = dir.head(DirectoryKey::Realm(req.to_realm)).is_none(),
                session_head_missing = dir.head(DirectoryKey::Session(req.session)).is_none(),
                source_node = ?subj.authority.node(),
                "CROSSING UNRESOLVED: dropped with no reply — the source ttl re-drive owns the retry",
            );
        }
        // No directory owner for the subject at all (authority already moved/revoked): a counted drop.
        (None, _, _) => runtime.crossing_subject_gone += 1,
    }
}

/// Slice 3f-C — consume a TRANSIENT `TransientCrossingRequest`: resolve the dest realm's owner and GRANT it
/// back to the transport-origin `from`. A transient is NOT a directory `OwnerRecord` (burst isolation — HR2),
/// so `from` (the connection the request arrived on) is the ONLY authoritative reply address; there is no
/// owner lookup for the source. The `batch` id is `crossing_transfer_id(req.subject, req.src_realm_fence)` —
/// deterministic per subject, so a redelivered request re-grants the SAME batch (absorbed at the source's
/// `on_transient_crossing_grant` no-op). The grant is `ReDriven` (the source re-requests) ⇒ plain `push_flow`
/// (Ephemeral), never `Retained`. D-43 #9: the resolved arm ALSO starts the `BatchHandoff` saga (keyed on the
/// same `batch`) so the dest's `BatchAdopted` lands on a live `AwaitAdopt` (see the inline note for why this
/// causes no double-emit).
///
/// MONOMORPHIC (no generic body): the branches are the 2-arm `Option` match plus the plain `if !contains_key`
/// saga-start guard (a `ReDriven` re-request whose saga is already live takes the skip arm) — all HR5-covered.
pub(crate) fn handle_transient_crossing_request(
    dir: &DirectoryCore,
    outbox: &mut OutboundBox,
    runtime: &mut SagaRuntimeRes,
    req: TransientCrossingRequest,
    from: NodeId,
) {
    match dir.head(DirectoryKey::Realm(req.to_realm)) {
        Some(rec) => {
            // Transients carry no per-entity attempt (no durable latch / abort-reply) → attempt 0.
            let batch = crossing_transfer_id(req.subject, req.src_realm_fence, 0);
            outbox.push_flow(
                from,
                MsgClass::Saga,
                &InterShardFlow::TransientCrossingGrant(TransientCrossingGrant {
                    subject: req.subject,
                    dest: rec.authority.node(),
                    to_realm: req.to_realm,
                    dst_realm_fence: rec.fence,
                    batch,
                    // Copy the dest realm's parent VERBATIM so the source can stamp it onto
                    // `TransientStatus::Crossing` for the batch's Area-frame rebind.
                    to_parent: req.to_parent,
                }),
            );
            runtime.transient_crossings_granted += 1;

            // D-43 #9: START the SAME `BatchHandoff` saga the machinery drives (HR2 one-machinery),
            // keyed on `batch` == the id the DEST acks `BatchAdopted` under, so `deliver` routes the
            // adopt onto a LIVE `AwaitAdopt` saga instead of the silent None early-return (the exact
            // symptom: `orch.batch_goes == []`, the dest stuck `Arriving`). This is the HR2/HR3 twin of
            // the durable `handle_crossing_request` (build ctx → `start_transfer`), differing ONLY in
            // `class: Transient` + the inert session/subject/gateway.
            //
            // NO DOUBLE-EMIT: the source emits its batch on the GRANT (the `Held→Crossing` flip in
            // `on_transient_crossing_grant`), NOT on the orchestrator-local go-token (`IssueTransientGo`
            // is wire-silent), so starting the saga at grant time is invisible to the source. The
            // `contains_key` fast-skip mirrors the durable arm — it avoids a redundant `pending` push when
            // the saga is already live from a prior tick's re-request; `process_starts`' own `contains_key`
            // is the AUTHORITATIVE one-saga guard (a transient takes NO directory lock, so a debris burst
            // stays burst-isolated).
            if !runtime.sagas.contains_key(&batch) {
                let ctx = SagaCtx {
                    transfer: batch,
                    // INERT — the transient short-path never reads `ctx.session` (no gateway route-swap).
                    session: SessionId::NONE,
                    // INERT provenance — a transient never enters the directory (`locks_directory_key`
                    // is false), so `subject` is never a CAS/lock key.
                    subject: DirectoryKey::Realm(req.to_realm),
                    // The dest realm-lease fence the batched go-token commits at.
                    expected_fence: rec.fence,
                    // The transport-origin connection = the source shard.
                    source: from,
                    dest: rec.authority.node(),
                    class: vd_core::entity_kind::DurabilityClass::Transient,
                    needs_provision: false,
                    from_realm: req.from_realm,
                    to_realm: req.to_realm,
                    // INERT for the transient batch path: the batch's pose rebind rides `TransientStatus::
                    // Crossing.to_parent` (carried on the GRANT below), never this ctx (the transient saga is
                    // the orchestrator-side `BatchHandoff` choreography, which builds no crossing envelope).
                    to_parent: None,
                    exterior: false,
                };
                // The `gateway` arg is INERT for a Transient (never read, never rendered by `views`) —
                // pass the in-scope `from` rather than fabricate a sentinel NodeId.
                runtime.start_transfer(ctx, from);
            }
        }
        // Unresolved dest realm: emit nothing (the source re-requests; the grant is idempotent, so a
        // later-resolving realm still grants). Counted only.
        None => runtime.transient_dest_unresolved += 1,
    }
}

/// ★ THE EXTERIOR CROSSING REQUEST (the ruler switch, slice 1; owner-approved 2026-09-03): a parent's
/// swept verdict moved its hull out of its bound or into a sibling, and it asks for the hull's
/// EXTERIOR — the `Ship` key that says who authors its placement — to move to the destination's shard.
/// Two heads, not three: the subject's (it must be held by the SENDER, or the request is unattested
/// and counted) and the destination realm's. No session: an exterior has no client input to cut.
/// A destination that is not running yet answers no head, and the source's ttl re-drive owns the
/// retry — the hull keeps flying in its old parent until the destination boots.
pub(crate) fn handle_exterior_crossing_request(
    runtime: &mut SagaRuntimeRes,
    dir: &DirectoryCore,
    req: ExteriorCrossingRequest,
    from: NodeId,
) {
    let transfer = crossing_transfer_id(req.subject, req.subject_fence, req.attempt);
    match (
        dir.head(req.subject),
        dir.head(DirectoryKey::Realm(req.to_realm)),
    ) {
        (Some(_), Some(_)) if runtime.sagas.contains_key(&transfer) => {}
        (Some(subj), _) if subj.authority.node() != from => {
            runtime.exterior_request_unattested += 1;
            tracing::warn!(
                transfer = ?transfer,
                subject = ?req.subject,
                holder = ?subj.authority.node(),
                sender = ?from,
                "EXTERIOR CROSSING UNATTESTED: the sender does not hold the exterior — dropped",
            );
        }
        (Some(subj), Some(dest_rec)) => {
            let ctx = SagaCtx {
                transfer,
                session: SessionId::NONE,
                subject: req.subject,
                expected_fence: subj.fence,
                source: subj.authority.node(),
                dest: dest_rec.authority.node(),
                class: vd_core::entity_kind::DurabilityClass::Durable,
                exterior: true,
                needs_provision: false,
                from_realm: req.from_realm,
                to_realm: req.to_realm,
                to_parent: None,
            };
            tracing::info!(
                transfer = ?ctx.transfer,
                subject = ?req.subject,
                from_realm = ?req.from_realm,
                to_realm = ?req.to_realm,
                attempt = req.attempt,
                source_node = ?ctx.source,
                dest_node = ?ctx.dest,
                expected_fence = ?ctx.expected_fence,
                "EXTERIOR CROSSING SAGA STARTED",
            );
            // No gateway takes part; the walk sends nothing there. The source node stands in the slot.
            runtime.start_transfer(ctx, from);
            runtime.exterior_crossings_started += 1;
        }
        (Some(_), None) => {
            runtime.crossing_unresolved += 1;
            tracing::warn!(
                transfer = ?transfer,
                subject = ?req.subject,
                to_realm = ?req.to_realm,
                "EXTERIOR CROSSING UNRESOLVED: the destination has no head yet — the source re-drives",
            );
        }
        (None, _) => runtime.crossing_subject_gone += 1,
    }
}
