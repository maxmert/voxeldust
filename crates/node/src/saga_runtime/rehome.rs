//! THE STANDING RE-HOME: recovering a subject whose owner is confirmed gone.
//!
//! Owns: the adopt envelope, the target selection (the LIVE directory head of the subject's realm —
//! never a realm-blind pick), the deterministic replay-stable transfer id a recovery mints, and the
//! queue drain that starts one.
//!
//! Does NOT own: the decision that an owner is gone (`liveness`) or the sequence that follows
//! (`execute` runs the same FSM every other transfer runs). A re-home is not a second machinery — it
//! is the same saga started by a different trigger (HR3).

use super::{
    LiveSaga, LivenessTracker, PendingReHome, SagaRuntimeRes, commit_result, dead_budget_elapsed,
    run_to_quiescence,
};
use std::collections::BTreeMap;
use vd_core::pose::{RealmId, StampedPose};
use vd_core::{EpochId, Fence, NodeId, SessionId, TransferId, UniverseTick};
use vd_sim::capability::{CapRequest, ShardProfile};
use vd_sim::directory::DirectoryCore;
use vd_sim::io::MsgClass;
use vd_sim::runtime::OutboundBox;
use vd_sim::saga::{self, BatchHandoffPhase, SagaCtx, SagaEvent, SagaState, SagaTuning};
use vd_wire::intershard::{
    InterShardFlow, RE_HOME_STEP, ReHomeCmd, ReHomeState, namespaced_transfer_id,
};
use vd_wire::seams::directory::DirectoryKey;

/// Build the `ReHome` adopt envelope the saga ships to a re-home `target` (D-37). Returns `None` (a
/// no-op) for a non-Entity subject or a missing flushed pose. A DEDICATED arm (NOT `Promote`): the target
/// RECONSTRUCTS the subject from `ReHomeState::PoseOnly`, having no pre-existing ghost. `new_fence` is the
/// post-CAS authority fence (fence rule 1). Monomorphic helper so the executor arm stays branchless (HR5).
pub(crate) fn build_rehome(
    ctx: &SagaCtx,
    new_fence: Fence,
    flush_pose: Option<StampedPose>,
    epoch: EpochId,
) -> Option<InterShardFlow> {
    let _entity = ctx.subject.transfer_subject_entity()?;
    // VERBATIM, in the source's own frame — see `build_crossing` for the whole reason. The re-home adopt is
    // the second place a pose enters a shard, and it is the receiver there that converts.
    let pose = flush_pose?;
    Some(InterShardFlow::ReHome(ReHomeCmd {
        transfer: ctx.transfer,
        universe_epoch: epoch,
        subject: ctx.subject,
        new_fence,
        step_id: RE_HOME_STEP,
        state: ReHomeState::PoseOnly(pose),
        source: ctx.source,
    }))
}

/// Push the re-home adopt to `target` if one can be built (Entity subject + a stashed pose), else a LOUD
/// no-op (mirrors `emit_crossing`). Holds the Some/None branch (covered both ways by unit tests) so the
/// executor arm stays a branchless dispatch (HR5).
pub(crate) fn emit_rehome(
    ctx: &SagaCtx,
    new_fence: Fence,
    target: NodeId,
    flush_pose: Option<StampedPose>,
    epoch: EpochId,
    outbox: &mut OutboundBox,
) {
    match build_rehome(ctx, new_fence, flush_pose, epoch) {
        Some(rehome) => outbox.push_flow(target, MsgClass::Saga, &rehome),
        None => tracing::warn!(
            transfer = ctx.transfer.0,
            "ReHomeAdopt skipped: non-Entity subject or no flushed pose"
        ),
    }
}

#[allow(clippy::too_many_arguments)] // the dead-aware decision needs the full saga + liveness + roster context
pub(crate) fn rehome_event_for(
    state: &SagaState,
    ctx: &SagaCtx,
    liveness: &LivenessTracker,
    dead_observed_since: &mut Option<(NodeId, UniverseTick)>,
    tuning: &SagaTuning,
    now: UniverseTick,
    roster: &BTreeMap<NodeId, ShardProfile>,
    req: &CapRequest,
    subject_owner: NodeId,
    pose_realm_owner: Option<NodeId>,
    dest_adopted: bool,
) -> SagaEvent {
    match state {
        // D-7d / R-6d3c transient batch hand-off: dest-dead abandons the source's retained copy; source-
        // dead splits BY PHASE — a POST-adopt phase self-promotes the dest (zero loss), PRE-adopt
        // `AwaitAdopt` discards-to-dest + counts the loss (`SourceUnreachablePreAdopt`, R-6d3c). BOTH the
        // source-dead and dest-dead resolutions are DESTRUCTIVE now (self-promoting an unadopted batch, or
        // discarding+accounting a loss, are both irreversible), so BOTH are budget-gated on
        // `abort_deadline_ticks` measured from `dead_observed_since` — a source that RESTARTS within budget
        // delivers via its durable outbox replay FIRST (the two recoveries race; the budget picks the
        // restart winner). R-6d3c ADDED the source-dead budget-gate (the former immediate self-promote
        // could not lose a race, but the AwaitAdopt discard MUST give the restart a chance).
        SagaState::BatchHandoff { phase, .. } => {
            // CA-1 S3/S4 OVER-DISCARD SAFETY (retires the R-6d4-F2 tripwire). Once the dest has ADOPTED an
            // `AwaitAdopt` batch (`dest_adopted`, latched in the PRE-SCAN inbox pass), the DEST is the
            // holder-of-record: its fate decides, NOT the source's. So DEFER to the dest-death ladder for
            // that case and NEVER take the source-death resolution — because the `EmitReSolicit` probe now
            // makes `is_confirmed_dead(source)` reachable in AwaitAdopt (it was inert before), and letting
            // the source `if` win would (a) over-discard a batch the dest holds when the dest is alive, and
            // (b) WEDGE forever when BOTH are dead (the source arm returns a bare Timeout that the dest-dead
            // `else if` can never reach). Deferring restores the pre-CA-1-S3 terminal: dest alive → re-drive
            // (the drain advances the phase off the latched `BatchAdopted`, then the post-adopt self-promote
            // handles the dead source); dest dead → the budget-gated `DestUnreachable` abandon+tombstone
            // (admin-visible via `dest_unreachable_resolutions`). The source-death resolutions
            // (discard-if-never-adopted / self-promote-if-post-adopt) apply only when the dest is NOT the
            // holder — i.e. `!defer_to_dest`.
            let defer_to_dest = matches!(phase, BatchHandoffPhase::AwaitAdopt) && dest_adopted;
            if liveness.is_confirmed_dead(ctx.source, now) && !defer_to_dest {
                // Keyed by ctx.source so a dest-dead-then-source-dead cause-switch re-anchors (does NOT
                // measure the source's restart-race budget from the dest's stale first-dead observation).
                if dead_budget_elapsed(dead_observed_since, ctx.source, now)
                    >= tuning.abort_deadline_ticks
                {
                    match phase {
                        // PRE-adopt (`!defer_to_dest` ⇒ dest never adopted here): the dest never received
                        // the batch → discard-to-dest + count the source-crash loss.
                        BatchHandoffPhase::AwaitAdopt => {
                            *dead_observed_since = None;
                            SagaEvent::SourceUnreachablePreAdopt
                        }
                        // POST-adopt phase: the dest provably holds the batch → zero-loss self-promote.
                        _ => {
                            *dead_observed_since = None;
                            SagaEvent::SourceUnreachable
                        }
                    }
                } else {
                    SagaEvent::Timeout // cheap re-drive while the restart-race budget accrues
                }
            } else if liveness.is_confirmed_dead(ctx.dest, now) {
                // The dest is confirmed dead — the DESTRUCTIVE budget gate (a recoverable blip that clears
                // in time never abandons a healthy dest). Reached for a post-adopt dest-death AND (via
                // `defer_to_dest`) for the AwaitAdopt-adopted-then-dead double-crash: both abandon the
                // source's retained copy + tombstone (the go-token can no longer promote a dead dest).
                if dead_budget_elapsed(dead_observed_since, ctx.dest, now)
                    >= tuning.abort_deadline_ticks
                {
                    SagaEvent::DestUnreachable
                } else {
                    SagaEvent::Timeout // cheap re-drive while the abort budget accrues (corpse won't ack)
                }
            } else {
                *dead_observed_since = None; // neither confirmed dead (or deferring to a LIVE dest) → re-drive
                SagaEvent::Timeout
            }
        }
        // D-37 CELL 1: a POST-commit `Demoting` saga whose SOURCE is confirmed dead self-promotes the
        // already-committed live dest (the ordered Demote toward the corpse will never ack). NON-destructive
        // — the directory already committed authority to the dest — so the cheap redrive deadline, no
        // budget. A live-but-slow source only re-drives (record_ack clears the evidence before confirmation).
        SagaState::Demoting { .. } => {
            if liveness.is_confirmed_dead(ctx.source, now) {
                SagaEvent::SourceUnreachable
            } else {
                SagaEvent::Timeout
            }
        }
        // D-37 CELL 2: a POST-commit `Promoting` saga whose committed DEST is confirmed dead FORWARD
        // re-homes onto a live capability-matched target — the dead dest can never `PromoteAck`, and
        // (unlike CELL 1's Demoting source-death) there is no already-committed live owner to self-promote
        // (the dest IS the committed owner). DESTRUCTIVE-budget gated exactly like the BatchHandoff dest-dead
        // ladder (the CSCALE-1 cure: `dead_observed_since` so a recoverable blip that clears in time never
        // re-homes a healthy dest). Once past budget, `select_rehome_target` picks THE live directory
        // owner of the stashed pose's realm — the one node that can place the pose (Stage-C fix; the
        // old lowest-live-roster pick sent the subject to a shard that refused the foreign frame):
        // Some(target) ⇒ `ReHomeTo` (the FSM bumps the fence to `target`); None (no placeable live
        // owner) ⇒ `Timeout`, the saga stays PARKED (honest RED, never a re-home a receiver refuses).
        //
        // ⚠️ The liveness check is on `subject_owner` — the CURRENT directory owner — NOT `ctx.dest` (the
        // original, now-stale dest). After ONE re-home the directory names the LIVE target, so a later fire
        // sees a LIVE owner ⇒ no re-home ⇒ the fence does NOT run away (the saga then merely re-drives /
        // parks on the D-36 starved watermark, owed). Checking `ctx.dest` would re-home EVERY fire (it stays
        // dead forever), bumping the fence past the journal-gated adopt into a FenceMismatch.
        SagaState::Promoting { .. } => {
            if liveness.is_confirmed_dead(subject_owner, now) {
                // Keyed by subject_owner (the committed dead dest); a Promoting saga only ever times this
                // one participant, but the keyed anchor keeps the budget honest if the owner ever changes.
                if dead_budget_elapsed(dead_observed_since, subject_owner, now)
                    >= tuning.abort_deadline_ticks
                {
                    match select_rehome_target(req, pose_realm_owner, roster, liveness, now) {
                        Some(target) => SagaEvent::ReHomeTo { target },
                        None => SagaEvent::Timeout, // no capable live target → stay parked
                    }
                } else {
                    SagaEvent::Timeout // cheap re-drive while the abort budget accrues (dead dest won't ack)
                }
            } else {
                *dead_observed_since = None; // dest healthy/recovered → clear stale budget, re-drive
                SagaEvent::Timeout
            }
        }
        // Every other phase (pre-commit + the forward-only Swapping/Releasing tail + the transient ReHoming
        // CAS-in-flight) → the idempotent Timeout re-drive.
        _ => SagaEvent::Timeout,
    }
}

/// D-37 target selection (Stage-C fix of the realm-blind pick): the forward-re-home target is THE
/// LIVE DIRECTORY OWNER OF THE POSE'S OWN REALM — the one node in the world that can lawfully place
/// the stashed pose (the receiver's `place_arriving_pose` accepts only its own frame and its direct
/// children's; the flushed pose is VERBATIM in the SOURCE realm's frame, so the source realm's
/// current owner is the placeable target — usually the source shard itself, or whoever the realm
/// re-homed to since). The old selection — the lowest live roster node satisfying the caps — was
/// realm-blind: for a P3 bare-point entity it degenerated to "the lowest live NodeId", whose shard
/// then rightly REFUSED the foreign-frame pose forever (the audit's HR3 critical).
///
/// `pose_realm_owner` is resolved by the caller from the directory head of the pose's frame realm —
/// the SAME fenced source of truth every authority answer comes from (never a peer claim, never a
/// roster guess). `None` (no stashed pose / a frame with no single realm owner / no live record) ⇒
/// the saga stays PARKED (honest RED, never a forced re-home a receiver must refuse); the next scan
/// fire re-resolves, so a demand-respawned realm is picked up as soon as its head lands. The
/// capability match stays: a roster profile that cannot satisfy the subject's caps refuses (HR3 —
/// a capability match, never a shard-kind test); a node absent from the static roster (a
/// demand-spawned shard) carries the strongest capability statement available — it already hosts
/// the realm the pose lives in — so it is accepted. Bitwise `&` (HR5).
pub(crate) fn select_rehome_target(
    req: &CapRequest,
    pose_realm_owner: Option<NodeId>,
    roster: &BTreeMap<NodeId, ShardProfile>,
    liveness: &LivenessTracker,
    now: UniverseTick,
) -> Option<NodeId> {
    let node = pose_realm_owner?;
    let alive = !liveness.is_confirmed_dead(node, now);
    let capable = roster.get(&node).is_none_or(|p| p.satisfies(req));
    (alive & capable).then_some(node)
}

/// D-37 Slice 3: a DETERMINISTIC, replay-stable `TransferId` for an orchestrator-minted STANDING re-home.
/// FNV-1a over the postcard bytes of `(subject, prev_fence)` into the low 120 bits, tagged with a fixed
/// high byte (`0x37`, the D-37 namespace) so it can NEVER collide with a client/gateway-assigned
/// `TransferId` (those are minted small + sequential in the connection plane, never with this tag). Pure +
/// branchless (HR5: no rng, no default hasher — the byte-identical seed-replay canary forbids both); the
/// same orphan + fence always derives the same id, so a re-attempt after a lost RAM enqueue is idempotent.
/// One orphan re-homes at most once per fence (the `in_transfer` lock dedups), so per-`(subject, fence)`
/// uniqueness suffices. Slice 3f-D (L2 DRY): the FNV body is the shared [`namespaced_transfer_id`]
/// primitive; only the `0x37` tag + the 2-arg seed are re-home's own.
pub(crate) fn rehome_transfer_id(subject: DirectoryKey, prev_fence: Fence) -> TransferId {
    let seed = postcard::to_allocvec(&(subject, prev_fence)).expect("encode rehome id seed");
    namespaced_transfer_id(0x37, &seed)
}

/// D-37 Slice 3: build the `SagaCtx` for a STANDING re-home. REAL fields carry the recovery: `subject`
/// (the orphan), `expected_fence = prev_fence` (Slice-4's `ReHomeCommit` CAS expectation, `fence+1`),
/// `source = dead_owner` (the `ReHome` envelope's provenance), `dest = target` (the selected live shard),
/// `class = Durable` (a standing re-home is always a durable per-key recovery — transient batches live and
/// die in one realm, never standing-re-homed). PLACEHOLDER fields — the `ReHome` envelope carries NO
/// session/realm and the Slice-4 adopt sources the entity's session/realm/pose from the RealmId-keyed
/// checkpoint reload (D-6/P7), NOT this ctx — so `session`/`from_realm` are NEVER read by the re-home path;
/// they are derived from the entity for replay-determinism + the shared WAL snapshot shape.
///
/// `to_realm` CAVEAT: `build_rehome` ships the flushed pose VERBATIM (the RECEIVER places it —
/// `place_arriving_pose`; the old rebind consumer is deleted, D-PLACE-1), so `to_realm` matters WHEN a
/// pose is present. This standing-reaper path is safe because it ALWAYS parks with `flushed_pose: None`
/// (a pre-flush death has no recoverable pose — see `process_rehome_starts`), so the entity-derived
/// placeholder stays inert. WHEN Slice-4/P7 sources a REAL pose from the RealmId-keyed checkpoint, it
/// MUST supply the true destination realm here (not the `System(entity)` placeholder) or the receiver
/// will place the pose against the wrong realm.
pub(crate) fn rehome_ctx(
    subject: DirectoryKey,
    prev_fence: Fence,
    dead_owner: NodeId,
    target: NodeId,
) -> SagaCtx {
    // Slice 3 only re-homes Entity keys (the reaper leaves Realm/Ship for Slice 4), so this is always Some.
    let entity = subject
        .transfer_subject_entity()
        .expect("a standing re-home subject is an Entity key")
        .0;
    SagaCtx {
        transfer: rehome_transfer_id(subject, prev_fence),
        // PLACEHOLDERS (never read by the re-home — see the doc above): entity-derived for determinism.
        session: SessionId(entity),
        from_realm: RealmId::System(entity as u64),
        to_realm: RealmId::System(entity as u64),
        // REAL recovery fields.
        subject,
        expected_fence: prev_fence,
        source: dead_owner,
        dest: target,
        class: vd_core::entity_kind::DurabilityClass::Durable,
        needs_provision: false,
        // `None`: a ★DEAD wire field either way (its consumer is deleted, D-PLACE-1/D-WIRE-1) — and the
        // standing reaper ALWAYS parks with `flushed_pose: None` (a pre-flush death has no recoverable
        // pose — see the `to_realm` CAVEAT above), so nothing downstream could even reach a pose here.
        to_parent: None,
    }
}

/// D-37 Slice 3: drain the standing re-home queue the reaper filled THIS sweep (a same-tick within-barrier
/// hand-off) and ARM each through the ONE machinery. The parked saga names NO forward target (Stage-C
/// fix of the realm-blind pick): a pre-flush death has no recoverable pose, so no node in the world can
/// be named as placeable — the park records the DEAD OWNER (truthful: authority stays at the corpse)
/// and the Slice-4 adopt, which arrives WITH the checkpoint pose, must select its target pose-aware
/// (`select_rehome_target` over the pose realm's live directory owner) at adopt time, never earlier.
/// `lock_transfer` makes the armed saga the SOLE owner of
/// the key's transfer lifecycle (so the next reaper sweep SKIPS it via the `in_transfer` gate — no
/// re-detection churn); `false` = already locked → skip (one key → one re-home saga). The saga ARMS via
/// `saga::start_rehome` and PARKS in `ReHoming` (CONSERVATIVE Slice-3 split: no `ReHomeCommit`, no adopt —
/// authority stays at the dead owner because a pre-flush death has no recoverable pose until the
/// RealmId-keyed redb lands, D-6/P7/Slice 4). `run_to_quiescence` + `commit_result` PERSIST the parked saga
/// (durable across kill-9; on reboot it rehydrates and the locked record keeps the reaper off it). Runs
/// inside the D-6 group-commit barrier right after the reaper, so the lock + the armed saga are durable the
/// SAME tick. NO fabricated pose (HR1: the dead owner's store is sealed; `flushed_pose: None`).
pub(crate) fn process_rehome_starts(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
) {
    for PendingReHome {
        subject,
        dead_owner,
        prev_fence,
    } in std::mem::take(&mut runtime.pending_rehome)
    {
        // The park's target IS the dead owner — the only truthful value while no pose exists (see
        // the fn doc). Slice 4 re-selects pose-aware; committing any live node here would be the
        // realm-blind pick the Stage-C audit condemned, deferred one slice.
        let target = dead_owner;
        let transfer = rehome_transfer_id(subject, prev_fence);
        if !dir.lock_transfer(subject, transfer) {
            continue; // already locked (a concurrent arm / a prior sweep's saga) → one key, one re-home
        }
        let ctx = rehome_ctx(subject, prev_fence, dead_owner, target);
        let (state, actions) = saga::start_rehome(target, prev_fence);
        runtime.sagas.insert(
            transfer,
            LiveSaga {
                ctx,
                state,
                // No client egress in the re-home tail (the adopt is shard→shard); `dead_owner` is an
                // inert provenance placeholder, never read while parked or in the Slice-4 adopt.
                gateway: dead_owner,
                since: now,
                flushed_pose: None, // HR1: the dead owner's store is sealed; the adopt pose is owed Slice 4
                dead_observed_since: None,
                dest_adopted: false,
                opened: now,
            },
        );
        // Parks immediately (start_rehome emits no actions); run_to_quiescence + commit_result PERSIST the
        // ReHoming snapshot so the armed re-home survives an orchestrator kill-9 (rehydrates parked).
        let (final_state, tombstone, rejected, batch_gos) = run_to_quiescence(
            &ctx, dead_owner, state, actions, dir, outbox, epoch, now, None,
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
}
