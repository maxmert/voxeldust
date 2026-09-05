//! THE TRANSIENT LANE: debris and projectiles, crossing in BATCHES.
//!
//! Owns: the transients this shard tracks, their Held/Arriving/Departing tiers, the closed-form
//! re-advance that keeps them moving, and the four-phase batch hand-off (release → adopt → promote
//! → complete) with its dead-source and dead-destination closures.
//!
//! Does NOT own: a second transfer machinery. Durable and transient are a POLICY FAN-OUT on ONE
//! set of machinery (HR2) — the difference is the commit point, not the code path. A transient is
//! never a directory record, which is why a thousand-piece burst writes zero directory rows; that
//! isolation is the whole reason the tier exists.

use super::{
    AppliedSteps, Dots, Placements, RealmAuthority, RealmRegions, RequestInFlight, StepOutcome,
    StubConfig, StubStats, flush_pose_for_dest, place_arriving_pose,
};
use crate::io::{Durability, MsgClass};
use crate::runtime::{ClockSample, OutboundBox};
use bevy_ecs::prelude::{Res, ResMut, Resource};
use std::collections::{BTreeMap, BTreeSet};
use vd_core::entity_kind::{DurabilityClass, EntityKind, continuity_of};
use vd_core::glam::DVec3;
use vd_core::kinematics::{self};
use vd_core::placement::PlacementLedger;
use vd_core::pose::{LatticePos, RealmId, StampedPose};
use vd_core::{EntityId, Fence, NodeId, TransferId};
use vd_wire::intershard::{
    InterShardFlow, TRANSFER_SCHEMA_VERSION, TRANSIENT_ABANDON_STEP, TRANSIENT_BATCH_STEP,
    TRANSIENT_COMPLETE_STEP, TRANSIENT_DISCARD_STEP, TRANSIENT_DROP_STEP, TRANSIENT_RELEASE_STEP,
    TransferAck, TransferEnvelope, TransientCrossingGrant, TransientHandoff, TransientItem,
    TransitionPayload,
};

/// One TRANSIENT (debris/projectile) this shard tracks (D-7). Pose-only for D-7a — the ballistic
/// `(pose0, v0)` blob that lets the dest re-advance closed-form is D-7b (`TransientItem.state`). The
/// `Held` subset (Arriving EXCLUDED) is the authoritative ground truth; a transient is NEVER a
/// directory `OwnerRecord` (a 1000-debris burst writes ZERO directory rows — burst isolation, HR2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transient {
    pub pose: StampedPose,
    /// The realm-lease fence the set is anchored to (set at adopt = the dest realm fence). The
    /// `TRANSIENT-AUTHORITY-HELD` oracle cross-checks this against the shard's realm fence + a
    /// committed go-token.
    pub anchor_fence: Fence,
    pub status: TransientStatus,
    /// Slice 3d — the frame-local offset at the END of the previous tick, the START endpoint of THIS
    /// tick's swept boundary segment in `evaluate_realm_boundaries` (the transient twin of
    /// `Dot::prev_offset`). Seeded to the pose offset at every construction site so tick-1's segment
    /// is degenerate, then written LAST each evaluation tick — anti-tunneling over the whole segment.
    /// INERT in production through P3 (the boundary registry is EMPTY).
    pub prev_offset: LatticePos,
}

/// A transient's lifecycle tier (D-7) — the Held-vs-Arriving split is the transient twin of the
/// durable Owned-vs-Ghost split that makes adopt-before-drop free of a double-held tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransientStatus {
    /// AUTHORITATIVELY held by this shard (COUNTED + rendered). `outbound` is `None` for a settled
    /// transient; it is set to the batch id once the item has been EMITTED in a crossing batch — the
    /// item stays authoritative (adopt-before-drop) until the orchestrator's `TransientRelease` for
    /// that batch flips it to `Departing`.
    Held { outbound: Option<TransferId> },
    /// A SOURCE-side pending crossing (the TEST-seeded boundary-heuristic stand-in — the autonomous
    /// geometric trigger is P4/P5): `emit_transient_batch` drains it into ONE `TransientBatch`
    /// envelope to `dest` and transitions the item to `Held{outbound: Some(batch)}`. The item is
    /// still COUNTED here (it has not left yet).
    Crossing {
        dest: NodeId,
        to_realm: RealmId,
        dst_realm_fence: Fence,
        batch: TransferId,
        /// ★DEAD FIELD — carried from the [`TransientCrossingGrant`] and read by NOTHING: the
        /// consumer it existed for (`rebind_pose_to_dest`) is DELETED (D-PLACE-1); the DEST forms
        /// its frame from its own ROSTER (`arrival_frame`). Kept only because the grant's wire shape
        /// carries it (postcard is positional); flag-day removal ledgered D-WIRE-1.
        to_parent: Option<RealmId>,
    },
    /// A mid-flight adopted copy at the DEST (UNCOUNTED — the Ghost analogue: excluded from the
    /// conservation count AND from rendering), tagged with its batch. Flips to `Held{outbound: None}`
    /// on the batch's `TransientDrop` (= PROMOTE; the transient twin of the ordered Ghost→Owned).
    Arriving { batch: TransferId },
    /// A SOURCE-side RELEASED copy (D-7b), retained UNCOUNTED and UNRENDERED across the handoff gap:
    /// on `TransientRelease` the source flips `Held{outbound:Some(b)}→Departing{b}` (so it stops both
    /// counting and rendering BEFORE the dest promotes — the holder set is never `{source,dest}`), and
    /// removes it only on `ReleaseComplete` (after the dest's promote-confirm). Retaining it (rather
    /// than removing on release) lets a dest-crash-mid-promote re-drive the promote against a
    /// still-extant copy. The source-side twin of the durable retained Ghost.
    Departing { batch: TransferId },
}

impl TransientStatus {
    /// Is this transient AUTHORITATIVELY held by its shard (COUNTED for conservation + render)? `Held`
    /// and the pre-emit `Crossing` count (the source still owns a crossing item until release);
    /// `Arriving` (dest mid-flight) and `Departing` (source released, retained) do NOT — the two
    /// UNCOUNTED tiers that make the holder set transit `{source}→{}→{dest}`, never `{source,dest}`.
    /// Monomorphic — the SINGLE answer to "does this shard hold this transient", mirroring
    /// `Authority::simulates` for durable dots.
    #[must_use]
    pub fn is_held(&self) -> bool {
        match self {
            TransientStatus::Held { .. } | TransientStatus::Crossing { .. } => true,
            TransientStatus::Arriving { .. } | TransientStatus::Departing { .. } => false,
        }
    }

    /// Is this transient MID-TRANSFER (D-7b.3) — i.e. losing it to a realm self-fence is an in-flight
    /// transfer LOSS (counted against the kind's `LossBudget`), NOT a resident eviction? Everything
    /// EXCEPT a settled `Held{outbound: None}`: a source item flagged-to-cross (`Crossing`), emitted
    /// and awaiting release (`Held{outbound: Some}`), released-and-retained (`Departing`), or a dest
    /// mid-flight copy (`Arriving`). Monomorphic predicate — the SINGLE answer to "is this a handover
    /// loss".
    #[must_use]
    pub fn is_in_handover(&self) -> bool {
        match self {
            TransientStatus::Held { outbound: None } => false,
            TransientStatus::Held { outbound: Some(_) }
            | TransientStatus::Crossing { .. }
            | TransientStatus::Arriving { .. }
            | TransientStatus::Departing { .. } => true,
        }
    }
}

/// The transients this shard tracks, by entity id (D-7). The `is_held()` subset is the
/// `TRANSIENT-AUTHORITY-HELD` oracle ground truth — anchored to the realm-lease fence, never the
/// directory. Sits beside `Dots`/`GhostColliderRegistration` (a sibling held-set, not a fork).
#[derive(Resource, Debug, Default)]
pub struct OwnedTransients(pub BTreeMap<EntityId, Transient>);

// ---------------------------------------------------------------------------
// Slice 3d/3e — the per-shard geometric transfer-TRIGGER state (`evaluate_realm_boundaries`).
// INERT in production through P3: the trigger early-returns on an EMPTY `RealmBoundaries` registry
// (the composer plants none — behaviour-identical), so ALL of the state below stays untouched in a
// real run; the tests populate `RealmBoundaries` to exercise every arm.
// ---------------------------------------------------------------------------

/// SOURCE consumer of the orchestrator's `TransientCrossingGrant` (Slice 3e): the grant carries the
/// resolved dest + fence + batch id for a transient this shard flagged via `TransientCrossingRequest`,
/// so flip the SOURCE Transient `Held → Crossing{dest, to_realm, dst_realm_fence, batch, to_parent}` — the
/// five grant fields match `TransientStatus::Crossing` EXACTLY (a straight field assign). `emit_transient_
/// batch` then ships it. A grant for a NON-Entity subject, an UNKNOWN transient, or a non-`Held` one (a
/// redelivery after the flip) is a counted no-op (degrade, never panic). Monomorphic so every arm is
/// covered once (HR5).
pub(crate) fn on_transient_crossing_grant(
    grant: TransientCrossingGrant,
    owned: &mut OwnedTransients,
    stats: &mut StubStats,
) {
    let Some(entity) = grant.subject.transfer_subject_entity() else {
        stats.transient_grant_no_entity += 1;
        return;
    };
    // Flip ONLY a SETTLED `Held{outbound: None}` transient → Crossing. A `Held{outbound: Some}` (already
    // emitted this batch), a `Crossing` (already flipped), or an Arriving/Departing (mid-handoff) item is
    // a counted no-op — so a REDELIVERED grant never re-flips + re-emits an in-flight batch (at-least-once).
    match owned.0.get_mut(&entity) {
        Some(t) if matches!(t.status, TransientStatus::Held { outbound: None }) => {
            t.status = TransientStatus::Crossing {
                dest: grant.dest,
                to_realm: grant.to_realm,
                dst_realm_fence: grant.dst_realm_fence,
                batch: grant.batch,
                to_parent: grant.to_parent,
            };
            stats.transient_grants_applied += 1;
        }
        // Unknown transient OR one already crossing/emitted/handing-off (a redelivery) — counted no-op.
        Some(_) | None => stats.transient_grant_noop += 1,
    }
}

/// Per-shard system (D-7b): RE-ADVANCE every AUTHORITATIVELY-HELD transient's pose by its closed-form
/// continuity each tick — debris is a MOVING object, so a frozen-on-cut pose would teleport. Runs on
/// BOTH source and dest, each from its OWN stamped origin (no double-advance: the dest re-advances
/// from the pose it ADOPTED, which the source already advanced to its emit tick, so by composability
/// of constant-velocity motion the dest lands exactly where the source's trajectory would). The
/// UNCOUNTED `Arriving` (dest mid-flight) + `Departing` (source released) tiers are SKIPPED — they are
/// not rendered, and the dest catches up in one closed-form step at promote. Per-shard-LOCAL over
/// `owned.0`, zero cross-shard read → `par_iter_mut`-ready for the D-7c burst (zero logic change). The
/// f64 pose feeds ONLY render + the batch payload, NEVER a control discriminant (Category-A; the
/// crossing trigger compares integer cells, P4/P5).
/// Re-stamp every SIMULATING durable dot's pose to the current tick (Stage B4 machinery fix — the
/// cure for rehome_one_mechanism §4u refutation 8). A durable pose advanced its stamp ONLY when
/// input applied, so an input-idle (parked) occupant was measured — by the containment scan AND the
/// hand-off flush, both of which resolve MOVING-child placements at the POSE's stamp — against a
/// world frozen at its last-input tick: a planet could sweep straight through a parked ship and no
/// crossing would ever fire. Transients already re-advance every tick ([`readvance_transients`]);
/// this is the SAME rule on the durable tier (HR2: one machinery — the class split was an accident,
/// not a design). Players are `Frozen` continuity, so [`kinematics::advance_continuity`] is a pure
/// re-stamp: position untouched — stopped means stopped at P3 (P5 inertia rides this same seam).
pub(crate) fn readvance_dots(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    in_flight: Res<RequestInFlight>,
    mut dots: ResMut<Dots>,
) {
    let now = clock.universe_tick;
    for dot in dots.0.values_mut().filter(|d| d.authority.simulates()) {
        // A dot whose CROSSING IS LATCHED keeps its stamp FROZEN (the up-observation arc's ride
        // measurement): from the latch to the demote BOTH shards legitimately own-emit the avatar
        // for a few ticks, and a re-stamped source row carries a tick as fresh as the destination's
        // — the client's track then flip-flops between the two SPACES at tick granularity for the
        // whole saga tail. Freezing the stamp at the latch makes the destination's rows strictly
        // newer, so the one legitimate frame change (the crossing cut) wins monotonically. The
        // freshness this system exists for (a parked ship measured against a moving world) is not
        // lost: a latched dot's re-decisions are suppressed anyway (the latch's Occupied arm), and
        // the flush re-validation measures the CONVERTED pose in the destination's own band.
        if in_flight.0.contains_key(&dot.entity) {
            continue;
        }
        // Mirrors `readvance_transients`: `saturating_sub` is monotonic-forward-only (a backward
        // target yields dt = 0), accel is ZERO (empty space through P4).
        let dt_s = now.0.saturating_sub(dot.pose.universe_tick.0) as f64 * config.tick_dt_s;
        dot.pose = kinematics::advance_continuity(
            continuity_of(dot.entity),
            dot.pose,
            DVec3::ZERO,
            dt_s,
            now,
        );
    }
}

pub(crate) fn readvance_transients(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    mut owned: ResMut<OwnedTransients>,
) {
    let now = clock.universe_tick;
    for (entity, t) in owned.0.iter_mut() {
        if t.status.is_held() {
            // ★ NO RE-CLAMP ON THE FLIGHT PATH (owner ruling 2026-08-27, the ungoverned rock: *"a
            // transfer NEVER changes that speed — a destination realm's ceiling limits what it may
            // ADD, never what a thing arrives with (a re-clamp is a jump, and a jump is a seam)"*).
            // This scaled every held transient's velocity down to the governed ceiling each tick —
            // and walked the whole roster to find it (the galaxy wedge, 2026-09-05). A piece of
            // debris arrives at its own speed and keeps it. Deleted 2026-09-05.
            // The SINGLE tick_dt_s chokepoint (no inline literal); `saturating_sub` enforces
            // monotonic-forward-only — a backward target yields dt=0 (no motion), never negative time.
            // accel = ZERO: a stub is empty space with no gravity field (P5's SphericalSpace introduces
            // the seed-derived analytic gravity — same primitive, non-zero accel, no system rewrite).
            let dt_s = now.0.saturating_sub(t.pose.universe_tick.0) as f64 * config.tick_dt_s;
            t.pose = kinematics::advance_continuity(
                continuity_of(*entity),
                t.pose,
                DVec3::ZERO,
                dt_s,
                now,
            );
        }
    }
}

/// SOURCE system (D-7): drain the pending transient crossings (`TransientStatus::Crossing`, the
/// TEST-seeded boundary-heuristic stand-in — the autonomous geometric trigger is P4/P5) into ONE
/// `TransientBatch` envelope per dest realm (G-TIER: one envelope per batch, never per item), and
/// transition each emitted item to `Held{outbound: Some(batch)}` (still authoritative — the source
/// holds it until the orchestrator's `TransientDrop`, adopt-before-drop). A shard without its realm
/// lease ships nothing (the Crossing items were dropped on the self-fence).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_transient_batch(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    regions: Res<RealmRegions>,
    placements: Res<Placements>,
    authority: Res<RealmAuthority>,
    mut owned: ResMut<OwnedTransients>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(src_realm_fence) = authority.0 else {
        return;
    };
    // One pass: collect each Crossing item into its batch group AND mark it sent in place (no second
    // fallible lookup — the in-place mutate avoids an uncoverable None arm, HR5).
    struct Group {
        dest: NodeId,
        to_realm: RealmId,
        dst_realm_fence: Fence,
        items: Vec<TransientItem>,
    }
    let mut batches: BTreeMap<TransferId, Group> = BTreeMap::new();
    for (entity, t) in owned.0.iter_mut() {
        if let TransientStatus::Crossing {
            dest,
            to_realm,
            dst_realm_fence,
            batch,
            ..
        } = t.status
        {
            // THE SOURCE'S HALF OF THE CONVERSION, through the SAME helper the durable flush uses
            // (HR3, one rule): convert only into a realm this shard AUTHORS the placement of, and
            // otherwise ship the pose VERBATIM in this shard's own frame, for whoever does know.
            //
            // This was the FOURTH relabel — an unconditional identity rebind that stamped the
            // destination's frame onto the pose without moving the number. See
            // `flush_pose_for_dest` for what that costs once realms stop sitting on top of each other.
            //
            // The RECEIVER'S half now exists too: `adopt_transient_batch` runs the SAME
            // `place_arriving_pose` guard as the durable ingress (the KNOWN GAP this comment used to
            // state is CLOSED — audit :105/:374/:384), so an UPWARD hand-off's child-frame pose gets
            // the child's placement ADDED by the receiver, and a pose the receiver cannot measure is
            // refused + counted, never stored verbatim.
            let Some(pose) = flush_pose_for_dest(
                t.pose,
                to_realm,
                &config,
                &regions,
                &placements.0,
                clock.universe_tick,
                &mut stats,
                true,
            ) else {
                // Counted + logged inside the helper. KEEP IT: the item goes back to plain `Held`, so
                // this shard is still its authority and it is still somewhere real. Shipping it anyway
                // would hand a peer a position nobody computed; dropping it while marking it sent would
                // strand it in a batch that never existed.
                t.status = TransientStatus::Held { outbound: None };
                continue;
            };
            batches
                .entry(batch)
                .or_insert(Group {
                    dest,
                    to_realm,
                    dst_realm_fence,
                    items: Vec::new(),
                })
                .items
                .push(TransientItem {
                    entity: *entity,
                    pose,
                    state: Vec::new(),
                });
            t.status = TransientStatus::Held {
                outbound: Some(batch),
            };
        }
    }
    for (batch, g) in batches {
        let env = TransferEnvelope {
            transfer_id: batch,
            universe_epoch: clock.epoch,
            schema_version: TRANSFER_SCHEMA_VERSION,
            fence: g.dst_realm_fence,
            step_id: TRANSIENT_BATCH_STEP,
            class: DurabilityClass::Transient,
            payload: TransitionPayload::TransientBatch {
                from_realm: config.realm,
                to_realm: g.to_realm,
                src_realm_fence,
                dst_realm_fence: g.dst_realm_fence,
                source_tick: clock.local_tick,
                items: g.items,
            },
        };
        outbox.push_flow_durable(
            g.dest,
            MsgClass::Saga,
            &InterShardFlow::Transfer(env),
            Durability::Retained,
        );
        stats.transients_emitted += 1;
    }
}

/// DEST adopt of a transient batch (D-7): journal the batch step idempotently; on FIRST delivery,
/// run THE RECEIVER'S CONVERSION on each item ([`place_arriving_pose`] — the SAME rule the durable
/// crossing and the D-37 re-home ingresses run; this was the third place a pose enters a shard and
/// the only one without the guard, audit :105/:374/:384) and insert the placed items into
/// `OwnedTransients` as the uncounted `Arriving` tier (anchored to the batch's committed realm
/// fence); ALWAYS ack `BatchAdopted` to the orchestrator (at-least-once — the ack GATES the
/// adopt-before-drop `TransientDrop`, so a lost ack must be re-ackable). A redelivery re-acks
/// WITHOUT re-adopting (the items are already Arriving/Held).
///
/// CONVERT-OR-REFUSE, as policy fan-out on the ONE machinery (HR2), not a fork: an item whose pose
/// this shard cannot measure (a frame it was never told the placement of, or an instant its book
/// does not retain) is REFUSED — counted (`transient_arrivals_unplaceable`) + logged, NEVER stored
/// verbatim to be read later as an own-frame number by AoI/containment. Where a DURABLE refusal
/// aborts the whole crossing (the source keeps authority until the saga re-drives), a TRANSIENT
/// item is one of a batched swarm whose source releases on the batch ack, so the refusal is an
/// accounted per-item loss within the Transient class budget — the batch still acks and the
/// placeable items still adopt.
#[allow(clippy::too_many_arguments)]
pub(crate) fn adopt_transient_batch(
    transfer: TransferId,
    to_realm: RealmId,
    dst_realm_fence: Fence,
    items: Vec<TransientItem>,
    config: &StubConfig,
    regions: &RealmRegions,
    placements: &PlacementLedger,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    match applied.journal_step(transfer, TRANSIENT_BATCH_STEP) {
        StepOutcome::FirstApply => {
            for item in items {
                // SANITIZE network input at the decode-ingress chokepoint (D-7b): a corrupt /
                // diverged sender could carry NaN/Inf, which would poison the ballistic
                // re-advance + the render — never trust the wire pose.
                let sanitized = item.pose.sanitized();
                // THE RECEIVER'S CONVERSION — an UPWARD hand-off arrives in the child's frame and
                // only THIS realm can add where it put that child (SL1); a pose already in this
                // realm's frame passes verbatim; anything else is the refusal arm below.
                let pose = match place_arriving_pose(
                    sanitized, to_realm, config, regions, placements, stats,
                ) {
                    Ok(placed) => placed,
                    Err(err) => {
                        stats.transient_arrivals_unplaceable += 1;
                        tracing::error!(
                            %err,
                            transfer = ?transfer,
                            entity = %item.entity,
                            arriving = ?sanitized.frame,
                            into = ?to_realm,
                            own = ?config.realm,
                            "refusing a transient item this shard cannot place — NOT adopted (a \
                             counted loss within the Transient class budget)"
                        );
                        continue;
                    }
                };
                owned.0.insert(
                    item.entity,
                    Transient {
                        pose,
                        anchor_fence: dst_realm_fence,
                        status: TransientStatus::Arriving { batch: transfer },
                        // Seed to the adopted pose offset: the first evaluation segment is degenerate.
                        prev_offset: pose.pos,
                    },
                );
                stats.transients_adopted += 1;
            }
        }
        StepOutcome::AlreadyApplied => stats.transients_adopt_redelivered += 1,
    }
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::BatchAdopted {
            transfer_id: transfer,
            step_id: TRANSIENT_BATCH_STEP,
        }),
    );
}

/// CA-1 S3/S4 DEST LIVENESS — re-emit `BatchAdopted` every tick for each DISTINCT batch this shard still
/// holds `Arriving`. This is the liveness half of the over-discard guarantee: the orchestrator's
/// `dest_adopted` latch is set from a `BatchAdopted` in the inbox, so re-presenting the ack every tick means
/// a transiently-lost single ack never strands the batch (the latch is re-established the next tick the ack
/// lane delivers) — and on the exact budget-maturity tick the ack is in the inbox to be latched pre-scan.
/// Gated PURELY on `Arriving`-presence (adopt is LEASE-FREE — NO authority/realm gate; a `self_fence`
/// would have already dropped the `Arriving` items via `self_fence_drop_transients`, so a self-fenced
/// holder naturally re-drives nothing). DISTINCT batches only (`BTreeSet`, deterministic order): a batch of
/// N items yields ONE ack, not N — the MMO-scale discipline (a 1000-bullet batch is one re-drive, never
/// 1000). Idempotent at the orchestrator: a re-driven `BatchAdopted` on a saga past `AwaitAdopt` is absorbed
/// by the FSM catch-all. TERMINATION: a permanently-orphaned `Arriving` item is dropped by the realm
/// self-fence (`self_fence_grace_ticks > 0`, enforced at prod boot) → the re-drive then stops (nothing
/// Arriving); in a finite test rig it is bounded by the run length.
pub(crate) fn redrive_pending_adoptions(
    config: Res<StubConfig>,
    owned: Res<OwnedTransients>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let mut batches: BTreeSet<TransferId> = BTreeSet::new();
    for t in owned.0.values() {
        if let TransientStatus::Arriving { batch } = t.status {
            batches.insert(batch);
        }
    }
    for batch in batches {
        outbox.push_flow(
            config.orchestrator,
            MsgClass::Saga,
            &InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id: batch,
                step_id: TRANSIENT_BATCH_STEP,
            }),
        );
        stats.batch_adopts_redriven += 1;
    }
}

/// SOURCE — phase 1 of the structural drop-before-promote (D-7b): on `TransientRelease` flip this
/// batch's `Held{outbound: Some(b)}` items to the UNCOUNTED `Departing{b}` tier (the source stops
/// counting + rendering BEFORE the dest promotes — a clean hand-off, NOT a loss) and ALWAYS ack
/// `DropApplied`(`TRANSIENT_RELEASE_STEP`) to gate the dest promote. Journaled idempotent: a
/// redelivery re-acks WITHOUT re-flipping (at-least-once). The item is RETAINED (not removed) so a
/// dest-crash-mid-promote can re-drive against it; `on_release_complete` retires it later.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_transient_release(
    rel: TransientHandoff,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
    config: &StubConfig,
    outbox: &mut OutboundBox,
) {
    match applied.journal_step(rel.transfer, TRANSIENT_RELEASE_STEP) {
        StepOutcome::FirstApply => {
            for t in owned.0.values_mut() {
                if let TransientStatus::Held { outbound: Some(b) } = t.status
                    && b == rel.transfer
                {
                    t.status = TransientStatus::Departing { batch: b };
                    stats.transients_handed_off += 1;
                }
            }
        }
        StepOutcome::AlreadyApplied => stats.transient_release_noop += 1,
    }
    // ALWAYS ack (at-least-once — the ack GATES the dest promote, so a lost ack must be re-ackable).
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: rel.transfer,
            step_id: TRANSIENT_RELEASE_STEP,
        }),
    );
}

/// DEST — phase 2 (PROMOTE, D-7b): on `TransientDrop` flip this batch's `Arriving{b}` items to
/// authoritative `Held{outbound: None}`, re-anchoring to the batch's commit fence, and ALWAYS ack
/// `DropApplied`(`TRANSIENT_DROP_STEP`) (the promote-confirm that drives the source `ReleaseComplete`).
/// Reachable only after the source released (the orchestrator gates it on the source's `DropApplied`),
/// so the source is already uncounted — the holder set is never `{source, dest}`. Journaled idempotent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_transient_promote(
    promote: TransientHandoff,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
    config: &StubConfig,
    outbox: &mut OutboundBox,
) {
    match applied.journal_step(promote.transfer, TRANSIENT_DROP_STEP) {
        StepOutcome::FirstApply => {
            for t in owned.0.values_mut() {
                if let TransientStatus::Arriving { batch } = t.status
                    && batch == promote.transfer
                {
                    t.status = TransientStatus::Held { outbound: None };
                    t.anchor_fence = promote.fence;
                    stats.transients_promoted += 1;
                }
            }
        }
        StepOutcome::AlreadyApplied => stats.transient_drop_noop += 1,
    }
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: promote.transfer,
            step_id: TRANSIENT_DROP_STEP,
        }),
    );
}

/// SOURCE — phase 3 (D-7b): on `ReleaseComplete` (after the dest's promote-confirm) RETIRE this
/// batch's retained `Departing{b}` items. STATE-idempotent (like the durable retained-ghost teardown,
/// `remove_retained_ghost`): a redelivery finds no `Departing` item and is a counted no-op
/// (`transient_release_noop`). D-7d: it now ALWAYS acks `DropApplied`(`TRANSIENT_COMPLETE_STEP`) — the
/// `SourceRetired` signal that drives the saga's `BatchHandoff` tail to `Done` (so a lost
/// `ReleaseComplete` is RE-DRIVEN by the saga's `AwaitComplete` Timeout, not left until a realm
/// self-fence). The ack fires on BOTH paths (removed + already-gone) so a redelivery is still ackable;
/// the orchestrator's tombstoned saga absorbs the duplicate as a no-op.
pub(crate) fn on_release_complete(
    rc: TransientHandoff,
    owned: &mut OwnedTransients,
    stats: &mut StubStats,
    config: &StubConfig,
    outbox: &mut OutboundBox,
) {
    let mut to_remove: Vec<EntityId> = Vec::new();
    for (entity, t) in owned.0.iter() {
        if let TransientStatus::Departing { batch } = t.status
            && batch == rc.transfer
        {
            to_remove.push(*entity);
        }
    }
    if to_remove.is_empty() {
        stats.transient_release_noop += 1;
    } else {
        for entity in to_remove {
            owned.0.remove(&entity);
        }
    }
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: rc.transfer,
            step_id: TRANSIENT_COMPLETE_STEP,
        }),
    );
}

/// SOURCE — D-7d dead-DEST resolution: on `TransientAbandon` (the dest died mid-handoff, so the only
/// promote target is gone) DROP this batch's retained items — `Departing{b}` (already released) OR
/// `Held{outbound: Some(b)}` (the dest died before the source released) — as an ACCOUNTED loss: each is
/// removed + bucketed into `transients_lost_in_handover` by kind (the SAME budget the realm self-fence
/// feeds — DRY, so `verify_transient_loss_budget` reads ONE honest population) + counted in
/// `transients_departure_cancelled`. Journaled idempotent (`TRANSIENT_ABANDON_STEP`): a redelivery
/// short-circuits, so the loss counts EXACTLY once. A corrupt kind tag (HR2 — never decode-to-default)
/// is still removed but NOT bucketed (no kind to attribute the loss to — the `Err` arm).
pub(crate) fn on_transient_abandon(
    abandon: TransientHandoff,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
) {
    match applied.journal_step(abandon.transfer, TRANSIENT_ABANDON_STEP) {
        StepOutcome::FirstApply => {
            let mut to_remove: Vec<EntityId> = Vec::new();
            for (entity, t) in owned.0.iter() {
                let in_batch = match t.status {
                    TransientStatus::Departing { batch } => batch == abandon.transfer,
                    TransientStatus::Held {
                        outbound: Some(batch),
                    } => batch == abandon.transfer,
                    _ => false,
                };
                if in_batch {
                    to_remove.push(*entity);
                }
            }
            for entity in to_remove {
                owned.0.remove(&entity);
                stats.transients_departure_cancelled += 1;
                if let Ok(kind) = EntityKind::from_tag(entity.kind_tag()) {
                    *stats.transients_lost_in_handover.entry(kind).or_insert(0) += 1;
                }
            }
        }
        StepOutcome::AlreadyApplied => stats.transient_release_noop += 1,
    }
}

/// DEST — R-6d3c NEVER-restart closure: on `TransientDiscard` (the source died in
/// `BatchHandoff::AwaitAdopt`, PRE-adopt, so the batch is being counted lost) REMOVE any
/// `Arriving{batch==transfer}` item as an ACCOUNTED loss AND POISON `(transfer, TRANSIENT_BATCH_STEP)`
/// — so a LATE outbox replay of the batch adopts as `AlreadyApplied` and never re-inserts an orphan
/// (the exact silent-loss interleave; `adopt_transient_batch` hits its `AlreadyApplied` arm,
/// `stub.rs` `TRANSIENT_BATCH_STEP`). Journaled idempotent by `(transfer, TRANSIENT_DISCARD_STEP)`: a
/// redelivery short-circuits, so the loss counts EXACTLY once. Ack-FREE — the resolving saga is
/// terminal (mirroring `on_transient_abandon`). A corrupt kind tag (HR2 — never decode-to-default) is
/// still removed + counted in `transients_discarded_source_crash` but NOT bucketed (the `Err` arm).
pub(crate) fn on_transient_discard(
    discard: TransientHandoff,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
) {
    match applied.journal_step(discard.transfer, TRANSIENT_DISCARD_STEP) {
        StepOutcome::FirstApply => {
            // POISON the adopt: record `(transfer, TRANSIENT_BATCH_STEP)` so a late replayed
            // `adopt_transient_batch` is `AlreadyApplied` (never re-inserts). The return value is
            // ignored — we only need the row present. (If the adopt already ran, this is a harmless
            // no-op insert; its `Arriving` items are then removed by the loop below.)
            let _ = applied.journal_step(discard.transfer, TRANSIENT_BATCH_STEP);
            let mut to_remove: Vec<EntityId> = Vec::new();
            for (entity, t) in owned.0.iter() {
                if let TransientStatus::Arriving { batch } = t.status
                    && batch == discard.transfer
                {
                    to_remove.push(*entity);
                }
            }
            for entity in to_remove {
                owned.0.remove(&entity);
                stats.transients_discarded_source_crash += 1;
                if let Ok(kind) = EntityKind::from_tag(entity.kind_tag()) {
                    *stats.transients_lost_in_handover.entry(kind).or_insert(0) += 1;
                }
            }
        }
        StepOutcome::AlreadyApplied => stats.transient_release_noop += 1,
    }
}

/// On a realm SELF-FENCE (the lease was taken over / revoked), DROP every transient this shard
/// tracked (D-7) — they were anchored to the now-lost lease with NO hand-off: a counted LOSS (the
/// declared-loss path; D-7b's `LossBudget` gate reads `transients_dropped`). Durable dots are
/// RETAINED (authority.rs owns them); only the held-set-anchored transients are lost. The `+= 0` on
/// an empty set is a covered straight-line no-op (the happy path never loses the realm).
pub(crate) fn self_fence_drop_transients(owned: &mut OwnedTransients, stats: &mut StubStats) {
    // GROSS eviction count (ops visibility) — every tier.
    stats.transients_dropped += owned.0.len() as u64;
    // HANDOVER-attributable per-kind LOSS (the D-7b.3 budget gate reads THIS, never the gross count):
    // only `is_in_handover()` items are an in-flight transfer loss; a settled `Held{outbound: None}`
    // is a resident eviction OUT of budget scope. A corrupt kind tag (HR2 — never decode-to-default)
    // is counted GROSS but NOT bucketed (the `Err` arm — it has no kind to attribute the loss to).
    for (entity, t) in owned.0.iter() {
        if t.status.is_in_handover()
            && let Ok(kind) = EntityKind::from_tag(entity.kind_tag())
        {
            *stats.transients_lost_in_handover.entry(kind).or_insert(0) += 1;
        }
    }
    owned.0.clear();
}
