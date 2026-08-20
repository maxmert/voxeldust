//! THE DESTINATION'S ADOPT: the crossed entity STATE, journaled exactly once.
//!
//! Owns: the idempotency journal keyed on `(transfer, step)` — the same key the gateway's RAM
//! journal and the durable table use, one machinery at three altitudes — the buffer that holds a
//! crossing which raced ahead of its dot's grant flip, and the apply that lands the pose on the
//! adopted dot.
//!
//! Does NOT own: the decision to accept. A crossing below the dot's recorded authority fence is a
//! stale replay and is refused here on the same rule every other receiver uses. Nor does it own
//! the conversion of the pose it applies — it receives a value already measured in this realm's
//! frame.

use super::{
    Dot, Dots, OwnedTransients, RealmRegions, StubConfig, StubStats, adopt_transient_batch,
    place_arriving_pose,
};
use crate::io::MsgClass;
use crate::runtime::OutboundBox;
use bevy_ecs::prelude::Resource;
use std::collections::{BTreeMap, BTreeSet};
use vd_core::placement::PlacementLedger;
use vd_core::pose::StampedPose;
use vd_core::{EntityId, EpochId, Fence, SessionId, TransferId};
use vd_wire::intershard::{InterShardFlow, TransferAck, TransferEnvelope, TransitionPayload};

/// The outcome of journaling one transferred-entity-state step (1d.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// This `(transfer, step_id)` was not yet applied — the caller MUST apply the effect.
    FirstApply,
    /// A redelivery of an already-applied step — the caller MUST NOT re-apply (re-ack only).
    AlreadyApplied,
}

/// The dest shard's idempotency journal for transferred-entity-STATE steps (1d.0): the IN-MEMORY
/// backing of the frozen `IdempotencyKey::TransferStep{transfer, step_id}` dedup. It keys on the
/// `(TransferId, step_id)` tuple — the SAME key the gateway's per-session RAM journal uses
/// (`gateway.rs` `recorded`/`journal`) and the durable redb `applied_steps` table will use at P3
/// (HR3 ONE machinery, many stores; the store differs per altitude, the KEY and the
/// consult-before-effect / record-after-effect discipline do not — DEFERRED D-22). It is NEVER a
/// dest-local key.
///
/// 1d.0 lands ONLY this primitive (NO Transfer-arm receiver yet). The 1d.1 receiver consults it
/// BEFORE applying a `StubCrossing` step and records AFTER — the ordering-discipline gate is 1d.1;
/// here we only prove the primitive is idempotent. (An in-mem backing cannot crash mid-step — the
/// durable crash window is a P3 concern, D-22.)
///
/// OWED (1d.1, when the receiver feeds this): a RETENTION BOUND — drop a transfer's steps on its
/// terminal (as the gateway RAM journal does, dropped-whole on terminal/`Bye`), so a long-lived
/// dest shard does not accumulate one entry per `(transfer, step)` forever. The set cannot grow in
/// 1d.0 (nothing journals into it yet); the bound belongs with the receiver that knows terminality.
#[derive(Resource, Debug, Default)]
pub struct AppliedSteps(BTreeSet<(TransferId, u32)>);

impl AppliedSteps {
    /// Journal one transfer step by its `IdempotencyKey::TransferStep` components (passed as the
    /// canonical `(transfer, step_id)`, never a dest-local id). Idempotent: the FIRST call records
    /// and returns [`StepOutcome::FirstApply`]; every redelivery of the same key returns
    /// [`StepOutcome::AlreadyApplied`] WITHOUT re-effect. A distinct `step_id` or `transfer` is
    /// independent.
    pub fn journal_step(&mut self, transfer: TransferId, step_id: u32) -> StepOutcome {
        if self.0.insert((transfer, step_id)) {
            StepOutcome::FirstApply
        } else {
            StepOutcome::AlreadyApplied
        }
    }

    /// Non-mutating probe: has `(transfer, step_id)` been journaled? Used by the relocated dest
    /// promote (1d.5b.3b) to gate the `Ghost→Owned` flip on the crossing pose having LANDED
    /// (`STUB_CROSSING_STEP` applied) — pose-before-promote, so a `Promote` racing ahead of its
    /// crossing never flips a poseless dot. Read-only (unlike `journal_step`, which records).
    #[must_use]
    pub fn is_applied(&self, transfer: TransferId, step_id: u32) -> bool {
        self.0.contains(&(transfer, step_id))
    }
}

/// One entity-state crossing held until its dot is adopted (1d.1). The `StubCrossing`, emitted by
/// the saga at the CAS, races AHEAD of the dest's adopt (which is gated behind
/// `OpenInputSlot`→`HeadRead`→grant-flip under the 1c.8 promote-before-demote model), so a
/// crossing that arrives before the grant flips is BUFFERED here and drained at the flip — the
/// arrival/adopt ordering is decoupled within a SINGLE delivery (no re-emit needed).
#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingCrossing {
    transfer: TransferId,
    step_id: u32,
    fence: Fence,
    pose: StampedPose,
}

/// Crossings buffered awaiting their dot's adopt grant-flip, keyed by the subject entity (1d.1).
/// Drained by `drain_pending_crossing` on the flip. In normal operation it holds ≤(concurrent
/// inbound transfers) entries transiently — the adopt lands within a few ticks.
///
/// OWED (alongside the 1d.0 `AppliedSteps` retention bound, D-22): a cleanup for an entry whose
/// entity NEVER adopts (a misrouted crossing). Both need terminal-awareness the dest shard does
/// not yet have, so both land with the journal-retention slice; neither grows per-tick in a
/// healthy run.
#[derive(Resource, Debug, Default)]
pub struct PendingCrossings(BTreeMap<EntityId, PendingCrossing>);

/// DEST side of the 1d.1 crossing: adopt the crossed entity STATE (pose only in 1d.1). Consumes
/// only `StubCrossing` (other payload kinds are a counted no-op). If the adopt grant has not
/// flipped yet (the crossing, emitted at CAS, races ahead of the dest's adopt under the 1c.8
/// promote-before-demote model), BUFFER it for the flip to drain — do NOT journal/ack until the
/// pose is actually applied.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_transfer_envelope(
    env: TransferEnvelope,
    current_epoch: EpochId,
    config: &StubConfig,
    regions: &RealmRegions,
    placements: &PlacementLedger,
    dots: &mut Dots,
    applied: &mut AppliedSteps,
    pending: &mut PendingCrossings,
    owned: &mut OwnedTransients,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    // transfer_protocol §3.3 fail-safe: a crossing leg whose `universe_epoch` does not match this
    // shard's current epoch is REFUSED — discarded, never applied/buffered/acked — so no entity is
    // ever placed at a stale celestial position. Checked here (before the payload match) so it
    // covers every payload of THIS (`Transfer`) arm — durable crossing, transient batch, initial spawn —
    // and a stale-epoch envelope cannot even strand a `PendingCrossings` entry. The OTHER pose-placing
    // ingress, `on_re_home` (the `InterShardFlow::ReHome` arm), carries the SAME guard, so §3.3 is UNIFORM
    // across both. Counted + fail-loud, mirroring the follower clock's `epoch_mismatches` counted-ignore.
    // (Epoch is exact-match by construction — unlike `schema_version`, a version FLOOR owed at D-31's
    // TLV-blob handshake.)
    if env.universe_epoch != current_epoch {
        stats.crossings_epoch_mismatch += 1;
        return;
    }
    let (entity, to_realm, pose) = match env.payload {
        // `to_realm` is READ, not assumed to be this shard's own realm: a co-hosting shard is the
        // destination for every link of the chain it holds, and the arrival has to be measured from the
        // centre of the realm the crossing NAMED. Assuming `config.realm` here is what re-converted an
        // already-placed pose back up into the parent and re-fired the crossing forever.
        TransitionPayload::StubCrossing {
            entity,
            to_realm,
            pose,
            ..
        } => (entity, to_realm, pose),
        // D-7: the DEST adopts a transient batch into its uncounted `Arriving` tier + acks
        // `BatchAdopted` (the gate that lets the orchestrator emit the adopt-before-drop
        // `TransientDrop`). Handled fully here — never the durable StubCrossing dot machinery.
        // `to_realm` is READ here too (audit :105/:374/:384 — it used to be discarded by `..`):
        // the adopt runs the SAME receiver-side conversion as the durable arm below.
        TransitionPayload::TransientBatch {
            to_realm,
            dst_realm_fence,
            items,
            ..
        } => {
            adopt_transient_batch(
                env.transfer_id,
                to_realm,
                dst_realm_fence,
                items,
                config,
                regions,
                placements,
                owned,
                applied,
                stats,
                outbox,
            );
            return;
        }
        TransitionPayload::InitialSpawn { .. } => {
            stats.crossings_unhandled += 1;
            return;
        }
    };
    // THE RECEIVER'S CONVERSION (see `place_arriving_pose` for the two directions and the third case
    // that is a refusal). A pose this shard cannot measure is DROPPED, not applied: applying it would
    // put the entity at a number nobody computed — which, at the scales this arc exists for, is the
    // player standing at the star instead of on the planet.
    let pose = match place_arriving_pose(pose, to_realm, config, regions, placements, stats) {
        Ok(placed) => placed,
        Err(err) => {
            stats.arrivals_unplaceable += 1;
            // Stage A: the refusal line carries the transfer id (joins the source's "CROSSING REQUEST
            // EMITTED" and the orchestrator's "CROSSING SAGA STARTED") plus the pose's stamp and FULL
            // position — a post-commit refusal here is the permanent strand (§4u refutation 4).
            tracing::error!(
                %err,
                transfer = ?env.transfer_id,
                arriving = ?pose.frame,
                at_tick = pose.universe_tick.0,
                pos_m = ?pose
                    .pos
                    .delta_m(vd_core::pose::LatticePos::ORIGIN, pose.frame.tier()),
                into = ?to_realm,
                own = ?config.realm,
                %entity,
                "refusing a crossing this shard cannot place — the entity is NOT adopted"
            );
            return;
        }
    };
    match dots
        .0
        .iter()
        .find(|(_, d)| crossing_target(d, entity))
        .map(|(s, _)| *s)
    {
        Some(session) => {
            let dot = dots.0.get_mut(&session).expect("just found");
            apply_crossing(
                dot,
                env.transfer_id,
                env.step_id,
                env.fence,
                pose,
                applied,
                config,
                stats,
                outbox,
            );
        }
        None => {
            // The adopt grant has not flipped yet: buffer for the flip to drain (no journal/ack).
            // Count a genuine FIRST buffering only — the saga re-emits the crossing at-least-once
            // (saga.rs Swapping timeout), and each redelivery while still-buffered overwrites the
            // SAME key (bounded), so an unconditional bump would inflate the counter for ONE
            // crossing. `insert` returning `None` is the first-insert signal (the same
            // FirstApply-vs-AlreadyApplied shape as `journal_step`).
            let first = pending
                .0
                .insert(
                    entity,
                    PendingCrossing {
                        transfer: env.transfer_id,
                        step_id: env.step_id,
                        fence: env.fence,
                        pose,
                    },
                )
                .is_none();
            if first {
                stats.crossings_buffered += 1;
            }
        }
    }
}

/// Whether a dot is the ADOPTED dest dot for `entity`: authority-held (`granted`), not departing.
/// Monomorphic predicate.
///
/// 1d.3 dropped a former `simulates()`-style term, and 1d.5b.3b made that DROP load-bearing: the
/// adopt dot stays `Ghost` (NOT simulating) from adopt all the way until the saga `Promote` flips it
/// in `on_saga_promote` (`apply_crossing` only STORES the pose now — it no longer promotes), so a
/// `!simulates()`-style guard would NEVER match the adopt dot and the crossing could never land. With
/// the term dropped, both the pre-promote Ghost AND a post-promote `Owned` redelivery match; the
/// journal returns `AlreadyApplied` on the redelivery, so it re-acks WITHOUT re-applying (a guard that
/// missed it would fall to `on_transfer_envelope`'s `None` arm and re-buffer into `PendingCrossings`
/// forever — a permanent strand + `crossings_buffered` inflation).
/// Sound because AUTHORITY-UNIQUE guarantees at most one granted non-departing dot per entity here,
/// AND the saga addresses the crossing envelope (`InterShardFlow::Transfer`) to the DEST node only,
/// so a widened predicate can never misland on the SOURCE's render-ready dot.
///
/// ⚠️ INTENTIONALLY identical-bodied to [`foreign_takeover_target`] AND [`flush_target`] — the
/// `twin-D1` triplet, DO NOT merge them (the tag is local DRY-legibility bookkeeping, NOT the
/// unrelated DEFERRED `D-1`). The NAMES are the documentation of intent: this finds the DEST dot to
/// APPLY a crossing; `foreign_takeover_target` finds the source's holder to DEMOTE on a takeover;
/// `flush_target` finds the source's holder to SHIP its pose. They serve disjoint call sites on
/// different shards and never fire on the same dot, so the shared body is safe — but a "DRY" merge
/// would break each call site's legibility.
#[must_use]
pub(crate) fn crossing_target(dot: &Dot, entity: EntityId) -> bool {
    (dot.entity == entity) & dot.granted & !dot.departing
}

/// Apply one crossing to its adopted dot (the shared immediate + drained path). Fence rule 1: a
/// crossing below the dot's recorded authority fence is a stale leftover (counted, dropped, NOT
/// acked). Otherwise consult-before-effect via the 1d.0 journal (FirstApply ⇒ sanitize + store;
/// redelivery ⇒ re-ack only), then ack the step either way.
#[allow(clippy::too_many_arguments)]
fn apply_crossing(
    dot: &mut Dot,
    transfer: TransferId,
    step_id: u32,
    fence: Fence,
    pose: StampedPose,
    applied: &mut AppliedSteps,
    config: &StubConfig,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    if fence.is_stale_against(dot.entity_fence) {
        stats.crossings_stale += 1;
        tracing::warn!(
            entity = %dot.entity,
            ?transfer,
            ?fence,
            entity_fence = ?dot.entity_fence,
            "crossing DROPPED as stale — below the dot's recorded authority fence"
        );
        return;
    }
    if applied.journal_step(transfer, step_id) == StepOutcome::FirstApply {
        // Never trust the network: sanitize to finite at this ingress before storing. The crossing
        // STORES the pose but the dot STAYS Ghost — the `Ghost→Owned` promote RELOCATED to
        // `on_saga_promote` (1d.5b.3b) so the demote-before-promote ordering is STRICT (the dest
        // becomes Owned only on the saga `Promote`, after the source has demoted). `on_saga_promote`
        // gates its flip on THIS journal entry (`STUB_CROSSING_STEP` applied) — pose-before-promote,
        // so the relocated promote still never emits a poseless origin-default frame.
        dot.pose = pose.sanitized();
        stats.crossings_applied += 1;
    }
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::Accepted {
            transfer_id: transfer,
            step_id,
        }),
    );
}

/// Drain a crossing buffered before its dot adopted (1d.1): on the adopt grant-flip, apply the
/// stored pose to the now-granted dot. No buffered crossing for the entity ⇒ no-op (an adopt with
/// nothing pending — e.g. a crossing that arrived after adopt, or none at all).
#[allow(clippy::too_many_arguments)]
pub(crate) fn drain_pending_crossing(
    dots: &mut BTreeMap<SessionId, Dot>,
    session: SessionId,
    entity: EntityId,
    applied: &mut AppliedSteps,
    pending: &mut PendingCrossings,
    config: &StubConfig,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let Some(crossing) = pending.0.remove(&entity) else {
        return;
    };
    let dot = dots.get_mut(&session).expect("the just-adopted dot");
    apply_crossing(
        dot,
        crossing.transfer,
        crossing.step_id,
        crossing.fence,
        crossing.pose,
        applied,
        config,
        stats,
        outbox,
    );
}
