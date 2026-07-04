//! G-SEALED conformance gate (`docs/design/sealed_shards.md` §1) — lifted from the
//! in-module unit test to the design-named INTEGRATION location so the closed-set
//! guarantee is a per-release gate, not a convention (audit SEAL-2).
//!
//! The closed taxonomy `InterShardFlow` (the FULL current 16-arm set — see `wire/src/lib.rs`
//! for the canonical enumeration; this header does NOT re-list it to avoid a second copy that
//! drifts, the exact staleness the `arm_tripwire` below structurally prevents) is the ONLY
//! shape that crosses a shard boundary; every arm has a
//! coherent `EffectClass`, and every SIDE-EFFECTING arm carries an idempotency key (so
//! an authority-gating payload can never ride a fire-and-forget channel). Two compile-
//! time tripwires keep this gate honest as arms are added: the exhaustive `match` in
//! `effect_class` (adding an arm without classifying it does not compile) and the
//! `arm_tripwire` below (adding an arm without representing it in `every_arm` here does
//! not compile) — so the per-release surface enumeration can never silently drift.

use vd_core::entity_kind::{DurabilityClass, EntityKind};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::{EntityId, EpochId, Fence, NodeId, SessionId, TickId, TransferId, UniverseTick};
use vd_wire::intershard::{
    EffectClass, FLUSH_SOURCE_STEP, FlowDurabilityClass, FlushSource, GhostFlow, IdempotencyKey,
    InterShardFlow, STUB_CROSSING_STEP, TransferAck, TransferEnvelope, TransferStepRejectReason,
    TransitionPayload,
};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp, DirectoryReply};
use vd_wire::seams::transfer_control::{PrepareResult, TransferControl, TransferControlAck};

fn pose() -> StampedPose {
    StampedPose::at_rest(
        FrameRef::SystemSpace { system_seed: 1 },
        vd_core::glam::DVec3::ZERO,
        UniverseTick(10),
    )
}

fn eid(kind: EntityKind) -> EntityId {
    EntityId::pack(kind, 1, 7, 3)
}

/// EVERY arm of the closed taxonomy, in one place — the per-release surface gate.
fn every_arm() -> Vec<InterShardFlow> {
    let envelope = |payload, class| {
        InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: TransferId(11),
            universe_epoch: EpochId(1),
            schema_version: 1,
            fence: Fence(2),
            step_id: 4,
            class,
            payload,
        })
    };
    vec![
        InterShardFlow::Ghost(GhostFlow::Spawn {
            entity: eid(EntityKind::Ship),
            pose: pose(),
            source_fence: Fence(1),
            since_tick: TickId(2),
        }),
        InterShardFlow::Ghost(GhostFlow::Delta {
            entity: eid(EntityKind::Ship),
            pose: pose(),
            source_fence: Fence(1),
            source_tick: TickId(3),
            seq: 1,
        }),
        InterShardFlow::Ghost(GhostFlow::Despawn {
            entity: eid(EntityKind::Ship),
            source_fence: Fence(1),
        }),
        envelope(
            TransitionPayload::InitialSpawn {
                entity: eid(EntityKind::Player),
                to_realm: RealmId::System(1),
                pose: pose(),
                state: vec![],
            },
            DurabilityClass::Durable,
        ),
        envelope(
            TransitionPayload::StubCrossing {
                entity: eid(EntityKind::Player),
                from_realm: RealmId::System(1),
                to_realm: RealmId::Planet(2),
                pose: pose(),
                state: vec![],
            },
            DurabilityClass::Durable,
        ),
        envelope(
            TransitionPayload::TransientBatch {
                from_realm: RealmId::System(1),
                to_realm: RealmId::Planet(2),
                src_realm_fence: Fence(3),
                dst_realm_fence: Fence(4),
                source_tick: TickId(6),
                items: vec![],
            },
            DurabilityClass::Transient,
        ),
        InterShardFlow::Directory(DirectoryOp::LeaseGrant {
            key: DirectoryKey::Realm(RealmId::System(1)),
            owner: AuthorityRef::Shard(NodeId(2)),
            fence: Fence(5),
        }),
        InterShardFlow::Directory(DirectoryOp::CommitCas {
            key: DirectoryKey::Realm(RealmId::System(1)),
            expected: Fence(5),
            transfer: TransferId(6),
            new_owner: AuthorityRef::Shard(NodeId(3)),
        }),
        InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Realm(RealmId::System(1)),
        }),
        // P2 arms (the route-swap saga) — SIDE-EFFECTING (TransferStep) for the command
        // + its ack; FIRE-AND-FORGET for the directory reply envelope.
        InterShardFlow::Saga(TransferControl::PrepareSubscribe {
            transfer: TransferId(7),
            session: SessionId(1),
            dest: NodeId(2),
        }),
        InterShardFlow::SagaAck(TransferControlAck::Prepared {
            transfer: TransferId(8),
            result: PrepareResult::Ready,
        }),
        InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Realm(RealmId::System(1)),
            record: None,
        }),
        // 1d.1 arms: the pose-flush request + the entity-state ack family (SIDE-EFFECTING,
        // TransferStep-keyed by their step phase).
        InterShardFlow::FlushSource(FlushSource {
            transfer: TransferId(9),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            step_id: FLUSH_SOURCE_STEP,
        }),
        InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: TransferId(10),
            step_id: FLUSH_SOURCE_STEP,
            pose: pose(),
            drained_seq: 5,
        }),
        // ALL three TransferAck inner arms ride the per-release surface gate (so a new inner variant
        // is forced through the InterShardFlow postcard roundtrip + effect-class classification, not
        // just the outer arm). Rejected has no producer yet (reserved, D-21) but is sealed here.
        InterShardFlow::TransferAck(TransferAck::Accepted {
            transfer_id: TransferId(10),
            step_id: STUB_CROSSING_STEP,
        }),
        InterShardFlow::TransferAck(TransferAck::Rejected {
            transfer_id: TransferId(10),
            step_id: STUB_CROSSING_STEP,
            reason: TransferStepRejectReason::SpatialPrecondition,
        }),
        // D-7a arm: the dest's TRANSIENT batch-adopt ack (SIDE-EFFECTING, TransferStep-keyed by the
        // transient batch phase) — sealed here so a new inner variant rides the surface gate.
        InterShardFlow::TransferAck(TransferAck::BatchAdopted {
            transfer_id: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
        }),
        // D-7b arm: the transient handoff DropApplied ack (release-ack / promote-confirm) — sealed
        // so a new TransferAck variant rides the surface gate + the roundtrip/classification.
        InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_RELEASE_STEP,
        }),
        // 1d.5b arms: the saga-pushed ordered demote/promote commands (SIDE-EFFECTING,
        // TransferStep-keyed by DEMOTE_STEP/PROMOTE_STEP).
        InterShardFlow::Demote(vd_wire::intershard::DemoteCmd {
            transfer: TransferId(10),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_owner_fence: Fence(6),
            step_id: vd_wire::intershard::DEMOTE_STEP,
        }),
        InterShardFlow::Promote(vd_wire::intershard::PromoteCmd {
            transfer: TransferId(10),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_fence: Fence(6),
            step_id: vd_wire::intershard::PROMOTE_STEP,
            source: vd_core::NodeId(2),
        }),
        // D-7b arms: the three transient structural drop-before-promote handoff commands (all
        // SIDE-EFFECTING, TransferStep-keyed; the source-only release/complete + the dest-only
        // promote — idempotent by journaled step + local held-status). Share `TransientHandoff`.
        InterShardFlow::TransientRelease(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_RELEASE_STEP,
            fence: Fence(6),
        }),
        InterShardFlow::TransientDrop(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_DROP_STEP,
            fence: Fence(6),
        }),
        InterShardFlow::ReleaseComplete(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_RELEASE_STEP,
            fence: Fence(6),
        }),
        // D-7d arm: the dead-DEST ABANDON (SIDE-EFFECTING, TransferStep-keyed by TRANSIENT_ABANDON_STEP;
        // shares `TransientHandoff` — a proper new action, classified by the same one arm).
        InterShardFlow::TransientAbandon(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_ABANDON_STEP,
            fence: Fence(6),
        }),
        // D-37 arm: the forward re-home adopt (SIDE-EFFECTING, TransferStep-keyed by RE_HOME_STEP) — a
        // DEDICATED arm carrying the `ReHomeState` payload, never a `Promote` reuse (no-repurpose).
        InterShardFlow::ReHome(vd_wire::intershard::ReHomeCmd {
            transfer: TransferId(10),
            universe_epoch: EpochId(1),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_fence: Fence(6),
            step_id: vd_wire::intershard::RE_HOME_STEP,
            state: vd_wire::intershard::ReHomeState::PoseOnly(pose()),
            source: NodeId(2),
        }),
        // R-6d3c arm: the orch→dest DISCARD-poison (SIDE-EFFECTING, TransferStep-keyed by
        // TRANSIENT_DISCARD_STEP; shares `TransientHandoff` — classified ReDriven, NOT producer-less,
        // so the golden pin below still asserts exactly TWO producer-less arms).
        InterShardFlow::TransientDiscard(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_DISCARD_STEP,
            fence: Fence(6),
        }),
    ]
}

/// Compile-time tripwire: adding an `InterShardFlow` arm MUST break this exhaustive
/// match (no wildcard), forcing the author back here to also represent it in
/// [`every_arm`] above — so the per-release surface gate can never silently omit an arm
/// (audit HR1-1). This is never called; its body is the structural assertion.
#[allow(dead_code)]
fn arm_tripwire(flow: &InterShardFlow) {
    match flow {
        InterShardFlow::Ghost(_)
        | InterShardFlow::Transfer(_)
        | InterShardFlow::Directory(_)
        | InterShardFlow::Saga(_)
        | InterShardFlow::SagaAck(_)
        | InterShardFlow::DirectoryReply(_)
        | InterShardFlow::FlushSource(_)
        | InterShardFlow::TransferAck(_)
        | InterShardFlow::Demote(_)
        | InterShardFlow::Promote(_)
        | InterShardFlow::TransientRelease(_)
        | InterShardFlow::TransientDrop(_)
        | InterShardFlow::ReleaseComplete(_)
        | InterShardFlow::TransientAbandon(_)
        | InterShardFlow::ReHome(_)
        | InterShardFlow::TransientDiscard(_) => {}
    }
}

/// NESTED-VARIANT tripwires (R-6d2b review). `arm_tripwire` is exhaustive only at the OUTER `InterShardFlow`
/// level (`Transfer(_)`/`Ghost(_)`), so a new NESTED `TransitionPayload`/`GhostFlow` variant — the natural
/// shape of a FUTURE producer-less durable flow (P9 cross-shard Signal, P6 durable BlockEdit forward) —
/// would NOT break compilation, would be omitted from `every_arm`, and would slip the `durability_class`
/// golden pin (which iterates `every_arm`) VACUOUSLY: it would ship pushed with the `Durability::Ephemeral`
/// default and be silently lost on a source crash (D-6 #1) with a GREEN suite. These wildcard-free matches
/// force a new nested variant to FAIL COMPILATION here until it is classified — an eyes-open edit that (with
/// the golden pin + the stub marker test it points at) forces representing it in `every_arm` and verifying
/// its push-site `Retained` marker. This is the exhaustive-by-construction guard the per-send-`Durability`
/// default relies on instead of the vetted explicit-4-arg. Never called; the body is the assertion.
#[allow(dead_code)]
fn payload_tripwire(p: &TransitionPayload) {
    match p {
        TransitionPayload::InitialSpawn { .. }
        | TransitionPayload::StubCrossing { .. }
        | TransitionPayload::TransientBatch { .. } => {}
    }
}

#[allow(dead_code)]
fn ghost_tripwire(g: &GhostFlow) {
    match g {
        GhostFlow::Spawn { .. } | GhostFlow::Delta { .. } | GhostFlow::Despawn { .. } => {}
    }
}

#[test]
fn every_arm_roundtrips_postcard_and_classifies_coherently() {
    for flow in every_arm() {
        // 1. Closed wire: every arm serializes and decodes byte-identically.
        let bytes = postcard::to_allocvec(&flow).expect("closed arm encodes");
        let back: InterShardFlow = postcard::from_bytes(&bytes).expect("closed arm decodes");
        assert_eq!(back, flow);

        // 2. G-SEALED: side-effecting arms MUST carry an idempotency key; nothing
        //    that mutates authority may ride a fire-and-forget channel.
        match flow.effect_class() {
            EffectClass::SideEffecting { idempotency } => match idempotency {
                IdempotencyKey::TransferStep { transfer, .. }
                | IdempotencyKey::FencedCas { transfer, .. } => {
                    assert_ne!(transfer, TransferId(0), "a real correlation id");
                }
                IdempotencyKey::FencedKey { fence } => {
                    assert_ne!(fence, Fence::GENESIS, "a real fence");
                }
            },
            EffectClass::FireAndForget => {
                // Fire-and-forget arms are re-derivable / latest-wins by construction
                // (ghost deltas, head reads). The exhaustive match in effect_class is
                // what makes this assertion total.
            }
        }
    }
}

#[test]
fn durability_class_pins_the_producer_less_reliable_set() {
    // R-6d §7 conformance: the wildcard-free `durability_class` match makes the classification TOTAL (a new
    // arm / GhostFlow / TransitionPayload variant fails to compile until classified). This golden pin asserts
    // the PRODUCER-LESS-RELIABLE set — the arms whose `push_flow` site MUST carry `Durability::Retained`, or
    // the one-shot is silently lost on a source crash — is EXACTLY {Ghost::Despawn, Transfer(TransientBatch)}.
    // Growing it is a deliberate edit that trips BOTH this pin AND (at R-6d2b) the marker-on-push test — so a
    // future durable Signal / block-edit forward cannot slip in producer-less without a marker.
    let mut producer_less = Vec::new();
    for flow in every_arm() {
        let expect = match &flow {
            InterShardFlow::Ghost(GhostFlow::Despawn { .. }) => {
                FlowDurabilityClass::ProducerLessReliable
            }
            InterShardFlow::Ghost(GhostFlow::Delta { .. }) => FlowDurabilityClass::Unreliable,
            InterShardFlow::Transfer(env)
                if matches!(env.payload, TransitionPayload::TransientBatch { .. }) =>
            {
                FlowDurabilityClass::ProducerLessReliable
            }
            _ => FlowDurabilityClass::ReDriven,
        };
        assert_eq!(
            flow.durability_class(),
            expect,
            "arm misclassified: {flow:?}"
        );
        if flow.durability_class() == FlowDurabilityClass::ProducerLessReliable {
            producer_less.push(flow);
        }
    }
    assert_eq!(
        producer_less.len(),
        2,
        "exactly two producer-less-reliable arms today (Ghost::Despawn + TransientBatch): {producer_less:?}"
    );
}

#[test]
fn side_effecting_transfer_carries_its_step_idempotency() {
    let durable = InterShardFlow::Transfer(TransferEnvelope {
        transfer_id: TransferId(99),
        universe_epoch: EpochId(1),
        schema_version: 1,
        fence: Fence(2),
        step_id: 7,
        class: DurabilityClass::Durable,
        payload: TransitionPayload::InitialSpawn {
            entity: eid(EntityKind::Player),
            to_realm: RealmId::System(1),
            pose: pose(),
            state: vec![],
        },
    });
    assert_eq!(
        durable.effect_class(),
        EffectClass::SideEffecting {
            idempotency: IdempotencyKey::TransferStep {
                transfer: TransferId(99),
                step_id: 7,
            },
        },
    );
}
