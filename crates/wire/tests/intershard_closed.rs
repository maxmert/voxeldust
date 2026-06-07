//! G-SEALED conformance gate (`docs/design/sealed_shards.md` §1) — lifted from the
//! in-module unit test to the design-named INTEGRATION location so the closed-set
//! guarantee is a per-release gate, not a convention (audit SEAL-2).
//!
//! The closed taxonomy `InterShardFlow{Ghost,Transfer,Directory}` is the ONLY shape
//! that crosses a shard boundary; every arm has a coherent `EffectClass`, and every
//! SIDE-EFFECTING arm carries an idempotency key (so an authority-gating payload can
//! never ride a fire-and-forget channel). The exhaustive `match` in `effect_class`
//! means adding an arm without classifying it does not compile.

use vd_core::entity_kind::{DurabilityClass, EntityKind};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::{EntityId, EpochId, Fence, NodeId, TickId, TransferId, UniverseTick};
use vd_wire::intershard::{
    EffectClass, GhostFlow, IdempotencyKey, InterShardFlow, TransferEnvelope, TransitionPayload,
};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp};

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
    ]
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
