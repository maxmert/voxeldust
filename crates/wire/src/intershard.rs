//! THE closed shard↔shard / shard↔orchestrator taxonomy (HR1 taxonomy 1 of 2;
//! `docs/design/sealed_shards.md` §0–§1). One reviewed file.
//!
//! Every byte crossing a shard-to-shard or shard-to-orchestrator boundary is exactly
//! one [`InterShardFlow`] arm. There is NO `Raw`, NO `Other`, NO `EcsSync` — to
//! express a new cross-shard flow you MUST add a variant here, under review.
//!
//! Arms freeze INCREMENTALLY with their first consumer (the closed-set guarantee is
//! the per-release conformance test below, not a day-one empty freeze):
//! - P0 (now): `Ghost`, `Transfer` (Durable class), `Directory`.
//! - P2: `Saga` (the saga→gateway transfer commands) + `SagaAck` (gateway→saga acks) —
//!   the route-swap saga drives the gateway exclusively through these.
//! - P3/P6: `Transfer` transient batches + `BlockEdit`.
//! - P8: `Coupling` (`EffectFree` ports). — variant reserved, payload lands with ships.
//! - P9: `Signal`.
//!
//! Effect classes (the G-SEALED invariant, enforced by `effect_class` + its test):
//! - SIDE-EFFECTING arms carry `(TransferId, step_id)` idempotency and are ack-driven.
//! - FIRE-AND-FORGET arms are effect-free or idempotently re-derivable: they may
//!   NEVER carry a transfer trigger or authority-gating discrete state.

use serde::{Deserialize, Serialize};
use vd_core::entity_kind::DurabilityClass;
use vd_core::pose::{RealmId, StampedPose};
use vd_core::{EntityId, EpochId, Fence, TickId, TransferId};

use crate::seams::directory::{DirectoryOp, DirectoryReply};
use crate::seams::transfer_control::{TransferControl, TransferControlAck};

/// The closed taxonomy. Compiler-forced exhaustive handling everywhere.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InterShardFlow {
    /// Replication into a neighbor's overlap band (spawn/delta/despawn).
    Ghost(GhostFlow),
    /// Authority handoff — both durability classes ride the same arm (HR3).
    Transfer(TransferEnvelope),
    /// Orchestrator-authoritative control ONLY (lease/CAS/clock — never spatial
    /// interest, which is shard-local by design).
    Directory(DirectoryOp),
    /// Saga → gateway transfer commands (P2): the route-swap saga drives the gateway
    /// through the phased, separately-acked command vocabulary (the gateway never
    /// decides a transfer). Side-effecting + ack-driven by `(transfer, phase)`.
    Saga(TransferControl),
    /// Gateway → saga acks for the `Saga` arm (one per command phase).
    SagaAck(TransferControlAck),
    /// Orchestrator → requester REPLY to a `Directory` op (P2): the standing head or the
    /// CAS outcome. Carried as its OWN arm (not bare bytes) so the requester can decode
    /// `InterShardFlow` ONCE and dispatch by variant — a `Saga(TransferControl)` and a
    /// directory reply both ride orchestrator→peer on `MsgClass::Saga`, and postcard is
    /// non-self-describing, so without this they would mis-decode into each other.
    DirectoryReply(DirectoryReply),
}

/// How an arm participates in side effects: the machine-checkable half of HR1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectClass {
    /// Mutates durable/authoritative state: must carry an idempotency key and be
    /// delivered via acked retry.
    SideEffecting { idempotency: IdempotencyKey },
    /// Latest-wins / re-derivable; loss is tolerated by design. May NEVER carry a
    /// transfer trigger or authority-gating discrete state.
    FireAndForget,
}

/// The idempotency mechanisms a side-effecting arm may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdempotencyKey {
    /// `(TransferId, step_id)` — journaled in `applied_steps` before any effect.
    TransferStep { transfer: TransferId, step_id: u32 },
    /// Idempotent by fence comparison (lease grants/revokes).
    FencedKey { fence: Fence },
    /// A fence CAS within a saga (commit/abort — mutually exclusive by rule 3).
    FencedCas {
        expected: Fence,
        transfer: TransferId,
    },
}

impl InterShardFlow {
    /// Classify an arm. EXHAUSTIVE by construction — adding a variant without
    /// classifying it does not compile, which is the G-SEALED conformance hook.
    #[must_use]
    pub fn effect_class(&self) -> EffectClass {
        match self {
            // Ghost spawn/despawn are reliable+acked but idempotently re-derivable
            // from band geometry; deltas are pure latest-wins mirrors.
            InterShardFlow::Ghost(_) => EffectClass::FireAndForget,
            InterShardFlow::Transfer(env) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: env.transfer_id,
                    step_id: env.step_id,
                },
            },
            InterShardFlow::Directory(op) => op.effect_class(),
            // Saga commands + acks are side-effecting + ack-driven; `(transfer, phase)`
            // is their `applied_steps` idempotency key (the gateway/saga no-op a
            // re-delivery at the same phase).
            InterShardFlow::Saga(cmd) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: cmd.transfer(),
                    step_id: cmd.step_id(),
                },
            },
            InterShardFlow::SagaAck(ack) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: ack.transfer(),
                    step_id: ack.step_id(),
                },
            },
            // A reply mutates NOTHING at the receiver: it reports the standing head / CAS
            // outcome, which the receiver treats as a HINT and pulls through to the
            // authority-of-record directory (a lost reply is recovered by re-reading the
            // head — the directory REQUEST it answers carries the real idempotency). So it
            // is re-derivable + loss-tolerated = FireAndForget; it carries no transfer
            // trigger and is never the authority of record (the CAS at the directory is).
            InterShardFlow::DirectoryReply(_) => EffectClass::FireAndForget,
        }
    }
}

/// Ghost replication: kinematic mirrors that NEVER independently integrate physics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GhostFlow {
    /// Reliable, acked: the neighbor inserts a kinematic ghost.
    Spawn {
        entity: EntityId,
        pose: StampedPose,
        source_fence: Fence,
        since_tick: TickId,
    },
    /// Datagram, 20 Hz, latest-wins. Deltas older than the ghost's `since_tick`
    /// or carrying a stale fence are dropped by data, not by stream ordering.
    Delta {
        entity: EntityId,
        pose: StampedPose,
        source_fence: Fence,
        source_tick: TickId,
        seq: u64,
    },
    /// Reliable, acked; REFUSED by the receiver while the entity is `in_transfer`
    /// (the directory field is enforced, not decorative).
    Despawn {
        entity: EntityId,
        source_fence: Fence,
    },
}

/// The transfer envelope: ONE shape for every entity kind and both durability
/// classes — the registry (`KindDef`) is the only per-kind variation (HR2/HR3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransferEnvelope {
    pub transfer_id: TransferId,
    pub universe_epoch: EpochId,
    /// Control-plane schema version (postcard, additive under minor negotiation).
    pub schema_version: u16,
    /// The authority fence this message was issued under; stale-fence messages are
    /// rejected by every receiver (fence rule 1).
    pub fence: Fence,
    /// `(transfer_id, step_id)` is the universal side-effect idempotency key.
    pub step_id: u32,
    pub class: DurabilityClass,
    pub payload: TransitionPayload,
}

/// Typed per-transition payloads — the compiler forbids a boarding message carrying
/// warp fields (the R4 god-struct is unrepresentable). Variants are added with their
/// phase; P0 carries what stub shards (P1–P3) need.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TransitionPayload {
    /// First spawn of an entity into a realm (login/bootstrap).
    InitialSpawn {
        entity: EntityId,
        to_realm: RealmId,
        pose: StampedPose,
        /// TLV state blob (`vd_core::tlv`), schema per `KindDef::blob_schema`.
        state: Vec<u8>,
    },
    /// A stub-shard boundary crossing (P2/P3 transfer machinery proving ground;
    /// the real transition classes — boarding/EVA/SOI/warp — land as additive
    /// variants with their phases).
    StubCrossing {
        entity: EntityId,
        from_realm: RealmId,
        to_realm: RealmId,
        pose: StampedPose,
        state: Vec<u8>,
    },
    /// Batched transient handover (debris/projectiles): items share ONE envelope and
    /// ONE batched `TransientGo` go-token — never per-item directory writes.
    TransientBatch {
        from_realm: RealmId,
        to_realm: RealmId,
        src_realm_fence: Fence,
        dst_realm_fence: Fence,
        source_tick: TickId,
        items: Vec<TransientItem>,
    },
}

/// One transient entity inside a batch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransientItem {
    pub entity: EntityId,
    pub pose: StampedPose,
    /// TLV blob, capped by the kind's `max_state_bytes`.
    pub state: Vec<u8>,
}

/// Acks for the side-effecting arms (delivered via the same flow channel).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TransferAck {
    /// Destination accepted and durably journaled the step.
    Accepted {
        transfer_id: TransferId,
        step_id: u32,
    },
    /// Destination refused; the entity stays authoritative on the source.
    Rejected {
        transfer_id: TransferId,
        step_id: u32,
        reason: TransferRejectReason,
    },
}

/// Typed rejection causes (never a stringly-typed warn-and-drop).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferRejectReason {
    SpatialPrecondition,
    StaleFence,
    EpochMismatch,
    VersionFloor,
    UnknownKind,
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::UniverseTick;
    use vd_core::entity_kind::EntityKind;
    use vd_core::glam::DVec3;
    use vd_core::pose::FrameRef;

    fn pose() -> StampedPose {
        StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 1 },
            DVec3::ZERO,
            UniverseTick(10),
        )
    }

    fn eid(kind: EntityKind) -> EntityId {
        EntityId::pack(kind, 1, 7, 3)
    }

    fn envelope(payload: TransitionPayload, class: DurabilityClass) -> TransferEnvelope {
        TransferEnvelope {
            transfer_id: TransferId(11),
            universe_epoch: EpochId(1),
            schema_version: 1,
            fence: Fence(2),
            step_id: 4,
            class,
            payload,
        }
    }

    /// G-SEALED: every arm has a coherent effect class, and side-effecting arms
    /// expose their idempotency key. The match in `effect_class` is exhaustive, so
    /// adding an arm without classifying it cannot compile.
    #[test]
    fn g_sealed_effect_classes() {
        let ghost = InterShardFlow::Ghost(GhostFlow::Delta {
            entity: eid(EntityKind::Player),
            pose: pose(),
            source_fence: Fence(1),
            source_tick: TickId(5),
            seq: 9,
        });
        assert_eq!(ghost.effect_class(), EffectClass::FireAndForget);

        let durable = InterShardFlow::Transfer(envelope(
            TransitionPayload::InitialSpawn {
                entity: eid(EntityKind::Player),
                to_realm: RealmId::System(1),
                pose: pose(),
                state: vec![],
            },
            DurabilityClass::Durable,
        ));
        assert_eq!(
            durable.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: 4
                }
            }
        );

        // Transient batches share the SAME side-effecting machinery (HR3): the
        // batched go-token is their commit point; they are NOT fire-and-forget.
        let transient = InterShardFlow::Transfer(envelope(
            TransitionPayload::TransientBatch {
                from_realm: RealmId::System(1),
                to_realm: RealmId::Planet(2),
                src_realm_fence: Fence(3),
                dst_realm_fence: Fence(4),
                source_tick: TickId(6),
                items: vec![TransientItem {
                    entity: eid(EntityKind::Debris),
                    pose: pose(),
                    state: vec![1],
                }],
            },
            DurabilityClass::Transient,
        ));
        assert_eq!(
            transient.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: 4
                }
            }
        );

        // Directory ops classify through the flow wrapper identically to direct calls.
        let directory = InterShardFlow::Directory(DirectoryOp::CommitCas {
            key: crate::seams::directory::DirectoryKey::Realm(RealmId::System(1)),
            expected: Fence(5),
            transfer: TransferId(6),
            new_owner: crate::seams::directory::AuthorityRef::Shard(vd_core::NodeId(2)),
        });
        assert_eq!(
            directory.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::FencedCas {
                    expected: Fence(5),
                    transfer: TransferId(6),
                }
            }
        );

        // Saga commands + acks ride the closed taxonomy as side-effecting arms keyed by
        // (transfer, phase) — the route swap is never untyped bytes (HR1). FreezeSource and
        // its ack SourceFrozen are both phase 2.
        let saga = InterShardFlow::Saga(TransferControl::FreezeSource {
            transfer: TransferId(11),
            session: vd_core::SessionId(3),
            marker_seq: 17,
        });
        assert_eq!(
            saga.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: 2,
                }
            }
        );
        let saga_ack = InterShardFlow::SagaAck(TransferControlAck::SourceFrozen {
            transfer: TransferId(11),
            drained_seq: 17,
        });
        assert_eq!(
            saga_ack.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: 2,
                }
            }
        );

        // A directory REPLY is a re-derivable report (the receiver pulls through to the
        // authority-of-record directory), never a transfer trigger → FireAndForget.
        let reply = InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: crate::seams::directory::DirectoryKey::Realm(RealmId::System(1)),
            record: None,
        });
        assert_eq!(reply.effect_class(), EffectClass::FireAndForget);
    }

    #[test]
    fn saga_arms_roundtrip() {
        for flow in [
            InterShardFlow::Saga(TransferControl::CommitAuthority {
                transfer: TransferId(11),
                session: vd_core::SessionId(3),
                new_fence: Fence(4),
            }),
            InterShardFlow::SagaAck(TransferControlAck::Committed {
                transfer: TransferId(11),
            }),
            // The reply arm must roundtrip distinctly from the Saga arms (the dispatch
            // split depends on the InterShardFlow tag discriminating them).
            InterShardFlow::DirectoryReply(DirectoryReply::CasResult {
                key: crate::seams::directory::DirectoryKey::Session(vd_core::SessionId(3)),
                outcome: crate::seams::directory::CasOutcome::Won {
                    new_fence: Fence(4),
                },
            }),
        ] {
            let bytes = postcard::to_allocvec(&flow).expect("encode");
            assert_eq!(
                postcard::from_bytes::<InterShardFlow>(&bytes).expect("decode"),
                flow
            );
        }
    }

    #[test]
    fn ghost_flow_roundtrips() {
        let flows = vec![
            GhostFlow::Spawn {
                entity: eid(EntityKind::Ship),
                pose: pose(),
                source_fence: Fence(1),
                since_tick: TickId(2),
            },
            GhostFlow::Delta {
                entity: eid(EntityKind::Ship),
                pose: pose(),
                source_fence: Fence(1),
                source_tick: TickId(3),
                seq: 1,
            },
            GhostFlow::Despawn {
                entity: eid(EntityKind::Ship),
                source_fence: Fence(1),
            },
        ];
        for flow in flows {
            let bytes = postcard::to_allocvec(&flow).expect("encode");
            assert_eq!(
                postcard::from_bytes::<GhostFlow>(&bytes).expect("decode"),
                flow
            );
        }
    }

    #[test]
    fn envelopes_and_acks_roundtrip() {
        let env = envelope(
            TransitionPayload::StubCrossing {
                entity: eid(EntityKind::Player),
                from_realm: RealmId::System(1),
                to_realm: RealmId::Planet(2),
                pose: pose(),
                state: vec![9, 9],
            },
            DurabilityClass::Durable,
        );
        let flow = InterShardFlow::Transfer(env);
        let bytes = postcard::to_allocvec(&flow).expect("encode");
        assert_eq!(
            postcard::from_bytes::<InterShardFlow>(&bytes).expect("decode"),
            flow
        );

        for ack in [
            TransferAck::Accepted {
                transfer_id: TransferId(1),
                step_id: 2,
            },
            TransferAck::Rejected {
                transfer_id: TransferId(1),
                step_id: 2,
                reason: TransferRejectReason::SpatialPrecondition,
            },
        ] {
            let bytes = postcard::to_allocvec(&ack).expect("encode");
            assert_eq!(
                postcard::from_bytes::<TransferAck>(&bytes).expect("decode"),
                ack
            );
        }
    }

    #[test]
    fn reject_reasons_are_typed_and_distinct() {
        let reasons = [
            TransferRejectReason::SpatialPrecondition,
            TransferRejectReason::StaleFence,
            TransferRejectReason::EpochMismatch,
            TransferRejectReason::VersionFloor,
            TransferRejectReason::UnknownKind,
        ];
        for (i, a) in reasons.iter().enumerate() {
            for (j, b) in reasons.iter().enumerate() {
                assert_eq!(a == b, i == j);
            }
        }
    }
}
