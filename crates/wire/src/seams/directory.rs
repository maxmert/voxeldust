//! The Ownership Directory seam: THE single authoritative answer to "who owns key X"
//! (`docs/design/transfer_protocol.md` §4, unified key space per the integration
//! resolutions). The orchestrator implements it; gateways and shards consume it.
//!
//! Consistency model: writes are linearizable (single writer); replicated caches are
//! HINTS — authority decisions always pull through, and the fence on every snapshot
//! frame is what makes a stale owner harmless (fence rule 5). `DirectoryOp` carries
//! ONLY orchestrator-authoritative control — spatial interest is shard-local and
//! never rides this seam (sealed_shards §6).

use serde::{Deserialize, Serialize};
use vd_core::pose::RealmId;
use vd_core::{EntityId, EpochId, Fence, NodeId, SessionId, TransferId, UniverseTick};

/// The unified directory key space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DirectoryKey {
    /// Session ownership: which gateway speaks for a connected client.
    Session(SessionId),
    /// Entity authority (Durable kinds only — transients are held-set anchored).
    Entity(EntityId),
    /// Realm ownership: the single writer of a realm's durable state.
    Realm(RealmId),
    /// Ship exterior-authority (the hull's host shard).
    Ship(EntityId),
}

/// Who holds authority for a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorityRef {
    Shard(NodeId),
    Gateway(NodeId),
}

/// One directory record. The fence IS the linearizability primitive; `in_transfer`
/// is ENFORCED (blocks concurrent sagas and ghost-despawn mid-transfer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnerRecord {
    pub authority: AuthorityRef,
    pub fence: Fence,
    /// Soft lease deadline against the analytic clock; renewals never touch durable
    /// storage — only ASSIGNMENTS are persisted.
    pub lease_expires: UniverseTick,
    pub in_transfer: Option<TransferId>,
}

/// Orchestrator-authoritative control operations.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DirectoryOp {
    /// Assign authority at a fresh fence (provision/spawn/recovery).
    LeaseGrant {
        key: DirectoryKey,
        owner: AuthorityRef,
        fence: Fence,
    },
    /// Soft liveness renewal (re-derivable; loss tolerated — the lease just lapses).
    LeaseRenew { key: DirectoryKey, fence: Fence },
    /// Revoke before reassignment (the holder must self-fence on receipt).
    LeaseRevoke { key: DirectoryKey, fence: Fence },
    /// THE commit point: CAS the fence as part of saga `transfer`; the winner
    /// flips authority to `new_owner` (a commit that cannot name the destination
    /// is unusable — the dest shard wins authority AT the CAS, nowhere else).
    CommitCas {
        key: DirectoryKey,
        expected: Fence,
        transfer: TransferId,
        new_owner: AuthorityRef,
    },
    /// The abort CAS — mutually exclusive with commit by construction (fence rule 3).
    AbortCas {
        key: DirectoryKey,
        expected: Fence,
        transfer: TransferId,
    },
    /// Hint head-read (carries no fence expectation; a mid-flip read never rejects).
    HeadRead { key: DirectoryKey },
    /// Analytic-clock sync (write-ahead ceiling discipline lives orchestrator-side).
    ClockSync {
        universe_tick: UniverseTick,
        epoch: EpochId,
    },
}

impl DirectoryOp {
    /// Effect classification for the G-SEALED conformance gate. CAS and grant/revoke
    /// mutate authority and are idempotent BY FENCE comparison; renewals, reads, and
    /// clock samples are re-derivable.
    #[must_use]
    pub fn effect_class(&self) -> super::super::intershard::EffectClass {
        use super::super::intershard::{EffectClass, IdempotencyKey};
        match self {
            DirectoryOp::LeaseGrant { fence, .. } | DirectoryOp::LeaseRevoke { fence, .. } => {
                EffectClass::SideEffecting {
                    idempotency: IdempotencyKey::FencedKey { fence: *fence },
                }
            }
            DirectoryOp::CommitCas {
                expected, transfer, ..
            }
            | DirectoryOp::AbortCas {
                expected, transfer, ..
            } => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::FencedCas {
                    expected: *expected,
                    transfer: *transfer,
                },
            },
            DirectoryOp::LeaseRenew { .. }
            | DirectoryOp::HeadRead { .. }
            | DirectoryOp::ClockSync { .. } => EffectClass::FireAndForget,
        }
    }
}

/// Replies to directory operations.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DirectoryReply {
    Head {
        key: DirectoryKey,
        record: Option<OwnerRecord>,
    },
    /// CAS outcome: the winner gets the new fence; the loser gets the current one
    /// and MUST no-op.
    CasResult {
        key: DirectoryKey,
        outcome: CasOutcome,
    },
    ClockNow {
        universe_tick: UniverseTick,
        epoch: EpochId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CasOutcome {
    Won { new_fence: Fence },
    Lost { current: Fence },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intershard::{EffectClass, IdempotencyKey};
    use vd_core::entity_kind::EntityKind;

    fn keys() -> Vec<DirectoryKey> {
        vec![
            DirectoryKey::Session(SessionId(1)),
            DirectoryKey::Entity(EntityId::pack(EntityKind::Player, 1, 2, 3)),
            DirectoryKey::Realm(RealmId::Planet(4)),
            DirectoryKey::Ship(EntityId::pack(EntityKind::Ship, 1, 5, 6)),
        ]
    }

    #[test]
    fn all_key_kinds_roundtrip_and_order() {
        let ks = keys();
        for k in &ks {
            let bytes = postcard::to_allocvec(k).expect("encode");
            assert_eq!(
                postcard::from_bytes::<DirectoryKey>(&bytes).expect("decode"),
                *k
            );
        }
        let mut sorted = ks.clone();
        sorted.sort();
        assert_eq!(sorted.len(), 4);
    }

    #[test]
    fn ops_classify_into_the_right_effect_classes() {
        let key = DirectoryKey::Realm(RealmId::System(9));
        let fenced = EffectClass::SideEffecting {
            idempotency: IdempotencyKey::FencedKey { fence: Fence(1) },
        };
        let cased = EffectClass::SideEffecting {
            idempotency: IdempotencyKey::FencedCas {
                expected: Fence(1),
                transfer: TransferId(2),
            },
        };
        let cases: Vec<(DirectoryOp, EffectClass)> = vec![
            (
                DirectoryOp::LeaseGrant {
                    key,
                    owner: AuthorityRef::Shard(NodeId(1)),
                    fence: Fence(1),
                },
                fenced,
            ),
            (
                DirectoryOp::LeaseRevoke {
                    key,
                    fence: Fence(1),
                },
                fenced,
            ),
            (
                DirectoryOp::CommitCas {
                    key,
                    expected: Fence(1),
                    transfer: TransferId(2),
                    new_owner: AuthorityRef::Shard(NodeId(8)),
                },
                cased,
            ),
            (
                DirectoryOp::AbortCas {
                    key,
                    expected: Fence(1),
                    transfer: TransferId(2),
                },
                cased,
            ),
            (
                DirectoryOp::LeaseRenew {
                    key,
                    fence: Fence(1),
                },
                EffectClass::FireAndForget,
            ),
            (DirectoryOp::HeadRead { key }, EffectClass::FireAndForget),
            (
                DirectoryOp::ClockSync {
                    universe_tick: UniverseTick(5),
                    epoch: EpochId(1),
                },
                EffectClass::FireAndForget,
            ),
        ];
        for (op, expected) in cases {
            assert_eq!(op.effect_class(), expected, "{op:?}");
        }
    }

    #[test]
    fn cas_idempotency_carries_expected_fence_and_transfer() {
        let op = DirectoryOp::CommitCas {
            new_owner: AuthorityRef::Shard(NodeId(8)),
            key: DirectoryKey::Session(SessionId(3)),
            expected: Fence(7),
            transfer: TransferId(8),
        };
        assert_eq!(
            op.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::FencedCas {
                    expected: Fence(7),
                    transfer: TransferId(8),
                },
            }
        );
    }

    #[test]
    fn records_and_replies_roundtrip() {
        let record = OwnerRecord {
            authority: AuthorityRef::Gateway(NodeId(4)),
            fence: Fence(2),
            lease_expires: UniverseTick(100),
            in_transfer: Some(TransferId(6)),
        };
        let replies = [
            DirectoryReply::Head {
                key: DirectoryKey::Session(SessionId(1)),
                record: Some(record),
            },
            DirectoryReply::Head {
                key: DirectoryKey::Session(SessionId(1)),
                record: None,
            },
            DirectoryReply::CasResult {
                key: DirectoryKey::Session(SessionId(1)),
                outcome: CasOutcome::Won {
                    new_fence: Fence(3),
                },
            },
            DirectoryReply::CasResult {
                key: DirectoryKey::Session(SessionId(1)),
                outcome: CasOutcome::Lost { current: Fence(9) },
            },
            DirectoryReply::ClockNow {
                universe_tick: UniverseTick(50),
                epoch: EpochId(1),
            },
        ];
        for reply in replies {
            let bytes = postcard::to_allocvec(&reply).expect("encode");
            assert_eq!(
                postcard::from_bytes::<DirectoryReply>(&bytes).expect("decode"),
                reply
            );
        }
    }
}
