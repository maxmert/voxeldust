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

impl DirectoryKey {
    /// The `EntityId` a transfer-DEST shard ADOPTS for this subject — the ONE home for the
    /// `DirectoryKey → adopt-target` policy (was duplicated byte-for-byte in `vd-sim` and `vd-node`,
    /// a lockstep-edit hazard; audit DRY-finding). `Some` ONLY for the per-entity `Entity` key:
    /// `Session`/`Realm` carry no adoptable entity, and `Ship` exterior-authority is a DISTINCT
    /// adopt path (the hull host, P8) deliberately `None` here so the per-entity crossing can never
    /// mis-adopt a ship hull — when P8 lands it consumes `Ship` explicitly; this stays the
    /// per-entity extraction.
    #[must_use]
    pub fn transfer_subject_entity(self) -> Option<EntityId> {
        match self {
            DirectoryKey::Entity(entity) => Some(entity),
            DirectoryKey::Session(_) | DirectoryKey::Realm(_) | DirectoryKey::Ship(_) => None,
        }
    }
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
    ///
    /// ## TTL ENFORCEMENT IS NOT YET IMPLEMENTED (binding pre-crash-matrix item)
    /// The deadline is WRITTEN on every grant/renew/commit but nothing DRIVES the
    /// lifecycle yet: no node sends `LeaseRenew` (the heartbeat producer is unbuilt)
    /// and no sweep reads `lease_expires` to reap a lapsed record — a crashed node's
    /// keys currently stay owned forever. Inert for P1's fixed roster (keys are
    /// explicitly revoked on detach); MUST land before the P2 crash/stagger matrix +
    /// P3 chaos: a renewal heartbeat from each authority holder AND an orchestrator
    /// expiry sweep gated on unreachable-confirmation (lapsed lease ⇒ ownership loss
    /// ONLY when the owner is confirmed unreachable). Do NOT assume crash recovery
    /// exists yet. (Audit XSI-1.)
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
    /// The abort CAS — mutually exclusive with commit by construction (fence rule 3). ⚠️ This is the
    /// FENCE-MOVING abort arm (bumps via `cas_next`). Do NOT route a TERMINAL abort through it: a terminal
    /// abort of a surviving source must be FENCE-NEUTRAL (the in-process path uses `DirectoryCore::abort_clear`)
    /// or it strands the source a fence behind + wedges a post-abort logout `LeaseRevoke` (FENCE-9; D-6 abort-path
    /// note). A future N-orchestrator router (D-32) carrying terminal aborts over this seam needs a fence-neutral
    /// `AbortClear` op, not this one.
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
    fn transfer_subject_entity_is_some_only_for_the_entity_key() {
        // The ONE adopt-target policy: Entity → Some; Session/Realm/Ship → None (Ship is the
        // distinct hull-host adopt path, P8). Covers both arms.
        let entity = EntityId::pack(EntityKind::Player, 1, 2, 3);
        assert_eq!(
            DirectoryKey::Entity(entity).transfer_subject_entity(),
            Some(entity)
        );
        assert_eq!(
            DirectoryKey::Session(SessionId(1)).transfer_subject_entity(),
            None
        );
        assert_eq!(
            DirectoryKey::Realm(RealmId::Planet(4)).transfer_subject_entity(),
            None
        );
        assert_eq!(
            DirectoryKey::Ship(EntityId::pack(EntityKind::Ship, 1, 5, 6)).transfer_subject_entity(),
            None
        );
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
