//! The in-memory Authority Directory core: ONE fenced, single-writer answer to
//! "who owns key X" (`docs/design/transfer_protocol.md` §4; the P3 redb backing
//! swaps in behind this same core via the Store seam — records never reshape).
//!
//! Binding semantics (fence rules 1–5):
//! - The fence CAS is the ONLY commit point; COMMIT and ABORT race on the same
//!   expected fence — the winner bumps it, the loser observes `Lost` and no-ops.
//! - Grants are idempotent BY FENCE: re-granting the same (key, owner, fence) is a
//!   success no-op; a different owner at a stale-or-equal fence is rejected with the
//!   current record (the caller compares — hints never decide authority).
//! - `in_transfer` is ENFORCED: a key locked by a saga refuses concurrent grants
//!   and revokes until commit/abort clears it.

use std::collections::BTreeMap;

use vd_core::{Fence, TransferId, UniverseTick};
use vd_wire::seams::directory::{AuthorityRef, CasOutcome, DirectoryKey, OwnerRecord};

/// Directory-side operational parameters (ONE reviewed struct — never inline
/// literals at use sites).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectoryTuning {
    /// Soft lease length granted on every grant/renewal, in universe ticks.
    pub lease_ttl_ticks: u64,
}

/// Outcome of a grant attempt (the wire reply is the resulting head record; this
/// typed outcome is for the orchestrator's own logic and tests).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GrantOutcome {
    /// The record now shows (owner, fence) as requested.
    Granted,
    /// Refused: the key is held at an equal-or-higher fence by someone else, or is
    /// locked by an in-flight transfer. The current record is returned unchanged.
    Refused { current: OwnerRecord },
}

/// Outcome of a revoke attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RevokeOutcome {
    Revoked,
    /// Stale fence or transfer-locked: the record stands.
    Refused {
        current: OwnerRecord,
    },
    UnknownKey,
}

/// The pure directory state machine. Single writer by construction (owned by the
/// orchestrator node; everyone else talks through the seam).
#[derive(Debug)]
pub struct DirectoryCore {
    records: BTreeMap<DirectoryKey, OwnerRecord>,
    tuning: DirectoryTuning,
}

impl DirectoryCore {
    #[must_use]
    pub fn new(tuning: DirectoryTuning) -> DirectoryCore {
        DirectoryCore {
            records: BTreeMap::new(),
            tuning,
        }
    }

    /// Assign authority at `fence`. Idempotent by fence; refuses stale fences,
    /// equal-fence owner changes, and transfer-locked keys.
    pub fn grant(
        &mut self,
        key: DirectoryKey,
        owner: AuthorityRef,
        fence: Fence,
        now: UniverseTick,
    ) -> GrantOutcome {
        let lease_expires = UniverseTick(now.0.saturating_add(self.tuning.lease_ttl_ticks));
        match self.records.get_mut(&key) {
            None => {
                self.records.insert(
                    key,
                    OwnerRecord {
                        authority: owner,
                        fence,
                        lease_expires,
                        in_transfer: None,
                    },
                );
                GrantOutcome::Granted
            }
            Some(record) => {
                if record.in_transfer.is_some() {
                    return GrantOutcome::Refused { current: *record };
                }
                if fence == record.fence && owner == record.authority {
                    // Idempotent re-grant: refresh the lease, change nothing else.
                    record.lease_expires = lease_expires;
                    return GrantOutcome::Granted;
                }
                if fence <= record.fence {
                    return GrantOutcome::Refused { current: *record };
                }
                *record = OwnerRecord {
                    authority: owner,
                    fence,
                    lease_expires,
                    in_transfer: None,
                };
                GrantOutcome::Granted
            }
        }
    }

    /// Soft liveness renewal: extends the lease iff the fence matches exactly.
    /// Loss-tolerant by design (the lease just lapses); returns whether it applied.
    pub fn renew(&mut self, key: DirectoryKey, fence: Fence, now: UniverseTick) -> bool {
        let ttl = self.tuning.lease_ttl_ticks;
        match self.records.get_mut(&key) {
            Some(record) if record.fence == fence => {
                record.lease_expires = UniverseTick(now.0.saturating_add(ttl));
                true
            }
            _ => false,
        }
    }

    /// Revoke before reassignment. Requires the exact current fence and no
    /// in-flight transfer; the record is removed (the key becomes grantable at a
    /// higher fence — the holder self-fences on receipt of the revoke).
    pub fn revoke(&mut self, key: DirectoryKey, fence: Fence) -> RevokeOutcome {
        match self.records.get(&key) {
            None => RevokeOutcome::UnknownKey,
            Some(record) => {
                if record.in_transfer.is_some() || record.fence != fence {
                    return RevokeOutcome::Refused { current: *record };
                }
                self.records.remove(&key);
                RevokeOutcome::Revoked
            }
        }
    }

    /// Mark a key as locked by a transfer saga (at most one per key; P2 drives
    /// this — the field's enforcement exists from day one).
    pub fn lock_transfer(&mut self, key: DirectoryKey, transfer: TransferId) -> bool {
        match self.records.get_mut(&key) {
            Some(record) if record.in_transfer.is_none() => {
                record.in_transfer = Some(transfer);
                true
            }
            _ => false,
        }
    }

    /// THE commit point: CAS on the fence. The winner flips authority to
    /// `new_owner`, bumps the fence, and clears the transfer lock; a loser (stale
    /// `expected`) observes `Lost` and MUST no-op (fence rule 3).
    ///
    /// SINGLE-KEY ONLY (correct for P2 — only childless dots cross). ⚠️ DEFERRED D-33: the
    /// compound ship handoff (P8) needs an ATOMIC N+1-key commit — bump `Ship(ShipId)` AND
    /// re-parent every `ChildOf` `OwnerRecord` under ONE lock so a passenger's authority can
    /// never flip on a different tick than its hull's. That grows additively to a
    /// `commit_cas_bundle(primary, slaved, …)`; this single-key form stays the N=0 case (HR3).
    pub fn commit_cas(
        &mut self,
        key: DirectoryKey,
        expected: Fence,
        new_owner: AuthorityRef,
        now: UniverseTick,
    ) -> CasOutcome {
        let ttl = self.tuning.lease_ttl_ticks;
        match self.records.get_mut(&key) {
            Some(record) => match Fence::cas_next(record.fence, expected) {
                Some(new_fence) => {
                    *record = OwnerRecord {
                        authority: new_owner,
                        fence: new_fence,
                        lease_expires: UniverseTick(now.0.saturating_add(ttl)),
                        in_transfer: None,
                    };
                    CasOutcome::Won { new_fence }
                }
                None => CasOutcome::Lost {
                    current: record.fence,
                },
            },
            None => CasOutcome::Lost {
                current: Fence::GENESIS,
            },
        }
    }

    /// The abort CAS: same fence race as commit (mutual exclusion by construction),
    /// but authority STAYS with the current owner; only the lock clears.
    pub fn abort_cas(&mut self, key: DirectoryKey, expected: Fence) -> CasOutcome {
        match self.records.get_mut(&key) {
            Some(record) => match Fence::cas_next(record.fence, expected) {
                Some(new_fence) => {
                    record.fence = new_fence;
                    record.in_transfer = None;
                    CasOutcome::Won { new_fence }
                }
                None => CasOutcome::Lost {
                    current: record.fence,
                },
            },
            None => CasOutcome::Lost {
                current: Fence::GENESIS,
            },
        }
    }

    /// Hint head-read (a mid-flip read never rejects).
    #[must_use]
    pub fn head(&self, key: DirectoryKey) -> Option<OwnerRecord> {
        self.records.get(&key).copied()
    }

    /// Every record, in key order — the oracle's AUTHORITY-UNIQUE ground truth and
    /// the admin snapshot's directory dump.
    pub fn entries(&self) -> impl Iterator<Item = (&DirectoryKey, &OwnerRecord)> {
        self.records.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::pose::RealmId;
    use vd_core::{NodeId, SessionId};

    const TUNING: DirectoryTuning = DirectoryTuning {
        lease_ttl_ticks: 100,
    };
    const NOW: UniverseTick = UniverseTick(50);

    fn key() -> DirectoryKey {
        DirectoryKey::Session(SessionId(7))
    }
    fn shard(n: u64) -> AuthorityRef {
        AuthorityRef::Shard(NodeId(n))
    }

    #[test]
    fn fresh_grant_creates_the_record_with_a_lease() {
        let mut dir = DirectoryCore::new(TUNING);
        assert_eq!(
            dir.grant(key(), shard(1), Fence(1), NOW),
            GrantOutcome::Granted
        );
        let record = dir.head(key()).expect("record exists");
        assert_eq!(record.authority, shard(1));
        assert_eq!(record.fence, Fence(1));
        assert_eq!(record.lease_expires, UniverseTick(150));
        assert_eq!(record.in_transfer, None);
    }

    #[test]
    fn regrant_same_owner_same_fence_is_an_idempotent_lease_refresh() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        assert_eq!(
            dir.grant(key(), shard(1), Fence(1), UniverseTick(80)),
            GrantOutcome::Granted
        );
        assert_eq!(
            dir.head(key()).expect("record").lease_expires,
            UniverseTick(180),
            "lease refreshed, nothing else moved"
        );
    }

    #[test]
    fn stale_or_equal_fence_owner_changes_are_refused_with_the_current_record() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(5), NOW);
        let current = dir.head(key()).expect("record");
        // Equal fence, different owner.
        assert_eq!(
            dir.grant(key(), shard(2), Fence(5), NOW),
            GrantOutcome::Refused { current }
        );
        // Stale fence.
        assert_eq!(
            dir.grant(key(), shard(2), Fence(4), NOW),
            GrantOutcome::Refused { current }
        );
        // A HIGHER fence wins ownership (recovery/reassignment path).
        assert_eq!(
            dir.grant(key(), shard(2), Fence(6), NOW),
            GrantOutcome::Granted
        );
        assert_eq!(dir.head(key()).expect("record").authority, shard(2));
    }

    #[test]
    fn transfer_lock_blocks_grants_and_revokes() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        assert!(dir.lock_transfer(key(), TransferId(9)));
        assert!(
            !dir.lock_transfer(key(), TransferId(10)),
            "one saga per key"
        );
        let current = dir.head(key()).expect("record");
        assert_eq!(
            dir.grant(key(), shard(2), Fence(2), NOW),
            GrantOutcome::Refused { current }
        );
        assert_eq!(
            dir.revoke(key(), Fence(1)),
            RevokeOutcome::Refused { current }
        );
    }

    #[test]
    fn lock_on_unknown_key_is_refused() {
        let mut dir = DirectoryCore::new(TUNING);
        assert!(!dir.lock_transfer(key(), TransferId(1)));
    }

    #[test]
    fn renew_extends_only_on_exact_fence() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(3), NOW);
        assert!(dir.renew(key(), Fence(3), UniverseTick(70)));
        assert_eq!(
            dir.head(key()).expect("record").lease_expires,
            UniverseTick(170)
        );
        assert!(!dir.renew(key(), Fence(2), UniverseTick(70)), "stale fence");
        assert!(
            !dir.renew(key(), Fence(4), UniverseTick(70)),
            "future fence"
        );
        assert!(
            !dir.renew(DirectoryKey::Realm(RealmId::Planet(1)), Fence(1), NOW),
            "unknown key"
        );
    }

    #[test]
    fn revoke_requires_exact_fence_and_frees_the_key() {
        let mut dir = DirectoryCore::new(TUNING);
        assert_eq!(dir.revoke(key(), Fence(1)), RevokeOutcome::UnknownKey);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        let current = dir.head(key()).expect("record");
        assert_eq!(
            dir.revoke(key(), Fence(2)),
            RevokeOutcome::Refused { current }
        );
        assert_eq!(dir.revoke(key(), Fence(1)), RevokeOutcome::Revoked);
        assert_eq!(dir.head(key()), None);
        // Re-grantable afterwards (at any fence — the key is fresh).
        assert_eq!(
            dir.grant(key(), shard(2), Fence(2), NOW),
            GrantOutcome::Granted
        );
    }

    #[test]
    fn commit_cas_flips_authority_and_clears_the_lock() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        assert!(dir.lock_transfer(key(), TransferId(5)));
        let outcome = dir.commit_cas(key(), Fence(1), shard(2), UniverseTick(60));
        assert_eq!(
            outcome,
            CasOutcome::Won {
                new_fence: Fence(2)
            }
        );
        let record = dir.head(key()).expect("record");
        assert_eq!(record.authority, shard(2));
        assert_eq!(record.in_transfer, None);
        assert_eq!(record.lease_expires, UniverseTick(160));
    }

    #[test]
    fn commit_and_abort_are_mutually_exclusive_on_the_same_fence() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        let won = dir.commit_cas(key(), Fence(1), shard(2), NOW);
        let new_fence = Fence(2);
        assert_eq!(won, CasOutcome::Won { new_fence });
        // The racing abort sees the bumped fence and LOSES — it must no-op.
        assert_eq!(
            dir.abort_cas(key(), Fence(1)),
            CasOutcome::Lost { current: new_fence }
        );
        assert_eq!(
            dir.head(key()).expect("record").authority,
            shard(2),
            "the loser changed nothing"
        );
    }

    #[test]
    fn abort_cas_keeps_authority_and_clears_the_lock() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        assert!(dir.lock_transfer(key(), TransferId(5)));
        let outcome = dir.abort_cas(key(), Fence(1));
        assert_eq!(
            outcome,
            CasOutcome::Won {
                new_fence: Fence(2)
            }
        );
        let record = dir.head(key()).expect("record");
        assert_eq!(record.authority, shard(1), "abort retains the source");
        assert_eq!(record.in_transfer, None);
        // And the racing commit now loses.
        assert_eq!(
            dir.commit_cas(key(), Fence(1), shard(2), NOW),
            CasOutcome::Lost { current: Fence(2) }
        );
    }

    #[test]
    fn cas_on_unknown_keys_loses_at_genesis() {
        let mut dir = DirectoryCore::new(TUNING);
        assert_eq!(
            dir.commit_cas(key(), Fence(1), shard(1), NOW),
            CasOutcome::Lost {
                current: Fence::GENESIS
            }
        );
        assert_eq!(
            dir.abort_cas(key(), Fence(1)),
            CasOutcome::Lost {
                current: Fence::GENESIS
            }
        );
    }

    #[test]
    fn entries_iterate_in_key_order_for_the_oracle() {
        let mut dir = DirectoryCore::new(TUNING);
        let k1 = DirectoryKey::Session(SessionId(1));
        let k2 = DirectoryKey::Session(SessionId(2));
        let _ = dir.grant(k2, shard(2), Fence(1), NOW);
        let _ = dir.grant(k1, shard(1), Fence(1), NOW);
        let keys: Vec<DirectoryKey> = dir.entries().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![k1, k2], "BTreeMap order, deterministic");
    }
}
