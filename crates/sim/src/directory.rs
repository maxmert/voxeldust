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

use vd_core::{Fence, TickId, TransferId, UniverseTick};
use vd_wire::seams::directory::{AuthorityRef, CasOutcome, DirectoryKey, OwnerRecord};

/// Directory-side operational parameters (ONE reviewed struct — never inline
/// literals at use sites). The D-3 lease-liveness knobs default INERT (renew/reaper
/// intervals 0 = no heartbeat / no sweep, exactly like `realm_recheck_interval == 0`)
/// so a default-tuned cluster behaves identically to pre-D-3; production sets them from env.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectoryTuning {
    /// Soft lease length granted on every grant/renewal, in universe ticks.
    pub lease_ttl_ticks: u64,
    /// How often (in the holder's LOCAL ticks) an authority holder re-sends `LeaseRenew`.
    /// `0` = INERT (no heartbeat producer fires). Prod ≈ `lease_ttl_ticks / 4`.
    pub lease_renew_interval_ticks: u64,
    /// How many renew intervals must fit inside one TTL (the renewal safety margin: a lease
    /// survives losing `min_renews_before_lapse - 1` consecutive renewals). Used by `validate`.
    pub min_renews_before_lapse: u32,
    /// A holder whose lease provably lapsed THIS many LOCAL ticks ago hard-stops its own authority
    /// (self-fence-before-grant, D-3 Slice 5) — strictly BEFORE the orchestrator may reassign.
    pub self_fence_grace_ticks: u64,
    /// The orchestrator waits `lease_ttl_ticks + max_self_fence_grace_ticks` after a lapse before
    /// reassigning — the upper bound on a holder's self-fence, so no overlap (= no split-brain).
    pub max_self_fence_grace_ticks: u64,
    /// How often (universe ticks) the orchestrator sweeps for lapsed-AND-confirmed-dead leases.
    /// `0` = INERT (no reaper). Prod a small interval (the sweep is O(directory) like `scan_deadlines`).
    pub reaper_interval_ticks: u64,
    /// A FIXED post-restart freeze (universe ticks) after a rehydrate before the reaper may act —
    /// belt-and-suspenders atop the RAM-only liveness tracker (which is the real CAP freeze: an empty
    /// tracker confirms nobody dead until fresh post-restart `NodeUnreachable`s re-accrue). NOT
    /// downtime-proportional (the virtual clock cannot measure wall-time).
    pub recovery_grace_ticks: u64,
}

/// A mis-tuned [`DirectoryTuning`] — rejected LOUD at boot (the bin calls `validate` after env read),
/// never a silent lease-liveness misconfiguration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DirectoryTuningError {
    #[error(
        "lease renewal is too sparse: lease_renew_interval_ticks ({interval}) * \
         min_renews_before_lapse ({renews}) = {product} must be <= lease_ttl_ticks ({ttl}) so a \
         lease survives losing renewals (a renewal heartbeat that cannot fit the TTL lapses a live holder)"
    )]
    RenewTooSparse {
        interval: u64,
        renews: u32,
        product: u64,
        ttl: u64,
    },
    #[error(
        "self_fence_grace_ticks ({grace}) must be > lease_ttl_ticks ({ttl}): a holder must keep \
         authority through its whole lease and only self-fence AFTER it provably lapsed"
    )]
    SelfFenceWithinTtl { ttl: u64, grace: u64 },
    #[error(
        "self_fence_grace_ticks ({grace}) must be <= lease_ttl_ticks ({ttl}) + \
         max_self_fence_grace_ticks ({max}): the holder must finish self-fencing BEFORE the \
         orchestrator's reassign-after window opens (no overlap = no split-brain)"
    )]
    SelfFenceOutlivesReassign { ttl: u64, grace: u64, max: u64 },
}

impl Default for DirectoryTuning {
    fn default() -> DirectoryTuning {
        DirectoryTuning {
            lease_ttl_ticks: 100,
            lease_renew_interval_ticks: 0,
            min_renews_before_lapse: 4,
            self_fence_grace_ticks: 150,
            max_self_fence_grace_ticks: 50,
            reaper_interval_ticks: 0,
            recovery_grace_ticks: 200,
        }
    }
}

impl DirectoryTuning {
    /// Reject a mis-tuned lease-liveness budget at boot. Enforces the binding ordering chain so the
    /// heartbeat keeps live leases alive AND a holder always self-fences before the orchestrator
    /// reassigns (the no-split-brain invariant). An INERT renew interval (`0`) skips the renewal
    /// margin check — there is no heartbeat to under-provision.
    ///
    /// # Errors
    /// [`DirectoryTuningError`] for a renewal too sparse to survive the TTL, or a self-fence grace
    /// that is inside the TTL (too eager) or outlives the orchestrator's reassign-after window (overlap).
    pub fn validate(&self) -> Result<(), DirectoryTuningError> {
        // ALL D-3 checks are gated on the heartbeat being ACTIVE (`lease_renew_interval_ticks != 0`):
        // when inert (renewal off — the pre-D-3 default, every existing in-process rig, and the dev
        // cluster) no lease is ever renewed, reaped, or self-fenced, so the ordering chain is vacuous and
        // a large `lease_ttl_ticks` with default graces is fine. The bitwise `&` keeps both operands
        // covered with no short-circuit branch (HR5, mirroring `SagaTuning::validate`'s `|`).
        let active = self.lease_renew_interval_ticks != 0;
        let product = self
            .lease_renew_interval_ticks
            .saturating_mul(self.min_renews_before_lapse as u64);
        if active & (product > self.lease_ttl_ticks) {
            return Err(DirectoryTuningError::RenewTooSparse {
                interval: self.lease_renew_interval_ticks,
                renews: self.min_renews_before_lapse,
                product,
                ttl: self.lease_ttl_ticks,
            });
        }
        if active & (self.self_fence_grace_ticks <= self.lease_ttl_ticks) {
            return Err(DirectoryTuningError::SelfFenceWithinTtl {
                ttl: self.lease_ttl_ticks,
                grace: self.self_fence_grace_ticks,
            });
        }
        if active
            & (self.self_fence_grace_ticks > self.lease_ttl_ticks + self.max_self_fence_grace_ticks)
        {
            return Err(DirectoryTuningError::SelfFenceOutlivesReassign {
                ttl: self.lease_ttl_ticks,
                grace: self.self_fence_grace_ticks,
                max: self.max_self_fence_grace_ticks,
            });
        }
        Ok(())
    }
}

/// THE proactive self-fence predicate (D-3 Slice 5) — ONE source of truth for the split-brain-critical
/// fence-rule-4 timing, shared by EVERY authority holder (the shard's `Realm`, the gateway's `Session`),
/// so the two can never drift apart. A holder self-fences (hard-stops its own authority) iff ALL hold:
///
/// - `held` — it currently holds the lease (an unowned holder has nothing to fence);
/// - `grace != 0` — the proactive timer is ARMED (`0` = INERT, the pre-D-3 default and every legacy rig);
/// - `recheck != 0` — the round-trip CONFIRMATION channel exists (without periodic head re-reads there is
///   no `confirmed` tick to measure against, so the timer must stay inert — never fire on a config that
///   cannot observe its own liveness);
/// - `local_tick - confirmed > grace` — the last round-trip confirmation is STALER than the grace, i.e.
///   the holder has been out of contact with the orchestrator longer than fence rule 4 permits. Saturating
///   so a (clock-skew) confirmation stamped in the future reads as zero staleness, never a panic.
///
/// `local_tick` MUST be the partition-surviving local clock (advances every `step_tick` regardless of
/// `ClockSync`); `confirmed` advances ONLY on a round-trip that affirmed ownership, so it freezes under a
/// partition while `local_tick` climbs. The split-brain-safe sizing `lease_ttl < grace <= lease_ttl +
/// max_self_fence_grace` (the holder self-fences AFTER its lease lapses but BEFORE the orchestrator's
/// reassign window opens) is enforced by [`DirectoryTuning::validate`]. Branchless bitwise `&` (HR5): every
/// operand is one region with no short-circuit, so a single armed call covers them all and only the
/// caller's `if` carries the two-arm branch.
#[must_use]
pub fn lease_self_fence_due(
    held: bool,
    grace: u64,
    recheck: u64,
    local_tick: TickId,
    confirmed: TickId,
) -> bool {
    held & (grace != 0) & (recheck != 0) & (local_tick.0.saturating_sub(confirmed.0) > grace)
}

/// THE per-tick cadence guard (D-3) — ONE source for "is a `tick`-periodic action due THIS tick", shared
/// by every lease-liveness producer (the shard's realm-recheck + Realm/Entity heartbeat, the gateway's
/// Session renew + recheck) so the `interval == 0 ⇒ INERT` convention can never be written two ways. `true`
/// iff the cadence is ARMED (`interval != 0` — `0` disables, the pre-D-3 default) AND `tick` lands on a
/// multiple of it. The `&&` short-circuits so `is_multiple_of` is never called with a `0` divisor.
#[must_use]
pub fn due_this_tick(interval: u64, tick: u64) -> bool {
    interval != 0 && tick.is_multiple_of(interval)
}

/// A mis-tuned proactive self-fence cadence — rejected LOUD at a NODE's boot (the shard/gateway bin calls
/// [`validate_self_fence_cadence`] after env read), never a silent mass-self-fence of HEALTHY holders. The
/// recheck cadence (`realm_recheck_interval` / `session_recheck_interval`) is a NODE-side knob the
/// orchestrator never sees, so this check CANNOT live in [`DirectoryTuning::validate`] — it is the node's
/// own guard, complementary to the orchestrator's `lease_ttl < grace <= lease_ttl + max` ordering chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SelfFenceCadenceError {
    #[error(
        "the proactive self-fence is ARMED (self_fence_grace_ticks = {grace}) but its confirmation \
         channel is OFF (recheck interval = 0): with no round-trip to re-arm the `confirmed` stamp, \
         every held lease would self-fence the instant the grace elapses"
    )]
    NoConfirmationChannel { grace: u64 },
    #[error(
        "self_fence_grace_ticks ({grace}) must be >= 2 * recheck interval ({recheck}) so a HEALTHY \
         holder observes at least one full affirming round-trip well inside the grace: the `confirmed` \
         stamp re-arms only once per recheck cadence, so a smaller grace lets it go stale past the \
         deadline between on-time replies and a perfectly live holder self-fences with no partition"
    )]
    GraceBelowRecheckMargin { grace: u64, recheck: u64 },
}

/// Reject a mis-tuned proactive self-fence cadence at a node's boot (D-3 Slice 5). When the self-fence is
/// ARMED (`grace != 0`) the confirmation channel MUST exist (`recheck != 0`) and the grace MUST span at
/// least two recheck cycles (`grace >= 2 * recheck`) — so a healthy holder, whose `confirmed` stamp
/// re-arms only once per recheck cadence (plus round-trip latency, assumed under one cadence), never
/// crosses the deadline between on-time replies. INERT configs (`grace == 0`) always pass — there is no
/// self-fence to mis-provision. This is the node-side complement to [`DirectoryTuning::validate`]'s
/// orchestrator-side `lease_ttl < grace <= lease_ttl + max` chain (the node cannot see `lease_ttl`).
///
/// # Errors
/// [`SelfFenceCadenceError`] when the self-fence is armed without a confirmation channel, or with a grace
/// below the two-recheck margin.
pub fn validate_self_fence_cadence(grace: u64, recheck: u64) -> Result<(), SelfFenceCadenceError> {
    if grace == 0 {
        return Ok(()); // INERT: no proactive self-fence to mis-provision
    }
    if recheck == 0 {
        return Err(SelfFenceCadenceError::NoConfirmationChannel { grace });
    }
    if grace < recheck.saturating_mul(2) {
        return Err(SelfFenceCadenceError::GraceBelowRecheckMargin { grace, recheck });
    }
    Ok(())
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
    /// D-6 (D-alpha) the per-tick DURABILITY DELTA: every key whose persisted snapshot CHANGED since the
    /// last group-commit. `Some(record)` ⇒ the row must be PUT at this value; `None` ⇒ the row was REMOVED
    /// and must be DELETEd. Every mutator arm that actually touches `records` records its post-state here
    /// (NO-CHANGE arms — `Refused`/`Lost`/`false`/`UnknownKey` — stage nothing); the group-commit barrier
    /// drains it ONCE per tick via [`take_dirty`](Self::take_dirty) straight into the same `commit()`.
    /// Within-tick collapse is automatic (last write to a key wins via `BTreeMap::insert`), so N renews of
    /// one key in a tick cost ONE staged delta. This REPLACES the old O(directory) delete-all-then-
    /// put-current reconcile: the store already holds every prior tick's snapshot, so persisting only this
    /// tick's deltas reconstructs the byte-identical durable set (the test-only differential oracle pins
    /// incremental == full). `restore` starts it EMPTY — restored rows are already durable, owing no delta.
    dirty: BTreeMap<DirectoryKey, Option<OwnerRecord>>,
}

impl DirectoryCore {
    #[must_use]
    pub fn new(tuning: DirectoryTuning) -> DirectoryCore {
        DirectoryCore {
            records: BTreeMap::new(),
            tuning,
            dirty: BTreeMap::new(),
        }
    }

    /// The directory's operational tuning (the D-3 reaper reads `reaper_interval_ticks` for its cadence
    /// and the lease/grace knobs; ONE reviewed config home — never an inline literal).
    #[must_use]
    pub fn tuning(&self) -> DirectoryTuning {
        self.tuning
    }

    /// Reconstruct the directory from durably-persisted records (D-6 orchestrator rehydrate). The
    /// persisted `(key, record)` pairs ARE the authority-of-record at the last group-commit, so they
    /// are installed VERBATIM (bypassing `grant`'s fence/lock logic — recovery RESTORES state, it does
    /// not re-decide it; the single commit point is unchanged). The `Directory` key family is persisted
    /// independently of the saga WAL (a distinct prefix) so io-prod can split it into its own file
    /// without a cross-file atomic transaction (the D-32 partitioning seam).
    #[must_use]
    pub fn restore(
        tuning: DirectoryTuning,
        records: impl IntoIterator<Item = (DirectoryKey, OwnerRecord)>,
    ) -> DirectoryCore {
        DirectoryCore {
            records: records.into_iter().collect(),
            tuning,
            dirty: BTreeMap::new(),
        }
    }

    /// D-6 (D-alpha) drain this tick's durable directory deltas (`key → Some(put)|None(delete)`), clearing
    /// the set. The group-commit barrier calls this ONCE per tick; the drained deltas stage straight into
    /// the same `commit()`, so the durable Directory family always matches RAM at the commit point. Empty
    /// between ticks with no directory mutation (the barrier then stages nothing — the per-family no-write
    /// guard, the write-amp win over the old full reconcile).
    #[must_use]
    pub fn take_dirty(&mut self) -> BTreeMap<DirectoryKey, Option<OwnerRecord>> {
        std::mem::take(&mut self.dirty)
    }

    /// Whether any durable directory delta is pending this tick (an assert/oracle hook; the barrier drains
    /// unconditionally). `false` exactly when no mutator changed a row since the last [`take_dirty`].
    #[must_use]
    pub fn has_dirty(&self) -> bool {
        !self.dirty.is_empty()
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
                let record = OwnerRecord {
                    authority: owner,
                    fence,
                    lease_expires,
                    in_transfer: None,
                };
                self.records.insert(key, record);
                self.dirty.insert(key, Some(record)); // new row → durable PUT
                GrantOutcome::Granted
            }
            Some(record) => {
                if record.in_transfer.is_some() {
                    return GrantOutcome::Refused { current: *record };
                }
                if fence == record.fence && owner == record.authority {
                    // Idempotent re-grant: refresh the lease, change nothing else. The lease_expires DID
                    // change, so it is a durable delta (the lease is part of the persisted record).
                    record.lease_expires = lease_expires;
                    self.dirty.insert(key, Some(*record));
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
                self.dirty.insert(key, Some(*record)); // owner/fence changed → durable PUT
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
                // D-alpha decision A: the lease HEARTBEAT participates — the refreshed lease_expires is a
                // durable delta so a kill-9 cannot resurrect a lease at a stale expiry the reaper would
                // misjudge. Within-tick collapse keeps repeated renews of one key to a single staged delta.
                self.dirty.insert(key, Some(*record));
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
                self.dirty.insert(key, None); // row removed → durable DELETE (the COMP-2 anti-zombie)
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
                self.dirty.insert(key, Some(*record)); // in_transfer set → durable PUT
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
                    self.dirty.insert(key, Some(*record)); // THE commit point → durable PUT
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
    ///
    /// ⚠️ This BUMPS the fence (via `cas_next`). It is the FENCE-MOVING abort arm — use it ONLY where the
    /// abort genuinely races a commit on a known-current `expected` fence. Do NOT route a TERMINAL abort
    /// through it: a terminal abort of a SURVIVING source must be FENCE-NEUTRAL (use [`abort_clear`]) or it
    /// strands the source one fence behind the directory and wedges a post-abort logout `LeaseRevoke`
    /// (FENCE-9; see the D-6 abort-path note). A future N-orchestrator router (D-32) wiring terminal aborts
    /// MUST dispatch to `abort_clear`, never here — the `abort_clear_is_fence_neutral` pin guards the producer.
    pub fn abort_cas(&mut self, key: DirectoryKey, expected: Fence) -> CasOutcome {
        match self.records.get_mut(&key) {
            Some(record) => match Fence::cas_next(record.fence, expected) {
                Some(new_fence) => {
                    record.fence = new_fence;
                    record.in_transfer = None;
                    self.dirty.insert(key, Some(*record)); // fence bumped + lock cleared → durable PUT
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

    /// Clear the transfer lock at TERMINAL abort (Slice 2a, closes D-1), RE-READING the head. The
    /// aborter's `expected_fence` (fixed at saga creation) is STALE by definition once any phase bumped
    /// the head, so a fence-matched [`abort_cas`](Self::abort_cas) would also Lose. This re-derives the
    /// current head and clears the lock IFF it is still held by THIS `transfer` — NEVER steals another
    /// saga's lock. The fence is UNCHANGED (FENCE-NEUTRAL): an abort is NOT an ownership change — this
    /// never writes `record.authority`, the source keeps the avatar — so the directory fence must stay
    /// put, keeping `directory.fence == source.entity_fence` (FENCE-9). That is load-bearing: a bumped
    /// fence would strand the surviving source one fence behind, and a post-abort `LeaseRevoke{fence}`
    /// (logout) / re-transfer keyed at the source's fence would be permanently Refused. A stale in-flight
    /// crossing cannot exist here to fence out: `EmitCrossing` is emitted only POST-commit (`CasWon →
    /// Swapping`) and post-commit phases NEVER abort, so the `in_transfer == Some(this)` guard is reachable
    /// only pre/at-CAS (⚠️ this no-bump correctness is contingent on crossings staying post-commit-only).
    /// Idempotent: a re-driven terminal whose lock already cleared (`None`) or moved to another transfer
    /// (`Some(other)`) is a `Lost` no-op — driven by the lock, NOT the fence value.
    pub fn abort_clear(&mut self, key: DirectoryKey, transfer: TransferId) -> CasOutcome {
        match self.records.get_mut(&key) {
            Some(record) if record.in_transfer == Some(transfer) => {
                record.in_transfer = None;
                self.dirty.insert(key, Some(*record)); // lock cleared (fence-neutral) → durable PUT
                CasOutcome::Won {
                    new_fence: record.fence,
                }
            }
            Some(record) => CasOutcome::Lost {
                current: record.fence,
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

    // A const can't use `..Default::default()`, so spell the D-3 knobs (the defaults) explicitly. Kept
    // INERT (renew/reaper 0) — these directory unit tests exercise grant/renew/revoke/CAS, not the D-3
    // heartbeat/reaper systems (those are node-level, tested there).
    const TUNING: DirectoryTuning = DirectoryTuning {
        lease_ttl_ticks: 100,
        lease_renew_interval_ticks: 0,
        min_renews_before_lapse: 4,
        self_fence_grace_ticks: 150,
        max_self_fence_grace_ticks: 50,
        reaper_interval_ticks: 0,
        recovery_grace_ticks: 200,
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
    fn abort_clear_clears_the_lock_by_re_reading_the_stale_head() {
        // Slice 2a (D-1): the aborter's expected fence is STALE (the head advanced) — abort_clear
        // re-reads + clears IFF still held by THIS transfer, keeping authority. FENCE-NEUTRAL: an abort
        // is not an ownership change, so the fence does NOT move (the lock cleared at the SAME fence).
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        assert!(dir.lock_transfer(key(), TransferId(5)));
        // The aborter does NOT know the head fence — it just names its transfer.
        let outcome = dir.abort_clear(key(), TransferId(5));
        assert_eq!(
            outcome,
            CasOutcome::Won {
                new_fence: Fence(1)
            }
        );
        let record = dir.head(key()).expect("record");
        assert_eq!(record.authority, shard(1), "abort retains the source owner");
        assert_eq!(record.in_transfer, None, "the lock cleared");
        assert_eq!(
            record.fence,
            Fence(1),
            "abort is fence-neutral: no ownership change, no bump (FENCE-9)"
        );
    }

    #[test]
    fn abort_clear_is_fence_neutral() {
        // Pin the fence-neutral contract at the unit boundary: a future re-introduction of the bump (or
        // a D-33 bundle-abort that bumps per passenger) is caught here. Authority unchanged, lock cleared,
        // fence stays put — so directory.fence == source.entity_fence (no surviving-source stranding).
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(3), NOW);
        assert!(dir.lock_transfer(key(), TransferId(7)));
        assert_eq!(
            dir.abort_clear(key(), TransferId(7)),
            CasOutcome::Won {
                new_fence: Fence(3)
            }
        );
        let record = dir.head(key()).expect("record");
        assert_eq!(record.fence, Fence(3), "the fence did not move on abort");
        assert_eq!(record.in_transfer, None, "the lock cleared");
        assert_eq!(record.authority, shard(1), "the source still owns");
    }

    #[test]
    fn abort_clear_is_an_idempotent_noop_when_already_clear() {
        // A re-driven terminal abort (the lock already cleared): Lost no-op, fence unchanged. The first
        // clear is fence-neutral (stays Fence(1)), so the second observes the same fence.
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        assert!(dir.lock_transfer(key(), TransferId(5)));
        let _ = dir.abort_clear(key(), TransferId(5)); // clears the lock, fence stays Fence(1)
        assert_eq!(
            dir.abort_clear(key(), TransferId(5)),
            CasOutcome::Lost { current: Fence(1) },
            "the second clear is a no-op (lock already None)"
        );
    }

    #[test]
    fn abort_clear_never_steals_another_sagas_lock() {
        // The lock is held by a DIFFERENT transfer: abort_clear must NOT clear it (a stale abort of
        // saga A cannot unlock a live saga B that re-acquired the key).
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        assert!(dir.lock_transfer(key(), TransferId(9)));
        assert_eq!(
            dir.abort_clear(key(), TransferId(5)),
            CasOutcome::Lost { current: Fence(1) },
            "a stale abort of transfer 5 leaves transfer 9's lock intact"
        );
        assert_eq!(
            dir.head(key()).expect("record").in_transfer,
            Some(TransferId(9)),
            "the other saga's lock survives"
        );
    }

    #[test]
    fn abort_clear_on_an_unknown_key_is_a_genesis_loss() {
        let mut dir = DirectoryCore::new(TUNING);
        assert_eq!(
            dir.abort_clear(key(), TransferId(5)),
            CasOutcome::Lost {
                current: Fence::GENESIS
            },
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

    #[test]
    fn due_this_tick_is_inert_at_zero_and_lands_on_multiples() {
        // The shared cadence guard (D-3): disarmed at interval 0 (the short-circuit never calls
        // is_multiple_of with a 0 divisor); otherwise true exactly on multiples of the interval.
        assert!(!due_this_tick(0, 5), "interval 0 ⇒ INERT, never due");
        assert!(due_this_tick(2, 4), "on a multiple of the interval ⇒ due");
        assert!(
            due_this_tick(2, 0),
            "genesis tick 0 is a multiple of everything"
        );
        assert!(!due_this_tick(2, 5), "off a multiple ⇒ not due");
    }

    #[test]
    fn validate_self_fence_cadence_guards_the_grace_against_the_recheck_cadence() {
        // INERT (grace 0) always passes — no self-fence to mis-provision (recheck value irrelevant).
        assert_eq!(validate_self_fence_cadence(0, 0), Ok(()));
        assert_eq!(validate_self_fence_cadence(0, 200), Ok(()));
        // Armed but NO confirmation channel ⇒ rejected (would self-fence the instant the grace elapses).
        assert_eq!(
            validate_self_fence_cadence(150, 0),
            Err(SelfFenceCadenceError::NoConfirmationChannel { grace: 150 })
        );
        // Armed with too small a grace vs the recheck cadence ⇒ rejected (the review's mass-fence trip:
        // grace 120 < 2 * recheck 200 = 400; a healthy holder would self-fence with no partition).
        assert_eq!(
            validate_self_fence_cadence(120, 200),
            Err(SelfFenceCadenceError::GraceBelowRecheckMargin {
                grace: 120,
                recheck: 200
            })
        );
        // Armed with the two-recheck margin ⇒ OK (the rigs' grace 5 vs recheck 2: 5 >= 4).
        assert_eq!(validate_self_fence_cadence(5, 2), Ok(()));
    }

    // ---- D-3 Slice 0: DirectoryTuning::validate (the lease-liveness ordering chain) ----------------

    #[test]
    fn directory_tuning_default_and_a_prod_config_validate() {
        assert_eq!(DirectoryTuning::default().validate(), Ok(()));
        // A representative prod config: heartbeat every ttl/4, 4 renews fit the ttl exactly.
        let prod = DirectoryTuning {
            lease_ttl_ticks: 100,
            lease_renew_interval_ticks: 25,
            min_renews_before_lapse: 4,
            self_fence_grace_ticks: 130,
            max_self_fence_grace_ticks: 40,
            reaper_interval_ticks: 8,
            recovery_grace_ticks: 200,
        };
        assert_eq!(prod.validate(), Ok(()));
    }

    #[test]
    fn directory_tuning_rejects_a_too_sparse_renewal() {
        // 30 * 4 = 120 > 100: a heartbeat that cannot fit 4 renews in the TTL lapses a live holder.
        let t = DirectoryTuning {
            lease_renew_interval_ticks: 30,
            min_renews_before_lapse: 4,
            ..DirectoryTuning::default()
        };
        assert_eq!(
            t.validate(),
            Err(DirectoryTuningError::RenewTooSparse {
                interval: 30,
                renews: 4,
                product: 120,
                ttl: 100,
            })
        );
    }

    #[test]
    fn directory_tuning_rejects_a_self_fence_within_the_ttl() {
        // grace == ttl: the holder would self-fence before its lease even expires (too eager). The check
        // is gated on an ACTIVE heartbeat, so set a valid renew interval to reach it.
        let t = DirectoryTuning {
            lease_renew_interval_ticks: 10,
            self_fence_grace_ticks: 100,
            ..DirectoryTuning::default()
        };
        assert_eq!(
            t.validate(),
            Err(DirectoryTuningError::SelfFenceWithinTtl {
                ttl: 100,
                grace: 100,
            })
        );
    }

    #[test]
    fn directory_tuning_rejects_a_self_fence_outliving_reassign() {
        // grace 151 > ttl 100 + max 50 = 150: the holder could still be live when the orchestrator
        // reassigns — a split-brain window. Gated on an active heartbeat, so set a valid renew interval.
        let t = DirectoryTuning {
            lease_renew_interval_ticks: 10,
            self_fence_grace_ticks: 151,
            max_self_fence_grace_ticks: 50,
            ..DirectoryTuning::default()
        };
        assert_eq!(
            t.validate(),
            Err(DirectoryTuningError::SelfFenceOutlivesReassign {
                ttl: 100,
                grace: 151,
                max: 50,
            })
        );
    }

    // ── D-6 (D-alpha): the per-tick durability DELTA set ─────────────────────────────────────────────
    // Every mutator arm that CHANGES a row stages a delta (`Some` = PUT, `None` = DELETE); every NO-CHANGE
    // arm (Refused/Lost/false/UnknownKey) stages NOTHING. The group-commit barrier drains the set once per
    // tick; these pins guarantee the incremental reconcile carries exactly the rows that moved.

    #[test]
    fn grant_fresh_stages_a_put_of_the_new_record() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        assert!(dir.has_dirty(), "a fresh grant is a durable change");
        let head = dir.head(key()).expect("record");
        let dirty = dir.take_dirty();
        assert_eq!(
            dirty.get(&key()),
            Some(&Some(head)),
            "the staged PUT is the current record"
        );
        assert!(!dir.has_dirty(), "take_dirty cleared the set");
        assert!(dir.take_dirty().is_empty(), "a second drain is empty");
    }

    #[test]
    fn grant_idempotent_lease_refresh_stages_a_put() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        let _ = dir.take_dirty(); // clear the create
        let _ = dir.grant(key(), shard(1), Fence(1), UniverseTick(80)); // re-grant: lease moves
        let head = dir.head(key()).expect("record");
        assert_eq!(head.lease_expires, UniverseTick(180));
        assert_eq!(
            dir.take_dirty().get(&key()),
            Some(&Some(head)),
            "the refreshed lease is a durable delta"
        );
    }

    #[test]
    fn grant_higher_fence_replace_stages_the_new_owner() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        let _ = dir.take_dirty();
        let _ = dir.grant(key(), shard(2), Fence(5), NOW);
        let head = dir.head(key()).expect("record");
        assert_eq!(head.fence, Fence(5));
        assert_eq!(dir.take_dirty().get(&key()), Some(&Some(head)));
    }

    #[test]
    fn grant_refused_arms_stage_nothing() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(5), NOW);
        let _ = dir.take_dirty();
        // equal-fence different-owner, and stale-fence — both Refused.
        let _ = dir.grant(key(), shard(2), Fence(5), NOW);
        let _ = dir.grant(key(), shard(2), Fence(4), NOW);
        assert!(!dir.has_dirty(), "a refused grant changes no row");
        // refused-while-locked is the third Refused arm (in_transfer.is_some()).
        let _ = dir.lock_transfer(key(), TransferId(1));
        let _ = dir.take_dirty();
        let _ = dir.grant(key(), shard(9), Fence(99), NOW);
        assert!(!dir.has_dirty(), "a grant onto a locked key changes no row");
    }

    #[test]
    fn renew_match_stages_a_put_mismatch_stages_nothing() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        let _ = dir.take_dirty();
        assert!(dir.renew(key(), Fence(1), UniverseTick(80)));
        let head = dir.head(key()).expect("record");
        assert_eq!(
            dir.take_dirty().get(&key()),
            Some(&Some(head)),
            "decision A: the heartbeat renew is a durable delta"
        );
        assert!(!dir.renew(key(), Fence(99), NOW), "fence mismatch");
        assert!(!dir.has_dirty(), "a no-op renew stages nothing");
    }

    #[test]
    fn lock_transfer_stages_a_put_relock_stages_nothing() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        let _ = dir.take_dirty();
        assert!(dir.lock_transfer(key(), TransferId(1)));
        let head = dir.head(key()).expect("record");
        assert_eq!(head.in_transfer, Some(TransferId(1)));
        assert_eq!(dir.take_dirty().get(&key()), Some(&Some(head)));
        assert!(!dir.lock_transfer(key(), TransferId(2)), "already locked");
        assert!(!dir.has_dirty(), "a refused lock stages nothing");
    }

    #[test]
    fn commit_cas_won_stages_lost_stages_nothing() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        let _ = dir.take_dirty();
        let _ = dir.commit_cas(key(), Fence(1), shard(2), NOW); // Won: flips owner, bumps fence
        let head = dir.head(key()).expect("record");
        assert_eq!(head.authority, shard(2));
        assert_eq!(dir.take_dirty().get(&key()), Some(&Some(head)));
        let _ = dir.commit_cas(key(), Fence(1), shard(3), NOW); // stale expected → Lost
        assert!(!dir.has_dirty(), "a lost CAS changes no row");
    }

    #[test]
    fn abort_cas_won_stages_the_bumped_unlocked_record() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        let _ = dir.lock_transfer(key(), TransferId(1));
        let _ = dir.take_dirty();
        let _ = dir.abort_cas(key(), Fence(1)); // Won: bumps fence, clears lock
        let head = dir.head(key()).expect("record");
        assert_eq!(head.in_transfer, None);
        assert_eq!(head.fence, Fence(2));
        assert_eq!(dir.take_dirty().get(&key()), Some(&Some(head)));
        let _ = dir.abort_cas(key(), Fence(1)); // stale → Lost
        assert!(!dir.has_dirty());
    }

    #[test]
    fn abort_clear_won_stages_the_unlocked_record_redrive_stages_nothing() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        let _ = dir.lock_transfer(key(), TransferId(7));
        let _ = dir.take_dirty();
        let _ = dir.abort_clear(key(), TransferId(7)); // Won: clears lock, fence-neutral
        let head = dir.head(key()).expect("record");
        assert_eq!(head.in_transfer, None);
        assert_eq!(head.fence, Fence(1), "abort_clear is fence-neutral");
        assert_eq!(dir.take_dirty().get(&key()), Some(&Some(head)));
        let _ = dir.abort_clear(key(), TransferId(7)); // lock already cleared → Lost
        assert!(!dir.has_dirty(), "a re-driven terminal stages nothing");
    }

    #[test]
    fn revoke_stages_a_delete_refused_stages_nothing() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        let _ = dir.take_dirty();
        assert_eq!(dir.revoke(key(), Fence(1)), RevokeOutcome::Revoked);
        assert_eq!(
            dir.take_dirty().get(&key()),
            Some(&None),
            "revoke stages a DELETE (the COMP-2 anti-zombie)"
        );
        // wrong-fence refuse + unknown-key on a now-absent record stage nothing.
        let _ = dir.grant(key(), shard(1), Fence(3), NOW);
        let _ = dir.take_dirty();
        let _ = dir.revoke(key(), Fence(99)); // Refused
        let _ = dir.revoke(DirectoryKey::Realm(RealmId::Planet(1)), Fence(1)); // UnknownKey
        assert!(!dir.has_dirty());
    }

    #[test]
    fn cas_on_a_missing_key_stages_nothing() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.commit_cas(key(), Fence(1), shard(1), NOW); // None → Lost
        let _ = dir.abort_cas(key(), Fence(1)); // None → Lost
        let _ = dir.abort_clear(key(), TransferId(1)); // None → Lost
        assert!(!dir.has_dirty(), "a CAS on an absent key changes no row");
    }

    #[test]
    fn within_tick_changes_to_one_key_collapse_to_one_delta() {
        let mut dir = DirectoryCore::new(TUNING);
        let _ = dir.grant(key(), shard(1), Fence(1), NOW);
        assert!(dir.renew(key(), Fence(1), UniverseTick(80)));
        assert!(dir.lock_transfer(key(), TransferId(1)));
        let head = dir.head(key()).expect("record");
        let dirty = dir.take_dirty();
        assert_eq!(dirty.len(), 1, "three mutations on one key → one delta");
        assert_eq!(
            dirty.get(&key()),
            Some(&Some(head)),
            "the last write wins (locked, lease-refreshed)"
        );
    }

    #[test]
    fn restore_starts_with_no_pending_deltas() {
        let rec = OwnerRecord {
            authority: shard(1),
            fence: Fence(3),
            lease_expires: UniverseTick(200),
            in_transfer: None,
        };
        let dir = DirectoryCore::restore(TUNING, [(key(), rec)]);
        assert!(
            !dir.has_dirty(),
            "restored rows are already durable — they owe no delta"
        );
    }
}
