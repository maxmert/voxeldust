//! DEAD OR MERELY SLOW: the discriminator, and the reaper that may act on it.
//!
//! Owns: the per-peer evidence ledger (consecutive unreachable notices, cleared by any success) and
//! the AND-gate that decides whether a directory record may be reaped — a lease that lapsed AND a
//! peer confirmed dead AND the post-restart freeze elapsed AND no live saga still holds the subject.
//!
//! Does NOT own: the timing budget. Every window here comes from the tuning struct, because the
//! ordering that prevents split-brain — a holder self-fences strictly before anyone may reassign —
//! is a relationship between budgets, not a property of any one of them.

use super::{LiveSaga, PendingReHome, SagaRuntimeRes};
use std::collections::BTreeMap;
use vd_core::{Fence, NodeId, TransferId, UniverseTick};
use vd_sim::directory::DirectoryCore;
use vd_sim::saga::LivenessTuning;
use vd_wire::seams::directory::{DirectoryKey, OwnerRecord};

/// D-3 dead-vs-slow EVIDENCE for one peer: how many CONSECUTIVE `NodeUnreachable` notices (no
/// intervening successful inbound) have been observed toward it, and when the run STARTED — the
/// abort-budget anchor, PRESERVED across the run (never re-armed per notice, unlike the saga's `since`).
#[derive(Clone, Copy, Debug)]
struct UnreachEvidence {
    consecutive: u32,
    first_unreachable_tick: UniverseTick,
    /// D-3 Strong-AND MONOTONE latch: set true the moment `consecutive` first reaches
    /// `n_consecutive_unreachable`, and held true across subsequent window RESETS (the redial backoff
    /// spaces late notices past the window, so the freshness-gated `is_confirmed_dead` PULSE expires long
    /// before the `should_reap` reassign horizon). Cleared only by `record_ack` removing the whole entry.
    /// This is what lets `should_reap` gate on `is_latched_dead` at `lease_expires + max` without the
    /// pulse having expired — the split-brain deadline can never be defeated by a stale evidence run.
    confirmed_dead_latched: bool,
}

/// D-3 the dead-vs-slow discriminator (the CSCALE-1 cure). Replaces the insert-only / never-cleared
/// `dead_participants` set: a peer is CONFIRMED dead only after `n_consecutive_unreachable`
/// `NodeUnreachable` notices within `unreachable_window_ticks` with NO intervening successful inbound —
/// so a single recoverable blip toward a HEALTHY peer never confirms it dead, and a recovered peer
/// un-marks itself on its next inbound (clear-on-ack). RAM-only: rebuilt EMPTY on rehydrate (the D-6
/// freeze — a restarted orchestrator confirms nobody dead until fresh post-restart notices re-accrue,
/// so an orchestrator outage never mass-orphans). The default tuning (`n == 1`) is kill-equivalent.
#[derive(Default)]
pub(crate) struct LivenessTracker {
    seen: std::collections::BTreeMap<NodeId, UnreachEvidence>,
    tuning: LivenessTuning,
}

impl LivenessTracker {
    pub(crate) fn new(tuning: LivenessTuning) -> LivenessTracker {
        LivenessTracker {
            seen: std::collections::BTreeMap::new(),
            tuning,
        }
    }

    /// A `NodeUnreachable` toward `node` at `now`: EXTEND its consecutive run, or START a fresh run if
    /// the window since the run's first notice has elapsed (a stale run is not evidence of a live death).
    pub(crate) fn record_unreachable(&mut self, node: NodeId, now: UniverseTick) {
        let ev = self.seen.entry(node).or_insert(UnreachEvidence {
            consecutive: 0,
            first_unreachable_tick: now,
            confirmed_dead_latched: false,
        });
        if now.0.saturating_sub(ev.first_unreachable_tick.0) > self.tuning.unreachable_window_ticks
        {
            // A stale run RESETS its consecutive count + anchor — but the `confirmed_dead_latched` MONOTONE
            // latch is deliberately PRESERVED (only `record_ack` clears it): the redial backoff makes late
            // notices arrive past the window, so an ongoing partition resets its run repeatedly, yet a peer
            // that once reached `n` stays confirmed dead for `should_reap`'s ttl+max horizon.
            ev.consecutive = 1;
            ev.first_unreachable_tick = now;
        } else {
            ev.consecutive = ev.consecutive.saturating_add(1);
        }
        if ev.consecutive >= self.tuning.n_consecutive_unreachable {
            ev.confirmed_dead_latched = true;
        }
    }

    /// ANY successful inbound from `node`: it is alive — clear its evidence, INCLUDING the monotone latch
    /// (the whole entry is removed; idempotent). The clear-on-ack that makes a recoverable blip non-fatal
    /// and un-latches a peer that came back.
    pub(crate) fn record_ack(&mut self, node: NodeId) {
        self.seen.remove(&node);
    }

    /// Is `node` CONFIRMED dead at `now`: enough consecutive notices AND the run still within the window
    /// (a too-old run is stale — re-confirmation needs fresh notices). Bitwise `&` keeps both operands
    /// covered with no short-circuit branch (HR5).
    pub(crate) fn is_confirmed_dead(&self, node: NodeId, now: UniverseTick) -> bool {
        let Some(ev) = self.seen.get(&node) else {
            return false;
        };
        let enough = ev.consecutive >= self.tuning.n_consecutive_unreachable;
        let fresh = now.0.saturating_sub(ev.first_unreachable_tick.0)
            <= self.tuning.unreachable_window_ticks;
        enough & fresh
    }

    /// D-3 Strong-AND: is `node` PERSISTENTLY confirmed dead — the MONOTONE latch (set the first tick its
    /// consecutive run reached `n`, held across window resets until a live inbound clears the whole entry).
    /// Unlike [`is_confirmed_dead`](LivenessTracker::is_confirmed_dead) (a freshness-gated PULSE that expires
    /// once the redial backoff spaces late notices past the window), the latch survives to `should_reap`'s
    /// `lease_expires + max` reassign horizon — so the ttl+max split-brain deadline is never defeated by a
    /// stale evidence run. RAM-only ⇒ empty on rehydrate (the D-6 mass-orphan freeze holds: a restarted
    /// orchestrator latches nobody until fresh post-restart notices re-accrue). No freshness/`now` term —
    /// the latch IS the persistence; the reassign-timing gate lives in `should_reap`.
    pub(crate) fn is_latched_dead(&self, node: NodeId) -> bool {
        self.seen
            .get(&node)
            .is_some_and(|ev| ev.confirmed_dead_latched)
    }

    /// Re-tune the discriminator (the prod config path sets it via `with_tunings`; a test scenario that
    /// needs a specific confirmation margin — e.g. the CSCALE-1 flap cells at `n = 3` — sets it here).
    pub(crate) fn set_tuning(&mut self, tuning: LivenessTuning) {
        self.tuning = tuning;
    }
}

/// D-3 Slice 4 + Strong-AND split-brain fix: the CAP AND-gate deciding if a directory record may be REAPED.
/// All THREE must hold (never OR): (1) the post-restart quiesce window has elapsed (a rebuilt orchestrator
/// does not reap until its liveness evidence has had time to re-accrue); (2) `now` is past the SPLIT-BRAIN
/// DEADLINE `lease_expires + max_self_fence_grace_ticks` — the upper bound (across the two clock domains,
/// sized by [`THETA_MAX`](vd_sim::directory::THETA_MAX) so it dominates a CPU-throttled holder's self-fence)
/// on WHEN a partitioned holder has provably hard-stopped its own authority. This subsumes the lapse gate
/// (a live lease has `lease_expires >= now`) and is the CORE FIX: the old gate reaped at `lease_expires`
/// (=ttl), while a holder self-fences at `grace > ttl` — a verified ~ttl→grace two-holder split-brain
/// window. Reaping only at ttl+max means the holder is PROVABLY self-fenced first (zero zombie window).
/// (3) the owner is PERSISTENTLY confirmed dead — [`is_latched_dead`](LivenessTracker::is_latched_dead), the
/// MONOTONE latch (NOT the freshness-gated `is_confirmed_dead` pulse, which expires far before ttl+max once
/// the redial backoff spaces notices past the window). The evidence gate that (a) never reaps a slow-but-
/// alive holder and (b) realizes the binding CAP choice: the RAM tracker is empty on rehydrate ⇒ nobody
/// latched ⇒ an orchestrator outage FREEZES recovery, NEVER mass-orphans. Three monomorphic `if`s (HR5; no
/// short-circuit `&&`); each corner is covered.
#[must_use]
pub(crate) fn should_reap(
    record: &OwnerRecord,
    now: UniverseTick,
    liveness: &LivenessTracker,
    quiesced_until: UniverseTick,
    max_self_fence_grace_ticks: u64,
) -> bool {
    if now.0 < quiesced_until.0 {
        return false;
    }
    let reassign_after = record
        .lease_expires
        .0
        .saturating_add(max_self_fence_grace_ticks);
    if now.0 <= reassign_after {
        return false;
    }
    if !liveness.is_latched_dead(record.authority.node()) {
        return false;
    }
    true
}

/// D-3 Slice 4 + D-37 Slice 3: the orchestrator expiry REAPER. Once per `reaper_interval_ticks` (re-armed
/// via `last_reap_tick`, never per tick — an O(directory) sweep), resolve every lapsed-AND-confirmed-dead
/// lease, fanned out by key family:
/// - **`Session`** → FULLY REVOKE (the client reconnects via its ResumeTicket — a well-defined path).
/// - **`Entity` (UNLOCKED)** → enqueue a STANDING re-home (D-37 Slice 3): recovered onto a live shard via
///   [`process_rehome_starts`], NOT revoked — revoking would `HeldNowhere`-strand the entity. A LOCKED
///   `Entity` (a saga already owns the key, e.g. an in-flight CELL-1/2 re-home) is LEFT to that saga.
/// - **`Realm`/`Ship`** → LEFT (the durable Realm/Ship re-home is owed Slice 4; the dead realm is the honest
///   `RealmHeldNowhere` residual — revoking a player's ship realm would freeze the ship forever).
///
/// Collect-then-act (the immutable `entries()` borrow ends before the `revoke`/`pending_rehome` mutations).
/// INERT when `reaper_interval_ticks == 0` (the pre-D-3 default). Runs INSIDE the D-6 group-commit barrier
/// (before the directory reconcile, and immediately before `process_rehome_starts`) so a revoke / a locked +
/// armed re-home is captured by the same tick's reconcile + commit (durable — never resurrects on a kill-9).
/// Whether a LIVE saga already owns this subject key's recovery. The standing re-home MUST consult this,
/// not just `record.in_transfer`: `commit_cas` CLEARS `in_transfer` at the commit point while a
/// POST-commit `Promoting`/`ReHoming` saga lives on (it re-homes a dead committed owner via
/// `scan_deadlines`), so a dead-owner key can be UNLOCKED yet still owned by an in-flight saga. Without
/// this check the reaper would arm a SECOND re-home on that key — a double recovery-arm that steals the
/// lock and leaks a parked saga (audit `wf_3b9eb7f0`). This makes "one re-home arm per key" an ENFORCED
/// invariant (the in_transfer lock alone does not, post-commit). O(live sagas) per reapable key — the
/// reaper is interval-paced; a subject→saga index is the MMO-scale optimization, not built now.
fn subject_has_live_saga(sagas: &BTreeMap<TransferId, LiveSaga>, subject: DirectoryKey) -> bool {
    sagas.values().any(|s| s.ctx.subject == subject)
}

pub(crate) fn reap_lapsed_leases(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    now: UniverseTick,
) {
    let interval = dir.tuning().reaper_interval_ticks;
    if (interval == 0) | (now.0.saturating_sub(runtime.last_reap_tick.0) < interval) {
        return;
    }
    runtime.last_reap_tick = now;
    let quiesced_until = runtime.liveness_quiesced_until;
    // The split-brain reassign deadline term (D-3 Strong-AND): `should_reap` reaps only PAST
    // `lease_expires + max_self_fence_grace_ticks`, so a partitioned holder has provably self-fenced first.
    let max_grace = dir.tuning().max_self_fence_grace_ticks;
    // ONE pass over the past-deadline-AND-latched-dead leases; collect per family, then act.
    let mut to_revoke: Vec<(DirectoryKey, Fence)> = Vec::new();
    let mut to_rehome: Vec<PendingReHome> = Vec::new();
    for (key, record) in dir.entries() {
        if !should_reap(record, now, &runtime.liveness, quiesced_until, max_grace) {
            continue;
        }
        match key {
            DirectoryKey::Session(_) => to_revoke.push((*key, record.fence)),
            // Standing re-home a dead-owner Entity ONLY when it is BOTH unlocked AND has no live saga: a
            // post-commit `Promoting` saga's key is UNLOCKED (commit_cas cleared the lock) yet still owned
            // by that in-flight saga (which re-homes it itself) — arming a second re-home here would
            // double-arm + leak. Bitwise `&` (HR5: both predicates always evaluated, no short-circuit gap).
            DirectoryKey::Entity(_)
                if record.in_transfer.is_none() & !subject_has_live_saga(&runtime.sagas, *key) =>
            {
                to_rehome.push(PendingReHome {
                    subject: *key,
                    dead_owner: record.authority.node(),
                    prev_fence: record.fence,
                });
            }
            // A LOCKED Entity, an Entity a live saga still owns, a Realm, or a Ship → left for the owning
            // saga / Slice 4 (never double-armed).
            DirectoryKey::Entity(_) | DirectoryKey::Realm(_) | DirectoryKey::Ship(_) => {}
        }
    }
    for (key, fence) in to_revoke {
        let _ = dir.revoke(key, fence);
    }
    runtime.pending_rehome.extend(to_rehome);
}
