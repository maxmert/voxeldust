//! P3 Slice 1 — the shard+gateway CRASH/FAULT RECOVERY MATRIX (PERMANENT gate; PLAN.md P3 headline).
//!
//! "Dots crossing a boundary while the operator kill-9s shards/gateway — never wedge, never duplicate,
//! never vanish." This is the proof that Slice 2a's recovery machinery (the R1 cure) holds under REAL
//! node failures, driven over the real fabric by the DRY `run_fault_scenario` harness (`vd-tests`).
//!
//! SCOPE (P3 Slice 1): SHARD + GATEWAY victims. The orchestrator-crash row needs the durable saga WAL
//! (D-6) and is a later slice. Drop/partition + the seed-driven breadth chaos are P3 Slice 1b.
//!
//! GREEN cells = recovery to a correct end state. The DEFERRED-D-37 cells (permanent kills that need
//! the forward re-home producer + D-3 lease liveness) are asserted HONEST — the dead-node-aware oracle
//! surfaces the orphan/park rather than false-passing a corpse — NOT recovered (that is owed, D-37).

use vd_core::entity_kind::DurabilityClass;
use vd_harness::fabric::CrashWhen;
use vd_tests::{DEST, EndState, Fault, SHARD, Scenario, assert_end_state, run_fault_scenario};

const SEED: u64 = 909;

/// The crash scenario is fully DETERMINISTIC under one seed — the R5 reproducibility cure extended to
/// crash+kill over the full cluster: the ground-truth reports AND the byte-comparable trace are
/// identical across two runs. A default-hasher `HashMap` or a wall-clock anywhere on the
/// orchestrator/gateway/shard/client path would turn this RED (so it is the standing canary that the
/// crash path stays seed-reproducible — the prerequisite for the seed-driven breadth chaos in 1b).
#[test]
fn p3_crash_matrix_is_byte_identical_under_one_seed() {
    let run = || {
        let (mut topo, entity, dead) = run_fault_scenario(
            SEED,
            Scenario {
                class: DurabilityClass::Durable,
                victim: SHARD,
                at_phase: "Demoting",
                crash_when: CrashWhen::PostInject,
                fault: Fault::CrashResurrect { after: 2 },
                standing_rehome: false,
            },
        );
        (topo.inspect_all(), topo.trace_bytes(), entity, dead)
    };
    assert_eq!(
        run(),
        run(),
        "the crash scenario is byte-identical under one seed"
    );
}

/// CRASH+RESURRECT the SOURCE while it is Demoting (post-commit): the fabric's at-least-once redelivers
/// the unacked Demote on resurrection (retry ~2 < redrive 8), so the saga reaches Done — the avatar
/// settles at the dest. `PostInject` (the undrained inbound is WIPED on crash) is the load-bearing case.
#[test]
fn p3_crash_resurrect_source_in_demoting_recovers_and_commits() {
    let (mut topo, entity, dead) = run_fault_scenario(
        SEED,
        Scenario {
            class: DurabilityClass::Durable,
            victim: SHARD,
            at_phase: "Demoting",
            crash_when: CrashWhen::PostInject,
            fault: Fault::CrashResurrect { after: 2 },
            standing_rehome: false,
        },
    );
    assert!(
        dead.is_empty(),
        "the crashed source was resurrected (live at assert time)"
    );
    assert_end_state(&mut topo, entity, &dead, EndState::SettledAt(DEST));
}

/// CRASH+RESURRECT the DEST while it is Promoting (post-commit): the Promote redelivers on resurrection,
/// the dest promotes, the saga completes. The avatar settles at the dest — no vanish, no double-hold.
#[test]
fn p3_crash_resurrect_dest_in_promoting_recovers_and_commits() {
    let (mut topo, entity, dead) = run_fault_scenario(
        SEED,
        Scenario {
            class: DurabilityClass::Durable,
            victim: DEST,
            at_phase: "Promoting",
            crash_when: CrashWhen::PostInject,
            fault: Fault::CrashResurrect { after: 2 },
            standing_rehome: false,
        },
    );
    assert!(dead.is_empty(), "the crashed dest was resurrected");
    assert_end_state(&mut topo, entity, &dead, EndState::SettledAt(DEST));
}

/// D-37 Slice 2c (CELL 2, the FORWARD-re-home cure) — the matrix REFUTED the design's "kill-DEST-pre-freeze
/// aborts to the live source" claim: the dest is NOT in the pre-freeze ack path, so killing it pre-freeze
/// does NOT trigger a pre-freeze abort. The saga sails to the orchestrator's LOCAL commit-CAS (committing
/// to the dest regardless of liveness), then would PARK in Promoting — the dead dest can never PromoteAck.
/// Now the orchestrator confirms the dest dead (D-3) and, past the abort budget, FORWARD re-homes the
/// committed entity onto a live capability-matched shard (`select_rehome_target` picks the lowest live
/// shard — here SHARD, since DEST is dead): ReHomeCommit bumps the fence past the dead dest (fence-monotone
/// — a resurrected corpse would be strictly stale), the live SHARD adopts the entity Owned from the pose.
/// The ENTITY recovers to SHARD. The dead DEST's REALM stays orphaned (the standing realm re-home is owed
/// Slice 3/4), so the cell is `EntityRecoveredRealmOrphaned{entity_at: SHARD}` until Slice 4 flips it to
/// SettledAt. (Was honest-RED `ParkedHalfOpen{DEST}` before D-37; gated on D-3 + D-6, both landed.)
#[test]
fn p3_kill_dest_pre_freeze_forward_rehomes_to_a_live_shard() {
    let (mut topo, entity, dead) = run_fault_scenario(
        SEED,
        Scenario {
            class: DurabilityClass::Durable,
            victim: DEST,
            at_phase: "Preparing",
            crash_when: CrashWhen::PostInject, // unused for Kill, but a valid cell coordinate
            fault: Fault::Kill,
            standing_rehome: false,
        },
    );
    assert_eq!(
        dead,
        [DEST].into_iter().collect(),
        "the dest stays permanently dead"
    );
    assert_end_state(
        &mut topo,
        entity,
        &dead,
        EndState::EntityRecoveredRealmOrphaned { entity_at: SHARD },
    );
}

/// D-37 Slice 0 (CELL 1, the ENTITY cure): PERMANENT KILL of the SOURCE while it is Demoting (post-commit).
/// The directory already committed to the dest; the ordered Demote would re-drive forever toward the dead
/// source. Now the orchestrator confirms the source dead (D-3) and the `scan_deadlines` re-home producer
/// injects `SourceUnreachable`, self-promoting the already-committed live dest (the corpse holds nothing).
/// The ENTITY recovers to DEST — no park, no entity orphan. The dead SOURCE shard's REALM (system-7) is
/// STILL orphaned: re-homing a dead shard's realm is the STANDING re-home owed at Slice 3/4, so the cell
/// is `EntityRecoveredRealmOrphaned` (GREEN entity, honest-RED realm) until Slice 4 flips it to SettledAt.
/// (Was fully honest-RED `ParkedHalfOpen{DEST}` before D-37; gated on D-3 lease liveness + D-6 WAL.)
#[test]
fn p3_kill_source_in_demoting_self_promotes_the_committed_dest() {
    let (mut topo, entity, dead) = run_fault_scenario(
        SEED,
        Scenario {
            class: DurabilityClass::Durable,
            victim: SHARD,
            at_phase: "Demoting",
            crash_when: CrashWhen::PostInject,
            fault: Fault::Kill,
            standing_rehome: false,
        },
    );
    assert_eq!(
        dead,
        [SHARD].into_iter().collect(),
        "the source stays permanently dead"
    );
    assert_end_state(
        &mut topo,
        entity,
        &dead,
        EndState::EntityRecoveredRealmOrphaned { entity_at: DEST },
    );
}

/// D-37 Slice 3 (CELL 3, the STANDING reaper-driven re-home): PERMANENT KILL of the SOURCE while it is
/// FREEZING (PRE-commit). The SourceFrozen ack never comes (source dead) → the freeze timeout fires
/// `abort_with_thaw` (compensators gateway-acked) → terminal Aborted → `abort_clear` clears the lock
/// (FENCE-NEUTRAL) + tombstones the saga. The CAS never ran, so authority never moved off the SOURCE —
/// now a CORPSE, the directory at a dead UNLOCKED owner with NO live saga (the pre-Slice-3
/// `DeadOwnerOrphan`). Slice 3a's machinery now recovers it: once the source is confirmed dead (D-3) and
/// its short-lease Entity record has lapsed, the expiry REAPER detects the orphan and ARMS a fresh
/// standing re-home — a saga PARKED in `ReHoming` at a LIVE capability-matched target, the key re-LOCKED.
/// CONSERVATIVE Slice-3 split: authority STAYS at the dead owner (no CAS) and NOTHING is fabricated
/// (HR1 — a pre-flush death has no recoverable pose; the dot ADOPT is owed Slice 4/D-6), so the cell is
/// `ParkedHalfOpen{SOURCE}` (a LIVE parked re-home saga, directory still at the dead owner — the
/// dead-aware oracle surfaces the exact `HeldNowhere` orphan, never a false pass). Slice 4 moves authority
/// (the `ReHomeCommit` CAS off the corpse) + reloads the state, flipping this to `EntityRecoveredRealmOrphaned`
/// then `SettledAt`. (Was honest-RED `DeadOwnerOrphan{SOURCE}` before Slice 3.)
#[test]
fn p3_kill_source_in_freezing_arms_a_standing_rehome() {
    let (mut topo, entity, dead) = run_fault_scenario(
        SEED,
        Scenario {
            class: DurabilityClass::Durable,
            victim: SHARD,
            at_phase: "Freezing",
            crash_when: CrashWhen::PostInject,
            fault: Fault::Kill,
            standing_rehome: true,
        },
    );
    assert_eq!(
        dead,
        [SHARD].into_iter().collect(),
        "the source stays permanently dead"
    );
    assert_end_state(
        &mut topo,
        entity,
        &dead,
        EndState::ParkedHalfOpen {
            authority_at: SHARD,
        },
    );
}
