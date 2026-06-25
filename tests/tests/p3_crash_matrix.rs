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
        },
    );
    assert!(dead.is_empty(), "the crashed dest was resurrected");
    assert_end_state(&mut topo, entity, &dead, EndState::SettledAt(DEST));
}

/// DEFERRED D-37 (honest RED) — the EMPIRICAL correction to the design's "kill-DEST-pre-freeze aborts
/// to the live source" claim, which the matrix REFUTED: the dest is NOT in the pre-freeze ack path
/// (Prepared/CutConfirmed/SourceFrozen come from the GATEWAY + SOURCE), so killing it pre-freeze does
/// NOT trigger a pre-freeze abort. The saga sails through to the orchestrator's LOCAL commit-CAS
/// (which commits to the dest regardless of its liveness), then PARKS in Promoting — the dead dest can
/// never ack PromoteAck. So the directory committed to a dead dest = `ParkedHalfOpen{DEST}`. There is
/// NO clean permanent-kill-to-LIVE-source cell (killing the source/gateway makes THEM dead); the
/// abort-to-live-source path is a NON-kill abort (a spatial rejection / a transient-fault pre-freeze
/// timeout, P3 Slice 1b). The dead-aware oracle surfaces this honestly; the forward re-home is D-37.
#[test]
fn p3_kill_dest_pre_freeze_commits_to_the_dead_dest_then_parks() {
    let (mut topo, entity, dead) = run_fault_scenario(
        SEED,
        Scenario {
            class: DurabilityClass::Durable,
            victim: DEST,
            at_phase: "Preparing",
            crash_when: CrashWhen::PostInject, // unused for Kill, but a valid cell coordinate
            fault: Fault::Kill,
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
        EndState::ParkedHalfOpen { authority_at: DEST },
    );
}

/// DEFERRED D-37 (honest RED): PERMANENT KILL of the SOURCE while it is Demoting (post-commit). The
/// directory already committed to the dest; the Demote re-drives forever toward the dead source and the
/// saga PARKS. The dead-node-aware oracle EXCLUDES the dead source's corpse claim and surfaces the
/// honest orphan (NOT a false-pass) — proving the foundation makes the gap observable. Recovery (a
/// forward re-home) is owed at D-37 (gated on D-3 lease liveness + D-6 WAL).
#[test]
fn p3_kill_source_in_demoting_parks_and_the_dead_aware_oracle_surfaces_it() {
    let (mut topo, entity, dead) = run_fault_scenario(
        SEED,
        Scenario {
            class: DurabilityClass::Durable,
            victim: SHARD,
            at_phase: "Demoting",
            crash_when: CrashWhen::PostInject,
            fault: Fault::Kill,
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
        EndState::ParkedHalfOpen { authority_at: DEST },
    );
}

/// DEFERRED D-37 (honest RED): PERMANENT KILL of the SOURCE while it is FREEZING (PRE-commit). The
/// SourceFrozen ack never comes (source dead) → the freeze timeout fires `abort_with_thaw` (the
/// compensators are gateway-acked) → terminal Aborted → `abort_clear` clears the lock (FENCE-NEUTRAL —
/// no bump) + tombstones the saga. But the CAS never ran, so authority never moved off the SOURCE —
/// which is now a CORPSE: `DeadOwnerOrphan{SOURCE}` (saga terminal, lock clear, directory at a dead
/// owner). The dead-aware oracle surfaces the exact orphan. (This is why "abort to the live source"
/// is NOT reachable by a permanent SOURCE kill — the aborted-to owner is dead.)
#[test]
fn p3_kill_source_in_freezing_tombstones_to_a_dead_owner_orphan() {
    let (mut topo, entity, dead) = run_fault_scenario(
        SEED,
        Scenario {
            class: DurabilityClass::Durable,
            victim: SHARD,
            at_phase: "Freezing",
            crash_when: CrashWhen::PostInject,
            fault: Fault::Kill,
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
        EndState::DeadOwnerOrphan {
            authority_at: SHARD,
        },
    );
}
