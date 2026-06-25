//! P3 D-6 S5 — the ORCHESTRATOR kill-9 recovery cells (PERMANENT gate; PLAN.md P3 + HR-durability).
//!
//! "kill -9 the orchestrator MID-handoff; the rebuilt process re-hydrates its durable WAL and drives the
//! in-flight transfer to completion — never wedge, never duplicate, never vanish." This is the durable
//! half of the P3 crash matrix the connection-plane/transfer designs demand: the saga producer dies and
//! comes back, and the at-least-once fabric + the persisted saga together close the gap. It completes the
//! orchestrator-crash ROW that `p3_crash_matrix.rs` deferred ("needs the durable saga WAL (D-6)").
//!
//! BOTH transfer classes, on the SAME `rebuild_orchestrator` machinery:
//! - TRANSIENT (the short FSM path — no client/cut/freeze): a live batch caught mid-`BatchHandoff`.
//! - DURABLE (the full saga — a player logs in, walks, crosses): caught mid-choreography. This is the
//!   class that carries PLAYERS, so the bar is zero loss + AUTHORITY-UNIQUE after the orchestrator's death.
//!
//! Each class has a true control pair: rebuilt against the RETAINED store → the saga re-hydrates and the
//! subject settles at DEST; rebuilt against a FRESH store → nothing recovers, proving the recovery is
//! genuinely persistence (not in-process World survival across the kill). The full durable C1–C9
//! crash-phase sweep + the cut-warmup cell ride this same machinery and are owed next (DEFERRED.md D-6).

use vd_tests::{
    DEST, EndState, assert_end_state, assert_transient_end_state, run_orch_kill_durable,
    run_orch_kill_transient,
};

// ---- TRANSIENT class -------------------------------------------------------------------------------

/// THE D-6 headline (transient): an orchestrator kill-9 mid-`BatchHandoff` recovers from the durable WAL
/// and the transient batch completes — the rebuilt process re-hydrates the in-flight saga, re-drives it on
/// the redelivered acks, and the debris settles SINGLY at the destination with zero loss.
#[test]
fn p3_orchestrator_kill_9_recovers_an_in_flight_transient_batch() {
    let mut out = run_orch_kill_transient(0xD6_5A6A, true);
    // RECOVERY EVIDENCE: the in-flight `BatchHandoff` saga came back from the retained WAL (NOT the
    // dropped World), captured the instant the rebuilt orchestrator booted, before it re-drove anything.
    assert!(
        out.recovered_states
            .iter()
            .any(|s| s.starts_with("BatchHandoff")),
        "the rebuilt orchestrator re-hydrated the in-flight transient saga from its durable store: {:?}",
        out.recovered_states,
    );
    // END STATE: the recovered handoff completed — the debris is held at DEST by exactly one live shard,
    // zero loss (a clean adopt-before-drop, re-driven across the orchestrator's death).
    assert_transient_end_state(
        &mut out.topo,
        out.subject,
        &out.dead,
        EndState::BatchCommittedAt { node: DEST },
    );
}

/// ANTI-THEATER control (transient): the SAME kill-9, but the orchestrator is rebuilt against a FRESH
/// (empty) store. It re-hydrates NOTHING — proof the headline cell's recovery rode the RETAINED durable
/// WAL, not the in-process World surviving the kill.
#[test]
fn p3_orchestrator_kill_9_without_a_durable_store_recovers_nothing() {
    let out = run_orch_kill_transient(0xD6_5A6B, false);
    assert!(
        out.recovered_states.is_empty(),
        "a non-durable restart re-hydrates no saga (the retained store is load-bearing): {:?}",
        out.recovered_states,
    );
}

// ---- DURABLE class (the player-carrying headline) --------------------------------------------------

/// THE D-6 headline (durable PLAYER): a player is crossing a boundary when the operator kill-9s the
/// orchestrator POST-commit (`Demoting` — the directory CAS already moved authority to DEST). The rebuilt
/// orchestrator re-hydrates the in-flight saga and re-drives the demote/promote forward to `Done`: the
/// avatar settles at DEST, held by exactly one shard, with no vanish and no double-hold. This is the
/// zero-loss bar for the class that carries players.
#[test]
fn p3_orchestrator_kill_9_recovers_a_durable_player_post_commit() {
    let mut out = run_orch_kill_durable(0xD6_D002, "Demoting", true);
    // RECOVERY EVIDENCE: the post-commit saga came back from the retained WAL, not the dropped World.
    assert!(
        out.recovered_states.iter().any(|s| s.starts_with("Demoting")),
        "the rebuilt orchestrator re-hydrated the in-flight durable saga from its durable store: {:?}",
        out.recovered_states,
    );
    // END STATE: AUTHORITY-UNIQUE + settled + the avatar held once at DEST (the dead-aware oracle sees no
    // dead nodes — the orchestrator is alive again).
    assert_end_state(&mut out.topo, out.subject, &out.dead, EndState::SettledAt(DEST));
}

/// ANTI-THEATER control (durable): the SAME post-commit kill-9, but rebuilt against a FRESH (empty) store.
/// The rebuilt orchestrator re-hydrates NOTHING — proof the durable player recovery rode the RETAINED WAL.
#[test]
fn p3_orchestrator_kill_9_durable_without_a_store_recovers_nothing() {
    let out = run_orch_kill_durable(0xD6_D003, "Demoting", false);
    assert!(
        out.recovered_states.is_empty(),
        "a non-durable restart re-hydrates no durable saga (the retained store is load-bearing): {:?}",
        out.recovered_states,
    );
}

/// The PRE-commit orchestrator kill-9 (the conservative-safe counterpart, now GREEN after the abort-path
/// fence-neutrality fix): a player crossing while the saga is still `Freezing` (authority NOT yet moved —
/// the directory CAS never ran). The rehydrated saga re-arms `since=0`, so the FIRST post-restart
/// `scan_deadlines` fires the (large) abort deadline → `abort_with_thaw` thaws the source → terminal
/// `Aborted`. The player stays ALIVE at SOURCE, authoritative, no loss. This is the documented operational
/// semantic: an orchestrator restart RE-DRIVES post-commit transfers forward to DEST but ABORTS in-flight
/// pre-commit transfers back to a live source (fail-safe — nothing was committed anywhere). The fence-
/// neutral `abort_clear` (this slice) is what makes the end state CLEAN: the surviving source's held fence
/// equals the directory's recorded fence (FENCE-9, asserted inside `AbortedToSource`), so the player is
/// neither stranded nor logout-wedged — the divergence that originally deferred this cell is fixed.
#[test]
fn p3_orchestrator_kill_9_pre_commit_aborts_a_durable_player_back_to_the_live_source() {
    let mut out = run_orch_kill_durable(0xD6_D004, "Freezing", true);
    // RECOVERY EVIDENCE: the pre-commit saga came back from the retained WAL (then the abort deadline fires).
    assert!(
        out.recovered_states.iter().any(|s| s.starts_with("Freezing")),
        "the rebuilt orchestrator re-hydrated the pre-commit durable saga from its durable store: {:?}",
        out.recovered_states,
    );
    // END STATE: aborted to the LIVE source, no loss — and (the fix's proof) NO fence divergence: the
    // `AbortedToSource` asserter now also runs AUTHORITY-UNIQUE (FENCE-9), which would RED on a bumped fence.
    assert_end_state(&mut out.topo, out.subject, &out.dead, EndState::AbortedToSource);
}
