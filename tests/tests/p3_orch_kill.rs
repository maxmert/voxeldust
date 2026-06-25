//! P3 D-6 S5 — the ORCHESTRATOR kill-9 recovery cells (PERMANENT gate; PLAN.md P3 + HR-durability).
//!
//! "kill -9 the orchestrator MID-handoff; the rebuilt process re-hydrates its durable WAL and drives the
//! in-flight transfer to completion — never wedge, never duplicate, never vanish." This is the durable
//! half of the P3 crash matrix the connection-plane/transfer designs demand: the saga producer dies and
//! comes back, and the at-least-once fabric + the persisted saga together close the gap.
//!
//! S5 SCOPE: a live TRANSIENT batch (the short FSM path — no client/cut/freeze) caught mid-`BatchHandoff`.
//! Two cells, a true control pair: (1) rebuilt against the RETAINED store → the saga re-hydrates and the
//! debris settles at DEST with zero loss; (2) rebuilt against a FRESH store → nothing recovers, proving
//! the recovery is genuinely persistence (not in-process World survival across the kill). The full durable
//! C1–C9 player matrix + the cut-warmup durable cell ride this same `rebuild_orchestrator` machinery and
//! are owed next (DEFERRED.md D-6).

use vd_tests::{DEST, EndState, assert_transient_end_state, run_orch_kill_transient};

/// THE D-6 headline: an orchestrator kill-9 mid-`BatchHandoff` recovers from the durable WAL and the
/// transient batch completes — the rebuilt process re-hydrates the in-flight saga, re-drives it on the
/// redelivered acks, and the debris settles SINGLY at the destination with zero loss.
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
        out.debris,
        &out.dead,
        EndState::BatchCommittedAt { node: DEST },
    );
}

/// ANTI-THEATER control: the SAME kill-9, but the orchestrator is rebuilt against a FRESH (empty) store.
/// It re-hydrates NOTHING — proof the headline cell's recovery rode the RETAINED durable WAL, not the
/// in-process World surviving the kill. (Without this, a harness that quietly kept the old World would
/// false-green the headline cell.)
#[test]
fn p3_orchestrator_kill_9_without_a_durable_store_recovers_nothing() {
    let out = run_orch_kill_transient(0xD6_5A6B, false);
    assert!(
        out.recovered_states.is_empty(),
        "a non-durable restart re-hydrates no saga (the retained store is load-bearing): {:?}",
        out.recovered_states,
    );
}
