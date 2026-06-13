//! P2 transfer definition-of-done gates (PLAN.md P2). PERMANENT — no later phase may
//! regress them.
//!
//! The D-28 cross-cut INPUT-CONSERVATION gate: a player's input survives a REAL cross-shard
//! authority handoff exactly once, in order, with the CUT_MARKER threaded end-to-end (client →
//! saga → gateway → dest). It is the FIRST proof of the core transfer thesis, and the first to
//! JOIN the two crate halves — the gateway's buffer/drain (connection-plane) and the dest's
//! `OpenInputSlot`/`apply_input` (sim) — in one scenario over the real fabric, driven by the
//! REAL saga producer (never hand-fed acks).
//!
//! SCOPE (D-28, focused): the transfer commits authority to the dest (directory CAS lands) and
//! the saga PARKS in `Demoting` — the demote/release tail (source release, authority-settled)
//! is 1c.8+ (DEFERRED D-28/D-29). This gate proves INPUT conservation across the cut, not the
//! final authority settle.

use vd_core::entity_kind::DurabilityClass;
use vd_core::pose::RealmId;
use vd_core::{AccountId, EntityId, NodeId, SessionId, TransferId};
use vd_harness::client::ScriptedClient;
use vd_harness::fabric::FaultFabric;
use vd_harness::oracle::verify_input_conservation;
use vd_harness::topology::{InspectReport, Topology};
use vd_sim::saga::SagaCtx;
use vd_tests::{
    DEST, ORCH, SHARD, gateway_buffered_count, live_sagas, p1_client, p2_cluster, read_subject,
    saga_states, trigger_transfer, walk_forward,
};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey};

const CLIENT: NodeId = NodeId(100);

fn report(reports: &[(NodeId, InspectReport)], id: NodeId) -> &InspectReport {
    &reports
        .iter()
        .find(|(n, _)| *n == id)
        .expect("node present")
        .1
}

fn with_client<R>(topo: &mut Topology, f: impl FnOnce(&mut ScriptedClient) -> R) -> R {
    let node = topo.node_mut(CLIENT).expect("client present");
    let client = node
        .as_any_mut()
        .expect("clients opt into downcasting")
        .downcast_mut::<ScriptedClient>()
        .expect("the node is a ScriptedClient");
    f(client)
}

/// Step (at most `max` ticks) until `cond` holds — a BOUNDED, deterministic poll on
/// deterministic state. Robust to unrelated timing shifts (unlike a hardcoded absolute tick);
/// the determinism sibling proves two runs poll byte-identically. Panics if never reached.
fn step_until(topo: &mut Topology, max: u64, mut cond: impl FnMut(&mut Topology) -> bool) {
    for _ in 0..max {
        topo.step();
        if cond(topo) {
            return;
        }
    }
    panic!("condition not reached within {max} ticks");
}

/// Drive ONE full cross-shard transfer of a logged-in player from [`SHARD`] to [`DEST`] over
/// the real fabric, returning the quiesced topology plus the session, transferred entity, and
/// the marker seq the run threaded (all observed LIVE, never hardcoded).
fn run_cut_transfer(fabric: &FaultFabric) -> (Topology, SessionId, EntityId, u64) {
    let mut topo = p2_cluster(fabric, 8);
    topo.add_node(Box::new(p1_client(
        fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    // WARMUP: the player logs in and walks; the SOURCE grants its avatar and the DEST wins its
    // own realm lease (so `OpenInputSlot` will not be deferred). The client is emitting a
    // contiguous seq stream that the source applies.
    step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        !report(&r, SHARD).held_entities.is_empty()
            && report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });

    // TRIGGER the REAL transfer of the avatar. The CAS expectation is the DIRECTORY record's
    // fence (read live), the subject the directory `Entity` key.
    let (session, entity, fence) = read_subject(&mut topo);
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: TransferId(1),
            session,
            subject: DirectoryKey::Entity(entity),
            expected_fence: fence,
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Durable,
            needs_provision: false,
        },
    );

    // The saga reaches `Freezing`: `FreezeSource` is in flight and the cut installs on the
    // gateway next tick. The client has already stamped the CUT_MARKER (in response to the
    // gateway's `RequestCut`) and PAUSED, so no seq > marker has leaked to the source.
    step_until(&mut topo, 40, |t| {
        saga_states(t).iter().any(|s| s.starts_with("Freezing"))
    });

    // RESUME now: the next inputs (seq > marker) reach the gateway AFTER the cut is installed,
    // so they BUFFER and are drained to the dest at commit — the buffer/drain path under test.
    with_client(&mut topo, ScriptedClient::resume_input);

    // The saga commits (directory CAS to DEST) and parks in `Demoting` (the demote/release
    // tail has no producer at 1c.7 — D-28 scope).
    step_until(&mut topo, 40, |t| {
        saga_states(t).iter().any(|s| s.starts_with("Demoting"))
    });

    // QUIESCE: stop emitting and let every in-flight post-marker input settle at the dest.
    with_client(&mut topo, ScriptedClient::pause_input);
    for _ in 0..8 {
        topo.step();
    }

    let marker = with_client(&mut topo, |c| c.marker_seq()).expect("the client stamped a marker");
    (topo, session, entity, marker)
}

#[test]
fn p2_dod_cross_cut_input_is_conserved_exactly_once() {
    let fabric = FaultFabric::new(909, 2);
    let (mut topo, session, entity, m) = run_cut_transfer(&fabric);

    // `inspect_all` yields nodes in NodeId order, so SOURCE (NodeId 3) precedes DEST (NodeId 4)
    // — a BINDING fixture invariant: the conservation oracle concatenates per-session applied
    // order in report-slice order WITHOUT sorting, so a clean cut stays strictly increasing
    // only if the source's `1..=M` precedes the dest's `M+1..`.
    let reports = topo.inspect_all();
    let src = report(&reports, SHARD);
    let dst = report(&reports, DEST);
    let orch = report(&reports, ORCH);

    // (2) EXACTLY-ONCE ACROSS THE CUT — the joined-halves proof. A lost drained input ⇒
    // Unaccounted; a double-apply across the seam ⇒ AppliedTwice; an out-of-order/replay apply
    // ⇒ NonMonotonicApply; a phantom ⇒ Phantom.
    verify_input_conservation(&reports).expect("cross-cut INPUT-CONSERVATION holds");

    // (3) THE PARTITION IS REAL (cut-specific, anti-vacuous): source applied only `<= marker`,
    // dest applied a NON-EMPTY batch all `> marker`, resuming exactly at `marker + 1`. (Holds by
    // construction because the client paused across the freeze window — no post-marker leak.)
    assert!(
        src.applied_inputs.iter().all(|(_, s)| *s <= m),
        "source applied only seqs <= the marker {m}: {:?}",
        src.applied_inputs,
    );
    assert!(
        dst.applied_inputs.iter().all(|(_, s)| *s > m),
        "dest applied only seqs > the marker {m}: {:?}",
        dst.applied_inputs,
    );
    assert!(
        !dst.applied_inputs.is_empty(),
        "the dest applied a NON-EMPTY post-marker batch",
    );
    assert_eq!(
        dst.applied_inputs.iter().map(|(_, s)| *s).min(),
        Some(m + 1),
        "the dest resumes exactly at marker+1 (no gap, no off-by-one)",
    );
    // The BUFFER/DRAIN path — the headline 1c.7 deliverable — was actually exercised: at least
    // one seq > marker hit the gateway AFTER the cut installed and was buffered (not merely
    // direct-forwarded to the dest after the route swapped). Pinned to a real observable so a
    // future tick-ordering shift that silently routes M+1 direct turns this RED, not green.
    assert!(
        gateway_buffered_count(&mut topo) >= 1,
        "the cut buffer held >= 1 frame — the gateway buffer/drain path was exercised",
    );

    // (4) THE MARKER ITSELF is applied at the SOURCE (the every-sent-accounted anchor: at marker
    // arrival no cut is installed yet, so it is forwarded-and-applied at source).
    assert!(
        src.applied_inputs.contains(&(session, m)),
        "the marker input (seq {m}) is applied at the source",
    );

    // (5) THE MARKER THREADED end-to-end: the resume watermark the gateway EMITTED in
    // `OpenInputSlot` (latched on the dest) equals the client's OWN marker seq — one live value,
    // no hardcoded literal on either side.
    assert_eq!(
        dst.dest_resume_seq,
        Some(m),
        "the dest seeded its resume watermark to the client's own marker seq",
    );

    // (6) NO SILENT WINDOW LOSS: the input logs saw the whole run (else conservation is a lie).
    assert_eq!(src.input_window_evictions, 0, "source log lost nothing");
    assert_eq!(dst.input_window_evictions, 0, "dest log lost nothing");

    // (7) THE COMMIT LANDED authority at DEST in the directory, and the saga PARKED post-commit
    // (the demote/release tail is 1c.8+; asserting Done here would be impossible — D-28).
    let entity_owner = orch
        .directory
        .iter()
        .find_map(|(k, r)| (*k == DirectoryKey::Entity(entity)).then_some(r.authority))
        .expect("the transferred avatar is recorded");
    assert_eq!(
        entity_owner,
        AuthorityRef::Shard(DEST),
        "authority committed to the dest shard",
    );
    assert_eq!(
        live_sagas(&mut topo),
        1,
        "the saga parks in Demoting after the commit (demote tail is 1c.8+)",
    );
}

/// The transfer run is fully deterministic under one seed — the standing in-process replay gate
/// (cross-process parity is P3 chaos scope). It compares not just `inspect_all` ground truth but
/// the harness TRACE (the purpose-built byte-comparable per-node step/sent/drain artifact) AND
/// the saga-FSM phase the choreography polls on — so a divergence in the transfer machinery PAST
/// the commit point (the saga's post-CAS phases and the gateway's route/buffer mutate NO
/// directory state, hence are invisible to `inspect_all` alone) cannot hide.
#[test]
fn p2_dod_cross_cut_transfer_is_byte_identical_under_same_seed() {
    let run = || {
        let (mut topo, _, _, marker) = run_cut_transfer(&FaultFabric::new(909, 2));
        let saga = saga_states(&mut topo);
        let live = live_sagas(&mut topo);
        // The gateway's cut-buffer fill count, compared DIRECTLY (not merely inferred from the
        // trace's per-tick send deferral) — so a divergence in the gateway's route/buffer
        // decision cannot hide behind an identical trace+reports+saga.
        let buffered = gateway_buffered_count(&mut topo);
        let reports = topo.inspect_all();
        let trace = topo.trace_bytes();
        (reports, trace, saga, live, buffered, marker)
    };
    assert_eq!(
        run(),
        run(),
        "identical seed ⇒ identical ground truth, trace, saga FSM phase, buffer fill, and marker",
    );
}

// ---------------------------------------------------------------------------------------------
// NEGATIVE CONTROLS — verified by LOCAL mutation (run each, confirm RED, revert), kept here so
// the gate's sensitivity is auditable. The oracle alone is blind to a clean-but-wrong partition,
// so assertions (3)/(5) are load-bearing beside it.
//
// VERIFIED to turn a DIFFERENT named assertion RED in THIS e2e gate:
//   (a) LOST DRAINED INPUT  — `for input_bytes in buffered.into_iter().skip(1)` in apply_commit
//                              ⇒ that seq is sent-but-applied-nowhere ⇒ verify_input_conservation
//                                 returns Unaccounted (assertion 2). [confirmed]
//   (c) MARKER MIS-THREAD   — emit `OpenInputSlot{ resume_from_seq: marker_seq + 5 }`
//                              ⇒ dest seeds M+5, rejects the drained batch ⇒ dest empty
//                                 (assertion 3); dest_resume_seq would also be Some(M+5) ≠ Some(M)
//                                 (assertion 5). [confirmed]
//   (e) NO PAUSE (leak)     — drop the client's pause-after-marker ⇒ seq > M forwarded to the
//                              source before the cut installs ⇒ source applied > M (assertion 3,
//                              the cut-correctness check the oracle CANNOT make). [confirmed]
//
// COVERED AT THE UNIT LEVEL (moot in THIS e2e gate, by design — documented honestly):
//   (b) DOUBLE-APPLY        — the dest's `last_applied_seq` dedup structurally rejects a replayed
//                              drained frame, so a naive double-apply cannot be injected here; the
//                              oracle's `AppliedTwice` arm is unit-tested in `harness::oracle`.
//   (d) OUT-OF-ORDER DRAIN  — the e2e freeze→commit window buffers a SINGLE frame (later
//                              post-marker inputs forward directly to the dest after the route
//                              swaps), so reversing a 1-element buffer is a no-op. Multi-frame
//                              drain ORDER is gated by the connection-plane unit test
//                              `commit_opens_the_dest_slot_then_drains_the_buffer_in_seq_order`.
// ---------------------------------------------------------------------------------------------
