//! Slice 3g — THE CORE CROSSING END-TO-END COMPOSITION PROOF (task #133 capstone).
//!
//! This gate proves the GEOMETRIC transfer-TRIGGER (`evaluate_realm_boundaries` / `should_commit`)
//! COMPOSES with THE transfer saga end-to-end, over the real cluster fabric, with NOTHING hand-fed.
//!
//! ANTI-VACUITY INVARIANT (enforced by construction — a grep of this file must find ZERO
//! `trigger_transfer` / `start_transfer` / `start_crossing_transfer`): the crossing is driven ONLY by
//! planting a geometric boundary (`plant_one_crossing_shell`) that the source's dwell detector commits
//! against. The request becomes a saga AUTONOMOUSLY — the shard→orch→shard route
//! (`CrossingRequest → handle_crossing_request → start_transfer → the choreography`) closes on its own,
//! or these tests could not pass. The composition proof is `report(ORCH).crossings_started >= 1`: the
//! source's `CrossingRequest` resolved its three directory heads and BECAME a real transfer saga.
//!
//! SCOPE (per the vetted plan): Test 1 durable happy-path, Test 2 dest render, Test 4 transient (HR2
//! second class), Test 5 determinism, Test 3 the pre-CAS ABORT leg, and the Tier-1 CRASH leg (an
//! orchestrator World-REBUILD between the abort-emit and the source ack — NOT a process SIGKILL; the
//! honest tier is in the test names). 3f-D4 (unresolved-dest re-drive) stays OUT OF SCOPE.
//!
//! MECHANISM-Y COMPOSITION: the abort/crash legs prove the crash-durable crossing-abort (`pending_abort_
//! replies`) COMPOSES at cluster tier. The abort is forced PRE-CAS by the gateway's one-shot
//! `reject_next_prepare` lever (`arm_gateway_reject`) — the gateway is the durable Prepare decider, so a
//! `Prepared{ Rejected(Spatial(Obstructed)) }` drives the saga's `abort_from_pre_freeze` (no `ThawSource`,
//! no `IssueCommitCas`), and the directory head NEVER moves off the source (the pre-CAS proof). The dest
//! stub never sees the Prepare, so `stub.rs` is untouched; no wire type is new (all reject arms exist).
//!
//! GEOMETRY HONESTY: the shell is centered at the dot's SPAWN offset, so the dot is born-inside and
//! commits after `n_entry` dwell ticks. This proves TRIGGER→TRANSFER composition; the pixel-visible
//! PHYSICAL traversal (a dot that walks ACROSS a boundary) remains the D-15/D-30 visual owed line.

use vd_core::entity_kind::EntityKind;
use vd_core::glam::DVec3;
use vd_core::pose::FrameRef;
use vd_core::{AccountId, EntityId, Fence, NodeId, TickId};
use vd_harness::client::ScriptedClient;
use vd_harness::fabric::{FaultFabric, LinkPolicy};
use vd_harness::oracle::{RenderSample, verify_authority_settled, verify_authority_unique};
use vd_harness::topology::{InspectReport, Topology};
use vd_sim::io::mem::MemStore;
use vd_tests::{
    DEST, ORCH, SHARD, arm_gateway_reject, dest_stub_config, live_sagas, p1_client, p2_cluster,
    p2_cluster_durable_orch, plant_one_crossing_shell, read_subject, realm_fence,
    rebuild_orchestrator, saga_states, seed_held_transient, stub_config, walk_forward,
};
use vd_wire::channels::SubId;
use vd_wire::intershard::crossing_transfer_id;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey};
use vd_wire::seams::transfer_control::{PrepareReject, SpatialReject};

const CLIENT: NodeId = NodeId(100);

/// A large finite render cursor so the client's `rendered` clamps to the FRESHEST delivered pose every
/// tick — the actual last-delivered crossing pose, sampled deterministically (mirrors p2_transfer_gates).
const TRACE_CURSOR: f64 = 1.0e9;

/// One captured tick of the subject's composited render: its sample (`None` ⇒ rendered nowhere) and the
/// subs holding a track for it. Built from DELIVERED BYTES only — never node internals.
#[derive(Clone, Debug, PartialEq)]
struct CapturedTick {
    tick: TickId,
    sample: Option<RenderSample>,
    subs_holding: Vec<SubId>,
}

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

/// Capture the subject's composited render sample + held-subs from the client's REAL `DeliveredView` this
/// tick (mirrors p2_transfer_gates::capture_subject). The subject is the client's OWN entity.
fn capture_subject(topo: &mut Topology) -> CapturedTick {
    let tick = topo.tick();
    with_client(topo, |c| {
        let view = &c.delivered_view;
        let sample = view.own_entity().and_then(|own| {
            view.rendered(TRACE_CURSOR)
                .into_iter()
                .find(|(e, _, _)| *e == own)
                .map(|(_, sub, pose)| RenderSample {
                    sub,
                    frame: pose.frame,
                    raw_pos: pose.pos,
                    world_pos: view.world_pos(&pose, TRACE_CURSOR),
                    orient: pose.orient,
                })
        });
        let subs_holding = view
            .own_entity()
            .map(|own| view.subs_holding(own))
            .unwrap_or_default();
        CapturedTick {
            tick,
            sample,
            subs_holding,
        }
    })
}

/// Step at most `max` ticks until `cond` holds, running `observe` after EVERY step (the per-tick render
/// sampler; `&mut |_| {}` for the gates that capture no trace). BOUNDED + deterministic. Panics if never.
fn step_until(
    topo: &mut Topology,
    max: u64,
    observe: &mut dyn FnMut(&mut Topology),
    mut cond: impl FnMut(&mut Topology) -> bool,
) {
    for _ in 0..max {
        topo.step();
        observe(topo);
        if cond(topo) {
            return;
        }
    }
    panic!("crossing-e2e condition not reached within {max} ticks");
}

/// Drive ONE AUTONOMOUS durable crossing: spin the cluster + a walking client, warm up, HARD-ASSERT the
/// M2 gate (the DEST holds the shell's `to_realm`) + the SHARD holds the dot, PLANT the crossing shell
/// (NO `trigger_transfer`), then let the geometric dwell fire the crossing and the saga self-drive to
/// terminal. Returns the quiesced topology + the transferred entity. `observe` runs after every step (the
/// render sampler; `&mut |_| {}` for the authority-only gates).
///
/// The drive mirrors `run_cut_transfer`'s step-to-terminal sequence, but with the trigger PLANTED
/// (geometric) rather than the saga hand-started. The client keeps walking; it AUTONOMOUSLY stamps the
/// CUT_MARKER on the saga's `RequestCut` (no test-driven pause/resume — the crossing-e2e asserts
/// authority + render, not input conservation, which is D-28's gate).
fn run_autonomous_crossing(
    fabric: &FaultFabric,
    observe: &mut dyn FnMut(&mut Topology),
) -> (Topology, EntityId) {
    let mut topo = p2_cluster(fabric, 8);
    topo.add_node(Box::new(p1_client(
        fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    // WARMUP: the player logs in + walks; the SOURCE grants its avatar and the DEST wins its realm lease.
    step_until(&mut topo, 80, observe, |t| {
        let r = t.inspect_all();
        !report(&r, SHARD).held_entities.is_empty()
            && report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == dest_stub_config().realm)
    });

    // M2 GATE (hard-assert, anti-vacuity): the DEST holds the exact realm the planted shell hands
    // authority to — else `handle_crossing_request` would silently count `crossing_unresolved`.
    let warm = topo.inspect_all();
    assert!(
        report(&warm, DEST)
            .held_realms
            .iter()
            .any(|(realm, _)| *realm == dest_stub_config().realm),
        "M2: the DEST holds the shell's to_realm {:?} — the crossing can resolve its dest head",
        dest_stub_config().realm,
    );
    let (subject, _) = *report(&warm, SHARD)
        .held_entities
        .first()
        .expect("the SHARD holds the walking avatar before the crossing is planted");

    // ARM the geometric trigger. From here the loop is AUTONOMOUS — no `trigger_transfer` anywhere.
    plant_one_crossing_shell(&mut topo);

    // The dwell fires an Inward commit after `n_entry` ticks → the shard emits a `CrossingRequest` →
    // the orchestrator resolves the three heads and STARTS the saga. Wait for the saga to reach Freezing
    // (it is genuinely live — anti-vacuity), then let it self-drive to terminal (`live_sagas == 0`).
    step_until(&mut topo, 40, observe, |t| {
        saga_states(t).iter().any(|s| s.starts_with("Freezing"))
    });
    step_until(&mut topo, 60, observe, |t| live_sagas(t) == 0);

    // QUIESCE: stop emitting and let the ordered demote/release tail land (the source dot self-fences to a
    // retained Ghost; the dest promotes to Owned) — mirror `run_cut_transfer`'s settle.
    with_client(&mut topo, ScriptedClient::pause_input);
    for _ in 0..8 {
        topo.step();
        observe(&mut topo);
    }

    // STEADY-STATE GUARD: the SOURCE reports nothing authoritative for the subject AND the DEST holds it.
    step_until(&mut topo, 40, observe, |t| {
        let r = t.inspect_all();
        let src = report(&r, SHARD);
        let dst = report(&r, DEST);
        let source_clear = !src.held_entities.iter().any(|(e, _)| *e == subject)
            && !src.pending_entities.contains(&subject)
            && !src.departing_entities.contains(&subject);
        let dest_holds = dst.held_entities.iter().any(|(e, _)| *e == subject);
        source_clear && dest_holds
    });

    (topo, subject)
}

/// TEST 1 — THE HEADLINE: a durable dot crosses a PLANTED boundary and the geometric trigger drives the
/// full transfer saga end-to-end (NO hand-feed). The composition proof is `crossings_started >= 1`.
#[test]
fn crossing_e2e_durable_dot_crosses_a_planted_boundary() {
    let fabric = FaultFabric::new(909, 2);
    let (mut topo, subject) = run_autonomous_crossing(&fabric, &mut |_| {});

    let reports = topo.inspect_all();
    let orch = report(&reports, ORCH);
    let src = report(&reports, SHARD);
    let dst = report(&reports, DEST);

    // (1) THE COMPOSITION PROOF (anti-vacuity, causal): the source's `CrossingRequest` BECAME a saga.
    // `crossings_started` bumps ONLY inside `handle_crossing_request`'s (Some,Some,Some) non-live arm,
    // i.e. only when a `CrossingRequest` actually became a `start_transfer` — the geometric trigger drove
    // the transfer, never a hand-fed `trigger_transfer` (which this file contains ZERO of).
    assert!(
        orch.crossings_started >= 1,
        "the geometric CrossingRequest became a transfer saga (crossings_started >= 1): {}",
        orch.crossings_started,
    );
    // Leg 1 fired at the source: the dwell detector emitted a `CrossingRequest`.
    assert!(
        src.crossings_requested >= 1,
        "the source's dwell detector emitted a CrossingRequest (crossings_requested >= 1): {}",
        src.crossings_requested,
    );

    // (2) THE DIRECTORY HEAD FLIPPED to the dest — authority committed (the CAS landed).
    let record = orch
        .directory
        .iter()
        .find_map(|(k, r)| (*k == DirectoryKey::Entity(subject)).then_some(*r))
        .expect("the crossed avatar is recorded in the directory");
    assert_eq!(
        record.authority,
        AuthorityRef::Shard(DEST),
        "the directory head flipped to the DEST shard (the crossing committed authority)",
    );

    // (3) SOURCE Ghost / DEST Owned: the source retains the avatar as a non-simulating Ghost (or has
    // dropped it), the dest holds it authoritatively.
    assert!(
        !src.held_entities.iter().any(|(e, _)| *e == subject),
        "the SOURCE no longer holds the crossed avatar authoritatively (it self-fenced to a Ghost)",
    );
    assert!(
        dst.held_entities.iter().any(|(e, _)| *e == subject),
        "the DEST holds the crossed avatar (Owned)",
    );

    // (4) POSITIVE LATCH CLEAR: the source's `RequestInFlight` crossing latch cleared on the saga's
    // Demote terminal, and no latch stands for the subject.
    assert!(
        src.crossing_latches_cleared >= 1,
        "the source's crossing latch cleared on the commit terminal (crossing_latches_cleared >= 1): {}",
        src.crossing_latches_cleared,
    );
    assert!(
        !src.in_flight_latches.contains(&subject),
        "no standing RequestInFlight latch for the crossed subject after commit: {:?}",
        src.in_flight_latches,
    );

    // (5) THE HR2 AUTHORITY ORACLES hold after the full crossing.
    verify_authority_unique(&reports)
        .expect("exactly one holder of the crossed entity, fence-matching the directory");
    verify_authority_settled(&reports)
        .expect("no lingering pending/departing anywhere after the crossing tail");
}

/// TEST 2 — THE DEST RENDERS THE CROSSED POSE (SEPARATE from Test 1's authority gate — HR5(d): never
/// `&&` render with authority). The dest renders the CROSSED source-realm frame (seed 7), NOT its own
/// seed-8 origin default — the load-bearing discriminator a dropped crossing would fail (mirror
/// p2_transfer_gates:719).
#[test]
fn crossing_e2e_renders_at_dest() {
    let fabric = FaultFabric::new(909, 2);
    let mut caps: Vec<CapturedTick> = Vec::new();
    let (_topo, _subject) =
        run_autonomous_crossing(&fabric, &mut |t| caps.push(capture_subject(t)));

    let rendered: Vec<RenderSample> = caps.iter().filter_map(|c| c.sample).collect();
    let dest_sample = *rendered
        .last()
        .expect("the subject renders at some tick across the crossing");

    // The dest renders the CROSSED pose re-expressed into the DEST realm's frame (`SystemSpace{seed:8}` —
    // post frame-rebinding, `transfer_frame` identity through P3). The drop discriminator is now the
    // non-origin `world_pos` below (the walked crossed pose, NOT the dest's ZERO adopt-default); this frame
    // assertion CONFIRMS the rebinding placed the render pose in the right realm.
    assert_eq!(
        dest_sample.frame,
        FrameRef::SystemSpace { system_seed: 8 },
        "the dest renders the crossed pose re-expressed into the DEST realm's frame (seed 8): {dest_sample:?}",
    );
    assert!(
        dest_sample.world_pos.is_finite(),
        "the rendered crossed world pose is finite (sanitized at ingress): {dest_sample:?}",
    );
    assert_ne!(
        dest_sample.world_pos,
        DVec3::ZERO,
        "the rendered pose is the walked crossed pose, not the origin-adopt default",
    );
}

/// TEST 4 — THE TRANSIENT (HR2 second-class) AUTONOMOUS CROSSING, to the boundary the cluster harness
/// actually closes. An OWNED `Held` Debris in the band → the geometric dwell fires the Transient fan-out
/// (`TransientCrossingRequest`) → the orchestrator AUTO-GRANTS it (`TransientCrossingGrant`) — proving the
/// SAME geometric trigger machinery fans a SECOND durability class out (HR2) with NO hand-fed batch.
///
/// SCOPE: this test asserts the AUTONOMOUS transient loop through Leg 1 + the auto-grant (the HR2
/// second-class proof reachable in the `p2_cluster` scaffold). The DOWNSTREAM batch handoff — the grant
/// starting a `BatchHandoff` saga so the dest promotes `Arriving→Held` (`batch_goes` non-empty, dest
/// `owned_transients` populated) — was an owed route this test originally FLAGGED (it was missing, not
/// hand-fed to force green); it is now LANDED (D-43 #9: `handle_transient_crossing_request` starts the
/// saga at grant, keyed on `batch`), and the full end-to-end transient composition is proven in
/// `crates/harness/src/topology.rs::a_transient_crossing_composes_end_to_end_dest_owns_and_go_token_recorded`
/// (a real 3-node topology where the dest ends `Held` with a recorded go-token). This test stays scoped
/// to Leg-1 + grant because the `p2_cluster` scaffold does not drive the multi-shard batch handoff.
#[test]
fn crossing_e2e_transient_crosses_via_batch_grant() {
    let fabric = FaultFabric::new(909, 2);
    let mut topo = p2_cluster(&fabric, 8);

    // WARMUP: BOTH shards win their realm leases (the source to cross from, the dest to adopt into).
    step_until(&mut topo, 80, &mut |_| {}, |t| {
        let r = t.inspect_all();
        report(&r, SHARD)
            .held_realms
            .iter()
            .any(|(realm, _)| *realm == stub_config().realm)
            && report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == dest_stub_config().realm)
    });

    // Seed an OWNED Held Debris transient INSIDE the band (near the shell center at spawn), then ARM the
    // trigger. The transient dwell fires `TransientCrossingRequest` AUTONOMOUSLY (no test-seeded Crossing
    // status, no hand-fed batch `trigger_transfer`).
    let src_fence = realm_fence(&mut topo, stub_config().realm);
    let debris = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 1, 0);
    seed_held_transient(&mut topo, debris, src_fence, DVec3::new(50.0, 0.0, 0.0));
    plant_one_crossing_shell(&mut topo);

    // Drive the trigger + the auto-grant round-trip (bounded, deterministic).
    step_until(&mut topo, 40, &mut |_| {}, |t| {
        let r = t.inspect_all();
        report(&r, ORCH).transient_crossings_granted >= 1
    });

    let reports = topo.inspect_all();
    let src = report(&reports, SHARD);
    let orch = report(&reports, ORCH);

    // (1) LEG 1: the transient dwell emitted a `TransientCrossingRequest` (the HR2 second-class Leg-1).
    assert!(
        src.transient_crossings_requested >= 1,
        "the transient dwell emitted a TransientCrossingRequest (>= 1): {}",
        src.transient_crossings_requested,
    );
    // (2) THE AUTO-GRANT (the HR2 composition proof reachable now): the orchestrator resolved the dest
    // realm head and GRANTED the crossing back to the source — the geometric trigger fanned a SECOND
    // durability class through the SAME `evaluate_realm_boundaries`/`fan_out_crossing` machinery, with no
    // hand-fed batch. (`transient_crossings_granted` bumps only inside the grant arm at saga_runtime.rs.)
    assert!(
        orch.transient_crossings_granted >= 1,
        "the orchestrator auto-granted the transient crossing (>= 1): {}",
        orch.transient_crossings_granted,
    );
}

/// TEST 5 — DETERMINISM: the full autonomous crossing (Test 1 authority + the Test 2 render observer) is
/// byte-identical under one seed. Float nondeterminism would surface in the `capture_subject` frame /
/// world_pos (the render sample) and in the InspectReport trace — mirror p2_transfer_gates:652.
#[test]
fn crossing_e2e_is_byte_identical_under_same_seed() {
    let run = || {
        let mut caps: Vec<CapturedTick> = Vec::new();
        let (mut topo, subject) = run_autonomous_crossing(&FaultFabric::new(909, 2), &mut |t| {
            caps.push(capture_subject(t))
        });
        let reports = topo.inspect_all();
        let trace = topo.trace_bytes();
        (reports, trace, subject, caps)
    };
    assert_eq!(
        run(),
        run(),
        "identical seed ⇒ identical ground truth, harness trace, subject, AND the per-tick render trace \
         (the autonomous crossing is deterministic)",
    );
}

// ===================================================================================================
// TEST 3 — the pre-CAS ABORT leg + the Tier-1 CRASH leg (Mechanism-Y composition at cluster tier).
// ===================================================================================================

/// The subject's directory head-holding node (the pre-CAS proof reads this: it MUST stay at the SOURCE).
/// Reads the same `report(ORCH).directory` the happy-path Test 1 does (arm-agnostic `.node()`, per Q-4).
fn head_node(reports: &[(NodeId, InspectReport)], subject: EntityId) -> NodeId {
    report(reports, ORCH)
        .directory
        .iter()
        .find_map(|(k, r)| (*k == DirectoryKey::Entity(subject)).then_some(r.authority.node()))
        .expect("the crossing subject is recorded in the directory")
}

/// Shared warmup for both abort legs: warm the cluster + walking client to the M2 gate (SHARD holds the
/// avatar, DEST holds `to_realm`), capture `subject` + its directory fence, GUARD the `0x39` crossing
/// namespace, ARM the one-shot gateway reject AS LATE AS POSSIBLE, then PLANT the crossing shell. Returns
/// `subject`. Verbatim the `run_autonomous_crossing` warmup + M2 gate — only the tail (arm + plant vs
/// self-drive) diverges. The client must already be added to `topo`.
fn warm_and_arm_pre_cas_abort(topo: &mut Topology) -> EntityId {
    // WARMUP: the player logs in + walks; the SOURCE grants its avatar and the DEST wins its realm lease.
    step_until(topo, 80, &mut |_| {}, |t| {
        let r = t.inspect_all();
        !report(&r, SHARD).held_entities.is_empty()
            && report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == dest_stub_config().realm)
    });

    // M2 GATE (hard-assert, anti-vacuity): the DEST holds the exact realm the planted shell hands authority
    // to — else `handle_crossing_request` silently counts `crossing_unresolved` and NO saga (nor abort) runs.
    let warm = topo.inspect_all();
    assert!(
        report(&warm, DEST)
            .held_realms
            .iter()
            .any(|(realm, _)| *realm == dest_stub_config().realm),
        "M2: the DEST holds the shell's to_realm {:?} — the crossing can resolve its dest head",
        dest_stub_config().realm,
    );
    let (subject, _) = *report(&warm, SHARD)
        .held_entities
        .first()
        .expect("the SHARD holds the walking avatar before the crossing is planted");

    // PRE-CONDITION (cheap namespace guard): the id the source will latch is `0x39`-namespaced. `fence` is the
    // subject's directory head fence — the closest observable to the `subject_fence` the source mints into the
    // wire request. A driver namespace regression fails HERE with a precise message, not later as "no reply".
    let (_session, subject_dir, fence): (_, EntityId, Fence) = read_subject(topo);
    assert_eq!(
        subject_dir, subject,
        "the directory subject IS the SHARD-held avatar"
    );
    assert_eq!(
        crossing_transfer_id(DirectoryKey::Entity(subject), fence, 0).0 >> 120,
        0x39,
        "the crossing transfer id is 0x39-namespaced (the crossing-origin Mechanism-Y discriminator)",
    );

    // ARM the one-shot reject AS LATE AS POSSIBLE (right before the plant), so no warmup/re-driven prepare
    // can spend it on the wrong transfer — then PLANT the shell (mints the 0x39 crossing on the dwell).
    arm_gateway_reject(topo, PrepareReject::Spatial(SpatialReject::Obstructed));
    plant_one_crossing_shell(topo);
    subject
}

/// TEST 3 — THE PRE-CAS ABORT LEG: a crossing-origin durable saga forced to REJECT at Prepare (the gateway's
/// one-shot lever) aborts BEFORE the directory CAS, clears the source latch via Mechanism-Y, and the head
/// NEVER moves off the source. Proves the crash-durable crossing-abort COMPOSES at cluster tier (the abort
/// staged a persisted latch-clear obligation and the source's ack reaped it), all geometrically triggered.
#[test]
fn crossing_e2e_pre_cas_abort_clears_the_source_latch() {
    let fabric = FaultFabric::new(0x3F_AB07, 2);
    let mut topo = p2_cluster(&fabric, 8);
    topo.add_node(Box::new(p1_client(
        &fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    let subject = warm_and_arm_pre_cas_abort(&mut topo);

    // Drive to terminal, CAPTURING the transient pre-ack staging (the load-bearing non-vacuity proof) and
    // the post-plant crossings count (proves the lever wasn't spent on a stray prepare).
    // Exit only once the crossing has STARTED (`crossings_started >= 1`, so the abort ran, not a vacuous
    // "no saga ever lived") AND tombstoned (`live_sagas == 0`). Observe the transient `pending_abort_replies
    // >= 1` staging along the way (the load-bearing non-vacuity proof).
    let mut saw_pending_pre_ack = false;
    let mut crossings_after_plant = 0u64;
    step_until(
        &mut topo,
        60,
        &mut |t| {
            let r = t.inspect_all();
            if report(&r, ORCH).pending_abort_replies >= 1 {
                saw_pending_pre_ack = true;
            }
            crossings_after_plant = report(&r, ORCH).crossings_started;
        },
        // Exit once the FULL Mechanism-Y round-trip completed: the crossing started + aborted (tombstoned),
        // the source cleared its latch, AND the source's ack reaped the staged reply (`pending == 0`).
        |t| {
            let r = t.inspect_all();
            report(&r, ORCH).crossings_started >= 1
                && live_sagas(t) == 0
                && report(&r, SHARD).crossing_latches_cleared >= 1
                && report(&r, ORCH).pending_abort_replies == 0
        },
    );

    let r = topo.inspect_all();

    // (a) NON-VACUITY / Mechanism-Y ran: the 0x39 crossing-origin durable tombstone STAGED a reply BEFORE the
    //     source ack reaped it. A non-crossing-origin abort stages NOTHING (the `is_crossing_origin` gate), so
    //     `>= 1` at any tick proves the 0x39 pre-freeze path specifically ran — the discriminator.
    assert!(
        saw_pending_pre_ack,
        "pending_abort_replies >= 1 BEFORE the ack — the 0x39 crossing-origin latch-clear reply staged",
    );
    // (b) exactly ONE crossing started after the plant: the lever wasn't spent on a stray prepare, and a
    //     second crossing didn't sneak through Ready (the one-shot fired for THIS crossing).
    assert_eq!(
        crossings_after_plant, 1,
        "exactly one durable crossing saga started post-plant",
    );
    // (c) the staged reply was REAPED by the source's CrossingAbortedAck (the round-trip completed).
    assert_eq!(
        report(&r, ORCH).pending_abort_replies,
        0,
        "pending_abort_replies == 0 after the source ack reaped the reply",
    );
    // (d) the source POSITIVELY cleared its crossing latch (via on_crossing_aborted, not a timeout).
    assert!(
        report(&r, SHARD).crossing_latches_cleared >= 1,
        "the source cleared its RequestInFlight latch on the abort (crossing_latches_cleared >= 1): {}",
        report(&r, SHARD).crossing_latches_cleared,
    );
    // (e) no standing latch for the subject.
    assert!(
        !report(&r, SHARD).in_flight_latches.contains(&subject),
        "no standing RequestInFlight latch for the aborted subject: {:?}",
        report(&r, SHARD).in_flight_latches,
    );
    // (f) no live sagas (the abort tombstoned).
    assert_eq!(live_sagas(&mut topo), 0, "the aborted saga tombstoned");

    // (g1) PRE-CAS PROOF — head-position necessary half: the directory head STAYED at the SOURCE (authority
    //      never handed off; `abort_from_pre_freeze` issues no `IssueCommitCas`). Arm-agnostic `.node()`.
    assert_eq!(
        head_node(&r, subject),
        SHARD,
        "pre-CAS abort: the directory head never moved off the SOURCE",
    );
    // (g2) PRE-CAS PROOF — sufficient half: the DEST never adopted the subject (a post-CAS-then-compensate
    //      path would have advanced a dest ownership counter). Combined with (a), pre-CAS is fully pinned.
    assert!(
        !report(&r, DEST)
            .held_entities
            .iter()
            .any(|(e, _)| *e == subject),
        "pre-CAS: the DEST never adopted the subject (no CAS ran)",
    );

    // The HR2 authority oracles hold after the abort tail (exactly one holder, no lingering pending).
    verify_authority_unique(&r).expect("exactly one holder of the aborted subject (the SOURCE)");
    verify_authority_settled(&r).expect("no lingering pending/departing after the abort tail");
}

/// TEST — THE TIER-1 CRASH LEG: an orchestrator REBUILD (World-rebuild via `rebuild_orchestrator` from the
/// RETAINED WAL — an honest analog of a kill-9, NOT a process SIGKILL) BETWEEN the abort-emit and the source
/// ack. The parked ack sits in the fabric while the orchestrator dies + re-hydrates; the persisted
/// `pending_abort_replies` entry rides the WAL, is restored, and the first post-restart scan re-emits so the
/// source clears/re-acks. Proves the crash-durable crossing-abort COMPOSES across an orchestrator restart.
///
/// TIER HONESTY (`..._survives_orchestrator_restart`, NOT `..._kill9`): a World-rebuild, not a `kill -9`. The
/// abort-reply crash *durability* is process-proven generically by `orchestrator_crash.rs` + WAL-specifically
/// by `rehydrate_restores_a_persisted_abort_reply_and_reemits_on_first_scan`; the crossing-specific SIGKILL
/// stays owed (DEFERRED D-43 — a composition gap, blocked on a shard-bin boundary-plant knob).
#[test]
fn crossing_e2e_abort_survives_orchestrator_restart() {
    let fabric = FaultFabric::new(0x3F_AB08, 2);
    let (mut topo, store) = p2_cluster_durable_orch(&fabric, 8); // RETAINED MemStore
    topo.add_node(Box::new(p1_client(
        &fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    let subject = warm_and_arm_pre_cas_abort(&mut topo);

    // PHASE 1 — HEALTHY link: drive until the crossing has STARTED, aborted, and STAGED its persisted reply
    // (`pending_abort_replies >= 1`). The link MUST stay healthy here: the `CrossingRequest` that starts the
    // saga rides SHARD -> ORCH — the SAME directed link we park below — so parking before the request
    // arrives would strand the crossing (it would never become a saga). Stop the INSTANT the reply is staged.
    step_until(&mut topo, 80, &mut |_| {}, |t| {
        report(&t.inspect_all(), ORCH).pending_abort_replies >= 1
    });

    // PHASE 2 — PARK the ack direction (SHARD -> ORCH Saga) so the source's `CrossingAbortedAck` requeues in
    // the fabric (partitioned = block + redeliver, never dropped) instead of reaping the staged reply. The
    // `CrossingAborted` re-emit flows ORCH -> SHARD (left healthy), so the source still receives it, clears
    // its latch, and acks — but that ack is now PARKED. This is the "abort emitted, ack in flight" instant.
    fabric.set_policy(
        SHARD,
        ORCH,
        LinkPolicy {
            partitioned: true,
            ..Default::default()
        },
    );
    // Drive (ack parked) until the SOURCE has RECEIVED a re-emitted `CrossingAborted` and cleared its latch —
    // `crossing_latches_cleared` bumps ONLY via `on_crossing_aborted` (the pre-CAS abort never commits, so
    // `on_saga_demote`, the other writer, cannot fire for this subject). With the ack parked, the ORCH still
    // holds the staged reply — so this is exactly "an emit fired + its ack is parked, entry not reaped".
    step_until(&mut topo, 80, &mut |_| {}, |t| {
        report(&t.inspect_all(), SHARD).crossing_latches_cleared >= 1
    });
    let pre = topo.inspect_all();
    assert!(
        report(&pre, ORCH).pending_abort_replies >= 1,
        "the abort reply is still staged (ack parked by the partition, not reaped) at the restart instant",
    );
    assert!(
        report(&pre, SHARD).crossing_latches_cleared >= 1,
        "an emit FIRED pre-restart and the source acked (the ack is the thing being parked)",
    );

    // REBUILD the orchestrator from the RETAINED store (the World-rebuild kill-9 analog).
    rebuild_orchestrator(&mut topo, &fabric, store.clone());

    // THE LOAD-BEARING CRASH ASSERTION: capture the instant AFTER rebuild, BEFORE any healed step. This (and
    // only this) proves the entry rode the WAL — not the throttle reset, not in-process World survival.
    assert!(
        report(&topo.inspect_all(), ORCH).pending_abort_replies >= 1,
        "the persisted abort-reply was RESTORED from the WAL by the cluster rehydrate (pre-heal capture)",
    );

    // HEAL + reap — a LABELED liveness check (NOT the crash proof: the post-restart re-emit is throttle-reset-
    // guaranteed, so this only shows the tail completes once the parked ack redelivers).
    fabric.set_policy(SHARD, ORCH, LinkPolicy::default());
    step_until(&mut topo, 60, &mut |_| {}, |t| {
        report(&t.inspect_all(), ORCH).pending_abort_replies == 0
    });

    let r = topo.inspect_all();
    assert_eq!(
        report(&r, ORCH).pending_abort_replies,
        0,
        "liveness: the reply was reaped post-restart by the redelivered source ack",
    );
    assert!(
        report(&r, SHARD).crossing_latches_cleared >= 1,
        "the source latch stayed cleared across the restart",
    );
    assert!(
        !report(&r, SHARD).in_flight_latches.contains(&subject),
        "no standing latch for the subject after the restart tail: {:?}",
        report(&r, SHARD).in_flight_latches,
    );
    assert_eq!(
        head_node(&r, subject),
        SHARD,
        "the directory head never moved — the pre-CAS abort survived the restart",
    );
}

/// MANDATORY NEGATIVE CONTROL (anti-theater): rebuild an IDENTICAL fixture against a FRESH empty
/// `MemStore::new()` and assert `pending_abort_replies == 0` immediately post-rebuild. Without this, the
/// crash-leg's pre-heal capture is confounded by an in-process-survival illusion; with it, the WAL-restore
/// assertion is PROVEN to read the store, not a leftover — the empty store recovers NOTHING.
#[test]
fn crossing_e2e_abort_reply_absent_from_empty_store_rebuild() {
    let fabric = FaultFabric::new(0x3F_AB09, 2);
    let (mut topo, _store) = p2_cluster_durable_orch(&fabric, 8);
    topo.add_node(Box::new(p1_client(
        &fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    let _subject = warm_and_arm_pre_cas_abort(&mut topo);

    // Drive until the abort reply is genuinely staged (the same window the crash leg captures) — so the
    // control is non-vacuous: a reply DID exist in the dying process, and the empty rebuild must still find 0.
    step_until(&mut topo, 80, &mut |_| {}, |t| {
        report(&t.inspect_all(), ORCH).pending_abort_replies >= 1
    });
    assert!(
        report(&topo.inspect_all(), ORCH).pending_abort_replies >= 1,
        "the abort reply is staged in the (about-to-die) orchestrator before the empty rebuild",
    );

    // REBUILD against a FRESH empty store (a NON-durable restart): nothing is persisted to recover.
    rebuild_orchestrator(&mut topo, &fabric, MemStore::new());

    // THE CONTROL: immediately post-rebuild, the empty store recovered NO abort reply — proving the crash
    // leg's `>= 1` restore reads the WAL, not an in-process survivor.
    assert_eq!(
        report(&topo.inspect_all(), ORCH).pending_abort_replies,
        0,
        "an EMPTY-store rebuild restores NO abort reply (the crash-leg restore reads the WAL, not RAM)",
    );
}
