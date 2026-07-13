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
//! second class), Test 5 determinism. The abort leg (Test 3), the crash leg, and 3f-D4 are OUT OF SCOPE.
//!
//! GEOMETRY HONESTY: the shell is centered at the dot's SPAWN offset, so the dot is born-inside and
//! commits after `n_entry` dwell ticks. This proves TRIGGER→TRANSFER composition; the pixel-visible
//! PHYSICAL traversal (a dot that walks ACROSS a boundary) remains the D-15/D-30 visual owed line.

use vd_core::glam::DVec3;
use vd_core::pose::FrameRef;
use vd_core::{AccountId, EntityId, NodeId, TickId};
use vd_harness::client::ScriptedClient;
use vd_harness::fabric::FaultFabric;
use vd_harness::oracle::{RenderSample, verify_authority_settled, verify_authority_unique};
use vd_harness::topology::{InspectReport, Topology};
use vd_core::entity_kind::EntityKind;
use vd_tests::{
    DEST, ORCH, SHARD, dest_stub_config, live_sagas, p1_client, p2_cluster, plant_one_crossing_shell,
    realm_fence, saga_states, seed_held_transient, stub_config, walk_forward,
};
use vd_wire::channels::SubId;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey};

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
    let (_topo, _subject) = run_autonomous_crossing(&fabric, &mut |t| caps.push(capture_subject(t)));

    let rendered: Vec<RenderSample> = caps.iter().filter_map(|c| c.sample).collect();
    let dest_sample = *rendered
        .last()
        .expect("the subject renders at some tick across the crossing");

    // The dest renders the CROSSED SOURCE-realm pose (`SystemSpace{seed:7}`), not its own adopt-default
    // seed-8 origin — a dropped crossing would leave seed 8. This is the same discriminator as the D-28
    // render gate: the dest's own integration moves `pos` but never `frame`, so a seed-7 frame at the dest
    // can ONLY come from a landed crossing.
    assert_eq!(
        dest_sample.frame,
        FrameRef::SystemSpace { system_seed: 7 },
        "the dest renders the CROSSED source-realm frame (seed 7), not the adopt-default (seed 8): {dest_sample:?}",
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
/// ⚠️ HONEST SCOPE (a real missing route, NOT hand-fed to force green): the AUTONOMOUS transient loop
/// closes through Leg 1 + the auto-grant, then STOPS. The grant path
/// (`saga_runtime::handle_transient_crossing_request`, saga_runtime.rs ~:1460) GRANTS but never
/// `start_transfer`s a `BatchHandoff` saga — unlike the DURABLE `handle_crossing_request` (:1440) which
/// starts one. So when the source ships the `TransientBatch` and the DEST adopts + acks `BatchAdopted`,
/// the orchestrator's `deliver` (saga_runtime.rs :1110) early-returns on the unknown transfer (no live
/// saga), and the item never promotes `Arriving→Held` (no `batch_goes`, empty dest `owned_transients`).
/// The missing production route is the batch-saga START on the transient-crossing-grant path — a
/// SATURATION of the geometric trigger into the transient handoff, distinct from this trigger-composition
/// slice. Asserting the auto-grant is the honest, non-vacuous HR2 proof reachable NOW; the downstream
/// batch-handoff composition is REPORTED as owed (see the task report), never faked here.
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
        let (mut topo, subject) =
            run_autonomous_crossing(&FaultFabric::new(909, 2), &mut |t| caps.push(capture_subject(t)));
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
