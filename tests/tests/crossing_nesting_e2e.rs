//! Slice 4b (task #133) — THE NESTING / SELF-HEAL COMPOSITION PROOF.
//!
//! This gate proves the re-scoped 4b claim that MAKES the direction-keyed latch unnecessary: after a
//! durable crossing COMMITS (an entity re-homes SHARD→DEST and DEST OWNS it), the DEST shard's
//! `evaluate_realm_boundaries` RE-EVALUATES that ADOPTED entity and fires a SUBSEQUENT `CrossingRequest`
//! when the entity crosses a boundary planted on DEST. Combined with the per-ENTITY serialization
//! (`RequestInFlight` is `BTreeMap<EntityId, TransferId>` + `fan_out_crossing`'s `Entry::Occupied`
//! suppress — there is never a second concurrent saga per entity), a dock-then-undock interleave
//! SELF-HEALS: the dest re-evaluates the adopted entity and fires the undock on its own. So NO
//! direction-keyed latch and NO `crossing_transfer_id` change are needed (see
//! `scripts/slice4b_design_adversary.md` FINDING-1; the design workflow + adversary + this verification
//! proved the simpler design).
//!
//! ANTI-VACUITY: the subsequent crossing is driven ONLY by planting a geometric boundary on DEST +
//! moving the adopted entity across it — never a hand-fed `trigger_transfer`. The load-bearing signal is
//! DEST's own `crossings_requested` going 0 → ≥ 1 for the SAME entity DEST adopted (with a negative
//! control that keeps it at 0 when no boundary is planted / nothing moves).

use vd_core::geometry::{Boundary, ContainmentBand, RealmRegion};
use vd_core::glam::DVec3;
use vd_core::pose::{LatticePos, RealmId, frame_for_realm};
use vd_core::{AccountId, EntityId, NodeId};
use vd_harness::client::ScriptedClient;
use vd_harness::fabric::{FabricTransport, FaultFabric};
use vd_harness::topology::{InspectReport, Topology};
use vd_node::ShardNode;
use vd_tests::{
    DEST, SHARD, dest_stub_config, live_sagas, p1_client, p2_cluster, plant_one_crossing_shell,
    saga_states, walk_forward,
};

const CLIENT: NodeId = NodeId(100);

fn report(reports: &[(NodeId, InspectReport)], id: NodeId) -> &InspectReport {
    &reports
        .iter()
        .find(|(n, _)| *n == id)
        .expect("node present")
        .1
}

/// Downcast a topology node to its concrete `ShardNode<FabricTransport>` (mirrors the private
/// `with_node` in vd-tests + `with_client` in crossing_e2e.rs).
fn with_shard<R>(
    topo: &mut Topology,
    id: NodeId,
    f: impl FnOnce(&mut ShardNode<FabricTransport>) -> R,
) -> R {
    let node = topo.node_mut(id).expect("node present");
    let shard = node
        .as_any_mut()
        .expect("ShardNode opts into downcasting")
        .downcast_mut::<ShardNode<FabricTransport>>()
        .expect("the node is a ShardNode");
    f(shard)
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

fn step_until(topo: &mut Topology, max: u64, mut cond: impl FnMut(&mut Topology) -> bool) {
    for _ in 0..max {
        topo.step();
        if cond(topo) {
            return;
        }
    }
    panic!("crossing-nesting condition not reached within {max} ticks");
}

/// Set the ADOPTED subject dot's frame-local offset on DEST (find the dot whose `entity == subject`).
/// Returns true if found + set. The adopted dot has NO client input reaching DEST (source retains the
/// connection, R2), so `process_inbound` never moves it — the manual write survives into
/// `evaluate_realm_boundaries` the same tick (schedule order: process_inbound → evaluate_realm_boundaries).
fn set_dest_subject_offset(topo: &mut Topology, subject: EntityId, off: DVec3) -> bool {
    with_shard(topo, DEST, |s| {
        let dots = s.world_mut().resource_mut::<vd_sim::stub::Dots>();
        for dot in dots.into_inner().0.values_mut() {
            if dot.entity == subject {
                dot.pose.pos = LatticePos::local(off);
                return true;
            }
        }
        false
    })
}

/// Plant a REGION forest into DEST's `RealmRegions` (the DEST variant of vd-tests' SHARD-only
/// `plant_crossing_boundaries`; C-3 CONTAINMENT).
fn plant_dest_regions(topo: &mut Topology, regions: Vec<RealmRegion>) {
    with_shard(topo, DEST, |s| {
        *s.world_mut().resource_mut::<vd_sim::stub::RealmRegions>() =
            vd_sim::stub::RealmRegions::new(regions);
    });
}

/// One `RealmRegion` shell at the origin of `realm`'s frame, radius `r`, nested under `parent`.
fn dest_region(realm: RealmId, parent: Option<RealmId>, r: f64) -> RealmRegion {
    let band = ContainmentBand::for_containment_velocity_safe(
        50.0,
        100.0,
        dest_stub_config().move_speed_mps,
        dest_stub_config().tick_dt_s,
        1.0,
    )
    .expect("valid dest containment band");
    RealmRegion {
        realm,
        center: LatticePos::local(DVec3::ZERO),
        frame: frame_for_realm(realm, None).expect("System realm always resolves a frame"),
        shape: Boundary::Shell { r },
        band,
        parent,
    }
}

/// Drive one autonomous durable crossing SHARD→DEST (mirrors crossing_e2e::run_autonomous_crossing),
/// returning the quiesced topo + the transferred subject. Uses vd-tests' PUBLIC helpers only.
fn drive_crossing(fabric: &FaultFabric) -> (Topology, EntityId) {
    let mut topo = p2_cluster(fabric, 8);
    topo.add_node(Box::new(p1_client(
        fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    // WARMUP: source grants the avatar; dest wins its realm lease.
    step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        !report(&r, SHARD).held_entities.is_empty()
            && report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == dest_stub_config().realm)
    });
    let warm = topo.inspect_all();
    let (subject, _) = *report(&warm, SHARD)
        .held_entities
        .first()
        .expect("the SHARD holds the walking avatar before the crossing");

    // ARM the geometric trigger on SHARD → the dwell fires a CrossingRequest → saga self-drives.
    plant_one_crossing_shell(&mut topo);
    step_until(&mut topo, 40, |t| {
        saga_states(t).iter().any(|s| s.starts_with("Freezing"))
    });
    step_until(&mut topo, 60, |t| live_sagas(t) == 0);

    // QUIESCE: stop input, let the demote/promote tail land (source→Ghost, dest→Owned).
    with_client(&mut topo, ScriptedClient::pause_input);
    for _ in 0..8 {
        topo.step();
    }
    step_until(&mut topo, 40, |t| {
        let r = t.inspect_all();
        let src = report(&r, SHARD);
        let dst = report(&r, DEST);
        !src.held_entities.iter().any(|(e, _)| *e == subject)
            && dst.held_entities.iter().any(|(e, _)| *e == subject)
    });

    (topo, subject)
}

/// THE SLICE-4b SELF-HEAL PROOF: DEST re-evaluates the ADOPTED entity and fires a SUBSEQUENT
/// `CrossingRequest` when the adopted entity crosses a boundary planted on DEST. This is the composition
/// property that makes the direction-keyed latch unnecessary — the destination shard re-drives the next
/// crossing on its own, so a dock-then-undock never needs two concurrent per-entity sagas on one shard.
#[test]
fn crossing_e2e_dest_reevaluates_adopted_entity_self_heal() {
    let fabric = FaultFabric::new(909, 2);
    let (mut topo, subject) = drive_crossing(&fabric);

    // ============ PRE-CONDITION (anti-vacuity baseline) ============
    let pre = topo.inspect_all();
    assert!(
        report(&pre, DEST)
            .held_entities
            .iter()
            .any(|(e, _)| *e == subject),
        "PRECOND: DEST OWNS the adopted subject after the crossing (held_entities): {:?}",
        report(&pre, DEST).held_entities,
    );
    assert_eq!(
        report(&pre, DEST).crossings_requested,
        0,
        "PRECOND: DEST has fired NOTHING yet (crossings_requested == 0) — the anti-vacuity baseline",
    );
    // Confirm SHARD (the source) fired the ORIGINAL crossing (>= 1), so the >= 1 we later read at DEST
    // cannot be confused with the source's counter.
    assert!(
        report(&pre, SHARD).crossings_requested >= 1,
        "the SOURCE fired the original crossing (SHARD.crossings_requested >= 1): {}",
        report(&pre, SHARD).crossings_requested,
    );

    // ============ PLANT A REGION FOREST ON DEST (not SHARD) ============
    // C-3 CONTAINMENT: root ⊃ own(System(8)) ⊃ child(System(99)), all origin-coincident. The subject
    // starts OUTSIDE the child (container == DEST's own realm ⇒ no re-home) and is moved INTO the child
    // (container flips to System(99) ⇒ DEST re-homes). `child.realm = System(99)` need not resolve at the
    // orchestrator — the CLAIM is only that DEST's `crossings_requested` bumps (inside fan_out_crossing, at
    // emit, BEFORE any orchestrator resolution).
    let center = DVec3::ZERO;
    let root = dest_region(RealmId::System(0), None, 1.0e9);
    let own = dest_region(
        dest_stub_config().realm,
        Some(RealmId::System(0)),
        100_000.0,
    );
    // A DISTINCT third realm — the deeper child region whose membership triggers the DEST re-home.
    let child = dest_region(RealmId::System(99), Some(dest_stub_config().realm), 1000.0);
    plant_dest_regions(&mut topo, vec![root, own, child]);

    // ============ MOVE THE ADOPTED ENTITY INTO THE DEST CHILD REGION ============
    // Start it OUTSIDE the child shell (2000 m out, > outset) but inside own ⇒ container == own(System(8)),
    // no re-home. The manual pose write lands the same tick as evaluate_realm_boundaries (schedule:
    // process_inbound → evaluate_realm_boundaries), and the adopted dot receives no DEST-side input so
    // nothing fights the write.
    assert!(
        set_dest_subject_offset(&mut topo, subject, DVec3::new(2000.0, 0.0, 0.0)),
        "the adopted subject dot is present on DEST and settable",
    );
    topo.step();

    // Move it INTO the child region (deep inside the shell): container flips to System(99) ⇒ re-home.
    // Hold at the center for a few ticks so the containment membership is stable through the commit.
    for _ in 0..8 {
        set_dest_subject_offset(&mut topo, subject, center);
        topo.step();
    }

    // ============ THE LOAD-BEARING ASSERT ============
    let post = topo.inspect_all();
    let dest = report(&post, DEST);

    // The subject is STILL the same DEST-owned entity that crossed in (not a native spawn).
    assert!(
        dest.held_entities.iter().any(|(e, _)| *e == subject),
        "the subject is still the SAME adopted entity DEST owns (not a fresh native spawn): {:?}",
        dest.held_entities,
    );
    // NON-VACUOUS: 0 before (asserted above), >= 1 now — DEST re-evaluated the adopted entity and fired a
    // subsequent CrossingRequest. This is the self-heal property (the dest re-drives the next crossing).
    assert!(
        dest.crossings_requested >= 1,
        "SELF-HEAL: DEST re-evaluated the ADOPTED entity and fired a subsequent CrossingRequest \
         (DEST.crossings_requested >= 1): {}",
        dest.crossings_requested,
    );
}

/// NEGATIVE CONTROL (anti-theater): the SAME committed crossing, but NO DEST boundary is ever planted and
/// the adopted entity is never moved. DEST.crossings_requested MUST stay 0 — proving the positive test's
/// `post >= 1` is CAUSED by the DEST-planted boundary + the adopted entity's motion, not a background
/// emission or a mislabeled counter.
#[test]
fn crossing_e2e_no_dest_boundary_means_dest_fires_nothing() {
    let fabric = FaultFabric::new(909, 2);
    let (mut topo, subject) = drive_crossing(&fabric);
    // Step a comparable number of ticks with NO boundary planted on DEST and no motion driven.
    for _ in 0..12 {
        topo.step();
    }
    let post = topo.inspect_all();
    let dest = report(&post, DEST);
    assert!(
        dest.held_entities.iter().any(|(e, _)| *e == subject),
        "DEST still owns the subject in the control",
    );
    assert_eq!(
        dest.crossings_requested, 0,
        "NEGATIVE CONTROL: with NO DEST boundary + no motion, DEST fires nothing (the positive \
         post >= 1 is caused by the plant+move, not background)",
    );
}
