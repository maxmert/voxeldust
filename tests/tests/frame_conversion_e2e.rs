//! THE UPWARD CROSSING, end to end over a real cluster — the owner's live symptom, stated in numbers.
//!
//! THE GROUND RULE this gate exists to enforce: *only the parent knows where the children are; a child has
//! no idea about its own position.* Every realm is centred on itself. A planet knows "an occupant 15 m from
//! my centre" and nothing else; its star knows "I put that planet 20 m out". Turning the first into "35 m
//! from the star" is ONE addition, and it belongs to the star, because the star is the only party that
//! holds the 20.
//!
//! WHAT WENT WRONG BEFORE. The saga relabelled the pose: it stamped the destination realm's frame onto the
//! occupant's position without moving the number. "15 m from the planet's centre" silently became "15 m
//! from the star's centre", and because the receiving shard then saw a pose already wearing its own frame,
//! its conversion took the same-frame shortcut and added nothing. The player crossed out of the planet and
//! appeared 20 m from where they left — at the star.
//!
//! SO THIS FILE ASSERTS THE NUMBER, not just the routing. It is RED before the slice that removed the
//! relabel (the parent reports the occupant at 15) and GREEN after (35). An authority-only assertion would
//! have passed either way, which is why the authority gates never caught this.
//!
//! ANTI-VACUITY, enforced by construction: a grep of this file finds ZERO `trigger_transfer`. The re-home
//! is driven by the REAL seed-forest containment detector reading the shard's own lineage, and carried by
//! the REAL transfer saga. Nothing is hand-fed.
//!
//! TIER: harness (in-process `Topology`) — the identical detector, saga, directory and shard code the
//! process bins run. The occupant's POSITION is scripted (as every containment gate at this tier does);
//! the re-home DECISION, the saga and the conversion are all real.
//!
//! ## What the second half of this file is for
//!
//! Everything above proves the SHARD's half at walk scale, where every realm sits still at its parent's
//! origin and every conversion is therefore very nearly the identity. That proves nothing about the case
//! the whole arc exists for: a realm that MOVES, and a world big enough that a millimetre stops being
//! representable if anyone inflates a position to universe scale on the way. The tests below supply those
//! measurements — coherence between the two render feeds ([`the_box_and_the_thing_standing_in_it_draw_at_one_point`]),
//! the precision the render path keeps at 1e13 m, the chord error of the gateway's linear placement carry
//! (which is what sizes `vd_sim::stub::PlacementCarry::BUDGET_MS`), and the observed rows-per-datagram
//! distribution.

use std::collections::{BTreeMap, BTreeSet};

use vd_core::UniverseTick;
use vd_core::celestial::{G, OrbitalElements, orbital_state};
use vd_core::frame::FrameContext;
use vd_core::glam::DVec3;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::worldgen::{UniverseConfig, WorldView, moving_children_for_config};
use vd_core::{AccountId, EntityId, NodeId};
use vd_harness::client::ScriptedClient;
use vd_harness::fabric::FaultFabric;
use vd_harness::topology::{InspectReport, Topology};
use vd_sim::stub::PlacementCarry;
use vd_tests::frame_fixture::{
    FAR_OCCUPANT_FROM_PLANET_M, FAR_SYSTEM_FROM_GALAXY_M, NEAR_PLANET_FROM_STAR_M, WorkedExample,
};
use vd_tests::{
    CHAIN_MID, CHAIN_TOP, DEST, FRAME_UNIVERSE_SEED, ORCH, SHARD, live_sagas, p1_client,
    p1_cluster, p2_cluster_area_in_planet_in_system, p2_cluster_planet_in_system, plant_regions,
    plant_seed_neighbourhood, plant_seed_neighbourhood_with_movers, set_shard_subject_offset,
    set_shard_subject_pose_now, set_shard_subject_pose_now_with, walk_forward,
};
use vd_wire::channels::RealmSnapshotDatagram;
use vd_wire::intershard::InterShardFlow;

const CLIENT: NodeId = NodeId(100);

/// The realm the player logs in to: the planet, hosted by [`SHARD`].
const PLANET: RealmId = RealmId::Planet(7);
/// The realm that AUTHORS where that planet sits: its star, hosted by [`DEST`].
const SYSTEM: RealmId = RealmId::System(7);

/// The number ONLY THE STAR HOLDS: where it put its planet, in its own frame. This is the seed forest's
/// `planet_offset_m`, written out as a literal rather than read back out of the world under test — a
/// fixture that recomputes its expectation with the subject's own expression cannot fail when the subject
/// is wrong.
const PLANET_FROM_STAR_M: f64 = 20.0;
/// The number ONLY THE PLANET HOLDS: how far out the occupant is walked, in the planet's own frame. Past
/// the planet's 10 m boundary and its 2 m hysteresis outset, so the containment detector genuinely
/// releases it; still well inside the star's 40 m boundary, so the star is where it belongs.
const OCCUPANT_FROM_PLANET_M: f64 = 15.0;
/// THE ANSWER, as a literal: the star adds its child's placement, 20 + 15. Before the relabel was removed
/// the parent reported 15 — the same number the child shipped, wearing the parent's frame.
const OCCUPANT_FROM_STAR_M: f64 = 35.0;
/// Where the occupant is put, in the STAR's frame, to cross back IN — 5 m past the planet's centre, so it
/// is comfortably inside that planet's 10 m boundary rather than balanced on its edge.
const OCCUPANT_ENTERS_FROM_STAR_M: f64 = 25.0;
/// THE ANSWER for the INWARD leg, as a literal: the star SUBTRACTS its child's placement, 25 − 20. A
/// relabel reports 25 instead — a position 25 m from the centre of a realm whose whole boundary is 10 m.
const OCCUPANT_INSIDE_PLANET_M: f64 = 5.0;
/// The planet's own boundary radius in the seed forest, written out as a literal. An occupant the planet
/// has ACCEPTED must be inside this, or the planet's own detector sees it outside itself on the very next
/// tick and hands it straight back — the flap.
const PLANET_BOUNDARY_R_M: f64 = 10.0;
/// The period given to the orbiting planet: long enough that its per-tick step (about 0.6 m) stays far
/// inside the 3 m entry/exit dead zone — so nothing in this gate can be explained by a body skipping the
/// band — and short enough that it sweeps over the waiting occupant within the step budget below.
const PLANET_ORBIT_PERIOD_S: f64 = 10.0;

/// THE TURN GIVEN TO THE PLANET: circular, in-plane, phase pinned to zero, on exactly the shell its static
/// region sits on — so the moving planet traces the same 20 m circle the static fixture puts it at, and the
/// only thing that changes between the two gates is whether it is standing still.
///
/// The central mass is chosen THROUGH the period (`T = 2π√(a³/μ)`, `μ = G·M`) rather than written down,
/// because the period is the thing being chosen and a mass constant is a number nobody can check.
fn planet_orbit() -> OrbitalElements {
    let a = PLANET_FROM_STAR_M;
    let mu = 4.0 * std::f64::consts::PI * std::f64::consts::PI * a.powi(3)
        / (PLANET_ORBIT_PERIOD_S * PLANET_ORBIT_PERIOD_S);
    OrbitalElements {
        sma: a,
        ecc: 0.0,
        inclination: 0.0,
        raan: 0.0,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: 0.0,
        central_mass: mu / G,
    }
}

fn report(reports: &[(NodeId, InspectReport)], id: NodeId) -> &InspectReport {
    &reports
        .iter()
        .find(|(n, _)| *n == id)
        .expect("node present")
        .1
}

fn with_client<R>(topo: &mut Topology, f: impl FnOnce(&mut ScriptedClient) -> R) -> R {
    with_client_at(topo, CLIENT, f)
}

/// Like [`with_client`] but for a NAMED client node — the two-player scenario holds two.
fn with_client_at<R>(
    topo: &mut Topology,
    node: NodeId,
    f: impl FnOnce(&mut ScriptedClient) -> R,
) -> R {
    let node = topo.node_mut(node).expect("client present");
    let client = node
        .as_any_mut()
        .expect("clients opt into downcasting")
        .downcast_mut::<ScriptedClient>()
        .expect("the node is a ScriptedClient");
    f(client)
}

/// Step at most `max` ticks until `cond` holds. Bounded + deterministic; returns whether it held.
fn step_until(topo: &mut Topology, max: u64, mut cond: impl FnMut(&mut Topology) -> bool) -> bool {
    for _ in 0..max {
        topo.step();
        if cond(topo) {
            return true;
        }
    }
    false
}

/// The pose the shard `node` reports for `subject` in its `InspectReport`, or `None` if it holds no such
/// entity. Read from the harness inspect surface, never from node internals.
fn held_pose(
    reports: &[(NodeId, InspectReport)],
    node: NodeId,
    subject: EntityId,
) -> Option<vd_core::pose::StampedPose> {
    report(reports, node)
        .held_poses
        .iter()
        .find(|(e, _)| *e == subject)
        .map(|(_, p)| *p)
}

/// Spin the parent-and-child cluster with a real logged-in player on the PLANET, plant the seed
/// neighbourhood on both shards, and return the topology plus the avatar's entity id.
fn boot_planet_and_star(fabric: &FaultFabric) -> (Topology, EntityId) {
    let mut topo = p2_cluster_planet_in_system(fabric, 8);
    topo.add_node(Box::new(p1_client(
        fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    // WARMUP: the player logs in on the PLANET shard and the STAR shard wins its realm lease.
    let warm = step_until(&mut topo, 120, |t| {
        let r = t.inspect_all();
        !report(&r, SHARD).held_entities.is_empty()
            && report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == SYSTEM)
    });
    assert!(
        warm,
        "the avatar is granted on the planet and the star shard holds its realm",
    );

    // Stop the scripted walk: this gate scripts the occupant's position explicitly (below), and a client
    // still driving inputs would fight it.
    with_client(&mut topo, ScriptedClient::pause_input);

    // ARM the REAL detector on both shards with the REAL seed geometry — the same
    // `realm_neighbourhood_for` the production shard boot computes. From here nothing is hand-fed.
    plant_seed_neighbourhood(&mut topo, SHARD, FRAME_UNIVERSE_SEED, PLANET);
    plant_seed_neighbourhood(&mut topo, DEST, FRAME_UNIVERSE_SEED, SYSTEM);

    let reports = topo.inspect_all();
    let (subject, _) = *report(&reports, SHARD)
        .held_entities
        .first()
        .expect("the planet shard holds the logged-in avatar");
    (topo, subject)
}

/// THE HEADLINE. A durable player walks out of the planet; the star — the only party that knows where it
/// put that planet — receives the occupant and ADDS the placement. The star must report 35, not 15.
#[test]
fn an_occupant_leaving_a_realm_arrives_at_its_parent_in_the_parents_frame() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, subject) = boot_planet_and_star(&fabric);

    // BEFORE: the planet holds the occupant, in the planet's OWN frame. That is all a planet ever knows.
    let before = topo.inspect_all();
    let start = held_pose(&before, SHARD, subject).expect("the planet holds the avatar");
    assert_eq!(
        start.frame,
        FrameRef::PlanetCentered { planet_seed: 7 },
        "the avatar is born measured from the planet's own centre",
    );

    // Walk it OUT past the planet's boundary and HOLD it there each tick (the scripted write lands before
    // the detector runs). The occupant's position is scripted ONLY on the planet — the star's copy is
    // whatever the star itself computes, which is the thing under test.
    //
    // CAPTURE THE ARRIVAL POSE the first tick the star reports holding it, rather than reading it at the
    // end: this gate is about the number the star computed on receipt, and reading later would let the
    // star's own simulation move it first.
    let out = DVec3::new(OCCUPANT_FROM_PLANET_M, 0.0, 0.0);
    let mut arrival: Option<vd_core::pose::StampedPose> = None;
    let landed = step_until(&mut topo, 400, |t| {
        set_shard_subject_offset(t, SHARD, subject, out);
        let r = t.inspect_all();
        if arrival.is_none() {
            arrival = held_pose(&r, DEST, subject);
        }
        arrival.is_some()
    });
    assert!(landed, "the occupant re-homes from the planet to its star");
    let arrived = arrival.expect("just captured");

    // The composition proof (anti-vacuity, causal): the planet's own detector emitted a CrossingRequest
    // and the orchestrator turned it into a real saga. There is no `trigger_transfer` in this file.
    let reports = topo.inspect_all();
    assert!(
        report(&reports, SHARD).crossings_requested >= 1,
        "the planet's containment detector emitted the crossing itself: {}",
        report(&reports, SHARD).crossings_requested,
    );
    assert!(
        report(&reports, ORCH).crossings_started >= 1,
        "that request became a real transfer saga: {}",
        report(&reports, ORCH).crossings_started,
    );

    // (1) THE FRAME: the star measures it in the STAR's frame. Not a relabel of the planet's.
    assert_eq!(
        arrived.frame,
        FrameRef::SystemSpace { system_seed: 7 },
        "the arrival is measured from the star's centre",
    );
    // (2) THE NUMBER, which is the whole gate: the star ADDED where it put its planet. 20 + 15.
    // Exact: every quantity in this fixture is exact in binary at this scale, so a tolerance here would
    // be hiding something.
    assert_eq!(
        arrived.pos.offset(),
        DVec3::new(OCCUPANT_FROM_STAR_M, 0.0, 0.0),
        "the star adds its child's placement: {PLANET_FROM_STAR_M} + {OCCUPANT_FROM_PLANET_M}. \
         Reporting {OCCUPANT_FROM_PLANET_M} here is the relabel bug: the frame changed, the number \
         did not, and the player is drawn {PLANET_FROM_STAR_M} m away — at the star.",
    );

    // (2b) AND IT STAYS. A pose that keeps the child's number while wearing the parent's frame lands the
    // occupant back inside the planet's boundary as the star reads it, so the two shards hand it back and
    // forth forever. The hand-off settling is therefore part of the same proof, not a separate nicety.
    let settled = step_until(&mut topo, 200, |t| live_sagas(t) == 0);
    let reports = topo.inspect_all();
    assert!(
        settled,
        "the hand-off settles instead of flapping: saga_states={:?}",
        vd_tests::saga_states(&mut topo),
    );
    // (3) And the planet no longer holds it authoritatively: one hand-off, not a fork.
    assert!(
        !report(&reports, SHARD)
            .held_entities
            .iter()
            .any(|(e, _)| *e == subject),
        "the planet let go when the star took over",
    );
}

/// THE DOWNWARD CROSSING — the direction the owner flew and found broken, and the one direction this
/// suite has never covered.
///
/// Entering is NOT the mirror of leaving. Leaving needs the parent to ADD a number only it holds; entering
/// needs the parent to SUBTRACT that same number BEFORE it hands the occupant over, because the child must
/// be able to accept the pose without doing any arithmetic about itself. Two different obligations, and
/// only one of them had a gate.
///
/// THE OWNER'S SYMPTOM, IN NUMBERS: crossing INTO the planet lands the player far outside it, while
/// leaving lands them correctly. That is exactly what a MISSING subtraction looks like — the star's number
/// (25 m from the star) worn as though it were the planet's (25 m from the planet's centre) on a realm
/// whose whole boundary is 10 m across. The player is then outside the realm that just accepted them, so
/// the detector hands them straight back and the two shards trade them for ever.
///
/// The occupant is walked OUT first, so the entry is made FROM the star, BY the star, on the same cluster
/// and the same saga as the upward leg — the round trip the owner actually flew, not a fresh login placed
/// by hand.
#[test]
fn an_occupant_entering_a_realm_lands_inside_it_measured_from_its_centre() {
    let fabric = FaultFabric::new(4243, 2);
    let (mut topo, subject) = boot_planet_and_star(&fabric);

    // LEG 1 — OUT to the star (the covered direction), so the occupant ends up held by the one party that
    // authors where the planet sits.
    let out = DVec3::new(OCCUPANT_FROM_PLANET_M, 0.0, 0.0);
    let left = step_until(&mut topo, 400, |t| {
        set_shard_subject_offset(t, SHARD, subject, out);
        held_pose(&t.inspect_all(), DEST, subject).is_some()
    });
    assert!(left, "the occupant reaches its star first");
    assert!(
        step_until(&mut topo, 200, |t| live_sagas(t) == 0),
        "the outward hand-off settles before the inward one begins",
    );

    // LEG 2 — back IN. Place the occupant in the STAR's frame, well inside the planet's boundary: the star
    // put that planet 20 m out and it is 10 m across, so 25 m from the star is 5 m from the planet's
    // centre. CAPTURE THE ARRIVAL POSE on the first tick the planet reports holding it — this gate is
    // about the number computed on receipt, and reading later would let the planet's own simulation move
    // it first.
    let inward = DVec3::new(OCCUPANT_ENTERS_FROM_STAR_M, 0.0, 0.0);
    let mut arrival: Option<vd_core::pose::StampedPose> = None;
    let landed = step_until(&mut topo, 400, |t| {
        set_shard_subject_offset(t, DEST, subject, inward);
        if arrival.is_none() {
            arrival = held_pose(&t.inspect_all(), SHARD, subject);
        }
        arrival.is_some()
    });
    assert!(
        landed,
        "the occupant re-homes from the star into its planet"
    );
    let arrived = arrival.expect("just captured");

    // The composition proof (anti-vacuity, causal): the star's own detector emitted the crossing and the
    // orchestrator turned it into a real saga. There is no `trigger_transfer` in this file.
    let reports = topo.inspect_all();
    assert!(
        report(&reports, DEST).crossings_requested >= 1,
        "the star's containment detector emitted the inward crossing itself: {}",
        report(&reports, DEST).crossings_requested,
    );

    // (1) THE FRAME: the planet measures the arrival from its OWN centre.
    assert_eq!(
        arrived.frame,
        FrameRef::PlanetCentered { planet_seed: 7 },
        "the arrival is measured from the planet's centre",
    );
    // (2) THE NUMBER, which is the whole gate: the star SUBTRACTED where it put its planet. 25 − 20.
    // Exact: every quantity in this fixture is exact in binary at this scale, so a tolerance here would be
    // hiding something.
    assert_eq!(
        arrived.pos.offset(),
        DVec3::new(OCCUPANT_INSIDE_PLANET_M, 0.0, 0.0),
        "the star subtracts its child's placement: {OCCUPANT_ENTERS_FROM_STAR_M} − {PLANET_FROM_STAR_M}. \
         Reporting {OCCUPANT_ENTERS_FROM_STAR_M} here is the relabel bug in the inward direction: the \
         frame changed, the number did not, and the player stands {OCCUPANT_ENTERS_FROM_STAR_M} m from \
         the centre of a realm 10 m across — outside the realm that just accepted them.",
    );

    // (3) AND IT STAYS. An occupant that lands outside the realm that accepted it is outside its boundary
    // on the very next tick, so the two shards hand it back and forth. One hand-off, then quiet.
    assert!(
        step_until(&mut topo, 200, |t| live_sagas(t) == 0),
        "the inward hand-off settles instead of flapping: saga_states={:?}",
        vd_tests::saga_states(&mut topo),
    );
    let reports = topo.inspect_all();
    assert!(
        !report(&reports, DEST)
            .held_entities
            .iter()
            .any(|(e, _)| *e == subject),
        "the star let go when the planet took over",
    );
}

/// THE SAME CROSSING, INTO A REALM THAT MOVES — the combination the owner flew and nothing in this tree
/// tests.
///
/// WHY THE STATIC TWIN ABOVE CANNOT CATCH THIS. When the child stands still, the number the star subtracts
/// is the same at every instant, so it does not matter WHICH instant the subtraction is made at. The moment
/// the child moves, the two shards can disagree about where it was, and the occupant lands somewhere that
/// was right a moment ago. Standing still hides the entire class.
///
/// WHAT WENT WRONG ON THE REAL CLUSTER, in the shard's own logs: the star and the planet traded one player
/// back and forth FOURTEEN times, about three times a second. An occupant placed OUTSIDE the boundary of
/// the realm that just accepted it is outside that realm on the very next tick, so the realm hands it back,
/// the star sees it inside again, and the two never converge. From the seat that reads as "I re-homed into
/// the planet and arrived far outside it".
///
/// SO THIS ASSERTS THE PROPERTY, NOT A LITERAL. With a moving child the correct arrival is not one number —
/// it depends on where the planet was when the star did the arithmetic — so pinning a literal would pin one
/// scheduling. What must hold at EVERY instant is the thing the flap violates: an occupant the planet has
/// accepted is INSIDE the planet. That is checkable, it is what the player experiences, and it cannot pass
/// by luck.
///
/// The planet is given its turn on the STAR only. The planet's own shard is never told that it moves —
/// under the ground rule it cannot be, and this gate would be meaningless if it were.
#[test]
fn an_occupant_entering_a_realm_that_moves_lands_inside_it_and_stays() {
    let fabric = FaultFabric::new(4244, 2);
    let (mut topo, subject) = boot_planet_and_star(&fabric);

    // GIVE THE PLANET ITS TURN — on the star, the only party entitled to know where its child is.
    plant_seed_neighbourhood_with_movers(
        &mut topo,
        DEST,
        FRAME_UNIVERSE_SEED,
        SYSTEM,
        BTreeMap::from([(PLANET, planet_orbit())]),
    );

    // LEG 1 — OUT to the star, so the occupant is held by the party that authors where the planet sits.
    let out = DVec3::new(OCCUPANT_FROM_PLANET_M, 0.0, 0.0);
    let left = step_until(&mut topo, 400, |t| {
        set_shard_subject_offset(t, SHARD, subject, out);
        held_pose(&t.inspect_all(), DEST, subject).is_some()
    });
    assert!(left, "the occupant reaches its star first");

    // LEG 2 — WAIT ON THE ORBIT. Hold the occupant at the planet's own epoch point, in the star's frame:
    // the planet is on that shell for its whole turn, so it sweeps over the waiting occupant once a period.
    // This is the real cluster's shape — the child arrives at the occupant, not the other way round.
    let wait_at = DVec3::new(PLANET_FROM_STAR_M, 0.0, 0.0);
    let mut arrival: Option<vd_core::pose::StampedPose> = None;
    let landed = step_until(&mut topo, 800, |t| {
        set_shard_subject_offset(t, DEST, subject, wait_at);
        if arrival.is_none() {
            arrival = held_pose(&t.inspect_all(), SHARD, subject);
        }
        arrival.is_some()
    });
    assert!(
        landed,
        "the planet sweeps over the waiting occupant and takes it",
    );
    let arrived = arrival.expect("just captured");
    let from_centre = arrived.pos.offset().length();
    println!(
        "[moving arrival] frame {:?}, {from_centre:.6} m from the planet's centre (boundary \
         {PLANET_BOUNDARY_R_M} m)",
        arrived.frame,
    );

    // (1) THE FRAME: measured from the planet's own centre.
    assert_eq!(
        arrived.frame,
        FrameRef::PlanetCentered { planet_seed: 7 },
        "the arrival is measured from the planet's centre",
    );
    // (2) INSIDE. The one property a hand-off must never violate: the realm that accepted the occupant
    // contains it. Everything the owner saw follows from breaking this.
    assert!(
        from_centre < PLANET_BOUNDARY_R_M,
        "the occupant landed {from_centre} m from the planet's centre, OUTSIDE its {PLANET_BOUNDARY_R_M} m \
         boundary. The planet's own detector sees that on the next tick and hands it straight back — the \
         flap. This is the moving-child case: the star subtracted where its planet was at some instant \
         other than the one the occupant's position was taken at.",
    );
    // (3) AND IT STAYS — one hand-off, then quiet. A flap shows up here as sagas that never stop.
    assert!(
        step_until(&mut topo, 300, |t| live_sagas(t) == 0),
        "the hand-off settles instead of flapping: saga_states={:?}",
        vd_tests::saga_states(&mut topo),
    );
}

/// THE CLIENT'S HALF OF A CROSSING — and the one the real cluster fails.
///
/// Every gate above asks a SHARD where the occupant is. That is the authority's answer, and on the live
/// cluster it is RIGHT: the shard logs show the star and the planet handing one player back and forth
/// fourteen times, so the crossings fire and authority moves. What the owner actually experienced was the
/// other half — after re-homing into the planet he was still drawn in the STAR's space, far outside the
/// planet he was standing in — and the process gate reports the same thing in its own words: the client's
/// reported location never becomes the planet, for the whole run.
///
/// A player's location is the frame their OWN delivered position is measured in. So this asks the CLIENT,
/// through the same delivered view the renderer draws from: after the planet takes you, are you told?
///
/// ANTI-VACUITY: the client must first be told it is at the STAR. Without that half, a client that is
/// never told anything at all would pass the inward half by never having been wrong.
#[test]
fn the_client_is_told_the_realm_it_is_standing_in_after_a_crossing() {
    let fabric = FaultFabric::new(4245, 2);
    let (mut topo, subject) = boot_planet_and_star(&fabric);

    // OUT to the star.
    let out = DVec3::new(OCCUPANT_FROM_PLANET_M, 0.0, 0.0);
    assert!(
        step_until(&mut topo, 400, |t| {
            set_shard_subject_offset(t, SHARD, subject, out);
            held_pose(&t.inspect_all(), DEST, subject).is_some()
        }),
        "the occupant reaches its star",
    );
    assert!(
        step_until(&mut topo, 200, |t| live_sagas(t) == 0),
        "the outward hand-off settles",
    );

    // THE ANTI-VACUITY HALF: the client is told it is at the star.
    let star_frame = FrameRef::SystemSpace { system_seed: 7 };
    assert!(
        step_until(&mut topo, 400, |t| {
            set_shard_subject_offset(t, DEST, subject, out);
            with_client(t, |c| c.delivered_view.own_location_frame()) == Some(star_frame)
        }),
        "the client is told it is at the star — without this the second half proves nothing",
    );

    // BACK IN: the star hands the occupant to its planet.
    let inward = DVec3::new(OCCUPANT_ENTERS_FROM_STAR_M, 0.0, 0.0);
    assert!(
        step_until(&mut topo, 400, |t| {
            set_shard_subject_offset(t, DEST, subject, inward);
            held_pose(&t.inspect_all(), SHARD, subject).is_some()
        }),
        "the planet takes the occupant",
    );

    // THE GATE. The planet owns the player; the player must be TOLD, within a bounded window, that the
    // position they are being drawn at is measured from the planet's centre. Until that happens the
    // renderer is drawing them in the star's space — which is a player standing on a planet, drawn beside
    // it. Whatever the last shard to speak was, the client's own row must end up in the owner's frame.
    let planet_frame = FrameRef::PlanetCentered { planet_seed: 7 };
    let told = step_until(&mut topo, 400, |t| {
        set_shard_subject_offset(t, DEST, subject, inward);
        with_client(t, |c| c.delivered_view.own_location_frame()) == Some(planet_frame)
    });
    let ended_at = with_client(&mut topo, |c| c.delivered_view.own_location_frame());
    assert!(
        told,
        "the planet owns the player and the client was never told: its own delivered position is still \
         measured in {ended_at:?}, so it is drawn in that realm's space while standing in another. This \
         is the owner's symptom — re-homed into the planet, drawn far outside it.",
    );
}

// ---- THE REAL WORLD'S PROPORTIONS ---------------------------------------------------------------
//
// Every gate above runs on the walk world, whose planet is a TEN-METRE body sitting still twenty metres
// from its star. The world the game actually boots is not that: a 150 m star system holding planets whose
// whole authority sphere is about FOUR metres, orbiting seventeen metres out at four and a half metres a
// second. The numbers below are that world's, taken from its own generator and from a live cluster run.
//
// WHY IT MATTERS, and it is the reason a green suite sat beside an unplayable game all day: the entry
// margin is a FIXED one metre. On a ten-metre planet that is a tenth of the body and an arrival can be a
// metre or two out and still land comfortably inside. On a four-metre planet it is a quarter, and the same
// arithmetic error puts the occupant OUTSIDE the realm that just accepted it — whereupon that realm hands
// it straight back, the parent hands it down again, and the player is trapped in the loop the owner flew.
//
/// The star system's own boundary in the shipped demand world.
const DEMAND_SYSTEM_SOI_M: f64 = 150.0;
/// A planet's WHOLE authority sphere there — a quarter the size of the walk world's, against the SAME
/// one-metre entry margin. This is the number that turns a tolerable arrival error into a trap.
const DEMAND_PLANET_SOI_M: f64 = 4.16;
/// How far out the inner planet orbits (`epoch_len` printed by the live demand gate).
const DEMAND_PLANET_ORBIT_M: f64 = 17.9;
/// Its period, derived from the speed the live cluster measured (13.27 m in 3 s ⇒ 4.42 m/s over a
/// circumference of 2·pi·17.9), so the planet sweeps the waiting occupant at the speed it really does.
const DEMAND_ORBIT_PERIOD_S: f64 = 25.4;

/// The demand world's turn for the planet: circular, in-plane, phase at zero, on its real shell.
fn demand_orbit() -> OrbitalElements {
    let a = DEMAND_PLANET_ORBIT_M;
    let mu = 4.0 * std::f64::consts::PI * std::f64::consts::PI * a.powi(3)
        / (DEMAND_ORBIT_PERIOD_S * DEMAND_ORBIT_PERIOD_S);
    OrbitalElements {
        sma: a,
        ecc: 0.0,
        inclination: 0.0,
        raan: 0.0,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: 0.0,
        central_mass: mu / G,
    }
}

/// The demand world's shape, three levels of it: an ambient root, the star system, and the planet.
///
/// The planet's stored centre is ZERO because it MOVES — its position comes from the star's ephemeris
/// every tick, and a stored centre beside a live placement would be counted twice. That is the shipped
/// generator's own rule for an orbiting body, reproduced here rather than invented.
fn demand_forest() -> Vec<vd_core::geometry::RealmRegion> {
    use vd_core::geometry::{AoiConfig, Boundary, ContainmentBand, RealmRegion};
    // The SHIPPED band: one metre in to acquire, two metres out to release.
    let band = ContainmentBand::for_containment_velocity_safe(1.0, 2.0, 0.0, 1.0, 0.0)
        .expect("the shipped containment band is valid");
    let root = RealmId::System(1); // the ambient root, as the live cluster names it
    vec![
        RealmRegion {
            realm: root,
            center: vd_core::pose::LatticePos::local(DVec3::ZERO),
            frame: FrameRef::SystemSpace { system_seed: 1 },
            shape: Boundary::Shell { r: 1.0e6 },
            band,
            aoi: AoiConfig::inert(),
            parent: None,
        },
        RealmRegion {
            realm: SYSTEM,
            center: vd_core::pose::LatticePos::local(DVec3::ZERO),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            shape: Boundary::Shell {
                r: DEMAND_SYSTEM_SOI_M,
            },
            band,
            aoi: AoiConfig::inert(),
            parent: Some(root),
        },
        RealmRegion {
            realm: PLANET,
            // ZERO: an orbiting body is placed LIVE by its parent, never from a stored centre.
            center: vd_core::pose::LatticePos::local(DVec3::ZERO),
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            shape: Boundary::Shell {
                r: DEMAND_PLANET_SOI_M,
            },
            band,
            aoi: AoiConfig::inert(),
            parent: Some(SYSTEM),
        },
    ]
}

/// THE OWNER'S LOOP, at the proportions it actually happens at.
///
/// He flew into a planet and the planet and its star traded him back and forth until he could not get
/// out. Every gate in this file said the crossing was fine, because every gate in this file uses a planet
/// two and a half times larger than the real one against the same fixed entry margin.
///
/// This asks the same question the moving gate above asks, at the real world's proportions, and it asks
/// it about the ONE property the loop violates: an occupant a realm has ACCEPTED is INSIDE that realm. A
/// realm that accepts somebody standing outside itself sees them outside on its very next tick and hands
/// them back — and there is no state in between for the loop to stop at.
#[test]
fn entering_a_real_sized_planet_lands_inside_it_and_does_not_trade_the_player_back() {
    let fabric = FaultFabric::new(4246, 2);
    let (mut topo, subject) = boot_planet_and_star(&fabric);

    // The REAL world on both shards: the star authors its planet's orbit; the planet is never told that
    // it moves, because under the ground rule it cannot be.
    plant_regions(
        &mut topo,
        DEST,
        demand_forest(),
        BTreeMap::from([(PLANET, demand_orbit())]),
    );
    plant_regions(&mut topo, SHARD, demand_forest(), BTreeMap::new());

    // OUT to the star first, so the hand-off INTO the planet is made by the star, from the star.
    let out = DVec3::new(DEMAND_PLANET_SOI_M * 3.0, 0.0, 0.0);
    assert!(
        step_until(&mut topo, 600, |t| {
            set_shard_subject_offset(t, SHARD, subject, out);
            held_pose(&t.inspect_all(), DEST, subject).is_some()
        }),
        "the occupant reaches its star first",
    );

    // WAIT ON THE ORBIT: hold the occupant at the planet's epoch point, in the star's frame. The planet
    // is on that shell for its whole turn, so it sweeps over the waiting occupant — which is exactly how
    // the owner met it, and how the live demand gate drives it.
    let wait_at = DVec3::new(DEMAND_PLANET_ORBIT_M, 0.0, 0.0);
    let mut arrival: Option<vd_core::pose::StampedPose> = None;
    let landed = step_until(&mut topo, 1200, |t| {
        set_shard_subject_offset(t, DEST, subject, wait_at);
        if arrival.is_none() {
            arrival = held_pose(&t.inspect_all(), SHARD, subject);
        }
        arrival.is_some()
    });
    assert!(
        landed,
        "the planet sweeps over the waiting occupant and takes it"
    );
    let arrived = arrival.expect("just captured");
    let from_centre = arrived.pos.offset().length();
    println!(
        "[real-sized arrival] {from_centre:.6} m from the planet's centre, boundary \
         {DEMAND_PLANET_SOI_M} m, entry margin 1 m"
    );

    assert!(
        from_centre < DEMAND_PLANET_SOI_M,
        "the occupant landed {from_centre} m from the centre of a {DEMAND_PLANET_SOI_M} m planet — \
         OUTSIDE the realm that just accepted it. The planet sees that on its next tick and hands them \
         back; the star sees them inside its planet and hands them down; and the player cannot leave. \
         This is the owner's loop, and the walk-scale gates cannot see it because their planet is two \
         and a half times larger against the same one-metre margin.",
    );
    assert!(
        step_until(&mut topo, 600, |t| live_sagas(t) == 0),
        "one hand-off, then quiet — a loop shows up here as sagas that never stop: {:?}",
        vd_tests::saga_states(&mut topo),
    );
}

/// THE SIBLING SEAM, made loud. A pose handed straight from one planet to a SIBLING planet used to land
/// with `Ok` at every hop: no error, no counter, no log anywhere — it was simply relabelled, so the
/// occupant moved by the whole distance between the two planets and nothing noticed.
///
/// Nobody ever told a planet where its sibling is, so there is no arithmetic it could do. The arrival is
/// now REFUSED and counted, the subject is not applied, and — because the receiver never acks a step it
/// refused — the source keeps the entity.
///
/// This drives the SAME production ingress the crossing above does, with the star's shard as the
/// receiver and a pose stamped in a realm it was never told the position of.
#[test]
fn a_sibling_handover_is_refused_loudly() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, subject) = boot_planet_and_star(&fabric);

    // A frame the star's neighbourhood does not carry: `System(8)` is the seed forest's OTHER star, a
    // SIBLING of System 7 under the galaxy. The star hosts its own realm, its ancestors and its own
    // children — never a sibling — so it has no placement for this and cannot place anything expressed
    // in it.
    let sibling_frame = FrameRef::SystemSpace { system_seed: 8 };
    assert!(
        !vd_core::worldgen::realm_neighbourhood_for(FRAME_UNIVERSE_SEED, SYSTEM)
            .iter()
            .any(|r| r.frame == sibling_frame),
        "the fixture is only meaningful if the star genuinely has no region for its sibling",
    );

    let before = topo.inspect_all();
    let source_held_before = report(&before, SHARD)
        .held_entities
        .iter()
        .any(|(e, _)| *e == subject);
    assert!(
        source_held_before,
        "the planet holds the avatar before the mis-routed hand-off",
    );

    // Push the occupant's pose into the SIBLING's frame on the planet shard, then let it leave. The
    // planet ships what it holds, tag and all — it has nothing to convert into its parent's frame — so
    // the star receives a pose in a frame it cannot measure.
    let out = DVec3::new(OCCUPANT_FROM_PLANET_M, 0.0, 0.0);
    let refused = step_until(&mut topo, 400, |t| {
        set_shard_subject_offset(t, SHARD, subject, out);
        vd_tests::set_shard_subject_frame(t, SHARD, subject, sibling_frame);
        report(&t.inspect_all(), DEST).arrivals_unplaceable >= 1
    });

    let reports = topo.inspect_all();
    assert!(
        refused,
        "the star refuses a pose it cannot measure (arrivals_unplaceable): {}",
        report(&reports, DEST).arrivals_unplaceable,
    );
    assert!(
        !report(&reports, DEST)
            .held_entities
            .iter()
            .any(|(e, _)| *e == subject),
        "the refused subject is NOT applied on the star",
    );
    assert!(
        report(&reports, SHARD)
            .held_entities
            .iter()
            .any(|(e, _)| *e == subject)
            || report(&reports, SHARD)
                .departing_entities
                .contains(&subject),
        "the source still has the entity — a refused hand-off strands nothing",
    );
    // Nothing else about the world moved: no OTHER shard picked it up either.
    let holders: BTreeSet<NodeId> = [SHARD, DEST]
        .into_iter()
        .filter(|n| {
            report(&reports, *n)
                .held_entities
                .iter()
                .any(|(e, _)| *e == subject)
        })
        .collect();
    assert!(
        holders.len() <= 1,
        "a refused hand-off never produces two holders: {holders:?}",
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// MOVING-REALM COHERENCE AND THE PRECISION GATE
//
// The gates above run at walk scale, where nothing moves and every placement is at its parent's origin.
// Everything below runs where it is hard: a realm that orbits, and a world 1e13 m across.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The cluster tick rate the shipped deployment runs at (`VD_TICK_HZ` in the k3d recipes). Every number
/// below that is expressed in TICKS is expressed at this rate.
const SHIPPED_TICK_HZ: u32 = 50;

/// One millimetre. The quantity the FAR fixture carries, and the reason that fixture is 1e13 m across:
/// at that distance consecutive doubles are ~1.95 mm apart, so a millimetre put through a value of that
/// size does not survive.
const ONE_MM_M: f64 = 0.001;

/// A nanometre — the bound the metre-scale assertions below use.
///
/// It is not a physical tolerance, it is a "did any arithmetic happen at the wrong magnitude" detector.
/// One f64 ulp at the ~150 m magnitudes in play is about 3e-14 m, so a nanometre leaves five orders of
/// magnitude of headroom for legitimate re-association — and is still a MILLION times smaller than the
/// ~1 mm a trip through the galaxy's own scale costs. Anything that fails this has been somewhere it had
/// no business being.
const NANOMETRE_M: f64 = 1.0e-9;

/// The moving realm's orbital period in the coherence gate, in seconds.
///
/// NAMED rather than a mass constant: what is actually being chosen is how fast the box moves, and it has
/// to be fast enough that a tick of drift between the two feeds would be visible (so the gate can fail) and
/// slow enough that a whole test run is a small arc of one turn.
const COHERENCE_ORBIT_PERIOD_S: f64 = 20.0;

/// How far the box must be seen to travel between the two coherence samples before the gate is believed.
///
/// Without it a realm that had quietly stopped moving would satisfy "both feeds agree" trivially — two
/// feeds agreeing about a stationary object is the walk-scale case that proves nothing. Derived from the
/// motion itself: a tenth of the distance the realm covers in one shard tick.
const COHERENCE_MIN_TRAVEL_FRACTION: f64 = 0.1;

/// The bound the two render feeds must agree to, in metres — the same [`NANOMETRE_M`] detector, for the
/// same reason.
///
/// Both lanes are composed by ONE expression through ONE table, so the only difference between them is
/// float re-association — nanometres at these magnitudes. The fault this gate exists for is METRES: the
/// realm boxes once composed against the universe root while the occupants standing in them composed
/// against their own shard, and a box and its rider drew in spaces offset by the shard's own position.
/// MEASURED, the gap is exactly zero; a nanometre is the slack, not the expectation.
const DRAW_COINCIDENCE_TOL_M: f64 = NANOMETRE_M;

/// THE REALM THAT MOVES: the seed forest's Area 7, a patch of ground the planet carries around with it as
/// it turns. Given orbital elements it stops riding the static centre its region carries and starts being
/// authored live, once a tick, on the realm lane — which is the only lane that can say where a moving body
/// is. Nothing about the machinery branches on realm KIND; an area on a turning surface and a planet on an
/// orbit are the same shape of problem, and this one nests one level deeper, which is what the gate needs.
const MOVING_AREA: RealmId = RealmId::Area(7);
/// That area's own frame — the frame an occupant STANDING ON IT is measured in.
const MOVING_AREA_FRAME: FrameRef = FrameRef::AreaLocal {
    planet_seed: 7,
    area_seed: 7,
};
/// The number ONLY THE PLANET HOLDS: how far its area sits from the planet's own centre. The seed forest's
/// own authored offset for Area 7, written out as a literal — and the radius the turn is given, so the
/// moving area traces exactly the shell the static fixture puts it on and stays inside the planet's 10 m
/// boundary (5 m out, 3 m half-extent).
const AREA_FROM_PLANET_M: f64 = 5.0;

/// The turn given to [`MOVING_AREA`]: circular, in-plane, phase pinned to zero.
///
/// The central mass is chosen THROUGH the period (`T = 2π√(a³/μ)`, `μ = G·M`) rather than written down,
/// because the period is the thing being chosen and a mass constant is a number nobody can check.
fn coherence_orbit() -> OrbitalElements {
    let a = AREA_FROM_PLANET_M;
    let mu = 4.0 * std::f64::consts::PI * std::f64::consts::PI * a.powi(3)
        / (COHERENCE_ORBIT_PERIOD_S * COHERENCE_ORBIT_PERIOD_S);
    OrbitalElements {
        sma: a,
        ecc: 0.0,
        inclination: 0.0,
        raan: 0.0,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: 0.0,
        central_mass: mu / G,
    }
}

/// What one client has been DELIVERED about the moving realm and about itself, at one instant.
#[derive(Clone, Copy, Debug)]
struct DrawnPair {
    /// The universe tick both feeds were sampled at.
    at: UniverseTick,
    /// Where the client would draw the realm's own box centre.
    box_m: DVec3,
    /// Where the client would draw the occupant standing at that realm's local origin.
    occupant_m: DVec3,
}

/// Read the two render feeds at ONE cursor, or `None` until both have delivered the same instant.
///
/// The cursor matters as much as the numbers. The two lanes are independent unreliable datagrams, so at any
/// wall-clock moment one is usually a tick ahead of the other; reading each at "whatever arrived last"
/// compares two different instants and measures the motion rather than the composition. This waits for a
/// tick BOTH feeds have delivered and samples both there, through the production interpolator.
fn drawn_pair(client: &ScriptedClient) -> Option<DrawnPair> {
    let entity = client.delivered_view.own_entity()?;
    let entity_tick = client.delivered_view.newest_tick()?;
    let realm_tick = client.realm_view.realm_newest_tick(MOVING_AREA)?;
    if entity_tick != realm_tick {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    // a universe tick index is far inside f64's exact-integer range
    let cursor = entity_tick.0 as f64;
    let occupant = client.delivered_view.render(cursor).get(&entity).copied()?;
    let realm_box = client.realm_view.realm_pose(MOVING_AREA, cursor)?;
    Some(DrawnPair {
        at: entity_tick,
        box_m: client.delivered_view.world_pos(&realm_box),
        occupant_m: client.delivered_view.world_pos(&occupant),
    })
}

/// (a) THE ANTI-DRIFT GATE — a moving realm's box and the rider standing at its centre must reduce to
/// ONE drawn point, under the SETTLED occupied-realm model (rehome_one_mechanism §4c/§4x, owner-decided).
///
/// THE FAULT THIS CATCHES ALREADY HAPPENED. The two render feeds were once produced in two different
/// spaces: the realm boxes measured from the universe root, the occupants standing in them measured from
/// their own shard's position. Each looked right on its own, and the pair drew in spaces offset by the
/// shard's own position — so the ground slid out from under the player. MEASURED at the client's own
/// consumers (2026-08-12): `realm row frame PlanetCentered{7} pos (4.97, 0.55, 0)` beside `occupant
/// frame AreaLocal{7,7} pos (0,0,0)` — a 5 m gap from one shard in one tick.
///
/// THE SETTLED MODEL this gate now asserts, in two halves:
///
/// **(1) THE PARENT-OBSERVER HALF — the anti-drift teeth.** The observer stands in the PLANET (the
/// area's parent) and rides the moving area's live placement, scripted each tick from the SAME
/// elements the shard authors with, at the SAME tick the pose is stamped at. Both lanes flow — the
/// realm row for the area (authored by the planet, shipped to a planet-standing observer as authored)
/// and the entity row for the rider — and they must reduce to one drawn point at a shared instant.
/// Neither lane is scripted; only the rider's position is.
///
/// **(2) THE OCCUPIED-REALM HALF — the one-space rule.** An occupant standing IN the moving area holds
/// NO streamed track for that area, ever: the realm you occupy ships no per-tick row for itself (its
/// centre in its own frame is the origin forever — SL3: it draws itself, as its outline, at your
/// origin), and the client's one-space ingress folds no row stated in another space. The stale
/// pre-crossing track that drew the ground 5 m from the player (the measured live defect) is therefore
/// UNREPRESENTABLE. The outline-at-origin drawing half is pinned by the vd-client scene tests (the
/// harness client carries no scene lane); what this e2e pins is the track's guaranteed absence.
#[test]
fn the_box_and_the_thing_standing_in_it_draw_at_one_point() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, subject) = boot_planet_and_star(&fabric);

    // MAKE THE AREA MOVE. The seed forest's realms are all static, and a static realm makes every
    // conversion the identity — which is the case that proves nothing.
    plant_seed_neighbourhood_with_movers(
        &mut topo,
        SHARD,
        FRAME_UNIVERSE_SEED,
        PLANET,
        BTreeMap::from([(MOVING_AREA, coherence_orbit())]),
    );

    let planet_frame = FrameRef::PlanetCentered { planet_seed: 7 };
    let elements = coherence_orbit();
    let tick_hz = 1.0 / vd_tests::stub_config().tick_dt_s;
    let area_at = move |tick: vd_core::UniverseTick| {
        vd_core::celestial::orbital_state(
            &elements,
            vd_core::celestial::secs_since_epoch(tick.0, tick_hz),
        )
        .position
    };
    // The rider co-moves BESIDE the area, radially outside its shell — never inside it (a rider AT
    // the centre is genuinely contained, and the detector rightly re-homes them in, which flips
    // their frame mid-sample; measured on the first cut of this rewrite). 1.7 × the 5 m orbit puts
    // the rider 3.5 m from the area's centre (shell half-extent 3 m, acquire needs ≥ 1 m INSIDE —
    // structurally unreachable at a constant 3.5 m) and 8.5 m from the planet's centre (10 m
    // boundary — solidly planet-contained). The rider's drawn point is then EXACTLY the box's
    // scaled by the same factor: one closed form, two lanes.
    const RIDE_FACTOR: f64 = 1.7;

    // ── HALF (1): the rider in the PLANET frame, riding beside the area's live placement, every tick. ──
    let mut samples: Vec<DrawnPair> = Vec::new();
    let got = step_until(&mut topo, 600, |t| {
        set_shard_subject_pose_now_with(t, SHARD, subject, planet_frame, |at| {
            area_at(at) * RIDE_FACTOR
        });
        let next = with_client(t, |c| drawn_pair(c));
        // Sample only once the SCRIPTED pose has round-tripped to the client: the login pose is the
        // planet's origin, and a boot-instant pair would measure the pre-script state (measured on
        // the first cut of this rewrite: an origin occupant beside a live box, gap = the radius).
        if let Some(pair) = next
            && pair.occupant_m.length() > 1.0
            && samples.last().is_none_or(|last| last.at != pair.at)
        {
            samples.push(pair);
        }
        samples.len() >= 2
    });
    assert!(
        got,
        "the client is delivered both render feeds for the moving area at a common instant: {} sample(s)",
        samples.len(),
    );

    // ANTI-VACUITY 1: the box must actually be MOVING between the two samples, or "both feeds agree" is the
    // walk-scale statement that two feeds agree about something standing still.
    let travelled = (samples[1].box_m - samples[0].box_m).length();
    // The reference is one SHARD tick of the turn, taken from the shard's own `tick_dt_s` — the rate the
    // ephemeris is actually sampled at in this cluster, not the shipped 50 Hz the accuracy bound above is
    // stated at. Reading it off the wrong clock makes this guard either vacuous or impossible to satisfy.
    let per_tick = std::f64::consts::TAU * AREA_FROM_PLANET_M / COHERENCE_ORBIT_PERIOD_S
        * vd_tests::stub_config().tick_dt_s;
    println!(
        "[coherence] ticks {:?}->{:?}  box travelled {travelled:.6} m  (one shard tick of the turn ~ {per_tick:.6} m)",
        samples[0].at, samples[1].at,
    );
    assert!(
        travelled > per_tick * COHERENCE_MIN_TRAVEL_FRACTION,
        "the area's box is genuinely turning between the samples: {travelled} m",
    );

    for s in &samples {
        let gap = (s.box_m * RIDE_FACTOR - s.occupant_m).length();
        println!(
            "[coherence] at {:?}  box {:?}  occupant {:?}  gap {gap:.3e} m",
            s.at, s.box_m, s.occupant_m,
        );
        // The rider is SCRIPTED at the area's closed-form placement × RIDE_FACTOR while the box
        // arrives through the live feed's stamped rows — the same closed form at the same tick, down
        // two independent lanes, so the scaled drawn difference is float re-association at ~10 m
        // magnitudes. A gap in METRES means the lanes composed in two spaces or at two instants —
        // fix the composition, never the tolerance.
        assert!(
            gap <= DRAW_COINCIDENCE_TOL_M,
            "the realm box and the rider beside it draw in ONE space: gap {gap} m at {:?}",
            s.at,
        );
        // ANTI-VACUITY 2 (the settled model's collapse guard): the row folded because it is STATED in
        // the observer's own space — the planet's, where the area orbits at ~5 m. A row that arrived
        // still measured from the STAR (the pre-§4c defect: ~15-25 m out) must have been skipped by
        // the one-space ingress, never folded; a drawn radius near the star's magnitude here means the
        // one-space rule stopped filtering.
        let r = s.box_m.length();
        assert!(
            (r - AREA_FROM_PLANET_M).abs() <= 1.0,
            "the drawn box orbits at the area's ~{AREA_FROM_PLANET_M} m radius in the OBSERVER'S space: {r} m",
        );
    }

    // ── HALF (2): THE TEST'S NAME, LITERALLY — the occupant STANDS IN the moving area, and the ──
    // box and the thing standing in it draw at ONE point, LIVE. The co-hosting shard authors the
    // area's placement AND holds the dot, so it serves this observer ONE space end to end: the
    // entity lane lifts the area-local pose through the placement it authors, and the realm lane —
    // grouped by the same delivered space (the cross-lane alignment this slice landed) — ships the
    // area's row in that space too. The 5 m split of the measured defect (`realm row
    // PlanetCentered{7} (4.97, 0.55, 0)` beside `occupant AreaLocal{7,7} (0,0,0)`) is closed: one
    // space, one point, and the box keeps MOVING while stood upon (the pre-alignment feed dried up
    // here — a frozen box under a riding player).
    let mut standing: Vec<DrawnPair> = Vec::new();
    let got = step_until(&mut topo, 600, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, MOVING_AREA_FRAME, DVec3::ZERO);
        let next = with_client(t, |c| drawn_pair(c));
        if let Some(pair) = next
            // Gate on the STANDING-IN state having round-tripped: the lifted occupant draws within
            // a metre of the box (never true of half (1)'s rider at 1.7 × the radius).
            && (pair.occupant_m - pair.box_m).length() <= 1.0
            && standing.last().is_none_or(|last| last.at != pair.at)
        {
            standing.push(pair);
        }
        standing.len() >= 2
    });
    assert!(
        got,
        "the standing-in occupant and the area's live box reach the client in one space: {} sample(s)",
        standing.len(),
    );
    let travelled = (standing[1].box_m - standing[0].box_m).length();
    assert!(
        travelled > per_tick * COHERENCE_MIN_TRAVEL_FRACTION,
        "the OCCUPIED area's box keeps turning while stood upon: {travelled} m — a frozen box under \
         a riding player is the pre-alignment starved feed",
    );
    for s in &standing {
        let gap = (s.box_m - s.occupant_m).length();
        println!(
            "[standing] at {:?}  box {:?}  occupant {:?}  gap {gap:.3e} m",
            s.at, s.box_m, s.occupant_m,
        );
        assert!(
            gap <= DRAW_COINCIDENCE_TOL_M,
            "the realm box and the thing standing in it draw at ONE point: gap {gap} m at {:?} — \
             the measured 5 m two-spaces split reborn; fix the composition, never the tolerance.",
            s.at,
        );
    }
}

/// (b) THE PRECISION GATE — a full round trip through the galaxy, on the FAR fixture, must return the
/// occupant to the metre it left.
///
/// The story: the neighbour star system is 1e13 m out, the planet 145 m from its star, the occupant 3.001 m
/// above the planet. The trip is PLANET → SYSTEM → GALAXY → SYSTEM → PLANET, every hop through the
/// production `transfer_frame` over the production `frame_context`, each addition made by the one party
/// that holds that number.
///
/// THE CONTROL IS PART OF THE PROOF. It computes what the removed fold computed — inflate the occupant to
/// the galaxy's own magnitude and take the difference — and demands that it DESTROY at least half the
/// millimetre the story carries. If that ever stops being true the fixture has drifted below the precision
/// cliff and this gate has become a green no-op, so the test fails loudly on that rather than quietly
/// stopping testing anything. "Not exactly equal" would not do: a fold loses an ulp at any magnitude.
///
/// HONESTY, AND IT IS WHY [`the_render_path_never_visits_the_galaxy`] SITS NEXT TO THIS ONE. Measured, the
/// round trip and the control return the SAME number — 3.001953125 m for an occupant that departed from
/// 3.001 m, a loss of 0.953 mm. A trip that genuinely visits the galaxy's frame costs an ulp at galaxy
/// scale, and there is no arrangement of the additions that avoids it: the occupant really is 1e13 m from
/// the galaxy's centre at the top of the walk, and 1e13 m cannot hold a millimetre. It comes back inside a
/// millimetre, but only just, and that is the whole point stated in the other direction — the arc's claim
/// is not that this trip is cheap, it is that NOBODY TAKES IT to draw a player standing on a planet. The
/// render path stops at the nearest common ancestor, which for a player and their own star system is the
/// star system, and the neighbouring test measures that it loses nothing at all.
#[test]
fn a_round_trip_returns_the_occupant_to_the_metre_it_left() {
    use vd_core::frame::transfer_frame;

    let fx = WorkedExample::far();
    let start = fx.occupant_pose();

    // UP: the planet ships "3.001, in my frame"; the SYSTEM adds where it put that planet.
    let at_system = transfer_frame(&start, fx.system_frame, &fx.frame_context(fx.system))
        .expect("the star places its own planet");
    // UP: the system ships "148.001, in my frame"; the GALAXY adds where it put that system.
    let at_galaxy = transfer_frame(&at_system, fx.galaxy_frame, &fx.frame_context(fx.galaxy))
        .expect("the galaxy places its own system");
    // DOWN: the GALAXY subtracts — it is the only party that holds the 1e13.
    let back_system = transfer_frame(&at_galaxy, fx.system_frame, &fx.frame_context(fx.galaxy))
        .expect("the galaxy places its own system");
    // DOWN: the SYSTEM subtracts, and the planet then accepts the result and does no arithmetic at all.
    let back_planet = transfer_frame(&back_system, fx.planet_frame, &fx.frame_context(fx.system))
        .expect("the star places its own planet");

    let departed = start.pos.offset().x;
    let returned = back_planet.pos.offset().x;
    let err = (returned - departed).abs();
    println!(
        "[far round trip] departed {departed:.9} m  returned {returned:.9} m  error {err:.9} m"
    );
    println!(
        "[far round trip] at the galaxy the occupant is legitimately {:.3} m out; one double step there is \
         {:.9} m",
        at_galaxy.pos.offset().x,
        (FAR_SYSTEM_FROM_GALAXY_M + ONE_MM_M) - FAR_SYSTEM_FROM_GALAXY_M,
    );
    assert_eq!(back_planet.frame, fx.planet_frame);
    assert!(
        err <= ONE_MM_M,
        "the occupant comes back to the millimetre it left: departed {departed}, returned {returned}, \
         error {err} m",
    );

    // THE CONTROL — the removed fold's arithmetic, written out. If this ever equals the occupant's height
    // the fixture is no longer above the precision cliff and the gate above has gone to sleep.
    let folded =
        ((FAR_SYSTEM_FROM_GALAXY_M + NEAR_PLANET_FROM_STAR_M + FAR_OCCUPANT_FROM_PLANET_M)
            - FAR_SYSTEM_FROM_GALAXY_M)
            - NEAR_PLANET_FROM_STAR_M;
    println!(
        "[control] folding through the galaxy's own magnitude returns {folded:.9} m for an occupant at \
         {FAR_OCCUPANT_FROM_PLANET_M} m — a loss of {:.9} m",
        (folded - FAR_OCCUPANT_FROM_PLANET_M).abs(),
    );
    // "Not exactly equal" is too weak to be a guard: at ANY magnitude the fold loses an ulp or two, so a
    // fixture shrunk for speed would still satisfy it while testing nothing. The demand is that the fold
    // destroy at least HALF the millimetre the story carries — i.e. that the quantity really is below the
    // representable step out there, which is the fixture's entire reason to exist.
    let control_loss = (folded - FAR_OCCUPANT_FROM_PLANET_M).abs();
    assert!(
        control_loss >= ONE_MM_M / 2.0,
        "THE FIXTURE HAS DRIFTED BELOW THE PRECISION CLIFF. Inflating the occupant to \
         {FAR_SYSTEM_FROM_GALAXY_M} m and back is supposed to destroy its millimetre, and it only cost \
         {control_loss} m; below half a millimetre this whole gate is green for free. Move the FAR \
         distance back out, do not relax this.",
    );
}

/// The universe seeds the chord measurement sweeps. One seed's forest is one draw; the bound the shipped
/// constant is set from has to hold for the worlds the generator actually produces, not for one of them.
const SKEW_SEED_SWEEP: u64 = 16;
/// The cluster tick rates that ship in this tree: 50 Hz is what the k3d recipes set `VD_TICK_HZ` to, and
/// 20 Hz is the `tick_dt_s = 0.05` every shard fixture runs at. The bound is rate-dependent — the reference
/// it is measured against (one tick of the realm's own motion) shrinks with the tick — so BOTH are checked.
const SHIPPED_TICK_RATES: [u32; 2] = [20, 50];
/// The tick offsets the per-mover chord table prints, so the shape of the error (quadratic in the skew) is
/// visible in the output rather than inferred from one number.
const CHORD_PROBE_TICKS: [u64; 4] = [1, 2, 5, 20];
/// How far the admissible-skew search walks before giving up. Far past any orbit that could be admitted, so
/// hitting it would itself be the finding.
const CHORD_SEARCH_CEILING: u64 = 2_000;

/// The largest whole-tick skew, at `tick_hz`, for which the straight-line carry of `e` stays closer to the
/// closed-form orbit than the distance that body moves in ONE tick.
///
/// The reference is the body's OWN per-tick motion on purpose. It makes the criterion scale-free — it holds
/// the same way for a 12-second demo orbit and a real planetary year — and it needs no invented tolerance,
/// because it compares the approximation against the resolution of the thing being approximated.
fn admissible_skew_ticks(e: &OrbitalElements, tick_hz: u32) -> u64 {
    let dt_s = 1.0 / f64::from(tick_hz);
    let base = orbital_state(e, 0.0);
    let per_tick = base.velocity.length() * dt_s;
    let mut admissible = 0u64;
    for ticks in 1..=CHORD_SEARCH_CEILING {
        #[allow(clippy::cast_precision_loss)] // a tick count in the thousands
        let secs = ticks as f64 * dt_s;
        let chord =
            ((base.position + base.velocity * secs) - orbital_state(e, secs).position).length();
        if chord > per_tick {
            break;
        }
        admissible = ticks;
    }
    admissible
}

/// Every orbiting body the shipped presets generate across the seed sweep.
fn shipped_movers() -> Vec<(String, RealmId, OrbitalElements)> {
    let mut movers = Vec::new();
    for seed in 0..SKEW_SEED_SWEEP {
        for (name, config) in [
            ("walk_scale", UniverseConfig::walk_scale()),
            ("visual_scale", UniverseConfig::visual_scale()),
        ] {
            let world = WorldView::generated(seed, &config);
            for region in world.regions() {
                for (realm, elements) in moving_children_for_config(seed, &config, region.realm) {
                    movers.push((format!("{name} seed {seed}"), realm, elements));
                }
            }
        }
    }
    movers
}

/// (c) THE MEASUREMENT THAT SIZES THE SKEW CAP.
///
/// CARRYING a moving realm's placement from the instant its author stamped it to the instant it is being
/// used at means walking a STRAIGHT LINE at the authored velocity. A straight line is a chord across an
/// arc, so it is only honest over a short arc — and how short "short" is had never been measured. Until
/// this test, the budget was an arrival-skew number wearing an accuracy label, and it was nearly twice
/// what the orbits allow.
///
/// The bound now lives with the party that AUTHORS the velocity (`vd_sim::stub::PlacementCarry`), not on
/// the router's transport tuning — a router cannot decide how far a simulation may extrapolate, and this
/// one no longer applies a placement at all. No lane carries linearly today (every level resolves its own
/// child's placement closed-form at the row's own instant), so the cap is currently unreachable by
/// construction; the measurement is kept because the first checkpoint-carried or signal-driven body brings
/// the carry back, and this is the bound it must respect.
///
/// THE BOUND IS DERIVED, not chosen: the error the carry introduces over the WHOLE skew window must stay
/// below the distance the realm itself moves in ONE tick. Below that the carry is worth less than the
/// quantisation of the feed that produced it; above it the carry is inventing motion. That criterion needs
/// no invented constant — it compares the mechanism against its own input — and it is what
/// [`admissible_skew_ticks`] searches for.
///
/// Everything here is measured against the closed-form `orbital_state`, over every mover the shipped
/// presets generate across sixteen seeds, at both shipped tick rates. The test RE-MEASURES on every run, so
/// the constant cannot creep back past the bound without this going red.
#[test]
fn the_linear_carry_chord_error_sizes_the_placement_skew_cap() {
    let movers = shipped_movers();
    assert!(
        !movers.is_empty(),
        "the shipped presets produce at least one ORBITING body — with none, this measures nothing",
    );

    // The shape of the error, printed for the tightest and the slackest body at the shipped rate, so the
    // quadratic growth is visible in the output rather than taken on trust.
    let hz = SHIPPED_TICK_HZ;
    let dt_s = 1.0 / f64::from(hz);
    let mut by_admissible: Vec<(u64, &(String, RealmId, OrbitalElements))> = movers
        .iter()
        .map(|m| (admissible_skew_ticks(&m.2, hz), m))
        .collect();
    by_admissible.sort_by_key(|(ticks, _)| *ticks);
    for (ticks, (name, realm, e)) in [
        by_admissible.first().expect("at least one mover"),
        by_admissible.last().expect("at least one mover"),
    ] {
        let base = orbital_state(e, 0.0);
        let per_tick = base.velocity.length() * dt_s;
        for probe in CHORD_PROBE_TICKS {
            #[allow(clippy::cast_precision_loss)] // a handful of ticks
            let secs = probe as f64 * dt_s;
            let chord =
                ((base.position + base.velocity * secs) - orbital_state(e, secs).position).length();
            println!(
                "[chord] {name} {realm:?} period {:.2} s |v| {:.3} m/s at {hz} Hz — {probe} tick(s): \
                 error {chord:.6e} m (one tick of its own motion {per_tick:.6e} m)",
                e.period(),
                base.velocity.length(),
            );
        }
        println!("[chord] {name} {realm:?}: admissible skew {ticks} ticks at {hz} Hz");
    }

    // THE BOUND, per shipped tick rate, over every mover in the sweep.
    for rate in SHIPPED_TICK_RATES {
        let mut worst = u64::MAX;
        let mut worst_who = String::new();
        for (name, realm, e) in &movers {
            let ticks = admissible_skew_ticks(e, rate);
            if ticks < worst {
                worst = ticks;
                worst_who = format!("{name} {realm:?} period {:.2} s", e.period());
            }
        }
        #[allow(clippy::cast_precision_loss)] // a tick count in the tens
        let worst_ms = worst as f64 * 1000.0 / f64::from(rate);
        let shipped = PlacementCarry::skew_ticks_for(rate);
        println!(
            "[chord] MEASURED at {rate} Hz over {} movers ({SKEW_SEED_SWEEP} seeds): the tightest admits \
             {worst} ticks ({worst_ms:.0} ms) — {worst_who}. Shipped cap {shipped} ticks \
             ({} ms budget).",
            movers.len(),
            PlacementCarry::BUDGET_MS,
        );
        assert!(
            shipped <= worst,
            "the shipped placement-skew cap at {rate} Hz ({shipped} ticks) must stay inside the measured \
             chord budget ({worst} ticks, {worst_ms:.0} ms). Raising PlacementCarry::BUDGET_MS past this \
             makes a shard invent where a planet was — move the measurement, not the number.",
        );
    }
}

/// The client counts the row-distribution measurement is taken at. One is the single-player case, and the
/// largest is the density gate's own co-location count — the shape a real crowd takes.
const ROW_DISTRIBUTION_CLIENTS: [usize; 3] = [1, 8, 128];
/// How many ticks each row-distribution run is driven for, once every client is live.
const ROW_DISTRIBUTION_TICKS: u64 = 40;

/// (e) THE ROWS-PER-DATAGRAM DISTRIBUTION, measured rather than assumed.
///
/// This decides an open design question and nothing else: whether hoisting the frame and a frame origin
/// into the datagram HEADER — so the gateway rewrites a header instead of transforming every row — would
/// be a bandwidth win or a loss. A header costs its bytes once per datagram and saves them once per row,
/// so the verdict flips sign at about four rows, and nobody had counted.
///
/// Counted at the CLIENT, on delivered bytes, over a real cluster.
#[test]
fn the_rows_per_snapshot_datagram_distribution_is_measured() {
    for clients in ROW_DISTRIBUTION_CLIENTS {
        let fabric = FaultFabric::new(4242, 2);
        let mut topo = p1_cluster(&fabric, clients);
        for i in 0..clients {
            let node = NodeId(CLIENT.0 + u64::try_from(i).expect("a small client index"));
            topo.add_node(Box::new(p1_client(
                &fabric,
                node,
                AccountId(1000 + u128::try_from(i).expect("a small client index")),
                walk_forward(),
            )));
        }
        // Let every avatar be granted, then run a window of ordinary play.
        let live = step_until(&mut topo, 400, |t| {
            report(&t.inspect_all(), SHARD).held_entities.len() >= clients
        });
        assert!(live, "every client's avatar is granted before the window");
        for _ in 0..ROW_DISTRIBUTION_TICKS {
            topo.step();
        }

        let mut histogram: BTreeMap<usize, u64> = BTreeMap::new();
        for i in 0..clients {
            let node = NodeId(CLIENT.0 + u64::try_from(i).expect("a small client index"));
            let per_client = {
                let n = topo.node_mut(node).expect("client present");
                n.as_any_mut()
                    .expect("clients opt into downcasting")
                    .downcast_mut::<ScriptedClient>()
                    .expect("the node is a ScriptedClient")
                    .view
                    .snapshot_rows
                    .clone()
            };
            for (rows, count) in per_client {
                *histogram.entry(rows).or_default() += count;
            }
        }
        let total: u64 = histogram.values().sum();
        let at_least_four: u64 = histogram.range(4..).map(|(_, c)| *c).sum();
        assert!(
            total > 0,
            "the window delivered at least one snapshot datagram"
        );
        #[allow(clippy::cast_precision_loss)] // counts in the thousands
        let share = at_least_four as f64 * 100.0 / total as f64;
        println!(
            "[rows/datagram] {clients} client(s): {total} delivered datagrams, histogram {histogram:?}, \
             {share:.1}% carried >= 4 rows (the byte verdict's break-even for a per-datagram header)",
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE CHAIN, ON THE PRODUCTION RELAY PATH
//
// Everything above this line stops after ONE leg. A planet tells its star where an occupant is; the star
// adds the placement it authored and that is where it ends. Nobody above the star ever hears about that
// player, so the position a third level would need cannot be produced anywhere in a running cluster.
//
// The story is three levels deep and each level's addition belongs to exactly one party. The area knows
// only "an occupant this far from my centre". The planet knows only "I put that area 5 m out". The star
// knows only "I put that planet 20 m out". Going up, each of them adds ITS number and states the result
// in its own frame, and sends that on. Nobody ever learns their own address, and no party holds two of
// the three numbers — which is exactly what makes a per-level addition distinguishable from one party
// folding the whole chain by itself.
//
// The gates below run that on three separate hosts, over the real directory, the real cadence, the real
// unreliable relay carrier and the real fold. `frame_fixture`'s
// `the_worked_example_walks_up_and_back_down_through_production_conversions` calls the conversion four
// times in a row with no shard, no relay and no tick behind it; it is the ARITHMETIC ORACLE and it is
// explicitly not this.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The deepest realm of the chain: the seed forest's Area 7, the box a player logs in to.
const CHAIN_AREA: RealmId = RealmId::Area(7);
/// Its own frame — the only frame the area shard is entitled to speak in.
const CHAIN_AREA_FRAME: FrameRef = FrameRef::AreaLocal {
    planet_seed: 7,
    area_seed: 7,
};
/// The number ONLY THE PLANET HOLDS: where it put its area, in its own frame. The seed forest's authored
/// offset, written out rather than read back out of the world under test.
const CHAIN_AREA_FROM_PLANET_M: f64 = 5.0;
/// The number ONLY THE STAR HOLDS: where it put its planet, in its own frame.
const CHAIN_PLANET_FROM_STAR_M: f64 = 20.0;
/// The number ONLY THE AREA HOLDS: where the occupant stands, measured from the area's own centre.
///
/// Well inside the area's 3 m half-extent and its containment hysteresis, so the player genuinely stays
/// where they are: this gate is about what the levels above compute, and a player wandering out mid-run
/// would replace the subject with a re-home.
const CHAIN_OCCUPANT_FROM_AREA_M: f64 = 0.25;
// (The per-level restated-position literals that lived here — 5.25 at the planet, 25.25 at the star —
// died with the occupant up-relay in Step 5 slice D: no level above the owner states an occupant's
// position at all any more. The bit that replaced the lane has no number to pin.)

/// Borrow one topology node as the concrete shard so the gate can read the resources the production code
/// writes. Nothing here reaches past what a shard legitimately holds.
fn with_shard<R>(
    topo: &mut Topology,
    id: NodeId,
    f: impl FnOnce(&mut vd_node::ShardNode<vd_harness::fabric::FabricTransport>) -> R,
) -> R {
    let node = topo.node_mut(id).expect("node present");
    let shard = node
        .as_any_mut()
        .expect("ShardNode opts into downcasting")
        .downcast_mut::<vd_node::ShardNode<vd_harness::fabric::FabricTransport>>()
        .expect("the node is a ShardNode");
    f(shard)
}

/// The SL7 occupancy bit `node` holds for its direct child `child`, if fresh — the upward-liveness
/// probe (Step 5: the ONE thing that crosses upward; the per-occupant pose relay is deleted).
fn child_bit(
    topo: &mut Topology,
    node: NodeId,
    child: vd_core::pose::RealmId,
) -> Option<vd_sim::stub::ChildLiveEntry> {
    with_shard(topo, node, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::ChildLiveness>()
            .0
            .get(&child)
            .copied()
    })
}

/// Does `node`'s AUTHORITY store (`Dots`) hold a pose for `subject`, in any authority state? The
/// anti-lane probe: it answers "does anyone above the owner OWN the occupant", which the deleted
/// per-occupant relay used to make true. Since slice E there is no other store to ask — the entity
/// lane's holding bay is a deleted type — so this plus the wire-silence window in the climb
/// scenario covers SL2's steady state whole.
fn holds_subject_pose(topo: &mut Topology, node: NodeId, subject: EntityId) -> bool {
    with_shard(topo, node, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::Dots>()
            .0
            .values()
            .any(|d| d.entity == subject)
    })
}

/// Every frame this shard can place, and the placement it holds for each — its WHOLE conversion context.
/// The ground rule reads directly off this: exactly one identity (itself), and its own children, and
/// nothing else in the world.
fn placeable_frames(topo: &mut Topology, node: NodeId) -> BTreeMap<String, bool> {
    with_shard(topo, node, |s| {
        let cfg = s.world_mut().resource::<vd_sim::stub::StubConfig>();
        let (realm, tick_hz) = (cfg.realm, 1.0 / cfg.tick_dt_s);
        let regions = s.world_mut().resource::<vd_sim::stub::RealmRegions>();
        let ctx = regions.frame_context(realm, tick_hz, UniverseTick(0));
        // Ask about EVERY frame the whole seed forest contains, not just the ones this shard holds — the
        // question is what it can answer, and a shard that could place something it was never told about
        // would only be caught by asking about that thing.
        vd_core::worldgen::realm_regions_for(FRAME_UNIVERSE_SEED)
            .iter()
            .map(|r| {
                (
                    format!("{:?}", r.realm),
                    ctx.placement(r.frame, UniverseTick(0)).is_some(),
                )
            })
            .collect()
    })
}

/// Boot the three-level chain with a real logged-in player standing in the AREA, every shard armed with
/// the REAL seed geometry and LIVE interest bands, and return the topology plus the avatar's entity id.
fn boot_the_chain(fabric: &FaultFabric) -> (Topology, EntityId) {
    boot_the_chain_with(fabric, &BTreeMap::new())
}

/// [`boot_the_chain`], with `movers_at_the_top` given a turn on the STAR shard.
///
/// The walk forest's bodies are all static, and a shard with no moving child authors nothing on the realm
/// lane, ships nothing and cascades nothing. A scenario about what descends the chain therefore has to put
/// something in motion up there, or it is measuring an empty feed and cannot fail.
fn boot_the_chain_with(
    fabric: &FaultFabric,
    movers_at_the_top: &BTreeMap<RealmId, OrbitalElements>,
) -> (Topology, EntityId) {
    let mut topo = p2_cluster_area_in_planet_in_system(fabric, 8);
    topo.add_node(Box::new(p1_client(
        fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    // WARMUP: the player is granted on the AREA shard and both levels above it hold their own realm
    // leases — without those the directory cannot tell anyone who their parent is.
    let warm = step_until(&mut topo, 200, |t| {
        let r = t.inspect_all();
        !report(&r, SHARD).held_entities.is_empty()
            && report(&r, CHAIN_MID)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == PLANET)
            && report(&r, CHAIN_TOP)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == SYSTEM)
    });
    assert!(
        warm,
        "the avatar is granted in the area and both levels above hold their realms",
    );

    // The scripted walk would fight the explicit position this gate sets, so stop it.
    with_client(&mut topo, ScriptedClient::pause_input);

    // ARM every shard with the REAL seed geometry and a LIVE interest band. From here nothing is
    // hand-fed: the parent lookups, the relay cadence, the folds and the carrier are all production.
    let (v_max, dt) = {
        let cfg = vd_tests::area_stub_config();
        (cfg.move_speed_mps * cfg.time_multiplier, cfg.tick_dt_s)
    };
    let none = BTreeMap::new();
    for (node, realm) in [
        (SHARD, CHAIN_AREA),
        (CHAIN_MID, PLANET),
        (CHAIN_TOP, SYSTEM),
    ] {
        vd_tests::plant_demand_neighbourhood_with_movers(
            &mut topo,
            node,
            FRAME_UNIVERSE_SEED,
            realm,
            v_max,
            dt,
            if node == CHAIN_TOP {
                movers_at_the_top
            } else {
                &none
            },
        );
    }

    let reports = topo.inspect_all();
    let (subject, _) = *report(&reports, SHARD)
        .held_entities
        .first()
        .expect("the area shard holds the logged-in avatar");
    (topo, subject)
}

/// THE ACCEPTANCE STORY, on the production liveness path, over three hosts (Step 5 rewrite).
///
/// The scenario used to measure the occupant's POSE climbing the chain — the per-occupant relay slice
/// D deleted, because a pose crossing a realm boundary is the SL2 breach. What climbs now is ONE BIT
/// per level, recursively: the area holds the player, so its bit beats at the planet; the planet holds
/// NOBODY of its own yet is live purely through its child's bit, so ITS bit beats at the star. No pose,
/// no entity set, no depth counter — and the star knows exactly one thing about everything below its
/// planet: somebody is in there.
#[test]
fn an_occupants_liveness_climbs_every_level_of_the_chain_one_bit_per_level() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, subject) = boot_the_chain(&fabric);

    // Hold the occupant at a known spot in the AREA's own frame, every tick, and wait for the chain to
    // reach the top. Only the area's copy is ever written; everything above is recursion.
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);
    let climbed = step_until(&mut topo, 600, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        child_bit(t, CHAIN_TOP, PLANET).is_some()
    });
    assert!(
        climbed,
        "liveness reaches TWO levels above the realm the player stands in",
    );

    // LEVEL 1 — the planet holds its AREA's fresh bit (the direct occupancy report).
    let mid_bit =
        child_bit(&mut topo, CHAIN_MID, CHAIN_AREA).expect("the planet holds its area's bit");
    // LEVEL 2 — the star holds its PLANET's bit. This is SL7's recursion, and its PREMISE is pinned
    // first: the planet itself holds NOBODY — no dot, no held transient — so the only thing that can
    // be keeping its bit beating is its own child's bit (the occupied-child observer). Without these
    // two probes a stray occupant at the middle level would satisfy the scenario without any
    // recursion ever being exercised.
    let (mid_dots, mid_transients) = with_shard(&mut topo, CHAIN_MID, |s| {
        (
            s.world_mut().resource::<vd_sim::stub::Dots>().0.len(),
            s.world_mut()
                .resource::<vd_sim::stub::OwnedTransients>()
                .0
                .len(),
        )
    });
    assert_eq!(mid_dots, 0, "the planet holds no dot of its own");
    assert_eq!(
        mid_transients, 0,
        "the planet holds no transient of its own"
    );
    let top_bit = child_bit(&mut topo, CHAIN_TOP, PLANET).expect("the star holds its planet's bit");

    // WHAT THE POSE PROBE MEANS, and why it is now the WHOLE question. `Dots` is the AUTHORITY
    // store: nobody above the area OWNS the subject. Both stores that ever held a foreign occupant's
    // pose above its owner are deleted TYPES now (the per-occupant retained store, slice D; the
    // entity lane's holding bay, slice E) — absence is structural, not a runtime zero. What remains
    // measurable is the WIRE: over a settle window, not one entity-lane frame may arrive anywhere on
    // the chain. That is SL2 at steady state, asserted on the transport rather than argued.
    assert!(
        !holds_subject_pose(&mut topo, CHAIN_MID, subject),
        "the planet OWNS no pose for the occupant — the liveness lane crossed one bit, not a pose",
    );
    assert!(
        !holds_subject_pose(&mut topo, CHAIN_TOP, subject),
        "nor does the star — no per-occupant relay exists to put an authority pose above the owner",
    );
    let mut entity_lane_frames = 0usize;
    for _ in 0..20 {
        topo.step();
        set_shard_subject_pose_now(&mut topo, SHARD, subject, CHAIN_AREA_FRAME, at);
        for node in [SHARD, CHAIN_MID, CHAIN_TOP] {
            entity_lane_frames += entity_relays_delivered_to(&mut topo, node);
        }
    }
    assert_eq!(
        entity_lane_frames, 0,
        "the entity lane is SILENT on every leg of an occupied chain — SL2 measured on the wire",
    );

    // The bit carries ordering guards and NOTHING else (the TYPE has no pose field), and its fence
    // names the SENDER's own realm authority — the zombie guard's input. Measured against each
    // sender's live lease, not restated from the receiver.
    let area_fence = with_shard(&mut topo, SHARD, |s| {
        s.world_mut().resource::<vd_sim::stub::RealmAuthority>().0
    })
    .expect("the area holds its realm lease");
    let planet_fence = with_shard(&mut topo, CHAIN_MID, |s| {
        s.world_mut().resource::<vd_sim::stub::RealmAuthority>().0
    })
    .expect("the planet holds its realm lease");
    assert_eq!(
        mid_bit.fence, area_fence,
        "the area's bit carries the area's OWN realm authority",
    );
    assert_eq!(
        top_bit.fence, planet_fence,
        "the planet's bit carries the planet's OWN realm authority",
    );

    // AND THE CHAIN STOPS WHERE THE WORLD DOES. Above the star is a galaxy, and no shard is running
    // it, so the star's parent resolve holds NOTHING — measured on the resource the emit gates on
    // (`aoi_up_relays_nothing_from_a_root_shard` pins the gate itself: an unresolved parent emits no
    // bit). Nothing counted levels to decide that — the beat simply has nowhere to go.
    let top_parent = with_shard(&mut topo, CHAIN_TOP, |s| {
        s.world_mut().resource::<vd_sim::stub::ParentRealmNode>().0
    });
    assert_eq!(
        top_parent, None,
        "the star resolved no parent node — its own bit goes no further",
    );
}

/// THE GROUND RULE, as an assertion that can fail rather than as prose: no shard in the chain can place
/// itself anywhere but at its own origin, and none of them can place a parent, a grandparent or a
/// sibling AT ALL.
///
/// This is the property the whole arc exists for, and it is the one a number-only assertion cannot
/// check: a cluster where one party held the whole forest and folded the chain by itself would produce
/// exactly the same 25.25.
#[test]
fn no_level_of_the_chain_can_place_itself_its_ancestors_or_a_sibling() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, _subject) = boot_the_chain(&fabric);

    // (node, its own realm, the realms it AUTHORED placements for and may therefore place)
    let expected: [(NodeId, RealmId, &[RealmId]); 3] = [
        (SHARD, CHAIN_AREA, &[]),
        (CHAIN_MID, PLANET, &[CHAIN_AREA]),
        (CHAIN_TOP, SYSTEM, &[PLANET, RealmId::Station(7)]),
    ];
    for (node, own, children) in expected {
        let placeable = placeable_frames(&mut topo, node);
        let can: BTreeSet<String> = placeable
            .iter()
            .filter(|(_, ok)| **ok)
            .map(|(realm, _)| realm.clone())
            .collect();
        let allowed: BTreeSet<String> = std::iter::once(own)
            .chain(children.iter().copied())
            .map(|r| format!("{r:?}"))
            .collect();
        assert_eq!(
            can, allowed,
            "{node:?} may place ITSELF (at its own origin, by definition) and the children it authored \
             placements for — and nothing else in the world. Its parent, its grandparents and every \
             sibling are absent on purpose: nobody told it where they are, so a conversion involving \
             one must fail loudly rather than answer confidently from numbers it invented.",
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE DOWN-CHAIN, ON THE PRODUCTION CASCADE PATH
//
// The mirror of everything above. Going up, each level ADDS the placement it authored and sends the
// result on. Coming down, each level SUBTRACTS the placement it authored for the child it is sending to,
// so what arrives at the bottom is measured from the bottom's own centre and the party standing there has
// nothing left to compute. The star holds "I put that planet 20 m out" and nobody else does; the planet
// holds "I put that area 5 m out" and nobody else does; so a station turning around the star arrives at
// the area 25 m nearer than the star stated it, in two separate subtractions made by two separate hosts.
//
// The gate below reads the bytes the fabric actually DELIVERED to each level's own inbox, not what a
// shard believes it sent.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The body the STAR is given a turn: the seed forest's Station 7, a direct child of System 7 and the only
/// thing in this fixture that moves. A static world makes every conversion the identity, which is the case
/// that proves nothing.
const CHAIN_STATION: RealmId = RealmId::Station(7);
/// The radius of that turn, matching the magnitude of the offset the seed forest already gives Station 7,
/// so the station stays where a station belongs — well inside the star's own 40 m boundary.
const CHAIN_STATION_ORBIT_R_M: f64 = 25.0;
/// Its period, chosen so the station covers metres per tick at 20 Hz: the descent has to be measured on a
/// value that is genuinely different from one datagram to the next.
const CHAIN_STATION_PERIOD_S: f64 = 60.0;
/// The whole distance the two levels subtract between them: `20 + 5`, and neither host holds both numbers.
const CHAIN_DOWN_2_M: f64 = CHAIN_PLANET_FROM_STAR_M + CHAIN_AREA_FROM_PLANET_M;
/// The slack allowed on a value that crossed two hosts. Two f64 subtractions in sequence are not bit-equal
/// to one subtraction of their sum, so an exact compare would be measuring the order of operations. At
/// these magnitudes the rounding is ~1e-14 m, while every defect this gate exists to catch is metres — a
/// missing subtraction is 5 or 20, a doubled one is 25. A nanometre sits a million times above the noise
/// and a billion times below the smallest real fault.
const CHAIN_DOWN_TOL_M: f64 = NANOMETRE_M;

/// The station's turn: circular, tilted and phase-shifted so all three components are non-zero (an
/// on-axis fixture cannot tell a subtraction on the wrong axis from a correct one). The central mass is
/// chosen THROUGH the period rather than written down, because the period is the thing being chosen.
fn station_orbit() -> OrbitalElements {
    let a = CHAIN_STATION_ORBIT_R_M;
    let mu = 4.0 * std::f64::consts::PI * std::f64::consts::PI * a.powi(3)
        / (CHAIN_STATION_PERIOD_S * CHAIN_STATION_PERIOD_S);
    OrbitalElements {
        sma: a,
        ecc: 0.0,
        inclination: 0.6,
        raan: 0.4,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: 0.9,
        central_mass: mu / G,
    }
}

/// The realm datagrams a shard was actually DELIVERED on its last step, read out of its own inbox.
///
/// This is wire truth: `InboundBox` holds exactly what the node drained from the fabric, so it says what
/// reached that host rather than what some other host believes it sent. Sampled straight after
/// `Topology::step`, before the next one overwrites it.
fn cascades_delivered_to(topo: &mut Topology, node: NodeId) -> Vec<RealmSnapshotDatagram> {
    with_shard(topo, node, |s| {
        s.world_mut()
            .resource::<vd_sim::runtime::InboundBox>()
            .0
            .iter()
            .filter_map(|m| match m {
                vd_sim::io::Inbound::Wire {
                    class: vd_sim::io::MsgClass::SignalDelta,
                    bytes,
                    ..
                } => match postcard::from_bytes::<InterShardFlow>(bytes) {
                    Ok(InterShardFlow::RealmCascade(rc)) => {
                        postcard::from_bytes(&rc.realm_snapshot_bytes).ok()
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect()
    })
}

/// What `node` AUTHORED for its own children at `tick`, in its own frame — the INPUT to the descent, read
/// through the production expression at the exact instant the rows under test are stamped at. Sampling it
/// at any other instant would measure the station's orbit rather than the conversion.
fn authored_by(
    topo: &mut Topology,
    node: NodeId,
    tick: UniverseTick,
) -> BTreeMap<RealmId, vd_core::pose::StampedPose> {
    with_shard(topo, node, |s| {
        let cfg = s.world_mut().resource::<vd_sim::stub::StubConfig>();
        let (realm, tick_hz) = (cfg.realm, 1.0 / cfg.tick_dt_s);
        s.world_mut()
            .resource::<vd_sim::stub::RealmRegions>()
            .authored_realm_snaps(realm, tick_hz, tick)
            .into_iter()
            .map(|r| (r.realm, r.pose))
            .collect()
    })
}

/// One shard's cascade bookkeeping: rows it restated, rows it refused, datagrams it passed further down.
fn cascade_counts(topo: &mut Topology, node: NodeId) -> (u64, u64, u64) {
    with_shard(topo, node, |s| {
        let st = s.world_mut().resource::<vd_sim::stub::StubStats>();
        (
            st.cascade_rows_converted,
            st.cascade_rows_dropped,
            st.realm_cascade_relayed,
        )
    })
}

/// THE ACCEPTANCE STORY GOING DOWN, on the production cascade path, over three hosts.
///
/// The star authors a turning station in its own frame. It subtracts the 20 it put its planet at and hands
/// the planet a station measured from the PLANET's centre. The planet subtracts the 5 it put its area at
/// and hands the area a station measured from the AREA's centre. The area accepts it and computes nothing
/// at all — its `cascade_rows_converted` never leaves zero, which is what tells a leaf apart from a
/// relaying level by measurement instead of by argument.
///
/// WHAT THIS REPLACES: the star's already-serialized bytes went down verbatim, in the STAR's frame, and
/// were re-fanned to the client unchanged. So one client held two feeds measured in two different spaces —
/// its own realm's, and its star's — and that is the whole reason a party downstream of every shard came
/// to believe it had to compose the two itself.
#[test]
fn the_authored_world_descends_one_subtraction_per_level_and_the_leaf_computes_nothing() {
    let fabric = FaultFabric::new(4242, 2);
    let movers = BTreeMap::from([(CHAIN_STATION, station_orbit())]);
    let (mut topo, subject) = boot_the_chain_with(&fabric, &movers);
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);

    // Run the real chain and collect what each level was DELIVERED. Two distinct arrivals at the bottom is
    // the minimum that can show the value is alive rather than a constant.
    let mut at_the_leaf: Vec<RealmSnapshotDatagram> = Vec::new();
    let mut ids_at_the_middle: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut ids_at_the_leaf: BTreeSet<(u64, u64)> = BTreeSet::new();
    let arrived = step_until(&mut topo, 900, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        for d in cascades_delivered_to(t, CHAIN_MID) {
            ids_at_the_middle.insert((d.frame_id, d.universe_tick.0));
        }
        for d in cascades_delivered_to(t, SHARD) {
            if at_the_leaf
                .last()
                .is_none_or(|p| p.universe_tick != d.universe_tick)
            {
                ids_at_the_leaf.insert((d.frame_id, d.universe_tick.0));
                at_the_leaf.push(d);
            }
        }
        at_the_leaf.len() >= 2
    });
    assert!(
        arrived,
        "the star's authored world reaches the realm the player is standing in, two levels below it: \
         {} arrival(s)",
        at_the_leaf.len(),
    );

    // ONE MEANING, ONE SPACE: every row the leaf was handed is measured from the leaf's own centre.
    for d in &at_the_leaf {
        for r in &d.realms {
            assert_eq!(
                r.pose.frame, CHAIN_AREA_FRAME,
                "the value's own label says which space it is measured in, and for a row arriving at \
                 the bottom of the chain that can only be the bottom's own frame",
            );
            assert_ne!(
                r.frame, r.pose.frame,
                "a row whose head equals its tail is a realm claiming to be its own parent; the \
                 recipient's own row is dropped precisely so this cannot happen",
            );
        }
    }

    // THE NUMBER, at each of the two arrivals, against what the STAR authored at that same instant.
    for d in &at_the_leaf {
        let row = d
            .realms
            .iter()
            .find(|r| r.realm == CHAIN_STATION)
            .expect("the star's turning station is what descends the chain");
        let from_the_star = authored_by(&mut topo, CHAIN_TOP, d.universe_tick)[&CHAIN_STATION]
            .pos
            .offset();
        let subtracted = DVec3::new(CHAIN_DOWN_2_M, 0.0, 0.0);
        let got = row.pose.pos.offset();
        println!(
            "[chain down] at {:?}  star authored {from_the_star:?}  leaf was handed {got:?}  \
             (two subtractions totalling {CHAIN_DOWN_2_M} m)",
            d.universe_tick,
        );
        assert!(
            (got - (from_the_star - subtracted)).length() <= CHAIN_DOWN_TOL_M,
            "the station arrives at the area measured from the AREA's centre: expected {:?}, got \
             {got:?}. Off by {CHAIN_PLANET_FROM_STAR_M} means the star shipped without subtracting; off \
             by {CHAIN_AREA_FROM_PLANET_M} means the planet re-relayed without subtracting; off by \
             {CHAIN_DOWN_2_M} the other way means a level subtracted a number it does not hold, or one \
             was counted twice.",
            from_the_star - subtracted,
        );
        // ANTI-VACUITY: the subtraction genuinely moved the number, on more than one axis.
        assert!((got - from_the_star).length() > 1.0);
        assert!(from_the_star.y.abs() > 1.0 && from_the_star.z.abs() > 1.0);
    }
    // ANTI-VACUITY: the station is genuinely turning, so this is a live feed and not one repeated value.
    let station_at = |d: &RealmSnapshotDatagram| {
        d.realms
            .iter()
            .find(|r| r.realm == CHAIN_STATION)
            .expect("the station row")
            .pose
            .pos
            .offset()
    };
    let travelled = (station_at(&at_the_leaf[1]) - station_at(&at_the_leaf[0])).length();
    assert!(
        travelled > CHAIN_DOWN_TOL_M,
        "the station moved between the two arrivals: {travelled} m",
    );

    // WHO DID THE ARITHMETIC. Both levels above subtract; the level the player stands on does not, ever.
    let (top_converted, top_dropped, top_relayed) = cascade_counts(&mut topo, CHAIN_TOP);
    let (mid_converted, mid_dropped, mid_relayed) = cascade_counts(&mut topo, CHAIN_MID);
    let (leaf_converted, leaf_dropped, leaf_relayed) = cascade_counts(&mut topo, SHARD);
    println!(
        "[chain down] rows restated: star {top_converted}, planet {mid_converted}, area \
         {leaf_converted}; datagrams passed further down: planet {mid_relayed}"
    );
    assert!(top_converted > 0, "the star restates for its planet");
    assert!(mid_converted > 0, "the planet restates for its area");
    assert_eq!(
        leaf_converted, 0,
        "THE LEAF ACCEPTS AND COMPUTES NOTHING — it was handed numbers already measured from its own \
         centre, and a level that had to convert on receipt would be a level being told about a space \
         it has no business knowing",
    );
    assert!(
        mid_relayed > 0,
        "the planet passes the star's world on down"
    );
    assert_eq!(leaf_relayed, 0, "the leaf has nobody below it");
    assert_eq!(top_relayed, 0, "the star receives no cascade of its own");
    assert_eq!((top_dropped, mid_dropped, leaf_dropped), (0, 0, 0));

    // THE frame_id GUARD, which replaces a structural guarantee with a checkable one. Restating the values
    // means opening the datagram, so a level COULD now stamp its own counter on someone else's rows — and
    // the client's staleness gate is a per-realm high-water fed by exactly one authoring shard's counter,
    // so it would ratchet those boxes past anything their author will produce for thousands of ticks and
    // freeze them for good. Every (counter, instant) pair the bottom saw must be one the middle saw first.
    assert!(ids_at_the_leaf.len() > 1 && ids_at_the_middle.len() > 1);
    assert!(
        ids_at_the_leaf.is_subset(&ids_at_the_middle),
        "no level minted a frame_id of its own on the way down: bottom saw {ids_at_the_leaf:?}, \
         middle saw {ids_at_the_middle:?}",
    );

    // THE RE-SERIALIZE COST, counted over a real window rather than reasoned about. The star used to
    // serialize its authored world ONCE and hand the same refcounted body to every active child; each child
    // now gets its own datagram, because each child gets different numbers. That is the trade the ground
    // rule makes, and this is the size of it: datagrams per tick per active child, at each level.
    const WINDOW_TICKS: u64 = 100;
    let (top_rows_before, _, _) = cascade_counts(&mut topo, CHAIN_TOP);
    let (mid_rows_before, _, mid_relayed_before) = cascade_counts(&mut topo, CHAIN_MID);
    let (mut to_mid, mut to_leaf) = (0u64, 0u64);
    for _ in 0..WINDOW_TICKS {
        topo.step();
        set_shard_subject_pose_now(&mut topo, SHARD, subject, CHAIN_AREA_FRAME, at);
        to_mid += cascades_delivered_to(&mut topo, CHAIN_MID).len() as u64;
        to_leaf += cascades_delivered_to(&mut topo, SHARD).len() as u64;
    }
    let (top_rows_after, _, _) = cascade_counts(&mut topo, CHAIN_TOP);
    let (mid_rows_after, _, mid_relayed_after) = cascade_counts(&mut topo, CHAIN_MID);
    #[allow(clippy::cast_precision_loss)] // counts in the hundreds
    let per = |n: u64| n as f64 / WINDOW_TICKS as f64;
    println!(
        "[chain down cost] over {WINDOW_TICKS} ticks: star restated {:.2} row/tick, planet restated \
         {:.2} row/tick and re-serialized {:.2} datagram/tick; delivered {:.2} datagram/tick to the \
         planet and {:.2} to the area",
        per(top_rows_after - top_rows_before),
        per(mid_rows_after - mid_rows_before),
        per(mid_relayed_after - mid_relayed_before),
        per(to_mid),
        per(to_leaf),
    );
    assert!(
        to_leaf > 0,
        "the window observed the chain actually running"
    );
}

/// The COST of the liveness chain, measured rather than assumed (Step 5 rewrite): what one beat costs
/// on the wire per leg, how often the middle level's beat lands at the top, and how the TTL bridges a
/// lossy link. The deleted pose relay cost a full pose + lineage per occupant per level per tick; the
/// bit costs a fence and a tick per REALM per cadence, and that difference is the whole deletion's
/// price tag, printed.
#[test]
fn the_per_leg_liveness_cost_and_rate_are_measured() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, subject) = boot_the_chain(&fabric);
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);
    let climbed = step_until(&mut topo, 600, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        child_bit(t, CHAIN_TOP, PLANET).is_some()
    });
    assert!(climbed, "the chain is running before its cost is quoted");

    // Bytes on the wire for each leg, built from what those legs actually carry — the deeper sender's
    // lineage is longer, so the two are not the same size and the difference is the thing to know.
    let mut leg = |node: NodeId| -> usize {
        // Every field read off the RUNNING shard — including the tick, because postcard varints are
        // length-dependent and a fabricated stamp would size a frame the cluster never ships.
        let (own_coord, fence, at) = with_shard(&mut topo, node, |s| {
            let coord = s
                .world_mut()
                .resource::<vd_sim::stub::StubConfig>()
                .own_coord
                .clone();
            let fence = s
                .world_mut()
                .resource::<vd_sim::stub::RealmAuthority>()
                .0
                .expect("a running chain level holds its lease");
            let at = s
                .world_mut()
                .resource::<vd_sim::runtime::ClockSample>()
                .universe_tick;
            (coord, fence, at)
        });
        postcard::to_allocvec(&InterShardFlow::ChildLive(vd_wire::intershard::ChildLive {
            child: own_coord,
            fence,
            at,
        }))
        .expect("closed wire enums serialize infallibly")
        .len()
    };
    let leg_1 = leg(SHARD); // area -> planet
    let leg_2 = leg(CHAIN_MID); // planet -> star

    // The RATE, counted at the top over a real window: how many beats of the PLANET's bit actually
    // land at the star per tick.
    let before = with_shard(&mut topo, CHAIN_TOP, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::StubStats>()
            .child_live_received
    });
    const WINDOW_TICKS: u64 = 100;
    for _ in 0..WINDOW_TICKS {
        set_shard_subject_pose_now(&mut topo, SHARD, subject, CHAIN_AREA_FRAME, at);
        topo.step();
    }
    let after = with_shard(&mut topo, CHAIN_TOP, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::StubStats>()
            .child_live_received
    });
    let beats = after - before;
    #[allow(clippy::cast_precision_loss)] // counts in the hundreds
    let per_tick = beats as f64 / WINDOW_TICKS as f64;
    #[allow(clippy::cast_precision_loss)] // a message size in the tens of bytes
    let bytes_per_s = per_tick * leg_2 as f64 / vd_tests::area_stub_config().tick_dt_s;
    println!(
        "[liveness] leg 1 (area->planet) {leg_1} B, leg 2 (planet->star) {leg_2} B; the star heard \
         {beats} beats in {WINDOW_TICKS} ticks = {per_tick:.2}/tick, {bytes_per_s:.0} B/s per REALM \
         (not per occupant) at {:.0} Hz",
        1.0 / vd_tests::area_stub_config().tick_dt_s,
    );
    assert!(beats > 0, "the window observed the chain actually beating");

    // DELIVERY at depth, on a perfect link first: how often each level hears a fresh beat.
    let clean_mid = freshness(&mut topo, CHAIN_MID, CHAIN_AREA, subject, at);
    let clean_top = freshness(&mut topo, CHAIN_TOP, PLANET, subject, at);
    println!(
        "[liveness delivery] perfect link: leg-1 level heard a beat on {clean_mid} of \
         {FRESHNESS_TICKS} ticks, leg-2 level on {clean_top}",
    );

    // AND UNDER A LOSSY LINK: the carrier is unreliable fire-and-forget, so a dropped beat is a tick
    // with no news, and the retain TTL is what bridges it — per level, so a 3-level chain bridges one
    // gap per level rather than one gap in total. The property that must hold under loss is that the
    // BIT NEVER BLINKS: staleness compounds, liveness does not flicker — asserted EVERY tick of the
    // window inside `freshness`, at both levels, not sampled once after it.
    //
    // ⚠ WHAT THIS HARNESS ACTUALLY MODELS, stated so the number is not over-read: its `drop_p` drops a
    // delivery ATTEMPT and re-queues the message, so a "lost" beat arrives LATE rather than never. So
    // this measures how much staleness compounds per level, NOT a (1-p)^depth end-to-end delivery.
    for (from, to) in [(SHARD, CHAIN_MID), (CHAIN_MID, CHAIN_TOP)] {
        fabric.set_policy(
            from,
            to,
            vd_harness::fabric::LinkPolicy {
                drop_p: LOSSY_LINK_DROP_P,
                ..vd_harness::fabric::LinkPolicy::default()
            },
        );
    }
    let lossy_mid = freshness(&mut topo, CHAIN_MID, CHAIN_AREA, subject, at);
    let lossy_top = freshness(&mut topo, CHAIN_TOP, PLANET, subject, at);
    println!(
        "[liveness delivery] drop_p {LOSSY_LINK_DROP_P} per leg: leg-1 level {lossy_mid} of \
         {FRESHNESS_TICKS}, leg-2 level {lossy_top}",
    );
    assert!(
        child_bit(&mut topo, CHAIN_TOP, PLANET).is_some(),
        "under a lossy link the bit is STALE, never GONE — the TTL bridges what the link drops",
    );
}

/// How many of [`FRESHNESS_TICKS`] ticks brought `node` NEWS about its NAMED direct child `child` —
/// a tick counts iff that child's bit's `last_seen` stamp moved. The occupant is held in place
/// throughout, so anything that changes is the transport. EVERY tick also asserts the bit is
/// PRESENT: that is the no-blink half of the lossy claim measured per tick rather than sampled once
/// after the window — under loss the bit may go stale (a tick with no news), never gone (the TTL
/// bridges the gap).
fn freshness(
    topo: &mut Topology,
    node: NodeId,
    child: vd_core::pose::RealmId,
    subject: EntityId,
    at: DVec3,
) -> u64 {
    let mut heard = 0;
    let mut prev = child_bit(topo, node, child).map(|e| e.last_seen);
    for _ in 0..FRESHNESS_TICKS {
        set_shard_subject_pose_now(topo, SHARD, subject, CHAIN_AREA_FRAME, at);
        topo.step();
        let bit = child_bit(topo, node, child);
        assert!(
            bit.is_some(),
            "the bit never blinks: stale under loss, never gone",
        );
        let now = bit.map(|e| e.last_seen);
        if now != prev {
            heard += 1;
        }
        prev = now;
    }
    heard
}

/// The window each freshness figure is counted over.
const FRESHNESS_TICKS: u64 = 200;
/// The per-attempt drop the lossy half of the delivery measurement induces on EVERY leg of the chain.
const LOSSY_LINK_DROP_P: f64 = 0.5;

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// SLICE 4 — THE SHAPE LANE DESCENDS THE SAME CHAIN.
//
// The live-pose lane above now hands the bottom of the chain numbers measured from the bottom's own
// centre. The RELIABLE SHAPE lane — the boxes themselves, shipped once when a realm comes into view —
// still went down as a single jump from the shard that authored a box to the shard that relayed the
// occupant, which on a chain of three levels or more is not where the player is. So the boxes and the
// player were measured from two different centres, and every level in between was skipped entirely.
//
// These gates read WIRE TRUTH at each host — what its own inbox was handed — and the client edge at the
// gateway, which is the last thing a shard says before the router touches it.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// Where the seed forest puts the AREA inside its planet — the number ONLY THE PLANET holds.
const CHAIN_AREA_FROM_PLANET_M2: f64 = 5.0;
/// Where it puts the PLANET inside its star — the number ONLY THE STAR holds.
const CHAIN_PLANET_FROM_STAR_M2: f64 = 20.0;
/// The half-extent of that area box, so "the player is inside the box they are standing in" is checked
/// against the box's real size rather than against a tolerance picked to make the gate pass.
const CHAIN_AREA_HALF_M: f64 = 3.0;

/// Every `ProxySceneSet` a shard was DELIVERED on its last step, read out of its own inbox — the reliable
/// shape lane's wire truth at that host, sampled the same way the cascade gate samples the pose lane.
fn scene_sets_delivered_to(
    topo: &mut Topology,
    node: NodeId,
) -> Vec<vd_wire::intershard::ChildSceneSet> {
    with_shard(topo, node, |s| {
        s.world_mut()
            .resource::<vd_sim::runtime::InboundBox>()
            .0
            .iter()
            .filter_map(|m| match m {
                vd_sim::io::Inbound::Wire {
                    class: vd_sim::io::MsgClass::Saga,
                    bytes,
                    ..
                } => match postcard::from_bytes::<InterShardFlow>(bytes) {
                    Ok(InterShardFlow::ChildSceneSet(p)) => Some(p),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    })
}

/// THE CLIENT EDGE: every realm-scene delta the LEAF shard put on the wire to the gateway, read out of the
/// gateway's own inbox. This is the last thing a shard says about the world before anything downstream of
/// it is involved at all, which is exactly what this whole arc is about.
fn scene_deltas_at_the_client_edge(
    topo: &mut Topology,
) -> Vec<(AccountId, Vec<vd_wire::channels::RealmShape>, Vec<RealmId>)> {
    with_shard(topo, vd_tests::GATEWAY, |s| {
        s.world_mut()
            .resource::<vd_sim::runtime::InboundBox>()
            .0
            .iter()
            .filter_map(|m| match m {
                vd_sim::io::Inbound::Wire {
                    from,
                    class: vd_sim::io::MsgClass::Control,
                    bytes,
                } if *from == SHARD => {
                    match postcard::from_bytes::<vd_wire::session_flow::ShardToGateway>(bytes) {
                        Ok(vd_wire::session_flow::ShardToGateway::RealmSceneDelta {
                            observer,
                            added,
                            removed,
                        }) => Some((observer, added, removed)),
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect()
    })
}

/// One shard's shape-lane bookkeeping (Step 5 slice C): outlines restated, outlines refused, sets
/// accepted from above, sets refused as addressed to somebody else, live bits with no roster entry.
fn scene_counts(topo: &mut Topology, node: NodeId) -> (u64, u64, u64, u64, u64) {
    with_shard(topo, node, |s| {
        let st = s.world_mut().resource::<vd_sim::stub::StubStats>();
        (
            st.proxy_scene_shapes_restated,
            st.proxy_scene_shapes_dropped,
            st.child_scene_received,
            st.child_scene_misrouted,
            st.child_scene_unaddressable,
        )
    })
}

/// THE ACCEPTANCE STORY FOR THE BOXES, over the same three hosts as the pose lane.
///
/// A box the STAR authored (the planet the player's realm hangs off) and a box the PLANET authored (the
/// area the player is standing in) both arrive at the AREA shard's client edge measured from the AREA's
/// centre — two different authoring parties, two different lengths of chain, one space at the bottom. And
/// the player's own pose, which the area shard has always held in its own frame and never converts, lands
/// in the same integer cell as the box it is standing in.
///
/// WHAT THIS REPLACES. The reflect was addressed at the shard that relayed the occupant up, and stopped
/// there. On this chain that is the PLANET, which hosts no client for this account — so the star's boxes
/// were counted as strays and thrown away, and the levels below were never told. The measurement of that
/// is in this gate too: the planet's stray count stays at zero and its accept count climbs instead.
#[test]
fn the_authored_boxes_descend_one_subtraction_per_level_into_the_space_the_player_stands_in() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, subject) = boot_the_chain(&fabric);
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);

    // Run the real chain and record, per hop, the FIRST tick anything arrived — the latency this
    // hop-by-hop descent costs is the difference between them, and it is a measurement, not a prediction.
    // Traced PER BOX, not per hop: the two levels start reflecting at very different times (the planet
    // has retained the player long before the star has heard of them at all), so a first-anything-arrived
    // stamp would measure how the chain warmed up rather than what a hop costs. Following the ONE box the
    // star authored down every leg is the measurement the plan asks for.
    let (mut t_mid, mut t_leaf, mut t_edge) = (None, None, None);
    let (mut t_area_leaf, mut t_area_edge) = (None, None);
    // The client edge is a STREAM of add/remove deltas, so the scene is what reconciling them leaves —
    // exactly what the party at the other end holds. Reading one message would measure message boundaries.
    let mut scene: BTreeMap<RealmId, vd_wire::channels::RealmShape> = BTreeMap::new();
    let arrived = step_until(&mut topo, 900, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        let now = t.tick().0;
        let holds = |sets: &[vd_wire::intershard::ChildSceneSet], realm: RealmId| {
            sets.iter()
                .any(|p| p.realms.iter().any(|s| s.realm == realm))
        };
        let to_mid = scene_sets_delivered_to(t, CHAIN_MID);
        let to_leaf = scene_sets_delivered_to(t, SHARD);
        if holds(&to_mid, PLANET) {
            t_mid = t_mid.or(Some(now));
        }
        if holds(&to_leaf, PLANET) {
            t_leaf = t_leaf.or(Some(now));
        }
        if holds(&to_leaf, CHAIN_AREA) {
            t_area_leaf = t_area_leaf.or(Some(now));
        }
        for (_, added, removed) in scene_deltas_at_the_client_edge(t) {
            for s in added {
                if s.realm == PLANET {
                    t_edge = t_edge.or(Some(now));
                }
                if s.realm == CHAIN_AREA {
                    t_area_edge = t_area_edge.or(Some(now));
                }
                scene.insert(s.realm, s);
            }
            for r in removed {
                scene.remove(&r);
            }
        }
        scene.len() >= 2
    });
    assert!(
        arrived,
        "the star's box and the planet's box both reach the client edge of the realm the player is \
         standing in: scene {scene:?}",
    );

    // ONE MEANING, ONE SPACE — every box the client is handed, measured from the player's own realm.
    let drawn: BTreeMap<RealmId, DVec3> = scene
        .values()
        .map(|s| (s.realm, s.center.offset()))
        .collect();
    println!("[boxes down] client edge: {drawn:?}");
    assert_eq!(
        drawn[&CHAIN_AREA],
        DVec3::ZERO,
        "the box the player is STANDING IN is measured from its own centre, which is the origin. \
         {CHAIN_AREA_FROM_PLANET_M2} means the planet reflected without subtracting.",
    );
    assert_eq!(
        drawn[&PLANET],
        DVec3::new(-CHAIN_AREA_FROM_PLANET_M2, 0.0, 0.0),
        "the box the STAR authored arrives measured from the AREA's centre. \
         {CHAIN_PLANET_FROM_STAR_M2} means the star reflected without subtracting; \
         {CHAIN_AREA_FROM_PLANET_M2} means the planet passed it on without subtracting; a number the \
         size of {CHAIN_DOWN_2_M} means a level subtracted something it does not hold.",
    );

    // THE BOX AND THE THING STANDING IN IT, on one point by construction. The area shard has always held
    // the player's pose in its own frame and converts nothing; the box just arrived there through two
    // subtractions made by two other hosts. They have to agree, and to the integer cell they do.
    let pose = with_shard(&mut topo, SHARD, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::Dots>()
            .0
            .values()
            .find(|d| d.entity == subject)
            .expect("the area shard holds the player")
            .pose
    });
    assert_eq!(pose.frame, CHAIN_AREA_FRAME, "and it never converted it");
    assert_eq!(
        pose.pos.cell(),
        vd_core::pose::LatticePos::local(drawn[&CHAIN_AREA]).cell(),
        "the player and the box they are inside sit in the same integer cell",
    );
    assert!(
        (pose.pos.offset() - drawn[&CHAIN_AREA]).length() < CHAIN_AREA_HALF_M,
        "and the player is inside that box: {:?} against a box of half-extent \
         {CHAIN_AREA_HALF_M} at {:?}",
        pose.pos.offset(),
        drawn[&CHAIN_AREA],
    );

    // THE PARENT LINK SURVIVES THE DESCENT. Restating a centre must not disturb which realm a box hangs
    // off — the receiver builds its nesting from that link alone and has no hierarchy of its own to fall
    // back on. Each box either arrives with its parent's box beside it, or names a realm the session's
    // own realm hangs off, which its login registry already gave it.
    let ancestry: BTreeSet<RealmId> =
        vd_core::worldgen::ancestor_realms(&vd_core::worldgen::realm_regions_for(0), CHAIN_AREA)
            .into_iter()
            .collect();
    for s in scene.values() {
        let Some(p) = s.parent else { continue };
        assert!(
            drawn.contains_key(&p) || ancestry.contains(&p),
            "{:?} hangs off {p:?}, which is neither in the scene nor a realm the session already \
             holds — an orphan must be refused, never rooted at an assumed origin",
            s.realm,
        );
    }

    // WHO DID THE ARITHMETIC, and where the old jump was losing everything.
    let (top_r, top_d, top_rel, top_mis, top_un) = scene_counts(&mut topo, CHAIN_TOP);
    let (mid_r, mid_d, mid_rel, mid_mis, mid_un) = scene_counts(&mut topo, CHAIN_MID);
    let (leaf_r, leaf_d, leaf_rel, leaf_mis, leaf_un) = scene_counts(&mut topo, SHARD);
    println!(
        "[boxes down] outlines restated: star {top_r}, planet {mid_r}, area {leaf_r}; sets accepted \
         from above: planet {mid_rel}, area {leaf_rel}; mis-addressed: star {top_mis}, planet \
         {mid_mis}, area {leaf_mis}"
    );
    assert!(top_r > 0, "the star restates for its planet");
    assert!(mid_r > 0, "the planet restates for its area");
    assert_eq!(
        leaf_r, 0,
        "THE LEAF RESTATES NOTHING — it was handed boxes already measured from its own centre",
    );
    assert!(
        mid_rel > 0,
        "the planet takes the star's boxes on for the hop below instead of drawing or dropping them",
    );
    assert!(
        leaf_rel > 0,
        "THE LEAF HOLDS A SET FROM ABOVE TOO (slice C's recursion): the realm the player stands in \
         folds its surroundings into the client scene from the one from-above holding",
    );
    assert_eq!(
        (top_mis, mid_mis, leaf_mis),
        (0, 0, 0),
        "nothing is dropped as mis-addressed anywhere on the chain — the planet is exactly where the \
         old per-occupant jump used to lose the star's whole world",
    );
    assert_eq!((top_d, mid_d, leaf_d), (0, 0, 0), "nothing refused");
    assert_eq!(
        (top_un, mid_un, leaf_un),
        (0, 0, 0),
        "no live bit outlived its roster entry"
    );
    assert_eq!(
        top_rel, 0,
        "the root has no parent, so nothing arrives from above it"
    );

    // THE PRICE OF GOING HOP BY HOP, in ticks, on the real topology. The reflect used to be one jump from
    // the authoring shard straight to the shard that relayed the occupant; it is now one delivery per
    // level, and that is what the ground rule costs on this lane.
    let (t_mid, t_leaf, t_edge) = (
        t_mid.expect("the star's box reached the planet"),
        t_leaf.expect("the star's box reached the area"),
        t_edge.expect("the star's box reached the client edge"),
    );
    let (t_area_leaf, t_area_edge) = (
        t_area_leaf.expect("the planet's box reached the area"),
        t_area_edge.expect("the planet's box reached the client edge"),
    );
    println!(
        "[boxes down] the STAR's box (two levels above the player) arrived: planet tick {t_mid}, area \
         tick {t_leaf}, client edge tick {t_edge} — the leg this slice ADDED costs {} tick(s), the \
         leaf-to-edge leg {} tick(s). Under the single jump the star's box reached the planet and went \
         no further at any price. The AREA's OWN box (the room the player is standing in, authored by \
         the planet) reached the client edge on tick {t_area_edge} and the planet's reflect of it \
         reached the area on tick {t_area_leaf}.",
        t_leaf - t_mid,
        t_edge - t_leaf,
    );
    assert!(
        t_leaf > t_mid && t_edge >= t_leaf,
        "each level is a separate delivery, in order",
    );
    // THE ROOM DOES NOT WAIT FOR THE CHAIN. The area's own box is the ONE outline in this scene with two
    // possible authors: the planet reflects it down (kept at the origin, like every path-child), and the
    // area's own shard states it directly out of the only geometric fact a realm holds about itself. They
    // agree on the value — the assertion above is on the reconciled scene, so it holds either way — but not
    // on the timing, and the timing is the whole point: the parent's copy costs an up-relay and a reflect
    // back, and on the top of a live chain it never comes at all. This used to be a subtraction of the two
    // and it OVERFLOWED the moment the leaf started answering for its own room, which is how the change
    // announced itself here.
    assert!(
        t_area_edge < t_area_leaf,
        "the room reached the client on tick {t_area_edge} but the chain only told this shard about it \
         on tick {t_area_leaf} — if that order ever reverses, the login scene is waiting on a round-trip \
         again",
    );

    // THE RE-DRIVE, counted rather than reasoned about. `ProxySceneSet` is reliable, so a hop-by-hop chain
    // that lost its send-on-change diffing at any level would turn into a full-set resend per tick per
    // level. The set is unchanged while the player stands still, so the honest number here is ZERO.
    const WINDOW_TICKS2: u64 = 100;
    let (mut to_mid, mut to_leaf) = (0u64, 0u64);
    for _ in 0..WINDOW_TICKS2 {
        topo.step();
        set_shard_subject_pose_now(&mut topo, SHARD, subject, CHAIN_AREA_FRAME, at);
        to_mid += scene_sets_delivered_to(&mut topo, CHAIN_MID).len() as u64;
        to_leaf += scene_sets_delivered_to(&mut topo, SHARD).len() as u64;
    }
    println!(
        "[boxes down] over {WINDOW_TICKS2} settled ticks: {to_mid} set(s) to the planet, {to_leaf} to \
         the area — send-on-change is preserved at EVERY hop, not just the first",
    );
    assert_eq!(
        (to_mid, to_leaf),
        (0, 0),
        "a settled chain re-ships nothing on the reliable lane at any level",
    );
}

// ─────────────────────────── SLICE 5 — THE CROSS-REALM ENTITY FEED ───────────────────────────

/// SLICE 6 — THE LOGIN SCENE ARRIVES FROM THE SHARD THAT OWNS THE ROOM, AND DOES NOT WAIT FOR THE CHAIN.
///
/// The box a player is standing inside used to be authored by the router: it held its own copy of the seed
/// forest, enumerated the home realm's whole ancestor chain out of it, and re-expressed every centre into a
/// space it picked. That is the ground rule's breach in the first message a player ever sees — the router is
/// the parent of no realm, so every number in it came from placements it had no business holding, and it was
/// a second, independent derivation of geometry the shards already own (free to disagree with the live one,
/// and it did).
///
/// It is stated by the realm's own shard now, out of the one geometric fact a realm holds about ITSELF: it
/// is at zero in its own frame. THE MEASUREMENT THAT SAYS SO IS A TIME. The shape lane does send this same
/// outline down from the parent — the path-child's box is kept at the origin — so on a warm chain the two
/// agree and reading the value alone cannot tell which party produced it. What tells them apart is that the
/// parent's copy costs an up-relay and a reflect back: this gate asserts the room reaches the client edge
/// STRICTLY BEFORE the first reflect from above reaches the leaf at all.
#[test]
fn the_room_the_player_is_standing_in_is_streamed_by_its_own_shard_before_the_chain_reaches_it() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, subject) = boot_the_chain(&fabric);
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);

    // The two instants this gate is about: when the AREA's own outline first reached the client edge, and
    // when the leaf was first handed ANY set from above. Nothing else is needed to tell the two authors
    // apart.
    let (mut t_room_at_edge, mut t_first_set_to_leaf) = (None, None);
    let mut room: Option<vd_wire::channels::RealmShape> = None;
    let settled = step_until(&mut topo, 900, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        let now = t.tick().0;
        if !scene_sets_delivered_to(t, SHARD).is_empty() {
            t_first_set_to_leaf = t_first_set_to_leaf.or(Some(now));
        }
        for (_, added, _) in scene_deltas_at_the_client_edge(t) {
            for s in added {
                if s.realm == CHAIN_AREA {
                    t_room_at_edge = t_room_at_edge.or(Some(now));
                    room = room.or(Some(s));
                }
            }
        }
        t_first_set_to_leaf.is_some() & t_room_at_edge.is_some()
    });
    assert!(
        settled,
        "both instants observed: room at the edge {t_room_at_edge:?}, first set from above {t_first_set_to_leaf:?}",
    );
    let (t_room, t_set) = (
        t_room_at_edge.expect("the room reached the client edge"),
        t_first_set_to_leaf.expect("a set from above reached the leaf"),
    );
    println!(
        "[the room] area's own outline at the client edge on tick {t_room}; first reflect from above \
         reached the leaf on tick {t_set}",
    );
    assert!(
        t_room < t_set,
        "the room the player is standing in reached the client on tick {t_room}, and the first thing \
         the chain said to this shard arrived on tick {t_set}. Equal or later means the outline came \
         from the parent's reflect after a round-trip — which is exactly what a player logging in on the \
         top of a live chain would never get.",
    );

    // And it is the realm's OWN origin in its OWN frame — no address, nothing folded, nothing subtracted.
    let room = room.expect("the room's outline");
    assert_eq!(
        room.center.offset(),
        DVec3::ZERO,
        "a realm is at zero in its own frame, wherever its parent has put it",
    );
    assert_eq!(
        room.frame, CHAIN_AREA_FRAME,
        "stated in the frame the player's own pose is already measured in",
    );
}

/// The second client's connection node. A second live player is what the ABSENCE contract needs (Step 5
/// slice E): with one occupant, "nobody's figure crosses a realm boundary" is vacuous — there is no other
/// figure to leak. Two players in two realms are the smallest world in which each edge's refusal to carry
/// the OTHER's figure, and the lane's wire-silence, are non-trivial measurements.
const CLIENT_B: NodeId = NodeId(101);

/// Where the traveller is walked to before it crosses: `-7` in the area's own frame. That is past the
/// area's 3 m half-extent AND past the 2 m hysteresis outset, so the real containment detector genuinely
/// releases it rather than holding it in the band; and it is `-2` in the planet's frame, comfortably inside
/// the planet's 10 m shell, so the planet is where it belongs and not the star.
const TRAVELLER_FROM_AREA_M: f64 = -7.0;
/// Where the traveller is held AFTER it has landed on the planet, in the planet's own frame — DELIBERATELY
/// not where it crossed. The area keeps a retained ghost of a departed player for the length of the
/// hand-off, frozen at the last position it owned; if the fixture parked the traveller where it left, that
/// stale mirror would report the expected number and the down-leg assertion could not fail. Well clear of
/// the area box (which spans 2..8 in the planet's frame) so the planet keeps it, and well inside the
/// planet's own 10 m shell so it does not leave for the star.
const TRAVELLER_FROM_PLANET_M: f64 = -6.0;
// (The restated-position literals that lived here — the traveller at `-11` in the area's frame, the
// stayer at `5.25` in the planet's — died with the entity relay in Step 5 slice E: no level states
// another realm's occupant at all any more. The absence itself is what the scenario now asserts.)

/// Boot the three-level chain with TWO logged-in players in the area, and return both avatars.
///
/// Both log in through the real flow onto the same shard, because a static cluster attaches every session
/// to one shard; one of them is then walked out through the real detector and the real saga, which is how
/// the fixture ends up with a player in each of two levels without anything being hand-placed.
fn boot_the_chain_with_two_players(fabric: &FaultFabric) -> (Topology, EntityId, EntityId) {
    let mut topo = p2_cluster_area_in_planet_in_system(fabric, 8);
    for (node, account) in [(CLIENT, 1000u128), (CLIENT_B, 1001)] {
        topo.add_node(Box::new(p1_client(
            fabric,
            node,
            AccountId(account),
            walk_forward(),
        )));
    }

    let warm = step_until(&mut topo, 300, |t| {
        let r = t.inspect_all();
        report(&r, SHARD).held_entities.len() >= 2
            && report(&r, CHAIN_MID)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == PLANET)
            && report(&r, CHAIN_TOP)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == SYSTEM)
    });
    assert!(
        warm,
        "both avatars are granted in the area and both levels above hold their realms",
    );
    for node in [CLIENT, CLIENT_B] {
        let client = topo.node_mut(node).expect("client present");
        client
            .as_any_mut()
            .expect("clients opt into downcasting")
            .downcast_mut::<ScriptedClient>()
            .expect("the node is a ScriptedClient")
            .pause_input();
    }

    let (v_max, dt) = {
        let cfg = vd_tests::area_stub_config();
        (cfg.move_speed_mps * cfg.time_multiplier, cfg.tick_dt_s)
    };
    for (node, realm) in [
        (SHARD, CHAIN_AREA),
        (CHAIN_MID, PLANET),
        (CHAIN_TOP, SYSTEM),
    ] {
        vd_tests::plant_demand_neighbourhood(
            &mut topo,
            node,
            FRAME_UNIVERSE_SEED,
            realm,
            v_max,
            dt,
        );
    }

    let reports = topo.inspect_all();
    let held = &report(&reports, SHARD).held_entities;
    let (stayer, _) = held[0];
    let (traveller, _) = held[1];
    (topo, stayer, traveller)
}

/// Every entity row a given shard put on the wire toward the gateway on its last step, read out of the
/// GATEWAY's own inbox. This is the last thing that shard says about who is where before anything
/// downstream of it is involved, which is the only place this arc's claim can honestly be measured.
fn entity_rows_at_the_client_edge(
    topo: &mut Topology,
    from_shard: NodeId,
) -> Vec<vd_wire::channels::EntitySnap> {
    with_shard(topo, vd_tests::GATEWAY, |s| {
        s.world_mut()
            .resource::<vd_sim::runtime::InboundBox>()
            .0
            .iter()
            .filter_map(|m| match m {
                vd_sim::io::Inbound::Wire {
                    from,
                    class: vd_sim::io::MsgClass::Snapshot,
                    bytes,
                } if *from == from_shard => {
                    match postcard::from_bytes::<vd_wire::session_flow::ShardToGateway>(bytes) {
                        Ok(vd_wire::session_flow::ShardToGateway::Frame {
                            snapshot_bytes, ..
                        }) => postcard::from_bytes::<vd_wire::channels::SnapshotDatagram>(
                            &snapshot_bytes,
                        )
                        .ok(),
                        _ => None,
                    }
                }
                _ => None,
            })
            .flat_map(|d| d.entities)
            .collect()
    })
}

/// How many TOMBSTONED entity-lane frames (either leg) a shard was delivered on its last step, read
/// out of its own inbox — wire truth. The lane is dead (Step 5 slice E), so every caller asserts
/// ZERO: the frames still DECODE (reserved discriminants), which is exactly what lets a silence
/// assertion mean "nothing was sent" rather than "nothing could be read".
fn entity_relays_delivered_to(topo: &mut Topology, node: NodeId) -> usize {
    with_shard(topo, node, |s| {
        s.world_mut()
            .resource::<vd_sim::runtime::InboundBox>()
            .0
            .iter()
            .filter(|m| match m {
                vd_sim::io::Inbound::Wire {
                    class: vd_sim::io::MsgClass::SignalDelta,
                    bytes,
                    ..
                } => matches!(
                    postcard::from_bytes::<InterShardFlow>(bytes),
                    Ok(InterShardFlow::EntityInterest(_) | InterShardFlow::EntityCascade(_))
                ),
                _ => false,
            })
            .count()
    })
}

/// The pose of `entity` in a shard's client-edge feed, if it is in there at all — and it must be in there
/// AT MOST ONCE. A row arriving twice from one shard is the ping-pong failure this lane's origin tag
/// exists to prevent, and it would otherwise look exactly like a working feed.
fn edge_row(
    rows: &[vd_wire::channels::EntitySnap],
    entity: EntityId,
) -> Option<vd_core::pose::StampedPose> {
    let hits: Vec<_> = rows.iter().filter(|r| r.entity == entity).collect();
    assert!(
        hits.len() <= 1,
        "an entity appears at most once in one shard's feed; {} copies means a batch came back to the \
         level it was sent from",
        hits.len(),
    );
    hits.first().map(|r| r.pose)
}

/// THE ACCEPTANCE STORY FOR THE OCCUPANTS, under the Step 5 slice E contract (owner-decided, design
/// §3/§8.2): two players standing in two different realms, and each one's client is handed ITS OWN
/// realm's occupants and NOBODY else's figure. The other player's whereabouts are visible as A REALM
/// — the occupied area's box, live in the scene — never as an avatar. SL2 at steady state: an
/// occupant's pose exists on the shard that owns them and on that shard's own clients, full stop.
///
/// This scenario used to assert the opposite (each edge handed BOTH players, restated hop by hop by
/// the entity relay). That relay shipped occupant poses across realm boundaries — the breach that
/// condemned it — and the owner accepted the visual loss: at parent scale a sibling realm's
/// occupants are sub-child-resolution, and the occupied realm IS their proxy (SL7). During a real
/// crossing a client holds subs on BOTH shards; the LEAVER'S OWN avatar rides them cleanly (the
/// one-space filter exempts it — the ride/round-trip gates cover exactly that), while a BYSTANDER's
/// view of the leaver is the retained ghost's frozen fill until slice F lands the leaver-vanish
/// eviction (D-4(a) escalation).
#[test]
fn two_players_in_two_realms_are_each_drawn_only_by_their_own_realm() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, stayer, traveller) = boot_the_chain_with_two_players(&fabric);

    // Hold the stayer where it is and walk the traveller OUT of the area box. The crossing is driven by
    // the real seed-forest detector and carried by the real saga; nothing here hands anything to anyone.
    let staying = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);
    let leaving = DVec3::new(TRAVELLER_FROM_AREA_M, 0.0, 0.0);
    let crossed = step_until(&mut topo, 600, |t| {
        set_shard_subject_pose_now(t, SHARD, stayer, CHAIN_AREA_FRAME, staying);
        set_shard_subject_pose_now(t, SHARD, traveller, CHAIN_AREA_FRAME, leaving);
        let r = t.inspect_all();
        report(&r, CHAIN_MID)
            .held_entities
            .iter()
            .any(|(e, _)| *e == traveller)
    });
    assert!(
        crossed,
        "the traveller re-homes from the area to its planet"
    );

    // THE REMOVE MESSAGE's precondition, pinned at the crossing instant (D-4(a)): the BYSTANDER —
    // the client whose own avatar is the stayer — HOLDS the traveller's delivered track from their
    // co-located time. This is the track that froze forever before the remove message existed, so
    // its presence here is what makes the eviction assert at the end non-vacuous.
    let bystander_node = [CLIENT, CLIENT_B]
        .into_iter()
        .find(|n| with_client_at(&mut topo, *n, |c| c.delivered_view.own_entity()) == Some(stayer))
        .expect("one of the two clients owns the stayer");
    assert!(
        with_client_at(&mut topo, bystander_node, |c| c
            .delivered_view
            .render(f64::from(u32::MAX))
            .contains_key(&traveller)),
        "the bystander holds the traveller's track from their co-located time",
    );

    // The composition proof (anti-vacuity, causal): the area's own detector emitted the crossing and the
    // orchestrator turned it into a real saga. There is no `trigger_transfer` anywhere in this file.
    let reports = topo.inspect_all();
    assert!(report(&reports, SHARD).crossings_requested >= 1);
    assert!(report(&reports, ORCH).crossings_started >= 1);

    // Hold BOTH: the stayer in the area's frame on the area shard, the traveller in the planet's frame
    // on the planet shard. Each is scripted only on the shard that owns it; what each level's client
    // edge says — and refuses to say — is what is under test.
    let held = DVec3::new(TRAVELLER_FROM_PLANET_M, 0.0, 0.0);
    let planet_frame = FrameRef::PlanetCentered { planet_seed: 7 };
    let mut area_feed = Vec::new();
    let mut planet_feed = Vec::new();
    let both = step_until(&mut topo, 600, |t| {
        set_shard_subject_pose_now(t, SHARD, stayer, CHAIN_AREA_FRAME, staying);
        set_shard_subject_pose_now(t, CHAIN_MID, traveller, planet_frame, held);
        area_feed = entity_rows_at_the_client_edge(t, SHARD);
        planet_feed = entity_rows_at_the_client_edge(t, CHAIN_MID);
        edge_row(&area_feed, stayer).is_some() & edge_row(&planet_feed, traveller).is_some()
    });
    assert!(
        both,
        "each level's client edge streams its OWN player: area feed {area_feed:?}, planet feed \
         {planet_feed:?}",
    );
    // SETTLE, then read: give any straggler from the crossing window (the second sub's feed, the
    // retained ghost fill) time to close, so the absence below is steady-state absence and not a
    // lucky early read.
    const SETTLE_TICKS: u64 = 40;
    for _ in 0..SETTLE_TICKS {
        topo.step();
        set_shard_subject_pose_now(&mut topo, SHARD, stayer, CHAIN_AREA_FRAME, staying);
        set_shard_subject_pose_now(&mut topo, CHAIN_MID, traveller, planet_frame, held);
        area_feed = entity_rows_at_the_client_edge(&mut topo, SHARD);
        planet_feed = entity_rows_at_the_client_edge(&mut topo, CHAIN_MID);
    }

    // (1) EACH EDGE DRAWS ITS OWN PLAYER, in its own space, at the scripted point — the half that
    // must keep working exactly as before.
    let stayer_here = edge_row(&area_feed, stayer).expect("the area draws its own player");
    assert_eq!(stayer_here.frame, CHAIN_AREA_FRAME);
    assert_eq!(stayer_here.pos.offset(), staying);
    let traveller_there =
        edge_row(&planet_feed, traveller).expect("the planet draws its own player");
    assert_eq!(traveller_there.frame, planet_frame);
    assert_eq!(traveller_there.pos.offset(), held);

    // (2) AND NEITHER EDGE CARRIES THE OTHER'S FIGURE — the accepted loss, stated as the
    // measurement it is. A row for the other realm's occupant here would mean a pose crossed a
    // realm boundary at steady state: the exact breach slice E deleted.
    assert!(
        edge_row(&area_feed, traveller).is_none(),
        "the area's edge carries NO figure for the planet's occupant — SL2 at steady state",
    );
    assert!(
        edge_row(&planet_feed, stayer).is_none(),
        "and the planet's edge carries NO figure for the area's occupant",
    );

    // (3) THE PROXY THAT REPLACES THE FIGURE (SL7) is a CONJUNCTION, and each probe below carries
    // one half: the drawn set proves the area's BOX is on the traveller's client (pure AoI
    // visibility — it would be drawn empty or full), and the BIT proves the area is genuinely
    // OCCUPIED-live (the box on screen is a realm holding somebody). Together: the stayer's
    // whereabouts reach the traveller as the occupied area itself, with error bounded by the area's
    // own size — the resolution the parent's decision is meaningful at.
    assert!(
        child_bit(&mut topo, CHAIN_MID, CHAIN_AREA).is_some(),
        "the occupied area's bit beats at the planet",
    );
    let traveller_scene_has_area = with_shard(&mut topo, CHAIN_MID, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::RenderSent>()
            .0
            .values()
            .any(|drawn| drawn.contains(&CHAIN_AREA))
    });
    assert!(
        traveller_scene_has_area,
        "the area's BOX is in the planet occupant's drawn scene (the visibility half of the proxy)",
    );

    // (4) THE NO-LEAK HALF, unchanged. Every level of the chain is asked about every realm in the
    // whole seed forest, and every one of them may answer for ITSELF and its OWN DIRECT CHILDREN
    // and nothing else. A level that could place its own parent would be a level that had been told
    // where it sits.
    for (node, own, children) in [
        (SHARD, "Area(7)", vec![]),
        (CHAIN_MID, "Planet(7)", vec!["Area(7)"]),
        (CHAIN_TOP, "System(7)", vec!["Planet(7)", "Station(7)"]),
    ] {
        for (realm, answered) in placeable_frames(&mut topo, node) {
            let allowed = realm.as_str() == own || children.contains(&realm.as_str());
            assert_eq!(
                answered, allowed,
                "{node:?} answered {realm} = {answered}; a shard knows where IT is only for itself \
                 and its own direct children, and nothing about any ancestor",
            );
        }
    }

    // (5) THE LANE IS SILENT, measured on the real topology over a real window — the deletion's
    // whole price tag: where entity-count-sized batches crossed every boundary of a live chain per
    // tick, nothing crosses at all. Frames of the tombstoned arms still DECODE (reserved
    // discriminants), so zero here means "nothing was sent", never "nothing could be read".
    const WINDOW_TICKS: u64 = 100;
    let mut lane_frames = 0usize;
    for _ in 0..WINDOW_TICKS {
        topo.step();
        set_shard_subject_pose_now(&mut topo, SHARD, stayer, CHAIN_AREA_FRAME, staying);
        set_shard_subject_pose_now(&mut topo, CHAIN_MID, traveller, planet_frame, held);
        for node in [SHARD, CHAIN_MID, CHAIN_TOP] {
            lane_frames += entity_relays_delivered_to(&mut topo, node);
        }
    }
    println!(
        "[entities across realms] over {WINDOW_TICKS} ticks on a 3-level chain with 2 players: \
         {lane_frames} entity-lane frames delivered (the deleted lane's whole remaining cost)",
    );
    assert_eq!(
        lane_frames, 0,
        "the tombstoned entity lane is SILENT on every leg while two realms are occupied",
    );

    // (6) THE REMOVE MESSAGE, end to end (proto_minor 14, D-4(a)): when the traveller's retained
    // ghost tears down at the band's destroy edge, the area emits `EntityRemoved`, the gateway
    // fans it as the reliable Event, and the BYSTANDER's real client EVICTS the leaver's track —
    // the figure VANISHES instead of freezing at the boundary forever. The whole lane, through
    // the production shard, gateway and client, measured on the delivered view.
    let vanished = step_until(&mut topo, 600, |t| {
        set_shard_subject_pose_now(t, SHARD, stayer, CHAIN_AREA_FRAME, staying);
        set_shard_subject_pose_now(t, CHAIN_MID, traveller, planet_frame, held);
        !with_client_at(t, bystander_node, |c| {
            c.delivered_view
                .render(f64::from(u32::MAX))
                .contains_key(&traveller)
        })
    });
    assert!(
        vanished,
        "the bystander's drawn copy of the leaver is EVICTED by the remove message — never a \
         frozen phantom at the boundary",
    );
    // And it STAYS gone. (On this ordered in-memory fabric no straggler exists to refuse — the
    // resurrect guard's straggler/return/prune arms are pinned in the vd-client unit tests; this
    // window measures only that nothing on the LIVE topology re-creates the evicted track.)
    for _ in 0..20 {
        topo.step();
        set_shard_subject_pose_now(&mut topo, SHARD, stayer, CHAIN_AREA_FRAME, staying);
        set_shard_subject_pose_now(&mut topo, CHAIN_MID, traveller, planet_frame, held);
    }
    assert!(
        !with_client_at(&mut topo, bystander_node, |c| {
            c.delivered_view
                .render(f64::from(u32::MAX))
                .contains_key(&traveller)
        }),
        "no straggler resurrects the evicted figure",
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// SLICE 10 — THE PRICE OF THE CHAIN, PUBLISHED.
//
// Every value a client draws now crosses one shard per level of the live chain instead of leaving the
// shard it was authored on and going straight out. That is the trade the ground rule deliberately makes:
// only the party that authored a placement may apply it, so the value has to visit that party. Nothing
// measured that cost. The one latency gate that ever sat over the render path measured the ROUTER's
// per-viewer re-expression, and that work does not happen any more — so the tree carried a gate over
// something that no longer exists and none over what replaced it.
//
// These gates measure the replacement, in the unit the cost is actually paid in: TICKS. A tick is the
// structural quantum here (`Topology::step` pumps deliveries at tick start, then steps nodes, so a message
// sent during tick N lands no earlier than N+1) and it is the same number on any machine, which a
// wall-clock microsecond in a debug build is not.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The window every figure below is sampled over. Long enough that a p99 has ~2 samples of tail in it,
/// short enough that the whole gate is one of the cheap tests in the file.
const CHAIN_LATENCY_TICKS: u64 = 200;

/// Levels of the live chain ABOVE the realm the player is standing in, on this fixture: the planet that
/// carries the area, and the star that carries the planet. Every per-hop budget below is stated against
/// this rather than against a written-out number of ticks, so a fixture that grows a level moves its own
/// budget with it and cannot silently start tolerating an extra hop.
const CHAIN_LEVELS_ABOVE_LEAF: u64 = 2;

/// Hops between the shard a player is standing on and that player's screen: shard → gateway, gateway →
/// client. Both are ordinary fabric deliveries and each costs the same one tick as a shard-to-shard leg;
/// naming them separately is what keeps the client-side budget a DERIVATION of the topology instead of a
/// number fitted to the measurement.
const CLIENT_EDGE_HOPS: u64 = 2;

/// The universe tick a node's own clock currently reads — and, for the shard that AUTHORS the realm lane,
/// the instant its rows this tick are stamped at (`emit_realm_frames` authors against exactly this value).
/// The universe clock is cluster-wide, so a difference of two such readings is real elapsed time.
fn universe_now(topo: &mut Topology, node: NodeId) -> UniverseTick {
    with_shard(topo, node, |s| {
        s.world_mut()
            .resource::<vd_sim::runtime::ClockSample>()
            .universe_tick
    })
}

/// The instant the shard that OWNS a dot has it stamped at, right now — the up leg's author-side reading.
///
/// WHY AGE IS MEASURED AGAINST THIS AND NOT AGAINST THE RECEIVER'S OWN CLOCK. A pose is stamped when it is
/// measured, and in this fixture that write happens between steps, so the value the leaf ships during a
/// step already carries the previous tick's instant. Reading `receiver.now − stamp` therefore reports one
/// tick that no hop spent, and quoting it as transport would put a fixture's stamping convention into a
/// production budget. Comparing the two ENDS of the same lane — what the author holds now against what the
/// consumer holds now — cancels the convention exactly and leaves the transit.
///
/// It was measured the other way first, and every series came out exactly one tick above its hop count
/// with zero spread, which is what a constant offset looks like and is how this was found.
fn own_dot_stamp(topo: &mut Topology, node: NodeId, subject: EntityId) -> Option<UniverseTick> {
    with_shard(topo, node, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::Dots>()
            .0
            .values()
            .find(|d| d.entity == subject)
            .map(|d| d.pose.universe_tick)
    })
}

/// One measured age series, reduced to the three numbers a budget is argued from.
struct Ages {
    /// How many samples the series holds. Quoted because a p99 over an empty set is zero, and zero would
    /// pass any `<= budget` assertion while proving that nothing ever arrived.
    n: usize,
    p50_ticks: u64,
    p99_ticks: u64,
    worst_ticks: u64,
}

/// Reduce a series of tick ages through the ONE percentile implementation the other two latency gates use
/// (`vd_harness::latency::percentile_unstable`, shared so SPIKE-2a, SPIKE-3a and this gate can never drift
/// in what "p99" means). It takes `Duration`s, so the ages ride through it as ticks-as-nanoseconds and come
/// back out as ticks — an exact integer round trip, no rate assumed anywhere in the reduction.
fn ages(samples: &[u64]) -> Ages {
    let durations: Vec<std::time::Duration> = samples
        .iter()
        .map(|t| std::time::Duration::from_nanos(*t))
        .collect();
    Ages {
        n: samples.len(),
        p50_ticks: vd_harness::latency::percentile_unstable(durations.clone(), 50).as_nanos()
            as u64,
        p99_ticks: vd_harness::latency::percentile_unstable(durations.clone(), 99).as_nanos()
            as u64,
        worst_ticks: vd_harness::latency::percentile_unstable(durations, 100).as_nanos() as u64,
    }
}

/// A tick age rendered in milliseconds at the cluster's real tick rate, so the figure can be read against
/// a human budget without the assertions depending on a rate.
#[allow(clippy::cast_precision_loss)] // ages in the single digits of ticks
fn ms(ticks: u64) -> f64 {
    ticks as f64 * vd_tests::area_stub_config().tick_dt_s * 1000.0
}

/// THE PER-LEVEL CHAIN LATENCY GATE — what the ground rule costs, up-leg and down-leg, in ticks.
///
/// WHY THIS GATE HAD TO BE BUILT. The only latency gate that ever covered the render path measured the
/// router re-expressing every body into every viewer's space. That work is gone: the chain hands each
/// client values already measured from the centre of the realm its player stands in, so the router's
/// remaining job is a leading-varint re-tag and a refcounted fan. The cost did not vanish, it MOVED — one
/// conversion and one shard hop per level of the live chain — and until this gate nothing anywhere
/// measured the thing that replaced the thing that was measured.
///
/// WHAT IS MEASURED. Age, in ticks, of a value at the party that consumes it, taken as `the newest instant
/// the AUTHOR of this lane has put on the wire − the instant the consumer is currently holding`, both read
/// at the same moment. Comparing the two ENDS of one lane is what makes the figure pure transit: a stamping
/// convention that shifts one end shifts the other identically and cancels, where `receiver's clock −
/// stamp` would fold it in. See [`own_dot_stamp`] for the measurement that established that. The universe
/// clock is cluster-wide, so the difference is elapsed time. Two legs, on ONE run of ONE topology:
///
/// * UP — the SL7 liveness beat, re-originated per level (the pose relay is dead — Step 5): the area's
///   bit at the planet, the planet's bit at the star. This is the leg the parents' warm-ahead and
///   cascade-targeting decisions ride on.
/// * DOWN — a turning station authored by the STAR, descending to the planet, to the area, and out to the
///   client. This is the leg the player sees the world move over.
///
/// THE BUDGET IS DERIVED, NOT FITTED. `Topology::step` pumps deliveries at tick start and then steps
/// nodes, so a message sent during tick N is delivered no earlier than N+1; and the shard schedule folds
/// an arriving relay in the SAME tick it lands (`process_inbound` opens group A, `evaluate_realm_aoi`
/// closes the tick after `emit_realm_frames`), so a re-relay leaves on the tick it arrived. One tick per
/// hop, therefore, and the budget is the hop count: [`CHAIN_LEVELS_ABOVE_LEAF`] for the up leg, plus
/// [`CLIENT_EDGE_HOPS`] for the leg that reaches a screen. If a level ever starts holding a relay for a
/// tick — a cadence gate, a batching buffer, a schedule reorder — this fails with the hop it cost.
#[test]
fn the_chain_pays_one_tick_per_level_each_way_and_the_price_is_measured() {
    let fabric = FaultFabric::new(4242, 2);
    let movers = BTreeMap::from([(CHAIN_STATION, station_orbit())]);
    let (mut topo, subject) = boot_the_chain_with(&fabric, &movers);
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);

    // WARM UP until BOTH legs are actually running end to end. Measuring before the chain is up would
    // fold the one-off spin-up into the steady-state figure and quote it as the per-hop price.
    let running = step_until(&mut topo, 600, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        child_bit(t, CHAIN_TOP, PLANET).is_some()
            && with_client(t, |c| c.realm_view.realm_newest_tick(CHAIN_STATION)).is_some()
    });
    assert!(
        running,
        "both legs of the chain are live before their price is quoted: the player's pose has reached the \
         star, and the star's station has reached the client",
    );

    // THE MEASUREMENT. One pass, every series sampled on the same ticks, so the up and down figures
    // describe one run of one cluster rather than two runs that might have settled differently.
    //
    // Every age is `what the AUTHOR of this lane holds right now − what this consumer holds right now`,
    // both read at the same instant, straight after the step. For the up leg the author is the area shard
    // holding the player's own dot; for the down leg it is the star, whose clock IS the instant its rows
    // this tick are authored at.
    let mut up: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut down_hop: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut down_client: Vec<u64> = Vec::new();
    // What each level below the star currently holds about the station. Carried ACROSS ticks rather than
    // sampled only on arrival ticks, because staleness at a consumer is how old what it is holding is —
    // a level that stopped receiving would otherwise contribute no samples and look perfect.
    let mut held_down: BTreeMap<NodeId, UniverseTick> = BTreeMap::new();
    for _ in 0..CHAIN_LATENCY_TICKS {
        set_shard_subject_pose_now(&mut topo, SHARD, subject, CHAIN_AREA_FRAME, at);
        // THE STAR'S AUTHORING INSTANT, sampled BEFORE the step because that is the value the rows this
        // step ships are stamped with. MEASURED, not assumed: a row landing at the planet on the step
        // whose post-step clock reads 19 carries 17, and all three shards' clocks are in exact lockstep,
        // so the row was authored one tick behind the authoring shard's own post-step clock.
        //
        // THAT IS A REAL FINDING AND IT IS NOT THIS FIXTURE. `register_clock_follower` installs
        // `observe_clock_syncs` with no ordering against the stub's chained authoring groups, so Bevy is
        // free to place it either side of `emit_realm_frames` — and here it lands AFTER, which means every
        // realm row every shard authors carries the clock value it held before that step's sync. It costs
        // one tick of world age on the realm lane everywhere, independent of chain depth, and it is
        // invisible on one shard because nobody has a second clock to compare against. Reported, not
        // fixed: an ordering constraint on a production schedule is not this slice's to change.
        let authored_down = universe_now(&mut topo, CHAIN_TOP);
        topo.step();

        // UP: how stale each level's newest CHILD-BIT is, per hop. The pose relay is deleted
        // (Step 5) and the bit is RE-ORIGINATED at every level — nothing forwards a stamp upward —
        // so each sample is a one-hop age: the receiver's "now" (the leaf's own stamp, one shared
        // universe clock) minus the tick the SENDING CHILD stamped at emit. Depth says WHERE the
        // sample was taken, never how many hops the stamp travelled (always one).
        let authored_up =
            own_dot_stamp(&mut topo, SHARD, subject).expect("the area holds its own dot");
        for (node, child, depth) in [(CHAIN_MID, CHAIN_AREA, 1u64), (CHAIN_TOP, PLANET, 2)] {
            if let Some(bit) = child_bit(&mut topo, node, child) {
                up.entry(depth)
                    .or_default()
                    .push(authored_up.0.saturating_sub(bit.at.0));
            }
        }

        // DOWN, per shard hop: how far behind the star's newest authored instant the newest station row
        // each level has been handed is. Read out of each receiving host's own inbox — wire truth.
        for (node, depth) in [(CHAIN_MID, 1u64), (SHARD, 2)] {
            for d in cascades_delivered_to(&mut topo, node) {
                if d.realms.iter().any(|r| r.realm == CHAIN_STATION) {
                    let slot = held_down.entry(node).or_insert(d.universe_tick);
                    *slot = (*slot).max(d.universe_tick);
                }
            }
            if let Some(held) = held_down.get(&node) {
                down_hop
                    .entry(depth)
                    .or_default()
                    .push(authored_down.0.saturating_sub(held.0));
            }
        }

        // DOWN, at the screen: the newest station instant the PRODUCTION realm consumer has accepted.
        // This is the number a player would feel, and the only one of the three that also carries the two
        // client-edge hops.
        if let Some(seen) =
            with_client(&mut topo, |c| c.realm_view.realm_newest_tick(CHAIN_STATION))
        {
            down_client.push(authored_down.0.saturating_sub(seen.0));
        }
    }

    let up1 = ages(up.get(&1).map_or(&[][..], Vec::as_slice));
    let up2 = ages(up.get(&2).map_or(&[][..], Vec::as_slice));
    let dn1 = ages(down_hop.get(&1).map_or(&[][..], Vec::as_slice));
    let dn2 = ages(down_hop.get(&2).map_or(&[][..], Vec::as_slice));
    let dnc = ages(&down_client);
    let hz = 1.0 / vd_tests::area_stub_config().tick_dt_s;
    println!(
        "[chain latency] {CHAIN_LATENCY_TICKS} ticks at {hz:.0} Hz, 3-level chain\n\
         [chain latency]   UP   depth 1 (planet)  n={:3}  p50 {} tick ({:.0} ms)  p99 {} ({:.0} ms)  max {}\n\
         [chain latency]   UP   depth 2 (star)    n={:3}  p50 {} tick ({:.0} ms)  p99 {} ({:.0} ms)  max {}\n\
         [chain latency]   DOWN depth 1 (planet)  n={:3}  p50 {} tick ({:.0} ms)  p99 {} ({:.0} ms)  max {}\n\
         [chain latency]   DOWN depth 2 (area)    n={:3}  p50 {} tick ({:.0} ms)  p99 {} ({:.0} ms)  max {}\n\
         [chain latency]   DOWN at the CLIENT     n={:3}  p50 {} tick ({:.0} ms)  p99 {} ({:.0} ms)  max {}",
        up1.n,
        up1.p50_ticks,
        ms(up1.p50_ticks),
        up1.p99_ticks,
        ms(up1.p99_ticks),
        up1.worst_ticks,
        up2.n,
        up2.p50_ticks,
        ms(up2.p50_ticks),
        up2.p99_ticks,
        ms(up2.p99_ticks),
        up2.worst_ticks,
        dn1.n,
        dn1.p50_ticks,
        ms(dn1.p50_ticks),
        dn1.p99_ticks,
        ms(dn1.p99_ticks),
        dn1.worst_ticks,
        dn2.n,
        dn2.p50_ticks,
        ms(dn2.p50_ticks),
        dn2.p99_ticks,
        ms(dn2.p99_ticks),
        dn2.worst_ticks,
        dnc.n,
        dnc.p50_ticks,
        ms(dnc.p50_ticks),
        dnc.p99_ticks,
        ms(dnc.p99_ticks),
        dnc.worst_ticks,
    );

    // ANTI-VACUITY FIRST. `percentile_unstable` is total: an empty series reduces to zero, and zero would
    // sail through every budget below while proving that not one value ever made the trip.
    for (name, a) in [
        ("up depth 1", &up1),
        ("up depth 2", &up2),
        ("down depth 1", &dn1),
        ("down depth 2", &dn2),
        ("down at the client", &dnc),
    ] {
        assert!(
            a.n > 0,
            "{name}: no sample at all — a zero p99 here would be an empty measurement passing as a fast one",
        );
    }

    // THE BUDGETS, each the hop count the topology forces and nothing more. The two measured levels ARE
    // the levels above the leaf, tied together here so a fixture that grows a level cannot leave the
    // client-side budget below quietly describing the old shape.
    assert_eq!(
        u64::try_from(up.len()).expect("two depths"),
        CHAIN_LEVELS_ABOVE_LEAF,
        "the up leg was sampled at every level above the leaf",
    );
    // THE UP BUDGET IS ONE TICK PER HOP AT EVERY LEVEL — depth never widens it, because the bit is
    // re-originated per level (each sample above is a one-hop age by construction). A depth-scaled
    // budget here would be slack at every level past the first and could hide a level sitting on a
    // beat for a whole extra tick. The END-TO-END recursion is bounded elsewhere: each level's bit
    // EXISTS only while its child's bit is fresh within the one TTL, and the climb scenario gates
    // the chain forming at all.
    for (depth, a) in [(1u64, &up1), (2, &up2)] {
        assert!(
            a.p99_ticks <= 1,
            "the up leg costs more than one tick per hop: at depth {depth} the p99 age is \
             {} ticks ({:.0} ms) against the one-hop budget ({:.0} ms). A level is holding a \
             beat instead of retaining it on the tick it arrives.",
            a.p99_ticks,
            ms(a.p99_ticks),
            ms(1),
        );
    }
    for (depth, a) in [(1u64, &dn1), (2, &dn2)] {
        assert!(
            a.p99_ticks <= depth,
            "the down leg costs more than one tick per level: at depth {depth} the p99 age on arrival is \
             {} ticks ({:.0} ms) against a derived budget of {depth} ({:.0} ms).",
            a.p99_ticks,
            ms(a.p99_ticks),
            ms(depth),
        );
    }
    let to_screen = CHAIN_LEVELS_ABOVE_LEAF + CLIENT_EDGE_HOPS;
    assert!(
        dnc.p99_ticks <= to_screen,
        "the world reaches the screen later than one tick per hop allows: p99 {} ticks ({:.0} ms) against \
         a derived budget of {to_screen} ({:.0} ms) = {CHAIN_LEVELS_ABOVE_LEAF} shard levels + \
         {CLIENT_EDGE_HOPS} client-edge hops.",
        dnc.p99_ticks,
        ms(dnc.p99_ticks),
        ms(to_screen),
    );
}

/// The per-attempt drop the lossy half induces on EVERY leg of the down chain.
///
/// A HALF is chosen deliberately: at a tenth the compounding is inside the noise of a 200-tick window, and
/// at a ninth-tenths nothing arrives anywhere and every depth reads the same. A half is the value at which
/// the difference BETWEEN depths is the largest thing in the measurement, which is the quantity under test.
const CHAIN_DOWN_DROP_P: f64 = 0.5;

/// THE COMPOUNDING GATE — what a lossy link costs a value that now has to visit one shard per level.
///
/// THE QUESTION THIS EXISTS FOR. The relay legs are `FireAndForget` + `Unreliable` on
/// `MsgClass::SignalDelta`. On a lane like that, a per-hop delivery probability `q` gives `q^depth` end to
/// end, and the retain window that today bridges ONE missing datagram has to bridge one per level. Whether
/// a drawn position may ride that lane at all is an owner decision, and it should be taken with numbers.
///
/// ⚠ WHAT THIS HARNESS CAN AND CANNOT SAY, stated before the numbers so they are not over-read. The
/// fabric's `drop_p` drops a delivery ATTEMPT and leaves the message in its unacked ledger to be
/// redelivered (`FaultFabric::drop_policy_loses_attempts_never_messages` is its own test of that). So a
/// "lost" relay here arrives LATE, never never. **The `q^depth` end-to-end delivery ratio of a genuinely
/// lossy datagram lane is therefore NOT MEASURABLE on this harness at any drop probability**, and no number
/// below should be read as one. What IS measured, and is the number that actually decides the question, is
/// how much STALENESS compounds per level: how old what each level is holding gets, and how often it gets
/// anything new. A drawn position rides this lane acceptably or not according to its age, and a late
/// datagram and a lost one are the same thing to a renderer.
///
/// Making the true-loss figure measurable means one more arm in `LinkPolicy` — a drop that does not
/// requeue — which is a change to shared Tier-A harness infrastructure and is deliberately not made here.
#[test]
fn the_chain_compounds_staleness_per_level_under_a_lossy_link() {
    let fabric = FaultFabric::new(4242, 2);
    let movers = BTreeMap::from([(CHAIN_STATION, station_orbit())]);
    let (mut topo, subject) = boot_the_chain_with(&fabric, &movers);
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);

    let running = step_until(&mut topo, 600, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        with_client(t, |c| c.realm_view.realm_newest_tick(CHAIN_STATION)).is_some()
    });
    assert!(
        running,
        "the down chain reaches the screen before it is made lossy"
    );

    // Sample the same three depths twice — once on a perfect link, once with EVERY leg of the descent
    // lossy — so the compounding is a difference between two measurements of one cluster rather than a
    // comparison against a remembered number.
    let measure = |topo: &mut Topology| -> [(f64, Ages); 3] {
        let mut age: [Vec<u64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut news = [0u64; 3];
        let mut held: [Option<UniverseTick>; 3] = [None; 3];
        for _ in 0..CHAIN_LATENCY_TICKS {
            set_shard_subject_pose_now(topo, SHARD, subject, CHAIN_AREA_FRAME, at);
            let authored = universe_now(topo, CHAIN_TOP);
            topo.step();
            for (i, node) in [CHAIN_MID, SHARD].into_iter().enumerate() {
                for d in cascades_delivered_to(topo, node) {
                    if d.realms.iter().any(|r| r.realm == CHAIN_STATION) {
                        let slot = held[i].get_or_insert(d.universe_tick);
                        *slot = (*slot).max(d.universe_tick);
                    }
                }
            }
            held[2] = with_client(topo, |c| c.realm_view.realm_newest_tick(CHAIN_STATION));
            for i in 0..3 {
                if let Some(h) = held[i] {
                    age[i].push(authored.0.saturating_sub(h.0));
                }
            }
            // "News" is counted from the age series itself: an age that did NOT grow by a tick means this
            // level was handed something newer on this tick. Deriving it from the same samples means the
            // two figures cannot disagree about what arrived.
            for i in 0..3 {
                let n = age[i].len();
                if n >= 2 && age[i][n - 1] <= age[i][n - 2] {
                    news[i] += 1;
                }
            }
        }
        #[allow(clippy::cast_precision_loss)] // counts bounded by the window
        let ratio = |i: usize| news[i] as f64 / CHAIN_LATENCY_TICKS as f64;
        [
            (ratio(0), ages(&age[0])),
            (ratio(1), ages(&age[1])),
            (ratio(2), ages(&age[2])),
        ]
    };

    let clean = measure(&mut topo);
    for (from, to) in [
        (CHAIN_TOP, CHAIN_MID),
        (CHAIN_MID, SHARD),
        (SHARD, vd_tests::GATEWAY),
        (vd_tests::GATEWAY, CLIENT),
    ] {
        fabric.set_policy(
            from,
            to,
            vd_harness::fabric::LinkPolicy {
                drop_p: CHAIN_DOWN_DROP_P,
                ..vd_harness::fabric::LinkPolicy::default()
            },
        );
    }
    let lossy = measure(&mut topo);

    let names = ["depth 1 (planet)", "depth 2 (area)  ", "depth 3 (CLIENT)"];
    println!(
        "[chain loss] down leg, {CHAIN_LATENCY_TICKS} ticks per arm, {CHAIN_DOWN_DROP_P} attempt-drop on \
         every leg of the descent"
    );
    for i in 0..3 {
        println!(
            "[chain loss]   {}  clean: news {:.2}/tick  p50 age {} ({:.0} ms)  p99 {} ({:.0} ms)   |   \
             lossy: news {:.2}/tick  p50 age {} ({:.0} ms)  p99 {} ({:.0} ms)  max {}",
            names[i],
            clean[i].0,
            clean[i].1.p50_ticks,
            ms(clean[i].1.p50_ticks),
            clean[i].1.p99_ticks,
            ms(clean[i].1.p99_ticks),
            lossy[i].0,
            lossy[i].1.p50_ticks,
            ms(lossy[i].1.p50_ticks),
            lossy[i].1.p99_ticks,
            ms(lossy[i].1.p99_ticks),
            lossy[i].1.worst_ticks,
        );
    }
    // The ARITHMETIC the harness cannot produce, printed beside the measurement and labelled as
    // arithmetic, because it is the figure the owner decision actually turns on.
    let q = 1.0 - CHAIN_DOWN_DROP_P;
    println!(
        "[chain loss]   ARITHMETIC, NOT MEASURED — on a lane that truly loses, per-hop delivery {q:.2} \
         gives {:.2} at depth 1, {:.2} at depth 2, {:.2} at depth 3. This harness redelivers, so the \
         measured columns above are staleness, not loss.",
        q,
        q * q,
        q * q * q,
    );

    // The clean arm is the anti-vacuity control: without it a lossy arm that measured nothing at all would
    // look like a lossy arm that measured a healthy link.
    assert!(
        clean[2].0 > 0.5,
        "the clean control actually delivered: the screen took news on {:.2} of ticks",
        clean[2].0,
    );
    // THE PROPERTIES, not the numbers. (1) the induced loss REACHED the lane — without this the whole
    // lossy arm could be measuring a healthy link and reporting it as a loss result. (2) staleness
    // COMPOUNDS: each level is strictly further behind than the one above it, which is what "in series"
    // means and what a chain that had quietly collapsed into one hop would fail.
    assert!(
        lossy[2].0 < clean[2].0,
        "the induced loss reached the lane: the screen took news on {:.2} of ticks lossy vs {:.2} clean",
        lossy[2].0,
        clean[2].0,
    );
    assert!(
        lossy[2].1.p50_ticks > lossy[1].1.p50_ticks,
        "staleness compounds past the second level: depth 3 p50 {} against depth 2 p50 {}",
        lossy[2].1.p50_ticks,
        lossy[1].1.p50_ticks,
    );
    assert!(
        lossy[1].1.p50_ticks > lossy[0].1.p50_ticks,
        "staleness compounds past the first level: depth 2 p50 {} against depth 1 p50 {}",
        lossy[1].1.p50_ticks,
        lossy[0].1.p50_ticks,
    );
    assert!(
        with_client(&mut topo, |c| c.realm_view.realm_newest_tick(CHAIN_STATION)).is_some(),
        "the screen still holds the world across the gaps — losing it here is what would make a client \
         draw nothing rather than draw something slightly old",
    );
}
