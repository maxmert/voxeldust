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
use vd_core::glam::DVec3;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::{AccountId, EntityId, NodeId};
use vd_harness::client::ScriptedClient;
use vd_harness::fabric::FaultFabric;
use vd_harness::topology::{InspectReport, Topology};
use vd_physics::celestial::{G, OrbitalElements, orbital_state};
use vd_physics::worldgen::{UniverseConfig, WorldView, moving_children_for_config};
use vd_sim::stub::PlacementCarry;
use vd_tests::frame_fixture::{
    FAR_OCCUPANT_FROM_PLANET_M, FAR_SYSTEM_FROM_GALAXY_M, NEAR_PLANET_FROM_STAR_M, WorkedExample,
};
use vd_tests::{
    CHAIN_MID, CHAIN_TOP, DEST, FRAME_UNIVERSE_SEED, GATEWAY, ORCH, SHARD, live_sagas, p1_client,
    p1_cluster, p2_cluster_area_in_planet_in_system, p2_cluster_planet_in_system, plant_regions,
    plant_seed_neighbourhood, plant_seed_neighbourhood_with_movers, set_shard_subject_offset,
    set_shard_subject_pose_now, set_shard_subject_pose_now_with, walk_forward,
};
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
/// Flatten a pose's lattice position to metres in its own frame (poses ride NORMALIZED since the
/// cell activation — `.offset()` raw is a sub-cell residual, never a position).
fn pose_m(p: &vd_core::pose::StampedPose) -> vd_core::glam::DVec3 {
    p.pos
        .delta_m(vd_core::pose::LatticePos::ORIGIN, p.frame.tier())
}

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
        pose_m(&arrived),
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
        pose_m(&arrived),
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
    let from_centre = pose_m(&arrived).length();
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
// from its star. The world the game actually boots is nothing like that shape — and this section
// deliberately refuses to say what shape it IS. Every quantity below is read out of the shipped
// generator AT USE, at the seed the shipped boot defaults to, and the run PRINTS the two that decide the
// outcome. Transcription is precisely what went wrong here, repeatedly: the copies this section used to
// carry stayed at the pre-S4 values when the apoapsis re-solve moved the world, and then the in-system
// true-size re-solve deleted the compressed geometry underneath them and the derived stellar mass cap
// re-drew every star on top of that. Each time the prose went on describing a world that no longer
// existed while the fixture quietly proved an EASIER arrival than the game ships (batch review, MAJOR —
// the exact drift class the walk gates were indicted for).
//
// WHY IT MATTERS, and it is the reason a green suite sat beside an unplayable game all day: the entry
// margin is FIXED. It is one number — the shipped containment inset — and it is the same number whatever
// the body it is applied to. So what decides whether an arrival lands INSIDE is the RATIO of that margin
// to the planet's own authority sphere, and nothing else. On the walk world's planet the margin is a
// small fraction of the body, and an arrival can be a metre or two out and still land comfortably inside.
// Where the ratio is large, the same arithmetic error puts the occupant OUTSIDE the realm that just
// accepted it — whereupon that realm hands it straight back, the parent hands it down again, and the
// player is trapped in the loop the owner flew. The RATIO is the whole point; what it currently equals is
// a measurement the gate below prints, never a claim written up here.

/// The universe seed THE world is generated from: the shipped DEFAULT every world-deriving process
/// reads (`vd_physics::worldgen::HOME_SEED`, the seed a player logs into), not a pinned ruler.
///
/// WHY IT IS NOT A LITERAL HERE, when literals elsewhere in this file are right. A test that pins a seed
/// is measuring the GENERATOR — it wants the same draw every run and does not care which world that is.
/// This section is measuring the DESTINATION: the proportions of the realm players actually arrive at.
/// Those two want different things from a seed, and this one has to follow the default the boot reads, or
/// it gates a world nobody logs into. These call sites passed a bare `0` while the section's own prose
/// claimed to describe "the world the shipped boot builds", which is the same sentence disagreeing with
/// itself.
const DEMAND_SEED: u64 = vd_physics::worldgen::HOME_SEED;

/// THE world's config, exactly as the shipped boot builds it. The two arguments feed the AoI band
/// only — inert in these fixtures (`AoiConfig::inert()`); the geometry (shells, orbits, mass) is the
/// same for any values, so the walk fixtures' own speed/tick pair is passed for parity.
fn demand_config() -> UniverseConfig {
    UniverseConfig::world(15.0, 0.05)
}

/// THE SHIPPED CONTAINMENT BAND, built by the shipped builder from the shipped config's own edges —
/// the acquire inset and the release outset a real boot arms every region with.
///
/// Built, not written down. The inset IS the entry margin this whole section is about, and a hand-copied
/// pair of edges here would be the identical defect the section exists to catch: prose and fixture
/// agreeing with each other while both drift away from what boots.
fn demand_band() -> vd_core::geometry::ContainmentBand {
    demand_config()
        .band
        .build()
        .expect("the shipped containment band is valid")
}

/// The star system's own boundary in the shipped demand world — read off the generator's SOLVED
/// region roster. Since the in-system re-solve there IS no config radius to quote: the shell is
/// whatever the ONE clearance law solves for THAT system's own drawn star, so it is stated here only
/// by where it comes from. (`worldgen::TARGET_SYSTEM_BOUND_HOME_M` looks like the number to write
/// down and is not: its own doc calls itself a SEED-0 provenance marker, and says the default seed's
/// home system solves to some thirty-two times it.)
fn demand_system_soi_m() -> f64 {
    let cfg = demand_config();
    vd_physics::worldgen::realm_regions_for_config(DEMAND_SEED, &cfg)
        .iter()
        .find(|r| r.realm == SYSTEM)
        .expect("THE world rosters its home system")
        .shape
        .finite_extent()
}

/// The INNER planet's WHOLE authority sphere there — its gravitational SOI at its drawn mass
/// (D-REAL-1), read off the same roster, against the SAME entry margin ([`demand_band`]'s inset).
fn demand_planet_soi_m() -> f64 {
    let cfg = demand_config();
    let inner = moving_children_for_config(DEMAND_SEED, &cfg, SYSTEM)
        .into_iter()
        .min_by(|a, b| a.1.sma.total_cmp(&b.1.sma))
        .map(|(r, _)| r)
        .expect("THE world's home system authors movers");
    vd_physics::worldgen::realm_regions_for_config(DEMAND_SEED, &cfg)
        .iter()
        .find(|r| r.realm == inner)
        .expect("the inner planet is rostered")
        .shape
        .finite_extent()
}

/// THE world's own INNER mover (smallest semi-major axis), from the same generator call the shipped
/// boot makes — its orbit radius and its real central mass (so the sweep speed is the world's own).
fn demand_inner_elements() -> OrbitalElements {
    moving_children_for_config(DEMAND_SEED, &demand_config(), SYSTEM)
        .into_iter()
        .min_by(|a, b| a.1.sma.total_cmp(&b.1.sma))
        .map(|(_, e)| e)
        .expect("THE world's home system authors movers")
}

/// How far out the fixture's planet orbits — the real inner mover's semi-major axis.
fn demand_planet_orbit_m() -> f64 {
    demand_inner_elements().sma
}

/// The demand world's turn for the planet: circular, in-plane, phase at zero, on the REAL inner
/// mover's shell around the REAL central mass — so the planet sweeps the waiting occupant at the
/// speed the shipped world actually turns at (the draw-dependent shape — ecc, inclination, phase —
/// is stripped so the wait-point geometry stays deterministic across seeds).
fn demand_orbit() -> OrbitalElements {
    let real = demand_inner_elements();
    OrbitalElements {
        sma: real.sma,
        ecc: 0.0,
        inclination: 0.0,
        raan: 0.0,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: 0.0,
        central_mass: real.central_mass,
    }
}

/// The demand world's shape, three levels of it: an ambient root, the star system, and the planet.
///
/// The planet's stored centre is ZERO because it MOVES — its position comes from the star's ephemeris
/// every tick, and a stored centre beside a live placement would be counted twice. That is the shipped
/// generator's own rule for an orbiting body, reproduced here rather than invented.
fn demand_forest() -> Vec<vd_core::geometry::RealmRegion> {
    use vd_core::geometry::{AoiConfig, Boundary, RealmRegion};
    // The SHIPPED band, built by the shipped builder off the shipped config's own edges — the pair of
    // literals that used to sit here was a hand copy of exactly that.
    let band = demand_band();
    let root = RealmId::System(1); // the ambient root, as the live cluster names it
    // `System(1)` IS THE GALAXY, so its shell is the galaxy's own, read off the shipped config. It was a
    // flat 1.0e6 m — an ambient parent orders of magnitude SMALLER than the one star system nested
    // inside it. Nothing here ever asked the root to contain its child, which is the only reason a world
    // that inverted could sit in a passing gate; deriving it means the two levels cannot disagree again.
    let root_shell = Boundary::Shell {
        r: demand_config().scale.galaxy_r_m,
    };
    vec![
        RealmRegion {
            realm: root,
            center: vd_core::geometry::ParentCentre::authored(vd_core::pose::LatticePos::ORIGIN),
            frame: FrameRef::SystemSpace { system_seed: 1 },
            shape: root_shell,
            look: Some(root_shell),
            band,
            aoi: AoiConfig::inert(),
            interior_band: AoiConfig::inert(),
            parent: None,
        },
        RealmRegion {
            realm: SYSTEM,
            center: vd_core::geometry::ParentCentre::authored(vd_core::pose::LatticePos::ORIGIN),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            shape: Boundary::Shell {
                r: demand_system_soi_m(),
            },
            look: Some(Boundary::Shell {
                r: demand_system_soi_m(),
            }),
            band,
            aoi: AoiConfig::inert(),
            interior_band: AoiConfig::inert(),
            parent: Some(root),
        },
        RealmRegion {
            realm: PLANET,
            // ZERO: an orbiting body is placed LIVE by its parent, never from a stored centre.
            center: vd_core::geometry::ParentCentre::authored(vd_core::pose::LatticePos::ORIGIN),
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            shape: Boundary::Shell {
                r: demand_planet_soi_m(),
            },
            look: Some(Boundary::Shell {
                r: demand_planet_soi_m(),
            }),
            band,
            aoi: AoiConfig::inert(),
            interior_band: AoiConfig::inert(),
            parent: Some(SYSTEM),
        },
    ]
}

/// THE OWNER'S LOOP, at the proportions it actually happens at.
///
/// He flew into a planet and the planet and its star traded him back and forth until he could not get
/// out. Every gate in this file said the crossing was fine, because every gate in this file runs on the
/// WALK world's planet, whose size makes the fixed entry margin a small fraction of the body — while the
/// shipped world applies that same fixed margin to a body of an entirely different size. The two ratios
/// are MEASURED and printed side by side by this gate, never asserted about in prose: the ratio is the
/// thing that matters, and every prose copy of what it equals has already gone stale once.
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
    let out = DVec3::new(demand_planet_soi_m() * 3.0, 0.0, 0.0);
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
    let wait_at = DVec3::new(demand_planet_orbit_m(), 0.0, 0.0);
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
    let from_centre = pose_m(&arrived).length();
    let planet_soi_m = demand_planet_soi_m();
    // THE RATIO THE SECTION HEADER IS ABOUT, measured instead of described: the shipped entry margin
    // against the body it is applied to — here, and on the walk world every gate above runs on. Both
    // sides are read at use (the margin off the shipped band, the walk radius off the walk preset), so
    // neither can go stale the way every transcribed copy in this section already has.
    let margin_m = demand_band().inset();
    let walk_planet_soi_m = UniverseConfig::walk_scale().planet.planet_soi_r_m;
    println!(
        "[real-sized arrival] {from_centre:.6} m from the planet's centre, boundary \
         {planet_soi_m} m, entry margin {margin_m} m"
    );
    println!(
        "[real-sized entry margin] {margin_m} m of margin is {:.6} of this {planet_soi_m} m planet, \
         against {:.6} of the walk gates' {walk_planet_soi_m} m one",
        margin_m / planet_soi_m,
        margin_m / walk_planet_soi_m,
    );

    assert!(
        from_centre < planet_soi_m,
        "the occupant landed {from_centre} m from the centre of a {planet_soi_m} m planet — \
         OUTSIDE the realm that just accepted it. The planet sees that on its next tick and hands them \
         back; the star sees them inside its planet and hands them down; and the player cannot leave. \
         This is the owner's loop, and the walk-scale gates cannot see it: their planet is \
         {walk_planet_soi_m} m against the same {margin_m} m margin, so that margin is {:.6} of their \
         body and {:.6} of this one.",
        margin_m / walk_planet_soi_m,
        margin_m / planet_soi_m,
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
        !vd_physics::worldgen::realm_neighbourhood_for(FRAME_UNIVERSE_SEED, SYSTEM)
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
        vd_physics::celestial::orbital_state(
            &elements,
            vd_core::kinematics::secs_since_epoch(tick.0, tick_hz),
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
/// production `transfer_frame` over the production authored books, each addition made by the one party
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
    let at_system = transfer_frame(&start, fx.system_frame, &fx.placement_book(fx.system))
        .expect("the star places its own planet");
    // UP: the system ships "148.001, in my frame"; the GALAXY adds where it put that system.
    let at_galaxy = transfer_frame(&at_system, fx.galaxy_frame, &fx.placement_book(fx.galaxy))
        .expect("the galaxy places its own system");
    // DOWN: the GALAXY subtracts — it is the only party that holds the 1e13.
    let back_system = transfer_frame(&at_galaxy, fx.system_frame, &fx.placement_book(fx.galaxy))
        .expect("the galaxy places its own system");
    // DOWN: the SYSTEM subtracts, and the planet then accepts the result and does no arithmetic at all.
    let back_planet = transfer_frame(&back_system, fx.planet_frame, &fx.placement_book(fx.system))
        .expect("the star places its own planet");

    let departed = pose_m(&start).x;
    let returned = pose_m(&back_planet).x;
    let err = (returned - departed).abs();
    println!(
        "[far round trip] departed {departed:.9} m  returned {returned:.9} m  error {err:.9} m"
    );
    println!(
        "[far round trip] at the galaxy the occupant is legitimately {:.3} m out; one double step there is \
         {:.9} m",
        pose_m(&at_galaxy).x,
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
///
/// HONEST SCOPE, because the name reads bigger than the measurement is: only the VISUAL preset
/// contributes anything. `walk_scale` sets `n_planets = 0` — the walk fixture forest is ambient-only and
/// authors no orbital child at any seed — so it is swept and returns nothing. It is swept anyway so that
/// the day the walk preset grows a mover, this measurement already covers it rather than needing to be
/// remembered.
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
/// presets generate across sixteen seeds — which in practice means the VISUAL preset's, since walk-scale
/// authors no orbital child at all (see [`shipped_movers`]) — at both shipped tick rates. The test
/// RE-MEASURES on every run, so the constant cannot creep back past the bound without this going red.
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
        let book = regions.author_book(realm, tick_hz, UniverseTick(0));
        // Ask about EVERY frame the whole seed forest contains, not just the ones this shard holds — the
        // question is what it can answer, and a shard that could place something it was never told about
        // would only be caught by asking about that thing.
        vd_physics::worldgen::realm_regions_for(FRAME_UNIVERSE_SEED)
            .iter()
            .map(|r| (format!("{:?}", r.realm), book.of(r.frame).is_some()))
            .collect()
    })
}

/// The chain's PLAYER-BUILT sibling at the STAR level: a second planet beside Planet 7, in range of the
/// occupied planet's own interest. The box the SL1 filter leaves as the lawful carrier of the two-
/// subtraction descent: the area IS entitled to see it, and it can only arrive through the star's
/// reflect (subtract 20) and the planet's onward reflect (subtract 5) — two hosts, two subtractions.
const CHAIN_PLANET2: RealmId = RealmId::Planet(8);
/// ...and at the PLANET level: a second area beside Area 7, in range of the occupied area's interest.
const CHAIN_AREA2: RealmId = RealmId::Area(8);
/// Where the fixture puts the second planet, in the STAR's frame. Off Planet 7 on a DIFFERENT axis
/// (21 m of +Y against the chain's all-X offsets), so a descent that subtracted on the wrong axis — or
/// twice — cannot pass; and 21 m centre-to-centre keeps the two 10 m shells disjoint while staying
/// inside the occupied planet's interest band (dist 21 − reach 10 = 11 ≤ spin-up 12).
const CHAIN_PLANET2_FROM_STAR: DVec3 = DVec3::new(CHAIN_PLANET_FROM_STAR_M, 21.0, 0.0);
/// Where it puts the second area, in the PLANET's frame — same different-axis discipline (7 m of +Y),
/// inside the planet's 10 m shell, disjoint from Area 7's box, inside the occupied area's interest.
const CHAIN_AREA2_FROM_PLANET: DVec3 = DVec3::new(CHAIN_AREA_FROM_PLANET_M, 7.0, 0.0);
/// The second planet's boundary — a shell the size a walk-scale planet takes (the seed forest's own
/// planet is the precedent; a player-built body carries its size as per-entity data).
const CHAIN_PLANET2_SOI_R_M: f64 = 10.0;
/// The second area's box half-extent — the size the seed forest's own area takes.
const CHAIN_AREA2_HALF_M: f64 = 3.0;

/// A PLAYER-BUILT sibling region with the SAME live band derivation the planted seed neighbourhood
/// carries (`walk_demand`'s one interest formula over the body's own extent — no second band source).
fn chain_sibling_region(
    realm: RealmId,
    parent: RealmId,
    center: DVec3,
    shape: vd_core::geometry::Boundary,
    v_max: f64,
    dt: f64,
) -> vd_core::geometry::RealmRegion {
    let cfg = UniverseConfig::walk_demand(v_max, dt);
    vd_core::geometry::RealmRegion {
        realm,
        center: vd_core::geometry::ParentCentre::authored(vd_core::pose::LatticePos::from_metres(
            center,
            vd_core::pose::Tier::Fine,
        )),
        frame: vd_core::pose::frame_for_realm(realm, Some(parent))
            .expect("a planet/area sibling has a canonical frame"),
        shape,
        look: Some(shape),
        band: cfg
            .band
            .build()
            .expect("containment band edges are valid by construction"),
        aoi: cfg
            .interest
            .build(shape.finite_extent(), 0.0)
            .expect("aoi band edges are valid by construction"),
        parent: Some(parent),
        // A planted single region states no children here — no interior, no interest (the
        // seed-forest rows get theirs stamped by `to_regions` from the full forest).
        interior_band: vd_core::geometry::AoiConfig::inert(),
    }
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
    // The PLAYER-BUILT siblings, one per level (the SL1 gate rewrite's lawful boxes): the star gets a
    // second planet on its roster, the planet a second area — placed exactly as a player would place
    // a structure, beside the seed neighbourhood, never inside a second world.
    let planet2 = chain_sibling_region(
        CHAIN_PLANET2,
        SYSTEM,
        CHAIN_PLANET2_FROM_STAR,
        vd_core::geometry::Boundary::Shell {
            r: CHAIN_PLANET2_SOI_R_M,
        },
        v_max,
        dt,
    );
    let area2 = chain_sibling_region(
        CHAIN_AREA2,
        PLANET,
        CHAIN_AREA2_FROM_PLANET,
        vd_core::geometry::Boundary::Aabb {
            half: DVec3::splat(CHAIN_AREA2_HALF_M),
        },
        v_max,
        dt,
    );
    let none = BTreeMap::new();
    let no_extras: Vec<vd_core::geometry::RealmRegion> = Vec::new();
    for (node, realm, extras) in [
        (SHARD, CHAIN_AREA, no_extras.clone()),
        (CHAIN_MID, PLANET, vec![area2]),
        (CHAIN_TOP, SYSTEM, vec![planet2]),
    ] {
        vd_tests::plant_demand_neighbourhood_with_movers_and_regions(
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
            &extras,
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
// (The two-level descent constants — the 20 + 5 the two hosts used to subtract between them, and the
// nanometre slack an f64 value crossing two hosts was allowed — died with the descent itself: window
// lane Slice C2, minor 19. No level restates another level's scenery any more, so there is no
// accumulated rounding to allow for.)

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
/// The newest universe tick a WINDOW LEVEL crossing the SHARD→GATEWAY edge carries THIS tick —
/// the leaf realm STATING ITS OWN authored rows straight to the connection plane, which since the
/// Slice-C2 deletion is the only way any picture leaves a realm at all. Decoded exactly as the
/// gateway decodes it, out of the gateway's own inbox: wire truth at that host.
fn own_level_ticks_at_the_gateway_edge(topo: &mut Topology) -> Option<UniverseTick> {
    with_shard(topo, GATEWAY, |g| {
        g.world_mut()
            .resource::<vd_sim::runtime::InboundBox>()
            .0
            .iter()
            .filter_map(|m| match m {
                vd_sim::io::Inbound::Wire {
                    class: vd_sim::io::MsgClass::RealmSnapshot,
                    bytes,
                    ..
                } => match postcard::from_bytes::<vd_wire::session_flow::ShardToGateway>(bytes) {
                    Ok(vd_wire::session_flow::ShardToGateway::WindowFrame { at, .. }) => Some(at),
                    _ => None,
                },
                _ => None,
            })
            .max()
    })
}

/// Every body statement crossing the SHARD→GATEWAY edge this tick — `(stating shard, subject,
/// statement)`. A realm's LOOK has exactly one lawful author since the deletion: the realm itself.
#[allow(clippy::type_complexity)]
fn window_bodies_at_the_gateway_edge(
    topo: &mut Topology,
) -> Vec<(NodeId, RealmId, vd_wire::session_flow::BodyStmt)> {
    with_shard(topo, GATEWAY, |g| {
        g.world_mut()
            .resource::<vd_sim::runtime::InboundBox>()
            .0
            .iter()
            .filter_map(|m| match m {
                vd_sim::io::Inbound::Wire { from, bytes, .. } => {
                    match postcard::from_bytes::<vd_wire::session_flow::ShardToGateway>(bytes) {
                        Ok(vd_wire::session_flow::ShardToGateway::WindowBody {
                            subject,
                            stmt,
                            ..
                        }) => Some((*from, subject, stmt)),
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect()
    })
}

/// EVERY TOMBSTONED SCENERY FRAME delivered into `node` this tick, by arm name — the measurement
/// that says the four old inter-realm picture lanes are dead rather than merely quiet. A frame of
/// any of them still DECODES (their discriminants are reserved forever), so this cannot pass
/// vacuously through a decode failure.
fn dead_scenery_into(topo: &mut Topology, node: NodeId) -> Vec<&'static str> {
    with_shard(topo, node, |s| {
        s.world_mut()
            .resource::<vd_sim::runtime::InboundBox>()
            .0
            .iter()
            .filter_map(|m| match m {
                vd_sim::io::Inbound::Wire { bytes, .. } => {
                    match postcard::from_bytes::<InterShardFlow>(bytes) {
                        Ok(InterShardFlow::RealmCascade(_)) => Some("RealmCascade"),
                        Ok(InterShardFlow::RealmObservation(_)) => Some("RealmObservation"),
                        Ok(InterShardFlow::RealmShapeObservation(_)) => {
                            Some("RealmShapeObservation")
                        }
                        Ok(InterShardFlow::ChildSceneSet(_)) => Some("ChildSceneSet"),
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect()
    })
}

/// The instant of the newest LEVEL inside the sealed statements `parent` currently holds for
/// `child` — the Q2 relay's cargo, opened HERE (a test may look; the parent structurally never
/// does: `vd-sim` deliberately never calls `open_relay_statements`).
fn relay_level_stamp(topo: &mut Topology, parent: NodeId, child: RealmId) -> Option<UniverseTick> {
    with_shard(topo, parent, |s| {
        let held = s.world_mut().resource::<vd_sim::stub::RelayHeld>();
        let bytes = held.statements_for(child)?;
        vd_wire::session_flow::open_relay_statements(&bytes)
            .ok()?
            .into_iter()
            .filter_map(|st| match st {
                vd_wire::session_flow::RelayedStatement::Level { at, .. } => Some(at),
                vd_wire::session_flow::RelayedStatement::Body { .. } => None,
            })
            .max()
    })
}

/// What `node` AUTHORED for its own children at `tick`, in its own frame — the INPUT to any
/// statement it makes, read through the production expression at the exact instant the rows under
/// test are stamped at. Sampling it at any other instant would measure the world's orbit rather
/// than the statement.
fn authored_by(
    topo: &mut Topology,
    node: NodeId,
    tick: UniverseTick,
) -> BTreeMap<RealmId, vd_core::pose::StampedPose> {
    with_shard(topo, node, |s| {
        let cfg = s.world_mut().resource::<vd_sim::stub::StubConfig>();
        let (realm, tick_hz) = (cfg.realm, 1.0 / cfg.tick_dt_s);
        let regions = s.world_mut().resource::<vd_sim::stub::RealmRegions>();
        regions
            .authored_realm_snaps(realm, &regions.author_book(realm, tick_hz, tick))
            .into_iter()
            .map(|r| (r.realm, r.pose))
            .collect()
    })
}

/// THE ACCEPTANCE STORY THE DELETION REPLACES IT WITH, over the same three hosts.
///
/// The star used to author a turning station, subtract where it put its planet, and hand the planet
/// a station measured from the PLANET's centre; the planet subtracted again and handed the area a
/// station measured from the AREA's. Three processes, two subtractions, three moments of time mixed
/// into one picture — and a level that had to OPEN and RE-STATE another level's scenery to pass it
/// on.
///
/// None of that happens any more (window lane Slice C2, minor 19). What this gate measures now is
/// the thing that replaced it, and it measures both halves:
///
/// * THE ABSENCE, on the wire: across a long live window, not one frame of any of the four
///   tombstoned scenery lanes is delivered into ANY of the three shards. Their discriminants are
///   reserved forever, so such a frame would still DECODE — this cannot pass by a decode failure.
/// * THE PRESENCE, on the living lanes: the leaf realm states its OWN authored rows straight to the
///   connection plane, once per tick, at its own stamp; and its self-authored statements climb one
///   hop up SEALED, for the parent to forward without ever reading them.
#[test]
fn the_authored_world_no_longer_descends_and_each_level_states_only_itself() {
    let fabric = FaultFabric::new(4242, 2);
    let movers = BTreeMap::from([(CHAIN_STATION, station_orbit())]);
    let (mut topo, subject) = boot_the_chain_with(&fabric, &movers);
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);

    // Run the real chain until BOTH living lanes are up: the leaf's own level is crossing the
    // gateway edge, and its sealed statements have climbed to its parent. Every tick, sweep all
    // three hosts' inboxes for a frame of any dead lane.
    let mut dead_seen: BTreeSet<&'static str> = BTreeSet::new();
    let mut edge_stamps: BTreeSet<u64> = BTreeSet::new();
    let running = step_until(&mut topo, 900, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        for node in [CHAIN_TOP, CHAIN_MID, SHARD] {
            dead_seen.extend(dead_scenery_into(t, node));
        }
        if let Some(seen) = own_level_ticks_at_the_gateway_edge(t) {
            edge_stamps.insert(seen.0);
        }
        // Two DISTINCT stamps at the edge is the minimum that shows a live feed rather than one
        // repeated value, and the relay must have reached the level above.
        (edge_stamps.len() >= 2) & relay_level_stamp(t, CHAIN_MID, CHAIN_AREA).is_some()
    });
    assert!(
        running,
        "the leaf states its own level to the connection plane and its sealed statements climb one \
         hop: {} distinct edge stamp(s)",
        edge_stamps.len(),
    );

    // ---- THE ABSENCE. Four lanes, three hosts, the whole warm-up window.
    assert_eq!(
        dead_seen,
        BTreeSet::new(),
        "a TOMBSTONED scenery lane is still speaking between realms: {dead_seen:?}. Their \
         discriminants decode forever, so this is a measurement of silence, not of garbage.",
    );

    // ---- THE PRESENCE, half 1: the leaf's own rows, at the leaf's own stamp, against what the
    // leaf itself authored at that instant. Nobody subtracted anything: the value is the authored
    // one, verbatim.
    let level = with_shard(&mut topo, GATEWAY, |g| {
        g.world_mut()
            .resource::<vd_sim::runtime::InboundBox>()
            .0
            .iter()
            .find_map(|m| match m {
                vd_sim::io::Inbound::Wire { bytes, .. } => {
                    match postcard::from_bytes::<vd_wire::session_flow::ShardToGateway>(bytes) {
                        Ok(vd_wire::session_flow::ShardToGateway::WindowFrame {
                            at, rows, ..
                        }) => Some((at, rows)),
                        _ => None,
                    }
                }
                _ => None,
            })
    });
    if let Some((at, rows)) = level {
        let authored = authored_by(&mut topo, SHARD, at);
        for r in &rows {
            assert_eq!(
                pose_m(&r.pose),
                pose_m(&authored[&r.realm]),
                "the leaf states its own authored row VERBATIM — no level subtracts for anyone now",
            );
        }
        println!(
            "[no descent] the leaf stated {} own row(s) at {at:?}",
            rows.len()
        );
    }

    // ---- THE PRESENCE, half 2: the Q2 relay climbs, SEALED, one hop per level. The parent holds
    // the child's own words; it never restates them, and it structurally cannot (this test opens
    // the seal; `vd-sim` never calls the opener at all).
    let relayed_to_planet = relay_level_stamp(&mut topo, CHAIN_MID, CHAIN_AREA)
        .expect("the area's seal reached the planet");
    let area_now = universe_now(&mut topo, SHARD);
    assert!(
        relayed_to_planet <= area_now,
        "the relayed level speaks at or before the child's own now: {relayed_to_planet:?} vs {area_now:?}",
    );
    println!(
        "[no descent] the area's SEALED statements sit at the planet, stamped {relayed_to_planet:?} \
         (the area's own clock reads {area_now:?})",
    );

    // ---- AND THE ABSENCE HOLDS OVER A SETTLED WINDOW, not just a warm-up: a lane that woke up
    // only once the chain settled would be invisible above.
    const WINDOW_TICKS: u64 = 100;
    let mut late_dead: BTreeSet<&'static str> = BTreeSet::new();
    for _ in 0..WINDOW_TICKS {
        topo.step();
        set_shard_subject_pose_now(&mut topo, SHARD, subject, CHAIN_AREA_FRAME, at);
        for node in [CHAIN_TOP, CHAIN_MID, SHARD] {
            late_dead.extend(dead_scenery_into(&mut topo, node));
        }
    }
    assert_eq!(
        late_dead,
        BTreeSet::new(),
        "a dead scenery lane woke up on a settled chain: {late_dead:?}",
    );
    println!("[no descent] {WINDOW_TICKS} settled ticks, zero inter-realm scenery frames anywhere");
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
    // THE RATE IS THE ASSERTION, not a print (finding 41 — `assert!(beats > 0)` is why the drift from
    // the contract's stated cadence was never caught): the bit beats at 1/cadence, measured against
    // the SENDER's own production expression (the planet ships the counted bit), within one beat for
    // the window's edges plus one for a possible occupancy-edge beat outside the cadence.
    let cadence = vd_sim::stub::aoi_recheck_cadence(&vd_tests::planet_stub_config());
    let expected = WINDOW_TICKS / cadence;
    assert!(
        (beats >= expected.saturating_sub(1)) && (beats <= expected + 1),
        "the star heard {beats} beats in {WINDOW_TICKS} ticks; the contract says 1 per {cadence}-tick \
         cadence (~{expected}), ± one beat (window edge / occupancy edge)",
    );

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

// (The three per-level placement constants that anchored this block's centre arithmetic died with
// the field — Slice C1, window_lane.md §2.4: a shape carries no position, so there is no
// subtraction left to assert against them. The chain's placements still live where they always
// did: in the seed forest the fixture boots.)

/// THE ACCEPTANCE STORY FOR THE BOXES, over the same three hosts — inverted by the deletion.
///
/// A box the STAR authored and a box the PLANET authored used to arrive at the AREA shard, restated
/// once per level into the AREA's space. The lane that carried them needed a runtime filter to stop
/// it handing the area its OWN placement, sign-flipped: the SL1 SELF-PLACEMENT FILTER.
///
/// The lane is deleted and the filter with it, by the owner's Q3 amendment (2026-08-16,
/// `docs/design/window_lane.md` §5 RULINGS) — retirement BY AMENDMENT, not erosion, because the
/// hazard itself ceases to exist. This gate is the END-TO-END half of that claim (the compile-level
/// half is `crates/wire/tests/intershard_closed.rs`
/// `no_living_inter_shard_payload_carries_a_realm_placement_or_a_centre`): across a long live
/// window on a real three-level chain, NOTHING addressed INTO a realm names that realm, and nothing
/// addressed into a realm carries any realm's placement at all.
///
/// It can fail: the sweep reads every byte delivered into each host and decodes it as the shard
/// dispatch does, and the tombstoned arms still decode.
#[test]
fn no_message_into_a_realm_names_that_realm_or_carries_a_placement() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, subject) = boot_the_chain(&fabric);
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);

    // The three hosts and the realm each one IS — "a message never tells its receiver about itself".
    let hosts = [
        (CHAIN_TOP, SYSTEM),
        (CHAIN_MID, PLANET),
        (SHARD, CHAIN_AREA),
    ];

    // THE CLASSIFIER, named so the sweep below and the POSITIVE CONTROL beneath it run the SAME
    // rule. Returns the offences one delivered frame commits against its receiver, and records
    // which LIVING picture lane it was (so a silent sweep cannot pass as a clean one).
    fn offences_of(
        node: NodeId,
        own_realm: RealmId,
        flow: &InterShardFlow,
        kinds: &mut BTreeSet<&'static str>,
    ) -> Vec<String> {
        match flow {
            // ★ THE FOUR DEAD SCENERY LANES. Each one either carried a realm's placement or
            // another realm's look; each one is now producer-less.
            InterShardFlow::RealmCascade(_) => vec![format!("{node:?} received a RealmCascade")],
            InterShardFlow::RealmObservation(_) => {
                vec![format!("{node:?} received a RealmObservation")]
            }
            InterShardFlow::RealmShapeObservation(_) => {
                vec![format!("{node:?} received a RealmShapeObservation")]
            }
            InterShardFlow::ChildSceneSet(cs) => vec![format!(
                "{node:?} received a ChildSceneSet naming {:?}",
                cs.realms.iter().map(|r| r.realm).collect::<Vec<_>>()
            )],
            // ★ THE ONE LIVING PICTURE FRAME a realm receives: its child's SEALED statements. It
            // is about the CHILD, never about the receiver, and the parent never opens it. Opened
            // HERE only to prove that: no statement inside names the receiving realm.
            InterShardFlow::WindowRelay(wr) => {
                kinds.insert("WindowRelay");
                // Slice 3 (look horizon): the sealed INTERIOR batches ride the same relay —
                // opened here too, because a realm told about itself inside a grandchild's
                // forwarded batch would be the identical offence one seal deeper.
                let named: Vec<RealmId> = std::iter::once(&wr.own)
                    .chain(wr.interior.iter().map(|e| &e.own))
                    .flat_map(|sealed| {
                        vd_wire::session_flow::open_relay_statements(sealed)
                            .expect("a sealed batch decodes")
                    })
                    .flat_map(|st| match st {
                        vd_wire::session_flow::RelayedStatement::Level { rows, .. } => {
                            rows.into_iter().map(|r| r.realm).collect::<Vec<_>>()
                        }
                        vd_wire::session_flow::RelayedStatement::Body { subject, .. } => {
                            vec![subject]
                        }
                    })
                    .collect();
                if named.contains(&own_realm) {
                    vec![format!(
                        "{node:?} was told about ITSELF ({own_realm:?}) inside a relay: {named:?}"
                    )]
                } else {
                    Vec::new()
                }
            }
            // The SL7 occupancy bit: one child coord, a fence and a tick. No geometry.
            InterShardFlow::ChildLive(_) => {
                kinds.insert("ChildLive");
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// A lineage coord for the chain's PLANET — the routing key a relay from the planet carries.
    fn planet_coord() -> vd_core::realm_coord::RealmCoord {
        vd_sim::stub::StubConfig::root_coord(SYSTEM)
            .child(vd_core::worldgen::level_of(PLANET).expect("a seed-lineage planet"))
    }

    // ★ THE POSITIVE CONTROL, before the sweep: a relay whose statements DO name the receiver is
    // caught. Without this, a sweep that found nothing would be indistinguishable from a rule that
    // can no longer find anything.
    {
        let mut kinds = BTreeSet::new();
        let planted = InterShardFlow::WindowRelay(vd_wire::intershard::WindowRelay {
            child: planet_coord(),
            realm_fence: vd_core::Fence(1),
            own: vd_wire::session_flow::seal_relay_statements(&[
                vd_wire::session_flow::RelayedStatement::Body {
                    subject: CHAIN_AREA, // the RECEIVER's own realm — the forbidden sentence
                    stmt: vd_wire::session_flow::BodyStmt::SelfLook { bag: vec![1, 2] },
                    authored_at: UniverseTick(1),
                },
            ]),
            interior: Vec::new(),
        });
        assert_eq!(
            offences_of(SHARD, CHAIN_AREA, &planted, &mut kinds).len(),
            1,
            "the detector must catch a realm being told about itself — otherwise the sweep below \
             proves nothing"
        );
        // And the SAME offence one seal deeper (slice 3): a grandchild batch naming the
        // receiver must be caught too, or the interior lane escapes the sweep.
        let planted_interior = InterShardFlow::WindowRelay(vd_wire::intershard::WindowRelay {
            child: planet_coord(),
            realm_fence: vd_core::Fence(1),
            own: vd_wire::session_flow::seal_relay_statements(&[]),
            interior: vec![vd_wire::intershard::InteriorRelay {
                child: PLANET,
                child_fence: vd_core::Fence(1),
                own: vd_wire::session_flow::seal_relay_statements(&[
                    vd_wire::session_flow::RelayedStatement::Body {
                        subject: CHAIN_AREA, // the receiver again, one hop deeper
                        stmt: vd_wire::session_flow::BodyStmt::SelfLook { bag: vec![3, 4] },
                        authored_at: UniverseTick(1),
                    },
                ]),
            }],
        });
        assert_eq!(
            offences_of(SHARD, CHAIN_AREA, &planted_interior, &mut kinds).len(),
            1,
            "the detector must catch the identical offence one seal deeper (the slice-3 \
             interior forward), or that lane escapes the sweep"
        );
        let clean = InterShardFlow::ChildLive(vd_wire::intershard::ChildLive {
            child: planet_coord(),
            fence: vd_core::Fence(1),
            at: UniverseTick(1),
        });
        assert!(
            offences_of(SHARD, CHAIN_AREA, &clean, &mut kinds).is_empty(),
            "and it must not fire on the occupancy bit — the sweep would then never be silent"
        );
    }

    // Warm the chain to steady state first: the bit has climbed both legs and the leaf is stating
    // its own level to the connection plane, so the sweep below runs over a LIVE cluster.
    let running = step_until(&mut topo, 900, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        child_bit(t, CHAIN_TOP, PLANET).is_some() & own_level_ticks_at_the_gateway_edge(t).is_some()
    });
    assert!(running, "the chain is live before the sweep runs");

    const SWEEP_TICKS: u64 = 150;
    let mut inbound_kinds: BTreeSet<&'static str> = BTreeSet::new();
    let mut offences: Vec<String> = Vec::new();
    let mut frames_seen = 0u64;
    for _ in 0..SWEEP_TICKS {
        topo.step();
        set_shard_subject_pose_now(&mut topo, SHARD, subject, CHAIN_AREA_FRAME, at);
        for (node, own_realm) in hosts {
            let inbound: Vec<Vec<u8>> = with_shard(&mut topo, node, |s| {
                s.world_mut()
                    .resource::<vd_sim::runtime::InboundBox>()
                    .0
                    .iter()
                    .filter_map(|m| match m {
                        vd_sim::io::Inbound::Wire { bytes, .. } => Some(bytes.to_vec()),
                        _ => None,
                    })
                    .collect()
            });
            for bytes in inbound {
                let Ok(flow) = postcard::from_bytes::<InterShardFlow>(&bytes) else {
                    continue; // not an inter-shard frame (the gateway/client lanes ride other types)
                };
                frames_seen += 1;
                offences.extend(offences_of(node, own_realm, &flow, &mut inbound_kinds));
            }
        }
    }
    assert!(frames_seen > 0, "the sweep read real traffic (non-vacuous)");
    assert!(
        inbound_kinds.contains("ChildLive") & inbound_kinds.contains("WindowRelay"),
        "the sweep saw BOTH living inter-realm lanes, so its silence about the dead ones means \
         something: saw {inbound_kinds:?}",
    );
    assert_eq!(
        offences,
        Vec::<String>::new(),
        "a realm was told about another realm's look, or about itself. The SL1 self-placement \
         filter was retired (owner Q3, 2026-08-16) BECAUSE this could no longer happen.",
    );
    println!(
        "[no self-placement] {SWEEP_TICKS} ticks, 3 hosts, {frames_seen} inter-shard frames read; \
         inbound picture lanes seen: {inbound_kinds:?}; zero offences",
    );

    // The player's pose never converted at its own shard — the half of the old assert that
    // survives everything: the shard holds the rider in the room's own frame, verbatim.
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
}

/// THE ROOM A PLAYER IS STANDING IN HAS EXACTLY ONE AUTHOR, AND IT IS THE ROOM.
///
/// The box a player is standing inside was once authored by the router, out of its own copy of the
/// seed forest; then by the room's own shard AND by its parent, which reflected the same outline
/// down (they agreed, so reading the value could not tell them apart — only the TIMING could).
///
/// Since the deletion the parent has no way to say it at all. This gate asserts the single author
/// directly: the room's look crosses the shard→gateway edge stated BY THE ROOM'S OWN SHARD, about
/// ITSELF, and no other host ever states a look about that realm.
#[test]
fn the_room_the_player_is_standing_in_has_exactly_one_author() {
    let fabric = FaultFabric::new(4242, 2);
    let (mut topo, subject) = boot_the_chain(&fabric);
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);

    let mut room_look_from: BTreeSet<NodeId> = BTreeSet::new();
    let mut foreign_look: Vec<String> = Vec::new();
    let mut looks_seen = 0u64;
    let arrived = step_until(&mut topo, 900, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        for (from, subject_realm, stmt) in window_bodies_at_the_gateway_edge(t) {
            looks_seen += 1;
            let is_look = matches!(stmt, vd_wire::session_flow::BodyStmt::SelfLook { .. });
            if is_look & (subject_realm == CHAIN_AREA) {
                room_look_from.insert(from);
            }
            // A LOOK about a realm the sender is not: structurally unrepresentable on this lane
            // (the gateway's own attestation refuses it), asserted here on the wire as well.
            if is_look & (subject_realm != CHAIN_AREA) {
                foreign_look.push(format!("{from:?} stated a look about {subject_realm:?}"));
            }
        }
        !room_look_from.is_empty()
    });
    assert!(
        arrived,
        "the room's own look reached the connection plane: {looks_seen} body statement(s) seen",
    );
    assert_eq!(
        room_look_from.into_iter().collect::<Vec<_>>(),
        vec![SHARD],
        "the room is described by its OWN shard and by nobody else (SL3 — a realm draws itself)",
    );
    assert_eq!(
        foreign_look,
        Vec::<String>::new(),
        "a shard stated a look about a realm it is not"
    );
}

// ─────────────────────────── SLICE 5 — THE CROSS-REALM ENTITY FEED ───────────────────────────

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
/// realm's occupants and NOBODY else's figure. The other player's whereabouts surface as A REALM —
/// the occupied area, kept alive by its own occupancy and announced by its one liveness bit; its
/// BOX draws only within the parent's visibility band (Slice C1 closed the beyond-visibility
/// interiors fan). SL2 at steady state: an occupant's pose exists on the shard that owns them and
/// on that shard's own clients, full stop.
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

    // THE PERMANENT TRIPWIRE's BASELINE (Step 5 slice F), taken after boot: the fixture logs both
    // avatars in BEFORE it plants the region forests, and in that pre-plant window a shard briefly
    // cannot NAME its own realm's local frame (production plants regions at spawn, before any
    // session — the window is fixture-only). From here — through a REAL crossing with a retained
    // ghost — the counter may never grow: the §4u corruption's last writer is deleted.
    let foreign_baseline: Vec<u64> = [SHARD, CHAIN_MID, CHAIN_TOP]
        .into_iter()
        .map(|node| {
            with_shard(&mut topo, node, |s| {
                s.world_mut()
                    .resource::<vd_sim::stub::StubStats>()
                    .entity_rows_foreign_labelled
            })
        })
        .collect();

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
    assert_eq!(pose_m(&stayer_here), staying);
    let traveller_there =
        edge_row(&planet_feed, traveller).expect("the planet draws its own player");
    assert_eq!(traveller_there.frame, planet_frame);
    assert_eq!(pose_m(&traveller_there), held);

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

    // (3) THE PROXY THAT REPLACES THE FIGURE (SL7), as the flag day left it (Slice C1). LIVENESS
    // is the occupied child's own doing: the area holds an occupant, so it keeps ITSELF alive and
    // its ONE BIT beats at its parent — the only thing about the stayer that crosses a boundary.
    assert!(
        child_bit(&mut topo, CHAIN_MID, CHAIN_AREA).is_some(),
        "the occupied area's bit beats at the planet",
    );
    assert!(
        with_shard(&mut topo, SHARD, |s| {
            s.world_mut()
                .resource::<vd_sim::stub::RealmAuthority>()
                .0
                .is_some()
        }),
        "the occupied area stays alive on its own occupancy (SL7 liveness)",
    );
    // VISIBILITY is the parent's per-observer band, and the traveller stands 11 m from an area
    // whose band tears down at 5.4 m — so the area's box is legitimately NOT in the traveller's
    // drawn scene. Before the flag day it was: the up-shape lane (now a tombstone) let interiors
    // fan to observers the band had never admitted. A child's look now rides the sibling-interior
    // relay to the GATEWAY composer and draws only inside membership ∪ chain (window_lane.md §2.2,
    // §5 Q2). Pinned by EQUALITY (HR5): the planet's one dot draws exactly its own realm's
    // outline — an area id here would be a beyond-visibility interior leaking again.
    let in_band: BTreeSet<RealmId> = with_shard(&mut topo, CHAIN_MID, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::AoiMembership>()
            .0
            .iter()
            .filter(|((obs, _), st)| matches!(obs, vd_sim::stub::ObserverId::Dot(_)) & st.in_band())
            .filter_map(|((_, path), _)| path.realm_id())
            .collect()
    });
    assert!(
        !in_band.contains(&CHAIN_AREA),
        "the planet's band does not admit the occupied area at 11 m against a 5.4 m tear-down: \
         its verdict is what gates every body the gateway draws, so an area id here would be a \
         beyond-visibility interior leaking again. In band: {in_band:?}",
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

    // (5b) THE PERMANENT TRIPWIRE (Step 5 slice F): since the post-boot baseline, no shard on this
    // chain emitted a row whose pose label it could not restate into its own frame — through a REAL
    // crossing whose retained ghost filled the hand-off window. The §4u corruption's last writer
    // (the ghost pose feed) is deleted; growth here is the deleted class resurfacing.
    for (node, base) in [SHARD, CHAIN_MID, CHAIN_TOP]
        .into_iter()
        .zip(&foreign_baseline)
    {
        let foreign = with_shard(&mut topo, node, |s| {
            s.world_mut()
                .resource::<vd_sim::stub::StubStats>()
                .entity_rows_foreign_labelled
        });
        assert_eq!(
            foreign, *base,
            "{node:?}: a foreign-labelled pose entered a locally-emitted row AFTER boot — the \
             deleted corruption class resurfaced",
        );
    }

    // (6) THE REMOVE MESSAGE, end to end (proto_minor 14, D-4(a)): the leaver's `EntityRemoved` is
    // fanned at HOLD CLOSURE (slice F retimed it there from the band-exit despawn — the SpawnV2
    // take-over proof closes the Source hold; the later band-exit emit is an idempotent re-send),
    // the gateway fans it as the reliable Event, and the BYSTANDER's real client EVICTS the
    // leaver's track — the figure VANISHES instead of freezing at the boundary forever. The whole
    // lane, through the production shard, gateway and client, measured on the delivered view.
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

/// Hops between the shard a player is standing on and the mesh lane's deepest surviving consumer,
/// the GATEWAY's parity intake (Slice C1 — the gateway→client fan of this lane is retired; the
/// screen's own latency now belongs to the composed feed and its process-tier gates). One ordinary
/// fabric delivery, costing the same one tick as a shard-to-shard leg; naming it separately keeps
/// the edge budget a DERIVATION of the topology instead of a number fitted to the measurement.
const GATEWAY_EDGE_HOPS: u64 = 1;

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
/// * DOWN — a turning station authored by the STAR, descending to the planet, to the area, and out to
///   the GATEWAY EDGE. Since the flag day (Slice C1) the deepest consumer of this mesh lane is the
///   gateway's parity comparator (the old client fan is retired; a screen sees only the COMPOSED
///   feed, whose latency the process-tier window gates measure) — so the deepest depth here reads
///   [`station_ticks_at_the_gateway_edge`], the identical bytes the retired fan used to forward.
///
/// THE BUDGET IS DERIVED, NOT FITTED. `Topology::step` pumps deliveries at tick start and then steps
/// nodes, so a message sent during tick N is delivered no earlier than N+1; and the shard schedule folds
/// an arriving relay in the SAME tick it lands (`process_inbound` opens group A, `evaluate_realm_aoi`
/// closes the tick after `emit_realm_frames`), so a re-relay leaves on the tick it arrived. One tick per
/// hop, therefore, and the budget is the hop count: [`CHAIN_LEVELS_ABOVE_LEAF`] for the up leg, plus
/// [`GATEWAY_EDGE_HOPS`] for the leg that reaches the gateway edge. If a level ever starts holding a relay for a
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
            && own_level_ticks_at_the_gateway_edge(t).is_some()
            && relay_level_stamp(t, CHAIN_MID, CHAIN_AREA).is_some()
    });
    assert!(
        running,
        "every leg of the chain is live before its price is quoted: the occupancy bit has climbed to \
         the star, the leaf's own level is crossing the gateway edge, and its sealed statements have \
         reached the level above",
    );

    // THE MEASUREMENT. One pass, every series sampled on the same ticks, so the up and down figures
    // describe one run of one cluster rather than two runs that might have settled differently.
    //
    // Every age is `what the AUTHOR of this lane holds right now − what this consumer holds right now`,
    // both read at the same instant, straight after the step. For the up leg the author is the area shard
    // holding the player's own dot; for the down leg it is the star, whose clock IS the instant its rows
    // this tick are authored at.
    let mut up: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut relay_hop: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut edge_ages: Vec<u64> = Vec::new();
    // The gateway edge's held newest: staleness at a consumer is how old what it is HOLDING is, so
    // a tick that received nothing still contributes the age of what it has.
    let mut held_edge: Option<UniverseTick> = None;
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
        // ONE universe clock: all three shards step in exact lockstep, so a single sample before
        // the step is the instant every statement shipped this step is stamped at.
        let authored = universe_now(&mut topo, SHARD);
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

        // THE RELAY, per level: how far behind now the newest LEVEL inside the sealed statements
        // each parent holds for its child is. This is the ONLY picture leg between realms since
        // the deletion, and it is exactly ONE HOP at every depth — a child's own words, held by
        // its parent, never re-shipped further. Its rate is send-on-change plus the AoI-cadence
        // re-assert, so a chain whose interiors are static beats at the cadence, by design.
        for (parent, child, depth) in [(CHAIN_MID, CHAIN_AREA, 1u64), (CHAIN_TOP, PLANET, 2)] {
            if let Some(stamp) = relay_level_stamp(&mut topo, parent, child) {
                relay_hop
                    .entry(depth)
                    .or_default()
                    .push(authored.0.saturating_sub(stamp.0));
            }
        }

        // THE PICTURE, at the gateway edge: the newest instant the LEAF's own level carries across
        // the shard→gateway hop — one hop, per tick, whatever the chain's depth. HELD across ticks
        // like the legs above.
        if let Some(seen) = own_level_ticks_at_the_gateway_edge(&mut topo) {
            let slot = held_edge.get_or_insert(seen);
            *slot = (*slot).max(seen);
        }
        if let Some(held) = held_edge {
            edge_ages.push(authored.0.saturating_sub(held.0));
        }
    }

    let up1 = ages(up.get(&1).map_or(&[][..], Vec::as_slice));
    let up2 = ages(up.get(&2).map_or(&[][..], Vec::as_slice));
    let rl1 = ages(relay_hop.get(&1).map_or(&[][..], Vec::as_slice));
    let rl2 = ages(relay_hop.get(&2).map_or(&[][..], Vec::as_slice));
    let dnc = ages(&edge_ages);
    let hz = 1.0 / vd_tests::area_stub_config().tick_dt_s;
    println!(
        "[chain latency] {CHAIN_LATENCY_TICKS} ticks at {hz:.0} Hz, 3-level chain\n\
         [chain latency]   UP   depth 1 (planet)  n={:3}  p50 {} tick ({:.0} ms)  p99 {} ({:.0} ms)  max {}\n\
         [chain latency]   UP   depth 2 (star)    n={:3}  p50 {} tick ({:.0} ms)  p99 {} ({:.0} ms)  max {}\n\
         [chain latency]   RELAY depth 1 (area→planet) n={:3}  p50 {} tick ({:.0} ms)  p99 {} ({:.0} ms)  max {}\n\
         [chain latency]   RELAY depth 2 (planet→star) n={:3}  p50 {} tick ({:.0} ms)  p99 {} ({:.0} ms)  max {}\n\
         [chain latency]   OWN LEVEL at the GW EDGE    n={:3}  p50 {} tick ({:.0} ms)  p99 {} ({:.0} ms)  max {}",
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
        rl1.n,
        rl1.p50_ticks,
        ms(rl1.p50_ticks),
        rl1.p99_ticks,
        ms(rl1.p99_ticks),
        rl1.worst_ticks,
        rl2.n,
        rl2.p50_ticks,
        ms(rl2.p50_ticks),
        rl2.p99_ticks,
        ms(rl2.p99_ticks),
        rl2.worst_ticks,
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
        ("relay depth 1", &rl1),
        ("relay depth 2", &rl2),
        ("own level at the gateway edge", &dnc),
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
    // THE UP BUDGET IS ONE TICK PER HOP PLUS THE BIT'S OWN CADENCE (finding 41): the bit beats on
    // its SENDER's AoI cadence — the contract's stated rate, ~25× less traffic than per-tick — so
    // between beats the held stamp ages by up to one full cadence, and that ageing is the
    // contract's own price, never a level sitting on a beat. Depth never widens it further: the bit
    // is re-originated per level (each sample above is a one-hop age by construction). The budget
    // is DERIVED from each sender's production expression, so a cadence change moves the gate and
    // the code together. The END-TO-END recursion is bounded elsewhere: each level's bit EXISTS
    // only while its child's bit is fresh within the one TTL (itself sized in cadences), and the
    // climb scenario gates the chain forming at all.
    for (depth, sender_cadence, a) in [
        (
            1u64,
            vd_sim::stub::aoi_recheck_cadence(&vd_tests::area_stub_config()),
            &up1,
        ),
        (
            2,
            vd_sim::stub::aoi_recheck_cadence(&vd_tests::planet_stub_config()),
            &up2,
        ),
    ] {
        assert!(
            a.p99_ticks <= sender_cadence,
            "the up leg ages past its own beat: at depth {depth} the p99 age is \
             {} ticks ({:.0} ms) against one hop + the sender's {sender_cadence}-tick cadence \
             ({:.0} ms). A level is holding a beat instead of retaining it on the tick it arrives.",
            a.p99_ticks,
            ms(a.p99_ticks),
            ms(sender_cadence),
        );
    }
    // THE RELAY BUDGET IS ONE HOP PLUS THE STATEMENT'S OWN CADENCE, AT EVERY DEPTH — and that
    // depth-independence is the deletion's whole structural win. The old down-cascade cost one
    // serialized hop PER LEVEL, so a deeper chain was a slower picture; the relay is a child's own
    // words held by its ONE parent and never re-shipped further, so depth 2 is priced exactly like
    // depth 1. The rate is send-on-change plus the AoI-cadence re-assert (a static interior states
    // nothing between beats), so the derived budget is one cadence + one tick of hop, read from
    // the SAME production expression the emitter uses.
    let relay_budget = vd_sim::stub::aoi_recheck_cadence(&vd_tests::area_stub_config()) + 1;
    for (depth, a) in [(1u64, &rl1), (2, &rl2)] {
        assert!(
            a.p99_ticks <= relay_budget,
            "the relay leg costs more than one hop plus its own cadence: at depth {depth} the p99 \
             age is {} ticks ({:.0} ms) against a derived budget of {relay_budget} ({:.0} ms). \
             Depth must NOT widen this — the relay is one hop at every level.",
            a.p99_ticks,
            ms(a.p99_ticks),
            ms(relay_budget),
        );
    }
    // THE PICTURE'S OWN BUDGET is now ONE hop, whatever the chain's depth: a realm states its
    // level straight to the connection plane. `CHAIN_LEVELS_ABOVE_LEAF` no longer appears in it,
    // and that absence IS the §2.13 claim — picture latency is max(one hop), not a sum over depth.
    assert!(
        dnc.p99_ticks <= GATEWAY_EDGE_HOPS,
        "the leaf's own level reaches the gateway edge later than one hop allows: p99 {} ticks \
         ({:.0} ms) against a derived budget of {GATEWAY_EDGE_HOPS} ({:.0} ms). Depth is not in \
         this budget and must never enter it.",
        dnc.p99_ticks,
        ms(dnc.p99_ticks),
        ms(GATEWAY_EDGE_HOPS),
    );
}

/// The per-attempt drop the lossy half induces on EVERY leg of the down chain.
///
/// A HALF is chosen deliberately: at a tenth the compounding is inside the noise of a 200-tick window, and
/// at a ninth-tenths nothing arrives anywhere and every depth reads the same. A half is the value at which
/// the difference BETWEEN depths is the largest thing in the measurement, which is the quantity under test.
const CHAIN_DOWN_DROP_P: f64 = 0.5;

/// THE ANTI-COMPOUNDING GATE — what a lossy link costs the picture now that it visits ONE hop.
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
fn the_picture_does_not_compound_staleness_per_level_under_a_lossy_link() {
    let fabric = FaultFabric::new(4242, 2);
    let movers = BTreeMap::from([(CHAIN_STATION, station_orbit())]);
    let (mut topo, subject) = boot_the_chain_with(&fabric, &movers);
    let at = DVec3::new(CHAIN_OCCUPANT_FROM_AREA_M, 0.0, 0.0);

    let running = step_until(&mut topo, 600, |t| {
        set_shard_subject_pose_now(t, SHARD, subject, CHAIN_AREA_FRAME, at);
        own_level_ticks_at_the_gateway_edge(t).is_some()
            & relay_level_stamp(t, CHAIN_MID, CHAIN_AREA).is_some()
    });
    assert!(
        running,
        "the picture reaches the gateway edge, and the relay its parent, before either is made lossy"
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
            let authored = universe_now(topo, SHARD);
            topo.step();
            for (i, (parent, child)) in [(CHAIN_MID, CHAIN_AREA), (CHAIN_TOP, PLANET)]
                .into_iter()
                .enumerate()
            {
                if let Some(stamp) = relay_level_stamp(topo, parent, child) {
                    let slot = held[i].get_or_insert(stamp);
                    *slot = (*slot).max(stamp);
                }
            }
            if let Some(seen) = own_level_ticks_at_the_gateway_edge(topo) {
                let slot = held[2].get_or_insert(seen);
                *slot = (*slot).max(seen);
            }
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

    let names = [
        "relay depth 1 (area→planet) ",
        "relay depth 2 (planet→star) ",
        "own level at the GATEWAY EDGE",
    ];
    println!(
        "[chain loss] {CHAIN_LATENCY_TICKS} ticks per arm, {CHAIN_DOWN_DROP_P} attempt-drop on every \
         leg of the chain"
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
    // THE PROPERTIES, not the numbers. (1) the induced loss REACHED the lane — without this the
    // whole lossy arm could be measuring a healthy link and reporting it as a loss result.
    assert!(
        lossy[2].0 < clean[2].0,
        "the induced loss reached the lane: the screen took news on {:.2} of ticks lossy vs {:.2} clean",
        lossy[2].0,
        clean[2].0,
    );
    // (2) STALENESS NO LONGER COMPOUNDS WITH DEPTH, and that inversion is the deletion's whole
    // point. When the picture descended hop by hop, each level was strictly further behind than
    // the one above it and this gate asserted exactly that — "in series" was the property under
    // test. The picture now leaves every realm in ONE hop to the connection plane, so the deeper
    // relay leg must NOT be worse than the shallower one by anything the depth explains: the two
    // relay legs are each one hop, and both are bounded by the same derived budget under loss.
    let lossy_budget =
        vd_sim::stub::aoi_recheck_cadence(&vd_tests::area_stub_config()) + CHAIN_LATENCY_TICKS / 4; // the redelivery allowance this fabric's attempt-drop costs
    assert!(
        lossy[1].1.p50_ticks <= lossy_budget,
        "the DEEPER relay leg is not paying for depth: depth 2 p50 {} against a derived budget of \
         {lossy_budget} (depth 1 p50 {})",
        lossy[1].1.p50_ticks,
        lossy[0].1.p50_ticks,
    );
    // (3) THE PICTURE STILL ARRIVES ACROSS THE GAPS. Under a half-drop the edge takes news on
    // roughly half of ticks — a client draws something slightly old, never nothing. Asserted as a
    // RATE over the whole lossy arm, not as a single sample: one empty tick's inbox is exactly
    // what an attempt-drop is, and reading one would be measuring the coin.
    assert!(
        lossy[2].0 > 0.0,
        "the gateway edge received no picture at all across the gaps: news {:.2}/tick",
        lossy[2].0,
    );
}
