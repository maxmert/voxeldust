//! task #149 — THE SOURCE==DEST RE-HOME END-TO-END PROOF (the un-hosted-child cure).
//!
//! Before this change a dot that re-homed into a realm ITS OWN shard also hosts (a co-hosted child) took a
//! LOCAL relabel short-circuit — it never drove the transfer saga, because `head(Realm(child))` resolved to
//! THIS SAME node (source==dest) and the old belief was that such a saga would strand. This gate PROVES the
//! opposite: a same-node crossing drives the ONE uniform orchestrator saga to completion — the gateway
//! self-acks the cut, the directory CAS bumps the fence, the route swap is an idempotent no-op — and the
//! dot's authoritative pose ends re-expressed into the child realm's frame.
//!
//! THE SUBJECT is a REAL durable player (a genuinely logged-in avatar via `p1_client`, with a directory
//! `OwnerRecord`), on a SINGLE shard that co-hosts {System 7, Planet 7, Area 7} (the worldgen forest's
//! nested chain). It is driven Planet 7's origin → Area 7's box, so the deepest container flips to Area 7,
//! a realm THIS node heads — the source==dest degenerate case.
//!
//! THE HEADLINE ASSERTIONS:
//! - the durable crossing saga STARTED and tombstoned (`crossings_started >= 1`, `live_sagas == 0`) — the
//!   co-hosted re-home really drove the saga (not the deleted local short-circuit);
//! - the directory `head(Entity(subject))` STAYS on SHARD (source==dest — authority never left the node);
//! - the dot's authoritative pose FRAME becomes `AreaLocal { planet_seed: 7, area_seed: 7 }` — the Area
//!   label FORMS (the parent-provenance `to_parent` threaded the enclosing Planet 7). This is the "Area
//!   label never flips" fix, proven end-to-end over the real gateway + saga + directory + detector.
//! - the REVERSE (Area 7 → Planet 7) drops the frame back to `PlanetCentered { planet_seed: 7 }`.
//! - IDEMPOTENCE: with the dot held stationary inside Area 7 after commit, NO further crossing fires
//!   (`crossings_started` stays constant) — no re-fire loop.
//!
//! TIER: HARNESS (in-process `Topology`) — the IDENTICAL gateway + saga + directory + geometric detector the
//! process bins run. The dot's position is SCRIPTED (`set_shard_subject_offset`); each re-home is a REAL,
//! autonomous, saga-driven directory CAS.

use std::collections::BTreeSet;
use vd_core::glam::DVec3;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::{AccountId, EntityId, NodeId};
use vd_harness::client::{InputCmd, ScriptedClient};
use vd_harness::fabric::FaultFabric;
use vd_harness::topology::{InspectReport, Topology};
use vd_sim::stub::DiscardReason;
use vd_tests::{
    ORCH, SHARD, entity_head_node, live_sagas, p1_client, p2_cluster_cohost,
    plant_seed_neighbourhood_held, read_subject, saga_states, set_shard_subject_offset,
    shard_subject_frame, walk_forward,
};

const UNIVERSE_SEED: u64 = 0;
const CLIENT: NodeId = NodeId(100);

/// THE PLANET PARKING SPOT — ONE point in the world, stated separately in every frame the dot can be
/// wearing on its way to and from it.
///
/// A bare offset is not a place. A pose carries the frame it is measured in, and a re-home CHANGES that
/// frame, so writing the same three numbers every tick MOVES the dot by the whole distance between the two
/// realms the instant it crosses. It used not to matter: a hand-off relabelled the frame and left the
/// number untouched, so one triple appeared to describe every space at once — which is exactly the fault
/// that drew the player a planet-radius off the surface they were standing on. With the hand-off actually
/// converting, a scripted position has to name its frame.
///
/// The point: inside Planet 7 (radius 10, sitting 20 m from its star) and clear of the Area 7 box (centred
/// 5 m from the planet, half-extent 3) by more than the 2 m containment outset, so the reverse leg cleanly
/// EXITS the area. In the star's frame that is x = 15; in the planet's, 15 − 20 = −5; in the area's,
/// −5 − 5 = −10. Each row is the SAME point, and the numbers below are what that point measures as from
/// each of those three centres:
///   star's frame    x =  15  →  |15−20|−10 = −5, inside the planet
///   planet's frame  x =  −5  →  |−5|−10    = −5 inside; |−5−5|−3 = +7, well clear of the area
///   area's frame    x = −10  →  |−10|−3    = +7 clear of the area, still −5 inside the planet
const PLANET_SPOT: [(FrameRef, f64); 3] = [
    (FrameRef::SystemSpace { system_seed: 7 }, 15.0),
    (FrameRef::PlanetCentered { planet_seed: 7 }, -5.0),
    (
        FrameRef::AreaLocal {
            planet_seed: 7,
            area_seed: 7,
        },
        -10.0,
    ),
];
/// THE AREA PARKING SPOT — the centre of Area 7's box, the same point in each frame the dot wears around
/// that crossing: 25 from the star, 5 from the planet that carries it, 0 from its own centre.
const AREA_SPOT: [(FrameRef, f64); 3] = [
    (FrameRef::SystemSpace { system_seed: 7 }, 25.0),
    (FrameRef::PlanetCentered { planet_seed: 7 }, 5.0),
    (
        FrameRef::AreaLocal {
            planet_seed: 7,
            area_seed: 7,
        },
        0.0,
    ),
];

/// Hold the scripted dot at `spot`, choosing the row that matches the frame its pose is CURRENTLY stamped
/// in. Answers `false` (and writes nothing) if the dot is missing or wearing a frame the table does not
/// describe — a caller waiting on a frame flip then simply keeps waiting rather than teleporting the dot.
fn park(topo: &mut Topology, subject: EntityId, spot: &[(FrameRef, f64)]) -> bool {
    let Some(frame) = shard_subject_frame(topo, SHARD, subject) else {
        return false;
    };
    let Some((_, x)) = spot.iter().find(|(f, _)| *f == frame) else {
        return false;
    };
    set_shard_subject_offset(topo, SHARD, subject, DVec3::new(*x, 0.0, 0.0))
}

fn report(reports: &[(NodeId, InspectReport)], id: NodeId) -> &InspectReport {
    &reports
        .iter()
        .find(|(n, _)| *n == id)
        .expect("node present")
        .1
}

fn step_until(topo: &mut Topology, max: u64, mut cond: impl FnMut(&mut Topology) -> bool) -> bool {
    for _ in 0..max {
        topo.step();
        if cond(topo) {
            return true;
        }
    }
    false
}

/// Spin the co-hosting cluster + a logged-in client, warm up until the shard grants its System 7 realm AND
/// the avatar, plant the seed neighbourhood (System 7 ⊃ Planet 7 ⊃ Area 7 — all co-hosted here), pause the
/// client input (positions are scripted), and PARK the dot at Planet 7's origin (container == Planet 7).
/// Returns the quiesced topo + the durable subject. The dot's owning frame starts `PlanetCentered{7}` after
/// the first Planet re-home settles.
fn warm_cohost_cluster_at_planet(fabric: &FaultFabric) -> (Topology, EntityId) {
    let mut topo = p2_cluster_cohost(fabric, 8);
    topo.add_node(Box::new(p1_client(
        fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    // WARMUP: the shard grants its System 7 realm (and, via co-hosting, Planet 7 + Area 7) AND the player
    // logs in + the shard grants the durable avatar.
    let ready = step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        report(&r, SHARD)
            .held_realms
            .iter()
            .any(|(rl, _)| *rl == RealmId::System(7))
            && !report(&r, SHARD).held_entities.is_empty()
    });
    assert!(
        ready,
        "the co-hosting shard grants its System 7 realm + the durable avatar before the crossing",
    );

    // Anti-vacuity: the shard actually co-hosts Planet 7 + Area 7 (else `head(Realm(Area 7))` would not
    // resolve to it and the saga could not resolve its dest head at all).
    let warm = topo.inspect_all();
    let held: Vec<RealmId> = report(&warm, SHARD)
        .held_realms
        .iter()
        .map(|(rl, _)| *rl)
        .collect();
    assert!(
        held.contains(&RealmId::Planet(7)) && held.contains(&RealmId::Area(7)),
        "the shard co-hosts BOTH Planet 7 and Area 7 (source==dest can resolve): held = {held:?}",
    );

    let (_session, subject, _fence) = read_subject(&mut topo);
    assert_eq!(
        entity_head_node(&mut topo, subject),
        Some(SHARD),
        "the durable player's directory head is the co-hosting login shard",
    );

    // Plant the SEED-DERIVED containment neighbourhood for the WHOLE co-hosted held set (System 7 ⊃ Planet 7
    // ⊃ Area 7 + ancestors). Area 7 is a GRANDCHILD of System 7, so a single-realm `plant_seed_neighbourhood`
    // (own + ancestors + DIRECT children) would omit it — the held-union planter includes it so the detector
    // can flip the deepest container to Area 7.
    let held = BTreeSet::from([RealmId::System(7), RealmId::Planet(7), RealmId::Area(7)]);
    plant_seed_neighbourhood_held(&mut topo, SHARD, UNIVERSE_SEED, &held);

    // Positions are SCRIPTED — pause the client input (the session stays Active so the route swap follows).
    let node = topo.node_mut(CLIENT).expect("client present");
    node.as_any_mut()
        .expect("clients opt into downcasting")
        .downcast_mut::<ScriptedClient>()
        .expect("the node is a ScriptedClient")
        .pause_input();

    // PARK at Planet 7's origin so the first (Planet) re-home settles — the dot's owning frame becomes
    // PlanetCentered{7} and its head STAYS on SHARD (Planet 7 is co-hosted). Bounded + deterministic.
    let at_planet = step_until(&mut topo, 200, |t| {
        park(t, subject, &PLANET_SPOT);
        shard_subject_frame(t, SHARD, subject) == Some(FrameRef::PlanetCentered { planet_seed: 7 })
            && live_sagas(t) == 0
    });
    assert!(
        at_planet,
        "the durable dot re-homed into co-hosted Planet 7 (pose frame == PlanetCentered{{7}}, head stays \
         SHARD): frame={:?}, head={:?}, saga_states={:?}",
        shard_subject_frame(&mut topo, SHARD, subject),
        entity_head_node(&mut topo, subject),
        saga_states(&mut topo),
    );
    // The head never left the node — the Planet re-home was a source==dest saga.
    assert_eq!(
        entity_head_node(&mut topo, subject),
        Some(SHARD),
        "a co-hosted Planet re-home keeps authority on THIS node (source==dest)",
    );

    (topo, subject)
}

/// THE SOURCE==DEST RE-HOME: a durable dot walks from co-hosted Planet 7 into co-hosted Area 7 (same node
/// as source and dest) and the uniform saga completes with the pose re-expressed into the Area frame — the
/// authoritative proof the local short-circuit deletion is correct AND that the Area label forms.
#[test]
fn source_equals_dest_rehome_into_area_flips_the_frame_and_stays_on_node() {
    let fabric = FaultFabric::new(0x149_A5EA, 2);
    let (mut topo, subject) = warm_cohost_cluster_at_planet(&fabric);

    let crossings_before = report(&topo.inspect_all(), SHARD).crossings_requested;

    // WALK into Area 7's box (x=25) — the deepest container flips to Area 7, a realm THIS node heads. The
    // crossing resolves `head(Realm(Area 7)) == SHARD` (source==dest) and drives the ONE uniform saga.
    let in_area = step_until(&mut topo, 260, |t| {
        park(t, subject, &AREA_SPOT);
        shard_subject_frame(t, SHARD, subject)
            == Some(FrameRef::AreaLocal {
                planet_seed: 7,
                area_seed: 7,
            })
            && live_sagas(t) == 0
    });
    assert!(
        in_area,
        "the durable dot re-homed into co-hosted Area 7: the pose frame flips to AreaLocal{{7,7}} AND the \
         saga tombstoned — got frame={:?}, head={:?}, saga_states={:?}",
        shard_subject_frame(&mut topo, SHARD, subject),
        entity_head_node(&mut topo, subject),
        saga_states(&mut topo),
    );

    let r = topo.inspect_all();
    // (1) THE COMPOSITION PROOF: a NEW durable crossing saga fired for the Planet→Area re-home (source==dest
    //     really drove the saga, not the deleted local relabel).
    assert!(
        report(&r, SHARD).crossings_requested > crossings_before,
        "the Planet→Area re-home emitted a fresh CrossingRequest (source==dest drove the saga): {} > {}",
        report(&r, SHARD).crossings_requested,
        crossings_before,
    );
    assert!(
        report(&r, ORCH).crossings_started >= 1,
        "the orchestrator STARTED the co-hosted crossing saga (crossings_started >= 1): {}",
        report(&r, ORCH).crossings_started,
    );
    // (2) SOURCE==DEST: the directory head NEVER left the node (a co-hosted re-home commits authority to the
    //     SAME shard — the CAS just bumps the fence).
    assert_eq!(
        entity_head_node(&mut topo, subject),
        Some(SHARD),
        "source==dest: authority stays on the co-hosting node across the Area re-home",
    );

    // (3) THE REVERSE: walk back to Planet 7's origin (out of the Area box) — the frame drops back to
    //     PlanetCentered{7}, proving the return leg is ALSO the uniform saga, still on-node.
    let back_at_planet = step_until(&mut topo, 260, |t| {
        park(t, subject, &PLANET_SPOT);
        shard_subject_frame(t, SHARD, subject) == Some(FrameRef::PlanetCentered { planet_seed: 7 })
            && live_sagas(t) == 0
    });
    assert!(
        back_at_planet,
        "the reverse Area 7 → Planet 7 re-home drops the frame back to PlanetCentered{{7}}: frame={:?}",
        shard_subject_frame(&mut topo, SHARD, subject),
    );
    assert_eq!(
        entity_head_node(&mut topo, subject),
        Some(SHARD),
        "source==dest: authority still on-node after the reverse re-home",
    );
}

/// IDEMPOTENCE (anti re-fire loop): after the source==dest saga commits, HOLD the dot stationary inside
/// Area 7 for many more ticks and assert NO further crossing fires (the started count stays constant) and
/// the owning frame stays AreaLocal. This catches a re-fire loop the deleted short-circuit could have hidden
/// (a committed co-hosted re-home whose owning realm never advances would re-fire every tick forever).
#[test]
fn a_committed_source_equals_dest_rehome_does_not_re_fire() {
    let fabric = FaultFabric::new(0x149_1DE1, 2);
    let (mut topo, subject) = warm_cohost_cluster_at_planet(&fabric);

    // Drive into Area 7 and let the saga commit (frame flips + tombstones).
    let committed = step_until(&mut topo, 260, |t| {
        park(t, subject, &AREA_SPOT);
        shard_subject_frame(t, SHARD, subject)
            == Some(FrameRef::AreaLocal {
                planet_seed: 7,
                area_seed: 7,
            })
            && live_sagas(t) == 0
    });
    assert!(
        committed,
        "the dot committed into Area 7 before the idempotence window"
    );

    // Snapshot the started/requested counts at the committed instant.
    let started_after_commit = report(&topo.inspect_all(), ORCH).crossings_started;

    // HOLD stationary inside Area 7 for >= 10 more ticks — NO new saga must start.
    for _ in 0..12 {
        park(&mut topo, subject, &AREA_SPOT);
        topo.step();
    }

    let r = topo.inspect_all();
    assert_eq!(
        report(&r, ORCH).crossings_started,
        started_after_commit,
        "IDEMPOTENCE: a committed source==dest re-home does not re-fire (crossings_started stayed constant)",
    );
    assert_eq!(
        shard_subject_frame(&mut topo, SHARD, subject),
        Some(FrameRef::AreaLocal {
            planet_seed: 7,
            area_seed: 7,
        }),
        "the dot stays owned in Area 7 across the idempotence window (owning realm did not thrash)",
    );
    assert_eq!(
        entity_head_node(&mut topo, subject),
        Some(SHARD),
        "authority stays on the co-hosting node across the idempotence window",
    );
    assert_eq!(
        live_sagas(&mut topo),
        0,
        "no lingering saga across the idempotence window",
    );
}

/// INPUT LIVENESS across a source==dest re-home — the regression the two tests above CANNOT catch (they
/// `pause_input()` + teleport the dot with `set_shard_subject_offset`, so no live input ever rides the
/// gateway across the commit). A REAL logged-in durable player WALKS with LIVE gateway-routed input from
/// System 7's origin across the System 7 → Planet 7 boundary (a co-hosted saga) and KEEPS walking.
///
/// THE BUG THIS GUARDS: every re-home demotes the source dot Owned→Ghost, so `simulates()` is false and the
/// shard's apply-input gate drops forwarded WSAD as `PendingAuthority` UNLESS the dest input slot armed
/// `input_active`. For a co-hosted (source==dest) re-home the `OpenInputSlot` lands on the session's
/// ALREADY-GRANTED source dot, so the arming must NOT be gated on `!granted` — otherwise input freezes for
/// the whole demote→promote window (WSAD dead, only client-side mouse-look survives: the exact window
/// symptom). THIS gate proves the live-input path COMPLETES end-to-end: a REAL player walks, crosses the
/// co-hosted boundary, stays on-node, and no input is dropped in the harness. The tight in-process harness
/// timing collapses the demote→promote window to ~nothing, so this e2e does NOT by itself discriminate the
/// fix (it is green both ways) — the fail-before/pass-after PROOF of the `input_active` bridge is the unit
/// test `stub::open_input_slot_arms_input_active_on_an_already_granted_cohosted_dot`, which the wider
/// process-tier latency window (where you saw the freeze) makes load-bearing.
#[test]
fn a_walking_player_keeps_its_input_across_a_source_equals_dest_rehome() {
    let fabric = FaultFabric::new(0x0A11_E777, 2);
    let mut topo = p2_cluster_cohost(&fabric, 8);
    // Drive the player toward +X, where Planet 7 sits (the existing tests teleport the dot to x=15 to reach
    // it). `walk_forward`'s `[1,0,0]` maps to local −Z (`local_axes_from_movement`), which walks AWAY from
    // Planet 7; `movement[1]` maps to world +X and the spawn orient is identity, so `[0,1,0]` drives the
    // player straight at the co-hosted System 7 → Planet 7 boundary — a REAL walk under live input.
    topo.add_node(Box::new(p1_client(
        &fabric,
        CLIENT,
        AccountId(1000),
        |_| {
            Some(InputCmd {
                movement: [0.0, 1.0, 0.0],
                look: [0.0, 0.0],
            })
        },
    )));

    // Warm up: the shard grants System 7 (+ co-hosted Planet 7 / Area 7) and the durable avatar logs in.
    let ready = step_until(&mut topo, 150, |t| {
        let r = t.inspect_all();
        report(&r, SHARD)
            .held_realms
            .iter()
            .any(|(rl, _)| *rl == RealmId::Planet(7))
            && !report(&r, SHARD).held_entities.is_empty()
    });
    assert!(
        ready,
        "the co-hosting shard grants Planet 7 + the durable avatar before the walk",
    );

    let held = BTreeSet::from([RealmId::System(7), RealmId::Planet(7), RealmId::Area(7)]);
    plant_seed_neighbourhood_held(&mut topo, SHARD, UNIVERSE_SEED, &held);
    let (session, subject, _fence) = read_subject(&mut topo);

    // Confirm the player is ALREADY Owned + walking (its input is being APPLIED) before we start counting —
    // so any `PendingAuthority` discard we see later is the re-home window, not a benign login-time gap.
    let moving = step_until(&mut topo, 80, |t| {
        report(&t.inspect_all(), SHARD)
            .applied_inputs
            .iter()
            .any(|(s, _)| *s == session)
    });
    assert!(
        moving,
        "the durable player is Owned + walking (input applied) before the crossing",
    );

    // WALK with LIVE gateway-routed input across the System 7 → Planet 7 boundary (a co-hosted saga) and keep
    // going ~40 ticks past the saga start (covering the demote→promote window). Track the WORST
    // `PendingAuthority` discard count seen for the session at any tick — the bounded InputLog is a rolling
    // window, so max-over-time is robust to eviction.
    let mut worst_pending_auth = 0usize;
    let mut crossed_at: Option<u32> = None;
    for tick in 0..500u32 {
        topo.step();
        let r = topo.inspect_all();
        worst_pending_auth = worst_pending_auth.max(
            report(&r, SHARD)
                .discarded_inputs
                .iter()
                .filter(|(s, _, reason)| {
                    *s == session && *reason == DiscardReason::PendingAuthority
                })
                .count(),
        );
        if crossed_at.is_none() && report(&r, ORCH).crossings_started >= 1 {
            crossed_at = Some(tick);
        }
        if crossed_at.is_some_and(|t0| tick - t0 >= 40) {
            break;
        }
    }

    // ANTI-VACUITY: the co-hosted re-home actually fired — which REQUIRED the player's own input to walk it to
    // the Planet 7 boundary (a stationary dot never crosses), and the head stayed on-node (source==dest).
    assert!(
        crossed_at.is_some(),
        "the player walked to + crossed the co-hosted System 7 → Planet 7 boundary under its own input",
    );
    assert_eq!(
        entity_head_node(&mut topo, subject),
        Some(SHARD),
        "the crossing was source==dest — authority stayed on the co-hosting node",
    );
    // THE SYMPTOM: NO forwarded WSAD input was EVER dropped as `PendingAuthority` across the demote→promote
    // window. Before the `input_active` fix this is > 0 (the frozen-player bug); after, it is 0.
    assert_eq!(
        worst_pending_auth, 0,
        "no forwarded WSAD input was dropped as PendingAuthority across the source==dest re-home window — \
         the input_active bridge holds across the source dot's demote→promote",
    );
}
