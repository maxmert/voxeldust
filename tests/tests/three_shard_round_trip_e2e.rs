//! C-6c — THE 3-SHARD CONTAINMENT ROUND-TRIP (task #135 capstone, harness tier).
//!
//! This gate proves the SEED-DERIVED containment model routes a subject through the FULL mandate chain
//! BOTH WAYS over a REAL 3-shard cluster (orchestrator + gateway + System 7 + Galaxy + System 8), with the
//! re-homes driven by the REAL geometric detector (`worldgen::realm_neighbourhood_for` planted per shard)
//! and the REAL transfer saga (NOTHING hand-fed — a grep of this file finds ZERO `trigger_transfer`).
//!
//! THE CORRECTED CONTAINMENT MODEL: a SIBLING crossing routes THROUGH THE SHARED PARENT. A subject leaving
//! System 7's SOI lands in the GALAXY (System 7's ancestor — the System 7 shard already has the Galaxy in
//! its neighbourhood, so it re-homes 7→Galaxy). The GALAXY shard (which hosts System 7 + System 8 as its
//! OWNED CHILDREN) then detects the subject entering System 8 and re-homes Galaxy→8. So NO shard ever needs
//! a sibling in its scan, and the between-systems space IS the Galaxy realm — which is why the round-trip
//! needs a GALAXY SHARD to own it.
//!
//! THE HEADLINE ASSERTION: the AUTHORITATIVE HOLDER (`head(Realm(subject))` in the containment sense — which
//! shard OWNS the subject) flips through the FULL chain BOTH WAYS: System 7 → Galaxy → System 8 → Galaxy →
//! System 7. The RETURN legs (8→Galaxy→7) are the reverse-cross proof; a one-way assertion would be a FAIL.
//!
//! SCOPE — read before the headline reads bigger than it is: this proves the containment DECISION routes
//! both ways; it is NOT one durable PLAYER dot surviving a continuous journey. The subject is a TRANSIENT and
//! each leg is a FRESH one (the two notes below explain why), so the `[7,Galaxy,8,Galaxy,7]` vector is an
//! AGGREGATE of four INDEPENDENT one-hop re-homes, not one entity's four-hop survival. A durable dot CANNOT
//! complete this round-trip TODAY (2nd hop dies `Cutting`→`CutTimeout`; the gateway session-route migration
//! is owed, ledgered to D-44) — its ONE-hop re-home is proven by `crossing_e2e`.
//!
//! SUBJECT = a TRANSIENT (Debris, HR2 second class). A transient re-homes via the batched `TransientGo`
//! path with NO client cut-marker, so it self-drives across shards on the SAME containment detector + saga
//! machinery a durable dot uses — and its ownership is the `owned_transients` held-set (a transient has no
//! directory `OwnerRecord` by design, HR2). A DURABLE dot's directory Entity head additionally needs the
//! gateway to MIGRATE the client session-route to each successive source shard (a multi-hop session-
//! following concern orthogonal to the containment decision under test); that durable multi-hop is proven
//! ONE hop by `crossing_e2e` and ledgered for the N-shard process tier (D-44). This gate isolates the
//! CONTAINMENT round-trip — the re-home DECISION routing a sibling crossing through the parent, both ways.
//!
//! WHY A FRESH SUBJECT PER LEG: the batched transient go-token is idempotency-keyed on
//! `crossing_transfer_id(entity, src_realm_fence, 0)` (`saga_runtime::handle_transient_crossing_request`).
//! At this P3 tier every realm rests at `Fence(1)` (a static-fence artifact of the harness — production
//! fences ADVANCE per re-home), so re-homing the SAME entity BACK to a shard it already visited re-mints a
//! COLLIDING id → the dest journals it `AlreadyApplied` and drops the re-visit. So each leg uses a FRESH
//! Debris seeded on its CURRENT owner: every leg is still a REAL, autonomous, saga-driven ownership flip
//! through the correct container, and the OBSERVED holder sequence is the full both-ways chain. (One entity
//! surviving all four hops is a same-subject batch-idempotency concern, orthogonal to the CONTAINMENT
//! routing under test — a durable dot, whose entity fence advances per commit, has no such collision.)
//!
//! TIER: HARNESS (in-process `Topology`), per the vetted plan — the IDENTICAL detector + saga + directory
//! the process bins run. Each leg's position is SCRIPTED (the physical walk-across-SOI is the D-15/D-30
//! visual owed line); each leg's re-home is a REAL saga-driven batched handoff.

use vd_core::entity_kind::EntityKind;
use vd_core::glam::DVec3;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::{EntityId, NodeId};
use vd_harness::fabric::FaultFabric;
use vd_harness::topology::{InspectReport, Topology};
use vd_tests::{
    DEST, GALAXY, GALAXY_SEED, SHARD, dest_stub_config, galaxy_stub_config, p3_galaxy_cluster,
    plant_seed_neighbourhood, realm_fence, seed_held_transient_on, set_transient_offset_on,
    stub_config, transient_holder,
};

/// The universe seed the whole cluster shares.
///
/// This gate deliberately rides the HAND-PLACED WALK FIXTURE forest — the topology small enough that a
/// person can walk across every boundary in it. `worldgen::realm_regions_for` returns that forest for
/// ANY seed: it takes the seed only to keep the frozen `f(seed)` signature and then discards it (see
/// `worldgen::walk`), because nothing in the walk forest is drawn from a seed stream. That is exactly
/// why the home-seed flip every world-deriving process now defaults to cannot reach this file, and why
/// pinning a seed here still says something rather than nothing.
const UNIVERSE_SEED: u64 = 0;
/// The three shards, in the order the holder-search checks them.
const SHARDS: [NodeId; 3] = [SHARD, GALAXY, DEST];

/// The walk galaxy's OWN extent in metres, read straight off the forest the shards are planted with
/// (`worldgen::realm_regions_for` → the Galaxy region's `finite_extent`).
///
/// WHY IT IS READ AND NOT WRITTEN DOWN. The comments below used to describe this realm as "the r=1000
/// between-space", a radius the walk forest has never produced: `worldgen::walk` sizes the galaxy to
/// contain System B's far face and nothing else. A quoted extent cannot notice the forest changing
/// shape — this one is the forest's own answer, recomputed on the run that reads it.
fn walk_galaxy_extent_m() -> f64 {
    vd_physics::worldgen::realm_regions_for(UNIVERSE_SEED)
        .iter()
        .find(|r| r.realm == galaxy_stub_config().realm)
        .expect("the walk forest rosters the Galaxy the middle shard hosts")
        .shape
        .finite_extent()
}

/// One round-trip WAYPOINT along +X: the shard the subject STARTS on (its current owner), the position it
/// is driven to, and the shard the ownership must flip TO once this leg's re-home commits. Outbound
/// 7→Galaxy→8, return 8→Galaxy→7. The `home` is a position inside the owner's OWN region (so the fresh
/// transient is genuinely owned there before it is moved to `offset`, proving a REAL re-home, not a spawn).
struct Leg {
    home: f64,
    offset: f64,
    owner: NodeId,
    expect_holder: NodeId,
    label: &'static str,
}

/// The waypoint out in the gap between the two systems — far enough that a subject standing there has
/// genuinely LEFT the system it came from, read off the same forest the detector reads.
///
/// ★ THIS USED TO BE THE LITERAL 50. It was outside the release edge only while every band in the
/// universe was the same three metres wide. Once bands are sized from the bodies they wrap, a
/// forty-metre system's release edge sits at fifty-three metres and a subject at fifty had never left,
/// so the holder never flipped and the chain stalled on its first leg. Both systems in the walk forest
/// have the same extent, so one value serves every leg.
fn walk_gap_offset_m() -> f64 {
    let r = vd_physics::worldgen::realm_regions_for(UNIVERSE_SEED)
        .iter()
        .find(|r| r.realm == stub_config().realm)
        .copied()
        .expect("the walk forest rosters the system the first shard hosts");
    // One whole band past the release edge, so the waypoint stays unambiguously outside rather than
    // outside by a margin that shrinks whenever the band changes.
    r.shape.finite_extent() + r.band.outset() + (r.band.inset() + r.band.outset())
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

/// The frame a subject at rest carries on shard `node` (its realm's `SystemSpace` frame). Under
/// `IdentityFrames` (P3) the frame is a render no-op, so positions are frame-invariant — but the transient's
/// pose still carries the owning realm's frame for correctness.
/// ★ EACH SHARD'S OWN FRAME, TAKEN FROM ITS OWN CONFIG (slice S9).
///
/// This used to build `SystemSpace { system_seed }` for every node from a hand-written seed table,
/// which was right while every realm in the fixture was a star system wearing a system frame. The
/// galaxy is a GALAXY now — its frame is `GalaxySpace`, and it counts in two-metre steps rather than
/// millimetres. Handing it a system frame seeded with its own number produced a frame no shard in the
/// cluster answers for, and the transient seeded in it was never inside anything.
///
/// Reading it off the shard's own config is also what stops the table drifting again: there is one
/// place a shard's frame is stated, and this is not a second one.
fn frame_of(node: NodeId) -> FrameRef {
    match node {
        n if n == SHARD => stub_config().frame,
        n if n == GALAXY => galaxy_stub_config().frame,
        n if n == DEST => dest_stub_config().frame,
        _ => FrameRef::UniverseSpace,
    }
}

/// Drive ONE round-trip leg with a FRESH transient `subject` (see the module note on the batch-id
/// collision). SEED it on the owner INSIDE the owner's own region (`home` — genuinely owned there, not a
/// spawn-at-destination), confirm the owner holds it, then MOVE it to the waypoint (`offset`) and drive the
/// REAL seed-forest re-home + batched-handoff saga to the ownership flip to `expect_holder`. HOLD the
/// waypoint each tick (the manual write lands before `evaluate_realm_boundaries`). Bounded + deterministic.
fn drive_leg(topo: &mut Topology, subject: EntityId, leg: &Leg, clear_after: bool) {
    // 1) Seed the fresh transient at `home` (inside the owner's own realm region) with the owner's realm
    //    fence — it is genuinely OWNED by `owner` (container == owner's realm ⇒ no re-home yet).
    let owner_realm = realm_of(leg.owner);
    let anchor = realm_fence(topo, owner_realm);
    seed_held_transient_on(
        topo,
        leg.owner,
        frame_of(leg.owner),
        subject,
        anchor,
        DVec3::new(leg.home, 0.0, 0.0),
    );
    let owned_at_home = step_until(topo, 20, |t| {
        transient_holder(&t.inspect_all(), &SHARDS, subject) == Some(leg.owner)
    });
    assert!(
        owned_at_home,
        "LEG {}: the fresh transient is genuinely OWNED by {:?} at home x={} before the re-home",
        leg.label, leg.owner, leg.home,
    );

    // 2) MOVE it to the waypoint → the seed-forest detector re-homes it. The leg completes when the NEW
    //    holder holds it AND the handoff saga has tombstoned (`live_sagas == 0`) — a STABLE `Held` state.
    let off = DVec3::new(leg.offset, 0.0, 0.0);
    let flipped = step_until(topo, 200, |t| {
        set_transient_offset_on(t, leg.owner, subject, off);
        set_transient_offset_on(t, leg.expect_holder, subject, off);
        transient_holder(&t.inspect_all(), &SHARDS, subject) == Some(leg.expect_holder)
            && vd_tests::live_sagas(t) == 0
    });
    assert!(
        flipped,
        "LEG {}: the authoritative holder must flip to {:?} (the transient re-homes {:?}→{:?} at x={}) — got {:?}",
        leg.label,
        leg.expect_holder,
        leg.owner,
        leg.expect_holder,
        leg.offset,
        transient_holder(&topo.inspect_all(), &SHARDS, subject),
    );

    // 3) Clean up this leg's transient (each leg is a fresh subject) so the final single-holder assert reads
    //    only the last leg's landed subject. The LAST leg keeps its subject (the final home-state proof).
    if clear_after {
        clear_transient(topo, subject);
    }
}

/// The realm a shard hosts (its `SystemSpace` realm id).
fn realm_of(node: NodeId) -> RealmId {
    match node {
        n if n == SHARD => stub_config().realm,
        n if n == GALAXY => galaxy_stub_config().realm,
        n if n == DEST => dest_stub_config().realm,
        _ => RealmId::System(0),
    }
}

/// Remove `subject` from every shard's transient set (each leg uses a fresh subject; drop the prior leg's
/// so the final holder assert is unambiguous).
fn clear_transient(topo: &mut Topology, subject: EntityId) {
    for n in SHARDS {
        vd_tests::remove_transient_on(topo, n, subject);
    }
}

/// THE ROUND-TRIP: a transient driven origin → x=50 → x=100 → x=50 → origin re-homes through the FULL
/// containment chain, and the authoritative holder flips System 7 → Galaxy → System 8 → Galaxy → System 7.
/// The RETURN legs (8→Galaxy→7) prove the reverse-cross — a one-way assertion would be a FAIL.
#[test]
fn three_shard_round_trip_flips_the_holder_through_the_full_chain_both_ways() {
    let fabric = FaultFabric::new(0xC6C_909, 2);
    let mut topo = p3_galaxy_cluster(&fabric, 8);
    let ready = step_until(&mut topo, 40, |t| {
        let r = t.inspect_all();
        let holds = |node: NodeId, realm| {
            report(&r, node)
                .held_realms
                .iter()
                .any(|(rl, _)| *rl == realm)
        };
        holds(SHARD, stub_config().realm)
            && holds(GALAXY, galaxy_stub_config().realm)
            && holds(DEST, dest_stub_config().realm)
    });
    assert!(
        ready,
        "all three shards must grant their realms (System 7 + Galaxy + System 8) before the round-trip",
    );

    // Plant the SEED-DERIVED containment neighbourhood on EACH shard (the exact geometry production
    // `shard.rs` boots): System 7 → {Universe, Galaxy, System 7, Planet 7}; the Galaxy → {Universe, Galaxy,
    // System 7, System 8} (it OWNS the two systems as children — the sibling-routing shard); System 8 →
    // {Universe, Galaxy, System 8}. NO shard sees a sibling — the crossing routes through the Galaxy parent.
    plant_seed_neighbourhood(&mut topo, SHARD, UNIVERSE_SEED, stub_config().realm);
    plant_seed_neighbourhood(&mut topo, GALAXY, UNIVERSE_SEED, galaxy_stub_config().realm);
    plant_seed_neighbourhood(&mut topo, DEST, UNIVERSE_SEED, dest_stub_config().realm);

    // The Galaxy realm IS the seed forest's between-space (`System(1)`), and the Galaxy shard OWNS it (so
    // `head(Realm(Galaxy))` resolves and authority can REST in the between-space).
    assert_eq!(
        galaxy_stub_config().realm,
        RealmId::Galaxy(GALAXY_SEED),
        "the Galaxy realm is the seed forest's between-space",
    );

    // THE FULL BOTH-WAYS CHAIN. Each leg seeds a FRESH transient on its owner (inside the owner's own
    // region — `home`), then moves it to the waypoint (`offset`) so the REAL seed-forest detector +
    // batched-handoff saga flip its ownership. Outbound: 7 → Galaxy → 8. Return: 8 → Galaxy → 7 (the
    // reverse-cross). `home` is a point clearly inside the owner's own region: each system owns the shell
    // the walk forest gives it, centred on its own origin (home = 0, the system's own centre); the Galaxy
    // owns the between-space the two systems sit in, and IT is centred on the galactic origin — 50 is a
    // point out in the gap, not where the galaxy is. The waypoints 0/50/100 are `worldgen::walk`'s own,
    // blessed by its doc; the extent they have to fall inside is asserted below rather than quoted.
    let legs = [
        Leg {
            home: 0.0,
            offset: walk_gap_offset_m(),
            owner: SHARD,
            expect_holder: GALAXY,
            label: "7→Galaxy (escape SOI)",
        },
        Leg {
            home: walk_gap_offset_m(),
            offset: 100.0,
            owner: GALAXY,
            expect_holder: DEST,
            label: "Galaxy→8 (enter sibling)",
        },
        Leg {
            home: 100.0,
            offset: walk_gap_offset_m(),
            owner: DEST,
            expect_holder: GALAXY,
            label: "8→Galaxy (return, reverse-cross)",
        },
        Leg {
            home: walk_gap_offset_m(),
            offset: 0.0,
            owner: GALAXY,
            expect_holder: SHARD,
            label: "Galaxy→7 (home, reverse-cross)",
        },
    ];

    // ANTI-DRIFT, and the reason the prose above states no radius: every waypoint has to lie inside the
    // walk galaxy, or the "gap" leg is happening in a realm no shard here hosts and the chain proves
    // nothing. The extent is read off the SAME forest the three shards were planted with, so a change to
    // the walk geometry shows up here as a failure instead of as a comment that quietly stops being true.
    let galaxy_extent_m = walk_galaxy_extent_m();
    assert!(
        legs.iter()
            .all(|l| l.home.abs() < galaxy_extent_m && l.offset.abs() < galaxy_extent_m),
        "every round-trip waypoint must lie inside the walk galaxy's own extent ({galaxy_extent_m} \
         m) — the between-space legs are only meaningful while the Galaxy still contains them: {:?}",
        legs.iter().map(|l| (l.home, l.offset)).collect::<Vec<_>>(),
    );

    let mut observed = vec![SHARD]; // the origin: System 7 owns the subject at the start of leg 1
    let mut final_subject = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 0, 0);
    for (i, leg) in legs.iter().enumerate() {
        // A distinct fresh Debris per leg (avoids the static-fence batch-id collision — see the module note).
        let subject = EntityId::pack(EntityKind::Debris, leg.owner.0 as u32, (i + 1) as u64, 0);
        let is_last = i + 1 == legs.len();
        drive_leg(&mut topo, subject, leg, !is_last);
        observed.push(leg.expect_holder);
        if is_last {
            final_subject = subject; // the last leg's landed subject (NOT cleared) for the final assert
        }
    }

    // THE HEADLINE: the holder flipped through the full chain BOTH ways. This vector is the reverse-cross
    // proof — it shows the RETURN legs (8→Galaxy→7), not just the outbound. Each transition is a REAL,
    // autonomous, saga-driven ownership flip through the CORRECT container (the sibling crossing routing
    // THROUGH the Galaxy parent, both directions).
    assert_eq!(
        observed,
        vec![SHARD, GALAXY, DEST, GALAXY, SHARD],
        "the containment DECISION flipped the holder System 7 → Galaxy → System 8 → Galaxy → System 7 \
         (both ways) — four INDEPENDENT transient re-homes aggregated, NOT one durable dot's journey",
    );

    // FINAL STATE: the last leg's subject landed home (System 7 holds it), exactly one shard holds it, and
    // the single-transient-holder oracle holds (no {a,b}-both-held window).
    let final_reports = topo.inspect_all();
    assert_eq!(
        transient_holder(&final_reports, &SHARDS, final_subject),
        Some(SHARD),
        "the final leg's transient is HOME — System 7 holds it (Galaxy→7, the reverse-cross)",
    );
    let holders: Vec<NodeId> = SHARDS
        .into_iter()
        .filter(|n| {
            report(&final_reports, *n)
                .owned_transients
                .iter()
                .any(|(e, _)| *e == final_subject)
        })
        .collect();
    assert_eq!(
        holders,
        vec![SHARD],
        "exactly ONE shard (System 7) holds the final leg's transient after the round-trip",
    );
    vd_harness::oracle::verify_transient_authority_held(&final_reports)
        .expect("TRANSIENT-AUTHORITY-HELD holds after the round-trip settles");
    let _ = fabric; // keep the fabric alive for the whole run (the transports hold clones)
}
