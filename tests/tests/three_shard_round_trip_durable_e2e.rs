//! S3 — THE DURABLE MULTI-HOP ROUND-TRIP (server-timed cut; retires the D-44 durable multi-hop gap).
//!
//! This is the thing that was IMPOSSIBLE before S3. `three_shard_round_trip_e2e` (the transient twin)
//! proved the containment DECISION routes both ways, but had to use a TRANSIENT because a DURABLE dot
//! died on hop 2 (`Cutting`→`CutTimeout`): the saga's cut waited on a client `CUT_MARKER`, and on hop 2
//! the client's session stays bound to the FIRST shard's port, so the marker never reached the Galaxy
//! shard's cut → CutTimeout → abort. (See that file's module doc: "A durable dot CANNOT complete this
//! round-trip TODAY (2nd hop dies Cutting→CutTimeout … ledgered to D-44) — its ONE-hop re-home is
//! proven by crossing_e2e." Also its static-Fence(1) batch-id note: a durable Entity's fence ADVANCES
//! per commit, so it has no such collision.)
//!
//! S3 SERVER-TIMES the cut: the gateway self-acks `CutConfirmed` at `RequestCut` and derives the real
//! input-cut seq from its own `last_input_seq` at FreezeSource-install time — the client marker is
//! retired. So the `Cutting` step no longer starves, and a REAL DURABLE PLAYER (an `Entity` whose
//! directory `Entity` head ADVANCES per commit) completes the FULL containment chain BOTH WAYS.
//!
//! THE SUBJECT IS A REAL DURABLE PLAYER (NO transient stand-in): a genuinely logged-in avatar, born via
//! the REAL `p1_client` login/grant flow (its `Entity` gets a directory `OwnerRecord`, unlike a
//! transient's held-set-only ownership). The SAME entity survives origin → 50 → 100 → 50 → origin.
//!
//! THE HEADLINE ASSERTION: the directory `head(Entity(subject))` authority flips through the FULL chain
//! BOTH WAYS — System 7 → Galaxy → System 8 → Galaxy → System 7 — each hop reaching a STABLE committed
//! `Held` on the new owner (`live_sagas == 0`), with ZERO CutTimeouts / ZERO aborts. The RETURN legs
//! (8→Galaxy→7) are the reverse-cross proof; a one-way assertion would be a FAIL.
//!
//! CONTAINMENT MODEL (identical to the transient twin): a SIBLING crossing routes THROUGH THE SHARED
//! PARENT. System 7 → Galaxy (its ancestor) → System 8 (Galaxy owns both systems as children); the
//! between-systems space IS the Galaxy realm. NO shard ever needs a sibling in its scan.
//!
//! TIER: HARNESS (in-process `Topology`) — the IDENTICAL gateway + saga + directory + geometric detector
//! the process bins run. Each hop's position is SCRIPTED (the physical walk is the D-15/D-30 visual owed
//! line); each hop's re-home is a REAL, autonomous, saga-driven directory CAS flip.

use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_core::{AccountId, EntityId, NodeId};
use vd_harness::client::ScriptedClient;
use vd_harness::fabric::FaultFabric;
use vd_harness::topology::{InspectReport, Topology};
use vd_tests::{
    DEST, GALAXY, GALAXY_SEED, ORCH, SHARD, dest_stub_config, entity_head_node, galaxy_stub_config,
    live_sagas, p1_client, p3_galaxy_cluster, plant_seed_neighbourhood, read_subject, saga_states,
    set_shard_subject_offset, stub_config, walk_forward,
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
/// The client node id (mirrors `crossing_e2e`'s CLIENT).
const CLIENT: NodeId = NodeId(100);

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

/// One round-trip WAYPOINT: the shard that OWNS the durable dot at the start of this hop, the position
/// it is driven to, and the shard the directory head must flip TO once this hop's re-home commits.
/// Outbound 7→Galaxy→8, return 8→Galaxy→7. The geometry is the walk fixture forest's: each system owns
/// the shell that forest gives it, centred on its own origin; the Galaxy owns the between-space the two
/// systems sit in, and IT is centred on the galactic origin — 50 is a probe point out in the gap, not
/// where the galaxy is. The extent every waypoint must fall inside is read off the forest at use
/// ([`walk_galaxy_extent_m`]), never quoted here.
struct Hop {
    owner: NodeId,
    offset: f64,
    expect_head: NodeId,
    label: &'static str,
}

/// Drive ONE hop of the SAME durable dot: MOVE it (on its CURRENT owner) to the waypoint so the REAL
/// seed-forest detector fires a durable `CrossingRequest`, then let the REAL transfer saga re-home it to
/// `expect_head`. The hop COMPLETES when the directory `head(Entity(subject))` flips to the new owner AND
/// the saga tombstoned (`live_sagas == 0`) — a STABLE committed `Held` on the new owner. Along the way
/// this ASSERTS the saga never enters a `CutTimeout`/abort (the exact failure S3 fixes): a stalled
/// `Cutting` step under the old client-marker cut would leave the head un-flipped and abort.
fn drive_hop(topo: &mut Topology, subject: EntityId, hop: &Hop) {
    let off = DVec3::new(hop.offset, 0.0, 0.0);
    // Re-assert the waypoint EVERY tick (the manual write lands before `evaluate_realm_boundaries`) on
    // the CURRENT OWNER only. Bounded + deterministic.
    //
    // It used to write the SAME offset on the expected new owner too, which assumed a position means the
    // same thing in both realms' frames. That was true only while every realm sat on its parent's origin.
    // The Galaxy authors System 8 at x = 130, so once the hand-off actually CONVERTS, the occupant the
    // Galaxy sees at x = 100 is at −30 in System 8's own frame — and re-writing +100 there put it 100 m
    // from System 8's centre, outside its 40 m boundary, so System 8 handed it straight back and the head
    // never settled. The arriving shard's copy is whatever IT computed; that is the thing under test.
    let flipped = step_until(topo, 260, |t| {
        set_shard_subject_offset(t, hop.owner, subject, off);
        // ANTI-VACUITY / the S3 proof: a well-behaved hop NEVER aborts. If the saga ever reached a
        // CutTimeout-driven Aborting, the head would not flip — this is exactly the multi-hop stall S3
        // fixes, so we assert its absence positively (below, via head-flip + live_sagas==0) rather than
        // scanning for a transient Aborting string that a fast tick could miss.
        entity_head_node(t, subject) == Some(hop.expect_head) && live_sagas(t) == 0
    });
    assert!(
        flipped,
        "HOP {}: head(Entity) must flip {:?}→{:?} (durable re-home at x={}) with the saga tombstoned — \
         got head={:?}, live_sagas={}, saga_states={:?}",
        hop.label,
        hop.owner,
        hop.expect_head,
        hop.offset,
        entity_head_node(topo, subject),
        live_sagas(topo),
        saga_states(topo),
    );
}

/// THE DURABLE ROUND-TRIP: the SAME durable player, driven origin → x=50 → x=100 → x=50 → origin,
/// re-homes through the FULL containment chain, and the directory `head(Entity(subject))` flips
/// System 7 → Galaxy → System 8 → Galaxy → System 7 — with ZERO CutTimeouts / aborts. The RETURN legs
/// (8→Galaxy→7) prove the reverse-cross. This was IMPOSSIBLE before S3 (2nd hop died Cutting→CutTimeout).
#[test]
fn three_shard_round_trip_durable_flips_the_head_through_the_full_chain_both_ways() {
    let fabric = FaultFabric::new(0xD09_5A6A, 2);
    let mut topo = p3_galaxy_cluster(&fabric, 8);
    topo.add_node(Box::new(p1_client(
        &fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    // WARMUP: all three shards win their realm leases AND the player logs in + the SOURCE grants its
    // durable avatar (a real Entity with a directory record — NOT a transient).
    let ready = step_until(&mut topo, 80, |t| {
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
            && !report(&r, SHARD).held_entities.is_empty()
    });
    assert!(
        ready,
        "all three shards grant their realms AND System 7 grants the durable avatar before the round-trip",
    );

    // The durable SUBJECT: the avatar's Entity id, read LIVE from the directory (it has a real
    // `OwnerRecord` — the durable discriminator; a transient would have none).
    let (_session, subject, start_fence) = read_subject(&mut topo);
    assert_eq!(
        entity_head_node(&mut topo, subject),
        Some(SHARD),
        "the durable player's directory head starts on System 7 (its login shard)",
    );

    // Plant the SEED-DERIVED containment neighbourhood on EACH shard (the exact geometry production
    // `shard.rs` boots): System 7 → {Universe, Galaxy, System 7, Planet 7}; the Galaxy → {Universe,
    // Galaxy, System 7, System 8} (it OWNS the two systems as children — the sibling-routing shard);
    // System 8 → {Universe, Galaxy, System 8}. NO shard sees a sibling — the crossing routes through
    // the Galaxy parent.
    plant_seed_neighbourhood(&mut topo, SHARD, UNIVERSE_SEED, stub_config().realm);
    plant_seed_neighbourhood(&mut topo, GALAXY, UNIVERSE_SEED, galaxy_stub_config().realm);
    plant_seed_neighbourhood(&mut topo, DEST, UNIVERSE_SEED, dest_stub_config().realm);

    // The Galaxy realm IS the seed forest's between-space (`System(1)`).
    assert_eq!(
        galaxy_stub_config().realm,
        RealmId::System(GALAXY_SEED),
        "the Galaxy realm is the seed forest's between-space (System(1))",
    );

    // Pause the client's input drive — the round-trip's positions are SCRIPTED via
    // `set_shard_subject_offset` (the physical walk is the D-15/D-30 visual line). The session stays
    // live (Active on the gateway), so the durable saga's route swap has a session to follow per hop.
    let node = topo.node_mut(CLIENT).expect("client present");
    node.as_any_mut()
        .expect("clients opt into downcasting")
        .downcast_mut::<ScriptedClient>()
        .expect("the node is a ScriptedClient")
        .pause_input();

    // THE FULL BOTH-WAYS CHAIN, driven on the SAME durable subject. Outbound: 7 → Galaxy → 8.
    // Return: 8 → Galaxy → 7 (the reverse-cross). Each hop moves the dot to a waypoint clearly inside
    // the target container: each system owns the shell the walk forest gives it, centred on its own
    // origin; the Galaxy owns the between-space the two systems sit in, and is itself centred on the
    // galactic origin — 50 is a point out in the gap, not where the galaxy is. The waypoints 0/50/100
    // are `worldgen::walk`'s own, blessed by its doc; the extent they have to fall inside is asserted
    // below rather than quoted.
    let hops = [
        Hop {
            owner: SHARD,
            offset: 50.0,
            expect_head: GALAXY,
            label: "7→Galaxy (escape SOI)",
        },
        Hop {
            owner: GALAXY,
            offset: 100.0,
            expect_head: DEST,
            label: "Galaxy→8 (enter sibling)",
        },
        Hop {
            owner: DEST,
            offset: 50.0,
            expect_head: GALAXY,
            label: "8→Galaxy (return, reverse-cross)",
        },
        Hop {
            owner: GALAXY,
            offset: 0.0,
            expect_head: SHARD,
            label: "Galaxy→7 (home, reverse-cross)",
        },
    ];

    // ANTI-DRIFT, and the reason the prose above states no radius: every waypoint has to lie inside the
    // walk galaxy, or the "gap" hop is happening in a realm no shard here hosts and the chain proves
    // nothing. The extent is read off the SAME forest the three shards were planted with, so a change to
    // the walk geometry shows up here as a failure instead of as a comment that quietly stops being true.
    let galaxy_extent_m = walk_galaxy_extent_m();
    assert!(
        hops.iter().all(|h| h.offset.abs() < galaxy_extent_m),
        "every round-trip waypoint must lie inside the walk galaxy's own extent ({galaxy_extent_m} \
         m) — the between-space hops are only meaningful while the Galaxy still contains them: {:?}",
        hops.iter().map(|h| h.offset).collect::<Vec<_>>(),
    );

    let mut observed = vec![SHARD]; // the origin: System 7 owns the durable head at the start of hop 1
    for hop in &hops {
        // Precondition: the CURRENT owner really holds the head (a REAL re-home from here, not a spawn).
        assert_eq!(
            entity_head_node(&mut topo, subject),
            Some(hop.owner),
            "HOP {}: the durable head is on {:?} before the hop",
            hop.label,
            hop.owner,
        );
        drive_hop(&mut topo, subject, hop);
        observed.push(hop.expect_head);
    }

    // THE HEADLINE: the DURABLE directory head flipped through the full chain BOTH ways. This vector is
    // the reverse-cross proof (it shows the RETURN legs 8→Galaxy→7, not just outbound). Each transition
    // is a REAL, autonomous, saga-driven directory CAS flip of the SAME durable Entity through the
    // correct container — the multi-hop that was IMPOSSIBLE before S3 (2nd hop CutTimeout).
    assert_eq!(
        observed,
        vec![SHARD, GALAXY, DEST, GALAXY, SHARD],
        "the durable head flipped System 7 → Galaxy → System 8 → Galaxy → System 7 (both ways) — ONE \
         durable player's four-hop journey, ZERO CutTimeouts",
    );

    // FINAL STATE: the durable player landed home (System 7 holds the head), and the fence ADVANCED per
    // commit across the four hops (the durable discriminator vs the transient's static-Fence(1)).
    let final_head = entity_head_node(&mut topo, subject).expect("the durable head is recorded");
    assert_eq!(
        final_head, SHARD,
        "the durable player is HOME — System 7 holds the directory head (Galaxy→7, the reverse-cross)",
    );
    let final_fence = with_final_fence(&mut topo, subject);
    assert!(
        final_fence > start_fence,
        "the durable Entity fence ADVANCED across the round-trip ({start_fence:?} → {final_fence:?}) — \
         four real commits, the durable discriminator (a transient rests at a static Fence(1))",
    );

    // No live sagas remain (the last hop tombstoned) and no orchestrator crossing is stuck unresolved.
    assert_eq!(live_sagas(&mut topo), 0, "the round-trip left no live saga");
    let _ = report(&topo.inspect_all(), ORCH); // keep ORCH observable in scope for the reader
    let _ = fabric; // keep the fabric alive for the whole run (the transports hold clones)
}

/// The fence the directory records for the durable subject's `Entity` head (the CAS fence, which
/// ADVANCES per commit). Read straight off the orchestrator's ONE directory.
fn with_final_fence(topo: &mut Topology, subject: EntityId) -> vd_core::Fence {
    let reports = topo.inspect_all();
    report(&reports, ORCH)
        .directory
        .iter()
        .find_map(|(k, r)| {
            (*k == vd_wire::seams::directory::DirectoryKey::Entity(subject)).then_some(r.fence)
        })
        .expect("the durable subject is recorded in the directory with a fence")
}
