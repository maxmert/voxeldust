//! ★ THE HULL LEAVES ITS STAR SYSTEM AND ENTERS THE NEXT ONE, ACROSS REAL SHARDS (the ruler switch,
//! `docs/design/realm_crossing_plan_2026-09-02.md` slices 0–5, the G-IDENTICAL acceptance).
//!
//! Six shards: the orchestrator, the gateway, System 7 and System 8 under one galaxy, and the hull's
//! own shard. A pilot aboard holds the stick forward. The hull states its push to whoever authors its
//! placement; System 7 integrates it and carries the hull out of its own shell; System 7's swept verdict
//! asks the orchestrator to move the hull's EXTERIOR to the galaxy; the one saga flushes, commits,
//! adopts, releases and promotes; the galaxy tells the hull its new lineage; the hull's next push
//! reaches the GALAXY. Then the hull flies on into System 8's shell and the same machinery hands it
//! DOWN, and a star system adopts it exactly as the galaxy did.
//!
//! WHAT IS MEASURED. Three things, none of them argued:
//! - THE IDENTICAL SIX NUMBERS: the drive System 7 held for the hull before the switch is the drive
//!   the galaxy holds after it, and the drive System 8 holds after the second switch — the same six
//!   integers, so two parent KINDS (a galaxy, a star system) adopt a hull byte-for-byte alike (HR4).
//! - NO GAP: on every tick of both crossings at least one shard holds the hull as a driven child. The
//!   old parent keeps its frozen row until the release and the new parent holds it from the envelope,
//!   so an observer's picture never lacks the hull (the ghost-row concern, plan §7 ask 7, measured at
//!   the authors).
//! - THE HULL'S OWN COORD moves with it: `Galaxy / hull` after the first switch, `Galaxy / System 8 /
//!   hull` after the second — the one datum a parent states to an adopted child (`LineageStated`).
//!
//! WHAT IT DELIBERATELY DOES NOT DO. It spawns no processes: the wire, the codec, the dispatch, the
//! guards, the saga, the physics and the authored rows are the shipped ones; only the transport is
//! the in-memory fabric, so the peer book (a process concern) is not under test here. The gateway's
//! chain splice for a session aboard is pinned by the gateway's own tests; the flight on the cluster
//! is the gate for the whole.
use vd_core::built::{BlueprintId, BuiltBody, BuiltFacts};
use vd_core::fence::Fence;
use vd_core::geometry::{AoiConfig, Boundary, ContainmentBand, ParentCentre, RealmRegion};
use vd_core::glam::{DQuat, DVec3};
use vd_core::ids::{AccountId, NodeId, SessionId, UniverseTick};
use vd_core::pose::{FrameRef, LatticePos, RealmId, Tier, frame_for_realm};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmKindTag, RealmLevel};
use vd_core::worldgen::level_of;
use vd_harness::fabric::FaultFabric;
use vd_harness::topology::Topology;
use vd_sim::stub::StubStats;
use vd_tests::{
    DEST, GALAXY, GALAXY_SEED, HULL_NODE, SHARD, galaxy_coord, hull_crossing_cluster, hull_entity,
    hull_realm,
};

/// System 7 sits at the galaxy's centre; System 8 sits 50 km down the hull's line of flight (the
/// pilot pushes along the nose, and the nose points down −z in the hull's own frame).
const SYSTEM_8_CENTRE_M: DVec3 = DVec3::new(0.0, 0.0, -50_000.0);
/// A star system's shell in this fixture; the hull's berth is 500 m from System 7's centre.
const SYSTEM_SHELL_M: f64 = 1_000.0;
const BERTH_M: DVec3 = DVec3::new(500.0, 0.0, 0.0);
const HULL_SHELL_M: f64 = 20.0;
/// The hull's rated push: 100 m/s² (stated in micro-units on its row), so it clears System 7's
/// shell in about five seconds and reaches System 8 in about half a minute of universe time.
const MAX_PUSH_MICRO_MPS2: i64 = 100_000_000;

fn galaxy() -> RealmId {
    RealmId::Galaxy(GALAXY_SEED)
}

fn system_coord(seed: u64) -> RealmCoord {
    galaxy_coord().child(RealmLevel::new(RealmKindTag::System, seed))
}

fn band() -> ContainmentBand {
    ContainmentBand::for_containment_velocity_safe(50.0, 100.0, 1.0, 0.05, 1.0)
        .expect("valid containment band")
}

/// One shell region: `centre_m` is in the PARENT's frame at the parent's tier (the unit trap the
/// memory names), the region's own frame is its realm's.
fn region(
    realm: RealmId,
    parent: Option<RealmId>,
    centre_m: DVec3,
    r: f64,
    parent_tier: Tier,
) -> RealmRegion {
    RealmRegion {
        realm,
        center: ParentCentre::authored(LatticePos::from_metres(centre_m, parent_tier)),
        frame: match realm {
            RealmId::Ship(ship) => FrameRef::ShipLocal { ship },
            other => frame_for_realm(other, None).expect("a seeded realm resolves a frame"),
        },
        shape: Boundary::Shell { r },
        look: Some(Boundary::Shell { r }),
        band: band(),
        // LIVE interest bands, as THE world's regions carry: a shard reads its parent's head (and its
        // children's) only on the area-of-interest cadence, and that cadence runs only for a forest
        // whose bands are live. Inert bands would leave the hull with no parent to state its push to.
        aoi: AoiConfig::for_velocity_safe(r, 2.0, 3.0, 100.0, 0.05, 20, 0.0)
            .expect("a live interest band"),
        parent,
    }
}

fn world_of(topo: &mut Topology, node: NodeId) -> &mut bevy_ecs::world::World {
    topo.node_mut(node)
        .expect("the node is in the rig")
        .as_any_mut()
        .expect("downcast")
        .downcast_mut::<vd_node::ShardNode<vd_harness::fabric::FabricTransport>>()
        .expect("a shard node")
        .world_mut()
}

fn stats(topo: &mut Topology, node: NodeId) -> StubStats {
    world_of(topo, node).resource::<StubStats>().clone()
}

/// Which shards hold the hull as a driven child right now.
fn holders(topo: &mut Topology) -> Vec<NodeId> {
    [SHARD, GALAXY, DEST]
        .into_iter()
        .filter(|node| {
            world_of(topo, *node)
                .resource::<vd_sim::stub::drive::DrivenChildren>()
                .0
                .contains_key(&hull_realm())
        })
        .collect()
}

fn held_drive(topo: &mut Topology, node: NodeId) -> Option<([i64; 3], [i64; 3])> {
    world_of(topo, node)
        .resource::<vd_sim::stub::drive::DrivenChildren>()
        .0
        .get(&hull_realm())
        .map(|c| c.drive)
}

/// Step until `done`, asserting on EVERY tick that somebody holds the hull; fails loud at `max`.
fn step_until(
    topo: &mut Topology,
    max: usize,
    what: &str,
    mut done: impl FnMut(&mut Topology) -> bool,
) -> usize {
    for i in 0..max {
        topo.step();
        let held_by = holders(topo);
        assert!(
            !held_by.is_empty(),
            "NO GAP: nobody held the hull on step {i} while waiting for {what}"
        );
        if done(topo) {
            return i + 1;
        }
    }
    let s7 = stats(topo, SHARD);
    let g = stats(topo, GALAXY);
    let s8 = stats(topo, DEST);
    let h = stats(topo, HULL_NODE);
    let hull_world = world_of(topo, HULL_NODE);
    let hull_authority = hull_world.resource::<vd_sim::stub::RealmAuthority>().0;
    let hull_parent = hull_world.resource::<vd_sim::stub::ParentRealmNode>().0;
    let hull_synced = hull_world.resource::<vd_sim::runtime::ClockSample>().synced;
    let s7_exterior = world_of(topo, SHARD)
        .resource::<vd_sim::stub::ExteriorAuthority>()
        .0
        .clone();
    let s7_authority = world_of(topo, SHARD)
        .resource::<vd_sim::stub::RealmAuthority>()
        .0;
    let orch = {
        let w = world_of(topo, vd_tests::ORCH);
        let r = w.resource::<vd_node::saga_runtime::SagaRuntimeRes>();
        (
            r.exterior_crossings_started(),
            r.exterior_request_unattested(),
            r.crossing_unresolved(),
            r.crossing_subject_gone(),
            r.views().len(),
        )
    };
    let g_more = (
        g.exterior_flush_unheld,
        g.exterior_flush_unplaceable,
        g.placement_book_miss,
        g.flush_anchor_not_own,
    );
    panic!(
        "{what} did not happen within {max} steps.\n\
         orchestrator: started/unattested/unresolved/gone/live sagas {orch:?}; \
         galaxy flush unheld/unplaceable/book_miss/anchor_not_own {g_more:?}\n\
         hull: authority {hull_authority:?} parent {hull_parent:?} synced {hull_synced}; \
         System 7: authority {s7_authority:?} exterior leases {s7_exterior:?}\n\
         System 7: decided {} requested {} flushed {} released {} drive_received {}\n\
         galaxy:   adopted {} promotes {} released {} decided {} requested {} drive_received {} \
         arrival_refused {} lineage_stated {}\n\
         System 8: adopted {} promotes {} drive_received {} lineage_stated {}\n\
         hull:     drive_sent {} lineage_applied {} held_unattested {} misrouted {} discarded {}",
        s7.exterior_crossings_decided,
        s7.exterior_crossings_requested,
        s7.exterior_flushed,
        s7.exterior_released,
        s7.child_drive_received,
        g.exterior_adopted,
        g.exterior_promotes,
        g.exterior_released,
        g.exterior_crossings_decided,
        g.exterior_crossings_requested,
        g.child_drive_received,
        g.exterior_arrival_refused,
        g.lineage_stated,
        s8.exterior_adopted,
        s8.exterior_promotes,
        s8.child_drive_received,
        s8.lineage_stated,
        h.child_drive_sent,
        h.lineage_applied,
        h.lineage_held_unattested,
        h.lineage_misrouted,
        h.lineage_discarded,
    );
}

/// A pilot aboard, holding the stick full forward along the nose.
fn pilot() -> vd_sim::stub::Dot {
    let pos = LatticePos::at(vd_core::glam::I64Vec3::ZERO, DVec3::ZERO);
    vd_sim::stub::Dot {
        last_stick: Some((DVec3::new(0.0, 0.0, -1.0), [0.0, 0.0, 0.0])),
        entity: vd_core::ids::EntityId::pack(vd_core::entity_kind::EntityKind::Player, 1, 1, 0),
        account: AccountId(1),
        session_fence: Fence(1),
        gateway: NodeId(2),
        granted: true,
        input_active: true,
        adopting: false,
        authority: vd_sim::authority::Authority::Owned { fence: Fence(1) },
        departing: false,
        entity_fence: Fence(1),
        pose: vd_core::pose::StampedPose {
            frame: FrameRef::ShipLocal {
                ship: hull_entity(),
            },
            pos,
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        },
        yaw: 0.0,
        pitch: 0.0,
        last_applied_seq: None,
        look_extent_m: vd_core::look::OCCUPANT_FIGURE_EXTENT_M,
        prev_offset: pos,
    }
}

fn hull_body() -> BuiltBody {
    BuiltBody {
        realm: hull_realm(),
        owner: AccountId(1),
        blueprint: BlueprintId(1),
        bound: Boundary::Shell { r: HULL_SHELL_M },
        look: Boundary::Shell { r: HULL_SHELL_M },
        facts: BuiltFacts {
            mass_g: 1_000_000_000,
            cross_section_mm2: 400_000_000,
            drag_micro: 0,
            max_push_micro_mps2: MAX_PUSH_MICRO_MPS2,
            max_turn_micro_radps2: 1_000_000,
        },
        fence: Fence(1),
    }
}

/// Plant the forest each shard evaluates, the berth on System 7, the body and the pilot on the hull.
fn plant(topo: &mut Topology) {
    let hull = hull_realm();
    let galaxy_root = region(galaxy(), None, DVec3::ZERO, 1.0e12, Tier::Galaxy);
    let system_7 = region(
        RealmId::System(7),
        Some(galaxy()),
        DVec3::ZERO,
        SYSTEM_SHELL_M,
        Tier::Galaxy,
    );
    let system_8 = region(
        RealmId::System(8),
        Some(galaxy()),
        SYSTEM_8_CENTRE_M,
        SYSTEM_SHELL_M,
        Tier::Galaxy,
    );
    let berth = region(
        hull,
        Some(RealmId::System(7)),
        BERTH_M,
        HULL_SHELL_M,
        Tier::Fine,
    );
    // System 7: its parent, itself, and the berthed hull.
    {
        let world = world_of(topo, SHARD);
        *world.resource_mut::<vd_sim::stub::RealmRegions>() =
            vd_sim::stub::RealmRegions::new(vec![galaxy_root, system_7, berth])
                .with_own_realm(RealmId::System(7));
        world
            .resource_mut::<vd_sim::stub::drive::DrivenChildren>()
            .0
            .insert(
                hull,
                vd_sim::stub::drive::DrivenChild {
                    // The travel since the berth, not the berth itself: a parent's pose for a driven
                    // child is the berth centre PLUS this, so a plant that repeats the berth here
                    // doubles it (measured: the hull ran a tangent to System 8's shell at x = 1000).
                    state: vd_sim::stub::drive::DrivenState {
                        pos_m: DVec3::ZERO,
                        vel_mps: DVec3::ZERO,
                        orient: DQuat::IDENTITY,
                        spin_radps: DVec3::ZERO,
                    },
                    drive: ([0; 3], [0; 3]),
                    drive_at: (Fence::GENESIS, UniverseTick(0)),
                    facts: None,
                    facts_at: (Fence::GENESIS, UniverseTick(0)),
                    frozen: false,
                },
            );
        // The berth's node, as the boot loader of a berth would learn it from the directory.
        world
            .resource_mut::<vd_sim::stub::ChildRealmNodes>()
            .0
            .insert(hull, HULL_NODE);
    }
    // The galaxy: itself and its two systems.
    {
        let world = world_of(topo, GALAXY);
        *world.resource_mut::<vd_sim::stub::RealmRegions>() =
            vd_sim::stub::RealmRegions::new(vec![galaxy_root, system_7, system_8])
                .with_own_realm(galaxy());
    }
    // System 8: its parent and itself.
    {
        let world = world_of(topo, DEST);
        *world.resource_mut::<vd_sim::stub::RealmRegions>() =
            vd_sim::stub::RealmRegions::new(vec![galaxy_root, system_8])
                .with_own_realm(RealmId::System(8));
    }
    // The hull: the realm it was berthed in, itself, its body, and the pilot at the stick.
    {
        let world = world_of(topo, HULL_NODE);
        *world.resource_mut::<vd_sim::stub::RealmRegions>() =
            vd_sim::stub::RealmRegions::new(vec![
                region(
                    RealmId::System(7),
                    None,
                    DVec3::ZERO,
                    SYSTEM_SHELL_M,
                    Tier::Galaxy,
                ),
                berth,
            ])
            .with_own_realm(hull);
        world.insert_resource(vd_sim::stub::drive::OwnBody(Some(hull_body())));
        world
            .resource_mut::<vd_sim::stub::Dots>()
            .0
            .insert(SessionId(1), pilot());
    }
}

#[test]
fn a_hull_leaves_its_star_system_for_the_galaxy_and_enters_the_next_system_with_its_six_numbers_intact()
 {
    let fabric = FaultFabric::new(7, 2);
    let mut topo = hull_crossing_cluster(&fabric, 4);
    plant(&mut topo);
    let hull = hull_realm();

    // ── 1. THE BERTH. The hull's push reaches System 7 and System 7 holds the six numbers. ─────
    let booted = step_until(&mut topo, 400, "System 7 to admit the hull's push", |t| {
        stats(t, SHARD).child_drive_received > 0
    });
    let sys7_drive = held_drive(&mut topo, SHARD).expect("System 7 holds the hull");
    assert_ne!(sys7_drive, ([0; 3], [0; 3]), "a real push was held");
    assert_eq!(
        stats(&mut topo, SHARD).child_drive_uncapable,
        0,
        "a star system integrates its children"
    );

    // ── 2. OUT. The hull clears System 7's shell; the exterior goes up to the galaxy. ──────────
    let out = step_until(&mut topo, 3_000, "the galaxy to promote the hull", |t| {
        stats(t, GALAXY).exterior_promotes >= 1
    });
    let s7 = stats(&mut topo, SHARD);
    let g = stats(&mut topo, GALAXY);
    assert_eq!(
        s7.exterior_crossings_decided, 1,
        "System 7 decided the exit once"
    );
    assert_eq!(s7.exterior_flushed, 1);
    assert_eq!(
        s7.exterior_released, 1,
        "System 7 let the hull go at the demote"
    );
    assert_eq!(
        g.exterior_adopted, 1,
        "the galaxy adopted the hull at the envelope"
    );
    assert_eq!(g.exterior_arrival_refused, 0);
    assert_eq!(
        holders(&mut topo),
        vec![GALAXY],
        "after the release only the galaxy holds it"
    );
    let adopted = *world_of(&mut topo, GALAXY)
        .resource::<vd_sim::stub::RealmRegions>()
        .direct_child(galaxy(), hull)
        .expect("the hull is on the galaxy's roster");
    assert_eq!(adopted.shape, Boundary::Shell { r: HULL_SHELL_M });

    // ── 3. THE HULL IS TOLD, and its next push reaches the GALAXY with the same six numbers. ───
    let told = step_until(&mut topo, 600, "the galaxy to admit the hull's push", |t| {
        stats(t, GALAXY).child_drive_received > 0
    });
    let h = stats(&mut topo, HULL_NODE);
    assert!(h.lineage_applied >= 1, "the hull applied its new lineage");
    assert_eq!(h.lineage_misrouted, 0);
    assert_eq!(h.lineage_discarded, 0);
    assert_eq!(
        world_of(&mut topo, HULL_NODE)
            .resource::<vd_sim::stub::StubConfig>()
            .own_coord,
        galaxy_coord().child(level_of(hull)),
        "the hull's own coord is the galaxy's plus its level"
    );
    assert_eq!(
        held_drive(&mut topo, GALAXY),
        Some(sys7_drive),
        "THE IDENTICAL SIX NUMBERS reached the galaxy"
    );
    assert_eq!(stats(&mut topo, GALAXY).child_drive_uncapable, 0);

    // ── 4. IN. The hull flies on into System 8's shell; the galaxy hands it DOWN. ──────────────
    let down = step_until(&mut topo, 6_000, "System 8 to promote the hull", |t| {
        stats(t, DEST).exterior_promotes >= 1
    });
    let g = stats(&mut topo, GALAXY);
    let s8 = stats(&mut topo, DEST);
    assert_eq!(
        g.exterior_crossings_decided, 1,
        "the galaxy decided the entry once"
    );
    assert_eq!(g.exterior_flushed, 1);
    assert_eq!(g.exterior_released, 1);
    assert_eq!(
        s8.exterior_adopted, 1,
        "System 8 adopted the hull at the envelope"
    );
    assert_eq!(s8.exterior_arrival_refused, 0);
    assert_eq!(holders(&mut topo), vec![DEST]);
    let adopted = *world_of(&mut topo, DEST)
        .resource::<vd_sim::stub::RealmRegions>()
        .direct_child(RealmId::System(8), hull)
        .expect("the hull is on System 8's roster");
    assert_eq!(adopted.shape, Boundary::Shell { r: HULL_SHELL_M });

    // ── 5. TOLD AGAIN: a star system adopts exactly as the galaxy did (HR4, two parent kinds). ─
    let told_again = step_until(&mut topo, 600, "System 8 to admit the hull's push", |t| {
        stats(t, DEST).child_drive_received > 0
    });
    assert_eq!(
        world_of(&mut topo, HULL_NODE)
            .resource::<vd_sim::stub::StubConfig>()
            .own_coord,
        system_coord(8).child(level_of(hull)),
        "the hull's own coord is System 8's plus its level"
    );
    assert_eq!(
        held_drive(&mut topo, DEST),
        Some(sys7_drive),
        "THE IDENTICAL SIX NUMBERS reached System 8"
    );
    let h = stats(&mut topo, HULL_NODE);
    assert!(h.lineage_applied >= 2);
    assert_eq!(h.lineage_misrouted, 0);
    assert_eq!(h.lineage_discarded, 0);
    eprintln!(
        "ruler switch: berth {booted} steps, out {out}, told {told}, down {down}, told again {told_again}"
    );
}
