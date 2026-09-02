//! ★ THE SHIP FLIES, ACROSS TWO REAL SHARDS (D-MOVE-2).
//!
//! A ship shard states what it is DOING; a star-system shard decides where it ends up. The two are
//! separate nodes and the six numbers travel the real wire between them — not a function call.
//!
//! This is the movement contract's own acceptance shape: *"a child states forces in its own frame;
//! the parent applies them; the parent authors the placement."*
//!
//! WHAT IT DELIBERATELY DOES NOT DO. It does not spawn processes. The wire, the codec, the dispatch,
//! the guards, the physics and the authored row are all the shipped ones; only the transport is the
//! in-memory fabric. A process-tier flight is a separate, slower gate.
use vd_core::fence::Fence;
use vd_core::ids::{NodeId, UniverseTick};
use vd_core::pose::{FrameRef, RealmId};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
use vd_harness::fabric::FaultFabric;
use vd_harness::topology::{StaggerPlan, Topology};
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::register_clock_follower;
use vd_sim::capability::{NodeKind, profiles};
use vd_sim::stub::{StubConfig, register_stub_shard};

const SYSTEM_NODE: NodeId = NodeId(11);
const SHIP_NODE: NodeId = NodeId(12);

/// The ship's lineage: a station under a star system. ★ A STATION, NOT A SHIP KIND — see the note in
/// `vd_sim::stub::drive`'s tests: the `Ship` tag exists now, but the world's own generated forest has
/// no ship in it, and this gate builds its forest by hand. What is under test is the LANE, which is
/// generic by construction, so the kind of the child changes nothing it measures.
fn ship_coord() -> RealmCoord {
    RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, 7),
        RealmLevel::new(RealmKindTag::Station, 5),
    ]))
    .expect("a two-level path has a leaf")
}

#[test]
fn a_ship_states_a_push_and_its_parent_authors_where_it_went() {
    let fabric = FaultFabric::new(7, 2);
    let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());

    // ── THE PARENT: a star system, which does the physics for what is inside it. ────────────────
    let mut system = build_app(
        NodeConfig {
            node_id: SYSTEM_NODE,
            kind: NodeKind::Shard(profiles::system().expect("a system profile")),
        },
        fabric.register(SYSTEM_NODE),
    );
    {
        let (world, schedule) = system.parts_mut();
        register_clock_follower(world, schedule);
        register_stub_shard(world, schedule, system_config());
    }
    topo.add_node(Box::new(system));

    // ── THE CHILD: a ship, which can push itself. ──────────────────────────────────────────────
    let mut ship = build_app(
        NodeConfig {
            node_id: SHIP_NODE,
            kind: NodeKind::Shard(profiles::ship().expect("a ship profile")),
        },
        fabric.register(SHIP_NODE),
    );
    {
        let (world, schedule) = ship.parts_mut();
        register_clock_follower(world, schedule);
        register_stub_shard(world, schedule, ship_config());
    }
    topo.add_node(Box::new(ship));

    // ── ARM BOTH SIDES with what the live cluster resolves for itself. ─────────────────────────
    // A directory grant and a parent route are the orchestrator's business; this gate is about the
    // MOVEMENT lane, so it states them and measures nothing about how they were reached.
    // THE CLOCK. Every authoring system is gated on a synced clock, because a shard that does not
    // know the universe tick must not author anything (D-Finding-1). A live cluster gets this from
    // the orchestrator; this gate states it, because what is under test is the movement lane.
    for node in [SHIP_NODE, SYSTEM_NODE] {
        let world = world_of(&mut topo, node);
        let mut clock = world.resource_mut::<vd_sim::runtime::ClockSample>();
        clock.synced = true;
    }
    world_of(&mut topo, SHIP_NODE)
        .insert_resource(vd_sim::stub::ParentRealmNode(Some(SYSTEM_NODE)));
    world_of(&mut topo, SHIP_NODE).insert_resource(vd_sim::stub::RealmAuthority(Some(Fence(3))));
    // THE PILOT AT THE CONTROLS, holding the stick full forward. A pilot's keys already arrive at
    // whichever shard holds them, so this is simply a pilot who is aboard.
    {
        let world = world_of(&mut topo, SHIP_NODE);
        let mut dots = world.resource_mut::<vd_sim::stub::Dots>();
        let mut dot = pilot();
        dot.last_stick = Some(([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
        dots.0.insert(vd_core::ids::SessionId(1), dot);
    }
    // The parent must know which node speaks for that child — the attestation every up-lane checks.
    {
        let world = world_of(&mut topo, SYSTEM_NODE);
        world
            .resource_mut::<vd_sim::stub::ChildRealmNodes>()
            .0
            .insert(ship_coord().lowered(), SHIP_NODE);
        world.insert_resource(vd_sim::stub::RealmAuthority(Some(Fence(3))));
    }

    // ── FLY IT. ────────────────────────────────────────────────────────────────────────────────
    for _ in 0..40 {
        topo.step();
    }

    // ★ THE PARENT RECEIVED THE PUSH. Nothing here reached into the ship and read its intent: the six
    // numbers crossed the wire, were decoded, passed four guards and were held.
    let (sys_received, sys_misrouted, sys_unattested, sys_uncapable) = {
        let s = world_of(&mut topo, SYSTEM_NODE).resource::<vd_sim::stub::StubStats>();
        (
            s.child_drive_received,
            s.child_drive_misrouted,
            s.child_drive_unattested,
            s.child_drive_uncapable,
        )
    };
    let sent = world_of(&mut topo, SHIP_NODE)
        .resource::<vd_sim::stub::StubStats>()
        .child_drive_sent;
    assert!(sent > 0, "the ship stated its push: sent {sent}");
    assert!(
        sys_received > 0,
        "the star system admitted it: received {sys_received}, misrouted {sys_misrouted}, \
         unattested {sys_unattested}, uncapable {sys_uncapable}"
    );
}

/// One node's world, by id.
fn world_of(topo: &mut Topology, node: NodeId) -> &mut bevy_ecs::world::World {
    topo.node_mut(node)
        .expect("the node is in the rig")
        .as_any_mut()
        .expect("downcast")
        .downcast_mut::<vd_node::ShardNode<vd_harness::fabric::FabricTransport>>()
        .expect("a shard node")
        .world_mut()
}

/// A pilot aboard — the minimum a dot needs to exist and hold a stick.
fn pilot() -> vd_sim::stub::Dot {
    let pos =
        vd_core::pose::LatticePos::at(vd_core::glam::I64Vec3::ZERO, vd_core::glam::DVec3::ZERO);
    vd_sim::stub::Dot {
        last_stick: None,
        entity: vd_core::ids::EntityId::pack(vd_core::entity_kind::EntityKind::Player, 1, 1, 0),
        account: vd_core::ids::AccountId(1),
        session_fence: Fence(1),
        gateway: NodeId(2),
        granted: true,
        input_active: true,
        adopting: false,
        authority: vd_sim::authority::Authority::Owned { fence: Fence(1) },
        departing: false,
        entity_fence: Fence(1),
        pose: vd_core::pose::StampedPose {
            frame: FrameRef::StationLocal { station_seed: 5 },
            pos,
            vel: vd_core::glam::DVec3::ZERO,
            orient: vd_core::glam::DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        },
        yaw: 0.0,
        pitch: 0.0,
        last_applied_seq: None,
        prev_offset: pos,
    }
}

fn system_config() -> StubConfig {
    StubConfig {
        realm: RealmId::System(7),
        own_coord: StubConfig::root_coord(RealmId::System(7)),
        held_realms: StubConfig::single_realm(RealmId::System(7)),
        frame: FrameRef::SystemSpace { system_seed: 7 },
        ..vd_tests::stub_config()
    }
}

fn ship_config() -> StubConfig {
    StubConfig {
        realm: ship_coord().lowered(),
        own_coord: ship_coord(),
        held_realms: StubConfig::single_realm(ship_coord().lowered()),
        frame: FrameRef::StationLocal { station_seed: 5 },
        ..vd_tests::stub_config()
    }
}
