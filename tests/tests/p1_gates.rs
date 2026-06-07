//! The P1 definition-of-done gates (PLAN.md P1 row). PERMANENT — no later phase
//! may regress them.
//!
//! - A scripted dot logs in (real Ed25519 ticket through gateway M0), walks at
//!   20 Hz, and SEES other dots — all on its single connection (target NodeId
//!   asserted constant inside the client).
//! - AUTHORITY-UNIQUE holds at EVERY tick once authority exists.
//! - INPUT-CONSERVATION holds over the whole run (after quiesce).
//! - The run is fully deterministic (two identical runs, identical ground truth).
//! - The volume run (user mandate): many clients, many messages, same invariants,
//!   bounded wall-clock.

use std::time::{Duration, Instant};

use vd_core::glam::DVec3;
use vd_core::{AccountId, NodeId};
use vd_harness::client::{ClientPhase, ScriptedClient};
use vd_harness::fabric::FaultFabric;
use vd_harness::oracle::{
    verify_authority_settled, verify_authority_unique, verify_input_conservation,
};
use vd_harness::topology::{InspectReport, Topology};
use vd_tests::{p1_client, p1_cluster, walk_forward};

/// Drive the topology while checking AUTHORITY-UNIQUE at every committed tick.
fn run_checked(topo: &mut Topology, ticks: u64) {
    for _ in 0..ticks {
        topo.step();
        let reports = topo.inspect_all();
        verify_authority_unique(&reports).expect("AUTHORITY-UNIQUE holds every tick");
    }
}

#[test]
fn p1_dod_two_dots_log_in_walk_and_see_each_other() {
    let fabric = FaultFabric::new(101, 2);
    let mut topo = p1_cluster(&fabric, 8);
    topo.add_node(Box::new(p1_client(
        &fabric,
        NodeId(100),
        AccountId(1000),
        walk_forward(),
    )));
    // The second dot stands still (it must still be SEEN).
    topo.add_node(Box::new(p1_client(
        &fabric,
        NodeId(101),
        AccountId(1001),
        |_| None,
    )));

    run_checked(&mut topo, 60);

    let reports = topo.inspect_all();
    // Both avatars exist on the shard and in the directory (the per-tick oracle
    // already proved exactly-one-owner; here we pin the count).
    let shard_report = shard_report(&reports);
    assert_eq!(shard_report.held_entities.len(), 2, "two dots spawned");

    // The walking client applied inputs; the idle one sent none.
    assert!(
        !shard_report.applied_inputs.is_empty(),
        "the walker's input reached the sim"
    );

    // Client-side wire truth: each client SEES both dots, and the walker's own
    // delivered pose has moved off the origin.
    let walker = client_view(&mut topo, NodeId(100));
    assert_eq!(
        walker.poses.len(),
        2,
        "walker sees itself AND the other dot"
    );
    let own = walker.own_entity.expect("authority announced");
    let own_pose = walker.poses[&own];
    assert!(
        own_pose.pos.distance(DVec3::ZERO) > 0.5,
        "the walker's delivered pose moved: {:?}",
        own_pose.pos
    );
    let idle = client_view(&mut topo, NodeId(101));
    assert_eq!(idle.poses.len(), 2, "the idle dot sees the walker too");
    let idle_own = idle.own_entity.expect("authority announced");
    assert_eq!(
        idle.poses[&idle_own].pos,
        DVec3::ZERO,
        "the idle dot never moved"
    );

    // Quiesce (no new inputs: drain in-flight), then INPUT-CONSERVATION exactly.
    quiesce_and_verify(&mut topo, &[NodeId(100), NodeId(101)]);
}

#[test]
fn p1_dod_runs_are_deterministic_end_to_end() {
    let run = || -> Vec<(NodeId, InspectReport)> {
        let fabric = FaultFabric::new(77, 2);
        let mut topo = p1_cluster(&fabric, 8);
        topo.add_node(Box::new(p1_client(
            &fabric,
            NodeId(100),
            AccountId(1),
            walk_forward(),
        )));
        run_checked(&mut topo, 50);
        topo.inspect_all()
    };
    assert_eq!(
        run(),
        run(),
        "identical seeds produce identical ground truth (sessions, entities, logs)"
    );
}

/// The user-mandated volume/load run: 32 concurrent sessions walking for 200
/// ticks through ONE gateway and ONE shard. Invariants hold at every tick; the
/// whole run stays inside a generous wall-clock property bound (catches
/// accidental O(n²) fan-out or contention collapse, not raw speed).
#[test]
fn p1_volume_32_clients_walk_under_invariants() {
    const CLIENTS: u64 = 32;
    let fabric = FaultFabric::new(202, 2);
    let mut topo = p1_cluster(&fabric, CLIENTS as usize);
    let ids: Vec<NodeId> = (0..CLIENTS).map(|n| NodeId(100 + n)).collect();
    for (n, id) in ids.iter().enumerate() {
        topo.add_node(Box::new(p1_client(
            &fabric,
            *id,
            AccountId(1_000 + n as u128),
            walk_forward(),
        )));
    }

    let started = Instant::now();
    run_checked(&mut topo, 200);
    let elapsed = started.elapsed();

    let reports = topo.inspect_all();
    let shard = shard_report(&reports);
    assert_eq!(
        shard.held_entities.len(),
        CLIENTS as usize,
        "all 32 spawned"
    );
    // Sustained input volume actually flowed (32 clients * most of 200 ticks).
    assert!(
        shard.applied_inputs.len() > 4_000,
        "expected thousands of applied inputs, got {}",
        shard.applied_inputs.len()
    );
    // Every client ended Active and saw the whole crowd.
    for id in &ids {
        let view = client_view(&mut topo, *id);
        assert_eq!(
            view.poses.len(),
            CLIENTS as usize,
            "every dot sees all {CLIENTS} dots"
        );
    }
    quiesce_and_verify(&mut topo, &ids);
    assert!(
        elapsed < Duration::from_secs(10),
        "volume run collapsed: {elapsed:?}"
    );
}

#[test]
fn p1_rejected_login_closes_and_leaves_no_state() {
    let fabric = FaultFabric::new(303, 2);
    let mut topo = p1_cluster(&fabric, 8);
    // A ticket signed by an IMPOSTOR auth service.
    let forged =
        vd_connection_plane::tickets::mint_login(&[0x66; 32], AccountId(9), vd_core::EpochId(1), 1);
    let client = ScriptedClient::new(
        fabric.register(NodeId(100)),
        vd_tests::GATEWAY,
        forged,
        |_| None,
    );
    topo.add_node(Box::new(client));
    run_checked(&mut topo, 20);

    let reports = topo.inspect_all();
    let shard = shard_report(&reports);
    assert_eq!(
        shard.held_entities.len(),
        0,
        "no avatar for a forged ticket"
    );
    let phase = with_client(&mut topo, NodeId(100), |c| c.phase());
    assert_eq!(phase, ClientPhase::Closed);
    let reason = with_client(&mut topo, NodeId(100), |c| {
        c.close_reason().map(str::to_owned)
    });
    assert_eq!(reason.as_deref(), Some("login ticket rejected"));
}

#[test]
fn p1_bye_releases_the_avatar_and_authority_stays_unique() {
    let fabric = FaultFabric::new(404, 2);
    let mut topo = p1_cluster(&fabric, 8);
    topo.add_node(Box::new(p1_client(
        &fabric,
        NodeId(100),
        AccountId(1),
        walk_forward(),
    )));
    run_checked(&mut topo, 30);
    assert_eq!(shard_report(&topo.inspect_all()).held_entities.len(), 1);

    with_client(&mut topo, NodeId(100), ScriptedClient::send_bye);
    run_checked(&mut topo, 10);
    let reports = topo.inspect_all();
    assert_eq!(
        shard_report(&reports).held_entities.len(),
        0,
        "the avatar despawned on logout"
    );
    quiesce_and_verify(&mut topo, &[NodeId(100)]);
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn shard_report(reports: &[(NodeId, InspectReport)]) -> &InspectReport {
    &reports
        .iter()
        .find(|(id, _)| *id == vd_tests::SHARD)
        .expect("shard present")
        .1
}

/// Pause every client, drain in-flight traffic, then assert conservation and
/// settled authority — the end-of-run exactness window.
fn quiesce_and_verify(topo: &mut Topology, clients: &[NodeId]) {
    for id in clients {
        // Closed clients (post-Bye) have nothing to pause but tolerate it.
        with_client(topo, *id, |c| c.pause_input());
    }
    // Two retry windows is ample for all in-flight datagrams on a perfect fabric.
    for _ in 0..6 {
        topo.step();
    }
    let reports = topo.inspect_all();
    verify_input_conservation(&reports).expect("INPUT-CONSERVATION holds");
    verify_authority_settled(&reports).expect("authority settled and unique at rest");
}

/// Borrow one scripted client from the topology.
fn with_client<R>(topo: &mut Topology, id: NodeId, f: impl FnOnce(&mut ScriptedClient) -> R) -> R {
    let node = topo.node_mut(id).expect("client present");
    let client = node
        .as_any_mut()
        .expect("clients opt into downcasting")
        .downcast_mut::<ScriptedClient>()
        .expect("the node is a ScriptedClient");
    f(client)
}

fn client_view(topo: &mut Topology, id: NodeId) -> vd_harness::client::DeliveredWorldView {
    with_client(topo, id, |c| c.view.clone())
}
