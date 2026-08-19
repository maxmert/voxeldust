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
use vd_tests::{GATEWAY, SHARD, p1_client, p1_cluster, walk_forward};

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
    // The DELIVERED position must be read as a TOTAL — whole-number part plus leftover. The integrator
    // now folds the leftover into the whole number every tick (so motion precision stops depending on how
    // far from the origin you are), which means the leftover alone is a sub-millimetre remainder and says
    // nothing about whether the walker moved.
    let travelled = own_pose
        .pos
        .delta_m(vd_core::pose::LatticePos::ORIGIN, vd_core::pose::Tier::Fine);
    assert!(
        travelled.distance(DVec3::ZERO) > 0.5,
        "the walker's delivered pose moved: {:?} (total {:?})",
        own_pose.pos,
        travelled
    );
    let idle = client_view(&mut topo, NodeId(101));
    assert_eq!(idle.poses.len(), 2, "the idle dot sees the walker too");
    let idle_own = idle.own_entity.expect("authority announced");
    assert_eq!(
        idle.poses[&idle_own].pos.offset(),
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

// ---- Density (hundreds-in-one-location) load fixture — named knobs (HR5 no-magic-numbers) ----------

/// The CI density gate: `>= 128` sessions is the audit-mandated floor for a dense-PvP soak claim; 128 is
/// the smallest power-of-two at that floor and puts the fan-out at ~16x the 32-client canary (D-9 is
/// O(N²) bytes/tick) — enough to make the D-9 baseline visible while `just gate` wall-clock stays sane.
const DENSITY_CLIENTS_DEFAULT: u64 = 128;
/// Ticks the crowd walks (same sustained window as the 32-client canary, not a burst).
const DENSITY_TICKS: u64 = 200;
/// The deterministic per-session login/attach handshake before a session's first input applies
/// (per-session, N-invariant on the loss-free fabric) — subtracted from the applied-input ideal.
const LOGIN_RAMP_TICKS: u64 = 10;
/// The applied-input floor as a % of the post-ramp ideal `n·(ticks-ramp)`. The run is DETERMINISTIC
/// (VirtualClock + fixed seed + loss-free fabric ⇒ exactly `n·(ticks-ramp)` applied, measured at N=128),
/// so 95% is a TIGHT floor with only ~5% slack — enough to absorb any per-session ramp jitter at a higher
/// soak N, yet tight enough to CATCH a real "inputs partly stopped flowing" regression (an 80% floor could
/// not). Widen this NAMED const (never a magic literal) only if a higher-N soak proves benign staggering.
const APPLIED_FLOOR_PCT: u64 = 95;
/// Client NodeId base: `100..100+N` — clear of ORCH(1)/GATEWAY(2)/SHARD(3)/DEST(4).
const DENSITY_NODE_BASE: u64 = 100;
/// Compile-time guard (zero runtime cost): the client NodeId base must clear the reserved low ids.
const _: () = assert!(DENSITY_NODE_BASE > 4);
/// Sane upper clamp so a fat-fingered `VD_DENSITY_CLIENTS` (e.g. a stray extra digit) is CAPPED rather
/// than wedging the machine building millions of in-process nodes — the harness soaks a few thousand, max.
const DENSITY_CLIENTS_MAX: u64 = 4096;
/// The GENEROUS, N-scaled wall-clock COLLAPSE detector (NOT a perf gate, NOT a D-9 byte gate): fixed
/// build/harness overhead + a fat per-client allowance absorbing the O(N²) fan-out. VirtualClock ⇒
/// wall-clock is pure CPU work, so a linear-per-client budget catches a quadratic-in-WALL-CLOCK
/// contention collapse WITHOUT ever failing when the (ledgered, P6) D-9 BYTE wall is hit.
const DENSITY_BASE_MS: u64 = 2_000;
const DENSITY_PER_CLIENT_MS: u64 = 200;

/// N for the density run: the named default, RAISABLE (never lowerable — CI always proves `>=` the floor)
/// via `VD_DENSITY_CLIENTS` for a soak. Mirrors the single-env-knob pattern of the other load tests.
fn density_clients() -> u64 {
    std::env::var("VD_DENSITY_CLIENTS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n >= DENSITY_CLIENTS_DEFAULT)
        .map(|n| n.min(DENSITY_CLIENTS_MAX))
        .unwrap_or(DENSITY_CLIENTS_DEFAULT)
}

/// The user-mandated "hundreds in ONE location" density gate (audit P4/P5 precursor). N>=128 concurrent
/// sessions walk for 200 ticks through ONE gateway + ONE shard; EVERY harness invariant holds at scale
/// (authority-unique per tick, input-conservation, authority-settled, no reliable shed, wire truth), and
/// the D-9 whole-realm-broadcast O(N²) byte baseline is REPORTED (never gated — D-9 is ledgered to P6:
/// this fixture MEASURES the wall, it does not fix it). Deterministic (VirtualClock + fixed-seed
/// FaultFabric), CI-runnable, env-RAISABLE for a soak. The tightly-pinned 32-client canary
/// `p1_volume_32_clients_walk_under_invariants` stays as the fast smaller sibling (never deleted).
#[test]
fn p1_volume_dense_hundreds_walk_under_invariants() {
    let n = density_clients();
    let fabric = FaultFabric::new(202, 2);
    let mut topo = p1_cluster(&fabric, n as usize);
    let ids: Vec<NodeId> = (0..n).map(|k| NodeId(DENSITY_NODE_BASE + k)).collect();
    for (k, id) in ids.iter().enumerate() {
        topo.add_node(Box::new(p1_client(
            &fabric,
            *id,
            AccountId(1_000 + k as u128),
            walk_forward(),
        )));
    }

    let started = Instant::now();
    run_checked(&mut topo, DENSITY_TICKS); // AUTHORITY-UNIQUE at every one of the 200 ticks
    let elapsed = started.elapsed();

    let reports = topo.inspect_all();
    let shard = shard_report(&reports);
    // All N avatars spawned.
    assert_eq!(shard.held_entities.len(), n as usize, "all {n} spawned");
    // Sustained input volume >= APPLIED_FLOOR_PCT% of the post-ramp ideal n·(ticks-ramp) — DERIVED from
    // named knobs + scales with N (replaces the 32-canary's fixed `> 4_000` which would under-assert ~5x).
    let applied_floor = n * (DENSITY_TICKS - LOGIN_RAMP_TICKS) * APPLIED_FLOOR_PCT / 100;
    assert!(
        shard.applied_inputs.len() as u64 >= applied_floor,
        "applied inputs {} below the {applied_floor} floor for {n} clients",
        shard.applied_inputs.len()
    );
    // No InputLog eviction ⇒ the applied-input window covers the WHOLE run, so INPUT-CONSERVATION below is
    // trustworthy (an eviction would truncate the log and false-green conservation). Cheap guard with big
    // headroom at N=128 (~24k applied vs the 1M InputLog cap ⇒ ~41x); it only becomes load-bearing at a
    // multi-thousand-client soak — kept because it costs nothing and guards conservation at ANY N.
    assert_eq!(
        shard.input_window_evictions, 0,
        "the InputLog window covered the whole run at {n} clients (conservation is trustworthy)"
    );
    // Every dot SEES the whole N-crowd — the crowd-visibility proof AND the D-9 pressure point (each
    // DeliveredWorldView carries all N poses). `poses` is insert-only over the run, so this asserts
    // "received a pose for all N at least once" — equivalent to "sees all N simultaneously" on the
    // loss-free lockstep fabric (continuous broadcast, no eviction on this path).
    for id in &ids {
        let view = client_view(&mut topo, *id);
        assert_eq!(view.poses.len(), n as usize, "every dot sees all {n} dots");
    }
    // Density overload alarms: no reliable transfer/control/membership frame shed, no node lied about bytes.
    topo.verify_no_reliable_shed()
        .expect("no reliable frame shed under crowd load");
    topo.verify_wire_truth()
        .expect("delivered == claimed per node under crowd load (no wire lie)");
    // INPUT-CONSERVATION + AUTHORITY-SETTLED at rest.
    quiesce_and_verify(&mut topo, &ids);

    // D-9 BASELINE — REPORTED, NEVER gated (documents the whole-realm-broadcast O(N²) fan-out ledgered to
    // P6). NOTE the UNIT: `Stepped.sent` is outbound MESSAGES accepted by the transport, NOT bytes — a
    // message-count PROXY for the O(N²) byte fan-out (each gateway message is ~one MTU-budgeted snapshot
    // datagram). Sum the per-tick counts the harness already records in its trace.
    let (mut gateway_msgs, mut shard_msgs, mut gateway_peak) = (0u64, 0u64, 0u64);
    for ev in topo.trace() {
        if let vd_harness::topology::TraceEvent::Stepped { node, sent, .. } = ev {
            if *node == GATEWAY {
                gateway_msgs += *sent;
                gateway_peak = gateway_peak.max(*sent);
            } else if *node == SHARD {
                shard_msgs += *sent;
            }
        }
    }
    eprintln!(
        "[density] D-9 BASELINE (outbound MESSAGE counts, not bytes) n={n} ticks={DENSITY_TICKS} \
         applied={} gateway_msgs_total={gateway_msgs} gateway_msgs_peak_tick={gateway_peak} \
         shard_msgs_total={shard_msgs} elapsed={elapsed:?}",
        shard.applied_inputs.len()
    );

    // GENEROUS N-scaled COLLAPSE detector (NOT a perf/byte gate).
    let budget = Duration::from_millis(DENSITY_BASE_MS + DENSITY_PER_CLIENT_MS * n);
    assert!(
        elapsed < budget,
        "density run collapsed: {elapsed:?} exceeded {budget:?} for {n} clients"
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
