//! S5b — the closed-loop dev-control drivers END-TO-END: a REAL client binary logs into a
//! REAL cluster over localhost QUIC, and `vdctl`-style `WalkTo`/`LookAt` requests drive its
//! own avatar to a world position / facing purely by injecting ordinary Move/Look at the
//! input seam (NO client prediction — the server stays sole authority; the loops steer on the
//! DELIVERED, lagged state). This is the wiring proof: the new `DevEntityRow.orient` field
//! flows end-to-end (look-at reconstructs facing from it), and an unreachable target times out
//! BOUNDED. The convergence MATH is proven exhaustively + at 100% coverage in
//! `vd-client-harness::nav`; this proves the bin loop + cadence gate + orient wire-through.
//!
//! Gated on `dev-control` (drives the client's loopback listener). Run with
//! `cargo test -p vd-bins --features dev-control`.
#![cfg(feature = "dev-control")]

use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, common_env, dev_auth_pubkey_hex, dev_auth_signing_key_hex,
    dev_roundtrip, gateway_env, orchestrator_env, reserve_tcp_addr, reserve_udp_addr, shard_env,
};
use vd_core::glam::{DQuat, DVec3};
use vd_devproto::{DevEntityRow, DevPhase, DevRequest, DevResponse, DevState};

const DEADLINE: Duration = Duration::from_secs(40);

fn devctl(port: u16, request: &DevRequest) -> Option<DevResponse> {
    dev_roundtrip(port, request).ok()
}

fn poll_state(port: u16) -> Option<DevState> {
    match devctl(port, &DevRequest::State)? {
        DevResponse::State { state } => Some(state),
        _ => None,
    }
}

/// The OWN entity's composited row (pos + the new orient), or `None` before it is delivered.
fn own_row(state: &DevState) -> Option<&DevEntityRow> {
    let own = state.own_entity.as_deref()?;
    state.entities.iter().find(|r| r.entity == own)
}

fn own_pos(state: &DevState) -> Option<DVec3> {
    own_row(state).map(|r| DVec3::from_array(r.pos))
}

/// World-forward reconstructed from the delivered orient quat — the exact facing the look-at
/// loop reads. Proves the `orient` field is wired end-to-end (client → wire → DevState → here).
fn own_forward(state: &DevState) -> Option<DVec3> {
    own_row(state).map(|r| {
        let o = r.orient;
        DQuat::from_xyzw(o[0], o[1], o[2], o[3]) * DVec3::NEG_Z
    })
}

#[test]
fn walk_to_and_look_at_converge_then_an_unreachable_target_times_out() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // ---- topology + trust ----------------------------------------------------
    let orch_addr = reserve_udp_addr();
    let gateway_addr = reserve_udp_addr();
    let shard_addr = reserve_udp_addr();
    let admin_addr = reserve_tcp_addr();
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();

    let trust_dir = std::env::temp_dir().join(format!("vd-nav-{}", std::process::id()));
    let trust = vd_io_prod::trust::ClusterTrust::generate("vd-nav").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    let orch_store = std::env::temp_dir().join(format!("vd-nav-{}-orch.redb", std::process::id()));
    let _ = std::fs::remove_file(&orch_store);
    let orch_store = orch_store.display().to_string();

    let addrs = ClusterAddrs {
        orchestrator: orch_addr,
        gateway: gateway_addr,
        shard: shard_addr,
        admin: admin_addr,
        ..ClusterAddrs::reserve()
    };
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    let spawn_node = |bin: &str, node_env: Vec<(&'static str, String)>| -> Child {
        vd_bins::spawn_node(bin, &common, &node_env).expect("spawn node")
    };
    let mut nodes = Cluster::new();
    nodes.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            orchestrator_env(&addrs, &DEV, &orch_store, vd_bins::ClusterShape::Single),
        ),
    );
    nodes.push(
        "vd-gateway",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-gateway"),
            gateway_env(
                &addrs,
                &dev_auth_pubkey_hex(),
                &DEV,
                vd_bins::ClusterShape::Single,
            ),
        ),
    );
    nodes.push(
        "vd-shard",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-shard"),
            shard_env(&addrs, &DEV, vd_bins::ClusterShape::Single),
        ),
    );
    let _nodes = nodes; // RAII: reaps the cluster on test end or panic

    // The client is its own kill-on-drop guard (NOT in the node Cluster) so the test can
    // `try_wait` it to prove a graceful close terminates the process.
    struct KillOnDrop(Child);
    impl Drop for KillOnDrop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    let mut client = {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
        for (k, v) in common.iter() {
            cmd.env(k, v);
        }
        cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
        cmd.args([
            "--name",
            "nav",
            "--agent-index",
            "0",
            "--gateway",
            &gateway_addr.to_string(),
            "--client-quic",
            &client_quic.port().to_string(),
            "--trust-dir",
            &trust_dir.display().to_string(),
            "--dev-control",
            &devctl_port.to_string(),
            "--allow-dev-control",
        ]);
        KillOnDrop(cmd.spawn().expect("spawn client"))
    };

    // ---- wait until Active AND the own row is delivered (so own_pose resolves) ----
    let started = Instant::now();
    let origin = loop {
        if let Some(pos) = poll_state(devctl_port)
            .filter(|s| s.phase == DevPhase::Active)
            .as_ref()
            .and_then(own_pos)
        {
            break pos;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "client never became Active with an own row"
        );
        std::thread::sleep(Duration::from_millis(100));
    };

    // ---- WalkTo: converge to a nearby world target purely via injected Move --------
    // EVERY NUMBER IS ONE SIM STEP, DERIVED (the true-scale restatement): the smallest move a
    // commanded step can make is the FOOT speed for one tick (`move_speed · dt` = 10 m at the
    // DEV cluster) — the geometric throttle map's floor — so a 2 m target with a 0.5 m arrival
    // band is not merely tight, it is UNREACHABLE: every tick overshoots it by 5×, and under
    // the speed law the carried velocity then ramps, widening the oscillation instead of
    // settling. The hop is stated in steps: 40 steps out, an arrival band of 2 steps (above one
    // step so a fixed-magnitude Move cannot oscillate, and above the ~100-150 ms delivered-pose
    // lag), and the standard 4-step brake so the last stretch tapers to the foot speed.
    let step_m = DEV.move_speed * DEV.tick_dt;
    let arrive_epsilon = 2.0 * step_m;
    let walk_target = origin + DVec3::new(40.0 * step_m, 0.0, 0.0);
    let walked = devctl(
        devctl_port,
        &DevRequest::WalkTo {
            target: walk_target.to_array(),
            arrive_epsilon,
            max_ticks: 600,
            max_step_m: 4.0 * step_m,
        },
    )
    .expect("walk response");
    let DevResponse::State { state: walked } = walked else {
        panic!("walk_to did not arrive (expected State): {walked:?}");
    };
    let landed = own_pos(&walked).expect("own pos after walk");
    assert!(
        landed.distance(walk_target) <= arrive_epsilon + 1e-6,
        "walk_to landed {landed:?} within {arrive_epsilon} of {walk_target:?}",
    );
    // It actually MOVED (not a vacuous already-there arrival): the origin was > epsilon away.
    assert!(
        origin.distance(walk_target) > arrive_epsilon,
        "the target must start outside the arrival band to prove motion",
    );

    // ---- LookAt: turn to face a world target; verify via the NEW orient field ------
    let align_epsilon = 0.02;
    let look_target = landed + DVec3::new(0.0, 0.0, -10.0);
    let looked = devctl(
        devctl_port,
        &DevRequest::LookAt {
            target: look_target.to_array(),
            align_epsilon,
            max_ticks: 400,
        },
    )
    .expect("look response");
    let DevResponse::State { state: looked } = looked else {
        panic!("look_at did not align (expected State): {looked:?}");
    };
    // Reconstruct facing from the delivered orient (end-to-end orient wire-through) and confirm
    // it points at the target within the tolerance the driver enforced.
    let forward = own_forward(&looked).expect("own forward after look");
    let want = (look_target - own_pos(&looked).expect("own pos after look")).normalize();
    let angle = forward.angle_between(want);
    assert!(
        angle <= align_epsilon + 1e-6,
        "look_at faced {forward:?} within {align_epsilon} rad of {want:?} (angle {angle})",
    );

    // ---- an UNREACHABLE target within a tiny budget TIMES OUT (bounded, not hung) --
    let t0 = Instant::now();
    let timed_out = devctl(
        devctl_port,
        &DevRequest::WalkTo {
            target: [1.0e9, 0.0, 0.0],
            arrive_epsilon,
            max_ticks: 10,
            max_step_m: 0.0,
        },
    )
    .expect("timeout response");
    assert!(
        matches!(timed_out, DevResponse::Timeout { .. }),
        "an unreachable walk_to reports a bounded Timeout: {timed_out:?}",
    );
    assert!(
        t0.elapsed() < Duration::from_secs(10),
        "the drive was bounded by max_ticks, not hung: {:?}",
        t0.elapsed(),
    );

    // A graceful close terminates the client process (frees the listener/ports).
    let _ = devctl(devctl_port, &DevRequest::Close);
    let exit = Instant::now();
    while client.0.try_wait().expect("try_wait").is_none() {
        assert!(exit.elapsed() < DEADLINE, "client did not exit after close");
        std::thread::sleep(Duration::from_millis(50));
    }

    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&orch_store);
}
