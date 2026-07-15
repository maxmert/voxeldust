//! SCALE-1 — the process-tier load/collapse gate: K REAL dev-control client
//! binaries log into one cluster over localhost QUIC concurrently and each must
//! reach a live, receiving, independent session. This is the structural proof that
//! the gateway fan-out + per-client snapshot routing scale to the per-slot client
//! cap (K = `max_clients_per_worktree`) without one client starving another or the
//! sessions collapsing into each other.
//!
//! Gated on `dev-control`: it drives the clients through their loopback listener, so
//! `CARGO_BIN_EXE_client` must be the listener-enabled build. Run with
//! `cargo test -p vd-bins --features dev-control`.
#![cfg(feature = "dev-control")]

use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, admin_get_body, common_env, dev_auth_pubkey_hex,
    dev_auth_signing_key_hex, dev_roundtrip, gateway_env, orchestrator_env, reserve_tcp_addr,
    reserve_udp_addr, shard_env,
};
use vd_core::NodeId;
use vd_devproto::{
    CLIENT_NODE_BASE, DevPhase, DevPortScheme, DevRequest, DevResponse, DevState, WaitField,
    WaitOp, WaitPredicate,
};

const DEADLINE: Duration = Duration::from_secs(40);
/// Floors a healthy client clears in a second or two of 20 Hz — below them is a
/// stall/collapse, not a slow boot.
const SNAPSHOT_FLOOR: u64 = 5;
const INPUT_FLOOR: u64 = 5;

/// One dev-control round-trip — the SHARED `vd_bins::dev_roundtrip` framing (one wire
/// definition for vdctl, this load test, and the render-smoke gate). `None` if the client
/// is not yet accepting (booting) — this test's poll loops tolerate that.
fn devctl(port: u16, request: &DevRequest) -> Option<DevResponse> {
    dev_roundtrip(port, request).ok()
}

fn poll_state(port: u16) -> Option<DevState> {
    match devctl(port, &DevRequest::State)? {
        DevResponse::State { state } => Some(state),
        _ => None,
    }
}

#[test]
fn k_clients_log_in_concurrently_each_live_receiving_and_independent() {
    let k = DevPortScheme::DEFAULT.max_clients_per_worktree;

    // ---- topology + trust ----------------------------------------------------
    let orch_addr = reserve_udp_addr();
    let gateway_addr = reserve_udp_addr();
    let shard_addr = reserve_udp_addr();
    let admin_addr = reserve_tcp_addr();
    // Per-client QUIC bind addr (seeded into the gateway book) + dev-control TCP port.
    let clients: Vec<(u16, SocketAddr, u16)> = (0..k)
        .map(|i| (i, reserve_udp_addr(), reserve_tcp_addr().port()))
        .collect();

    let trust_dir = std::env::temp_dir().join(format!("vd-load-{}", std::process::id()));
    let trust = vd_io_prod::trust::ClusterTrust::generate("vd-load").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    // D-6: the orchestrator's durable Store (temp scratch ⇒ VD_STORE_EPHEMERAL_OK via orchestrator_env).
    let orch_store = std::env::temp_dir().join(format!("vd-load-{}-orch.redb", std::process::id()));
    let _ = std::fs::remove_file(&orch_store);
    let orch_store = orch_store.display().to_string();

    let addrs = ClusterAddrs {
        orchestrator: orch_addr,
        gateway: gateway_addr,
        shard: shard_addr,
        admin: admin_addr,
        orchestrator_probe: reserve_tcp_addr(),
        gateway_probe: reserve_tcp_addr(),
        shard_probe: reserve_tcp_addr(),
        shard_b: reserve_tcp_addr(),
        shard_b_probe: reserve_tcp_addr(),
    };
    let client_book: Vec<(NodeId, SocketAddr)> = clients
        .iter()
        .map(|(i, quic, _)| (NodeId(CLIENT_NODE_BASE + u64::from(*i)), *quic))
        .collect();
    let common = common_env(&trust_dir.display().to_string(), &DEV);

    // ---- spawn the cluster ---------------------------------------------------
    let spawn_node = |bin: &str, node_env: Vec<(&'static str, String)>| -> Child {
        vd_bins::spawn_node(bin, &common, &node_env).expect("spawn node")
    };
    let mut guard = Cluster::new();
    guard.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            orchestrator_env(&addrs, &DEV, &orch_store, false),
        ),
    );
    guard.push(
        "vd-gateway",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-gateway"),
            gateway_env(&addrs, &client_book, &dev_auth_pubkey_hex(), &DEV, false),
        ),
    );
    guard.push(
        "vd-shard",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-shard"),
            shard_env(&addrs, &DEV, false),
        ),
    );

    // ---- spawn K real dev-control clients ------------------------------------
    for (i, quic, devctl_port) in &clients {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
        for (k, v) in common.iter() {
            cmd.env(k, v);
        }
        cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
        cmd.args([
            "--name",
            &format!("load-{i}"),
            "--agent-index",
            &i.to_string(),
            "--gateway",
            &gateway_addr.to_string(),
            "--client-quic",
            &quic.port().to_string(),
            "--trust-dir",
            &trust_dir.display().to_string(),
            "--dev-control",
            &devctl_port.to_string(),
            "--allow-dev-control",
        ]);
        guard.push("client", cmd.spawn().expect("spawn client"));
    }
    let _guard = guard; // RAII: kills every node AND client on test end or panic

    // ---- converge: all K live, receiving, and sending ------------------------
    let ports: Vec<u16> = clients.iter().map(|(_, _, p)| *p).collect();
    let started = Instant::now();
    let states = loop {
        let snapshot: Vec<Option<DevState>> = ports.iter().map(|p| poll_state(*p)).collect();
        let all_ready = snapshot.iter().all(|s| {
            s.as_ref().is_some_and(|st| {
                st.phase == vd_devproto::DevPhase::Active
                    && st.snapshots_applied >= SNAPSHOT_FLOOR
                    && st.sent_input_count >= INPUT_FLOOR
            })
        });
        if all_ready {
            break snapshot.into_iter().map(Option::unwrap).collect::<Vec<_>>();
        }
        assert!(
            started.elapsed() < DEADLINE,
            "K={k} clients did not all converge; last poll: {snapshot:?}",
        );
        std::thread::sleep(Duration::from_millis(100));
    };

    // ---- assert independence + no collapse -----------------------------------
    // Every client owns a DISTINCT session — the N-session independence proof (no
    // shared/aliased session under concurrent login).
    let mut sessions: Vec<&String> = states
        .iter()
        .map(|s| s.session.as_ref().expect("welcomed ⇒ session"))
        .collect();
    sessions.sort();
    sessions.dedup();
    assert_eq!(
        sessions.len() as u16,
        k,
        "each of K clients must hold a distinct session: {states:?}",
    );

    // Each client is live AND clean: snapshots landing, input flowing, ZERO decode /
    // foreign-peer faults (collapse detection — a starved or cross-wired client trips).
    for state in &states {
        assert!(
            state.snapshots_applied >= SNAPSHOT_FLOOR,
            "starved: {state:?}"
        );
        assert!(
            state.sent_input_count >= INPUT_FLOOR,
            "not sending: {state:?}"
        );
        assert_eq!(state.decode_errors, 0, "decode faults: {state:?}");
        assert_eq!(state.foreign_peer_drops, 0, "cross-wired peer: {state:?}");
        assert!(state.own_entity.is_some(), "no authority: {state:?}");
        // Snapshots applied ⇒ the composited view is non-empty (at least its own dot).
        assert!(!state.entities.is_empty(), "renders no entity: {state:?}");
    }

    // The admin endpoint corroborates: K sessions + the realm + K entities, all fenced.
    let directory = admin_directory(admin_addr);
    let session_rows = directory
        .iter()
        .filter(|a| a.as_str() == "gateway:node-2")
        .count();
    assert_eq!(
        session_rows as u16, k,
        "directory shows K sessions: {directory:?}"
    );

    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&orch_store);
}

/// The orchestrator admin directory's `authority` column (what a 2am `curl` shows).
fn admin_directory(addr: SocketAddr) -> Vec<String> {
    let started = Instant::now();
    loop {
        if let Some(value) = admin_get_body(addr, "/admin/snapshot", None)
            .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok())
        {
            return value["directory"]
                .as_array()
                .expect("directory array")
                .iter()
                .map(|e| e["authority"].as_str().expect("authority").to_owned())
                .collect();
        }
        assert!(
            started.elapsed() < DEADLINE,
            "admin endpoint never answered"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn pred(field: WaitField, op: WaitOp, value: u64) -> WaitPredicate {
    WaitPredicate { field, op, value }
}

/// The dev-control wait/close contract end-to-end: a satisfied wait fires, an
/// unsatisfiable wait TIMES OUT BOUNDED (never hangs — the H1 guard), and a
/// graceful `close` terminates the client PROCESS (freeing its listener + ports).
#[test]
fn wait_until_fires_times_out_bounded_and_close_terminates_the_process() {
    let orch_addr = reserve_udp_addr();
    let gateway_addr = reserve_udp_addr();
    let shard_addr = reserve_udp_addr();
    let admin_addr = reserve_tcp_addr();
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();

    let trust_dir = std::env::temp_dir().join(format!("vd-wait-{}", std::process::id()));
    let trust = vd_io_prod::trust::ClusterTrust::generate("vd-wait").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    // D-6: the orchestrator's durable Store (temp scratch ⇒ VD_STORE_EPHEMERAL_OK via orchestrator_env).
    let orch_store = std::env::temp_dir().join(format!("vd-wait-{}-orch.redb", std::process::id()));
    let _ = std::fs::remove_file(&orch_store);
    let orch_store = orch_store.display().to_string();

    let addrs = ClusterAddrs {
        orchestrator: orch_addr,
        gateway: gateway_addr,
        shard: shard_addr,
        admin: admin_addr,
        orchestrator_probe: reserve_tcp_addr(),
        gateway_probe: reserve_tcp_addr(),
        shard_probe: reserve_tcp_addr(),
        shard_b: reserve_tcp_addr(),
        shard_b_probe: reserve_tcp_addr(),
    };
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    let spawn_node = |bin: &str, node_env: Vec<(&'static str, String)>| -> Child {
        vd_bins::spawn_node(bin, &common, &node_env).expect("spawn node")
    };
    let mut nodes = Cluster::new();
    nodes.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            orchestrator_env(&addrs, &DEV, &orch_store, false),
        ),
    );
    nodes.push(
        "vd-gateway",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-gateway"),
            gateway_env(&addrs, &client_book, &dev_auth_pubkey_hex(), &DEV, false),
        ),
    );
    nodes.push(
        "vd-shard",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-shard"),
            shard_env(&addrs, &DEV, false),
        ),
    );
    let _nodes = nodes; // RAII: reaps the cluster on test end or panic

    // The client is its own kill-on-drop guard (NOT in the node Cluster) so the test
    // can `try_wait` it to prove a graceful close terminates the process.
    struct KillOnDrop(Child);
    impl Drop for KillOnDrop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let mut client = KillOnDrop({
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
        for (k, v) in common.iter() {
            cmd.env(k, v);
        }
        cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
        cmd.args([
            "--name",
            "wait",
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
        cmd.spawn().expect("spawn client")
    });

    // Wait until the client is reachable + Active before exercising the waits.
    let started = Instant::now();
    loop {
        if poll_state(devctl_port).is_some_and(|s| s.phase == DevPhase::Active) {
            break;
        }
        assert!(started.elapsed() < DEADLINE, "client never became Active");
        std::thread::sleep(Duration::from_millis(100));
    }

    // 1. A SATISFIED wait returns a State immediately.
    let satisfied = devctl(
        devctl_port,
        &DevRequest::WaitUntil {
            predicate: pred(WaitField::Active, WaitOp::Eq, 1),
            max_ticks: 400,
        },
    )
    .expect("response");
    assert!(
        matches!(satisfied, DevResponse::State { .. }),
        "a satisfied wait fires with State: {satisfied:?}",
    );

    // 2. An UNSATISFIABLE wait times out — BOUNDED by max_ticks, not hung.
    let t0 = Instant::now();
    let timed_out = devctl(
        devctl_port,
        &DevRequest::WaitUntil {
            predicate: pred(WaitField::EntityCount, WaitOp::Eq, 999),
            max_ticks: 20,
        },
    )
    .expect("response");
    let elapsed = t0.elapsed();
    assert!(
        matches!(timed_out, DevResponse::Timeout { .. }),
        "an unsatisfiable wait reports Timeout: {timed_out:?}",
    );
    assert!(
        elapsed < Duration::from_secs(10),
        "the wait was bounded by max_ticks, not hung: {elapsed:?}",
    );

    // 3. A graceful `close` terminates the client PROCESS (the loop breaks on Closed,
    // main returns, the runtime + listener shut down).
    let acked = devctl(devctl_port, &DevRequest::Close).expect("response");
    assert!(
        matches!(acked, DevResponse::Ack),
        "close is acked: {acked:?}"
    );
    let exit_deadline = Instant::now();
    loop {
        if client.0.try_wait().expect("try_wait").is_some() {
            break;
        }
        assert!(
            exit_deadline.elapsed() < DEADLINE,
            "the client did not exit after a graceful close",
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&orch_store);
}
