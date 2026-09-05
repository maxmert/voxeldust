//! THE VANISHED-CLIENT GATE — a login AFTER a kill.
//!
//! A player's window dies. The process is gone, so it never says `Bye`. The gateway still holds that
//! session. The player starts the window again with the SAME name, so the SAME node id and the SAME
//! QUIC port come back. The second login MUST land.
//!
//! Two facts in this crate make it land, and this gate flies both:
//!   1. THE CLIENT STAMPS A REAL LAUNCH. Each client process reports a launch stamp that is strictly
//!      higher than the dead process's, so the transport reads the second window as a NEW process
//!      (`Inbound::PeerReset { cause: Reincarnated }`) and drops the dead one's buffered frames.
//!   2. A CLIENT IS A LEARNED PEER, NEVER A BOOKED ONE. No cluster books a client any more, so the
//!      gateway never DIALS one. A dialed lane would re-dial the second window at the same address
//!      and replay the dead window's retained frames — an old Welcome, old subscriptions — into it.
//!
//! The cluster is the SAME static single-shard cluster `dev_control_nav` flies.
//!
//! Gated on `dev-control`. Run with
//! `cargo test --release -p vd-bins --features dev-control --test login_after_kill`.
#![cfg(feature = "dev-control")]

use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, admin_get_body, common_env, dev_auth_pubkey_hex,
    dev_auth_signing_key_hex, dev_roundtrip, gateway_env, orchestrator_env, reserve_tcp_addr,
    reserve_udp_addr, shard_env,
};
use vd_devproto::{DevPhase, DevRequest, DevResponse, DevState};
use vd_wire::admin::{AdminSnapshot, GatewayView};

/// How long a login may take, DERIVED from the cluster's own numbers — never a bare literal.
///
/// Two things must finish before the second window is Active:
///   1. THE GATEWAY MUST NOTICE THE DEAD ONE. Its lane toward the killed client re-dials with a
///      backoff and declares the peer unreachable only after `confirm_unreachable_after_retries`
///      failed replays, each spaced by at most `DEFAULT_REDIAL_BACKOFF_MAX`.
///   2. THE HOME MUST BE READY. That is one boot budget — `DEV.boot_ticks_p99` ticks at
///      `DEV.tick_hz`, the same number the gateway and the orchestrator wait on.
fn login_deadline() -> Duration {
    let confirm_window = vd_io_prod::mesh::DEFAULT_REDIAL_BACKOFF_MAX
        * vd_io_prod::mesh::DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES;
    // Integer milliseconds: the tick budget times a second, divided by the tick rate.
    let boot_budget = Duration::from_millis(DEV.boot_ticks_p99 * 1_000 / u64::from(DEV.tick_hz));
    confirm_window + boot_budget
}

fn devctl(port: u16, request: &DevRequest) -> Option<DevResponse> {
    dev_roundtrip(port, request).ok()
}

fn poll_state(port: u16) -> Option<DevState> {
    match devctl(port, &DevRequest::State)? {
        DevResponse::State { state } => Some(state),
        _ => None,
    }
}

/// True once the client is Active AND its OWN entity row has been delivered — the point at which
/// the player can see themselves, not merely the point at which the socket opened.
fn active_with_own_row(port: u16) -> bool {
    let Some(state) = poll_state(port) else {
        return false;
    };
    if state.phase != DevPhase::Active {
        return false;
    }
    let Some(own) = state.own_entity.as_deref() else {
        return false;
    };
    state.entities.iter().any(|r| r.entity == own)
}

fn gateway_view(addr: SocketAddr) -> Option<GatewayView> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str::<AdminSnapshot>(&body).ok()?.gateway
}

/// A client process that dies with its guard. `kill` is SIGKILL on unix — no `Bye` is ever sent.
struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn a_killed_client_relogs_in_on_the_same_node_id_and_port() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports.
    let _tier = vd_bins::cluster_tier();

    let addrs = ClusterAddrs {
        orchestrator: reserve_udp_addr(),
        gateway: reserve_udp_addr(),
        shard: reserve_udp_addr(),
        admin: reserve_tcp_addr(),
        gateway_admin: Some(reserve_tcp_addr()),
        ..ClusterAddrs::reserve()
    };
    let gw_admin = addrs.gateway_admin.expect("gateway admin bound above");
    // The ONE client QUIC port both windows bind, in turn — the second window comes back on the
    // address the dead one held, which is exactly what made the old gateway replay into it.
    let client_quic = reserve_udp_addr();
    let devctl_a = reserve_tcp_addr().port();
    let devctl_b = reserve_tcp_addr().port();

    let trust_dir = std::env::temp_dir().join(format!("vd-relogin-{}", std::process::id()));
    let trust = vd_io_prod::trust::ClusterTrust::generate("vd-relogin").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    let orch_store =
        std::env::temp_dir().join(format!("vd-relogin-{}-orch.redb", std::process::id()));
    let _ = std::fs::remove_file(&orch_store);
    let orch_store = orch_store.display().to_string();

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

    // Both windows are the SAME player: the same name and the same agent index, so the same node id.
    let spawn_client = |devctl_port: u16| -> KillOnDrop {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
        for (k, v) in common.iter() {
            cmd.env(k, v);
        }
        cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
        cmd.args([
            "--name",
            "pilot",
            "--agent-index",
            "0",
            "--gateway",
            &addrs.gateway.to_string(),
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

    let await_active = |devctl_port: u16, what: &str| {
        let deadline = login_deadline();
        let started = Instant::now();
        while !active_with_own_row(devctl_port) {
            assert!(
                started.elapsed() < deadline,
                "{what} never became Active with its own row inside {deadline:?}",
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    };

    // ---- window A logs in ----------------------------------------------------
    let mut window_a = spawn_client(devctl_a);
    await_active(devctl_a, "the first window");

    // ---- window A is KILLED: SIGKILL, so no Bye ever reaches the gateway ------
    window_a.0.kill().expect("kill the first window");
    window_a.0.wait().expect("reap the first window");
    drop(window_a);

    // ---- window B: SAME node id, SAME QUIC port, a new dev-control port -------
    let _window_b = spawn_client(devctl_b);
    await_active(devctl_b, "the second window after the kill");

    // ---- the gateway says WHY the dead session went ---------------------------
    // counters land with the gateway half
    let view = gateway_view(gw_admin).expect("the gateway serves its admin snapshot");
    let closed_lost = view.sessions_closed_peer_lost;
    let closed_reincarnated = view.sessions_closed_peer_reincarnated;
    let replaced = view.sessions_replaced_by_relogin;
    assert!(
        closed_lost + closed_reincarnated + replaced >= 1,
        "the gateway must record the dead session ending: peer_lost={closed_lost}, \
         peer_reincarnated={closed_reincarnated}, replaced_by_relogin={replaced}",
    );

    // Window B closes gracefully; its guard reaps it either way.
    let _ = devctl(devctl_b, &DevRequest::Close);

    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&orch_store);
}
