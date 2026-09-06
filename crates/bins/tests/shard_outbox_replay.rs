//! THE REAL SHARD'S BOOT REPLAY — a message survives the shard that owed it (foundation slice 4).
//!
//! A shard says some things exactly once. "This occupant left your interest" is one of them: nobody
//! re-says it, because the occupant is still alive and simply out of one observer's hold. If the shard
//! dies between saying it and hearing the acknowledgement, the notice is gone and a figure stays frozen
//! on that player's screen forever.
//!
//! The durable outbox is the cure: the frame is on disk BEFORE it leaves, and the next boot re-drives it.
//! `outbox_sigkill_restart` proves that machinery on a purpose-built test node. THIS gate proves it on the
//! SHIPPED PATH — the real `vd-shard` binary, booted by the same `shard_env` every dev cluster and every
//! process gate boots, reaching the real gateway over real QUIC.
//!
//! HOW THE ROW GETS THERE. The test plants the row the way the shard itself would: a real mesh, bound as
//! the shard's own node id, sends ONE `Retained` control frame toward the gateway while the gateway is not
//! yet running. The send stages and fsyncs the outbox row BEFORE it dials (the durable-before-send gate),
//! the dial then fails, and the row stays retained. No test-only seam is used, so the row on disk is the
//! row a live shard would have written.
//!
//! WHAT IS OBSERVED. The notice names a session the gateway never minted, so the gateway counts it as
//! `interest_recipient_unsubscribed` — a counter that can move for no other reason in this cluster, and
//! one an ordinary login never touches. The twin boots the SAME cluster on an EMPTY outbox file and holds
//! the counter at zero over the same wait, so the positive result cannot come from anything else.
//!
//! Gated on `dev-control` (the client the last step logs in). Run with
//! `cargo test --release -p vd-bins --features dev-control --test shard_outbox_replay`.
#![cfg(feature = "dev-control")]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, GATEWAY, SHARD, common_env, dev_auth_pubkey_hex,
    dev_auth_signing_key_hex, dev_roundtrip, gateway_env, orchestrator_env, reserve_tcp_addr,
    reserve_udp_addr, shard_env,
};
use vd_core::{EntityId, EpochId, Fence, SessionId, UniverseTick};
use vd_devproto::{DevPhase, DevRequest, DevResponse, DevState};
use vd_io_prod::outbox::{NodeOutbox, OutboxSink, SharedOutbox};
use vd_io_prod::runtime::EnvConfig;
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Durability, MsgClass, Transport};
use vd_wire::admin::{AdminSnapshot, GatewayView};
use vd_wire::session_flow::ShardToGateway;

/// How long the replayed notice may take to reach the gateway: one boot budget for the shard (the same
/// number the gateway and the orchestrator wait on) plus the mesh's own confirm window, because the shard
/// replays the row while the gateway may still be booting and the retransmit timer re-drives it.
fn replay_deadline() -> Duration {
    let confirm_window = vd_io_prod::mesh::DEFAULT_REDIAL_BACKOFF_MAX
        * vd_io_prod::mesh::DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES;
    let boot_budget = Duration::from_millis(DEV.boot_ticks_p99 * 1_000 / u64::from(DEV.tick_hz));
    confirm_window + boot_budget
}

/// THE SESSION THE GATEWAY NEVER MINTED. The notice names it, so the gateway can only count the delivery
/// as one for an unsubscribed recipient — which is the observation this gate reads.
const ABSENT_SESSION: SessionId = SessionId(424_242);

/// The addresses + files one run of the cluster owns.
struct Rig {
    addrs: ClusterAddrs,
    gateway_admin: SocketAddr,
    trust_dir: PathBuf,
    work_dir: PathBuf,
    orch_store: String,
    trust: ClusterTrust,
}

impl Rig {
    fn new(tag: &'static str) -> Rig {
        let addrs = ClusterAddrs {
            orchestrator: reserve_udp_addr(),
            gateway: reserve_udp_addr(),
            shard: reserve_udp_addr(),
            admin: reserve_tcp_addr(),
            gateway_admin: Some(reserve_tcp_addr()),
            ..ClusterAddrs::reserve()
        };
        let gateway_admin = addrs.gateway_admin.expect("bound above");
        let trust_dir = std::env::temp_dir().join(format!("{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&trust_dir);
        let trust = ClusterTrust::generate(tag).expect("trust");
        trust.write_der_dir(&trust_dir).expect("trust dir");
        let orch_store =
            std::env::temp_dir().join(format!("{tag}-{}-orch.redb", std::process::id()));
        let _ = std::fs::remove_file(&orch_store);
        Rig {
            addrs,
            gateway_admin,
            trust_dir,
            work_dir: vd_bins::fresh_work_dir(tag),
            orch_store: orch_store.display().to_string(),
            trust,
        }
    }

    fn common(&self) -> Vec<(&'static str, String)> {
        common_env(&self.trust_dir.display().to_string(), &DEV)
    }

    fn shard_env(&self) -> Vec<(&'static str, String)> {
        shard_env(
            &self.addrs,
            &DEV,
            vd_bins::ClusterShape::Single,
            &self.work_dir,
        )
    }

    /// The outbox file the shard will open — read out of the very env the shard is given, so this gate
    /// can never plant a row in a file the shard does not read.
    fn outbox_path(&self) -> PathBuf {
        let env = self.shard_env();
        let value = env
            .iter()
            .find(|(k, _)| *k == "VD_OUTBOX_PATH")
            .map(|(_, v)| v.clone())
            .expect("every shard env names its durable outbox");
        PathBuf::from(value)
    }

    /// THE LABEL the shard will check the file against — built from the SAME env the shard boots with, so
    /// a file this test writes is a file that shard accepts.
    fn outbox_stamp(&self) -> vd_core::store_stamp::StoreStamp {
        let mut vars: BTreeMap<String, String> = BTreeMap::new();
        for (key, value) in self.common().into_iter().chain(self.shard_env()) {
            vars.insert(key.to_owned(), value);
        }
        vd_bins::durable_stamp(
            &EnvConfig::new(vars),
            vd_core::store_stamp::StoreRole::Outbox,
            EpochId(0),
        )
        .expect("the shard's own outbox label")
    }

    /// Boot orchestrator + gateway + shard, exactly as `dev_control_nav` does.
    fn boot(&self) -> Cluster {
        let common = self.common();
        let spawn = |bin: &str, node_env: Vec<(&'static str, String)>| -> Child {
            vd_bins::spawn_node(bin, &common, &node_env).expect("spawn node")
        };
        let mut nodes = Cluster::new();
        nodes.push(
            "vd-orchestrator",
            spawn(
                env!("CARGO_BIN_EXE_vd-orchestrator"),
                orchestrator_env(
                    &self.addrs,
                    &DEV,
                    &self.orch_store,
                    vd_bins::ClusterShape::Single,
                ),
            ),
        );
        nodes.push(
            "vd-gateway",
            spawn(
                env!("CARGO_BIN_EXE_vd-gateway"),
                gateway_env(
                    &self.addrs,
                    &dev_auth_pubkey_hex(),
                    &DEV,
                    vd_bins::ClusterShape::Single,
                ),
            ),
        );
        nodes.push(
            "vd-shard",
            spawn(env!("CARGO_BIN_EXE_vd-shard"), self.shard_env()),
        );
        nodes
    }

    fn cleanup(&self) {
        let _ = std::fs::remove_dir_all(&self.trust_dir);
        let _ = std::fs::remove_dir_all(&self.work_dir);
        let _ = std::fs::remove_file(&self.orch_store);
    }
}

/// The gateway's own view of itself, or `None` while it is still coming up.
fn gateway_view(addr: SocketAddr) -> Option<GatewayView> {
    let body = vd_bins::admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str::<AdminSnapshot>(&body).ok()?.gateway
}

fn unsubscribed_count(addr: SocketAddr) -> u64 {
    gateway_view(addr).map_or(0, |g| g.interest_recipient_unsubscribed)
}

/// PLANT ONE RETAINED NOTICE, THE WAY A LIVE SHARD WOULD.
///
/// A real mesh bound as the shard's node id sends one `Retained` control frame toward the gateway, which
/// is not running yet: the row is staged and fsynced before the dial, the dial fails, the row stays. The
/// file is then closed and re-opened to prove the row is genuinely on disk (and that the lock is free for
/// the shard about to boot).
fn plant_one_retained_notice(rig: &Rig) {
    let payload = postcard::to_allocvec(&ShardToGateway::EntityOutOfInterest {
        realm_fence: Fence(1),
        session: ABSENT_SESSION,
        entity: EntityId(5),
        at: UniverseTick(60),
    })
    .expect("closed wire enums serialize infallibly");

    let path = rig.outbox_path();
    {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("runtime");
        let outbox = NodeOutbox::open(
            &path,
            vd_io_prod::store::StoreTuning::default(),
            rig.outbox_stamp(),
        )
        .expect("open the shard's outbox before it boots");
        let shared: SharedOutbox = std::sync::Arc::new(std::sync::Mutex::new(
            Box::new(outbox) as Box<dyn OutboxSink + Send>
        ));
        let peers: BTreeMap<vd_core::NodeId, SocketAddr> =
            [(GATEWAY, rig.addrs.gateway)].into_iter().collect();
        let cfg = vd_io_prod::mesh::MeshConfig::new(
            SHARD,
            reserve_udp_addr(),
            peers,
            DEV.outbound_cap as usize,
            1, // this planting process's own incarnation; the shard re-frames at its own on replay
            vd_bins::world_generation(),
        );
        let (mut transport, control) =
            vd_io_prod::mesh::spawn_mesh(runtime.handle(), &rig.trust, &cfg, Some(shared.clone()))
                .expect("plant mesh");
        transport
            .send_durable(
                GATEWAY,
                MsgClass::Control,
                vd_sim::io::bytes(payload),
                Durability::Retained,
            )
            .expect("stage the retained notice");

        // Wait for the row to be RETAINED on disk (the durable-before-send gate does the fsync).
        let started = Instant::now();
        loop {
            let rows = shared
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .scan_all()
                .len();
            if rows == 1 {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(30),
                "the retained notice never reached the outbox (rows {rows})"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        drop(control);
        drop(transport);
        drop(runtime); // ends the peer-writer tasks, releasing their handles on the sink
        drop(shared); // closes the file — the shard is about to open it
    }

    // Re-open it as the shard will: the label matches, and exactly one row waits to be re-driven.
    // The retry covers the moment the planting mesh's last task lets go of the file — the shard would
    // meet the same wait, and a file still locked here would fail its boot instead.
    let reopened = {
        let started = Instant::now();
        loop {
            match NodeOutbox::open(
                &path,
                vd_io_prod::store::StoreTuning::default(),
                rig.outbox_stamp(),
            ) {
                Ok(ob) => break ob,
                Err(e) => assert!(
                    started.elapsed() < Duration::from_secs(10),
                    "the planted outbox never re-opened: {e}"
                ),
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    };
    assert_eq!(
        reopened.scan_all().len(),
        1,
        "exactly one retained row waits on disk for the shard's boot replay"
    );
    drop(reopened);
}

/// Create the outbox file with the shard's own label and NO rows — the twin's starting state.
fn plant_empty_outbox(rig: &Rig) {
    let outbox = NodeOutbox::open(
        rig.outbox_path(),
        vd_io_prod::store::StoreTuning::default(),
        rig.outbox_stamp(),
    )
    .expect("open an empty outbox");
    assert_eq!(outbox.scan_all().len(), 0, "the twin starts with no rows");
    drop(outbox);
}

/// A client process that dies with its guard.
struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn poll_state(port: u16) -> Option<DevState> {
    match dev_roundtrip(port, &DevRequest::State).ok()? {
        DevResponse::State { state } => Some(state),
        _ => None,
    }
}

/// Log ONE real client in and wait until it is Active — the proof that a shard holding an open outbox
/// still runs the ordinary game.
fn login_and_wait_active(rig: &Rig, common: &[(&'static str, String)]) {
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    for (k, v) in common {
        cmd.env(k, v);
    }
    cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
    cmd.args([
        "--name",
        "outbox",
        "--agent-index",
        "0",
        "--gateway",
        &rig.addrs.gateway.to_string(),
        "--client-quic",
        &client_quic.port().to_string(),
        "--trust-dir",
        &rig.trust_dir.display().to_string(),
        "--dev-control",
        &devctl_port.to_string(),
        "--allow-dev-control",
    ]);
    let _client = KillOnDrop(cmd.spawn().expect("spawn client"));

    let started = Instant::now();
    let deadline = replay_deadline();
    loop {
        if poll_state(devctl_port).is_some_and(|s| s.phase == DevPhase::Active) {
            return;
        }
        assert!(
            started.elapsed() < deadline,
            "the client never became Active on a shard running with its outbox open"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn a_retained_notice_survives_the_shard_and_reaches_the_gateway_at_boot() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let rig = Rig::new("vd-outbox-replay");
    plant_one_retained_notice(&rig);

    let common = rig.common();
    let _nodes = rig.boot(); // RAII: reaps the cluster on test end or panic

    // The shard's boot replay re-drives the row; the gateway counts a notice for a session it never
    // minted. Nothing else in this cluster moves that counter.
    let started = Instant::now();
    let deadline = replay_deadline();
    loop {
        let count = unsubscribed_count(rig.gateway_admin);
        if count >= 1 {
            break;
        }
        assert!(
            started.elapsed() < deadline,
            "the replayed notice never reached the gateway (count {count} after {:?})",
            started.elapsed()
        );
        std::thread::sleep(Duration::from_millis(100));
    }

    // And the shard runs the ordinary game with that file open.
    login_and_wait_active(&rig, &common);
    rig.cleanup();
}

#[test]
fn an_empty_outbox_replays_nothing() {
    // FIRST statement: hold the process tier for the whole body.
    let _tier = vd_bins::cluster_tier();
    let rig = Rig::new("vd-outbox-replay-empty");
    plant_empty_outbox(&rig);

    let _nodes = rig.boot();

    // The cluster genuinely runs — the shard answers its readiness probe — and over the SAME wait the
    // counter never leaves zero. So the positive run's count came from the replayed row and nothing else.
    let started = Instant::now();
    let deadline = replay_deadline();
    let mut shard_ready = false;
    while started.elapsed() < deadline {
        assert_eq!(
            unsubscribed_count(rig.gateway_admin),
            0,
            "an empty outbox re-drives nothing, yet the gateway counted an unknown-session notice"
        );
        shard_ready |= vd_bins::http_get_status(
            rig.addrs.shard_probe,
            "/readyz",
            Some(Duration::from_secs(2)),
        ) == Some(200);
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        shard_ready,
        "the shard must actually have booted (and replayed its empty outbox) for the zero to mean anything"
    );
    assert!(
        gateway_view(rig.gateway_admin).is_some(),
        "the gateway must be answering for the zero to mean anything"
    );
    rig.cleanup();
}
