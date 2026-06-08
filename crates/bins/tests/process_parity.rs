//! The P1 process-tier parity gate: the SAME logical scenario the in-process
//! `p1_gates` proves — login, walk, see other dots, one connection target —
//! running as REAL BINARIES over localhost QUIC under cluster mTLS.
//!
//! Tier-1 (in-process) never solely gates a release: this tier validates the
//! quinn handshakes, the mesh bridge semantics, the bins' wiring, and the
//! admin endpoint — everything the VirtualClock tier cannot see.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, DEV_AUTH_SEED, GATEWAY, common_env, dev_auth_pubkey_hex,
    gateway_env, orchestrator_env, shard_env,
};
use vd_core::glam::DVec3;
use vd_core::pose::StampedPose;
use vd_core::{AccountId, EntityId, EpochId, NodeId, SessionId, TickId};
use vd_devproto::CLIENT_NODE_BASE;
use vd_io_prod::mesh::{MeshConfig, MeshTransport, spawn_mesh};
use vd_io_prod::runtime::TickPacer;
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Inbound, MsgClass, Transport};
use vd_wire::channels::{
    ClientControlMsg, InputDatagram, ServerControlMsg, SnapshotDatagram, SnapshotVerdict, SubId,
    classify_snapshot,
};
use vd_wire::version::ProtoVersion;

const DEADLINE: Duration = Duration::from_secs(30);

fn reserve_addr() -> SocketAddr {
    let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve");
    socket.local_addr().expect("addr")
}

/// A minimal real-protocol client over the production mesh transport.
struct ProcessClient {
    transport: MeshTransport,
    session: Option<SessionId>,
    sub: Option<SubId>,
    own_entity: Option<EntityId>,
    poses: BTreeMap<EntityId, StampedPose>,
    last_frame: Option<u64>,
    next_seq: u64,
    tick: u64,
}

impl ProcessClient {
    fn new(transport: MeshTransport) -> ProcessClient {
        ProcessClient {
            transport,
            session: None,
            sub: None,
            own_entity: None,
            poses: BTreeMap::new(),
            last_frame: None,
            next_seq: 0,
            tick: 0,
        }
    }

    fn send_hello(&mut self, account: AccountId) {
        let hello = ClientControlMsg::Hello {
            version: ProtoVersion::CURRENT,
            login: vd_connection_plane::tickets::mint_login(
                &DEV_AUTH_SEED,
                account,
                EpochId(1),
                account.0 as u64,
            ),
        };
        let bytes = postcard::to_allocvec(&hello).expect("encode");
        let _ = self
            .transport
            .send(GATEWAY, MsgClass::Control, bytes.into());
    }

    /// One client tick: drain, decode (asserting the single-peer invariant),
    /// optionally walk.
    fn step(&mut self, walk: bool) {
        for msg in self.transport.drain_inbound() {
            match msg {
                Inbound::Wire { from, class, bytes } => {
                    assert_eq!(from, GATEWAY, "one connection target, ever");
                    match class {
                        MsgClass::Control => self.on_control(&bytes),
                        MsgClass::Snapshot => self.on_snapshot(&bytes),
                        other => panic!("unexpected class {other:?}"),
                    }
                }
                Inbound::NodeUnreachable { .. } => {
                    // The gateway wasn't up yet for an early frame; retried below.
                }
            }
        }
        self.tick += 1;
        if walk && self.sub.is_some() {
            self.next_seq += 1;
            let input = InputDatagram {
                seq: self.next_seq,
                is_cut_marker: false,
                client_tick: TickId(self.tick),
                movement: [1.0, 0.0, 0.0],
                look: [0.0, 0.0],
                action_bits: 0,
            };
            let bytes = postcard::to_allocvec(&input).expect("encode");
            let _ = self.transport.send(GATEWAY, MsgClass::Input, bytes.into());
        }
    }

    fn on_control(&mut self, bytes: &[u8]) {
        match postcard::from_bytes::<ServerControlMsg>(bytes).expect("decode control") {
            ServerControlMsg::Welcome { session, .. } => self.session = Some(session),
            ServerControlMsg::SubscriptionOpened { sub, .. } => self.sub = Some(sub),
            ServerControlMsg::AuthorityChanged { entity, .. } => self.own_entity = Some(entity),
            ServerControlMsg::Close { reason } => panic!("gateway closed the session: {reason}"),
            other => panic!("unexpected control message in P1: {other:?}"),
        }
    }

    fn on_snapshot(&mut self, bytes: &[u8]) {
        let snap: SnapshotDatagram = postcard::from_bytes(bytes).expect("decode snapshot");
        // THE shared §6.3 gate — the SAME vd_wire fn the production client and the
        // in-process ScriptedClient call (no third hand-inlined copy to drift), so
        // the real-binary parity gate is held to the very SSOT it validates.
        match classify_snapshot(self.sub, self.last_frame, snap.sub, snap.frame_id) {
            SnapshotVerdict::Apply => {
                self.last_frame = Some(snap.frame_id);
                for entity in snap.entities {
                    self.poses.insert(entity.entity, entity.pose);
                }
            }
            SnapshotVerdict::DropForeignSub | SnapshotVerdict::DropStale => {}
        }
    }
}

#[test]
fn p1_parity_real_binaries_over_quic() {
    // ---- topology: addresses, trust bundle, child processes ------------------
    let orch_addr = reserve_addr();
    let gateway_addr = reserve_addr();
    let shard_addr = reserve_addr();
    let client_a_addr = reserve_addr();
    let client_b_addr = reserve_addr();
    let admin_addr = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve tcp");
        listener.local_addr().expect("addr")
    };

    let trust_dir = std::env::temp_dir().join(format!("vd-parity-{}", std::process::id()));
    let trust = ClusterTrust::generate("vd-parity").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");

    // The cluster contract is the SHARED vd_bins source of truth — the exact env,
    // params, roster, dev identity, and peer-book the launcher uses. If they ever
    // drift, THIS gate is what bring-up is really validated against.
    let addrs = ClusterAddrs {
        orchestrator: orch_addr,
        gateway: gateway_addr,
        shard: shard_addr,
        admin: admin_addr,
    };
    let clients = [
        (NodeId(CLIENT_NODE_BASE), client_a_addr),
        (NodeId(CLIENT_NODE_BASE + 1), client_b_addr),
    ];
    let auth_pubkey_hex = dev_auth_pubkey_hex();
    let common = common_env(&trust_dir.display().to_string(), &DEV);

    let spawn = |bin: &str, node_env: Vec<(&'static str, String)>| -> Child {
        let mut cmd = Command::new(bin);
        for (k, v) in common.iter().chain(node_env.iter()) {
            cmd.env(k, v);
        }
        cmd.spawn().expect("spawn child binary")
    };

    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        spawn(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            orchestrator_env(&addrs, &DEV),
        ),
    );
    cluster.push(
        "vd-gateway",
        spawn(
            env!("CARGO_BIN_EXE_vd-gateway"),
            gateway_env(&addrs, &clients, &auth_pubkey_hex, &DEV),
        ),
    );
    cluster.push(
        "vd-shard",
        spawn(env!("CARGO_BIN_EXE_vd-shard"), shard_env(&addrs, &DEV)),
    );
    let _guard = cluster; // RAII: kill the children on test end or panic

    // ---- two real-protocol clients over the production mesh ------------------
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    // STRUCTURAL one-connection invariant: the clients' address books contain
    // ONLY the gateway — no other node is reachable, by construction.
    let client_book = BTreeMap::from([(GATEWAY, gateway_addr)]);
    let mesh = |id: NodeId, bind: SocketAddr| -> MeshTransport {
        let (t, _c) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(id, bind, client_book.clone(), 64),
        )
        .expect("client mesh");
        std::mem::forget(_c); // keep the endpoint alive for the test's duration
        t
    };
    let mut walker = ProcessClient::new(mesh(NodeId(CLIENT_NODE_BASE), client_a_addr));
    let mut idle = ProcessClient::new(mesh(NodeId(CLIENT_NODE_BASE + 1), client_b_addr));

    // ---- drive the scenario at real tick rate --------------------------------
    let started = Instant::now();
    let mut pacer = TickPacer::new(DEV.tick_hz);
    let mut hello_retry = Instant::now();
    walker.send_hello(AccountId(1000));
    idle.send_hello(AccountId(1001));
    loop {
        walker.step(true);
        idle.step(false);
        // Children may still be booting: retry Hello until welcomed (idempotent).
        if hello_retry.elapsed() > Duration::from_millis(500) {
            hello_retry = Instant::now();
            if walker.session.is_none() {
                walker.send_hello(AccountId(1000));
            }
            if idle.session.is_none() {
                idle.send_hello(AccountId(1001));
            }
        }
        let done = walker.poses.len() == 2
            && idle.poses.len() == 2
            && walker.own_entity.is_some_and(|own| {
                walker
                    .poses
                    .get(&own)
                    .is_some_and(|p| p.pos.distance(DVec3::ZERO) > 0.5)
            });
        if done {
            break;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "parity scenario did not converge: walker(session={:?} sub={:?} poses={} moved={:?}) \
             idle(session={:?} poses={})",
            walker.session,
            walker.sub,
            walker.poses.len(),
            walker
                .own_entity
                .and_then(|e| walker.poses.get(&e))
                .map(|p| p.pos),
            idle.session,
            idle.poses.len(),
        );
        let _ = pacer.wait();
    }

    // ---- the same logical assertions as the in-process tier ------------------
    assert!(walker.session.is_some(), "walker logged in");
    assert!(idle.session.is_some(), "idle logged in");
    assert_ne!(walker.session, idle.session, "distinct sessions");
    assert_eq!(walker.sub, Some(SubId(0)), "M0: the one subscription");
    let idle_own = idle
        .own_entity
        .expect("authority announced to the idle dot");
    assert_eq!(
        idle.poses[&idle_own].pos,
        DVec3::ZERO,
        "the idle dot never moved"
    );

    // ---- the 2am curl: the admin endpoint shows the live directory -----------
    let snapshot = http_get_json(admin_addr, "/admin/snapshot");
    let entries = snapshot["directory"].as_array().expect("directory array");
    let authorities: Vec<&str> = entries
        .iter()
        .map(|e| e["authority"].as_str().expect("authority"))
        .collect();
    assert!(
        authorities.contains(&"shard:node-3"),
        "realm + entity records: {authorities:?}"
    );
    assert!(
        authorities.contains(&"gateway:node-2"),
        "session records: {authorities:?}"
    );
    // Exactly: 1 realm + 2 sessions + 2 entities = 5 records, every one fenced.
    assert_eq!(entries.len(), 5, "directory dump: {entries:?}");
    let _ = std::fs::remove_dir_all(&trust_dir);
}

/// A raw HTTP/1.1 GET returning parsed JSON (what `curl` would do at 2am).
fn http_get_json(addr: SocketAddr, path: &str) -> serde_json::Value {
    let started = Instant::now();
    loop {
        let mut conn = std::net::TcpStream::connect(addr).expect("admin reachable");
        let request =
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        conn.write_all(request.as_bytes()).expect("request");
        let mut response = String::new();
        conn.read_to_string(&mut response).expect("response");
        let body = response
            .split_once("\r\n\r\n")
            .map(|(_, b)| b.to_owned())
            .expect("http body");
        if let Ok(value) = serde_json::from_str(&body) {
            return value;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "admin endpoint never answered"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}
