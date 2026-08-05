//! The P1 process-tier parity gate: the SAME logical scenario the in-process
//! `p1_gates` proves — login, walk, see other dots, one connection target —
//! running as REAL BINARIES over localhost QUIC under cluster mTLS.
//!
//! Tier-1 (in-process) never solely gates a release: this tier validates the
//! quinn handshakes, the mesh bridge semantics, the bins' wiring, and the
//! admin endpoint — everything the VirtualClock tier cannot see.

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::process::Child;
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, DEV_AUTH_SEED, GATEWAY, admin_get_body, common_env,
    dev_auth_pubkey_hex, gateway_env, orchestrator_env, reserve_tcp_addr, reserve_udp_addr,
    shard_env, spawn_node,
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

/// A minimal real-protocol client over the production mesh transport.
struct ProcessClient {
    transport: MeshTransport,
    session: Option<SessionId>,
    held_subs: BTreeSet<SubId>,
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
            held_subs: BTreeSet::new(),
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
                Inbound::SendShed { .. } => {
                    // A local send-shed (R-4d M3): this test client's sends are tiny + its buffer
                    // ample, so this never fires — present for Inbound exhaustiveness.
                }
            }
        }
        self.tick += 1;
        if walk && !self.held_subs.is_empty() {
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
            ServerControlMsg::SubscriptionOpened { sub, .. } => {
                self.held_subs.insert(sub);
            }
            // S6: the node-agnostic own-entity signal (minor 2) names the avatar by EntityId alone —
            // this pure-renderer parity client learns which entity is itself WITHOUT any node info.
            ServerControlMsg::OwnEntity { entity } => self.own_entity = Some(entity),
            // Legacy node-aware `AuthorityChanged` — still emitted to old clients; a minor-2 client
            // ignores it (OwnEntity is the own-entity source of truth).
            ServerControlMsg::AuthorityChanged { .. } => {}
            ServerControlMsg::Close { reason } => panic!("gateway closed the session: {reason}"),
            // The cluster tick rate (minor 1) — this minimal parity client does not
            // interpolate; ignore it (the real client learns its render rate from it).
            ServerControlMsg::UniverseRate { .. } => {}
            other => panic!("unexpected control message in P1: {other:?}"),
        }
    }

    fn on_snapshot(&mut self, bytes: &[u8]) {
        let snap: SnapshotDatagram = postcard::from_bytes(bytes).expect("decode snapshot");
        // THE shared §6.3 gate — the SAME vd_wire fn the production client and the
        // in-process ScriptedClient call (no third hand-inlined copy to drift), so
        // the real-binary parity gate is held to the very SSOT it validates.
        match classify_snapshot(&self.held_subs, self.last_frame, snap.sub, snap.frame_id) {
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
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // ---- topology: addresses, trust bundle, child processes ------------------
    let orch_addr = reserve_udp_addr();
    let gateway_addr = reserve_udp_addr();
    let shard_addr = reserve_udp_addr();
    let client_a_addr = reserve_udp_addr();
    let client_b_addr = reserve_udp_addr();
    let admin_addr = reserve_tcp_addr();
    // RLM RG-4c non-vacuity control: give the STATIC gateway its own admin endpoint so this test can prove a
    // static gateway NEVER receives a reactive greeting nor mints a dynamic home — making the demand-login
    // e2e's `presence_announces >= 1` / `dynamic_shards >= 1` assertions non-vacuous.
    let gw_admin = reserve_tcp_addr();

    let trust_dir = std::env::temp_dir().join(format!("vd-parity-{}", std::process::id()));
    let trust = ClusterTrust::generate("vd-parity").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    // D-6: the orchestrator's durable Store (temp scratch ⇒ VD_STORE_EPHEMERAL_OK via orchestrator_env).
    // Remove any stale file from a recycled-pid prior run so this parity spawn boots at genesis.
    let orch_store =
        std::env::temp_dir().join(format!("vd-parity-{}-orch.redb", std::process::id()));
    let _ = std::fs::remove_file(&orch_store);
    let orch_store = orch_store.display().to_string();

    // The cluster contract is the SHARED vd_bins source of truth — the exact env,
    // params, roster, dev identity, and peer-book the launcher uses. If they ever
    // drift, THIS gate is what bring-up is really validated against.
    let addrs = ClusterAddrs {
        orchestrator: orch_addr,
        gateway: gateway_addr,
        shard: shard_addr,
        admin: admin_addr,
        gateway_admin: Some(gw_admin),
        ..ClusterAddrs::reserve()
    };
    let clients = [
        (NodeId(CLIENT_NODE_BASE), client_a_addr),
        (NodeId(CLIENT_NODE_BASE + 1), client_b_addr),
    ];
    let auth_pubkey_hex = dev_auth_pubkey_hex();
    let common = common_env(&trust_dir.display().to_string(), &DEV);

    let spawn = |bin: &str, node_env: Vec<(&'static str, String)>| -> Child {
        spawn_node(bin, &common, &node_env).expect("spawn child binary")
    };

    let mut cluster = Cluster::new();
    // Single-shard parity: every builder passes `dual=false` (the inert arm — byte-identical to the
    // pre-Track-R env; proven by `single_shard_env_is_byte_identical_to_dual_false`).
    cluster.push(
        "vd-orchestrator",
        spawn(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            orchestrator_env(&addrs, &DEV, &orch_store, vd_bins::ClusterShape::Single),
        ),
    );
    cluster.push(
        "vd-gateway",
        spawn(
            env!("CARGO_BIN_EXE_vd-gateway"),
            gateway_env(
                &addrs,
                &clients,
                &auth_pubkey_hex,
                &DEV,
                vd_bins::ClusterShape::Single,
            ),
        ),
    );
    cluster.push(
        "vd-shard",
        spawn(
            env!("CARGO_BIN_EXE_vd-shard"),
            shard_env(&addrs, &DEV, vd_bins::ClusterShape::Single),
        ),
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
            &MeshConfig::new(id, bind, client_book.clone(), 64, 0),
            None,
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
                    .is_some_and(|p| {
                        // TOTAL displacement — whole-number part plus leftover. The integrator folds the
                        // leftover into the whole number every tick, so the leftover alone is a
                        // sub-millimetre remainder and never reaches this threshold however far the
                        // walker walks: this loop would spin until its deadline.
                        p.pos
                            .delta_m(
                                vd_core::pose::LatticePos::local(DVec3::ZERO),
                                vd_core::pose::Tier::Fine,
                            )
                            .distance(DVec3::ZERO)
                            > 0.5
                    })
            });
        if done {
            break;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "parity scenario did not converge: walker(session={:?} subs={:?} poses={} moved={:?}) \
             idle(session={:?} poses={})",
            walker.session,
            walker.held_subs,
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
    assert_eq!(
        walker.held_subs,
        BTreeSet::from([SubId(0)]),
        "M0: the one subscription"
    );
    let idle_own = idle
        .own_entity
        .expect("authority announced to the idle dot");
    assert_eq!(
        idle.poses[&idle_own].pos.offset(),
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

    // ---- RLM RG-4c non-vacuity control: a STATIC gateway never greets nor mints ----
    // Both clients logged into the PRE-BOOKED shard (node-3), so the gateway received NO reactive greeting
    // and minted NO dynamic home. This is the falsifiable twin of the demand-login e2e: it proves that test's
    // `presence_announces >= 1` / `dynamic_shards >= 1` could only come from the demand path, never a static one.
    let gw_snapshot = http_get_json(gw_admin, "/admin/snapshot");
    let gw = &gw_snapshot["gateway"];
    assert_eq!(
        gw["presence_announces"].as_u64(),
        Some(0),
        "a static gateway must receive NO reactive greeting: {gw}",
    );
    assert_eq!(
        gw["dynamic_shards"].as_u64(),
        Some(0),
        "a static login routes to the pre-booked shard, never a demand-spawned node: {gw}",
    );

    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&orch_store);
}

/// A raw HTTP/1.1 GET returning parsed JSON (what `curl` would do at 2am).
fn http_get_json(addr: SocketAddr, path: &str) -> serde_json::Value {
    let started = Instant::now();
    loop {
        if let Some(value) = admin_get_body(addr, path, None)
            .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok())
        {
            return value;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "admin endpoint never answered"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}
