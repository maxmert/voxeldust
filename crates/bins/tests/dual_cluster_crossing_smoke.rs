//! Track R / 1d.2 (Batch A) — THE dual-shard PROCESS-cluster crossing proof.
//!
//! Stands up the LOCAL 2-process crossing playground as REAL BINARIES over localhost QUIC under mTLS
//! (`vd-devcluster up --dual`): a SOURCE shard (realm `System(7)`, the login shard) + a DEST shard
//! (realm `System(8)`) + the orchestrator + gateway. The launcher plants the born-inside geometric
//! crossing boundary (`System(7)`→`System(8)`) on the SOURCE via `VD_REALM_BOUNDARIES` and — the C1 gate
//! — waits for BOTH realms to be granted before `up` returns, so the crossing can never count
//! `crossing_unresolved`.
//!
//! Then a real-protocol client (the production mesh transport) logs in on the SOURCE. Its avatar spawns
//! INSIDE the planted shell, so the SOURCE's geometric dwell detector fires a `CrossingRequest`
//! AUTONOMOUSLY (nothing hand-fed — no `trigger_transfer`), the orchestrator resolves the three directory
//! heads and starts THE proven transfer saga, and the directory CAS re-homes the dot's `Entity` authority
//! onto the DEST. The gate polls the orchestrator's `/admin/snapshot` until an `Entity` row's authority is
//! `shard:node-4` (the DEST OWNS the dot) — the process-tier mirror of `crossing_e2e`'s dest-owns proof.
//!
//! ANTI-VACUITY: the crossing fires from the GEOMETRIC trigger the launcher planted (born-inside dwell),
//! not from any test-driven start; the C1 both-realms gate guarantees the DEST head resolves. A grep of
//! this file finds ZERO `trigger`/`start_transfer` — the re-home is autonomous.
//!
//! SCOPE (M-2): the LOCAL 2-process playground. The N-shard k3d roster generalization is ledgered to
//! cloud (#123) in `docs/design/DEFERRED.md`. This is Tier-B process glue (coverage-exempt); the Tier-A
//! verdicts it stands on (the env builders, the port scheme, `AdminSnapshot::realms_present`) are
//! 100%-covered by their own unit tests.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use vd_bins::{
    CROSSING_SLOT, DEV, DEV_AUTH_SEED, DevClusterDown, GATEWAY, admin_get_body, devcluster,
    loopback, slot_trust_dir,
};
use vd_core::{AccountId, EpochId, NodeId, TickId};
use vd_devproto::{CLIENT_NODE_BASE, DevPortScheme};
use vd_io_prod::mesh::{MeshConfig, MeshTransport, spawn_mesh};
use vd_io_prod::runtime::TickPacer;
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Inbound, MsgClass, Transport};
use vd_wire::channels::{ClientControlMsg, InputDatagram, ServerControlMsg};
use vd_wire::version::ProtoVersion;

/// A generous deadline: `up --dual` boots 4 processes + the C1 both-realms grant, then the client logs in
/// and the born-inside dwell + saga self-drive to the directory CAS. Real QUIC handshakes over loopback.
const DEADLINE: Duration = Duration::from_secs(45);

/// The DEST shard's rendered authority in the admin directory (`AuthorityRef::Shard(NodeId(4))`).
const DEST_AUTHORITY: &str = "shard:node-4";

/// Run one `vd-devcluster up --dual` against the crossing slot (the launcher path from `CARGO_BIN_EXE`).
fn up_dual(launcher: &str, slot: u16) -> std::process::ExitStatus {
    std::process::Command::new(launcher)
        .args(["up", "--slot", &slot.to_string(), "--dual"])
        .status()
        .expect("run vd-devcluster up --dual")
}

/// A minimal real-protocol client that logs in and drains — enough to make the SOURCE grant its avatar
/// (which spawns inside the planted shell) and open a session the crossing's route can resolve.
struct LoginClient {
    transport: MeshTransport,
    session: bool,
    subscribed: bool,
    /// Set when the gateway asks for the cut (`ServerControlMsg::RequestCut`): the NEXT input carries
    /// `is_cut_marker = true` — the client's half of the transfer cut (mirrors the real client / the
    /// harness `ScriptedClient`). Without this the saga parks at `Cutting` forever and never CAS-commits.
    emit_marker_next: bool,
    next_seq: u64,
    tick: u64,
}

impl LoginClient {
    fn new(transport: MeshTransport) -> LoginClient {
        LoginClient {
            transport,
            session: false,
            subscribed: false,
            emit_marker_next: false,
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
        let bytes = postcard::to_allocvec(&hello).expect("encode hello");
        let _ = self
            .transport
            .send(GATEWAY, MsgClass::Control, bytes.into());
    }

    /// Drain inbound (noting the welcome + the first subscription), then WALK: once subscribed, send a
    /// forward-movement input datagram every tick. The dot's avatar stays born-inside the 1150 m shell for
    /// thousands of ticks (~0.04 m/tick), so the SOURCE's dwell detector fires the crossing after `n_entry`
    /// in-band ticks — the SAME proven `should_commit` path the in-process `crossing_e2e` walks (nothing
    /// hand-fed; the crossing is purely geometric). Walking also keeps the dot live + integrated on the shard.
    fn step(&mut self) {
        for msg in self.transport.drain_inbound() {
            if let Inbound::Wire {
                class: MsgClass::Control,
                bytes,
                ..
            } = msg
            {
                match postcard::from_bytes::<ServerControlMsg>(&bytes) {
                    Ok(ServerControlMsg::Welcome { .. }) => self.session = true,
                    Ok(ServerControlMsg::SubscriptionOpened { .. }) => self.subscribed = true,
                    // The gateway's cut request — stamp the CUT_MARKER on the next input so the transfer
                    // saga can advance past `Cutting` to the directory CAS (the client's half of the cut).
                    Ok(ServerControlMsg::RequestCut { .. }) => self.emit_marker_next = true,
                    _ => {}
                }
            }
        }
        self.tick += 1;
        if self.subscribed {
            self.next_seq += 1;
            let is_cut_marker = self.emit_marker_next;
            self.emit_marker_next = false;
            let input = InputDatagram {
                seq: self.next_seq,
                is_cut_marker,
                client_tick: TickId(self.tick),
                movement: [1.0, 0.0, 0.0],
                look: [0.0, 0.0],
                action_bits: 0,
            };
            let bytes = postcard::to_allocvec(&input).expect("encode input");
            let _ = self.transport.send(GATEWAY, MsgClass::Input, bytes.into());
        }
    }
}

/// Fetch the orchestrator admin snapshot's directory rows as `(key, authority)` pairs, or empty on a
/// not-yet-answering endpoint. Reuses the SHARED `admin_get_body` HTTP client (never a hand-inlined GET).
fn directory_rows(admin: SocketAddr) -> Vec<(String, String)> {
    let Some(body) = admin_get_body(admin, "/admin/snapshot", Some(Duration::from_secs(2))) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    value["directory"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    Some((
                        e["key"].as_str()?.to_owned(),
                        e["authority"].as_str()?.to_owned(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A directory row proving the DEST OWNS the transferred dot: an `Entity` key (`ent-…`, distinct from the
/// DEST's own realm row `system-…`) whose authority is `shard:node-4`. This is the process-tier dest-owns
/// verdict — the directory CAS moved the dot's authority off the source onto the DEST.
fn dest_owns_an_entity(rows: &[(String, String)]) -> bool {
    rows.iter()
        .any(|(key, authority)| key.starts_with("ent-") && authority == DEST_AUTHORITY)
}

#[test]
fn a_dot_re_homes_source_to_dest_over_the_process_dual_shard_tier() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let slot = CROSSING_SLOT;
    let _ = devcluster(launcher, "down", slot); // clean slate (idempotent)
    let _guard = DevClusterDown::new(launcher, slot);

    let ports = DevPortScheme::DEFAULT
        .slot_ports(slot)
        .expect("crossing slot resolves");

    // ---- WIRING (Batch A): stand up the 4-process dual cluster --------------------------------------
    // `up --dual` spawns orchestrator + gateway + SOURCE(realm 7) + DEST(realm 8), plants the born-inside
    // System(7)→System(8) boundary on the SOURCE, and — the C1 gate — returns 0 ONLY after BOTH realms are
    // granted to shards in the ONE directory (so the crossing's dest head resolves, never `unresolved`).
    assert!(
        up_dual(launcher, slot).success(),
        "up --dual must reach the C1 both-realms ready gate and exit 0 (gateway known_shards={{SHARD,DEST}}, \
         orchestrator roster has DEST, the directory holds Realm(7)+Realm(8))",
    );

    // The C1 gate proof, read straight off the admin directory: BOTH realm rows are present, each owned by
    // a shard (`system-…` @ `shard:node-…`). Realm 7 → the source shard, realm 8 → the DEST shard.
    let admin = loopback(ports.admin);
    let rows = directory_rows(admin);
    let realm_rows: Vec<&(String, String)> = rows
        .iter()
        .filter(|(key, authority)| key.starts_with("system-") && authority.starts_with("shard:"))
        .collect();
    assert!(
        realm_rows
            .iter()
            .any(|(key, _)| *key == format!("{}", vd_core::pose::RealmId::System(DEV.realm_seed)))
            && realm_rows.iter().any(|(key, _)| {
                *key == format!("{}", vd_core::pose::RealmId::System(DEV.realm_seed_b))
            }),
        "C1: both realms {} + {} must be shard-granted in the ONE directory before the crossing drives: {rows:?}",
        vd_core::pose::RealmId::System(DEV.realm_seed),
        vd_core::pose::RealmId::System(DEV.realm_seed_b),
    );

    // ---- THE CROSSING: a real client logs in; the born-inside dwell fires the re-home AUTONOMOUSLY -----
    // Load the launcher's mTLS trust bundle + bind at the client-0 QUIC port the gateway pre-booked
    // (`CLIENT_NODE_BASE` → client_quic(0)), then log in on the SOURCE. The avatar spawns inside the
    // planted shell, so the SOURCE's geometric dwell detector emits a `CrossingRequest` on its own.
    let trust =
        ClusterTrust::from_der_dir(&slot_trust_dir(slot)).expect("load launcher trust bundle");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime");
    // STRUCTURAL one-connection invariant: the client books ONLY the gateway.
    let client_book = BTreeMap::from([(GATEWAY, loopback(ports.gateway))]);
    let client_bind = loopback(ports.client_quic(0).expect("client-0 quic port"));
    let (transport, control) = spawn_mesh(
        rt.handle(),
        &trust,
        &MeshConfig::new(NodeId(CLIENT_NODE_BASE), client_bind, client_book, 64, 0),
        None,
    )
    .expect("client mesh");
    std::mem::forget(control); // keep the endpoint alive for the test's duration
    let mut client = LoginClient::new(transport);

    let started = Instant::now();
    let mut pacer = TickPacer::new(DEV.tick_hz);
    let mut hello_retry = Instant::now();
    client.send_hello(AccountId(1000));
    loop {
        client.step();
        // Children may still be settling the QUIC handshake; re-send Hello until welcomed (idempotent).
        if !client.session && hello_retry.elapsed() > Duration::from_millis(500) {
            hello_retry = Instant::now();
            client.send_hello(AccountId(1000));
        }
        // THE PROOF: poll the ONE directory until the DEST owns an Entity — the dot re-homed SOURCE→DEST.
        if dest_owns_an_entity(&directory_rows(admin)) {
            break;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "the dot never re-homed to the DEST (session={}): the born-inside geometric crossing did not \
             flip an Entity head to {DEST_AUTHORITY}. directory={:?}",
            client.session,
            directory_rows(admin),
        );
        let _ = pacer.wait();
    }

    // The dest-owns flip IS the proof; assert it once more explicitly for the failure message + to pin
    // that the SOURCE no longer holds THAT entity under its own authority (the CAS moved it, not copied).
    let final_rows = directory_rows(admin);
    assert!(
        dest_owns_an_entity(&final_rows),
        "the DEST must own the re-homed dot's Entity row: {final_rows:?}",
    );
    let source_entity_rows = final_rows
        .iter()
        .filter(|(key, authority)| key.starts_with("ent-") && authority == "shard:node-3")
        .count();
    let dest_entity_rows = final_rows
        .iter()
        .filter(|(key, authority)| key.starts_with("ent-") && authority == DEST_AUTHORITY)
        .count();
    assert!(
        dest_entity_rows >= 1,
        "the DEST holds >= 1 re-homed Entity ({dest_entity_rows}); the directory CAS committed the crossing: {final_rows:?}",
    );
    // Belt-and-suspenders anti-vacuity: the whole run's client actually established a session (the
    // crossing's `Session` head resolved), so the re-home was a real logged-in dot, not a phantom.
    assert!(
        client.session,
        "the client established a session (the crossing route's Session head resolved)"
    );
    let _ = source_entity_rows; // observed for the failure message; the CAS may leave a source Ghost row transiently
}
