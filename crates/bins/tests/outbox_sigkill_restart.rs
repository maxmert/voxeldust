//! R-6d4-D — the PROCESS-TIER SIGKILL-restart durable-outbox + boot-counter-by-EXACTLY-1 proof.
//!
//! Boots a real `vd-outbox-testnode` child (source = NodeId(1)) that SEEDS one durable-and-unacked
//! `ReliableFrame` outbox row to an IN-PROCESS receiver B (NodeId(2), a real `MeshTransport` this test holds —
//! a killed shard has NO admin surface, so B is how we observe the re-drive), SIGKILLs it in the durable
//! window, then restarts it on the SAME outbox + boot-state dir. Asserts: (a) the boot-2 replay RE-DROVE the
//! seeded row (B receives the payload), and (b) the durable BootCounter incremented by EXACTLY 1 (the process
//! incarnation resolved exactly once across the crash). Plus an anti-theater twin: a FRESH outbox re-drives
//! ZERO rows (B stays empty) — so the positive path is not a false-positive from some other delivery.
//!
//! `store-test-hooks` ONLY (the seed seam + `BootCounter::current` are feature-gated, ABSENT from release).
//! The RESTART case (survives-a-kill); the NEVER-restart prod closure is CA-1-era (out of scope, ledgered).
#![cfg(feature = "store-test-hooks")]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use vd_bins::{BOOT_COUNTER_NAME, Cluster, book, reserve_udp_addr, spawn_node};
use vd_core::NodeId;
use vd_io_prod::boot::BootCounter;
use vd_io_prod::mesh::{MeshConfig, MeshControl, MeshTransport, spawn_mesh};
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Inbound, Transport};

const SOURCE: NodeId = NodeId(1);
const B: NodeId = NodeId(2);
const DEADLINE: Duration = Duration::from_secs(30);
const BIN: &str = env!("CARGO_BIN_EXE_vd-outbox-testnode");

/// RAII removal of a run's on-disk temp artifacts (cleared at start too — pid+tag stable).
struct TempPaths {
    store: std::path::PathBuf,
    trust_dir: std::path::PathBuf,
    boot_dir: std::path::PathBuf,
}
impl TempPaths {
    fn new(tag: &str) -> TempPaths {
        let base = std::env::temp_dir();
        let pid = std::process::id();
        let p = TempPaths {
            store: base.join(format!("vd-outbox-sigkill-{tag}-{pid}.redb")),
            trust_dir: base.join(format!("vd-outbox-sigkill-trust-{tag}-{pid}")),
            boot_dir: base.join(format!("vd-outbox-sigkill-boot-{tag}-{pid}")),
        };
        p.cleanup();
        std::fs::create_dir_all(&p.boot_dir).expect("boot dir");
        p
    }
    fn cleanup(&self) {
        let _ = std::fs::remove_file(&self.store);
        let _ = std::fs::remove_file(self.store.with_extension("seeded"));
        let _ = std::fs::remove_file(self.store.with_extension("ready"));
        let _ = std::fs::remove_dir_all(&self.trust_dir);
        let _ = std::fs::remove_dir_all(&self.boot_dir);
    }
    fn counter_path(&self) -> std::path::PathBuf {
        self.boot_dir.join(BOOT_COUNTER_NAME)
    }
}
impl Drop for TempPaths {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// The source child's env. `seed = Some(payload)` ⇒ boot 1 (seed-and-idle); `None` ⇒ boot 2
/// (boot_mesh_and_replay). NO `VD_PROCESS_INCARNATION` — so the durable BootCounter is used (the whole point).
fn source_env(
    paths: &TempPaths,
    book_str: &str,
    addr_src: SocketAddr,
    seed: Option<u8>,
) -> Vec<(&'static str, String)> {
    let mut env = vec![
        ("VD_NODE_ID", "1".to_owned()),
        ("VD_BIND", addr_src.to_string()),
        ("VD_PEERS", book_str.to_owned()),
        ("VD_OUTBOUND_CAP", "64".to_owned()),
        ("VD_TRUST_DIR", paths.trust_dir.display().to_string()),
        ("VD_OUTBOX_PATH", paths.store.display().to_string()),
        ("VD_OUTBOX_EPHEMERAL_OK", "1".to_owned()),
        ("VD_BOOT_STATE_DIR", paths.boot_dir.display().to_string()),
        ("VD_BOOT_STATE_EPHEMERAL_OK", "1".to_owned()),
    ];
    if let Some(payload) = seed {
        env.push(("VD_OUTBOX_TEST_SEED", payload.to_string()));
    }
    env
}

/// Poll a marker file to existence (deterministic — the child writes it at a known point), or fail loud.
fn wait_for_marker(marker: &std::path::Path, what: &str) {
    let start = Instant::now();
    while !marker.exists() {
        std::thread::sleep(Duration::from_millis(10));
        assert!(
            start.elapsed() < DEADLINE,
            "timed out waiting for {what} marker {marker:?}"
        );
    }
}

/// Drain B's inbound, collecting the first payload byte of every delivered reliable `Wire` event.
fn drain_b(b: &mut MeshTransport, out: &mut Vec<u8>) {
    for ev in b.drain_inbound() {
        if let Inbound::Wire { bytes, .. } = ev
            && !bytes.is_empty()
        {
            out.push(bytes[0]);
        }
    }
}

/// Spin up an in-process receiver B (a real `MeshTransport` on `addr_b`) + the trust both sides share. Returns
/// the runtime (held so B's tasks live), B's transport+control, and the shared book string for `VD_PEERS`.
fn spin_up_b(
    paths: &TempPaths,
    addr_src: SocketAddr,
    addr_b: SocketAddr,
) -> (tokio::runtime::Runtime, MeshTransport, MeshControl, String) {
    let trust = ClusterTrust::generate("vd-outbox-sigkill").expect("trust");
    trust
        .write_der_dir(&paths.trust_dir)
        .expect("write trust der"); // the child reads VD_TRUST_DIR
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("rt");
    let book_map: BTreeMap<NodeId, SocketAddr> =
        [(SOURCE, addr_src), (B, addr_b)].into_iter().collect();
    let cfg_b = MeshConfig::new(B, addr_b, book_map, 256, 1, vd_bins::world_generation());
    let (b, ctl_b) = spawn_mesh(rt.handle(), &trust, &cfg_b, None).expect("spawn B");
    let book_str = book(&[(SOURCE, addr_src), (B, addr_b)]);
    (rt, b, ctl_b, book_str)
}

#[test]
fn sigkill_restart_redrives_the_durable_outbox_row_and_boots_counter_by_exactly_one() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    const PAYLOAD: u8 = 0x5D;
    let paths = TempPaths::new("redrive");
    let (addr_src, addr_b) = (reserve_udp_addr(), reserve_udp_addr());
    let (_rt, mut b, _ctl_b, book_str) = spin_up_b(&paths, addr_src, addr_b);

    // BOOT 1: seed a durable row to B, then idle for the SIGKILL. Cluster RAII-reaps on any panic.
    let mut cluster = Cluster::new();
    let child = spawn_node(
        BIN,
        &[],
        &source_env(&paths, &book_str, addr_src, Some(PAYLOAD)),
    )
    .expect("spawn boot-1 source");
    cluster.push("src", child);
    wait_for_marker(&paths.store.with_extension("seeded"), "boot-1 seed-durable");
    let v1 = BootCounter::current(&paths.counter_path())
        .expect("read v1")
        .expect("boot-1 minted a durable incarnation v1");

    // SIGKILL + reap (frees addr_src + the redb lock), then RESTART on the SAME env/addr (no seed).
    cluster.kill_and_reap("src");
    let child2 = spawn_node(BIN, &[], &source_env(&paths, &book_str, addr_src, None))
        .expect("spawn boot-2 source");
    cluster.push("src2", child2);

    // The boot-2 replay re-drives the retained row over real QUIC to the in-process B (first-contact Accept).
    let mut got: Vec<u8> = Vec::new();
    let start = Instant::now();
    while !got.contains(&PAYLOAD) {
        drain_b(&mut b, &mut got);
        std::thread::sleep(Duration::from_millis(5));
        assert!(
            start.elapsed() < DEADLINE,
            "timed out: boot-2 replay did not re-drive the seeded row to B (got {got:?})"
        );
    }

    // The INDEPENDENT exactly-once check: the durable counter incremented by EXACTLY 1 across the crash (not
    // a delivery precondition — read after B observed the re-drive, so boot 2's counter write is complete).
    let v2 = BootCounter::current(&paths.counter_path())
        .expect("read v2")
        .expect("boot-2 minted a durable incarnation v2");
    assert_eq!(
        v2,
        v1 + 1,
        "the process incarnation resolved EXACTLY once across the SIGKILL-restart"
    );
}

#[test]
fn fresh_outbox_re_drives_zero_rows() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // Anti-theater twin: NO seed ⇒ a fresh/empty outbox ⇒ boot-2 replay re-drives NOTHING, so B stays EMPTY
    // over a settle generous enough that the positive path would have landed. Proves the positive test is not
    // a false-positive from an unrelated delivery. Still asserts v2 == v1 + 1 (a genuine restart happened).
    let paths = TempPaths::new("fresh");
    let (addr_src, addr_b) = (reserve_udp_addr(), reserve_udp_addr());
    let (_rt, mut b, _ctl_b, book_str) = spin_up_b(&paths, addr_src, addr_b);
    let ready = paths.store.with_extension("ready");

    // BOOT 1: no seed ⇒ boot_mesh_and_replay on an empty outbox (0 re-driven), then idle.
    let mut cluster = Cluster::new();
    let child = spawn_node(BIN, &[], &source_env(&paths, &book_str, addr_src, None))
        .expect("spawn boot-1 (no seed)");
    cluster.push("src", child);
    wait_for_marker(&ready, "boot-1 ready");
    let v1 = BootCounter::current(&paths.counter_path())
        .expect("v1")
        .expect("v1 minted");

    // Restart (still no seed).
    cluster.kill_and_reap("src");
    let _ = std::fs::remove_file(&ready); // so we poll boot-2's fresh marker
    let child2 = spawn_node(BIN, &[], &source_env(&paths, &book_str, addr_src, None))
        .expect("spawn boot-2 (no seed)");
    cluster.push("src2", child2);
    wait_for_marker(&ready, "boot-2 ready");

    // B must receive NOTHING over a bounded settle (the positive path delivers well within this window).
    let mut got: Vec<u8> = Vec::new();
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(5) {
        drain_b(&mut b, &mut got);
        assert!(
            got.is_empty(),
            "a fresh (unseeded) outbox re-drove nothing, yet B received {got:?}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let v2 = BootCounter::current(&paths.counter_path())
        .expect("v2")
        .expect("v2 minted");
    assert_eq!(
        v2,
        v1 + 1,
        "a genuine restart happened (counter + 1) — the 0-re-drive is not vacuous"
    );
}
