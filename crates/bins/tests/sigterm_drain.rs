//! Cloud-ready k3d Slice 1 — the GATE-WIRED SIGTERM graceful-drain proof.
//!
//! Closes the coverage gap the post-impl review flagged: the drain path was proven only by the manual
//! `just image-drain-smoke` docker recipe (not in `cargo test` / the gate). This spawns a REAL
//! `vd-orchestrator` + `vd-shard`, waits until the cluster bootstraps (a durable directory grant exists),
//! sends **SIGTERM** to the orchestrator (the exact signal a k8s pod-stop / rolling-deploy sends), and
//! asserts:
//!   1. the orchestrator EXITS CLEANLY (code 0) WELL within a grace window — the loop broke, the final
//!      parked outbox flushed, and `RedbStore::Drop` joined the off-tick writer, instead of the old
//!      unconditional loop that only ever died on SIGKILL (a hang would surface as a non-exit → FAIL, the
//!      cargo analog of docker's 137 escalation);
//!   2. re-spawning the orchestrator on the SAME store REHYDRATES the grant — so the drain's fsync is
//!      LOAD-BEARING (a clean exit that lost the store would still fail here).
//!
//! Tier-B process glue. Uses `libc::kill(pid, SIGTERM)` because `std::process::Child` exposes only SIGKILL
//! (`.kill()`); the send is one FFI call, test-only.

use std::process::Child;
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, admin_get_body, common_env, orchestrator_env, reserve_tcp_addr,
    reserve_udp_addr, shard_env, spawn_node,
};
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::AdminSnapshot;

const DEADLINE: Duration = Duration::from_secs(30);
/// The drain must complete WELL inside a real k8s `terminationGracePeriodSeconds`; 15s is generous for an
/// idle solo orchestrator's flush + writer-join (sub-second in practice). Exceeding it ⇒ the drain HUNG.
const DRAIN_GRACE: Duration = Duration::from_secs(15);

fn admin(addr: std::net::SocketAddr) -> Option<AdminSnapshot> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str(&body).ok()
}

/// SIGKILL + reap on drop — panic-safety so a failed assertion before the explicit SIGTERM never leaks the
/// orchestrator child (std `Child` does NOT kill on drop). `take` disarms it for the explicit SIGTERM path.
struct KillOnDrop(Option<Child>);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

#[test]
fn sigterm_drains_the_orchestrator_cleanly_and_the_store_survives() {
    // ---- topology + trust ----
    let orch1 = reserve_udp_addr();
    let gateway = reserve_udp_addr(); // in the book; no gateway spawned (orch+shard alone bootstraps)
    let shard = reserve_udp_addr();
    let admin1 = reserve_tcp_addr();
    let trust_dir = std::env::temp_dir().join(format!("vd-sigterm-{}", std::process::id()));
    let trust = ClusterTrust::generate("vd-sigterm").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    // The DURABLE store survives the SIGTERM drain + restart (NOT removed between boots — it is the subject).
    let store = std::env::temp_dir().join(format!("vd-sigterm-{}.redb", std::process::id()));
    let _ = std::fs::remove_file(&store);
    let store_str = store.display().to_string();
    let common = common_env(&trust_dir.display().to_string(), &DEV);

    // ---- boot 1: orchestrator + shard (the shard grants realm A into the durable directory) ----
    let addrs1 = ClusterAddrs {
        orchestrator: orch1,
        gateway,
        shard,
        admin: admin1,
        orchestrator_probe: reserve_tcp_addr(),
        gateway_probe: reserve_tcp_addr(),
        shard_probe: reserve_tcp_addr(),
        shard_b: reserve_tcp_addr(),
        shard_b_probe: reserve_tcp_addr(),
        galaxy: reserve_udp_addr(),
        galaxy_probe: reserve_tcp_addr(),
        planet: reserve_udp_addr(),
        planet_probe: reserve_tcp_addr(),
        station: reserve_udp_addr(),
        station_probe: reserve_tcp_addr(),
        area: reserve_udp_addr(),
        area_probe: reserve_tcp_addr(),
    };
    let mut orch1_child = KillOnDrop(Some(
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &common,
            &orchestrator_env(&addrs1, &DEV, &store_str, vd_bins::ClusterShape::Single),
        )
        .expect("spawn orch"),
    ));
    let mut cluster = Cluster::new(); // the shard + the restarted orchestrator (RAII-reaped)
    cluster.push(
        "vd-shard",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-shard"),
            &common,
            &shard_env(&addrs1, &DEV, vd_bins::ClusterShape::Single),
        )
        .expect("spawn shard"),
    );

    // ---- wait until the cluster bootstraps: grant A (the shard's realm) is durably in the directory ----
    let a_key = vd_core::pose::RealmId::System(DEV.realm_seed).to_string();
    let started = Instant::now();
    loop {
        if let Some(s) = admin(admin1)
            && s.cluster_bootstrapped()
            && s.directory.iter().any(|e| e.key == a_key)
        {
            break;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "cluster never bootstrapped (grant A {a_key} absent) — orch/shard mesh up?"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    // ---- SIGTERM the orchestrator; it MUST drain + exit 0 WELL within grace (not hang, not crash) ----
    let mut victim = orch1_child.0.take().expect("orch1 child present");
    let pid = victim.id() as libc::pid_t;
    // SAFETY: `pid` is a live child we own (not yet reaped); SIGTERM is a valid signal. One FFI call.
    let rc = unsafe { libc::kill(pid, libc::SIGTERM) };
    assert_eq!(
        rc,
        0,
        "libc::kill(SIGTERM) failed: {}",
        std::io::Error::last_os_error()
    );
    let sent = Instant::now();
    let status = loop {
        match victim
            .try_wait()
            .expect("try_wait on the orchestrator child")
        {
            Some(status) => break status,
            None => {
                assert!(
                    sent.elapsed() < DRAIN_GRACE,
                    "the orchestrator did not exit within {DRAIN_GRACE:?} of SIGTERM — the graceful drain \
                     HUNG (in k8s this would be the SIGKILL escalation, the exact regression this guards)"
                );
                std::thread::sleep(Duration::from_millis(25));
            }
        }
    };
    drop(orch1_child); // disarmed (None) — no double-kill
    assert!(
        status.success(),
        "SIGTERM must drain the orchestrator CLEANLY (exit 0); got {status:?}. A signal/non-zero exit means \
         the drain path failed — the whole point of Slice 1 is that a routine pod-stop is NOT a crash-path"
    );

    // ---- boot 2: re-spawn on the SAME store; grant A must REHYDRATE (the drain's fsync was load-bearing) ----
    let orch2 = reserve_udp_addr();
    let admin2 = reserve_tcp_addr();
    let addrs2 = ClusterAddrs {
        orchestrator: orch2,
        gateway,
        shard,
        admin: admin2,
        orchestrator_probe: reserve_tcp_addr(),
        gateway_probe: reserve_tcp_addr(),
        shard_probe: reserve_tcp_addr(),
        shard_b: reserve_tcp_addr(),
        shard_b_probe: reserve_tcp_addr(),
        galaxy: reserve_udp_addr(),
        galaxy_probe: reserve_tcp_addr(),
        planet: reserve_udp_addr(),
        planet_probe: reserve_tcp_addr(),
        station: reserve_udp_addr(),
        station_probe: reserve_tcp_addr(),
        area: reserve_udp_addr(),
        area_probe: reserve_tcp_addr(),
    };
    cluster.push(
        "vd-orchestrator-restart",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &common,
            &orchestrator_env(&addrs2, &DEV, &store_str, vd_bins::ClusterShape::Single),
        )
        .expect("respawn orchestrator"),
    );
    let _guard = cluster;

    let restart_started = Instant::now();
    let recovered = loop {
        if let Some(s) = admin(admin2) {
            break s;
        }
        assert!(
            restart_started.elapsed() < DEADLINE,
            "the restarted orchestrator never served admin (rehydrate panic / wedge on the drained store?)"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(
        recovered.directory.iter().any(|e| e.key == a_key),
        "grant A ({a_key}) must REHYDRATE from the SIGTERM-drained store — the drain's final fsync is \
         load-bearing, not just a clean exit: {:?}",
        recovered.directory
    );

    // cleanup (the Cluster RAII reaps the children; drop the durable artifacts).
    let _ = std::fs::remove_file(&store);
    let _ = std::fs::remove_dir_all(&trust_dir);
}
