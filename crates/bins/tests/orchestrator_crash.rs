//! D-6 Slice D-delta — the process-tier SIGKILL-mid-fsync crash proof.
//!
//! A real `kill -9` of the RedbStore-backed orchestrator WHILE a directory grant's batch sits SUBMITTED
//! but NOT yet fsynced must lose AT MOST ONE batch and recover to a CONSISTENT, forward-only state — no
//! corruption, no zombie authority, the clock resumes forward. This is the capstone that proves the C2
//! off-tick-fsync + block-on-prior durability under a genuine crash (not just a graceful drop).
//!
//! Determinism (the hard part): the group-commit barrier stages the Clock key EVERY tick, so the durable
//! batch seq ≈ tick and never quiesces — a seq-number pause would be racy. Instead the orchestrator (under
//! `store-test-hooks`, via `VD_STORE_TEST_SENTINEL_SEED`) plants a DISTINCT realm grant (B) once the
//! shard's realm grant (A) is already in a PRIOR tick's batch; by block-on-prior, A is durable BEFORE B's
//! batch is submitted. The off-tick writer then PARKS on B's batch BEFORE fsync (content-keyed, not
//! seq-keyed) and writes a marker file — an honest, decoupled "submitted-but-pre-fsync window is open"
//! signal written by the WRITER thread itself (the bin loop is, by then, blocked on the persist-before-
//! effect gate waiting for B to become durable, so it cannot signal). The test SIGKILLs in that window.
//!
//! Gated on `store-test-hooks`: only that build of `vd-orchestrator` carries the pause hook.
#![cfg(feature = "store-test-hooks")]

use std::process::Child;
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, admin_get_body, common_env, orchestrator_env, reserve_tcp_addr,
    reserve_udp_addr, shard_env, spawn_node,
};
use vd_core::pose::RealmId;
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::AdminSnapshot;

const DEADLINE: Duration = Duration::from_secs(30);
/// Disjoint from `DEV.realm_seed` (7) and the saga-test seeds (5, 8) — the planted, must-be-LOST grant B.
const SENTINEL_SEED: u64 = 0x0D6D_E17A;

/// SIGKILL + REAP on drop — panic-safety so a failed assertion before the explicit kill never leaks the
/// orchestrator child (std `Child` does NOT kill on drop). `take` disarms it for the explicit kill+reap.
struct KillOnDrop(Option<Child>);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

fn admin(addr: std::net::SocketAddr) -> Option<AdminSnapshot> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str(&body).ok()
}

#[test]
fn sigkill_mid_fsync_loses_at_most_one_batch_and_recovers_consistently() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // ---- topology + trust ----
    let orch1 = reserve_udp_addr();
    let gateway = reserve_udp_addr();
    let shard = reserve_udp_addr();
    let admin1 = reserve_tcp_addr();
    let trust_dir = std::env::temp_dir().join(format!("vd-orchcrash-{}", std::process::id()));
    let trust = ClusterTrust::generate("vd-orchcrash").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    // The DURABLE store survives the kill-9 + restart (NOT removed between boots — it IS the crash subject).
    let store = std::env::temp_dir().join(format!("vd-orchcrash-{}.redb", std::process::id()));
    let _ = std::fs::remove_file(&store);
    let marker = store.with_extension("paused");
    let _ = std::fs::remove_file(&marker);
    let store_str = store.display().to_string();
    let common = common_env(&trust_dir.display().to_string(), &DEV);

    // ---- boot 1: orchestrator (sentinel-armed) + shard (grants realm A) ----
    let addrs1 = ClusterAddrs {
        orchestrator: orch1,
        gateway,
        shard,
        admin: admin1,
        ..ClusterAddrs::reserve()
    };
    let mut orch_env = orchestrator_env(&addrs1, &DEV, &store_str, vd_bins::ClusterShape::Single);
    orch_env.push(("VD_STORE_TEST_SENTINEL_SEED", SENTINEL_SEED.to_string()));
    let mut orch1_child = KillOnDrop(Some(
        spawn_node(env!("CARGO_BIN_EXE_vd-orchestrator"), &common, &orch_env).expect("spawn orch"),
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

    // ---- wait for the writer to PARK on the sentinel batch (the decoupled, honest window signal) ----
    let started = Instant::now();
    let mut pre_kill_tick = 0u64;
    while !marker.exists() {
        if let Some(s) = admin(admin1) {
            pre_kill_tick = s.universe_tick;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "the writer never parked on the sentinel batch (marker absent) — sentinel not planted/staged?"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    // Pre-kill anti-vacuity: grant A (a shard realm) IS present live, AND the sentinel B IS present, so
    // "B absent after restart" proves the LOSS of a real row, not the absence of a never-existent one.
    let live = admin(admin1).expect("admin reachable while the sim thread is parked");
    let a_key = RealmId::System(DEV.realm_seed).to_string();
    let b_key = RealmId::System(SENTINEL_SEED).to_string();
    assert!(
        live.cluster_bootstrapped() && live.directory.iter().any(|e| e.key == a_key),
        "grant A (shard realm {a_key}) present pre-kill: {:?}",
        live.directory
    );
    assert!(
        live.directory.iter().any(|e| e.key == b_key),
        "grant B (sentinel realm {b_key}) staged-and-visible pre-kill: {:?}",
        live.directory
    );

    // ---- SIGKILL the orchestrator mid-fsync, then REAP (releases the redb file lock + ports) ----
    let mut victim = orch1_child.0.take().expect("orch1 child present");
    victim.kill().expect("SIGKILL the orchestrator");
    victim.wait().expect("reap the killed orchestrator");
    drop(orch1_child); // disarmed (None) — no double-kill

    // ---- boot 2: restart on the SAME store, FRESH addrs, NO sentinel (clean recovery) ----
    let orch2 = reserve_udp_addr();
    let admin2 = reserve_tcp_addr();
    let addrs2 = ClusterAddrs {
        orchestrator: orch2,
        gateway,
        shard,
        admin: admin2,
        ..ClusterAddrs::reserve()
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

    // ---- poll the restarted admin: a 200 proves it BOOTED CLEAN (rehydrate did not panic / wedge) ----
    let restart_started = Instant::now();
    let recovered = loop {
        if let Some(s) = admin(admin2) {
            break s;
        }
        assert!(
            restart_started.elapsed() < DEADLINE,
            "the restarted orchestrator never served admin (rehydrate panic / wedge on the recovered store?)"
        );
        std::thread::sleep(Duration::from_millis(50));
    };

    // (a) grant A SURVIVED the kill-9.
    assert!(
        recovered.cluster_bootstrapped() && recovered.directory.iter().any(|e| e.key == a_key),
        "grant A (shard realm {a_key}) LOST across kill-9: {:?}",
        recovered.directory
    );
    // (b) grant B (the in-flight, never-fsynced batch) was LOST — the ≤1-batch-loss claim.
    assert!(
        !recovered.directory.iter().any(|e| e.key == b_key),
        "grant B (sentinel realm {b_key}) should be LOST (paused pre-fsync at kill): {:?}",
        recovered.directory
    );
    // (c) the clock resumed FORWARD (recover jumps to the persisted ceiling, never rewinds) and keeps ticking.
    assert!(
        recovered.universe_tick >= pre_kill_tick,
        "clock rewound across recovery: {} < {pre_kill_tick}",
        recovered.universe_tick
    );
    let later = loop {
        if let Some(s) = admin(admin2)
            && s.universe_tick > recovered.universe_tick
        {
            break s;
        }
        assert!(
            restart_started.elapsed() < DEADLINE.saturating_mul(2),
            "the recovered clock did not resume ticking forward"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(later.universe_tick > recovered.universe_tick);
    // (d) CONSISTENT recovery: no zombie saga from the lost grant, no orphan transfer lock.
    assert!(
        recovered.sagas.is_empty(),
        "no zombie saga should survive the lost in-flight grant: {:?}",
        recovered.sagas
    );
    assert!(
        recovered.directory.iter().all(|e| e.in_transfer.is_none()),
        "no orphan transfer lock on a recovered row: {:?}",
        recovered.directory
    );

    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&store);
    let _ = std::fs::remove_file(&marker);
}

#[test]
fn grant_a_recovery_assertion_fails_against_a_fresh_store() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // ANTI-THEATER twin: the SAME `cluster_bootstrapped()` predicate the recovery test relies on for
    // "grant A survived" must be FALSE on a fresh store with NO shard granting a realm — so the recovery
    // assertion is falsifiable (it fails exactly when A was never durable), not vacuous.
    let orch = reserve_udp_addr();
    let gateway = reserve_udp_addr();
    let shard = reserve_udp_addr();
    let admin_addr = reserve_tcp_addr();
    let trust_dir = std::env::temp_dir().join(format!("vd-orchcrash-fresh-{}", std::process::id()));
    let trust = ClusterTrust::generate("vd-orchcrash-fresh").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    let store =
        std::env::temp_dir().join(format!("vd-orchcrash-fresh-{}.redb", std::process::id()));
    let _ = std::fs::remove_file(&store);
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    let addrs = ClusterAddrs {
        orchestrator: orch,
        gateway,
        shard,
        admin: admin_addr,
        ..ClusterAddrs::reserve()
    };
    // NO shard ⇒ no realm is ever granted.
    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &common,
            &orchestrator_env(
                &addrs,
                &DEV,
                &store.display().to_string(),
                vd_bins::ClusterShape::Single,
            ),
        )
        .expect("spawn orch"),
    );
    let _guard = cluster;
    // It must serve admin AND never report a bootstrapped cluster.
    let started = Instant::now();
    let mut saw_admin = false;
    while started.elapsed() < Duration::from_secs(4) {
        if let Some(s) = admin(admin_addr) {
            saw_admin = true;
            assert!(
                !s.cluster_bootstrapped(),
                "no shard ⇒ the cluster must NOT be bootstrapped: {:?}",
                s.directory
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(saw_admin, "the orchestrator served admin at least once");
    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&store);
}
