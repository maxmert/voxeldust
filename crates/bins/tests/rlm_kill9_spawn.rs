//! RLM Step 5e-5 PROCESS GATE — the kill-9 crash-safety CAPSTONE of the real-process realm spawner.
//!
//! Drives the REAL orchestrator bin (its `SpawnCore<ProcLaunchBackend>` over a REAL `launch.redb`, forking
//! REAL `vd-shard` processes) through the crash windows the durable write-ahead ledger exists to survive.
//! Because the reconciler is INERT in the bin (`RlmTuning::default()`), a store-test-hooks boot-spawn hook
//! (`VD_RLM_TEST_SPAWN_COORD`) drives exactly ONE real spawn so there is something to crash.
//!
//! THREE arms (each on a DISJOINT, bounded F2 port band via `VD_RLM_FIRST_PORT`/`VD_RLM_PORT_LIMIT`,
//! `--test-threads=1`):
//!  - **D1 (mid-fsync, pre-fork):** `VD_RLM_TEST_LAUNCH_PAUSE` parks the launch.redb writer just before it
//!    fsyncs the v1 write-ahead batch. `spawn_realm` blocks in `flush()` — STRICTLY before `backend.launch()`
//!    — so a SIGKILL here forks NO child (proven DIRECTLY: the child probe port never answers `/whoami`) and
//!    leaves launch.redb EMPTY (the un-fsynced batch is lost). boot-2 rehydrates clean and re-drives the
//!    spawn EXACTLY ONCE at the genesis id — no double-spawn, no F2 id-reuse-of-a-live-child, no orphan.
//!  - **ADOPT (full-anchor survivor):** boot-1 spawns a survivor that boots + serves `/whoami`; the
//!    orchestrator is killed but the child survives; boot-2 rehydrates + ADOPTS it (the `/whoami` cookie
//!    probe), so the boot-spawn idempotency guard sees it already launched ⇒ NO relaunch (one ledger row).
//!  - **D6 (anti-vacuity CONTROL):** the SAME survivor, but boot-2 rebuilds via `water_only`
//!    (`VD_RLM_TEST_REHYDRATE_DISABLE`) ⇒ no adopt ⇒ the guard re-spawns ⇒ TWO rows / two shards for one
//!    realm — the double-spawn the recovery seed prevents (proving rehydrate/adopt is load-bearing).
//!
//! The forked-before-v2 ORPHAN window (v1 fsynced, child forked, crash before the v2 commit) is DELIBERATELY
//! NOT exercised here — D1 sits in the pre-fork window on purpose, so NO orphan is produced. Reaping that
//! residual orphan needs an orphan-head sweep against the armed reconciler; it is ledgered to 5f (D-RLM-5).
//!
//! Gated on `store-test-hooks`: only that build of `vd-orchestrator` carries the boot-spawn + pause hooks.
#![cfg(feature = "store-test-hooks")]

use std::net::{Ipv4Addr, SocketAddr};
use std::process::Child;
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, admin_get_body, common_env, orchestrator_env,
    pid_alive, reserve_tcp_addr, reserve_udp_addr, signal_group, spawn_node,
};
use vd_core::NodeId;
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
use vd_io_prod::store::{RedbStore, StoreTuning};
use vd_io_prod::trust::ClusterTrust;
use vd_node::rlm_spawn::LaunchIntent;
use vd_node::saga_runtime::rlm_launch_prefix;
use vd_sim::io::Store;
use vd_wire::admin::AdminSnapshot;

const DEADLINE: Duration = Duration::from_secs(30);
/// Let the idle launch.redb writer fsync the v2 (pid:Some) confirm before a kill (v2 is block-on-prior, NOT
/// flushed — only v1 is). Generous vs the ms an idle writer actually needs, well under `DEADLINE`.
const V2_FSYNC_SETTLE: Duration = Duration::from_secs(2);

/// A Planet lineage `[Universe, Galaxy(g), System(s), Planet(p)]`. Planet(2,7,7) EXISTS in the default
/// (Walk-scale) seed forest — an arbitrary seed has an empty containment forest and the shard refuses to boot.
fn planet(g: u64, s: u64, p: u64) -> RealmCoord {
    RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::Universe, 0),
        RealmLevel::new(RealmKindTag::Galaxy, g),
        RealmLevel::new(RealmKindTag::System, s),
        RealmLevel::new(RealmKindTag::Planet, p),
    ]))
    .expect("planet path has a leaf")
}

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

fn admin(addr: SocketAddr) -> Option<AdminSnapshot> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str(&body).ok()
}

/// A full `ClusterAddrs` with fresh reserved ports for one orchestrator boot (no gateway/shard peers are
/// spawned — the orchestrator boots standalone, exactly as `orchestrator_crash.rs` does).
fn addrs() -> ClusterAddrs {
    ClusterAddrs {
        orchestrator: reserve_udp_addr(),
        gateway: reserve_udp_addr(),
        shard: reserve_udp_addr(),
        admin: reserve_tcp_addr(),
        gateway_admin: None,
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
    }
}

/// The forked shard's must-parse boot params, added to the ORCHESTRATOR's env so its spawn-anchor harvest
/// forwards them to the child (a shard refuses to boot without them). `common_env` supplies trust + tick-hz.
fn push_shard_anchors(env: &mut Vec<(&'static str, String)>) {
    env.extend([
        ("VD_SNAPSHOT_BUDGET", DEV.snapshot_budget.to_string()),
        ("VD_TICK_DT", DEV.tick_dt.to_string()),
        ("VD_SPEED", DEV.move_speed.to_string()),
        ("VD_MINT_SEED", DEV.mint_seed.to_string()),
        ("VD_INPUT_LOG_CAP", DEV.input_log_cap.to_string()),
    ]);
}

/// Reopen `launch.redb` read-only and decode the durable launch rows to `(node, coord, pid)` — reading back
/// through the SAME frozen postcard shape the spawner wrote. DROPS the store (joins its writer, releases the
/// redb process-exclusive lock) before returning, so a subsequent boot can open it. Caller must ensure NO
/// orchestrator holds the file (kill + reap first).
fn launch_rows(path: &std::path::Path) -> Vec<(NodeId, RealmCoord, Option<u32>)> {
    let (store, _durability) =
        RedbStore::open(path, StoreTuning::default()).expect("reopen launch.redb");
    let rows: Vec<(NodeId, RealmCoord, Option<u32>)> = store
        .scan(&rlm_launch_prefix())
        .into_iter()
        .map(|(_, v)| {
            let i: LaunchIntent = postcard::from_bytes(&v).expect("decode LaunchIntent");
            (i.node, i.coord, i.pid)
        })
        .collect();
    drop(store);
    rows
}

/// Reap every forked shard named by a durable row's confirmed pid — SIGKILL its process group + poll until
/// gone. A row with no pid (v2 not yet durable) has no reapable handle here; the settle wait makes that rare.
fn reap_forked(rows: &[(NodeId, RealmCoord, Option<u32>)]) {
    for (_, _, pid) in rows {
        if let Some(pid) = pid {
            signal_group(*pid, "KILL");
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(5) && pid_alive(*pid) {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn wait_for_marker(marker: &std::path::Path) {
    let start = Instant::now();
    while !marker.exists() {
        assert!(
            start.elapsed() < DEADLINE,
            "the launch.redb writer never parked on the v1 batch (marker absent) — pause not wired?"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn wait_for_admin(addr: SocketAddr) {
    let start = Instant::now();
    while admin(addr).is_none() {
        assert!(
            start.elapsed() < DEADLINE,
            "the orchestrator never served admin (boot-spawn wedged / rehydrate panic?)"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn wait_for_whoami(probe: SocketAddr) {
    let start = Instant::now();
    while admin_get_body(probe, "/whoami", Some(Duration::from_millis(300))).is_none() {
        assert!(
            start.elapsed() < DEADLINE,
            "the forked shard never served /whoami (did not boot?)"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// One test's private trust dir + store paths (the store survives across boots — it IS the crash subject).
struct Fixture {
    trust_dir: std::path::PathBuf,
    common: Vec<(&'static str, String)>,
    store_str: String,
    launch_path: std::path::PathBuf,
    store: std::path::PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.trust_dir);
        let _ = std::fs::remove_file(&self.store);
        let _ = std::fs::remove_file(&self.launch_path);
        let _ = std::fs::remove_file(self.launch_path.with_extension("paused"));
    }
}

fn fixture(tag: &str) -> Fixture {
    let trust = ClusterTrust::generate("vd-rlm-kill9").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-rlm-kill9-{tag}-{}", std::process::id()));
    let trust_dir = base.join("trust");
    std::fs::create_dir_all(&trust_dir).expect("trust dir");
    trust.write_der_dir(&trust_dir).expect("write trust");
    let store = base.join("orchestrator.redb");
    let launch_path = store.with_file_name(vd_bins::LAUNCH_STORE_NAME);
    let _ = std::fs::remove_file(&store);
    let _ = std::fs::remove_file(&launch_path);
    let _ = std::fs::remove_file(launch_path.with_extension("paused"));
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    Fixture {
        store_str: store.display().to_string(),
        trust_dir,
        common,
        launch_path,
        store,
    }
}

/// Build a boot env for the orchestrator: base + shard anchors + the RLM test hooks the caller supplies.
fn orch_env(
    f: &Fixture,
    a: &ClusterAddrs,
    hooks: &[(&'static str, String)],
) -> Vec<(&'static str, String)> {
    let mut env = orchestrator_env(a, &DEV, &f.store_str, ClusterShape::Single);
    push_shard_anchors(&mut env);
    env.extend(hooks.iter().cloned());
    env
}

#[test]
fn d1_sigkill_mid_fsync_forks_no_child_and_rehydrates_clean() {
    let f = fixture("d1");
    let hex = planet(2, 7, 7).path().to_env_string();
    let child_probe: SocketAddr = (Ipv4Addr::LOCALHOST, 42_001).into(); // FIRST_PORT 42000 ⇒ probe 42001

    // ---- boot-1: spawn-coord + PAUSE + first-port. The boot-spawn parks in flush() before the fork. ----
    let a1 = addrs();
    let env1 = orch_env(
        &f,
        &a1,
        &[
            ("VD_RLM_TEST_SPAWN_COORD", hex.clone()),
            ("VD_RLM_FIRST_PORT", "42000".to_string()),
            ("VD_RLM_PORT_LIMIT", "43000".to_string()),
            ("VD_RLM_TEST_LAUNCH_PAUSE", "1".to_string()),
        ],
    );
    let mut orch1 = KillOnDrop(Some(
        spawn_node(env!("CARGO_BIN_EXE_vd-orchestrator"), &f.common, &env1).expect("spawn orch1"),
    ));

    // The writer parked on the v1 batch (the honest pre-fsync signal); admin is NEVER served (the main
    // thread is stuck in flush() during setup), so we poll the marker, not admin.
    let marker = f.launch_path.with_extension("paused");
    wait_for_marker(&marker);

    // DIRECT no-fork witness: the child probe port never binds (flush() sits strictly before backend.launch,
    // so no vd-shard was forked). A pgid scan would be vacuous — a forked shard leads its OWN group.
    assert!(
        admin_get_body(child_probe, "/whoami", Some(Duration::from_millis(300))).is_none(),
        "a vd-shard forked in the pre-fork mid-fsync window (child probe answered /whoami)"
    );

    // SIGKILL + reap the parked orchestrator (releases the launch.redb lock + ports).
    let mut victim = orch1.0.take().expect("orch1 present");
    victim.kill().expect("SIGKILL orch1");
    victim.wait().expect("reap orch1");
    drop(orch1); // disarmed

    // The un-fsynced v1 batch was LOST — launch.redb has NO row, so rehydrate is clean (no orphan, water
    // stays genesis). This is the D1 durability guarantee (flush parks before the fork).
    assert!(
        launch_rows(&f.launch_path).is_empty(),
        "the un-fsynced v1 write-ahead survived the kill — flush did not park pre-fsync?"
    );
    let _ = std::fs::remove_file(&marker);

    // ---- boot-2: SAME store, NO pause ⇒ the boot-spawn re-drives spawn_realm EXACTLY ONCE at genesis. ----
    let a2 = addrs();
    let env2 = orch_env(
        &f,
        &a2,
        &[
            ("VD_RLM_TEST_SPAWN_COORD", hex.clone()),
            ("VD_RLM_FIRST_PORT", "42000".to_string()),
            ("VD_RLM_PORT_LIMIT", "43000".to_string()),
        ],
    );
    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator-boot2",
        spawn_node(env!("CARGO_BIN_EXE_vd-orchestrator"), &f.common, &env2).expect("spawn orch2"),
    );
    wait_for_admin(a2.admin); // proves the boot-spawn completed (the re-drive forked its shard)
    std::thread::sleep(V2_FSYNC_SETTLE); // let the idle writer fsync the v2 (pid:Some) confirm
    drop(cluster); // reap boot-2 orchestrator (the forked shard survives — reaped below)

    // EXACTLY ONE row, minted at the GENESIS id (no F2 id-reuse-of-a-live-child: id 1000 never held a live
    // child, since flush parked before the boot-1 fork), for the requested coord. NO double-spawn.
    let rows = launch_rows(&f.launch_path);
    assert_eq!(
        rows.len(),
        1,
        "boot-2 re-drove the spawn exactly once: {rows:?}"
    );
    assert_eq!(
        rows[0].0,
        NodeId(1_000),
        "the re-drive minted the genesis id"
    );
    assert_eq!(
        rows[0].1,
        planet(2, 7, 7),
        "the re-drive spawned the requested coord"
    );
    assert!(rows[0].2.is_some(), "the completed spawn recorded its pid");
    reap_forked(&rows);
}

#[test]
fn adopt_a_survivor_is_recovered_without_relaunch() {
    let f = fixture("adopt");
    let hex = planet(2, 7, 7).path().to_env_string();
    let child_probe: SocketAddr = (Ipv4Addr::LOCALHOST, 43_001).into(); // FIRST_PORT 43000 ⇒ probe 43001

    // ---- boot-1: NORMAL. The boot-spawn forks a survivor that boots + serves /whoami. ----
    let a1 = addrs();
    let env1 = orch_env(
        &f,
        &a1,
        &[
            ("VD_RLM_TEST_SPAWN_COORD", hex.clone()),
            ("VD_RLM_FIRST_PORT", "43000".to_string()),
            ("VD_RLM_PORT_LIMIT", "44000".to_string()),
        ],
    );
    let mut orch1 = KillOnDrop(Some(
        spawn_node(env!("CARGO_BIN_EXE_vd-orchestrator"), &f.common, &env1).expect("spawn orch1"),
    ));
    wait_for_whoami(child_probe); // the survivor booted
    std::thread::sleep(V2_FSYNC_SETTLE); // its v2 (pid:Some) row is durable before we kill the orchestrator

    // Kill the ORCHESTRATOR ONLY (the survivor shard is a separate process group ⇒ it lives on).
    let mut victim = orch1.0.take().expect("orch1 present");
    victim.kill().expect("SIGKILL orch1");
    victim.wait().expect("reap orch1");
    drop(orch1);

    // ---- boot-2: NORMAL, same store/coord ⇒ rehydrate ADOPTS the survivor (the /whoami cookie probe), so
    //      the boot-spawn idempotency guard sees it already launched ⇒ NO relaunch. ----
    let a2 = addrs();
    let env2 = orch_env(
        &f,
        &a2,
        &[
            ("VD_RLM_TEST_SPAWN_COORD", hex.clone()),
            ("VD_RLM_FIRST_PORT", "43000".to_string()),
            ("VD_RLM_PORT_LIMIT", "44000".to_string()),
        ],
    );
    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator-boot2",
        spawn_node(env!("CARGO_BIN_EXE_vd-orchestrator"), &f.common, &env2).expect("spawn orch2"),
    );
    wait_for_admin(a2.admin);
    drop(cluster); // reap boot-2 orchestrator

    // EXACTLY ONE row (the adopted survivor — no relaunch), at the SAME genesis id boot-1 minted.
    let rows = launch_rows(&f.launch_path);
    assert_eq!(
        rows.len(),
        1,
        "the survivor was adopted, not relaunched (one ledger row): {rows:?}"
    );
    assert_eq!(
        rows[0].0,
        NodeId(1_000),
        "same node id — adopted, not re-minted"
    );
    assert_eq!(rows[0].1, planet(2, 7, 7));
    // The survivor is still the same live incarnation (its /whoami still answers).
    assert!(
        admin_get_body(child_probe, "/whoami", Some(Duration::from_millis(300))).is_some(),
        "the adopted survivor is still alive"
    );
    reap_forked(&rows);
}

#[test]
fn d6_control_an_unseeded_rebuild_double_spawns_a_survivor() {
    let f = fixture("d6");
    let hex = planet(2, 7, 7).path().to_env_string();
    let child_probe: SocketAddr = (Ipv4Addr::LOCALHOST, 44_001).into(); // FIRST_PORT 44000 ⇒ probe 44001

    // ---- boot-1: NORMAL — a survivor boots (same as the ADOPT arm). ----
    let a1 = addrs();
    let env1 = orch_env(
        &f,
        &a1,
        &[
            ("VD_RLM_TEST_SPAWN_COORD", hex.clone()),
            ("VD_RLM_FIRST_PORT", "44000".to_string()),
            ("VD_RLM_PORT_LIMIT", "45000".to_string()),
        ],
    );
    let mut orch1 = KillOnDrop(Some(
        spawn_node(env!("CARGO_BIN_EXE_vd-orchestrator"), &f.common, &env1).expect("spawn orch1"),
    ));
    wait_for_whoami(child_probe);
    std::thread::sleep(V2_FSYNC_SETTLE);
    let mut victim = orch1.0.take().expect("orch1 present");
    victim.kill().expect("SIGKILL orch1");
    victim.wait().expect("reap orch1");
    drop(orch1);

    // ---- boot-2: REHYDRATE-DISABLED (water_only) ⇒ no adopt ⇒ empty seed ⇒ the guard re-spawns. ----
    let a2 = addrs();
    let env2 = orch_env(
        &f,
        &a2,
        &[
            ("VD_RLM_TEST_SPAWN_COORD", hex.clone()),
            ("VD_RLM_FIRST_PORT", "44000".to_string()),
            ("VD_RLM_PORT_LIMIT", "45000".to_string()),
            ("VD_RLM_TEST_REHYDRATE_DISABLE", "1".to_string()),
        ],
    );
    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator-boot2",
        spawn_node(env!("CARGO_BIN_EXE_vd-orchestrator"), &f.common, &env2).expect("spawn orch2"),
    );
    wait_for_admin(a2.admin);
    std::thread::sleep(V2_FSYNC_SETTLE); // the SECOND spawn's v2 row becomes durable
    drop(cluster);

    // TWO rows, DISTINCT node ids, BOTH for the one coord — the double-spawn a `water_only` (unseeded)
    // rebuild produces. This is the falsifiable twin of the ADOPT arm: rehydrate/adopt is load-bearing.
    let rows = launch_rows(&f.launch_path);
    assert_eq!(
        rows.len(),
        2,
        "an unseeded rebuild double-spawned the survivor (two ledger rows): {rows:?}"
    );
    let ids: std::collections::BTreeSet<NodeId> = rows.iter().map(|r| r.0).collect();
    assert_eq!(
        ids.len(),
        2,
        "the two spawns took DISTINCT F2 ids (no id-reuse): {rows:?}"
    );
    assert!(
        rows.iter().all(|r| r.1 == planet(2, 7, 7)),
        "both rows serve the same coord: {rows:?}"
    );
    reap_forked(&rows);
}

#[test]
fn demand_with_static_forest_fails_loud() {
    // RLM 5f-1 SAFETY: an ARMED demand reconciler must NEVER boot atop externally pre-spawned static heads
    // (it would reap them — they carry no demand cell). `orchestrator_env` marks the static-boot mode
    // (VD_STATIC_FOREST), so adding VD_DEMAND is the exact misconfig the XOR gate rejects: the orchestrator
    // must EXIT non-zero and NEVER serve admin (the reconciler is never armed on a bad config).
    let f = fixture("demandxor");
    let a = addrs();
    let env = orch_env(&f, &a, &[("VD_DEMAND", "1".to_string())]);
    let mut child =
        spawn_node(env!("CARGO_BIN_EXE_vd-orchestrator"), &f.common, &env).expect("spawn orch");
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().expect("try_wait orch") {
            break s;
        }
        assert!(
            admin(a.admin).is_none(),
            "the orchestrator served admin despite the VD_DEMAND + VD_STATIC_FOREST misconfig"
        );
        assert!(
            start.elapsed() < DEADLINE,
            "the misconfigured orchestrator never exited (the XOR gate did not fire?)"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(
        !status.success(),
        "boot MUST fail loud on VD_DEMAND + VD_STATIC_FOREST, got {status:?}"
    );
}
