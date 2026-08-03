//! D-6 Slice D-delta — the orchestrator's HR1 ephemeral-store boot guard (always compiled; no feature).
//!
//! `VD_STORE_PATH` is REQUIRED and a temp-dir path is REJECTED LOUD unless the explicit dev/test opt-in
//! `VD_STORE_EPHEMERAL_OK` is set — the production-safety net against the old `/tmp/{shard_id}` data-loss
//! bug. This locks BOTH arms of that guard (reject without the opt-in, accept with it), so a refactor that
//! silently inverts the condition fails the gate.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, admin_get_body, common_env, orchestrator_env, spawn_node,
};
use vd_io_prod::trust::ClusterTrust;

fn addrs() -> ClusterAddrs {
    ClusterAddrs::reserve()
}

#[test]
fn a_temp_store_without_ephemeral_ok_refuses_to_boot() {
    let addrs = addrs();
    let trust_dir = std::env::temp_dir().join(format!("vd-bootreject-{}", std::process::id()));
    ClusterTrust::generate("vd-bootreject")
        .expect("trust")
        .write_der_dir(&trust_dir)
        .expect("trust dir");
    // A temp-dir store path (the guard's reject target) — with the dev escape STRIPPED.
    let temp_store =
        std::env::temp_dir().join(format!("vd-bootreject-{}.redb", std::process::id()));
    let mut node_env = orchestrator_env(
        &addrs,
        &DEV,
        &temp_store.display().to_string(),
        vd_bins::ClusterShape::Single,
    );
    node_env.retain(|(k, _)| *k != "VD_STORE_EPHEMERAL_OK");
    let common = common_env(&trust_dir.display().to_string(), &DEV);

    // A REAL reserved VD_BIND so the mesh binds and execution actually REACHES the store guard (a garbage
    // bind would non-zero-exit for the WRONG reason — guarded against by the stderr-reason assertions).
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vd-orchestrator"));
    for (k, v) in common.iter().chain(node_env.iter()) {
        cmd.env(k, v);
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::piped());
    let output = cmd
        .spawn()
        .expect("spawn orchestrator")
        .wait_with_output()
        .expect("wait orchestrator");

    assert!(
        !output.status.success(),
        "the orchestrator MUST exit non-zero on a temp store with no VD_STORE_EPHEMERAL_OK"
    );
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(
        err.contains("Refusing to boot"),
        "the ephemeral guard must reject with its message; stderr: {err}"
    );
    // Rigor: prove it was the STORE GUARD, not an earlier trust/bind/config failure.
    assert!(
        !err.contains("VD_TRUST_DIR") && !err.contains("VD_BIND"),
        "rejected for the wrong reason (not the store guard); stderr: {err}"
    );
    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&temp_store);
}

#[test]
fn a_temp_store_with_ephemeral_ok_boots_past_the_guard() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // The ACCEPT arm: `orchestrator_env` keeps VD_STORE_EPHEMERAL_OK=1, so a temp store boots and serves
    // admin (the guard's accept + the "under a TEMP dir … dev/test" warn ran).
    let addrs = addrs();
    let admin_addr = addrs.admin;
    let trust_dir = std::env::temp_dir().join(format!("vd-bootaccept-{}", std::process::id()));
    ClusterTrust::generate("vd-bootaccept")
        .expect("trust")
        .write_der_dir(&trust_dir)
        .expect("trust dir");
    let store = std::env::temp_dir().join(format!("vd-bootaccept-{}.redb", std::process::id()));
    let _ = std::fs::remove_file(&store);
    let common = common_env(&trust_dir.display().to_string(), &DEV);
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
    let started = Instant::now();
    let mut served = false;
    while started.elapsed() < Duration::from_secs(10) {
        if admin_get_body(admin_addr, "/admin/snapshot", Some(Duration::from_secs(2))).is_some() {
            served = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        served,
        "with VD_STORE_EPHEMERAL_OK a temp-store orchestrator must boot past the guard + serve admin"
    );
    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&store);
}
