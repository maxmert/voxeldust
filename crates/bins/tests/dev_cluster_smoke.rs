//! Slice-0 process-tier smoke: the LOCAL-PROCESS dev cluster (`vd-devcluster`,
//! no k3d) comes UP — the stub shard grants its realm to the directory over real
//! QUIC under mTLS, which is what `up` waits for — and tears DOWN WITHOUT LEAKING:
//! every node process is dead, the workdir is gone, and the ports are bindable
//! again. A second case proves a SIGKILL-mid-claim crash state is always
//! recoverable (no wedged slot). This is the foundation the HR6 agent loop
//! (`dev-cluster.sh` + `vdctl`) and the G-RENDER-SMOKE gate stand on.

use std::net::UdpSocket;
use std::process::{Command, Stdio};

// The slot constants, the on-disk layout, the launcher runner, and the down-on-drop guard
// are the SHARED vd_bins definitions (one registry/layout for every process-tier test —
// they can never drift from the launcher or collide with each other).
use vd_bins::{
    DEMAND_SMOKE_SLOT, DEV, DevClusterDown, RECOVERY_SLOT, SMOKE_SLOT, common_env, devcluster,
    slot_runfile, slot_workdir,
};
use vd_devproto::DevPortScheme;

/// The `VD_PROCESS_INCARNATION` value from a rendered `common_env`.
fn incarnation_of(env: &[(&'static str, String)]) -> u64 {
    env.iter()
        .find(|(k, _)| *k == "VD_PROCESS_INCARNATION")
        .and_then(|(_, v)| v.parse().ok())
        .expect("common_env exports VD_PROCESS_INCARNATION")
}

/// The node PIDs recorded in the runfile (empty if not up).
fn recorded_pids(slot: u16) -> Vec<u32> {
    std::fs::read_to_string(slot_runfile(slot))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .collect()
}

fn alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn run(launcher: &str, sub: &str, slot: u16) -> std::process::ExitStatus {
    devcluster(launcher, sub, slot)
}

#[test]
fn dev_cluster_comes_up_over_quic_and_tears_down_without_leaking() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let _ = run(launcher, "down", SMOKE_SLOT); // clean slate (idempotent)
    let _guard = DevClusterDown::new(launcher, SMOKE_SLOT);

    // `up` exits 0 ONLY after the shard granted its realm (the QUIC mesh works).
    assert!(
        run(launcher, "up", SMOKE_SLOT).success(),
        "dev-cluster up should reach ready and exit 0"
    );

    // All three node PIDs are recorded (the kill record) and alive.
    let pids = recorded_pids(SMOKE_SLOT);
    assert_eq!(pids.len(), 3, "three node PIDs recorded, got {pids:?}");
    assert!(
        pids.iter().all(|p| alive(*p)),
        "all node processes alive while up: {pids:?}"
    );

    // `status` reports the live processes + a ready admin.
    let status = Command::new(launcher)
        .args(["status", "--slot", &SMOKE_SLOT.to_string()])
        .output()
        .expect("run status");
    let out = String::from_utf8_lossy(&status.stdout);
    assert!(out.contains("3/3"), "status shows 3/3 alive, got: {out}");
    assert!(out.contains("READY"), "status should be READY, got: {out}");

    // `down` succeeds.
    assert!(
        run(launcher, "down", SMOKE_SLOT).success(),
        "down should succeed"
    );

    // PROVE no leak: every recorded process is dead, the workdir is gone, and the
    // QUIC ports are bindable again (no orphan holding them).
    assert!(
        pids.iter().all(|p| !alive(*p)),
        "every node process must be dead after down: {pids:?}"
    );
    assert!(
        !slot_workdir(SMOKE_SLOT).exists(),
        "the workdir must be removed after down"
    );
    let ports = DevPortScheme::DEFAULT
        .slot_ports(SMOKE_SLOT)
        .expect("ports");
    for port in [ports.orchestrator, ports.gateway, ports.shard] {
        assert!(
            UdpSocket::bind(("127.0.0.1", port)).is_ok(),
            "QUIC port {port} must be free after down (no leaked node)"
        );
    }
}

#[test]
fn demand_cluster_up_boots_orchestrator_and_gateway_only_and_reaches_ready() {
    // RLM demand-walk (VU): `up --demand` stands up the demand cluster (orchestrator + gateway, NO pre-booked
    // shard) — the exact cluster the rlm_demand_login/demand-walk proofs build in-harness, now LAUNCHABLE so a
    // windowed client can fly it. Readiness is "the orchestrator is up + serving" (a demand cluster has NO
    // static realm to wait for — its worlds spin up on login/AoI demand, proven by rlm_demand_login).
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let slot = DEMAND_SMOKE_SLOT.to_string();
    let _ = Command::new(launcher)
        .args(["down", "--slot", &slot])
        .status(); // clean slate (idempotent)
    let _guard = DevClusterDown::new(launcher, DEMAND_SMOKE_SLOT);

    // `up --demand` exits 0 once the orchestrator is up + serving (no shard needs to grant a realm first).
    let status = Command::new(launcher)
        .args(["up", "--demand", "--slot", &slot])
        .status()
        .expect("run up --demand");
    assert!(
        status.success(),
        "up --demand should reach ready and exit 0"
    );

    // Exactly TWO node PIDs recorded (orchestrator + gateway) — NO static login shard was booked.
    let pids = recorded_pids(DEMAND_SMOKE_SLOT);
    assert_eq!(
        pids.len(),
        2,
        "a demand cluster is orchestrator + gateway ONLY (no pre-booked shard), got {pids:?}"
    );
    assert!(
        pids.iter().all(|p| alive(*p)),
        "both demand nodes alive while up: {pids:?}"
    );
    // Teardown + leak-freedom is proven by DevClusterDown (drop reaps) + the Single up/down test above.
}

#[test]
fn two_launches_export_a_strictly_increasing_process_incarnation() {
    // R-3' precondition: a node that restarts while a peer keeps running must come up at a STRICTLY HIGHER
    // VD_PROCESS_INCARNATION than the peer's surviving mesh ledger holds, or its seq-reset-to-0 frames are
    // silently deduped. `common_env` stamps a per-launch wall-clock-ms incarnation; two launches must differ.
    let first = incarnation_of(&common_env("trust", &DEV));
    std::thread::sleep(std::time::Duration::from_millis(2));
    let second = incarnation_of(&common_env("trust", &DEV));
    assert!(
        first > 0 && second > first,
        "launch incarnation must be monotone and nonzero: {first} -> {second}"
    );
}

#[test]
fn a_sigkill_mid_claim_crash_state_is_always_recoverable() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // A SIGKILL/Ctrl-C/OOM between the O_EXCL claim and the first recorded pid
    // leaves the runfile present but EMPTY (no pids). This must NOT wedge the slot:
    // `down` clears it and a fresh `up` succeeds. (The runfile IS the claim, so
    // there is no separate lock to strand.) Regression test for the audited
    // lock-before-runfile wedge.
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let _ = run(launcher, "down", RECOVERY_SLOT); // clean slate
    let _guard = DevClusterDown::new(launcher, RECOVERY_SLOT);

    // Forge the crash state: workdir + empty runfile, no pids.
    let wd = slot_workdir(RECOVERY_SLOT);
    std::fs::create_dir_all(&wd).expect("mk workdir");
    std::fs::write(slot_runfile(RECOVERY_SLOT), "").expect("empty runfile");

    // `down` must recover it (not print "already down" while leaving it wedged).
    assert!(
        run(launcher, "down", RECOVERY_SLOT).success(),
        "down must clear a claim-only crash state"
    );
    assert!(!wd.exists(), "down must remove the stranded workdir");

    // A fresh `up` must now succeed — the slot is NOT stuck on a stale claim.
    assert!(
        run(launcher, "up", RECOVERY_SLOT).success(),
        "up must succeed after the crash state was cleared"
    );
    assert_eq!(
        recorded_pids(RECOVERY_SLOT).len(),
        3,
        "up records all three pids"
    );
    assert!(
        run(launcher, "down", RECOVERY_SLOT).success(),
        "final teardown"
    );
}
