//! Slice-0 process-tier smoke: the LOCAL-PROCESS dev cluster (`vd-devcluster`,
//! no k3d) comes UP — the stub shard grants its realm to the directory over real
//! QUIC under mTLS, which is what `up` waits for — and tears DOWN WITHOUT LEAKING:
//! every node process is dead, the workdir is gone, and the ports are bindable
//! again. A second case proves a SIGKILL-mid-claim crash state is always
//! recoverable (no wedged slot). This is the foundation the HR6 agent loop
//! (`dev-cluster.sh` + `vdctl`) and the G-RENDER-SMOKE gate stand on.

use std::net::UdpSocket;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use vd_devproto::{DevPortScheme, WORKTREE_SLOT_CEILING};

/// Test-reserved slots ABOVE the worktree auto-derivation ceiling (so they can
/// never coincide with a developer's running `dev-cluster.sh up` for some
/// worktree — whose slot is always `< WORKTREE_SLOT_CEILING` — and the test's
/// pre-clean `down` can't clobber it), yet still below the high range Docker grabs
/// on macOS (10000+; these slots map to ports 9560-9623).
const SMOKE_SLOT: u16 = WORKTREE_SLOT_CEILING + 16; // 80
const RECOVERY_SLOT: u16 = WORKTREE_SLOT_CEILING + 17; // 81

fn workdir(slot: u16) -> PathBuf {
    std::env::temp_dir()
        .join("vd-devcluster")
        .join(format!("slot-{slot}"))
}

/// The node PIDs recorded in the runfile (empty if not up).
fn recorded_pids(slot: u16) -> Vec<u32> {
    std::fs::read_to_string(workdir(slot).join("cluster.pids"))
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
    Command::new(launcher)
        .args([sub, "--slot", &slot.to_string()])
        .status()
        .unwrap_or_else(|e| panic!("run {sub}: {e}"))
}

/// Tear the cluster down even if an assertion panics — never leak child processes.
struct DownGuard(u16);
impl Drop for DownGuard {
    fn drop(&mut self) {
        let _ = run(env!("CARGO_BIN_EXE_vd-devcluster"), "down", self.0);
    }
}

#[test]
fn dev_cluster_comes_up_over_quic_and_tears_down_without_leaking() {
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let _ = run(launcher, "down", SMOKE_SLOT); // clean slate (idempotent)
    let _guard = DownGuard(SMOKE_SLOT);

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
        !workdir(SMOKE_SLOT).exists(),
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
fn a_sigkill_mid_claim_crash_state_is_always_recoverable() {
    // A SIGKILL/Ctrl-C/OOM between the O_EXCL claim and the first recorded pid
    // leaves the runfile present but EMPTY (no pids). This must NOT wedge the slot:
    // `down` clears it and a fresh `up` succeeds. (The runfile IS the claim, so
    // there is no separate lock to strand.) Regression test for the audited
    // lock-before-runfile wedge.
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let _ = run(launcher, "down", RECOVERY_SLOT); // clean slate
    let _guard = DownGuard(RECOVERY_SLOT);

    // Forge the crash state: workdir + empty runfile, no pids.
    let wd = workdir(RECOVERY_SLOT);
    std::fs::create_dir_all(&wd).expect("mk workdir");
    std::fs::write(wd.join("cluster.pids"), "").expect("empty runfile");

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
