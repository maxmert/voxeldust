//! Cloud-ready k3d Slice 2 — the PROCESS-tier proof that the cloud footgun preflight is WIRED LIVE into the
//! server bins (not merely unit-tested in `vd-io-prod`). A bin that forgot to call `enforce_cloud_preflight`
//! / `resolve_node_d3` at boot would come up GREEN with `VD_PROFILE=cloud` + a footgun set — the exact
//! silent-fail-open the DEFERRED.md:984 split-brain hole is about. These tests assert the real binary instead
//! exits NON-ZERO with its actionable Display guidance in stderr (the same fail-loud posture as the HR1 store
//! guard, `boot_guard.rs`). The four negatives cover each preflight arm, in order; the last is GATEWAY-only —
//! it proves the gateway's `Some(dev_auth_pubkey_bytes())` wiring is live (an orchestrator/shard passing
//! `None` cannot trigger it). The positive control proves cloud mode is not accidentally always-failing: a
//! COHERENT cloud config boots the real orchestrator and serves admin.
//!
//! A watchdog bounds every spawn: if a (mis-wired) bin runs forever instead of refusing to boot, the child is
//! killed and `run_to_exit` returns `None`, so the test FAILS cleanly rather than hanging the whole suite.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, admin_get_body, common_env, dev_auth_pubkey_hex, gateway_env,
    orchestrator_env, spawn_node,
};
use vd_io_prod::trust::ClusterTrust;

fn addrs() -> ClusterAddrs {
    ClusterAddrs::reserve()
}

/// A trust bundle under `$TMPDIR` (a real DER dir every node needs before it reaches the preflight/store).
fn write_trust(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("vd-cloudpf-{tag}-{}", std::process::id()));
    ClusterTrust::generate("vd-cloudpf")
        .expect("trust")
        .write_der_dir(&dir)
        .expect("trust dir");
    dir
}

/// A scratch dir under the cargo target dir — a REAL (non-`$TMPDIR`) path, so the cloud preflight's
/// under-temp durable-root rejection does NOT fire for a legitimately-persistent volume. Unique per tag+pid.
fn non_temp_dir(tag: &str) -> std::path::PathBuf {
    let d = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("cloud-preflight-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("create non-temp scratch dir");
    d
}

/// Spawn `bin` with `envs`, wait for it to exit ON ITS OWN, and return `(success, stderr)`. A watchdog kills
/// a child that runs past the deadline and returns `None` — so a bin that boots GREEN when it should refuse
/// makes the caller's `.expect(...)` fail loudly instead of hanging the suite forever.
fn run_to_exit(bin: &str, envs: &[(&'static str, String)]) -> Option<(bool, String)> {
    let mut cmd = Command::new(bin);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn bin");
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if child.try_wait().expect("try_wait").is_some() {
            let out = child.wait_with_output().expect("wait_with_output");
            return Some((
                out.status.success(),
                String::from_utf8_lossy(&out.stderr).into_owned(),
            ));
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Preflight arm 1 (ephemeral escape). The dev `orchestrator_env` sets `VD_STORE_EPHEMERAL_OK=1` — in cloud
/// that is the FIRST footgun the preflight vetoes. Just selecting `VD_PROFILE=cloud` on the standard dev env
/// must refuse to boot.
#[test]
fn orchestrator_cloud_rejects_the_ephemeral_store_escape() {
    let addrs = addrs();
    let trust = write_trust("eph");
    let store = std::env::temp_dir().join(format!("vd-cloudpf-eph-{}.redb", std::process::id()));
    let mut envs = common_env(&trust.display().to_string(), &DEV);
    envs.extend(orchestrator_env(
        &addrs,
        &DEV,
        &store.display().to_string(),
        vd_bins::ClusterShape::Single,
    ));
    envs.push(("VD_PROFILE", "cloud".to_owned()));

    let (ok, err) = run_to_exit(env!("CARGO_BIN_EXE_vd-orchestrator"), &envs)
        .expect("orchestrator must EXIT (refuse to boot), not run forever");
    assert!(
        !ok,
        "cloud + VD_STORE_EPHEMERAL_OK must refuse to boot; stderr: {err}"
    );
    assert!(
        err.contains("cloud profile forbids") && err.contains("VD_STORE_EPHEMERAL_OK"),
        "must reject with the actionable ephemeral-escape guidance (not a bare Debug dump); stderr: {err}"
    );
    let _ = std::fs::remove_dir_all(&trust);
}

/// Preflight arm 2 (manual incarnation). Strip the earlier ephemeral footgun so we REACH the incarnation
/// check; `common_env` supplies the forbidden `VD_PROCESS_INCARNATION` (the durable M3 counter is mandatory
/// in cloud).
#[test]
fn orchestrator_cloud_rejects_a_manual_process_incarnation() {
    let addrs = addrs();
    let trust = write_trust("inc");
    let store = std::env::temp_dir().join(format!("vd-cloudpf-inc-{}.redb", std::process::id()));
    let mut envs = common_env(&trust.display().to_string(), &DEV);
    envs.extend(orchestrator_env(
        &addrs,
        &DEV,
        &store.display().to_string(),
        vd_bins::ClusterShape::Single,
    ));
    envs.retain(|(k, _)| *k != "VD_STORE_EPHEMERAL_OK");
    envs.push(("VD_PROFILE", "cloud".to_owned()));

    let (ok, err) = run_to_exit(env!("CARGO_BIN_EXE_vd-orchestrator"), &envs)
        .expect("orchestrator must EXIT (refuse to boot), not run forever");
    assert!(
        !ok,
        "cloud + a manual VD_PROCESS_INCARNATION must refuse to boot; stderr: {err}"
    );
    assert!(
        err.contains("cloud profile forbids") && err.contains("VD_PROCESS_INCARNATION"),
        "must reject the manual-incarnation footgun with guidance; stderr: {err}"
    );
    let _ = std::fs::remove_dir_all(&trust);
}

/// Preflight arm 3 (missing durable root). Strip BOTH earlier footguns so we reach the durable-root check;
/// declare no `VD_STORE_DURABLE_ROOT` — a cloud pod MUST name its persistent volume.
#[test]
fn orchestrator_cloud_requires_a_durable_root() {
    let addrs = addrs();
    let trust = write_trust("root");
    let store = std::env::temp_dir().join(format!("vd-cloudpf-root-{}.redb", std::process::id()));
    let mut envs = common_env(&trust.display().to_string(), &DEV);
    envs.extend(orchestrator_env(
        &addrs,
        &DEV,
        &store.display().to_string(),
        vd_bins::ClusterShape::Single,
    ));
    envs.retain(|(k, _)| *k != "VD_STORE_EPHEMERAL_OK" && *k != "VD_PROCESS_INCARNATION");
    envs.push(("VD_PROFILE", "cloud".to_owned()));

    let (ok, err) = run_to_exit(env!("CARGO_BIN_EXE_vd-orchestrator"), &envs)
        .expect("orchestrator must EXIT (refuse to boot), not run forever");
    assert!(
        !ok,
        "cloud without VD_STORE_DURABLE_ROOT must refuse to boot; stderr: {err}"
    );
    assert!(
        err.contains("VD_STORE_DURABLE_ROOT"),
        "must reject the missing durable root with guidance; stderr: {err}"
    );
    let _ = std::fs::remove_dir_all(&trust);
}

/// Preflight arm 4 (dev auth key) — GATEWAY-ONLY. Proves the gateway's `Some(dev_auth_pubkey_bytes())` wiring
/// is LIVE: `gateway_env` sets `VD_AUTH_PUBKEY` to the built-in dev verifying key, which a cloud gateway must
/// refuse. Strip the manual incarnation + supply a real (non-temp) durable root so the preflight passes the
/// earlier arms and reaches the gateway dev-key veto (no `VD_DEMAND` here, so the 5f-3e veto does not fire).
#[test]
fn gateway_cloud_vetoes_the_built_in_dev_auth_key() {
    let addrs = addrs();
    let trust = write_trust("dev");
    let durable = non_temp_dir("dev");
    let mut envs = common_env(&trust.display().to_string(), &DEV);
    envs.extend(gateway_env(
        &addrs,
        &[],
        &dev_auth_pubkey_hex(),
        &DEV,
        vd_bins::ClusterShape::Single,
    ));
    envs.retain(|(k, _)| *k != "VD_PROCESS_INCARNATION");
    envs.push(("VD_PROFILE", "cloud".to_owned()));
    // BOTH durable roots present + non-temp so the preflight passes the durable-root arms and REACHES the
    // dev-key veto (checked last).
    envs.push(("VD_STORE_DURABLE_ROOT", durable.display().to_string()));
    envs.push(("VD_BOOT_DURABLE_ROOT", durable.display().to_string()));

    let (ok, err) = run_to_exit(env!("CARGO_BIN_EXE_vd-gateway"), &envs)
        .expect("gateway must EXIT (refuse to boot), not run forever");
    assert!(
        !ok,
        "a cloud gateway using the dev auth key must refuse to boot; stderr: {err}"
    );
    assert!(
        err.contains("refuses the built-in DEV auth verifying key"),
        "must reject the dev auth key with guidance (proves the gateway dev-pubkey wiring is live); stderr: {err}"
    );
    let _ = std::fs::remove_dir_all(&trust);
    let _ = std::fs::remove_dir_all(&durable);
}

/// Positive control — cloud mode is NOT accidentally always-failing. A COHERENT cloud config boots the real
/// orchestrator + serves admin: strip the two dev footguns, supply the durable root (store lives UNDER it) +
/// the M3 boot-state dir (the durable monotone incarnation REPLACES the forbidden manual one), and DROP the
/// dev `VD_LEASE_TTL=10000` so cloud derives its own internally-coherent D-3 set (derived grace=150 must
/// exceed ttl; the dev ttl would break `grace > ttl`).
#[test]
fn orchestrator_cloud_boots_green_with_a_coherent_config() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let addrs = addrs();
    let admin_addr = addrs.admin;
    let trust = write_trust("green");
    let durable = non_temp_dir("green");
    let boot_durable = non_temp_dir("green-boot");
    // The M3 boot-state dir lives UNDER the boot durable root (cloud force-validates the allow-list).
    let boot_state = boot_durable.join("state");
    std::fs::create_dir_all(&boot_state).expect("boot state dir");
    let store = durable.join("orch.redb");

    let mut common = common_env(&trust.display().to_string(), &DEV);
    common.retain(|(k, _)| *k != "VD_PROCESS_INCARNATION");
    let mut node_env = orchestrator_env(
        &addrs,
        &DEV,
        &store.display().to_string(),
        vd_bins::ClusterShape::Single,
    );
    node_env.retain(|(k, _)| *k != "VD_STORE_EPHEMERAL_OK" && *k != "VD_LEASE_TTL");
    node_env.push(("VD_PROFILE", "cloud".to_owned()));
    node_env.push(("VD_STORE_DURABLE_ROOT", durable.display().to_string()));
    node_env.push(("VD_BOOT_DURABLE_ROOT", boot_durable.display().to_string()));
    node_env.push(("VD_BOOT_STATE_DIR", boot_state.display().to_string()));

    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        spawn_node(env!("CARGO_BIN_EXE_vd-orchestrator"), &common, &node_env).expect("spawn orch"),
    );
    let _guard = cluster;

    let started = Instant::now();
    let mut served = false;
    while started.elapsed() < Duration::from_secs(15) {
        if admin_get_body(admin_addr, "/admin/snapshot", Some(Duration::from_secs(2))).is_some() {
            served = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        served,
        "a COHERENT cloud config must boot the orchestrator + serve admin (cloud mode is not always-fail)"
    );
    let _ = std::fs::remove_dir_all(&trust);
    let _ = std::fs::remove_dir_all(&durable);
    let _ = std::fs::remove_dir_all(&boot_durable);
}

/// RLM 5f-3e (any role) — the demand-arming veto is LIVE in a real bin. The ORCHESTRATOR is the DoS ACTOR (it
/// ACTS on injected demands) and passes `None` to the preflight, so this proves the veto is NOT gateway-only —
/// gateway-only scoping would leave the exact configuration this closes wide open, since a bundle-holding
/// client can inject a `RealmDemand` by dialing the orchestrator directly on the mesh. `VD_PROFILE=cloud` +
/// `VD_DEMAND=1` on the standard dev env ⇒ refuse to boot (the veto is checked FIRST in the cloud block, so no
/// other footgun needs stripping to reach it).
#[test]
fn orchestrator_cloud_refuses_an_armed_demand_route() {
    let addrs = addrs();
    let trust = write_trust("demand");
    let store = std::env::temp_dir().join(format!("vd-cloudpf-demand-{}.redb", std::process::id()));
    let mut envs = common_env(&trust.display().to_string(), &DEV);
    envs.extend(orchestrator_env(
        &addrs,
        &DEV,
        &store.display().to_string(),
        vd_bins::ClusterShape::Single,
    ));
    envs.push(("VD_PROFILE", "cloud".to_owned()));
    envs.push(("VD_DEMAND", "1".to_owned()));

    let (ok, err) = run_to_exit(env!("CARGO_BIN_EXE_vd-orchestrator"), &envs)
        .expect("orchestrator must EXIT (refuse to boot), not run forever");
    assert!(
        !ok,
        "a cloud node with VD_DEMAND armed must refuse to boot; stderr: {err}"
    );
    assert!(
        err.contains("refuses VD_DEMAND"),
        "must reject the armed demand route with guidance (proves the 5f-3e veto is live for ANY role, not \
         only the gateway); stderr: {err}"
    );
    let _ = std::fs::remove_dir_all(&trust);
}

/// RLM 5f-3e enabling-condition pin (a STATIC assertion over the shipped manifests, not a process spawn). The
/// whole cloud footgun preflight — the 5f-3e demand veto, the dev-key veto, the ephemeral-store escapes —
/// fires ONLY under `Profile::Cloud`, which `resolve_profile` derives from `VD_PROFILE`; absent ⇒ DevTest, a
/// deliberate fail-OPEN. The only thing that flips the shipped cluster into cloud is the `vd-cluster-env`
/// ConfigMap setting `VD_PROFILE: "cloud"` AND every server manifest importing it via `envFrom`. Pin both, so
/// a manifest edit that drops either — silently disarming EVERY cloud security control while `just gate`
/// stays green — fails HERE instead.
#[test]
fn shipped_cloud_manifests_pin_the_cloud_profile() {
    let deploy = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deploy/k3d");
    let read = |name: &str| {
        std::fs::read_to_string(deploy.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"))
    };
    let configmap = read("10-configmap.yaml");
    assert!(
        configmap.contains("VD_PROFILE: \"cloud\""),
        "the vd-cluster-env ConfigMap MUST set VD_PROFILE=cloud — without it every cloud footgun veto (the \
         5f-3e demand veto, the dev-key veto, the ephemeral escapes) is silently disarmed into DevTest"
    );
    for node in ["30-orch.yaml", "40-gateway.yaml", "50-shard.yaml"] {
        let manifest = read(node);
        assert!(
            manifest.contains("vd-cluster-env"),
            "{node} MUST import the vd-cluster-env ConfigMap (envFrom) or the node boots WITHOUT \
             VD_PROFILE=cloud — fail-open into DevTest with every cloud veto disarmed"
        );
    }
    // THE WORLD INPUTS, shared. Both the gateway and the shard build the world from these: the gateway
    // decides which realm a login lands in, the shard simulates what is around them once there. If the two
    // are handed different numbers they describe different universes from the same seed — a login placed by
    // one geometry and then simulated by another. Pinned in the SHARED object so a retune cannot move one
    // and not the other, and asserted absent from the per-pod manifests so nothing can locally override it.
    for key in ["VD_TICK_DT", "VD_SPEED"] {
        assert!(
            configmap.contains(key),
            "the vd-cluster-env ConfigMap MUST carry {key} — the gateway REFUSES to boot without it (it \
             cannot build the world), and a per-pod copy would let the login and simulating sides drift"
        );
        // Matched as an ENV ENTRY (`name: VD_…`), not as any mention — the manifests DOCUMENT these keys in
        // comments, and a substring search would read its own documentation as a violation.
        let entry = format!("name: {key}");
        for node in ["40-gateway.yaml", "50-shard.yaml"] {
            assert!(
                !read(node).contains(&entry),
                "{node} must NOT set {key} itself — a per-pod value silently overrides the shared one and \
                 lets the gateway resolve logins by different geometry than the shard simulates"
            );
        }
    }
}
