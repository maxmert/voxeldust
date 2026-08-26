//! RLM Step 5c-2b PROCESS GATE — the real-process proof of `ProcLaunchBackend` (the "hands" behind the
//! 100%-covered `SpawnCore` kernel). vd-bins is Tier-B (§1.1), so `ProcLaunchBackend`'s launch/liveness/
//! teardown syscalls are proven HERE, forking a REAL `vd-shard`, not by the coverage gate.
//!
//! It drives the backend DIRECTLY (playing the kernel's role: `mint_cookie` → build `LaunchSpec` →
//! `launch`), which is the right shape for a Tier-B backend unit — the kernel→backend integration is
//! Tier-A-proven in `vd_node::rlm_spawn`. It asserts, against real forked shards:
//!   1. A spawned PLANET boots (its coord/profile resolve without a crash) and echoes the EXACT minted
//!      incarnation cookie on `/whoami` — the cookie plumbed end-to-end (the Step-5e pid-reuse guard).
//!   2. A spawned GALAXY-lineage shard boots too (`profile_for(Galaxy)` did NOT refuse — the un-collapsed
//!      `VD_OWN_COORD` carries `signal_relay`; a refusal would exit before binding the probe).
//!   3. `is_alive` is true while running; `teardown` makes the process EXIT and REAPS it — `pid_alive`
//!      goes false within the grace window (a `<defunct>` zombie would still answer `kill -0`, so a false
//!      answer proves the D2 reap, not just the exit).
//!
//! NOTE (scope, per the 5c vet): the "self-granted realm head appears in the orchestrator `/admin/snapshot`"
//! integration (a shard dialing a live orchestrator + the directory) is proven where a full demand cluster
//! runs — `node_per_realm_walk` (the spawn-path self-grant) and slice 5f (the orchestrator driving
//! `ProcLaunchBackend`). This gate proves the ProcLaunchBackend MECHANISM: fork → boot-with-coord → identity
//! → teardown → reap.
//!
//! Run: `cargo test -p vd-bins --test rlm_proc_spawn_smoke -- --nocapture`

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use vd_bins::proc_launch::{ProcLaunchBackend, ProcSpawnTuning};
use vd_bins::{DEV, common_env, pid_alive, reserve_tcp_addr, reserve_udp_addr};
use vd_core::NodeId;
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
use vd_io_prod::mesh::{MeshConfig, MeshControl, spawn_mesh};
use vd_io_prod::trust::ClusterTrust;
use vd_node::rlm_spawn::{LaunchBackend, LaunchSpec};

/// A Planet lineage `[Universe, Galaxy(g), System(s), Planet(p)]`.
fn planet(g: u64, s: u64, p: u64) -> RealmCoord {
    RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::Universe, 0),
        RealmLevel::new(RealmKindTag::Galaxy, g),
        RealmLevel::new(RealmKindTag::System, s),
        RealmLevel::new(RealmKindTag::Planet, p),
    ]))
    .expect("planet path has a leaf")
}

/// A Galaxy lineage `[Universe, Galaxy(g)]` — the profile that must resolve to `signal_relay`.
fn galaxy(g: u64) -> RealmCoord {
    RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::Universe, 0),
        RealmLevel::new(RealmKindTag::Galaxy, g),
    ]))
    .expect("galaxy path has a leaf")
}

/// Poll `admin_get_body(probe, "/whoami")` until it returns a body or the deadline passes.
fn poll_whoami(probe: SocketAddr, deadline: Duration) -> Option<String> {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if let Some(body) =
            vd_bins::admin_get_body(probe, "/whoami", Some(Duration::from_millis(500)))
        {
            return Some(body);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    None
}

/// A throwaway mesh + trust bundle: the trust is WRITTEN to a dir (the forked shards read it via
/// `VD_TRUST_DIR`); the `MeshControl` is this test-process's own (the `book_peer` target — harmless here,
/// the shards dial out on their own). Returns everything the shards + backend need.
struct Harness {
    _rt: tokio::runtime::Runtime,
    control: Arc<MeshControl>,
    trust_dir: std::path::PathBuf,
    workdir: std::path::PathBuf,
}

fn harness() -> Harness {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("rt");
    let trust = ClusterTrust::generate("rlm-proc-spawn-smoke").expect("trust");
    let base = std::env::temp_dir().join(format!("rlm-proc-spawn-{}", std::process::id()));
    let trust_dir = base.join("trust");
    let workdir = base.join("work");
    std::fs::create_dir_all(&trust_dir).expect("trust dir");
    std::fs::create_dir_all(&workdir).expect("work dir");
    trust.write_der_dir(&trust_dir).expect("write trust");
    let mesh_addr = reserve_udp_addr();
    let (_tx, ctl) = spawn_mesh(
        rt.handle(),
        &trust,
        &MeshConfig::new(
            NodeId(1),
            mesh_addr,
            std::collections::BTreeMap::new(),
            64,
            0,
            vd_bins::world_generation(),
        ),
        None,
    )
    .expect("mesh");
    Harness {
        _rt: rt,
        control: Arc::new(ctl),
        trust_dir,
        workdir,
    }
}

/// The fixed anchors every forked shard inherits — the shard boot-env contract MINUS the per-realm vars
/// `ProcLaunchBackend::child_env` supplies (`VD_NODE_ID`/`VD_BIND`/`VD_PROBE_ADDR`/`VD_OWN_COORD`/
/// `VD_INCARNATION_COOKIE`/`VD_REALM_KIND`/`VD_REALM_SEED`). `common_env` gives the trust + transport knobs;
/// the rest are the shard's must-parse operational params (from `DEV`). `VD_ORCH`/`VD_PEERS` name an
/// unreachable orchestrator — the shard boots + binds its probe (so `/whoami` answers) without ever
/// syncing; the mechanism proof needs no live clock.
fn anchors(trust_dir: &std::path::Path) -> Vec<(&'static str, String)> {
    let mut env = common_env(trust_dir.to_str().expect("utf8 trust dir"), &DEV);
    env.extend([
        ("VD_SNAPSHOT_BUDGET", DEV.snapshot_budget.to_string()),
        ("VD_TICK_DT", DEV.tick_dt.to_string()),
        ("VD_SPEED", DEV.move_speed.to_string()),
        ("VD_ORCH", "1".to_string()),
        ("VD_MINT_SEED", DEV.mint_seed.to_string()),
        ("VD_INPUT_LOG_CAP", DEV.input_log_cap.to_string()),
        // NOTE: VD_PEERS is NO LONGER an anchor — RLM 5d makes ProcLaunchBackend::child_env emit it from
        // the kernel-computed LaunchSpec.peers (the ancestor closure). The specs below fill it.
    ]);
    env
}

#[test]
fn proc_launch_backend_forks_boots_identifies_and_reaps_a_real_shard() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let h = harness();
    let tuning = ProcSpawnTuning {
        exe: "vd-shard",
        workdir: h.workdir.clone(),
        drain_grace: Duration::from_secs(3),
        anchors: anchors(&h.trust_dir),
        // Fast probe cadence + a generous read timeout for the adopt leg below (the orphan cookie-probe).
        orphan_probe_interval: Duration::from_millis(100),
        probe_timeout: Duration::from_millis(500),
    };
    let backend = ProcLaunchBackend::new(Arc::clone(&h.control), tuning);
    let boot_deadline = Duration::from_secs(30);

    // The RLM 5d VD_PEERS book the kernel would compute (ancestor closure ∪ anchors). Here a fixture
    // naming an unreachable orchestrator — the child parses it into its peer book + boots (a real ancestor
    // closure is exercised by the Tier-A `closure_peers_*` tests; this proves ProcLaunchBackend EMITS it and
    // the forked shard ACCEPTS a non-empty VD_PEERS end-to-end).
    let peers = vec![(NodeId(1), SocketAddr::from((Ipv4Addr::LOCALHOST, 1)))];

    // --- 1+2: a Planet and a Galaxy shard both boot; the Planet echoes its EXACT cookie on /whoami. ---
    // The spawned coord is THE world's own inner planet, derived through `world_roster` — never a
    // stated seed: a realm THE world does not contain has an empty containment forest and the shard
    // refuses to boot (0 ambient roots), which is exactly the death the retired walk-forest coords
    // hid (D-WORLD-8).
    let roster = vd_bins::world_roster(&DEV);
    // ★ S9: the galaxy is a GALAXY, not the `System(1)` stand-in it used to borrow. The seed it
    // carries is unchanged, so the shard boots the identical realm — only its name is honest now.
    let vd_core::pose::RealmId::Galaxy(galaxy_seed) = roster.galaxy else {
        panic!("the galaxy realm is a Galaxy, got {}", roster.galaxy);
    };
    let vd_core::pose::RealmId::System(home_seed) = roster.home else {
        panic!("the home realm is a System, got {}", roster.home);
    };
    let vd_core::pose::RealmId::Planet(inner_seed) = roster.inner else {
        panic!("the inner mover is a Planet, got {}", roster.inner);
    };
    let planet_node = NodeId(1_000);
    let planet_bind = reserve_udp_addr();
    let planet_probe = reserve_tcp_addr();
    let planet_cookie = backend.mint_cookie(planet_node);
    let planet_pid = backend
        .launch(&LaunchSpec {
            node: planet_node,
            coord: planet(galaxy_seed, home_seed, inner_seed),
            addr: planet_bind,
            probe: planet_probe,
            cookie: planet_cookie,
            peers: peers.clone(),
        })
        .expect("planet launch");

    let galaxy_node = NodeId(1_001);
    let galaxy_probe = reserve_tcp_addr();
    let galaxy_cookie = backend.mint_cookie(galaxy_node);
    let galaxy_pid = backend
        .launch(&LaunchSpec {
            node: galaxy_node,
            // ★ THE COLLAPSE THIS LEG USED TO EXERCISE IS GONE (slice S9), and its removal is why
            // the seed here changed from `2` to the world's own. A Galaxy LEVEL used to lower to the
            // `System(1)` stand-in WHATEVER its level seed said, so `galaxy(2)` silently became THE
            // world's galaxy and booted its neighbourhood. That was the lossy behaviour, not a
            // feature: a galaxy keeps its seed now, `Galaxy(2)` is a galaxy this world does not
            // contain, and a shard asked to host it finds an empty forest and refuses — correctly.
            //
            // What this leg pins is unchanged: the PROFILE. `profile_for(Galaxy)` must not refuse
            // (`VD_OWN_COORD` carries `signal_relay`).
            coord: galaxy(galaxy_seed),
            addr: reserve_udp_addr(),
            probe: galaxy_probe,
            cookie: galaxy_cookie,
            peers: peers.clone(),
        })
        .expect("galaxy launch");

    let planet_whoami = poll_whoami(planet_probe, boot_deadline)
        .expect("planet shard boots + serves /whoami before the deadline");
    assert_eq!(
        planet_whoami,
        planet_cookie.to_env_string(),
        "the planet shard echoes the EXACT incarnation cookie the backend minted"
    );

    let galaxy_whoami = poll_whoami(galaxy_probe, boot_deadline)
        .expect("galaxy shard boots (profile_for(Galaxy) did not refuse) + serves /whoami");
    assert_eq!(
        galaxy_whoami,
        galaxy_cookie.to_env_string(),
        "the galaxy shard echoes its cookie — signal_relay profile booted cleanly"
    );

    assert!(
        backend.is_alive(planet_node),
        "the planet shard is alive while running (try_wait = still-running)"
    );

    // --- 3: teardown EXITS and REAPS the planet shard (D2). ---
    backend.teardown(planet_node);
    let reap_deadline = Duration::from_secs(10);
    let start = Instant::now();
    while start.elapsed() < reap_deadline && pid_alive(planet_pid) {
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        !pid_alive(planet_pid),
        "teardown reaped the planet shard — pid {planet_pid} is fully gone (a zombie would still answer \
         kill -0, so this proves the reap, not just the exit)"
    );
    assert!(
        !backend.is_alive(planet_node),
        "the torn-down node is no longer a live slot"
    );

    // --- 4 (RLM 5e-4): ORPHAN ADOPTION over the /whoami cookie-probe. ---------------------------------
    // Model an orchestrator RESTART: a SECOND backend (the rebuilt process) inherits NO `Child` for the
    // still-running galaxy shard. It ADOPTS it from the persisted `(pid, cookie, probe)` — exactly what
    // `SpawnCore::rehydrate` does per survivor. Liveness is then the incarnation-cookie probe, not `try_wait`.
    let restarted = ProcLaunchBackend::new(
        Arc::clone(&h.control),
        ProcSpawnTuning {
            exe: "vd-shard",
            workdir: h.workdir.clone(),
            drain_grace: Duration::from_secs(3),
            anchors: anchors(&h.trust_dir),
            orphan_probe_interval: Duration::from_millis(100),
            probe_timeout: Duration::from_millis(500),
        },
    );
    restarted.adopt(galaxy_node, galaxy_pid, galaxy_cookie, galaxy_probe);

    // adopt seeds the cache optimistic-ALIVE (defers the first probe), so let it lapse to force a REAL probe;
    // is_alive then dials /whoami and requires the echoed cookie to EQUAL the adopted one (the pid-reuse guard).
    std::thread::sleep(Duration::from_millis(150));
    assert!(
        restarted.is_alive(galaxy_node),
        "the adopted orphan is confirmed alive via the /whoami cookie-probe (cookie matches)"
    );

    // Kill the galaxy shard; once its pid is gone, the probe can no longer connect, so the cookie-probe reads
    // DEAD — the orphan-path negative case (a recycled pid answering a DIFFERENT cookie reads dead the same way).
    backend.teardown(galaxy_node);
    let reap_deadline = Duration::from_secs(10);
    let start = Instant::now();
    while start.elapsed() < reap_deadline && pid_alive(galaxy_pid) {
        std::thread::sleep(Duration::from_millis(100));
    }
    std::thread::sleep(Duration::from_millis(150)); // lapse the probe cache so is_alive re-probes
    assert!(
        !restarted.is_alive(galaxy_node),
        "the adopted orphan reads DEAD once the shard exits — the cookie-probe cannot connect"
    );

    std::thread::sleep(Duration::from_millis(500));
    let _ = std::fs::remove_dir_all(h.workdir.parent().unwrap_or(&h.workdir));
}
