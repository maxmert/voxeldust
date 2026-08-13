//! RLM — THE demand-loop PROCESS proofs: the demand-LOGIN (RG-4c) + the demand-WALK.
//!
//! Boots a DEMAND cluster in real OS processes — orchestrator + gateway ONLY, with NO shard pre-booked
//! ([`ClusterShape::Demand`]) — then logs in ONE real dev-control client. The design pass (wf_5c3f7fbb) found
//! this path had NEVER run outside a private in-process test helper; this gate proves it end to end:
//!
//!   login → the gateway's dynamic-home route emits a `RealmDemand` → the orchestrator's ARMED reconciler
//!   spawns the home realm shard in a fresh process → that shard greets its booked peers (the reactive
//!   greeting, RG-0..3) so the gateway learns its return connection with NO pre-booked address → the gateway
//!   routes the login onto the spawn-minted node → the client goes Active and receives snapshots.
//!
//! The proof is read from the gateway's `/admin/snapshot` (the RG-4a observability): a nonzero
//! `dynamic_shards` means a login routed to a spawn-minted node (not a static `config.shard`), a nonzero
//! `presence_announces` means the demand shard's reactive greeting arrived, and `home_bootstrap_timeouts ==
//! 0` + `undecodable == 0` mean it converged cleanly. The orchestrator's `rlm.spins_requested` corroborates.
//!
//! `--test-threads=1` + a DEDICATED RLM port band ([`RLM_DEMAND_FIRST_PORT`]) so the demand-spawned shards'
//! deterministic ports never collide with `rlm_kill9_spawn`'s band when the two binaries run in parallel.
//!
//! Gated on `dev-control`: it drives the client through its loopback listener.
#![cfg(feature = "dev-control")]

use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, admin_get_body, common_env,
    dev_auth_pubkey_hex, dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows,
    orchestrator_env, reap_forked, reserve_tcp_addr, reserve_udp_addr,
};
use vd_core::NodeId;
use vd_core::glam::DVec3;
use vd_devproto::{CLIENT_NODE_BASE, DevEntityRow, DevPhase, DevRequest, DevResponse, DevState};
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::{AdminSnapshot, GatewayView, RlmView};

/// Demand: the login must SPAWN a real shard process THEN converge — a longer budget than a static login.
const DEADLINE: Duration = Duration::from_secs(60);
/// A healthy client clears this in a second or two of 20 Hz once its home is up — below it is a stall.
const SNAPSHOT_FLOOR: u64 = 5;

fn devctl(port: u16, request: &DevRequest) -> Option<DevResponse> {
    dev_roundtrip(port, request).ok()
}

fn poll_state(port: u16) -> Option<DevState> {
    match devctl(port, &DevRequest::State)? {
        DevResponse::State { state } => Some(state),
        _ => None,
    }
}

fn admin(addr: SocketAddr) -> Option<AdminSnapshot> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str(&body).ok()
}

fn gateway_view(addr: SocketAddr) -> Option<GatewayView> {
    admin(addr)?.gateway
}

/// A full `ClusterAddrs` for a demand cluster: the orchestrator + gateway binds + admin endpoints. The
/// `shard`/`shard_b`/… fields are UNUSED (no static shard is spawned — the demand-spawned shard mints its own
/// ports from the RLM band), but the struct is total, so every field gets a fresh reservation. `gateway_admin`
/// is `Some` so the gateway serves the `/admin/snapshot` this test reads the proof from.
fn demand_addrs(gateway_admin: SocketAddr) -> ClusterAddrs {
    ClusterAddrs {
        gateway_admin: Some(gateway_admin),
        ..ClusterAddrs::reserve()
    }
}

/// A test-private trust dir + the orchestrator's durable store + its `launch.redb` sidecar. Cleaned on drop.
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
    }
}

fn fixture(tag: &str) -> Fixture {
    let trust = ClusterTrust::generate("vd-rlm-demand").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-rlm-demand-{tag}-{}", std::process::id()));
    let trust_dir = base.join("trust");
    std::fs::create_dir_all(&trust_dir).expect("trust dir");
    trust.write_der_dir(&trust_dir).expect("write trust");
    let store = base.join("orchestrator.redb");
    let launch_path = store.with_file_name(vd_bins::LAUNCH_STORE_NAME);
    let _ = std::fs::remove_file(&store);
    let _ = std::fs::remove_file(&launch_path);
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    Fixture {
        store_str: store.display().to_string(),
        trust_dir,
        common,
        launch_path,
        store,
    }
}

/// SIGKILL + poll-until-gone every demand-spawned shard the orchestrator forked, on drop. Declared BEFORE the
/// `Cluster` so it drops AFTER it — by then the `Cluster` has killed + reaped the orchestrator (releasing the
/// `launch.redb` lock), so `launch_rows` can reopen the ledger to name the forked pids. This is the panic-safe
/// cleanup: the forked shards lead their OWN process groups, so the `Cluster`'s group-kill never reaches them.
struct ForkedReaper(std::path::PathBuf);
impl Drop for ForkedReaper {
    fn drop(&mut self) {
        if self.0.exists() {
            reap_forked(&launch_rows(&self.0));
        }
    }
}

/// Spawn one real dev-control client that logs into `gateway` and drives itself through its loopback listener.
fn spawn_client(f: &Fixture, gateway: SocketAddr, quic_port: u16, devctl_port: u16) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    for (k, v) in &f.common {
        cmd.env(k, v);
    }
    cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
    cmd.args([
        "--name",
        "demand-0",
        "--agent-index",
        "0",
        "--gateway",
        &gateway.to_string(),
        "--client-quic",
        &quic_port.to_string(),
        "--trust-dir",
        &f.trust_dir.display().to_string(),
        "--dev-control",
        &devctl_port.to_string(),
        "--allow-dev-control",
    ]);
    cmd.spawn().expect("spawn client")
}

/// Poll the client's dev-control listener until it is Active AND receiving, or fail LOUD with the last client
/// state + the gateway's view (so a stall names WHY — never went Active, or Active but starved).
fn await_active(devctl_port: u16, gateway_admin: SocketAddr, deadline: Duration) -> DevState {
    let started = Instant::now();
    loop {
        if let Some(st) = poll_state(devctl_port)
            && st.phase == DevPhase::Active
            && st.snapshots_applied >= SNAPSHOT_FLOOR
        {
            return st;
        }
        assert!(
            started.elapsed() < deadline,
            "the demand login never converged: last client state {:?}; gateway view {:?}",
            poll_state(devctl_port),
            gateway_view(gateway_admin),
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Push a fresh demand gateway onto `cluster` — the SAME env each time, so a restart re-binds the SAME addrs.
fn push_demand_gateway(
    cluster: &mut Cluster,
    f: &Fixture,
    a: &ClusterAddrs,
    p: &DevClusterParams,
    client_book: &[(NodeId, SocketAddr)],
) {
    cluster.push(
        "vd-gateway",
        vd_bins::spawn_node(
            env!("CARGO_BIN_EXE_vd-gateway"),
            &f.common,
            &gateway_env(
                a,
                client_book,
                &dev_auth_pubkey_hex(),
                p,
                ClusterShape::Demand,
            ),
        )
        .expect("spawn gateway"),
    );
}

/// Boot a demand cluster (orchestrator + gateway) + one dev-control client — NO shard pre-booked. The only
/// way to a world is the armed reconciler spinning one up on the client's login demand (+ its AoI as it moves).
fn boot_demand_login(
    f: &Fixture,
    a: &ClusterAddrs,
    p: &DevClusterParams,
    client_book: &[(NodeId, SocketAddr)],
    client_quic: SocketAddr,
    devctl_port: u16,
) -> Cluster {
    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        vd_bins::spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &f.common,
            &orchestrator_env(a, p, &f.store_str, ClusterShape::Demand),
        )
        .expect("spawn orchestrator"),
    );
    push_demand_gateway(&mut cluster, f, a, p, client_book);
    cluster.push(
        "client",
        spawn_client(f, a.gateway, client_quic.port(), devctl_port),
    );
    cluster
}

#[test]
fn a_demand_login_spins_up_a_fresh_home_and_lands_with_no_prebooked_shard() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let f = fixture("login");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];

    // The forked-shard reaper drops LAST (declared first) — after the Cluster has reaped the orchestrator.
    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &DEV, &client_book, client_quic, devctl_port);

    // ---- THE proof: the login went Active, and it did so by spawning + reaching a fresh home shard. ----
    let state = await_active(devctl_port, gw_admin, DEADLINE);
    assert!(
        state.own_entity.is_some(),
        "no authority after login: {state:?}"
    );
    // RLM realistic-demo Slice 3: the login lands at the star home realm (System 7) of the compressed-real
    // visual-demand geometry — the inner planets already in visibility spin up, the outer 2 are culled.
    assert_eq!(
        state.location.as_deref(),
        Some("System 7"),
        "the demand login lands at the star home realm: {state:?}",
    );
    assert_eq!(state.decode_errors, 0, "decode faults: {state:?}");
    assert_eq!(state.foreign_peer_drops, 0, "cross-wired peer: {state:?}");

    let gw = gateway_view(gw_admin).expect("the demand gateway serves its admin snapshot");
    assert!(
        gw.dynamic_shards >= 1,
        "the login must have routed to a DEMAND-SPAWNED node (not a static shard): {gw:?}",
    );
    assert!(
        gw.presence_announces >= 1,
        "the demand shard's reactive greeting must have reached the gateway: {gw:?}",
    );
    assert_eq!(
        gw.home_bootstrap_timeouts, 0,
        "the home realm bootstrapped within its TTL (no timeout): {gw:?}",
    );
    assert_eq!(gw.undecodable, 0, "no undecodable frames: {gw:?}");
    assert_eq!(gw.sessions_open, 1, "exactly the one login session: {gw:?}");

    let orch = admin(a.admin).expect("the orchestrator serves its admin snapshot");
    assert!(
        orch.rlm.spins_requested >= 1,
        "the armed reconciler requested at least one spawn: {:?}",
        orch.rlm,
    );
    assert_eq!(orch.rlm.spins_failed, 0, "no spawn failed: {:?}", orch.rlm);
}

#[test]
fn a_gateway_restart_re_learns_the_running_demand_shard_via_the_reactive_greeting() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // The reactive greeting's SELF-HEAL property (RG-1): the connection between a demand-spawned shard and the
    // gateway is learned (not pre-booked), so it must survive a GATEWAY restart with no manual re-plumb. This
    // is the SERVER-side re-heal — the client's session does NOT resume (reconnect-without-replay is D-11,
    // deferred), but the still-running shard must re-teach a FRESH gateway where it is.
    let f = fixture("reheal");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];

    let _reaper = ForkedReaper(f.launch_path.clone());
    let mut cluster = boot_demand_login(&f, &a, &DEV, &client_book, client_quic, devctl_port);

    // The login spawned a home shard + the gateway learned it via the reactive greeting.
    await_active(devctl_port, gw_admin, DEADLINE);
    let before = gateway_view(gw_admin).expect("gateway view");
    assert!(
        before.presence_announces >= 1 && before.dynamic_shards >= 1,
        "the demand shard is up + learned before the restart: {before:?}",
    );

    // SIGKILL + reap the gateway, then boot a FRESH one on the SAME addrs (a fresh process ⇒ its
    // presence_announces resets to 0). The still-running demand shard is UNAFFECTED — it leads its OWN process
    // group, so the Cluster's per-child kill never touched it. D-34: we restart the GATEWAY (the learner), not
    // the shard.
    cluster.kill_and_reap("vd-gateway");
    push_demand_gateway(&mut cluster, &f, &a, &DEV, &client_book);
    let _cluster = cluster;

    // THE re-heal proof: the fresh gateway RE-LEARNS the still-running shard via the greeting ALONE. The shard
    // greets-on-silence (RG-1) — having stopped hearing from the old gateway it re-dials the SAME booked
    // address + re-sends its ShardPresence, so the fresh gateway learns the return connection with NO
    // pre-booked address + NO re-login. `presence_announces` climbs from 0 back to >= 1.
    let started = Instant::now();
    loop {
        if let Some(gw) = gateway_view(gw_admin)
            && gw.presence_announces >= 1
        {
            break;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "the restarted gateway never re-learned the demand shard via the reactive greeting: {:?}",
            gateway_view(gw_admin),
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// REPRO (frozen-planet bug): after a demand login lands at the star, the client MUST receive the shard's
/// LIVE realm-pose feed (`RealmSnapshotDatagram`) — that is what makes the orbiting planets MOVE on the
/// client. A human flying `demand-visual-run.sh` sees the planets FROZEN; the diagnosis is the client's
/// `realm_view` is (near-)empty ⇒ `realm_frames_applied == 0`. This gate reproduces it in a real process:
/// boot the demand cluster + a headless client, wait a few seconds near the star, and assert the client
/// applied at least one live realm frame. FAIL here == the frozen-planet bug, reproduced.
#[test]
fn a_demand_login_applies_live_realm_frames_so_the_planets_are_not_frozen() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let f = fixture("realmframes");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &DEV, &client_book, client_quic, devctl_port);

    let state = await_active(devctl_port, gw_admin, DEADLINE);
    assert_eq!(
        state.location.as_deref(),
        Some("System 7"),
        "the demand login lands at the star home realm: {state:?}",
    );
    eprintln!(
        "[repro] at Active: realm_frames_applied={} snapshots_applied={} universe_tick={:?} location={:?}",
        state.realm_frames_applied, state.snapshots_applied, state.universe_tick, state.location,
    );

    // (i) The client IS receiving live realm frames (the feed lands). A healthy shard clears this in a
    // second or two of 20 Hz.
    let started = Instant::now();
    let mut frames = state.realm_frames_applied;
    loop {
        if let Some(st) = poll_state(devctl_port) {
            frames = st.realm_frames_applied;
            if frames > 0 {
                break;
            }
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the client never applied a LIVE realm frame (realm_frames_applied stayed {frames})",
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    // (ii) THE frozen-planet assertion: the shard's UNIVERSE TICK must ADVANCE. Every planet's orbital pose
    // is `orbital_state(elements, secs_since_epoch(universe_tick, hz))` — if `universe_tick` is FROZEN the
    // planets are stuck at one ephemeris instant and appear frozen on the client, EVEN THOUGH realm frames
    // keep flowing (each carries the SAME pose). The client's `universe_tick` is the shard's, carried in the
    // snapshot, so a frozen clock is directly observable here.
    let t0 = poll_state(devctl_port)
        .and_then(|s| s.universe_tick)
        .expect("a universe tick once snapshots are landing");
    std::thread::sleep(Duration::from_secs(4));
    let t1 = poll_state(devctl_port)
        .and_then(|s| s.universe_tick)
        .expect("a universe tick after the wait");
    eprintln!(
        "[repro] realm_frames_applied={frames}; universe_tick t0={t0} t1={t1} (advanced by {})",
        t1.saturating_sub(t0),
    );
    assert!(
        t1 > t0,
        "FROZEN PLANETS reproduced: the demand home shard's universe_tick did NOT advance over 4s \
         (t0={t0} t1={t1}). The planets' ephemeris time is stuck ⇒ they are frozen on the client, even \
         though realm_frames_applied kept climbing to {frames}. The demand-spawned shard is not receiving \
         the ongoing ClockSync broadcast (DynamicClockPeers).",
    );
    eprintln!("[repro] universe_tick advanced {t0} -> {t1} — the planets are LIVE on the client");
}

/// REPRO (frozen-planet, DECISIVE — reads the DRAWN scene): the twin above proved realm frames ARRIVE and the
/// clock advances, but never checked what the client actually DRAWS. This reads `DevState.realm_boxes` (the
/// overlaid render scene the renderer publishes) and splits the two possible freezes: (a) NO `Planet` row ⇒ the
/// `RealmSceneDelta` render-scene never reached the client (only the static container shells are drawn — looks
/// frozen); (b) a `Planet` row whose center does NOT move over ~3 s ⇒ the render-scene landed but the live
/// realm-pose feed is not overlaying it (drawn at its static epoch center — frozen). A human flying the demand
/// cluster sees FROZEN planets from the start; this pins WHICH of the two it is.
#[test]
fn a_demand_login_draws_moving_planets_not_a_frozen_scene() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let f = fixture("drawnplanets");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &DEV, &client_book, client_quic, devctl_port);

    let state = await_active(devctl_port, gw_admin, DEADLINE);
    assert_eq!(
        state.location.as_deref(),
        Some("System 7"),
        "login at the star: {state:?}"
    );

    // Wait until the client has BOTH applied a live realm frame AND drawn ≥1 planet box.
    let started = Instant::now();
    let mut last = state;
    loop {
        if let Some(st) = poll_state(devctl_port) {
            let planets = st
                .realm_boxes
                .iter()
                .filter(|b| b.realm.starts_with("Planet("))
                .count();
            let ok = st.realm_frames_applied > 0 && planets > 0;
            last = st;
            if ok {
                break;
            }
        }
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "(a) NO planet drawn: realm_frames_applied={} realm_boxes={:?} — the RealmSceneDelta \
             render-scene never reached the client (only the container shells are drawn = frozen)",
            last.realm_frames_applied,
            last.realm_boxes,
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    eprintln!("[repro] drawn realm_boxes t0: {:?}", last.realm_boxes);
    let planet0: Vec<_> = last
        .realm_boxes
        .iter()
        .filter(|b| b.realm.starts_with("Planet("))
        .cloned()
        .collect();

    // (b) a drawn planet's center MOVES over ~3 s — the pose feed is overlaying (orbiting, not frozen).
    std::thread::sleep(Duration::from_secs(3));
    let t1 = poll_state(devctl_port).expect("state after wait");
    eprintln!("[repro] drawn realm_boxes t1: {:?}", t1.realm_boxes);
    let mut best_move = 0.0_f64;
    for p0 in &planet0 {
        if let Some(p1) = t1.realm_boxes.iter().find(|b| b.realm == p0.realm) {
            let d = ((p1.center[0] - p0.center[0]).powi(2)
                + (p1.center[1] - p0.center[1]).powi(2)
                + (p1.center[2] - p0.center[2]).powi(2))
            .sqrt();
            eprintln!(
                "[repro] {} moved {d:.4}m ({:?} -> {:?})",
                p0.realm, p0.center, p1.center
            );
            best_move = best_move.max(d);
        }
    }
    assert!(
        best_move > 1e-3,
        "(b) FROZEN PLANETS: a drawn planet's center did NOT move over 3 s (best {best_move:.5} m) — the \
         render-scene landed but the realm-pose feed is not overlaying the streamed boxes. t0={planet0:?} \
         t1={:?}",
        t1.realm_boxes,
    );
    eprintln!("[repro] a drawn planet moved {best_move:.4}m over 3s — LIVE, not frozen");
}

/// FULL-SPEED PARKED PROOF (the containment-flap fix): a stationary ship at the star must keep the planets
/// ORBITING indefinitely at full orbit speed (NO slowdown), PAST the planet spin-up. With the OLD buggy
/// containment a moving planet's phantom capture-center (`live_pos + epoch`) swept the star origin every
/// half-orbit (~7 s for the inner planet at full speed), wrongly re-homing the parked ship INTO the planet ⇒
/// the sibling feed froze. With a mover's `region.center = 0` the capture-zone is always ON the planet, so a
/// parked ship stays ~13.7 m outside forever. Asserts, over a window past spin-up: the ship never leaves
/// "System 7", the realm feed keeps climbing, and a drawn planet keeps moving. FLAPS + FREEZES on the pre-fix
/// shard (the speed-dependent freeze the human saw at full speed but not under the orbit-slowdown flag).
#[test]
fn a_parked_ship_keeps_the_planets_orbiting_at_full_speed() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let f = fixture("parkedfull");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &DEV, &client_book, client_quic, devctl_port);
    let start = await_active(devctl_port, gw_admin, DEADLINE);
    assert_eq!(
        start.location.as_deref(),
        Some("System 7"),
        "login at the star: {start:?}"
    );

    let planets = |st: &DevState| -> Vec<(String, [f64; 3])> {
        st.realm_boxes
            .iter()
            .filter(|b| b.realm.starts_with("Planet("))
            .map(|b| (b.realm.clone(), b.center))
            .collect()
    };
    // Poll ~28 s (PAST spin-up + several inner-planet orbits). The ship NEVER moves (no WalkTo).
    let started = Instant::now();
    let mut first: Option<DevState> = None;
    let mut last = start;
    while started.elapsed() < Duration::from_secs(28) {
        if let Some(st) = poll_state(devctl_port) {
            assert_eq!(
                st.location.as_deref(),
                Some("System 7"),
                "the PARKED ship was wrongly re-homed out of System 7 (the flap) at {}s: loc={:?}",
                started.elapsed().as_secs(),
                st.location,
            );
            if first.is_none() && st.realm_frames_applied > 0 && !planets(&st).is_empty() {
                first = Some(st.clone());
            }
            last = st;
        }
        std::thread::sleep(Duration::from_millis(400));
    }
    let first = first.expect("the client drew a planet + applied a realm frame");
    assert!(
        last.realm_frames_applied > first.realm_frames_applied + 20,
        "the realm feed FROZE while parked: {} -> {} frames",
        first.realm_frames_applied,
        last.realm_frames_applied,
    );
    let (p0, p1) = (planets(&first), planets(&last));
    let moved = p0
        .iter()
        .filter_map(|(r, c0)| {
            p1.iter().find(|(r1, _)| r1 == r).map(|(_, c1)| {
                ((c1[0] - c0[0]).powi(2) + (c1[1] - c0[1]).powi(2) + (c1[2] - c0[2]).powi(2)).sqrt()
            })
        })
        .fold(0.0_f64, f64::max);
    assert!(
        moved > 1.0,
        "no drawn planet moved over the window — frozen (best {moved:.3} m)"
    );
    eprintln!(
        "[parked] stayed in System 7; feed {} -> {}; a planet moved {moved:.2} m — LIVE at full speed",
        first.realm_frames_applied, last.realm_frames_applied,
    );
}

/// The orbit-slowdown knob, PANIC-SAFE (Stage B4). The crossing tests slow the orbits through
/// `VD_VISUAL_ORBIT_SLOWDOWN` (value from `VD_TEST_ORBIT_SLOWDOWN`, default 300; set that to 1 for a
/// flight-speed run). The old bare `set_var`/`remove_var` pair leaked the quasi-freeze into the
/// FULL-SPEED tests that follow in this binary whenever an assert fired between the two calls — one
/// crossing failure then cascaded into spurious "frozen world" failures downstream. The guard's `Drop`
/// runs on unwind, so the knob clears however the test ends. (`--test-threads=1` serializes the
/// binary and shards read the knob only at boot, so holding it for the whole test body is equivalent
/// to the old mid-test reset.)
struct OrbitSlowdown;

impl OrbitSlowdown {
    fn engage() -> OrbitSlowdown {
        // SAFETY: set before any cluster boot; the binary runs single-threaded (`--test-threads=1`).
        unsafe {
            std::env::set_var(
                "VD_VISUAL_ORBIT_SLOWDOWN",
                std::env::var("VD_TEST_ORBIT_SLOWDOWN").unwrap_or_else(|_| "300".to_owned()),
            );
        }
        OrbitSlowdown
    }
}

impl Drop for OrbitSlowdown {
    fn drop(&mut self) {
        // SAFETY: same single-threaded discipline as `engage`.
        unsafe { std::env::remove_var("VD_VISUAL_ORBIT_SLOWDOWN") }
    }
}

/// The INNER planet's `(realm, elements)` — the mover whose EPOCH position is CLOSEST to the star (well
/// inside System 7's 150 m SOI, so approaching it never crosses the System boundary).
///
/// THROUGH THE PRODUCTION BOOT (Stage B4): `vd_bins::boot_regions_and_movers` applies the
/// `VD_VISUAL_ORBIT_SLOWDOWN` knob to the star mass exactly as the booting shard does, so the elements
/// the fixture aims by ARE the elements the star authors with. The old direct
/// `moving_children_for_config` read the UNMODIFIED config: under the gated 300x default the fixture's
/// aim point orbited 300x faster than the real planet — a phantom target that merely swept the right
/// circle. Call it AFTER the [`OrbitSlowdown`] guard engages.
fn inner_planet_mover(
    p: &DevClusterParams,
) -> (vd_core::pose::RealmId, vd_core::celestial::OrbitalElements) {
    let system = vd_core::pose::RealmId::System(7);
    let held = std::collections::BTreeSet::from([system]);
    let (_regions, moving) =
        vd_bins::boot_regions_and_movers(p.universe_seed, &held, system, p.move_speed, p.tick_dt);
    moving
        .into_iter()
        .min_by(|a, b| {
            vd_core::celestial::orbital_state(&a.1, 0.0)
                .position
                .length()
                .total_cmp(
                    &vd_core::celestial::orbital_state(&b.1, 0.0)
                        .position
                        .length(),
                )
        })
        .expect("the visual-demand forest has planet movers")
}

/// THE CROSSING PROOF (moving-frame crossing fix): flying an occupant INTO a demand-spawned orbiting planet
/// RE-HOMES cleanly — `location` flips to the planet realm, no flap. DECISIVE isolation: a large
/// `VD_VISUAL_ORBIT_SLOWDOWN` quasi-freezes the orbits (the EPOCH position is slowdown-invariant), so the
/// target is a FIXED point, and the INNER planet sits well inside System 7's 150 m SOI (no System-boundary
/// confound). Fly straight to the inner planet's epoch center and assert `location` flips to the planet.
///
/// WHY IT NOW WORKS (was the flap): (1) a mover's `region.center` is ZERO, so the planet's SOI is centered
/// on the planet itself (not shifted ~one orbit off) — the source + the planet shard compute the SAME
/// containment; (2) the SOURCE rebases the flushed pose into the planet's LIVE frame at flush
/// (`on_flush_source` via `LocalFrames`), so the planet shard reads the occupant at its own origin (inside),
/// AGREEING with the source. Both together ⇒ the two shards no longer disagree ⇒ no flap ⇒ the re-home
/// completes. (Was `#[ignore]`d as the known-failing repro before the fix landed.)
///
/// THEN THE CASCADE PROOF (the "world stale after re-home" fix): once landed on the planet — which authors
/// nothing itself — the client's realm feed must NOT freeze. The star shard CASCADES its authored planet
/// orbits DOWN to this active child, which re-fans them, so `realm_frames_applied` keeps climbing on the
/// planet shard (the whole system still orbits around the crossed-in player). A frozen counter here is the bug.
#[test]
fn a_flying_occupant_re_homes_into_an_inner_planet_no_boundary_flap() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let _slowdown = OrbitSlowdown::engage();
    let f = fixture("innerrehome");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let p = DEV;
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];
    let deadline = Duration::from_secs(150);

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &p, &client_book, client_quic, devctl_port);

    let landed = await_active(devctl_port, gw_admin, deadline);
    assert_eq!(
        landed.location.as_deref(),
        Some("System 7"),
        "login at the star: {landed:?}"
    );

    let (planet, elements) = inner_planet_mover(&p);
    let planet_seed = match planet {
        vd_core::pose::RealmId::Planet(s) => s,
        other => panic!("inner mover is a planet, got {other:?}"),
    };
    // AIM AT WHERE THE PLANET IS NOW, re-aimed every leg — the way a pilot flies.
    //
    // It used to aim at the planet's EPOCH centre, which is where that planet sits only while it barely
    // moves. That is exactly what the 300x orbit slowdown this fixture sets buys, and it made the fixture
    // unable to fail on a world that moves: run it at flight speed and it chases a point the planet left
    // long ago — measured, closest approach 17.93 m against a 4.16 m boundary while the occupant wandered
    // 12, 31, 41 and 23 m out. A fixture that can only pass on a nearly-still world cannot gate a game
    // that is played on a moving one.
    let tick_hz = 1.0 / p.tick_dt;
    let planet_at = |tick: u64| {
        vd_core::celestial::orbital_state(
            &elements,
            vd_core::celestial::secs_since_epoch(tick, tick_hz),
        )
        .position
    };
    // RENDEZVOUS AND PARK (Stage B4; run-3/run-4 measurements). Chasing the live centre — even with a
    // led aim — keeps the ship at chase speed through the band, and the flush re-validation then
    // RIGHTLY refuses a crossing whose subject is already gone: ~19 m of travel between the decision
    // and the flush, 15+ aborted attempts, best approach 0.39 m, zero commits. A pilot lands by
    // ARRIVING EARLY: fly to where the planet WILL BE, stop, and let it sweep over the parked ship —
    // the relative speed is then the planet's own ~6 m/s (~0.13 m per tick), the decision is still
    // true at the flush, and the crossing commits. At the gated 300x default the rendezvous point is
    // in effect the current centre, so this degenerates to fly-there-and-arrive.
    const RENDEZVOUS_TICKS: u64 = 200;
    // How long past the rendezvous instant to stay parked before re-planning: one full sweep of the
    // planet's shell (~8.3 m at ~6.3 m/s ≈ 66 ticks) plus margin for the crossing pipeline.
    const SWEEP_GRACE_TICKS: u64 = 120;
    let epoch = planet_at(0);
    let want = vd_core::pose::FrameRef::PlanetCentered { planet_seed }.label();
    eprintln!(
        "[repro] inner planet {planet:?} epoch_len={:.1} — flying to its LIVE centre, expect {want:?}",
        epoch.length(),
    );

    let started = Instant::now();
    let mut best = f64::INFINITY;
    let mut loc = String::new();
    let mut leg = 0u32;
    loop {
        let st = {
            let mut got = None;
            for _ in 0..20 {
                if let Some(s) = poll_state(devctl_port) {
                    got = Some(s);
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            match got {
                Some(s) => s,
                None => {
                    assert!(
                        started.elapsed() < deadline,
                        "dev-control unreachable (best {best:.2} m, loc {loc:?})"
                    );
                    continue;
                }
            }
        };
        loc = st.location.clone().unwrap_or_default();
        if loc == want {
            break;
        }
        // FLY TO THE RENDEZVOUS — where the planet will be `RENDEZVOUS_TICKS` from the client's last
        // report (the star authors these elements; the fixture derives them through the same boot the
        // star uses) — then PARK there and let the planet sweep over the stationary ship.
        let now_tick = st.universe_tick.unwrap_or(0);
        let target = planet_at(now_tick + RENDEZVOUS_TICKS);
        eprintln!(
            "[repro] inner leg {leg}: tick={now_tick} loc={loc:?} own_len={:?} best={best:.2} — \
             rendezvous at tick {}",
            own_pos(&st).map(|v| v.length()),
            now_tick + RENDEZVOUS_TICKS,
        );
        leg += 1;
        let flew = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: target.to_array(),
                arrive_epsilon: 0.5,
                max_ticks: RENDEZVOUS_TICKS,
                // THE BRAKE (nav::walk_to), sized to the FEEDBACK LAG (run-7 measurement): the
                // controller steers by the DELIVERED pose, which trails the server by the
                // interpolation buffer (~2-3 ticks). A brake sized to ONE step still overshoots
                // and declares arrival on the lagged pose mid-hunt, parking 10-45 m off. Sizing
                // it to (1 + lag) steps commands at most dist/4 per tick — a monotone, no-overshoot
                // approach on which the lagged arrival test is honest.
                max_step_m: 4.0 * p.move_speed * p.tick_dt,
            },
        );
        // CUT THE THROTTLE before parking (run-6 measurement): the Move input is STICKY — a leg
        // that times out leaves the last axes held, and a park that sleeps on a held throttle is
        // a full-speed straight-line runaway (measured: ~1 km out of the system in one window).
        let _ = devctl(
            devctl_port,
            &DevRequest::Move {
                axes: [0.0, 0.0, 0.0],
            },
        );
        eprintln!(
            "[repro]   leg {}: walk outcome {}",
            leg - 1,
            match &flew {
                Some(DevResponse::State { .. }) => "ARRIVED",
                Some(DevResponse::Timeout { .. }) => "TIMEOUT",
                other => {
                    let _ = other;
                    "OTHER"
                }
            },
        );
        // PARKED: hold until the sweep instant (plus grace) or the flip, whichever first.
        let wait_until = now_tick + RENDEZVOUS_TICKS + SWEEP_GRACE_TICKS;
        loop {
            std::thread::sleep(Duration::from_millis(200));
            let Some(s) = poll_state(devctl_port) else {
                break;
            };
            loc = s.location.clone().unwrap_or_default();
            if let Some(pos) = own_pos(&s) {
                best = best.min((pos - planet_at(s.universe_tick.unwrap_or(0))).length());
            }
            if loc == want || s.universe_tick.unwrap_or(0) > wait_until {
                break;
            }
        }
        if loc == want {
            break;
        }
        assert!(
            started.elapsed() < deadline,
            "NO CROSSING (inner): flew rendezvous legs at the inner planet {planet:?} for {}s but \
             location never left \"System 7\" (loc {loc:?}); closest approach {best:.2} m vs ~4.16 m SOI.",
            started.elapsed().as_secs(),
        );
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    eprintln!("[repro] inner location flipped to {loc:?} (closest approach {best:.2} m)");
    assert_eq!(loc, want);
    // (The orbit-slowdown knob now clears at the `OrbitSlowdown` guard's drop — panic-safe, end of test.)

    // ── THE CASCADE PROOF (the "world stale after re-home" fix) ────────────────────────────────────────────
    // The player is now on the PLANET shard, which authors NOTHING itself (it has no moving children). WITHOUT
    // the per-realm observation cascade its realm feed FREEZES — every other planet stops orbiting on the
    // client, the exact symptom the user hit ("all froze; only re-homing back to the star moved again"). WITH
    // the fix the STAR shard keeps authoring every planet's orbit and CASCADES those poses DOWN to this active
    // child, which re-fans them, so the client's `realm_frames_applied` keeps CLIMBING while it sits on the
    // planet. First let any in-flight star frames DRAIN, so every frame counted past the baseline can ONLY be
    // the cascade; then require a healthy climb (a frozen feed never reaches it).
    std::thread::sleep(Duration::from_secs(1)); // let in-flight star frames drain first
    // Prove the feed keeps climbing WHILE the occupant is homed to a CHILD realm (a planet) — where WITHOUT the
    // cascade nothing would author its realm view. `realm_frames_applied` is a monotone client counter, so we
    // baseline on the first on-planet poll and require a healthy climb. The occupant may legitimately re-home
    // BETWEEN the clustered inner planets (the compressed visual scale packs their SOIs close, and a stopped
    // ship sits where two overlap) — that is fine; ANY planet is a child realm with no self-authoring, so a live
    // feed there is the cascade at work. A brief hop to the star (which DOES self-author) is simply not counted.
    const CASCADE_MIN_FRAMES: u64 = 20;
    let cascade_deadline = Instant::now() + Duration::from_secs(25);
    let mut on_planet_baseline: Option<u64> = None;
    let mut latest = 0u64;
    let mut saw_planet = false;
    // PARKED IS RIDING (Stage B4): once ON the planet the player's pose is PLANET-LOCAL, and the
    // planet's own motion is its parent's business — a parked occupant rides the realm by
    // construction (the realm model's whole point). No station-keeping here: walking is exactly what
    // pushed the run-2 player OFF the planet (a WalkTo target stated in the OLD system frame while
    // the player's stream was already planet-local).
    loop {
        std::thread::sleep(Duration::from_millis(300));
        if let Some(s) = poll_state(devctl_port) {
            if s.location
                .as_deref()
                .is_some_and(|l| l.starts_with("Planet"))
            {
                saw_planet = true;
                latest = s.realm_frames_applied;
                let base = *on_planet_baseline.get_or_insert(latest);
                if latest >= base + CASCADE_MIN_FRAMES {
                    break;
                }
            }
        }
        assert!(
            Instant::now() < cascade_deadline,
            "WORLD STALE on a planet shard: realm_frames_applied did not climb by {CASCADE_MIN_FRAMES} in 25 s \
             (saw_planet={saw_planet}, baseline={on_planet_baseline:?}, latest={latest}); the observation \
             cascade is not re-fanning the star's authored orbits — the system is frozen on the client.",
        );
    }
    assert!(
        saw_planet,
        "the occupant never settled on a child realm to prove the cascade"
    );
    eprintln!(
        "[repro] post-crossing cascade LIVE on a planet shard: realm_frames_applied climbed to {latest} \
         (the whole system keeps orbiting around the crossed-in player)"
    );
}

/// ★ Symptom-B (primary) + Symptom-A acceptance — a full System→Planet→System ROUND TRIP (two rehomes)
/// commits BOTH crossings and the occupant RIDES its realm throughout (no teleport-to-origin). Fly IN to the
/// inner planet (System→Planet), assert the occupant's world pose tracks its streamed planet box (Symptom A:
/// composed against its live realm placement, not rendered at raw frame-local), DWELL so System's retained
/// proxy ages out (the reap window), then fly back INWARD toward the star (Planet→System).
///
/// Symptom B (primary) — the reported "everything froze on the way back": WITHOUT the crossing dest-realm
/// keep-alive the return-dest System is reaped mid-crossing → the re-driven `CrossingRequest` parks
/// unresolved → the player lands on no live authority and EVERYTHING (player + world) freezes. WITH the fix
/// the SOURCE shard keep-alive-demands System (+ its whole chain) for the crossing's duration — re-spinning
/// it if the dwell already reaped it — so the return COMMITS: `location` returns to "System 7", the player is
/// on a live authority, and its own pose keeps advancing. That commit is the acceptance gate here (the
/// fly-out loop only breaks when `location == "System 7"`, which is unreachable if the return parked).
///
/// RESIDUAL (owed, NOT asserted — see DEFERRED.md D-RLM-14 / floating-origin S6 / VU-6): after the return
/// commits, the SURROUNDING realm feed (per-tick `RealmSnapshot`) does not yet re-advance on the cross. The
/// dest's read sub re-opens only from the `SubscriptionReady` at `on_saga_promote`, and the realm SCENE is
/// re-streamed only at a home-entry `Active` promote (never on a cross) — so a returned player is live and
/// rides its own realm, but its neighbours can stall until the warp re-stream lands. The total freeze (the
/// user's report) is fixed; this narrower residual is the next slice. The tail below OBSERVES it (logs the
/// owed status) rather than panicking, so the landed fixes gate green; flip it to a hard assert when D-RLM-14
/// lands. This is a ROUND-TRIP: two rehomes, proving the crossing machinery survives repeated System↔Planet
/// moves. A large epoch-invariant orbit slowdown keeps the fly targets fixed; the knob is reset at the END (a
/// leak would quasi-freeze a later full-speed test).
#[test]
fn a_planet_to_system_return_commits_both_rehomes_and_the_player_rides() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let _slowdown = OrbitSlowdown::engage();
    let f = fixture("returnnofreeze");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let p = DEV;
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];
    let deadline = Duration::from_secs(150);

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &p, &client_book, client_quic, devctl_port);

    let landed = await_active(devctl_port, gw_admin, deadline);
    assert_eq!(
        landed.location.as_deref(),
        Some("System 7"),
        "login at the star: {landed:?}"
    );

    let (planet, elements) = inner_planet_mover(&p);
    let planet_seed = match planet {
        vd_core::pose::RealmId::Planet(s) => s,
        other => panic!("inner mover is a planet, got {other:?}"),
    };
    let epoch = vd_core::celestial::orbital_state(&elements, 0.0).position;
    let want_planet = vd_core::pose::FrameRef::PlanetCentered { planet_seed }.label();

    // Poll dev-control with a short retry (it may briefly blink between ticks).
    let poll = |until: Instant| -> DevState {
        loop {
            for _ in 0..20 {
                if let Some(s) = poll_state(devctl_port) {
                    return s;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            assert!(Instant::now() < until, "dev-control unreachable");
        }
    };

    // ── FLY IN, park-then-nudge (the space-change trap): a WalkTo aimed at the epoch in SYSTEM ──
    // numbers keeps driving after the crossing flips the avatar into PLANET space — the same target
    // array is then 17.9 m away in the NEW space and the remaining budget walks the avatar straight
    // back out of the 4.16 m shell (measured; the pre-B1 stale commits masked it). So: approach to a
    // PARK POINT 6 m short of the epoch (outside the shell — no crossing can fire), settle, then
    // nudge inward in 2-tick legs, polling the flip BETWEEN nudges — at most one nudge of
    // misdirected drive (~a metre) can ever land after the flip.
    let in_deadline = Instant::now() + deadline;
    // PARK INSIDE THE ACQUIRE EDGE: the shell is ~4.16 m and containment acquires at ≥1 m inside
    // (~3.16 m from the centre), so an arrive tolerance of 2.5 m puts the PARKED ship past the edge
    // — the crossing fires on a stationary avatar, and there is no remaining drive budget for the
    // space-change trap to misdirect. The heavy brake (20× a full-speed step) keeps the terminal
    // approach at walking pace, so the drive's own lag never overshoots the tolerance.
    let _ = devctl(
        devctl_port,
        &DevRequest::WalkTo {
            target: epoch.to_array(),
            arrive_epsilon: 2.5,
            max_ticks: 600,
            max_step_m: 20.0 * DEV.move_speed * DEV.tick_dt,
        },
    );
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    loop {
        if poll(in_deadline).location.as_deref() == Some(want_planet.as_str()) {
            break;
        }
        assert!(
            Instant::now() < in_deadline,
            "NO CROSSING IN: location never became {want_planet:?}. The shards DO hand the player over \
             (their logs show it), so the question this failure has to answer is what the ROUTER did with \
             the frames afterwards — gateway: {:?}",
            gateway_view(gw_admin),
        );
        std::thread::sleep(Duration::from_millis(150));
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    eprintln!("[repro] crossed IN to {want_planet:?}");

    // ── RIDE THE REALM (Symptom A), THE SETTLED ONE-SPACE MODEL (crossing-render slice, §4x/§4y):
    // standing ON the planet, the client stands IN the planet's space — its own pose AND the planet's
    // box are BOTH planet-local, so they coincide near the origin, and "riding the orbit" is true by
    // construction (the world moves around you; you do not chase your own ground). The assertion that
    // still catches the ORIGINAL teleport bug is the GAP: a player rendered at raw frame-local against
    // a box still drawn at its parent-space orbit reads a gap of the whole orbit (~17 m), and a stale
    // pre-crossing box (the measured "I landed on the planet and I am outside it") reads metres. The
    // OLD precondition here (`center.length() > 12`) asserted the pre-slice model — the box drawn in
    // the PARENT's space while its rider drew in the planet's — which is exactly the two-space split
    // the slice made unrepresentable; it flipped to red the moment the cure landed, as it should.
    let want_box = format!("Planet({planet_seed})");
    // SETTLE, then assert: the saga tail (commit → the source's demote) legitimately overlaps two
    // owners for a few ticks, and the drawn pose belongs to whichever space's row is freshest — a
    // transient the crossing cut ends. Poll until the pair COINCIDES twice in a row (the settled
    // one-space state), bounded; the assert then reads the settled values.
    const PLANET_RIDE_TOL_M: f64 = 8.0;
    let (own, center) = {
        let until = Instant::now() + Duration::from_secs(15);
        let mut streak = 0u32;
        let mut latest: Option<(DVec3, DVec3)> = None;
        loop {
            let s = poll(until);
            if let (Some(own), Some(bx)) = (
                own_pos(&s),
                s.realm_boxes.iter().find(|b| b.realm == want_box),
            ) {
                let center = DVec3::from_array(bx.center);
                latest = Some((own, center));
                // BOTH halves of the settled state: the pair coincides AND the pair sits at the
                // observer's own-space origin — a coincident pair out at the parent-space orbit is
                // the stale pre-cut render, coherent but in the WRONG space (measured).
                streak = if (own - center).length() < PLANET_RIDE_TOL_M
                    && center.length() < PLANET_RIDE_TOL_M
                {
                    streak + 1
                } else {
                    0
                };
                if streak >= 2 {
                    break latest.expect("just set");
                }
            }
            assert!(
                Instant::now() < until,
                "RIDE NEVER SETTLED: own pose and the planet box never coincided after cross-in \
                 (last {latest:?}) — two spaces in one picture (the pre-slice split, or a stale \
                 pre-crossing box)."
            );
            std::thread::sleep(Duration::from_millis(150));
        }
    };
    assert!(
        center.length() < PLANET_RIDE_TOL_M,
        "the OCCUPIED planet's box draws at the observer's own space origin (SL3: the realm you stand \
         in draws itself at your origin), not at its parent-space orbit: {:.2} m",
        center.length(),
    );
    eprintln!(
        "[repro] RIDES THE REALM (one space): own {own:?} beside the planet box {center:?} (gap {:.2} m)",
        (own - center).length()
    );

    // Dwell so System's retained proxy ages out — the exact reap window the keep-alive must survive/re-spin.
    std::thread::sleep(Duration::from_secs(3));

    // ── FLY OUT INWARD toward the star (≈ -epoch in the planet frame) until `location` returns to System 7. ──
    // The inner planet is the CLOSEST to the star and well inside System 7's 150 m SOI, so heading INWARD
    // exits the planet's tiny SOI straight into OPEN System space — no sibling confound (siblings orbit
    // farther out). This is the RETURN crossing whose dest is the OLD, reap-eligible System.
    let out_deadline = Instant::now() + Duration::from_secs(90);
    loop {
        if poll(out_deadline).location.as_deref() == Some("System 7") {
            break;
        }
        let _ = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: (-epoch * 1.5).to_array(),
                arrive_epsilon: 1.0,
                max_ticks: 40,
                max_step_m: 4.0 * DEV.move_speed * DEV.tick_dt,
            },
        );
        assert!(
            Instant::now() < out_deadline,
            "RETURN FROZE: flew inward toward the star but location never returned to \"System 7\" — the \
             return-dest System was reaped mid-crossing and the re-driven crossing parked unresolved.",
        );
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    eprintln!(
        "[repro] RETURN committed — back on System 7 (reap survived; player on a live authority)"
    );

    // ── THE WORLD KEEPS MOVING AFTER YOU CROSS BACK — now a GATE, not an observation. ──
    // This was an observe-and-log for a defect that was ledgered as owed (D-RLM-14 / floating-origin S6 /
    // VU-6): "after a return crossing the surrounding realm feed does not re-advance". A 17-agent audit
    // (2026-08-06) could not reproduce it — 3 runs at HEAD and 1 at 1bb73e8, all LIVE — and then found why
    // it CANNOT happen: the per-tick placement forwarder gates on nothing but a live session and an open
    // channel to the sending shard, and that channel is opened by the very event that makes the crossing
    // commit. A committed crossing cannot leave this feed shut. The ledger described a stall with no
    // mechanism.
    //
    // So the honest work of that slice is to LOCK IN what turned out to be already true. An observation
    // that only logs cannot fail, and behaviour nobody asserts is behaviour waiting to regress quietly.
    // This now fails the day anyone puts a staleness gate, a sequence check or an authority test on that
    // delivery path — which is exactly the change that would break it.
    let base = poll(Instant::now() + Duration::from_secs(5)).realm_frames_applied;
    let observe_until = Instant::now() + Duration::from_secs(15);
    let mut neighbour_feed_live = false;
    // Assigned by the first pass of the loop below (which always runs before either read of it), so
    // seeding it with `base` would be a value nothing can observe — and `-D warnings` says so.
    let mut last_seen;
    loop {
        std::thread::sleep(Duration::from_millis(300));
        let s = poll(observe_until);
        last_seen = s.realm_frames_applied;
        if s.location.as_deref() == Some("System 7") && s.realm_frames_applied >= base + 20 {
            neighbour_feed_live = true;
            break;
        }
        if Instant::now() >= observe_until {
            break;
        }
    }
    // The commit gate MUST still hold after the observation window (the player stays on the live authority).
    assert_eq!(
        poll(Instant::now() + Duration::from_secs(5))
            .location
            .as_deref(),
        Some("System 7"),
        "RETURN REGRESSED: the player did not stay homed on System 7 after the return commit — the reap \
         keep-alive did not hold the return-dest live.",
    );
    assert!(
        neighbour_feed_live,
        "POST-RETURN FREEZE: the player crossed back to System 7 and is live there, but the SURROUNDING \
         realms stopped advancing — the per-tick placement feed did not resume. Applied frames went from \
         {base} to {last_seen} in 15s (needed {} more). The planets are frozen on screen while the player \
         moves among them. Look first at whatever now gates the gateway's per-tick placement forward: it \
         used to gate on nothing but a live session plus an open channel to the sending shard.",
        20,
    );
    eprintln!(
        "[repro] post-return neighbour feed LIVE on System 7 ({base} -> {last_seen} applied frames) — the \
         surrounding realms keep advancing across the cross"
    );
    // (The orbit-slowdown knob clears at the `OrbitSlowdown` guard's drop — panic-safe.)
}

/// LIVE-BUG REPRO (the user's report): after SEVERAL Planet→System→Planet→System rehomes everything FROZE —
/// the client stopped responding to WASD (input dead) and the planets stopped moving. ONE round-trip passes
/// (`a_planet_to_system_return...`); this drives N cycles and asserts the player stays LIVE each cycle: every
/// crossing COMPLETES (a freeze surfaces as a `location` that never flips), AND the own home shard's
/// `universe_tick` keeps advancing across the return (a stalled clock == the sim stopped == WASD dead). The
/// epoch-fixed fly targets (300× orbit slowdown) keep the crossings robust across cycles. Knob reset at the end.
#[test]
fn repeated_planet_system_roundtrips_do_not_freeze() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let _slowdown = OrbitSlowdown::engage();
    let f = fixture("repeatroundtrip");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let p = DEV;
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];
    let deadline = Duration::from_secs(240);

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &p, &client_book, client_quic, devctl_port);

    let landed = await_active(devctl_port, gw_admin, deadline);
    assert_eq!(
        landed.location.as_deref(),
        Some("System 7"),
        "login at the star: {landed:?}"
    );

    let (planet, elements) = inner_planet_mover(&p);
    let planet_seed = match planet {
        vd_core::pose::RealmId::Planet(s) => s,
        other => panic!("inner mover is a planet, got {other:?}"),
    };
    let epoch = vd_core::celestial::orbital_state(&elements, 0.0).position;
    let want_planet = vd_core::pose::FrameRef::PlanetCentered { planet_seed }.label();

    // The own home shard's sim clock (a stalled tick == the sim stopped == WASD dead). Short retry for blinks.
    let tick_now = || -> u64 {
        for _ in 0..30 {
            if let Some(t) = poll_state(devctl_port).and_then(|s| s.universe_tick) {
                return t;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        0
    };
    // Drive a crossing: WalkTo `target` until `location` flips to `loc`. On a timeout, SEPARATE the two
    // failure modes this gate must never confuse — the whole point of the fixture is the FREEZE, and a
    // crude open-loop walk that simply has not arrived yet is NOT one:
    //   * the clock STALLED / the pose stopped moving  ⇒ THE FREEZE (the user's report: the player's
    //     authority stopped feeding the client, so nothing the client does has any effect);
    //   * the clock is live and the pose is still moving ⇒ the client is HEALTHY and merely still in
    //     transit (a fixture-steering shortfall), reported as such and never as a freeze.
    let cross_to = |loc: &str, target: DVec3, phase: &str, cycle: usize| {
        let dl = Instant::now() + Duration::from_secs(90);
        let t_start = tick_now();
        let mut seen_poses: Vec<DVec3> = Vec::new();
        while poll_state(devctl_port).and_then(|s| s.location).as_deref() != Some(loc) {
            let _ = devctl(
                devctl_port,
                &DevRequest::WalkTo {
                    target: target.to_array(),
                    arrive_epsilon: 1.0,
                    max_ticks: 40,
                    max_step_m: 4.0 * DEV.move_speed * DEV.tick_dt,
                },
            );
            if let Some(p) = poll_state(devctl_port).as_ref().and_then(own_pos) {
                seen_poses.push(p);
            }
            if Instant::now() >= dl {
                let t_end = tick_now();
                let moved = seen_poses
                    .iter()
                    .zip(seen_poses.iter().skip(1))
                    .any(|(a, b)| a.distance(*b) > 1e-9);
                assert!(
                    t_end <= t_start || !moved,
                    "at cycle {cycle} ({phase}) the client is LIVE (tick {t_start} -> {t_end}, own pose \
                     moving) but did not reach {loc:?} inside the window — the open-loop WalkTo steering \
                     did not arrive. NOT the freeze; tighten the fixture's steering, not the server.",
                );
                panic!(
                    "FREEZE at cycle {cycle} ({phase}): the client STOPPED (tick {t_start} -> {t_end}, \
                     pose moved: {moved}) and location never became {loc:?} — the player's authority \
                     stopped feeding the client (the WASD-dead / frozen-planets report).",
                );
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = devctl(
            devctl_port,
            &DevRequest::Move {
                axes: [0.0, 0.0, 0.0],
            },
        );
    };

    const CYCLES: usize = 5;
    for cycle in 0..CYCLES {
        cross_to(&want_planet, epoch, "fly-in", cycle);
        eprintln!("[repro] cycle {cycle}: crossed IN to {want_planet}");
        std::thread::sleep(Duration::from_secs(3)); // reap window — System's retained proxy ages out
        cross_to("System 7", -epoch * 1.5, "fly-out", cycle);
        // DIAGNOSE which feed dies after the return: snapshots_applied = the ENTITY feed (the own pose — its
        // stall is the real WASD-dead freeze); realm_frames_applied = the REALM feed (the D-RLM-14 silence);
        // own pose moving under injected input proves input still lands. The steer target is the PLANET (the
        // next cycle's fly-in), NEVER a point further out: dragging the avatar away from the loop's own
        // target would leave each cycle a longer trip than the last and eventually time out for want of
        // travel time — a fixture artifact that masquerades as a freeze.
        let t0 = tick_now();
        for i in 0..8 {
            let _ = devctl(
                devctl_port,
                &DevRequest::WalkTo {
                    target: epoch.to_array(),
                    arrive_epsilon: 1.0,
                    max_ticks: 40,
                    max_step_m: 4.0 * DEV.move_speed * DEV.tick_dt,
                },
            );
            let s = poll_state(devctl_port);
            eprintln!(
                "[diag c{cycle}+{i}s] loc={:?} utick={:?} snaps={:?} realmf={:?} own={:?}",
                s.as_ref().and_then(|s| s.location.clone()),
                s.as_ref().and_then(|s| s.universe_tick),
                s.as_ref().map(|s| s.snapshots_applied),
                s.as_ref().map(|s| s.realm_frames_applied),
                s.as_ref().and_then(own_pos),
            );
            if let Some(adm) = admin(gw_admin) {
                let g = adm.gateway.as_ref();
                eprintln!(
                    "[adm  c{cycle}+{i}s] reaped={} force_reap={} in_unroutable={:?} in_buffered={:?} \
                     in_dropped={:?} self_fenced_lapsed={:?} transfer_unroutable={:?} tc_parked={:?} \
                     frame_sub_desync={:?} home_boot_to={:?}",
                    adm.rlm.teardowns_reaped,
                    adm.rlm.force_reaps,
                    g.map(|g| g.inputs_unroutable),
                    g.map(|g| g.inputs_buffered_for_dest),
                    g.map(|g| g.dest_inputs_dropped),
                    g.map(|g| g.sessions_self_fenced_lapsed),
                    g.map(|g| g.transfer_unroutable),
                    g.map(|g| g.transfer_control_parked),
                    g.map(|g| g.frame_sub_desync),
                    g.map(|g| g.home_bootstrap_timeouts),
                );
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        let t1 = tick_now();
        assert!(
            t1 > t0,
            "FREEZE at cycle {cycle} (post-return): the own shard's universe_tick STALLED ({t0} -> {t1}) — the \
             player's authority stopped feeding the client (WASD dead), even though location is System 7.",
        );
        eprintln!("[repro] cycle {cycle}: returned to System 7 — tick live ({t0} -> {t1})");
    }
    eprintln!("[repro] survived {CYCLES} Planet↔System round-trips — no freeze");
    // (The orbit-slowdown knob clears at the `OrbitSlowdown` guard's drop — panic-safe.)
}

/// THE EXIT-THE-SYSTEM GATE (owner report 2026-08-13: "when I exit the system planets are frozen").
/// A logged-in occupant flies OUT of System 7's 150 m shell, re-homes UP to the galaxy, parks — and
/// the system's planets MUST keep orbiting on its screen. The lane under test is the up-observation
/// relay (owner-approved, PROTO_MINOR 10): the SYSTEM ships the rows it authors one hop up in its own
/// frame; the GALAXY adds the one placement it authors and re-fans to its observers, so the rows the
/// client folds are stated in the space it stands in (the one-space rule passes them). Runs at FULL
/// orbit speed (no slowdown knob): the inner planets sweep several metres over the window, so a
/// frozen feed cannot hide.
#[test]
fn exiting_the_system_keeps_its_planets_orbiting() {
    let _tier = vd_bins::cluster_tier();
    let f = fixture("exitfreeze");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let p = DEV;
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &p, &client_book, client_quic, devctl_port);
    let start = await_active(devctl_port, gw_admin, Duration::from_secs(150));
    assert_eq!(
        start.location.as_deref(),
        Some("System 7"),
        "login at the star: {start:?}"
    );

    // FLY OUT: +X to 220 m — past the 150 m shell into the galaxy — with the braked walk, then park.
    let exit_deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let loc = poll_state(devctl_port).and_then(|s| s.location);
        if loc.as_deref().is_some_and(|l| l != "System 7") {
            eprintln!("[exit] re-homed OUT of System 7 into {loc:?}");
            break;
        }
        let _ = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: [220.0, 0.0, 0.0],
                arrive_epsilon: 1.0,
                max_ticks: 200,
                max_step_m: 4.0 * p.move_speed * p.tick_dt,
            },
        );
        assert!(
            Instant::now() < exit_deadline,
            "NO EXIT: flew to 220 m but location never left \"System 7\"",
        );
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );

    // PARKED OUTSIDE: the system's planets must keep arriving AND keep moving on the drawn scene.
    let planets = |st: &DevState| -> Vec<(String, [f64; 3])> {
        st.realm_boxes
            .iter()
            .filter(|b| b.realm.starts_with("Planet("))
            .map(|b| (b.realm.clone(), b.center))
            .collect()
    };
    let base = poll_state(devctl_port).expect("state after exit");
    let base_frames = base.realm_frames_applied;
    let observe_until = Instant::now() + Duration::from_secs(25);
    let mut latest = base.clone();
    let mut feed_live = false;
    while Instant::now() < observe_until {
        std::thread::sleep(Duration::from_millis(300));
        if let Some(st) = poll_state(devctl_port) {
            latest = st;
            if latest.realm_frames_applied >= base_frames + 20 {
                feed_live = true;
                break;
            }
        }
    }
    assert!(
        feed_live,
        "PLANETS FROZEN AFTER EXIT: realm_frames_applied did not climb by 20 in 25 s outside the \
         system ({} -> {}); the up-observation relay is not reaching the galaxy-standing observer.",
        base_frames, latest.realm_frames_applied,
    );
    // And the DRAWN boxes moved — folding without drawing would be the stale-box defect reborn.
    let (p0, p1) = (planets(&base), planets(&latest));
    assert!(
        !p0.is_empty(),
        "the system's planet boxes are still in the drawn scene after exit"
    );
    let moved = p0
        .iter()
        .filter_map(|(r, c0)| {
            p1.iter().find(|(r1, _)| r1 == r).map(|(_, c1)| {
                ((c1[0] - c0[0]).powi(2) + (c1[1] - c0[1]).powi(2) + (c1[2] - c0[2]).powi(2)).sqrt()
            })
        })
        .fold(0.0_f64, f64::max);
    assert!(
        moved > 1.0,
        "no drawn planet moved over the window from OUTSIDE the system — frozen (best {moved:.3} m)"
    );
    eprintln!(
        "[exit] planets LIVE from the galaxy: feed {} -> {}, best box motion {moved:.2} m",
        base_frames, latest.realm_frames_applied,
    );
}

// The bootstrap-TTL fail-safe leg (a demand login whose home never boots must be CLOSED at
// `bootstrap_ttl_ticks`, not hang) is DEFERRED — see DEFERRED.md D-RLM-9. An empirical spike here found the
// obvious inductions do NOT reach the TTL: with no orchestrator (or the gateway dropped from its peers) the
// gateway never CLOCK-SYNCS, so its seed injector stays inert and no demand is ever emitted — the login sits
// with an open session and `home_bootstrap_timeouts == 0` (a separate pre-sync-hold concern). The TTL fires
// only for a SYNCED gateway whose home genuinely fails to boot, which CA-1 learning (the orchestrator learns
// the gateway from the demand frame and delivers the grant anyway) makes delicate to induce.

// ---- The demand-FLY: a MOVING occupant streams a culled planet in AHEAD + reaps the vacated realm BEHIND ---

/// Poll one orchestrator RLM gauge until it holds STEADY for [`STABLE_HOLDS`] consecutive reads or
/// `cap` elapses, then return the settled count — the shared settle for [`settle_spins`] (the
/// stationary occupant has finished spinning up everything already in its visibility) and
/// [`settle_reaps`] (the realms the occupant already vacated have finished draining out).
const STABLE_HOLDS: u32 = 5;
fn settle_gauge(admin_addr: SocketAddr, cap: Duration, read: impl Fn(&RlmView) -> u64) -> u64 {
    let started = Instant::now();
    let mut last = read(&orch_rlm(admin_addr));
    let mut holds = 0u32;
    while started.elapsed() < cap {
        std::thread::sleep(Duration::from_millis(300));
        let now = read(&orch_rlm(admin_addr));
        if now == last {
            holds += 1;
            if holds >= STABLE_HOLDS {
                break;
            }
        } else {
            last = now;
            holds = 0;
        }
    }
    last
}

/// The settled `spins_requested` — the baseline a fly-out must climb ABOVE for a NON-VACUOUS
/// spin-up-ahead: the neighbour star is asleep at login (the ring is wider than the wake radius),
/// so it is NOT in this baseline and can only spin up as the ship approaches.
fn settle_spins(admin_addr: SocketAddr, cap: Duration) -> u64 {
    settle_gauge(admin_addr, cap, |r| r.spins_requested)
}

/// The settled `teardowns_reaped` — the baseline the return leg must climb ABOVE for a NON-VACUOUS
/// reap-behind: the home system's own vacated planets reap during the outbound flight, so the
/// return's assertion must start from wherever that left the gauge.
fn settle_reaps(admin_addr: SocketAddr, cap: Duration) -> u64 {
    settle_gauge(admin_addr, cap, |r| r.teardowns_reaped)
}

fn own_row(state: &DevState) -> Option<&DevEntityRow> {
    let own = state.own_entity.as_deref()?;
    state.entities.iter().find(|r| r.entity == own)
}

fn own_pos(state: &DevState) -> Option<DVec3> {
    own_row(state).map(|r| DVec3::from_array(r.pos))
}

/// The orchestrator's RLM view (`spins_requested` / `spins_failed` / `teardowns_reaped` …); a zeroed view if
/// the admin endpoint blinks (a poll loop tolerates that).
fn orch_rlm(admin_addr: SocketAddr) -> RlmView {
    admin(admin_addr).map(|s| s.rlm).unwrap_or_default()
}

/// The nearest ring NEIGHBOUR of the home star: its realm and its authored centre in the galaxy's
/// frame — THROUGH THE PRODUCTION BOOT (the same discipline as `inner_planet_mover`): the regions come
/// from `vd_bins::boot_regions_and_movers` hosted AT the galaxy, so the aim IS the placement the
/// galaxy authors. The home star (System 7, ring index 0) sits at the galactic origin; every other
/// star is a ring sibling, all equidistant from home — the lowest seed is picked for determinism.
fn neighbour_system(p: &DevClusterParams) -> (vd_core::pose::RealmId, DVec3) {
    let galaxy = vd_core::pose::RealmId::System(vd_bins::GALAXY_SEED);
    let held = std::collections::BTreeSet::from([galaxy]);
    let (regions, _moving) =
        vd_bins::boot_regions_and_movers(p.universe_seed, &held, galaxy, p.move_speed, p.tick_dt);
    let (seed, region) = regions
        .into_iter()
        .filter(|r| r.parent == Some(galaxy))
        .filter_map(|r| match r.realm {
            vd_core::pose::RealmId::System(seed) if seed != 7 => Some((seed, r)),
            _ => None,
        })
        .min_by_key(|(seed, _)| *seed)
        .expect("the multi-star galaxy has a ring neighbour");
    assert_eq!(
        region.center.cell(),
        vd_core::glam::I64Vec3::ZERO,
        "a ring placement fits inside one lattice cell",
    );
    (vd_core::pose::RealmId::System(seed), region.center.offset())
}

/// Drive ONE walk leg to the 3D `target`, settle the sticky Move, read the delivered own pose, and assert the
/// occupant ARRIVED — movement never froze across the demand spin-ups. A generous `max_ticks` so a slow spawn
/// never times out mid-leg (the point is that the dot keeps MOVING while realms stream in around it).
fn walk_leg(devctl_port: u16, leg: &str, target: DVec3, max_ticks: u64) -> DVec3 {
    let _ = devctl(
        devctl_port,
        &DevRequest::WalkTo {
            target: target.to_array(),
            arrive_epsilon: 2.0,
            max_ticks,
            max_step_m: 4.0 * DEV.move_speed * DEV.tick_dt,
        },
    )
    .unwrap_or_else(|| panic!("leg {leg}: no walk response"));
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    let landed = own_pos(&poll_state(devctl_port).expect("state after leg")).expect("own pos");
    assert!(
        (landed - target).length() <= 3.0,
        "leg {leg} FROZE: the occupant stalled at {landed:?}, never reaching {target:?} — a demand spin-up \
         stalled the walk, or the walk itself froze.",
    );
    landed
}

#[test]
fn a_flying_occupant_streams_a_neighbour_system_in_ahead_then_the_vacated_realm_is_reaped() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // THE demand-FLY, rewritten against the multi-star ring (the D-RLM-16 rewrite): since the
    // wider-visibility world (32fcc5c) every planet of the HOME system is already visible at login,
    // so the genuinely culled realm is the NEIGHBOUR STAR ~12 km down the ring — asleep, because the
    // ring is deliberately 1.05× the wake radius. The occupant exits the home system, flies toward
    // the neighbour, and parks INSIDE its wake band while still ~10 km outside its 150 m shell:
    //   1. the neighbour's shard spins up AHEAD (the demand loop — spins climb above the settled
    //      login baseline, warp as a consequence of movement);
    //   2. its planets' BOXES stream into the drawn scene (slice C's shape mirror: the live system
    //      ships its interior outlines one hop up, the galaxy adds the one placement it authors);
    //   3. those boxes MOVE (the rows lane animating them at full orbit speed);
    // all while `location` never leaves the between-space — the world arrives ahead of any crossing.
    // Flying back out of the band, the vacated neighbour empties and is reaped BEHIND. This is the
    // owner-flown gap of 2026-08-13 ("approaching another star system, its planets were not
    // loading, though inside my AoI and visibility"), as a gate.
    let f = fixture("fly");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let p = DEV;
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];
    // Spawn + exit + a ~2 km flight + the drain/quiesce teardown windows ⇒ a materially longer
    // budget than a login.
    let deadline = Duration::from_secs(120);

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &p, &client_book, client_quic, devctl_port);

    // The login lands the occupant at the home star; everything already in view spins up and settles
    // — the baseline the approach must climb above for a NON-VACUOUS spin-up-ahead.
    let landed = await_active(devctl_port, gw_admin, deadline);
    assert_eq!(
        landed.location.as_deref(),
        Some("System 7"),
        "the login lands at the star home realm: {landed:?}",
    );
    let settled_spins = settle_spins(a.admin, Duration::from_secs(30));

    // The aim, through the production boot: the neighbour star's authored ring placement. The
    // precondition ties the flight to the multi-star geometry — the neighbour is far beyond the home
    // shell, so it CANNOT be in the settled baseline (the ring is wider than the wake radius).
    let (neighbour, centre) = neighbour_system(&p);
    let ring_r = centre.length();
    assert!(
        ring_r > 1_000.0,
        "the ring neighbour {neighbour} is far outside the home shell: {ring_r} m",
    );

    // ---- LEG 0: exit the home system (re-home UP into the galaxy) — the exit-gate pattern. ----
    let exit_deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let loc = poll_state(devctl_port).and_then(|s| s.location);
        if loc.as_deref().is_some_and(|l| l != "System 7") {
            eprintln!("[fly] re-homed OUT of System 7 into {loc:?}");
            break;
        }
        let _ = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: [220.0, 0.0, 0.0],
                arrive_epsilon: 1.0,
                max_ticks: 200,
                max_step_m: 4.0 * p.move_speed * p.tick_dt,
            },
        );
        assert!(
            Instant::now() < exit_deadline,
            "NO EXIT: flew to 220 m but location never left \"System 7\"",
        );
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    let galaxy_loc = poll_state(devctl_port)
        .and_then(|s| s.location)
        .expect("a location outside the home system");

    // The drawn scene BEFORE the approach: whatever planet boxes it holds are the HOME system's; the
    // sleeping neighbour's interior is nowhere in it.
    let planet_boxes = |st: &DevState| -> Vec<(String, [f64; 3])> {
        st.realm_boxes
            .iter()
            .filter(|b| b.realm.starts_with("Planet("))
            .map(|b| (b.realm.clone(), b.center))
            .collect()
    };
    let baseline_planets: std::collections::BTreeSet<String> =
        planet_boxes(&poll_state(devctl_port).expect("state after exit"))
            .into_iter()
            .map(|(name, _)| name)
            .collect();

    // ---- LEG 1: fly toward the neighbour, park INSIDE its wake band, far outside its shell. ----
    // The wake radius is ring/1.05 (both derive from the one visibility angle), so parking at
    // 0.85·ring leaves a >1 km margin inside the band and >10 km outside any containment.
    walk_leg(devctl_port, "toward-neighbour-star", centre * 0.15, 3_000);
    let out = poll_state(devctl_port).expect("state after leg 1");
    assert_eq!(
        out.location.as_deref(),
        Some(galaxy_loc.as_str()),
        "parked in the between-space — the world must arrive AHEAD of any crossing: {out:?}",
    );
    assert!(
        out.own_entity.is_some(),
        "the occupant kept authority across the approach: {out:?}",
    );
    assert_eq!(
        out.decode_errors, 0,
        "the client saw no decode faults across the approach: {out:?}",
    );

    // 1. Spin-up-AHEAD: entering the wake band demanded the sleeping neighbour (a climb above the
    // settled baseline is that spin-up — nothing else was left to wake). Deliberately NO
    // `spins_failed == 0` here: the serial suite reuses ONE fixed RLM port band, so a spawn can
    // transiently collide with the previous test's not-yet-expired TIME_WAIT sockets — the
    // reconciler backs off and retries by design, and the proof the spawn SUCCEEDED is step 2 (the
    // boxes cannot stream from a shard that is not running). Asserting zero here measured fixture
    // hygiene, not the product (flushed 2026-08-13: fail_streak 1, then a clean retry).
    let started = Instant::now();
    loop {
        let flew = orch_rlm(a.admin);
        if flew.spins_requested > settled_spins {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the neighbour star never spun up ahead (spins stayed {settled_spins}): {flew:?}",
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    // 2. THE OWNER'S GAP, as a gate: planet boxes the baseline never held stream into the drawn
    // scene while the ship sits parked 10 km out — the neighbour's interior, boxes ahead of arrival.
    let discover_deadline = Instant::now() + Duration::from_secs(45);
    let fresh: std::collections::BTreeMap<String, [f64; 3]> = loop {
        let now: std::collections::BTreeMap<String, [f64; 3]> = poll_state(devctl_port)
            .map(|st| {
                planet_boxes(&st)
                    .into_iter()
                    .filter(|(name, _)| !baseline_planets.contains(name))
                    .collect()
            })
            .unwrap_or_default();
        if !now.is_empty() {
            break now;
        }
        if Instant::now() >= discover_deadline {
            let drawn: Vec<String> = poll_state(devctl_port)
                .map(|st| st.realm_boxes.iter().map(|b| b.realm.clone()).collect())
                .unwrap_or_default();
            panic!(
                "NO NEW PLANET BOXES: the neighbour's interior never reached the drawn scene \
                 (baseline {baseline_planets:?}; full drawn scene now {drawn:?})",
            );
        }
        std::thread::sleep(Duration::from_millis(300));
    };
    eprintln!(
        "[fly] the neighbour's interior streamed in ahead: {:?}",
        fresh.keys().collect::<Vec<_>>()
    );

    // 3. ...and the streamed boxes MOVE — outlines alone would be the stale-box defect reborn.
    let observe_until = Instant::now() + Duration::from_secs(25);
    let mut best = 0.0_f64;
    while Instant::now() < observe_until && best <= 1.0 {
        std::thread::sleep(Duration::from_millis(300));
        if let Some(st) = poll_state(devctl_port) {
            for (name, c1) in planet_boxes(&st) {
                if let Some(c0) = fresh.get(&name) {
                    let d = ((c1[0] - c0[0]).powi(2)
                        + (c1[1] - c0[1]).powi(2)
                        + (c1[2] - c0[2]).powi(2))
                    .sqrt();
                    best = best.max(d);
                }
            }
        }
    }
    assert!(
        best > 1.0,
        "the neighbour's streamed boxes never moved (best {best:.3} m) — outlines without rows",
    );

    // ---- LEG 2: fly back out of the band → the vacated neighbour reaps BEHIND. ----
    // The home system's own vacated planets reap during the outbound flight; settle that first so
    // the return's climb can only be the neighbour. Anti-thrash rests entirely on the ~1 s grace
    // (equal AoI factors ⇒ no geometric dead-zone), so the reap LAGS the geometric exit by the grace
    // tail + drain + quiesce — a polled assertion, bounded by the deadline.
    let settled_reaps = settle_reaps(a.admin, Duration::from_secs(30));
    walk_leg(
        devctl_port,
        "back-toward-home",
        centre.normalize() * 220.0,
        3_000,
    );
    let started = Instant::now();
    loop {
        let r = orch_rlm(a.admin);
        if r.teardowns_reaped > settled_reaps {
            break;
        }
        assert!(
            started.elapsed() < deadline,
            "the vacated neighbour was never torn down/reaped (reaped stayed {settled_reaps}): {r:?}",
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}
