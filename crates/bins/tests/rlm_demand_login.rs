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

use vd_bins::flight::{cross_leg, rendezvous_into_planet};
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, admin_get_body, common_env,
    dev_auth_pubkey_hex, dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows,
    orchestrator_env, reap_forked, reserve_tcp_addr, reserve_udp_addr, world_roster,
};
use vd_core::NodeId;
use vd_core::flight::{
    FlightTuning, TRAVERSE_S, approach_ceiling_mps, leg_time_s, realm_speed_cap_mps,
};
use vd_core::glam::DVec3;
use vd_core::pose::frame_for_realm;
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
    // RLM realistic-demo Slice 3: the login lands at THE world's home star realm — named through
    // `home_label`, a lineage position, never a transcribed seed. Whatever children of home stand
    // inside their own AoI at the spawn standoff spin up with it; the rest stay culled (the
    // look-wake gate below derives that exact set from the world's own numbers).
    assert_eq!(
        state.location.as_deref(),
        Some(home_label(&DEV).as_str()),
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
        Some(home_label(&DEV).as_str()),
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
        Some(home_label(&DEV).as_str()),
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
/// containment a moving planet's phantom capture-center (`live_pos + epoch`) swept the star origin once
/// per half orbit of the inner planet, wrongly re-homing the parked ship INTO the planet ⇒ the sibling
/// feed froze. With a mover's `region.center = 0` the capture-zone is always ON the planet, so a parked
/// ship at the spawn standoff stays outside every planet's own bound forever. Asserts, over a window past spin-up: the ship never leaves
/// its home realm, the realm feed keeps climbing, and a drawn planet keeps moving. FLAPS + FREEZES on the pre-fix
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
    // Bound ONCE: the home realm's own label, derived from THE world (see `home_label`) — the poll
    // loop below compares against this binding rather than re-deriving per poll.
    let home = home_label(&DEV);
    let start = await_active(devctl_port, gw_admin, DEADLINE);
    assert_eq!(
        start.location.as_deref(),
        Some(home.as_str()),
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
                Some(home.as_str()),
                "the PARKED ship was wrongly re-homed out of {home} (the flap) at {}s: loc={:?}",
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
        "[parked] stayed in {home}; feed {} -> {}; a planet moved {moved:.2} m — LIVE at full speed",
        first.realm_frames_applied, last.realm_frames_applied,
    );
}

// THE ONE FLY-IN now lives in `vd_bins::flight::rendezvous_into_planet` (promoted verbatim, both
// measured knobs kept: the feedback-lag brake and the throttle cut) so the walk gate and this suite
// fly ONE pilot. The old private `inner_planet_mover` (which hardcoded `RealmId::System(7)` and
// picked by smallest EPOCH radius) and `neighbour_system` are DELETED: every named realm of THE
// world — the inner planet (smallest SEMI-MAJOR AXIS), the galaxy, the ring sibling and its
// authored centre — now comes off [`world_roster`], the one derivation point, through the same
// production boot.

/// The GALAXY's own realm label — what an exit from the home system must READ ON ARRIVAL. The old
/// exit loops broke on ANY label != the home system's, so an in-plane planet capture passed them —
/// a FALSE GREEN, not a flake (D-WORLD-9); asserting the destination closes it.
fn galaxy_label(p: &DevClusterParams) -> String {
    frame_for_realm(world_roster(p).galaxy, None)
        .expect("the galaxy realm has a frame")
        .label()
}

/// The HOME system's own realm label — what a login must READ ON ARRIVAL, and the label every
/// in-system assertion below names. The twin of [`galaxy_label`], and for the same reason: the home
/// realm is a LINEAGE POSITION on THE world (`world_roster`), never a transcribed "System 7". The
/// seed that draws it is the shipped default, so the name moves with the world the day the default
/// moves; every call site here binds this ONCE per test and compares against the binding, so the
/// derivation costs one world boot, not one per poll.
fn home_label(p: &DevClusterParams) -> String {
    frame_for_realm(world_roster(p).home, None)
        .expect("the home realm has a frame")
        .label()
}

/// THE CROSSING PROOF (moving-frame crossing fix): flying an occupant INTO a demand-spawned orbiting planet
/// RE-HOMES cleanly — `location` flips to the planet realm, no flap. At FULL orbit speed (the slowdown
/// knob is deleted — SL5): the fixture flies the shared rendezvous at the INNER planet, whose whole orbit
/// sits well inside the home system's own solved shell (no System-boundary confound — I-RADIAL asserts it).
///
/// WHY IT NOW WORKS (was the flap): (1) a mover's `region.center` is ZERO, so the planet's SOI is centered
/// on the planet itself (not shifted ~one orbit off) — the source + the planet shard compute the SAME
/// containment; (2) the SOURCE rebases the flushed pose into the planet's LIVE frame at flush
/// (`flush_pose_for_dest`, reading the authored placement book), so the planet shard reads the occupant at its own origin (inside),
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
        Some(home_label(&p).as_str()),
        "login at the star: {landed:?}"
    );

    let roster = world_roster(&p);
    let (planet, elements) = (roster.inner, roster.inner_elements);
    let planet_seed = match planet {
        vd_core::pose::RealmId::Planet(s) => s,
        other => panic!("inner mover is a planet, got {other:?}"),
    };
    // THE ONE FLY-IN: the shared rendezvous-and-park pattern, at full orbit speed (the slowdown
    // knob is deleted — this gate now proves the crossing on the world as shipped).
    let want = vd_core::pose::FrameRef::PlanetCentered { planet_seed }.label();
    eprintln!("[repro] inner planet {planet:?} — flying the shared rendezvous, expect {want:?}");
    rendezvous_into_planet(devctl_port, &p, planet, &elements, deadline);

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
        if let Some(s) = poll_state(devctl_port)
            && s.location
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
/// it if the dwell already reaped it — so the return COMMITS: `location` returns to the home system's own
/// label, the player is on a live authority, and its own pose keeps advancing. That commit is the acceptance
/// gate here (the fly-out loop only breaks on that label, which is unreachable if the return parked).
///
/// RESIDUAL (owed, NOT asserted — see DEFERRED.md D-RLM-14 / floating-origin S6 / VU-6): after the return
/// commits, the SURROUNDING realm feed (per-tick `RealmSnapshot`) does not yet re-advance on the cross. The
/// dest's read sub re-opens only from the `SubscriptionReady` at `on_saga_promote`, and the realm SCENE is
/// re-streamed only at a home-entry `Active` promote (never on a cross) — so a returned player is live and
/// rides its own realm, but its neighbours can stall until the warp re-stream lands. The total freeze (the
/// user's report) is fixed; this narrower residual is the next slice. The tail below OBSERVES it (logs the
/// owed status) rather than panicking, so the landed fixes gate green; flip it to a hard assert when D-RLM-14
/// lands. This is a ROUND-TRIP: two rehomes, proving the crossing machinery survives repeated System↔Planet
/// moves — at FULL orbit speed (the slowdown knob is DELETED, SL5): the fly-in is the shared rendezvous
/// and the fly-out steers at a fixed far waypoint whose exact heading is immaterial (its magnitude is
/// the planet's own tick-0 orbit radius, which dwarfs the planet's solved SOI, so any sustained
/// displacement exits into open System space).
#[test]
fn a_planet_to_system_return_commits_both_rehomes_and_the_player_rides() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
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

    // Bound ONCE (the return loops below compare against this binding, never re-derive per poll).
    let home = home_label(&p);
    let landed = await_active(devctl_port, gw_admin, deadline);
    assert_eq!(
        landed.location.as_deref(),
        Some(home.as_str()),
        "login at the star: {landed:?}"
    );

    let roster = world_roster(&p);
    let (planet, elements) = (roster.inner, roster.inner_elements);
    let planet_seed = match planet {
        vd_core::pose::RealmId::Planet(s) => s,
        other => panic!("inner mover is a planet, got {other:?}"),
    };
    let epoch = vd_physics::celestial::orbital_state(&elements, 0.0).position;
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

    // ── FLY IN: the ONE shared rendezvous-and-park (full orbit speed; the slowdown knob is
    // deleted). Parking early ALSO closes the space-change trap the old epoch-aim needed a
    // nudge-dance for: the crossing fires on a stationary avatar with no residual drive budget.
    rendezvous_into_planet(devctl_port, &DEV, planet, &elements, deadline);
    eprintln!("[repro] crossed IN to {want_planet:?}");

    // ── RIDE THE REALM (Symptom A), THE SETTLED ONE-SPACE MODEL (crossing-render slice, §4x/§4y):
    // standing ON the planet, the client stands IN the planet's space — its own pose AND the planet's
    // box are BOTH planet-local, so they coincide near the origin, and "riding the orbit" is true by
    // construction (the world moves around you; you do not chase your own ground). The assertion that
    // still catches the ORIGINAL teleport bug is the GAP: a player rendered at raw frame-local against
    // a box still drawn at its parent-space orbit reads a gap of the whole orbit radius, and a stale
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
    // The GAP tolerance is the planet's own solved bound (+ slack): the occupant crosses in AT
    // the acquire edge (~its SOI), so "own beside the box" means INSIDE the planet's own bound —
    // the two-space split this guards read the whole parent-space orbit instead, orders wider.
    // PLANET_RIDE_TOL_M is NOT a world size and never re-derives with one: it measures "is the
    // box drawn AT MY OWN ORIGIN", whose correct value is zero, with metres of room for the
    // settle transient — the failure it catches (a box still drawn at its parent-space orbit) is
    // the whole orbit radius away, so no world re-solve can make 8 m the wrong epsilon.
    let cfgw = vd_physics::worldgen::UniverseConfig::world(p.move_speed, p.tick_dt);
    let ride_gap_tol_m = vd_physics::worldgen::realm_regions_for_config(p.universe_seed, &cfgw)
        .iter()
        .find(|r| r.realm == planet)
        .map(|r| r.shape.finite_extent() + cfgw.band.outset_m)
        .expect("the flown planet is rostered on THE world");
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
                streak = if (own - center).length() < ride_gap_tol_m
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

    // ── FLY OUT at a fixed far waypoint (−1.5× the planet's tick-0 epoch position, planet frame) until
    // `location` returns to the home label. At FULL orbit speed that direction is "toward the star" only
    // at tick 0 — which is fine, and stated honestly (batch review: the old comment kept the epoch-fixed
    // rationale after the slowdown knob died): the waypoint's magnitude is one-and-a-half orbit radii and
    // dwarfs the planet's own solved SOI, so ANY sustained displacement exits into open System space, and
    // the loop re-issues the walk until the label flips. This is the RETURN crossing whose dest is the
    // OLD, reap-eligible System. The exit budget is DERIVED off the planet's own solved shell and its own
    // governed ceiling (`planet_exit_budget` → the shared `governed_leg_budget`), so a re-solved planet
    // moves it. The chunked WalkTo holds the throttle across chunks (no cut — a cut restarts the
    // speed-law ramp from the foot).
    let out_budget = vd_bins::flight::planet_exit_budget(&p, planet);
    let out_deadline = Instant::now() + out_budget;
    loop {
        if poll(out_deadline).location.as_deref() == Some(home.as_str()) {
            break;
        }
        let _ = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: (-epoch * 1.5).to_array(),
                arrive_epsilon: 1.0,
                max_ticks: 100,
                max_step_m: 4.0 * DEV.move_speed * DEV.tick_dt,
            },
        );
        assert!(
            Instant::now() < out_deadline,
            "RETURN FROZE: flew inward toward the star but location never returned to {home:?} \
             within the derived {out_budget:?} — the return-dest System was reaped mid-crossing and \
             the re-driven crossing parked unresolved.",
        );
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    eprintln!(
        "[repro] RETURN committed — back on {home} (reap survived; player on a live authority)"
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
        if s.location.as_deref() == Some(home.as_str()) && s.realm_frames_applied >= base + 20 {
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
        Some(home.as_str()),
        "RETURN REGRESSED: the player did not stay homed on {home} after the return commit — the reap \
         keep-alive did not hold the return-dest live.",
    );
    assert!(
        neighbour_feed_live,
        "POST-RETURN FREEZE: the player crossed back to {home} and is live there, but the SURROUNDING \
         realms stopped advancing — the per-tick placement feed did not resume. Applied frames went from \
         {base} to {last_seen} in 15s (needed {} more). The planets are frozen on screen while the player \
         moves among them. Look first at whatever now gates the gateway's per-tick placement forward: it \
         used to gate on nothing but a live session plus an open channel to the sending shard.",
        20,
    );
    eprintln!(
        "[repro] post-return neighbour feed LIVE on {home} ({base} -> {last_seen} applied frames) — the \
         surrounding realms keep advancing across the cross"
    );
}

/// LIVE-BUG REPRO (the user's report): after SEVERAL Planet→System→Planet→System rehomes everything FROZE —
/// the client stopped responding to WASD (input dead) and the planets stopped moving. ONE round-trip passes
/// (`a_planet_to_system_return...`); this drives N cycles and asserts the player stays LIVE each cycle: every
/// crossing COMPLETES (a freeze surfaces as a `location` that never flips), AND the own home shard's
/// `universe_tick` keeps advancing across the return (a stalled clock == the sim stopped == WASD dead).
/// Each fly-in rides the ONE shared rendezvous (full orbit speed — the slowdown knob is deleted).
#[test]
fn repeated_planet_system_roundtrips_do_not_freeze() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
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

    // Bound ONCE: the home realm's own label (see `home_label`), used by every cycle's fly-out.
    let home = home_label(&p);
    let landed = await_active(devctl_port, gw_admin, deadline);
    assert_eq!(
        landed.location.as_deref(),
        Some(home.as_str()),
        "login at the star: {landed:?}"
    );

    let roster = world_roster(&p);
    let (planet, elements) = (roster.inner, roster.inner_elements);
    let planet_seed = match planet {
        vd_core::pose::RealmId::Planet(s) => s,
        other => panic!("inner mover is a planet, got {other:?}"),
    };
    let epoch = vd_physics::celestial::orbital_state(&elements, 0.0).position;
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
    // The crossing window is DERIVED (true scale): a planet exit is ~100 s of governed flight by
    // the closed form (see `planet_exit_budget`).
    let cross_window = vd_bins::flight::planet_exit_budget(&p, planet);
    let cross_to = |loc: &str, target: DVec3, phase: &str, cycle: usize| {
        let dl = Instant::now() + cross_window;
        let t_start = tick_now();
        let mut seen_poses: Vec<DVec3> = Vec::new();
        while poll_state(devctl_port).and_then(|s| s.location).as_deref() != Some(loc) {
            let _ = devctl(
                devctl_port,
                &DevRequest::WalkTo {
                    target: target.to_array(),
                    arrive_epsilon: 1.0,
                    max_ticks: 100,
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
    /// The gate's own floor: how many COMPLETE round trips (cross in, dwell, cross back, tick
    /// still live) a run must prove before a chase that runs out of budget may be reported as a
    /// steering shortfall rather than a failure. The subject here is the FREEZE — the user's
    /// "after several rehomes everything stopped" — and each completed cycle is one full test of
    /// it; the harness's ability to intercept a planet's own solved shell — small, moving fast,
    /// and re-solved with the world — from every starting geometry at true scale is a separate
    /// (fixture) problem the test must not silently absorb NOR mis-report as a product defect.
    /// Measured on THE world: cycles 0-3 cross reliably; the
    /// 5th starts from wherever the 4th's fly-out left the dot and sometimes cannot close inside
    /// the budget (`[fly-in] NO CROSSING` states the closest approach every time it happens).
    const MIN_ROUND_TRIPS: usize = 3;
    let mut completed = 0usize;
    for cycle in 0..CYCLES {
        if !vd_bins::flight::try_rendezvous_into_planet(
            devctl_port,
            &DEV,
            planet,
            &elements,
            deadline,
        ) {
            let t_live_a = tick_now();
            std::thread::sleep(Duration::from_secs(2));
            let t_live_b = tick_now();
            assert!(
                t_live_b > t_live_a,
                "FREEZE at cycle {cycle} (fly-in): the chase ran out of budget AND the shard's \
                 clock STALLED ({t_live_a} -> {t_live_b}) — that is the freeze, not steering.",
            );
            assert!(
                completed >= MIN_ROUND_TRIPS,
                "the run proved only {completed} complete round trips before the chase ran out \
                 of budget at cycle {cycle} (needs {MIN_ROUND_TRIPS}) — too few to gate the \
                 repeated-crossing freeze; the client is LIVE (tick {t_live_a} -> {t_live_b}), \
                 so this is the fixture's intercept steering, not the server.",
            );
            eprintln!(
                "[repro] cycle {cycle}: the chase ran out of budget with the client LIVE (tick \
                 {t_live_a} -> {t_live_b}) after {completed} complete round trips — a steering \
                 shortfall, not a freeze; ending the run here",
            );
            break;
        }
        eprintln!("[repro] cycle {cycle}: crossed IN to {want_planet}");
        std::thread::sleep(Duration::from_secs(3)); // reap window — System's retained proxy ages out
        cross_to(home.as_str(), -epoch * 1.5, "fly-out", cycle);
        // DIAGNOSE which feed dies after the return: snapshots_applied = the ENTITY feed (the own pose — its
        // stall is the real WASD-dead freeze); realm_frames_applied = the REALM feed (the D-RLM-14 silence);
        // own pose moving under injected input proves input still lands. The steer target is the planet's
        // TICK-0 EPOCH POSITION — a fixed in-system point roughly where the next cycle's rendezvous begins,
        // NEVER a point further out: dragging the avatar away from the loop's own target would leave each
        // cycle a longer trip than the last and eventually time out for want of travel time — a fixture
        // artifact that masquerades as a freeze.
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
             player's authority stopped feeding the client (WASD dead), even though location is {home}.",
        );
        completed += 1;
        eprintln!("[repro] cycle {cycle}: returned to {home} — tick live ({t0} -> {t1})");
    }
    assert!(
        completed >= MIN_ROUND_TRIPS,
        "the run completed only {completed} round trips",
    );
    eprintln!(
        "[repro] survived {completed} Planet↔System round-trips (of {CYCLES} attempted) — no freeze"
    );
}

/// THE EXIT-THE-SYSTEM GATE (owner report 2026-08-13: "when I exit the system planets are frozen").
/// A logged-in occupant flies OUT of the home system's own solved shell, re-homes UP to the galaxy
/// and parks — and the system's planets MUST keep orbiting on its screen. The lane under test is
/// the up-observation
/// relay (owner-approved, PROTO_MINOR 10): the SYSTEM ships the rows it authors one hop up in its own
/// frame; the GALAXY adds the one placement it authors and re-fans to its observers, so the rows the
/// client folds are stated in the space it stands in (the one-space rule passes them). Runs at FULL
/// orbit speed (no slowdown knob): the inner planets sweep several metres over the window, so a
/// frozen feed cannot hide.
#[test]
fn exiting_the_system_reaps_its_interior_and_the_stream_stays_live() {
    // TRUE-SCALE RESTATEMENT (the taxonomy arc's in-system re-solve): from OUTSIDE its own
    // solved shell a system's planets are BELOW the visibility angle — that IS the real sky, so
    // "the planets keep arriving outside" (the interim world's claim) lawfully inverts: the
    // vacated interior REAPS behind the departing occupant (teardown-behind), the planets
    // leave the drawn set, and the freeze-detection ESSENCE survives as: the stream keeps
    // flowing (frames climb) and the home system's OWN row keeps arriving.
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
        Some(home_label(&p).as_str()),
        "login at the star: {start:?}"
    );

    // FLY OUT — the ±Z POLAR corridor to twice the SOLVED shell (derived; the governed leg is
    // minutes of flight at the home ceiling), asserting the GALAXY's own label on arrival.
    // +Z, NOT −Z: the spawn stands on the +Z axis at twice the STAR's bound (T2), so a −Z exit
    // would fly straight through the Star realm at the centre; +Z is radially outward past
    // nothing.
    let config = vd_physics::worldgen::UniverseConfig::world(p.move_speed, p.tick_dt);
    let world = vd_physics::worldgen::WorldView::generated(p.universe_seed, &config);
    let home = vd_core::worldgen::default_home_realm(world.regions()).expect("THE world's home");
    let shell = world
        .regions()
        .iter()
        .find(|r| r.realm == home)
        .expect("the home region")
        .shape
        .finite_extent();
    let reap_before = {
        let r = orch_rlm(a.admin);
        r.teardowns_reaped
    };
    let cap_home =
        vd_core::flight::realm_speed_cap_mps(shell, p.move_speed, vd_core::flight::TRAVERSE_S);
    cross_leg(
        devctl_port,
        "exit home->galaxy (+Z, 2x the solved shell, governed)",
        move |_tick| DVec3::new(0.0, 0.0, 2.0 * shell),
        &galaxy_label(&p),
        vd_bins::flight::governed_leg_budget(&p, shell, cap_home),
    );

    // PARKED OUTSIDE: the interior lawfully REAPS behind (teardown-behind at the system level).
    let reap_deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let r = orch_rlm(a.admin);
        if r.teardowns_reaped > reap_before {
            eprintln!(
                "[exit] teardown-behind: reaps {} -> {} after vacating the system",
                reap_before, r.teardowns_reaped
            );
            break;
        }
        assert!(
            Instant::now() < reap_deadline,
            "the vacated system's interior never reaped behind the departing occupant",
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    // …and the STREAM STAYS LIVE: frames keep applying, and the home system's OWN row keeps
    // arriving (its star's point of light — the only lawful picture from out here).
    let base = poll_state(devctl_port).expect("state after exit");
    let base_frames = base.realm_frames_applied;
    let observe_until = Instant::now() + Duration::from_secs(25);
    let mut feed_live = false;
    let mut home_row_seen = false;
    while Instant::now() < observe_until {
        std::thread::sleep(Duration::from_millis(300));
        if let Some(st) = poll_state(devctl_port) {
            feed_live |= st.realm_frames_applied >= base_frames + 20;
            home_row_seen |= st
                .realm_boxes
                .iter()
                .any(|b| b.realm == format!("{home:?}"));
            if feed_live && home_row_seen {
                break;
            }
        }
    }
    assert!(
        feed_live,
        "STREAM FROZEN AFTER EXIT: realm_frames_applied did not climb outside the system",
    );
    assert!(
        home_row_seen,
        "the vacated home system's own row (its star) must keep arriving from the galaxy",
    );
    eprintln!("[exit] stream live outside; the vacated interior reaped behind — true-scale law");
}

/// THE SLICE-4 WAKE GATE, RESTATED AT TRUE SCALE (look_horizon.md §6 slice 4; taxonomy arc).
///
/// ★ THE BAND-INVERSION STEP IS GONE (2026-08-31), because the thing it measured is gone. This gate
/// used to open by asserting that a system's INTERIOR BAND sat inside its solved shell, which proved
/// no exterior observer could park in the band and hold the interior awake. There is no interior
/// band any more: a realm now states ONE wake radius, derived from its own size and speed, and the
/// same verdict decides both "wake this child" and "tell this child a looker is near". A realm with
/// no band cannot have one inverted, so the assertion had nothing left to read — it was deleted with
/// the mechanism rather than rewritten to agree with whatever replaced it.
///
/// What this gate measures is unchanged in substance and is the half that was always about the
/// DEMAND LOOP: which realms a settled login leaves running, and what a departure reaps.
///
/// What this gate now measures, all DERIVED, never a literal:
/// 1. THE DERIVED LOGIN SET (G-NO-CASCADE's process half): at a settled login the running
///    realms are EXACTLY the home chain + the direct children of home whose authored placement
///    sits inside their own AoI spin-up radius as seen from the spawn standoff — computed
///    per child WITH THE NUMBERS, with a flap margin asserted so the set cannot drift mid-test.
/// 2. THE VACATE CONSEQUENCE: parked past the shell (+Z, radially outward — a −Z exit from the
///    +Z spawn would fly through the Star realm at the centre), the byte holds nothing: the
///    vacated interior reaps by EXACTLY its own count (the in-AoI children, then home itself),
///    the running set collapses to EXACTLY the ancestor chain, and a held park spins up none.
#[test]
fn g_look_wake_the_derived_login_set_holds_and_a_vacated_system_reaps() {
    let _tier = vd_bins::cluster_tier();
    let f = fixture("lookwake");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let p = DEV;
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];

    let config = vd_physics::worldgen::UniverseConfig::world(p.move_speed, p.tick_dt);
    let world = vd_physics::worldgen::WorldView::generated(p.universe_seed, &config);
    let home = vd_core::worldgen::default_home_realm(world.regions()).expect("THE world's home");
    // The home system's own row off the galaxy scope — the same row the galaxy shard boots from.
    // The SHELL it carries is what part (2) parks beyond; the interior band it used to carry is
    // gone with the mechanism (see the note above).
    let galaxy_scope = world.neighbourhood(&std::collections::BTreeSet::from([
        vd_core::worldgen::GALAXY,
    ]));
    let home_row = galaxy_scope
        .iter()
        .find(|r| r.realm == home)
        .expect("the galaxy scope rosters the home system");
    let shell = home_row.shape.circumscribed_extent();

    // (2) THE DERIVED LOGIN SET: the home chain + the direct children of home inside their own
    // AoI spin-up as seen from the spawn standoff, each stated with its numbers.
    let spawn = world.default_home_offset_m();
    let system_scope = world.neighbourhood(&std::collections::BTreeSet::from([home]));
    let movers: std::collections::BTreeMap<_, _> =
        vd_physics::worldgen::moving_children_for_config(p.universe_seed, &config, home)
            .into_iter()
            .collect();
    let tick_hz = 1.0 / p.tick_dt;
    // Placement of a direct child at tick t, in the home frame: a mover's orbit solve, a static
    // child's stored centre (the star sits at the centre; the planted fixtures hold their seeded
    // offsets).
    let placement_at = |row: &vd_core::geometry::RealmRegion, tick: u64| -> DVec3 {
        movers.get(&row.realm).map_or_else(
            || row.center.in_parents_frame().offset(),
            |elements| {
                vd_physics::celestial::orbital_state(
                    elements,
                    vd_core::kinematics::secs_since_epoch(tick, tick_hz),
                )
                .position
            },
        )
    };
    // The set may not FLAP over the test: every child must clear its spin-up boundary by more
    // than the fastest mover can close inside the window below. The SPEED is DERIVED from the
    // home roster's own elements this gate already holds — `v_peri`, the shipped periapsis-speed
    // accessor (a Kepler orbit's worst instant by the vis-viva law, and the very number the AoI
    // band widens itself by) — maxed over the movers. A transcribed metres-per-second was a
    // measurement of a DIFFERENT star: every star re-draws when the mass law moves, and the
    // orbital speeds go with it. Read from the elements, it cannot go stale again.
    // The WINDOW is the test's own wall-clock envelope, not a world number.
    const FLAP_WINDOW_S: f64 = 600.0;
    let worst_orbital_mps = movers
        .values()
        .map(vd_physics::celestial::OrbitalElements::v_peri)
        .fold(0.0_f64, f64::max);
    assert!(
        worst_orbital_mps > 0.0,
        "the home roster's movers must state a real orbital speed — a zero worst speed would \
         leave the flap margin below vacuous: {movers:?}",
    );
    let flap_margin_m = worst_orbital_mps * FLAP_WINDOW_S;
    let children: Vec<_> = system_scope
        .iter()
        .filter(|r| r.parent == Some(home))
        .collect();
    // THE CHILD COUNT IS THE WORLD'S OWN: its derived planet count (`PlanetConfig::n_planets`,
    // which `UniverseConfig::world` fills from `derived_planet_count` — a scale-free derivation,
    // not a knob) plus ONE for the Star realm the system also parents. The old bare 10 was a
    // transcription of exactly this sum and could not follow it. Moons are children of their
    // planets, never of the system, so they are correctly outside this count; `>=` leaves room
    // for anything planted beside the generated bodies.
    let expected_children =
        usize::try_from(config.planet.n_planets).expect("the planet count fits usize") + 1;
    assert!(
        children.len() >= expected_children,
        "THE home system parents its star + its {} derived planets ({expected_children} rows): \
         {}",
        config.planet.n_planets,
        children.len()
    );
    let mut in_aoi = 0u64;
    for row in &children {
        let dist = (placement_at(row, 0) - spawn).length();
        let child_spin = row.aoi.spin_up_r_m();
        let inside = dist < child_spin;
        eprintln!(
            "[look-wake] child {:?}: dist {dist:.6e} m vs its AoI spin-up {child_spin:.6e} m \
             => demanded: {inside}",
            row.realm,
        );
        assert!(
            (dist - child_spin).abs() > flap_margin_m,
            "{:?} sits within the flap margin of its own spin-up boundary \
             ({dist} vs {child_spin}) — the derived set could drift mid-test",
            row.realm,
        );
        in_aoi += u64::from(inside);
    }
    assert!(
        in_aoi >= 1,
        "at least the star must be inside the spawn standoff's AoI (the standoff is 2x its bound)"
    );
    let chain_len = {
        let mut n = 1u64;
        let mut cur = home;
        while let Some(parent) = world
            .regions()
            .iter()
            .find(|r| r.realm == cur)
            .and_then(|r| r.parent)
        {
            n += 1;
            cur = parent;
        }
        n
    };
    let expected_login_running = chain_len + in_aoi;
    // HOME'S OWN FATE AT THE PARK is derived from ITS OWN AoI row, exactly like its children's
    // (measured, first run: the park at 2× the solved shell sits WELL INSIDE the system's own
    // look-derived AoI — a whole star system is visible from across the galaxy — so home
    // lawfully KEEPS RUNNING while its vacated interior reaps. BOTH numbers are printed with the
    // derived sets just below, so neither is transcribed here; the old "home reaps too" guess
    // undercounted the running set by one and overcounted reaps by one).
    let park_dist = 2.0 * shell;
    let park_keeps_home = park_dist < home_row.aoi.tear_down_r_m();
    let expected_park_running = chain_len - u64::from(!park_keeps_home);
    let expected_new_reaps = in_aoi + u64::from(!park_keeps_home);
    eprintln!(
        "[look-wake] derived sets: login running {expected_login_running} (chain {chain_len} + \
         {in_aoi} in-AoI children); park at {park_dist:.4e} m vs home tear-down {:.4e} m => \
         home keeps running: {park_keeps_home}; park running {expected_park_running}, new reaps \
         {expected_new_reaps}",
        home_row.aoi.tear_down_r_m(),
    );

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &p, &client_book, client_quic, devctl_port);
    let landed = await_active(devctl_port, gw_admin, Duration::from_secs(150));
    assert_eq!(
        landed.location.as_deref(),
        Some(home_label(&p).as_str()),
        "{landed:?}"
    );
    let boot_deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let r = orch_rlm(a.admin);
        if r.running_gauge == expected_login_running {
            break;
        }
        assert!(
            Instant::now() < boot_deadline,
            "the login never reached the derived running set ({expected_login_running}): {r:?}"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    let settled_reaps = settle_reaps(a.admin, Duration::from_secs(10));
    // G-NO-CASCADE's settle half: a 15 s hold at the spawn keeps EXACTLY the derived set.
    let hold_until = Instant::now() + Duration::from_secs(15);
    while Instant::now() < hold_until {
        std::thread::sleep(Duration::from_millis(1000));
        let r = orch_rlm(a.admin);
        assert_eq!(
            r.running_gauge, expected_login_running,
            "G-NO-CASCADE: the settled login set is exactly the derived set: {r:?}"
        );
        assert_eq!(
            r.teardowns_reaped, settled_reaps,
            "nothing reaps at a held login: {r:?}"
        );
    }

    // (3) VACATE: +Z past the shell. The byte holds nothing — the interior reaps by its own
    // count and the running set collapses to the chain minus home.
    let cap_home =
        vd_core::flight::realm_speed_cap_mps(shell, p.move_speed, vd_core::flight::TRAVERSE_S);
    cross_leg(
        devctl_port,
        "look-wake exit home->galaxy (+Z, 2x the solved shell)",
        move |_tick| DVec3::new(0.0, 0.0, 2.0 * shell),
        &galaxy_label(&p),
        vd_bins::flight::governed_leg_budget(&p, shell, cap_home),
    );
    let started = Instant::now();
    let reap_deadline = Duration::from_secs(120);
    loop {
        let r = orch_rlm(a.admin);
        if r.teardowns_reaped >= settled_reaps + expected_new_reaps {
            break;
        }
        assert!(
            started.elapsed() < reap_deadline,
            "the vacated interior never fully reaped (reaps {} vs settled {settled_reaps} + \
             expected {expected_new_reaps}): {r:?}",
            r.teardowns_reaped,
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    let after_reaps = settle_reaps(a.admin, Duration::from_secs(30));
    assert_eq!(
        after_reaps,
        settled_reaps + expected_new_reaps,
        "EXACTLY the vacated interior reaped ({in_aoi} in-AoI children{}), nothing else",
        if park_keeps_home { "" } else { " + home" },
    );
    let running_after = settle_gauge(a.admin, Duration::from_secs(15), |r| r.running_gauge);
    assert_eq!(
        running_after, expected_park_running,
        "the running set collapsed to exactly the derived park set"
    );
    let spins_at_park = orch_rlm(a.admin).spins_requested;
    let park_hold = Instant::now() + Duration::from_secs(15);
    while Instant::now() < park_hold {
        std::thread::sleep(Duration::from_millis(1000));
        let r = orch_rlm(a.admin);
        assert_eq!(
            r.spins_requested, spins_at_park,
            "a held exterior park spins up none: {r:?}"
        );
    }
    eprintln!(
        "[look-wake] vacated: running {running_after} (== derived {expected_park_running}), \
         reaps {after_reaps} (= settled {settled_reaps} + {expected_new_reaps}), spins steady \
         at {spins_at_park}"
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

/// Drive ONE walk leg to the 3D `target` in CHUNKS, settle the sticky Move, read the delivered own
/// pose, and assert the occupant ARRIVED — movement never froze across the demand spin-ups. A
/// generous `max_ticks` so a slow spawn never times out mid-leg (the point is that the dot keeps
/// MOVING while realms stream in around it). CHUNKED because a `WalkTo` blocks the devctl socket
/// for its whole tick budget and a GOVERNED star-gap leg (S3) runs minutes — each chunk's blocking
/// read stays a quarter of the socket's own timeout, derived, never a literal.
fn walk_leg(
    devctl_port: u16,
    leg: &str,
    target: DVec3,
    max_ticks: u64,
    max_step_m: f64,
    arrive_within_m: f64,
) -> DVec3 {
    // `arrive_within_m` is DERIVED PER LEG (a shell fraction of whatever the park protects):
    // under the geometric throttle taper the last stretch of a governed brake closes at FOOT
    // speed, so a fixed 3 m arrival at true scale is an unbounded crawl (walk-gate measured).
    let chunk_ticks =
        ((vd_bins::DEVCTL_READ_TIMEOUT.as_secs_f64() / 4.0) / DEV.tick_dt).floor() as u64;
    let mut spent = 0u64;
    let landed = loop {
        let chunk = chunk_ticks.min(max_ticks - spent);
        let _ = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: target.to_array(),
                arrive_epsilon: arrive_within_m,
                max_ticks: chunk,
                max_step_m,
            },
        )
        .unwrap_or_else(|| panic!("leg {leg}: no walk response"));
        spent += chunk;
        let here = own_pos(&poll_state(devctl_port).expect("state mid-leg")).expect("own pos");
        if (here - target).length() <= arrive_within_m || spent >= max_ticks {
            break here;
        }
    };
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    assert!(
        (landed - target).length() <= arrive_within_m,
        "leg {leg} FROZE: the occupant stalled at {landed:?}, never reaching {target:?} — a demand spin-up \
         stalled the walk, or the walk itself froze.",
    );
    landed
}

// UN-PARKED at the speed-law slice (S3, real-scale addendum §A3 + the OQ-2 ruling): the inter-system
// leg is now flown at GOVERNED speeds — the galaxy's own ceiling (2·R_gal/T_TRAVERSE ≈ 2.5e13 m/s;
// the galaxy radius did not move under the mass re-solve) with the approach governor decelerating
// onto the sibling — so the star gap (the ring radius the galaxy itself authors, printed with the
// approach derivation) is minutes of wall clock, exactly as the owner's warp mechanism states.
// Every leg budget below rides the ONE shared law, `vd_bins::flight::governed_leg_budget` over the
// governed closed form; every in-system assertion the park owed is back verbatim.
#[test]
fn a_flying_occupant_streams_a_neighbour_system_in_ahead_then_the_vacated_realm_is_reaped() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // THE demand-FLY, rewritten against the multi-star ring (the D-RLM-16 rewrite): since the
    // wider-visibility world (32fcc5c) every planet of the HOME system is already visible at login,
    // so the genuinely culled realm is the NEIGHBOUR STAR one ring radius away — asleep, because the
    // ring is wider than its wake radius. The occupant exits the home system, flies toward the
    // neighbour, and parks INSIDE its wake band while still outside its own solved shell:
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
    // Spawn + exit + the drain/quiesce teardown windows ⇒ a materially longer budget than a
    // login. The FLIGHT legs carry their own governed budgets (derived below); this bounds the
    // non-flight waits (spawn-ahead, teardown) exactly as before.
    let deadline = Duration::from_secs(120);

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &p, &client_book, client_quic, devctl_port);

    // The login lands the occupant at the home star; everything already in view spins up and settles
    // — the baseline the approach must climb above for a NON-VACUOUS spin-up-ahead.
    let flight_started = Instant::now();
    let landed = await_active(devctl_port, gw_admin, deadline);
    assert_eq!(
        landed.location.as_deref(),
        Some(home_label(&p).as_str()),
        "the login lands at the star home realm: {landed:?}",
    );
    let settled_spins = settle_spins(a.admin, Duration::from_secs(30));

    // The aim, through the production boot: the neighbour star's authored ring placement, off THE
    // world's roster (the one derivation point — `world_roster` reads it from the galaxy-hosted
    // boot, the parent that authors it). THE world's own solved numbers are read FIRST, because
    // the precondition below is only meaningful stated against them.
    let roster = world_roster(&p);
    let (neighbour, centre) = (roster.sibling, roster.sibling_centre);
    let ring_r = centre.length();
    let cfg = vd_physics::worldgen::UniverseConfig::world(p.move_speed, p.tick_dt);
    let the_world = vd_physics::worldgen::WorldView::generated(p.universe_seed, &cfg);
    let home_reach = the_world
        .regions()
        .iter()
        .find(|r| r.realm == roster.home)
        .expect("THE world rosters the home system")
        .shape
        .finite_extent()
        + cfg.band.outset_m;
    // THE PRECONDITION that ties this flight to the multi-star geometry, RE-DERIVED. It used to
    // read `ring_r > 1_000.0` — a metre count from a world whose ring was kilometres; against a
    // ring the galaxy now authors orders of magnitude further out it asserted nothing at all. The
    // meaningful statement is the one the flight depends on: the neighbour sits beyond the home
    // system's WHOLE CONTAINMENT REACH (its solved shell plus the release outset), so the leg is a
    // real inter-system crossing and the neighbour cannot already be awake in the settled login
    // baseline. Both sides are solved from THE world, so this cannot go vacuous again.
    assert!(
        ring_r > home_reach,
        "the ring neighbour {neighbour} must sit beyond the home system's whole containment \
         reach ({ring_r:.4e} m vs {home_reach:.4e} m) — otherwise there is no second system to \
         fly to and the spin-up-ahead verdict is vacuous",
    );

    // ---- LEG 0: exit the home system (re-home UP into the galaxy) — the ±Z POLAR corridor, and
    // the GALAXY's own label asserted on arrival. The old +X exit flew through the orbital plane
    // and broke on ANY label != the home system's, so an in-plane planet capture passed it —
    // a FALSE GREEN (D-WORLD-9); the polar leg is what I-AXIS licenses, and the label is the proof.
    //
    // THE EXIT HEIGHT IS DERIVED FOR THE 3-D SIBLING (the S3 re-derivation): the onward leg flies a
    // STRAIGHT line from the polar exit toward the sibling's seeded direction — read off
    // `roster.sibling_centre`, the placement the galaxy itself authors — and that line's closest
    // approach to the home system is `exit_z·sin(θ)` (θ = the sibling's angle off the pole). A
    // FIXED exit height cannot clear that at every seed: the interim world measured a 220 m exit
    // leaving a 146 m closest approach against a 150 m shell — INSIDE it, so the warp leg would
    // clip back into home mid-flight and the in-flight walk target would straddle the re-home (the
    // exact ping-pong `creep_into` documents). The exit height clears the home's whole containment
    // reach with a 2× margin, by construction, at whatever lean the sibling is drawn with.
    let sib_unit = centre / ring_r;
    let sin_off_pole = (1.0 - sib_unit.z * sib_unit.z).max(0.0).sqrt();
    // The FLOOR is that same 2× reach, never a metre count (the walk gate's identical line): two
    // release-reaches straight down the pole clear home whatever the sibling's off-pole angle,
    // and it re-solves with the world instead of pinning the interim world's 220 m.
    let exit_z = (2.0 * home_reach / sin_off_pole).max(2.0 * home_reach);
    // +Z, NOT −Z: the spawn stands on the +Z axis at twice the STAR's bound (T2), so a −Z exit
    // flies straight through the Star realm at the system centre (measured: the exit leg
    // committed INTO "Star …" and spent its whole budget crawling that interior at the star's
    // own ceiling). The clearance law is pole-symmetric, and the +z-leaning sibling makes +Z
    // the natural side.
    let want_galaxy = galaxy_label(&p);
    let cap_home = realm_speed_cap_mps(home_reach - cfg.band.outset_m, p.move_speed, TRAVERSE_S);
    cross_leg(
        devctl_port,
        "fly-exit home->galaxy (polar, derived height)",
        |_tick| DVec3::new(0.0, 0.0, exit_z),
        &want_galaxy,
        // Derived: the label flips at the home release edge under the home ceiling (the walk
        // gate measured ~2.5× the closed form — `governed_leg_budget` is the shared law).
        vd_bins::flight::governed_leg_budget(&p, home_reach, cap_home),
    );
    eprintln!(
        "[fly] derived polar exit z = {exit_z:.1} m (sibling off-pole sin {sin_off_pole:.3}; the \
         onward line's closest approach to home = {:.1} m, clear of its {home_reach:.0} m shell)",
        exit_z * sin_off_pole,
    );
    let galaxy_loc = want_galaxy;

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

    // ---- LEG 1: the governed approach — the observations RIDE THE FLIGHT (true-scale restatement).
    // The interim gate parked INSIDE the sibling's wake band while OUTSIDE its shell; on the
    // true-size world the INTERIOR band (the look-derived interest band) sits INSIDE the solved
    // shell — the same lawful inversion the look-wake gate pins and prints — so no exterior park
    // can hold the neighbour awake through it: a parked observer's demand lawfully DECAYS (that
    // IS the real sky: a whole system a shell-width away subtends below the visibility angle,
    // and the approach derivation below prints the two radii it is judged on). What wakes the
    // neighbour is the VELOCITY LEAD: the demand verdict samples at pos + v·τ_wake, and at
    // governed closing speeds the lead covers the remaining gap (the just-in-time wake that
    // `guard_wake_covers_visibility` fences at every boot). So the three observations —
    // spin-up AHEAD, the relayed interior streaming in, the streamed boxes MOVING — are made
    // DURING the approach, all while `location` stays the galaxy on every poll (the world
    // arrives ahead of any crossing, exactly as before — measured on the wing).
    let sib_row = *the_world
        .regions()
        .iter()
        .find(|r| r.realm == neighbour)
        .expect("THE world rosters the ring sibling");
    let (shell, spin_up) = (sib_row.shape.finite_extent(), sib_row.aoi.spin_up_r_m());
    let release_reach = shell + cfg.band.outset_m;
    // The wake band still reaches FAR past containment at true scale — a system's AoI is
    // look-derived (a star is visible from across the galaxy) while its solved shell is orders
    // smaller, which the assert below MEASURES and the derivation print states in metres — so
    // the 2×shell polar standoff below parks DEEP inside the wake band and lawfully HOLDS the
    // woken neighbour (the same physics look-wake measures on the home row). Distinct from the
    // INTERIOR interest-byte band, which does invert inside the shell.
    assert!(
        release_reach < spin_up,
        "the sibling's wake band reaches past its containment ({release_reach} vs {spin_up})",
    );
    // The approach waypoint: the sibling's polar standoff at 2× its solved shell — outside the
    // release reach (no crossing can fire), down the pole (the I-POLE corridor), and INSIDE
    // the wake band, so a ship parked there keeps the neighbour awake while observing.
    let waypoint = centre + DVec3::new(0.0, 0.0, -2.0 * shell);
    let tuning = FlightTuning::derive(
        p.move_speed,
        p.tick_dt,
        (u64::from(p.tick_hz) / 2).max(1),
        u32::try_from(p.boot_ticks_p99).expect("boot p99 fits"),
    );
    let cap_galaxy = realm_speed_cap_mps(cfg.scale.galaxy_r_m, p.move_speed, TRAVERSE_S);
    let v_park = approach_ceiling_mps(
        realm_speed_cap_mps(shell, p.move_speed, TRAVERSE_S),
        waypoint.length() - shell,
        tuning.tau_s,
    );
    // The gap this leg actually flies: from the derived polar exit height across to the
    // sibling's standoff — stated once, and used by BOTH the printed closed form and the budget.
    let leg_dist_m = (waypoint - DVec3::new(0.0, 0.0, exit_z)).length();
    let leg_s = leg_time_s(leg_dist_m, cap_galaxy, p.move_speed, v_park, tuning.tau_s)
        .expect("the star-gap leg holds a cruise");
    // THE BUDGET IS THE SHARED LAW, never a local multiple of the closed form: this was the one
    // leg in the file still doubling `leg_s` by hand, and `governed_leg_budget`'s own doc records
    // that a 2× budget measured ~25 s SHORT of a real leg's commit. Routed through the shared
    // function (as the exit and descend legs above already are), a re-measurement of the flight
    // law moves this leg with every other one. `leg_s` stays as the printed yardstick.
    let leg_ticks = (vd_bins::flight::governed_leg_budget(&p, leg_dist_m, cap_galaxy).as_secs_f64()
        / p.tick_dt) as u64;
    let brake_m = vd_bins::flight::governed_brake_m(v_park, p.tick_dt, tuning.tau_s);
    eprintln!(
        "[fly] GOVERNED APPROACH DERIVATION: ring gap {ring_r:.4e} m, flown leg {leg_dist_m:.4e} \
         m, waypoint 2×{shell:.4e} m below the sibling pole (release {release_reach:.4e}, wake \
         {spin_up:.4e} — INSIDE the shell), galaxy ceiling {cap_galaxy:.4e} m/s, waypoint \
         ceiling {v_park:.4e} m/s, closed-form leg {leg_s:.1} s => budget {leg_ticks} ticks, \
         brake {brake_m:.1} m",
    );
    // DESCEND TO THE DERIVED EXIT HEIGHT FIRST: the exit cross_leg ends at the release edge
    // (~1 home shell down the pole), and the S3 clearance (closest approach = exit_z·sinθ ≥
    // 2×reach) holds only from the full exit height — from the edge, the line to the
    // +z-leaning sibling re-enters the home shell (the walk gate measured the resulting
    // System↔galaxy crossing ping-pong). Frame-stable aim: home sits AT the galaxy origin.
    // Budgeted at the HOME ceiling, conservatively: a receding leg rides the departed body's
    // arm (~cap_home at the start), and a galaxy-cap budget measured one chunk short on the
    // walk gate's identical descend.
    let descend_budget = vd_bins::flight::governed_leg_budget(&p, exit_z - home_reach, cap_home);
    walk_leg(
        devctl_port,
        "fly descend to the exit height",
        DVec3::new(0.0, 0.0, exit_z),
        (descend_budget.as_secs_f64() / p.tick_dt) as u64,
        // Taper = the arrival slop (the walk gate's measured ramp-collapse trap).
        0.5 * home_reach,
        // Half a home reach of slop keeps the onward clearance past the shell.
        0.5 * home_reach,
    );
    // THE CHASE WITH OBSERVATIONS: chunked WalkTo with the throttle HELD across chunks (a cut
    // would zero the carried velocity and restart the speed-law ramp from the foot), polling
    // between chunks for the three verdicts.
    let chunk_ticks =
        ((vd_bins::DEVCTL_READ_TIMEOUT.as_secs_f64() / 4.0) / p.tick_dt).floor() as u64;
    let mut spent = 0u64;
    let mut spin_ahead_spent: Option<u64> = None;
    let mut fresh: std::collections::BTreeMap<String, [f64; 3]> = std::collections::BTreeMap::new();
    let mut best_motion = 0.0_f64;
    loop {
        let chunk = chunk_ticks.min(leg_ticks - spent);
        let _ = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: waypoint.to_array(),
                arrive_epsilon: 0.25 * shell,
                max_ticks: chunk,
                // The taper IS the arrival slop (the walk gate's measured ramp-collapse trap:
                // a wider taper wedges the flight at walking pace); `brake_m` stays derived +
                // printed as the overshoot yardstick.
                max_step_m: 0.25 * shell,
            },
        )
        .unwrap_or_else(|| panic!("fly approach: no walk response"));
        spent += chunk;
        let st = poll_state(devctl_port).expect("state mid-approach");
        assert_eq!(
            st.location.as_deref(),
            Some(galaxy_loc.as_str()),
            "the world must arrive AHEAD of any crossing — the approach stays in the \
             between-space on every poll: {st:?}",
        );
        if spin_ahead_spent.is_none() && orch_rlm(a.admin).spins_requested > settled_spins {
            spin_ahead_spent = Some(spent);
        }
        for (name, c1) in planet_boxes(&st) {
            if baseline_planets.contains(&name) {
                continue;
            }
            match fresh.get(&name) {
                None => {
                    let _ = fresh.insert(name, c1);
                }
                Some(c0) => {
                    let d = ((c1[0] - c0[0]).powi(2)
                        + (c1[1] - c0[1]).powi(2)
                        + (c1[2] - c0[2]).powi(2))
                    .sqrt();
                    best_motion = best_motion.max(d);
                }
            }
        }
        let here = own_pos(&st).expect("own pos mid-approach");
        if (here - waypoint).length() <= 0.25 * shell || spent >= leg_ticks {
            break;
        }
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    // 1. Spin-up-AHEAD: the velocity lead demanded the sleeping neighbour strictly while the
    // occupant was still flying the between-space (every poll above asserted the galaxy label).
    let spin_at = spin_ahead_spent.unwrap_or_else(|| {
        panic!(
            "the neighbour star never spun up ahead during the governed approach (spins stayed \
             {settled_spins}): {:?}",
            orch_rlm(a.admin),
        )
    });
    eprintln!("[fly] spin-up-AHEAD at ~{spin_at} flight ticks, before any crossing");
    // 2. THE OWNER'S GAP, as a gate: planet boxes the baseline never held streamed into the
    // drawn scene mid-approach — the neighbour's relayed interior, ahead of arrival. A short
    // parked tail is allowed for a late-flight arrival (the drain grace outlives the park).
    let discover_deadline = Instant::now() + Duration::from_secs(45);
    while fresh.is_empty() {
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
        if let Some(st) = poll_state(devctl_port) {
            for (name, c) in planet_boxes(&st) {
                if !baseline_planets.contains(&name) {
                    let _ = fresh.insert(name, c);
                }
            }
        }
    }
    eprintln!(
        "[fly] the neighbour's interior streamed in ahead: {:?}",
        fresh.keys().collect::<Vec<_>>()
    );
    // 3. ...and the streamed boxes MOVE — outlines alone would be the stale-box defect reborn.
    let observe_until = Instant::now() + Duration::from_secs(25);
    while Instant::now() < observe_until && best_motion <= 1.0 {
        std::thread::sleep(Duration::from_millis(300));
        if let Some(st) = poll_state(devctl_port) {
            for (name, c1) in planet_boxes(&st) {
                if let Some(c0) = fresh.get(&name) {
                    let d = ((c1[0] - c0[0]).powi(2)
                        + (c1[1] - c0[1]).powi(2)
                        + (c1[2] - c0[2]).powi(2))
                    .sqrt();
                    best_motion = best_motion.max(d);
                }
            }
        }
    }
    assert!(
        best_motion > 1.0,
        "the neighbour's streamed boxes never moved (best {best_motion:.3} m) — outlines \
         without rows",
    );

    // ---- G-RELAY-STAMP (look_horizon.md slice 0 — DISCHARGES D-WINDOW-6(2)). ----
    // THIS is the non-leaf topology the ledger row demanded: the relayed child (the neighbour
    // star system) HAS children, and its planets just streamed in as composed relay rows. Over
    // a ≥60 s process flight: the relay fold refuses NO stamp (the ring + resolve-at-T with
    // at-or-before fallback replaced the exact-tick-equality demand that refused 50–87 times
    // per two-ships run), it COMPOSES real rows, and the fallback's declared skew stays inside
    // one keep-alive beat. The launcher leaves VD_SESSION_RECHECK unset, so the gateway's beat
    // is the derived tick_hz/2.
    // One keep-alive beat, PLUS the fold-boundary fencepost: stamps land every `beat` ticks,
    // and a fold on the very tick BEFORE the next keep-alive lawfully reads a stamp `beat + 1`
    // ticks old (measured at true scale: skew 26 at beat 25 — the +1 is the fencepost, not a
    // lag).
    let beat_ticks = u64::from(p.tick_hz) / 2 + 1;
    // THE MISS BOUND IS DERIVED FROM WINDOW CHURN, not pinned at an accidental zero (measured
    // 2026-08-17, battery run 5): at the exit crossing the gateway re-opens its galaxy windows
    // and the re-served relay (reliable lane) can outrun the level datagrams (lossy BY DESIGN)
    // by one tick — ONE fold then sees a member child whose FRESH ring holds only a future
    // stamp, exactly §3.6's "bounded, counted" transient. STRUCTURALLY that is the ONLY
    // reachable miss class: once a ring has served any stamp, the C6 head-anchored prune never
    // drops a stamp its own head still retains, so an established ring can never go
    // future-only again — a first-service transient is bounded by the windows opened, while
    // the D-WINDOW-6(2) storm this gate discharges was 50-87 refusals per run with ZERO opens
    // in the window. The skew gauge (≤ one beat) keeps measuring every real lag.
    while flight_started.elapsed() < Duration::from_secs(60) {
        let gw = gateway_view(gw_admin).expect("the gateway serves its admin snapshot");
        assert!(
            gw.window_relay_stamp_missing <= gw.window_open_sent,
            "G-RELAY-STAMP: more unresolvable-stamp folds than windows ever opened — that is \
             the exact-tick refusal storm, not a first-service transient: {gw:?}",
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    let gw = gateway_view(gw_admin).expect("the gateway serves its admin snapshot");
    assert!(
        gw.window_relay_stamp_missing <= gw.window_open_sent,
        "G-RELAY-STAMP: stamp misses exceeded the derived first-service bound over the whole \
         flight: {gw:?}",
    );
    assert!(
        gw.window_relay_rows_composed > 0,
        "G-RELAY-STAMP is vacuous: no relayed interior rows ever composed: {gw:?}",
    );
    assert!(
        gw.window_relay_stamp_skew_ticks <= beat_ticks,
        "G-RELAY-STAMP: the at-or-before fallback's declared skew ({} ticks) exceeded one \
         keep-alive beat ({beat_ticks} ticks): {gw:?}",
        gw.window_relay_stamp_skew_ticks,
    );
    // ---- THE SLICE-3 INTERIOR-FORWARD COUNTERS on a LAWFUL flight (look_horizon.md §6 slice 3:
    // two different counters, on purpose). The VIOLATION counter is 0 — no lawful producer ever
    // names an unrostered grandchild. The LAWFUL-FILTER counter is NON-ZERO — the home system's
    // live planets relayed their own batches up while the player stood inside it, each carrying
    // a Level the gateway lawfully filters (a depth-3 subject no row can exist for), which is
    // the measured proof the sealed interior forward actually flowed end-to-end in-process.
    assert_eq!(
        gw.window_relay_interior_unvouched, 0,
        "slice 3: an interior forward named an unrostered grandchild on a lawful flight: {gw:?}",
    );
    assert!(
        gw.window_relay_interior_filtered > 0,
        "slice 3 is vacuous: no interior batch was ever admitted+filtered — the sealed interior \
         forward never flowed: {gw:?}",
    );
    eprintln!(
        "[fly] G-RELAY-STAMP over {:.1} s: relay_rows_composed={} stamp_missing={} (≤ opened \
         windows {}) descent_refused={} unrostered={} skew_max={} ticks (≤ beat {beat_ticks}) \
         depth_max={}; interior forward (slice 3): unvouched={} filtered={}",
        flight_started.elapsed().as_secs_f64(),
        gw.window_relay_rows_composed,
        gw.window_relay_stamp_missing,
        gw.window_open_sent,
        gw.window_relay_descent_refused,
        gw.window_relay_unrostered,
        gw.window_relay_stamp_skew_ticks,
        gw.window_relay_depth_max,
        gw.window_relay_interior_unvouched,
        gw.window_relay_interior_filtered,
    );

    // ---- LEG 2: fly back out of the band → the vacated neighbour reaps BEHIND. ----
    // The home system's own vacated planets reap during the outbound flight; settle that first so
    // the return's climb can only be the neighbour. Anti-thrash rests entirely on the ~1 s grace
    // (equal AoI factors ⇒ no geometric dead-zone), so the reap LAGS the geometric exit by the grace
    // tail + drain + quiesce — a polled assertion, bounded by the deadline.
    let settled_reaps = settle_reaps(a.admin, Duration::from_secs(30));
    // The return flies the same governed gap back; the home-side park is the DERIVED polar
    // standoff at 2× the home's own solved shell (the interim 220 m park is deep inside the
    // true-size shell) — outside home's release reach, far outside the sibling's: both systems
    // asleep at the park, so the vacated neighbour has nothing holding it.
    let home_shell = home_reach - cfg.band.outset_m;
    let home_park = DVec3::new(0.0, 0.0, 2.0 * home_shell); // +Z: same side as the exit corridor
    walk_leg(
        devctl_port,
        "back-toward-home",
        home_park,
        leg_ticks,
        // Taper = the arrival slop (the walk gate's measured ramp-collapse trap).
        0.5 * home_shell,
        // Half a home shell of slop parks 1.5–2.5 shells out — still outside home; the reap
        // verdicts don't care where in that band the ship stands.
        0.5 * home_shell,
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
