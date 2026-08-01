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
        orchestrator: reserve_udp_addr(),
        gateway: reserve_udp_addr(),
        shard: reserve_udp_addr(),
        admin: reserve_tcp_addr(),
        gateway_admin: Some(gateway_admin),
        orchestrator_probe: reserve_tcp_addr(),
        gateway_probe: reserve_tcp_addr(),
        shard_probe: reserve_tcp_addr(),
        shard_b: reserve_udp_addr(),
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

/// The INNER planet's `(realm, elements)` — the mover whose EPOCH position is CLOSEST to the star (well
/// inside System 7's 150 m SOI, so approaching it never crosses the System boundary).
fn inner_planet_mover(
    p: &DevClusterParams,
) -> (vd_core::pose::RealmId, vd_core::celestial::OrbitalElements) {
    let config = vd_core::worldgen::UniverseConfig::visual_demand(p.move_speed, p.tick_dt);
    vd_core::worldgen::moving_children_for_config(
        p.universe_seed,
        &config,
        vd_core::pose::RealmId::System(7),
    )
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
    // SAFETY: the process-wide env is set before any cluster boot; `--test-threads=1` serializes tests. It is
    // RESET the moment this test's crossing completes (below) so a later orbit-speed-dependent test
    // (`a_parked_ship…`) does not inherit the quasi-freeze — this shard reads the value only at boot, so
    // resetting after its cluster is up does not affect this test.
    unsafe {
        std::env::set_var("VD_VISUAL_ORBIT_SLOWDOWN", "300");
    }
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
    // The epoch position is slowdown-invariant; with slowdown=300 the planet barely drifts over the approach.
    let epoch = vd_core::celestial::orbital_state(&elements, 0.0).position;
    let want = vd_core::pose::FrameRef::PlanetCentered { planet_seed }.label();
    eprintln!(
        "[repro] inner planet {planet:?} epoch_len={:.1} — flying to its FIXED epoch center, expect {want:?}",
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
        if let Some(pos) = own_pos(&st) {
            best = best.min((pos - epoch).length());
        }
        if leg.is_multiple_of(5) {
            eprintln!(
                "[repro] inner leg {leg}: tick={:?} loc={loc:?} own_len={:?} best={best:.2}",
                st.universe_tick,
                own_pos(&st).map(|v| v.length()),
            );
        }
        leg += 1;
        let _ = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: epoch.to_array(),
                arrive_epsilon: 1.0,
                max_ticks: 40,
            },
        );
        assert!(
            started.elapsed() < deadline,
            "NO CROSSING (inner): chased the inner planet {planet:?}'s fixed epoch center for {}s but \
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
    // Reset the orbit-slowdown NOW (the crossing is done; this test's shards already booted with it) so a later
    // full-speed-orbit test in this process never inherits the quasi-freeze.
    unsafe {
        std::env::remove_var("VD_VISUAL_ORBIT_SLOWDOWN");
    }

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
    unsafe {
        std::env::set_var("VD_VISUAL_ORBIT_SLOWDOWN", "300");
    }
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

    // ── FLY IN: chase the planet's FIXED epoch center until `location` flips to the planet. ──
    let in_deadline = Instant::now() + deadline;
    loop {
        if poll(in_deadline).location.as_deref() == Some(want_planet.as_str()) {
            break;
        }
        let _ = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: epoch.to_array(),
                arrive_epsilon: 1.0,
                max_ticks: 40,
            },
        );
        assert!(
            Instant::now() < in_deadline,
            "NO CROSSING IN: location never became {want_planet:?}"
        );
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    eprintln!("[repro] crossed IN to {want_planet:?}");

    // ── RIDE THE REALM (Symptom A): the own player's WORLD pose tracks the destination planet's streamed
    // box center (rides the orbit), NOT the star origin (no "teleport"). `own_pos` is now the world_pos-
    // mapped own pose; the planet box center is its streamed placement. Discriminates on the frozen
    // geometry: inner orbit ≈17 m, planet SOI ≈4.16 m — so a tolerance of 8 m separates "rides it" (gap
    // < SOI < 8, own.len ≈17 > 8) from the bug (raw frame-local: gap ≈17 > 8 AND own.len ≈0 < 8).
    let want_box = format!("Planet({planet_seed})");
    let ride = {
        let until = Instant::now() + Duration::from_secs(15);
        loop {
            let s = poll(until);
            if own_pos(&s).is_some() && s.realm_boxes.iter().any(|b| b.realm == want_box) {
                break s;
            }
            assert!(
                Instant::now() < until,
                "own world pose / planet box never appeared after cross-in"
            );
        }
    };
    let own = own_pos(&ride).expect("own world pose");
    let center = DVec3::from_array(
        ride.realm_boxes
            .iter()
            .find(|b| b.realm == want_box)
            .expect("planet box")
            .center,
    );
    const PLANET_RIDE_TOL_M: f64 = 8.0;
    assert!(
        center.length() > 12.0,
        "precondition: the planet box sits out at its orbit (was {:.2} m)",
        center.length()
    );
    assert!(
        (own - center).length() < PLANET_RIDE_TOL_M,
        "RIDE FAILED: own world pose {own:?} did NOT ride the planet box {center:?} (gap {:.2} m of a \
         {:.2} m orbit) — rendered at raw frame-local (teleport-to-origin), not composed with its realm.",
        (own - center).length(),
        center.length(),
    );
    assert!(
        own.length() > PLANET_RIDE_TOL_M,
        "RIDE FAILED: own world pose {own:?} sits at the star origin (len {:.2}) — did not ride the moving \
         planet.",
        own.length(),
    );
    eprintln!(
        "[repro] RIDES THE REALM: own {own:?} tracks the planet box {center:?} (gap {:.2} m) — no teleport",
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

    // ── OWED (D-RLM-14 / floating-origin S6 / VU-6): the SURROUNDING realm feed re-advancing on the return. ──
    // The return COMMIT above is the Symptom-B (primary) gate — reaching "System 7" is unreachable if the
    // return parked, so the total freeze is proven fixed. What is NOT yet guaranteed is that System re-ships
    // its per-tick `RealmSnapshot` feed to the returned client: the dest read sub re-opens only from the
    // `SubscriptionReady` at `on_saga_promote`, and the realm SCENE is re-streamed only at a home-entry
    // `Active` promote (never on a cross). So we OBSERVE the neighbour feed here and LOG its status — we do
    // NOT panic, because the re-establishment-on-cross is a ledgered next slice, not a regression. Flip this
    // to a hard assert (`>= base + 20`) when D-RLM-14 lands.
    let base = poll(Instant::now() + Duration::from_secs(5)).realm_frames_applied;
    let observe_until = Instant::now() + Duration::from_secs(15);
    let mut neighbour_feed_live = false;
    loop {
        std::thread::sleep(Duration::from_millis(300));
        let s = poll(observe_until);
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
    if neighbour_feed_live {
        eprintln!(
            "[repro] post-return neighbour feed LIVE on System 7 — the surrounding realms re-advanced on the \
             cross (D-RLM-14 already satisfied here; keep the observation)"
        );
    } else {
        eprintln!(
            "[repro] OWED (D-RLM-14 / floating-origin S6 / VU-6): the return COMMITTED and the player is live \
             on System 7, but the SURROUNDING realm feed did not re-advance within the window — the \
             return-crossing read-sub / realm-scene re-stream is the next slice. NOT a regression (the total \
             freeze is fixed); ledgered."
        );
    }
    unsafe {
        std::env::remove_var("VD_VISUAL_ORBIT_SLOWDOWN");
    }
}

// The bootstrap-TTL fail-safe leg (a demand login whose home never boots must be CLOSED at
// `bootstrap_ttl_ticks`, not hang) is DEFERRED — see DEFERRED.md D-RLM-9. An empirical spike here found the
// obvious inductions do NOT reach the TTL: with no orchestrator (or the gateway dropped from its peers) the
// gateway never CLOCK-SYNCS, so its seed injector stays inert and no demand is ever emitted — the login sits
// with an open session and `home_bootstrap_timeouts == 0` (a separate pre-sync-hold concern). The TTL fires
// only for a SYNCED gateway whose home genuinely fails to boot, which CA-1 learning (the orchestrator learns
// the gateway from the demand frame and delivers the grant anyway) makes delicate to induce.

// ---- The demand-FLY: a MOVING occupant streams a culled planet in AHEAD + reaps the vacated realm BEHIND ---

/// RLM realistic-demo Slice 3 — how far SHORT of the outer planet's center the ship stops on the fly-out. It
/// must clear the planet's ~4.16 m SOI (so the occupant stays OUTSIDE the planet's containment and `location`
/// never flips) yet fall well inside its ~59.5 m visibility band (so the planet's shard spins up AHEAD). 20 m
/// sits between: ≫ the 4.16 m SOI, ≪ the 59.5 m band even after the outer planet's ~3 m/s orbital drift over
/// the fly-out. The spin-up-ahead LEAD is a natural consequence of visibility-radius ≫ crossing-radius, NOT a
/// predictive horizon — `boot_ticks_p99` stays 0 (its `DEV` value), so a realm boots exactly when it becomes
/// VISIBLE, and the ~15 m/s ship then spends seconds flying the rest of the way toward containment.
const OUTER_APPROACH_SHORT_M: f64 = 20.0;

/// Poll the orchestrator's `spins_requested` until it holds STEADY for [`STABLE_HOLDS`] consecutive reads (the
/// stationary occupant has finished spinning up every realm ALREADY in its visibility — the star + the inner
/// planets) or `cap` elapses, then return the settled count. This is the baseline the fly-out must climb ABOVE
/// for a NON-VACUOUS spin-up-ahead: the outer planet is culled at login (its ~142 m orbit is beyond the
/// ~59.5 m band from the star), so it is NOT in this baseline — it can only spin up as the ship approaches.
const STABLE_HOLDS: u32 = 5;
fn settle_spins(admin_addr: SocketAddr, cap: Duration) -> u64 {
    let started = Instant::now();
    let mut last = orch_rlm(admin_addr).spins_requested;
    let mut holds = 0u32;
    while started.elapsed() < cap {
        std::thread::sleep(Duration::from_millis(300));
        let now = orch_rlm(admin_addr).spins_requested;
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

/// The outer (most distant) planet's EPOCH position under the demand cluster's `visual-demand` geometry — the
/// deterministic aim point for the fly-out. Computed from the SAME `(seed, config)` the demand-spawned shard
/// boots from (region centers are the tick-0 orbital epoch, mass-independent), so the test and the server
/// agree on where the planet is. Since `occupant_v_max = move_speed · time_multiplier` (multiplier 1.0 by
/// default) and no orbit-slowdown is set, this matches the shard's forest.
fn outer_planet_pos(p: &DevClusterParams) -> DVec3 {
    let config = vd_core::worldgen::UniverseConfig::visual_demand(p.move_speed, p.tick_dt);
    // A mover's region `center` is ZERO — its position is AUTHORED by its frame/ephemeris each tick, not baked
    // into the center (the moving-frame containment fix). So derive each planet's EPOCH position from its
    // orbital elements at tick 0, exactly as the shard authors it (mirroring `inner_planet_mover`), and take the
    // FARTHEST — the culled outer planet.
    vd_core::worldgen::moving_children_for_config(
        p.universe_seed,
        &config,
        vd_core::pose::RealmId::System(7),
    )
    .into_iter()
    .map(|(_realm, elements)| vd_core::celestial::orbital_state(&elements, 0.0).position)
    .max_by(|a, b| a.length().total_cmp(&b.length()))
    .expect("the visual-demand forest has orbiting planets")
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
fn a_flying_occupant_streams_a_culled_planet_in_ahead_then_the_vacated_realm_is_reaped() {
    // THE demand-FLY — warp as a consequence of movement, on the compressed-real 5-planet Kepler geometry. A
    // logged-in occupant sits at the star (System 7); the OUTER planet is culled (beyond the ~59.5 m
    // visibility band from the star, so its box + shard do NOT exist yet). The occupant flies TOWARD it: as it
    // crosses into the planet's visibility band its shard spins up AHEAD (the box streams in), while the
    // occupant is STILL well outside the planet's ~4.16 m containment (`location` never flips — the box
    // arrives ahead of any crossing, exactly because visibility 59.5 m ≫ containment 4 m). Flying back to the
    // star, the vacated realm EMPTIES and its shard is reaped. No loading, no teleport, no predictive horizon
    // (`boot_ticks_p99` = 0) — the lead is the geometry.
    let f = fixture("fly");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let p = DEV; // move_speed 15 m/s, boot_ticks_p99 0 (no warm-ahead), the visual-demand scale.
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];
    // Spawn + a long fly-out/in + the drain/quiesce teardown windows ⇒ a materially longer budget than a login.
    let deadline = Duration::from_secs(120);

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &p, &client_book, client_quic, devctl_port);

    // The login lands the occupant at the star (System 7); the inner planets in view spin up, the outer are
    // culled by angular size.
    let landed = await_active(devctl_port, gw_admin, deadline);
    assert_eq!(
        landed.location.as_deref(),
        Some("System 7"),
        "the login lands at the star home realm: {landed:?}",
    );
    // Let the STATIONARY spin-ups (star + the inner planets already in visibility) settle — the baseline the
    // fly-out must climb above for a non-vacuous spin-up-ahead of the CULLED outer planet.
    let settled_spins = settle_spins(a.admin, Duration::from_secs(30));

    // ---- LEG 1: fly toward the CULLED outer planet → its box/shard streams in AHEAD of containment ----
    // Aim at a point ~20 m short (radially) of the outer planet's epoch position: inside its ~59.5 m
    // visibility band (spins it up) but well outside its ~4.16 m SOI (the occupant never enters containment).
    let outer = outer_planet_pos(&p);
    // Precondition tying the fly to the Slice-0 compressed-real geometry: the outer of the 5 Kepler planets
    // orbits at ~142 render-m (its epoch magnitude within a small eccentricity) — CULLED from the star (142 m
    // ≫ the ~59.5 m visibility band) yet still inside System 7's 150 m SOI. So the planet does NOT exist at
    // login and approaching it is a NON-VACUOUS spin-up-ahead.
    assert!(
        (130.0..150.0).contains(&outer.length()),
        "the outer planet is at its frozen ~142 m orbit, culled from the star: {}",
        outer.length(),
    );
    let aim = outer.normalize() * (outer.length() - OUTER_APPROACH_SHORT_M);
    walk_leg(devctl_port, "toward-outer-planet", aim, 3_000);
    let out = poll_state(devctl_port).expect("state after leg 1");
    // The occupant is STILL outside every planet's containment (it stopped ~20 m short) — `location` did NOT
    // flip to a planet. So any box that streamed in did so AHEAD of any containment entry.
    assert_eq!(
        out.location.as_deref(),
        Some("System 7"),
        "the occupant stays OUTSIDE planet containment across the fly-out (the box streams in ahead): {out:?}",
    );
    assert!(
        out.own_entity.is_some(),
        "the occupant kept authority across the fly-out: {out:?}",
    );
    assert_eq!(
        out.decode_errors, 0,
        "the client saw no decode faults across the fly-out: {out:?}",
    );
    let flew = orch_rlm(a.admin);
    // NON-VACUOUS: the culled outer planet was NOT in the settled baseline (beyond the star's 59.5 m band), so
    // a climb above it is the outer planet's shard spinning up AHEAD as the ship crossed into its visibility.
    assert!(
        flew.spins_requested > settled_spins,
        "the fly-out streamed a CULLED planet in ahead (spins {} > settled {settled_spins}): {flew:?}",
        flew.spins_requested,
    );
    assert_eq!(
        flew.spins_failed, 0,
        "no spawn failed across the fly-out: {flew:?}"
    );

    // ---- LEG 2: fly back to the star → the vacated realm reaps BEHIND ----
    // Fly to the origin (the star). The realm(s) the occupant flew away from EMPTY → self-report Empty → the
    // reconciler drains + quiesces + reaps them, while the still-occupied home (System 7) stays up.
    walk_leg(devctl_port, "back-to-star", DVec3::ZERO, 3_000);
    let back = poll_state(devctl_port).expect("state after leg 2");
    assert_eq!(
        back.location.as_deref(),
        Some("System 7"),
        "the occupant is back at the star: {back:?}",
    );
    // The vacated realm is torn down + its real shard process reaped. Anti-thrash rests entirely on the ~1 s
    // grace (equal AoI factors ⇒ no geometric dead-zone), so the reap LAGS the geometric exit by the grace
    // TAIL + drain + quiesce — a polled assertion (never asserted instantly), bounded by the deadline.
    let started = Instant::now();
    loop {
        let r = orch_rlm(a.admin);
        if r.teardowns_reaped >= 1 {
            break;
        }
        assert!(
            started.elapsed() < deadline,
            "the vacated realm was never torn down/reaped (teardowns_reaped stayed 0): {r:?}",
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}
