//! **G-FLOWN-SYMPTOM** — look_horizon.md §6 SLICE 5: the owner's own flight, fixed, asserted in
//! pixels — plus **G-IDENTICAL** (HR4) and **G-NOTHING-OWED** (THE LAW GATE).
//!
//! One DEMAND cluster (orchestrator + gateway, no shard pre-booked) booted onto THE world **plus
//! the planted player-built pair** (`VD_FIXTURE_PLANT=station-area` — the SL5 fixture-forest
//! doctrine's process path; `vd_physics::worldgen::station_area_plant` is the ONE derivation,
//! and every process of the cluster boots it through `vd_bins::process_world_config`), and one
//! real headless GPU client. The flight:
//!
//! 1. **PARK B** (G-IDENTICAL, the AREA half): standing inside the home system, park on the +Z
//!    polar axis outside the inner planet's shell and inside the planet's interior band. The
//!    planted AREA — a player-built region two levels below the observer — draws its OWN picture
//!    through the slice-3 sealed interior forward, at its camera-model size, parented on its
//!    PLANET. Fly out across the planet's interior stop level and back: the marker handover
//!    happens exactly once each way, no blank frame, no radius step above the readback quantum,
//!    and the reverse handover fits a derived budget that STATES the extra relay hop apart.
//! 2. **PARK A** (the flown symptom): exit the polar corridor, walk to the derived park —
//!    outside the 150 m shell, inside the 444.104489631 m interior band, along the OUTER
//!    planet's instantaneous radius vector. Every planet's presence is its OWN picture (DevState
//!    body kind + the composed provenance + real painted pixels) — RED for all five before
//!    slice 4; the outer planet's drawn radius is derived from the camera model, far above the
//!    three-pixel floor; all five planets draw DISTINCT radii; every planet row's parent is the
//!    star system THROUGHOUT. The planted STATION — a player-built realm KIND — passes the
//!    identical assertions in the same scene (G-IDENTICAL's station half).
//! 3. **OUT AND BACK across the stop level**: the marker handover happens exactly once per
//!    subject, no blank frame, no radius step above the quantum; the reverse handover fits the
//!    derived wake budget whose EXTRA RELAY HOP is stated as its own term.
//! 4. **G-NOTHING-OWED** at both parks: the generator (plus the plant spec) is the ORACLE,
//!    consulted OUT-OF-BAND — every generated subject whose true angular size exceeds the
//!    minimum has a drawn row carrying its OWN picture, at any depth, independent of the
//!    carrier. The oracle set is asserted non-empty and to include subjects at depth 2.
//!
//! Every bound is DERIVED from THE (planted) world's own numbers and the landed cadences; every
//! wait is on a signal, never a sleep standing in for one. GPU-required + LOCAL, exactly like
//! the other render gates.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::flight::cross_leg;
use vd_bins::pixel::{Author, Presence, Subject, own_pose};
use vd_bins::proc_launch::EARLY_DEATH_WINDOW;
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, admin_get_body, common_env,
    dev_auth_pubkey_hex, dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows,
    orchestrator_env, reap_forked, reserve_tcp_addr, reserve_udp_addr,
};
use vd_client_harness::assert::magenta_pixel_count;
use vd_client_harness::camera::{CaptureCamera, DOT_MIN_APPARENT_RADIUS_PX, ScreenAabb, ScreenPos};
use vd_client_harness::manifest::{MANIFEST_FILENAME, RunManifest};
use vd_client_harness::verdict::{dot_pixels_distinct_from_surround, projected_point_aabb};
use vd_client_render::{CAPTURE_H, CAPTURE_W};
use vd_core::NodeId;
use vd_core::glam::DVec3;
use vd_core::pose::{LatticePos, RealmId, frame_for_realm};
use vd_devproto::{CLIENT_NODE_BASE, DevPhase, DevRequest, DevResponse, DevState};
use vd_io_prod::trust::ClusterTrust;
use vd_physics::celestial::OrbitalElements;
use vd_physics::motion::Motion;
use vd_physics::worldgen::{StationAreaPlant, UniverseConfig, WorldView, station_area_plant};
use vd_wire::admin::{AdminSnapshot, GatewayView, RlmView};

// ---------------------------------------------------------------------------------------------
// THE DERIVED BUDGET — every term an expression over THE world's numbers or a landed cadence.
// ---------------------------------------------------------------------------------------------

/// The RLM reconciler's own interval (one tick).
const RECONCILE_TICKS: u64 = 1;
/// The interest byte's parent→child hop (its own reliable lane, slice 4): one tick.
const INTEREST_HOP_TICKS: u64 = 1;
/// The woken child authors its first sealed statement batch: one tick.
const AUTHOR_TICKS: u64 = 1;
/// Its parent forwards on the tick it receives (the landed Q2 relay — `process_inbound` runs
/// before `emit_realm_frames` on the same schedule).
const PARENT_FORWARD_TICKS: u64 = 1;
/// **THE EXTRA RELAY HOP** — slice 3's sealed interior forward: a depth-2 subject's picture rides
/// one hop further than a direct child's (child batch → parent holds → the GRANDPARENT forwards
/// it, sealed, to the gateway), so its wake pays exactly ONE more forward tick. §6 slice 5
/// demands this term stated APART, so a relay too slow to serve a deep wake fails BY NAME
/// instead of hiding inside a boot number.
const EXTRA_RELAY_HOP_TICKS: u64 = 1;
/// The gateway folds and ships on its next tick.
const COMPOSE_TICKS: u64 = 1;
/// The client applies the composed level and the renderer draws it.
const DRAW_TICKS: u64 = 1;

/// THE (planted) world — the one derivation this gate reads world numbers from (SL5: one world;
/// the plant is player-built content ON it, the same content every cluster process boots via
/// `VD_FIXTURE_PLANT`).
fn planted_config() -> UniverseConfig {
    UniverseConfig::world(DEV.move_speed, DEV.tick_dt).with_station_area_plant()
}

fn metres_per_tick() -> f64 {
    DEV.move_speed * DEV.tick_dt
}

/// The AoI re-check cadence in ticks — the beat the demand fold, the interest byte and the
/// window statements all breathe on (the shard's own expression).
fn aoi_cadence_ticks() -> u64 {
    (u64::from(DEV.tick_hz) / 2).max(1)
}

/// The reconciler LAUNCHES serially: each fork is watched for [`EARLY_DEATH_WINDOW`] on the
/// reconcile thread before the next child forks, so a wake that spins `n` children up in one
/// sweep pays `(n − 1)` windows before the LAST child even exists.
fn spawn_serialization_ticks(n_children: u64) -> u64 {
    let window_ticks = (EARLY_DEATH_WINDOW.as_secs_f64() / DEV.tick_dt).ceil() as u64;
    n_children.saturating_sub(1) * window_ticks
}

/// THE WAKE BUDGET for a DEPTH-2 subject, in ticks: the observer-realm's AoI beat + the interest
/// byte's hop + the vacated realm's own beat (the down-proxy fold) + the reconcile tick + the
/// spawn serialization + the cluster's own MEASURED boot + author + the Q2 parent forward + THE
/// EXTRA RELAY HOP (stated apart above) + compose + draw.
fn wake_budget_ticks(boot_ticks: u64, n_children: u64) -> u64 {
    2 * aoi_cadence_ticks()
        + INTEREST_HOP_TICKS
        + RECONCILE_TICKS
        + spawn_serialization_ticks(n_children)
        + boot_ticks
        + AUTHOR_TICKS
        + PARENT_FORWARD_TICKS
        + EXTRA_RELAY_HOP_TICKS
        + COMPOSE_TICKS
        + DRAW_TICKS
}

/// A footprint change smaller than one pixel is not a change anyone could see: one pixel is the
/// readback's own quantum, not a fitted tolerance.
const READBACK_QUANTUM_PX: f64 = 1.0;
/// How much wider than the drawn footprint a probed rectangle is bracketed — the shared factor
/// every pixel gate brackets by.
const RECT_BRACKET: f64 = 2.0;
/// A capture's ring probe width — the shared minimum apparent radius.
const PROBE_RING_PX: f64 = DOT_MIN_APPARENT_RADIUS_PX;

/// The login must SPAWN a real shard process before it can converge (under battery load a login
/// has been measured near 150 s).
const LOGIN_DEADLINE: Duration = Duration::from_secs(150);
/// A healthy client clears this once its home is up — below it is a stall.
const SNAPSHOT_FLOOR: u64 = 5;
/// One leg of this flight, bounded generously (walks + parked waits + captures).
const LEG_DEADLINE: Duration = Duration::from_secs(300);
/// How long the gate waits for a TEAR-DOWN's marker handovers: demand grace + teardown cooldown
/// plus drain and the look TTL, all wall-clock across five processes — bounded loosely (the wake
/// gate's own reap deadline class); the design puts NO latency budget on the outward direction.
const TEARDOWN_DEADLINE: Duration = Duration::from_secs(180);
/// The sampling interval — one tick of the shipped clock.
const SAMPLE_POLL: Duration = Duration::from_millis(20);
/// One walk chunk while watching, in ticks.
const WATCHED_CHUNK_TICKS: u64 = 5;
/// The settle discipline for orchestrator gauges (the wake gate's shape).
const STABLE_HOLDS: u32 = 5;

// ---------------------------------------------------------------------------------------------
// THE ORACLE — the generator (plus the plant spec), consulted OUT-OF-BAND.
// ---------------------------------------------------------------------------------------------

/// Everything G-NOTHING-OWED needs to enumerate the world without asking the picture: the
/// lowered regions, every mover's closed-form elements, and the ONE visibility factor
/// `cot(θ_min/2)`.
struct Oracle {
    regions: Vec<vd_core::geometry::RealmRegion>,
    movers: BTreeMap<RealmId, OrbitalElements>,
    factor: f64,
}

fn oracle() -> Oracle {
    let config = planted_config();
    let world = WorldView::generated(DEV.universe_seed, &config);
    let regions = world.regions().to_vec();
    let parents: std::collections::BTreeSet<RealmId> =
        regions.iter().filter_map(|r| r.parent).collect();
    let movers = parents
        .iter()
        .flat_map(|p| {
            vd_physics::worldgen::moving_children_for_config(DEV.universe_seed, &config, *p)
        })
        .collect();
    Oracle {
        regions,
        movers,
        factor: config.interest.spin_up_factor,
    }
}

impl Oracle {
    fn region(&self, realm: RealmId) -> &vd_core::geometry::RealmRegion {
        self.regions
            .iter()
            .find(|r| r.realm == realm)
            .unwrap_or_else(|| panic!("the oracle rosters {realm:?}"))
    }

    fn extent(&self, realm: RealmId) -> f64 {
        self.region(realm).shape.circumscribed_extent()
    }

    /// A body's offset in its PARENT's frame at `tick`: the closed-form orbit for a mover, the
    /// authored static offset otherwise — the same split the boot's reach map states.
    fn local_pos(&self, realm: RealmId, tick: u64) -> DVec3 {
        match self.movers.get(&realm) {
            Some(e) => {
                Motion::Kepler(*e)
                    .state_at(tick as f64 * DEV.tick_dt)
                    .origin
            }
            None => {
                let r = self.region(realm);
                let tier = r
                    .parent
                    .map_or(r.frame.tier(), |p| self.region(p).frame.tier());
                r.center.delta_m(LatticePos::local(DVec3::ZERO), tier)
            }
        }
    }

    /// A body's ABSOLUTE position at `tick` — the parent-chain fold the ORACLE (a test, not a
    /// realm) may lawfully perform.
    fn abs_pos(&self, realm: RealmId, tick: u64) -> DVec3 {
        let mut pos = self.local_pos(realm, tick);
        let mut cur = self.region(realm).parent;
        while let Some(p) = cur {
            pos += self.local_pos(p, tick);
            cur = self.region(p).parent;
        }
        pos
    }

    /// How many parent steps from `realm` up to `origin` — `None` when `origin` is not an
    /// ancestor (the subject sits outside the origin's subtree).
    fn depth_below(&self, realm: RealmId, origin: RealmId) -> Option<usize> {
        let mut steps = 0usize;
        let mut cur = realm;
        loop {
            if cur == origin {
                return Some(steps);
            }
            cur = self.region(cur).parent?;
            steps += 1;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Cluster scaffolding (the demand shape, planted).
// ---------------------------------------------------------------------------------------------

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct Fixture {
    trust_dir: std::path::PathBuf,
    common: Vec<(&'static str, String)>,
    store_str: String,
    launch_path: std::path::PathBuf,
    store: std::path::PathBuf,
    cwd: std::path::PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.trust_dir);
        let _ = std::fs::remove_file(&self.store);
        let _ = std::fs::remove_file(&self.launch_path);
    }
}

fn fixture(tag: &str) -> Fixture {
    let trust = ClusterTrust::generate("vd-look-pixels").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-look-{tag}-{}", std::process::id()));
    let trust_dir = base.join("trust");
    std::fs::create_dir_all(&trust_dir).expect("trust dir");
    trust.write_der_dir(&trust_dir).expect("write trust");
    let cwd = base.join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("capture cwd");
    let store = base.join("orchestrator.redb");
    let launch_path = store.with_file_name(vd_bins::LAUNCH_STORE_NAME);
    let _ = std::fs::remove_file(&store);
    let _ = std::fs::remove_file(&launch_path);
    let mut common = common_env(&trust_dir.display().to_string(), &DEV);
    // THE PLANT (look_horizon slice 5, G-IDENTICAL): every process of this cluster — the
    // orchestrator (whose spawn anchors forward it to every demand-spawned shard), the gateway,
    // and the client — boots THE world plus the station/area pair. One env value, one world.
    common.push(("VD_FIXTURE_PLANT", "station-area".to_owned()));
    Fixture {
        store_str: store.display().to_string(),
        trust_dir,
        common,
        launch_path,
        store,
        cwd,
    }
}

/// SIGKILL + reap every demand-spawned shard the orchestrator forked, on drop.
struct ForkedReaper(std::path::PathBuf);
impl Drop for ForkedReaper {
    fn drop(&mut self) {
        if self.0.exists() {
            reap_forked(&launch_rows(&self.0));
        }
    }
}

fn demand_addrs(gateway_admin: SocketAddr) -> ClusterAddrs {
    ClusterAddrs {
        gateway_admin: Some(gateway_admin),
        ..ClusterAddrs::reserve()
    }
}

fn spawn_capture_client(
    f: &Fixture,
    gateway: SocketAddr,
    name: &str,
    quic_port: u16,
    devctl_port: u16,
) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    for (k, v) in &f.common {
        cmd.env(k, v);
    }
    cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
    cmd.current_dir(&f.cwd);
    cmd.args([
        "--name",
        name,
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
        "--capture",
        "--capture-pilot",
    ]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().expect("spawn capture client")
}

fn boot_demand_cluster(
    f: &Fixture,
    a: &ClusterAddrs,
    p: &DevClusterParams,
    client_book: &[(NodeId, SocketAddr)],
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
    cluster
}

fn admin(addr: SocketAddr) -> Option<AdminSnapshot> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str(&body).ok()
}

fn orch_rlm(addr: SocketAddr) -> RlmView {
    admin(addr).map(|s| s.rlm).unwrap_or_default()
}

fn gateway_view(addr: SocketAddr) -> Option<GatewayView> {
    admin(addr)?.gateway
}

fn await_active(devctl_port: u16, gw_admin: SocketAddr, deadline: Duration) -> DevState {
    let started = Instant::now();
    loop {
        if let Ok(DevResponse::State { state }) = dev_roundtrip(devctl_port, &DevRequest::State)
            && state.phase == DevPhase::Active
            && state.snapshots_applied >= SNAPSHOT_FLOOR
            && !state.realm_boxes.is_empty()
        {
            return state;
        }
        assert!(
            started.elapsed() < deadline,
            "the demand login never converged: gateway view {:?}",
            gateway_view(gw_admin),
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn await_listener(port: u16, child: &mut Child) {
    let started = Instant::now();
    loop {
        if std::net::TcpStream::connect(vd_bins::loopback(port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "the capture client exited before serving dev-control ({status}) — GPU \
                 precondition: this gate needs a working adapter",
            );
        }
        assert!(
            started.elapsed() < Duration::from_secs(90),
            "the capture client never opened its dev-control listener on {port}",
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn label_of(realm: RealmId) -> String {
    frame_for_realm(realm, None)
        .expect("THE world's realms all have frames")
        .label()
        .to_owned()
}

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

/// Wait (bounded) for the reap gauge to CLIMB to the derived count, then settle and assert it
/// stops EXACTLY there — the wake gate's discipline: a teardown's reap lands seconds after the
/// marker handover (grace + cooldown + drain), so a settle alone reads a stable-but-early zero,
/// while the climb-then-settle proves both that the derived set reaped and that nothing else did.
fn await_reaps_exact(admin_addr: SocketAddr, expected: u64, what: &str) {
    let started = Instant::now();
    loop {
        let r = orch_rlm(admin_addr);
        if r.teardowns_reaped >= expected {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "{what}: the derived reaps ({expected}) never landed: {r:?}",
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    let settled = settle_gauge(admin_addr, Duration::from_secs(15), |r| r.teardowns_reaped);
    assert_eq!(
        settled, expected,
        "{what}: exactly the derived set reaped, nothing else"
    );
}

/// Wait (bounded) until the orchestrator's running gauge equals the DERIVED set — the wake
/// gate's discipline: a gauge sampled during the spawn-backoff window under-reads, so the gate
/// waits for the derived number instead of trusting one sample.
fn await_running(admin_addr: SocketAddr, expected: u64, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let r = orch_rlm(admin_addr);
        if r.running_gauge == expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: the running set never reached the derived {expected}: {r:?}",
        );
        std::thread::sleep(Duration::from_millis(500));
    }
}

// ---------------------------------------------------------------------------------------------
// Flight helpers.
// ---------------------------------------------------------------------------------------------

fn throttle_stop(devctl: u16) {
    let reply = dev_roundtrip(
        devctl,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    assert!(
        matches!(reply, Ok(DevResponse::Ack)),
        "all-stop refused: {reply:?}"
    );
}

fn walk_chunk(devctl: u16, aim: DVec3, ticks: u64) {
    let _ = dev_roundtrip(
        devctl,
        &DevRequest::WalkTo {
            target: aim.to_array(),
            arrive_epsilon: 2.0,
            max_ticks: ticks,
            max_step_m: 4.0 * DEV.move_speed * DEV.tick_dt,
        },
    );
    throttle_stop(devctl);
}

/// Turn the pilot's eyes toward `target` and let the in-flight orientation deltas land (a settle
/// signal: two identical delivered facings — never a sleep standing in for one).
fn look_at(devctl: u16, target: DVec3) {
    let reply = dev_roundtrip(
        devctl,
        &DevRequest::LookAt {
            target: target.to_array(),
            align_epsilon: 0.05,
            max_ticks: 900,
        },
    );
    assert!(
        matches!(reply, Ok(DevResponse::State { .. })),
        "the avatar never turned toward {target:?}: {reply:?}",
    );
    let mut last: Option<DVec3> = None;
    for _ in 0..40 {
        std::thread::sleep(SAMPLE_POLL);
        let st = vd_bins::pixel::poll(devctl);
        let (_, orient) = own_pose(&st).expect("a delivered pose to aim from");
        let facing = orient * DVec3::NEG_Z;
        if last.is_some_and(|f: DVec3| f.abs_diff_eq(facing, 1e-12)) {
            return;
        }
        last = Some(facing);
    }
}

/// Aim the eyes at a realm's CURRENT drawn centre.
fn face_realm(devctl: u16, realm: RealmId) {
    let st = vd_bins::pixel::poll(devctl);
    let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
    let centre = vd_bins::pixel::subject(&st, &camera, realm).centre_m;
    look_at(devctl, centre);
}

// ---------------------------------------------------------------------------------------------
// The leg instrument: presence, parent law, flip counting, radius steps.
// ---------------------------------------------------------------------------------------------

/// One observed author flip of one watched subject.
#[derive(Clone, Copy, Debug)]
struct FlipRec {
    tick: u64,
    to: Author,
    /// Radius at the sample before / at the flip, and the CAMERA-MODEL expectation at both —
    /// the step assert allows the quantum plus the model's own change over the gap (the model
    /// tracks each sample's own camera, so the observer's motion and turns cancel out).
    r_before_px: f64,
    r_at_px: f64,
    model_before_px: f64,
    model_at_px: f64,
    distance_m: f64,
}

/// The record of one watched leg.
struct LegRecord {
    /// Flips per watched subject, in watch order.
    flips: Vec<Vec<FlipRec>>,
    /// The first sample tick at which the observer stood within the trigger radius of the
    /// trigger realm's drawn centre (the wake-crossing instant the reverse budget measures from)
    /// and the sampling gap preceding it.
    trigger_tick: Option<u64>,
    trigger_gap: u64,
}

/// The camera-model expectation for an extent-sized subject: the projected footprint of the
/// ORACLE's radius at the drawn centre, floored at the shared minimum apparent radius — the same
/// Tier-A projection the renderer draws by, fed the out-of-band number. NEVER a literal.
fn model_radius_px(camera: &CaptureCamera, centre: DVec3, extent_m: f64) -> f64 {
    projected_point_aabb(camera, centre, extent_m)
        .map_or(0.0, |r| (r.max.x - r.min.x) * 0.5)
        .max(DOT_MIN_APPARENT_RADIUS_PX)
}

/// THE PARENT LAW, asserted on EVERY sample of every leg: every planet row's parent is the star
/// system (never the galaxy — slice 0's G-PARENT-TRUE, held THROUGHOUT the flight as §6 slice 5
/// demands), and each planted row parents on its own planted parent.
struct ParentLaw {
    home_debug: String,
    station_debug: String,
    area_debug: String,
    area_parent_debug: String,
}

impl ParentLaw {
    fn assert(&self, st: &DevState, leg: &str) {
        for b in &st.realm_boxes {
            if b.realm.starts_with("Planet(") {
                assert_eq!(
                    b.parent.as_deref(),
                    Some(self.home_debug.as_str()),
                    "{leg}: planet row {} must parent on the star system",
                    b.realm,
                );
            } else if b.realm == self.station_debug {
                assert_eq!(
                    b.parent.as_deref(),
                    Some(self.home_debug.as_str()),
                    "{leg}: the planted station must parent on the star system",
                );
            } else if b.realm == self.area_debug {
                assert_eq!(
                    b.parent.as_deref(),
                    Some(self.area_parent_debug.as_str()),
                    "{leg}: the planted area must parent on its planet",
                );
            }
        }
    }
}

/// Drive toward `park` (coasting once there) while WATCHING `watch`: THE PRESENCE LAW on every
/// sample (no watched subject is ever drawn by NOBODY — these legs cross no realm boundary, so
/// not even the epoch-bump rebuild allowance applies), THE PARENT LAW on every sample, every
/// author flip recorded with its radius step, and the wake-trigger crossing timed against the
/// `trigger` realm's drawn centre. `face` keeps the pilot's eyes on a subject so projections
/// stay in-frustum across the dive.
#[allow(clippy::too_many_arguments)]
fn watch_leg(
    devctl: u16,
    leg: &str,
    watch: &[(RealmId, f64)],
    park: DVec3,
    trigger: Option<(RealmId, f64)>,
    face: Option<RealmId>,
    parent_law: &ParentLaw,
    deadline: Duration,
    mut done: impl FnMut(&DevState, &[Vec<FlipRec>]) -> bool,
) -> LegRecord {
    let started = Instant::now();
    let mut flips: Vec<Vec<FlipRec>> = vec![Vec::new(); watch.len()];
    let mut prev: Vec<Option<(Author, f64, f64)>> = vec![None; watch.len()];
    let mut prev_tick: Option<u64> = None;
    let mut trigger_tick: Option<u64> = None;
    let mut trigger_gap = 0u64;
    loop {
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        let (pos, _) = own_pose(&st).expect("a delivered pose mid-leg");
        let tick = st.universe_tick.unwrap_or_default();
        let gap = prev_tick.map_or(0, |p| tick.saturating_sub(p));
        parent_law.assert(&st, leg);
        if let Some((t_realm, t_radius)) = trigger
            && trigger_tick.is_none()
        {
            let centre = vd_bins::pixel::subject(&st, &camera, t_realm).centre_m;
            if (pos - centre).length() <= t_radius {
                trigger_tick = Some(tick);
                trigger_gap = gap;
            }
        }
        let mut faced_out = false;
        for (i, &(realm, extent)) in watch.iter().enumerate() {
            let s = vd_bins::pixel::subject(&st, &camera, realm);
            let author = match s.presence {
                Presence::Drawn(a) => a,
                Presence::Absent => panic!(
                    "{leg}: THE PRESENCE LAW broken — {realm:?} drawn by NOBODY at tick {tick} \
                     (no realm boundary was crossed on this leg, so no rebuild allowance applies)",
                ),
            };
            faced_out |= s.rect.is_none();
            let dist = (s.centre_m - pos).length();
            let model = model_radius_px(&camera, s.centre_m, extent);
            if let Some((p_author, p_r, p_model)) = prev[i]
                && p_author != author
            {
                flips[i].push(FlipRec {
                    tick,
                    to: author,
                    r_before_px: p_r,
                    r_at_px: s.radius_px,
                    model_before_px: p_model,
                    model_at_px: model,
                    distance_m: dist,
                });
            }
            prev[i] = Some((author, s.radius_px, model));
        }
        if done(&st, &flips) {
            throttle_stop(devctl);
            return LegRecord {
                flips,
                trigger_tick,
                trigger_gap,
            };
        }
        assert!(
            started.elapsed() < deadline,
            "{leg}: never reached its stop condition (flips {:?}, trigger {trigger_tick:?})",
            flips.iter().map(Vec::len).collect::<Vec<_>>(),
        );
        prev_tick = Some(tick);
        // Keep the eyes on the faced subject when a watched projection has left the frustum
        // (the dive changes bearings by tens of degrees) — the model allowance tracks the
        // per-sample cameras, so a turn cannot fake or hide a radius step.
        if faced_out && let Some(realm) = face {
            face_realm(devctl, realm);
        }
        if (pos - park).length() > 5.0 {
            walk_chunk(devctl, park, WATCHED_CHUNK_TICKS);
        } else {
            std::thread::sleep(SAMPLE_POLL);
        }
    }
}

/// Every flip of a leg happened EXACTLY ONCE per subject, TO the expected author, with the
/// radius step across the handover inside the readback quantum plus the camera model's own
/// change over the bracketing gap (slice 1's size-matching: the extent-sized marker and the
/// look draw the SAME true angular size).
fn assert_flips(record: &LegRecord, watch: &[(RealmId, f64)], to: Author, leg: &str) {
    for (i, &(realm, _)) in watch.iter().enumerate() {
        let f = &record.flips[i];
        assert_eq!(
            f.len(),
            1,
            "{leg}: {realm:?} must hand over EXACTLY once (saw {f:?})",
        );
        let flip = f[0];
        assert_eq!(
            flip.to, to,
            "{leg}: {realm:?} handed over to the wrong author"
        );
        let step = (flip.r_at_px - flip.r_before_px).abs();
        let allowance = READBACK_QUANTUM_PX + (flip.model_at_px - flip.model_before_px).abs();
        assert!(
            step <= allowance,
            "{leg}: {realm:?} handover RADIUS STEP {step:.3} px exceeds the quantum plus the \
             model's own change ({allowance:.3} px) — the marker and the look disagree on the \
             subject's size (flip {flip:?})",
        );
        eprintln!(
            "[look] {leg}: {realm:?} handed over at tick {} ({:.1} m out), radius {:.2} → {:.2} \
             px (model {:.2} → {:.2} px)",
            flip.tick,
            flip.distance_m,
            flip.r_before_px,
            flip.r_at_px,
            flip.model_before_px,
            flip.model_at_px,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Pixel verdicts (the shared local-probe discipline).
// ---------------------------------------------------------------------------------------------

fn decode(cwd: &std::path::Path, rel: &str) -> (Vec<u8>, usize, usize, [u8; 4]) {
    let img = image::open(cwd.join(rel))
        .unwrap_or_else(|e| panic!("open capture {rel}: {e}"))
        .to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let buf = img.into_raw();
    let i = (w - 1) * 4;
    let clear = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
    (buf, w, h, clear)
}

/// Inflate a projected rectangle by the shared bracketing factor about its own centre (never
/// below the shared minimum apparent radius), plus `pad_px` on every side — the MEASURED
/// straddle drift of a fast-moving subject, so the probed rectangle covers the paint wherever
/// between the two agreeing polls the readback actually happened.
fn bracket(rect: ScreenAabb, pad_px: f64) -> ScreenAabb {
    let cx = f64::midpoint(rect.min.x, rect.max.x);
    let cy = f64::midpoint(rect.min.y, rect.max.y);
    let hw =
        ((rect.max.x - rect.min.x) * 0.5).max(DOT_MIN_APPARENT_RADIUS_PX) * RECT_BRACKET + pad_px;
    let hh =
        ((rect.max.y - rect.min.y) * 0.5).max(DOT_MIN_APPARENT_RADIUS_PX) * RECT_BRACKET + pad_px;
    ScreenAabb {
        min: ScreenPos {
            x: cx - hw,
            y: cy - hh,
        },
        max: ScreenPos {
            x: cx + hw,
            y: cy + hh,
        },
    }
}

fn nonclear_in(rgba: &[u8], w: usize, h: usize, clear: [u8; 4], rect: ScreenAabb) -> u64 {
    let x0 = rect.min.x.floor().max(0.0) as usize;
    let y0 = rect.min.y.floor().max(0.0) as usize;
    let x1 = (rect.max.x.ceil().max(0.0) as usize).min(w);
    let y1 = (rect.max.y.ceil().max(0.0) as usize).min(h);
    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * w + x) * 4;
            if rgba[i..i + 4] != clear {
                n += 1;
            }
        }
    }
    n
}

/// The subject actually PAINTED at its composed position (§2.11's local probe — never a
/// full-frame search). `distinct` additionally requires the pixels to stand apart from their
/// own local background — asserted where the lawful background is empty space, and skipped
/// where the subject lawfully NESTS inside another drawn body (the planted area sits within its
/// planet's disc, so the surround-distinct probe's premise fails by nesting, not by absence).
fn assert_painted(
    rgba: &(Vec<u8>, usize, usize, [u8; 4]),
    subject: &Subject,
    pad_px: f64,
    distinct: bool,
    what: &str,
) -> u64 {
    let (buf, w, h, clear) = rgba;
    let rect = bracket(
        subject
            .rect
            .unwrap_or_else(|| panic!("{what}: the subject must project in front of the eye")),
        pad_px,
    );
    let painted = nonclear_in(buf, *w, *h, *clear, rect);
    assert!(
        painted > 0,
        "{what}: NOTHING painted at the composed position (rect {rect:?}, footprint {:.2} px)",
        subject.radius_px,
    );
    if distinct {
        assert!(
            dot_pixels_distinct_from_surround(buf, *w, *h, rect, PROBE_RING_PX),
            "{what}: the pixels at the composed position are indistinguishable from the local \
             background",
        );
    }
    painted
}

/// **G-NOTHING-OWED** (THE LAW GATE, §3.3.4/§6 slice 5): from this very camera pose, enumerate
/// the generated forest OUT-OF-BAND (the oracle), compute each subject's true angular size, and
/// assert EVERY subject above the minimum angle has a drawn row carrying its OWN picture — at
/// any depth, independent of the carrier, the bound and the wake rule. Returns
/// `(owed, depth-2 owed)` so the caller proves non-vacuity.
fn assert_nothing_owed(
    st: &DevState,
    oracle: &Oracle,
    origin: RealmId,
    label: &str,
) -> (usize, usize) {
    let tick = st.universe_tick.expect("a ticked capture");
    let (eye_local, _) = own_pose(st).expect("a delivered pose");
    let eye = oracle.abs_pos(origin, tick) + eye_local;
    let mut owed = Vec::new();
    let mut depth2 = 0usize;
    for r in oracle.regions.iter().filter(|r| r.parent.is_some()) {
        let extent = r.shape.circumscribed_extent();
        let d = (oracle.abs_pos(r.realm, tick) - eye).length();
        // A subject is OWED when its true angular size exceeds the minimum AND the eye stands
        // OUTSIDE it: a realm that CONTAINS the eye is not a subject at all — the owner's
        // standing ruling that a containment boundary is never drawn as an object (its interior
        // IS the scene), so no angular size is stateable for it.
        if d > extent && d <= extent * oracle.factor {
            let depth = oracle.depth_below(r.realm, origin);
            let row = st
                .realm_boxes
                .iter()
                .find(|b| b.realm == format!("{:?}", r.realm))
                .unwrap_or_else(|| {
                    panic!(
                        "G-NOTHING-OWED ({label}): {:?} subtends the minimum angle ({d:.1} m \
                         against a reach of {:.1} m) yet has NO drawn row at all",
                        r.realm,
                        extent * oracle.factor,
                    )
                });
            assert_eq!(
                row.body_kind,
                "look",
                "G-NOTHING-OWED ({label}): {:?} subtends the minimum angle ({d:.1} m against a \
                 reach of {:.1} m) yet is drawn as a {} — the owner's law owes it its OWN picture",
                r.realm,
                extent * oracle.factor,
                row.body_kind,
            );
            if depth == Some(2) {
                depth2 += 1;
            }
            owed.push((r.realm, d, depth));
        }
    }
    eprintln!(
        "[look] G-NOTHING-OWED ({label}): {} subjects owed at this pose, every one drawn with \
         its OWN picture; {depth2} at depth 2: {owed:?}",
        owed.len(),
    );
    assert!(
        !owed.is_empty(),
        "G-NOTHING-OWED ({label}) is vacuous: nothing owed"
    );
    (owed.len(), depth2)
}

/// HR6: every capture is in the run manifest with the state dump attesting each drawn row's
/// lawful author.
fn assert_manifest_attests(cwd: &std::path::Path, labels: &[&str]) {
    let runs = cwd.join("runs");
    let run_dir = std::fs::read_dir(&runs)
        .unwrap_or_else(|e| panic!("no runs dir at {}: {e}", runs.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.join(MANIFEST_FILENAME).exists())
        .unwrap_or_else(|| panic!("no run manifest under {}", runs.display()));
    let manifest = RunManifest::from_json(
        &std::fs::read_to_string(run_dir.join(MANIFEST_FILENAME)).expect("read manifest"),
    )
    .expect("parse manifest");
    for label in labels {
        let entry = manifest
            .captures
            .iter()
            .find(|c| c.path.contains(label))
            .unwrap_or_else(|| panic!("capture '{label}' is not in the run manifest"));
        let state_rel = entry
            .state_path
            .as_ref()
            .unwrap_or_else(|| panic!("capture '{label}' has no state dump"));
        let dump: DevState = serde_json::from_str(
            &std::fs::read_to_string(run_dir.join(state_rel)).expect("read state dump"),
        )
        .expect("parse state dump");
        assert!(
            !dump.realm_boxes.is_empty(),
            "capture '{label}': no drawn rows"
        );
        for row in &dump.realm_boxes {
            assert!(
                row.body_kind == "look" || row.body_kind == "marker",
                "capture '{label}': row {} has no lawful author ({})",
                row.realm,
                row.body_kind,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// THE DERIVED GEOMETRY — shared by the companion and the flight.
// ---------------------------------------------------------------------------------------------

struct Derived {
    plant: StationAreaPlant,
    home: RealmId,
    outer: RealmId,
    outer_elements: OrbitalElements,
    /// The home system's shell radius (150 m) and interior band (444.104489631 /
    /// 469.104489631 m).
    shell: f64,
    spin: f64,
    tear: f64,
    /// Park A: outside the shell, inside the interior band — their midpoint, along the outer
    /// planet's instantaneous radius vector.
    park_a_r: f64,
    /// The out-park past the tear-down radius by half the interior band's width, DIVED off the
    /// orbital plane so no ring sibling can enter its own wake band at ANY azimuth.
    park_out_r: f64,
    /// The inner planet's shell and interior band (the planted area's park geometry).
    p_shell: f64,
    p_spin: f64,
    p_tear: f64,
    /// Park B / its out-park: +Z POLAR offsets from the SYSTEM centre — on the axis the range
    /// to an in-plane orbiter is `sqrt(z² + r²)`, phase-free, so a static park holds its band
    /// for the whole orbit and can never be captured.
    park_b_z: f64,
    park_bout_z: f64,
    inner_apo: f64,
}

fn derived() -> Derived {
    let config = planted_config();
    let plant = station_area_plant(DEV.universe_seed, &config);
    let world = WorldView::generated(DEV.universe_seed, &config);
    let home = plant.station_parent;
    let home_row = world
        .regions()
        .iter()
        .find(|r| r.realm == home)
        .expect("the home system is rostered");
    let shell = home_row.shape.circumscribed_extent();
    let spin = home_row.interior_band.spin_up_r_m();
    let tear = home_row.interior_band.tear_down_r_m();
    let movers = vd_physics::worldgen::moving_children_for_config(DEV.universe_seed, &config, home);
    let (outer, outer_elements) = movers
        .iter()
        .max_by(|a, b| a.1.sma.total_cmp(&b.1.sma))
        .copied()
        .expect("the home system orbits planets");
    let inner_elements = movers
        .iter()
        .find(|(r, _)| *r == plant.area_parent)
        .expect("the area's host planet is a mover")
        .1;
    let inner_apo = inner_elements.sma * (1.0 + inner_elements.ecc);
    let p_row = world
        .regions()
        .iter()
        .find(|r| r.realm == plant.area_parent)
        .expect("the inner planet is rostered");
    let p_shell = p_row.shape.circumscribed_extent();
    let p_spin = p_row.interior_band.spin_up_r_m();
    let p_tear = p_row.interior_band.tear_down_r_m();
    let step = metres_per_tick();
    Derived {
        park_a_r: f64::midpoint(shell, spin),
        park_out_r: tear + 0.5 * (spin - shell),
        // Park B: the polar range to the planet is sqrt(z² + r²) ≥ z, so the +Z offset is the
        // midpoint of the band that keeps the WHOLE orbit inside (shell .. sqrt(spin² − apo²)).
        park_b_z: f64::midpoint(p_shell, (p_spin * p_spin - inner_apo * inner_apo).sqrt()),
        // Its out-park: past the planet's interior tear-down at every phase (range ≥ z), inside
        // the system shell by two occupant steps.
        park_bout_z: f64::midpoint(p_tear, shell - 2.0 * step),
        plant,
        home,
        outer,
        outer_elements,
        shell,
        spin,
        tear,
        p_shell,
        p_spin,
        p_tear,
        inner_apo,
    }
}

// ---------------------------------------------------------------------------------------------
// THE COMPANION — the derivations closed before any process runs (milliseconds, no GPU).
// ---------------------------------------------------------------------------------------------

#[test]
fn g_flown_symptom_budgets_parks_and_the_plant_are_derived_and_lawful() {
    let d = derived();
    let oracle = oracle();
    let step = metres_per_tick();
    eprintln!(
        "[look] DERIVED on THE planted world (seed {}, {:.0} m/s, {:.3} s/tick):\n  \
         home {:?} shell {:.3} m · interior band {:.9} / {:.9} m\n  \
         park A {:.3} m (along {:?}'s instantaneous radius vector) · out-park {:.3} m (dived)\n  \
         station {:?} at {:?} extent {:.4} m · area {:?} on {:?} at {:?} extent {:.4} m\n  \
         inner planet shell {:.4} m · interior band {:.9} / {:.9} m · park B z {:.3} m · out z \
         {:.3} m · apo {:.3} m\n  \
         wake budget at zero boot: 1 child {} ticks · 6 children {} ticks (of which spawn \
         serialization {} and THE EXTRA RELAY HOP {EXTRA_RELAY_HOP_TICKS})",
        DEV.universe_seed,
        DEV.move_speed,
        DEV.tick_dt,
        d.home,
        d.shell,
        d.spin,
        d.tear,
        d.park_a_r,
        d.outer,
        d.park_out_r,
        d.plant.station,
        d.plant.station_offset_m,
        d.plant.station_extent_m,
        d.plant.area,
        d.plant.area_parent,
        d.plant.area_offset_m,
        d.plant.area_extent_m,
        d.p_shell,
        d.p_spin,
        d.p_tear,
        d.park_b_z,
        d.park_bout_z,
        d.inner_apo,
        wake_budget_ticks(0, 1),
        wake_budget_ticks(0, 6),
        spawn_serialization_ticks(6),
    );
    // The interior band on THE planted world is the pinned pair, and the parks bracket it.
    assert!((d.spin - 444.104_489_631).abs() < 1.0e-9, "spin {}", d.spin);
    assert!((d.tear - 469.104_489_631).abs() < 1.0e-9, "tear {}", d.tear);
    assert!(
        d.shell < d.park_a_r && d.park_a_r < d.spin,
        "park A brackets"
    );
    assert!(
        d.park_out_r > d.tear,
        "the out-park is past the tear-down radius"
    );
    // The out-park keeps every RING SIBLING asleep at EVERY azimuth: it dives off the orbital
    // plane, so the range to a sibling exceeds the sibling's own tear-down radius even when the
    // outer planet's radius vector points straight at one.
    let ring_r = oracle
        .regions
        .iter()
        .filter(|r| r.parent == Some(vd_core::worldgen::GALAXY) && r.realm != d.home)
        .map(|r| {
            r.center
                .delta_m(LatticePos::local(DVec3::ZERO), r.frame.tier())
                .length()
        })
        .fold(f64::INFINITY, f64::min);
    let sibling_tear = oracle
        .regions
        .iter()
        .find(|r| r.parent == Some(vd_core::worldgen::GALAXY) && r.realm != d.home)
        .expect("a ring sibling exists")
        .aoi
        .tear_down_r_m();
    let dive_z = (d.park_out_r * d.park_out_r - d.park_a_r * d.park_a_r).sqrt();
    let worst_sibling_range = ((ring_r - d.park_a_r).powi(2) + dive_z * dive_z).sqrt();
    assert!(
        worst_sibling_range > sibling_tear + step,
        "the dived out-park must keep every ring sibling asleep at every azimuth: worst range \
         {worst_sibling_range:.1} m vs sibling tear-down {sibling_tear:.1} m",
    );
    assert!(
        ring_r - d.park_a_r > sibling_tear + step,
        "park A keeps the ring asleep at every azimuth: {:.1} vs {:.1}",
        ring_r - d.park_a_r,
        sibling_tear,
    );
    // The exit-to-park-A walk clears the shell at every point: the exit parks at (0,0,−park_a)
    // and the park is in-plane, so the segment's closest approach to the system centre is
    // exactly park_a/√2 — above the shell by a wide, azimuth-independent margin.
    let closest = d.park_a_r / std::f64::consts::SQRT_2;
    assert!(
        closest > d.shell + 2.0 * step,
        "the exit→park walk grazes the shell: closest {closest:.1} m vs shell {:.1} m",
        d.shell,
    );
    // Park B and its out-park hold their bands for the WHOLE orbit (the polar phase-free
    // geometry), and the +Z legs clear the planted station by more than its extent plus a step.
    assert!(
        d.park_b_z > d.p_shell,
        "park B is outside the planet's shell"
    );
    let worst_b = (d.park_b_z * d.park_b_z + d.inner_apo * d.inner_apo).sqrt();
    assert!(
        worst_b < d.p_spin - step,
        "park B stays inside the planet's interior band at apoapsis: {worst_b:.2} vs {:.2}",
        d.p_spin,
    );
    assert!(
        d.park_bout_z > d.p_tear + step,
        "the out-park is past the planet's interior tear-down at every phase: {:.2} vs {:.2}",
        d.park_bout_z,
        d.p_tear,
    );
    assert!(
        d.park_bout_z < d.shell - step,
        "the out-park never leaves the system: {:.2} vs shell {:.2}",
        d.park_bout_z,
        d.shell,
    );
    assert!(
        d.plant.station_offset_m.x - d.plant.station_extent_m > step,
        "the +Z park legs clear the planted station: x {:.2} − extent {:.2} vs one step {step}",
        d.plant.station_offset_m.x,
        d.plant.station_extent_m,
    );
    // The planted area is OWED at park B at every orbital phase, and NEVER owed at park A (a
    // depth-3 subject for a galaxy observer — consistent with the carrier's arity by
    // MEASUREMENT, not assumption).
    let area_reach = d.plant.area_extent_m * oracle.factor;
    let worst_b_area = ((d.park_b_z - d.plant.area_offset_m.z).powi(2)
        + (d.inner_apo + d.plant.area_offset_m.z).powi(2))
    .sqrt();
    assert!(
        worst_b_area < area_reach,
        "the area is owed at park B at every phase: {worst_b_area:.2} vs reach {area_reach:.2}",
    );
    assert!(
        d.park_a_r - d.inner_apo - d.plant.area_offset_m.z > area_reach,
        "the area is NEVER owed at park A (depth 3 for a galaxy observer)",
    );
    // The outer planet is a GUARANTEED depth-2 owed subject at park A: parked along its radius
    // vector, the range is at most park − perihelion, far inside its visibility reach.
    let outer_reach = oracle.extent(d.outer) * oracle.factor;
    let outer_worst_d = d.park_a_r - d.outer_elements.sma * (1.0 - d.outer_elements.ecc);
    assert!(
        outer_worst_d < outer_reach,
        "the outer planet is owed at park A: {outer_worst_d:.1} m vs reach {outer_reach:.1} m",
    );
    // The six-child wake budget fits inside the leg deadline with real headroom.
    assert!(
        (wake_budget_ticks(0, 6) as f64) * DEV.tick_dt < LEG_DEADLINE.as_secs_f64() * 0.5,
        "the wake budget must fit the leg deadline",
    );
}

// ---------------------------------------------------------------------------------------------
// THE GATE.
// ---------------------------------------------------------------------------------------------

#[test]
fn g_flown_symptom_every_subject_draws_itself_at_the_owners_park() {
    // FIRST statement: hold the process tier for the whole body.
    let _tier = vd_bins::cluster_tier();

    let d = derived();
    let oracle = oracle();
    let f = fixture("flown");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_cluster(&f, &a, &DEV, &client_book);
    let mut client = ChildGuard(spawn_capture_client(
        &f,
        a.gateway,
        "g-flown",
        client_quic.port(),
        devctl,
    ));
    await_listener(devctl, &mut client.0);

    let home_label = label_of(d.home);
    let galaxy_label = label_of(vd_core::worldgen::GALAXY);
    let landed = await_active(devctl, gw_admin, LOGIN_DEADLINE);
    assert_eq!(
        landed.location.as_deref(),
        Some(home_label.as_str()),
        "the login lands at the home star: {landed:?}",
    );

    // THE DERIVED RUNNING SETS on the planted world: the chain (home + ancestors to the root) +
    // the 5 planets + the planted STATION (all in the system's visibility fold at login) + the
    // planted AREA (the spawn stands at the home centre, inside the inner planet's interior
    // band at every orbital phase, so the interest byte wakes it from login — the slice-4 lane
    // exercised one level deeper than the wake gate).
    let chain_len = {
        let mut n = 1u64;
        let mut cur = d.home;
        while let Some(parent) = oracle.region(cur).parent {
            n += 1;
            cur = parent;
        }
        n
    };
    let n_system_children = oracle
        .regions
        .iter()
        .filter(|r| r.parent == Some(d.home))
        .count() as u64; // the 5 planets + the planted station
    assert_eq!(n_system_children, 6, "5 planets + the planted station");
    let running_full = chain_len + n_system_children + 1; // + the planted area
    let running_no_area = chain_len + n_system_children;
    let running_chain = chain_len;
    await_running(a.admin, running_full, "login (planted world)");
    let mut reap_base = settle_gauge(a.admin, Duration::from_secs(10), |r| r.teardowns_reaped);
    eprintln!(
        "[look] LOGIN settled: running {running_full} (chain {chain_len} + {n_system_children} \
         system children + the area), reaps {reap_base}",
    );

    let parent_law = ParentLaw {
        home_debug: format!("{:?}", d.home),
        station_debug: format!("{:?}", d.plant.station),
        area_debug: format!("{:?}", d.plant.area),
        area_parent_debug: format!("{:?}", d.plant.area_parent),
    };
    let area_watch = [(d.plant.area, d.plant.area_extent_m)];

    // ============================ PHASE B — G-IDENTICAL, the AREA half =========================
    let park_b = DVec3::new(0.0, 0.0, d.park_b_z);
    let _ = watch_leg(
        devctl,
        "park-B approach",
        &[],
        park_b,
        None,
        None,
        &parent_law,
        LEG_DEADLINE,
        |st, _| own_pose(st).is_some_and(|(p, _)| (p - park_b).length() <= 5.0),
    );
    face_realm(devctl, d.plant.area_parent);
    // The straddle WATCHES the planet (whose footprint out-paces its own drift); the area rides
    // the same hull at the same speed but is a quarter its size, so its paint is probed with the
    // straddle's own MEASURED drift as padding instead.
    let cap_b = vd_bins::pixel::straddle(
        devctl,
        "identical-area",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &[d.plant.area_parent],
        vd_bins::pixel::pilot_camera,
    );
    {
        let rgba = decode(&f.cwd, &cap_b.shot);
        assert_eq!(
            magenta_pixel_count(&rgba.0),
            0,
            "identical-area: a missing-asset pixel"
        );
        let (eye, _) = own_pose(&cap_b.post).expect("a pose at park B");
        let area = vd_bins::pixel::subject(&cap_b.post, &cap_b.camera, d.plant.area);
        let planet = vd_bins::pixel::subject(&cap_b.post, &cap_b.camera, d.plant.area_parent);
        // G-IDENTICAL: the identical assertions, one level deeper, on a PLAYER-BUILT kind — the
        // area's presence is its OWN picture (the slice-4 wake + the slice-3 interior forward),
        // its delivered radius is THE world's own planted number, and its drawn footprint
        // matches the camera model at the measured range.
        assert_eq!(
            area.presence,
            Presence::Drawn(Author::SelfLook),
            "the planted area draws its OWN picture at park B",
        );
        assert_eq!(
            planet.presence,
            Presence::Drawn(Author::SelfLook),
            "its planet draws itself too",
        );
        let row = cap_b
            .post
            .realm_boxes
            .iter()
            .find(|b| b.realm == format!("{:?}", d.plant.area))
            .expect("the area's row is in the diagnosis surface");
        assert!(
            (row.extent_m - d.plant.area_extent_m).abs() < 1.0e-9,
            "the delivered extent ({}) is THE world's own planted number ({})",
            row.extent_m,
            d.plant.area_extent_m,
        );
        let d_area = (area.centre_m - eye).length();
        let expected_px = model_radius_px(&cap_b.camera, area.centre_m, d.plant.area_extent_m);
        assert!(
            (area.radius_px - expected_px).abs() < READBACK_QUANTUM_PX,
            "the area draws at its camera-model size: measured {:.2} px vs derived \
             {expected_px:.2} px",
            area.radius_px,
        );
        assert!(
            expected_px > 2.0 * (DOT_MIN_APPARENT_RADIUS_PX + READBACK_QUANTUM_PX),
            "far above the three-pixel floor: {expected_px:.2} px",
        );
        // DISTINCT from its planet wherever the camera model separates the two.
        let planet_model = model_radius_px(&cap_b.camera, planet.centre_m, d.p_shell);
        if (planet_model - expected_px).abs() > READBACK_QUANTUM_PX {
            assert_eq!(
                area.radius_px > planet.radius_px,
                expected_px > planet_model,
                "the area and its planet draw at distinct radii in the model's order",
            );
        }
        let painted = assert_painted(&rgba, &area, cap_b.drift_px, false, "identical-area/area");
        eprintln!(
            "[look] G-IDENTICAL (area): OWN picture at {d_area:.1} m — {:.2} px (derived \
             {expected_px:.2} px), {painted} px painted, planet {:.2} px, straddle drift \
             {:.2} px",
            area.radius_px, planet.radius_px, cap_b.drift_px,
        );
        assert_nothing_owed(&cap_b.post, &oracle, d.home, "park B");
    }

    // Outward across the planet's interior stop level: the area hands over to its planet's
    // marker EXACTLY once, no blank frame, no radius step above the quantum.
    let park_bout = DVec3::new(0.0, 0.0, d.park_bout_z);
    let out_b = watch_leg(
        devctl,
        "area-out",
        &area_watch,
        park_bout,
        None,
        Some(d.plant.area_parent),
        &parent_law,
        TEARDOWN_DEADLINE,
        |_, flips| !flips[0].is_empty(),
    );
    assert_flips(&out_b, &area_watch, Author::ParentMarker, "area-out");
    await_reaps_exact(a.admin, reap_base + 1, "area-out (exactly the area)");
    reap_base += 1;
    await_running(a.admin, running_no_area, "area-out");

    // Back in: the reverse handover within the derived budget, THE EXTRA RELAY HOP stated apart.
    let boot_ticks = orch_rlm(a.admin).boot_ticks_observed_max;
    let back_b = watch_leg(
        devctl,
        "area-return",
        &area_watch,
        park_b,
        Some((d.plant.area_parent, d.p_spin)),
        Some(d.plant.area_parent),
        &parent_law,
        LEG_DEADLINE,
        |_, flips| !flips[0].is_empty(),
    );
    assert_flips(&back_b, &area_watch, Author::SelfLook, "area-return");
    let budget_b = wake_budget_ticks(boot_ticks, 1);
    let trigger_b = back_b
        .trigger_tick
        .expect("the return leg crossed the planet's interior spin-up radius");
    let measured_b = back_b.flips[0][0].tick.saturating_sub(trigger_b);
    eprintln!(
        "[look] AREA REVERSE HANDOVER: {measured_b} ticks from the {:.3} m spin-up crossing vs \
         derived {budget_b} + {} resolution (2×{} AoI + {INTEREST_HOP_TICKS} interest hop + \
         {RECONCILE_TICKS} reconcile + {} spawn serialization + {boot_ticks} MEASURED boot + \
         {AUTHOR_TICKS} author + {PARENT_FORWARD_TICKS} Q2 forward + {EXTRA_RELAY_HOP_TICKS} \
         EXTRA RELAY HOP + {COMPOSE_TICKS} compose + {DRAW_TICKS} draw)",
        d.p_spin,
        back_b.trigger_gap + 1,
        aoi_cadence_ticks(),
        spawn_serialization_ticks(1),
    );
    assert!(
        measured_b <= budget_b + back_b.trigger_gap + 1,
        "G-IDENTICAL (area, reverse): {measured_b} ticks past the derived {budget_b} — if the \
         EXTRA RELAY HOP term is what broke it, the interior forward is too slow to serve a wake",
    );
    await_running(a.admin, running_full, "area-return");

    // ============================ PHASE A — THE FLOWN SYMPTOM ==================================
    cross_leg(
        devctl,
        "flown-exit home->galaxy (polar corridor)",
        |_tick| DVec3::new(0.0, 0.0, -d.park_a_r),
        &galaxy_label,
        Duration::from_secs(120),
    );
    // Leaving vacated the system: its occupancy observers are gone, so the interest byte to the
    // inner planet decays and the AREA reaps — THE CASCADE CAP measured in-process (an
    // interest-derived observer produces no interest of its own; the wake stops exactly two
    // levels below the observer).
    await_reaps_exact(
        a.admin,
        reap_base + 1,
        "post-exit (G-NO-CASCADE: exactly the area goes down when the system vacates)",
    );
    reap_base += 1;
    await_running(a.admin, running_no_area, "post-exit");

    // The radial park: aim from the DRAWN scene (the outer planet's instantaneous radius
    // vector), twice — the second walk corrects the first walk's sweep.
    for _ in 0..2 {
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        let home_c = vd_bins::pixel::subject(&st, &camera, d.home).centre_m;
        let outer_c = vd_bins::pixel::subject(&st, &camera, d.outer).centre_m;
        let u = (outer_c - home_c).normalize();
        let park_a = home_c + u * d.park_a_r;
        let _ = watch_leg(
            devctl,
            "park-A approach",
            &[],
            park_a,
            None,
            None,
            &parent_law,
            LEG_DEADLINE,
            |st, _| own_pose(st).is_some_and(|(p, _)| (p - park_a).length() <= 5.0),
        );
    }
    face_realm(devctl, d.home);

    let planets: Vec<RealmId> = oracle
        .regions
        .iter()
        .filter(|r| r.parent == Some(d.home) && matches!(r.realm, RealmId::Planet(_)))
        .map(|r| r.realm)
        .collect();
    assert_eq!(planets.len(), 5, "THE home system authors 5 planets");
    let mut watch_a: Vec<(RealmId, f64)> =
        planets.iter().map(|p| (*p, oracle.extent(*p))).collect();
    watch_a.push((d.plant.station, d.plant.station_extent_m));
    let watch_realms: Vec<RealmId> = watch_a.iter().map(|(r, _)| *r).collect();

    let cap_a = vd_bins::pixel::straddle(
        devctl,
        "flown-park",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch_realms,
        vd_bins::pixel::pilot_camera,
    );
    {
        let rgba = decode(&f.cwd, &cap_a.shot);
        assert_eq!(
            magenta_pixel_count(&rgba.0),
            0,
            "flown-park: a missing-asset pixel"
        );
        let (eye, _) = own_pose(&cap_a.post).expect("a pose at park A");
        let home_c = vd_bins::pixel::subject(&cap_a.post, &cap_a.camera, d.home).centre_m;
        let park_r = (eye - home_c).length();
        assert!(
            d.shell < park_r && park_r < d.spin,
            "the park stands outside the shell, inside the interior band: {park_r:.1} m",
        );
        // Every planet's presence is its OWN picture — RED for all five before slice 4 — plus
        // the planted station: the same assertions on a player-built KIND (G-IDENTICAL).
        let mut measured: Vec<(RealmId, f64, f64)> = Vec::new(); // (realm, radius_px, model_px)
        for &(realm, extent) in &watch_a {
            let s = vd_bins::pixel::subject(&cap_a.post, &cap_a.camera, realm);
            assert_eq!(
                s.presence,
                Presence::Drawn(Author::SelfLook),
                "{realm:?}'s presence at the owner's park is its OWN picture, not a parent \
                 marker",
            );
            let row = cap_a
                .post
                .realm_boxes
                .iter()
                .find(|b| b.realm == format!("{realm:?}"))
                .expect("a watched row");
            assert!(
                (row.extent_m - extent).abs() < 1.0e-9,
                "{realm:?}: delivered extent {} vs THE world's {extent}",
                row.extent_m,
            );
            let model = model_radius_px(&cap_a.camera, s.centre_m, extent);
            assert!(
                (s.radius_px - model).abs() < READBACK_QUANTUM_PX,
                "{realm:?}: measured {:.2} px vs camera-model {model:.2} px",
                s.radius_px,
            );
            let painted = assert_painted(
                &rgba,
                &s,
                cap_a.drift_px,
                realm == d.outer || realm == d.plant.station,
                &format!("flown-park/{realm:?}"),
            );
            measured.push((realm, s.radius_px, model));
            eprintln!(
                "[look] PARK A {realm:?}: OWN picture at {:.1} m — {:.2} px (model {model:.2} \
                 px), {painted} px painted",
                (s.centre_m - eye).length(),
                s.radius_px,
            );
        }
        // The OUTER planet: far above the three-pixel floor (the ~23 px class the design names
        // for this park), at the camera-model size.
        let outer_m = measured
            .iter()
            .find(|(r, ..)| *r == d.outer)
            .expect("outer measured");
        assert!(
            outer_m.1 > 2.0 * (DOT_MIN_APPARENT_RADIUS_PX + READBACK_QUANTUM_PX),
            "the outer planet's drawn radius {:.2} px must sit far above the {:.0} px floor",
            outer_m.1,
            DOT_MIN_APPARENT_RADIUS_PX,
        );
        // DISTINCT radii — not five identical dots: every planet pair the camera model
        // separates by more than the quantum must MEASURE separated, in the model's own order;
        // and the model must separate enough pairs for the assert to have teeth.
        let mut separated = 0u32;
        for i in 0..planets.len() {
            for j in (i + 1)..planets.len() {
                let (ri, mi) = (measured[i].1, measured[i].2);
                let (rj, mj) = (measured[j].1, measured[j].2);
                if (mi - mj).abs() > READBACK_QUANTUM_PX {
                    separated += 1;
                    assert_eq!(
                        ri > rj,
                        mi > mj,
                        "planets {:?} and {:?} must draw at DISTINCT radii in the model's order \
                         ({ri:.2} vs {rj:.2} px, model {mi:.2} vs {mj:.2})",
                        measured[i].0,
                        measured[j].0,
                    );
                }
            }
        }
        assert!(
            separated >= 3,
            "the five planets collapsed toward identical dots: only {separated} \
             model-separated pairs",
        );
        eprintln!(
            "[look] PARK A: five planets DISTINCT ({separated} separated pairs), straddle \
             drift {:.2} px",
            cap_a.drift_px,
        );
        // THE LAW GATE, from this very pose.
        let (owed, depth2) =
            assert_nothing_owed(&cap_a.post, &oracle, vd_core::worldgen::GALAXY, "park A");
        assert!(
            depth2 >= 1,
            "park A's oracle set includes depth-2 subjects ({owed} owed)"
        );
    }
    // The composed provenance: the deep pictures rode the slice-3 interior forward, lawfully.
    {
        let gw = gateway_view(gw_admin).expect("the gateway serves its admin snapshot");
        assert_eq!(
            gw.window_relay_interior_unvouched, 0,
            "no unvouched interior batch was ever admitted: {gw:?}",
        );
        assert!(
            gw.window_relay_interior_filtered > 0,
            "the interior forward's lawful filter ran (depth-3 parts dropped): {gw:?}",
        );
        assert!(
            gw.window_relays_ingested > 0,
            "the relay lane carried the pictures"
        );
        eprintln!(
            "[look] PROVENANCE: relays ingested {} · interior filtered {} · interior unvouched 0",
            gw.window_relays_ingested, gw.window_relay_interior_filtered,
        );
    }

    // ---- OUTWARD across the stop level: six look→marker handovers, exactly once each. ----
    let park_out = {
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        let home_c = vd_bins::pixel::subject(&st, &camera, d.home).centre_m;
        let (pos, _) = own_pose(&st).expect("a pose before the out-leg");
        let u = (pos - home_c).normalize();
        let dive_z = (d.park_out_r * d.park_out_r - d.park_a_r * d.park_a_r).sqrt();
        home_c + u * d.park_a_r + DVec3::new(0.0, 0.0, -dive_z)
    };
    let out_a = watch_leg(
        devctl,
        "flown-out",
        &watch_a,
        park_out,
        None,
        Some(d.home),
        &parent_law,
        TEARDOWN_DEADLINE,
        |_, flips| flips.iter().all(|f| !f.is_empty()),
    );
    assert_flips(&out_a, &watch_a, Author::ParentMarker, "flown-out");
    await_reaps_exact(
        a.admin,
        reap_base + n_system_children,
        "flown-out (exactly the system's interior)",
    );
    await_running(a.admin, running_chain, "flown-out");
    face_realm(devctl, d.home);
    let cap_out = vd_bins::pixel::straddle(
        devctl,
        "flown-out",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch_realms,
        vd_bins::pixel::pilot_camera,
    );
    for &(realm, _) in &watch_a {
        let s = vd_bins::pixel::subject(&cap_out.post, &cap_out.camera, realm);
        assert_eq!(
            s.presence,
            Presence::Drawn(Author::ParentMarker),
            "{realm:?} is its parent's point of light past the stop level",
        );
    }

    // ---- BACK IN: the reverse handover, within the derived budget, the relay hop stated. ----
    let boot_ticks = orch_rlm(a.admin).boot_ticks_observed_max;
    let park_ret = {
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        let home_c = vd_bins::pixel::subject(&st, &camera, d.home).centre_m;
        let (pos, _) = own_pose(&st).expect("a pose before the return");
        let mut back = pos - home_c;
        back.z = 0.0; // climb back onto the in-plane park along the same azimuth
        home_c + back.normalize() * d.park_a_r
    };
    let back_a = watch_leg(
        devctl,
        "flown-return",
        &watch_a,
        park_ret,
        Some((d.home, d.spin)),
        Some(d.home),
        &parent_law,
        LEG_DEADLINE,
        |_, flips| flips.iter().all(|f| !f.is_empty()),
    );
    assert_flips(&back_a, &watch_a, Author::SelfLook, "flown-return");
    let trigger_a = back_a
        .trigger_tick
        .expect("the return crossed the interior spin-up radius");
    let budget_a = wake_budget_ticks(boot_ticks, n_system_children);
    let last_flip = back_a
        .flips
        .iter()
        .map(|f| f[0].tick)
        .max()
        .expect("six flips");
    let measured_a = last_flip.saturating_sub(trigger_a);
    eprintln!(
        "[look] REVERSE HANDOVER (all {n_system_children} subjects): {measured_a} ticks from \
         the {:.3} m crossing vs derived {budget_a} + {} resolution (2×{} AoI + \
         {INTEREST_HOP_TICKS} interest hop + {RECONCILE_TICKS} reconcile + {} spawn \
         serialization + {boot_ticks} MEASURED boot + {AUTHOR_TICKS} author + \
         {PARENT_FORWARD_TICKS} Q2 forward + {EXTRA_RELAY_HOP_TICKS} EXTRA RELAY HOP + \
         {COMPOSE_TICKS} compose + {DRAW_TICKS} draw)",
        d.spin,
        back_a.trigger_gap + 1,
        aoi_cadence_ticks(),
        spawn_serialization_ticks(n_system_children),
    );
    assert!(
        measured_a <= budget_a + back_a.trigger_gap + 1,
        "the reverse handover took {measured_a} ticks, past the derived {budget_a} — if the \
         EXTRA RELAY HOP term is what broke it, the interior forward is too slow to serve a wake",
    );
    await_running(a.admin, running_no_area, "flown-return");
    face_realm(devctl, d.home);
    let cap_ret = vd_bins::pixel::straddle(
        devctl,
        "flown-return",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch_realms,
        vd_bins::pixel::pilot_camera,
    );
    for &(realm, _) in &watch_a {
        let s = vd_bins::pixel::subject(&cap_ret.post, &cap_ret.camera, realm);
        assert_eq!(
            s.presence,
            Presence::Drawn(Author::SelfLook),
            "{realm:?} draws itself again after the return",
        );
    }
    assert_nothing_owed(
        &cap_ret.post,
        &oracle,
        vd_core::worldgen::GALAXY,
        "park A (returned)",
    );

    // HR6: every capture, with every drawn row's provenance, is in the run manifest.
    assert_manifest_attests(
        &f.cwd,
        &["identical-area", "flown-park", "flown-out", "flown-return"],
    );
    drop(client);
}
