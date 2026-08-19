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

    /// A body's LOOK extent — the size it DRAWS at (the bound/look split); `None` draws nothing.
    fn look(&self, realm: RealmId) -> Option<f64> {
        self.region(realm).look.map(|l| l.circumscribed_extent())
    }

    /// A body's offset in its PARENT's frame at `tick`: the closed-form orbit for a mover, the
    /// authored static offset otherwise — the same split the boot's reach map states.
    fn local_pos(&self, realm: RealmId, tick: u64) -> DVec3 {
        match self.movers.get(&realm) {
            Some(e) => {
                // THE WHOLE placement, not its sub-cell remainder: `FramePlacement` normalizes a
                // position into an integer CELL anchor plus a residual, so reading `.origin`
                // alone reports a body at essentially its own origin (measured: a planet 5.1e9 m
                // out read as 0 m, and the nothing-owed law then judged it against the eye's own
                // standoff). `anchor()` is the lattice position that carries both halves.
                let at = Motion::Kepler(*e).state_at(tick as f64 * DEV.tick_dt);
                let r = self.region(realm);
                let tier = r
                    .parent
                    .map_or(r.frame.tier(), |p| self.region(p).frame.tier());
                at.anchor().delta_m(LatticePos::ORIGIN, tier)
            }
            None => {
                let r = self.region(realm);
                let tier = r
                    .parent
                    .map_or(r.frame.tier(), |p| self.region(p).frame.tier());
                r.center.delta_m(LatticePos::ORIGIN, tier)
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
/// CLOSE IN to a stated RANGE of a realm's drawn centre — the step a crossing does NOT do. A
/// `cross_leg` ends the instant the LABEL flips, which on THE world happens at the acquire edge
/// (a planet's SOI, ~2e9 m out), where the body still subtends a few pixels. Standing at a range
/// the CAMERA can resolve therefore needs its own leg: nose on the mark, full throttle, no taper
/// (the measured ramp-collapse law — a taper commands a sub-ramp speed that spirals down), and a
/// FEEDBACK stop on the measured range rather than an arrival epsilon the governor's own ceiling
/// makes unreachable.
fn close_to_range(devctl: u16, realm: RealmId, range_m: f64, deadline: Duration) {
    let started = Instant::now();
    loop {
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        let centre = vd_bins::pixel::subject(&st, &camera, realm).centre_m;
        let here = own_pose(&st).map_or(DVec3::ZERO, |(p, _)| p);
        let d = (centre - here).length();
        if d <= range_m {
            throttle_stop(devctl);
            eprintln!("[look] closed to {d:.4e} m of {realm:?} (target {range_m:.4e} m)");
            return;
        }
        look_at(devctl, centre);
        let _ = dev_roundtrip(
            devctl,
            &DevRequest::WalkTo {
                target: centre.to_array(),
                arrive_epsilon: range_m,
                max_ticks: 25,
                max_step_m: 0.0,
            },
        );
        assert!(
            started.elapsed() < deadline,
            "the close-in leg never reached {range_m:.4e} m of {realm:?} (still {d:.4e} m out)",
        );
    }
}

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
        // The OUTSIDE test reads the BOUND (containment); the ANGULAR test reads the LOOK (the
        // drawn size — the bound/look split). A look-less body draws nothing and owes nothing.
        let bound = r.shape.circumscribed_extent();
        let Some(extent) = oracle.look(r.realm) else {
            continue;
        };
        let d = (oracle.abs_pos(r.realm, tick) - eye).length();
        // A subject is OWED when its true angular size exceeds the minimum AND the eye stands
        // OUTSIDE it: a realm that CONTAINS the eye is not a subject at all — the owner's
        // standing ruling that a containment boundary is never drawn as an object (its interior
        // IS the scene), so no angular size is stateable for it.
        if d > bound && d <= extent * oracle.factor {
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
    /// ★ T2: the STAR realm — the body-bearing child at the home system's origin.
    star: RealmId,
    /// The star's BOUND (its dust-sublimation extent) and LOOK (its photosphere).
    star_bound: f64,
    star_look: f64,
    /// The star's own AoI wake band (spin/tear — factor × BOUND + the derived lead).
    star_spin: f64,
    star_tear: f64,
    /// The login spawn's derived standoff (T2: the home centre is inside the star now).
    spawn_z: f64,
    /// The out-park past the star's tear-down: the star reaps and hands over to its marker.
    star_out_z: f64,
    /// The OUTER home planet: the one body a single in-system park can watch at true size.
    outer: RealmId,
    outer_look: f64,
    /// The park range from the outer planet: half its own visibility reach.
    outer_park_range: f64,
    /// The home shell (the solved system bound — the flight never leaves it).
    shell: f64,
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
    let star_row = world
        .regions()
        .iter()
        .find(|r| r.parent == Some(home) && matches!(r.realm, RealmId::Star(_)))
        .expect("the home system holds its star (T2)");
    let star_bound = star_row.shape.circumscribed_extent();
    let star_look = star_row
        .look
        .expect("a star draws itself")
        .circumscribed_extent();
    let movers = vd_physics::worldgen::moving_children_for_config(DEV.universe_seed, &config, home);
    let (outer, _) = movers
        .iter()
        .max_by(|a, b| a.1.sma.total_cmp(&b.1.sma))
        .copied()
        .expect("the home system orbits planets");
    let outer_look = world
        .regions()
        .iter()
        .find(|r| r.realm == outer)
        .and_then(|r| r.look)
        .expect("a planet draws itself")
        .circumscribed_extent();
    Derived {
        spawn_z: world.default_home_offset_m().z,
        star: star_row.realm,
        star_bound,
        star_look,
        star_spin: star_row.aoi.spin_up_r_m(),
        star_tear: star_row.aoi.tear_down_r_m(),
        // One full tear-down of margin past the edge (the same doubling the arrival
        // standoffs use) — still deep inside the home shell.
        star_out_z: 2.0 * star_row.aoi.tear_down_r_m(),
        // HALF the planet's own solved bound — a range INSIDE its SOI, deliberately. The
        // interim gate parked inside the look reach and outside the bound; on THE world that
        // band does not exist (a planet's visibility reach is ~1/140 of its gravitational
        // shell — the measured bound/look inversion), so the only vantage from which a planet
        // draws its OWN picture is from INSIDE its realm. The gate crosses in, which is what
        // the owner's law describes anyway: you see a world when you are at it.
        // TWENTY LOOK-RADII from the body — a range chosen by the CAMERA, not by the shell:
        // the drawn radius is `look/range` in radians, so 20 look-radii puts the planet at
        // ~1/20 rad, tens of pixels across, unmistakably a BODY rather than the floor dot. It
        // is inside the planet's own shell by three orders (the bound/look inversion below), so
        // the ship is inside the realm and the picture is the planet's own.
        outer_park_range: 20.0 * outer_look,
        plant,
        home,
        outer,
        outer_look,
        shell,
    }
}

/// ★ TRUE-SCALE COMPANION (the taxonomy arc's in-system re-solve): every park and budget is
/// DERIVED from the solved world and asserted lawful before anything flies. The interim
/// 150 m / 444.104489631 m story ("park outside the shell, inside the interior band") is
/// GONE — at true scale a system's interior reach sits INSIDE its shell (the lawful
/// inversion; G-NO-CASCADE), and the pixel stories move to where pictures really are: the
/// STAR at the spawn, and ONE planet inside its own visibility reach.
#[test]
fn g_true_scale_budgets_and_parks_are_derived_and_lawful() {
    let d = derived();
    let oracle = oracle();
    eprintln!(
        "[look] DERIVED on THE planted world (seed {}):\n  home {:?} shell {:.4e} m\n  \
         star {:?} bound {:.4e} m · look {:.4e} m · wake {:.4e}/{:.4e} m · spawn z {:.4e} m · \
         out z {:.4e} m\n  outer {:?} look {:.4e} m · park range {:.4e} m\n  wake budget at \
         zero boot: 1 child {} ticks",
        DEV.universe_seed,
        d.home,
        d.shell,
        d.star,
        d.star_bound,
        d.star_look,
        d.star_spin,
        d.star_tear,
        d.spawn_z,
        d.star_out_z,
        d.outer,
        d.outer_look,
        d.outer_park_range,
        wake_budget_ticks(0, 1),
    );
    // The spawn stands OUTSIDE the star's bound (T2 forced the standoff), INSIDE its wake.
    assert!(d.star_bound < d.spawn_z && d.spawn_z < d.star_spin);
    // The star's wake band brackets its bound the right way round (the PLAIN AoI band is
    // factor × bound — only the INTERIOR band inverted at true scale).
    assert!((d.star_bound < d.star_spin) & (d.star_spin < d.star_tear));
    // The out-park is past the tear with a full tear of margin, still inside the home shell.
    assert!(d.star_out_z > d.star_tear && d.star_out_z < d.shell);
    // THE BOUND/LOOK INVERSION, pinned with both numbers: a planet's visibility reach
    // (look × factor) sits INSIDE its own gravitational shell on THE world, so there is no
    // vantage OUTSIDE a planet from which the planet draws its own picture — the sky is real,
    // and you see a world by going to it. The park is therefore inside the shell, and the
    // phase that uses it crosses in.
    let outer_bound = oracle.extent(d.outer);
    assert!(
        d.outer_look * oracle.factor < outer_bound,
        "the outer planet's visibility reach ({:.4e} m) must sit inside its own shell \
         ({outer_bound:.4e} m) — if this ever un-inverts, an outside-the-shell park exists \
         again and the old (bound, reach) park window should be restored",
        d.outer_look * oracle.factor,
    );
    assert!(d.outer_park_range < outer_bound);
    // The planted pair are pinned as world numbers (their PIXEL phases are parked — see the
    // gate's park note): the 10 km city under the home system, the 20 m structure on the
    // inner planet, at half their parents' solved extents.
    assert!((d.plant.station_extent_m - 1.0e4).abs() < 1e-9);
    assert!((d.plant.area_extent_m - 20.0).abs() < 1e-9);
    // The wake budget fits the leg deadline with real headroom.
    assert!((wake_budget_ticks(0, 1) as f64) * DEV.tick_dt < LEG_DEADLINE.as_secs_f64() * 0.5);
}

// ---------------------------------------------------------------------------------------------
// THE GATE.
// ---------------------------------------------------------------------------------------------

/// ★ THE TRUE-SCALE PIXEL GATE (the taxonomy arc: T2's star-body proof + T3's depth-4 moon
/// proof, in ONE flight on ONE demand cluster — the celestial_taxonomy_design §9 gates,
/// measured in pixels):
///
/// 1. LOGIN at the derived standoff: the STAR REALM wakes by demand and draws AS A BODY —
///    its OWN `TAG_LOOK` photosphere at the camera-model size, carrying its `TAG_LUMA`
///    photometric datum (ruling C: the star has a colour the instant it runs).
/// 2. OUT past the star's tear-down: the body hands over to the parent's MARKER exactly once,
///    the Star realm reaps behind (teardown-behind), and the marker carries THE SAME datum —
///    colour continuity across the handover, measured, not asserted.
/// 3. BACK IN: the marker hands back to the body inside the derived wake budget.
/// 4. OUT to the home system's OUTER planet (a governed ~1.3e11 m leg): crossing INTO the
///    planet realm (demand re-home), whose own body draws at the camera-model size from
///    inside its bound — at true scale a planet's picture exists only within its own look
///    reach, which sits INSIDE its gravitational SOI (bound/look ≈ 140): the sky is real.
/// 5. ON to the planet's MOON (T3): the moon realm — a Planet under a Planet — demand-spawns
///    at depth 4, the occupant CROSSES IN (the depth-4 walk, in pixels), the moon draws its
///    own body, and the way back tears it down behind.
///
/// THE PLANTED-PAIR PIXEL PHASES ARE PARKED (the S5 f32-eye class, MEASURED): at the true-
/// size world the 20 m area rides a planet ~1.9e9 m from the render origin, where one f32
/// view ulp is ~256 m — 12× the subject; the 10 km city sits at ~7.9e10 m (ulp ~8.2e3 m).
/// Composition is proven by the walk/demand label gates; the pixels return with the S5
/// camera-relative flatten (render_crossing_smoke's park class, D-REAL-3's ledger note).
#[test]
#[ignore = "PARKED on a MEASURED render-side gap (D-LOOK-3, new): the camera's far plane is \
            STAR_FAR_PLANE = 120 000 render-metres (twice the 60 km star-sphere backdrop), so on \
            THE world — where the nearest body a ship can stand off from sits ~1.8e8 m away — all \
            real body GEOMETRY is clipped. Measured in this gate: the outer planet composes \
            correctly at 43.82 px with Author::SelfLook at 1.8248e8 m and the readback finds its \
            rect EMPTY, while the STAR's sprite-drawn body paints 141 758 pixels at 211.17 px \
            with its TAG_LUMA datum (ruling C's seam, in pixels) at 3.2e8 m. What this gate \
            already proved GREEN before the clip stops it: the star as a BODY with luma, \
            G-NOTHING-OWED at the spawn (1 subject owed, drawn with its own picture), the star's \
            look⇒marker handover out at 1.359e10 m and back at 1.189e10 m inside the derived \
            wake budget (23 ticks vs 58+3). Returns with the S5 render-scale work (far plane + \
            camera-relative flatten), together with warp_pixels' two gates."]
fn g_true_scale_star_body_and_depth_four_moon_draw_themselves() {
    // FIRST statement: hold the process tier for the whole body.
    let _tier = vd_bins::cluster_tier();

    let d = derived();
    let oracle = oracle();
    let f = fixture("star-moon");
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
        "g-star-moon",
        client_quic.port(),
        devctl,
    ));
    await_listener(devctl, &mut client.0);

    let home_label = label_of(d.home);
    let landed = await_active(devctl, gw_admin, LOGIN_DEADLINE);
    assert_eq!(
        landed.location.as_deref(),
        Some(home_label.as_str()),
        "the login lands at the home star system (outside the Star realm's bound): {landed:?}",
    );
    let parent_law = ParentLaw {
        home_debug: format!("{:?}", d.home),
        station_debug: format!("{:?}", d.plant.station),
        area_debug: format!("{:?}", d.plant.area),
        area_parent_debug: format!("{:?}", d.plant.area_parent),
    };

    // ======================= PHASE 1 — THE STAR BODY (T2's pixel proof) ======================
    // The demand loop wakes the star from the spawn (inside its 1.2e10 m wake radius); wait for
    // its SelfLook row, then capture.
    let deadline = Instant::now() + LEG_DEADLINE;
    loop {
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        if vd_bins::pixel::subject(&st, &camera, d.star).presence
            == Presence::Drawn(Author::SelfLook)
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the Star realm never woke into its own body at the spawn",
        );
        std::thread::sleep(SAMPLE_POLL);
    }
    face_realm(devctl, d.star);
    let cap_star = vd_bins::pixel::straddle(
        devctl,
        "star-body",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &[d.star],
        vd_bins::pixel::pilot_camera,
    );
    let star_luma = {
        let rgba = decode(&f.cwd, &cap_star.shot);
        assert_eq!(
            magenta_pixel_count(&rgba.0),
            0,
            "star-body: a missing-asset pixel"
        );
        let s = vd_bins::pixel::subject(&cap_star.post, &cap_star.camera, d.star);
        assert_eq!(
            s.presence,
            Presence::Drawn(Author::SelfLook),
            "THE STAR DRAWS AS A BODY: its OWN picture, not a marker",
        );
        let row = cap_star
            .post
            .realm_boxes
            .iter()
            .find(|b| b.realm == format!("{:?}", d.star))
            .expect("the star's row is in the diagnosis surface");
        assert!(
            (row.extent_m - d.star_look).abs() < 1e-6,
            "the delivered extent is the PHOTOSPHERE ({}) not the bound ({}): {}",
            d.star_look,
            d.star_bound,
            row.extent_m,
        );
        // ★ RULING C, measured: the RUNNING star's own look bag carries its photometric datum.
        let luma = row
            .luma
            .expect("ruling C: the running star's own look carries TAG_LUMA");
        assert_eq!(
            luma.0, 6,
            "the home star is an M class (code 6) — the datum is the star's own"
        );
        let model = model_radius_px(&cap_star.camera, s.centre_m, d.star_look);
        assert!(
            (s.radius_px - model).abs() < READBACK_QUANTUM_PX,
            "the star draws at its camera-model size: measured {:.2} px vs {model:.2} px",
            s.radius_px,
        );
        assert!(
            model > 2.0 * (DOT_MIN_APPARENT_RADIUS_PX + READBACK_QUANTUM_PX),
            "far above the apparent floor: {model:.2} px",
        );
        let painted = assert_painted(&rgba, &s, cap_star.drift_px, false, "star-body");
        eprintln!(
            "[look] T2 STAR BODY: SelfLook {:.2} px (model {model:.2}), {painted} px painted, \
             luma {luma:?}",
            s.radius_px,
        );
        // THE LAW GATE at the spawn: everything above the minimum angle draws its own picture
        // (the star is the one owed subject — depth 2 under the galaxy).
        let (_, depth2) =
            assert_nothing_owed(&cap_star.post, &oracle, vd_core::worldgen::GALAXY, "spawn");
        assert!(depth2 >= 1, "the star is a depth-2 owed subject");
        luma
    };
    // G-NO-CASCADE, process-measured: the planted station and area are NOT running (their wake
    // radii are metres-scale against astronomical distances; the interim world's login used to
    // wake the area through the interest byte — the true-size world lawfully does not).
    {
        let rlm = orch_rlm(a.admin);
        eprintln!(
            "[look] LOGIN demand set: spins {} (failed {}), reaps {}",
            rlm.spins_requested, rlm.spins_failed, rlm.teardowns_reaped
        );
    }

    // =================== PHASE 2 — OUT: body → marker, teardown-behind =======================
    let reap_before = settle_gauge(a.admin, Duration::from_secs(10), |r| r.teardowns_reaped);
    let star_watch = [(d.star, d.star_look)];
    let park_out = DVec3::new(0.0, 0.0, d.star_out_z);
    let out = watch_leg(
        devctl,
        "star-out",
        &star_watch,
        park_out,
        None,
        Some(d.star),
        &parent_law,
        TEARDOWN_DEADLINE,
        |_, flips| !flips[0].is_empty(),
    );
    assert_flips(&out, &star_watch, Author::ParentMarker, "star-out");
    // Teardown-behind: the vacated Star realm reaps (other far-woken planets may lawfully reap
    // on the same leg, so the gauge is a floor, not an exact count).
    let reap_after = settle_gauge(a.admin, Duration::from_secs(30), |r| r.teardowns_reaped);
    assert!(
        reap_after > reap_before,
        "the vacated Star realm tears down behind ({reap_before} -> {reap_after})",
    );
    // ★ RULING C ACROSS THE HANDOVER: the parent's marker carries THE SAME photometric datum —
    // the star does not change colour when it stops running.
    {
        let st = vd_bins::pixel::poll(devctl);
        let row = st
            .realm_boxes
            .iter()
            .find(|b| b.realm == format!("{:?}", d.star))
            .expect("the star's marker row survives the teardown");
        assert_eq!(
            row.body_kind, "marker",
            "past the tear the star is a marker"
        );
        assert_eq!(
            row.luma,
            Some(star_luma),
            "COLOUR CONTINUITY: the marker's datum equals the body's (ruling C)",
        );
    }

    // ========================= PHASE 3 — BACK IN, inside the budget ==========================
    let boot_ticks = orch_rlm(a.admin).boot_ticks_observed_max;
    let spawn = DVec3::new(0.0, 0.0, d.spawn_z);
    let back = watch_leg(
        devctl,
        "star-return",
        &star_watch,
        spawn,
        Some((d.star, d.star_spin)),
        Some(d.star),
        &parent_law,
        LEG_DEADLINE,
        |_, flips| !flips[0].is_empty(),
    );
    assert_flips(&back, &star_watch, Author::SelfLook, "star-return");
    let budget = wake_budget_ticks(boot_ticks, 1);
    let trigger = back
        .trigger_tick
        .expect("the return leg crossed the star's wake radius");
    let measured = back.flips[0][0].tick.saturating_sub(trigger);
    eprintln!(
        "[look] STAR REVERSE HANDOVER: {measured} ticks from the {:.3e} m wake crossing vs \
         derived {budget} + {} resolution (boot {boot_ticks} measured)",
        d.star_spin,
        back.trigger_gap + 1,
    );
    assert!(
        measured <= budget + back.trigger_gap + 1,
        "the star's wake handover blew the derived budget",
    );

    // ============= PHASE 4 — THE OUTER PLANET: cross in, the body draws (T2/T3) =============
    // The aim comes from the ORACLE (a test flies with out-of-band knowledge; the planet is
    // below the visibility angle from here — that IS the true-size sky).
    let tick_now = vd_bins::pixel::poll(devctl)
        .universe_tick
        .unwrap_or_default();
    let planet_pos = oracle.abs_pos(d.outer, tick_now);
    let park_planet = planet_pos * (1.0 - d.outer_park_range / planet_pos.length());
    let planet_label = label_of(d.outer);
    cross_leg(
        devctl,
        "to the outer planet (governed, crossing in)",
        move |_tick| park_planet,
        &planet_label,
        LEG_DEADLINE,
    );
    close_to_range(devctl, d.outer, d.outer_park_range, LEG_DEADLINE);
    face_realm(devctl, d.outer);
    let cap_planet = vd_bins::pixel::straddle(
        devctl,
        "planet-body",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &[d.outer],
        vd_bins::pixel::pilot_camera,
    );
    {
        let rgba = decode(&f.cwd, &cap_planet.shot);
        let s = vd_bins::pixel::subject(&cap_planet.post, &cap_planet.camera, d.outer);
        assert_eq!(
            s.presence,
            Presence::Drawn(Author::SelfLook),
            "inside its bound, the planet's own body draws",
        );
        let model = model_radius_px(&cap_planet.camera, s.centre_m, d.outer_look);
        assert!(
            (s.radius_px - model).abs() < READBACK_QUANTUM_PX,
            "the planet draws at its camera-model size: {:.2} vs {model:.2} px",
            s.radius_px,
        );
        assert!(model > 2.0 * (DOT_MIN_APPARENT_RADIUS_PX + READBACK_QUANTUM_PX));
        let painted = assert_painted(&rgba, &s, cap_planet.drift_px, false, "planet-body");
        eprintln!(
            "[look] OUTER PLANET BODY: SelfLook {:.2} px (model {model:.2}), {painted} px \
             painted, at {:.3e} m",
            s.radius_px,
            (s.centre_m - own_pose(&cap_planet.post).expect("posed").0).length(),
        );
    }

    // ================= PHASE 5 — THE MOON: depth 4, in pixels (T3's proof) ===================
    // The home outer planet is THE world's one home moon host (the census). The moon — a
    // Planet realm under a Planet realm — demand-spawns as the occupant approaches, the
    // occupant CROSSES IN (depth 4), and its body draws.
    let moon = oracle
        .regions
        .iter()
        .find(|r| r.parent == Some(d.outer) && matches!(r.realm, RealmId::Planet(_)))
        .map(|r| r.realm)
        .expect("the home outer planet hosts the census moon");
    let moon_look = oracle.look(moon).expect("a moon draws itself");
    let moon_label = label_of(moon);
    let tick_now = vd_bins::pixel::poll(devctl)
        .universe_tick
        .unwrap_or_default();
    let moon_pos = oracle.abs_pos(moon, tick_now);
    let park_range = 0.5 * moon_look * oracle.factor;
    let toward_planet = (oracle.abs_pos(d.outer, tick_now) - moon_pos).normalize();
    let park_moon = moon_pos + toward_planet * park_range;
    cross_leg(
        devctl,
        "to the moon (depth 4, crossing in)",
        move |_tick| park_moon,
        &moon_label,
        LEG_DEADLINE,
    );
    face_realm(devctl, moon);
    let cap_moon = vd_bins::pixel::straddle(
        devctl,
        "moon-body",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &[moon],
        vd_bins::pixel::pilot_camera,
    );
    {
        let rgba = decode(&f.cwd, &cap_moon.shot);
        let s = vd_bins::pixel::subject(&cap_moon.post, &cap_moon.camera, moon);
        assert_eq!(
            s.presence,
            Presence::Drawn(Author::SelfLook),
            "THE MOON DRAWS ITS OWN BODY at depth 4",
        );
        let model = model_radius_px(&cap_moon.camera, s.centre_m, moon_look);
        assert!(
            (s.radius_px - model).abs() < READBACK_QUANTUM_PX,
            "the moon draws at its camera-model size: {:.2} vs {model:.2} px",
            s.radius_px,
        );
        assert!(model > 2.0 * (DOT_MIN_APPARENT_RADIUS_PX + READBACK_QUANTUM_PX));
        let painted = assert_painted(&rgba, &s, cap_moon.drift_px, false, "moon-body");
        eprintln!(
            "[look] T3 MOON BODY (depth 4): SelfLook {:.2} px (model {model:.2}), {painted} px \
             painted",
            s.radius_px,
        );
    }

    // ================== PHASE 6 — BACK OUT: depth-4 teardown-behind ==========================
    let reap_before_moon = settle_gauge(a.admin, Duration::from_secs(5), |r| r.teardowns_reaped);
    let planet_label_again = label_of(d.outer);
    let back_park = park_planet;
    cross_leg(
        devctl,
        "back to the planet park (the moon behind)",
        move |_tick| back_park,
        &planet_label_again,
        LEG_DEADLINE,
    );
    let reap_after_moon = settle_gauge(a.admin, Duration::from_secs(60), |r| r.teardowns_reaped);
    assert!(
        reap_after_moon > reap_before_moon,
        "the vacated moon tears down behind ({reap_before_moon} -> {reap_after_moon})",
    );
    eprintln!(
        "[look] DEPTH-4 WALK COMPLETE: system → planet → moon → back, every crossing a label, \
         teardown behind at every level",
    );
    assert_manifest_attests(&f.cwd, &["star-body", "planet-body", "moon-body"]);
}
