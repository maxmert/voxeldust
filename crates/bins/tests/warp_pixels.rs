//! **G-WARP-PIXELS + G-HANDOVER** — THE WARP ACCEPTANCE, in real pixels
//! (`docs/design/window_lane.md` §2.8 + §4 Slice D).
//!
//! One demand cluster (orchestrator + gateway, NO shard pre-booked), one real headless client with
//! a GPU, and one flight down THE world's own star ring: out of the home system, across the gap,
//! into the ring sibling 12 031.398 m away.
//!
//! WHAT IT PROVES, in the owner's words: *the destination is a point of light at departure, it
//! grows monotonically as you approach, and it hands over flicker-free into a live system that
//! draws itself; behind you the system you left shrinks to a dot and tears down.* No loading
//! screen, no toggle, no teleport — the same picture the whole way.
//!
//! WHY IT CAN FAIL. Every bound is DERIVED from THE world's solved geometry and the landed
//! cadences, then compared against a measurement of the real cluster:
//!
//! * the destination is asleep at departure only because the ring (12 031.398 m) is wider than the
//!   visibility wake radius (11 458.475 m) — the 572.924 m margin the world's own solver produced;
//! * the WAKE budget is the demand cadence + the reconcile tick + the cluster's OWN measured boot
//!   latency + the woken shard's first statement beat + the Q2 relay hop + the compose/draw ticks.
//!   The owner ruled that the relay hop is measured HERE: its share is asserted APART, so a relay
//!   too slow to serve a wake fails this gate by name rather than hiding inside a boot number;
//! * the DEPARTURE budget is the AoI grace, the verdict cadence and the same delivery ticks — the
//!   one law read backwards — with the gateway's roster-loss window as its outer bound.
//!
//! BUDGETS ARE COUNTED IN TICKS AND REPORTED IN METRES. Every stage above is a tick count off a
//! landed expression; the owner-facing "footprint" is that tick count times the occupant's own
//! travel per tick. Measuring in ticks is what makes the measurement honest whether the ship is
//! flying or parked when the handover lands.
//!
//! THE CAMERA IS THE PILOT'S (`--capture-pilot`): the default capture framing fits the whole drawn
//! scene, and on a flight between two FIXED points of the ring that framing barely moves, so a
//! system you fly toward could not grow on screen at all. The warp is a statement about what the
//! pilot sees, so the agent's eyes look through the pilot's eyes — reconstructed by the gate from
//! the delivered pose through the very same Tier-A expression the renderer places its camera by.
//!
//! Dot detection is DevState-driven with LOCAL pixel probes at the projected position (§2.11) —
//! never a full-frame search at a 3-pixel footprint. Every wait is on the demand loop's own signal
//! (a gauge, a label, an author flip), never a sleep literal standing in for one.
//!
//! GPU-required + LOCAL, exactly like the other render gates.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::flight::cross_leg;
use vd_bins::pixel::{Author, Presence, Straddled, Subject, own_pose};
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, admin_get_body, common_env,
    dev_auth_pubkey_hex, dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows,
    orchestrator_env, reap_forked, reserve_tcp_addr, reserve_udp_addr, world_roster,
};
use vd_client_harness::assert::magenta_pixel_count;
use vd_client_harness::camera::{
    CaptureCamera, DOT_MIN_APPARENT_RADIUS_PX, FIT_FOV_Y, ScreenAabb, ScreenPos,
};
use vd_client_harness::manifest::{MANIFEST_FILENAME, RunManifest};
use vd_client_harness::verdict::dot_pixels_distinct_from_surround;
use vd_client_render::{CAPTURE_H, CAPTURE_W};
use vd_core::NodeId;
use vd_core::glam::DVec3;
use vd_core::pose::{RealmId, frame_for_realm};
use vd_devproto::{CLIENT_NODE_BASE, DevPhase, DevRequest, DevResponse, DevState};
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::{AdminSnapshot, GatewayView, RlmView};

// ---------------------------------------------------------------------------------------------
// THE DERIVED BUDGET — every term an expression over THE world's numbers or a landed cadence.
// ---------------------------------------------------------------------------------------------

/// The RLM reconciler's own interval (`RlmTuning::cloud_with_boot`: one tick).
const RECONCILE_TICKS: u64 = 1;
/// The Q2 relay hop in shard ticks: ONE at the woken child to author its sealed statements, ONE at
/// the parent to forward them (the parent forwards in the tick it receives — `process_inbound` runs
/// before `emit_realm_frames` on the same schedule, so there is no store-and-forward tick).
const RELAY_FORWARD_TICKS: u64 = 2;
/// The membership verdict's shard→gateway hop, in ticks.
const MEMBERSHIP_HOP_TICKS: u64 = 1;
/// The gateway folds and ships on its next tick.
const COMPOSE_TICKS: u64 = 1;
/// The client applies the composed level and the renderer draws it.
const DRAW_TICKS: u64 = 1;

/// THE world, exactly as every shard of the cluster boots it — the one place this gate reads a
/// world number from (SL5: one world, no preset, no test variant).
fn world() -> vd_physics::worldgen::UniverseConfig {
    vd_physics::worldgen::UniverseConfig::world(DEV.move_speed, DEV.tick_dt)
}

/// THE VISIBILITY FACTOR — `cot(θ_min/2)`, THE world's one visibility law.
fn visibility_factor() -> f64 {
    world().interest.spin_up_factor
}

/// A star system's own extent (its SOI shell radius) — what its self-look states.
fn system_extent_m() -> f64 {
    world().stellar.system_soi_r_m
}

/// A star system's WAKE radius: its extent times the visibility factor — the distance at which the
/// parent's AoI verdict demands it, and at which its own look may take over the drawing.
fn spin_up_r_m() -> f64 {
    system_extent_m() * visibility_factor()
}

/// A star system's TEAR-DOWN radius: the wake radius plus the velocity lead the band's own
/// derivation adds (`|v_rel| · dt · (K_SAFETY + extra)`), through the one shared constructor.
fn tear_down_r_m() -> f64 {
    let cfg = world();
    vd_core::geometry::AoiConfig::for_velocity_safe(
        cfg.stellar.system_soi_r_m,
        cfg.interest.spin_up_factor,
        cfg.interest.tear_down_factor,
        cfg.interest.occupant_v_max_mps,
        cfg.interest.tick_dt_s,
        cfg.interest.grace_ticks,
        cfg.interest.k_safety_extra,
    )
    .expect("THE world's system band is well-formed")
    .tear_down_r_m()
}

/// The star ring's radius — the distance between the home system and its ring sibling.
fn ring_r_m() -> f64 {
    world().stellar.system_ring_r_m
}

/// The occupant's travel per tick at the shipped speed — the ONE tick↔metre conversion, and the
/// crossing gates' one-tick true-motion bound.
fn metres_per_tick() -> f64 {
    DEV.move_speed * DEV.tick_dt
}

/// The AoI re-check cadence in ticks — the beat the demand fold, the SL7 bit, the membership
/// verdict and the window statements all breathe on (the shard's own expression).
fn aoi_cadence_ticks() -> u64 {
    let hz = (1.0 / DEV.tick_dt).round() as u64;
    (hz / 2).max(1)
}

/// The AoI hysteresis grace in ticks, PLUS the tick it drops on — the fold holds for `grace_ticks`
/// evaluations and releases on the next, so the fence-post is counted here rather than assumed.
fn grace_hold_ticks() -> u64 {
    u64::from(world().interest.grace_ticks) + 1
}

/// The gateway's roster-loss window in ticks (`WindowTuning::look_ttl_ticks` = two keep-alive beats
/// + one), derived off the gateway's own beat exactly as the composer derives it.
fn look_ttl_ticks() -> u64 {
    let beat = (u64::from(DEV.tick_hz) / 2).max(1);
    2 * beat + 1
}

/// THE WAKE BUDGET in ticks (§2.8, symmetric half one): the parent's demand beat, the reconcile
/// tick, the cluster's OWN measured boot latency, the woken shard's first-statement beat, the Q2
/// relay hop, the compose tick and the draw tick — each counted once.
fn wake_budget_ticks(boot_ticks: u64) -> u64 {
    2 * aoi_cadence_ticks()
        + RECONCILE_TICKS
        + boot_ticks
        + RELAY_FORWARD_TICKS
        + COMPOSE_TICKS
        + DRAW_TICKS
}

/// THE Q2 RELAY SHARE of the wake budget — everything after the destination's shard is up. Owner
/// ruling: the relay hop is measured at the wake moment and asserted APART, so a relay too slow to
/// serve a wake fails BY NAME (the D-WINDOW-2 trigger) instead of being absorbed by a boot number.
fn relay_share_budget_ticks() -> u64 {
    wake_budget_ticks(0)
}

/// THE DEPARTURE BUDGET in ticks (§2.8, symmetric half two): the AoI grace the departing realm
/// keeps, the cadence its verdict ships on, and the same delivery/draw ticks.
fn departure_budget_ticks() -> u64 {
    grace_hold_ticks() + aoi_cadence_ticks() + MEMBERSHIP_HOP_TICKS + COMPOSE_TICKS + DRAW_TICKS
}

/// The departure budget's OUTER bound — plus the gateway's roster-loss window, by which a departed
/// realm's stored look must be gone from the composer, not merely un-drawn.
fn departure_budget_full_ticks() -> u64 {
    departure_budget_ticks() + look_ttl_ticks()
}

/// Ticks → the owner-facing footprint in metres: how far the occupant travels while a budget runs.
fn footprint_m(ticks: u64) -> f64 {
    metres_per_tick() * ticks as f64
}

// ---------------------------------------------------------------------------------------------
// Flight geometry — derived parks, never picked distances.
// ---------------------------------------------------------------------------------------------

/// The polar-corridor exit out of the home system — the ONE licensed departure (I-AXIS/I-POLE).
/// Derived: one tear-down band outside the home shell, so the exit clears containment AND the
/// hysteresis band by construction.
fn polar_exit_z_m() -> f64 {
    -(system_extent_m() + (tear_down_r_m() - spin_up_r_m()) + system_extent_m())
}

/// THE ARRIVAL STANDOFF: the closest approach at which the destination's BRACKETED footprint still
/// fits well inside the readback, so the local pixel probe has real background to compare against.
/// `radius_px(d) = extent/d · (h/2)/tan(fov/2)`; requiring `RECT_BRACKET · radius_px ≤ h/4` gives
/// `d ≥ 2 · RECT_BRACKET · extent / tan(fov/2)`.
fn arrival_standoff_m() -> f64 {
    2.0 * RECT_BRACKET * system_extent_m() / (FIT_FOV_Y * 0.5).tan()
}

/// THE DEPARTURE PARK: where the ship waits for the system BEHIND it to hand back to its parent's
/// marker. It must be past the home system's tear-down radius (so the verdict can drop) and
/// outside the destination's own shell (so the point of light behind has empty space around it,
/// not a translucent wall). The two constraints leave the band
/// `(extent, ring − tear_down)`; the park is its MIDPOINT — maximum margin on both sides.
fn departure_park_m() -> f64 {
    f64::midpoint(system_extent_m(), ring_r_m() - tear_down_r_m())
}

/// How much wider than the drawn footprint a probed rectangle is bracketed — the same 2× the
/// crossing gate brackets its dot rect by, absorbing the sub-frame skew between the sampled state
/// and the readback frame.
const RECT_BRACKET: f64 = 2.0;
/// A capture's ring probe width in pixels — the SHARED minimum apparent radius, so a point of
/// light's local background is probed exactly one marker-radius out from its own rect.
const PROBE_RING_PX: f64 = DOT_MIN_APPARENT_RADIUS_PX;
/// A footprint shrink smaller than one pixel is not a shrink anyone could see: one pixel is the
/// readback's own quantum, not a fitted tolerance.
const READBACK_QUANTUM_PX: f64 = 1.0;

/// The login must SPAWN a real shard process before it can converge.
const LOGIN_DEADLINE: Duration = Duration::from_secs(60);
/// A healthy client clears this in a second or two once its home is up — below it is a stall.
const SNAPSHOT_FLOOR: u64 = 5;
/// The whole outbound flight, bounded generously: ~24 s of travel at the shipped speed plus the
/// per-chunk round trips, the parked waits and the captures.
const FLIGHT_DEADLINE: Duration = Duration::from_secs(300);
/// One walk chunk on a WATCHED leg, in ticks: short enough that the handover it brackets is still
/// resolved near the clock's own quantum, long enough that the leg is not one round trip per tick.
const WATCHED_CHUNK_TICKS: u64 = 5;
/// The sampling interval — ONE tick of the shipped clock, so a handover's measured latency is
/// resolved to the universe clock's own quantum rather than to the gate's polling habits.
const SAMPLE_POLL: Duration = Duration::from_millis(20);
/// Full forward on the sticky throttle: the warp IS `move_speed` on held axes, along the facing a
/// `LookAt` already pointed at the destination.
const FULL_AHEAD: [f32; 3] = [1.0, 0.0, 0.0];
/// How far past its own closest approach a watched realm may open again before the flight is
/// declared to have sailed past it — one system extent, so a straight run that ends INSIDE the
/// destination is fine and one that misses is loud.
fn pass_by_slack_m() -> f64 {
    system_extent_m()
}
/// THE RE-AIM TRIGGER, as a WORLD miss rather than a frame fraction: turn back onto the subject
/// once the current heading would pass it by more than its OWN EXTENT at its OWN range. Derived,
/// and it tightens as the range closes — which a frame fraction cannot: half a frame is ~20° of
/// heading error, which is 75 m of miss at ring range but 500 m of miss on the run-in, and a gate
/// that only corrects at 20° sails past what it was flying to (MEASURED 2026-08-16: closest
/// approach 476.7 m against a 349.0 m park). It is twice the aim's own converged tolerance, so a
/// fresh turn always has margin and the loop cannot thrash.
fn reaim_miss_m() -> f64 {
    system_extent_m()
}
/// The throttle released.
const ALL_STOP: [f32; 3] = [0.0, 0.0, 0.0];
/// How many settle polls the aim's converge loop takes before it measures the residual — bounded,
/// and it exits early the moment two consecutive polls report the identical facing.
const AIM_SETTLE_POLLS: u32 = 40;

// ---------------------------------------------------------------------------------------------
// Cluster scaffolding (the demand shape, as the RLM process gates boot it).
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
    let trust = ClusterTrust::generate("vd-warp-pixels").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-warp-{tag}-{}", std::process::id()));
    let trust_dir = base.join("trust");
    std::fs::create_dir_all(&trust_dir).expect("trust dir");
    trust.write_der_dir(&trust_dir).expect("write trust");
    let cwd = base.join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("capture cwd");
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
        cwd,
    }
}

/// SIGKILL + reap every demand-spawned shard the orchestrator forked, on drop (they lead their own
/// process groups, so the `Cluster`'s group-kill never reaches them). Declared BEFORE the cluster
/// so it drops AFTER it — by then the orchestrator has released the launch ledger's lock.
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

/// Spawn ONE real headless CAPTURE client in the PILOT view — the agent's eyes looking through the
/// pilot's eyes.
fn spawn_capture_client(
    f: &Fixture,
    gateway: SocketAddr,
    name: &str,
    agent_index: u32,
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
        &agent_index.to_string(),
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

/// Poll until the client is Active AND receiving AND drawing, or fail LOUD with the gateway's view.
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

/// Wait for the client's dev-control listener (the process is up and serving).
fn await_listener(port: u16, child: &mut Child) {
    let started = Instant::now();
    loop {
        if std::net::TcpStream::connect(vd_bins::loopback(port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "the capture client exited before serving dev-control ({status}) — GPU \
                 precondition: this gate needs a working adapter (WGPU_BACKENDS/WGPU_POWER_PREF)",
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

// ---------------------------------------------------------------------------------------------
// The flight instrument.
// ---------------------------------------------------------------------------------------------

/// One sample of the flight: WHEN (the universe clock), WHERE, and how each watched realm is drawn.
#[derive(Clone, Debug)]
struct Sample {
    tick: u64,
    /// The composed picture's ORIGIN EPOCH at this sample — bumped by the gateway on every chain
    /// change, which is the one moment a window is torn down and rebuilt.
    epoch: u64,
    pos: DVec3,
    travelled_m: f64,
    /// One entry per watched realm, in the order the caller listed them.
    subjects: Vec<Subject>,
}

impl Sample {
    /// The distance from the occupant to the `i`-th watched realm's drawn centre.
    fn distance(&self, i: usize) -> f64 {
        (self.subjects[i].centre_m - self.pos).length()
    }
}

/// A watched author FLIP: the universe tick it was first observed at, the distance it happened at,
/// and the drawn footprint at that moment.
#[derive(Clone, Copy, Debug)]
struct Flip {
    tick: u64,
    distance_m: f64,
    footprint_px: f64,
    travelled_m: f64,
    /// The subject's PROJECTED CENTRE in the frame BEFORE the flip and in the frame OF the flip —
    /// §2.8's "centroid path continuous". THE POSITION AUTHOR NEVER CHANGES across a handover (the
    /// parent's placement row positions the subject before, during and after its spin-up; only the
    /// LOOK payload upgrades), so the drawn centre may move only by the observer's own travel
    /// between the two samples. A jump here would mean the swap moved the thing, not just redrew it.
    centre_before_px: Option<(f64, f64)>,
    centre_at_px: Option<(f64, f64)>,
    /// The universe ticks between the sample BEFORE this one and this one — the measurement's own
    /// resolution, and nothing else's. A gap elsewhere in the leg (a course correction's turn, say)
    /// says nothing about how precisely THIS instant was observed, and folding it in would only
    /// loosen the budget this event is judged against.
    gap_ticks: u64,
}

/// The whole flight's record — what both directions of G-HANDOVER read.
struct Record {
    samples: Vec<Sample>,
    /// Where each watched realm first crossed a named radius: `(tick, travel, the gap preceding
    /// that sample)`.
    crossings: Vec<Option<(u64, f64, u64)>>,
    /// The observed author flips per watched realm.
    flips: Vec<Option<Flip>>,
    /// Every sample at which a watched realm was drawn by NOBODY: `(watch index, tick, epoch,
    /// distance)`. Lawful only while the composer is rebuilding its windows across an epoch bump.
    absences: Vec<(usize, u64, u64, f64)>,
}

/// Sample the composed picture once, through the pilot camera the renderer actually used.
fn sample(devctl: u16, watch: &[RealmId], travelled: &mut f64, last: &mut Option<DVec3>) -> Sample {
    let st = vd_bins::pixel::poll(devctl);
    let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
    let (pos, _) = own_pose(&st).expect("the avatar has a delivered pose");
    if let Some(prev) = *last {
        *travelled += (pos - prev).length();
    }
    *last = Some(pos);
    Sample {
        tick: st.universe_tick.unwrap_or_default(),
        epoch: st.origin.as_ref().map_or(0, |(_, e)| *e),
        pos,
        travelled_m: *travelled,
        subjects: watch
            .iter()
            .map(|r| vd_bins::pixel::subject(&st, &camera, *r))
            .collect(),
    }
}

/// Set the sticky throttle. THE WARP IS A HELD THROTTLE (`vd_bins::lib`: "warp IS `move_speed` on
/// held axes") — full forward along the avatar's own facing, which a `LookAt` has already pointed
/// at the destination. Holding it, rather than walking in bursts, is BOTH the production input
/// path and what lets this gate sample the handover at the universe clock's own resolution: a
/// burst-and-stop flight costs three round trips per sample and could only resolve the handover to
/// tens of ticks, which is coarser than the budget it is measuring.
fn throttle(devctl: u16, axes: [f32; 3]) {
    let reply = dev_roundtrip(devctl, &DevRequest::Move { axes });
    assert!(
        matches!(reply, Ok(DevResponse::Ack)),
        "the throttle {axes:?} was refused: {reply:?}",
    );
}

/// THE AIM TOLERANCE, derived: a straight run down the ring must END INSIDE the destination, so
/// the angular error may not carry the ship past its own shell over the ring's length. Half the
/// system's angular radius at ring range leaves the arrival comfortably inside.
fn aim_tolerance_rad() -> f64 {
    (system_extent_m() / ring_r_m()).atan() * 0.5
}
/// How many turns the converge loop may take before the aim is declared un-gettable.
const AIM_TRIES: u32 = 8;

/// One bounded WALK burst toward `aim`, then the sticky throttle released. `WalkTo` steers by the
/// delivered pose into LOCAL axes, so the ship's heading is independent of where it is looking —
/// which is what lets a leg close on one realm while watching another. The brake is the shared
/// feedback-lag term every cluster gate walks by (`vd_bins::flight`).
fn walk_chunk(devctl: u16, aim: DVec3, ticks: u64) {
    let _ = dev_roundtrip(
        devctl,
        &DevRequest::WalkTo {
            target: aim.to_array(),
            arrive_epsilon: 3.0,
            max_ticks: ticks,
            max_step_m: 4.0 * DEV.move_speed * DEV.tick_dt,
        },
    );
    throttle(devctl, ALL_STOP);
}

/// Turn the avatar — and therefore the pilot's eyes — to face `target`, and PROVE the DELIVERED
/// facing settled within [`aim_tolerance_rad`].
///
/// WHY A CONVERGE LOOP AND NOT ONE CALL. The dev-control look-at loop steers by the DELIVERED
/// orientation, so it stops the moment that lagging pose reads aligned — and whatever it has
/// already put on the wire still lands afterwards. `vd_client_harness::nav`'s brake makes that
/// residual small (measured on THE world: 0.6030 rad without it, 0.0353 rad with it), but small is
/// not zero, and 0.0353 rad over the 12 031 m ring is a 425 m miss — which is exactly how this
/// flight first sailed PAST its destination. Re-driving from the settled pose divides the residual
/// again each time, so the aim converges instead of freezing wherever the lag left it.
fn look_at(devctl: u16, target: DVec3) {
    let tol = aim_tolerance_rad();
    for attempt in 0..AIM_TRIES {
        let reply = dev_roundtrip(
            devctl,
            &DevRequest::LookAt {
                target: target.to_array(),
                align_epsilon: tol,
                max_ticks: 900,
            },
        );
        assert!(
            matches!(reply, Ok(DevResponse::State { .. })),
            "the avatar never finished turning toward {target:?}: {reply:?}",
        );
        // Let the in-flight deltas land: poll until the delivered facing holds still, then measure
        // the TRUE 3-D error against the target. A settle signal, not a sleep standing in for one.
        let mut last: Option<DVec3> = None;
        let mut error = f64::INFINITY;
        for _ in 0..AIM_SETTLE_POLLS {
            std::thread::sleep(SAMPLE_POLL);
            let st = vd_bins::pixel::poll(devctl);
            let (pos, orient) = own_pose(&st).expect("a delivered pose to aim from");
            let facing = orient * DVec3::NEG_Z;
            error = facing.angle_between((target - pos).normalize());
            if last.is_some_and(|f: DVec3| f.abs_diff_eq(facing, 1e-12)) {
                break;
            }
            last = Some(facing);
        }
        if error <= tol {
            eprintln!(
                "[warp] AIM: settled {error:.5} rad off {target:?} after {} turn(s) \
                 (tolerance {tol:.5} rad = a {:.1} m miss over the {:.0} m ring)",
                attempt + 1,
                ring_r_m() * tol.tan(),
                ring_r_m(),
            );
            return;
        }
    }
    panic!(
        "the aim never converged on {target:?} within {tol:.5} rad after {AIM_TRIES} turns — \
         the look-at loop is not reducing its error",
    );
}

/// HOW A LEG IS FLOWN. A held throttle drives the ship along its own FACING — which is the warp's
/// real input path and the only way to sample at the clock's own resolution — but it ties where the
/// ship goes to where it looks. A leg that must WATCH one realm while CLOSING on another therefore
/// walks instead: `WalkTo` steers by the delivered pose into local axes, so the heading is
/// independent of the facing, at the cost of one round trip per chunk.
#[derive(Clone, Copy, Debug)]
enum Drive {
    /// Hold these axes for the whole leg (full ahead, or all stop).
    Throttle([f32; 3]),
    /// Walk toward the `toward`-th watched realm's own drawn centre in bounded chunks, and coast
    /// (poll only) once inside `stop_within` of it — so the ship parks itself where the leg wants
    /// it while the eyes stay wherever they were pointed.
    Walk {
        toward: usize,
        chunk_ticks: u64,
        stop_within: f64,
    },
}

/// Drive a leg and record the composed picture at the clock's own resolution until `stop` says so,
/// then release the throttle. `radii` names, per watched realm, the radius whose crossing is
/// timed; `flip_to` names the author each watched realm is expected to hand over to.
///
/// THE PRESENCE LAW is asserted on EVERY sample of the whole flight, not only the captured ones.
#[allow(clippy::too_many_arguments)]
fn fly_recording(
    devctl: u16,
    watch: &[RealmId],
    radii: &[Option<(f64, bool)>], // (radius, true = crossing is "beyond", false = "within")
    flip_to: &[Option<Author>],
    // The index in `watch` the flight is aimed at — the pass-by guard's subject.
    toward: usize,
    // The index in `watch` the EYES are on — the re-aim's subject. Equal to `toward` on a leg flown
    // forward; the realm being left behind on a leg flown astern.
    facing: usize,
    drive: Drive,
    deadline: Duration,
    mut stop: impl FnMut(&Sample, &[Option<Flip>]) -> bool,
) -> Record {
    if let Drive::Throttle(axes) = drive {
        throttle(devctl, axes);
    }
    let started = Instant::now();
    let mut travelled = 0.0_f64;
    let mut last: Option<DVec3> = None;
    let mut samples: Vec<Sample> = Vec::new();
    let mut crossings: Vec<Option<(u64, f64, u64)>> = vec![None; watch.len()];
    let mut flips: Vec<Option<Flip>> = vec![None; watch.len()];
    let mut prev_author: Vec<Option<Author>> = vec![None; watch.len()];
    let mut absences: Vec<(usize, u64, u64, f64)> = Vec::new();
    let mut closest = f64::INFINITY;
    loop {
        let s = sample(devctl, watch, &mut travelled, &mut last);
        // The gap preceding THIS sample — the resolution of anything observed at it.
        let gap = samples
            .last()
            .map_or(0, |prev| s.tick.saturating_sub(prev.tick));
        let mut chasing = false;
        for i in 0..watch.len() {
            // THE PRESENCE LAW, per realm per sample. An ABSENCE is recorded rather than fatal
            // here: it is lawful ONLY while the composer is rebuilding its windows across an
            // origin-epoch bump (a missing statement means the thing is not drawn — THE DRAW LAW),
            // and the caller asserts that bound over the whole flight.
            let author = match s.subjects[i].presence {
                Presence::Drawn(a) => a,
                Presence::Absent => {
                    absences.push((i, s.tick, s.epoch, s.distance(i)));
                    prev_author[i] = None;
                    continue;
                }
            };
            if let Some((radius, beyond)) = radii[i] {
                let d = s.distance(i);
                let crossed = if beyond { d > radius } else { d <= radius };
                if crossed && crossings[i].is_none() {
                    crossings[i] = Some((s.tick, s.travelled_m, gap));
                }
                if crossings[i].is_some() && flips[i].is_none() {
                    chasing = true;
                }
            }
            if let Some(want) = flip_to[i]
                && prev_author[i].is_some_and(|p| p != want)
                && author == want
                && flips[i].is_none()
            {
                flips[i] = Some(Flip {
                    tick: s.tick,
                    distance_m: s.distance(i),
                    footprint_px: s.subjects[i].radius_px,
                    travelled_m: s.travelled_m,
                    centre_before_px: samples.last().and_then(|p| p.subjects[i].centre_px()),
                    centre_at_px: s.subjects[i].centre_px(),
                    gap_ticks: gap,
                });
            }
            prev_author[i] = Some(author);
        }
        // THE PASS-BY GUARD, on the realm the flight is AIMED at (the one it flies toward; the one
        // it leaves behind opens by design): a flight that sails past its destination must say so,
        // loudly and with the miss distance, instead of running to the deadline out in the dark.
        let d = s.distance(toward);
        closest = closest.min(d);
        assert!(
            d <= closest + pass_by_slack_m(),
            "the flight SAILED PAST {:?}: closest approach {closest:.1} m, now {d:.1} m and \
             opening — the aim missed by more than one system extent",
            watch[toward],
        );
        // RE-AIM — a pilot corrects course. A single fixed heading over a 12 km run cannot hold a
        // subject centred: the aim's own residual becomes a growing angle as the range closes, and
        // a subject that drifts to the frame edge is measured through a badly off-axis projection.
        // Never while CHASING a flip: the handover's centroid continuity is measured across two
        // bracketing samples, and turning between them would be the GATE moving the picture rather
        // than the world.
        let drifted = s.subjects[facing].centre_px().is_none_or(|(x, y)| {
            let (cx, cy) = (CAPTURE_W as f64 * 0.5, CAPTURE_H as f64 * 0.5);
            let off_px = (x - cx).hypot(y - cy);
            let range = s.distance(facing);
            let m_per_px = 2.0 * range * (FIT_FOV_Y * 0.5).tan() / CAPTURE_H as f64;
            off_px * m_per_px > reaim_miss_m()
        });
        let reaim = (drifted & !chasing).then_some(s.subjects[facing].centre_m);
        let done = stop(&s, &flips);
        samples.push(s);
        if done {
            throttle(devctl, ALL_STOP);
            let _ = drive;
            return Record {
                samples,
                crossings,
                flips,
                absences,
            };
        }
        assert!(
            started.elapsed() < deadline,
            "the flight never reached its stop condition (travelled {travelled:.1} m, crossings \
             {crossings:?}, flips seen {:?}, chasing {chasing})",
            flips.iter().map(Option::is_some).collect::<Vec<_>>(),
        );
        if let Some(target) = reaim {
            look_at(devctl, target);
        }
        match drive {
            Drive::Throttle(_) => std::thread::sleep(SAMPLE_POLL),
            Drive::Walk {
                toward: t,
                chunk_ticks,
                stop_within,
            } => {
                let sample = samples.last().expect("a sample was just pushed");
                if sample.distance(t) > stop_within {
                    walk_chunk(devctl, sample.subjects[t].centre_m, chunk_ticks);
                } else {
                    std::thread::sleep(SAMPLE_POLL);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pixel verdicts over a straddled capture.
// ---------------------------------------------------------------------------------------------

fn decode(cwd: &std::path::Path, rel: &str) -> (Vec<u8>, usize, usize, [u8; 4]) {
    let img = image::open(cwd.join(rel))
        .unwrap_or_else(|e| panic!("open capture {rel}: {e}"))
        .to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let buf = img.into_raw();
    // The clear colour is self-calibrated from the top-right corner — deep space, and the one
    // corner the HUD never reaches.
    let i = (w - 1) * 4;
    let clear = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
    (buf, w, h, clear)
}

/// Inflate a projected rectangle by the shared bracketing factor about its own centre, never below
/// the shared minimum apparent radius (a point of light's rect is its footprint, not a point).
fn bracket(rect: ScreenAabb) -> ScreenAabb {
    let cx = f64::midpoint(rect.min.x, rect.max.x);
    let cy = f64::midpoint(rect.min.y, rect.max.y);
    let hw = ((rect.max.x - rect.min.x) * 0.5).max(DOT_MIN_APPARENT_RADIUS_PX) * RECT_BRACKET;
    let hh = ((rect.max.y - rect.min.y) * 0.5).max(DOT_MIN_APPARENT_RADIUS_PX) * RECT_BRACKET;
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

/// THE LOCAL PIXEL PROBE (§2.11): the watched realm actually PAINTED at the position the composed
/// picture put it, and those pixels are distinguishable from their own local background — never a
/// full-frame search. Returns the painted-pixel count for the run log.
fn assert_painted(cap: &Straddled, cwd: &std::path::Path, subject: &Subject, what: &str) -> u64 {
    let (rgba, w, h, clear) = decode(cwd, &cap.shot);
    assert_eq!(
        magenta_pixel_count(&rgba),
        0,
        "{what}: a magenta (missing-asset) pixel in the capture",
    );
    let rect = bracket(
        subject
            .rect
            .unwrap_or_else(|| panic!("{what}: the subject must project in front of the eye")),
    );
    let painted = nonclear_in(&rgba, w, h, clear, rect);
    assert!(
        painted > 0,
        "{what}: NOTHING was painted at the composed position (rect {rect:?}, footprint \
         {:.2} px, author {:?})",
        subject.radius_px,
        subject.presence,
    );
    assert!(
        dot_pixels_distinct_from_surround(&rgba, w, h, rect, PROBE_RING_PX),
        "{what}: the pixels at the composed position are indistinguishable from the local \
         background — nothing legible was drawn there",
    );
    painted
}

/// §2.8's CENTROID PATH CONTINUOUS, at the handover itself: the drawn centre across the author
/// flip may move only by the observer's OWN travel between the two bracketing samples, projected.
/// One position author for life — the parent's placement row places the subject before, during and
/// after its spin-up, and only the LOOK payload upgrades — so the swap redraws the thing where it
/// already was. Any larger move means the handover MOVED it, which is the jump §2.8 forbids.
fn assert_centroid_continuous(flip: &Flip, camera: &CaptureCamera, gap_ticks: u64, what: &str) {
    let (Some((ax, ay)), Some((bx, by))) = (flip.centre_before_px, flip.centre_at_px) else {
        panic!(
            "{what}: the subject must project in front of the eye on both sides of the handover"
        );
    };
    let moved_px = (bx - ax).hypot(by - ay);
    // The observer's own travel over the bracketing gap, projected at the subject's range — the
    // one-tick true-motion bound the crossing gates use, scaled to the samples actually taken.
    let travel_m = metres_per_tick() * (gap_ticks + 1) as f64;
    let m_per_px = 2.0 * flip.distance_m * (camera.fov_y * 0.5).tan() / camera.height as f64;
    let allowance_px = travel_m / m_per_px;
    eprintln!(
        "[warp] CENTROID ({what}): the drawn centre moved {moved_px:.2} px across the handover \
         (from {ax:.1},{ay:.1} to {bx:.1},{by:.1}) against the observer's own {travel_m:.1} m of \
         travel over {} sample tick(s) = {allowance_px:.2} px at {:.0} m",
        gap_ticks + 1,
        flip.distance_m,
    );
    assert!(
        moved_px <= allowance_px,
        "{what}: THE CENTROID JUMPED {moved_px:.2} px across the handover, past the          {allowance_px:.2} px the observer's own travel allows — the position author changed at          the swap, which it must never do",
    );
}

/// THE PRESENCE LAW over a whole leg: a watched realm may be drawn by NOBODY only while the
/// composer is rebuilding its windows across an ORIGIN-EPOCH BUMP — a chain change tears the old
/// windows down and the new ones must be re-served their statements before anything can be drawn
/// from them. THE DRAW LAW's own sentence covers that gap ("a missing statement means the thing is
/// not drawn"); what this asserts is that it is BOUNDED and that it happens nowhere else.
///
/// The bound is derived: the gateway's keep-alive beat (the slowest re-assert of a send-on-change
/// statement) plus the delivery and draw ticks.
fn assert_absences_only_bridge_an_epoch_bump(rec: &Record, watch: &[RealmId], leg: &str) {
    let bound =
        (u64::from(DEV.tick_hz) / 2).max(1) + MEMBERSHIP_HOP_TICKS + COMPOSE_TICKS + DRAW_TICKS;
    if rec.absences.is_empty() {
        eprintln!("[warp] PRESENCE ({leg}): every sample drew every watched realm — no gap at all");
        return;
    }
    // Group each absence against the tick its epoch was first seen at.
    let epoch_start = |epoch: u64| -> u64 {
        rec.samples
            .iter()
            .find(|s| s.epoch == epoch)
            .map_or(0, |s| s.tick)
    };
    let mut worst = 0u64;
    for &(i, tick, epoch, dist) in &rec.absences {
        let since = tick.saturating_sub(epoch_start(epoch));
        worst = worst.max(since);
        assert!(
            since <= bound,
            "{leg}: THE PRESENCE LAW broken — {:?} was drawn by NOBODY at tick {tick} ({dist:.1} m \
             out), {since} ticks into epoch {epoch}, past the derived window-rebuild bound of \
             {bound} ticks. Neither its own look nor its parent's marker reached the picture.",
            watch[i],
        );
    }
    eprintln!(
        "[warp] PRESENCE ({leg}): {} samples with a watched realm undrawn, all inside the \
         window-rebuild bound (worst {worst} ticks into an epoch, bound {bound})",
        rec.absences.len(),
    );
}

/// THE PRESENCE LAW, per watched realm per frame: exactly one of {marker, body} — never zero.
/// (Never BOTH is unrepresentable: a composed row carries one bag, and the wire types cannot
/// express a third pixel source.)
fn assert_presence(subject: &Subject, what: &str) -> Author {
    match subject.presence {
        Presence::Drawn(author) => author,
        Presence::Absent => panic!(
            "{what}: THE PRESENCE LAW broken — the realm is drawn by NOBODY (neither its own look \
             nor its parent's marker reached the picture)",
        ),
    }
}

/// G-SHEAR, the PIXEL half: every drawn row of one captured frame was composed from a moment the
/// composer could lawfully hold — none newer than the fold, none older than the declared retention
/// (a held stratum and a relayed interior carry their own stamps BY DESIGN, and say so per row).
fn assert_same_tick_composition(st: &DevState, what: &str) -> Vec<u64> {
    let mut ticks: Vec<u64> = st
        .realm_boxes
        .iter()
        .filter_map(|b| b.newest_tick)
        .collect();
    ticks.sort_unstable();
    ticks.dedup();
    if let (Some(&lo), Some(&hi)) = (ticks.first(), ticks.last()) {
        let span = hi - lo;
        let retention = look_ttl_ticks();
        assert!(
            span <= retention,
            "{what}: G-SHEAR — the drawn rows span {span} ticks, past the composer's declared \
             retention of {retention} (stamps {ticks:?}); a mixed-age picture reached the pixels",
        );
    }
    ticks
}

// ---------------------------------------------------------------------------------------------
// THE GATES
// ---------------------------------------------------------------------------------------------

#[test]
fn g_handover_the_symmetric_budgets_are_derived_and_fit_the_worlds_own_geometry() {
    // The SYMMETRIC half of G-HANDOVER, closed before any process runs: both budgets are
    // derivations over THE world's solved numbers and the landed cadences, and each must FIT the
    // distance the geometry itself provides for it. A world-numbers change that breaks the warp
    // acceptance fails here in milliseconds, not only in a four-minute flight.
    let (ring, wake, tear, extent) = (
        ring_r_m(),
        spin_up_r_m(),
        tear_down_r_m(),
        system_extent_m(),
    );
    let margin = ring - wake;
    let band = tear - wake;
    eprintln!(
        "[handover] DERIVED on THE world (seed {}, {:.0} m/s, {:.3} s/tick, {:.1} m/tick):\n  \
         visibility factor cot(θ/2) = {:.9}\n  \
         system extent {extent:.3} m · wake radius {wake:.6} m · tear-down radius {tear:.6} m\n  \
         star ring {ring:.6} m · asleep-at-departure margin {margin:.6} m · tear-down band {band:.3} m\n  \
         cadences (ticks): AoI {} · grace-hold {} · reconcile {RECONCILE_TICKS} · relay {RELAY_FORWARD_TICKS} \
         · membership hop {MEMBERSHIP_HOP_TICKS} · compose {COMPOSE_TICKS} · draw {DRAW_TICKS} · \
         roster-loss window {}\n  \
         WAKE budget at zero boot = {} ticks = {:.1} m (+ {:.1} m per boot tick); \
         Q2 RELAY SHARE = {} ticks = {:.1} m\n  \
         DEPARTURE budget = {} ticks = {:.1} m; with the roster-loss window = {} ticks = {:.1} m\n  \
         flight parks: polar exit z {:.1} m · arrival standoff {:.1} m · departure park {:.1} m",
        DEV.universe_seed,
        DEV.move_speed,
        DEV.tick_dt,
        metres_per_tick(),
        visibility_factor(),
        aoi_cadence_ticks(),
        grace_hold_ticks(),
        look_ttl_ticks(),
        wake_budget_ticks(0),
        footprint_m(wake_budget_ticks(0)),
        metres_per_tick(),
        relay_share_budget_ticks(),
        footprint_m(relay_share_budget_ticks()),
        departure_budget_ticks(),
        footprint_m(departure_budget_ticks()),
        departure_budget_full_ticks(),
        footprint_m(departure_budget_full_ticks()),
        polar_exit_z_m(),
        arrival_standoff_m(),
        departure_park_m(),
    );
    // THE geometric precondition the whole acceptance rests on: a destination is genuinely asleep
    // when you leave, because the world's own solver left more ring than wake radius.
    assert!(
        margin > 0.0,
        "THE world no longer leaves a system asleep at departure (ring {ring:.3} m ≤ wake {wake:.3} m)",
    );
    // SYMMETRY: neither budget may exceed the distance the geometry provides for it. The wake must
    // complete inside the wake radius (or a system is entered before it draws itself); the whole
    // departure — including the roster-loss window — must complete inside the ring (or the system
    // you left never becomes a dot before you reach the next one).
    assert!(
        footprint_m(wake_budget_ticks(0)) < wake,
        "the zero-boot wake footprint {:.1} m must fit inside the wake radius {wake:.1} m",
        footprint_m(wake_budget_ticks(0)),
    );
    assert!(
        footprint_m(departure_budget_full_ticks()) < ring,
        "the full departure footprint {:.1} m must fit inside the ring {ring:.1} m",
        footprint_m(departure_budget_full_ticks()),
    );
    // The tear-down band IS the velocity lead: it must cover more than one tick of travel, or the
    // hysteresis could not bracket a crossing at all.
    assert!(
        band > metres_per_tick(),
        "the tear-down band {band:.3} m must exceed one tick of travel {:.3} m",
        metres_per_tick(),
    );
    // The departure park must exist: past the home system's tear-down radius AND outside the
    // destination's own shell — otherwise the point of light behind has a translucent wall around
    // it instead of empty space, and no local pixel probe could be made honest.
    let park = departure_park_m();
    assert!(
        park > extent && park < ring - tear,
        "the departure park {park:.3} m must sit between the destination shell {extent:.3} m and \
         the home tear-down crossing at {:.3} m",
        ring - tear,
    );
}

#[test]
fn g_warp_pixels_a_point_of_light_grows_hands_over_and_the_one_behind_shrinks_to_a_dot() {
    // FIRST statement: hold the process tier for the whole body (it outlives the cluster reap).
    let _tier = vd_bins::cluster_tier();

    let f = fixture("warp");
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
        "g-warp",
        0,
        client_quic.port(),
        devctl,
    ));
    await_listener(devctl, &mut client.0);

    let roster = world_roster(&DEV);
    let home_label = label_of(roster.home);
    let galaxy_label = label_of(roster.galaxy);
    let sibling_label = label_of(roster.sibling);
    let landed = await_active(devctl, gw_admin, LOGIN_DEADLINE);
    assert_eq!(
        landed.location.as_deref(),
        Some(home_label.as_str()),
        "the login lands at the home star: {landed:?}",
    );
    let (wake, tear, extent, ring) = (
        spin_up_r_m(),
        tear_down_r_m(),
        system_extent_m(),
        ring_r_m(),
    );
    eprintln!(
        "[warp] THE world: ring {ring:.3} m · wake {wake:.3} m · tear-down {tear:.3} m · extent \
         {extent:.3} m · asleep-at-departure margin {:.3} m · {:.1} m per tick",
        ring - wake,
        metres_per_tick(),
    );

    // ---- Leg 0: out of the home system, up the licensed polar corridor. ----
    cross_leg(
        devctl,
        "warp-exit home->galaxy (polar corridor)",
        |_tick| DVec3::new(0.0, 0.0, polar_exit_z_m()),
        &galaxy_label,
        Duration::from_secs(120),
    );

    // ---- DEPARTURE: face the destination and prove it is A POINT OF LIGHT. ----
    // Index 0 = the destination (the ring sibling); index 1 = the home system left behind.
    let watch = [roster.sibling, roster.home];
    look_at(devctl, roster.sibling_centre);
    let cap = vd_bins::pixel::straddle(
        devctl,
        "warp-departure",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch,
        vd_bins::pixel::pilot_camera,
    );
    let dest = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.sibling);
    let (here, _) = own_pose(&cap.post).expect("a delivered pose at departure");
    let dist0 = (dest.centre_m - here).length();
    assert!(
        dist0 > wake,
        "PRECONDITION: at departure the destination must be OUTSIDE its own wake radius \
         (measured {dist0:.1} m vs wake {wake:.1} m) — otherwise it was never asleep and the \
         handover this gate measures would be vacuous",
    );
    assert_eq!(
        assert_presence(&dest, "departure/destination"),
        Author::ParentMarker,
        "at departure the destination is its PARENT'S point of light — no server runs for it",
    );
    let painted = assert_painted(&cap, &f.cwd, &dest, "departure/destination");
    let departure_footprint_px = dest.radius_px;
    eprintln!(
        "[warp] DEPARTURE: the destination at {dist0:.1} m is a MARKER — footprint \
         {departure_footprint_px:.2} px, {painted} pixels painted, straddle drift {:.2} px, \
         drawn stamps {:?}",
        cap.drift_px,
        assert_same_tick_composition(&cap.post, "departure"),
    );

    // ---- G-PARENT-TRUE (look_horizon.md slice 0, the process half): in the departure scene —
    // the observer standing in the GALAXY, the home system's planets arriving as RELAYED
    // interior rows — every planet row's parent must be its STAR SYSTEM, never the galaxy.
    // Before slice 0 the composer derived a relayed row's parent from the chain index, which
    // named the GRANDparent (the galaxy) on every relayed row (measured RED at the unit tier).
    {
        // The boxes render realm ids in Debug form ("System(7)"), so the galaxy compares in
        // that same rendering — never the display label ("System 7").
        let galaxy_debug = format!("{:?}", roster.galaxy);
        let planet_parents: Vec<(String, Option<String>)> = cap
            .post
            .realm_boxes
            .iter()
            .filter(|b| b.realm.starts_with("Planet("))
            .map(|b| (b.realm.clone(), b.parent.clone()))
            .collect();
        assert!(
            !planet_parents.is_empty(),
            "G-PARENT-TRUE is vacuous: no planet rows in the departure scene \
             (drawn: {:?})",
            cap.post
                .realm_boxes
                .iter()
                .map(|b| b.realm.clone())
                .collect::<Vec<_>>(),
        );
        for (planet, parent) in &planet_parents {
            let parent = parent
                .as_deref()
                .unwrap_or_else(|| panic!("G-PARENT-TRUE: {planet} delivered with NO parent"));
            assert!(
                parent.starts_with("System(") && parent != galaxy_debug.as_str(),
                "G-PARENT-TRUE: {planet}'s delivered parent is {parent:?} — it must be its \
                 star system, never the galaxy ({galaxy_debug:?})",
            );
        }
        eprintln!(
            "[warp] G-PARENT-TRUE: planet rows parent on their star system: {planet_parents:?}"
        );
    }

    // ---- SLICE 6 (look_horizon.md §5.4/§6): ROWS PER FOLD, pinned on the departure fixture. ----
    // The per-session clone is the census wall and it is LINEAR in rows per fold; the look
    // horizon's §5.4 promise is that the design did NOT change rows per fold (relayed interior
    // rows composed before it — only their tag upgraded from marker to picture). Pinned as the
    // DERIVED set off THE world, not a count: the origin galaxy's one in-band child (the home
    // system, drawn by its own look) + that child's whole interior (its 5 planets, drawn by
    // their own relayed pictures) + one parent-authored marker per out-of-band galaxy child
    // (the ring siblings). Containing realms draw nothing (a containment boundary is never
    // drawn as an object). Growth here is the wall moving — this assert is where it gets loud.
    {
        let the_world = vd_bins::boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt);
        let mut expected: std::collections::BTreeSet<String> = the_world
            .regions()
            .iter()
            .filter(|r| r.parent == Some(vd_core::worldgen::GALAXY))
            .map(|r| format!("{:?}", r.realm))
            .collect();
        expected.extend(
            vd_physics::worldgen::moving_children_for_config(
                DEV.universe_seed,
                &world(),
                roster.home,
            )
            .iter()
            .map(|(p, _)| format!("{p:?}")),
        );
        let actual: std::collections::BTreeSet<String> = cap
            .post
            .realm_boxes
            .iter()
            .map(|b| b.realm.clone())
            .collect();
        assert_eq!(
            actual, expected,
            "SLICE-6 ROWS PER FOLD: the departure fixture's composed set must be exactly the \
             derived one (in-band child + its interior + sibling markers)",
        );
        eprintln!(
            "[warp] SLICE-6 ROWS PER FOLD pinned: {} rows at the departure fixture: {actual:?}",
            actual.len(),
        );
    }

    // ---- THE APPROACH: fly the ring, timing the wake handover on the universe clock. ----
    let standoff = arrival_standoff_m();
    let approach = fly_recording(
        devctl,
        &watch,
        &[Some((wake, false)), None],
        &[Some(Author::SelfLook), None],
        0,
        0,
        Drive::Throttle(FULL_AHEAD),
        FLIGHT_DEADLINE,
        |s, flips| flips[0].is_some() && s.distance(0) <= standoff,
    );
    assert_absences_only_bridge_an_epoch_bump(&approach, &watch, "approach");
    let boot_ticks = orch_rlm(a.admin).boot_ticks_observed_max;

    // ---- G-HANDOVER, the WAKE direction. ----
    let (wake_tick, wake_travel, wake_gap) = approach.crossings[0]
        .expect("the destination crossed its own wake radius during the approach");
    let swap = approach.flips[0]
        .expect("the destination handed over from its parent's marker to its own look");
    let measured = swap.tick.saturating_sub(wake_tick);
    // THE MEASUREMENT'S OWN RESOLUTION: the gap preceding the crossing sample plus the gap
    // preceding the flip sample — the two instants this latency is the difference of, and nothing
    // else in the leg.
    let resolution = wake_gap + swap.gap_ticks;
    let budget = wake_budget_ticks(boot_ticks) + resolution;
    eprintln!(
        "[warp] WAKE HANDOVER: crossed the {wake:.1} m line at tick {wake_tick} ({wake_travel:.1} m \
         travelled); the destination's own look took over at tick {} ({:.1} m out, footprint \
         {:.2} px, {:.1} m travelled). MEASURED {measured} ticks = {:.1} m vs DERIVED budget \
         {} ticks = {:.1} m (2×{} AoI cadence + {RECONCILE_TICKS} reconcile + {boot_ticks} MEASURED \
         boot + {RELAY_FORWARD_TICKS} Q2 relay + {COMPOSE_TICKS} compose + {DRAW_TICKS} draw, \
         + {} ticks bracketing the two observed instants).",
        swap.tick,
        swap.distance_m,
        swap.footprint_px,
        swap.travelled_m,
        footprint_m(measured),
        budget,
        footprint_m(budget),
        aoi_cadence_ticks(),
        resolution,
    );
    assert!(
        measured <= budget,
        "G-HANDOVER (wake): the destination took {measured} ticks ({:.1} m of travel) to start \
         drawing itself, past the derived budget of {budget} ticks ({:.1} m)",
        footprint_m(measured),
        footprint_m(budget),
    );
    // THE Q2 RELAY SHARE, asserted APART (owner ruling).
    let relay_share = measured.saturating_sub(boot_ticks);
    let relay_budget = relay_share_budget_ticks() + resolution;
    eprintln!(
        "[warp] Q2 RELAY SHARE: {relay_share} ticks = {:.1} m of the wake is post-boot (the \
         parent-relay hop + the cadences + compose + draw) vs DERIVED {relay_budget} ticks = {:.1} m",
        footprint_m(relay_share),
        footprint_m(relay_budget),
    );
    assert!(
        relay_share <= relay_budget,
        "G-HANDOVER (wake, THE Q2 RELAY HOP): the parent-relay leg cost {relay_share} ticks \
         ({:.1} m) after the destination's shard was up, past the derived {relay_budget} ticks \
         ({:.1} m). THE RELAY HOP BREAKS THE WAKE BUDGET — this is the ONE measurement D-WINDOW-2 \
         is gated on; the direct live-sibling window is a FRESH owner ask, never taken for \
         convenience. STOP and report.",
        footprint_m(relay_share),
        footprint_m(relay_budget),
    );
    assert_centroid_continuous(&swap, &cap.camera, swap.gap_ticks, "wake/destination");
    // NO POP: the destination must still be near a point when it takes over — its own look starts
    // drawing no closer than the wake radius minus the whole budget it was allowed.
    assert!(
        swap.distance_m >= wake - footprint_m(budget),
        "G-HANDOVER (wake): the destination only started drawing itself at {:.1} m, inside the \
         {:.1} m the derived budget allows — it POPPED in at size",
        swap.distance_m,
        wake - footprint_m(budget),
    );

    // ---- THE GROWTH CURVE: monotone, and strictly bigger at the standoff than at departure. ----
    let curve: Vec<(f64, f64)> = approach
        .samples
        .iter()
        .map(|s| (s.distance(0), s.subjects[0].radius_px))
        .collect();
    let mut worst_shrink = 0.0_f64;
    for pair in curve.windows(2) {
        worst_shrink = worst_shrink.max(pair[0].1 - pair[1].1);
    }
    eprintln!(
        "[warp] GROWTH CURVE: {} samples · {:.2} px at {:.0} m → {:.2} px at {:.0} m · worst \
         sample-to-sample shrink {worst_shrink:.3} px (the readback's own quantum is \
         {READBACK_QUANTUM_PX:.1} px)",
        curve.len(),
        curve[0].1,
        curve[0].0,
        curve[curve.len() - 1].1,
        curve[curve.len() - 1].0,
    );
    assert!(
        worst_shrink < READBACK_QUANTUM_PX,
        "G-WARP-PIXELS: the destination's footprint SHRANK by {worst_shrink:.3} px while the \
         occupant was approaching it — the growth is not monotone",
    );
    assert!(
        curve[curve.len() - 1].1 > curve[0].1,
        "G-WARP-PIXELS: the destination never grew ({:.2} px → {:.2} px)",
        curve[0].1,
        curve[curve.len() - 1].1,
    );

    // ---- THE ARRIVAL PICTURE: the destination draws ITSELF, large, in real pixels. ----
    let cap = vd_bins::pixel::straddle(
        devctl,
        "warp-arrival",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch,
        vd_bins::pixel::pilot_camera,
    );
    let dest = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.sibling);
    assert_eq!(
        assert_presence(&dest, "arrival/destination"),
        Author::SelfLook,
        "on arrival the destination draws ITSELF — a running realm authors its own look",
    );
    let painted = assert_painted(&cap, &f.cwd, &dest, "arrival/destination");
    assert!(
        dest.radius_px > departure_footprint_px,
        "the arrival footprint {:.2} px must exceed the departure point of light \
         {departure_footprint_px:.2} px",
        dest.radius_px,
    );
    eprintln!(
        "[warp] ARRIVAL: destination footprint {:.2} px, {painted} pixels painted, drawn stamps {:?}",
        dest.radius_px,
        assert_same_tick_composition(&cap.post, "arrival"),
    );

    // ---- THE DEPARTURE DIRECTION, in pixels: press on to the derived park, turn, and watch the
    // system behind hand back to its parent's marker. ----
    let park = departure_park_m();
    // TURN AND LOOK BACK FIRST, then run in to the park while WATCHING what is behind. "It shrinks
    // to a dot behind you" is a statement about what you SEE when you look at it, so the whole
    // window — the tear-down crossing AND the handover that follows it — has to be recorded with
    // the eyes already on the system being left. That is why this leg WALKS rather than holding a
    // throttle: walking steers by the delivered pose, so the ship keeps closing on the destination
    // while facing the other way. Aimed at the home system's OWN drawn centre in the frame the
    // picture is composed in now, never a remembered coordinate.
    let home_seen = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.home);
    look_at(devctl, home_seen.centre_m);
    let behind = fly_recording(
        devctl,
        &watch,
        &[None, Some((tear, true))],
        &[None, Some(Author::ParentMarker)],
        0,
        1,
        Drive::Walk {
            toward: 0,
            chunk_ticks: WATCHED_CHUNK_TICKS,
            stop_within: park,
        },
        FLIGHT_DEADLINE,
        |_, flips| flips[1].is_some(),
    );
    assert_absences_only_bridge_an_epoch_bump(&behind, &watch, "departure");
    let (tear_tick, tear_travel, tear_gap) = behind.crossings[1]
        .expect("the home system crossed its own tear-down radius behind the flight");
    let back = behind.flips[1]
        .expect("the home system handed back to its parent's marker behind the flight");
    let measured_back = back.tick.saturating_sub(tear_tick);
    let back_resolution = tear_gap + back.gap_ticks;
    let back_budget = departure_budget_ticks() + back_resolution;
    eprintln!(
        "[warp] DEPARTURE HANDOVER: the home system crossed its {tear:.1} m tear-down radius at \
         tick {tear_tick} ({tear_travel:.1} m travelled) and stopped drawing itself at tick {} \
         ({:.1} m out, footprint {:.2} px). MEASURED {measured_back} ticks = {:.1} m vs DERIVED \
         budget {back_budget} ticks = {:.1} m ({} grace-hold + {} AoI cadence + \
         {MEMBERSHIP_HOP_TICKS} hop + {COMPOSE_TICKS} compose + {DRAW_TICKS} draw, + {} ticks \
         bracketing the two observed instants).",
        back.tick,
        back.distance_m,
        back.footprint_px,
        footprint_m(measured_back),
        footprint_m(back_budget),
        grace_hold_ticks(),
        aoi_cadence_ticks(),
        back_resolution,
    );
    assert!(
        measured_back <= back_budget,
        "G-HANDOVER (departure): the home system kept drawing its own body for {measured_back} \
         ticks ({:.1} m of travel) past its tear-down radius, over the derived {back_budget} ticks \
         ({:.1} m)",
        footprint_m(measured_back),
        footprint_m(back_budget),
    );
    assert_centroid_continuous(&back, &cap.camera, back.gap_ticks, "departure/home");
    // The departure footprint must also fit the ROSTER-LOSS outer bound — a departed realm's stored
    // look is gone from the composer, not merely un-drawn.
    assert!(
        measured_back <= departure_budget_full_ticks(),
        "G-HANDOVER (departure): the handover took {measured_back} ticks, past even the \
         roster-loss outer bound of {} ticks",
        departure_budget_full_ticks(),
    );

    // The eyes are already on the system behind (the leg was flown astern), so this capture
    // needs no turn — only the picture as it stands at the park.
    let cap = vd_bins::pixel::straddle(
        devctl,
        "warp-behind",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch,
        vd_bins::pixel::pilot_camera,
    );
    let home_now = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.home);
    assert_eq!(
        assert_presence(&home_now, "behind/home"),
        Author::ParentMarker,
        "behind the flight the home system is a point of light again — its parent's marker resumed",
    );
    let painted = assert_painted(&cap, &f.cwd, &home_now, "behind/home");
    // SHRUNK TO A POINT OF LIGHT, in pixels — re-derived for look_horizon.md slice 1 (Q2
    // APPROVED): the departed system's marker now carries its parent's ONE stated radius, so it
    // draws at its TRUE angular size — computed here INDEPENDENTLY from THE world's own system
    // extent and the measured distance, through the same shared floor expression the renderer
    // scales by. Before the slice this marker sat at the bare 3 px floor (an ~3.8× size pop
    // against the body it had just been); the marker-vs-own-look distinction is the AUTHOR
    // assert above (DevState body kind), no longer a size reading.
    let dist_behind = (home_now.centre_m - cap.camera.eye).length();
    let expected_world = vd_client_harness::camera::marker_world_radius(
        extent,
        dist_behind,
        cap.camera.fov_y,
        cap.camera.height as f64,
    );
    let expected_px = vd_client_harness::verdict::projected_point_aabb(
        &cap.camera,
        home_now.centre_m,
        expected_world,
    )
    .map_or(0.0, |r| (r.max.x - r.min.x) * 0.5);
    assert!(
        (home_now.radius_px - expected_px).abs() < READBACK_QUANTUM_PX,
        "the system behind must be drawn at its extent's true angular size \
         ({expected_px:.2} px at {dist_behind:.1} m), not at {:.2} px",
        home_now.radius_px,
    );
    assert!(
        expected_px > DOT_MIN_APPARENT_RADIUS_PX + READBACK_QUANTUM_PX,
        "NON-VACUOUS: at this range the extent-sized point must sit ABOVE the bare floor \
         (expected {expected_px:.2} px vs floor {DOT_MIN_APPARENT_RADIUS_PX:.1} px) — the \
         slice-1 growth, not the old constant dot",
    );
    eprintln!(
        "[warp] BEHIND: the home system is a MARKER at footprint {:.2} px, {painted} pixels \
         painted, drawn stamps {:?}",
        home_now.radius_px,
        assert_same_tick_composition(&cap.post, "behind"),
    );

    // ---- NON-VACUOUS: the roster-loss machinery really ran, and the world tore a realm down. ----
    let gw = gateway_view(gw_admin).expect("the gateway serves its admin snapshot");
    let rlm = orch_rlm(a.admin);
    eprintln!(
        "[warp] MACHINERY: window_looks_pruned {} · window_relay_levels_pruned {} · \
         window_relays_ingested {} · spins_requested {} · teardowns_reaped {}",
        gw.window_looks_pruned,
        gw.window_relay_levels_pruned,
        gw.window_relays_ingested,
        rlm.spins_requested,
        rlm.teardowns_reaped,
    );
    assert!(
        gw.window_relays_ingested > 0,
        "the destination's own look reached the composer through the Q2 PARENT RELAY — a zero \
         here means the wake handover was served by something else: {gw:?}",
    );
    assert!(
        rlm.spins_requested > 0,
        "the demand loop spun a realm up during this flight: {rlm:?}",
    );

    // ---- THE JOURNEY COMPLETES: fly in and stand inside the destination. ----
    // The destination's OWN drawn centre in the frame the picture is composed in right now — the
    // journey's last leg is aimed at what the picture says, never at a remembered coordinate. Flown
    // on the SAME held throttle the whole warp was flown on (turn, then hold), because that is both
    // the production input path and the one this flight has already proven itself on.
    let arrive_at = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.sibling).centre_m;
    look_at(devctl, arrive_at);
    fly_until_label(
        devctl,
        &sibling_label,
        roster.sibling,
        Duration::from_secs(120),
    );

    // ---- HR6: every capture, and every drawn row's provenance, is in the run manifest. ----
    assert_manifest_attests(&f.cwd, &["warp-departure", "warp-arrival", "warp-behind"]);

    // The client holds the GPU — tear it down before the cluster's own drop.
    drop(client);
}

/// The design's stated close range for G-LOOK-GROWTH (look_horizon.md slice 1's pixel gate:
/// "as the camera closes from 11 km to 200 m") — checked at runtime against the world's own
/// geometry so the whole approach provably stays OUTSIDE the system shell (the planet remains a
/// parent-authored point of light for every sample; waking it is slice 4's work, not this
/// gate's).
const GROWTH_END_RANGE_M: f64 = 200.0;

/// G-LOOK-GROWTH — look_horizon.md slice 1's PIXEL GATE (Q2 APPROVED 2026-08-17): a planet's
/// POINT OF LIGHT grows monotonically — and STRICTLY once above the shared apparent floor — as
/// the camera closes from ~11.4 km (just inside the home system's wake band) down to 200 m of
/// the inner planet, and it NEVER pops and NEVER blanks. The whole leg is flown outside the
/// 150 m shell, so every sample is the parent's marker: the growth measured is the point of
/// light's OWN — the extent-sized marker that replaced the constant three-pixel dot (before the
/// slice this curve was FLAT at the floor from 11 km all the way to the wake handover, then
/// popped ~3.8× into the body's true size).
///
/// This is also the STATION-LAPSE law in pixels for the planet's own lapse: the planets' shards
/// died when the occupant left the system (their looks pruned on the roster-loss window), and
/// the drawn set NEVER blanked — each degraded to its parent's correctly-sized marker.
#[test]
fn g_look_growth_a_planets_point_of_light_grows_strictly_on_approach() {
    let _tier = vd_bins::cluster_tier();
    let f = fixture("g-growth");
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
        "g-growth",
        0,
        client_quic.port(),
        devctl,
    ));
    await_listener(devctl, &mut client.0);

    let roster = world_roster(&DEV);
    let home_label = label_of(roster.home);
    let galaxy_label = label_of(roster.galaxy);
    let landed = await_active(devctl, gw_admin, LOGIN_DEADLINE);
    assert_eq!(
        landed.location.as_deref(),
        Some(home_label.as_str()),
        "the login lands at the home star: {landed:?}",
    );
    // The planet's TRUE circumscribed extent, off the SAME boot the shard runs (SL5 — one world,
    // one derivation): what the parent's one-radius marker states, and what the growth curve is
    // judged against.
    let inner_extent_m = vd_bins::boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt)
        .regions()
        .iter()
        .find(|r| r.realm == roster.inner)
        .expect("THE world rosters the inner planet")
        .shape
        .circumscribed_extent();
    // The approach flies the LICENSED POLAR CORRIDOR (I-AXIS), which makes the close range
    // PHASE-FREE: coming down the ±Z axis, the range to an in-plane orbiter is
    // `sqrt(z² + r_orbit²)` — nearly independent of where the planet is on its orbit — so
    // stopping at 200 m from the planet leaves the observer `sqrt(200² − apoapsis²)` up the
    // axis, provably outside the shell at EVERY orbital phase (no far-side race exists on the
    // axis). Asserted against the world's own numbers; an in-flight centre guard backs it.
    let inner_apoapsis_m = roster.inner_elements.sma * (1.0 + roster.inner_elements.ecc);
    let polar_close_axis_m =
        (GROWTH_END_RANGE_M * GROWTH_END_RANGE_M - inner_apoapsis_m * inner_apoapsis_m).sqrt();
    assert!(
        polar_close_axis_m > system_extent_m() + 2.0 * metres_per_tick(),
        "the polar close range must clear the {:.1} m shell at every phase \
         (axis distance {polar_close_axis_m:.1} m at apoapsis {inner_apoapsis_m:.1} m)",
        system_extent_m(),
    );
    let centre_guard_m = system_extent_m() + 2.0 * metres_per_tick();

    // ---- Leg 0: out of the home system, up the licensed polar corridor. ----
    cross_leg(
        devctl,
        "growth-exit home->galaxy (polar corridor)",
        |_tick| DVec3::new(0.0, 0.0, polar_exit_z_m()),
        &galaxy_label,
        Duration::from_secs(120),
    );

    // ---- Leg 1: fly polar OUT to the growth start — just inside the wake band, so the home
    // system stays demanded (its relays keep flowing) while its interior stays asleep. ----
    let band_m = tear_down_r_m() - spin_up_r_m();
    let growth_start_m = spin_up_r_m() - 2.0 * band_m;
    let started = Instant::now();
    {
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        let home = vd_bins::pixel::subject(&st, &camera, roster.home);
        // Straight down the licensed −Z polar corridor from the home centre (I-POLE/I-AXIS).
        look_at(
            devctl,
            home.centre_m + DVec3::new(0.0, 0.0, -growth_start_m),
        );
    }
    throttle(devctl, FULL_AHEAD);
    loop {
        std::thread::sleep(SAMPLE_POLL);
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        let home = vd_bins::pixel::subject(&st, &camera, roster.home);
        let (pos, _) = own_pose(&st).expect("a delivered pose on the out-leg");
        if (home.centre_m - pos).length() >= growth_start_m {
            throttle(devctl, ALL_STOP);
            break;
        }
        assert!(
            started.elapsed() < FLIGHT_DEADLINE,
            "the out-leg never reached the growth start ({growth_start_m:.1} m)",
        );
    }

    // ---- THE START: the planet is a POINT OF LIGHT at the shared floor — present, painted,
    // parent-authored (its own look lapsed when the system emptied), never a blank. ----
    let watch = [roster.inner, roster.home];
    {
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        let inner = vd_bins::pixel::subject(&st, &camera, roster.inner);
        look_at(devctl, inner.centre_m);
    }
    let cap0 = vd_bins::pixel::straddle(
        devctl,
        "growth-start",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch,
        vd_bins::pixel::pilot_camera,
    );
    let start = vd_bins::pixel::subject(&cap0.post, &cap0.camera, roster.inner);
    let (pos0, _) = own_pose(&cap0.post).expect("a delivered pose at the growth start");
    let dist0 = (start.centre_m - pos0).length();
    assert_eq!(
        assert_presence(&start, "growth-start/inner"),
        Author::ParentMarker,
        "at the start the planet's shard is down (the system holds nobody) — its lapsed look \
         degraded to the parent's marker, never to nothing",
    );
    assert!(
        (start.radius_px - DOT_MIN_APPARENT_RADIUS_PX).abs() < READBACK_QUANTUM_PX,
        "at {dist0:.1} m the planet's point sits at the shared floor \
         ({DOT_MIN_APPARENT_RADIUS_PX:.1} px), measured {:.2} px",
        start.radius_px,
    );
    let painted0 = assert_painted(&cap0, &f.cwd, &start, "growth-start/inner");
    eprintln!(
        "[growth] START: the inner planet at {dist0:.1} m is a MARKER at {:.2} px \
         ({painted0} pixels painted); extent {inner_extent_m:.4} m; floor-crossing range \
         {:.1} m",
        start.radius_px,
        inner_extent_m * (cap0.camera.height as f64 * 0.5)
            / ((FIT_FOV_Y * 0.5).tan() * DOT_MIN_APPARENT_RADIUS_PX),
    );

    // ---- THE APPROACH: close from ~11.4 km to 200 m, recording every sample. The stop is the
    // planet range; the centre guard is the loud backstop the polar geometry makes unreachable
    // (a trip into the shell would re-home the observer and void every marker premise). ----
    let record = fly_recording(
        devctl,
        &watch,
        &[None, None],
        &[None, None],
        0,
        0,
        Drive::Throttle(FULL_AHEAD),
        FLIGHT_DEADLINE,
        |s, _| {
            assert!(
                s.distance(1) > centre_guard_m,
                "the approach GRAZED the shell (centre range {:.1} m ≤ guard {centre_guard_m:.1} \
                 m) — the polar corridor failed to keep the close range outside containment",
                s.distance(1),
            );
            s.distance(0) <= GROWTH_END_RANGE_M
        },
    );
    // NEVER BLANKS: the planet was drawn at EVERY sample of the approach — no epoch bump
    // happened (no crossing), so not even the bridging allowance applies.
    let inner_absences: Vec<_> = record.absences.iter().filter(|(i, ..)| *i == 0).collect();
    assert!(
        inner_absences.is_empty(),
        "the planet's point of light BLANKED mid-approach: {inner_absences:?}",
    );
    // MONOTONE, and STRICT — at the measurement's own resolution. The measured footprint carries
    // a small off-axis projection term (the shared floor expression sizes the world radius by the
    // EUCLIDEAN eye distance while the projection divides by view depth — ~0.7 % at a few degrees
    // off-axis, measured 3.0218 px at the floor on the first red run) plus the planet's own
    // orbital swing, both bounded well inside the readback quantum. So: the footprint may never
    // fall more than one quantum below its running maximum (never pops, never shrinks), and
    // between any two samples whose MODEL growth — the extent's pure angular size at the two
    // measured distances — exceeds one quantum, the measured footprint must STRICTLY rise.
    let mut curve: Vec<(u64, f64, f64)> = Vec::new(); // (tick, distance, radius_px)
    for s in &record.samples {
        if curve.last().is_none_or(|(t, ..)| *t != s.tick) {
            curve.push((s.tick, s.distance(0), s.subjects[0].radius_px));
        }
    }
    assert!(
        curve.len() >= 32,
        "the approach must be a real curve, not a handful of samples: {} points",
        curve.len()
    );
    // The extent's pure on-axis angular radius in pixels at range `d`, FLOORED at the shared
    // apparent minimum — the model of the DRAWN size the strictness resolution is derived from
    // (the same camera constants the pilot capture projects with). The floor matters: across the
    // floor-dominated far segment the drawn size IS the constant floor, so no strict growth can
    // be owed there however far the camera closes (the second red run demanded strictness
    // between 11 456.6 m and 2 636.1 m — both floored, model 0.30 px vs 1.30 px — and measured
    // only the ±0.002 px off-axis wobble: 3.0016 -> 3.0004).
    let model_px = |d: f64| {
        (inner_extent_m / d * (CAPTURE_H as f64 * 0.5) / (FIT_FOV_Y * 0.5).tan())
            .max(DOT_MIN_APPARENT_RADIUS_PX)
    };
    let mut running_max = f64::NEG_INFINITY;
    let mut strict_pairs = 0u64;
    let (mut anchor_d, mut anchor_f) = (curve[0].1, curve[0].2);
    for &(tick, d, f) in &curve {
        running_max = running_max.max(f);
        assert!(
            f >= running_max - READBACK_QUANTUM_PX,
            "the point of light SHRANK on approach: {f:.4} px at tick {tick} ({d:.1} m) \
             against a running maximum of {running_max:.4} px",
        );
        if model_px(d) - model_px(anchor_d) >= READBACK_QUANTUM_PX {
            assert!(
                f > anchor_f,
                "the growth must be STRICT across a quantum of model growth: \
                 {anchor_f:.4} px ({anchor_d:.1} m) -> {f:.4} px ({d:.1} m)",
            );
            strict_pairs += 1;
            (anchor_d, anchor_f) = (d, f);
        }
    }
    assert!(
        strict_pairs > 0,
        "the strict arm never ran — the curve never rose above the floor (still the old \
         constant dot)",
    );

    // ---- THE END: 200 m out, the point of light is the planet's TRUE angular size. ----
    // SLICE 4's WAKE, AWAITED WITHIN ITS OWN DERIVED BUDGET (never a sleep literal): the
    // approach crossed the system's 444.104489631 m interior spin-up radius in its last second,
    // so the interest byte has just landed and the vacated system owes its planets' own
    // pictures within the wake budget at the cluster's measured boot. Poll the diagnosis
    // surface for the marker⇒look flip, bounded by that budget in wall time — doubled, stated:
    // the budget counts UNIVERSE ticks and this wait is wall-clock across five processes, so
    // one factor of two absorbs pacing jitter and the poll's own sampling gap (the same
    // resolution allowance the in-flight handover measurement adds explicitly).
    let boot_ticks = orch_rlm(a.admin).boot_ticks_observed_max;
    let wake_secs = wake_budget_ticks(boot_ticks) as f64 * DEV.tick_dt;
    let wake_deadline = Instant::now() + Duration::from_secs_f64(2.0 * wake_secs);
    let inner_name = format!("{:?}", roster.inner);
    loop {
        let st = vd_bins::pixel::poll(devctl);
        if st
            .realm_boxes
            .iter()
            .any(|b| (b.realm == inner_name) & (b.body_kind == "look"))
        {
            break;
        }
        assert!(
            Instant::now() < wake_deadline,
            "standing inside the vacated system's interior band, the planets' own pictures \
             never arrived within the derived wake budget ({wake_secs:.1} s at measured boot \
             {boot_ticks} ticks, doubled for wall-clock jitter) — the slice-4 wake failed",
        );
        std::thread::sleep(SAMPLE_POLL);
    }
    let cap1 = vd_bins::pixel::straddle(
        devctl,
        "growth-end",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch,
        vd_bins::pixel::pilot_camera,
    );
    let end = vd_bins::pixel::subject(&cap1.post, &cap1.camera, roster.inner);
    let (pos1, _) = own_pose(&cap1.post).expect("a delivered pose at the growth end");
    let dist1 = (end.centre_m - pos1).length();
    // SLICE 4 LANDED (look_horizon.md §6 slice 4; Q1 APPROVED, owner 2026-08-17): this assert
    // used to pin the OLD posture ("waking it from outside is slice 4's work" — ParentMarker),
    // and it went RED the run the interest bit landed, exactly as its own message predicted.
    // Standing here — outside the 150 m shell, INSIDE the system's 444.104489631 m interior
    // band — the vacated system now holds the interest byte, its down-proxy wakes the planets,
    // and the planet's OWN picture arrives through the sealed interior forward (slice 3). The
    // planets are running when you look at them: presence is the planet's OWN statement.
    assert_eq!(
        assert_presence(&end, "growth-end/inner"),
        Author::SelfLook,
        "standing inside the vacated system's interior band, the planet draws its OWN picture \
         (the slice-4 wake + the slice-3 sealed interior forward)",
    );
    // The delivered one-radius datum IS the world's own extent (the wire's extent against the
    // out-of-band boot derivation — not the gate reading its own output back).
    let delivered_extent = cap1
        .post
        .realm_boxes
        .iter()
        .find(|b| b.realm == format!("{:?}", roster.inner))
        .expect("the inner planet's box is in the diagnosis surface")
        .extent_m;
    assert!(
        (delivered_extent - inner_extent_m).abs() < 1.0e-9,
        "the delivered marker radius ({delivered_extent}) must be THE world's own circumscribed \
         extent ({inner_extent_m})",
    );
    let expected_end_world = vd_client_harness::camera::marker_world_radius(
        inner_extent_m,
        dist1,
        cap1.camera.fov_y,
        cap1.camera.height as f64,
    );
    let expected_end_px = vd_client_harness::verdict::projected_point_aabb(
        &cap1.camera,
        end.centre_m,
        expected_end_world,
    )
    .map_or(0.0, |r| (r.max.x - r.min.x) * 0.5);
    assert!(
        (end.radius_px - expected_end_px).abs() < READBACK_QUANTUM_PX,
        "at {dist1:.1} m the point must draw at its true angular size ({expected_end_px:.2} px), \
         measured {:.2} px",
        end.radius_px,
    );
    assert!(
        end.radius_px > start.radius_px + READBACK_QUANTUM_PX,
        "the point of light must have GROWN over the leg: {:.2} px -> {:.2} px",
        start.radius_px,
        end.radius_px,
    );
    let painted1 = assert_painted(&cap1, &f.cwd, &end, "growth-end/inner");
    eprintln!(
        "[growth] G-LOOK-GROWTH GREEN: {:.2} px at {dist0:.1} m -> {:.2} px at {dist1:.1} m over \
         {} distinct-tick samples ({strict_pairs} strict pairs above the floor); {painted1} \
         pixels painted at the close range; no blank sample, no shrink, no pop.",
        start.radius_px,
        end.radius_px,
        curve.len(),
    );
    assert_manifest_attests(&f.cwd, &["growth-start", "growth-end"]);
    drop(client);
}

/// THE JOURNEY'S LAST LEG: hold the throttle until the picture says the ship STANDS IN `want` —
/// the leg's outcome is a REALM LABEL, never a coordinate. Fails with the distance still to run, so
/// a stalled crossing names itself instead of timing out silently.
fn fly_until_label(devctl: u16, want: &str, subject_realm: RealmId, deadline: Duration) {
    let started = Instant::now();
    throttle(devctl, FULL_AHEAD);
    loop {
        std::thread::sleep(SAMPLE_POLL);
        let st = vd_bins::pixel::poll(devctl);
        if st.location.as_deref() == Some(want) {
            throttle(devctl, ALL_STOP);
            eprintln!("[warp] ARRIVED: the ship stands in {want}");
            return;
        }
        if started.elapsed() >= deadline {
            let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
            let sub = vd_bins::pixel::subject(&st, &camera, subject_realm);
            let (pos, _) = own_pose(&st).expect("a delivered pose");
            throttle(devctl, ALL_STOP);
            panic!(
                "the ship never crossed into {want}: it is at {pos:?}, {:.1} m from the \
                 destination's drawn centre {:?}, still standing in {:?}",
                (sub.centre_m - pos).length(),
                sub.centre_m,
                st.location,
            );
        }
    }
}

/// HR6: every capture this gate took is in the run manifest, and each carries the STATE DUMP that
/// attests every drawn row's provenance (which realm, which lawful author, which universe tick).
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
            .unwrap_or_else(|| panic!("capture '{label}' has no state dump to attest its rows"));
        let dump: DevState = serde_json::from_str(
            &std::fs::read_to_string(run_dir.join(state_rel)).expect("read state dump"),
        )
        .expect("parse state dump");
        assert!(
            !dump.realm_boxes.is_empty(),
            "capture '{label}': the attested picture has no drawn rows",
        );
        for row in &dump.realm_boxes {
            assert!(
                row.body_kind == "look" || row.body_kind == "marker",
                "capture '{label}': row {} has no lawful author ({})",
                row.realm,
                row.body_kind,
            );
        }
        eprintln!(
            "[warp] manifest attests '{label}': {} drawn rows, each with its lawful author",
            dump.realm_boxes.len(),
        );
    }
}
