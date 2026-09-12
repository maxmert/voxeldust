//! **M8-1 — THE RESIDENCY BAND ON A MOVING EYE** (the voxel foundation, slice 8 step 4; ruling V14
//! D8-3 (A): the reach plus the interpolation buffer, no new data).
//!
//! Every picture of steps 8p to 3 was a STILL stand. This gate moves the eye over the home planet
//! three ways and reads THE BAND'S GAP every frame — the stamp's `chunks_urgent`, the wanted
//! chunks inside their rung's own territory that are not resident. The band is complete when the
//! gap is zero; the assertion is that it is zero on every FRAME of the walk. The two hull legs are
//! MEASURED and reported, never asserted — see `HULL_LEG_MPS`: at 240 m/s the band holds while the
//! eye stays over the finest ring's territory and breaks the moment that ring enters; at 528 m/s
//! it breaks throughout. The ask goes to the owner with the numbers (the slice discussion §16.3).
//!
//! 1. **THE WALK.** The own character walks on foot over the ground at the foot speed (1.4 m/s):
//!    the placeholder walk the suit ruling keeps until the suit lands (on a floor the foot speed
//!    is the character's own).
//! 2. **THE HULL AT 240 m/s.** A player-built hull is berthed INSIDE THE PLANET'S REALM, 500 m
//!    over the highest ground along its path (read from the recipe), the character boards it (walks into its wake, the crossing commits), and
//!    pushes: its stick is the hull's push (the temporary control seam), scaled by the hull's own
//!    rating. The gate holds the push until the hull passes 240 m/s and releases it; the hull
//!    coasts (the live ambient has no pull and no drag), and the gate reads the band for a minute.
//! 3. **THE HULL AT 528 m/s.** The gate holds the push again until 528 m/s and reads the band
//!    again — and REPORTS it: the gap per rung, the deepest queue, the lead. MEASURED 2026-09-09 (eighth run):
//!    the lead is applied (the wanted set is computed one buffer ahead) and the band still goes
//!    incomplete on every sample, 1 822 urgent chunks at the worst, across rungs 0 to 5, the queue
//!    at 2 798 — a THROUGHPUT wall (the eye sweeps more chunks per second than the workers build),
//!    which no lead of a few chunks can bridge. The lever is the owner's (the slice discussion
//!    §16), so this leg reports and the ask goes up with its numbers.
//!
//! The speeds are MEASURED, never assumed: the planet's row moves through the hull's frame, and
//! two polls a known interval apart give the speed the gate then names. The gates fly the shipped
//! path (the suit ruling of 2026-09-05): berth a hull, board it, push — never a walking dot at
//! warp. SL10 clause 7 holds on the client: its lead is the interpolation buffer's own delivered
//! poses, and this gate is what measures whether that lead is enough (D8-3 A). It is, on the walk
//! and at 240 m/s; at 528 m/s the band fails by throughput, not by lead, so the ask that goes to
//! the owner is not R-18 (a stated lead) but the build rate (the slice discussion §16).
//!
//! GPU-required + LOCAL, like the other picture gates; `just terrain-moving-eye`. Under a plain
//! `cargo test --workspace` this file compiles to zero tests.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::memory::MemoryRead;
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, common_env, dev_auth_pubkey_hex,
    dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows, orchestrator_env,
    realm_store_path, reap_forked, reserve_tcp_addr, reserve_udp_addr,
};
use vd_client_harness::capture::{capture_rel_path, probe_rel_for, state_rel_for};
use vd_client_harness::manifest::CaptureKind;
use vd_client_harness::pop::{Frame, PairRead, frame_camera, judge_pair, sum_reads};
use vd_core::EntityId;
use vd_core::entity_kind::EntityKind;
use vd_core::glam::DVec3;
use vd_core::pose::{RealmId, frame_for_realm};
use vd_devproto::{DevPhase, DevRequest, DevResponse, DevState, WaitField, WaitOp, WaitPredicate};
use vd_io_prod::trust::ClusterTrust;
use vd_terrain::Gf;

const CLIENT_ACCOUNT_BASE: u64 = 1000;
/// The walker's eye over the ground.
const EYE_HEIGHT_M: f64 = 1.8;
/// How far over the highest ground along its path the hull flies, and how densely that ground is
/// read from the recipe (at rung 2, with the dropped octaves' bound added).
const HULL_CLEARANCE_M: f64 = 500.0;
const PATH_SAMPLE_M: f64 = 100.0;
/// The day side, as the picture gate stands it: the star this high over the horizon.
const SUN_ELEVATION_DEG: f64 = 15.0;
const SUN_OFF_NOSE_DEG: f64 = 120.0;
/// THE FOOT SPEED the walk leg commands, in metres per second: the character's own walk.
const WALK_MPS: f64 = 1.4;
/// THE HULL LEGS' speeds, in metres per second: a fast hull and the fastest one the slice names
/// (ruling V14 M8-1). Both are read and REPORTED, never asserted. MEASURED on six runs of
/// 2026-09-09/10 at 240 m/s, 1 400 m down to 960 m over rising ground: three runs green (the queue
/// at 102, 144, 109), three red — the seventh (68 urgent, the queue at 333, beside a stray shard),
/// the eleventh (7 urgent on 24 frames, the queue at 213, in the leg's last two seconds, when the
/// eye fell under 990 m and rung 0 entered) and the twelfth (34 urgent on 59 frames, the queue at
/// 338). At 528 m/s the band is incomplete on every frame (the
/// queue at 2 800, rungs 0 to 5). A leg that flaps at the machine's edge is reported, not asserted.
/// The hull's legs (ruling V14 D8-5: a hull at 1.4, 240 and 528 m/s — the walking pace flown, never
/// a walking dot; the suit ruling). The slow leg first, from rest.
const HULL_LEG_MPS: [f64; 3] = [1.4, 240.0, 528.0];
/// How far below a leg's named speed the measured speed may fall.
const SPEED_SHORTFALL: f64 = 0.9;
/// How long each leg is read, and how often.
const LEG_S: f64 = 60.0;
const SAMPLE_MS: u64 = 40;
/// THE POP DETECTOR (step 6, ruling V14 M8-3): every this many seconds of a leg the client
/// records a pair of consecutive frames (the picture, the probe and the frame's own stamp), and
/// after the leg the pairs are judged: the colour step at every pixel whose rung changed between
/// the two frames, against the floor of the pixels whose rung did not. Two frames at the
/// client's own rate.
const POP_PAIR_EVERY_S: f64 = 2.0;
const POP_PAIR_FPS: u32 = 60;
/// THE TURNING LEG (step 6, D-TERRAIN-5 item 11): at 240 m/s the hull holds its turn axis for
/// half of this at the leg's start and the opposite axis for the other half, then coasts on the
/// new heading — the wanted set on a heading the lead never asked for, and the drawn scene under
/// a hull that yaws. MEASURED on the first turning leg (a five-second hold, no counter-turn): the
/// turn axis is a torque and the hull kept spinning at about fifty degrees a second for the
/// rest of the leg, so the counter-turn brings the spin back to rest (a torque of the same
/// size for the same time), and the yaw is read on the course.
const TURN_S: f64 = 4.0;
const TURN_AXES: [f32; 3] = [0.0, 1.0, 0.0];
/// The turn axis's angular acceleration, degrees a second per second of hold (MEASURED on the
/// second turning leg: 29° after 1.3 s, 128° after 2.6 s).
const TURN_ACCEL_DEG_S2: f64 = 37.0;
const SPIN_RATE_WINDOW: Duration = Duration::from_millis(500);
/// At rest within this: the shortest hold the round trip allows (about 0.05 s) changes the rate
/// by two degrees a second, so a smaller rate cannot be corrected without overshooting
/// (MEASURED: 1.3 → 2.7°/s on a 0.04 s hold). The one stop rule of `cancel_spin`.
const SPIN_REST_DEG_S: f64 = 2.0;
const SPIN_CANCEL_ROUNDS: u32 = 4;
const SPIN_HOLD_MAX_S: f64 = 3.0;
/// How long a push may take to reach a leg's speed before the gate gives up.
const PUSH_DEADLINE: Duration = Duration::from_secs(60);
/// The interval two polls stand apart when a speed is measured.
const SPEED_INTERVAL: Duration = Duration::from_millis(2_000);
/// How far from the character the hull is berthed, along its nose: inside its wake by far, and a
/// half-minute walk.
const BERTH_STANDOFF_M: f64 = 40.0;
const LOGIN_DEADLINE: Duration = Duration::from_secs(120);
const BOARDING_DEADLINE: Duration = Duration::from_secs(300);
/// THE BOARDINGS (DEFERRED item 15, the boarding that never settled — one run in five): the
/// gate boards once; a diagnosis flight sets `VD_BOARDINGS=<n>` and boards `n` times with `n`
/// fresh pilots before the legs, each settle printing its course, so the race recurs with its
/// course in the log.
const BOARDINGS_ENV: &str = "VD_BOARDINGS";
/// How far apart a diagnosis flight's berths stand along the flight path, in metres.
const BERTH_SPACING_M: f64 = 200.0;
const TERRAIN_WAIT_TICKS: u64 = 3_600;
/// The hull's own push, ten gravities (the shipyard's own default), whole micro-metres per second
/// per second: stated here so the legs' push times are derived, never guessed.
const HULL_PUSH_MICRO_MPS2: i64 = 98_100_000;

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct Fixture {
    base: PathBuf,
    trust_dir: PathBuf,
    common: Vec<(&'static str, String)>,
    store_str: String,
    launch_path: PathBuf,
    cwd: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Kept on a failure, or on request (`VD_KEEP_FIXTURE`): the recorded pairs and their dumps
        // are the pop detector's raw material for an offline reading.
        if std::thread::panicking() || std::env::var_os(KEEP_FIXTURE_ENV).is_some() {
            eprintln!(
                "terrain_moving_eye: the fixture is KEPT for diagnosis at {}",
                self.base.display()
            );
            return;
        }
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

/// Keep the fixture (the runs with every recorded pair) after a green flight.
const KEEP_FIXTURE_ENV: &str = "VD_KEEP_FIXTURE";

fn fixture() -> Fixture {
    let trust = ClusterTrust::generate("vd-terrain-moving-eye").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-terrain-moving-eye-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let trust_dir = base.join("trust");
    std::fs::create_dir_all(&trust_dir).expect("trust dir");
    trust.write_der_dir(&trust_dir).expect("write trust");
    let cwd = base.join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("capture cwd");
    let store = base.join("orchestrator.redb");
    let launch_path = store.with_file_name(vd_bins::LAUNCH_STORE_NAME);
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    Fixture {
        store_str: store.display().to_string(),
        base,
        trust_dir,
        common,
        launch_path,
        cwd,
    }
}

struct ForkedReaper(PathBuf);
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

/// Boot the demand cluster with the spawn poses inside the home planet.
fn boot_demand_cluster(
    f: &Fixture,
    a: &ClusterAddrs,
    p: &DevClusterParams,
    spawn_poses: &str,
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
    let mut gw = gateway_env(a, &dev_auth_pubkey_hex(), p, ClusterShape::Demand);
    gw.push(("VD_SPAWN_POSES", spawn_poses.to_owned()));
    cluster.push(
        "vd-gateway",
        vd_bins::spawn_node(env!("CARGO_BIN_EXE_vd-gateway"), &f.common, &gw)
            .expect("spawn gateway"),
    );
    cluster
}

/// The capture client for one leg: the agent index picks the account, the pilot view rides the
/// own character.
fn spawn_capture_client(
    f: &Fixture,
    gateway: SocketAddr,
    name: &str,
    agent_index: u64,
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

fn await_listener(port: u16, child: &mut Child) {
    let started = Instant::now();
    loop {
        if std::net::TcpStream::connect(vd_bins::loopback(port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "the capture client exited before serving dev-control ({status}) — GPU \
                 precondition: this gate needs a working adapter (WGPU_BACKENDS/WGPU_POWER_PREF)"
            );
        }
        assert!(
            started.elapsed() < LOGIN_DEADLINE,
            "the capture client never served dev-control on {port}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn round_trip(port: u16, req: &DevRequest) -> DevResponse {
    dev_roundtrip(port, req).unwrap_or_else(|e| {
        panic!("dev-control round-trip on {port} failed: {e} — capture client gone?")
    })
}

fn poll(port: u16) -> DevState {
    match round_trip(port, &DevRequest::State) {
        DevResponse::State { state } => state,
        other => panic!("a state poll answered {other:?}"),
    }
}

fn await_active(devctl: u16) -> DevState {
    let started = Instant::now();
    loop {
        if let Ok(DevResponse::State { state }) = dev_roundtrip(devctl, &DevRequest::State)
            && state.phase == DevPhase::Active
            && state.snapshots_applied >= 5
            && !state.realm_boxes.is_empty()
        {
            return state;
        }
        assert!(
            started.elapsed() < LOGIN_DEADLINE,
            "the demand login onto the planet never converged"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Wait until the ladder has landed whole: a first chunk drawn, then nothing pending. The wait
/// POLLS and prints its time course (MEASURED 2026-09-10: a boarding that never settled — the
/// lead eye 5 591 km from the drawn one, 238 000 builds in three minutes — left only its last
/// state behind; the course is what a diagnosis needs).
fn await_settled(devctl: u16, leg: &str) {
    let reply = round_trip(
        devctl,
        &DevRequest::WaitUntil {
            predicate: WaitPredicate {
                field: WaitField::TerrainChunksDrawn,
                op: WaitOp::Ge,
                value: 1,
            },
            max_ticks: TERRAIN_WAIT_TICKS,
        },
    );
    assert!(
        matches!(reply, DevResponse::State { .. }),
        "{leg}: the terrain never drew a chunk: {reply:?}"
    );
    let started = Instant::now();
    let deadline = Duration::from_millis(TERRAIN_WAIT_TICKS * 1_000 / 20);
    let mut printed = 0u64;
    loop {
        let st = poll(devctl);
        let stamp = st
            .terrain_stamp
            .as_ref()
            .unwrap_or_else(|| panic!("{leg}: no terrain stamp while settling: {st:?}"));
        if stamp.chunks_pending == 0 {
            return;
        }
        let secs = started.elapsed().as_secs();
        if secs / SETTLE_COURSE_S > printed {
            printed = secs / SETTLE_COURSE_S;
            let own = vd_bins::pixel::own_pose(&st).map(|(p, _)| p);
            eprintln!(
                "terrain_moving_eye/{leg}: settling t {secs:4} s — {} drawn, {} pending, {} \
                 urgent {:?}, {} revealed, lead {:.1} m, {:.0} m up, rungs {}..{}, origin \
                 {:?}, own {:?}, windows {:?}",
                stamp.chunks_drawn,
                stamp.chunks_pending,
                stamp.chunks_urgent,
                stamp.urgent_per_rung,
                stamp.chunks_revealed,
                stamp.lead_m,
                stamp.altitude_m,
                stamp.rung_min,
                stamp.rung_max,
                st.origin,
                own,
                st.entity_windows
            );
        }
        assert!(
            started.elapsed() < deadline,
            "{leg}: the terrain never settled (pending {} after {deadline:?}): {st:?}",
            stamp.chunks_pending
        );
        std::thread::sleep(Duration::from_millis(SETTLE_POLL_MS));
    }
}

/// The settle wait's poll period, and how often it prints its course.
const SETTLE_POLL_MS: u64 = 250;
const SETTLE_COURSE_S: u64 = 2;

/// Set the sticky throttle: the character's walk on foot, or the hull's push once boarded.
fn throttle(devctl: u16, axes: [f32; 3]) {
    let reply = round_trip(devctl, &DevRequest::Move { axes });
    assert!(
        matches!(reply, DevResponse::Ack),
        "the throttle {axes:?} was refused: {reply:?}"
    );
}

fn label_of(realm: RealmId) -> String {
    frame_for_realm(realm, None)
        .expect("every realm of THE world has a frame")
        .label()
}

/// What one leg measured: the worst gap, the deepest queue, the fewest chunks on screen, the
/// samples taken, and the lead the band ran on.
#[derive(Debug, Default)]
struct LegRead {
    samples: u64,
    max_urgent: u64,
    urgent_samples: u64,
    /// THE FRAMES WITH A GAP across the leg, from the renderer's own counter (every frame, not
    /// every poll): the band's verdict.
    gap_frames: u64,
    /// THE THREE RATES' counters across the leg (M8-2a): frames drawn, chunks built and the
    /// nanoseconds spent, chunks harvested, harvests that filled their cap.
    frames: u64,
    built: u64,
    build_nanos: u64,
    harvested: u64,
    harvest_full: u64,
    harvest_nanos: u64,
    /// The parent cache across the leg: hits, builds, waits.
    parent_hits: u64,
    parent_builds: u64,
    parent_waits: u64,
    max_revealed: u64,
    revealed_samples: u64,
    max_pending: u64,
    min_drawn: u64,
    /// THE CLIENT'S MEMORY at the leg's end and its growth over the leg (ruling V17 item 1),
    /// from the operating system's footprint: a client whose footprint grows with the chunks it
    /// uploads, past what it draws, leaks the upload.
    memory_end: MemoryRead,
    memory_grew: MemoryRead,
    max_lead_m: f64,
    /// THE POP DETECTOR'S reading across the leg's pairs (step 6): how many pairs were judged,
    /// how many were not two consecutive frames (left out), and the sum of the judged.
    pop_pairs: u64,
    pop_skipped: u64,
    pop: PairRead,
}

/// One recorded frame of a pair, read back from the run: its picture, its probe and its stamp.
struct PairFrame {
    rgba: image::RgbaImage,
    probe: image::RgbaImage,
    stamp: vd_devproto::DevTerrainStamp,
    capture_frame: Option<u64>,
}

fn pair_frame(run: &Path, label: &str, index: u32) -> Option<PairFrame> {
    let rel = capture_rel_path(CaptureKind::Frame, &format!("{label}-{index:04}"));
    let rgba = image::open(run.join(&rel)).ok()?.to_rgba8();
    let probe = image::open(run.join(probe_rel_for(&rel))).ok()?.to_rgba8();
    let state: DevState =
        serde_json::from_str(&std::fs::read_to_string(run.join(state_rel_for(&rel))).ok()?).ok()?;
    Some(PairFrame {
        rgba,
        probe,
        capture_frame: state.capture_frame,
        stamp: state.terrain_stamp?,
    })
}

/// A recorded frame as the detector reads it: its bytes and the camera its stamp states.
fn frame_of(f: &PairFrame, w: usize, h: usize) -> Frame<'_> {
    Frame {
        rgba: f.rgba.as_raw(),
        probe: f.probe.as_raw(),
        width: w,
        height: h,
        camera: frame_camera(f.stamp.eye_body_m, f.stamp.camera_body_xyzw, w, h),
        // The overlay's rectangle, the stamp's own: its readouts are not the ground. An empty
        // rectangle (no overlay drawn) masks nothing.
        hud: {
            let r = f.stamp.hud_rect_px.map(f64::from);
            ((r[2] > r[0]) & (r[3] > r[1])).then_some(r)
        },
    }
}

/// THE POP DETECTOR's judgement of a leg's pairs: the judged pairs' sum, and how many pairs were
/// left out — a frame missing, or the two not consecutive (the stamp's own frame counter), or
/// not of one size or one body.
fn judge_pairs(pairs: &[(PathBuf, String)]) -> (u64, u64, PairRead) {
    let cell = |rung: u8| f64::from(vd_seed::ladder::cell_m(rung));
    let mut reads = Vec::new();
    let mut skipped = 0u64;
    for (run, label) in pairs {
        let (Some(a), Some(b)) = (pair_frame(run, label, 0), pair_frame(run, label, 1)) else {
            skipped += 1;
            eprintln!("terrain_moving_eye: the pair {label} lacks a frame, a probe or a stamp");
            continue;
        };
        let consecutive = b.stamp.frames == a.stamp.frames + 1;
        let same = a.rgba.dimensions() == b.rgba.dimensions() && a.stamp.realm == b.stamp.realm;
        if !consecutive || !same {
            skipped += 1;
            eprintln!(
                "terrain_moving_eye: the pair {label} is terrain frames {} and {} ({}), left out \
                 — capture frames {:?} and {:?}, ticks {:?} and {:?}, the eye moved {:.3} m",
                a.stamp.frames,
                b.stamp.frames,
                if same {
                    "not consecutive"
                } else {
                    "not one picture"
                },
                a.capture_frame,
                b.capture_frame,
                a.stamp.tick,
                b.stamp.tick,
                (DVec3::from_array(b.stamp.eye_body_m) - DVec3::from_array(a.stamp.eye_body_m))
                    .length()
            );
            continue;
        }
        let (w, h) = a.rgba.dimensions();
        let (w, h) = (w as usize, h as usize);
        let (fa, fb) = (frame_of(&a, w, h), frame_of(&b, w, h));
        reads.push(judge_pair(&fa, &fb, &cell));
    }
    (reads.len() as u64, skipped, sum_reads(&reads))
}

/// Read the band for `secs` seconds, one sample every `SAMPLE_MS`: the instrument is the stamp
/// the renderer writes every frame, polled through dev-control — never a sleep standing in for
/// a measurement.
fn read_band(
    devctl: u16,
    client_pid: u32,
    leg: &str,
    secs: f64,
    cwd: &Path,
    planet: RealmId,
    turn_until_s: Option<f64>,
) -> LegRead {
    let mut read = LegRead {
        min_drawn: u64::MAX,
        ..LegRead::default()
    };
    let memory_start = MemoryRead::of(client_pid);
    let started = Instant::now();
    // The counters at the first and the last sample: the leg's differences.
    let mut first: Option<[u64; 10]> = None;
    let mut last = [0u64; 10];
    let mut last_drawn = 0u64;
    let slug: String = leg
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let mut pairs: Vec<(PathBuf, String)> = Vec::new();
    let mut next_pair_s = 0.0;
    // THE TURN: the axis held from the leg's start to half of `turn_until_s`, the opposite axis
    // to `turn_until_s`, then released (a torque and its counter-torque).
    let mut phase = u8::from(turn_until_s.is_some());
    if phase == 1 {
        throttle(devctl, TURN_AXES);
    }
    while started.elapsed().as_secs_f64() < secs {
        let t = started.elapsed().as_secs_f64();
        if phase == 1 && turn_until_s.is_some_and(|until| t >= until * 0.5) {
            throttle(devctl, [-TURN_AXES[0], -TURN_AXES[1], -TURN_AXES[2]]);
            phase = 2;
        }
        if phase == 2 && turn_until_s.is_some_and(|until| t >= until) {
            throttle(devctl, [0.0, 0.0, 0.0]);
            let residual = cancel_spin(devctl, planet, leg);
            eprintln!(
                "terrain_moving_eye/{leg}: the turn is done at t {:.1} s, the residual spin \
                 {residual:.1}°/s",
                started.elapsed().as_secs_f64()
            );
            phase = 3;
        }
        // THE POP DETECTOR's pair: two frames at the client's rate, on the leg's clock.
        if started.elapsed().as_secs_f64() >= next_pair_s {
            let label = format!("pop-{slug}-{:03}", pairs.len());
            let reply = round_trip(
                devctl,
                &DevRequest::Record {
                    fps: POP_PAIR_FPS,
                    secs: 2.0 / f64::from(POP_PAIR_FPS),
                    label: Some(label.clone()),
                    consecutive: true,
                },
            );
            match reply {
                DevResponse::Recorded { path, frames: 2 } => {
                    let run = Path::new(&path);
                    let run = if run.is_absolute() {
                        run.to_path_buf()
                    } else {
                        cwd.join(run)
                    };
                    pairs.push((run, label));
                }
                other => {
                    read.pop_skipped += 1;
                    eprintln!(
                        "terrain_moving_eye/{leg}: the pair {label} was not recorded: {other:?}"
                    );
                }
            }
            next_pair_s += POP_PAIR_EVERY_S;
        }
        let st = poll(devctl);
        let stamp = st
            .terrain_stamp
            .as_ref()
            .unwrap_or_else(|| panic!("{leg}: no terrain stamp while the leg runs: {st:?}"));
        last = [
            stamp.urgent_frames,
            stamp.frames,
            stamp.built_chunks,
            stamp.build_nanos,
            stamp.harvested,
            stamp.harvest_full,
            stamp.parent_hits,
            stamp.parent_builds,
            stamp.parent_waits,
            stamp.harvest_nanos,
        ];
        first.get_or_insert(last);
        // THE TIME COURSE, every twenty-fifth sample: how the queue and the gap move, so a
        // capacity wall (the queue grows steadily) and a release storm (the screen empties at
        // once) read differently.
        if read.samples.is_multiple_of(25) {
            eprintln!(
                "terrain_moving_eye/{leg}: t {:5.1} s — {} drawn, {} pending, {} urgent {:?}, \
                 {} revealed, lead {:.1} m, {:.0} m up, rungs {}..{} {:?}, the planet turned \
                 {:.1}° in the window",
                started.elapsed().as_secs_f64(),
                stamp.chunks_drawn,
                stamp.chunks_pending,
                stamp.chunks_urgent,
                stamp.urgent_per_rung,
                stamp.chunks_revealed,
                stamp.lead_m,
                stamp.altitude_m,
                stamp.rung_min,
                stamp.rung_max,
                stamp.chunks_per_rung,
                turned_deg(planet_facing(&st, planet))
            );
        }
        read.samples += 1;
        read.max_urgent = read.max_urgent.max(stamp.chunks_urgent);
        read.urgent_samples += u64::from(stamp.chunks_urgent > 0);
        read.max_revealed = read.max_revealed.max(stamp.chunks_revealed);
        read.revealed_samples += u64::from(stamp.chunks_revealed > 0);
        read.max_pending = read.max_pending.max(stamp.chunks_pending);
        read.min_drawn = read.min_drawn.min(stamp.chunks_drawn);
        last_drawn = stamp.chunks_drawn;
        read.max_lead_m = read.max_lead_m.max(stamp.lead_m);
        std::thread::sleep(Duration::from_millis(SAMPLE_MS));
    }
    let (judged, skipped, pop) = judge_pairs(&pairs);
    read.pop_pairs = judged;
    read.pop_skipped += skipped;
    read.pop = pop;
    let first = first.unwrap_or(last);
    // A counter that went backwards (a client restart mid-leg) is refused, not wrapped.
    assert!(
        first.iter().zip(last.iter()).all(|(a, b)| a <= b),
        "{leg}: a stamp counter went backwards — the client restarted during the leg"
    );
    read.gap_frames = last[0].saturating_sub(first[0]);
    read.frames = last[1].saturating_sub(first[1]);
    read.built = last[2].saturating_sub(first[2]);
    read.build_nanos = last[3].saturating_sub(first[3]);
    read.harvested = last[4].saturating_sub(first[4]);
    read.harvest_full = last[5].saturating_sub(first[5]);
    read.parent_hits = last[6].saturating_sub(first[6]);
    read.parent_builds = last[7].saturating_sub(first[7]);
    read.parent_waits = last[8].saturating_sub(first[8]);
    read.harvest_nanos = last[9].saturating_sub(first[9]);
    read.memory_end = MemoryRead::of(client_pid);
    read.memory_grew = read.memory_end.since(memory_start);
    // THE CLIENT'S MEMORY over the leg (ruling V17 item 1): the footprint at the end, and what
    // grew, beside the chunks the leg uploaded — the growth per uploaded chunk, not per drawn
    // chunk, is what a leak reads as.
    eprintln!(
        "terrain_moving_eye/{leg}: THE MEMORY — at the end {}; over the leg it grew by {} \
         while {} chunks were harvested and the screen ended at {} chunks",
        read.memory_end, read.memory_grew, read.harvested, last_drawn
    );
    // THE THREE RATES (M8-2a, ruling V15): what the workers build, what the engine harvests, and
    // the frames — per second of the leg — with the mean build time and the share of frames whose
    // harvest filled its cap. A harvest at its cap on most frames is the wall.
    eprintln!(
        "terrain_moving_eye/{leg}: THE THREE RATES — {:.1} frames/s, the workers ran {:.0} \
         jobs/s ({:.2} ms of wall time each, a wait on a sibling's parent included), the engine \
         harvested {:.0} chunks/s ({:.2} ms of the main thread each), the harvest filled its cap \
         on {} of {} frames",
        read.frames as f64 / secs,
        read.built as f64 / secs,
        if read.built > 0 {
            read.build_nanos as f64 / read.built as f64 / 1.0e6
        } else {
            0.0
        },
        read.harvested as f64 / secs,
        if read.harvested > 0 {
            read.harvest_nanos as f64 / read.harvested as f64 / 1.0e6
        } else {
            0.0
        },
        read.harvest_full,
        read.frames
    );
    eprintln!(
        "terrain_moving_eye/{leg}: THE PARENT CACHE — {} hits, {} builds, {} waits ({:.0} % hit)",
        read.parent_hits,
        read.parent_builds,
        read.parent_waits,
        if read.parent_hits + read.parent_builds > 0 {
            100.0 * read.parent_hits as f64 / (read.parent_hits + read.parent_builds) as f64
        } else {
            0.0
        }
    );
    eprintln!(
        "terrain_moving_eye/{leg}: {} samples over {secs:.0} s — the band's gap peaked at {} \
         urgent chunks ({} samples with a gap, {} FRAMES with a gap); the reveals past the \
         horizon at {} chunks ({} samples); the queue at {} pending, {} chunks on screen at the \
         least, the lead up to {:.1} m",
        read.samples,
        read.max_urgent,
        read.urgent_samples,
        read.gap_frames,
        read.max_revealed,
        read.revealed_samples,
        read.max_pending,
        read.min_drawn,
        read.max_lead_m
    );
    eprintln!(
        "terrain_moving_eye/{leg}: THE POP DETECTOR — {} pairs judged ({} left out), {} pixels \
         compared ({} nearer than the limit, which reached {:.0} m; {} off the first frame), {} \
         crossed a rung boundary, the floor at {} levels, {} past the floor, the widest {} \
         levels; per boundary (finer rung, crossed, past the floor, widest) {:?}",
        read.pop_pairs,
        read.pop_skipped,
        read.pop.compared,
        read.pop.near_skipped,
        read.pop.near_limit_m,
        read.pop.off_frame,
        read.pop.crossed(),
        read.pop.floor(),
        read.pop.over_floor(),
        read.pop.widest(),
        read.pop
            .boundary_reads()
            .iter()
            .map(|b| (b.finer, b.crossed, b.over_floor, b.widest))
            .collect::<Vec<_>>()
    );
    read
}

/// THE PLANET'S MOTION THROUGH THE EYE'S FRAME: where the planet's row stands in the picture. For
/// a character on the planet it stands still; for a character inside a flying hull it moves at
/// the hull's own speed, the other way.
/// The planet's box facing in the pilot's window: how the planet's frame turns in the frame the
/// window draws (the hull's, aboard), as a quaternion.
fn planet_facing(state: &DevState, planet: RealmId) -> vd_core::glam::DQuat {
    let label = format!("{planet:?}");
    let row = state
        .realm_boxes
        .iter()
        .find(|b| b.realm == label)
        .unwrap_or_else(|| panic!("the planet's row {label} is in the window: {state:?}"));
    vd_core::glam::DQuat::from_array(row.facing).normalize()
}

/// How far a facing has turned from the identity, in degrees, 0 to 360 (its own axis's sense).
fn turned_deg(q: vd_core::glam::DQuat) -> f64 {
    let v = DVec3::new(q.x, q.y, q.z).length();
    (2.0 * v.atan2(q.w)).to_degrees().rem_euclid(360.0)
}

fn planet_centre(state: &DevState, planet: RealmId) -> DVec3 {
    let label = format!("{planet:?}");
    let row = state
        .realm_boxes
        .iter()
        .find(|b| b.realm == label)
        .unwrap_or_else(|| panic!("the planet's row {label} is in the window: {state:?}"));
    DVec3::from_array(row.center)
}

/// The speed of the eye through the planet, MEASURED from two polls `SPEED_INTERVAL` apart, in
/// metres per second.
fn measure_speed(devctl: u16, planet: RealmId) -> f64 {
    let a = poll(devctl);
    let t0 = Instant::now();
    let c0 = hull_in_planet(&a, planet);
    std::thread::sleep(SPEED_INTERVAL);
    let b = poll(devctl);
    let dt = t0.elapsed().as_secs_f64();
    let c1 = hull_in_planet(&b, planet);
    (c1 - c0).length() / dt
}

/// THE HULL'S PLACE IN THE PLANET'S FRAME, from the planet's box in the pilot's window: the
/// facing's inverse on the centre's negative. Rotation-invariant — MEASURED with the box's centre
/// alone: a hull spinning five degrees a second read 587 km/s, the centre swinging on a 6 371 km
/// lever, and the push to 528 m/s never fired.
fn hull_in_planet(state: &DevState, planet: RealmId) -> DVec3 {
    planet_facing(state, planet).inverse() * -planet_centre(state, planet)
}

/// THE YAW RATE of the hull, degrees a second, signed about the turn axis: the planet's facing in
/// the window read twice, `SPIN_RATE_WINDOW` apart.
fn yaw_rate_deg_s(devctl: u16, planet: RealmId) -> f64 {
    let q0 = planet_facing(&poll(devctl), planet);
    let t0 = Instant::now();
    std::thread::sleep(SPIN_RATE_WINDOW);
    let q1 = planet_facing(&poll(devctl), planet);
    let dt = t0.elapsed().as_secs_f64();
    let r = (q1 * q0.inverse()).normalize();
    let axis = DVec3::new(r.x, r.y, r.z);
    let angle = 2.0 * axis.length().atan2(r.w);
    let signed = if axis.y < 0.0 { -angle } else { angle };
    signed.to_degrees() / dt
}

/// CANCEL THE SPIN: the turn axis is a torque (MEASURED: a hold turned the hull at
/// `TURN_ACCEL_DEG_S2` a second for every second held, and it kept spinning after release), so
/// after the counter-turn a residual rate remains; each round reads the rate and holds the
/// opposite axis for the time that rate takes to cancel, until the hull is at rest within
/// `SPIN_REST_DEG_S`. The product's own answer is a ship's safety block (ruling 2026-08-27, item
/// 4: slowing is gameplay); the instrument closes the loop itself.
fn cancel_spin(devctl: u16, planet: RealmId, leg: &str) -> f64 {
    let mut rate = yaw_rate_deg_s(devctl, planet);
    for round in 0..SPIN_CANCEL_ROUNDS {
        if rate.abs() < SPIN_REST_DEG_S {
            break;
        }
        let hold_s = (rate.abs() / TURN_ACCEL_DEG_S2).min(SPIN_HOLD_MAX_S);
        let sign = if rate > 0.0 { 1.0 } else { -1.0 };
        throttle(
            devctl,
            [
                -sign * TURN_AXES[0],
                -sign * TURN_AXES[1],
                -sign * TURN_AXES[2],
            ],
        );
        std::thread::sleep(Duration::from_secs_f64(hold_s));
        throttle(devctl, [0.0, 0.0, 0.0]);
        let after = yaw_rate_deg_s(devctl, planet);
        eprintln!(
            "terrain_moving_eye/{leg}: spin round {round}: {rate:.1}°/s, held the opposite axis \
             {hold_s:.2} s, now {after:.1}°/s"
        );
        rate = after;
    }
    rate
}

/// Push the hull along its nose until it passes `target_mps`, then release the stick: the hull
/// coasts. The push time is derived from the hull's own rating and the speed it has; the speed is
/// then measured, and the push repeated while it falls short.
fn push_to(devctl: u16, planet: RealmId, target_mps: f64) -> f64 {
    let push_mps2 = HULL_PUSH_MICRO_MPS2 as f64 / 1.0e6;
    let started = Instant::now();
    let mut speed = measure_speed(devctl, planet);
    while speed < target_mps {
        assert!(
            started.elapsed() < PUSH_DEADLINE,
            "the hull never reached {target_mps} m/s (at {speed:.1} m/s)"
        );
        let need_s = (target_mps - speed) / push_mps2;
        throttle(devctl, [1.0, 0.0, 0.0]);
        std::thread::sleep(Duration::from_secs_f64(need_s.max(0.05)));
        throttle(devctl, [0.0, 0.0, 0.0]);
        speed = measure_speed(devctl, planet);
        eprintln!("terrain_moving_eye: pushed {need_s:.2} s, the hull flies at {speed:.1} m/s");
    }
    speed
}

/// A stand on the planet: the character's offset in the planet's frame and its facing.
struct Stand {
    offset_m: DVec3,
    orient: vd_core::glam::DQuat,
}

fn stand(d: DVec3, height_m: f64, surface_m: f64, forward: DVec3) -> Stand {
    let up = (d - forward * forward.dot(d)).normalize();
    let right = forward.cross(up).normalize();
    let basis = vd_core::glam::DMat3::from_cols(right, up, -forward);
    Stand {
        offset_m: d * (surface_m + height_m),
        orient: vd_core::glam::DQuat::from_mat3(&basis).normalize(),
    }
}

fn spawn_entry(account: u64, planet_seed: u64, st: &Stand) -> String {
    format!(
        "{account}=Planet({planet_seed}):{},{},{}@{},{},{},{}",
        st.offset_m.x,
        st.offset_m.y,
        st.offset_m.z,
        st.orient.x,
        st.orient.y,
        st.orient.z,
        st.orient.w
    )
}

/// The shipyard stand-in's mint for the hull with serial `seq`, restated so the gate knows the
/// hull's name.
fn minted_hull(seq: u64) -> RealmId {
    RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, seq, 0))
}

/// Write a hull into THE PLANET'S file and its own: a berth inside the planet's realm, stated
/// in the planet's frame, where the planet's shard reads it when it boots. Serial `seq` names
/// it (a diagnosis flight plants one hull per boarding: a boarded hull keeps its pilot's body,
/// and a third pilot found the berth blocked — MEASURED 2026-09-11, the crossing never
/// committed; players collide).
fn plant_hull(f: &Fixture, planet: RealmId, berth_m: DVec3, seq: u64) -> RealmId {
    let parent_store = realm_store_path(&f.base, planet);
    let ship_store = realm_store_path(&f.base, minted_hull(seq));
    let out = Command::new(env!("CARGO_BIN_EXE_vd-build-ship"))
        .args([
            "--parent-store",
            &parent_store,
            "--ship-store",
            &ship_store,
            "--owner",
            "1001",
            "--seq",
            &seq.to_string(),
            "--max-push-micro-mps2",
            &HULL_PUSH_MICRO_MPS2.to_string(),
            "--berth-x-m",
            &berth_m.x.to_string(),
            "--berth-y-m",
            &berth_m.y.to_string(),
            "--berth-z-m",
            &berth_m.z.to_string(),
        ])
        .output()
        .expect("the shipyard's stand-in runs");
    assert!(
        out.status.success(),
        "vd-build-ship: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let printed = String::from_utf8_lossy(&out.stdout);
    let hull = minted_hull(seq);
    assert!(
        printed.contains(&hull.to_string()),
        "the tool names the hull it built ({hull}); it printed {printed:?}",
    );
    hull
}

/// The verdict of one leg: the band never incomplete on any sample.
fn assert_band_complete(leg: &str, read: &LegRead) {
    assert!(read.samples > 0, "{leg}: no sample");
    assert!(read.min_drawn > 0, "{leg}: the ground left the screen");
    assert_eq!(
        read.gap_frames,
        0,
        "{leg}: THE BAND WENT INCOMPLETE on {} frames — {} urgent chunks missing at the worst \
         sample, on {} of {} samples (the queue peaked at {} pending, the lead ran up to {:.1} m). \
         Read the time course: a queue that grows steadily is a throughput wall (the workers' \
         build rate), a gap with a shallow queue is the band's rule (the lead, the territory, the \
         skyline).",
        read.gap_frames,
        read.max_urgent,
        read.urgent_samples,
        read.samples,
        read.max_pending,
        read.max_lead_m
    );
    assert_eq!(
        read.max_urgent, 0,
        "{leg}: a sample saw a gap no frame counted"
    );
}

/// ONE JOB AT A TIME, enforced: a failed run keeps its fixture and its shards keep running; a
/// later run beside them measures the machine, not the band (the seventh run of 2026-09-09
/// failed the 240 m/s leg beside such a shard). The gate refuses to start while any process of
/// an earlier fixture runs.
fn refuse_stray_fixtures() {
    let out = Command::new("ps")
        .args(["-Ao", "command"])
        .output()
        .expect("ps lists the processes");
    let listing = String::from_utf8_lossy(&out.stdout);
    let strays: Vec<&str> = listing
        .lines()
        .filter(|l| l.contains("vd-terrain-moving-eye-") && !l.contains(" ps "))
        .collect();
    assert!(
        strays.is_empty(),
        "an earlier run's fixture still runs; stop it first (one job at a time): {strays:#?}"
    );
}

#[test]
fn the_band_stays_complete_on_a_walk_and_on_two_hull_legs() {
    refuse_stray_fixtures();
    let _tier = vd_bins::cluster_tier();
    let body = vd_bins::home_body(DEV.universe_seed).expect("the home planet");
    let planet = RealmId::Planet(body.seed());
    // THE STAND, as the picture gate finds it: the day side, on the equator, the star over the
    // shoulder. The nose points along the ground.
    let orbit = vd_bins::home_orbit(DEV.universe_seed).expect("the home planet's orbit");
    let sun = -vd_physics::celestial::orbital_state(&orbit, 0.0)
        .position
        .normalize();
    let along = sun.cross(DVec3::Z).normalize();
    let zenith = (90.0_f64 - SUN_ELEVATION_DEG).to_radians();
    let d = (sun * zenith.cos() + along * zenith.sin()).normalize();
    let dir = [Gf::from_f64(d.x), Gf::from_f64(d.y), Gf::from_f64(d.z)];
    let h = vd_terrain::height::height_m(&body, dir, 0).to_f64();
    let toward_sun = (sun - d * sun.dot(d)).normalize();
    let ahead =
        vd_core::glam::DQuat::from_axis_angle(d, SUN_OFF_NOSE_DEG.to_radians()) * toward_sun;
    let level = ahead;
    // The walker faces LEVEL: it walks along its nose, and a nose tilted down walks into the
    // ground (MEASURED on the first run: 8° down, under the surface after 13 m, and with the eye
    // under the surface the horizon is zero, no wall is raised, and everything within the reach
    // is wanted — 1 930 chunks pending on a walk).
    let walker = stand(d, EYE_HEIGHT_M, h, level);
    // THE HULL'S ALTITUDE, FROM THE RECIPE: the hull flies level along the planet frame's `−Z`
    // (its own nose) for the longest leg's distance, and the ground under that path rises where
    // it rises; the berth stands `HULL_CLEARANCE_M` over the highest ground along the path, read
    // from the recipe (MEASURED on the first hull leg at a stated 300 m: the hull flew into rising
    // ground 12 km on, the eye stood under the surface, and the ladder collapsed to a dozen
    // chunks). Nothing here states a height; the world does.
    // The whole flight: both legs at their speeds for their read and their two speed polls, and
    // the two pushes (a push from rest to `v` at the hull's rating covers `v² / 2a`), a fifth
    // over (refutation R4-4: the first form read the probe leg alone, 40 km of a 50 km flight).
    let push_mps2 = HULL_PUSH_MICRO_MPS2 as f64 / 1.0e6;
    let read_s = LEG_S + 2.0 * SPEED_INTERVAL.as_secs_f64();
    let leg_m = HULL_LEG_MPS
        .iter()
        .map(|v| v * read_s + v * v / (2.0 * push_mps2))
        .sum::<f64>()
        * 1.2;
    let mut highest = h;
    let mut s_m = 0.0;
    while s_m <= leg_m {
        let p = d * h + DVec3::NEG_Z * s_m;
        let q = p.normalize();
        let hq = vd_terrain::height::height_m(
            &body,
            [Gf::from_f64(q.x), Gf::from_f64(q.y), Gf::from_f64(q.z)],
            2,
        )
        .to_f64();
        highest = highest.max(hq + body.dropped_bound_m(2).to_f64());
        // Between two samples the surface may stand higher than at either (the bound holds AT a
        // sampled direction): the clearance carries that (refutation R4-30).
        if ((s_m / PATH_SAMPLE_M).round() as u64).is_multiple_of(20) {
            eprintln!(
                "terrain_moving_eye: the ground {:.0} m along the path stands {:.0} m over the stand",
                s_m,
                hq - h
            );
        }
        s_m += PATH_SAMPLE_M;
    }
    let hull_m = highest - h + HULL_CLEARANCE_M;
    eprintln!(
        "terrain_moving_eye: the ground along the {leg_m:.0} m path rises to {:.0} m over the \
         stand's surface; the hull flies {hull_m:.0} m over it",
        highest - h
    );
    let pilot = stand(d, hull_m, h, level);
    let boardings: u64 = std::env::var(BOARDINGS_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);
    // One pilot's stand and one berth per boarding, spaced along the path so no boarded hull's
    // body stands in a later pilot's way.
    let mut pilots: Vec<Stand> = Vec::new();
    let mut berths: Vec<DVec3> = Vec::new();
    for b in 0..boardings {
        let db = (d * h + level * (BERTH_SPACING_M * b as f64)).normalize();
        let hb = vd_terrain::height::height_m(
            &body,
            [Gf::from_f64(db.x), Gf::from_f64(db.y), Gf::from_f64(db.z)],
            0,
        )
        .to_f64();
        let pb = stand(db, hull_m + (h - hb).max(0.0), hb, level);
        berths.push(pb.offset_m + level * BERTH_STANDOFF_M);
        pilots.push(pb);
    }
    // THE BERTH: the hull's centre `BERTH_STANDOFF_M` ahead of the pilot's stand, level with it,
    // in the planet's frame. The hull's own nose is the planet frame's `−Z`: at this stand that
    // is within a degree of level (the stand lies on the equator's plane), so a push flies the
    // hull along the ground, rising slowly as the ground curves away.
    let berth = pilot.offset_m + level * BERTH_STANDOFF_M;
    let mut spawn_list = vec![spawn_entry(CLIENT_ACCOUNT_BASE, body.seed(), &walker)];
    for (b, pb) in pilots.iter().enumerate() {
        spawn_list.push(spawn_entry(
            CLIENT_ACCOUNT_BASE + 1 + b as u64,
            body.seed(),
            pb,
        ));
    }
    let spawn_poses = spawn_list.join(";");
    eprintln!(
        "terrain_moving_eye: {planet:?}, surface {h:.1} m, the walker at {:?}, the pilot at {:?}, \
         the hull berthed at {berth:?} (nose −Z, {:.2}° off level)",
        walker.offset_m,
        pilot.offset_m,
        (DVec3::NEG_Z.dot(d)).asin().to_degrees()
    );

    let f = fixture();
    let hulls: Vec<RealmId> = berths
        .iter()
        .enumerate()
        .map(|(b, bm)| plant_hull(&f, planet, *bm, 1 + b as u64))
        .collect();
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_cluster(&f, &a, &DEV, &spawn_poses);

    // ---- LEG 1: THE WALK.
    let walk = {
        let client_quic = reserve_udp_addr();
        let devctl = reserve_tcp_addr().port();
        let mut client = ChildGuard(spawn_capture_client(
            &f,
            a.gateway,
            "walk",
            0,
            client_quic.port(),
            devctl,
        ));
        await_listener(devctl, &mut client.0);
        let landed = await_active(devctl);
        assert_eq!(
            landed.location.as_deref(),
            Some(label_of(planet).as_str()),
            "the walker lands on the planet: {landed:?}"
        );
        await_settled(devctl, "walk");
        // THE FOOT SPEED is COMMANDED BY FEEDBACK: the share seeds from the shared parameter and
        // the shard's own speed law scales it; the gate measures the speed the shard gives and
        // corrects the share until the walk is the foot speed (MEASURED on the first run: the
        // seed alone walked at 0.52 m/s). The walk covers about 84 m in its minute — a rung-0
        // chunk and a third: the eye's own chunk changes once.
        let speed = walk_at(devctl, WALK_MPS);
        let read = read_band(devctl, client.0.id(), "walk", LEG_S, &f.cwd, planet, None);
        throttle(devctl, [0.0, 0.0, 0.0]);
        eprintln!("terrain_moving_eye/walk: the character walked at {speed:.2} m/s");
        assert!(
            speed >= WALK_MPS * SPEED_SHORTFALL,
            "the walk is slower than the foot speed: {speed:.2} m/s"
        );
        let _ = round_trip(devctl, &DevRequest::Close);
        read
    };

    // ---- LEGS 2 AND 3: THE HULL. (A diagnosis flight boards `boardings` times with fresh
    // pilots and flies the legs on the last.)
    let (slow, fast, turning, fastest) = {
        let mut boarded: Option<(ChildGuard, u16)> = None;
        for b in 0..boardings {
            let client_quic = reserve_udp_addr();
            let devctl = reserve_tcp_addr().port();
            let mut client = ChildGuard(spawn_capture_client(
                &f,
                a.gateway,
                "hull",
                1 + b,
                client_quic.port(),
                devctl,
            ));
            await_listener(devctl, &mut client.0);
            let landed = await_active(devctl);
            assert_eq!(
                landed.location.as_deref(),
                Some(label_of(planet).as_str()),
                "the pilot lands on the planet: {landed:?}"
            );
            let tag = format!("hull {}, before boarding", b + 1);
            await_settled(devctl, &tag);
            // BOARD: walk into the hull's wake; its shard spawns by the ordinary demand and the
            // crossing commits when it runs. The aim is the berth in the planet's frame, which
            // the pilot stands in until the crossing.
            let berth_b = berths[b as usize];
            let hull_b = hulls[b as usize];
            vd_bins::flight::cross_leg(
                devctl,
                "into the hull",
                move |_tick| berth_b,
                &label_of(hull_b),
                BOARDING_DEADLINE,
            );
            let aboard = poll(devctl);
            assert_eq!(
                aboard.location.as_deref(),
                Some(label_of(hull_b).as_str()),
                "the pilot is aboard: {aboard:?}"
            );
            // From inside the hull the planet's ground is drawn: the ladder lands whole.
            let tag = format!("hull {}, aboard", b + 1);
            await_settled(devctl, &tag);
            eprintln!(
                "terrain_moving_eye: boarding {} of {boardings} settled",
                b + 1
            );
            if let Some((prev, prev_devctl)) = boarded.take() {
                let _ = round_trip(prev_devctl, &DevRequest::Close);
                drop(prev);
            }
            boarded = Some((client, devctl));
        }
        let (client, devctl) = boarded.expect("at least one boarding");
        let mut reads = Vec::new();
        for (i, target) in HULL_LEG_MPS.iter().enumerate() {
            let leg = format!("hull {target} m/s");
            // `push_to` returns at or past the target, or panics at its deadline: the speed
            // needs no second assertion (refutation R4-7).
            let speed = push_to(devctl, planet, *target);
            let read = read_band(devctl, client.0.id(), &leg, LEG_S, &f.cwd, planet, None);
            eprintln!(
                "terrain_moving_eye/{leg}: leg {} coasted at {speed:.1} m/s",
                i + 2
            );
            reads.push(read);
        }
        // THE TURNING LEG, LAST (after the straight legs, so their readings keep their history —
        // MEASURED with the turn before the 528 m/s leg: the push then fired along the turned
        // nose, the hull climbed, the ground left the view). The heading before and after, from
        // the planet's box as the pilot's window states it (the hull's own box is its frame; the
        // planet turns in it when the hull yaws).
        let leg = "hull turning";
        let before = planet_facing(&poll(devctl), planet);
        let read = read_band(
            devctl,
            client.0.id(),
            leg,
            LEG_S,
            &f.cwd,
            planet,
            Some(TURN_S),
        );
        let after = planet_facing(&poll(devctl), planet);
        let turned = before.angle_between(after).to_degrees();
        eprintln!(
            "terrain_moving_eye/{leg}: the turn and its counter-turn over {TURN_S} s left the \
             heading {turned:.1}° from the start (the planet's box facing before {before:?}, \
             after {after:?})"
        );
        reads.push(read);
        let _ = round_trip(devctl, &DevRequest::Close);
        let turning = reads.pop().expect("four legs");
        let fastest = reads.pop().expect("four legs");
        let fast = reads.pop().expect("four legs");
        let slow = reads.pop().expect("four legs");
        (slow, fast, turning, fastest)
    };
    // The verdicts, after every leg has flown (so every leg's numbers are always in the log).
    report_leg(&format!("hull {} m/s", HULL_LEG_MPS[0]), &slow);
    report_leg(&format!("hull {} m/s", HULL_LEG_MPS[1]), &fast);
    report_leg(&format!("hull {} m/s", HULL_LEG_MPS[2]), &fastest);
    report_leg("hull turning", &turning);
    assert_band_complete("walk", &walk);
    eprintln!(
        "terrain_moving_eye: THE BAND HELD on every frame of the walk — {walk:?}; the hull legs \
         read {slow:?}, {fast:?}, {turning:?} and {fastest:?}"
    );
}

/// A HULL LEG'S REPORT: measured and named, never asserted (the file's doc). The ground must stay
/// on screen (coarser rungs cover what the finer ones miss); the rest is the number the owner
/// reads.
fn report_leg(leg: &str, read: &LegRead) {
    assert!(read.samples > 0, "{leg}: no sample");
    assert!(read.min_drawn > 0, "{leg}: the ground left the screen");
    if read.max_urgent == 0 {
        eprintln!("terrain_moving_eye/{leg}: THE BAND HELD on every frame");
    } else {
        eprintln!(
            "terrain_moving_eye/{leg}: THE BAND WENT INCOMPLETE (reported, not asserted) on {} \
             frames: {} urgent chunks missing at the worst sample, on {} of {} samples; the queue \
             peaked at {} pending; the lead ran up to {:.1} m. A throughput wall: the eye sweeps \
             more chunks per second than the workers build. The lever is the owner's (slice 8 \
             discussion §16).",
            read.gap_frames,
            read.max_urgent,
            read.urgent_samples,
            read.samples,
            read.max_pending,
            read.max_lead_m
        );
    }
}

/// Walk at `target_mps` by feedback: a share of the stick, the speed measured, the share
/// corrected, until the walk is within a tenth of the target (or the correction loop is spent,
/// and the gate then judges the measured speed).
fn walk_at(devctl: u16, target_mps: f64) -> f64 {
    let mut share = target_mps / DEV.move_speed;
    let mut speed = 0.0;
    let mut round = 0;
    while round < 5 {
        throttle(devctl, [share as f32, 0.0, 0.0]);
        // The stick reaches the shard and the delivered pose the picture: a settling pause
        // before the two polls, so the measurement is the steady walk.
        std::thread::sleep(SPEED_INTERVAL);
        speed = measure_speed_of_walker(devctl);
        eprintln!("terrain_moving_eye/walk: share {share:.5} of the stick walks at {speed:.2} m/s");
        if (speed - target_mps).abs() <= target_mps * 0.1 {
            break;
        }
        if speed > 0.0 {
            share *= target_mps / speed;
        } else {
            share *= 2.0;
        }
        round += 1;
    }
    speed
}

/// The walker's speed, MEASURED from its own row two polls apart.
fn measure_speed_of_walker(devctl: u16) -> f64 {
    let a = poll(devctl);
    let t0 = Instant::now();
    let p0 = vd_bins::pixel::own_pose(&a)
        .expect("the walker's own row")
        .0;
    std::thread::sleep(SPEED_INTERVAL);
    let b = poll(devctl);
    let dt = t0.elapsed().as_secs_f64();
    let p1 = vd_bins::pixel::own_pose(&b)
        .expect("the walker's own row")
        .0;
    (p1 - p0).length() / dt
}
