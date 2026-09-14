//! ★ THE TWO CLIENT SEAMS THE CARD NEEDS, MEASURED (ruling F9 item 2, step 1).
//!
//! Ruling F9 item 2 wires the card in as a SECOND BUILDER beside the CPU share. Two seams stood
//! unproven before any of that could be written:
//!
//! 1. A WORKER THREAD SUBMITTING COMPUTE to the renderer's own device while the renderer draws.
//! 2. A WORKER'S `device.poll(wait)` while the renderer keeps submitting frames.
//!
//! Both are wgpu questions, not game questions: the renderer's device is one object, the card is
//! one queue, and a worker that waits on the whole device waits on the renderer's frames too. So
//! this gate runs the REAL capture client and starts a worker thread part way through it — the
//! seam probe (`vd_client_render::gpu_check::spawn_seam_probe`), which dispatches the recipe's box
//! chain on the renderer's own device and reads it back, over and over, for the rest of the run.
//!
//! What it measures: THE FRAMES the client drew per second and THE WORST SINGLE FRAME of the last
//! second — THE PROBE ITSELF STATES THEM, because it is the only thing on the right thread at the
//! right moment. The terrain system publishes both through
//! `vd_client_render::terrain::FrameMeter`, and the probe reports them every two seconds beside
//! its own rate: first while it IDLES, then while it BUILDS. This gate reads the client's log and
//! states the two phases.
//!
//! The card itself is OFF here (`VD_TERRAIN_GPU=0`): this gate asks what the SEAM costs, never what
//! the builder buys. The moving-eye flight is the judge of the builder.
//!
//! GPU-REQUIRED AND LOCAL, exactly like the moving eye and the picture gate: it needs a working
//! GPU adapter, it is not in `just gate`, and it is run by hand with `just gpu-seam`. Under a plain
//! `cargo test --workspace` this file compiles to zero tests.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::TcpStream;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    DevClusterDown, GPU_SEAM_SLOT, dev_auth_signing_key_hex, dev_roundtrip, devcluster, loopback,
    record_extra_pid, slot_trust_dir, slot_workdir,
};
use vd_devproto::{DevPortScheme, DevRequest, DevResponse, WaitField, WaitOp, WaitPredicate};

const CLIENT_NAME: &str = "gpu-seam";
/// How long to wait for the capture client's dev-control listener to come up.
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const READY_POLL: Duration = Duration::from_millis(200);
/// Step-tick budget for the "delivered frame arrived" wait (about 30 s at the 20 Hz client step).
const CAPTURE_WAIT_TICKS: u64 = 600;
/// THE PROBE STARTS this many seconds after the client does, so one run holds BOTH phases and the
/// two are compared on one binary, one scene and one machine. Fifteen seconds is seven of the
/// probe's own two-second report windows, less the first one this gate throws away.
const PROBE_AT_S: f64 = 15.0;
/// THE PROBE'S OWN REPORT WINDOW, which this gate counts in (`SEAM_REPORT_S` in the client).
const REPORT_S: f64 = 2.0;
/// HOW MANY WINDOWS THE BUSY PHASE IS READ OVER: nine, so the busy phase is the longer of the two
/// and the run is still under a minute. THE RUN IS DERIVED from the two, never typed (review
/// item 10).
const BUSY_WINDOWS: f64 = 9.0;
const RUN_S: f64 = PROBE_AT_S + BUSY_WINDOWS * REPORT_S + REPORT_S;
/// How often the client is looked at while it runs — it must only be seen to be alive.
const POLL: Duration = Duration::from_millis(200);
/// ★ THE FIRST QUIET WINDOW IS THROWN AWAY (review item 5): the client's own start is inside it —
/// it is uploading the whole near ladder, the shaders are compiling, and its worst frame reads
/// three times every later one (MEASURED: 75.2 ms against a steady 20.9). The instrument must
/// answer what a worker COSTS, and a window that holds the start answers something else.
const QUIET_WINDOWS_DROPPED: usize = 1;
/// ★ HOW FAR THE BUSY PHASE'S WORST FRAME MAY STAND ABOVE THE QUIET PHASE'S (review item 11): a
/// fifth. The quiet windows' own worst frames spread by about a twentieth between themselves
/// (MEASURED: 20.77 to 20.89 ms over six windows), so a fifth is four times that spread — wide
/// enough that the run's own noise never reds the gate, narrow enough that one dropped frame in
/// ten does.
const FRAME_SPREAD: f64 = 0.20;

/// Kill the capture client on drop.
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn round_trip(port: u16, req: &DevRequest) -> DevResponse {
    dev_roundtrip(port, req).unwrap_or_else(|e| {
        panic!("dev-control round-trip on {port} failed: {e} — capture client gone? (GPU?)")
    })
}

fn await_listener(port: u16, child: &mut Child) {
    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        if TcpStream::connect(loopback(port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "the --capture client exited early ({status}) before its dev-control listener \
                 came up — this gate requires a working GPU adapter"
            );
        }
        assert!(
            Instant::now() < deadline,
            "dev-control listener on {port} never came up within {READY_TIMEOUT:?}"
        );
        std::thread::sleep(READY_POLL);
    }
}

/// One piped stream of the client, as the reader thread wants it.
fn pipe_of<T: std::io::Read + Send + 'static>(stream: T) -> Box<dyn std::io::Read + Send> {
    Box::new(stream)
}

/// THE LINE WITHOUT ITS COLOURS. The client writes its log with the terminal's own escapes around
/// every field name, so `phase="idle"` never appears as those characters; the escapes come off
/// before anything is read. (Two runs were lost to this before it was written down.)
fn plain(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        // An escape runs to its final letter; everything between it is the terminal's.
        for tail in chars.by_ref() {
            if tail.is_ascii_alphabetic() {
                break;
            }
        }
    }
    out
}

/// One field of a probe line, as the client's log writes it (`name=value`).
fn field(line: &str, name: &str) -> Option<f64> {
    let at = line.find(&format!("{name}="))? + name.len() + 1;
    let rest = &line[at..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() | (c == '.') | (c == '-') | (c == 'e')))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// The probe's lines of one phase, newest last. A line carries its phase by name.
fn phase_lines<'a>(log: &'a str, phase: &str) -> Vec<&'a str> {
    log.lines()
        .filter(|l| l.contains("THE SEAM PROBE") && l.contains(&format!("phase=\"{phase}\"")))
        .collect()
}

/// The mean of a field over a phase's lines, and the largest reading of another.
fn mean_and_max(lines: &[&str], mean_of: &str, max_of: &str) -> (f64, f64) {
    let values: Vec<f64> = lines.iter().filter_map(|l| field(l, mean_of)).collect();
    let mean = if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    };
    let worst = lines
        .iter()
        .filter_map(|l| field(l, max_of))
        .fold(0.0f64, f64::max);
    (mean, worst)
}

/// ★ THE SEAM MEASUREMENT. One capture client; the probe joins it half way through.
#[test]
fn a_worker_thread_may_build_on_the_renderers_own_device_while_it_draws() {
    let _tier = vd_bins::cluster_tier();
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let _ = devcluster(launcher, "down", GPU_SEAM_SLOT);
    let _down = DevClusterDown::new(launcher, GPU_SEAM_SLOT);
    assert!(
        devcluster(launcher, "up", GPU_SEAM_SLOT).success(),
        "dev-cluster up should reach ready and exit 0"
    );

    let ports = DevPortScheme::DEFAULT
        .slot_ports(GPU_SEAM_SLOT)
        .expect("slot ports");
    let gateway = loopback(ports.gateway);
    let devctl = ports.dev_control(0).expect("dev-control port");
    let client_quic = ports.client_quic(0).expect("client-quic port");
    let trust_dir = slot_trust_dir(GPU_SEAM_SLOT);
    let signing_key = dev_auth_signing_key_hex();
    let cwd = slot_workdir(GPU_SEAM_SLOT).join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("make client cwd");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    cmd.current_dir(&cwd)
        .env("VD_AUTH_SIGNING_KEY", &signing_key)
        // The card builds NOTHING here: this gate measures the seam, never the builder.
        .env("VD_TERRAIN_GPU", "0")
        .env("VD_TERRAIN_GPU_SEAM", PROBE_AT_S.to_string())
        .args([
            "--name",
            CLIENT_NAME,
            "--agent-index",
            "0",
            "--gateway",
            &gateway.to_string(),
            "--client-quic",
            &client_quic.to_string(),
            "--trust-dir",
            trust_dir.to_str().expect("utf8 trust dir"),
            "--dev-control",
            &devctl.to_string(),
            "--allow-dev-control",
            "--capture",
        ]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    // The probe writes what it measures to the client's own log, so the log is the instrument.
    // Both streams are PIPED and read on a thread of this gate, which is the one reading that
    // cannot be defeated by a buffer nobody flushed.
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = ChildGuard(cmd.spawn().expect("spawn capture client"));
    let lines: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    for stream in [
        child.0.stdout.take().map(pipe_of),
        child.0.stderr.take().map(pipe_of),
    ]
    .into_iter()
    .flatten()
    {
        let into = std::sync::Arc::clone(&lines);
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::BufReader::new(stream)
                .lines()
                .map_while(Result::ok)
            {
                into.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(plain(&line));
            }
        });
    }
    record_extra_pid(GPU_SEAM_SLOT, child.0.id()).expect("record capture-client pid");
    await_listener(devctl, &mut child.0);

    let live = round_trip(
        devctl,
        &DevRequest::WaitUntil {
            predicate: WaitPredicate {
                field: WaitField::UniverseTick,
                op: WaitOp::Ge,
                value: 1,
            },
            max_ticks: CAPTURE_WAIT_TICKS,
        },
    );
    assert!(
        matches!(live, DevResponse::State { .. }),
        "client should reach a delivered frame, got {live:?}"
    );

    // The run: hold the client for the whole window. The probe idles, then builds, and states
    // the renderer's frames either way into the log.
    let started = Instant::now();
    while started.elapsed().as_secs_f64() < RUN_S {
        std::thread::sleep(POLL);
        // One poll a second, so a client that died is caught while the window runs.
        if let Ok(Some(status)) = child.0.try_wait() {
            panic!("the capture client exited ({status}) during the seam window");
        }
    }
    let log = lines
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .join("\n");
    let idle = phase_lines(&log, "idle");
    let building = phase_lines(&log, "building");
    assert!(
        !idle.is_empty() & !building.is_empty(),
        "the probe did not report both phases ({} idle lines, {} building lines) over {} lines \
         of the client's log",
        idle.len(),
        building.len(),
        log.lines().count()
    );
    for line in idle.iter().chain(building.iter()) {
        eprintln!("gpu_seam: {}", line.trim());
    }
    // ★ The first quiet window holds the client's own start, so it is dropped by NAME, never by
    // picking the number that suits (review item 5).
    assert!(
        idle.len() > QUIET_WINDOWS_DROPPED,
        "the quiet phase must hold more than the {QUIET_WINDOWS_DROPPED} window this gate drops"
    );
    let settled = &idle[QUIET_WINDOWS_DROPPED..];
    let (quiet_fps, quiet_worst) = mean_and_max(settled, "frames_per_s", "worst_frame_ms");
    let (start_fps, start_worst) = mean_and_max(
        &idle[..QUIET_WINDOWS_DROPPED],
        "frames_per_s",
        "worst_frame_ms",
    );
    let (probed_fps, probed_worst) = mean_and_max(&building, "frames_per_s", "worst_frame_ms");
    let last = building.last().expect("a building line");
    eprintln!(
        "gpu_seam: THE SEAM — quiet: {quiet_fps:.1} frames/s, worst frame {quiet_worst:.1} ms \
         ({} windows, the client's own start dropped: {start_fps:.1} frames/s, worst frame \
         {start_worst:.1} ms); with a worker building on the device: {probed_fps:.1} frames/s, \
         worst frame {probed_worst:.1} ms ({} windows)",
        settled.len(),
        building.len()
    );
    eprintln!(
        "gpu_seam: THE PROBE — {:.0} boxes a second, {:.2} ms a box, the worst box {:.2} ms, {:.0} \
         stalls past half a second; the frames held {:.1} % of the quiet rate",
        field(last, "boxes_per_s").unwrap_or(0.0),
        field(last, "mean_ms").unwrap_or(0.0),
        field(last, "worst_ms").unwrap_or(0.0),
        field(last, "stalls").unwrap_or(0.0),
        100.0 * probed_fps / quiet_fps.max(1.0e-9)
    );

    // ★ THE ONLY ASSERTION: THE RENDERER KEPT DRAWING. A hang on a submit or a poll — the fault
    // this gate exists to refuse — stops the frame counter dead, and this catches it. Everything
    // else is a MEASUREMENT the ruling reads, never a threshold argued here.
    assert!(
        probed_fps > 0.0,
        "the renderer stopped drawing while a worker thread built on its device — the seam FAILS"
    );
    assert_eq!(
        field(last, "stalls").unwrap_or(-1.0),
        0.0,
        "a box took past half a second — a submit or a poll STALLED behind the renderer"
    );
    // ★ AND THE FRAMES KEPT THEIR SHAPE (review item 11): the worst single frame while the worker
    // builds stands within the quiet phase's own spread. A hitch of one frame in ten reds this;
    // the run's own noise does not.
    let ceiling = quiet_worst * (1.0 + FRAME_SPREAD);
    assert!(
        probed_worst <= ceiling,
        "the worst frame rose from {quiet_worst:.1} ms to {probed_worst:.1} ms while a worker \
         built on the renderer's own device — past the {FRAME_SPREAD:.0} spread this gate allows"
    );
}
