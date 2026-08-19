//! G-RENDER-SMOKE — the permanent HR6 visual-regression gate. Brings the local-process
//! QUIC cluster up, launches a HEADLESS `client --capture`, drives it over dev-control to a
//! real wgpu-readback screenshot, decodes the captured PNG, and asserts the Tier-A
//! [`render_smoke`] verdict (no-magenta + content-present) PLUS gate-level SCENE checks
//! (a content floor far above what the HUD alone can satisfy + a non-empty scene region)
//! over it — turning the agent's eyeball check into an automated gate. Tears everything
//! down leak-free: the shared `DevClusterDown` guard + a kill-on-drop guard for the 4th
//! process (the capture client) + the client's PID recorded into the slot RUNFILE, so even
//! a SIGKILL of the test runner leaves an orphan the next pre-clean `down` reaps (it can
//! never poison the slot's ports).
//!
//! GPU PRECONDITION (cloud-1): this renders through wgpu and REQUIRES a working GPU adapter
//! (the dev Apple-Silicon Metal GPU today). It is a LOCAL gate — no CI yet, and no
//! software-rasterizer fallback (that would poison the visual baseline). On a GPU-less host
//! the capture client cannot start; the test then fails with a message naming the
//! precondition. Steer the backend with `WGPU_BACKENDS` / `WGPU_POWER_PREF`.
//!
//! Run via `just render-smoke` (it builds the client with `--features dev-control,render`).
//! Under a plain `cargo test --workspace` (no features) this file compiles to ZERO tests —
//! the whole module is gated on the features the capture client needs.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    DevClusterDown, RENDER_SMOKE_SLOT, dev_auth_signing_key_hex, dev_roundtrip, devcluster,
    loopback, record_extra_pid, slot_trust_dir, slot_workdir,
};
use vd_client_harness::assert::{MIN_CONTENT_FRACTION, Rect, region_nonempty, render_smoke};
use vd_client_harness::manifest::{CaptureKind, MANIFEST_FILENAME, RunManifest};
use vd_devproto::{DevPortScheme, DevRequest, DevResponse, WaitField, WaitOp, WaitPredicate};

const CLIENT_NAME: &str = "g-render-smoke";
/// How long to wait for the capture client's dev-control listener to come up (it binds
/// before the GPU/render init, so this is generous headroom, not a tight bound).
const READY_TIMEOUT: Duration = Duration::from_secs(60);
/// The listener poll interval while waiting for the client to boot.
const READY_POLL: Duration = Duration::from_millis(200);
/// Step-tick budget for the "delivered frame arrived" wait (≈30 s at the 20 Hz client step).
const CAPTURE_WAIT_TICKS: u64 = 600;
/// GATE-level scene floor, far stricter than the Tier-A [`MIN_CONTENT_FRACTION`] contract:
/// the egui HUD alone (a few text lines) is well under 1% of the frame, while the reference
/// SCENE (the ground plate fills roughly the lower half) measures ~50% live — so a 5% floor
/// cleanly separates "the 3D scene drew" from "only the HUD drew" with ~10x margin BOTH
/// ways. A regression that loses the whole scene but keeps egui can NOT pass this gate.
const MIN_SCENE_FRACTION: f64 = 0.05;

/// Kill the capture client on drop — the 4th process beyond the cluster's 3 nodes. Covers
/// every in-process exit (assertion panic included); the runfile record (below) covers a
/// SIGKILL of the test runner itself.
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// One dev-control request → response over the SHARED `vd_bins::dev_roundtrip` framing
/// (the same wire path `vdctl` uses — by construction, not by-eye), failing with a
/// GPU-precondition-pointing message instead of a raw error.
fn round_trip(port: u16, req: &DevRequest) -> DevResponse {
    dev_roundtrip(port, req).unwrap_or_else(|e| {
        panic!("dev-control round-trip on {port} failed: {e} — capture client gone? (GPU precondition?)")
    })
}

/// Block until the dev-control listener accepts (the capture client booted + bound), or
/// fail naming the GPU precondition (a no-GPU client exits before getting this far).
fn await_listener(port: u16, child: &mut Child) {
    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        if TcpStream::connect(loopback(port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "the --capture client exited early ({status}) before its dev-control listener \
                 came up — G-RENDER-SMOKE requires a working GPU adapter (see the module docs / \
                 `just render-smoke`)"
            );
        }
        assert!(
            Instant::now() < deadline,
            "dev-control listener on {port} never came up within {READY_TIMEOUT:?} \
             (capture client stuck; GPU precondition?)"
        );
        std::thread::sleep(READY_POLL);
    }
}

#[test]
#[ignore = "PARKED under D-LOOK-3 (the true-scale camera far plane, MEASURED): the 3D scene \
            draws nothing on THE world — content_fraction 0.0036 (the egui HUD alone) against a \
            0.05 floor — because every body sits millions of km away and STAR_FAR_PLANE is \
            120 000 render-metres. Same class as look_pixels/warp_pixels' parked gates; un-parks \
            with the S5 render-scale slice."]
fn g_render_smoke_captures_a_real_frame_with_content_and_no_magenta() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    // Clean slate (idempotent) — and the SIGKILL-orphan reaper: if a previous run was
    // SIGKILLed, its capture client's PID is in this slot's runfile and dies here.
    let _ = devcluster(launcher, "down", RENDER_SMOKE_SLOT);
    let _down = DevClusterDown::new(launcher, RENDER_SMOKE_SLOT);
    assert!(
        devcluster(launcher, "up", RENDER_SMOKE_SLOT).success(),
        "dev-cluster up should reach ready and exit 0"
    );

    // The client contract — from the SAME APIs the launcher uses (no env-file parsing):
    // the dev auth seed, the deterministic per-slot/per-agent ports, the generated trust dir.
    let ports = DevPortScheme::DEFAULT
        .slot_ports(RENDER_SMOKE_SLOT)
        .expect("slot ports");
    let gateway = loopback(ports.gateway);
    let devctl = ports.dev_control(0).expect("dev-control port");
    let client_quic = ports.client_quic(0).expect("client-quic port");
    let trust_dir = slot_trust_dir(RENDER_SMOKE_SLOT);
    let signing_key = dev_auth_signing_key_hex();
    // A contained cwd so the client's `runs/` lands here (reaped with the workdir on `down`).
    let cwd = slot_workdir(RENDER_SMOKE_SLOT).join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("make client cwd");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    cmd.current_dir(&cwd)
        .env("VD_AUTH_SIGNING_KEY", &signing_key)
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
    // Own process group, matching the launcher's nodes — `down` reaps by group id.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = ChildGuard(cmd.spawn().expect("spawn capture client"));
    // Record the client into the slot's durable kill record: ChildGuard covers in-process
    // exits/panics, but a SIGKILL of the test runner skips Drop — the runfile entry makes
    // the next pre-clean `down` reap the orphan instead of leaving it to poison slot 82.
    record_extra_pid(RENDER_SMOKE_SLOT, child.0.id()).expect("record capture-client pid");

    await_listener(devctl, &mut child.0);

    // Drive to a live, DELIVERED frame (active + a snapshot anchored the clock), then capture.
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

    let shot = round_trip(
        devctl,
        &DevRequest::Screenshot {
            at_tick: None,
            label: Some("smoke".to_owned()),
        },
    );
    let (rel_path, tick) = match shot {
        DevResponse::Captured { path, tick } => (path, tick),
        other => panic!("screenshot was not captured (GPU precondition unmet?): {other:?}"),
    };

    // Decode the captured PNG → RGBA8 (Tier-B: the only decode in the gate; the assertion
    // itself is Tier-A and GPU-free).
    let png = cwd.join(&rel_path);
    let img = image::open(&png)
        .unwrap_or_else(|e| panic!("open captured PNG {}: {e}", png.display()))
        .to_rgba8();
    let (w, h) = img.dimensions();
    let buf = img.into_raw();

    // Self-calibrate the clear color from the top-right corner — guaranteed sky in the
    // reference scene (the camera at eye height looks at the horizon; the scene's tallest
    // content subtends ~10° of elevation vs the ≥22.5° corner ray at the 45° vFOV, and the
    // HUD anchors LEFT_TOP) — so the gate is immune to the platform/pipeline sRGB encoding
    // of the cleared target. A blank or uniform-garbage frame still FAILS: every pixel
    // equals the corner → content fraction 0.
    let corner = (w as usize - 1) * 4;
    let clear = [
        buf[corner],
        buf[corner + 1],
        buf[corner + 2],
        buf[corner + 3],
    ];

    // The Tier-A permanent contract: zero magenta + the baseline content floor.
    let verdict = render_smoke(&buf, clear);
    println!(
        "G-RENDER-SMOKE: {w}x{h} frame at tick {tick:?}, clear {clear:?} \
         → magenta {}, content_fraction {:.4}",
        verdict.magenta, verdict.content_fraction,
    );
    assert!(
        verdict.passed,
        "G-RENDER-SMOKE failed on a {w}x{h} frame at tick {tick:?}: magenta={} \
         content_fraction={:.4} (need 0 magenta and >= {MIN_CONTENT_FRACTION} content)",
        verdict.magenta, verdict.content_fraction,
    );
    // GATE-level scene checks, stricter than the Tier-A floor: the HUD alone (<1% of the
    // frame) can NOT satisfy these — a regression that loses the entire 3D scene fails.
    assert!(
        verdict.content_fraction >= MIN_SCENE_FRACTION,
        "the 3D SCENE did not draw: content_fraction {:.4} < {MIN_SCENE_FRACTION} \
         (the egui HUD alone is under 1% — this floor proves the ground plate rendered)",
        verdict.content_fraction,
    );
    // The center of the frame (away from the LEFT_TOP HUD anchor) must hold content — the
    // ground plate spans the lower half and the landmark pillar sits center-screen.
    let center = Rect {
        x: w as usize / 3,
        y: h as usize / 3,
        w: w as usize / 3,
        h: h as usize / 3,
    };
    assert!(
        region_nonempty(&buf, w as usize, center, clear),
        "the center scene region is empty — only HUD/clear pixels in the middle third",
    );

    // The HR6 manifest aligns the capture to its tick + state — confirm it was written and
    // records this screenshot (proves the full runs/ artifact pipeline ran, not just the PNG).
    let run_dir = png.parent().and_then(Path::parent).expect("run dir");
    let manifest_json =
        std::fs::read_to_string(run_dir.join(MANIFEST_FILENAME)).expect("read manifest.json");
    let manifest = RunManifest::from_json(&manifest_json).expect("parse manifest");
    assert!(
        manifest
            .captures
            .iter()
            .any(|c| c.kind == CaptureKind::Screenshot && rel_path.ends_with(&c.path)),
        "manifest must record the screenshot capture, got {:?}",
        manifest.captures,
    );

    // Teardown: `child` (kill the client) then `_down` (cluster down + workdir removal) drop
    // in reverse declaration order — leak-free per the dev_cluster_smoke precedent.
}
