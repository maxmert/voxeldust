//! G-RENDER-SMOKE — the permanent HR6 visual-regression gate. Brings the local-process
//! QUIC cluster up, launches a HEADLESS `client --capture`, drives it over dev-control to a
//! real wgpu-readback screenshot, decodes the captured PNG, and asserts the Tier-A
//! [`render_smoke`] verdict (no-magenta + content-present) over it — turning the agent's
//! eyeball check into an automated gate. Tears everything down leak-free (the
//! `dev_cluster_smoke` `DownGuard` precedent + a kill-on-drop guard for the 4th process,
//! the capture client, which the cluster's PID reaper does not own).
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

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{dev_auth_signing_key_hex, loopback};
use vd_client_harness::assert::{MIN_CONTENT_FRACTION, render_smoke};
use vd_client_harness::manifest::{CaptureKind, RunManifest};
use vd_devproto::{
    DevPortScheme, DevRequest, DevResponse, WORKTREE_SLOT_CEILING, WaitField, WaitOp, WaitPredicate,
};

/// Test-reserved slot ABOVE the worktree auto-derivation ceiling (so it can never coincide
/// with a developer's running `dev-cluster.sh up`), DISTINCT from `dev_cluster_smoke`'s
/// 80/81 so a parallel run never collides.
const RENDER_SMOKE_SLOT: u16 = WORKTREE_SLOT_CEILING + 18; // 82
const CLIENT_NAME: &str = "g-render-smoke";
/// How long to wait for the capture client's dev-control listener to come up (it binds
/// before the GPU/render init, so this is generous headroom, not a tight bound).
const READY_TIMEOUT: Duration = Duration::from_secs(60);
/// Step-tick budget for the "delivered frame arrived" wait (≈30 s at the 20 Hz client step).
const CAPTURE_WAIT_TICKS: u64 = 600;

fn workdir(slot: u16) -> PathBuf {
    std::env::temp_dir()
        .join("vd-devcluster")
        .join(format!("slot-{slot}"))
}

fn devcluster(sub: &str, slot: u16) -> std::process::ExitStatus {
    Command::new(env!("CARGO_BIN_EXE_vd-devcluster"))
        .args([sub, "--slot", &slot.to_string()])
        .status()
        .unwrap_or_else(|e| panic!("vd-devcluster {sub}: {e}"))
}

/// Tear the cluster down on drop (even on panic) — the `dev_cluster_smoke` precedent.
struct DownGuard(u16);
impl Drop for DownGuard {
    fn drop(&mut self) {
        let _ = devcluster("down", self.0);
    }
}

/// Kill the capture client on drop — the 4th process beyond the cluster's 3 nodes (the
/// `down` PID reaper only owns the recorded node PIDs, not this child).
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// One dev-control request → response (a fresh connection per request, like `vdctl`).
fn round_trip(port: u16, req: &DevRequest) -> DevResponse {
    let stream = TcpStream::connect(loopback(port)).unwrap_or_else(|e| {
        panic!("dev-control connect {port}: {e} — capture client gone? (GPU precondition?)")
    });
    let mut writer = stream.try_clone().expect("clone stream");
    let mut line = serde_json::to_string(req).expect("encode request");
    line.push('\n');
    writer.write_all(line.as_bytes()).expect("write request");
    writer.flush().ok();
    let mut reply = String::new();
    BufReader::new(stream)
        .read_line(&mut reply)
        .expect("read response");
    serde_json::from_str(reply.trim()).expect("decode response")
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
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[test]
fn g_render_smoke_captures_a_real_frame_with_content_and_no_magenta() {
    let _ = devcluster("down", RENDER_SMOKE_SLOT); // clean slate (idempotent)
    let _down = DownGuard(RENDER_SMOKE_SLOT);
    assert!(
        devcluster("up", RENDER_SMOKE_SLOT).success(),
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
    let trust_dir = workdir(RENDER_SMOKE_SLOT).join("trust");
    let signing_key = dev_auth_signing_key_hex();
    // A contained cwd so the client's `runs/` lands here (reaped with the workdir on `down`).
    let cwd = workdir(RENDER_SMOKE_SLOT).join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("make client cwd");

    let mut child = ChildGuard(
        Command::new(env!("CARGO_BIN_EXE_client"))
            .current_dir(&cwd)
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
            ])
            .spawn()
            .expect("spawn capture client"),
    );

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

    // Self-calibrate the clear color from the top-right corner (reference-scene sky), so the
    // gate is immune to the platform/pipeline sRGB encoding of the cleared target. A blank or
    // uniform-garbage frame still FAILS: every pixel equals the corner → content fraction 0.
    let corner = (w as usize - 1) * 4;
    let clear = [
        buf[corner],
        buf[corner + 1],
        buf[corner + 2],
        buf[corner + 3],
    ];

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

    // The HR6 manifest aligns the capture to its tick + state — confirm it was written and
    // records this screenshot (proves the full runs/ artifact pipeline ran, not just the PNG).
    let run_dir = png.parent().and_then(Path::parent).expect("run dir");
    let manifest_json =
        std::fs::read_to_string(run_dir.join("manifest.json")).expect("read manifest.json");
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
