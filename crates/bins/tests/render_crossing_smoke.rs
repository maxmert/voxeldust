//! G-RENDER-CROSSING-SMOKE — the Visual Crossing Playground pixel proof (Slice V4): a real dot
//! WALKS ACROSS a live realm boundary and the PROVEN transfer is captured in actual pixels — the
//! dot's pixels move from box A's screen region to box B's, over the wire, end to end.
//!
//! It stands up the DUAL-shard process cluster (`up --dual`) with an INJECTED walk-into crossing
//! trigger (via the `VD_DEVCLUSTER_BOUNDARIES` launcher hook, in place of the default born-inside
//! shell — which re-homes authority WITHOUT moving the dot and so cannot show a pixel crossing).
//! A headless `client --capture --realm-boxes` renders TWO translucent colored boxes — box A
//! (`System(7)`, at the origin) and box B (`System(8)`, offset on +X). The client's avatar spawns
//! inside box A; `WalkTo` drives it +X across the boundary; the SOURCE's geometric dwell detector
//! fires the crossing AUTONOMOUSLY, the real client stamps the in-band CUT_MARKER (its passive,
//! server-authoritative half of the cut), the directory CAS re-homes authority onto the DEST, and
//! the client auto-acquires the DEST subscription (a server-side route swap it never initiates).
//!
//! CAPTURE + VERDICT. BEFORE (dot in box A) and AFTER (dot in box B) screenshots, each asserting:
//! - STATE (deterministic): `location` flips `System 7` → `System 8` (the client's OWN authoritative
//!   FrameRef re-homed — it received + composited a DEST-authoritative track); `expected_box` flips
//!   A → B (the composited world pos is geometrically inside the other box); and the world pos
//!   ADVANCED on +X well past the crossing point — a pose that ONLY the DEST could have delivered
//!   (the source sub's last pose is behind the crossing; the dot at box B center proves DEST
//!   delivery, a STRONGER non-vacuity gate than a transient both-subs overlap).
//! - PIXELS (corroboration): the dot's projected screen region holds non-clear pixels INSIDE box A's
//!   rectangle BEFORE and box B's rectangle AFTER (the two rectangles are DISJOINT by construction),
//!   and NOT inside box A's rectangle after — the pixels moved. Zero magenta on both frames.
//!
//! GPU PRECONDITION (same as the other visual gates): renders through wgpu, REQUIRES a working GPU
//! adapter, LOCAL-only (no CI, no software-raster fallback). Run via `just render-crossing-smoke`.
//! Under a plain `cargo test --workspace` (no features) this file compiles to ZERO tests. Tier-B
//! process glue (coverage-exempt); the Tier-A verdicts it stands on are 100%-covered by their own
//! unit tests, and the client's cut handling is covered in `vd_client::net`.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::crossing_playground::{self, BOX_B_CENTER};
use vd_bins::{
    DEV, DevClusterDown, admin_get_body, dev_auth_signing_key_hex, dev_roundtrip, devcluster,
    loopback, record_extra_pid, slot_trust_dir, slot_workdir,
};
use vd_client::realm_scene::{BoxShape, RealmScene};
use vd_client_harness::assert::magenta_pixel_count;
use vd_client_harness::camera::{CaptureCamera, ScreenAabb, fit_camera_to_scene};
use vd_client_harness::manifest::{CaptureKind, MANIFEST_FILENAME, RunManifest};
use vd_client_harness::verdict::{
    dot_pixels_within_box_region, expected_box, projected_point_aabb,
};
use vd_client_render::{CAPTURE_H, CAPTURE_W};
use vd_core::glam::{DVec3, I64Vec3};
use vd_core::pose::{FrameRef, LatticePos, RealmId};
use vd_devproto::{DevPortScheme, DevRequest, DevResponse, WORKTREE_SLOT_CEILING};

const CLIENT_NAME: &str = "g-render-crossing";
/// A DISTINCT test-reserved slot, above `WORKTREE_SLOT_CEILING` and clear of the other visual/process
/// gates (render-smoke +18, render-boxes/crossing-e2e +19) so a live cluster can never collide.
const RENDER_CROSSING_SLOT: u16 = WORKTREE_SLOT_CEILING + 22; // 86: G-RENDER-CROSSING-SMOKE

// The playground geometry (box A@SOURCE origin, box B@DEST +X, the walk-into shell trigger + two-box
// scene) is the SINGLE-SOURCED `vd_bins::crossing_playground` fixture set, shared with the human
// `crossing-playground` launcher. The box CENTRES are no longer imported for the projection (slice 5:
// each box's centre is read from the scene and reduced through the one chokepoint); BOX_B_CENTER
// remains as the server-side WalkTo target, which is an absolute world point, not a render-space one.
/// The dot's on-screen world radius for its projected rectangle (brackets the billboard marker).
const DOT_WORLD_RADIUS: f64 = 2.0;
/// The AFTER capture waits for the dot to reach here (well past the ≈ +38.5 crossing point, near box B
/// center): a pose the source sub can NEVER have delivered, so it proves the DEST stream delivered.
const CROSSING_CONFIRMED_X: f64 = 44.0;
/// A generous LOCAL deadline: boot + login + a ≈ 50 m walk at `DEV.move_speed` + the re-home propagation.
const CROSSING_DEADLINE: Duration = Duration::from_secs(70);
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const READY_POLL: Duration = Duration::from_millis(200);
const CROSSING_POLL: Duration = Duration::from_millis(100);

/// Kill the capture client on drop — the 5th process beyond the dual cluster's 4 nodes.
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn round_trip(port: u16, req: &DevRequest) -> DevResponse {
    dev_roundtrip(port, req).unwrap_or_else(|e| {
        panic!("dev-control round-trip on {port} failed: {e} — capture client gone? (GPU precondition?)")
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
                "the --capture client exited early ({status}) before its dev-control listener came \
                 up — G-RENDER-CROSSING-SMOKE requires a working GPU adapter (see the module docs)"
            );
        }
        assert!(
            Instant::now() < deadline,
            "dev-control listener on {port} never came up within {READY_TIMEOUT:?} (GPU precondition?)"
        );
        std::thread::sleep(READY_POLL);
    }
}

/// The orchestrator admin directory as `(key, authority)` rows (diagnosis on a crossing timeout:
/// distinguishes "never fired / source still owns" from "committed on the DEST but the client did
/// not observe it"). Empty on a not-yet-answering endpoint.
fn directory_rows(admin: std::net::SocketAddr) -> Vec<(String, String)> {
    let Some(body) = admin_get_body(admin, "/admin/snapshot", Some(Duration::from_secs(2))) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    value["directory"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    Some((
                        e["key"].as_str()?.to_owned(),
                        e["authority"].as_str()?.to_owned(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The orchestrator's in-flight saga states as `(transfer, state)` — the leg diagnostic: a saga parked
/// in `Promoting` means the DEST promoted but the client-delivery watermark (D-36) starved; a saga stuck
/// in an earlier state (`Cutting`/`Preparing`) means the crossing never got past that leg.
fn saga_states(admin: std::net::SocketAddr) -> Vec<(String, String)> {
    let Some(body) = admin_get_body(admin, "/admin/snapshot", Some(Duration::from_secs(2))) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    value["sagas"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    Some((
                        e["transfer"].as_str()?.to_owned(),
                        e["state"].as_str()?.to_owned(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The current delivered state (a `State` round-trip).
fn poll_state(port: u16) -> vd_devproto::DevState {
    match round_trip(port, &DevRequest::State) {
        DevResponse::State { state } => state,
        other => panic!("expected DevState, got {other:?}"),
    }
}

/// The OWN entity's delivered row (id + composited world pose), or panic naming the state.
fn own_pos(state: &vd_devproto::DevState) -> DVec3 {
    let own = state
        .own_entity
        .as_deref()
        .unwrap_or_else(|| panic!("no own_entity yet in {state:?}"));
    let row = state
        .entities
        .iter()
        .find(|r| r.entity == own)
        .unwrap_or_else(|| panic!("own entity {own} not in the delivered rows: {state:?}"));
    DVec3::new(row.pos[0], row.pos[1], row.pos[2])
}

/// The box's projected screen rectangle via the SAME `fit_camera_to_scene` camera the render framed
/// with — the box center + its bounding-sphere radius projected through the Tier-A `CaptureCamera`.
fn box_screen_aabb(
    scene: &RealmScene,
    camera: &CaptureCamera,
    realm: RealmId,
    origin: LatticePos,
) -> ScreenAabb {
    let rbox = scene.get(realm).expect("box in scene");
    let radius = match rbox.shape {
        BoxShape::Box { half } => half.length(),
        BoxShape::Sphere { r } => r,
    };
    // The centre reduced against the client's render origin through the ONE chokepoint — the same
    // space the pixels are in. This test never spells the subtraction itself.
    projected_point_aabb(camera, rbox.draw_center(origin), radius)
        .expect("the box center projects in front of the camera")
}

/// The render origin THIS state sample was expressed in: the space its reported positions, and the
/// pixels drawn at that moment, both live in. Taken from the sample itself rather than fetched
/// separately, so a position can never be paired with an origin from a different instant.
fn state_origin(state: &vd_devproto::DevState) -> LatticePos {
    let o = state.render_origin;
    LatticePos::at(
        I64Vec3::new(o.cell[0], o.cell[1], o.cell[2]),
        DVec3::new(o.offset[0], o.offset[1], o.offset[2]),
    )
}

/// Decode a captured PNG → (rgba, w, h, self-calibrated clear color from the top-right corner).
fn decode_capture(cwd: &Path, rel_path: &str) -> (Vec<u8>, usize, usize, [u8; 4]) {
    let png = cwd.join(rel_path);
    let img = image::open(&png)
        .unwrap_or_else(|e| panic!("open captured PNG {}: {e}", png.display()))
        .to_rgba8();
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let buf = img.into_raw();
    let corner = (w - 1) * 4;
    let clear = [
        buf[corner],
        buf[corner + 1],
        buf[corner + 2],
        buf[corner + 3],
    ];
    (buf, w, h, clear)
}

/// Screenshot at the current tick, returning the run-relative path (fails loud naming the GPU precondition).
fn screenshot(port: u16, label: &str) -> String {
    match round_trip(
        port,
        &DevRequest::Screenshot {
            at_tick: None,
            label: Some(label.to_owned()),
        },
    ) {
        DevResponse::Captured { path, .. } => path,
        other => {
            panic!("screenshot '{label}' was not captured (GPU precondition unmet?): {other:?}")
        }
    }
}

/// `vd-devcluster up --slot N --dual` with the SOURCE crossing geometry INJECTED via the
/// `VD_DEVCLUSTER_BOUNDARIES` launcher hook (the approved bin dev-config override).
fn up_dual_with_trigger(launcher: &str, slot: u16, boundaries_path: &Path) -> bool {
    Command::new(launcher)
        .args(["up", "--slot", &slot.to_string(), "--dual"])
        .env("VD_DEVCLUSTER_BOUNDARIES", boundaries_path)
        .status()
        .expect("run vd-devcluster up --dual")
        .success()
}

#[test]
fn g_render_crossing_smoke_dot_pixels_move_from_box_a_to_box_b() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let _ = devcluster(launcher, "down", RENDER_CROSSING_SLOT); // clean slate (idempotent)
    let _down = DevClusterDown::new(launcher, RENDER_CROSSING_SLOT);

    let ports = DevPortScheme::DEFAULT
        .slot_ports(RENDER_CROSSING_SLOT)
        .expect("slot ports");
    let gateway = loopback(ports.gateway);
    let devctl = ports.dev_control(0).expect("dev-control port");
    let client_quic = ports.client_quic(0).expect("client-quic port");
    let trust_dir = slot_trust_dir(RENDER_CROSSING_SLOT);
    let signing_key = dev_auth_signing_key_hex();
    let cwd = slot_workdir(RENDER_CROSSING_SLOT).join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("make client cwd");

    // Two files (single-sourcing is impossible for a crossing — see crossing_playground::scene): the SHARD
    // trigger (injected via the launcher hook) and the CLIENT two-box scene.
    let trigger_path = cwd.join("trigger.json");
    std::fs::write(
        &trigger_path,
        serde_json::to_string(&crossing_playground::trigger(&DEV)).expect("serialize trigger"),
    )
    .expect("write trigger.json");
    let scene_boundaries = crossing_playground::scene(&DEV);
    let scene = RealmScene::from_boundaries(&scene_boundaries).expect("the scene projects");
    let boxes_json = cwd.join("boxes.json");
    std::fs::write(
        &boxes_json,
        serde_json::to_string(&scene_boundaries).expect("serialize boxes.json"),
    )
    .expect("write boxes.json");

    // Bring up the dual cluster with the injected walk-into trigger (C1 both-realms gate → exit 0).
    assert!(
        up_dual_with_trigger(launcher, RENDER_CROSSING_SLOT, &trigger_path),
        "up --dual with the injected trigger must reach the C1 both-realms ready gate and exit 0",
    );

    // Launch the headless capture client with the two-box scene (the 5th process).
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
            "--realm-boxes",
            boxes_json.to_str().expect("utf8 boxes.json path"),
        ]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = ChildGuard(cmd.spawn().expect("spawn capture client"));
    record_extra_pid(RENDER_CROSSING_SLOT, child.0.id()).expect("record capture-client pid");
    await_listener(devctl, &mut child.0);

    // ---- BEFORE: the dot spawns inside box A (System 7). Wait for a delivered frame, then capture. --
    let system_a = FrameRef::SystemSpace {
        system_seed: DEV.realm_seed,
    }
    .label();
    let system_b = FrameRef::SystemSpace {
        system_seed: DEV.realm_seed_b,
    }
    .label();
    let before_state = {
        let deadline = Instant::now() + CROSSING_DEADLINE;
        loop {
            let s = poll_state(devctl);
            if s.own_entity.is_some()
                && !s.entities.is_empty()
                && s.location.as_deref() == Some(system_a.as_str())
            {
                break s;
            }
            assert!(
                Instant::now() < deadline,
                "the dot never delivered inside box A (System 7): {s:?}"
            );
            std::thread::sleep(CROSSING_POLL);
        }
    };
    let pos_before = own_pos(&before_state);

    // THE ONE SPACE (slice 5). Everything below — the fitted camera, the two box rectangles, the
    // containment verdicts — is computed against the client's LIVE render origin, the same origin it
    // reduced `pos_before` with and drew its pixels with. Read from the client, never assumed to be
    // zero: the scene this test holds is ABSOLUTE, the positions the client reports are RELATIVE, and
    // comparing the two without the origin is only accidentally right while the origin happens to be
    // zero. Fitted HERE (after the client is up) rather than before boot, because that is the first
    // moment the origin is knowable.
    let origin = state_origin(&before_state);
    let camera = fit_camera_to_scene(&scene, origin, CAPTURE_W as usize, CAPTURE_H as usize)
        .expect("the two-box scene frames to a camera");
    let box_a_region = box_screen_aabb(&scene, &camera, RealmId::System(DEV.realm_seed), origin);
    let box_b_region = box_screen_aabb(&scene, &camera, RealmId::System(DEV.realm_seed_b), origin);

    // S6 (pure renderer): the client is NODE-AGNOSTIC — it no longer tracks an authoritative sub.
    // The client-tier re-home proof is the OWN entity's LOCATION label flipping realms (source→dest),
    // which is the own entity's delivered authoritative FrameRef re-expressed by the server.
    let location_before = before_state.location.clone();
    assert_eq!(
        expected_box(&scene, origin, pos_before),
        Some(RealmId::System(DEV.realm_seed)),
        "BEFORE: the dot's world pos {pos_before} must be geometrically inside box A",
    );
    let before_shot = screenshot(devctl, "before");
    let (before_rgba, w, h, before_clear) = decode_capture(&cwd, &before_shot);
    let dot_region_before = projected_point_aabb(&camera, pos_before, DOT_WORLD_RADIUS)
        .expect("the dot projects in front of the camera (before)");
    assert_eq!(
        magenta_pixel_count(&before_rgba),
        0,
        "BEFORE frame must be magenta-free"
    );
    assert!(
        dot_pixels_within_box_region(
            &before_rgba,
            w,
            h,
            before_clear,
            dot_region_before,
            box_a_region
        ),
        "BEFORE: the dot's pixels must fall inside box A's projected region",
    );

    // ---- DRIVE: WalkTo box B center. The dot walks +X across the trigger; the SOURCE dwell detector
    // fires the crossing AUTONOMOUSLY, the real client stamps the CUT_MARKER, and authority re-homes. --
    let admin = loopback(ports.admin);
    let walk = round_trip(
        devctl,
        &DevRequest::WalkTo {
            target: [BOX_B_CENTER.x, BOX_B_CENTER.y, BOX_B_CENTER.z],
            arrive_epsilon: 3.0,
            max_ticks: 6000,
        },
    );
    assert!(
        matches!(
            walk,
            DevResponse::State { .. } | DevResponse::Timeout { .. }
        ),
        "WalkTo should return a delivered state, got {walk:?}",
    );
    // STOP the dot at box B: WalkTo's last Move is sticky (the client keeps applying it), so without a
    // zero-Move the dot walks straight through box B. It stops INSIDE box B's SOI, so the saga (which
    // needs the subject in-band) can still complete.
    let _ = round_trip(
        devctl,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );

    // ---- WAIT for the CLIENT-observed re-home: location flips to System 8 AND the dot advanced to box B
    // center — a pose ONLY the DEST could have delivered (the source sub is behind the crossing point). --
    let after_state = {
        let deadline = Instant::now() + CROSSING_DEADLINE;
        loop {
            let s = poll_state(devctl);
            // The client-observed re-home, on THREE signals: (1) the HUD `location` label reads the DEST
            // realm ("System 8") — the frame-rebinding re-expresses the crossed pose into System(8)'s frame,
            // so the authoritative FrameRef (and its label) flips; (2) the composited authoritative world pos
            // is geometrically inside box B; (3) it advanced past the crossing point (a pose only the DEST
            // could have delivered). The label is now a HARD gate, not a lagging cosmetic signal.
            if s.location.as_deref() == Some(system_b.as_str())
                && expected_box(&scene, state_origin(&s), own_pos(&s))
                    == Some(RealmId::System(DEV.realm_seed_b))
                && own_pos(&s).x >= CROSSING_CONFIRMED_X
            {
                break s;
            }
            if Instant::now() >= deadline {
                // CLIENT-SIDE leg diagnostic (printed BEFORE the panic): did the DEST sub ever open
                // (own row's authoritative_sub changed off the login sub 0)? is the client receiving
                // frames (snapshots_applied)? authoritative_sub still 0 ⇒ the client never got the DEST
                // sub (SubscriptionReady never reached it: promote deferred, or the reply was lost).
                eprintln!(
                    "CLIENT LEG: phase={:?} snapshots_applied={} entities={:?}",
                    s.phase,
                    s.snapshots_applied,
                    s.entities
                        .iter()
                        .map(|r| (r.entity.clone(), r.authoritative_sub, r.pos[0]))
                        .collect::<Vec<_>>(),
                );
                panic!(
                    "the dot never re-homed to box B (System 8) with pos.x >= {CROSSING_CONFIRMED_X}: \
                     location={:?} pos={:?}. admin directory (ent-* @ shard:node-4 = DEST owns ⇒ the \
                     crossing COMMITTED but the client did not observe it): {:?}. SAGA STATES \
                     (Promoting parked = D-36 watermark; empty/Done = a different leg): {:?}",
                    s.location,
                    s.own_entity.as_ref().map(|_| own_pos(&s)),
                    directory_rows(admin),
                    saga_states(admin),
                );
            }
            std::thread::sleep(CROSSING_POLL);
        }
    };
    let pos_after = own_pos(&after_state);
    // THE SHARED CAMERA'S PREMISE, made loud. The one fitted camera and the two box rectangles above
    // were built against the BEFORE origin, and every check below reuses them for the AFTER frame.
    // That is only sound while the crossing does not re-pin the client's render origin. It does not
    // today; when re-anchoring on a crossing lands, this fires, and the fix is to fit a camera and
    // rectangles per frame rather than to relax this line.
    assert_eq!(
        state_origin(&after_state),
        origin,
        "the render origin moved across the crossing — the shared camera and both box rectangles \
         were fitted in the BEFORE space, so every geometric check below is comparing two different \
         spaces. Fit the camera and the rectangles PER FRAME instead of loosening this assertion.",
    );

    // ---- AFTER: capture the dot inside box B. ---------------------------------------------------------
    let after_shot = screenshot(devctl, "after");
    let (after_rgba, aw, ah, after_clear) = decode_capture(&cwd, &after_shot);
    let dot_region_after = projected_point_aabb(&camera, pos_after, DOT_WORLD_RADIUS)
        .expect("the dot projects in front of the camera (after)");

    let location_after = after_state.location.clone();
    let final_directory = directory_rows(admin);
    let dest_owns = final_directory
        .iter()
        .any(|(key, authority)| key.starts_with("ent-") && authority == "shard:node-4");
    println!(
        "G-RENDER-CROSSING-SMOKE: {w}x{h} · box A screen {:?} box B screen {:?} · pos {pos_before} → \
         {pos_after} · location {location_before:?} → {location_after:?} · DEST-owns={dest_owns} \
         (the location label flipped to the DEST realm — the frame-rebinding re-expresses the pose)",
        (box_a_region.min.x as i32, box_a_region.max.x as i32),
        (box_b_region.min.x as i32, box_b_region.max.x as i32),
    );

    // NON-VACUITY: this must be a REAL transfer, not just the dot walking into box B's region.
    // (a) the directory CAS committed authority onto the DEST; (b) the CLIENT (a NODE-AGNOSTIC pure
    // renderer, S6) observed the re-home — the own dot's LOCATION label flipped realms (source→dest),
    // i.e. its delivered authoritative FrameRef re-homed. This is the client-tier proof (not pixels),
    // and it needs NO node awareness (the client never learns which shard owns the avatar).
    assert!(
        dest_owns,
        "the DEST (shard:node-4) must OWN the re-homed dot — the transfer committed, not just a walk: {final_directory:?}",
    );
    assert_ne!(
        location_after, location_before,
        "the client must have observed the authority re-home (own dot's LOCATION label flips off \
         the source realm {location_before:?} onto the DEST realm) — the client-tier proof, not just pixels",
    );
    assert_eq!(
        location_after.as_deref(),
        Some(system_b.as_str()),
        "the client's location label reads the DEST realm (System 8) after the re-home",
    );

    // STATE: the composited world pos is now geometrically inside box B, and advanced far on +X.
    assert_eq!(
        expected_box(&scene, origin, pos_after),
        Some(RealmId::System(DEV.realm_seed_b)),
        "AFTER: the dot's world pos {pos_after} must be geometrically inside box B",
    );
    assert!(
        pos_after.x - pos_before.x > 30.0,
        "AFTER: the dot must have physically advanced far on +X ({} → {})",
        pos_before.x,
        pos_after.x,
    );
    // PIXELS: the dot's pixels are now inside box B's region and NO LONGER inside box A's — they moved.
    assert_eq!(
        magenta_pixel_count(&after_rgba),
        0,
        "AFTER frame must be magenta-free"
    );
    assert!(
        dot_pixels_within_box_region(
            &after_rgba,
            aw,
            ah,
            after_clear,
            dot_region_after,
            box_b_region
        ),
        "AFTER: the dot's pixels must fall inside box B's projected region",
    );
    assert!(
        !dot_pixels_within_box_region(
            &after_rgba,
            aw,
            ah,
            after_clear,
            dot_region_after,
            box_a_region
        ),
        "AFTER: the dot's pixels must NO LONGER fall inside box A's region (they moved A→B)",
    );

    // HR6: both screenshots recorded in the run manifest (the capture pipeline ran end to end).
    let run_dir = cwd.join(&after_shot);
    let run_dir = run_dir.parent().and_then(Path::parent).expect("run dir");
    let manifest = RunManifest::from_json(
        &std::fs::read_to_string(run_dir.join(MANIFEST_FILENAME)).expect("read manifest.json"),
    )
    .expect("parse manifest");
    for (label, shot) in [("before", &before_shot), ("after", &after_shot)] {
        assert!(
            manifest
                .captures
                .iter()
                .any(|c| c.kind == CaptureKind::Screenshot && shot.ends_with(&c.path)),
            "manifest must record the '{label}' screenshot, got {:?}",
            manifest.captures,
        );
    }
}
