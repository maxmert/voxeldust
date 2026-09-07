//! G-STAR-SKY-PIXELS (S11) — THE GALAXY IS ON SCREEN, where the seed says it is.
//!
//! Every other link in the star lane is already proven, and each by its own test: the catalogue
//! crosses the wire (`process_parity`), the client holds every star its shard states
//! (`process_parity`'s census gate), the anchor is found in the catalogue itself and the subtraction
//! is exact (`render_snapshot`'s unit tests), and the cloud is four vertices per star sharing one
//! mesh (`StarCloud`'s unit tests).
//!
//! ★ WHAT WAS NEVER PROVEN IS THE LAST LINK: that a pixel lights up. A shader that compiles is not a
//! shader that draws — the day this was written, a clean compile hid a startup panic AND a camera
//! that pointed the scene off the edge of the frame. Only running the real client on the real GPU
//! found either.
//!
//! ★ THE GATE DERIVES ITS OWN EXPECTATION. The client reports the sky it HOLDS
//! (`DevState.sky = (generation, stars)`); this test computes independently, from the world seed,
//! WHERE each star must land. Asserting the client's drawn count against the client's held count
//! would be checking one source against itself, and two numbers from one source always agree.
//!
//! ★ AND THE TWO COUNTS DIFFER, WHICH IS THE POINT. The catalogue contains the observer's OWN
//! system, and that star is never drawn — you are standing inside it, and a point of light at zero
//! distance sits on the camera. So the drawn set is `stars - 1`. A self-reported count would have
//! papered over exactly that.
//!
//! SCALE, STATED HONESTLY: THE world holds three star systems, so with the observer inside one this
//! gate proves TWO points of light. The 150,000-star census is S12. This is the chain's last link,
//! not a crowd.
//!
//! GPU PRECONDITION (as `render_smoke`): renders through wgpu and REQUIRES a working GPU adapter.
//! A LOCAL gate — no CI, and no software-rasterizer fallback, which would poison the baseline.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    DEV, DevClusterDown, boot_regions_and_movers, dev_auth_signing_key_hex, dev_roundtrip,
    devcluster, loopback, record_extra_pid, slot_trust_dir, slot_workdir, star_catalogue_for_boot,
    world_roster,
};
use vd_client_harness::assert::{MAGENTA, magenta_pixel_count};
use vd_client_harness::camera::ScreenAabb;
use vd_client_harness::manifest::{CaptureKind, MANIFEST_FILENAME, RunManifest};
use vd_client_harness::verdict::projected_point_aabb;
use vd_client_render::{CAPTURE_H, CAPTURE_W};
use vd_core::glam::DVec3;
use vd_devproto::WORKTREE_SLOT_CEILING;
use vd_devproto::{DevPortScheme, DevRequest, DevResponse, WaitField, WaitOp, WaitPredicate};

const STAR_SKY_SLOT: u16 = WORKTREE_SLOT_CEILING + 21; // 85: G-STAR-SKY-PIXELS
const CLIENT_NAME: &str = "g-star-sky-pixels";
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const READY_POLL: Duration = Duration::from_millis(200);
const CAPTURE_WAIT_TICKS: u64 = 600;
/// How long to wait for the sky to arrive and complete, after the scene is live. The catalogue is
/// request-driven and the gateway asks on the liveness beat, so this covers a beat period with
/// headroom — never a sleep literal in the body.
const SKY_TIMEOUT: Duration = Duration::from_secs(30);
const SKY_POLL: Duration = Duration::from_millis(100);

/// The search half-size, in pixels, around a star's projected centre.
///
/// ★ WHY A SEARCH AND NOT AN EXACT PIXEL. A star sits at the APPARENT-SIZE FLOOR — about one pixel —
/// so its rectangle is roughly a single sample, and a one-pixel target is unforgiving of any
/// disagreement between the projection this test computes and the sprite the shader places. Sources
/// of legitimate disagreement: the state is sampled one frame before the screenshot, and the camera
/// refits per frame over a scene whose planets are moving.
///
/// This is the SAME allowance the box gate makes for the same reason (`RIM_PLANET_DRIFT_PX = 12`).
/// It is a bracket around a computed position — NOT a licence to search the frame: at 12 px this
/// covers 0.03 % of a 1284x720 image, so a hit is still a statement about WHERE the star drew.
const STAR_PROBE_HALF_PX: f64 = 12.0;
/// Two polls this far apart with the same delivered pose mean the avatar is at rest: longer than the
/// interpolation buffer, so a look input still in flight cannot hide between them.
const POSE_SETTLE_POLL: Duration = Duration::from_millis(300);
const POSE_SETTLE_TIMEOUT: Duration = Duration::from_secs(10);
/// A star bright enough to be a multi-pixel blob on its own — rare in a sky of 233,220, so its
/// presence or absence at the predicted place is a sharp verdict where a faint one-pixel star is not.
const BRIGHT_STAR_AMPLITUDE: f64 = 0.3;

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn round_trip(port: u16, req: &DevRequest) -> DevResponse {
    dev_roundtrip(port, req).unwrap_or_else(|e| panic!("dev-control {req:?}: {e}"))
}

fn await_listener(port: u16, child: &mut Child) {
    let deadline = Instant::now() + READY_TIMEOUT;
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "the --capture client exited early ({status}) before its dev-control listener came \
                 up — G-STAR-SKY-PIXELS requires a working GPU adapter"
            );
        }
        if TcpStream::connect(loopback(port)).is_ok() {
            return;
        }
        std::thread::sleep(READY_POLL);
    }
    panic!("the capture client's dev-control listener never came up within {READY_TIMEOUT:?}");
}

/// Non-clear pixels inside `region` (top-left origin).
fn nonclear_in_region(rgba: &[u8], w: usize, h: usize, clear: [u8; 4], region: ScreenAabb) -> u64 {
    let x0 = region.min.x.floor().clamp(0.0, w as f64) as usize;
    let y0 = region.min.y.floor().clamp(0.0, h as f64) as usize;
    let x1 = (region.max.x.ceil().clamp(0.0, w as f64) as usize).max(x0);
    let y1 = (region.max.y.ceil().clamp(0.0, h as f64) as usize).max(y0);
    let mut count = 0u64;
    for py in y0..y1 {
        for px in x0..x1 {
            let idx = (py * w + px) * 4;
            if let Some(pixel) = rgba.get(idx..idx + 4)
                && pixel != clear.as_slice()
            {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn g_star_sky_pixels_draws_the_galaxy_where_the_seed_puts_it() {
    let _tier = vd_bins::cluster_tier();
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let _ = devcluster(launcher, "down", STAR_SKY_SLOT);
    let _down = DevClusterDown::new(launcher, STAR_SKY_SLOT);
    // ★ A DUAL CLUSTER, AND THAT IS NOT AN INCIDENTAL CHOICE (S11, measured 2026-08-27).
    //
    // A shard folds its sky from the realms IT booted. On a SINGLE cluster the only shard is the home
    // star system, so its catalogue holds exactly ONE star — its own — and the drawable set is
    // `1 - 1 = 0`. There is literally nothing to draw, and this gate failed its own precondition
    // proving it.
    //
    // `--dual` boots the GALAXY shard, which parents every star system in the world. That is the only
    // cluster shape in which another star exists to be seen.
    //
    // ⚠ THIS IS THE OPEN S12 QUESTION, MADE CONCRETE: a player standing in a star system must see the
    // whole galaxy, and the shard they are subscribed to does not hold it. The ledger names the three
    // candidate answers and picks none. This gate now depends on that question being answered — if
    // the chosen answer changes which shard states the sky, this setup changes with it.
    assert!(
        Command::new(launcher)
            .args(["up", "--slot", &STAR_SKY_SLOT.to_string(), "--dual"])
            .status()
            .expect("run vd-devcluster up --dual")
            .success(),
        "dev-cluster up --dual should reach ready and exit 0"
    );

    let ports = DevPortScheme::DEFAULT
        .slot_ports(STAR_SKY_SLOT)
        .expect("slot ports");
    let gateway = loopback(ports.gateway);
    let devctl = ports.dev_control(0).expect("dev-control port");
    let client_quic = ports.client_quic(0).expect("client-quic port");
    let trust_dir = slot_trust_dir(STAR_SKY_SLOT);
    let signing_key = dev_auth_signing_key_hex();
    let cwd = slot_workdir(STAR_SKY_SLOT).join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("make client cwd");

    // ★ THE EXPECTATION, FROM THE WORLD ITSELF — derived exactly as the SHARD derives its sky, so
    // this gate measures the drawing and not a disagreement about which stars exist.
    // The observer stands in the home system; the SKY is stated by the GALAXY shard, which parents
    // every system. So the expectation is derived from the galaxy's forest, exactly as that shard
    // derives it at boot.
    // THE world's own names for both realms — never constructed here. `world_roster` derives the
    // login realm and its parent from the regions themselves, which is what the cluster boots.
    let roster = world_roster(&DEV);
    let own = roster.home;
    let galaxy = roster.galaxy;
    let held: std::collections::BTreeSet<vd_core::pose::RealmId> = [galaxy].into_iter().collect();
    let (shard_regions, _) = boot_regions_and_movers(
        DEV.universe_seed,
        &held,
        galaxy,
        DEV.move_speed,
        DEV.tick_dt,
    );
    let (rows, catalogue_generation, _) = star_catalogue_for_boot(
        DEV.universe_seed,
        DEV.move_speed,
        DEV.tick_dt,
        &shard_regions,
    );
    println!(
        "  the gate's catalogue: {} stars, generation {catalogue_generation:#x}; first rows {:?}",
        rows.len(),
        rows.iter()
            .take(3)
            .map(|r| (r.realm, r.cell))
            .collect::<Vec<_>>()
    );
    assert!(
        rows.len() >= 2,
        "this world must state at least two stars, or the gate proves nothing: {}",
        rows.len()
    );

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
            // ★ PILOT VIEW, and it is required rather than preferred (S11, measured 2026-08-27).
            //
            // The diagnostic capture camera FITS the local scene — the home system's shell and its
            // planets. The stars are 4.5 light years away in whatever direction the seed put them, so
            // they land outside the frame: MEASURED at (1216, -2114) and (-638, -31) on a 1284x720
            // frame, both IN FRONT of the eye and neither on it. Nothing was wrong with the drawing;
            // the camera was not looking at the sky.
            //
            // Pilot view renders from the avatar's own eye along its delivered facing, which is a
            // direction this test can then AIM.
            "--capture-pilot",
        ]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = ChildGuard(cmd.spawn().expect("spawn capture client"));
    record_extra_pid(STAR_SKY_SLOT, child.0.id()).expect("record capture-client pid");
    await_listener(devctl, &mut child.0);

    // THE SCENE SIGNAL — the composed fold's own arrival, the same one the box gate waits on. The
    // camera reconstruction needs a drawn scene to frame.
    let live = round_trip(
        devctl,
        &DevRequest::WaitUntil {
            predicate: WaitPredicate {
                field: WaitField::RealmFramesApplied,
                op: WaitOp::Ge,
                value: 1,
            },
            max_ticks: CAPTURE_WAIT_TICKS,
        },
    );
    assert!(
        matches!(live, DevResponse::State { .. }),
        "client should reach a delivered composed fold, got {live:?}"
    );

    // THE SKY SIGNAL — poll until the client reports a WHOLE catalogue. It arrives on the liveness
    // beat's own cadence, which is slower than the scene, so waiting on the scene alone would
    // photograph a client that holds no sky yet and blame the renderer.
    let deadline = Instant::now() + SKY_TIMEOUT;
    let state = loop {
        let DevResponse::State { state } = round_trip(devctl, &DevRequest::State) else {
            panic!("dev-control State should answer with a state");
        };
        if state.sky.is_some() {
            break state;
        }
        assert!(
            Instant::now() < deadline,
            "the client never assembled a whole sky within {SKY_TIMEOUT:?} — the catalogue is \
             request-driven, so this means the ask, the answer or the assembly failed"
        );
        std::thread::sleep(SKY_POLL);
    };

    let (generation, held_stars) = state.sky.expect("the loop broke on Some");
    assert!(generation != 0, "the generation is folded from content");
    println!(
        "  the client holds: {held_stars} stars, generation {generation:#x}, anchor {:?}",
        state.sky_anchor
    );
    // ★ THE SAME SKY, BY CONTENT (2026-09-06): the gate's expectation and the client's held sky must
    // be the SAME catalogue — one count proves nothing (two different skies of equal size fooled
    // this gate: every star "in view" was compared against a picture of other stars). The
    // generation is a hash of the rows, so equality here is equality of every cell and look.
    assert_eq!(
        generation, catalogue_generation,
        "the client holds a DIFFERENT sky than the gate derived from the world (same seed, same \
         config?) — the pixel verdict below would compare unrelated stars"
    );
    assert_eq!(
        held_stars as usize,
        rows.len(),
        "the client holds exactly the stars this shard states"
    );

    // WHERE EACH STAR MUST LAND, computed BEFORE aiming — the same exact integer subtraction the
    // client makes, recomputed here rather than read back from it.

    // WHERE EACH STAR MUST LAND — the same exact integer subtraction the client makes, recomputed
    // here rather than read back from it.
    // ★ THE ORACLE PLACES THE SKY THE WAY THE CLIENT DOES (owner ruling 2026-09-02 R1): every star
    // as metres from the SKY ANCHOR — the observer's origin realm in the galaxy's frame, which at
    // login is the home system's own cell (a generated system is exactly cell-aligned) — minus the
    // eye's own offset inside that realm, the spawn clearing. The observer's own star is IN the
    // drawn set: a point of light at its true place, not an exclusion.
    let anchor = rows
        .iter()
        .find(|r| r.realm == own)
        .map(|r| r.cell)
        .expect("the observer's own system is IN the catalogue");
    let spawn_m =
        vd_bins::boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt).default_home_offset_m();
    let edge_m = vd_core::pose::Tier::Galaxy.cell_edge_m();
    // Each star with its LOOK (class + luminosity): the same two numbers the renderer's star law
    // reads, so the gate can say which stars the law culls as too faint to paint alone.
    let drawn: Vec<(vd_core::pose::RealmId, DVec3, u8, f64)> = rows
        .iter()
        .map(|r| {
            (
                r.realm,
                DVec3::new(
                    (i128::from(r.cell.x) - i128::from(anchor.x)) as f64 * edge_m,
                    (i128::from(r.cell.y) - i128::from(anchor.y)) as f64 * edge_m,
                    (i128::from(r.cell.z) - i128::from(anchor.z)) as f64 * edge_m,
                ) - spawn_m,
                r.class_code,
                r.luma_lsun,
            )
        })
        .collect();
    assert_eq!(
        drawn.len(),
        rows.len(),
        "every star of the catalogue is drawn"
    );
    // The client says so too: the drawn count is the whole census, and the anchor it was placed by
    // is the home system's cell — measured against the world, not the client's own arithmetic.
    let state = {
        let started = Instant::now();
        loop {
            let st = vd_bins::pixel::poll(devctl);
            if st.stars_drawn > 0 {
                break st;
            }
            assert!(
                started.elapsed() < SKY_TIMEOUT,
                "the sky was held ({held_stars} stars) and never drawn — the anchor never arrived \
                 (sky_anchor = {:?}) or the renderer never placed it",
                st.sky_anchor
            );
            std::thread::sleep(SKY_POLL);
        }
    };
    assert_eq!(
        state.stars_drawn as usize,
        rows.len(),
        "the whole catalogue is on screen"
    );
    assert_eq!(
        state.sky_anchor.map(|a| a.map(f64::round)),
        Some([
            (anchor.x as f64 * edge_m).round(),
            (anchor.y as f64 * edge_m).round(),
            (anchor.z as f64 * edge_m).round(),
        ]),
        "the anchor the client holds is the home system's own galaxy cell, in metres"
    );

    // Aim at a star that is NOT the one we stand in — a point of light at zero distance cannot be
    // looked at.
    let (target_realm, target_pos, _, _) = *drawn
        .iter()
        .find(|(realm, _, _, _)| *realm != own)
        .expect("at least one other star");
    let aim = round_trip(
        devctl,
        &DevRequest::LookAt {
            target: target_pos.to_array(),
            align_epsilon: 0.05,
            max_ticks: 900,
        },
    );
    assert!(
        matches!(aim, DevResponse::State { .. }),
        "the avatar never turned toward {target_realm:?} at {target_pos:?}: {aim:?}"
    );
    // ★ WAIT FOR THE DELIVERED POSE TO SETTLE (2026-09-07, the cause of every red this gate ever
    // gave). The look-at drive returns the moment the DELIVERED pose reads aligned, while the last
    // look inputs are still in flight — server, snapshot, the interpolation buffer — so the pose
    // read next is a pose the renderer moves past before the frame is captured. MEASURED with the
    // renderer's own star probe: a pure 3.2° yaw between the gate's camera and the frame's, 49 px at
    // the centre, more at the edges, different every run. Two polls a buffer apart with the same
    // pose mean the avatar is at rest; only then is a camera built and a frame taken.
    let settled = {
        let started = Instant::now();
        let mut last = vd_bins::pixel::own_pose(&vd_bins::pixel::poll(devctl));
        loop {
            std::thread::sleep(POSE_SETTLE_POLL);
            let now = vd_bins::pixel::own_pose(&vd_bins::pixel::poll(devctl));
            if now.is_some() && now == last {
                break true;
            }
            if started.elapsed() > POSE_SETTLE_TIMEOUT {
                break false;
            }
            last = now;
        }
    };
    assert!(
        settled,
        "the avatar's delivered pose never came to rest after the look-at"
    );

    // The camera the client draws through, reconstructed AFTER the turn — from the avatar's own eye
    // and its delivered facing, which is what `--capture-pilot` renders.
    let state = vd_bins::pixel::poll(devctl);
    let camera = vd_bins::pixel::pilot_camera(&state, CAPTURE_W as usize, CAPTURE_H as usize);

    let shot = round_trip(
        devctl,
        &DevRequest::Screenshot {
            at_tick: None,
            label: Some("star-sky".to_owned()),
        },
    );
    let DevResponse::Captured { path: rel_path, .. } = shot else {
        panic!("Screenshot should answer with a captured path, got {shot:?}");
    };

    let png = cwd.join(&rel_path);
    // KEEP THE EVIDENCE: the cluster's shutdown reaps the slot directory, capture included, so a
    // red verdict used to leave nothing to look at. The capture is copied beside the run's own log
    // before any pixel is judged; the path is printed so a reader can open it.
    if let Ok(keep) = std::env::var("VD_KEEP_CAPTURE") {
        let kept = std::path::Path::new(&keep).join("star_sky_capture.png");
        match std::fs::copy(&png, &kept) {
            Ok(_) => println!("capture kept at {}", kept.display()),
            Err(e) => println!("capture NOT kept ({e}) — {}", kept.display()),
        }
    }
    let img = image::open(&png)
        .unwrap_or_else(|e| panic!("open captured PNG {}: {e}", png.display()))
        .to_rgba8();
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let rgba = img.as_raw();

    // A broken shader paints Bevy's magenta sentinel. Assert it first: a magenta frame would satisfy
    // "non-clear pixels near the star" while proving the opposite of what this gate claims.
    assert_eq!(
        magenta_pixel_count(rgba),
        0,
        "the frame carries the {MAGENTA:?} shader-failure sentinel"
    );

    // The clear colour, read from the frame's own corner (as the sibling gates do).
    let corner = (w - 1) * 4;
    let clear: [u8; 4] = rgba[corner..corner + 4].try_into().expect("corner pixel");

    // ★ THE VERDICT. Every star that projects in FRONT of the eye must have painted pixels within a
    // bracket of where the seed says it is. A star behind the camera paints nothing and is not a
    // failure — the fitted capture view stands at the home system, and a sibling star is routinely
    // behind you.
    let mut checked = 0usize;
    let mut behind = 0usize;
    let mut offframe = 0usize;
    let mut culled = 0usize;
    let mut misses: Vec<(vd_core::pose::RealmId, f64, f64, f64)> = Vec::new();
    let star_tuning = vd_client::realm_scene::StarTuning::default();
    for (realm, pos, class_code, luma_lsun) in &drawn {
        // ★ THE STAR LAW DECIDES WHO MUST PAINT (2026-09-06): a far, dim star's drawn amplitude
        // falls below the law's cull level and the renderer does not draw it alone — it is part
        // of the summed glow. The gate reads the SAME Tier-A law (`star_draw`, the numbers the
        // renderer's uniforms come from), so it demands a pixel only from a star the law draws.
        // MEASURED: a star 1.56e18 m away, in frame at (996, 132), painted nothing and the gate
        // called the star lane broken; the law had culled it. Counted by name, never silent.
        let look = vd_client::realm_scene::marker_look(*class_code, *luma_lsun);
        let draw =
            vd_client::realm_scene::star_draw(look.base_radius_m, pos.length(), &star_tuning);
        if draw.amplitude <= 0.0 {
            println!(
                "  {realm:?}: CULLED by the star law (flux {:.3e}), {:.3e} m away",
                draw.flux,
                pos.length()
            );
            culled += 1;
            continue;
        }
        let Some(rect) = projected_point_aabb(&camera, *pos, 0.0) else {
            // DIAGNOSED, not merely skipped: "no star was in view" has two causes and they call for
            // different fixes, so the gate must say which one it met.
            println!("  {realm:?}: BEHIND the eye, at {:.3e} m", pos.length());
            behind += 1;
            continue;
        };
        let probe = ScreenAabb {
            min: vd_client_harness::camera::ScreenPos {
                x: rect.min.x - STAR_PROBE_HALF_PX,
                y: rect.min.y - STAR_PROBE_HALF_PX,
            },
            max: vd_client_harness::camera::ScreenPos {
                x: rect.max.x + STAR_PROBE_HALF_PX,
                y: rect.max.y + STAR_PROBE_HALF_PX,
            },
        };
        // Off-frame is not a drawing failure either.
        // ON-FRAME is the STAR'S OWN footprint, never the probe's (2026-09-06): a star ten pixels
        // left of the frame is inside a 12 px probe but no pixel can paint there, and the gate read
        // it as "in view, painted nothing" — the ledgered edge-of-frame red. The probe stays the
        // search window; the frame's own edge decides membership.
        if rect.max.x < 0.0 || rect.max.y < 0.0 || rect.min.x > w as f64 || rect.min.y > h as f64 {
            println!(
                "  {realm:?}: OFF-FRAME at ({:.1}, {:.1}), frame is {w}x{h}",
                rect.min.x, rect.min.y
            );
            offframe += 1;
            continue;
        }
        if draw.amplitude >= BRIGHT_STAR_AMPLITUDE {
            println!(
                "  {realm:?}: IN VIEW at ({:.1}, {:.1}), {:.3e} m away, amplitude {:.4}, crop {:.2} px",
                rect.min.x,
                rect.min.y,
                pos.length(),
                draw.amplitude,
                draw.crop_px
            );
        }
        let lit = nonclear_in_region(rgba, w, h, clear, probe);
        // A CENSUS, not a first-miss panic (2026-09-06): the first red used to stop the walk at star
        // 962 of 233,220, and every diagnosis read that prefix as the whole sky. Every miss is kept
        // and the verdict comes after the walk, with the bright misses named first.
        if lit == 0 {
            misses.push((*realm, rect.min.x, rect.min.y, draw.amplitude));
        }
        checked += 1;
    }
    // ★ THE STAR PROBE, READ BACK (2026-09-06): the renderer's own projection of its brightest
    // stars against this gate's prediction (the math) and against the pixels at the renderer's
    // position (the raster). Printed per star; the two disagreements are then told apart.
    let predicted: std::collections::BTreeMap<String, (f64, f64)> = drawn
        .iter()
        .filter_map(|(realm, pos, _, _)| {
            let rect = projected_point_aabb(&camera, *pos, 0.0)?;
            Some((format!("{realm:?}"), (rect.min.x, rect.min.y)))
        })
        .collect();
    let mut probe_math_agree = 0usize;
    let mut probe_raster_agree = 0usize;
    for probe in &state.star_probe {
        let core = ScreenAabb {
            min: vd_client_harness::camera::ScreenPos {
                x: probe.x - 4.0,
                y: probe.y - 4.0,
            },
            max: vd_client_harness::camera::ScreenPos {
                x: probe.x + 4.0,
                y: probe.y + 4.0,
            },
        };
        let painted = nonclear_in_region(rgba, w, h, clear, core);
        let mine = predicted.get(&probe.realm).copied();
        let delta = mine.map(|(x, y)| ((x - probe.x).powi(2) + (y - probe.y).powi(2)).sqrt());
        probe_math_agree += usize::from(delta.is_some_and(|d| d <= 2.0));
        probe_raster_agree += usize::from(painted > 0);
        println!(
            "  PROBE {} amp {:.3}: renderer ({:.1}, {:.1}); gate {:?}; delta {:?} px; pixels within 4 px of the renderer's spot: {painted}",
            probe.realm, probe.amplitude, probe.x, probe.y, mine, delta
        );
    }
    println!(
        "  probe: {} stars; math agrees (<=2 px) for {probe_math_agree}; pixels at the renderer's spot for {probe_raster_agree}",
        state.star_probe.len()
    );
    misses.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    let bright_misses: Vec<_> = misses
        .iter()
        .filter(|m| m.3 >= BRIGHT_STAR_AMPLITUDE)
        .collect();
    println!(
        "  misses {} of {checked} in view ({} bright); bright misses: {:?}",
        misses.len(),
        bright_misses.len(),
        bright_misses.iter().take(12).collect::<Vec<_>>()
    );
    for (realm, x, y, amplitude) in &misses {
        println!("  MISS {realm:?} at ({x:.1}, {y:.1}) amplitude {amplitude:.4}");
    }
    // ★ THE VERDICT (2026-09-07): every star the law draws paints within the probe — bright or
    // faint. Two runs of 41,000+ stars in view found ZERO misses once the two real causes were cured
    // (the stale camera pose the gate read, and the f32 underflow that culled every star under
    // amplitude 0.11 in the shader); a miss here is a regression, never noise.
    assert!(
        misses.is_empty(),
        "{} stars drew NO pixel within {STAR_PROBE_HALF_PX} px of where the seed puts them (the \
         brightest: {:?}) — the star lane paints them elsewhere or not at all",
        misses.len(),
        misses.first()
    );

    // ★ ANTI-VACUITY. Every star could be behind the eye or off-frame, and the loop above would pass
    // having asserted nothing at all. That is the shape of a gate that certifies its own silence.
    println!("  checked {checked}, behind {behind}, off-frame {offframe}, culled {culled}");
    assert!(
        checked > 0,
        "no star was in front of the camera and on-frame, so this gate proved NOTHING — the \
         capture framing must place at least one star in view for the verdict to mean anything \
         ({behind} behind the eye, {offframe} off-frame, of {} drawable)",
        drawn.len()
    );

    // The manifest records the capture, as the sibling gates require.
    let manifest_path = cwd
        .join(
            Path::new(&rel_path)
                .parent()
                .expect("shots dir")
                .parent()
                .expect("run dir"),
        )
        .join(MANIFEST_FILENAME);
    if let Ok(text) = std::fs::read_to_string(&manifest_path)
        && let Ok(manifest) = serde_json::from_str::<RunManifest>(&text)
    {
        assert!(
            manifest
                .captures
                .iter()
                .any(|c| c.kind == CaptureKind::Screenshot),
            "the run manifest records the PNG capture"
        );
    }

    println!(
        "G-STAR-SKY-PIXELS: {}x{} frame, clear {clear:?}, sky generation {generation:#x}, \
         {held_stars} stars held, {} drawable, {checked} checked in view",
        w,
        h,
        drawn.len()
    );
}
