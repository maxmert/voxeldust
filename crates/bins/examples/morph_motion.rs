//! ★ THE GROUND'S OWN MOTION IN FLIGHT (2026-10-01, the owner from the window: "On the flight rungs adjusting
//! height all the time, and because of that it looks like waves"): for each recorded pair of a moving-eye
//! leg, every ground pixel the probe marks MOVING (its vertex slides on the geomorph, the sink or the arrival
//! ramp) is read like the motion truth reads a still one — its written motion against the reference through
//! the last frame's camera — so the residual is the ground's own slide on the screen in one frame, in
//! render pixels. By rung, and by distance.
//! `cargo run --release -p vd-bins --features dev-control --example morph_motion -- <run dir> <label prefix>`

use std::collections::BTreeMap;
use std::path::Path;

use vd_client_harness::motion_census::{MOTION_SENTINEL_BITS, MotionPicture, half_to_f32};
use vd_client_harness::probe::{PROBE_KIND_TERRAIN, decode_probe, probe_moving};
use vd_devproto::DevState;

struct Frame {
    probe: Vec<u8>,
    motion: MotionPicture,
    camera: vd_client_harness::camera::CaptureCamera,
    jitter: [f64; 2],
    frames: u64,
}

fn open(run: &Path, stem: &str) -> Option<Frame> {
    let probe = image::open(run.join(format!("frames/{stem}.probe.png")))
        .ok()?
        .to_rgba8();
    let (w, h) = (probe.width() as usize, probe.height() as usize);
    let motion = MotionPicture::from_bytes(
        &std::fs::read(run.join(format!("frames/{stem}.motion.bin"))).ok()?,
    )
    .ok()?;
    let state: DevState = serde_json::from_str(
        &std::fs::read_to_string(run.join(format!("state/{stem}.json"))).ok()?,
    )
    .ok()?;
    let stamp = state.terrain_stamp?;
    Some(Frame {
        camera: vd_client_harness::pop::frame_camera(
            stamp.eye_body_m,
            stamp.camera_body_xyzw,
            w,
            h,
        ),
        jitter: stamp
            .reconstruction
            .map_or([0.0; 2], |r| r.jitter_px.map(f64::from)),
        frames: stamp.frames,
        probe: probe.into_raw(),
        motion,
    })
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let run = Path::new(&a[0]);
    let prefix = &a[1];
    // The labels of the pairs: `<prefix>-NNN-0000` and `-0001`.
    let mut labels: Vec<String> = std::fs::read_dir(run.join("frames"))
        .expect("the run's frames")
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix.as_str()) && n.ends_with("-0001.probe.png"))
        .map(|n| n.trim_end_matches("-0001.probe.png").to_owned())
        .collect();
    labels.sort();
    // rung -> (pixels moving, pixels still, residuals of the moving)
    let mut by_rung: BTreeMap<u8, (u64, u64, Vec<f64>)> = BTreeMap::new();
    let mut by_km: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    let mut eye_steps = Vec::new();
    for label in &labels {
        let (Some(last), Some(now)) = (
            open(run, &format!("{label}-0000")),
            open(run, &format!("{label}-0001")),
        ) else {
            continue;
        };
        if now.frames != last.frames + 1 {
            continue;
        }
        eye_steps.push((now.camera.eye - last.camera.eye).length());
        let (w, h) = (now.camera.width, now.camera.height);
        for (i, px) in now.probe.chunks_exact(4).enumerate() {
            let p = decode_probe([px[0], px[1], px[2]]);
            if p.kind != PROBE_KIND_TERRAIN {
                continue;
            }
            let row = by_rung.entry(p.rung).or_insert((0, 0, Vec::new()));
            if !probe_moving(px[3]) {
                row.1 += 1;
                continue;
            }
            row.0 += 1;
            let bits = now.motion.motion[i];
            if bits == [MOTION_SENTINEL_BITS; 2] {
                continue;
            }
            let d = f64::from(vd_seed::ladder::cell_m(p.rung)) * f64::from(p.cells);
            let [x, y] = vd_client::reconstruction::sample_point(
                [(i % w) as i64, (i / w) as i64],
                now.jitter,
            );
            let Some(reference) = last.camera.project_point(now.camera.unproject(x, y, d)) else {
                continue;
            };
            let own_x = x - f64::from(half_to_f32(bits[0])) * w as f64;
            let own_y = y - f64::from(half_to_f32(bits[1])) * h as f64;
            let residual = (own_x - reference.x).hypot(own_y - reference.y);
            row.2.push(residual);
            by_km.entry((d / 1000.0) as u32).or_default().push(residual);
        }
    }
    eye_steps.sort_by(f64::total_cmp);
    println!(
        "{} pairs of {prefix}; the eye's step a frame {:.2} m at the median",
        labels.len(),
        eye_steps.get(eye_steps.len() / 2).copied().unwrap_or(0.0)
    );
    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(f64::total_cmp);
        v.get(((v.len() as f64 - 1.0) * q).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    for (rung, (moving, still, mut res)) in by_rung {
        let share = moving as f64 / (moving + still).max(1) as f64;
        println!(
            "rung {rung:2}: {moving:8} moving of {:8} ground pixels ({:.1} %); their own slide a frame: median {:.3} px, p90 {:.3}, p99 {:.3}, the widest {:.3}",
            moving + still,
            100.0 * share,
            pct(&mut res, 0.5),
            pct(&mut res, 0.9),
            pct(&mut res, 0.99),
            pct(&mut res, 1.0)
        );
    }
    for (km, mut res) in by_km {
        if res.len() < 500 {
            continue;
        }
        println!(
            "{km:4}–{:4} km: {:8} moving pixels; their slide median {:.3} px, p99 {:.3}",
            km + 1,
            res.len(),
            pct(&mut res, 0.5),
            pct(&mut res, 0.99)
        );
    }
}
