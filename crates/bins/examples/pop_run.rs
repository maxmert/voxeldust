//! A diagnostic (not a gate): THE POP DETECTOR over a recorded CONSECUTIVE sequence
//! (`vdctl record --consecutive`), every pair of neighbouring frames judged — the same judge the
//! moving-eye flight samples once every two seconds (`vd_client_harness::pop::judge_pair`).
//!
//! On a STILL eye no band moves, so every pixel whose rung changes between two frames is a chunk
//! that arrived or left: the reading is the arrivals' own pop (the far-water design §10, slice
//! R1c: a late chunk used to rise by up to twice the coarser rung's sink in one frame).
//!
//! ```text
//! cargo run --release -p vd-bins --example pop_run -- <run dir> <label>
//! ```
//!
//! **Example.** A pilot logs in over the steep coast and the client records four seconds while the
//! rings stream in. Before the ramp each landing chunk steps its pixels by up to tens of levels in
//! one frame; with the ramp the first frame of a landing chunk shows its stand-in's surface.

use std::path::{Path, PathBuf};

use vd_client_harness::capture::{capture_rel_path, probe_rel_for, state_rel_for};
use vd_client_harness::manifest::CaptureKind;
use vd_client_harness::pop::{Frame, frame_camera, judge_pair, over_floor, sum_reads};
use vd_devproto::DevState;

/// One recorded frame: the picture, the probe and the frame's stamp.
struct Recorded {
    rgba: image::RgbaImage,
    probe: image::RgbaImage,
    stamp: vd_devproto::DevTerrainStamp,
}

fn load(run: &Path, label: &str, index: u32) -> Option<Recorded> {
    let rel = capture_rel_path(CaptureKind::Frame, &format!("{label}-{index:04}"));
    let rgba = image::open(run.join(&rel)).ok()?.to_rgba8();
    let probe = image::open(run.join(probe_rel_for(&rel))).ok()?.to_rgba8();
    let state: DevState =
        serde_json::from_str(&std::fs::read_to_string(run.join(state_rel_for(&rel))).ok()?).ok()?;
    Some(Recorded {
        rgba,
        probe,
        stamp: state.terrain_stamp?,
    })
}

fn frame(f: &Recorded) -> Frame<'_> {
    let (w, h) = f.rgba.dimensions();
    let (w, h) = (w as usize, h as usize);
    Frame {
        rgba: f.rgba.as_raw(),
        probe: f.probe.as_raw(),
        width: w,
        height: h,
        camera: frame_camera(f.stamp.eye_body_m, f.stamp.camera_body_xyzw, w, h),
        hud: {
            let r = f.stamp.hud_rect_px.map(f64::from);
            ((r[2] > r[0]) & (r[3] > r[1])).then_some(r)
        },
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(run), Some(label)) = (args.first(), args.get(1)) else {
        eprintln!("usage: pop_run <run dir> <label>");
        std::process::exit(2);
    };
    let run = PathBuf::from(run);
    let cell = |rung: u8| f64::from(vd_seed::ladder::cell_m(rung));
    let mut reads = Vec::new();
    let (mut skipped, mut judged) = (0u64, 0u64);
    let mut before = load(&run, label, 0);
    let mut index = 1u32;
    while let Some(after) = load(&run, label, index) {
        if let Some(a) = before.as_ref() {
            let consecutive = after.stamp.frames == a.stamp.frames + 1;
            let same = a.rgba.dimensions() == after.rgba.dimensions()
                && a.stamp.realm == after.stamp.realm;
            if consecutive & same {
                let read = judge_pair(&frame(a), &frame(&after), &cell);
                let boundaries: Vec<(u8, u64, u64, u8)> = read
                    .boundary_reads()
                    .iter()
                    .filter(|b| b.crossed > 0)
                    .map(|b| (b.finer, b.crossed, b.over_floor, b.widest))
                    .collect();
                if !boundaries.is_empty() {
                    println!(
                        "pair {:04}: floor {}, per boundary (finer rung, crossed, past the floor, widest) {boundaries:?}",
                        index - 1,
                        read.floor()
                    );
                }
                reads.push(read);
                judged += 1;
            } else {
                skipped += 1;
            }
        }
        before = Some(after);
        index += 1;
    }
    let sum = sum_reads(&reads);
    // Against FIXED thresholds too: a ramping chunk's pixels change a little each frame, which
    // lifts the still pixels' floor, so two runs are compared at the same levels as well.
    let fixed: Vec<(u8, u64, u64)> = [2u8, 3, 5, 10, 20]
        .iter()
        .map(|&t| (t, over_floor(&sum.flipped, t), over_floor(&sum.still, t)))
        .collect();
    println!(
        "POP-RUN-FIXED {label}: (threshold in levels, boundary pixels past it, still pixels past it) {fixed:?}"
    );
    println!(
        "POP-RUN {label}: {judged} pairs judged ({skipped} left out), {} pixels compared, {} crossed a rung boundary, the floor at {} levels, {} past the floor, the widest {} levels; per boundary (finer rung, crossed, past the floor, widest) {:?}",
        sum.compared,
        sum.crossed(),
        sum.floor(),
        sum.over_floor(),
        sum.widest(),
        sum.boundary_reads()
            .iter()
            .map(|b| (b.finer, b.crossed, b.over_floor, b.widest))
            .collect::<Vec<_>>()
    );
}
