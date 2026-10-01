//! A diagnostic (not a gate): THE POP DETECTOR OVER A MOVING-EYE FLIGHT'S RECORDED PAIRS, pooled per
//! leg (the temporal path's T2, 2026-09-30: the pass rules are set on these numbers). A flight kept
//! with `VD_KEEP_FIXTURE=1` holds every pair it judged under `capture-cwd/runs/<client run>/frames/`
//! as `pop-<leg>-NNN-0000` and `-0001`; this reads them all, judges each pair as the gate does — the
//! RENDER picture with its silhouettes apart, and under the reconstruction the DISPLAY picture with
//! the probe under each display pixel — and writes the pooled step histograms per leg as JSON, so a
//! pass rule can be tried against flights that must pass and a control that must fail.
//!
//! ```text
//! cargo run --release -p vd-bins --example pop_legs -- <runs dir> <out.json>
//! ```
//!
//! With `POP_LEGS_PAIRS` set it prints each pair's reading; with `POP_LEGS_MAP=<text>` it paints a map of
//! every pair whose label holds the text (`maps/` beside the JSON: the picture dimmed, a boundary pixel
//! at or under the floor dark green, past it red, an edge past it cyan, a still pixel past it yellow);
//! with `POP_LEGS_WIDE=<levels>` it lists every boundary pixel that stepped at least that far, with its
//! place and distance; with `POP_LEGS_PATCHES=<levels>` it writes every CONNECTED PATCH of boundary
//! pixels that stepped at least that far (eight neighbours) to `patches.csv` beside the JSON — its
//! leg, pair, picture, area in pixels and in square arcminutes of the picture's own field, and its mean
//! and widest step: the pops as a player's eye meets them, a patch at a time. With `POP_LEGS_EYE=1` it
//! reads every pair by THE EYE'S RULE (`vd_client_harness::eye`): each patch and each window with its
//! visibility level to `eye.csv` beside the JSON (a patch from half the eye's threshold up), and per
//! leg, picture and group the patches and windows the eye sees; `POP_LEGS_EYE_MAP=<text>` also paints
//! every step whose label holds the text (`maps/<label>-<picture>-eye.png`: each patch the eye sees —
//! held on a boundary red, on still ground yellow, on a silhouette cyan, a flicker magenta).
//!
//! **Example.** The four-frame flight of 2026-10-01: the eye reads no pop it sees on the walk, at 528 m/s
//! or in the turn, one two-pixel patch at 240 m/s, and the login's pops at the landing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use vd_client_harness::capture::{capture_rel_path, probe_rel_for, render_rel_for, state_rel_for};
use vd_client_harness::eye::{Group, Part, REFERENCE_DISPLAY, Sighting, read_step};
use vd_client_harness::manifest::CaptureKind;
use vd_client_harness::pop::{
    Frame, PairRead, PixelClass, PixelRead, RENDER_EDGE_REACH_PX, display_edge_reach_px,
    edge_separation_per_m, frame_camera, judge_pair_mapped, sum_reads,
};
use vd_client_harness::probe::{decode_probe, display_probe};
use vd_devproto::DevState;

struct Recorded {
    rgba: image::RgbaImage,
    probe: image::RgbaImage,
    render: Option<image::RgbaImage>,
    stamp: vd_devproto::DevTerrainStamp,
}

fn load(run: &Path, label: &str, index: u32) -> Option<Recorded> {
    let rel = capture_rel_path(CaptureKind::Frame, &format!("{label}-{index:04}"));
    let rgba = image::open(run.join(&rel)).ok()?.to_rgba8();
    let probe = image::open(run.join(probe_rel_for(&rel))).ok()?.to_rgba8();
    let state: DevState =
        serde_json::from_str(&std::fs::read_to_string(run.join(state_rel_for(&rel))).ok()?).ok()?;
    let render = image::open(run.join(render_rel_for(&rel)))
        .ok()
        .map(|i| i.to_rgba8());
    Some(Recorded {
        rgba,
        probe,
        render,
        stamp: state.terrain_stamp?,
    })
}

fn jitter(f: &Recorded) -> [f64; 2] {
    f.stamp
        .reconstruction
        .map_or([0.0; 2], |r| r.jitter_px.map(f64::from))
}

fn hud(f: &Recorded) -> Option<[f64; 4]> {
    let r = f.stamp.hud_rect_px.map(f64::from);
    ((r[2] > r[0]) & (r[3] > r[1])).then_some(r)
}

/// The render picture as the gate reads it: the probe's grid, the frame's jitter, the silhouettes apart.
fn render_frame(f: &Recorded) -> Frame<'_> {
    let (w, h) = f.probe.dimensions();
    let (w, h) = (w as usize, h as usize);
    Frame {
        rgba: f.render.as_ref().map_or(f.rgba.as_raw(), |r| r.as_raw()),
        probe: f.probe.as_raw(),
        width: w,
        height: h,
        camera: frame_camera(f.stamp.eye_body_m, f.stamp.camera_body_xyzw, w, h),
        hud: hud(f),
        jitter_px: jitter(f),
        edge_separation_per_m: if f.stamp.reconstruction.is_some() {
            edge_separation_per_m(w, h)
        } else {
            0.0
        },
        edge_reach_px: RENDER_EDGE_REACH_PX,
    }
}

/// The probe under each display pixel.
fn display_probe_of(f: &Recorded) -> Vec<u8> {
    let (dw, dh) = f.rgba.dimensions();
    let (pw, ph) = f.probe.dimensions();
    display_probe(
        f.probe.as_raw(),
        [pw as usize, ph as usize],
        [dw as usize, dh as usize],
        jitter(f),
    )
}

/// The display picture as the gate reads it: the unjittered camera, the silhouettes apart.
fn display_frame<'a>(f: &'a Recorded, probe: &'a [u8]) -> Frame<'a> {
    let (w, h) = f.rgba.dimensions();
    let (w, h) = (w as usize, h as usize);
    let (rw, rh) = f.probe.dimensions();
    let (rw, rh) = (rw as usize, rh as usize);
    Frame {
        rgba: f.rgba.as_raw(),
        probe,
        width: w,
        height: h,
        camera: frame_camera(f.stamp.eye_body_m, f.stamp.camera_body_xyzw, w, h),
        hud: hud(f),
        jitter_px: [0.0; 2],
        edge_separation_per_m: edge_separation_per_m(rw, rh),
        edge_reach_px: display_edge_reach_px(rw, w),
    }
}

/// THE MAP of one judged picture: the picture dimmed, each compared pixel painted by its class and
/// whether its step stood past `floor`.
fn map_of(frame: &Frame<'_>, marks: &[PixelRead], floor: u8) -> image::RgbaImage {
    let (w, h) = (frame.width as u32, frame.height as u32);
    let mut map = image::RgbaImage::from_fn(w, h, |x, y| {
        let i = ((y * w + x) * 4) as usize;
        let dim = |c: u8| (f32::from(c) * 0.35) as u8;
        image::Rgba([
            dim(frame.rgba[i]),
            dim(frame.rgba[i + 1]),
            dim(frame.rgba[i + 2]),
            255,
        ])
    });
    for m in marks {
        let past = m.step > floor;
        let colour = match (m.class, past) {
            (PixelClass::Boundary(_), false) => Some([0, 90, 0]),
            (PixelClass::Boundary(_), true) => Some([255, 40, 40]),
            (PixelClass::Edge, true) => Some([0, 200, 255]),
            (PixelClass::Still, true) => Some([255, 220, 0]),
            _ => None,
        };
        if let Some([r, g, b]) = colour {
            map.put_pixel(m.x as u32, m.y as u32, image::Rgba([r, g, b, 255]));
        }
    }
    map
}

/// Every boundary pixel of a judged picture that stepped at least `wide` levels, with its distance.
fn wide_lines(
    label: &str,
    which: &str,
    frame: &Frame<'_>,
    marks: &[PixelRead],
    wide: u8,
) -> Vec<String> {
    marks
        .iter()
        .filter(|m| matches!(m.class, PixelClass::Boundary(_)) & (m.step >= wide))
        .map(|m| {
            let i = (m.y * frame.width + m.x) * 4;
            let p = decode_probe([frame.probe[i], frame.probe[i + 1], frame.probe[i + 2]]);
            let d = f64::from(vd_seed::ladder::cell_m(p.rung)) * f64::from(p.cells);
            format!(
                "WIDE {label} {which} ({}, {}): step {}, {:?}, drawn by rung {} at {d:.0} m",
                m.x, m.y, m.step, m.class, p.rung
            )
        })
        .collect()
}

/// ★ ONE CONNECTED PATCH of boundary pixels that stepped at least a threshold (eight neighbours): its
/// area in pixels, its mean and widest step, and the finer rung most of its pixels crossed.
struct Patch {
    area: usize,
    mean: f64,
    widest: u8,
    rung: u8,
}

/// The patches of one judged picture.
fn patches_of(frame: &Frame<'_>, marks: &[PixelRead], min: u8) -> Vec<Patch> {
    let (w, h) = (frame.width, frame.height);
    let mut step = vec![0u8; w * h];
    let mut rung = vec![0u8; w * h];
    let mut on = vec![false; w * h];
    for m in marks {
        if let PixelClass::Boundary(r) = m.class
            && m.step >= min
        {
            let i = m.y * w + m.x;
            on[i] = true;
            step[i] = m.step;
            rung[i] = r;
        }
    }
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !on[start] {
            continue;
        }
        on[start] = false;
        stack.push(start);
        let (mut area, mut sum, mut widest) = (0usize, 0u64, 0u8);
        let mut rungs = [0usize; 32];
        while let Some(i) = stack.pop() {
            area += 1;
            sum += u64::from(step[i]);
            widest = widest.max(step[i]);
            rungs[usize::from(rung[i]) & 31] += 1;
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if (nx >= 0) & (ny >= 0) & (nx < w as i64) & (ny < h as i64) {
                        let j = ny as usize * w + nx as usize;
                        if on[j] {
                            on[j] = false;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        let rung = (0..32).max_by_key(|r| rungs[*r]).unwrap_or(0) as u8;
        out.push(Patch {
            area,
            mean: sum as f64 / area as f64,
            widest,
            rung,
        });
    }
    out
}

/// Judges a pair and, when asked, maps it and lists its widest boundary pixels.
fn judged(
    label: &str,
    which: &str,
    before: &Frame<'_>,
    after: &Frame<'_>,
    cell: &dyn Fn(u8) -> f64,
    ask: &Ask,
) -> PairRead {
    let mut marks = Vec::new();
    let keep = ask.map.as_ref().is_some_and(|m| label.contains(m.as_str()))
        | ask.wide.is_some()
        | ask.patches.is_some();
    let read = judge_pair_mapped(before, after, cell, &mut |p| {
        if keep {
            marks.push(p);
        }
    });
    if let Some(wide) = ask.wide {
        for line in wide_lines(label, which, after, &marks, wide) {
            println!("{line}");
        }
    }
    if let Some(min) = ask.patches {
        // One pixel's side in arcminutes of the picture's own vertical field.
        let arcmin = after.camera.fov_y.to_degrees() * 60.0 / after.height as f64;
        let mut rows = ask.patch_rows.borrow_mut();
        for p in patches_of(after, &marks, min) {
            rows.push(format!(
                "{},{label},{which},{},{:.2},{:.2},{},{}",
                leg_of(label),
                p.area,
                p.area as f64 * arcmin * arcmin,
                p.mean,
                p.widest,
                p.rung
            ));
        }
    }
    if ask.map.as_ref().is_some_and(|m| label.contains(m.as_str())) {
        let path = ask.maps_dir.join(format!("{label}-{which}.png"));
        std::fs::create_dir_all(&ask.maps_dir).expect("the maps dir");
        map_of(after, &marks, read.floor().max(1))
            .save(&path)
            .expect("write the map");
        println!("MAP {}", path.display());
    }
    read
}

/// ★ THE EYE'S READING of one judged picture: its patches and windows to the rows, and the leg's tally.
/// ★ THE EYE'S READING of one step of a recorded pair (the step's two pictures, and the frames around it
/// when the flight recorded them): the judge reads the step's second picture against each of the others,
/// then the eye's patches and windows go to the rows and the leg's tally.
fn eye_read(label: &str, which: &str, frames: &[Frame<'_>], cell: &dyn Fn(u8) -> f64, ask: &Ask) {
    // The display picture carries the engine's screen dither; the render picture does not.
    let Some(reading) = read_step(frames, cell, which == "display", REFERENCE_DISPLAY) else {
        return;
    };
    // The step's second picture: the middle two of four, else the second recorded.
    let second = &frames[usize::from(frames.len() >= 4) + 1];
    let [_, before_marks, after_marks] = &reading.reads;
    let leg = leg_of(label);
    let mut rows = ask.eye_rows.borrow_mut();
    let mut tally = ask.eye_tally.borrow_mut();
    let found = &reading.found;
    if ask
        .eye_map
        .as_ref()
        .is_some_and(|m| label.contains(m.as_str()))
    {
        let path = ask.maps_dir.join(format!("{label}-{which}-eye.png"));
        std::fs::create_dir_all(&ask.maps_dir).expect("the maps dir");
        eye_map_of(second, found)
            .save(&path)
            .expect("write the eye's map");
        println!("EYE MAP {}", path.display());
    }
    // ★ How many of the seen held patches' pixels the frame before or after the step could not read
    // (the judge's near limit grows with the eye's travel, two frames for the frame before).
    let read_at = |marks: &[PixelRead]| {
        let mut at = vec![false; second.width * second.height];
        for m in marks {
            at[m.y * second.width + m.x] = true;
        }
        at
    };
    let (had_before, had_after) = (read_at(before_marks), read_at(after_marks));
    {
        let mut unread = ask.eye_unread.borrow_mut();
        for (p, pixels) in found
            .iter()
            .filter(|(p, _)| p.seen() & (p.part == Part::Held))
        {
            let t = unread
                .entry((leg_of(label), which.to_owned(), p.group))
                .or_default();
            t.0 += pixels.len() as u64;
            t.1 += pixels.iter().filter(|i| !had_before[**i]).count() as u64;
            t.2 += pixels.iter().filter(|i| !had_after[**i]).count() as u64;
        }
    }
    for (p, _) in found {
        let t = tally
            .entry((
                leg.clone(),
                which.to_owned(),
                format!("patch {:?}", p.part),
                p.group,
                0,
            ))
            .or_default();
        t.0 += u64::from(p.seen());
        if p.visibility > t.1 {
            *t = (
                t.0,
                p.visibility,
                format!(
                    "{label} ({}, {}) {} px rung {}",
                    p.at.0, p.at.1, p.pixels, p.rung
                ),
            );
        }
        if p.visibility >= 0.5 {
            rows.push(format!(
                "{leg},{label},{which},patch,{:?},{:?},{},{},{},{:.1},{:.5},{:.2},{:.5},{:.3},{},{}",
                p.part,
                p.group,
                p.rung,
                p.sign,
                p.pixels,
                p.area_arcmin2,
                p.contrast,
                p.field_cd_m2,
                p.threshold,
                p.visibility,
                p.at.0,
                p.at.1
            ));
        }
    }
    let wins = &reading.windows;
    // ★ THE VERDICT: the pops the eye sees in this step.
    {
        let seen = &reading.pops;
        let mut verdict = ask.eye_verdict.borrow_mut();
        let t = verdict.entry((leg.clone(), which.to_owned())).or_default();
        t.0 += 1;
        t.1 += u64::from(!seen.is_empty());
        t.2 += seen.len() as u64;
        for s in seen {
            println!(
                "EYE POP {label} {which}: {:?} {:?} rung {} {} px ({:.0}′²) at ({}, {}), contrast {:.4} on {:.1} cd/m², threshold {:.4}, level {:.2}",
                s.meeting,
                s.group,
                s.rung,
                s.pixels,
                s.area_arcmin2,
                s.at.0,
                s.at.1,
                s.contrast,
                s.field_cd_m2,
                s.threshold,
                s.visibility
            );
            if s.visibility > t.3 {
                t.3 = s.visibility;
                t.4 = format!(
                    "{label} {:?} {} px at ({}, {})",
                    s.group, s.pixels, s.at.0, s.at.1
                );
            }
        }
    }
    for (win, seen) in wins {
        let (win, seen) = (win, *seen);
        let side = (win.pixels as f64).sqrt() as usize;
        let t = tally
            .entry((
                leg.clone(),
                which.to_owned(),
                format!("window {:?}", win.part),
                win.group,
                side,
            ))
            .or_default();
        t.0 += seen;
        if win.visibility > t.1 {
            *t = (
                t.0,
                win.visibility,
                format!("{label} ({}, {})", win.at.0, win.at.1),
            );
        }
        rows.push(format!(
            "{leg},{label},{which},window,{:?},{:?},{},{},{},{:.1},{:.5},{:.2},{:.5},{:.3},{},{}",
            win.part,
            win.group,
            side,
            win.sign,
            seen,
            win.area_arcmin2,
            win.contrast,
            win.field_cd_m2,
            win.threshold,
            win.visibility,
            win.at.0,
            win.at.1
        ));
    }
}

/// ★ THE EYE'S MAP of one step: the second picture dimmed, each patch the eye SEES painted by its part
/// and group — held on a rung boundary red, held on still ground yellow, held on a silhouette cyan, a
/// flicker magenta.
fn eye_map_of(second: &Frame<'_>, found: &[(Sighting, Vec<usize>)]) -> image::RgbaImage {
    let (w, h) = (second.width as u32, second.height as u32);
    let mut map = image::RgbaImage::from_fn(w, h, |x, y| {
        let i = ((y * w + x) * 4) as usize;
        let dim = |c: u8| (f32::from(c) * 0.35) as u8;
        image::Rgba([
            dim(second.rgba[i]),
            dim(second.rgba[i + 1]),
            dim(second.rgba[i + 2]),
            255,
        ])
    });
    for (s, pixels) in found.iter().filter(|(s, _)| s.seen()) {
        let colour = match (s.part, s.group) {
            (Part::Held, Group::Boundary) => [255, 40, 40],
            (Part::Held, Group::Still) => [255, 220, 0],
            (Part::Held, Group::Edge) => [0, 200, 255],
            (Part::Flicker, _) => [255, 0, 255],
        };
        for i in pixels {
            map.put_pixel(
                (*i % second.width) as u32,
                (*i / second.width) as u32,
                image::Rgba([colour[0], colour[1], colour[2], 255]),
            );
        }
    }
    map
}

/// ★ THE EYE'S STEP of a recorded pair's frames: with four, the middle two and the frames around them;
/// with three, the first two and the frame after; with two, the pair alone.
fn eye_step(label: &str, which: &str, frames: &[Frame<'_>], cell: &dyn Fn(u8) -> f64, ask: &Ask) {
    eye_read(label, which, frames, cell, ask);
}

/// The eye's tally per (leg, picture, method, group, window side): how many the eye sees, the most
/// visible one's level and where it stands.
type EyeTally = BTreeMap<(String, String, String, Group, usize), (u64, f64, String)>;

/// The eye's verdict per (leg, picture): the steps read, the steps with a pop, the pops, and the most
/// visible one's level and place.
type EyeVerdict = BTreeMap<(String, String), (u64, u64, u64, f64, String)>;

/// Per (leg, picture, group): the seen held patches' pixels, and how many the frames around missed.
type EyeUnread = BTreeMap<(String, String, Group), (u64, u64, u64)>;

/// What the run was asked for beyond the pooled reading.
struct Ask {
    map: Option<String>,
    wide: Option<u8>,
    maps_dir: PathBuf,
    /// The patches' threshold, and the patches' rows as they are read.
    patches: Option<u8>,
    patch_rows: std::cell::RefCell<Vec<String>>,
    /// The eye's rule, its rows and its tally, and the steps it maps.
    eye: bool,
    eye_map: Option<String>,
    /// ★ THE VERDICT per (leg, picture): the steps read, the steps with a pop, the pops, and the most
    /// visible one's level and place.
    eye_verdict: std::cell::RefCell<EyeVerdict>,
    /// Per (leg, picture, group): the seen held patches' pixels, and how many of them the frame before
    /// and the frame after the step could not read.
    eye_unread: std::cell::RefCell<EyeUnread>,
    eye_rows: std::cell::RefCell<Vec<String>>,
    eye_tally: std::cell::RefCell<EyeTally>,
}

fn hist_json(hist: &[u64; 256]) -> serde_json::Value {
    serde_json::Value::from(hist.to_vec())
}

fn read_json(read: &PairRead) -> serde_json::Value {
    let boundaries: BTreeMap<String, serde_json::Value> = read
        .boundaries
        .iter()
        .map(|(rung, hist)| (rung.to_string(), hist_json(hist)))
        .collect();
    serde_json::json!({
        "compared": read.compared,
        "still": hist_json(&read.still),
        "flipped": hist_json(&read.flipped),
        "edges": hist_json(&read.edges),
        "boundaries": boundaries,
    })
}

/// Every pair label in a client run's frames: `pop-<leg>-NNN` with a `-0000` picture.
fn pair_labels(run: &Path) -> Vec<String> {
    let mut labels: Vec<String> = std::fs::read_dir(run.join("frames"))
        .map(|dir| {
            dir.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().into_string().ok())
                .filter_map(|name| name.strip_suffix("-0000.png").map(str::to_owned))
                .filter(|label| label.starts_with("pop-"))
                .collect()
        })
        .unwrap_or_default();
    labels.sort();
    labels
}

/// The leg of a pair label: `pop-hull-240-m-s-017` → `hull-240-m-s`.
fn leg_of(label: &str) -> String {
    let body = label.trim_start_matches("pop-");
    body.rsplit_once('-')
        .map_or(body, |(leg, _)| leg)
        .to_owned()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(runs), Some(out)) = (args.first(), args.get(1)) else {
        eprintln!("usage: pop_legs <runs dir> <out.json>");
        std::process::exit(2);
    };
    let cell = |rung: u8| f64::from(vd_seed::ladder::cell_m(rung));
    let ask = Ask {
        map: std::env::var("POP_LEGS_MAP").ok(),
        wide: std::env::var("POP_LEGS_WIDE")
            .ok()
            .and_then(|s| s.parse().ok()),
        maps_dir: Path::new(out)
            .parent()
            .unwrap_or(Path::new("."))
            .join("maps"),
        patches: std::env::var("POP_LEGS_PATCHES")
            .ok()
            .and_then(|s| s.parse().ok()),
        patch_rows: std::cell::RefCell::new(Vec::new()),
        eye: std::env::var_os("POP_LEGS_EYE").is_some(),
        eye_map: std::env::var("POP_LEGS_EYE_MAP").ok(),
        eye_unread: std::cell::RefCell::new(BTreeMap::new()),
        eye_verdict: std::cell::RefCell::new(BTreeMap::new()),
        eye_rows: std::cell::RefCell::new(Vec::new()),
        eye_tally: std::cell::RefCell::new(BTreeMap::new()),
    };
    let mut legs: BTreeMap<String, (Vec<PairRead>, Vec<PairRead>)> = BTreeMap::new();
    let mut client_runs: Vec<PathBuf> = std::fs::read_dir(runs)
        .expect("the runs dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("frames").is_dir())
        .collect();
    client_runs.sort();
    for run in &client_runs {
        for label in pair_labels(run) {
            let (Some(a), Some(b)) = (load(run, &label, 0), load(run, &label, 1)) else {
                eprintln!("pop_legs: {label} lacks a frame");
                continue;
            };
            if (b.stamp.frames != a.stamp.frames + 1)
                || (a.rgba.dimensions() != b.rgba.dimensions())
            {
                eprintln!("pop_legs: {label} is not two consecutive frames of one picture");
                continue;
            }
            let entry = legs.entry(leg_of(&label)).or_default();
            // ★ The frames after the pair, when the flight recorded them (the eye's held part): with four
            // the eye's step is the middle two, the first and the last around it.
            let c = load(run, &label, 2).filter(|c| {
                (c.stamp.frames == b.stamp.frames + 1)
                    & (c.rgba.dimensions() == b.rgba.dimensions())
            });
            let d = c.as_ref().and_then(|c| {
                load(run, &label, 3).filter(|d| {
                    (d.stamp.frames == c.stamp.frames + 1)
                        & (d.rgba.dimensions() == c.rgba.dimensions())
                })
            });
            let recorded: Vec<&Recorded> = [Some(&a), Some(&b), c.as_ref(), d.as_ref()]
                .into_iter()
                .flatten()
                .collect();
            let render = judged(
                &label,
                "render",
                &render_frame(&a),
                &render_frame(&b),
                &cell,
                &ask,
            );
            if ask.eye {
                let frames: Vec<Frame<'_>> = recorded.iter().map(|f| render_frame(f)).collect();
                eye_step(&label, "render", &frames, &cell, &ask);
            }
            let mut line = format!(
                "PAIR {label}: render {} past {} of {} boundary pixels (the widest {})",
                render.floor(),
                render.over_floor(),
                render.crossed(),
                render.widest()
            );
            entry.0.push(render);
            if a.render.is_some() & b.render.is_some() {
                let (pa, pb) = (display_probe_of(&a), display_probe_of(&b));
                let display = judged(
                    &label,
                    "display",
                    &display_frame(&a, &pa),
                    &display_frame(&b, &pb),
                    &cell,
                    &ask,
                );
                if ask.eye {
                    let probes: Vec<Vec<u8>> =
                        recorded.iter().map(|f| display_probe_of(f)).collect();
                    let frames: Vec<Frame<'_>> = recorded
                        .iter()
                        .zip(&probes)
                        .map(|(f, p)| display_frame(f, p))
                        .collect();
                    eye_step(&label, "display", &frames, &cell, &ask);
                }
                line.push_str(&format!(
                    "; display {} past {} of {} (the widest {}), per boundary {:?}",
                    display.floor(),
                    display.over_floor(),
                    display.crossed(),
                    display.widest(),
                    display
                        .boundary_reads()
                        .iter()
                        .filter(|b| b.over_floor > 0)
                        .map(|b| (b.finer, b.crossed, b.over_floor, b.widest))
                        .collect::<Vec<_>>()
                ));
                entry.1.push(display);
            }
            if std::env::var_os("POP_LEGS_PAIRS").is_some() {
                println!("{line}");
            }
        }
    }
    let mut json = serde_json::Map::new();
    for (leg, (renders, displays)) in &legs {
        let (r, d) = (sum_reads(renders), sum_reads(displays));
        println!(
            "POP-LEG {leg}: {} pairs; render: floor {}, {} of {} boundary pixels past it, the widest {}, {} silhouette pixels (the widest {}); display: {} pairs, floor {}, {} of {} past it, the widest {}",
            renders.len(),
            r.floor(),
            r.over_floor(),
            r.crossed(),
            r.widest(),
            r.edge_read().0,
            r.edge_read().1,
            displays.len(),
            d.floor(),
            d.over_floor(),
            d.crossed(),
            d.widest()
        );
        json.insert(
            leg.clone(),
            serde_json::json!({ "pairs": renders.len(), "render": read_json(&r), "display": read_json(&d) }),
        );
    }
    std::fs::write(
        out,
        serde_json::to_string(&serde_json::Value::Object(json)).expect("json"),
    )
    .expect("write the json");
    if ask.patches.is_some() {
        let csv = Path::new(out)
            .parent()
            .unwrap_or(Path::new("."))
            .join("patches.csv");
        let mut text =
            String::from("leg,pair,picture,area_px,area_arcmin2,mean_step,widest,rung\n");
        for row in ask.patch_rows.borrow().iter() {
            text.push_str(row);
            text.push('\n');
        }
        std::fs::write(&csv, text).expect("write the patches");
        println!("PATCHES {}", csv.display());
    }
    if ask.eye {
        let csv = Path::new(out)
            .parent()
            .unwrap_or(Path::new("."))
            .join("eye.csv");
        let mut text = String::from(
            "leg,pair,picture,method,part,group,rung_or_side,sign,pixels_or_seen,area_arcmin2,contrast,field_cd_m2,threshold,visibility,x,y\n",
        );
        for row in ask.eye_rows.borrow().iter() {
            text.push_str(row);
            text.push('\n');
        }
        std::fs::write(&csv, text).expect("write the eye's rows");
        for ((leg, which, method, group, side), (seen, best, at)) in ask.eye_tally.borrow().iter() {
            println!(
                "EYE {leg} {which} {method} {group:?} side {side}: {seen} seen, the most visible {best:.2} at {at}"
            );
        }
        for ((leg, which, group), (px, no_before, no_after)) in ask.eye_unread.borrow().iter() {
            println!(
                "EYE UNREAD {leg} {which} {group:?}: {px} pixels in seen held patches, {no_before} unread by the frame before, {no_after} by the frame after"
            );
        }
        for ((leg, which), (steps, with_pops, n, best, at)) in ask.eye_verdict.borrow().iter() {
            println!(
                "EYE VERDICT {leg} {which}: {with_pops} of {steps} steps hold a pop the eye sees ({n} pops), the most visible {best:.2} at {at}"
            );
        }
        println!("EYE ROWS {}", csv.display());
    }
}
