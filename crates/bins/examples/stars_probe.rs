//! ★ THE STAR CENSUS, ONE PAIR UNDER THE LENS (2026-10-01, G-STARS): for a candidate turn frame and
//! today's frame of the nearest camera (both kept by the picture gate's stars flight under
//! `<folder>/stars/{today,candidate}/turn`), every reference star of today's frame — its place, its
//! peak over the sky, its predicted place in the candidate (the census's own turn of its direction),
//! what the candidate's probe and picture show there, and the nearest candidate star.
//! `cargo run --release -p vd-bins --features dev-control --example stars_probe -- <folder> <candidate frame>`
//! — or `<folder> census`: the whole census of the kept frames, as the picture gate reads it.

use std::path::Path;

use vd_client_harness::camera::CaptureCamera;
use vd_client_harness::probe::decode_probe;
use vd_client_harness::stars::{StarFrame, StarReference, nearest_frame, stars_in};
use vd_core::glam::DVec3;
use vd_devproto::DevState;

struct File {
    rgba: Vec<u8>,
    probe: Vec<u8>,
    w: usize,
    h: usize,
    camera: CaptureCamera,
    hud: Option<[f64; 4]>,
}

fn open(dir: &Path, k: u32) -> Option<File> {
    let pic = image::open(dir.join(format!("{k:04}.png")))
        .ok()?
        .to_rgba8();
    let probe = image::open(dir.join(format!("{k:04}.probe.png")))
        .ok()?
        .to_rgba8();
    let state: DevState =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{k:04}.json"))).ok()?)
            .ok()?;
    let stamp = state.terrain_stamp?;
    let (w, h) = (pic.width() as usize, pic.height() as usize);
    let r = stamp.hud_rect_px.map(f64::from);
    Some(File {
        camera: vd_client_harness::pop::frame_camera(
            stamp.eye_body_m,
            stamp.camera_body_xyzw,
            w,
            h,
        ),
        hud: ((r[2] > r[0]) & (r[3] > r[1])).then_some(r),
        w,
        h,
        rgba: pic.into_raw(),
        probe: probe.into_raw(),
    })
}

fn frame(f: &File) -> StarFrame<'_> {
    StarFrame {
        rgba: &f.rgba,
        width: f.w,
        height: f.h,
        probe: &f.probe,
        probe_width: f.w,
        probe_height: f.h,
        hud: f.hud,
        camera: Some(f.camera),
    }
}

/// A camera moved to the origin: a star's direction projects through it exactly.
fn at_origin(c: &CaptureCamera) -> CaptureCamera {
    CaptureCamera {
        eye: DVec3::ZERO,
        target: c.target - c.eye,
        ..*c
    }
}

/// THE WHOLE CENSUS of the kept frames, as the picture gate reads it (`read_star_stand`): the still
/// against today's first frame, the turn paired by the nearest camera within one of today's steps.
fn census(folder: &Path) {
    use vd_client_harness::stars::{CENSUS_FRAMES, StarCensus, turn_between};
    let print = |what: &str, c: &StarCensus| {
        let lost = c.missing_events();
        let hidden = lost
            .iter()
            .filter(|e| e.sighting == vd_client_harness::stars::Sighting::Hidden)
            .count();
        println!(
            "{what}: {} frames, {} checks, {} present, {} out of view, {} under the overlay, {} lost ({hidden} hidden), {} extra; the light pooled {:+.4}, the worst frame {:+.4}; {:?}",
            c.frame_count(),
            c.checks(),
            c.present(),
            c.out_of_view(),
            c.overlaid(),
            lost.len(),
            c.extra(),
            c.light_change(),
            c.frames
                .iter()
                .map(vd_client_harness::stars::FrameComparison::light_change)
                .fold(0.0_f64, |a, b| if b.abs() > a.abs() { b } else { a }),
            c.verdict()
        );
    };
    if let Some(first) = open(&folder.join("today/still"), 0) {
        let reference = StarReference::of(&frame(&first));
        let mut c = StarCensus::default();
        let mut k = 0u32;
        while c.frame_count() < CENSUS_FRAMES {
            let Some(f) = open(&folder.join("candidate/still"), k) else {
                break;
            };
            c.add(&reference, &frame(&f));
            k += 1;
        }
        print("still", &c);
    }
    let mut todays: Vec<(vd_client_harness::camera::CaptureCamera, StarReference)> = Vec::new();
    let mut j = 0u32;
    while let Some(f) = open(&folder.join("today/turn"), j) {
        todays.push((f.camera, StarReference::of(&frame(&f))));
        j += 1;
    }
    let cameras: Vec<_> = todays.iter().map(|t| t.0).collect();
    let mut steps: Vec<f64> = cameras
        .windows(2)
        .map(|p| turn_between(&p[0], &p[1]))
        .collect();
    steps.sort_by(f64::total_cmp);
    let step = steps[steps.len() / 2];
    let mut c = StarCensus::default();
    let (mut k, mut unpaired) = (0u32, 0u32);
    let mut per_frame = Vec::new();
    let mut pairs_used: Vec<usize> = Vec::new();
    while c.frame_count() < CENSUS_FRAMES {
        let Some(f) = open(&folder.join("candidate/turn"), k) else {
            break;
        };
        k += 1;
        let Some((r, turn)) = nearest_frame(&cameras, &f.camera) else {
            break;
        };
        if turn > step {
            unpaired += 1;
            continue;
        }
        c.add(&todays[r].1, &frame(&f));
        pairs_used.push(r);
        let last = c.frames.last().expect("a frame");
        let checks = last.sightings.iter().filter(|s| s.is_check()).count();
        let present = last.sightings.iter().filter(|s| s.is_present()).count();
        if checks == 0 && !last.sightings.is_empty() {
            let away = last
                .sightings
                .iter()
                .filter(|s| **s == vd_client_harness::stars::Sighting::OutOfView)
                .count();
            println!(
                "  frame {}: no check — {} reference stars, {away} out of view, {} under the overlay",
                k - 1,
                last.sightings.len(),
                last.sightings.len() - away
            );
        }
        if last.light_change().abs() > 0.15 {
            for (star, sighting) in last.reference.iter().zip(&last.sightings) {
                if let vd_client_harness::stars::Sighting::Present(j) = *sighting {
                    let c = last.read.stars[j];
                    println!(
                        "  frame {}: today's star ({:.2}, {:.2}) peak {} over {} light {:.5} area {} -> the candidate's ({:.2}, {:.2}) peak {} over {} light {:.5} area {}",
                        k - 1,
                        star.x,
                        star.y,
                        star.peak,
                        star.background,
                        star.light,
                        star.area,
                        c.x,
                        c.y,
                        c.peak,
                        c.background,
                        c.light,
                        c.area
                    );
                }
            }
        }
        per_frame.push((k - 1, checks, checks - present, last.light_change()));
    }
    println!(
        "turning: today's median step {:.2}°, {unpaired} candidate frames unpaired",
        step.to_degrees()
    );
    // THE CROWDING of the turn's reference stars: each checked star's nearest other star of today's
    // frame, in pixels, for the present and the lost apart (bins: under 2, 2–3, 3–4, 4–6, 6 and over).
    let bin = |d: f64| match d {
        d if d < 2.0 => 0,
        d if d < 3.0 => 1,
        d if d < 4.0 => 2,
        d if d < 6.0 => 3,
        _ => 4,
    };
    let (mut present_bins, mut lost_bins) = ([0u64; 5], [0u64; 5]);
    for (fc, r) in c.frames.iter().zip(&pairs_used) {
        let reference = &todays[*r].1;
        for (k, (sighting, &i)) in fc.sightings.iter().zip(&reference.checked).enumerate() {
            let _ = k;
            if !sighting.is_check() {
                continue;
            }
            let s = reference.read.stars[i];
            let nearest = reference
                .read
                .stars
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, o)| ((o.x - s.x).powi(2) + (o.y - s.y).powi(2)).sqrt())
                .fold(f64::MAX, f64::min);
            if sighting.is_present() {
                present_bins[bin(nearest)] += 1;
            } else {
                lost_bins[bin(nearest)] += 1;
                if nearest >= 4.0 {
                    let within = |rad: f64| {
                        reference
                            .read
                            .stars
                            .iter()
                            .filter(|o| ((o.x - s.x).powi(2) + (o.y - s.y).powi(2)).sqrt() < rad)
                            .count()
                            - 1
                    };
                    println!(
                        "  a lost star clear of 4 px: today's frame {r} star at ({:.1}, {:.1}) peak {} over {}, light {:.5}; the nearest other {nearest:.1} px; stars within 6 / 9 px: {} / {}; the sighting {sighting:?}",
                        s.x,
                        s.y,
                        s.peak,
                        s.background,
                        s.light,
                        within(6.0),
                        within(9.0)
                    );
                }
            }
        }
    }
    println!(
        "the nearest other star of today's frame (under 2, 2–3, 3–4, 4–6, 6 px and over): present {present_bins:?}, lost {lost_bins:?}"
    );
    print("turning", &c);
    for (k, checks, lost, light) in per_frame.iter().filter(|p| p.1 > 0) {
        println!("  frame {k}: {checks} checks, {lost} lost, the light {light:+.4}");
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let folder = Path::new(&a[0]).join("stars");
    if a.get(1).is_some_and(|v| v == "census") {
        census(&folder);
        return;
    }
    let k: u32 = a.get(1).map_or(0, |v| v.parse().expect("a frame"));
    let mut cams = Vec::new();
    let mut j = 0u32;
    while let Some(f) = open(&folder.join("today/turn"), j) {
        cams.push(f.camera);
        j += 1;
    }
    let cand = open(&folder.join("candidate/turn"), k).expect("the candidate frame");
    let (r, turn) = nearest_frame(&cams, &cand.camera).expect("today's frames");
    let today = open(&folder.join("today/turn"), r as u32).expect("today's frame");
    println!(
        "candidate {k} pairs with today {r}, {:.2}° apart",
        turn.to_degrees()
    );
    let reference = StarReference::of(&frame(&today));
    let read = stars_in(&frame(&cand));
    let (from, to) = (at_origin(&today.camera), at_origin(&cand.camera));
    let (mut open_sky, mut covered, mut far) = (0, 0, 0);
    // THE OFFSETS: each predicted place to its nearest candidate star (dx, dy, distance), the matched
    // (within 1.5 px) and the lost apart. A camera error moves every place one way; merged blobs
    // scatter the lost ones in every direction.
    let mut matched: Vec<(f64, f64)> = Vec::new();
    let mut lost: Vec<(f64, f64, f64, f64)> = Vec::new();
    for &i in &reference.checked {
        let s = reference.read.stars[i];
        let Some(p) = to.project_point(from.unproject(s.x, s.y, 1.0)) else {
            continue;
        };
        let inside = (p.x >= 9.0)
            & (p.y >= 9.0)
            & (p.x <= cand.w as f64 - 9.0)
            & (p.y <= cand.h as f64 - 9.0);
        if !inside {
            continue;
        }
        let (nearest, dx, dy) = read
            .stars
            .iter()
            .map(|c| {
                (
                    ((c.x - p.x).powi(2) + (c.y - p.y).powi(2)).sqrt(),
                    c.x - p.x,
                    c.y - p.y,
                )
            })
            .fold((f64::MAX, 0.0, 0.0), |a, b| if b.0 < a.0 { b } else { a });
        if nearest <= 1.5 {
            matched.push((dx, dy));
            continue;
        }
        lost.push((p.x, p.y, dx, dy));
        let (px, py) = (p.x as usize, p.y as usize);
        let o = (py * cand.w + px) * 4;
        let kind = decode_probe([cand.probe[o], cand.probe[o + 1], cand.probe[o + 2]]).kind;
        let rgb = [cand.rgba[o], cand.rgba[o + 1], cand.rgba[o + 2]];
        if kind == 0 {
            open_sky += 1;
        } else {
            covered += 1;
        }
        far += usize::from(nearest > 6.0);
        println!(
            "  lost: today ({:.1}, {:.1}) peak {} over {} -> ({:.1}, {:.1}); the candidate's probe kind {kind}, picture {rgb:?}; the nearest candidate star {nearest:.1} px",
            s.x, s.y, s.peak, s.background, p.x, p.y
        );
    }
    println!(
        "{} reference stars; lost on open sky {open_sky}, under the candidate's ground {covered}; {far} with no candidate star within 6 px",
        reference.checked.len()
    );
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
    };
    println!(
        "the matched: {} stars, the median offset ({:+.2}, {:+.2}) px",
        matched.len(),
        median(matched.iter().map(|m| m.0).collect()),
        median(matched.iter().map(|m| m.1).collect())
    );
    let near: Vec<&(f64, f64, f64, f64)> = lost
        .iter()
        .filter(|l| (l.2 * l.2 + l.3 * l.3).sqrt() <= 6.0)
        .collect();
    println!(
        "the lost with a candidate star within 6 px: {}, the median offset ({:+.2}, {:+.2}) px, the mean direction's length {:.2} (1 = one way, 0 = every way)",
        near.len(),
        median(near.iter().map(|l| l.2).collect()),
        median(near.iter().map(|l| l.3).collect()),
        {
            let (sx, sy) = near.iter().fold((0.0, 0.0), |a, l| {
                let d = (l.2 * l.2 + l.3 * l.3).sqrt().max(1e-9);
                (a.0 + l.2 / d, a.1 + l.3 / d)
            });
            (sx * sx + sy * sy).sqrt() / (near.len().max(1) as f64)
        }
    );
    let (ys, xs): (Vec<f64>, Vec<f64>) = lost.iter().map(|l| (l.1, l.0)).unzip();
    println!(
        "the lost places: x {:.0}..{:.0}, y {:.0}..{:.0}",
        xs.iter().copied().fold(f64::MAX, f64::min),
        xs.iter().copied().fold(f64::MIN, f64::max),
        ys.iter().copied().fold(f64::MAX, f64::min),
        ys.iter().copied().fold(f64::MIN, f64::max)
    );
}
