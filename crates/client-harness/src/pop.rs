//! ★ THE POP DETECTOR (slice 8 step 6; ruling V14 M8-3, D8-2, D8-5) — the pure half.
//!
//! A pop is a change of the picture that happens in ONE frame instead of across a crossfade
//! band: a chunk that appears at full detail at once, a hole, a shading step where one rung
//! hands over to the next. Under the seamless law (SL8) a pop is a defect. The ruling's
//! instrument is a FRAME-TO-FRAME DIFFERENCE ON THE RUNG BOUNDARIES: two consecutive frames, and
//! at every pixel where the rung that drew it changed between the two, the colour before and
//! after must agree within the picture's own noise.
//!
//! **The reprojection.** The eye moves between the two frames (four metres a frame at 240 m/s),
//! so a pixel of the second frame is compared with the pixel of the FIRST frame that showed the
//! same ground: the probe states each pixel's distance from the eye, the stamp states each
//! frame's drawn camera, so the second frame's pixel becomes a point in the body's frame and
//! projects into the first frame through the same pixel model the ruler gate proved
//! ([`CaptureCamera::project_point`]). Near the eye a stated distance (half a cell coarse) is too
//! coarse for that, so pixels nearer than [`near_limit_m`] are left out; on a straight leg the
//! rung boundaries stand far past it, and the limit each pair used is reported, since an eye
//! whose stated travel is not its motion (a spinning parent's chord, D-TERRAIN-5 item 21) raises
//! it into the picture.
//!
//! **The neighbourhood.** A reprojected pixel lands between pixels of the first frame, and the
//! ground's shading changes by tens of levels from one pixel to the next at a crease or a
//! shadow's edge, so the nearest pixel alone reads a step where the ground did not change.
//! MEASURED at 240 m/s with the nearest pixel: a floor of 43 levels, which hides every pop
//! below it. Every pixel therefore compares against the BEST match among the reprojected pixel
//! and its eight neighbours ([`NEIGHBOURHOOD`]): a shading edge that merely slid a pixel reads
//! zero, a region that changed its shade reads its change. A pixel is a BOUNDARY pixel when any
//! terrain pixel of that neighbourhood was drawn by another rung than the pixel's own — the rung
//! is read from the whole neighbourhood, never from the centre alone (a boundary that slid a
//! pixel and a half would otherwise count as still and put its own handover into the floor).
//!
//! **The floor.** The picture's own noise is measured on the same pairs: the still pixels'
//! steps, pooled over every pair of a leg, give a distribution; the step under which
//! [`FLOOR_SHARE`] of them lie is the floor. A boundary pixel that steps past the floor is a pop
//! candidate; the widest step and their count are the reading. The histograms are what a pair
//! carries, so a leg's sum is the pooled histograms and every number is derived from them ONCE —
//! a floor, a count past it and a widest that cannot contradict each other.
//!
//! Example: the hull flies at 240 m/s one kilometre over the home planet. Between two frames the
//! rung-2/rung-3 boundary at 3.5 km moved four metres nearer; 600 pixels along it changed rung.
//! Reprojected, 598 of them differ by at most two levels, under the leg's floor; two differ by
//! 30 levels, where a rung-3 crease stood under a rung-2 slope — a pop the detector names by
//! rung and count.

use std::collections::BTreeMap;

use vd_core::glam::{DQuat, DVec3};

use crate::camera::{CaptureCamera, FIT_FOV_Y};
use crate::probe::{PROBE_KIND_TERRAIN, decode_probe};

/// The share of the still pixels' steps under the floor: the floor is the step under which
/// this share of them lie (a thousandth stands above it).
pub const FLOOR_SHARE: f64 = 0.999;

/// How far a reprojected pixel may miss for a stated distance's own coarseness (half a cell of
/// its rung): the near limit keeps the miss under this many pixels.
pub const NEAR_LIMIT_PX: f64 = 0.25;

/// The neighbourhood a reprojected pixel is matched and classified in: this many pixels each way.
pub const NEIGHBOURHOOD: i64 = 1;

/// A histogram of colour steps, one bin per level.
pub type StepHist = Box<[u64; 256]>;

fn empty_hist() -> StepHist {
    Box::new([0u64; 256])
}

/// One frame of a pair: its picture (RGBA, eight bits), its probe (the same size), its size and
/// the camera it was drawn with, in the body's frame.
pub struct Frame<'a> {
    pub rgba: &'a [u8],
    pub probe: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub camera: CaptureCamera,
}

impl Frame<'_> {
    /// Whether the frame's bytes are one picture and one probe of its stated size.
    fn well_formed(&self) -> bool {
        let n = self.width * self.height * 4;
        (n > 0) & (self.rgba.len() == n) & (self.probe.len() == n)
    }
}

/// The camera a frame was drawn with, from the stamp's drawn eye and camera rotation in the
/// body's frame (the same field of view every pilot capture uses).
#[must_use]
pub fn frame_camera(
    eye_body_m: [f64; 3],
    camera_body_xyzw: [f64; 4],
    width: usize,
    height: usize,
) -> CaptureCamera {
    let rot = DQuat::from_array(camera_body_xyzw).normalize();
    let eye = DVec3::from_array(eye_body_m);
    CaptureCamera {
        eye,
        target: eye + rot * DVec3::NEG_Z,
        up: rot * DVec3::Y,
        fov_y: FIT_FOV_Y,
        width,
        height,
    }
}

/// THE NEAR LIMIT in metres: a pixel's stated distance is within half a cell of the true one,
/// and a depth error `e` moves the reprojected pixel by about `focal · travel · e / d²` pixels
/// for an eye that travelled `travel` metres; the limit is the distance where that miss equals
/// [`NEAR_LIMIT_PX`]. Zero travel (a still eye) limits nothing.
#[must_use]
pub fn near_limit_m(cell_m: f64, travel_m: f64, focal_px: f64) -> f64 {
    (focal_px * travel_m * cell_m * 0.5 / NEAR_LIMIT_PX).sqrt()
}

/// The focal length in pixels of a frame: half its height over the tangent of half the field.
#[must_use]
pub fn focal_px(height: usize, fov_y: f64) -> f64 {
    height as f64 * 0.5 / (fov_y * 0.5).tan()
}

/// THE FLOOR of a histogram of steps: the smallest step under which at least `share` of the
/// counted pixels lie (zero for an empty histogram).
#[must_use]
pub fn floor_of(hist: &[u64; 256], share: f64) -> u8 {
    let total: u64 = hist.iter().sum();
    let need = (total as f64 * share).ceil() as u64;
    let mut seen = 0u64;
    let mut step = 0usize;
    while step < 255 && seen + hist[step] < need {
        seen += hist[step];
        step += 1;
    }
    step as u8
}

/// The widest step a histogram holds.
#[must_use]
pub fn widest_of(hist: &[u64; 256]) -> u8 {
    hist.iter().rposition(|n| *n > 0).unwrap_or(0) as u8
}

/// How many steps of a histogram lie past a floor.
#[must_use]
pub fn over_floor(hist: &[u64; 256], floor: u8) -> u64 {
    hist[usize::from(floor) + 1..].iter().sum()
}

/// What one rung boundary read: the finer of the two rungs, how many pixels crossed it, how
/// many stepped past the floor, and the widest step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundaryRead {
    pub finer: u8,
    pub crossed: u64,
    pub over_floor: u64,
    pub widest: u8,
}

/// What a pair of consecutive frames read — or a leg's pairs, summed: the counts add and the
/// histograms pool, so every derived number is one reading of the whole.
#[derive(Clone, Debug, PartialEq)]
pub struct PairRead {
    /// Terrain pixels of the second frame compared with the first (past the near limit, inside
    /// the first frame, on terrain there too).
    pub compared: u64,
    /// Terrain pixels nearer than the near limit, left out.
    pub near_skipped: u64,
    /// Terrain pixels whose ground fell outside the first frame, or on no terrain there.
    pub off_frame: u64,
    /// The widest near limit any pair applied, in metres (the finest rung's is the smallest; a
    /// spinning parent's chord raises it into the picture).
    pub near_limit_m: f64,
    /// The steps of the still pixels, and of the boundary pixels.
    pub still: StepHist,
    pub flipped: StepHist,
    /// The boundary pixels' steps per boundary, keyed by the finer rung.
    pub boundaries: BTreeMap<u8, StepHist>,
}

impl Default for PairRead {
    fn default() -> PairRead {
        PairRead {
            compared: 0,
            near_skipped: 0,
            off_frame: 0,
            near_limit_m: 0.0,
            still: empty_hist(),
            flipped: empty_hist(),
            boundaries: BTreeMap::new(),
        }
    }
}

impl PairRead {
    /// The floor: the step under which [`FLOOR_SHARE`] of the still pixels lie.
    #[must_use]
    pub fn floor(&self) -> u8 {
        floor_of(&self.still, FLOOR_SHARE)
    }

    /// How many pixels crossed a rung boundary.
    #[must_use]
    pub fn crossed(&self) -> u64 {
        self.flipped.iter().sum()
    }

    /// Boundary pixels that stepped past the floor.
    #[must_use]
    pub fn over_floor(&self) -> u64 {
        over_floor(&self.flipped, self.floor())
    }

    /// The widest boundary step.
    #[must_use]
    pub fn widest(&self) -> u8 {
        widest_of(&self.flipped)
    }

    /// The reading per boundary, finest first, against the one floor.
    #[must_use]
    pub fn boundary_reads(&self) -> Vec<BoundaryRead> {
        let floor = self.floor();
        self.boundaries
            .iter()
            .map(|(finer, hist)| BoundaryRead {
                finer: *finer,
                crossed: hist.iter().sum(),
                over_floor: over_floor(hist, floor),
                widest: widest_of(hist),
            })
            .collect()
    }
}

/// THE BEST MATCH of the second frame's pixel `i` (a byte offset) among the first frame's
/// terrain pixels around `(qx, qy)`, and whether any of them was drawn by another rung than
/// `rung`: the smallest widest-channel step over the [`NEIGHBOURHOOD`] (the centre included),
/// and the finest rung a differing neighbour was drawn at. `None` when no terrain pixel stands
/// in the neighbourhood (sky, or the frame's edge).
fn best_match(
    before: &Frame<'_>,
    after_rgba: &[u8],
    i: usize,
    qx: usize,
    qy: usize,
    rung: u8,
) -> Option<(u8, Option<u8>)> {
    let mut best: Option<u8> = None;
    let mut other_rung: Option<u8> = None;
    let mut dy = -NEIGHBOURHOOD;
    while dy <= NEIGHBOURHOOD {
        let mut dx = -NEIGHBOURHOOD;
        while dx <= NEIGHBOURHOOD {
            let (x, y) = (qx as i64 + dx, qy as i64 + dy);
            let inside =
                (x >= 0) & (y >= 0) & (x < before.width as i64) & (y < before.height as i64);
            if inside {
                let j = (y as usize * before.width + x as usize) * 4;
                let pa = decode_probe([before.probe[j], before.probe[j + 1], before.probe[j + 2]]);
                if pa.kind == PROBE_KIND_TERRAIN {
                    let step = (0..3)
                        .map(|k| after_rgba[i + k].abs_diff(before.rgba[j + k]))
                        .max()
                        .unwrap_or(0);
                    best = Some(best.map_or(step, |b| b.min(step)));
                    if pa.rung != rung {
                        other_rung = Some(other_rung.map_or(pa.rung, |r| r.min(pa.rung)));
                    }
                }
            }
            dx += 1;
        }
        dy += 1;
    }
    best.map(|b| (b, other_rung))
}

/// THE JUDGE of a pair: the second frame's terrain pixels reprojected into the first; the rung
/// boundaries' steps and the still pixels' steps as histograms. `cell_m` names a rung's cell in
/// metres (the ladder's own). A frame that is not one picture and one probe of its stated size
/// reads nothing.
#[must_use]
pub fn judge_pair(before: &Frame<'_>, after: &Frame<'_>, cell_m: &dyn Fn(u8) -> f64) -> PairRead {
    let mut read = PairRead::default();
    if !(before.well_formed() & after.well_formed()) {
        return read;
    }
    let travel = (after.camera.eye - before.camera.eye).length();
    let focal = focal_px(after.height, after.camera.fov_y);
    let mut y = 0usize;
    while y < after.height {
        let mut x = 0usize;
        while x < after.width {
            let i = (y * after.width + x) * 4;
            let pb = decode_probe([after.probe[i], after.probe[i + 1], after.probe[i + 2]]);
            if pb.kind != PROBE_KIND_TERRAIN {
                x += 1;
                continue;
            }
            let cell = cell_m(pb.rung);
            let d = cell * f64::from(pb.cells);
            let limit = near_limit_m(cell, travel, focal);
            read.near_limit_m = read.near_limit_m.max(limit);
            if d < limit {
                read.near_skipped += 1;
                x += 1;
                continue;
            }
            let point = after.camera.unproject(x as f64 + 0.5, y as f64 + 0.5, d);
            let q = before.camera.project_point(point);
            let hit = q.and_then(|q| {
                let inside = (q.x >= 0.0)
                    & (q.y >= 0.0)
                    & (q.x < before.width as f64)
                    & (q.y < before.height as f64);
                inside.then_some((q.x as usize, q.y as usize))
            });
            let matched =
                hit.and_then(|(qx, qy)| best_match(before, after.rgba, i, qx, qy, pb.rung));
            let Some((step, other)) = matched else {
                read.off_frame += 1;
                x += 1;
                continue;
            };
            read.compared += 1;
            match other {
                None => read.still[usize::from(step)] += 1,
                Some(other_rung) => {
                    read.flipped[usize::from(step)] += 1;
                    read.boundaries
                        .entry(pb.rung.min(other_rung))
                        .or_insert_with(empty_hist)[usize::from(step)] += 1;
                }
            }
            x += 1;
        }
        y += 1;
    }
    read
}

fn add_hist(into: &mut [u64; 256], from: &[u64; 256]) {
    let mut k = 0;
    while k < 256 {
        into[k] += from[k];
        k += 1;
    }
}

/// The sum of many pairs' readings: the counts add, the histograms pool, the near limit takes
/// the widest.
#[must_use]
pub fn sum_reads(reads: &[PairRead]) -> PairRead {
    let mut total = PairRead::default();
    for r in reads {
        total.compared += r.compared;
        total.near_skipped += r.near_skipped;
        total.off_frame += r.off_frame;
        total.near_limit_m = total.near_limit_m.max(r.near_limit_m);
        add_hist(&mut total.still, &r.still);
        add_hist(&mut total.flipped, &r.flipped);
        for (finer, hist) in &r.boundaries {
            add_hist(
                total.boundaries.entry(*finer).or_insert_with(empty_hist),
                hist,
            );
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{ProbePixel, encode_probe};

    fn cell(rung: u8) -> f64 {
        f64::from(1u32 << rung)
    }

    /// A frame of `w × h` whose every pixel is terrain at `rung`, `cells` away, painted `grey`.
    fn painted(
        w: usize,
        h: usize,
        rung: u8,
        cells: u16,
        grey: u8,
        camera: CaptureCamera,
    ) -> (Vec<u8>, Vec<u8>, CaptureCamera) {
        let probe = encode_probe(ProbePixel {
            kind: PROBE_KIND_TERRAIN,
            rung,
            cells,
        });
        let mut rgba = Vec::with_capacity(w * h * 4);
        let mut pr = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            rgba.extend_from_slice(&[grey, grey, grey, 255]);
            pr.extend_from_slice(&[probe[0], probe[1], probe[2], 255]);
        }
        (rgba, pr, camera)
    }

    fn still_camera(w: usize, h: usize) -> CaptureCamera {
        frame_camera([1000.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], w, h)
    }

    fn frame<'a>(
        rgba: &'a [u8],
        probe: &'a [u8],
        w: usize,
        h: usize,
        camera: CaptureCamera,
    ) -> Frame<'a> {
        Frame {
            rgba,
            probe,
            width: w,
            height: h,
            camera,
        }
    }

    fn hist_with(steps: &[(u8, u64)]) -> StepHist {
        let mut h = empty_hist();
        for (s, n) in steps {
            h[usize::from(*s)] += n;
        }
        h
    }

    #[test]
    fn the_frame_camera_looks_along_the_rotation_with_its_up() {
        let cam = frame_camera([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], 8, 4);
        assert_eq!(cam.eye, DVec3::new(1.0, 2.0, 3.0));
        assert_eq!(cam.target, DVec3::new(1.0, 2.0, 2.0));
        assert_eq!(cam.up, DVec3::Y);
        assert_eq!((cam.width, cam.height), (8, 4));
        assert!((cam.fov_y - FIT_FOV_Y).abs() < 1e-12);
    }

    /// A pixel's centre unprojected at a distance and projected back lands on itself.
    #[test]
    fn unproject_then_project_round_trips_every_pixel() {
        let cam = frame_camera([5.0, 6.0, 7.0], [0.1, 0.2, 0.3, 0.9], 12, 8);
        for (x, y) in [(0, 0), (11, 7), (6, 4), (3, 1)] {
            let (px, py) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            let point = cam.unproject(px, py, 250.0);
            assert!(((point - cam.eye).length() - 250.0).abs() < 1e-9);
            let back = cam.project_point(point).expect("in front");
            assert!((back.x - px).abs() < 1e-9, "({x}, {y}) x");
            assert!((back.y - py).abs() < 1e-9, "({x}, {y}) y");
        }
    }

    #[test]
    fn the_near_limit_grows_with_travel_and_cell_and_is_zero_when_still() {
        assert_eq!(near_limit_m(1.0, 0.0, 870.0), 0.0);
        let a = near_limit_m(1.0, 4.0, 870.0);
        let b = near_limit_m(4.0, 4.0, 870.0);
        let c = near_limit_m(1.0, 16.0, 870.0);
        assert!((a - (870.0 * 4.0 * 0.5 / NEAR_LIMIT_PX).sqrt()).abs() < 1e-9);
        assert!((b - 2.0 * a).abs() < 1e-9);
        assert!((c - 2.0 * a).abs() < 1e-9);
        assert!((focal_px(720, FIT_FOV_Y) - 360.0 / (FIT_FOV_Y * 0.5).tan()).abs() < 1e-9);
    }

    #[test]
    fn the_floor_the_widest_and_the_count_past_the_floor_read_a_histogram() {
        let mut hist = [0u64; 256];
        assert_eq!(floor_of(&hist, FLOOR_SHARE), 0);
        assert_eq!(widest_of(&hist), 0);
        hist[0] = 997;
        hist[1] = 2;
        hist[7] = 1;
        assert_eq!(floor_of(&hist, FLOOR_SHARE), 1);
        assert_eq!(floor_of(&hist, 0.5), 0);
        assert_eq!(floor_of(&hist, 1.0), 7);
        assert_eq!(widest_of(&hist), 7);
        assert_eq!(over_floor(&hist, 1), 1);
        assert_eq!(over_floor(&hist, 7), 0);
        let mut top = [0u64; 256];
        top[255] = 1;
        assert_eq!(floor_of(&top, 1.0), 255);
        assert_eq!(over_floor(&top, 255), 0);
    }

    /// The same frame twice, from a still eye: every pixel compared, none a boundary, the floor
    /// zero, no limit applied.
    #[test]
    fn an_unchanged_frame_from_a_still_eye_reads_nothing() {
        let (w, h) = (6, 4);
        let (rgba, probe, cam) = painted(w, h, 2, 500, 120, still_camera(w, h));
        let a = frame(&rgba, &probe, w, h, cam);
        let b = frame(&rgba, &probe, w, h, still_camera(w, h));
        let read = judge_pair(&a, &b, &cell);
        assert_eq!(read.compared, (w * h) as u64);
        assert_eq!(
            (
                read.crossed(),
                read.floor(),
                read.over_floor(),
                read.widest()
            ),
            (0, 0, 0, 0)
        );
        assert_eq!((read.near_skipped, read.off_frame), (0, 0));
        assert_eq!(read.near_limit_m, 0.0);
        assert!(read.boundary_reads().is_empty());
    }

    /// A boundary moved: two pixels of the top row changed rung; they and every pixel whose
    /// neighbourhood holds them are boundary pixels; one of the two stepped 30 levels.
    #[test]
    fn a_rung_boundary_is_read_with_its_neighbourhood_and_its_widest_step() {
        let (w, h) = (6, 4);
        let (rgba_a, probe_a, cam_a) = painted(w, h, 2, 500, 120, still_camera(w, h));
        let (mut rgba_b, mut probe_b, cam_b) = painted(w, h, 2, 500, 120, still_camera(w, h));
        // Pixels 3 and 4 of the top row now belong to rung 3 (at the same distance in metres).
        for x in [3usize, 4] {
            let p = encode_probe(ProbePixel {
                kind: PROBE_KIND_TERRAIN,
                rung: 3,
                cells: 250,
            });
            probe_b[x * 4..x * 4 + 3].copy_from_slice(&p);
        }
        // Pixel 4 also shades 30 levels darker; pixel 1 (a still pixel) drifts by one level.
        rgba_b[4 * 4] = 90;
        rgba_b[4] = 121;
        let a = frame(&rgba_a, &probe_a, w, h, cam_a);
        let b = frame(&rgba_b, &probe_b, w, h, cam_b);
        let read = judge_pair(&a, &b, &cell);
        assert_eq!(read.compared, (w * h) as u64);
        // The boundary pixels: in the first frame every pixel is rung 2, so a second-frame pixel
        // of rung 3 (3 and 4) sees another rung in its neighbourhood; a second-frame pixel of
        // rung 2 sees only rung 2 there. Two boundary pixels, keyed by the finer rung 2.
        assert_eq!(read.crossed(), 2);
        // 22 still pixels, one at step 1: the floor at a thousandth's share is that step.
        assert_eq!(read.floor(), 1);
        assert_eq!((read.over_floor(), read.widest()), (1, 30));
        assert_eq!(
            read.boundary_reads(),
            vec![BoundaryRead {
                finer: 2,
                crossed: 2,
                over_floor: 1,
                widest: 30
            }]
        );
        // The other way round: the first frame holds the rung-3 pair, the second is all rung 2:
        // the second frame's pixels whose neighbourhood holds a rung-3 pixel are the boundary —
        // pixels 2..5 of the top row and of the row below it.
        let read = judge_pair(&b, &a, &cell);
        assert_eq!(read.crossed(), 8);
        assert_eq!(read.boundary_reads()[0].finer, 2);
    }

    /// A shading edge that slid one pixel between the frames reads zero through the
    /// neighbourhood; a pixel darker than every neighbour reads its own step; a corner pixel has
    /// fewer neighbours and still reads.
    #[test]
    fn the_neighbourhood_forgives_a_slid_edge_and_reads_a_changed_shade() {
        let (w, h) = (6, 4);
        let (mut rgba_a, probe, cam) = painted(w, h, 2, 500, 120, still_camera(w, h));
        let (mut rgba_b, _, _) = painted(w, h, 2, 500, 120, still_camera(w, h));
        // A bright column at x = 2 in the first frame, at x = 3 in the second: an edge that slid.
        for y in 0..h {
            rgba_a[(y * w + 2) * 4] = 200;
            rgba_b[(y * w + 3) * 4] = 200;
        }
        // The corner pixel (0, 0) of the second frame shades 50 levels darker than everything.
        rgba_b[0] = 70;
        let a = frame(&rgba_a, &probe, w, h, cam);
        let b = frame(&rgba_b, &probe, w, h, still_camera(w, h));
        let read = judge_pair(&a, &b, &cell);
        assert_eq!(read.compared, (w * h) as u64);
        // 23 still pixels at zero, one (the corner) at 50: the floor is that step.
        assert_eq!(read.floor(), 50);
        assert_eq!(best_match(&a, &rgba_b, 0, 0, 0, 2), Some((50, None)));
        assert_eq!(best_match(&a, &rgba_b, 3 * 4, 3, 0, 2), Some((0, None)));
    }

    /// An eye that travelled: the near pixels are skipped and the limit reported; a frame turned
    /// around sees none of the other's ground; a first frame of sky is off-frame too.
    #[test]
    fn travel_skips_the_near_pixels_and_a_turned_frame_is_off_frame() {
        let (w, h) = (6, 4);
        let far = still_camera(w, h);
        let mut near_eye = still_camera(w, h);
        near_eye.eye += DVec3::new(0.0, 0.0, -4.0);
        near_eye.target += DVec3::new(0.0, 0.0, -4.0);
        // Rung 0 at 5 cells: five metres, under the near limit for a four-metre travel.
        let (rgba_a, probe_a, _) = painted(w, h, 0, 5, 100, far);
        let (rgba_b, probe_b, _) = painted(w, h, 0, 5, 100, far);
        let a = frame(&rgba_a, &probe_a, w, h, far);
        let b = frame(&rgba_b, &probe_b, w, h, near_eye);
        let read = judge_pair(&a, &b, &cell);
        assert_eq!(read.near_skipped, (w * h) as u64);
        assert_eq!(read.compared, 0);
        let expected = near_limit_m(1.0, 4.0, focal_px(h, FIT_FOV_Y));
        assert!((read.near_limit_m - expected).abs() < 1e-9);
        // Turned around: the ground of the second frame lies behind the first frame's camera.
        let turned = frame_camera([1000.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], w, h);
        let (rgba_c, probe_c, _) = painted(w, h, 2, 500, 100, far);
        let c = frame(&rgba_c, &probe_c, w, h, turned);
        let read = judge_pair(&a, &c, &cell);
        assert_eq!(read.off_frame, (w * h) as u64);
        assert_eq!(read.compared, 0);
        // Sky in the first frame where the second sees ground: off-frame as well.
        let mut sky_probe = probe_c.clone();
        for px in sky_probe.chunks_exact_mut(4) {
            px[0] = 0;
        }
        let sky = frame(&rgba_c, &sky_probe, w, h, far);
        let ground = frame(&rgba_c, &probe_c, w, h, far);
        let read = judge_pair(&sky, &ground, &cell);
        assert_eq!(read.off_frame, (w * h) as u64);
        // A pixel of the first frame just outside the viewport: a wider second frame.
        let (rgba_d, probe_d, _) = painted(w + 2, h, 2, 500, 100, still_camera(w + 2, h));
        let d = frame(&rgba_d, &probe_d, w + 2, h, still_camera(w + 2, h));
        let read = judge_pair(&ground, &d, &cell);
        assert!(read.off_frame > 0, "{read:?}");
        assert!(read.compared > 0, "{read:?}");
    }

    /// A frame of no size, or of bytes that do not match its size, reads nothing; the sky of
    /// the second frame is never compared.
    #[test]
    fn a_malformed_frame_reads_nothing_and_the_sky_is_not_compared() {
        let (w, h) = (6, 4);
        let (rgba, probe, cam) = painted(w, h, 2, 500, 120, still_camera(w, h));
        let empty = frame(&[], &[], 0, 0, still_camera(0, 0));
        let full = frame(&rgba, &probe, w, h, cam);
        assert_eq!(judge_pair(&empty, &full, &cell), PairRead::default());
        assert_eq!(judge_pair(&full, &empty, &cell), PairRead::default());
        let short = frame(&rgba[..8], &probe, w, h, still_camera(w, h));
        assert_eq!(judge_pair(&full, &short, &cell), PairRead::default());
        let short_probe = frame(&rgba, &probe[..8], w, h, still_camera(w, h));
        assert_eq!(judge_pair(&short_probe, &full, &cell), PairRead::default());
        let mut sky = probe.clone();
        for px in sky.chunks_exact_mut(4) {
            px[0] = 0;
        }
        let none = frame(&rgba, &sky, w, h, still_camera(w, h));
        let read = judge_pair(&full, &none, &cell);
        assert_eq!(read.compared, 0);
    }

    /// The sum pools the histograms: one floor for the leg, one count past it, one widest —
    /// never a floor from one pair against a count from another.
    #[test]
    fn the_sum_of_reads_pools_the_histograms_into_one_reading() {
        let mut a = PairRead {
            compared: 10,
            near_skipped: 1,
            off_frame: 2,
            near_limit_m: 40.0,
            still: hist_with(&[(0, 9), (9, 1)]),
            flipped: hist_with(&[(3, 3)]),
            ..PairRead::default()
        };
        a.boundaries.insert(2, hist_with(&[(3, 3)]));
        let mut b = PairRead {
            compared: 2000,
            near_skipped: 0,
            off_frame: 0,
            near_limit_m: 12.0,
            still: hist_with(&[(0, 1996), (2, 4)]),
            flipped: hist_with(&[(1, 1), (30, 2)]),
            ..PairRead::default()
        };
        b.boundaries.insert(1, hist_with(&[(1, 1)]));
        b.boundaries.insert(2, hist_with(&[(30, 2)]));
        let sum = sum_reads(&[a.clone(), b.clone()]);
        assert_eq!(
            (sum.compared, sum.near_skipped, sum.off_frame),
            (2010, 1, 2)
        );
        assert!((sum.near_limit_m - 40.0).abs() < 1e-12);
        // Pair a alone would read a floor of 9 (its one still pixel at 9 is past a thousandth);
        // pooled, 2 010 still pixels with one at 9: the floor is 2 and the 30s stand past it.
        assert_eq!(a.floor(), 9);
        assert_eq!(sum.floor(), 2);
        assert_eq!((sum.crossed(), sum.over_floor(), sum.widest()), (6, 5, 30));
        assert_eq!(
            sum.boundary_reads(),
            vec![
                BoundaryRead {
                    finer: 1,
                    crossed: 1,
                    over_floor: 0,
                    widest: 1
                },
                BoundaryRead {
                    finer: 2,
                    crossed: 5,
                    over_floor: 5,
                    widest: 30
                },
            ]
        );
        assert_eq!(sum_reads(&[]), PairRead::default());
    }
}
