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

use vd_client::reconstruction::{
    ReconstructionSettings, corner_factor, depth_separation, nearest_sample, sample_point,
};

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

/// THE OVERLAY'S MARGIN in pixels: the stamp's rectangle grown by the glyphs' antialiasing, as
/// the picture gate grows it. MEASURED without the mask: the overlay's readouts change every
/// frame while a hull turns, the probe marks the ground under them as terrain, and those digit
/// pixels set the turning leg's floor at 70 levels.
pub const HUD_MARGIN_PX: f64 = 2.0;

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
    /// THE OVERLAY'S RECTANGLE (left, top, right, bottom, pixels), the stamp's `hud_rect_px`:
    /// nothing under it is judged or matched — its readouts are not the ground.
    pub hud: Option<[f64; 4]>,
    /// THE FRAME'S JITTER (the temporal path's T2; the stamp's reconstruction row): pixel `i` sampled
    /// the scene at `i + 0.5 + jitter`. Nought without the reconstruction.
    pub jitter_px: [f64; 2],
    /// ★ THE OCCLUSION EDGES, JUDGED APART (the temporal path's T2; the design §12.3): the depth test's
    /// separation per metre of distance (`vd_client::reconstruction::depth_separation`, on the RENDER
    /// grid, where the depth test runs). A pixel whose 3 × 3 neighbourhood holds another kind (the sky,
    /// the ruler) or ground farther apart than this share of the farther distance stands on a
    /// SILHOUETTE, and so does every pixel within the frame's [`Frame::edge_reach_px`] of one, in
    /// EITHER frame of the pair (the second frame's own place, and the first frame's place the pixel
    /// reprojects to: a silhouette that moved between the two); its step goes to [`PairRead::edges`]
    /// and is neither still nor a boundary. Nought judges no edge apart (a picture without the
    /// reconstruction).
    pub edge_separation_per_m: f64,
    /// ★ HOW FAR A SILHOUETTE REACHES in this frame's pixels: one for the render picture (the 3 × 3
    /// itself), [`display_edge_reach_px`] for the display picture, whose pixel gathers render samples
    /// around its nearest one and reads its history along a dilated motion.
    pub edge_reach_px: usize,
}

impl Frame<'_> {
    /// Whether the frame's bytes are one picture and one probe of its stated size.
    fn well_formed(&self) -> bool {
        let n = self.width * self.height * 4;
        (n > 0) & (self.rgba.len() == n) & (self.probe.len() == n)
    }

    /// Whether a pixel lies under the overlay (its rectangle grown by [`HUD_MARGIN_PX`]).
    fn under_overlay(&self, x: usize, y: usize) -> bool {
        self.hud.is_some_and(|r| {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            (px >= r[0] - HUD_MARGIN_PX)
                & (py >= r[1] - HUD_MARGIN_PX)
                & (px <= r[2] + HUD_MARGIN_PX)
                & (py <= r[3] + HUD_MARGIN_PX)
        })
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

/// ★ THE OCCLUSION EDGE'S SEPARATION for a capture frame of `width × height` (T2): the reconstruction's own
/// depth test (`depth_separation` at the proposed settings), with the corner factor of the capture camera's
/// field ([`FIT_FOV_Y`] over the frame's aspect) — one rule for the history and the gates.
///
/// Example: the reference view (1 284 × 720, 45°) separates 0.0264 of the farther distance: a rock 40 m
/// away in front of a valley 3 km away is an edge (a gap of 2 960 m against 79 m).
#[must_use]
pub fn edge_separation_per_m(width: usize, height: usize) -> f64 {
    let y_scale = 1.0 / (FIT_FOV_Y * 0.5).tan();
    let x_scale = y_scale * height as f64 / width as f64;
    depth_separation(
        [width as u32, height as u32],
        corner_factor(x_scale, y_scale),
        &ReconstructionSettings::PROPOSED,
    )
}

/// THE RENDER PICTURE'S SILHOUETTE REACH in its own pixels: the 3 × 3 itself (a render pixel is one
/// sample; nothing gathers its neighbours).
pub const RENDER_EDGE_REACH_PX: usize = 1;

/// ★ THE DISPLAY PICTURE'S SILHOUETTE REACH in display pixels (T2; the design §4.2): a display pixel
/// gathers the 3 × 3 render samples around its nearest one (step 4.1), and reads its history along the
/// motion of the nearest surface of THEIR 3 × 3 (the dilated motion, step 1.2) — render samples two
/// render pixels either way, in display pixels (rounded up).
///
/// Example: at one half the display picture of 2 568 × 1 440 over a render picture of 1 284 × 720 reads
/// four display pixels either way of a near dune's crest as its silhouette; at one, two.
#[must_use]
pub fn display_edge_reach_px(render_width: usize, display_width: usize) -> usize {
    (2 * display_width).div_ceil(render_width.max(1))
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
    /// Terrain pixels whose ground fell outside the first frame, or on no terrain there, or — under
    /// the separation ([`Frame::edge_separation_per_m`]) — behind a nearer surface there: ground the
    /// first frame did not show (a disocclusion, the reconstruction's own depth test).
    pub off_frame: u64,
    /// The widest near limit any pair applied, in metres (the finest rung's is the smallest; a
    /// spinning parent's chord raises it into the picture).
    pub near_limit_m: f64,
    /// The steps of the still pixels, and of the boundary pixels.
    pub still: StepHist,
    pub flipped: StepHist,
    /// The boundary pixels' steps per boundary, keyed by the finer rung.
    pub boundaries: BTreeMap<u8, StepHist>,
    /// ★ The steps of the pixels on an occlusion edge ([`Frame::edge_separation_per_m`]): a jittered
    /// render picture draws a silhouette as a stair that moves between two frames, so these steps
    /// are read apart from the seam law.
    pub edges: StepHist,
}

/// ★ HOW THE JUDGE READ ONE PIXEL of the second frame: the histogram its step went to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelClass {
    /// No other rung in its neighbourhood: its step belongs to the picture's noise (the floor).
    Still,
    /// On an occlusion edge ([`Frame::edge_separation_per_m`]): read apart.
    Edge,
    /// Another rung in its neighbourhood: the boundary keyed by the finer of the two rungs.
    Boundary(u8),
}

/// One compared pixel as the judge read it: its place in the second frame, its step, its class, the
/// first frame's pixel it matched best (`y · width + x` there) — the pixel its step was read against —
/// and the place its ground reprojected to in the first frame's SAMPLE GRID (the first frame's pixel
/// `(i, j)` sampled at `(i, j)`; between them, the ground lies between their samples).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelRead {
    pub x: usize,
    pub y: usize,
    pub step: u8,
    pub class: PixelClass,
    pub matched: usize,
    pub place: [f64; 2],
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
            edges: empty_hist(),
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

    /// How many pixels stood on an occlusion edge, and the widest step among them.
    #[must_use]
    pub fn edge_read(&self) -> (u64, u8) {
        (self.edges.iter().sum(), widest_of(&self.edges))
    }

    /// Counts one compared pixel's step in the histogram of its class.
    fn count(&mut self, class: PixelClass, step: u8) {
        let s = usize::from(step);
        match class {
            PixelClass::Still => self.still[s] += 1,
            PixelClass::Edge => self.edges[s] += 1,
            PixelClass::Boundary(finer) => {
                self.flipped[s] += 1;
                self.boundaries.entry(finer).or_insert_with(empty_hist)[s] += 1;
            }
        }
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
/// `rung`: the smallest widest-channel step over the [`NEIGHBOURHOOD`] (the centre included), the
/// first frame's pixel that holds it (`y · width + x`; the first in reading order among equals), and
/// the finest rung a differing neighbour was drawn at. `None` when no terrain pixel stands in the
/// neighbourhood (sky, or the frame's edge).
fn best_match(
    before: &Frame<'_>,
    after_rgba: &[u8],
    i: usize,
    qx: usize,
    qy: usize,
    rung: u8,
) -> Option<(u8, usize, Option<u8>)> {
    let mut best: Option<(u8, usize)> = None;
    let mut other_rung: Option<u8> = None;
    let mut dy = -NEIGHBOURHOOD;
    while dy <= NEIGHBOURHOOD {
        let mut dx = -NEIGHBOURHOOD;
        while dx <= NEIGHBOURHOOD {
            let (x, y) = (qx as i64 + dx, qy as i64 + dy);
            let inside =
                (x >= 0) & (y >= 0) & (x < before.width as i64) & (y < before.height as i64);
            if inside & !before.under_overlay(x as usize, y as usize) {
                let j = (y as usize * before.width + x as usize) * 4;
                let pa = decode_probe([before.probe[j], before.probe[j + 1], before.probe[j + 2]]);
                if pa.kind == PROBE_KIND_TERRAIN {
                    let step = (0..3)
                        .map(|k| after_rgba[i + k].abs_diff(before.rgba[j + k]))
                        .max()
                        .unwrap_or(0);
                    if best.is_none_or(|(b, _)| step < b) {
                        best = Some((step, j / 4));
                    }
                    if pa.rung != rung {
                        other_rung = Some(other_rung.map_or(pa.rung, |r| r.min(pa.rung)));
                    }
                }
            }
            dx += 1;
        }
        dy += 1;
    }
    best.map(|(b, at)| (b, at, other_rung))
}

/// ★ WHETHER THE PIXEL AT `(x, y)` STANDS ON A SILHOUETTE of `frame`: a neighbour of its 3 × 3 (inside
/// the frame, off the overlay) of another kind, or — both ground — whose distance differs from the
/// pixel's own by more than the frame's separation times the farther of the two.
fn silhouette(frame: &Frame<'_>, x: usize, y: usize, cell_m: &dyn Fn(u8) -> f64) -> bool {
    let i = (y * frame.width + x) * 4;
    let p = decode_probe([frame.probe[i], frame.probe[i + 1], frame.probe[i + 2]]);
    let d = cell_m(p.rung) * f64::from(p.cells);
    let mut dy = -NEIGHBOURHOOD;
    while dy <= NEIGHBOURHOOD {
        let mut dx = -NEIGHBOURHOOD;
        while dx <= NEIGHBOURHOOD {
            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
            let inside =
                (nx >= 0) & (ny >= 0) & (nx < frame.width as i64) & (ny < frame.height as i64);
            if inside & !frame.under_overlay(nx as usize, ny as usize) {
                let j = (ny as usize * frame.width + nx as usize) * 4;
                let n = decode_probe([frame.probe[j], frame.probe[j + 1], frame.probe[j + 2]]);
                let dn = cell_m(n.rung) * f64::from(n.cells);
                let ground = (n.kind == PROBE_KIND_TERRAIN) & (p.kind == PROBE_KIND_TERRAIN);
                let apart = ground & ((dn - d).abs() > frame.edge_separation_per_m * dn.max(d));
                if (n.kind != p.kind) | apart {
                    return true;
                }
            }
            dx += 1;
        }
        dy += 1;
    }
    false
}

/// ★ WHETHER THE GROUND AT `point` WAS HIDDEN IN THE FIRST FRAME (T2; the reconstruction's own depth
/// test, the design §4.2 step 2): the first frame's pixel `(qx, qy)` shows ground NEARER than the
/// point by more than the separation times the farther of the two — ground a near crest covered, which
/// no pixel of the first frame showed. Nought separation tests nothing.
///
/// Example: the hull flies at 240 m/s and a near ridge 3.7 km off sinks out of the picture's bottom
/// rows; the far ground 24.5 km off that it covered shows in the second frame, where the first frame
/// showed the ridge — hidden there, so it is not compared.
fn disoccluded(
    before: &Frame<'_>,
    point: DVec3,
    (qx, qy): (usize, usize),
    cell_m: &dyn Fn(u8) -> f64,
) -> bool {
    let j = (qy * before.width + qx) * 4;
    let pa = decode_probe([before.probe[j], before.probe[j + 1], before.probe[j + 2]]);
    let da = cell_m(pa.rung) * f64::from(pa.cells);
    let dp = (point - before.camera.eye).length();
    let sep = before.edge_separation_per_m;
    (sep > 0.0) & (pa.kind == PROBE_KIND_TERRAIN) & (dp - da > sep * dp.max(da))
}

/// Every pixel within `k` of a set pixel of `on` (a `w × h` grid), in both directions: the square
/// `2k + 1` wide around each; `k` nought copies the grid.
fn dilate(on: &[bool], w: usize, h: usize, k: usize) -> Vec<bool> {
    let mut wide = vec![false; w * h];
    for i in (0..on.len()).filter(|&i| on[i]) {
        let (x, y) = (i % w, i / w);
        for xx in x.saturating_sub(k)..(x + k + 1).min(w) {
            wide[y * w + xx] = true;
        }
    }
    let mut tall = vec![false; w * h];
    for i in (0..wide.len()).filter(|&i| wide[i]) {
        let (x, y) = (i % w, i / w);
        for yy in y.saturating_sub(k)..(y + k + 1).min(h) {
            tall[yy * w + x] = true;
        }
    }
    tall
}

/// ★ THE SILHOUETTE BAND of a frame: every pixel within [`Frame::edge_reach_px`] of a pixel whose 3 × 3
/// crosses a silhouette ([`silhouette`]); all clear without the separation (nothing is read apart).
fn silhouette_band(frame: &Frame<'_>, cell_m: &dyn Fn(u8) -> f64) -> Vec<bool> {
    let judged = usize::from(frame.edge_separation_per_m > 0.0);
    let w = frame.width;
    let on: Vec<bool> = (0..frame.width * frame.height * judged)
        .map(|i| silhouette(frame, i % w, i / w, cell_m))
        .collect();
    dilate(
        &on,
        w,
        on.len() / w.max(1),
        frame.edge_reach_px.saturating_sub(1),
    )
}

/// Whether pixel `i` lies in a band (a band left empty holds no pixel).
fn in_band(band: &[bool], i: usize) -> bool {
    band.get(i).copied().unwrap_or(false)
}

/// THE JUDGE of a pair: the second frame's terrain pixels reprojected into the first; the rung
/// boundaries' steps and the still pixels' steps as histograms. `cell_m` names a rung's cell in
/// metres (the ladder's own). A frame that is not one picture and one probe of its stated size
/// reads nothing.
#[must_use]
pub fn judge_pair(before: &Frame<'_>, after: &Frame<'_>, cell_m: &dyn Fn(u8) -> f64) -> PairRead {
    judge_pair_mapped(before, after, cell_m, &mut |_| {})
}

/// ★ THE JUDGE, PIXEL BY PIXEL: [`judge_pair`], and every compared pixel handed to `on_pixel` as it is
/// counted — one judge for the gate and for a diagnostic that maps where a leg's steps stand.
///
/// Example: a map of the 240 m/s leg's first pair paints each boundary pixel past the floor red over the
/// picture, and the red lies along the rung-2 band under the hull.
#[must_use]
pub fn judge_pair_mapped(
    before: &Frame<'_>,
    after: &Frame<'_>,
    cell_m: &dyn Fn(u8) -> f64,
    on_pixel: &mut dyn FnMut(PixelRead),
) -> PairRead {
    let mut read = PairRead::default();
    if !(before.well_formed() & after.well_formed()) {
        return read;
    }
    let (band_before, band_after) = (
        silhouette_band(before, cell_m),
        silhouette_band(after, cell_m),
    );
    let travel = (after.camera.eye - before.camera.eye).length();
    let focal = focal_px(after.height, after.camera.fov_y);
    let mut y = 0usize;
    while y < after.height {
        let mut x = 0usize;
        while x < after.width {
            let i = (y * after.width + x) * 4;
            let pb = decode_probe([after.probe[i], after.probe[i + 1], after.probe[i + 2]]);
            if (pb.kind != PROBE_KIND_TERRAIN) | after.under_overlay(x, y) {
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
            // ★ THE SAMPLE POINTS (the temporal path's T2): the second frame's pixel sampled the scene at
            // its jittered point; the first frame's pixel whose sample stands nearest the reprojected
            // place is the match's centre (`vd_client::reconstruction`, one rule for both).
            let [sx, sy] = sample_point([x as i64, y as i64], after.jitter_px);
            let point = after.camera.unproject(sx, sy, d);
            let q = before.camera.project_point(point);
            let place = q.map_or([0.0; 2], |q| {
                [
                    q.x - 0.5 - before.jitter_px[0],
                    q.y - 0.5 - before.jitter_px[1],
                ]
            });
            let hit = q
                .and_then(|q| {
                    let [qx, qy] = nearest_sample([q.x, q.y], before.jitter_px);
                    let inside = (qx >= 0)
                        & (qy >= 0)
                        & (qx < before.width as i64)
                        & (qy < before.height as i64);
                    inside.then_some((qx as usize, qy as usize))
                })
                .filter(|&q| !disoccluded(before, point, q, cell_m));
            let matched = hit.and_then(|(qx, qy)| {
                best_match(before, after.rgba, i, qx, qy, pb.rung)
                    .map(|m| (m, qy * before.width + qx))
            });
            let Some(((step, at, other), q)) = matched else {
                read.off_frame += 1;
                x += 1;
                continue;
            };
            read.compared += 1;
            let class = if in_band(&band_after, y * after.width + x) | in_band(&band_before, q) {
                PixelClass::Edge
            } else {
                other.map_or(PixelClass::Still, |o| PixelClass::Boundary(pb.rung.min(o)))
            };
            read.count(class, step);
            on_pixel(PixelRead {
                x,
                y,
                step,
                class,
                matched: at,
                place,
            });
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

/// ★ THE LEVEL RULE (the temporal path's T2; a READING since ruling B27 made the eye's rule,
/// `crate::eye`, the seam law): a boundary of a leg holds while the pixels that crossed it AND stepped
/// past the floor number at most `share` of the pixels that crossed it, plus `per_pair` for each pair
/// the leg judged. The floor already stands at
/// the still pixels' 99.9 % point, so a thousandth of any pixels steps past it by the picture's own
/// noise; the per-pair term forgives a lone speckle of the ground's own (a still pixel shows one as
/// often), never a line of them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeamRule {
    pub share: f64,
    pub per_pair: u64,
}

/// ★ THE LEVEL RULE'S NUMBERS: one hundredth of the crossed pixels plus one a pair. MEASURED on the passing
/// flights of 2026-09-30 (the walk, a turn, 240 and 528 m/s, at one and at one half, both pictures): the
/// fullest boundary used a fifth of this allowance (the walk's rung 2, 49 of 21 647 crossed, 45 of them
/// one level past the floor).
pub const SEAM_RULE: SeamRule = SeamRule {
    share: 0.01,
    per_pair: 1,
};

impl SeamRule {
    /// The allowance of a boundary that `crossed` pixels crossed over `pairs` pairs.
    #[must_use]
    pub fn allowance(&self, crossed: u64, pairs: u64) -> f64 {
        self.share * crossed as f64 + (self.per_pair * pairs) as f64
    }

    /// The boundaries of a leg's pooled reading over `pairs` pairs that break the rule, finest first.
    #[must_use]
    pub fn breaches(&self, read: &PairRead, pairs: u64) -> Vec<BoundaryRead> {
        read.boundary_reads()
            .into_iter()
            .filter(|b| b.over_floor as f64 > self.allowance(b.crossed, pairs))
            .collect()
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
        add_hist(&mut total.edges, &r.edges);
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
            hud: None,
            jitter_px: [0.0; 2],
            edge_separation_per_m: 0.0,
            edge_reach_px: 1,
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

    /// The reference view's separation: `Ksep × Kfov × the diagonal`, 0.0264 of the farther distance.
    #[test]
    fn the_edge_separation_is_the_depth_tests_own() {
        let s = edge_separation_per_m(1284, 720);
        assert!((s - 0.0264).abs() < 0.0005, "{s}");
        // Twice the grid, twice the separation per metre (the diagonal doubles; the corner factor holds).
        let s2 = edge_separation_per_m(2568, 1440);
        assert!((s2 - 2.0 * s).abs() < 1e-12, "{s2}");
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

    /// ★ THE OCCLUSION EDGE (T2): a near ridge (the left half, 200 m) in front of far ground (the right half,
    /// 2 000 m). With the depth test's separation the pixels beside the gap, and the pixels beside the sky,
    /// read apart; the rest stay still. Without it (nought, a picture without the reconstruction) nothing is
    /// judged apart. Under the overlay a neighbour is not read.
    #[test]
    fn a_silhouette_is_read_apart_from_the_seam_law() {
        let (w, h) = (8, 4);
        let (mut rgba, mut probe, cam) = painted(w, h, 2, 500, 120, still_camera(w, h));
        let near = encode_probe(ProbePixel {
            kind: PROBE_KIND_TERRAIN,
            rung: 2,
            cells: 50,
        });
        for y in 0..h {
            for x in 0..4 {
                let i = (y * w + x) * 4;
                probe[i..i + 3].copy_from_slice(&near);
                rgba[i..i + 3].copy_from_slice(&[60, 60, 60]);
            }
        }
        // The top-right pixel is sky.
        let sky = encode_probe(ProbePixel {
            kind: 0,
            rung: 0,
            cells: 0,
        });
        let top_right = (w - 1) * 4;
        probe[top_right..top_right + 3].copy_from_slice(&sky);
        let a = Frame {
            edge_separation_per_m: 0.0264,
            ..frame(&rgba, &probe, w, h, cam)
        };
        let b = Frame {
            edge_separation_per_m: 0.0264,
            ..frame(&rgba, &probe, w, h, still_camera(w, h))
        };
        let read = judge_pair(&a, &b, &cell);
        // Columns 3 and 4 hold the gap (8 pixels); (6, 0), (6, 1) and (7, 1) border the sky at (7, 0): 11
        // pixels read apart, every one of them unchanged.
        let (edges, widest) = read.edge_read();
        assert_eq!((edges, widest), (11, 0));
        assert_eq!(read.compared, (w * h - 1) as u64);
        assert_eq!(read.still.iter().sum::<u64>(), read.compared - edges);
        // Without the separation the same pair judges no edge apart.
        let a0 = frame(&rgba, &probe, w, h, still_camera(w, h));
        let b0 = frame(&rgba, &probe, w, h, still_camera(w, h));
        let read0 = judge_pair(&a0, &b0, &cell);
        assert_eq!(read0.edge_read(), (0, 0));
        // Under the overlay a neighbour is not read: an overlay over the near half hides the gap from the
        // far pixels of column 4, which read still again.
        let hud = Some([0.0 - HUD_MARGIN_PX, 0.0, 4.0 - HUD_MARGIN_PX - 0.5, 4.0]);
        let ah = Frame {
            hud,
            ..Frame {
                edge_separation_per_m: 0.0264,
                ..frame(&rgba, &probe, w, h, still_camera(w, h))
            }
        };
        let bh = Frame {
            hud,
            ..Frame {
                edge_separation_per_m: 0.0264,
                ..frame(&rgba, &probe, w, h, still_camera(w, h))
            }
        };
        let readh = judge_pair(&ah, &bh, &cell);
        assert!(readh.edge_read().0 < edges, "{:?}", readh.edge_read());
        // The pooled read keeps the edges.
        let total = sum_reads(&[read.clone(), read]);
        assert_eq!(total.edge_read(), (22, 0));
    }

    /// ★ THE DISPLAY PICTURE'S REACH: two render pixels either way, in display pixels, rounded up.
    #[test]
    fn the_display_reach_is_two_render_pixels_in_display_pixels() {
        assert_eq!(display_edge_reach_px(1284, 1284), 2);
        assert_eq!(display_edge_reach_px(1284, 2568), 4);
        assert_eq!(display_edge_reach_px(1284, 1926), 3);
        assert_eq!(display_edge_reach_px(0, 8), 16);
    }

    /// The square around each set pixel; nought copies the grid; the grid's edge holds it.
    #[test]
    fn a_band_grows_by_its_reach_and_stops_at_the_grid() {
        let (w, h) = (5, 4);
        let mut on = vec![false; w * h];
        on[w + 1] = true;
        assert_eq!(dilate(&on, w, h, 0), on);
        let grown = dilate(&on, w, h, 1);
        let set: Vec<(usize, usize)> = (0..w * h)
            .filter(|&i| grown[i])
            .map(|i| (i % w, i / w))
            .collect();
        assert_eq!(
            set,
            vec![
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2)
            ]
        );
        assert!(dilate(&on, w, h, 9).iter().all(|&b| b));
        assert!(!in_band(&[], 3));
        assert!(in_band(&grown, w));
    }

    /// ★ THE BAND IN BOTH FRAMES AND ITS REACH: a rock 200 m off at the corner (0, 0) in front of ground
    /// 2 000 m off, in the FIRST frame only (a silhouette that moved away). The corner's own ground was
    /// hidden there (off the frame); the second frame's pixels whose places in the first frame lie in its
    /// band read apart, and a reach of two widens that band.
    #[test]
    fn a_silhouette_that_moved_is_read_apart_where_it_stood_and_the_reach_widens_it() {
        let (w, h) = (6, 4);
        let (rgba, probe, _) = painted(w, h, 2, 500, 120, still_camera(w, h));
        let mut rock = probe.clone();
        rock[..3].copy_from_slice(&encode_probe(ProbePixel {
            kind: PROBE_KIND_TERRAIN,
            rung: 2,
            cells: 50,
        }));
        let judged = |reach: usize| {
            let before = Frame {
                edge_separation_per_m: 0.0264,
                edge_reach_px: reach,
                ..frame(&rgba, &rock, w, h, still_camera(w, h))
            };
            let after = Frame {
                edge_separation_per_m: 0.0264,
                edge_reach_px: reach,
                ..frame(&rgba, &probe, w, h, still_camera(w, h))
            };
            let mut edges = Vec::new();
            let read = judge_pair_mapped(&before, &after, &cell, &mut |p| {
                if p.class == PixelClass::Edge {
                    edges.push((p.x, p.y));
                }
            });
            (read, edges)
        };
        let (read, edges) = judged(1);
        assert_eq!(edges, vec![(1, 0), (0, 1), (1, 1)]);
        assert_eq!((read.edge_read(), read.off_frame), ((3, 0), 1));
        let (_, wide) = judged(2);
        assert_eq!(wide.len(), 8, "{wide:?}");
        assert!(wide.contains(&(2, 2)));
    }

    /// ★ A DISOCCLUSION IS NOT COMPARED: a rock 200 m off covers pixel (2, 1) in the first frame and is
    /// gone in the second, where the ground 2 000 m off shows. Under the separation that pixel's ground
    /// was hidden in the first frame and counts off the frame; without it the pair compares the ground
    /// with the rock.
    #[test]
    fn ground_the_first_frame_did_not_show_is_not_compared() {
        let (w, h) = (6, 4);
        let (rgba, probe, _) = painted(w, h, 2, 500, 120, still_camera(w, h));
        let mut rock = probe.clone();
        let mut dark = rgba.clone();
        let at = (w + 2) * 4;
        rock[at..at + 3].copy_from_slice(&encode_probe(ProbePixel {
            kind: PROBE_KIND_TERRAIN,
            rung: 2,
            cells: 50,
        }));
        dark[at..at + 3].copy_from_slice(&[20, 20, 20]);
        let pair = |sep: f64| {
            let before = Frame {
                edge_separation_per_m: sep,
                ..frame(&dark, &rock, w, h, still_camera(w, h))
            };
            let after = Frame {
                edge_separation_per_m: sep,
                ..frame(&rgba, &probe, w, h, still_camera(w, h))
            };
            judge_pair(&before, &after, &cell)
        };
        let hidden = pair(0.0264);
        assert_eq!((hidden.compared, hidden.off_frame), ((w * h - 1) as u64, 1));
        let plain = pair(0.0);
        assert_eq!((plain.compared, plain.off_frame), ((w * h) as u64, 0));
        // Without the separation the rock's pixel is judged, and (its 3 × 3 all ground but the rock) it
        // matches the ground beside it: nothing past a still frame's floor.
        assert_eq!(plain.widest(), 0);
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

    /// ★ THE MAPPED JUDGE: every compared pixel reaches the map once, in the class whose histogram
    /// counted it — the boundary pair of the test above at (3, 0) and (4, 0), the 30-level step at (4, 0).
    #[test]
    fn the_mapped_judge_hands_every_compared_pixel_to_its_map() {
        let (w, h) = (6, 4);
        let (rgba_a, probe_a, cam_a) = painted(w, h, 2, 500, 120, still_camera(w, h));
        let (mut rgba_b, mut probe_b, cam_b) = painted(w, h, 2, 500, 120, still_camera(w, h));
        for x in [3usize, 4] {
            let p = encode_probe(ProbePixel {
                kind: PROBE_KIND_TERRAIN,
                rung: 3,
                cells: 250,
            });
            probe_b[x * 4..x * 4 + 3].copy_from_slice(&p);
        }
        rgba_b[4 * 4] = 90;
        let a = frame(&rgba_a, &probe_a, w, h, cam_a);
        let b = frame(&rgba_b, &probe_b, w, h, cam_b);
        let mut map = Vec::new();
        let read = judge_pair_mapped(&a, &b, &cell, &mut |p| map.push(p));
        assert_eq!(read, judge_pair(&a, &b, &cell));
        assert_eq!(map.len() as u64, read.compared);
        let boundary: Vec<PixelRead> = map
            .iter()
            .copied()
            .filter(|p| p.class != PixelClass::Still)
            .collect();
        let bare: Vec<(usize, usize, u8, PixelClass, usize)> = boundary
            .iter()
            .map(|p| (p.x, p.y, p.step, p.class, p.matched))
            .collect();
        assert_eq!(
            bare,
            vec![
                (3, 0, 0, PixelClass::Boundary(2), 2),
                (4, 0, 30, PixelClass::Boundary(2), 3),
            ]
        );
        // A still eye: the ground reprojects onto its own sample, pixel (x, 0) at (x, 0) of the grid.
        for p in &boundary {
            assert!((p.place[0] - p.x as f64).abs() < 1e-9);
            assert!((p.place[1] - p.y as f64).abs() < 1e-9);
        }
        // An occlusion edge reaches the map as an edge: a rock 200 m off at the corner (0, 0) in front of
        // ground 2 000 m off; the rock and its three neighbours stand on the silhouette.
        let mut rock = probe_a.clone();
        rock[..3].copy_from_slice(&encode_probe(ProbePixel {
            kind: PROBE_KIND_TERRAIN,
            rung: 2,
            cells: 50,
        }));
        let (sa, sb) = (
            Frame {
                edge_separation_per_m: 0.0264,
                ..frame(&rgba_a, &rock, w, h, still_camera(w, h))
            },
            Frame {
                edge_separation_per_m: 0.0264,
                ..frame(&rgba_a, &rock, w, h, still_camera(w, h))
            },
        );
        let mut edges = Vec::new();
        let read = judge_pair_mapped(&sa, &sb, &cell, &mut |p| {
            if p.class == PixelClass::Edge {
                edges.push((p.x, p.y));
            }
        });
        assert_eq!(edges, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        assert_eq!(read.edge_read(), (4, 0));
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
        assert_eq!(best_match(&a, &rgba_b, 0, 0, 0, 2), Some((50, 0, None)));
        // The slid bright column: the best match is the first frame's (2, 0), a pixel to the left.
        assert_eq!(best_match(&a, &rgba_b, 3 * 4, 3, 0, 2), Some((0, 2, None)));
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

    /// ★ THE JITTER (the temporal path's T2): a frame one pixel wide, from a still eye — unjittered, the
    /// second frame's pixel lands on the first's; with the second frame's sample at 0.8 and the first's
    /// at 0.2, the nearest sample of the first frame to 0.8 lies one pixel on, outside the frame.
    #[test]
    fn the_jitter_moves_the_sample_points_of_both_frames() {
        let (w, h) = (1, 1);
        let (rgba, probe, cam) = painted(w, h, 2, 500, 100, still_camera(w, h));
        let a = frame(&rgba, &probe, w, h, cam);
        let b = frame(&rgba, &probe, w, h, still_camera(w, h));
        assert_eq!(judge_pair(&a, &b, &cell).compared, 1);
        let mut a_jittered = frame(&rgba, &probe, w, h, cam);
        a_jittered.jitter_px = [-0.3, 0.0];
        let mut b_jittered = frame(&rgba, &probe, w, h, still_camera(w, h));
        b_jittered.jitter_px = [0.3, 0.0];
        let read = judge_pair(&a_jittered, &b_jittered, &cell);
        assert_eq!((read.compared, read.off_frame), (0, 1));
    }

    /// THE OVERLAY IS NOT THE GROUND: pixels under the second frame's rectangle are not judged,
    /// and pixels under the first frame's rectangle are no match — a readout that changed by 50
    /// levels between the frames counts nothing.
    #[test]
    fn the_overlay_is_neither_judged_nor_matched() {
        let (w, h) = (8, 4);
        let (rgba_a, probe, cam) = painted(w, h, 2, 500, 120, still_camera(w, h));
        let (mut rgba_b, _, _) = painted(w, h, 2, 500, 120, still_camera(w, h));
        // A readout in the top-left corner changed by 50 levels.
        rgba_b[0] = 70;
        rgba_b[4] = 70;
        let mut a = frame(&rgba_a, &probe, w, h, cam);
        let mut b = frame(&rgba_b, &probe, w, h, still_camera(w, h));
        let unmasked = judge_pair(&a, &b, &cell);
        assert_eq!(unmasked.compared, (w * h) as u64);
        assert_eq!(unmasked.floor(), 50);
        // The rectangle over those two pixels (its margin reaches their neighbours too).
        a.hud = Some([0.0, 0.0, 1.0, 0.0]);
        b.hud = Some([0.0, 0.0, 1.0, 0.0]);
        let masked = judge_pair(&a, &b, &cell);
        assert!(masked.compared < unmasked.compared);
        assert_eq!(masked.floor(), 0);
        // The first frame's rectangle alone: the second frame's changed pixels find no match
        // under it and count as off the frame, never as a step.
        b.hud = None;
        let half = judge_pair(&a, &b, &cell);
        assert_eq!(half.compared, unmasked.compared - 2);
        assert_eq!(half.off_frame, 2);
        assert_eq!(half.floor(), 0);
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

    /// ★ THE SEAM RULE: a boundary passes at its allowance and breaks one pixel past it; the allowance
    /// grows with the crossed pixels and with the pairs.
    #[test]
    fn the_seam_rule_names_the_boundaries_past_their_allowance() {
        let mut read = PairRead {
            still: hist_with(&[(0, 100_000)]),
            ..PairRead::default()
        };
        // Rung 2: 1 000 crossed, 12 past the floor (nought); rung 5: 1 000 crossed, 13 past it.
        read.boundaries.insert(2, hist_with(&[(0, 988), (30, 12)]));
        read.boundaries.insert(5, hist_with(&[(0, 987), (30, 13)]));
        assert_eq!(SEAM_RULE.allowance(1_000, 2), 12.0);
        let bad = SEAM_RULE.breaches(&read, 2);
        assert_eq!(
            bad,
            vec![BoundaryRead {
                finer: 5,
                crossed: 1_000,
                over_floor: 13,
                widest: 30
            }]
        );
        assert!(SEAM_RULE.breaches(&read, 3).is_empty());
        assert!(SEAM_RULE.breaches(&PairRead::default(), 0).is_empty());
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
