//! ★ THE MOTION CENSUS (the temporal path's slice T1 items 5 and 6; gate G-MOTION-COVERAGE) — the
//! pure half.
//!
//! The reconstruction (T2) finds each pixel of this frame in last frame's picture by the pixel's
//! MOTION. A drawn thing that writes no motion would smear there. The census finds such things: a
//! pixel where a surface stands but no surface wrote a motion.
//!
//! **The sentinel.** The renderer clears the motion target to [`MOTION_SENTINEL`] in both channels,
//! the largest finite half float. A motion is half the step of a clip place in one frame, so this
//! value is a step of 131 008 clip widths in one frame, and no drawn thing takes it. A pixel that
//! keeps the sentinel after the frame had no surface that wrote a motion.
//!
//! **The capture.** Beside each picture the capture writes the picture camera's motion target and
//! its depth after the frame ([`MotionPicture`], the file [`crate::capture::motion_rel_for`]). The
//! depth is reverse depth: nought where no surface stands. A pixel HOLDS A DEPTH when its depth is
//! above nought.
//!
//! **The reading** ([`MotionCensus`]): the pixels that hold a depth, the pixels among them that keep
//! the sentinel (the uncovered pixels; the gate wants none), the first of these, and the pixels that
//! keep the sentinel without a depth (the sky; if none keeps it, the stated sentinel is not the
//! stored clear value, and a count of nought uncovered pixels is empty).
//!
//! **Example.** The census flight stands on the hill with the prepass on. Every ground pixel holds a
//! depth and a motion of nought, and the sky keeps the sentinel. A lit body drawn by a material with
//! no prepass would hold a depth in the main pass and keep the sentinel, and the census would count
//! its pixels as uncovered.
//!
//! **THE MOTION TRUTH** ([`motion_truth`], gate G-MOTION-TRUTH on the still ground): for each pixel
//! of a frame that the probe calls terrain and does NOT mark as moving against its body
//! ([`crate::probe::probe_moving`]), the REFERENCE names the pixel's previous place by the pop
//! detector's reprojection — its probe distance through this frame's camera to a point of the body,
//! and that point through last frame's camera. The pixel's MOTION names its own previous place. The
//! residual is the distance between the two, in render pixels. A stated distance is half a cell
//! coarse, and that miss moves the reference by about `focal · travel · cell / 2 / d²` pixels, so a
//! pixel is judged only where this stays under [`TRUTH_REFERENCE_MISS_PX`]. The moving ground (a
//! band's morph or sink, a rising chunk) is judged by the Tier-A rule (`vd_client::motion`), because
//! there the reference's "still ground" is not the truth. The still pixels of chunks NEW in the frame,
//! landing or rebuilt ([`crate::probe::probe_new`]), are judged apart as well: they are the design's
//! new-chunk rows, where the engine's own rule gives a new entity no previous place, and a defect there
//! alone would hide under a thousandth of the whole picture.
//!
//! Example: the hull flies at 240 m/s over the belt. A pixel of rung 4 at 9 km holds still against
//! the planet; its probe distance, through the two stamped cameras, puts its previous place 3.2
//! pixels left, and its motion names a previous place within 0.01 pixel of that.

use crate::camera::CaptureCamera;
use crate::probe::{PROBE_KIND_TERRAIN, decode_probe, probe_moving, probe_new};

/// ★ THE SENTINEL: the largest finite half float (65 504), in both channels of the motion target.
/// The renderer clears to it and the census reads it back, one value named once.
pub const MOTION_SENTINEL: f32 = 65_504.0;

/// The sentinel as the target stores it: the half float's bits.
pub const MOTION_SENTINEL_BITS: u16 = 0x7BFF;

/// The file's first eight bytes.
pub const MOTION_MAGIC: [u8; 8] = *b"VDMOTN01";

/// The bytes a pixel takes in each part of the capture: two half floats of motion, one 32-bit float
/// of depth.
pub const MOTION_TEXEL_BYTES: usize = 4;

/// The header: the magic word, the width and the height.
const HEADER_BYTES: usize = 16;

/// THE REFERENCE'S OWN MISS the truth judges under, in render pixels (the design's T1: the pop
/// detector's near limit, taken at 0.02 pixel in place of its 0.25).
pub const TRUTH_REFERENCE_MISS_PX: f64 = 0.02;

/// THE PROPOSED TARGET (the design's T1): the residual under which [`TRUTH_SHARE`] of the judged
/// pixels must stand, in render pixels.
pub const TRUTH_RESIDUAL_PX: f64 = 0.05;
/// The share of the judged pixels whose residual stands under [`TRUTH_RESIDUAL_PX`].
pub const TRUTH_SHARE: f64 = 0.999;

/// The residual histogram's bin, in render pixels, and its bins; the last bin holds every residual
/// of a pixel or more.
pub const TRUTH_BIN_PX: f64 = 0.001;
pub const TRUTH_BINS: usize = 1001;

/// ★ ONE CAPTURED MOTION PICTURE: the motion target (the two half floats of each pixel, as bits) and
/// the depth, row by row from the top.
#[derive(Clone, Debug, PartialEq)]
pub struct MotionPicture {
    pub width: u32,
    pub height: u32,
    pub motion: Vec<[u16; 2]>,
    pub depth: Vec<f32>,
}

/// Why a motion file does not read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionReadError {
    /// Fewer bytes than the header, or than the header states.
    Short,
    /// The first eight bytes are not [`MOTION_MAGIC`].
    Magic,
}

/// ★ THE CENSUS'S READING of one motion picture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionCensus {
    /// The pixels that hold a depth: a surface stands there.
    pub held: u64,
    /// The pixels that hold a depth and keep the sentinel: a surface that wrote no motion.
    pub uncovered: u64,
    /// The first uncovered pixel (column, row), from the top row.
    pub first_uncovered: Option<(u32, u32)>,
    /// The pixels that keep the sentinel and hold no depth (the sky).
    pub sentinel_sky: u64,
}

impl MotionPicture {
    /// The picture from the copier's raw buffer: the motion rows, then the depth rows, each row
    /// padded to `padded_row` bytes. `None` when a row cannot hold the width or the buffer cannot
    /// hold both parts.
    #[must_use]
    pub fn from_padded(width: u32, height: u32, padded_row: usize, raw: &[u8]) -> Option<Self> {
        Self::from_padded_window(width, height, height, padded_row, raw)
    }

    /// ★ THE RENDER GRID'S WINDOW (the temporal path's T2; the design §12.1): the top-left `width` ×
    /// `height` pixels of a copy `copied_rows` tall. Under an up-scale the picture camera's targets
    /// hold the DISPLAY size and the drawn picture fills their top-left render rectangle, so the
    /// capture copies the whole targets and keeps the render grid. `None` when a row cannot hold the
    /// width, the window is taller than the copy, or the buffer cannot hold both parts.
    #[must_use]
    pub fn from_padded_window(
        width: u32,
        height: u32,
        copied_rows: u32,
        padded_row: usize,
        raw: &[u8],
    ) -> Option<Self> {
        let w = width as usize;
        let h = height as usize;
        let rows = copied_rows as usize;
        let fits = (w * MOTION_TEXEL_BYTES <= padded_row)
            & (h <= rows)
            & (2 * padded_row * rows <= raw.len());
        if !fits {
            return None;
        }
        let texel = |part: usize, x: usize, y: usize| -> [u8; 4] {
            let at = part * padded_row * rows + y * padded_row + x * MOTION_TEXEL_BYTES;
            [raw[at], raw[at + 1], raw[at + 2], raw[at + 3]]
        };
        let pixels = || (0..h).flat_map(move |y| (0..w).map(move |x| (x, y)));
        let motion = pixels()
            .map(|(x, y)| {
                let t = texel(0, x, y);
                [
                    u16::from_le_bytes([t[0], t[1]]),
                    u16::from_le_bytes([t[2], t[3]]),
                ]
            })
            .collect();
        let depth = pixels()
            .map(|(x, y)| f32::from_le_bytes(texel(1, x, y)))
            .collect();
        Some(MotionPicture {
            width,
            height,
            motion,
            depth,
        })
    }

    /// The file's bytes: the header, the motion, then the depth, little-endian.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + 2 * MOTION_TEXEL_BYTES * self.depth.len());
        out.extend_from_slice(&MOTION_MAGIC);
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        for [x, y] in &self.motion {
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
        }
        for d in &self.depth {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out
    }

    /// The picture from a file's bytes.
    ///
    /// # Errors
    /// [`MotionReadError::Magic`] when the magic word is not the file's; [`MotionReadError::Short`]
    /// when the bytes end before the header or before the pixels it states.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MotionReadError> {
        let header = bytes.get(..HEADER_BYTES).ok_or(MotionReadError::Short)?;
        if header[..8] != MOTION_MAGIC {
            return Err(MotionReadError::Magic);
        }
        let word = |at: usize| {
            u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
        };
        let (width, height) = (word(8), word(12));
        let pixels = width as usize * height as usize;
        let row = MOTION_TEXEL_BYTES * width as usize;
        let body = bytes
            .get(HEADER_BYTES..HEADER_BYTES + 2 * MOTION_TEXEL_BYTES * pixels)
            .ok_or(MotionReadError::Short)?;
        // The body is the two parts unpadded: a row is its width's texels exactly.
        Self::from_padded(width, height, row, body).ok_or(MotionReadError::Short)
    }

    /// ★ THE CENSUS: the pixels that hold a depth, those that also keep the sentinel, and the sky's
    /// sentinel pixels.
    #[must_use]
    pub fn census(&self) -> MotionCensus {
        let mut read = MotionCensus::default();
        let width = self.width.max(1);
        for (i, (motion, depth)) in self.motion.iter().zip(&self.depth).enumerate() {
            let held = *depth > 0.0;
            let kept = *motion == [MOTION_SENTINEL_BITS; 2];
            read.held += u64::from(held);
            read.sentinel_sky += u64::from(kept & !held);
            let uncovered = held & kept;
            read.uncovered += u64::from(uncovered);
            let index = u32::try_from(i).unwrap_or(u32::MAX);
            let here = Some((index % width, index / width));
            read.first_uncovered = read.first_uncovered.or(here.filter(|_| uncovered));
        }
        read
    }
}

/// One frame for the truth: its probe (RGBA, eight bits), its motion picture (the same size) and the
/// camera it was drawn with, in the body's frame.
pub struct TruthFrame<'a> {
    pub probe: &'a [u8],
    pub motion: &'a MotionPicture,
    pub camera: CaptureCamera,
    /// THE FRAME'S JITTER (the temporal path's T2; the stamp's reconstruction row): render pixel `i`
    /// sampled the scene at `i + 0.5 + jitter`, so the reference unprojects that point. Nought without
    /// the reconstruction.
    pub jitter_px: [f64; 2],
}

/// ★ A SET OF RESIDUALS: the pixels judged, their residuals in bins of [`TRUTH_BIN_PX`] (the last bin
/// holds a pixel or more), and the largest, in render pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct Residuals {
    pub judged: u64,
    pub hist: Box<[u64; TRUTH_BINS]>,
    pub worst_px: f64,
}

impl Default for Residuals {
    fn default() -> Self {
        Residuals {
            judged: 0,
            hist: Box::new([0; TRUTH_BINS]),
            worst_px: 0.0,
        }
    }
}

impl Residuals {
    fn add(&mut self, residual: f64) {
        self.judged += 1;
        self.worst_px = self.worst_px.max(residual);
        let bin = ((residual / TRUTH_BIN_PX) as usize).min(TRUTH_BINS - 1);
        self.hist[bin] += 1;
    }

    /// The residual under which `share` of the judged pixels stand: the upper edge of the bin where
    /// the count reaches that share (nought when nothing was judged).
    #[must_use]
    pub fn at(&self, share: f64) -> f64 {
        let wanted = (share * self.judged as f64).ceil() as u64;
        let mut seen = 0u64;
        let mut bin = 0usize;
        while (bin < TRUTH_BINS) & (seen < wanted) {
            seen += self.hist[bin];
            bin += 1;
        }
        bin as f64 * TRUTH_BIN_PX
    }

    /// Whether the set meets the proposed target: [`TRUTH_SHARE`] of its pixels under
    /// [`TRUTH_RESIDUAL_PX`]. An empty set meets it.
    #[must_use]
    pub fn meets_target(&self) -> bool {
        self.at(TRUTH_SHARE) <= TRUTH_RESIDUAL_PX + TRUTH_BIN_PX * 0.5
    }

    /// The sum of two sets: the counts add, the histograms pool, the worst of both.
    #[must_use]
    pub fn plus(&self, other: &Residuals) -> Residuals {
        let mut hist = self.hist.clone();
        for (into, from) in hist.iter_mut().zip(other.hist.iter()) {
            *into += from;
        }
        Residuals {
            judged: self.judged + other.judged,
            hist,
            worst_px: self.worst_px.max(other.worst_px),
        }
    }
}

/// ★ THE TRUTH'S READING of one pair.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TruthRead {
    /// Every still ground pixel judged.
    pub still: Residuals,
    /// Of those, the pixels of chunks NEW in this frame, landing or rebuilt (the design's new-chunk
    /// rows: the engine's own rule gives a new entity no previous place).
    pub new_rows: Residuals,
    /// The ground pixels the probe marks as moving (left to the Tier-A rule).
    pub moving: u64,
    /// Of the moving pixels, those of chunks new in this frame.
    pub new_moving: u64,
    /// The ground pixels nearer than the reference's limit.
    pub near: u64,
    /// The ground pixels whose reference falls behind last frame's eye.
    pub behind: u64,
    /// The ground pixels that kept the motion sentinel (the census's count, met here too).
    pub uncovered: u64,
    /// The widest distance the limit kept out, in metres.
    pub limit_m: f64,
}

impl TruthRead {
    /// Whether the pair meets the proposed target on every still pixel AND on the new chunks'
    /// still pixels apart (a defect of the new rows alone hides under a thousandth of the whole).
    #[must_use]
    pub fn meets_target(&self) -> bool {
        self.still.meets_target() & self.new_rows.meets_target()
    }

    /// The sum of two readings: the counts add, the residuals pool, the widest limit.
    #[must_use]
    pub fn plus(&self, other: &TruthRead) -> TruthRead {
        TruthRead {
            still: self.still.plus(&other.still),
            new_rows: self.new_rows.plus(&other.new_rows),
            moving: self.moving + other.moving,
            new_moving: self.new_moving + other.new_moving,
            near: self.near + other.near,
            behind: self.behind + other.behind,
            uncovered: self.uncovered + other.uncovered,
            limit_m: self.limit_m.max(other.limit_m),
        }
    }
}

/// ★ G-MOTION-TRUTH ON ONE PAIR: every still ground pixel of `now` (terrain in its probe, not marked
/// moving, farther than the reference's limit) against the reference through `last_camera`.
/// `cell_m` names a rung's cell in metres (the ladder's own). A frame whose probe or motion is not
/// its stated size reads nothing.
#[must_use]
pub fn motion_truth(
    last_camera: &CaptureCamera,
    now: &TruthFrame<'_>,
    cell_m: &dyn Fn(u8) -> f64,
) -> TruthRead {
    let mut read = TruthRead::default();
    let (w, h) = (now.camera.width, now.camera.height);
    let fits = (now.probe.len() == w * h * 4)
        & (now.motion.motion.len() == w * h)
        & (now.motion.width as usize == w);
    if !fits {
        return read;
    }
    let travel = (now.camera.eye - last_camera.eye).length();
    let focal = crate::pop::focal_px(h, now.camera.fov_y);
    for (i, px) in now.probe.chunks_exact(4).enumerate() {
        let p = decode_probe([px[0], px[1], px[2]]);
        if p.kind != PROBE_KIND_TERRAIN {
            continue;
        }
        let moving = probe_moving(px[3]);
        let new = probe_new(px[3]);
        read.moving += u64::from(moving);
        read.new_moving += u64::from(moving & new);
        let bits = now.motion.motion[i];
        let kept = bits == [MOTION_SENTINEL_BITS; 2];
        read.uncovered += u64::from(kept & !moving);
        let cell = cell_m(p.rung);
        let d = cell * f64::from(p.cells);
        let limit = (focal * travel * cell * 0.5 / TRUTH_REFERENCE_MISS_PX).sqrt();
        let near = d < limit;
        read.near += u64::from(near & !moving & !kept);
        read.limit_m = read.limit_m.max(limit);
        if moving | kept | near {
            continue;
        }
        let [x, y] = vd_client::reconstruction::sample_point(
            [(i % w) as i64, (i / w) as i64],
            now.jitter_px,
        );
        let Some(reference) = last_camera.project_point(now.camera.unproject(x, y, d)) else {
            read.behind += 1;
            continue;
        };
        let own_x = x - f64::from(half_to_f32(bits[0])) * w as f64;
        let own_y = y - f64::from(half_to_f32(bits[1])) * h as f64;
        let residual = (own_x - reference.x).hypot(own_y - reference.y);
        read.still.add(residual);
        if new {
            read.new_rows.add(residual);
        }
    }
    read
}

/// A half float's value from its bits, exactly (every half float is a 32-bit float).
#[must_use]
pub fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let fraction = f32::from(bits & 0x3FF);
    let magnitude = match exponent {
        0 => fraction * 2f32.powi(-24),
        31 => {
            if fraction == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        e => (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
    };
    sign * magnitude
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::glam::DVec3;

    /// A 3 × 2 picture: the top row ground (a depth) with a motion of nought, a sentinel, and a
    /// small motion; the bottom row sky (no depth) with the sentinel, nought, and the sentinel.
    fn picture() -> MotionPicture {
        let s = [MOTION_SENTINEL_BITS; 2];
        MotionPicture {
            width: 3,
            height: 2,
            motion: vec![[0, 0], s, [0x3C00, 0xBC00], s, [0, 0], s],
            depth: vec![0.5, 0.25, 0.125, 0.0, 0.0, 0.0],
        }
    }

    /// ★ THE SENTINEL IS THE LARGEST FINITE HALF FLOAT, and its bits read back to its value.
    #[test]
    fn the_sentinel_is_the_largest_finite_half_float() {
        assert_eq!(half_to_f32(MOTION_SENTINEL_BITS), MOTION_SENTINEL);
        assert_eq!(half_to_f32(MOTION_SENTINEL_BITS + 1), f32::INFINITY);
    }

    /// The half floats read exactly: one, minus one, the smallest subnormal, nought, a fraction,
    /// the infinities and a NaN.
    #[test]
    fn a_half_float_reads_exactly() {
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xBC00), -1.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3E00), 1.5);
        assert_eq!(half_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(half_to_f32(0x7E00).is_nan());
    }

    /// ★ THE CENSUS: a ground pixel that keeps the sentinel is uncovered, the ground pixels with a
    /// motion are covered, and the sky's sentinel pixels are counted apart.
    #[test]
    fn the_census_counts_the_uncovered_ground_and_the_sky_apart() {
        let read = picture().census();
        assert_eq!(
            read,
            MotionCensus {
                held: 3,
                uncovered: 1,
                first_uncovered: Some((1, 0)),
                sentinel_sky: 2,
            }
        );
        let mut covered = picture();
        covered.motion[1] = [0, 0];
        assert_eq!(covered.census().uncovered, 0);
        assert_eq!(covered.census().first_uncovered, None);
    }

    /// The first uncovered pixel is the first in reading order, and a later one does not replace it.
    #[test]
    fn the_first_uncovered_pixel_is_the_first_in_reading_order() {
        let mut p = picture();
        p.depth[3] = 0.75;
        let read = p.census();
        assert_eq!(read.uncovered, 2);
        assert_eq!(read.first_uncovered, Some((1, 0)));
    }

    /// ★ THE FILE: the bytes read back to the picture; a wrong magic word and a short file are
    /// refused.
    #[test]
    fn the_file_reads_back_and_refuses_a_wrong_or_short_file() {
        let p = picture();
        let bytes = p.to_bytes();
        assert_eq!(bytes.len(), HEADER_BYTES + 2 * 4 * 6);
        assert_eq!(MotionPicture::from_bytes(&bytes), Ok(p));
        let mut wrong = bytes.clone();
        wrong[0] = b'X';
        assert_eq!(
            MotionPicture::from_bytes(&wrong),
            Err(MotionReadError::Magic)
        );
        assert_eq!(
            MotionPicture::from_bytes(&bytes[..HEADER_BYTES - 1]),
            Err(MotionReadError::Short)
        );
        assert_eq!(
            MotionPicture::from_bytes(&bytes[..bytes.len() - 1]),
            Err(MotionReadError::Short)
        );
    }

    /// A half float's bits for a value, rounded to the nearest: a carry out of the fraction adds one
    /// to the exponent through the bits' sum, and nought reads nought.
    fn to_half(v: f32) -> u16 {
        let a = f64::from(v.abs());
        let e = a.log2().floor();
        let fraction = ((a / e.exp2() - 1.0) * 1024.0).round();
        (u16::from(v < 0.0) << 15) | ((((e + 15.0) as u16) << 10) + fraction as u16)
    }

    #[test]
    fn the_test_encoder_rounds_to_the_nearest_half_float() {
        assert_eq!(to_half(1.0), 0x3C00);
        assert_eq!(to_half(-1.5), 0xBE00);
        assert_eq!(to_half(0.0), 0);
        // 1.9999 rounds up to two: the fraction's carry lands in the exponent.
        assert_eq!(to_half(1.9999), 0x4000);
    }

    const W: usize = 64;
    const H: usize = 48;
    /// One cell of the fixture's rung: a millimetre, so a stated distance misses by half of one.
    const CELL_M: f64 = 0.001;

    fn camera(eye: DVec3, look: DVec3) -> CaptureCamera {
        CaptureCamera {
            eye,
            target: eye + look,
            up: DVec3::Y,
            fov_y: std::f64::consts::FRAC_PI_4,
            width: W,
            height: H,
        }
    }

    /// ★ THE FIXTURE: the eye stepped one metre to the right between the frames, over a flat wall
    /// 50 m ahead. The probe states each pixel's distance to the wall in millimetres; the motion
    /// names the previous place the reference names, as half floats, moved by `shift_px`.
    fn fixture(shift_px: f64) -> (CaptureCamera, Vec<u8>, MotionPicture) {
        fixture_at(shift_px, [0.0; 2])
    }

    /// The fixture drawn with a jitter: each pixel's probe and motion are the surface's at its sample
    /// point `i + 0.5 + jitter` (the temporal path's T2).
    fn fixture_at(shift_px: f64, jitter: [f64; 2]) -> (CaptureCamera, Vec<u8>, MotionPicture) {
        let now = camera(DVec3::ZERO, DVec3::NEG_Z);
        let last = camera(DVec3::new(-1.0, 0.0, 0.0), DVec3::NEG_Z);
        let mut probe = Vec::with_capacity(W * H * 4);
        let mut motion = Vec::with_capacity(W * H);
        for i in 0..W * H {
            let (x, y) = (
                (i % W) as f64 + 0.5 + jitter[0],
                (i / W) as f64 + 0.5 + jitter[1],
            );
            let ray = now.unproject(x, y, 1.0);
            let cells = (50.0 / -ray.z / CELL_M).round() as u16;
            let [r, g, b] = crate::probe::encode_probe(crate::probe::ProbePixel {
                kind: PROBE_KIND_TERRAIN,
                rung: 0,
                cells,
            });
            probe.extend_from_slice(&[r, g, b, crate::probe::PROBE_ALPHA_STILL]);
            let q = last
                .project_point(now.unproject(x, y, f64::from(cells) * CELL_M))
                .expect("the wall stands ahead of both eyes");
            let mx = ((x - q.x + shift_px) / W as f64) as f32;
            let my = ((y - q.y) / H as f64) as f32;
            motion.push([to_half(mx), to_half(my)]);
        }
        let picture = MotionPicture {
            width: W as u32,
            height: H as u32,
            motion,
            depth: vec![0.5; W * H],
        };
        (last, probe, picture)
    }

    /// ★ THE TRUTH: the motion that names the reference's previous place meets the target on every
    /// pixel; THE CONTROL (the design's T1 control 3): the same motion moved by 0.2 render pixel
    /// fails it.
    #[test]
    fn the_true_motion_meets_the_target_and_a_fifth_of_a_pixel_off_fails_it() {
        let (last, probe, picture) = fixture(0.0);
        let now = TruthFrame {
            probe: &probe,
            motion: &picture,
            camera: camera(DVec3::ZERO, DVec3::NEG_Z),
            jitter_px: [0.0; 2],
        };
        let read = motion_truth(&last, &now, &|_| CELL_M);
        assert_eq!(read.still.judged, (W * H) as u64);
        assert!(read.meets_target(), "{}", read.still.at(TRUTH_SHARE));
        assert!(read.still.worst_px < 0.01, "{}", read.still.worst_px);
        let (last, probe, shifted) = fixture(0.2);
        let now = TruthFrame {
            probe: &probe,
            motion: &shifted,
            camera: camera(DVec3::ZERO, DVec3::NEG_Z),
            jitter_px: [0.0; 2],
        };
        let off = motion_truth(&last, &now, &|_| CELL_M);
        assert!(!off.meets_target());
        assert!((off.still.at(TRUTH_SHARE) - 0.2).abs() < 0.01);
    }

    /// ★ A JITTERED FRAME (the temporal path's T2): the probe and the motion drawn at the sample points
    /// `i + 0.5 + jitter`, judged at the frame's jitter, meet the target as tightly as an unjittered one.
    #[test]
    fn a_jittered_frame_is_judged_at_its_sample_points() {
        let jitter = [0.375, -0.25];
        let (last, probe, picture) = fixture_at(0.0, jitter);
        let now = TruthFrame {
            probe: &probe,
            motion: &picture,
            camera: camera(DVec3::ZERO, DVec3::NEG_Z),
            jitter_px: jitter,
        };
        let read = motion_truth(&last, &now, &|_| CELL_M);
        assert_eq!(read.still.judged, (W * H) as u64);
        assert!(read.still.worst_px < 0.01, "{}", read.still.worst_px);
    }

    /// The pixels the truth leaves out, each counted apart: the moving ground (a new chunk's
    /// too), the ground that kept the sentinel, the ground nearer than the reference's limit, and the
    /// sky (not counted); a new chunk's still pixel is judged in both sets; and a last eye that
    /// looks away puts every reference behind it.
    #[test]
    fn the_truth_counts_what_it_leaves_out() {
        use crate::probe::probe_alpha;
        let (last, mut probe, mut picture) = fixture(0.0);
        probe[3] = probe_alpha(true, false);
        picture.motion[1] = [MOTION_SENTINEL_BITS; 2];
        probe[2 * 4 + 1] = 0;
        probe[2 * 4 + 2] = 100;
        probe[3 * 4] = 0;
        probe[4 * 4 + 3] = probe_alpha(false, true);
        probe[5 * 4 + 3] = probe_alpha(true, true);
        let now = TruthFrame {
            probe: &probe,
            motion: &picture,
            camera: camera(DVec3::ZERO, DVec3::NEG_Z),
            jitter_px: [0.0; 2],
        };
        let read = motion_truth(&last, &now, &|_| CELL_M);
        assert_eq!(
            (read.moving, read.new_moving, read.uncovered, read.near),
            (2, 1, 1, 1)
        );
        assert_eq!(
            (read.still.judged, read.new_rows.judged),
            ((W * H - 5) as u64, 1)
        );
        assert!(read.limit_m > 0.1);
        let away = camera(DVec3::new(-1.0, 0.0, 0.0), DVec3::Z);
        let behind = motion_truth(&away, &now, &|_| CELL_M);
        assert_eq!(behind.behind, (W * H - 5) as u64);
        assert_eq!(behind.still.judged, 0);
        assert_eq!(behind.still.at(TRUTH_SHARE), 0.0);
    }

    /// ★ THE NEW-CHUNK ROWS APART (the design's control 1 in its unit form): a new chunk's pixel a
    /// whole pixel off hides under a thousandth of the still ground, so the whole meets the target —
    /// and the new rows alone fail it, so the pair fails.
    #[test]
    fn a_new_chunk_off_by_a_pixel_fails_the_pair_though_the_whole_meets_the_target() {
        let (last, mut probe, mut picture) = fixture(0.0);
        probe[4 * 4 + 3] = crate::probe::probe_alpha(false, true);
        let [x, y] = picture.motion[4];
        picture.motion[4] = [to_half(half_to_f32(x) + 1.0 / W as f32), y];
        let now = TruthFrame {
            probe: &probe,
            motion: &picture,
            camera: camera(DVec3::ZERO, DVec3::NEG_Z),
            jitter_px: [0.0; 2],
        };
        let read = motion_truth(&last, &now, &|_| CELL_M);
        assert!(read.still.meets_target());
        assert!(!read.new_rows.meets_target());
        assert!(!read.meets_target());
        assert!(
            (read.new_rows.worst_px - 1.0).abs() < 0.01,
            "{}",
            read.new_rows.worst_px
        );
    }

    /// A frame whose probe is not its size reads nothing; two readings sum their counts and pool
    /// their residuals; a share past one reads the histogram's end.
    #[test]
    fn a_misfit_frame_reads_nothing_and_readings_sum() {
        let (last, probe, picture) = fixture(0.0);
        let now = TruthFrame {
            probe: &probe[4..],
            motion: &picture,
            camera: camera(DVec3::ZERO, DVec3::NEG_Z),
            jitter_px: [0.0; 2],
        };
        assert_eq!(motion_truth(&last, &now, &|_| CELL_M), TruthRead::default());
        let now = TruthFrame {
            probe: &probe,
            motion: &picture,
            camera: camera(DVec3::ZERO, DVec3::NEG_Z),
            jitter_px: [0.0; 2],
        };
        let one = motion_truth(&last, &now, &|_| CELL_M);
        let two = one.plus(&one);
        assert_eq!(two.still.judged, 2 * one.still.judged);
        assert_eq!(two.still.hist.iter().sum::<u64>(), 2 * one.still.judged);
        assert_eq!(two.still.worst_px, one.still.worst_px);
        assert_eq!(two.limit_m, one.limit_m);
        assert_eq!(one.still.at(1.5), TRUTH_BINS as f64 * TRUTH_BIN_PX);
    }

    /// ★ THE COPIER'S BUFFER: each row padded, the motion part then the depth part; the padding is
    /// dropped, and a row narrower than the width or a buffer shorter than both parts is refused.
    #[test]
    fn the_padded_buffer_drops_its_padding_and_refuses_what_does_not_fit() {
        let p = picture();
        let padded_row = 16;
        let mut raw = vec![0xEE_u8; 2 * padded_row * 2];
        for (i, [x, y]) in p.motion.iter().enumerate() {
            let at = (i / 3) * padded_row + (i % 3) * 4;
            raw[at..at + 2].copy_from_slice(&x.to_le_bytes());
            raw[at + 2..at + 4].copy_from_slice(&y.to_le_bytes());
        }
        for (i, d) in p.depth.iter().enumerate() {
            let at = 2 * padded_row + (i / 3) * padded_row + (i % 3) * 4;
            raw[at..at + 4].copy_from_slice(&d.to_le_bytes());
        }
        assert_eq!(
            MotionPicture::from_padded(3, 2, padded_row, &raw),
            Some(p.clone())
        );
        // ★ THE WINDOW: the top-left 2 × 1 of the 3 × 2 copy; a window taller than the copy is refused.
        assert_eq!(
            MotionPicture::from_padded_window(2, 1, 2, padded_row, &raw),
            Some(MotionPicture {
                width: 2,
                height: 1,
                motion: p.motion[..2].to_vec(),
                depth: p.depth[..2].to_vec(),
            })
        );
        assert_eq!(
            MotionPicture::from_padded_window(3, 3, 2, padded_row, &raw),
            None
        );
        assert_eq!(MotionPicture::from_padded(3, 2, 11, &raw), None);
        assert_eq!(
            MotionPicture::from_padded(3, 2, padded_row, &raw[1..]),
            None
        );
    }
}
