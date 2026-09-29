//! ★ THE SHARPNESS METER (G-SHARP; the temporal path, slice T0 item 2, §12.4) — the pure half.
//!
//! The meter states how much fine detail the ground shows in one picture. The temporal path
//! (`docs/investigation/2026-09-28/temporal_path_design.md`) can make the ground softer: its
//! reconstruction mixes earlier frames, and its up-scale draws fewer pixels than the display
//! shows. The owner judges that softness at D5 and D8, and this reading stands beside each
//! picture. T0 records today's multisampled picture as the "before" row. T2 and T4 read the
//! candidates against it.
//!
//! **The pixels that count.** The probe states what drew each pixel. An EDGE pixel has another
//! kind or another rung in its 3 × 3 probe neighbourhood (§11). The meter reads no edge pixel: not
//! as a centre, and not inside the Laplacian or the window of a centre. The reason: the probe
//! states the kind at the centre of a pixel only, but multisampling blends the whole pixel. So a
//! ground pixel beside the sky can show a part of the sky's colour. Thus a counted pixel has its
//! whole 5 × 5 ([`MEMBER_REACH`]) on ground of its own rung. The meter treats a pixel under the
//! overlay (the stamp's rectangle, grown by [`HUD_MARGIN_PX`]) as another kind, because the
//! overlay's digits are not ground.
//!
//! **The luma.** The picture holds sRGB code values (the capture's target has the engine's default
//! sRGB format). The meter weighs the code values with the BT.709 luma weights ([`LUMA_WEIGHTS`])
//! and does not linearise them, for three reasons:
//! - The owner judges with the eye, and sRGB code values are near even steps of lightness. So a
//!   detail in the shade weighs about as much as the same visible detail in the sun. Linear light
//!   weighs the sunlit detail many times more, and at the dusk stand the shade nearly drops out.
//! - The other picture gates count levels of these bytes (the pop detector's floor, G-FLICKER's
//!   one level, the exact pictures' one level). The raw energy uses the same level.
//! - The 8-bit rounding and the tone map's dither are even in levels. So their part of the raw
//!   energy (about 20 × 1/12 ≈ 1.7 levels² for independent rounding) is the same at every
//!   brightness.
//!
//! The weights are integers in ten-thousandths of a level. So the luma of a grey pixel is its level
//! exactly, and every sum is an exact integer that pools with no rounding.
//!
//! **The raw energy** is the mean squared response of the four-neighbour Laplacian ([`LAPLACIAN`])
//! over the counted pixels, in levels². The Laplacian gives 0 on flat ground and on a smooth slope
//! of shading. It responds to a change from one pixel to the next, and a blur removes that change
//! first.
//!
//! **The contrast** is the mean squared deviation of the luma from its own mean in the 3 × 3
//! window of each counted pixel ([`WINDOW`]), averaged over the counted pixels, in levels². **The
//! normalised energy** is the raw energy divided by the contrast. A darker stand multiplies each
//! deviation by one factor, so the ratio stays the same. The meter divides the two SUMS. It never
//! averages the ratio of each pixel: a nearly flat pixel with a dither of one level has the ratio
//! of pixel noise, and flat ground is frequent. Reference values: a smooth slope reads 0, stripes
//! one pixel wide read 18, pixel noise reads about 22.5, a checkerboard reads 64.8.
//!
//! **The bands.** The caller can split the reading by distance. The distance of a counted pixel is
//! its probe's cells times the cell of its rung in metres. The band edges are in metres and rise
//! strictly. A pixel at an edge goes into the farther band. With no edges, one band holds every
//! counted pixel.
//!
//! **The controls** are the tests below. White noise blurred by a 3 × 3 box keeps 1/50.6 of its raw
//! energy and 1/3.19 of its normalised energy ([`BLUR_CONTROL_RAW_DROP`],
//! [`BLUR_CONTROL_NORMALISED_DROP`], ± [`BLUR_CONTROL_TOLERANCE`]). The same noise sharpened reads
//! higher. A flat picture reads 0. The sky, the ruler, the overlay and a blended horizon row never
//! reach the reading.
//!
//! Example: the pilot stands on foot on the home planet and looks along the grass to the horizon.
//! The meter reads the grass and the rocks below the horizon. The sky, the star and the ruler ball
//! give nothing. The two ground rows next to the horizon give nothing too, because the edge of the
//! horizon is G-EDGE's reading. If the temporal path at half size makes the grass as soft as a
//! 3 × 3 box blur does, grass with detail at the size of a pixel loses about 98 % of its raw energy
//! and about two thirds of its normalised energy.

use std::ops::AddAssign;

use crate::pop::HUD_MARGIN_PX;
use crate::probe::{PROBE_KIND_TERRAIN, ProbePixel, decode_probe, probe_byte};

/// THE LUMA WEIGHTS of ITU-R BT.709 (item 3.2: E′Y = 0.2126 E′R + 0.7152 E′G + 0.0722 E′B), in
/// ten-thousandths of a level. The sRGB primaries are the BT.709 primaries, so these weights fit
/// the picture. They add up to [`LUMA_SCALE`], so the luma of a grey pixel is its level exactly.
pub const LUMA_WEIGHTS: [u32; 3] = [2_126, 7_152, 722];

/// The luma's unit: one level of a code value is this many units (the weights' denominator).
pub const LUMA_SCALE: u32 = 10_000;

/// THE LAPLACIAN: four times the centre minus its four side neighbours, the kernel of the common
/// variance-of-the-Laplacian focus measure. For white noise under a 3 × 3 box blur it drops the
/// normalised energy 3.19 times, where the eight-neighbour kernel drops it 1.77 times. So it reads
/// the softness more clearly.
pub const LAPLACIAN: [[i64; 3]; 3] = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]];

/// THE WINDOW: the side of the square in which the contrast reads the deviation of each counted
/// pixel. It is the Laplacian's own 3 × 3. A wider window reads a blur more strongly (for white
/// noise, a normalised drop of 4.43 at 5 × 5 and 4.94 at 7 × 7), but it also reads the slope of
/// the shading, and it removes more pixels at every edge. The raw energy already reads a blur
/// strongly on one stand.
pub const WINDOW: usize = LAPLACIAN.len();

/// THE MEMBER REACH: every pixel this many pixels each way from a counted pixel is ground of its
/// rung. The window reaches one pixel, and each pixel that it reads must not be an edge pixel
/// (§11's 3 × 3 rule), which reaches one pixel more.
pub const MEMBER_REACH: usize = WINDOW / 2 + 1;

/// THE BLUR CONTROL'S RAW DROP: how many times the raw energy of white noise falls under a 3 × 3
/// box blur. White noise of variance s² gives a raw energy of 20 s² (the kernel's squared weights:
/// 4² + 4 × 1²). The squared weights of the kernel convolved with the box add up to 32/81, so the
/// blurred noise gives 32/81 s², and the drop is 20 ÷ 32/81 = 405/8 = 50.625.
pub const BLUR_CONTROL_RAW_DROP: f64 = 405.0 / 8.0;

/// THE BLUR CONTROL'S NORMALISED DROP: how many times the normalised energy of white noise falls
/// under a 3 × 3 box blur. The contrast of white noise is s² less the variance of its window's
/// mean (s²/9) = 8/9 s², so the noise reads 20 ÷ 8/9 = 22.5. The contrast of the blurred noise is
/// s²/9 less the variance of its window's mean (361/6561 s²) = 368/6561 s², so it reads
/// 32/81 ÷ 368/6561 = 162/23 ≈ 7.04. The drop is 22.5 ÷ 162/23 = 115/36 ≈ 3.194.
pub const BLUR_CONTROL_NORMALISED_DROP: f64 = 115.0 / 36.0;

/// THE CONTROL'S TOLERANCE, a share of each drop. MEASURED over 1 000 seeds of the control's own
/// picture (128 × 128 pixels, levels 64 to 191): one standard deviation is 1.7 % of the raw drop
/// and 1.6 % of the normalised drop. The 8-bit rounding of the blurred picture moves the raw drop
/// by 0.3 % and the normalised drop by 0.2 %. So ±15 % holds about nine standard deviations, and
/// it still refuses the eight-neighbour kernel (a normalised drop of 1.77) and a 5 × 5 window
/// (4.43).
pub const BLUR_CONTROL_TOLERANCE: f64 = 0.15;

/// The pixels in the window.
const WINDOW_PIXELS: u128 = (WINDOW * WINDOW) as u128;

/// The window's pixel count squared, the divisor that turns a window's `n Σy² − (Σy)²` into its
/// mean squared deviation.
const WINDOW_PIXELS_SQ: f64 = (WINDOW_PIXELS * WINDOW_PIXELS) as f64;

/// One squared level in squared luma units.
const LEVEL_SQ: f64 = LUMA_SCALE as f64 * LUMA_SCALE as f64;

/// THE LUMA of one pixel's sRGB code values, in ten-thousandths of a level ([`LUMA_SCALE`]).
#[must_use]
pub const fn luma(rgb: [u8; 3]) -> u32 {
    LUMA_WEIGHTS[0] * rgb[0] as u32
        + LUMA_WEIGHTS[1] * rgb[1] as u32
        + LUMA_WEIGHTS[2] * rgb[2] as u32
}

/// One picture and its probe, the same size, as the capture reads them back: RGBA8, row-major, four
/// bytes a pixel.
#[derive(Clone, Copy, Debug)]
pub struct Picture<'a> {
    /// The picture: sRGB code values (the fourth byte is not read).
    pub rgba: &'a [u8],
    /// The probe: what drew each pixel, and how far it is ([`crate::probe`]).
    pub probe: &'a [u8],
    /// The width in pixels.
    pub width: usize,
    /// The height in pixels.
    pub height: usize,
    /// THE OVERLAY'S RECTANGLE (left, top, right, bottom, pixels), the stamp's `hud_rect_px`. The
    /// meter reads nothing under it.
    pub hud: Option<[f64; 4]>,
}

impl Picture<'_> {
    /// Whether the bytes are one picture and one probe of the stated size.
    fn well_formed(&self) -> bool {
        let n = self.width * self.height * 4;
        (n > 0) & (self.rgba.len() == n) & (self.probe.len() == n)
    }

    /// Whether a pixel lies under the overlay (its rectangle grown by [`HUD_MARGIN_PX`]), as the
    /// pop detector reads the same rectangle.
    fn under_overlay(&self, x: usize, y: usize) -> bool {
        self.hud.is_some_and(|r| {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            (px >= r[0] - HUD_MARGIN_PX)
                & (py >= r[1] - HUD_MARGIN_PX)
                & (px <= r[2] + HUD_MARGIN_PX)
                & (py <= r[3] + HUD_MARGIN_PX)
        })
    }

    /// The probe pixel at `(x, y)`.
    fn probe_at(&self, x: usize, y: usize) -> ProbePixel {
        let i = (y * self.width + x) * 4;
        decode_probe([self.probe[i], self.probe[i + 1], self.probe[i + 2]])
    }

    /// The pixel's member code: the probe's byte for its kind and rung when it is ground outside
    /// the overlay, else 0. No ground byte is 0, so two pixels share a non-zero code only when both
    /// are ground of one rung.
    fn member(&self, x: usize, y: usize) -> u8 {
        let p = self.probe_at(x, y);
        let ground = (p.kind == PROBE_KIND_TERRAIN) & !self.under_overlay(x, y);
        u8::from(ground) * probe_byte(p.kind, p.rung)
    }

    /// The luma at `(x, y)`.
    fn luma_at(&self, x: usize, y: usize) -> u32 {
        let i = (y * self.width + x) * 4;
        luma([self.rgba[i], self.rgba[i + 1], self.rgba[i + 2]])
    }

    /// THE SUMS OF ONE PIXEL: its squared Laplacian response and its window's squared deviation,
    /// or `None` when a pixel of its 5 × 5 is not ground of its rung. The caller keeps the whole
    /// 5 × 5 inside the frame.
    fn pixel_sums(&self, x: usize, y: usize) -> Option<SharpSums> {
        let code = self.member(x, y);
        let mut clean = code != 0;
        for qy in y - MEMBER_REACH..=y + MEMBER_REACH {
            for qx in x - MEMBER_REACH..=x + MEMBER_REACH {
                clean &= self.member(qx, qy) == code;
            }
        }
        let (mut laplacian, mut sum, mut sum_sq) = (0i64, 0u128, 0u128);
        for (dy, row) in LAPLACIAN.iter().enumerate() {
            for (dx, weight) in row.iter().enumerate() {
                let l = self.luma_at(x + dx - WINDOW / 2, y + dy - WINDOW / 2);
                laplacian += weight * i64::from(l);
                sum += u128::from(l);
                sum_sq += u128::from(l) * u128::from(l);
            }
        }
        clean.then_some(SharpSums {
            pixels: 1,
            laplacian_sq: u128::from(laplacian.unsigned_abs()).pow(2),
            deviation_sq: WINDOW_PIXELS * sum_sq - sum * sum,
        })
    }
}

/// The sums of a reading, in exact integers: they add over pixels, bands and pictures, and every
/// number derives from them once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharpSums {
    /// The counted ground pixels.
    pub pixels: u64,
    /// The squared Laplacian responses, in squared luma units.
    pub laplacian_sq: u128,
    /// The windows' squared deviations: for each counted pixel `n Σy² − (Σy)²` over its window
    /// (`n` pixels), which is `n²` times its mean squared deviation, in squared luma units.
    pub deviation_sq: u128,
}

impl SharpSums {
    /// THE RAW ENERGY: the mean squared Laplacian response, in levels² (0 with no pixel).
    #[must_use]
    pub fn raw_energy(&self) -> f64 {
        self.laplacian_sq as f64 / (self.pixels.max(1) as f64 * LEVEL_SQ)
    }

    /// THE CONTRAST: the mean squared deviation in the window, in levels² (0 with no pixel).
    #[must_use]
    pub fn contrast(&self) -> f64 {
        self.deviation_sq as f64 / (self.pixels.max(1) as f64 * WINDOW_PIXELS_SQ * LEVEL_SQ)
    }

    /// THE NORMALISED ENERGY: the raw energy divided by the contrast, a pure number. The window
    /// holds every tap of the Laplacian, so a window with no deviation gives no response either.
    /// Thus a picture with no contrast reads 0.
    #[must_use]
    pub fn normalised(&self) -> f64 {
        self.laplacian_sq as f64 * WINDOW_PIXELS_SQ / self.deviation_sq.max(1) as f64
    }
}

impl AddAssign for SharpSums {
    fn add_assign(&mut self, other: SharpSums) {
        self.pixels += other.pixels;
        self.laplacian_sq += other.laplacian_sq;
        self.deviation_sq += other.deviation_sq;
    }
}

/// What a picture read: one sum per distance band, the nearest first. Band `i` holds the pixels
/// from edge `i - 1` (the eye for band 0) to edge `i` (no end for the last band).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharpRead {
    /// The sums of each band: one more band than the caller gave edges.
    pub bands: Vec<SharpSums>,
}

impl SharpRead {
    /// The sums of every band together: the whole ground of the picture.
    #[must_use]
    pub fn total(&self) -> SharpSums {
        let mut total = SharpSums::default();
        for band in &self.bands {
            total += *band;
        }
        total
    }
}

/// Whether band edges are numbers that rise strictly.
fn rising(edges_m: &[f64]) -> bool {
    edges_m.iter().all(|e| e.is_finite()) & edges_m.windows(2).all(|pair| pair[0] < pair[1])
}

/// THE METER: the sums of every counted ground pixel of a picture, one per distance band.
/// `band_edges_m` cuts the bands (empty for one band); `cell_m` gives a rung's cell in metres (the
/// ladder's own). A picture that is not one picture and one probe of its stated size, or edges
/// that do not rise strictly, read nothing.
#[must_use]
pub fn read_sharpness(
    picture: &Picture<'_>,
    band_edges_m: &[f64],
    cell_m: &dyn Fn(u8) -> f64,
) -> SharpRead {
    let mut read = SharpRead {
        bands: vec![SharpSums::default(); band_edges_m.len() + 1],
    };
    if !(picture.well_formed() & rising(band_edges_m)) {
        return read;
    }
    for y in MEMBER_REACH..picture.height.saturating_sub(MEMBER_REACH) {
        for x in MEMBER_REACH..picture.width.saturating_sub(MEMBER_REACH) {
            if let Some(sums) = picture.pixel_sums(x, y) {
                let p = picture.probe_at(x, y);
                let distance_m = cell_m(p.rung) * f64::from(p.cells);
                let band = band_edges_m.partition_point(|edge| *edge <= distance_m);
                read.bands[band] += sums;
            }
        }
    }
    read
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{PROBE_KIND_NONE, PROBE_KIND_RULER, encode_probe};

    /// A rung's cell in metres: the ladder doubles it at each rung.
    fn cell(rung: u8) -> f64 {
        f64::from(1u32 << rung)
    }

    /// The four bytes of one probe pixel.
    fn probe_px(kind: u8, rung: u8, cells: u16) -> [u8; 4] {
        let p = encode_probe(ProbePixel { kind, rung, cells });
        [p[0], p[1], p[2], 255]
    }

    /// A grey pixel of one level.
    fn grey(level: u8) -> [u8; 4] {
        [level, level, level, 255]
    }

    /// A picture or a probe of `w × h` pixels, each pixel's bytes from its position.
    fn paint(w: usize, h: usize, px: &dyn Fn(usize, usize) -> [u8; 4]) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                out.extend_from_slice(&px(x, y));
            }
        }
        out
    }

    /// A probe of ground at one rung and one distance everywhere.
    fn ground(w: usize, h: usize, rung: u8, cells: u16) -> Vec<u8> {
        paint(w, h, &|_, _| probe_px(PROBE_KIND_TERRAIN, rung, cells))
    }

    fn picture<'a>(rgba: &'a [u8], probe: &'a [u8], w: usize, h: usize) -> Picture<'a> {
        Picture {
            rgba,
            probe,
            width: w,
            height: h,
            hud: None,
        }
    }

    /// The whole reading of a picture, in one band.
    fn whole(p: &Picture<'_>) -> SharpSums {
        read_sharpness(p, &[], &cell).total()
    }

    /// A reading of nothing, with `bands` bands.
    fn nothing(bands: usize) -> SharpRead {
        SharpRead {
            bands: vec![SharpSums::default(); bands],
        }
    }

    /// WHITE NOISE: the `i`-th output of splitmix64 from `seed`, as a level from 64 to 191.
    fn noise(seed: u64, i: usize) -> u8 {
        let mut z = seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        64 + (z >> 57) as u8
    }

    /// THE CONTROL'S PICTURES of `w × h` pixels from one noise on a `(w + 2) × (h + 2)` source, so
    /// that each pixel reads a whole 3 × 3 of the source: the sharp picture (the source's inside),
    /// the blurred picture (the rounded mean of the source's 3 × 3), and the sharpened picture (an
    /// unsharp mask at half strength, `x + (x − mean) / 2`, rounded). The levels 64 to 191 keep the
    /// sharpened picture between 8 and 247, so nothing clips.
    fn control_pictures(seed: u64, w: usize, h: usize) -> [Vec<u8>; 3] {
        let sw = w + 2;
        let source: Vec<u32> = (0..sw * (h + 2))
            .map(|i| u32::from(noise(seed, i)))
            .collect();
        let at = |x: usize, y: usize| source[y * sw + x];
        let box_sum = |x: usize, y: usize| {
            let mut s = 0;
            for dy in 0..3 {
                for dx in 0..3 {
                    s += at(x + dx, y + dy);
                }
            }
            s
        };
        let sharp = paint(w, h, &|x, y| grey(at(x + 1, y + 1) as u8));
        let blurred = paint(w, h, &|x, y| grey(((box_sum(x, y) + 4) / 9) as u8));
        let sharpened = paint(w, h, &|x, y| {
            grey(((27 * at(x + 1, y + 1) - box_sum(x, y) + 9) / 18) as u8)
        });
        [sharp, blurred, sharpened]
    }

    #[test]
    fn the_luma_weighs_bt_709_and_reads_a_grey_pixel_as_its_level() {
        assert_eq!(luma([255, 0, 0]), 2_126 * 255);
        assert_eq!(luma([0, 255, 0]), 7_152 * 255);
        assert_eq!(luma([0, 0, 255]), 722 * 255);
        assert_eq!(luma([77, 77, 77]), 77 * LUMA_SCALE);
        assert_eq!(LUMA_WEIGHTS.iter().sum::<u32>(), LUMA_SCALE);
        assert_eq!((WINDOW, MEMBER_REACH), (3, 2));
    }

    /// Stripes, a checkerboard and a slope read their stated values exactly; stripes in one
    /// channel read that channel's weight, and the fourth byte is not read.
    #[test]
    fn the_reference_patterns_read_their_stated_values() {
        let (w, h) = (8, 6);
        let probe = ground(w, h, 1, 40);
        let counted = ((w - 2 * MEMBER_REACH) * (h - 2 * MEMBER_REACH)) as u64;
        // Stripes one pixel wide, 10 levels apart: each response is 20 levels, each window's
        // deviation 200/9 levels², the ratio 18.
        let stripes = paint(w, h, &|x, _| grey(100 + 10 * (x % 2) as u8));
        let s = whole(&picture(&stripes, &probe, w, h));
        assert_eq!(s.pixels, counted);
        assert_eq!(
            (s.raw_energy(), s.contrast(), s.normalised()),
            (400.0, 200.0 / 9.0, 18.0)
        );
        // A checkerboard: each response is 40 levels, each deviation 2 000/81 levels², the
        // ratio 64.8.
        let checker = paint(w, h, &|x, y| grey(100 + 10 * ((x + y) % 2) as u8));
        let c = whole(&picture(&checker, &probe, w, h));
        assert_eq!(
            (c.raw_energy(), c.contrast(), c.normalised()),
            (1_600.0, 2_000.0 / 81.0, 64.8)
        );
        // A smooth slope of 5 levels a pixel: no response, a deviation of 50/3 levels², the
        // ratio 0.
        let slope = paint(w, h, &|x, _| grey(50 + 5 * x as u8));
        let r = whole(&picture(&slope, &probe, w, h));
        assert_eq!(
            (r.raw_energy(), r.contrast(), r.normalised()),
            (0.0, 50.0 / 3.0, 0.0)
        );
        // The stripes in the green channel alone, with another fourth byte: the luma steps 7.152
        // levels, so the raw energy is 4 × 7.152² levels², and the ratio stays 18.
        let green = paint(w, h, &|x, _| [0, 100 + 10 * (x % 2) as u8, 0, 17]);
        let g = whole(&picture(&green, &probe, w, h));
        assert_eq!((g.raw_energy(), g.normalised()), (204.604_416, 18.0));
    }

    /// ★ THE CONTROL: white noise under a known 3 × 3 box blur. Both energies fall, each by a
    /// factor inside the stated range. A picture that did not soften (the same noise from another
    /// seed) reads a factor near 1, which the range refuses: the meter reads the softness, and
    /// only the softness.
    #[test]
    fn the_blur_control_drops_both_energies_inside_the_stated_range() {
        let (w, h) = (128, 128);
        let probe = ground(w, h, 2, 500);
        let [sharp, blurred, _] = control_pictures(0, w, h);
        let before = whole(&picture(&sharp, &probe, w, h));
        let after = whole(&picture(&blurred, &probe, w, h));
        let counted = ((w - 2 * MEMBER_REACH) * (h - 2 * MEMBER_REACH)) as u64;
        assert_eq!((before.pixels, after.pixels), (counted, counted));
        // White noise reads 20 s² over 8/9 s² = 22.5.
        let noise_reads = 20.0 / (8.0 / 9.0);
        assert!((before.normalised() / noise_reads - 1.0).abs() <= BLUR_CONTROL_TOLERANCE);
        let raw_drop = before.raw_energy() / after.raw_energy();
        let normalised_drop = before.normalised() / after.normalised();
        assert!((raw_drop / BLUR_CONTROL_RAW_DROP - 1.0).abs() <= BLUR_CONTROL_TOLERANCE);
        assert!(
            (normalised_drop / BLUR_CONTROL_NORMALISED_DROP - 1.0).abs() <= BLUR_CONTROL_TOLERANCE
        );
        // The negative case: another sharp noise is no blur, and the range refuses it.
        let [other, _, _] = control_pictures(1, w, h);
        let other = whole(&picture(&other, &probe, w, h));
        let raw_same = before.raw_energy() / other.raw_energy();
        let normalised_same = before.normalised() / other.normalised();
        assert!((raw_same / BLUR_CONTROL_RAW_DROP - 1.0).abs() > BLUR_CONTROL_TOLERANCE);
        assert!(
            (normalised_same / BLUR_CONTROL_NORMALISED_DROP - 1.0).abs() > BLUR_CONTROL_TOLERANCE
        );
    }

    /// THE CONTROL, the other way: the same noise sharpened reads higher on both energies (for
    /// white noise at half strength, 2.29 times the raw energy and 1.044 times the normalised).
    #[test]
    fn the_sharpened_control_reads_higher_on_both_energies() {
        let (w, h) = (128, 128);
        let probe = ground(w, h, 2, 500);
        let [sharp, _, sharpened] = control_pictures(0, w, h);
        let before = whole(&picture(&sharp, &probe, w, h));
        let after = whole(&picture(&sharpened, &probe, w, h));
        assert!(after.raw_energy() > before.raw_energy());
        assert!(after.normalised() > before.normalised());
    }

    /// THE CONTROL: a flat picture reads 0 on every number (never a division by zero), and its
    /// pixels still count.
    #[test]
    fn a_flat_picture_reads_zero_on_every_number() {
        let (w, h) = (9, 7);
        let flat = paint(w, h, &|_, _| grey(128));
        let probe = ground(w, h, 0, 20);
        let sums = whole(&picture(&flat, &probe, w, h));
        assert_eq!(sums.pixels, 15);
        assert_eq!(
            (sums.raw_energy(), sums.contrast(), sums.normalised()),
            (0.0, 0.0, 0.0)
        );
        // No pixel at all reads 0 too.
        let none = SharpSums::default();
        assert_eq!(
            (none.raw_energy(), none.contrast(), none.normalised()),
            (0.0, 0.0, 0.0)
        );
    }

    /// THE CONTROL: the sky never counts. Wild noise in the sky over flat ground reads 0, and the
    /// two ground rows next to the sky are left out; a probe of sky alone counts no pixel.
    #[test]
    fn the_sky_never_counts() {
        let (w, h) = (12, 12);
        let rgba = paint(w, h, &|x, y| {
            let sky = y < 6;
            grey(u8::from(sky) * noise(3, y * w + x) + u8::from(!sky) * 90)
        });
        let probe = paint(w, h, &|_, y| {
            probe_px(PROBE_KIND_TERRAIN * u8::from(y >= 6), 0, 30)
        });
        let sums = whole(&picture(&rgba, &probe, w, h));
        // Rows 8 and 9 (the ground's rows 6 and 7 are next to the sky), columns 2 to 9.
        assert_eq!(sums.pixels, 16);
        assert_eq!(sums.raw_energy(), 0.0);
        let sky = paint(w, h, &|_, _| probe_px(PROBE_KIND_NONE, 0, 0));
        assert_eq!(whole(&picture(&rgba, &sky, w, h)), SharpSums::default());
    }

    /// The ruler is another kind: its noise never counts, and the ground within two pixels of it
    /// is left out.
    #[test]
    fn the_ruler_never_counts() {
        let (w, h) = (12, 12);
        let on_ruler = |x: usize, y: usize| (5..=6).contains(&x) & (5..=6).contains(&y);
        let rgba = paint(w, h, &|x, y| {
            let r = on_ruler(x, y);
            grey(u8::from(r) * noise(4, y * w + x) + u8::from(!r) * 90)
        });
        let probe = paint(w, h, &|x, y| {
            let kind = [PROBE_KIND_TERRAIN, PROBE_KIND_RULER][usize::from(on_ruler(x, y))];
            probe_px(kind, 0, 30)
        });
        let sums = whole(&picture(&rgba, &probe, w, h));
        // 8 × 8 centres, less the 6 × 6 whose 5 × 5 holds a ruler pixel.
        assert_eq!(sums.pixels, 64 - 36);
        assert_eq!(sums.raw_energy(), 0.0);
    }

    /// The overlay is another kind: its digits count as ground without the rectangle, and never
    /// with it; its margin reaches past the rectangle.
    #[test]
    fn the_overlay_never_counts() {
        let (w, h) = (14, 14);
        let digits = |x: usize, y: usize| (4..=6).contains(&x) & (4..=5).contains(&y);
        let rgba = paint(w, h, &|x, y| {
            let d = digits(x, y);
            grey(u8::from(d) * noise(5, y * w + x) + u8::from(!d) * 90)
        });
        let probe = ground(w, h, 0, 30);
        let mut pic = picture(&rgba, &probe, w, h);
        let open = whole(&pic);
        assert_eq!(open.pixels, 100);
        assert!(open.raw_energy() > 0.0);
        // The rectangle over the digits, grown by the margin, covers columns 2 to 8 and rows 2 to
        // 7; a centre counts when its 5 × 5 stays off it: 100 centres less 9 × 8.
        pic.hud = Some([4.0, 4.0, 7.0, 6.0]);
        let masked = whole(&pic);
        assert_eq!(masked.pixels, 100 - 72);
        assert_eq!(masked.raw_energy(), 0.0);
        assert!(pic.under_overlay(2, 7));
        assert!(pic.under_overlay(8, 2));
        assert!(!pic.under_overlay(1, 2));
        assert!(!pic.under_overlay(2, 1));
        assert!(!pic.under_overlay(9, 2));
        assert!(!pic.under_overlay(2, 8));
    }

    /// THE EDGE RULE READS TWO PIXELS: the probe calls a pixel ground by its centre, but the
    /// multisampled picture blends the whole pixel with the sky. The pixel below it has only
    /// ground in its 3 × 3 probe neighbourhood, but its Laplacian would read the blend (50 levels);
    /// the meter reads no pixel within two of the sky, so flat ground reads 0.
    #[test]
    fn a_blended_horizon_row_never_reaches_the_reading() {
        let (w, h) = (9, 10);
        // The sky (200) over rows 0 to 2, the blend (150) on row 3, the ground (100) below.
        let rgba = paint(w, h, &|_, y| grey([200, 200, 200, 150, 100][y.min(4)]));
        let probe = paint(w, h, &|_, y| {
            probe_px(PROBE_KIND_TERRAIN * u8::from(y >= 3), 0, 12)
        });
        let pic = picture(&rgba, &probe, w, h);
        let sums = whole(&pic);
        // Rows 5 to 7, columns 2 to 6.
        assert_eq!(sums.pixels, 15);
        assert_eq!(sums.raw_energy(), 0.0);
        // Row 4 has only ground in its 3 × 3, but its 5 × 5 reaches the sky.
        let code = pic.member(4, 4);
        let mut ground_3x3 = code != 0;
        for y in 3..=5 {
            for x in 3..=5 {
                ground_3x3 &= pic.member(x, y) == code;
            }
        }
        assert!(ground_3x3);
        assert_eq!(pic.pixel_sums(4, 4), None);
        assert_eq!(pic.luma_at(4, 3), 150 * LUMA_SCALE);
    }

    /// Another rung is an edge: the pixels within two of the rung boundary are left out.
    #[test]
    fn another_rung_is_an_edge() {
        let (w, h) = (10, 7);
        let stripes = paint(w, h, &|x, _| grey(100 + 10 * (x % 2) as u8));
        let probe = paint(w, h, &|x, _| {
            probe_px(PROBE_KIND_TERRAIN, 2 + u8::from(x >= 5), 300)
        });
        let sums = whole(&picture(&stripes, &probe, w, h));
        // Columns 2 and 7, rows 2 to 4.
        assert_eq!(sums.pixels, 6);
        assert_eq!(sums.normalised(), 18.0);
    }

    /// The bands split the counted pixels by distance; a pixel at an edge goes into the farther
    /// band; the total is the whole reading.
    #[test]
    fn the_bands_split_the_counted_pixels_by_distance() {
        let (w, h) = (12, 5);
        let stripes = paint(w, h, &|x, _| grey(100 + 10 * (x % 2) as u8));
        // Rung 2 (4 m cells): the columns stand at 40 m, 100 m, 400 m and 4 000 m.
        let probe = paint(w, h, &|x, _| {
            probe_px(PROBE_KIND_TERRAIN, 2, [10, 25, 100, 1_000][x / 3])
        });
        let pic = picture(&stripes, &probe, w, h);
        let read = read_sharpness(&pic, &[100.0, 1_000.0], &cell);
        let pixels: Vec<u64> = read.bands.iter().map(|b| b.pixels).collect();
        // Centres 2 to 9 of row 2: column 2 is nearer than 100 m; columns 3 to 8 stand at 100 m
        // and at 400 m; column 9 stands past 1 000 m.
        assert_eq!(pixels, vec![1, 6, 1]);
        assert_eq!(read.bands[1].normalised(), 18.0);
        assert_eq!(read.total(), whole(&pic));
        assert_eq!(read.total().pixels, 8);
    }

    /// A darker stand reads the same normalised energy: half the deviations give a quarter of
    /// both sums, and a brighter base level changes nothing. (For the power-law part of the sRGB
    /// curve a dimmer light multiplies the code values by one factor.)
    #[test]
    fn a_darker_stand_reads_the_same_normalised_energy() {
        let (w, h) = (16, 12);
        let probe = ground(w, h, 3, 900);
        let dim = paint(w, h, &|x, y| grey(30 + noise(7, y * w + x) / 4));
        let bright = paint(w, h, &|x, y| grey(2 * (30 + noise(7, y * w + x) / 4)));
        let lifted = paint(w, h, &|x, y| grey(100 + noise(7, y * w + x) / 4));
        let d = whole(&picture(&dim, &probe, w, h));
        let b = whole(&picture(&bright, &probe, w, h));
        let l = whole(&picture(&lifted, &probe, w, h));
        assert_eq!(
            (b.laplacian_sq, b.deviation_sq),
            (4 * d.laplacian_sq, 4 * d.deviation_sq)
        );
        assert_eq!(b.normalised(), d.normalised());
        assert!(d.raw_energy() > 0.0);
        assert_eq!(l, d);
    }

    /// A picture whose bytes do not fit its size, and edges that do not rise strictly, read
    /// nothing; a picture too small for one 5 × 5 counts no pixel.
    #[test]
    fn a_malformed_picture_or_band_list_reads_nothing() {
        let (w, h) = (6, 6);
        let stripes = paint(w, h, &|x, _| grey(100 + 10 * (x % 2) as u8));
        let probe = ground(w, h, 1, 40);
        let good = picture(&stripes, &probe, w, h);
        assert_eq!(whole(&good).pixels, 4);
        let short_picture = picture(&stripes[..8], &probe, w, h);
        assert_eq!(read_sharpness(&short_picture, &[], &cell), nothing(1));
        let short_probe = picture(&stripes, &probe[..8], w, h);
        assert_eq!(read_sharpness(&short_probe, &[50.0], &cell), nothing(2));
        assert_eq!(
            read_sharpness(&picture(&[], &[], 0, 0), &[], &cell),
            nothing(1)
        );
        let falling: &[f64] = &[100.0, 50.0];
        let equal: &[f64] = &[70.0, 70.0];
        let not_a_number: &[f64] = &[f64::NAN];
        let endless: &[f64] = &[10.0, f64::INFINITY];
        for edges in [falling, equal, not_a_number, endless] {
            assert_eq!(
                read_sharpness(&good, edges, &cell),
                nothing(edges.len() + 1)
            );
        }
        let small = picture(&stripes[..64], &probe[..64], 4, 4);
        assert_eq!(read_sharpness(&small, &[], &cell), nothing(1));
    }
}
