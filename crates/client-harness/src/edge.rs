//! ★ THE EDGE METER (G-EDGE; the temporal path's slice T0, item 2) — the pure half.
//!
//! The meter reads how the picture blends the horizon and the limb: the share of the edge pixels
//! that show a mix of the sky and the ground, and the number of levels of that mix. It makes step
//! 19's measure an instrument (the edge kept: 0.41 against a 0.43 reference, commit `4c155d0`).
//! T0 reads today's multisampled picture as the "before" row. Each later slice of the temporal
//! path reads its own picture against that row (`docs/investigation/2026-09-28/
//! temporal_path_design.md` §11 T0, §12.4).
//!
//! **The edge.** The probe states what drew each pixel: its "nothing" kind is the sky, its terrain
//! kind is the ground. An EDGE pixel is a pixel whose closed four-neighbourhood (the pixel and its
//! four neighbours inside the frame) holds a sky pixel and a ground pixel. So the meter counts both
//! sides of the boundary: the last sky pixel and the first ground pixel of a column. When a
//! straight edge crosses a pixel away from the frame's border, that pixel is always an edge pixel:
//! along the axis on which the edge's normal leans most, the pixel's centre and the centre of the
//! next pixel across the edge stand on the two sides of it. So no blended pixel of the horizon
//! falls outside the set. The meter leaves out an edge pixel that also touches another kind (the
//! ruler) and counts it, because a third colour can mix into it.
//!
//! **The references.** A PURE pixel is a sky or ground pixel whose closed four-neighbourhood holds
//! only its own kind. An edge pixel takes its local sky and its local ground from the pure pixels
//! of each kind in the square of [`REFERENCE_RADIUS_PX`] round it: the median of each channel. A
//! median ignores a star beside the limb; a mean does not. The meter leaves out an edge pixel
//! without a pure pixel of each kind in its square, and counts it.
//!
//! **The blend.** The blend fraction `t` of a pixel is the projection of (pixel − ground) onto
//! (sky − ground), clamped to [0, 1]: 0 is the ground's colour, 1 is the sky's colour. Under an
//! anti-alias, `t` is the share of the pixel that the sky covers. A pixel is BLENDED when `t` lies
//! strictly between [`BLEND_LOW`] and [`BLEND_HIGH`]. Under the minimum contrast
//! ([`MIN_CONTRAST_SQ`]) the noise of the bytes can move `t` past a margin, so the meter leaves
//! such a pixel out and counts it.
//!
//! **The colour space.** The meter reads the picture's own bytes (sRGB, after the engine's tone
//! map), as every pixel gate of this crate does. The anti-alias mixes linear light. So a pixel
//! that the sky covers by half reads over one half where a bright sky stands over a dark ground,
//! and under one half where a black sky stands over a bright ground (the limb). This bend moves
//! the levels. For four samples it keeps their count and keeps them inside the margins (see
//! [`BLEND_LOW`]).
//!
//! **The reading.** The count of edge pixels, the counts left out, and a histogram of `t` in
//! [`EDGE_BINS`] bins: the two pure ends, and between them the blended pixels by their nearest
//! sixteenth. The blended count, the blended share and the number of blend levels come from the
//! histogram. The readings of many frames add ([`sum_edge_reads`]), so a still stand and a turn
//! each give one reading.
//!
//! **What the probe does not draw.** The probe draws the ground and the ruler only. A hull, a star
//! or the sun's disc stands in the probe's sky. A hole in the ground is "nothing" in the probe
//! too, so its rim reads as an edge; the hole instruments keep the stands' holes at zero.
//!
//! **One size only.** The meter reads a picture and a probe of ONE size, and it refuses any other
//! pair ([`read_edges`] reads nothing). Its edge set and its reference radius count on one probe
//! pixel under each picture pixel. That is T0's case and every case at the render scale 1. At T2
//! the probe stays at the render size and the display picture grows by 1/s (the design's §12.1),
//! so an up-scaled probe would put the probe's line up to one render pixel off the true edge. T2
//! owes the meter two things before G-EDGE reads a display picture at 0.67 or 0.5: a probe of its
//! own size, read under each display pixel's centre, and an edge band and a reference radius in
//! render pixels (1/s and 2/s display pixels), with a control at 0.5 (DEFERRED "THE SPEED STEP").
//! Example: at 0.5 the capture's 1 284 × 720 probe stands under a 2 568 × 1 440 picture. The true
//! horizon crosses display row 9.1, the up-scale blends rows 8 and 9, and a doubled probe puts
//! its line at row 10: row 8 then falls outside the set and serves as a "pure" sky reference.
//!
//! Example: the pilot stands on foot on the home planet and looks level at the horizon. In each
//! column the probe marks two edge pixels: the last sky pixel and the first ground pixel. Under
//! four samples a pixel, three columns in four hold one pixel whose samples split between the sky
//! and the ground. So 0.375 of the edge pixels read blended, at three levels: a quarter, a half
//! and three quarters of sky. A sky pass that reads one sample a pixel makes a stair of the same
//! edge: the share reads 0 and the levels read 0.

use crate::pop::HUD_MARGIN_PX;
use crate::probe::{PROBE_KIND_NONE, PROBE_KIND_TERRAIN, decode_probe};

/// THE REFERENCE RADIUS in pixels: an edge pixel reads its local sky and ground in the square of
/// this many pixels each way round it (5 × 5 pixels). Basis: two is the smallest radius that
/// always reaches a pure pixel of each side of a straight edge. Along the axis on which the edge's
/// normal leans most, the pixel two steps away on the far side is clear of the edge with its four
/// neighbours. The pixel one step away can be an edge pixel itself, because the edge band is two
/// pixels thick. A wider square reads colours farther from the edge: the sky's gradient and the
/// ground's shading.
pub const REFERENCE_RADIUS_PX: usize = 2;

/// THE LEVEL GRID: the meter reads a blend to the nearest sixteenth. Basis: an anti-alias of 2,
/// 4, 8 or 16 samples a pixel blends at k/n, which is a whole number of sixteenths, so each of its
/// levels has a bin of its own: four samples read three levels. A temporal history blends between
/// those levels and fills more bins (the design's weight floor in motion is 1/16).
pub const BLEND_STEPS: usize = 16;

/// The bins of the histogram. Bin 0 holds the pure pixels on the ground's side (`t` at or under
/// [`BLEND_LOW`]). Bin [`BLEND_STEPS`] holds the pure pixels on the sky's side (`t` at or over
/// [`BLEND_HIGH`]). Bins 1 to 15 hold the blended pixels by their nearest sixteenth.
pub const EDGE_BINS: usize = BLEND_STEPS + 1;

/// THE GROUND'S MARGIN: a pixel is blended when its fraction lies strictly between this margin
/// and [`BLEND_HIGH`]. A pixel within one sixteenth of its side's colour reads pure. Basis:
/// - over the minimum contrast the noise of the bytes moves `t` by one sixteenth at most
///   ([`MIN_CONTRAST_SQ`]), so noise alone never makes a pure pixel read blended;
/// - the sRGB bend keeps every level of four samples inside the margins. The sweep in this
///   module's tests checks every grey pair over the floor (50 400 pairs): the levels lie from
///   0.117 to 0.883, and the three levels of each pair fall in three bins.
///
/// A blend of one sample in sixteen stands on the margin and reads pure.
pub const BLEND_LOW: f64 = 1.0 / 16.0;

/// THE SKY'S MARGIN: see [`BLEND_LOW`].
pub const BLEND_HIGH: f64 = 15.0 / 16.0;

/// THE NOISE OF ONE PICTURE BYTE, in levels: half a level of rounding to eight bits, and about half
/// a level of the engine's deband dither. The engine turns the dither on for every 3D camera
/// (bevy_core_pipeline 0.18.1, `core_3d/mod.rs:145`) and adds ±0.5/255 in a gamma-2.2 space next
/// to the sRGB store (`tonemapping.wgsl:25-30`). The picture camera keeps the dither on; the probe
/// camera turns it off.
pub const PIXEL_NOISE_LEVELS: f64 = 1.0;

/// THE MINIMUM CONTRAST, squared, in levels²: the meter leaves out an edge pixel whose local sky
/// and ground differ by less. Basis: the pixel and its reference each carry up to
/// [`PIXEL_NOISE_LEVELS`] a channel, so their difference carries up to twice that a channel. Over
/// the three channels that is a step of 2·√3 levels in length. The step moves `t` by its length
/// over the contrast, and the floor keeps that move at or under [`BLEND_LOW`]. So the floor is
/// 2·√3·[`PIXEL_NOISE_LEVELS`] / [`BLEND_LOW`] = 32·√3 ≈ 55.4 levels, 3 072 squared.
pub const MIN_CONTRAST_SQ: f64 =
    3.0 * (2.0 * PIXEL_NOISE_LEVELS / BLEND_LOW) * (2.0 * PIXEL_NOISE_LEVELS / BLEND_LOW);

/// The class bits of a probe pixel and of a neighbourhood: the sky, the ground, another kind.
const SKY: u8 = 1;
const GROUND: u8 = 2;
const OTHER: u8 = 4;

/// The pixels in the square of [`REFERENCE_RADIUS_PX`]: the most one side can collect.
const WINDOW_AREA: usize = (2 * REFERENCE_RADIUS_PX + 1) * (2 * REFERENCE_RADIUS_PX + 1);

/// One picture and its probe, as the meter reads them: the picture's RGBA bytes, the probe's RGBA
/// bytes of the same size, the size, and the overlay's rectangle.
pub struct EdgeFrame<'a> {
    pub rgba: &'a [u8],
    pub probe: &'a [u8],
    pub width: usize,
    pub height: usize,
    /// THE OVERLAY'S RECTANGLE (left, top, right, bottom, pixels), the stamp's `hud_rect_px`. The
    /// meter judges no pixel under it and takes no reference from it. The rectangle grows by
    /// [`HUD_MARGIN_PX`], as the pop detector grows it.
    pub hud: Option<[f64; 4]>,
}

impl EdgeFrame<'_> {
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

    /// The picture's colour at pixel `i` (row-major).
    fn colour(&self, i: usize) -> [u8; 3] {
        let j = i * 4;
        [self.rgba[j], self.rgba[j + 1], self.rgba[j + 2]]
    }
}

/// What the meter read on one frame, or on many frames summed. The counts add and the histograms
/// pool, so the share and the levels are one reading of the whole. Every edge pixel is in one
/// count: `edge` = `other_kind` + `unreferenced` + `low_contrast` + [`EdgeRead::measured`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EdgeRead {
    /// The edge pixels outside the overlay.
    pub edge: u64,
    /// Left out: edge pixels that touch another kind than the sky and the ground (the ruler).
    pub other_kind: u64,
    /// Left out: edge pixels without a pure sky pixel or a pure ground pixel in their square.
    pub unreferenced: u64,
    /// Left out: edge pixels whose local sky and ground differ by less than the minimum contrast.
    pub low_contrast: u64,
    /// The measured edge pixels by their blend fraction (see [`EDGE_BINS`]).
    pub hist: [u64; EDGE_BINS],
}

impl EdgeRead {
    /// The edge pixels that the meter measured: every bin of the histogram.
    #[must_use]
    pub fn measured(&self) -> u64 {
        self.hist.iter().sum()
    }

    /// The edge pixels that the meter left out, for any of the three reasons.
    #[must_use]
    pub fn left_out(&self) -> u64 {
        self.other_kind + self.unreferenced + self.low_contrast
    }

    /// The blended pixels: the bins between the two pure ends.
    #[must_use]
    pub fn blended(&self) -> u64 {
        self.hist[1..BLEND_STEPS].iter().sum()
    }

    /// THE BLENDED SHARE: the blended pixels over the measured pixels (0 when the meter measured
    /// none).
    #[must_use]
    pub fn blended_share(&self) -> f64 {
        self.blended() as f64 / self.measured().max(1) as f64
    }

    /// THE BLEND LEVELS: how many bins between the two pure ends hold a blended pixel.
    #[must_use]
    pub fn levels(&self) -> usize {
        self.hist[1..BLEND_STEPS].iter().filter(|n| **n > 0).count()
    }
}

/// THE BLEND FRACTION of a pixel between its local sky and its local ground: the projection of
/// (pixel − ground) onto (sky − ground), clamped to [0, 1]. `None` when the two references differ
/// by less than the minimum contrast ([`MIN_CONTRAST_SQ`]).
#[must_use]
pub fn blend_fraction(pixel: [u8; 3], sky: [f64; 3], ground: [f64; 3]) -> Option<f64> {
    let mut along = 0.0;
    let mut contrast_sq = 0.0;
    for ((c, s), g) in pixel.into_iter().zip(sky).zip(ground) {
        let d = s - g;
        along += (f64::from(c) - g) * d;
        contrast_sq += d * d;
    }
    (contrast_sq >= MIN_CONTRAST_SQ).then(|| (along / contrast_sq).clamp(0.0, 1.0))
}

/// THE BIN of a blend fraction in [0, 1]: 0 at or under [`BLEND_LOW`], [`BLEND_STEPS`] at or over
/// [`BLEND_HIGH`], and between them the nearest sixteenth (1 to 15).
#[must_use]
pub fn blend_bin(t: f64) -> usize {
    let nearest = ((t * BLEND_STEPS as f64).round() as usize).clamp(1, BLEND_STEPS - 1);
    if t <= BLEND_LOW {
        0
    } else if t >= BLEND_HIGH {
        BLEND_STEPS
    } else {
        nearest
    }
}

/// The class bit of one probe pixel: the sky, the ground, or another kind.
fn kind_bit(px: &[u8]) -> u8 {
    match decode_probe([px[0], px[1], px[2]]).kind {
        PROBE_KIND_NONE => SKY,
        PROBE_KIND_TERRAIN => GROUND,
        _ => OTHER,
    }
}

/// The class bits of each pixel's closed four-neighbourhood: the pixel and its four neighbours
/// inside the frame. A neighbour outside the frame reads as the pixel itself.
fn hoods(kinds: &[u8], width: usize, height: usize) -> Vec<u8> {
    (0..kinds.len())
        .map(|i| {
            let (x, y) = (i % width, i / width);
            kinds[i]
                | kinds[i - usize::from(x > 0)]
                | kinds[i + usize::from(x + 1 < width)]
                | kinds[i - width * usize::from(y > 0)]
                | kinds[i + width * usize::from(y + 1 < height)]
        })
        .collect()
}

/// The median of a run of levels that is not empty: the middle value, or the mean of the two
/// middle values of an even count. It sorts the run in place.
fn median_of(run: &mut [u8]) -> f64 {
    run.sort_unstable();
    let n = run.len();
    (f64::from(run[(n - 1) / 2]) + f64::from(run[n / 2])) * 0.5
}

/// The pure pixels of one side round an edge pixel: their channels, as the square collects them.
#[derive(Default)]
struct Side {
    n: usize,
    channels: [[u8; WINDOW_AREA]; 3],
}

impl Side {
    fn push(&mut self, colour: [u8; 3]) {
        for (channel, level) in self.channels.iter_mut().zip(colour) {
            channel[self.n] = level;
        }
        self.n += 1;
    }

    /// The side's colour: the median of each channel, or `None` when the side has no pixel. It
    /// sorts each channel's run in place.
    fn median(&mut self) -> Option<[f64; 3]> {
        let n = self.n;
        (n > 0).then(|| {
            self.channels
                .each_mut()
                .map(|channel| median_of(&mut channel[..n]))
        })
    }
}

/// What the meter did with one edge pixel.
enum Judged {
    OtherKind,
    Unreferenced,
    LowContrast,
    Bin(usize),
}

/// THE JUDGE of the edge pixel at `(x, y)`: another kind in its neighbourhood, no reference of a
/// side, too little contrast, or the bin of its blend fraction.
fn judge(frame: &EdgeFrame<'_>, hood: &[u8], x: usize, y: usize) -> Judged {
    let i = y * frame.width + x;
    if hood[i] & OTHER != 0 {
        return Judged::OtherKind;
    }
    let (mut sky, mut ground) = (Side::default(), Side::default());
    let r = REFERENCE_RADIUS_PX;
    for wy in y.saturating_sub(r)..=(y + r).min(frame.height - 1) {
        for wx in x.saturating_sub(r)..=(x + r).min(frame.width - 1) {
            let j = wy * frame.width + wx;
            let free = !frame.under_overlay(wx, wy);
            if free & (hood[j] == SKY) {
                sky.push(frame.colour(j));
            }
            if free & (hood[j] == GROUND) {
                ground.push(frame.colour(j));
            }
        }
    }
    let (Some(s), Some(g)) = (sky.median(), ground.median()) else {
        return Judged::Unreferenced;
    };
    blend_fraction(frame.colour(i), s, g).map_or(Judged::LowContrast, |t| Judged::Bin(blend_bin(t)))
}

/// THE EDGE METER on one frame: every edge pixel outside the overlay, judged against its local
/// sky and ground. A frame that is not one picture and one probe of its stated size reads nothing.
#[must_use]
pub fn read_edges(frame: &EdgeFrame<'_>) -> EdgeRead {
    let mut read = EdgeRead::default();
    if !frame.well_formed() {
        return read;
    }
    let kinds: Vec<u8> = frame.probe.chunks_exact(4).map(kind_bit).collect();
    let hood = hoods(&kinds, frame.width, frame.height);
    for (i, bits) in hood.iter().enumerate() {
        let (x, y) = (i % frame.width, i / frame.width);
        let edge = (bits & SKY != 0) & (bits & GROUND != 0);
        if !edge | frame.under_overlay(x, y) {
            continue;
        }
        read.edge += 1;
        match judge(frame, &hood, x, y) {
            Judged::OtherKind => read.other_kind += 1,
            Judged::Unreferenced => read.unreferenced += 1,
            Judged::LowContrast => read.low_contrast += 1,
            Judged::Bin(bin) => read.hist[bin] += 1,
        }
    }
    read
}

/// The sum of many frames' readings: the counts add and the histograms pool, so the share and the
/// levels of a still stand or of a turn are one reading of all its frames.
#[must_use]
pub fn sum_edge_reads(reads: &[EdgeRead]) -> EdgeRead {
    let mut total = EdgeRead::default();
    for r in reads {
        total.edge += r.edge;
        total.other_kind += r.other_kind;
        total.unreferenced += r.unreferenced;
        total.low_contrast += r.low_contrast;
        for (sum, n) in total.hist.iter_mut().zip(r.hist) {
            *sum += n;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::probe::{PROBE_KIND_RULER, ProbePixel, encode_probe};

    /// The probe's bytes of one kind (the rung and the distance do not matter to the meter).
    fn probe_px(kind: u8) -> [u8; 4] {
        let p = encode_probe(ProbePixel {
            kind,
            rung: 3,
            cells: 900,
        });
        [p[0], p[1], p[2], 255]
    }

    /// One coverage sample a pixel, at its centre: the stair that the defect draws.
    const ONE: [(f64, f64); 1] = [(0.5, 0.5)];

    /// The standard four-sample pattern (Vulkan, Direct3D), from the pixel's top-left corner.
    const FOUR: [(f64, f64); 4] = [
        (0.375, 0.125),
        (0.875, 0.375),
        (0.125, 0.625),
        (0.625, 0.875),
    ];

    /// Sixteen samples a pixel on a 4 × 4 grid.
    fn sixteen() -> Vec<(f64, f64)> {
        (0..16u32)
            .map(|k| {
                (
                    (f64::from(k % 4) + 0.5) / 4.0,
                    (f64::from(k / 4) + 0.5) / 4.0,
                )
            })
            .collect()
    }

    /// A day sky and a grass ground: each channel's step (160, 160, 224) divides by sixteen.
    const DAY_SKY: [u8; 3] = [200, 216, 248];
    const GRASS: [u8; 3] = [40, 56, 24];
    /// Space and a lit limb: each channel's step divides by sixteen.
    const SPACE: [u8; 3] = [0, 0, 0];
    const LIT: [u8; 3] = [160, 176, 208];

    /// The width and height of the horizon scenes.
    const W: usize = 40;
    const H: usize = 24;

    /// A tilted horizon (a banked hull): the sky over a line that falls 0.3 rows a column, from
    /// row 7.3 at the left to row 19.3 at the right.
    fn tilted(x: f64, y: f64) -> bool {
        y < 7.3 + 0.3 * x
    }

    /// The limb of the planet from orbit: the ground inside a disc, the sky round it.
    fn limb(cx: f64, cy: f64, r: f64) -> impl Fn(f64, f64) -> bool {
        move |x, y| (x - cx).powi(2) + (y - cy).powi(2) > r * r
    }

    /// A drawn scene: the picture, the probe, and what the drawing knows about its own blend.
    struct Drawn {
        w: usize,
        h: usize,
        rgba: Vec<u8>,
        probe: Vec<u8>,
        /// The pixels whose samples split between the sky and the ground.
        partial: u64,
        /// Those pixels by the count of their samples on the sky.
        counts: BTreeMap<usize, u64>,
    }

    impl Drawn {
        fn frame(&self) -> EdgeFrame<'_> {
            EdgeFrame {
                rgba: &self.rgba,
                probe: &self.probe,
                width: self.w,
                height: self.h,
                hud: None,
            }
        }
    }

    /// Draw a scene of `w × h`: `sky_at(x, y)` says whether a point (pixels, y down) shows the
    /// sky. The probe reads each pixel's centre, with no anti-alias, as the probe camera does. The
    /// picture holds `mix(k)` for `k` of the samples on the sky.
    fn draw(
        w: usize,
        h: usize,
        sky_at: &dyn Fn(f64, f64) -> bool,
        samples: &[(f64, f64)],
        mix: &dyn Fn(usize) -> [u8; 3],
    ) -> Drawn {
        let mut drawn = Drawn {
            w,
            h,
            rgba: Vec::with_capacity(w * h * 4),
            probe: Vec::with_capacity(w * h * 4),
            partial: 0,
            counts: BTreeMap::new(),
        };
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x as f64, y as f64);
                let k = samples
                    .iter()
                    .filter(|(sx, sy)| sky_at(px + sx, py + sy))
                    .count();
                let kind = if sky_at(px + 0.5, py + 0.5) {
                    PROBE_KIND_NONE
                } else {
                    PROBE_KIND_TERRAIN
                };
                let c = mix(k);
                drawn.rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
                drawn.probe.extend_from_slice(&probe_px(kind));
                if (k > 0) & (k < samples.len()) {
                    drawn.partial += 1;
                    *drawn.counts.entry(k).or_insert(0) += 1;
                }
            }
        }
        drawn
    }

    /// The byte mix of `k` sky samples of `n`: exact when each channel's step divides by `n`.
    fn lerp_bytes(sky: [u8; 3], ground: [u8; 3], k: usize, n: usize) -> [u8; 3] {
        let mut c = [0u8; 3];
        for ((out, s), g) in c.iter_mut().zip(sky).zip(ground) {
            let (s, g) = (i64::from(s), i64::from(g));
            *out = (g + (s - g) * k as i64 / n as i64) as u8;
        }
        c
    }

    /// The sRGB byte's linear light, and back: the GPU's own conversion at the picture's store.
    fn to_linear(b: u8) -> f64 {
        let x = f64::from(b) / 255.0;
        if x <= 0.040_45 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    }

    fn to_srgb(l: f64) -> u8 {
        let x = if l <= 0.003_130_8 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        (x * 255.0).round() as u8
    }

    /// The mix of `k` sky samples of `n` in linear light, stored as sRGB bytes: what the GPU's
    /// multisample resolve and the picture's store make.
    fn lerp_light(sky: [u8; 3], ground: [u8; 3], k: usize, n: usize) -> [u8; 3] {
        let f = k as f64 / n as f64;
        let mut c = [0u8; 3];
        for ((out, s), g) in c.iter_mut().zip(sky).zip(ground) {
            let (s, g) = (to_linear(s), to_linear(g));
            *out = to_srgb(g + (s - g) * f);
        }
        c
    }

    /// The bins between the pure ends that hold a pixel.
    fn occupied(read: &EdgeRead) -> BTreeSet<usize> {
        (1..BLEND_STEPS).filter(|k| read.hist[*k] > 0).collect()
    }

    /// ★ CONTROL — THE DEFECT: a hard horizon (one sample a pixel, the stair that a one-sample
    /// sky pass draws) reads a share of 0 and no level.
    #[test]
    fn control_a_hard_horizon_reads_a_share_of_zero() {
        let drawn = draw(W, H, &tilted, &ONE, &|k| lerp_bytes(DAY_SKY, GRASS, k, 1));
        assert_eq!(drawn.partial, 0);
        let read = read_edges(&drawn.frame());
        // Two edge pixels a column: the last sky pixel and the first ground pixel. The line falls
        // less than a row a column, so no column adds a third.
        assert_eq!(read.edge, 2 * W as u64);
        assert_eq!(read.left_out(), 0);
        assert_eq!(read.blended(), 0);
        assert_eq!(read.blended_share(), 0.0);
        assert_eq!(read.levels(), 0);
        let mut hist = [0u64; EDGE_BINS];
        hist[0] = W as u64;
        hist[BLEND_STEPS] = W as u64;
        assert_eq!(read.hist, hist);
    }

    /// ★ CONTROL — FOUR SAMPLES: the tilted horizon under the standard four-sample pattern reads
    /// its known share and exactly three levels, a quarter, a half and three quarters (sixteenths
    /// 4, 8 and 12). Then the same edge mixed in linear light and stored as sRGB bytes, as the GPU
    /// stores it: the levels move, their count does not.
    ///
    /// THE KNOWN SHARE, by hand. Along the line, the four samples stand at the heights
    /// `sy − 0.3·sx`: 0.0125, 0.1125, 0.5875 and 0.6875 of a row. In each run of ten columns, the
    /// line crosses its row at 0.3, 0.6, 0.9, 0.2, 0.5, 0.8, 0.1, 0.4, 0.7 and 0.0 of the row. The
    /// samples split where that height lies between 0.0125 and 0.6875: six columns in ten, one
    /// pixel each. So 24 of the 80 edge pixels blend (0.3): 4 with one sample on the sky (at 0.1),
    /// 16 with two, 4 with three (at 0.6).
    #[test]
    fn control_four_samples_read_their_known_share_and_three_levels() {
        let drawn = draw(W, H, &tilted, &FOUR, &|k| lerp_bytes(DAY_SKY, GRASS, k, 4));
        assert_eq!(drawn.partial, 24);
        assert_eq!(drawn.counts, BTreeMap::from([(1, 4), (2, 16), (3, 4)]));
        let read = read_edges(&drawn.frame());
        assert_eq!(read.edge, 2 * W as u64);
        assert_eq!(read.left_out(), 0);
        // Every split pixel is an edge pixel and reads blended; no other pixel reads blended.
        assert_eq!(read.blended(), drawn.partial);
        assert_eq!(read.blended_share(), 0.3);
        assert_eq!(read.levels(), 3);
        assert_eq!(occupied(&read), BTreeSet::from([4, 8, 12]));
        for (k, n) in &drawn.counts {
            assert_eq!(read.hist[4 * k], *n, "{k} samples of four on the sky");
        }
        let light = draw(W, H, &tilted, &FOUR, &|k| lerp_light(DAY_SKY, GRASS, k, 4));
        let read = read_edges(&light.frame());
        assert_eq!(read.blended(), light.partial);
        assert_eq!(read.levels(), 3);
        assert_eq!(occupied(&read), BTreeSet::from([7, 11, 14]));
    }

    /// ★ CONTROL — LOW CONTRAST: a hazy horizon whose ground stands 31 levels a channel under its
    /// sky is left out, and every edge pixel is counted as low contrast. At 32 levels a channel the
    /// same edge stands on the floor and the meter measures it.
    #[test]
    fn control_a_low_contrast_edge_is_left_out_and_counted() {
        let (haze, far_ground) = ([180, 190, 200], [149, 159, 169]);
        let drawn = draw(W, H, &tilted, &FOUR, &|k| {
            lerp_bytes(haze, far_ground, k, 4)
        });
        let read = read_edges(&drawn.frame());
        assert_eq!(read.edge, 2 * W as u64);
        assert_eq!(read.low_contrast, read.edge);
        assert_eq!(read.measured(), 0);
        assert_eq!(read.blended_share(), 0.0);
        assert_eq!(read.levels(), 0);
        let on_floor = [148, 158, 168];
        let drawn = draw(W, H, &tilted, &FOUR, &|k| lerp_bytes(haze, on_floor, k, 4));
        let read = read_edges(&drawn.frame());
        assert_eq!(read.low_contrast, 0);
        assert_eq!(read.measured(), read.edge);
        assert_eq!(read.blended(), drawn.partial);
        assert_eq!(read.levels(), 3);
    }

    /// THE EXAMPLE of the module: a level horizon under four samples, at 64 heights across one
    /// row, pooled. A column blends when the line stands between the first and the last sample's
    /// height, three columns in four, and each column has two edge pixels: 3/8 of them blend.
    #[test]
    fn a_level_horizon_under_four_samples_reads_three_eighths_over_every_height() {
        let reads: Vec<EdgeRead> = (0..64u32)
            .map(|j| {
                let line = 10.0 + (f64::from(j) + 0.5) / 64.0;
                let drawn = draw(W, H, &|_, y| y < line, &FOUR, &|k| {
                    lerp_bytes(DAY_SKY, GRASS, k, 4)
                });
                read_edges(&drawn.frame())
            })
            .collect();
        let sum = sum_edge_reads(&reads);
        assert_eq!(sum.edge, 64 * 2 * W as u64);
        assert_eq!(sum.left_out(), 0);
        assert_eq!(sum.blended_share(), 0.375);
        assert_eq!(sum.levels(), 3);
    }

    /// Sixteen samples a pixel read more levels than four: each count of samples on the sky reads
    /// its own sixteenth. One sample in sixteen (and fifteen) stands on a margin and reads pure.
    #[test]
    fn sixteen_samples_read_more_levels_than_four() {
        let samples = sixteen();
        let drawn = draw(W, H, &tilted, &samples, &|k| {
            lerp_bytes(DAY_SKY, GRASS, k, 16)
        });
        let read = read_edges(&drawn.frame());
        assert_eq!(read.left_out(), 0);
        let inside: BTreeMap<usize, u64> = drawn
            .counts
            .iter()
            .filter(|(k, _)| (2..BLEND_STEPS - 1).contains(*k))
            .map(|(k, n)| (*k, *n))
            .collect();
        assert_eq!(read.blended(), inside.values().sum::<u64>());
        assert_eq!(occupied(&read), inside.keys().copied().collect());
        assert!(read.levels() > 3, "{read:?}");
    }

    /// THE REFERENCE RADIUS reaches both sides of the limb at every angle: round a disc every
    /// direction of the edge's normal occurs, and no edge pixel lacks a reference. Every pixel
    /// whose four samples split is an edge pixel and reads blended.
    #[test]
    fn the_radius_reaches_both_sides_of_the_limb_at_every_angle() {
        for (cx, cy, r) in [(32.3, 31.7, 26.0), (32.1, 32.6, 17.5), (31.8, 32.2, 9.25)] {
            let drawn = draw(64, 64, &limb(cx, cy, r), &FOUR, &|k| {
                lerp_bytes(SPACE, LIT, k, 4)
            });
            let read = read_edges(&drawn.frame());
            assert!(read.edge > 0, "radius {r}");
            assert_eq!(read.left_out(), 0, "radius {r}: {read:?}");
            assert_eq!(read.blended(), drawn.partial, "radius {r}");
            assert_eq!(read.levels(), 3, "radius {r}");
        }
    }

    /// A star beside the limb moves no reference: the median of the pure sky ignores it, and the
    /// reading does not change.
    #[test]
    fn a_star_beside_the_limb_moves_no_reference() {
        let mut drawn = draw(64, 64, &limb(32.3, 31.7, 20.0), &FOUR, &|k| {
            lerp_bytes(SPACE, LIT, k, 4)
        });
        let before = read_edges(&drawn.frame());
        // The disc's top edge band is rows 11 and 12 at column 32; the star stands at row 10.
        let i = (10 * 64 + 32) * 4;
        drawn.rgba[i..i + 3].copy_from_slice(&[255, 255, 255]);
        assert_eq!(read_edges(&drawn.frame()), before);
        assert_eq!(before.levels(), 3);
    }

    /// The median of a side: none for no pixel, the middle value of an odd count (one star among
    /// three pixels is ignored), the mean of the two middle values of an even count.
    #[test]
    fn the_median_of_a_side_ignores_a_star_and_reads_an_even_count_midway() {
        let mut side = Side::default();
        assert_eq!(side.median(), None);
        for level in [0u8, 255, 0] {
            side.push([level, level, 10]);
        }
        assert_eq!(side.median(), Some([0.0, 0.0, 10.0]));
        side.push([40, 40, 30]);
        assert_eq!(side.median(), Some([20.0, 20.0, 10.0]));
    }

    /// The blend fraction projects, clamps, and needs the contrast. At the floor, a pixel two levels
    /// a channel off its reference (one level of its own noise, one of the reference's) stands
    /// exactly on a margin and reads pure.
    #[test]
    fn the_blend_fraction_projects_clamps_and_needs_the_contrast() {
        let (sky, ground) = ([200.0, 216.0, 248.0], [40.0, 56.0, 24.0]);
        assert_eq!(blend_fraction([40, 56, 24], sky, ground), Some(0.0));
        assert_eq!(blend_fraction([200, 216, 248], sky, ground), Some(1.0));
        assert_eq!(blend_fraction([120, 136, 136], sky, ground), Some(0.5));
        // Past either colour, the fraction clamps.
        assert_eq!(blend_fraction([255, 255, 255], sky, ground), Some(1.0));
        assert_eq!(blend_fraction([0, 0, 0], sky, ground), Some(0.0));
        // The floor: 32 levels a channel stands on it (3 072 levels squared); 31 is under it.
        assert_eq!(MIN_CONTRAST_SQ, 3072.0);
        assert!(blend_fraction([100; 3], [132.0; 3], [100.0; 3]).is_some());
        assert_eq!(blend_fraction([100; 3], [131.0; 3], [100.0; 3]), None);
        // The noise at the floor, toward the sky from the ground and toward the ground from the sky.
        let low = blend_fraction([102; 3], [132.0; 3], [100.0; 3]).expect("on the floor");
        let high = blend_fraction([130; 3], [132.0; 3], [100.0; 3]).expect("on the floor");
        assert_eq!((low, high), (BLEND_LOW, BLEND_HIGH));
        assert_eq!((blend_bin(low), blend_bin(high)), (0, BLEND_STEPS));
    }

    /// The bin: the pure ends at the margins, and between them the nearest sixteenth.
    #[test]
    fn a_blend_bin_puts_the_margins_at_the_ends_and_the_rest_at_the_nearest_sixteenth() {
        assert_eq!(blend_bin(0.0), 0);
        assert_eq!(blend_bin(BLEND_LOW), 0);
        assert_eq!(blend_bin(BLEND_LOW + 1e-9), 1);
        assert_eq!(blend_bin(0.25), 4);
        assert_eq!(blend_bin(0.27), 4);
        assert_eq!(blend_bin(0.5), 8);
        assert_eq!(blend_bin(0.75), 12);
        assert_eq!(blend_bin(BLEND_HIGH - 1e-9), BLEND_STEPS - 1);
        assert_eq!(blend_bin(BLEND_HIGH), BLEND_STEPS);
        assert_eq!(blend_bin(1.0), BLEND_STEPS);
    }

    /// THE BEND, checked for [`BLEND_LOW`]'s basis: on every grey pair over the floor, four samples
    /// mixed in linear light and stored as sRGB bytes read three levels in three bins, inside the
    /// margins.
    #[test]
    fn the_srgb_bend_keeps_four_samples_at_three_levels_inside_the_margins() {
        // The conversion itself: both pieces of the curve, and a round trip on every byte.
        assert_eq!((to_linear(0), to_linear(255)), (0.0, 1.0));
        assert_eq!((to_srgb(0.0), to_srgb(1.0)), (0, 255));
        for b in 0..=255u8 {
            assert_eq!(to_srgb(to_linear(b)), b);
        }
        let (mut lowest, mut highest, mut pairs) = (1.0_f64, 0.0_f64, 0u32);
        for sky in 0..=255u8 {
            for ground in 0..=255u8 {
                let d = f64::from(sky) - f64::from(ground);
                if 3.0 * d * d < MIN_CONTRAST_SQ {
                    continue;
                }
                pairs += 1;
                let (sky_f, ground_f) = ([f64::from(sky); 3], [f64::from(ground); 3]);
                let mut bins = BTreeSet::new();
                for k in 1..4 {
                    let c = lerp_light([sky; 3], [ground; 3], k, 4);
                    let t = blend_fraction(c, sky_f, ground_f).expect("over the floor");
                    lowest = lowest.min(t);
                    highest = highest.max(t);
                    bins.insert(blend_bin(t));
                }
                assert_eq!(bins.len(), 3, "sky {sky}, ground {ground}");
            }
        }
        assert_eq!(pairs, 50_400);
        assert!(lowest > BLEND_LOW, "{lowest}");
        assert!(highest < BLEND_HIGH, "{highest}");
        assert!((lowest - 0.117).abs() < 5e-4, "{lowest}");
        assert!((highest - 0.883).abs() < 5e-4, "{highest}");
    }

    /// THE OVERLAY is neither judged nor a reference. Its rectangle over the first nine columns,
    /// grown by the margin, takes columns 0 to 10 out of the edge. A rectangle over the sky of a
    /// level horizon, down to the row next to the edge band, leaves the edge judged but without a
    /// sky reference.
    #[test]
    fn the_overlay_is_neither_judged_nor_a_reference() {
        let drawn = draw(W, H, &tilted, &ONE, &|k| lerp_bytes(DAY_SKY, GRASS, k, 1));
        let mut frame = drawn.frame();
        frame.hud = Some([0.0, 0.0, 9.0, H as f64]);
        let read = read_edges(&frame);
        assert_eq!(read.edge, 2 * (W as u64 - 11));
        assert_eq!(read.left_out(), 0);
        // A level horizon: rows 0 to 9 are sky, the edge band is rows 9 and 10. The rectangle
        // reaches row 7, grown to row 8: the rows of pure sky next to the band.
        let level = draw(W, H, &|_, y| y < 10.0, &ONE, &|k| {
            lerp_bytes(DAY_SKY, GRASS, k, 1)
        });
        let mut frame = level.frame();
        frame.hud = Some([0.0, 0.0, W as f64, 7.0]);
        let read = read_edges(&frame);
        assert_eq!(read.edge, 2 * W as u64);
        assert_eq!(read.unreferenced, read.edge);
    }

    /// THE FRAME'S BORDER: an edge band on the top row has no pure sky, on the bottom row no pure
    /// ground, and in a frame of two rows neither; each such edge pixel is counted unreferenced.
    #[test]
    fn an_edge_on_the_frame_border_is_left_out_as_unreferenced() {
        for (h, line) in [(H, 1.0), (H, H as f64 - 1.0), (2, 1.0)] {
            let drawn = draw(W, h, &|_, y| y < line, &ONE, &|k| {
                lerp_bytes(DAY_SKY, GRASS, k, 1)
            });
            let read = read_edges(&drawn.frame());
            assert_eq!(read.edge, 2 * W as u64, "the line at row {line}");
            assert_eq!(read.unreferenced, read.edge, "the line at row {line}");
        }
    }

    /// AN EDGE THAT TOUCHES THE RULER is left out and counted. The ruler's top pixel stands on the
    /// first ground row of a level horizon: it and its two ground neighbours touch it, and the sky
    /// pixel over it loses its ground neighbour and leaves the edge.
    #[test]
    fn an_edge_that_touches_the_ruler_is_left_out_and_counted() {
        let mut drawn = draw(W, H, &|_, y| y < 10.0, &ONE, &|k| {
            lerp_bytes(DAY_SKY, GRASS, k, 1)
        });
        let i = (10 * W + 20) * 4;
        drawn.probe[i..i + 4].copy_from_slice(&probe_px(PROBE_KIND_RULER));
        let read = read_edges(&drawn.frame());
        assert_eq!(read.edge, 2 * W as u64 - 1);
        assert_eq!(read.other_kind, 3);
        assert_eq!(read.measured(), 2 * W as u64 - 4);
        assert_eq!(read.blended(), 0);
    }

    /// A frame of no size, or of bytes that do not match its size, reads nothing.
    #[test]
    fn a_malformed_frame_reads_nothing() {
        let drawn = draw(W, H, &tilted, &FOUR, &|k| lerp_bytes(DAY_SKY, GRASS, k, 4));
        assert_eq!(read_edges(&drawn.frame()).edge, 2 * W as u64);
        let empty = EdgeFrame {
            rgba: &[],
            probe: &[],
            width: 0,
            height: 0,
            hud: None,
        };
        assert_eq!(read_edges(&empty), EdgeRead::default());
        let mut short = drawn.frame();
        short.rgba = &drawn.rgba[..8];
        assert_eq!(read_edges(&short), EdgeRead::default());
        let mut short_probe = drawn.frame();
        short_probe.probe = &drawn.probe[..8];
        assert_eq!(read_edges(&short_probe), EdgeRead::default());
    }

    /// The sum pools the counts and the histograms: one share and one level count for the whole
    /// leg, never a share of one frame against the levels of another.
    #[test]
    fn the_sum_pools_the_counts_and_the_histograms() {
        let mut a = EdgeRead {
            edge: 10,
            other_kind: 1,
            unreferenced: 2,
            low_contrast: 3,
            ..EdgeRead::default()
        };
        a.hist[0] = 2;
        a.hist[4] = 1;
        a.hist[BLEND_STEPS] = 1;
        let mut b = EdgeRead {
            edge: 6,
            ..EdgeRead::default()
        };
        b.hist[8] = 3;
        b.hist[BLEND_STEPS] = 3;
        assert_eq!((a.blended_share(), a.levels()), (0.25, 1));
        let sum = sum_edge_reads(&[a.clone(), b.clone()]);
        assert_eq!(
            (sum.edge, sum.other_kind, sum.unreferenced, sum.low_contrast),
            (16, 1, 2, 3)
        );
        assert_eq!(sum.left_out(), 6);
        assert_eq!((sum.measured(), sum.blended(), sum.levels()), (10, 4, 2));
        assert_eq!(sum.blended_share(), 0.4);
        assert_eq!(sum_edge_reads(&[]), EdgeRead::default());
        assert_eq!(EdgeRead::default().blended_share(), 0.0);
    }
}
