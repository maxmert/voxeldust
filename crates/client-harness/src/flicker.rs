//! ★ THE FLICKER METER (G-FLICKER; the temporal path, slice T0 item 2; design §11 and §12.4) — the
//! pure half.
//!
//! A flicker is a change of a pixel from one frame to the next while the eye and the ground stand
//! still. The temporal path builds each frame from the frames before it, and it shifts each frame
//! by a part of a pixel (the jitter). So a pixel can change a little in each frame although nothing
//! moves: an edge crawls, a thin ridge comes and goes, a shadow's edge steps. This meter measures
//! that change over [`STILL_FRAMES`] frames at a still stand. Slice T0 reads today's picture with it
//! first: these are the "before" rows that every later slice compares with (design §11, M-6).
//!
//! **The step.** For each pixel and each pair of consecutive frames, the step is the largest
//! difference of the colour channels, in levels of 255 ([`COLOUR_CHANNELS`]). One step is one
//! COUNT: one pixel in one pair of frames.
//!
//! **The regions.** Each count goes into ONE region. The first rule that claims the pixel wins:
//! 1. the overlay: the meter leaves out a pixel under the stamp's rectangle, because its readouts
//!    change in each frame;
//! 2. the stars: a pixel of the caller's star mask;
//! 3. the shadow edges: a pixel of the caller's shadow-edge mask;
//! 4. the edges: a pixel whose 3 × 3 probe neighbourhood holds two kinds or two rungs, in either
//!    probe of the pair ([`EDGE_NEIGHBOURHOOD_PX`]);
//! 5. the sky, the ground or the other drawn things, by the probe's kind at the pixel.
//!
//! A mask names its pixels on purpose, so it claims them before the probe's own classes. A still
//! has one probe for all its frames, or one probe a frame; a pair reads both of its probes.
//!
//! **The reading.** Each region holds a histogram of its steps. The histogram gives the share of
//! the counts within one level and within two levels, the largest step and the pixel count. The
//! pairs of a still add up as in the pop detector: the counts add and the histograms pool.
//!
//! **The targets (PROPOSED; the owner rules them).** At a still stand after the history converges,
//! 99.9 % of the content changes by one level a frame or less, and 99.9 % of the edges by two
//! levels or less ([`TARGET_SHARE`], [`CONTENT_TARGET_LEVELS`], [`EDGE_TARGET_LEVELS`]). The
//! content is every region except the edges. An empty region has nothing to judge, and its verdict
//! says so (`None`).
//!
//! Example (an illustration, not a measurement): a pilot stands on foot on the home planet and
//! looks at the horizon for 64 frames. The sky and the ground stay within one level. The ridge at
//! the horizon falls in the edge region. If the jitter crawls on the ridge by three levels in some
//! frames, the edges read a largest step of 3. If that crawl touches more than a thousandth of the
//! edge counts, the edges fail their two-level target.

use crate::pop::{HUD_MARGIN_PX, StepHist, over_floor, widest_of};
use crate::probe::{PROBE_KIND_NONE, PROBE_KIND_TERRAIN, decode_probe};

/// THE FRAMES that the design reads at a still stand: two whole jitter cycles of 32 phases at the
/// render scale 0.5 (design §4.3 and §11 T0 item 2). The meter reads any number of frames. A
/// capture that follows the design takes this many.
pub const STILL_FRAMES: usize = 64;

/// THE CHANNELS that a step reads: red, green and blue. The window shows these three. The alpha has
/// its own gate (G-ALPHA: every display pixel reads 255, design §7.7). The pop detector reads the
/// same three channels.
pub const COLOUR_CHANNELS: usize = 3;

/// THE EDGE'S REACH in pixels each way: 1 gives the design's 3 × 3 neighbourhood. The jitter moves
/// an edge by less than one pixel, so a band of this reach holds the edge in every phase. The
/// reconstruction dilates the motion over the same 3 × 3 (design §4.2).
pub const EDGE_NEIGHBOURHOOD_PX: usize = 1;

/// THE PROPOSED SHARE (design §12.4; the owner rules it): a region meets its bound when this share
/// of its counts, or more, lies within the bound. The pop detector's floor takes the same share of
/// the still pixels ([`crate::pop::FLOOR_SHARE`]).
pub const TARGET_SHARE: f64 = 0.999;

/// THE PROPOSED BOUND FOR THE CONTENT, in levels a frame (design §12.4; the owner rules it). It is
/// the one level of ruling V18, the tolerance of the exact pictures.
pub const CONTENT_TARGET_LEVELS: u8 = 1;

/// THE PROPOSED BOUND FOR THE EDGES, in levels a frame (design §12.4; the owner rules it). An edge
/// gets a wider bound, because the jitter still crawls on it after the history converges: about
/// 1/64 of the edge's contrast at the history's floor (design §4.2, an ESTIMATE).
pub const EDGE_TARGET_LEVELS: u8 = 2;

/// The number of regions.
pub const REGION_COUNT: usize = 6;

/// The bytes of one pixel in a picture and in a probe: RGBA8.
const PIXEL_BYTES: usize = 4;

/// THE REGIONS of a still stand. Each count goes into one region, by the claim order of the
/// module's text: the overlay (left out), the stars, the shadow edges, the edges, then the kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    /// What the probe calls nothing, off the edges: the sky, or a hole in the ground (the hole
    /// census judges the holes).
    Sky = 0,
    /// The terrain, off the edges.
    Ground = 1,
    /// A pixel whose 3 × 3 probe neighbourhood holds two kinds or two rungs, in either probe of
    /// the pair.
    Edges = 2,
    /// A pixel of the caller's star mask.
    Stars = 3,
    /// A pixel of the caller's shadow-edge mask.
    ShadowEdges = 4,
    /// Any other drawn kind, off the edges: the ruler today.
    Other = 5,
}

impl Region {
    /// Every region, in the order of a report and of [`FlickerRead::regions`].
    pub const ALL: [Region; REGION_COUNT] = [
        Region::Sky,
        Region::Ground,
        Region::Edges,
        Region::Stars,
        Region::ShadowEdges,
        Region::Other,
    ];

    /// THE CONTENT: every region except the edges. The design's content target applies to it.
    pub const CONTENT: [Region; REGION_COUNT - 1] = [
        Region::Sky,
        Region::Ground,
        Region::Stars,
        Region::ShadowEdges,
        Region::Other,
    ];

    /// The proposed bound of the region, in levels a frame: [`EDGE_TARGET_LEVELS`] for the edges,
    /// [`CONTENT_TARGET_LEVELS`] for the content.
    #[must_use]
    pub const fn target_levels(self) -> u8 {
        match self {
            Region::Edges => EDGE_TARGET_LEVELS,
            _ => CONTENT_TARGET_LEVELS,
        }
    }
}

/// THE STILL STAND: what stays the same in every frame of a still. It holds the pictures' size,
/// the caller's two masks and the overlay's rectangle. A mask holds one `bool` a pixel, row-major.
/// `None` means that the caller has no such mask, and its region stays empty.
#[derive(Clone, Copy, Debug)]
pub struct StillStand<'a> {
    /// The columns of each picture and probe.
    pub width: usize,
    /// The rows of each picture and probe.
    pub height: usize,
    /// The star mask: `true` on the pixels of a star (for example, from the star census).
    pub stars: Option<&'a [bool]>,
    /// The shadow-edge mask: `true` on the ground where the edge of a shadow falls.
    pub shadow_edges: Option<&'a [bool]>,
    /// THE OVERLAY'S RECTANGLE (left, top, right, bottom, in pixels): the stamp's `hud_rect_px`.
    /// The meter grows it by [`HUD_MARGIN_PX`], as the pop detector does. The text of the readouts
    /// changes from frame to frame, so the caller gives the union of the rectangles of all the
    /// frames. The meter judges nothing under the grown rectangle.
    pub hud: Option<[f64; 4]>,
}

impl StillStand<'_> {
    /// Whether a pair and the masks match the stand's size: a picture and a probe of four bytes a
    /// pixel for each frame, and one `bool` a pixel in each mask.
    fn fits(&self, before: &StillFrame<'_>, after: &StillFrame<'_>) -> bool {
        let pixels = self.width * self.height;
        let bytes = pixels * PIXEL_BYTES;
        (pixels > 0)
            & (before.rgba.len() == bytes)
            & (before.probe.len() == bytes)
            & (after.rgba.len() == bytes)
            & (after.probe.len() == bytes)
            & self.stars.is_none_or(|mask| mask.len() == pixels)
            & self.shadow_edges.is_none_or(|mask| mask.len() == pixels)
    }

    /// Whether a pixel lies under the overlay: its centre inside the rectangle grown by
    /// [`HUD_MARGIN_PX`] (the pop detector's rule).
    fn under_overlay(&self, x: usize, y: usize) -> bool {
        self.hud.is_some_and(|r| {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            (px >= r[0] - HUD_MARGIN_PX)
                & (py >= r[1] - HUD_MARGIN_PX)
                & (px <= r[2] + HUD_MARGIN_PX)
                & (py <= r[3] + HUD_MARGIN_PX)
        })
    }

    /// The region that claims a pixel in a pair with the probes `before` and `after`.
    fn region_of(&self, before: &[u8], after: &[u8], x: usize, y: usize) -> Region {
        let i = y * self.width + x;
        let centre = kind_rung(before, i * PIXEL_BYTES);
        if self.stars.is_some_and(|mask| mask[i]) {
            Region::Stars
        } else if self.shadow_edges.is_some_and(|mask| mask[i]) {
            Region::ShadowEdges
        } else if self.one_class_around(before, after, x, y, centre) {
            kind_region(centre.0)
        } else {
            Region::Edges
        }
    }

    /// Whether both probes name the kind and the rung `centre` at every pixel of the pixel's
    /// neighbourhood inside the frame (the pixel itself included).
    fn one_class_around(
        &self,
        before: &[u8],
        after: &[u8],
        x: usize,
        y: usize,
        centre: (u8, u8),
    ) -> bool {
        let x0 = x.saturating_sub(EDGE_NEIGHBOURHOOD_PX);
        let x1 = (x + EDGE_NEIGHBOURHOOD_PX).min(self.width - 1);
        let y0 = y.saturating_sub(EDGE_NEIGHBOURHOOD_PX);
        let y1 = (y + EDGE_NEIGHBOURHOOD_PX).min(self.height - 1);
        let mut one = true;
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                let o = (ny * self.width + nx) * PIXEL_BYTES;
                one &= (kind_rung(before, o) == centre) & (kind_rung(after, o) == centre);
            }
        }
        one
    }
}

/// ONE FRAME of a still: its picture and its probe, each RGBA8, row-major, four bytes a pixel, of
/// the stand's size. The probe must have the picture's size: at a render scale under one, the
/// caller scales the probe to the display picture first. A still with one probe gives that probe to
/// every frame. A still with one probe a frame gives each frame its own probe.
#[derive(Clone, Copy, Debug)]
pub struct StillFrame<'a> {
    /// The picture.
    pub rgba: &'a [u8],
    /// The probe (the codec of [`crate::probe`]).
    pub probe: &'a [u8],
}

/// WHAT ONE REGION READ: a histogram of its steps. `steps[s]` counts how often a pixel of the region
/// changed by `s` levels from one frame to the next. Each pixel counts once in each pair of frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionRead {
    /// The histogram of the steps, one bin a level.
    pub steps: StepHist,
}

impl Default for RegionRead {
    fn default() -> RegionRead {
        RegionRead {
            steps: Box::new([0; 256]),
        }
    }
}

impl RegionRead {
    /// The counts: one for each pixel of the region in each pair of frames.
    #[must_use]
    pub fn counts(&self) -> u64 {
        self.steps.iter().sum()
    }

    /// The share of the counts that changed by `levels` or less, or `None` for an empty region.
    #[must_use]
    pub fn share_within(&self, levels: u8) -> Option<f64> {
        let counts = self.counts();
        let within = counts - over_floor(&self.steps, levels);
        (counts > 0).then(|| within as f64 / counts as f64)
    }

    /// The largest step of the region (0 for an empty region).
    #[must_use]
    pub fn largest(&self) -> u8 {
        widest_of(&self.steps)
    }

    /// Whether the region meets a bound: [`TARGET_SHARE`] of its counts or more change by `levels`
    /// or less. `None` for an empty region: it has nothing to judge.
    #[must_use]
    pub fn meets(&self, levels: u8) -> Option<bool> {
        self.share_within(levels).map(|share| share >= TARGET_SHARE)
    }

    /// Pool another region's histogram into this one.
    fn add(&mut self, other: &RegionRead) {
        for (mine, theirs) in self.steps.iter_mut().zip(other.steps.iter()) {
            *mine += theirs;
        }
    }
}

/// WHAT A PAIR OF FRAMES READ, or the pairs of a still, summed. Every count is one pixel in one pair
/// of frames. The counts add and the histograms pool, so each number comes from the whole still.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FlickerRead {
    /// The pairs of frames in the reading.
    pub pairs: u64,
    /// The counts that the overlay left out.
    pub under_overlay: u64,
    /// The counts whose own probe pixel names another kind or rung in the second frame than in the
    /// first. The pair then reads the pixel as an edge, unless a mask claims it. At a still stand
    /// whose probe does not change, this count is 0.
    pub probe_changed: u64,
    /// The regions' readings, in the order of [`Region::ALL`] (the index is `Region as usize`).
    pub regions: [RegionRead; REGION_COUNT],
}

impl FlickerRead {
    /// The reading of one region.
    #[must_use]
    pub fn region(&self, region: Region) -> &RegionRead {
        &self.regions[region as usize]
    }

    /// THE PIXEL COUNT of a region: its counts over the pairs, rounded down. At a still stand whose
    /// probe does not change, every pair counts the same pixels, so the number is exact. It is 0 for
    /// a reading of no pair.
    #[must_use]
    pub fn pixels(&self, region: Region) -> u64 {
        self.region(region).counts() / self.pairs.max(1)
    }

    /// THE POOL of some regions: their histograms added into one. A gate uses it when the owner
    /// rules a group of regions other than [`Region::CONTENT`].
    #[must_use]
    pub fn pooled(&self, regions: &[Region]) -> RegionRead {
        let mut pool = RegionRead::default();
        for region in regions {
            pool.add(self.region(*region));
        }
        pool
    }

    /// THE CONTENT: the histograms of [`Region::CONTENT`] pooled into one.
    #[must_use]
    pub fn content(&self) -> RegionRead {
        self.pooled(&Region::CONTENT)
    }

    /// Whether a region meets its proposed target ([`Region::target_levels`] at [`TARGET_SHARE`]).
    /// `None` for an empty region.
    #[must_use]
    pub fn meets_target(&self, region: Region) -> Option<bool> {
        self.region(region).meets(region.target_levels())
    }

    /// Whether the pooled content meets the proposed content target ([`CONTENT_TARGET_LEVELS`] at
    /// [`TARGET_SHARE`]). `None` when the content is empty.
    #[must_use]
    pub fn content_meets_target(&self) -> Option<bool> {
        self.content().meets(CONTENT_TARGET_LEVELS)
    }

    /// Add another reading into this one.
    fn add(&mut self, other: &FlickerRead) {
        self.pairs += other.pairs;
        self.under_overlay += other.under_overlay;
        self.probe_changed += other.probe_changed;
        for (mine, theirs) in self.regions.iter_mut().zip(other.regions.iter()) {
            mine.add(theirs);
        }
    }
}

/// The kind and the rung that a probe states at the byte offset `o`.
fn kind_rung(probe: &[u8], o: usize) -> (u8, u8) {
    let p = decode_probe([probe[o], probe[o + 1], probe[o + 2]]);
    (p.kind, p.rung)
}

/// The region of a pixel off the edges, by the probe's kind.
fn kind_region(kind: u8) -> Region {
    match kind {
        PROBE_KIND_NONE => Region::Sky,
        PROBE_KIND_TERRAIN => Region::Ground,
        _ => Region::Other,
    }
}

/// THE STEP of the pixel at the byte offset `o`: the largest difference of the colour channels
/// between the two pictures, in levels.
fn step_at(before: &[u8], after: &[u8], o: usize) -> u8 {
    (0..COLOUR_CHANNELS).fold(0, |largest, k| {
        largest.max(before[o + k].abs_diff(after[o + k]))
    })
}

/// THE READING OF ONE PAIR of consecutive frames at a still stand: the step of each pixel goes into
/// the region that claims the pixel. A pair whose pictures, probes or masks do not match the stand's
/// size reads nothing (`pairs` stays 0).
#[must_use]
pub fn read_pair(
    stand: &StillStand<'_>,
    before: &StillFrame<'_>,
    after: &StillFrame<'_>,
) -> FlickerRead {
    if !stand.fits(before, after) {
        return FlickerRead::default();
    }
    let mut read = FlickerRead {
        pairs: 1,
        ..FlickerRead::default()
    };
    for y in 0..stand.height {
        for x in 0..stand.width {
            if stand.under_overlay(x, y) {
                read.under_overlay += 1;
                continue;
            }
            let o = (y * stand.width + x) * PIXEL_BYTES;
            let changed = kind_rung(before.probe, o) != kind_rung(after.probe, o);
            read.probe_changed += u64::from(changed);
            let region = stand.region_of(before.probe, after.probe, x, y);
            let step = usize::from(step_at(before.rgba, after.rgba, o));
            read.regions[region as usize].steps[step] += 1;
        }
    }
    read
}

/// THE FOLD OF A STILL: every pair of consecutive frames, read and summed. A still of fewer than two
/// frames reads nothing. A malformed pair adds nothing, so `pairs` then stays under the number of
/// frames less one.
#[must_use]
pub fn read_still(stand: &StillStand<'_>, frames: &[StillFrame<'_>]) -> FlickerRead {
    let mut total = FlickerRead::default();
    for (before, after) in frames.iter().zip(frames.iter().skip(1)) {
        total.add(&read_pair(stand, before, after));
    }
    total
}

/// The sum of many readings: the counts add and the histograms pool.
#[must_use]
pub fn sum_reads(reads: &[FlickerRead]) -> FlickerRead {
    let mut total = FlickerRead::default();
    for read in reads {
        total.add(read);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{PROBE_KIND_RULER, ProbePixel, encode_probe};

    /// The test pictures' size: 8 columns, 6 rows.
    const W: usize = 8;
    const H: usize = 6;

    /// One probe pixel: a kind at a rung, 100 cells away.
    fn at(kind: u8, rung: u8) -> [u8; 4] {
        let p = encode_probe(ProbePixel {
            kind,
            rung,
            cells: 100,
        });
        [p[0], p[1], p[2], 255]
    }

    /// One opaque grey picture pixel.
    fn grey(level: u8) -> [u8; 4] {
        [level, level, level, 255]
    }

    /// A W × H picture or probe, painted pixel by pixel.
    fn paint(rule: &dyn Fn(usize, usize) -> [u8; 4]) -> Vec<u8> {
        let mut out = Vec::with_capacity(W * H * PIXEL_BYTES);
        for y in 0..H {
            for x in 0..W {
                out.extend_from_slice(&rule(x, y));
            }
        }
        out
    }

    /// THE HORIZON: the sky in the rows 0–2, the ground at rung 2 in the rows 3–5. The rows 2 and 3
    /// see both kinds in their 3 × 3, so they are the edge. The rows 0–1 are the sky and the rows
    /// 4–5 the ground: 16 pixels in each region.
    fn horizon() -> Vec<u8> {
        paint(&|_, y| {
            if y < 3 {
                at(PROBE_KIND_NONE, 0)
            } else {
                at(PROBE_KIND_TERRAIN, 2)
            }
        })
    }

    /// The stand of the test pictures: no mask, no overlay.
    fn stand() -> StillStand<'static> {
        StillStand {
            width: W,
            height: H,
            stars: None,
            shadow_edges: None,
            hud: None,
        }
    }

    /// The frames of a still with one probe for all of them.
    fn frames<'a>(pictures: &[&'a [u8]], probe: &'a [u8]) -> Vec<StillFrame<'a>> {
        pictures
            .iter()
            .map(|&rgba| StillFrame { rgba, probe })
            .collect()
    }

    /// THE CONTROL: the same picture in each of the design's 64 frames. Every region reads all its
    /// counts within one level and a largest step of 0. The empty regions judge nothing.
    #[test]
    fn a_still_sequence_reads_every_count_within_one_level() {
        let probe = horizon();
        let picture = paint(&|x, y| grey((x * 10 + y * 20) as u8));
        let pictures = [picture.as_slice(); STILL_FRAMES];
        let read = read_still(&stand(), &frames(&pictures, &probe));
        assert_eq!(read.pairs, (STILL_FRAMES - 1) as u64);
        assert_eq!(Region::ALL.map(|r| read.pixels(r)), [16, 16, 16, 0, 0, 0]);
        for r in [Region::Sky, Region::Ground, Region::Edges] {
            assert_eq!(
                read.region(r).share_within(CONTENT_TARGET_LEVELS),
                Some(1.0)
            );
            assert_eq!(read.region(r).largest(), 0);
            assert_eq!(read.meets_target(r), Some(true));
        }
        for r in [Region::Stars, Region::ShadowEdges, Region::Other] {
            assert_eq!(read.region(r).counts(), 0);
            assert_eq!(read.meets_target(r), None);
        }
        assert_eq!(read.content().counts(), 32 * read.pairs);
        assert_eq!(read.content_meets_target(), Some(true));
        assert_eq!((read.under_overlay, read.probe_changed), (0, 0));
    }

    /// THE DEFECT, TWO LEVELS: the ground alternates by two levels from frame to frame. The meter
    /// counts every step in the ground, and the ground fails its one-level target. The sky and the
    /// edges stay still. The same flicker on the edges stays inside their two-level target, and a
    /// three-level flicker there does not.
    #[test]
    fn a_two_level_flicker_counts_in_its_region_and_fails_the_one_level_share() {
        let probe = horizon();
        let even = paint(&|_, _| grey(100));
        let odd = paint(&|_, y| grey(if y >= 4 { 102 } else { 100 }));
        let pictures: Vec<&[u8]> = (0..STILL_FRAMES)
            .map(|k| {
                if k.is_multiple_of(2) {
                    even.as_slice()
                } else {
                    odd.as_slice()
                }
            })
            .collect();
        let read = read_still(&stand(), &frames(&pictures, &probe));
        let ground = read.region(Region::Ground);
        assert_eq!(ground.counts(), 16 * read.pairs);
        assert_eq!((ground.steps[2], ground.largest()), (ground.counts(), 2));
        assert_eq!(ground.share_within(CONTENT_TARGET_LEVELS), Some(0.0));
        assert_eq!(ground.share_within(EDGE_TARGET_LEVELS), Some(1.0));
        assert_eq!(read.meets_target(Region::Ground), Some(false));
        assert_eq!(read.meets_target(Region::Sky), Some(true));
        assert_eq!(read.meets_target(Region::Edges), Some(true));
        assert_eq!(read.region(Region::Edges).largest(), 0);
        // Pooled, half of the content counts step by two levels: the content fails too.
        assert_eq!(
            read.content().share_within(CONTENT_TARGET_LEVELS),
            Some(0.5)
        );
        assert_eq!(read.content_meets_target(), Some(false));
        // A group that the owner can rule in its place, the drawn content alone: here it holds the
        // ground's own steps. A pool of no region judges nothing.
        let drawn = read.pooled(&[Region::Ground, Region::ShadowEdges, Region::Other]);
        assert_eq!(drawn, *ground);
        assert_eq!(read.pooled(&[]).meets(CONTENT_TARGET_LEVELS), None);

        // Two levels on the edge rows: inside the edges' target, outside the content's bound.
        let edge_odd = paint(&|_, y| grey(if (2..4).contains(&y) { 102 } else { 100 }));
        let seq: [&[u8]; 3] = [&even, &edge_odd, &even];
        let read = read_still(&stand(), &frames(&seq, &probe));
        let edges = read.region(Region::Edges);
        assert_eq!((edges.counts(), edges.largest()), (32, 2));
        assert_eq!(edges.meets(CONTENT_TARGET_LEVELS), Some(false));
        assert_eq!(read.meets_target(Region::Edges), Some(true));
        // Three levels on the edge rows fail the edges' own target.
        let edge_three = paint(&|_, y| grey(if (2..4).contains(&y) { 103 } else { 100 }));
        let seq: [&[u8]; 2] = [&even, &edge_three];
        let read = read_still(&stand(), &frames(&seq, &probe));
        assert_eq!(read.region(Region::Edges).largest(), 3);
        assert_eq!(read.meets_target(Region::Edges), Some(false));
    }

    /// THE DEFECT, ONE FRAME: the middle frame of the design's 64 is 30 levels darker in red alone.
    /// Every region's largest step reads 30, twice for each pixel: into the frame and out of it. A
    /// change of the alpha alone reads nothing, and a step of the blue alone reads in full.
    #[test]
    fn a_one_frame_step_shows_as_the_largest_step() {
        let probe = horizon();
        let still = paint(&|_, _| grey(100));
        let stepped = paint(&|_, _| [70, 100, 100, 255]);
        let mut pictures = [still.as_slice(); STILL_FRAMES];
        pictures[STILL_FRAMES / 2] = &stepped;
        let read = read_still(&stand(), &frames(&pictures, &probe));
        for r in [Region::Sky, Region::Ground, Region::Edges] {
            let region = read.region(r);
            assert_eq!((region.largest(), region.steps[30]), (30, 32));
            assert_eq!(region.steps[0], 16 * read.pairs - 32);
            assert_eq!(read.meets_target(r), Some(false));
        }
        let s = stand();
        let first = StillFrame {
            rgba: &still,
            probe: &probe,
        };
        let alpha = paint(&|_, _| [100, 100, 100, 55]);
        let read = read_pair(
            &s,
            &first,
            &StillFrame {
                rgba: &alpha,
                probe: &probe,
            },
        );
        assert_eq!(read.region(Region::Ground).largest(), 0);
        let blue = paint(&|_, _| [100, 100, 130, 255]);
        let read = read_pair(
            &s,
            &first,
            &StillFrame {
                rgba: &blue,
                probe: &probe,
            },
        );
        assert_eq!(read.region(Region::Ground).largest(), 30);
    }

    /// THE CLAIMS: a mask claims its pixels before the probe's classes, and the star mask claims
    /// them before the shadow-edge mask. A star that brightens counts in the stars, never in the sky
    /// around it.
    #[test]
    fn a_mask_claims_its_pixels_before_the_probe_and_the_stars_before_the_shadows() {
        let probe = horizon();
        let index = |x: usize, y: usize| y * W + x;
        let mut stars = [false; W * H];
        // A star in the sky, one on the edge rows and one on the ground.
        for (x, y) in [(1, 0), (1, 2), (1, 4)] {
            stars[index(x, y)] = true;
        }
        let mut shadows = [false; W * H];
        // A shadow's edge on the ground star (the star wins), on the ground and on the edge rows.
        for (x, y) in [(1, 4), (5, 4), (5, 3)] {
            shadows[index(x, y)] = true;
        }
        let s = StillStand {
            stars: Some(&stars),
            shadow_edges: Some(&shadows),
            ..stand()
        };
        let still = paint(&|_, _| grey(100));
        // The star in the sky brightens by five levels in the second frame.
        let twinkle = paint(&|x, y| grey(if (x, y) == (1, 0) { 105 } else { 100 }));
        let read = read_pair(
            &s,
            &StillFrame {
                rgba: &still,
                probe: &probe,
            },
            &StillFrame {
                rgba: &twinkle,
                probe: &probe,
            },
        );
        // The sky, the ground, the edges, the stars, the shadow edges, the other things.
        assert_eq!(Region::ALL.map(|r| read.pixels(r)), [15, 14, 14, 3, 2, 0]);
        assert_eq!(read.region(Region::Stars).largest(), 5);
        assert_eq!(read.region(Region::Sky).largest(), 0);
    }

    /// THE OVERLAY: a readout under the stamp's rectangle changes by 50 levels, and the meter leaves
    /// out every pixel under the rectangle grown by the glyphs' margin. Without the rectangle the
    /// readout reads as a step of 50.
    #[test]
    fn the_overlay_is_left_out() {
        let probe = horizon();
        let still = paint(&|_, _| grey(100));
        let readout = paint(&|x, y| grey(if (x < 2) & (y < 1) { 50 } else { 100 }));
        let before = StillFrame {
            rgba: &still,
            probe: &probe,
        };
        let after = StillFrame {
            rgba: &readout,
            probe: &probe,
        };
        let corner = StillStand {
            hud: Some([0.0, 0.0, 1.0, 0.0]),
            ..stand()
        };
        let read = read_pair(&corner, &before, &after);
        // Grown by two pixels, the rectangle covers the columns 0–2 of the rows 0–1: six pixels.
        assert_eq!(read.under_overlay, 6);
        assert_eq!(read.pixels(Region::Sky), 10);
        assert_eq!(read.region(Region::Sky).largest(), 0);
        let bare = read_pair(&stand(), &before, &after);
        assert_eq!(bare.under_overlay, 0);
        assert_eq!(bare.region(Region::Sky).largest(), 50);
        // A rectangle in the middle: the columns 2–5 of the rows 1–4 have their centres within two
        // pixels of it, on all four sides.
        let middle = StillStand {
            hud: Some([4.0, 3.0, 4.0, 3.0]),
            ..stand()
        };
        assert_eq!(read_pair(&middle, &before, &after).under_overlay, 16);
    }

    /// A PROBE A FRAME: the middle probe of three names another rung under one ground pixel and the
    /// ground under one sky pixel. Each pair counts both pixels as changed and reads the 3 × 3 around
    /// each of them as an edge. The same pictures with the one steady probe read the plain horizon.
    #[test]
    fn a_probe_that_changes_between_frames_makes_its_pixels_edges() {
        let steady = horizon();
        let changed = paint(&|x, y| match (x, y) {
            (5, 5) => at(PROBE_KIND_TERRAIN, 3),
            (2, 0) => at(PROBE_KIND_TERRAIN, 2),
            _ if y < 3 => at(PROBE_KIND_NONE, 0),
            _ => at(PROBE_KIND_TERRAIN, 2),
        });
        let picture = paint(&|_, _| grey(100));
        let seq = [
            StillFrame {
                rgba: &picture,
                probe: &steady,
            },
            StillFrame {
                rgba: &picture,
                probe: &changed,
            },
            StillFrame {
                rgba: &picture,
                probe: &steady,
            },
        ];
        let read = read_still(&stand(), &seq);
        assert_eq!(read.pairs, 2);
        assert_eq!(read.probe_changed, 4);
        // The six ground pixels around (5, 5) and the six sky pixels around (2, 0) join the edges.
        assert_eq!(Region::ALL.map(|r| read.pixels(r)), [10, 10, 28, 0, 0, 0]);
        let one_probe = read_still(&stand(), &frames(&[picture.as_slice(); 3], &steady));
        assert_eq!(
            Region::ALL.map(|r| one_probe.pixels(r)),
            [16, 16, 16, 0, 0, 0]
        );
        assert_eq!(one_probe.probe_changed, 0);
    }

    /// THE KINDS AND THE RUNGS: a ruler ball in the corner reads as another drawn thing where its
    /// 3 × 3 (trimmed by the frame) holds only the ruler, and its rim reads as an edge. Two rungs side
    /// by side make an edge of the two columns where they meet.
    #[test]
    fn another_kind_and_another_rung_read_their_own_regions() {
        let picture = paint(&|_, _| grey(100));
        let seq: [&[u8]; 2] = [&picture, &picture];
        let ruler = paint(&|x, y| {
            if (x < 3) & (y < 3) {
                at(PROBE_KIND_RULER, 2)
            } else {
                at(PROBE_KIND_TERRAIN, 2)
            }
        });
        let read = read_still(&stand(), &frames(&seq, &ruler));
        // The inner 2 × 2 of the ruler is another drawn thing. The ruler's rim (5 pixels) and the
        // ground around it (7 pixels) are the edge.
        assert_eq!(Region::ALL.map(|r| read.pixels(r)), [0, 32, 12, 0, 0, 4]);
        assert_eq!(read.meets_target(Region::Other), Some(true));
        let rungs = paint(&|x, _| at(PROBE_KIND_TERRAIN, if x < 4 { 1 } else { 2 }));
        let read = read_still(&stand(), &frames(&seq, &rungs));
        assert_eq!(Region::ALL.map(|r| read.pixels(r)), [0, 36, 12, 0, 0, 0]);
    }

    /// A MALFORMED PAIR reads nothing: a picture, a probe or a mask of the wrong size, or a stand of
    /// no pixel. In a still such a pair adds nothing.
    #[test]
    fn a_malformed_pair_reads_nothing() {
        let probe = horizon();
        let picture = paint(&|_, _| grey(100));
        let good = StillFrame {
            rgba: &picture,
            probe: &probe,
        };
        let short_picture = StillFrame {
            rgba: &picture[..8],
            probe: &probe,
        };
        let short_probe = StillFrame {
            rgba: &picture,
            probe: &probe[..8],
        };
        let s = stand();
        assert_eq!(read_pair(&s, &good, &good).pairs, 1);
        for (before, after) in [
            (short_picture, good),
            (good, short_picture),
            (short_probe, good),
            (good, short_probe),
        ] {
            assert_eq!(read_pair(&s, &before, &after), FlickerRead::default());
        }
        let nothing = StillFrame {
            rgba: &[],
            probe: &[],
        };
        let empty = StillStand {
            width: 0,
            height: 0,
            ..stand()
        };
        assert_eq!(
            read_pair(&empty, &nothing, &nothing),
            FlickerRead::default()
        );
        let wrong_mask = [true; 3];
        let stars = StillStand {
            stars: Some(&wrong_mask),
            ..stand()
        };
        assert_eq!(read_pair(&stars, &good, &good), FlickerRead::default());
        let shadows = StillStand {
            shadow_edges: Some(&wrong_mask),
            ..stand()
        };
        assert_eq!(read_pair(&shadows, &good, &good), FlickerRead::default());
        assert_eq!(read_still(&s, &[good, short_picture, good]).pairs, 0);
        assert_eq!(read_still(&s, &[good, good, short_picture]).pairs, 1);
    }

    /// THE SUM: the pairs of a still pool into the same reading as the fold of its frames. A still
    /// of no frame or of one frame, and a sum of no reading, read nothing.
    #[test]
    fn the_sum_pools_the_pairs_and_matches_the_fold() {
        let probe = horizon();
        let even = paint(&|_, _| grey(100));
        let odd = paint(&|_, y| grey(if y >= 4 { 102 } else { 100 }));
        let s = stand();
        let first = StillFrame {
            rgba: &even,
            probe: &probe,
        };
        let second = StillFrame {
            rgba: &odd,
            probe: &probe,
        };
        let there = read_pair(&s, &first, &second);
        let back = read_pair(&s, &second, &first);
        let sum = sum_reads(&[there.clone(), back]);
        assert_eq!(sum, read_still(&s, &[first, second, first]));
        assert_eq!(sum.pairs, 2);
        assert_eq!(sum.region(Region::Ground).steps[2], 32);
        assert_eq!(sum.pixels(Region::Ground), there.pixels(Region::Ground));
        assert_eq!(sum_reads(&[]), FlickerRead::default());
        assert_eq!(read_still(&s, &[]), FlickerRead::default());
        assert_eq!(read_still(&s, &[first]), FlickerRead::default());
        assert_eq!(FlickerRead::default().pixels(Region::Sky), 0);
        assert_eq!(FlickerRead::default().content_meets_target(), None);
    }

    /// THE SHARE AND THE BOUND read a histogram: exactly the target share meets the bound, one
    /// count less does not, and an empty region judges nothing.
    #[test]
    fn the_share_and_the_bound_read_a_histogram() {
        let mut region = RegionRead::default();
        assert_eq!(
            (region.counts(), region.largest(), region.share_within(1)),
            (0, 0, None)
        );
        assert_eq!(region.meets(CONTENT_TARGET_LEVELS), None);
        region.steps[0] = 998;
        region.steps[2] = 1;
        region.steps[7] = 1;
        assert_eq!(region.counts(), 1000);
        assert_eq!(region.largest(), 7);
        assert_eq!(region.share_within(1), Some(0.998));
        assert_eq!(region.share_within(2), Some(0.999));
        assert_eq!(region.share_within(u8::MAX), Some(1.0));
        assert_eq!(region.meets(1), Some(false));
        assert_eq!(region.meets(2), Some(true));
    }

    /// THE TARGETS: the edges take the two-level bound and every content region the one-level
    /// bound; the lists hold every region once, in the order of the index.
    #[test]
    fn the_edges_take_the_two_level_bound_and_the_content_the_one_level_bound() {
        assert_eq!(Region::Edges.target_levels(), EDGE_TARGET_LEVELS);
        for region in Region::CONTENT {
            assert_eq!(region.target_levels(), CONTENT_TARGET_LEVELS);
        }
        assert!(!Region::CONTENT.contains(&Region::Edges));
        for (k, region) in Region::ALL.iter().enumerate() {
            assert_eq!(*region as usize, k);
        }
    }
}
