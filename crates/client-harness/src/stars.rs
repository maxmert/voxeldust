//! ★ THE STAR CENSUS (G-STARS; the temporal path, slice T0 item 2; the design's §11 and §12.3–12.4)
//! — the pure half.
//!
//! The temporal path draws the stars after the reconstruction, in the points pass (§7.7). The core
//! of a star is 0.6 display pixels wide. A jittered history can hold such a point in one frame and
//! lose it in the next frame, and then the sky blinks and thins (the design's risk 7). This meter
//! compares the stars of a candidate run with the stars of TODAY'S picture, frame by frame. So a
//! star that blinks, a star that goes out and a sky that dims each show as a number.
//!
//! **A star.** A star is a small bright blob on the sky. The probe tells which pixels are sky (kind
//! [`PROBE_KIND_NONE`]); the ground, the ruler and the overlay's readouts are never sky. The meter
//! reads the luma of each sky pixel ([`luma_level`]) and its local sky background: the median luma
//! of the sky pixels in a window around the pixel ([`BACKGROUND_WINDOW_RADIUS_PX`]). A pixel is a
//! star pixel when its luma stands [`STAR_THRESHOLD_LEVELS`] or more over that background. Star
//! pixels that touch, also at a corner, make one blob. ★ A blob splits at its peaks: a peak that
//! stands [`PEAK_SPLIT_LEVELS`] or more over its pass (the lowest pixel on the best path to a
//! brighter peak) is its own star, so two stars whose wings touch stay two stars. A blob with one
//! peak is one part. A part is a star when three rules hold:
//! - its area is [`STAR_MAX_AREA_PX`] or less, so the sun disc in the owner's window is not a star;
//! - its white core is [`STAR_MAX_WHITE_PX`] or less, so a small sun disc is not a star;
//! - its flat top is [`STAR_MAX_FLAT_PX`] or less, so a lit moon is not a star. The flat top is the
//!   set of pixels under the top level whose light is [`FLAT_TOP_SHARE`] of the part's peak light
//!   or more.
//!
//! For each star the meter records the centroid, the peak, the area and the light. The light is the
//! sum of the star pixels' linear light over their background ([`linear_light`]), so it adds up as
//! light adds up on the screen.
//!
//! **The reference.** The census judges a candidate run against TODAY'S picture: today's renderer
//! at the same stand, at the same ticks and through the same cameras (the "before" row, M-6). The
//! census reads each reference frame once ([`StarReference::of`]). The reference stars of a frame
//! are its stars that peak [`CENSUS_PEAK_LEVELS`] or more over their background and stand on open
//! sky, [`STAR_MAX_RADIUS_PX`] or more inside the frame, and as far from every other star of the
//! frame: a crowded star is no reference star, because two runs that turn place it on two
//! sub-pixel places, and close peaks then split and merge differently ([`crowded`]).
//!
//! **One shape for a still stand and a turn.** [`StarCensus::add`] reads one candidate frame
//! against one reference frame. So the caller reads a run one frame at a time, and the 64 display
//! pictures of a run do not have to stay in memory at the same time:
//! - at a still stand, one reference frame serves every candidate frame;
//! - in a turn, reference frame k serves candidate frame k (today's run at the same ticks);
//! - ★ in a turn of two runs, today's frame of the NEAREST CAMERA serves each candidate frame
//!   ([`nearest_frame`]): two runs cannot turn through the same cameras, because the pilot's look
//!   reaches the server at its own ticks.
//!
//! [`star_census`] reads whole runs in the first two shapes.
//!
//! **The check of a frame.** The census puts each reference star at its place in the candidate
//! frame. A star is at infinity, so the census turns the star's DIRECTION from the reference camera
//! to the candidate camera and projects it. The two cameras are the same, so the place does not
//! move, and a small difference of the stamped cameras moves no star. A star of the candidate
//! within [`MATCH_RADIUS_PX`] of the place answers for the reference star. One star of the
//! candidate answers for one reference star only, the nearest pair first. So each reference star
//! gets one sighting: present; missing (the candidate shows open sky and no star); hidden (the
//! candidate's ground covers the place); under the overlay; or out of view. A star out of view is NO
//! CHECK: the candidate's camera does not look at it, so the frame says nothing of it. Through one
//! camera that never happens; through today's frame of the nearest camera in a turn, a star at the
//! frame's edge stands in one view and out of the other. A star under the candidate's overlay, or
//! within [`STAR_MAX_RADIUS_PX`] of it, is no check either: the overlay is no part of the world's
//! picture, its readouts change its rectangle from frame to frame, and it cuts the light of a star
//! at its edge. A frame that looks at none of its reference's
//! stars is refused, so a wrong camera never passes by its silence. The EXTRA stars of the
//! candidate are its stars that peak [`CENSUS_PEAK_LEVELS`] or more on open sky and answer for no
//! star of the reference: for example a new blink, or a smear that splits a star.
//!
//! **The light.** In each frame the census adds up the light of the reference stars that the
//! candidate shows, once in the reference frame and once in the candidate frame. The change of the
//! frame is (candidate − reference) ÷ reference. The sum weighs each star by its light, so a bright
//! star carries more of it than a faint star. The census reads no median of the stars' own changes:
//! the 8-bit rounding hides a small change of a faint star. Example: a star pixel at level 20 over
//! the dusk sky keeps its level when its light drops by 3 %.
//!
//! **The verdict (§12.4; the owner rules the targets).** [`StarCensus::verdict`] passes a run when
//! each reference star is present in each candidate frame, and the light of each frame changes by
//! [`LIGHT_CHANGE_TARGET`] or less, over [`CENSUS_FRAMES`] frames. A verdict on missing data is
//! never a pass. So the verdict refuses a run with an unreadable frame, a run of fewer frames, a
//! run with a frame that looks at none of its reference's stars, a run with no reference star, and
//! a run read against its own frame 0. The extra stars do not
//! change the verdict, because §12.4 names only the lost stars and the light. The census reports
//! them.
//!
//! **The run's own reading.** [`StarCensus::add_own`] reads a run against the run's own frame 0:
//! the first frame names the stars, and a camera that turns carries them into each later frame. T0
//! reads today's picture so (item 2: the stars of each frame and their light, still and turning),
//! and it learns whether today's sky blinks by itself. This reading is NOT the verdict. A run that
//! loses a star in every frame lacks it in frame 0 too, so it reads steady here. The verdict
//! refuses it.
//!
//! Example: the pilot stands on foot on the home planet at the dusk stand and looks at the belt's
//! ridge. Today's picture shows 212 stars over the ridge, and 180 of them are reference stars. A
//! moon over the ridge (54 px at level 220) is not a star: its flat top holds all 54 pixels. The
//! candidate run shows 211 stars in frame 37: the star 3 px over the ridge does not show. The
//! census states one missing event (frame 37, star 41) and fails the run. A candidate run that
//! loses that star in all 64 frames gets 64 missing events and fails too, because its reference is
//! today's picture and not its own frame 0.

use std::collections::BTreeMap;

use vd_core::glam::DVec3;

use crate::camera::{CaptureCamera, ScreenPos};
use crate::pop::HUD_MARGIN_PX;
use crate::probe::{PROBE_KIND_NONE, decode_probe};

/// The frames of one census run. The design's instruments each read 64 frames at a stand (T0
/// item 2).
pub const CENSUS_FRAMES: usize = 64;

/// How far over its local sky background, in 8-bit luma levels, a pixel must stand to be a star
/// pixel. The basis: a reference star peaks [`CENSUS_PEAK_LEVELS`] or more over its background.
/// When the star stands on the corner of a pixel, the star law's core (σ 0.6 px) gives each of the
/// four pixels around it half its peak: exp(−(0.5² + 0.5²) ÷ (2 · 0.6²)) = 0.50. Half of 10 levels,
/// less half a level of 8-bit rounding, is more than 4. So a reference star stays over the
/// threshold at every sub-pixel place in a view that turns. And 4 levels stand over the still sky's
/// own noise: the 8-bit rounding and the tonemapper's dither move a sky pixel by about one level.
pub const STAR_THRESHOLD_LEVELS: u8 = 4;

/// ★ A SECOND PEAK IS A SECOND STAR (2026-10-01, G-STARS at the solved dusk stand): inside one blob,
/// a peak is its own star when it stands this many levels or more over its PASS — the lowest pixel
/// on the best path that joins it to a brighter peak. It is the rule of a star pixel
/// ([`STAR_THRESHOLD_LEVELS`]), read with the pass as the background: a peak with a rise of 4 levels
/// over the pass makes its own blob over a sky at the pass's level. A smaller rise is a bump of the
/// brighter star, for example the tonemapper's dither on a wing (about one level). MEASURED first, in
/// the turning census of the dusk stand (one frame, 450 reference stars): the matched stars stood at
/// a median offset of (0.00, 0.00) px from their predicted places, so the cameras are exact; 41 of
/// the 44 lost stars had a candidate star 1.5–5 px away, scattered in every direction (the mean
/// direction's length 0.03–0.18 over four frames), where a random place of that sky has a star
/// within 6 px about one time in twelve. Two stars whose wings touch at the threshold made one blob
/// in one picture and two in the other, and the one blob's centroid answered for one of them only.
/// Example: at the dusk stand a star of peak 107 hangs 4 px under a star of peak 136 over a sky of
/// 18. The temporal path's softer wing joins them through two pixels at 22 and 23, so one blob holds
/// both, and its pass is at 22. The lower star stands 85 levels over its pass, so it stays its own
/// star, at its own place, as in today's picture.
pub const PEAK_SPLIT_LEVELS: u8 = STAR_THRESHOLD_LEVELS;

/// How far a star must peak over its background, in 8-bit luma levels, to be a reference star (and,
/// in a candidate frame, an extra star). The basis: the dimmest star that the star law draws on a
/// black sky. The law culls a star at 0.003 of full scale in linear light
/// (`vd_client::realm_scene::StarTuning::cull_level`), and the straight toe of the sRGB curve
/// encodes that to 12.92 × 0.003 × 255 = 9.9 levels.
pub const CENSUS_PEAK_LEVELS: u8 = 10;

/// The half-width of the window whose median is a pixel's sky background, in pixels: a window of 31
/// × 31 pixels. The basis: the largest star ([`STAR_MAX_AREA_PX`]) covers about a quarter of it, so
/// the median stays on the sky. A window with the pixel at its centre also removes a smooth
/// gradient of the sky (a straight gradient exactly), for example the glow over the horizon at
/// dusk.
pub const BACKGROUND_WINDOW_RADIUS_PX: usize = 15;

/// The radius of the widest star, in pixels. The basis: the star law crops the sprite of a star
/// where its halo fades under the cull, r = σ_halo · √(2 · ln(A · w_halo ÷ cull)). The radius grows
/// only with the root of the logarithm of the amplitude A: 4.9 px for the nearest neighbour (A ≈
/// 11) and 9.08 px for A = 1 000 000. A reference star also stands this far inside the frame, so
/// the frame's edge cuts none of its light, and this far from every other star of its frame, so no
/// other star's light falls on its place ([`StarReference::crowded`]).
pub const STAR_MAX_RADIUS_PX: f64 = 9.0;

/// The largest area of a star, in pixels: a disc of [`STAR_MAX_RADIUS_PX`] (π · 9² = 254.5). A
/// larger blob is not a star. Examples: the sun disc in the owner's full-screen window (one sun at
/// one AU is 0.565° wide: 27 px at 2 168 rows over 45°, about 580 px) and a lit moon.
pub const STAR_MAX_AREA_PX: u64 = 254;

/// The largest white core of a star, in pixels: the pixels at the top luma level. The basis: the
/// star law keeps the white core of its brightest star inside a radius of 3 px (its own test
/// `a_near_bright_star_is_not_a_white_ball`), a disc of π · 3² = 28.3 px. The sun disc draws far
/// over full scale, so it is white over its face (UNMEASURED through today's tonemapper). At the
/// capture's reference view (45° over 720 rows) one sun at one AU is 9 px wide: 64 white pixels. So
/// a small sun disc is not a star, also where its area is the area of a star.
pub const STAR_MAX_WHITE_PX: u64 = 28;

/// The share of a blob's peak light that marks its flat top: one half. The light of each pixel is
/// its linear light over its background. The basis: this share splits the core of a star from a
/// disc with a wide margin on both sides.
/// - The star law's core falls steeply. When a star stands on the corner of a pixel, the four
///   pixels around it get the same light, and the next pixels get less than a tenth of it (0.094
///   with the halo). So a star that no channel clips has a flat top of 4 pixels or fewer.
/// - A lit moon falls slowly. A flat disc keeps all its pixels at its peak. A lit sphere at full
///   phase keeps three quarters of its disc over half its peak: √(1 − ρ²) ≥ ½ where ρ² ≤ ¾.
pub const FLAT_TOP_SHARE: f64 = 0.5;

/// The largest flat top of a star, in pixels: the pixels under the top level whose light is
/// [`FLAT_TOP_SHARE`] of the blob's peak light or more. A blob with a wider flat top is not a star.
/// The basis: the widest flat top of a star that the census keeps (a white core of
/// [`STAR_MAX_WHITE_PX`] or less), in the star law's model. That model is the law's two Gaussians
/// (`StarTuning`) in each class colour (`MARKER_CLASS_SRGB`), each channel clipped at full scale
/// ("above 1.0 the core clips"), over the dusk sky. A bright star clips its core to white, and its
/// flat top is the ring where its light falls from full scale to half of it. The reddest class (M)
/// has the widest ring: its blue channel clips last, so its luma stays near the top longest. The
/// test `the_flat_top_bound_follows_the_star_law` sweeps the seven classes, the amplitudes and the
/// sub-pixel places. The widest flat top is 28 pixels: an M star on a pixel's corner at an
/// amplitude of about 256 (a white core of 16 pixels) and again at about 380 (a white core of 24
/// pixels). A brighter M star has a white core of more than 28 pixels. A lit moon has a wider flat
/// top. Example: a moon 0.52° wide at the capture's reference view (16 px a degree) covers about 54
/// pixels. A flat disc of it at level 220 holds all 54 in its flat top, and a lit sphere of 57
/// pixels holds 45. UNMEASURED through today's tonemapper: the law's model clips each channel; the
/// tonemapper bends the top.
pub const STAR_MAX_FLAT_PX: u64 = 28;

/// How far a star may stand from its predicted place in a candidate frame and still answer for it,
/// in pixels. The basis: the pixel model puts a turned star within one pixel of the drawn one (the
/// ruler gate holds the model to the drawn camera within one pixel, [`crate::probe`]), and the
/// centroid of a star moves by less than half a pixel with its sub-pixel place (the threshold cuts
/// its edge pixels).
pub const MATCH_RADIUS_PX: f64 = 1.5;

/// The target for the light (§12.4: the summed light within 1 %; the owner rules it). In each
/// frame, the candidate's light of the reference stars that it shows differs from the reference's
/// light of the same stars by this share or less. §7.7 names its first cause: the air's
/// transmittance comes from the table, not from the pixel's march.
pub const LIGHT_CHANGE_TARGET: f64 = 0.01;

/// The luma weights of the red, green and blue bytes, per ten thousand (ITU-R BT.709): how much of
/// each channel the eye sees as brightness.
pub const LUMA_WEIGHTS: [u32; 3] = [2_126, 7_152, 722];

/// The sum of [`LUMA_WEIGHTS`].
const LUMA_SCALE: u32 = 10_000;

/// The sRGB curve (IEC 61966-2-1): the end of its straight toe, the slope of the toe, the offset of
/// the curve and its exponent.
const SRGB_TOE: f64 = 0.040_45;
const SRGB_TOE_SLOPE: f64 = 12.92;
const SRGB_OFFSET: f64 = 0.055;
const SRGB_EXPONENT: f64 = 2.4;

/// THE LUMA of an 8-bit pixel: the [`LUMA_WEIGHTS`] on its bytes, rounded to a level (0..=255).
#[must_use]
pub fn luma_level(rgb: [u8; 3]) -> u8 {
    let sum = LUMA_WEIGHTS[0] * u32::from(rgb[0])
        + LUMA_WEIGHTS[1] * u32::from(rgb[1])
        + LUMA_WEIGHTS[2] * u32::from(rgb[2]);
    ((sum + LUMA_SCALE / 2) / LUMA_SCALE) as u8
}

/// THE LINEAR LIGHT of an 8-bit level (0 black, 1 full scale): the sRGB curve decoded. Light adds
/// up in this unit, so a star's light is the same when it covers one pixel or spreads over four.
#[must_use]
pub fn linear_light(level: u8) -> f64 {
    let c = f64::from(level) / 255.0;
    if c <= SRGB_TOE {
        c / SRGB_TOE_SLOPE
    } else {
        ((c + SRGB_OFFSET) / (1.0 + SRGB_OFFSET)).powf(SRGB_EXPONENT)
    }
}

/// The linear light of every level, for the per-pixel sums.
fn linear_table() -> [f64; 256] {
    let mut table = [0.0; 256];
    for (level, slot) in table.iter_mut().enumerate() {
        *slot = linear_light(level as u8);
    }
    table
}

/// One frame of a run.
#[derive(Clone, Copy)]
pub struct StarFrame<'a> {
    /// The picture: RGBA, eight bits a channel, row after row, `width × height` pixels.
    pub rgba: &'a [u8],
    pub width: usize,
    pub height: usize,
    /// The probe (RGBA, `probe_width × probe_height` pixels). At the render scale 1 it has the size
    /// of the picture. Under the up-scale it has the size of the render picture (§12.1), and each
    /// picture pixel then reads the probe pixel under its centre.
    pub probe: &'a [u8],
    pub probe_width: usize,
    pub probe_height: usize,
    /// The overlay's rectangle (left, top, right, bottom, pixels; the stamp's `hud_rect_px`), grown
    /// by [`HUD_MARGIN_PX`]. The pixels under it are not sky: its readouts change from frame to
    /// frame, and they are not stars.
    pub hud: Option<[f64; 4]>,
    /// The camera that drew the frame, with the picture's size as its viewport (for example
    /// [`crate::pop::frame_camera`] from the stamp). A camera with another viewport makes the frame
    /// unreadable: its places would miss the picture's stars. `None` at a still stand: the stars
    /// keep their places. The census turns a star from one camera to the other only when both
    /// frames give one.
    pub camera: Option<CaptureCamera>,
}

impl StarFrame<'_> {
    /// Whether the frame's bytes are one picture and one probe of their stated sizes, and its
    /// camera (when it gives one) has the picture's size as its viewport. The census reads nothing
    /// of a frame that fails this check: its probe could be too short to index.
    fn well_formed(&self) -> bool {
        let n = self.width * self.height;
        let probe_n = self.probe_width * self.probe_height;
        let viewport = self
            .camera
            .is_none_or(|c| (c.width == self.width) & (c.height == self.height));
        (n > 0)
            & (probe_n > 0)
            & (self.rgba.len() == n * 4)
            & (self.probe.len() == probe_n * 4)
            & viewport
    }

    /// Whether the picture pixel `(x, y)` is sky: the probe pixel under its centre shows nothing,
    /// and the overlay does not cover it.
    fn is_sky(&self, x: usize, y: usize) -> bool {
        let px = (2 * x + 1) * self.probe_width / (2 * self.width);
        let py = (2 * y + 1) * self.probe_height / (2 * self.height);
        let j = (py * self.probe_width + px) * 4;
        let kind = decode_probe([self.probe[j], self.probe[j + 1], self.probe[j + 2]]).kind;
        (kind == PROBE_KIND_NONE) & !self.under_overlay(x, y)
    }

    /// ★ Whether the place `(x, y)` stands within `reach` of the overlay's rectangle grown by
    /// [`HUD_MARGIN_PX`]: a star there has pixels under the overlay, which cuts its light.
    fn near_overlay(&self, x: f64, y: f64, reach: f64) -> bool {
        self.hud.is_some_and(|r| {
            let m = HUD_MARGIN_PX + reach;
            (x >= r[0] - m) & (y >= r[1] - m) & (x <= r[2] + m) & (y <= r[3] + m)
        })
    }

    /// Whether the overlay's rectangle, grown by [`HUD_MARGIN_PX`], covers the pixel `(x, y)`.
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

/// One star of a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Star {
    /// The centroid, in pixels from the frame's top-left corner, weighted by each pixel's light.
    pub x: f64,
    pub y: f64,
    /// The luma of the brightest pixel.
    pub peak: u8,
    /// The sky background under the brightest pixel.
    pub background: u8,
    /// The sum of the star pixels' linear light over their background.
    pub light: f64,
    /// The number of star pixels.
    pub area: u64,
}

/// What one frame shows: its stars, and the blobs that are not stars.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameStars {
    /// The stars, blob after blob in the order of each blob's first pixel (row after row); the stars
    /// of one blob in the order of their own first pixel.
    pub stars: Vec<Star>,
    /// The parts of blobs ([`PEAK_SPLIT_LEVELS`]) larger than [`STAR_MAX_AREA_PX`]: for example the
    /// sun disc in a large picture, a moon.
    pub too_large: u64,
    /// The other parts with a white core larger than [`STAR_MAX_WHITE_PX`]: for example the sun
    /// disc in a small picture.
    pub too_white: u64,
    /// The other parts with a flat top larger than [`STAR_MAX_FLAT_PX`]: for example a lit moon
    /// that is not white.
    pub too_flat: u64,
    /// The sky pixels of the frame. A frame with no sky shows no star and proves nothing.
    pub sky: u64,
}

impl FrameStars {
    /// The summed light of the frame's stars: the "before" row's light of a frame (T0 item 2).
    #[must_use]
    pub fn light(&self) -> f64 {
        self.stars.iter().map(|s| s.light).sum()
    }
}

/// A window of sky levels as a histogram, with its lower median kept up to date while the window
/// slides (Huang's running median).
struct Window {
    hist: [u32; 256],
    count: u32,
    /// The level of the median at the last read.
    at: usize,
    /// The levels in the window under `at`.
    below: u32,
}

impl Window {
    fn new() -> Window {
        Window {
            hist: [0; 256],
            count: 0,
            at: 0,
            below: 0,
        }
    }

    fn add(&mut self, level: u8) {
        let v = usize::from(level);
        self.hist[v] += 1;
        self.count += 1;
        self.below += u32::from(v < self.at);
    }

    fn remove(&mut self, level: u8) {
        let v = usize::from(level);
        self.hist[v] -= 1;
        self.count -= 1;
        self.below -= u32::from(v < self.at);
    }

    /// Adds the sky levels of column `x` in the rows `rows` (both ends included).
    fn enter(&mut self, level: &[u8], sky: &[bool], w: usize, x: usize, rows: (usize, usize)) {
        for y in rows.0..=rows.1 {
            let i = y * w + x;
            if sky[i] {
                self.add(level[i]);
            }
        }
    }

    /// Removes the sky levels of column `x` in the rows `rows` (both ends included).
    fn leave(&mut self, level: &[u8], sky: &[bool], w: usize, x: usize, rows: (usize, usize)) {
        for y in rows.0..=rows.1 {
            let i = y * w + x;
            if sky[i] {
                self.remove(level[i]);
            }
        }
    }

    /// The lower median of the levels in the window: the level at rank (count − 1) ÷ 2 in their
    /// order. The caller counts one level or more.
    fn median(&mut self) -> u8 {
        let rank = (self.count - 1) / 2;
        while self.below > rank {
            self.at -= 1;
            self.below -= self.hist[self.at];
        }
        while self.below + self.hist[self.at] <= rank {
            self.below += self.hist[self.at];
            self.at += 1;
        }
        self.at as u8
    }
}

/// THE SKY BACKGROUND of every sky pixel: the lower median of the sky levels in the window of
/// [`BACKGROUND_WINDOW_RADIUS_PX`] around it, cut at the frame's edge. The window slides along each
/// row: a column enters on the right and a column leaves on the left. A pixel that is not sky gets
/// no background (0), and it is never a star pixel.
fn background_levels(level: &[u8], sky: &[bool], w: usize, h: usize) -> Vec<u8> {
    let r = BACKGROUND_WINDOW_RADIUS_PX;
    let mut background = vec![0u8; w * h];
    for y in 0..h {
        let rows = (y.saturating_sub(r), (y + r).min(h - 1));
        let mut window = Window::new();
        let mut entering = 0usize;
        for x in 0..w {
            while entering <= (x + r).min(w - 1) {
                window.enter(level, sky, w, entering, rows);
                entering += 1;
            }
            if x > r {
                window.leave(level, sky, w, x - r - 1, rows);
            }
            let i = y * w + x;
            if sky[i] {
                background[i] = window.median();
            }
        }
    }
    background
}

/// A blob while the flood fill grows it.
#[derive(Default)]
struct Blob {
    area: u64,
    white: u64,
    light: f64,
    moment_x: f64,
    moment_y: f64,
    peak: u8,
    background: u8,
}

impl Blob {
    fn add(&mut self, x: usize, y: usize, level: u8, background: u8, light: &[f64; 256]) {
        let over = light[usize::from(level)] - light[usize::from(background)];
        self.area += 1;
        self.white += u64::from(level == u8::MAX);
        self.light += over;
        self.moment_x += over * (x as f64 + 0.5);
        self.moment_y += over * (y as f64 + 0.5);
        if level > self.peak {
            self.peak = level;
            self.background = background;
        }
    }

    /// The star the blob makes. Each star pixel stands over its background, so the light is more
    /// than zero.
    fn star(&self) -> Star {
        Star {
            x: self.moment_x / self.light,
            y: self.moment_y / self.light,
            peak: self.peak,
            background: self.background,
            light: self.light,
            area: self.area,
        }
    }
}

/// THE FLAT TOP of a blob: how many of its pixels stand under the top level with a light over their
/// background of [`FLAT_TOP_SHARE`] of the blob's peak light or more. The white pixels are the white
/// core's, so the flat top leaves them out: a small sun disc is white over its face, and
/// [`STAR_MAX_WHITE_PX`] judges it.
fn flat_top(members: &[Member], light: &[f64; 256]) -> u64 {
    let over = |m: &Member| light[usize::from(m.level)] - light[usize::from(m.background)];
    let peak = members.iter().map(over).fold(0.0, f64::max);
    members
        .iter()
        .filter(|&m| (m.level < u8::MAX) & (over(m) >= FLAT_TOP_SHARE * peak))
        .count() as u64
}

/// One pixel of a blob: its index in the frame (row after row), its luma level and its background.
#[derive(Clone, Copy, Debug)]
struct Member {
    index: usize,
    level: u8,
    background: u8,
}

/// The empty mark of [`split`]'s tables: a pixel not flooded yet, a peak that joined no other.
const NONE: usize = usize::MAX;

/// The eight neighbours of a pixel, also at a corner.
const NEIGHBOURS: [(i64, i64); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// The peak that heads a pixel's flood: follows the flood's links up, and halves the path on the way.
fn flood_root(flood: &mut [usize], mut s: usize) -> usize {
    while flood[s] != s {
        flood[s] = flood[flood[s]];
        s = flood[s];
    }
    s
}

/// ★ THE STARS OF ONE BLOB ([`PEAK_SPLIT_LEVELS`]): the blob's pixels, split into one part for each
/// peak that stands [`PEAK_SPLIT_LEVELS`] or more over its pass. The pixels flood the blob from the
/// brightest down (the first pixel first among equal levels): a pixel with no flooded neighbour starts
/// a peak; a pixel that touches two floods is their pass, and the flood of the lower peak (the later
/// one among equals) joins the higher one's; a lower peak that stands under [`PEAK_SPLIT_LEVELS`] over
/// that pass is no star of its own, and its pixels are the higher peak's. Each pixel takes the peak of
/// its brightest flooded neighbour (the first among equals). The parts come in the order of their
/// first pixel, and each part keeps the blob's own order of its pixels — so a blob with one peak is
/// one part, the same pixels in the same order, and its star is the whole blob's to the last bit.
/// `w` is the frame's width.
fn split(members: &[Member], w: usize) -> Vec<Vec<Member>> {
    let n = members.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&s| (std::cmp::Reverse(members[s].level), members[s].index));
    let mut rank = vec![0usize; n];
    for (k, &s) in order.iter().enumerate() {
        rank[s] = k;
    }
    let slot: BTreeMap<usize, usize> = members
        .iter()
        .enumerate()
        .map(|(s, m)| (m.index, s))
        .collect();
    let mut flood = vec![NONE; n];
    let mut owner = vec![NONE; n];
    let mut joins = vec![NONE; n];
    for &s in &order {
        let (x, y) = ((members[s].index % w) as i64, (members[s].index / w) as i64);
        let flooded: Vec<usize> = NEIGHBOURS
            .iter()
            .filter_map(|&(dx, dy)| {
                let (nx, ny) = (x + dx, y + dy);
                ((nx >= 0) & (ny >= 0) & (nx < w as i64))
                    .then(|| slot.get(&(ny as usize * w + nx as usize)).copied())
                    .flatten()
            })
            .filter(|&t| flood[t] != NONE)
            .collect();
        // THE PIXEL'S PEAK: its brightest flooded neighbour's; a pixel with none starts its own.
        owner[s] = flooded
            .iter()
            .copied()
            .min_by_key(|&t| rank[t])
            .map_or(s, |t| owner[t]);
        // THE PASS: the floods the pixel touches join the flood of the highest peak among them.
        let mut roots: Vec<usize> = flooded.iter().map(|&t| flood_root(&mut flood, t)).collect();
        roots.sort_by_key(|&r| rank[r]);
        roots.dedup();
        let top = roots.first().copied().unwrap_or(s);
        flood[s] = top;
        for &r in roots.iter().skip(1) {
            flood[r] = top;
            joins[r] = if members[r].level - members[s].level >= PEAK_SPLIT_LEVELS {
                NONE
            } else {
                top
            };
        }
    }
    // THE PARTS: each pixel's peak, followed through the peaks that joined a higher one.
    let star_of = |mut p: usize| {
        while joins[p] != NONE {
            p = joins[p];
        }
        p
    };
    let mut parts: BTreeMap<usize, Vec<Member>> = BTreeMap::new();
    for (s, m) in members.iter().enumerate() {
        parts.entry(star_of(owner[s])).or_default().push(*m);
    }
    let mut parts: Vec<Vec<Member>> = parts.into_values().collect();
    parts.sort_by_key(|part| part.iter().map(|m| m.index).min());
    parts
}

/// THE STARS OF ONE FRAME: the star pixels over their sky background, grouped into 8-connected
/// blobs; each blob split at its peaks ([`PEAK_SPLIT_LEVELS`]); the parts over [`STAR_MAX_AREA_PX`],
/// [`STAR_MAX_WHITE_PX`] or [`STAR_MAX_FLAT_PX`] are counted and are not stars. A frame whose bytes
/// are not one picture and one probe of their stated sizes shows nothing, and so does a frame whose
/// camera has another viewport.
#[must_use]
pub fn stars_in(frame: &StarFrame<'_>) -> FrameStars {
    let mut read = FrameStars::default();
    if !frame.well_formed() {
        return read;
    }
    let (w, h) = (frame.width, frame.height);
    let light = linear_table();
    let mut level = Vec::with_capacity(w * h);
    let mut sky = Vec::with_capacity(w * h);
    for (i, px) in frame.rgba.chunks_exact(4).enumerate() {
        let open = frame.is_sky(i % w, i / w);
        level.push(luma_level([px[0], px[1], px[2]]));
        sky.push(open);
        read.sky += u64::from(open);
    }
    let background = background_levels(&level, &sky, w, h);
    let mut lit: Vec<bool> = (0..w * h)
        .map(|i| {
            sky[i]
                & (u16::from(level[i])
                    >= u16::from(background[i]) + u16::from(STAR_THRESHOLD_LEVELS))
        })
        .collect();
    let mut stack = Vec::new();
    let mut members = Vec::new();
    for start in 0..w * h {
        if !lit[start] {
            continue;
        }
        lit[start] = false;
        stack.push(start);
        members.clear();
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            members.push(Member {
                index: i,
                level: level[i],
                background: background[i],
            });
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    let inside = (nx >= 0) & (ny >= 0) & (nx < w as i64) & (ny < h as i64);
                    if inside {
                        let j = ny as usize * w + nx as usize;
                        if lit[j] {
                            lit[j] = false;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        for part in split(&members, w) {
            let mut blob = Blob::default();
            for m in &part {
                blob.add(m.index % w, m.index / w, m.level, m.background, &light);
            }
            let large = blob.area > STAR_MAX_AREA_PX;
            let white = !large & (blob.white > STAR_MAX_WHITE_PX);
            let flat = !large & !white & (flat_top(&part, &light) > STAR_MAX_FLAT_PX);
            read.too_large += u64::from(large);
            read.too_white += u64::from(white);
            read.too_flat += u64::from(flat);
            if large | white | flat {
                continue;
            }
            read.stars.push(blob.star());
        }
    }
    read
}

/// What a candidate frame shows at the place of a reference star.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sighting {
    /// A star of the frame answers for it: its index in that frame's stars.
    Present(usize),
    /// The frame shows open sky at the place, and no star of the frame answers for it.
    Missing,
    /// The ground of the frame covers the place, and no star of the frame answers for it.
    Hidden,
    /// ★ The place stands under the frame's overlay, or within [`STAR_MAX_RADIUS_PX`] of it (2026-10-01).
    /// It is NO CHECK, also where a star answers: the overlay is no part of the world's picture, its
    /// readouts change its rectangle from frame to frame, and it hides the star's pixels under it, so
    /// the frame says nothing of the star or of its light ([`Sighting::is_check`]) — the same reach the
    /// frame's own edge keeps. MEASURED at the solved dusk stand: the turning census's "hidden" losses
    /// were stars under the lower edge of a candidate's overlay that today's shorter readout left on
    /// the sky; and the worst frame's light (−76 %) was one star 0.3 px past the overlay's right edge,
    /// whose pixels under it the frame did not count.
    Overlaid,
    /// The place is off the frame (or nearer to its edge than [`STAR_MAX_RADIUS_PX`]) or behind the
    /// camera. It is NO CHECK: the frame's camera does not look at the star, so the frame says
    /// nothing of it ([`Sighting::is_check`]).
    OutOfView,
}

impl Sighting {
    /// Whether the frame shows the star.
    #[must_use]
    pub fn is_present(&self) -> bool {
        matches!(self, Sighting::Present(_))
    }

    /// ★ WHETHER THE SIGHTING CHECKS THE STAR: the frame's camera looks at its place, and the frame
    /// shows the world there (present, missing or hidden). A star out of view is no check. Through
    /// one camera on both sides it never happens; in a turn, today's frame of the nearest camera
    /// ([`nearest_frame`]) looks a little past the candidate's, so a star at the frame's edge stands
    /// in one view and out of the other. A star under the frame's overlay is no check either
    /// ([`Sighting::Overlaid`]).
    #[must_use]
    pub fn is_check(&self) -> bool {
        !matches!(self, Sighting::OutOfView | Sighting::Overlaid)
    }
}

/// A reference star that a candidate frame does not show: the frame, the star (its index among that
/// frame's reference stars) and what the frame shows at its place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MissingEvent {
    pub frame: usize,
    pub star: usize,
    /// [`Sighting::Missing`] or [`Sighting::Hidden`].
    pub sighting: Sighting,
}

/// Where the place of a star stands in a frame.
enum Place {
    /// Off the frame (or nearer to its edge than [`STAR_MAX_RADIUS_PX`]), or behind the camera.
    Out,
    /// In the frame, but the ground covers the place.
    Covered(ScreenPos),
    /// In the frame, but under the overlay or within [`STAR_MAX_RADIUS_PX`] of it.
    Overlaid(ScreenPos),
    /// On the sky of the frame.
    Open(ScreenPos),
}

impl Place {
    /// The place where a star of the frame can answer: none when it is out of view. A covered place
    /// can still get an answer. Example: the candidate's probe puts the belt's ridge one pixel
    /// higher than today's, over the centre of a star, and the star still shows on the sky above
    /// it.
    fn spot(&self) -> Option<ScreenPos> {
        match self {
            Place::Out => None,
            Place::Covered(p) | Place::Overlaid(p) | Place::Open(p) => Some(*p),
        }
    }
}

/// Where the place `at` of a star stands in a frame: out of view, under the overlay, covered by the
/// ground, or on open sky.
fn place(frame: &StarFrame<'_>, at: Option<ScreenPos>) -> Place {
    let Some(p) = at else {
        return Place::Out;
    };
    let m = STAR_MAX_RADIUS_PX;
    let inside = (p.x >= m)
        & (p.y >= m)
        & (p.x <= frame.width as f64 - m)
        & (p.y <= frame.height as f64 - m);
    if !inside {
        return Place::Out;
    }
    if frame.near_overlay(p.x, p.y, STAR_MAX_RADIUS_PX) {
        return Place::Overlaid(p);
    }
    if frame.is_sky(p.x as usize, p.y as usize) {
        Place::Open(p)
    } else {
        Place::Covered(p)
    }
}

/// Whether a star of a frame joins the census of that frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Join {
    /// Its peak stands under [`CENSUS_PEAK_LEVELS`] over its background.
    Faint,
    /// It stands nearer to the frame's edge than [`STAR_MAX_RADIUS_PX`], or off the sky.
    Edge,
    /// It joins.
    Yes,
}

/// Whether a star of a frame joins the census: it peaks [`CENSUS_PEAK_LEVELS`] or more over its
/// background and stands on open sky, [`STAR_MAX_RADIUS_PX`] or more inside the frame.
fn join(frame: &StarFrame<'_>, star: &Star) -> Join {
    if u16::from(star.peak) < u16::from(star.background) + u16::from(CENSUS_PEAK_LEVELS) {
        return Join::Faint;
    }
    match place(
        frame,
        Some(ScreenPos {
            x: star.x,
            y: star.y,
        }),
    ) {
        Place::Open(_) => Join::Yes,
        Place::Out | Place::Covered(_) | Place::Overlaid(_) => Join::Edge,
    }
}

/// The camera moved to the origin: the same view, without the eye's position. A star's direction
/// projects through it exactly (a star is at infinity).
fn at_origin(camera: &CaptureCamera) -> CaptureCamera {
    CaptureCamera {
        eye: DVec3::ZERO,
        target: camera.target - camera.eye,
        ..*camera
    }
}

/// The place of a star of one frame in another frame. When one of the two frames gives no camera,
/// it is the same place. When both give one, it is the star's direction through the first camera,
/// projected through the second; `None` when the direction is behind the second camera.
fn predict(
    x: f64,
    y: f64,
    from: Option<&CaptureCamera>,
    to: Option<&CaptureCamera>,
) -> Option<ScreenPos> {
    match (from, to) {
        (Some(a), Some(b)) => at_origin(b).project_point(at_origin(a).unproject(x, y, 1.0)),
        _ => Some(ScreenPos { x, y }),
    }
}

/// The cell of the matching grid that a coordinate falls in. The cells are [`MATCH_RADIUS_PX`]
/// wide, so each star within the radius of a place stands in the place's cell or in one of the
/// eight cells around it.
fn cell(v: f64) -> i64 {
    (v / MATCH_RADIUS_PX).floor() as i64
}

/// The stars of a frame by the cell of the matching grid that they stand in.
type Grid = BTreeMap<(i64, i64), Vec<usize>>;

/// The matching grid of a frame's stars.
fn grid_of(stars: &[Star]) -> Grid {
    let mut grid = Grid::new();
    for (j, star) in stars.iter().enumerate() {
        grid.entry((cell(star.x), cell(star.y)))
            .or_default()
            .push(j);
    }
    grid
}

/// The stars of a frame within [`MATCH_RADIUS_PX`] of the place `p` of star `i`, as pairs (the
/// squared distance, the star of the place, the frame's star).
fn near(grid: &Grid, stars: &[Star], p: ScreenPos, i: usize, pairs: &mut Vec<(f64, usize, usize)>) {
    let (cx, cy) = (cell(p.x), cell(p.y));
    for dy in -1..=1 {
        for dx in -1..=1 {
            let Some(list) = grid.get(&(cx + dx, cy + dy)) else {
                continue;
            };
            for &j in list {
                let d2 = (stars[j].x - p.x).powi(2) + (stars[j].y - p.y).powi(2);
                if d2 <= MATCH_RADIUS_PX * MATCH_RADIUS_PX {
                    pairs.push((d2, i, j));
                }
            }
        }
    }
}

/// THE ANSWERS of a frame's stars to some places: each place gets a star of the frame within
/// [`MATCH_RADIUS_PX`], the nearest pair first, and a pair counts only when neither of its two ends
/// has a partner yet. A star marked in `taken` answers for no place; each star that answers is
/// marked in it. A place of `None` gets no answer.
fn assign(
    spots: &[Option<ScreenPos>],
    grid: &Grid,
    stars: &[Star],
    taken: &mut [bool],
) -> Vec<Option<usize>> {
    let mut pairs = Vec::new();
    for (i, spot) in spots.iter().enumerate() {
        if let Some(p) = spot {
            near(grid, stars, *p, i, &mut pairs);
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut answers = vec![None; spots.len()];
    for (_, i, j) in pairs {
        if answers[i].is_none() & !taken[j] {
            answers[i] = Some(j);
            taken[j] = true;
        }
    }
    answers
}

/// The sighting of a reference star: the star of the frame that answers for it, or else what the
/// frame shows at its place.
fn sighting(answer: Option<usize>, at: &Place) -> Sighting {
    match (answer, at) {
        (_, Place::Overlaid(_)) => Sighting::Overlaid,
        (Some(j), _) => Sighting::Present(j),
        (None, Place::Out) => Sighting::OutOfView,
        (None, Place::Covered(_)) => Sighting::Hidden,
        (None, Place::Open(_)) => Sighting::Missing,
    }
}

/// ★ A CROWDED STAR (2026-10-01, G-STARS at the solved dusk stand): another star of its frame stands
/// nearer than [`STAR_MAX_RADIUS_PX`], the widest star's radius, so that star's light can fall on its
/// place. A crowded star is no reference star. MEASURED: two runs never turn through the same cameras,
/// so each star falls on a new sub-pixel place, and in a crowded patch close peaks split and merge
/// differently from picture to picture ([`PEAK_SPLIT_LEVELS`]). In the turning census of the dusk
/// stand (64 frames, 9 185 checks) the lost share fell with the distance of the nearest other star:
/// 30 % under 2 px, 7 % at 2–3 px, 1.3 % at 3–4 px, 0.4 % at 4–6 px (each of those nine in a cluster
/// of 3–5 stars within 6 px), and 0 of 1 806 at 6 px and over. Example: in the milky band over the
/// belt, a star of peak 35 with three stars within 6 px is crowded; a star of peak 33 alone in a
/// clear patch is a reference star.
fn crowded(stars: &[Star], i: usize) -> bool {
    let s = stars[i];
    stars.iter().enumerate().any(|(j, o)| {
        (j != i)
            & ((o.x - s.x).powi(2) + (o.y - s.y).powi(2) < STAR_MAX_RADIUS_PX * STAR_MAX_RADIUS_PX)
    })
}

/// THE REFERENCE of one frame: TODAY'S picture at the stand, read once. Example: at a still stand
/// at dusk, the gate reads frame 0 of today's run once, and it serves each of the 64 candidate
/// frames.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StarReference {
    /// What the frame shows: its stars, and the blobs that are not stars.
    pub read: FrameStars,
    /// The reference stars, as indices into `read.stars`: the stars that peak
    /// [`CENSUS_PEAK_LEVELS`] or more over their background and stand on open sky,
    /// [`STAR_MAX_RADIUS_PX`] or more inside the frame and from every other star of the frame.
    pub checked: Vec<usize>,
    /// The stars left out for a peak under [`CENSUS_PEAK_LEVELS`].
    pub faint: usize,
    /// The stars left out for a place nearer to the frame's edge than [`STAR_MAX_RADIUS_PX`], or
    /// off the sky.
    pub edge: usize,
    /// ★ The stars left out for another star of the frame nearer than [`STAR_MAX_RADIUS_PX`]: a
    /// crowded star ([`crowded`]).
    pub crowded: usize,
    /// Whether the frame could be read. An unreadable reference names no star, and each candidate
    /// frame read against it is unreadable too.
    pub readable: bool,
    /// The camera that drew the frame.
    camera: Option<CaptureCamera>,
}

impl StarReference {
    /// Reads one reference frame: its stars, and which of them the census checks.
    #[must_use]
    pub fn of(frame: &StarFrame<'_>) -> StarReference {
        let mut reference = StarReference {
            read: stars_in(frame),
            readable: frame.well_formed(),
            camera: frame.camera,
            ..StarReference::default()
        };
        for (j, star) in reference.read.stars.iter().enumerate() {
            match join(frame, star) {
                Join::Faint => reference.faint += 1,
                Join::Edge => reference.edge += 1,
                Join::Yes if crowded(&reference.read.stars, j) => reference.crowded += 1,
                Join::Yes => reference.checked.push(j),
            }
        }
        reference
    }
}

/// One candidate frame against its reference frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameComparison {
    /// Whether both frames could be read. An unreadable pair checks nothing.
    pub readable: bool,
    /// What the candidate frame shows.
    pub read: FrameStars,
    /// The reference stars of the pair, with their places and their light in the reference frame.
    pub reference: Vec<Star>,
    /// What the candidate frame shows at the place of each reference star, in the same order.
    pub sightings: Vec<Sighting>,
    /// The extra stars of the candidate frame: its stars that peak [`CENSUS_PEAK_LEVELS`] or more
    /// over their background, stand on open sky [`STAR_MAX_RADIUS_PX`] or more inside the frame,
    /// and answer for no star of the reference frame. The faint stars of the reference and the
    /// stars at its edge answer too, so a star that the reference shows is never an extra.
    pub extra: usize,
    /// ★ Whether the frame's camera looks away from its reference's picture: the centre of its view
    /// stands outside the reference frame, or behind its camera ([`StarRefusal::LooksAway`]). A
    /// frame of a turn whose few stars all sat at the reference's edge leaves them out of view by
    /// the pairing's own small turn; it looks into the reference's picture, and is not one of these.
    pub looks_away: bool,
}

impl FrameComparison {
    /// The light of the reference stars that the candidate frame shows: (their light in the
    /// reference frame, their light in the candidate frame).
    #[must_use]
    pub fn lights(&self) -> (f64, f64) {
        let mut sums = (0.0, 0.0);
        for (star, sighting) in self.reference.iter().zip(&self.sightings) {
            if let Sighting::Present(j) = *sighting {
                sums.0 += star.light;
                sums.1 += self.read.stars[j].light;
            }
        }
        sums
    }

    /// The change of the frame's light: (candidate − reference) ÷ reference, from
    /// [`lights`](Self::lights). It is 0 when the candidate shows no reference star, because there
    /// is no light to compare; the lost stars fail the verdict on their own.
    #[must_use]
    pub fn light_change(&self) -> f64 {
        let (reference, candidate) = self.lights();
        (candidate - reference) / reference.max(f64::MIN_POSITIVE)
    }
}

/// Reads a well-formed candidate frame against a readable reference.
fn compare(reference: &StarReference, frame: &StarFrame<'_>) -> FrameComparison {
    let read = stars_in(frame);
    let grid = grid_of(&read.stars);
    let mut taken = vec![false; read.stars.len()];
    let (from, to) = (reference.camera.as_ref(), frame.camera.as_ref());
    let mut places: Vec<Option<ScreenPos>> = reference
        .read
        .stars
        .iter()
        .map(|s| predict(s.x, s.y, from, to))
        .collect();
    let wheres: Vec<Place> = reference
        .checked
        .iter()
        .map(|&i| place(frame, places[i]))
        .collect();
    let spots: Vec<Option<ScreenPos>> = wheres.iter().map(Place::spot).collect();
    let answers = assign(&spots, &grid, &read.stars, &mut taken);
    // The other stars of the reference (faint, or at its edge) take their own stars of the
    // candidate next, so a star that the reference shows too is never an extra.
    for &i in &reference.checked {
        places[i] = None;
    }
    assign(&places, &grid, &read.stars, &mut taken);
    let extra = read
        .stars
        .iter()
        .zip(&taken)
        .filter(|&(star, &t)| !t & (join(frame, star) == Join::Yes))
        .count();
    FrameComparison {
        readable: true,
        reference: reference
            .checked
            .iter()
            .map(|&i| reference.read.stars[i])
            .collect(),
        sightings: answers
            .iter()
            .zip(&wheres)
            .map(|(answer, at)| sighting(*answer, at))
            .collect(),
        extra,
        looks_away: looks_away(from, to),
        read,
    }
}

/// ★ Whether the camera `to` looks away from the picture of the camera `from`: the centre of its
/// view projects outside that picture, or behind its camera. Without both cameras the two frames
/// share one view. Example: a candidate frame stamped with the camera of the turn's far side looks
/// away; a frame 5° off its partner's camera looks into its picture.
fn looks_away(from: Option<&CaptureCamera>, to: Option<&CaptureCamera>) -> bool {
    let (Some(a), Some(b)) = (from, to) else {
        return false;
    };
    at_origin(a)
        .project_point(b.target - b.eye)
        .is_none_or(|c| {
            (c.x < 0.0) | (c.y < 0.0) | (c.x > a.width as f64) | (c.y > a.height as f64)
        })
}

/// THE VERDICT of a census against the design's proposed target (§12.4; the owner rules it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarVerdict {
    /// Each reference star is present in each candidate frame, and the light of each frame holds
    /// within [`LIGHT_CHANGE_TARGET`].
    Pass,
    /// The candidate run misses the target.
    Fail(StarFault),
    /// The census cannot judge the run.
    Refused(StarRefusal),
}

/// Why a candidate run misses the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarFault {
    /// A candidate frame does not show a reference star that its camera looks at: it is missing or
    /// hidden there ([`StarCensus::missing_events`]).
    StarLost,
    /// In a frame, the light of the reference stars changes by more than [`LIGHT_CHANGE_TARGET`].
    LightChanged,
}

/// Why the census cannot judge a run. A verdict on missing data is never a pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarRefusal {
    /// The census read the run against its own frame 0 ([`StarCensus::add_own`]). That reading is
    /// blind to a star that the run loses in every frame.
    OwnRun,
    /// A frame of the run, or its reference frame, could not be read.
    Unreadable,
    /// The run has fewer than [`CENSUS_FRAMES`] frames.
    TooFewFrames,
    /// A candidate frame checks none of its reference frame's stars, and its camera looks away from
    /// the reference's picture (the centre of its view stands outside it). Example: a candidate frame
    /// whose stamped camera looks back along the turn. The frame checks nothing, and a wrong camera
    /// must never pass by its silence.
    LooksAway,
    /// The run checks no star: the reference shows none, or every one stands out of its candidate
    /// frame's view or under its overlay.
    NoStars,
}

/// THE CENSUS of a candidate run: each candidate frame against its reference frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StarCensus {
    /// Each candidate frame against its reference frame, in the order of the run.
    pub frames: Vec<FrameComparison>,
    /// The run's own frame 0, when the census reads the run against itself
    /// ([`add_own`](Self::add_own)).
    own: Option<StarReference>,
}

impl StarCensus {
    /// Reads one more candidate frame against its reference frame: today's picture at the same
    /// stand and the same tick. The census reads nothing of an unreadable frame (or of a frame
    /// against an unreadable reference): it counts the frame as unreadable, and the verdict refuses
    /// the run.
    pub fn add(&mut self, reference: &StarReference, candidate: &StarFrame<'_>) {
        let readable = reference.readable & candidate.well_formed();
        let frame = if readable {
            compare(reference, candidate)
        } else {
            FrameComparison::default()
        };
        self.frames.push(frame);
    }

    /// THE RUN'S OWN READING: reads one more frame of a run against the run's own frame 0. The
    /// first frame that this reads names the stars. It is not the verdict: the verdict refuses a
    /// census that holds such a frame ([`StarRefusal::OwnRun`]).
    pub fn add_own(&mut self, frame: &StarFrame<'_>) {
        let own = self.own.take().unwrap_or_else(|| StarReference::of(frame));
        self.add(&own, frame);
        self.own = Some(own);
    }

    /// The frames read.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// The frames that could not be read.
    #[must_use]
    pub fn unreadable(&self) -> usize {
        self.frames.iter().filter(|f| !f.readable).count()
    }

    /// The checks of the run: one for each reference star in each candidate frame whose camera looks
    /// at it ([`Sighting::is_check`]).
    #[must_use]
    pub fn checks(&self) -> usize {
        self.frames
            .iter()
            .flat_map(|f| &f.sightings)
            .filter(|s| s.is_check())
            .count()
    }

    /// The reference stars out of their candidate frame's view, over the run: no checks.
    #[must_use]
    pub fn out_of_view(&self) -> usize {
        self.frames
            .iter()
            .flat_map(|f| &f.sightings)
            .filter(|s| matches!(s, Sighting::OutOfView))
            .count()
    }

    /// The reference stars under their candidate frame's overlay, over the run: no checks
    /// ([`Sighting::Overlaid`]).
    #[must_use]
    pub fn overlaid(&self) -> usize {
        self.frames
            .iter()
            .flat_map(|f| &f.sightings)
            .filter(|s| matches!(s, Sighting::Overlaid))
            .count()
    }

    /// The frames that check none of their reference's stars and whose camera looks away from the
    /// reference's picture ([`FrameComparison::looks_away`], [`StarRefusal::LooksAway`]). A frame
    /// whose stars stand under its overlay, or whose few edge stars the pairing's small turn carries
    /// out of view, looks into the reference's picture and checks nothing; it is not one of these.
    #[must_use]
    pub fn looking_away(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| f.looks_away & !f.sightings.iter().any(Sighting::is_check))
            .count()
    }

    /// The checks that found the reference star present.
    #[must_use]
    pub fn present(&self) -> usize {
        self.frames
            .iter()
            .flat_map(|f| &f.sightings)
            .filter(|s| s.is_present())
            .count()
    }

    /// Each check that did not find the reference star present, in the order of the frames.
    #[must_use]
    pub fn missing_events(&self) -> Vec<MissingEvent> {
        let mut events = Vec::new();
        for (frame, comparison) in self.frames.iter().enumerate() {
            for (star, sighting) in comparison.sightings.iter().enumerate() {
                if sighting.is_check() & !sighting.is_present() {
                    events.push(MissingEvent {
                        frame,
                        star,
                        sighting: *sighting,
                    });
                }
            }
        }
        events
    }

    /// The extra stars of all candidate frames. The verdict does not read them (§12.4 names the
    /// lost stars and the light); G-FLICKER and G-GHOST judge a blink and a smear.
    #[must_use]
    pub fn extra(&self) -> usize {
        self.frames.iter().map(|f| f.extra).sum()
    }

    /// The stars that each candidate frame shows: the reference stars and the others.
    #[must_use]
    pub fn visible_counts(&self) -> Vec<usize> {
        self.frames.iter().map(|f| f.read.stars.len()).collect()
    }

    /// The largest change of one frame's light ([`FrameComparison::light_change`]), by its size.
    /// The verdict reads this. It reads each frame on its own, because a star that dims in one
    /// frame (half a blink) moves that frame's light, while a pool of 64 frames moves 64 times
    /// less.
    #[must_use]
    pub fn worst_light_change(&self) -> f64 {
        self.frames
            .iter()
            .map(|f| f.light_change().abs())
            .fold(0.0, f64::max)
    }

    /// The change of the light pooled over all frames. It reads a change that every frame shares:
    /// for example the air's transmittance from the table (§7.7, Difference 1). The verdict does
    /// not read it: [`worst_light_change`](Self::worst_light_change) holds each frame to the
    /// target.
    #[must_use]
    pub fn light_change(&self) -> f64 {
        let (reference, candidate) = self
            .frames
            .iter()
            .map(FrameComparison::lights)
            .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        (candidate - reference) / reference.max(f64::MIN_POSITIVE)
    }

    /// THE VERDICT against the design's proposed target (§12.4). The refusals come first: a run
    /// read against itself, an unreadable frame, fewer than [`CENSUS_FRAMES`] frames, a frame whose
    /// camera looks at none of its reference's stars, and a run that checks no star. Then a lost star fails
    /// the run, and then a light change of more than [`LIGHT_CHANGE_TARGET`] in a frame.
    #[must_use]
    pub fn verdict(&self) -> StarVerdict {
        if self.own.is_some() {
            return StarVerdict::Refused(StarRefusal::OwnRun);
        }
        if self.unreadable() > 0 {
            return StarVerdict::Refused(StarRefusal::Unreadable);
        }
        if self.frames.len() < CENSUS_FRAMES {
            return StarVerdict::Refused(StarRefusal::TooFewFrames);
        }
        if self.looking_away() > 0 {
            return StarVerdict::Refused(StarRefusal::LooksAway);
        }
        if self.checks() == 0 {
            return StarVerdict::Refused(StarRefusal::NoStars);
        }
        if self.present() < self.checks() {
            return StarVerdict::Fail(StarFault::StarLost);
        }
        if self.worst_light_change() > LIGHT_CHANGE_TARGET {
            return StarVerdict::Fail(StarFault::LightChanged);
        }
        StarVerdict::Pass
    }
}

/// THE TURN BETWEEN TWO CAMERAS, radians: the angle of the rotation that carries one camera's basis
/// onto the other's, whatever their eyes. A star is at infinity, so this angle alone moves it.
#[must_use]
pub fn turn_between(a: &CaptureCamera, b: &CaptureCamera) -> f64 {
    let (ra, ua, ba) = a.basis();
    let (rb, ub, bb) = b.basis();
    let trace = ra.dot(rb) + ua.dot(ub) + ba.dot(bb);
    ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
}

/// ★ THE PAIRING OF A TURN: the frame of today's run whose camera turns least from a candidate
/// frame's `camera`, and that turn in radians (the first such frame on a tie); `None` for no frame.
/// Two runs cannot turn through the same cameras: the pilot's look reaches the server at its own
/// ticks. So each candidate frame reads today's frame that looked most nearly where it looks, and
/// the census turns each star from the one camera to the other ([`Sighting::is_check`]).
///
/// Example: the pilot turns 90° a second at the dusk stand and the window draws 40 frames a second,
/// so today's frames stand 2.25° apart. A candidate frame's camera stands at most half of that from
/// its partner's: 1.1°, about 18 pixels at the capture's 0.0625° a pixel.
#[must_use]
pub fn nearest_frame(cameras: &[CaptureCamera], camera: &CaptureCamera) -> Option<(usize, f64)> {
    cameras
        .iter()
        .map(|c| turn_between(c, camera))
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

/// THE CENSUS of a whole candidate run against today's picture. One reference frame serves each
/// candidate frame (a still stand). Otherwise reference frame k serves candidate frame k (a turn),
/// and a frame that has no partner on the other side is unreadable.
#[must_use]
pub fn star_census(references: &[StarFrame<'_>], candidates: &[StarFrame<'_>]) -> StarCensus {
    let mut census = StarCensus::default();
    if let [reference] = references {
        let reference = StarReference::of(reference);
        for candidate in candidates {
            census.add(&reference, candidate);
        }
        return census;
    }
    for k in 0..references.len().max(candidates.len()) {
        match (references.get(k), candidates.get(k)) {
            (Some(reference), Some(candidate)) => {
                census.add(&StarReference::of(reference), candidate);
            }
            _ => census.frames.push(FrameComparison::default()),
        }
    }
    census
}

/// THE RUN'S OWN READING of a whole run: frame 0 names the stars, and each frame looks for them
/// ([`StarCensus::add_own`]). It is not the verdict.
#[must_use]
pub fn own_census(frames: &[StarFrame<'_>]) -> StarCensus {
    let mut census = StarCensus::default();
    for frame in frames {
        census.add_own(frame);
    }
    census
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::FIT_FOV_Y;
    use crate::probe::{PROBE_KIND_TERRAIN, ProbePixel, encode_probe};
    use vd_client::realm_scene::{MARKER_CLASS_SRGB, StarTuning};

    /// The sky's level in the synthetic pictures: a dark sky at dusk.
    const SKY: u8 = 10;
    /// A bright star's level.
    const BRIGHT: u8 = 200;

    /// A picture of `w × h` pixels at one grey level.
    fn grey(w: usize, h: usize, level: u8) -> Vec<u8> {
        [level, level, level, 255].repeat(w * h)
    }

    /// A probe of `w × h` pixels: sky above the row `ground`, the ground from that row down.
    fn probe_of(w: usize, h: usize, ground: usize) -> Vec<u8> {
        let t = encode_probe(ProbePixel {
            kind: PROBE_KIND_TERRAIN,
            rung: 0,
            cells: 40,
        });
        let mut bytes = [0, 0, 0, 255].repeat(w * ground);
        bytes.extend([t[0], t[1], t[2], 255].repeat(w * (h - ground)));
        bytes
    }

    /// Marks the probe pixel `(x, y)` as the ground.
    fn ground_at(probe: &mut [u8], w: usize, x: usize, y: usize) {
        let t = encode_probe(ProbePixel {
            kind: PROBE_KIND_TERRAIN,
            rung: 0,
            cells: 40,
        });
        let i = (y * w + x) * 4;
        probe[i..i + 3].copy_from_slice(&t);
    }

    /// Paints the pixel `(x, y)` a grey `level`.
    fn paint(rgba: &mut [u8], w: usize, x: usize, y: usize, level: u8) {
        let i = (y * w + x) * 4;
        rgba[i..i + 3].copy_from_slice(&[level; 3]);
    }

    /// Paints a star as a plus: `peak` at `(x, y)`, `arm` on the four pixels beside it.
    fn plus(rgba: &mut [u8], w: usize, x: usize, y: usize, peak: u8, arm: u8) {
        paint(rgba, w, x, y, peak);
        for (px, py) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
            paint(rgba, w, px, py, arm);
        }
    }

    /// Paints the block `xs × ys` a grey `level`.
    fn fill(
        rgba: &mut [u8],
        w: usize,
        xs: std::ops::Range<usize>,
        ys: std::ops::Range<usize>,
        level: u8,
    ) {
        for y in ys {
            for x in xs.clone() {
                paint(rgba, w, x, y, level);
            }
        }
    }

    /// Paints the pixels whose centres stand within `r` of the point `c` a grey `level`.
    fn disc(rgba: &mut [u8], w: usize, c: (f64, f64), r: f64, level: u8) {
        let reach = r.ceil() as usize + 1;
        let (x0, y0) = (c.0 as usize, c.1 as usize);
        for y in y0 - reach..=y0 + reach {
            for x in x0 - reach..=x0 + reach {
                let (dx, dy) = (x as f64 + 0.5 - c.0, y as f64 + 0.5 - c.1);
                if dx * dx + dy * dy <= r * r {
                    paint(rgba, w, x, y, level);
                }
            }
        }
    }

    /// A still frame whose probe has the picture's size.
    fn frame<'a>(rgba: &'a [u8], probe: &'a [u8], w: usize, h: usize) -> StarFrame<'a> {
        StarFrame {
            rgba,
            width: w,
            height: h,
            probe,
            probe_width: w,
            probe_height: h,
            hud: None,
            camera: None,
        }
    }

    /// The linear light of one pixel at `level` over the sky.
    fn over_sky(level: u8) -> f64 {
        linear_light(level) - linear_light(SKY)
    }

    /// The 8-bit level of a linear light: the sRGB curve encoded and rounded (the inverse of
    /// [`linear_light`]), with the light clipped at full scale.
    fn srgb_level(v: f64) -> u8 {
        let c = if v <= SRGB_TOE / SRGB_TOE_SLOPE {
            v * SRGB_TOE_SLOPE
        } else {
            (1.0 + SRGB_OFFSET) * v.powf(1.0 / SRGB_EXPONENT) - SRGB_OFFSET
        };
        (c.min(1.0) * 255.0).round() as u8
    }

    fn assert_star(s: &Star, at: (f64, f64), peak: u8, background: u8, light: f64, area: u64) {
        assert!((s.x - at.0).abs() < 1e-9, "x of {s:?}");
        assert!((s.y - at.1).abs() < 1e-9, "y of {s:?}");
        assert_eq!((s.peak, s.background, s.area), (peak, background, area));
        assert!((s.light - light).abs() < 1e-12, "{s:?} against {light}");
    }

    /// A pixel of a blob at `level` over `background`, for the flat top's own count.
    fn member(level: u8, background: u8) -> Member {
        Member {
            index: 0,
            level,
            background,
        }
    }

    /// The pixels of [`plus`], as `(x, y, level)`.
    fn plus_pixels(x: usize, y: usize, peak: u8, arm: u8) -> Vec<(usize, usize, u8)> {
        vec![
            (x, y, peak),
            (x - 1, y, arm),
            (x + 1, y, arm),
            (x, y - 1, arm),
            (x, y + 1, arm),
        ]
    }

    /// The star the pixels `px` (`(x, y, level)` over the sky) make: its centroid and its light.
    fn expected_star(px: &[(usize, usize, u8)]) -> ((f64, f64), f64) {
        let light: f64 = px.iter().map(|p| over_sky(p.2)).sum();
        let moment = |at: fn(&(usize, usize, u8)) -> usize| {
            px.iter()
                .map(|p| over_sky(p.2) * (at(p) as f64 + 0.5))
                .sum::<f64>()
                / light
        };
        ((moment(|p| p.0), moment(|p| p.1)), light)
    }

    /// Each reference star's sightings over the frames, as letters: P present, M missing, H hidden,
    /// O out of view, V under the overlay, U an unreadable frame. The reference is the same in each frame (a still
    /// stand, or the run's own reading), so star `i` is the same star in each frame.
    fn codes(census: &StarCensus) -> Vec<String> {
        let stars = census.frames.first().map_or(0, |f| f.sightings.len());
        (0..stars)
            .map(|i| {
                census
                    .frames
                    .iter()
                    .map(|f| match f.sightings.get(i) {
                        Some(Sighting::Present(_)) => 'P',
                        Some(Sighting::Missing) => 'M',
                        Some(Sighting::Hidden) => 'H',
                        Some(Sighting::OutOfView) => 'O',
                        Some(Sighting::Overlaid) => 'V',
                        None => 'U',
                    })
                    .collect()
            })
            .collect()
    }

    /// A field of six reference stars on a 96 × 64 sky over the ground from row 56: single pixels
    /// and pluses at several peaks, the dimmest at exactly the census peak.
    fn field() -> (usize, usize, Vec<u8>, Vec<u8>) {
        let (w, h) = (96, 64);
        let mut rgba = grey(w, h, SKY);
        paint(&mut rgba, w, 50, 12, 150);
        plus(&mut rgba, w, 20, 15, BRIGHT, 100);
        plus(&mut rgba, w, 75, 20, 255, 120);
        paint(&mut rgba, w, 30, 35, 60);
        plus(&mut rgba, w, 60, 40, 180, 90);
        paint(&mut rgba, w, 85, 45, SKY + CENSUS_PEAK_LEVELS);
        (w, h, rgba, probe_of(w, h, 56))
    }

    /// A picture whose stars give `share` of their light over the sky, encoded again. The sky keeps
    /// its level.
    fn dimmed(rgba: &[u8], share: f64) -> Vec<u8> {
        let sky = linear_light(SKY);
        rgba.chunks_exact(4)
            .flat_map(|px| {
                let level = srgb_level(sky + share * (linear_light(px[0]) - sky));
                [level, level, level, px[3]]
            })
            .collect()
    }

    /// CONTROL (a): a candidate like the reference passes. Each reference star is present in each
    /// of the 64 frames, no star is extra, and no light moves.
    #[test]
    fn a_candidate_like_the_reference_passes() {
        let (w, h, rgba, probe) = field();
        let today = frame(&rgba, &probe, w, h);
        let census = star_census(&[today], &[today; CENSUS_FRAMES]);
        assert_eq!(census.frame_count(), CENSUS_FRAMES);
        assert_eq!(census.unreadable(), 0);
        assert_eq!((census.checks(), census.present()), (6 * 64, 6 * 64));
        assert_eq!(census.missing_events(), Vec::new());
        assert_eq!(census.extra(), 0);
        assert_eq!(census.visible_counts(), vec![6; CENSUS_FRAMES]);
        assert_eq!(codes(&census), vec!["P".repeat(CENSUS_FRAMES); 6]);
        assert_eq!(census.frames[9].sightings[2], Sighting::Present(2));
        let total = stars_in(&today).light();
        assert!(total > 0.0);
        assert_eq!(census.frames[9].lights(), (total, total));
        assert_eq!(census.worst_light_change(), 0.0);
        assert_eq!(census.light_change(), 0.0);
        assert_eq!(census.verdict(), StarVerdict::Pass);
    }

    /// THE REVIEW'S CASE (b): a candidate that loses one reference star in EVERY frame fails. Its
    /// own reading is blind to it (its frame 0 lacks the star too), so the verdict refuses that
    /// reading.
    #[test]
    fn a_star_lost_in_every_frame_fails_against_today_s_picture() {
        let (w, h, rgba, probe) = field();
        // The candidate drops the star at (30, 35), reference star 3, in all 64 frames.
        let mut lost = rgba.clone();
        paint(&mut lost, w, 30, 35, SKY);
        let candidate = [frame(&lost, &probe, w, h); CENSUS_FRAMES];
        let census = star_census(&[frame(&rgba, &probe, w, h)], &candidate);
        let events: Vec<MissingEvent> = (0..CENSUS_FRAMES)
            .map(|frame| MissingEvent {
                frame,
                star: 3,
                sighting: Sighting::Missing,
            })
            .collect();
        assert_eq!(census.missing_events(), events);
        assert_eq!(census.present(), 5 * 64);
        assert_eq!(census.verdict(), StarVerdict::Fail(StarFault::StarLost));
        // The light compares the stars that the candidate shows, so it holds.
        assert_eq!(census.worst_light_change(), 0.0);
        // The run's own reading names the stars from its own frame 0: five, all present, no event.
        let own = own_census(&candidate);
        assert_eq!(codes(&own), vec!["P".repeat(CENSUS_FRAMES); 5]);
        assert_eq!(own.missing_events(), Vec::new());
        assert_eq!(own.verdict(), StarVerdict::Refused(StarRefusal::OwnRun));
    }

    /// CONTROL (c): a star lost in one frame is one missing event. It names the frame, and the run
    /// fails.
    #[test]
    fn a_star_lost_in_one_frame_names_that_frame() {
        let (w, h, rgba, probe) = field();
        // Frame 37 drops the star at (30, 35): reference star 3.
        let mut dropped = rgba.clone();
        paint(&mut dropped, w, 30, 35, SKY);
        let mut candidate = vec![frame(&rgba, &probe, w, h); CENSUS_FRAMES];
        candidate[37] = frame(&dropped, &probe, w, h);
        let census = star_census(&[frame(&rgba, &probe, w, h)], &candidate);
        assert_eq!(
            census.missing_events(),
            vec![MissingEvent {
                frame: 37,
                star: 3,
                sighting: Sighting::Missing,
            }]
        );
        assert_eq!(census.present(), 6 * 64 - 1);
        assert_eq!(census.visible_counts()[37], 5);
        // The later stars of frame 37 answer at their own indices there.
        assert_eq!(census.frames[37].sightings[5], Sighting::Present(4));
        // Frame 37's light leaves the lost star out on both sides, so it holds.
        assert_eq!(census.frames[37].light_change(), 0.0);
        assert_eq!(census.verdict(), StarVerdict::Fail(StarFault::StarLost));
    }

    /// The light reads a frame on its own: a star that dims in one frame moves that frame's light
    /// past the target, and the pooled light stays under it.
    #[test]
    fn a_star_dimmed_in_one_frame_fails_on_the_light() {
        let (w, h, rgba, probe) = field();
        // Frame 20 dims the peak of the plus at (20, 15), reference star 1, from 200 to 170.
        let mut dim = rgba.clone();
        paint(&mut dim, w, 20, 15, 170);
        let mut candidate = vec![frame(&rgba, &probe, w, h); CENSUS_FRAMES];
        candidate[20] = frame(&dim, &probe, w, h);
        let census = star_census(&[frame(&rgba, &probe, w, h)], &candidate);
        assert_eq!(census.missing_events(), Vec::new());
        let (reference, dimmed) = census.frames[20].lights();
        assert!((reference - dimmed - (over_sky(BRIGHT) - over_sky(170))).abs() < 1e-12);
        let change = (dimmed - reference) / reference;
        assert_eq!(census.frames[20].light_change(), change);
        assert_eq!(census.worst_light_change(), -change);
        assert!(census.worst_light_change() > LIGHT_CHANGE_TARGET);
        let pooled = (dimmed - reference) / (64.0 * reference);
        assert!((census.light_change() - pooled).abs() < 1e-12);
        assert!(census.light_change().abs() < LIGHT_CHANGE_TARGET);
        assert_eq!(census.verdict(), StarVerdict::Fail(StarFault::LightChanged));
    }

    /// (f) THE LIGHT TARGET: a candidate whose stars are all 2 % dimmer fails, and one whose stars
    /// are 0.5 % dimmer passes. The 8-bit rounding shows: at 0.5 % only the white pixel moves.
    #[test]
    fn a_sky_two_percent_dimmer_fails_and_half_a_percent_passes() {
        let (w, h, rgba, probe) = field();
        let today = frame(&rgba, &probe, w, h);
        let today_light = stars_in(&today).light();
        let mut readings = Vec::new();
        for share in [0.98, 0.995] {
            let dim = dimmed(&rgba, share);
            let candidate = frame(&dim, &probe, w, h);
            let census = star_census(&[today], &[candidate; CENSUS_FRAMES]);
            assert_eq!(census.missing_events(), Vec::new());
            // Every star answers, so the frame's light is the light of all the stars of each frame.
            assert_eq!(
                census.frames[0].lights(),
                (today_light, stars_in(&candidate).light())
            );
            readings.push((census.worst_light_change(), census.verdict()));
        }
        assert!(readings[0].0 > 0.02, "{readings:?}");
        assert!(readings[0].0 < 0.021, "{readings:?}");
        assert_eq!(readings[0].1, StarVerdict::Fail(StarFault::LightChanged));
        assert!(readings[1].0 > 0.002, "{readings:?}");
        assert!(readings[1].0 < 0.003, "{readings:?}");
        assert_eq!(readings[1].1, StarVerdict::Pass);
        // The module's claim: a faint star's pixel at level 20 keeps its level at 3 % less light.
        assert_eq!(dimmed(&[20, 20, 20, 255], 0.97), [20, 20, 20, 255]);
    }

    /// A camera on the home planet's surface, turned `yaw_deg` from the nose (-Z) toward -X, its
    /// eye moved `travel_m` along X.
    fn yawed(yaw_deg: f64, travel_m: f64, w: usize, h: usize) -> CaptureCamera {
        let yaw = yaw_deg.to_radians();
        let eye = DVec3::new(travel_m, 6_371_000.0, 0.0);
        CaptureCamera {
            eye,
            target: eye + DVec3::new(-yaw.sin(), 0.0, -yaw.cos()),
            up: DVec3::Y,
            fov_y: FIT_FOV_Y,
            width: w,
            height: h,
        }
    }

    /// A star's direction at an azimuth and an elevation in degrees, in the frame of [`yawed`]:
    /// azimuth 0 is the nose (-Z), and the azimuth grows toward -X as the yaw does.
    fn direction(azimuth_deg: f64, elevation_deg: f64) -> DVec3 {
        let (az, el) = (azimuth_deg.to_radians(), elevation_deg.to_radians());
        DVec3::new(-az.sin() * el.cos(), el.sin(), -az.cos() * el.cos())
    }

    /// The frames of a run that turns: each picture over the probe, with its camera.
    fn turn_frames<'a>(
        pictures: &'a [Vec<u8>],
        probe: &'a [u8],
        cams: &[CaptureCamera],
    ) -> Vec<StarFrame<'a>> {
        pictures
            .iter()
            .zip(cams)
            .map(|(p, c)| StarFrame {
                camera: Some(*c),
                ..frame(p, probe, c.width, c.height)
            })
            .collect()
    }

    /// A sky with a one-pixel star under each direction that the camera projects into its frame.
    fn sky_of(camera: &CaptureCamera, dirs: &[DVec3]) -> Vec<u8> {
        let (w, h) = (camera.width, camera.height);
        let mut rgba = grey(w, h, SKY);
        for d in dirs {
            if let Some(q) = camera.project_point(camera.eye + *d) {
                let inside = (q.x >= 0.0) & (q.y >= 0.0) & (q.x < w as f64) & (q.y < h as f64);
                if inside {
                    paint(&mut rgba, w, q.x as usize, q.y as usize, BRIGHT);
                }
            }
        }
        rgba
    }

    /// CONTROL (d): a turning run against today's turning run, frame by frame. The run turns 63°,
    /// so each star of frame 0 leaves the view, and later frames check the stars that come in. An
    /// identical run passes. A run that loses, in frame 50 only, a star that frame 0 never shows
    /// fails, and the event names the frame.
    #[test]
    fn a_turning_run_is_compared_frame_by_frame() {
        // 192 × 128: the band's stars stand 16 px apart or more, clear of each other ([`crowded`]).
        let (w, h) = (192, 128);
        // A band of stars all along the turn: one each 3° of azimuth from -30° to 147°, at three
        // elevations.
        let dirs: Vec<DVec3> = (0..60)
            .map(|i| direction(3.0 * f64::from(i) - 30.0, [4.0, 9.0, 14.0][i as usize % 3]))
            .collect();
        // The pilot turns 1° a frame.
        let cams: Vec<CaptureCamera> = (0..CENSUS_FRAMES)
            .map(|k| yawed(k as f64, 0.0, w, h))
            .collect();
        let open = probe_of(w, h, h);
        let today: Vec<Vec<u8>> = cams.iter().map(|c| sky_of(c, &dirs)).collect();
        // The candidate loses the star at azimuth 60° (index 30) in frame 50.
        let mut lossy = today.clone();
        let others: Vec<DVec3> = [&dirs[..30], &dirs[31..]].concat();
        lossy[50] = sky_of(&cams[50], &others);
        let (today_run, lossy_run) = (
            turn_frames(&today, &open, &cams),
            turn_frames(&lossy, &open, &cams),
        );
        // Frame 0 does not show the lost star, and frame 63 shows none of frame 0's stars.
        let lost = cams[0]
            .project_point(cams[0].eye + dirs[30])
            .expect("ahead");
        assert!(lost.x < 0.0, "{lost:?}");
        let first = StarReference::of(&today_run[0]);
        let seen: Vec<DVec3> = first
            .checked
            .iter()
            .map(|&i| {
                let s = first.read.stars[i];
                cams[0].unproject(s.x, s.y, 1.0) - cams[0].eye
            })
            .collect();
        // Azimuths −27° to 27° stand clear of the frame's edge (9 px, 3.2° at this size).
        assert_eq!(seen.len(), 19);
        assert_eq!(sky_of(&cams[63], &seen), grey(w, h, SKY));
        assert_eq!(StarReference::of(&today_run[63]).checked.len(), 19);
        // Today's run against itself, frame k against frame k: every star present.
        let same = star_census(&today_run, &today_run);
        assert_eq!(same.checks(), same.present());
        assert_eq!(same.extra(), 0);
        assert_eq!(same.verdict(), StarVerdict::Pass);
        let census = star_census(&today_run, &lossy_run);
        let events = census.missing_events();
        assert_eq!(events.len(), 1);
        assert_eq!(
            (events[0].frame, events[0].sighting),
            (50, Sighting::Missing)
        );
        let star = census.frames[50].reference[events[0].star];
        let place = cams[50]
            .project_point(cams[50].eye + dirs[30])
            .expect("ahead");
        assert!((star.x - place.x).abs() < 1.0, "{star:?} {place:?}");
        assert!((star.y - place.y).abs() < 1.0, "{star:?} {place:?}");
        assert_eq!(census.verdict(), StarVerdict::Fail(StarFault::StarLost));
    }

    /// ★ CONTROL (f): A TURN OF TWO RUNS, paired by the nearest camera. Today turns 4° a frame; the
    /// candidate turns through cameras 1.9° past today's, as a second run's look does. Each
    /// candidate frame pairs with today's frame k (1.9° off); the stars at the frames' edges stand
    /// in one view and out of the other and are no checks; an identical sky passes. A star lost in
    /// frame 20 alone fails, and the event names the frame.
    #[test]
    fn a_turn_of_two_runs_pairs_each_frame_with_the_nearest_camera() {
        // 192 × 128: the band's stars stand 16 px apart or more, clear of each other ([`crowded`]).
        let (w, h) = (192, 128);
        // A band of stars all along the turn: one each 3° of azimuth from -30° to 297°, at three
        // elevations.
        let dirs: Vec<DVec3> = (0..110)
            .map(|i| direction(3.0 * f64::from(i) - 30.0, [4.0, 9.0, 14.0][i as usize % 3]))
            .collect();
        let today_cams: Vec<CaptureCamera> = (0..CENSUS_FRAMES)
            .map(|k| yawed(4.0 * k as f64, 0.0, w, h))
            .collect();
        let cams: Vec<CaptureCamera> = (0..CENSUS_FRAMES)
            .map(|k| yawed(4.0 * k as f64 + 1.9, 0.0, w, h))
            .collect();
        let open = probe_of(w, h, h);
        let today: Vec<Vec<u8>> = today_cams.iter().map(|c| sky_of(c, &dirs)).collect();
        let candidate: Vec<Vec<u8>> = cams.iter().map(|c| sky_of(c, &dirs)).collect();
        let mut lossy = candidate.clone();
        // A star of frame 20 near the middle of its view: azimuth 81° (index 37).
        let lost = cams[20]
            .project_point(cams[20].eye + dirs[37])
            .expect("ahead");
        assert!((lost.x - 96.0).abs() < 40.0, "{lost:?}");
        let others: Vec<DVec3> = [&dirs[..37], &dirs[38..]].concat();
        lossy[20] = sky_of(&cams[20], &others);
        let references: Vec<StarReference> = turn_frames(&today, &open, &today_cams)
            .iter()
            .map(StarReference::of)
            .collect();
        let census_of = |pictures: &[Vec<u8>]| {
            let mut census = StarCensus::default();
            for (k, f) in turn_frames(pictures, &open, &cams).iter().enumerate() {
                let (r, turn) = nearest_frame(&today_cams, &cams[k]).expect("today's run");
                assert_eq!(r, k);
                assert!((turn.to_degrees() - 1.9).abs() < 1e-9, "{turn}");
                census.add(&references[r], f);
            }
            census
        };
        let same = census_of(&candidate);
        assert!(same.out_of_view() > 0, "the edges: {}", same.out_of_view());
        assert_eq!(same.checks(), same.present());
        assert_eq!(same.looking_away(), 0);
        assert_eq!(same.verdict(), StarVerdict::Pass);
        let census = census_of(&lossy);
        assert_eq!(
            census.missing_events(),
            vec![MissingEvent {
                frame: 20,
                star: census.frames[20]
                    .sightings
                    .iter()
                    .position(|s| *s == Sighting::Missing)
                    .expect("the lost star"),
                sighting: Sighting::Missing
            }]
        );
        assert_eq!(census.verdict(), StarVerdict::Fail(StarFault::StarLost));
    }

    /// ★ A FRAME THAT LOOKS AWAY: a candidate frame whose camera looks back along the turn looks at
    /// none of its reference's stars; the census refuses the run, because a wrong camera must never
    /// pass by its silence.
    #[test]
    fn a_frame_that_looks_at_none_of_its_reference_s_stars_is_refused() {
        // 192 × 128: the stars stand 16 px apart or more, clear of each other ([`crowded`]).
        let (w, h) = (192, 128);
        let dirs: Vec<DVec3> = (0..20)
            .map(|i| direction(3.0 * f64::from(i) - 30.0, [4.0, 9.0][i as usize % 2]))
            .collect();
        let ahead = yawed(0.0, 0.0, w, h);
        let back = yawed(180.0, 0.0, w, h);
        let open = probe_of(w, h, h);
        let today = sky_of(&ahead, &dirs);
        let reference = StarReference::of(&StarFrame {
            camera: Some(ahead),
            ..frame(&today, &open, w, h)
        });
        assert!(!reference.checked.is_empty());
        let mut census = StarCensus::default();
        for k in 0..CENSUS_FRAMES {
            let camera = if k == 7 { back } else { ahead };
            census.add(
                &reference,
                &StarFrame {
                    camera: Some(camera),
                    ..frame(&today, &open, w, h)
                },
            );
        }
        assert_eq!(census.looking_away(), 1);
        assert_eq!(census.missing_events(), Vec::new());
        assert_eq!(
            census.verdict(),
            StarVerdict::Refused(StarRefusal::LooksAway)
        );
    }

    /// ★ A FRAME WHOSE EDGE STAR LEAVES BY THE PAIRING'S SMALL TURN looks into its reference's
    /// picture: it checks nothing, and it is not refused for looking away. Today's frame holds one
    /// clear star 12 px inside its right edge; the candidate's camera stands 4° to the left (11 px at
    /// this size), the star leaves its view, and a run of such frames checks no star.
    #[test]
    fn a_frame_whose_edge_star_leaves_by_a_small_turn_does_not_look_away() {
        let (w, h) = (192, 128);
        let ahead = yawed(0.0, 0.0, w, h);
        let turned = yawed(4.0, 0.0, w, h);
        let open = probe_of(w, h, h);
        let dir = ahead.unproject(w as f64 - 11.5, 40.5, 1.0) - ahead.eye;
        let today = sky_of(&ahead, &[dir]);
        let reference = StarReference::of(&StarFrame {
            camera: Some(ahead),
            ..frame(&today, &open, w, h)
        });
        assert_eq!(reference.checked.len(), 1);
        let later = sky_of(&turned, &[dir]);
        let mut census = StarCensus::default();
        for _ in 0..CENSUS_FRAMES {
            census.add(
                &reference,
                &StarFrame {
                    camera: Some(turned),
                    ..frame(&later, &open, w, h)
                },
            );
        }
        assert_eq!(census.frames[0].sightings, vec![Sighting::OutOfView]);
        assert!(!census.frames[0].looks_away);
        assert_eq!((census.checks(), census.looking_away()), (0, 0));
        assert_eq!(census.verdict(), StarVerdict::Refused(StarRefusal::NoStars));
    }

    /// The turn between two cameras is the angle of their orientations alone; the nearest frame is
    /// the first of equals, and no frame pairs with nothing.
    #[test]
    fn the_turn_between_two_cameras_ignores_the_eye() {
        let (w, h) = (96, 64);
        let a = yawed(0.0, 0.0, w, h);
        let b = yawed(30.0, 5_000.0, w, h);
        assert!((turn_between(&a, &b).to_degrees() - 30.0).abs() < 1e-9);
        assert!(turn_between(&a, &a).abs() < 1e-6);
        let cams = [yawed(10.0, 0.0, w, h), yawed(-10.0, 0.0, w, h)];
        let (index, turn) = nearest_frame(&cams, &a).expect("two frames");
        assert_eq!(index, 0);
        assert!((turn.to_degrees() - 10.0).abs() < 1e-9);
        assert_eq!(nearest_frame(&[], &a), None);
    }

    /// CONTROL (e): the census counts the candidate's extra stars. A new bright star on open sky is
    /// an extra. A reference star that was too faint to check and shows brighter now is not; nor is
    /// a new faint star, nor a new star at the frame's edge. The extras do not change the verdict.
    #[test]
    fn an_extra_star_is_counted_and_does_not_change_the_verdict() {
        let (w, h, mut rgba, probe) = field();
        // A faint star of today's picture: a star, too faint for the census.
        paint(&mut rgba, w, 10, 25, SKY + CENSUS_PEAK_LEVELS - 1);
        let today = StarReference::of(&frame(&rgba, &probe, w, h));
        assert_eq!((today.checked.len(), today.faint), (6, 1));
        let mut brighter = rgba.clone();
        paint(&mut brighter, w, 10, 25, BRIGHT);
        let mut more = brighter.clone();
        paint(&mut more, w, 40, 25, BRIGHT);
        paint(&mut more, w, 45, 30, SKY + CENSUS_PEAK_LEVELS - 1);
        paint(&mut more, w, 3, 30, BRIGHT);
        let mut census = StarCensus::default();
        for k in 0..CENSUS_FRAMES {
            let picture = if k == 12 { &more } else { &brighter };
            census.add(&today, &frame(picture, &probe, w, h));
        }
        assert_eq!(census.frames[12].read.stars.len(), 10);
        assert_eq!(census.frames[12].extra, 1);
        assert_eq!(census.extra(), 1);
        assert_eq!(census.missing_events(), Vec::new());
        assert_eq!(census.verdict(), StarVerdict::Pass);
    }

    /// THE RUN'S OWN READING through a turn: it finds each star of frame 0 again through the
    /// camera's turn; a star that leaves the view, turns behind the eye or sets under the ground is
    /// not missing; and the same pictures read without their cameras lose every star.
    #[test]
    fn the_run_s_own_reading_follows_the_turn() {
        let (w, h) = (160, 120);
        let c0 = yawed(0.0, 0.0, w, h);
        // Five stars on pixel centres of frame 0, as directions from its eye.
        let pixels = [(30, 30), (140, 40), (80, 60), (120, 90), (60, 100)];
        let dirs: Vec<DVec3> = pixels
            .iter()
            .map(|&(x, y)| c0.unproject(f64::from(x) + 0.5, f64::from(y) + 0.5, 1.0) - c0.eye)
            .collect();
        // Frame 1 turns 2° and travels 1 km; frame 2 turns 8°; frame 3 looks back; frame 4 turns
        // 2° over a ground that rises to row 85.
        let cams = [
            c0,
            yawed(2.0, 1_000.0, w, h),
            yawed(8.0, 0.0, w, h),
            yawed(180.0, 0.0, w, h),
            yawed(2.0, 0.0, w, h),
        ];
        let open = probe_of(w, h, h);
        let risen = probe_of(w, h, 85);
        let probes = [&open, &open, &open, &open, &risen];
        let pictures: Vec<Vec<u8>> = cams.iter().map(|c| sky_of(c, &dirs)).collect();
        let frames: Vec<StarFrame<'_>> = (0..5)
            .map(|k| StarFrame {
                camera: Some(cams[k]),
                ..frame(&pictures[k], probes[k], w, h)
            })
            .collect();
        let census = own_census(&frames);
        assert_eq!(
            codes(&census),
            ["PPPOP", "PPOOP", "PPPOP", "PPPOH", "PPPOH"]
        );
        let events = census.missing_events();
        let missing = events.iter().filter(|e| e.sighting == Sighting::Missing);
        assert_eq!(missing.count(), 0);
        // The six stars out of view are no checks and no events; the two set stars are hidden.
        assert_eq!((census.checks(), census.out_of_view()), (19, 6));
        assert_eq!(
            events,
            vec![
                MissingEvent {
                    frame: 4,
                    star: 3,
                    sighting: Sighting::Hidden
                },
                MissingEvent {
                    frame: 4,
                    star: 4,
                    sighting: Sighting::Hidden
                },
            ]
        );
        // Frame 3 looks back: it looks at none of frame 0's stars.
        assert_eq!(census.looking_away(), 1);
        assert_eq!(census.verdict(), StarVerdict::Refused(StarRefusal::OwnRun));
        // The eye's travel does not move a star; a star behind the eye has no place.
        let travelled = predict(30.5, 30.5, Some(&c0), Some(&cams[1])).expect("in front");
        let turned = predict(30.5, 30.5, Some(&c0), Some(&cams[4])).expect("in front");
        assert!((travelled.x - turned.x).abs() < 1e-9);
        assert!((travelled.y - turned.y).abs() < 1e-9);
        assert!(
            travelled.x - 30.5 > 4.0,
            "the 2° turn moves the star: {travelled:?}"
        );
        assert_eq!(predict(30.5, 30.5, Some(&c0), Some(&cams[3])), None);
        // The same two pictures without their cameras: every star moved about 5 px, so each one is
        // missing in frame 1.
        let still = own_census(&[
            frame(&pictures[0], &open, w, h),
            frame(&pictures[1], &open, w, h),
        ]);
        assert_eq!(codes(&still), ["PM"; 5]);
    }

    /// A star on the candidate's sky answers also where the candidate's probe covers the place (the
    /// candidate's ridge one pixel higher). Where no star answers, the covered place is hidden, and
    /// a hidden reference star fails the verdict: the candidate does not show it on its sky.
    #[test]
    fn a_covered_place_can_still_answer_and_a_hidden_star_fails() {
        let (w, h) = (64, 48);
        let open = probe_of(w, h, h);
        let mut today = grey(w, h, SKY);
        paint(&mut today, w, 30, 20, BRIGHT);
        paint(&mut today, w, 40, 30, BRIGHT);
        let reference = StarReference::of(&frame(&today, &open, w, h));
        // The candidate's probe marks the pixels (30, 20) and (40, 30) as the ground. The first
        // star shows one pixel to the right, on the sky; the second does not show.
        let mut covered = open.clone();
        ground_at(&mut covered, w, 30, 20);
        ground_at(&mut covered, w, 40, 30);
        let mut moved = grey(w, h, SKY);
        paint(&mut moved, w, 31, 20, BRIGHT);
        let mut census = StarCensus::default();
        for _ in 0..CENSUS_FRAMES {
            census.add(&reference, &frame(&moved, &covered, w, h));
        }
        assert_eq!(
            census.frames[0].sightings,
            vec![Sighting::Present(0), Sighting::Hidden]
        );
        assert_eq!(census.missing_events().len(), CENSUS_FRAMES);
        assert_eq!(census.verdict(), StarVerdict::Fail(StarFault::StarLost));
    }

    /// ★ A PLACE UNDER THE CANDIDATE'S OVERLAY IS NO CHECK ([`Sighting::Overlaid`]; the dusk stand's
    /// case): today's overlay, grown by its margin and the widest star's radius, ends 3 px over a
    /// star; the candidate's readout is one line longer, and its reach covers the star. RED before: the
    /// star was hidden, a lost star in every frame. The other star, clear of both rectangles, is present; and a candidate whose
    /// overlay covers every reference star still looks at them (none is out of its view), checks
    /// nothing, and the verdict refuses a run of such frames: it checks no star.
    #[test]
    fn a_place_under_the_candidate_s_overlay_is_no_check() {
        let (w, h) = (64, 48);
        let open = probe_of(w, h, h);
        let mut sky = grey(w, h, SKY);
        paint(&mut sky, w, 30, 26, BRIGHT);
        paint(&mut sky, w, 52, 34, BRIGHT);
        let today = StarFrame {
            hud: Some([20.0, 4.0, 36.0, 12.0]),
            ..frame(&sky, &open, w, h)
        };
        let reference = StarReference::of(&today);
        assert_eq!(reference.checked.len(), 2);
        let longer = StarFrame {
            hud: Some([20.0, 4.0, 36.0, 20.0]),
            ..frame(&sky, &open, w, h)
        };
        let mut census = StarCensus::default();
        for _ in 0..CENSUS_FRAMES {
            census.add(&reference, &longer);
        }
        // The candidate shows the first star on its sky (its pixels clear the overlay itself), and
        // answers for it; within the overlay's reach it is still no check.
        assert_eq!(
            census.frames[0].sightings,
            vec![Sighting::Overlaid, Sighting::Present(1)]
        );
        assert_eq!(census.extra(), 0);
        assert_eq!(
            (census.checks(), census.overlaid(), census.out_of_view()),
            (CENSUS_FRAMES, CENSUS_FRAMES, 0)
        );
        assert_eq!(census.missing_events(), Vec::new());
        assert_eq!(codes(&census)[0], "V".repeat(CENSUS_FRAMES));
        assert_eq!(census.verdict(), StarVerdict::Pass);
        let over_both = StarFrame {
            hud: Some([20.0, 4.0, 44.0, 34.0]),
            ..frame(&sky, &open, w, h)
        };
        let mut census = StarCensus::default();
        for _ in 0..CENSUS_FRAMES {
            census.add(&reference, &over_both);
        }
        assert_eq!((census.checks(), census.looking_away()), (0, 0));
        assert_eq!(census.verdict(), StarVerdict::Refused(StarRefusal::NoStars));
    }

    /// CONTROL: the sun is not a star. The sun disc in the owner's window is larger than the widest
    /// star; the sun disc at the capture's reference view has the area of a star but a white face.
    #[test]
    fn a_sun_sized_blob_is_not_a_star() {
        let (w, h) = (128, 96);
        let open = probe_of(w, h, h);
        let mut rgba = grey(w, h, SKY);
        disc(&mut rgba, w, (30.5, 30.5), 12.0, 255);
        disc(&mut rgba, w, (80.5, 30.5), 4.5, 255);
        // A lit moon: larger than the widest star, and not white.
        fill(&mut rgba, w, 100..116, 60..76, 150);
        // A bright near star: its white core (25 px) is inside the star law's bound.
        fill(&mut rgba, w, 26..31, 70..75, 255);
        let read = stars_in(&frame(&rgba, &open, w, h));
        assert_eq!((read.too_large, read.too_white, read.too_flat), (2, 1, 0));
        assert_eq!(read.stars.len(), 1);
        assert_star(
            &read.stars[0],
            (28.5, 72.5),
            255,
            SKY,
            25.0 * over_sky(255),
            25,
        );
    }

    /// CONTROL: a moon is not a star. A moon 0.52° wide at the capture's reference view covers 54
    /// pixels. At level 220 it is not white, and its area is a star's; its flat top holds all 54
    /// pixels. A lit sphere of 57 pixels (full phase) keeps 45 of them over half its peak.
    #[test]
    fn a_moon_is_not_a_star() {
        let (w, h) = (96, 64);
        let open = probe_of(w, h, h);
        let mut rgba = grey(w, h, SKY);
        disc(&mut rgba, w, (20.75, 20.5), 4.1, 220);
        // A lit sphere at full phase: its light falls as √(1 − ρ²) from the centre to the limb.
        let (sky, top) = (linear_light(SKY), linear_light(220));
        let (cx, cy, r) = (60.5, 30.5, 4.2);
        for y in 24..38 {
            for x in 54..68 {
                let rho2 =
                    ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)) / (r * r);
                let shade = (1.0 - rho2).max(0.0).sqrt();
                let level = srgb_level(sky + (top - sky) * shade);
                paint(&mut rgba, w, x, y, level);
            }
        }
        let read = stars_in(&frame(&rgba, &open, w, h));
        assert_eq!((read.too_large, read.too_white, read.too_flat), (0, 0, 2));
        assert_eq!(read.stars, Vec::new());
        // The flat tops themselves: the flat disc's and the lit sphere's.
        let light = linear_table();
        let flat = vec![member(220, SKY); 54];
        assert_eq!(flat_top(&flat, &light), 54);
        let mut shaded = Vec::new();
        for y in 24..38 {
            for x in 54..68 {
                let level = luma_level([rgba[(y * w + x) * 4]; 3]);
                shaded.push(member(level, SKY));
            }
        }
        shaded.retain(|m| m.level >= SKY + STAR_THRESHOLD_LEVELS);
        assert_eq!(shaded.len(), 57);
        assert_eq!(flat_top(&shaded, &light), 45);
    }

    /// The bounds are exact: the widest star, the largest white core and the widest flat top are
    /// stars, and one pixel more is not.
    #[test]
    fn the_largest_star_white_core_and_flat_top_are_stars_and_one_pixel_more_is_not() {
        let (w, h) = (96, 96);
        let open = probe_of(w, h, h);
        let mut rgba = grey(w, h, SKY);
        // 254 pixels at a low level with one peak pixel: a 16 × 16 block less two corners; 255
        // pixels: less one corner. The flat top of each is the peak pixel alone.
        fill(&mut rgba, w, 8..24, 8..24, 30);
        paint(&mut rgba, w, 8, 8, SKY);
        paint(&mut rgba, w, 23, 8, SKY);
        paint(&mut rgba, w, 15, 15, 150);
        fill(&mut rgba, w, 56..72, 8..24, 30);
        paint(&mut rgba, w, 56, 8, SKY);
        paint(&mut rgba, w, 63, 15, 150);
        // 28 white pixels, and 29.
        fill(&mut rgba, w, 8..12, 40..47, 255);
        fill(&mut rgba, w, 56..60, 40..47, 255);
        paint(&mut rgba, w, 60, 40, 255);
        // A flat top of 28 pixels at level 150, and of 29.
        fill(&mut rgba, w, 8..12, 70..77, 150);
        fill(&mut rgba, w, 56..60, 70..77, 150);
        paint(&mut rgba, w, 60, 70, 150);
        let read = stars_in(&frame(&rgba, &open, w, h));
        assert_eq!((read.too_large, read.too_white, read.too_flat), (1, 1, 1));
        let shapes: Vec<(u64, u8)> = read.stars.iter().map(|s| (s.area, s.peak)).collect();
        assert_eq!(
            shapes,
            vec![
                (STAR_MAX_AREA_PX, 150),
                (STAR_MAX_WHITE_PX, 255),
                (STAR_MAX_FLAT_PX, 150)
            ]
        );
    }

    /// The star law's intensity at the squared distance `r2` (px²) from the centre of a star of the
    /// amplitude `a`: its two Gaussians (`StarTuning`), and nothing at the cull or under it (the
    /// sprite's fragment stage discards it there).
    fn law_intensity(law: &StarTuning, a: f64, r2: f64) -> f64 {
        let core = (-0.5 * r2 / (law.core_sigma_px * law.core_sigma_px)).exp();
        let halo = (-0.5 * r2 / (law.halo_sigma_px * law.halo_sigma_px)).exp();
        let i = a * (core + law.halo_weight * halo);
        i * f64::from(u8::from(i > law.cull_level))
    }

    /// One pixel of a star of the law over the sky: each channel adds the class colour times the
    /// intensity to the sky (the sprite shader's `color.rgb * intensity`, added), clips at full
    /// scale (the law: "above 1.0 the core clips") and is encoded to sRGB.
    fn law_pixel(intensity: f64, colour: [f32; 3]) -> [u8; 3] {
        let sky = linear_light(SKY);
        colour.map(|c| srgb_level(sky + f64::from(c) * intensity))
    }

    /// The reach of the patch that [`law_top`] reads, in pixels each way from the star's pixel. The
    /// flat top of a star that the census keeps stays within 4 px of its centre (3.9 px for the M
    /// star at an amplitude of 380), and the patch holds 5.5 px or more around each place.
    const LAW_PATCH: i64 = 6;

    /// The flat top and the white core of one star of the law, on a patch around the star's centre
    /// `at` (the pixel `(0, 0)` has its centre at (0.5, 0.5)). The census's own count
    /// ([`flat_top`]) reads the star pixels.
    fn law_top(law: &StarTuning, a: f64, colour: [f32; 3], at: (f64, f64)) -> (u64, u64) {
        let mut members = Vec::new();
        for y in -LAW_PATCH..=LAW_PATCH {
            for x in -LAW_PATCH..=LAW_PATCH {
                let r2 = (x as f64 + 0.5 - at.0).powi(2) + (y as f64 + 0.5 - at.1).powi(2);
                let level = luma_level(law_pixel(law_intensity(law, a, r2), colour));
                if level >= SKY + STAR_THRESHOLD_LEVELS {
                    members.push(member(level, SKY));
                }
            }
        }
        let white = members.iter().filter(|m| m.level == u8::MAX).count() as u64;
        (flat_top(&members, &linear_table()), white)
    }

    /// The sub-pixel places of the sweep, from the centre of a pixel to its corner.
    const LAW_PLACES: [(f64, f64); 6] = [
        (0.5, 0.5),
        (0.75, 0.5),
        (1.0, 0.5),
        (0.75, 0.75),
        (1.0, 0.75),
        (1.0, 1.0),
    ];

    /// The step of the sweep's amplitude: 1.5 %. On a pixel's corner, the M star's widest flat top
    /// holds over 2 % of the amplitude (from 377 to 386), so the sweep lands on it. It holds from 255
    /// to 258 too, and the sweep lands there first (at 255.6).
    const LAW_STEP: f64 = 1.015;

    /// One star of the law in the sweep: its flat top, its amplitude and its sub-pixel place.
    #[derive(Clone, Copy, Debug, Default)]
    struct LawStar {
        flat: u64,
        amplitude: f64,
        at: (f64, f64),
    }

    /// What the sweep finds in one class colour: the star of the widest flat top that the census
    /// keeps, and the brightest star that the census keeps (the largest amplitude).
    #[derive(Clone, Copy, Debug, Default)]
    struct LawClass {
        widest: LawStar,
        brightest: LawStar,
    }

    /// The sweep of the star law: for each class colour, each amplitude from 0.1 up to where no
    /// place keeps the white core within [`STAR_MAX_WHITE_PX`], and each sub-pixel place. Returns
    /// what it finds in each class, and the widest flat top of a star that no channel clips.
    fn law_sweep() -> (Vec<LawClass>, u64) {
        let law = StarTuning::default();
        let sky = linear_light(SKY);
        let mut classes = Vec::new();
        let mut unclipped = 0;
        for colour in MARKER_CLASS_SRGB {
            let mut class = LawClass::default();
            let mut amplitude = 0.1;
            loop {
                let mut kept = false;
                for at in LAW_PLACES {
                    let (flat, white) = law_top(&law, amplitude, colour, at);
                    let keep = white <= STAR_MAX_WHITE_PX;
                    let star = LawStar {
                        flat,
                        amplitude,
                        at,
                    };
                    kept |= keep;
                    if keep & (flat > class.widest.flat) {
                        class.widest = star;
                    }
                    if keep {
                        class.brightest = star;
                    }
                    // The peak of the profile is at most a · (1 + w_halo): no channel clips under
                    // it.
                    let top = sky + amplitude * (1.0 + law.halo_weight);
                    unclipped = unclipped.max(flat * u64::from(top < 1.0));
                }
                if !kept {
                    break;
                }
                amplitude *= LAW_STEP;
            }
            classes.push(class);
        }
        (classes, unclipped)
    }

    /// THE FLAT-TOP BOUND FOLLOWS THE STAR LAW (`StarTuning`, `MARKER_CLASS_SRGB`): a change of the
    /// law turns this test red, and the bound moves with it. The widest flat top of a star that the
    /// census keeps is the bound, and it is the M class's. A star that no channel clips holds the
    /// four pixels around a pixel's corner.
    #[test]
    fn the_flat_top_bound_follows_the_star_law() {
        let (classes, unclipped) = law_sweep();
        let flats: Vec<u64> = classes.iter().map(|c| c.widest.flat).collect();
        assert_eq!(flats, [20, 17, 17, 17, 17, 20, 28]);
        assert_eq!(flats.iter().max(), Some(&STAR_MAX_FLAT_PX));
        // The M star of the widest flat top stands on a pixel's corner, with a white core of 16
        // pixels, at an amplitude of about 256.
        let law = StarTuning::default();
        let m = classes[6].widest;
        assert_eq!(m.at, (1.0, 1.0));
        assert_eq!(
            law_top(&law, m.amplitude, MARKER_CLASS_SRGB[6], m.at),
            (28, 16)
        );
        assert!((m.amplitude - 256.0).abs() < 2.0, "{m:?}");
        assert_eq!(unclipped, 4);
        // The share: on a pixel's corner, the next pixels get less than a tenth of the corner's
        // four.
        let next = law_intensity(&law, 1.0, 2.5) / law_intensity(&law, 1.0, 0.5);
        assert!((next - 0.094).abs() < 0.001, "{next}");
        assert!(next < FLAT_TOP_SHARE);
    }

    /// The brightest star that the law draws and the census keeps stays a star through
    /// [`stars_in`], in each class colour: the star of the largest amplitude, and the star of the
    /// widest flat top, each drawn into a frame.
    #[test]
    fn the_brightest_star_the_law_draws_stays_a_star() {
        let law = StarTuning::default();
        let (classes, _) = law_sweep();
        let m = classes[6].brightest;
        assert!(m.amplitude > 370.0, "{m:?}");
        let (w, h) = (48, 48);
        let open = probe_of(w, h, h);
        let stars = classes.iter().flat_map(|c| [c.brightest, c.widest]);
        for (k, star) in stars.enumerate() {
            let colour = MARKER_CLASS_SRGB[k / 2];
            let mut rgba = grey(w, h, SKY);
            let (cx, cy) = (24.0 + star.at.0, 24.0 + star.at.1);
            for y in 0..h {
                for x in 0..w {
                    let r2 = (x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2);
                    let i = (y * w + x) * 4;
                    let px = law_pixel(law_intensity(&law, star.amplitude, r2), colour);
                    rgba[i..i + 3].copy_from_slice(&px);
                }
            }
            let read = stars_in(&frame(&rgba, &open, w, h));
            assert_eq!(
                (
                    read.stars.len(),
                    read.too_large,
                    read.too_white,
                    read.too_flat
                ),
                (1, 0, 0, 0),
                "the class {colour:?}: {star:?}"
            );
        }
    }

    /// Stars, and what is not a star: the blobs touch at a corner, the threshold is exact, a star
    /// on the frame's edge counts, and a bright pixel on the ground is never a star.
    #[test]
    fn a_star_is_a_small_bright_blob_on_the_sky() {
        let (w, h) = (64, 48);
        let mut rgba = grey(w, h, SKY);
        paint(&mut rgba, w, 50, 10, 120);
        paint(&mut rgba, w, 51, 11, 120);
        plus(&mut rgba, w, 20, 13, BRIGHT, 100);
        paint(&mut rgba, w, 40, 20, 150);
        paint(&mut rgba, w, 10, 25, SKY + STAR_THRESHOLD_LEVELS);
        paint(&mut rgba, w, 30, 25, SKY + STAR_THRESHOLD_LEVELS - 1);
        paint(&mut rgba, w, 0, 30, 180);
        paint(&mut rgba, w, 30, 44, BRIGHT);
        let ground = probe_of(w, h, 40);
        let read = stars_in(&frame(&rgba, &ground, w, h));
        assert_eq!(
            (read.sky, read.too_large, read.too_white, read.too_flat),
            (64 * 40, 0, 0, 0)
        );
        assert_eq!(read.stars.len(), 5);
        assert_star(
            &read.stars[0],
            (51.0, 11.0),
            120,
            SKY,
            2.0 * over_sky(120),
            2,
        );
        let plus_light = over_sky(BRIGHT) + 4.0 * over_sky(100);
        assert_star(&read.stars[1], (20.5, 13.5), BRIGHT, SKY, plus_light, 5);
        assert_star(&read.stars[2], (40.5, 20.5), 150, SKY, over_sky(150), 1);
        let faint = SKY + STAR_THRESHOLD_LEVELS;
        assert_star(&read.stars[3], (10.5, 25.5), faint, SKY, over_sky(faint), 1);
        assert_star(&read.stars[4], (0.5, 30.5), 180, SKY, over_sky(180), 1);
        let total = 2.0 * over_sky(120) + plus_light + over_sky(150) + over_sky(faint);
        assert!((read.light() - total - over_sky(180)).abs() < 1e-12);
    }

    /// ★ TWO STARS WHOSE WINGS TOUCH STAY TWO STARS ([`PEAK_SPLIT_LEVELS`]; the dusk stand's case): a
    /// star of peak 136 over a star of peak 107, 6 px apart. Today's picture keeps their wings apart;
    /// the candidate's softer wing joins them through three pixels (22, 20, 22), so one blob holds
    /// both. RED before: that blob was one star at (20.5, 12.83), 2.33 px from the upper star's place
    /// and 3.67 px from the lower star's. Now the blob splits at its pass (the 20): each star keeps its
    /// own pixels; the upper star takes the pass and the wing pixel it floods first, the lower star its
    /// own wing pixel. Today's picture names the same two stars, and 6 px apart both are crowded, so
    /// the census checks neither ([`crowded`]).
    #[test]
    fn two_stars_whose_wings_touch_stay_two_stars() {
        let (w, h) = (48, 32);
        let open = probe_of(w, h, h);
        let mut today = grey(w, h, SKY);
        plus(&mut today, w, 20, 10, 136, 60);
        plus(&mut today, w, 20, 16, 107, 50);
        let mut joined = today.clone();
        paint(&mut joined, w, 20, 12, 22);
        paint(&mut joined, w, 20, 13, 20);
        paint(&mut joined, w, 20, 14, 22);
        let read = stars_in(&frame(&joined, &open, w, h));
        assert_eq!(read.stars.len(), 2);
        let mut upper = plus_pixels(20, 10, 136, 60);
        upper.extend([(20, 12, 22), (20, 13, 20)]);
        let (at, light) = expected_star(&upper);
        assert_star(&read.stars[0], at, 136, SKY, light, 7);
        let mut lower = plus_pixels(20, 16, 107, 50);
        lower.push((20, 14, 22));
        let (at, light) = expected_star(&lower);
        assert_star(&read.stars[1], at, 107, SKY, light, 6);
        // Today's picture names the same two stars; 6 px apart, both are crowded ([`crowded`]).
        let reference = StarReference::of(&frame(&today, &open, w, h));
        assert_eq!(reference.read.stars.len(), 2);
        assert_eq!((reference.checked.len(), reference.crowded), (0, 2));
    }

    /// A BUMP IS NO STAR, AND THE SPLIT'S BOUND IS EXACT: a star of peak 136 with a wing of two
    /// pixels under it, 24 and then a bump. A bump of 26 stands 2 levels over its pass (the 24), so
    /// the blob is one star, the whole blob's to the last bit; a bump one level under
    /// [`PEAK_SPLIT_LEVELS`] over the pass is the star's too, and a bump at the bound is a star of its
    /// own.
    #[test]
    fn a_bump_under_the_split_levels_stays_one_star() {
        let (w, h) = (48, 32);
        let open = probe_of(w, h, h);
        let picture = |bump: u8| {
            let mut rgba = grey(w, h, SKY);
            plus(&mut rgba, w, 20, 10, 136, 60);
            paint(&mut rgba, w, 20, 12, 24);
            paint(&mut rgba, w, 20, 13, bump);
            rgba
        };
        let stars = |bump: u8| stars_in(&frame(&picture(bump), &open, w, h)).stars;
        let one = stars(26);
        assert_eq!(one.len(), 1);
        let mut whole = plus_pixels(20, 10, 136, 60);
        whole.extend([(20, 12, 24), (20, 13, 26)]);
        let (at, light) = expected_star(&whole);
        assert_star(&one[0], at, 136, SKY, light, 7);
        assert_eq!(stars(24 + PEAK_SPLIT_LEVELS - 1).len(), 1);
        let two = stars(24 + PEAK_SPLIT_LEVELS);
        assert_eq!(two.len(), 2);
        let bump = 24 + PEAK_SPLIT_LEVELS;
        assert_star(&two[1], (20.5, 13.5), bump, SKY, over_sky(bump), 1);
    }

    #[test]
    fn the_background_is_the_median_of_the_sky_in_the_window() {
        // 40 × 24: the sky at 10 on the left half and at 60 on the right half; the ground (250,
        // bright) from row 12 down.
        let (w, h) = (40, 24);
        let mut level = Vec::new();
        let mut sky = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let open = y < 12;
                let sky_level = 10 + 50 * u8::from(x >= 20);
                level.push(sky_level * u8::from(open) + 250 * u8::from(!open));
                sky.push(open);
            }
        }
        let background = background_levels(&level, &sky, w, h);
        let at = |x: usize, y: usize| background[y * w + x];
        assert_eq!((at(0, 0), at(39, 0)), (10, 60));
        // The window of column 19 holds 16 columns at 10 and 15 at 60; that of column 20, 15 and
        // 16.
        assert_eq!((at(19, 5), at(20, 5)), (10, 60));
        // The window of (5, 11) holds as many ground pixels as sky pixels. Counted, the 250s would
        // lift its median to 60; they are not sky, so it stays at 10.
        assert_eq!(at(5, 11), 10);
        // The ground gets no background.
        assert_eq!(at(5, 12), 0);
    }

    #[test]
    fn the_window_keeps_its_lower_median_while_it_slides() {
        let mut window = Window::new();
        window.add(50);
        assert_eq!(window.median(), 50);
        for v in [10, 10, 90] {
            window.add(v);
        }
        // 10 10 50 90: the second, 10. The median moved down.
        assert_eq!(window.median(), 10);
        for v in [90, 90, 90] {
            window.add(v);
        }
        // 10 10 50 90 90 90 90: the fourth, 90. It moved up past 50.
        assert_eq!(window.median(), 90);
        for v in [90, 90, 10] {
            window.remove(v);
        }
        // 10 50 90 90: the second, 50.
        assert_eq!(window.median(), 50);
        window.remove(10);
        // 50 90 90: the second, 90.
        assert_eq!(window.median(), 90);
    }

    /// One star of a frame answers for one place, the nearest pair first; a star in a cell around
    /// the place but past the radius answers for none ([`assign`]). The census reads it so: a clear
    /// reference star takes the nearest candidate star, and the others are extras.
    #[test]
    fn one_star_answers_for_one_reference_star() {
        let at = |x: f64, y: f64| ScreenPos { x, y };
        let star = |x: f64, y: f64| Star {
            x,
            y,
            peak: BRIGHT,
            background: SKY,
            light: over_sky(BRIGHT),
            area: 1,
        };
        // (21.5, 20.5) stands 1 px from the first place and 1.41 px from the second; (40.5, 30.5)
        // and (42.5, 31.5) stand 1 px and 1.41 px from the third; (39.5, 32.5) stands 2.83 px from it.
        let stars = [
            star(21.5, 20.5),
            star(40.5, 30.5),
            star(42.5, 31.5),
            star(39.5, 32.5),
        ];
        let spots = [
            Some(at(20.5, 20.5)),
            Some(at(22.5, 21.5)),
            Some(at(41.5, 30.5)),
            None,
        ];
        let mut taken = vec![false; stars.len()];
        let answers = assign(&spots, &grid_of(&stars), &stars, &mut taken);
        assert_eq!(answers, vec![Some(0), None, Some(1), None]);
        assert_eq!(taken, vec![true, true, false, false]);
        // The census on two clear reference stars.
        let (w, h) = (64, 48);
        let open = probe_of(w, h, h);
        let mut first = grey(w, h, SKY);
        for (x, y) in [(20, 20), (41, 30)] {
            paint(&mut first, w, x, y, BRIGHT);
        }
        let mut later = grey(w, h, SKY);
        for (x, y) in [(21, 20), (40, 30), (42, 31), (39, 32)] {
            paint(&mut later, w, x, y, BRIGHT);
        }
        let census = star_census(&[frame(&first, &open, w, h)], &[frame(&later, &open, w, h)]);
        assert_eq!(
            census.frames[0].sightings,
            vec![Sighting::Present(0), Sighting::Present(1)]
        );
        assert_eq!(census.missing_events(), Vec::new());
        // (42, 31) and (39, 32) answer for no reference star: two extra stars.
        assert_eq!(census.extra(), 2);
    }

    /// ★ A CROWDED STAR IS NO REFERENCE STAR ([`crowded`]): two stars 8 px apart are both crowded, and
    /// at 9 px — the widest star's radius — both are reference stars; a faint star crowds a bright
    /// one too, for its light falls on the bright star's place all the same.
    #[test]
    fn a_crowded_star_is_no_reference_star() {
        let (w, h) = (64, 48);
        let open = probe_of(w, h, h);
        let reference_of = |pixels: &[(usize, usize, u8)]| {
            let mut rgba = grey(w, h, SKY);
            for &(x, y, level) in pixels {
                paint(&mut rgba, w, x, y, level);
            }
            StarReference::of(&frame(&rgba, &open, w, h))
        };
        let near = reference_of(&[(20, 20, BRIGHT), (28, 20, BRIGHT)]);
        assert_eq!((near.checked.len(), near.crowded), (0, 2));
        let clear = reference_of(&[(20, 20, BRIGHT), (29, 20, BRIGHT)]);
        assert_eq!((clear.checked.len(), clear.crowded), (2, 0));
        let faint = reference_of(&[(20, 20, BRIGHT), (25, 20, SKY + CENSUS_PEAK_LEVELS - 1)]);
        assert_eq!((faint.checked.len(), faint.crowded, faint.faint), (0, 1, 1));
    }

    #[test]
    fn the_reference_leaves_out_faint_stars_and_stars_at_the_edge() {
        let (w, h) = (64, 48);
        let mut probe = probe_of(w, h, h);
        let mut rgba = grey(w, h, SKY);
        // A bright star 4.5 px from the left edge, nearer than the widest star's radius.
        paint(&mut rgba, w, 4, 20, BRIGHT);
        // A star one level under the census peak: a star, but too faint for the census.
        paint(&mut rgba, w, 30, 20, SKY + CENSUS_PEAK_LEVELS - 1);
        // A ring of eight pixels around a pixel of the ground: a star whose centre is not on the
        // sky.
        fill(&mut rgba, w, 19..22, 29..32, BRIGHT);
        paint(&mut rgba, w, 20, 30, SKY);
        ground_at(&mut probe, w, 20, 30);
        paint(&mut rgba, w, 40, 30, BRIGHT);
        let reference = StarReference::of(&frame(&rgba, &probe, w, h));
        assert_eq!(reference.read.stars.len(), 4);
        assert_eq!(reference.read.stars[2].area, 8);
        assert_eq!((reference.faint, reference.edge), (1, 2));
        assert_eq!(reference.checked, vec![3]);
        assert!(reference.readable);
        // One frame is not a run.
        let one = frame(&rgba, &probe, w, h);
        let census = star_census(&[one], &[one]);
        assert_eq!(
            census.verdict(),
            StarVerdict::Refused(StarRefusal::TooFewFrames)
        );
    }

    #[test]
    fn a_census_of_no_stars_proves_nothing() {
        let empty = star_census(&[], &[]);
        assert_eq!(empty, StarCensus::default());
        assert_eq!(own_census(&[]), StarCensus::default());
        assert_eq!((empty.frame_count(), empty.checks()), (0, 0));
        assert_eq!(empty.missing_events(), Vec::new());
        assert_eq!(empty.worst_light_change(), 0.0);
        assert_eq!(empty.light_change(), 0.0);
        assert_eq!(empty.visible_counts(), Vec::<usize>::new());
        assert_eq!(codes(&empty), Vec::<String>::new());
        assert_eq!(
            empty.verdict(),
            StarVerdict::Refused(StarRefusal::TooFewFrames)
        );
        // 64 frames of the ground alone: no sky, no star, no verdict.
        let (w, h) = (16, 12);
        let rgba = grey(w, h, SKY);
        let ground = probe_of(w, h, 0);
        let blind = frame(&rgba, &ground, w, h);
        let census = star_census(&[blind], &[blind; CENSUS_FRAMES]);
        assert_eq!(census.frames[0].read.sky, 0);
        assert_eq!(census.checks(), 0);
        assert_eq!(census.verdict(), StarVerdict::Refused(StarRefusal::NoStars));
    }

    /// Two runs of different lengths cannot be paired: each frame without a partner is unreadable.
    #[test]
    fn runs_of_two_lengths_are_unreadable_where_they_differ() {
        let (w, h, rgba, probe) = field();
        let today = frame(&rgba, &probe, w, h);
        let census = star_census(&[today, today], &[today, today, today]);
        assert_eq!(census.frame_count(), 3);
        assert_eq!(census.unreadable(), 1);
        assert_eq!(census.checks(), 12);
        assert_eq!(
            census.verdict(),
            StarVerdict::Refused(StarRefusal::Unreadable)
        );
    }

    /// FINDING 3: a malformed frame after a good one reads nothing. A short probe does not panic,
    /// and a short picture gives no false missing events. The verdict refuses the run.
    #[test]
    fn a_malformed_later_frame_is_unreadable_and_the_verdict_refuses() {
        let (w, h, rgba, probe) = field();
        let good = frame(&rgba, &probe, w, h);
        // Half of each buffer: the probe ends at row 32, above three of the six reference stars.
        let short_probe = StarFrame {
            probe: &probe[..probe.len() / 2],
            ..good
        };
        let short_picture = StarFrame {
            rgba: &rgba[..rgba.len() / 2],
            ..good
        };
        let wrong_camera = StarFrame {
            camera: Some(yawed(0.0, 0.0, w + 1, h)),
            ..good
        };
        for bad in [short_probe, short_picture, wrong_camera] {
            // The run's own reading: frame 0 is good and frame 1 is bad.
            let own = own_census(&[good, bad]);
            assert_eq!(own.missing_events(), Vec::new());
            assert_eq!(own.unreadable(), 1);
            assert_eq!(codes(&own), ["PU"; 6]);
            // Against today's picture: 63 good frames and the bad one.
            let mut candidate = vec![good; CENSUS_FRAMES];
            candidate[5] = bad;
            let census = star_census(&[good], &candidate);
            assert_eq!(census.unreadable(), 1);
            assert_eq!(census.frames[5], FrameComparison::default());
            assert_eq!(census.missing_events(), Vec::new());
            assert_eq!(
                census.verdict(),
                StarVerdict::Refused(StarRefusal::Unreadable)
            );
            // A bad reference frame names no star, and each frame read against it is unreadable.
            let reference = StarReference::of(&bad);
            assert_eq!(reference.checked, Vec::<usize>::new());
            assert!(!reference.readable);
            let against = star_census(&[bad], &[good; CENSUS_FRAMES]);
            assert_eq!(against.unreadable(), CENSUS_FRAMES);
            assert_eq!(
                against.verdict(),
                StarVerdict::Refused(StarRefusal::Unreadable)
            );
        }
    }

    #[test]
    fn the_probe_may_have_the_render_picture_s_size() {
        // A 64 × 48 picture over a 32 × 24 probe (the up-scale at s = 0.5): the probe's ground from
        // its row 12 covers the picture from row 24 down.
        let (w, h) = (64, 48);
        let mut rgba = grey(w, h, SKY);
        paint(&mut rgba, w, 20, 10, BRIGHT);
        paint(&mut rgba, w, 20, 30, BRIGHT);
        let small = probe_of(32, 24, 12);
        let f = StarFrame {
            probe_width: 32,
            probe_height: 24,
            ..frame(&rgba, &small, w, h)
        };
        assert!(f.is_sky(0, 23));
        assert!(!f.is_sky(0, 24));
        let read = stars_in(&f);
        assert_eq!((read.sky, read.stars.len()), (64 * 24, 1));
        assert_star(
            &read.stars[0],
            (20.5, 10.5),
            BRIGHT,
            SKY,
            over_sky(BRIGHT),
            1,
        );
    }

    #[test]
    fn the_overlay_hides_its_readouts() {
        let (w, h) = (64, 48);
        let mut rgba = grey(w, h, SKY);
        paint(&mut rgba, w, 20, 10, BRIGHT);
        let open = probe_of(w, h, h);
        let bare = frame(&rgba, &open, w, h);
        assert_eq!(stars_in(&bare).stars.len(), 1);
        // The rectangle (18, 8)–(22, 12) grown by the margin covers the columns 16..=23 and the
        // rows 6..=13: 64 pixels that are not sky, the star among them.
        let covered = StarFrame {
            hud: Some([18.0, 8.0, 22.0, 12.0]),
            ..bare
        };
        let read = stars_in(&covered);
        assert_eq!((read.sky, read.stars.len()), (64 * 48 - 64, 0));
        // An overlay elsewhere hides nothing of the star.
        let beside = StarFrame {
            hud: Some([40.0, 30.0, 50.0, 40.0]),
            ..bare
        };
        assert_eq!(stars_in(&beside).stars.len(), 1);
    }

    #[test]
    fn a_malformed_frame_shows_nothing() {
        let (w, h) = (8, 6);
        let rgba = grey(w, h, SKY);
        let open = probe_of(w, h, h);
        let good = frame(&rgba, &open, w, h);
        assert_eq!(stars_in(&good).sky, 48);
        let seen = StarFrame {
            camera: Some(yawed(0.0, 0.0, w, h)),
            ..good
        };
        assert_eq!(stars_in(&seen).sky, 48);
        for bad in [
            StarFrame { width: 0, ..good },
            StarFrame {
                rgba: &rgba[..8],
                ..good
            },
            StarFrame {
                probe: &open[..8],
                ..good
            },
            StarFrame {
                probe_width: 0,
                ..good
            },
            StarFrame {
                camera: Some(yawed(0.0, 0.0, w, h + 1)),
                ..good
            },
        ] {
            assert_eq!(stars_in(&bad), FrameStars::default());
        }
    }

    #[test]
    fn the_luma_and_the_linear_light_follow_their_standards() {
        assert_eq!(luma_level([0, 0, 0]), 0);
        assert_eq!(luma_level([255, 255, 255]), 255);
        assert_eq!(luma_level([200, 200, 200]), 200);
        // The eye weighs green most and blue least.
        assert_eq!(luma_level([255, 0, 0]), 54);
        assert_eq!(luma_level([0, 255, 0]), 182);
        assert_eq!(luma_level([0, 0, 255]), 18);
        assert_eq!(LUMA_WEIGHTS.iter().sum::<u32>(), LUMA_SCALE);
        // Black and full scale; a level on the straight toe; a level on the curve.
        assert_eq!(linear_light(0), 0.0);
        assert_eq!(linear_light(255), 1.0);
        assert!((linear_light(10) - 10.0 / 255.0 / SRGB_TOE_SLOPE).abs() < 1e-15);
        assert!((linear_light(128) - 0.215_860_5).abs() < 1e-6);
        // The light rises at every level, so every star pixel adds light over its background.
        let table = linear_table();
        assert_eq!(table.windows(2).filter(|p| p[1] <= p[0]).count(), 0);
        // The test's encoder is the inverse of the decoder at every level.
        let back: Vec<u8> = table.iter().map(|&v| srgb_level(v)).collect();
        assert_eq!(back, (0..=255).collect::<Vec<u8>>());
    }

    /// The thresholds follow the star law's own numbers (`StarTuning`): a change of the law turns
    /// this test red, and the thresholds move with it.
    #[test]
    fn the_thresholds_follow_the_star_law() {
        let law = StarTuning::default();
        // The census peak: the cull through the straight toe of the sRGB curve.
        assert_eq!(
            (SRGB_TOE_SLOPE * law.cull_level * 255.0).round(),
            f64::from(CENSUS_PEAK_LEVELS)
        );
        // The threshold: a reference star on a pixel's corner gives each pixel half its peak; less
        // half a level of rounding, that stays at the threshold or over it.
        let corner = (-(2.0 * 0.5_f64.powi(2)) / (2.0 * law.core_sigma_px.powi(2))).exp();
        assert!((corner - 0.5).abs() < 0.001);
        assert!(f64::from(CENSUS_PEAK_LEVELS) * corner - 0.5 >= f64::from(STAR_THRESHOLD_LEVELS));
        // The widest star: the crop at an amplitude of a million, in whole pixels; the nearest
        // neighbour's crop.
        let crop = |amplitude: f64| {
            law.halo_sigma_px * (2.0 * (amplitude * law.halo_weight / law.cull_level).ln()).sqrt()
        };
        assert_eq!(crop(1.0e6).floor(), STAR_MAX_RADIUS_PX);
        assert!((crop(10.9) - 4.9).abs() < 0.01);
        // The areas: the disc of the widest star, and the white core of radius 3.
        let disc_area = |r: f64| (std::f64::consts::PI * r * r) as u64;
        assert_eq!(disc_area(STAR_MAX_RADIUS_PX), STAR_MAX_AREA_PX);
        assert_eq!(disc_area(3.0), STAR_MAX_WHITE_PX);
        // The largest star covers about a quarter of the window.
        let window = (2 * BACKGROUND_WINDOW_RADIUS_PX + 1).pow(2) as u64;
        assert_eq!(window, 961);
        assert!(window > 3 * STAR_MAX_AREA_PX);
    }
}
