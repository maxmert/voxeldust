//! ★ THE GHOST RIG (the temporal path, slice T0 item 2; the gate G-GHOST) — the pure half.
//!
//! A ghost is a trail that a moving thing leaves behind it on the picture. A temporal
//! reconstruction builds each display pixel from this frame and from the frames before it. Where a
//! moving hull box uncovers the sky, the history can still hold the box's colour, and the picture
//! then shows a fading copy of the box behind it. The design
//! (`docs/investigation/2026-09-28/temporal_path_design.md`, §12.4) proposes a bound: a trail of
//! at most one display pixel behind a hull box and a dot at 240 and 528 m/s. The owner sets the
//! bound at D9. This module measures the trail on one picture.
//!
//! **The inputs.** For each frame the caller gives the picture (RGBA, eight bits a channel, row
//! after row), the object's TRUE footprint, the footprint's travel on the screen since the last
//! frame, and the background. The object is the moving thing: a hull box or a dot. The caller
//! computes the footprint from the stamped camera and the object's pose, through
//! [`CaptureCamera::project_point`](crate::camera::CaptureCamera::project_point) at the size of
//! the DISPLAY picture. A hull box gives its projected corners, and the footprint is their convex
//! hull. A dot gives its projected centre and its drawn radius. The travel is the footprint's
//! centre now minus its centre in the last frame ([`Footprint::centre`]). The caller can also give
//! the object's settled colour (see "The fade-in" below).
//!
//! **The scene.** The rig runs on the scene of the picture census (G-PICTURE-HOLES): the picture
//! clears to the census's sentinel colour, the air pass is off, and the star cloud is off. The
//! stand frames the object against that clear colour, so that no ground and no other body is
//! behind the object within the reach. Then only the background is behind the object over the
//! whole reach. The caller states the clear colour as the picture stores it
//! ([`Background::Colour`]). This module does not name the sentinel: the background is always the
//! caller's colour. The rig checks what it can. A pixel behind the object that no mix of the object
//! and the background can make (a star, the sun's glow, the horizon) makes the line unreadable
//! ([`Unreadable::ForeignBehind`]). The rig never reads such a pixel as a trail. A thing with a
//! colour inside that mix (a grey rock behind a grey hull on a grey sky) looks like a trail to the
//! rig, so the scene must keep such things out.
//!
//! **The scan lines.** Three lines ([`SCAN_SHARES`]) cross the footprint along the travel. On each
//! line the meter reads:
//! 1. the object's colour: the mean of the pixels that the footprint covers completely;
//! 2. the background: a colour that the caller states, or the mean of the pixels ahead of the
//!    leading edge ([`Background::Ahead`]), where the object was not drawn before;
//! 3. the pixels behind the trailing edge, one step of one display pixel at a time.
//!
//! **The trail.** Each pixel behind gets a BLEND SHARE ([`blend_share`]): 0 is the background, 1 is
//! the object's colour. A pixel with a share of [`TRAIL_SHARE`] or more is a trail pixel. Only a
//! pixel that the footprint does not touch can be a trail pixel: a pixel that the footprint
//! touches is part of the object's own edge. Each line reads its trail in two ways:
//! - THE RUN ([`LineTrail::run_px`]): the steps from the trailing edge to the last trail pixel of
//!   the unbroken run from that edge. This is the smear: the trail that still shows.
//! - THE FARTHEST ([`LineTrail::farthest_px`]): the steps from the trailing edge to the farthest
//!   trail pixel within the reach ([`reach_px`]). A temporal path with a wrong motion can leave a
//!   faint copy of the object at its last place, with the background between the copy and the
//!   object. The run stops at that gap. The farthest reads the far edge of the copy.
//!
//! The farthest is never shorter than the run. THE GATE READS THE FARTHEST, so on a lawful scene
//! the run and the farthest are both within the bound.
//!
//! **The fade-in (risk 24 of the design).** A new dot can fade in over a few frames. The rig takes
//! the object's colour from the same picture, so a dim dot is its own reference, and its trail
//! reads 0. So the caller can state the object's SETTLED colour ([`GhostFrame::expected`]): a
//! stated colour, or the object's colour that the rig reads on a later settled frame
//! ([`LineTrail::object`]). Then each line also reads its COVER SHARE
//! ([`LineTrail::cover_share`]): the blend share of the object's colour between the background and
//! the settled colour. The settled colour reads 1.0. A cover share under [`COVER_SHARE_MIN`] fails
//! the gate. Without a settled colour the cover share is `None`, and G-GHOST does not read risk 24.
//!
//! **The reading.** Each line reads its trail, or the reason why it cannot ([`Unreadable`]). A
//! frame's reading, and the pooled reading of a leg ([`sum_reads`]), give the longest trail, the
//! median trail and the lowest cover share. The gate passes only when every line reads, the
//! longest trail is within the bound, and no cover share is under [`COVER_SHARE_MIN`]
//! ([`GhostRead::passes`], [`GHOST_BOUND_PX`]).
//!
//! **Example.** A hull flies at 240 m/s, one kilometre from the pilot's eye, across the owner's
//! full-screen window. Its box moves about 10 display pixels a frame, so the scan behind can read
//! up to 330 steps. Today's picture (four samples a pixel) keeps no history, so its lines should
//! read a run of 0 and a farthest of 0; slice T0 measures that "before" row. If a reconstruction
//! keeps the box's colour in the history where the box uncovers the sky, the middle line reads,
//! for example, a run of 12 pixels, and the gate fails at the bound of 1. If the star cloud stays
//! on by mistake, a star 200 pixels behind the box on the middle line makes that line unreadable
//! (`ForeignBehind`). It never reads as a trail of 200.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};

use crate::camera::{ScreenAabb, ScreenPos};

/// THE TRAIL SHARE: a pixel behind the footprint is a trail pixel when its blend share toward the
/// object's colour is this share or more (an eighth of the object–background contrast).
///
/// Basis: the share must stand above the two things behind an object that are not a ghost.
/// - The reconstruction's own edge. The design weighs render samples with a Lanczos-2 weight
///   (§4.2). At the floor scale (1/3, the widest spread that the floor law of §4.4 allows) a
///   converged honest edge puts a share of at most 0.081 into the second pixel behind it (its
///   centre 1.5 display pixels, half a render pixel, from the edge), and less at every larger
///   scale. At an eighth, an honest edge reads a trail of 1 pixel or less.
/// - The picture's own noise at the smallest contrast that a line reads ([`MIN_CONTRAST_LEVELS`]).
///
/// A higher share would not see a fainter ghost.
pub const TRAIL_SHARE: f64 = 0.125;

/// THE NOISE in levels on each channel: the largest change that noise alone makes to a pixel.
///
/// Basis: the proposed flicker target (G-FLICKER, §12.4) lets an edge pixel change by 2 levels a
/// frame (and a content pixel by 1 level). [`MIN_CONTRAST_LEVELS`], [`FOREIGN_SHARE`] and
/// [`SETTLED_MIN_CONTRAST_LEVELS`] take their noise from this constant.
pub const NOISE_LEVELS: f64 = 2.0;

/// THE SMALLEST CONTRAST that a line reads, in levels of 255 on the widest channel between the
/// object's colour and the background.
///
/// Basis: noise of [`NOISE_LEVELS`] on each channel, in its worst direction, moves a blend share by
/// at most 1.37 × 2 / 32 = 0.085 at this contrast (1.37 is the largest value of
/// `|d|₁ · max|dᵢ| / |d|²` for a colour step `d`). That is less than [`TRAIL_SHARE`], so noise
/// alone never makes a trail pixel.
pub const MIN_CONTRAST_LEVELS: f64 = 32.0;

/// THE PROPOSED BOUND in display pixels: a trail of at most one pixel behind a hull box and a dot
/// at 240 and 528 m/s (the design's proposed G-GHOST target, §12.4). The owner sets the bound at
/// D9. The gate gives this constant to [`GhostRead::passes`].
pub const GHOST_BOUND_PX: u32 = 1;

/// THE SCAN LINES: where the three lines cross the footprint, as shares of its width across the
/// travel.
///
/// Basis: the middle line reads the centre of the trail, and the two quarter lines read its
/// sides. Each line is a quarter of the width inside the silhouette, so it crosses the footprint
/// well away from the footprint's sides. On a dot at its smallest drawn radius (3 display pixels,
/// [`DOT_MIN_APPARENT_RADIUS_PX`](crate::camera::DOT_MIN_APPARENT_RADIUS_PX)) the quarter lines are
/// 1.5 pixels from the centre, and they still cross pixels that the dot covers completely.
pub const SCAN_SHARES: [f64; 3] = [0.25, 0.5, 0.75];

/// THE FILTER'S REACH in display pixels: the scan behind reads at least this far, and the
/// background ahead starts after it.
///
/// Basis: the reconstruction weighs render samples up to two render pixels from a display pixel
/// (Lanczos-2, §4.2). At the floor scale (1/3) two render pixels are six display pixels. An honest
/// edge cannot spread farther than that.
pub const FILTER_REACH_PX: u32 = 6;

/// THE AHEAD SPAN: the number of steps after [`FILTER_REACH_PX`] that [`Background::Ahead`]
/// averages into the background.
///
/// Basis: the mean of 8 pixels decreases a pixel's own noise of one level (the proposed flicker
/// target for content pixels, §12.4) to about a third of a level (1/√8). The span also checks that
/// the background is even: one pixel of the span with a share of [`TRAIL_SHARE`] or more makes the
/// line unreadable ([`Unreadable::UnevenBackground`]).
pub const AHEAD_SPAN_PX: u32 = 8;

/// THE FOREIGN MARGIN, as a share of a channel's lawful range. A pixel behind the footprint is
/// FOREIGN when one of its channels is outside the lawful range by more than this share of the
/// range's width plus [`NOISE_LEVELS`]. A foreign pixel makes the line unreadable
/// ([`Unreadable::ForeignBehind`]).
///
/// The lawful range of a channel goes from its lowest to its highest level over the background,
/// the pixels on the line that the footprint covers completely, and the settled colour when the
/// caller states it ([`GhostFrame::expected`]). Basis:
/// - A mix of the object and the background is inside the range on each channel when the mix is
///   made one channel at a time on a curve that only rises (the light before the sRGB curve, or
///   the stored levels). Such a curve keeps a value between its two ends. So a trail is never
///   foreign.
/// - The reconstruction's own edge goes past the background. The step response of the Lanczos-2
///   weight goes down to −0.044 of the contrast one render pixel from the edge, at every scale (a
///   test calculates it again). An eighth of the range covers it.
/// - The tone map mixes the channels, so it can bend a mix out of the range. The eighth leaves
///   0.08 of the range for that bend. The bend is UNMEASURED.
/// - Noise adds [`NOISE_LEVELS`] on each channel. A channel where the object and the background
///   have the same level gets only this margin.
///
/// The check never makes a line pass. A foreign pixel that the margin misses reads as before: a
/// trail pixel, or the background when its share is under [`TRAIL_SHARE`].
///
/// Example: a hull box (200, 180, 90) passes the night sky (20, 30, 60). The red range goes from 20
/// to 200, and its margin is 2 + 180 / 8 = 24.5 levels. A white star (255, 255, 255) is 30.5 levels
/// past the margin, so it is foreign. The box's own edge rings down to red 12, inside the margin.
pub const FOREIGN_SHARE: f64 = TRAIL_SHARE;

/// THE HISTORY'S MOVING FLOOR: the smallest weight that the reconstruction gives the new frame
/// while a pixel moves one pixel a frame or more. The design proposes 1/16 (§4.2); the owner sets
/// it at D9. It is the basis of [`REACH_FRAMES`].
pub const MOVING_FLOOR: f64 = 1.0 / 16.0;

/// THE REACH IN FRAMES: the scan behind reads the footprint's travel of this number of frames.
///
/// Basis: a history ghost fades by at least [`MOVING_FLOOR`] each frame. A ghost at the full share
/// keeps (15/16)^32 = 0.127 after 32 frames and (15/16)^33 = 0.119 after 33 frames, which is less
/// than [`TRAIL_SHARE`]. So no history ghost shows farther behind than 33 frames of travel. A test
/// calculates this number again from the two constants.
pub const REACH_FRAMES: f64 = 33.0;

/// THE COVER TOLERANCE in display pixels: a footprint that goes less than this distance into a
/// pixel's square does not touch that pixel.
///
/// Basis: such a sliver covers at most √2/512 of the pixel (its longest chord is the diagonal), so
/// it changes the pixel by less than one level, even at the full contrast of 255 levels. The
/// tolerance also absorbs the rounding of a float when an edge is on a pixel boundary.
pub const COVER_TOLERANCE_PX: f64 = 1.0 / 512.0;

/// THE SMALLEST COVER SHARE: a line whose object shows less than this share of its settled
/// contrast has not settled, and the gate fails ([`GhostRead::passes`]).
///
/// Basis: the same eighth that makes a pixel behind the object a trail pixel ([`TRAIL_SHARE`])
/// makes a part of the object's own colour that is missing. Noise alone moves the cover share by
/// less than an eighth ([`SETTLED_MIN_CONTRAST_LEVELS`]).
///
/// Example: a pilot's dot at night shows 60 % of its settled contrast in its first frame (risk 24
/// of the design: a new dot fades in over a few frames). Its cover share is 0.6, and the gate
/// fails. The settled dot reads 1.0 and passes.
pub const COVER_SHARE_MIN: f64 = 1.0 - TRAIL_SHARE;

/// THE SMALLEST SETTLED CONTRAST, in levels on the widest channel between the settled colour and
/// the background. A line whose stated settled colour is nearer to the background does not read
/// ([`Unreadable::LowContrast`]).
///
/// Basis: a settled colour that the caller reads on a later frame brings the noise of that frame
/// too. The noise of the two frames ([`NOISE_LEVELS`] each, 4 levels together), in its worst
/// direction, moves the cover share by at most 1.37 × 4 / 64 = 0.085 at this contrast. That is the
/// share that the noise of one frame gives at [`MIN_CONTRAST_LEVELS`], and it is less than the
/// eighth that [`COVER_SHARE_MIN`] leaves.
pub const SETTLED_MIN_CONTRAST_LEVELS: f64 = 2.0 * MIN_CONTRAST_LEVELS;

/// The object's true footprint on the picture, in display pixels (the origin at the top left,
/// `+y` down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Footprint<'a> {
    /// A hull box: its projected corners, in any order. The footprint is their convex hull, which
    /// is the silhouette of the box.
    Hull(&'a [ScreenPos]),
    /// A dot: its projected centre and its drawn radius.
    Dot { centre: ScreenPos, radius_px: f64 },
}

impl Footprint<'_> {
    /// The footprint's centre: the dot's centre, or the mean of the hull's points. The travel is
    /// this centre now minus the same centre in the last frame. A hull with no points has no
    /// centre (NaN), and the meter reads such a travel as no travel.
    #[must_use]
    pub fn centre(&self) -> DVec2 {
        match self {
            Footprint::Hull(points) => {
                points
                    .iter()
                    .fold(DVec2::ZERO, |sum, p| sum + DVec2::new(p.x, p.y))
                    / points.len() as f64
            }
            Footprint::Dot { centre, .. } => DVec2::new(centre.x, centre.y),
        }
    }
}

/// The four corners of a projected rectangle, for [`Footprint::Hull`].
#[must_use]
pub fn rect_corners(rect: ScreenAabb) -> [ScreenPos; 4] {
    [
        rect.min,
        ScreenPos {
            x: rect.max.x,
            y: rect.min.y,
        },
        rect.max,
        ScreenPos {
            x: rect.min.x,
            y: rect.max.y,
        },
    ]
}

/// Where a line reads the background.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Background {
    /// The caller states the background: a plain fixture colour, or a run's sentinel clear colour.
    Colour([u8; 3]),
    /// Each line reads the background ahead of the leading edge: the mean of [`AHEAD_SPAN_PX`]
    /// steps after [`FILTER_REACH_PX`] steps.
    Ahead,
}

/// One frame for the rig.
#[derive(Clone, Copy, Debug)]
pub struct GhostFrame<'a> {
    /// The picture: RGBA, eight bits a channel, `width × height` pixels, row after row.
    pub rgba: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub footprint: Footprint<'a>,
    /// The footprint's travel since the last frame, in display pixels. Its direction is the
    /// direction of the scan lines. Its length sets the reach ([`reach_px`]).
    pub travel_px: DVec2,
    pub background: Background,
    /// THE SETTLED COLOUR of the object, in levels (0 to 255 on each channel), when the caller
    /// knows it: a stated colour (for example the colour that a fixture gives its dot), or the
    /// object's colour that the rig reads on a later settled frame ([`LineTrail::object`]). The
    /// rig then reads each line's cover share ([`LineTrail::cover_share`]). `None`: the rig reads
    /// no cover share, and G-GHOST does not read risk 24 (a dot that fades in).
    pub expected: Option<DVec3>,
}

/// Why a line reads no trail. A line that reads no trail never passes the gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Unreadable {
    /// The bytes are not one RGBA picture of the stated size, or the settled colour is not a
    /// finite colour.
    Malformed,
    /// The travel is zero or not finite: the frame has no direction for its scan lines.
    NoTravel,
    /// The footprint has no area: fewer than three hull points that are not on one line, a radius
    /// of zero or less, or a coordinate that is not finite.
    NoFootprint,
    /// The line crosses no pixel of the picture that the footprint covers completely, so the
    /// object's colour is not known.
    NoObject,
    /// The background ahead goes out of the picture, or it holds no pixel clear of the footprint.
    NoBackground,
    /// The object and the background differ by less than [`MIN_CONTRAST_LEVELS`], or the stated
    /// settled colour and the background differ by less than [`SETTLED_MIN_CONTRAST_LEVELS`].
    LowContrast,
    /// A pixel of the background ahead differs from the background's mean by a share of
    /// [`TRAIL_SHARE`] or more, so the line cannot tell a trail from the background.
    UnevenBackground,
    /// The scan behind goes out of the picture before [`FILTER_REACH_PX`] steps, or it meets no
    /// pixel clear of the footprint.
    NoRoomBehind,
    /// A pixel behind the footprint, within the reach, is not the object, not the background and
    /// not a mix of the two: it is FOREIGN ([`FOREIGN_SHARE`]). So the scene breaks the rig's
    /// requirement (a star, the sun's glow or the horizon is behind the object), and the line
    /// cannot tell a trail from that thing. Example: a white star 100 pixels behind a hull box at
    /// night makes the middle line unreadable. It never reads as a trail of 100.
    ForeignBehind,
}

/// What one line reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineTrail {
    /// THE RUN in display pixels: the steps from the trailing edge to the last trail pixel of the
    /// unbroken run from that edge, or 0. This is the smear, the trail that still shows. A pixel
    /// that the footprint touches (the object's own edge) does not stop the run. The first pixel
    /// clear of the footprint with a share under [`TRAIL_SHARE`] stops it.
    pub run_px: u32,
    /// THE FARTHEST in display pixels: the steps from the trailing edge to the farthest trail pixel
    /// within the reach, or 0 when no pixel behind is a trail pixel. It reads a detached copy of
    /// the object (a double image at the object's last place) past the gap where the run stops.
    /// It is never shorter than the run. THE GATE READS THE FARTHEST ([`GhostRead::trails`]).
    pub farthest_px: u32,
    /// The farthest reading ahead of the leading edge, within [`FILTER_REACH_PX`] steps. An honest
    /// spread (the reconstruction's filter, a glow) shows on the two sides, a ghost only behind.
    pub lead_px: u32,
    /// The steps that the scan behind read: the reach, or fewer where the picture ends.
    pub reach_px: u32,
    /// The largest blend share of the pixels behind that are clear of the footprint: the strength
    /// of the ghost. It is negative when every such pixel is past the background, away from the
    /// object's colour.
    pub peak_share: f64,
    /// The object–background contrast on the widest channel, in levels.
    pub contrast: f64,
    /// THE OBJECT'S COLOUR on the line, in levels: the mean of the pixels that the footprint
    /// covers completely. The trail's shares read against it. A caller can take it from a later
    /// settled frame as the settled colour of an earlier frame ([`GhostFrame::expected`]).
    pub object: DVec3,
    /// THE COVER SHARE: the blend share of the object's colour between the background (0) and the
    /// stated settled colour (1). A dot that shows 60 % of its settled contrast reads 0.6.
    /// `None` when the caller states no settled colour: then G-GHOST does not read risk 24.
    pub cover_share: Option<f64>,
}

/// The reading of one scan line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineRead {
    /// The line's place across the footprint ([`SCAN_SHARES`]).
    pub at: f64,
    pub read: Result<LineTrail, Unreadable>,
}

/// What a frame reads, or the pooled reading of a leg of frames: the trails and the runs of the
/// lines that read (display pixels → the number of lines), the lines that do not read, by reason,
/// and the lowest cover share. Every number comes from these counts, so the longest and the median
/// trail of a leg are one reading of the whole leg.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GhostRead {
    /// Each line of one frame, in the order of [`SCAN_SHARES`]. A pooled reading keeps no lines.
    pub lines: Vec<LineRead>,
    /// THE GATE'S TRAILS: the farthest reading of each line that reads
    /// ([`LineTrail::farthest_px`]).
    pub trails: BTreeMap<u32, u64>,
    /// The run of each line that reads ([`LineTrail::run_px`]). A run is never longer than its
    /// line's farthest reading, so the gate does not read it again.
    pub runs: BTreeMap<u32, u64>,
    pub unreadable: BTreeMap<Unreadable, u64>,
    /// THE LOWEST COVER SHARE of the lines that read one ([`LineTrail::cover_share`]), or `None`
    /// when no line reads one (no settled colour).
    pub cover_low: Option<f64>,
}

impl GhostRead {
    /// The number of lines that read a trail.
    #[must_use]
    pub fn readable(&self) -> u64 {
        self.trails.values().sum()
    }

    /// The longest trail, or `None` when no line reads.
    #[must_use]
    pub fn max_px(&self) -> Option<u32> {
        self.trails.keys().next_back().copied()
    }

    /// The median trail: the smallest trail that at least half of the lines that read do not go
    /// past. For an even count it is the lower of the two middle trails. `None` when no line reads.
    #[must_use]
    pub fn median_px(&self) -> Option<u32> {
        let need = self.readable().div_ceil(2);
        let mut seen = 0u64;
        self.trails.iter().find_map(|(trail, n)| {
            seen += n;
            (seen >= need).then_some(*trail)
        })
    }

    /// THE GATE: every line reads, at least one line reads, the longest trail (the farthest
    /// reading) is `bound_px` or less, and no cover share is under [`COVER_SHARE_MIN`]. The run
    /// of a line is never longer than its farthest reading, so the run is within the bound too. A
    /// reading with no line never passes.
    #[must_use]
    pub fn passes(&self, bound_px: u32) -> bool {
        self.unreadable.is_empty()
            & self.max_px().is_some_and(|longest| longest <= bound_px)
            & self.cover_low.is_none_or(|low| low >= COVER_SHARE_MIN)
    }
}

/// THE BLEND SHARE of a pixel: the mix `a` for which `(1 − a) · background + a · object` comes
/// nearest to the pixel (least squares over R, G and B). 0 is the background and 1 is the object's
/// colour. A pixel with a share of more than one half is nearer to the object than to the
/// background. The caller keeps the object and the background apart ([`MIN_CONTRAST_LEVELS`]).
#[must_use]
pub fn blend_share(pixel: DVec3, object: DVec3, background: DVec3) -> f64 {
    let d = object - background;
    (pixel - background).dot(d) / d.length_squared()
}

/// THE REACH in steps for a travel of `travel_px` display pixels a frame: [`REACH_FRAMES`] frames
/// of travel, and never less than [`FILTER_REACH_PX`].
#[must_use]
pub fn reach_px(travel_px: f64) -> u32 {
    ((REACH_FRAMES * travel_px).ceil() as u32).max(FILTER_REACH_PX)
}

/// How a pixel's square stands against the footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cover {
    /// The footprint covers the whole square.
    Inside,
    /// The footprint covers a part of the square.
    Edge,
    /// The square is clear of the footprint.
    Outside,
}

/// The footprint as geometry.
#[derive(Clone, Debug, PartialEq)]
enum Shape {
    /// A convex polygon whose every turn is positive, and its bounding box.
    Hull {
        v: Vec<DVec2>,
        min: DVec2,
        max: DVec2,
    },
    Disc {
        c: DVec2,
        r: f64,
    },
}

/// Adds `p` to a chain of the monotone hull: first it removes each last point that does not make a
/// positive turn toward `p`.
fn push_turn(chain: &mut Vec<DVec2>, p: DVec2) {
    while chain.len() >= 2 {
        let (a, b) = (chain[chain.len() - 2], chain[chain.len() - 1]);
        if (b - a).perp_dot(p - b) > 0.0 {
            break;
        }
        chain.pop();
    }
    chain.push(p);
}

/// The convex hull of the points (Andrew's monotone chain): every turn positive, no point on a
/// straight run. Fewer than three vertices means that the points have no area.
fn convex_hull(points: &[ScreenPos]) -> Vec<DVec2> {
    let mut pts: Vec<DVec2> = points.iter().map(|p| DVec2::new(p.x, p.y)).collect();
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts.dedup();
    let mut lower: Vec<DVec2> = Vec::with_capacity(pts.len());
    for p in &pts {
        push_turn(&mut lower, *p);
    }
    let mut upper: Vec<DVec2> = Vec::with_capacity(pts.len());
    for p in pts.iter().rev() {
        push_turn(&mut upper, *p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// The footprint's geometry, or `None` when it has no area.
fn shape_of(footprint: &Footprint<'_>) -> Option<Shape> {
    match footprint {
        Footprint::Hull(points) => {
            let finite = points.iter().all(|p| p.x.is_finite() & p.y.is_finite());
            let v = if finite {
                convex_hull(points)
            } else {
                Vec::new()
            };
            let min = v.iter().fold(DVec2::INFINITY, |m, p| m.min(*p));
            let max = v.iter().fold(DVec2::NEG_INFINITY, |m, p| m.max(*p));
            (v.len() >= 3).then_some(Shape::Hull { v, min, max })
        }
        Footprint::Dot { centre, radius_px } => {
            let c = DVec2::new(centre.x, centre.y);
            let r = *radius_px;
            (c.is_finite() & r.is_finite() & (r > 0.0)).then_some(Shape::Disc { c, r })
        }
    }
}

/// The class of a square from its two tests: clear first, then covered, else on the edge.
fn cover_of(inside: bool, outside: bool) -> Cover {
    if outside {
        Cover::Outside
    } else if inside {
        Cover::Inside
    } else {
        Cover::Edge
    }
}

impl Shape {
    /// How the square of the pixel `(x, y)` stands against the footprint, within
    /// [`COVER_TOLERANCE_PX`]. For the hull: a separating axis (the square's two axes, or an edge
    /// of the hull) makes the square clear; all four corners inside every edge make it covered.
    fn cover(&self, x: usize, y: usize) -> Cover {
        let lo = DVec2::new(x as f64, y as f64);
        let corners = [lo, lo + DVec2::X, lo + DVec2::ONE, lo + DVec2::Y];
        match self {
            Shape::Hull { v, min, max } => {
                let mut outside = (max.x <= lo.x + COVER_TOLERANCE_PX)
                    | (min.x >= lo.x + 1.0 - COVER_TOLERANCE_PX)
                    | (max.y <= lo.y + COVER_TOLERANCE_PX)
                    | (min.y >= lo.y + 1.0 - COVER_TOLERANCE_PX);
                let mut inside = true;
                for (a, b) in v.iter().zip(v.iter().cycle().skip(1)) {
                    let edge = (*b - *a).normalize();
                    // The signed distance of each corner from the edge's line, positive inside.
                    let (near, far) = corners.iter().fold(
                        (f64::INFINITY, f64::NEG_INFINITY),
                        |(near, far), k| {
                            let d = edge.perp_dot(*k - *a);
                            (near.min(d), far.max(d))
                        },
                    );
                    outside |= far <= COVER_TOLERANCE_PX;
                    inside &= near >= -COVER_TOLERANCE_PX;
                }
                cover_of(inside, outside)
            }
            Shape::Disc { c, r } => {
                let nearest = c.clamp(lo, lo + DVec2::ONE).distance(*c);
                let farthest = corners.iter().map(|k| k.distance(*c)).fold(0.0, f64::max);
                cover_of(
                    farthest <= r + COVER_TOLERANCE_PX,
                    nearest >= r - COVER_TOLERANCE_PX,
                )
            }
        }
    }

    /// The footprint's extent along the unit direction `n`: the smallest and largest `p · n`.
    fn extent(&self, n: DVec2) -> (f64, f64) {
        match self {
            Shape::Hull { v, .. } => v
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                    (lo.min(p.dot(n)), hi.max(p.dot(n)))
                }),
            Shape::Disc { c, r } => (c.dot(n) - r, c.dot(n) + r),
        }
    }

    /// THE CHORD of the line `t · m + o · n` (the travel `m`, its normal `n`): the `t` where the
    /// line goes into the footprint and the `t` where it comes out. The line crosses the footprint
    /// inside its extent across `m` (every scan line does), so a hull edge that the line crosses
    /// has its two ends on the two sides of the line.
    fn chord(&self, m: DVec2, n: DVec2, o: f64) -> (f64, f64) {
        match self {
            Shape::Hull { v, .. } => {
                let mut span = (f64::INFINITY, f64::NEG_INFINITY);
                for (a, b) in v.iter().zip(v.iter().cycle().skip(1)) {
                    let (an, bn) = (a.dot(n) - o, b.dot(n) - o);
                    if an * bn <= 0.0 {
                        let t = a.dot(m) + (b.dot(m) - a.dot(m)) * an / (an - bn);
                        span = (span.0.min(t), span.1.max(t));
                    }
                }
                span
            }
            Shape::Disc { c, r } => {
                let half = (r * r - (c.dot(n) - o).powi(2)).sqrt();
                (c.dot(m) - half, c.dot(m) + half)
            }
        }
    }
}

/// The picture that the lines read.
struct Picture<'a> {
    rgba: &'a [u8],
    width: usize,
    height: usize,
}

impl Picture<'_> {
    /// The pixel that holds the point `p`, or `None` outside the picture.
    fn pixel_at(&self, p: DVec2) -> Option<(usize, usize)> {
        let f = p.floor();
        let inside =
            (f.x >= 0.0) & (f.y >= 0.0) & (f.x < self.width as f64) & (f.y < self.height as f64);
        inside.then_some((f.x as usize, f.y as usize))
    }

    /// The pixel's colour, in levels.
    fn colour(&self, x: usize, y: usize) -> DVec3 {
        let i = (y * self.width + x) * 4;
        DVec3::new(
            f64::from(self.rgba[i]),
            f64::from(self.rgba[i + 1]),
            f64::from(self.rgba[i + 2]),
        )
    }
}

/// One frame's geometry, which its three lines share.
struct Setup<'a> {
    pic: Picture<'a>,
    shape: Shape,
    /// The travel's unit direction, and its normal.
    m: DVec2,
    n: DVec2,
    /// The footprint's extent across the travel (along `n`).
    across: (f64, f64),
    /// The picture's extent along the travel (along `m`).
    along: (f64, f64),
    reach: u32,
    background: Background,
    /// The settled colour, when the caller states it.
    expected: Option<DVec3>,
}

/// The frame's geometry, or the reason why no line of the frame can read.
fn setup<'a>(frame: &GhostFrame<'a>) -> Result<Setup<'a>, Unreadable> {
    let bytes = frame.width * frame.height * 4;
    let lost_colour = frame.expected.is_some_and(|e| !e.is_finite());
    if (bytes == 0) | (frame.rgba.len() != bytes) | lost_colour {
        return Err(Unreadable::Malformed);
    }
    let travel = frame.travel_px.length();
    let moving = (travel > 0.0) & travel.is_finite();
    if !moving {
        return Err(Unreadable::NoTravel);
    }
    let shape = shape_of(&frame.footprint).ok_or(Unreadable::NoFootprint)?;
    let m = frame.travel_px / travel;
    let n = m.perp();
    let (w, h) = (frame.width as f64, frame.height as f64);
    let along = [
        DVec2::ZERO,
        DVec2::new(w, 0.0),
        DVec2::new(0.0, h),
        DVec2::new(w, h),
    ]
    .iter()
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), k| {
        (lo.min(k.dot(m)), hi.max(k.dot(m)))
    });
    Ok(Setup {
        pic: Picture {
            rgba: frame.rgba,
            width: frame.width,
            height: frame.height,
        },
        across: shape.extent(n),
        shape,
        m,
        n,
        along,
        reach: reach_px(travel),
        background: frame.background,
        expected: frame.expected,
    })
}

/// The object's colours on one line, in levels.
struct ObjectColours {
    /// The mean of the pixels that the footprint covers completely.
    mean: DVec3,
    /// The lowest and the highest level of those pixels on each channel.
    lo: DVec3,
    hi: DVec3,
}

/// THE OBJECT'S COLOURS on the line at `o`: the pixels that the footprint covers completely along
/// the chord `[t_in, t_out]`, one sample a pixel step; `None` when the line crosses no such pixel
/// in the picture.
fn object_colours(s: &Setup<'_>, o: f64, t_in: f64, t_out: f64) -> Option<ObjectColours> {
    let end = t_out.min(s.along.1);
    let mut t = t_in.max(s.along.0) + 0.5;
    let mut seen: Vec<DVec3> = Vec::new();
    while t < end {
        let whole = s
            .pic
            .pixel_at(s.m * t + s.n * o)
            .filter(|&(x, y)| s.shape.cover(x, y) == Cover::Inside);
        if let Some((x, y)) = whole {
            seen.push(s.pic.colour(x, y));
        }
        t += 1.0;
    }
    let colours = ObjectColours {
        mean: seen.iter().sum::<DVec3>() / seen.len() as f64,
        lo: seen.iter().fold(DVec3::INFINITY, |m, c| m.min(*c)),
        hi: seen.iter().fold(DVec3::NEG_INFINITY, |m, c| m.max(*c)),
    };
    (!seen.is_empty()).then_some(colours)
}

/// THE SCAN from the point `from` along the unit direction `dir`, one step of one display pixel at
/// a time, for at most `steps` steps: `Some(colour)` for a pixel clear of the footprint, `None` for
/// a pixel that the footprint touches. The scan stops where it goes out of the picture. Each step
/// samples [`COVER_TOLERANCE_PX`] short of its full distance, so on a line along a pixel row a step
/// is in the next pixel exactly when that pixel is clear.
fn walk(s: &Setup<'_>, from: DVec2, dir: DVec2, steps: u32) -> Vec<Option<DVec3>> {
    let mut out = Vec::new();
    for k in 1..=steps {
        let Some((x, y)) = s
            .pic
            .pixel_at(from + dir * (f64::from(k) - COVER_TOLERANCE_PX))
        else {
            break;
        };
        out.push((s.shape.cover(x, y) == Cover::Outside).then_some(s.pic.colour(x, y)));
    }
    out
}

/// THE BACKGROUND and the pixels that must be even around it: the stated colour (no pixels), or
/// the mean of the ahead span's clear pixels (those pixels).
fn background_of(
    background: Background,
    ahead: &[Option<DVec3>],
) -> Result<(DVec3, Vec<DVec3>), Unreadable> {
    match background {
        Background::Colour(c) => Ok((
            DVec3::new(f64::from(c[0]), f64::from(c[1]), f64::from(c[2])),
            Vec::new(),
        )),
        Background::Ahead => {
            let full = ahead.len() == (FILTER_REACH_PX + AHEAD_SPAN_PX) as usize;
            let span: Vec<DVec3> = ahead
                .iter()
                .skip(FILTER_REACH_PX as usize)
                .flatten()
                .copied()
                .collect();
            if !full | span.is_empty() {
                return Err(Unreadable::NoBackground);
            }
            let mean = span.iter().sum::<DVec3>() / span.len() as f64;
            Ok((mean, span))
        }
    }
}

/// THE FARTHEST step of a scan whose pixel is a trail pixel (1 is the first step), or 0.
fn farthest(scan: &[Option<DVec3>], share: &dyn Fn(DVec3) -> f64) -> u32 {
    scan.iter()
        .rposition(|c| c.is_some_and(|c| share(c) >= TRAIL_SHARE))
        .map_or(0, |k| k as u32 + 1)
}

/// THE RUN of a scan: the farthest trail pixel before the first gap (a pixel clear of the
/// footprint with a share under [`TRAIL_SHARE`]), or 0. A pixel that the footprint touches is no
/// gap.
fn run(scan: &[Option<DVec3>], share: &dyn Fn(DVec3) -> f64) -> u32 {
    let gap = scan
        .iter()
        .position(|c| c.is_some_and(|c| share(c) < TRAIL_SHARE))
        .unwrap_or(scan.len());
    farthest(&scan[..gap], share)
}

/// Whether the pixel `c` is FOREIGN to the lawful range `[lo, hi]`: one of its channels is outside
/// the range by more than [`FOREIGN_SHARE`] of the range's width plus [`NOISE_LEVELS`].
fn foreign(c: DVec3, lo: DVec3, hi: DVec3) -> bool {
    let margin = (hi - lo) * FOREIGN_SHARE + NOISE_LEVELS;
    (c.cmplt(lo - margin) | c.cmpgt(hi + margin)).any()
}

/// The lower of two cover shares. A missing share takes no part.
fn lowest(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    a.into_iter().chain(b).reduce(f64::min)
}

/// THE LINE at `at` across the footprint: its trail, or the reason why it reads none.
fn read_line(s: &Setup<'_>, at: f64) -> Result<LineTrail, Unreadable> {
    let o = s.across.0 + at * (s.across.1 - s.across.0);
    let (t_in, t_out) = s.shape.chord(s.m, s.n, o);
    let object = object_colours(s, o, t_in, t_out).ok_or(Unreadable::NoObject)?;
    let point = |t: f64| s.m * t + s.n * o;
    let ahead = walk(s, point(t_out), s.m, FILTER_REACH_PX + AHEAD_SPAN_PX);
    let (background, span) = background_of(s.background, &ahead)?;
    let contrast = (object.mean - background).abs().max_element();
    let settled_faint = s
        .expected
        .is_some_and(|e| (e - background).abs().max_element() < SETTLED_MIN_CONTRAST_LEVELS);
    if (contrast < MIN_CONTRAST_LEVELS) | settled_faint {
        return Err(Unreadable::LowContrast);
    }
    let share = |c: DVec3| blend_share(c, object.mean, background);
    if span.iter().any(|c| share(*c).abs() >= TRAIL_SHARE) {
        return Err(Unreadable::UnevenBackground);
    }
    let behind = walk(s, point(t_in), -s.m, s.reach);
    let clear: Vec<DVec3> = behind.iter().flatten().copied().collect();
    if (behind.len() < FILTER_REACH_PX as usize) | clear.is_empty() {
        return Err(Unreadable::NoRoomBehind);
    }
    // The lawful range: the background, the object's covered pixels and the settled colour.
    let (lo, hi) = s.expected.map_or((object.lo, object.hi), |e| {
        (object.lo.min(e), object.hi.max(e))
    });
    let (lo, hi) = (lo.min(background), hi.max(background));
    if clear.iter().any(|c| foreign(*c, lo, hi)) {
        return Err(Unreadable::ForeignBehind);
    }
    let lead = &ahead[..ahead.len().min(FILTER_REACH_PX as usize)];
    Ok(LineTrail {
        run_px: run(&behind, &share),
        farthest_px: farthest(&behind, &share),
        lead_px: farthest(lead, &share),
        reach_px: behind.len() as u32,
        peak_share: clear
            .iter()
            .map(|c| share(*c))
            .fold(f64::NEG_INFINITY, f64::max),
        contrast,
        object: object.mean,
        cover_share: s.expected.map(|e| blend_share(object.mean, e, background)),
    })
}

/// THE JUDGE of one frame: the trail on each of the three scan lines, and the lines that cannot
/// read, by reason. A frame that cannot read at all (its bytes, its settled colour, its travel,
/// its footprint) gives the same reason on each line.
#[must_use]
pub fn judge_frame(frame: &GhostFrame<'_>) -> GhostRead {
    let reads = match setup(frame) {
        Ok(s) => SCAN_SHARES.map(|at| read_line(&s, at)),
        Err(why) => SCAN_SHARES.map(|_| Err(why)),
    };
    let mut read = GhostRead::default();
    for (at, line) in SCAN_SHARES.iter().zip(reads) {
        match line {
            Ok(trail) => {
                *read.trails.entry(trail.farthest_px).or_insert(0) += 1;
                *read.runs.entry(trail.run_px).or_insert(0) += 1;
                read.cover_low = lowest(read.cover_low, trail.cover_share);
            }
            Err(why) => *read.unreadable.entry(why).or_insert(0) += 1,
        }
        read.lines.push(LineRead {
            at: *at,
            read: line,
        });
    }
    read
}

/// The pooled reading of many frames (a leg): the counts add, the lowest cover share of the leg
/// stays, and the lines stay behind.
#[must_use]
pub fn sum_reads(reads: &[GhostRead]) -> GhostRead {
    let mut total = GhostRead::default();
    for r in reads {
        for (trail, n) in &r.trails {
            *total.trails.entry(*trail).or_insert(0) += n;
        }
        for (run, n) in &r.runs {
            *total.runs.entry(*run).or_insert(0) += n;
        }
        for (why, n) in &r.unreadable {
            *total.unreadable.entry(*why).or_insert(0) += n;
        }
        total.cover_low = lowest(total.cover_low, r.cover_low);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The night sky behind the rig (a plain fixture colour).
    const SKY: [u8; 3] = [20, 30, 60];
    /// A pure cyan sky: a clear colour like the census's sentinel, stated here as a plain fixture
    /// colour.
    const CYAN: [u8; 3] = [0, 255, 255];
    /// The hull box's drawn colour.
    const HULL: [u8; 3] = [200, 180, 90];
    /// THE HONEST EDGE: the converged edge of the design's Lanczos-2 reconstruction at the floor
    /// scale (1/3), with the edge on a pixel boundary. These are the shares of the pixels at 0.5,
    /// 1.5, 2.5, 3.5, 4.5 and 5.5 display pixels from the edge: the kernel's step response at 1/6,
    /// 1/2, 5/6, 7/6, 3/2 and 11/6 of a render pixel. Past the second pixel the edge rings below
    /// the background. A test calculates the six shares again.
    const HONEST_EDGE: [f64; 6] = [0.3381, 0.0808, -0.0338, -0.0376, -0.0113, -0.0004];
    /// The rig's picture: 48 × 20 display pixels.
    const W: usize = 48;
    const H: usize = 20;
    /// The rig's hull box: the columns 20..30 and the rows 5..15.
    const BOX: ScreenAabb = ScreenAabb {
        min: ScreenPos { x: 20.0, y: 5.0 },
        max: ScreenPos { x: 30.0, y: 15.0 },
    };
    /// Four display pixels a frame to the right.
    const RIGHT: DVec2 = DVec2::new(4.0, 0.0);

    fn level(c: [u8; 3]) -> DVec3 {
        DVec3::new(f64::from(c[0]), f64::from(c[1]), f64::from(c[2]))
    }

    fn canvas_of(w: usize, h: usize, sky: [u8; 3]) -> Vec<u8> {
        [sky[0], sky[1], sky[2], 255].repeat(w * h)
    }

    fn canvas(w: usize, h: usize) -> Vec<u8> {
        canvas_of(w, h, SKY)
    }

    fn paint(rgba: &mut [u8], w: usize, x: usize, y: usize, c: [u8; 3]) {
        let i = (y * w + x) * 4;
        rgba[i..i + 3].copy_from_slice(&c);
    }

    /// The colour `share` of the way from the sky to the hull, rounded to a level.
    fn mix(share: f64) -> [u8; 3] {
        let m = |s: u8, o: u8| (f64::from(s) + share * (f64::from(o) - f64::from(s))).round() as u8;
        [m(SKY[0], HULL[0]), m(SKY[1], HULL[1]), m(SKY[2], HULL[2])]
    }

    /// The rig's picture: the sky, the box painted `object`, and `behind[i]` painted `i + 1`
    /// pixels behind its trailing (left) edge on every row of the box.
    fn rig_with(object: [u8; 3], behind: &[[u8; 3]]) -> Vec<u8> {
        let mut rgba = canvas(W, H);
        for y in 5..15 {
            for x in 20..30 {
                paint(&mut rgba, W, x, y, object);
            }
            for (i, c) in behind.iter().enumerate() {
                paint(&mut rgba, W, 19 - i, y, *c);
            }
        }
        rgba
    }

    /// The rig's picture with a trail of the stated shares behind the hull box.
    fn rig(trail: &[f64]) -> Vec<u8> {
        let behind: Vec<[u8; 3]> = trail.iter().map(|s| mix(*s)).collect();
        rig_with(HULL, &behind)
    }

    fn frame<'a>(
        rgba: &'a [u8],
        footprint: Footprint<'a>,
        travel: DVec2,
        background: Background,
    ) -> GhostFrame<'a> {
        GhostFrame {
            rgba,
            width: W,
            height: H,
            footprint,
            travel_px: travel,
            background,
            expected: None,
        }
    }

    /// Each line's trail (its farthest reading, which the gate reads), or its reason.
    fn trails(read: &GhostRead) -> Vec<Result<u32, Unreadable>> {
        read.lines
            .iter()
            .map(|l| l.read.map(|t| t.farthest_px))
            .collect()
    }

    /// Each line's run, or its reason.
    fn runs(read: &GhostRead) -> Vec<Result<u32, Unreadable>> {
        read.lines
            .iter()
            .map(|l| l.read.map(|t| t.run_px))
            .collect()
    }

    /// Each line's cover share, or its reason.
    fn covers(read: &GhostRead) -> Vec<Result<Option<f64>, Unreadable>> {
        read.lines
            .iter()
            .map(|l| l.read.map(|t| t.cover_share))
            .collect()
    }

    /// Each line's lead.
    fn leads(read: &GhostRead) -> Vec<Result<u32, Unreadable>> {
        read.lines
            .iter()
            .map(|l| l.read.map(|t| t.lead_px))
            .collect()
    }

    /// The rig's box, judged with a stated background and with the background read ahead.
    fn judge_both(rgba: &[u8], travel: DVec2) -> [GhostRead; 2] {
        let corners = rect_corners(BOX);
        [Background::Colour(SKY), Background::Ahead]
            .map(|bg| judge_frame(&frame(rgba, Footprint::Hull(&corners), travel, bg)))
    }

    /// THE CONTROL "a 3-pixel trail": a hull box moving right with a 3-pixel fading trail reads 3
    /// on every line (the run and the farthest), against the stated sky and against the sky read
    /// ahead. The gate fails. With no settled colour the lines read no cover share.
    #[test]
    fn the_three_pixel_trail_control_reads_three() {
        let rgba = rig(&[0.75, 0.5, 0.25]);
        for read in judge_both(&rgba, RIGHT) {
            assert_eq!(trails(&read), vec![Ok(3), Ok(3), Ok(3)]);
            assert_eq!(runs(&read), vec![Ok(3), Ok(3), Ok(3)]);
            assert_eq!(covers(&read), vec![Ok(None), Ok(None), Ok(None)]);
            assert_eq!(read.cover_low, None);
            assert_eq!(leads(&read), vec![Ok(0), Ok(0), Ok(0)]);
            assert_eq!(read.max_px(), Some(3));
            assert_eq!(read.median_px(), Some(3));
            assert_eq!(read.readable(), 3);
            assert!(!read.passes(GHOST_BOUND_PX));
            let ats: Vec<f64> = read.lines.iter().map(|l| l.at).collect();
            assert_eq!(ats, SCAN_SHARES.to_vec());
            let middle = read.lines[1].read.expect("the middle line reads");
            // The scan behind goes from the edge at x = 20 to the picture's edge: 20 steps.
            assert_eq!(middle.reach_px, 20);
            let strongest = blend_share(level(mix(0.75)), level(HULL), level(SKY));
            assert!((middle.peak_share - strongest).abs() < 1e-12);
            assert!((middle.contrast - 180.0).abs() < 1e-12);
            assert_eq!(middle.object, level(HULL));
        }
    }

    /// THE CONTROL "a clean moving box": the same box with no trail reads 0, and the gate passes.
    #[test]
    fn a_clean_moving_box_reads_zero_and_passes() {
        let rgba = rig(&[]);
        for read in judge_both(&rgba, RIGHT) {
            assert_eq!(trails(&read), vec![Ok(0), Ok(0), Ok(0)]);
            assert_eq!(runs(&read), vec![Ok(0), Ok(0), Ok(0)]);
            assert_eq!((read.max_px(), read.median_px()), (Some(0), Some(0)));
            assert!(read.passes(GHOST_BOUND_PX));
            let middle = read.lines[1].read.expect("the middle line reads");
            assert!(middle.peak_share.abs() < 1e-12);
        }
    }

    /// A dot of radius 3 (the smallest drawn radius) at `centre` on a `w × h` sky: every pixel that
    /// the disc overlaps is painted `colour`, and `trail[i]` (a share from the sky to the hull) is
    /// painted `i + 1` pixels left of each row's first dot pixel.
    fn dot_painted(w: usize, h: usize, centre: DVec2, colour: [u8; 3], trail: &[f64]) -> Vec<u8> {
        let mut rgba = canvas(w, h);
        for y in 0..h {
            let mut first: Option<usize> = None;
            for x in 0..w {
                let lo = DVec2::new(x as f64, y as f64);
                if centre.clamp(lo, lo + DVec2::ONE).distance(centre) < 3.0 {
                    paint(&mut rgba, w, x, y, colour);
                    first = first.or(Some(x));
                }
            }
            if let Some(x0) = first {
                for (i, share) in trail.iter().enumerate() {
                    paint(&mut rgba, w, x0 - 1 - i, y, mix(*share));
                }
            }
        }
        rgba
    }

    /// The hull-coloured dot at `centre`, with `trail` behind it.
    fn dot_picture_at(w: usize, h: usize, centre: DVec2, trail: &[f64]) -> Vec<u8> {
        dot_painted(w, h, centre, HULL, trail)
    }

    /// The rig's dot at (20, 10).
    fn dot_picture(trail: &[f64]) -> Vec<u8> {
        dot_picture_at(W, H, DVec2::new(20.0, 10.0), trail)
    }

    fn dot_at(x: f64, y: f64) -> Footprint<'static> {
        Footprint::Dot {
            centre: ScreenPos { x, y },
            radius_px: 3.0,
        }
    }

    fn dot() -> Footprint<'static> {
        dot_at(20.0, 10.0)
    }

    /// THE CONTROL "a dot with a 2-pixel trail": it reads 2 on every line; a clean dot reads 0.
    #[test]
    fn a_dot_with_a_two_pixel_trail_reads_two() {
        let travel = DVec2::new(5.0, 0.0);
        let smeared = dot_picture(&[2.0 / 3.0, 1.0 / 3.0]);
        let clean = dot_picture(&[]);
        for bg in [Background::Colour(SKY), Background::Ahead] {
            let read = judge_frame(&frame(&smeared, dot(), travel, bg));
            assert_eq!(trails(&read), vec![Ok(2), Ok(2), Ok(2)]);
            assert!(!read.passes(GHOST_BOUND_PX));
            let read = judge_frame(&frame(&clean, dot(), travel, bg));
            assert_eq!(trails(&read), vec![Ok(0), Ok(0), Ok(0)]);
            assert!(read.passes(GHOST_BOUND_PX));
        }
    }

    /// A double image: the box's copy one frame behind (two pixels at 0.6, four pixels back), with
    /// the sky between. The farthest reads the copy's far edge, 6, never the clear sky next to the
    /// box. The run stops at that sky and reads 0: only the farthest sees a detached copy.
    #[test]
    fn a_detached_copy_reads_its_far_edge() {
        let rgba = rig(&[0.0, 0.0, 0.0, 0.0, 0.6, 0.6]);
        for read in judge_both(&rgba, RIGHT) {
            assert_eq!(trails(&read), vec![Ok(6), Ok(6), Ok(6)]);
            assert_eq!(runs(&read), vec![Ok(0), Ok(0), Ok(0)]);
            assert!(!read.passes(GHOST_BOUND_PX));
        }
    }

    /// THE TWO READINGS: a smear of two pixels, two pixels of sky, then a faint copy at 0.6. The run
    /// reads the smear (2), and the farthest reads the copy's far edge (5); the gate reads 5. A
    /// smear longer than the reach fills the reach on the two readings: a slow box (a reach of 6
    /// steps) with a smear of 8 pixels reads 6 and 6.
    #[test]
    fn the_run_reads_the_smear_and_the_farthest_reads_the_copy() {
        let rgba = rig(&[0.75, 0.5, 0.0, 0.0, 0.6]);
        for read in judge_both(&rgba, RIGHT) {
            assert_eq!(runs(&read), vec![Ok(2), Ok(2), Ok(2)]);
            assert_eq!(trails(&read), vec![Ok(5), Ok(5), Ok(5)]);
            assert_eq!(read.runs, BTreeMap::from([(2, 3)]));
            assert_eq!(read.trails, BTreeMap::from([(5, 3)]));
            assert_eq!(read.max_px(), Some(5));
            assert!(!read.passes(GHOST_BOUND_PX));
            assert!(read.passes(5));
        }
        let [slow, _] = judge_both(&rig(&[0.5; 8]), DVec2::new(0.1, 0.0));
        assert_eq!(runs(&slow), vec![Ok(6), Ok(6), Ok(6)]);
        assert_eq!(trails(&slow), vec![Ok(6), Ok(6), Ok(6)]);
    }

    /// THE SHARE IS THE LINE: a trail pixel at 0.11 of the contrast is not a trail, one at 0.14
    /// is; a pixel darker than the sky (away from the hull, as far as the honest edge rings) is
    /// never a trail. When every pixel behind is that dark, the peak share is negative.
    #[test]
    fn the_trail_share_decides_what_a_trail_is() {
        let under = mix(0.11);
        let over = mix(0.14);
        assert!(blend_share(level(under), level(HULL), level(SKY)) < TRAIL_SHARE);
        assert!(blend_share(level(over), level(HULL), level(SKY)) >= TRAIL_SHARE);
        for read in judge_both(&rig_with(HULL, &[under]), RIGHT) {
            assert_eq!(trails(&read), vec![Ok(0), Ok(0), Ok(0)]);
        }
        for read in judge_both(&rig_with(HULL, &[over]), RIGHT) {
            assert_eq!(trails(&read), vec![Ok(1), Ok(1), Ok(1)]);
        }
        let dark = mix(DEEPEST_RING);
        let [read, _] = judge_both(&rig_with(HULL, &[dark; 20]), RIGHT);
        assert_eq!(trails(&read), vec![Ok(0), Ok(0), Ok(0)]);
        let middle = read.lines[1].read.expect("the middle line reads");
        let darker = blend_share(level(dark), level(HULL), level(SKY));
        assert!(darker < 0.0);
        assert!((middle.peak_share - darker).abs() < 1e-12);
    }

    /// THE BASIS OF THE SHARE: the honest edge ([`HONEST_EDGE`]) shows on the two sides of the
    /// box: its first pixel is over the trail share, its second pixel is under it, and past that it
    /// rings below the sky. The line reads a run of 1, a farthest of 1 and a lead of 1, within the
    /// bound. The ringing is never foreign.
    #[test]
    fn an_honest_edge_at_the_floor_scale_reads_one_on_the_two_sides() {
        let mut rgba = rig(&HONEST_EDGE);
        for y in 5..15 {
            for (i, share) in HONEST_EDGE.iter().enumerate() {
                paint(&mut rgba, W, 30 + i, y, mix(*share));
            }
        }
        for read in judge_both(&rgba, RIGHT) {
            assert_eq!(trails(&read), vec![Ok(1), Ok(1), Ok(1)]);
            assert_eq!(runs(&read), vec![Ok(1), Ok(1), Ok(1)]);
            assert_eq!(leads(&read), vec![Ok(1), Ok(1), Ok(1)]);
            assert!(read.passes(GHOST_BOUND_PX));
        }
    }

    /// The Lanczos-2 weight at `x` render pixels, for `0 < x < 2`.
    fn lanczos2(x: f64) -> f64 {
        let sinc = |x: f64| (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x);
        sinc(x) * sinc(x / 2.0)
    }

    /// The integral of the Lanczos-2 weight from `a` to 2 render pixels, for `a ≥ 0`, by the
    /// midpoint rule (no sample falls on 0).
    fn lanczos2_tail(a: f64) -> f64 {
        let steps = 20_000u32;
        let h = (2.0 - a) / f64::from(steps);
        (0..steps)
            .map(|i| lanczos2(a + (f64::from(i) + 0.5) * h))
            .sum::<f64>()
            * h
    }

    /// The step response of the Lanczos-2 weight `y` render pixels from an edge: the part of the
    /// kernel's weight on the object's side of the edge.
    fn step_response(y: f64) -> f64 {
        lanczos2_tail(y) / (2.0 * lanczos2_tail(0.0))
    }

    /// The deepest ringing of the honest edge, one render pixel from the edge.
    const DEEPEST_RING: f64 = -0.0444;

    /// THE BASIS OF THE FOREIGN MARGIN: the step response of the Lanczos-2 weight gives the honest
    /// edge again (its six shares), and its deepest ringing (−0.044, one render pixel from the
    /// edge, where the weight changes sign). The margin's share covers that ringing. On the rig's
    /// colours the deepest ringing is 8 red levels under the sky, inside the red margin of 24.5.
    #[test]
    fn the_foreign_margin_covers_the_edges_own_ringing() {
        let at = [1.0 / 6.0, 0.5, 5.0 / 6.0, 7.0 / 6.0, 1.5, 11.0 / 6.0];
        for (y, share) in at.iter().zip(HONEST_EDGE) {
            assert!((step_response(*y) - share).abs() < 1e-4);
        }
        assert!((step_response(1.0) - DEEPEST_RING).abs() < 1e-4);
        assert!(step_response(0.95) > step_response(1.0));
        assert!(step_response(1.05) > step_response(1.0));
        assert!(DEEPEST_RING.abs() < FOREIGN_SHARE);
        let (sky, hull) = (level(SKY), level(HULL));
        let ring = level(mix(DEEPEST_RING));
        assert_eq!(ring, DVec3::new(12.0, 23.0, 59.0));
        assert!(!foreign(ring, sky.min(hull), sky.max(hull)));
    }

    /// THE BASIS OF THE CONTRAST: noise of 2 levels on every channel, in its worst direction, at
    /// the smallest contrast (32 levels) is not a trail (a share of 0.085); one level less of
    /// contrast and the line does not read.
    #[test]
    fn noise_at_the_smallest_contrast_is_never_a_trail() {
        let faint = [SKY[0] + 32, SKY[1] + 12, SKY[2] + 12];
        let noise = [SKY[0] + 2, SKY[1] + 2, SKY[2] + 2];
        let share = blend_share(level(noise), level(faint), level(SKY));
        assert!((share - 112.0 / 1312.0).abs() < 1e-12);
        for read in judge_both(&rig_with(faint, &[noise; 4]), RIGHT) {
            assert_eq!(trails(&read), vec![Ok(0), Ok(0), Ok(0)]);
        }
        let fainter = [SKY[0] + 31, SKY[1] + 12, SKY[2] + 12];
        for read in judge_both(&rig_with(fainter, &[]), RIGHT) {
            assert_eq!(trails(&read), vec![Err(Unreadable::LowContrast); 3]);
        }
    }

    /// The blend share is the least-squares mix: the sky is 0, the hull is 1, the midpoint is one
    /// half, and a colour off the sky–hull line reads its projection on that line.
    #[test]
    fn the_blend_share_is_the_least_squares_mix() {
        let (sky, hull) = (level(SKY), level(HULL));
        assert_eq!(blend_share(sky, hull, sky), 0.0);
        assert_eq!(blend_share(hull, hull, sky), 1.0);
        assert!((blend_share((sky + hull) * 0.5, hull, sky) - 0.5).abs() < 1e-12);
        let d = hull - sky;
        let off_line = sky + d * 0.25 + d.normalize().any_orthonormal_vector() * 40.0;
        assert!((blend_share(off_line, hull, sky) - 0.25).abs() < 1e-12);
    }

    /// THE REACH follows the travel: 33 frames of it, never less than the filter's reach; a travel
    /// that is not a number reads the floor, an endless one reads to the picture's edge. The reach
    /// in frames comes again from the history's floor and the trail share.
    #[test]
    fn the_reach_follows_the_travel_and_the_history_floor() {
        assert_eq!(reach_px(0.1), FILTER_REACH_PX);
        assert_eq!(reach_px(4.0), 132);
        assert_eq!(reach_px(10.0), 330);
        assert_eq!(reach_px(f64::NAN), FILTER_REACH_PX);
        assert_eq!(reach_px(f64::INFINITY), u32::MAX);
        let keep = 1.0 - MOVING_FLOOR;
        assert_eq!((TRAIL_SHARE.ln() / keep.ln()).ceil(), REACH_FRAMES);
        assert!(keep.powf(REACH_FRAMES - 1.0) >= TRAIL_SHARE);
        assert!(keep.powf(REACH_FRAMES) < TRAIL_SHARE);
    }

    /// A slow box reads only the filter's reach behind it: a copy 8 pixels back is out of it.
    #[test]
    fn a_slow_box_reads_only_the_filters_reach() {
        let rgba = rig(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]);
        let [stated, _] = judge_both(&rgba, DVec2::new(0.1, 0.0));
        assert_eq!(trails(&stated), vec![Ok(0), Ok(0), Ok(0)]);
        let middle = stated.lines[1].read.expect("the middle line reads");
        assert_eq!(middle.reach_px, FILTER_REACH_PX);
        let [fast, _] = judge_both(&rgba, RIGHT);
        assert_eq!(trails(&fast), vec![Ok(8), Ok(8), Ok(8)]);
    }

    /// The travel sets the direction: a box moving left trails on its right, and a box moving down
    /// trails above it.
    #[test]
    fn the_travel_sets_the_direction_of_the_lines() {
        let mut left = rig(&[]);
        for y in 5..15 {
            for (i, share) in [0.75, 0.5, 0.25].iter().enumerate() {
                paint(&mut left, W, 30 + i, y, mix(*share));
            }
        }
        for read in judge_both(&left, DVec2::new(-4.0, 0.0)) {
            assert_eq!(trails(&read), vec![Ok(3), Ok(3), Ok(3)]);
        }
        // The rig turned on its side: 20 × 48, the box over the columns 5..15 and the rows 20..30.
        let (w, h) = (H, W);
        let mut down = canvas(w, h);
        for x in 5..15 {
            for y in 20..30 {
                paint(&mut down, w, x, y, HULL);
            }
            for (i, share) in [0.75, 0.5].iter().enumerate() {
                paint(&mut down, w, x, 19 - i, mix(*share));
            }
        }
        let corners = [
            ScreenPos { x: 5.0, y: 20.0 },
            ScreenPos { x: 15.0, y: 20.0 },
            ScreenPos { x: 15.0, y: 30.0 },
            ScreenPos { x: 5.0, y: 30.0 },
        ];
        for bg in [Background::Colour(SKY), Background::Ahead] {
            let read = judge_frame(&GhostFrame {
                rgba: &down,
                width: w,
                height: h,
                footprint: Footprint::Hull(&corners),
                travel_px: DVec2::new(0.0, 4.0),
                background: bg,
                expected: None,
            });
            assert_eq!(trails(&read), vec![Ok(2), Ok(2), Ok(2)]);
        }
    }

    /// A diamond (a box turned 45°) at (24, 12) with a half-diagonal of 8, painted on every pixel it
    /// overlaps, and `trail` shares painted left of each row's first diamond pixel.
    fn diamond_picture(trail: &[f64]) -> Vec<u8> {
        let centre = DVec2::new(24.0, 12.0);
        let (w, h) = (56, 26);
        let mut rgba = canvas(w, h);
        for y in 0..h {
            let mut first: Option<usize> = None;
            for x in 0..w {
                let lo = DVec2::new(x as f64, y as f64);
                let d = (centre.clamp(lo, lo + DVec2::ONE) - centre).abs();
                if d.x + d.y < 8.0 {
                    paint(&mut rgba, w, x, y, HULL);
                    first = first.or(Some(x));
                }
            }
            if let Some(x0) = first {
                for (i, share) in trail.iter().enumerate() {
                    paint(&mut rgba, w, x0 - 1 - i, y, mix(*share));
                }
            }
        }
        rgba
    }

    /// A SLANTED EDGE IS NEVER A TRAIL: a diamond moving right, its edge pixels painted with the
    /// full hull colour, reads 0 — a pixel that the footprint touches is the object's own edge. A
    /// trail of two pixels behind it reads its distance from the edge along each line: 3 on the
    /// upper quarter line (the first pixel behind the edge there is an edge pixel), 2 on the
    /// middle line and on the lower quarter line. The edge pixel does not stop the run, so the
    /// run reads the same.
    #[test]
    fn a_slanted_edge_is_never_a_trail() {
        let corners = [
            ScreenPos { x: 24.0, y: 4.0 },
            ScreenPos { x: 32.0, y: 12.0 },
            ScreenPos { x: 24.0, y: 20.0 },
            ScreenPos { x: 16.0, y: 12.0 },
        ];
        for (trail, expected) in [
            (vec![], vec![Ok(0), Ok(0), Ok(0)]),
            (vec![0.5, 0.5], vec![Ok(3), Ok(2), Ok(2)]),
        ] {
            let rgba = diamond_picture(&trail);
            for bg in [Background::Colour(SKY), Background::Ahead] {
                let read = judge_frame(&GhostFrame {
                    rgba: &rgba,
                    width: 56,
                    height: 26,
                    footprint: Footprint::Hull(&corners),
                    travel_px: RIGHT,
                    background: bg,
                    expected: None,
                });
                assert_eq!(trails(&read), expected);
                assert_eq!(runs(&read), expected);
            }
        }
    }

    /// A dot that moves along a diagonal, painted clean on every pixel it overlaps, reads 0: the
    /// scan steps along the diagonal and never reads the dot's own staircase edge as a trail.
    #[test]
    fn a_clean_dot_on_a_diagonal_reads_zero() {
        let rgba = dot_picture_at(48, 48, DVec2::new(24.0, 24.0), &[]);
        for background in [Background::Colour(SKY), Background::Ahead] {
            for travel_px in [DVec2::new(3.0, 3.0), DVec2::new(-2.0, 5.0)] {
                let read = judge_frame(&GhostFrame {
                    rgba: &rgba,
                    width: 48,
                    height: 48,
                    footprint: dot_at(24.0, 24.0),
                    travel_px,
                    background,
                    expected: None,
                });
                assert_eq!(trails(&read), vec![Ok(0), Ok(0), Ok(0)]);
            }
        }
    }

    /// A projected box gives eight corners in any order; the hull of them is the silhouette, and
    /// the reading is the rectangle's. The centre of the eight points gives the travel.
    #[test]
    fn eight_projected_corners_read_as_their_silhouette() {
        let rgba = rig(&[0.75, 0.5, 0.25]);
        let points = [
            ScreenPos { x: 24.0, y: 8.0 },
            ScreenPos { x: 30.0, y: 15.0 },
            ScreenPos { x: 26.0, y: 8.0 },
            ScreenPos { x: 20.0, y: 5.0 },
            ScreenPos { x: 26.0, y: 12.0 },
            ScreenPos { x: 20.0, y: 15.0 },
            ScreenPos { x: 24.0, y: 12.0 },
            ScreenPos { x: 30.0, y: 5.0 },
        ];
        let read = judge_frame(&frame(
            &rgba,
            Footprint::Hull(&points),
            RIGHT,
            Background::Ahead,
        ));
        assert_eq!(trails(&read), vec![Ok(3), Ok(3), Ok(3)]);
        assert_eq!(Footprint::Hull(&points).centre(), DVec2::new(25.0, 10.0));
        assert_eq!(dot().centre(), DVec2::new(20.0, 10.0));
        let later = [ScreenPos { x: 29.0, y: 10.0 }];
        let travel = Footprint::Hull(&later).centre() - Footprint::Hull(&points).centre();
        assert_eq!(travel, RIGHT);
        assert!(Footprint::Hull(&[]).centre().is_nan());
    }

    /// The hull keeps only the silhouette's corners, every turn positive, and drops a point on a
    /// straight run and a repeated point; points on one line have no area.
    #[test]
    fn the_hull_keeps_the_silhouette_with_positive_turns() {
        let p = |x: f64, y: f64| ScreenPos { x, y };
        let hull = convex_hull(&[
            p(0.0, 0.0),
            p(4.0, 0.0),
            p(2.0, 0.0),
            p(4.0, 4.0),
            p(0.0, 4.0),
            p(2.0, 2.0),
            p(4.0, 4.0),
        ]);
        assert_eq!(
            hull,
            vec![
                DVec2::new(0.0, 0.0),
                DVec2::new(4.0, 0.0),
                DVec2::new(4.0, 4.0),
                DVec2::new(0.0, 4.0),
            ]
        );
        assert!(convex_hull(&[p(0.0, 0.0), p(1.0, 1.0), p(3.0, 3.0)]).len() < 3);
        assert!(convex_hull(&[p(1.0, 1.0)]).is_empty());
        assert!(convex_hull(&[]).is_empty());
    }

    /// The cover classes of a pixel square: inside, on the edge, clear — for the hull and for the
    /// disc, and clear within the tolerance.
    #[test]
    fn a_pixel_square_is_inside_on_the_edge_or_clear() {
        let corners = rect_corners(BOX);
        let hull = shape_of(&Footprint::Hull(&corners)).expect("the box has an area");
        assert_eq!(hull.cover(24, 9), Cover::Inside);
        assert_eq!(hull.cover(19, 9), Cover::Outside);
        assert_eq!(hull.cover(24, 20), Cover::Outside);
        let tilted = [
            ScreenPos { x: 24.0, y: 4.0 },
            ScreenPos { x: 32.0, y: 12.0 },
            ScreenPos { x: 24.0, y: 20.0 },
            ScreenPos { x: 16.0, y: 12.0 },
        ];
        let diamond = shape_of(&Footprint::Hull(&tilted)).expect("the diamond has an area");
        assert_eq!(diamond.cover(19, 8), Cover::Edge);
        // Only a corner of this square touches the diamond: it is clear.
        assert_eq!(diamond.cover(18, 8), Cover::Outside);
        // A square beyond the diamond's corner: clear by the square's own axes.
        assert_eq!(diamond.cover(32, 12), Cover::Outside);
        let sliver = [
            ScreenPos {
                x: 20.0 - COVER_TOLERANCE_PX / 2.0,
                y: 5.0,
            },
            ScreenPos { x: 30.0, y: 5.0 },
            ScreenPos { x: 30.0, y: 15.0 },
            ScreenPos {
                x: 20.0 - COVER_TOLERANCE_PX / 2.0,
                y: 15.0,
            },
        ];
        let within = shape_of(&Footprint::Hull(&sliver)).expect("an area");
        assert_eq!(within.cover(19, 9), Cover::Outside);
        let disc = shape_of(&dot()).expect("the dot has an area");
        assert_eq!(disc.cover(19, 9), Cover::Inside);
        assert_eq!(disc.cover(17, 9), Cover::Edge);
        assert_eq!(disc.cover(16, 10), Cover::Outside);
    }

    /// THE WHOLE FRAME CANNOT READ: its bytes, its settled colour, its travel or its footprint;
    /// each line says why, and the gate never passes.
    #[test]
    fn a_frame_that_cannot_read_says_why_on_every_line() {
        let rgba = rig(&[]);
        let corners = rect_corners(BOX);
        let hull = Footprint::Hull(&corners);
        let bg = Background::Colour(SKY);
        let whole = |read: GhostRead, why: Unreadable| {
            assert_eq!(trails(&read), vec![Err(why); 3]);
            assert_eq!(read.unreadable.get(&why), Some(&3));
            assert!(!read.passes(GHOST_BOUND_PX));
            assert_eq!((read.max_px(), read.median_px()), (None, None));
        };
        whole(
            judge_frame(&frame(&rgba[..8], hull, RIGHT, bg)),
            Unreadable::Malformed,
        );
        let mut empty = frame(&[], hull, RIGHT, bg);
        empty.width = 0;
        whole(judge_frame(&empty), Unreadable::Malformed);
        // A settled colour with a channel that is not a number is not a colour.
        let mut lost = frame(&rgba, hull, RIGHT, bg);
        lost.expected = Some(DVec3::new(f64::NAN, 180.0, 90.0));
        whole(judge_frame(&lost), Unreadable::Malformed);
        for travel in [DVec2::ZERO, DVec2::NAN, DVec2::new(f64::INFINITY, 0.0)] {
            whole(
                judge_frame(&frame(&rgba, hull, travel, bg)),
                Unreadable::NoTravel,
            );
        }
        let two = [ScreenPos { x: 1.0, y: 1.0 }, ScreenPos { x: 9.0, y: 9.0 }];
        let lost = [
            ScreenPos { x: 20.0, y: 5.0 },
            ScreenPos {
                x: f64::NAN,
                y: 5.0,
            },
            ScreenPos { x: 30.0, y: 15.0 },
        ];
        let flat = |radius_px: f64, x: f64| Footprint::Dot {
            centre: ScreenPos { x, y: 10.0 },
            radius_px,
        };
        for footprint in [
            Footprint::Hull(&two),
            Footprint::Hull(&lost),
            flat(0.0, 20.0),
            flat(f64::INFINITY, 20.0),
            flat(3.0, f64::NAN),
        ] {
            whole(
                judge_frame(&frame(&rgba, footprint, RIGHT, bg)),
                Unreadable::NoFootprint,
            );
        }
    }

    /// EACH LINE CANNOT READ for its own reason: a footprint thinner than a pixel or off the
    /// picture (no object), a box at the picture's right side with the background read ahead (no
    /// background), a stripe in the ahead span (an uneven background), a box at the picture's left
    /// side (no room behind).
    #[test]
    fn a_line_that_cannot_read_says_why() {
        let rgba = rig(&[]);
        let bg = Background::Ahead;
        let thin = [
            ScreenPos { x: 20.0, y: 5.0 },
            ScreenPos { x: 30.0, y: 5.0 },
            ScreenPos { x: 30.0, y: 5.5 },
            ScreenPos { x: 20.0, y: 5.5 },
        ];
        let above = [
            ScreenPos { x: 20.0, y: -15.0 },
            ScreenPos { x: 30.0, y: -15.0 },
            ScreenPos { x: 30.0, y: -5.0 },
            ScreenPos { x: 20.0, y: -5.0 },
        ];
        for points in [&thin, &above] {
            let read = judge_frame(&frame(&rgba, Footprint::Hull(points), RIGHT, bg));
            assert_eq!(trails(&read), vec![Err(Unreadable::NoObject); 3]);
        }
        // The rig moved to the right side: its leading edge 6 pixels from the picture's edge.
        let right_side = [
            ScreenPos { x: 32.0, y: 5.0 },
            ScreenPos { x: 42.0, y: 5.0 },
            ScreenPos { x: 42.0, y: 15.0 },
            ScreenPos { x: 32.0, y: 15.0 },
        ];
        let mut near_edge = canvas(W, H);
        for y in 5..15 {
            for x in 32..42 {
                paint(&mut near_edge, W, x, y, HULL);
            }
        }
        let fp = Footprint::Hull(&right_side);
        let read = judge_frame(&frame(&near_edge, fp, RIGHT, bg));
        assert_eq!(trails(&read), vec![Err(Unreadable::NoBackground); 3]);
        assert_eq!(read.unreadable.get(&Unreadable::NoBackground), Some(&3));
        // With the sky stated, the same frame reads: the lead reads as far as the picture goes.
        let read = judge_frame(&frame(&near_edge, fp, RIGHT, Background::Colour(SKY)));
        assert_eq!(trails(&read), vec![Ok(0), Ok(0), Ok(0)]);
        // A stripe at the ahead span's third pixel (x = 30 + 6 + 2) makes the sky uneven; with
        // the sky stated, the stripe is only sky ahead and the line reads.
        let mut striped = rig(&[]);
        for y in 5..15 {
            paint(&mut striped, W, 38, y, mix(0.5));
        }
        let [stated, ahead] = judge_both(&striped, RIGHT);
        assert_eq!(trails(&ahead), vec![Err(Unreadable::UnevenBackground); 3]);
        assert_eq!(trails(&stated), vec![Ok(0), Ok(0), Ok(0)]);
        // The rig moved to the left side: 3 pixels behind it before the picture's edge.
        let left_side = [
            ScreenPos { x: 3.0, y: 5.0 },
            ScreenPos { x: 13.0, y: 5.0 },
            ScreenPos { x: 13.0, y: 15.0 },
            ScreenPos { x: 3.0, y: 15.0 },
        ];
        let mut cramped = canvas(W, H);
        for y in 5..15 {
            for x in 3..13 {
                paint(&mut cramped, W, x, y, HULL);
            }
        }
        let read = judge_frame(&frame(&cramped, Footprint::Hull(&left_side), RIGHT, bg));
        assert_eq!(trails(&read), vec![Err(Unreadable::NoRoomBehind); 3]);
    }

    /// A stripe inside the filter's reach ahead is the lead, not the background: the line reads,
    /// and the lead names the stripe's step.
    #[test]
    fn a_spread_ahead_is_the_lead_and_not_the_background() {
        let mut rgba = rig(&[0.5]);
        for y in 5..15 {
            paint(&mut rgba, W, 32, y, mix(0.5));
        }
        for read in judge_both(&rgba, RIGHT) {
            assert_eq!(trails(&read), vec![Ok(1), Ok(1), Ok(1)]);
            assert_eq!(leads(&read), vec![Ok(3), Ok(3), Ok(3)]);
        }
    }

    /// THE LEG: the frames pool into one reading; the longest and the median trail come from the
    /// pooled count; one unreadable line fails the leg; a leg of no frames never passes.
    #[test]
    fn a_leg_pools_its_frames_into_one_reading() {
        let trailed = rig(&[0.75, 0.5, 0.25]);
        let clean = rig(&[]);
        let [a, _] = judge_both(&trailed, RIGHT);
        let [b, _] = judge_both(&clean, RIGHT);
        let c = judge_frame(&frame(&clean, dot(), DVec2::ZERO, Background::Ahead));
        let leg = sum_reads(&[a, b.clone(), b.clone()]);
        assert!(leg.lines.is_empty());
        assert_eq!(leg.trails, BTreeMap::from([(0, 6), (3, 3)]));
        assert_eq!(leg.runs, BTreeMap::from([(0, 6), (3, 3)]));
        assert_eq!(leg.cover_low, None);
        assert_eq!(
            (leg.readable(), leg.max_px(), leg.median_px()),
            (9, Some(3), Some(0))
        );
        assert!(!leg.passes(GHOST_BOUND_PX));
        assert!(leg.passes(3));
        let clean_leg = sum_reads(&[b.clone(), b.clone()]);
        assert!(clean_leg.passes(GHOST_BOUND_PX));
        let broken = sum_reads(&[b, c]);
        assert_eq!(
            broken.unreadable,
            BTreeMap::from([(Unreadable::NoTravel, 3)])
        );
        assert!(!broken.passes(GHOST_BOUND_PX));
        let none = sum_reads(&[]);
        assert_eq!(none, GhostRead::default());
        assert_eq!(
            (none.readable(), none.max_px(), none.median_px()),
            (0, None, None)
        );
        assert!(!none.passes(GHOST_BOUND_PX));
        // The median of an even count is the lower of the two middle trails.
        let even = GhostRead {
            trails: BTreeMap::from([(0, 1), (2, 1)]),
            ..GhostRead::default()
        };
        assert_eq!(even.median_px(), Some(0));
    }

    /// The wide rig: a 160 × 20 picture of `sky`, the hull box over the columns 120..130 and the
    /// rows 5..15, and one pixel of `star` at (20, 10), 100 steps behind the box on the middle line.
    fn wide_rig(sky: [u8; 3], star: [u8; 3]) -> Vec<u8> {
        let w = 160;
        let mut rgba = canvas_of(w, H, sky);
        for y in 5..15 {
            for x in 120..130 {
                paint(&mut rgba, w, x, y, HULL);
            }
        }
        paint(&mut rgba, w, 20, 10, star);
        rgba
    }

    /// The wide rig's box, judged at 10 display pixels a frame (a reach of 330 steps).
    fn judge_wide(rgba: &[u8], background: Background) -> GhostRead {
        let corners = rect_corners(ScreenAabb {
            min: ScreenPos { x: 120.0, y: 5.0 },
            max: ScreenPos { x: 130.0, y: 15.0 },
        });
        judge_frame(&GhostFrame {
            rgba,
            width: 160,
            height: H,
            footprint: Footprint::Hull(&corners),
            travel_px: DVec2::new(10.0, 0.0),
            background,
            expected: None,
        })
    }

    /// THE REVIEW'S CASE: a clean hull box passes a white star at night. The star is 100 pixels
    /// behind the box on the middle line, and the box moves 10 pixels a frame (a reach of 330
    /// steps). The star's blend share is 1.47, so the farthest reading alone reads a trail of 100.
    /// But no mix of the hull and the sky makes the star's colour. So the middle line does not
    /// read (`ForeignBehind`), and it never reads a trail. The rig cannot tell a far star from a
    /// detached copy by its place, so it reads the colour and says that the scene breaks its
    /// requirement. The gate fails and names that reason. The two quarter lines do not cross the
    /// star and read 0. On a pure cyan sky the star's share is 0.70, a share that a mix can have,
    /// but its red channel is 28 levels past the hull's red and the margin: foreign too. Without
    /// the star the box reads 0 on every line and passes.
    #[test]
    fn a_star_behind_a_clean_box_is_foreign_and_never_a_trail() {
        let white = [255, 255, 255];
        let at_night = blend_share(level(white), level(HULL), level(SKY));
        assert!((at_night - 81_900.0 / 55_800.0).abs() < 1e-12);
        let on_cyan = blend_share(level(white), level(HULL), level(CYAN));
        assert!((on_cyan - 51_000.0 / 72_850.0).abs() < 1e-12);
        for sky in [SKY, CYAN] {
            for background in [Background::Colour(sky), Background::Ahead] {
                let read = judge_wide(&wide_rig(sky, white), background);
                let foreign = Err(Unreadable::ForeignBehind);
                assert_eq!(trails(&read), vec![Ok(0), foreign, Ok(0)]);
                assert_eq!(runs(&read), vec![Ok(0), foreign, Ok(0)]);
                assert_eq!(
                    read.unreadable,
                    BTreeMap::from([(Unreadable::ForeignBehind, 1)])
                );
                assert!(!read.passes(GHOST_BOUND_PX));
                let clean = judge_wide(&wide_rig(sky, sky), background);
                assert_eq!(trails(&clean), vec![Ok(0), Ok(0), Ok(0)]);
                assert!(clean.passes(GHOST_BOUND_PX));
                let middle = clean.lines[1].read.expect("the middle line reads");
                assert_eq!(middle.reach_px, 120);
            }
        }
    }

    /// The sRGB curve: a stored level (0 to 255) to light (0 to 1).
    fn to_light(level: f64) -> f64 {
        let c = level / 255.0;
        if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Light (0 to 1) to a stored level (0 to 255), rounded.
    fn to_level(light: f64) -> f64 {
        let c = if light <= 0.003_130_8 {
            light * 12.92
        } else {
            1.055 * light.powf(1.0 / 2.4) - 0.055
        };
        (c * 255.0).round()
    }

    /// The mix `a` of the sky and the hull, made in light and stored through the sRGB curve.
    fn light_mix(sky: DVec3, hull: DVec3, a: f64) -> DVec3 {
        let one = |s: f64, h: f64| to_level((1.0 - a) * to_light(s) + a * to_light(h));
        DVec3::new(one(sky.x, hull.x), one(sky.y, hull.y), one(sky.z, hull.z))
    }

    /// A MIX IN LIGHT IS NEVER FOREIGN. The reconstruction mixes in light, and the picture stores
    /// the sRGB curve of that light. Such a mix of the hull and the sky, at every share from 0 to 1
    /// in steps of 0.05, stays inside the lawful range on each channel, on the night sky and on
    /// the cyan sky. But it is not on the straight line between the two stored colours: on the
    /// cyan sky the half mix (146, 222, 196) is 48 levels off that line. So a check of the
    /// distance from that line would read an honest trail as foreign; the rig checks each channel.
    #[test]
    fn a_mix_in_light_is_never_foreign() {
        let hull = level(HULL);
        for sky in [level(SKY), level(CYAN)] {
            for step in 0..=20 {
                let p = light_mix(sky, hull, f64::from(step) / 20.0);
                assert!(!foreign(p, sky.min(hull), sky.max(hull)));
            }
        }
        let (sky, half) = (level(CYAN), light_mix(level(CYAN), hull, 0.5));
        assert_eq!(half, DVec3::new(146.0, 222.0, 196.0));
        let on_line = sky + (hull - sky) * blend_share(half, hull, sky);
        assert!((half - on_line).length() > 48.0);
    }

    /// A COLOUR THAT NO MIX CAN MAKE IS FOREIGN. Behind the hull box on the night sky the red
    /// margin ends 24.5 levels past the hull's red of 200. One pixel behind the box at red 224
    /// reads as a trail pixel; at red 225 the lines do not read. Black and a ground colour are
    /// foreign too. A channel where the hull and the sky have the same level gets only the noise:
    /// a pixel 2 levels past it reads, 3 levels past it does not.
    #[test]
    fn a_colour_that_no_mix_can_make_is_foreign() {
        for read in judge_both(&rig_with(HULL, &[[224, 105, 75]]), RIGHT) {
            assert_eq!(trails(&read), vec![Ok(1), Ok(1), Ok(1)]);
        }
        let ground = [70, 90, 40];
        for stray in [[225, 105, 75], [0, 0, 0], ground] {
            for read in judge_both(&rig_with(HULL, &[stray]), RIGHT) {
                assert_eq!(trails(&read), vec![Err(Unreadable::ForeignBehind); 3]);
                assert_eq!(
                    read.unreadable,
                    BTreeMap::from([(Unreadable::ForeignBehind, 3)])
                );
            }
        }
        let flat = [200, 180, SKY[2]];
        for read in judge_both(&rig_with(flat, &[[110, 105, SKY[2] + 2]]), RIGHT) {
            assert_eq!(trails(&read), vec![Ok(1), Ok(1), Ok(1)]);
        }
        for read in judge_both(&rig_with(flat, &[[110, 105, SKY[2] + 3]]), RIGHT) {
            assert_eq!(trails(&read), vec![Err(Unreadable::ForeignBehind); 3]);
        }
    }

    /// The rig's dot at (20, 10) at 60 % of its settled contrast: the colour 60 % of the way from
    /// the sky to the hull.
    fn dim_dot(trail: &[f64]) -> Vec<u8> {
        dot_painted(W, H, DVec2::new(20.0, 10.0), mix(0.6), trail)
    }

    /// A frame of the rig's dot at 5 display pixels a frame, with the settled colour `expected`.
    fn dot_frame(rgba: &[u8], background: Background, expected: Option<DVec3>) -> GhostFrame<'_> {
        let mut f = frame(rgba, dot(), DVec2::new(5.0, 0.0), background);
        f.expected = expected;
        f
    }

    /// THE FADE-IN (risk 24): a pilot's dot at night shows 60 % of its settled contrast in its
    /// first frame. Without a settled colour the dim dot is its own reference: every line reads 0,
    /// and the gate passes (the review's finding). With the settled colour stated, every line reads
    /// a cover share of 0.6, and the gate fails. The settled dot reads 1.0 and passes. A missing
    /// settled colour reads `None` and changes nothing else.
    #[test]
    fn a_dot_that_fades_in_fails_against_its_settled_colour() {
        let (dim, settled) = (dim_dot(&[]), dot_picture(&[]));
        let hull = Some(level(HULL));
        for bg in [Background::Colour(SKY), Background::Ahead] {
            let blind = judge_frame(&dot_frame(&dim, bg, None));
            assert_eq!(trails(&blind), vec![Ok(0), Ok(0), Ok(0)]);
            assert_eq!(covers(&blind), vec![Ok(None), Ok(None), Ok(None)]);
            assert!(blind.passes(GHOST_BOUND_PX));
            let faded = judge_frame(&dot_frame(&dim, bg, hull));
            assert_eq!(trails(&faded), vec![Ok(0), Ok(0), Ok(0)]);
            assert_eq!(covers(&faded), vec![Ok(Some(0.6)); 3]);
            assert_eq!(faded.cover_low, Some(0.6));
            assert!(!faded.passes(GHOST_BOUND_PX));
            let with = judge_frame(&dot_frame(&settled, bg, hull));
            assert_eq!(covers(&with), vec![Ok(Some(1.0)); 3]);
            assert!(with.passes(GHOST_BOUND_PX));
            let without = judge_frame(&dot_frame(&settled, bg, None));
            for (a, b) in with.lines.iter().zip(&without.lines) {
                let a = a.read.expect("the line reads");
                assert_eq!(
                    b.read,
                    Ok(LineTrail {
                        cover_share: None,
                        ..a
                    })
                );
            }
            assert_eq!((&with.trails, &with.runs), (&without.trails, &without.runs));
            assert_eq!((with.cover_low, without.cover_low), (Some(1.0), None));
            assert!(without.passes(GHOST_BOUND_PX));
        }
    }

    /// THE SETTLED COLOUR WIDENS THE LAWFUL RANGE: a ghost of the settled dot (80 % of the settled
    /// contrast) lies behind the dim dot. It is a mix of the settled colour and the sky. With the
    /// settled colour stated, it reads as a trail pixel (1), and the cover share fails the gate.
    /// Without the settled colour it is brighter than any mix of the dim dot and the sky, so it is
    /// foreign.
    #[test]
    fn a_ghost_of_the_settled_dot_reads_as_a_trail() {
        let ghosted = dim_dot(&[0.8]);
        for bg in [Background::Colour(SKY), Background::Ahead] {
            let blind = judge_frame(&dot_frame(&ghosted, bg, None));
            assert_eq!(trails(&blind), vec![Err(Unreadable::ForeignBehind); 3]);
            let read = judge_frame(&dot_frame(&ghosted, bg, Some(level(HULL))));
            assert_eq!(trails(&read), vec![Ok(1), Ok(1), Ok(1)]);
            assert_eq!(read.cover_low, Some(0.6));
            assert!(!read.passes(GHOST_BOUND_PX));
        }
    }

    /// THE SETTLED COLOUR FROM A LATER FRAME: the caller reads the dot's colour on a settled frame
    /// (the middle line's object colour) and states it for the first frame. The first frame's
    /// fade-in fails the same way.
    #[test]
    fn a_settled_colour_read_on_a_later_frame_finds_the_fade_in() {
        let bg = Background::Colour(SKY);
        let later = judge_frame(&dot_frame(&dot_picture(&[]), bg, None));
        let settled = later.lines[1].read.expect("the middle line reads").object;
        assert_eq!(settled, level(HULL));
        let first = judge_frame(&dot_frame(&dim_dot(&[]), bg, Some(settled)));
        assert_eq!(first.cover_low, Some(0.6));
        assert!(!first.passes(GHOST_BOUND_PX));
    }

    /// THE BASIS OF THE SETTLED CONTRAST: a settled colour 64 levels from the sky on its widest
    /// channel reads. There the dot shows 4 levels under the stated colour on each channel (2
    /// levels of noise in each of the two frames, in the worst direction). The cover share is
    /// 4800 / 5248 = 0.915: the noise takes 0.085, and the dot still passes. A settled colour 63
    /// levels from the sky does not read.
    #[test]
    fn a_settled_colour_needs_twice_the_smallest_contrast() {
        let settled = [SKY[0] + 64, SKY[1] + 24, SKY[2] + 24];
        let noisy = [settled[0] - 4, settled[1] - 4, settled[2] - 4];
        let rgba = dot_painted(W, H, DVec2::new(20.0, 10.0), noisy, &[]);
        let bg = Background::Colour(SKY);
        let read = judge_frame(&dot_frame(&rgba, bg, Some(level(settled))));
        let low = read.cover_low.expect("the lines read a cover share");
        assert!((low - 4800.0 / 5248.0).abs() < 1e-12);
        assert!(low >= COVER_SHARE_MIN);
        assert!(read.passes(GHOST_BOUND_PX));
        let nearer = Some(level(settled) - DVec3::X);
        let read = judge_frame(&dot_frame(&rgba, bg, nearer));
        assert_eq!(trails(&read), vec![Err(Unreadable::LowContrast); 3]);
    }

    /// THE LEG KEEPS ITS LOWEST COVER SHARE: a leg with one faded frame fails; a leg of settled
    /// frames passes; a frame without a settled colour takes no part.
    #[test]
    fn a_leg_keeps_its_lowest_cover_share() {
        let bg = Background::Colour(SKY);
        let hull = Some(level(HULL));
        let settled = dot_picture(&[]);
        let steady = judge_frame(&dot_frame(&settled, bg, hull));
        let faded = judge_frame(&dot_frame(&dim_dot(&[]), bg, hull));
        let blind = judge_frame(&dot_frame(&settled, bg, None));
        let leg = sum_reads(&[steady.clone(), faded, blind.clone()]);
        assert_eq!(leg.cover_low, Some(0.6));
        assert!(!leg.passes(GHOST_BOUND_PX));
        let leg = sum_reads(&[blind.clone(), steady]);
        assert_eq!(leg.cover_low, Some(1.0));
        assert!(leg.passes(GHOST_BOUND_PX));
        let leg = sum_reads(&[blind.clone(), blind]);
        assert_eq!(leg.cover_low, None);
        assert!(leg.passes(GHOST_BOUND_PX));
    }
}
