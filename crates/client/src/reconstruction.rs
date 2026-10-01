//! ★ THE RECONSTRUCTION'S RULES (the temporal path's slice T2; `docs/investigation/2026-09-28/
//! temporal_path_design.md` §4.2–§4.4 and §6): the pure half that the render world runs and the
//! reconstruction's shaders transliterate, step for step — the jitter's phases and its clock, the
//! floor law of the render scale, the reset's causes, the sample filter and its narrowing per pixel,
//! the upsample with its de-ringing, the count of earned weight and its cap by motion, the tone
//! weight, the colour box that rectifies the history, the depth test that finds a surface hidden in the
//! last frame, and the reactive mask that finds a pixel the transparent pass changed.
//!
//! ★ THE UPSAMPLE (FSR 2.2's `ComputeUpsampledColorAndWeight`, MEASURED needed on 2026-09-30): a
//! frame gives a display pixel only the weight its samples EARN — the sum of their Lanczos-2 weights
//! — and its colour is clamped into the range its samples span. At half scale the filter is so narrow
//! that a display pixel between two samples earns almost nothing, and the first build divided by that
//! near-nothing: the owner saw bright flashes along every dark-and-bright edge (the dune against the
//! haze), a row of pixels brighter than the sky that walked along the edge with the jitter.
//!
//! **One settings struct** ([`ReconstructionSettings`]): every constant of the reconstruction with
//! its basis (the project's rule: no magic numbers). The look constants — the floors, the box's
//! width — are PROPOSED by the design and ruled by the owner (D7, D9).
//!
//! **Example.** The owner stands on foot at full screen (2 168 rows) with the render scale at 0.5:
//! the jitter runs 32 phases of the Halton (2, 3) sequence, the floor law allows a scale down to
//! 720 ÷ 2 168 = 0.332, and a still pixel's new-frame weight falls from one to its floor of 1/64
//! over its first 64 frames, while a pixel that moves a whole render pixel a frame keeps a weight
//! of at least 1/16 and never trails further behind.

use vd_core::geometry::REFERENCE_VIEW_ROWS_PX;

/// The smallest render scale the rules accept: a display of 64 times the reference view's rows.
/// A lawful scale ([`lawful_scale`]) never reaches it on any display that exists; it only keeps the
/// phase count finite for a caller that passes nonsense.
pub const SCALE_MIN: f64 = 1.0 / 64.0;

/// ★ THE ONE SETTINGS STRUCT (the design §4.2: every constant of the reconstruction in one named
/// struct, each with its basis).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructionSettings {
    /// The new frame's weight floor for a still pixel (PROPOSED 1/64, the design §4.2; the owner's
    /// D9): the jitter's remaining crawl is about this share of an edge's contrast, under one level
    /// on most edges.
    pub still_floor: f64,
    /// The new frame's weight floor for a pixel that moves (PROPOSED 1/16, the design §4.2; D9): the
    /// history trails a moving edge by at most about this share of its motion.
    pub moving_floor: f64,
    /// The motion under which a pixel counts as still, in render pixels a frame (PROPOSED 0.01, the
    /// design §4.2: a pilot in orbit sees the planet turn 0.02 px a frame and must converge).
    pub still_motion_px: f64,
    /// The motion from which the moving floor holds, in render pixels a frame (PROPOSED 1.0).
    pub moving_motion_px: f64,
    /// The history's colour box: the new samples' mean ± this many deviations, in YCoCg, cut to the
    /// samples' own range (FSR 2.2's rectification box, [`rectify`]; PROPOSED 1.25, the owner's look at
    /// D7).
    pub box_deviations: f64,
    /// The sharpen after the tone map (the design's stage 11): the engine's contrast-adaptive
    /// sharpening strength in `[0, 1]` (PROPOSED 0.5 for the first look; the owner's D7). FSR 2 ships
    /// its own sharpen for the same reason: its filter over the jittered samples is soft at full size.
    pub sharpen: f64,
    /// The depth test's separation constant (FSR 2.2's `Ksep`, `ffx_fsr2_depth_clip.h`): two surfaces
    /// stand apart when their view depths differ by more than this × the field's corner factor × the
    /// render picture's diagonal in pixels, of the farther one — 2.6 % of the depth on the capture's
    /// 1 284 × 720 grid at its 45° view.
    pub depth_separation_k: f64,
    /// Under this motion, in DISPLAY pixels a frame, the rebuilt previous depth takes a pixel as still
    /// (FSR 2.2: 0.1 — a smaller motion is the reprojection's noise).
    pub rebuild_still_display_px: f64,
    /// A bilinear tap under this weight neither writes nor reads the rebuilt previous depth (FSR 2.2:
    /// 0.01).
    pub bilinear_tap_floor: f64,
    /// ★ THE REACTIVE MASK (the design §4.2 step 3; FSR 2's automatic reactive mask with its sample's
    /// defaults, REPORTED): the tone-mapped change the transparent pass made to a render pixel, its
    /// largest channel, times this scale (PROPOSED 1.0)...
    pub reactive_scale: f64,
    /// ...counts from this change on (PROPOSED 0.2: a faint glass leaves the history alone)...
    pub reactive_threshold: f64,
    /// ...as this reactive value (PROPOSED 0.9: the pixel takes mostly the new frame, and its colour box
    /// shrinks to a tenth).
    pub reactive_value: f64,
    /// A frame's upsample weight under this counts as none (FSR 2.2's `FSR2_EPSILON`): the display pixel
    /// lies between this frame's samples, and its history carries it.
    pub upsample_weight_min: f64,
}

impl ReconstructionSettings {
    /// The design's proposed values, until the owner rules D7 and D9.
    pub const PROPOSED: ReconstructionSettings = ReconstructionSettings {
        still_floor: 1.0 / 64.0,
        moving_floor: 1.0 / 16.0,
        still_motion_px: 0.01,
        moving_motion_px: 1.0,
        box_deviations: 1.25,
        sharpen: 0.5,
        depth_separation_k: 1.37e-5,
        rebuild_still_display_px: 0.1,
        bilinear_tap_floor: 0.01,
        reactive_scale: 1.0,
        reactive_threshold: 0.2,
        reactive_value: 0.9,
        upsample_weight_min: 1e-3,
    };
}

/// FSR 2.2's reference render size for the depth test's power: at this diagonal and above, the
/// power is three.
pub const DEPTH_CLIP_REFERENCE_PX: [f64; 2] = [1_920.0, 1_080.0];

/// The smallest reverse-Z depth the rules divide by: the sky's nought reads as a view depth of
/// `near × 10³⁰`, farther than every surface and still finite in the GPU's single floats.
pub const DEPTH_FLOOR: f64 = 1e-30;

/// The narrowest the sample filter becomes: FSR 2's cap (`ComputeMaxKernelWeight`), so a render
/// sample always reaches the display pixels around it.
pub const KERNEL_BIAS_MAX: f64 = 1.99;

/// ★ THE FILTER'S NARROWING at render scale `s` (FSR 2's kernel weight): the Lanczos-2 filter reads the
/// render offset times `1 ÷ s`, capped at [`KERNEL_BIAS_MAX`] — so the filter spans about two DISPLAY
/// pixels at every scale: two render pixels at 1.0, one at 0.5.
#[must_use]
pub fn kernel_bias(render_scale: f64) -> f64 {
    (1.0 / render_scale.clamp(SCALE_MIN, 1.0)).min(KERNEL_BIAS_MAX)
}

/// The share of a phase under which ⌈8 ÷ s²⌉ reads a whole number as whole: a scale read off two
/// whole sizes (720 ÷ 1 080 for two thirds) squares to a value a last bit off, and 18 would read 19.
const PHASE_ROUNDING: f64 = 1e-9;

/// ★ THE JITTER'S PHASES at render scale `s`: ⌈8 ÷ s²⌉ (FSR 2's rule; the design §4.3) — 8 at 1.0, 15
/// at 0.75, 18 at 0.67 and at two thirds, 32 at 0.5 — so each display pixel sees about eight samples
/// of its own per cycle at every scale.
#[must_use]
pub fn jitter_phases(render_scale: f64) -> u32 {
    let s = render_scale.clamp(SCALE_MIN, 1.0);
    (8.0 / (s * s) - PHASE_ROUNDING).ceil() as u32
}

/// The Halton sequence's value at `index` in `base`: the digits of `index` in `base` mirrored about
/// the point (the radical inverse), in `[0, 1)`.
#[must_use]
pub fn halton(index: u32, base: u32) -> f64 {
    let b = f64::from(base.max(2));
    let mut i = index;
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= b;
        r += f * f64::from(i % base.max(2));
        i /= base.max(2);
    }
    r
}

/// ★ THE JITTER of `phase` in a cycle of `phases`, in render pixels, each axis in `[−0.5, 0.5)`: the
/// Halton (2, 3) point at index `phase mod phases + 1`, less a half. The index starts at ONE, because
/// FSR 2 says that a jitter "should never generate a null vector" (Bevy's own first phase is (0, 0),
/// the design's F3).
///
/// **The sign.** The jitter is WHERE A RENDER PIXEL'S SAMPLE LIES: render pixel `i` samples the scene
/// at `i + 0.5 + jitter` (x to the right, y down). The engine's projection offset takes the jitter as
/// it is: it draws a scene point at `p` as `p − offset`, so the pixel centre `c` sees `c + offset`
/// (`vd_client_render`'s test of the engine's own projection proves it). FSR 2 states the same
/// geometry with the opposite sign — its projection moves a point by `+jitter`, so its pixel sees
/// `c − jitter` — and its code reads its jitter that way.
#[must_use]
pub fn jitter(phase: u32, phases: u32) -> [f64; 2] {
    let index = phase % phases.max(1) + 1;
    [halton(index, 2) - 0.5, halton(index, 3) - 0.5]
}

/// ★ THE JITTER'S CLOCK (the design §4.3): the phase counts from nought at each reset, so two captures
/// of one stand see the same phases (Bevy's sequence takes the frame count, F3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JitterClock {
    pub phase: u32,
}

impl JitterClock {
    /// The next frame's phase: nought on a reset, else one more (wrapping).
    pub fn advance(&mut self, reset: bool) -> u32 {
        self.phase = u32::from(!reset) * self.phase.wrapping_add(1);
        self.phase
    }
}

/// ★ THE CONVERGED STILL (the design §12.2): after a reset, the history's count reaches its cap
/// (`1 ÷ still_floor`, a frame a pixel at most) only after that many frames; a converged still is
/// taken at the first frame of a whole jitter cycle at or past it, so two captures of one stand read
/// the same history and the same phase (DERIVED: 64 frames at 8 phases and at 32).
///
/// **Example.** At one half (32 phases, a floor of 1/64) the capture resets, waits 64 frames (two
/// whole cycles) and takes the 65th, whose phase is the cycle's first.
#[must_use]
pub fn converged_frames(phases: u32, settings: &ReconstructionSettings) -> u32 {
    let cap = (1.0 / settings.still_floor).ceil() as u32;
    let phases = phases.max(1);
    cap.div_ceil(phases) * phases
}

/// ★ THE NEAREST SAMPLE to a point `at` of the render grid (the gather's centre, the design §4.2
/// step 4.1): render pixel `i`'s sample lies at `i + 0.5 + jitter` ([`jitter`]), so the pixel whose
/// sample lies within half a pixel of `at` is `⌊at − jitter⌋` on each axis.
#[must_use]
pub fn nearest_sample(at: [f64; 2], jitter: [f64; 2]) -> [i64; 2] {
    [0, 1].map(|i| (at[i] - jitter[i]).floor() as i64)
}

/// ★ WHERE RENDER PIXEL `pixel` SAMPLED THE SCENE on the render grid: `pixel + 0.5 + jitter` ([`jitter`]).
/// The gates reproject a jittered picture through it (the design §12.3).
#[must_use]
pub fn sample_point(pixel: [i64; 2], jitter: [f64; 2]) -> [f64; 2] {
    [0, 1].map(|i| pixel[i] as f64 + 0.5 + jitter[i])
}

/// The offset of render pixel `pixel`'s sample from `at`, in render pixels: what the sample filter
/// ([`lanczos2`]) weighs.
#[must_use]
pub fn sample_offset(pixel: [i64; 2], jitter: [f64; 2], at: [f64; 2]) -> [f64; 2] {
    let point = sample_point(pixel, jitter);
    [0, 1].map(|i| point[i] - at[i])
}

/// ★ THE FLOOR LAW (the design §4.4): one render pixel may never cover more than one pixel of the
/// reference view, so the render scale may not fall under the reference view's rows ÷ the display's
/// rows — 0.332 on the owner's 2 168 rows, 0.667 on 1 080, one on 720 or fewer. At the floor no
/// ladder cell at its switch distance and no shipped occupant is smaller than one render pixel.
#[must_use]
pub fn scale_floor(display_rows: u32) -> f64 {
    (REFERENCE_VIEW_ROWS_PX / f64::from(display_rows.max(1))).min(1.0)
}

/// The render scale a wish may take: the wish, never under the floor law and never over one.
#[must_use]
pub fn lawful_scale(wanted: f64, display_rows: u32) -> f64 {
    wanted.clamp(scale_floor(display_rows), 1.0)
}

/// The render size for a display size at a lawful scale: each side rounded to a whole pixel, never
/// nought (the engine's own override takes whole pixels).
#[must_use]
pub fn render_size(display: [u32; 2], scale: f64) -> [u32; 2] {
    display.map(|side| ((f64::from(side) * scale).round() as u32).max(1))
}

/// ★ THE CAPTURE'S DISPLAY SIZE (the design §12.1, "the capture keeps its grid"): the capture's RENDER
/// picture stays the reference view, and its display picture grows to the render grid ÷ `s`, each
/// side rounded. The render size at `s` of that display is the grid again, exactly: a side's
/// rounding moves it by at most half a display pixel, which is at most `s ÷ 2` of a render pixel.
#[must_use]
pub fn display_size(render: [u32; 2], scale: f64) -> [u32; 2] {
    let s = scale.clamp(SCALE_MIN, 1.0);
    render.map(|side| (f64::from(side) / s).round() as u32)
}

/// The view depth (along the view's axis) of a reverse-Z depth under the engine's infinite projection:
/// `near ÷ depth`; the sky's nought reads as the farthest depth there is ([`DEPTH_FLOOR`]).
#[must_use]
pub fn view_depth(depth: f64, near: f64) -> f64 {
    near / depth.max(DEPTH_FLOOR)
}

/// The field's corner factor (FSR 2.2's `Kfov`): the corner ray's length over the centre ray's at one
/// view depth, from the projection's two scales (`P₀₀`, `P₁₁`): `√(1 + 1/P₀₀² + 1/P₁₁²)` — 1.31 for the
/// 45° view on a 16 : 9 picture.
#[must_use]
pub fn corner_factor(x_scale: f64, y_scale: f64) -> f64 {
    (1.0 + 1.0 / (x_scale * x_scale) + 1.0 / (y_scale * y_scale)).sqrt()
}

/// The depth test's separation per metre of depth: `Ksep × Kfov × the render diagonal in pixels`.
#[must_use]
pub fn depth_separation(render: [u32; 2], corner: f64, settings: &ReconstructionSettings) -> f64 {
    let [w, h] = render.map(f64::from);
    settings.depth_separation_k * corner * w.hypot(h)
}

/// The depth test's power (FSR 2.2): one for an empty picture, rising in a straight line to three at
/// [`DEPTH_CLIP_REFERENCE_PX`]'s diagonal and above — a larger picture separates its depths sharper.
#[must_use]
pub fn depth_clip_power(render: [u32; 2]) -> f64 {
    let [w, h] = render.map(f64::from);
    let [rw, rh] = DEPTH_CLIP_REFERENCE_PX;
    1.0 + 2.0 * (w.hypot(h) / rw.hypot(rh)).clamp(0.0, 1.0)
}

/// ★ THE BILINEAR FOOTPRINT of a point on a pixel grid (pixel `i` centred at `i + 0.5`): the pixel at
/// the footprint's lower corner, and the weights of its four taps in the order (0, 0), (1, 0), (0, 1),
/// (1, 1) — tap `k` at the corner plus `(k & 1, k >> 1)`. The weights sum to one.
#[must_use]
pub fn bilinear_footprint(at: [f64; 2]) -> ([i64; 2], [f64; 4]) {
    let corner = at.map(|v| v - 0.5);
    let base = corner.map(f64::floor);
    let [fx, fy] = [corner[0] - base[0], corner[1] - base[1]];
    (
        base.map(|v| v as i64),
        [
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ],
    )
}

/// ★ THE DEPTH TEST (the design §4.2 step 2; FSR 2.2's `ComputeDepthClip`): how much of a pixel was
/// HIDDEN in the last frame, from nought (seen) to one (a disocclusion), from its current depth and the
/// rebuilt previous depth at its last-frame place — each tap `(the nearest depth that stood there,
/// its bilinear weight)`, the taps off the picture or under the tap floor left out by the caller.
/// A tap whose surface stood NEARER than this pixel's counts: the pixel is kept by the share
/// `(separation × the farther depth ÷ the gap)^power`, so a gap under the separation keeps the history
/// and a wide one drops it smoothly. A pixel with no nearer tap was seen.
///
/// **Example.** The eye walks along the belt's ridge; a valley 3 km away comes into view behind a rock
/// 40 m away. The valley's pixel finds the rock's depth at its last-frame place — a gap of 2 960 m
/// against a separation of 75 m — and takes no history: no smear of the rock over the valley.
#[must_use]
pub fn depth_clip(
    current: f64,
    taps: &[(f64, f64)],
    near: f64,
    separation: f64,
    power: f64,
) -> f64 {
    let here = view_depth(current, near);
    let (kept, weight) = taps
        .iter()
        .map(|&(before, w)| (view_depth(before, near), w))
        .filter(|&(before, _)| here > before)
        .fold((0.0, 0.0), |(kept, weight), (before, w)| {
            let share = (separation * here.max(before) / (here - before)).clamp(0.0, 1.0);
            (kept + share.powf(power) * w, weight + w)
        });
    if weight > 0.0 {
        (1.0 - kept / weight).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// ★ THE REACTIVE VALUE of a render pixel (the design §4.2 step 3; FSR 2's automatic reactive mask): the
/// change the transparent pass made to it — `before` and `after` the pass, each through the reversible
/// tone map `c ÷ (1 + max c)` — its largest channel times the scale; under the threshold nought, else the
/// settings' reactive value. A pixel that a translucent thing crossed takes mostly the new frame; the
/// opaque world keeps its history.
///
/// **Example.** A hull's translucent box slides across the belt: its pixels change by far more than the
/// threshold and read 0.9, so they keep a tenth of their count and leave no trail behind the box.
#[must_use]
pub fn reactive(before: [f64; 3], after: [f64; 3], settings: &ReconstructionSettings) -> f64 {
    let tone = |c: [f64; 3]| {
        let c = c.map(|v| v.max(0.0));
        let m = c.into_iter().fold(0.0_f64, f64::max);
        c.map(|v| v / (1.0 + m))
    };
    let (b, a) = (tone(before), tone(after));
    let change = [0, 1, 2]
        .map(|i| (a[i] - b[i]).abs())
        .into_iter()
        .fold(0.0_f64, f64::max)
        * settings.reactive_scale;
    f64::from(u8::from(change >= settings.reactive_threshold)) * settings.reactive_value
}

/// ★ THE FILTER'S NARROWING FOR ONE PIXEL (FSR 2.2's kernel bias): the scale's narrowest
/// ([`kernel_bias`]) widened toward FSR 2's widest, `max(1, (1 + narrowest) × 0.3)`, by `widen` in
/// `[0, 1]` — the largest of a quarter of the pixel's depth test, its reactive value and one for a pixel
/// that starts again — so a pixel with no history always gathers samples: at half scale its filter
/// spans two render pixels, not one.
#[must_use]
pub fn pixel_kernel_bias(narrowest: f64, widen: f64) -> f64 {
    let widest = ((1.0 + narrowest) * 0.3).max(1.0);
    narrowest + (widest - narrowest) * widen.clamp(0.0, 1.0)
}

/// ★ A SAMPLE'S UPSAMPLE WEIGHT (FSR 2.2's `GetUpsampleLanczosWeight`, its reference form): Lanczos-2 of
/// the sample's DISTANCE from the display pixel's centre, in render pixels, times the pixel's kernel bias.
#[must_use]
pub fn upsample_weight(offset: [f64; 2], bias: f64) -> f64 {
    lanczos2((offset[0] * bias).hypot(offset[1] * bias))
}

/// ★ THE UPSAMPLE of one channel (FSR 2.2's, with its de-ringing): each sample `(its offset from the
/// display pixel's centre, its value, whether it lies on the picture)`, the nearest first. The value is
/// the weighted mean CLAMPED into the range the samples span — Lanczos-2's negative lobes cannot
/// overshoot an edge — and the weight is the sum of the weights, this frame's EARNED weight. A sum
/// under the floor earns nothing: the value is the nearest sample's, and the pixel's history carries it.
///
/// **Example.** At half scale a display pixel on the dune's edge lies half way between a dune sample
/// and a haze sample: both sit on Lanczos-2's zero at a biased distance of one, the sum is a few
/// thousandths, and the pixel keeps its history this frame — where the first build divided by those
/// thousandths and drew a pixel brighter than the sky.
#[must_use]
pub fn upsample(
    samples: &[([f64; 2], f64, bool)],
    bias: f64,
    settings: &ReconstructionSettings,
) -> (f64, f64) {
    let (lo, hi) = samples.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(lo, hi), &(_, v, _)| (lo.min(v), hi.max(v)),
    );
    let (sum, weight) = samples.iter().filter(|&&(_, _, on)| on).fold(
        (0.0, 0.0),
        |(sum, weight), &(offset, v, _)| {
            let w = upsample_weight(offset, bias);
            (sum + v * w, weight + w)
        },
    );
    let nearest = samples.first().map_or(0.0, |&(_, v, _)| v);
    if weight > settings.upsample_weight_min {
        ((sum / weight).clamp(lo, hi), weight)
    } else {
        (nearest, 0.0)
    }
}

/// ★ THE NEW COUNT of a pixel, in earned weight (FSR 2's accumulation): the history's count lowered by
/// the stronger of the depth test and the reactive value (to nought on a restart: a reset, or a pixel
/// from off the picture), plus this frame's earned weight, never over the cap by motion
/// ([`count_cap`]). Motion alone never restarts it.
#[must_use]
pub fn next_count(history: f64, earned: f64, cap: f64, lowering: f64, restart: bool) -> f64 {
    let kept = if restart {
        0.0
    } else {
        history * (1.0 - lowering.clamp(0.0, 1.0))
    };
    (kept + earned).min(cap)
}

/// ★ THE COUNT'S CAP BY MOTION: one over the weight floor by motion ([`weight_floor`]) — 64 for a still
/// pixel, 16 from a pixel a frame — so a frame that earns a whole weight never falls under the design's
/// floors, and a moving pixel never trails further behind.
#[must_use]
pub fn count_cap(motion_px: f64, settings: &ReconstructionSettings) -> f64 {
    1.0 / weight_floor(motion_px, settings)
}

/// ★ THE NEW FRAME'S BLEND WEIGHT: its earned weight over the new count — one on a restart, and nought
/// for a frame that earned nothing (the history carries the pixel). A pixel with no count at all (a
/// restart that earned nothing) takes the frame whole: it has no history to keep.
#[must_use]
pub fn blend_weight(earned: f64, count: f64) -> f64 {
    if count > 0.0 {
        (earned / count).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// ★ WHY THE HISTORY RESTARTS (the design §6.1): only where no history exists or it holds another
/// picture. The history is NEVER restarted on the player's path by motion, a crossing, a swap, a
/// render-scale change or a harvest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResetCause {
    /// The view's first frame: no history exists.
    FirstFrame,
    /// The device was lost: the history's textures are gone.
    DeviceLost,
    /// The display size changed (a window resize, the full-screen switch): the history lives at
    /// the display size.
    DisplaySize,
    /// A tool's jump: a gate's teleport, the capture's scene-fitted camera, a stand's start —
    /// counted apart from the player's path.
    ToolJump,
}

/// ★ THE RESET DECISION of one frame, the first cause that holds: no history yet, a lost device, a
/// new display size, a tool's jump. `None` — the history carries on.
#[must_use]
pub fn reset_cause(
    has_history: bool,
    device_lost: bool,
    display_size: [u32; 2],
    history_size: [u32; 2],
    tool_jump: bool,
) -> Option<ResetCause> {
    let causes = [
        (!has_history, ResetCause::FirstFrame),
        (device_lost, ResetCause::DeviceLost),
        (display_size != history_size, ResetCause::DisplaySize),
        (tool_jump, ResetCause::ToolJump),
    ];
    causes
        .iter()
        .find(|(holds, _)| *holds)
        .map(|(_, cause)| *cause)
}

/// ★ THE NEW FRAME'S WEIGHT FLOOR by the pixel's motion (the design §4.2 step 4.4): the still floor
/// under `still_motion_px`, the moving floor from `moving_motion_px`, and between them a straight
/// line in the motion's logarithm, so the floor changes smoothly with the motion and never jumps.
#[must_use]
pub fn weight_floor(motion_px: f64, settings: &ReconstructionSettings) -> f64 {
    let lo = settings.still_motion_px.ln();
    let hi = settings.moving_motion_px.ln();
    let t = ((motion_px.max(f64::MIN_POSITIVE).ln() - lo) / (hi - lo)).clamp(0.0, 1.0);
    settings.still_floor + (settings.moving_floor - settings.still_floor) * t
}

/// ★ THE SAMPLE FILTER: Lanczos-2, `sinc(x) · sinc(x/2)` inside two pixels, nought outside — the
/// weight of a render sample by its distance from a display pixel's centre under this frame's
/// jitter (FSR 2's upsample).
#[must_use]
pub fn lanczos2(x: f64) -> f64 {
    let sinc = |v: f64| {
        let p = std::f64::consts::PI * v;
        if p == 0.0 { 1.0 } else { p.sin() / p }
    };
    f64::from(u8::from(x.abs() < 2.0)) * sinc(x) * sinc(x * 0.5)
}

/// ★ THE TONE WEIGHT (Karis 2014): a sample counts by `1 ÷ (1 + its luminance)`, so one bright point
/// cannot dominate a pixel's accumulated colour; the weight is reversed on the output.
#[must_use]
pub fn tone_weight(luminance: f64) -> f64 {
    1.0 / (1.0 + luminance.max(0.0))
}

/// Linear RGB to YCoCg (the colour box's space: luminance apart from the two chroma axes).
#[must_use]
pub fn rgb_to_ycocg(rgb: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = rgb;
    [
        0.25 * r + 0.5 * g + 0.25 * b,
        0.5 * r - 0.5 * b,
        -0.25 * r + 0.5 * g - 0.25 * b,
    ]
}

/// YCoCg to linear RGB (the inverse of [`rgb_to_ycocg`]).
#[must_use]
pub fn ycocg_to_rgb(ycocg: [f64; 3]) -> [f64; 3] {
    let [y, co, cg] = ycocg;
    [y + co - cg, y + cg, y - co - cg]
}

/// ★ THE HISTORY'S RECTIFICATION (the design §4.2 step 4.3, as FSR 2.2's `RectifyHistory`,
/// `ffx_fsr2_accumulate.h`, 2026-10-01): the history clamped, axis by axis in YCoCg, into the box of
/// `centre ± extent` cut to the samples' own range `lo..=hi` — the box's NEAREST point, so two
/// colours never move apart (the clamp onto an axis-aligned box is non-expansive on every axis). The
/// bounds follow the shader's `min(max(h, lower), upper)`: a box that rounding inverts takes its upper
/// bound, and nothing panics.
///
/// ★ WHY NOT THE CLIP TOWARD THE CENTRE (Playdead 2016, the first build): it moves the history along
/// the ray to the box's centre, and on a box (not a ball) that move can carry two near colours apart.
/// With a new frame's weight of 1/64 nothing pulled them back faster than the jitter's eight boxes
/// pushed, so any difference grew until the box's size bounded it. MEASURED (2026-10-01, ruling B28
/// item 3: the shader's blend replayed on the solved ground stand's captured frames): two inputs
/// 1e-5 apart ended 4–7 levels apart under the clip and the 16-bit history (the picture gate's own
/// G-DETERMINISM: the ground 4, the hill 7), 1.77e-3 apart in tone even with a 32-bit history (177
/// times the input); under this rectification within one level, and the picture at one jitter phase
/// repeats exactly every cycle (0 pixels moved, about 190 a cycle under the clip). Example: on the
/// hill stand's dune slope, a history a hair outside its pixel's box lands on the box's nearest face
/// in both of two flights, and the two flights draw the slope alike.
#[must_use]
pub fn rectify(
    history: [f64; 3],
    centre: [f64; 3],
    extent: [f64; 3],
    lo: [f64; 3],
    hi: [f64; 3],
) -> [f64; 3] {
    [0, 1, 2].map(|i| {
        let lower = (centre[i] - extent[i]).max(lo[i]);
        let upper = (centre[i] + extent[i]).min(hi[i]);
        history[i].max(lower).min(upper)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: ReconstructionSettings = ReconstructionSettings::PROPOSED;

    fn close(a: f64, b: f64, within: f64) -> bool {
        (a - b).abs() <= within
    }

    /// ★ THE PHASES: ⌈8 ÷ s²⌉ at the design's scales, and a nonsense scale held at the rules' bounds.
    #[test]
    fn the_jitter_runs_eight_over_the_scale_squared_phases() {
        assert_eq!([1.0, 0.75, 0.67, 0.5].map(jitter_phases), [8, 15, 18, 32]);
        // Two thirds read off the capture's sizes squares a last bit off; it is still 18.
        assert_eq!(jitter_phases(720.0 / 1_080.0), 18);
        assert_eq!(jitter_phases(2.0 / 3.0), 18);
        assert_eq!(jitter_phases(2.0), 8);
        assert_eq!(
            jitter_phases(0.0),
            (8.0 / (SCALE_MIN * SCALE_MIN)).ceil() as u32
        );
    }

    /// The Halton sequence's first values in bases 2 and 3, and nought at index nought; a base under
    /// two reads as two.
    #[test]
    fn the_halton_sequence_mirrors_the_digits() {
        assert_eq!(
            [0, 1, 2, 3, 4].map(|i| halton(i, 2)),
            [0.0, 0.5, 0.25, 0.75, 0.125]
        );
        let third = [1, 2, 3].map(|i| halton(i, 3));
        assert!(close(third[0], 1.0 / 3.0, 1e-12));
        assert!(close(third[1], 2.0 / 3.0, 1e-12));
        assert!(close(third[2], 1.0 / 9.0, 1e-12));
        assert_eq!(halton(3, 1), halton(3, 2));
    }

    /// ★ NO NULL JITTER: every phase of the cycle at each scale lies in [−0.5, 0.5) on both axes, none
    /// is (0, 0), all are distinct, and the cycle repeats.
    #[test]
    fn every_phase_is_a_distinct_offset_and_none_is_null() {
        for phases in [8u32, 15, 18, 32] {
            let points: Vec<[f64; 2]> = (0..phases).map(|p| jitter(p, phases)).collect();
            for [x, y] in &points {
                assert!((-0.5..0.5).contains(x), "{x}");
                assert!((-0.5..0.5).contains(y), "{y}");
                assert_ne!([*x, *y], [0.0, 0.0]);
            }
            for (i, a) in points.iter().enumerate() {
                for b in &points[i + 1..] {
                    assert_ne!(a, b);
                }
            }
            assert_eq!(jitter(phases, phases), jitter(0, phases));
        }
        // A cycle of nought phases reads as a cycle of one.
        assert_eq!(jitter(5, 0), jitter(0, 1));
    }

    /// ★ THE CLOCK: one more a frame, nought on a reset; two captures from a reset see one sequence.
    /// Whole cycles at or past the cap: 64 at 8 and at 32 phases, 72 at 18, 75 at 15, the cap alone
    /// at one phase; a phase count of nought reads as one.
    #[test]
    fn a_converged_still_waits_whole_cycles_past_the_count_cap() {
        let s = ReconstructionSettings::PROPOSED;
        assert_eq!(converged_frames(8, &s), 64);
        assert_eq!(converged_frames(32, &s), 64);
        assert_eq!(converged_frames(18, &s), 72);
        assert_eq!(converged_frames(15, &s), 75);
        assert_eq!(converged_frames(1, &s), 64);
        assert_eq!(converged_frames(0, &s), 64);
    }

    #[test]
    fn the_jitter_clock_counts_from_nought_at_each_reset() {
        let mut clock = JitterClock::default();
        assert_eq!([false, false, false].map(|r| clock.advance(r)), [1, 2, 3]);
        assert_eq!(clock.advance(true), 0);
        assert_eq!(clock.advance(false), 1);
        let mut wrapped = JitterClock { phase: u32::MAX };
        assert_eq!(wrapped.advance(false), 0);
    }

    /// ★ THE FLOOR LAW: 720 ÷ the rows, never over one; a wish held between the floor and one.
    #[test]
    fn the_floor_law_keeps_a_render_pixel_within_a_reference_pixel() {
        assert!(close(scale_floor(2_168), 720.0 / 2_168.0, 1e-12));
        assert!(close(scale_floor(1_080), 2.0 / 3.0, 1e-12));
        assert_eq!(scale_floor(720), 1.0);
        assert_eq!(scale_floor(480), 1.0);
        assert_eq!(scale_floor(0), 1.0);
        assert_eq!(lawful_scale(0.5, 2_168), 0.5);
        assert!(close(lawful_scale(0.2, 2_168), 720.0 / 2_168.0, 1e-12));
        assert_eq!(lawful_scale(1.5, 2_168), 1.0);
        assert_eq!(lawful_scale(0.5, 1_080), 2.0 / 3.0);
    }

    /// ★ THE NARROWING: one at full size, 1 ÷ s below, never past FSR 2's cap.
    #[test]
    fn the_filter_narrows_by_one_over_the_scale_to_the_cap() {
        assert_eq!(kernel_bias(1.0), 1.0);
        assert!(close(kernel_bias(0.75), 4.0 / 3.0, 1e-12));
        assert_eq!(kernel_bias(0.5), KERNEL_BIAS_MAX);
        assert_eq!(kernel_bias(0.2), KERNEL_BIAS_MAX);
        assert_eq!(kernel_bias(3.0), 1.0);
    }

    /// The render size: each side scaled and rounded, never nought.
    #[test]
    fn the_render_size_rounds_each_side_and_never_reads_nought() {
        assert_eq!(render_size([3_456, 2_168], 0.5), [1_728, 1_084]);
        assert_eq!(render_size([1_284, 720], 1.0), [1_284, 720]);
        assert_eq!(render_size([1, 1], 0.4), [1, 1]);
    }

    /// ★ THE NEAREST SAMPLE: its sample lies within half a pixel of the point on each axis, whatever the
    /// jitter, and its neighbours' samples lie one pixel further on.
    #[test]
    fn the_nearest_sample_lies_within_half_a_pixel_of_the_point() {
        for k in 0..400 {
            let at = [0.37 * f64::from(k), 1.91 * f64::from(k % 17) + 0.004];
            let j = jitter(k as u32, 32);
            let near = nearest_sample(at, j);
            let offset = sample_offset(near, j, at);
            // Within half a pixel: a tie goes to the lower pixel, so the offset lies in (−½, ½].
            for axis in offset {
                assert!((axis > -0.5) & (axis <= 0.5), "{at:?} {j:?} {offset:?}");
            }
            let right = sample_offset([near[0] + 1, near[1]], j, at);
            assert!(close(right[0], offset[0] + 1.0, 1e-9));
        }
        // A worked case: the point (10.5, 3.5) with the jitter (0.25, −0.25) — pixel (10, 3) samples
        // (10.75, 3.25), a quarter pixel off on each axis.
        assert_eq!(sample_point([10, 3], [0.25, -0.25]), [10.75, 3.25]);
        assert_eq!(sample_point([0, 0], [0.0, 0.0]), [0.5, 0.5]);
        assert_eq!(nearest_sample([10.5, 3.5], [0.25, -0.25]), [10, 3]);
        assert_eq!(
            sample_offset([10, 3], [0.25, -0.25], [10.5, 3.5]),
            [0.25, -0.25]
        );
    }

    /// ★ THE CAPTURE KEEPS ITS GRID: the display is the grid ÷ s, and the render size at s of that
    /// display is the grid again at every scale.
    #[test]
    fn the_capture_display_grows_by_one_over_the_scale_and_renders_its_grid_again() {
        assert_eq!(display_size([1_284, 720], 0.5), [2_568, 1_440]);
        assert_eq!(display_size([1_284, 720], 1.0), [1_284, 720]);
        assert_eq!(display_size([1_284, 720], 0.67), [1_916, 1_075]);
        assert_eq!(display_size([1_284, 720], 3.0), [1_284, 720]);
        for grid in [[1_284_u32, 720], [1, 1], [3_456, 2_168]] {
            for k in 25..=100 {
                let s = f64::from(k) / 100.0;
                assert_eq!(
                    render_size(display_size(grid, s), s),
                    grid,
                    "{grid:?} at {s}"
                );
            }
        }
    }

    /// The view depth: near ÷ depth, the sky's nought the farthest and finite.
    #[test]
    fn a_depth_reads_as_near_over_depth_and_the_sky_as_the_farthest() {
        assert_eq!(view_depth(0.5, 0.1), 0.2);
        assert_eq!(view_depth(1.0, 0.1), 0.1);
        let sky = view_depth(0.0, 0.1);
        assert!(sky.is_finite(), "{sky}");
        assert!(sky > 1e28, "{sky}");
    }

    /// ★ THE SEPARATION: the corner factor of the 45° view on 16 : 9, and the capture's separation of
    /// about 2.6 % of the depth; the power from one to three.
    #[test]
    fn the_separation_scales_with_the_field_and_the_picture() {
        let tan = (std::f64::consts::PI / 8.0).tan();
        let k = corner_factor(1.0 / (tan * 16.0 / 9.0), 1.0 / tan);
        assert!(close(
            k,
            (1.0 + (tan * 16.0 / 9.0).powi(2) + tan * tan).sqrt(),
            1e-12
        ));
        assert!(close(k, 1.3091, 1e-4), "{k}");
        let sep = depth_separation([1_284, 720], k, &S);
        assert!(close(sep, 0.0264, 1e-3), "{sep}");
        assert_eq!(depth_clip_power([0, 0]), 1.0);
        assert_eq!(depth_clip_power([1_920, 1_080]), 3.0);
        assert_eq!(depth_clip_power([3_456, 2_168]), 3.0);
        assert!(close(depth_clip_power([960, 540]), 2.0, 1e-12));
    }

    /// ★ THE FOOTPRINT: a pixel centre is one tap alone; half way between four centres, a quarter each;
    /// the weights always sum to one.
    #[test]
    fn the_bilinear_footprint_weighs_the_four_nearest_centres() {
        assert_eq!(
            bilinear_footprint([3.5, 7.5]),
            ([3, 7], [1.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(bilinear_footprint([4.0, 8.0]), ([3, 7], [0.25; 4]));
        let (base, w) = bilinear_footprint([0.2, 0.9]);
        assert_eq!(base, [-1, 0]);
        assert!(close(w.iter().sum::<f64>(), 1.0, 1e-12));
        assert!(close(w[1], 0.7 * 0.6, 1e-12), "{w:?}");
    }

    /// ★ THE DEPTH TEST: a far valley behind a near rock is a disocclusion; the same surface, a
    /// farther one, a gap under the separation or no tap at all keep the history; the sky behind a
    /// surface is uncovered; partial cover is weighed by the taps.
    #[test]
    fn a_surface_hidden_last_frame_takes_no_history() {
        let near = 0.1;
        let sep = 0.025;
        let depth_at = |m: f64| near / m;
        // The valley at 3 000 m, the rock at 40 m: hidden.
        let hidden = depth_clip(depth_at(3_000.0), &[(depth_at(40.0), 1.0)], near, sep, 3.0);
        assert!(hidden > 0.999, "{hidden}");
        // The same surface, or a farther one, or nothing that stood there: seen.
        assert_eq!(
            depth_clip(depth_at(500.0), &[(depth_at(500.0), 1.0)], near, sep, 3.0),
            0.0
        );
        assert_eq!(
            depth_clip(depth_at(500.0), &[(depth_at(900.0), 1.0)], near, sep, 3.0),
            0.0
        );
        assert_eq!(
            depth_clip(depth_at(500.0), &[(0.0, 1.0)], near, sep, 3.0),
            0.0
        );
        assert_eq!(depth_clip(depth_at(500.0), &[], near, sep, 3.0), 0.0);
        // A gap within the separation (the ground's own slope across a pixel): kept.
        assert_eq!(
            depth_clip(depth_at(500.0), &[(depth_at(495.0), 1.0)], near, sep, 3.0),
            0.0
        );
        // The sky behind a surface that moved away: uncovered (all but the separation's cube).
        let sky = depth_clip(0.0, &[(depth_at(40.0), 1.0)], near, sep, 3.0);
        assert!(close(sky, 1.0 - sep.powi(3), 1e-9), "{sky}");
        // Between the ends it rises smoothly with the gap.
        let gaps = [520.0, 560.0, 700.0]
            .map(|m| depth_clip(depth_at(m), &[(depth_at(500.0), 1.0)], near, sep, 3.0));
        assert!(gaps.windows(2).all(|w| w[0] < w[1]), "{gaps:?}");
    }

    /// ★ THE REACTIVE VALUE: nought where the transparent pass changed nothing or little; the settings'
    /// value from the threshold on; the change read through the tone map, so a bright pixel's large
    /// change counts as much as a dim pixel's.
    #[test]
    fn a_pixel_the_transparent_pass_changed_is_reactive() {
        let sky = [0.2, 0.3, 0.6];
        assert_eq!(reactive(sky, sky, &S), 0.0);
        // A faint glass: under the threshold.
        assert_eq!(reactive(sky, [0.25, 0.33, 0.62], &S), 0.0);
        // A hull's box over the sky: far over it.
        assert_eq!(reactive(sky, [0.9, 0.8, 0.2], &S), S.reactive_value);
        // Negative light reads as black.
        assert_eq!(reactive([-1.0, -1.0, -1.0], [0.0; 3], &S), 0.0);
        // Through the tone map: 10 → 20 is a small change of a bright pixel.
        assert_eq!(reactive([10.0; 3], [20.0; 3], &S), 0.0);
    }

    /// ★ THE COUNT: one more than the history's, capped; a full disocclusion or a restart starts at one;
    /// a partial one lowers it in proportion.
    #[test]
    fn the_count_rises_by_one_and_the_depth_test_lowers_it() {
        assert_eq!(next_count(7.0, 1.0, 64.0, 0.0, false), 8.0);
        assert_eq!(next_count(64.0, 1.0, 64.0, 0.0, false), 64.0);
        assert_eq!(next_count(7.0, 1.0, 64.0, 1.0, false), 1.0);
        assert_eq!(next_count(8.0, 1.0, 64.0, 0.5, false), 5.0);
        assert_eq!(next_count(7.0, 0.25, 64.0, 0.0, true), 0.25);
        // A frame that earned nothing keeps the count: the history carries the pixel.
        assert_eq!(next_count(7.0, 0.0, 64.0, 0.0, false), 7.0);
    }

    /// ★ THE RESET DECISION: the first cause that holds, in order; none on the player's path.
    #[test]
    fn the_history_restarts_only_without_history_on_a_lost_device_a_new_size_or_a_tool_jump() {
        let size = [3_456, 2_168];
        assert_eq!(reset_cause(true, false, size, size, false), None);
        assert_eq!(
            reset_cause(false, true, size, [0, 0], true),
            Some(ResetCause::FirstFrame)
        );
        assert_eq!(
            reset_cause(true, true, size, [0, 0], true),
            Some(ResetCause::DeviceLost)
        );
        assert_eq!(
            reset_cause(true, false, size, [1_728, 1_084], true),
            Some(ResetCause::DisplaySize)
        );
        assert_eq!(
            reset_cause(true, false, size, size, true),
            Some(ResetCause::ToolJump)
        );
    }

    /// ★ THE FLOOR BY MOTION: the still floor under a hundredth of a pixel, the moving floor from one
    /// pixel, the line in the logarithm between (a tenth of a pixel is half way), and nought motion
    /// is still.
    #[test]
    fn the_weight_floor_rises_smoothly_from_the_still_to_the_moving_floor() {
        assert_eq!(weight_floor(0.0, &S), S.still_floor);
        assert_eq!(weight_floor(0.005, &S), S.still_floor);
        assert_eq!(weight_floor(1.0, &S), S.moving_floor);
        assert_eq!(weight_floor(40.0, &S), S.moving_floor);
        let mid = weight_floor(0.1, &S);
        assert!(
            close(mid, (S.still_floor + S.moving_floor) * 0.5, 1e-12),
            "{mid}"
        );
        // Monotone between the ends.
        let steps = [0.02, 0.05, 0.2, 0.5].map(|m| weight_floor(m, &S));
        assert!(steps.windows(2).all(|w| w[0] < w[1]), "{steps:?}");
    }

    /// ★ THE FILTER FOR ONE PIXEL: the scale's narrowest while the pixel keeps its history; FSR 2's widest
    /// for one that starts again — one at half scale, so its filter spans two render pixels.
    #[test]
    fn a_pixel_that_starts_again_gathers_with_the_widest_filter() {
        assert_eq!(pixel_kernel_bias(KERNEL_BIAS_MAX, 0.0), KERNEL_BIAS_MAX);
        assert_eq!(pixel_kernel_bias(KERNEL_BIAS_MAX, 1.0), 1.0);
        assert_eq!(pixel_kernel_bias(1.0, 1.0), 1.0);
        assert!(close(
            pixel_kernel_bias(KERNEL_BIAS_MAX, 0.5),
            (KERNEL_BIAS_MAX + 1.0) * 0.5,
            1e-12
        ));
        // A narrowest of four would widen to (1 + 4) × 0.3 = 1.5, over one.
        assert!(close(pixel_kernel_bias(4.0, 1.0), 1.5, 1e-12));
        assert_eq!(pixel_kernel_bias(KERNEL_BIAS_MAX, 7.0), 1.0);
        // The radial weight: one on the sample, Lanczos-2 of the biased distance beside it.
        assert_eq!(upsample_weight([0.0, 0.0], KERNEL_BIAS_MAX), 1.0);
        assert!(close(
            upsample_weight([0.3, 0.4], 1.0),
            lanczos2(0.5),
            1e-12
        ));
    }

    /// The 3 × 3 samples of a display pixel on the dune's edge at half scale, the nearest first: the dune
    /// (0.2) in the column at `dark_x`, the haze (0.8) in the two columns after it.
    fn edge(dark_x: f64) -> Vec<([f64; 2], f64, bool)> {
        let mut samples = Vec::new();
        for dy in [0.0, -1.0, 1.0] {
            for (dx, v) in [(1.0, 0.8), (0.0, 0.2), (2.0, 0.8)] {
                samples.push(([dark_x + dx, dy], v, true));
            }
        }
        samples
    }

    /// ★ THE FLASH, MEASURED AND CURED (2026-09-30, the owner: *"constant flashes of light ... where dark
    /// and bright parts are connected"*): a display pixel 0.6 render pixel from the dune's sample and 0.4
    /// from the haze's — the first build's filter (separable, normalized by its sum) read 1.25 there, above
    /// the haze's 0.8: a pixel brighter than the sky. The upsample stays inside the samples' range. Half
    /// way between them the pixel earns almost nothing and keeps its history; on a sample it earns a
    /// whole weight and reads that sample.
    #[test]
    fn a_pixel_between_a_dark_and_a_bright_sample_never_overshoots() {
        let bias = kernel_bias(0.5);
        let near_haze = edge(-0.6);
        let first_build = {
            let (sum, weight) = near_haze.iter().fold((0.0, 0.0), |(s, w), &(o, v, _)| {
                let k = lanczos2(o[0] * bias) * lanczos2(o[1] * bias);
                (s + v * k, w + k)
            });
            sum / weight
        };
        assert!(first_build > 1.2, "{first_build}");
        let (value, earned) = upsample(&near_haze, bias, &S);
        assert!((0.2..=0.8).contains(&value), "{value}");
        assert!(earned > S.upsample_weight_min, "{earned}");
        // Half way: almost nothing earned.
        let (value, earned) = upsample(&edge(-0.5), bias, &S);
        assert!((0.2..=0.8).contains(&value), "{value}");
        assert!(earned < 0.01, "{earned}");
        // On a sample: a whole weight, that sample's value.
        let on = vec![
            ([0.0, 0.0], 0.8, true),
            ([1.0, 0.0], 0.2, true),
            ([-1.0, 0.0], 0.2, true),
        ];
        let (value, earned) = upsample(&on, bias, &S);
        assert!(close(value, 0.8, 1e-9), "{value}");
        // (The two neighbours stand just inside Lanczos-2's negative lobe, at a biased 1.99.)
        assert!(close(earned, 1.0, 1e-3), "{earned}");
        // A sample off the picture earns nothing; with none on it, the nearest's value and no weight.
        assert_eq!(upsample(&[([0.0, 0.0], 0.8, false)], bias, &S), (0.8, 0.0));
        assert_eq!(upsample(&[], bias, &S), (0.0, 0.0));
    }

    /// ★ THE BLEND: the earned weight over the count, the count capped by motion — a frame that earns a
    /// whole weight keeps the design's floors; motion alone never restarts the count.
    #[test]
    fn the_blend_weight_is_the_earned_weight_over_the_count_capped_by_motion() {
        assert_eq!(blend_weight(1.0, 1.0), 1.0);
        assert_eq!(blend_weight(1.0, 4.0), 0.25);
        assert_eq!(blend_weight(0.0, 4.0), 0.0);
        assert_eq!(blend_weight(0.0, 0.0), 1.0);
        // The caps by motion hold the design's floors for a frame that earns a whole weight.
        assert_eq!(count_cap(0.0, &S), 1.0 / S.still_floor);
        assert_eq!(count_cap(3.0, &S), 1.0 / S.moving_floor);
        let still = next_count(1_000.0, 1.0, count_cap(0.0, &S), 0.0, false);
        assert_eq!(blend_weight(1.0, still), S.still_floor);
        let moving = next_count(1_000.0, 1.0, count_cap(3.0, &S), 0.0, false);
        assert_eq!(blend_weight(1.0, moving), S.moving_floor);
    }

    /// ★ THE FILTER: one at the centre, nought at the pixel's whole steps and from two pixels on, and
    /// symmetric; its negative lobe lies between one and two pixels.
    #[test]
    fn the_lanczos_filter_is_one_at_the_centre_and_nought_past_two_pixels() {
        assert_eq!(lanczos2(0.0), 1.0);
        assert!(close(lanczos2(1.0), 0.0, 1e-12));
        assert_eq!(lanczos2(2.0), 0.0);
        assert_eq!(lanczos2(-3.5), 0.0);
        assert_eq!(lanczos2(0.4), lanczos2(-0.4));
        assert!(lanczos2(0.5) > 0.0);
        assert!(lanczos2(1.5) < 0.0);
    }

    /// The tone weight: one for black, a half at luminance one, and a negative luminance counts as
    /// black.
    #[test]
    fn the_tone_weight_tames_a_bright_sample() {
        assert_eq!(tone_weight(0.0), 1.0);
        assert_eq!(tone_weight(1.0), 0.5);
        assert_eq!(tone_weight(-2.0), 1.0);
        assert!(tone_weight(100.0) < 0.01);
    }

    /// YCoCg reads back to RGB exactly, and white is luminance with no chroma.
    #[test]
    fn ycocg_round_trips_and_white_has_no_chroma() {
        for rgb in [[1.0, 1.0, 1.0], [0.25, 0.5, 0.75], [3.0, 0.0, 0.1]] {
            let back = ycocg_to_rgb(rgb_to_ycocg(rgb));
            for i in 0..3 {
                assert!(close(back[i], rgb[i], 1e-12), "{rgb:?} {back:?}");
            }
        }
        assert_eq!(rgb_to_ycocg([1.0, 1.0, 1.0]), [1.0, 0.0, 0.0]);
    }

    /// ★ THE RECTIFICATION: a history inside the box stays; outside it is clamped axis by axis onto
    /// the box (two axes out move independently); the box is cut to the samples' own range; a box of
    /// nought on an axis holds only its centre there; a box that rounding inverts takes its upper bound.
    #[test]
    fn the_history_is_clamped_into_the_box_cut_to_the_samples_range() {
        let c = [0.5, 0.0, 0.0];
        let e = [0.1, 0.05, 0.05];
        let wide = ([-1.0; 3], [1.0; 3]);
        let r = |h: [f64; 3], e: [f64; 3], (lo, hi): ([f64; 3], [f64; 3])| rectify(h, c, e, lo, hi);
        assert_eq!(r([0.55, 0.01, -0.02], e, wide), [0.55, 0.01, -0.02]);
        let out = r([0.9, 0.0, 0.0], e, wide);
        assert!(close(out[0], 0.6, 1e-12), "{out:?}");
        // Two axes out: each clamped on its own axis.
        let both = r([0.7, 0.2, 0.0], e, wide);
        assert!(close(both[0], 0.6, 1e-12), "{both:?}");
        assert!(close(both[1], 0.05, 1e-12), "{both:?}");
        // The samples' own range cuts the box: the deviations reach 0.6, the samples only 0.55.
        let cut = r([0.9, 0.0, 0.0], e, ([0.45, -1.0, -1.0], [0.55, 1.0, 1.0]));
        assert!(close(cut[0], 0.55, 1e-12), "{cut:?}");
        // A flat box: only the centre on that axis.
        let flat = r([0.5, 0.3, 0.0], [0.1, 0.0, 0.1], wide);
        assert_eq!(flat[1], 0.0);
        // A box inverted by rounding (the range just under the centre): the upper bound.
        let inverted = rectify([0.7; 3], [0.5; 3], [0.0; 3], [0.5 + 1e-9; 3], [0.5; 3]);
        assert_eq!(inverted, [0.5; 3]);
    }

    /// ★ TWO COLOURS NEVER MOVE APART (ruling B28 item 3, the amplifier's cure): over many pairs of
    /// histories and many boxes, the rectification carries no two colours farther apart on any axis.
    /// CONTROL: the clip toward the centre, the first build's rule, fails the same reading — near a
    /// corner of a flat box, two histories that differ only on the narrow axis land 0.048 apart on the
    /// wide one.
    #[test]
    fn the_rectification_never_carries_two_colours_apart() {
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut worst = 0.0_f64;
        for _ in 0..20_000 {
            let centre = [next(), next(), next()];
            let extent = [next().abs(), next().abs() * 0.1, next().abs()];
            let lo = [0, 1, 2].map(|i| centre[i] - next().abs());
            let hi = [0, 1, 2].map(|i| centre[i] + next().abs());
            let a = [next() * 2.0, next() * 2.0, next() * 2.0];
            let b = [0, 1, 2].map(|i| a[i] + next() * 0.01);
            let (ra, rb) = (
                rectify(a, centre, extent, lo, hi),
                rectify(b, centre, extent, lo, hi),
            );
            for i in 0..3 {
                worst = worst.max((ra[i] - rb[i]).abs() - (a[i] - b[i]).abs());
            }
        }
        assert!(worst <= 1e-12, "a pair moved apart by {worst}");
        // The control: the clip toward the centre on a box of (1, 0.1), two histories (2, 0.21) and
        // (2, 0.19) that differ only on the narrow axis.
        let clip = |h: [f64; 2]| {
            let reach = (h[0].abs() / 1.0).max(h[1].abs() / 0.1).max(1.0);
            [h[0] / reach, h[1] / reach]
        };
        let (ca, cb) = (clip([2.0, 0.21]), clip([2.0, 0.19]));
        let gap = (ca[0] - cb[0]).abs();
        assert!(gap > 0.04, "the clip carried them {gap} apart");
        let wide = rectify(
            [2.0, 0.21, 0.0],
            [0.0; 3],
            [1.0, 0.1, 1.0],
            [-9.0; 3],
            [9.0; 3],
        );
        let narrow = rectify(
            [2.0, 0.19, 0.0],
            [0.0; 3],
            [1.0, 0.1, 1.0],
            [-9.0; 3],
            [9.0; 3],
        );
        assert_eq!(wide[0], narrow[0]);
    }
}
