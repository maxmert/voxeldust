//! ★ THE EYE'S RULE (the temporal path's T2, 2026-10-01; the owner: "I need realistic experience, not
//! sure that common rule for all distances will work here" → "Patches by eye") — the pure half.
//!
//! A pop is a change of the picture that a player's EYE sees. One count of levels cannot say that for
//! every distance: a step of 30 levels on a speck a pixel wide is gone before the eye finds it, and a
//! step of two levels over a band across the whole view is a flash. So the rule judges each change
//! as the eye meets it: a PATCH of the picture that got brighter (or darker) at once, by its size on
//! the eye and its contrast against the light around it, against the eye's published thresholds.
//!
//! **The thresholds.** Blackwell (1946, JOSA 36, 624) measured, for a uniform disc on a uniform field,
//! the contrast that his observers saw half of the time, for discs 0.6′ to 6° across on fields from
//! starlight to daylight. Crumey (2014, MNRAS 442, 2600) fitted the whole table with one smooth law:
//! under RICCÒ's area the eye sums a patch's light, so the threshold falls with the area (`R / A`);
//! past it the threshold stands at the Weber fraction `C∞`; one exponent `q` joins the two:
//! `C_th = ((R / A)^q + C∞^q)^(1/q)` ([`threshold_contrast`]). `R` ([`ricco_sr`]), `C∞`
//! ([`weber_contrast`]) and `q` ([`joining_exponent`]) are functions of the field's luminance, with
//! the paper's coefficients for Blackwell's data (its equations 25 and 28, 39 and 40, 42 to 44).
//!
//! **The display.** A picture's codes are light on a screen: sRGB (IEC 61966-2-1), white at 200 cd/m²
//! and black at a thousandth of it ([`REFERENCE_DISPLAY`]); the luminance is the CIE Y of the linear
//! light. **The seat.** The picture's vertical field fills the same angle on the eye
//! ([`arcmin_per_px`]): the ground stands where the camera says, so a patch subtends on the eye what it
//! subtends in the world — at the capture's 45° over 720 rows, a pixel's side is 3.75′.
//!
//! **The change.** Each pixel the pop judge compared (`crate::pop::judge_pair_mapped`) carries where its
//! ground reprojected in the other frame; its change is the luminance now less the luminance there —
//! read BETWEEN that frame's samples (MEASURED: against the nearest sample alone, a smooth slope that
//! slid part of a pixel read one level darker over a whole dune, 29 131 "seen" patches of still ground on
//! the 240 m/s leg; between the samples, 7 084), or at the judge's own best sample where an edge of
//! shade slid — whichever explains the light better ([`light_at`]). The display picture is read UNDER
//! the engine's screen dither ([`screen_dither`], a pattern fixed on the screen that the ground slides
//! under), and every pixel carries its ROUNDING BAND: one code of its eight bits, which no reading can
//! resolve ([`Display::light_image`]).
//!
//! **Held, or a flicker.** A pop HOLDS: the ground stood still before the step and keeps its new light
//! after it. The flight records four frames a pair; the eye's step is the middle two, and a change
//! counts as held only when both frames after the step differ from both frames before it, all in one
//! direction ([`held_of`]; a lone frame on one side would let that frame's own dip read as a held
//! step). The rest of the change is a flicker, seen for one frame ([`ONE_FRAME`]). The eye adapts to
//! the mean luminance of the second picture over [`ADAPTING_FIELD_DEG`] around.
//!
//! **Two ways to meet a change**: a PATCH ([`patches`]) — the pixels of one class (still, edge, a rung
//! boundary) and one sign of change BEYOND the rounding band, joined through their eight neighbours: its
//! area is its pixels, its contrast its summed change over its summed field; and a WINDOW ([`windows`])
//! — a square of the picture at each size from one pixel to 128, where the eye's own summing is
//! modelled on the whole change: a changed pixel and an unchanged one average, a brighter one and a
//! darker one cancel, and so does the rounding. A compact pop shows as a patch; a faint pop over a wide
//! field shows in a window at least [`WIDE_WINDOW_ARCMIN`] wide ([`pops`]).
//!
//! **The verdict.** The visibility level is the contrast over the threshold times the part's temporal
//! factor ([`HELD_STEP`] for a held change, as Blackwell's disc held for six seconds; [`ONE_FRAME`] for
//! a flicker). At one or more the eye sees it half of the time ([`SEEN`]).
//!
//! Example: the hull flies at 240 m/s over the dunes; between two frames a rung-2 band 1.5° long and
//! 0.1° wide under the hull steps two levels brighter and stays — 0.15 square degrees at 3 % on a
//! 40 cd/m² field, where the eye's threshold is 0.3 %: visibility level ten, a pop the rule names by its
//! rung. A lone pixel that stepped one level darker for one frame — 14 square arcminutes at 1.3 % where
//! the threshold is 2 %, and a flicker besides — stays unseen.

use crate::pop::{Frame, PixelClass, PixelRead, judge_pair_mapped};

/// ★ THE DISPLAY THE EYE LOOKS AT: the luminance of its white and of its black, cd/m².
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Display {
    pub white_cd_m2: f64,
    pub black_cd_m2: f64,
}

/// THE REFERENCE DISPLAY (a stated seat, the research's defaults): white at 200 cd/m², black at a
/// thousandth of it; no light reflected off the screen.
pub const REFERENCE_DISPLAY: Display = Display {
    white_cd_m2: 200.0,
    black_cd_m2: 0.2,
};

/// The CIE Y weights of linear sRGB's red, green and blue.
const Y_WEIGHTS: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// THE LINEAR LIGHT of an sRGB code (IEC 61966-2-1), whole or between the whole codes, clamped to the
/// codes' range: the straight segment under 0.04045, the power 2.4 over it.
#[must_use]
pub fn srgb_linear(code: f64) -> f64 {
    let c = code.clamp(0.0, 255.0) / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// THE LINEAR LIGHT of each whole sRGB code ([`srgb_linear`]).
#[must_use]
pub fn srgb_linear_table() -> [f64; 256] {
    let mut t = [0.0; 256];
    for (code, v) in t.iter_mut().enumerate() {
        *v = srgb_linear(code as f64);
    }
    t
}

/// ★ THE ENGINE'S SCREEN DITHER (Bevy 0.18's `screen_space_dither` under `DebandDither::Enabled`, every
/// 3D camera's default, which the game's cameras keep): the part of a code, per channel, that the
/// tone-mapping pass added before the picture was rounded to eight bits, at pixel `(x, y)` — a pattern
/// fixed on the SCREEN, which the ground slides under as the eye moves. MEASURED on the walk's display
/// picture: the ground's step from a pixel to its neighbour 0.9 of a level, 0.55 with the dither taken
/// off; the render picture holds none (0.01).
#[must_use]
pub fn screen_dither(x: usize, y: usize) -> [f64; 3] {
    let v = 171.0 * (x as f64 + 0.5) + 231.0 * (y as f64 + 0.5);
    [103.0, 71.0, 97.0].map(|m: f64| (v / m).rem_euclid(1.0) - 0.5)
}

impl Display {
    /// The luminance of a pixel's code on this display, cd/m².
    #[must_use]
    pub fn luminance(&self, table: &[f64; 256], rgb: [u8; 3]) -> f64 {
        let y = Y_WEIGHTS[0] * table[usize::from(rgb[0])]
            + Y_WEIGHTS[1] * table[usize::from(rgb[1])]
            + Y_WEIGHTS[2] * table[usize::from(rgb[2])];
        self.black_cd_m2 + (self.white_cd_m2 - self.black_cd_m2) * y
    }

    /// The luminance of every pixel of an RGBA picture, cd/m².
    #[must_use]
    pub fn luminance_image(&self, rgba: &[u8]) -> Vec<f64> {
        let table = srgb_linear_table();
        rgba.chunks_exact(4)
            .map(|p| self.luminance(&table, [p[0], p[1], p[2]]))
            .collect()
    }

    /// The luminance of codes whole or between the whole codes, cd/m².
    #[must_use]
    pub fn luminance_of_codes(&self, codes: [f64; 3]) -> f64 {
        let y = Y_WEIGHTS[0] * srgb_linear(codes[0])
            + Y_WEIGHTS[1] * srgb_linear(codes[1])
            + Y_WEIGHTS[2] * srgb_linear(codes[2]);
        self.black_cd_m2 + (self.white_cd_m2 - self.black_cd_m2) * y
    }

    /// ★ A PICTURE AS LIGHT: each pixel's luminance, cd/m², read under the engine's dither when the
    /// picture carries it ([`screen_dither`]); and each pixel's ROUNDING BAND — the luminance of one code
    /// at its level, half a code each way, which no reading of an eight-bit picture resolves.
    #[must_use]
    pub fn light_image(&self, rgba: &[u8], width: usize, dithered: bool) -> (Vec<f64>, Vec<f64>) {
        let w = width.max(1);
        rgba.chunks_exact(4)
            .enumerate()
            .map(|(i, p)| {
                let d = screen_dither(i % w, i / w).map(|v| v * f64::from(u8::from(dithered)));
                let c = [0, 1, 2].map(|k| f64::from(p[k]) - d[k]);
                let light = self.luminance_of_codes(c);
                let band = self.luminance_of_codes(c.map(|v| v + 0.5))
                    - self.luminance_of_codes(c.map(|v| v - 0.5));
                (light, band)
            })
            .unzip()
    }
}

/// Steradians in one square arcminute.
pub const SR_PER_ARCMIN2: f64 =
    (std::f64::consts::PI / 10_800.0) * (std::f64::consts::PI / 10_800.0);

/// ★ RICCÒ'S CONSTANT in steradians on a field of `field` cd/m² (Crumey 2014, equations 25 and 28, the
/// whole range of Blackwell's fields): under Riccò's area the threshold contrast is `R / A`.
#[must_use]
pub fn ricco_sr(field: f64) -> f64 {
    let (a1, a2, a3, a4, a5) = (5.949e-8, -2.389e-7, 2.459e-7, 4.120e-4, -4.225e-4);
    let s = field.powf(-0.25);
    let r = (a1 * s * s + a2 * s + a3).sqrt() + a4 * s + a5;
    r * r
}

/// ★ THE WEBER CONTRAST of a large patch on a field of `field` cd/m² (Crumey 2014, equations 39 and
/// 40): the threshold the contrast falls to as the area grows.
#[must_use]
pub fn weber_contrast(field: f64) -> f64 {
    let (b1, b2, b3, b4, b5) = (9.606e-6, -4.112e-5, 5.019e-5, 4.837e-3, -4.884e-3);
    let s = field.powf(-0.25);
    (b1 * s * s + b2 * s + b3).sqrt() + b4 * s + b5
}

/// THE JOINING EXPONENT on a field of `field` cd/m² (Crumey 2014, equations 42 to 44): day, twilight,
/// night.
#[must_use]
pub fn joining_exponent(field: f64) -> f64 {
    if field >= 3.40 {
        1.146 - 0.0885 * field.log10()
    } else if field >= 0.193 {
        0.8861 + 0.4 * field.log10()
    } else {
        0.6
    }
}

/// ★ THE EYE'S THRESHOLD CONTRAST for a patch of `area_sr` steradians on a field of `field` cd/m²
/// (Crumey 2014, equation 41): the contrast that Blackwell's observers saw half of the time.
#[must_use]
pub fn threshold_contrast(area_sr: f64, field: f64) -> f64 {
    let q = joining_exponent(field);
    ((ricco_sr(field) / area_sr).powf(q) + weber_contrast(field).powf(q)).powf(1.0 / q)
}

/// ★ THE SEAT: a pixel's side in arcminutes of the eye when the picture's vertical field `fov_y`
/// (radians) fills the same angle on the eye over its `rows`.
#[must_use]
pub fn arcmin_per_px(fov_y: f64, rows: usize) -> f64 {
    fov_y.to_degrees() * 60.0 / rows.max(1) as f64
}

/// THE ADAPTING FIELD: the side, in degrees, of the square the eye's adaptation is read over (the
/// research: about half a degree around the patch).
pub const ADAPTING_FIELD_DEG: f64 = 0.5;

/// ★ THE TEMPORAL FACTOR of a HELD step: a pop holds its new state, as Blackwell's disc held (a change
/// seen for one frame only needs 1.7 times the contrast, Graham and Kemp 1938 — not this rule's case).
pub const HELD_STEP: f64 = 1.0;

/// THE VISIBILITY LEVEL at which the eye sees a change half of the time.
pub const SEEN: f64 = 1.0;

/// THE WINDOWS' SIDES in pixels, a power of two apart (a window steps by half its side).
pub const WINDOW_SIDES: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// ★ THE WIDE WINDOW, arcminutes: a window this wide or wider is judged on its whole change, where the
/// eight-bit rounding averages out; a narrower change counts only as a patch, beyond the rounding. At
/// the capture's seat (3.75′ a pixel) the rounding's spread over a 16-pixel window is about 0.03 of a
/// code, where the eye's threshold stands near 0.17 of a code.
pub const WIDE_WINDOW_ARCMIN: f64 = 60.0;

/// The smallest side of [`WINDOW_SIDES`] at least [`WIDE_WINDOW_ARCMIN`] wide at a seat of
/// `arcmin_per_px`; the widest side when none is.
#[must_use]
pub fn wide_side_px(arcmin_per_px: f64) -> usize {
    WINDOW_SIDES
        .iter()
        .copied()
        .find(|side| *side as f64 * arcmin_per_px >= WIDE_WINDOW_ARCMIN)
        .unwrap_or(WINDOW_SIDES[WINDOW_SIDES.len() - 1])
}

/// ★ THE POPS THE EYE SEES in a step: every HELD patch, and every held window at least
/// [`WIDE_WINDOW_ARCMIN`] wide, on the ground (still ground or a rung boundary) that the eye sees half
/// of the time. An occlusion edge's change is read apart (a silhouette's stair moves with the jitter),
/// and a flicker is the flicker meter's.
#[must_use]
pub fn pops(found: &[Sighting], wins: &[(Sighting, u64)], arcmin_per_px: f64) -> Vec<Sighting> {
    let wide = wide_side_px(arcmin_per_px) as u64;
    let ground = |s: &Sighting| (s.part == Part::Held) & (s.group != Group::Edge) & s.seen();
    found
        .iter()
        .filter(|s| ground(s))
        .chain(
            wins.iter()
                .map(|(s, _)| s)
                .filter(|s| ground(s) & (s.pixels >= wide * wide)),
        )
        .cloned()
        .collect()
}

/// A SUMMED-AREA TABLE of a `w × h` grid: any rectangle's sum in four reads.
pub struct Summed {
    w: usize,
    h: usize,
    s: Vec<f64>,
}

impl Summed {
    /// The table of a grid `v` of `w × h` values.
    #[must_use]
    pub fn of(v: &[f64], w: usize, h: usize) -> Summed {
        let mut s = vec![0.0; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row = 0.0;
            for x in 0..w {
                row += v[y * w + x];
                s[(y + 1) * (w + 1) + x + 1] = s[y * (w + 1) + x + 1] + row;
            }
        }
        Summed { w, h, s }
    }

    /// The sum over the columns `x0..x1` and the rows `y0..y1`, clamped to the grid.
    #[must_use]
    pub fn sum(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let (x1, y1) = (x1.min(self.w), y1.min(self.h));
        let (x0, y0) = (x0.min(x1), y0.min(y1));
        let at = |x: usize, y: usize| self.s[y * (self.w + 1) + x];
        at(x1, y1) - at(x0, y1) - at(x1, y0) + at(x0, y0)
    }

    /// The mean over the square of side `side` centred on `(x, y)`, clamped to the grid.
    #[must_use]
    pub fn mean_around(&self, x: usize, y: usize, side: usize) -> f64 {
        let half = side / 2;
        let (x0, y0) = (x.saturating_sub(half), y.saturating_sub(half));
        let (x1, y1) = ((x0 + side).min(self.w), (y0 + side).min(self.h));
        self.sum(x0, y0, x1, y1) / ((x1 - x0) * (y1 - y0)).max(1) as f64
    }
}

/// THE GROUP a pixel's change is met in: the picture's own change (still), an occlusion edge, or a
/// rung boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    Still,
    Edge,
    Boundary,
}

impl Group {
    /// The group of a judged pixel's class.
    #[must_use]
    pub fn of(class: PixelClass) -> Group {
        match class {
            PixelClass::Still => Group::Still,
            PixelClass::Edge => Group::Edge,
            PixelClass::Boundary(_) => Group::Boundary,
        }
    }

    const ALL: [Group; 3] = [Group::Still, Group::Edge, Group::Boundary];
}

/// ★ THE TEMPORAL FACTOR of a change seen for ONE frame (a flicker; Graham and Kemp 1938): the eye
/// needs 1.7 times the contrast of a held step.
pub const ONE_FRAME: f64 = 1.7;

/// ★ WHICH PART OF A CHANGE the eye meets: the part that HELD into the next frame (a pop), or the part
/// that did not (a flicker, seen for one frame). A pair read without its next frame holds all of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Part {
    Held,
    Flicker,
}

impl Part {
    /// The part's temporal factor: [`HELD_STEP`] or [`ONE_FRAME`].
    #[must_use]
    pub fn factor(self) -> f64 {
        match self {
            Part::Held => HELD_STEP,
            Part::Flicker => ONE_FRAME,
        }
    }

    const ALL: [Part; 2] = [Part::Held, Part::Flicker];
}

/// HOW A CHANGE WAS MET: as a patch of changed pixels, or as a window's whole change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Meeting {
    Patch,
    Window,
}

/// ★ ONE THING THE EYE MEETS: a patch or a window of one group's change, held or flickering.
#[derive(Clone, Debug, PartialEq)]
pub struct Sighting {
    pub meeting: Meeting,
    pub part: Part,
    pub group: Group,
    /// For a boundary: the finer rung most of its pixels crossed; else nought.
    pub rung: u8,
    /// Brighter (+1) or darker (−1).
    pub sign: i8,
    /// Its changed pixels (a patch), or its side squared (a window).
    pub pixels: u64,
    pub area_arcmin2: f64,
    /// Its contrast: the summed change over the summed field.
    pub contrast: f64,
    /// Its field: the mean adapting luminance, cd/m².
    pub field_cd_m2: f64,
    /// The eye's threshold for its area on its field.
    pub threshold: f64,
    /// Its visibility level: the contrast over the part's temporal factor times the threshold.
    pub visibility: f64,
    /// Where it stands: a pixel of it (a patch's first in reading order, a window's corner).
    pub at: (usize, usize),
}

impl Sighting {
    /// Whether the eye sees it half of the time.
    #[must_use]
    pub fn seen(&self) -> bool {
        self.visibility >= SEEN
    }
}

/// ★ A STEP AS THE EYE MEETS IT: the step's two pictures (RGBA), the frame before the step and the frame
/// after it when the flight recorded them (`None`: a change is read as held on that side), their size,
/// the seat's pixel, the display.
pub struct EyePair<'a> {
    pub first: &'a [u8],
    pub second: &'a [u8],
    pub before: Option<&'a [u8]>,
    pub after: Option<&'a [u8]>,
    pub width: usize,
    pub height: usize,
    pub arcmin_per_px: f64,
    pub display: Display,
    /// Whether the pictures carry the engine's screen dither ([`screen_dither`]): the display picture
    /// does, the render picture does not.
    pub dithered: bool,
}

/// ★ THE JUDGE'S READS OF A STEP: the second picture against the first (`step`, the pixels judged), and
/// the same second picture against the frame before the step and against the frame after it (empty
/// without that frame) — so every frame is read at the same ground as the second picture's pixel.
pub struct StepReads<'a> {
    pub step: &'a [PixelRead],
    pub before: &'a [PixelRead],
    pub after: &'a [PixelRead],
}

/// What every judged pixel changed by (its held part and its flicker), and the field the eye adapts to.
struct Changes {
    /// The luminance change of each judged pixel of the second picture, cd/m² (nought elsewhere): the
    /// jump that held and the rest, each beyond the rounding band (what a patch is judged on).
    held: Vec<f64>,
    flicker: Vec<f64>,
    /// The same without the rounding band: a window's mean, where the rounding averages out.
    held_wide: Vec<f64>,
    flicker_wide: Vec<f64>,
    group: Vec<Option<(Group, u8)>>,
    /// The second picture's luminance, summed.
    field: Summed,
    /// The adapting field's side in pixels.
    adapt_px: usize,
}

impl Changes {
    fn part(&self, part: Part) -> &[f64] {
        match part {
            Part::Held => &self.held,
            Part::Flicker => &self.flicker,
        }
    }

    fn part_wide(&self, part: Part) -> &[f64] {
        match part {
            Part::Held => &self.held_wide,
            Part::Flicker => &self.flicker_wide,
        }
    }
}

/// ★ A FRAME'S LIGHT AT A PIXEL'S GROUND, nearest `target`: between the frame's samples where the ground
/// reprojected (a smooth slope that slid part of a pixel), or at the judge's own best sample (an edge of
/// shade that slid) — whichever stands nearer the target.
fn light_at(frame: &[f64], w: usize, h: usize, r: &PixelRead, target: f64) -> f64 {
    let own = frame[r.matched];
    let between = bilinear(frame, w, h, r.place).unwrap_or(own);
    if (target - between).abs() < (target - own).abs() {
        between
    } else {
        own
    }
}

/// ★ THE PART THAT HELD: of the lights a pixel's ground showed after the step (`after`) and before it
/// (`before`), every after less every before — the smallest when all agree in sign, else nought.
///
/// Example: a rung-2 band steps from 38 to 45 cd/m² between the step's frames and stays: before 38, 38,
/// after 45, 45 — every difference 7, held 7. A pixel that dipped to 37 for the step's first frame only:
/// before 38, 37, after 38, 38 — the differences 0 and 1 do not all agree, held nought.
#[must_use]
pub fn held_of(after: &[f64], before: &[f64]) -> f64 {
    agreeing(after.iter().flat_map(|a| before.iter().map(move |b| a - b)))
}

/// Of a set of differences, the one nearest nought when all agree in sign, else nought.
fn agreeing(diffs: impl Iterator<Item = f64>) -> f64 {
    let (lo, hi) = diffs.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), d| {
        (lo.min(d), hi.max(d))
    });
    if lo > 0.0 {
        lo
    } else if hi < 0.0 {
        hi
    } else {
        0.0
    }
}

/// A difference less the rounding band `band` toward nought: the part an eight-bit picture proves.
fn beyond(d: f64, band: f64) -> f64 {
    d.signum() * (d.abs() - band).max(0.0)
}

/// The spread of one side's lights.
fn spread(side: &[f64]) -> f64 {
    let (lo, hi) = side
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    hi - lo
}

/// ★ THE JUMP AT THE STEP: the part that held ([`held_of`]) less the ground's own change within either
/// side of the step (the frame before against the step's first, the frame after against its second) —
/// a pop is a jump between two still states, and a steady drift is not one — every difference first
/// taken toward nought by the rounding band `band` (nought: none). MEASURED without the sides' change:
/// on the walk 918 held patches of still ground read seen with all four frames read — every later
/// frame differed from every earlier one alike, as a steady morph does, and as a stated distance half a
/// cell off does (its misreading grows with each frame of travel).
///
/// Example: the ground's light drifts 36, 37, 38, 39 cd/m² over the four frames: held 1, within 1, the
/// jump nought. A rung-2 band 38, 38, 45, 45: held 7, within nought, the jump 7 — 6.3 beyond a
/// rounding band of 0.7.
#[must_use]
pub fn jump_of(after: &[f64], before: &[f64], band: f64) -> f64 {
    let held = agreeing(
        after
            .iter()
            .flat_map(|a| before.iter().map(move |b| beyond(a - b, band))),
    );
    let within = beyond(spread(after), band).max(beyond(spread(before), band));
    beyond(held, within)
}

/// Each judged second-picture pixel's read in another judge's reads, by its place in the second picture.
fn by_pixel(reads: &[PixelRead], w: usize, h: usize) -> Vec<Option<&PixelRead>> {
    let mut at = vec![None; w * h];
    for r in reads {
        at[r.y * w + r.x] = Some(r);
    }
    at
}

/// A frame around the step as light, with the judge's reads of it laid by the second picture's pixel;
/// `None` without the frame.
type Around<'a> = Option<(Vec<f64>, Vec<Option<&'a PixelRead>>)>;

fn around<'a>(pair: &EyePair<'_>, frame: Option<&[u8]>, reads: &'a [PixelRead]) -> Around<'a> {
    frame.map(|f| {
        (
            pair.display.light_image(f, pair.width, pair.dithered).0,
            by_pixel(reads, pair.width, pair.height),
        )
    })
}

fn changes(pair: &EyePair<'_>, reads: &StepReads<'_>) -> Changes {
    let (w, h) = (pair.width, pair.height);
    let (first, _) = pair.display.light_image(pair.first, w, pair.dithered);
    let (second, band) = pair.display.light_image(pair.second, w, pair.dithered);
    let (before, after) = (
        around(pair, pair.before, reads.before),
        around(pair, pair.after, reads.after),
    );
    let mut held = vec![0.0; w * h];
    let mut flicker = vec![0.0; w * h];
    let mut held_wide = vec![0.0; w * h];
    let mut flicker_wide = vec![0.0; w * h];
    let mut group = vec![None; w * h];
    for r in reads.step {
        let i = r.y * w + r.x;
        let now = second[i];
        let was = light_at(&first, w, h, r, now);
        // The frame before the step, nearest what the second picture shows (the kindest reading); the
        // frame after it, nearest what the first showed (the change must hold against that too).
        let mut befores = vec![was];
        befores.extend(
            before
                .as_ref()
                .and_then(|(f, at)| at[i].map(|b| light_at(f, w, h, b, now))),
        );
        let mut afters = vec![now];
        afters.extend(
            after
                .as_ref()
                .and_then(|(f, at)| at[i].map(|a| light_at(f, w, h, a, was))),
        );
        // A patch reads what the picture proves beyond its rounding; a wide window reads the whole
        // change, its rounding averaging out over the window.
        let kept = jump_of(&afters, &befores, band[i]);
        held[i] = kept;
        flicker[i] = beyond(now - was, band[i]) - kept;
        let kept_wide = jump_of(&afters, &befores, 0.0);
        held_wide[i] = kept_wide;
        flicker_wide[i] = (now - was) - kept_wide;
        let rung = match r.class {
            PixelClass::Boundary(rung) => rung,
            _ => 0,
        };
        group[i] = Some((Group::of(r.class), rung));
    }
    let adapt_px = ((ADAPTING_FIELD_DEG * 60.0 / pair.arcmin_per_px).round() as usize).max(1);
    Changes {
        held,
        flicker,
        held_wide,
        flicker_wide,
        group,
        field: Summed::of(&second, w, h),
        adapt_px,
    }
}

/// ★ A GRID'S VALUE BETWEEN ITS SAMPLES at `place` (sample `(i, j)` at `(i, j)`): the four around it
/// weighed by nearness; `None` when one of the four falls off the grid.
#[must_use]
pub fn bilinear(grid: &[f64], w: usize, h: usize, place: [f64; 2]) -> Option<f64> {
    let (fx, fy) = (place[0].floor(), place[1].floor());
    let inside = (fx >= 0.0) & (fy >= 0.0) & (fx + 1.0 < w as f64) & (fy + 1.0 < h as f64);
    inside.then(|| {
        let (x, y) = (fx as usize, fy as usize);
        let (tx, ty) = (place[0] - fx, place[1] - fy);
        let at = |dx: usize, dy: usize| grid[(y + dy) * w + x + dx];
        let top = at(0, 0) * (1.0 - tx) + at(1, 0) * tx;
        let bottom = at(0, 1) * (1.0 - tx) + at(1, 1) * tx;
        top * (1.0 - ty) + bottom * ty
    })
}

/// The sighting of a summed change over a field, on an area.
#[allow(clippy::too_many_arguments)]
fn sighting(
    part: Part,
    group: Group,
    rung: u8,
    pixels: u64,
    area_px: f64,
    change: f64,
    field_sum: f64,
    field_mean: f64,
    arcmin_per_px: f64,
    at: (usize, usize),
) -> Sighting {
    let area_arcmin2 = area_px * arcmin_per_px * arcmin_per_px;
    let contrast = change.abs() / field_sum.max(f64::MIN_POSITIVE);
    let threshold = threshold_contrast(area_arcmin2 * SR_PER_ARCMIN2, field_mean);
    Sighting {
        meeting: Meeting::Patch,
        part,
        group,
        rung,
        sign: if change < 0.0 { -1 } else { 1 },
        pixels,
        area_arcmin2,
        contrast,
        field_cd_m2: field_mean,
        threshold,
        visibility: contrast / (part.factor() * threshold),
        at,
    }
}

/// ★ THE EYE'S READING OF ONE RECORDED STEP — what the flight gate and the reader both run.
pub struct StepReading {
    /// Every patch, with its pixels in the step's second picture.
    pub found: Vec<(Sighting, Vec<usize>)>,
    /// The most visible window per part, group and side, and how many of that side the eye sees.
    pub windows: Vec<(Sighting, u64)>,
    /// ★ The pops the eye sees ([`pops`]).
    pub pops: Vec<Sighting>,
    /// The judge's reads of the step's second picture against its first, against the frame before and
    /// against the frame after (empty without that frame).
    pub reads: [Vec<PixelRead>; 3],
    /// The seat's pixel, arcminutes.
    pub arcmin_per_px: f64,
}

/// ★ THE EYE'S STEP of a pair's recorded frames, consecutive and each as the pop judge reads it: with
/// four or more, the middle two of the first four and the frames around them; with three, the first two
/// and the frame after; with two, the pair alone; with fewer, none. `dithered`: the pictures carry the
/// engine's screen dither ([`screen_dither`]; the display picture does).
///
/// Example: the flight records `pop-hull-240-m-s-017-0000` to `-0003`; the eye reads the step from the
/// second frame to the third, the first and the fourth on either side.
#[must_use]
pub fn read_step(
    frames: &[Frame<'_>],
    cell_m: &dyn Fn(u8) -> f64,
    dithered: bool,
    display: Display,
) -> Option<StepReading> {
    let (first, second, before, after) = match frames {
        [f0, f1, f2, f3, ..] => (f1, f2, Some(f0), Some(f3)),
        [f0, f1, f2] => (f0, f1, None, Some(f2)),
        [f0, f1] => (f0, f1, None, None),
        _ => return None,
    };
    let against = |other: Option<&Frame<'_>>| {
        let mut marks = Vec::new();
        if let Some(other) = other {
            let _ = judge_pair_mapped(other, second, cell_m, &mut |p| marks.push(p));
        }
        marks
    };
    let reads = [against(Some(first)), against(before), against(after)];
    let pair = EyePair {
        first: first.rgba,
        second: second.rgba,
        before: before.map(|f| f.rgba),
        after: after.map(|f| f.rgba),
        width: second.width,
        height: second.height,
        arcmin_per_px: arcmin_per_px(second.camera.fov_y, second.height),
        display,
        dithered,
    };
    let (found, wins) = {
        let step = StepReads {
            step: &reads[0],
            before: &reads[1],
            after: &reads[2],
        };
        (patches_with_pixels(&pair, &step), windows(&pair, &step))
    };
    let sightings: Vec<Sighting> = found.iter().map(|(s, _)| s.clone()).collect();
    let seen = pops(&sightings, &wins, pair.arcmin_per_px);
    Some(StepReading {
        found,
        windows: wins,
        pops: seen,
        reads,
        arcmin_per_px: pair.arcmin_per_px,
    })
}

/// ★ THE PATCHES of a judged step: for each part (held, flicker), every set of judged pixels of one
/// group and one sign of change, joined through their eight neighbours; a pixel whose change is nought
/// joins none.
#[must_use]
pub fn patches(pair: &EyePair<'_>, reads: &StepReads<'_>) -> Vec<Sighting> {
    patches_with_pixels(pair, reads)
        .into_iter()
        .map(|(s, _)| s)
        .collect()
}

/// [`patches`], each with its pixels (`y · width + x` of the second picture) — where a map paints it.
#[must_use]
pub fn patches_with_pixels(
    pair: &EyePair<'_>,
    reads: &StepReads<'_>,
) -> Vec<(Sighting, Vec<usize>)> {
    let c = changes(pair, reads);
    Part::ALL
        .iter()
        .flat_map(|part| patches_of(pair, &c, *part))
        .collect()
}

fn patches_of(pair: &EyePair<'_>, c: &Changes, part: Part) -> Vec<(Sighting, Vec<usize>)> {
    let (w, h) = (pair.width, pair.height);
    let delta = c.part(part);
    let key = |i: usize| {
        c.group[i]
            .filter(|_| delta[i] != 0.0)
            .map(|(g, _)| (g, delta[i] > 0.0))
    };
    let mut taken = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        let Some(k) = key(start).filter(|_| !taken[start]) else {
            continue;
        };
        taken[start] = true;
        stack.push(start);
        let (mut pixels, mut change, mut field) = (0u64, 0.0, 0.0);
        let mut rungs = [0u64; 32];
        let mut members = Vec::new();
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            members.push(i);
            pixels += 1;
            change += delta[i];
            field += c.field.mean_around(x, y, c.adapt_px);
            rungs[usize::from(c.group[i].map_or(0, |(_, r)| r)) & 31] += 1;
            for ny in y.saturating_sub(1)..(y + 2).min(h) {
                for nx in x.saturating_sub(1)..(x + 2).min(w) {
                    let j = ny * w + nx;
                    if !taken[j] && key(j) == Some(k) {
                        taken[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        let rung = (0..32u8)
            .max_by_key(|r| rungs[usize::from(*r)])
            .unwrap_or(0);
        let rung = if k.0 == Group::Boundary { rung } else { 0 };
        let seen = sighting(
            part,
            k.0,
            rung,
            pixels,
            pixels as f64,
            change,
            field,
            field / pixels as f64,
            pair.arcmin_per_px,
            (start % w, start / w),
        );
        members.sort_unstable();
        out.push((seen, members));
    }
    out
}

/// ★ THE WINDOWS of a judged step: for each part and each group and each side of [`WINDOW_SIDES`], the window (a
/// square stepping by half its side) whose summed change the eye sees best — the change of that
/// group's pixels only, every other pixel's taken as nought — and how many windows of that side the
/// eye sees half of the time.
#[must_use]
pub fn windows(pair: &EyePair<'_>, reads: &StepReads<'_>) -> Vec<(Sighting, u64)> {
    let c = changes(pair, reads);
    let mut out = Vec::new();
    for part in Part::ALL {
        for g in Group::ALL {
            out.extend(windows_of(pair, &c, part, g));
        }
    }
    out
}

fn windows_of(pair: &EyePair<'_>, c: &Changes, part: Part, g: Group) -> Vec<(Sighting, u64)> {
    let (w, h) = (pair.width, pair.height);
    let delta = c.part_wide(part);
    let mut out = Vec::new();
    {
        let mine: Vec<f64> = (0..w * h)
            .map(|i| {
                if c.group[i].is_some_and(|(gi, _)| gi == g) {
                    delta[i]
                } else {
                    0.0
                }
            })
            .collect();
        let judged: Vec<f64> = (0..w * h)
            .map(|i| f64::from(u8::from(c.group[i].is_some_and(|(gi, _)| gi == g))))
            .collect();
        let (sum, count) = (Summed::of(&mine, w, h), Summed::of(&judged, w, h));
        for side in WINDOW_SIDES {
            let stride = (side / 2).max(1);
            let mut best: Option<Sighting> = None;
            let mut seen = 0u64;
            for y0 in (0..h.saturating_sub(side - 1)).step_by(stride) {
                for x0 in (0..w.saturating_sub(side - 1)).step_by(stride) {
                    if count.sum(x0, y0, x0 + side, y0 + side) == 0.0 {
                        continue;
                    }
                    let change = sum.sum(x0, y0, x0 + side, y0 + side);
                    if change == 0.0 {
                        continue;
                    }
                    let around = side.max(c.adapt_px);
                    let field_mean = c.field.mean_around(x0 + side / 2, y0 + side / 2, around);
                    let s = Sighting {
                        meeting: Meeting::Window,
                        ..sighting(
                            part,
                            g,
                            0,
                            (side * side) as u64,
                            (side * side) as f64,
                            change,
                            field_mean * (side * side) as f64,
                            field_mean,
                            pair.arcmin_per_px,
                            (x0, y0),
                        )
                    };
                    seen += u64::from(s.seen());
                    if best.as_ref().is_none_or(|b| s.visibility > b.visibility) {
                        best = Some(s);
                    }
                }
            }
            if let Some(b) = best {
                out.push((b, seen));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A relative difference.
    fn near(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(f64::MIN_POSITIVE)
    }

    /// The sRGB codes' light: the straight segment, the curve, white; the display's luminance of them.
    #[test]
    fn a_code_is_light_on_the_reference_display() {
        let t = srgb_linear_table();
        assert_eq!(t[0], 0.0);
        assert!(near(t[10], 0.003_035_269_835, 1e-9));
        assert!(near(t[128], 0.215_860_500_114, 1e-9));
        assert!(near(t[255], 1.0, 1e-12));
        let d = REFERENCE_DISPLAY;
        assert!(near(d.luminance(&t, [255, 255, 255]), 200.0, 1e-12));
        assert!(near(d.luminance(&t, [0, 0, 0]), 0.2, 1e-12));
        assert!(near(d.luminance(&t, [128, 128, 128]), 43.328_927_923, 1e-9));
        let img = d.luminance_image(&[0, 0, 0, 255, 255, 255, 255, 0]);
        assert_eq!(img.len(), 2);
        assert!(near(img[1], 200.0, 1e-12));
    }

    /// ★ THE THRESHOLD LAW against the values its equations give (computed apart), across day,
    /// twilight and night: Riccò's constant, the Weber contrast, the joining exponent, and the
    /// threshold of a capture pixel and of a large patch.
    #[test]
    fn the_threshold_law_gives_blackwells_contrasts() {
        let cases = [
            (0.05, 2.817_946e-7, 7.833_207e-3, 0.6),
            (1.0, 6.118_526e-8, 4.274_574e-3, 0.8861),
            (40.0, 2.008_299e-8, 2.985_628e-3, 1.004_218),
            (343.0, 1.282_216e-8, 2.655_079e-3, 0.921_626),
        ];
        for (field, r, c, q) in cases {
            assert!(near(ricco_sr(field), r, 1e-6), "R at {field}");
            assert!(near(weber_contrast(field), c, 1e-6), "C∞ at {field}");
            assert!(near(joining_exponent(field), q, 1e-6), "q at {field}");
        }
        let sr = |arcmin2: f64| arcmin2 * SR_PER_ARCMIN2;
        assert!(near(
            threshold_contrast(sr(14.0625), 40.0),
            1.982_813e-2,
            1e-6
        ));
        assert!(near(
            threshold_contrast(sr(1406.25), 40.0),
            3.151_653e-3,
            1e-6
        ));
        assert!(near(
            threshold_contrast(sr(14.0625), 1.0),
            5.786_800e-2,
            1e-6
        ));
        // Blackwell's 3.6′ disc at 100 foot-lamberts: about 1.8 %.
        assert!(near(
            threshold_contrast(sr(10.18), 343.0),
            1.821_004e-2,
            1e-5
        ));
        // The joining exponent's three laws meet within a thousandth at their borders.
        assert!((joining_exponent(3.40) - joining_exponent(3.399_999)).abs() < 1e-3);
        assert!((joining_exponent(0.193) - joining_exponent(0.192_999)).abs() < 1e-3);
    }

    /// The law's two limbs: a speck's threshold is Riccò's `R / A`, a field's is the Weber contrast.
    #[test]
    fn a_speck_obeys_ricco_and_a_field_obeys_weber() {
        let field = 40.0;
        let speck = 0.01 * SR_PER_ARCMIN2;
        assert!(near(
            threshold_contrast(speck, field),
            ricco_sr(field) / speck,
            1e-3
        ));
        let sky = 1.0e7 * SR_PER_ARCMIN2;
        assert!(near(
            threshold_contrast(sky, field),
            weber_contrast(field),
            1e-3
        ));
        // A larger patch and a brighter field each lower the threshold.
        assert!(threshold_contrast(2.0 * speck, field) < threshold_contrast(speck, field));
        assert!(threshold_contrast(speck, 2.0 * field) < threshold_contrast(speck, field));
    }

    /// The seat: 45° over 720 rows is 3.75′ a pixel; no rows reads as one.
    #[test]
    fn the_seat_gives_a_pixel_its_angle() {
        assert!(near(arcmin_per_px(45f64.to_radians(), 720), 3.75, 1e-12));
        assert!(near(arcmin_per_px(45f64.to_radians(), 0), 2700.0, 1e-12));
    }

    /// The summed table: a rectangle's sum, a rectangle past the grid clamped, a square's mean at a
    /// corner.
    #[test]
    fn the_summed_table_sums_any_rectangle() {
        // 1 2 3
        // 4 5 6
        let t = Summed::of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        assert_eq!(t.sum(0, 0, 3, 2), 21.0);
        assert_eq!(t.sum(1, 0, 3, 2), 16.0);
        assert_eq!(t.sum(1, 1, 9, 9), 11.0);
        assert_eq!(t.sum(5, 5, 9, 9), 0.0);
        // The 2 × 2 square around (0, 0) holds 1, 2, 4, 5.
        assert_eq!(t.mean_around(0, 0, 2), 3.0);
        assert_eq!(t.mean_around(2, 1, 1), 6.0);
        assert_eq!(t.mean_around(1, 1, 0), 0.0);
    }

    /// The groups of the judge's classes; the parts' temporal factors; a sighting seen at one.
    #[test]
    fn groups_parts_and_the_seen_line() {
        assert_eq!(Group::of(PixelClass::Still), Group::Still);
        assert_eq!(Group::of(PixelClass::Edge), Group::Edge);
        assert_eq!(Group::of(PixelClass::Boundary(3)), Group::Boundary);
        assert_eq!(Part::Held.factor(), HELD_STEP);
        assert_eq!(Part::Flicker.factor(), ONE_FRAME);
        let at = |visibility: f64| Sighting {
            meeting: Meeting::Patch,
            part: Part::Held,
            group: Group::Still,
            rung: 0,
            sign: 1,
            pixels: 1,
            area_arcmin2: 1.0,
            contrast: 0.0,
            field_cd_m2: 1.0,
            threshold: 1.0,
            visibility,
            at: (0, 0),
        };
        assert!(at(SEEN).seen());
        assert!(!at(0.99).seen());
    }

    /// ★ THE HELD PART: every after-light less every before-light; all of one sign keep the smallest,
    /// a mixed set keeps nought (the doc's two examples, a darkening, a one-sided step).
    #[test]
    fn a_change_holds_only_when_both_sides_agree() {
        assert_eq!(held_of(&[45.0, 45.0], &[38.0, 38.0]), 7.0);
        assert_eq!(held_of(&[38.0, 38.0], &[38.0, 37.0]), 0.0);
        assert_eq!(held_of(&[45.0, 40.0], &[38.0, 39.0]), 1.0);
        assert_eq!(held_of(&[36.0, 37.0], &[38.0, 38.0]), -1.0);
        assert_eq!(held_of(&[30.0], &[38.0]), -8.0);
        assert_eq!(held_of(&[40.0, 36.0], &[38.0]), 0.0);
    }

    /// ★ THE JUMP: a step between two still sides keeps its size; a steady drift keeps nought; a step on a
    /// drift keeps the step; a lone frame on each side (no spread) keeps what held; nothing held, nought.
    #[test]
    fn a_pop_is_a_jump_between_two_still_sides() {
        assert_eq!(jump_of(&[45.0, 45.0], &[38.0, 38.0], 0.0), 7.0);
        assert_eq!(jump_of(&[38.0, 39.0], &[36.0, 37.0], 0.0), 0.0);
        assert_eq!(jump_of(&[46.0, 47.0], &[36.0, 37.0], 0.0), 8.0);
        assert_eq!(jump_of(&[30.0], &[38.0], 0.0), -8.0);
        assert_eq!(jump_of(&[38.0, 38.0], &[38.0, 37.0], 0.0), 0.0);
        assert_eq!(jump_of(&[31.0, 30.0], &[38.0, 38.0], 0.0), -6.0);
        // ★ Beyond a rounding band of 0.5: the step keeps 6.5; a one-band wobble on each side is no
        // spread; a change of a band or less is nothing.
        assert_eq!(jump_of(&[45.0, 45.0], &[38.0, 38.0], 0.5), 6.5);
        assert_eq!(jump_of(&[45.0, 45.5], &[38.0, 38.5], 0.5), 6.0);
        assert_eq!(jump_of(&[38.5, 38.5], &[38.0, 38.0], 0.5), 0.0);
    }

    /// Codes between the whole codes: the same curve as the table, clamped to the codes' range.
    #[test]
    fn a_code_between_the_codes_is_light_too() {
        let t = srgb_linear_table();
        assert_eq!(srgb_linear(128.0), t[128]);
        assert!(srgb_linear(127.5) > t[127]);
        assert!(srgb_linear(127.5) < t[128]);
        assert_eq!(srgb_linear(-3.0), 0.0);
        assert_eq!(srgb_linear(300.0), 1.0);
        let d = REFERENCE_DISPLAY;
        assert!(near(d.luminance_of_codes([120.0; 3]), 37.726_590_306, 1e-9));
        assert!(near(
            d.luminance_of_codes([119.7, 120.2, 120.4]),
            37.799_467_716,
            1e-9
        ));
    }

    /// ★ THE ENGINE'S DITHER at known pixels (its own formula, computed apart); a flat colour dithered
    /// and rounded as the engine rounds it reads flatter under the dither than over it; the rounding
    /// band is one code's light.
    #[test]
    fn the_dither_is_read_off_and_the_band_is_one_code() {
        let near3 = |a: [f64; 3], b: [f64; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-6);
        assert!(near3(
            screen_dither(0, 0),
            [0.451_456_311, 0.330_985_915, -0.427_835_052]
        ));
        assert!(near3(
            screen_dither(1, 0),
            [0.111_650_485, -0.260_563_380, 0.335_051_546]
        ));
        assert!(near3(
            screen_dither(5, 3),
            [0.480_582_524, 0.133_802_817, -0.469_072_165]
        ));
        let (w, h) = (16usize, 8usize);
        let flat = 120.3;
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let d = screen_dither(i % w, i / w);
                let c = [0, 1, 2].map(|k| (flat + d[k]).round() as u8);
                [c[0], c[1], c[2], 255]
            })
            .collect();
        let d = REFERENCE_DISPLAY;
        let spread = |v: &[f64]| {
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - v.iter().copied().fold(f64::INFINITY, f64::min)
        };
        let (over, _) = d.light_image(&rgba, w, false);
        let (under, band) = d.light_image(&rgba, w, true);
        assert!(spread(&under) < spread(&over));
        // Under the dither every pixel stands within half a code of the flat colour's light.
        let truth = d.luminance_of_codes([flat; 3]);
        assert!(
            under
                .iter()
                .zip(&band)
                .all(|(l, b)| (l - truth).abs() <= 0.5 * b + 1e-9)
        );
        // The band is one code's light at the pixel's own codes, half a code each way.
        let d0 = screen_dither(0, 0);
        let c0 = [0, 1, 2].map(|k| f64::from(rgba[k]) - d0[k]);
        let one_code =
            d.luminance_of_codes(c0.map(|v| v + 0.5)) - d.luminance_of_codes(c0.map(|v| v - 0.5));
        assert!(near(band[0], one_code, 1e-12));
    }

    /// ★ THE WIDE WINDOW's side at the capture's seat, at the window's, at a huge pixel and a tiny one.
    #[test]
    fn the_wide_window_spans_a_degree() {
        assert_eq!(wide_side_px(3.75), 16);
        assert_eq!(wide_side_px(1.875), 32);
        assert_eq!(wide_side_px(100.0), 1);
        assert_eq!(wide_side_px(0.1), 128);
    }

    /// ★ THE POPS: a seen held patch on a boundary and a seen held window a degree wide count; an edge,
    /// an unseen patch, a flicker and a narrow window do not.
    #[test]
    fn the_pops_are_the_seen_held_changes_of_the_ground() {
        let at = |part: Part, group: Group, pixels: u64, visibility: f64| Sighting {
            meeting: Meeting::Patch,
            part,
            group,
            rung: 2,
            sign: 1,
            pixels,
            area_arcmin2: 1.0,
            contrast: 0.1,
            field_cd_m2: 40.0,
            threshold: 0.01,
            visibility,
            at: (0, 0),
        };
        let found = vec![
            at(Part::Held, Group::Boundary, 4, 3.0),
            at(Part::Held, Group::Edge, 4, 9.0),
            at(Part::Held, Group::Still, 4, 0.5),
            at(Part::Flicker, Group::Still, 4, 9.0),
        ];
        let window = |pixels: u64, visibility: f64| Sighting {
            meeting: Meeting::Window,
            ..at(Part::Held, Group::Still, pixels, visibility)
        };
        let wins = vec![(window(16 * 16, 1.5), 3), (window(8 * 8, 4.0), 9)];
        let got = pops(&found, &wins, 3.75);
        assert_eq!(got.len(), 2);
        assert_eq!((got[0].group, got[0].pixels), (Group::Boundary, 4));
        assert_eq!(
            (got[1].meeting, got[1].group, got[1].pixels),
            (Meeting::Window, Group::Still, 256)
        );
    }

    /// Between the samples: the four around weighed by nearness; a sample itself; off the grid, none.
    #[test]
    fn a_grid_is_read_between_its_samples() {
        // 0 2
        // 4 6
        let g = [0.0, 2.0, 4.0, 6.0];
        assert_eq!(bilinear(&g, 2, 2, [0.5, 0.5]), Some(3.0));
        assert_eq!(bilinear(&g, 2, 2, [0.0, 0.0]), Some(0.0));
        assert_eq!(bilinear(&g, 2, 2, [0.25, 0.0]), Some(0.5));
        assert_eq!(bilinear(&g, 2, 2, [1.0, 0.5]), None);
        assert_eq!(bilinear(&g, 2, 2, [-0.5, 0.0]), None);
    }

    fn read(x: usize, y: usize, w: usize, class: PixelClass, place: [f64; 2]) -> PixelRead {
        PixelRead {
            x,
            y,
            step: 0,
            class,
            matched: y * w + x,
            place,
        }
    }

    /// ★ A FRAME'S LIGHT AT THE GROUND: between the samples when that stands nearer the target, else the
    /// judge's own best sample; off the grid between samples, the judge's sample alone.
    #[test]
    fn a_frame_is_read_nearest_the_target() {
        // 0 10
        // 0 10
        let frame = [0.0, 10.0, 0.0, 10.0];
        let slid = read(0, 0, 2, PixelClass::Still, [0.5, 0.0]);
        assert_eq!(light_at(&frame, 2, 2, &slid, 6.0), 5.0);
        assert_eq!(light_at(&frame, 2, 2, &slid, 1.0), 0.0);
        let edge = read(1, 1, 2, PixelClass::Still, [1.5, 1.0]);
        assert_eq!(light_at(&frame, 2, 2, &edge, 3.0), 10.0);
    }

    /// Another judge's reads laid by the second picture's pixel.
    #[test]
    fn reads_are_found_by_their_pixel() {
        let reads = [
            read(2, 1, 3, PixelClass::Still, [2.0, 1.0]),
            read(0, 0, 3, PixelClass::Edge, [0.0, 0.0]),
        ];
        let at = by_pixel(&reads, 3, 2);
        assert_eq!(at.iter().filter(|r| r.is_some()).count(), 2);
        assert_eq!(at[5].map(|r| r.class), Some(PixelClass::Still));
        assert_eq!(at[0].map(|r| r.class), Some(PixelClass::Edge));
        assert!(at[1].is_none());
    }

    /// A 6 × 4 step of grey (120) frames, the frame before it and the frame after it: a 2 × 2 block of a
    /// rung boundary steps ten codes brighter and holds; a still pixel at (0, 3) steps ten codes darker
    /// and holds; a still pixel at (5, 0) flashes twenty codes brighter and falls back; a still pixel at
    /// (4, 3) dipped ten codes in the step's first frame only (so it reads ten codes brighter in the
    /// step); a still pixel at (0, 0) drifts five codes a frame (110, 115, 120, 125). The frame before
    /// holds no read of (5, 0).
    struct Scene {
        frames: [Vec<u8>; 4],
        step: Vec<PixelRead>,
        before: Vec<PixelRead>,
        after: Vec<PixelRead>,
    }

    fn scene() -> Scene {
        let (w, h) = (6usize, 4usize);
        let grey = || -> Vec<u8> { (0..w * h).flat_map(|_| [120, 120, 120, 255]).collect() };
        let mut frames = [grey(), grey(), grey(), grey()];
        let set = |img: &mut Vec<u8>, x: usize, y: usize, v: u8| {
            img[(y * w + x) * 4..(y * w + x) * 4 + 3].copy_from_slice(&[v, v, v]);
        };
        for (x, y) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            set(&mut frames[2], x, y, 130);
            set(&mut frames[3], x, y, 130);
        }
        set(&mut frames[2], 0, 3, 110);
        set(&mut frames[3], 0, 3, 110);
        set(&mut frames[2], 5, 0, 140);
        set(&mut frames[1], 4, 3, 110);
        for (k, v) in [110u8, 115, 120, 125].into_iter().enumerate() {
            set(&mut frames[k], 0, 0, v);
        }
        let class = |x: usize, y: usize| {
            if (1..3).contains(&x) & (1..3).contains(&y) {
                PixelClass::Boundary(2)
            } else {
                PixelClass::Still
            }
        };
        let step: Vec<PixelRead> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                read(x, y, w, class(x, y), [x as f64, y as f64])
            })
            .collect();
        let before = step
            .iter()
            .copied()
            .filter(|r| (r.x, r.y) != (5, 0))
            .collect();
        let after = step.clone();
        Scene {
            frames,
            step,
            before,
            after,
        }
    }

    fn pair(scene: &Scene, outer: bool) -> EyePair<'_> {
        EyePair {
            first: &scene.frames[1],
            second: &scene.frames[2],
            before: outer.then_some(scene.frames[0].as_slice()),
            after: outer.then_some(scene.frames[3].as_slice()),
            width: 6,
            height: 4,
            arcmin_per_px: 3.75,
            display: REFERENCE_DISPLAY,
            dithered: false,
        }
    }

    fn reads(scene: &Scene) -> StepReads<'_> {
        StepReads {
            step: &scene.step,
            before: &scene.before,
            after: &scene.after,
        }
    }

    /// ★ THE PATCHES: with the frames around the step, the block and the dark pixel jumped, the drift, the
    /// dip and the flash did not; without them every change reads as held. Each patch's level is its
    /// contrast over its part's factor times the eye's threshold for its area on its field.
    #[test]
    fn the_patches_split_the_held_from_the_flicker() {
        let sc = scene();
        let with = patches(&pair(&sc, true), &reads(&sc));
        let brief = |s: &Sighting| (s.part, s.group, s.rung, s.sign, s.pixels, s.at);
        let mut got: Vec<_> = with.iter().map(brief).collect();
        got.sort_by_key(|b| (b.0, b.1, b.5));
        assert_eq!(
            got,
            vec![
                (Part::Held, Group::Still, 0, -1, 1, (0, 3)),
                (Part::Held, Group::Boundary, 2, 1, 4, (1, 1)),
                (Part::Flicker, Group::Still, 0, 1, 1, (0, 0)),
                (Part::Flicker, Group::Still, 0, 1, 1, (4, 3)),
                (Part::Flicker, Group::Still, 0, 1, 1, (5, 0)),
            ]
        );
        for s in &with {
            let area = s.pixels as f64 * 3.75 * 3.75;
            assert!(near(s.area_arcmin2, area, 1e-12));
            let th = threshold_contrast(area * SR_PER_ARCMIN2, s.field_cd_m2);
            assert!(near(s.threshold, th, 1e-12));
            assert!(near(
                s.visibility,
                s.contrast / (s.part.factor() * th),
                1e-12
            ));
        }
        // Ten codes over a 2 × 2 block (56′²) on a 40 cd/m² field: plainly seen.
        let block = with
            .iter()
            .find(|s| s.group == Group::Boundary)
            .expect("the block");
        assert!(block.seen());
        // The block's pixels, in reading order.
        let with_pixels = patches_with_pixels(&pair(&sc, true), &reads(&sc));
        let (_, block_px) = with_pixels
            .iter()
            .find(|(s, _)| s.group == Group::Boundary)
            .expect("the block's pixels");
        assert_eq!(block_px, &vec![7, 8, 13, 14]);
        let alone = StepReads {
            step: &sc.step,
            before: &[],
            after: &[],
        };
        let without = patches(&pair(&sc, false), &alone);
        assert!(without.iter().all(|s| s.part == Part::Held));
        assert_eq!(without.len(), 5);
    }

    /// ★ THE STEP READ FROM RECORDED FRAMES (a still eye, every pixel rung 2 at 2 km): with four frames the
    /// block and the dark pixel are the pops the eye sees, read against all three other frames; with
    /// three the step is the first two (where nothing jumped) and the frame after; with two the pair
    /// alone; with one, nothing.
    #[test]
    fn a_recorded_step_is_read_as_the_eye_meets_it() {
        use crate::probe::{PROBE_KIND_TERRAIN, ProbePixel, encode_probe};
        let sc = scene();
        let (w, h) = (6usize, 4usize);
        let p = encode_probe(ProbePixel {
            kind: PROBE_KIND_TERRAIN,
            rung: 2,
            cells: 500,
        });
        let probe: Vec<u8> = (0..w * h).flat_map(|_| [p[0], p[1], p[2], 255]).collect();
        let frames: Vec<Frame<'_>> = sc
            .frames
            .iter()
            .map(|rgba| Frame {
                rgba,
                probe: &probe,
                width: w,
                height: h,
                camera: crate::pop::frame_camera([0.0; 3], [0.0, 0.0, 0.0, 1.0], w, h),
                hud: None,
                jitter_px: [0.0; 2],
                edge_separation_per_m: 0.0,
                edge_reach_px: 1,
            })
            .collect();
        let cell = |rung: u8| f64::from(1u32 << rung);
        let four = read_step(&frames, &cell, false, REFERENCE_DISPLAY).expect("four frames");
        assert_eq!(four.reads.map(|r| r.len()), [24, 24, 24]);
        // The patches the eye sees; at this seat (675′ a pixel) every window is a degree wide, so the
        // windows over them count too.
        let mut at: Vec<(usize, usize)> = four
            .pops
            .iter()
            .filter(|s| s.meeting == Meeting::Patch)
            .map(|s| s.at)
            .collect();
        at.sort_unstable();
        assert_eq!(at, vec![(0, 3), (1, 1)]);
        assert!(four.pops.iter().any(|s| s.meeting == Meeting::Window));
        assert!(near(four.arcmin_per_px, 45.0 * 60.0 / 4.0, 1e-9));
        let three = read_step(&frames[..3], &cell, false, REFERENCE_DISPLAY).expect("three");
        assert_eq!(three.reads.map(|r| r.len()), [24, 0, 24]);
        assert!(three.pops.is_empty());
        let two = read_step(&frames[..2], &cell, false, REFERENCE_DISPLAY).expect("two");
        assert_eq!(two.reads.map(|r| r.len()), [24, 0, 0]);
        assert!(read_step(&frames[..1], &cell, false, REFERENCE_DISPLAY).is_none());
    }

    /// ★ THE WINDOWS: the held block is best met by the 2 × 2 window on it; a group with no pixel and a
    /// window larger than the picture meet nothing; the brighter flash is the most visible flicker.
    #[test]
    fn the_windows_meet_the_block_at_its_own_size() {
        let sc = scene();
        let wins = windows(&pair(&sc, true), &reads(&sc));
        let find = |part: Part, g: Group, side: usize| {
            wins.iter().find(|(s, _)| {
                (s.part == part) & (s.group == g) & (s.pixels == (side * side) as u64)
            })
        };
        let (best2, seen2) = find(Part::Held, Group::Boundary, 2).expect("a 2 × 2 window");
        assert_eq!((best2.meeting, best2.at), (Meeting::Window, (1, 1)));
        assert!(*seen2 >= 1);
        let (best1, _) = find(Part::Held, Group::Boundary, 1).expect("a pixel window");
        assert!(best2.visibility > best1.visibility);
        assert!(find(Part::Held, Group::Boundary, 8).is_none());
        assert!(wins.iter().all(|(s, _)| s.group != Group::Edge));
        let (flash, _) = find(Part::Flicker, Group::Still, 1).expect("the flash");
        assert_eq!(flash.at, (5, 0));
        assert_eq!(flash.sign, 1);
    }
}
