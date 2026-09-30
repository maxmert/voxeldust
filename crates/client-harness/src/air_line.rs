//! ★ THE LINE AT THE EYE'S LEVEL (G-AIR-UNDER; ruling B25, 2026-09-30) — the pure half.
//!
//! The owner flew low over the dry sea floor and the haze vanished. MEASURED: the air model's ground
//! sphere stood at the ladder's datum, and an eye under it was lifted onto it, so every ray under the
//! eye's level met the sphere at once and crossed no air. The picture showed a HARD LINE at the eye's
//! level: the ground over it hazed, the ground under it bare. A ray over the level and a ray under it
//! differ by a hair, so the air they cross differs by a hair too; a line at the level is the model's
//! defect and never the world's.
//!
//! **The reading.** The meter reads the ground's colour in bands of rows on both sides of the eye's
//! level row: each band's mean over the probe's GROUND pixels (a band is ground where at least
//! [`GROUND_SHARE_MIN`] of its pixels are). The step between two neighbouring bands is the largest of
//! the three channels' differences, in levels. The step AT THE LEVEL is the step between the band that
//! ends at the level row and the band that starts there; every other pair of neighbouring bands in the
//! window gives the steps ELSEWHERE.
//!
//! **The verdict.** The eye's level is not special: its step is no larger than the largest step
//! elsewhere in the window. The ground's own shading and the air's own slope make the steps elsewhere;
//! a line adds a step at the level alone.
//!
//! **The controls** are the tests below, and the numbers the gate was set on (the stand 2 135 m under
//! the old sphere, eight-row bands, 2026-09-30): the step at the level 28.8 levels against 6.9 at most
//! elsewhere BEFORE the cure; 2.3 against 3.7 AFTER it; the stand 2 002 m over the sphere, 0.7 against
//! 1.7.
//!
//! Example: the hull hangs 2 km over the dry sea floor, 20 km off the owner's coast, and looks back at
//! the coast 2° under its level. The meter reads twenty bands of eight rows round the level row. The
//! defect's picture steps 28.8 levels at the level and fails; the cured picture steps 2.3 and passes.

use crate::probe::{PROBE_KIND_TERRAIN, decode_probe};

/// The least share of a band's pixels that must be ground for the band to be read: a band mostly sky
/// (over the horizon) or mostly ruler reads no ground colour.
pub const GROUND_SHARE_MIN: f64 = 0.5;

/// One picture's reading at the eye's level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelStep {
    /// The step between the two bands that meet at the level row, in levels.
    pub at_level: f64,
    /// The largest step between any other two neighbouring bands of the window, in levels.
    pub largest_elsewhere: f64,
    /// How many other pairs of neighbouring bands were read.
    pub pairs_elsewhere: usize,
}

impl LevelStep {
    /// The verdict: the step at the level is no larger than the largest step elsewhere.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.at_level <= self.largest_elsewhere
    }
}

/// One band's mean ground colour, or `None` where the band leaves the picture or holds too little
/// ground.
fn band_mean(
    rgba: &[u8],
    probe: &[u8],
    w: usize,
    h: usize,
    first_row: isize,
    rows: usize,
) -> Option<[f64; 3]> {
    let first = usize::try_from(first_row).ok()?;
    let end = first + rows;
    if end > h {
        return None;
    }
    let mut sum = [0u64; 3];
    let mut ground = 0u64;
    for i in first * w..end * w {
        let p = &probe[4 * i..4 * i + 3];
        let is_ground = decode_probe([p[0], p[1], p[2]]).kind == PROBE_KIND_TERRAIN;
        ground += u64::from(is_ground);
        for (c, s) in sum.iter_mut().enumerate() {
            *s += u64::from(is_ground) * u64::from(rgba[4 * i + c]);
        }
    }
    let share = ground as f64 / (rows * w) as f64;
    (share >= GROUND_SHARE_MIN).then(|| sum.map(|s| s as f64 / ground as f64))
}

/// The step between two bands: the largest channel difference, in levels.
fn step(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f64::max)
}

/// ★ THE READING: bands of `band_rows` rows, `bands_each_side` on each side of the boundary at
/// `level_row` (the first row under the eye's level), over the RGBA8 picture and its RGBA8 probe of
/// `w × h`. `None` where either band at the level is not ground, or no other pair of neighbouring
/// bands is.
#[must_use]
pub fn level_step(
    rgba: &[u8],
    probe: &[u8],
    w: usize,
    h: usize,
    level_row: usize,
    band_rows: usize,
    bands_each_side: usize,
) -> Option<LevelStep> {
    let side = bands_each_side as isize;
    let bands: Vec<Option<[f64; 3]>> = (-side..side)
        .map(|k| {
            let first = level_row as isize + k * band_rows as isize;
            band_mean(rgba, probe, w, h, first, band_rows)
        })
        .collect();
    // The pair at the level: the band that ends at the level row and the band that starts there.
    let at = bands_each_side;
    let at_level = step(bands[at - 1]?, bands[at]?);
    let elsewhere: Vec<f64> = bands
        .windows(2)
        .enumerate()
        .filter(|(i, _)| i + 1 != at)
        .filter_map(|(_, pair)| Some(step(pair[0]?, pair[1]?)))
        .collect();
    let largest_elsewhere = elsewhere.iter().copied().reduce(f64::max)?;
    Some(LevelStep {
        at_level,
        largest_elsewhere,
        pairs_elsewhere: elsewhere.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{PROBE_KIND_NONE, probe_byte};

    const W: usize = 4;
    const H: usize = 40;

    /// A picture whose every row is one grey, `level(y)`, all ground (or sky where `sky(y)`). The two
    /// rules are trait objects, so the helper has ONE instance and its branch is counted once (HR5).
    fn picture(level: &dyn Fn(usize) -> u8, sky: &dyn Fn(usize) -> bool) -> (Vec<u8>, Vec<u8>) {
        let mut rgba = Vec::with_capacity(W * H * 4);
        let mut probe = Vec::with_capacity(W * H * 4);
        for y in 0..H {
            for _ in 0..W {
                let v = level(y);
                rgba.extend_from_slice(&[v, v, v, 255]);
                let kind = if sky(y) {
                    PROBE_KIND_NONE
                } else {
                    PROBE_KIND_TERRAIN
                };
                probe.extend_from_slice(&[probe_byte(kind, 3), 0, 9, 255]);
            }
        }
        (rgba, probe)
    }

    /// A smooth slope of one level a row passes: the step at the level is the slope's own.
    #[test]
    fn a_smooth_slope_passes() {
        let (rgba, probe) = picture(&|y| 60 + y as u8, &|_| false);
        let s = level_step(&rgba, &probe, W, H, 20, 4, 4).expect("ground on both sides");
        assert_eq!(s.at_level, 4.0);
        assert_eq!(s.largest_elsewhere, 4.0);
        assert_eq!(s.pairs_elsewhere, 6);
        assert!(s.passes());
    }

    /// A hard line at the level (the defect's picture) fails: the level steps 30 levels, nothing else
    /// steps at all.
    #[test]
    fn a_line_at_the_level_fails() {
        let (rgba, probe) = picture(&|y| if y < 20 { 90 } else { 60 }, &|_| false);
        let s = level_step(&rgba, &probe, W, H, 20, 4, 4).expect("ground on both sides");
        assert_eq!(s.at_level, 30.0);
        assert_eq!(s.largest_elsewhere, 0.0);
        assert!(!s.passes());
    }

    /// Only the ground counts: a sky band over the horizon is not read, and a picture whose band at
    /// the level is sky reads nothing.
    #[test]
    fn sky_bands_are_not_read_and_a_level_in_the_sky_reads_nothing() {
        // The top two bands are sky: the pairs elsewhere drop from six to four.
        let (rgba, probe) = picture(&|y| 60 + y as u8, &|y| y < 12);
        let s = level_step(&rgba, &probe, W, H, 20, 4, 4).expect("ground at the level");
        assert_eq!(s.pairs_elsewhere, 4);
        // The band over the level is sky.
        let (rgba, probe) = picture(&|y| 60 + y as u8, &|y| (16..20).contains(&y));
        assert_eq!(level_step(&rgba, &probe, W, H, 20, 4, 4), None);
        // The band under the level is sky.
        let (rgba, probe) = picture(&|y| 60 + y as u8, &|y| (20..24).contains(&y));
        assert_eq!(level_step(&rgba, &probe, W, H, 20, 4, 4), None);
    }

    /// A window that leaves the picture reads only its bands inside it; with no other whole pair, it
    /// reads nothing.
    #[test]
    fn bands_outside_the_picture_are_not_read() {
        let (rgba, probe) = picture(&|y| 60 + y as u8, &|_| false);
        // Above the top: the band from row −4 is outside.
        let s = level_step(&rgba, &probe, W, H, 4, 4, 2).expect("one other pair inside");
        assert_eq!(s.pairs_elsewhere, 1);
        // Below the bottom: the bands past row 40 are outside.
        let s = level_step(&rgba, &probe, W, H, 36, 4, 2).expect("one other pair inside");
        assert_eq!(s.pairs_elsewhere, 1);
        // One band each side: no other pair at all.
        assert_eq!(level_step(&rgba, &probe, W, H, 20, 4, 1), None);
    }

    /// A band just under half ground is not read; one at half is.
    #[test]
    fn a_band_is_ground_from_half_its_pixels() {
        // Rows 16..20 over the level: two of the four rows sky (half ground): read.
        let (rgba, probe) = picture(&|y| 60 + y as u8, &|y| (16..18).contains(&y));
        assert!(level_step(&rgba, &probe, W, H, 20, 4, 4).is_some());
        // Three of the four rows sky: not read.
        let (rgba, probe) = picture(&|y| 60 + y as u8, &|y| (16..19).contains(&y));
        assert_eq!(level_step(&rgba, &probe, W, H, 20, 4, 4), None);
    }

    /// The step is the largest channel difference: a change in blue alone is a step.
    #[test]
    fn the_step_is_the_largest_channel() {
        assert_eq!(step([1.0, 2.0, 3.0], [1.0, 2.0, 13.0]), 10.0);
        assert_eq!(step([9.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 8.0);
    }
}
