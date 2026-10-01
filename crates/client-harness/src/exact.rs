//! ★ THE EXACT PICTURE'S COMPARE (ruling V18; the temporal path's G-DETERMINISM) — the pure half.
//!
//! Two pictures of one stand and their probes are compared on the CONTENT only: a pixel that either
//! probe marks as drawn (the ground or the ruler), and not under the overlays' rectangle (its readouts
//! carry the frame's tick, so they differ between two runs by design). The reading is the count of
//! content pixels, how many differ, and the widest channel step; the picture gate allows a step of one
//! level (ruling V18), and G-DETERMINISM asks the same of two runs of one stand.
//!
//! **The control lives here, not in the game** (the owner, 2026-10-01: "Please make sure that we don't
//! build any seams inside the game to support the tests"): the design's "a one-level change at one
//! pixel must show" is this module's own test.
//!
//! Example: the ground stand's converged still of a second flight against the first: 921 600 pixels,
//! 412 000 of them content; if one of them is one level brighter, the reading is one pixel and a widest
//! step of one — inside the tolerance, and seen.

use crate::probe::{PROBE_KIND_NONE, decode_probe};

/// What a compare read: the content pixels, how many of them differ, and the widest channel step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContentRead {
    pub content: usize,
    pub differing: usize,
    pub widest: u8,
}

/// Whether pixel `i` of an RGBA probe marks something drawn.
fn drawn(probe: &[u8], i: usize) -> bool {
    decode_probe([probe[i * 4], probe[i * 4 + 1], probe[i * 4 + 2]]).kind != PROBE_KIND_NONE
}

/// The widest channel step of pixel `i` between two RGBA pictures (red, green, blue).
fn step_at(a: &[u8], b: &[u8], i: usize) -> u8 {
    (0..3)
        .map(|k| a[i * 4 + k].abs_diff(b[i * 4 + k]))
        .max()
        .unwrap_or(0)
}

/// Whether pixel `i` of a `width`-wide picture is CONTENT: drawn in either probe, and outside the
/// overlays' rectangle `hud` (left, top, right, bottom, pixels, inclusive).
fn is_content(
    before_probe: &[u8],
    after_probe: &[u8],
    width: usize,
    hud: [f32; 4],
    i: usize,
) -> bool {
    let (x, y) = ((i % width.max(1)) as f32, (i / width.max(1)) as f32);
    let under_hud = (x >= hud[0]) & (x <= hud[2]) & (y >= hud[1]) & (y <= hud[3]);
    !under_hud & (drawn(before_probe, i) | drawn(after_probe, i))
}

/// ★ THE COMPARE of two pictures of one stand (RGBA, `width` wide, the same size) with their probes,
/// leaving out the overlays' rectangle `hud`.
#[must_use]
pub fn compare_content(
    before: &[u8],
    after: &[u8],
    before_probe: &[u8],
    after_probe: &[u8],
    width: usize,
    hud: [f32; 4],
) -> ContentRead {
    let pixels = before.len().min(after.len()) / 4;
    (0..pixels)
        .filter(|&i| is_content(before_probe, after_probe, width, hud, i))
        .fold(ContentRead::default(), |r, i| {
            let step = step_at(before, after, i);
            ContentRead {
                content: r.content + 1,
                differing: r.differing + usize::from(step > 0),
                widest: r.widest.max(step),
            }
        })
}

/// THE DIFFERENCE IMAGE for the owner: every content pixel that differs white, everything else black
/// (RGBA, the pictures' size).
#[must_use]
pub fn difference_image(
    before: &[u8],
    after: &[u8],
    before_probe: &[u8],
    after_probe: &[u8],
    width: usize,
    hud: [f32; 4],
) -> Vec<u8> {
    let pixels = before.len().min(after.len()) / 4;
    (0..pixels)
        .flat_map(|i| {
            let changed = is_content(before_probe, after_probe, width, hud, i)
                & (step_at(before, after, i) > 0);
            let v = if changed { 255 } else { 0 };
            [v, v, v, 255]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{PROBE_KIND_TERRAIN, ProbePixel, encode_probe};

    /// A 4 × 3 grey picture, and a probe whose top row is sky and the rest ground.
    fn scene() -> (Vec<u8>, Vec<u8>) {
        let picture: Vec<u8> = (0..12).flat_map(|_| [100, 90, 80, 255]).collect();
        let ground = encode_probe(ProbePixel {
            kind: PROBE_KIND_TERRAIN,
            rung: 1,
            cells: 300,
        });
        let probe: Vec<u8> = (0..12)
            .flat_map(|i| {
                if i < 4 {
                    [0, 0, 0, 255]
                } else {
                    [ground[0], ground[1], ground[2], 255]
                }
            })
            .collect();
        (picture, probe)
    }

    /// No overlay at all.
    const NO_HUD: [f32; 4] = [-10.0, -10.0, -10.0, -10.0];

    /// Two identical pictures read nothing; the content is the ground's eight pixels.
    #[test]
    fn identical_pictures_differ_nowhere() {
        let (p, probe) = scene();
        let r = compare_content(&p, &p, &probe, &probe, 4, NO_HUD);
        assert_eq!(
            r,
            ContentRead {
                content: 8,
                differing: 0,
                widest: 0
            }
        );
        assert!(
            difference_image(&p, &p, &probe, &probe, 4, NO_HUD)
                .chunks_exact(4)
                .all(|px| px[0] == 0)
        );
    }

    /// ★ THE CONTROL (the design's "a one-level change at one pixel must show"): one ground pixel one
    /// level brighter reads one pixel and a widest step of one, and the difference image marks it.
    #[test]
    fn a_one_level_change_at_one_pixel_shows() {
        let (p, probe) = scene();
        let mut q = p.clone();
        q[6 * 4 + 1] += 1;
        let r = compare_content(&p, &q, &probe, &probe, 4, NO_HUD);
        assert_eq!((r.differing, r.widest), (1, 1));
        let diff = difference_image(&p, &q, &probe, &probe, 4, NO_HUD);
        let white: Vec<usize> = diff
            .chunks_exact(4)
            .enumerate()
            .filter(|(_, px)| px[0] == 255)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(white, vec![6]);
    }

    /// The sky (drawn in neither probe) and the overlay's rectangle are left out; a pixel drawn in one
    /// probe only is content.
    #[test]
    fn the_sky_and_the_overlay_are_left_out() {
        let (p, probe) = scene();
        let mut q = p.clone();
        q[4] = 200; // the sky pixel (1, 0)
        q[9 * 4] = 200; // a ground pixel at (1, 2), under the overlay below
        let hud = [0.5, 1.5, 1.5, 2.5];
        let r = compare_content(&p, &q, &probe, &probe, 4, hud);
        assert_eq!((r.content, r.differing), (7, 0));
        // The sky pixel at (1, 0) drawn by the second probe alone becomes content, and differs.
        let mut probe_b = probe.clone();
        probe_b[4..8].copy_from_slice(&probe[16..20]);
        let r = compare_content(&p, &q, &probe, &probe_b, 4, hud);
        assert_eq!((r.content, r.differing, r.widest), (8, 1, 100));
    }
}
