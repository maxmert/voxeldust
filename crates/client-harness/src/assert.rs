//! The HR6 visual-regression assertions the capture gate (G-RENDER-SMOKE) runs over a
//! wgpu readback buffer. Each is a plain function over an RGBA8 byte slice (or an f32
//! slice) — zero GPU, deterministic, 100% region+branch coverable on synthetic buffers
//! (validated as a standalone spike before this crate existed). NO magic numbers: every
//! threshold is a named const.

/// RGBA8: 4 bytes per pixel.
const CHANNELS: usize = 4;

/// The shader / missing-texture "default pink" sentinel — its presence on screen means
/// something failed to draw with a real material.
pub const MAGENTA: [u8; 4] = [255, 0, 255, 255];

/// Rec.601 luma coefficients (perceptual brightness).
const LUMA_R: f64 = 0.299;
const LUMA_G: f64 = 0.587;
const LUMA_B: f64 = 0.114;

/// G-RENDER-SMOKE content floor: more than this fraction of non-clear pixels proves the
/// scene actually drew something (not just a cleared frame).
pub const MIN_CONTENT_FRACTION: f64 = 0.001;

/// A pixel rectangle in the readback image (e.g. a HUD anchor region).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// Count exact-magenta pixels — the no-magenta sentinel asserts this is 0.
#[must_use]
pub fn magenta_pixel_count(rgba: &[u8]) -> u64 {
    let mut count = 0;
    for px in rgba.chunks_exact(CHANNELS) {
        if px == MAGENTA.as_slice() {
            count += 1;
        }
    }
    count
}

/// Fraction of pixels differing from `clear` — content-present proof.
#[must_use]
pub fn content_present_fraction(rgba: &[u8], clear: [u8; 4]) -> f64 {
    let total = rgba.len() / CHANNELS;
    if total == 0 {
        return 0.0;
    }
    let mut non_clear = 0usize;
    for px in rgba.chunks_exact(CHANNELS) {
        if px != clear.as_slice() {
            non_clear += 1;
        }
    }
    non_clear as f64 / total as f64
}

/// Mean Rec.601 luminance over all pixels (0.0..=255.0); 0.0 for an empty buffer.
#[must_use]
pub fn mean_luminance(rgba: &[u8]) -> f64 {
    let total = rgba.len() / CHANNELS;
    if total == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for px in rgba.chunks_exact(CHANNELS) {
        sum += LUMA_R * f64::from(px[0]) + LUMA_G * f64::from(px[1]) + LUMA_B * f64::from(px[2]);
    }
    sum / total as f64
}

/// Whether mean luminance is within `[lo, hi]` (catches all-black / blown-out frames).
#[must_use]
pub fn luminance_in_range(rgba: &[u8], lo: f64, hi: f64) -> bool {
    let mean = mean_luminance(rgba);
    lo <= mean && mean <= hi
}

/// Whether a rectangle contains any non-clear pixel (HUD-anchor non-empty). Pixels of
/// the rect outside the image are skipped (an oversized rect cannot panic).
#[must_use]
pub fn region_nonempty(rgba: &[u8], width: usize, rect: Rect, clear: [u8; 4]) -> bool {
    for ry in rect.y..rect.y + rect.h {
        for rx in rect.x..rect.x + rect.w {
            let idx = (ry * width + rx) * CHANNELS;
            if let Some(px) = rgba.get(idx..idx + CHANNELS)
                && px != clear.as_slice()
            {
                return true;
            }
        }
    }
    false
}

/// Differing-pixel count between two readbacks (static-scene frame-stability: a still
/// scene differs by 0). Compares the shorter pixel count.
#[must_use]
pub fn differing_pixel_count(a: &[u8], b: &[u8]) -> u64 {
    a.chunks_exact(CHANNELS)
        .zip(b.chunks_exact(CHANNELS))
        .filter(|(x, y)| x != y)
        .count() as u64
}

/// NaN count in a float (HDR) readback — the no-NaN sentinel asserts 0.
#[must_use]
pub fn nan_count_f32(buf: &[f32]) -> u64 {
    buf.iter().filter(|x| x.is_nan()).count() as u64
}

/// The G-RENDER-SMOKE verdict over one readback: no magenta AND content present.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmokeVerdict {
    pub magenta: u64,
    pub content_fraction: f64,
    pub passed: bool,
}

/// Evaluate the permanent G-RENDER-SMOKE gate over a readback buffer.
#[must_use]
pub fn render_smoke(rgba: &[u8], clear: [u8; 4]) -> SmokeVerdict {
    let magenta = magenta_pixel_count(rgba);
    let content_fraction = content_present_fraction(rgba, clear);
    let passed = magenta == 0 && content_fraction >= MIN_CONTENT_FRACTION;
    SmokeVerdict {
        magenta,
        content_fraction,
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLEAR: [u8; 4] = [10, 20, 30, 255];
    const WHITE: [u8; 4] = [255, 255, 255, 255];

    fn solid(width: usize, height: usize, fill: [u8; 4]) -> Vec<u8> {
        fill.iter()
            .copied()
            .cycle()
            .take(width * height * CHANNELS)
            .collect()
    }
    fn set_px(buf: &mut [u8], width: usize, x: usize, y: usize, px: [u8; 4]) {
        let i = (y * width + x) * CHANNELS;
        buf[i..i + CHANNELS].copy_from_slice(&px);
    }

    #[test]
    fn magenta_sentinel_counts_only_exact_magenta() {
        let mut buf = solid(2, 2, CLEAR);
        assert_eq!(magenta_pixel_count(&buf), 0);
        set_px(&mut buf, 2, 1, 0, MAGENTA);
        set_px(&mut buf, 2, 0, 1, MAGENTA);
        assert_eq!(magenta_pixel_count(&buf), 2);
        set_px(&mut buf, 2, 1, 1, [254, 0, 255, 255]);
        assert_eq!(magenta_pixel_count(&buf), 2);
    }

    #[test]
    fn content_fraction_zero_when_empty_and_all_clear() {
        assert_eq!(content_present_fraction(&[], CLEAR), 0.0);
        assert_eq!(content_present_fraction(&solid(4, 4, CLEAR), CLEAR), 0.0);
    }

    #[test]
    fn content_fraction_counts_non_clear() {
        let mut buf = solid(2, 2, CLEAR);
        set_px(&mut buf, 2, 0, 0, WHITE);
        assert!((content_present_fraction(&buf, CLEAR) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn mean_luminance_empty_zero_white_full_black_zero() {
        assert_eq!(mean_luminance(&[]), 0.0);
        assert!((mean_luminance(&solid(2, 2, WHITE)) - 255.0).abs() < 1e-9);
        assert_eq!(mean_luminance(&solid(1, 1, [0, 0, 0, 255])), 0.0);
    }

    #[test]
    fn luminance_range_below_inside_above() {
        let buf = solid(2, 2, [100, 100, 100, 255]); // mean luma == 100
        assert!(!luminance_in_range(&buf, 150.0, 200.0));
        assert!(luminance_in_range(&buf, 50.0, 150.0));
        assert!(!luminance_in_range(&buf, 0.0, 50.0));
    }

    #[test]
    fn region_detects_content_clear_and_oob() {
        let mut buf = solid(4, 4, CLEAR);
        let r = Rect {
            x: 0,
            y: 0,
            w: 2,
            h: 2,
        };
        assert!(!region_nonempty(&buf, 4, r, CLEAR));
        set_px(&mut buf, 4, 1, 1, WHITE);
        assert!(region_nonempty(&buf, 4, r, CLEAR));
        let big = Rect {
            x: 2,
            y: 2,
            w: 10,
            h: 10,
        };
        assert!(!region_nonempty(&solid(4, 4, CLEAR), 4, big, CLEAR));
    }

    #[test]
    fn differing_pixels_zero_then_counts_then_ignores_extra() {
        let a = solid(2, 2, CLEAR);
        assert_eq!(differing_pixel_count(&a, &a), 0);
        let mut b = a.clone();
        set_px(&mut b, 2, 0, 0, WHITE);
        assert_eq!(differing_pixel_count(&a, &b), 1);
        assert_eq!(differing_pixel_count(&a, &solid(1, 1, CLEAR)), 0);
    }

    #[test]
    fn nan_sentinel_counts_nans() {
        assert_eq!(nan_count_f32(&[0.0, 1.0, -2.5]), 0);
        assert_eq!(nan_count_f32(&[0.0, f32::NAN, f32::NAN]), 2);
    }

    #[test]
    fn render_smoke_passes_only_with_no_magenta_and_content() {
        let clear_frame = solid(8, 8, CLEAR);
        let v = render_smoke(&clear_frame, CLEAR);
        assert_eq!(v.magenta, 0);
        assert!(!v.passed);
        let mut good = solid(8, 8, CLEAR);
        for x in 0..8 {
            set_px(&mut good, 8, x, 0, WHITE);
        }
        assert!(render_smoke(&good, CLEAR).passed);
        let mut bad = good.clone();
        set_px(&mut bad, 8, 0, 0, MAGENTA);
        let v = render_smoke(&bad, CLEAR);
        assert_eq!(v.magenta, 1);
        assert!(!v.passed);
    }
}
