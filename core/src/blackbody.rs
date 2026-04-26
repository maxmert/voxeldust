//! Blackbody-temperature → linear sRGB conversion.
//!
//! Self-contained implementation of Tanner Helland's algorithm (2012)
//! followed by an sRGB → linear conversion.
//!
//! Tanner Helland's curve is a piecewise empirical fit to the chromaticity
//! locus of an ideal blackbody radiator across the temperature range
//! ~1 000 K – 40 000 K, expressed as 8-bit gamma-encoded sRGB. We then
//! linearise each channel (the standard sRGB transfer function) so the
//! output is suitable for direct use as a Bevy `Color::linear_rgb` /
//! `DirectionalLight.color` value.
//!
//! Source: <https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html>
//!
//! This file (along with `physics_constants.rs` and
//! `client/src/config/graphics.rs`) is exempt from
//! `client/tests/no_magic_numbers.rs` — the regression coefficients below are
//! algorithm-defined, not arbitrary. They are named with their role.

// ─── Tanner Helland regression coefficients ────────────────────────────────
//
// The algorithm operates on `T_h = temperature_K / TANNER_TEMP_DIVISOR`. Each
// channel below ships its own piecewise constants.

/// Constant divisor applied to Kelvin before evaluating the regression.
/// (`T_h = T / 100`).
const TANNER_TEMP_DIVISOR: f64 = 100.0;

/// Below this scaled temperature, the red channel saturates.
const TANNER_RED_LOWER_BOUND: f64 = 66.0;
/// Below this scaled temperature, the blue channel is zero.
const TANNER_BLUE_LOWER_BOUND: f64 = 19.0;
/// Above this scaled temperature, the blue channel saturates.
const TANNER_BLUE_UPPER_BOUND: f64 = 66.0;
/// Above this scaled temperature, the green channel switches branches.
const TANNER_GREEN_BRANCH: f64 = 66.0;

/// Coefficient + exponent for red on the high-temperature branch:
/// `R_8bit = COEFF * (T_h - SHIFT)^EXPONENT`.
const TANNER_RED_HOT_COEFF: f64 = 329.698727446;
const TANNER_RED_HOT_SHIFT: f64 = 60.0;
const TANNER_RED_HOT_EXPONENT: f64 = -0.1332047592;

/// Coefficients for green on the cool branch (T_h ≤ 66):
/// `G_8bit = COEFF * ln(T_h) + OFFSET`.
const TANNER_GREEN_COOL_COEFF: f64 = 99.4708025861;
const TANNER_GREEN_COOL_OFFSET: f64 = -161.1195681661;

/// Coefficients for green on the hot branch (T_h > 66):
/// `G_8bit = COEFF * (T_h - SHIFT)^EXPONENT`.
const TANNER_GREEN_HOT_COEFF: f64 = 288.1221695283;
const TANNER_GREEN_HOT_SHIFT: f64 = 60.0;
const TANNER_GREEN_HOT_EXPONENT: f64 = -0.0755148492;

/// Coefficients for blue on the warm branch (19 < T_h < 66):
/// `B_8bit = COEFF * ln(T_h - SHIFT) + OFFSET`.
const TANNER_BLUE_WARM_COEFF: f64 = 138.5177312231;
const TANNER_BLUE_WARM_SHIFT: f64 = 10.0;
const TANNER_BLUE_WARM_OFFSET: f64 = -305.0447927307;

/// Maximum 8-bit channel value (the algorithm clamps to this).
const TANNER_MAX_8BIT: f64 = 255.0;

// ─── sRGB transfer-function constants ──────────────────────────────────────

/// Threshold of the linear-segment portion of the sRGB transfer function.
const SRGB_LINEAR_THRESHOLD: f64 = 0.04045;
/// Slope of the linear-segment portion.
const SRGB_LINEAR_SLOPE: f64 = 12.92;
/// Affine offset of the gamma-segment portion (`(c + offset) / scale`).
const SRGB_GAMMA_OFFSET: f64 = 0.055;
const SRGB_GAMMA_SCALE: f64 = 1.055;
const SRGB_GAMMA_EXPONENT: f64 = 2.4;

// ─── Public API ────────────────────────────────────────────────────────────

/// Convert a blackbody-radiator temperature in Kelvin to a linear-sRGB RGB
/// triple (each component in `[0, 1]`).
///
/// Useful range: ~1 000 K – 40 000 K. Outside this band the regression
/// extrapolates and the return value loses physical meaning, but channels
/// remain clamped to `[0, 1]`.
pub fn temperature_to_linear_rgb(temperature_k: f64) -> [f32; 3] {
    let srgb_8bit = tanner_helland_srgb_8bit(temperature_k);
    let r_lin = srgb8_to_linear(srgb_8bit[0] / TANNER_MAX_8BIT);
    let g_lin = srgb8_to_linear(srgb_8bit[1] / TANNER_MAX_8BIT);
    let b_lin = srgb8_to_linear(srgb_8bit[2] / TANNER_MAX_8BIT);
    [r_lin as f32, g_lin as f32, b_lin as f32]
}

// ─── Implementation ────────────────────────────────────────────────────────

/// Tanner Helland's piecewise approximation, returning gamma-encoded sRGB
/// channel values in `[0, 255]`.
fn tanner_helland_srgb_8bit(temperature_k: f64) -> [f64; 3] {
    let t_h = temperature_k / TANNER_TEMP_DIVISOR;

    // Red.
    let r = if t_h <= TANNER_RED_LOWER_BOUND {
        TANNER_MAX_8BIT
    } else {
        TANNER_RED_HOT_COEFF * (t_h - TANNER_RED_HOT_SHIFT).powf(TANNER_RED_HOT_EXPONENT)
    };

    // Green.
    let g = if t_h <= TANNER_GREEN_BRANCH {
        TANNER_GREEN_COOL_COEFF * t_h.ln() + TANNER_GREEN_COOL_OFFSET
    } else {
        TANNER_GREEN_HOT_COEFF * (t_h - TANNER_GREEN_HOT_SHIFT).powf(TANNER_GREEN_HOT_EXPONENT)
    };

    // Blue.
    let b = if t_h >= TANNER_BLUE_UPPER_BOUND {
        TANNER_MAX_8BIT
    } else if t_h <= TANNER_BLUE_LOWER_BOUND {
        0.0
    } else {
        TANNER_BLUE_WARM_COEFF * (t_h - TANNER_BLUE_WARM_SHIFT).ln() + TANNER_BLUE_WARM_OFFSET
    };

    [
        r.clamp(0.0, TANNER_MAX_8BIT),
        g.clamp(0.0, TANNER_MAX_8BIT),
        b.clamp(0.0, TANNER_MAX_8BIT),
    ]
}

/// Standard sRGB → linear-sRGB transfer function (per channel, input in `[0, 1]`).
fn srgb8_to_linear(c: f64) -> f64 {
    if c <= SRGB_LINEAR_THRESHOLD {
        c / SRGB_LINEAR_SLOPE
    } else {
        ((c + SRGB_GAMMA_OFFSET) / SRGB_GAMMA_SCALE).powf(SRGB_GAMMA_EXPONENT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_red_dwarf_is_red_dominant() {
        let rgb = temperature_to_linear_rgb(3_000.0);
        assert!(
            rgb[0] > rgb[2],
            "expected red > blue at 3000 K, got {:?}",
            rgb
        );
    }

    #[test]
    fn sun_like_is_balanced() {
        let rgb = temperature_to_linear_rgb(5_780.0);
        // Sun colour: warm-white. All channels nontrivial.
        for c in rgb {
            assert!(
                c > 0.3,
                "Sun-like channel below 0.3: {:?}",
                rgb,
            );
        }
    }

    #[test]
    fn hot_blue_giant_is_blue_dominant() {
        let rgb = temperature_to_linear_rgb(30_000.0);
        assert!(
            rgb[2] >= rgb[0],
            "expected blue >= red at 30000 K, got {:?}",
            rgb
        );
    }

    #[test]
    fn output_in_range() {
        for &t in &[1_000.0_f64, 3_000.0, 5_780.0, 10_000.0, 30_000.0, 40_000.0] {
            let rgb = temperature_to_linear_rgb(t);
            for c in rgb {
                assert!((0.0..=1.0).contains(&c), "channel out of range at T={}: {:?}", t, rgb);
            }
        }
    }

    #[test]
    fn deterministic() {
        let a = temperature_to_linear_rgb(5_780.0);
        let b = temperature_to_linear_rgb(5_780.0);
        assert_eq!(a, b);
    }
}
