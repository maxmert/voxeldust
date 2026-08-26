//! Physical distances that are NOT coordinate units.
//!
//! A coordinate unit is the step one whole number of the position lattice counts, and it lives on
//! [`Tier`](crate::pose::Tier). Everything here is a plain distance in metres — something the world is
//! measured *against*, never something positions are counted *in*.
//!
//! The distinction earned its own module (slice S8). The light-year below used to BE the coarse rung's
//! step, and while it was, two different jobs shared one constant: how far apart stars are, and how a
//! galaxy counts. Those pull in opposite directions — the first is a physical fact, the second must be a
//! power of two so that re-bucketing a position is exact. When the ladder gave the galaxy a two-metre step,
//! keeping the light-year under a name like `COARSE_CELL_EDGE_M` would have left the next reader believing
//! a light-year still counted something.

/// One light-year in metres — the IAU julian light-year, exact by definition: the speed of light
/// (299,792,458 m/s, itself exact by definition of the metre) times a julian year of 365.25 days.
///
/// **A DISTANCE, NOT A UNIT.** Nothing counts in light-years. This is the scale interstellar gaps are
/// reported against and the floor a star gap is judged by; positions are counted in the steps on
/// [`Tier`](crate::pose::Tier).
///
/// Stated as its derivation rather than as the digits, so it cannot be mistyped and so the definition it
/// comes from is visible. It works out to exactly 9,460,730,472,580,800 m.
pub const LIGHT_YEAR_M: f64 = 299_792_458.0 * 365.25 * 86_400.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_light_year_is_the_exact_iau_definition() {
        // The exact integer the IAU definition produces — pinned so a mistyped factor is loud, and
        // asserted as an equality against the DIGITS the derivation must reproduce.
        assert_eq!(LIGHT_YEAR_M, 9_460_730_472_580_800.0);
        // And it is NOT a power of two, which is the whole reason it stopped being a coordinate step:
        // re-bucketing a position at a non-power-of-two edge is not exactly idempotent.
        let bits = LIGHT_YEAR_M.to_bits();
        assert_ne!(
            bits & ((1 << 52) - 1),
            0,
            "a power of two has a zero mantissa"
        );
    }
}
