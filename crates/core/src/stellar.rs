//! THE CITED STELLAR AND EARTH CALIBRATION CONSTANTS, AND THE TWO MASS RELATIONS EVERY HOST READS.
//!
//! Slice 8s moved these here from `vd-physics::taxonomy` and `vd-physics::worldgen::charter`, because
//! the client's sky (`vd-client::sky`) reads the same numbers the census and the charter author read:
//! the Sun's luminosity and radius, the astronomical unit, the Stefan–Boltzmann constant, Earth's
//! surface pressure and mean surface temperature, and the mass–luminosity and mass–radius tables the
//! star row's luminosity is turned back into a radius with. ONE source, two readers (HR3); the
//! physics crate re-exports every name unchanged. Every value is a CITED physical constant or a
//! PUBLISHED fit coefficient — the no-magic-numbers discipline.

/// Nominal solar luminosity in W — IAU 2015 Resolution B3, exactly defined.
pub const L_SUN_W: f64 = 3.828e26;
/// Nominal solar radius in m — IAU 2015 Resolution B3.
pub const R_SUN_M: f64 = 6.957e8;
/// The astronomical unit in m — IAU 2012 Resolution B2, exactly defined.
pub const AU_M: f64 = 1.495978707e11;
/// Stefan-Boltzmann constant, W m^-2 K^-4 — CODATA 2018, exact under the SI redefinition.
pub const STEFAN_BOLTZMANN_W_M2_K4: f64 = 5.670374419e-8;
/// Boltzmann constant, J/K — SI 2019, exactly defined.
pub const BOLTZMANN_J_K: f64 = 1.380649e-23;
/// Atomic mass unit, kg — CODATA 2018.
pub const ATOMIC_MASS_KG: f64 = 1.66053906660e-27;

/// Earth's mean sea-level pressure, Pa (the ISA standard atmosphere): the calibration body of the
/// charter's pressure law (ruling T9) and of the sky's aerosol scaling.
pub const EARTH_P_SURF_PA: f64 = 101_325.0;
/// Earth's mean surface temperature, K: the charter thermostat's set point (8b stage 4) and the
/// sky's surface-density calibration.
pub const EARTH_MEAN_SURFACE_K: f64 = 288.0;
/// Earth's standard surface gravity, m/s² (the ISA value the charter's pressure law is stated at).
pub const EARTH_G_MPS2: f64 = 9.81;
/// The mean molecular weight of dry air, atomic units.
pub const EARTH_MU: f64 = 28.96;

/// Earth's isothermal scale height in metres, `k·T / (μ·m_u·g)` at the three Earth words above —
/// the SAME law the charter author states a body's scale height with, so a ratio `H / H⊕` is one
/// law over two bodies. ≈ 8 428 m (`scale_height_m` pins it).
#[must_use]
pub fn earth_scale_height_m() -> f64 {
    scale_height_m(EARTH_MEAN_SURFACE_K, EARTH_MU, EARTH_G_MPS2)
}

/// The isothermal scale height `k·T / (μ·m_u·g)`, metres.
#[must_use]
pub fn scale_height_m(t_k: f64, mu: f64, g_mps2: f64) -> f64 {
    BOLTZMANN_J_K * t_k / (mu * ATOMIC_MASS_KG * g_mps2)
}

/// Broken-power-law mass-luminosity segments `(mass_hi, coeff, exponent)` ascending (Duric 2004,
/// "Advanced Astrophysics"): `L/Lsun = coeff * M^exponent`. Nearly continuous at the breaks (the
/// physics crate's `mass_luminosity_is_continuous` tripwire guards it).
pub const MLR_SEGMENTS: [(f64, f64, f64); 3] =
    [(0.43, 0.23, 2.3), (2.0, 1.0, 4.0), (55.0, 1.4, 3.5)];

/// Demircan & Kahraman 1991 (Ap&SS 181:313) stellar mass-radius segments in SOLAR units:
/// `R/Rsun = 1.06*M^0.945` below 1.66 Msun, `1.2917*M^0.555` above.
pub const STELLAR_MR_SEGMENTS: [(f64, f64, f64); 2] =
    [(1.66, 1.06, 0.945), (f64::MAX, 1.2917, 0.555)];

/// Segmented broken-power-law `coeff * x^exponent` over ascending `(x_hi, coeff, exponent)`
/// rows; `x` above every break reuses the top row. THE one evaluator every mass-radius /
/// mass-luminosity table goes through (HR3 — one machinery, tables as data).
#[must_use]
pub fn segmented_power_law(x: f64, segments: &[(f64, f64, f64)]) -> f64 {
    let (_, coeff, exponent) = segments
        .iter()
        .find(|(hi, _, _)| x <= *hi)
        .copied()
        .unwrap_or(segments[segments.len() - 1]);
    coeff * x.powf(exponent)
}

/// A star's photospheric radius in metres from its mass (Demircan & Kahraman 1991 through the one
/// segmented evaluator). ONE function, every call site by design: the System's look, the Star
/// realm's own look and the sky's sun disc all read THIS (taxonomy design par 5.2).
#[must_use]
pub fn star_radius_m(mass_msun: f64) -> f64 {
    R_SUN_M * segmented_power_law(mass_msun, &STELLAR_MR_SEGMENTS)
}

/// The main-sequence mass, solar units, whose luminosity is `luma_lsun` under [`MLR_SEGMENTS`]:
/// the inverse of the segmented law, row by row — the first row whose inverse lands at or under
/// its own mass limit answers, and the top row's inverse answers past the last limit (the same
/// tail rule the forward law has). The rows are NEARLY continuous, not exactly: at a break the two
/// rows' luminosities differ by a few percent, and a luminosity inside that jump has no mass on
/// either row — it lands ON the break (`max(lo)`), so the inverse is continuous and monotone
/// where the forward law is not. The star row ships a luminosity and no mass, so the sun disc (8s)
/// needs this way back.
#[must_use]
pub fn mass_from_luminosity_msun(luma_lsun: f64) -> f64 {
    let mut lo = 0.0;
    for (hi, coeff, exponent) in MLR_SEGMENTS {
        let mass = (luma_lsun / coeff).powf(1.0 / exponent);
        if mass <= hi {
            return mass.max(lo);
        }
        lo = hi;
    }
    let (_, coeff, exponent) = MLR_SEGMENTS[MLR_SEGMENTS.len() - 1];
    (luma_lsun / coeff).powf(1.0 / exponent)
}

/// A star's radius in metres from its luminosity alone: the mass by [`mass_from_luminosity_msun`],
/// then [`star_radius_m`]. The Sun gives the Sun's radius exactly (both tables are unity at 1 Msun).
#[must_use]
pub fn star_radius_from_luminosity_m(luma_lsun: f64) -> f64 {
    star_radius_m(mass_from_luminosity_msun(luma_lsun))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_scale_height_is_eight_and_a_half_kilometres() {
        let h = earth_scale_height_m();
        assert!((h - 8_428.0).abs() < 1.0, "{h}");
    }

    #[test]
    fn segmented_power_law_selects_a_row_and_reuses_the_top_tail() {
        // In the first row, in the second, and past the last break (the tail reuses the top row).
        assert!((segmented_power_law(1.0, &STELLAR_MR_SEGMENTS) - 1.06).abs() < 1e-12);
        assert!(
            (segmented_power_law(2.0, &STELLAR_MR_SEGMENTS) - 1.2917 * 2f64.powf(0.555)).abs()
                < 1e-12
        );
        assert!((segmented_power_law(100.0, &MLR_SEGMENTS) - 1.4 * 100f64.powf(3.5)).abs() < 1e-6);
    }

    #[test]
    fn the_sun_is_the_sun() {
        assert!((mass_from_luminosity_msun(1.0) - 1.0).abs() < 1e-12);
        assert!((star_radius_m(1.0) - 1.06 * R_SUN_M).abs() < 1.0);
        assert!((star_radius_from_luminosity_m(1.0) - 1.06 * R_SUN_M).abs() < 1.0);
    }

    #[test]
    fn the_luminosity_inverse_walks_every_segment_and_the_tail() {
        // Round trips through each forward row: a red dwarf (row 1), a sun (row 2), a hot star
        // (row 3) and a mass past the last break (the tail).
        for mass in [0.3, 1.5, 10.0, 80.0] {
            let luma = segmented_power_law(mass, &MLR_SEGMENTS);
            let back = mass_from_luminosity_msun(luma);
            assert!(
                (back - mass).abs() < 1e-9 * mass,
                "{mass} -> {luma} -> {back}"
            );
        }
    }

    #[test]
    fn a_luminosity_inside_a_breaks_jump_lands_on_the_break() {
        // The rows are nearly continuous: at 0.43 Msun row 1 says 0.0333 L☉ and row 2 says 0.0342.
        // A luminosity between them has no mass on either row; the inverse answers the break
        // itself, so the mass never jumps as the luminosity grows through the gap.
        let below = segmented_power_law(0.43, &MLR_SEGMENTS);
        let above = 0.43f64.powi(4);
        assert!(below < above, "{below} {above}");
        let inside = 0.5 * (below + above);
        let mass = mass_from_luminosity_msun(inside);
        assert!((mass - 0.43).abs() < 1e-12, "{mass}");
        // And monotone across the gap: just under and just over land on either side of the break.
        assert!(mass_from_luminosity_msun(below * 0.999) < 0.43);
        assert!(mass_from_luminosity_msun(above * 1.001) > 0.43);
    }

    #[test]
    fn a_luminosity_between_the_rows_takes_the_row_it_lands_in() {
        // The rows are nearly continuous, so a luminosity just past a break is claimed by the next
        // row, never by two: the inverse of row 1 lands ABOVE row 1's mass limit and is refused.
        let luma_at_break = segmented_power_law(0.43, &MLR_SEGMENTS) * 1.05;
        let mass = mass_from_luminosity_msun(luma_at_break);
        assert!(mass > 0.43, "{mass}");
        assert!((segmented_power_law(mass, &MLR_SEGMENTS) - luma_at_break).abs() < 1e-9);
    }
}
