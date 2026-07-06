//! Closed-form celestial math — Category A (analytic, bit-deterministic, zero per-tick
//! accumulation). A node that joins at tick T is instantly correct because position is
//! `f(seed, universe_tick)`, never `Σ f(0..T)`.
//!
//! Ported from the reference repo (`core/src/system.rs`, branch `ecs-system`) WITH its
//! semantics preserved; the two SOI functions are deliberately DISTINCT and must never
//! be merged (`docs/design/generic_transfer.md` §1.3):
//! - [`planet_soi`] — Hill-sphere style, mass-ratio based, METERS (planet capture).
//! - [`system_soi`] (in `crate::geometry`) — luminosity based, GALAXY UNITS (warp range).
//!
//! Determinism note: `solve_kepler` is iterative. Cross-binary bit-equality is NOT yet
//! gated — the current `p0_gates` determinism test only re-runs the SAME binary twice on
//! one host, which cannot catch cross-build FMA-contraction/vectorization divergence. A
//! real cross-target-cpu build-and-diff gate (SPIKE-6a: FMA contraction forbidden here +
//! pinned/exact deps + a two-way build byte-diff) is OWED and lands with P4 terrain
//! (`docs/design/DEFERRED.md`, `roadmap.json` SPIKE-6a). AUTHORITY never depends on this
//! bit-equality regardless — the source shard computes transfer poses and ships them
//! (`transfer_protocol.md` §6.1); the hole is determinism-hygiene for terrain, not authority.

/// Gravitational constant in m³/(kg·s²).
pub const G: f64 = 6.674e-11;

/// Hill-sphere SOI exponent: `r_soi = sma * (m_planet / m_star)^(2/5)`.
pub const SOI_EXPONENT: f64 = 0.4;

/// Normalize an angle to `[0, TAU)`. Rust's `%` on f64 can return negative values.
#[must_use]
pub fn normalize_angle(angle: f64) -> f64 {
    let a = angle % core::f64::consts::TAU;
    if a < 0.0 {
        a + core::f64::consts::TAU
    } else {
        a
    }
}

/// Closed-form mean anomaly at elapsed time `time_s` since epoch:
/// `M(t) = M₀ + n·t`, normalized. THE entry point for "where is this body at tick T".
#[must_use]
pub fn mean_anomaly_at(mean_anomaly_epoch: f64, mean_motion: f64, time_s: f64) -> f64 {
    normalize_angle(mean_anomaly_epoch + mean_motion * time_s)
}

/// Solve Kepler's equation `M = E - e·sin(E)` for the eccentric anomaly `E`.
/// Newton-Raphson; converges for all `e < 1`. Input `M` may be any angle.
#[must_use]
pub fn solve_kepler(mean_anomaly: f64, eccentricity: f64) -> f64 {
    let m = normalize_angle(mean_anomaly);
    let e = eccentricity;

    // Initial guess E = M + e·sin(M): good for low eccentricity, adequate everywhere
    // we generate (worlds cap e well below parabolic).
    let mut ea = m + e * m.sin();

    for _ in 0..50 {
        let sin_ea = ea.sin();
        let cos_ea = ea.cos();
        let f = ea - e * sin_ea - m;
        let fp = 1.0 - e * cos_ea;
        if fp.abs() < 1e-15 {
            break; // degenerate derivative (e -> 1 near pericenter)
        }
        let delta = f / fp;
        ea -= delta;
        if delta.abs() < 1e-12 {
            break; // converged
        }
    }
    ea
}

/// True anomaly from eccentric anomaly:
/// `ν = 2·atan2(√(1+e)·sin(E/2), √(1−e)·cos(E/2))`.
#[must_use]
pub fn eccentric_to_true_anomaly(ecc_anomaly: f64, eccentricity: f64) -> f64 {
    let e = eccentricity;
    let half_e = ecc_anomaly / 2.0;
    let y = (1.0 + e).sqrt() * half_e.sin();
    let x = (1.0 - e).sqrt() * half_e.cos();
    2.0 * y.atan2(x)
}

/// Planet sphere-of-influence radius in METERS (Hill-sphere style):
/// `r_soi = sma · (m_planet / m_star)^0.4`.
///
/// Distinct from `crate::geometry::system_soi` by design — different formula,
/// different units, different gameplay meaning. A CI test asserts they stay distinct.
#[must_use]
pub fn planet_soi(sma_m: f64, planet_mass_kg: f64, star_mass_kg: f64) -> f64 {
    sma_m * (planet_mass_kg / star_mass_kg).powf(SOI_EXPONENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn normalize_angle_covers_all_quadrants() {
        assert_eq!(normalize_angle(0.0), 0.0);
        assert!((normalize_angle(-0.5) - (core::f64::consts::TAU - 0.5)).abs() < 1e-12);
        assert!((normalize_angle(core::f64::consts::TAU + 1.0) - 1.0).abs() < 1e-12);
        assert!((normalize_angle(-3.0 * core::f64::consts::TAU)).abs() < 1e-9);
    }

    #[test]
    fn kepler_circular_orbit_is_identity() {
        for m in [0.0, 1.0, 3.0, 6.0] {
            assert!((solve_kepler(m, 0.0) - m).abs() < 1e-12, "e=0 means E=M");
        }
    }

    #[test]
    fn kepler_inverts_the_forward_equation_at_high_eccentricity() {
        // Generate E, compute M = E - e·sin(E), solve back: must recover E.
        let e = 0.95;
        for ea_expected in [0.1, 1.0, 2.5, 4.0, 6.0] {
            let m = ea_expected - e * f64::sin(ea_expected);
            let ea = solve_kepler(m, e);
            // Compare via the forward equation (E itself may differ by 2π aliasing).
            let m_back = ea - e * ea.sin();
            assert!(
                (normalize_angle(m_back) - normalize_angle(m)).abs() < 1e-9,
                "residual too large at E={ea_expected}"
            );
        }
    }

    #[test]
    fn true_anomaly_equals_eccentric_anomaly_for_circular_orbits() {
        for ea in [0.0f64, 0.7, 2.0] {
            let nu = eccentric_to_true_anomaly(ea, 0.0);
            // atan2 form returns (-π, π]; compare normalized.
            assert!((normalize_angle(nu) - normalize_angle(ea)).abs() < 1e-12);
        }
    }

    #[test]
    fn true_anomaly_leads_eccentric_anomaly_before_apocenter() {
        // For 0 < E < π on an eccentric orbit, ν > E (the body sweeps faster near
        // pericenter in angle terms).
        let nu = eccentric_to_true_anomaly(1.0, 0.5);
        assert!(nu > 1.0);
    }

    #[test]
    fn mean_anomaly_at_is_closed_form_and_normalized() {
        let m0 = 1.0;
        let n = 2.0e-7; // rad/s
        let m = mean_anomaly_at(m0, n, 1.0e8);
        assert!((0.0..core::f64::consts::TAU).contains(&m));
        // Closed form: evaluating at t directly equals epoch-shifted evaluation.
        let direct = mean_anomaly_at(m0, n, 5.0e7);
        let shifted = mean_anomaly_at(mean_anomaly_at(m0, n, 2.0e7), n, 3.0e7);
        assert!((direct - shifted).abs() < 1e-9);
    }

    #[test]
    fn planet_soi_matches_the_earth_sun_reference() {
        // Earth: sma 1.496e11 m, 5.972e24 kg; Sun 1.989e30 kg.
        // Hill-style r_soi = sma * (m/M)^0.4 ≈ 9.2e8 m (the real SOI is ~9.25e8 m).
        let r = planet_soi(1.496e11, 5.972e24, 1.989e30);
        assert!(
            (8.0e8..1.1e9).contains(&r),
            "Earth-Sun SOI out of expected band: {r:e}"
        );
    }

    #[test]
    fn gravitational_constant_is_si() {
        assert!((G - 6.674e-11).abs() < 1e-14);
    }

    #[test]
    fn kepler_degenerate_derivative_guard_terminates() {
        // e = 1.0 (parabolic, outside the generated domain but inside the input
        // domain) with M = 0: the initial guess is E = 0, where
        // f'(E) = 1 - e·cos(E) = 0 exactly — the degenerate-derivative guard must
        // break out rather than divide by ~0.
        let ea = solve_kepler(0.0, 1.0);
        assert_eq!(ea, 0.0);
    }

    proptest! {
        /// The solver's contract: for any input, the returned E satisfies Kepler's
        /// equation to tight tolerance across the eccentricity range we generate.
        #[test]
        fn kepler_residual_is_tiny(
            m in -100.0f64..100.0,
            e in 0.0f64..0.97,
        ) {
            let ea = solve_kepler(m, e);
            let residual = normalize_angle(ea - e * ea.sin()) - normalize_angle(m);
            // Residual modulo TAU: accept wrap at the boundary.
            let wrapped = residual.abs().min((residual.abs() - core::f64::consts::TAU).abs());
            prop_assert!(wrapped < 1e-8, "residual {wrapped} for M={m}, e={e}");
        }

        /// normalize_angle is idempotent and always lands in [0, TAU).
        #[test]
        fn normalize_angle_is_total(a in -1.0e6f64..1.0e6) {
            let n = normalize_angle(a);
            prop_assert!((0.0..core::f64::consts::TAU).contains(&n));
            prop_assert!((normalize_angle(n) - n).abs() < 1e-12);
        }
    }
}
