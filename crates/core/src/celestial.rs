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
//!
//! D-45(a) Slice 1 extends this module from the anomaly chain (which stopped at true
//! anomaly) to the full elements→Cartesian ephemeris: [`OrbitalElements`] →
//! [`orbital_state`] via [`solve_kepler_fixed`]. `solve_kepler_fixed` runs a CONSTANT
//! [`KEPLER_FIXED_ITERS`] Newton steps (no data-dependent iteration count) — closing the
//! variable-step half of the determinism story (Tier 1). The residual libm-transcendental
//! divergence (Tier 2: `sin`/`cos`/`sqrt`/`atan2` + glam rotation) is STILL owed to the
//! SPIKE-6a cross-target build-and-diff gate at step-2/P4; Slice-1 consumers evaluate the
//! ephemeris ONCE at boot for a static epoch center, so per-tick cross-host drift does not
//! yet apply. Authority remains immune (the source ships transfer poses).

use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

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

/// Fixed Newton-Raphson iteration count for [`solve_kepler_fixed`] — a COMPILE-TIME
/// constant so every host runs the identical number of steps (bit-equal iteration count
/// across builds; the data-dependent-iteration determinism hole that adaptive
/// [`solve_kepler`] carries is closed). 32 steps converges to a ~1e-15 forward residual
/// across the whole *generated* eccentricity domain (`e ≤ KEPLER_ECC_MAX`, measured
/// worst-case ≈9e-16 at e=0.97 — Newton reaches 1e-12 in ≤6 steps, so 32 is safe
/// overkill). It is NOT sufficient as `e → 1` (e.g. e=0.999 leaves ≈1 rad after 32
/// steps): the generator MUST keep eccentricity at or below [`KEPLER_ECC_MAX`].
pub const KEPLER_FIXED_ITERS: usize = 32;

/// Upper eccentricity bound at which [`solve_kepler_fixed`]'s fixed [`KEPLER_FIXED_ITERS`]
/// count is verified to converge (to <1e-9 forward residual). The seed universe
/// generator's eccentricity sampler MUST stay at or below this — a fail-loud cross-slice
/// invariant: raising it forces a conscious re-verification (or increase) of the iteration
/// count, rather than silently returning garbage ephemeris for near-parabolic orbits.
pub const KEPLER_ECC_MAX: f64 = 0.97;

/// Branchless magnitude floor on Newton's derivative `f'(E) = 1 − e·cos(E)` in
/// [`solve_kepler_fixed`]. At `e=1, E=0` the derivative is exactly 0; clamping its
/// MAGNITUDE (sign-preserving) up to this floor keeps the step finite with NO
/// data-dependent `if` (unlike adaptive [`solve_kepler`]'s `if fp.abs() < 1e-15 { break }`),
/// so there is no uncoverable false arm. Same magnitude threshold, applied as a clamp.
pub const KEPLER_DENOM_FLOOR: f64 = 1.0e-15;

/// The six classical Keplerian orbital elements plus the parent mass, in SI (metres,
/// radians, kg). `mu`, `mean_motion`, and `period` are DERIVED accessors, never stored
/// (no redundant magic numbers). ONE instance describes ANY body orbiting ANY parent —
/// planet-around-star, moon-around-planet, station-around-planet — because the parent is
/// `central_mass` DATA feeding `μ = G·central_mass`, never a shard-kind `match` (HR3).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitalElements {
    /// Semi-major axis `a` in metres (`a > 0`).
    pub sma: f64,
    /// Eccentricity `e`, dimensionless (`0 ≤ e < 1`; circular = 0). The generator caps
    /// this at [`KEPLER_ECC_MAX`]; the elliptic map is undefined for `e ≥ 1`.
    pub ecc: f64,
    /// Inclination `i` in radians (tilt of the orbit plane from the parent reference plane).
    pub inclination: f64,
    /// Right ascension of the ascending node `Ω` (RAAN) in radians, measured about `+Z`.
    pub raan: f64,
    /// Argument of periapsis `ω` in radians, in the orbit plane from the ascending node.
    pub arg_periapsis: f64,
    /// Mean anomaly at epoch `M₀` in radians.
    pub mean_anomaly_epoch: f64,
    /// Parent (central) body mass in kg — DATA feeding `μ = G·central_mass` (HR3).
    pub central_mass: f64,
}

impl OrbitalElements {
    /// Standard gravitational parameter `μ = G · central_mass` in m³/s².
    #[must_use]
    pub fn mu(&self) -> f64 {
        G * self.central_mass
    }

    /// Mean motion `n = √(μ / a³)` in rad/s (always positive; retrograde is expressed via
    /// `inclination`, never a negative `n`).
    #[must_use]
    pub fn mean_motion(&self) -> f64 {
        (self.mu() / self.sma.powi(3)).sqrt()
    }

    /// Orbital period `T = 2π / n` in seconds.
    #[must_use]
    pub fn period(&self) -> f64 {
        core::f64::consts::TAU / self.mean_motion()
    }
}

/// A body's instantaneous Cartesian state (position + velocity) in the PARENT inertial
/// frame — the output of [`orbital_state`]. Position in metres, velocity in m/s. Slice-3
/// consumes only `position` (baked once as a static tick-0 epoch center); `velocity`
/// carries for step-2's per-tick `FramePlacement`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitalState {
    /// Position in the parent inertial frame, metres (fine / mm tier).
    pub position: DVec3,
    /// Velocity in the parent inertial frame, m/s.
    pub velocity: DVec3,
}

/// Solve Kepler's equation `M = E − e·sin(E)` for the eccentric anomaly `E` in EXACTLY
/// [`KEPLER_FIXED_ITERS`] Newton-Raphson steps — NO early-out, with a branchless
/// sign-preserving denominator floor ([`KEPLER_DENOM_FLOOR`]). The constant iteration
/// count makes the step count bit-equal across builds (closing the data-dependent-iteration
/// determinism hole) and keeps the body HR5-clean (no data-dependent branch ⇒ no
/// uncoverable arm). Adaptive [`solve_kepler`] is retained for adaptive/test callers.
/// Converges for `e ≤ `[`KEPLER_ECC_MAX`].
#[must_use]
pub fn solve_kepler_fixed(mean_anomaly: f64, eccentricity: f64) -> f64 {
    let m = normalize_angle(mean_anomaly);
    let e = eccentricity;
    // Same initial guess as the adaptive solver: good for low e, adequate to KEPLER_ECC_MAX.
    let mut ea = m + e * m.sin();
    for _ in 0..KEPLER_FIXED_ITERS {
        let f = ea - e * ea.sin() - m;
        let fp = 1.0 - e * ea.cos();
        // HR5: branchless sign-preserving floor — NEVER rewrite as `if fp.abs() < FLOOR`
        // (that reintroduces an uncoverable false arm). `copysign(_, +0.0) == +|x|`, so at
        // the exact `e=1, E=0` degenerate (fp = 0) the step is `0 / FLOOR = 0` and E holds.
        let denom = fp.abs().max(KEPLER_DENOM_FLOOR).copysign(fp);
        ea -= f / denom;
    }
    ea
}

/// Closed-form Cartesian state of a body at `time_s` seconds since epoch (Category A: pure
/// `f(elements, time_s)`, no per-tick accumulation). The chain: mean anomaly `M(t)=M₀+n·t`
/// → eccentric anomaly `E` (via [`solve_kepler_fixed`]) → true anomaly `ν` → perifocal
/// position `r` and velocity (angular-momentum form) → the 3-1-3 `Rz(Ω)·Rx(i)·Rz(ω)`
/// rotation applied to BOTH vectors → parent-inertial state (m, m/s). This is the
/// elements→`DVec3` piece the anomaly chain stopped short of. Defined for `0 ≤ e < 1`
/// (see [`KEPLER_ECC_MAX`]).
#[must_use]
pub fn orbital_state(elements: &OrbitalElements, time_s: f64) -> OrbitalState {
    let e = elements.ecc;
    let mu = elements.mu();
    let n = elements.mean_motion();

    let mean = mean_anomaly_at(elements.mean_anomaly_epoch, n, time_s);
    let ea = solve_kepler_fixed(mean, e);
    let nu = eccentric_to_true_anomaly(ea, e);
    let (sin_nu, cos_nu) = nu.sin_cos();

    // Perifocal position: r = a(1 − e·cosE), in the (P̂, Q̂) plane (z = 0).
    let r = elements.sma * (1.0 - e * ea.cos());
    let r_pqw = DVec3::new(r * cos_nu, r * sin_nu, 0.0);

    // Perifocal velocity (angular-momentum form): v = (μ/h)·(−sinν, e+cosν, 0) in m/s,
    // with h = √(μ·p), p = a(1−e²). Reduces to the circular tangential √(μ/a) at e = 0.
    let p = elements.sma * (1.0 - e * e);
    let h = (mu * p).sqrt();
    let v_scale = mu / h;
    let v_pqw = DVec3::new(-v_scale * sin_nu, v_scale * (e + cos_nu), 0.0);

    // 3-1-3 (Z-X-Z) perifocal→inertial. glam composes right-to-left, so the rightmost
    // Rz(ω) is applied FIRST — a direct transcription of the matrix Rz(Ω)·Rx(i)·Rz(ω). The
    // same rigid orthonormal rotation maps position and velocity identically (never
    // re-derive velocity in the inertial frame).
    let rot = DQuat::from_axis_angle(DVec3::Z, elements.raan)
        * DQuat::from_axis_angle(DVec3::X, elements.inclination)
        * DQuat::from_axis_angle(DVec3::Z, elements.arg_periapsis);

    OrbitalState {
        position: rot * r_pqw,
        velocity: rot * v_pqw,
    }
}

/// Elapsed seconds since epoch for a universe `tick`: `tick / tick_hz`. Time is SECONDS,
/// not ticks — `tick_hz` is a per-shard knob passed in (a global `TICKS_PER_SECOND` const
/// would be a magic number silently splitting a 10 Hz vs 50 Hz shard). Feed the result to
/// [`orbital_state`] as `time_s`. Contract: `tick_hz > 0` — validated once at config load,
/// not guarded here (a guard would reintroduce an HR5 branch for a caller-contract violation).
#[must_use]
#[allow(clippy::cast_precision_loss)] // universe ticks stay well below 2^53 for astronomical spans
pub fn secs_since_epoch(tick: u64, tick_hz: f64) -> f64 {
    tick as f64 / tick_hz
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

    // ---- D-45(a) Slice 1: 3D Kepler ephemeris ----------------------------------------

    const TEST_MASS: f64 = 5.972e24; // kg (Earth)
    const TEST_SMA: f64 = 7.0e6; // m (low-orbit scale — moderate magnitudes for tests)

    /// A circular, equatorial, un-rotated orbit — the analytic-golden baseline.
    fn circular() -> OrbitalElements {
        OrbitalElements {
            sma: TEST_SMA,
            ecc: 0.0,
            inclination: 0.0,
            raan: 0.0,
            arg_periapsis: 0.0,
            mean_anomaly_epoch: 0.0,
            central_mass: TEST_MASS,
        }
    }

    /// An eccentric, inclined, fully-rotated orbit — for the invariant/property tests.
    fn eccentric() -> OrbitalElements {
        OrbitalElements {
            sma: 1.2e7,
            ecc: 0.3,
            inclination: 0.5,
            raan: 0.4,
            arg_periapsis: 0.9,
            mean_anomaly_epoch: 0.2,
            central_mass: TEST_MASS,
        }
    }

    /// Circular speed `vc = √(μ/a)`.
    fn vc(e: &OrbitalElements) -> f64 {
        (e.mu() / e.sma).sqrt()
    }

    #[test]
    fn derived_accessors_match_closed_form() {
        let e = circular();
        // mu is exactly G·central_mass (a pure multiply).
        assert_eq!(e.mu(), G * TEST_MASS);
        // mean_motion recomputed with the SAME powi(3) expression the accessor uses.
        let n_expected = (G * TEST_MASS / TEST_SMA.powi(3)).sqrt();
        assert!((e.mean_motion() - n_expected).abs() < 1e-12 * n_expected);
        // period = 2π / n.
        let t_expected = core::f64::consts::TAU / n_expected;
        assert!((e.period() - t_expected).abs() < 1e-6 * t_expected);
    }

    #[test]
    fn secs_since_epoch_is_tick_over_hz() {
        // A 50 Hz shard at tick 50 and a 10 Hz shard at tick 10 are BOTH 1.0 s — tick_hz
        // is the per-shard knob, not a global const. Exact (these divide cleanly).
        assert_eq!(secs_since_epoch(0, 50.0), 0.0);
        assert_eq!(secs_since_epoch(50, 50.0), 1.0);
        assert_eq!(secs_since_epoch(10, 10.0), 1.0);
    }

    #[test]
    fn orbital_state_circular_equatorial_at_m0() {
        let e = circular();
        let s = orbital_state(&e, 0.0);
        // M = 0 ⇒ ν = 0 ⇒ position on +x at radius a; prograde velocity on +y (CCW/+z).
        assert!(s.position.abs_diff_eq(DVec3::new(TEST_SMA, 0.0, 0.0), 1e-6));
        assert!(s.velocity.abs_diff_eq(DVec3::new(0.0, vc(&e), 0.0), 1e-6));
    }

    #[test]
    fn orbital_state_circular_at_m_half_pi() {
        let e = circular();
        let t = core::f64::consts::FRAC_PI_2 / e.mean_motion(); // M = π/2
        let s = orbital_state(&e, t);
        assert!(s.position.abs_diff_eq(DVec3::new(0.0, TEST_SMA, 0.0), 1e-6));
        assert!(s.velocity.abs_diff_eq(DVec3::new(-vc(&e), 0.0, 0.0), 1e-6));
    }

    #[test]
    fn orbital_state_circular_at_m_pi() {
        let e = circular();
        let t = core::f64::consts::PI / e.mean_motion(); // M = π
        let s = orbital_state(&e, t);
        assert!(
            s.position
                .abs_diff_eq(DVec3::new(-TEST_SMA, 0.0, 0.0), 1e-6)
        );
        assert!(s.velocity.abs_diff_eq(DVec3::new(0.0, -vc(&e), 0.0), 1e-6));
    }

    #[test]
    fn orbital_state_inclination_probe_is_the_handedness_gate() {
        // e=0, i=π/2, Ω=ω=0, ν=0. Rx(π/2) fixes +x and sends +y → +z. THE glam
        // multiply-order/handedness gate: a wrong order or left-handed axis lands velocity
        // on −z (or leaves it on +y) and fails loudly.
        let e = OrbitalElements {
            inclination: core::f64::consts::FRAC_PI_2,
            ..circular()
        };
        let s = orbital_state(&e, 0.0);
        let v = vc(&e);
        assert!(s.position.abs_diff_eq(DVec3::new(TEST_SMA, 0.0, 0.0), 1e-6));
        assert!(s.velocity.abs_diff_eq(DVec3::new(0.0, 0.0, v), 1e-6));
        // Orbit normal r×v = −ŷ·(a·vc) = Rx(π/2)·ẑ.
        let expected_h = DVec3::new(0.0, -TEST_SMA * v, 0.0);
        assert!(
            s.position
                .cross(s.velocity)
                .abs_diff_eq(expected_h, 1e-9 * TEST_SMA * v)
        );
    }

    #[test]
    fn orbital_state_raan_probe() {
        // e=0, i=0, Ω=π/2, ν=0. Rz(π/2): position +x → +y, velocity +y → −x. Confirms
        // Rz(Ω) handedness independent of the ω spin.
        let e = OrbitalElements {
            raan: core::f64::consts::FRAC_PI_2,
            ..circular()
        };
        let s = orbital_state(&e, 0.0);
        let v = vc(&e);
        assert!(s.position.abs_diff_eq(DVec3::new(0.0, TEST_SMA, 0.0), 1e-6));
        assert!(s.velocity.abs_diff_eq(DVec3::new(-v, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn orbital_state_full_period_returns_to_start() {
        let e = eccentric();
        let s0 = orbital_state(&e, 0.0);
        let s_t = orbital_state(&e, e.period());
        // M(T) = M₀ + n·T = M₀ + 2π ≡ M₀ (mod 2π) ⇒ identical state.
        assert!(s_t.position.abs_diff_eq(s0.position, 1e-3));
        assert!(s_t.velocity.abs_diff_eq(s0.velocity, 1e-6));
        // Guard against a trivial pass: a quarter period MUST have moved.
        let quarter = orbital_state(&e, e.period() / 4.0);
        assert!(!quarter.position.abs_diff_eq(s0.position, 1e-3));
    }

    #[test]
    fn orbital_state_radius_equals_a_one_minus_e_cos_e() {
        let e = eccentric();
        let n = e.mean_motion();
        for &t in &[0.0, 500.0, 1500.0, 3000.0, 6000.0] {
            let s = orbital_state(&e, t);
            let mean = mean_anomaly_at(e.mean_anomaly_epoch, n, t);
            let ea = solve_kepler_fixed(mean, e.ecc);
            let r_expected = e.sma * (1.0 - e.ecc * ea.cos());
            // |r| is rotation-independent (orthonormal preserves norm).
            assert!((s.position.length() - r_expected).abs() < 1e-6 * e.sma);
        }
    }

    #[test]
    fn orbital_state_vis_viva_holds() {
        let e = eccentric();
        let mu = e.mu();
        for &t in &[0.0, 800.0, 2000.0, 5000.0] {
            let s = orbital_state(&e, t);
            let r = s.position.length();
            let v2 = s.velocity.length_squared();
            let expected = mu * (2.0 / r - 1.0 / e.sma);
            assert!((v2 - expected).abs() < 1e-9 * (mu / e.sma));
        }
    }

    #[test]
    fn orbital_state_specific_energy_is_constant() {
        let e = eccentric();
        let mu = e.mu();
        let expected = -mu / (2.0 * e.sma); // ε = −μ/2a, t-independent
        for &t in &[0.0, 900.0, 2500.0, 4200.0] {
            let s = orbital_state(&e, t);
            let r = s.position.length();
            let energy = 0.5 * s.velocity.length_squared() - mu / r;
            assert!((energy - expected).abs() < 1e-6 * (mu / e.sma));
        }
    }

    #[test]
    fn orbital_state_angular_momentum_magnitude_is_constant() {
        let e = eccentric();
        let h_expected = (e.mu() * e.sma * (1.0 - e.ecc * e.ecc)).sqrt();
        for &t in &[0.0, 700.0, 2100.0, 4800.0] {
            let s = orbital_state(&e, t);
            let h = s.position.cross(s.velocity).length();
            assert!((h - h_expected).abs() < 1e-6 * h_expected);
        }
    }

    #[test]
    fn kepler_fixed_matches_adaptive_residual() {
        // A NEGATIVE raw M lights normalize_angle's `a < 0.0` arm from WITHIN the fixed
        // solver's own coverage set (not borrowed from the normalize_angle tests).
        for &m in &[-1.0f64, 0.0, 0.5, 2.5, 5.0, 12.0] {
            for &ecc in &[0.0, 0.2, 0.6, KEPLER_ECC_MAX] {
                let ea = solve_kepler_fixed(m, ecc);
                // Forward residual (E may alias by 2π ⇒ compare via the equation).
                let resid = normalize_angle(ea - ecc * ea.sin()) - normalize_angle(m);
                let wrapped = resid
                    .abs()
                    .min((resid.abs() - core::f64::consts::TAU).abs());
                assert!(wrapped < 1e-9, "fixed residual {wrapped} at M={m}, e={ecc}");
                // Agrees with the adaptive solver via the same forward-residual measure.
                let ea_adaptive = solve_kepler(m, ecc);
                let d = normalize_angle(ea - ecc * ea.sin())
                    - normalize_angle(ea_adaptive - ecc * ea_adaptive.sin());
                let d_wrapped = d.abs().min((d.abs() - core::f64::consts::TAU).abs());
                assert!(
                    d_wrapped < 1e-9,
                    "fixed vs adaptive disagree at M={m}, e={ecc}"
                );
            }
        }
    }

    #[test]
    fn kepler_fixed_degenerate_is_branchless_and_finite() {
        // e=1, M=0: f'(E)=0 exactly at E=0. The branchless copysign floor keeps the step
        // finite and E pinned at 0.0 — same result as the adaptive guard, no branch.
        let ea = solve_kepler_fixed(0.0, 1.0);
        assert_eq!(ea, 0.0);
        assert!(ea.is_finite());
    }

    #[test]
    fn orbital_state_is_deterministic_replay() {
        let e = eccentric();
        let a = orbital_state(&e, 1234.5);
        let b = orbital_state(&e, 1234.5);
        // Same binary, same input ⇒ bit-identical (the step-1 determinism proof).
        assert_eq!(
            a.position.to_array().map(f64::to_bits),
            b.position.to_array().map(f64::to_bits)
        );
        assert_eq!(
            a.velocity.to_array().map(f64::to_bits),
            b.velocity.to_array().map(f64::to_bits)
        );
    }

    #[test]
    fn orbital_state_e_zero_i_zero_is_nonsingular() {
        // e=0 AND i=0 with Ω=ω=0: no 1/e, no 1/sin(i), no atan2(0,0) — no special-case.
        let e = circular();
        let n = e.mean_motion();
        for &frac in &[0.0, 0.25, 0.5, 0.75] {
            let t = frac * core::f64::consts::TAU / n;
            let s = orbital_state(&e, t);
            assert!(s.position.is_finite());
            assert!(s.velocity.is_finite());
            assert!((s.position.length() - TEST_SMA).abs() < 1e-6 * TEST_SMA);
        }
    }

    #[test]
    fn orbital_elements_serde_round_trips() {
        let e = eccentric();
        let bytes = postcard::to_allocvec(&e).expect("encode");
        let back: OrbitalElements = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(e, back);
    }

    #[test]
    fn orbital_state_serde_round_trips() {
        let s = orbital_state(&eccentric(), 3600.0);
        let bytes = postcard::to_allocvec(&s).expect("encode");
        let back: OrbitalState = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(s, back);
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
