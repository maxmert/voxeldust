//! Named physical constants — the only place in the lighting / celestial-physics
//! pipeline where bare numeric literals are permitted.
//!
//! Every constant has:
//!   * a name with universal physical meaning,
//!   * units in the doc comment,
//!   * a citation or note explaining provenance.
//!
//! Modules under `core::stellar`, `core::geophysics`, `core::planet_rotation`
//! and `client/src/lighting` reference these by name. The `tests/no_magic_numbers.rs`
//! integration test fails if any of those modules contains a bare numeric literal
//! that does not come from this file (or from `client::lighting::quality` for
//! rendering-fidelity knobs that have no physical meaning).

/// Mathematical pi. Used in surface-area / inverse-square-law formulas.
/// Units: dimensionless.
pub const PI: f64 = std::f64::consts::PI;

/// Tau (2π). Used for full-rotation angles in planet rotation calculations.
/// Units: radians per full revolution.
pub const TAU: f64 = std::f64::consts::TAU;

// ─── Universal physical constants ──────────────────────────────────────────

/// Newtonian gravitational constant.
/// Units: m³ · kg⁻¹ · s⁻²
/// Source: CODATA 2018.
pub const G_NEWTONIAN: f64 = 6.674_30e-11;

/// Stefan-Boltzmann constant. Used for radiative flux from a blackbody:
/// `L = 4π·R² · σ · T⁴`.
/// Units: W · m⁻² · K⁻⁴
/// Source: CODATA 2018.
pub const SIGMA_STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Boltzmann constant. Used for atmospheric scale-height: `H = k·T / (m̄·g)`.
/// Units: J · K⁻¹
/// Source: CODATA 2018, exact.
pub const K_BOLTZMANN: f64 = 1.380_649e-23;

/// Planck constant. Used in blackbody-spectral derivations (the `blackbody`
/// crate uses this internally; we expose it for any future spectral work).
/// Units: J · s
/// Source: CODATA 2018, exact.
pub const H_PLANCK: f64 = 6.626_070_15e-34;

/// Speed of light in vacuum. Used in spectral / Planck derivations.
/// Units: m · s⁻¹
/// Source: CODATA 2018, exact.
pub const C_LIGHT: f64 = 299_792_458.0;

/// Avogadro's number. Used to convert between molar and per-molecule
/// quantities (atmospheric mean molecular mass).
/// Units: mol⁻¹
/// Source: CODATA 2018, exact.
pub const N_AVOGADRO: f64 = 6.022_140_76e23;

/// Universal gas constant. R = N_A · k_B.
/// Units: J · K⁻¹ · mol⁻¹
pub const R_GAS: f64 = N_AVOGADRO * K_BOLTZMANN;

// ─── Solar reference values (the "1 ☉" units) ──────────────────────────────

/// Solar mass.
/// Units: kg
/// Source: IAU 2015 nominal solar mass.
pub const M_SUN_KG: f64 = 1.988_47e30;

/// Solar radius.
/// Units: m
/// Source: IAU 2015 nominal solar radius.
pub const R_SUN_M: f64 = 6.957e8;

/// Solar luminosity (total radiative power).
/// Units: W
/// Source: IAU 2015 nominal solar luminosity.
pub const L_SUN_W: f64 = 3.828e26;

/// Solar effective temperature (Sol's blackbody-equivalent surface temperature).
/// Used as the reference for the mass-temperature relation.
/// Units: K
/// Source: IAU 2015.
pub const T_SUN_K: f64 = 5772.0;

/// Astronomical unit (mean Earth-Sun distance). Used as a reference distance
/// for illuminance computations.
/// Units: m
/// Source: IAU 2012, exact by definition.
pub const AU_M: f64 = 1.495_978_707e11;

/// Luminous efficacy of solar radiation reaching the ground (after atmosphere).
/// Used to convert radiometric power (watts) to photometric illuminance (lux):
/// `lux ≈ W/m² · η_sun`.
/// Units: lm · W⁻¹
/// Source: nominal value cited in CIE photometry tables for daylight.
pub const LUMINOUS_EFFICACY_SUN: f64 = 93.0;

// ─── Main-sequence stellar relations (used by core::stellar) ───────────────

/// Piecewise mass-luminosity relation thresholds (in M☉). The relation
/// changes shape at these mass boundaries; see Wikipedia "Mass–luminosity
/// relation" / standard stellar-structure texts (Hansen, Kawaler, Trimble).
pub const MAIN_SEQUENCE_M_L_LOW_THRESHOLD_SOLAR: f64 = 0.43;
pub const MAIN_SEQUENCE_M_L_SOLAR_THRESHOLD_SOLAR: f64 = 2.0;
pub const MAIN_SEQUENCE_M_L_HIGH_THRESHOLD_SOLAR: f64 = 55.0;

// Low-mass branch (M < 0.43 M☉): `L/L☉ = COEFF · (M/M☉)^EXPONENT`.
pub const MAIN_SEQUENCE_M_L_LOW_COEFF: f64 = 0.23;
pub const MAIN_SEQUENCE_M_L_LOW_EXPONENT: f64 = 2.3;

// Solar-mass branch (0.43 ≤ M < 2 M☉): `L/L☉ = (M/M☉)^EXPONENT`.
pub const MAIN_SEQUENCE_M_L_SOLAR_EXPONENT: f64 = 4.0;

// Intermediate branch (2 ≤ M < 55 M☉): `L/L☉ = COEFF · (M/M☉)^EXPONENT`.
pub const MAIN_SEQUENCE_M_L_INTERMEDIATE_COEFF: f64 = 1.4;
pub const MAIN_SEQUENCE_M_L_INTERMEDIATE_EXPONENT: f64 = 3.5;

// Very-high-mass branch (M ≥ 55 M☉): `L/L☉ = COEFF · (M/M☉)`.
// Asymptotic Eddington-luminosity-limited regime.
pub const MAIN_SEQUENCE_M_L_VERY_HIGH_COEFF: f64 = 32_000.0;

/// Mass-radius relation branch threshold (1 M☉). Below this the cooler
/// radiative-and-convective envelope dominates; above, radiative envelopes.
pub const MAIN_SEQUENCE_M_R_THRESHOLD_SOLAR: f64 = 1.0;

/// Mass-radius relation exponent for low-mass main-sequence stars
/// (`M < M☉`): `R/R☉ ≈ (M/M☉)^β_low`.
/// Units: dimensionless.
/// Source: Demircan & Kahraman 1991.
pub const MAIN_SEQUENCE_M_R_EXPONENT_LOW: f64 = 0.8;

/// Mass-radius relation exponent for high-mass main-sequence stars
/// (`M ≥ M☉`): `R/R☉ ≈ (M/M☉)^β_high`.
/// Units: dimensionless.
/// Source: Demircan & Kahraman 1991.
pub const MAIN_SEQUENCE_M_R_EXPONENT_HIGH: f64 = 0.57;

/// Piecewise main-sequence mass-luminosity relation. Returns `L/L☉`.
///
/// The piecewise form fits empirical data across the full main sequence
/// (0.08 M☉ red dwarfs to 90 M☉ O-class giants) better than a single
/// power law would. Each branch uses named constants above.
///
/// Pure function — bit-identical output for bit-identical input on the
/// same machine.
pub fn ms_luminosity_solar_from_mass_solar(m_solar: f64) -> f64 {
    if m_solar < MAIN_SEQUENCE_M_L_LOW_THRESHOLD_SOLAR {
        MAIN_SEQUENCE_M_L_LOW_COEFF * m_solar.powf(MAIN_SEQUENCE_M_L_LOW_EXPONENT)
    } else if m_solar < MAIN_SEQUENCE_M_L_SOLAR_THRESHOLD_SOLAR {
        m_solar.powf(MAIN_SEQUENCE_M_L_SOLAR_EXPONENT)
    } else if m_solar < MAIN_SEQUENCE_M_L_HIGH_THRESHOLD_SOLAR {
        MAIN_SEQUENCE_M_L_INTERMEDIATE_COEFF
            * m_solar.powf(MAIN_SEQUENCE_M_L_INTERMEDIATE_EXPONENT)
    } else {
        MAIN_SEQUENCE_M_L_VERY_HIGH_COEFF * m_solar
    }
}

/// Piecewise main-sequence mass-radius relation. Returns `R/R☉`.
pub fn ms_radius_solar_from_mass_solar(m_solar: f64) -> f64 {
    if m_solar < MAIN_SEQUENCE_M_R_THRESHOLD_SOLAR {
        m_solar.powf(MAIN_SEQUENCE_M_R_EXPONENT_LOW)
    } else {
        m_solar.powf(MAIN_SEQUENCE_M_R_EXPONENT_HIGH)
    }
}

// ─── Rayleigh / Mie / atmosphere references (used by core::geophysics) ─────

/// Reference wavelengths for the RGB scattering bins, used to compute per-band
/// Rayleigh coefficients (`σ_R(λ) ∝ λ⁻⁴`) and per-band absorption.
/// Units: m
/// Source: typical engine convention (red 680 nm, green 550 nm, blue 440 nm).
pub const WAVELENGTH_RED_M: f64 = 680e-9;
pub const WAVELENGTH_GREEN_M: f64 = 550e-9;
pub const WAVELENGTH_BLUE_M: f64 = 440e-9;

/// Atmospheric refractive index minus one, at standard temperature and
/// pressure for dry air. Used in the Rayleigh cross-section formula:
/// `σ_R = (8π³ · (n²-1)²) / (3 · N² · λ⁴)`.
/// Units: dimensionless.
/// Source: Bucholtz 1995 / Penndorf 1957.
pub const REFRACTIVE_INDEX_AIR_MINUS_ONE_STP: f64 = 2.78e-4;

/// Mean molecular mass of dry Earth air at sea level (78 % N₂ + 21 % O₂ + 1 % Ar).
/// Reference baseline for atmospheric scale-height; per-planet composition
/// scales this proportionally.
/// Units: kg / molecule
/// Source: standard atmospheric tables.
pub const MEAN_MOLECULAR_MASS_AIR_KG: f64 = 4.81e-26;

/// Standard atmospheric pressure at Earth sea level (1 atm).
/// Units: Pa
/// Source: ISO 2533 standard atmosphere.
pub const STANDARD_PRESSURE_PA: f64 = 101_325.0;

/// One atmosphere expressed in atm units (the dimensionless reference).
/// Provided as a named constant so per-planet `surface_pressure_atm` ratios
/// have a clean reference point.
/// Units: atm.
pub const REFERENCE_PRESSURE_ATM: f64 = 1.0;

// ─── Photometric environment defaults ──────────────────────────────────────

/// Average integrated luminance of the starfield + scattered light in a
/// populated star system, in candela per square metre. This is the
/// **physical** "darkness floor" inside a star system — not zero, because
/// scattered starlight from the host star's photometric outfall, light
/// from nearby planets at high albedo, and the integrated emission from
/// surrounding stars + nebula contribute a small but non-zero ambient
/// fill. Reference values:
///
///   * Cosmic microwave background photometric integration: ~10⁻¹⁰ cd/m²
///     (effectively zero).
///   * Average interstellar starlight integrated over the sky: ~10⁻⁴ cd/m².
///   * Earth full moonlight on lunar surface from Earthshine: ~10² cd/m².
///   * Inside a populated star system, far from the host star: ~10¹ cd/m².
///
/// We use 100 cd/m² — calibrated against Bevy 0.18's PBR ambient pipeline
/// (`pbr_ambient.wgsl::ambient_light`) at our `ev100 = 12` exposure setting
/// to land anti-sun-facing voxel faces at ~0.10 sRGB display luminance.
/// That's a visible "dim grey" rather than pitch black — matching what
/// real space photography shows on shadowed surfaces of orbiting craft
/// (where Earthshine + integrated starlight provide modest fill) without
/// washing out the strong directional contrast on sunward faces.
///
/// Voxel cubes have only 6 axis-aligned normals, so from any viewing
/// angle most visible faces share a single NdotL value. With pure black
/// ambient (~10 cd/m²), camera-visible-but-anti-sun faces collapse to
/// near-zero brightness and the hull reads as fully dark. With higher
/// ambient (~200 cd/m²) the NdotL contrast washes out. 100 cd/m² is the
/// LDR-pipeline sweet spot until Phase 2's runtime-baked IBL ships.
pub const STARFIELD_AMBIENT_FLOOR_CD_PER_M2: f64 = 100.0;
