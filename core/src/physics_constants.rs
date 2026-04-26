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

/// Refractive index minus one for pure CO₂ at STP. CO₂-dominated
/// atmospheres scatter more strongly (per molecule) than N₂/O₂ — used
/// in `core::geophysics` to weight Rayleigh scattering by composition.
/// Source: Bideau-Mehu et al. 1973.
pub const REFRACTIVE_INDEX_CO2_MINUS_ONE_STP: f64 = 4.49e-4;

/// Refractive index minus one for pure CH₄ (methane) at STP. Drives
/// scattering colour for methane-rich Titan-type atmospheres.
/// Source: NIST gas refractometry tables.
pub const REFRACTIVE_INDEX_METHANE_MINUS_ONE_STP: f64 = 4.44e-4;

/// Refractive index minus one for pure H₂ at STP. Drives scattering
/// colour for hydrogen-dominated gas-giant-like atmospheres.
/// Source: NIST.
pub const REFRACTIVE_INDEX_HYDROGEN_MINUS_ONE_STP: f64 = 1.39e-4;

/// Refractive index minus one for pure N₂ at STP. Used as the reference
/// when composition is N₂-dominated (Earth-like, Titan-like).
/// Source: NIST.
pub const REFRACTIVE_INDEX_NITROGEN_MINUS_ONE_STP: f64 = 2.97e-4;

/// Refractive index minus one for pure water vapour at STP — drives
/// the scattering colour shift on humid worlds.
/// Source: NIST.
pub const REFRACTIVE_INDEX_WATER_VAPOUR_MINUS_ONE_STP: f64 = 2.49e-4;

/// Mean molecular mass of dry Earth air at sea level (78 % N₂ + 21 % O₂ + 1 % Ar).
/// Reference baseline for atmospheric scale-height; per-planet composition
/// scales this proportionally.
/// Units: kg / molecule
/// Source: standard atmospheric tables.
pub const MEAN_MOLECULAR_MASS_AIR_KG: f64 = 4.81e-26;

/// Per-species molecular masses (kg per molecule). Used by
/// `core::geophysics` to compute composition-weighted mean molecular
/// mass and per-species atmospheric retention via Jeans escape.
/// Sources: NIST atomic-weights × standard isotopic abundance.
pub const M_MOLECULE_HYDROGEN_KG: f64 = 3.347e-27; // H₂
pub const M_MOLECULE_HELIUM_KG: f64 = 6.646e-27; // He
pub const M_MOLECULE_NITROGEN_KG: f64 = 4.652e-26; // N₂
pub const M_MOLECULE_OXYGEN_KG: f64 = 5.314e-26; // O₂
pub const M_MOLECULE_WATER_KG: f64 = 2.992e-26; // H₂O
pub const M_MOLECULE_METHANE_KG: f64 = 2.664e-26; // CH₄
pub const M_MOLECULE_AMMONIA_KG: f64 = 2.829e-26; // NH₃
pub const M_MOLECULE_CO2_KG: f64 = 7.308e-26; // CO₂
pub const M_MOLECULE_SULPHUR_DIOXIDE_KG: f64 = 1.064e-25; // SO₂

/// Loschmidt's number: number density of an ideal gas at standard
/// temperature and pressure (273.15 K, 101 325 Pa). Reference scale for
/// atmospheric Rayleigh scattering (`σ_R ∝ 1/N²` per Penndorf 1957).
/// Units: m⁻³
/// Source: CODATA 2018.
pub const N_LOSCHMIDT: f64 = 2.686_780_111e25;

/// Standard atmospheric pressure at Earth sea level (1 atm).
/// Units: Pa
/// Source: ISO 2533 standard atmosphere.
pub const STANDARD_PRESSURE_PA: f64 = 101_325.0;

/// Standard temperature at Earth sea level (273.15 K = 0 °C). Used as
/// the reference for STP-relative number-density computations.
/// Units: K
/// Source: ISO 2533.
pub const STANDARD_TEMPERATURE_K: f64 = 273.15;

/// One atmosphere expressed in atm units (the dimensionless reference).
/// Provided as a named constant so per-planet `surface_pressure_atm` ratios
/// have a clean reference point.
/// Units: atm.
pub const REFERENCE_PRESSURE_ATM: f64 = 1.0;

// ─── Earth reference values (used by core::geophysics) ─────────────────────

/// Earth mass — the reference scale for terrestrial-planet bookkeeping.
/// Units: kg.
/// Source: IAU 2015 nominal value.
pub const M_EARTH_KG: f64 = 5.972_2e24;

/// Earth equatorial radius.
/// Units: m.
/// Source: IAU 2015 nominal value.
pub const R_EARTH_M: f64 = 6.378_137e6;

/// Earth surface gravity at equator.
/// Units: m · s⁻².
/// Source: ISO 80000.
pub const G_EARTH_MS2: f64 = 9.806_65;

/// Earth Bond albedo — used as the reference and as the default ground
/// albedo when broadcasting `Atmosphere.ground_albedo`.
/// Units: dimensionless.
/// Source: Stephens et al. 2015 (CERES EBAF data).
pub const ALBEDO_EARTH: f64 = 0.306;

/// Total mass of Earth's atmosphere. Reference for scaling atmospheric
/// column mass by planet retention class.
/// Units: kg
/// Source: Trenberth & Smith 2005.
pub const M_ATMOSPHERE_EARTH_KG: f64 = 5.148e18;

/// Earth's mean orbital distance from the Sun — reference distance for
/// stellar irradiance computations.
/// Units: m. Identical to `AU_M` numerically; named separately so
/// formulas read in their natural physical phrasing.
pub const ORBITAL_DISTANCE_EARTH_M: f64 = AU_M;

// ─── Atmospheric retention / classification (used by core::geophysics) ─────

/// Jeans-escape retention threshold: a species is retained on geological
/// timescales if its thermal velocity is less than the planet's escape
/// velocity divided by this factor. 6 ≈ "retained for ~Gyr"; smaller
/// factors mean faster loss.
/// Units: dimensionless.
/// Source: Pierrehumbert (2010), Principles of Planetary Climate, §1.
pub const JEANS_RETENTION_VELOCITY_RATIO: f64 = 6.0;

/// Minimum O₂ mole fraction required for an ozone layer to form (assuming
/// adequate stellar UV flux). Below this threshold the ozone term is
/// suppressed entirely. Earth's atmosphere has ~21 % O₂; the formation
/// threshold is much lower (~5 %) because ozone catalysis runs even on a
/// thin O₂ atmosphere.
/// Units: mole fraction.
/// Source: Kasting & Donahue 1980 (atmospheric oxygenation).
pub const OZONE_FORMATION_O2_FRACTION_THRESHOLD: f64 = 0.05;

/// Minimum stellar effective temperature required for ozone formation:
/// the host star must emit enough UV (λ < 240 nm) to dissociate O₂. F-,
/// G- and earlier-class stars cross this threshold; M-dwarfs do not.
/// Units: K.
/// Source: Segura et al. 2003.
pub const OZONE_FORMATION_T_STAR_K_THRESHOLD: f64 = 5000.0;

/// Number of scale heights used to define the atmosphere's "top" — the
/// altitude at which the density has dropped to `e^-N` of surface
/// density. 6 scale heights ≈ 99.75 % of atmospheric column mass below.
/// Units: dimensionless.
pub const ATMOSPHERE_TOP_SCALE_HEIGHTS: f64 = 6.0;

/// Coefficient `8` in the Maxwell-Boltzmann mean thermal speed:
/// `v_th = √(C · k_B · T / (π · m))`. Comes from integrating the
/// Maxwell-Boltzmann speed distribution against the speed weighting.
/// Units: dimensionless.
/// Source: any kinetic-theory text (e.g. Reif §7.10).
pub const KINETIC_THERMAL_VELOCITY_COEFF: f64 = 8.0;

/// Coefficient `8` in Penndorf 1957's Rayleigh-scattering cross-section
/// formula: `σ_R = (C · π³ · (n²−1)²) / (3 · N² · λ⁴)`. The 8/3 factor
/// emerges from the integration of the Rayleigh phase function over the
/// full sphere.
/// Units: dimensionless.
/// Source: Penndorf 1957, "Tables of the Refractive Index for Standard
/// Air and the Rayleigh Scattering Coefficient".
pub const PENNDORF_RAYLEIGH_COEFF: f64 = 8.0;

/// Default Mie phase-function asymmetry parameter (Henyey-Greenstein g)
/// for atmospheric aerosols. 0 = isotropic, 1 = fully forward-scattering.
/// 0.76 is the value Bevy ships in `ScatteringMedium::earthlike` and
/// matches in-situ measurements of terrestrial aerosols.
/// Units: dimensionless.
pub const MIE_PHASE_ASYMMETRY_DEFAULT: f64 = 0.76;

/// Per-class baseline aerosol scattering coefficient at the surface, in
/// `m⁻¹`. The geophysics module multiplies this by per-planet density
/// modifiers (e.g. dustier worlds × 5–20). Values are the published
/// Hillaire-2020 / Bevy-`earthlike` coefficient for a clear Earth-like
/// atmosphere.
/// Units: m⁻¹
pub const MIE_SCATTERING_BASE: f64 = 0.444e-6;

/// Per-class baseline aerosol absorption coefficient at the surface, in
/// `m⁻¹`. Pairs with `MIE_SCATTERING_BASE` for the Mie term's
/// extinction. Source: Bevy 0.18 `ScatteringMedium::earthlike` (line
/// `medium.rs:145` — `absorption: 3.996e-6`).
/// Units: m⁻¹
pub const MIE_ABSORPTION_BASE: f64 = 3.996e-6;

/// Default Mie aerosol scale height — a condensed-aerosol layer typically
/// sits in the lower troposphere. Earth observation: ~1.2 km.
/// Units: m
pub const MIE_SCALE_HEIGHT_DEFAULT_M: f64 = 1_200.0;

/// Default ozone absorption coefficients per RGB at the surface for an
/// Earth-equivalent ozone column. Scaled per-planet by the integrated
/// O₂ fraction. Source: Bevy 0.18 `ScatteringMedium::earthlike`
/// (`absorption: Vec3::new(0.650e-6, 1.881e-6, 0.085e-6)`).
/// Units: m⁻¹
pub const OZONE_ABSORPTION_EARTH_R: f64 = 0.650e-6;
pub const OZONE_ABSORPTION_EARTH_G: f64 = 1.881e-6;
pub const OZONE_ABSORPTION_EARTH_B: f64 = 0.085e-6;

/// Ozone-layer geometry on Earth: layer centre and full width.
/// Units: m
/// Source: standard atmospheric tables.
pub const OZONE_LAYER_CENTRE_EARTH_M: f64 = 25_000.0;
pub const OZONE_LAYER_WIDTH_EARTH_M: f64 = 15_000.0;

// ─── Planet rotation (used by core::planet_rotation) ───────────────────────

/// Conversion: seconds in one Earth (sidereal) day. Provided as a named
/// reference value for diagnostics + tests. Sidereal-day value chosen
/// (vs. solar day = 86 400 s exact) because rotation is measured against
/// the inertial frame, matching how `PlanetRotationParams.period_s` is
/// defined.
/// Units: s.
/// Source: IERS standard sidereal day = 23h 56m 4.0905 s ≈ 86 164.0905 s.
pub const SECONDS_PER_EARTH_DAY: f64 = 86_164.090_5;

/// Anchor for the rotation-period distribution: rotation period is built
/// as `t_freefall × FREEFALL_MULTIPLIER × seed_factor`, where
/// `t_freefall = √(R³ / (G·M))` is the planet's gravitational free-fall
/// timescale (a physical timescale derived from its bulk M and R alone).
///
/// The named multiplier is calibrated against Earth's observed values:
///   * Earth: M = 5.972·10²⁴ kg, R = 6.378·10⁶ m → t_ff ≈ 806 s.
///   * Earth sidereal day = 86 164 s.
///   * 86 164 / 806 ≈ 107.
///
/// So `FREEFALL_MULTIPLIER = 107` lands an Earth-bulk planet at one
/// Earth-day, and the rotation period of a *different* planet flows
/// from its own physics: heavier worlds get larger `G·M`, smaller
/// `t_ff`, faster rotation; lighter worlds rotate slower. The log-uniform
/// `seed_factor` (≈ ×3 envelope) covers the formation-history variance
/// that physics alone can't predict.
/// Units: dimensionless.
pub const PLANET_ROTATION_FREEFALL_MULTIPLIER: f64 = 107.0;

/// Log-uniform variance amplitude applied to the physics baseline. The
/// per-planet seed picks a multiplier in `[1/AMP, AMP]` so the rotation
/// period spans a factor of ≈ AMP² across same-bulk planets. Set to 3.0
/// so an Earth-bulk world lands in `[28 800, 259 200] s ≈ [8h, 72h]` —
/// covering Jupiter-fast through Mars-slow without hitting the
/// gameplay-implausible extremes.
/// Units: dimensionless.
pub const PLANET_ROTATION_VARIANCE_AMPLITUDE: f64 = 3.0;

/// Per-planet axial-tilt range in radians: ±45° about the orbital
/// plane normal. Bounds picked to land typical worlds in
/// the small-tilt regime (Earth = 23.5°, Mars = 25°), with a few
/// dramatic outliers but excluding extreme cases like Uranus (≈ 98°)
/// which would invert the rotation axis.
/// Units: rad.
pub const PLANET_OBLIQUITY_MIN_RAD: f64 = -PI / 4.0;
pub const PLANET_OBLIQUITY_MAX_RAD: f64 = PI / 4.0;

/// Probability per-planet that the rotation runs retrograde (Venus-like).
/// Solar-system observation is 2 of 8 (Venus, Uranus) ≈ 25 %, but most
/// retrograde is from late-stage tidal interaction rather than the
/// initial spin-up; for a fresh-formation procedural galaxy the
/// formation-rate is closer to ~5 %.
/// Units: dimensionless probability.
pub const PLANET_RETROGRADE_PROBABILITY: f64 = 0.05;

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
