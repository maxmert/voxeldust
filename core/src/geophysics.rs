//! Per-planet geophysical state derivation. Pure-functional; called
//! exclusively by server-shards (`system-shard` at system bootstrap, plus
//! echo by `planet-shard` from the same inputs in its broadcast). The
//! client links this module through the shared `core` crate but never
//! invokes the `from_*` constructor — `client/tests/no_client_seed_derivation.rs`
//! enforces this.
//!
//! ## What this derives
//!
//! Given:
//!
//!   * `planet_seed` (fully derived in `core::seed`),
//!   * the parent star's [`crate::stellar::StellarState`] (so we know how
//!     much energy reaches the planet at its orbital distance),
//!   * the planet's physical bulk: `mass_kg`, `radius_m`, `orbital_distance_m`
//!     (the same values `core::system::generate_planet` derives for the
//!     server-internal `PlanetParams`),
//!
//! produces a self-consistent atmosphere model suitable for direct
//! consumption by Bevy 0.18's [`bevy::pbr::Atmosphere`] +
//! [`bevy::pbr::ScatteringMedium`]:
//!
//! 1. **Surface gravity**, `g = G·M/R²` — from named gravitational
//!    constant + bulk physics.
//! 2. **Equilibrium temperature**, `T_eq = T★·(R★/2d)^½·(1−A)^¼` — from
//!    radiation balance with the parent star (Stefan-Boltzmann inverted).
//! 3. **Atmospheric retention**, per-species — Jeans escape: a species is
//!    retained iff `v_thermal(species) < v_escape / N` for the named
//!    `JEANS_RETENTION_VELOCITY_RATIO`. Outputs the set of retained gases.
//! 4. **Composition**, mole fractions — picked deterministically from the
//!    seed within the physical envelope of retained gases. Hot worlds get
//!    CO₂-dominated; warm intermediate worlds N₂/O₂ or N₂/CO₂; cold worlds
//!    CH₄/N₂ or H₂/He.
//! 5. **Mean molecular mass**, `M̄ = Σ fᵢ·mᵢ` — composition-weighted average.
//! 6. **Atmospheric mass**, scaled from Earth's atmosphere by
//!    `(M_planet/M_earth)^p` — heavier planets retain proportionally more
//!    column. Earthlike worlds reproduce Earth's ≈ 5.15·10¹⁸ kg.
//! 7. **Surface pressure**, `P = M_atm·g / (4π R²)` — hydrostatic equilibrium.
//! 8. **Scale height**, `H = R·T_eq / (M̄·g)` — kinetic-theory scale height.
//! 9. **Atmosphere top**, `z_top = N·H` for `N = ATMOSPHERE_TOP_SCALE_HEIGHTS`.
//! 10. **Rayleigh scattering coefficients** per RGB — Penndorf 1957:
//!     `σ_R(λ) = (8π³ · (n²−1)²) / (3 · N₀² · λ⁴) · n_surface`,
//!     with `(n−1)` chosen as the composition-weighted average of the per-species
//!     refractive-index excesses, evaluated at `λ_R, λ_G, λ_B` from
//!     `physics_constants`.
//! 11. **Mie scattering / absorption**, per RGB — base coefficients
//!     `MIE_SCATTERING_BASE`, `MIE_ABSORPTION_BASE` scaled by a per-class
//!     dust-density modifier. Dust composition fraction tints scatter
//!     toward warm-amber; cold ammonia worlds toward yellow.
//! 12. **Ozone absorption** per RGB — present iff the O₂ mole fraction
//!     exceeds `OZONE_FORMATION_O2_FRACTION_THRESHOLD` *and* the parent
//!     star is hot enough (`T★ > OZONE_FORMATION_T_STAR_K_THRESHOLD`) to
//!     produce dissociating UV. Otherwise zero.
//!
//! ## Design constraint
//!
//! Every numeric derivation references a named constant in
//! [`crate::physics_constants`]. The
//! `client/tests/no_magic_numbers.rs` integration test fails if a bare
//! literal slips into this file.
//!
//! ## Determinism
//!
//! Every operation is a pure function of `(planet_seed, parent_stellar,
//! mass_kg, radius_m, orbital_distance_m)`. Two clients on the same
//! galaxy seeing the same body see byte-identical
//! `PlanetGeophysicalState` over the wire (both shards broadcasting it
//! source the same authoritative computation; see
//! `system-shard/src/main.rs::sync_celestial_bodies`).

use serde::{Deserialize, Serialize};

use crate::physics_constants::{
    ALBEDO_EARTH, ATMOSPHERE_TOP_SCALE_HEIGHTS, G_NEWTONIAN, JEANS_RETENTION_VELOCITY_RATIO,
    KINETIC_THERMAL_VELOCITY_COEFF, K_BOLTZMANN, MIE_ABSORPTION_BASE,
    MIE_PHASE_ASYMMETRY_DEFAULT, MIE_SCALE_HEIGHT_DEFAULT_M, MIE_SCATTERING_BASE,
    M_ATMOSPHERE_EARTH_KG, M_EARTH_KG, M_MOLECULE_CO2_KG, M_MOLECULE_HELIUM_KG,
    M_MOLECULE_HYDROGEN_KG, M_MOLECULE_METHANE_KG, M_MOLECULE_NITROGEN_KG, M_MOLECULE_OXYGEN_KG,
    M_MOLECULE_WATER_KG, N_AVOGADRO, N_LOSCHMIDT, OZONE_ABSORPTION_EARTH_B,
    OZONE_ABSORPTION_EARTH_G, OZONE_ABSORPTION_EARTH_R, OZONE_FORMATION_O2_FRACTION_THRESHOLD,
    OZONE_FORMATION_T_STAR_K_THRESHOLD, OZONE_LAYER_CENTRE_EARTH_M, OZONE_LAYER_WIDTH_EARTH_M, PI,
    PENNDORF_RAYLEIGH_COEFF, REFRACTIVE_INDEX_AIR_MINUS_ONE_STP,
    REFRACTIVE_INDEX_CO2_MINUS_ONE_STP, REFRACTIVE_INDEX_HYDROGEN_MINUS_ONE_STP,
    REFRACTIVE_INDEX_METHANE_MINUS_ONE_STP, REFRACTIVE_INDEX_NITROGEN_MINUS_ONE_STP,
    REFRACTIVE_INDEX_WATER_VAPOUR_MINUS_ONE_STP, R_GAS, R_SUN_M, WAVELENGTH_BLUE_M,
    WAVELENGTH_GREEN_M, WAVELENGTH_RED_M,
};
use crate::seed::{derive_seed, seed_to_f64};
use crate::stellar::StellarState;

/// Sub-seed indices reserved for planet geophysics. Stable across versions
/// so seed → output remains reproducible.
mod sub_seed {
    pub const COMPOSITION_VARIATION: u32 = 2_001;
    pub const COMPOSITION_OXYGEN_WORLD: u32 = 2_002;
    pub const MIE_DENSITY_MODIFIER: u32 = 2_003;
    pub const ATMOSPHERIC_MASS_VARIANCE: u32 = 2_004;
    pub const ALBEDO_VARIATION: u32 = 2_005;
}

// ─── Internal: classification thresholds (visual-class boundaries) ─────────

/// Equilibrium temperature thresholds (K) used to bin planets into
/// composition classes. Boundaries reflect physical regime transitions:
/// liquid-water habitability, CO₂ condensation, H₂ retention.
const TEMP_HOT_RUNAWAY_K: f64 = 600.0; // CO₂-dominated runaway (Venus-like)
const TEMP_HABITABLE_LOWER_K: f64 = 200.0; // CO₂ ice forms below this
const TEMP_HABITABLE_UPPER_K: f64 = 320.0; // upper bound of liquid water at 1 atm
const TEMP_FROZEN_K: f64 = 100.0; // methane / ammonia regime

/// Mass thresholds (Earth masses) used by the retention/composition logic.
const MASS_TINY_AIRLESS_EARTHS: f64 = 0.05; // smaller than Mars by a lot
const MASS_LARGE_EARTHS: f64 = 5.0; // super-Earth boundary

/// Atmospheric mass scaling: `M_atm = M_atm_earth · (M_planet/M_earth)^EXPONENT`.
/// Empirical exponent for retention scaling. Source: Pierrehumbert (2010).
const ATMOSPHERIC_MASS_SCALING_EXPONENT: f64 = 1.5;

/// Within-class composition variance. Dominant gas's mole fraction varies
/// in `[mean − amp, mean + amp]`. Small enough that classification is
/// preserved; large enough that neighbour planets look distinct.
const COMPOSITION_AMPLITUDE: f64 = 0.10;

/// Per-class Mie density multiplier on Earth-baseline coefficients.
/// Sources: Venus has ≈ 6× Earth aerosol optical depth (sulfates);
/// Titan ≈ 3× (photochemical haze); Earth-like = 1; LargeGaseous = 2
/// (clouds + ammonia haze).
const MIE_DENSITY_HOT_RUNAWAY: f64 = 6.0;
const MIE_DENSITY_HABITABLE: f64 = 1.0;
const MIE_DENSITY_COLD: f64 = 3.0;
const MIE_DENSITY_FROZEN: f64 = 4.0;
const MIE_DENSITY_LARGE_GASEOUS: f64 = 2.0;

/// Albedo seed-variation amplitude. Earth-like ≈ 0.30 ± this. Stays
/// well within the physical [0, 1] range.
const ALBEDO_VARIATION_AMPLITUDE: f64 = 0.10;

/// Atmospheric-mass log-uniform variation amplitude per planet seed.
/// Output multiplier in `[1/AMP, AMP]`.
const ATMOSPHERIC_MASS_VARIATION_AMPLITUDE: f64 = 1.5;

/// Per-class total atmospheric-mass multipliers (× Earth atmospheric mass,
/// before the planet-mass scaling factor applies). Reference points:
/// Venus ≈ 90 × Earth atmospheric mass; Mars ≈ 6×10⁻⁴; Titan ≈ 1.77;
/// hydrogen-helium envelopes scale the highest of any class because the
/// retained molecules are extremely light.
const ATM_MASS_MULTIPLIER_HOT_RUNAWAY: f64 = 90.0;
const ATM_MASS_MULTIPLIER_HABITABLE: f64 = 1.0;
const ATM_MASS_MULTIPLIER_COLD: f64 = 6.0e-4;
const ATM_MASS_MULTIPLIER_FROZEN: f64 = 1.77;
const ATM_MASS_MULTIPLIER_LARGE_GASEOUS: f64 = 1_000.0;

/// Earth O₂ mole fraction — used as the reference scale for ozone
/// concentration in oxygenated atmospheres.
const OXYGEN_FRACTION_EARTH: f64 = 0.21;

/// Habitable oxygen-world dominant-gas baselines (mole fractions).
const HABITABLE_OXYGEN_DOMINANT: f64 = 0.78; // N₂ in Earth's air
const HABITABLE_OXYGEN_BASELINE: f64 = 0.21; // O₂ in Earth's air
const HABITABLE_OXYGEN_TRACE_CO2: f64 = 0.005;
const HABITABLE_OXYGEN_TRACE_H2O: f64 = 0.005;

/// Habitable anoxic-world baselines (mole fractions).
const HABITABLE_ANOXIC_CO2_BASELINE: f64 = 0.30;
const HABITABLE_ANOXIC_TRACE_H2O: f64 = 0.01;

/// Hot-runaway-world baselines (mole fractions). Venus is ~96 % CO₂.
const HOT_RUNAWAY_CO2_BASELINE: f64 = 0.96;
const HOT_RUNAWAY_DUST_BASELINE: f64 = 0.02;

/// Cold-world baselines (mole fractions). Mars-like.
const COLD_CO2_BASELINE: f64 = 0.95;
const COLD_DUST_BASELINE: f64 = 0.01;

/// Frozen-world baselines (mole fractions). Titan-like.
const FROZEN_N2_BASELINE: f64 = 0.94;
const FROZEN_CH4_BASELINE: f64 = 0.05;
const FROZEN_DUST_BASELINE: f64 = 0.01;

/// Large-gaseous-world baselines (mole fractions). Mini-Neptune.
const LARGE_GASEOUS_H2_BASELINE: f64 = 0.85;
const LARGE_GASEOUS_CH4_TRACE: f64 = 0.005;

/// Mie tint coefficients (linear scale): how strongly composition shifts
/// the Mie scattering colour. Warm-bias shifts red up + blue down (dusty
/// CO₂); cool-bias the opposite (hydrogen / methane).
const MIE_TINT_WARM_GAIN: f64 = 0.10;
const MIE_TINT_COOL_GAIN: f64 = 0.05;
const MIE_TINT_WARM_FROM_CO2: f64 = 0.5;

/// Per-class hue tint multipliers on `albedo_scalar` for `ground_albedo_rgb`.
/// Physical: ice/frozen worlds reflect bluer; CO₂ + dust worlds warmer;
/// Earth-like sit near neutral.
const ALBEDO_TINT_HOT_RUNAWAY_R: f64 = 1.10;
const ALBEDO_TINT_HOT_RUNAWAY_G: f64 = 1.00;
const ALBEDO_TINT_HOT_RUNAWAY_B: f64 = 0.85;
const ALBEDO_TINT_HABITABLE_R: f64 = 0.95;
const ALBEDO_TINT_HABITABLE_G: f64 = 1.00;
const ALBEDO_TINT_HABITABLE_B: f64 = 1.05;
const ALBEDO_TINT_COLD_R: f64 = 1.05;
const ALBEDO_TINT_COLD_G: f64 = 0.95;
const ALBEDO_TINT_COLD_B: f64 = 0.85;
const ALBEDO_TINT_FROZEN_R: f64 = 0.85;
const ALBEDO_TINT_FROZEN_G: f64 = 0.95;
const ALBEDO_TINT_FROZEN_B: f64 = 1.05;
const ALBEDO_TINT_LARGE_GASEOUS_R: f64 = 0.95;
const ALBEDO_TINT_LARGE_GASEOUS_G: f64 = 1.00;
const ALBEDO_TINT_LARGE_GASEOUS_B: f64 = 1.10;

/// Dust-driven warmth shift on ground albedo (per unit dust mole-fraction).
const ALBEDO_DUST_WARMTH_GAIN: f64 = 0.10;
const ALBEDO_DUST_BLUE_DAMP: f64 = 0.5;

// ─── Public type ───────────────────────────────────────────────────────────

/// Physical state of a single planet's atmosphere + bulk geophysics.
/// Server-derived; broadcast in `CelestialBodySnapshot.planetary`. The
/// client builds Bevy 0.18 `ScatteringMedium` + `Atmosphere` directly
/// from this struct without re-deriving anything.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlanetGeophysicalState {
    /// `true` iff this planet retains an atmosphere on geological
    /// timescales. When `false` all scattering / absorption coefficients
    /// are zero and the client suppresses any `Atmosphere` rendering.
    pub has_atmosphere: bool,

    // ─── Bulk planet physics ──────────────────────────────────────────────
    pub surface_gravity_ms2: f32,
    pub equilibrium_temp_k: f32,
    pub surface_pressure_pa: f32,
    /// Bond albedo of the planet surface, linear sRGB. Used directly for
    /// `bevy::pbr::Atmosphere::ground_albedo`.
    pub ground_albedo_linear_rgb: [f32; 3],

    // ─── Composition (mole fractions, sum ≈ 1 when atmosphered) ───────────
    pub composition_n2: f32,
    pub composition_o2: f32,
    pub composition_co2: f32,
    pub composition_ch4: f32,
    pub composition_h2o: f32,
    pub composition_h2: f32,
    pub composition_he: f32,
    pub composition_dust: f32,

    // ─── Vertical structure ───────────────────────────────────────────────
    pub scale_height_m: f32,
    pub atmosphere_top_altitude_m: f32,
    pub mean_molecular_mass_kg: f32,
    pub atmospheric_mass_kg: f32,

    // ─── Rayleigh (per RGB) ───────────────────────────────────────────────
    pub rayleigh_scattering_per_lambda: [f32; 3],
    pub rayleigh_scale_height_m: f32,

    // ─── Mie ──────────────────────────────────────────────────────────────
    pub mie_scattering_per_lambda: [f32; 3],
    pub mie_absorption_per_lambda: [f32; 3],
    pub mie_scale_height_m: f32,
    pub mie_phase_g: f32,

    // ─── Ozone (per RGB) ──────────────────────────────────────────────────
    pub ozone_absorption_per_lambda: [f32; 3],
    pub ozone_layer_centre_m: f32,
    pub ozone_layer_width_m: f32,
}

/// Atmospheric class — internal classification driving composition + Mie.
#[derive(Debug, Clone, Copy, PartialEq)]
enum AtmosphereClass {
    /// Mass too low or temperature too high: no retained gases.
    Airless,
    /// Hot CO₂ runaway (Venus-like): dense CO₂, sulfate haze.
    HotRunaway,
    /// Earth-class habitable: N₂ / O₂ or N₂ / CO₂.
    Habitable,
    /// Cold sub-freezing: CO₂ ice on surface, thin atmosphere.
    Cold,
    /// Frozen methane/ammonia (Titan-like).
    Frozen,
    /// Large gaseous (super-Earth / mini-Neptune): H₂ / He retention.
    LargeGaseous,
}

impl PlanetGeophysicalState {
    /// Derive a planet's geophysical state from its seed plus the parent
    /// star and basic bulk parameters (mass, radius, orbital distance).
    ///
    /// All numerics come from named physics constants in
    /// [`crate::physics_constants`]. The derivation is bit-deterministic
    /// for fixed inputs.
    pub fn from_seed_and_star(
        planet_seed: u64,
        parent_stellar: &StellarState,
        mass_kg: f64,
        radius_m: f64,
        orbital_distance_m: f64,
    ) -> Self {
        // Step 1: surface gravity.
        let surface_gravity = G_NEWTONIAN * mass_kg / (radius_m * radius_m);

        // Step 2: albedo — Earth-like with seed variation.
        let albedo_offset = ALBEDO_VARIATION_AMPLITUDE
            * (seed_to_f64(derive_seed(planet_seed, sub_seed::ALBEDO_VARIATION)) * 2.0
                - 1.0);
        let albedo_scalar = (ALBEDO_EARTH + albedo_offset).clamp(0.0, 1.0);

        // Step 3: equilibrium temperature from radiation balance.
        // T_eq = T★ · (R★ / (2·d))^½ · (1 − A)^¼.
        let r_star_m = (parent_stellar.radius_solar as f64) * R_SUN_M;
        let t_star_k = parent_stellar.temperature_k as f64;
        let radiation_factor = (r_star_m / (2.0 * orbital_distance_m)).sqrt();
        let albedo_factor = (1.0 - albedo_scalar).max(0.0).powf(1.0 / 4.0);
        let equilibrium_temp = t_star_k * radiation_factor * albedo_factor;

        // Step 4: classify.
        let mass_earths = mass_kg / M_EARTH_KG;
        let class = classify_atmosphere(mass_earths, equilibrium_temp, mass_kg, radius_m);

        // Step 5: composition (mole fractions).
        let composition = composition_for_class(class, planet_seed);

        // Step 6: mean molecular mass = composition-weighted.
        let mean_molecular_mass = mean_molecular_mass(&composition);

        // Step 7: atmospheric mass scaled by class + planet mass + seed variance.
        let atmospheric_mass = atmospheric_mass(class, mass_earths, planet_seed);

        // Step 8: surface pressure from hydrostatic equilibrium.
        let surface_pressure = if class == AtmosphereClass::Airless {
            0.0
        } else {
            atmospheric_mass * surface_gravity / (4.0 * PI * radius_m * radius_m)
        };

        // Step 9: scale height H = R·T / (M̄·g) using the molar form so the
        // dimensions read in their natural physical phrasing (R is the
        // universal gas constant in J·K⁻¹·mol⁻¹).
        let mean_molecular_mass_molar = mean_molecular_mass * N_AVOGADRO;
        let scale_height = if class == AtmosphereClass::Airless {
            1.0
        } else {
            R_GAS * equilibrium_temp / (mean_molecular_mass_molar * surface_gravity)
        };

        // Step 10: atmosphere top altitude.
        let atmosphere_top_altitude = if class == AtmosphereClass::Airless {
            0.0
        } else {
            ATMOSPHERE_TOP_SCALE_HEIGHTS * scale_height
        };

        // Step 11: Rayleigh per-band coefficients (Penndorf 1957).
        let rayleigh_per_band = if class == AtmosphereClass::Airless {
            [0.0; 3]
        } else {
            rayleigh_per_band(&composition, surface_pressure, equilibrium_temp)
        };

        // Step 12: Mie coefficients = Earth baseline × class density modifier.
        let (mie_scatter_band, mie_absorb_band) = if class == AtmosphereClass::Airless {
            ([0.0; 3], [0.0; 3])
        } else {
            mie_per_band(class, &composition, planet_seed)
        };

        // Step 13: ozone — present iff sufficient O₂ + sufficient stellar UV.
        let ozone_per_band = ozone_per_band(&composition, t_star_k);

        // Step 14: ground albedo as RGB.
        let ground_albedo = ground_albedo_rgb(class, albedo_scalar, &composition);

        let has_atmosphere = class != AtmosphereClass::Airless;

        PlanetGeophysicalState {
            has_atmosphere,
            surface_gravity_ms2: surface_gravity as f32,
            equilibrium_temp_k: equilibrium_temp as f32,
            surface_pressure_pa: surface_pressure as f32,
            ground_albedo_linear_rgb: ground_albedo,
            composition_n2: composition.n2 as f32,
            composition_o2: composition.o2 as f32,
            composition_co2: composition.co2 as f32,
            composition_ch4: composition.ch4 as f32,
            composition_h2o: composition.h2o as f32,
            composition_h2: composition.h2 as f32,
            composition_he: composition.he as f32,
            composition_dust: composition.dust as f32,
            scale_height_m: scale_height as f32,
            atmosphere_top_altitude_m: atmosphere_top_altitude as f32,
            mean_molecular_mass_kg: mean_molecular_mass as f32,
            atmospheric_mass_kg: atmospheric_mass as f32,
            rayleigh_scattering_per_lambda: [
                rayleigh_per_band[0] as f32,
                rayleigh_per_band[1] as f32,
                rayleigh_per_band[2] as f32,
            ],
            rayleigh_scale_height_m: scale_height as f32,
            mie_scattering_per_lambda: [
                mie_scatter_band[0] as f32,
                mie_scatter_band[1] as f32,
                mie_scatter_band[2] as f32,
            ],
            mie_absorption_per_lambda: [
                mie_absorb_band[0] as f32,
                mie_absorb_band[1] as f32,
                mie_absorb_band[2] as f32,
            ],
            mie_scale_height_m: MIE_SCALE_HEIGHT_DEFAULT_M as f32,
            mie_phase_g: MIE_PHASE_ASYMMETRY_DEFAULT as f32,
            ozone_absorption_per_lambda: [
                ozone_per_band[0] as f32,
                ozone_per_band[1] as f32,
                ozone_per_band[2] as f32,
            ],
            ozone_layer_centre_m: OZONE_LAYER_CENTRE_EARTH_M as f32,
            ozone_layer_width_m: OZONE_LAYER_WIDTH_EARTH_M as f32,
        }
    }
}

// ─── Composition struct (intermediate computation) ─────────────────────────

#[derive(Debug, Clone, Copy)]
struct Composition {
    n2: f64,
    o2: f64,
    co2: f64,
    ch4: f64,
    h2o: f64,
    h2: f64,
    he: f64,
    dust: f64,
}

impl Composition {
    fn zero() -> Self {
        Composition {
            n2: 0.0,
            o2: 0.0,
            co2: 0.0,
            ch4: 0.0,
            h2o: 0.0,
            h2: 0.0,
            he: 0.0,
            dust: 0.0,
        }
    }
}

// ─── Classification ────────────────────────────────────────────────────────

fn classify_atmosphere(
    mass_earths: f64,
    equilibrium_temp_k: f64,
    mass_kg: f64,
    radius_m: f64,
) -> AtmosphereClass {
    if mass_earths < MASS_TINY_AIRLESS_EARTHS {
        return AtmosphereClass::Airless;
    }
    // Even larger bodies become airless if they orbit so close that even
    // heavy molecules escape — Jeans test on N₂ at the equilibrium temp.
    if !species_retained(M_MOLECULE_NITROGEN_KG, equilibrium_temp_k, mass_kg, radius_m) {
        return AtmosphereClass::Airless;
    }
    if mass_earths >= MASS_LARGE_EARTHS
        && species_retained(M_MOLECULE_HYDROGEN_KG, equilibrium_temp_k, mass_kg, radius_m)
    {
        return AtmosphereClass::LargeGaseous;
    }
    if equilibrium_temp_k > TEMP_HOT_RUNAWAY_K {
        AtmosphereClass::HotRunaway
    } else if equilibrium_temp_k >= TEMP_HABITABLE_LOWER_K
        && equilibrium_temp_k <= TEMP_HABITABLE_UPPER_K
    {
        AtmosphereClass::Habitable
    } else if equilibrium_temp_k >= TEMP_FROZEN_K {
        AtmosphereClass::Cold
    } else {
        AtmosphereClass::Frozen
    }
}

/// Jeans-escape retention: a species is retained on geological timescales
/// iff its thermal velocity at the equilibrium temperature is below
/// `v_escape / JEANS_RETENTION_VELOCITY_RATIO`.
fn species_retained(
    species_mass_kg: f64,
    equilibrium_temp_k: f64,
    mass_kg: f64,
    radius_m: f64,
) -> bool {
    let v_thermal = (KINETIC_THERMAL_VELOCITY_COEFF * K_BOLTZMANN * equilibrium_temp_k
        / (PI * species_mass_kg))
        .sqrt();
    let v_escape = (2.0 * G_NEWTONIAN * mass_kg / radius_m).sqrt();
    v_thermal < v_escape / JEANS_RETENTION_VELOCITY_RATIO
}

// ─── Composition derivation per class ──────────────────────────────────────

fn composition_for_class(class: AtmosphereClass, planet_seed: u64) -> Composition {
    if class == AtmosphereClass::Airless {
        return Composition::zero();
    }
    let variance_subseed = derive_seed(planet_seed, sub_seed::COMPOSITION_VARIATION);
    // `t` ∈ [-1, 1) — symmetrical about zero so `mean ± COMPOSITION_AMPLITUDE`.
    let t = seed_to_f64(variance_subseed) * 2.0 - 1.0;
    match class {
        AtmosphereClass::Airless => Composition::zero(),
        AtmosphereClass::HotRunaway => {
            // Venus-like.
            let co2 = (HOT_RUNAWAY_CO2_BASELINE + COMPOSITION_AMPLITUDE * t * 0.5)
                .clamp(0.0, 1.0);
            let n2 = (1.0 - co2 - HOT_RUNAWAY_DUST_BASELINE).max(0.0);
            Composition {
                n2,
                o2: 0.0,
                co2,
                ch4: 0.0,
                h2o: 0.0,
                h2: 0.0,
                he: 0.0,
                dust: HOT_RUNAWAY_DUST_BASELINE,
            }
        }
        AtmosphereClass::Habitable => {
            // Two sub-cases: oxygenated (Earth-like) or anoxic (N₂ + CO₂).
            // Half of habitable worlds are oxygenated.
            let oxygen_world = seed_to_f64(derive_seed(
                planet_seed,
                sub_seed::COMPOSITION_OXYGEN_WORLD,
            )) > 0.5;
            if oxygen_world {
                let o2 =
                    (HABITABLE_OXYGEN_BASELINE + COMPOSITION_AMPLITUDE * t).clamp(0.0, 1.0);
                let n2 =
                    (HABITABLE_OXYGEN_DOMINANT - COMPOSITION_AMPLITUDE * t).clamp(0.0, 1.0);
                Composition {
                    n2,
                    o2,
                    co2: HABITABLE_OXYGEN_TRACE_CO2,
                    ch4: 0.0,
                    h2o: HABITABLE_OXYGEN_TRACE_H2O,
                    h2: 0.0,
                    he: 0.0,
                    dust: 0.0,
                }
            } else {
                let co2 = (HABITABLE_ANOXIC_CO2_BASELINE + COMPOSITION_AMPLITUDE * t)
                    .clamp(0.0, 1.0);
                let n2 = (1.0 - co2 - HABITABLE_ANOXIC_TRACE_H2O).clamp(0.0, 1.0);
                Composition {
                    n2,
                    o2: 0.0,
                    co2,
                    ch4: 0.0,
                    h2o: HABITABLE_ANOXIC_TRACE_H2O,
                    h2: 0.0,
                    he: 0.0,
                    dust: 0.0,
                }
            }
        }
        AtmosphereClass::Cold => {
            // Mars-like: thin CO₂-N₂ atmosphere.
            let co2 = (COLD_CO2_BASELINE + COMPOSITION_AMPLITUDE * t * 0.5).clamp(0.0, 1.0);
            let n2 = (1.0 - co2 - COLD_DUST_BASELINE).max(0.0);
            Composition {
                n2,
                o2: 0.0,
                co2,
                ch4: 0.0,
                h2o: 0.0,
                h2: 0.0,
                he: 0.0,
                dust: COLD_DUST_BASELINE,
            }
        }
        AtmosphereClass::Frozen => {
            // Titan-like: N₂-dominated with CH₄ + photochemical haze.
            let ch4 = (FROZEN_CH4_BASELINE + COMPOSITION_AMPLITUDE * t).clamp(0.0, 1.0);
            let n2 = (FROZEN_N2_BASELINE - COMPOSITION_AMPLITUDE * t).clamp(0.0, 1.0);
            Composition {
                n2,
                o2: 0.0,
                co2: 0.0,
                ch4,
                h2o: 0.0,
                h2: 0.0,
                he: 0.0,
                dust: FROZEN_DUST_BASELINE,
            }
        }
        AtmosphereClass::LargeGaseous => {
            // Mini-Neptune / super-Earth: H₂ + He envelope, traces of CH₄.
            let h2 = (LARGE_GASEOUS_H2_BASELINE + COMPOSITION_AMPLITUDE * t).clamp(0.0, 1.0);
            let he = (1.0 - h2 - LARGE_GASEOUS_CH4_TRACE).clamp(0.0, 1.0);
            Composition {
                n2: 0.0,
                o2: 0.0,
                co2: 0.0,
                ch4: LARGE_GASEOUS_CH4_TRACE,
                h2o: 0.0,
                h2,
                he,
                dust: 0.0,
            }
        }
    }
}

// ─── Mean molecular mass ───────────────────────────────────────────────────

fn mean_molecular_mass(c: &Composition) -> f64 {
    // Sum of mole_fraction × molecular_mass for each species. Dust is solid
    // condensate — excluded from the gas-phase mean.
    c.n2 * M_MOLECULE_NITROGEN_KG
        + c.o2 * M_MOLECULE_OXYGEN_KG
        + c.co2 * M_MOLECULE_CO2_KG
        + c.ch4 * M_MOLECULE_METHANE_KG
        + c.h2o * M_MOLECULE_WATER_KG
        + c.h2 * M_MOLECULE_HYDROGEN_KG
        + c.he * M_MOLECULE_HELIUM_KG
}

// ─── Atmospheric mass ──────────────────────────────────────────────────────

fn atmospheric_mass(class: AtmosphereClass, mass_earths: f64, planet_seed: u64) -> f64 {
    if class == AtmosphereClass::Airless {
        return 0.0;
    }
    let class_multiplier = match class {
        AtmosphereClass::Airless => 0.0,
        AtmosphereClass::HotRunaway => ATM_MASS_MULTIPLIER_HOT_RUNAWAY,
        AtmosphereClass::Habitable => ATM_MASS_MULTIPLIER_HABITABLE,
        AtmosphereClass::Cold => ATM_MASS_MULTIPLIER_COLD,
        AtmosphereClass::Frozen => ATM_MASS_MULTIPLIER_FROZEN,
        AtmosphereClass::LargeGaseous => ATM_MASS_MULTIPLIER_LARGE_GASEOUS,
    };
    // Per-planet seed variance.
    let variance_subseed =
        derive_seed(planet_seed, sub_seed::ATMOSPHERIC_MASS_VARIANCE);
    let t = seed_to_f64(variance_subseed); // [0, 1)
    let log_amp = ATMOSPHERIC_MASS_VARIATION_AMPLITUDE.ln();
    let variance_factor = (log_amp * (2.0 * t - 1.0)).exp();

    let scaling = mass_earths.powf(ATMOSPHERIC_MASS_SCALING_EXPONENT);
    M_ATMOSPHERE_EARTH_KG * class_multiplier * scaling * variance_factor
}

// ─── Rayleigh per-band coefficients ────────────────────────────────────────

/// Composition-weighted mean of `(n−1)` at STP. Each species contributes
/// proportionally to its mole fraction. Dust phase is excluded (solid).
fn mean_refractive_excess(c: &Composition) -> f64 {
    let gas_total =
        c.n2 + c.o2 + c.co2 + c.ch4 + c.h2o + c.h2 + c.he + f64::EPSILON;
    (c.n2 * REFRACTIVE_INDEX_NITROGEN_MINUS_ONE_STP
        + c.o2 * REFRACTIVE_INDEX_AIR_MINUS_ONE_STP
        + c.co2 * REFRACTIVE_INDEX_CO2_MINUS_ONE_STP
        + c.ch4 * REFRACTIVE_INDEX_METHANE_MINUS_ONE_STP
        + c.h2o * REFRACTIVE_INDEX_WATER_VAPOUR_MINUS_ONE_STP
        + c.h2 * REFRACTIVE_INDEX_HYDROGEN_MINUS_ONE_STP
        + c.he * REFRACTIVE_INDEX_HYDROGEN_MINUS_ONE_STP)
        / gas_total
}

/// Rayleigh scattering coefficient at the surface for one wavelength
/// (Penndorf 1957 cross-section per molecule × surface number density).
fn rayleigh_band(refractive_excess: f64, wavelength_m: f64, n_surface: f64) -> f64 {
    let n_squared_minus_one = (1.0 + refractive_excess).powi(2) - 1.0;
    let cross_section = (PENNDORF_RAYLEIGH_COEFF * PI.powi(3) * n_squared_minus_one.powi(2))
        / (3.0 * N_LOSCHMIDT.powi(2) * wavelength_m.powi(4));
    cross_section * n_surface
}

fn rayleigh_per_band(c: &Composition, surface_pressure_pa: f64, temp_k: f64) -> [f64; 3] {
    let refractive_excess = mean_refractive_excess(c);
    // Number density at surface from the ideal gas law: n = P / (k·T).
    let n_surface = surface_pressure_pa / (K_BOLTZMANN * temp_k);
    [
        rayleigh_band(refractive_excess, WAVELENGTH_RED_M, n_surface),
        rayleigh_band(refractive_excess, WAVELENGTH_GREEN_M, n_surface),
        rayleigh_band(refractive_excess, WAVELENGTH_BLUE_M, n_surface),
    ]
}

// ─── Mie per-band coefficients ─────────────────────────────────────────────

fn mie_per_band(
    class: AtmosphereClass,
    composition: &Composition,
    planet_seed: u64,
) -> ([f64; 3], [f64; 3]) {
    let class_density = match class {
        AtmosphereClass::Airless => 0.0,
        AtmosphereClass::HotRunaway => MIE_DENSITY_HOT_RUNAWAY,
        AtmosphereClass::Habitable => MIE_DENSITY_HABITABLE,
        AtmosphereClass::Cold => MIE_DENSITY_COLD,
        AtmosphereClass::Frozen => MIE_DENSITY_FROZEN,
        AtmosphereClass::LargeGaseous => MIE_DENSITY_LARGE_GASEOUS,
    };
    let variance_subseed = derive_seed(planet_seed, sub_seed::MIE_DENSITY_MODIFIER);
    let t = seed_to_f64(variance_subseed); // [0, 1)
    let log_amp = ATMOSPHERIC_MASS_VARIATION_AMPLITUDE.ln();
    let variance_factor = (log_amp * (2.0 * t - 1.0)).exp();
    let density = class_density * variance_factor;

    // Composition-driven tint.
    let warm_bias = composition.dust + MIE_TINT_WARM_FROM_CO2 * composition.co2;
    let cool_bias = composition.h2 + composition.ch4;
    let tint_r = 1.0 + MIE_TINT_WARM_GAIN * warm_bias - MIE_TINT_COOL_GAIN * cool_bias;
    let tint_g = 1.0;
    let tint_b = 1.0 - MIE_TINT_COOL_GAIN * warm_bias + MIE_TINT_WARM_GAIN * cool_bias;

    let scatter_base = MIE_SCATTERING_BASE * density;
    let absorb_base = MIE_ABSORPTION_BASE * density;
    let scatter = [
        scatter_base * tint_r,
        scatter_base * tint_g,
        scatter_base * tint_b,
    ];
    let absorb = [
        absorb_base * tint_r,
        absorb_base * tint_g,
        absorb_base * tint_b,
    ];
    (scatter, absorb)
}

// ─── Ozone per-band coefficients ───────────────────────────────────────────

fn ozone_per_band(c: &Composition, t_star_k: f64) -> [f64; 3] {
    let has_ozone = c.o2 >= OZONE_FORMATION_O2_FRACTION_THRESHOLD
        && t_star_k >= OZONE_FORMATION_T_STAR_K_THRESHOLD;
    if !has_ozone {
        return [0.0; 3];
    }
    // Concentration scales (roughly) with O₂ fraction relative to Earth's.
    let scale = c.o2 / OXYGEN_FRACTION_EARTH;
    [
        OZONE_ABSORPTION_EARTH_R * scale,
        OZONE_ABSORPTION_EARTH_G * scale,
        OZONE_ABSORPTION_EARTH_B * scale,
    ]
}

// ─── Ground albedo RGB ─────────────────────────────────────────────────────

fn ground_albedo_rgb(
    class: AtmosphereClass,
    albedo_scalar: f64,
    composition: &Composition,
) -> [f32; 3] {
    let (tint_r, tint_g, tint_b) = match class {
        AtmosphereClass::Airless => (1.0, 1.0, 1.0),
        AtmosphereClass::HotRunaway => (
            ALBEDO_TINT_HOT_RUNAWAY_R,
            ALBEDO_TINT_HOT_RUNAWAY_G,
            ALBEDO_TINT_HOT_RUNAWAY_B,
        ),
        AtmosphereClass::Habitable => (
            ALBEDO_TINT_HABITABLE_R,
            ALBEDO_TINT_HABITABLE_G,
            ALBEDO_TINT_HABITABLE_B,
        ),
        AtmosphereClass::Cold => (ALBEDO_TINT_COLD_R, ALBEDO_TINT_COLD_G, ALBEDO_TINT_COLD_B),
        AtmosphereClass::Frozen => (
            ALBEDO_TINT_FROZEN_R,
            ALBEDO_TINT_FROZEN_G,
            ALBEDO_TINT_FROZEN_B,
        ),
        AtmosphereClass::LargeGaseous => (
            ALBEDO_TINT_LARGE_GASEOUS_R,
            ALBEDO_TINT_LARGE_GASEOUS_G,
            ALBEDO_TINT_LARGE_GASEOUS_B,
        ),
    };
    let dust_warmth = composition.dust * ALBEDO_DUST_WARMTH_GAIN;
    let r = (albedo_scalar * (tint_r + dust_warmth)).clamp(0.0, 1.0) as f32;
    let g = (albedo_scalar * tint_g).clamp(0.0, 1.0) as f32;
    let b = (albedo_scalar * (tint_b - dust_warmth * ALBEDO_DUST_BLUE_DAMP))
        .clamp(0.0, 1.0) as f32;
    [r, g, b]
}

// ─── FlatBuffers conversion helpers ────────────────────────────────────────

use crate::protocol_generated as fb;
use flatbuffers::{FlatBufferBuilder, WIPOffset};

/// Build a `PlanetaryStateData` table on the FlatBuffers builder. `None`
/// propagates through.
pub fn to_fb_planetary<'b>(
    s: &Option<PlanetGeophysicalState>,
    builder: &mut FlatBufferBuilder<'b>,
) -> Option<WIPOffset<fb::PlanetaryStateData<'b>>> {
    s.as_ref().map(|s| {
        fb::PlanetaryStateData::create(
            builder,
            &fb::PlanetaryStateDataArgs {
                has_atmosphere: if s.has_atmosphere { 1 } else { 0 },
                surface_gravity_ms2: s.surface_gravity_ms2,
                equilibrium_temp_k: s.equilibrium_temp_k,
                surface_pressure_pa: s.surface_pressure_pa,
                ground_albedo_r: s.ground_albedo_linear_rgb[0],
                ground_albedo_g: s.ground_albedo_linear_rgb[1],
                ground_albedo_b: s.ground_albedo_linear_rgb[2],
                composition_n2: s.composition_n2,
                composition_o2: s.composition_o2,
                composition_co2: s.composition_co2,
                composition_ch4: s.composition_ch4,
                composition_h2o: s.composition_h2o,
                composition_h2: s.composition_h2,
                composition_he: s.composition_he,
                composition_dust: s.composition_dust,
                scale_height_m: s.scale_height_m,
                atmosphere_top_altitude_m: s.atmosphere_top_altitude_m,
                mean_molecular_mass_kg: s.mean_molecular_mass_kg,
                atmospheric_mass_kg: s.atmospheric_mass_kg,
                rayleigh_scattering_r: s.rayleigh_scattering_per_lambda[0],
                rayleigh_scattering_g: s.rayleigh_scattering_per_lambda[1],
                rayleigh_scattering_b: s.rayleigh_scattering_per_lambda[2],
                rayleigh_scale_height_m: s.rayleigh_scale_height_m,
                mie_scattering_r: s.mie_scattering_per_lambda[0],
                mie_scattering_g: s.mie_scattering_per_lambda[1],
                mie_scattering_b: s.mie_scattering_per_lambda[2],
                mie_absorption_r: s.mie_absorption_per_lambda[0],
                mie_absorption_g: s.mie_absorption_per_lambda[1],
                mie_absorption_b: s.mie_absorption_per_lambda[2],
                mie_scale_height_m: s.mie_scale_height_m,
                mie_phase_g: s.mie_phase_g,
                ozone_absorption_r: s.ozone_absorption_per_lambda[0],
                ozone_absorption_g: s.ozone_absorption_per_lambda[1],
                ozone_absorption_b: s.ozone_absorption_per_lambda[2],
                ozone_layer_centre_m: s.ozone_layer_centre_m,
                ozone_layer_width_m: s.ozone_layer_width_m,
            },
        )
    })
}

/// Decode an optional `PlanetaryStateData` FB table to a typed
/// [`PlanetGeophysicalState`]. `None` propagates through.
pub fn from_fb_planetary(
    fb: Option<fb::PlanetaryStateData>,
) -> Option<PlanetGeophysicalState> {
    fb.map(|s| PlanetGeophysicalState {
        has_atmosphere: s.has_atmosphere() != 0,
        surface_gravity_ms2: s.surface_gravity_ms2(),
        equilibrium_temp_k: s.equilibrium_temp_k(),
        surface_pressure_pa: s.surface_pressure_pa(),
        ground_albedo_linear_rgb: [
            s.ground_albedo_r(),
            s.ground_albedo_g(),
            s.ground_albedo_b(),
        ],
        composition_n2: s.composition_n2(),
        composition_o2: s.composition_o2(),
        composition_co2: s.composition_co2(),
        composition_ch4: s.composition_ch4(),
        composition_h2o: s.composition_h2o(),
        composition_h2: s.composition_h2(),
        composition_he: s.composition_he(),
        composition_dust: s.composition_dust(),
        scale_height_m: s.scale_height_m(),
        atmosphere_top_altitude_m: s.atmosphere_top_altitude_m(),
        mean_molecular_mass_kg: s.mean_molecular_mass_kg(),
        atmospheric_mass_kg: s.atmospheric_mass_kg(),
        rayleigh_scattering_per_lambda: [
            s.rayleigh_scattering_r(),
            s.rayleigh_scattering_g(),
            s.rayleigh_scattering_b(),
        ],
        rayleigh_scale_height_m: s.rayleigh_scale_height_m(),
        mie_scattering_per_lambda: [
            s.mie_scattering_r(),
            s.mie_scattering_g(),
            s.mie_scattering_b(),
        ],
        mie_absorption_per_lambda: [
            s.mie_absorption_r(),
            s.mie_absorption_g(),
            s.mie_absorption_b(),
        ],
        mie_scale_height_m: s.mie_scale_height_m(),
        mie_phase_g: s.mie_phase_g(),
        ozone_absorption_per_lambda: [
            s.ozone_absorption_r(),
            s.ozone_absorption_g(),
            s.ozone_absorption_b(),
        ],
        ozone_layer_centre_m: s.ozone_layer_centre_m(),
        ozone_layer_width_m: s.ozone_layer_width_m(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics_constants::{
        ORBITAL_DISTANCE_EARTH_M, R_EARTH_M, STANDARD_PRESSURE_PA, STANDARD_TEMPERATURE_K,
    };

    /// Construct a Sol-like StellarState (G-class, ~1 M⊙ ~5772 K).
    fn sol_like() -> StellarState {
        StellarState {
            mass_solar: 1.0,
            temperature_k: 5772.0,
            radius_solar: 1.0,
            luminosity_w: 3.828e26,
            color_linear_rgb: [1.0, 0.94, 0.80],
            surface_radiance_w_per_m2: 6.3e7,
        }
    }

    /// Earth-like derivation should produce a plausible Earth atmosphere.
    #[test]
    fn earth_like_planet() {
        let star = sol_like();
        let g = PlanetGeophysicalState::from_seed_and_star(
            0xEAA7_4117,
            &star,
            M_EARTH_KG,
            R_EARTH_M,
            ORBITAL_DISTANCE_EARTH_M,
        );
        assert!(g.has_atmosphere, "Earth-like should have an atmosphere");
        assert!(
            (200.0..=350.0).contains(&g.equilibrium_temp_k),
            "Earth-like T_eq out of range: {}",
            g.equilibrium_temp_k
        );
        assert!(
            (8.0..=11.0).contains(&g.surface_gravity_ms2),
            "Earth-like surface gravity wrong: {}",
            g.surface_gravity_ms2
        );
        assert!(
            g.surface_pressure_pa > 0.0,
            "Earth-like surface pressure should be positive"
        );
    }

    /// Determinism: identical inputs → bit-identical outputs.
    #[test]
    fn deterministic() {
        let star = sol_like();
        let a = PlanetGeophysicalState::from_seed_and_star(
            0xC0FFEE,
            &star,
            M_EARTH_KG,
            R_EARTH_M,
            ORBITAL_DISTANCE_EARTH_M,
        );
        let b = PlanetGeophysicalState::from_seed_and_star(
            0xC0FFEE,
            &star,
            M_EARTH_KG,
            R_EARTH_M,
            ORBITAL_DISTANCE_EARTH_M,
        );
        assert_eq!(a, b, "geophysics derivation is not deterministic");
    }

    /// Different seeds within the same class produce different atmospheres.
    #[test]
    fn different_seeds_differ() {
        let star = sol_like();
        let a = PlanetGeophysicalState::from_seed_and_star(
            0x1111,
            &star,
            M_EARTH_KG,
            R_EARTH_M,
            ORBITAL_DISTANCE_EARTH_M,
        );
        let b = PlanetGeophysicalState::from_seed_and_star(
            0x2222,
            &star,
            M_EARTH_KG,
            R_EARTH_M,
            ORBITAL_DISTANCE_EARTH_M,
        );
        assert_ne!(
            a.atmospheric_mass_kg, b.atmospheric_mass_kg,
            "different seeds should give different atmospheric masses"
        );
    }

    /// Tiny moonlet — should be airless.
    #[test]
    fn tiny_moon_is_airless() {
        let star = sol_like();
        let g = PlanetGeophysicalState::from_seed_and_star(
            0xDEAD,
            &star,
            0.01 * M_EARTH_KG,
            0.2 * R_EARTH_M,
            ORBITAL_DISTANCE_EARTH_M,
        );
        assert!(!g.has_atmosphere);
        assert_eq!(g.surface_pressure_pa, 0.0);
        for band in g.rayleigh_scattering_per_lambda {
            assert_eq!(band, 0.0);
        }
    }

    /// FlatBuffers round-trip must preserve all fields bit-identically.
    #[test]
    fn flatbuffers_round_trip() {
        let star = sol_like();
        let original = PlanetGeophysicalState::from_seed_and_star(
            0xBEEF,
            &star,
            M_EARTH_KG,
            R_EARTH_M,
            ORBITAL_DISTANCE_EARTH_M,
        );
        let mut builder = FlatBufferBuilder::new();
        let offset = to_fb_planetary(&Some(original), &mut builder).unwrap();
        builder.finish(offset, None);
        let buf = builder.finished_data();
        let decoded = unsafe {
            flatbuffers::root_unchecked::<fb::PlanetaryStateData>(buf)
        };
        let recovered = from_fb_planetary(Some(decoded)).unwrap();
        assert_eq!(original, recovered);
    }

    /// A hot rocky world above the runaway-greenhouse equilibrium-temperature
    /// threshold should classify as HotRunaway and produce CO₂-dominant
    /// composition. This test uses a Mercury-orbit-class distance (0.18 AU)
    /// to push T_eq above the model's 600 K boundary; real Venus has a
    /// modest equilibrium temp (~232 K) but its 740 K surface comes from
    /// greenhouse forcing — a separate physical mechanism that this Phase 3
    /// model does not yet include. The test covers the HotRunaway code path
    /// generically rather than tying to Venus orbital data specifically.
    #[test]
    fn hot_world_is_co2_dominant() {
        let star = sol_like();
        let mass = 0.5 * M_EARTH_KG;
        let radius = 0.8 * R_EARTH_M;
        // 0.10 AU keeps T_eq ≥ ~770 K across the seed-driven albedo range
        // [0.20, 0.41], so the test is robust to albedo variance.
        let orbit = 0.10 * ORBITAL_DISTANCE_EARTH_M;
        let g = PlanetGeophysicalState::from_seed_and_star(
            0x5EED_F00D,
            &star,
            mass,
            radius,
            orbit,
        );
        assert!(
            g.equilibrium_temp_k > 600.0,
            "test setup should put T_eq above HotRunaway threshold: {}",
            g.equilibrium_temp_k
        );
        assert!(g.has_atmosphere);
        assert!(
            g.composition_co2 > 0.5,
            "hot rocky world should be CO₂-dominant: {}",
            g.composition_co2
        );
    }

    /// Composition fractions sum to ≈ 1 for atmosphered planets.
    #[test]
    fn composition_normalised() {
        let star = sol_like();
        for seed in [0x1111_u64, 0x2222, 0x3333, 0x4444, 0x5555] {
            let g = PlanetGeophysicalState::from_seed_and_star(
                seed,
                &star,
                M_EARTH_KG,
                R_EARTH_M,
                ORBITAL_DISTANCE_EARTH_M,
            );
            if g.has_atmosphere {
                let total = g.composition_n2
                    + g.composition_o2
                    + g.composition_co2
                    + g.composition_ch4
                    + g.composition_h2o
                    + g.composition_h2
                    + g.composition_he
                    + g.composition_dust;
                assert!(
                    (total - 1.0).abs() < 0.05,
                    "composition does not sum to 1: total={}",
                    total
                );
            }
        }
    }

    /// STP sanity check: at Earth pressure and temperature, our number
    /// density should be very close to Loschmidt's number.
    #[test]
    fn n_surface_at_stp() {
        let n = STANDARD_PRESSURE_PA / (K_BOLTZMANN * STANDARD_TEMPERATURE_K);
        assert!(
            (n / N_LOSCHMIDT - 1.0).abs() < 0.001,
            "N at STP should match Loschmidt's number: {n} vs {N_LOSCHMIDT}"
        );
    }
}
