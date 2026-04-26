//! Stellar physical state derivation. Pure-functional; called exclusively by
//! server-shards (galaxy-shard at galaxy generation, plus echo by system /
//! planet / ship shards in their broadcasts). The client links this module
//! through the shared `core` crate but never invokes the `from_*` constructor
//! at runtime — `client/tests/no_client_seed_derivation.rs` enforces this.
//!
//! All state derives from a single seed-driven input: a continuous mass
//! within the spectral class's astrophysical mass range. Everything else
//! follows from main-sequence stellar physics:
//!
//! 1. Mass within the class's astrophysical range, picked from a
//!    `system_seed`-derived sub-seed.
//! 2. Radius from `ms_radius_solar_from_mass_solar(M)` (Demircan & Kahraman
//!    1991 mass-radius relation).
//! 3. Luminosity from `ms_luminosity_solar_from_mass_solar(M)` (piecewise M-L
//!    relation; standard stellar-structure texts).
//! 4. Effective temperature from Stefan-Boltzmann law: `L = 4π · R² · σ · T⁴`,
//!    inverted on `T`. Self-consistent with the derived `L` and `R`.
//! 5. Linear-sRGB rendering colour from `crate::blackbody::temperature_to_linear_rgb(T)`.
//! 6. Surface radiance for HDR emissive sun-disk rendering: `σ · T⁴`.
//!
//! Every formula references named constants from `crate::physics_constants`;
//! `client/tests/no_magic_numbers.rs` enforces this.

use serde::{Deserialize, Serialize};

use crate::galaxy::StarClass;
use crate::physics_constants::{
    L_SUN_W, PI, R_SUN_M, SIGMA_STEFAN_BOLTZMANN, ms_luminosity_solar_from_mass_solar,
    ms_radius_solar_from_mass_solar,
};
use crate::seed::{derive_seed, seed_to_f64};

/// Sub-seed-index reserved for stellar-mass within-class variance. Stable
/// across versions so seed → output remains reproducible.
mod sub_seed {
    pub const STELLAR_MASS_WITHIN_CLASS: u32 = 1_001;
}

/// Physical state of a single star, derived deterministically server-side.
/// Broadcast to clients in `StarCatalogEntry.stellar` and
/// `CelestialBodySnapshot.stellar` (body_id == 0). The client renders from
/// these values without re-deriving anything.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StellarState {
    /// Mass in solar masses.
    pub mass_solar: f32,
    /// Effective surface temperature in Kelvin.
    pub temperature_k: f32,
    /// Radius in solar radii.
    pub radius_solar: f32,
    /// Total radiative power in watts (the absolute luminosity).
    pub luminosity_w: f32,
    /// Pre-computed linear sRGB colour for direct use as `DirectionalLight.color`
    /// and as the base colour of the sun-disk emissive material. Components in
    /// `[0, 1]`. Computed by `crate::blackbody` from `temperature_k`.
    pub color_linear_rgb: [f32; 3],
    /// Surface radiance `σ · T⁴` in W/m². Used to compute the HDR emissive
    /// value of the sun-disk billboard / icosphere so that bloom + tonemap
    /// produce a believable star intensity. Real-world reference: the Sun's
    /// surface radiance is ~6.3×10⁷ W/m².
    pub surface_radiance_w_per_m2: f32,
}

impl StellarState {
    /// Derive a star's physical state from its spectral class and system seed.
    ///
    /// Internally:
    /// 1. Mass is sampled continuously within the class's astrophysical range
    ///    (variance from a stable sub-seed of the system seed).
    /// 2. Luminosity, radius, temperature, colour, and surface radiance all
    ///    flow from the mass via main-sequence physics in `physics_constants`.
    pub fn from_class_and_seed(class: StarClass, system_seed: u64) -> Self {
        // Step 1: pick a continuous mass within the class's astrophysical range.
        let (m_low, m_high) = class.mass_range_solar();
        let mass_subseed = derive_seed(system_seed, sub_seed::STELLAR_MASS_WITHIN_CLASS);
        let t = seed_to_f64(mass_subseed); // [0, 1)
        let mass_solar_f64: f64 = m_low + (m_high - m_low) * t;

        // Step 2: radius from the mass-radius relation.
        let radius_solar_f64: f64 = ms_radius_solar_from_mass_solar(mass_solar_f64);

        // Step 3: luminosity from the piecewise mass-luminosity relation.
        let luminosity_solar_f64: f64 = ms_luminosity_solar_from_mass_solar(mass_solar_f64);
        let luminosity_w_f64: f64 = luminosity_solar_f64 * L_SUN_W;

        // Step 4: temperature from L = 4π · R² · σ · T⁴, inverted.
        // T = (L / (4π · R² · σ))^(1/4).
        let radius_m_f64: f64 = radius_solar_f64 * R_SUN_M;
        let denom: f64 = 4.0 * PI * radius_m_f64 * radius_m_f64 * SIGMA_STEFAN_BOLTZMANN;
        let temperature_k_f64: f64 = (luminosity_w_f64 / denom).powf(1.0 / 4.0);

        // Step 5: rendering colour from blackbody temperature.
        let color_linear_rgb: [f32; 3] =
            crate::blackbody::temperature_to_linear_rgb(temperature_k_f64);

        // Step 6: surface radiance for HDR emissive rendering.
        let surface_radiance_w_per_m2_f64: f64 =
            SIGMA_STEFAN_BOLTZMANN * temperature_k_f64.powi(4);

        StellarState {
            mass_solar: mass_solar_f64 as f32,
            temperature_k: temperature_k_f64 as f32,
            radius_solar: radius_solar_f64 as f32,
            luminosity_w: luminosity_w_f64 as f32,
            color_linear_rgb,
            surface_radiance_w_per_m2: surface_radiance_w_per_m2_f64 as f32,
        }
    }
}

// ─── FlatBuffers conversion helpers ────────────────────────────────────────
//
// Used by `client_message::*` and `shard_message::*` to round-trip
// `StellarState` through the wire format. The typed value lives in the Rust
// data structs (`CelestialBodyData.stellar`, `StarCatalogEntryData.stellar`,
// `CelestialBodySnapshotData.stellar`); the on-the-wire form is the
// `StellarStateData` table generated from `protocol/voxeldust.fbs`.

use crate::protocol_generated as fb;
use flatbuffers::{FlatBufferBuilder, WIPOffset};

/// Build a `StellarStateData` table on the FlatBuffers builder, returning the
/// offset suitable for use as a child of a parent table's `stellar` field.
/// `None` propagates through.
pub fn to_fb_stellar<'b>(
    s: &Option<StellarState>,
    builder: &mut FlatBufferBuilder<'b>,
) -> Option<WIPOffset<fb::StellarStateData<'b>>> {
    s.as_ref().map(|s| {
        fb::StellarStateData::create(
            builder,
            &fb::StellarStateDataArgs {
                mass_solar: s.mass_solar,
                temperature_k: s.temperature_k,
                radius_solar: s.radius_solar,
                luminosity_w: s.luminosity_w,
                color_linear_r: s.color_linear_rgb[0],
                color_linear_g: s.color_linear_rgb[1],
                color_linear_b: s.color_linear_rgb[2],
                surface_radiance_w_per_m2: s.surface_radiance_w_per_m2,
            },
        )
    })
}

/// Decode an optional `StellarStateData` FlatBuffers table to a typed
/// `StellarState`. `None` propagates through.
pub fn from_fb_stellar(fb: Option<fb::StellarStateData>) -> Option<StellarState> {
    fb.map(|s| StellarState {
        mass_solar: s.mass_solar(),
        temperature_k: s.temperature_k(),
        radius_solar: s.radius_solar(),
        luminosity_w: s.luminosity_w(),
        color_linear_rgb: [s.color_linear_r(), s.color_linear_g(), s.color_linear_b()],
        surface_radiance_w_per_m2: s.surface_radiance_w_per_m2(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity-check the Sol-like derivation: G-class with a plausible seed.
    /// Expect mass close to 1 M☉, T close to T_SUN, radius close to 1 R☉.
    #[test]
    fn sol_like_g_class() {
        let s = StellarState::from_class_and_seed(StarClass::G, 0xC0FFEE);
        assert!(
            (0.8..1.04).contains(&s.mass_solar),
            "G-class mass out of range: {}",
            s.mass_solar
        );
        // T_eff for G-class: 5 200 – 6 000 K.
        assert!(
            (4_500.0..6_500.0).contains(&s.temperature_k),
            "G-class T_eff implausible: {}",
            s.temperature_k
        );
        // Surface radiance for Sol: 6.3e7 W/m². Allow ±50 %.
        assert!(
            s.surface_radiance_w_per_m2 > 3.0e7
                && s.surface_radiance_w_per_m2 < 1.0e8,
            "Sol-like surface radiance out of range: {}",
            s.surface_radiance_w_per_m2
        );
        // Color components must be in [0, 1].
        for c in s.color_linear_rgb {
            assert!(
                (0.0..=1.0).contains(&c),
                "color component out of range: {}",
                c
            );
        }
    }

    /// O-class hot blue giant — temperature should be high (>20 000 K).
    #[test]
    fn o_class_blue() {
        let s = StellarState::from_class_and_seed(StarClass::O, 0xDEAD_BEEF);
        assert!(
            s.temperature_k > 20_000.0,
            "O-class T should exceed 20 000 K, got {}",
            s.temperature_k
        );
        // O-class: blue should dominate red.
        assert!(
            s.color_linear_rgb[2] >= s.color_linear_rgb[0],
            "O-class colour not blue-dominant: {:?}",
            s.color_linear_rgb
        );
    }

    /// M-class red dwarf — temperature should be low (<4 000 K).
    #[test]
    fn m_class_red() {
        let s = StellarState::from_class_and_seed(StarClass::M, 0xFEEDFACE);
        assert!(
            s.temperature_k < 4_000.0,
            "M-class T should be below 4 000 K, got {}",
            s.temperature_k
        );
        assert!(
            s.color_linear_rgb[0] >= s.color_linear_rgb[2],
            "M-class colour not red-dominant: {:?}",
            s.color_linear_rgb
        );
    }

    /// Determinism: identical inputs produce bit-identical outputs.
    #[test]
    fn deterministic() {
        let a = StellarState::from_class_and_seed(StarClass::G, 0xABCD_1234);
        let b = StellarState::from_class_and_seed(StarClass::G, 0xABCD_1234);
        assert_eq!(a, b);
    }

    /// Different seeds within the same class produce different masses (and
    /// therefore different L / T / colour).
    #[test]
    fn different_seeds_different_mass() {
        let a = StellarState::from_class_and_seed(StarClass::G, 0x1111);
        let b = StellarState::from_class_and_seed(StarClass::G, 0x2222);
        assert_ne!(a.mass_solar, b.mass_solar);
    }
}
