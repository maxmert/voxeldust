//! THE SKY'S LAWS (slice 8s, `docs/investigation/2026-09-08/landforms/slice_8s_design.md` §4).
//!
//! The renderer draws a planet's air as a VOLUME around a sphere (Hillaire 2020, as Bevy ships it,
//! with the planet-centre patch). Every number that volume reads is computed HERE, from the body's
//! charter words and the star row, by a published law with a stated calibration body — never typed,
//! never tuned (ruling T9). This module is renderer-free (Tier-A): it returns plain numbers that
//! `vd-client-render` maps onto the engine's structs and nothing else.
//!
//! Example: the home planet's charter states one Earth-like Rayleigh optical depth over an 8 km
//! scale height, so its zenith is blue at noon and red at sunset. The 1 239 Pa Mars-thin body the
//! 8b report found states an optical depth forty times thinner, so the same law gives a near-black
//! zenith with a faint band on the horizon and the stars out by day. Nothing here names either body.

use vd_core::look::{BodyCharter, CHARTER_FLAG_HAS_AIR, CHARTER_FLAG_HAS_SEA};
use vd_core::stellar::{
    AU_M, EARTH_MEAN_SURFACE_K, EARTH_P_SURF_PA, earth_scale_height_m,
    star_radius_from_luminosity_m,
};

/// The wavelength the charter's Rayleigh optical depth is stated at, nm.
pub const RAYLEIGH_REFERENCE_NM: f64 = 550.0;
/// The three wavelengths the engine's RGB sky is sampled at, nm — MEASURED from Bevy's own Earth
/// term (`13.558/5.802 = (680/550)⁴`, `33.1/13.558 = (550/440)⁴`): Bruneton's calibration.
pub const RAYLEIGH_SAMPLE_NM: [f64; 3] = [680.0, 550.0, 440.0];
/// Bevy's (Bruneton's) Earth Rayleigh scattering at 550 nm, m⁻¹ — the calibration the ratio print
/// compares a body's own `τ_vis / H` against. Not a term anybody reads.
pub const EARTH_RAYLEIGH_550_PER_M: f64 = 13.558e-6;
/// WHERE THE AIR IS IGNORED: the shell's top is `R + H · ln 65 536` — the altitude where the density
/// falls below ONE HALF-FLOAT STEP of the surface density (the engine's tables are `Rgba16Float`),
/// so the shell ends by the format's own quantum, never by a typed kilometre count. `16 · ln 2`.
pub const SHELL_TOP_SCALE_HEIGHTS: f64 = 11.090_354_888_959_125;

/// ★ THE AEROSOL PLACEHOLDER (design §4.3, owner 2026-09-18, ledgered): Bevy's Earth aerosol term
/// (Bruneton's calibration) — scaled by the surface number density and the scale height. Its only
/// inputs are charter words; the SOURCE law (dust from aridity, salt from the sea, ash) is 8c/8e/8o's.
pub const EARTH_MIE_SCATTERING_PER_M: f64 = 0.444e-6;
pub const EARTH_MIE_ABSORPTION_PER_M: f64 = 3.996e-6;
pub const EARTH_MIE_ASYMMETRY: f64 = 0.8;
/// The aerosol's scale height on Earth, m (the boundary layer's), scaled by `H / H⊕`.
pub const EARTH_MIE_SCALE_HEIGHT_M: f64 = 1_200.0;

/// ★ OZONE (design §4.4, owner 2026-09-18: PRESENT where the body is earth-like — air AND a sea, the
/// charter's own bits; the liquid-water clause of ruling T9). Bruneton's Earth absorption, scaled by
/// the surface number density. The layer sits at a PRESSURE LEVEL (photochemistry), so its altitude
/// is `H · ln(p_surf / P_PEAK)` — Earth: 8.4 km · ln(101 325 / 2 500) ≈ 31 km against the measured
/// 22–28 km maximum; the calibration is the 25 hPa level.
pub const EARTH_OZONE_ABSORPTION_PER_M: [f64; 3] = [0.650e-6, 1.881e-6, 0.085e-6];
pub const OZONE_PEAK_PRESSURE_PA: f64 = 2_500.0;
/// The ozone tent's full width on Earth, m (Bruneton's profile: ±15 km about the peak), scaled by
/// `H / H⊕`.
pub const EARTH_OZONE_LAYER_WIDTH_M: f64 = 30_000.0;

/// The charter's fixed-point denominators.
const Q12: f64 = 4096.0;

/// How a term's density falls with altitude, in the engine's own parameter `p` (`1` at the surface,
/// `0` at the shell's top).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Falloff {
    /// `e^(−z / (share · shell))`: the share is the scale height over the shell's height.
    Exponential { scale_share: f64 },
    /// A triangular layer: `center` and `width` in `p`.
    Tent { center: f64, width: f64 },
}

/// How a term scatters light by angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    Rayleigh,
    Mie { asymmetry: f64 },
    Isotropic,
}

/// One scattering term of a body's air: per-metre coefficients at the three sample wavelengths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Term {
    pub scattering_per_m: [f64; 3],
    pub absorption_per_m: [f64; 3],
    pub falloff: Falloff,
    pub phase: Phase,
}

/// A body's air, as the volume reads it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyTerms {
    /// The ground's radius, m (the ladder's datum).
    pub bottom_radius_m: f64,
    /// The shell's top, m from the centre.
    pub top_radius_m: f64,
    /// The bond albedo, for the multiple-scattering ground bounce (design §4.6, an interim).
    pub ground_albedo: f64,
    pub rayleigh: Term,
    pub mie: Term,
    pub ozone: Option<Term>,
    /// `τ_vis / H` at 550 nm, m⁻¹ — printed as a ratio against [`EARTH_RAYLEIGH_550_PER_M`].
    pub rayleigh_550_per_m: f64,
}

/// The words the air's laws read, widened once.
#[derive(Clone, Copy, Debug)]
struct AirWords {
    scale_height_m: f64,
    p_surf_pa: f64,
    tau_vis: f64,
    t_surface_k: f64,
}

/// The four air words, all present or none: a charter that states air and lacks a word is refused,
/// like a malformed charter word anywhere else on the client.
fn air_words(c: &BodyCharter) -> Option<AirWords> {
    let scale_height_m = f64::from(c.scale_height_m?);
    let p_surf_pa = f64::from(c.p_surf_pa?);
    let tau_vis = f64::from(c.tau_vis_q12?) / Q12;
    let t_surface_k = f64::from(c.t_surface_mk?) / 1000.0;
    Some(AirWords {
        scale_height_m,
        p_surf_pa,
        tau_vis,
        t_surface_k,
    })
}

/// Rayleigh's law: the coefficient at each sample wavelength from the one at the reference.
fn rayleigh_rgb(beta_reference: f64) -> [f64; 3] {
    RAYLEIGH_SAMPLE_NM.map(|nm| beta_reference * (RAYLEIGH_REFERENCE_NM / nm).powi(4))
}

/// The ozone layer, where the surface pressure stands above the peak's level (a body whose whole
/// air is thinner than the 25 hPa level has no layer to place).
fn ozone_term(w: &AirWords, density_ratio: f64, height_ratio: f64, shell_m: f64) -> Option<Term> {
    let centre_m = w.scale_height_m * (w.p_surf_pa / OZONE_PEAK_PRESSURE_PA).ln();
    if centre_m <= 0.0 {
        return None;
    }
    let width_m = EARTH_OZONE_LAYER_WIDTH_M * height_ratio;
    Some(Term {
        scattering_per_m: [0.0; 3],
        absorption_per_m: EARTH_OZONE_ABSORPTION_PER_M.map(|a| a * density_ratio),
        falloff: Falloff::Tent {
            center: 1.0 - centre_m / shell_m,
            width: width_m / shell_m,
        },
        phase: Phase::Isotropic,
    })
}

/// THE AIR OF ONE BODY from its charter and its ground radius; `None` for a body without air, or
/// with air and a missing word.
#[must_use]
pub fn sky_terms(charter: &BodyCharter, radius_m: f64) -> Option<SkyTerms> {
    if charter.flags & CHARTER_FLAG_HAS_AIR == 0 {
        return None;
    }
    let w = air_words(charter)?;
    let shell_m = w.scale_height_m * SHELL_TOP_SCALE_HEIGHTS;
    let top_radius_m = radius_m + shell_m;
    // Rayleigh, exact by construction: the zenith optical depth equals the charter's own word.
    let rayleigh_550_per_m = w.tau_vis / w.scale_height_m;
    let rayleigh = Term {
        scattering_per_m: rayleigh_rgb(rayleigh_550_per_m),
        absorption_per_m: [0.0; 3],
        falloff: Falloff::Exponential {
            scale_share: w.scale_height_m / shell_m,
        },
        phase: Phase::Rayleigh,
    };
    // The surface number density against Earth's, `(p/T) / (p⊕/T⊕)`, and the scale height against
    // Earth's under the same law: the two ratios the placeholder terms scale by.
    let density_ratio = (w.p_surf_pa / w.t_surface_k) / (EARTH_P_SURF_PA / EARTH_MEAN_SURFACE_K);
    let height_ratio = w.scale_height_m / earth_scale_height_m();
    let mie = Term {
        scattering_per_m: [EARTH_MIE_SCATTERING_PER_M * density_ratio; 3],
        absorption_per_m: [EARTH_MIE_ABSORPTION_PER_M * density_ratio; 3],
        falloff: Falloff::Exponential {
            scale_share: EARTH_MIE_SCALE_HEIGHT_M * height_ratio / shell_m,
        },
        phase: Phase::Mie {
            asymmetry: EARTH_MIE_ASYMMETRY,
        },
    };
    let earth_like = charter.flags & CHARTER_FLAG_HAS_SEA != 0;
    let ozone = if earth_like {
        ozone_term(&w, density_ratio, height_ratio, shell_m)
    } else {
        None
    };
    Some(SkyTerms {
        bottom_radius_m: radius_m,
        top_radius_m,
        ground_albedo: f64::from(charter.bond_albedo_q12) / Q12,
        rayleigh,
        mie,
        ozone,
        rayleigh_550_per_m,
    })
}

/// THE SUN DISC'S ANGULAR DIAMETER, radians, from the star row's luminosity and the body's charter
/// insolation: the distance is `AU · √(L / S)` (the inverse-square law the charter's insolation was
/// stated with), the radius is the star's from its luminosity (`vd_core::stellar`). `None` where the
/// insolation or the luminosity is zero (no star lights this body).
#[must_use]
pub fn sun_disk_angle_rad(luma_lsun: f64, insolation_q12: u32) -> Option<f64> {
    let insolation = f64::from(insolation_q12) / Q12;
    if insolation <= 0.0 || luma_lsun <= 0.0 {
        return None;
    }
    let distance_m = AU_M * (luma_lsun / insolation).sqrt();
    Some(2.0 * star_radius_from_luminosity_m(luma_lsun) / distance_m)
}

/// THE PICK'S KEY (design §7): how large a body's air stands in the eye's sky, `top / distance`.
/// The eye gets the air of the body with the largest key; a body without air is never a candidate.
#[must_use]
pub fn shell_angle(top_radius_m: f64, distance_m: f64) -> f64 {
    top_radius_m / distance_m
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::stellar::{R_SUN_M, scale_height_m};

    /// A charter at Earth's own words: the scale height by the shared law, Earth's pressure, the
    /// Rayleigh depth the charter author calibrates on (0.0973), the thermostat's set point.
    fn earth_like() -> BodyCharter {
        let h = scale_height_m(EARTH_MEAN_SURFACE_K, 28.96, 9.81).round() as u32;
        BodyCharter {
            gravity_mm_s2: 9_810,
            bulk_density_kgm3: 5_514,
            escape_velocity_mps: 11_186,
            insolation_q12: 4096,
            t_eq_mk: 255_000,
            t_surface_mk: Some(288_000),
            bond_albedo_q12: 1_229,
            mu_q8: Some(7_414),
            scale_height_m: Some(h),
            p_surf_pa: Some(101_325),
            tau_vis_q12: Some(399),
            tau_ir_q12: Some(3_523),
            day_s: Some(86_164),
            obliquity_cos_q1024: Some(939),
            water_km3: Some(1_386_000_000),
            sea_offset_mm: Some(0),
            elastic_thickness_m: Some(35_000),
            ecc_q16: 1_094,
            year_s: 31_558_150,
            flags: CHARTER_FLAG_HAS_AIR | CHARTER_FLAG_HAS_SEA | 2,
        }
    }

    const R_M: f64 = 6.371e6;

    /// The exponential share of a falloff, or `None` for a layer — a `match` whose both arms the
    /// tests walk (a `let … else { panic!() }` leaves an arm nobody walks, HR5).
    fn exp_share(f: Falloff) -> Option<f64> {
        match f {
            Falloff::Exponential { scale_share } => Some(scale_share),
            Falloff::Tent { .. } => None,
        }
    }

    /// A layer's `(center, width)`, or `None` for an exponential falloff.
    fn tent_of(f: Falloff) -> Option<(f64, f64)> {
        match f {
            Falloff::Tent { center, width } => Some((center, width)),
            Falloff::Exponential { .. } => None,
        }
    }

    #[test]
    fn an_airless_body_has_no_sky() {
        let mut c = earth_like();
        c.flags &= !CHARTER_FLAG_HAS_AIR;
        assert_eq!(sky_terms(&c, R_M), None);
    }

    #[test]
    fn a_charter_with_air_and_a_missing_word_is_refused() {
        let mut no_h = earth_like();
        no_h.scale_height_m = None;
        assert_eq!(sky_terms(&no_h, R_M), None);
        let mut no_p = earth_like();
        no_p.p_surf_pa = None;
        assert_eq!(sky_terms(&no_p, R_M), None);
        let mut no_tau = earth_like();
        no_tau.tau_vis_q12 = None;
        assert_eq!(sky_terms(&no_tau, R_M), None);
        let mut no_t = earth_like();
        no_t.t_surface_mk = None;
        assert_eq!(sky_terms(&no_t, R_M), None);
    }

    #[test]
    fn rayleigh_is_the_charters_depth_over_its_scale_height_and_follows_the_fourth_power() {
        let c = earth_like();
        let s = sky_terms(&c, R_M).expect("air");
        let h = f64::from(c.scale_height_m.expect("h"));
        let tau = f64::from(c.tau_vis_q12.expect("tau")) / 4096.0;
        assert!((s.rayleigh_550_per_m - tau / h).abs() < 1e-18);
        // Green IS the reference; red and blue follow (550/λ)⁴.
        let [r, g, b] = s.rayleigh.scattering_per_m;
        assert!((g - s.rayleigh_550_per_m).abs() < 1e-18);
        assert!((r / g - (550.0f64 / 680.0).powi(4)).abs() < 1e-12);
        assert!((b / g - (550.0f64 / 440.0).powi(4)).abs() < 1e-12);
        assert_eq!(s.rayleigh.absorption_per_m, [0.0; 3]);
        assert_eq!(s.rayleigh.phase, Phase::Rayleigh);
        // At Earth's words the derived coefficient stands within a few percent of Bruneton's.
        let ratio = s.rayleigh_550_per_m / EARTH_RAYLEIGH_550_PER_M;
        assert!((0.8..1.2).contains(&ratio), "{ratio}");
    }

    #[test]
    fn the_shell_ends_at_the_half_float_quantum_and_the_falloff_is_the_scale_height() {
        let c = earth_like();
        let s = sky_terms(&c, R_M).expect("air");
        let h = f64::from(c.scale_height_m.expect("h"));
        let shell = s.top_radius_m - s.bottom_radius_m;
        assert!((shell - h * 16.0 * std::f64::consts::LN_2).abs() < 1e-6);
        assert_eq!(s.bottom_radius_m, R_M);
        let scale_share = exp_share(s.rayleigh.falloff).expect("rayleigh falls off exponentially");
        assert!((scale_share * shell - h).abs() < 1e-9);
        assert_eq!(tent_of(s.rayleigh.falloff), None);
    }

    #[test]
    fn at_earths_words_the_placeholder_terms_are_bruneton_s_exactly() {
        let c = earth_like();
        let s = sky_terms(&c, R_M).expect("air");
        let h = f64::from(c.scale_height_m.expect("h"));
        let shell = s.top_radius_m - s.bottom_radius_m;
        // The density ratio is 1 at Earth's pressure and temperature; the height ratio is 1 up to
        // the charter word's rounding to whole metres.
        let height_ratio = h / earth_scale_height_m();
        for ch in 0..3 {
            assert!((s.mie.scattering_per_m[ch] - EARTH_MIE_SCATTERING_PER_M).abs() < 1e-18);
            assert!((s.mie.absorption_per_m[ch] - EARTH_MIE_ABSORPTION_PER_M).abs() < 1e-18);
        }
        let scale_share = exp_share(s.mie.falloff).expect("mie falls off exponentially");
        assert!((scale_share * shell - EARTH_MIE_SCALE_HEIGHT_M * height_ratio).abs() < 1e-6);
        assert_eq!(
            s.mie.phase,
            Phase::Mie {
                asymmetry: EARTH_MIE_ASYMMETRY
            }
        );
        let ozone = s.ozone.expect("an earth-like body has an ozone layer");
        assert_eq!(ozone.scattering_per_m, [0.0; 3]);
        for (got, earth) in ozone
            .absorption_per_m
            .iter()
            .zip(EARTH_OZONE_ABSORPTION_PER_M)
        {
            assert!((got - earth).abs() < 1e-18);
        }
        let (center, width) = tent_of(ozone.falloff).expect("ozone is a layer");
        assert_eq!(exp_share(ozone.falloff), None);
        // The layer's centre: H · ln(p / 25 hPa) ≈ 31 km on Earth; the width 30 km.
        let centre_m = (1.0 - center) * shell;
        assert!(
            (centre_m - h * (101_325.0f64 / 2_500.0).ln()).abs() < 1e-6,
            "{centre_m}"
        );
        assert!((centre_m - 31_200.0).abs() < 600.0, "{centre_m}");
        assert!((width * shell - EARTH_OZONE_LAYER_WIDTH_M * height_ratio).abs() < 1e-6);
        assert_eq!(ozone.phase, Phase::Isotropic);
        assert!((s.ground_albedo - 1_229.0 / 4096.0).abs() < 1e-12);
    }

    #[test]
    fn a_body_without_a_sea_has_no_ozone_and_a_thin_air_has_no_layer_to_place() {
        let mut dry = earth_like();
        dry.flags &= !CHARTER_FLAG_HAS_SEA;
        assert_eq!(sky_terms(&dry, R_M).expect("air").ozone, None);
        // A sea under an air thinner than the 25 hPa level: the layer's centre would be underground.
        let mut thin = earth_like();
        thin.p_surf_pa = Some(1_239);
        assert_eq!(sky_terms(&thin, R_M).expect("air").ozone, None);
    }

    #[test]
    fn a_thinner_air_scales_the_placeholder_terms_by_its_surface_density() {
        let mut thin = earth_like();
        thin.p_surf_pa = Some(50_662);
        let s = sky_terms(&thin, R_M).expect("air");
        let ratio = s.mie.scattering_per_m[1] / EARTH_MIE_SCATTERING_PER_M;
        assert!((ratio - 50_662.0 / 101_325.0).abs() < 1e-12, "{ratio}");
    }

    #[test]
    fn the_sun_disc_is_half_a_degree_at_one_au_and_none_without_a_star() {
        let angle = sun_disk_angle_rad(1.0, 4096).expect("lit");
        // 2 · 1.06 R☉ / AU (the fit's 1.06 at one solar mass).
        assert!((angle - 2.0 * 1.06 * R_SUN_M / AU_M).abs() < 1e-15);
        let deg = angle.to_degrees();
        assert!((deg - 0.565).abs() < 0.005, "{deg}");
        assert_eq!(sun_disk_angle_rad(1.0, 0), None);
        assert_eq!(sun_disk_angle_rad(0.0, 4096), None);
        // Four times the luminosity at the same insolation stands twice as far: the disc is the
        // star's larger radius over twice the distance.
        let far = sun_disk_angle_rad(4.0, 4096).expect("lit");
        let expect = 2.0 * star_radius_from_luminosity_m(4.0) / (2.0 * AU_M);
        assert!((far - expect).abs() < 1e-15);
    }

    #[test]
    fn the_pick_key_is_the_shell_over_the_distance() {
        assert!((shell_angle(6.45e6, 1.29e7) - 0.5).abs() < 1e-12);
    }
}
