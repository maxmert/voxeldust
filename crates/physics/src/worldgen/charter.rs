//! ★ THE CHARTER'S AUTHOR (the landform arc, slice 8b stage 2; ruling V13 L12, crossing A1 approved;
//! `docs/investigation/2026-09-08/landforms/slice_8b_design.md` §2.0, §4.1 and §4.2).
//!
//! Owns: THE ONE FLOAT→INTEGER DOOR through which a body's physical facts leave the census and
//! become the whole numbers both hosts read (SL10), and the read of a body's own subtree that
//! gathers them.
//!
//! Does NOT own: any draw of its own. Every number here comes from a row the census already draws —
//! the taxonomy row (`BodyTaxon`), the illuminating star's photometrics, the orbit the generator
//! placed the body on. The words the arc's later stages draw — the spin, the tilt, the two optical
//! depths, the surface pressure, the surface temperature, the elastic thickness, the water inventory
//! and the sea — are stated ABSENT here, never as a zero somebody could read as a fact.
//!
//! **★ A2 IS UNSPENT, AND THIS FILE IS WHY (§4.2).** Ruling V13 approved a lane carrying the orbit
//! half from a star system down to its planet. MEASURED: it is not needed. A shard boots its own
//! subtree AND its lineage (`shard_boot_world`), so a planet's shard already holds its parent chain
//! up to its star system, its own mass and its own orbit. It derives its charter LOCALLY and nothing
//! crosses a realm boundary — SL6's own instruction, *"find the local formulation first"*. The
//! approval stays in hand for the day a body's facts stop being a pure function of the seed.
//!
//! **THE RULE ON ABSENCE.** A shard that cannot derive its charter states NO surface (§4.2 rule 2).
//! It does not guess and it does not default. So [`body_charter_in_subtree`] answers `None` for a
//! body the forest does not name, a body with no taxonomy row (a star, a station, a built hull), a
//! chain that names no star, a body on no orbit, or any fact that will not fit its stated unit.
//!
//! **Example.** The home planet's own shard boots, reads its own row out of the subtree it already
//! built, and states `gravity_mm_s2 = 9 818` — whole millimetres per second squared. Every client
//! that sees the home planet then holds that same whole number.

use super::{GeneratedBody, UniverseConfig, body_facts_in_forest, orbital_of, realm_subtree};
use crate::celestial::OrbitalElements;
use crate::taxonomy::{
    AU_M, FROST_COEFF_AU, KOPPARAPU_FLUX_CONSERVATIVE, M_SUN_KG, SYSTEM_AGE_GYR,
    escape_velocity_mps, frost_line_radius_au, surface_gravity_mps2,
};
use core::f64::consts::TAU;
use vd_core::look::{
    BodyCharter, CHARTER_FLAG_HAS_AIR, CHARTER_FLAG_SOLID_SURFACE, CHARTER_FLAG_TIDALLY_LOCKED,
    CHARTER_STAR_CLASS_SHIFT,
};
use vd_core::pose::RealmId;
use vd_core::rng::{SplitMix64, child_seed};

/// ★ THE DOOR (§2.0): a full-precision number becomes ONE whole number in a stated unit, once.
///
/// The `libm` arithmetic happens above this line; below it nothing is ever a float again. A value
/// that is not finite, is negative, or does not fit the unit is REFUSED — the charter is then not
/// derivable and the realm states no surface, which is the honest answer and not a clamp.
///
/// Example: the home planet's gravity of 9.818 m/s² at a scale of 1 000 becomes 9 818 mm/s².
#[must_use]
pub fn quantise_u32(value: f64, scale: f64) -> Option<u32> {
    let scaled = (value * scale).floor();
    if !scaled.is_finite() {
        return None;
    }
    if scaled < 0.0 {
        return None;
    }
    if scaled > f64::from(u32::MAX) {
        return None;
    }
    // The three refusals above leave a floored value inside [0, u32::MAX], so the cast is exact.
    Some(scaled as u32)
}

/// [`quantise_u32`]'s wide twin — for a number no 32-bit word holds (an orbital period in seconds).
#[must_use]
pub fn quantise_u64(value: f64, scale: f64) -> Option<u64> {
    let scaled = (value * scale).floor();
    if !scaled.is_finite() {
        return None;
    }
    if scaled < 0.0 {
        return None;
    }
    if scaled > U64_MAX_AS_F64 {
        return None;
    }
    Some(scaled as u64)
}

/// `u64::MAX` as the nearest `f64` — the bound [`quantise_u64`] compares against, stated once so the
/// comparison is not a literal nobody can check.
const U64_MAX_AS_F64: f64 = 18_446_744_073_709_551_615.0;

/// The bulk density `M / ((4/3)πR³)` in kg/m³ — the census stores the mass and the radius and never
/// this, so it is computed on the spot, exactly as `home_body_facts` computes it.
#[must_use]
pub fn bulk_density_kgm3(mass_kg: f64, radius_m: f64) -> f64 {
    let volume = 4.0 / 3.0 * core::f64::consts::PI * radius_m * radius_m * radius_m;
    mass_kg / volume
}

/// ★ THE CHARTER OF ONE BODY, read from the subtree a shard already boots (§4.2 rule 1).
///
/// `None` when the body's facts are not all in hand; the caller then states no surface.
#[must_use]
pub fn body_charter_in_subtree(
    seed_universe: u64,
    config: &UniverseConfig,
    held: &std::collections::BTreeSet<RealmId>,
    lineage: &std::collections::BTreeSet<RealmId>,
    body: RealmId,
) -> Option<BodyCharter> {
    charter_in_forest(&realm_subtree(seed_universe, config, held, lineage), body)
}

/// The arithmetic of [`body_charter_in_subtree`] over a forest already in hand — split from the
/// generate-and-read shell so its two refusals are reachable from a unit test on a hand forest. THE
/// world drives the first (a star system carries no taxonomy row); the second — a body with a
/// taxonomy row that sits on no orbit — the world never produces, and a branch nothing drives is a
/// branch nothing checks.
pub(crate) fn charter_in_forest(bodies: &[GeneratedBody], body: RealmId) -> Option<BodyCharter> {
    let facts = body_facts_in_forest(bodies, body)?;
    let orbit = bodies
        .iter()
        .find(|b| b.realm == body)
        .and_then(|b| orbital_of(b.placement))?;
    let seed = body_seed(body)?;
    let moons = companions_in_forest(bodies, body);
    let formation_sma_m = formation_sma_in_forest(bodies, body, &orbit)?;
    charter_of(&facts, &orbit, seed, &moons, formation_sma_m)
}

/// Where the body FORMED, as the water model reads it (§2.4.1): a planet in its own orbit; a moon
/// in its planet's sub-disc, so its formation zone is ITS PLANET'S orbit about the star. A moon whose
/// planet sits on no orbit is a shape the world never produces; it states no charter.
fn formation_sma_in_forest(
    bodies: &[GeneratedBody],
    body: RealmId,
    own_orbit: &OrbitalElements,
) -> Option<f64> {
    let parent = bodies
        .iter()
        .find(|b| b.realm == body)
        .and_then(|b| b.parent)?;
    match parent {
        RealmId::Planet(_) => bodies
            .iter()
            .find(|b| b.realm == parent)
            .and_then(|b| orbital_of(b.placement))
            .map(|o| o.sma),
        _ => Some(own_orbit.sma),
    }
}

/// A body's own seed — the number its realm id carries. A planet and a moon are both `Planet`
/// realms (the owner, 2026-08: a moon is a planet), and nothing else the forest names carries a
/// taxonomy row, so any other kind states no charter.
fn body_seed(body: RealmId) -> Option<u64> {
    match body {
        RealmId::Planet(seed) => Some(seed),
        _ => None,
    }
}

/// One moon as the obliquity's torque ratio reads it (§2.3): its mass, and the semi-major axis of
/// its orbit about the body whose tilt it steadies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Companion {
    /// The moon's mass in kg (the census's own row).
    pub mass_kg: f64,
    /// The moon's semi-major axis about its planet in metres (the orbit the generator placed it on).
    pub sma_m: f64,
}

/// The moons of `body` as the forest holds them. A child with no taxonomy row or on no orbit is
/// not a moon the law can read and is left out — THE world produces neither, and a unit test on a
/// hand forest drives both.
pub(crate) fn companions_in_forest(bodies: &[GeneratedBody], body: RealmId) -> Vec<Companion> {
    bodies
        .iter()
        .filter(|b| b.parent == Some(body))
        .filter_map(|b| {
            let taxon = b.taxon?;
            let orbit = orbital_of(b.placement)?;
            Some(Companion {
                mass_kg: taxon.mass_kg,
                sma_m: orbit.sma,
            })
        })
        .collect()
}

/// How many words of the record are plain `u32` quantisations of one fact.
const PLAIN_WORDS: usize = 7;

/// ★ THE TWO WORDS THE RELIEF LAW READS (the landform arc, slice 8b stage 3), from a body's mass
/// and radius: the surface gravity in whole mm/s² and the bulk density in whole kg/m³.
///
/// ONE SOURCE (HR3): [`charter_of`] splices these two answers straight into its own refusal, so the
/// charter a realm STATES and the pair a measurement reads for the same body can never be two
/// different numbers. Each answer is `None` where the fact will not fit its unit, exactly as
/// [`quantise_u32`] states it; the caller decides what an absent pair means.
///
/// **Example.** The home planet's 5.972 × 10²⁴ kg at 6 370.7 km gives `[Some(9 818), Some(5 513)]`,
/// and the relief law then caps its mountains at 8 840 m.
#[must_use]
pub fn relief_words(mass_kg: f64, radius_m: f64) -> [Option<u32>; 2] {
    [
        quantise_u32(surface_gravity_mps2(mass_kg, radius_m), 1_000.0),
        quantise_u32(bulk_density_kgm3(mass_kg, radius_m), 1.0),
    ]
}

// ---- THE DRAWS OF STAGE 4 (`slice_8b_design.md` §2.2, §2.3, §2.5, §2.6) ---------------------
//
// Every draw below reads ONE new seed stream, salted off the body's own seed, so not one draw the
// forest already makes moves (the same rule the terrain's own salts follow). THE ORDER IS FROZEN:
// the day, the tilt's family, the tilt — every unit is taken whether the law uses it or not, so a
// body that is tidally locked still consumes its day's unit and the draws behind it stand where
// they stood. ★ THE PRESSURE IS NOT A DRAW (ruling T9): it is computed from the mass, the gravity
// and the radius. The spin and the tilt stay draws because they are HISTORY the facts cannot
// recover (a body's rotation and its tilt come from its formation and its impacts), and the law
// then bounds and damps them.

/// The charter's own draw stream, beside the forest's salts ("CHARTER").
const CHARTER_SALT: u64 = 0x0043_4841_5254_4552;
/// THE SPIN's band, log-uniform, anchored on the solar system's rocky bodies
/// (`05_climate_biomes_weather.md` §3.3 B): six hours to a hundred hours.
const DAY_MIN_S: f64 = 6.0 * 3_600.0;
const DAY_MAX_S: f64 = 100.0 * 3_600.0;
/// THE TIDAL LOCK's calibration body: Mercury, despun at 0.387 AU around the Sun in 4.5 Gyr
/// (§2.2). The despinning time scales as `a⁶`, so the locking distance is a sharp threshold and
/// the law is `a_lock = 0.387 AU · M^(1/3) · (t / 4.5 Gyr)^(1/6)` with `M` in solar masses — a
/// CALIBRATION ON ONE BODY, exactly as the relief law's yield stress is calibrated on Everest.
const LOCK_CALIBRATION_SMA_AU: f64 = 0.387;
const LOCK_CALIBRATION_AGE_GYR: f64 = 4.5;
/// THE TILT's prior (`05` §3.3 A; ★ ASSUMED per the design's ask 4 (a), pending the owner): nine
/// bodies in ten draw `cos ε` uniform over `[0.2, 1.0]`, the tenth over `[−1.0, 0.2]` — a
/// Uranus-like curiosity. The nine-to-one split is a TASTE, not a physics result, and it is written
/// here as the owner's number to move.
const OBLIQUITY_COMMON_SHARE: f64 = 0.9;
const OBLIQUITY_COS_SPLIT: f64 = 0.2;
/// The common band's own median, the point the damping pulls toward.
const OBLIQUITY_COS_MEDIAN: f64 = 0.6;
/// ★ THE SURFACE PRESSURE IS DERIVED, NEVER DRAWN (ruling T9, 2026-09-18: *"we don't randomly
/// generate any physic numbers"*; the design's ask 6 band is REFUSED). A rocky body's air is baked
/// out of its rock, so the mass of air it outgasses scales with its mass — ONE calibration body,
/// Earth; the pressure is that air's weight over the surface, `p = g · M_air / (4πR²)`; the
/// census's own shoreline verdict says whether the air is kept. So
/// `p = p⊕ · (M/M⊕) · (g/g⊕) · (R⊕/R)²`, and a body that is Earth's twin in mass, gravity and
/// radius reads one bar BY CONSTRUCTION. The scatter of real planets around the law (Venus, Mars)
/// is NOT modelled; if it is ever wanted it is a NAMED modulation around this value (T9).
const EARTH_MASS_KG: f64 = 5.972e24;
/// THE RAYLEIGH DEPTH at 550 nm, calibrated on Earth (Bodhaine et al. 1999: 0.0973 at one bar,
/// 9.81 m/s², a mean molecular weight of 28.96 u): the column is `p / (g · μ)`.
const EARTH_RAYLEIGH_550: f64 = 0.0973;
use vd_core::stellar::EARTH_P_SURF_PA;
const EARTH_G_MPS2: f64 = 9.81;
const EARTH_MU: f64 = 28.96;
/// THE GREY GREENHOUSE calibrated on Earth: `T_s = T_eq · (1 + ¾·τ_ir)^(1/4)`, and Earth's 288 K
/// over its 254.3 K equilibrium gives `τ_ir⊕ = 0.860`.
const EARTH_TAU_IR: f64 = 0.860;
/// THE CARBONATE–SILICATE THERMOSTAT's set point (Walker, Hays & Kasting 1981; ★ ASSUMED per the
/// design's ask 2 (b), pending the owner): inside the conservative habitable band the CO₂ column
/// rises on a cold world and falls on a warm one until the surface stands at Earth's own mean —
/// a CALIBRATION ON ONE BODY. Outside the band the thermostat cannot hold and the greenhouse is
/// Earth's own (the maximum-greenhouse floor is not modelled: UNMEASURED).
const THERMOSTAT_SET_K: f64 = vd_core::stellar::EARTH_MEAN_SURFACE_K;
/// THE ELASTIC THICKNESS (Watts 2001): `T_e` tracks the depth to a fixed isotherm, so it falls as
/// the surface heat flow rises; radiogenic heat scales with `ρ·R` and decays with age. Earth's
/// 35 km at its own `ρ·R` and 4.54 Gyr is the reference; the decay is a rational `1 / (1 + t/τ)`
/// (`exp` is banned inside the fence), with `τ = 4.5 Gyr` — the value that reproduces the
/// design's own `q/q⊕ = 0.9516` at 5.0 against 4.54 Gyr.
const TE_REF_M: f64 = 35_000.0;
const EARTH_RHO_KGM3: f64 = 5_514.0;
const EARTH_RADIUS_M: f64 = 6.371e6;
const EARTH_AGE_GYR: f64 = 4.54;
const RADIOGENIC_TAU_GYR: f64 = 4.5;

// ---- THE WATER INVENTORY (stage 5, `slice_8b_design.md` §2.4; ruling T6 ask 2: "based on the
// physical laws or some models") -----------------------------------------------------------------
//
// THE MODEL, and every law in it is either already in the code or a published relation with one
// calibration body. (1) THE FORMATION ZONE against the star's snow line — the code's own
// `frost_line_radius_au` (Hayashi 1981, `2.7·√L`): inside it a body accretes dry rock, outside it
// solar-composition condensates that are about HALF ICE by mass (Lodders 2003); between, a log-linear
// ramp in `r = a / a_frost` anchored on Earth (`r = 1/2.7`, an ocean of 2.26 × 10⁻⁴ of the mass) and
// on the condensate endpoint (`r = 1`, one half). (2) THE RETENTION against the star's heat — the
// census's OWN shoreline-and-envelope verdict (`BodyTaxon::atmosphere`: Zahnle & Catling 2017's
// cosmic shoreline), and the runaway greenhouse at the conservative band's inner edge (a body over
// it holds its water as steam, the steam is photolysed and the hydrogen leaves). The verdict is HARD
// (★ ASSUMED per the design's ask 5 (a): what the census's own atmosphere draw does). (3) The
// volume, in whole cubic kilometres — one km³ over the home planet is two microns of sea level.

/// Earth's own formation ratio, `1 AU / 2.7 AU`.
const EARTH_FROST_RATIO: f64 = 1.0 / FROST_COEFF_AU;
/// Earth's ocean over Earth's mass: 1.35 × 10²¹ kg over 5.97 × 10²⁴ kg.
const EARTH_WATER_MASS_FRACTION: f64 = 2.26e-4;
/// The ice fraction of solar-composition condensates beyond the snow line (Lodders 2003).
const ICE_MASS_FRACTION: f64 = 0.5;
/// Liquid water, for the volume.
const WATER_DENSITY_KGM3: f64 = 1_000.0;

/// The water mass fraction a body accreted at formation ratio `r = a / a_frost`: Earth's at Earth's
/// ratio, one half at and beyond the snow line, a log-linear ramp between. The home planet, at
/// `r = 0.428`: 4.58 × 10⁻⁴, about twice Earth's.
#[must_use]
pub fn water_mass_fraction(frost_ratio: f64) -> f64 {
    if frost_ratio >= 1.0 {
        return ICE_MASS_FRACTION;
    }
    let log_earth = EARTH_WATER_MASS_FRACTION.log10();
    let log_ice = ICE_MASS_FRACTION.log10();
    let along = (frost_ratio - EARTH_FROST_RATIO) / (1.0 - EARTH_FROST_RATIO);
    10f64.powf(log_earth + along * (log_ice - log_earth))
}

/// The water a body HOLDS, in cubic metres: its accreted fraction of its mass over the density of
/// water — or NOTHING, when the shoreline stripped its air (`retained` is the census's own verdict)
/// or the star boils it off past the runaway edge. A stated zero is a fact about a dry world, which
/// is why the record carries it as `Some(0)` and never as an absence.
#[must_use]
pub fn water_inventory_m3(
    mass_kg: f64,
    frost_ratio: f64,
    retained: bool,
    insolation_rel: f64,
) -> f64 {
    let (_, inner) = KOPPARAPU_FLUX_CONSERVATIVE;
    let steamed = insolation_rel > inner;
    if !retained | steamed {
        return 0.0;
    }
    water_mass_fraction(frost_ratio) * mass_kg / WATER_DENSITY_KGM3
}

/// A signed reading through the door: floored to a whole number of `1/scale`, refused when not
/// finite or outside an `i32` — the tilt's cosine is the one word that may be negative.
#[must_use]
pub fn quantise_i32(value: f64, scale: f64) -> Option<i32> {
    let scaled = (value * scale).floor();
    if !scaled.is_finite() {
        return None;
    }
    if scaled < f64::from(i32::MIN) {
        return None;
    }
    if scaled > f64::from(i32::MAX) {
        return None;
    }
    Some(scaled as i32)
}

/// A unit draw mapped log-uniformly onto `[lo, hi]`: `lo · (hi/lo)^u`.
fn log_uniform(u: f64, lo: f64, hi: f64) -> f64 {
    lo * (hi / lo).powf(u)
}

/// The break-up period `2π√(R/g)`: a body cannot turn faster than the period at which a point on
/// its equator flies off. The home planet: 5 060 s, 1.4 h.
#[must_use]
pub fn break_up_period_s(radius_m: f64, gravity_mps2: f64) -> f64 {
    TAU * (radius_m / gravity_mps2).sqrt()
}

/// The tidal-locking distance about a central mass, in metres, after `age_gyr` (§2.2). Our Moon:
/// 8.3 × 10⁸ m about Earth against its 3.8 × 10⁸ m orbit — locked, as it is.
#[must_use]
pub fn tidal_lock_radius_m(central_mass_kg: f64, age_gyr: f64) -> f64 {
    LOCK_CALIBRATION_SMA_AU
        * AU_M
        * (central_mass_kg / M_SUN_KG).cbrt()
        * (age_gyr / LOCK_CALIBRATION_AGE_GYR).powf(1.0 / 6.0)
}

/// THE SPIN: the day in seconds and whether the body is tidally locked. Inside the lock radius the
/// day IS the year; outside it the day is a log-uniform draw over the band, floored at the break-up
/// period (a draw under it would fly the body apart, so the band's floor is the higher of the two).
#[must_use]
pub fn spin_s(orbit: &OrbitalElements, radius_m: f64, gravity_mps2: f64, u: f64) -> (f64, bool) {
    if orbit.sma < tidal_lock_radius_m(orbit.central_mass, SYSTEM_AGE_GYR) {
        return (orbit.period(), true);
    }
    let p_min = break_up_period_s(radius_m, gravity_mps2);
    let lo = if p_min > DAY_MIN_S { p_min } else { DAY_MIN_S };
    let hi = if DAY_MAX_S > lo { DAY_MAX_S } else { lo };
    (log_uniform(u, lo, hi), false)
}

/// The lunisolar torque ratio `Λ = Σ (m_moon / M★) · (a_orbit / a_moon)³` (Laskar, Joutel &
/// Robutel 1993; Earth's is 2.18). Zero for a moonless body.
#[must_use]
pub fn torque_ratio(central_mass_kg: f64, orbit_sma_m: f64, moons: &[Companion]) -> f64 {
    let mut sum = 0.0;
    for moon in moons {
        sum += (moon.mass_kg / central_mass_kg) * (orbit_sma_m / moon.sma_m).powi(3);
    }
    sum
}

/// THE TILT as a cosine: the prior's draw (`u_family` picks the band, `u_cos` the place in it),
/// then damped toward the common band's median by `1 / (1 + Λ)` — a body with a large close moon
/// keeps a steady tilt, a moonless body wanders the whole prior.
#[must_use]
pub fn obliquity_cos(u_family: f64, u_cos: f64, torque_ratio: f64) -> f64 {
    let raw = if u_family < OBLIQUITY_COMMON_SHARE {
        OBLIQUITY_COS_SPLIT + u_cos * (1.0 - OBLIQUITY_COS_SPLIT)
    } else {
        -1.0 + u_cos * (OBLIQUITY_COS_SPLIT + 1.0)
    };
    OBLIQUITY_COS_MEDIAN + (raw - OBLIQUITY_COS_MEDIAN) / (1.0 + torque_ratio)
}

/// THE SURFACE PRESSURE of a body that keeps its air (ruling T9): Earth's, scaled by the mass
/// (the outgassed air), the gravity (its weight) and the inverse square of the radius (the area it
/// presses on). The home planet, Earth's twin: 101 408 Pa.
#[must_use]
pub fn surface_pressure_pa(mass_kg: f64, radius_m: f64, gravity_mps2: f64) -> f64 {
    EARTH_P_SURF_PA
        * (mass_kg / EARTH_MASS_KG)
        * (gravity_mps2 / EARTH_G_MPS2)
        * (EARTH_RADIUS_M / radius_m).powi(2)
}

/// The Rayleigh optical depth at 550 nm: Earth's, scaled by the column `p / (g · μ)`. The home
/// planet at one bar and 28 u: 0.1005.
#[must_use]
pub fn rayleigh_depth(p_surf_pa: f64, gravity_mps2: f64, mu: f64) -> f64 {
    EARTH_RAYLEIGH_550
        * (p_surf_pa / EARTH_P_SURF_PA)
        * (EARTH_G_MPS2 / gravity_mps2)
        * (EARTH_MU / mu)
}

/// THE GREENHOUSE: `(τ_ir, T_surface)`. No air under a surface: no greenhouse, the surface stands
/// at its equilibrium. Air inside the conservative habitable band: the thermostat sets the surface
/// at its set point and `τ_ir` follows, `τ = (4/3)·((T_set/T_eq)⁴ − 1)`, floored at zero for a
/// world already warmer than the set point. Air outside the band: Earth's own greenhouse.
#[must_use]
pub fn greenhouse(t_eq_k: f64, insolation_rel: f64, air_over_a_surface: bool) -> (f64, f64) {
    if !air_over_a_surface {
        return (0.0, t_eq_k);
    }
    let (outer, inner) = KOPPARAPU_FLUX_CONSERVATIVE;
    let in_band = (insolation_rel >= outer) & (insolation_rel <= inner);
    let tau = if in_band {
        let need = 4.0 / 3.0 * ((THERMOSTAT_SET_K / t_eq_k).powi(4) - 1.0);
        if need > 0.0 { need } else { 0.0 }
    } else {
        EARTH_TAU_IR
    };
    (tau, t_eq_k * (1.0 + 0.75 * tau).powf(0.25))
}

// ---- THE LIQUID RULE (stage 6, `slice_8b_design.md` §3.5) --------------------------------------

/// Water's triple point: below this pressure there is no liquid phase at any temperature.
const WATER_TRIPLE_POINT_PA: f64 = 611.657;
const WATER_TRIPLE_POINT_K: f64 = 273.16;
/// Water's boiling point at one bar, and the Clausius–Clapeyron constants that move it with the
/// pressure: the latent heat of vaporisation and the specific gas constant of water vapour.
const WATER_BOIL_1BAR_K: f64 = 373.15;
const WATER_LATENT_J_KG: f64 = 2.26e6;
const WATER_GAS_CONSTANT_J_KG_K: f64 = 461.5;

/// Water's boiling point at surface pressure `p` (Clausius–Clapeyron, integrated from one bar):
/// `1/T = 1/373.15 − (R/L)·ln(p/p⊕)`.
#[must_use]
pub fn water_boiling_point_k(p_surf_pa: f64) -> f64 {
    let inverse = 1.0 / WATER_BOIL_1BAR_K
        - (WATER_GAS_CONSTANT_J_KG_K / WATER_LATENT_J_KG) * (p_surf_pa / EARTH_P_SURF_PA).ln();
    1.0 / inverse
}

/// Whether water is LIQUID at a surface: over the triple point in both pressure and temperature,
/// and under the boiling point at that pressure. A body that fails it carries its water as ice or
/// as steam and gets no sea (§3.5). ★ MEASURED on the home planet: at its DRAWN 1 239 Pa water
/// boils at 279 K, and its thermostat holds the surface at 288 K — so its two oceans are STEAM and
/// it states no sea. That is the pressure draw's own consequence (the design's ask 6), and it is
/// the number ruling T6 ask 2 asked to see.
#[must_use]
pub fn water_is_liquid(t_surface_k: f64, p_surf_pa: f64) -> bool {
    let over_triple = (p_surf_pa > WATER_TRIPLE_POINT_PA) & (t_surface_k > WATER_TRIPLE_POINT_K);
    over_triple & (t_surface_k < water_boiling_point_k(p_surf_pa))
}

/// The lithosphere's effective elastic thickness in metres (§2.6). The home planet: 36.8 km.
#[must_use]
pub fn elastic_thickness_m(bulk_density_kgm3: f64, radius_m: f64, age_gyr: f64) -> f64 {
    let heat = bulk_density_kgm3 * radius_m / (1.0 + age_gyr / RADIOGENIC_TAU_GYR);
    let heat_ref = EARTH_RHO_KGM3 * EARTH_RADIUS_M / (1.0 + EARTH_AGE_GYR / RADIOGENIC_TAU_GYR);
    TE_REF_M * heat_ref / heat
}

/// The record, from one body's facts and its orbit. `None` when any fact will not fit its unit.
///
/// ★ THE WORDS NO KERNEL READS YET (the design's ask 7, taken as ASSUMED pending the owner's
/// answer): the spin, the tilt, the pressure, the two optical depths, the surface temperature and
/// the elastic thickness are DRAWN here (stage 4) and read by no kernel until 8c, 8s, 8e and 18;
/// the water inventory is MODELLED here (stage 5) and the sea is stage 6's, stated ABSENT. Carrying the record
/// now is free while the ground may still move (ruling T1's free window) and impossible after the
/// freeze (slice 14).
#[must_use]
pub fn charter_of(
    facts: &super::BodyFacts,
    orbit: &OrbitalElements,
    seed: u64,
    moons: &[Companion],
    formation_sma_m: f64,
) -> Option<BodyCharter> {
    let taxon = &facts.taxon;
    // ONE refusal for the seven plain words, so a fact that will not fit its unit takes the same
    // path whichever fact it is (HR3: one machinery, never seven).
    let [gravity_word, density_word] = relief_words(taxon.mass_kg, taxon.radius_m);
    let stated: [Option<u32>; PLAIN_WORDS] = [
        gravity_word,
        density_word,
        quantise_u32(escape_velocity_mps(taxon.mass_kg, taxon.radius_m), 1.0),
        quantise_u32(taxon.insolation_rel, 4_096.0),
        quantise_u32(taxon.t_eq_k, 1_000.0),
        quantise_u32(taxon.bond_albedo, 4_096.0),
        quantise_u32(orbit.ecc, 65_536.0),
    ];
    let mut words = [0u32; PLAIN_WORDS];
    for (slot, value) in words.iter_mut().zip(stated) {
        *slot = value?;
    }
    let [
        gravity_mm_s2,
        bulk_density_kgm3,
        escape_velocity_mps,
        insolation_q12,
        t_eq_mk,
        bond_albedo_q12,
        ecc_q16,
    ] = words;
    // THE DRAWS (stage 4), in their frozen order, off the body's own stream.
    let mut stream = SplitMix64::new(child_seed(seed, CHARTER_SALT, 0));
    let u_day = stream.next_f64();
    let u_family = stream.next_f64();
    let u_cos = stream.next_f64();
    let gravity_mps2 = surface_gravity_mps2(taxon.mass_kg, taxon.radius_m);
    let (day_s, locked) = spin_s(orbit, taxon.radius_m, gravity_mps2, u_day);
    let lambda = torque_ratio(orbit.central_mass, orbit.sma, moons);
    let tilt_cos = obliquity_cos(u_family, u_cos, lambda);
    // A surface pressure is a fact about a SURFACE: a giant has an envelope and no floor to stand
    // on, so it states none — and with it no Rayleigh depth, no greenhouse and no lithosphere.
    let solid = !taxon.class.def().is_giant;
    let p_surf_pa = if solid {
        taxon
            .atmosphere
            .map(|_| surface_pressure_pa(taxon.mass_kg, taxon.radius_m, gravity_mps2))
    } else {
        None
    };
    let mu = taxon.atmosphere.map(|a| a.mean_molecular_weight);
    let tau_vis = p_surf_pa
        .zip(mu)
        .map(|(p, m)| rayleigh_depth(p, gravity_mps2, m));
    let (tau_ir, t_surface_k) = greenhouse(taxon.t_eq_k, taxon.insolation_rel, p_surf_pa.is_some());
    // The lithosphere reads the charter's OWN density word — the integer both hosts hold (SL10),
    // never a second float of the same fact.
    let elastic = solid
        .then(|| elastic_thickness_m(f64::from(bulk_density_kgm3), taxon.radius_m, SYSTEM_AGE_GYR));
    let lock_bit = if locked {
        CHARTER_FLAG_TIDALLY_LOCKED
    } else {
        0
    };
    // THE WATER (stage 5): the formation zone against the star's own snow line, the census's own
    // retention verdict, the runaway edge — a volume in whole cubic kilometres, zero when dry.
    let frost_au = frost_line_radius_au(facts.star.luma_lsun, FROST_COEFF_AU);
    let frost_ratio = formation_sma_m / AU_M / frost_au;
    let water_m3 = water_inventory_m3(
        taxon.mass_kg,
        frost_ratio,
        taxon.atmosphere.is_some(),
        taxon.insolation_rel,
    );
    Some(BodyCharter {
        gravity_mm_s2,
        bulk_density_kgm3,
        escape_velocity_mps,
        insolation_q12,
        t_eq_mk,
        t_surface_mk: Some(quantise_u32(t_surface_k, 1_000.0)?),
        bond_albedo_q12,
        mu_q8: optional_word(mu, 256.0)?,
        scale_height_m: optional_word(taxon.atmosphere.map(|a| a.scale_height_m), 1.0)?,
        p_surf_pa: optional_word(p_surf_pa, 1.0)?,
        tau_vis_q12: optional_word(tau_vis, 4_096.0)?,
        tau_ir_q12: Some(quantise_u32(tau_ir, 4_096.0)?),
        day_s: Some(quantise_u32(day_s, 1.0)?),
        obliquity_cos_q1024: Some(quantise_i32(tilt_cos, 1_024.0)?),
        water_km3: Some(quantise_u64(water_m3 / 1.0e9, 1.0)?),
        sea_offset_mm: None,
        elastic_thickness_m: optional_word(elastic, 1.0)?,
        ecc_q16,
        year_s: quantise_u64(orbit.period(), 1.0)?,
        flags: charter_flags(facts) | lock_bit,
    })
}

/// A word that is absent because the BODY has no such fact — an airless planet's molecular weight.
/// `Some(None)` is "the body states none"; `None` is "the body has one and it does not fit", which
/// refuses the whole charter. The two must never collapse into each other.
fn optional_word(value: Option<f64>, scale: f64) -> Option<Option<u32>> {
    match value {
        Some(v) => quantise_u32(v, scale).map(Some),
        None => Some(None),
    }
}

/// The flag word: the air, the ground, and the star that lights it. Only the bits whose author
/// exists today are set; a bit a later stage authors is not a bit of this word yet.
fn charter_flags(facts: &super::BodyFacts) -> u32 {
    let air = if facts.taxon.atmosphere.is_some() {
        CHARTER_FLAG_HAS_AIR
    } else {
        0
    };
    let ground = if facts.taxon.class.def().is_giant {
        0
    } else {
        CHARTER_FLAG_SOLID_SURFACE
    };
    air | ground | (u32::from(facts.star.class as u8) << CHARTER_STAR_CLASS_SHIFT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taxonomy::{Atmosphere, BodyTaxon, PlanetType, SpectralClass};
    use crate::worldgen::{BodyFacts, Placement, StarPhotometrics};

    fn star() -> StarPhotometrics {
        StarPhotometrics {
            mass_msun: 0.953,
            class: SpectralClass::G,
            luma_lsun: 0.823,
        }
    }

    fn taxon(class: PlanetType, air: bool) -> BodyTaxon {
        BodyTaxon {
            class,
            mass_kg: 5.972e24,
            radius_m: 6.371e6,
            insolation_rel: 0.748,
            t_eq_k: 236.785,
            bond_albedo: 0.2998,
            atmosphere: air.then_some(Atmosphere {
                mean_molecular_weight: 28.0,
                scale_height_m: 7_160.0,
                reference_density_kgm3: None,
            }),
        }
    }

    fn facts(class: PlanetType, air: bool) -> BodyFacts {
        BodyFacts {
            system: RealmId::System(7),
            star: star(),
            taxon: taxon(class, air),
            earth_like: false,
        }
    }

    fn orbit() -> OrbitalElements {
        OrbitalElements {
            sma: 1.5e11,
            ecc: 0.0155,
            inclination: 0.0,
            raan: 0.0,
            arg_periapsis: 0.0,
            mean_anomaly_epoch: 0.0,
            central_mass: 1.9e30,
        }
    }

    /// THE DOOR, both ways, on every refusal it states.
    #[test]
    fn the_float_to_integer_door_floors_what_fits_and_refuses_what_does_not() {
        assert_eq!(quantise_u32(9.8215, 1_000.0), Some(9_821));
        assert_eq!(quantise_u32(0.0, 1.0), Some(0));
        assert_eq!(quantise_u32(f64::NAN, 1.0), None, "not a number");
        assert_eq!(quantise_u32(f64::INFINITY, 1.0), None, "not finite");
        assert_eq!(quantise_u32(-0.5, 1.0), None, "a negative fact");
        assert_eq!(quantise_u32(5.0e9, 1.0), None, "over a 32-bit word");
        assert_eq!(quantise_u32(f64::from(u32::MAX), 1.0), Some(u32::MAX));
        assert_eq!(quantise_u64(34_727_239.6, 1.0), Some(34_727_239));
        assert_eq!(quantise_u64(f64::NAN, 1.0), None);
        assert_eq!(quantise_u64(-1.0, 1.0), None);
        assert_eq!(quantise_u64(1.0e30, 1.0), None, "over a 64-bit word");
        assert_eq!(quantise_u64(0.0, 1.0), Some(0));
    }

    /// The bulk density is the mass over the sphere's volume, and Earth's numbers give Earth's.
    #[test]
    fn the_bulk_density_is_the_mass_over_the_volume() {
        let rho = bulk_density_kgm3(5.972e24, 6.371e6);
        assert!((rho - 5_513.0).abs() < 2.0, "Earth's bulk density: {rho}");
    }

    /// A word the BODY has none of is `Some(None)`; a word the body HAS that will not fit refuses
    /// the whole charter. The two answers must never collapse into one.
    #[test]
    fn an_absent_word_and_a_word_that_will_not_fit_are_different_answers() {
        assert_eq!(optional_word(Some(28.0), 256.0), Some(Some(7_168)));
        assert_eq!(optional_word(None, 256.0), Some(None), "the body has none");
        assert_eq!(optional_word(Some(-1.0), 1.0), None, "it does not fit");
    }

    /// The flag word's four arms: air and no air, ground and no ground.
    #[test]
    fn the_flag_word_states_the_air_the_ground_and_the_star() {
        let earth = charter_flags(&facts(PlanetType::Rocky, true));
        assert_eq!(
            earth,
            CHARTER_FLAG_HAS_AIR | CHARTER_FLAG_SOLID_SURFACE | (4 << CHARTER_STAR_CLASS_SHIFT)
        );
        let bare_rock = charter_flags(&facts(PlanetType::Rocky, false));
        assert_eq!(
            bare_rock,
            CHARTER_FLAG_SOLID_SURFACE | (4 << CHARTER_STAR_CLASS_SHIFT),
            "a bare rock has ground and no air"
        );
        let giant = charter_flags(&facts(PlanetType::GasGiant, true));
        assert_eq!(
            giant,
            CHARTER_FLAG_HAS_AIR | (4 << CHARTER_STAR_CLASS_SHIFT),
            "a gas giant has air and nothing to stand on"
        );
    }

    /// The record, whole, from one body's facts — and the refusal when a fact will not fit.
    #[test]
    fn the_record_carries_the_census_words_and_states_the_later_stages_absent() {
        let charter = charter_of(&facts(PlanetType::Rocky, true), &orbit(), 7, &[], 1.5e11)
            .expect("an earth-like body states a charter");
        assert_eq!(charter.gravity_mm_s2, 9_819);
        assert_eq!(charter.insolation_q12, 3_063);
        assert_eq!(charter.mu_q8, Some(7_168));
        assert_eq!(charter.scale_height_m, Some(7_160));
        assert_eq!(charter.ecc_q16, 1_015);
        // The sea (stage 6) is ABSENT, never zero. The water (stage 5) is MODELLED: at 1.0027 AU
        // about a 0.823 L☉ star the snow line stands at 2.449 AU, the formation ratio is 0.4094,
        // the fraction 3.64 × 10⁻⁴ and the volume 2.18 × 10⁹ km³ — about 1.6 Earth oceans.
        assert_eq!(charter.sea_offset_mm, None);
        let water = charter.water_km3.expect("a wet world");
        assert!(
            (2_150_000_000..=2_200_000_000).contains(&water),
            "the water {water} km³"
        );
        // The words stage 4 draws are PRESENT on a rocky world with air, inside their bands.
        let day = charter.day_s.expect("a day");
        assert!(
            (21_600..=360_000).contains(&day),
            "the day {day} s in the band"
        );
        let tilt = charter.obliquity_cos_q1024.expect("a tilt");
        assert!((-1_024..=1_024).contains(&tilt), "the tilt's cosine {tilt}");
        // Earth's twin in mass, gravity and radius reads Earth's pressure: 101 423 Pa MEASURED (the
        // test body's own gravity, 9.8195 m/s², against the calibration's 9.81).
        assert_eq!(charter.p_surf_pa, Some(101_423));
        assert!(charter.tau_vis_q12.is_some(), "a Rayleigh depth under air");
        // The thermostat: 0.748 S⊕ is inside the conservative band, so the surface stands at the
        // set point (288 K exactly) and the greenhouse is the one that puts it there, 1.584 in
        // 1/4096.
        assert_eq!(charter.t_surface_mk, Some(288_000));
        assert_eq!(charter.tau_ir_q12, Some(6_490));
        // 36.787 km from the charter's own density word (5 513 kg/m³), the design's 36.8 km.
        assert_eq!(charter.elastic_thickness_m, Some(36_787));
        assert_eq!(
            charter.flags & CHARTER_FLAG_TIDALLY_LOCKED,
            0,
            "the home orbit spins freely"
        );
        // An airless body states no molecular weight, no scale height, no pressure, no Rayleigh
        // depth and no greenhouse: its surface stands at its own equilibrium.
        let bare = charter_of(&facts(PlanetType::Rocky, false), &orbit(), 7, &[], 1.5e11)
            .expect("a bare rock");
        assert_eq!((bare.mu_q8, bare.scale_height_m), (None, None));
        assert_eq!((bare.p_surf_pa, bare.tau_vis_q12), (None, None));
        assert_eq!(bare.tau_ir_q12, Some(0));
        assert_eq!(bare.t_surface_mk, Some(bare.t_eq_mk));
        assert_eq!(
            bare.water_km3,
            Some(0),
            "the shoreline stripped it: a stated dry world"
        );
        assert!(
            bare.elastic_thickness_m.is_some(),
            "a rock has a lithosphere"
        );
        // A giant has an envelope and no floor: no pressure, no Rayleigh depth, no greenhouse and
        // no lithosphere, and its surface word is its equilibrium.
        let giant = charter_of(&facts(PlanetType::GasGiant, true), &orbit(), 7, &[], 1.5e11)
            .expect("a giant states a charter");
        assert_eq!(giant.flags & CHARTER_FLAG_SOLID_SURFACE, 0);
        assert_eq!((giant.p_surf_pa, giant.tau_vis_q12), (None, None));
        assert_eq!(
            (giant.tau_ir_q12, giant.t_surface_mk),
            (Some(0), Some(giant.t_eq_mk))
        );
        assert_eq!(giant.elastic_thickness_m, None);
        assert!(
            giant.mu_q8.is_some(),
            "the envelope's weight is still stated"
        );
        assert!(
            giant.water_km3.expect("stated") > 0,
            "an envelope holds its volatiles"
        );
        // ONE plain word that will not fit refuses the whole record: a body of no radius has an
        // infinite gravity, and an infinite fact is not a fact.
        let mut broken = facts(PlanetType::Rocky, true);
        broken.taxon.radius_m = 0.0;
        assert_eq!(
            charter_of(&broken, &orbit(), 7, &[], 1.5e11),
            None,
            "an infinite gravity"
        );
        // A body whose atmosphere will not fit refuses too — the optional word's own refusal.
        let mut sour = facts(PlanetType::Rocky, true);
        sour.taxon.atmosphere = Some(Atmosphere {
            mean_molecular_weight: -1.0,
            scale_height_m: 7_160.0,
            reference_density_kgm3: None,
        });
        assert_eq!(charter_of(&sour, &orbit(), 7, &[], 1.5e11), None);
        // A year no 64-bit word holds refuses too.
        let mut still = orbit();
        still.central_mass = f64::MIN_POSITIVE;
        assert_eq!(
            charter_of(&facts(PlanetType::Rocky, true), &still, 7, &[], 1.5e11),
            None
        );
    }

    /// ★ THE HOME PLANET'S OWN SHARD DERIVES ITS CHARTER FROM THE SUBTREE IT BOOTS (§4.2 rule 1) —
    /// no message, no arm, nothing across a realm boundary. The whole twenty words are pinned in
    /// `crates/bins/tests/home_body_pin.rs` (`charter_pin`); this drives the read itself.
    #[test]
    fn a_planets_own_subtree_states_its_charter_and_a_star_system_states_none() {
        let config = UniverseConfig::world(1.0, 0.05);
        let held = std::collections::BTreeSet::from([vd_core::worldgen::HOME_SYSTEM]);
        let lineage = std::collections::BTreeSet::from([vd_core::worldgen::GALAXY]);
        let charter = body_charter_in_subtree(
            crate::worldgen::HOME_SEED,
            &config,
            &held,
            &lineage,
            vd_core::worldgen::HOME_PLANET,
        )
        .expect("the home planet's shard derives its own charter");
        assert_eq!(charter.gravity_mm_s2, 9_818, "the home planet's gravity");
        assert_eq!(charter.bulk_density_kgm3, 5_513);
        // 2.736 × 10⁹ km³ — about two Earth oceans, the design's own 2.737 × 10⁹ to 0.04 %.
        assert_eq!(
            charter.water_km3,
            Some(2_735_928_089),
            "the water, stage 5's model"
        );
        // A star system has no body: no taxonomy row, so no charter, so its shard states no surface.
        assert_eq!(
            body_charter_in_subtree(
                crate::worldgen::HOME_SEED,
                &config,
                &held,
                &lineage,
                vd_core::worldgen::HOME_SYSTEM,
            ),
            None,
            "a star system states no charter"
        );
    }

    /// THE TWO REFUSALS OF THE FOREST READ, on a hand forest — the world drives the first and never
    /// the second.
    #[test]
    fn a_body_with_no_row_and_a_body_on_no_orbit_state_no_charter() {
        let system = RealmId::System(7);
        let planet = RealmId::Planet(7);
        let bodies = vec![
            GeneratedBody {
                realm: system,
                parent: None,
                shape: vd_core::geometry::Boundary::Shell { r: 1.0e12 },
                placement: Placement::StaticOffset(glam::DVec3::ZERO),
                photometrics: Some(star()),
                taxon: None,
                look: None,
            },
            GeneratedBody {
                realm: planet,
                parent: Some(system),
                shape: vd_core::geometry::Boundary::Shell { r: 6.371e6 },
                // A taxonomy row and NO ORBIT — the shape the world never produces.
                placement: Placement::StaticOffset(glam::DVec3::ZERO),
                photometrics: None,
                taxon: Some(taxon(PlanetType::Rocky, true)),
                look: None,
            },
        ];
        assert_eq!(
            charter_in_forest(&bodies, RealmId::Planet(99)),
            None,
            "a body the forest does not name"
        );
        assert_eq!(
            charter_in_forest(&bodies, system),
            None,
            "a star carries no taxonomy row"
        );
        assert_eq!(
            charter_in_forest(&bodies, planet),
            None,
            "a body on no orbit states no year"
        );
        // The same body ON an orbit does state one — the true arm of both refusals.
        let mut orbiting = bodies;
        orbiting[1].placement = Placement::Orbital(orbit());
        assert!(charter_in_forest(&orbiting, planet).is_some());
    }

    /// THE SIGNED DOOR: the tilt's cosine is the one word that may be negative.
    #[test]
    fn the_signed_door_floors_a_reading_and_refuses_what_does_not_fit() {
        assert_eq!(quantise_i32(-0.5, 1_024.0), Some(-512));
        assert_eq!(quantise_i32(0.999_9, 1_024.0), Some(1_023));
        assert_eq!(quantise_i32(f64::NAN, 1_024.0), None);
        assert_eq!(quantise_i32(-3.0e9, 1.0), None, "under an i32");
        assert_eq!(quantise_i32(3.0e9, 1.0), None, "over an i32");
    }

    /// THE SPIN: the innermost planet of a system like the home one is locked, the home orbit is
    /// not; a locked day is the year; a free day is in the band and never under the break-up
    /// period; the draw is a function of the unit.
    #[test]
    fn the_innermost_planet_is_tidally_locked_and_the_home_orbit_spins_freely() {
        let radius_m = 6.371e6;
        let g = surface_gravity_mps2(5.972e24, radius_m);
        let lock_au = tidal_lock_radius_m(orbit().central_mass, SYSTEM_AGE_GYR) / AU_M;
        assert!(
            (lock_au - 0.387_9).abs() < 0.001,
            "the lock radius {lock_au} AU"
        );
        let mut inner = orbit();
        inner.sma = 0.36 * AU_M;
        let (locked_day, locked) = spin_s(&inner, radius_m, g, 0.5);
        assert!(locked);
        assert!(
            (locked_day - inner.period()).abs() < 1e-6,
            "a locked day is the year"
        );
        let (free_day, free) = spin_s(&orbit(), radius_m, g, 0.5);
        assert!(!free);
        assert!((DAY_MIN_S..=DAY_MAX_S).contains(&free_day));
        assert!(free_day > break_up_period_s(radius_m, g));
        // The unit's ends are the band's ends.
        let (lo, _) = spin_s(&orbit(), radius_m, g, 0.0);
        let (hi, _) = spin_s(&orbit(), radius_m, g, 1.0);
        assert!((lo - DAY_MIN_S).abs() < 1e-6);
        assert!((hi - DAY_MAX_S).abs() < 1e-6);
        // A body whose break-up period stands over the band's floor draws from that period up: a
        // dust ball of 100 km and one millionth of a g.
        let dust_g = 1.0e-5;
        let p_min = break_up_period_s(1.0e5, dust_g);
        assert!(p_min > DAY_MAX_S, "the case the band cannot hold");
        let (dust_day, _) = spin_s(&orbit(), 1.0e5, dust_g, 0.0);
        assert!(
            (dust_day - p_min).abs() < 1e-6,
            "the floor is the break-up period"
        );
        // Our Moon about Earth: locked, as it is.
        let moon_orbit = OrbitalElements {
            sma: 3.844e8,
            central_mass: 5.972e24,
            ..orbit()
        };
        assert!(
            spin_s(&moon_orbit, 1.737e6, 1.62, 0.5).1,
            "the Moon is locked"
        );
        // The record carries the lock bit and the year as the day.
        let charter =
            charter_of(&facts(PlanetType::Rocky, true), &inner, 7, &[], 1.5e11).expect("locked");
        assert_eq!(
            charter.flags & CHARTER_FLAG_TIDALLY_LOCKED,
            CHARTER_FLAG_TIDALLY_LOCKED
        );
        assert_eq!(u64::from(charter.day_s.expect("a day")), charter.year_s);
    }

    /// THE TILT: a moonless body keeps its drawn tilt whole; a large close moon damps the draw
    /// toward the common band's median by `1 / (1 + Λ)`; Earth's Moon gives the published 2.18.
    #[test]
    fn a_moonless_body_keeps_its_drawn_tilt_and_a_large_moon_damps_it() {
        let raw_common = OBLIQUITY_COS_SPLIT + 0.25 * (1.0 - OBLIQUITY_COS_SPLIT);
        assert!(
            (obliquity_cos(0.5, 0.25, 0.0) - raw_common).abs() < 1e-12,
            "undamped"
        );
        let raw_odd = -1.0 + 0.25 * (OBLIQUITY_COS_SPLIT + 1.0);
        assert!(
            (obliquity_cos(0.95, 0.25, 0.0) - raw_odd).abs() < 1e-12,
            "the tenth body"
        );
        let moon = [Companion {
            mass_kg: 7.35e22,
            sma_m: 3.844e8,
        }];
        let lambda = torque_ratio(1.989e30, 1.496e11, &moon);
        assert!(
            (lambda - 2.18).abs() < 0.01,
            "Earth's lunisolar torque ratio, {lambda}"
        );
        let damped = obliquity_cos(0.5, 0.25, lambda);
        let pull = (damped - OBLIQUITY_COS_MEDIAN) / (raw_common - OBLIQUITY_COS_MEDIAN);
        assert!((pull - 1.0 / (1.0 + lambda)).abs() < 1e-12);
        assert_eq!(torque_ratio(1.989e30, 1.496e11, &[]), 0.0, "moonless");
    }

    /// THE RAYLEIGH DEPTH on the home planet at one bar reads Earth's within a tenth, and the
    /// record quantises it to 412 in 1/4096 — the design's own number.
    #[test]
    fn the_rayleigh_depth_at_one_bar_reads_the_designs_number() {
        let tau = rayleigh_depth(101_325.0, 9.821, 28.0);
        assert!((tau - 0.100_5).abs() < 5e-4, "{tau}");
        assert_eq!(quantise_u32(tau, 4_096.0), Some(411));
    }

    /// THE GREENHOUSE: an airless surface stands at its equilibrium; a cold world inside the band
    /// is warmed to the set point; a warm world inside the band is not cooled under its
    /// equilibrium; outside the band the greenhouse is Earth's.
    #[test]
    fn the_thermostat_warms_a_cold_habitable_world_and_leaves_the_rest_to_the_laws() {
        assert_eq!(greenhouse(236.785, 0.748, false), (0.0, 236.785));
        let (tau, t_s) = greenhouse(236.785, 0.748, true);
        assert!(
            (t_s - THERMOSTAT_SET_K).abs() < 1e-9,
            "the set point, {t_s}"
        );
        assert!((tau - 1.584).abs() < 1e-3, "the design's 1.584, {tau}");
        let (hot_tau, hot_t) = greenhouse(300.0, 1.05, true);
        assert_eq!(hot_tau, 0.0);
        assert!((hot_t - 300.0).abs() < 1e-9);
        let (cold_tau, cold_t) = greenhouse(236.785, 0.3, true);
        assert!((cold_tau - EARTH_TAU_IR).abs() < 1e-12);
        assert!(
            (cold_t - 268.2).abs() < 0.1,
            "the design's own snowball number, {cold_t}"
        );
        let (over_tau, _) = greenhouse(300.0, 1.5, true);
        assert!(
            (over_tau - EARTH_TAU_IR).abs() < 1e-12,
            "past the inner edge"
        );
    }

    /// THE ELASTIC THICKNESS of the home planet reads the design's own 36.8 km, and a younger,
    /// hotter body's is thinner.
    #[test]
    fn the_elastic_thickness_reads_the_designs_number_and_thins_with_heat() {
        let home = elastic_thickness_m(5_513.0, 6.371e6, SYSTEM_AGE_GYR);
        assert!((home - 36_780.0).abs() < 60.0, "{home}");
        assert!(
            elastic_thickness_m(5_513.0, 6.371e6, 1.0) < home,
            "a younger body runs hotter"
        );
        assert!(
            elastic_thickness_m(3_344.0, 3.5e5, SYSTEM_AGE_GYR) > home,
            "a small cold moon"
        );
    }

    /// THE DRAWS ARE A FUNCTION OF THE SEED: two seeds differ, one seed repeats.
    #[test]
    fn the_charter_draws_are_a_function_of_the_body_seed() {
        let a = charter_of(&facts(PlanetType::Rocky, true), &orbit(), 7, &[], 1.5e11).expect("a");
        let b = charter_of(&facts(PlanetType::Rocky, true), &orbit(), 8, &[], 1.5e11).expect("b");
        let again =
            charter_of(&facts(PlanetType::Rocky, true), &orbit(), 7, &[], 1.5e11).expect("a again");
        assert_eq!(a, again);
        assert_ne!(
            (a.day_s, a.obliquity_cos_q1024),
            (b.day_s, b.obliquity_cos_q1024),
            "another seed draws another history"
        );
        assert_eq!(
            a.p_surf_pa, b.p_surf_pa,
            "the pressure is a fact, not a draw (T9)"
        );
    }

    /// THE MOONS THE FOREST READ HANDS THE LAW: a moon with a row and an orbit counts; a child with
    /// no row or on no orbit is not a moon the law can read; a body that is not a planet has no seed.
    #[test]
    fn the_forest_read_gathers_a_planets_moons_and_only_a_planet_carries_a_seed() {
        let system = RealmId::System(7);
        let planet = RealmId::Planet(7);
        let moon = |realm: RealmId, taxon: Option<BodyTaxon>, placement: Placement| GeneratedBody {
            realm,
            parent: Some(planet),
            shape: vd_core::geometry::Boundary::Shell { r: 1.0e6 },
            placement,
            photometrics: None,
            taxon,
            look: None,
        };
        let moon_orbit = OrbitalElements {
            sma: 3.844e8,
            central_mass: 5.972e24,
            ..orbit()
        };
        let bodies = vec![
            GeneratedBody {
                realm: system,
                parent: None,
                shape: vd_core::geometry::Boundary::Shell { r: 1.0e12 },
                placement: Placement::StaticOffset(glam::DVec3::ZERO),
                photometrics: Some(star()),
                taxon: None,
                look: None,
            },
            GeneratedBody {
                realm: planet,
                parent: Some(system),
                shape: vd_core::geometry::Boundary::Shell { r: 6.371e6 },
                placement: Placement::Orbital(orbit()),
                photometrics: None,
                taxon: Some(taxon(PlanetType::Rocky, true)),
                look: None,
            },
            moon(
                RealmId::Planet(71),
                Some(taxon(PlanetType::Rocky, false)),
                Placement::Orbital(moon_orbit),
            ),
            moon(RealmId::Planet(72), None, Placement::Orbital(moon_orbit)),
            moon(
                RealmId::Planet(73),
                Some(taxon(PlanetType::Rocky, false)),
                Placement::StaticOffset(glam::DVec3::ZERO),
            ),
        ];
        assert_eq!(
            companions_in_forest(&bodies, planet),
            vec![Companion {
                mass_kg: 5.972e24,
                sma_m: 3.844e8,
            }]
        );
        // A system's own companions are its planets — the same read, one level up.
        assert_eq!(
            companions_in_forest(&bodies, system),
            vec![Companion {
                mass_kg: 5.972e24,
                sma_m: 1.5e11,
            }]
        );
        assert_eq!(
            companions_in_forest(&bodies, RealmId::Planet(71)),
            vec![],
            "a moon has none"
        );
        assert_eq!(body_seed(planet), Some(7));
        assert_eq!(body_seed(system), None);
        // The charter read through the forest damps the planet's tilt by that moon, and the same
        // read on a moonless forest does not: the two differ in the tilt alone among the draws.
        let with_moon = charter_in_forest(&bodies, planet).expect("with a moon");
        let without = charter_in_forest(&bodies[..2], planet).expect("moonless");
        assert_eq!(with_moon.day_s, without.day_s);
        assert_ne!(with_moon.obliquity_cos_q1024, without.obliquity_cos_q1024);
        // The moon of that planet: airless, so the shoreline leaves it DRY — a stated zero.
        let moon_charter =
            charter_in_forest(&bodies, RealmId::Planet(71)).expect("a moon's charter");
        assert_eq!(moon_charter.water_km3, Some(0));
        assert_eq!(
            formation_sma_in_forest(&bodies, RealmId::Planet(71), &moon_orbit),
            Some(1.5e11),
            "a moon forms in its planet's zone"
        );
        assert_eq!(
            formation_sma_in_forest(&bodies, planet, &orbit()),
            Some(1.5e11),
            "a planet forms in its own"
        );
        // A moon whose planet sits on no orbit: the shape the world never produces, refused.
        let mut adrift = bodies.clone();
        adrift[1].placement = Placement::StaticOffset(glam::DVec3::ZERO);
        assert_eq!(
            formation_sma_in_forest(&adrift, RealmId::Planet(71), &moon_orbit),
            None
        );
        assert_eq!(
            formation_sma_in_forest(&bodies, RealmId::Planet(99), &orbit()),
            None,
            "unnamed"
        );
    }

    /// THE WATER MODEL: Earth's fraction at Earth's ratio, one half at and past the snow line, the
    /// home planet's 4.58 × 10⁻⁴ at its 0.428, monotone between; a stripped or steamed body holds
    /// nothing; a wet one holds its fraction of its mass over the density of water.
    #[test]
    fn the_water_model_is_anchored_on_earth_and_the_snow_line_and_dries_the_stripped_and_the_steamed()
     {
        assert!((water_mass_fraction(EARTH_FROST_RATIO) - EARTH_WATER_MASS_FRACTION).abs() < 1e-12);
        assert_eq!(water_mass_fraction(1.0), ICE_MASS_FRACTION);
        assert_eq!(water_mass_fraction(2.0), ICE_MASS_FRACTION);
        let home = water_mass_fraction(0.428_15);
        assert!(
            (home - 4.58e-4).abs() < 0.02e-4,
            "the design's 4.58e-4: {home}"
        );
        assert!(
            water_mass_fraction(0.6) > home,
            "monotone toward the snow line"
        );
        assert_eq!(
            water_inventory_m3(5.972e24, 0.428_15, false, 0.748),
            0.0,
            "stripped"
        );
        assert_eq!(
            water_inventory_m3(5.972e24, 0.428_15, true, 1.2),
            0.0,
            "steamed"
        );
        let wet_km3 = water_inventory_m3(5.972e24, 0.428_15, true, 0.748) / 1.0e9;
        assert!(
            (wet_km3 - 2.737e9).abs() < 0.01e9,
            "the design's 2.737e9 km³: {wet_km3}"
        );
    }

    /// THE PRESSURE LAW (T9): Earth reads one bar; a half-mass body of Earth's density reads
    /// under half a bar; the home planet's twin reads Earth's, and water is liquid under it.
    #[test]
    fn the_surface_pressure_is_earths_scaled_by_the_mass_the_gravity_and_the_area() {
        let earth = surface_pressure_pa(EARTH_MASS_KG, EARTH_RADIUS_M, EARTH_G_MPS2);
        assert!((earth - EARTH_P_SURF_PA).abs() < 1e-6, "{earth}");
        // At Earth's density the radius scales as M^(1/3), the gravity as M^(1/3) and the area as
        // M^(2/3), so the pressure scales as M^(2/3): a half-mass body reads 0.63 bar.
        let half_r = EARTH_RADIUS_M * 0.5f64.cbrt();
        let half_g = surface_gravity_mps2(EARTH_MASS_KG * 0.5, half_r);
        let half = surface_pressure_pa(EARTH_MASS_KG * 0.5, half_r, half_g);
        let expected = EARTH_P_SURF_PA * 0.5f64.powf(2.0 / 3.0) * (half_g / (0.5f64.cbrt() * 9.81));
        assert!((half - expected).abs() < 1.0, "{half} against {expected}");
        // Two assertions, never one `&&` (its short-circuit is an uncoverable branch, HR5).
        assert!(half < 0.7 * EARTH_P_SURF_PA, "{half}");
        assert!(half > 0.55 * EARTH_P_SURF_PA, "{half}");
        let home = surface_pressure_pa(5.972e24, 6.371e6, 9.818);
        assert!(water_is_liquid(288.0, home), "a sea under a derived air");
    }

    /// THE LIQUID RULE: Earth's surface holds liquid water; the home planet at its DRAWN 1 239 Pa
    /// does not — its 288 K surface stands over the 279 K boiling point at that pressure, so its two
    /// oceans are steam; under the triple point nothing is liquid; a frozen surface is not.
    #[test]
    fn water_is_liquid_on_earth_and_steam_on_the_home_planets_thin_air() {
        assert!(water_is_liquid(288.0, 101_325.0), "Earth");
        let boils = water_boiling_point_k(1_239.0);
        assert!(
            (boils - 279.4).abs() < 0.5,
            "the boiling point at 1 239 Pa: {boils}"
        );
        assert!(
            !water_is_liquid(288.0, 1_239.0),
            "the home planet's drawn air: steam"
        );
        assert!(
            water_is_liquid(278.0, 1_239.0),
            "a cooler surface under the same air: liquid"
        );
        assert!(!water_is_liquid(288.0, 500.0), "under the triple point");
        assert!(!water_is_liquid(260.0, 101_325.0), "frozen");
        assert!(
            (water_boiling_point_k(101_325.0) - 373.15).abs() < 1e-9,
            "one bar"
        );
    }
}
