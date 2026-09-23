//! ★ THE CLIMATE INSIDE THE SCHEDULE (the landform arc, slice 8c stage C3; `slice_8c_design.md`
//! §5; the investigation's `05_climate_biomes_weather.md` §4–§5.6; `03_erosion_rivers.md` §4.5, §4.9).
//!
//! A climate is what the air does on average over a body. The solve needs it because relief and
//! rain are a loop: mountains lift the air and make rain, rain cuts the mountains. So every
//! `CLIMATE_EVERY` passes the climate is recomputed over the relief AS IT THEN STANDS, as a closed
//! form over the macro lattice — no heap, no order, `O(1)` a node (03 §4.5 requires it: an upwind
//! march would be the solve's dominant cost). Its rows, each from charter words by a published law
//! with a calibration body (ruling T9):
//!
//! - **the temperature**: the charter's mean surface temperature spread by latitude with the
//!   annual-mean insolation profile (North 1975) and an energy-balance contrast (Budyko–Sellers,
//!   calibrated on Earth's 38 K annual-mean equator-to-pole difference; no air carries no heat, so
//!   an airless body's poles fall much further), then cooled with height by THE LAPSE RATE
//!   `Γ = f_moist · g / c_p` (a law of the gas and the gravity: Earth 9.76 K/km dry, 6.5 K/km
//!   environmental);
//! - **the circulation**: how many belts of rising and sinking air the spin allows (Held & Hou 1980,
//!   calibrated on Earth's 30° Hadley edge): a slow or tidally locked body has ONE cell pole to
//!   pole, a fast one several; the surface wind comes from the east in the first cell (the trades),
//!   from the west in the second, and so on;
//! - **the rain, at three scales**: the water the air can hold (Clausius–Clapeyron in the Tetens
//!   form, calibrated on Earth's mean 990 mm/yr), the zonal belt (wet where air rises, dry where it
//!   sinks: the subtropical desert belt), and the OROGRAPHIC LIFT with its RAIN SHADOW read from the
//!   upwind neighbour's height over the chord;
//! - **the frost line and the ELA**: the height at which the air reaches freezing over each
//!   latitude, and the equilibrium line altitude above it in a dry climate (03 §4.9, calibration the
//!   Andes: 0 °C at 4 800 m, the ELA near 6 000 m);
//! - **the aridity**: the potential evaporation against the rain (the UNEP aridity classes).
//!
//! **Example.** On the home planet the trade wind blows from the east onto a coastal range. The
//! windward valley's node reads a rise from its upwind neighbour, the lift factor doubles its rain,
//! and the discharge cuts a deep trunk valley; the node behind the crest reads a fall, its rain
//! shadow halves the rain, and the same-sized basin stays a high dry upland. Nothing here runs on the
//! tick: the climate is computed inside the solve, once per body, and the pictures read its rows.

// ★ A CLIMATE MAY DIVIDE: it runs once per body on the CPU, never in a kernel (03 §5.3).
#![allow(
    clippy::integer_division,
    clippy::modulo_arithmetic,
    reason = "the climate runs on the CPU once per body, never in a kernel; integer-exact on every host"
)]

use vd_recipe::Gi;
use vd_recipe::bend::DIR_BITS;
use vd_recipe::height::POLE_AXIS;

use crate::body::BodyDefinition;
use crate::gf::Gf;
use crate::macro_lattice::{MacroLattice, NO_NODE};
use crate::solve::{P_MAX_MM_YR, P_MIN_MM_YR, SolveWords, Z_STEPS_PER_M};

/// The climate over a body's macro lattice: one word per node in every row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Climate {
    /// The mean surface air temperature at the node, tenths of a kelvin.
    pub temperature_dk: Vec<i16>,
    /// The rain, mm/yr: zero on an airless or a dry body, else in `[P_MIN_MM_YR, P_MAX_MM_YR]`.
    pub rain_mm_yr: Vec<u32>,
    /// The aridity: 0 wet, 255 hyper-arid — the potential evaporation against the rain.
    pub aridity_q8: Vec<u8>,
    /// ★ THE POTENTIAL EVAPORATION at the node, whole mm/yr ([`pet_mm_yr`]) — the number the
    /// lakes' water budget reads beside the rain (ruling W11).
    pub pet_mm_yr: Vec<u32>,
    /// ★ THE RUNOFF at the node, whole mm/yr ([`runoff_mm_yr`]): the share of the rain that leaves
    /// the node as water instead of going back into the air, by the Turc–Pike partition. What the
    /// rivers carry and what a lake downstream is fed by (ruling W11).
    pub runoff_mm_yr: Vec<u32>,
    /// The height (sixteenths from the ladder radius) at which the air reaches freezing over the
    /// node's latitude; `i32::MAX` where it never freezes, `i32::MIN` where it is always frozen.
    pub frost_z: Vec<i32>,
    /// The equilibrium line altitude at the node: the frost line plus the dry-climate offset.
    pub ela_z: Vec<i32>,
    /// The stencil slot (0..8) of the neighbour the surface wind comes FROM; [`NO_WIND`] where
    /// there is no air.
    pub upwind: Vec<u8>,
    /// ★ THE LAPSE RATE the rows were computed with, millikelvin a kilometre ([`lapse_rate_k_m`]):
    /// an INTEGER, so the climate stays comparable word for word. The ice's mass balance reads it
    /// to turn a height over the equilibrium line into a temperature.
    pub lapse_mk_km: u32,
}

/// The upwind word of a node with no wind over it (an airless body).
pub const NO_WIND: u8 = 255;

// ---- THE LAWS AND THEIR CALIBRATIONS ---------------------------------------------------------

/// The solar constant at Earth, W/m²: what `insolation_q12 = 4 096` means.
pub const SOLAR_CONSTANT_W_M2: f64 = 1_361.0;
/// The gas constant, J/(mol·K).
pub const GAS_CONSTANT_J_MOL_K: f64 = 8.314;
/// ★ THE MOIST FACTOR of the lapse rate: condensing water gives heat back, so the environmental
/// lapse rate is this share of the dry adiabat (calibration Earth: 6.5 over 9.76 K/km; 03 §4.9).
pub const LAPSE_MOIST_SHARE: f64 = 0.666;
/// ★ THE ENERGY-BALANCE CONSTANTS (Budyko–Sellers; North 1975), calibration Earth: the outgoing
/// longwave's sensitivity `B` in W/(m²·K) and the meridional diffusion `D` in W/(m²·K) at Earth's
/// surface pressure. The latitude contrast is `T₂ = S̄(1 − α)·s₂ / (B + 6D)`; `D` scales with the
/// surface pressure, because thick air carries heat poleward and no air carries none (05 §5.2).
pub const OLR_SENSITIVITY_W_M2_K: f64 = 1.9;
pub const EARTH_DIFFUSION_W_M2_K: f64 = 0.44;
/// Earth's surface pressure, Pa: the diffusion's calibration.
pub const EARTH_P_SURF_PA: f64 = 101_325.0;
/// ★ THE HADLEY EDGE's law (Held & Hou 1980): `sin²φ_H = 5·ΔH·g·H / (3·Ω²·a²)`; `ΔH = 0.388` is a
/// REVERSE FIT to Earth's 30° (05 §5.3: it reproduces Venus and Jupiter and misses Mars by one cell).
pub const HELD_HOU_DELTA_H: f64 = 0.388;
/// The most cells a hemisphere holds (05 §5.3's clamp).
pub const CELLS_MAX: usize = 6;
/// ★ THE TETENS FORM of Clausius–Clapeyron: `e_s = 610.78 · exp(17.27·t / (t + 237.3))` pascals
/// for `t` in °C (calibration: water over Earth's temperatures).
pub const TETENS_A_PA: f64 = 610.78;
pub const TETENS_B: f64 = 17.27;
pub const TETENS_C_C: f64 = 237.3;
/// The kelvin of 0 °C.
pub const FREEZING_K: f64 = 273.15;
/// Earth's mean surface temperature, K, and Earth's mean rain, mm/yr: the supply's calibration —
/// an Earth-like column at Earth's mean temperature receives Earth's mean rain before the belt and
/// the relief act on it.
pub const EARTH_MEAN_SURFACE_K: f64 = 288.0;
pub const EARTH_MEAN_RAIN_MM_YR: f64 = 990.0;
/// ★ THE BELT's two-point anchor (05 §5.4, U15): an Earth-like column at the rising branch gets
/// about 2 000 mm/yr and under the sinking branch about 100 mm/yr. As shares of the supply at those
/// latitudes' temperatures: the rising share and the sinking share.
pub const BELT_RISING_SHARE: f64 = 1.0;
pub const BELT_SINKING_SHARE: f64 = 0.09;
/// ★ THE OROGRAPHIC GAIN per unit of upwind slope (Smith 2003's linear model, calibration the
/// Cascades: a kilometre of rise over eight doubles the rain on the windward side). A stated
/// calibration on one range, named as such. The lee side's shadow is its reciprocal.
pub const OROGRAPHIC_GAIN: f64 = 8.0;
/// The lift factor's bounds: a cliff does not make forty times the rain.
pub const LIFT_MIN: f64 = 0.1;
pub const LIFT_MAX: f64 = 4.0;
/// ★ OHMURA'S ELA CURVE (Ohmura, Kasser & Funk 1992, *Climate at the equilibrium line of glaciers*,
/// J. Glaciol. 38(130), 397–411, <https://doi.org/10.1017/S0022143000002276>): over 70 mid- and
/// high-latitude glaciers the annual precipitation at the equilibrium line and the ABLATION-SEASON
/// (June–August) air temperature there satisfy `P = 645 + 296·T + 9·T²`, `P` in mm w.e./yr and `T`
/// in °C, with a standard error of 200 mm w.e. The paper's own three-way anchor is the point
/// `1 °C, 350 mm w.e., 7 W/m²`, and the curve reads `P(1 °C) = 950 mm`.
///
/// Calibration body: Ohmura's 70 glaciers (tropical glaciers excluded by the paper itself, so a
/// node wetter than the fit ever saw is an EXTRAPOLATION and is named as one).
pub const OHMURA_A_MM: f64 = 645.0;
pub const OHMURA_B_MM_K: f64 = 296.0;
pub const OHMURA_C_MM_K2: f64 = 9.0;
/// ★ THE SEASONAL INSOLATION COEFFICIENT `s₁(ε) = 2·sin ε` — the FIRST Legendre term of the
/// daily-mean insolation, the same expansion [`insolation_s2`] reads its second term from (North
/// 1975; North & Coakley 1979 state `s₁ = −0.796` for Earth's 23.44° tilt, and `2·sin 23.44° =
/// 0.796`). The local summer's daily insolation stands `S̄·s₁·|sin φ|` over the annual mean at the
/// latitude, so a body with a big tilt has big seasons whatever its annual mean does.
pub const SEASON_S1_PER_SIN: f64 = 2.0;
/// ★ THE SEASONAL MIXED LAYER of an ocean, metres: the depth of water that answers the year's own
/// cycle (the standard seasonal energy-balance depth; calibration body EARTH's mid-latitude ocean).
pub const OCEAN_MIXED_LAYER_M: f64 = 50.0;
/// Sea water's volumetric heat capacity, J/(m³·K): 1 030 kg/m³ × 3 900 J/(kg·K).
pub const SEA_WATER_HEAT_J_M3_K: f64 = 4.0e6;
/// Rock's volumetric heat capacity, J/(m³·K): 2 500 kg/m³ × 800 J/(kg·K).
pub const ROCK_HEAT_J_M3_K: f64 = 2.0e6;
/// Rock's thermal diffusivity, m²/s — the number the ANNUAL SKIN DEPTH `√(2κ/ω)` reads, so the
/// ground's own share of the seasonal heat store is a law and never a dial.
pub const ROCK_DIFFUSIVITY_M2_S: f64 = 1.0e-6;
/// ★ THE POTENTIAL EVAPORATION at Earth's mean temperature, mm/yr (calibration: Earth's land mean),
/// scaled with the water-holding capacity of the air.
pub const EARTH_PET_MM_YR: f64 = 1_000.0;
/// The UNEP aridity classes read on `PET/P`: humid under 0.5 (the byte's zero), hyper-arid over
/// 20 (`P/PET < 0.05`, the byte's 255), the byte logarithmic between.
pub const ARIDITY_WET_RATIO: f64 = 0.5;
pub const ARIDITY_HYPER_RATIO: f64 = 20.0;
/// The mean molecular weight's unit, 1/256 of an atomic unit, in kg/mol.
const MU_Q8_KG_MOL: f64 = 1.0e-3 / 256.0;
/// The sines of the Hadley edge at which the cell count changes: `round(90°/φ_H) = k` when
/// `φ_H` lies in `(90/(k+½), 90/(k−½)]`, so the count is 1 above `sin 60°`, 2 above `sin 36°`, 3
/// above `sin 25.7°`, 4 above `sin 20°`, 5 above `sin 16.4°`, else 6 — literal sines, never a call.
const CELL_COUNT_SINES: [f64; 5] = [
    0.866_025_4,
    0.587_785_3,
    0.433_883_7,
    0.342_020_1,
    0.281_732_6,
];
/// The sines of the cell edges for one to six cells a hemisphere, `sin(k · 90° / n)`: literal.
const CELL_EDGE_SINES: [[f64; 7]; 6] = [
    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [
        0.0,
        std::f64::consts::FRAC_1_SQRT_2,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    [0.0, 0.5, 0.866_025_4, 1.0, 1.0, 1.0, 1.0],
    [
        0.0,
        0.382_683_4,
        std::f64::consts::FRAC_1_SQRT_2,
        0.923_879_5,
        1.0,
        1.0,
        1.0,
    ],
    [
        0.0,
        0.309_017_0,
        0.587_785_3,
        0.809_017_0,
        0.951_056_5,
        1.0,
        1.0,
    ],
    [
        0.0,
        0.258_819_0,
        0.5,
        std::f64::consts::FRAC_1_SQRT_2,
        0.866_025_4,
        0.965_925_8,
        1.0,
    ],
];

/// The circulation of a body: how many cells a hemisphere holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Circulation {
    /// The cells per hemisphere, 1 to [`CELLS_MAX`].
    pub cells: usize,
}

/// The words the per-node laws read, computed ONCE per climate from the charter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClimateLaws {
    /// The mean surface temperature, K.
    pub t_mean_k: Gf,
    /// The latitude contrast `T₂`, K (the coefficient of `P₂(sin lat)`).
    pub t2_k: Gf,
    /// The environmental lapse rate, K/m (zero on an airless body).
    pub lapse_k_m: Gf,
    /// The circulation.
    pub circulation: Circulation,
    /// Whether it rains at all: the body holds air AND water.
    pub rains: bool,
    /// ★ THE ABSORBED FLUX `S̄·(1 − α)`, W/m²: the mean insolation the ground keeps. The annual
    /// contrast and the SEASON read the same number.
    pub absorbed_w_m2: Gf,
    /// ★ THE SEASONAL INSOLATION COEFFICIENT `s₁ = 2·sin ε` ([`SEASON_S1_PER_SIN`]).
    pub s1: Gf,
    /// ★ THE SEASONAL DAMPING `|B + 2D + iCω|`, W/(m²·K) ([`seasonal_damping_w_m2_k`]): what the
    /// year's own forcing must push against — the outgoing longwave, the `n = 1` diffusion, and
    /// the surface's heat store, which is what really holds the season down.
    pub seasonal_damping_w_m2_k: Gf,
}

fn dot(a: [Gf; 3], b: [Gf; 3]) -> Gf {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `v / |v|`; a zero vector stays zero (a stated answer, not a division by zero).
fn normalise(v: [Gf; 3]) -> [Gf; 3] {
    let len = dot(v, v).sqrt();
    if len > Gf::ZERO {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

/// A node's direction as fenced floats.
fn dir_of(dir: [Gi; 3]) -> [Gf; 3] {
    let one = Gf::from_i64(1 << DIR_BITS);
    [
        Gf::from_i64(dir[0].raw()) / one,
        Gf::from_i64(dir[1].raw()) / one,
        Gf::from_i64(dir[2].raw()) / one,
    ]
}

/// The second Legendre polynomial `P₂(x) = (3x² − 1)/2`: the shape of the annual-mean insolation.
#[must_use]
pub fn legendre_p2(x: Gf) -> Gf {
    (Gf::from_f64(3.0) * x * x - Gf::ONE) * Gf::HALF
}

/// ★ THE INSOLATION PROFILE's coefficient `s₂(ε) = (5/16)·(3 sin²ε − 2)` (North 1975): the poles
/// get less than the equator until the tilt passes 54.7°, where the sign turns. Earth's 23.4° gives
/// −0.477 against the published −0.482. Reads the charter's COSINE of the obliquity.
#[must_use]
pub fn insolation_s2(obliquity_cos: Gf) -> Gf {
    let sin2 = Gf::ONE - obliquity_cos * obliquity_cos;
    Gf::from_f64(5.0 / 16.0) * (Gf::from_f64(3.0) * sin2 - Gf::TWO)
}

/// ★ THE LATITUDE CONTRAST `T₂ = S̄(1 − α)·s₂ / (B + 6D)` (the one-mode energy balance), with the
/// mean flux `S̄ = S⊕·insolation/4`, the bond albedo, and the diffusion scaled by the surface
/// pressure over Earth's. Earth: `340·0.7·(−0.477)/(1.9 + 2.64) = −25 K`, a 38 K equator-to-pole
/// difference in the annual mean. An airless body has no diffusion and its poles fall 80 K.
#[must_use]
pub fn latitude_contrast_k(insolation_rel: Gf, bond_albedo: Gf, s2: Gf, p_surf_pa: Gf) -> Gf {
    let mean_flux = Gf::from_f64(SOLAR_CONSTANT_W_M2) * insolation_rel / Gf::from_f64(4.0);
    let diffusion =
        Gf::from_f64(EARTH_DIFFUSION_W_M2_K) * p_surf_pa / Gf::from_f64(EARTH_P_SURF_PA);
    mean_flux * (Gf::ONE - bond_albedo) * s2
        / (Gf::from_f64(OLR_SENSITIVITY_W_M2_K) + Gf::from_f64(6.0) * diffusion)
}

/// ★ THE LAPSE RATE, K/m: `Γ = f_moist · g / c_p` with `c_p = (7/2)·R/μ` for a diatomic gas (03
/// §4.9). Earth: `0.666 · 9.81 / 1 004 = 6.5 K/km`. A body with no air (no `μ`) has no lapse.
#[must_use]
pub fn lapse_rate_k_m(gravity_mm_s2: u32, mu_q8: Option<u32>) -> Gf {
    let Some(mu) = mu_q8 else {
        return Gf::ZERO;
    };
    let g = Gf::from_i64(i64::from(gravity_mm_s2)) / Gf::from_i64(1_000);
    let mu_kg_mol = Gf::from_i64(i64::from(mu)) * Gf::from_f64(MU_Q8_KG_MOL);
    let c_p = Gf::from_f64(3.5) * Gf::from_f64(GAS_CONSTANT_J_MOL_K) / mu_kg_mol;
    Gf::from_f64(LAPSE_MOIST_SHARE) * g / c_p
}

/// ★ THE CIRCULATION from the spin (Held & Hou 1980): `sin²φ_H = 5·ΔH·g·H / (3·Ω²·a²)` with
/// `Ω = 2π/day`; the cell count is `round(90°/φ_H)` clamped to one and [`CELLS_MAX`], read off a
/// literal table of sines so no arcsine is called. No air (no scale height) or no spin means one
/// cell; a tidally locked body's day is its year, so it gets one too.
#[must_use]
pub fn circulation(
    radius_m: f64,
    gravity_mm_s2: u32,
    scale_height_m: Option<u32>,
    day_s: Option<u32>,
) -> Circulation {
    let (Some(h), Some(day)) = (scale_height_m, day_s) else {
        return Circulation { cells: 1 };
    };
    if day == 0 {
        return Circulation { cells: 1 };
    }
    let g = Gf::from_i64(i64::from(gravity_mm_s2)) / Gf::from_i64(1_000);
    let omega = Gf::TWO * Gf::from_f64(std::f64::consts::PI) / Gf::from_i64(i64::from(day));
    let a = Gf::from_f64(radius_m);
    let sin2 = Gf::from_f64(5.0 * HELD_HOU_DELTA_H) * g * Gf::from_i64(i64::from(h))
        / (Gf::from_f64(3.0) * omega * omega * a * a);
    let sin_edge = sin2.sqrt();
    let mut cells = CELLS_MAX;
    let mut k = 0;
    while k < CELL_COUNT_SINES.len() {
        if sin_edge > Gf::from_f64(CELL_COUNT_SINES[k]) {
            cells = k + 1;
            break;
        }
        k += 1;
    }
    Circulation { cells }
}

/// The cell a latitude lies in (0 from the equator) and the position within it from its rising
/// branch (0) to its sinking branch (1), read on the sine of the latitude against the literal edge
/// table (a stated approximation: the position is linear in the sine, not in the angle). An even
/// cell rises at its equator-side edge and sinks at its pole-side edge; an odd cell the other way —
/// so on a three-cell body the equator rises, 30° sinks, 60° rises and the pole sinks.
#[must_use]
pub fn cell_position(circulation: Circulation, sin_lat_abs: Gf) -> (usize, Gf) {
    let edges = &CELL_EDGE_SINES[circulation.cells - 1];
    let mut k = 0;
    while k + 1 < circulation.cells && sin_lat_abs >= Gf::from_f64(edges[k + 1]) {
        k += 1;
    }
    let lo = Gf::from_f64(edges[k]);
    let hi = Gf::from_f64(edges[k + 1]);
    let along = ((sin_lat_abs - lo) / (hi - lo)).clamp(Gf::ZERO, Gf::ONE);
    // An even cell rises at its equator-side edge; an odd cell rises at its pole-side edge.
    let from_rising = if k % 2 == 0 { along } else { Gf::ONE - along };
    (k, from_rising)
}

/// ★ THE BELT FACTOR: the share of the supply that falls, wet at the rising branch and dry at the
/// sinking one, faded by the bump `(1 − u²)²` so the desert belt is narrow and the wet belt wide.
#[must_use]
pub fn belt_factor(from_rising: Gf) -> Gf {
    let r = Gf::ONE - from_rising * from_rising;
    Gf::from_f64(BELT_SINKING_SHARE) + Gf::from_f64(BELT_RISING_SHARE - BELT_SINKING_SHARE) * r * r
}

/// ★ THE SATURATION VAPOUR PRESSURE (Tetens), pascals, at a temperature in kelvin.
#[must_use]
pub fn saturation_vapour_pa(t_k: Gf) -> Gf {
    let t_c = t_k - Gf::from_f64(FREEZING_K);
    Gf::from_f64(TETENS_A_PA)
        * (Gf::from_f64(TETENS_B) * t_c / (t_c + Gf::from_f64(TETENS_C_C))).exp()
}

/// ★ THE SUPPLY: the rain an Earth-like column receives at a temperature before the belt and the
/// relief act — Earth's mean rain scaled by the water the air can hold.
#[must_use]
pub fn supply_mm_yr(t_k: Gf) -> Gf {
    Gf::from_f64(EARTH_MEAN_RAIN_MM_YR) * saturation_vapour_pa(t_k)
        / saturation_vapour_pa(Gf::from_f64(EARTH_MEAN_SURFACE_K))
}

/// ★ THE OROGRAPHIC LIFT AND THE RAIN SHADOW: the factor on the rain from the upwind slope `rise /
/// chord` — a rise multiplies, a fall divides, both by `1 + gain·|slope|`, bounded.
#[must_use]
pub fn lift_factor(rise_m: Gf, chord_m: Gf) -> Gf {
    let slope = rise_m / chord_m;
    let gain = Gf::ONE + Gf::from_f64(OROGRAPHIC_GAIN) * slope.abs();
    let factor = if slope >= Gf::ZERO {
        gain
    } else {
        Gf::ONE / gain
    };
    factor.clamp(Gf::from_f64(LIFT_MIN), Gf::from_f64(LIFT_MAX))
}

/// The mean surface temperature at a latitude and a height, kelvin.
#[must_use]
pub fn temperature_k(laws: &ClimateLaws, sin_lat: Gf, height_m: Gf) -> Gf {
    laws.t_mean_k + laws.t2_k * legendre_p2(sin_lat) - laws.lapse_k_m * height_m
}

/// ★ THE SEASONAL INSOLATION COEFFICIENT `s₁(ε) = 2·sin ε` ([`SEASON_S1_PER_SIN`]). Reads the
/// charter's COSINE of the obliquity, as [`insolation_s2`] does.
#[must_use]
pub fn insolation_s1(obliquity_cos: Gf) -> Gf {
    let sin2 = (Gf::ONE - obliquity_cos * obliquity_cos).greater(Gf::ZERO);
    Gf::from_f64(SEASON_S1_PER_SIN) * sin2.sqrt()
}

/// ★ THE SEASONAL HEAT STORE of a body's surface, J/(m²·K): the share of the surface under water
/// times the ocean's seasonal mixed layer, plus the dry share times the ground's own ANNUAL SKIN
/// DEPTH `√(2κ/ω)` — how deep the year's wave reaches into rock. `year_s` is the body's own year.
/// A body with no year (no orbit stated) stores nothing, and its season is radiative.
#[must_use]
pub fn seasonal_capacity_j_m2_k(wet_share: Gf, year_s: u64) -> Gf {
    if year_s == 0 {
        return Gf::ZERO;
    }
    let omega = Gf::TWO * Gf::from_f64(std::f64::consts::PI) / Gf::from_i64(year_s as i64);
    let skin_m = (Gf::TWO * Gf::from_f64(ROCK_DIFFUSIVITY_M2_S) / omega).sqrt();
    let wet = wet_share.clamp(Gf::ZERO, Gf::ONE);
    wet * Gf::from_f64(OCEAN_MIXED_LAYER_M) * Gf::from_f64(SEA_WATER_HEAT_J_M3_K)
        + (Gf::ONE - wet) * skin_m * Gf::from_f64(ROCK_HEAT_J_M3_K)
}

/// ★ THE SEASONAL DAMPING `|B + 2D + iCω|`, W/(m²·K): the one-mode energy balance's answer to a
/// forcing that turns once a year — the outgoing longwave `B`, the meridional diffusion at the
/// `n = 1` mode's eigenvalue `n(n+1) = 2`, and the surface heat store `C` turning with `ω = 2π/yr`.
/// The store dominates on any body with an ocean, which is WHY a season is tens of kelvin small
/// instead of the fifty the bare radiation balance would give.
///
/// ★ ITS STATED LIMIT: one column stands for the ocean AND the land, so a continental interior
/// really runs hotter in its own summer than this law says. The ice line it gives is therefore
/// LOW, never high, and the census's G-ICE line measures the consequence.
#[must_use]
pub fn seasonal_damping_w_m2_k(p_surf_pa: Gf, capacity_j_m2_k: Gf, year_s: u64) -> Gf {
    let diffusion =
        Gf::from_f64(EARTH_DIFFUSION_W_M2_K) * p_surf_pa / Gf::from_f64(EARTH_P_SURF_PA);
    let radiative = Gf::from_f64(OLR_SENSITIVITY_W_M2_K) + Gf::TWO * diffusion;
    if year_s == 0 {
        return radiative;
    }
    let omega = Gf::TWO * Gf::from_f64(std::f64::consts::PI) / Gf::from_i64(year_s as i64);
    let store = capacity_j_m2_k * omega;
    (radiative * radiative + store * store).sqrt()
}

/// ★ THE ABLATION SEASON'S WARMTH over the annual mean at a latitude, kelvin: the seasonal
/// insolation anomaly `S̄(1 − α)·s₁·|sin φ|` over the seasonal damping. Zero at the equator, where
/// there is no season, and greatest at the pole, where the whole year's tilt lands.
#[must_use]
pub fn seasonal_amplitude_k(laws: &ClimateLaws, sin_lat: Gf) -> Gf {
    if laws.seasonal_damping_w_m2_k <= Gf::ZERO {
        return Gf::ZERO;
    }
    laws.absorbed_w_m2 * laws.s1 * sin_lat.abs() / laws.seasonal_damping_w_m2_k
}

/// The ABLATION-SEASON temperature at a latitude and a height over the body's datum, kelvin.
#[must_use]
pub fn summer_k(laws: &ClimateLaws, sin_lat: Gf, height_m: Gf) -> Gf {
    temperature_k(laws, sin_lat, height_m) + seasonal_amplitude_k(laws, sin_lat)
}

/// ★ THE EQUILIBRIUM LINE'S OWN SUMMER TEMPERATURE for a rain, °C: Ohmura's curve
/// `P = 645 + 296·T + 9·T²` inverted for `T` (the rising root). A dry glacier needs a colder
/// summer than a wet one, because it has less snow to lose: `P = 350 mm` gives −1.0 °C, `P = 950`
/// gives +1.0 °C, `P = 2 000` gives +4.1 °C.
///
/// **Example.** A range on the home planet's dry lee takes 300 mm a year, so its snow only
/// outlives the summer where the summer stands under −1 °C; the wet windward side of the same
/// range keeps snow at +4 °C, and its line is the lower of the two.
#[must_use]
pub fn ohmura_ela_summer_c(rain_mm_yr: u32) -> Gf {
    let b = Gf::from_f64(OHMURA_B_MM_K);
    let c4 = Gf::from_f64(4.0) * Gf::from_f64(OHMURA_C_MM_K2);
    let disc = b * b - c4 * (Gf::from_f64(OHMURA_A_MM) - Gf::from_i64(i64::from(rain_mm_yr)));
    (disc.greater(Gf::ZERO).sqrt() - b) / (Gf::TWO * Gf::from_f64(OHMURA_C_MM_K2))
}

/// ★ THE BODY'S TEMPERATURE DATUM in sixteenths: the height the charter's MEAN SURFACE temperature
/// stands at. A body with a sea has its mean surface AT the sea (Earth: 288 K at sea level); a body
/// with none has it at the area-weighted mean of its ground. Before this law the lapse cooled from
/// the LADDER RADIUS, a geometric datum the sea stands kilometres above, and the whole planet came
/// out kilometres' worth of lapse too cold — MEASURED on the home planet: the sea surface read
/// −11 °C, the rain fell to a sixth of Earth's, and every land node stood over the freezing line.
#[must_use]
pub fn datum_z(lattice: &MacroLattice, z: &[i32], sea_z: Option<i32>) -> i32 {
    if let Some(sea) = sea_z {
        return sea;
    }
    let mut weighted = 0i128;
    let mut area = 0u128;
    for (k, &h) in z.iter().enumerate() {
        let a = u128::from(lattice.area_m2(k as u32));
        weighted += i128::from(h) * (a as i128);
        area += a;
    }
    if area == 0 {
        return 0;
    }
    (weighted / (area as i128)) as i32
}

/// ★ THE WET SHARE of a body's surface: the area under the sea over the whole area, zero where
/// there is no sea. The season's heat store reads it ([`seasonal_capacity_j_m2_k`]).
#[must_use]
pub fn wet_share(lattice: &MacroLattice, z: &[i32], sea_z: Option<i32>) -> Gf {
    let Some(sea) = sea_z else {
        return Gf::ZERO;
    };
    let mut wet = 0u128;
    let mut all = 0u128;
    for (k, &h) in z.iter().enumerate() {
        let a = u128::from(lattice.area_m2(k as u32));
        all += a;
        if h <= sea {
            wet += a;
        }
    }
    if all == 0 {
        return Gf::ZERO;
    }
    Gf::from_i64(wet as i64) / Gf::from_i64(all as i64)
}

/// ★ THE FROST LINE at a latitude, in sixteenths from the ladder radius: the height at which the
/// ANNUAL MEAN air reaches freezing, measured from the body's own `datum_z`; `i32::MAX` where the
/// lapse is zero and the ground is warm, `i32::MIN` where it is zero and cold.
#[must_use]
pub fn frost_line_z(laws: &ClimateLaws, sin_lat: Gf, datum_z: i32) -> i32 {
    let at_datum = laws.t_mean_k + laws.t2_k * legendre_p2(sin_lat);
    line_z(laws, at_datum - Gf::from_f64(FREEZING_K), datum_z)
}

/// ★ THE EQUILIBRIUM LINE at a node, in sixteenths from the ladder radius (ruling B2 step 2): the
/// height at which the ABLATION-SEASON temperature falls to the one Ohmura's curve names for the
/// node's own rain. THE LINE IS A SUMMER LINE, never the annual mean — Egholm et al. 2009 take the
/// `Ts = 0` isotherm of the ablation-season temperature, Ohmura et al. 1992 put the line at about
/// +1 °C in June–August, and the mean-annual freezing line reaches sea level near 60° on Earth,
/// where the real line stands 1 100–1 500 m up.
///
/// **Example.** The pilot flies the belt at the equator: the season there is nothing, so the line
/// sits where the year's own mean puts it and she sees no ice under 5 000 m. She flies north and
/// the line RISES with the summer, then falls again where the summer runs out.
#[must_use]
pub fn ela_z(laws: &ClimateLaws, sin_lat: Gf, rain_mm_yr: u32, datum_z: i32) -> i32 {
    let at_datum = summer_k(laws, sin_lat, Gf::ZERO);
    let threshold = Gf::from_f64(FREEZING_K) + ohmura_ela_summer_c(rain_mm_yr);
    line_z(laws, at_datum - threshold, datum_z)
}

/// The height a temperature excess reaches at the lapse rate, in sixteenths from the ladder
/// radius: the datum plus `over / Γ`, saturating where there is no lapse to climb.
fn line_z(laws: &ClimateLaws, over: Gf, datum_z: i32) -> i32 {
    if laws.lapse_k_m <= Gf::ZERO {
        return if over >= Gf::ZERO { i32::MAX } else { i32::MIN };
    }
    let height_m = over / laws.lapse_k_m;
    let steps =
        height_m * Gf::from_i64(i64::from(Z_STEPS_PER_M)) + Gf::from_i64(i64::from(datum_z));
    steps
        .clamp(
            Gf::from_i64(i64::from(i32::MIN) + 1),
            Gf::from_i64(i64::from(i32::MAX) - 1),
        )
        .to_i64_floor() as i32
}

/// ★ THE POTENTIAL EVAPORATION at a temperature, mm/yr: Earth's land mean scaled by the water the
/// air can hold at that temperature (the Tetens form of Clausius–Clapeyron). Calibration body:
/// EARTH, whose land mean is [`EARTH_PET_MM_YR`] at [`EARTH_MEAN_SURFACE_K`]. The aridity byte and
/// the lakes' water budget read this ONE law, so a node's dryness and its lake's evaporation can
/// never disagree.
///
/// **Example.** A node on the belt at 300 K asks for 1 630 mm a year and gets 100 mm of rain: it
/// is a desert, and a hollow beside it evaporates every river that reaches it.
#[must_use]
pub fn pet_mm_yr(t_k: Gf) -> Gf {
    Gf::from_f64(EARTH_PET_MM_YR) * saturation_vapour_pa(t_k)
        / saturation_vapour_pa(Gf::from_f64(EARTH_MEAN_SURFACE_K))
}

/// ★ THE RUNOFF, mm/yr: what is left of the rain after the ground has given the air as much as the
/// air can take AND the ground can spare — the Turc–Pike form of the Budyko partition,
/// `E = P / √(1 + (P/PET)²)`, so `Q = P − E` (Turc 1954, *Le bilan d'eau des sols*, Annales
/// Agronomiques 5, 491–569; Pike 1964, J. Hydrology 2(2), 116–123,
/// <https://doi.org/10.1016/0022-1694(64)90022-8>; the framework is Budyko 1974). Calibration body:
/// EARTH's own catchments, where the curve was fitted.
///
/// ★ WHY NOT `max(0, P − PET)`. That reading is a DEFECT and it was measured as one: Earth's land
/// mean rain is about 750 mm/yr against a potential evaporation over 1 000 mm/yr, so the difference
/// is negative nearly everywhere and EARTH'S OWN RIVERS would carry nothing. MEASURED on the home
/// planet (2026-09-22, the first lake census under ruling W11): every one of 74 682 hollows ran dry
/// and 31 nodes of 8.87 million held water. The Budyko partition says what the difference cannot:
/// a dry basin still yields a little water, and a wet one yields most of its rain.
///
/// **Example.** A node on the belt's windward slope gets 2 000 mm of rain and asks for 1 000: it
/// hands 900 to the air and 1 100 to the river. A node in the lee gets 100 and asks for 1 000: it
/// hands nearly all 100 back and the river gets a trickle — which is why the hollow below it is a
/// pan and not a lake.
#[must_use]
pub fn runoff_mm_yr(rain: Gf, pet: Gf) -> Gf {
    if pet <= Gf::ZERO {
        return rain;
    }
    let ratio = rain / pet;
    rain - rain / (Gf::ONE + ratio * ratio).sqrt()
}

/// ★ THE ARIDITY BYTE: the potential evaporation (Earth's land mean scaled by the air's water
/// capacity) over the rain, on the UNEP classes' logarithmic axis — 0 at a ratio of 0.5 and under
/// (humid), 255 at 20 and over (hyper-arid). No rain at all is hyper-arid.
#[must_use]
pub fn aridity_q8(t_k: Gf, rain_mm_yr: u32) -> u8 {
    if rain_mm_yr == 0 {
        return 255;
    }
    let ratio = pet_mm_yr(t_k) / Gf::from_i64(i64::from(rain_mm_yr));
    let lo = Gf::from_f64(ARIDITY_WET_RATIO).ln();
    let hi = Gf::from_f64(ARIDITY_HYPER_RATIO).ln();
    let share = ((ratio.ln() - lo) / (hi - lo)).clamp(Gf::ZERO, Gf::ONE);
    (share * Gf::from_f64(255.0) + Gf::HALF).to_i64_floor() as u8
}

/// ★ THE SURFACE WIND at a node: a unit tangent vector, zonal from the east in an even cell and
/// from the west in an odd one (the trades, the westerlies, the polar easterlies), with a
/// meridional part toward the cell's rising branch. The spin is prograde about `+Z`. At the exact
/// pole the local east is undefined and the wind is the zero vector.
#[must_use]
pub fn surface_wind(circulation: Circulation, d: [Gf; 3]) -> [Gf; 3] {
    let sin_lat = d[POLE_AXIS];
    let (cell, from_rising) = cell_position(circulation, sin_lat.abs());
    let east = normalise([-d[1], d[0], Gf::ZERO]);
    let north = normalise([
        -d[0] * sin_lat,
        -d[1] * sin_lat,
        Gf::ONE - sin_lat * sin_lat,
    ]);
    // The wind BLOWS toward the west in an even cell (it comes from the east).
    let zonal = if cell % 2 == 0 { -Gf::ONE } else { Gf::ONE };
    // It blows toward the rising branch: equatorward in an even cell, poleward in an odd one, on
    // the hemisphere's own side; weaker as it nears the branch.
    let toward_pole = if cell % 2 == 0 { -Gf::ONE } else { Gf::ONE };
    let hemisphere = if sin_lat >= Gf::ZERO {
        Gf::ONE
    } else {
        -Gf::ONE
    };
    let meridional = toward_pole * hemisphere * from_rising;
    normalise([
        zonal * east[0] + meridional * north[0],
        zonal * east[1] + meridional * north[1],
        zonal * east[2] + meridional * north[2],
    ])
}

/// The stencil slot of the neighbour the wind comes FROM: the neighbour whose offset points most
/// against the wind; the first slot on a tie (the zero wind at the pole).
#[must_use]
pub fn upwind_slot(lattice: &MacroLattice, node: u32, d: [Gf; 3], wind: [Gf; 3]) -> u8 {
    let mut best = 0u8;
    let mut best_dot = Gf::from_f64(f64::MAX);
    for (slot, m) in lattice.neighbours(node).iter().enumerate() {
        if *m == NO_NODE {
            continue;
        }
        let dm = dir_of(lattice.direction(*m));
        let against = dot([dm[0] - d[0], dm[1] - d[1], dm[2] - d[2]], wind);
        if against < best_dot {
            best_dot = against;
            best = slot as u8;
        }
    }
    best
}

/// The per-body words of the laws from the charter.
#[must_use]
pub fn climate_laws(body: &BodyDefinition, words: &SolveWords, wet_share: Gf) -> ClimateLaws {
    let t_mean_k =
        Gf::from_i64(i64::from(words.t_surface_mk.unwrap_or(words.t_eq_mk))) / Gf::from_i64(1_000);
    let obliquity_cos =
        Gf::from_i64(i64::from(words.obliquity_cos_q1024.unwrap_or(1_024))) / Gf::from_i64(1_024);
    let p_surf = Gf::from_i64(i64::from(words.p_surf_pa.unwrap_or(0)));
    let insolation_rel = Gf::from_i64(i64::from(words.insolation_q12)) / Gf::from_i64(4_096);
    let albedo = Gf::from_i64(i64::from(words.bond_albedo_q12)) / Gf::from_i64(4_096);
    let t2_k = latitude_contrast_k(insolation_rel, albedo, insolation_s2(obliquity_cos), p_surf);
    let absorbed_w_m2 =
        Gf::from_f64(SOLAR_CONSTANT_W_M2) * insolation_rel / Gf::from_f64(4.0) * (Gf::ONE - albedo);
    let capacity = seasonal_capacity_j_m2_k(wet_share, words.year_s);
    ClimateLaws {
        t_mean_k,
        t2_k,
        absorbed_w_m2,
        s1: insolation_s1(obliquity_cos),
        seasonal_damping_w_m2_k: seasonal_damping_w_m2_k(p_surf, capacity, words.year_s),
        lapse_k_m: lapse_rate_k_m(body.facts().gravity_mm_s2, words.mu_q8),
        circulation: circulation(
            body.ladder().radius_m(),
            body.facts().gravity_mm_s2,
            words.scale_height_m,
            words.day_s,
        ),
        rains: words.has_air() && words.water_km3 > 0,
    }
}

/// ★ THE CLIMATE over `lattice` for the relief `z` as it stands. `sea_z` IS read, twice: it names
/// the body's own temperature DATUM ([`datum_z`]) and it gives the wet share the season's heat
/// store reads. The supply is still a temperature law alone (no fetch, no continental term — a
/// stated limit of the O(1) form).
#[must_use]
pub fn climate(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &SolveWords,
    z: &[i32],
    sea_z: Option<i32>,
) -> Climate {
    let datum = datum_z(lattice, z, sea_z);
    let laws = climate_laws(body, words, wet_share(lattice, z, sea_z));
    let n = lattice.node_count();
    let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let mut out = Climate {
        temperature_dk: Vec::with_capacity(n),
        rain_mm_yr: Vec::with_capacity(n),
        aridity_q8: Vec::with_capacity(n),
        pet_mm_yr: Vec::with_capacity(n),
        runoff_mm_yr: Vec::with_capacity(n),
        frost_z: Vec::with_capacity(n),
        ela_z: Vec::with_capacity(n),
        upwind: Vec::with_capacity(n),
        lapse_mk_km: (laws.lapse_k_m * Gf::from_i64(1_000_000) + Gf::HALF)
            .to_i64_floor()
            .clamp(0, i64::from(u32::MAX)) as u32,
    };
    for node in 0..n as u32 {
        let i = node as usize;
        let d = dir_of(lattice.direction(node));
        let sin_lat = d[POLE_AXIS];
        // ★ THE AIR OVER WATER STANDS AT THE WATER'S OWN SURFACE, never at the floor under it: a
        // node four kilometres down on the abyssal plain carries the sea's air, not air four
        // kilometres' worth of lapse warmer. MEASURED before this line: the sea floor read 344 K,
        // asked for 16 160 mm of evaporation a year and was given 6 708 mm of rain.
        let air_z = match sea_z {
            Some(sea) => z[i].max(sea),
            None => z[i],
        };
        let height_m = Gf::from_i64(i64::from(air_z) - i64::from(datum)) / steps;
        let t_k = temperature_k(&laws, sin_lat, height_m);
        let (rain, slot) = if laws.rains {
            let wind = surface_wind(laws.circulation, d);
            let slot = upwind_slot(lattice, node, d, wind);
            let m = lattice.neighbours(node)[usize::from(slot)];
            let up_z = match sea_z {
                Some(sea) => z[m as usize].max(sea),
                None => z[m as usize],
            };
            let rise_m = Gf::from_i64(i64::from(air_z) - i64::from(up_z)) / steps;
            let chord = Gf::from_i64(i64::from(lattice.chord_m(node, m).max(1)));
            let (_, from_rising) = cell_position(laws.circulation, sin_lat.abs());
            let mm = supply_mm_yr(t_k) * belt_factor(from_rising) * lift_factor(rise_m, chord);
            let rain = (mm + Gf::HALF)
                .to_i64_floor()
                .clamp(i64::from(P_MIN_MM_YR), i64::from(P_MAX_MM_YR))
                as u32;
            (rain, slot)
        } else {
            (0, NO_WIND)
        };
        let frost = frost_line_z(&laws, sin_lat, datum);
        out.temperature_dk.push(
            (t_k * Gf::from_f64(10.0) + Gf::HALF)
                .to_i64_floor()
                .clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16,
        );
        out.rain_mm_yr.push(rain);
        out.aridity_q8.push(aridity_q8(t_k, rain));
        let pet = pet_mm_yr(t_k);
        out.pet_mm_yr.push(
            (pet + Gf::HALF)
                .to_i64_floor()
                .clamp(0, i64::from(u32::MAX)) as u32,
        );
        out.runoff_mm_yr.push(
            (runoff_mm_yr(Gf::from_i64(i64::from(rain)), pet) + Gf::HALF)
                .to_i64_floor()
                .clamp(0, i64::from(rain)) as u32,
        );
        out.frost_z.push(frost);
        out.ela_z.push(ela_z(&laws, sin_lat, rain, datum));
        out.upwind.push(slot);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::{home_moon, home_moon_solve_words, home_solve_words};
    use crate::land::initial_land;
    use crate::macro_lattice::STENCIL;

    fn gf(v: f64) -> Gf {
        Gf::from_f64(v)
    }

    /// ★ GATE G-ICE, AS A LAW TEST AND NOT A WORLD READING (ruling B5, 2026-09-22).
    ///
    /// The home planet's own ice covers 4 % of its land, and the cause is its 55° tilt — a seed
    /// identity choice the owner refused to edit. So the gate asks the LAW, on an EARTH-LIKE
    /// CHARTER, and the home planet's own share stays a census reading.
    ///
    /// **THE BODY.** Earth's obliquity 23.4° (`cos = 0.918`), Earth's insolation, Earth's bond
    /// albedo 0.306, Earth's surface pressure and mean surface temperature 288 K, Earth's year and
    /// day, Earth's water. The home planet's own RADIUS and GRAVITY stand under it, because the
    /// body is what the generator can build; its gravity is 9.818 m/s² against Earth's 9.807, and
    /// its radius is about half Earth's, which moves the Hadley cell's reach and nothing in the
    /// snowline's own law.
    ///
    /// **THE LAND.** One elevation everywhere: EARTH'S MEAN LAND ELEVATION, 840 m over the sea
    /// (the published hypsometric mean). So the ice's share is set by LATITUDE alone, which is what
    /// the law is being asked about, and not by a synthetic mountain range somebody chose.
    ///
    /// **THE NUMBER AND ITS BODY.** Earth carries ice on about 10 % of its land today (NSIDC), and
    /// about 30 % at a glacial maximum (Egholm et al. 2009). A flat 840 m Earth reads 10 % exactly
    /// when its snowline crosses 840 m near 63° of latitude, which is where Earth's own snowline
    /// crosses it. The band below is 4 % to 25 %: wide enough that the radius we cannot give the
    /// body does not decide it, narrow enough that the home planet's own 4 % tilt-driven reading
    /// would go RED here.
    #[test]
    fn g_ice_the_snowline_law_on_an_earth_like_charter() {
        let body = crate::home::home_planet();
        // A coarse lattice of the body's own: the snowline is a function of latitude and height,
        // and 2 166 nodes read it as well as eight million.
        let lattice = crate::macro_lattice::MacroLattice::of(&body)
            .expect("a macro lattice")
            .coarser(6)
            .expect("a coarse level");
        let words = SolveWords {
            // Earth's own charter, word for word.
            insolation_q12: 4_096,
            t_eq_mk: 255_000,
            t_surface_mk: Some(288_000),
            bond_albedo_q12: 1_253,
            mu_q8: Some(7_416),
            scale_height_m: Some(8_500),
            p_surf_pa: Some(101_325),
            tau_ir_q12: Some(2_458),
            day_s: Some(86_400),
            obliquity_cos_q1024: Some(940),
            ecc_q16: 1_097,
            year_s: 31_557_600,
            water_km3: 1_386_000_000,
            ..crate::home::home_solve_words()
        };
        // ONE ELEVATION FOR THE LAND and one for the floor: Earth's mean land, 840 m over the
        // sea, on 29 % of the nodes - Earth's own land share - and Earth's mean ocean depth,
        // 3 682 m, under the rest. The land is laid by the node's own index and NOT by latitude,
        // so the share the ice takes is the share of the SPHERE its snowline reaches, which is the
        // question the law is being asked.
        let land_z = 840 * Z_STEPS_PER_M;
        let floor_z = -3_682 * Z_STEPS_PER_M;
        let is_land = |i: usize| i % 100 < 29;
        let z: Vec<i32> = (0..lattice.node_count())
            .map(|i| if is_land(i) { land_z } else { floor_z })
            .collect();
        let climate = climate(&body, &lattice, &words, &z, Some(0));
        let land = (0..lattice.node_count()).filter(|&i| is_land(i)).count();
        let under_ice = (0..lattice.node_count())
            .filter(|&i| is_land(i) && climate.ela_z[i] != i32::MAX && land_z > climate.ela_z[i])
            .count();
        let share = under_ice as f64 * 100.0 / land as f64;
        let mut lines: Vec<i32> = climate
            .ela_z
            .iter()
            .copied()
            .filter(|&e| e != i32::MAX && e != i32::MIN)
            .collect();
        lines.sort_unstable();
        println!(
            "G-ICE on an Earth-like charter: the ice covers {share:.2} % of the land ({under_ice} of {land} nodes); the snowline over {} nodes reads lowest {:.0} m, median {:.0} m, highest {:.0} m",
            lines.len(),
            f64::from(*lines.first().unwrap_or(&0)) / f64::from(Z_STEPS_PER_M),
            f64::from(lines[lines.len() / 2]) / f64::from(Z_STEPS_PER_M),
            f64::from(*lines.last().unwrap_or(&0)) / f64::from(Z_STEPS_PER_M)
        );
        assert!(
            (4.0..=25.0).contains(&share),
            "the ice covers {share:.2} % of an Earth-like charter's land; Earth carries about 10 %. The snowline over {} nodes: lowest {:.0} m, median {:.0} m, highest {:.0} m",
            lines.len(),
            f64::from(*lines.first().unwrap_or(&0)) / f64::from(Z_STEPS_PER_M),
            f64::from(lines[lines.len() / 2]) / f64::from(Z_STEPS_PER_M),
            f64::from(*lines.last().unwrap_or(&0)) / f64::from(Z_STEPS_PER_M)
        );
        // AND THE LINE ITSELF: at the equator Earth's snowline stands four to five kilometres up,
        // and the law must put it there or the share above is a coincidence.
        let mut equator = i32::MAX;
        for node in 0..lattice.node_count() as u32 {
            let d = dir_of(lattice.direction(node));
            if d[POLE_AXIS].to_f64().abs() < 0.05 {
                equator = equator.min(climate.ela_z[node as usize]);
            }
        }
        let equator_m = f64::from(equator) / f64::from(Z_STEPS_PER_M);
        assert!(
            (2_000.0..=7_000.0).contains(&equator_m),
            "the equatorial snowline stands at {equator_m:.0} m; Earth's is 4 500 to 5 000"
        );
    }

    /// Earth's tilt gives `s₂ ≈ −0.48`; no tilt −0.625; the sign turns past 54.7°.
    #[test]
    fn the_insolation_coefficient_on_earths_tilt() {
        let earth = insolation_s2(gf(0.917_5)).to_f64();
        assert!((earth + 0.477).abs() < 0.005, "{earth}");
        assert_eq!(insolation_s2(Gf::ONE), gf(-0.625));
        assert!(insolation_s2(gf(0.5)).to_f64() > 0.0);
        assert_eq!(legendre_p2(Gf::ONE), Gf::ONE);
        assert_eq!(legendre_p2(Gf::ZERO), -Gf::HALF);
    }

    /// The energy balance on Earth's numbers: about 25 K of contrast (a 38 K pole-to-equator
    /// difference), and four times more with no air to carry the heat.
    #[test]
    fn the_latitude_contrast_on_earth_and_without_air() {
        let earth = latitude_contrast_k(Gf::ONE, gf(0.3), gf(-0.477), gf(EARTH_P_SURF_PA)).to_f64();
        assert!((earth + 25.0).abs() < 1.0, "{earth}");
        let airless = latitude_contrast_k(Gf::ONE, gf(0.3), gf(-0.477), Gf::ZERO).to_f64();
        assert!((airless + 59.8).abs() < 1.0, "{airless}");
    }

    /// The lapse rate on Earth's air: 6.5 K/km; none without air.
    #[test]
    fn the_lapse_rate_on_earths_air() {
        let earth = lapse_rate_k_m(9_810, Some(7_413)).to_f64() * 1_000.0;
        assert!((earth - 6.5).abs() < 0.1, "{earth}");
        assert_eq!(lapse_rate_k_m(9_810, None), Gf::ZERO);
    }

    /// The circulation: Earth's day, radius and scale height give three cells; a slow spin one;
    /// no air one; a day of zero one; a fast spin six.
    #[test]
    fn the_circulation_from_the_spin() {
        assert_eq!(
            circulation(6.371e6, 9_810, Some(8_400), Some(86_164)).cells,
            3
        );
        assert_eq!(
            circulation(6.371e6, 9_810, Some(8_400), Some(86_164 * 100)).cells,
            1
        );
        assert_eq!(circulation(6.371e6, 9_810, None, Some(86_164)).cells, 1);
        assert_eq!(circulation(6.371e6, 9_810, Some(8_400), None).cells, 1);
        assert_eq!(circulation(6.371e6, 9_810, Some(8_400), Some(0)).cells, 1);
        assert_eq!(
            circulation(6.371e6, 9_810, Some(8_400), Some(3_600)).cells,
            CELLS_MAX
        );
        assert_eq!(
            circulation(6.371e6, 9_810, Some(8_400), Some(120_630)).cells,
            2
        );
    }

    /// The position in a three-cell hemisphere: the equator rises, 30° sinks, 60° rises, the
    /// pole sinks; the belt factor is wet at the rising branch and dry at the sinking one.
    #[test]
    fn the_cell_position_and_the_belt() {
        let three = Circulation { cells: 3 };
        assert_eq!(cell_position(three, Gf::ZERO), (0, Gf::ZERO));
        let (k, u) = cell_position(three, gf(0.5));
        assert_eq!(k, 1);
        assert_eq!(u, Gf::ONE);
        let (k, u) = cell_position(three, gf(0.866_025_4));
        assert_eq!(k, 2);
        assert_eq!(u, Gf::ZERO);
        let (k, u) = cell_position(three, Gf::ONE);
        assert_eq!(k, 2);
        assert_eq!(u, Gf::ONE);
        assert_eq!(belt_factor(Gf::ZERO), gf(BELT_RISING_SHARE));
        assert_eq!(belt_factor(Gf::ONE), gf(BELT_SINKING_SHARE));
        let (k, u) = cell_position(Circulation { cells: 1 }, gf(0.3));
        assert_eq!(k, 0);
        assert!((u.to_f64() - 0.3).abs() < 1e-12);
    }

    /// Tetens at 0 °C and at Earth's mean; the supply at Earth's mean is Earth's mean rain and
    /// it grows with the temperature.
    #[test]
    fn the_vapour_pressure_and_the_supply() {
        let freezing = saturation_vapour_pa(gf(FREEZING_K)).to_f64();
        assert!((freezing - 610.78).abs() < 0.01, "{freezing}");
        let mean = saturation_vapour_pa(gf(288.0)).to_f64();
        assert!((mean - 1_700.0).abs() < 20.0, "{mean}");
        assert_eq!(
            supply_mm_yr(gf(EARTH_MEAN_SURFACE_K)),
            gf(EARTH_MEAN_RAIN_MM_YR)
        );
        assert!(supply_mm_yr(gf(300.0)) > supply_mm_yr(gf(280.0)));
    }

    /// The lift: a rise doubles the rain at the calibration's slope, a fall halves it, both bounded.
    #[test]
    fn the_orographic_lift_and_shadow() {
        assert_eq!(lift_factor(gf(1_000.0), gf(8_000.0)), Gf::TWO);
        assert_eq!(lift_factor(gf(-1_000.0), gf(8_000.0)), Gf::HALF);
        assert_eq!(lift_factor(gf(8_000.0), gf(8_000.0)), gf(LIFT_MAX));
        assert_eq!(lift_factor(gf(-16_000.0), gf(8_000.0)), gf(LIFT_MIN));
        assert_eq!(lift_factor(Gf::ZERO, gf(8_000.0)), Gf::ONE);
    }

    /// ★ THE FROST LINE, THE SEASON AND THE SUMMER ELA (ruling B2 step 2). The line is a SUMMER
    /// line: it stands over the annual freezing line by the season's own warmth, and it stands
    /// LOWER in a wet climate than in a dry one, because Ohmura's 70 glaciers say a wet glacier
    /// keeps its snow at a warmer summer. The saturated words stand where there is no lapse.
    #[test]
    fn the_frost_line_the_season_and_the_summer_ela() {
        // Earth's own numbers: 238 W/m² absorbed, a 23.44° tilt, a year of 31.56 Ms, 71 % wet.
        let capacity = seasonal_capacity_j_m2_k(gf(0.71), 31_556_926);
        let earth = ClimateLaws {
            t_mean_k: gf(288.0),
            t2_k: gf(-25.0),
            lapse_k_m: gf(0.006_5),
            circulation: Circulation { cells: 3 },
            rains: true,
            absorbed_w_m2: gf(238.0),
            s1: insolation_s1(gf(0.917_5)),
            seasonal_damping_w_m2_k: seasonal_damping_w_m2_k(
                gf(EARTH_P_SURF_PA),
                capacity,
                31_556_926,
            ),
        };
        // ★ THE COEFFICIENT NORTH & COAKLEY REPORT: `s₁ = 0.796` at Earth's tilt.
        assert!(
            (earth.s1.to_f64() - 0.796).abs() < 0.001,
            "{}",
            earth.s1.to_f64()
        );
        // The season is nothing at the equator and greatest at the pole, and the ocean's heat
        // store holds it to a few kelvin — never the fifty the bare radiation balance would give.
        assert_eq!(seasonal_amplitude_k(&earth, Gf::ZERO), Gf::ZERO);
        let polar = seasonal_amplitude_k(&earth, Gf::ONE).to_f64();
        assert!((3.0..12.0).contains(&polar), "{polar}");
        // The equator stands at 300.5 K: freezing 4 200 m up.
        let frost = frost_line_z(&earth, Gf::ZERO, 0);
        let frost_m = f64::from(frost) / f64::from(Z_STEPS_PER_M);
        assert!((frost_m - 4_207.7).abs() < 1.0, "{frost_m}");
        // The pole stands at 263 K: freezing 1 561 m under the ground.
        assert!(frost_line_z(&earth, Gf::ONE, 0) < 0);
        // ★ OHMURA'S CURVE, at the three points the paper itself names.
        assert!((ohmura_ela_summer_c(950).to_f64() - 1.0).abs() < 0.05);
        assert!((ohmura_ela_summer_c(350).to_f64() + 1.03).abs() < 0.05);
        assert!((ohmura_ela_summer_c(2_000).to_f64() - 4.07).abs() < 0.05);
        // The equator has no season, so its line is the frost line shifted by Ohmura's threshold
        // alone: a dry line stands HIGHER than a wet one.
        let dry = ela_z(&earth, Gf::ZERO, 200, 0);
        let wet = ela_z(&earth, Gf::ZERO, 3_000, 0);
        assert!(dry > wet, "{dry} {wet}");
        assert!(
            dry > frost,
            "a dry glacier needs a colder summer than freezing"
        );
        // The datum carries the whole line with it, exactly.
        assert_eq!(ela_z(&earth, Gf::ZERO, 200, 1_600), dry + 1_600);
        assert_eq!(frost_line_z(&earth, Gf::ZERO, 1_600), frost + 1_600);
        let airless = ClimateLaws {
            lapse_k_m: Gf::ZERO,
            ..earth
        };
        assert_eq!(frost_line_z(&airless, Gf::ZERO, 0), i32::MAX);
        assert_eq!(frost_line_z(&airless, Gf::ONE, 0), i32::MIN);
        assert_eq!(ela_z(&airless, Gf::ZERO, 0, 0), i32::MAX);
        assert_eq!(ela_z(&airless, Gf::ONE, 0, 0), i32::MIN);
        // A body with no year stores no heat, so its season is the radiative one — bigger.
        assert_eq!(seasonal_capacity_j_m2_k(gf(0.71), 0), Gf::ZERO);
        let bare = seasonal_damping_w_m2_k(gf(EARTH_P_SURF_PA), Gf::ZERO, 0);
        assert!(bare < earth.seasonal_damping_w_m2_k);
        assert_eq!(
            seasonal_amplitude_k(
                &ClimateLaws {
                    seasonal_damping_w_m2_k: Gf::ZERO,
                    ..earth
                },
                Gf::ONE
            ),
            Gf::ZERO
        );
        let cold = ClimateLaws {
            t_mean_k: gf(200.0),
            ..earth
        };
        // A cold body with air: the equator's ground stands at 212.5 K, so the frost line lies
        // 9 331 m under the ground — a finite negative height, not the saturated word.
        let cold_frost = frost_line_z(&cold, Gf::ZERO, 0);
        assert!(cold_frost < 0);
        assert_ne!(cold_frost, i32::MIN);
        let cold_m = f64::from(cold_frost) / f64::from(Z_STEPS_PER_M);
        assert!((cold_m + 9_330.8).abs() < 1.0, "{cold_m}");
        // A tilt of zero has no season at all; a tilt of ninety degrees has the most there is.
        assert_eq!(insolation_s1(Gf::ONE), Gf::ZERO);
        assert_eq!(insolation_s1(Gf::ZERO), Gf::TWO);
    }

    /// The aridity byte: Earth's mean is humid, no rain is hyper-arid, a desert is in between.
    #[test]
    fn the_aridity_byte() {
        assert_eq!(aridity_q8(gf(288.0), 3_000), 0);
        assert_eq!(aridity_q8(gf(288.0), 0), 255);
        assert_eq!(aridity_q8(gf(288.0), 10), 255);
        let desert = aridity_q8(gf(295.0), 100);
        assert!(desert > 128, "{desert}");
        assert!(desert < 255, "{desert}");
        let mean = aridity_q8(gf(288.0), 990);
        assert!(mean > 0, "{mean}");
        assert!(mean < 128, "{mean}");
    }

    /// The surface wind: from the east on the equator (it blows west), from the west at 45° on a
    /// three-cell body, the zero vector at the pole; its upwind slot on the moon's lattice names a
    /// real neighbour whose offset points against the wind.
    #[test]
    fn the_surface_wind_and_the_upwind_slot() {
        let three = Circulation { cells: 3 };
        let equator = surface_wind(three, [Gf::ONE, Gf::ZERO, Gf::ZERO]);
        assert!(equator[1] < Gf::ZERO, "the trades blow west: {equator:?}");
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mid = surface_wind(three, [gf(s), Gf::ZERO, gf(s)]);
        assert!(mid[1] > Gf::ZERO, "the westerlies blow east: {mid:?}");
        assert!(mid[2] > Gf::ZERO, "toward the storm belt: {mid:?}");
        let south = surface_wind(three, [gf(s), Gf::ZERO, gf(-s)]);
        assert!(
            south[2] < Gf::ZERO,
            "toward the southern storm belt: {south:?}"
        );
        let pole = surface_wind(three, [Gf::ZERO, Gf::ZERO, Gf::ONE]);
        assert_eq!(pole, [Gf::ZERO; 3]);
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let node = lattice.index(vd_seed::bend::Face::PosX, 20, 20);
        let d = dir_of(lattice.direction(node));
        let wind = surface_wind(three, d);
        let slot = upwind_slot(&lattice, node, d, wind);
        let m = lattice.neighbours(node)[usize::from(slot)];
        assert_ne!(m, NO_NODE);
        let dm = dir_of(lattice.direction(m));
        assert!(dot([dm[0] - d[0], dm[1] - d[1], dm[2] - d[2]], wind) < Gf::ZERO);
        // A corner node's missing slot is skipped; the zero wind takes the first slot.
        let corner = lattice.index(vd_seed::bend::Face::PosX, 0, 0);
        let dc = dir_of(lattice.direction(corner));
        assert_eq!(upwind_slot(&lattice, corner, dc, [Gf::ZERO; 3]), 1);
        assert_eq!(STENCIL[0], (-1, -1));
    }

    /// ★ THE DRIVER ON THE HOME MOON: airless and dry — no rain, no wind, no lapse; the poles
    /// colder than the equator; frozen everywhere the equilibrium profile is under freezing.
    #[test]
    fn the_moons_climate_is_airless() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let words = home_moon_solve_words();
        let z = vec![0i32; lattice.node_count()];
        let c = climate(&moon, &lattice, &words, &z, None);
        assert_eq!(c.rain_mm_yr.len(), lattice.node_count());
        assert!(c.rain_mm_yr.iter().all(|&r| r == 0));
        assert!(c.upwind.iter().all(|&u| u == NO_WIND));
        assert!(c.aridity_q8.iter().all(|&a| a == 255));
        let laws = climate_laws(&moon, &words, Gf::ZERO);
        assert_eq!(laws.lapse_k_m, Gf::ZERO);
        assert!(!laws.rains);
        assert_eq!(laws.circulation.cells, 1);
        let equator = lattice.index(vd_seed::bend::Face::PosX, 34, 34);
        let pole = lattice.index(vd_seed::bend::Face::PosZ, 34, 34);
        assert!(c.temperature_dk[equator as usize] > c.temperature_dk[pole as usize]);
        assert!(c.frost_z.iter().all(|&f| f == i32::MAX || f == i32::MIN));
        assert!(c.frost_z.contains(&i32::MAX));
        assert!(c.frost_z.contains(&i32::MIN));
        // ★ THE SUMMER LINE ON A DRY BODY stands at or over the annual freezing line: there is no
        // rain, so Ohmura's curve asks for a summer colder than freezing before snow can last.
        assert!(c.ela_z.iter().zip(&c.frost_z).all(|(&e, &f)| e >= f));
        assert!(c.ela_z.contains(&i32::MAX));
    }

    /// ★ THE HOME PLANET'S WORDS ON THE MOON'S LATTICE (a stated neighbourhood): it rains
    /// everywhere between the floor and the ceiling, the tropics are wetter than the 30° belt, a
    /// windward node is wetter than its lee, the temperature falls with height, and the ELA stands
    /// at or over the frost line.
    #[test]
    fn the_home_planets_climate_on_the_moons_lattice() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let words = home_solve_words();
        let land = initial_land(&moon, &lattice, &words.land());
        let c = climate(&moon, &lattice, &words, &land.z, land.sea_z);
        let n = lattice.node_count();
        assert!(c.rain_mm_yr.iter().all(|&r| r >= P_MIN_MM_YR));
        assert!(c.rain_mm_yr.iter().all(|&r| r <= P_MAX_MM_YR));
        assert!(c.upwind.iter().all(|&u| u < 8));
        let laws = climate_laws(&moon, &words, Gf::HALF);
        assert!(laws.rains);
        assert!(laws.lapse_k_m > Gf::ZERO);
        // The belt: the moon's small radius and the home planet's day give ONE cell (Held–Hou), so
        // the equator rises and the pole sinks — the rain's mean near the equator against the polar
        // cap, on a flat field.
        let flat = vec![0i32; n];
        let f = climate(&moon, &lattice, &words, &flat, None);
        let mean_at = |lo: f64, hi: f64| {
            let mut sum = 0u64;
            let mut count = 0u64;
            for node in 0..n as u32 {
                let s = dir_of(lattice.direction(node))[POLE_AXIS].abs().to_f64();
                if s >= lo && s < hi {
                    sum += u64::from(f.rain_mm_yr[node as usize]);
                    count += 1;
                }
            }
            sum as f64 / count as f64
        };
        assert_eq!(laws.circulation.cells, 1);
        let tropics = mean_at(0.0, 0.1);
        let polar = mean_at(0.9, 1.01);
        assert!(
            tropics > 3.0 * polar,
            "tropics {tropics} against the polar cap {polar}"
        );
        // Windward against lee: a ridge one node wide across the wind on the flat field.
        let mut ridge = flat.clone();
        let node = lattice.index(vd_seed::bend::Face::PosX, 30, 30);
        ridge[node as usize] = 1_000 * Z_STEPS_PER_M;
        let r = climate(&moon, &lattice, &words, &ridge, None);
        let slot = r.upwind[node as usize];
        let up = lattice.neighbours(node)[usize::from(slot)];
        let (di, dj) = STENCIL[usize::from(slot)];
        let (_, i, j) = lattice.split(node);
        let lee = lattice.index(vd_seed::bend::Face::PosX, i - di, j - dj);
        assert!(
            r.rain_mm_yr[node as usize] > f.rain_mm_yr[node as usize],
            "the windward face gains"
        );
        assert!(
            r.rain_mm_yr[lee as usize] < f.rain_mm_yr[lee as usize],
            "the lee loses"
        );
        assert_eq!(r.rain_mm_yr[up as usize], f.rain_mm_yr[up as usize]);
        assert!(
            r.temperature_dk[node as usize] < f.temperature_dk[node as usize],
            "colder aloft"
        );
        // ★ THE LINE IS A SUMMER LINE (ruling B2 step 2): the season stands over the annual mean,
        // so a latitude with a season carries its line higher than the equator's, and a DRY node's
        // line stands over a WET one's, because Ohmura's 70 glaciers say a wet glacier keeps its
        // snow at a warmer summer than a dry one does.
        assert!(seasonal_amplitude_k(&laws, Gf::ZERO) == Gf::ZERO);
        assert!(seasonal_amplitude_k(&laws, Gf::ONE) > Gf::ZERO);
        let dry_line = ela_z(&laws, Gf::HALF, 200, 0);
        let wet_line = ela_z(&laws, Gf::HALF, 3_000, 0);
        assert!(dry_line > wet_line, "{dry_line} {wet_line}");
        assert_eq!(c.ela_z.len(), n);
    }

    /// ★ THE POTENTIAL EVAPORATION AND THE RUNOFF (ruling W11). The evaporation is Earth's own land
    /// mean at Earth's own mean temperature and climbs with the air's water capacity. The runoff is
    /// the Turc–Pike partition: a wet node hands most of its rain to the river, a node as wet as it
    /// is thirsty hands about three parts in ten, a desert hands almost nothing, and a node with no
    /// evaporation at all hands everything. Every reading is under the rain, and never negative.
    #[test]
    fn the_runoff_is_the_turc_pike_share_of_the_rain() {
        assert_eq!(
            pet_mm_yr(gf(EARTH_MEAN_SURFACE_K)).to_i64_floor(),
            EARTH_PET_MM_YR as i64
        );
        assert!(pet_mm_yr(gf(300.0)) > pet_mm_yr(gf(288.0)));
        // No evaporation at all: every millimetre leaves.
        assert_eq!(runoff_mm_yr(gf(1_000.0), Gf::ZERO), gf(1_000.0));
        assert_eq!(runoff_mm_yr(Gf::ZERO, gf(1_000.0)), Gf::ZERO);
        // As wet as it is thirsty: `1 − 1/√2` of the rain.
        let even = runoff_mm_yr(gf(1_000.0), gf(1_000.0)).to_i64_floor();
        assert_eq!(even, 292);
        // Twice as wet as thirsty, and a desert at a tenth.
        let wet = runoff_mm_yr(gf(2_000.0), gf(1_000.0)).to_i64_floor();
        assert_eq!(wet, 1_105);
        let desert = runoff_mm_yr(gf(100.0), gf(1_000.0));
        // ★ THE DEFECT THIS LAW REPLACED: the plain difference of the rain and the potential
        // evaporation is NEGATIVE over most of Earth's land, so it would dry every river. The
        // partition never is: a desert node still yields a little water.
        assert!(desert > Gf::ZERO);
        assert!(desert < gf(1.0));
        // The climate's own rows carry both, and the runoff never passes the rain.
        let body = home_moon();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let words = home_solve_words();
        let z = vec![0; lattice.node_count()];
        let c = climate(&body, &lattice, &words, &z, Some(0));
        assert_eq!(c.pet_mm_yr.len(), lattice.node_count());
        for i in 0..lattice.node_count() {
            assert!(c.runoff_mm_yr[i] <= c.rain_mm_yr[i]);
        }
        assert!(c.pet_mm_yr.iter().any(|&p| p > 0));
        assert!(c.runoff_mm_yr.iter().any(|&q| q > 0));
    }
}
