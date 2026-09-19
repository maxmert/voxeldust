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
    /// The height (sixteenths from the ladder radius) at which the air reaches freezing over the
    /// node's latitude; `i32::MAX` where it never freezes, `i32::MIN` where it is always frozen.
    pub frost_z: Vec<i32>,
    /// The equilibrium line altitude at the node: the frost line plus the dry-climate offset.
    pub ela_z: Vec<i32>,
    /// The stencil slot (0..8) of the neighbour the surface wind comes FROM; [`NO_WIND`] where
    /// there is no air.
    pub upwind: Vec<u8>,
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
/// ★ THE DRY-CLIMATE ELA OFFSET (03 §4.9, calibration the Andes): the ELA stands this far over the
/// frost line where no rain falls, and on it where the rain reaches `ELA_WET_MM_YR`.
pub const ELA_DRY_OFFSET_M: f64 = 1_200.0;
pub const ELA_WET_MM_YR: f64 = 2_000.0;
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

/// ★ THE FROST LINE at a latitude, in sixteenths: the height at which the air reaches freezing;
/// `i32::MAX` where the lapse is zero and the ground is warm, `i32::MIN` where it is zero and cold.
#[must_use]
pub fn frost_line_z(laws: &ClimateLaws, sin_lat: Gf) -> i32 {
    let at_ground = laws.t_mean_k + laws.t2_k * legendre_p2(sin_lat);
    let over = at_ground - Gf::from_f64(FREEZING_K);
    if laws.lapse_k_m <= Gf::ZERO {
        return if over >= Gf::ZERO { i32::MAX } else { i32::MIN };
    }
    let height_m = over / laws.lapse_k_m;
    (height_m * Gf::from_i64(i64::from(Z_STEPS_PER_M)))
        .clamp(
            Gf::from_i64(i64::from(i32::MIN) + 1),
            Gf::from_i64(i64::from(i32::MAX) - 1),
        )
        .to_i64_floor() as i32
}

/// ★ THE ELA over the frost line: `Δ_dry · (1 − P/P_wet)`, at least zero — a dry range keeps its
/// snow higher than its frost line, a wet one does not. Saturates where the frost line does.
#[must_use]
pub fn ela_z(frost_z: i32, rain_mm_yr: u32) -> i32 {
    if frost_z == i32::MAX || frost_z == i32::MIN {
        return frost_z;
    }
    let dryness = (Gf::ONE - Gf::from_i64(i64::from(rain_mm_yr)) / Gf::from_f64(ELA_WET_MM_YR))
        .greater(Gf::ZERO);
    let offset =
        (Gf::from_f64(ELA_DRY_OFFSET_M) * dryness * Gf::from_i64(i64::from(Z_STEPS_PER_M)))
            .to_i64_floor();
    (i64::from(frost_z) + offset).clamp(i64::from(i32::MIN) + 1, i64::from(i32::MAX) - 1) as i32
}

/// ★ THE ARIDITY BYTE: the potential evaporation (Earth's land mean scaled by the air's water
/// capacity) over the rain, on the UNEP classes' logarithmic axis — 0 at a ratio of 0.5 and under
/// (humid), 255 at 20 and over (hyper-arid). No rain at all is hyper-arid.
#[must_use]
pub fn aridity_q8(t_k: Gf, rain_mm_yr: u32) -> u8 {
    if rain_mm_yr == 0 {
        return 255;
    }
    let pet = Gf::from_f64(EARTH_PET_MM_YR) * saturation_vapour_pa(t_k)
        / saturation_vapour_pa(Gf::from_f64(EARTH_MEAN_SURFACE_K));
    let ratio = pet / Gf::from_i64(i64::from(rain_mm_yr));
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
pub fn climate_laws(body: &BodyDefinition, words: &SolveWords) -> ClimateLaws {
    let t_mean_k =
        Gf::from_i64(i64::from(words.t_surface_mk.unwrap_or(words.t_eq_mk))) / Gf::from_i64(1_000);
    let obliquity_cos =
        Gf::from_i64(i64::from(words.obliquity_cos_q1024.unwrap_or(1_024))) / Gf::from_i64(1_024);
    let p_surf = Gf::from_i64(i64::from(words.p_surf_pa.unwrap_or(0)));
    let t2_k = latitude_contrast_k(
        Gf::from_i64(i64::from(words.insolation_q12)) / Gf::from_i64(4_096),
        Gf::from_i64(i64::from(words.bond_albedo_q12)) / Gf::from_i64(4_096),
        insolation_s2(obliquity_cos),
        p_surf,
    );
    ClimateLaws {
        t_mean_k,
        t2_k,
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

/// ★ THE CLIMATE over `lattice` for the relief `z` as it stands. `sea_z` is not read: the supply is
/// a temperature law alone (no fetch, no continental term — a stated limit of the O(1) form).
#[must_use]
pub fn climate(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &SolveWords,
    z: &[i32],
    sea_z: Option<i32>,
) -> Climate {
    let _ = sea_z;
    let laws = climate_laws(body, words);
    let n = lattice.node_count();
    let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let mut out = Climate {
        temperature_dk: Vec::with_capacity(n),
        rain_mm_yr: Vec::with_capacity(n),
        aridity_q8: Vec::with_capacity(n),
        frost_z: Vec::with_capacity(n),
        ela_z: Vec::with_capacity(n),
        upwind: Vec::with_capacity(n),
    };
    for node in 0..n as u32 {
        let i = node as usize;
        let d = dir_of(lattice.direction(node));
        let sin_lat = d[POLE_AXIS];
        let height_m = Gf::from_i64(i64::from(z[i])) / steps;
        let t_k = temperature_k(&laws, sin_lat, height_m);
        let (rain, slot) = if laws.rains {
            let wind = surface_wind(laws.circulation, d);
            let slot = upwind_slot(lattice, node, d, wind);
            let m = lattice.neighbours(node)[usize::from(slot)];
            let rise_m = Gf::from_i64(i64::from(z[i]) - i64::from(z[m as usize])) / steps;
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
        let frost = frost_line_z(&laws, sin_lat);
        out.temperature_dk.push(
            (t_k * Gf::from_f64(10.0) + Gf::HALF)
                .to_i64_floor()
                .clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16,
        );
        out.rain_mm_yr.push(rain);
        out.aridity_q8.push(aridity_q8(t_k, rain));
        out.frost_z.push(frost);
        out.ela_z.push(ela_z(frost, rain));
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

    /// The frost line and the ELA: over the frost line in a dry climate, on it in a wet one, and
    /// the saturated words where the lapse is zero.
    #[test]
    fn the_frost_line_and_the_ela() {
        let earth = ClimateLaws {
            t_mean_k: gf(288.0),
            t2_k: gf(-25.0),
            lapse_k_m: gf(0.006_5),
            circulation: Circulation { cells: 3 },
            rains: true,
        };
        // The equator stands at 300.5 K: freezing 4 200 m up.
        let frost = frost_line_z(&earth, Gf::ZERO);
        let frost_m = f64::from(frost) / f64::from(Z_STEPS_PER_M);
        assert!((frost_m - 4_207.7).abs() < 1.0, "{frost_m}");
        // The pole stands at 263 K: freezing 1 561 m under the ground.
        assert!(frost_line_z(&earth, Gf::ONE) < 0);
        assert_eq!(ela_z(frost, 2_000), frost);
        assert_eq!(ela_z(frost, 4_000), frost);
        assert_eq!(ela_z(frost, 0), frost + 1_200 * Z_STEPS_PER_M);
        assert_eq!(ela_z(frost, 1_000), frost + 600 * Z_STEPS_PER_M);
        let airless = ClimateLaws {
            lapse_k_m: Gf::ZERO,
            ..earth
        };
        assert_eq!(frost_line_z(&airless, Gf::ZERO), i32::MAX);
        assert_eq!(frost_line_z(&airless, Gf::ONE), i32::MIN);
        assert_eq!(ela_z(i32::MAX, 0), i32::MAX);
        assert_eq!(ela_z(i32::MIN, 0), i32::MIN);
        let cold = ClimateLaws {
            t_mean_k: gf(200.0),
            ..earth
        };
        // A cold body with air: the equator's ground stands at 212.5 K, so the frost line lies
        // 9 331 m under the ground — a finite negative height, not the saturated word.
        let cold_frost = frost_line_z(&cold, Gf::ZERO);
        assert!(cold_frost < 0);
        assert_ne!(cold_frost, i32::MIN);
        let cold_m = f64::from(cold_frost) / f64::from(Z_STEPS_PER_M);
        assert!((cold_m + 9_330.8).abs() < 1.0, "{cold_m}");
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
        let laws = climate_laws(&moon, &words);
        assert_eq!(laws.lapse_k_m, Gf::ZERO);
        assert!(!laws.rains);
        assert_eq!(laws.circulation.cells, 1);
        let equator = lattice.index(vd_seed::bend::Face::PosX, 34, 34);
        let pole = lattice.index(vd_seed::bend::Face::PosZ, 34, 34);
        assert!(c.temperature_dk[equator as usize] > c.temperature_dk[pole as usize]);
        assert!(c.frost_z.iter().all(|&f| f == i32::MAX || f == i32::MIN));
        assert!(c.frost_z.contains(&i32::MAX));
        assert!(c.frost_z.contains(&i32::MIN));
        assert_eq!(c.ela_z, c.frost_z);
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
        let laws = climate_laws(&moon, &words);
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
        for k in 0..n {
            assert!(c.ela_z[k] >= c.frost_z[k]);
        }
    }
}
