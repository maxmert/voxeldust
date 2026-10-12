//! ★ THE SOLVE — its CORE (the landform arc, slice 8c stage C1; `slice_8c_design.md` §4; the
//! investigation's `03_erosion_rivers.md` §4.3–4.6, §5).
//!
//! Once per body, on the macro lattice, a short list of passes turns a starting surface into an
//! eroded one: the water is routed downhill (D8), the pits are filled to their spill level (the
//! priority flood), the flats are resolved by a distance field, the rain is accumulated downstream
//! into a DISCHARGE, and the stream power law cuts every node toward its receiver's base level —
//! one implicit sweep, one division per node, unconditionally stable. Integers between passes;
//! the one float is the chord's square root under the fence, floored to a metre before it is
//! compared. Every order is a stated order over integers, so two hosts cannot disagree, and the
//! solve runs on ONE thread, off the tick (03 §5.3).
//!
//! **What C1 leaves out, by name** (the design's C2–C5): the initial land from plates and isostasy
//! (the starting surface here is the recipe's own coarse relief at the node's rung), the climate
//! (the rain is [`P_MIN_MM_YR`] everywhere — 03 §4.5's stated fallback: the rivers are right in
//! shape and wrong in size), the uplift, the sediment budget, the flexural rebound, the talus, the
//! ice, the craters, the coast and the artifact. C1 exists to MEASURE the core's wall time and
//! memory on THE world's bodies before those are built.
//!
//! **Example.** On the home planet's moon (27 744 nodes) the flood seeds the heap with every node
//! under the drawn sea, pops them lowest first, and raises each pit to the level at which its water
//! could leave. The rain that falls on a highland node then runs node to node down the receivers
//! to the sea, and the sweep lowers each node toward the level of the node below it, harder where
//! more water passes. A lake node is skipped, so the hole the flood found stays a lake.

// ★ A SOLVE MAY DIVIDE (ruling F7's rule is about the KERNELS a card runs). The implicit update is
// one division per node on the CPU, single-threaded, off the tick, never a GPU kernel (03 §5.3);
// the quotient is computed with `div_euclid` on 128-bit words, which is integer-exact on every host.
#![allow(
    clippy::integer_division,
    clippy::modulo_arithmetic,
    reason = "the solve runs on the CPU once per body, never in a kernel; integer-exact on every host"
)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use vd_recipe::Gi;
use vd_recipe::bend::DIR_BITS;

use crate::body::BodyDefinition;
use crate::gf::Gf;
use crate::height::height;
use crate::macro_lattice::{MacroLattice, NO_NODE, chord_between};
use crate::units::LENGTH_BITS;

/// The macro height's unit: sixteenths of a metre, from the ladder radius. A planet's relief is at
/// most 18 000 m — 288 000 steps, four decimal orders under the word.
pub const Z_STEPS_PER_M: i32 = 16;
/// The shift from the recipe's length word (gap steps of 1/128 m at [`LENGTH_BITS`] fraction bits)
/// to sixteenths: `128 / 16 = 2³`.
const Z_SHIFT: u32 = LENGTH_BITS + 3;
/// The water level's word for a node that holds no standing water.
pub const DRY: i32 = i32::MIN;
/// The rain's floor, in millimetres a year: a hyper-arid basin still carries a river's SHAPE, and a
/// logarithm cannot hold zero (03 §4.5). Until the climate lands (C3), this is the rain everywhere.
pub const P_MIN_MM_YR: u32 = 1;
/// The rain's ceiling, stated ONCE: Earth's wettest station, Mawsynram, is about 11 900 mm/yr.
pub const P_MAX_MM_YR: u32 = 10_000;
/// A body with no node under its sea has no outlet; the flood then seeds the heap with this many
/// lowest nodes by `(height, index)` — one per face, a stated world constant (03 §4.4).
pub const OUTLETS_WITHOUT_SEA: usize = 6;

// ★ THE SCHEDULE's RESOLUTION (part of the world tag, so a change is a new world; 03 §5.5). It is how finely the age
// is stepped, never a dial: the physical dial is the age, and it is the system's own. ★ PHASE 2 (ruling B39: THE WORLD
// CONVERGES IN TIME; B47: the gate's tolerance) — the passes stand on THE MIXED AXIS, short at the present and long in
// the deep past, and the readings at this axis and at the axis twice as fine agree within the stated tolerance. ★ B56
// (2026-10-05, the owner: "414 steps"): under B54's noise rule the gate FAILED on the 214-pass axis (half a million
// years last, 25 million longest: the old plains 11–17 % high, land over 1 km 3 points) and PASSED on this one against
// the axis twice as fine (every share within 0.4 points, every height within 4.5 %; the census's gate section item 14).
/// ★ THE MIXED AXIS's last pass, years (rulings B39's gate, B56): a quarter of a million years at the present.
pub const MIXED_T_MIN_YR: u64 = 250_000;
/// ★ THE MIXED AXIS's longest pass, years: twelve and a half million years in the deep past.
pub const MIXED_T_MAX_YR: u64 = 12_500_000;
/// ★ THE MIXED AXIS's ratio between neighbouring recent passes: each pass is this many times the next, until a pass
/// reaches [`MIXED_T_MAX_YR`].
pub const MIXED_RATIO: f64 = 1.25;
// ★ THE CLIMATE AND THE RUNNING SEA EVERY PASS (ruling B69 item 1; B68, plate 5's cause): re-solved every twentieth pass,
// the sea stood still from 242 to 17 million years ago on the 414-pass axis while the young belts rose, and ONE late
// re-solve raised it 53–60 m — 17 million years ago on one axis, 8.6 on the other — so plate 5's old plains read 303 m
// at 414 passes and 263 m at 828 (−14.3 %). Re-solved every pass, the sea follows the ranges as they rise: 269 → 257 m
// (−9.1 %), and the gate passes on every reading. No spacing is left to state: the climate and the sea are part of
// every pass, like the flood below. Its price: the climate is about a fifth of a first solve.
// ★ THE FLOOD, THE FLATS AND THE RECEIVERS EVERY PASS (2026-09-22, the lake census): at ten, a hollow the uplift, the
// rebound or the talus made between two floods was cut by nobody, and the spill of a lake was re-found four times in five
// billion years — so the pits GREW from 98 k after the first ten passes to 146 k at the end. A river breaches a hollow; a
// flood every pass lets it.
/// The talus relaxations the age holds, spread over its passes (C3).
pub const TALUS_PASSES: u32 = 8;
/// ★ THE SWEEPS A PASS RUNS on its one routing, each a share of the pass's time, with a rebound after each (ruling
/// B41 item 1: a resolution the convergence gate reads; B45: the rebound's spacing settles at 12.5 Myr or finer).
/// ★ EIGHT (ruling B62 item 1): the joint law's one implicit step per sweep under-erodes decaying high ground when the
/// step is long — MEASURED on the older landscape's relief, plate 5's mean read 207 / 194 / 185 m with 2 / 4 / 8 sweeps
/// on the 414-pass axis against 188 m on the 828-pass axis, for about 15 % more time.
pub const SUB_SWEEPS: u32 = 8;
/// ★ THE ROUTINGS A PASS RUNS (ruling B65 item 1): the water routed — the flood, the receivers, the budget's own paths
/// and its closed hollows — this many times a pass, each routing followed by its share of the pass's [`SUB_SWEEPS`]; the
/// uplift, the talus, the ice and the peak law stay once a pass. A resolution the convergence gate reads (B39): each
/// routing finds the basins and the water's ways afresh, so more routings let the rivers capture an old plain's hollows
/// sooner. MEASURED on generator 26 (the bench's routings counted apart from the passes): plate 13's mean read 263 /
/// 218–232 / 214–215 m at 414 / 828 / 1 656 routings over the age — 414 and 828 passes agreeing within a metre at the
/// same routing count — and its cut grew 7.7 % then 3.5 % with each doubling.
pub const ROUTINGS: u32 = 4;

/// ★ THE JOINT LAW (ruling B43 item 2, F49; Adams, Whipple, Forte, Heimsath & Hodges 2020, *Climate controls on
/// erosion in tectonically active landscapes*, Science Advances 6, eaaz3166, eq. 1b and its regional best fit):
/// `E = K_lp · Q · S^n`, `Q` the discharge — "the product of drainage area and mean upstream rainfall", m³/yr —
/// `K_lp = 2.2 × 10⁻⁹ m⁻²`, `n = 2.2`. Calibration body: 142 basins of Bhutan and central-eastern Nepal (¹⁰Be). The
/// rock's share reads against the crystalline basement, the calibration body's gneiss. Its authors: "we do not
/// suggest that this represents a universal n value" — a stated limit.
pub const JOINT_K_LP: f64 = 2.2e-9;
/// The joint law's slope exponent.
pub const JOINT_N: f64 = 2.2;
/// The joint law's implicit step solves `x − x₀ + c·(x/L)^n = 0` by Newton from `x₀` (convex and rising, so the steps
/// fall to the root without passing it) at most this many times, stopping where a step moves under
/// [`JOINT_TOLERANCE_M`].
pub const JOINT_NEWTON: usize = 60;
/// The Newton step under which the joint law's root stands, metres (a micrometre, far under a sixteenth).
pub const JOINT_TOLERANCE_M: f64 = 1.0e-6;
/// ★ THE REWORKING LAW's rate, per Gyr (ruling B45 item 2): 30.0 % of Earth's continental area is younger than 0.35 Ga
/// (Artemieva 2006, *Global 1° × 1° thermal model TC1 for the continental lithosphere*, Tectonophysics 416, Table 4),
/// so `λ = −ln(0.700)/0.35 Gyr` = 1.019 per Gyr; the law then gives 53.4 % younger than 0.75 Ga against the measured
/// 53.7 %, and 95.1 % younger than 2.95 Ga against 88.8 % (the old cratons survive longer — a stated limit). A plate's
/// raise time is `−ln(1 − u)/λ` for its own age share `u`.
pub const REWORK_RATE_PER_GYR: f64 = 1.019;
/// A node's erosion share of a pass under the landscape's time, in 1/65 536 (a whole pass is `ACTIVE_FULL`).
pub const ACTIVE_FULL: u32 = 1 << 16;
/// ★ THE PEAK LAW's rounds in one pass (limits as law, the belts' design L0): a fixed count, one answer on every host,
/// each a sweep in node order.
pub const PEAK_ROUNDS: usize = 256;
/// ★ THE FLEXURAL REBOUND's smoothing (03 §4.7): the removal is restricted up the pyramid to the
/// level whose node is nearest the flexural parameter, smoothed there by this many passes of the
/// nine-node mean (two passes spread over about 1.15 coarse nodes — the parameter itself), and
/// prolonged back to the nearest parent. A stated count, part of the world tag.
pub const SMOOTH_PASSES: u32 = 2;
/// ★ THE TALUS's move: half the excess leaves a node in one pass, split among the lower neighbours
/// in proportion to each one's excess — an explicit non-linear diffusion that cannot overshoot
/// (03 §4.8; the fraction at most one half).
pub const TALUS_FRACTION_DEN: u32 = 2;
/// The talus's tangent word: a tangent in 1/65536.
pub const TAN_BITS: u32 = 16;
/// ★ THE ANGLE OF REPOSE's two ends as tangents: loose dry rock stands at 35° (`tan = 0.70`, 8a's
/// `TAN_REPOSE`, calibration: scree), a wet soil-covered slope at 25° (`tan = 0.47`, calibration:
/// vegetated hillslopes' threshold). A node's tangent reads its aridity between them.
pub const TAN_REPOSE_DRY: f64 = 0.70;
pub const TAN_REPOSE_WET: f64 = 0.47;
/// ★ THE ICE's laws. A glacier's thickness under the perfect-plastic law (Nye 1952; Paterson
/// 1994, calibration: valley glaciers): `H = τ / (ρ_ice · g · S)` with the basal yield stress
/// `τ = 100 kPa` and the ice's density — thick on a gentle trunk, thin on a steep ridge, which IS
/// the pooling the design describes, in closed form.
pub const ICE_YIELD_STRESS_PA: f64 = 1.0e5;
pub const ICE_DENSITY_KGM3: f64 = 917.0;
/// ★ EGHOLM'S FIVE NUMBERS (Egholm, Nielsen, Pedersen & Lesemann 2009, *Glacial effects limiting
/// mountain height*, Nature 460, 884–887, <https://doi.org/10.1038/nature08263>; ruling B2 step 2).
/// The whole ice law is a mass balance against the ABLATION-SEASON temperature `Ts`, in metres a
/// year a kelvin — accumulate `−0.1·min(0, Ts)`, ablate `−0.15·max(0, Ts)` at the surface and
/// `−0.05` at a bed that stands over its melting point — and an erosion law `ė = kₑ·|u_s|` with
/// the exponent ONE, `kₑ` dimensionless and calibrated by Egholm so the mean glacial erosion stays
/// under a millimetre a year. Calibration body: Egholm's own Alpine and Andean ranges; the
/// observational half is that no peak on Earth stands more than about 1 500 m over its snowline.
///
/// ★ NO EXTENT IS EVER DRAWN. The balance decides where ice is: a node keeps ice only where the
/// ice its catchment delivers outlives the ablation there, so *"the snowline acts like a climatic
/// base level limiting the down-valley extent of glacial erosion"* and a small catchment stalls at
/// the line on its own.
pub const ICE_ACCUMULATION_M_YR_K: f64 = 0.1;
pub const ICE_ABLATION_M_YR_K: f64 = 0.15;
pub const ICE_BASAL_MELT_M_YR_K: f64 = 0.05;
pub const GLACIAL_EROSION_K: f64 = 1.0e-4;
/// ★ THE BED'S OWN TEMPERATURE: Earth's continental mean geothermal flux, W/m² (Pollack, Hurter &
/// Johnson 1993; Davies 2013 — the calibration body is Earth's continents), the thermal
/// conductivity of ice, W/(m·K) (Paterson 1994), and the pressure-melting gradient, K a metre of
/// ice (Paterson 1994, pure ice). COLD-BASED ICE DOES NOT ERODE, so a thin cold cap on a polar
/// plateau leaves the ground alone and a thick temperate trunk cuts it.
pub const GEOTHERMAL_FLUX_W_M2: f64 = 0.065;
pub const ICE_CONDUCTIVITY_W_M_K: f64 = 2.1;
pub const ICE_PRESSURE_MELT_K_PER_M: f64 = 8.7e-4;
/// ★ THE GLACIAL EPOCH the cut integrates over, years: an unrecoverable HISTORY of the body (ruling
/// T9 allows one), stated as Earth's Quaternary — the ice's cut never reads the tick (ruling T3).
pub const GLACIAL_EPOCH_YR: u64 = 2_600_000;
/// ★ THE COAST's band: the height over and under the sea the waves work, from a storm wave of
/// Earth's height scaled by the air's density over Earth's and by Earth's gravity over the body's
/// (a wind-driven wave's height goes as `ρ_air · U² / g`; calibration: Earth's 10 m storm wave).
pub const EARTH_STORM_WAVE_M: f64 = 10.0;
/// Earth's surface air density, kg/m³, the wave law's calibration.
pub const EARTH_AIR_DENSITY_KGM3: f64 = 1.225;

/// The facies bits a node carries out of the solve (the artifact's byte, C4).
pub const FACIES_SEA: u8 = 1;
pub const FACIES_LAKE: u8 = 2;
pub const FACIES_COAST: u8 = 4;
pub const FACIES_ICE: u8 = 8;

/// ★ STRAHLER'S BANDS of the hypsometric integral (Strahler 1952, the calibration of gate G-AGE):
/// a basin whose integral stands over 0.60 is in YOUTH (little cut), between 0.35 and 0.60 in
/// MATURITY (the balance of uplift and cutting), under 0.35 in OLD AGE (worn to its base). The
/// integral is `(mean − min) / (max − min)` of the basin's land heights, by area.
pub const HI_MATURE: (f64, f64) = (0.35, 0.60);
/// The bits a cached direction keeps: `[i32; 3]` at 30 fraction bits, so the direction table costs
/// twelve bytes a node; the chord it feeds is floored to a whole metre, and a step of 2⁻³⁰ is six
/// millimetres on the home planet.
const CACHE_BITS: u32 = 30;

/// ★ THE CHARTER WORDS THE SOLVE READS (slice 8c stage C3), as whole numbers stated by the body's
/// own realm (ruling V13 L12) — the subset of `vd_core::look::BodyCharter` the land, the climate, the
/// craters and the ice read, with the census's system age. The generator may name no core crate, so
/// the words are restated here field for field; `crates/bins/tests/home_body_pin.rs` proves the
/// home planet's and its moon's are the census's own, and the flag bits are the charter's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolveWords {
    /// The body's whole water inventory, whole km³ (zero on a dry body).
    pub water_km3: u64,
    /// The lithosphere's effective elastic thickness, whole metres.
    pub elastic_thickness_m: u32,
    /// Insolation relative to Earth's, in 1/4096.
    pub insolation_q12: u32,
    /// Equilibrium temperature, whole millikelvin.
    pub t_eq_mk: u32,
    /// Mean SURFACE temperature, whole millikelvin (the thermostat's), where stated.
    pub t_surface_mk: Option<u32>,
    /// Bond albedo, in 1/4096.
    pub bond_albedo_q12: u32,
    /// The atmosphere's mean molecular weight, in 1/256 atomic units; ABSENT on an airless body.
    pub mu_q8: Option<u32>,
    /// The atmosphere's isothermal scale height, whole metres; ABSENT on an airless body.
    pub scale_height_m: Option<u32>,
    /// Surface pressure, whole pascals; ABSENT on an airless body.
    pub p_surf_pa: Option<u32>,
    /// The grey greenhouse optical depth, in 1/4096.
    pub tau_ir_q12: Option<u32>,
    /// The rotation period, whole seconds.
    pub day_s: Option<u32>,
    /// The COSINE of the obliquity, in 1/1024.
    pub obliquity_cos_q1024: Option<i32>,
    /// The orbit's eccentricity, in 1/65536.
    pub ecc_q16: u32,
    /// The orbital period, whole seconds.
    pub year_s: u64,
    /// The charter's flag word ([`WORD_FLAG_HAS_AIR`] and its siblings).
    pub flags: u32,
    /// ★ THE EROSIONAL AGE in years: the system's own age from the census.
    pub age_yr: u64,
}

/// The charter flag: the body holds an atmosphere (`vd_core::look::CHARTER_FLAG_HAS_AIR`).
pub const WORD_FLAG_HAS_AIR: u32 = 1;
/// The charter flag: the body has a surface you can stand on.
pub const WORD_FLAG_SOLID_SURFACE: u32 = 2;
/// The charter flag: the body is tidally locked.
pub const WORD_FLAG_TIDALLY_LOCKED: u32 = 4;
/// The charter flag: the body has a sea.
pub const WORD_FLAG_HAS_SEA: u32 = 8;

impl SolveWords {
    /// The two words the initial land reads.
    #[must_use]
    pub const fn land(&self) -> crate::land::LandWords {
        crate::land::LandWords {
            water_km3: self.water_km3,
            elastic_thickness_m: self.elastic_thickness_m,
        }
    }

    /// Whether the body holds an atmosphere.
    #[must_use]
    pub const fn has_air(&self) -> bool {
        self.flags & WORD_FLAG_HAS_AIR != 0
    }

    /// Whether the body is tidally locked.
    #[must_use]
    pub const fn tidally_locked(&self) -> bool {
        self.flags & WORD_FLAG_TIDALLY_LOCKED != 0
    }
}

/// The schedule of one solve: the age the passes step through, the mixed axis that steps it, and the cadences.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// ★ THE EROSIONAL AGE in years: the SYSTEM'S OWN age from the census (the design's ask 3).
    pub age_yr: u64,
    /// The mixed axis's last pass and its longest pass, years ([`mixed_times`]).
    pub t_min_yr: u64,
    pub t_max_yr: u64,
    pub talus_passes: u32,
    pub sub_sweeps: u32,
    /// The routings a pass ([`ROUTINGS`]), at most one a sweep.
    pub routings: u32,
}

impl Schedule {
    /// The standard schedule for a body of a system `age_yr` old.
    #[must_use]
    pub const fn standard(age_yr: u64) -> Schedule {
        Schedule {
            age_yr,
            t_min_yr: MIXED_T_MIN_YR,
            t_max_yr: MIXED_T_MAX_YR,
            talus_passes: TALUS_PASSES,
            sub_sweeps: SUB_SWEEPS,
            routings: ROUTINGS,
        }
    }

    /// ★ WHETHER THE WATER IS ROUTED AGAIN before sweep `s` of a pass (ruling B65 item 1): the pass's [`Schedule::routings`]
    /// (at least one, at most one a sweep) share its sweeps in order — sweep `s` belongs to routing `s · R / S` — and a
    /// later routing starts where that number grows. The first routing stands before the pass's first sweep.
    ///
    /// **Example.** Eight sweeps and four routings: the water is routed before the pass, then again before sweeps 2, 4
    /// and 6.
    #[must_use]
    pub fn routes_before(&self, s: u32) -> bool {
        let sweeps = self.sub_sweeps.max(1);
        let routings = self.routings.clamp(1, sweeps);
        s.checked_sub(1)
            .is_some_and(|before| s * routings / sweeps > before * routings / sweeps)
    }

    /// The passes' boundaries in years from the solve's start ([`mixed_times`]).
    #[must_use]
    pub fn times(&self) -> Vec<u64> {
        mixed_times(self.age_yr, self.t_min_yr, self.t_max_yr)
    }
}

/// ★ THE MIXED AXIS's boundaries in years from the solve's start, `0` first and the age last (ruling B39's gate): from
/// the present backwards each pass is [`MIXED_RATIO`] times the next, the last `t_min` years, until a pass reaches
/// `t_max`; then even passes of `t_max` back to the start, the first taking the remainder, so the passes sum to the
/// age exactly. The lengths grow by the fence's own multiply and round half up, so every host states the same axis.
///
/// **Example.** The home planet's five billion years: the last pass a quarter of a million years, the one before
/// 312 500, then 390 625, … up to 12.5 million years about 18 passes back; 414 passes in all.
#[must_use]
pub fn mixed_times(age: u64, t_min: u64, t_max: u64) -> Vec<u64> {
    let mut lengths_from_present: Vec<u64> = Vec::new();
    let top = Gf::from_i64(t_max as i64);
    let mut length = Gf::from_i64(t_min as i64);
    let mut covered = 0u64;
    while covered < age {
        let l = ((length + Gf::HALF).floor().to_i64_floor() as u64)
            .min(t_max)
            .max(1)
            .min(age - covered);
        lengths_from_present.push(l);
        covered += l;
        length = (length * Gf::from_f64(MIXED_RATIO)).lesser(top);
    }
    let mut times = Vec::with_capacity(lengths_from_present.len() + 1);
    let mut t = 0u64;
    times.push(t);
    for l in lengths_from_present.into_iter().rev() {
        t += l;
        times.push(t);
    }
    times
}

/// What one routing pass found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteReport {
    /// Whether the sea seeded the flood (false: the lowest nodes did).
    pub sea_seeded: bool,
    /// The outlets the flood started from.
    pub outlets: usize,
    /// Nodes the flood RAISED — the lake nodes.
    pub raised: usize,
    /// The pits' volume: every raise summed, in steps (2026-09-22, the lake census).
    pub pit_steps: u64,
    /// Nodes routed by the flat's distance field rather than by a lower neighbour.
    pub flat: usize,
    /// Nodes the distance field never reached: a defect (gate G-DRAINAGE).
    pub undrained: usize,
    /// Nodes left out of the topological order: a cycle, a defect.
    pub cyclic: usize,
    /// ★ THE INLAND POCKETS the flood did NOT seed (2026-09-24, ruling B7 item 1): the bodies at or
    /// under the sea's level that no path joins to the ocean, and their nodes. The flood raises
    /// each one like any other hollow.
    pub pockets: usize,
    pub pocket_nodes: usize,
}

/// What one stream-power sweep did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SweepReport {
    /// Nodes cut by at least one step.
    pub lowered: usize,
    /// Lake nodes skipped, so the hole stays a lake.
    pub skipped_lake: usize,
    /// The deepest single cut, in steps.
    pub max_cut: i32,
    /// Every cut summed, in steps.
    pub total_cut: u64,
    /// ★ Lake nodes the sediment FILLED this pass (2026-09-22): the cut of the catchment, carried
    /// down the tree, laid in the hollow it drains into.
    pub filled: usize,
    /// The fill summed over those nodes, in steps.
    pub total_fill: u64,
    /// ★ THE ROCK'S WAY (the water's own paths only; sixteenths·m²): the rock that reached the sea, the rock a closed
    /// body full of rock passed over its spill ([`MacroSolve::deposit_basins`]), the rock dropped where the rivers' own
    /// way ends at an inland end no body holds or where a full body's every way is refused (`rock_stuck`; none is
    /// expected), and the level fill's own remainder under one sixteenth.
    pub rock_to_sea: u128,
    pub rock_over_spill: u128,
    pub rock_dropped: u128,
    pub rock_stuck: u128,
    pub rock_remainder: u128,
    /// The ways the rock past a full body's spill refused ([`DepositReport::refused`]).
    pub rock_refused: usize,
    /// ★ The closed bodies' water nodes the capture drained this sweep ([`MacroSolve::drain_to_saddles`]).
    pub drained: usize,
}

/// The joint law's words for one sweep ([`MacroSolve::sweep`]).
struct JointStep {
    dt_yr: Gf,
    steps_m: Gf,
    basement: Gf,
    k_lp: Gf,
    n_exp: Gf,
    tolerance: Gf,
}

/// One node's state as a sweep's core holds it ([`MacroSolve::sweep_group`]): its receiver is its place in the core's
/// own list.
#[derive(Clone, Copy)]
struct SweepCell {
    z: i32,
    z_flood: i32,
    pit: u32,
    ocean: bool,
    receiver: u32,
    chord: u32,
    discharge: u64,
    erodibility_q8: u32,
    active: u32,
    removed: u32,
    pending: i32,
    cut: u32,
}

impl SweepCell {
    /// The sweep's base level at this node as a receiver ([`MacroSolve::sweep_base`]'s rule, [`base_level`]).
    fn base(&self, sea_z: i32) -> i32 {
        base_level(self.z, self.z_flood, self.pit, self.ocean, sea_z)
    }
}

/// ★ THE SWEEP'S BASE LEVEL from a node's words (see [`MacroSolve::sweep_base`]): a lake at its flood level, the
/// connected ocean at the sea's level, every other node at its ground — never under the ground.
fn base_level(z: i32, z_flood: i32, pit: u32, ocean: bool, sea_z: i32) -> i32 {
    let level = if pit > 0 {
        z_flood
    } else if ocean {
        sea_z
    } else {
        DRY
    };
    z.max(level)
}

/// The sweep's work dealt to the cores ([`MacroSolve::basin_groups`]): valid while the routing's receivers and order
/// stand.
pub struct BasinGroups {
    lists: Vec<Vec<u32>>,
    skip: Vec<usize>,
    local: Vec<u32>,
    /// The list that sweeps each node ([`NO_NODE`] for a node no list sweeps: a read-only root, or a lone outlet).
    group: Vec<u32>,
}

/// What the deposit with the budget's closed hollows did ([`MacroSolve::deposit_basins`]): the nodes it raised and the
/// fill summed in steps, and the rock's way in sixteenths·m² (see [`SweepReport::rock_to_sea`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepositReport {
    pub filled: usize,
    pub total_fill: u64,
    pub to_sea: u128,
    pub over_spill: u128,
    pub dropped: u128,
    /// The part of `dropped` that found NO WAY: a full body whose every way is refused and whose parent's ring is no
    /// closed body.
    pub stuck: u128,
    pub remainder: u128,
    /// The ways refused because the water's way from their gate ends where the rest may not go: in a body the rest has
    /// already passed (the full body itself among them), at an inland end no body holds, or on a ring of receivers.
    pub refused: usize,
}

/// One core's share of the deposit's walk down the trees ([`MacroSolve::deposit_basins_grouped`]): the nodes it raised and
/// by how much, the rock it booked to each closed body, and its sums.
#[derive(Default)]
struct WalkPart {
    raised: Vec<(u32, i32)>,
    held: Vec<(u32, u128)>,
    out: DepositReport,
}

/// The deposit's work over the closed bodies ([`MacroSolve::deposit_basins`]): each body's representative after the
/// unions, the nodes it fills, the rock due to it, whether it stands at its spill (full of rock), and the bodies due to
/// fill.
struct RockWork {
    owner: Vec<u32>,
    nodes: Vec<Vec<u32>>,
    held: Vec<u128>,
    full: Vec<bool>,
    due: Vec<u32>,
}

/// A closed body's representative after the deposit's unions ([`MacroSolve::deposit_basins`]): a body that rose into
/// its parent's ring is the parent's.
fn representative(owner: &[u32], mut b: u32) -> u32 {
    while owner[b as usize] != b {
        b = owner[b as usize];
    }
    b
}

/// The whole solve's record, pass by pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SolveReport {
    pub routes: Vec<RouteReport>,
    pub sweeps: Vec<SweepReport>,
}

/// ★ THE FULL SOLVE's record (stage C3): what every pass found, and the gates' readings.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FullReport {
    pub routes: Vec<RouteReport>,
    pub sweeps: Vec<SweepReport>,
    /// The craters stamped, each in its own pass.
    pub craters: usize,
    /// ★ THE RECORD ITSELF (ruling B2 step 2's gate G-CRATER): every kept crater with its node,
    /// its width and the age of the impact that made it, so the census can count the wide ones and
    /// read the age histogram the resurfacing law is supposed to skew toward the recent.
    pub crater_record: Vec<crate::craters::Crater>,
    /// Every rebound: the level, the nodes lifted, the greatest lift.
    pub rebounds: Vec<(u32, usize, i32)>,
    /// Every talus pass: the nodes that shed, the worst excess before it.
    pub talus: Vec<(usize, i64)>,
    /// The ice: the nodes under ice at the end, the thickest ice in metres there, and the stepped ice's cut summed over
    /// the age (sixteenths × nodes).
    pub ice: (usize, f64, i64),
    /// The stepped ice's steps over the age (ruling B44 item 1), and of them the steps the one-year floor set (a
    /// reading: zero says the floor never acted).
    pub ice_steps: u32,
    pub ice_floored: u32,
    /// ★ THE BELTS (phase 2; [`crate::belts`]): the critical wedges, each node's kind and plateau mark, and the law's
    /// readings.
    pub belts: crate::belts::Belts,
    /// ★ THE LANDSCAPE'S TIME (ruling B45 item 2): each plate's raise time, years before the present.
    pub raise_yr: Vec<u64>,
    /// ★ THE PEAK LAW (limits as law): the collapses over the passes and their rounds, and the largest excess left over
    /// the cap after any pass, in sixteenths (zero: it settled every time).
    pub peak: (usize, i32),
    /// Gate G-AGE's reading: the integrals of every basin of at least [`AGE_GATE_MIN_NODES`]
    /// land nodes.
    pub integrals: Vec<BasinIntegral>,
    /// The sediment budget: basins with a deposit.
    pub deposits: usize,
    /// The coast band, sixteenths.
    pub coast_band: i32,
    /// ★ THE SEA RE-SOLVED (stage C5): its level in sixteenths over the final field, and the share
    /// of the globe's area under it (gate G-SEA's reading); `None` and zero on a dry body.
    pub sea_z: Option<i32>,
    pub ocean_share: f64,
    /// ★ THE LAKES' BUDGET (ruling W11): what the depression hierarchy and the water budget found.
    pub lakes: crate::lakes::LakeReport,
    /// ★ THE SEA BEFORE THE LAKES WERE TAKEN OUT OF THE INVENTORY, sixteenths (ruling W11 step 4);
    /// `None` on a dry body or where the lakes' volume left the level where it stood.
    pub sea_before_lakes: Option<i32>,
    /// ★ THE LAKES' SEA FIXED POINT (2026-09-24, ruling B7 item 1; plan question 8): how many times
    /// the sea was re-solved over the inventory less the lakes, and whether it then stood still to
    /// the sixteenth (`false` where the bound [`LAKE_SEA_SOLVES`] stopped it first).
    pub lake_sea_solves: u32,
    pub lake_sea_converged: bool,
    /// Whether any connected sea solve re-seeded from another body (`land::ConnectedSea`).
    pub sea_reseeded: bool,
    /// ★ Whether any connected sea solve's answer did NOT balance (`land::ConnectedSea::settled`;
    /// review finding 4 of plan step 2): its level was solved over one body and its ocean is
    /// another's.
    pub sea_unsettled: bool,
    /// ★ THE WATER'S OWN TREE (plan step 2b; ruling B8 item 2): what the shipped receivers found
    /// and what following them found ([`MacroSolve::follow_the_water`]).
    pub tree: crate::lakes::TreeReport,
}

/// ★ WHAT THE LAKES' SEA FIXED POINT DID (plan question 8; review finding 7 of plan step 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LakeSea {
    /// The sea before the first re-solve, sixteenths.
    pub first: i32,
    /// How many times the sea was re-solved and the lakes read again.
    pub solves: u32,
    /// Whether the sea then stood still to the sixteenth; `false` where the bound stopped it.
    pub converged: bool,
    /// Whether a connected sea solve inside it re-seeded, and whether one did not balance.
    pub reseeded: bool,
    pub unsettled: bool,
}

/// ★ THE LAKES' SEA FIXED POINT (ruling W11 step 4; ruling B7 item 1; plan question 8): the lakes'
/// standing water is part of the body's one inventory, so the sea is re-solved over the inventory
/// LESS the lakes (`volume_m3`, the lakes the state holds now), and a new sea moves the ocean, the
/// pockets and the lakes again: `again` routes nothing, it reads the climate and the lakes over the
/// state as it stands after the new sea's routing and returns their volume. The re-solve repeats
/// until the sea stands still to the sixteenth, at most `bound` times — numerical resolution, never
/// physics — and the answer says whether it stood still.
///
/// **Example.** A pocket in a dry interior fills with 40 km³ of water. The sea stood over the
/// whole inventory, so it is re-solved over the inventory less those 40 km³ and falls; the new sea
/// changes no pocket, the lakes read again hold the same 40 km³, and the second solve stands still.
pub fn settle_lakes_sea(
    state: &mut MacroSolve,
    water_km3: u64,
    bound: u32,
    volume_m3: u128,
    routes: &mut Vec<RouteReport>,
    again: &mut dyn FnMut(&MacroSolve) -> u128,
) -> LakeSea {
    let mut fix = LakeSea {
        first: state.sea_z,
        solves: 0,
        converged: false,
        reseeded: false,
        unsettled: false,
    };
    let mut volume = volume_m3;
    for _ in 0..bound {
        let km3 = (volume / LAKE_KM3_M3) as u64;
        let left = water_km3.saturating_sub(km3);
        let lattice = state.lattice;
        // A lake that held the whole inventory would leave no sea to solve: the level stands.
        let (level, reseeded, settled) = match crate::land::sea_level_connected_on(
            &lattice,
            &state.z,
            &state.area,
            left,
            false,
            state.cores,
        ) {
            Some(sea) => (sea.level, sea.reseeded, sea.settled),
            None => (state.sea_z, false, true),
        };
        fix.reseeded |= reseeded;
        fix.unsettled |= !settled;
        fix.converged = level == state.sea_z;
        if fix.converged {
            break;
        }
        fix.solves += 1;
        state.sea_z = level;
        routes.push(state.route());
        volume = again(state);
    }
    fix
}

/// ★ THE MOST TIMES the sea is re-solved over the inventory less the lakes (plan question 8): a
/// NUMERICAL RESOLUTION of the fixed point, never a physical number; the report states whether the
/// sea stood still inside it.
pub const LAKE_SEA_SOLVES: u32 = 4;

/// The cubic metres in a cubic kilometre: the lakes' volume against the charter's inventory.
pub const LAKE_KM3_M3: u128 = 1_000_000_000;

/// A basin must hold this many land nodes to be read by gate G-AGE: a hundred nodes is about
/// 6 700 km² on the home planet, a river basin and not a gully.
pub const AGE_GATE_MIN_NODES: u32 = 100;

/// ★ GATE G-AGE's verdict over a reading: the area-weighted MEDIAN integral (by node count) of
/// the basins, in 1/256, and whether it stands inside Strahler's maturity band. `None` with no
/// basin to read.
#[must_use]
pub fn age_gate(integrals: &[BasinIntegral]) -> Option<(u8, bool)> {
    let total: u64 = integrals.iter().map(|b| u64::from(b.nodes)).sum();
    if total == 0 {
        return None;
    }
    let mut sorted: Vec<&BasinIntegral> = integrals.iter().collect();
    sorted.sort_by_key(|b| (b.integral_q8, b.outlet));
    let mut acc = 0u64;
    let mut median = sorted[0].integral_q8;
    for b in sorted {
        acc += u64::from(b.nodes);
        median = b.integral_q8;
        if acc * 2 >= total {
            break;
        }
    }
    let (lo, hi) = HI_MATURE;
    let value = f64::from(median) / 256.0;
    Some((median, value >= lo && value <= hi))
}

/// The solve's state over one body's macro lattice. Every row is one word a node; the words are
/// public so an instrument can read them, and only the passes below write them.
#[derive(Clone, Debug, PartialEq)]
pub struct MacroSolve {
    pub lattice: MacroLattice,
    /// The sea's level in sixteenths from the ladder radius: a node at or under it is an outlet.
    pub sea_z: i32,
    /// The terrain, sixteenths from the ladder radius.
    pub z: Vec<i32>,
    /// The routing surface: the terrain with every pit raised to its spill level.
    pub z_flood: Vec<i32>,
    /// The receiver: while the passes run, the ROUTING's D8 receiver on the flooded surface
    /// ([`NO_NODE`] for an outlet); ★ once the solve ends, THE WATER'S OWN WAY (plan step 2b; ruling
    /// B8 item 2; [`crate::lakes::water_tree`]) — the ground's descent, across a spilling lake to its
    /// spill, over the spill into the side the budget pours into, and [`NO_NODE`] where the water
    /// ENDS: the ocean, a closed lake's water, a dry hollow's pit.
    pub receiver: Vec<u32>,
    /// The chord to the receiver in whole metres; zero for an outlet.
    pub chord: Vec<u32>,
    /// The node's true area in whole square metres.
    pub area: Vec<u64>,
    /// The rain at the node, mm/yr, in `[P_MIN_MM_YR, P_MAX_MM_YR]`.
    pub rain: Vec<u32>,
    /// The discharge, mm·m²/yr: the rain accumulated downstream while the passes run (the stream
    /// power law's drainage), and ★ THE BUDGET'S OWN WATER once the solve ends (plan step 2b; ruling
    /// B8 item 2, amending B7 item 2; [`MacroSolve::follow_the_water`]) — the runoff accumulated
    /// over the water's own receivers, less at each spill what its spilling lake hands the air, so a
    /// hollow that never fills passes nothing on and the row into the sea is the budget's own. The
    /// row the artifact ships.
    pub discharge: Vec<u64>,
    /// The topological order of the receiver tree, LEAVES FIRST, outlets last.
    pub order: Vec<u32>,
    /// ★ THE SINKS the shipped row takes off (plan step 2b): at each spill node of a spilling lake,
    /// the water its lake hands the air, sorted by node; empty while the passes run.
    pub spill_sinks: Vec<(u32, u64)>,
    /// ★ THE BODY'S SEED ([`BodyDefinition::seed`]): the flat step's split reads its own stream
    /// under it (plan step 2b, question 4).
    pub seed: u64,
    /// ★ THE UPLIFT OVER THE AGE (C2's belts; C3): what each node gains by the end of the age,
    /// sixteenths, applied pass by pass; negative where a trench subsides.
    pub uplift: Vec<i32>,
    /// ★ THE REMOVAL FIELD: what the sweeps have cut from each node so far, sixteenths — the
    /// sediment budget's input.
    pub removed: Vec<u32>,
    /// The NET removal the rebound has not yet answered, sixteenths, SIGNED (2026-09-22): the cut
    /// less the sediment laid on the node — a filled basin is a load and sinks, an eroded range is
    /// lightened and rises. Consumed by every rebound.
    pub pending: Vec<i32>,
    /// ★ THE REBOUND'S CARRIED FRACTIONS (ruling B45 item 1: the rebound without its floors): each node's part of a
    /// sixteenth its last rebound owed but could not lay, carried to its next rebound, so the lifts sum exactly over
    /// the age.
    pub rebound_carry: Vec<Gf>,
    /// ★ THIS PASS's cut per node, sixteenths (2026-09-22): the sediment the deposit carries.
    pub cut_now: Vec<u32>,
    /// ★ THE ERODIBILITY of each node's rock as a share of `K₀` in 1/256 (W7 step 3), from its
    /// province ([`crate::strata::Province::erodibility_q8`]); 256 where no province is stated.
    pub erodibility_q8: Vec<u32>,
    /// ★ THE PIT under each node AS THE ROUTING FOUND IT, sixteenths (2026-09-22): the flood level
    /// over the terrain at route time, zero for a node the flood did not raise. The sweep skips a
    /// node with a pit (a lake does not cut) and the deposit fills up to it — never `z_flood − z`
    /// read after the sweep, which is every cut node's own cut (MEASURED: the fill equalled the cut
    /// to a tenth of a percent, pass after pass, and no lake ever drained).
    pub pit: Vec<u32>,
    /// ★ THE ROCK PROVINCE of each node (slice 8d step 2), as its code byte: the initial land's own
    /// reading, carried through the solve untouched. The rivers cut the ground; they do not change
    /// which crust it is. A state built with no initial land reads
    /// [`crate::strata::DEFAULT_PROVINCE`] everywhere, because it has no plates to read.
    pub province: Vec<u8>,
    /// ★ THE STANDING WATER at each node, sixteenths, or [`DRY`] (ruling W11, 2026-09-22): the
    /// level the LAKES' WATER BUDGET settled ([`crate::lakes::budget`]), never the routing flood.
    /// The flood is a scratch surface that decides the receivers, the flats and the order and
    /// nothing else; a hollow is a lake only where a finite amount of water stands in it. Empty of
    /// water until the budget runs at the end of the full solve, so every pass reads a dry field.
    pub water_z: Vec<i32>,
    /// ★ THE OCEAN (2026-09-24, ruling B7 item 1: *"an inland pocket is a closed basin, not the
    /// sea"*): whether each node belongs to the CONNECTED ocean at the sea's level — the largest
    /// connected water under it ([`crate::land::ocean_mask`]). Every [`MacroSolve::route`] labels it
    /// again at the sea's level over the field as it stands. The sea's facies, the water level, the
    /// sweep's base level, the budget's root and the climate's sea read THIS bit, never `z ≤ sea`.
    pub ocean: Vec<bool>,
    /// The node directions at [`CACHE_BITS`], read by every chord.
    dir: Vec<[i32; 3]>,
    /// ★ THE CORES THE PASSES' STEPS MAY USE (ruling B64): an operational word the caller states, never part of the
    /// world — the same bytes come out on any number ([`crate::par`]).
    pub cores: crate::par::Cores,
}

/// ★ THE BUDGET'S CLOSED HOLLOWS (see [`MacroSolve::follow_the_budget`]): for each closed body (a hollow whose inflow
/// stays under what its water hands the air), its nodes — the lake and its dry rim — and its spill level, sixteenths;
/// and ★ WHERE ITS ROCK GOES ONCE IT IS FULL OF ROCK ([`MacroSolve::deposit_basins`]): the ways over its saddle in the
/// water's own order, and the closed body its parent's ring belongs to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Basins {
    /// Each node's closed body's slot, or `u32::MAX`.
    pub slot: Vec<u32>,
    /// Each closed body's nodes and spill level.
    pub bodies: Vec<(Vec<u32>, i32)>,
    /// Each closed body's ways over its saddle, steepest first (the saddle's own order), its own side left out: the
    /// side's closed body, or [`OVER_TO_SEA`] for the ocean's side, or [`OVER_TO_FULL`] for a hollow that spills; and
    /// the side's gate, where the rock enters it. Empty where the body never met another (no saddle).
    pub sides: Vec<Vec<(u32, u32)>>,
    /// The closed body its parent's ring belongs to, or `u32::MAX` where the parent spills, is the ocean, or is none.
    pub parent: Vec<u32>,
    /// Where the rock goes when every side is full and the parent's ring is no closed body: the body's own saddle node,
    /// from which the parent's water runs across its lake to its spill, or [`NO_NODE`] where it never met another.
    pub parent_gate: Vec<u32>,
}

/// A side of a closed body's saddle that is the ocean's ([`Basins::sides`]).
pub const OVER_TO_SEA: u32 = u32::MAX - 1;
/// A side of a closed body's saddle that is a hollow which spills ([`Basins::sides`]).
pub const OVER_TO_FULL: u32 = u32::MAX;

/// ★ THE LEVEL FILL of one closed hollow: `heights` its nodes' `(height, area)` sorted by height (sixteenths, m²),
/// `volume` the rock it holds (sixteenths × m²); returns the one level, sixteenths, that the volume pays for over the
/// lowest nodes, never over `spill` (a remainder under one sixteenth of the raised area stays unlaid). Every node under
/// the level is then raised to it.
///
/// **Example.** A dry hollow in the plateau's lee holds three nodes at 0, 10 and 20 sixteenths, each 1 m², and its rim's
/// rivers bring 15 sixteenths·m²: the lowest node rises to 10 (10 paid), then the two lowest share the last 5 and stand at
/// 12.
#[must_use]
pub fn level_fill(heights: &[(i32, u64)], volume: u128, spill: i32) -> i32 {
    let spill = i64::from(spill);
    let Some(&(lowest, _)) = heights.first() else {
        return spill as i32;
    };
    let (mut paid, mut area_below, mut level) = (0u128, 0u128, i64::from(lowest));
    let mut k = 0usize;
    loop {
        let z_here = heights
            .get(k)
            .map_or(spill, |&(z, _)| i64::from(z).min(spill));
        let cost = area_below * (z_here - level).max(0) as u128;
        if paid + cost > volume {
            level += ((volume - paid) / area_below.max(1)) as i64;
            break;
        }
        paid += cost;
        level = level.max(z_here);
        if (k >= heights.len()) | (z_here >= spill) {
            break;
        }
        area_below += u128::from(heights[k].1);
        k += 1;
    }
    level.min(spill) as i32
}

/// ★ THE JOINT LAW'S IMPLICIT ROOT (see [`MacroSolve::sweep`]): the `x` that solves `x − x₀ + c·(x/L)^n = 0`, by Newton
/// from `x₀` — the function is convex and rising, so the steps fall to the root from above without passing it — at most
/// [`JOINT_NEWTON`] steps, stopping where a step moves under `tolerance`; the powers through the fence's own
/// logarithm and exponential ([`Gf::powf`]).
#[must_use]
pub fn joint_root(x0: Gf, c: Gf, length: Gf, n: Gf, tolerance: Gf) -> Gf {
    let mut x = x0;
    for _ in 0..JOINT_NEWTON {
        let s = x / length;
        let sn = s.powf(n);
        let g = x - x0 + c * sn;
        let dg = Gf::ONE + c * n * (sn / s) / length;
        let next = (x - g / dg).greater(Gf::ZERO);
        let moved = (x - next).abs();
        x = next;
        if moved < tolerance {
            break;
        }
    }
    x
}

/// ★ THE LANDSCAPE'S TIME (ruling B45 item 2; ruling B46): each plate's raise time, years before the present, under
/// the reworking law — `−ln(1 − u)/λ` for the plate's own age share `u` ([`REWORK_RATE_PER_GYR`]; the share is a draw
/// on `[0, 1 − 2⁻⁵³]`, so the logarithm is always finite). The law's own domain is the system's age: a plate the law
/// has not raised since the body formed keeps its first ground, as old as the body. A plate's ground erodes only since
/// it was last raised. ★ A BODY WHOSE PLATES TURN NOTHING OVER
/// (`reworks` false: no ridge opens an ocean, so the plates' turnover is none — a stagnant lid) reworks no ground, and
/// every plate is as old as the system. A body whose plates turn reads Earth's rate — a STATED INTERIM (DEFERRED
/// D-TERRAIN-10) until the rate is scaled by each body's own turnover: the home planet's plates turn over in about
/// 225 Myr (MEASURED in the bench, run 30) against Earth's ~190 Myr.
///
/// **Example.** A plate whose age share is a half was raised 680 million years ago: its old plains wear for 0.68 Gyr,
/// not for the system's five. The home moon, a stagnant lid, keeps the whole age and every crater it ever took.
#[must_use]
pub fn raise_times(plates: &[crate::land::Plate], age_yr: u64, reworks: bool) -> Vec<u64> {
    plates
        .iter()
        .map(|p| {
            if !reworks {
                return age_yr;
            }
            let t_gyr = -(Gf::ONE - p.age).ln() / Gf::from_f64(REWORK_RATE_PER_GYR);
            ((t_gyr * Gf::from_f64(1e9)).to_i64_floor() as u64).min(age_yr)
        })
        .collect()
}

/// A node's share of the pass `[t0, t1)` after its erosion's start, in 1/[`ACTIVE_FULL`]: none before the start, the
/// whole pass after it, the part after it in between — an exact integer share.
#[must_use]
pub fn active_share(start_yr: u64, t0: u64, t1: u64) -> u32 {
    if start_yr >= t1 {
        0
    } else if start_yr <= t0 {
        ACTIVE_FULL
    } else {
        (u128::from(t1 - start_yr) * u128::from(ACTIVE_FULL) / u128::from(t1 - t0)) as u32
    }
}

/// ★ THE BELTS ON THE LAND (phase 2): every node where a wedge rises loses its share of the room's even uplift (the
/// wedge is its belt now), and a wedge node off its plateau reads the folded belt's rock while a plateau keeps its own.
/// Returns each node's even uplift over the age, sixteenths.
#[must_use]
pub fn belt_land(belts: &crate::belts::Belts, land: &mut crate::land::Land) -> Vec<i64> {
    let mut even: Vec<i64> = land.uplift.iter().map(|&u| i64::from(u)).collect();
    for (k, &t) in belts.total.iter().enumerate() {
        if t > 0 {
            even[k] = 0;
            if !belts.plateau[k] {
                land.province[k] = crate::strata::Province::FoldedBelt.code();
            }
        }
    }
    even
}

/// ★ EACH NODE'S EROSION START, years from the solve's start (the landscape's time): its plate's last raise, or its
/// belt's window where that opened later.
#[must_use]
pub fn erosion_starts(
    belts: &crate::belts::Belts,
    raise_yr: &[u64],
    plate: &[u8],
    age_yr: u64,
) -> Vec<u64> {
    plate
        .iter()
        .enumerate()
        .map(|(k, &p)| {
            let raised = age_yr - raise_yr[usize::from(p)];
            if belts.total[k] > 0 {
                raised.min(belts.start_yr[k])
            } else {
                raised
            }
        })
        .collect()
}

/// ★ THE UPLIFT OF ONE PASS `[t0, t1)`: every node its even total's share of the pass's time (an exact cumulative
/// split, so the passes sum to the total to the sixteenth and a trench steps down), and a wedge node its wedge's own
/// rise over the pass.
pub fn pass_uplift(
    z: &mut [i32],
    even_total: &[i64],
    belts: &crate::belts::Belts,
    age_yr: u64,
    t0: u64,
    t1: u64,
) {
    for (h, &even) in z.iter_mut().zip(even_total) {
        *h += crate::belts::window_step(even, 0, age_yr, t0, t1) as i32;
    }
    for w in &belts.wedge {
        z[w.node as usize] += crate::belts::wedge_step(w, t0, t1) as i32;
    }
}

/// One basin's hypsometric integral (gate G-AGE).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BasinIntegral {
    /// The basin's outlet node.
    pub outlet: u32,
    /// Its land nodes (over the sea).
    pub nodes: u32,
    /// The integral in 1/256.
    pub integral_q8: u8,
}

/// The sea's level of `body` in sixteenths from the ladder radius.
#[must_use]
pub fn sea_level(body: &BodyDefinition) -> i32 {
    match body.sea_m() {
        Some(m) => m.saturating_mul(Z_STEPS_PER_M),
        None => i32::MIN,
    }
}

/// The rung whose cell is about a node: `log₂` of the node's size in rung-0 cells, at most the
/// body's top rung. The starting surface is the recipe's relief AT THAT RUNG — the octaves the
/// survival rule keeps there, which is exactly the relief a node can resolve.
#[must_use]
pub fn node_rung(body: &BodyDefinition, lattice: &MacroLattice) -> u8 {
    let log2 = (31 - lattice.cells_per_node.leading_zeros()) as u8;
    log2.min(body.ladder().rungs - 1)
}

/// ★ THE STARTING SURFACE: the recipe's own relief at every node's direction, at the node's rung,
/// in sixteenths from the ladder radius (C2 replaces it with the initial land).
#[must_use]
pub fn initial_surface(body: &BodyDefinition, lattice: &MacroLattice) -> Vec<i32> {
    let rung = node_rung(body, lattice);
    (0..lattice.node_count() as u32)
        .map(|node| {
            ((height(body, lattice.direction(node), rung) - body.radius).raw() >> Z_SHIFT) as i32
        })
        .collect()
}

/// ★ THE OUTLETS OF A BODY WITH NO SEA ([`OUTLETS_WITHOUT_SEA`]): its lowest nodes by `(height, index)`, the ones the
/// routing seeds its flood from and the lakes' budget takes as its root, so both agree where the water leaves.
///
/// **Example.** The dry moon has no ocean: its six lowest crater floors are where every flood starts.
#[must_use]
pub(crate) fn outlets_without_sea(z: &[i32]) -> Vec<u32> {
    let mut lowest: BinaryHeap<(i32, u32)> = BinaryHeap::new();
    for (i, &h) in z.iter().enumerate() {
        lowest.push((h, i as u32));
        if lowest.len() > OUTLETS_WITHOUT_SEA {
            lowest.pop();
        }
    }
    lowest.into_iter().map(|(_, node)| node).collect()
}

/// ★ THE TOPOLOGICAL ORDER of a receiver tree, LEAVES FIRST: a node is emitted once every donor
/// is; the receiver whose last donor was emitted is pushed. A stack, so the order is a stated one. A
/// node on a ring of receivers is never emitted, so `n − order.len()` counts the ring.
///
/// **Example.** A trunk off the belt's range and its two tributaries: each tributary's head comes
/// first, then the confluence, then the trunk down to the sea.
#[must_use]
pub(crate) fn order_of(receiver: &[u32]) -> Vec<u32> {
    let n = receiver.len();
    let mut donors = vec![0u8; n];
    for &r in receiver {
        if r != NO_NODE {
            donors[r as usize] += 1;
        }
    }
    let mut stack: Vec<u32> = (0..n as u32).filter(|&i| donors[i as usize] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(node) = stack.pop() {
        order.push(node);
        let r = receiver[node as usize];
        if r != NO_NODE {
            donors[r as usize] -= 1;
            if donors[r as usize] == 0 {
                stack.push(r);
            }
        }
    }
    order
}

/// Whether neighbour `m` (drop `dz_m` over chord `l_m`) is a STEEPER descent than the best so far
/// (`dz_best` over `l_best`, node `best`): cross-multiplied integers, and on an exact tie the
/// smaller global index. A total order; no float ratio is ever compared.
#[must_use]
pub fn steeper(dz_m: i64, l_m: u32, m: u32, dz_best: i64, l_best: u32, best: u32) -> bool {
    let lhs = dz_m * i64::from(l_best);
    let rhs = dz_best * i64::from(l_m);
    lhs > rhs || (lhs == rhs && m < best)
}

impl MacroSolve {
    /// The solve's state for `body`, at its starting surface; `None` where the body has no
    /// lattice.
    #[must_use]
    pub fn new(body: &BodyDefinition) -> Option<MacroSolve> {
        let lattice = body.macro_lattice()?;
        let n = lattice.node_count();
        let z = initial_surface(body, &lattice);
        let dir: Vec<[i32; 3]> = (0..n as u32)
            .map(|node| {
                let d = lattice.direction(node);
                [cache(d[0]), cache(d[1]), cache(d[2])]
            })
            .collect();
        let area = (0..n as u32).map(|node| lattice.area_m2(node)).collect();
        Some(MacroSolve {
            lattice,
            sea_z: sea_level(body),
            z_flood: z.clone(),
            z,
            receiver: vec![NO_NODE; n],
            chord: vec![0; n],
            area,
            rain: vec![P_MIN_MM_YR; n],
            discharge: vec![0; n],
            order: Vec::with_capacity(n),
            uplift: vec![0; n],
            removed: vec![0; n],
            pending: vec![0; n],
            rebound_carry: vec![Gf::ZERO; n],
            cut_now: vec![0; n],
            pit: vec![0; n],
            erodibility_q8: vec![256; n],
            province: vec![crate::strata::DEFAULT_PROVINCE.code(); n],
            water_z: vec![DRY; n],
            ocean: vec![false; n],
            spill_sinks: Vec::new(),
            seed: body.seed(),
            dir,
            cores: crate::par::Cores::default(),
        })
    }

    /// ★ THE SOLVE'S STATE FROM THE INITIAL LAND (stage C2): the land's heights and its sea level
    /// as the start, in place of the recipe's noise. A body with no sea gets a level under every
    /// node, so the flood seeds from its lowest nodes.
    #[must_use]
    pub fn from_land(body: &BodyDefinition, land: &crate::land::Land) -> Option<MacroSolve> {
        let mut state = MacroSolve::new(body)?;
        state.z.clone_from(&land.z);
        state.z_flood.clone_from(&land.z);
        state.uplift.clone_from(&land.uplift);
        state.province.clone_from(&land.province);
        state.erodibility_q8 = land
            .province
            .iter()
            .map(|&p| {
                crate::strata::Province::from_code(p)
                    .map_or(256, crate::strata::Province::erodibility_q8)
            })
            .collect();
        state.sea_z = land.sea_z.unwrap_or(i32::MIN);
        state.ocean.clone_from(&land.ocean);
        Some(state)
    }

    /// ★ THE SEA AS THE CLIMATE READS IT: this state's level and its connected ocean.
    #[must_use]
    pub fn sea_view(&self) -> crate::climate::SeaView<'_> {
        crate::climate::SeaView {
            level: self.sea_z,
            ocean: &self.ocean,
        }
    }

    /// ★ THE OCEAN LABELLED AGAIN at the sea's level over the field as it stands (2026-09-24, ruling
    /// B7 item 1), and the pockets under the level it leaves out.
    pub fn label_ocean(&mut self) -> (usize, usize) {
        let mask =
            crate::land::ocean_mask_on(&self.lattice, &self.z, &self.area, self.sea_z, self.cores);
        self.ocean = mask.ocean;
        (mask.pockets, mask.pocket_nodes)
    }

    /// The nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.z.len()
    }

    /// The bytes the state holds per node, summed over its rows — what the design's memory model
    /// is measured against.
    #[must_use]
    pub fn bytes_per_node(&self) -> usize {
        4 + 4 + 4 + 4 + 8 + 4 + 8 + 4 + 4 + 4 + 8 + 4 + 12 + 1 + 4 + 1
    }

    /// ★ THE BASIN OF EVERY NODE: the outlet its water reaches, read down the order (outlets first,
    /// each node after its receiver). An outlet is its own basin.
    #[must_use]
    pub fn basins(&self) -> Vec<u32> {
        let mut basin = vec![NO_NODE; self.node_count()];
        for &node in self.order.iter().rev() {
            let r = self.receiver[node as usize];
            basin[node as usize] = if r == NO_NODE {
                node
            } else {
                basin[r as usize]
            };
        }
        basin
    }

    /// ★ GATE G-AGE's READING: the hypsometric integral of every basin with at least `min_nodes`
    /// LAND nodes (every node that is not the connected ocean; 2026-09-24, ruling B7 item 1), by
    /// area — `(mean − min) / (max − min)` in 1/256 — in outlet order. A basin whose land is flat
    /// (max = min) is left out: it has no integral.
    #[must_use]
    pub fn hypsometric_integrals(&self, min_nodes: u32) -> Vec<BasinIntegral> {
        let n = self.node_count();
        let basin = self.basins();
        // 1. Land nodes per basin.
        let mut count = vec![0u32; n];
        for i in 0..n {
            count[basin[i] as usize] += u32::from(!self.ocean[i]);
        }
        // 2. A dense slot for every basin that qualifies, in outlet order.
        let mut slot = vec![u32::MAX; n];
        let mut outlets = Vec::new();
        for (i, &c) in count.iter().enumerate() {
            if c >= min_nodes && c > 0 {
                slot[i] = outlets.len() as u32;
                outlets.push(i as u32);
            }
        }
        let m = outlets.len();
        let (mut lo, mut hi) = (vec![i32::MAX; m], vec![i32::MIN; m]);
        let (mut sum, mut area) = (vec![0i128; m], vec![0i128; m]);
        for i in 0..n {
            let s = slot[basin[i] as usize];
            if (s == u32::MAX) | self.ocean[i] {
                continue;
            }
            let s = s as usize;
            lo[s] = lo[s].min(self.z[i]);
            hi[s] = hi[s].max(self.z[i]);
            sum[s] += i128::from(self.area[i]) * i128::from(self.z[i]);
            area[s] += i128::from(self.area[i]);
        }
        let mut out = Vec::with_capacity(m);
        for s in 0..m {
            if hi[s] == lo[s] {
                continue;
            }
            // (mean − min) / (max − min) = (sum − min·area) / ((max − min)·area), in 1/256 —
            // ONE integer division at the end, never a floored mean first.
            let over = sum[s] - i128::from(lo[s]) * area[s];
            let span = (i128::from(hi[s]) - i128::from(lo[s])) * area[s];
            let q8 = (over * 256 / span).clamp(0, 255) as u8;
            out.push(BasinIntegral {
                outlet: outlets[s],
                nodes: count[outlets[s] as usize],
                integral_q8: q8,
            });
        }
        out
    }

    /// ★ THE SEDIMENT BUDGET: what the sweeps removed, summed per basin in cubic metres — the
    /// deposit a delta or an alluvial fan is made of (8d reads it). Only basins with a removal.
    #[must_use]
    pub fn sediment_by_basin(&self) -> std::collections::BTreeMap<u32, u64> {
        let basin = self.basins();
        let mut out = std::collections::BTreeMap::new();
        for (i, &b) in basin.iter().enumerate() {
            if self.removed[i] == 0 {
                continue;
            }
            let m3 = (u128::from(self.removed[i]) * u128::from(self.area[i])
                / u128::from(Z_STEPS_PER_M as u32)) as u64;
            *out.entry(b).or_insert(0u64) += m3;
        }
        out
    }

    /// ★ THE WATER LEVEL AT A NODE (ruling W11, 2026-09-22): the level the LAKES' BUDGET settled
    /// where standing water covers the node, the sea's level on the CONNECTED OCEAN (2026-09-24,
    /// ruling B7 item 1: an inland pocket is a closed basin, and its water is its budget's),
    /// [`DRY`] elsewhere — one expression, no branch on a landform kind. It no longer reads the
    /// routing flood: a hollow the rain cannot fill is a dry basin, not a lake.
    ///
    /// **Example.** A pilot follows a river into a hollow in the belt's interior. The hollow's own
    /// nodes stand metres under the flood's spill, and every one of them says DRY: the river that
    /// reaches it evaporates, and what she lands on is a salt pan.
    #[must_use]
    pub fn water_level(&self, node: u32) -> i32 {
        let i = node as usize;
        if self.water_z[i] != DRY {
            self.water_z[i]
        } else if self.ocean[i] {
            self.sea_z
        } else {
            DRY
        }
    }

    /// ★ THE SWEEP'S BASE LEVEL at a node, CURED (ruling B43 item 1): a lake the routing found stands at its flood
    /// level (its spill, or the level the lakes' cure lowered it to), the connected ocean at the sea's level, and every
    /// other node at its ground as it stands NOW — never under the ground. MEASURED (the belts' design §14): the rule
    /// before this read `z_flood > z` as a lake, which is true of every node the sweep has just cut (its ground now
    /// under its stale flood level), so the implicit sweep climbed one node per routing and the relief depended on the
    /// routing count. One rule for the state and for a sweep's cells ([`base_level`]); public for the instruments that
    /// read the sweep's floor.
    #[must_use]
    pub fn sweep_base(&self, node: u32) -> i32 {
        let i = node as usize;
        base_level(
            self.z[i],
            self.z_flood[i],
            self.pit[i],
            self.ocean[i],
            self.sea_z,
        )
    }

    /// ★ A CLOSED BODY'S SPILL AS IT STANDS NOW (ruling B62 item 2; as B47 made a lake's level follow its outlet): the
    /// spill the routing found — its saddle node's height then — or the saddle's height now where a river has cut the
    /// saddle lower since. MEASURED before this: a basin could open only when the water was routed again, once a pass,
    /// so a coarser axis kept rock in hollows already captured (the gate's miss on two old plains, plates 5 and 13).
    fn spill_now(&self, basins: &Basins, b: usize) -> i32 {
        // A body with no saddle stated never met another: it keeps the routing's spill.
        let gate = basins.parent_gate.get(b).copied().unwrap_or(NO_NODE);
        let stored = basins.bodies[b].1;
        if gate == NO_NODE {
            stored
        } else {
            stored.min(self.z[gate as usize])
        }
    }

    /// ★ THE CAPTURE INSIDE THE SWEEP (ruling B62 item 2): every closed body's standing water drains to its spill as it
    /// stands now ([`MacroSolve::spill_now`]) — a river that cut the body's saddle under its lake lets the lake out in the
    /// same sweep, not at the next routing, and the floor it uncovers is cut like any dry node. Returns the nodes drained.
    fn drain_to_saddles(&mut self, basins: &Basins) -> usize {
        let mut drained = 0usize;
        for b in 0..basins.bodies.len() {
            let spill = self.spill_now(basins, b);
            for &v in &basins.bodies[b].0 {
                let i = v as usize;
                if (self.pit[i] > 0) & (self.z_flood[i] > spill) {
                    let level = self.z[i].max(spill);
                    self.z_flood[i] = level;
                    self.pit[i] = (level - self.z[i]) as u32;
                    drained += 1;
                }
            }
        }
        drained
    }

    /// A node's cached direction at [`CACHE_BITS`] (plan step 2b: the flat step's bearing).
    pub(crate) fn dir_of(&self, node: u32) -> [i32; 3] {
        self.dir[node as usize]
    }

    /// The chord between two nodes from the cached directions, whole metres.
    pub(crate) fn chord_m(&self, a: u32, b: u32) -> u32 {
        chord_between(
            self.lattice.radius_m(),
            uncache(self.dir[a as usize]),
            uncache(self.dir[b as usize]),
        )
    }

    /// ★ THE ROUTING: the ocean labelled at the sea's level, the priority flood, the D8 receivers on
    /// the flooded surface, the flats by a distance field, and the topological order of the
    /// receiver tree.
    ///
    /// ★ ONLY THE CONNECTED OCEAN SEEDS THE FLOOD (2026-09-24, ruling B7 item 1; plan step 2 part
    /// 3a). An inland pocket under the sea's level is a closed basin: the flood raises it to its
    /// spill like any hollow, the sweep skips it, the deposit fills it, and the budget decides its
    /// water. MEASURED at generator 19: 148 pockets, 59 470 nodes, 3.2 million km² were seeded as
    /// sea and drawn at the sea's level.
    pub fn route(&mut self) -> RouteReport {
        let n = self.node_count();
        let mut report = RouteReport::default();
        (report.pockets, report.pocket_nodes) = self.label_ocean();
        // 1. THE PRIORITY FLOOD: the outlets are the seeds, and every node is raised to the least, over the paths from
        //    an outlet, of the highest node on the path — ★ ON THE SOLVE'S CORES, the same bytes (ruling B72 item 1,
        //    [`crate::flood::fill`]).
        let threads = self.cores.threads();
        let mut seed = self.ocean.clone();
        report.sea_seeded = seed.contains(&true);
        if !report.sea_seeded {
            for node in outlets_without_sea(&self.z) {
                seed[node as usize] = true;
            }
        }
        // ★ Every whole-field count, pit and pick below ON THE SOLVE'S CORES (ruling B72 item 1): each node reads only
        //   the flood and its own neighbours, and the counts are integer sums.
        report.outlets = crate::par::per_range(n, threads, |r| r.filter(|&i| seed[i]).count())
            .into_iter()
            .sum();
        self.z_flood = crate::flood::fill(&self.lattice, &self.z, &seed, self.cores);
        self.pit = crate::par::map(n, threads, |i| (self.z_flood[i] - self.z[i]).max(0) as u32);
        (report.raised, report.pit_steps) = crate::par::per_range(n, threads, |r| {
            let pits = &self.pit[r];
            (
                pits.iter().filter(|&&p| p > 0).count(),
                pits.iter().map(|&p| u64::from(p)).sum::<u64>(),
            )
        })
        .into_iter()
        .fold((0, 0), |(a, b), (c, d)| (a + c, b + d));
        // 2. THE RECEIVERS: the steepest lower neighbour on the flooded surface — ★ ON THE SOLVE'S CORES (ruling B64):
        //    each node reads only the flooded surface, so the nodes are cut into ranges and their picks laid in order.
        let picks = crate::par::map(n, threads, |i| {
            let node = i as u32;
            if seed[i] {
                return (NO_NODE, 0);
            }
            let mut best = NO_NODE;
            let (mut dz_best, mut l_best) = (0i64, 0u32);
            for m in self.lattice.neighbours(node) {
                if m == NO_NODE || self.z_flood[m as usize] >= self.z_flood[i] {
                    continue;
                }
                let dz = i64::from(self.z_flood[i]) - i64::from(self.z_flood[m as usize]);
                let l = self.chord_m(node, m);
                if best == NO_NODE || steeper(dz, l, m, dz_best, l_best, best) {
                    best = m;
                    dz_best = dz;
                    l_best = l;
                }
            }
            (best, l_best)
        });
        // An outlet or a node with a lower neighbour is routed (its distance zero); every other node waits for the flats.
        self.receiver = crate::par::map(n, threads, |i| picks[i].0);
        self.chord = crate::par::map(n, threads, |i| picks[i].1);
        let mut dist: Vec<u64> = crate::par::map(n, threads, |i| {
            if seed[i] | (picks[i].0 != NO_NODE) {
                0
            } else {
                u64::MAX
            }
        });
        drop(picks);
        drop(seed);
        // 3. ★ THE FLATS, BY THE DISTANCE IN METRES AND NEVER BY THE HOP COUNT (ruling B2 step 3,
        //    2026-09-22; Cordonnier, Bovy & Braun 2019 §2.3.2). A filled hollow is a FLAT, and a
        //    hop count charges the stencil's diagonal — which is √2 of a side — the same one step
        //    as a row. So the cheapest path across a lake ran on the diagonal, planet-wide, and the
        //    receiver tree drew the scratches the owner saw. Here the distance to the flat's own
        //    draining shore is the lattice's OWN CHORD IN WHOLE METRES, summed along the path — a
        //    shortest path, so it depends on no visit order, and a diagonal step pays its length.
        //    Cordonnier rank on the straight line to one outlet and say plainly that it *"does not
        //    yield the perfect path patterns that one would obtain by including obstacles in the
        //    computation of the Euclidean distance"*; the summed chord IS that obstacle-aware
        //    distance, and our flats have a whole SHORE of outlets rather than the one node a
        //    straight line could be measured to.
        //
        //    ONLY THE SHORE SEEDS THE WALK. A routed node with no unreached neighbour of its own
        //    flooded height touches no flat, so seeding every routed node would put eight million
        //    rows in a heap to no purpose (SL9: a cost that grows with the count is a defect).
        //    ★ The shore is found ON THE SOLVE'S CORES (ruling B72 item 1): each node reads only the finished receivers
        //    and the flooded surface, and the ranges' shores join in index order.
        let shores: Vec<u32> =
            crate::par::join(crate::par::per_range(n, self.cores.threads(), |r| {
                r.filter(|&i| {
                    (dist[i] == 0)
                        && self.lattice.neighbours(i as u32).iter().any(|&m| {
                            m != NO_NODE
                                && dist[m as usize] == u64::MAX
                                && self.z_flood[m as usize] == self.z_flood[i]
                        })
                })
                .map(|i| i as u32)
                .collect::<Vec<u32>>()
            }));
        let mut walk: BinaryHeap<Reverse<(u64, u32)>> =
            shores.into_iter().map(|node| Reverse((0, node))).collect();
        while let Some(Reverse((d, node))) = walk.pop() {
            if d > dist[node as usize] {
                continue;
            }
            for m in self.lattice.neighbours(node) {
                if m == NO_NODE || self.z_flood[m as usize] != self.z_flood[node as usize] {
                    continue;
                }
                let step = d + u64::from(self.chord_m(node, m));
                if step < dist[m as usize] {
                    dist[m as usize] = step;
                    walk.push(Reverse((step, m)));
                }
            }
        }
        drop(walk);
        let (flat, undrained) = self.assign_flat_receivers(&dist);
        report.flat = flat;
        report.undrained = undrained;
        drop(dist);
        // 4. THE ORDER: leaves first ([`order_of`]).
        self.order = order_of(&self.receiver);
        report.cyclic = n - self.order.len();
        report
    }

    /// ★ THE ROUTING A WET PASS READS (ruling B73 item 1, "for now"): the ocean labelled at the sea's level, and every
    /// ocean node's flood level its own ground (an outlet holds no room for rock). In a body with water the budget's own
    /// tree ([`MacroSolve::follow_the_budget`]) then lays the receivers, their chords, the order, the discharge and every
    /// land node's level, so the full routing's flood, flat walk and order were replaced before anything read them
    /// (MEASURED 2026-10-09: a quarter of a pass's time on 14 cores). Returns the routing's report with the ocean's
    /// words alone — the pockets, the seeding and the outlets.
    ///
    /// **Example.** A pass of the home planet's solve routes its water four times; each time it now labels the ocean
    /// under the sea that stands and hands the land to the budget, which finds the lakes and the rivers itself.
    pub fn label_for_budget(&mut self) -> RouteReport {
        let (n, threads) = (self.node_count(), self.cores.threads());
        let mut report = RouteReport::default();
        (report.pockets, report.pocket_nodes) = self.label_ocean();
        report.outlets =
            crate::par::per_range(n, threads, |r| r.filter(|&i| self.ocean[i]).count())
                .into_iter()
                .sum();
        report.sea_seeded = report.outlets > 0;
        self.z_flood = crate::par::map(n, threads, |i| {
            [self.z_flood[i], self.z[i]][usize::from(self.ocean[i])]
        });
        self.pit = crate::par::map(n, threads, |i| [self.pit[i], 0][usize::from(self.ocean[i])]);
        report
    }

    /// ★ THE FLAT'S RECEIVERS from the distance field (ruling B2 step 3, 2026-09-22): a node the
    /// field reached (a distance over zero) takes the neighbour of equal flooded height that stands
    /// NEAREST THE SHORE THROUGH ITSELF — the smallest `distance + chord`, in whole metres — and on
    /// an exact tie the smaller index; a node it never reached is counted UNDRAINED (a defect, gate
    /// G-DRAINAGE) and keeps no receiver. Returns `(flat, undrained)`.
    ///
    /// The cost is the one the distance field itself is built of, so the winner's distance is
    /// exactly this node's own less the chord between them: the distance FALLS along every step and
    /// the flat's tree can hold no ring. Before this the cost was the HOP COUNT, which charged the
    /// stencil's diagonal the same as its row and sent every lake's water down one diagonal.
    ///
    /// **Example.** A filled hollow on the belt is eight nodes across. Its water used to leave by
    /// the north-east diagonal whatever the hollow's shape, because eight diagonal hops counted
    /// eight and eight row hops counted eight. Now the eight diagonal steps cost 92 700 m and the
    /// eight row steps cost 65 500 m, so the water leaves by the nearest shore, as water does.
    fn assign_flat_receivers(&mut self, dist: &[u64]) -> (usize, usize) {
        let (n, threads) = (self.node_count(), self.cores.threads());
        // ★ ON THE SOLVE'S CORES (ruling B64): each flat node reads only the finished distances and the flooded surface.
        let picks = crate::par::map(n, threads, |i| {
            if (dist[i] == 0) | (dist[i] == u64::MAX) {
                return (NO_NODE, 0);
            }
            let node = i as u32;
            let mut best = NO_NODE;
            let mut best_cost = 0u64;
            let mut best_chord = 0u32;
            for m in self.lattice.neighbours(node) {
                if m == NO_NODE
                    || dist[m as usize] == u64::MAX
                    || self.z_flood[m as usize] != self.z_flood[i]
                {
                    continue;
                }
                let chord = self.chord_m(node, m);
                let cost = dist[m as usize] + u64::from(chord);
                if best == NO_NODE || cost < best_cost || (cost == best_cost && m < best) {
                    best = m;
                    best_cost = cost;
                    best_chord = chord;
                }
            }
            (best, best_chord)
        });
        // ★ The picks laid and counted ON THE SOLVE'S CORES too (ruling B72 item 1): a node reached by the field takes
        //   its pick, a routed node keeps its receiver, a node never reached is counted.
        let on_flat = |i: usize| (dist[i] != 0) & (dist[i] != u64::MAX);
        let (flat, undrained) = crate::par::per_range(n, threads, |r| {
            r.fold((0usize, 0usize), |(f, u), i| {
                (
                    f + usize::from(on_flat(i)),
                    u + usize::from(dist[i] == u64::MAX),
                )
            })
        })
        .into_iter()
        .fold((0, 0), |(a, b), (c, d)| (a + c, b + d));
        let receiver = crate::par::map(n, threads, |i| {
            [self.receiver[i], picks[i].0][usize::from(on_flat(i))]
        });
        let chord = crate::par::map(n, threads, |i| {
            [self.chord[i], picks[i].1][usize::from(on_flat(i))]
        });
        self.receiver = receiver;
        self.chord = chord;
        (flat, undrained)
    }

    /// ★ THE DISCHARGE: the rain on every node's own area, accumulated down the receiver tree in
    /// the order, leaves first. Exact integer sums, whatever the order.
    pub fn accumulate(&mut self) {
        for i in 0..self.node_count() {
            self.discharge[i] = u64::from(self.rain[i]) * self.area[i];
        }
        for k in 0..self.order.len() {
            let node = self.order[k] as usize;
            let r = self.receiver[node];
            if r != NO_NODE {
                self.discharge[r as usize] += self.discharge[node];
            }
        }
    }

    /// ★ THE WATER'S OWN TREE, FOLLOWED (plan step 2b; ruling B8 item 2, amending B7 item 2; status
    /// §7 W10 and W11). Called ONCE, after the solve's last routing and after its budget: the tree
    /// ([`crate::lakes::water_tree`], or [`crate::lakes::dry_tree`] on a body with no water) becomes
    /// the receivers, their chords and their order, and the discharge row becomes THE BUDGET'S OWN
    /// WATER — the Turc–Pike runoff on every node's own area, accumulated down the tree, leaves
    /// first, and at each spill node less the water its spilling lakes hand the air (the tree's
    /// SINKS). So a hollow that never fills passes nothing on (its rivers end in it), a spilling
    /// hollow passes its inflow less its lake's evaporation, and the row into the sea is the
    /// cascade's own. Exact integer sums: a node carries at most `P_MAX_MM_YR · area`, and the
    /// globe's whole area times that is under `2⁶⁴`. Returns the tree's report with the ring and
    /// shortfall counts, both zero by construction.
    ///
    /// **Example.** A desert river off a dry range ends in a salt pan and carries nothing on; a wet
    /// valley's river crosses a lake that fills to its sill and leaves it smaller by what the lake
    /// hands the air. Before this step the row carried both on to the sea.
    pub fn follow_the_water(
        &mut self,
        tree: crate::lakes::WaterTree,
        runoff: &[u32],
    ) -> crate::lakes::TreeReport {
        let n = self.node_count();
        let mut report = tree.report;
        self.receiver = tree.receiver;
        // ★ Each node's chord to its receiver ON THE SOLVE'S CORES (ruling B72 item 1): a node reads only its own way.
        self.chord = crate::par::map(n, self.cores.threads(), |i| {
            let r = self.receiver[i];
            if r == NO_NODE {
                0
            } else {
                self.chord_m(i as u32, r)
            }
        });
        self.order = order_of(&self.receiver);
        report.cyclic = n - self.order.len();
        let mut sink = vec![0u64; n];
        for &(node, water) in &tree.sinks {
            sink[node as usize] += water;
        }
        for ((d, &q), &a) in self.discharge.iter_mut().zip(runoff).zip(&self.area) {
            *d = u64::from(q) * a;
        }
        for k in 0..self.order.len() {
            let node = self.order[k] as usize;
            report.shortfall += usize::from(self.discharge[node] < sink[node]);
            self.discharge[node] = self.discharge[node].saturating_sub(sink[node]);
            let r = self.receiver[node];
            if r != NO_NODE {
                self.discharge[r as usize] += self.discharge[node];
            }
        }
        self.spill_sinks = tree.sinks;
        report
    }

    /// ★ THE RIVER BASINS AS THE SWEEP'S WORK (ruling B64): every node of the sweep's order in the drainage tree its
    /// receivers lead to — rooted at an outlet, or at a node outside the order (on a ring of receivers), which the sweep
    /// reads and never moves — and the trees dealt to the solve's cores, the largest first, each to the least loaded,
    /// ties to the smaller root and the lower core. Each list holds its read-only roots first (`skip` of them), then its
    /// nodes in the sweep's own order, outlets first; `local` is every listed node's place in its list.
    #[must_use]
    pub fn basin_groups(&self) -> BasinGroups {
        let n = self.node_count();
        let threads = self.cores.threads();
        let mut in_order = vec![false; n];
        for &v in &self.order {
            in_order[v as usize] = true;
        }
        // The tree of every ordered node, outlets first: an outlet is its own root; a node's receiver outside the order
        // is a root the sweep never moves. `size` counts each root's ordered nodes.
        let mut tree = vec![NO_NODE; n];
        let mut size = vec![0u32; n];
        for &v in self.order.iter().rev() {
            let r = self.receiver[v as usize];
            let root = if r == NO_NODE {
                v
            } else if in_order[r as usize] {
                tree[r as usize]
            } else {
                r
            };
            tree[v as usize] = root;
            size[root as usize] += 1;
        }
        // The trees with a node the sweep moves (an outlet with no donor moves nothing and is left out), dealt to the
        // cores: the largest first, each to the least loaded.
        let mut trees: Vec<(u32, u32)> = (0..n as u32)
            .filter_map(|root| {
                // A node that roots no tree counts nothing; an ordered root counts itself, and moves not.
                let moving = size[root as usize].saturating_sub(u32::from(in_order[root as usize]));
                (moving > 0).then_some((moving, root))
            })
            .collect();
        trees.sort_by_key(|&(k, root)| (std::cmp::Reverse(k), root));
        let mut load = vec![0u64; threads];
        let mut core_of = vec![u32::MAX; n];
        for (k, root) in trees {
            let core = (0..threads).min_by_key(|&c| (load[c], c)).unwrap_or(0);
            load[core] += u64::from(k);
            core_of[root as usize] = core as u32;
        }
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); threads];
        let mut skip = vec![0usize; threads];
        // The read-only roots outside the order first, each once, in index order.
        for (root, &core) in core_of.iter().enumerate() {
            if (core != u32::MAX) & !in_order[root] {
                lists[core as usize].push(root as u32);
                skip[core as usize] += 1;
            }
        }
        for &v in self.order.iter().rev() {
            let core = core_of[tree[v as usize] as usize];
            if core != u32::MAX {
                lists[core as usize].push(v);
            }
        }
        let mut local = vec![NO_NODE; n];
        let mut group = vec![NO_NODE; n];
        for (g, list) in lists.iter().enumerate() {
            for (k, &v) in list.iter().enumerate() {
                local[v as usize] = k as u32;
            }
            for &v in list.iter().skip(skip[g]) {
                group[v as usize] = g as u32;
            }
        }
        BasinGroups {
            lists,
            skip,
            local,
            group,
        }
    }

    /// One core's share of a sweep ([`MacroSolve::sweep`]): its trees' nodes copied into cells, swept in the list's order
    /// (the solve's own, outlets first) by the joint law, and returned with the share's counts. A root (a read-only
    /// node, or an outlet) keeps its cell as it came.
    fn sweep_group(
        &self,
        list: &[u32],
        skip: usize,
        local: &[u32],
        active: Option<&[u32]>,
        law: &JointStep,
    ) -> (Vec<SweepCell>, SweepReport) {
        let mut cells: Vec<SweepCell> = list
            .iter()
            .map(|&v| {
                let i = v as usize;
                let r = self.receiver[i];
                SweepCell {
                    z: self.z[i],
                    z_flood: self.z_flood[i],
                    pit: self.pit[i],
                    ocean: self.ocean[i],
                    receiver: if r == NO_NODE {
                        NO_NODE
                    } else {
                        local[r as usize]
                    },
                    chord: self.chord[i],
                    discharge: self.discharge[i],
                    erodibility_q8: self.erodibility_q8[i],
                    active: active.map_or(ACTIVE_FULL, |a| a[i]),
                    removed: self.removed[i],
                    pending: self.pending[i],
                    cut: self.cut_now[i],
                }
            })
            .collect();
        let mut report = SweepReport::default();
        for k in skip..cells.len() {
            let r = cells[k].receiver;
            if r == NO_NODE {
                continue;
            }
            let base = cells[r as usize].base(self.sea_z);
            let cell = &mut cells[k];
            // ★ THE LAKES' CURE (ruling B46 item 2): a lake node's level becomes its receiver's base as it stands now
            // (the outlet's new height, or the next lake node's new level), never under the node's own floor — so a
            // lake's level follows its outlet as the outlet wears down, inside the sweep; the node is skipped while it
            // still stands under water and cut like any dry node once it is dry.
            if cell.pit > 0 {
                let level = base.max(cell.z);
                cell.z_flood = level;
                cell.pit = (i64::from(level) - i64::from(cell.z)) as u32;
                if cell.pit > 0 {
                    report.skipped_lake += 1;
                    continue;
                }
            }
            if (cell.active == 0) | (cell.z <= base) {
                continue;
            }
            let x0 = Gf::from_i64(i64::from(cell.z - base)) / law.steps_m;
            let length = Gf::from_i64(i64::from(cell.chord.max(1)));
            let discharge_m3_yr = Gf::from_i64(cell.discharge as i64) / Gf::from_i64(1_000);
            let rock = Gf::from_i64(i64::from(cell.erodibility_q8)) / law.basement;
            let c = law.dt_yr * Gf::from_i64(i64::from(cell.active))
                / Gf::from_i64(i64::from(ACTIVE_FULL))
                * law.k_lp
                * rock
                * discharge_m3_yr;
            let x = joint_root(x0, c, length, law.n_exp, law.tolerance);
            let cut = cell.z - (base + (x * law.steps_m).to_i64_floor() as i32);
            if cut > 0 {
                report.lowered += 1;
                report.max_cut = report.max_cut.max(cut);
                report.total_cut += cut as u64;
                cell.z -= cut;
                cell.removed = cell.removed.saturating_add(cut as u32);
                cell.pending = cell.pending.saturating_add(cut);
                cell.cut = cut as u32;
            }
        }
        (cells, report)
    }

    /// ★ THE JOINT LAW'S SWEEP (ruling B43 item 2), implicit and outlets first (Braun & Willett 2013): a node `x` metres
    /// over its receiver's base after the pass's uplift, `x₀`, settles where `x − x₀ + c·(x/L)^n = 0`, `c = Δt · K_lp ·
    /// share · Q` ([`JOINT_K_LP`], [`JOINT_N`]; `L` the chord to the receiver; `Q` the discharge in m³/yr; the share the
    /// rock's erodibility over the crystalline basement's and the node's own share of the pass under the landscape's
    /// time, `active`, in 1/[`ACTIVE_FULL`]), the root found by Newton from `x₀`. An outlet is left alone; a lake node
    /// follows its outlet ([`SweepCell::base`]) and is skipped while it stays under water — a lake does
    /// not cut — and then the SEDIMENT FILLS IT: the closed hollows of the budget's water where `basins` states them
    /// ([`MacroSolve::deposit_basins`]), the routing's lakes otherwise ([`MacroSolve::deposit`]).
    ///
    /// **Example.** A river leaves the belt's wedge with 3 km³ a year over a 2 % slope on gneiss: in a quarter-million
    /// year sweep it cuts its node about 30 m toward the node below, and a node beside it that drains only its own
    /// square cuts a few metres.
    pub fn sweep(
        &mut self,
        dt_yr: Gf,
        active: Option<&[u32]>,
        basins: Option<&Basins>,
    ) -> SweepReport {
        let groups = self.basin_groups();
        self.sweep_grouped(dt_yr, active, basins, &groups)
    }

    /// [`MacroSolve::sweep`] over river basins already dealt ([`MacroSolve::basin_groups`]): the deal depends only on
    /// the routing (the receivers and the order), so the sweeps of one routing share it.
    pub fn sweep_grouped(
        &mut self,
        dt_yr: Gf,
        active: Option<&[u32]>,
        basins: Option<&Basins>,
        groups: &BasinGroups,
    ) -> SweepReport {
        let mut report = SweepReport::default();
        let law = JointStep {
            dt_yr,
            steps_m: Gf::from_i64(i64::from(Z_STEPS_PER_M)),
            basement: Gf::from_i64(i64::from(
                crate::strata::Province::CrystallineBasement.erodibility_q8(),
            )),
            k_lp: Gf::from_f64(JOINT_K_LP),
            n_exp: Gf::from_f64(JOINT_N),
            tolerance: Gf::from_f64(JOINT_TOLERANCE_M),
        };
        self.cut_now.iter_mut().for_each(|c| *c = 0);
        if let Some(b) = basins {
            report.drained = self.drain_to_saddles(b);
        }
        // ★ THE JOINT LAW BY RIVER BASIN ON THE SOLVE'S CORES (ruling B64): a node reads only its own state and its
        // receiver's, and both lie in one drainage tree, so whole trees go to each thread; each thread copies its trees'
        // nodes into its own cells, sweeps them outlets first in the solve's own order, and the cells come back in a
        // fixed order — the very state one thread leaves.
        let swept = crate::par::per_range(groups.lists.len(), self.cores.threads(), |r| {
            r.map(|g| {
                self.sweep_group(
                    &groups.lists[g],
                    groups.skip[g],
                    &groups.local,
                    active,
                    &law,
                )
            })
            .collect::<Vec<_>>()
        });
        let swept: Vec<(Vec<SweepCell>, SweepReport)> = swept.into_iter().flatten().collect();
        for (_, part) in &swept {
            report.lowered += part.lowered;
            report.skipped_lake += part.skipped_lake;
            report.max_cut = report.max_cut.max(part.max_cut);
            report.total_cut += part.total_cut;
        }
        // ★ The swept cells gathered back ON THE SOLVE'S CORES (ruling B72 item 1): each node reads the cell its list
        //   swept, or keeps its own values where no list swept it.
        let (n, threads) = (self.node_count(), self.cores.threads());
        let cell = |i: usize| -> Option<&SweepCell> {
            let g = groups.group[i];
            if g == NO_NODE {
                return None;
            }
            Some(&swept[g as usize].0[groups.local[i] as usize])
        };
        self.z = crate::par::map(n, threads, |i| cell(i).map_or(self.z[i], |c| c.z));
        self.z_flood = crate::par::map(n, threads, |i| {
            cell(i).map_or(self.z_flood[i], |c| c.z_flood)
        });
        self.pit = crate::par::map(n, threads, |i| cell(i).map_or(self.pit[i], |c| c.pit));
        self.removed = crate::par::map(n, threads, |i| {
            cell(i).map_or(self.removed[i], |c| c.removed)
        });
        self.pending = crate::par::map(n, threads, |i| {
            cell(i).map_or(self.pending[i], |c| c.pending)
        });
        self.cut_now = crate::par::map(n, threads, |i| cell(i).map_or(self.cut_now[i], |c| c.cut));
        drop(swept);
        let deposit = match basins {
            Some(b) => self.deposit_basins_grouped(b, groups),
            None => {
                let (filled, total_fill) = self.deposit();
                DepositReport {
                    filled,
                    total_fill,
                    ..DepositReport::default()
                }
            }
        };
        report.filled = deposit.filled;
        report.total_fill = deposit.total_fill;
        report.rock_to_sea = deposit.to_sea;
        report.rock_over_spill = deposit.over_spill;
        report.rock_dropped = deposit.dropped;
        report.rock_stuck = deposit.stuck;
        report.rock_remainder = deposit.remainder;
        report.rock_refused = deposit.refused;
        report
    }

    /// ★ THE DEPOSIT (2026-09-22, the lake census: 16 % of the land was lake against Earth's 3.7 %,
    /// because a hollow the flood found was never filled and never breached). This pass's cut is
    /// carried down the receiver tree, leaves first, as a VOLUME (steps × area); at a LAKE node
    /// (its flood level over its terrain) the volume fills the node up to its spill level and the
    /// rest goes on to the receiver; at an outlet — the sea — what is left sinks (a delta is 8d's
    /// own later piece). Exact integers, order-free on the tree. Returns the lake nodes filled and
    /// the fill summed, in steps.
    ///
    /// The fill is a LOAD the rebound does not read (the rebound answers removal only); named here,
    /// UNMEASURED, because a filled lake is a few hundred metres over its own basin and the flexure
    /// of that stands far under a node's step.
    ///
    /// **Example.** A crater bowl 300 m deep sits under a range the rivers cut. Every pass the
    /// range's cut runs into the bowl; after a few passes the bowl is a plain at the spill level,
    /// the next flood finds no pit, and the river that fed it now runs across.
    pub fn deposit(&mut self) -> (usize, u64) {
        let n = self.node_count();
        let mut flux = vec![0u128; n];
        let (mut filled, mut total_fill) = (0usize, 0u64);
        for k in 0..self.order.len() {
            let node = self.order[k];
            let i = node as usize;
            let area = u128::from(self.area[i].max(1));
            flux[i] += u128::from(self.cut_now[i]) * area;
            if self.pit[i] > 0 && flux[i] > 0 {
                let room = u128::from(self.pit[i]) * area;
                let laid = flux[i].min(room);
                let dz = (laid / area) as i32;
                if dz > 0 {
                    self.z[i] += dz;
                    self.pit[i] -= dz as u32;
                    // A fill is a load the rebound answers with the opposite sign: without it the
                    // crust rose under mass that never left (MEASURED: the sea floor lifted over
                    // the sea on 1.4 million nodes and the re-solved sea drowned 99 % of the globe).
                    self.pending[i] = self.pending[i].saturating_sub(dz);
                    flux[i] -= u128::from(dz as u32) * area;
                    filled += 1;
                    total_fill += dz as u64;
                }
            }
            let r = self.receiver[i];
            if r != NO_NODE {
                flux[r as usize] += flux[i];
            }
        }
        (filled, total_fill)
    }

    /// ★ THE WATER'S OWN PATHS FOR THE EROSION (ruling B51 item 2; ruling B8 in the passes): the solve's own lakes'
    /// budget and water tree over the routing that stands — the receivers, their chords and their order become the
    /// water's, and the discharge is each node's RAIN on its area summed down them with no sink (the joint law's own
    /// `Q = A·R`, `A` the area that truly drains to the node), so a CLOSED lake ends its rivers and its basin's rain
    /// reaches no river below its rim. THE LAKES BECOME THE BUDGET'S: each land node's flood level is the budget's
    /// standing water, or its own ground where none stands, so the sweep skips only water and cuts a closed hollow's
    /// dry rim down to its lake, and the deposit fills a lake only up to its water. Returns the closed hollows for the
    /// basin fill ([`MacroSolve::deposit_basins`]); a body with no water takes the dry tree and keeps the routing's
    /// hollows, and answers `None`.
    pub fn follow_the_budget(
        &mut self,
        climate: &crate::climate::Climate,
        has_water: bool,
    ) -> Option<Basins> {
        let rain = self.rain.clone();
        if !has_water {
            let tree = crate::lakes::dry_tree(self);
            let _ = self.follow_the_water(tree, &rain);
            return None;
        }
        let lakes = crate::lakes::budget(self, climate);
        let mut tree = crate::lakes::water_tree(self, &lakes);
        tree.sinks.clear();
        let _ = self.follow_the_water(tree, &rain);
        Some(self.lakes_from_budget(&lakes))
    }

    /// The budget's standing water as the solve's lakes (see [`MacroSolve::follow_the_budget`]); returns the closed
    /// hollows: for each closed body (a hollow whose inflow stays under what its water hands the air), its nodes —
    /// every node whose own depression it covers, the lake and its dry rim, as the budget's own level solve gathers
    /// them — and its spill level.
    fn lakes_from_budget(&mut self, lakes: &crate::lakes::Lakes) -> Basins {
        use crate::lakes::{NO_DEP, OCEAN};
        let (n, threads) = (self.node_count(), self.cores.threads());
        // ★ Each land node's level ON THE SOLVE'S CORES (ruling B72 item 1): the budget's water over it, or its own
        //   ground; an ocean node keeps the routing's.
        let level = |i: usize| -> i32 { lakes.water[i].max(self.z[i]) };
        self.z_flood = crate::par::map(n, threads, |i| {
            [level(i), self.z_flood[i]][usize::from(self.ocean[i])]
        });
        self.pit = crate::par::map(n, threads, |i| {
            [(level(i) - self.z[i]) as u32, self.pit[i]][usize::from(self.ocean[i])]
        });
        let mut slot_of_dep = vec![u32::MAX; lakes.deps.len()];
        let mut bodies: Vec<(Vec<u32>, i32)> = Vec::new();
        let mut dep_of_body: Vec<usize> = Vec::new();
        for (d, dep) in lakes.deps.iter().enumerate().skip(1) {
            // ★ Every hollow that does not spill is a body the rock can fill: a WATER body (it holds its own lake, its
            // dry rim with it) or, where its water stands in its children, its own DRY RING, which the rock rises into
            // once its children are full of rock ([`MacroSolve::deposit_basins`]).
            let b = lakes.body_of[d];
            if !dep.full & ((b == d as u32) | (b == NO_DEP)) {
                slot_of_dep[d] = bodies.len() as u32;
                bodies.push((Vec::new(), dep.spill_z));
                dep_of_body.push(d);
            }
        }
        // The closed body a depression's nodes belong to — its water body, or its own dry ring where it holds no water —
        // or `u32::MAX` (the ocean, none, or a hollow that spills).
        let body_slot = |d: u32| -> u32 {
            if (d == NO_DEP) | (d == OCEAN) {
                return u32::MAX;
            }
            let b = lakes.body_of[d as usize];
            slot_of_dep[if b == NO_DEP { d } else { b } as usize]
        };
        let mut slot = vec![u32::MAX; n];
        for (i, (&h, here)) in lakes.home.iter().zip(slot.iter_mut()).enumerate() {
            let s = body_slot(h);
            if s != u32::MAX {
                *here = s;
                bodies[s as usize].0.push(i as u32);
            }
        }
        // ★ WHERE A BODY'S ROCK GOES ONCE IT IS FULL OF ROCK ([`MacroSolve::deposit_basins`]): the ways over its saddle
        // in the saddle's own order (the water's), its own side left out, and its parent's ring.
        let mut sides = Vec::with_capacity(bodies.len());
        let mut parent = Vec::with_capacity(bodies.len());
        let mut parent_gate = Vec::with_capacity(bodies.len());
        for &d in &dep_of_body {
            let dep = &lakes.deps[d];
            let ways: Vec<(u32, u32)> = lakes
                .saddles
                .get(dep.saddle as usize)
                .map_or(&[][..], |s| &s.sides[..])
                .iter()
                .filter(|k| k.root != d as u32)
                .map(|k| {
                    let b = body_slot(k.root);
                    let to = if k.root == OCEAN {
                        OVER_TO_SEA
                    } else if b == u32::MAX {
                        OVER_TO_FULL
                    } else {
                        b
                    };
                    (to, k.gate)
                })
                .collect();
            sides.push(ways);
            parent.push(body_slot(dep.parent));
            // Where the parent spills, the rock enters its ring at this hollow's saddle node and follows the parent's
            // own water across its lake to its spill and on.
            parent_gate.push(
                lakes
                    .saddles
                    .get(dep.saddle as usize)
                    .map_or(NO_NODE, |s| s.node),
            );
        }
        Basins {
            slot,
            bodies,
            sides,
            parent,
            parent_gate,
        }
    }

    /// ★ THE DEPOSIT WITH THE BUDGET'S CLOSED HOLLOWS (ruling B51 item 2): the solve's own deposit, line for line, down
    /// the water's own paths — at a lake node the volume fills the node up to its water and the rest goes on — except
    /// that the rock reaching a CLOSED hollow's end (a node under its lake, or its dry pit, where the water's path
    /// stops) stays in that hollow: the hollow is then filled as ONE body, its lowest nodes raised to a common level the
    /// volume pays for, never over its spill ([`level_fill`]) — a closed basin keeps its rock and fills from its floor
    /// up, as its water does. Rock at the sea leaves.
    ///
    /// ★ A HOLLOW FULL OF ROCK PASSES THE REST ON (2026-10-04, the convergence gate's cause; ruling B51's own words:
    /// rock past a full hollow's spill leaves): once a closed body's fill reaches its spill, the rock left over goes
    /// over its saddle the way its water would go if it spilled — to the steepest side that is the ocean's or a closed
    /// body not yet full of rock ([`Basins::sides`]), down the water's way from that side's gate, filling every lake's
    /// room on the way, into the sea or into another closed body, which fills in its turn; where every side is full of
    /// rock, the rock rises into the parent's ring, and the parent fills as ONE body with its full children up to the
    /// parent's own spill ([`Basins::parent`]), or leaves by the parent's own exit ([`Basins::parent_gate`]).
    ///
    /// ★ THE REST PASSES EACH BODY ONCE (2026-10-05: a run of the belts' bench spun 90 minutes in this work list,
    /// because a side's water ran back into the full body itself, and the rest went round that side for ever). The rest
    /// is passed on at once along its whole way ([`MacroSolve::pass_on`]): into the sea, into a closed body not yet full
    /// of rock, which keeps it, or THROUGH a body already full of rock, which passes it on by its own way in turn. A way
    /// that ends in a body this rest has already passed (the full body itself among them), at an inland end no body
    /// holds, or on a ring of receivers is refused and counted, and the next way is tried. So the cascade ENDS by
    /// construction: one rest passes each body at most once, and every body the work list fills again is one not yet
    /// full of rock (it turns full once, for good) or a parent a full child rose into (once). The test
    /// `a_way_that_runs_back_into_a_full_hollow_is_refused` hangs without the rule. MEASURED: refusing every way that
    /// ends in a body full of rock instead (this cure's first form) dropped 1.15 × 10¹⁶ m³ of rock in one solve of the
    /// home planet (81 085 ways refused) — a full body is no dead end: it passes the rock on.
    /// MEASURED before (the belts' bench's rock meter, the home planet): the level fill DROPPED every volume over the
    /// spill — 125.6 × 10¹⁵ m³ of the 754 × 10¹⁵ m³ the rivers cut (17 %) vanished, less on an axis twice as fine
    /// (112.4), so the world depended on the pass length (ruling B39). Returns the deposit's own reading.
    ///
    /// **Example.** A salt pan on the plateau's lee holds 2 km³ under its sill and its range sends 5 km³ in one pass:
    /// the pan fills to its sill, and the other 3 km³ run over the sill into the next pan down the water's way, which
    /// keeps what it has room for and passes the rest on to the sea.
    pub fn deposit_basins(&mut self, basins: &Basins) -> DepositReport {
        let groups = self.basin_groups();
        self.deposit_basins_grouped(basins, &groups)
    }

    /// [`MacroSolve::deposit_basins`] over river basins already dealt ([`MacroSolve::basin_groups`]: the sweep's own
    /// deal, which the deposit after it shares). ★ THE WALK DOWN THE TREES ON THE SOLVE'S CORES (ruling B72 item 1): a
    /// node's rock reaches only its own receiver, and both lie in one drainage tree, so each core walks its own trees,
    /// leaves first in the solve's own order, and lists what it laid and where its rock left the trees; an outlet that
    /// no tree holds (no donor) carries only its own cut. The lists are applied after the walk — each node is raised
    /// once, by the one core that walked it, and every sum is an integer sum — so the state is the one a single core
    /// leaves. The bodies' fill after the walk passes rock between bodies, and stays one body after another.
    pub fn deposit_basins_grouped(
        &mut self,
        basins: &Basins,
        groups: &BasinGroups,
    ) -> DepositReport {
        let nb = basins.bodies.len();
        let threads = self.cores.threads();
        let mut w = RockWork {
            owner: (0..nb as u32).collect(),
            nodes: basins.bodies.iter().map(|(m, _)| m.clone()).collect(),
            held: vec![0u128; nb],
            full: vec![false; nb],
            due: Vec::new(),
        };
        let mut out = DepositReport::default();
        let trees = crate::par::per_range(groups.lists.len(), threads, |r| {
            r.map(|g| self.deposit_group(&groups.lists[g], groups.skip[g], &groups.local, basins))
                .collect::<Vec<_>>()
        });
        let lone = crate::par::per_range(self.node_count(), threads, |r| {
            let mut part = WalkPart::default();
            for i in r {
                if (self.cut_now[i] == 0)
                    | (groups.local[i] != NO_NODE)
                    | (self.receiver[i] != NO_NODE)
                {
                    continue;
                }
                let cut = u128::from(self.cut_now[i]) * u128::from(self.area[i].max(1));
                self.walk_step(i, cut, basins, &mut part);
            }
            part
        });
        for part in trees.into_iter().flatten().chain(lone) {
            for (v, dz) in part.raised {
                let i = v as usize;
                self.z[i] += dz;
                self.pit[i] -= dz as u32;
                self.pending[i] = self.pending[i].saturating_sub(dz);
            }
            for (slot, f) in part.held {
                w.held[slot as usize] += f;
            }
            out.filled += part.out.filled;
            out.total_fill += part.out.total_fill;
            out.to_sea += part.out.to_sea;
            out.dropped += part.out.dropped;
        }
        // THE BODIES, as a work list: a body due to fill takes its rock, and the rest of a body that then stands at its
        // spill (full of rock) passes on ([`MacroSolve::pass_on`]).
        w.due = (0..nb as u32)
            .rev()
            .filter(|&b| w.held[b as usize] > 0)
            .collect();
        while let Some(b) = w.due.pop() {
            let b = representative(&w.owner, b) as usize;
            let volume = std::mem::take(&mut w.held[b]);
            let spill = self.spill_now(basins, b);
            let (paid, at_spill) = self.fill_body(&mut w.nodes[b], volume, spill, &mut out);
            let rest = volume - paid;
            // Under the spill, the rest is the level's own remainder (under one sixteenth of the raised area).
            out.remainder += rest * u128::from(!at_spill);
            w.full[b] |= at_spill;
            if !at_spill | (rest == 0) {
                continue;
            }
            out.over_spill += rest;
            self.pass_on(b, rest, basins, &mut w, &mut out);
        }
        out
    }

    /// ★ THE REST PASSED ON from the body `from`, which stands at its spill full of rock ([`MacroSolve::deposit_basins`]):
    /// over the body's saddle the way its water would go if it spilled — the first side that is the ocean's or a closed
    /// body not yet full of rock ([`Basins::sides`]) — and down the water's way from that side's gate, filling every
    /// lake's room on the way: into the sea; into a closed body not yet full of rock, which keeps it (due to fill); or
    /// THROUGH a body already full of rock, which passes it on by its own way in turn. Where no side's way is left, the
    /// rock rises into the parent's ring ([`Basins::parent`]) or leaves by the parent's own exit
    /// ([`Basins::parent_gate`]), and else it is dropped and counted (`stuck`). The rest passes each body at most once:
    /// a way back to a body it passed is refused ([`MacroSolve::way_end`]).
    fn pass_on(
        &mut self,
        from: usize,
        mut volume: u128,
        basins: &Basins,
        w: &mut RockWork,
        out: &mut DepositReport,
    ) {
        let nb = basins.bodies.len();
        let mut passed = vec![from as u32];
        let mut b = from;
        loop {
            let mut way = None;
            for &(to, gate) in basins.sides.get(b).map_or(&[][..], Vec::as_slice) {
                let side_open = (to == OVER_TO_SEA)
                    | (((to as usize) < nb) && !w.full[representative(&w.owner, to) as usize]);
                if !side_open {
                    continue;
                }
                match self.way_end(gate, basins, &w.owner, &passed) {
                    Some(end) => {
                        way = Some((gate, end));
                        break;
                    }
                    None => out.refused += 1,
                }
            }
            if way.is_none() {
                let p = basins.parent.get(b).copied().unwrap_or(u32::MAX);
                if (p as usize) < nb {
                    // No side's way is left: the rock rises into the parent's ring, which fills as one body with its
                    // full children.
                    let p = representative(&w.owner, p) as usize;
                    w.owner[b] = p as u32;
                    let children = std::mem::take(&mut w.nodes[b]);
                    w.nodes[p].extend(children);
                    w.held[p] += volume;
                    w.due.push(p as u32);
                    return;
                }
                // The parent's own exit.
                let gate = basins.parent_gate.get(b).copied().unwrap_or(NO_NODE);
                if gate != NO_NODE {
                    way = self
                        .way_end(gate, basins, &w.owner, &passed)
                        .map(|end| (gate, end));
                    out.refused += usize::from(way.is_none());
                }
            }
            let Some((gate, end)) = way else {
                out.dropped += volume;
                out.stuck += volume;
                return;
            };
            let mut v = gate;
            loop {
                self.fill_room(v as usize, &mut volume, out);
                if v == end {
                    break;
                }
                v = self.receiver[v as usize];
            }
            let s = basins.slot[end as usize];
            if s == u32::MAX {
                out.to_sea += volume;
                return;
            }
            let r = representative(&w.owner, s);
            if !w.full[r as usize] {
                w.held[r as usize] += volume;
                w.due.push(r);
                return;
            }
            // Through a body full of rock: it passes the rest on by its own way.
            passed.push(r);
            b = r as usize;
        }
    }

    /// One core's walk down its trees ([`MacroSolve::deposit_basins_grouped`]): the group's nodes leaves first (its list
    /// read backwards, its read-only roots last and never walked), each node's rock gathered from its donors, its room
    /// filled, the rest handed to its receiver — or, at the tree's end, booked to the closed body, the sea, or as
    /// dropped. A receiver that is a read-only root (on a ring of receivers) takes the rock and never passes it on.
    fn deposit_group(&self, list: &[u32], skip: usize, local: &[u32], basins: &Basins) -> WalkPart {
        let mut flux = vec![0u128; list.len()];
        let mut part = WalkPart::default();
        for k in (skip..list.len()).rev() {
            let i = list[k] as usize;
            let cut = u128::from(self.cut_now[i]) * u128::from(self.area[i].max(1));
            let f = self.walk_step(i, flux[k] + cut, basins, &mut part);
            let r = self.receiver[i];
            if r != NO_NODE {
                flux[local[r as usize] as usize] += f;
            }
        }
        part
    }

    /// One node of the deposit's walk: the room it holds filled from the rock `volume` reaching it (listed in `part`,
    /// applied after the walk), and at a tree's end (no receiver) the rest booked — to the node's closed body, to the
    /// sea, or as dropped. Returns the rest.
    fn walk_step(&self, i: usize, mut volume: u128, basins: &Basins, part: &mut WalkPart) -> u128 {
        let area = u128::from(self.area[i].max(1));
        let dz = self.room_dz(i, volume);
        if dz > 0 {
            volume -= u128::from(dz as u32) * area;
            part.raised.push((i as u32, dz));
            part.out.filled += 1;
            part.out.total_fill += dz as u64;
        }
        if self.receiver[i] == NO_NODE {
            if basins.slot[i] != u32::MAX {
                part.held.push((basins.slot[i], volume));
            } else if self.ocean[i] {
                part.out.to_sea += volume;
            } else {
                part.out.dropped += volume;
            }
        }
        volume
    }

    /// The steps a node's lake room takes from the rock `volume` reaching it: what the volume pays for over the node's
    /// area, never over its water.
    fn room_dz(&self, i: usize, volume: u128) -> i32 {
        let area = u128::from(self.area[i].max(1));
        let room = u128::from(self.pit[i]) * area;
        (volume.min(room) / area) as i32
    }

    /// One node's lake room filled from the volume passing it (the deposit's own step): the node rises by what the
    /// volume pays for over its area, never over its water.
    fn fill_room(&mut self, i: usize, volume: &mut u128, out: &mut DepositReport) {
        let area = u128::from(self.area[i].max(1));
        let dz = self.room_dz(i, *volume);
        if dz > 0 {
            self.z[i] += dz;
            self.pit[i] -= dz as u32;
            self.pending[i] = self.pending[i].saturating_sub(dz);
            *volume -= u128::from(dz as u32) * area;
            out.filled += 1;
            out.total_fill += dz as u64;
        }
    }

    /// The node the water's way from `from` ends at, where the rest may go there ([`MacroSolve::pass_on`]): the sea, or a
    /// closed body the rest has not passed (full of rock or not); `None` where the way ends in a body in `passed` — the
    /// full body itself among them, where a side's water runs back into it — at an inland end no body holds, or on a
    /// ring of receivers (a way longer than the body's node count).
    fn way_end(&self, from: u32, basins: &Basins, owner: &[u32], passed: &[u32]) -> Option<u32> {
        let mut v = from;
        for _ in 0..self.node_count() {
            let i = v as usize;
            if self.receiver[i] == NO_NODE {
                let s = basins.slot[i];
                let open = if s == u32::MAX {
                    self.ocean[i]
                } else {
                    !passed.contains(&representative(owner, s))
                };
                return open.then_some(v);
            }
            v = self.receiver[i];
        }
        None
    }

    /// One closed body filled from its floor with `volume` ([`level_fill`]): returns the volume laid and whether the
    /// level stands at the spill.
    fn fill_body(
        &mut self,
        nodes: &mut [u32],
        volume: u128,
        spill: i32,
        out: &mut DepositReport,
    ) -> (u128, bool) {
        nodes.sort_unstable_by_key(|&v| (self.z[v as usize], v));
        let heights: Vec<(i32, u64)> = nodes
            .iter()
            .map(|&v| (self.z[v as usize], self.area[v as usize].max(1)))
            .collect();
        let level = level_fill(&heights, volume, spill);
        let mut paid = 0u128;
        for &v in nodes.iter() {
            let i = v as usize;
            if self.z[i] >= level {
                break;
            }
            let dz = level - self.z[i];
            self.z[i] += dz;
            self.pit[i] = self.pit[i].saturating_sub(dz as u32);
            self.pending[i] = self.pending[i].saturating_sub(dz);
            paid += u128::from(dz as u32) * u128::from(self.area[i].max(1));
            out.filled += 1;
            out.total_fill += dz as u64;
        }
        (paid, level == spill)
    }

    /// The pyramid level whose node is nearest `alpha_m` (the flexural parameter), among the
    /// levels the edge halves to.
    #[must_use]
    pub fn coarse_level(&self, alpha_m: f64) -> u32 {
        let mut best = 0u32;
        let mut best_distance = f64::MAX;
        let mut k = 0u32;
        while let Some(coarse) = self.lattice.coarser(k) {
            let distance = (coarse.node_m() - alpha_m).abs();
            if distance < best_distance {
                best_distance = distance;
                best = k;
            }
            k += 1;
        }
        best
    }

    /// ★ THE FLEXURAL REBOUND (03 §4.7): the removal since the last rebound is restricted up the
    /// pyramid to the level nearest the flexural parameter `alpha_m` (the mean over each block),
    /// smoothed there by [`SMOOTH_PASSES`] nine-node means through the seam table, and `ρ_crust /
    /// ρ_mantle` of it is added to the terrain — the crust floats up under the lightened load.
    /// ★ WITHOUT ITS FLOORS (ruling B45 item 1): the means are exact, and each node's fraction of a
    /// sixteenth is carried to its next rebound ([`MacroSolve::rebound_carry`]) — MEASURED, the
    /// floored form lost an eighth of a metre a node at every rebound, so its answer moved with the
    /// rebound's spacing. Returns the level used, the nodes lifted and the greatest lift, in
    /// sixteenths.
    ///
    /// ★ THE SMOOTH READ (phase 1 of the ground step; the belts' design L10; ruling B37 item 3,
    /// which MEASURED 131 km grid steps where each node took its block's lift whole). Each block
    /// owes the lift its old read gave — its nodes times the crust's share of its smoothed field, in
    /// whole sixteenths — and hands it over the nodes around its centre by the BILINEAR weights of
    /// their positions in the face's index space: a node reads its own block's centre and the three
    /// next to it on its own side, across a face seam through the coarse lattice's own neighbour
    /// table (the partner face's axes as the table states them), and at a cube corner, where the
    /// diagonal centre does not exist, from the three that do (the diagonal's weight moves to the
    /// two side centres and leaves the own one, the plane through three points). The hand-over runs
    /// in the nodes' index order with each block's running sum, so every block lifts EXACTLY what it
    /// owes and the total lift is conserved to the sixteenth (review finding P14).
    ///
    /// **Example.** The home planet's rivers cut a 400 m valley into a range; the rebound lifts the
    /// 150 km around it by about 40 m, so the ridge beside the valley ends up TALLER than it
    /// started, and the range keeps its snow line instead of wearing down into hills — and the lift
    /// fades across the next block's ground as a ramp, never as a 131 km step a pilot sees from
    /// orbit.
    pub fn rebound(&mut self, alpha_m: f64) -> (u32, usize, i32) {
        let n = self.node_count();
        let k = self.coarse_level(alpha_m);
        let coarse = self.lattice.coarser(k).unwrap_or(self.lattice);
        let m = coarse.node_count();
        // 1. Restrict: the mean of each 2^k × 2^k block, exact (ruling B45 item 1: no floor).
        // The field is SIGNED (2026-09-22): a block whose sediment outweighs its cut sinks.
        // ★ ON THE SOLVE'S CORES (ruling B64): by bands of whole block rows, so each block sums its own nodes in index
        // order on one thread and the bands' fields join in order — the field one thread sums.
        let rows = 6 * coarse.edge as usize;
        let row_nodes = (self.lattice.edge as usize) << k;
        let bands = crate::par::per_range(rows, self.cores.threads(), |r| {
            let mut field = vec![Gf::ZERO; r.len() * coarse.edge as usize];
            let first = (r.start * coarse.edge as usize) as u32;
            let mut parent = Vec::with_capacity(r.len() * row_nodes);
            for node in (r.start * row_nodes) as u32..(r.end * row_nodes) as u32 {
                let (face, i, j) = self.lattice.split(node);
                let c = coarse.index(face, i >> k, j >> k);
                parent.push(c);
                field[(c - first) as usize] += Gf::from_i32(self.pending[node as usize]);
            }
            (field, parent)
        });
        let mut field = Vec::with_capacity(m);
        let mut parent = Vec::with_capacity(n);
        for (f, p) in bands {
            field.extend(f);
            parent.extend(p);
        }
        self.pending.iter_mut().for_each(|p| *p = 0);
        let block = Gf::from_i64(1i64 << (2 * k));
        for f in &mut field {
            *f = *f / block;
        }
        // 2. Smooth: the nine-node mean, a stated number of times, through the seam table, exact.
        for _ in 0..SMOOTH_PASSES {
            let next: Vec<Gf> = (0..m as u32)
                .map(|c| {
                    let mut sum = field[c as usize];
                    let mut count = Gf::ONE;
                    for nb in coarse.neighbours(c) {
                        if nb != NO_NODE {
                            sum += field[nb as usize];
                            count += Gf::ONE;
                        }
                    }
                    sum / count
                })
                .collect();
            field = next;
        }
        // 3. What each block owes — or sinks, under a load: its nodes times the crust's share.
        let share = Gf::from_f64(
            crate::land::CONTINENTAL_CRUST_DENSITY_KGM3 / crate::land::MANTLE_DENSITY_KGM3,
        );
        let owed: Vec<Gf> = field.iter().map(|&f| f * share * block).collect();
        drop(field);
        // 4. ★ THE SMOOTH READ, and each node's fraction of a sixteenth carried to its next rebound.
        let ring: Vec<[u32; 8]> = (0..m as u32).map(|c| coarse.neighbours(c)).collect();
        let lifts = self.smooth_lifts(k, &parent, &ring, &owed);
        let (mut lifted, mut max_lift) = (0usize, 0i32);
        for ((z, carry), &lift) in self.z.iter_mut().zip(&mut self.rebound_carry).zip(&lifts) {
            let exact = lift + *carry;
            let whole = exact.floor();
            *carry = exact - whole;
            let step = whole.to_i64_floor() as i32;
            if step != 0 {
                *z += step;
                lifted += 1;
                max_lift = max_lift.max(step);
            }
        }
        (k, lifted, max_lift)
    }

    /// ★ THE HAND-OVER of the smooth read (see [`MacroSolve::rebound`]): each coarse block's `owed` lift, sixteenths
    /// summed over its nodes, handed to the nodes that read it in proportion to their [`MacroSolve::smooth_read`]
    /// weights over the block's own weight sum — so each block hands out exactly what it owes (to the float's last
    /// bit) and the total lift is conserved (review finding P14). Returns each node's lift, in sixteenths.
    fn smooth_lifts(&self, k: u32, parent: &[u32], ring: &[[u32; 8]], owed: &[Gf]) -> Vec<Gf> {
        let n = self.node_count();
        let threads = self.cores.threads();
        // ★ ON THE SOLVE'S CORES (ruling B64): the weights are whole numbers, so each range's sums add in any order to
        // the same total; each node's lift reads only the sums and its own read.
        let parts = crate::par::per_range(n, threads, |r| {
            let mut sums = vec![0u64; ring.len()];
            for node in r.start as u32..r.end as u32 {
                for (c, w) in self.smooth_read(node, k, parent, ring) {
                    sums[c as usize] += u64::from(w);
                }
            }
            sums
        });
        let mut weight_sum = vec![0u64; ring.len()];
        for part in parts {
            for (sum, w) in weight_sum.iter_mut().zip(part) {
                *sum += w;
            }
        }
        crate::par::map(n, threads, |node| {
            let mut lift = Gf::ZERO;
            for (c, w) in self.smooth_read(node as u32, k, parent, ring) {
                let c = c as usize;
                lift += owed[c] * Gf::from_i64(i64::from(w)) / Gf::from_i64(weight_sum[c] as i64);
            }
            lift
        })
    }

    /// ★ ONE NODE'S SMOOTH READ of the coarse level `k` (see [`MacroSolve::rebound`]): its own
    /// block's centre `parent[node]` and the three centres next to it on the node's own side, each
    /// with its bilinear weight in units of `1 / (2^(k+1))²` (they sum to that square); `ring` is the
    /// coarse lattice's neighbour table. A centre that does not exist (the diagonal at a cube corner)
    /// gives its weight to the two side centres and takes it from the own one.
    fn smooth_read(&self, node: u32, k: u32, parent: &[u32], ring: &[[u32; 8]]) -> [(u32, u32); 4] {
        let (_, i, j) = self.lattice.split(node);
        let half = 1i32 << k;
        let d = 2 * half;
        // The node's centre from its block's centre, in halves of a fine node: odd, under `d`.
        let a = 2 * (i & (half - 1)) + 1 - half;
        let b = 2 * (j & (half - 1)) + 1 - half;
        let (di, dj) = (if a < 0 { -1 } else { 1 }, if b < 0 { -1 } else { 1 });
        let (a, b) = (a.abs(), b.abs());
        // The stencil's slot of an offset: its row-major place among the nine, the middle left out
        // (`crate::macro_lattice::STENCIL`'s own order).
        let slot = |si: i32, sj: i32| {
            let r = ((sj + 1) * 3 + (si + 1)) as usize;
            r - usize::from(r > 4)
        };
        let c = parent[node as usize];
        let around = ring[c as usize];
        let (side_i, side_j, diag) = (
            around[slot(di, 0)],
            around[slot(0, dj)],
            around[slot(di, dj)],
        );
        let (mut w_c, mut w_i, mut w_j, mut w_d) =
            ((d - a) * (d - b), a * (d - b), (d - a) * b, a * b);
        if diag == NO_NODE {
            w_i += w_d;
            w_j += w_d;
            w_c -= w_d;
            w_d = 0;
        }
        [
            (c, w_c as u32),
            (side_i, w_i as u32),
            (side_j, w_j as u32),
            (if diag == NO_NODE { c } else { diag }, w_d as u32),
        ]
    }

    /// ★ THE TALUS (03 §4.8): nothing stands steeper than its angle of repose. For every node the
    /// EXCESS over each lower neighbour is the drop past `chord · tan θ`; half the LARGEST excess
    /// leaves the node, split among the lower neighbours in proportion to their excess, booked in
    /// a second buffer and applied at the end, so the answer never depends on the visiting order
    /// and the mass is conserved to the sixteenth. `tan_q16` is the node's tangent in 1/65536 (its
    /// aridity's, [`tan_repose_q16`]). Returns the nodes that shed and the greatest single excess
    /// seen before the pass, in sixteenths — the number that must fall pass after pass.
    pub fn talus(&mut self, tan_q16: &[u32]) -> (usize, i64) {
        let n = self.node_count();
        let mut delta = vec![0i64; n];
        let (mut shed, mut worst) = (0usize, 0i64);
        for node in 0..n as u32 {
            let i = node as usize;
            let ring = self.lattice.neighbours(node);
            let mut excess = [0i64; 8];
            let mut total = 0i64;
            let mut largest = 0i64;
            for (slot, &m) in ring.iter().enumerate() {
                if m == NO_NODE {
                    continue;
                }
                let drop = i64::from(self.z[i]) - i64::from(self.z[m as usize]);
                if drop <= 0 {
                    continue;
                }
                // chord [m] × 16 [sixteenths a metre] × tan [1/2¹⁶] = chord × tan >> 12.
                let limit =
                    (u64::from(self.chord_m(node, m)) * u64::from(tan_q16[i])) >> (TAN_BITS - 4);
                let e = drop - limit as i64;
                if e > 0 {
                    excess[slot] = e;
                    total += e;
                    largest = largest.max(e);
                }
            }
            if total == 0 {
                continue;
            }
            shed += 1;
            worst = worst.max(largest);
            // ★ HALF THE LARGEST excess leaves the node — never half the SUM over eight neighbours
            // (MEASURED on the home planet: the sum's half sent a peak twenty kilometres under its
            // neighbours and the pair grew a hundred-thousandfold over eight passes). Bounded by
            // the largest pair, the node never falls under the level its steepest neighbour sets.
            let moving = largest / i64::from(TALUS_FRACTION_DEN);
            for (slot, &m) in ring.iter().enumerate() {
                if excess[slot] > 0 {
                    let part = moving * excess[slot] / total;
                    delta[m as usize] += part;
                    delta[i] -= part;
                }
            }
        }
        for (z, d) in self.z.iter_mut().zip(&delta) {
            *z += *d as i32;
        }
        (shed, worst)
    }

    /// ★ THE ICE, AS A MASS BALANCE (ruling B2 step 2; Egholm et al. 2009): the walks of the receiver tree the water
    /// already uses, and NO EXTENT IS DRAWN anywhere in them.
    ///
    /// 1. **The balance.** At each node the ablation-season temperature stands `Γ·(ela − z)` from the one the
    ///    equilibrium line names, so the node gains `0.1·|Ts|` of ice a year above the line and loses `0.15·Ts` below
    ///    it, less `0.05·T` at a bed the year's own mean keeps over freezing. That balance times the node's area is its
    ///    own contribution, and the tree carries it downstream: where the sum runs out the glacier ENDS. A node under
    ///    the sea or under the routing's standing water contributes nothing — water only where the routing found a
    ///    lake (`pit > 0`), never where a bed the ice has just cut stands under its stale flood level (ruling B44
    ///    item 1: MEASURED, the older rule chopped every glacier at its first cut node).
    /// 2. **The distance to the margin**, up the same tree: how far the ice under a node must still travel before it
    ///    melts.
    /// 3. **The thickness and the speed.** The thickness is the LESSER of the two perfect-plastic readings of one yield
    ///    stress — `τ/(ρ·g·S)` over the slope to the receiver (Nye 1952; Paterson 1994) and `√(2·τ·d/ρ·g)` over the
    ///    distance to the margin (the plastic ice-sheet profile). The ice's own speed is its flux over its
    ///    cross-section, `u = q/H`, greatest at the line, so the overdeepening falls at the equilibrium line and below
    ///    every confluence without anybody placing it (MacGregor et al. 2000). The bed erodes at `kₑ·u` only where it
    ///    stands at its pressure-melting point: `T_bed = T_year + G·H/k_ice` against `T_melt`.
    ///
    /// Returns the mask, and for every warm bed its erosion rate in metres a year and its thickness in metres (zeros
    /// elsewhere), and the nodes under ice and the greatest thickness in metres.
    #[allow(
        clippy::type_complexity,
        reason = "the flow's five readings, read by the mask and the stepped cut"
    )]
    fn ice_flow(
        &self,
        ela_z: &[i32],
        temperature_dk: &[i16],
        lapse_mk_km: u32,
        gravity_mm_s2: u32,
    ) -> (Vec<bool>, Vec<Gf>, Vec<Gf>, usize, Gf) {
        let n = self.node_count();
        let mut mask = vec![false; n];
        let mut rate = vec![Gf::ZERO; n];
        let mut thick = vec![Gf::ZERO; n];
        let (mut under, mut thickest) = (0usize, Gf::ZERO);
        let g = Gf::from_i64(i64::from(gravity_mm_s2)) / Gf::from_i64(1_000);
        let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
        let lapse = Gf::from_i64(i64::from(lapse_mk_km)) / Gf::from_i64(1_000_000);
        let rho_g = Gf::from_f64(ICE_DENSITY_KGM3) * g;
        let tau = Gf::from_f64(ICE_YIELD_STRESS_PA);
        let dry = |i: usize| !self.ocean[i] & (self.pit[i] == 0);
        // 1. THE BALANCE, and the flux down the tree the water uses.
        let mut flux: Vec<Gf> = (0..n)
            .map(|i| {
                let b = if dry(i) {
                    ice_balance_m_yr(self.z[i], ela_z[i], lapse, celsius(temperature_dk[i]))
                } else {
                    Gf::ZERO
                };
                b * Gf::from_i64(self.area[i] as i64)
            })
            .collect();
        for k in 0..self.order.len() {
            let node = self.order[k] as usize;
            if !dry(node) || flux[node] <= Gf::ZERO {
                flux[node] = Gf::ZERO;
                continue;
            }
            mask[node] = true;
            under += 1;
            let r = self.receiver[node];
            if r != NO_NODE {
                flux[r as usize] = flux[r as usize] + flux[node];
            }
        }
        // 2. THE DISTANCE TO THE MARGIN, outlets first (the order, reversed).
        let mut margin_m = vec![0u32; n];
        for k in (0..self.order.len()).rev() {
            let node = self.order[k] as usize;
            let r = self.receiver[node];
            if mask[node] {
                let carried = if r != NO_NODE && mask[r as usize] {
                    margin_m[r as usize]
                } else {
                    0
                };
                // The last ice node still holds ice across its OWN width, so the margin stands one chord out from it
                // and the plastic profile there is a thickness, never a zero.
                margin_m[node] = carried.saturating_add(self.chord[node].max(1));
            }
        }
        // 3. THE THICKNESS AND THE SPEED of every warm bed.
        for node in 0..n as u32 {
            let i = node as usize;
            let r = self.receiver[i];
            if !mask[i] || r == NO_NODE {
                continue;
            }
            let drop = Gf::from_i64(i64::from(self.z[i]) - i64::from(self.z[r as usize])) / steps;
            if drop <= Gf::ZERO {
                continue;
            }
            let slope = drop / Gf::from_i64(i64::from(self.chord[i].max(1)));
            let plastic = tau / (rho_g * slope);
            let sheet = (Gf::TWO * tau * Gf::from_i64(i64::from(margin_m[i])) / rho_g).sqrt();
            let thickness = plastic.lesser(sheet);
            thickest = thickest.greater(thickness);
            let bed_c = celsius(temperature_dk[i])
                + Gf::from_f64(GEOTHERMAL_FLUX_W_M2) * thickness
                    / Gf::from_f64(ICE_CONDUCTIVITY_W_M_K);
            let melt_c = -Gf::from_f64(ICE_PRESSURE_MELT_K_PER_M) * thickness;
            if bed_c < melt_c {
                continue;
            }
            let width = Gf::from_i64(self.area[i].max(1) as i64).sqrt();
            rate[i] = Gf::from_f64(GLACIAL_EROSION_K) * flux[i] / (width * thickness);
            thick[i] = thickness;
        }
        (mask, rate, thick, under, thickest)
    }

    /// ★ THE ICE'S MASK over the field as it stands ([`MacroSolve::ice_flow`]): where ice lies, the nodes under it and
    /// the greatest thickness in metres — the reading the facies byte carries; it cuts nothing.
    ///
    /// **Example.** On the home planet a range stands over its summer line: the snow that falls on its head runs down
    /// one trunk, the trunk keeps ice a hundred kilometres BELOW the line because the catchment feeds it, and the
    /// ridge beside it holds a cold thin cap. The moon, whose line stands above every node, holds none.
    #[must_use]
    pub fn ice_mask(
        &self,
        ela_z: &[i32],
        temperature_dk: &[i16],
        lapse_mk_km: u32,
        gravity_mm_s2: u32,
    ) -> (Vec<bool>, usize, Gf) {
        let (mask, _, _, under, thickest) =
            self.ice_flow(ela_z, temperature_dk, lapse_mk_km, gravity_mm_s2);
        (mask, under, thickest)
    }

    /// ★ THE ICE STEPPED IN TIME WITH NO CAP (ruling B44 item 1): the ice's cut over `years` in steps, each the longest
    /// that lowers no warm bed by more than its own ice — a resolution, never a limit on the cut — the flow read again
    /// over the field each step cut ([`MacroSolve::ice_flow`]); each node's fraction of a sixteenth carried to its
    /// next step, and every cut booked in `pending` for the rebound (ruling B42 item 2: the ice's cut lightens the
    /// crust as the rivers' does). A step is never under one year — the bench's guard (ruling B44's copy) that keeps the
    /// count of steps finite — so where a bed would cut its own ice in less, that step lowers it by more: the steps the
    /// year set are COUNTED, a reading that says whether the guard ever acts. Returns the cut summed over the steps
    /// (sixteenths × nodes), the steps taken, and the steps the year set.
    ///
    /// **Example.** In the glacial epoch's last half-million years a valley glacier on the home planet's wedge moves
    /// 40 m a year over a warm bed: the first step is the time it takes to cut its own 300 m of ice, the next reads the
    /// deepened trough again, and the trough ends some hundreds of metres deeper, with the range rising a little under
    /// it.
    pub fn ice_stepped(
        &mut self,
        ela_z: &[i32],
        temperature_dk: &[i16],
        lapse_mk_km: u32,
        gravity_mm_s2: u32,
        years: u64,
    ) -> (i64, u32, u32) {
        let n = self.node_count();
        let steps_m = Gf::from_i64(i64::from(Z_STEPS_PER_M));
        let mut remaining = Gf::from_i64(years as i64);
        let (mut total, mut taken, mut floored) = (0i64, 0u32, 0u32);
        let mut carry = vec![Gf::ZERO; n];
        while remaining > Gf::ZERO {
            let (_, rate, thick, _, _) =
                self.ice_flow(ela_z, temperature_dk, lapse_mk_km, gravity_mm_s2);
            // The step: the longest that lowers no warm bed by more than its own ice, at least a year.
            let mut step = remaining;
            for (&r, &h) in rate.iter().zip(&thick) {
                if r > Gf::ZERO {
                    step = step.lesser(h / r);
                }
            }
            floored += u32::from((step < Gf::ONE) & (remaining > Gf::ONE));
            let step = step.greater(Gf::ONE).lesser(remaining);
            // The cut over the step, the fraction of a sixteenth carried to the next step.
            let mut any = false;
            for i in 0..n {
                if rate[i] <= Gf::ZERO {
                    continue;
                }
                let exact = rate[i] * step * steps_m + carry[i];
                let whole = exact.floor();
                carry[i] = exact - whole;
                let cut = whole.to_i64_floor() as i32;
                if cut > 0 {
                    any = true;
                    self.z[i] -= cut;
                    total += i64::from(cut);
                    self.pending[i] = self.pending[i].saturating_add(cut);
                }
            }
            remaining -= step;
            taken += 1;
            if !any & rate.iter().all(|&r| r <= Gf::ZERO) {
                break;
            }
        }
        (total, taken, floored)
    }

    /// ★ THE BODY'S DATUM, sixteenths: the sea's level on a body with a sea, the ladder's datum on a dry body — the
    /// level the charter's surface words hold at (ruling B25 item 1), and the base the peak law reads a height over.
    #[must_use]
    pub fn datum_z(&self) -> i32 {
        if self.sea_z == i32::MIN {
            0
        } else {
            self.sea_z
        }
    }

    /// ★ THE PEAK LAW per node (limits as law, the belts' design L0; ruling B37): a node whose height over the body's
    /// datum ([`MacroSolve::datum_z`]) passes `cap` (the body's relief cap by its own law,
    /// [`crate::land::relief_cap_m`]) collapses where it stands; its excess volume goes to its lowest neighbour, laid
    /// over that neighbour's area to the whole sixteenth below (the remainder, under one sixteenth of the neighbour's
    /// area, stays unlaid), and the rounds repeat until no node stands over the cap or [`PEAK_ROUNDS`] end — so the
    /// debris runs out downhill until it finds room. Returns the collapses summed over the rounds and the largest
    /// excess still over the cap after the last round, in sixteenths.
    ///
    /// **Example.** A wedge's crest on a small dry world rises past the height its rock can hold; the node sheds its
    /// excess into the valley beside it, which stands under the cap and keeps it.
    pub fn peak_law(&mut self, cap: i32) -> (usize, i32) {
        let n = self.node_count();
        let ceiling = i64::from(self.datum_z()) + i64::from(cap);
        let mut collapsed = 0usize;
        for _ in 0..PEAK_ROUNDS {
            let mut moved = 0usize;
            for i in 0..n {
                let excess = i64::from(self.z[i]) - ceiling;
                if excess <= 0 {
                    continue;
                }
                let mut lowest = NO_NODE;
                for m in self.lattice.neighbours(i as u32) {
                    let lower = (m != NO_NODE)
                        && ((lowest == NO_NODE) || (self.z[m as usize] < self.z[lowest as usize]));
                    if lower {
                        lowest = m;
                    }
                }
                self.z[i] -= excess as i32;
                let volume = i128::from(excess) * i128::from(self.area[i]);
                self.z[lowest as usize] +=
                    volume.div_euclid(i128::from(self.area[lowest as usize])) as i32;
                moved += 1;
            }
            collapsed += moved;
            if moved == 0 {
                break;
            }
        }
        let left = self
            .z
            .iter()
            .map(|&z| (i64::from(z) - ceiling).max(0))
            .max()
            .unwrap_or(0);
        (collapsed, left as i32)
    }

    /// ★ THE FACIES BYTE of every node: the sea (the CONNECTED ocean, 2026-09-24, ruling B7 item 1),
    /// a lake, the coast (within `band` sixteenths of the sea's level), the ice — the bits 8d and 8e
    /// read. A body with no water has no sea and no lake: the flood's raised pits on a dry moon are
    /// closed basins, not lakes. An inland pocket is never SEA: it is a lake where its budget put
    /// water, and dry ground where it did not.
    #[must_use]
    pub fn facies(&self, ice: &[bool], band: i32, has_water: bool) -> Vec<u8> {
        (0..self.node_count() as u32)
            .map(|node| {
                let i = node as usize;
                let mut f = 0u8;
                if has_water & self.ocean[i] {
                    f |= FACIES_SEA;
                }
                // ★ A LAKE IS WATER THAT STANDS (ruling W11), never the routing flood's own fill.
                if has_water & (self.water_z[i] > self.z[i]) {
                    f |= FACIES_LAKE;
                }
                if (i64::from(self.z[i]) - i64::from(self.sea_z)).abs() <= i64::from(band) {
                    f |= FACIES_COAST;
                }
                if ice[i] {
                    f |= FACIES_ICE;
                }
                f
            })
            .collect()
    }

    /// ★ THE OCEAN SHARE (stage C5, gate G-SEA): the CONNECTED ocean's area over the globe's area
    /// (2026-09-24, ruling B7 item 1: a pocket under the level is not the sea), a plain number for
    /// the report; zero with no sea.
    #[must_use]
    pub fn ocean_share(&self) -> f64 {
        if self.sea_z == i32::MIN {
            return 0.0;
        }
        let mut wet = 0u128;
        let mut all = 0u128;
        for (&a, &ocean) in self.area.iter().zip(&self.ocean) {
            all += u128::from(a);
            wet += u128::from(a) * u128::from(ocean);
        }
        if all == 0 {
            return 0.0;
        }
        wet as f64 / all as f64
    }

    /// The lowest and the highest terrain, in sixteenths from the ladder radius.
    #[must_use]
    pub fn range(&self) -> (i32, i32) {
        self.z
            .iter()
            .fold((i32::MAX, i32::MIN), |(lo, hi), &z| (lo.min(z), hi.max(z)))
    }
}

/// A climate row's temperature in degrees celsius.
#[must_use]
pub fn celsius(temperature_dk: i16) -> Gf {
    Gf::from_i64(i64::from(temperature_dk)) / Gf::from_i64(10)
        - Gf::from_f64(crate::climate::FREEZING_K)
}

/// ★ THE ICE'S MASS BALANCE at one node, metres a year (Egholm et al. 2009): the ablation-season
/// temperature stands `Γ·(ela − z)` from the equilibrium line's own, so a node over the line gains
/// [`ICE_ACCUMULATION_M_YR_K`] a kelvin and a node under it loses [`ICE_ABLATION_M_YR_K`] a
/// kelvin, less [`ICE_BASAL_MELT_M_YR_K`] a kelvin at a bed the year's mean keeps over freezing.
///
/// **Example.** A node 600 m over its line on a 6.3 K/km lapse is 3.8 K colder than the line, so
/// it gains 0.38 m of ice a year; the valley 1 200 m under the line loses 1.13 m a year, and the
/// glacier reaches it only while its catchment sends more than that.
#[must_use]
pub fn ice_balance_m_yr(z: i32, ela_z: i32, lapse_k_m: Gf, t_year_c: Gf) -> Gf {
    let over_m =
        Gf::from_i64(i64::from(z) - i64::from(ela_z)) / Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let ts = -lapse_k_m * over_m;
    let accumulation = Gf::from_f64(ICE_ACCUMULATION_M_YR_K) * -ts.lesser(Gf::ZERO);
    let ablation = Gf::from_f64(ICE_ABLATION_M_YR_K) * ts.greater(Gf::ZERO);
    let basal = Gf::from_f64(ICE_BASAL_MELT_M_YR_K) * t_year_c.greater(Gf::ZERO);
    accumulation - ablation - basal
}

/// ★ THE TANGENT OF REPOSE at an aridity (0 wet .. 255 hyper-arid), in 1/65536: the wet end faded
/// to the dry end by the aridity's share.
#[must_use]
pub fn tan_repose_q16(aridity_q8: u8) -> u32 {
    let share = Gf::from_i64(i64::from(aridity_q8)) / Gf::from_i64(255);
    let tan = Gf::from_f64(TAN_REPOSE_WET)
        + (Gf::from_f64(TAN_REPOSE_DRY) - Gf::from_f64(TAN_REPOSE_WET)) * share;
    (tan * Gf::from_i64(1 << TAN_BITS)).to_i64_floor() as u32
}

/// ★ THE COAST's BAND in sixteenths: the storm wave's height under the body's air and gravity — the
/// air's density from the surface pressure over the scale height and the gravity, `ρ = p / (g·H)`.
/// An airless body has no waves and a zero band.
#[must_use]
pub fn coast_band(words: &SolveWords, gravity_mm_s2: u32) -> i32 {
    let (Some(p), Some(h)) = (words.p_surf_pa, words.scale_height_m) else {
        return 0;
    };
    let g = Gf::from_i64(i64::from(gravity_mm_s2)) / Gf::from_i64(1_000);
    let air = Gf::from_i64(i64::from(p)) / (g * Gf::from_i64(i64::from(h.max(1))));
    let earth_g = Gf::from_f64(9.81);
    let wave = Gf::from_f64(EARTH_STORM_WAVE_M)
        * (air / Gf::from_f64(EARTH_AIR_DENSITY_KGM3))
        * (earth_g / g);
    (wave * Gf::from_i64(i64::from(Z_STEPS_PER_M))).to_i64_floor() as i32
}

/// A direction word at the bend's bits, kept at [`CACHE_BITS`].
fn cache(g: Gi) -> i32 {
    (g.raw() >> (DIR_BITS - CACHE_BITS)) as i32
}

/// A cached direction back at the bend's bits (the dropped bits are zero: a stated rounding).
fn uncache(d: [i32; 3]) -> [Gi; 3] {
    [
        Gi::new(i64::from(d[0]) << (DIR_BITS - CACHE_BITS)),
        Gi::new(i64::from(d[1]) << (DIR_BITS - CACHE_BITS)),
        Gi::new(i64::from(d[2]) << (DIR_BITS - CACHE_BITS)),
    ]
}

/// ★ THE DRIVER: the mixed axis over `body` from the recipe's own relief (the C1 start) — a routing and an
/// accumulation every pass, then the joint law's sweeps on that routing. No climate (the rain stays [`P_MIN_MM_YR`]),
/// no uplift, no rebound: the core's drainage, for the instruments that read only it. `None` where the body has no
/// lattice.
#[must_use]
pub fn solve(body: &BodyDefinition, schedule: Schedule) -> Option<(MacroSolve, SolveReport)> {
    let mut state = MacroSolve::new(body)?;
    let mut report = SolveReport::default();
    let times = schedule.times();
    for pair in times.windows(2) {
        report.routes.push(state.route());
        state.accumulate();
        let dt = Gf::from_i64((pair[1] - pair[0]) as i64)
            / Gf::from_i64(i64::from(schedule.sub_sweeps.max(1)));
        for s in 0..schedule.sub_sweeps.max(1) {
            // The pass's later routings (ruling B65 item 1), as the full solve's.
            if schedule.routes_before(s) {
                report.routes.push(state.route());
                state.accumulate();
            }
            report.sweeps.push(state.sweep(dt, None, None));
        }
    }
    Some((state, report))
}

/// One routing of a pass of the full solve: a body with water labels its ocean for the budget, whose own tree routes
/// the rest ([`MacroSolve::label_for_budget`]; ruling B73 item 1); a dry body floods and routes its ground, which its
/// sweeps and deposit read, and accumulates its discharge.
fn route_for_pass(state: &mut MacroSolve, has_water: bool) -> RouteReport {
    if has_water {
        return state.label_for_budget();
    }
    let report = state.route();
    state.accumulate();
    report
}

/// ★ THE FULL SOLVE (stage C3; the design's §4.11; ★ PHASE 2 OF THE GROUND STEP, rulings B39–B51): the land; the belts
/// by their influx as critical wedges ([`crate::belts`]); the landscape's time; then the MIXED AXIS — each pass stamps
/// the craters whose impact falls in it, re-solves the running sea and the climate over the relief as it stands (every
/// pass, ruling B69 item 1), routes and follows the budget's own water ([`MacroSolve::follow_the_budget`]), sheds the talus, steps the
/// ice through the glacial epoch's share it holds, raises the belts and the even uplift by the pass's share of time,
/// and runs `sub_sweeps` sweeps of the joint law with a rebound after each, then the peak law — and at the end the
/// talus's remainder, the sea re-solved, the final routing, the ice's mask, the lakes and the sea's fixed point, and
/// the readings. NO LIMIT CLAMPS A HEIGHT (the belts' design L0): no clip of the craters, no re-clamp of the uplift,
/// no envelope — the peak law is the one limit, and it is a law. The facies byte comes out beside the state. ONE
/// thread, off the tick; `None` where the body has no lattice.
///
/// **Example.** The home planet's shard solves its planet once: fourteen plates; three wedges rising at their
/// margins inside their own windows; 214 passes from five billion years ago to the present, the last half a million
/// years long; rivers fed by the rain the ranges lift out of the wind, ending in closed salt basins where the water
/// runs out; the crust floating up under every cut; the ice cutting its troughs in the last 2.6 million years. The
/// artifact is what it keeps; the ground never changes again.
#[must_use]
pub fn solve_full(
    body: &BodyDefinition,
    words: &SolveWords,
    schedule: Schedule,
) -> Option<(MacroSolve, Vec<u8>, FullReport)> {
    solve_full_on(body, words, schedule, crate::par::Cores::default())
}

/// [`solve_full`] ON THE SOLVE'S CORES (ruling B64): the first land, every pass's climate, routing, descent, sweeps and
/// rebounds on `cores` threads — the same state, facies and report on any number of them, so the artifact's digest does
/// not move. The passes stay one after another.
///
/// **Example.** The home planet's shard solves its planet on its share of the host's cores; the artifact it stores is
/// the one a single core would have stored, byte for byte.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn solve_full_on(
    body: &BodyDefinition,
    words: &SolveWords,
    schedule: Schedule,
    cores: crate::par::Cores,
) -> Option<(MacroSolve, Vec<u8>, FullReport)> {
    let lattice = body.macro_lattice()?;
    let mut land = crate::land::initial_land_on(body, &lattice, &words.land(), cores);
    let n = lattice.node_count();
    let has_water = words.water_km3 > 0;
    let mut report = FullReport::default();
    // ★ THE BELTS BY THEIR INFLUX (phase 2): the critical wedges replace the room's belts where they rise; a wedge node
    // off its plateau reads the folded belt's rock, a plateau keeps its own.
    let belts = crate::belts::belts(body, &lattice, words, &land, schedule.age_yr);
    let even_total = belt_land(&belts, &mut land);
    // ★ THE LANDSCAPE'S TIME: each node erodes only since its ground was last raised — its plate's raise, or its belt's
    // window where that opened later.
    let raise_yr = raise_times(&land.plates, schedule.age_yr, belts.report.t_m_yr > 0);
    let erosion_start = erosion_starts(&belts, &raise_yr, &land.plate, schedule.age_yr);
    let mut state = MacroSolve::from_land(body, &land)?;
    state.cores = cores;
    let gravity = body.facts().gravity_mm_s2;
    // The craters: the impact record over the surface's own retention (ruling B2 step 2), each stamped in the pass its
    // impact falls in (ruling B37, the craters' own time), never one older than its ground's last raise.
    let pre_climate =
        crate::climate::climate_on(body, &lattice, words, &land.z, land.sea_view(), cores);
    let surface = crate::craters::Surface {
        crust: &land.crust,
        boundary_m: &land.boundary_m,
        plate: &land.plate,
        plates: &land.plates,
        province: &land.province,
        rain_mm_yr: &pre_climate.rain_mm_yr,
    };
    let craters = crate::craters::crater_population(body, &lattice, words, &surface);
    drop(pre_climate);
    state.z_flood.clone_from(&state.z);
    let peak_cap = (Gf::from_f64(crate::land::relief_cap_m(body))
        * Gf::from_i64(i64::from(Z_STEPS_PER_M)))
    .to_i64_floor() as i32;
    let alpha_m = crate::land::flexural_parameter_m(words.elastic_thickness_m, gravity);
    let times = schedule.times();
    let passes = times.len() - 1;
    let sub_sweeps = schedule.sub_sweeps.max(1);
    let epoch_start = schedule.age_yr.saturating_sub(GLACIAL_EPOCH_YR);
    // ★ A DRY BODY'S LINE stands over every node (03 §4.9), whatever its cold — allocated once.
    let no_ice: Vec<i32> = if has_water {
        Vec::new()
    } else {
        vec![i32::MAX; n]
    };
    let mut climate: Option<crate::climate::Climate> = None;
    let mut talus_done = 0u32;
    for pass in 0..passes {
        let (t0, t1) = (times[pass], times[pass + 1]);
        let last = pass + 1 == passes;
        let due: Vec<crate::craters::Crater> = craters
            .iter()
            .copied()
            .filter(|c| {
                let at = schedule.age_yr.saturating_sub(c.age_yr);
                (at >= erosion_start[c.node as usize])
                    & (at >= t0)
                    & ((at < t1) | (last & (at <= t1)))
            })
            .collect();
        crate::craters::apply_craters(&mut state.z, &lattice, &due, gravity);
        report.crater_record.extend_from_slice(&due);
        // ★ THE RUNNING SEA (W7 step 2) over the CONNECTED ocean only (ruling B7 item 1), and the climate over the relief
        // as it stands — both every pass (ruling B69 item 1).
        if let Some(sea) = crate::land::sea_level_connected_on(
            &lattice,
            &state.z,
            &state.area,
            words.water_km3,
            false,
            cores,
        ) {
            state.sea_z = sea.level;
            state.ocean = sea.mask.ocean;
            report.sea_reseeded |= sea.reseeded;
            report.sea_unsettled |= !sea.settled;
        }
        let c: &crate::climate::Climate = climate.insert(crate::climate::climate_on(
            body,
            &lattice,
            words,
            &state.z,
            has_water.then_some(state.sea_view()),
            cores,
        ));
        state.rain.clone_from(&c.rain_mm_yr);
        report.routes.push(route_for_pass(&mut state, has_water));
        let mut basins = state.follow_the_budget(c, has_water);
        // ★ THE SCREE AND THE ICE, INSIDE THE PASS (ruling B2 step 2): the talus keeps its own total, spread over the
        // passes; the ice cuts the glacial epoch's share this pass holds, stepped (ruling B44 item 1).
        let talus_due = (pass as u64 + 1) * u64::from(schedule.talus_passes) / passes as u64
            > pass as u64 * u64::from(schedule.talus_passes) / passes as u64;
        if talus_due {
            let tan: Vec<u32> = c.aridity_q8.iter().map(|&a| tan_repose_q16(a)).collect();
            report.talus.push(state.talus(&tan));
            talus_done += 1;
        }
        let epoch = crate::belts::overlap(epoch_start, schedule.age_yr, t0, t1);
        if epoch > 0 {
            let line: &[i32] = if has_water { &c.ela_z } else { &no_ice };
            let (cut, steps, floored) =
                state.ice_stepped(line, &c.temperature_dk, c.lapse_mk_km, gravity, epoch);
            report.ice.2 += cut;
            report.ice_steps += steps;
            report.ice_floored += floored;
        }
        pass_uplift(&mut state.z, &even_total, &belts, schedule.age_yr, t0, t1);
        // ★ THE SWEEPS on the pass's routings, each a share of the pass's time, a rebound after each (ruling B45: the
        // rebound without its floors, at 12.5 Myr or finer); ★ THE WATER ROUTED AGAIN before each later share of them
        // (ruling B65 item 1, [`Schedule::routes_before`]).
        let active: Vec<u32> = crate::par::map(n, cores.threads(), |i| {
            active_share(erosion_start[i], t0, t1)
        });
        let dt = Gf::from_i64((t1 - t0) as i64) / Gf::from_i64(i64::from(sub_sweeps));
        let mut cut_sum = vec![0u32; n];
        let mut groups = state.basin_groups();
        for s in 0..sub_sweeps {
            if schedule.routes_before(s) {
                report.routes.push(route_for_pass(&mut state, has_water));
                basins = state.follow_the_budget(c, has_water);
                groups = state.basin_groups();
            }
            report
                .sweeps
                .push(state.sweep_grouped(dt, Some(&active), basins.as_ref(), &groups));
            cut_sum = crate::par::map(n, cores.threads(), |i| {
                cut_sum[i].saturating_add(state.cut_now[i])
            });
            report.rebounds.push(state.rebound(alpha_m));
        }
        state.cut_now = cut_sum;
        // ★ THE PEAK LAW after every pass, on the height that stands.
        let (collapsed, left) = state.peak_law(peak_cap);
        report.peak.0 += collapsed;
        report.peak.1 = report.peak.1.max(left);
    }
    report.craters = report.crater_record.len();
    // The climate over the final relief, for the talus's angle and the ice's line.
    let climate = match climate {
        Some(c) => c,
        None => crate::climate::climate_on(
            body,
            &lattice,
            words,
            &state.z,
            has_water.then_some(state.sea_view()),
            cores,
        ),
    };
    // The talus relaxations the passes had no room to hold: the count the schedule states is the count that runs.
    let tan: Vec<u32> = climate
        .aridity_q8
        .iter()
        .map(|&a| tan_repose_q16(a))
        .collect();
    while talus_done < schedule.talus_passes {
        report.talus.push(state.talus(&tan));
        talus_done += 1;
    }
    drop(tan);
    // ★ THE SEA, RE-SOLVED over the eroded field (stage C5) over the CONNECTED ocean (ruling B7 item 1).
    if let Some(sea) = crate::land::sea_level_connected_on(
        &lattice,
        &state.z,
        &state.area,
        words.water_km3,
        false,
        cores,
    ) {
        state.sea_z = sea.level;
        report.sea_reseeded |= sea.reseeded;
        report.sea_unsettled |= !sea.settled;
    }
    // The final routing, so the receivers, the basins and the facies read the final land.
    report.routes.push(state.route());
    // ★ THE ICE'S OWN LAST READING over the final field: the mask the facies byte carries.
    let ice_mask = {
        let line: &[i32] = if has_water { &climate.ela_z } else { &no_ice };
        let (mask, under, thickest) =
            state.ice_mask(line, &climate.temperature_dk, climate.lapse_mk_km, gravity);
        report.ice.0 = under;
        report.ice.1 = thickest.to_f64();
        mask
    };
    // ★ THE LAKES (ruling W11): the depression hierarchy over the final field and every hollow's own water budget, and
    // the sea re-solved over the inventory less the lakes — a fixed point (ruling B7 item 1; plan question 8).
    let mut final_climate = climate;
    let mut budget = None;
    if has_water {
        // The climate and the lakes over the field as it stands: read once here, and again by the same rule each time
        // the sea moves; the last reading is the one the solve keeps.
        let mut read: Option<(crate::climate::Climate, crate::lakes::Lakes)> = None;
        let mut again = |s: &MacroSolve| -> u128 {
            let climate =
                crate::climate::climate_on(body, &lattice, words, &s.z, Some(s.sea_view()), cores);
            let lakes = crate::lakes::budget(s, &climate);
            let volume = lakes.report.volume_m3;
            read = Some((climate, lakes));
            volume
        };
        let volume = again(&state);
        let fix = settle_lakes_sea(
            &mut state,
            words.water_km3,
            LAKE_SEA_SOLVES,
            volume,
            &mut report.routes,
            &mut again,
        );
        let (climate, lakes) = read.expect("the lakes were read over the field");
        report.lake_sea_solves = fix.solves;
        report.lake_sea_converged = fix.converged;
        report.sea_reseeded |= fix.reseeded;
        report.sea_unsettled |= fix.unsettled;
        report.sea_before_lakes = (state.sea_z != fix.first).then_some(fix.first);
        report.lakes = lakes.report.clone();
        state.water_z.clone_from(&lakes.water);
        final_climate = climate;
        budget = Some(lakes);
    }
    report.sea_z = has_water.then_some(state.sea_z);
    report.ocean_share = state.ocean_share();
    report.coast_band = coast_band(words, gravity);
    // ★ GATE G-AGE AND THE SEDIMENT BUDGET READ THE EROSION'S LAST ROUTING (plan step 2b, question 7), read BEFORE the
    // tree is swapped for the water's own below.
    report.integrals = state.hypsometric_integrals(AGE_GATE_MIN_NODES);
    report.deposits = state.sediment_by_basin().len();
    // ★ THE WATER'S OWN TREE AND ITS DISCHARGE (plan step 2b; ruling B8 item 2): the shipped receivers follow the
    // budget's water, and the shipped row is the budget's own runoff less what each spilling lake hands the air.
    let tree = match &budget {
        Some(lakes) => crate::lakes::water_tree(&state, lakes),
        None => crate::lakes::dry_tree(&state),
    };
    drop(budget);
    report.tree = state.follow_the_water(tree, &final_climate.runoff_mm_yr);
    drop(final_climate);
    report.belts = belts;
    report.raise_yr = raise_yr;
    let facies = state.facies(&ice_mask, report.coast_band, has_water);
    Some((state, facies, report))
}

/// ★ TWO HOLLOWS DUG INTO A SOLVED FIELD — compiled into the tests only, never into the game. Since phase 2 the wet moon
/// field the tests solve holds no hollow at all (MEASURED: over five billion years the rock that reaches each hollow
/// fills it, ruling B51 item 2), so the laws that read lakes are tested over the same field with two hollows dug into
/// it: a stepped bowl of nine nodes at the driest land node no river reaches whose own runoff still pays for a lake on
/// its middle (the budget's own rates: `PET > P` and `8·Q > PET − P`), so its lake stands where the air takes what falls
/// in; and a pit 30 m under its lowest neighbour in the land's largest river away from the sea, which fills and spills.
#[cfg(test)]
pub(crate) mod dug {
    use super::*;

    /// Digs the two hollows into `state` (whose receivers are the solve's own), routes it again, and returns the
    /// climate over the new field and the lakes' budget over it, with the state's water set to the budget's.
    pub(crate) fn two_hollows(
        body: &BodyDefinition,
        words: &SolveWords,
        state: &mut MacroSolve,
    ) -> (crate::climate::Climate, crate::lakes::Lakes) {
        let n = state.node_count();
        let lattice = body.macro_lattice().expect("a lattice");
        let climate =
            crate::climate::climate(body, &lattice, words, &state.z, Some(state.sea_view()));
        let inland = |s: &MacroSolve, i: usize| {
            !s.ocean[i]
                && s.lattice
                    .neighbours(i as u32)
                    .iter()
                    .all(|&m| (m != NO_NODE) && !s.ocean[m as usize])
        };
        let mut donors = vec![0u32; n];
        for i in 0..n {
            let r = state.receiver[i];
            if r != NO_NODE {
                donors[r as usize] += 1;
            }
        }
        let arid = |i: usize| -> i64 {
            i64::from(climate.pet_mm_yr[i]) - i64::from(climate.rain_mm_yr[i])
        };
        // The semi-arid summits, driest first: the first whose dug bowl the budget leaves CLOSED is the bowl (MEASURED
        // after the deposit's cure, 2026-10-05: the driest one's dug ring took in water from beyond the summit and its
        // lake spilled, so the bowl is chosen by the budget's own verdict, not assumed).
        let mut summits: Vec<usize> = (0..n)
            .filter(|&i| {
                inland(state, i)
                    & (donors[i] == 0)
                    & (arid(i) > 0)
                    & (8 * i64::from(climate.runoff_mm_yr[i]) > arid(i))
            })
            .collect();
        summits.sort_unstable_by_key(|&i| std::cmp::Reverse((arid(i), i)));
        assert!(!summits.is_empty(), "a semi-arid summit");
        let trunk = (0..n)
            .filter(|&i| inland(state, i) && !summits[..summits.len().min(32)].contains(&i))
            .max_by_key(|&i| (state.discharge[i], i))
            .expect("a trunk");
        let lowest = state
            .lattice
            .neighbours(trunk as u32)
            .iter()
            .map(|&m| state.z[m as usize])
            .min()
            .expect("a neighbour");
        state.z[trunk] = lowest - 30 * Z_STEPS_PER_M;
        summits
            .iter()
            .take(32)
            .find_map(|&summit| {
                // Each bowl is dug on a copy of the field, and the copy is kept only where the budget closes its lake.
                let mut trial = state.clone();
                let ring = trial.lattice.neighbours(summit as u32);
                let mut bowl: Vec<u32> = ring.to_vec();
                bowl.push(summit as u32);
                let low = bowl
                    .iter()
                    .flat_map(|&v| trial.lattice.neighbours(v))
                    .filter(|m| (*m != NO_NODE) & !bowl.contains(m))
                    .map(|m| trial.z[m as usize])
                    .min()
                    .expect("a rim");
                for &m in &ring {
                    trial.z[m as usize] = low - 30 * Z_STEPS_PER_M;
                }
                trial.z[summit] = low - 60 * Z_STEPS_PER_M;
                trial.route();
                let climate = crate::climate::climate(
                    body,
                    &lattice,
                    words,
                    &trial.z,
                    Some(trial.sea_view()),
                );
                let lakes = crate::lakes::budget(&trial, &climate);
                (lakes.report.partial > 0).then(|| {
                    trial.water_z.clone_from(&lakes.water);
                    *state = trial;
                    (climate, lakes)
                })
            })
            .expect("a closed bowl among the 32 driest semi-arid summits")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_planet};
    use crate::macro_lattice::STENCIL;
    use vd_seed::bend::Face;

    /// ★ THE MIXED AXIS (rulings B39's gate, B56): the home system's age in 414 passes — the last a quarter of a million
    /// years, each pass before it a quarter longer (rounded half up) until 12.5 million years, the first taking the
    /// remainder — so the boundaries start at 0 and end at the age exactly. An age under the last pass is one pass, no
    /// age no pass, and a last pass longer than the longest makes every pass the longest.
    #[test]
    fn the_mixed_axis_steps_the_age_exactly() {
        let times = Schedule::standard(HOME_SYSTEM_AGE_YR).times();
        assert_eq!(
            times,
            mixed_times(HOME_SYSTEM_AGE_YR, MIXED_T_MIN_YR, MIXED_T_MAX_YR)
        );
        assert_eq!(times.len(), 415);
        assert_eq!(times[0], 0);
        assert_eq!(times[414], HOME_SYSTEM_AGE_YR);
        let lengths: Vec<u64> = times.windows(2).map(|w| w[1] - w[0]).collect();
        assert_eq!(&lengths[410..], &[488_281, 390_625, 312_500, 250_000]);
        assert!(lengths.iter().all(|&l| (1..=MIXED_T_MAX_YR).contains(&l)));
        assert_eq!(lengths[1], MIXED_T_MAX_YR);
        // The axis before B56, for the rule itself.
        assert_eq!(mixed_times(0, 500_000, 25_000_000), vec![0]);
        assert_eq!(mixed_times(300_000, 500_000, 25_000_000), vec![0, 300_000]);
        assert_eq!(
            mixed_times(1_000_000, 500_000, 25_000_000),
            vec![0, 500_000, 1_000_000]
        );
        assert_eq!(mixed_times(10, 7, 3), vec![0, 1, 4, 7, 10]);
    }

    /// The steeper test is a total order over integers: a steeper drop wins, an equal slope with a
    /// longer chord loses, and an exact tie takes the smaller index.
    #[test]
    fn steeper_is_a_total_order_with_the_smaller_index_on_a_tie() {
        assert!(steeper(20, 100, 7, 10, 100, 3));
        assert!(!steeper(10, 200, 7, 10, 100, 3));
        assert!(steeper(20, 200, 2, 10, 100, 3));
        assert!(!steeper(20, 200, 4, 10, 100, 3));
    }

    /// The node's rung on the home planet is 13 (a node is 2¹³ cells) and on the moon 13 too, and
    /// the starting surface holds the recipe's relief at that rung within the body's own band.
    #[test]
    fn the_starting_surface_is_the_recipes_relief_at_the_nodes_rung() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        assert_eq!(node_rung(&moon, &lattice), 13.min(moon.ladder().rungs - 1));
        let z = initial_surface(&moon, &lattice);
        assert_eq!(z.len(), lattice.node_count());
        let band = (moon.relief_bound_m(0) * f64::from(Z_STEPS_PER_M)) as i32;
        assert!(z.iter().all(|&s| s.abs() <= band), "outside the band");
        assert!(z.iter().any(|&s| s != 0), "a flat moon");
        let home = home_planet();
        assert_eq!(
            node_rung(&home, &MacroLattice::of(&home).expect("a lattice")),
            13
        );
    }

    /// ★ THE DRIVER ON A REAL SMALL BODY OF THE WORLD (06 §3.3): the home moon's whole axis — 414
    /// routings, two sweeps on each; every routing drains every node to an outlet (gate
    /// G-DRAINAGE) with no cycle; the sweeps only lower; the relief stays inside the band.
    /// ★ THE SAME BYTES ON ANY NUMBER OF CORES (ruling B64): a short wet solve of the home moon — the home planet's air
    /// and a sea over part of it, the first sixty million years of the axis — on one core and on three gives the same
    /// state, facies and report: the first land, every climate, routing, descent, sweep by river basin and rebound on
    /// the cores, every sum in one order.
    /// ★ A WET PASS READS ONLY THE ROUTING'S OCEAN LABEL (ruling B73 item 1): from one wet state whose ground moved
    /// since its last routing, the full routing and its discharge followed by the budget's own tree, and the ocean's
    /// label followed by the same tree, leave the same state, word for word, and the same closed hollows.
    #[test]
    fn a_wet_pass_reads_only_the_routings_ocean_label() {
        let moon = home_moon();
        let words = SolveWords {
            water_km3: 3_000_000,
            ..crate::home::home_solve_words()
        };
        let schedule = Schedule {
            age_yr: 60_000_000,
            ..Schedule::standard(HOME_SYSTEM_AGE_YR)
        };
        let (mut state, _, _) = crate::test_solves::solved(&moon, &words, schedule);
        // The ground moves after the last routing (a block of nodes lifted, another cut), as a pass's sweeps move it.
        for (i, z) in state.z.iter_mut().enumerate().step_by(7) {
            *z += [-40, 25][i & 1];
        }
        let lattice = state.lattice;
        let climate = crate::climate::climate_on(
            &moon,
            &lattice,
            &words,
            &state.z,
            Some(state.sea_view()),
            crate::par::Cores(1),
        );
        let mut full = state.clone();
        let routed = full.route();
        full.accumulate();
        let by_routing = full.follow_the_budget(&climate, true);
        let mut label = state;
        let labelled = label.label_for_budget();
        let by_label = label.follow_the_budget(&climate, true);
        assert!(
            routed.raised > 0,
            "the moved ground holds hollows the routing fills"
        );
        assert_eq!(
            (
                labelled.pockets,
                labelled.pocket_nodes,
                labelled.outlets,
                labelled.sea_seeded
            ),
            (
                routed.pockets,
                routed.pocket_nodes,
                routed.outlets,
                routed.sea_seeded
            )
        );
        assert_eq!(by_label, by_routing);
        assert_eq!(label, full);
    }

    #[test]
    fn the_cores_give_the_same_bytes() {
        let moon = home_moon();
        let words = SolveWords {
            water_km3: 3_000_000,
            ..crate::home::home_solve_words()
        };
        let schedule = Schedule {
            age_yr: 60_000_000,
            ..Schedule::standard(HOME_SYSTEM_AGE_YR)
        };
        let one = solve_full_on(&moon, &words, schedule, crate::par::Cores(1)).expect("a solve");
        let three = solve_full_on(&moon, &words, schedule, crate::par::Cores(3)).expect("a solve");
        assert!(one.2.sweeps.iter().any(|s| s.lowered > 0), "the rivers cut");
        assert!(
            one.2.routes.len() > ROUTINGS as usize,
            "the passes route more than once"
        );
        assert_eq!(one.0.z, three.0.z);
        assert_eq!(one.0.water_z, three.0.water_z);
        assert_eq!(one.1, three.1);
        assert_eq!(one.2, three.2);
        assert_eq!(one.0, three.0);
    }

    /// ★ A RING OF RECEIVERS IN THE SWEEP BY RIVER BASIN (ruling B64): two nodes that drain into each other stand
    /// outside the sweep's order, and a third that drains into the ring is swept against the ring's node as it stands —
    /// a read-only root of its tree — the same on one core and on two.
    #[test]
    fn a_ring_of_receivers_is_a_read_only_root() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        state.sea_z = i32::MIN;
        let _ = state.route();
        state.accumulate();
        // Three nodes of one face row: `a` and `b` drain into each other, `c` drains into `a` over its own height.
        let (a, b, c) = (
            state.lattice.index(Face::PosX, 30, 30),
            state.lattice.index(Face::PosX, 31, 30),
            state.lattice.index(Face::PosX, 32, 30),
        );
        state.receiver[a as usize] = b;
        state.receiver[b as usize] = a;
        state.receiver[c as usize] = a;
        state.chord[c as usize] = 8_192;
        state.z[c as usize] = state.z[a as usize] + 500 * Z_STEPS_PER_M;
        state.discharge[c as usize] = 1_000_000_000_000_000;
        state.order = order_of(&state.receiver);
        assert!(!state.order.contains(&a));
        assert!(state.order.contains(&c));
        let groups = state.basin_groups();
        assert!(
            groups.lists.iter().any(|l| l.contains(&a)),
            "the ring's node is a read-only root"
        );
        let mut two = state.clone();
        two.cores = crate::par::Cores(2);
        let dt = Gf::from_i64(1_000_000);
        let r1 = state.sweep(dt, None, None);
        let r2 = two.sweep(dt, None, None);
        assert_eq!(r1, r2);
        assert_eq!(state.z, two.z);
        assert!(
            state.cut_now[c as usize] > 0,
            "the node over the ring is cut"
        );
    }

    /// ★ THE ROUTINGS SHARE THE SWEEPS (ruling B65 item 1): eight sweeps and four routings route again before sweeps 2,
    /// 4 and 6; three routings before 3 and 6; one routing never again; more routings than sweeps route before every
    /// sweep but the first; and no schedule routes again before its first sweep.
    #[test]
    fn the_routings_share_the_sweeps() {
        let at = |sweeps: u32, routings: u32| -> Vec<u32> {
            let schedule = Schedule {
                sub_sweeps: sweeps,
                routings,
                ..Schedule::standard(HOME_SYSTEM_AGE_YR)
            };
            (0..sweeps).filter(|&s| schedule.routes_before(s)).collect()
        };
        assert_eq!(at(8, 4), vec![2, 4, 6]);
        assert_eq!(at(8, 3), vec![3, 6]);
        assert_eq!(at(8, 1), Vec::<u32>::new());
        assert_eq!(at(4, 9), vec![1, 2, 3]);
        assert_eq!(at(8, 0), Vec::<u32>::new());
        assert_eq!(Schedule::standard(HOME_SYSTEM_AGE_YR).routings, ROUTINGS);
    }

    #[test]
    fn the_home_moon_solves_and_every_node_drains() {
        let moon = home_moon();
        let before = MacroSolve::new(&moon).expect("a state");
        let (state, report) =
            solve(&moon, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        assert_eq!(report.sweeps.len(), 414 * SUB_SWEEPS as usize);
        assert_eq!(report.routes.len(), 414 * ROUTINGS as usize);
        for r in &report.routes {
            assert_eq!((r.undrained, r.cyclic), (0, 0), "{r:?}");
            assert!(r.outlets > 0);
        }
        assert!(report.sweeps[0].lowered > 0);
        // ★ THE SWEEPS ONLY LOWER, AND THE DEPOSIT ONLY FILLS A HOLLOW TO ITS SPILL (2026-09-22): a
        // node stands where it started or lower, unless the sediment raised it, and then it stands
        // at or under the flood level the last routing found.
        assert!(
            report.sweeps.iter().any(|s| s.filled > 0),
            "some hollow filled"
        );
        for (i, (&a, &b)) in before.z.iter().zip(state.z.iter()).enumerate() {
            // One bound, no short circuit (HR5 rule d): at or under the higher of its start and its spill.
            let spill = state.z_flood[i];
            assert!(
                b <= a.max(spill),
                "node {i} rose from {a} to {b} past its spill {spill}"
            );
        }
        let (lo, hi) = state.range();
        let (lo0, hi0) = before.range();
        assert!(lo >= lo0, "the floor fell: {lo0} to {lo}");
        assert!(hi <= hi0, "the ceiling rose: {hi0} to {hi}");
        // 64 before the rock map; the province's own byte makes 65 (slice 8d step 2), the lakes'
        // own water word makes 69 (ruling W11), the ocean's own bit makes 70 (ruling B7), and the
        // exact rebound's carried fraction makes 78 (ruling B45 item 1).
        assert_eq!(state.bytes_per_node(), 78);
        assert_eq!(state.node_count(), 27_744);
        // The discharge is conserved: what leaves at the outlets is every node's own rain.
        let total: u64 = (0..state.node_count())
            .map(|i| u64::from(state.rain[i]) * state.area[i])
            .sum();
        let out: u64 = (0..state.node_count())
            .filter(|&i| state.receiver[i] == NO_NODE)
            .map(|i| state.discharge[i])
            .sum();
        assert_eq!(out, total);
    }

    /// ★ THE UPLIFT OF THE PASSES sums to each node's even total to the sixteenth over the mixed axis, a negative total
    /// steps down, and a wedge node rises by its even total plus its own wedge's rise by the end; a sweep of no time
    /// cuts nothing, and neither does a sweep whose every node has no share of the pass.
    #[test]
    fn the_uplift_steps_sum_exactly_and_a_trench_subsides() {
        use crate::belts::{Belts, SIDE_PRO, WedgeNode, wedge_area_m2, wedge_height_m};
        let age = HOME_SYSTEM_AGE_YR;
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        let before = state.z.clone();
        let mut even = vec![1_601i64; n];
        even[0] = -803;
        let wedge = WedgeNode {
            node: 1,
            side: SIDE_PRO,
            dist_m: Gf::ZERO,
            f0_m2_yr: Gf::from_f64(10.0),
            start_yr: age - 50_000_000,
            ceil_m: Gf::from_f64(3_000.0),
            tau_yr: None,
        };
        let belts = Belts {
            total: vec![0; n],
            start_yr: vec![age; n],
            kind: vec![crate::belts::BELT_NONE; n],
            plateau: vec![false; n],
            wedge: vec![wedge],
            report: crate::belts::BeltReport::default(),
        };
        for w in Schedule::standard(age).times().windows(2) {
            pass_uplift(&mut state.z, &even, &belts, age, w[0], w[1]);
        }
        assert_eq!(state.z[0] - before[0], -803);
        let (h, _) = wedge_height_m(&wedge, wedge_area_m2(&wedge, age));
        let rise = (h * Gf::from_i64(i64::from(Z_STEPS_PER_M))).to_i64_floor();
        assert!(rise > 0, "{rise}");
        assert_eq!(i64::from(state.z[1] - before[1]), 1_601 + rise);
        for (z, b) in state.z.iter().zip(&before).skip(2) {
            assert_eq!(z - b, 1_601);
        }
        // No time, and no share: no cut.
        state.route();
        state.accumulate();
        assert_eq!(state.sweep(Gf::ZERO, None, None).lowered, 0);
        let none = vec![0u32; n];
        let s = state.sweep(Gf::from_i64(125_000_000), Some(&none), None);
        assert_eq!(s.lowered, 0);
        assert!(state.removed.iter().all(|&r| r == 0));
    }

    /// ★ THE BASINS AND THEIR INTEGRALS on the plateau: every node drains to the one outlet, so the
    /// one basin holds every land node; its integral over the land (the plateau at 16 with the
    /// pit at 8) stands near one; a flat basin has no integral; the sediment after a sweep is
    /// booked to that basin.
    #[test]
    fn basins_integrals_and_sediment_on_the_plateau() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![Z_STEPS_PER_M; n];
        state.sea_z = 0;
        let outlet = state.lattice.index(Face::PosX, 3, 3);
        state.z[outlet as usize] = 0;
        let pit = state.lattice.index(Face::NegZ, 30, 30);
        state.z[pit as usize] = Z_STEPS_PER_M / 2;
        state.route();
        let basins = state.basins();
        assert!(basins.iter().all(|&b| b == outlet));
        let integrals = state.hypsometric_integrals(100);
        assert_eq!(integrals.len(), 1);
        assert_eq!(integrals[0].outlet, outlet);
        assert_eq!(integrals[0].nodes as usize, n - 1);
        assert!(integrals[0].integral_q8 >= 254, "{:?}", integrals[0]);
        assert_eq!(state.hypsometric_integrals(n as u32), Vec::new());
        assert_eq!(
            state.hypsometric_integrals(0).len(),
            1,
            "a floor of zero still needs land"
        );
        // A flat land (the pit filled) has no integral.
        state.z[pit as usize] = Z_STEPS_PER_M;
        assert_eq!(state.hypsometric_integrals(100), Vec::new());
        // The sediment: nothing removed yet; after a sweep the basin holds the cut volume.
        assert!(state.sediment_by_basin().is_empty());
        state.accumulate();
        state.sweep(Gf::from_i64(125_000_000), None, None);
        let sediment = state.sediment_by_basin();
        assert_eq!(sediment.len(), 1);
        let total: u64 = (0..n)
            .map(|i| u64::from(state.removed[i]) * state.area[i] / u64::from(Z_STEPS_PER_M as u32))
            .sum();
        let booked = sediment[&outlet];
        assert!(booked > 0);
        assert!(booked <= total);
        assert!(
            total - booked <= n as u64,
            "the per-node floors: {total} vs {booked}"
        );
    }

    /// The moon's lattice as a cone on face `PosX`: every node 100 m plus one metre per node of distance from the
    /// outlet at (20, 20), the other faces at 160 m, the outlet at the sea's level 0 — every river on it cuts.
    fn cone() -> (MacroSolve, u32) {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        for node in 0..n as u32 {
            let (face, i, j) = state.lattice.split(node);
            let d = if face == Face::PosX {
                (i - 20).abs().max((j - 20).abs())
            } else {
                60
            };
            state.z[node as usize] = Z_STEPS_PER_M * (100 + d);
        }
        state.sea_z = 0;
        let outlet = state.lattice.index(Face::PosX, 20, 20);
        state.z[outlet as usize] = 0;
        (state, outlet)
    }

    /// ★ THE DEPOSIT (W7): on the cone with one pit, a stated cut upstream of the pit is laid in the pit up to its
    /// spill and no further, the rest runs on to the outlet and sinks, and a node that was cut is NOT refilled (the pit
    /// is the routing's, never the cut's); the fill is booked against the rebound.
    #[test]
    fn the_deposit_fills_the_pit_to_its_spill_and_no_cut_node() {
        let (mut state, _) = cone();
        let pit = state.lattice.index(Face::PosX, 30, 30);
        state.z[pit as usize] = Z_STEPS_PER_M * 70;
        state.route();
        state.accumulate();
        // The spill is the downhill neighbour at nine nodes: 109 m; the pit is 39 m deep.
        assert_eq!(
            state.pit[pit as usize] as i32,
            Z_STEPS_PER_M * 39,
            "the pit's depth"
        );
        let above = state.lattice.index(Face::PosX, 31, 31);
        assert_eq!(state.receiver[above as usize], pit);
        let beside = state.lattice.index(Face::PosX, 21, 20);
        let before = state.z[beside as usize];
        state.cut_now[above as usize] = (Z_STEPS_PER_M * 50) as u32;
        state.cut_now[beside as usize] = Z_STEPS_PER_M as u32;
        let (filled, total) = state.deposit();
        assert_eq!(filled, 1, "the one pit filled");
        assert_eq!(total, (Z_STEPS_PER_M * 39) as u64);
        assert_eq!(
            state.z[pit as usize],
            Z_STEPS_PER_M * 109,
            "filled to its spill and no further"
        );
        assert_eq!(state.pit[pit as usize], 0);
        assert_eq!(state.pending[pit as usize], -Z_STEPS_PER_M * 39);
        assert_eq!(
            state.z[beside as usize], before,
            "a cut node is never refilled"
        );
    }

    /// ★ THE LAKES' CURE (ruling B46 item 2): a lake's level follows its outlet INSIDE the sweep. On the cone a pit 12 m
    /// under its spill drains over the spill node; when the spill node alone takes a trunk's discharge for a thousand
    /// years and is cut, the lake's level falls to the spill's new height in the same sweep and the lake is skipped;
    /// when the trunk below the spill takes it too for a million years, the spill falls under the pit's own floor, the
    /// pit is dry ground and the sweep cuts it like any slope.
    #[test]
    fn a_lakes_level_follows_its_outlet_inside_the_sweep() {
        let (mut state, _) = cone();
        let pit = state.lattice.index(Face::PosX, 23, 23);
        state.z[pit as usize] = Z_STEPS_PER_M * 90;
        state.route();
        state.accumulate();
        let spill = state.lattice.index(Face::PosX, 22, 22);
        assert_eq!(state.receiver[pit as usize], spill);
        assert_eq!(state.pit[pit as usize] as i32, Z_STEPS_PER_M * 12);
        let routed = state.clone();
        // The spill alone takes a trunk's water for a thousand years: it is cut, and the lake follows it.
        state.discharge[spill as usize] = 1_000_000_000_000_000;
        let report = state.sweep(Gf::from_i64(1_000), None, None);
        let z_spill = state.z[spill as usize];
        assert!(
            z_spill < Z_STEPS_PER_M * 102,
            "the spill was cut: {z_spill}"
        );
        assert!(
            z_spill > Z_STEPS_PER_M * 90,
            "and stands over the pit's floor: {z_spill}"
        );
        assert!(report.skipped_lake >= 1);
        assert_eq!(state.z_flood[pit as usize], z_spill);
        assert_eq!(
            state.pit[pit as usize] as i32,
            z_spill - state.z[pit as usize]
        );
        // The trunk below the spill takes the water too: the spill falls under the pit's floor, the pit is dry, and the
        // sweep cuts it.
        let mut dry = routed;
        let below = state.lattice.index(Face::PosX, 21, 21);
        dry.discharge[spill as usize] = 1_000_000_000_000_000;
        dry.discharge[below as usize] = 1_000_000_000_000_000;
        dry.sweep(Gf::from_i64(1_000_000), None, None);
        let cut_to = dry.z[spill as usize];
        assert!(cut_to < Z_STEPS_PER_M * 90, "{cut_to}");
        assert_eq!(dry.pit[pit as usize], 0);
        assert!(
            dry.z[pit as usize] < Z_STEPS_PER_M * 90,
            "the dry pit was cut"
        );
    }

    /// ★ THE REBOUND of a stated cut: the rebound lifts the cut node and its surroundings by the crust's share of the
    /// smoothed cut, spread over the level nearest the flexural parameter, and consumes the pending removal; ★ EXACT
    /// (ruling B45 item 1): the whole sixteenths lifted plus the fractions carried are the crust's share of the cut,
    /// and every carried fraction stands under one sixteenth; a rebound with nothing pending lifts nothing.
    #[test]
    fn the_rebound_lifts_the_crusts_share_of_the_cut() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![Z_STEPS_PER_M * 100; n];
        let at = state.lattice.index(Face::PosX, 20, 20);
        state.pending[at as usize] = 800;
        for m in state.lattice.neighbours(at) {
            state.pending[m as usize] = 160;
        }
        let cut: i64 = state.pending.iter().map(|&p| i64::from(p)).sum();
        let before = state.z.clone();
        // A parameter of about four nodes: level 2 (32 768 m against 8 192 · 4).
        assert_eq!(state.coarse_level(33_000.0), 2);
        assert_eq!(state.coarse_level(1.0), 0);
        let (level, lifted, max_lift) = state.rebound(33_000.0);
        assert_eq!(level, 2);
        assert!(lifted > 0);
        assert!(max_lift > 0);
        assert!(state.pending.iter().all(|&p| p == 0));
        let lift: i64 = (0..n).map(|i| i64::from(state.z[i] - before[i])).sum();
        let carried: f64 = state.rebound_carry.iter().map(|c| c.to_f64()).sum();
        let share = crate::land::CONTINENTAL_CRUST_DENSITY_KGM3 / crate::land::MANTLE_DENSITY_KGM3;
        let owed = share * cut as f64;
        assert!(
            (lift as f64 + carried - owed).abs() < 1e-6 * owed,
            "lift {lift} and carried {carried} against {owed}"
        );
        assert!(
            state
                .rebound_carry
                .iter()
                .all(|&c| (c >= Gf::ZERO) & (c < Gf::ONE))
        );
        // Nothing pending: nothing lifted (every carried fraction stands under a sixteenth).
        let again = state.z.clone();
        assert_eq!(state.rebound(33_000.0), (2, 0, 0));
        assert_eq!(state.z, again);
        // The finest level when the parameter is under a node.
        let (level, _, _) = state.rebound(1.0);
        assert_eq!(level, 0);
    }

    /// ★ THE SMOOTH READ (phase 1 of the ground step; the belts' design L10): one block of the
    /// home moon's level 2 owes 16 sixteenths a node; the hand-over lifts EXACTLY its owed total
    /// (P14: conserved to the sixteenth), the lift ramps out over the next block's ground — no
    /// step between two neighbours exceeds half the block read's step of 16 (MEASURED 7 inside a
    /// face, a diagonal step across the ramp's top; 8 at a cube corner) — and a block at a cube
    /// corner, whose diagonal centre does not exist, hands over its total exactly too.
    #[test]
    fn the_smooth_read_hands_each_block_its_lift_as_a_ramp() {
        let moon = home_moon();
        let state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        let k = 2u32;
        let coarse = state.lattice.coarser(k).expect("level 2");
        let m = coarse.node_count();
        let parent: Vec<u32> = (0..n as u32)
            .map(|node| {
                let (face, i, j) = state.lattice.split(node);
                coarse.index(face, i >> k, j >> k)
            })
            .collect();
        let ring: Vec<[u32; 8]> = (0..m as u32).map(|c| coarse.neighbours(c)).collect();
        for (face, ci, cj) in [(Face::PosX, 8, 8), (Face::NegZ, 0, 0)] {
            let block = coarse.index(face, ci, cj);
            let mut owed = vec![Gf::ZERO; m];
            owed[block as usize] = Gf::from_i64(16 << (2 * k));
            let lifts = state.smooth_lifts(k, &parent, &ring, &owed);
            let total: f64 = lifts.iter().map(|l| l.to_f64()).sum();
            assert!(
                (total - f64::from(16u32 << (2 * k))).abs() < 1e-9,
                "the block at {face:?} lifts what it owes: {total}"
            );
            assert!(lifts.iter().all(|&l| l >= Gf::ZERO));
            let step = (0..n as u32)
                .flat_map(|a| {
                    let lifts = &lifts;
                    state
                        .lattice
                        .neighbours(a)
                        .into_iter()
                        .filter(|&b| b != NO_NODE)
                        .map(move |b| (lifts[a as usize] - lifts[b as usize]).abs().to_f64())
                })
                .fold(0.0f64, |a, b| if b > a { b } else { a });
            assert!(step <= 8.0, "the largest step {step} at {face:?}");
        }
        // The stencil's slot formula is the stencil's own order.
        for (slot, &(di, dj)) in crate::macro_lattice::STENCIL.iter().enumerate() {
            let r = ((dj + 1) * 3 + (di + 1)) as usize;
            assert_eq!(r - usize::from(r > 4), slot);
        }
    }

    /// ★ THE TALUS on a stated wall: a node standing a kilometre over its neighbours sheds half its
    /// excess to them, the mass is conserved to the sixteenth, and the worst excess falls pass
    /// after pass (03 §14 M13). The tangent of repose reads the aridity between its two ends.
    #[test]
    fn the_talus_sheds_half_the_excess_and_conserves_the_mass() {
        assert_eq!(tan_repose_q16(255), (0.70 * 65_536.0) as u32);
        assert_eq!(tan_repose_q16(0), (0.47 * 65_536.0) as u32);
        let mid = tan_repose_q16(128);
        assert!(mid > tan_repose_q16(0));
        assert!(mid < tan_repose_q16(255));
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![0; n];
        let peak = state.lattice.index(Face::PosZ, 30, 30);
        state.z[peak as usize] = 1_000 * Z_STEPS_PER_M;
        let tan = vec![tan_repose_q16(255); n];
        let mass_before: i64 = state.z.iter().map(|&z| i64::from(z)).sum();
        let (shed, worst) = state.talus(&tan);
        // The excess of a kilometre drop past 8 192 · 0.70 = 5 734 m: none, nothing sheds.
        assert_eq!((shed, worst), (0, 0));
        // A five-kilometre-tall spike over an 8 km chord stands past 35°: it sheds.
        state.z[peak as usize] = 10_000 * Z_STEPS_PER_M;
        let mass_before =
            mass_before - 1_000 * i64::from(Z_STEPS_PER_M) + 10_000 * i64::from(Z_STEPS_PER_M);
        let (shed, worst) = state.talus(&tan);
        assert_eq!(shed, 1);
        assert!(worst > 0);
        let mass_after: i64 = state.z.iter().map(|&z| i64::from(z)).sum();
        assert_eq!(mass_after, mass_before, "the mass moved, never made");
        assert!(state.z[peak as usize] < 10_000 * Z_STEPS_PER_M);
        for m in state.lattice.neighbours(peak) {
            assert!(state.z[m as usize] > 0, "a neighbour received talus");
        }
        let (_, worse) = state.talus(&tan);
        assert!(worse < worst, "the excess fell: {worst} then {worse}");
    }

    /// ★ THE ICE IS A MASS BALANCE, AND NO EXTENT IS DRAWN (ruling B2 step 2). A line over every
    /// node leaves no ice at all. A line under one peak gives that peak ice, and the glacier ENDS
    /// where the ablation below eats what the catchment sends; ★ STEPPED WITH NO CAP (ruling B44
    /// item 1) its cut over the epoch passes the ice's own first thickness, in steps, and every cut
    /// is booked for the rebound. The sea and a lake are not ice's ground. An outlet holds ice and
    /// cuts nothing, because it has nowhere to flow. A bed the year keeps far under freezing is
    /// cold-based and erodes nothing.
    #[test]
    fn the_ice_is_a_mass_balance_and_a_cold_bed_cuts_nothing() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![Z_STEPS_PER_M * 100; n];
        state.sea_z = 0;
        let outlet = state.lattice.index(Face::PosX, 20, 20);
        state.z[outlet as usize] = 0;
        let peak = state.lattice.index(Face::PosX, 21, 20);
        state.z[peak as usize] = 3_000 * Z_STEPS_PER_M;
        let pit = state.lattice.index(Face::NegY, 10, 10);
        state.z[pit as usize] = 50 * Z_STEPS_PER_M;
        state.route();
        let temperate = vec![2_531i16; n];
        let lapse = 6_500u32;
        let untouched = state.clone();
        let no_ice = vec![i32::MAX; n];
        let (mask, under, thickest) = state.ice_mask(&no_ice, &temperate, lapse, 330);
        assert_eq!(under, 0);
        assert_eq!(thickest, Gf::ZERO);
        assert!(mask.iter().all(|&m| !m));
        assert_eq!(
            state.ice_stepped(&no_ice, &temperate, lapse, 330, GLACIAL_EPOCH_YR),
            (0, 1, 0)
        );
        // The line at 2 000 m: the peak stands 1 000 m over it and gains ice; every neighbour
        // stands 1 900 m under it and loses more than the peak sends, so the glacier is one node.
        let line = vec![2_000 * Z_STEPS_PER_M; n];
        let before = state.z[peak as usize];
        let (mask, under, thickest) = state.ice_mask(&line, &temperate, lapse, 330);
        assert_eq!(under, 1);
        assert!(mask[peak as usize]);
        assert!(thickest > Gf::ZERO);
        let (cut, steps, floored) =
            state.ice_stepped(&line, &temperate, lapse, 330, GLACIAL_EPOCH_YR);
        assert_eq!(floored, 0, "no step under a year");
        assert!(cut > 0);
        assert_eq!(
            i64::from(before - state.z[peak as usize]),
            cut,
            "the peak alone is cut"
        );
        assert_eq!(
            i64::from(state.pending[peak as usize]),
            cut,
            "booked for the rebound"
        );
        assert!(steps >= 2, "{steps} steps");
        let thick = thickest.to_f64();
        assert!(
            Gf::from_i64(cut) > thickest * Gf::from_i64(i64::from(Z_STEPS_PER_M)),
            "no cap at the ice's own thickness: {cut} against {thick}"
        );
        // ★ THE SAME PEAK ON A COLD BED cuts nothing: the year's mean is 70 K under freezing, so
        // the geothermal flux through the ice cannot bring the bed to its melting point.
        let mut cold_state = untouched;
        let frozen = vec![2_000i16; n];
        let z_before = cold_state.z.clone();
        let (cold_mask, cold_under, _) = cold_state.ice_mask(&line, &frozen, lapse, 330);
        assert_eq!(cold_under, 1);
        assert!(cold_mask[peak as usize]);
        assert_eq!(
            cold_state.ice_stepped(&line, &frozen, lapse, 330, GLACIAL_EPOCH_YR),
            (0, 1, 0)
        );
        assert_eq!(cold_state.z, z_before);
        // The line at the sea: the sea's outlet and the pit (a lake) are not ice's ground; the
        // flat plateau has ice and cuts nothing, because it has no drop to its receiver.
        let z_before = state.z.clone();
        let low = vec![0; n];
        let (mask, under, _) = state.ice_mask(&low, &temperate, lapse, 330);
        assert!(!mask[outlet as usize]);
        assert!(!mask[pit as usize]);
        assert!(under > 0);
        let flat = state.lattice.index(Face::PosY, 5, 5);
        assert!(mask[flat as usize]);
        let _ = state.ice_stepped(&low, &temperate, lapse, 330, 1_000);
        assert_eq!(state.z[flat as usize], z_before[flat as usize]);
        // No sea: the lowest nodes are outlets, so they hold ice and cut nothing (no receiver);
        // and a node one metre over the line with a one-sixteenth drop over eight kilometres cuts
        // under a sixteenth in a thousand years: its fraction is carried, never laid.
        let mut dry = MacroSolve::new(&moon).expect("a state");
        dry.z = vec![Z_STEPS_PER_M * 100; n];
        dry.sea_z = i32::MIN;
        let top = dry.lattice.index(Face::PosZ, 40, 40);
        let below = dry.lattice.index(Face::PosZ, 41, 40);
        dry.z[top as usize] = Z_STEPS_PER_M * 100 + Z_STEPS_PER_M;
        dry.z[below as usize] = Z_STEPS_PER_M * 100 + Z_STEPS_PER_M - 1;
        dry.route();
        let outlet = (0..n)
            .find(|&i| dry.receiver[i] == NO_NODE)
            .expect("an outlet");
        let line = vec![Z_STEPS_PER_M * 100; n];
        let z_before = dry.z.clone();
        let (mask, _, _) = dry.ice_mask(&line, &temperate, lapse, 330);
        let _ = dry.ice_stepped(&line, &temperate, lapse, 330, 1_000);
        assert!(mask[outlet]);
        assert_eq!(dry.z[outlet], z_before[outlet]);
        assert!(mask[top as usize]);
        assert_eq!(
            dry.z[top as usize], z_before[top as usize],
            "a cut under a sixteenth is no cut"
        );
    }

    /// The mass balance itself: a node over its line gains, a node under it loses more per kelvin,
    /// and a bed the year keeps over freezing melts on top of that.
    #[test]
    fn the_mass_balance_gains_over_the_line_and_loses_under_it() {
        let lapse = Gf::from_f64(0.006_5);
        let line = 1_000 * Z_STEPS_PER_M;
        let cold = Gf::from_f64(-10.0);
        let above = ice_balance_m_yr(line + 600 * Z_STEPS_PER_M, line, lapse, cold);
        assert!((above.to_f64() - 0.39).abs() < 0.001, "{}", above.to_f64());
        let on = ice_balance_m_yr(line, line, lapse, cold);
        assert_eq!(on, Gf::ZERO);
        let below = ice_balance_m_yr(line - 1_200 * Z_STEPS_PER_M, line, lapse, cold);
        assert!((below.to_f64() + 1.17).abs() < 0.001, "{}", below.to_f64());
        // A warm year takes its own share at the bed, on top of the surface's balance.
        let warm = ice_balance_m_yr(line + 600 * Z_STEPS_PER_M, line, lapse, Gf::from_f64(4.0));
        let warm_m = warm.to_f64();
        assert!((warm_m - (above.to_f64() - 0.2)).abs() < 0.001, "{warm_m}");
        assert_eq!(
            celsius(2_731),
            Gf::from_f64(273.1) - Gf::from_f64(crate::climate::FREEZING_K)
        );
    }

    /// ★ THE FACIES AND THE COAST BAND on the plateau: the outlet is sea and coast, the pit a lake,
    /// an ice node ice; an airless body has no band and the home planet's is a few metres.
    #[test]
    fn facies_and_the_coast_band() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![Z_STEPS_PER_M; n];
        state.sea_z = 0;
        let outlet = state.lattice.index(Face::PosX, 3, 3);
        state.z[outlet as usize] = 0;
        let pit = state.lattice.index(Face::NegZ, 30, 30);
        state.z[pit as usize] = Z_STEPS_PER_M / 2;
        state.route();
        let mut ice = vec![false; n];
        let icy = state.lattice.index(Face::PosY, 7, 7);
        ice[icy as usize] = true;
        // ★ THE FILL IS NOT WATER (ruling W11): the flood raised the pit, and until the budget
        // states a level the node is dry land. The budget's own level makes it a lake.
        assert!(state.z_flood[pit as usize] > state.z[pit as usize]);
        assert_eq!(state.facies(&ice, 4, true)[pit as usize], 0);
        state.water_z[pit as usize] = Z_STEPS_PER_M;
        let facies = state.facies(&ice, 4, true);
        assert_eq!(facies[outlet as usize], FACIES_SEA | FACIES_COAST);
        assert_eq!(facies[pit as usize], FACIES_LAKE);
        assert_eq!(facies[icy as usize], FACIES_ICE);
        assert_eq!(facies[state.lattice.index(Face::PosY, 8, 8) as usize], 0);
        // A dry body: the same hollows are closed basins, never lakes, and there is no sea.
        let dry = state.facies(&ice, 4, false);
        assert_eq!(dry[outlet as usize], FACIES_COAST);
        assert_eq!(dry[pit as usize], 0);
        assert_eq!(coast_band(&crate::home::home_moon_solve_words(), 330), 0);
        let home_band = coast_band(&crate::home::home_solve_words(), 9_818);
        assert!((100..=200).contains(&home_band), "{home_band}");
    }

    /// ★ THE PEAK LAW (limits as law, the belts' design L0): on the moon's lattice a node twice the cap over the datum
    /// collapses where it stands and its excess runs to its lowest neighbour, round after round, until no node stands
    /// over the cap — the volume moved to the sixteenth below on each neighbour; a node at a cube corner, whose eighth
    /// neighbour does not exist, sheds to the seven that do; a field under the cap is left alone; and on a DRY body the
    /// datum is the ladder's (ruling B25 item 1), so a node over the radius is read over it and nothing wraps.
    #[test]
    fn the_peak_law_collapses_a_node_over_its_cap_and_moves_its_volume() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![0; n];
        state.sea_z = 0;
        assert_eq!(state.datum_z(), 0);
        let cap = 1_000 * Z_STEPS_PER_M;
        assert_eq!(
            state.peak_law(cap),
            (0, 0),
            "a field under the cap is left alone"
        );
        let peak = state.lattice.index(Face::PosX, 30, 30);
        let corner = state.lattice.index(Face::PosX, 0, 0);
        assert!(state.lattice.neighbours(corner).contains(&NO_NODE));
        state.z[peak as usize] = 2 * cap;
        state.z[corner as usize] = 3 * cap;
        let volume = |s: &MacroSolve| -> i128 {
            (0..n)
                .map(|i| i128::from(s.z[i]) * i128::from(s.area[i]))
                .sum()
        };
        let before = volume(&state);
        let (collapsed, left) = state.peak_law(cap);
        assert!(collapsed >= 2, "{collapsed}");
        assert_eq!(left, 0, "it settled");
        assert!(state.z.iter().all(|&z| z <= cap));
        assert_eq!(state.z[peak as usize], cap);
        let lost = before - volume(&state);
        let largest = i128::from(*state.area.iter().max().expect("an area"));
        assert!(
            (0..collapsed as i128 * largest).contains(&lost),
            "{lost} lost over {collapsed} collapses"
        );
        // A dry body: the datum is the ladder's, and a node over the radius is read over it.
        state.sea_z = i32::MIN;
        assert_eq!(state.datum_z(), 0);
        state.z[peak as usize] = 3 * cap;
        let (collapsed, left) = state.peak_law(cap);
        assert!(collapsed >= 1);
        assert_eq!(left, 0);
        assert_eq!(state.z[peak as usize], cap);
        // A sea over the datum lifts the ceiling with it.
        state.sea_z = 100;
        assert_eq!(state.datum_z(), 100);
        state.z[peak as usize] = cap + 100;
        assert_eq!(state.peak_law(cap), (0, 0));
    }

    /// ★ THE FULL SOLVE ON THE MOON (airless, dry, a stagnant lid): craters stamped, each in its own pass; no rain so
    /// no cut and no deposit; no ice, no coast band; no ridge opens an ocean, so no belt rises and no plate is reworked
    /// (every raise time is the system's age); every routing drained, seeded from the six lowest nodes with no sea;
    /// the peak law settled every pass and no node stands over its cap over the ladder's datum; no integral read (no
    /// sea, so no land over it); the facies carry no sea, no coast and no ice.
    #[test]
    fn the_full_solve_on_the_airless_moon() {
        let moon = home_moon();
        let words = crate::home::home_moon_solve_words();
        let (state, facies, report) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        assert!(report.craters > 0);
        assert_eq!(report.craters, report.crater_record.len());
        assert!(
            report.sweeps.iter().all(|s| s.lowered == 0),
            "no rain, no cut"
        );
        assert_eq!(report.deposits, 0);
        assert_eq!(report.ice, (0, 0.0, 0));
        assert_eq!(report.coast_band, 0);
        assert_eq!(report.belts.report.t_m_yr, 0);
        assert!(report.belts.wedge.is_empty());
        assert!(report.raise_yr.iter().all(|&t| t == HOME_SYSTEM_AGE_YR));
        assert_eq!(state.sea_z, i32::MIN);
        assert_eq!(report.peak.1, 0, "the peak law settled");
        let cap = (crate::land::relief_cap_m(&moon) * f64::from(Z_STEPS_PER_M)) as i32;
        assert!(state.z.iter().all(|&z| z <= cap));
        assert_eq!(report.rebounds.len(), 414 * SUB_SWEEPS as usize);
        assert_eq!(report.talus.len(), 8);
        assert_eq!(report.routes.len(), 414 * ROUTINGS as usize + 1);
        for r in &report.routes {
            assert!(!r.sea_seeded);
            assert_eq!(
                (r.outlets, r.undrained, r.cyclic),
                (OUTLETS_WITHOUT_SEA, 0, 0)
            );
        }
        assert!(
            facies
                .iter()
                .all(|&f| f & (FACIES_SEA | FACIES_LAKE | FACIES_COAST | FACIES_ICE) == 0)
        );
        assert_eq!(state.node_count(), facies.len());
    }

    /// A schedule of no age, so no pass: the climate is computed once at the end for the talus and
    /// the ice; the land and the talus still run, and no crater falls inside a pass.
    #[test]
    fn a_schedule_of_no_passes_still_reads_a_climate() {
        let moon = home_moon();
        let words = crate::home::home_moon_solve_words();
        let (_, _, report) = crate::test_solves::solved(&moon, &words, Schedule::standard(0));
        assert_eq!(report.sweeps.len(), 0);
        assert_eq!(report.routes.len(), 1);
        assert_eq!(report.talus.len(), 8);
        assert_eq!(report.craters, 0);
    }

    /// ★ THE FULL SOLVE WITH RAIN, on the moon's lattice under the home planet's words as a stated
    /// neighbourhood: the rivers cut, the deposits exist, the rebound lifts, the coast has a band,
    /// the sea's facies exist, and gate G-AGE reads integrals.
    #[test]
    fn the_full_solve_with_the_home_planets_words_on_the_moons_lattice() {
        let moon = home_moon();
        // The home planet's air and climate, with a sea that covers PART of the moon (the home
        // planet's whole ocean would drown it: two Earth oceans on a 355 km moon stand 1 700 km deep).
        let words = SolveWords {
            water_km3: 3_000_000,
            ..crate::home::home_solve_words()
        };
        let (state, facies, report) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        assert!(report.sweeps.iter().any(|s| s.lowered > 0), "the rain cuts");
        assert!(report.deposits > 0);
        assert!(report.rebounds.iter().any(|&(_, lifted, _)| lifted > 0));
        assert!(report.coast_band > 0);
        assert!(facies.iter().any(|&f| f & FACIES_SEA != 0));
        assert!(facies.iter().any(|&f| f & FACIES_COAST != 0));
        assert!(!report.integrals.is_empty());
        // ★ THE SEA RE-SOLVED (C5): the report names the level the state holds, and the ocean
        // share is the area under it — a part of the moon, never all of it, never none.
        assert_eq!(report.sea_z, Some(state.sea_z));
        // Two statements, never one `and`: a short-circuit hides the second half from the count.
        let share = report.ocean_share;
        assert!(share > 0.0, "{share}");
        assert!(share < 1.0, "{share}");
        assert!((state.ocean_share() - report.ocean_share).abs() < 1e-12);
        let wet_nodes = state.z.iter().filter(|&&z| z <= state.sea_z).count();
        let nodes = state.node_count();
        assert!(wet_nodes > 0, "{wet_nodes}");
        assert!(wet_nodes < nodes, "{wet_nodes} of {nodes}");
        // A dry state reads no share; a sea over every node reads one once the routing has labelled
        // its ocean (2026-09-24, ruling B7: the share is the CONNECTED ocean's).
        let mut dry = MacroSolve::new(&moon).expect("a state");
        dry.sea_z = i32::MIN;
        assert_eq!(dry.ocean_share(), 0.0);
        dry.sea_z = i32::MAX;
        assert_eq!(dry.ocean_share(), 0.0, "no ocean labelled yet");
        dry.route();
        assert_eq!(dry.ocean_share(), 1.0);
        dry.z.clear();
        dry.area.clear();
        assert_eq!(dry.ocean_share(), 0.0, "no area, no share");
        // A body with no macro lattice has no solve of any kind.
        let rock = moon.without_macro_lattice();
        assert!(MacroSolve::new(&rock).is_none());
        assert!(solve(&rock, Schedule::standard(HOME_SYSTEM_AGE_YR)).is_none());
        assert!(solve_full(&rock, &words, Schedule::standard(HOME_SYSTEM_AGE_YR)).is_none());
        // The seed-built body states no sea, so the bench's own start is dry.
        assert_eq!(sea_level(&moon), i32::MIN);
        assert_eq!(sea_level(&moon.with_sea_m(Some(-2))), -2 * Z_STEPS_PER_M);
        let (median, mature) = age_gate(&report.integrals).expect("a reading");
        assert!(median > 0);
        let _ = mature;
        assert_eq!(age_gate(&[]), None);
        let young = [BasinIntegral {
            outlet: 1,
            nodes: 200,
            integral_q8: 250,
        }];
        assert_eq!(age_gate(&young), Some((250, false)));
        let mature = [
            BasinIntegral {
                outlet: 1,
                nodes: 200,
                integral_q8: 120,
            },
            BasinIntegral {
                outlet: 2,
                nodes: 100,
                integral_q8: 60,
            },
        ];
        assert_eq!(age_gate(&mature), Some((120, true)));
        let old = [BasinIntegral {
            outlet: 3,
            nodes: 50,
            integral_q8: 60,
        }];
        assert_eq!(age_gate(&old), Some((60, false)));
        assert!(state.z.len() == facies.len());
        // The words' flags, read: the home planet turns, its moon is locked.
        assert!(!words.tidally_locked());
        assert!(crate::home::home_moon_solve_words().tidally_locked());
        assert!(words.has_air());
        assert!(!crate::home::home_moon_solve_words().has_air());
    }

    /// The flood's two seedings on the moon: with the sea under every node the six lowest nodes
    /// are the outlets; with the sea over every node every node is an outlet and nothing routes.
    #[test]
    fn the_flood_seeds_from_the_sea_or_from_the_lowest_nodes() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        state.sea_z = i32::MIN + 1;
        let r = state.route();
        assert!(!r.sea_seeded);
        assert_eq!(r.outlets, OUTLETS_WITHOUT_SEA);
        assert_eq!((r.undrained, r.cyclic), (0, 0));
        let lowest = state.range().0;
        let outlets: Vec<usize> = (0..state.node_count())
            .filter(|&i| state.receiver[i] == NO_NODE)
            .collect();
        assert_eq!(outlets.len(), OUTLETS_WITHOUT_SEA);
        assert!(outlets.iter().any(|&i| state.z[i] == lowest));
        for &i in &outlets {
            assert_eq!(state.water_level(i as u32), DRY);
        }
        state.sea_z = i32::MAX;
        let r = state.route();
        assert!(r.sea_seeded);
        assert_eq!(r.outlets, state.node_count());
        assert_eq!((r.raised, r.flat, r.undrained, r.cyclic), (0, 0, 0, 0));
        assert_eq!(state.order.len(), state.node_count());
        assert_eq!(
            state.sweep(Gf::from_i64(1 << 30), None, None),
            SweepReport::default()
        );
    }

    /// ★ THE GATE THE ROUTING FILL USED TO FAIL (ruling W11, 2026-09-22): a plateau one metre over
    /// the sea with one outlet and one pit. The flood RAISES the pit to the plateau — the scratch
    /// surface the receivers, the flats and the sweep's base level all read — and the pit's own
    /// WATER LEVEL is DRY, because no budget ever put water in it. Before W11 `water_level` read
    /// the fill and this test read `plateau`. A long sweep then cuts the plateau's rivers back from
    /// the outlet, the lake's level follows its own outlet down, and no node that drains into the
    /// lake is cut under the lake's level.
    #[test]
    fn the_routing_fill_is_not_water() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        let plateau = Z_STEPS_PER_M;
        state.z = vec![plateau; n];
        state.sea_z = 0;
        let outlet = state.lattice.index(Face::PosX, 3, 3);
        state.z[outlet as usize] = 0;
        let mid = state.lattice.edge as i32 / 2;
        let pit = state.lattice.index(Face::NegZ, mid, mid);
        state.z[pit as usize] = plateau / 2;
        let r = state.route();
        assert!(r.sea_seeded);
        assert_eq!(r.outlets, 1);
        assert_eq!(r.raised, 1);
        assert_eq!((r.undrained, r.cyclic), (0, 0));
        // Eight nodes touch the outlet and take it by slope; every other node is flat.
        assert_eq!(r.flat, n - 1 - 8);
        // ★ THE FILL IS A SCRATCH SURFACE: it raised the pit to the plateau and the pit's own
        // water word says DRY. The sweep still reads the fill as its base level.
        assert_eq!(state.z_flood[pit as usize], plateau);
        assert_eq!(state.pit[pit as usize] as i32, plateau - plateau / 2);
        assert_eq!(state.water_level(pit), DRY);
        assert_eq!(state.sweep_base(pit), plateau);
        assert_eq!(state.facies(&vec![false; n], 0, true)[pit as usize], 0);
        assert_eq!(state.water_level(outlet), 0);
        assert_eq!(state.sweep_base(outlet), 0);
        let dry = state.lattice.index(Face::PosY, 1, 1);
        assert_eq!(state.water_level(dry), DRY);
        assert_eq!(state.sweep_base(dry), plateau);
        // And a stated lake level makes the same node water again, at the level the budget says.
        state.water_z[pit as usize] = plateau;
        assert_eq!(state.water_level(pit), plateau);
        assert_eq!(
            state.facies(&vec![false; n], 0, true)[pit as usize],
            FACIES_LAKE
        );
        state.water_z[pit as usize] = DRY;
        for m in state.lattice.neighbours(outlet) {
            assert_eq!(state.receiver[m as usize], outlet);
            assert!(state.chord[m as usize] > 0);
        }
        state.accumulate();
        let total: u64 = state.area.iter().sum::<u64>() * u64::from(P_MIN_MM_YR);
        assert_eq!(state.discharge[outlet as usize], total);
        let s = state.sweep(Gf::from_i64(125_000_000), None, None);
        assert_eq!(s.skipped_lake, 1);
        assert!(s.lowered >= 1, "{s:?}");
        assert!(s.max_cut > 0);
        // ★ THE LAKE FOLLOWED ITS OUTLET (ruling B46 item 2): its level stands at or under the plateau it was raised to,
        // over its own floor, and the deposit filled it at most to that level.
        let level = state.z_flood[pit as usize];
        assert!(level <= plateau, "{level}");
        assert!(level > plateau / 2, "{level}");
        assert!(state.z[pit as usize] >= plateau / 2);
        assert!(state.z[pit as usize] <= level);
        // A node whose receiver is the lake reads the lake's LEVEL as its base, so it is never cut
        // below that level toward the hole.
        let donors: Vec<usize> = (0..n).filter(|&i| state.receiver[i] == pit).collect();
        assert!(!donors.is_empty());
        for i in donors {
            assert!(state.z[i] >= level, "{i}: {} under {level}", state.z[i]);
        }
        // The stencil order is the fixed one, and a routed node's receiver is a stencil neighbour.
        assert_eq!(STENCIL[0], (-1, -1));
        let some = state.lattice.index(Face::PosX, 3, 4);
        assert!(
            state
                .lattice
                .neighbours(some)
                .contains(&state.receiver[some as usize])
        );
    }

    /// A plateau at 16, nine nodes of sea at 0, and a pocket of two nodes far inland under the sea:
    /// its pit at −50 and a wall at −20.
    fn pocket_plateau() -> (MacroSolve, usize, usize) {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![Z_STEPS_PER_M; n];
        state.sea_z = 0;
        for i in 2..=4 {
            for j in 2..=4 {
                state.z[state.lattice.index(Face::PosX, i, j) as usize] = 0;
            }
        }
        let q = state.lattice.index(Face::NegZ, 40, 40) as usize;
        let wall = state.lattice.index(Face::NegZ, 41, 40) as usize;
        state.z[q] = -50;
        state.z[wall] = -20;
        (state, q, wall)
    }

    /// ★ TESTS 17 AND 18 (plan step 2 part 3a; ruling B7 item 1): THE ROUTING SEEDS THE OCEAN ALONE
    /// AND RAISES A POCKET LIKE ANY HOLLOW; the water level, the facies and the ocean share leave
    /// the pocket out. The pocket's nodes are no outlets: each has a receiver, the flood raised
    /// them to the plateau, and the sweep's base level is that spill. Its water level is DRY (no
    /// budget put water there), its facies carry no sea, and the share is the ocean's nine nodes.
    #[test]
    fn the_routing_seeds_the_ocean_alone_and_raises_a_pocket() {
        let (mut state, q, wall) = pocket_plateau();
        let n = state.node_count();
        let r = state.route();
        assert!(r.sea_seeded);
        assert_eq!(r.outlets, 9);
        assert_eq!((r.pockets, r.pocket_nodes), (1, 2));
        assert_eq!((r.undrained, r.cyclic), (0, 0));
        for node in [q, wall] {
            assert!(!state.ocean[node]);
            assert_ne!(state.receiver[node], NO_NODE, "a pocket node is no outlet");
            assert_eq!(state.z_flood[node], Z_STEPS_PER_M, "raised to its spill");
            assert_eq!(state.water_level(node as u32), DRY);
            assert_eq!(state.sweep_base(node as u32), Z_STEPS_PER_M);
        }
        assert_eq!(state.pit[q] as i32, Z_STEPS_PER_M + 50);
        let facies = state.facies(&vec![false; n], 0, true);
        assert_eq!(facies[q] & FACIES_SEA, 0);
        let sea = state.lattice.index(Face::PosX, 3, 3) as usize;
        assert_eq!(facies[sea] & FACIES_SEA, FACIES_SEA);
        assert_eq!(state.water_level(sea as u32), 0);
        let ocean_area: u64 = (0..n)
            .filter(|&k| state.ocean[k])
            .map(|k| state.area[k])
            .sum();
        let all: u64 = state.area.iter().sum();
        assert!((state.ocean_share() - ocean_area as f64 / all as f64).abs() < 1e-15);
        assert_eq!((0..n).filter(|&k| state.ocean[k]).count(), 9);
        // The pocket is land to the basins' integrals too: its basin counts every node that is not
        // the ocean, the two under the sea's level included, and its floor is the pocket's.
        let basins = state.basins();
        let basin = basins[q];
        let land = (0..n)
            .filter(|&k| basins[k] == basin && !state.ocean[k])
            .count() as u32;
        let integral = state
            .hypsometric_integrals(1)
            .into_iter()
            .find(|b| b.outlet == basin)
            .expect("the pocket's basin has an integral");
        assert_eq!(integral.nodes, land);
        assert!(
            integral.integral_q8 > 200,
            "a plateau over a deep pocket: {integral:?}"
        );
        // The ice reads it as no ground for ice: it stands under the flood's fill.
        let line = vec![i32::MIN; n];
        let (mask, _, _) = state.ice_mask(&line, &vec![2_000; n], 6_500, 330);
        assert!(!mask[q]);
        assert!(!mask[sea]);
    }

    /// ★ THE LAKES' SEA FIXED POINT WITH NO LAKE (plan question 8): a wet body whose ground yields no
    /// runoff — the airless moon's own words, stated with a sea — holds no lake, so the sea over the
    /// inventory less the lakes is the sea itself: no re-solve, the fixed point stood still at once,
    /// and no "sea before the lakes" is reported.
    #[test]
    fn a_wet_body_with_no_lake_keeps_its_sea_through_the_fixed_point() {
        let moon = home_moon();
        let words = SolveWords {
            water_km3: 3_000_000,
            ..crate::home::home_moon_solve_words()
        };
        let (state, _, report) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        assert_eq!(report.lakes.volume_m3, 0);
        assert_eq!(report.lake_sea_solves, 0);
        assert!(report.lake_sea_converged);
        assert_eq!(report.sea_before_lakes, None);
        assert_eq!(report.sea_z, Some(state.sea_z));
        assert!(!report.sea_reseeded);
        assert!(!report.sea_unsettled);
    }

    /// The moon's lattice with a sea over face `PosX` (−1 000 m, the rest at 500 m) and the sea the
    /// inventory `water_km3` puts over the connected ocean, routed.
    fn sea_face(water_km3: u64) -> MacroSolve {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        for k in 0..n {
            state.z[k] = if state.lattice.split(k as u32).0 == Face::PosX {
                -1_000 * Z_STEPS_PER_M
            } else {
                500 * Z_STEPS_PER_M
            };
        }
        let lattice = state.lattice;
        state.sea_z =
            crate::land::sea_level_connected(&lattice, &state.z, &state.area, water_km3, false)
                .expect("a sea")
                .level;
        state.route();
        state
    }

    /// ★ THE LAKES' SEA FIXED POINT WITH A LAKE THAT MOVES THE SEA (review finding 7 of plan step
    /// 2). Four statements, each of which could fail, on the moon's sea face with the lakes' volume
    /// stated (a pocket's lake in a dry interior holding a tenth of the inventory):
    ///
    /// 1. the sea is re-solved ONCE and then stands still: one solve, converged, one more route, the
    ///    first sea kept, and the final sea exactly the connected sea over the inventory less the
    ///    lakes — at least one sixteenth under the first;
    /// 2. a bound of one stops the loop after the sea moved: one solve, NOT converged;
    /// 3. lakes that hold the whole inventory leave no sea to solve: the level stands, converged,
    ///    no solve;
    /// 4. the lakes read again after each new sea are the ones the loop solves against: a volume
    ///    that grows at each read keeps the sea moving until the bound stops it.
    #[test]
    fn the_lakes_sea_fixed_point_re_solves_until_the_sea_stands_still() {
        let water_km3 = 400_000;
        let lakes_m3 = u128::from(water_km3 / 10) * LAKE_KM3_M3;
        let mut state = sea_face(water_km3);
        let first = state.sea_z;
        let mut routes = Vec::new();
        let mut reads = 0u32;
        let fix = settle_lakes_sea(
            &mut state,
            water_km3,
            LAKE_SEA_SOLVES,
            lakes_m3,
            &mut routes,
            &mut |_| {
                reads += 1;
                lakes_m3
            },
        );
        assert_eq!(fix.first, first);
        assert_eq!((fix.solves, fix.converged), (1, true));
        assert_eq!((reads, routes.len()), (1, 1));
        assert!(!fix.reseeded);
        assert!(!fix.unsettled);
        let lattice = state.lattice;
        let want = crate::land::sea_level_connected(
            &lattice,
            &state.z,
            &state.area,
            water_km3 - water_km3 / 10,
            false,
        )
        .expect("a sea")
        .level;
        assert_eq!(state.sea_z, want);
        assert!(
            state.sea_z < first,
            "the sea fell: {first} to {}",
            state.sea_z
        );
        // 2. A bound of one.
        let mut bounded = sea_face(water_km3);
        let fix = settle_lakes_sea(
            &mut bounded,
            water_km3,
            1,
            lakes_m3,
            &mut Vec::new(),
            &mut |_| lakes_m3,
        );
        assert_eq!((fix.solves, fix.converged), (1, false));
        assert_eq!(bounded.sea_z, want);
        // 3. Lakes that hold the whole inventory.
        let mut drained = sea_face(water_km3);
        let all = u128::from(water_km3) * LAKE_KM3_M3;
        let fix = settle_lakes_sea(
            &mut drained,
            water_km3,
            LAKE_SEA_SOLVES,
            all,
            &mut Vec::new(),
            &mut |_| all,
        );
        assert_eq!((fix.solves, fix.converged), (0, true));
        assert_eq!(drained.sea_z, first);
        // 4. A volume that grows at every read: the sea never stands still inside the bound.
        let mut growing = sea_face(water_km3);
        let mut volume = lakes_m3;
        let fix = settle_lakes_sea(
            &mut growing,
            water_km3,
            LAKE_SEA_SOLVES,
            lakes_m3,
            &mut Vec::new(),
            &mut |_| {
                volume += lakes_m3;
                volume
            },
        );
        assert_eq!((fix.solves, fix.converged), (LAKE_SEA_SOLVES, false));
    }

    /// The node classes of the water's own tree over a solved state and its budget: a node of a
    /// spilling lake's hollow, the spill of one, or under a closed lake's water.
    fn tree_classes(
        state: &MacroSolve,
        lakes: &crate::lakes::Lakes,
    ) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
        let n = state.node_count();
        let body = |i: usize| -> u32 {
            let h = lakes.home[i];
            if (h == crate::lakes::NO_DEP) | (h == crate::lakes::OCEAN) {
                crate::lakes::NO_DEP
            } else {
                lakes.body_of[h as usize]
            }
        };
        let mut region = vec![false; n];
        let mut spill = vec![false; n];
        let mut closed = vec![false; n];
        for i in 0..n {
            let b = body(i);
            let full = (b != crate::lakes::NO_DEP) && lakes.deps[b as usize].full;
            region[i] = full;
            closed[i] = !full & (lakes.water[i] != DRY);
        }
        for (d, dep) in lakes.deps.iter().enumerate().skip(1) {
            if (lakes.body_of[d] == d as u32) & dep.full {
                spill[dep.spill_node as usize] = true;
            }
        }
        (region, spill, closed)
    }

    /// The five statements of test T10 over a state whose receivers, order and row the water's own tree set, and the
    /// budget and the climate over that state's field: returns how many nodes stood in a spilling lake's hollow, at a
    /// spill, and under a closed lake's water.
    fn check_the_waters_tree(
        state: &MacroSolve,
        lakes: &crate::lakes::Lakes,
        runoff_mm_yr: &[u32],
    ) -> (usize, usize, usize) {
        let n = state.node_count();
        // 1. The row again, with the sinks.
        let mut sink = vec![0u64; n];
        for &(node, water) in &state.spill_sinks {
            sink[node as usize] += water;
        }
        let mut again: Vec<u64> = (0..n)
            .map(|i| u64::from(runoff_mm_yr[i]) * state.area[i])
            .collect();
        for &node in &state.order {
            let i = node as usize;
            assert!(again[i] >= sink[i], "a sink is short at node {i}");
            again[i] -= sink[i];
            let r = state.receiver[i];
            if r != NO_NODE {
                again[r as usize] += again[i];
            }
        }
        assert_eq!(again, state.discharge);
        // 2. The land's row into the ocean is the cascade's.
        let (mut to_sea, mut ends) = (0u128, 0u128);
        for i in 0..n {
            if state.ocean[i] {
                continue;
            }
            let r = state.receiver[i];
            if r == NO_NODE {
                ends += u128::from(state.discharge[i]);
            } else if state.ocean[r as usize] {
                to_sea += u128::from(state.discharge[i]);
            }
        }
        assert_eq!(to_sea, lakes.report.to_ocean);
        // 3. The ends and the sinks hold what the budget holds.
        let sinks: u128 = state.spill_sinks.iter().map(|&(_, w)| u128::from(w)).sum();
        assert_eq!(ends + sinks, lakes.report.evaporated_total);
        // 4. No floor rises, no ring.
        let floor = |i: usize| -> i32 {
            let w = state.water_level(i as u32);
            if w == DRY {
                state.z[i]
            } else {
                state.z[i].max(w)
            }
        };
        let mut steps = 0usize;
        for i in 0..n {
            let r = state.receiver[i];
            if r == NO_NODE {
                continue;
            }
            steps += 1;
            assert!(
                floor(r as usize) <= floor(i),
                "the floor rises from {i} to {r}"
            );
            assert!(
                state.lattice.neighbours(i as u32).contains(&r),
                "{r} is no neighbour of {i}"
            );
            assert_eq!(state.chord[i], state.chord_m(i as u32, r));
        }
        let land = state.ocean.iter().filter(|&&o| !o).count();
        assert!(steps > land / 2, "{steps} steps over {land} land nodes");
        assert_eq!(state.order.len(), n);
        // 5. The dry ground takes the descent.
        let (region, spill, closed) = tree_classes(state, lakes);
        let mut dry = 0usize;
        for i in 0..n {
            if region[i] | spill[i] | closed[i] {
                continue;
            }
            dry += 1;
            assert_eq!(state.receiver[i], lakes.descent.down[i], "node {i}");
        }
        assert!(dry > land / 2, "{dry} dry nodes of {land}");
        let count = |v: &[bool]| v.iter().filter(|&&b| b).count();
        (count(&region), count(&spill), count(&closed))
    }

    /// ★ TEST T10 (plan step 2b; ruling B8 item 2; replaces step 2's test 19): THE SHIPPED ROW IS
    /// THE BUDGET'S OWN WATER over THE WATER'S OWN TREE ([`check_the_waters_tree`]'s five statements):
    ///
    /// 1. the runoff accumulated again over the shipped receivers, less the stated sinks at their
    ///    spill nodes, is the discharge row EXACTLY, at every node, and no sink is short;
    /// 2. the row that reaches the ocean from the land is the cascade's `to_ocean` to the unit;
    /// 3. the rows at every end on the land plus every sink is the water the budget holds;
    /// 4. along every receiver the FLOOR (the ground, or the water's surface where water stands)
    ///    never rises, and no receiver is left out of the order;
    /// 5. every node that is not a spilling lake's, a spill, or under a closed lake's water takes the
    ///    budget's own descent on the ground, and every receiver is a neighbour.
    ///
    /// First over the full solve on the moon's lattice under the home planet's words (a sea over part of the moon),
    /// with the budget and the climate over the final field read again — ★ since the deposit's cure that field keeps a
    /// few hollows (MEASURED: 24; before it none, ruling B51 item 2); then over the same field with two hollows dug into it — a
    /// stepped bowl at a summit, whose lake stands where the air takes what falls in (a closed lake), and a pit in a
    /// trunk river, which fills and spills — re-routed, so every class of the tree is met.
    ///
    /// Before step 2b the row carried the budget's held water to the sea and the floor rose where
    /// the routing fill crossed a parent's dry ring.
    #[test]
    fn the_shipped_row_is_the_budgets_own_water_over_the_waters_own_tree() {
        let moon = home_moon();
        let words = SolveWords {
            water_km3: 3_000_000,
            ..crate::home::home_solve_words()
        };
        let (mut state, _, report) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let lattice = moon.macro_lattice().expect("a lattice");
        let climate =
            crate::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let lakes = crate::lakes::budget(&state, &climate);
        assert_eq!(lakes.report, report.lakes, "the budget is the solve's own");
        assert_eq!((report.tree.cyclic, report.tree.shortfall), (0, 0));
        assert_eq!(report.tree.exit_disagree, 0);
        // ★ Since the deposit's cure (2026-10-04: a hollow full of rock passes the rest on) the solved field keeps a few
        // hollows (MEASURED: 24 depressions; before the cure none); the tree over them meets the same five rules.
        let (region, spill, closed) = check_the_waters_tree(&state, &lakes, &climate.runoff_mm_yr);
        assert_eq!(
            lakes.report.depressions == 0,
            region + spill + closed == 0,
            "a class of the tree only where a hollow stands"
        );
        // Two hollows dug: a bowl at a semi-arid summit and a pit in the land's largest river away from the sea.
        let (climate, lakes) = dug::two_hollows(&moon, &words, &mut state);
        assert!(
            lakes.report.spilling > 0,
            "a lake spills: {:?}",
            lakes.report
        );
        assert!(
            lakes.report.partial > 0,
            "a lake is closed: {:?}",
            lakes.report
        );
        let tree = crate::lakes::water_tree(&state, &lakes);
        let followed = state.follow_the_water(tree, &climate.runoff_mm_yr);
        assert_eq!((followed.cyclic, followed.shortfall), (0, 0));
        assert!(!state.spill_sinks.is_empty());
        let (region, spill, closed) = check_the_waters_tree(&state, &lakes, &climate.runoff_mm_yr);
        assert!(region > 0, "a spilling lake's hollow");
        assert!(spill > 0, "a spill");
        assert!(closed > 0, "a closed lake");
    }

    /// ★ TEST T11 (plan step 2b): THE TREE MOVES NO HEIGHT AND NO WATER. The solved state is
    /// cloned, the water's tree is built and followed again on the clone, and the ground, the fill,
    /// the pits, the water, the ocean, the sea and the rock are the state's own; the receivers and
    /// the row come out the same, because the tree is a function of the state and its budget.
    #[test]
    fn following_the_water_moves_no_height_and_no_water() {
        let moon = home_moon();
        let words = SolveWords {
            water_km3: 3_000_000,
            ..crate::home::home_solve_words()
        };
        let (state, _, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let lattice = moon.macro_lattice().expect("a lattice");
        let climate =
            crate::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let lakes = crate::lakes::budget(&state, &climate);
        let mut again = state.clone();
        again.receiver = vec![NO_NODE; again.node_count()];
        again.discharge = vec![0; again.node_count()];
        let tree = crate::lakes::water_tree(&again, &lakes);
        let report = again.follow_the_water(tree, &climate.runoff_mm_yr);
        assert_eq!(report.cyclic, 0);
        assert_eq!(again.z, state.z);
        assert_eq!(again.z_flood, state.z_flood);
        assert_eq!(again.pit, state.pit);
        assert_eq!(again.water_z, state.water_z);
        assert_eq!(again.ocean, state.ocean);
        assert_eq!(again.sea_z, state.sea_z);
        assert_eq!(again.province, state.province);
        assert_eq!(again.receiver, state.receiver);
        assert_eq!(again.discharge, state.discharge);
        assert_eq!(again.spill_sinks, state.spill_sinks);
    }

    /// ★ TEST T12 (plan step 2b, question 8): ON A BODY WITH NO WATER THE WATER'S WAY IS THE
    /// GROUND'S OWN DESCENT, and every root and every pit ends it: the airless moon's receivers are
    /// the descent's, and its rows carry nothing (no rain, no runoff).
    #[test]
    fn a_dry_bodys_receivers_are_the_grounds_descent() {
        let moon = home_moon();
        let words = crate::home::home_moon_solve_words();
        let (state, _, report) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        assert_eq!(state.receiver, crate::lakes::descent(&state).down);
        assert!(state.spill_sinks.is_empty());
        assert_eq!(report.tree.cyclic, 0);
        assert!(state.discharge.iter().all(|&d| d == 0));
        assert_eq!(state.order.len(), state.node_count());
    }

    /// ★ THE UNDRAINED COUNTER on a stated distance field: a node the field never reached is counted
    /// and keeps no receiver; a reached node skips an unreached neighbour and takes the reached one.
    #[test]
    fn an_unreached_node_is_counted_undrained_and_skipped_as_a_neighbour() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z_flood = vec![0; n];
        let mut dist = vec![u64::MAX; n];
        let a = state.lattice.index(Face::PosX, 20, 20);
        let ring = state.lattice.neighbours(a);
        // One routed neighbour at distance 0, the rest unreached; `a` itself a chord away.
        dist[ring[7] as usize] = 0;
        dist[a as usize] = u64::from(state.chord_m(a, ring[7]));
        let (flat, undrained) = state.assign_flat_receivers(&dist);
        assert_eq!(flat, 1);
        assert_eq!(undrained, n - 2);
        assert_eq!(state.receiver[a as usize], ring[7]);
        assert_eq!(state.receiver[ring[0] as usize], NO_NODE);
    }

    /// ★ THE FLAT'S RECEIVER TAKES THE NEARER NEIGHBOUR, NOT THE SMALLER INDEX (ruling B2 step 3,
    /// 2026-09-22). On the plateau every routed neighbour of the outlet stands at distance zero, so
    /// a flat node touching three of them chose the smallest INDEX before — the stencil's own
    /// `(-1, -1)` corner, one fixed diagonal planet-wide — and now chooses the one its chord is
    /// SHORTEST to, which is the row neighbour, because a diagonal is the root of two of a row.
    #[test]
    fn a_flat_node_between_three_routed_nodes_takes_the_nearest_and_never_the_smaller_index() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![Z_STEPS_PER_M; n];
        state.sea_z = 0;
        let outlet = state.lattice.index(Face::PosX, 10, 10);
        state.z[outlet as usize] = 0;
        state.route();
        // (12, 10) touches (11, 9), (11, 10) and (11, 11), all routed to the outlet at distance 0.
        let node = state.lattice.index(Face::PosX, 12, 10);
        let row = state.lattice.index(Face::PosX, 11, 10);
        let diagonal = state.lattice.index(Face::PosX, 11, 9);
        assert!(diagonal < row, "the diagonal holds the smaller index");
        assert!(
            state.chord_m(node, row) < state.chord_m(node, diagonal),
            "the row is the shorter chord"
        );
        assert_eq!(state.receiver[node as usize], row);
        assert_eq!(state.chord[node as usize], state.chord_m(node, row));
    }

    /// ★ A FIELD WITH NO FLAT ROUTES EXACTLY AS IT DID (ruling B2 step 3): the cure touches the
    /// flats and nothing else. The cone is a stated field with one low point, so the routing reports
    /// no flat at all — and every receiver is the steepest lower neighbour, derived here a second
    /// time and compared.
    #[test]
    fn a_field_with_no_flat_keeps_the_steepest_descent_it_always_had() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        // ★ THE CONE WITHOUT A FLOAT (the fence, SL10 clause 4): the dot of two unit directions
        // FALLS as the angle between them grows, so the negated dot is already a cone. The shift
        // brings the product of two 40-bit directions back into a signed word and still leaves
        // about a thousand million levels — far finer than twenty-eight thousand nodes can tie on.
        const DOT_SHIFT: u32 = 2 * vd_recipe::bend::DIR_BITS - 30;
        let raw = |node: u32| -> [i128; 3] {
            let d = state.lattice.direction(node);
            [
                i128::from(d[0].raw()),
                i128::from(d[1].raw()),
                i128::from(d[2].raw()),
            ]
        };
        let apex = raw(0);
        let cone: Vec<i32> = (0..n as u32)
            .map(|node| {
                let v = raw(node);
                let dot = v[0] * apex[0] + v[1] * apex[1] + v[2] * apex[2];
                -((dot >> DOT_SHIFT) as i32)
            })
            .collect();
        state.z = cone;
        let report = state.route();
        assert_eq!(report.flat, 0, "a cone holds no flat");
        assert_eq!(report.undrained, 0);
        assert_eq!(report.raised, 0, "and the flood raises nothing");
        for node in 0..n as u32 {
            let i = node as usize;
            if state.receiver[i] == NO_NODE {
                continue;
            }
            let (mut best, mut dz_best, mut l_best) = (NO_NODE, 0i64, 0u32);
            for m in state.lattice.neighbours(node) {
                if m == NO_NODE || state.z_flood[m as usize] >= state.z_flood[i] {
                    continue;
                }
                let dz = i64::from(state.z_flood[i]) - i64::from(state.z_flood[m as usize]);
                let l = state.chord_m(node, m);
                if best == NO_NODE || steeper(dz, l, m, dz_best, l_best, best) {
                    best = m;
                    dz_best = dz;
                    l_best = l;
                }
            }
            assert_eq!(state.receiver[i], best, "node {node}");
            assert_eq!(state.chord[i], l_best, "node {node}");
        }
    }

    /// ★ THE JOINT LAW'S ROOT: no time leaves the height; at `n = 1` the root is the linear scheme's own `x₀/(1 + c/L)`;
    /// at the law's `n = 2.2` the root solves its equation to the tolerance, stands under `x₀` and over nothing, and a
    /// larger `c` cuts deeper.
    #[test]
    fn the_joint_root_solves_the_implicit_step() {
        let tol = Gf::from_f64(JOINT_TOLERANCE_M);
        let (x0, l) = (Gf::from_i64(100), Gf::from_i64(8_192));
        assert_eq!(joint_root(x0, Gf::ZERO, l, Gf::from_f64(JOINT_N), tol), x0);
        let linear = joint_root(x0, Gf::from_i64(8_192), l, Gf::ONE, tol);
        assert!((linear.to_f64() - 50.0).abs() < 1e-6, "{}", linear.to_f64());
        let n = Gf::from_f64(JOINT_N);
        let mut last = x0;
        for c in [1.0e3, 1.0e6, 1.0e9] {
            let c = Gf::from_f64(c);
            let x = joint_root(x0, c, l, n, tol);
            let g = x - x0 + c * (x / l).powf(n);
            assert!(g.abs() < Gf::from_f64(1e-6), "{}", g.to_f64());
            assert!(x < last, "{} under {}", x.to_f64(), last.to_f64());
            assert!(x > Gf::ZERO);
            last = x;
        }
    }

    /// ★ THE LANDSCAPE'S TIME: a plate whose age share is nothing was raised now; a half share 680 million years ago
    /// (`ln 2 / 1.019` Gyr); the largest share the draw can give is older than the system and so as old as the body; a
    /// body whose plates turn nothing over keeps every plate at the system's age. A node's share of a pass is none
    /// before its erosion's start, all of it after, and the exact part in between.
    #[test]
    fn the_landscapes_time_raises_each_plate_and_shares_each_pass() {
        let plate = |age: Gf| crate::land::Plate {
            site: [Gf::ONE, Gf::ZERO, Gf::ZERO],
            drift: [Gf::ZERO; 3],
            affinity: Gf::ZERO,
            age,
            crust_scatter: Gf::ONE,
        };
        let top = Gf::from_i64((1i64 << 53) - 1) / Gf::from_i64(1i64 << 53);
        let plates = [plate(Gf::ZERO), plate(Gf::HALF), plate(top)];
        let age = HOME_SYSTEM_AGE_YR;
        let t = raise_times(&plates, age, true);
        assert_eq!(t[0], 0);
        let half = t[1];
        assert!((half as f64 - 0.693_147e9 / 1.019).abs() < 1.0e3, "{half}");
        assert_eq!(t[2], age);
        assert_eq!(raise_times(&plates, age, false), vec![age; 3]);
        assert_eq!(active_share(500, 100, 500), 0);
        assert_eq!(active_share(100, 100, 500), ACTIVE_FULL);
        assert_eq!(active_share(400, 100, 500), ACTIVE_FULL / 4);
    }

    /// ★ THE LEVEL FILL: the doc's hollow (0, 10 and 20 sixteenths, 1 m² each) takes 15 and stands at 12; no node holds
    /// the spill; more rock than the hollow holds stops at the spill; a remainder under one sixteenth of the raised area
    /// stays unlaid; a node over the spill counts at the spill.
    #[test]
    fn the_level_fill_raises_the_lowest_nodes_to_one_level_under_the_spill() {
        let hollow = [(0, 1u64), (10, 1), (20, 1)];
        assert_eq!(level_fill(&hollow, 15, 100), 12);
        assert_eq!(level_fill(&[], 15, 100), 100);
        assert_eq!(level_fill(&hollow, 1_000, 30), 30);
        assert_eq!(level_fill(&hollow, 16, 100), 13);
        assert_eq!(level_fill(&hollow, 17, 100), 13);
        assert_eq!(level_fill(&[(0, 2), (40, 1)], 10, 30), 5);
        assert_eq!(level_fill(&[(0, 1), (40, 1)], 35, 30), 30);
        assert_eq!(level_fill(&hollow, 0, 100), 0);
    }

    /// ★ THE BELTS ON THE LAND: a node where a wedge rises loses its even uplift, and off its plateau it reads the folded
    /// belt's rock while a plateau node keeps its own; each node's erosion starts at its plate's raise, or at its belt's
    /// window where that opened later.
    #[test]
    fn the_belts_take_the_land_and_open_the_landscapes_time() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let mut land =
            crate::land::initial_land(&moon, &lattice, &crate::home::home_moon_land_words());
        let n = lattice.node_count();
        land.uplift = vec![100; n];
        let mut belts = crate::belts::Belts {
            total: vec![0; n],
            start_yr: vec![HOME_SYSTEM_AGE_YR; n],
            kind: vec![crate::belts::BELT_NONE; n],
            plateau: vec![false; n],
            wedge: Vec::new(),
            report: crate::belts::BeltReport::default(),
        };
        belts.total[1] = 50;
        belts.total[2] = 70;
        belts.plateau[2] = true;
        belts.start_yr[1] = HOME_SYSTEM_AGE_YR - 1_000;
        belts.start_yr[2] = 10;
        let province_2 = land.province[2];
        let even = belt_land(&belts, &mut land);
        assert_eq!((even[0], even[1], even[2]), (100, 0, 0));
        assert_eq!(land.province[1], crate::strata::Province::FoldedBelt.code());
        assert_eq!(land.province[2], province_2);
        let raise = vec![2_000u64; land.plates.len()];
        let start = erosion_starts(&belts, &raise, &land.plate, HOME_SYSTEM_AGE_YR);
        let raised = HOME_SYSTEM_AGE_YR - 2_000;
        assert_eq!(start[0], raised);
        assert_eq!(start[1], raised);
        assert_eq!(start[2], 10);
    }

    /// ★ THE DEPOSIT WITH THE BUDGET'S CLOSED HOLLOWS (ruling B51 item 2): on the cone a closed hollow of three nodes ends
    /// its water at its dry floor; the rock cut upstream reaches that end, stays in the hollow, and fills it from its
    /// floor up to one level under its spill; a hollow that holds no rock is left alone; and the rock past the
    /// hollows runs on to the sea.
    #[test]
    fn a_closed_hollow_keeps_its_rock_and_fills_from_its_floor() {
        let (mut state, _) = cone();
        let floor = state.lattice.index(Face::PosX, 30, 30);
        let a = state.lattice.index(Face::PosX, 31, 30);
        let b = state.lattice.index(Face::PosX, 30, 31);
        state.z[floor as usize] = Z_STEPS_PER_M * 90;
        state.z[a as usize] = Z_STEPS_PER_M * 95;
        state.z[b as usize] = Z_STEPS_PER_M * 100;
        state.route();
        // The water's own tree: the hollow's floor ends its river (a closed lake ends its rivers), its pit dry.
        state.receiver[floor as usize] = NO_NODE;
        for v in [floor, a, b] {
            state.pit[v as usize] = 0;
        }
        let spill = Z_STEPS_PER_M * 109;
        let far = state.lattice.index(Face::NegZ, 10, 10);
        let basins = Basins {
            slot: (0..state.node_count() as u32)
                .map(|v| {
                    if [floor, a, b].contains(&v) {
                        0
                    } else if v == far {
                        1
                    } else {
                        u32::MAX
                    }
                })
                .collect(),
            bodies: vec![(vec![floor, a, b], spill), (vec![far], spill)],
            ..Basins::default()
        };
        let above = state.lattice.index(Face::PosX, 31, 31);
        let up = state.receiver[above as usize];
        assert!(
            [floor, a, b].contains(&up),
            "the cut drains into the hollow"
        );
        let beside = state.lattice.index(Face::PosX, 21, 20);
        state.cut_now[above as usize] = (Z_STEPS_PER_M * 20) as u32;
        state.cut_now[beside as usize] = Z_STEPS_PER_M as u32;
        let held =
            u128::from(state.cut_now[above as usize]) * u128::from(state.area[above as usize]);
        let mut heights: Vec<(i32, u64)> = [floor, a, b]
            .iter()
            .map(|&v| (state.z[v as usize], state.area[v as usize]))
            .collect();
        heights.sort_unstable();
        let level = level_fill(&heights, held, spill);
        let far_before = state.z[far as usize];
        let before = state.z.clone();
        let d = state.deposit_basins(&basins);
        assert!(level > Z_STEPS_PER_M * 95, "{level}");
        assert!(level < spill);
        for v in [floor, a, b] {
            let z = state.z[v as usize];
            assert!((z == level) | (z > level), "{v}: {z} against {level}");
        }
        assert_eq!(state.z[floor as usize], level);
        assert!(d.filled >= 2);
        assert!(d.total_fill > 0);
        assert_eq!(
            state.z[far as usize], far_before,
            "a hollow with no rock is left alone"
        );
        // Nothing passed a spill and nothing was dropped: the hollow kept its rock, the rest of the cut reached the sea.
        assert_eq!((d.over_spill, d.dropped), (0, 0));
        assert_eq!(
            d.to_sea,
            u128::from(state.cut_now[beside as usize]) * u128::from(state.area[beside as usize])
        );
        assert_eq!(
            rock_cut(&state),
            rock_laid(&state, &before) + d.to_sea + d.remainder
        );
    }

    /// The wet moon's solved field (the home planet's words with a sea over part of the moon), for the dug tests.
    fn dug_state() -> (MacroSolve, SolveWords, BodyDefinition) {
        let moon = home_moon();
        let words = SolveWords {
            water_km3: 3_000_000,
            ..crate::home::home_solve_words()
        };
        let (state, _, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        (state, words, moon)
    }

    /// The rock the sweep's cut stands for, sixteenths·m² (the deposit's tests).
    fn rock_cut(state: &MacroSolve) -> u128 {
        state
            .cut_now
            .iter()
            .zip(&state.area)
            .map(|(&c, &a)| u128::from(c) * u128::from(a.max(1)))
            .sum()
    }

    /// The rock the deposit laid since `before`, sixteenths·m² (the deposit's tests).
    fn rock_laid(state: &MacroSolve, before: &[i32]) -> u128 {
        state
            .z
            .iter()
            .zip(before)
            .zip(&state.area)
            .map(|((&z, &b), &a)| u128::from((z - b) as u32) * u128::from(a.max(1)))
            .sum()
    }

    /// The cone with the closed hollow of the deposit's first test dug in it and a second closed hollow beside it
    /// (each with a dry floor that ends its water), routed; returns the state, the first hollow's nodes, the second's
    /// floor, the node whose cut drains into the first, and a gate on the cone's flank whose way runs to the sea.
    /// ★ AN OUTLET NO RIVER TREE HOLDS CARRIES ITS OWN CUT (ruling B72 item 1): a node with no receiver and no donor
    /// lies in no tree the cores walk; its own cut still fills its own room, and the rest leaves by the tree's-end rule —
    /// into the sea from an ocean node, dropped and counted at an inland end no closed body holds.
    #[test]
    fn an_outlet_no_tree_holds_books_its_own_cut() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        state.order = order_of(&state.receiver);
        let n = state.node_count();
        let (sea, inland) = (3usize, 4usize);
        state.ocean[sea] = true;
        state.pit[inland] = 2;
        state.cut_now[sea] = 10;
        state.cut_now[inland] = 10;
        let (area_sea, area_inland) = (u128::from(state.area[sea]), u128::from(state.area[inland]));
        let z = state.z[inland];
        let d = state.deposit_basins(&Basins {
            slot: vec![u32::MAX; n],
            ..Basins::default()
        });
        assert_eq!(d.to_sea, 10 * area_sea);
        assert_eq!(d.dropped, 8 * area_inland);
        assert_eq!((d.filled, d.total_fill), (1, 2));
        assert_eq!((state.z[inland], state.pit[inland]), (z + 2, 0));
        assert_eq!(state.pending[inland], -2);
    }

    fn two_dug_hollows() -> (MacroSolve, [u32; 3], u32, u32, u32) {
        let (mut state, _) = cone();
        let floor = state.lattice.index(Face::PosX, 30, 30);
        let a = state.lattice.index(Face::PosX, 31, 30);
        let b = state.lattice.index(Face::PosX, 30, 31);
        let second = state.lattice.index(Face::PosX, 35, 20);
        state.z[floor as usize] = Z_STEPS_PER_M * 90;
        state.z[a as usize] = Z_STEPS_PER_M * 95;
        state.z[b as usize] = Z_STEPS_PER_M * 100;
        state.z[second as usize] = Z_STEPS_PER_M * 80;
        state.route();
        for v in [floor, second] {
            state.receiver[v as usize] = NO_NODE;
        }
        for v in [floor, a, b, second] {
            state.pit[v as usize] = 0;
        }
        let above = state.lattice.index(Face::PosX, 31, 31);
        let gate = state.lattice.index(Face::PosX, 27, 27);
        (state, [floor, a, b], second, above, gate)
    }

    /// The slots of the two dug hollows: the first hollow body 0, the second's floor body 1, a ring of two flank nodes
    /// body 2.
    fn dug_slots(state: &MacroSolve, first: [u32; 3], second: u32, ring: [u32; 2]) -> Vec<u32> {
        (0..state.node_count() as u32)
            .map(|v| {
                if first.contains(&v) {
                    0
                } else if v == second {
                    1
                } else if ring.contains(&v) {
                    2
                } else {
                    u32::MAX
                }
            })
            .collect()
    }

    /// ★ A CUT SADDLE LETS A CLOSED HOLLOW GO IN THE SAME SWEEP (ruling B62 item 2): the first hollow is a closed body
    /// whose lake stands at 105 m under its 109 m spill; a river has cut its saddle to 92 m. The sweep drains the lake to
    /// the saddle before it cuts (the floor keeps 2 m of water, the two higher nodes stand dry), and the deposit fills
    /// the hollow only to the saddle and passes the rest to the sea — no rock lost. Before the cure the lake stood and the
    /// hollow kept the rock until the next routing.
    #[test]
    fn a_cut_saddle_lets_a_closed_hollow_go_in_the_same_sweep() {
        let (mut state, first, second, above, gate) = two_dug_hollows();
        let spill = Z_STEPS_PER_M * 109;
        let rim = state.lattice.index(Face::PosX, 32, 30);
        state.z[rim as usize] = Z_STEPS_PER_M * 92;
        for v in first {
            let i = v as usize;
            state.z_flood[i] = Z_STEPS_PER_M * 105;
            state.pit[i] = (state.z_flood[i] - state.z[i]) as u32;
        }
        let basins = Basins {
            slot: dug_slots(&state, first, second, [NO_NODE, NO_NODE]),
            bodies: vec![(first.to_vec(), spill), (vec![second], spill)],
            sides: vec![vec![(OVER_TO_SEA, gate)], Vec::new()],
            parent: vec![u32::MAX, u32::MAX],
            parent_gate: vec![rim, NO_NODE],
        };
        let report = state.sweep(Gf::ZERO, None, Some(&basins));
        assert_eq!(
            report.drained, 3,
            "every node of the lake drains to the saddle"
        );
        let floor = first[0] as usize;
        assert_eq!(state.z_flood[floor], Z_STEPS_PER_M * 92);
        assert_eq!(state.pit[floor], (Z_STEPS_PER_M * 2) as u32);
        for v in &first[1..] {
            assert_eq!(state.pit[*v as usize], 0, "the higher nodes stand dry");
        }
        state.cut_now[above as usize] = (Z_STEPS_PER_M * 200) as u32;
        let cut =
            u128::from(state.cut_now[above as usize]) * u128::from(state.area[above as usize]);
        let before = state.z.clone();
        let d = state.deposit_basins(&basins);
        assert_eq!(
            state.z[floor],
            Z_STEPS_PER_M * 92,
            "filled to the saddle, never the old spill"
        );
        assert!(d.over_spill > 0);
        assert!(d.to_sea > 0);
        assert_eq!(rock_laid(&state, &before) + d.to_sea + d.remainder, cut);
    }

    /// ★ A HOLLOW FULL OF ROCK PASSES THE REST ON (2026-10-04, the convergence gate's cause): the first hollow holds
    /// 42 m of rock over a node under its 109 m spill and its range sends 200 m; it fills to the spill, and the rest
    /// goes over its saddle to the ocean's side and down the cone to the sea. No rock is lost.
    #[test]
    fn a_hollow_full_of_rock_passes_the_rest_over_its_spill_to_the_sea() {
        let (mut state, first, second, above, gate) = two_dug_hollows();
        let spill = Z_STEPS_PER_M * 109;
        let basins = Basins {
            slot: dug_slots(&state, first, second, [NO_NODE, NO_NODE]),
            bodies: vec![(first.to_vec(), spill), (vec![second], spill)],
            sides: vec![vec![(OVER_TO_FULL, gate), (OVER_TO_SEA, gate)], Vec::new()],
            parent: vec![u32::MAX, u32::MAX],
            parent_gate: vec![NO_NODE, NO_NODE],
        };
        state.cut_now[above as usize] = (Z_STEPS_PER_M * 200) as u32;
        let before = state.z.clone();
        let d = state.deposit_basins(&basins);
        for v in first {
            assert_eq!(
                state.z[v as usize], spill,
                "every node of the full hollow at its spill"
            );
        }
        assert!(d.over_spill > 0);
        assert_eq!(
            d.to_sea, d.over_spill,
            "the rest runs down the cone to the sea"
        );
        assert_eq!(d.dropped, 0);
        assert_eq!(
            rock_cut(&state),
            rock_laid(&state, &before) + d.to_sea + d.remainder
        );
    }

    /// The rest goes into the next closed hollow its saddle leads to, which keeps it; and where that one fills too and
    /// every side is full of rock, the rock rises into the parent's ring, which fills as one body with its full child.
    #[test]
    fn the_rest_fills_the_next_hollow_and_then_the_parents_ring() {
        let (mut state, first, second, above, _) = two_dug_hollows();
        let spill = Z_STEPS_PER_M * 109;
        let ring = [
            state.lattice.index(Face::PosX, 36, 20),
            state.lattice.index(Face::PosX, 37, 20),
        ];
        let ring_spill = Z_STEPS_PER_M * 140;
        let basins = |state: &MacroSolve| Basins {
            slot: dug_slots(state, first, second, ring),
            bodies: vec![
                (first.to_vec(), spill),
                (vec![second], spill),
                (ring.to_vec(), ring_spill),
            ],
            sides: vec![vec![(1, second)], vec![(0, first[0])], Vec::new()],
            parent: vec![2, 2, u32::MAX],
            parent_gate: vec![NO_NODE, NO_NODE, NO_NODE],
        };
        // 60 m over the first hollow: 42 m fill it, the rest fills the second from its floor (80 m to about 98 m).
        state.cut_now[above as usize] = (Z_STEPS_PER_M * 60) as u32;
        let floor2 = state.z[second as usize];
        let before = state.z.clone();
        let first_round = basins(&state);
        let d = state.deposit_basins(&first_round);
        assert!(
            state.z[second as usize] > floor2,
            "the next hollow kept the rest"
        );
        assert!(state.z[second as usize] < spill);
        assert!(d.over_spill > 0);
        assert_eq!((d.to_sea, d.dropped), (0, 0));
        assert_eq!(rock_cut(&state), rock_laid(&state, &before) + d.remainder);
        // 60 m again: the first hollow is full, the second fills to its spill, and every side being full the rest rises
        // into the parent's ring — the ring's two flank nodes and the second's floor at one level under the ring's spill.
        let before = state.z.clone();
        let second_round = basins(&state);
        let d = state.deposit_basins(&second_round);
        let level = state.z[second as usize];
        assert_eq!(state.z[ring[0] as usize], level);
        assert_eq!(state.z[ring[1] as usize], level);
        assert!(
            level > spill,
            "the full child filled with its parent's ring: {level}"
        );
        assert!(level < ring_spill);
        assert_eq!((d.to_sea, d.dropped), (0, 0));
        assert_eq!(rock_cut(&state), rock_laid(&state, &before) + d.remainder);
    }

    /// Where every side is full and the parent's ring is no closed body, the rest leaves by the parent's own exit; and
    /// where there is no way at all, or the way runs in a ring of receivers, it is dropped and counted.
    #[test]
    fn the_rest_leaves_by_the_parents_exit_or_is_dropped_and_counted() {
        let (mut state, first, second, above, gate) = two_dug_hollows();
        let spill = Z_STEPS_PER_M * 109;
        let slot = dug_slots(&state, first, NO_NODE, [NO_NODE, NO_NODE]);
        assert_ne!(second, NO_NODE);
        state.cut_now[above as usize] = (Z_STEPS_PER_M * 100) as u32;
        let by_exit = Basins {
            slot: slot.clone(),
            bodies: vec![(first.to_vec(), spill)],
            sides: vec![vec![(OVER_TO_FULL, gate)]],
            parent: vec![u32::MAX],
            parent_gate: vec![gate],
        };
        let before = state.z.clone();
        let d = state.deposit_basins(&by_exit);
        assert!(d.over_spill > 0);
        assert_eq!((d.to_sea, d.dropped, d.refused), (d.over_spill, 0, 0));
        assert_eq!(
            rock_cut(&state),
            rock_laid(&state, &before) + d.to_sea + d.remainder
        );
        // No way at all (a body that never met another): the rest is dropped and counted.
        let nowhere = Basins {
            slot: slot.clone(),
            bodies: vec![(first.to_vec(), spill)],
            ..Basins::default()
        };
        let d = state.deposit_basins(&nowhere);
        assert_eq!(d.dropped, d.over_spill);
        assert!(d.dropped > 0);
        // A ring of receivers past the spill: the way is refused, and the rest dropped and counted, never a hang.
        let (g1, g2) = (gate, state.receiver[gate as usize]);
        state.receiver[g2 as usize] = g1;
        // The order is the receivers' own (a routing makes both): the ring's two nodes leave it.
        state.order = order_of(&state.receiver);
        let ring = Basins {
            slot,
            bodies: vec![(first.to_vec(), spill)],
            sides: vec![vec![(OVER_TO_SEA, g1)]],
            ..Basins::default()
        };
        let d = state.deposit_basins(&ring);
        assert_eq!(
            (d.dropped, d.stuck, d.refused),
            (d.over_spill, d.over_spill, 1)
        );
        // An overflow whose way ends at an inland node no body holds: the way is refused, and with no other way the rest
        // is dropped and counted.
        let (mut state, first, _, above, gate) = two_dug_hollows();
        state.receiver[gate as usize] = NO_NODE;
        state.cut_now[above as usize] = (Z_STEPS_PER_M * 100) as u32;
        let d = state.deposit_basins(&Basins {
            slot: dug_slots(&state, first, NO_NODE, [NO_NODE, NO_NODE]),
            bodies: vec![(first.to_vec(), spill)],
            sides: vec![vec![(OVER_TO_SEA, gate)]],
            ..Basins::default()
        });
        assert!(d.over_spill > 0);
        assert_eq!(
            (d.dropped, d.stuck, d.refused),
            (d.over_spill, d.over_spill, 1)
        );
        // An inland end no body holds drops the rock that reaches it.
        let (mut state, first, _, _, _) = two_dug_hollows();
        let lone = state.lattice.index(Face::PosX, 25, 33);
        state.receiver[lone as usize] = NO_NODE;
        state.cut_now[lone as usize] = Z_STEPS_PER_M as u32;
        let d = state.deposit_basins(&Basins {
            slot: dug_slots(&state, first, NO_NODE, [NO_NODE, NO_NODE]),
            bodies: vec![(first.to_vec(), spill)],
            ..Basins::default()
        });
        assert_eq!(
            d.dropped,
            u128::from(state.cut_now[lone as usize]) * u128::from(state.area[lone as usize])
        );
    }

    /// ★ A WAY THAT RUNS BACK INTO A FULL HOLLOW IS REFUSED (2026-10-05: a run of the belts' bench spun for 90 minutes
    /// in the deposit's work list). The first hollow's first side is the ocean's, but the water's way from its gate runs
    /// back down into the hollow itself, which is then full of rock; the rest takes the next side, into the second
    /// hollow, which keeps it. Before the cure the rest went round that one side for ever.
    #[test]
    fn a_way_that_runs_back_into_a_full_hollow_is_refused() {
        let (mut state, first, second, above, _) = two_dug_hollows();
        let spill = Z_STEPS_PER_M * 109;
        let basins = Basins {
            slot: dug_slots(&state, first, second, [NO_NODE, NO_NODE]),
            bodies: vec![(first.to_vec(), spill), (vec![second], spill)],
            sides: vec![vec![(OVER_TO_SEA, above), (1, second)], Vec::new()],
            parent: vec![u32::MAX, u32::MAX],
            parent_gate: vec![NO_NODE, NO_NODE],
        };
        state.cut_now[above as usize] = (Z_STEPS_PER_M * 60) as u32;
        let floor2 = state.z[second as usize];
        let before = state.z.clone();
        let d = state.deposit_basins(&basins);
        assert!(d.over_spill > 0);
        assert_eq!(
            d.refused, 1,
            "the ocean's side runs back into the full hollow"
        );
        assert!(
            state.z[second as usize] > floor2,
            "the next hollow kept the rest"
        );
        assert_eq!((d.to_sea, d.dropped), (0, 0));
        assert_eq!(rock_cut(&state), rock_laid(&state, &before) + d.remainder);
        // The parent's own exit refused the same way: with no other way the rest is dropped and counted.
        let (mut state, first, _, above, gate) = two_dug_hollows();
        state.cut_now[above as usize] = (Z_STEPS_PER_M * 60) as u32;
        let d = state.deposit_basins(&Basins {
            slot: dug_slots(&state, first, NO_NODE, [NO_NODE, NO_NODE]),
            bodies: vec![(first.to_vec(), spill)],
            sides: vec![vec![(OVER_TO_FULL, gate)]],
            parent: vec![u32::MAX],
            parent_gate: vec![above],
        });
        assert!(d.over_spill > 0);
        assert_eq!(
            (d.dropped, d.stuck, d.refused),
            (d.over_spill, d.over_spill, 1)
        );
    }

    /// ★ THE REST PASSES THROUGH A FULL HOLLOW, AND NEVER TWICE (2026-10-05). The second hollow's floor has no room
    /// (body 0, its spill at its own height); the first hollow (body 1) fills, and its side's water ends on that floor:
    /// the rest passes THROUGH the full floor and on by the floor's own way, to the sea — nothing refused, nothing
    /// dropped. Where the floor's own way leads back into the first hollow, the rest would go round the two for ever:
    /// that way is refused, and with no other way the rest is dropped and counted.
    #[test]
    fn the_rest_passes_through_a_full_hollow_and_never_twice() {
        let (mut state, first, second, above, gate) = two_dug_hollows();
        let ring = [
            state.lattice.index(Face::PosX, 36, 20),
            state.lattice.index(Face::PosX, 37, 20),
        ];
        let slot: Vec<u32> = (0..state.node_count() as u32)
            .map(|v| {
                if v == second {
                    0
                } else if first.contains(&v) {
                    1
                } else if ring.contains(&v) {
                    2
                } else {
                    u32::MAX
                }
            })
            .collect();
        let floor_way = |way: u32| Basins {
            slot: slot.clone(),
            bodies: vec![
                (vec![second], state.z[second as usize]),
                (first.to_vec(), Z_STEPS_PER_M * 109),
                (ring.to_vec(), Z_STEPS_PER_M * 140),
            ],
            sides: vec![vec![(OVER_TO_SEA, way)], vec![(2, second)], Vec::new()],
            parent: vec![u32::MAX; 3],
            parent_gate: vec![NO_NODE; 3],
        };
        let through = floor_way(gate);
        let round = floor_way(above);
        state.cut_now[second as usize] = (Z_STEPS_PER_M * 10) as u32;
        state.cut_now[above as usize] = (Z_STEPS_PER_M * 60) as u32;
        let (start, floor) = (state.clone(), state.z[second as usize]);
        let before = start.z.clone();
        let d = state.deposit_basins(&through);
        assert_eq!(
            state.z[second as usize], floor,
            "the full floor kept no rock"
        );
        assert!(d.over_spill > 0);
        assert_eq!((d.to_sea, d.refused, d.dropped), (d.over_spill, 0, 0));
        assert_eq!(
            rock_cut(&state),
            rock_laid(&state, &before) + d.to_sea + d.remainder
        );
        let mut state = start;
        let d = state.deposit_basins(&round);
        assert_eq!(d.refused, 1, "the way back into the first hollow");
        assert!(d.stuck > 0);
        assert_eq!((d.dropped, d.to_sea), (d.stuck, 0));
        assert_eq!(
            rock_cut(&state),
            rock_laid(&state, &before) + d.dropped + d.remainder
        );
    }

    /// A body that rose into its parent's ring answers for the parent.
    #[test]
    fn a_body_that_rose_into_its_parent_answers_for_it() {
        assert_eq!(representative(&[0, 0, 1], 2), 0);
        assert_eq!(representative(&[0, 1, 2], 2), 2);
    }

    /// ★ THE BUDGET'S OWN HOLLOWS KEEP EVERY ROCK: on the dug state (a semi-arid stepped bowl and a trunk pit, the
    /// budget's own closed bodies, sides and parents), one sweep's cut equals what the deposit laid, what reached the
    /// sea, and the level's own remainder — nothing dropped.
    #[test]
    fn the_budgets_own_hollows_keep_every_rock() {
        let (mut state, words, body) = dug_state();
        let (climate, _) = dug::two_hollows(&body, &words, &mut state);
        state.route();
        state.accumulate();
        let basins = state
            .follow_the_budget(&climate, true)
            .expect("a wet body has closed hollows");
        assert!(!basins.bodies.is_empty());
        assert_eq!(basins.sides.len(), basins.bodies.len());
        let before = state.z.clone();
        let r = state.sweep(Gf::from_i64(2_000_000), None, Some(&basins));
        assert_eq!(r.rock_dropped, 0);
        assert_eq!(
            rock_cut(&state),
            rock_laid_net(&state, &before) + r.rock_to_sea + r.rock_remainder
        );
    }

    /// The rock laid since `before`, net of the sweep's own cut (the deposit's tests after a whole sweep).
    fn rock_laid_net(state: &MacroSolve, before: &[i32]) -> u128 {
        state
            .z
            .iter()
            .zip(before)
            .zip(&state.cut_now)
            .zip(&state.area)
            .map(|(((&z, &b), &c), &a)| {
                u128::from((i64::from(z) - i64::from(b) + i64::from(c)) as u64)
                    * u128::from(a.max(1))
            })
            .sum()
    }
}
