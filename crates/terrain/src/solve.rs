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
use vd_recipe::root::isqrt;

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

// ★ THE SCHEDULE's COST KNOBS (the design's ask 5, the recommended values taken; part of the world
// tag, so a change is a new world; 03 §5.5). They are NUMERICAL RESOLUTION — how finely the age is
// stepped — never a dial: the physical dial is the age, and it is the system's own.
/// The stream-power sweeps the age is divided into.
pub const PASSES: u32 = 40;
/// The climate is recomputed over the relief as it then stands every this many passes (C3).
pub const CLIMATE_EVERY: u32 = 10;
/// The flood, the flats and the receivers are recomputed every this many passes.
pub const FLOOD_EVERY: u32 = 10;
/// The flexural rebound runs every this many passes (C3).
pub const ISOSTASY_EVERY: u32 = 5;
/// The talus relaxations after the last sweep (C3).
pub const TALUS_PASSES: u32 = 8;

/// ★ THE ERODIBILITY, calibration body EARTH: `K₀ = 2 × 10⁻⁶ /yr` for the stream power law at
/// `m = ½, n = 1` with the drainage AREA in square metres — the middle of the published range
/// (Stock & Montgomery 1999, 10⁻⁶ to 10⁻⁴; Whipple & Tucker 1999). The law here reads the
/// DISCHARGE in mm·m²/yr, so the constant carries `1/√(mm per m)` with it ([`Schedule::gain`]).
/// C3 ties it down against the per-basin hypsometric integral (gate G-AGE); C1 states it.
pub const K0_PER_YR: f64 = 2.0e-6;
/// Millimetres in a metre: the discharge's unit against the law's.
pub const MM_PER_M: f64 = 1_000.0;
/// The fraction bits of the sweep's coefficient `Δt·K·√Q/L`.
pub const GAIN_BITS: u32 = 24;
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
/// The glacial erosion rate at the reference `H · S = 300 m × 0.05` (Hallet, Hunter & Bogen 1996,
/// calibration: temperate Alpine glaciers, about one millimetre a year), scaled by `H · S`.
pub const GLACIAL_RATE_M_PER_YR: f64 = 1.0e-3;
pub const GLACIAL_REFERENCE_HS_M: f64 = 15.0;
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

/// The schedule of one solve: the age the passes step through, and the two counts C1 reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule {
    /// ★ THE EROSIONAL AGE in years: the SYSTEM'S OWN age from the census (the design's ask 3).
    pub age_yr: u64,
    pub passes: u32,
    pub flood_every: u32,
    pub climate_every: u32,
    pub isostasy_every: u32,
    pub talus_passes: u32,
    /// The erodibility the sweep reads ([`K0_PER_YR`] by default; the bench's calibration scan
    /// states others).
    pub k0_per_yr: f64,
}

impl Schedule {
    /// The standard schedule for a body of a system `age_yr` old.
    #[must_use]
    pub const fn standard(age_yr: u64) -> Schedule {
        Schedule {
            age_yr,
            passes: PASSES,
            flood_every: FLOOD_EVERY,
            climate_every: CLIMATE_EVERY,
            isostasy_every: ISOSTASY_EVERY,
            talus_passes: TALUS_PASSES,
            k0_per_yr: K0_PER_YR,
        }
    }

    /// ★ THE SWEEP'S GAIN: `Δt · K₀/√(mm per m) · 2^GAIN_BITS`, floored — the one product of the
    /// law's constant and the age's step, computed ONCE under the fence. `Δt = age / passes`, so a
    /// finer schedule steps a smaller `Δt` and the answer converges to the same landscape (03 §4.6:
    /// the passes are a resolution, not a dial).
    #[must_use]
    pub fn gain(&self) -> u64 {
        let dt = Gf::from_i64(self.age_yr as i64) / Gf::from_i64(i64::from(self.passes));
        let k = Gf::from_f64(self.k0_per_yr) / Gf::from_f64(MM_PER_M).sqrt();
        (dt * k * Gf::from_i64(1 << GAIN_BITS)).to_i64_floor() as u64
    }
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
    /// Nodes routed by the flat's distance field rather than by a lower neighbour.
    pub flat: usize,
    /// Nodes the distance field never reached: a defect (gate G-DRAINAGE).
    pub undrained: usize,
    /// Nodes left out of the topological order: a cycle, a defect.
    pub cyclic: usize,
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
    /// The craters stamped at the start.
    pub craters: usize,
    /// Every rebound: the level, the nodes lifted, the greatest lift.
    pub rebounds: Vec<(u32, usize, i32)>,
    /// Every talus pass: the nodes that shed, the worst excess before it.
    pub talus: Vec<(usize, i64)>,
    /// The ice: the nodes under ice, the thickest ice in metres, the deepest cut.
    pub ice: (usize, f64, i32),
    /// The envelope: the greatest `|z|` before, and whether the scale was applied.
    pub envelope: (i32, bool),
    /// Gate G-AGE's reading: the integrals of every basin of at least [`AGE_GATE_MIN_NODES`]
    /// land nodes.
    pub integrals: Vec<BasinIntegral>,
    /// The sediment budget: basins with a deposit.
    pub deposits: usize,
    /// The coast band, sixteenths.
    pub coast_band: i32,
}

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
    /// The D8 receiver; [`NO_NODE`] for an outlet.
    pub receiver: Vec<u32>,
    /// The chord to the receiver in whole metres; zero for an outlet.
    pub chord: Vec<u32>,
    /// The node's true area in whole square metres.
    pub area: Vec<u64>,
    /// The rain at the node, mm/yr, in `[P_MIN_MM_YR, P_MAX_MM_YR]`.
    pub rain: Vec<u32>,
    /// The discharge: the rain accumulated downstream, mm·m²/yr.
    pub discharge: Vec<u64>,
    /// The topological order of the receiver tree, LEAVES FIRST, outlets last.
    pub order: Vec<u32>,
    /// ★ THE UPLIFT OVER THE AGE (C2's belts; C3): what each node gains by the end of the age,
    /// sixteenths, applied pass by pass; negative where a trench subsides.
    pub uplift: Vec<i32>,
    /// ★ THE REMOVAL FIELD: what the sweeps have cut from each node so far, sixteenths — the
    /// sediment budget's input.
    pub removed: Vec<u32>,
    /// The removal the rebound has not yet answered, sixteenths; consumed by every rebound.
    pub pending: Vec<u32>,
    /// The node directions at [`CACHE_BITS`], read by every chord.
    dir: Vec<[i32; 3]>,
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
    ((body.sea_radius - body.radius).raw() >> Z_SHIFT) as i32
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
        let lattice = MacroLattice::of(body)?;
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
            dir,
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
        state.sea_z = land.sea_z.unwrap_or(i32::MIN);
        Some(state)
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
        4 + 4 + 4 + 4 + 8 + 4 + 8 + 4 + 4 + 4 + 4 + 12
    }

    /// ★ THE UPLIFT OF ONE PASS at a node: the age's share that pass carries, as the difference of
    /// two exact cumulative shares `total · (pass + 1) / passes − total · pass / passes`, so the
    /// forty steps sum to the total to the sixteenth and a trench's negative total steps down.
    #[must_use]
    pub fn uplift_step(total: i32, pass: u32, passes: u32) -> i32 {
        let total = i64::from(total);
        let passes = i64::from(passes.max(1));
        let after = (total * (i64::from(pass) + 1)).div_euclid(passes);
        let before = (total * i64::from(pass)).div_euclid(passes);
        (after - before) as i32
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
    /// LAND nodes (over the sea), by area — `(mean − min) / (max − min)` in 1/256 — in outlet
    /// order. A basin whose land is flat (max = min) is left out: it has no integral.
    #[must_use]
    pub fn hypsometric_integrals(&self, min_nodes: u32) -> Vec<BasinIntegral> {
        let n = self.node_count();
        let basin = self.basins();
        // 1. Land nodes per basin.
        let mut count = vec![0u32; n];
        for i in 0..n {
            if self.z[i] > self.sea_z {
                count[basin[i] as usize] += 1;
            }
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
            if s == u32::MAX || self.z[i] <= self.sea_z {
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

    /// The water level at a node: a lake's spill level where the flood raised it, the sea's level
    /// under the sea, [`DRY`] elsewhere — one expression, no branch on a landform kind.
    #[must_use]
    pub fn water_level(&self, node: u32) -> i32 {
        let i = node as usize;
        if self.z_flood[i] > self.z[i] {
            self.z_flood[i]
        } else if self.z[i] <= self.sea_z {
            self.sea_z
        } else {
            DRY
        }
    }

    /// The chord between two nodes from the cached directions, whole metres.
    fn chord_m(&self, a: u32, b: u32) -> u32 {
        chord_between(
            self.lattice.radius_m(),
            uncache(self.dir[a as usize]),
            uncache(self.dir[b as usize]),
        )
    }

    /// ★ THE ROUTING: the priority flood, the D8 receivers on the flooded surface, the flats by a
    /// distance field, and the topological order of the receiver tree.
    pub fn route(&mut self) -> RouteReport {
        let n = self.node_count();
        let mut report = RouteReport::default();
        // 1. THE PRIORITY FLOOD. The outlets seed a heap keyed by `(height, index)`; popping the
        //    lowest first, every unvisited neighbour is raised to at least the popped level.
        let mut visited = vec![false; n];
        let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::new();
        let mut dist = vec![u32::MAX; n];
        for i in 0..n {
            self.receiver[i] = NO_NODE;
            self.chord[i] = 0;
            if self.z[i] <= self.sea_z {
                visited[i] = true;
                self.z_flood[i] = self.z[i];
                dist[i] = 0;
                heap.push(Reverse((self.z[i], i as u32)));
            }
        }
        report.sea_seeded = !heap.is_empty();
        if heap.is_empty() {
            let mut lowest: BinaryHeap<(i32, u32)> = BinaryHeap::new();
            for i in 0..n {
                lowest.push((self.z[i], i as u32));
                if lowest.len() > OUTLETS_WITHOUT_SEA {
                    lowest.pop();
                }
            }
            for (z, node) in lowest {
                let i = node as usize;
                visited[i] = true;
                self.z_flood[i] = z;
                dist[i] = 0;
                heap.push(Reverse((z, node)));
            }
        }
        report.outlets = heap.len();
        while let Some(Reverse((level, node))) = heap.pop() {
            for m in self.lattice.neighbours(node) {
                if m == NO_NODE || visited[m as usize] {
                    continue;
                }
                visited[m as usize] = true;
                let raised = self.z[m as usize].max(level);
                self.z_flood[m as usize] = raised;
                heap.push(Reverse((raised, m)));
            }
        }
        drop(heap);
        drop(visited);
        report.raised = (0..n).filter(|&i| self.z_flood[i] > self.z[i]).count();
        // 2. THE RECEIVERS: the steepest lower neighbour on the flooded surface.
        for node in 0..n as u32 {
            let i = node as usize;
            if dist[i] == 0 {
                continue;
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
            if best != NO_NODE {
                self.receiver[i] = best;
                self.chord[i] = l_best;
                dist[i] = 0;
            }
        }
        // 3. THE FLATS: a breadth-first distance to the nearest routed node over equal flooded
        //    heights, level by level (a distance depends on no visit order), then the receiver of a
        //    flat node is its neighbour with the smaller distance, ties to the smaller index.
        let mut frontier: Vec<u32> = (0..n as u32).filter(|&i| dist[i as usize] == 0).collect();
        let mut level = 0u32;
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for &node in &frontier {
                for m in self.lattice.neighbours(node) {
                    if m == NO_NODE
                        || dist[m as usize] != u32::MAX
                        || self.z_flood[m as usize] != self.z_flood[node as usize]
                    {
                        continue;
                    }
                    dist[m as usize] = level + 1;
                    next.push(m);
                }
            }
            frontier = next;
            level += 1;
        }
        let (flat, undrained) = self.assign_flat_receivers(&dist);
        report.flat = flat;
        report.undrained = undrained;
        drop(dist);
        // 4. THE ORDER: leaves first. A node is emitted once every donor is; the receiver whose
        //    last donor was emitted is pushed. A stack, so the order is a stated one.
        let mut donors = vec![0u8; n];
        for i in 0..n {
            let r = self.receiver[i];
            if r != NO_NODE {
                donors[r as usize] += 1;
            }
        }
        let mut stack: Vec<u32> = (0..n as u32).filter(|&i| donors[i as usize] == 0).collect();
        self.order.clear();
        while let Some(node) = stack.pop() {
            self.order.push(node);
            let r = self.receiver[node as usize];
            if r != NO_NODE {
                donors[r as usize] -= 1;
                if donors[r as usize] == 0 {
                    stack.push(r);
                }
            }
        }
        report.cyclic = n - self.order.len();
        report
    }

    /// ★ THE FLAT'S RECEIVERS from the distance field: a node the distance field reached (a distance
    /// over zero) takes its neighbour of equal flooded height with the SMALLER distance, ties to the
    /// smaller index; a node it never reached is counted UNDRAINED (a defect, gate G-DRAINAGE) and
    /// keeps no receiver. Returns `(flat, undrained)`.
    fn assign_flat_receivers(&mut self, dist: &[u32]) -> (usize, usize) {
        let (mut flat, mut undrained) = (0usize, 0usize);
        for i in 0..self.node_count() {
            if dist[i] == 0 {
                continue;
            }
            if dist[i] == u32::MAX {
                undrained += 1;
                continue;
            }
            flat += 1;
            let node = i as u32;
            let mut best = NO_NODE;
            for m in self.lattice.neighbours(node) {
                if m == NO_NODE
                    || dist[m as usize] == u32::MAX
                    || self.z_flood[m as usize] != self.z_flood[i]
                {
                    continue;
                }
                if best == NO_NODE
                    || dist[m as usize] < dist[best as usize]
                    || (dist[m as usize] == dist[best as usize] && m < best)
                {
                    best = m;
                }
            }
            self.receiver[i] = best;
            self.chord[i] = self.chord_m(node, best);
        }
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

    /// ★ THE STREAM POWER SWEEP, implicit (Braun & Willett 2013), outlets first:
    /// `z' = (z + c·b) / (1 + c)` with `c = Δt·K·√Q / L` and `b` the receiver's BASE LEVEL —
    /// its water level where it stands under water, its terrain height otherwise, one expression.
    /// An outlet is left alone; a lake node is skipped, so the hole the flood found stays a lake.
    /// ★ THE UPLIFT FIRST (C3): every node rises by its pass's share of the age's uplift before
    /// the cut, so a range is the balance of the two; `pass` of `passes` says which share.
    pub fn sweep(&mut self, gain: u64, pass: u32, passes: u32) -> SweepReport {
        let mut report = SweepReport::default();
        let one = 1i128 << GAIN_BITS;
        for i in 0..self.node_count() {
            self.z[i] += MacroSolve::uplift_step(self.uplift[i], pass, passes);
        }
        for k in (0..self.order.len()).rev() {
            let node = self.order[k];
            let i = node as usize;
            let r = self.receiver[i];
            if r == NO_NODE {
                continue;
            }
            if self.z_flood[i] > self.z[i] {
                report.skipped_lake += 1;
                continue;
            }
            let base = self.z[r as usize].max(self.water_level(r));
            let c = (u128::from(gain) * u128::from(isqrt(self.discharge[i])))
                / u128::from(self.chord[i].max(1));
            let c = c as i128;
            let numerator = (i128::from(self.z[i]) << GAIN_BITS) + c * i128::from(base);
            let cut = i128::from(self.z[i]) - numerator.div_euclid(one + c);
            let cut = cut as i32;
            if cut > 0 {
                report.lowered += 1;
                report.max_cut = report.max_cut.max(cut);
                report.total_cut += cut as u64;
                self.z[i] -= cut;
                self.removed[i] = self.removed[i].saturating_add(cut as u32);
                self.pending[i] = self.pending[i].saturating_add(cut as u32);
            }
        }
        report
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
    /// pyramid to the level nearest the flexural parameter `alpha_m` (an integer mean over each
    /// block, exact and order-free), smoothed there by [`SMOOTH_PASSES`] nine-node means through
    /// the seam table, prolonged back to every node's nearest parent, and `ρ_crust / ρ_mantle` of
    /// it is added to the terrain — the crust floats up under the lightened load. Returns the
    /// level used, the nodes lifted and the greatest lift, in sixteenths.
    ///
    /// **Example.** The home planet's rivers cut a 400 m valley into a range; the rebound lifts the
    /// 150 km around it by about 40 m, so the ridge beside the valley ends up TALLER than it
    /// started, and the range keeps its snow line instead of wearing down into hills.
    pub fn rebound(&mut self, alpha_m: f64) -> (u32, usize, i32) {
        let n = self.node_count();
        let k = self.coarse_level(alpha_m);
        let coarse = self.lattice.coarser(k).unwrap_or(self.lattice);
        let m = coarse.node_count();
        // 1. Restrict: the mean of each 2^k × 2^k block.
        let mut field = vec![0u64; m];
        let mut parent = Vec::with_capacity(n);
        for node in 0..n as u32 {
            let (face, i, j) = self.lattice.split(node);
            let c = coarse.index(face, i >> k, j >> k);
            parent.push(c);
            field[c as usize] += u64::from(self.pending[node as usize]);
            self.pending[node as usize] = 0;
        }
        for f in &mut field {
            *f >>= 2 * k;
        }
        // 2. Smooth: the nine-node mean, a stated number of times, through the seam table.
        let mut pass = 0;
        while pass < SMOOTH_PASSES {
            let mut next = vec![0u64; m];
            for c in 0..m as u32 {
                let mut sum = field[c as usize];
                let mut count = 1u64;
                for nb in coarse.neighbours(c) {
                    if nb != NO_NODE {
                        sum += field[nb as usize];
                        count += 1;
                    }
                }
                next[c as usize] = sum / count;
            }
            field = next;
            pass += 1;
        }
        // 3. Prolong and lift.
        let share = Gf::from_f64(
            crate::land::CONTINENTAL_CRUST_DENSITY_KGM3 / crate::land::MANTLE_DENSITY_KGM3,
        );
        let (mut lifted, mut max_lift) = (0usize, 0i32);
        for node in 0..n {
            let lift =
                (Gf::from_i64(field[parent[node] as usize] as i64) * share).to_i64_floor() as i32;
            if lift > 0 {
                self.z[node] += lift;
                lifted += 1;
                max_lift = max_lift.max(lift);
            }
        }
        (k, lifted, max_lift)
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

    /// ★ THE ICE (03 §4.9): above the equilibrium line `ela_z` snow outlives the summer; the
    /// glacier's thickness follows the perfect-plastic law over the slope to the receiver, capped
    /// by the height above the line, and the trunk is over-deepened by the glacial erosion law
    /// over the glacial epoch, at most by the ice's own thickness. A node under water, an outlet
    /// or a node with no drop to its receiver is not cut. Returns the ice mask, and (the nodes
    /// under ice, the greatest thickness in metres, the greatest cut in sixteenths).
    ///
    /// **Example.** On the home planet a 6 000 m range stands over its snow line: its trunk
    /// valleys are cut into troughs while the ridges between them stay knife-edged; the moon,
    /// whose line stands above every node, keeps every crater.
    pub fn ice(&mut self, ela_z: &[i32], gravity_mm_s2: u32) -> (Vec<bool>, usize, f64, i32) {
        let n = self.node_count();
        let mut mask = vec![false; n];
        let (mut under, mut thickest, mut deepest) = (0usize, 0.0f64, 0i32);
        let g = Gf::from_i64(i64::from(gravity_mm_s2)) / Gf::from_i64(1_000);
        let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
        let rate = Gf::from_f64(GLACIAL_RATE_M_PER_YR) * Gf::from_i64(GLACIAL_EPOCH_YR as i64)
            / Gf::from_f64(GLACIAL_REFERENCE_HS_M);
        for node in 0..n as u32 {
            let i = node as usize;
            if self.z[i] < ela_z[i] || self.z_flood[i] > self.z[i] || self.z[i] <= self.sea_z {
                continue;
            }
            mask[i] = true;
            under += 1;
            let r = self.receiver[i];
            if r == NO_NODE {
                continue;
            }
            let drop = Gf::from_i64(i64::from(self.z[i]) - i64::from(self.z[r as usize])) / steps;
            if drop <= Gf::ZERO {
                continue;
            }
            let slope = drop / Gf::from_i64(i64::from(self.chord[i].max(1)));
            let above = Gf::from_i64(i64::from(self.z[i]) - i64::from(ela_z[i])) / steps;
            let plastic =
                Gf::from_f64(ICE_YIELD_STRESS_PA) / (Gf::from_f64(ICE_DENSITY_KGM3) * g * slope);
            let thickness = plastic.lesser(above);
            let cut_m = (rate * thickness * slope).lesser(thickness);
            let cut = (cut_m * steps).to_i64_floor() as i32;
            if thickness.to_f64() > thickest {
                thickest = thickness.to_f64();
            }
            if cut > 0 {
                self.z[i] -= cut;
                deepest = deepest.max(cut);
            }
        }
        (mask, under, thickest, deepest)
    }

    /// ★ THE FACIES BYTE of every node: the sea, a lake, the coast (within `band` sixteenths of the
    /// sea's level), the ice — the bits 8d and 8e read. A body with no water has no sea and no
    /// lake: the flood's raised pits on a dry moon are closed basins, not lakes.
    #[must_use]
    pub fn facies(&self, ice: &[bool], band: i32, has_water: bool) -> Vec<u8> {
        (0..self.node_count() as u32)
            .map(|node| {
                let i = node as usize;
                let mut f = 0u8;
                if has_water && self.z[i] <= self.sea_z {
                    f |= FACIES_SEA;
                }
                if has_water && self.z_flood[i] > self.z[i] {
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

    /// ★ THE ENVELOPE (03 §4.13): `|z| ≤ relief` at every node, or every height is scaled DOWN
    /// uniformly by `relief / max|z|` — one fenced division, one floor a node — so the shape is
    /// kept and the containment band never moves. Returns the greatest `|z|` before, and whether
    /// the scale was applied.
    pub fn envelope(&mut self, relief: i32) -> (i32, bool) {
        let mut worst = 0i32;
        for &z in &self.z {
            worst = worst.max(z.abs());
        }
        if worst <= relief {
            return (worst, false);
        }
        let scale = Gf::from_i64(i64::from(relief)) / Gf::from_i64(i64::from(worst));
        for z in &mut self.z {
            *z = (Gf::from_i64(i64::from(*z)) * scale).to_i64_floor() as i32;
        }
        (worst, true)
    }

    /// The lowest and the highest terrain, in sixteenths from the ladder radius.
    #[must_use]
    pub fn range(&self) -> (i32, i32) {
        self.z
            .iter()
            .fold((i32::MAX, i32::MIN), |(lo, hi), &z| (lo.min(z), hi.max(z)))
    }
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

/// ★ THE DRIVER: the schedule over `body` from the recipe's own relief (the C1 start) — a routing
/// and an accumulation every `flood_every` passes, a sweep every pass. `None` where the body has
/// no lattice.
#[must_use]
pub fn solve(body: &BodyDefinition, schedule: Schedule) -> Option<(MacroSolve, SolveReport)> {
    run(MacroSolve::new(body)?, schedule)
}

/// ★ THE DRIVER FROM THE INITIAL LAND (stage C2): the same schedule from the land the plates,
/// the isostasy and the belts made under the charter's words.
#[must_use]
pub fn solve_land(
    body: &BodyDefinition,
    words: &crate::land::LandWords,
    schedule: Schedule,
) -> Option<(MacroSolve, SolveReport)> {
    let lattice = MacroLattice::of(body)?;
    let land = crate::land::initial_land(body, &lattice, words);
    run(MacroSolve::from_land(body, &land)?, schedule)
}

/// ★ THE FULL SOLVE (stage C3; the design's §4.11): the land, the craters stamped on it, then the
/// schedule — the climate over the relief as it stands every `climate_every` passes (its rain is
/// the discharge's source), a routing every `flood_every`, a sweep with the uplift every pass, a
/// rebound every `isostasy_every` — then the talus relaxations, the ice, the coast band, the
/// envelope, and the two readings: gate G-AGE's integrals and the sediment budget. The facies
/// byte comes out beside the state. ONE thread, off the tick; `None` where the body has no
/// lattice.
///
/// **Example.** The home planet's shard solves its planet once: the plates, fourteen; the
/// craters the air let through; forty sweeps of rivers fed by the rain the ranges themselves
/// lift out of the wind; the ridges floating up as the valleys empty; the last scree settling;
/// the high troughs cut by ice. The artifact is what it keeps; the ground never changes again.
#[must_use]
pub fn solve_full(
    body: &BodyDefinition,
    words: &SolveWords,
    schedule: Schedule,
) -> Option<(MacroSolve, Vec<u8>, FullReport)> {
    let lattice = MacroLattice::of(body)?;
    let land = crate::land::initial_land(body, &lattice, &words.land());
    let mut state = MacroSolve::from_land(body, &land)?;
    let gravity = body.facts().gravity_mm_s2;
    let mut report = FullReport::default();
    // The craters: the impact record, stamped before the water works.
    let craters = crate::craters::crater_population(body, &lattice, words);
    crate::craters::apply_craters(&mut state.z, &lattice, &craters, gravity);
    report.craters = craters.len();
    drop(craters);
    // A crater never breaks the envelope: its bowl and its rim are CLIPPED at the relief (a local
    // rule, so a big basin on a small moon does not scale the whole moon down at the end).
    let relief = envelope_steps(body);
    for z in &mut state.z {
        *z = (*z).clamp(-relief, relief);
    }
    // ★ AND THE UPLIFT IS CLAMPED AGAINST THE CRATERED FIELD: a crater floor clipped at −relief
    // whose trench then subsides would leave the envelope (MEASURED: −9 333 m under 8 276 m).
    for (u, &z) in state.uplift.iter_mut().zip(&state.z) {
        *u = (*u).clamp(-relief - z, relief - z);
    }
    state.z_flood.clone_from(&state.z);
    let gain = schedule.gain();
    let alpha_m = crate::land::flexural_parameter_m(words.elastic_thickness_m, gravity);
    let mut since_route = schedule.flood_every;
    let mut since_climate = schedule.climate_every;
    let mut since_rebound = 0;
    let mut climate = None;
    for pass in 0..schedule.passes {
        if since_climate >= schedule.climate_every {
            let c = crate::climate::climate(body, &lattice, words, &state.z, land.sea_z);
            state.rain.clone_from(&c.rain_mm_yr);
            climate = Some(c);
            since_climate = 0;
            // A new rain is a new discharge on the tree as it stands.
            since_route = schedule.flood_every;
        }
        since_climate += 1;
        if since_route >= schedule.flood_every {
            report.routes.push(state.route());
            state.accumulate();
            since_route = 0;
        }
        since_route += 1;
        report.sweeps.push(state.sweep(gain, pass, schedule.passes));
        since_rebound += 1;
        if since_rebound >= schedule.isostasy_every {
            report.rebounds.push(state.rebound(alpha_m));
            since_rebound = 0;
        }
    }
    // The climate over the final relief, for the talus's angle and the ice's line.
    let climate = match climate {
        Some(c) => c,
        None => crate::climate::climate(body, &lattice, words, &state.z, land.sea_z),
    };
    let tan: Vec<u32> = climate
        .aridity_q8
        .iter()
        .map(|&a| tan_repose_q16(a))
        .collect();
    for _ in 0..schedule.talus_passes {
        report.talus.push(state.talus(&tan));
    }
    drop(tan);
    // No water, no ice: a dry body's line stands over every node (03 §4.9), whatever its cold.
    let no_ice = vec![i32::MAX; state.node_count()];
    let line: &[i32] = if words.water_km3 == 0 {
        &no_ice
    } else {
        &climate.ela_z
    };
    let (ice, under, thickest, deepest) = state.ice(line, gravity);
    report.ice = (under, thickest, deepest);
    drop(climate);
    report.envelope = state.envelope(relief);
    // The final routing, so the water levels, the basins and the facies read the final land.
    report.routes.push(state.route());
    report.coast_band = coast_band(words, gravity);
    let facies = state.facies(&ice, report.coast_band, words.water_km3 > 0);
    report.integrals = state.hypsometric_integrals(AGE_GATE_MIN_NODES);
    report.deposits = state.sediment_by_basin().len();
    Some((state, facies, report))
}

/// ★ THE ENVELOPE's BOUND in sixteenths: the lesser of the relief the seed drew and the band the
/// ladder holds at rung 0 (on the home moon the draw stands over the band), so the solve's field
/// fits the grid on every body. C4 subtracts the fine octaves' sum from it when `Z` replaces the
/// coarse octaves (03 §4.13).
#[must_use]
pub fn envelope_steps(body: &BodyDefinition) -> i32 {
    let relief = body.relief_m();
    let band = body.relief_bound_m(0);
    let bound = if band < relief { band } else { relief };
    (Gf::from_f64(bound) * Gf::from_i64(i64::from(Z_STEPS_PER_M))).to_i64_floor() as i32
}

/// The schedule over a state.
fn run(mut state: MacroSolve, schedule: Schedule) -> Option<(MacroSolve, SolveReport)> {
    let gain = schedule.gain();
    let mut report = SolveReport::default();
    let mut since_route = schedule.flood_every;
    for pass in 0..schedule.passes {
        if since_route >= schedule.flood_every {
            report.routes.push(state.route());
            state.accumulate();
            since_route = 0;
        }
        since_route += 1;
        report.sweeps.push(state.sweep(gain, pass, schedule.passes));
    }
    Some((state, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_planet};
    use crate::macro_lattice::STENCIL;
    use vd_seed::bend::Face;

    /// The gain: five billion years over forty passes, times the Earth-calibrated erodibility over
    /// the root of a thousand, at twenty-four fraction bits — about 132 million.
    #[test]
    fn the_gain_is_the_ages_step_times_the_erodibility() {
        let gain = Schedule::standard(HOME_SYSTEM_AGE_YR).gain();
        let expect = 5.0e9 / 40.0 * 2.0e-6 / 1000f64.sqrt() * f64::from(1u32 << GAIN_BITS);
        assert_eq!(gain, expect.floor() as u64);
        assert!((132_000_000..133_000_000).contains(&gain), "{gain}");
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

    /// ★ THE DRIVER ON A REAL SMALL BODY OF THE WORLD (06 §3.3): the home moon's whole schedule —
    /// four routings, forty sweeps; every routing drains every node to an outlet (gate
    /// G-DRAINAGE) with no cycle; the sweeps only lower; the relief stays inside the band.
    #[test]
    fn the_home_moon_solves_and_every_node_drains() {
        let moon = home_moon();
        let before = MacroSolve::new(&moon).expect("a state");
        let (state, report) =
            solve(&moon, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        assert_eq!(report.sweeps.len(), 40);
        assert_eq!(report.routes.len(), 4);
        for r in &report.routes {
            assert_eq!((r.undrained, r.cyclic), (0, 0), "{r:?}");
            assert!(r.outlets > 0);
        }
        assert!(report.sweeps[0].lowered > 0);
        for (i, (&a, &b)) in before.z.iter().zip(state.z.iter()).enumerate() {
            assert!(b <= a, "node {i} rose from {a} to {b}");
        }
        let (lo, hi) = state.range();
        let (lo0, hi0) = before.range();
        assert!(lo >= lo0, "the floor fell: {lo0} to {lo}");
        assert!(hi <= hi0, "the ceiling rose: {hi0} to {hi}");
        assert_eq!(state.bytes_per_node(), 64);
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

    /// ★ THE DRIVER FROM THE LAND on the moon: a stagnant lid with no sea, so the flood seeds from
    /// the six lowest nodes, every node drains, and the sweeps only lower.
    #[test]
    fn the_moon_solves_from_its_initial_land() {
        let moon = home_moon();
        let (state, report) = solve_land(
            &moon,
            &crate::home::home_moon_land_words(),
            Schedule::standard(HOME_SYSTEM_AGE_YR),
        )
        .expect("a solve");
        assert_eq!(state.sea_z, i32::MIN);
        assert_eq!(report.routes.len(), 4);
        for r in &report.routes {
            assert!(!r.sea_seeded);
            assert_eq!(
                (r.outlets, r.undrained, r.cyclic),
                (OUTLETS_WITHOUT_SEA, 0, 0)
            );
        }
        let band = (moon.relief_bound_m(0) * f64::from(Z_STEPS_PER_M)) as i32;
        assert!(state.z.iter().all(|&z| z.abs() <= band));
    }

    /// ★ THE UPLIFT'S STEPS sum to the total to the sixteenth over forty passes, a negative total
    /// steps down, and a zero gain leaves the cut at nothing: the moon under a stated uplift of
    /// 100 m everywhere rises by exactly 1 600 sixteenths over the schedule.
    #[test]
    fn the_uplift_steps_sum_exactly_and_a_trench_subsides() {
        let mut sum = 0;
        for pass in 0..40 {
            sum += MacroSolve::uplift_step(1_601, pass, 40);
        }
        assert_eq!(sum, 1_601);
        let mut down = 0;
        for pass in 0..40 {
            down += MacroSolve::uplift_step(-803, pass, 40);
        }
        assert_eq!(down, -803);
        assert_eq!(MacroSolve::uplift_step(1_600, 3, 0), 1_600);
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        let before = state.z.clone();
        state.uplift = vec![1_600; n];
        state.route();
        for pass in 0..40 {
            let s = state.sweep(0, pass, 40);
            assert_eq!(s.lowered, 0);
        }
        for (i, &b) in before.iter().enumerate() {
            assert_eq!(state.z[i] - b, 1_600);
        }
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
        state.sweep(Schedule::standard(HOME_SYSTEM_AGE_YR).gain(), 0, 1);
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

    /// ★ THE REBOUND on the plateau: one sweep cuts the nodes at the outlet; the rebound lifts
    /// them and their surroundings by the crust's share of the smoothed cut, spread over the
    /// level nearest the flexural parameter, and consumes the pending removal; a rebound with
    /// nothing pending lifts nothing. The mass lifted is the crust's share of the mass cut,
    /// within the block means' floors.
    #[test]
    fn the_rebound_lifts_the_crusts_share_of_the_cut() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![Z_STEPS_PER_M * 100; n];
        state.sea_z = 0;
        let outlet = state.lattice.index(Face::PosX, 20, 20);
        state.z[outlet as usize] = 0;
        state.route();
        state.accumulate();
        state.sweep(Schedule::standard(HOME_SYSTEM_AGE_YR).gain(), 0, 1);
        let cut: u64 = state.pending.iter().map(|&p| u64::from(p)).sum();
        assert!(cut > 0);
        let before = state.z.clone();
        // A parameter of about four nodes: level 2 (32 768 m against 8 192 · 4).
        assert_eq!(state.coarse_level(33_000.0), 2);
        assert_eq!(state.coarse_level(1.0), 0);
        let (level, lifted, max_lift) = state.rebound(33_000.0);
        assert_eq!(level, 2);
        assert!(lifted > 0);
        assert!(max_lift > 0);
        assert!(state.pending.iter().all(|&p| p == 0));
        let lift: u64 = (0..n).map(|i| (state.z[i] - before[i]) as u64).sum();
        assert!(lift > 0);
        assert!(lift <= cut, "lift {lift} over the cut {cut}");
        // Nothing pending: nothing lifted.
        let again = state.z.clone();
        assert_eq!(state.rebound(33_000.0), (2, 0, 0));
        assert_eq!(state.z, again);
        // The finest level when the parameter is under a node.
        let (level, _, _) = state.rebound(1.0);
        assert_eq!(level, 0);
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

    /// ★ THE ICE on a stated range: with the line under a peak the peak's slope to its receiver is
    /// cut by at most the ice's thickness and the mask marks it; with the line over every node
    /// (an airless moon) nothing is under ice; a node under the sea or in a lake, an outlet, and
    /// a node with no drop are never cut.
    #[test]
    fn the_ice_cuts_the_trunk_under_the_line_and_never_the_water() {
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
        let no_ice = vec![i32::MAX; n];
        let (mask, under, thickest, deepest) = state.ice(&no_ice, 330);
        assert_eq!((under, deepest), (0, 0));
        assert_eq!(thickest, 0.0);
        assert!(mask.iter().all(|&m| !m));
        // The line at 2 000 m: only the peak stands over it.
        let line = vec![2_000 * Z_STEPS_PER_M; n];
        let before = state.z[peak as usize];
        let (mask, under, thickest, deepest) = state.ice(&line, 330);
        assert_eq!(under, 1);
        assert!(mask[peak as usize]);
        assert!(thickest > 0.0);
        assert!(
            thickest <= 1_000.0,
            "capped by the height over the line: {thickest}"
        );
        assert!(deepest > 0);
        assert!(before - state.z[peak as usize] == deepest);
        assert!(deepest <= (thickest * f64::from(Z_STEPS_PER_M)) as i32 + 1);
        // The line at the sea: the outlet, the pit (a lake) and the flat plateau (no drop to a
        // receiver at the same height) are under the line yet not cut.
        let z_before = state.z.clone();
        let low = vec![0; n];
        let (mask, under, _, _) = state.ice(&low, 330);
        assert!(!mask[outlet as usize]);
        assert!(!mask[pit as usize]);
        assert!(under > 0);
        let flat = state.lattice.index(Face::PosY, 5, 5);
        assert!(mask[flat as usize]);
        assert_eq!(state.z[flat as usize], z_before[flat as usize]);
        // No sea: the six lowest nodes are outlets, under the line yet never cut (no receiver);
        // and a node one metre over the line with a one-sixteenth drop over eight kilometres has
        // a cut that floors to nothing.
        let mut dry = MacroSolve::new(&moon).expect("a state");
        dry.z = vec![Z_STEPS_PER_M * 100; n];
        dry.sea_z = i32::MIN;
        let top = dry.lattice.index(Face::PosZ, 40, 40);
        let under = dry.lattice.index(Face::PosZ, 41, 40);
        dry.z[top as usize] = Z_STEPS_PER_M * 100 + Z_STEPS_PER_M;
        dry.z[under as usize] = Z_STEPS_PER_M * 100 + Z_STEPS_PER_M - 1;
        dry.route();
        let outlet = (0..n)
            .find(|&i| dry.receiver[i] == NO_NODE)
            .expect("an outlet");
        let line = vec![Z_STEPS_PER_M * 100; n];
        let z_before = dry.z.clone();
        let (mask, _, _, _) = dry.ice(&line, 330);
        assert!(mask[outlet]);
        assert_eq!(dry.z[outlet], z_before[outlet]);
        assert!(mask[top as usize]);
        assert_eq!(
            dry.z[top as usize], z_before[top as usize],
            "a cut under a sixteenth is no cut"
        );
    }

    /// ★ THE FACIES, THE COAST BAND AND THE ENVELOPE on the plateau: the outlet is sea and coast,
    /// the pit a lake, an ice node ice; an airless body has no band and the home planet's is a
    /// few metres; the envelope leaves a field inside the relief alone and scales one outside it
    /// down to the relief.
    #[test]
    fn facies_coast_band_and_envelope() {
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
        let facies = state.facies(&ice, 4, true);
        assert_eq!(facies[outlet as usize], FACIES_SEA | FACIES_COAST);
        assert_eq!(facies[pit as usize], FACIES_LAKE);
        assert_eq!(facies[icy as usize], FACIES_ICE);
        assert_eq!(facies[state.lattice.index(Face::PosY, 8, 8) as usize], 0);
        // A dry body: the same pits are closed basins, never lakes, and there is no sea.
        let dry = state.facies(&ice, 4, false);
        assert_eq!(dry[outlet as usize], FACIES_COAST);
        assert_eq!(dry[pit as usize], 0);
        assert_eq!(coast_band(&crate::home::home_moon_solve_words(), 330), 0);
        let home_band = coast_band(&crate::home::home_solve_words(), 9_818);
        assert!((100..=200).contains(&home_band), "{home_band}");
        // The envelope.
        let relief = 2 * Z_STEPS_PER_M;
        assert_eq!(state.envelope(relief), (Z_STEPS_PER_M, false));
        state.z[icy as usize] = 8 * Z_STEPS_PER_M;
        assert_eq!(state.envelope(relief), (8 * Z_STEPS_PER_M, true));
        assert_eq!(state.z[icy as usize], relief);
        assert_eq!(
            state.z[state.lattice.index(Face::PosY, 8, 8) as usize],
            Z_STEPS_PER_M / 4
        );
    }

    /// ★ THE FULL SOLVE ON THE MOON (airless, dry, a stagnant lid): craters stamped, no rain so no
    /// cut and no deposit, no ice, no coast band, every routing drained, the envelope holding, no
    /// integral read (no sea, so no land over it: every node counts as land — the basins of the
    /// six lowest nodes are read); the facies carry no sea, no coast and no ice.
    #[test]
    fn the_full_solve_on_the_airless_moon() {
        let moon = home_moon();
        let words = crate::home::home_moon_solve_words();
        let (state, facies, report) =
            solve_full(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        assert!(report.craters > 0);
        assert!(
            report.sweeps.iter().all(|s| s.lowered == 0),
            "no rain, no cut"
        );
        assert_eq!(report.deposits, 0);
        assert_eq!(report.ice, (0, 0.0, 0));
        assert_eq!(report.coast_band, 0);
        let relief = envelope_steps(&moon);
        assert!(
            state.z.iter().all(|&z| z.abs() <= relief),
            "the envelope holds"
        );
        assert!(
            relief < (moon.relief_m() * f64::from(Z_STEPS_PER_M)) as i32,
            "the band binds"
        );
        assert_eq!(
            envelope_steps(&home_planet()),
            (home_planet().relief_m() * 16.0) as i32
        );
        assert_eq!(report.rebounds.len(), 8);
        assert_eq!(report.talus.len(), 8);
        assert_eq!(report.routes.len(), 5);
        assert!(report.routes.iter().all(|r| r.undrained == 0));
        assert!(report.routes.iter().all(|r| r.cyclic == 0));
        assert!(
            facies
                .iter()
                .all(|&f| f & (FACIES_SEA | FACIES_LAKE | FACIES_COAST | FACIES_ICE) == 0)
        );
        assert_eq!(state.node_count(), facies.len());
        let band = (moon.relief_bound_m(0) * f64::from(Z_STEPS_PER_M)) as i32;
        assert!(state.z.iter().all(|&z| z.abs() <= band));
    }

    /// A schedule of no passes: the climate is computed once at the end for the talus and the
    /// ice; the land, the craters, the talus and the envelope still run.
    #[test]
    fn a_schedule_of_no_passes_still_reads_a_climate() {
        let moon = home_moon();
        let words = crate::home::home_moon_solve_words();
        let schedule = Schedule {
            passes: 0,
            ..Schedule::standard(HOME_SYSTEM_AGE_YR)
        };
        let (_, _, report) = solve_full(&moon, &words, schedule).expect("a solve");
        assert_eq!(report.sweeps.len(), 0);
        assert_eq!(report.routes.len(), 1);
        assert_eq!(report.talus.len(), 8);
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
            solve_full(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        assert!(report.sweeps.iter().any(|s| s.lowered > 0), "the rain cuts");
        assert!(report.deposits > 0);
        assert!(report.rebounds.iter().any(|&(_, lifted, _)| lifted > 0));
        assert!(report.coast_band > 0);
        assert!(facies.iter().any(|&f| f & FACIES_SEA != 0));
        assert!(facies.iter().any(|&f| f & FACIES_COAST != 0));
        assert!(!report.integrals.is_empty());
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
        assert_eq!(state.sweep(1 << 30, 0, 1), SweepReport::default());
    }

    /// ★ THE KERNELS ON A STATED NEIGHBOURHOOD (06 §3.3): a plateau one metre over the sea with
    /// one outlet and one pit. The flood raises the pit to the plateau (a lake with the plateau's
    /// spill level), the plateau is a flat that drains to the outlet by distance, the lake node is
    /// skipped by the sweep, its neighbours' base level is the lake's level, and the nodes at the
    /// outlet are cut.
    #[test]
    fn a_pit_becomes_a_lake_and_a_plateau_drains_by_distance() {
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
        assert_eq!(state.water_level(pit), plateau);
        assert_eq!(state.water_level(outlet), 0);
        let dry = state.lattice.index(Face::PosY, 1, 1);
        assert_eq!(state.water_level(dry), DRY);
        for m in state.lattice.neighbours(outlet) {
            assert_eq!(state.receiver[m as usize], outlet);
            assert!(state.chord[m as usize] > 0);
        }
        state.accumulate();
        let total: u64 = state.area.iter().sum::<u64>() * u64::from(P_MIN_MM_YR);
        assert_eq!(state.discharge[outlet as usize], total);
        let s = state.sweep(Schedule::standard(HOME_SYSTEM_AGE_YR).gain(), 0, 1);
        assert_eq!(s.skipped_lake, 1);
        assert!(s.lowered >= 8, "{s:?}");
        assert!(s.max_cut > 0);
        assert_eq!(
            state.z[pit as usize],
            plateau / 2,
            "the lake keeps its hole"
        );
        // A node whose receiver is the lake reads the lake's LEVEL as its base, so it is not cut
        // below the plateau toward the hole.
        let donors: Vec<usize> = (0..n).filter(|&i| state.receiver[i] == pit).collect();
        assert!(!donors.is_empty());
        for i in donors {
            assert_eq!(state.z[i], plateau);
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

    /// ★ THE UNDRAINED COUNTER on a stated distance field: a node the field never reached is counted
    /// and keeps no receiver; a reached node skips an unreached neighbour and takes the reached one.
    #[test]
    fn an_unreached_node_is_counted_undrained_and_skipped_as_a_neighbour() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z_flood = vec![0; n];
        let mut dist = vec![u32::MAX; n];
        let a = state.lattice.index(Face::PosX, 20, 20);
        let ring = state.lattice.neighbours(a);
        // One routed neighbour at distance 0, the rest unreached; `a` itself at distance 1.
        dist[ring[7] as usize] = 0;
        dist[a as usize] = 1;
        let (flat, undrained) = state.assign_flat_receivers(&dist);
        assert_eq!(flat, 1);
        assert_eq!(undrained, n - 2);
        assert_eq!(state.receiver[a as usize], ring[7]);
        assert_eq!(state.receiver[ring[0] as usize], NO_NODE);
    }

    /// The tie in the flat's receiver choice: two neighbours at one distance keep the smaller index.
    /// On the plateau every routed neighbour of the outlet is at distance zero, so a flat node
    /// adjacent to two of them takes the smaller.
    #[test]
    fn a_flat_node_between_two_routed_nodes_takes_the_smaller_index() {
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
        let candidates = [
            state.lattice.index(Face::PosX, 11, 9),
            state.lattice.index(Face::PosX, 11, 10),
            state.lattice.index(Face::PosX, 11, 11),
        ];
        assert_eq!(
            state.receiver[node as usize],
            *candidates.iter().min().expect("three")
        );
    }
}
