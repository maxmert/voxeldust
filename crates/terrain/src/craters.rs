//! ★ THE CRATERS (the landform arc, slice 8c stage C3; `slice_8c_design.md` §4.9; the
//! investigation's `02_planet_layout.md` §4.7).
//!
//! A crater is the record of one impact. A surface keeps them over its age, and the air above it
//! decides which impactors reach the ground at all: a thick atmosphere burns the small ones, so a
//! body's smallest crater grows with its surface pressure. This module puts that record on the
//! macro lattice, ONCE per body, before the solve's rivers run — the rivers then erase the old
//! craters where they cut, so a cratered highland and a smooth wet lowland come out of the same
//! passes, with no branch on a body kind (HR4).
//!
//! Three published laws, each with its calibration body (ruling T9):
//! - **the production function** (Neukum, Ivanov & Hartmann 2001, the lunar chronology): how many
//!   craters wider than one kilometre a square kilometre collects over an age;
//! - **the size-frequency slope** (02 §4.7): make the diameter ten times smaller and find a hundred
//!   times as many, `N(>D) ∝ D⁻²`;
//! - **the depth, the rim and the ejecta** (Pike 1977, the Moon; McGetchin, Settle & Head 1973):
//!   a simple crater is a bowl a fifth as deep as it is wide; past the simple-to-complex transition,
//!   which scales as `1/g` (the Moon's 15 km at its own gravity), the floor flattens and a central
//!   peak stands; the rim is a raised ring and the ejecta a skirt that thins as the cube of the
//!   distance out to one radius beyond the rim.
//!
//! Two more numbers bound the record: **the atmospheric screening** (a law calibrated on Earth and
//! checked on Venus) and **the saturation** (Gault 1970's geometric limit, at the share the lunar
//! highlands actually reach — Hartmann 1984). A draw exists only for the identity choices — WHERE
//! a crater sits and its size along the law's own distribution — and the count's fractional part.
//!
//! **Example.** The home moon is airless and five billion years old, so its lattice collects a few
//! hundred craters wider than two nodes, largest first, and keeps every one: a pilot walking its
//! plain crosses a sixteen-kilometre bowl with a raised rim and a skirt of ejecta. The home planet
//! under 101 kPa of air collects craters of the same sizes, and the solve's rivers cut them away
//! wherever the rain fell.

// ★ THE CRATERS MAY DIVIDE: they run once per body on the CPU, never in a kernel (03 §5.3).
#![allow(
    clippy::integer_division,
    clippy::modulo_arithmetic,
    reason = "the crater record runs on the CPU once per body, never in a kernel; integer-exact on every host"
)]

use vd_seed::bend::Face;
use vd_seed::rng::{SplitMix64, child_seed};

use crate::body::{BodyDefinition, draw_unit, salt};
use crate::gf::Gf;
use crate::macro_lattice::{MacroLattice, NO_NODE};
use crate::solve::{SolveWords, Z_STEPS_PER_M};

/// One crater: the node under its centre, its rim-to-rim diameter in whole metres, and the age of
/// the impact that made it, years before now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crater {
    pub node: u32,
    pub diameter_m: u32,
    pub age_yr: u64,
}

/// ★ WHAT THE RETENTION LAW READS about a surface (ruling B2 step 2): the initial land's own rows
/// and the climate over them. Nothing here is new data — every row already exists for another
/// reason, and the craters simply ask them how long this ground has stood.
pub struct Surface<'a> {
    /// The crust share, 0 oceanic to 255 continental: only OCEANIC crust is consumed at a trench.
    pub crust: &'a [u8],
    /// The distance to the nearest plate boundary, whole metres; `u32::MAX` on a one-plate body.
    pub boundary_m: &'a [u32],
    /// The node's plate.
    pub plate: &'a [u8],
    /// The plates, whose drift carries the speed share the seed drew.
    pub plates: &'a [crate::land::Plate],
    /// The rock province, whose four rocks give the erodibility.
    pub province: &'a [u8],
    /// The rain over the node, mm/yr: no rain, no denudation.
    pub rain_mm_yr: &'a [u32],
}

// ---- THE LAWS AND THEIR CALIBRATIONS ---------------------------------------------------------

/// ★ THE PRODUCTION FUNCTION (Neukum, Ivanov & Hartmann 2001, the lunar chronology): the count of
/// craters wider than one kilometre per square kilometre after an age `T` in Gyr,
/// `N(>1 km) = a·(e^{bT} − 1) + c·T`. The three constants, from the calibrated lunar samples.
pub const NEUKUM_A_PER_KM2: f64 = 5.44e-14;
/// The exponential's rate, per Gyr (the late heavy bombardment's tail).
pub const NEUKUM_B_PER_GYR: f64 = 6.93;
/// The linear rate, per km² per Gyr (the steady flux since).
pub const NEUKUM_C_PER_KM2_GYR: f64 = 8.38e-4;
/// The size-frequency slope: `N(>D) ∝ D^{−SFD_SLOPE}` (02 §4.7, the −2 power).
pub const SFD_SLOPE: f64 = 2.0;
/// ★ THE SATURATION (Gault 1970): the geometric limit `1.54·D⁻²` per km² with `D` in km, at which
/// a new crater erases an old one on average.
pub const GEOMETRIC_SATURATION_PER_KM2: f64 = 1.54;
/// The share of the geometric limit a real surface reaches before the count stops growing: the
/// lunar highlands, the oldest surface with a measured count, sit near five percent (Hartmann 1984).
pub const EMPIRICAL_SATURATION_SHARE: f64 = 0.05;
/// ★ THE ATMOSPHERIC SCREENING: the smallest crater a surface under pressure `p` collects, from
/// Earth's floor of a few hundred metres (the smallest fresh craters on Earth, Meteor Crater's
/// impactor being the smallest iron that arrives whole; stony impactors under it burst in the air)
/// scaled by `(p/p⊕)^{SCREENING_EXPONENT}`; checked on Venus, where nothing under about three
/// kilometres exists at 92 bar (Phillips et al. 1992): `250 · 92^0.55 = 3 000 m`.
pub const EARTH_SCREENING_FLOOR_M: f64 = 250.0;
/// Earth's surface pressure, pascals — the screening law's calibration.
pub const EARTH_P_SURF_PA: f64 = 101_325.0;
/// The screening's pressure exponent, fitted through Earth and Venus.
pub const SCREENING_EXPONENT: f64 = 0.55;
/// A crater narrower than this many macro nodes is the third dimension's (8f), not the lattice's.
pub const CRATER_MIN_NODES: u32 = 2;
/// No crater is wider than the body's radius: the largest lunar basin stands near that bound.
pub const CRATER_MAX_RADIUS_SHARE: f64 = 1.0;
/// ★ THE SIMPLE-TO-COMPLEX TRANSITION (Pike 1977): a bowl gives way to a flat floor with a central
/// peak past 15 km on the Moon, and the transition diameter scales as `1/g`.
pub const MOON_TRANSITION_M: f64 = 15_000.0;
/// The Moon's surface gravity, mm/s² — the transition law's calibration.
pub const MOON_GRAVITY_MM_S2: u32 = 1_620;
/// A simple crater's depth over its diameter (Pike 1977: `d = 0.196·D^{1.01}`, a fifth).
pub const SIMPLE_DEPTH_SHARE: f64 = 0.2;
/// A complex crater's depth in km, `1.044·D^{0.301}` with `D` in km (Pike 1977).
pub const COMPLEX_DEPTH_KM_COEFF: f64 = 1.044;
pub const COMPLEX_DEPTH_EXPONENT: f64 = 0.301;
/// A simple crater's rim height over its diameter (Pike 1977: `h = 0.036·D^{1.014}`).
pub const SIMPLE_RIM_SHARE: f64 = 0.036;
/// A complex crater's rim height in km, `0.236·D^{0.399}` with `D` in km (Pike 1977).
pub const COMPLEX_RIM_KM_COEFF: f64 = 0.236;
pub const COMPLEX_RIM_EXPONENT: f64 = 0.399;
/// A complex crater's flat floor reaches this share of the radius; its central peak this share.
pub const COMPLEX_FLOOR_SHARE: f64 = 0.5;
pub const COMPLEX_PEAK_SHARE: f64 = 0.2;
/// A complex crater's central peak stands this share of the depth over the floor (Pike 1977's
/// peaks reach a tenth to a third of the depth; the tenth is the stated shape).
pub const COMPLEX_PEAK_HEIGHT_SHARE: f64 = 0.1;
/// The ejecta's reach beyond the rim, in crater radii (the continuous blanket, McGetchin 1973).
pub const EJECTA_REACH_RADII: f64 = 1.0;
/// The ejecta thins as this power of the distance from the centre (McGetchin 1973: the cube).
pub const EJECTA_DECAY_POWER: f64 = 3.0;

/// ★ EARTH'S MEAN PLATE SPEED, mm/yr (the plate-motion models' own mean; calibration body EARTH):
/// the number the plate's own drift SHARE — a seed identity choice, drawn once per plate — scales
/// to give a real speed. A patch of ocean floor is made at one boundary and consumed at the next,
/// so the time it can hold a crater is the distance to that boundary over this speed. Earth's
/// ocean floor reaches about 200 Myr, which is what a half-ocean of ~10 000 km at 50 mm/yr gives.
pub const EARTH_MEAN_PLATE_SPEED_MM_YR: f64 = 50.0;
/// ★ EARTH'S OUTCROP DENUDATION RATE, metres a million years (Portenga & Bierman 2011,
/// *Understanding Earth's eroding surface with ¹⁰Be*, GSA Today 21(8), 4–10): the mean lowering of
/// bare bedrock outcrops, measured with cosmogenic beryllium. ★ UNVERIFIED as an opened source in
/// this session; the figure is the one the research report names as the calibration.
/// The outcrops the method needs carry quartz, so the calibration's own rock is the CRYSTALLINE
/// BASEMENT, and every other province scales by its erodibility against that one.
pub const EARTH_OUTCROP_DENUDATION_M_PER_MYR: f64 = 12.0;
/// ★ EARTH'S LAND MEAN RAIN, mm/yr: the denudation's other half. A surface no rain falls on loses
/// nothing, which is why an airless dry moon keeps the whole of its record without anybody
/// branching on a body kind (HR4).
///
/// ★ WHY THE RAIN AND NOT THE RUNOFF. The runoff is the Turc–Pike REMAINDER after the air has
/// taken what it can, so it collapses toward zero over any semi-arid ground — MEASURED on the home
/// planet: 374 mm of rain a year over the land left 75 mm of runoff, and the p10 node left 0.3 m
/// a million years of denudation, which kept craters for four billion years on ground that really
/// wears. Denudation is chemical as well as mechanical, and the water that does the chemistry is
/// the water that FALLS. The calibration body is Earth's land, where 750 mm a year of rain goes
/// with Portenga & Bierman's measured 12 m a million years of outcrop lowering.
pub const EARTH_LAND_MEAN_RAIN_MM_YR: f64 = 750.0;
/// The years in a megayear, for the denudation stated in metres a million years.
const YR_PER_MYR: f64 = 1.0e6;

/// The metres in a kilometre, for the laws stated in kilometres.
const M_PER_KM: f64 = 1_000.0;
/// The years in a gigayear, for the chronology stated in Gyr.
const YR_PER_GYR: f64 = 1.0e9;

/// The count of craters wider than one kilometre per square kilometre after `age_yr` — the
/// production function alone, before saturation.
#[must_use]
pub fn production_per_km2(age_yr: u64) -> Gf {
    let t = Gf::from_i64(age_yr as i64) / Gf::from_f64(YR_PER_GYR);
    Gf::from_f64(NEUKUM_A_PER_KM2) * ((Gf::from_f64(NEUKUM_B_PER_GYR) * t).exp() - Gf::ONE)
        + Gf::from_f64(NEUKUM_C_PER_KM2_GYR) * t
}

/// The count of craters wider than `diameter_m` per square kilometre after `age_yr`: the
/// production function scaled down the size-frequency slope, capped at the empirical saturation.
#[must_use]
pub fn density_per_km2(age_yr: u64, diameter_m: Gf) -> Gf {
    let d_km = diameter_m / Gf::from_f64(M_PER_KM);
    let slope = d_km.powf(-Gf::from_f64(SFD_SLOPE));
    let produced = production_per_km2(age_yr) * slope;
    let saturated = Gf::from_f64(EMPIRICAL_SATURATION_SHARE)
        * Gf::from_f64(GEOMETRIC_SATURATION_PER_KM2)
        * slope;
    produced.lesser(saturated)
}

/// The smallest crater the air lets through, metres: zero on an airless body.
#[must_use]
pub fn screening_floor_m(p_surf_pa: Option<u32>) -> Gf {
    match p_surf_pa {
        Some(p) if p > 0 => {
            let ratio = Gf::from_i64(i64::from(p)) / Gf::from_f64(EARTH_P_SURF_PA);
            Gf::from_f64(EARTH_SCREENING_FLOOR_M) * ratio.powf(Gf::from_f64(SCREENING_EXPONENT))
        }
        _ => Gf::ZERO,
    }
}

/// The smallest crater the record keeps: the wider of the screening floor and
/// [`CRATER_MIN_NODES`] macro nodes.
#[must_use]
pub fn floor_diameter_m(lattice: &MacroLattice, p_surf_pa: Option<u32>) -> Gf {
    let nodes = Gf::from_i64(i64::from(CRATER_MIN_NODES) * i64::from(lattice.cells_per_node));
    screening_floor_m(p_surf_pa).greater(nodes)
}

/// The whole count from an expected count: its floor, plus one where the fractional part beats a
/// unit draw — one draw, never a loop whose trips depend on the data.
#[must_use]
pub fn count_of(expected: Gf, unit: Gf) -> u64 {
    let whole = expected.floor();
    let extra = u64::from(expected - whole > unit);
    whole.to_i64_floor().max(0) as u64 + extra
}

/// A diameter along the size-frequency distribution from a unit draw: `D = D_floor · u^{−1/slope}`
/// with `u` in `(0, 1]`, capped at the body's radius.
#[must_use]
pub fn diameter_of(floor_m: Gf, unit: Gf, radius_m: Gf) -> Gf {
    let u = Gf::ONE - unit;
    let d = floor_m * u.powf(-Gf::ONE / Gf::from_f64(SFD_SLOPE));
    d.lesser(radius_m * Gf::from_f64(CRATER_MAX_RADIUS_SHARE))
}

/// The simple-to-complex transition diameter under a gravity, metres.
#[must_use]
pub fn transition_diameter_m(gravity_mm_s2: u32) -> Gf {
    Gf::from_f64(MOON_TRANSITION_M) * Gf::from_i64(i64::from(MOON_GRAVITY_MM_S2))
        / Gf::from_i64(i64::from(gravity_mm_s2.max(1)))
}

/// A crater's shape in metres: its depth, its rim height and whether it is complex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub radius_m: Gf,
    pub depth_m: Gf,
    pub rim_m: Gf,
    pub complex: bool,
}

/// The shape of a crater `diameter_m` wide under a gravity (Pike 1977).
#[must_use]
pub fn shape_of(diameter_m: Gf, gravity_mm_s2: u32) -> Shape {
    let complex = diameter_m > transition_diameter_m(gravity_mm_s2);
    let d_km = diameter_m / Gf::from_f64(M_PER_KM);
    let (depth_m, rim_m) = if complex {
        (
            Gf::from_f64(COMPLEX_DEPTH_KM_COEFF)
                * d_km.powf(Gf::from_f64(COMPLEX_DEPTH_EXPONENT))
                * Gf::from_f64(M_PER_KM),
            Gf::from_f64(COMPLEX_RIM_KM_COEFF)
                * d_km.powf(Gf::from_f64(COMPLEX_RIM_EXPONENT))
                * Gf::from_f64(M_PER_KM),
        )
    } else {
        (
            diameter_m * Gf::from_f64(SIMPLE_DEPTH_SHARE),
            diameter_m * Gf::from_f64(SIMPLE_RIM_SHARE),
        )
    };
    Shape {
        radius_m: diameter_m * Gf::HALF,
        depth_m,
        rim_m,
        complex,
    }
}

/// ★ THE PROFILE, one scalar kernel: the height change in metres at a distance `r_m` from the
/// centre. Inside the rim a simple crater is a paraboloid from `−depth` at the centre to `+rim` at
/// the rim; a complex crater has a flat floor to [`COMPLEX_FLOOR_SHARE`] of the radius, a wall
/// rising to the rim, and a central peak. Beyond the rim the ejecta thins as the cube of the
/// distance out to one radius past the rim, and is an EXACT ZERO further out (the scan law's first
/// condition, 02 §4.0).
#[must_use]
pub fn profile_m(shape: &Shape, r_m: Gf) -> Gf {
    let x = r_m / shape.radius_m;
    let reach = Gf::ONE + Gf::from_f64(EJECTA_REACH_RADII);
    if x >= reach {
        return Gf::ZERO;
    }
    if x >= Gf::ONE {
        return shape.rim_m * (Gf::ONE / x).powf(Gf::from_f64(EJECTA_DECAY_POWER));
    }
    if !shape.complex {
        return -shape.depth_m + (shape.depth_m + shape.rim_m) * x * x;
    }
    let floor = Gf::from_f64(COMPLEX_FLOOR_SHARE);
    let peak = Gf::from_f64(COMPLEX_PEAK_SHARE);
    if x < peak {
        return -shape.depth_m
            + shape.depth_m * Gf::from_f64(COMPLEX_PEAK_HEIGHT_SHARE) * (Gf::ONE - x / peak);
    }
    if x < floor {
        return -shape.depth_m;
    }
    -shape.depth_m + (shape.depth_m + shape.rim_m) * (x - floor) / (Gf::ONE - floor)
}

/// ★ THE PLATE CLOCK at a node, years: how long a patch of OCEANIC crust can stand before the
/// plate carrying it reaches a boundary — the distance to that boundary over the plate's own
/// speed, which is the drift share the seed drew for that plate times Earth's mean plate speed.
/// Continental crust is not consumed, and a one-plate body has no boundary to reach, so both
/// answer `u64::MAX`: no clock at all.
///
/// **Example.** A node on the home planet's ocean floor stands 3 000 km from its trench on a plate
/// drifting at 40 mm a year: it has 75 million years left, and no crater older than that survives
/// on it. The shield across the sea is continental and keeps whatever the rain leaves it.
#[must_use]
pub fn plate_clock_yr(surface: &Surface, node: u32) -> u64 {
    let i = node as usize;
    if surface.crust[i] >= 128 {
        return u64::MAX;
    }
    let plate = surface.plates.get(usize::from(surface.plate[i]));
    let Some(plate) = plate else {
        return u64::MAX;
    };
    let d = plate.drift;
    let share = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let speed_mm_yr = share * Gf::from_f64(EARTH_MEAN_PLATE_SPEED_MM_YR);
    if speed_mm_yr <= Gf::ZERO {
        return u64::MAX;
    }
    let mm = Gf::from_i64(i64::from(surface.boundary_m[i])) * Gf::from_f64(M_PER_KM);
    (mm / speed_mm_yr).to_i64_floor().max(0) as u64
}

/// ★ THE DENUDATION RATE at a node, metres a million years: Earth's measured outcrop rate scaled
/// by the node's own rock (its province's erodibility against the crystalline basement's, which is
/// the rock the calibration was measured on) and by its own rain against Earth's land mean. A node
/// no rain falls on loses nothing, whatever its rock.
#[must_use]
pub fn denudation_m_per_myr(surface: &Surface, node: u32) -> Gf {
    let i = node as usize;
    let Some(province) = crate::strata::Province::from_code(surface.province[i]) else {
        return Gf::ZERO;
    };
    let reference = crate::strata::Province::CrystallineBasement.erodibility_q8();
    let rock = Gf::from_i64(i64::from(province.erodibility_q8()))
        / Gf::from_i64(i64::from(reference.max(1)));
    let water =
        Gf::from_i64(i64::from(surface.rain_mm_yr[i])) / Gf::from_f64(EARTH_LAND_MEAN_RAIN_MM_YR);
    Gf::from_f64(EARTH_OUTCROP_DENUDATION_M_PER_MYR) * rock * water
}

/// ★ THE SURFACE'S CRATER RETENTION AGE at a node for a crater `depth_m` deep, years: the LESSER
/// of the body's own age, the plate clock, and the time the denudation needs to take the crater's
/// whole depth away. THE PRODUCTION FUNCTION IS UNTOUCHED — only the age it integrates over moves,
/// and it moves because the surface itself is new, which is exactly what Earth's record shows: 45 %
/// of its 190 structures are younger than 200 Ma, 4.4 % of its history, because the ocean floor
/// resets on a 200 Myr clock and a craton does not.
///
/// **Example.** A 20 km crater is 2.5 km deep. On the home planet's shield, eroding at about a
/// metre a million years, it stands for two billion years. On the shelf, eroding forty times
/// faster, it is gone in fifty million. On the ocean floor the trench takes the whole surface
/// first. On the airless moon nothing takes any of it, and the moon's count does not move.
#[must_use]
pub fn retention_age_yr(surface: &Surface, node: u32, depth_m: Gf, age_yr: u64) -> u64 {
    let mut keep = age_yr.min(plate_clock_yr(surface, node));
    let rate = denudation_m_per_myr(surface, node);
    if rate > Gf::ZERO {
        let years = (depth_m / rate * Gf::from_f64(YR_PER_MYR))
            .to_i64_floor()
            .max(0) as u64;
        keep = keep.min(years);
    }
    keep
}

/// ★ THE AGE OF ONE KEPT CRATER, years before now: the formation time `t` at which the production
/// function has delivered `unit` of the craters a surface of retention age `retention_yr` holds,
/// `N(t) = unit · N(retention)`. Found by a FIXED number of bisections over the monotone
/// production function, so the answer is the same on every host and no loop's trips depend on the
/// data. Where the linear term rules — any surface much younger than the late bombardment's tail —
/// the answer is flat in time, and the histogram's skew toward the recent comes from the SURFACE,
/// not from the chronology.
#[must_use]
pub fn formation_age_yr(retention_yr: u64, unit: Gf) -> u64 {
    let target = production_per_km2(retention_yr) * unit;
    let (mut lo, mut hi) = (0u64, retention_yr);
    let mut step = 0;
    while step < AGE_BISECTIONS {
        let mid = lo + (hi - lo) / 2;
        if production_per_km2(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
        step += 1;
    }
    hi
}

/// The bisections the formation age takes: enough to split a five-billion-year span to the year.
pub const AGE_BISECTIONS: u32 = 64;

/// ★ THE CANDIDATE COUNT: how many craters the production function alone would stamp on `body`, at
/// the body's whole age and with no surface renewal at all. The retained record is drawn from
/// exactly these candidates, so a body whose surface never renews keeps every one of them and its
/// count cannot move (HR4: one machinery, no branch on a body kind).
#[must_use]
pub fn expected_count(body: &BodyDefinition, lattice: &MacroLattice, words: &SolveWords) -> u64 {
    let mut rng = SplitMix64::new(child_seed(body.seed(), salt::CRATERS, 0));
    let floor_m = floor_diameter_m(lattice, words.p_surf_pa);
    let area_km2 = Gf::from_i64(
        (0..lattice.node_count() as u32)
            .map(|n| lattice.area_m2(n))
            .sum::<u64>() as i64,
    ) / Gf::from_f64(M_PER_KM * M_PER_KM);
    let expected = density_per_km2(words.age_yr, floor_m) * area_km2;
    count_of(expected, draw_unit(&mut rng))
}

/// ★ THE CRATER POPULATION of `body` over `lattice`: the expected count from the laws over the
/// body's whole area, its size along the size-frequency distribution, its place a draw of a face
/// and a cell — every draw on the body's own crater stream in a frozen order (a count's fraction,
/// then per crater a face, a row, a column, a size) — sorted largest first, ties by index.
#[must_use]
pub fn crater_population(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &SolveWords,
    surface: &Surface,
) -> Vec<Crater> {
    let mut rng = SplitMix64::new(child_seed(body.seed(), salt::CRATERS, 0));
    // ★ THE RETENTION READS ITS OWN STREAM (index 1 of the body's crater salt), so a surface that
    // renews nothing draws the same faces, the same cells and the same sizes it always drew and
    // its record is byte for byte the one it had.
    let mut keep_rng = SplitMix64::new(child_seed(body.seed(), salt::CRATERS, 1));
    let floor_m = floor_diameter_m(lattice, words.p_surf_pa);
    let area_km2 = Gf::from_i64(
        (0..lattice.node_count() as u32)
            .map(|n| lattice.area_m2(n))
            .sum::<u64>() as i64,
    ) / Gf::from_f64(M_PER_KM * M_PER_KM);
    let expected = density_per_km2(words.age_yr, floor_m) * area_km2;
    let count = count_of(expected, draw_unit(&mut rng));
    let radius_m = Gf::from_f64(body.ladder().radius_m());
    let edge = u64::from(lattice.edge);
    let gravity = body.facts().gravity_mm_s2;
    let mut craters: Vec<Crater> = Vec::new();
    for _ in 0..count {
        let face = Face::from_index(rng.range_u64(0, 6) as u8).unwrap_or(Face::NegZ);
        let i = rng.range_u64(0, edge) as i32;
        let j = rng.range_u64(0, edge) as i32;
        let diameter = diameter_of(floor_m, draw_unit(&mut rng), radius_m);
        let node = lattice.index(face, i, j);
        // ★ THE SURFACE DECIDES WHETHER THE RECORD KEPT IT. The production function over the
        // surface's own retention age against the production over the body's whole age is the
        // share of the candidates this ground still shows.
        let depth_m = shape_of(diameter, gravity).depth_m;
        let retention = retention_age_yr(surface, node, depth_m, words.age_yr);
        let whole = density_per_km2(words.age_yr, diameter);
        let kept = density_per_km2(retention, diameter);
        let share = if whole > Gf::ZERO {
            (kept / whole).clamp(Gf::ZERO, Gf::ONE)
        } else {
            Gf::ZERO
        };
        if draw_unit(&mut keep_rng) >= share {
            continue;
        }
        craters.push(Crater {
            node,
            diameter_m: diameter.floor().to_i64_floor() as u32,
            age_yr: formation_age_yr(retention, draw_unit(&mut keep_rng)),
        });
    }
    craters.sort_by(|a, b| b.diameter_m.cmp(&a.diameter_m).then(a.node.cmp(&b.node)));
    craters
}

/// ★ APPLY THE CRATERS to the node heights (sixteenths of a metre), in list order, each on the
/// field the earlier ones left. The nodes a crater reaches are found by a breadth-first walk from
/// its centre over the neighbours, kept while the chord to the centre is inside the ejecta's
/// reach; a stamp per node marks the current crater's walk, so no crater scans the lattice.
pub fn apply_craters(
    z: &mut [i32],
    lattice: &MacroLattice,
    craters: &[Crater],
    gravity_mm_s2: u32,
) {
    let mut stamp = vec![0u32; lattice.node_count()];
    let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let mut frontier: Vec<u32> = Vec::new();
    let mut next: Vec<u32> = Vec::new();
    for (k, crater) in craters.iter().enumerate() {
        let mark = k as u32 + 1;
        let shape = shape_of(Gf::from_i64(i64::from(crater.diameter_m)), gravity_mm_s2);
        let reach_m = shape.radius_m * (Gf::ONE + Gf::from_f64(EJECTA_REACH_RADII));
        frontier.clear();
        frontier.push(crater.node);
        stamp[crater.node as usize] = mark;
        while !frontier.is_empty() {
            next.clear();
            for &node in &frontier {
                let r_m = Gf::from_i64(i64::from(lattice.chord_m(crater.node, node)));
                z[node as usize] += (profile_m(&shape, r_m) * steps).to_i64_floor() as i32;
                for m in lattice.neighbours(node) {
                    if m == NO_NODE || stamp[m as usize] == mark {
                        continue;
                    }
                    stamp[m as usize] = mark;
                    if Gf::from_i64(i64::from(lattice.chord_m(crater.node, m))) < reach_m {
                        next.push(m);
                    }
                }
            }
            std::mem::swap(&mut frontier, &mut next);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::{home_moon, home_moon_solve_words};

    fn gf(v: f64) -> Gf {
        Gf::from_f64(v)
    }

    /// The Moon's numbers: after four billion years the production function gives about 0.06
    /// craters wider than a kilometre per square kilometre, under the saturation's 0.077 (the
    /// lunar highlands); a billion years later the exponential's tail runs past the cap and the
    /// cap binds; a young surface of ten million years stands far under both.
    #[test]
    fn the_production_function_on_the_moons_numbers() {
        let four = production_per_km2(4_000_000_000).to_f64();
        assert!((0.055..0.070).contains(&four), "{four}");
        let produced = density_per_km2(4_000_000_000, gf(1_000.0)).to_f64();
        assert!((produced - four).abs() < 1e-12, "{produced}");
        let capped = density_per_km2(5_000_000_000, gf(1_000.0)).to_f64();
        assert!((capped - 0.077).abs() < 1e-9, "{capped}");
        let young = density_per_km2(10_000_000, gf(1_000.0)).to_f64();
        assert!((8.0e-6..9.0e-6).contains(&young), "{young}");
        // Ten times wider, a hundred times fewer.
        let wide = density_per_km2(5_000_000_000, gf(10_000.0)).to_f64();
        assert!((wide * 100.0 - capped).abs() < 1e-9);
    }

    /// The screening: airless keeps everything, Earth keeps only craters over about 250 m, Venus
    /// nothing under three kilometres.
    #[test]
    fn the_screening_on_earth_and_venus() {
        assert_eq!(screening_floor_m(None), Gf::ZERO);
        assert_eq!(screening_floor_m(Some(0)), Gf::ZERO);
        let earth = screening_floor_m(Some(101_325)).to_f64();
        assert!((earth - 250.0).abs() < 1e-9, "{earth}");
        let venus = screening_floor_m(Some(9_200_000)).to_f64();
        assert!((2_800.0..3_200.0).contains(&venus), "{venus}");
    }

    /// The floor is the wider of the screening and two nodes: on the moon's lattice two nodes
    /// (16 384 m) beat Earth's 250 m; a pressure a hundred times Venus's beats even that.
    #[test]
    fn the_floor_is_the_wider_of_screening_and_two_nodes() {
        let lattice = MacroLattice::of(&home_moon()).expect("a lattice");
        assert_eq!(floor_diameter_m(&lattice, None).to_f64(), 16_384.0);
        assert_eq!(floor_diameter_m(&lattice, Some(101_325)).to_f64(), 16_384.0);
        let venus = floor_diameter_m(&lattice, Some(920_000_000)).to_f64();
        assert!(venus > 16_384.0, "{venus}");
    }

    /// The count from an expected value: the floor, plus one where the fraction beats the draw;
    /// a negative expectation counts nothing.
    #[test]
    fn the_count_takes_the_fraction_by_one_draw() {
        assert_eq!(count_of(gf(3.7), gf(0.5)), 4);
        assert_eq!(count_of(gf(3.7), gf(0.9)), 3);
        assert_eq!(count_of(gf(0.0), gf(0.5)), 0);
        assert_eq!(count_of(gf(-1.0), gf(0.5)), 0);
    }

    /// The size along the distribution: the smallest draw gives the floor, a draw near one gives a
    /// crater capped at the radius, and a quarter draw gives the floor over the root of three
    /// quarters.
    #[test]
    fn the_diameter_follows_the_size_frequency_slope() {
        let floor = gf(16_384.0);
        let r = gf(354_632.9);
        assert_eq!(diameter_of(floor, Gf::ZERO, r), floor);
        assert_eq!(diameter_of(floor, gf(0.999_999_999), r), r);
        let q = diameter_of(floor, gf(0.25), r).to_f64();
        assert!((q - 16_384.0 / 0.75f64.sqrt()).abs() < 1e-6, "{q}");
    }

    /// The shapes: a 10 km crater on the Moon is simple — 2 km deep, a 360 m rim — and on Earth
    /// (transition 2.5 km) complex — about 2.1 km deep with a 590 m rim; the transition scales as
    /// one over gravity.
    #[test]
    fn the_shapes_on_the_moon_and_on_earth() {
        assert!((transition_diameter_m(9_810).to_f64() - 2_477.0).abs() < 1.0);
        assert_eq!(transition_diameter_m(MOON_GRAVITY_MM_S2).to_f64(), 15_000.0);
        assert!(transition_diameter_m(0).to_f64() > 15_000.0);
        let moon = shape_of(gf(10_000.0), MOON_GRAVITY_MM_S2);
        assert!(!moon.complex);
        assert_eq!(moon.depth_m.to_f64(), 2_000.0);
        assert_eq!(moon.rim_m.to_f64(), 360.0);
        assert_eq!(moon.radius_m.to_f64(), 5_000.0);
        let earth = shape_of(gf(10_000.0), 9_810);
        assert!(earth.complex);
        let depth = earth.depth_m.to_f64();
        assert!((2_050.0..2_150.0).contains(&depth), "{depth}");
        let rim = earth.rim_m.to_f64();
        assert!((570.0..610.0).contains(&rim), "{rim}");
    }

    /// The profile: the bowl's centre, the rim, the ejecta's decay, the exact zero past the reach;
    /// the complex crater's peak, floor and wall.
    #[test]
    fn the_profile_at_every_station() {
        let simple = shape_of(gf(10_000.0), MOON_GRAVITY_MM_S2);
        assert_eq!(profile_m(&simple, Gf::ZERO).to_f64(), -2_000.0);
        assert_eq!(profile_m(&simple, gf(5_000.0)).to_f64(), 360.0);
        let half = profile_m(&simple, gf(2_500.0)).to_f64();
        assert!((half - (-2_000.0 + 2_360.0 * 0.25)).abs() < 1e-9, "{half}");
        let skirt = profile_m(&simple, gf(7_500.0)).to_f64();
        assert!((skirt - 360.0 / (1.5 * 1.5 * 1.5)).abs() < 1e-9, "{skirt}");
        assert_eq!(profile_m(&simple, gf(10_000.0)), Gf::ZERO);
        assert_eq!(profile_m(&simple, gf(50_000.0)), Gf::ZERO);
        let complex = shape_of(gf(10_000.0), 9_810);
        let d = complex.depth_m.to_f64();
        let centre = profile_m(&complex, Gf::ZERO).to_f64();
        assert!((centre - (-d + 0.1 * d)).abs() < 1e-9, "{centre}");
        assert_eq!(profile_m(&complex, gf(1_500.0)).to_f64(), -d);
        assert_eq!(
            profile_m(&complex, gf(5_000.0)).to_f64(),
            complex.rim_m.to_f64()
        );
        let wall = profile_m(&complex, gf(3_750.0)).to_f64();
        assert!(wall > -d);
        assert!(wall < complex.rim_m.to_f64());
    }

    /// The land and the climate a [`Surface`] reads, for one body under one set of words.
    fn surface_rows(
        body: &BodyDefinition,
        lattice: &MacroLattice,
        words: &SolveWords,
    ) -> (crate::land::Land, crate::climate::Climate) {
        let land = crate::land::initial_land(body, lattice, &words.land());
        let climate = crate::climate::climate(body, lattice, words, &land.z, land.sea_z);
        (land, climate)
    }

    /// A [`Surface`] over the rows [`surface_rows`] built.
    fn surface_of<'a>(
        land: &'a crate::land::Land,
        climate: &'a crate::climate::Climate,
    ) -> Surface<'a> {
        Surface {
            crust: &land.crust,
            boundary_m: &land.boundary_m,
            plate: &land.plate,
            plates: &land.plates,
            province: &land.province,
            rain_mm_yr: &climate.rain_mm_yr,
        }
    }

    /// ★ THE DRIVER ON THE HOME MOON: airless and five billion years old, it collects a few hundred
    /// craters wider than two nodes, largest first, every node valid; applying them changes the
    /// heights only within each crater's reach, the bowl stands under the rim, and the same seed
    /// draws the same record.
    #[test]
    fn the_home_moon_collects_its_craters_largest_first() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let words = home_moon_solve_words();
        let (land, clim) = surface_rows(&moon, &lattice, &words);
        let surface = surface_of(&land, &clim);
        let craters = crater_population(&moon, &lattice, &words, &surface);
        assert!((100..2_000).contains(&craters.len()), "{}", craters.len());
        for pair in craters.windows(2) {
            assert!(pair[0].diameter_m >= pair[1].diameter_m);
        }
        for c in &craters {
            assert!((c.node as usize) < lattice.node_count());
            assert!(c.diameter_m >= 16_384);
            assert!(f64::from(c.diameter_m) <= moon.ladder().radius_m());
        }
        assert_eq!(
            crater_population(&moon, &lattice, &words, &surface),
            craters
        );
        // One crater applied alone: the bowl under the rim, nothing changed past its reach.
        let one = craters[craters.len() / 2];
        let mut z = vec![0i32; lattice.node_count()];
        apply_craters(&mut z, &lattice, &[one], moon.facts().gravity_mm_s2);
        let shape = shape_of(
            Gf::from_i64(i64::from(one.diameter_m)),
            moon.facts().gravity_mm_s2,
        );
        let reach = (shape.radius_m * Gf::TWO).to_f64();
        assert!(z[one.node as usize] < 0, "the bowl is a hole");
        let mut rim_max = i32::MIN;
        for (k, &h) in z.iter().enumerate() {
            let r = f64::from(lattice.chord_m(one.node, k as u32));
            if r >= reach {
                assert_eq!(h, 0, "node {k} at {r} m changed past the reach");
            }
            if h > rim_max {
                rim_max = h;
            }
        }
        assert!(rim_max > 0, "a raised rim");
        assert!(z[one.node as usize] < rim_max);
        // The whole record, applied: it moves the heights.
        let mut all = vec![0i32; lattice.node_count()];
        apply_craters(&mut all, &lattice, &craters, moon.facts().gravity_mm_s2);
        assert!(all.iter().any(|&h| h != 0));
    }

    /// The moon's lattice under the home planet's air and age: the screening floor (250 m) stays
    /// under two nodes, so the count is the same law's; a Venus-like pressure raises the floor and
    /// thins the record; a pressure a hundred times Venus's raises the floor over two nodes.
    #[test]
    fn a_thick_air_thins_the_record() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let dry_words = home_moon_solve_words();
        let (land, clim) = surface_rows(&moon, &lattice, &dry_words);
        let surface = surface_of(&land, &clim);
        let airless = crater_population(&moon, &lattice, &dry_words, &surface).len();
        // The SAME dry surface under Earth's pressure: the screening floor alone is read here, so
        // the record's size is the only thing the air can change.
        let mut earth_air = home_moon_solve_words();
        earth_air.p_surf_pa = Some(101_325);
        let under_earth_air = crater_population(&moon, &lattice, &earth_air, &surface).len();
        assert_eq!(under_earth_air, airless);
        let mut venus_air = home_moon_solve_words();
        venus_air.p_surf_pa = Some(920_000_000);
        let craters = crater_population(&moon, &lattice, &venus_air, &surface);
        assert!(craters.len() < airless, "{} under {airless}", craters.len());
        let floor = floor_diameter_m(&lattice, venus_air.p_surf_pa).to_f64();
        for c in &craters {
            assert!(f64::from(c.diameter_m) >= floor.floor());
        }
    }

    /// ★ GATE G-CRATER's LAW, ON TWO BODIES (ruling B2 step 2). The airless dry moon renews
    /// nothing — no water to wear it, no plate to swallow it — so it keeps EVERY candidate the
    /// production function drew, and its count cannot move. The same lattice under the home
    /// planet's air and water keeps a small share of them, because the rain takes the rest. THE
    /// READING THAT PROVES THE LAW READS RESURFACING AND NOT A BODY KIND (HR4).
    #[test]
    fn the_retention_thins_a_wet_record_and_never_the_airless_moons() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let dry_words = home_moon_solve_words();
        let (land, clim) = surface_rows(&moon, &lattice, &dry_words);
        let dry = surface_of(&land, &clim);
        let kept = crater_population(&moon, &lattice, &dry_words, &dry);
        assert_eq!(
            kept.len() as u64,
            expected_count(&moon, &lattice, &dry_words),
            "an airless dry surface keeps every candidate"
        );
        let oldest = kept.iter().map(|c| c.age_yr).max().expect("a record");
        assert!(oldest <= dry_words.age_yr, "{oldest}");
        assert!(
            oldest * 2 > dry_words.age_yr,
            "the moon's record reaches back to the bombardment: {oldest}"
        );
        // ★ THE SAME GROUND UNDER EARTH'S OWN LAND MEAN RAIN, and nothing else changed: the rain
        // takes most of the record away, because a crater is gone once the ground it sits in has
        // fallen by the crater's own depth.
        let wet_rain = vec![EARTH_LAND_MEAN_RAIN_MM_YR as u32; lattice.node_count()];
        let wet = Surface {
            rain_mm_yr: &wet_rain,
            ..surface_of(&land, &clim)
        };
        let wet_kept = crater_population(&moon, &lattice, &dry_words, &wet);
        assert!(
            wet_kept.len() * 10 < kept.len(),
            "the rain takes most of the record: {} of {}",
            wet_kept.len(),
            kept.len()
        );
        for c in &wet_kept {
            assert!(c.age_yr <= dry_words.age_yr);
        }
    }

    /// ★ THE TWO CLOCKS, each on its own: the plate's and the rain's. Continental crust is never
    /// consumed; a plate that does not move never reaches a boundary; a node no water reaches
    /// loses nothing; a province byte nobody owns answers no denudation rather than a guess.
    #[test]
    fn the_retention_reads_the_plate_and_the_rain() {
        let plates = vec![
            crate::land::Plate {
                site: [Gf::ONE, Gf::ZERO, Gf::ZERO],
                drift: [Gf::ZERO, Gf::HALF, Gf::ZERO],
                affinity: Gf::ZERO,
                age: Gf::ZERO,
                crust_scatter: Gf::ONE,
            },
            crate::land::Plate {
                site: [Gf::ONE, Gf::ZERO, Gf::ZERO],
                drift: [Gf::ZERO, Gf::ZERO, Gf::ZERO],
                affinity: Gf::ZERO,
                age: Gf::ZERO,
                crust_scatter: Gf::ONE,
            },
        ];
        let crust = [0u8, 255, 0, 0];
        let boundary_m = [3_000_000u32, 3_000_000, 3_000_000, 3_000_000];
        let plate = [0u8, 0, 1, 9];
        let province = [
            crate::strata::Province::FlatShelf.code(),
            crate::strata::Province::CrystallineBasement.code(),
            crate::strata::Province::CrystallineBasement.code(),
            9,
        ];
        let rain_mm_yr = [750u32, 0, 750, 750];
        let s = Surface {
            crust: &crust,
            boundary_m: &boundary_m,
            plate: &plate,
            plates: &plates,
            province: &province,
            rain_mm_yr: &rain_mm_yr,
        };
        // Oceanic crust on a plate drifting at half Earth's mean speed, 3 000 km from its
        // boundary: 3 000 km at 25 mm a year is 120 million years.
        assert_eq!(plate_clock_yr(&s, 0), 120_000_000);
        // Continental crust, a plate that does not move, and a plate nobody named: no clock.
        assert_eq!(plate_clock_yr(&s, 1), u64::MAX);
        assert_eq!(plate_clock_yr(&s, 2), u64::MAX);
        assert_eq!(plate_clock_yr(&s, 3), u64::MAX);
        // The rain: the shelf wears four times faster than the basement the rate was measured on,
        // a node no water reaches wears not at all, and an unnamed province answers zero.
        let shelf = denudation_m_per_myr(&s, 0).to_f64();
        let shield = denudation_m_per_myr(&s, 2).to_f64();
        assert!(shelf > shield, "{shelf} over {shield}");
        assert!(
            (shield - EARTH_OUTCROP_DENUDATION_M_PER_MYR).abs() < 0.001,
            "{shield}"
        );
        assert_eq!(denudation_m_per_myr(&s, 1), Gf::ZERO);
        assert_eq!(denudation_m_per_myr(&s, 3), Gf::ZERO);
        // The retention: the lesser of the body's age, the plate's clock and the rain's.
        let age = 5_000_000_000u64;
        assert_eq!(retention_age_yr(&s, 1, Gf::from_f64(1_000.0), age), age);
        let shield_keeps = retention_age_yr(&s, 2, Gf::from_f64(1_000.0), age);
        assert_eq!(shield_keeps, 83_333_333);
        let ocean_keeps = retention_age_yr(&s, 0, Gf::from_f64(100_000.0), age);
        assert_eq!(ocean_keeps, 120_000_000, "the trench beats the rain");
    }

    /// The formation age: the production function's own inverse over the retention age, rising
    /// with the draw, zero at zero and the whole retention at the top.
    #[test]
    fn the_formation_age_walks_the_chronology() {
        let span = 400_000_000u64;
        assert_eq!(formation_age_yr(span, Gf::ZERO), 0);
        assert_eq!(formation_age_yr(span, Gf::ONE), span);
        let half = formation_age_yr(span, Gf::HALF);
        assert!(half > 0 && half < span, "{half}");
        let quarter = formation_age_yr(span, Gf::from_f64(0.25));
        assert!(quarter < half, "{quarter} under {half}");
        assert_eq!(formation_age_yr(0, Gf::HALF), 0);
    }
}
