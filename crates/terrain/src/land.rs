//! ★ THE INITIAL LAND (the landform arc, slice 8c stage C2; `slice_8c_design.md` §3; the
//! investigation's `02_planet_layout.md` §4.0–4.3 and §7.1).
//!
//! The land before water touched it, ONCE per node of the macro lattice, as a closed form of the
//! seed and the charter:
//!
//! - **the plates** (L1): the seed places plate sites on the sphere; every node belongs to the
//!   nearest site (a Voronoi partition — "each point goes to its nearest site"); the two nearest
//!   sites' drifts say whether their boundary pushes together, pulls apart or slides past; the
//!   node keeps its plate, its distance to that boundary and the boundary's kind;
//! - **isostasy on a crust field, with the water load** (L2): the crust floats on the mantle like a
//!   raft on water — a thick light raft rides high — and its thickness is a smooth FIELD (a slow
//!   noise plus a per-plate character), so a continent's edge falls INSIDE a plate: a passive
//!   margin, the ordinary coast. Where the sea stands the water's weight pushes the floor down;
//! - **orogeny under the relief law** (L3): a belt where plates collide, a trench where an ocean
//!   floor dives, a ridge where an ocean floor is made, a rift where a continent tears — one
//!   machinery, one profile per boundary kind, every profile an exact zero outside its width;
//! - **the sea**: the level that holds the charter's water inventory over THIS field, with the load
//!   inside the bisection (02 §7.1) — ruling T8's second hump, finally with a basin to sit in.
//!
//! Every physical number is a law with a calibration body (ruling T9), named beside its constant.
//! A seed draw exists only for the identity choices: where a site sits, which way it drifts, how
//! continental a plate is, how old a belt is, and the crust's NAMED scatter. Floats run under the
//! fence and are floored ONCE per node into sixteenths of a metre; the solve reads integers.
//!
//! ★ GENERATOR 26 (rulings B59–B61, 2026-10-06; each law measured first in the bench's land copy,
//! `vd_bins::land_copy`, against the convergence gate):
//!
//! - **a node is its cell's mean** ([`CELL_SUBCELLS`]): every term that depends on a plate — its
//!   continental character, its crust's scatter, its boundary's profile — is read on the node's
//!   sub-cells and averaged, so a plate's step falls inside the cells it crosses and never whole on
//!   one node row (W16, B16: a node stands for its cell). MEASURED before: 21.9 % of the boundary
//!   coastline stepped along the node rows;
//! - **the margins by kind** ([`margin_width_m`]): on the side of a boundary whose crust column
//!   stands higher, the crust share and the crust's scatter ramp from the other plate's at the
//!   boundary to the plate's own over the margin's published width;
//! - **the sourced spread** ([`EARTH_CONTINENT_SPREAD`]): each plate's crust scatter on Earth's
//!   stable continents' measured spread;
//! - **the older landscape's relief** ([`older_landscape_m`]): the relief a plate carried when its
//!   erosion began, by the continents' published spectrum, weighted by each node's crust share and
//!   read by the freeboard law. MEASURED before: 87 % of the first land stood on an exact flat;
//! - **the crust share read continuously**: the room's profiles blend the continental and the
//!   oceanic profile by each sub-cell's own share, in place of a yes-or-no test at one half.
//!
//! **Example.** The home planet draws fourteen plates from its seed. A pilot flies west: the
//! ground rises for 300 km into a range, drops into a valley of old sea floor, and rises again
//! into a line of islands. One boundary where two plates push together drew all three, and the
//! crust field put the beach the pilot landed on 200 km from any boundary at all.

// ★ A LAND MAY DIVIDE: it runs once per body on the CPU, never in a kernel (03 §5.3).
#![allow(
    clippy::integer_division,
    clippy::modulo_arithmetic,
    reason = "the initial land runs on the CPU once per body, never in a kernel; integer-exact on every host"
)]

use vd_recipe::Gi;
use vd_recipe::bend::DIR_BITS;
use vd_recipe::height::{AMP_BITS, Octave, relief};
use vd_recipe::noise::NOISE_BITS;
use vd_seed::bend::{Face, direction};
use vd_seed::rng::{SplitMix64, child_seed};

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::body::{BodyDefinition, draw_unit, frequency_of, salt};
use crate::gf::Gf;
use crate::macro_lattice::{MacroLattice, NO_NODE};
use crate::solve::Z_STEPS_PER_M;
use crate::strata::Province;

/// ★ THE CHARTER WORDS THE INITIAL LAND READS, as whole numbers, stated by the body's own realm
/// (ruling V13 L12): the water inventory and the lithosphere's elastic thickness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LandWords {
    /// The body's whole water inventory, whole cubic kilometres (zero on a dry body).
    pub water_km3: u64,
    /// The lithosphere's effective elastic thickness, whole metres (the charter's derived word).
    pub elastic_thickness_m: u32,
}

// ---- THE LAWS AND THEIR CALIBRATIONS ---------------------------------------------------------

/// ★ THE ISOSTATIC STEP's law: the crust thicknesses scale with the RELIEF LAW's CAP — the lesser of
/// the strength arm `σ_y/(ρ_c·g)` (02 §4.2's `1/g` at fixed composition) and the shape arm
/// `0.077·R` (which keeps a small body's crust from growing past what its size can hold: the `1/g`
/// law alone would give a 350 km moon a thousand kilometres of crust). Calibration: Earth, whose
/// cap is Everest's 8 850 m and whose dry step is 4 455 m.
pub const EARTH_RELIEF_CAP_M: f64 = 8_850.0;
/// The mantle's density, kg/m³ — the raft floats on this (calibration: Earth's upper mantle).
pub const MANTLE_DENSITY_KGM3: f64 = 3_300.0;
/// Continental crust, kg/m³ (Earth).
pub const CONTINENTAL_CRUST_DENSITY_KGM3: f64 = 2_800.0;
/// Oceanic crust, kg/m³ (Earth).
pub const OCEANIC_CRUST_DENSITY_KGM3: f64 = 2_900.0;
/// Sea water, kg/m³ (Earth).
pub const WATER_DENSITY_KGM3: f64 = 1_030.0;
/// Continental crust thickness on Earth, metres; a body's is this times its relief cap over
/// Earth's ([`EARTH_RELIEF_CAP_M`]).
pub const EARTH_CONTINENTAL_CRUST_M: f64 = 35_000.0;
/// Oceanic crust thickness on Earth, metres; the same scaling.
pub const EARTH_OCEANIC_CRUST_M: f64 = 7_000.0;
/// ★ THE NAMED SCATTER of a plate's continental thickness around the computed value (ruling T9: a
/// named scatter around a computed value is lawful; a craton is thicker than a young platform),
/// SOURCED (ruling B59 item 3, generator 26): Earth's stable continents' spread of their mean
/// crust, σ 1.4 km on 40.4 km over seven continents (the research's reading of CRUST1.0, Laske et
/// al. 2013; `docs/investigation/2026-10-06/platform_margin_research.md`, Appendix A). A plate
/// draws its factor uniformly with that σ: `1 ± σ·√3`. MEASURED before: the range [0.7, 1.4]
/// named no source and is about 5.6 times Earth's spread; where two continental plates met they
/// stood in steps of 1 939 m (10 m on the first land with the margins).
pub const EARTH_CONTINENT_SPREAD: f64 = 1.4 / 40.4;
/// ★ THE CELL'S SUB-CELLS a side (ruling B59 item 2, generator 26): a node holds the mean of its
/// plate terms over `K × K` sub-cells of the recipe's own bend on a lattice `K` times finer, so the
/// sub-cells tile the cell exactly. A resolution of the mean, never a knob: 1 km sub-cells on the
/// home planet's 8 km nodes.
pub const CELL_SUBCELLS: u32 = 8;
/// ★ THE ANDEAN INTERFACE'S DIP, degrees: the seismogenic zone's mean dip over Earth's subduction
/// zones, 23 ± 8° (Heuret, Lallemand, Funiciello, Piromallo & Faccenna 2011, G³ 12, Q01004).
pub const ANDEAN_DIP_DEG: f64 = 23.0;
/// The tangent of [`ANDEAN_DIP_DEG`], stated as a number because the fence offers no tangent (a
/// test computes it again by the fenced series).
pub const ANDEAN_DIP_TAN: f64 = 0.424_474_816_209_604_76;
/// ★ THE NECKING WAVELENGTH over the strong layer's thickness: 3 to 4 (Fletcher & Hallet 1983, as
/// Chenin, Schmalholz, Manatschal & Karner 2018 apply it, their Appendix A2); the middle of the
/// stated range.
pub const NECKING_WAVELENGTH_RATIO: f64 = 3.5;
/// ★ THE BULK EXTENSION when the neck forms: 12 to 30 % (the same source); the middle of the stated
/// range.
pub const NECKING_STRETCH: f64 = 0.21;
/// ★ A TRANSFORM MARGIN'S NECKING WIDTH, metres: Earth's measured ~50 km (Mercier de Lépinay et
/// al. 2016, 78 margins). A named calibration until a law exists (the research's §4.3).
pub const TRANSFORM_NECKING_M: f64 = 50_000.0;
/// ★ THE CONTINENTS' TOPOGRAPHY SPECTRUM's exponent `β`: power ∝ `k^−β` (Gagnon, Lovejoy &
/// Schertzer 2006, Nonlin. Processes Geophys. 13); the field's Hurst exponent is `(β − 1)/2`.
pub const SPECTRUM_BETA: f64 = 2.09;
/// ★ EARTH'S STABLE CONTINENTS' HEIGHT DIFFERENCE, metres, at [`SPECTRUM_LAG_M`] (the research's
/// reading of ETOPO1 over the stable continents' cells): the older landscape's one calibration of
/// size, a NAMED scatter for a history the facts cannot recover (rulings T9, B60).
pub const SPECTRUM_RMS_M: f64 = 131.0;
/// The lag of [`SPECTRUM_RMS_M`], metres.
pub const SPECTRUM_LAG_M: f64 = 110_000.0;
/// The older landscape's longest wave, metres: a continent's own scale, from which the spectrum's
/// octaves halve down to two of the body's nodes (the shortest wave its lattice holds).
pub const SPECTRUM_LONGEST_M: f64 = 4_096_000.0;
/// ★ THE CONTINENTAL SHARE of the surface: the crust field's threshold is the area quantile that
/// gives continental crust this share (calibration: Earth, 40 % — 29 % land plus the shelves).
/// No published law gives another body's share; it is the one calibration with no scaling.
pub const CONTINENTAL_SHARE: f64 = 0.40;
/// The crust field's transition band in field units — the width, in the affinity, over which the
/// crust fades from oceanic to continental: the continental slope. A tenth of the field's span, so
/// at the field's longest wavelength the slope is about a hundred kilometres wide (Earth's are
/// 20–100 km).
pub const AFFINITY_BAND: f64 = 0.25;
/// The affinity field's longest wavelength, metres: a continent-sized swell (02 §4.2, ~3 000 km).
pub const AFFINITY_WAVE_M: u32 = 3_000_000;
/// The affinity field's octaves: three, each half the wavelength and half the amplitude of the one
/// before (02 §4.2).
pub const AFFINITY_OCTAVES: usize = 3;
/// The per-plate character: a plate is mostly continental or mostly oceanic, an offset drawn in
/// `[−PLATE_AFFINITY, PLATE_AFFINITY)` added to the field — the reason a plate has a character.
pub const PLATE_AFFINITY: f64 = 0.5;
/// ★ HOW MANY PLATES: the driver is the ratio of the body's radius to its lithosphere's thickness
/// — a thin lithosphere on a big body breaks into many plates (02 §4.7) — calibrated on Earth:
/// about fifteen plates at `R/T_e = 6 371 km / 35 km`. Under two the body is a STAGNANT LID (one
/// unbroken shell: no boundaries, no belts, like Venus and Mars today).
pub const EARTH_PLATES: f64 = 15.0;
/// Earth's radius, metres (the plate law's calibration).
pub const EARTH_RADIUS_M: f64 = 6.371e6;
/// Earth's elastic thickness, metres (the plate law's calibration; the census's own reference).
pub const EARTH_ELASTIC_THICKNESS_M: f64 = 35_000.0;
/// No body in the literature carries more than about forty plates (02 §4.7).
pub const PLATES_MAX: u32 = 40;
/// A boundary whose normal rate is under this share of the relative drift SLIDES PAST (a transform
/// boundary): a scarp, no uplift.
pub const TRANSFORM_SHARE: f64 = 0.1;
/// ★ THE FLEXURAL PARAMETER's law (Turcotte & Schubert): `α = (4D / (Δρ·g))^¼` with the flexural
/// rigidity `D = E·T_e³ / (12·(1 − ν²))`; Young's modulus of the lithosphere, Pa.
pub const YOUNG_MODULUS_PA: f64 = 70.0e9;
/// Poisson's ratio of the lithosphere.
pub const POISSON_RATIO: f64 = 0.25;
/// A belt's half-width in flexural parameters: the crust bends over about two of them on each
/// side of the load (Earth: `α ≈ 120 km`, belts 250–500 km wide).
pub const BELT_HALF_WIDTHS: f64 = 2.0;
/// A trench or a rift is a quarter of a belt wide.
pub const NARROW_SHARE: f64 = 0.25;
/// A trench's depth, a ridge's height, an arc's height and a rift's depth as shares of the belt
/// amplitude the relief law caps: a trench dives half as deep as a belt stands, a ridge rides a
/// quarter (young hot crust rides 2–3 km over the abyssal plain on Earth, against 8 km of belt).
pub const TRENCH_SHARE: f64 = 0.5;
pub const ARC_SHARE: f64 = 0.5;
pub const RIDGE_SHARE: f64 = 0.25;
pub const RIFT_SHARE: f64 = 0.25;
/// ★ THE FREEBOARD LAW (2026-09-22, ruling W7 step 2; Wise 1974, "Continental margins, freeboard
/// and the volumes of continents and oceans through time"): a continent's thickness is not a
/// constant of the body but the balance the sea imposes over the ages — erosion thins a platform
/// that stands over the sea and sediment thickens one that stands under it, until THE PLATFORM
/// STANDS AT THE SEA. So the continental thickness is SOLVED: the factor on the Earth-scaled
/// thickness at which the sea the inventory fills stands at the continental platform's own
/// area-weighted median. Calibration: Earth, whose platform (the shelf edge) stands at its sea.
/// The factor is searched in this range and clamped at its ends — a body whose water fits no
/// crust in the range is a WATER WORLD or a DESERT WORLD and says so in its land.
/// MEASURED before the law on the home planet (twice Earth's water by its formation ratio): the
/// sea stood 861 m over the platform, and every continent the rivers graded to it drowned.
pub const FREEBOARD_FACTOR_RANGE: (f64, f64) = (0.5, 3.0);
/// ★ EARTH'S LAND SHARE, the freeboard law's second calibration number: 29 % of Earth's surface
/// stands over the sea on 40 % of continental crust ([`CONTINENTAL_SHARE`]), so the sea stands at
/// the continental crust's area quantile `1 − 0.29 / 0.40 = 0.275` — the shelves (the lowest
/// 27.5 % of the crust) under it, the platform over it. The median put half the crust under a
/// few metres of water and the coast came out as a checkerboard of one-node islands.
pub const EARTH_LAND_SHARE: f64 = 0.29;
/// The bisections of the freeboard factor: twenty halvings of a 2.5-wide range is a thousandth.
pub const FREEBOARD_BISECTIONS: u32 = 20;
/// The hypsometry's bin, metres.
pub const HYPSOMETRY_BIN_M: u32 = 250;
/// The hypsometry's quantile histogram bins for the crust threshold.
const QUANTILE_BINS: usize = 4_096;

/// The kind of the nearest plate boundary at a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    /// One plate: no boundary at all (a stagnant lid).
    None = 0,
    /// The plates push together: a belt, a trench, an arc.
    Convergent = 1,
    /// The plates pull apart: a ridge, a rift.
    Divergent = 2,
    /// The plates slide past: a scarp, no uplift.
    Transform = 3,
}

/// One plate: its site, its drift (a tangent vector whose length is its speed share), its
/// continental character, its belts' age share, and its crust's named scatter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plate {
    pub site: [Gf; 3],
    pub drift: [Gf; 3],
    pub affinity: Gf,
    pub age: Gf,
    pub crust_scatter: Gf,
}

/// The nearest boundary at a node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Boundary {
    /// The node's own plate.
    pub plate: u8,
    /// The second-nearest plate, `None` on a one-plate body.
    pub other: Option<u8>,
    /// The sine of the angle from the node to the bisector plane — the boundary distance over the
    /// radius (02 §4.1: under one percent from the arc inside every profile).
    pub sine: Gf,
    pub kind: Kind,
    /// The convergence rate `−(v · n̂)`, a share in `[−2, 2]`: positive when they push together.
    pub convergence: Gf,
}

/// ★ THE INITIAL LAND of one body over its macro lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct Land {
    /// The heights in sixteenths from the ladder radius — the ISOSTASY, the water load applied
    /// under the sea. The belts are not in it: they RISE during the age ([`Land::uplift`]).
    pub z: Vec<i32>,
    /// ★ THE UPLIFT OVER THE AGE, sixteenths: what each node's belt, trench, arc, ridge or rift
    /// adds to it by the end of the age. The solve applies it pass by pass while the rivers cut,
    /// so a range is the balance of the two (C3); a trench's is negative — it subsides.
    pub uplift: Vec<i32>,
    /// The sea's level in sixteenths; `None` on a body with no water.
    pub sea_z: Option<i32>,
    /// The node's plate.
    pub plate: Vec<u8>,
    /// The nearest boundary's kind, as its discriminant.
    pub kind: Vec<u8>,
    /// The distance to the nearest boundary, whole metres (`u32::MAX` on a one-plate body).
    pub boundary_m: Vec<u32>,
    /// The crust share `S` in `0..=255`: 0 oceanic, 255 continental.
    pub crust: Vec<u8>,
    /// ★ THE ROCK PROVINCE of each node ([`province_of`]), as its code byte. Derived here, where
    /// the crust and the uplift stand, and carried to the artifact's row — the solve throws the
    /// crust and the belt fields away, so nobody downstream could compute it again.
    pub province: Vec<u8>,
    /// The plates drawn.
    pub plates: Vec<Plate>,
    /// The crust field's threshold, the area quantile.
    pub threshold: Gf,
    /// ★ THE FREEBOARD FACTOR the law solved ([`FREEBOARD_FACTOR_RANGE`]): the continental
    /// thickness over the Earth-scaled one. ONE on a body with no water.
    pub freeboard_factor: Gf,
    /// ★ THE OCEAN (2026-09-24, ruling B7 item 1): whether each node belongs to the CONNECTED
    /// ocean at the sea's level ([`ocean_mask`]); every node `false` on a body with no water. Only
    /// these nodes carry the water's load.
    pub ocean: Vec<bool>,
    /// The bodies at or under the sea's level that are NOT the ocean on this first land, and their
    /// nodes: the inland pockets the sea no longer counts (ruling B7 item 1; plan question 9).
    pub pockets: usize,
    pub pocket_nodes: usize,
}

impl Land {
    /// ★ THE SEA AS THE CLIMATE READS IT: the level and the connected ocean's mask; `None` on a body
    /// with no water.
    #[must_use]
    pub fn sea_view(&self) -> Option<crate::climate::SeaView<'_>> {
        self.sea_z.map(|level| crate::climate::SeaView {
            level,
            ocean: &self.ocean,
        })
    }
}

/// ★ THE PLATE LAW: `round(EARTH_PLATES · (R / T_e) / (R⊕ / T_e⊕))`, at least one, at most
/// [`PLATES_MAX`]. The home planet (6 342 km over 36 789 m) draws fourteen; its moon, whose
/// lithosphere is three times its radius, is a stagnant lid.
#[must_use]
pub fn plate_count(radius_m: f64, elastic_thickness_m: u32) -> u32 {
    let ratio = Gf::from_f64(radius_m) / Gf::from_i64(i64::from(elastic_thickness_m.max(1)));
    let earth = Gf::from_f64(EARTH_RADIUS_M) / Gf::from_f64(EARTH_ELASTIC_THICKNESS_M);
    let count = (Gf::from_f64(EARTH_PLATES) * ratio / earth + Gf::HALF).floor();
    let count = count.to_i64_floor().clamp(1, i64::from(PLATES_MAX));
    count as u32
}

/// The flexural parameter `α` in metres for an elastic thickness, under the body's gravity: the
/// length over which the lithosphere bends under a load.
#[must_use]
pub fn flexural_parameter_m(elastic_thickness_m: u32, gravity_mm_s2: u32) -> f64 {
    let te = Gf::from_i64(i64::from(elastic_thickness_m));
    let rigidity = Gf::from_f64(YOUNG_MODULUS_PA) * te * te * te
        / (Gf::from_f64(12.0) * (Gf::ONE - Gf::from_f64(POISSON_RATIO * POISSON_RATIO)));
    let g = Gf::from_i64(i64::from(gravity_mm_s2)) / Gf::from_i64(1_000);
    let drho = Gf::from_f64(MANTLE_DENSITY_KGM3 - CONTINENTAL_CRUST_DENSITY_KGM3);
    (Gf::from_f64(4.0) * rigidity / (drho * g))
        .sqrt()
        .sqrt()
        .to_f64()
}

/// A unit direction from three unit draws through the LADDER's bend (02 §7.1: never a Gaussian,
/// never a rejection loop): a face and a position on it, the bend's own equal-area design making
/// the draw near-uniform.
fn draw_direction(rng: &mut SplitMix64) -> [Gf; 3] {
    let face = Face::from_index(rng.range_u64(0, 6) as u8).unwrap_or(Face::NegZ);
    let a = (draw_unit(rng) * Gf::TWO - Gf::ONE).to_f64();
    let b = (draw_unit(rng) * Gf::TWO - Gf::ONE).to_f64();
    let d = direction(face, a, b);
    [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])]
}

fn dot(a: [Gf; 3], b: [Gf; 3]) -> Gf {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [Gf; 3], b: [Gf; 3]) -> [Gf; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [Gf; 3], k: Gf) -> [Gf; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

/// `v / |v|`; a zero vector stays zero (a stated answer, not a division by zero).
fn normalise(v: [Gf; 3]) -> [Gf; 3] {
    let len = dot(v, v).sqrt();
    if len > Gf::ZERO {
        scale(v, Gf::ONE / len)
    } else {
        v
    }
}

/// ★ THE PLATES, drawn from the body's own land stream in a FROZEN order: for each plate its site,
/// its drift (a second direction projected onto the tangent plane at the site, scaled by a speed
/// share), its affinity, its age share and its crust scatter — every unit taken whether the law
/// uses it or not, so a later reader of the stream stands where it stood.
#[must_use]
pub fn draw_plates(seed: u64, count: u32) -> Vec<Plate> {
    let mut rng = SplitMix64::new(child_seed(seed, salt::LAND, 0));
    (0..count)
        .map(|_| {
            let site = draw_direction(&mut rng);
            let toward = draw_direction(&mut rng);
            let radial = dot(toward, site);
            let tangent = normalise(sub(toward, scale(site, radial)));
            let speed = draw_unit(&mut rng);
            let affinity = (draw_unit(&mut rng) * Gf::TWO - Gf::ONE) * Gf::from_f64(PLATE_AFFINITY);
            let age = draw_unit(&mut rng);
            // The sourced spread: uniform with Earth's stable continents' σ, `1 ± σ·√3`.
            let half = Gf::from_f64(EARTH_CONTINENT_SPREAD) * Gf::from_i64(3).sqrt();
            let crust_scatter = (Gf::ONE - half) + draw_unit(&mut rng) * (Gf::TWO * half);
            Plate {
                site,
                drift: scale(tangent, speed),
                affinity,
                age,
                crust_scatter,
            }
        })
        .collect()
}

/// ★ THE BOUNDARY at a direction: the nearest and second-nearest sites by a scan of the whole list
/// in index order (the scan law, 02 §4.0; a tie keeps the smaller index), the sine distance to
/// their bisector plane, and the kind from the two drifts: `v = drift₂ − drift₁`, `n̂` from site 1
/// toward site 2, `c = −(v · n̂)`.
#[must_use]
pub fn boundary_of(plates: &[Plate], d: [Gf; 3]) -> Boundary {
    let (mut best, mut best_dot) = (0usize, Gf::from_f64(-2.0));
    let (mut second, mut second_dot) = (usize::MAX, Gf::from_f64(-2.0));
    for (k, p) in plates.iter().enumerate() {
        let s = dot(d, p.site);
        if s > best_dot {
            second = best;
            second_dot = best_dot;
            best = k;
            best_dot = s;
        } else if s > second_dot {
            second = k;
            second_dot = s;
        }
    }
    if plates.len() < 2 {
        return Boundary {
            plate: best as u8,
            other: None,
            sine: Gf::ONE,
            kind: Kind::None,
            convergence: Gf::ZERO,
        };
    }
    let (p1, p2) = (&plates[best], &plates[second]);
    let n = normalise(sub(p2.site, p1.site));
    let v = sub(p2.drift, p1.drift);
    let normal_rate = dot(v, n);
    let speed = dot(v, v).sqrt();
    let kind = if normal_rate.abs() < speed * Gf::from_f64(TRANSFORM_SHARE) {
        Kind::Transform
    } else if normal_rate < Gf::ZERO {
        Kind::Convergent
    } else {
        Kind::Divergent
    };
    Boundary {
        plate: best as u8,
        other: Some(second as u8),
        sine: dot(d, n).abs(),
        kind,
        convergence: -normal_rate,
    }
}

/// The quintic smoothstep `x³(10 − 15x + 6x²)` on `[0, 1]`, clamped: the crust's fade from
/// oceanic to continental.
#[must_use]
pub fn smoothstep(x: Gf) -> Gf {
    let x = x.clamp(Gf::ZERO, Gf::ONE);
    x * x * x * (Gf::from_f64(10.0) - Gf::from_f64(15.0) * x + Gf::from_f64(6.0) * x * x)
}

/// The crust share at an affinity against the threshold: the smoothstep over the band.
#[must_use]
pub fn crust_share(affinity: Gf, threshold: Gf) -> Gf {
    smoothstep((affinity - threshold) / Gf::from_f64(AFFINITY_BAND) + Gf::HALF)
}

/// ★ AIRY ISOSTASY: the elevation over the compensation depth of a crust of thickness `t` and
/// density `ρ` on a mantle of density `ρ_m`: `t · (1 − ρ/ρ_m)`. The thickness and the density are
/// each the oceanic value faded to the continental one by the crust share; the thicknesses scale
/// with the body's relief cap over Earth's, and the continental one carries the plate's named
/// scatter. Computed with Earth's numbers: a 35 km continent at 2 800 stands 5 303 m over the
/// compensation depth, a 7 km ocean floor at 2 900 stands 848 m — a dry step of 4 455 m.
#[must_use]
pub fn isostatic_height_m(share: Gf, relief_cap_m: f64, crust_scatter: Gf) -> Gf {
    let cap_ratio = Gf::from_f64(relief_cap_m) / Gf::from_f64(EARTH_RELIEF_CAP_M);
    let t_c = Gf::from_f64(EARTH_CONTINENTAL_CRUST_M) * cap_ratio * crust_scatter;
    let t_o = Gf::from_f64(EARTH_OCEANIC_CRUST_M) * cap_ratio;
    let t = t_o + (t_c - t_o) * share;
    let rho = Gf::from_f64(OCEANIC_CRUST_DENSITY_KGM3)
        + Gf::from_f64(CONTINENTAL_CRUST_DENSITY_KGM3 - OCEANIC_CRUST_DENSITY_KGM3) * share;
    t * (Gf::ONE - rho / Gf::from_f64(MANTLE_DENSITY_KGM3))
}

/// The bump `(1 − x²)²` on `|x| < 1`, an EXACT zero outside: every profile is built of it, so a
/// node far from a boundary contributes exactly nothing (the scan law's first condition).
#[must_use]
pub fn bump(x: Gf) -> Gf {
    let x2 = x * x;
    if x2 < Gf::ONE {
        let r = Gf::ONE - x2;
        r * r
    } else {
        Gf::ZERO
    }
}

/// ★ THE OROGENY at a node, metres: one machinery, one profile per boundary kind and crust pair.
/// `amplitude_m` is the belt's cap under the relief law times the convergence share and the age
/// share; `width_m` the belt's half-width (the flexural parameter times [`BELT_HALF_WIDTHS`]);
/// `distance_m` the boundary distance; `own_continental` the node's own side, `other_continental`
/// the far plate's character.
///
/// | kind, crust pair | landform | profile |
/// |---|---|---|
/// | convergent, this side continental | a belt (a collision belt or a cordillera) | `+A · bump(b/W)` |
/// | convergent, this side oceanic, the other continental | a trench (the ocean floor dives) | `−A/2 · bump(b/(W/4))` |
/// | convergent, ocean + ocean | a trench on the lower-indexed plate, an island arc on the other | as the trench; `+A/2 · bump(b/W)` |
/// | divergent, this side continental | a rift valley | `−A/4 · bump(b/(W/4))` |
/// | divergent, this side oceanic | a mid-ocean ridge | `+A/4 · bump(b/W)` |
/// | transform, or no boundary | a scarp, no uplift | `0` |
#[must_use]
pub fn uplift_m(
    kind: Kind,
    own_continental: bool,
    other_continental: bool,
    own_is_lower_index: bool,
    distance_m: Gf,
    amplitude_m: Gf,
    width_m: Gf,
) -> Gf {
    let wide = bump(distance_m / width_m);
    let narrow = bump(distance_m / (width_m * Gf::from_f64(NARROW_SHARE)));
    match (kind, own_continental, other_continental) {
        (Kind::Convergent, true, _) => amplitude_m * wide,
        (Kind::Convergent, false, true) => -amplitude_m * Gf::from_f64(TRENCH_SHARE) * narrow,
        (Kind::Convergent, false, false) => {
            if own_is_lower_index {
                -amplitude_m * Gf::from_f64(TRENCH_SHARE) * narrow
            } else {
                amplitude_m * Gf::from_f64(ARC_SHARE) * wide
            }
        }
        (Kind::Divergent, true, _) => -amplitude_m * Gf::from_f64(RIFT_SHARE) * narrow,
        (Kind::Divergent, false, _) => amplitude_m * Gf::from_f64(RIDGE_SHARE) * wide,
        (Kind::Transform | Kind::None, _, _) => Gf::ZERO,
    }
}

/// ★ THE PROVINCE OF ONE NODE (slice 8d step 2; `slice_8d_design.md` §3.5): which rock map the
/// node's beds are drawn from, from FIVE WORDS THE LAND ALREADY HOLDS and nothing else — the
/// crust share, the uplift over the age, whether the node stands under its own sea, its
/// boundary's kind, and whether the plate across that boundary is continental.
///
/// ★ NO NUMBER IS DRAWN HERE (ruling T9). Each test is the land's OWN word:
///
/// * `crust >= 128` is the crust share's own half: the node's crust is more continental than
///   oceanic. The crust share is the cell's mean of the smoothstep of the affinity against the area
///   quantile, so 128 IS the threshold the quantile placed, not a tuning dial. (Since generator 26
///   the room's profiles blend by the share itself; the province names the rock the larger share
///   made.)
/// * `uplift != 0` says the plates are moving this ground: a belt, an arc, a ridge, a rift or a
///   trench. Zero is zero. A transform boundary and a plate's interior both uplift nothing.
/// * `z < sea` is the sea the land's own inventory solved for.
///
/// | crust | uplift | under the sea | province | the landform |
/// |---|---|---|---|---|
/// | continental | rises | — | folded belt | a collision belt, a cordillera |
/// | continental | sinks where the plates pull apart | — | rift basalt | a continental rift valley, flooded with lava |
/// | continental | sinks where the plates push together | — | deep sediment | a forearc under the trench's inner wall |
/// | continental | still | no | crystalline basement | the old shield |
/// | continental | still | yes | flat shelf | the drowned platform |
/// | oceanic | rises where the plates pull apart, or against an ocean | — | rift basalt | a mid-ocean ridge or an island arc |
/// | oceanic | rises where it dives under a continent | — | deep sediment | the diving plate's own floor |
/// | oceanic | sinks or still | — | deep sediment | a trench, and the abyssal plain |
///
/// **Example.** The pilot flies west over the home planet. She leaves the shield (crystalline
/// basement), crosses a range the plates raised (folded belt), drops over the beach and the shelf
/// (flat shelf), and then over the deep floor (deep sediment). Every rock she mines on that leg
/// comes from the province she stands on.
#[must_use]
pub fn province_of(
    crust: u8,
    uplift: i32,
    z: i32,
    sea_z: Option<i32>,
    kind: Kind,
    other_continental: bool,
) -> Province {
    let continental = crust >= 128;
    let drowned = matches!(sea_z, Some(level) if z < level);
    if uplift > 0 {
        if continental {
            return Province::FoldedBelt;
        }
        // ★ THE DIVING PLATE MAKES NO ARC (ruling B63 item 2): oceanic crust that rises where it
        // dives under a continent — the crust share's blend gives its partly continental cells a
        // part of the belt's profile — is the diving plate's sediment, never an arc's basalt.
        // MEASURED before: 25 073 such nodes of the home planet named rift basalt (generator 25:
        // none).
        if (kind == Kind::Convergent) & other_continental {
            return Province::DeepSediment;
        }
        return Province::RiftBasalt;
    }
    if uplift < 0 {
        // ★ A SINKING CONTINENT IS A RIFT ONLY WHERE THE PLATES PULL APART (ruling B63 item 2): where
        // they push together it is a forearc under the trench's inner wall, which fills with
        // sediment like the trench beside it. MEASURED before: 12 609 forearc nodes of the home
        // planet named rift basalt.
        if continental & (kind == Kind::Divergent) {
            return Province::RiftBasalt;
        }
        return Province::DeepSediment;
    }
    if !continental {
        return Province::DeepSediment;
    }
    if drowned {
        return Province::FlatShelf;
    }
    Province::CrystallineBasement
}

/// The affinity field's octaves: three smooth swells, each half the wavelength and half the
/// amplitude of the one before, on the body's land stream.
fn affinity_octaves(body: &BodyDefinition) -> [Octave; AFFINITY_OCTAVES] {
    let radius = Gf::from_f64(body.ladder().radius_m());
    let mut wave = Gf::from_i64(i64::from(AFFINITY_WAVE_M));
    let mut amp = 1i64 << AMP_BITS;
    let mut out = [Octave::smooth(0, Gi::ZERO, Gi::ZERO, Gi::ZERO); AFFINITY_OCTAVES];
    for (k, slot) in out.iter_mut().enumerate() {
        let (fi, ff) = frequency_of(radius / wave);
        *slot = Octave::smooth(
            child_seed(body.seed(), salt::LAND, 1 + k as u64),
            fi,
            ff,
            Gi::new(amp),
        );
        wave = wave / Gf::TWO;
        amp >>= 1;
    }
    out
}

/// The affinity noise at a node's direction, as a unit-amplitude sum in `[−1.75, 1.75]`.
fn affinity_noise(octaves: &[Octave], dir: [Gi; 3]) -> Gf {
    Gf::from_i64(relief(octaves, dir).raw()) / Gf::from_i64(1 << NOISE_BITS)
}

/// ★ THE SUB-CELLS' BOUNDARIES of node `node` (ruling B59 item 2): the boundary at each of its
/// [`CELL_SUBCELLS`]² sub-cells, the recipe's own bend on a lattice `K` times finer (`inv_sub` is
/// its reciprocal), row by row, into `out`.
fn sub_boundaries(
    lattice: &MacroLattice,
    plates: &[Plate],
    node: u32,
    inv_sub: Gi,
    out: &mut Vec<Boundary>,
) {
    out.clear();
    let (face, i, j) = lattice.split(node);
    let k = CELL_SUBCELLS as i32;
    for b in 0..k {
        for a in 0..k {
            let d =
                vd_recipe::bend::direction(i32::from(face.index()), i * k + a, j * k + b, inv_sub);
            out.push(boundary_of(plates, dir_of(d)));
        }
    }
}

/// ★ THE MARGIN'S WIDTH BY KIND, metres (ruling B59 item 2), for the plate `own` whose crust column
/// stands higher; `freeboard` is the continental thickness factor and `elastic_thickness_m` the
/// body's lithosphere:
///
/// - an ANDEAN margin (convergent): the coast stands over the crossing of the plate interface and
///   the upper plate's Moho (Ruff & Tichelaar 1996; Melnick 2016), `x = (z_M − d_T)/tan δ` — `z_M`
///   the plate's own continental crust, `d_T` the ocean floor's depth under its platform (the land's
///   own Airy step), `δ` [`ANDEAN_DIP_DEG`];
/// - a RIFTED margin (divergent): the necking instability, `W = ½(1 + ε)·λ`, `λ`
///   [`NECKING_WAVELENGTH_RATIO`] times the strong layer, `ε` [`NECKING_STRETCH`]; the strong
///   layer is the model's own decoupling level — the upper crust over the weak lower crust, the
///   belts' law's [`crate::belts::INDIA_STORED_CRUST_M`] carried by the elastic thickness, a stated
///   identification;
/// - a TRANSFORM margin: Earth's measured necking, [`TRANSFORM_NECKING_M`];
/// - no boundary: no margin.
///
/// **Example.** Earth's own crusts (a 35 km continent over a 7 km ocean floor) put an Andean coast
/// about 72 km inland of the trench; the home planet's continents, which the freeboard law
/// thickens, put it about 94 km inland. The continent's crust thins toward the trench over that
/// width, so its height falls at about 3.7°, as Earth's continental slopes fall at 3–4°.
#[must_use]
pub fn margin_width_m(
    kind: Kind,
    own: &Plate,
    cap_m: f64,
    freeboard: Gf,
    elastic_thickness_m: u32,
) -> Gf {
    let cap_ratio = Gf::from_f64(cap_m) / Gf::from_f64(EARTH_RELIEF_CAP_M);
    match kind {
        Kind::Convergent => {
            let rm = Gf::from_f64(MANTLE_DENSITY_KGM3);
            let t_c =
                Gf::from_f64(EARTH_CONTINENTAL_CRUST_M) * cap_ratio * own.crust_scatter * freeboard;
            let t_o = Gf::from_f64(EARTH_OCEANIC_CRUST_M) * cap_ratio;
            let d_t = t_c * (Gf::ONE - Gf::from_f64(CONTINENTAL_CRUST_DENSITY_KGM3) / rm)
                - t_o * (Gf::ONE - Gf::from_f64(OCEANIC_CRUST_DENSITY_KGM3) / rm);
            (t_c - d_t) / Gf::from_f64(ANDEAN_DIP_TAN)
        }
        Kind::Divergent => {
            let strong = Gf::from_f64(crate::belts::INDIA_STORED_CRUST_M)
                * Gf::from_i64(i64::from(elastic_thickness_m))
                / Gf::from_f64(EARTH_ELASTIC_THICKNESS_M);
            Gf::HALF
                * (Gf::ONE + Gf::from_f64(NECKING_STRETCH))
                * Gf::from_f64(NECKING_WAVELENGTH_RATIO)
                * strong
        }
        Kind::Transform => Gf::from_f64(TRANSFORM_NECKING_M),
        Kind::None => Gf::ZERO,
    }
}

/// ★ A FIELD OF OCTAVES over the lattice, zero mean and unit variance by area: each octave
/// `(wave_m, amplitude)` is the recipe's smooth noise at that wave on the body's stream `stream`,
/// its amplitude in the recipe's whole steps.
fn unit_field(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    stream: u64,
    octaves: &[(Gf, Gf)],
) -> Vec<Gf> {
    let radius = Gf::from_f64(body.ladder().radius_m());
    let built: Vec<Octave> = octaves
        .iter()
        .enumerate()
        .map(|(k, &(wave, amp))| {
            let (fi, ff) = frequency_of(radius / wave);
            Octave::smooth(
                child_seed(body.seed(), stream, 1 + k as u64),
                fi,
                ff,
                Gi::new((amp * Gf::from_i64(1 << AMP_BITS) + Gf::HALF).to_i64_floor()),
            )
        })
        .collect();
    let mut field: Vec<Gf> = (0..lattice.node_count() as u32)
        .map(|node| affinity_noise(&built, lattice.direction(node)))
        .collect();
    let (mut sum, mut sum2, mut total) = (Gf::ZERO, Gf::ZERO, Gf::ZERO);
    for (node, &f) in field.iter().enumerate() {
        let a = Gf::from_i64(lattice.area_m2(node as u32) as i64);
        sum += f * a;
        sum2 += f * f * a;
        total += a;
    }
    let mean = sum / total;
    let sd = (sum2 / total - mean * mean).sqrt();
    for f in &mut field {
        *f = (*f - mean) / sd;
    }
    field
}

/// The root-mean-square difference of `field` between nodes `lag` apart along the face rows and
/// columns.
fn rms_difference(lattice: &MacroLattice, field: &[Gf], lag: i32) -> Gf {
    let edge = lattice.edge as i32;
    let (mut s, mut c) = (Gf::ZERO, 0i64);
    for node in 0..lattice.node_count() as u32 {
        let (face, i, j) = lattice.split(node);
        if i + lag < edge {
            let d = field[lattice.index(face, i + lag, j) as usize] - field[node as usize];
            s += d * d;
            c += 1;
        }
        if j + lag < edge {
            let d = field[lattice.index(face, i, j + lag) as usize] - field[node as usize];
            s += d * d;
            c += 1;
        }
    }
    (s / Gf::from_i64(c)).sqrt()
}

/// ★ THE OLDER LANDSCAPE'S RELIEF, metres a node (rulings B59 item 1, B60 item 1): the relief a
/// plate carried when its erosion began, which no fact in hand recovers — a NAMED scatter (ruling
/// T9) with the continents' published spectrum ([`SPECTRUM_BETA`]; Hurst `(β − 1)/2`) from
/// [`SPECTRUM_LONGEST_M`] down to two of the body's nodes, sized so that its height difference at
/// [`SPECTRUM_LAG_M`] along the face rows and columns is Earth's stable continents'
/// [`SPECTRUM_RMS_M`]. The land weights it by each node's crust share: an ocean floor carries none.
///
/// **Example.** A pilot over the home planet's old plains sees low rises and shallow basins about
/// 130 m apart in height over 110 km, the older landscape the plate carried when its erosion
/// began, and rivers that wind between them; before generator 26 the same plains started as one
/// exact flat and their first rivers ran in straight lines.
#[must_use]
pub fn older_landscape_m(body: &BodyDefinition, lattice: &MacroLattice) -> Vec<Gf> {
    let hurst = (Gf::from_f64(SPECTRUM_BETA) - Gf::ONE) / Gf::TWO;
    let ratio = Gf::TWO.powf(-hurst);
    let shortest = Gf::from_f64(lattice.node_m()) * Gf::TWO;
    let mut octaves = Vec::new();
    let (mut wave, mut amp) = (Gf::from_f64(SPECTRUM_LONGEST_M), Gf::ONE);
    while wave >= shortest {
        octaves.push((wave, amp));
        wave = wave / Gf::TWO;
        amp *= ratio;
    }
    let field = unit_field(body, lattice, salt::OLDER_LANDSCAPE, &octaves);
    // The height difference at the lag, between the two whole lags around it.
    let lag = Gf::from_f64(SPECTRUM_LAG_M) / Gf::from_f64(lattice.node_m());
    let below = lag.floor();
    let frac = lag - below;
    let l0 = below.to_i64_floor() as i32;
    let at = rms_difference(lattice, &field, l0) * (Gf::ONE - frac)
        + rms_difference(lattice, &field, l0 + 1) * frac;
    let scale = Gf::from_f64(SPECTRUM_RMS_M) / at;
    field.into_iter().map(|f| f * scale).collect()
}

/// A node's direction as fenced floats.
pub(crate) fn dir_of(dir: [Gi; 3]) -> [Gf; 3] {
    let one = Gf::from_i64(1 << DIR_BITS);
    [
        Gf::from_i64(dir[0].raw()) / one,
        Gf::from_i64(dir[1].raw()) / one,
        Gf::from_i64(dir[2].raw()) / one,
    ]
}

/// ★ THE AREA QUANTILE of a field: the value under which `share` of the area lies, by a histogram
/// of [`QUANTILE_BINS`] over the field's span. Integer bins, an integer area sum: one answer on
/// every host.
#[must_use]
pub fn area_quantile(values: &[Gf], area: &[u64], share: f64) -> Gf {
    let (mut lo, mut hi) = (Gf::from_f64(f64::MAX), Gf::from_f64(f64::MIN));
    for &v in values {
        lo = lo.lesser(v);
        hi = hi.greater(v);
    }
    let span = hi - lo;
    if span <= Gf::ZERO {
        return lo;
    }
    let mut bins = vec![0u128; QUANTILE_BINS];
    let scale = Gf::from_i64(QUANTILE_BINS as i64) / span;
    for (k, &v) in values.iter().enumerate() {
        let bin = ((v - lo) * scale)
            .to_i64_floor()
            .clamp(0, QUANTILE_BINS as i64 - 1) as usize;
        bins[bin] += u128::from(area[k]);
    }
    let total: u128 = bins.iter().sum();
    let target = (Gf::from_f64(share) * Gf::from_i64(total as i64)).to_i64_floor() as u128;
    let mut acc = 0u128;
    let mut bin = 0usize;
    while bin + 1 < QUANTILE_BINS && acc + bins[bin] < target {
        acc += bins[bin];
        bin += 1;
    }
    lo + Gf::from_i64(bin as i64) / scale
}

/// ★ THE SEA OVER EVERY NODE OF THE FIELD, WITH THE LOAD (02 §7.1, §4.2): the level `L`, in
/// sixteenths, at which the water under it equals the inventory, where a node under the level sinks
/// by the water's weight — and the deeper column weighs more, so the balance is the fixed point
/// `sink = (L − h + sink)·ρ_w/ρ_m`, whose closed form is `sink = (L − h)·ρ_w/(ρ_m − ρ_w)` and the
/// water column `(L − h)/(1 − ρ_w/ρ_m)`, linear in `L` (Earth's 4 455 m dry step becomes the
/// 6 476 m loaded one). An integer bisection on the level between the lowest node and the level
/// that would drown the whole field; `None` for no water. `with_load` false reads a field whose
/// isostatic sink is already in it (the eroded field at the end of the solve), so the water's own
/// load is not counted twice.
///
/// ★ PRIVATE SINCE 2026-09-24 (ruling B7 item 1): this counts EVERY node under the level as sea,
/// an inland pocket included. The body's sea is [`sea_level_connected`], which starts here and
/// then counts the connected ocean only.
fn sea_level_over(z_dry: &[i32], area: &[u64], water_km3: u64, with_load: bool) -> Option<i32> {
    if water_km3 == 0 {
        return None;
    }
    let target = sea_target(water_km3, with_load);
    let (mut lo, mut hi) = sea_bracket(z_dry, area, target);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if water_under(z_dry, area, mid as i32) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(hi as i32)
}

/// The inventory as the bisection's target, in area·sixteenths: the inventory in m³ times the steps
/// a metre, over the load where the field is dry.
fn sea_target(water_km3: u64, with_load: bool) -> u128 {
    let load = if with_load {
        Gf::ONE / (Gf::ONE - Gf::from_f64(WATER_DENSITY_KGM3 / MANTLE_DENSITY_KGM3))
    } else {
        Gf::ONE
    };
    let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let target_m3_steps = Gf::from_i64(water_km3 as i64) * Gf::from_f64(1.0e9) * steps / load;
    target_m3_steps.to_f64() as u128
}

/// The bisection's first bracket: the lowest node, and the level that would hold the whole target
/// over the highest node.
fn sea_bracket(z: &[i32], area: &[u64], target: u128) -> (i64, i64) {
    let total_area: u128 = area.iter().map(|&a| u128::from(a)).sum();
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for &h in z {
        lo = lo.min(h);
        hi = hi.max(h);
    }
    let column = (target / total_area.max(1)) as i64 + 1;
    (i64::from(lo), i64::from(hi) + column)
}

/// The water under a level over the dry field, in area·sixteenths (the load's factor left out:
/// it is applied to the target once).
fn water_under(z_dry: &[i32], area: &[u64], level: i32) -> u128 {
    let mut sum = 0u128;
    for (k, &z) in z_dry.iter().enumerate() {
        if z < level {
            sum += u128::from(area[k]) * (u128::from((i64::from(level) - i64::from(z)) as u64));
        }
    }
    sum
}

/// No body: a node over the level in [`ocean_mask`]'s labels.
const NO_BODY: u32 = u32::MAX;

/// ★ THE OCEAN'S MASK AT A LEVEL (2026-09-24, ruling B7 item 1; plan step 2 part 3a).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OceanMask {
    /// Whether each node belongs to the ocean.
    pub ocean: Vec<bool>,
    /// The bodies at or under the level that are NOT the ocean: the inland pockets.
    pub pockets: usize,
    /// Their nodes.
    pub pocket_nodes: usize,
}

/// ★ THE OCEAN IS THE LARGEST CONNECTED WATER UNDER THE LEVEL (2026-09-24, ruling B7 item 1: *"an
/// inland pocket is a closed basin, not the sea"*; plan question 1). Every node at or under `level`
/// is labelled by a walk through the lattice's OWN eight neighbours, the cube seams included; the
/// body with the largest AREA is the ocean, and on an exact tie the body whose lowest node index is
/// smaller (the census's own rule, `lake_census` `inland_sea_lines`). Every other body is a POCKET:
/// a closed basin whose water comes from its budget, never from the sea. A level under every node
/// gives an empty ocean.
///
/// **Example.** A crater floor 300 km inland stands 40 m under the sea's level. No path through the
/// lattice joins it to the ocean, so it is a pocket: the solve's flood raises it like any hollow,
/// and the lake budget, not the sea, decides whether water stands in it.
#[must_use]
pub fn ocean_mask(lattice: &MacroLattice, z: &[i32], area: &[u64], level: i32) -> OceanMask {
    let n = z.len();
    let mut label = vec![NO_BODY; n];
    // Each body's area and node count, in the order its lowest index was met.
    let mut bodies: Vec<(u128, usize)> = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    for start in 0..n {
        if (label[start] != NO_BODY) | (z[start] > level) {
            continue;
        }
        let id = bodies.len() as u32;
        let (mut body_area, mut nodes) = (0u128, 0usize);
        label[start] = id;
        stack.push(start as u32);
        while let Some(v) = stack.pop() {
            body_area += u128::from(area[v as usize]);
            nodes += 1;
            for m in lattice.neighbours(v) {
                if m == NO_NODE {
                    continue;
                }
                let mi = m as usize;
                if (label[mi] == NO_BODY) & (z[mi] <= level) {
                    label[mi] = id;
                    stack.push(m);
                }
            }
        }
        bodies.push((body_area, nodes));
    }
    // The largest by area; a later body replaces it only when STRICTLY larger, so a tie keeps the
    // body met first — the one whose lowest index is smaller.
    let mut ocean = NO_BODY;
    let mut largest = 0u128;
    for (k, &(body_area, _)) in bodies.iter().enumerate() {
        if ocean == NO_BODY || body_area > largest {
            ocean = k as u32;
            largest = body_area;
        }
    }
    let ocean_nodes = if ocean == NO_BODY {
        0
    } else {
        bodies[ocean as usize].1
    };
    OceanMask {
        ocean: label
            .iter()
            .map(|&b| (b != NO_BODY) & (b == ocean))
            .collect(),
        pockets: bodies.len().saturating_sub(1),
        pocket_nodes: bodies.iter().map(|b| b.1).sum::<usize>() - ocean_nodes,
    }
}

/// ★ THE BODY'S SEA OVER THE CONNECTED OCEAN (2026-09-24, ruling B7 item 1; plan step 2 part 3a).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectedSea {
    /// The level, sixteenths.
    pub level: i32,
    /// The ocean's mask at that level.
    pub mask: OceanMask,
    /// Whether the first seed's body was not the largest at the level, so the solve seeded again
    /// once from the body that was (stated, and reported).
    pub reseeded: bool,
    /// ★ WHETHER THE ANSWER BALANCES (2026-09-25, review finding 4 of plan step 2): the water the
    /// level was solved over is exactly the water over the answer's own ocean — every node joined
    /// to the seed body at the level is the ocean, and no other node is. `false` where the re-seed
    /// body is no longer the largest at its own level, so the largest-body rule and the inventory
    /// name two different bodies: the level is the re-seed body's and the mask is another's. No
    /// single seed balances such a field; the answer is stated and reported, never silent.
    pub settled: bool,
}

/// ★ THE SEA OVER THE CONNECTED OCEAN ONLY (2026-09-24, ruling B7 item 1: *"the sea's level is
/// solved over the CONNECTED ocean only. One law, no special case"*). The inventory fills the ocean;
/// a pocket under the level that no path joins to the ocean holds none of it.
///
/// 1. `L0` = the level over EVERY node ([`sea_level_over`]).
/// 2. The bodies at `L0` ([`ocean_mask`]). ONE body: `L0` is exact, because the connected volume
///    and the whole volume are one at `L0` and the connected one is never more under it.
/// 3. More: a priority flood from the ocean at `L0` gives each node the lowest level at which it
///    joins that ocean (`reach`), and the level is bisected again over
///    `V(L) = Σ (L − z)·a` for `reach ≤ L`, `z < L` — monotone in `L`, the same target and load.
/// 4. The bodies at the answer. Where the seed's body is not the largest there, the flood and the
///    bisection run ONCE more from the body that is, and the answer says so.
/// 5. ★ THE CHECK (review finding 4 of plan step 2): the nodes joined to the seed at the answer's
///    level against the answer's ocean ([`ConnectedSea::settled`]). Where the re-seed body lost the
///    largest-body rule at its own level, the answer says so and the solve reports it.
///
/// `None` for no water.
///
/// **Example.** The home planet's inventory stood over 148 pockets and 3.2 million km² of inland
/// ground under its level (MEASURED at generator 19). Counted as ocean, their volume held the sea
/// down; over the connected ocean alone the same water stands higher, by the pockets' volume over
/// the ocean's area.
#[must_use]
pub fn sea_level_connected(
    lattice: &MacroLattice,
    z: &[i32],
    area: &[u64],
    water_km3: u64,
    with_load: bool,
) -> Option<ConnectedSea> {
    let first = sea_level_over(z, area, water_km3, with_load)?;
    let mask = ocean_mask(lattice, z, area, first);
    if mask.pockets == 0 {
        // One body under `L0`: it holds the whole inventory, so the answer balances by itself.
        return Some(ConnectedSea {
            level: first,
            mask,
            reseeded: false,
            settled: true,
        });
    }
    let target = sea_target(water_km3, with_load);
    let (level, reach) = connected_level(lattice, z, area, target, &mask.ocean, first);
    let at = ocean_mask(lattice, z, area, level);
    // A node of the first ocean stands in the seed's body at every level over `L0`, so the seed's
    // body is the largest exactly where the new mask holds it.
    let seed = mask.ocean.iter().position(|&o| o).unwrap_or(0);
    if at.ocean[seed] {
        let settled = joined_is_ocean(&reach, &at.ocean, level);
        return Some(ConnectedSea {
            level,
            mask: at,
            reseeded: false,
            settled,
        });
    }
    let (level, reach) = connected_level(lattice, z, area, target, &at.ocean, first);
    let mask = ocean_mask(lattice, z, area, level);
    let settled = joined_is_ocean(&reach, &mask.ocean, level);
    Some(ConnectedSea {
        level,
        mask,
        reseeded: true,
        settled,
    })
}

/// ★ Whether the nodes JOINED to the seed at `level` (`reach ≤ level`) are exactly the `ocean`
/// (review finding 4 of plan step 2): then the water the bisection counted is the water over the
/// ocean, and the answer balances.
fn joined_is_ocean(reach: &[i32], ocean: &[bool], level: i32) -> bool {
    reach.iter().zip(ocean).all(|(&r, &o)| (r <= level) == o)
}

/// The level at which the water joined to the `seed` body holds `target`, bisected between one step
/// under `first` (where even every node holds less) and the level that drowns the whole field; and
/// the level at which each node joins the seed, which the answer is checked against.
fn connected_level(
    lattice: &MacroLattice,
    z: &[i32],
    area: &[u64],
    target: u128,
    seed: &[bool],
    first: i32,
) -> (i32, Vec<i32>) {
    let reach = reach_from(lattice, z, seed);
    let (_, mut hi) = sea_bracket(z, area, target);
    let mut lo = i64::from(first) - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if water_joined(z, area, &reach, mid as i32) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (hi as i32, reach)
}

/// ★ THE LEVEL AT WHICH EACH NODE JOINS THE SEED (a priority flood, lowest first, ties to the
/// smaller index): a seed node's own height, and every other node the least, over the paths to the
/// seed, of the highest node on the path.
fn reach_from(lattice: &MacroLattice, z: &[i32], seed: &[bool]) -> Vec<i32> {
    let n = z.len();
    let mut reach = vec![i32::MAX; n];
    let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::new();
    for i in 0..n {
        if seed[i] {
            reach[i] = z[i];
            heap.push(Reverse((z[i], i as u32)));
        }
    }
    while let Some(Reverse((level, node))) = heap.pop() {
        for m in lattice.neighbours(node) {
            if m == NO_NODE || reach[m as usize] != i32::MAX {
                continue;
            }
            let raised = z[m as usize].max(level);
            reach[m as usize] = raised;
            heap.push(Reverse((raised, m)));
        }
    }
    reach
}

/// The water under a level over the nodes joined to the seed at that level, in area·sixteenths.
fn water_joined(z: &[i32], area: &[u64], reach: &[i32], level: i32) -> u128 {
    let mut sum = 0u128;
    for (k, &h) in z.iter().enumerate() {
        if (h < level) & (reach[k] <= level) {
            sum += u128::from(area[k]) * (u128::from((i64::from(level) - i64::from(h)) as u64));
        }
    }
    sum
}

/// ★ THE SEA'S LOAD ON THE OCEAN ALONE (2026-09-24, ruling B7 item 1): every OCEAN node sinks by
/// the water's weight ([`loaded_height`]); a pocket under the level carries no sea, so it keeps its
/// dry height, and a node over the level has nothing to carry.
///
/// **Example.** The ocean's floor 4 km under the level sinks about 1.8 km more under its water; a
/// pocket 200 m under the level in a continent's middle holds no sea and stands where the crust
/// put it.
#[must_use]
pub fn load_ocean(z_dry: &[i32], ocean: &[bool], level: i32) -> Vec<i32> {
    z_dry
        .iter()
        .zip(ocean)
        .map(|(&h, &wet)| if wet { loaded_height(h, level) } else { h })
        .collect()
}

/// The loaded height of a node under the sea: `h − (L − h)·ρ_w/(ρ_m − ρ_w)`, floored to a
/// sixteenth.
#[must_use]
pub fn loaded_height(z_dry: i32, level: i32) -> i32 {
    if z_dry >= level {
        return z_dry;
    }
    let sink = Gf::from_i64(i64::from(level) - i64::from(z_dry))
        * Gf::from_f64(WATER_DENSITY_KGM3 / (MANTLE_DENSITY_KGM3 - WATER_DENSITY_KGM3));
    (Gf::from_i64(i64::from(z_dry)) - sink).to_i64_floor() as i32
}

/// The relief law's CAP in metres: the lesser of its two arms — what the crust's thicknesses scale
/// by ([`EARTH_RELIEF_CAP_M`]).
#[must_use]
pub fn relief_cap_m(body: &BodyDefinition) -> f64 {
    let (strength, shape) = body.relief_arms_m();
    if strength < shape { strength } else { shape }
}

/// ★ THE INITIAL LAND of `body` over `lattice`, under its charter's land words: the plates drawn
/// from the body's own seed by the plate law, then [`initial_land_from`].
#[must_use]
pub fn initial_land(body: &BodyDefinition, lattice: &MacroLattice, words: &LandWords) -> Land {
    initial_land_on(body, lattice, words, crate::par::Cores::default())
}

/// [`initial_land`] ON THE SOLVE'S CORES (ruling B64): the same land on any number of them.
#[must_use]
pub fn initial_land_on(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &LandWords,
    cores: crate::par::Cores,
) -> Land {
    let plates = draw_plates(
        body.seed(),
        plate_count(body.ladder().radius_m(), words.elastic_thickness_m),
    );
    initial_land_from_on(body, lattice, words, plates, cores)
}

/// The initial land over STATED plates — the shipped kernel on a stated neighbourhood (06 §3.3),
/// so a test can hand a real small body of the world two plates and a sea.
#[must_use]
pub fn initial_land_from(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &LandWords,
    plates: Vec<Plate>,
) -> Land {
    initial_land_from_on(body, lattice, words, plates, crate::par::Cores::default())
}

/// [`initial_land_from`] ON THE SOLVE'S CORES (ruling B64): each node's cell mean on its own thread, the same land on
/// any number of them.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn initial_land_from_on(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &LandWords,
    plates: Vec<Plate>,
    cores: crate::par::Cores,
) -> Land {
    let n = lattice.node_count();
    let gravity = body.facts().gravity_mm_s2;
    let cap_m = relief_cap_m(body);
    let octaves = affinity_octaves(body);
    let radius = Gf::from_f64(body.ladder().radius_m());
    let width =
        Gf::from_f64(flexural_parameter_m(words.elastic_thickness_m, gravity) * BELT_HALF_WIDTHS);
    let relief = Gf::from_f64(body.relief_m());
    // 1. The plates and the affinity at every node's centre.
    let mut boundaries = Vec::with_capacity(n);
    let mut affinity = Vec::with_capacity(n);
    let mut noise = Vec::with_capacity(n);
    let mut area = Vec::with_capacity(n);
    for node in 0..n as u32 {
        let gi = lattice.direction(node);
        let d = dir_of(gi);
        let b = boundary_of(&plates, d);
        let a = affinity_noise(&octaves, gi);
        affinity.push(a + plates[usize::from(b.plate)].affinity);
        noise.push(a);
        boundaries.push(b);
        area.push(lattice.area_m2(node));
    }
    // 2. The crust threshold: the continental share of the AREA, over the centres' field.
    let threshold = area_quantile(&affinity, &area, 1.0 - CONTINENTAL_SHARE);
    // ★ THE FIRST ESTIMATE of the freeboard factor (generator 26), which the Andean margin's width
    // reads: the law over the node centres' own plates, before the margins and the older landscape.
    // MEASURED in the bench: the margins move the factor by a part in ten thousand, so the coast's
    // width is the solved one's within about ten metres.
    let first = {
        let mut part_o = Vec::with_capacity(n);
        let mut part_c = Vec::with_capacity(n);
        let mut crust = Vec::with_capacity(n);
        for k in 0..n {
            let share = crust_share(affinity[k], threshold);
            let own = &plates[usize::from(boundaries[k].plate)];
            let (o, c) = isostatic_parts_m(share, cap_m, own.crust_scatter);
            part_o.push(o);
            part_c.push(c);
            crust.push((share * Gf::from_f64(255.0) + Gf::HALF).to_i64_floor() as u8);
        }
        let flat = vec![Gf::ZERO; n];
        freeboard_factor(
            lattice,
            &part_o,
            &part_c,
            &crust,
            &area,
            words.water_km3,
            &flat,
        )
    };
    // ★ ONE SUB-CELL'S CRUST SHARE AND SCATTER (rulings B59 item 2, B61 item 1): its own plate's,
    // or — on the side of a boundary whose crust column stands higher, by the land's own Airy parts
    // at the first estimate — a linear ramp from the other plate's terms at the boundary to its own
    // over the margin's width ([`margin_width_m`]), so the crust thins toward the ocean as a
    // margin's crust does and a thick continent thins toward a thinner one. The lower side keeps
    // its own, so the two sides meet at the boundary. MEASURED before (the bench): a plate's
    // offset sign alone left cliffs of 46° at the 99th percentile.
    let terms = |k: usize, sb: &Boundary| -> (Gf, Gf) {
        let own = &plates[usize::from(sb.plate)];
        let own_share = crust_share(noise[k] + own.affinity, threshold);
        let Some(o) = sb.other else {
            return (own_share, own.crust_scatter);
        };
        let other = &plates[usize::from(o)];
        let other_share = crust_share(noise[k] + other.affinity, threshold);
        let column = |share: Gf, scatter: Gf| -> Gf {
            let (o, c) = isostatic_parts_m(share, cap_m, scatter);
            o + first * c
        };
        if column(own_share, own.crust_scatter) <= column(other_share, other.crust_scatter) {
            return (own_share, own.crust_scatter);
        }
        let w = margin_width_m(sb.kind, own, cap_m, first, words.elastic_thickness_m);
        let t = (radius * sb.sine / w).clamp(Gf::ZERO, Gf::ONE);
        (
            other_share + (own_share - other_share) * t,
            other.crust_scatter + (own.crust_scatter - other.crust_scatter) * t,
        )
    };
    // 3. Isostasy, dry, about the area-weighted mean — each node the MEAN of its sub-cells' Airy
    //    parts and crust shares (ruling B59 item 2) — and the room the belts have under the relief.
    //    ★ THE CONTINENTAL THICKNESS IS SOLVED BY THE FREEBOARD LAW (W7 step 2): a node's dry
    //    height is linear in the thickness factor, `A + f·B` (the oceanic part and the continental
    //    part of the Airy sum), so the field at any factor is one pass, and the factor is bisected
    //    until the sea the inventory fills stands at the continental platform's shelf quantile —
    //    with the older landscape's relief in the platform (ruling B60: MEASURED in the bench, the
    //    relief added after the law left the home planet with 18.4 % land against 24.6 %).
    let inv_sub = vd_recipe::bend::inv_n_of(lattice.edge * CELL_SUBCELLS);
    let count = Gf::from_i64(i64::from(CELL_SUBCELLS * CELL_SUBCELLS));
    // ★ ON THE SOLVE'S CORES (ruling B64): each node's mean reads only its own sub-cells, so the nodes are cut into
    // ranges, each with its own scratch, and joined in order.
    let parts = crate::par::per_range(n, cores.threads(), |r| {
        let mut subs = Vec::with_capacity((CELL_SUBCELLS * CELL_SUBCELLS) as usize);
        r.map(|k| {
            sub_boundaries(lattice, &plates, k as u32, inv_sub, &mut subs);
            let (mut o_sum, mut c_sum, mut s_sum) = (Gf::ZERO, Gf::ZERO, Gf::ZERO);
            for sb in &subs {
                let (share, scatter) = terms(k, sb);
                let (o, c) = isostatic_parts_m(share, cap_m, scatter);
                o_sum += o;
                c_sum += c;
                s_sum += share;
            }
            (
                o_sum / count,
                c_sum / count,
                (s_sum / count * Gf::from_f64(255.0) + Gf::HALF).to_i64_floor() as u8,
            )
        })
        .collect::<Vec<_>>()
    });
    let mut part_o = Vec::with_capacity(n);
    let mut part_c = Vec::with_capacity(n);
    let mut crust = Vec::with_capacity(n);
    for (o, c, s) in parts.into_iter().flatten() {
        part_o.push(o);
        part_c.push(c);
        crust.push(s);
    }
    let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    // The older landscape's relief, by each node's crust share: an ocean floor carries none.
    let older: Vec<Gf> = older_landscape_m(body, lattice)
        .into_iter()
        .zip(&crust)
        .map(|(r, &c)| Gf::from_i64(i64::from(c)) / Gf::from_i64(255) * r)
        .collect();
    let freeboard_factor = freeboard_factor(
        lattice,
        &part_o,
        &part_c,
        &crust,
        &area,
        words.water_km3,
        &older,
    );
    let (dry, tallest) = dry_field(&part_o, &part_c, &area, freeboard_factor);
    let room = (relief - tallest).greater(Gf::ZERO);
    // 4. Orogeny: the belts under the room — THE UPLIFT OVER THE AGE, kept beside the land; each
    //    node the mean of its sub-cells' profiles. ★ THE CRUST SHARE READ CONTINUOUSLY (ruling B61
    //    item 1): a sub-cell's profile blends the continental and the oceanic profile by its own
    //    crust share, where the yes-or-no test at one half put a belt's edge on the half-share line.
    //    ★ THE ANDEAN MARGIN'S TRENCH (ruling B60 item 1): a continent's side of an ocean diving
    //    under it holds the trench's inner wall — the outer wall's own profile, so the trough is one
    //    trough with its axis on the boundary — and its wedge rises from the coast inland (the
    //    belts' law).
    let mut uplift: Vec<i32> = crate::par::per_range(n, cores.threads(), |r| {
        let mut subs = Vec::with_capacity((CELL_SUBCELLS * CELL_SUBCELLS) as usize);
        r.map(|k| {
            sub_boundaries(lattice, &plates, k as u32, inv_sub, &mut subs);
            let mut sum = Gf::ZERO;
            for sb in &subs {
                let own = &plates[usize::from(sb.plate)];
                let (other_continental, other_lower) = match sb.other {
                    Some(o) => (plates[usize::from(o)].affinity >= Gf::ZERO, o < sb.plate),
                    None => (false, false),
                };
                let amplitude = room
                    * (sb.convergence.abs() / Gf::TWO).clamp(Gf::ZERO, Gf::ONE)
                    * (Gf::ONE - own.age * Gf::HALF);
                let distance = radius * sb.sine;
                let andean = (sb.kind == Kind::Convergent)
                    & (own.affinity >= Gf::ZERO)
                    & sb.other
                        .is_some_and(|o| plates[usize::from(o)].affinity < Gf::ZERO);
                let (continental, oceanic) = if andean {
                    let wall = uplift_m(
                        Kind::Convergent,
                        false,
                        true,
                        true,
                        distance,
                        amplitude,
                        width,
                    );
                    (wall, wall)
                } else {
                    (
                        uplift_m(
                            sb.kind,
                            true,
                            other_continental,
                            !other_lower,
                            distance,
                            amplitude,
                            width,
                        ),
                        uplift_m(
                            sb.kind,
                            false,
                            other_continental,
                            !other_lower,
                            distance,
                            amplitude,
                            width,
                        ),
                    )
                };
                let s = terms(k, sb).0;
                sum += continental * s + oceanic * (Gf::ONE - s);
            }
            (sum / count * steps).to_i64_floor() as i32
        })
        .collect::<Vec<i32>>()
    })
    .into_iter()
    .flatten()
    .collect();
    // The dry field in sixteenths, the older landscape's relief in sixteenths added to it.
    let z_dry: Vec<i32> = dry
        .iter()
        .zip(&older)
        .map(|(&h, &r)| (h * steps).to_i64_floor() as i32 + (r * steps).to_i64_floor() as i32)
        .collect();
    // 5. The sea, with the load — ★ OVER THE CONNECTED OCEAN ONLY (2026-09-24, ruling B7 item 1):
    //    a pocket under the level that no path joins to the ocean carries neither the sea's water
    //    nor its load.
    let sea = sea_level_connected(lattice, &z_dry, &area, words.water_km3, true);
    let (sea_z, ocean, pockets, pocket_nodes) = match sea {
        Some(s) => (
            Some(s.level),
            s.mask.ocean,
            s.mask.pockets,
            s.mask.pocket_nodes,
        ),
        None => (None, vec![false; n], 0, 0),
    };
    let z: Vec<i32> = match sea_z {
        Some(level) => load_ocean(&z_dry, &ocean, level),
        None => z_dry,
    };
    // ★ THE UPLIFT KEEPS THE ENVELOPE ON BOTH SIDES: a trench's subsidence may not carry a node
    // under `−relief` any more than a belt may carry one over `+relief` (MEASURED on the home
    // planet: the trenches reached −9 333 m under an 8 276 m relief and the whole planet was
    // scaled down by a ninth at the end). Each node's total is clamped against its LOADED height,
    // so its land plus its uplift stays inside the relief.
    let relief_steps = (relief * steps).to_i64_floor() as i32;
    for (u, &h) in uplift.iter_mut().zip(&z) {
        *u = (*u).clamp(-relief_steps - h, relief_steps - h);
    }
    // ★ THE ROCK MAP, drawn LAST (slice 8d step 2): the crust, the uplift as the envelope clamped
    // it, and the loaded height against the sea the inventory solved for. It is read here because
    // the solve keeps none of those three words to the end.
    let province: Vec<u8> = crust
        .iter()
        .zip(&uplift)
        .zip(&z)
        .zip(&boundaries)
        .map(|(((&c, &u), &h), b)| {
            let other_continental = b
                .other
                .is_some_and(|o| plates[usize::from(o)].affinity >= Gf::ZERO);
            province_of(c, u, h, sea_z, b.kind, other_continental).code()
        })
        .collect();
    Land {
        z,
        uplift,
        sea_z,
        province,
        plate: boundaries.iter().map(|b| b.plate).collect(),
        kind: boundaries.iter().map(|b| b.kind as u8).collect(),
        boundary_m: boundaries
            .iter()
            .map(|b| match b.other {
                Some(_) => (radius * b.sine).to_i64_floor() as u32,
                None => u32::MAX,
            })
            .collect(),
        crust,
        plates,
        threshold,
        freeboard_factor,
        ocean,
        pockets,
        pocket_nodes,
    }
}

/// ★ THE TWO PARTS OF A NODE'S AIRY HEIGHT ([`isostatic_height_m`] split): the oceanic part, which
/// no thickness factor touches, and the continental part, which the freeboard factor multiplies —
/// `height(f) = o + f · c`. The continental thickness alone scales: the density is the crust's own.
#[must_use]
pub fn isostatic_parts_m(share: Gf, relief_cap_m: f64, crust_scatter: Gf) -> (Gf, Gf) {
    let cap_ratio = Gf::from_f64(relief_cap_m) / Gf::from_f64(EARTH_RELIEF_CAP_M);
    let t_c = Gf::from_f64(EARTH_CONTINENTAL_CRUST_M) * cap_ratio * crust_scatter;
    let t_o = Gf::from_f64(EARTH_OCEANIC_CRUST_M) * cap_ratio;
    let rho = Gf::from_f64(OCEANIC_CRUST_DENSITY_KGM3)
        + Gf::from_f64(CONTINENTAL_CRUST_DENSITY_KGM3 - OCEANIC_CRUST_DENSITY_KGM3) * share;
    let buoyancy = Gf::ONE - rho / Gf::from_f64(MANTLE_DENSITY_KGM3);
    (t_o * (Gf::ONE - share) * buoyancy, t_c * share * buoyancy)
}

/// The dry field at a freeboard factor, centred on its area-weighted mean, and the tallest
/// magnitude in it (the belts' room is the relief less it).
#[must_use]
pub fn dry_field(part_o: &[Gf], part_c: &[Gf], area: &[u64], factor: Gf) -> (Vec<Gf>, Gf) {
    let n = part_o.len();
    let mut dry = Vec::with_capacity(n);
    let mut sum = Gf::ZERO;
    let mut total = Gf::ZERO;
    for k in 0..n {
        let h = part_o[k] + factor * part_c[k];
        let a = Gf::from_i64(area[k] as i64);
        sum += h * a;
        total += a;
        dry.push(h);
    }
    let mean = sum / total;
    let mut tallest = Gf::ZERO;
    for h in &mut dry {
        *h -= mean;
        tallest = tallest.greater(h.abs());
    }
    (dry, tallest)
}

/// ★ THE FREEBOARD FACTOR ([`FREEBOARD_FACTOR_RANGE`]): the continental thickness factor at which
/// the sea the inventory fills stands at the continental crust's shelf quantile
/// ([`EARTH_LAND_SHARE`] over [`CONTINENTAL_SHARE`], from the bottom). The sea
/// falls against the platform as the factor grows (a taller platform deepens the basins), so a
/// bisection finds it; a factor at either end of the range is a water world or a desert world.
/// ONE on a body with no water (no sea to stand at).
///
/// ★ THE SEA HERE IS THE CONNECTED OCEAN'S (2026-09-24, ruling B7 item 1; plan question 9): the
/// same law the first land and the solve read, so a pocket on the first land holds no sea here
/// either.
///
/// ★ THE PLATFORM CARRIES ITS RELIEF (generator 26, ruling B60): `relief_m` is the surface relief
/// each node carries over its Airy height (the older landscape's), so the law reads the sea and
/// the platform's quantile on the field the land will stand on.
#[must_use]
pub fn freeboard_factor(
    lattice: &MacroLattice,
    part_o: &[Gf],
    part_c: &[Gf],
    crust: &[u8],
    area: &[u64],
    water_km3: u64,
    relief_m: &[Gf],
) -> Gf {
    if water_km3 == 0 {
        return Gf::ONE;
    }
    let n = part_o.len();
    let continental: Vec<usize> = (0..n).filter(|&k| crust[k] >= 128).collect();
    if continental.is_empty() {
        return Gf::ONE;
    }
    let cont_area: Vec<u64> = continental.iter().map(|&k| area[k]).collect();
    let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    // The sea's height over the platform at a factor, in sixteenths.
    let over = |factor: Gf| -> i64 {
        let (mut dry, _) = dry_field(part_o, part_c, area, factor);
        for (d, &r) in dry.iter_mut().zip(relief_m) {
            *d += r;
        }
        let z_dry: Vec<i32> = dry
            .iter()
            .map(|&h| (h * steps).to_i64_floor() as i32)
            .collect();
        let sea = sea_level_connected(lattice, &z_dry, area, water_km3, true)
            .map_or(i32::MIN, |s| s.level);
        let platform_values: Vec<Gf> = continental.iter().map(|&k| dry[k]).collect();
        let platform = (area_quantile(
            &platform_values,
            &cont_area,
            1.0 - EARTH_LAND_SHARE / CONTINENTAL_SHARE,
        ) * steps)
            .to_i64_floor();
        i64::from(sea) - platform
    };
    let (mut lo, mut hi) = (
        Gf::from_f64(FREEBOARD_FACTOR_RANGE.0),
        Gf::from_f64(FREEBOARD_FACTOR_RANGE.1),
    );
    if over(lo) <= 0 {
        return lo;
    }
    if over(hi) >= 0 {
        return hi;
    }
    let mut k = 0;
    while k < FREEBOARD_BISECTIONS {
        let mid = (lo + hi) * Gf::HALF;
        if over(mid) > 0 {
            lo = mid;
        } else {
            hi = mid;
        }
        k += 1;
    }
    (lo + hi) * Gf::HALF
}

/// ★ THE HYPSOMETRY: the area under each height bin of [`HYPSOMETRY_BIN_M`], from the lowest
/// node up. The curve's shape is gate G-LAND-HYPSOMETRY's evidence.
#[must_use]
pub fn hypsometry(z: &[i32], area: &[u64]) -> (i32, Vec<u64>) {
    let lo = z.iter().copied().min().unwrap_or(0);
    let hi = z.iter().copied().max().unwrap_or(0);
    let bin = HYPSOMETRY_BIN_M as i64 * i64::from(Z_STEPS_PER_M);
    let bins = ((i64::from(hi) - i64::from(lo)) / bin + 1) as usize;
    let mut out = vec![0u64; bins];
    for (k, &h) in z.iter().enumerate() {
        out[((i64::from(h) - i64::from(lo)) / bin) as usize] += area[k];
    }
    (lo, out)
}

/// ★ GATE G-LAND-HYPSOMETRY's reading: the two largest humps of a hypsometry and the valley
/// between them. `Some((low_bin, high_bin, valley_share))` where `valley_share` is the shallowest
/// bin between the humps over the LOWER hump; `None` when the curve has one hump. Two humps are
/// two local maxima of the smoothed curve, each at least a tenth of the largest bin.
#[must_use]
pub fn humps(hist: &[u64]) -> Option<(usize, usize, f64)> {
    let smooth: Vec<u64> = (0..hist.len())
        .map(|k| {
            let l = if k > 0 { hist[k - 1] } else { 0 };
            let r = if k + 1 < hist.len() { hist[k + 1] } else { 0 };
            l + 2 * hist[k] + r
        })
        .collect();
    let top = smooth.iter().copied().max().unwrap_or(0);
    let floor = top / 10;
    let mut peaks: Vec<usize> = Vec::new();
    for k in 0..smooth.len() {
        let l = if k > 0 { smooth[k - 1] } else { 0 };
        let r = if k + 1 < smooth.len() {
            smooth[k + 1]
        } else {
            0
        };
        if smooth[k] >= floor && smooth[k] > l && smooth[k] >= r {
            peaks.push(k);
        }
    }
    // The two largest peaks, then in height order.
    peaks.sort_by_key(|&k| std::cmp::Reverse(smooth[k]));
    let (a, b) = (*peaks.first()?, *peaks.get(1)?);
    let (low, high) = (a.min(b), a.max(b));
    let valley = smooth[low..=high].iter().copied().min().unwrap_or(0);
    let lower = smooth[low].min(smooth[high]);
    Some((low, high, valley as f64 / lower as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::{home_land_words, home_moon, home_moon_land_words, home_planet};
    use vd_seed::bend::Face;

    fn gf(v: f64) -> Gf {
        Gf::from_f64(v)
    }

    /// The plate law: the home planet draws fourteen, Earth itself fifteen, the moon one.
    #[test]
    fn the_plate_law_on_the_home_bodies() {
        assert_eq!(plate_count(home_planet().ladder().radius_m(), 36_789), 14);
        assert_eq!(plate_count(EARTH_RADIUS_M, 35_000), 15);
        assert_eq!(plate_count(home_moon().ladder().radius_m(), 1_094_578), 1);
        assert_eq!(plate_count(EARTH_RADIUS_M, 0), PLATES_MAX);
    }

    /// The flexural parameter on Earth's numbers: about 120 km at 35 km of elastic thickness.
    #[test]
    fn the_flexural_parameter_on_earth() {
        let alpha = flexural_parameter_m(35_000, 9_810);
        assert!((110_000.0..135_000.0).contains(&alpha), "{alpha}");
    }

    /// Isostasy on Earth's numbers: the continent 5 303 m, the floor 848 m, a dry step of 4 455 m.
    #[test]
    fn isostasy_reproduces_the_dry_step() {
        let cont = isostatic_height_m(Gf::ONE, EARTH_RELIEF_CAP_M, Gf::ONE).to_f64();
        let ocean = isostatic_height_m(Gf::ZERO, EARTH_RELIEF_CAP_M, Gf::ONE).to_f64();
        assert!((cont - 5_303.0).abs() < 1.0, "{cont}");
        assert!((ocean - 848.5).abs() < 1.0, "{ocean}");
        assert!((cont - ocean - 4_454.5).abs() < 1.0);
        // The loaded floor stands 6 476 m under the continent when the sea fills to the continent.
        let level = 5_303 * Z_STEPS_PER_M;
        let floor = loaded_height(848 * Z_STEPS_PER_M, level) / Z_STEPS_PER_M;
        assert!((5_303 - floor - 6_476).abs() <= 2, "{floor}");
        assert_eq!(loaded_height(100, 50), 100);
    }

    /// The smoothstep and the bump: their ends, their middles, and the exact zero outside.
    #[test]
    fn the_profiles_are_bounded_and_exactly_zero_outside() {
        assert_eq!(smoothstep(gf(-1.0)), Gf::ZERO);
        assert_eq!(smoothstep(gf(2.0)), Gf::ONE);
        assert_eq!(smoothstep(Gf::HALF), Gf::HALF);
        assert_eq!(bump(Gf::ZERO), Gf::ONE);
        assert_eq!(bump(gf(1.0)), Gf::ZERO);
        assert_eq!(bump(gf(-3.0)), Gf::ZERO);
        assert_eq!(bump(Gf::HALF), gf(0.5625));
        assert_eq!(crust_share(gf(0.3), gf(0.3)), Gf::HALF);
        assert_eq!(crust_share(gf(-1.0), gf(0.3)), Gf::ZERO);
        assert_eq!(crust_share(gf(1.0), gf(0.3)), Gf::ONE);
    }

    /// Every profile arm: a belt, a trench under a continent, the ocean-ocean pair (a trench on
    /// the lower-indexed plate, an arc on the other), a rift, a ridge, a scarp, no boundary.
    #[test]
    fn every_uplift_profile() {
        let (a, w) = (gf(8_000.0), gf(240_000.0));
        let at = |kind, own, other, lower, b: f64| {
            uplift_m(kind, own, other, lower, gf(b), a, w).to_f64()
        };
        assert_eq!(at(Kind::Convergent, true, false, true, 0.0), 8_000.0);
        assert_eq!(at(Kind::Convergent, true, false, true, 240_000.0), 0.0);
        assert_eq!(at(Kind::Convergent, false, true, true, 0.0), -4_000.0);
        assert_eq!(at(Kind::Convergent, false, true, true, 60_000.0), 0.0);
        assert_eq!(at(Kind::Convergent, false, false, true, 0.0), -4_000.0);
        assert_eq!(at(Kind::Convergent, false, false, false, 0.0), 4_000.0);
        assert_eq!(at(Kind::Divergent, true, true, true, 0.0), -2_000.0);
        assert_eq!(at(Kind::Divergent, false, true, true, 0.0), 2_000.0);
        assert_eq!(at(Kind::Transform, true, true, true, 0.0), 0.0);
        assert_eq!(at(Kind::None, false, false, false, 0.0), 0.0);
    }

    /// A stated pair of plates: two sites on the equator drifting toward each other are convergent
    /// with the sine distance to their bisector; drifting apart, divergent; sliding, transform. One
    /// plate: no boundary. A tie in the scan keeps the smaller index.
    #[test]
    fn the_boundary_of_a_stated_pair() {
        let plate = |site: [f64; 3], drift: [f64; 3]| Plate {
            site: [gf(site[0]), gf(site[1]), gf(site[2])],
            drift: [gf(drift[0]), gf(drift[1]), gf(drift[2])],
            affinity: Gf::ZERO,
            age: Gf::ZERO,
            crust_scatter: Gf::ONE,
        };
        let s = std::f64::consts::FRAC_1_SQRT_2;
        // Plate 0 at +X drifting toward +Y; plate 1 at +Y drifting toward +X: they push together.
        let push = [
            plate([1.0, 0.0, 0.0], [0.0, 0.5, 0.0]),
            plate([0.0, 1.0, 0.0], [0.5, 0.0, 0.0]),
        ];
        let b = boundary_of(&push, [gf(0.6), gf(0.8), Gf::ZERO]);
        assert_eq!((b.plate, b.other), (1, Some(0)));
        assert_eq!(b.kind, Kind::Convergent);
        assert!(b.convergence > Gf::ZERO);
        // On the bisector (the 45° line) the sine is zero; at +X it is sin 45°.
        let on = boundary_of(&push, [gf(s), gf(s), Gf::ZERO]);
        assert!(on.sine.to_f64().abs() < 1e-12);
        let far = boundary_of(&push, [Gf::ONE, Gf::ZERO, Gf::ZERO]);
        assert!((far.sine.to_f64() - s).abs() < 1e-12);
        assert_eq!(far.plate, 0);
        // Drifting apart.
        let pull = [
            plate([1.0, 0.0, 0.0], [0.0, -0.5, 0.0]),
            plate([0.0, 1.0, 0.0], [-0.5, 0.0, 0.0]),
        ];
        assert_eq!(
            boundary_of(&pull, [gf(0.6), gf(0.8), Gf::ZERO]).kind,
            Kind::Divergent
        );
        // Sliding past: both drift along +Z.
        let slide = [
            plate([1.0, 0.0, 0.0], [0.0, 0.0, 0.5]),
            plate([0.0, 1.0, 0.0], [0.0, 0.0, 0.3]),
        ];
        assert_eq!(
            boundary_of(&slide, [gf(0.6), gf(0.8), Gf::ZERO]).kind,
            Kind::Transform
        );
        // One plate.
        let one = boundary_of(&push[..1], [Gf::ONE, Gf::ZERO, Gf::ZERO]);
        assert_eq!((one.other, one.kind, one.sine), (None, Kind::None, Gf::ONE));
        // A tie: the same site twice keeps index 0 as nearest and 1 as second.
        let twin = [push[0], push[0]];
        let t = boundary_of(&twin, [Gf::ONE, Gf::ZERO, Gf::ZERO]);
        assert_eq!((t.plate, t.other), (0, Some(1)));
        // A zero vector normalises to itself.
        assert_eq!(normalise([Gf::ZERO; 3]), [Gf::ZERO; 3]);
    }

    /// The plates drawn from a seed are unit sites with tangent drifts of length under one, and
    /// the same seed draws the same plates.
    #[test]
    fn drawn_plates_are_unit_sites_with_tangent_drifts() {
        let plates = draw_plates(home_planet().seed(), 14);
        assert_eq!(plates.len(), 14);
        for p in &plates {
            assert!((dot(p.site, p.site).to_f64() - 1.0).abs() < 1e-9);
            assert!(
                dot(p.site, p.drift).to_f64().abs() < 1e-9,
                "the drift is tangent"
            );
            assert!(dot(p.drift, p.drift).to_f64() < 1.0);
            assert!(p.affinity.to_f64().abs() <= PLATE_AFFINITY);
            // The sourced spread: `1 ± σ·√3` with Earth's stable continents' σ.
            let half = EARTH_CONTINENT_SPREAD * 3f64.sqrt();
            assert!((1.0 - half..1.0 + half).contains(&p.crust_scatter.to_f64()));
        }
        assert_eq!(draw_plates(home_planet().seed(), 14), plates);
        assert_ne!(draw_plates(home_moon().seed(), 14), plates);
    }

    /// The area quantile on a stated field: the value under which the share of the area lies; a
    /// flat field answers itself.
    #[test]
    fn the_area_quantile_on_a_stated_field() {
        let values: Vec<Gf> = (0..1000).map(|k| gf(f64::from(k) / 1000.0)).collect();
        let area = vec![1u64; 1000];
        let q = area_quantile(&values, &area, 0.6).to_f64();
        assert!((q - 0.6).abs() < 0.002, "{q}");
        assert_eq!(area_quantile(&[gf(0.3); 4], &[1; 4], 0.6), gf(0.3));
        // Heavier nodes weigh more: two nodes, the low one holding nine tenths of the area.
        let q = area_quantile(&[Gf::ZERO, Gf::ONE], &[9, 1], 0.5).to_f64();
        assert!(q < 0.5, "{q}");
    }

    /// The sea over a stated field: a two-level field, half at 0 and half at 1 000 m, with the
    /// water to fill the low half to 500 m — the level lands at 500 m divided by the load; no
    /// water gives no sea; more water than the field can hold drowns it.
    #[test]
    fn the_sea_over_a_two_level_field() {
        let low = 0i32;
        let high = 1_000 * Z_STEPS_PER_M;
        let z = vec![low, high];
        let area = vec![1_000_000_000_000u64, 1_000_000_000_000u64];
        // 500 m of water over 10¹² m² = 5 × 10¹⁴ m³ = 500 000 km³; the load lets the column stand
        // 1.454 times taller than the dry gap, so the level stands 500 / 1.454 = 344 m over the
        // dry floor (and 500 m over the loaded one).
        let level = sea_level_over(&z, &area, 500_000, true).expect("a sea");
        let level_m = f64::from(level) / f64::from(Z_STEPS_PER_M);
        assert!((level_m - 343.9).abs() < 1.0, "{level_m}");
        let column_m = f64::from(level - loaded_height(low, level)) / f64::from(Z_STEPS_PER_M);
        assert!((column_m - 500.0).abs() < 1.0, "{column_m}");
        assert_eq!(sea_level_over(&z, &area, 0, true), None);
        let flooded = sea_level_over(&z, &area, 10_000_000, true).expect("a sea");
        assert!(flooded > high);
        assert!(loaded_height(low, level) < low);
        assert_eq!(loaded_height(high, level), high);
    }

    /// The moon's lattice and its node areas.
    fn moon_areas() -> (MacroLattice, Vec<u64>) {
        let lattice = MacroLattice::of(&home_moon()).expect("a lattice");
        let area = (0..lattice.node_count() as u32)
            .map(|k| lattice.area_m2(k))
            .collect();
        (lattice, area)
    }

    /// ★ TEST 13 (plan step 2 part 3a; ruling B7 item 1): THE SEA OVER THE CONNECTED OCEAN. Land at
    /// 1 000 m everywhere, an ocean of one whole face and a pocket of a hundred nodes on another
    /// face, both floors at −1 000 m. Over every node the inventory stands at `L0`; over the ocean
    /// alone it stands at the least level whose water over the ocean's own area holds the whole
    /// target — higher by the pocket's share, which is the pocket's volume at `L0` over the ocean's
    /// area within one step. Without the pocket the answer is `L0` itself, on the one-body path.
    #[test]
    fn the_connected_sea_stands_over_the_ocean_alone() {
        let (lattice, area) = moon_areas();
        let n = lattice.node_count();
        let land = 1_000 * Z_STEPS_PER_M;
        let floor = -1_000 * Z_STEPS_PER_M;
        let mut z = vec![land; n];
        let (mut ocean_area, mut pocket_area) = (0u128, 0u128);
        for k in 0..n {
            let (face, i, j) = lattice.split(k as u32);
            if face == Face::PosX {
                z[k] = floor;
                ocean_area += u128::from(area[k]);
            } else if face == Face::NegX && (20..30).contains(&i) && (20..30).contains(&j) {
                z[k] = floor;
                pocket_area += u128::from(area[k]);
            }
        }
        let water_km3 = 20_000;
        let target = sea_target(water_km3, false);
        let first = sea_level_over(&z, &area, water_km3, false).expect("a sea");
        let sea = sea_level_connected(&lattice, &z, &area, water_km3, false).expect("a sea");
        assert!(!sea.reseeded);
        assert!(sea.settled);
        assert!(holds_within_a_step(&z, &area, &sea, target));
        assert_eq!((sea.mask.pockets, sea.mask.pocket_nodes), (1, 100));
        // By hand: the least level `L` with `(L − floor)·A ≥ target`, over the ocean's area alone
        // and over both.
        let least = |a: u128| i32::try_from(target.div_ceil(a)).expect("a level") + floor;
        assert_eq!(first, least(ocean_area + pocket_area));
        assert_eq!(sea.level, least(ocean_area));
        let pocket_share = (i64::from(first - floor) * pocket_area as i64) / ocean_area as i64;
        assert!(
            (i64::from(sea.level - first) - pocket_share).abs() <= 1,
            "{} over {first}: the pocket's share is {pocket_share}",
            sea.level
        );
        for k in 0..n {
            assert_eq!(sea.mask.ocean[k], lattice.split(k as u32).0 == Face::PosX);
        }
        // Raise the pocket over the land: one body, and the level is `L0` exactly.
        let lone: Vec<i32> = (0..n)
            .map(|k| {
                if lattice.split(k as u32).0 == Face::PosX {
                    floor
                } else {
                    land
                }
            })
            .collect();
        let one = sea_level_connected(&lattice, &lone, &area, water_km3, false).expect("a sea");
        assert_eq!(
            one.level,
            sea_level_over(&lone, &area, water_km3, false).expect("a sea")
        );
        assert!(one.settled);
        assert!(holds_within_a_step(&lone, &area, &one, target));
        assert_eq!((one.mask.pockets, one.mask.pocket_nodes), (0, 0));
        assert_eq!(sea_level_connected(&lattice, &lone, &area, 0, false), None);
    }

    /// ★ TEST 14 (plan step 2 part 3a; plan question 1): THE LARGEST BODY IS THE OCEAN, AND A TIE
    /// GOES TO THE BODY WHOSE LOWEST NODE INDEX IS SMALLER. Two single nodes of one area on two
    /// faces: the first by index is the ocean. A later body of two nodes is larger and wins. A
    /// level under every node gives no ocean at all.
    #[test]
    fn the_largest_body_is_the_ocean_and_a_tie_takes_the_smaller_index() {
        let (lattice, area) = moon_areas();
        let n = lattice.node_count();
        let mut z = vec![100; n];
        let a = lattice.index(Face::PosX, 10, 10) as usize;
        let b = lattice.index(Face::NegX, 10, 10) as usize;
        assert_eq!(area[a], area[b], "one area on the two faces");
        assert!(a < b);
        z[a] = 0;
        z[b] = 0;
        let tie = ocean_mask(&lattice, &z, &area, 0);
        assert!(tie.ocean[a]);
        assert!(!tie.ocean[b]);
        assert_eq!((tie.pockets, tie.pocket_nodes), (1, 1));
        let c = lattice.index(Face::NegX, 11, 10) as usize;
        z[c] = 0;
        let larger = ocean_mask(&lattice, &z, &area, 0);
        assert!(!larger.ocean[a]);
        assert!(larger.ocean[b]);
        assert!(larger.ocean[c]);
        let none = ocean_mask(&lattice, &z, &area, -1);
        assert!(none.ocean.iter().all(|&o| !o));
        assert_eq!((none.pockets, none.pocket_nodes), (0, 0));
        // Every node under the level: one body over the whole sphere, the cube's corners included.
        let all = ocean_mask(&lattice, &z, &area, i32::MAX);
        assert!(all.ocean.iter().all(|&o| o));
    }

    /// ★ TEST 15 (plan step 2 part 3a): THE LOAD FALLS ON THE OCEAN ALONE. An ocean node under the
    /// level sinks by `loaded_height`; a pocket node under the same level keeps its dry height; a
    /// node over the level is untouched.
    #[test]
    fn the_sea_loads_the_ocean_and_never_a_pocket() {
        let level = 0;
        let z_dry = [-1_600, -1_600, 400];
        let ocean = [true, false, false];
        let loaded = load_ocean(&z_dry, &ocean, level);
        assert_eq!(loaded[0], loaded_height(-1_600, level));
        assert!(loaded[0] < -1_600);
        assert_eq!(loaded[1], -1_600, "a pocket carries no sea");
        assert_eq!(loaded[2], 400);
    }

    /// ★ TEST 16 (plan step 2 part 3a): THE RE-SEED. At `L0` a shallow basin of a hundred nodes is
    /// the largest body, so it seeds the connected solve; over it alone the water must stand so
    /// high that a second basin — ten deep nodes and a wide shelf just over `L0` — floods past it.
    /// The seed's body is then not the largest, so the solve seeds again once from the body that
    /// is, states it, and its ocean is that body.
    #[test]
    fn the_connected_sea_seeds_again_from_the_body_that_is_largest() {
        let (lattice, area) = moon_areas();
        let n = lattice.node_count();
        let mut z = vec![1_000 * Z_STEPS_PER_M; n];
        let (mut shallow, mut deep) = (Vec::new(), Vec::new());
        for (k, h) in z.iter_mut().enumerate().take(n) {
            let (face, i, j) = lattice.split(k as u32);
            if face == Face::PosX && (10..20).contains(&i) && (10..20).contains(&j) {
                *h = 0;
                shallow.push(k);
            } else if face == Face::NegX && (10..20).contains(&i) && j == 10 {
                *h = -1_600;
                deep.push(k);
            } else if face == Face::NegX && (10..50).contains(&i) && (11..41).contains(&j) {
                *h = 80;
            }
        }
        let first = sea_level_over(&z, &area, 72, false).expect("a sea");
        assert!(
            (0..80).contains(&first),
            "L0 {first} between the shallow floor and the shelf"
        );
        assert!(ocean_mask(&lattice, &z, &area, first).ocean[shallow[0]]);
        let sea = sea_level_connected(&lattice, &z, &area, 72, false).expect("a sea");
        assert!(sea.reseeded);
        assert!(sea.level > 80, "the shelf floods: {}", sea.level);
        assert!(sea.mask.ocean[deep[0]]);
        assert!(!sea.mask.ocean[shallow[0]]);
        assert!(
            sea.settled,
            "the re-seed body is the largest at its own level"
        );
        assert!(holds_within_a_step(&z, &area, &sea, sea_target(72, false)));
    }

    /// ★ THE ANSWER'S OCEAN HOLDS THE TARGET, WITHIN ONE STEP: the water over the answer's own ocean
    /// at its level reaches the target, and one step lower over the same wet area it does not.
    fn holds_within_a_step(z: &[i32], area: &[u64], sea: &ConnectedSea, target: u128) -> bool {
        let (mut held, mut wet) = (0u128, 0u128);
        for k in 0..z.len() {
            if sea.mask.ocean[k] & (z[k] < sea.level) {
                let a = u128::from(area[k]);
                held += a * u128::from((i64::from(sea.level) - i64::from(z[k])) as u64);
                wet += a;
            }
        }
        (held >= target) & (held - wet < target)
    }

    /// ★ TEST 16b (review finding 4 of plan step 2): A RE-SEED THAT DOES NOT SETTLE IS SAID. At `L0`
    /// a basin A of twenty nodes, 1 000 m deep, is the largest body, beside a narrow hole B of ten
    /// nodes, 10 000 m deep, with a shelf of thirty nodes at 500 m around it. Over A alone the
    /// water stands at about 4 000 m, where B and its flooded shelf are larger than A, so the solve
    /// seeds again from B. Over B the target stands UNDER the shelf, where A is the largest body
    /// again. No single seed balances this field: the answer's level is B's and its ocean is A's,
    /// and the answer says so — never a silent sea whose ocean holds another volume.
    #[test]
    fn a_reseed_whose_body_loses_the_largest_rule_is_stated_unsettled() {
        let (lattice, area) = moon_areas();
        let n = lattice.node_count();
        let m = Z_STEPS_PER_M;
        let mut z = vec![8_000 * m; n];
        let (mut a_nodes, mut hole) = (Vec::new(), Vec::new());
        for (k, h) in z.iter_mut().enumerate().take(n) {
            let (face, i, j) = lattice.split(k as u32);
            if face == Face::PosX && (24..44).contains(&i) && j == 34 {
                *h = -1_000 * m;
                a_nodes.push(k);
            } else if face == Face::NegX && (24..34).contains(&i) && j == 34 {
                *h = -10_000 * m;
                hole.push(k);
            } else if face == Face::NegX && (24..54).contains(&i) && j == 35 {
                *h = 500 * m;
            }
        }
        // The target: the hole alone brim full to the level 0.
        let water_m3: u128 = hole.iter().map(|&k| u128::from(area[k]) * 10_000).sum();
        let water_km3 = (water_m3 / 1_000_000_000) as u64;
        let first = sea_level_over(&z, &area, water_km3, false).expect("a sea");
        let at_first = ocean_mask(&lattice, &z, &area, first);
        assert!(at_first.ocean[a_nodes[0]], "A is the largest at L0 {first}");
        assert_eq!(at_first.pockets, 1);
        let sea = sea_level_connected(&lattice, &z, &area, water_km3, false).expect("a sea");
        assert!(sea.reseeded);
        assert!(
            (-1_000 * m..500 * m).contains(&sea.level),
            "over B the level stands under the shelf: {}",
            sea.level
        );
        assert!(sea.mask.ocean[a_nodes[0]], "A is the largest again");
        assert!(!sea.mask.ocean[hole[0]]);
        assert!(!sea.settled);
        assert!(!holds_within_a_step(
            &z,
            &area,
            &sea,
            sea_target(water_km3, false)
        ));
    }

    /// The hypsometry and the humps on stated curves: two humps with a valley, one hump, an empty
    /// curve.
    #[test]
    fn humps_on_stated_curves() {
        let two = [1u64, 8, 20, 8, 1, 0, 1, 6, 30, 6, 1];
        let (low, high, valley) = humps(&two).expect("two humps");
        assert_eq!((low, high), (2, 8));
        assert!(valley < 0.1, "{valley}");
        let one = [1u64, 4, 9, 4, 1];
        assert_eq!(humps(&one), None);
        assert_eq!(humps(&[]), None);
        let (lo, hist) = hypsometry(&[0, 16 * 250, 16 * 250 + 1, 16 * 1_000], &[1, 2, 3, 4]);
        assert_eq!(lo, 0);
        assert_eq!(hist, vec![1, 5, 0, 0, 4]);
    }

    /// ★ THE DRIVER ON THE MOON: a stagnant lid — one plate, no boundary anywhere, no sea — whose
    /// crust field still floats two levels (highlands and lowlands), so its hypsometry has two
    /// humps; the heights stay inside the relief law's band.
    #[test]
    fn the_moons_initial_land_is_a_stagnant_lid_with_two_levels() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let land = initial_land(&moon, &lattice, &home_moon_land_words());
        assert_eq!(land.plates.len(), 1);
        assert_eq!(land.sea_z, None);
        assert!(land.kind.iter().all(|&k| k == Kind::None as u8));
        assert!(land.boundary_m.iter().all(|&b| b == u32::MAX));
        assert!(land.plate.iter().all(|&p| p == 0));
        assert!(
            land.uplift.iter().all(|&u| u == 0),
            "a stagnant lid lifts nothing"
        );
        let continental =
            land.crust.iter().filter(|&&c| c >= 128).count() as f64 / land.crust.len() as f64;
        assert!(
            (0.3..0.5).contains(&continental),
            "continental share {continental}"
        );
        let band = (moon.relief_bound_m(0) * f64::from(Z_STEPS_PER_M)) as i32;
        assert!(land.z.iter().all(|&z| z.abs() <= band));
        let area: Vec<u64> = (0..lattice.node_count() as u32)
            .map(|n| lattice.area_m2(n))
            .collect();
        let (_, hist) = hypsometry(&land.z, &area);
        let (low, high, valley) = humps(&hist).expect("two humps");
        assert!(high > low);
        assert!(valley < 0.5, "valley {valley}");
    }

    /// The relief cap is the strength arm on the home planet and the shape arm on its moon.
    #[test]
    fn the_relief_cap_takes_the_lesser_arm() {
        let home = home_planet();
        let (strength, shape) = home.relief_arms_m();
        assert!(strength < shape);
        assert_eq!(relief_cap_m(&home), strength);
        let moon = home_moon();
        let (strength, shape) = moon.relief_arms_m();
        assert!(shape < strength);
        assert_eq!(relief_cap_m(&moon), shape);
    }

    /// The scan over three plates: the third, farthest, is neither the nearest nor the second.
    #[test]
    fn a_third_farther_plate_is_neither_nearest_nor_second() {
        let plate = |site: [f64; 3]| Plate {
            site: [gf(site[0]), gf(site[1]), gf(site[2])],
            drift: [Gf::ZERO; 3],
            affinity: Gf::ZERO,
            age: Gf::ZERO,
            crust_scatter: Gf::ONE,
        };
        let three = [
            plate([1.0, 0.0, 0.0]),
            plate([0.0, 1.0, 0.0]),
            plate([-1.0, 0.0, 0.0]),
        ];
        let b = boundary_of(&three, [gf(0.9), gf(0.1), Gf::ZERO]);
        assert_eq!((b.plate, b.other), (0, Some(1)));
        // A first plate that is the farthest: the scan replaces it as nearest, then as second.
        let b = boundary_of(&three, [gf(-0.7), gf(0.7), Gf::ZERO]);
        assert_eq!((b.plate, b.other), (1, Some(2)));
    }

    /// The area quantile at a share of one walks to the last bin.
    #[test]
    fn the_area_quantile_at_a_full_share_is_the_top() {
        let values: Vec<Gf> = (0..100).map(|k| gf(f64::from(k))).collect();
        let q = area_quantile(&values, &[1; 100], 1.0).to_f64();
        assert!(q > 98.9, "{q}");
    }

    /// ★ THE MULTI-PLATE LAND ON A REAL SMALL BODY: the moon handed two stated plates that push
    /// together along the +X/+Y bisector, and an ocean's worth of water. The boundary's kind is
    /// stated at every node, a trench stands at it, the sea exists and loads the floor under it,
    /// the far plate's character is read, and the hypsometry keeps two humps.
    #[test]
    fn the_moon_with_two_stated_plates_and_a_sea() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let plate = |site: [f64; 3], drift: [f64; 3], affinity: f64| Plate {
            site: [gf(site[0]), gf(site[1]), gf(site[2])],
            drift: [gf(drift[0]), gf(drift[1]), gf(drift[2])],
            affinity: gf(affinity),
            age: gf(0.2),
            crust_scatter: Gf::ONE,
        };
        let plates = vec![
            plate([1.0, 0.0, 0.0], [0.0, 0.5, 0.0], 0.3),
            plate([0.0, 1.0, 0.0], [0.5, 0.0, 0.0], -0.3),
        ];
        // Three million cubic kilometres: about two kilometres of water over the moon's surface,
        // so the sea covers the lowlands and the floors sink under it, and the whole stays in the band.
        // A THIN lithosphere (5 km), so the belt's half-width (two flexural parameters, about
        // 130 km at the moon's gravity) leaves nodes on the far side of the moon outside it.
        let words = LandWords {
            water_km3: 3_000_000,
            elastic_thickness_m: 5_000,
        };
        let land = initial_land_from(&moon, &lattice, &words, plates);
        assert_eq!(land.plates.len(), 2);
        assert!(land.kind.iter().all(|&k| k == Kind::Convergent as u8));
        assert!(land.plate.contains(&0));
        assert!(land.plate.contains(&1));
        assert!(land.boundary_m.iter().all(|&b| b != u32::MAX));
        let sea = land.sea_z.expect("a sea");
        let n = lattice.node_count();
        let under = (0..n).filter(|&k| land.z[k] <= sea).count();
        assert!(under > 0, "nothing under the sea");
        assert!(under < n, "under {under} of {n}: everything under the sea");
        // ★ THE MARGIN (generator 26, ruling B60 item 1): plate 0's continent meets plate 1's ocean
        // at a trench whose inner wall stands on the continent's side, so near the boundary the
        // ground sinks on BOTH sides (the wedge that rises from the coast inland is the belts'
        // law's, not the room's); far from any boundary the uplift is EXACTLY zero (the bump's
        // exact zero outside its width).
        for plate in [0u8, 1] {
            let near: Vec<i64> = (0..n)
                .filter(|&k| (land.boundary_m[k] < 20_000) & (land.plate[k] == plate))
                .map(|k| i64::from(land.uplift[k]))
                .collect();
            assert!(!near.is_empty(), "plate {plate} reaches the boundary");
            let sum: i64 = near.iter().sum();
            assert!(
                sum < 0,
                "plate {plate} at the boundary: a trench, not {sum}"
            );
        }
        assert!(
            (0..n)
                .filter(|&k| land.boundary_m[k] > 250_000)
                .all(|k| land.uplift[k] == 0),
            "the far field lifts"
        );
        assert!(
            (0..n).filter(|&k| land.boundary_m[k] > 250_000).count() > 0,
            "a far node exists"
        );
        let band = (moon.relief_bound_m(0) * f64::from(Z_STEPS_PER_M)) as i32;
        assert!(land.z.iter().all(|&z| z.abs() <= band));
        let area: Vec<u64> = (0..n as u32).map(|k| lattice.area_m2(k)).collect();
        let (_, hist) = hypsometry(&land.z, &area);
        assert!(humps(&hist).is_some());
    }

    /// ★ THE FREEBOARD LAW (W7 step 2), three statements that could each fail: the Airy parts sum
    /// to the Airy height at a factor of one; on the wet two-plate moon the solved sea stands at the
    /// continental platform's median within one quantile bin, with the factor inside the range and
    /// off its ends; and a dry body's factor is one.
    #[test]
    fn the_freeboard_law_puts_the_sea_at_the_platform() {
        for share in [0.0, 0.3, 1.0] {
            let (o, c) = isostatic_parts_m(gf(share), EARTH_RELIEF_CAP_M, gf(1.2));
            let whole = isostatic_height_m(gf(share), EARTH_RELIEF_CAP_M, gf(1.2));
            assert!(((o + c) - whole).to_f64().abs() < 1e-6, "{share}");
        }
        let moon = home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let plate = |site: [f64; 3], drift: [f64; 3], affinity: f64| Plate {
            site: [gf(site[0]), gf(site[1]), gf(site[2])],
            drift: [gf(drift[0]), gf(drift[1]), gf(drift[2])],
            affinity: gf(affinity),
            age: gf(0.2),
            crust_scatter: Gf::ONE,
        };
        let plates = vec![
            plate([1.0, 0.0, 0.0], [0.0, 0.5, 0.0], 0.3),
            plate([0.0, 1.0, 0.0], [0.5, 0.0, 0.0], -0.3),
        ];
        // Twenty million cubic kilometres: at Earth's thickness the sea would overtop the moon's
        // platform, so the law must thicken the crust to hold it under the shelf edge.
        let words = LandWords {
            water_km3: 20_000_000,
            elastic_thickness_m: 5_000,
        };
        let land = initial_land_from(&moon, &lattice, &words, plates.clone());
        let f = land.freeboard_factor.to_f64();
        assert!(
            f > FREEBOARD_FACTOR_RANGE.0,
            "the factor {f} sits inside the range"
        );
        assert!(
            f < FREEBOARD_FACTOR_RANGE.1,
            "the factor {f} sits inside the range"
        );
        let sea = land.sea_z.expect("a sea");
        // The continental crust's share OVER the sea, on the LOADED land (a node over the sea is
        // not loaded, so its dry and loaded heights agree): Earth's 29 of 40, within the
        // quantile's own bin.
        let n = lattice.node_count();
        let (mut over, mut total) = (0u64, 0u64);
        for k in (0..n).filter(|&k| land.crust[k] >= 128) {
            let a = lattice.area_m2(k as u32);
            total += a;
            over += u64::from(land.z[k] > sea) * a;
        }
        let share = over as f64 / total as f64;
        let want = EARTH_LAND_SHARE / CONTINENTAL_SHARE;
        assert!(
            (share - want).abs() < 0.02,
            "the crust over the sea {share} against Earth's {want}"
        );
        let dry = initial_land_from(
            &moon,
            &lattice,
            &LandWords {
                water_km3: 0,
                elastic_thickness_m: 5_000,
            },
            plates,
        );
        assert_eq!(dry.freeboard_factor, Gf::ONE);
        // A stated field of ocean floor alone (no node of continental crust) holds no platform for
        // its sea to stand at: the law has no continent to thicken, and the factor is ONE. (The
        // crust share follows the area quantile, so a body's own plates always lay a continent.)
        let flat = vec![Gf::ZERO; n];
        let area: Vec<u64> = (0..n as u32).map(|k| lattice.area_m2(k)).collect();
        assert_eq!(
            freeboard_factor(&lattice, &flat, &flat, &vec![0u8; n], &area, 1_000, &flat),
            Gf::ONE
        );
    }

    /// ★ FAILING FIRST (slice 8d step 2): THE PROVINCE RULE'S SIX ARMS, each on the words the land
    /// itself holds. RED before this step: `province_of` did not exist.
    #[test]
    fn the_province_rule_reads_the_crust_the_uplift_and_the_sea() {
        let sea = Some(0);
        let (push, pull) = (Kind::Convergent, Kind::Divergent);
        // Continental crust (the solve's own `>= 128` test), rising: a collision belt.
        assert_eq!(
            province_of(200, 12, -50, sea, push, true),
            Province::FoldedBelt
        );
        // Continental, sinking where the plates pull apart: a rift valley the lava fills.
        assert_eq!(
            province_of(200, -12, -50, sea, pull, true),
            Province::RiftBasalt
        );
        // ★ Continental, sinking where the plates push together (ruling B63 item 2): a forearc
        // under the trench's inner wall, which fills with sediment.
        assert_eq!(
            province_of(200, -12, -50, sea, push, false),
            Province::DeepSediment
        );
        // Continental, still, above its sea: the old shield.
        assert_eq!(
            province_of(128, 0, 1, sea, push, true),
            Province::CrystallineBasement
        );
        // Continental, still, under its sea: the drowned platform.
        assert_eq!(
            province_of(128, 0, -1, sea, push, true),
            Province::FlatShelf
        );
        // Oceanic, rising: a ridge, or an island arc where two oceans meet.
        assert_eq!(
            province_of(127, 9, -900, sea, pull, true),
            Province::RiftBasalt
        );
        assert_eq!(
            province_of(127, 9, -900, sea, push, false),
            Province::RiftBasalt
        );
        // ★ Oceanic, rising where it dives under a continent (ruling B63 item 2): the diving
        // plate's own floor, no arc.
        assert_eq!(
            province_of(127, 9, -900, sea, push, true),
            Province::DeepSediment
        );
        // Oceanic, sinking: a trench, which fills with its own sediment.
        assert_eq!(
            province_of(0, -9, -900, sea, push, true),
            Province::DeepSediment
        );
        // Oceanic, still: the abyssal plain.
        assert_eq!(
            province_of(0, 0, -900, sea, Kind::None, false),
            Province::DeepSediment
        );
        // A DRY body has no sea, so no node is drowned and no shelf exists.
        assert_eq!(
            province_of(200, 0, -900, None, Kind::None, false),
            Province::CrystallineBasement
        );
    }

    /// ★ FAILING FIRST (slice 8d step 2): THE ROCK MAP OVER A REAL SMALL BODY OF THE WORLD — the
    /// moon under two stated convergent plates and a sea, the same fixture the belts stand on.
    ///
    /// Four statements. (1) Every node carries a province a reader can name. (2) Every node the
    /// margin moves is deep sediment (ruling B63 item 2): the room raises no continental crust on
    /// this body, so no node is a belt. (3) The dry shield appears. (4) Which
    /// codes are ABSENT is stated, never shrugged at: this body's two plates only push together, so
    /// nothing pulls apart and the diving plate makes no arc — the rift basalt is absent for that
    /// reason; the folded belt is the belts' law's, which needs a ridge to time it; and the first
    /// land holds no shelf (generator 26: no pocket, the continent clear of its sea).
    #[test]
    fn the_moons_rock_map_names_a_province_for_every_node() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let plate = |site: [f64; 3], drift: [f64; 3], affinity: f64| Plate {
            site: [gf(site[0]), gf(site[1]), gf(site[2])],
            drift: [gf(drift[0]), gf(drift[1]), gf(drift[2])],
            affinity: gf(affinity),
            age: gf(0.2),
            crust_scatter: Gf::ONE,
        };
        let plates = vec![
            plate([1.0, 0.0, 0.0], [0.0, 0.5, 0.0], 0.3),
            plate([0.0, 1.0, 0.0], [0.5, 0.0, 0.0], -0.3),
        ];
        let words = LandWords {
            water_km3: 3_000_000,
            elastic_thickness_m: 5_000,
        };
        let land = initial_land_from(&moon, &lattice, &words, plates);
        let n = lattice.node_count();
        let sea = land.sea_z.expect("a sea");
        assert_eq!(land.province.len(), n);
        let mut counts = std::collections::BTreeMap::new();
        let (mut diving, mut risen_continent) = (0usize, 0usize);
        for k in 0..n {
            let p = Province::from_code(land.province[k]).expect("a named province");
            *counts.entry(p).or_insert(0usize) += 1;
            // ★ (2) WHERE THE PLATES PUSH TOGETHER, NO RIFT AND NO ARC ON THIS BODY (ruling B63
            // item 2): every node the margin moves that is no belt — the trench's floor and walls, a
            // forearc the inner wall sinks, the diving plate's cells the crust share lifts — is
            // sediment. A belt node (continental crust the room raised) would be the folded belt
            // (the province rule's own test); this body holds none, counted below.
            let belt = (land.uplift[k] > 0) & (land.crust[k] >= 128);
            risen_continent += usize::from(belt);
            if (land.uplift[k] != 0) & !belt {
                assert_eq!(p, Province::DeepSediment, "node {k}: moved by the margin");
                diving += usize::from((land.uplift[k] > 0) & (land.plate[k] == 1));
            }
        }
        let held = |p: Province| counts.get(&p).copied().unwrap_or_default();
        // The diving plate's own cells rise where the crust share gives them a part of the belt's
        // profile (MEASURED: 436 nodes, all sediment).
        assert!(diving > 0, "the diving plate rises nowhere: {counts:?}");
        // THE BELT IS ABSENT, AND THAT IS A MEASUREMENT (generator 26): the Andean margin's room holds
        // the trench, and its belt is the belts' law's wedge, which needs a ridge to time it — this
        // body's two plates only push together, so no wedge rises (the belts' law's own domain;
        // `belts::tests` raise one on four plates). MEASURED: the room raises no node of
        // continental crust here — the margin's room is the trench.
        assert_eq!(
            risen_continent, 0,
            "the room raised continental crust: {counts:?}"
        );
        assert_eq!(held(Province::FoldedBelt), 0, "{counts:?}");
        assert!(
            held(Province::CrystallineBasement) > 0,
            "no shield: {counts:?}"
        );
        assert!(
            held(Province::DeepSediment) > 0,
            "no ocean floor: {counts:?}"
        );
        // (4) THREE CODES ARE ABSENT (the belt above, the rift and the shelf below), AND EACH
        // ABSENCE IS A MEASUREMENT, not a shrug.
        //
        // THE RIFT: both of this body's boundaries push together, so nothing pulls apart, and the
        // diving plate makes no arc: what the margin sinks or the crust share lifts there is deep
        // sediment, never rift basalt (ruling B63 item 2); a divergent plate pair is what draws a
        // rift, and the home planet's fourteen plates hold several.
        assert_eq!(held(Province::RiftBasalt), 0, "a rift with no rifting");
        assert!(land.kind.iter().all(|&k| k == Kind::Convergent as u8));
        // THE SHELF: three million cubic kilometres of water stands under every node of
        // continental crust IN THE OCEAN on this moon — the isostasy floats the continent clear of
        // its own sea — so the ocean drowns no continent. MEASURED here, node by node.
        let drowned = (0..n)
            .filter(|&k| (land.crust[k] >= 128) & (land.z[k] < sea) & land.ocean[k])
            .count();
        assert_eq!(drowned, 0, "continental crust under this moon's ocean");
        // ★ THE SHELF IS ABSENT ON THIS FIRST LAND, AND THAT IS A MEASUREMENT (generator 26). Before
        // it, the only shelf stood in a dry pocket under the connected sea's level (one pocket of
        // 2 955 nodes, ruling B7 item 1); the margins now thin the continent toward its trench and
        // the older landscape lies over it, and the first land holds no pocket at all — so no
        // continental floor stands under the level anywhere, in the ocean or out of it. The shelf's
        // own arm is the province rule's test above.
        assert_eq!(
            (land.pockets, land.pocket_nodes),
            (0, 0),
            "a pocket on the first land: {counts:?}"
        );
        assert_eq!(held(Province::FlatShelf), 0, "{counts:?}");
    }

    /// ★ THE DIP'S TANGENT is the tangent of the published dip: the fenced series of the sine and the cosine of 23°
    /// agree with [`ANDEAN_DIP_TAN`] to the word's own precision.
    #[test]
    fn the_dip_tangent_is_the_dips() {
        let x = gf(ANDEAN_DIP_DEG) * gf(std::f64::consts::PI) / Gf::from_i64(180);
        let (mut sin, mut cos) = (x, Gf::ONE);
        let (mut term_s, mut term_c) = (x, Gf::ONE);
        for k in 1..12i64 {
            term_s = -term_s * x * x / Gf::from_i64((2 * k) * (2 * k + 1));
            term_c = -term_c * x * x / Gf::from_i64((2 * k - 1) * (2 * k));
            sin += term_s;
            cos += term_c;
        }
        let tan = (sin / cos).to_f64();
        assert!((tan - ANDEAN_DIP_TAN).abs() < 1e-15, "{tan}");
    }

    /// ★ THE MARGINS' WIDTHS BY KIND (ruling B59 item 2), each against its law written out again: an Andean coast where
    /// the plate interface at 23° meets the Moho of Earth's own crusts (about 72 km), a rifted margin of half
    /// `(1 + ε)·3.5` strong layers (about 40 km over Earth's lithosphere), a transform margin at Earth's measured 50 km,
    /// and no margin without a boundary.
    #[test]
    fn the_margins_widths_by_kind() {
        let own = Plate {
            site: [Gf::ONE, Gf::ZERO, Gf::ZERO],
            drift: [Gf::ZERO; 3],
            affinity: gf(0.3),
            age: gf(0.2),
            crust_scatter: Gf::ONE,
        };
        let te = EARTH_ELASTIC_THICKNESS_M as u32;
        let andean =
            margin_width_m(Kind::Convergent, &own, EARTH_RELIEF_CAP_M, Gf::ONE, te).to_f64();
        let (t_c, t_o) = (EARTH_CONTINENTAL_CRUST_M, EARTH_OCEANIC_CRUST_M);
        let d_t = t_c * (1.0 - CONTINENTAL_CRUST_DENSITY_KGM3 / MANTLE_DENSITY_KGM3)
            - t_o * (1.0 - OCEANIC_CRUST_DENSITY_KGM3 / MANTLE_DENSITY_KGM3);
        let want = (t_c - d_t) / ANDEAN_DIP_TAN;
        assert!((andean - want).abs() < 1e-6, "{andean} against {want}");
        assert!((71_000.0..73_000.0).contains(&andean), "{andean}");
        // A thicker crust (the freeboard factor) moves the coast inland.
        assert!(
            margin_width_m(Kind::Convergent, &own, EARTH_RELIEF_CAP_M, gf(1.4), te).to_f64()
                > andean
        );
        let rifted =
            margin_width_m(Kind::Divergent, &own, EARTH_RELIEF_CAP_M, Gf::ONE, te).to_f64();
        let want = 0.5
            * (1.0 + NECKING_STRETCH)
            * NECKING_WAVELENGTH_RATIO
            * crate::belts::INDIA_STORED_CRUST_M;
        assert!((rifted - want).abs() < 1e-6, "{rifted} against {want}");
        assert_eq!(
            margin_width_m(Kind::Transform, &own, EARTH_RELIEF_CAP_M, Gf::ONE, te).to_f64(),
            TRANSFORM_NECKING_M
        );
        assert_eq!(
            margin_width_m(Kind::None, &own, EARTH_RELIEF_CAP_M, Gf::ONE, te),
            Gf::ZERO
        );
    }

    /// ★ THE OLDER LANDSCAPE (ruling B60) on a real body of the world, the home moon: its area-weighted mean is zero, its
    /// height difference at 110 km along the face rows and columns is Earth's stable continents' 131 m, it holds no two
    /// equal neighbours, and the same seed draws the same field.
    #[test]
    fn the_older_landscape_holds_earths_relief_at_its_lag() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let field = older_landscape_m(&moon, &lattice);
        let n = lattice.node_count();
        assert_eq!(field.len(), n);
        let (mut sum, mut total) = (0.0f64, 0.0f64);
        for (k, f) in field.iter().enumerate() {
            let a = lattice.area_m2(k as u32) as f64;
            sum += f.to_f64() * a;
            total += a;
        }
        assert!((sum / total).abs() < 1e-6, "the mean {}", sum / total);
        let lag = SPECTRUM_LAG_M / lattice.node_m();
        let (l0, frac) = (lag.floor() as i32, lag - lag.floor());
        let at = rms_difference(&lattice, &field, l0).to_f64() * (1.0 - frac)
            + rms_difference(&lattice, &field, l0 + 1).to_f64() * frac;
        assert!(
            (at - SPECTRUM_RMS_M).abs() < 1e-6,
            "{at} m at {SPECTRUM_LAG_M} m"
        );
        let flat = (0..n as u32)
            .filter(|&k| {
                lattice
                    .neighbours(k)
                    .iter()
                    .all(|&m| m == NO_NODE || field[m as usize] == field[k as usize])
            })
            .count();
        assert_eq!(flat, 0, "an exact flat in the older landscape");
        assert_eq!(older_landscape_m(&moon, &lattice), field);
    }

    /// ★ GENERATOR 26's FIRST LAND on the moon's two stated plates (rulings B59–B61): (1) the margin closes the plates'
    /// step — the crust shares on the two sides of the boundary agree within a few steps at the boundary while the
    /// plates' interiors differ by most of the scale; (2) no node of continental crust stands on an exact flat (the root
    /// ruling B59 measured: 87 % of the old first land did).
    #[test]
    fn the_margin_closes_the_plates_step_and_no_continent_is_flat() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let plate = |site: [f64; 3], drift: [f64; 3], affinity: f64| Plate {
            site: [gf(site[0]), gf(site[1]), gf(site[2])],
            drift: [gf(drift[0]), gf(drift[1]), gf(drift[2])],
            affinity: gf(affinity),
            age: gf(0.2),
            crust_scatter: Gf::ONE,
        };
        let plates = vec![
            plate([1.0, 0.0, 0.0], [0.0, 0.5, 0.0], 0.3),
            plate([0.0, 1.0, 0.0], [0.5, 0.0, 0.0], -0.3),
        ];
        let words = LandWords {
            water_km3: 3_000_000,
            elastic_thickness_m: 5_000,
        };
        let land = initial_land_from(&moon, &lattice, &words, plates);
        let n = lattice.node_count();
        // (1) The mean crust share of each plate's nodes within one node of the boundary, and on the farthest third of
        // the plate from it (the moon is 355 km in radius, so its farthest node stands about 350 km from the boundary).
        let node_m = lattice.node_m() as u32;
        let mean_crust = |plate: u8, lo: u32, hi: u32| -> f64 {
            let picked: Vec<f64> = (0..n)
                .filter(|&k| {
                    (land.plate[k] == plate)
                        & (land.boundary_m[k] >= lo)
                        & (land.boundary_m[k] < hi)
                })
                .map(|k| f64::from(land.crust[k]))
                .collect();
            assert!(!picked.is_empty(), "plate {plate} has nodes at {lo}–{hi} m");
            picked.iter().sum::<f64>() / picked.len() as f64
        };
        let farthest = (0..n).map(|k| land.boundary_m[k]).max().expect("a node");
        let deep = farthest / 3 * 2;
        let (near0, near1) = (mean_crust(0, 0, node_m), mean_crust(1, 0, node_m));
        let (far0, far1) = (mean_crust(0, deep, u32::MAX), mean_crust(1, deep, u32::MAX));
        assert!(
            (near0 - near1).abs() < 40.0,
            "at the boundary {near0} against {near1}"
        );
        assert!(far0 - far1 > 100.0, "deep inside {far0} against {far1}");
        // (2) No continental node stands on an exact flat.
        let flat = (0..n as u32)
            .filter(|&k| land.crust[k as usize] >= 128)
            .filter(|&k| {
                lattice
                    .neighbours(k)
                    .iter()
                    .filter(|&&m| m != NO_NODE)
                    .all(|&m| land.z[m as usize] == land.z[k as usize])
            })
            .count();
        assert_eq!(flat, 0, "continental nodes on an exact flat");
    }

    /// The home planet's land words are the pinned ones.
    #[test]
    fn the_home_land_words() {
        assert_eq!(home_land_words().water_km3, 2_735_928_089);
        assert_eq!(home_moon_land_words().elastic_thickness_m, 1_094_578);
    }
}
