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

use crate::body::{BodyDefinition, draw_unit, frequency_of, salt};
use crate::gf::Gf;
use crate::macro_lattice::MacroLattice;
use crate::solve::Z_STEPS_PER_M;

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
/// named scatter around a computed value is lawful): a craton is thicker than a young platform.
pub const CRUST_SCATTER: (f64, f64) = (0.7, 1.4);
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
    /// The heights in sixteenths from the ladder radius, the water load applied under the sea.
    pub z: Vec<i32>,
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
    /// The plates drawn.
    pub plates: Vec<Plate>,
    /// The crust field's threshold, the area quantile.
    pub threshold: Gf,
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
            let (lo, hi) = CRUST_SCATTER;
            let crust_scatter = Gf::from_f64(lo) + draw_unit(&mut rng) * Gf::from_f64(hi - lo);
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

/// A node's direction as fenced floats.
fn dir_of(dir: [Gi; 3]) -> [Gf; 3] {
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

/// ★ THE SEA OVER THE NODE FIELD, WITH THE LOAD (02 §7.1, §4.2): the level `L`, in sixteenths, at
/// which the water under it equals the inventory, where a node under the level sinks by the
/// water's weight — and the deeper column weighs more, so the balance is the fixed point
/// `sink = (L − h + sink)·ρ_w/ρ_m`, whose closed form is `sink = (L − h)·ρ_w/(ρ_m − ρ_w)` and the
/// water column `(L − h)/(1 − ρ_w/ρ_m)`, linear in `L` (Earth's 4 455 m dry step becomes the
/// 6 476 m loaded one). An integer bisection on the level between the lowest node and the level
/// that would drown the whole field; `None` for no water.
#[must_use]
pub fn sea_level(z_dry: &[i32], area: &[u64], water_km3: u64) -> Option<i32> {
    if water_km3 == 0 {
        return None;
    }
    let load = Gf::ONE / (Gf::ONE - Gf::from_f64(WATER_DENSITY_KGM3 / MANTLE_DENSITY_KGM3));
    // The target in area·sixteenths: the inventory in m³ times the steps a metre, over the load.
    let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let target_m3_steps = Gf::from_i64(water_km3 as i64) * Gf::from_f64(1.0e9) * steps / load;
    let target = target_m3_steps.to_f64() as u128;
    let total_area: u128 = area.iter().map(|&a| u128::from(a)).sum();
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for &z in z_dry {
        lo = lo.min(z);
        hi = hi.max(z);
    }
    let column = (target / total_area.max(1)) as i64 + 1;
    let mut hi = i64::from(hi) + column;
    let mut lo = i64::from(lo);
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
    let plates = draw_plates(
        body.seed(),
        plate_count(body.ladder().radius_m(), words.elastic_thickness_m),
    );
    initial_land_from(body, lattice, words, plates)
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
    let n = lattice.node_count();
    let gravity = body.facts().gravity_mm_s2;
    let cap_m = relief_cap_m(body);
    let octaves = affinity_octaves(body);
    let radius = Gf::from_f64(body.ladder().radius_m());
    let width =
        Gf::from_f64(flexural_parameter_m(words.elastic_thickness_m, gravity) * BELT_HALF_WIDTHS);
    let relief = Gf::from_f64(body.relief_m());
    // 1. The plates and the affinity at every node.
    let mut boundaries = Vec::with_capacity(n);
    let mut affinity = Vec::with_capacity(n);
    let mut area = Vec::with_capacity(n);
    for node in 0..n as u32 {
        let gi = lattice.direction(node);
        let d = dir_of(gi);
        let b = boundary_of(&plates, d);
        affinity.push(affinity_noise(&octaves, gi) + plates[usize::from(b.plate)].affinity);
        boundaries.push(b);
        area.push(lattice.area_m2(node));
    }
    // 2. The crust threshold: the continental share of the AREA.
    let threshold = area_quantile(&affinity, &area, 1.0 - CONTINENTAL_SHARE);
    // 3. Isostasy, dry, about the area-weighted mean — and the room the belts have under the
    //    relief: THE ENVELOPE (03 §4.13) holds by construction, `|z| ≤ relief`, because the belts'
    //    amplitude is the relief less the tallest platform.
    let mut dry = Vec::with_capacity(n);
    let mut crust = Vec::with_capacity(n);
    let mut sum = Gf::ZERO;
    let total_area = Gf::from_i64(area.iter().sum::<u64>() as i64);
    for k in 0..n {
        let b = boundaries[k];
        let share = crust_share(affinity[k], threshold);
        let own = &plates[usize::from(b.plate)];
        let iso = isostatic_height_m(share, cap_m, own.crust_scatter);
        sum += iso * Gf::from_i64(area[k] as i64);
        dry.push(iso);
        crust.push((share * Gf::from_f64(255.0) + Gf::HALF).to_i64_floor() as u8);
    }
    let mean = sum / total_area;
    let mut tallest = Gf::ZERO;
    for h in &mut dry {
        *h -= mean;
        tallest = tallest.greater(h.abs());
    }
    let room = (relief - tallest).greater(Gf::ZERO);
    // 4. Orogeny: the belts under the room.
    for k in 0..n {
        let b = boundaries[k];
        let own = &plates[usize::from(b.plate)];
        let (other_continental, other_lower) = match b.other {
            Some(o) => (plates[usize::from(o)].affinity >= Gf::ZERO, o < b.plate),
            None => (false, false),
        };
        let amplitude = room
            * (b.convergence.abs() / Gf::TWO).clamp(Gf::ZERO, Gf::ONE)
            * (Gf::ONE - own.age * Gf::HALF);
        dry[k] += uplift_m(
            b.kind,
            crust[k] >= 128,
            other_continental,
            !other_lower,
            radius * b.sine,
            amplitude,
            width,
        );
    }
    let mean = Gf::ZERO;
    let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let z_dry: Vec<i32> = dry
        .iter()
        .map(|&h| ((h - mean) * steps).to_i64_floor() as i32)
        .collect();
    // 5. The sea, with the load.
    let sea_z = sea_level(&z_dry, &area, words.water_km3);
    let z = match sea_z {
        Some(level) => z_dry.iter().map(|&h| loaded_height(h, level)).collect(),
        None => z_dry,
    };
    Land {
        z,
        sea_z,
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
    }
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
            assert!((0.7..1.4).contains(&p.crust_scatter.to_f64()));
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
        let level = sea_level(&z, &area, 500_000).expect("a sea");
        let level_m = f64::from(level) / f64::from(Z_STEPS_PER_M);
        assert!((level_m - 343.9).abs() < 1.0, "{level_m}");
        let column_m = f64::from(level - loaded_height(low, level)) / f64::from(Z_STEPS_PER_M);
        assert!((column_m - 500.0).abs() < 1.0, "{column_m}");
        assert_eq!(sea_level(&z, &area, 0), None);
        let flooded = sea_level(&z, &area, 10_000_000).expect("a sea");
        assert!(flooded > high);
        assert!(loaded_height(low, level) < low);
        assert_eq!(loaded_height(high, level), high);
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
    /// stated at every node, a belt or a trench stands near it, the sea exists and loads the floor
    /// under it, the far plate's character is read, and the hypsometry keeps two humps.
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
        let words = LandWords {
            water_km3: 3_000_000,
            elastic_thickness_m: 30_000,
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
        // The belts: near the boundary on the continental side the land stands over the far field.
        let near_max = (0..n)
            .filter(|&k| land.boundary_m[k] < 20_000 && land.crust[k] >= 128)
            .map(|k| land.z[k])
            .max()
            .expect("a near node");
        let far_max = (0..n)
            .filter(|&k| land.boundary_m[k] > 300_000)
            .map(|k| land.z[k])
            .max()
            .expect("a far node");
        assert!(
            near_max > far_max,
            "belt {near_max} over the far field {far_max}"
        );
        let band = (moon.relief_bound_m(0) * f64::from(Z_STEPS_PER_M)) as i32;
        assert!(land.z.iter().all(|&z| z.abs() <= band));
        let area: Vec<u64> = (0..n as u32).map(|k| lattice.area_m2(k)).collect();
        let (_, hist) = hypsometry(&land.z, &area);
        assert!(humps(&hist).is_some());
    }

    /// The home planet's land words are the pinned ones.
    #[test]
    fn the_home_land_words() {
        assert_eq!(home_land_words().water_km3, 2_735_928_089);
        assert_eq!(home_moon_land_words().elastic_thickness_m, 1_094_578);
    }
}
