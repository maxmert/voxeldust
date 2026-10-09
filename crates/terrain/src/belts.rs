//! ★ THE BELTS' LAW (phase 2 of the ground step; rulings B37, B40, B41–B42, B49–B51; the belts' design
//! `docs/investigation/2026-10-02/belts_and_time_design.md` §3, §12, §17).
//!
//! A belt is the rock a convergent margin takes in, stacked into a CRITICAL WEDGE (Roe, Stolar & Willett 2008,
//! *Response of a steady-state critical wedge orogen to changes in climate and tectonic forcing*, Eq. 35): its
//! surface cross-section is `(1 − ρ_c/ρ_m)` of the rock's, a triangle of two tapers whose crest rises to the force
//! balance's height and then widens into a plateau behind it. Every number it reads is the body's own or a stated
//! calibration:
//!
//! - **the influx**: the convergence (the plates' own drift shares times [`PLATE_SPEED_PER_SHARE_MM_YR`], ruling B40
//!   item 2) times the crust taken in — at a collision the incoming continent's crust ABOVE THE DECOUPLING LEVEL
//!   (P2, Replumaz et al. 2010: [`INDIA_STORED_CRUST_M`], carried by the elastic thickness), at an Andean margin
//!   Norabuena et al. 1998's share of the upper plate's own crust ([`ANDEAN_SHARE`], the registered interim);
//! - **the window**: a seeded share of the plates' turnover (Parsons: twice the oceanic area over the ridges'
//!   opening), and at a collision the influx falls as `e^(−t/τ)` (S2, Clark 2012: `τ = 1/ε̇` across Tibet);
//! - **the tapers**: the front flank at the critical taper of Davis, Suppe & Dahlen 1983 with Taiwan's frictions
//!   ([`PRO_TAPER_TAN`]), the back flank at Taiwan's measured 6° ([`RETRO_TAPER_TAN`]); the incoming plate (the one
//!   whose drift carries it faster toward the other) takes the front flank;
//! - **the ceiling**: the force balance `ΔF = ρ_c·g·h·[L + h·ρ_m/(2Δρ)]` with the force Tibet's mean height holds
//!   ([`TIBET_MEAN_M`]), carried by the elastic thickness;
//! - **one level a pair** (ruling B51 item 1, Clark & Royden 2000): the lower crust flows until a plateau is level, so
//!   each node's ceiling is its pair's level less its own platform.
//!
//! **Example.** Two continents of the home planet close at 40 mm a year. The faster one dives: its crust down to
//! 20 km is scraped into a wedge whose front rises 3.8 m a kilometre inland from the suture and whose back falls at
//! 6°. The wedge grows for its window of about 70 million years; when its crest reaches the 3 km the plates' force
//! holds, the rest of the rock widens a level plateau behind it.

// ★ THE BELTS' LAW MAY DIVIDE: it runs once per body on the CPU, never in a kernel, and its one integer quotient is
// the window's exact split of a whole number of sixteenths.
#![allow(
    clippy::integer_division,
    reason = "the belts' law runs on the CPU once per body, never in a kernel"
)]

use std::collections::BTreeMap;

use crate::body::BodyDefinition;
use crate::gf::Gf;
use crate::land::{
    CONTINENTAL_CRUST_DENSITY_KGM3, EARTH_CONTINENTAL_CRUST_M, EARTH_ELASTIC_THICKNESS_M,
    EARTH_RELIEF_CAP_M, Kind, Land, MANTLE_DENSITY_KGM3, boundary_of, dir_of, margin_width_m,
    relief_cap_m,
};
use crate::macro_lattice::MacroLattice;
use crate::solve::{SolveWords, Z_STEPS_PER_M};

/// ★ THE PLATE SPEED A WHOLE DRIFT SHARE STANDS FOR, mm/yr (ruling B40 item 2): a plate's drift share is drawn
/// uniform on `[0, 1)`, so its mean is a half; Earth's mean plate speed is
/// [`crate::craters::EARTH_MEAN_PLATE_SPEED_MM_YR`] (checked against Earth's 53.4 mm/yr of ridge opening), so a whole
/// share is twice it. The belts and the craters' plate clock read this one speed.
pub const PLATE_SPEED_PER_SHARE_MM_YR: f64 = 2.0 * crate::craters::EARTH_MEAN_PLATE_SPEED_MM_YR;

/// ★ THE ANDEAN SHARE: the part of the convergence that shortens the upper plate of an ocean under a continent —
/// Earth's ratio as a STATED CALIBRATION (Norabuena et al. 1998: 10–15 mm/yr of inland shortening against 60–80 mm/yr
/// of convergence; the middle, 12.5/70). A REGISTERED INTERIM (ruling B42 item 3): the Andean margins' law replaces it
/// (the belts' design §18).
pub const ANDEAN_SHARE: f64 = 12.5 / 70.0;

/// The window draw's salt: the ASCII of "WINDOW".
pub const WINDOW_SALT: u64 = 0x5749_4e44_4f57;

/// ★ P2's calibration (Replumaz et al. 2010): the thickness of India's crust stored in the collision (in the
/// thickened crust or moved out sideways), metres, of the craton's 38 km — the decoupling level.
pub const INDIA_STORED_CRUST_M: f64 = 19_000.0;

/// ★ S2's calibration (Clark 2012): the constant contractional strain rate across Tibet, per second.
pub const TIBET_STRAIN_RATE_PER_S: f64 = 7.03e-16;

/// Seconds in a Julian year (365.25 days): S2's strain rate against the solve's years.
pub const SECONDS_PER_YEAR: f64 = 31_557_600.0;

/// ★ THE FORCE'S CALIBRATION BODY (ruling B40 item 1): the Tibetan Plateau's measured mean height, metres (Fielding et
/// al. 1994, Geology 22: the arid central and northern interior), over Earth's continental crust of 35 km, under
/// Earth's gravity; the force it implies is carried to a body by the ratio of its elastic thickness to Earth's.
pub const TIBET_MEAN_M: f64 = 5_023.0;

/// Earth's surface gravity for the force's calibration, m/s².
pub const EARTH_GRAVITY_M_S2: f64 = 9.81;

/// ★ THE FRONT (PRO) FLANK's slope as its tangent: the subaerial Coulomb wedge's critical taper (Davis, Suppe &
/// Dahlen 1983; in the form of Mouthereau et al. 2006, Eq. 1) `α + β = [β + μ_b(1 − λ_b)] / [1 + (1 − λ)·2 sin φ/(1 −
/// sin φ)]`, `φ = atan μ`, at Taiwan's numbers `μ = 1.03`, `μ_b = 0.85`, `λ = λ_b = 0.7`, `β = 6°`: `α` = 0.037 815 rad
/// (2.17°; Taiwan's measured pro flank is about 2°, Roe et al. 2008, p. 823). Stated as the tangent of that angle,
/// because the fence offers no tangent; the bench's test `the_critical_taper_at_taiwan` checks both numbers.
pub const PRO_TAPER_TAN: f64 = 0.037_833_035_141_091_13;

/// ★ THE BACK (RETRO) FLANK's slope as its tangent: Taiwan's measured retro flank, 6° (Roe et al. 2008, p. 823).
pub const RETRO_TAPER_TAN: f64 = 0.105_104_235_265_676_46;

/// A node's belt kind: no belt.
pub const BELT_NONE: u8 = 0;
/// A node's belt kind: a collision of two continents.
pub const BELT_COLLISION: u8 = 1;
/// A node's belt kind: an ocean under a continent (an Andean margin).
pub const BELT_ANDEAN: u8 = 2;

/// A wedge node on the incoming plate's side of a collision (the front flank).
pub const SIDE_PRO: u8 = 0;
/// A wedge node on the other side of a collision (the plateau and the back flank).
pub const SIDE_RETRO: u8 = 1;
/// A wedge node on the continent of an Andean margin (the back flank from the coast, the plateau, the front ramp
/// inland).
pub const SIDE_ANDEAN: u8 = 2;

/// ★ A WEDGE NODE: what the critical wedge reads at one belt node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WedgeNode {
    pub node: u32,
    /// [`SIDE_PRO`], [`SIDE_RETRO`] or [`SIDE_ANDEAN`].
    pub side: u8,
    /// The node's distance to the boundary, m.
    pub dist_m: Gf,
    /// The rock the margin takes in when its window opens, m² a year a metre of margin.
    pub f0_m2_yr: Gf,
    /// The window's start, years from the solve's start.
    pub start_yr: u64,
    /// The crest's ceiling over the node's platform, m (the force balance's height, or the pair's level less the
    /// platform).
    pub ceil_m: Gf,
    /// S2's e-folding time, years, at a slowing collision.
    pub tau_yr: Option<Gf>,
}

/// What the belts' law found on a body (the census's readings).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeltReport {
    /// The wedge nodes that rise: all, at collisions, at Andean margins, and on a plateau at the end.
    pub nodes: usize,
    pub collision_nodes: usize,
    pub andean_nodes: usize,
    pub plateau_nodes: usize,
    /// The plates' turnover `t_m`, years.
    pub t_m_yr: u64,
    /// The rock the wedges raise by the end (each node's surface rise times its area), m³: at collisions and at Andean
    /// margins.
    pub volume_m3: [u64; 2],
}

/// ★ THE BELTS of one body: each node's surface rise by the end (sixteenths; zero where no belt), its window's start
/// in years from the solve's start, its kind, whether it stands on a plateau at the end, and the wedge nodes that
/// rise.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Belts {
    pub total: Vec<i64>,
    pub start_yr: Vec<u64>,
    pub kind: Vec<u8>,
    pub plateau: Vec<bool>,
    pub wedge: Vec<WedgeNode>,
    pub report: BeltReport,
}

/// The rock a wedge node's margin has taken in by `t_yr` (years from the solve's start), m² a metre of margin: the
/// influx over the window so far, falling as `e^(−t/τ)` at a slowing collision.
#[must_use]
pub fn wedge_area_m2(w: &WedgeNode, t_yr: u64) -> Gf {
    if t_yr <= w.start_yr {
        return Gf::ZERO;
    }
    let dt = Gf::from_i64((t_yr - w.start_yr) as i64);
    match w.tau_yr {
        Some(tau) => w.f0_m2_yr * tau * (Gf::ONE - (-(dt / tau)).exp()),
        None => w.f0_m2_yr * dt,
    }
}

/// ★ THE WEDGE'S SURFACE over the platform at a wedge node, m, for the rock taken in `a_m2` (Roe et al. 2008, Eq. 35:
/// the surface's cross-section is `(1 − ρ_c/ρ_m)` of the rock's, a triangle of the two tapers; past the ceiling a
/// trapezoid whose plateau widens behind the crest). Returns the height and whether the node stands on the plateau. A
/// ceiling of nothing (a platform over its pair's level) holds a plateau of no height.
#[must_use]
pub fn wedge_height_m(w: &WedgeNode, a_m2: Gf) -> (Gf, bool) {
    let (tp, tr) = (Gf::from_f64(PRO_TAPER_TAN), Gf::from_f64(RETRO_TAPER_TAN));
    let cot_sum = Gf::ONE / tp + Gf::ONE / tr;
    let rm = Gf::from_f64(MANTLE_DENSITY_KGM3);
    let surface = a_m2 * (rm - Gf::from_f64(CONTINENTAL_CRUST_DENSITY_KGM3)) / rm;
    let h_tri = (Gf::TWO * surface / cot_sum).sqrt();
    let (h, plateau) = if h_tri <= w.ceil_m {
        (h_tri, Gf::ZERO)
    } else if w.ceil_m <= Gf::ZERO {
        (Gf::ZERO, Gf::from_f64(f64::INFINITY))
    } else {
        (
            w.ceil_m,
            (surface - w.ceil_m * w.ceil_m * cot_sum / Gf::TWO) / w.ceil_m,
        )
    };
    let d = w.dist_m;
    let on_plateau = plateau > Gf::ZERO;
    match w.side {
        SIDE_PRO => ((h - d * tp).greater(Gf::ZERO), false),
        SIDE_RETRO => {
            if d <= plateau {
                (h, on_plateau)
            } else {
                ((h - (d - plateau) * tr).greater(Gf::ZERO), false)
            }
        }
        _ => {
            let rise = h / tr;
            if d < rise {
                (d * tr, false)
            } else if d < rise + plateau {
                (h, true)
            } else {
                ((h - (d - rise - plateau) * tp).greater(Gf::ZERO), false)
            }
        }
    }
}

/// A wedge node's surface rise over `[t0, t1)`, sixteenths: the floored height at `t1` less the floored height at
/// `t0`, so the steps sum exactly to the end's height.
#[must_use]
pub fn wedge_step(w: &WedgeNode, t0: u64, t1: u64) -> i64 {
    let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let (h0, _) = wedge_height_m(w, wedge_area_m2(w, t0));
    let (h1, _) = wedge_height_m(w, wedge_area_m2(w, t1));
    (h1 * steps).to_i64_floor() - (h0 * steps).to_i64_floor()
}

/// The share of a window `[start, end)` that falls in `[t0, t1)`, as an exact cumulative split of `total`:
/// `cum(t) = ⌊total · clamp(t − start, 0, end − start)/(end − start)⌋`, the step `cum(t1) − cum(t0)`.
#[must_use]
pub fn window_step(total: i64, start: u64, end: u64, t0: u64, t1: u64) -> i64 {
    if end <= start {
        return 0;
    }
    let span = i128::from(end - start);
    let cum = |t: u64| -> i128 {
        let inside = i128::from(t.clamp(start, end) - start);
        (i128::from(total) * inside).div_euclid(span)
    };
    (cum(t1) - cum(t0)) as i64
}

/// The years of `[a0, a1)` inside `[t0, t1)`.
#[must_use]
pub fn overlap(a0: u64, a1: u64, t0: u64, t1: u64) -> u64 {
    a1.min(t1).saturating_sub(a0.max(t0))
}

/// The force balance's height, metres: the root of `ρ_c·g·[L·h + h²·ρ_m/(2Δρ)] = F_b`.
#[must_use]
pub fn force_height_m(force_n_m: Gf, crust_m: Gf, gravity_m_s2: Gf) -> Gf {
    let rc = Gf::from_f64(CONTINENTAL_CRUST_DENSITY_KGM3);
    let rm = Gf::from_f64(MANTLE_DENSITY_KGM3);
    let a = rc * gravity_m_s2 * rm / (Gf::TWO * (rm - rc));
    let b = rc * gravity_m_s2 * crust_m;
    (-b + (b * b + Gf::from_f64(4.0) * a * force_n_m).sqrt()) / (Gf::TWO * a)
}

/// The force a belt's ceiling holds, N a metre of margin: Tibet's mean height over Earth's crust under Earth's
/// gravity, carried to the body by its elastic thickness.
#[must_use]
pub fn ceiling_force_n_m(elastic_thickness_m: u32) -> Gf {
    let rc = Gf::from_f64(CONTINENTAL_CRUST_DENSITY_KGM3);
    let rm = Gf::from_f64(MANTLE_DENSITY_KGM3);
    let h = Gf::from_f64(TIBET_MEAN_M);
    rc * Gf::from_f64(EARTH_GRAVITY_M_S2)
        * h
        * (Gf::from_f64(EARTH_CONTINENTAL_CRUST_M) + h * rm / (Gf::TWO * (rm - rc)))
        * Gf::from_i64(i64::from(elastic_thickness_m))
        / Gf::from_f64(EARTH_ELASTIC_THICKNESS_M)
}

/// A unit draw from a hash: the top 53 bits over `2^53`.
fn unit_of(h: u64) -> Gf {
    Gf::from_i64((h >> 11) as i64) / Gf::from_i64(1i64 << 53)
}

fn sub3(a: [Gf; 3], b: [Gf; 3]) -> [Gf; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [Gf; 3], b: [Gf; 3]) -> Gf {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A pair's key: its two plates, the smaller first.
fn pair_key(a: u8, b: u8) -> u16 {
    (u16::from(a.min(b)) << 8) | u16::from(a.max(b))
}

/// ★ THE BELTS' LAW on a body's initial land over its lattice (see the module): every node with continental crust
/// whose nearest boundary closes becomes a wedge node, sized by its margin's rock over its window; a body with one
/// plate, or with no ridge to open an ocean, has no belt.
///
/// ★ THE WEDGE READS THE NODE'S CRUST SHARE (generator 26, ruling B61 item 1): a node of crust share `s` takes in `s`
/// of the margin's rock — the cell's mean of the law's own variable — in place of a yes-or-no test at one half. MEASURED
/// before (the bench's Andean coast tile, 200–345 km inside plate 1): the solved ground fell 3 648 → 570 m between two
/// nodes of share 135 and 124, a cliff along the continent's half-share line that stepped with the node rows; after,
/// 4 167 → 2 051 m over five nodes, and the coast slope's 99th percentile 19° → 10°.
///
/// ★ THE ANDEAN WEDGE RISES FROM THE COAST (generator 26, rulings B59 item 2, B60 item 1): an Andean wedge node reads
/// its distance from the coast the margin's law places ([`margin_width_m`]: the crossing of the plate interface and
/// the Moho) in place of its distance from the trench, and a node between the trench and the coast (the forearc)
/// raises no wedge; the land's trench wall stands there.
#[must_use]
pub fn belts(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &SolveWords,
    land: &Land,
    age_yr: u64,
) -> Belts {
    let n = lattice.node_count();
    let radius = Gf::from_f64(body.ladder().radius_m());
    let g = Gf::from_i64(i64::from(body.facts().gravity_mm_s2)) / Gf::from_i64(1_000);
    let continental_m = Gf::from_f64(EARTH_CONTINENTAL_CRUST_M) * Gf::from_f64(relief_cap_m(body))
        / Gf::from_f64(EARTH_RELIEF_CAP_M)
        * land.freeboard_factor;
    let speed_m_yr = Gf::from_f64(PLATE_SPEED_PER_SHARE_MM_YR) / Gf::from_i64(1_000);
    let te = Gf::from_i64(i64::from(words.elastic_thickness_m));
    let decoupling_m =
        Gf::from_f64(INDIA_STORED_CRUST_M) * te / Gf::from_f64(EARTH_ELASTIC_THICKNESS_M);
    let tau_yr = Gf::ONE / (Gf::from_f64(TIBET_STRAIN_RATE_PER_S) * Gf::from_f64(SECONDS_PER_YEAR));
    let force = ceiling_force_n_m(words.elastic_thickness_m);
    let node_m = Gf::from_f64(lattice.node_m());
    let bounds: Vec<crate::land::Boundary> = (0..n as u32)
        .map(|k| boundary_of(&land.plates, dir_of(lattice.direction(k))))
        .collect();
    // The plates' clock: the ridges' opening (the nodes within half a node of a divergent boundary's bisector) and the
    // oceanic crust's area — Parsons's turnover `t_m = 2 · area / C0`.
    let (mut c0, mut ocean) = (Gf::ZERO, Gf::ZERO);
    for (k, (b, &crust)) in bounds.iter().zip(&land.crust).enumerate() {
        let area = Gf::from_i64(lattice.area_m2(k as u32) as i64);
        if crust < 128 {
            ocean += area;
        }
        let ridge = (b.kind == Kind::Divergent) & (radius * b.sine < node_m / Gf::TWO);
        if ridge {
            c0 += area / node_m * b.convergence.abs() * speed_m_yr;
        }
    }
    let mut out = Belts {
        total: vec![0; n],
        start_yr: vec![age_yr; n],
        kind: vec![BELT_NONE; n],
        plateau: vec![false; n],
        wedge: Vec::new(),
        report: BeltReport::default(),
    };
    if c0 <= Gf::ZERO {
        return out;
    }
    let t_m = Gf::TWO * ocean / c0;
    out.report.t_m_yr = t_m.to_i64_floor() as u64;
    // Every wedge node and its pair.
    let cap_m = relief_cap_m(body);
    let mut wedge: Vec<(WedgeNode, u16)> = Vec::new();
    for (k, &b) in bounds.iter().enumerate() {
        let closing = (b.kind == Kind::Convergent) & (land.crust[k] > 0);
        let Some(o) = b.other.filter(|_| closing) else {
            continue;
        };
        let s = Gf::from_i64(i64::from(land.crust[k])) / Gf::from_i64(255);
        let own = &land.plates[usize::from(b.plate)];
        let other = &land.plates[usize::from(o)];
        let v_n = b.convergence * speed_m_yr;
        let collision = other.affinity >= Gf::ZERO;
        // The boundary's normal from the own plate's site toward the other's (as `boundary_of` reads it), and each
        // plate's drift toward the other: the faster one is the incoming plate.
        // Two plates that share a boundary stand at two different sites, so the normal has a length.
        let nv = sub3(other.site, own.site);
        let len = dot3(nv, nv).sqrt();
        let nn = [nv[0] / len, nv[1] / len, nv[2] / len];
        let own_toward = dot3(own.drift, nn);
        let other_toward = -dot3(other.drift, nn);
        let (side, f0) = if collision {
            let pro = own_toward > other_toward;
            let incoming = if pro { own } else { other };
            let kept_m = (continental_m * incoming.crust_scatter).lesser(decoupling_m);
            (if pro { SIDE_PRO } else { SIDE_RETRO }, v_n * kept_m)
        } else {
            (
                SIDE_ANDEAN,
                Gf::from_f64(ANDEAN_SHARE) * v_n * continental_m * own.crust_scatter,
            )
        };
        // The node takes in its crust share of the margin's rock.
        let f0 = f0 * s;
        // An Andean wedge rises from the coast; the forearc raises none.
        let mut dist_m = radius * b.sine;
        if side == SIDE_ANDEAN {
            let coast = margin_width_m(
                Kind::Convergent,
                own,
                cap_m,
                land.freeboard_factor,
                words.elastic_thickness_m,
            );
            if dist_m < coast {
                continue;
            }
            dist_m -= coast;
        }
        let key = pair_key(b.plate, o);
        let share = unit_of(vd_seed::rng::child_seed(
            body.seed(),
            WINDOW_SALT,
            u64::from(key),
        ));
        let window = (share * t_m).to_i64_floor() as u64;
        wedge.push((
            WedgeNode {
                node: k as u32,
                side,
                dist_m,
                f0_m2_yr: f0,
                // The window opens no earlier than the body formed: the law's own domain.
                start_yr: age_yr.saturating_sub(window),
                ceil_m: force_height_m(force, continental_m * own.crust_scatter, g),
                tau_yr: collision.then_some(tau_yr),
            },
            key,
        ));
    }
    // ONE LEVEL A PAIR: the area-weighted mean over the pair's wedge nodes of their platform plus their force height.
    let steps_m = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let platform = |w: &WedgeNode| Gf::from_i32(land.z[w.node as usize]) / steps_m;
    let mut sums: BTreeMap<u16, (Gf, Gf)> = BTreeMap::new();
    for (w, key) in &wedge {
        let a = Gf::from_i64(lattice.area_m2(w.node) as i64);
        let e = sums.entry(*key).or_insert((Gf::ZERO, Gf::ZERO));
        e.0 += (platform(w) + w.ceil_m) * a;
        e.1 += a;
    }
    for (w, key) in &mut wedge {
        let (sum, area) = sums[key];
        w.ceil_m = (sum / area - platform(w)).greater(Gf::ZERO);
    }
    // The nodes that rise by the end.
    for (w, _) in wedge {
        let (h, on) = wedge_height_m(&w, wedge_area_m2(&w, age_yr));
        if h <= Gf::ZERO {
            continue;
        }
        let k = w.node as usize;
        out.total[k] = (h * steps_m).to_i64_floor();
        out.start_yr[k] = w.start_yr;
        out.plateau[k] = on;
        let volume = (h * Gf::from_i64(lattice.area_m2(w.node) as i64)).to_i64_floor() as u64;
        out.report.nodes += 1;
        out.report.plateau_nodes += usize::from(on);
        if w.side == SIDE_ANDEAN {
            out.kind[k] = BELT_ANDEAN;
            out.report.andean_nodes += 1;
            out.report.volume_m3[1] += volume;
        } else {
            out.kind[k] = BELT_COLLISION;
            out.report.collision_nodes += 1;
            out.report.volume_m3[0] += volume;
        }
        out.wedge.push(w);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(side: u8, dist_m: f64, ceil_m: f64, tau_yr: Option<f64>) -> WedgeNode {
        WedgeNode {
            node: 0,
            side,
            dist_m: Gf::from_f64(dist_m),
            f0_m2_yr: Gf::from_f64(100.0),
            start_yr: 1_000,
            ceil_m: Gf::from_f64(ceil_m),
            tau_yr: tau_yr.map(Gf::from_f64),
        }
    }

    /// The rock taken in grows with the window's years, and a slowing collision takes in less than a steady one.
    #[test]
    fn the_wedge_takes_in_its_margins_rock() {
        let steady = node(SIDE_PRO, 0.0, 3_000.0, None);
        assert_eq!(wedge_area_m2(&steady, 500), Gf::ZERO);
        assert_eq!(wedge_area_m2(&steady, 1_000), Gf::ZERO);
        assert_eq!(wedge_area_m2(&steady, 2_000), Gf::from_f64(100_000.0));
        let slowing = node(SIDE_PRO, 0.0, 3_000.0, Some(1_000.0));
        let a = wedge_area_m2(&slowing, 2_000);
        assert!(a < Gf::from_f64(100_000.0));
        assert!(
            a > Gf::from_f64(60_000.0),
            "about (1 − 1/e) of the steady rock"
        );
    }

    /// ★ The wedge grows as a triangle until its crest meets the ceiling, then a plateau widens behind it; each side
    /// reads its own flank, and a ceiling of nothing holds a plateau of no height.
    #[test]
    fn a_wedge_grows_a_plateau_at_its_ceiling() {
        let small = Gf::from_f64(1.0e6);
        let big = Gf::from_f64(1.0e10);
        let retro = node(SIDE_RETRO, 0.0, 3_000.0, None);
        let (h, on) = wedge_height_m(&retro, small);
        assert!(h > Gf::ZERO);
        assert!(h < Gf::from_f64(3_000.0));
        assert!(!on, "a triangle has no plateau");
        let (h, on) = wedge_height_m(&retro, big);
        assert_eq!(h, Gf::from_f64(3_000.0));
        assert!(on, "past the ceiling the plateau widens");
        // Far behind the plateau the back flank falls to nothing.
        let far = node(SIDE_RETRO, 1.0e9, 3_000.0, None);
        assert_eq!(wedge_height_m(&far, big), (Gf::ZERO, false));
        // The front flank falls from the crest at the critical taper.
        let pro = node(SIDE_PRO, 10_000.0, 3_000.0, None);
        let (h, on) = wedge_height_m(&pro, big);
        assert!(!on);
        assert!(h < Gf::from_f64(3_000.0));
        assert!(h > Gf::from_f64(2_600.0));
        let pro_far = node(SIDE_PRO, 1.0e9, 3_000.0, None);
        assert_eq!(wedge_height_m(&pro_far, big), (Gf::ZERO, false));
        // An Andean wedge: the steep flank from the coast, the plateau, the front ramp inland.
        let coast = node(SIDE_ANDEAN, 1_000.0, 3_000.0, None);
        let (h, on) = wedge_height_m(&coast, big);
        assert!(!on);
        assert_eq!(h, Gf::from_f64(1_000.0) * Gf::from_f64(RETRO_TAPER_TAN));
        let mid = node(SIDE_ANDEAN, 100_000.0, 3_000.0, None);
        assert_eq!(wedge_height_m(&mid, big), (Gf::from_f64(3_000.0), true));
        let inland = node(SIDE_ANDEAN, 1.0e9, 3_000.0, None);
        assert_eq!(wedge_height_m(&inland, big), (Gf::ZERO, false));
        // A ceiling of nothing: a plateau of no height on the retro and Andean sides, nothing on the front.
        let flat_retro = node(SIDE_RETRO, 5_000.0, 0.0, None);
        assert_eq!(wedge_height_m(&flat_retro, big), (Gf::ZERO, true));
        let flat_andean = node(SIDE_ANDEAN, 5_000.0, 0.0, None);
        assert_eq!(wedge_height_m(&flat_andean, big), (Gf::ZERO, true));
        let flat_pro = node(SIDE_PRO, 5_000.0, 0.0, None);
        assert_eq!(wedge_height_m(&flat_pro, big), (Gf::ZERO, false));
    }

    /// The steps of a window sum exactly to its total, a slowing step closes the same total, and nothing moves outside
    /// the window.
    #[test]
    fn the_steps_sum_exactly() {
        assert_eq!(window_step(100, 10, 10, 0, 50), 0);
        let steps: i64 = (0..10)
            .map(|p| window_step(1_001, 100, 1_100, p * 120, (p + 1) * 120))
            .sum();
        assert_eq!(steps, 1_001);
        assert_eq!(window_step(1_001, 100, 1_100, 0, 100), 0);
        assert_eq!(overlap(10, 20, 0, 15), 5);
        assert_eq!(overlap(10, 20, 30, 40), 0);
        let w = node(SIDE_RETRO, 0.0, 3_000.0, Some(500.0));
        let sum: i64 = (0..20)
            .map(|p| wedge_step(&w, p * 200, (p + 1) * 200))
            .sum();
        let (h, _) = wedge_height_m(&w, wedge_area_m2(&w, 4_000));
        assert_eq!(
            sum,
            (h * Gf::from_i64(i64::from(Z_STEPS_PER_M))).to_i64_floor()
        );
    }

    /// The force balance answers Tibet's mean height for Tibet's force over Earth's crust under Earth's gravity.
    #[test]
    fn the_force_balance_answers_tibet_for_tibets_force() {
        let force = ceiling_force_n_m(35_000);
        let h = force_height_m(
            force,
            Gf::from_f64(EARTH_CONTINENTAL_CRUST_M),
            Gf::from_f64(EARTH_GRAVITY_M_S2),
        );
        assert!((h - Gf::from_f64(TIBET_MEAN_M)).abs() < Gf::from_f64(1e-6));
    }

    /// Four plates on the home moon's lattice (the moon itself is a stagnant lid, so its land is restated here): two
    /// continents close (a collision), an ocean dives under the first continent (an Andean margin), and a ridge opens
    /// between the first continent and the other ocean (the plates' clock).
    fn four_plates() -> (BodyDefinition, MacroLattice, SolveWords, Land) {
        let body = crate::home::home_moon();
        let lattice = body.macro_lattice().expect("a lattice");
        let words = crate::home::home_moon_solve_words();
        let mut land = crate::land::initial_land(&body, &lattice, &words.land());
        let plate = |site: [f64; 3], drift: [f64; 3], affinity: f64| crate::land::Plate {
            site: site.map(Gf::from_f64),
            drift: drift.map(Gf::from_f64),
            affinity: Gf::from_f64(affinity),
            age: Gf::from_f64(0.5),
            crust_scatter: Gf::ONE,
        };
        land.plates = vec![
            plate([1.0, 0.0, 0.0], [-0.3, 0.3, 0.0], 1.0),
            plate([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], 1.0),
            plate([-1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], -1.0),
            plate([0.0, 0.0, 1.0], [0.4, 0.0, 0.0], -1.0),
        ];
        for k in 0..lattice.node_count() {
            let b = boundary_of(&land.plates, dir_of(lattice.direction(k as u32)));
            let continental = land.plates[usize::from(b.plate)].affinity >= Gf::ZERO;
            land.crust[k] = if continental { 255 } else { 0 };
        }
        (body, lattice, words, land)
    }

    /// ★ The law raises a collision's front and back flanks and an Andean margin, each risen node's total is its
    /// wedge's floored height at the end inside its own window, and every pair's ceiling stands at one level over the
    /// platforms that reach it.
    #[test]
    fn four_plates_raise_a_collision_and_an_andean_margin() {
        let (body, lattice, words, land) = four_plates();
        let b = belts(&body, &lattice, &words, &land, words.age_yr);
        assert!(b.report.t_m_yr > 0);
        assert!(b.report.collision_nodes > 0);
        assert!(b.report.andean_nodes > 0);
        assert_eq!(b.report.nodes, b.wedge.len());
        assert!(b.report.volume_m3[0] > 0);
        assert!(b.report.volume_m3[1] > 0);
        let sides: Vec<u8> = b.wedge.iter().map(|w| w.side).collect();
        for side in [SIDE_PRO, SIDE_RETRO, SIDE_ANDEAN] {
            assert!(sides.contains(&side), "a wedge node on side {side}");
        }
        let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
        for w in &b.wedge {
            let k = w.node as usize;
            let (h, on) = wedge_height_m(w, wedge_area_m2(w, words.age_yr));
            assert_eq!(b.total[k], (h * steps).to_i64_floor());
            assert_eq!(b.plateau[k], on);
            assert!(w.start_yr < words.age_yr);
            assert_eq!(b.start_yr[k], w.start_yr);
            let kind = if w.side == SIDE_ANDEAN {
                BELT_ANDEAN
            } else {
                BELT_COLLISION
            };
            assert_eq!(b.kind[k], kind);
            assert_eq!(
                w.tau_yr.is_some(),
                w.side != SIDE_ANDEAN,
                "only a collision slows"
            );
        }
        // One level a pair: the nodes whose platform the level clears stand at the same platform plus ceiling.
        let level = |w: &WedgeNode| Gf::from_i32(land.z[w.node as usize]) / steps + w.ceil_m;
        let collision: Vec<&WedgeNode> = b
            .wedge
            .iter()
            .filter(|w| (w.side != SIDE_ANDEAN) & (w.ceil_m > Gf::ZERO))
            .collect();
        let first = level(collision[0]);
        for w in &collision {
            assert!((level(w) - first).abs() < Gf::from_f64(1e-6));
        }
    }

    /// A body with one plate (the home moon's stagnant lid) has no ridge and no belt.
    #[test]
    fn a_stagnant_lid_has_no_belt() {
        let moon = crate::home::home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let words = crate::home::home_moon_solve_words();
        let land = crate::land::initial_land(&moon, &lattice, &words.land());
        let b = belts(&moon, &lattice, &words, &land, words.age_yr);
        assert_eq!(b.report, BeltReport::default());
        assert!(b.total.iter().all(|&t| t == 0));
        assert!(b.wedge.is_empty());
    }
}
