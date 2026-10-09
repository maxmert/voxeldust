//! ★ THE ONE COPY OF THE BELTS' LAW FOR THE BENCHES (rulings B60 item 3, B61 item 1). A line-for-line copy of
//! `vd_terrain::belts::belts` (generator 26) in which two of its laws can be taken out — IN THE COPY ONLY; nothing in the
//! game reads this module (the owner, 2026-10-01: no test seams in the game; ruling B38). With both on the copy is the
//! game's law (THE CONTROL: the example `coast_bench` compares every node's total, start, kind and plateau).
//!
//! 1. ★ THE WEDGE READS THE NODE'S CRUST SHARE (`share`): a node of crust share `s` takes in `s` of the margin's rock.
//!    Out, a wedge rises only on a node whose crust share is at least one half (`crust ≥ 128`), as generator 25 did —
//!    MEASURED then on the Andean coast tile (face 3, 200–345 km inside plate 1): the solved ground fell 3 648 → 570 m
//!    between two nodes of share 135 and 124, a cliff along the continent's half-share line.
//! 2. ★ THE ANDEAN WEDGE RISES FROM THE COAST (`andean_coast`; ruling B59 item 2): an Andean wedge node reads its
//!    distance from the coast the margin's law places (`vd_terrain::land::margin_width_m`), and a node between the
//!    trench and the coast raises no wedge. Out, it reads its distance from the trench.
//!
//! **Example.** `phase2-share` in the belts' bench takes the first law out: plate 1's Andean plateau stands 4 km high to
//! the continent's half-share line and drops 3 km in one node again, and the bench prints the coast slope's percentiles.

use std::collections::BTreeMap;

use vd_recipe::Gi;
use vd_recipe::bend::DIR_BITS;
use vd_terrain::BodyDefinition;
use vd_terrain::Gf;
use vd_terrain::belts::{
    ANDEAN_SHARE, BELT_ANDEAN, BELT_COLLISION, BELT_NONE, BeltReport, Belts, INDIA_STORED_CRUST_M,
    PLATE_SPEED_PER_SHARE_MM_YR, SECONDS_PER_YEAR, SIDE_ANDEAN, SIDE_PRO, SIDE_RETRO,
    TIBET_STRAIN_RATE_PER_S, WINDOW_SALT, WedgeNode, ceiling_force_n_m, force_height_m,
    wedge_area_m2, wedge_height_m,
};
use vd_terrain::land::{
    EARTH_CONTINENTAL_CRUST_M, EARTH_ELASTIC_THICKNESS_M, EARTH_RELIEF_CAP_M, Kind, Land,
    boundary_of, relief_cap_m,
};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::solve::{SolveWords, Z_STEPS_PER_M};

/// The copy of `vd_terrain::belts::unit_of` (private).
fn unit_of(h: u64) -> Gf {
    Gf::from_i64((h >> 11) as i64) / Gf::from_i64(1i64 << 53)
}

fn sub3(a: [Gf; 3], b: [Gf; 3]) -> [Gf; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [Gf; 3], b: [Gf; 3]) -> Gf {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn pair_key(a: u8, b: u8) -> u16 {
    (u16::from(a.min(b)) << 8) | u16::from(a.max(b))
}

/// The copy of `vd_terrain::land::dir_of` (crate-private).
fn dir_of(dir: [Gi; 3]) -> [Gf; 3] {
    let one = Gf::from_i64(1 << DIR_BITS);
    [
        Gf::from_i64(dir[0].raw()) / one,
        Gf::from_i64(dir[1].raw()) / one,
        Gf::from_i64(dir[2].raw()) / one,
    ]
}

/// The copy of `vd_terrain::belts::belts` with the laws a bench may take out (see the module); every line not marked
/// `LAW OUT` is the game's own.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn belts_copy(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &SolveWords,
    land: &Land,
    age_yr: u64,
    share: bool,
    andean_coast: bool,
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
    let bounds: Vec<vd_terrain::land::Boundary> = (0..n as u32)
        .map(|k| boundary_of(&land.plates, dir_of(lattice.direction(k))))
        .collect();
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
    let words_land = words.land();
    let mut wedge: Vec<(WedgeNode, u16)> = Vec::new();
    for (k, &b) in bounds.iter().enumerate() {
        // LAW OUT (`share`): the half-share test in place of the share.
        let s = Gf::from_i64(i64::from(land.crust[k])) / Gf::from_i64(255);
        let closing = if share {
            (b.kind == Kind::Convergent) & (land.crust[k] > 0)
        } else {
            (b.kind == Kind::Convergent) & (land.crust[k] >= 128)
        };
        let Some(o) = b.other.filter(|_| closing) else {
            continue;
        };
        let own = &land.plates[usize::from(b.plate)];
        let other = &land.plates[usize::from(o)];
        let v_n = b.convergence * speed_m_yr;
        let collision = other.affinity >= Gf::ZERO;
        let nv = sub3(other.site, own.site);
        let len = dot3(nv, nv).sqrt();
        let nn = [nv[0] / len, nv[1] / len, nv[2] / len];
        let own_toward = dot3(own.drift, nn);
        let other_toward = -dot3(other.drift, nn);
        let (side, mut f0) = if collision {
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
        if share {
            f0 *= s;
        }
        // LAW OUT (`andean_coast`): the Andean wedge's distance from the trench in place of the coast.
        let mut dist_m = radius * b.sine;
        if andean_coast & (side == SIDE_ANDEAN) {
            let x = vd_terrain::land::margin_width_m(
                Kind::Convergent,
                own,
                relief_cap_m(body),
                land.freeboard_factor,
                words_land.elastic_thickness_m,
            );
            if dist_m < x {
                continue;
            }
            dist_m -= x;
        }
        let key = pair_key(b.plate, o);
        let window_share = unit_of(vd_seed::rng::child_seed(
            body.seed(),
            WINDOW_SALT,
            u64::from(key),
        ));
        let window = (window_share * t_m).to_i64_floor() as u64;
        wedge.push((
            WedgeNode {
                node: k as u32,
                side,
                dist_m,
                f0_m2_yr: f0,
                start_yr: age_yr.saturating_sub(window),
                ceil_m: force_height_m(force, continental_m * own.crust_scatter, g),
                tau_yr: collision.then_some(tau_yr),
            },
            key,
        ));
    }
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
