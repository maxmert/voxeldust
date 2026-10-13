//! ★ ANY PLANET AS A STAND (2026-10-13; the owner chose the Earth-like search's #3, `Planet(18348864443750092586)`:
//! "Let's go to #3 first"). Bench only; THE world (SL5) through the generator's own functions, no variant:
//!
//! - `solve PLANET OUT.art [cores]`: the planet's star system found among the systems nearest home, its charter and body
//!   read as its shard reads them, its 8 km world solved by the game's own first solve (`artifact_worker::run_solve_on`,
//!   the same bytes on any number of cores, ruling B64), written to `OUT.art`, and its 25 km relief printed;
//! - `stands PLANET WORLD.art OUT_DIR`: three stands chosen by the world's own 25 km relief — the land node whose window
//!   reads nearest Kansas's 46 m (a plain), Germany's 175 m (hills) and 1 500 m (a belt) — each one's 128 m tile solved by
//!   the game's tile solve (`l1::solve_l1`), read by `relief_scales`' rule (the relief at 1 km and 5 km over land
//!   windows) and drawn from the side by its camera (2 km up, 8° down, 60° wide).
//!
//! `cargo run --release -p vd-bins --example planet_stand -- solve 18348864443750092586 /tmp/p3.art 8`

use std::collections::BTreeSet;

use vd_core::pose::RealmId;
use vd_core::worldgen::{GALAXY, HOME_SYSTEM};
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::Artifact;
use vd_terrain::fine::delaunay::NONE;
use vd_terrain::fine::geometry::locate;
use vd_terrain::fine::l1::{
    BUFFER_NODES, COARSE_ROUNDS, COARSE_RUNG, Domain, FINE_ROUND_BOUND, L1_RUNG, MAX_PASSES,
    REBUILD_ROUNDS, SETTLE_M, TILE_NODES, TileKey, U0_SEA, cells_per_node_shift, decode_class,
    domain, solve_l1,
};
use vd_terrain::fine::points::chart_cell;
use vd_terrain::fine::tree::{Tree, accumulate, tree};
use vd_terrain::gf::Gf;
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::{JOINT_K_LP, JOINT_N};

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;

/// The planet's body, charter and system, found among the systems nearest home.
fn find(planet: u64) -> (BodyDefinition, vd_core::look::BodyCharter, RealmId) {
    let config =
        vd_physics::worldgen::UniverseConfig::world(vd_bins::DEV.move_speed, vd_bins::DEV.tick_dt);
    let view = vd_physics::worldgen::system_layer_view(UNIVERSE, &config);
    let regions = view.regions();
    let galaxy = regions
        .iter()
        .find(|r| r.realm == GALAXY)
        .expect("the galaxy");
    let home_at = regions
        .iter()
        .find(|r| r.realm == HOME_SYSTEM)
        .expect("the home system")
        .center
        .metres_in(galaxy);
    let mut near: Vec<(f64, RealmId)> = regions
        .iter()
        .filter(|r| matches!(r.realm, RealmId::System(_)) && r.parent == Some(GALAXY))
        .map(|r| ((r.center.metres_in(galaxy) - home_at).length(), r.realm))
        .collect();
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    let lineage = BTreeSet::from([GALAXY]);
    let target = RealmId::Planet(planet);
    for (_, system) in near {
        let held = BTreeSet::from([system]);
        let (rows, _) =
            vd_physics::worldgen::shard_boot_world(UNIVERSE, &config, &held, system, &lineage);
        let Some(row) = rows.iter().find(|r| r.realm == target) else {
            continue;
        };
        let radius_m = match row.look {
            Some(vd_core::geometry::Boundary::Shell { r }) => r,
            _ => panic!("the planet is round"),
        };
        let charter = vd_bins::body_charter(UNIVERSE, &held, &lineage, target).expect("a charter");
        let body = BodyDefinition::from_seed(planet, radius_m, vd_bins::facts_of_charter(&charter))
            .expect("a ladder");
        return (body, charter, system);
    }
    panic!("the planet is in none of the galaxy's systems");
}

/// The 25 km relief of a world: each land node with its eight neighbours.
fn relief25(artifact: &Artifact, lattice: &MacroLattice) -> Vec<Option<f32>> {
    let sea = i32::from(artifact.sea_m);
    let z = |n: u32| i32::from(artifact.rows[n as usize].z_m);
    (0..lattice.node_count() as u32)
        .map(|n| {
            if z(n) <= sea {
                return None;
            }
            let (mut lo, mut hi) = (z(n), z(n));
            for m in lattice.neighbours(n) {
                if m == NO_NODE || z(m) <= sea {
                    return None;
                }
                lo = lo.min(z(m));
                hi = hi.max(z(m));
            }
            Some((hi - lo) as f32)
        })
        .collect()
}

fn summary(name: &str, r: &mut [f32]) {
    r.sort_by(f32::total_cmp);
    if r.is_empty() {
        println!("  {name}: no land window");
        return;
    }
    let q = |f: f64| r[((r.len() - 1) as f64 * f).round() as usize];
    let under = |m: f32| 100.0 * r.iter().filter(|v| **v < m).count() as f64 / r.len() as f64;
    println!(
        "  {name:<40}: p10 {:>5.0} p50 {:>5.0} p90 {:>5.0} m | under 20 m {:>5.1} %, 50 m {:>5.1} %, 100 m {:>5.1} % | {} windows",
        q(0.1),
        q(0.5),
        q(0.9),
        under(20.0),
        under(50.0),
        under(100.0),
        r.len()
    );
}

/// The relief over `win × win` windows of land (every cell over the sea) on a `w`-wide raster of metres.
fn reliefs(z: &[f32], w: usize, win: usize) -> Vec<f32> {
    let h = z.len() / w;
    let mut out = Vec::new();
    for wy in 0..h / win {
        for wx in 0..w / win {
            let (mut lo, mut hi, mut land) = (f32::INFINITY, f32::NEG_INFINITY, true);
            for y in wy * win..(wy + 1) * win {
                for x in wx * win..(wx + 1) * win {
                    let v = z[y * w + x];
                    land &= v.is_finite() && v > 0.0;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            if land {
                out.push(hi - lo);
            }
        }
    }
    out
}

/// ★ THE SIDE VIEW (relief_scales' camera): from the raster's middle, 2 km over the ground, looking up its rows.
fn render(z: &[f32], w: usize, cell_m: f64, path: &str) {
    let h = z.len() / w;
    let (wpx, hpx) = (960u32, 480u32);
    let fov = 60f64.to_radians();
    let (ex, ey) = ((w / 2) as f64 * cell_m, (h * 7 / 8) as f64 * cell_m);
    let ground = |x: f64, y: f64| -> Option<f64> {
        let (cx, cy) = (x / cell_m, y / cell_m);
        if cx < 0.0 || cy < 0.0 || cx >= (w - 1) as f64 || cy >= (h - 1) as f64 {
            return None;
        }
        let (ix, iy) = (cx as usize, cy as usize);
        let (fx, fy) = (cx - ix as f64, cy - iy as f64);
        let v = |a: usize, b: usize| f64::from(z[b * w + a]);
        let g = v(ix, iy) * (1.0 - fx) * (1.0 - fy)
            + v(ix + 1, iy) * fx * (1.0 - fy)
            + v(ix, iy + 1) * (1.0 - fx) * fy
            + v(ix + 1, iy + 1) * fx * fy;
        g.is_finite().then_some(g.max(0.0))
    };
    let eye_z = ground(ex, ey).unwrap_or(0.0) + 2_000.0;
    let pitch = -8f64.to_radians();
    let sun = {
        let (az, el) = (225f64.to_radians(), 25f64.to_radians());
        [el.cos() * az.sin(), -el.cos() * az.cos(), el.sin()]
    };
    let mut img = image::RgbImage::new(wpx, hpx);
    for py in 0..hpx {
        for px in 0..wpx {
            let ax = ((f64::from(px) + 0.5) / f64::from(wpx) - 0.5) * fov;
            let ay = (0.5 - (f64::from(py) + 0.5) / f64::from(hpx)) * fov * f64::from(hpx)
                / f64::from(wpx)
                + pitch;
            let d = [ax.sin() * ay.cos(), -ax.cos() * ay.cos(), ay.sin()];
            let (mut t, mut hit) = (0.0f64, None);
            while t < 60_000.0 {
                let (x, y, zz) = (ex + d[0] * t, ey + d[1] * t, eye_z + d[2] * t);
                match ground(x, y) {
                    None => break,
                    Some(gz) if zz <= gz => {
                        hit = Some((x, y, gz, t));
                        break;
                    }
                    _ => {}
                }
                t += cell_m * 0.5 * (1.0 + t / 8_000.0);
            }
            let sky = [150.0, 180.0, 215.0];
            let c = match hit {
                None => sky,
                Some((x, y, gz, t)) => {
                    let dzx =
                        ground(x + cell_m, y).unwrap_or(gz) - ground(x - cell_m, y).unwrap_or(gz);
                    let dzy =
                        ground(x, y + cell_m).unwrap_or(gz) - ground(x, y - cell_m).unwrap_or(gz);
                    let n = [-dzx, -dzy, 2.0 * cell_m];
                    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    let lit = ((n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]) / l).max(0.0);
                    let base = if gz <= 0.0 {
                        [40.0, 70.0, 120.0]
                    } else {
                        [170.0, 160.0, 140.0]
                    };
                    let shade = 0.25 + 0.75 * lit;
                    let haze = 1.0 - (-t / 45_000.0).exp();
                    [0, 1, 2].map(|k| base[k] * shade * (1.0 - haze) + sky[k] * haze)
                }
            };
            img.put_pixel(px, py, image::Rgb(c.map(|v| v.clamp(0.0, 255.0) as u8)));
        }
    }
    img.save(path).expect("the picture saves");
}

// ★ THE FORM BENCH (`forms`, 2026-10-13; the owner: "Ok, proceed" to one bench round on #3's hills): the game's 128 m
// tile beside variants of HOW the game embeds the law under the 8 km world, one switch at a time, each read for its
// relief, its ROWS and its BRANCHING beside Earth's tiles, and drawn from the side and from above. The game's
// `solve_l1` is the control; the bench's copy of it (below) must give the same heights with every switch off.

/// One variant: which of the game's embedding rules it changes.
#[derive(Clone, Copy)]
struct Variant {
    name: &'static str,
    /// Every point on the domain's outer rim is an outlet at the 8 km surface (the B80 bench's open region), beside the
    /// game's rule (outlets only where an 8 km river leaves).
    open_rim: bool,
    /// The rise read off the 8 km world by the same law (`U = K·Q·S^n`, each node's drop to its receiver and its
    /// discharge), held fixed — no rounds pull a node's mean to its row.
    law_rise: bool,
    /// The game's coarse start (points four times further apart, solved first); off, the fine rounds start from the 8 km
    /// surface itself.
    coarse_start: bool,
    /// ★ S1 THE SOIL SLOPE (the landforms design §2.0, revision 2, ruling B79): each point's slope the gentler of the
    /// stream law's and the soil law's, so divides round and channels start where the two meet.
    soil: bool,
}

const VARIANTS: [Variant; 6] = [
    Variant {
        name: "A_today",
        open_rim: false,
        law_rise: false,
        coarse_start: true,
        soil: false,
    },
    Variant {
        name: "B_open_rim",
        open_rim: true,
        law_rise: false,
        coarse_start: true,
        soil: false,
    },
    Variant {
        name: "C_law_rise",
        open_rim: false,
        law_rise: true,
        coarse_start: true,
        soil: false,
    },
    Variant {
        name: "D_open_rim_law_rise",
        open_rim: true,
        law_rise: true,
        coarse_start: true,
        soil: false,
    },
    Variant {
        name: "E_no_coarse_start",
        open_rim: false,
        law_rise: false,
        coarse_start: false,
        soil: false,
    },
    Variant {
        name: "S_soil_slope",
        open_rim: false,
        law_rise: false,
        coarse_start: true,
        soil: true,
    },
];

/// S1's creep rate (m²/yr) — the landforms design's interim: Roering, Kirchner & Dietrich 1999, the Oregon Coast Range.
const SOIL_KH_M2_YR: f64 = 0.003;
/// S1's rock over soil density (Roering, Perron & Kirchner 2007).
const ROCK_OVER_SOIL: f64 = 2.0;

/// ★ S1 THE SOIL SLOPE (Roering, Kirchner & Dietrich 1999; Roering, Perron & Kirchner 2007): the flux
/// `q = K_h·S / (1 − (S/S_c)²)` carries the rock a steady hillslope sheds, `q = (ρ_r/ρ_s)·E·x` (E the rate, x the area
/// drained per unit of contour), in the design's closed form `S = 2q / (K_h + √(K_h² + 4q²/S_c²))`; `S_c` the level's
/// own threshold (tan 35° at 128 m, ruling B82).
fn soil_slope(e: Gf, x: Gf) -> Gf {
    let sc = vd_terrain::body::TAN_REPOSE;
    let q = ROCK_OVER_SOIL * e.to_f64() * x.to_f64();
    Gf::from_f64(
        2.0 * q
            / (SOIL_KH_M2_YR + (SOIL_KH_M2_YR * SOIL_KH_M2_YR + 4.0 * q * q / (sc * sc)).sqrt()),
    )
}

/// Each point's area drained per unit of contour on the tree `t`: the area through it over its own width (the square
/// root of its own area).
fn drained_per_width(d: &Domain, t: &Tree) -> Vec<Gf> {
    accumulate(t, &d.geom.area)
        .iter()
        .zip(&d.geom.area)
        .map(|(&a, &own)| a / own.sqrt())
        .collect()
}

/// Every point on the domain's outer rim made an outlet at its base (the 8 km surface there); returns how many.
fn open_rim(d: &mut Domain, key: TileKey, shift: u32) -> usize {
    let (ni0, nj0) = (
        key.tx * TILE_NODES - BUFFER_NODES,
        key.ty * TILE_NODES - BUFFER_NODES,
    );
    let s = (TILE_NODES + 2 * BUFFER_NODES) << shift;
    let mut n = 0;
    for p in 0..d.cells.len() {
        let c = d.cells[p];
        let (a, b) = (c.i - (ni0 << shift), c.j - (nj0 << shift));
        if c.face == key.face.index()
            && (a == 0 || b == 0 || a == s - 1 || b == s - 1)
            && !d.outlet[p]
        {
            d.outlet[p] = true;
            n += 1;
        }
    }
    n
}

/// The rise of each read node by the joint law off the 8 km world: `U = K·Q·S^n` (S the node's drop to its receiver
/// over the link, Q its discharge); a sea node, a node with no receiver or no drop takes the sea's rise.
fn law_rises(d: &Domain, lattice: &MacroLattice, artifact: &Artifact, radius_m: f64) -> Vec<Gf> {
    use vd_terrain::artifact::{RECEIVER_NONE, RECEIVER_SLOT_MASK};
    let unit = |n: u32| {
        let s = (1u64 << 40) as f64;
        let g = lattice.direction(n);
        [
            g[0].raw() as f64 / s,
            g[1].raw() as f64 / s,
            g[2].raw() as f64 / s,
        ]
    };
    d.refs
        .iter()
        .map(|&m| {
            let row = &artifact.rows[m as usize];
            let f = row.receiver_facies;
            if row.z_m <= artifact.sea_m || f & RECEIVER_NONE != 0 {
                return Gf::from_f64(U0_SEA);
            }
            let r = lattice.neighbours(m)[(f & RECEIVER_SLOT_MASK) as usize];
            if r == NO_NODE {
                return Gf::from_f64(U0_SEA);
            }
            let (a, b) = (unit(m), unit(r));
            let link = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                .sqrt()
                * radius_m;
            let drop = f64::from(row.z_m) - f64::from(artifact.rows[r as usize].z_m);
            if drop <= 0.0 {
                return Gf::from_f64(U0_SEA);
            }
            let q = decode_class(row.discharge_log, 2) as f64 / 1_000.0;
            Gf::from_f64(JOINT_K_LP * q * (drop / link).powf(JOINT_N))
        })
        .collect()
}

// The bench's copy of the game's rounds (`l1.rs`, private there), with the node hold as a switch.
fn node_means(d: &Domain, h: &[Gf]) -> Vec<Gf> {
    let nl = d.nodes.len();
    let (mut sum, mut area) = (vec![Gf::ZERO; nl], vec![Gf::ZERO; nl]);
    for ((&k, &hp), &a) in d.pnode.iter().zip(h).zip(&d.geom.area) {
        sum[k as usize] += hp * a;
        area[k as usize] += a;
    }
    (0..nl)
        .map(|k| {
            if area[k] > Gf::ZERO {
                sum[k] / area[k]
            } else {
                Gf::ZERO
            }
        })
        .collect()
}

fn point_rises(d: &Domain, u: &[Gf]) -> Vec<Gf> {
    let ln_u: Vec<Gf> = u.iter().map(|x| x.ln_fine()).collect();
    (0..d.base.len())
        .map(|p| {
            let mut s = Gf::ZERO;
            for t in d.wstart[p] as usize..d.wstart[p + 1] as usize {
                s += d.wval[t] * ln_u[d.wnode[t] as usize];
            }
            s.exp_fine()
        })
        .collect()
}

fn steady_pass(d: &Domain, t: &Tree, q: &[Gf], up: &[Gf], h: &mut [Gf], soil: bool) {
    let x = if soil {
        drained_per_width(d, t)
    } else {
        Vec::new()
    };
    let (k_lp, inv_n) = (Gf::from_f64(JOINT_K_LP), Gf::from_f64(1.0 / JOINT_N));
    let tan = Gf::from_f64(vd_terrain::body::TAN_REPOSE);
    let tiny = Gf::from_f64(f64::MIN_POSITIVE);
    for &c in &t.order {
        let c = c as usize;
        let r = t.recv[c];
        if d.outlet[c] || r == NONE {
            h[c] = d.base[c];
            continue;
        }
        let mut s = (up[c] / (k_lp * q[c].greater(tiny)))
            .powf_fine(inv_n)
            .lesser(tan);
        if soil {
            s = s.lesser(soil_slope(up[c], x[c]));
        }
        h[c] = h[r as usize] + s * t.link[c];
    }
}

fn round_slopes(d: &Domain, u: &[Gf], ln_kq: &[Gf], x: Option<&[Gf]>) -> Vec<Gf> {
    let inv_n = Gf::from_f64(1.0 / JOINT_N);
    let tan = Gf::from_f64(vd_terrain::body::TAN_REPOSE);
    let ln_u: Vec<Gf> = u.iter().map(|x| x.ln_fine()).collect();
    (0..d.base.len())
        .map(|p| {
            let mut l = Gf::ZERO;
            for t in d.wstart[p] as usize..d.wstart[p + 1] as usize {
                l += d.wval[t] * ln_u[d.wnode[t] as usize];
            }
            let s = ((l - ln_kq[p]) * inv_n).exp_fine().lesser(tan);
            match x {
                Some(x) => s.lesser(soil_slope(l.exp_fine(), x[p])),
                None => s,
            }
        })
        .collect()
}

fn integrate(d: &Domain, t: &Tree, slopes: &[Gf], h: &mut [Gf]) {
    for &c in &t.order {
        let c = c as usize;
        let r = t.recv[c];
        if d.outlet[c] || r == NONE {
            h[c] = d.base[c];
            continue;
        }
        h[c] = h[r as usize] + slopes[c] * t.link[c];
    }
}

fn moved_max(d: &Domain, now: &[Gf], last: &[Gf]) -> Gf {
    (0..d.nodes.len())
        .filter(|&k| d.held[k])
        .map(|k| (now[k] - last[k]).abs())
        .fold(Gf::ZERO, Gf::greater)
}

fn half_step(d: &Domain, u: &mut [Gf], means: &[Gf]) {
    let half_n = Gf::from_f64(0.5 * JOINT_N);
    for (k, &mean) in means.iter().enumerate() {
        let r = d.node_ref[k] as usize;
        if d.held[k] && mean > Gf::ZERO && r < u.len() {
            u[r] *= (d.node_z[k] / mean).powf_fine(half_n);
        }
    }
}

fn coarse_rounds(d: &Domain, u: &mut [Gf], h: &mut [Gf], rounds: usize, hold: bool, soil: bool) {
    let settle = Gf::from_f64(SETTLE_M);
    for _ in 0..rounds {
        let up = point_rises(d, u);
        let mut last = node_means(d, h);
        let mut passes = 0;
        while passes < MAX_PASSES {
            passes += 1;
            let t = tree(h, &d.outlet, &d.graph, &d.geom.len);
            let q = accumulate(&t, &d.water);
            steady_pass(d, &t, &q, &up, h, soil);
            let now = node_means(d, h);
            let moved = moved_max(d, &now, &last);
            last = now;
            if moved <= settle {
                break;
            }
        }
        if hold {
            half_step(d, u, &last);
        }
    }
}

fn network(d: &Domain, h: &[Gf]) -> (Tree, Vec<Gf>, Vec<Gf>) {
    let t = tree(h, &d.outlet, &d.graph, &d.geom.len);
    let x = drained_per_width(d, &t);
    let (k_lp, tiny) = (Gf::from_f64(JOINT_K_LP), Gf::from_f64(f64::MIN_POSITIVE));
    let ln_kq = accumulate(&t, &d.water)
        .iter()
        .map(|&qc| (k_lp * qc.greater(tiny)).ln_fine())
        .collect();
    (t, ln_kq, x)
}

fn fine_rounds(d: &Domain, u: &mut [Gf], h: &mut [Gf], hold: bool, soil: bool) -> usize {
    let settle = Gf::from_f64(SETTLE_M);
    let (mut t, mut ln_kq, mut x) = network(d, h);
    let mut last = node_means(d, h);
    for round in 1..=FINE_ROUND_BOUND {
        let slopes = round_slopes(d, u, &ln_kq, soil.then_some(x.as_slice()));
        integrate(d, &t, &slopes, h);
        let now = node_means(d, h);
        let moved = moved_max(d, &now, &last);
        if hold {
            half_step(d, u, &now);
        }
        last = now;
        if moved <= settle {
            return round;
        }
        if round % REBUILD_ROUNDS == 0 {
            (t, ln_kq, x) = network(d, h);
        }
    }
    FINE_ROUND_BOUND
}

/// A variant's raster, its side, its rounds and its fit.
type Solved = (Vec<f32>, usize, usize, (f64, f64));
/// A variant's fine domain, its rises, its heights, its rounds and its fit.
type Points = (Domain, Vec<Gf>, Vec<Gf>, usize, (f64, f64));

/// One variant's solved points (the bench's copy of `solve_l1` with the variant's switches): the fine domain, its rises
/// (one a read node), its heights, the rounds run, and the held nodes' fit to their rows (largest, mean).
fn solve_points(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    key: TileKey,
    v: Variant,
) -> Option<Points> {
    let hold = !v.law_rise;
    let mut fine = domain(body, lattice, artifact, L1_RUNG, key)?;
    if v.open_rim {
        open_rim(
            &mut fine,
            key,
            cells_per_node_shift(body, lattice, L1_RUNG)?,
        );
    }
    let fine_law = law_rises(&fine, lattice, artifact, body.radius_m());
    let (mut u, mut h) = if v.coarse_start {
        let mut coarse = domain(body, lattice, artifact, COARSE_RUNG, key)?;
        if v.open_rim {
            open_rim(
                &mut coarse,
                key,
                cells_per_node_shift(body, lattice, COARSE_RUNG)?,
            );
        }
        let mut uc = if v.law_rise {
            law_rises(&coarse, lattice, artifact, body.radius_m())
        } else {
            coarse.ref_u0.clone()
        };
        let mut hc = coarse.base.clone();
        coarse_rounds(&coarse, &mut uc, &mut hc, COARSE_ROUNDS, hold, v.soil);
        let u: Vec<Gf> = if v.law_rise {
            fine_law
        } else {
            fine.refs
                .iter()
                .zip(&fine.ref_u0)
                .map(|(m, &u0)| coarse.refs.binary_search(m).map_or(u0, |i| uc[i]))
                .collect()
        };
        let mut h = fine.base.clone();
        let mut from = 0u32;
        for ((hp, dir), &out) in h.iter_mut().zip(&fine.dirs).zip(&fine.outlet) {
            if out {
                continue;
            }
            if let Some((t, w)) = locate(&coarse.dirs, &coarse.graph, from, dir) {
                let tri = coarse.graph.tris[t as usize];
                *hp = w[0] * hc[tri[0] as usize]
                    + w[1] * hc[tri[1] as usize]
                    + w[2] * hc[tri[2] as usize];
                from = t;
            }
        }
        (u, h)
    } else {
        let u = if v.law_rise {
            fine_law
        } else {
            fine.ref_u0.clone()
        };
        (u, fine.base.clone())
    };
    let rounds = fine_rounds(&fine, &mut u, &mut h, hold, v.soil);
    // The fit: each held node's fine mean against its row.
    let means = node_means(&fine, &h);
    let gaps: Vec<f64> = (0..fine.nodes.len())
        .filter(|&k| fine.held[k])
        .map(|k| (means[k] - fine.node_z[k]).abs().to_f64())
        .collect();
    let fit = (
        gaps.iter().copied().fold(0.0, f64::max),
        gaps.iter().sum::<f64>() / gaps.len().max(1) as f64,
    );
    Some((fine, u, h, rounds, fit))
}

/// One variant's tile: its raster, side, rounds and fit.
fn solve_variant(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    key: TileKey,
    v: Variant,
) -> Option<Solved> {
    let (fine, _, h, rounds, fit) = solve_points(body, lattice, artifact, key, v)?;
    let (z, side) = raster_of(body, lattice, key, &fine, &h)?;
    Some((z, side, rounds, fit))
}

/// The raster of the heights `h` on `fine` (as `solve_l1`'s): the tile and half its buffer, metres over the sea, and
/// its side.
fn raster_of(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    key: TileKey,
    fine: &Domain,
    h: &[Gf],
) -> Option<(Vec<f32>, usize)> {
    let n_r = body.ladder().cells_per_edge(L1_RUNG);
    let shift = cells_per_node_shift(body, lattice, L1_RUNG)?;
    let half = BUFFER_NODES >> 1;
    let (i0, j0) = (
        (key.tx * TILE_NODES - half) << shift,
        (key.ty * TILE_NODES - half) << shift,
    );
    let side = ((TILE_NODES + 2 * half) << shift) as usize;
    let inv_n = vd_recipe::bend::inv_n_of(n_r);
    let mut z = Vec::with_capacity(side * side);
    let mut row_from = 0u32;
    for b in 0..side as i32 {
        let mut from = row_from;
        for a in 0..side as i32 {
            let Some(c) = chart_cell(key.face, i0 + a, j0 + b, n_r as i32) else {
                z.push(f32::NAN);
                continue;
            };
            let q = vd_recipe::bend::direction(i32::from(c.face), c.i, c.j, inv_n);
            let (t, w) = locate(&fine.dirs, &fine.graph, from, &q)?;
            let tri = fine.graph.tris[t as usize];
            let zz =
                w[0] * h[tri[0] as usize] + w[1] * h[tri[1] as usize] + w[2] * h[tri[2] as usize];
            z.push(((zz * Gf::from_i64(4) + Gf::HALF).floor().to_i64_floor() as f32) / 4.0);
            if a == 0 {
                row_from = t;
            }
            from = t;
        }
    }
    Some((z, side))
}

/// ★ THE ROWS NUMBER: how strongly the local slopes line up on ONE axis. The heights less their 5 km box mean (so a
/// region's tilt counts nothing), the slope at each cell, and the length of the mean of `e^(2iθ)` weighted by the slope
/// (θ the slope's direction; the doubled angle counts the two sides of a ridge as one axis). Nought: slopes in every
/// direction (a branching network); one: every slope across one axis (parallel ridges). Returns it and the axis's
/// direction in degrees.
fn rows_number(z: &[f32], w: usize) -> (f64, f64) {
    let h = z.len() / w;
    let r = 19usize;
    // The summed-area tables of the finite heights and their count.
    let (mut s, mut c) = (
        vec![0.0f64; (w + 1) * (h + 1)],
        vec![0.0f64; (w + 1) * (h + 1)],
    );
    for y in 0..h {
        for x in 0..w {
            let v = z[y * w + x];
            let (vs, vc) = if v.is_finite() {
                (f64::from(v), 1.0)
            } else {
                (0.0, 0.0)
            };
            let at = (y + 1) * (w + 1) + x + 1;
            s[at] = vs + s[at - 1] + s[at - (w + 1)] - s[at - (w + 1) - 1];
            c[at] = vc + c[at - 1] + c[at - (w + 1)] - c[at - (w + 1) - 1];
        }
    }
    let boxed = |t: &[f64], x0: usize, y0: usize, x1: usize, y1: usize| {
        t[y1 * (w + 1) + x1] - t[y0 * (w + 1) + x1] - t[y1 * (w + 1) + x0] + t[y0 * (w + 1) + x0]
    };
    let detrended = |x: usize, y: usize| -> f64 {
        let (x0, y0, x1, y1) = (
            x.saturating_sub(r),
            y.saturating_sub(r),
            (x + r + 1).min(w),
            (y + r + 1).min(h),
        );
        let n = boxed(&c, x0, y0, x1, y1);
        f64::from(z[y * w + x]) - boxed(&s, x0, y0, x1, y1) / n.max(1.0)
    };
    let (mut cx, mut cy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let near = [
                z[y * w + x - 1],
                z[y * w + x + 1],
                z[(y - 1) * w + x],
                z[(y + 1) * w + x],
            ];
            if near.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let gx = (detrended(x + 1, y) - detrended(x - 1, y)) / 2.0;
            let gy = (detrended(x, y + 1) - detrended(x, y - 1)) / 2.0;
            let g = (gx * gx + gy * gy).sqrt();
            if g == 0.0 {
                continue;
            }
            // e^(2iθ) = ((gx² − gy²) + i·2·gx·gy) / g², weighted by g.
            cx += (gx * gx - gy * gy) / g;
            cy += 2.0 * gx * gy / g;
            sw += g;
        }
    }
    let r2 = (cx * cx + cy * cy).sqrt() / sw.max(f64::MIN_POSITIVE);
    (r2, 0.5 * cy.atan2(cx).to_degrees())
}

/// ★ THE BRANCHING: the drainage of a raster (the depressions filled by a priority flood, each cell draining to its
/// steepest of eight neighbours), its channels (cells draining at least `head_km2`), and Strahler's orders: the count of
/// streams of each order, the largest order, Horton's bifurcation ratio over the orders (the geometric mean of
/// `N_k / N_k+1`; Earth's branching networks read 3–5, Horton 1945, Strahler 1957) and the drainage density (km of
/// channel a km²). Returns the orders' counts, the ratio, the density, and each cell's order (0 off the channels).
fn branching(z: &[f32], w: usize, cell_m: f64, head_km2: f64) -> (Vec<usize>, f64, f64, Vec<u8>) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let h = z.len() / w;
    let n = w * h;
    let key = |v: f64| (v * 1_000.0).round() as i64;
    // The priority flood (Barnes et al. 2014) with a millimetre's rise across a flat, from the raster's edge and its holes.
    let mut f = vec![f64::NAN; n];
    let mut heap = BinaryHeap::new();
    let mut seen = vec![false; n];
    for y in 0..h {
        for x in 0..w {
            let k = y * w + x;
            if !z[k].is_finite() {
                seen[k] = true;
                continue;
            }
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                f[k] = f64::from(z[k]);
                seen[k] = true;
                heap.push(Reverse((key(f[k]), k)));
            }
        }
    }
    let nbr = [
        (-1i64, -1i64),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    while let Some(Reverse((_, k))) = heap.pop() {
        let (x, y) = ((k % w) as i64, (k / w) as i64);
        for (dx, dy) in nbr {
            let (a, b) = (x + dx, y + dy);
            if a < 0 || b < 0 || a >= w as i64 || b >= h as i64 {
                continue;
            }
            let m = b as usize * w + a as usize;
            if seen[m] {
                continue;
            }
            seen[m] = true;
            f[m] = f64::from(z[m]).max(f[k] + 1e-3);
            heap.push(Reverse((key(f[m]), m)));
        }
    }
    // Each cell's receiver: the steepest drop over its eight neighbours (none at the edge or with no drop).
    let mut recv = vec![usize::MAX; n];
    for y in 0..h {
        for x in 0..w {
            let k = y * w + x;
            if !f[k].is_finite() {
                continue;
            }
            let mut best = (0.0f64, usize::MAX);
            for (dx, dy) in nbr {
                let (a, b) = (x as i64 + dx, y as i64 + dy);
                if a < 0 || b < 0 || a >= w as i64 || b >= h as i64 {
                    continue;
                }
                let m = b as usize * w + a as usize;
                if !f[m].is_finite() {
                    continue;
                }
                let s = (f[k] - f[m]) / ((dx * dx + dy * dy) as f64).sqrt();
                if s > best.0 {
                    best = (s, m);
                }
            }
            recv[k] = best.1;
        }
    }
    // The areas, from the highest cell down, and the orders on the same walk.
    let mut by_height: Vec<usize> = (0..n).filter(|&k| f[k].is_finite()).collect();
    by_height.sort_by(|&a, &b| f[b].total_cmp(&f[a]));
    let cell_km2 = cell_m * cell_m / 1e6;
    let mut area = vec![cell_km2; n];
    let (mut top, mut top_count) = (vec![0u8; n], vec![0u8; n]);
    let mut order = vec![0u8; n];
    for &k in &by_height {
        if area[k] >= head_km2 {
            order[k] = match (top[k], top_count[k]) {
                (0, _) => 1,
                (o, c) if c >= 2 => o + 1,
                (o, _) => o,
            };
        }
        let r = recv[k];
        if r != usize::MAX {
            area[r] += area[k];
            if order[k] > top[r] {
                top[r] = order[k];
                top_count[r] = 1;
            } else if order[k] == top[r] && order[k] > 0 {
                top_count[r] += 1;
            }
        }
    }
    // A stream of order k ends where its water leaves the raster or enters a cell of another order.
    let max = order.iter().copied().max().unwrap_or(0) as usize;
    let mut count = vec![0usize; max + 1];
    let mut channel = 0usize;
    for k in 0..n {
        if order[k] == 0 {
            continue;
        }
        channel += 1;
        let r = recv[k];
        if r == usize::MAX || order[r] != order[k] {
            count[order[k] as usize] += 1;
        }
    }
    let counts = count[1..].to_vec();
    let rb = if max >= 2 && counts[max - 1] > 0 {
        (counts[0] as f64 / counts[max - 1] as f64).powf(1.0 / (max - 1) as f64)
    } else {
        f64::NAN
    };
    let land_km2 = by_height.len() as f64 * cell_km2;
    let density = channel as f64 * cell_m / 1_000.0 / land_km2.max(f64::MIN_POSITIVE);
    (counts, rb, density, order)
}

/// ★ THE DRAINAGE MAP: the raster from above, shaded by the same sun, its channels drawn by their order (light blue the
/// first order, deep blue the fourth and over).
fn drainage_map(z: &[f32], w: usize, cell_m: f64, order: &[u8], path: &str) {
    let h = z.len() / w;
    let mut img = image::RgbImage::new(w as u32, h as u32);
    let sun = {
        let (az, el) = (225f64.to_radians(), 35f64.to_radians());
        [el.cos() * az.sin(), -el.cos() * az.cos(), el.sin()]
    };
    let at = |x: usize, y: usize| f64::from(z[y.min(h - 1) * w + x.min(w - 1)]);
    for y in 0..h {
        for x in 0..w {
            let k = y * w + x;
            let c = if !z[k].is_finite() {
                [255.0, 255.0, 255.0]
            } else {
                let dzx = at(x + 1, y) - at(x.saturating_sub(1), y);
                let dzy = at(x, y + 1) - at(x, y.saturating_sub(1));
                let nrm = [-dzx, -dzy, 2.0 * cell_m];
                let l = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
                let lit = ((nrm[0] * sun[0] + nrm[1] * sun[1] + nrm[2] * sun[2]) / l).max(0.0);
                let g = 90.0 + 150.0 * lit;
                match order[k] {
                    0 => [g, g * 0.97, g * 0.9],
                    1 => [150.0, 200.0, 240.0],
                    2 => [90.0, 150.0, 230.0],
                    3 => [40.0, 90.0, 210.0],
                    _ => [10.0, 30.0, 150.0],
                }
            };
            img.put_pixel(
                x as u32,
                y as u32,
                image::Rgb(c.map(|v: f64| v.clamp(0.0, 255.0) as u8)),
            );
        }
    }
    img.save(path).expect("the map saves");
}

/// An SRTM 1″ tile averaged to ~128 m cells (`relief_scales`' rule), its middle `side × side` cells.
fn earth_crop(path: &str, side: usize) -> (Vec<f32>, f64) {
    let raw = std::fs::read(path).expect("the tile reads");
    let n = 3601usize;
    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let lat: f64 = name[1..3].parse::<f64>().expect("a latitude")
        * if name.starts_with('S') { -1.0 } else { 1.0 };
    let row_m = 30.87;
    let col_m = row_m * (lat + 0.5).to_radians().cos();
    let (ky, kx) = (4usize, (128.0 / col_m).round().max(1.0) as usize);
    let (w, h) = ((n - 1) / kx, (n - 1) / ky);
    let (x0, y0) = ((w - side.min(w)) / 2, (h - side.min(h)) / 2);
    let mut z = vec![f32::NAN; side * side];
    for by in 0..side.min(h) {
        for bx in 0..side.min(w) {
            let (mut s, mut ok) = (0.0f64, true);
            for y in (y0 + by) * ky..(y0 + by + 1) * ky {
                for x in (x0 + bx) * kx..(x0 + bx + 1) * kx {
                    let v = i16::from_be_bytes([raw[2 * (y * n + x)], raw[2 * (y * n + x) + 1]]);
                    ok &= v != i16::MIN;
                    s += f64::from(v);
                }
            }
            if ok {
                z[by * side + bx] = (s / (kx * ky) as f64) as f32;
            }
        }
    }
    (z, (row_m * ky as f64 + col_m * kx as f64) / 2.0)
}

/// Each cell's mean over the `(2r+1)²` box around it (finite cells only; `NaN` where the cell is not finite).
fn box_means(z: &[f32], w: usize, r: usize) -> Vec<f64> {
    let h = z.len() / w;
    let (mut s, mut c) = (
        vec![0.0f64; (w + 1) * (h + 1)],
        vec![0.0f64; (w + 1) * (h + 1)],
    );
    for y in 0..h {
        for x in 0..w {
            let v = z[y * w + x];
            let (vs, vc) = if v.is_finite() {
                (f64::from(v), 1.0)
            } else {
                (0.0, 0.0)
            };
            let at = (y + 1) * (w + 1) + x + 1;
            s[at] = vs + s[at - 1] + s[at - (w + 1)] - s[at - (w + 1) - 1];
            c[at] = vc + c[at - 1] + c[at - (w + 1)] - c[at - (w + 1) - 1];
        }
    }
    let boxed = |t: &[f64], x0: usize, y0: usize, x1: usize, y1: usize| {
        t[y1 * (w + 1) + x1] - t[y0 * (w + 1) + x1] - t[y1 * (w + 1) + x0] + t[y0 * (w + 1) + x0]
    };
    (0..w * h)
        .map(|k| {
            let (x, y) = (k % w, k / w);
            if !z[k].is_finite() {
                return f64::NAN;
            }
            let (x0, y0, x1, y1) = (
                x.saturating_sub(r),
                y.saturating_sub(r),
                (x + r + 1).min(w),
                (y + r + 1).min(h),
            );
            boxed(&s, x0, y0, x1, y1) / boxed(&c, x0, y0, x1, y1).max(1.0)
        })
        .collect()
}

/// ★ THE SMALL-SCALE SHARE: of the relief inside 5 km (the heights less their 5 km box mean), the share of its variance
/// that lives under ~640 m (the heights less their 640 m box mean). Earth's smooth hillslopes keep it low; ground cut into
/// small sharp ridges at every point raises it.
fn small_scale_share(z: &[f32], w: usize) -> f64 {
    let (m5, m640) = (box_means(z, w, 19), box_means(z, w, 2));
    let (mut big, mut small) = (0.0f64, 0.0f64);
    for k in 0..z.len() {
        if z[k].is_finite() && m5[k].is_finite() {
            big += (f64::from(z[k]) - m5[k]).powi(2);
            small += (f64::from(z[k]) - m640[k]).powi(2);
        }
    }
    small / big.max(f64::MIN_POSITIVE)
}

/// ★ A DIAGNOSTIC, NOT A CANDIDATE: the raster crept by linear diffusion (Culling 1960; `∂z/∂t = D·∇²z`) over `dt_m2` =
/// D·t square metres, with no rise and no rivers — only to see whether smooth hillslopes alone make the picture read as
/// Earth's. Explicit steps, each under the scheme's bound (D·dt/dx² ≤ 1/4).
fn crept(z: &[f32], w: usize, cell_m: f64, dt_m2: f64) -> Vec<f32> {
    let h = z.len() / w;
    let steps = (4.0 * dt_m2 / (cell_m * cell_m)).ceil().max(1.0) as usize;
    let a = dt_m2 / steps as f64 / (cell_m * cell_m);
    let mut cur: Vec<f64> = z.iter().map(|&v| f64::from(v)).collect();
    for _ in 0..steps {
        let mut next = cur.clone();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let k = y * w + x;
                let lap = cur[k - 1] + cur[k + 1] + cur[k - w] + cur[k + w] - 4.0 * cur[k];
                if lap.is_finite() {
                    next[k] = cur[k] + a * lap;
                }
            }
        }
        cur = next;
    }
    cur.iter().map(|&v| v as f32).collect()
}

/// ★ THE HYPSOMETRIC INTEGRAL (Strahler 1952) over `win × win` land windows: `(mean − lowest) / (highest − lowest)`, the
/// median. High: the ground stands near its windows' tops (rounded, convex hills cut by narrow valleys); low: it lies near
/// their floors (wide low ground under sharp crests).
fn hypsometric(z: &[f32], w: usize, win: usize) -> f64 {
    let h = z.len() / w;
    let mut out = Vec::new();
    for wy in 0..h / win {
        for wx in 0..w / win {
            let (mut lo, mut hi, mut sum, mut land) = (f64::MAX, f64::MIN, 0.0f64, true);
            for y in wy * win..(wy + 1) * win {
                for x in wx * win..(wx + 1) * win {
                    let v = f64::from(z[y * w + x]);
                    land &= v.is_finite() && v > 0.0;
                    lo = lo.min(v);
                    hi = hi.max(v);
                    sum += v;
                }
            }
            if land && hi - lo > 1.0 {
                out.push((sum / (win * win) as f64 - lo) / (hi - lo));
            }
        }
    }
    out.sort_by(f64::total_cmp);
    out.get(out.len() / 2).copied().unwrap_or(f64::NAN)
}

// ★ THE FORWARD BENCH (`evolve`, 2026-10-13; the owner: "Please continue" after the form bench): the same joint law run
// FORWARD IN TIME — the Fastscape method (Braun & Willett 2013, Geomorphology 180: each step the rise, the drainage of
// the moment, and the stream law solved implicitly from the outlets up, Newton's method for n ≠ 1) — with and without
// the hillslope creep law (Culling 1960, linear, `∂z/∂t = D·∇²z`), from an old flat surface raised at today's rises. The
// pictures show the ground on its way to the balance today's level solves at once. Bench only.

/// The creep's rate (m²/yr): a mid-range Earth value (Fernandes & Dietrich 1997, 0.0036–0.01; Roering et al. 1999, the
/// Oregon Coast Range, 0.003; Hurst et al. 2012, the Feather River, 0.0086). Bench only.
const CREEP_D_M2_YR: f64 = 0.005;

/// The shares of the way to the balance at which the forward run is drawn.
const SNAPSHOTS: [f64; 4] = [0.25, 0.5, 0.75, 0.9];

/// Each link's cotangent weight `(cot α + cot β) / 2` in `graph.adj`'s order (Pinkall & Polthier 1993; with the mixed
/// Voronoi areas, the Laplacian of Meyer et al. 2003). A hull link of one obtuse triangle reads a negative weight; the
/// bench takes it as nought (no flux across that link).
fn cotan_weights(d: &Domain, radius_m: f64) -> Vec<f64> {
    let s = radius_m / (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let pos: Vec<[f64; 3]> = d
        .dirs
        .iter()
        .map(|v| {
            [
                v[0].raw() as f64 * s,
                v[1].raw() as f64 * s,
                v[2].raw() as f64 * s,
            ]
        })
        .collect();
    let g = &d.graph;
    let slot = |a: u32, b: u32| -> usize {
        let (lo, hi) = (
            g.start[a as usize] as usize,
            g.start[a as usize + 1] as usize,
        );
        lo + g.adj[lo..hi]
            .iter()
            .position(|&m| m == b)
            .expect("a triangle's side is a link")
    };
    let mut w = vec![0.0f64; g.adj.len()];
    for t in &g.tris {
        for k in 0..3 {
            let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
            let (pa, pb, pc) = (pos[a as usize], pos[b as usize], pos[c as usize]);
            let (u, v) = (
                [pa[0] - pc[0], pa[1] - pc[1], pa[2] - pc[2]],
                [pb[0] - pc[0], pb[1] - pc[1], pb[2] - pc[2]],
            );
            let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            let cr = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let len = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
            if len > 0.0 {
                let half_cot = 0.5 * dot / len;
                w[slot(a, b)] += half_cot;
                w[slot(b, a)] += half_cot;
            }
        }
    }
    for x in &mut w {
        *x = x.max(0.0);
    }
    w
}

/// ★ ONE FORWARD STEP of `dt` years: the rise, the drainage, the implicit stream law (the joint law `E = K·Q·S^n`, held
/// under the 35° repose as the steady level is), and, with `creep`, the creep law in explicit sub-steps under the
/// scheme's bound.
fn forward_step(d: &Domain, up: &[f64], weights: &[f64], h: &mut [f64], dt: f64, creep: bool) {
    let n = h.len();
    for p in 0..n {
        if !d.outlet[p] {
            h[p] += up[p] * dt;
        }
    }
    let hg: Vec<Gf> = h.iter().map(|&v| Gf::from_f64(v)).collect();
    let t = tree(&hg, &d.outlet, &d.graph, &d.geom.len);
    let q = accumulate(&t, &d.water);
    let tan = vd_terrain::body::TAN_REPOSE;
    for &c in &t.order {
        let c = c as usize;
        if d.outlet[c] {
            h[c] = d.base[c].to_f64();
            continue;
        }
        let r = t.recv[c];
        if r == NONE {
            continue;
        }
        let (hr, h0, l) = (h[r as usize], h[c], t.link[c].to_f64());
        if h0 <= hr {
            continue;
        }
        let f = JOINT_K_LP * q[c].to_f64() * dt;
        let mut x = h0;
        for _ in 0..60 {
            let s = (x - hr) / l;
            let g = x - h0 + f * s.powf(JOINT_N);
            let gp = 1.0 + f * JOINT_N * s.powf(JOINT_N - 1.0) / l;
            let dx = g / gp;
            x = (x - dx).max(hr + 1e-9);
            if dx.abs() < 1e-6 {
                break;
            }
        }
        h[c] = x.min(hr + tan * l);
    }
    if !creep {
        return;
    }
    // The creep IMPLICIT (backward Euler, unconditionally stable): `(A_i + a·Σw_ij)·x_i − a·Σ w_ij·x_j = A_i·h_i` with
    // `a = D·dt`, the outlets held, solved by Gauss–Seidel sweeps until no point moves a millimetre (at most 200).
    let g = &d.graph;
    let a = CREEP_D_M2_YR * dt;
    let old = h.to_vec();
    for _ in 0..200 {
        let mut moved = 0.0f64;
        for p in 0..n {
            if d.outlet[p] {
                continue;
            }
            let (lo, hi) = (g.start[p] as usize, g.start[p + 1] as usize);
            let (mut sw, mut swh) = (0.0, 0.0);
            for (&m, &w) in g.adj[lo..hi].iter().zip(&weights[lo..hi]) {
                sw += w;
                swh += w * h[m as usize];
            }
            let area = d.geom.area[p].to_f64();
            let x = (area * old[p] + a * swh) / (area + a * sw);
            moved = moved.max((x - h[p]).abs());
            h[p] = x;
        }
        if moved < 1e-3 {
            break;
        }
    }
}

/// ★ THE FORWARD RUN: from the old flat surface (every point at the lowest river exit's level, the outlets at their own),
/// raised at today's rises, forward until the ground's mean rise reaches 97 % of today's balance (or six response times);
/// drawn and read at each share of [`SNAPSHOTS`].
#[allow(clippy::too_many_arguments)]
fn forward_run(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    key: TileKey,
    d: &Domain,
    u: &[Gf],
    h_today: &[Gf],
    creep: bool,
    even: bool,
    out: &str,
) {
    let n = d.base.len();
    let mut up: Vec<f64> = point_rises(d, u).iter().map(|x| x.to_f64()).collect();
    if even {
        // ONE RISE for the whole domain: the mean of today's over its points (no node's own rise).
        let (sum, count) = (0..n)
            .filter(|&p| !d.outlet[p])
            .fold((0.0, 0usize), |(s, c), p| (s + up[p], c + 1));
        let one = sum / count.max(1) as f64;
        up.iter_mut().for_each(|x| *x = one);
    }
    let weights = cotan_weights(d, body.radius_m());
    let low = (0..n)
        .filter(|&p| d.outlet[p] && d.base[p] > Gf::ZERO)
        .map(|p| d.base[p].to_f64())
        .fold(f64::INFINITY, f64::min);
    let low = if low.is_finite() { low } else { 0.0 };
    let mut h: Vec<f64> = (0..n)
        .map(|p| if d.outlet[p] { d.base[p].to_f64() } else { low })
        .collect();
    let inner: Vec<usize> = (0..n).filter(|&p| !d.outlet[p]).collect();
    let mean =
        |v: &dyn Fn(usize) -> f64| inner.iter().map(|&p| v(p)).sum::<f64>() / inner.len() as f64;
    let target = mean(&|p| h_today[p].to_f64() - low);
    let mean_up = mean(&|p| up[p]);
    let tau = target / mean_up;
    let dt = tau / 150.0;
    let name = match (even, creep) {
        (false, false) => "F_stream",
        (false, true) => "F_creep",
        (true, false) => "F_even_stream",
        (true, true) => "F_even_creep",
    };
    println!(
        "{name}: the start {low:.0} m, today's mean rise over it {target:.0} m, the mean rise rate {:.1} m/Myr, the response time {:.2} Myr, a step {:.1} kyr",
        mean_up * 1e6,
        tau / 1e6,
        dt / 1e3
    );
    let started = std::time::Instant::now();
    let mut next = 0usize;
    for step in 1..=900 {
        forward_step(d, &up, &weights, &mut h, dt, creep);
        let f = mean(&|p| h[p] - low) / target;
        let last = f >= 0.97 || step == 900;
        if (next < SNAPSHOTS.len() && f >= SNAPSHOTS[next]) || last {
            let label = if last && next >= SNAPSHOTS.len() {
                format!("{name}_end")
            } else {
                format!(
                    "{name}_{:.0}",
                    SNAPSHOTS[next.min(SNAPSHOTS.len() - 1)] * 100.0
                )
            };
            println!(
                "  {label}: {:.2} Myr ({step} steps, {:.0} s), {:.0} % of the way to the balance",
                step as f64 * dt / 1e6,
                started.elapsed().as_secs_f64(),
                f * 100.0
            );
            let hg: Vec<Gf> = h.iter().map(|&v| Gf::from_f64(v)).collect();
            if let Some((z, side)) = raster_of(body, lattice, key, d, &hg) {
                read_form(&label, &z, side, 128.0, out);
            }
            next += 1;
            if last {
                break;
            }
        }
    }
}

/// One raster's readings and pictures under `name`.
fn read_form(name: &str, z: &[f32], w: usize, cell_m: f64, out: &str) {
    let (mut r1, mut r5) = (reliefs(z, w, 8), reliefs(z, w, 39));
    summary(&format!("{name}, 1 km"), &mut r1);
    summary(&format!("{name}, 5 km"), &mut r5);
    let (rows, axis) = rows_number(z, w);
    let (counts, rb, density, order) = branching(z, w, cell_m, HEAD_KM2);
    println!(
        "  {name:<40}: HYPSOMETRIC 2 km {:.2} 5 km {:.2} | SMALL-SCALE SHARE {:.3} | ROWS {rows:.3} (axis {axis:>4.0}°) | BRANCHING orders 1.. {counts:?}, highest {}, bifurcation {rb:.2}, density {density:.2} km/km²",
        hypsometric(z, w, 16),
        hypsometric(z, w, 39),
        small_scale_share(z, w),
        counts.len()
    );
    render(z, w, cell_m, &format!("{out}/{name}_side.png"));
    drainage_map(z, w, cell_m, &order, &format!("{out}/{name}_drainage.png"));
}

/// The channel head's area for the branching (half a km²: about 31 cells of 128 m).
const HEAD_KM2: f64 = 0.5;

/// ★ A WINDOW BERTH AT A TILE'S MIDDLE (2026-10-13; the owner: "Yes, please" to a window look on hills): the hull's
/// centre over the tile's middle node, upright, its belly a metre over the highest 128 m ground near the middle (the
/// game's tile solve; the shard's stand then puts it on its own footprint's ground). The nose: with a star direction,
/// 120° off the star along the ground (the picture gate's "over the shoulder"); without, to the planet's north pole.
fn print_berth(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    key: TileKey,
    star: Option<vd_core::glam::DVec3>,
) {
    use vd_core::glam::{DMat3, DQuat, DVec3};
    let half = TILE_NODES / 2;
    let node = lattice.index(
        key.face,
        key.tx * TILE_NODES + half,
        key.ty * TILE_NODES + half,
    );
    let g = lattice.direction(node);
    let s = (1u64 << 40) as f64;
    let up = DVec3::new(
        g[0].raw() as f64 / s,
        g[1].raw() as f64 / s,
        g[2].raw() as f64 / s,
    )
    .normalize();
    let level = match star {
        Some(sun) => {
            let flat = (sun - up * up.dot(sun)).normalize();
            let side = up.cross(flat).normalize();
            let off = 120f64.to_radians();
            (flat * off.cos() + side * off.sin()).normalize()
        }
        None => (DVec3::Z - up * up.z).normalize(),
    };
    if let Some(sun) = star {
        println!(
            "the star {:.1}° up at the middle node",
            up.dot(sun).clamp(-1.0, 1.0).asin().to_degrees()
        );
    }
    let right = level.cross(up).normalize();
    let orient = DQuat::from_mat3(&DMat3::from_cols(right, up, -level)).normalize();
    let l1 = solve_l1(body, lattice, artifact, key).expect("the tile solves");
    // The raster's cells around the middle node's centre (the raster starts half a buffer before the tile; a node is
    // `1 << shift` cells, its centre half of that in): the highest of the four nearest.
    let (w, r) = (l1.side, body.radius_m());
    let cells = 1usize << cells_per_node_shift(body, lattice, L1_RUNG).expect("nests");
    let mid = ((BUFFER_NODES >> 1) + half) as usize * cells + cells / 2;
    let mut highest = f64::MIN;
    for y in mid - 1..mid + 1 {
        for x in mid - 1..mid + 1 {
            highest = highest.max(f64::from(l1.quarters[y * w + x]) / 4.0);
        }
    }
    let sea = f64::from(artifact.sea_m);
    let centre = up * (r + sea + highest + 3.0 + 1.0);
    println!(
        "tile {key:?}: the middle node {node}, the ground there up to {highest:.1} m over the sea"
    );
    println!(
        "SPAWN 1000=Planet({}):{},{},{}@{},{},{},{}",
        body.seed(),
        centre.x,
        centre.y,
        centre.z,
        orient.x,
        orient.y,
        orient.z,
        orient.w
    );
    println!(
        "BERTH --berth-x-m {} --berth-y-m {} --berth-z-m {} --berth-facing-x {} --berth-facing-y {} --berth-facing-z {} --berth-facing-w {}",
        centre.x, centre.y, centre.z, orient.x, orient.y, orient.z, orient.w
    );
}

/// Each whole land tile (all 64 nodes inland land) with its nodes' median 25 km relief.
fn tile_medians(rel: &[Option<f32>], lattice: &MacroLattice) -> Vec<((u8, i32, i32), f32)> {
    let mut by_tile: std::collections::BTreeMap<(u8, i32, i32), Vec<f32>> =
        std::collections::BTreeMap::new();
    for (n, r) in rel.iter().enumerate() {
        if let Some(r) = r {
            let (face, i, j) = lattice.split(n as u32);
            by_tile
                .entry((
                    face.index(),
                    i.div_euclid(TILE_NODES),
                    j.div_euclid(TILE_NODES),
                ))
                .or_default()
                .push(*r);
        }
    }
    let full = (TILE_NODES * TILE_NODES) as usize;
    by_tile
        .into_iter()
        .filter(|(_, v)| v.len() == full)
        .map(|(k, mut v)| {
            v.sort_by(f32::total_cmp);
            (k, v[v.len() / 2])
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let planet: u64 = args.get(1).expect("a planet seed").parse().expect("a seed");
    let (body, charter, system) = find(planet);
    let lattice = body.macro_lattice().expect("a lattice");
    println!(
        "Planet({planet}) in {system:?}: radius {:.0} km, {} nodes of the 8 km lattice, {} an edge",
        body.radius_m() / 1_000.0,
        lattice.node_count(),
        lattice.edge
    );
    match args.first().map(String::as_str) {
        Some("solve") => {
            let out = args.get(2).expect("OUT.art");
            let cores: usize = args.get(3).map_or(8, |s| s.parse().expect("cores"));
            let words = vd_bins::solve_words_of(&charter);
            let started = std::time::Instant::now();
            let artifact = vd_bins::artifact_worker::run_solve_on(
                &vd_bins::artifact_worker::SolveJob { body, words },
                vd_terrain::par::Cores(cores),
            )
            .expect("the planet solves");
            vd_bins::artifact_file::write(out, &artifact).expect("the artifact writes");
            let land = artifact
                .rows
                .iter()
                .filter(|r| r.z_m > artifact.sea_m)
                .count();
            println!(
                "solved in {:.1} s on {cores} cores: the sea {} m, land {:.1} % of the nodes; written to {out}",
                started.elapsed().as_secs_f64(),
                artifact.sea_m,
                100.0 * land as f64 / artifact.rows.len() as f64
            );
            let mut r: Vec<f32> = relief25(&artifact, &lattice)
                .into_iter()
                .flatten()
                .collect();
            summary("the whole land, 3 × 3 nodes (~25 km)", &mut r);
        }
        Some("stands") => {
            let world = args.get(2).expect("WORLD.art");
            let out = args.get(3).expect("OUT_DIR");
            std::fs::create_dir_all(out).expect("the output directory");
            let artifact = vd_bins::artifact_file::read(world).expect("the world reads");
            let rel = relief25(&artifact, &lattice);
            let mut all: Vec<f32> = rel.iter().flatten().copied().collect();
            summary("the whole land, 3 × 3 nodes (~25 km)", &mut all);
            // Each whole tile's nodes' 25 km windows (a tile counts when every one of its 64 nodes is inland land).
            let mut by_tile: std::collections::BTreeMap<(u8, i32, i32), Vec<f32>> =
                std::collections::BTreeMap::new();
            for (n, r) in rel.iter().enumerate() {
                if let Some(r) = r {
                    let (face, i, j) = lattice.split(n as u32);
                    by_tile
                        .entry((
                            face.index(),
                            i.div_euclid(TILE_NODES),
                            j.div_euclid(TILE_NODES),
                        ))
                        .or_default()
                        .push(*r);
                }
            }
            let full = (TILE_NODES * TILE_NODES) as usize;
            let medians: Vec<((u8, i32, i32), f32)> = by_tile
                .into_iter()
                .filter(|(_, v)| v.len() == full)
                .map(|(k, mut v)| {
                    v.sort_by(f32::total_cmp);
                    (k, v[v.len() / 2])
                })
                .collect();
            println!("  {} whole land tiles", medians.len());
            for (name, target) in [("plain", 46.0f32), ("hills", 175.0), ("belt", 1_500.0)] {
                // The whole land tile whose nodes' median 25 km window reads nearest the target.
                let Some(&((f, tx, ty), r)) = medians
                    .iter()
                    .min_by(|a, b| (a.1 - target).abs().total_cmp(&(b.1 - target).abs()))
                else {
                    println!("  {name}: no whole land tile");
                    continue;
                };
                let key = TileKey {
                    face: vd_seed::bend::Face::from_index(f).expect("a face"),
                    tx,
                    ty,
                };
                let started = std::time::Instant::now();
                let Some(tile) = solve_l1(&body, &lattice, &artifact, key) else {
                    println!("  {name}: the tile {key:?} does not solve");
                    continue;
                };
                let z: Vec<f32> = tile.quarters.iter().map(|&q| q as f32 / 4.0).collect();
                let w = tile.side;
                let (mut r1, mut r5) = (reliefs(&z, w, 8), reliefs(&z, w, 39));
                println!(
                    "{name}: tile {key:?} (its nodes' median 25 km window {r:.0} m), solved in {:.1} s",
                    started.elapsed().as_secs_f64()
                );
                summary(&format!("{name}, 1 km"), &mut r1);
                summary(&format!("{name}, 5 km"), &mut r5);
                render(&z, w, 128.0, &format!("{out}/{name}.png"));
            }
            println!("pictures in {out}");
        }
        Some("forms") => {
            let world = args.get(2).expect("WORLD.art");
            let tile: Vec<i32> = args
                .get(3)
                .expect("F,TX,TY")
                .split(',')
                .map(|x| x.trim().parse().expect("an integer"))
                .collect();
            let out = args.get(4).expect("OUT_DIR");
            std::fs::create_dir_all(out).expect("the output directory");
            let artifact = vd_bins::artifact_file::read(world).expect("the world reads");
            let key = TileKey {
                face: vd_seed::bend::Face::from_index(tile[0] as u8).expect("a face"),
                tx: tile[1],
                ty: tile[2],
            };
            println!(
                "THE FORM BENCH: tile {key:?}; the channel head {HEAD_KM2} km²; each raster the tile and half its buffer"
            );
            // The control: the game's own solve, and the bench's copy with every switch off.
            let started = std::time::Instant::now();
            let game = solve_l1(&body, &lattice, &artifact, key).expect("the tile solves");
            let game_z: Vec<f32> = game.quarters.iter().map(|&q| q as f32 / 4.0).collect();
            println!(
                "the game's solve_l1: {} points, the fine rounds {} ({:.1} s)",
                game.points,
                game.passes.last().copied().unwrap_or(0),
                started.elapsed().as_secs_f64()
            );
            for v in VARIANTS {
                let started = std::time::Instant::now();
                let Some((z, side, rounds, fit)) =
                    solve_variant(&body, &lattice, &artifact, key, v)
                else {
                    println!("{}: the domain does not build", v.name);
                    continue;
                };
                println!(
                    "{}: the fine rounds {rounds}{}, the held nodes' mean off their row: largest {:.0} m, mean {:.0} m ({:.1} s)",
                    v.name,
                    if rounds == FINE_ROUND_BOUND {
                        " (the bound: never settled)"
                    } else {
                        ""
                    },
                    fit.0,
                    fit.1,
                    started.elapsed().as_secs_f64()
                );
                if v.name == "A_today" {
                    for dt in [10_000.0f64, 40_000.0] {
                        let c = crept(&z, side, 128.0, dt);
                        read_form(&format!("A_crept_{dt:.0}"), &c, side, 128.0, out);
                    }
                    let same = z.len() == game_z.len()
                        && z.iter()
                            .zip(&game_z)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                    println!("  THE COPY IS EXACT (A against the game's solve_l1): {same}");
                }
                read_form(v.name, &z, side, 128.0, out);
            }
            for path in &args[5..] {
                let name = std::path::Path::new(path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("earth");
                let (z, cell_m) = earth_crop(path, game.side);
                read_form(&format!("Earth_{name}"), &z, game.side, cell_m, out);
            }
            println!("pictures in {out}");
        }
        Some("evolve") => {
            let world = args.get(2).expect("WORLD.art");
            let tile: Vec<i32> = args
                .get(3)
                .expect("F,TX,TY")
                .split(',')
                .map(|x| x.trim().parse().expect("an integer"))
                .collect();
            let out = args.get(4).expect("OUT_DIR");
            std::fs::create_dir_all(out).expect("the output directory");
            let artifact = vd_bins::artifact_file::read(world).expect("the world reads");
            let key = TileKey {
                face: vd_seed::bend::Face::from_index(tile[0] as u8).expect("a face"),
                tx: tile[1],
                ty: tile[2],
            };
            println!("THE FORWARD BENCH: tile {key:?}; the creep {CREEP_D_M2_YR} m²/yr");
            let (fine, u, h, rounds, _) =
                solve_points(&body, &lattice, &artifact, key, VARIANTS[0])
                    .expect("the tile solves");
            println!(
                "today's balance: {} points, the fine rounds {rounds}",
                fine.dirs.len()
            );
            // `nodes`: today's rises, node by node; `even`: one rise for the whole domain; both by default.
            let rises = args.get(5).map_or("both", String::as_str);
            for even in [false, true] {
                if (rises == "nodes" && even) || (rises == "even" && !even) {
                    continue;
                }
                for creep in [false, true] {
                    forward_run(&body, &lattice, key, &fine, &u, &h, creep, even, out);
                }
            }
            println!("pictures in {out}");
        }
        Some("berth") => {
            let world = args.get(2).expect("WORLD.art");
            let tile: Vec<i32> = args
                .get(3)
                .expect("F,TX,TY")
                .split(',')
                .map(|x| x.trim().parse().expect("an integer"))
                .collect();
            let artifact = vd_bins::artifact_file::read(world).expect("the world reads");
            let digest = artifact.digest();
            println!("the world's digest [{:#x}, {:#x}]", digest[0], digest[1]);
            let key = TileKey {
                face: vd_seed::bend::Face::from_index(tile[0] as u8).expect("a face"),
                tx: tile[1],
                ty: tile[2],
            };
            print_berth(&body, &lattice, &artifact, key, None);
        }
        Some("sunlit") => {
            // ★ HILLS IN DAYLIGHT (2026-10-13; the owner woke on the dark side): no body spins (the picture gate's
            // reading), so the star's direction in the planet's frame is minus the planet's place on its orbit; the
            // whole land tile with the star 25–60° up over its middle whose nodes' median 25 km relief reads nearest
            // `TARGET` metres (Germany's 175 m by default), berthed as `berth` does with the star over the shoulder.
            use vd_core::glam::DVec3;
            let world = args.get(2).expect("WORLD.art");
            let target: f32 = args.get(3).map_or(175.0, |s| s.parse().expect("metres"));
            let artifact = vd_bins::artifact_file::read(world).expect("the world reads");
            let digest = artifact.digest();
            println!("the world's digest [{:#x}, {:#x}]", digest[0], digest[1]);
            let orbit = vd_bins::home_orbit(UNIVERSE).expect("the home planet's orbit");
            let sun = -vd_physics::celestial::orbital_state(&orbit, 0.0)
                .position
                .normalize();
            let rel = relief25(&artifact, &lattice);
            let s40 = (1u64 << 40) as f64;
            let lit: Vec<((u8, i32, i32), f32)> = tile_medians(&rel, &lattice)
                .into_iter()
                .filter(|&((f, tx, ty), _)| {
                    let face = vd_seed::bend::Face::from_index(f).expect("a face");
                    let g = lattice.direction(lattice.index(
                        face,
                        tx * TILE_NODES + TILE_NODES / 2,
                        ty * TILE_NODES + TILE_NODES / 2,
                    ));
                    let d = DVec3::new(
                        g[0].raw() as f64 / s40,
                        g[1].raw() as f64 / s40,
                        g[2].raw() as f64 / s40,
                    )
                    .normalize();
                    let up_deg = d.dot(sun).clamp(-1.0, 1.0).asin().to_degrees();
                    (25.0..=60.0).contains(&up_deg)
                })
                .collect();
            println!("{} whole land tiles with the star 25–60° up", lit.len());
            let &((f, tx, ty), r) = lit
                .iter()
                .min_by(|a, b| (a.1 - target).abs().total_cmp(&(b.1 - target).abs()))
                .expect("a lit tile");
            println!("the pick: its nodes' median 25 km relief {r:.0} m (target {target:.0} m)");
            let key = TileKey {
                face: vd_seed::bend::Face::from_index(f).expect("a face"),
                tx,
                ty,
            };
            print_berth(&body, &lattice, &artifact, key, Some(sun));
        }
        _ => panic!(
            "solve PLANET OUT.art [cores] | stands PLANET WORLD.art OUT_DIR | forms PLANET WORLD.art F,TX,TY OUT_DIR [EARTH.hgt…]"
        ),
    }
}
