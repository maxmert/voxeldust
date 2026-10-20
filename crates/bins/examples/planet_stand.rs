//! ★ ANY PLANET AS A STAND (2026-10-13; the owner chose the Earth-like search's #3, `Planet(18348864443750092586)`:
//! "Let's go to #3 first"). Bench only; THE world (SL5) through the generator's own functions, no variant:
//!
//! - `solve PLANET OUT.art [cores]`: the planet's star system found among the systems nearest home, its charter and body
//!   read as its shard reads them, its 8 km world solved by the game's own first solve (`artifact_worker::run_solve_on`,
//!   the same bytes on any number of cores, ruling B64), written to `OUT.art`, and its 25 km relief printed;
//! - `stands PLANET WORLD.art OUT_DIR`: three stands chosen by the world's own 25 km relief — the land node whose window
//!   reads nearest Kansas's 46 m (a plain), Germany's 175 m (hills) and 1 500 m (a belt) — each one's 128 m tile solved by
//!   the game's tile solve (`l1::solve_l1`), read by `relief_scales`' rule (the relief at 1 km and 5 km over land
//!   windows) and drawn from the side by its camera (2 km up, 8° down, 60° wide);
//! - `maps PLANET WORLD.art OUT_DIR TAG [LAT0 LAT1 LON0 LON1]`: the world as the rain bench's three maps (relief, rain,
//!   rivers; optionally only a window between two latitudes and two east longitudes) and its readings;
//! - `moon-with PLANET [cores]`: a bench reading, never a world — the home moon's small lattice solved by the game's
//!   worker with PLANET's words, and its temperature rows read;
//! - `diff PLANET A.art B.art`: two worlds of one body, the nodes that differ in each row field;
//! - `lakes PLANET WORLD.art`: G-OUTLETS — every lake closed (no node passes water on) or spilling, with their areas;
//! - `water1 PLANET WORLD.art F,TX,TY OUT_DIR`: B99 step 4's slice W1 on the bench — the 128 m tile under today's outlets
//!   and under each standing water's (B105 Q1 (b)), each filled by the same first water map, the readings and side views;
//! - `words PLANET`: the solve's words as the shard reads them from the charter;
//! - `store-get PLANET OUT.art` / `store-put PLANET IN.art` (the store file named by `VD_REALM_STORE`): a stored world out
//!   to a file under the key its own head names, or a solved world into the store under this build's key by the shard's
//!   own writer and read back as the shard reads it.
//!
//! `cargo run --release -p vd-bins --example planet_stand -- solve 18348864443750092586 /tmp/p3.art 8`

use std::collections::{BTreeMap, BTreeSet};

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
/// The sea's word a book is made with when no world is read (the codec's check): nought metres, a sea that applies.
fn artifact_sea(_body: &vd_terrain::BodyDefinition) -> i16 {
    0
}

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

/// ★ A VIEW FROM ANY EYE: from the raster cell `(ex, ey)`, `eye_m` over the ground, looking along `heading_deg` (0 up
/// the rows, 90 along them) and `pitch_deg` down, 60° wide; the same sun and haze as [`render`]. For a walker's eye and a
/// low flight over the 16 m ground beside the 128 m curve.
#[allow(clippy::too_many_arguments)]
fn render_view(
    z: &[f32],
    w: usize,
    cell_m: f64,
    (ex, ey): (f64, f64),
    eye_m: f64,
    heading_deg: f64,
    pitch_deg: f64,
    path: &str,
) {
    render_view_water(
        z,
        None,
        w,
        cell_m,
        (ex, ey),
        eye_m,
        heading_deg,
        pitch_deg,
        path,
    );
}

/// [`render_view`] with an optional WATER surface (`water`: each cell's water level over the sea, NaN where dry): the
/// ray meets the higher of the smooth ground and the water, and a water hit is drawn as still water (the sky's colour
/// over a dark blue, by the angle) — the bench's first look at the fill, no wave, no clarity. THE SHORE BY THE COLUMN
/// RULE (the one-volume revision §3.4): a point takes the level of its wet corner samples and is wet where the smooth
/// (bilinear) ground stands under it, so the shore follows the ground's own crossing of the level, never a cell's square.
#[allow(clippy::too_many_arguments)]
fn render_view_water(
    z: &[f32],
    water: Option<&[f32]>,
    w: usize,
    cell_m: f64,
    (ex, ey): (f64, f64),
    eye_m: f64,
    heading_deg: f64,
    pitch_deg: f64,
    path: &str,
) {
    let h = z.len() / w;
    // The water's level at a point: the highest level among its four corner samples that hold water.
    let level_at = |x: f64, y: f64| -> Option<f64> {
        let wl = water?;
        let (cx, cy) = (x / cell_m - 0.5, y / cell_m - 0.5);
        if cx < 0.0 || cy < 0.0 || cx >= (w - 1) as f64 || cy >= (h - 1) as f64 {
            return None;
        }
        let (ix, iy) = (cx as usize, cy as usize);
        [(ix, iy), (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1)]
            .into_iter()
            .map(|(a, b)| b * w + a)
            .filter(|&k| wl[k].is_finite() && wl[k] > z[k])
            .map(|k| f64::from(wl[k]))
            .reduce(f64::max)
    };
    let (wpx, hpx) = (960u32, 540u32);
    let fov = 60f64.to_radians();
    let ground = |x: f64, y: f64| -> Option<f64> {
        let (cx, cy) = (x / cell_m - 0.5, y / cell_m - 0.5);
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
        g.is_finite().then_some(g)
    };
    let (exm, eym) = (ex * cell_m, ey * cell_m);
    let eye_z = ground(exm, eym).unwrap_or(0.0) + eye_m;
    let (hd, pitch) = (heading_deg.to_radians(), -pitch_deg.to_radians());
    let sun = {
        let (az, el) = (225f64.to_radians(), 25f64.to_radians());
        [el.cos() * az.sin(), -el.cos() * az.cos(), el.sin()]
    };
    let far = ((w.max(h)) as f64 * cell_m * 1.5).min(60_000.0);
    let mut img = image::RgbImage::new(wpx, hpx);
    for py in 0..hpx {
        for px in 0..wpx {
            let ax = ((f64::from(px) + 0.5) / f64::from(wpx) - 0.5) * fov + hd;
            let ay = (0.5 - (f64::from(py) + 0.5) / f64::from(hpx)) * fov * f64::from(hpx)
                / f64::from(wpx)
                + pitch;
            let d = [ax.sin() * ay.cos(), -ax.cos() * ay.cos(), ay.sin()];
            let (mut t, mut hit) = (0.0f64, None);
            while t < far {
                let (x, y, zz) = (exm + d[0] * t, eym + d[1] * t, eye_z + d[2] * t);
                match ground(x, y) {
                    None => break,
                    Some(gz) => {
                        let top = level_at(x, y).map_or(gz, |l| l.max(gz));
                        if zz <= top {
                            hit = Some((x, y, gz, t, top > gz));
                            break;
                        }
                    }
                }
                t += (cell_m * 0.2).max(0.5) * (1.0 + t / (20.0 * eye_m.max(5.0)));
            }
            let sky = [150.0, 180.0, 215.0];
            let c = match hit {
                None => sky,
                Some((_, _, _, t, true)) => {
                    // Still water: the sky's colour by the angle of view over a dark blue.
                    let grazing = (-d[2]).clamp(0.0, 1.0);
                    let deep = [30.0, 60.0, 85.0];
                    let mix = 0.35 + 0.5 * (1.0 - grazing);
                    let haze = 1.0 - (-t / 45_000.0).exp();
                    [0, 1, 2].map(|k| {
                        (deep[k] * (1.0 - mix) + sky[k] * mix) * (1.0 - haze) + sky[k] * haze
                    })
                }
                Some((x, y, gz, t, false)) => {
                    let e = cell_m * 0.5;
                    let dzx = ground(x + e, y).unwrap_or(gz) - ground(x - e, y).unwrap_or(gz);
                    let dzy = ground(x, y + e).unwrap_or(gz) - ground(x, y - e).unwrap_or(gz);
                    let n = [-dzx, -dzy, 2.0 * e];
                    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    let lit = ((n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]) / l).max(0.0);
                    let base = [170.0, 160.0, 140.0];
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

/// The relief census of a 16 m raster: the share of its links over 35°, 45° and 63° (each cell with its east and south
/// neighbour), and the steepest.
fn steep_census(name: &str, z: &[f32], w: usize, cell_m: f64) {
    let h = z.len() / w;
    let (mut n, mut over) = (0usize, [0usize; 3]);
    let mut steepest = 0.0f64;
    let bounds = [35f64, 45.0, 63.0].map(|d| d.to_radians().tan());
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let a = z[y * w + x];
            for b in [z[y * w + x + 1], z[(y + 1) * w + x]] {
                if a.is_finite() && b.is_finite() {
                    let s = f64::from((a - b).abs()) / cell_m;
                    n += 1;
                    steepest = steepest.max(s);
                    for (k, &bd) in bounds.iter().enumerate() {
                        if s > bd {
                            over[k] += 1;
                        }
                    }
                }
            }
        }
    }
    let pc = |k: usize| 100.0 * over[k] as f64 / n.max(1) as f64;
    println!(
        "  {name:<40}: links over 35° {:.2} %, 45° {:.3} %, 63° {:.4} %, the steepest {:.1}°",
        pc(0),
        pc(1),
        pc(2),
        steepest.atan().to_degrees()
    );
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

/// The rain bench's rain scale (`climate_bench`'s own): a logarithmic ramp from 10 mm/yr (pale sand) through 250 and
/// 1 000 to 4 000 mm/yr (deep blue); the sea grey-blue.
fn rain_colour(mm_yr: f64) -> [u8; 3] {
    if mm_yr.is_nan() {
        return [70, 80, 96];
    }
    const STOPS: [(f64, [f64; 3]); 6] = [
        (10.0, [244.0, 236.0, 214.0]),
        (100.0, [222.0, 196.0, 140.0]),
        (250.0, [190.0, 200.0, 120.0]),
        (600.0, [110.0, 170.0, 90.0]),
        (1_500.0, [40.0, 120.0, 150.0]),
        (4_000.0, [20.0, 40.0, 120.0]),
    ];
    let x = mm_yr.max(STOPS[0].0).log10();
    let mut c = STOPS[STOPS.len() - 1].1;
    for w in STOPS.windows(2) {
        let (a, b) = (w[0].0.log10(), w[1].0.log10());
        if x <= b {
            let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
            c = [0, 1, 2].map(|k| w[0].1[k] + (w[1].1[k] - w[0].1[k]) * t);
            break;
        }
    }
    c.map(|v| v.round() as u8)
}

/// The year's mean temperature, °C, on one ramp: deep blue at −60, white at 0, red at 40, dark red from 100.
fn temperature_colour(c: f64) -> [u8; 3] {
    const STOPS: [(f64, [f64; 3]); 7] = [
        (-60.0, [40.0, 40.0, 140.0]),
        (-20.0, [90.0, 150.0, 220.0]),
        (0.0, [240.0, 240.0, 240.0]),
        (20.0, [240.0, 190.0, 90.0]),
        (40.0, [210.0, 80.0, 40.0]),
        (100.0, [120.0, 20.0, 20.0]),
        (160.0, [60.0, 0.0, 0.0]),
    ];
    let x = c.clamp(STOPS[0].0, STOPS[STOPS.len() - 1].0);
    let mut out = STOPS[STOPS.len() - 1].1;
    for w in STOPS.windows(2) {
        if x <= w[1].0 {
            let t = (x - w[0].0) / (w[1].0 - w[0].0);
            out = [0, 1, 2].map(|k| w[0].1[k] + (w[1].1[k] - w[0].1[k]) * t);
            break;
        }
    }
    out.map(|v| v.round() as u8)
}

/// The temperature row's class in °C (class 0 is 173 K, one class a kelvin).
fn row_celsius(class: u8) -> f64 {
    f64::from(vd_terrain::artifact::TEMPERATURE_FLOOR_DK) / 10.0 + f64::from(class) - 273.15
}

/// ★ ANY WORLD AS THE RAIN BENCH'S THREE MAPS (`maps`; `climate_bench art-maps` drew them for the home planet only): one
/// even longitude–latitude picture each — `TAG_relief.png` (the ground by its height over the datum, shaded by a light
/// from the north-west; the sea's floor in blues, shaded the same, so a world under one sea shows its floor),
/// `TAG_rain.png` (the rain rows on the one rain scale), `TAG_rivers.png` (the discharge rows) and `TAG_temperature.png`
/// (the year's mean temperature row) — and the readings. THE
/// DATUM is the sea's level; a dry world (no sea) and a world under one sea have none, so their datum is the median
/// ground, the caption says so, and the land takes sand and rock by height in place of the lowland's green.
fn world_maps(
    lattice: &MacroLattice,
    radius_m: f64,
    a: &Artifact,
    out: &str,
    tag: &str,
    window: Option<[f64; 4]>,
) {
    const MAP_W: u32 = 1_440;
    let [lat0, lat1, lon0, lon1] = window.unwrap_or([-90.0, 90.0, 0.0, 360.0]);
    std::fs::create_dir_all(out).expect("the output folder");
    let wet = |i: usize| {
        (a.rows[i].receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
            & vd_terrain::solve::FACIES_SEA
            != 0
    };
    let sea_stated = a.sea_m != i16::MIN && a.sea_m != i16::MAX;
    let datum = if sea_stated {
        f64::from(a.sea_m)
    } else {
        let mut z: Vec<i16> = a.rows.iter().map(|r| r.z_m).collect();
        z.sort_unstable();
        f64::from(z[z.len() / 2])
    };
    let node_at = |lat: f64, lon: f64| {
        let d = vd_core::glam::DVec3::new(
            lat.to_radians().cos() * lon.to_radians().cos(),
            lat.to_radians().cos() * lon.to_radians().sin(),
            lat.to_radians().sin(),
        );
        vd_bins::drainage_gates::node_under(lattice, d) as usize
    };
    let h = ((f64::from(MAP_W) * (lat1 - lat0) / (lon1 - lon0)).round() as u32).max(1);
    let (mut relief, mut rain, mut rivers, mut temperature) = (
        image::RgbImage::new(MAP_W, h),
        image::RgbImage::new(MAP_W, h),
        image::RgbImage::new(MAP_W, h),
        image::RgbImage::new(MAP_W, h),
    );
    let height = |lat: f64, lon: f64| f64::from(a.rows[node_at(lat, lon)].z_m) - datum;
    let step = (lon1 - lon0) / f64::from(MAP_W);
    let km_per_px = 2.0 * std::f64::consts::PI * radius_m / 1_000.0 * step / 360.0;
    for y in 0..h {
        let lat = lat1 - (f64::from(y) + 0.5) * (lat1 - lat0) / f64::from(h);
        let dlon = (lat.to_radians().cos()).max(0.05);
        for x in 0..MAP_W {
            let lon = lon0 + (f64::from(x) + 0.5) * step;
            let i = node_at(lat, lon);
            let hm = f64::from(a.rows[i].z_m) - datum;
            let dzx = (height(lat, lon + step) - height(lat, lon - step))
                / (2.0 * km_per_px * 1_000.0 * dlon);
            let dzy =
                (height(lat + step, lon) - height(lat - step, lon)) / (2.0 * km_per_px * 1_000.0);
            // A light from the north-west, 45° up; the slope exaggerated ten times on the globe, three in a window.
            let ex = if window.is_some() { 3.0 } else { 10.0 };
            let (nx, ny, nz) = (-dzx * ex, -dzy * ex, 1.0);
            let l = (nx * nx + ny * ny + nz * nz).sqrt();
            let shade = ((-nx * 0.5 + ny * 0.5 + nz * 0.707) / l).clamp(0.15, 1.0);
            // A world with no stated sea takes sand and rock by height (its low ground is no green lowland).
            let base: [f64; 3] = if wet(i) {
                [50.0, 80.0, 130.0]
            } else if !sea_stated {
                if hm < 0.0 {
                    [175.0, 150.0, 115.0]
                } else if hm < 1_000.0 {
                    [160.0, 125.0, 90.0]
                } else if hm < 2_500.0 {
                    [130.0, 105.0, 85.0]
                } else {
                    [200.0, 195.0, 190.0]
                }
            } else if hm < 300.0 {
                [110.0, 140.0, 90.0]
            } else if hm < 1_000.0 {
                [150.0, 150.0, 100.0]
            } else if hm < 2_500.0 {
                [150.0, 120.0, 90.0]
            } else {
                [200.0, 195.0, 190.0]
            };
            relief.put_pixel(x, y, image::Rgb(base.map(|v| (v * shade).round() as u8)));
            let r = &a.rows[i];
            temperature.put_pixel(
                x,
                y,
                image::Rgb(temperature_colour(row_celsius(r.temperature))),
            );
            let lo = decode_class(r.rain, 4) as f64;
            let hi = decode_class(r.rain.saturating_add(1), 4) as f64;
            rain.put_pixel(
                x,
                y,
                image::Rgb(rain_colour(if wet(i) { f64::NAN } else { 0.5 * (lo + hi) })),
            );
            let q = f64::from(r.discharge_log);
            let v = if wet(i) {
                20.0
            } else {
                ((q - 60.0) * 2.0).clamp(0.0, 255.0)
            };
            rivers.put_pixel(
                x,
                y,
                image::Rgb([(v * 0.3) as u8, (v * 0.6) as u8, v as u8]),
            );
        }
    }
    for (img, kind) in [
        (&relief, "relief"),
        (&rain, "rain"),
        (&rivers, "rivers"),
        (&temperature, "temperature"),
    ] {
        img.save(format!("{out}/{tag}_{kind}.png")).expect("saves");
    }
    let mut all_c: Vec<f64> = a.rows.iter().map(|r| row_celsius(r.temperature)).collect();
    all_c.sort_by(f64::total_cmp);
    let area_mean = all_c.iter().sum::<f64>() / all_c.len() as f64;
    println!(
        "{tag}: the year's mean temperature over every node: mean {area_mean:.1} °C, 10th {:.1}, median {:.1}, 90th {:.1}, coldest {:.1}, warmest {:.1}",
        all_c[all_c.len() / 10],
        all_c[all_c.len() / 2],
        all_c[all_c.len() * 9 / 10],
        all_c[0],
        all_c[all_c.len() - 1]
    );
    let land: Vec<usize> = (0..a.rows.len()).filter(|&i| !wet(i)).collect();
    println!(
        "{tag}: the datum {} ({datum:.0} m over the ladder radius); the land {:.2} % of the nodes",
        if sea_stated {
            "the sea's level"
        } else {
            "the median ground (no sea stated)"
        },
        100.0 * land.len() as f64 / a.rows.len() as f64
    );
    if land.is_empty() {
        return;
    }
    let mut land_rain: Vec<f64> = land
        .iter()
        .map(|&i| {
            let r = &a.rows[i];
            0.5 * (decode_class(r.rain, 4) as f64
                + decode_class(r.rain.saturating_add(1), 4) as f64)
        })
        .collect();
    land_rain.sort_by(f64::total_cmp);
    let dry = land_rain.iter().filter(|&&p| p < 250.0).count();
    let mut land_h: Vec<f64> = land
        .iter()
        .map(|&i| f64::from(a.rows[i].z_m) - datum)
        .collect();
    land_h.sort_by(f64::total_cmp);
    let over =
        |m: f64| 100.0 * land_h.iter().filter(|&&v| v > m).count() as f64 / land_h.len() as f64;
    println!(
        "{tag}: the land's rain median {:.0} mm/yr (10th {:.0}, 90th {:.0}), under 250 mm {:.1} %; its height median {:.0} m, 90th {:.0} m, top {:.0} m; over 1 / 2 / 3 km {:.1} / {:.1} / {:.1} %",
        land_rain[land_rain.len() / 2],
        land_rain[land_rain.len() / 10],
        land_rain[land_rain.len() * 9 / 10],
        100.0 * dry as f64 / land_rain.len() as f64,
        land_h[land_h.len() / 2],
        land_h[land_h.len() * 9 / 10],
        land_h[land_h.len() - 1],
        over(1_000.0),
        over(2_000.0),
        over(3_000.0)
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

/// Each standing water's spread on a raster (the water map's own rule, `fine::water::raster_water`, without the rivers):
/// how many waters reach each cell.
fn spread_counts(ground: &[i32], side: usize, node_level: &[Option<i32>]) -> Vec<u8> {
    use vd_terrain::fine::water::DRY;
    let n = ground.len();
    let mut reach = vec![0u8; n];
    let mut levels: Vec<i32> = node_level.iter().flatten().copied().collect();
    levels.sort_unstable();
    levels.dedup();
    let under = |k: usize, lam: i32| ground[k] != DRY && ground[k] < lam;
    for lam in levels {
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = (0..n)
            .filter(|&k| node_level[k] == Some(lam) && under(k, lam))
            .collect();
        for &k in &stack {
            seen[k] = true;
        }
        while let Some(k) = stack.pop() {
            reach[k] = reach[k].saturating_add(1);
            let x = k % side;
            let around = [
                (x > 0).then(|| k - 1),
                (x + 1 < side).then(|| k + 1),
                (k >= side).then(|| k - side),
                (k + side < n).then(|| k + side),
            ];
            for m in around.into_iter().flatten() {
                if !seen[m] && under(m, lam) {
                    seen[m] = true;
                    stack.push(m);
                }
            }
        }
    }
    reach
}

/// THE WATER'S FACES AND STEPS on a square of `side` cells (each cell its ground, its water level or `none`, both in
/// quarter metres; `cut` the tiles' side where the square is stitched from tiles): a FACE is a cell's water beside an open
/// cell whose ground stands under that level (the water would stand as a wall); a STEP is two neighbouring waters more than
/// a metre apart. Returns `[faces inside, faces across a cut, steps inside, steps across a cut]`, and the faces whose
/// water is a river's (`river`, when given) apart: `[river faces, standing faces]`.
fn faces_and_steps(
    ground: &[i32],
    water: &[Option<i32>],
    side: usize,
    cut: Option<usize>,
    river: Option<&[bool]>,
) -> ([usize; 4], [usize; 2]) {
    use vd_terrain::fine::water::DRY;
    let (mut n, mut by_kind) = ([0usize; 4], [0usize; 2]);
    for y in 0..side {
        for x in 0..side {
            let a = y * side + x;
            for (b, crosses) in [
                (x + 1 < side).then(|| (a + 1, cut.is_some_and(|c| (x + 1) % c == 0))),
                (y + 1 < side).then(|| (a + side, cut.is_some_and(|c| (y + 1) % c == 0))),
            ]
            .into_iter()
            .flatten()
            {
                let across = usize::from(crosses);
                match (water[a], water[b]) {
                    (Some(la), Some(lb)) => {
                        if (la - lb).abs() > 4 {
                            n[2 + across] += 1;
                        }
                    }
                    (Some(l), None) | (None, Some(l)) => {
                        let (wet, open) = if water[a].is_some() { (a, b) } else { (b, a) };
                        if ground[open] != DRY && ground[open] < l {
                            n[across] += 1;
                            by_kind[usize::from(!river.is_some_and(|r| r[wet]))] += 1;
                        }
                    }
                    (None, None) => {}
                }
            }
        }
    }
    (n, by_kind)
}

/// ★ SLICE W1 ITEM 5 — THE WATER MAP'S READINGS (the one-volume revision §3.10) on the 3 × 3 tiles around `F,TX,TY`,
/// solved under the game's rule and finished as the planet's shard finishes them:
/// - G-DIVIDE: raw cells two standing waters reach (the higher holds them); its control, the same spread on today's
///   outlets' ground (`OutletRule::Today`);
/// - G-FACE: a water beside an open cell whose ground stands under its level, in the raw rasters (a river's bank and a
///   standing water apart) and on the finished square; its control, the raw map with the connection search switched off;
///   and the water's steps over a metre between two neighbouring waters;
/// - G-SEAM-WATER: the finished square's faces and steps across the tiles' cuts;
/// - the pockets: finished dry cells under the sea's level;
///
/// and four side views on the finished square — a coast, a lake shore, a river, a pocket — each from 1.5 km on its dry
/// side, 150 m up.
fn water_gates(body: &BodyDefinition, lattice: &MacroLattice, args: &[String]) {
    use vd_seed::bend::Face;
    use vd_terrain::fine::finish::block_keys;
    use vd_terrain::fine::l1::{L1Tile, OutletRule, solve_l1_with};
    use vd_terrain::fine::level::{FineLevel, LevelGeom};
    use vd_terrain::fine::stitch::{Field, MISSING};
    use vd_terrain::fine::tile::{WET_FULL, finish_water, raw_field, water_planes};
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
        face: Face::from_index(tile[0] as u8).expect("a face"),
        tx: tile[1],
        ty: tile[2],
    };
    let geom = LevelGeom::of(FineLevel::L1, body, lattice).expect("a level");
    let ring: Vec<TileKey> = block_keys(key, geom).into_iter().flatten().collect();
    assert!(
        ring.len() == 9 && ring.iter().all(|k| k.face == key.face),
        "the bench reads tiles inside one face"
    );
    let mut all: Vec<TileKey> = ring
        .iter()
        .flat_map(|&k| block_keys(k, geom).into_iter().flatten())
        .collect();
    all.sort();
    all.dedup();
    let solve_all = |keys: &[TileKey], rule: OutletRule| -> BTreeMap<TileKey, L1Tile> {
        std::thread::scope(|sc| {
            let handles: Vec<_> = keys
                .iter()
                .map(|&k| {
                    let artifact = &artifact;
                    sc.spawn(move || {
                        solve_l1_with(body, lattice, artifact, k, rule).map(|t| (k, t))
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| h.join().expect("a solve"))
                .collect()
        })
    };
    let started = std::time::Instant::now();
    let game = solve_all(&all, OutletRule::StandingWater);
    let today = solve_all(&ring, OutletRule::Today);
    println!(
        "{} tiles under the game's rule and {} under today's solved in {:.1} s",
        game.len(),
        today.len(),
        started.elapsed().as_secs_f64()
    );
    // Each raw cell's node's standing water (quarter metres over the sea), as the solve reads it.
    let shift = cells_per_node_shift(body, lattice, L1_RUNG).expect("the rung nests");
    let n_r = body.ladder().cells_per_edge(L1_RUNG) as i32;
    let sea = i32::from(artifact.sea_m);
    let identity = |k: TileKey, side: usize| -> Vec<Option<i32>> {
        use vd_terrain::artifact::{DRY_M, FACIES_SHIFT};
        let half = BUFFER_NODES >> 1;
        let (i0, j0) = (
            (k.tx * TILE_NODES - half) << shift,
            (k.ty * TILE_NODES - half) << shift,
        );
        (0..side * side)
            .map(|c| {
                let cell = chart_cell(k.face, i0 + (c % side) as i32, j0 + (c / side) as i32, n_r)?;
                let f = Face::from_index(cell.face)?;
                let r = &artifact.rows[lattice.index(f, cell.i >> shift, cell.j >> shift) as usize];
                if (r.receiver_facies >> FACIES_SHIFT) & vd_terrain::solve::FACIES_SEA != 0 {
                    Some(0)
                } else if r.water_m != DRY_M && r.water_m > r.z_m {
                    Some(4 * (i32::from(r.water_m) - sea))
                } else {
                    None
                }
            })
            .collect()
    };
    // G-DIVIDE ON EVERY SOLVED TILE (the ring's neighbours too): the cells two standing waters reach, and the pairs of
    // levels that meet (metres).
    let mut met: BTreeMap<(i32, i32), usize> = BTreeMap::new();
    let mut all_contacts = 0usize;
    for (k, g) in &game {
        let id = identity(*k, g.side);
        let mut levels: Vec<i32> = id.iter().flatten().copied().collect();
        levels.sort_unstable();
        levels.dedup();
        let reach = spread_counts(&g.quarters, g.side, &id);
        for (c, &r) in reach.iter().enumerate() {
            if r >= 2 {
                all_contacts += 1;
                // The lowest water that reaches the cell is the one under it; the cell holds the highest.
                let lo = levels
                    .iter()
                    .copied()
                    .find(|&l| g.quarters[c] < l)
                    .unwrap_or(0);
                *met.entry((lo / 4, g.water[c] / 4)).or_default() += 1;
            }
        }
    }
    println!(
        "G-DIVIDE on all {} solved tiles: {all_contacts} cells two standing waters reach; the levels that meet (lower, held; m): {:?}",
        game.len(),
        met.iter().collect::<Vec<_>>()
    );
    // THE RAW READINGS on the ring's tiles.
    let (mut contacts, mut contacts_today, mut cells) = (0usize, 0usize, 0usize);
    let (mut raw_faces, mut raw_kinds, mut nc_faces) = ([0usize; 4], [0usize; 2], [0usize; 4]);
    for k in &ring {
        let (Some(g), Some(t)) = (game.get(k), today.get(k)) else {
            continue;
        };
        let id = identity(*k, g.side);
        cells += g.side * g.side;
        contacts += spread_counts(&g.quarters, g.side, &id)
            .iter()
            .filter(|&&c| c >= 2)
            .count();
        contacts_today += spread_counts(&t.quarters, t.side, &id)
            .iter()
            .filter(|&&c| c >= 2)
            .count();
        let water: Vec<Option<i32>> = g
            .water
            .iter()
            .map(|&w| (w != vd_terrain::fine::water::DRY).then_some(w))
            .collect();
        let river: Vec<bool> = g.river.iter().map(|&b| b != i32::MIN).collect();
        let (f, kinds) = faces_and_steps(&g.quarters, &water, g.side, None, Some(&river));
        // The control: every cell under its node's level wet, no connection searched (the rivers as built).
        let unsearched: Vec<Option<i32>> = (0..id.len())
            .map(|c| {
                let lake = id[c].filter(|&l| g.quarters[c] != MISSING && g.quarters[c] < l);
                let bank = (g.river[c] != i32::MIN).then_some(g.river[c]);
                lake.max(bank)
            })
            .collect();
        let (fc, _) = faces_and_steps(&g.quarters, &unsearched, g.side, None, None);
        for i in 0..4 {
            raw_faces[i] += f[i];
            nc_faces[i] += fc[i];
        }
        raw_kinds[0] += kinds[0];
        raw_kinds[1] += kinds[1];
    }
    println!("THE RAW RASTERS (the ring's nine tiles, {cells} cells):");
    println!(
        "  G-DIVIDE: {contacts} cells two standing waters reach (today's outlets' ground: {contacts_today})"
    );
    println!(
        "  G-FACE: {} faces — {} at a river's bank, {} at a standing water; steps over 1 m: {} (connection switched off: {} faces, {} steps)",
        raw_faces[0], raw_kinds[0], raw_kinds[1], raw_faces[2], nc_faces[0], nc_faces[2]
    );
    // THE FINISHED SQUARE: the ring finished with its water, stitched 3 × 3.
    let raws: BTreeMap<TileKey, (Field, Field, Field)> = game
        .iter()
        .map(|(&k, t)| {
            let (level, wet) = water_planes(&t.water, &t.river_share);
            (
                k,
                (
                    raw_field(k, geom, t.side, t.quarters.clone()),
                    raw_field(k, geom, t.side, level),
                    raw_field(k, geom, t.side, wet),
                ),
            )
        })
        .collect();
    let s0 = geom.side() as usize;
    let w = 3 * s0;
    let (mut z, mut level, mut wet) = (
        vec![MISSING; w * w],
        vec![MISSING; w * w],
        vec![0i32; w * w],
    );
    for &k in &ring {
        let keys = block_keys(k, geom);
        let pick = |plane: usize| -> [Option<&Field>; 9] {
            std::array::from_fn(|i| {
                keys[i].and_then(|kk| raws.get(&kk)).map(|r| match plane {
                    0 => &r.0,
                    1 => &r.1,
                    _ => &r.2,
                })
            })
        };
        let block: [Option<(TileKey, &Field)>; 9] =
            std::array::from_fn(|i| keys[i].and_then(|kk| raws.get(&kk).map(|r| (kk, &r.0))));
        let f = finish_water(block, pick(1), pick(2), geom).expect("a tile finishes");
        let (ox, oy) = (
            (k.tx - key.tx + 1) as usize * s0,
            (k.ty - key.ty + 1) as usize * s0,
        );
        for y in 0..s0 {
            for x in 0..s0 {
                let (src, dst) = (y * s0 + x, (oy + y) * w + ox + x);
                z[dst] = f.heights[0][src];
                level[dst] = f.level[0][src];
                wet[dst] = f.wet[0][src];
            }
        }
    }
    let water: Vec<Option<i32>> = (0..w * w)
        .map(|c| (wet[c] == WET_FULL && level[c] != MISSING).then_some(level[c]))
        .collect();
    let (f, _) = faces_and_steps(&z, &water, w, Some(s0), None);
    let pockets = (0..w * w)
        .filter(|&c| water[c].is_none() && z[c] != MISSING && z[c] < 0)
        .count();
    let wet_cells = water.iter().flatten().count();
    // The finished square kept for a closer reading: its heights and its water (quarter metres, little-endian words,
    // `i32::MIN` where none), row by row.
    let dump = |name: &str, v: &mut dyn Iterator<Item = i32>| {
        let bytes: Vec<u8> = v.flat_map(i32::to_le_bytes).collect();
        std::fs::write(format!("{out}/{name}"), bytes).expect("the dump writes");
    };
    dump("finished_z.i32", &mut z.iter().copied());
    dump(
        "finished_water.i32",
        &mut water.iter().map(|l| l.unwrap_or(i32::MIN)),
    );
    println!(
        "THE FINISHED SQUARE (3 × 3 tiles, {} cells, {wet_cells} under water; {w} a side):",
        w * w
    );
    println!(
        "  G-FACE: {} faces and {} steps over 1 m inside the tiles",
        f[0], f[2]
    );
    println!(
        "  G-SEAM-WATER: {} faces and {} steps over 1 m across the tiles' cuts",
        f[1], f[3]
    );
    println!(
        "  pockets: {pockets} dry cells under the sea's level ({:.2} km²)",
        pockets as f64 * 0.128 * 0.128
    );
    // THE SIDE VIEWS.
    let zf: Vec<f32> = z
        .iter()
        .map(|&v| {
            if v == MISSING {
                f32::NAN
            } else {
                v as f32 / 4.0
            }
        })
        .collect();
    let wl: Vec<f32> = water
        .iter()
        .map(|l| l.map_or(f32::NAN, |l| l as f32 / 4.0))
        .collect();
    let centre = (w / 2) as i64;
    let near_centre = |c: usize| {
        let (x, y) = ((c % w) as i64, (c / w) as i64);
        (x - centre).pow(2) + (y - centre).pow(2)
    };
    let shore = |c: usize| {
        let (x, y) = (c % w, c / w);
        water[c].is_some()
            && x > 0
            && y > 0
            && x + 1 < w
            && y + 1 < w
            && [c - 1, c + 1, c - w, c + w]
                .iter()
                .any(|&m| water[m].is_none() && z[m] != MISSING)
    };
    let window = |c: usize| -> Vec<i32> {
        let (x, y) = ((c % w) as i64, (c / w) as i64);
        let mut v = Vec::new();
        for dy in -4..=4i64 {
            for dx in -4..=4i64 {
                let (xx, yy) = (x + dx, y + dy);
                if xx >= 0
                    && yy >= 0
                    && (xx as usize) < w
                    && (yy as usize) < w
                    && let Some(l) = water[yy as usize * w + xx as usize]
                {
                    v.push(l);
                }
            }
        }
        v
    };
    let coast = (0..w * w)
        .filter(|&c| shore(c) && water[c] == Some(0))
        .min_by_key(|&c| near_centre(c));
    let lake = (0..w * w)
        .filter(|&c| {
            shore(c)
                && water[c].is_some_and(|l| l != 0)
                && window(c).iter().filter(|&&l| Some(l) == water[c]).count() >= 40
        })
        .min_by_key(|&c| near_centre(c));
    // A river: the middle tile's carved channel cell nearest the square's centre (its raw raster reaches half a node past
    // the tile each side).
    let river = game.get(&key).and_then(|g| {
        let margin = (g.side - s0) / 2;
        (0..g.side * g.side)
            .filter(|&c| g.river[c] != i32::MIN)
            .filter_map(|c| {
                let (a, b) = (c % g.side, c / g.side);
                let (x, y) = ((a + s0).checked_sub(margin)?, (b + s0).checked_sub(margin)?);
                (x < w && y < w).then_some(y * w + x)
            })
            .min_by_key(|&c| near_centre(c))
    });
    let pocket = (0..w * w)
        .filter(|&c| water[c].is_none() && z[c] != MISSING && z[c] < 0)
        .min_by_key(|&c| z[c]);
    for (name, target) in [
        ("coast", coast),
        ("lake", lake),
        ("river", river),
        ("pocket", pocket),
    ] {
        let Some(t) = target else {
            println!("  side view {name}: none on this square");
            continue;
        };
        let (tx, ty) = ((t % w) as f64, (t / w) as f64);
        let dist = 1_500.0 / 128.0;
        let eye = (0..8)
            .filter_map(|i| {
                let h = f64::from(i) * 45.0;
                let (sx, sy) = (h.to_radians().sin(), -h.to_radians().cos());
                let (ex, ey) = (tx - sx * dist, ty - sy * dist);
                let inside = ex >= 1.0 && ey >= 1.0 && ex < (w - 2) as f64 && ey < (w - 2) as f64;
                let c = ey as usize * w + ex as usize;
                (inside && water[c].is_none() && z[c] != MISSING).then_some((h, ex, ey, z[c]))
            })
            .max_by_key(|e| e.3);
        let Some((h, ex, ey, _)) = eye else {
            println!("  side view {name}: no dry eye 1.5 km around it");
            continue;
        };
        let path = format!("{out}/side_{name}.png");
        render_view_water(&zf, Some(&wl), w, 128.0, (ex, ey), 150.0, h, 6.0, &path);
        println!(
            "  side view {name}: the target cell ({tx}, {ty}) at {:.1} m (water {}), the eye 1.5 km away facing {h}°: {path}",
            f64::from(z[t]) / 4.0,
            water[t].map_or("none".to_string(), |l| format!(
                "{:.1} m",
                f64::from(l) / 4.0
            ))
        );
    }
}

/// ★ THE TILE SOLVE'S PROBE (bench only): `solve_l1_with`'s steps under the game's rule on the tile `F,TX,TY`, each
/// stage's rises and heights read for words that are not finite, and the nodes whose rises left the finite range.
fn l1_probe(body: &BodyDefinition, lattice: &MacroLattice, args: &[String]) {
    use vd_terrain::fine::l1::{
        Waters, base_levels, domain, exits_at_bed, rounds, standing_water_outlets,
    };
    let world = args.get(2).expect("WORLD.art");
    let tile: Vec<i32> = args
        .get(3)
        .expect("F,TX,TY")
        .split(',')
        .map(|x| x.trim().parse().expect("an integer"))
        .collect();
    let artifact = vd_bins::artifact_file::read(world).expect("the world reads");
    let key = TileKey {
        face: vd_seed::bend::Face::from_index(tile[0] as u8).expect("a face"),
        tx: tile[1],
        ty: tile[2],
    };
    let bad = |v: &[Gf]| v.iter().filter(|x| !x.to_f64().is_finite()).count();
    let range = |v: &[Gf]| {
        let f: Vec<f64> = v
            .iter()
            .map(|x| x.to_f64())
            .filter(|x| x.is_finite())
            .collect();
        (
            f.iter().copied().fold(f64::INFINITY, f64::min),
            f.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let mut coarse = domain(body, lattice, &artifact, COARSE_RUNG, key).expect("a coarse domain");
    let none = BTreeMap::new();
    let waters = Waters::new(lattice, &artifact, &none);
    coarse.outlet = standing_water_outlets(&coarse, &waters);
    base_levels(&mut coarse, &waters);
    exits_at_bed(&mut coarse, body, lattice, &artifact);
    println!(
        "coarse: {} points, {} outlets, {} nodes ({} held); the base not finite at {} points",
        coarse.dirs.len(),
        coarse.outlet.iter().filter(|&&o| o).count(),
        coarse.nodes.len(),
        coarse.held.iter().filter(|&&h| h).count(),
        bad(&coarse.base)
    );
    let mut uc = coarse.ref_u0.clone();
    let mut hc = coarse.base.clone();
    for round in 0..COARSE_ROUNDS {
        let passes = rounds(&coarse, &mut uc, &mut hc, 1);
        let (lo, hi) = range(&uc);
        println!(
            "  coarse round {round}: {passes:?} passes; rises {lo:.3e} … {hi:.3e}, {} not finite; heights {} not finite",
            bad(&uc),
            bad(&hc)
        );
    }
    // The nodes: row, base level, held, and the mean of their coarse points.
    let nl = coarse.nodes.len();
    let (mut sum, mut area) = (vec![0.0f64; nl], vec![0.0f64; nl]);
    for ((&k, &hp), &a) in coarse.pnode.iter().zip(&hc).zip(&coarse.geom.area) {
        sum[k as usize] += hp.to_f64() * a.to_f64();
        area[k as usize] += a.to_f64();
    }
    for k in 0..nl {
        let r = coarse.node_ref[k] as usize;
        let u = uc.get(r).map_or(f64::NAN, |x| x.to_f64());
        let mean = sum[k] / area[k];
        if !u.is_finite() || !(1e-12..1e3).contains(&u) || !mean.is_finite() {
            // Its points: how many are outlets, their lowest base, and its 8 km chain to its end.
            let pts: Vec<usize> = (0..coarse.pnode.len())
                .filter(|&p| coarse.pnode[p] as usize == k)
                .collect();
            let outs = pts.iter().filter(|&&p| coarse.outlet[p]).count();
            let low = pts
                .iter()
                .map(|&p| coarse.base[p].to_f64())
                .fold(f64::INFINITY, f64::min);
            let mut chain = Vec::new();
            let mut g = coarse.nodes[k];
            for _ in 0..12 {
                let row = &artifact.rows[g as usize];
                let f = row.receiver_facies;
                chain.push(format!(
                    "{g}{}:{}{}",
                    if coarse.nodes.contains(&g) {
                        ""
                    } else {
                        "(out)"
                    },
                    i32::from(row.z_m) - i32::from(artifact.sea_m),
                    if row.water_m != vd_terrain::artifact::DRY_M && row.water_m > row.z_m {
                        format!("~{}", i32::from(row.water_m) - i32::from(artifact.sea_m))
                    } else {
                        String::new()
                    }
                ));
                if f & vd_terrain::artifact::RECEIVER_NONE != 0 {
                    chain.push("END".into());
                    break;
                }
                g = lattice.neighbours(g)[(f & vd_terrain::artifact::RECEIVER_SLOT_MASK) as usize];
            }
            println!(
                "    its {} points: {outs} outlets, the lowest base {low:.1}; its chain {}",
                pts.len(),
                chain.join(" → ")
            );
            println!(
                "  node {k} (global {}): row {:.1}, base {:.1}, held {}, mean {:.3}, rise {u:.3e}",
                coarse.nodes[k],
                coarse.node_z[k].to_f64(),
                coarse.node_base[k].to_f64(),
                coarse.held[k],
                mean
            );
        }
    }
}

/// ★ THE SPILL COUNT (the spill law's short design, bench only): for `N` spilling lakes of the 8 km world (half the
/// largest, half spread over the rest by size), the 128 m tile that holds each lake's BRIM node (the first node its water's
/// own receivers reach outside the lake) is solved under the game's rule; in its raw raster the LOWEST PATH from the lake's
/// own cells to any cell of a node outside the lake whose row stands more than a metre under the lake's level gives the
/// lake's spill on the fine ground. A spill under the 8 km level is a rim the 128 m ground cut under it: the lake's water
/// would run out there. Prints each lake and the counts.
fn spill_count(body: &BodyDefinition, lattice: &MacroLattice, args: &[String]) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use vd_seed::bend::Face;
    use vd_terrain::artifact::{DRY_M, FACIES_SHIFT, RECEIVER_NONE, RECEIVER_SLOT_MASK};
    use vd_terrain::fine::l1::{OutletRule, solve_l1_with};
    use vd_terrain::macro_lattice::NO_NODE;
    let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art")).expect("the world reads");
    let want: usize = args.get(3).map_or(40, |s| s.parse().expect("N"));
    let n = a.rows.len();
    let sea_m = i32::from(a.sea_m);
    let is_sea =
        |i: usize| (a.rows[i].receiver_facies >> FACIES_SHIFT) & vd_terrain::solve::FACIES_SEA != 0;
    let lake = |i: usize| {
        let r = &a.rows[i];
        !is_sea(i) && r.water_m != DRY_M && r.water_m > r.z_m
    };
    let receiver = |i: u32| {
        let f = a.rows[i as usize].receiver_facies;
        if f & RECEIVER_NONE != 0 {
            NO_NODE
        } else {
            lattice.neighbours(i)[(f & RECEIVER_SLOT_MASK) as usize]
        }
    };
    // Every spilling lake: its nodes, area, level (m over the sea) and brim node.
    struct Lake {
        nodes: BTreeSet<u32>,
        area_km2: f64,
        level_m: i32,
        brim: u32,
    }
    let mut seen = vec![false; n];
    let mut lakes: Vec<Lake> = Vec::new();
    for start in 0..n {
        if seen[start] || !lake(start) {
            continue;
        }
        let (mut stack, mut nodes, mut area, mut spills) =
            (vec![start as u32], BTreeSet::new(), 0f64, false);
        seen[start] = true;
        while let Some(i) = stack.pop() {
            nodes.insert(i);
            area += lattice.area_m2(i) as f64;
            spills |= a.rows[i as usize].receiver_facies & RECEIVER_NONE == 0;
            for m in lattice.neighbours(i) {
                if m != NO_NODE && !seen[m as usize] && lake(m as usize) {
                    seen[m as usize] = true;
                    stack.push(m);
                }
            }
        }
        if !spills {
            continue;
        }
        let mut k = start as u32;
        for _ in 0..n {
            if !nodes.contains(&k) {
                break;
            }
            let r = receiver(k);
            if r == NO_NODE {
                break;
            }
            k = r;
        }
        if nodes.contains(&k) {
            continue;
        }
        lakes.push(Lake {
            level_m: i32::from(a.rows[start].water_m) - sea_m,
            nodes,
            area_km2: area / 1e6,
            brim: k,
        });
    }
    lakes.sort_by(|x, y| y.area_km2.total_cmp(&x.area_km2));
    let total = lakes.len();
    // The sample: the largest half, and the rest spread evenly over the smaller lakes.
    let half = want / 2;
    let mut pick: Vec<usize> = (0..half.min(total)).collect();
    let rest = total.saturating_sub(half);
    let others = want - pick.len();
    for i in 0..others.min(rest) {
        pick.push(half + i * rest / others.max(1));
    }
    pick.dedup();
    println!("{total} spilling lakes; {} sampled", pick.len());
    let shift = cells_per_node_shift(body, lattice, L1_RUNG).expect("the rung nests");
    let n_r = body.ladder().cells_per_edge(L1_RUNG) as i32;
    let tile_of = |node: u32| {
        let (f, ni, nj) = lattice.split(node);
        TileKey {
            face: f,
            tx: ni.div_euclid(TILE_NODES),
            ty: nj.div_euclid(TILE_NODES),
        }
    };
    // (index, area, level m, floor m, the fine spill m or none seen, cells under the level)
    type Row = (usize, f64, i32, f64, Option<f64>);
    let rows: Vec<Row> = std::thread::scope(|sc| {
        let handles: Vec<_> = pick
            .iter()
            .map(|&li| {
                let (lk, a) = (&lakes[li], &a);
                sc.spawn(move || {
                    let key = tile_of(lk.brim);
                    let t = solve_l1_with(body, lattice, a, key, OutletRule::StandingWater)?;
                    let half = BUFFER_NODES >> 1;
                    let (i0, j0) = (
                        (key.tx * TILE_NODES - half) << shift,
                        (key.ty * TILE_NODES - half) << shift,
                    );
                    let side = t.side;
                    let lam = 4 * lk.level_m;
                    let node_of = |c: usize| -> Option<u32> {
                        let cell = chart_cell(
                            key.face,
                            i0 + (c % side) as i32,
                            j0 + (c / side) as i32,
                            n_r,
                        )?;
                        let f = Face::from_index(cell.face)?;
                        Some(lattice.index(f, cell.i >> shift, cell.j >> shift))
                    };
                    let nodes: Vec<Option<u32>> = (0..side * side).map(node_of).collect();
                    let is_lake = |c: usize| nodes[c].is_some_and(|k| lk.nodes.contains(&k));
                    let is_low = |c: usize| {
                        nodes[c].is_some_and(|k| {
                            !lk.nodes.contains(&k)
                                && (i32::from(a.rows[k as usize].z_m) - sea_m) * 4 < lam - 4
                        })
                    };
                    let floor = (0..side * side)
                        .filter(|&c| is_lake(c) && t.quarters[c] != i32::MIN)
                        .map(|c| t.quarters[c])
                        .min()
                        .map_or(f64::NAN, |q| f64::from(q) / 4.0);
                    // The lowest path: the greatest ground along it, from the lake's cells, to a low node's cell.
                    let mut best = vec![i32::MAX; side * side];
                    let mut heap = BinaryHeap::new();
                    for (c, b) in best.iter_mut().enumerate() {
                        if is_lake(c) && t.quarters[c] != i32::MIN {
                            *b = t.quarters[c];
                            heap.push(Reverse((t.quarters[c], c)));
                        }
                    }
                    let mut spill = None;
                    while let Some(Reverse((h, c))) = heap.pop() {
                        if h > best[c] {
                            continue;
                        }
                        if h >= lam {
                            break;
                        }
                        if is_low(c) {
                            spill = Some(f64::from(h) / 4.0);
                            break;
                        }
                        let x = c % side;
                        let around = [
                            (x > 0).then(|| c - 1),
                            (x + 1 < side).then(|| c + 1),
                            (c >= side).then(|| c - side),
                            (c + side < side * side).then(|| c + side),
                        ];
                        for m in around.into_iter().flatten() {
                            if t.quarters[m] == i32::MIN {
                                continue;
                            }
                            let hm = h.max(t.quarters[m]);
                            if hm < best[m] {
                                best[m] = hm;
                                heap.push(Reverse((hm, m)));
                            }
                        }
                    }
                    Some((li, lk.area_km2, lk.level_m, floor, spill))
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().expect("a solve"))
            .collect()
    });
    let (mut leaks, mut drains) = (0usize, 0usize);
    let mut drops = Vec::new();
    for &(li, area, level, floor, spill) in &rows {
        let drop = spill.map(|s| f64::from(level) - s);
        if let Some(d) = drop {
            leaks += 1;
            drops.push(d);
            if spill.is_some_and(|s| s <= floor) {
                drains += 1;
            }
        }
        println!(
            "  lake #{li}: {area:.0} km², level {level} m, its floor in the tile {floor:.1} m — {}",
            match (spill, drop) {
                (Some(s), Some(d)) =>
                    format!("SPILLS at {s:.1} m on the fine ground, {d:.1} m under its level"),
                _ => "holds its level in the tile".to_string(),
            }
        );
    }
    drops.sort_by(f64::total_cmp);
    let q = |f: f64| {
        drops
            .get(((drops.len().max(1) - 1) as f64 * f).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    println!(
        "{} of {} sampled lakes spill under their level on the 128 m ground (the drop p10 / p50 / p90 / most: {:.1} / {:.1} / {:.1} / {:.1} m); {drains} spill at or under their floor in the tile (drain whole)",
        leaks,
        rows.len(),
        q(0.1),
        q(0.5),
        q(0.9),
        q(1.0)
    );
}

/// ★ THE FULL SPILL COUNT (ruling B115 item 3, bench only): EVERY lake of the 8 km world, closed and spilling, over ALL
/// its 128 m tiles (each tile that holds a lake node or a node beside one, solved once under the game's rule and kept as
/// its own square). Each lake's tiles are joined into one map, and the lowest path from the lake's own cells to any cell of
/// a node outside the lake whose row stands more than a metre under the lake's level gives its spill on the 128 m ground.
/// A lake whose tiles cross a cube face is not read (counted). Writes `OUT_DIR/lakes.csv` and prints the counts.
fn spill_full(body: &BodyDefinition, lattice: &MacroLattice, args: &[String]) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vd_terrain::artifact::{DRY_M, FACIES_SHIFT, RECEIVER_NONE};
    use vd_terrain::fine::l1::{OutletRule, solve_l1_with};
    use vd_terrain::macro_lattice::NO_NODE;
    let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art")).expect("the world reads");
    let out = args.get(3).expect("OUT_DIR");
    std::fs::create_dir_all(out).expect("the output directory");
    let n = a.rows.len();
    let sea_m = i32::from(a.sea_m);
    let is_sea =
        |i: usize| (a.rows[i].receiver_facies >> FACIES_SHIFT) & vd_terrain::solve::FACIES_SEA != 0;
    let lake = |i: usize| {
        let r = &a.rows[i];
        !is_sea(i) && r.water_m != DRY_M && r.water_m > r.z_m
    };
    let tn = TILE_NODES;
    let tile_of = |node: u32| {
        let (f, ni, nj) = lattice.split(node);
        TileKey {
            face: f,
            tx: ni.div_euclid(tn),
            ty: nj.div_euclid(tn),
        }
    };
    struct Lake {
        nodes: BTreeSet<u32>,
        area_km2: f64,
        level_q: i32,
        spilling: bool,
        tiles: BTreeSet<TileKey>,
    }
    let mut seen = vec![false; n];
    let mut lakes: Vec<Lake> = Vec::new();
    for start in 0..n {
        if seen[start] || !lake(start) {
            continue;
        }
        let (mut stack, mut nodes, mut area, mut spilling) =
            (vec![start as u32], BTreeSet::new(), 0f64, false);
        let mut tiles = BTreeSet::new();
        seen[start] = true;
        while let Some(i) = stack.pop() {
            nodes.insert(i);
            area += lattice.area_m2(i) as f64;
            spilling |= a.rows[i as usize].receiver_facies & RECEIVER_NONE == 0;
            tiles.insert(tile_of(i));
            for m in lattice.neighbours(i) {
                if m == NO_NODE {
                    continue;
                }
                tiles.insert(tile_of(m));
                if !seen[m as usize] && lake(m as usize) {
                    seen[m as usize] = true;
                    stack.push(m);
                }
            }
        }
        lakes.push(Lake {
            level_q: 4 * (i32::from(a.rows[start].water_m) - sea_m),
            nodes,
            area_km2: area / 1e6,
            spilling,
            tiles,
        });
    }
    let crossing = |lk: &Lake| {
        lk.tiles
            .iter()
            .any(|t| t.face != lk.tiles.first().expect("a tile").face)
    };
    let all: BTreeSet<TileKey> = lakes
        .iter()
        .filter(|lk| !crossing(lk))
        .flat_map(|lk| lk.tiles.iter().copied())
        .collect();
    let all: Vec<TileKey> = all.into_iter().collect();
    let workers = std::thread::available_parallelism()
        .map_or(4, |c| c.get())
        .saturating_sub(2)
        .max(1);
    println!(
        "{} lakes ({} spilling), {} across a face edge not read; {} tiles to solve on {workers} threads",
        lakes.len(),
        lakes.iter().filter(|l| l.spilling).count(),
        lakes.iter().filter(|l| crossing(l)).count(),
        all.len()
    );
    // Each tile's own square (the raw raster less its half-node margin), solved once.
    let shift = cells_per_node_shift(body, lattice, L1_RUNG).expect("the rung nests");
    let s0 = (TILE_NODES << shift) as usize;
    let squares: Mutex<BTreeMap<TileKey, Vec<i32>>> = Mutex::new(BTreeMap::new());
    let next = AtomicUsize::new(0);
    let started = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&key) = all.get(i) else {
                        break;
                    };
                    let Some(t) = solve_l1_with(body, lattice, &a, key, OutletRule::StandingWater)
                    else {
                        continue;
                    };
                    let m = (t.side - s0) / 2;
                    let mut sq = Vec::with_capacity(s0 * s0);
                    for y in 0..s0 {
                        sq.extend_from_slice(
                            &t.quarters[(m + y) * t.side + m..(m + y) * t.side + m + s0],
                        );
                    }
                    let mut map = squares.lock().expect("the squares");
                    map.insert(key, sq);
                    if map.len().is_multiple_of(100) {
                        println!(
                            "  {} tiles solved in {:.0} s",
                            map.len(),
                            started.elapsed().as_secs_f64()
                        );
                    }
                }
            });
        }
    });
    let squares = squares.into_inner().expect("the squares");
    println!(
        "{} tiles solved in {:.0} s",
        squares.len(),
        started.elapsed().as_secs_f64()
    );
    // Each lake: its mosaic, its flood.
    let mut csv = String::from("lake,area_km2,spilling,level_m,floor_m,spill_m,drop_m,drains\n");
    // (spilling, drop m or none, drains whole, area)
    let mut results: Vec<(bool, Option<f64>, bool, f64)> = Vec::new();
    for (li, lk) in lakes.iter().enumerate() {
        if crossing(lk) {
            continue;
        }
        let (tx0, ty0) = (
            lk.tiles.iter().map(|t| t.tx).min().expect("a tile"),
            lk.tiles.iter().map(|t| t.ty).min().expect("a tile"),
        );
        let (tw, th) = (
            (lk.tiles.iter().map(|t| t.tx).max().expect("a tile") - tx0 + 1) as usize,
            (lk.tiles.iter().map(|t| t.ty).max().expect("a tile") - ty0 + 1) as usize,
        );
        let (w, h) = (tw * s0, th * s0);
        let mut z = vec![i32::MIN; w * h];
        for t in &lk.tiles {
            let Some(sq) = squares.get(t) else {
                continue;
            };
            let (ox, oy) = ((t.tx - tx0) as usize * s0, (t.ty - ty0) as usize * s0);
            for y in 0..s0 {
                z[(oy + y) * w + ox..(oy + y) * w + ox + s0]
                    .copy_from_slice(&sq[y * s0..(y + 1) * s0]);
            }
        }
        let face = lk.tiles.first().expect("a tile").face;
        let node_of = |c: usize| {
            let (x, y) = ((c % w) as i32, (c / w) as i32);
            let (ci, cj) = (tx0 * (tn << shift) + x, ty0 * (tn << shift) + y);
            lattice.index(face, ci >> shift, cj >> shift)
        };
        let lam = lk.level_q;
        let mut best = vec![i32::MAX; w * h];
        let mut heap = BinaryHeap::new();
        let mut floor = i32::MAX;
        for (c, b) in best.iter_mut().enumerate() {
            if z[c] != i32::MIN && lk.nodes.contains(&node_of(c)) {
                *b = z[c];
                floor = floor.min(z[c]);
                heap.push(Reverse((z[c], c)));
            }
        }
        let mut spill = None;
        while let Some(Reverse((hc, c))) = heap.pop() {
            if hc > best[c] {
                continue;
            }
            if hc >= lam {
                break;
            }
            let k = node_of(c);
            if !lk.nodes.contains(&k) && 4 * (i32::from(a.rows[k as usize].z_m) - sea_m) < lam - 4 {
                spill = Some(hc);
                break;
            }
            let x = c % w;
            let around = [
                (x > 0).then(|| c - 1),
                (x + 1 < w).then(|| c + 1),
                (c >= w).then(|| c - w),
                (c + w < w * h).then(|| c + w),
            ];
            for m in around.into_iter().flatten() {
                if z[m] == i32::MIN {
                    continue;
                }
                let hm = hc.max(z[m]);
                if hm < best[m] {
                    best[m] = hm;
                    heap.push(Reverse((hm, m)));
                }
            }
        }
        let drop = spill.map(|s| f64::from(lam - s) / 4.0);
        let drains = spill.is_some_and(|s| s <= floor);
        results.push((lk.spilling, drop, drains, lk.area_km2));
        csv.push_str(&format!(
            "{li},{:.1},{},{:.2},{:.2},{},{},{}\n",
            lk.area_km2,
            lk.spilling,
            f64::from(lam) / 4.0,
            f64::from(floor) / 4.0,
            spill.map_or(String::new(), |s| format!("{:.2}", f64::from(s) / 4.0)),
            drop.map_or(String::new(), |d| format!("{d:.2}")),
            drains
        ));
    }
    std::fs::write(format!("{out}/lakes.csv"), csv).expect("the table writes");
    for (kind, spilling) in [("spilling", true), ("closed", false)] {
        let of: Vec<&(bool, Option<f64>, bool, f64)> =
            results.iter().filter(|r| r.0 == spilling).collect();
        let mut drops: Vec<f64> = of.iter().filter_map(|r| r.1).collect();
        drops.sort_by(f64::total_cmp);
        let q = |f: f64| {
            drops
                .get(((drops.len().max(1) - 1) as f64 * f).round() as usize)
                .copied()
                .unwrap_or(0.0)
        };
        let area: f64 = of.iter().filter(|r| r.1.is_some()).map(|r| r.3).sum();
        println!(
            "{kind}: {} of {} lakes spill under their 8 km level on the 128 m ground ({area:.0} km²); the drop p10 / p50 / p90 / most {:.1} / {:.1} / {:.1} / {:.1} m; {} drain whole",
            drops.len(),
            of.len(),
            q(0.1),
            q(0.5),
            q(0.9),
            q(1.0),
            of.iter().filter(|r| r.2).count()
        );
    }
}

/// ★ ONE LAKE'S SPILL, READ TWO WAYS (bench only; ruling B115): the lake whose first node is `LAKE`, its tiles solved
/// under the game's rule; each tile's raw raster gives its own pass (`fine::spill::raster_passes`, the rule the shard would
/// run), and the tiles' own squares joined give the lake's pass over its whole map, with the path drawn
/// (`OUT_DIR/lake_<LAKE>_spill.png`: the ground shaded, the lake's own cells blue, the path red, its end yellow).
fn spill_lake(body: &BodyDefinition, lattice: &MacroLattice, args: &[String]) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use vd_seed::bend::Face;
    use vd_terrain::artifact::DRY_M;
    use vd_terrain::fine::l1::{OutletRule, solve_l1_with};
    use vd_terrain::fine::spill::{NO_LAKE, raster_passes};
    use vd_terrain::macro_lattice::NO_NODE;
    let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art")).expect("the world reads");
    // The lake: a node index, the full count's lake number `#N`, or a tile `F,TX,TY` whose middle node's lake (the lake
    // node nearest the tile's middle).
    let arg = args.get(3).expect("LAKE or F,TX,TY");
    let first: u32 = if let Some(number) = arg.strip_prefix('#') {
        // The full count's lake number (`spill-full`'s `lakes.csv`): its lakes in node order, each the connected lake nodes.
        let want: usize = number.parse().expect("a lake number");
        let lake = |i: usize| {
            let r = &a.rows[i];
            (r.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
                & vd_terrain::solve::FACIES_SEA
                == 0
                && r.water_m != DRY_M
                && r.water_m > r.z_m
        };
        let mut seen = vec![false; a.rows.len()];
        let mut found = None;
        let mut count = 0usize;
        for start in 0..a.rows.len() {
            if seen[start] || !lake(start) {
                continue;
            }
            if count == want {
                found = Some(start as u32);
                break;
            }
            count += 1;
            seen[start] = true;
            let mut stack = vec![start as u32];
            while let Some(i) = stack.pop() {
                for m in lattice.neighbours(i) {
                    if m != NO_NODE && !seen[m as usize] && lake(m as usize) {
                        seen[m as usize] = true;
                        stack.push(m);
                    }
                }
            }
        }
        found.expect("the lake number exists")
    } else if arg.contains(',') {
        let t: Vec<i32> = arg
            .split(',')
            .map(|x| x.trim().parse().expect("an integer"))
            .collect();
        let f = Face::from_index(t[0] as u8).expect("a face");
        let mid = TILE_NODES / 2;
        (0..TILE_NODES * TILE_NODES)
            .map(|q| (q % TILE_NODES, q / TILE_NODES))
            .map(|(dx, dy)| {
                (
                    (dx - mid).pow(2) + (dy - mid).pow(2),
                    lattice.index(f, t[1] * TILE_NODES + dx, t[2] * TILE_NODES + dy),
                )
            })
            .filter(|&(_, k)| {
                let r = &a.rows[k as usize];
                r.water_m != DRY_M && r.water_m > r.z_m
            })
            .min()
            .expect("a lake node in the tile")
            .1
    } else {
        arg.parse().expect("a node index")
    };
    let out = args.get(4).expect("OUT_DIR");
    std::fs::create_dir_all(out).expect("the output directory");
    let sea_m = i32::from(a.sea_m);
    let word = a.rows[first as usize].water_m;
    assert!(word != DRY_M, "the node holds no water");
    let lam = 4 * (i32::from(word) - sea_m);
    let mut nodes = BTreeSet::from([first]);
    let mut stack = vec![first];
    while let Some(i) = stack.pop() {
        for m in lattice.neighbours(i) {
            let r = &a.rows[m as usize];
            if m != NO_NODE && r.water_m == word && r.water_m > r.z_m && nodes.insert(m) {
                stack.push(m);
            }
        }
    }
    // The lake's id: its lowest node index (`fine::l1::Waters::lake_of`).
    let first = *nodes.first().expect("a lake node");
    let tn = TILE_NODES;
    let tile_of = |node: u32| {
        let (f, ni, nj) = lattice.split(node);
        TileKey {
            face: f,
            tx: ni.div_euclid(tn),
            ty: nj.div_euclid(tn),
        }
    };
    let tiles: BTreeSet<TileKey> = nodes
        .iter()
        .flat_map(|&i| {
            std::iter::once(i)
                .chain(lattice.neighbours(i).into_iter().filter(|&m| m != NO_NODE))
                .map(tile_of)
                .collect::<Vec<_>>()
        })
        .collect();
    let face = tiles.first().expect("a tile").face;
    assert!(
        tiles.iter().all(|t| t.face == face),
        "the bench reads a lake inside one face"
    );
    println!(
        "lake {first}: {} nodes at {} m, {} tiles; {} of its nodes end their water (no receiver)",
        nodes.len(),
        f64::from(lam) / 4.0,
        tiles.len(),
        nodes
            .iter()
            .filter(
                |&&k| a.rows[k as usize].receiver_facies & vd_terrain::artifact::RECEIVER_NONE != 0
            )
            .count()
    );
    let shift = cells_per_node_shift(body, lattice, L1_RUNG).expect("the rung nests");
    let n_r = body.ladder().cells_per_edge(L1_RUNG) as i32;
    let solved: Vec<(TileKey, vd_terrain::fine::l1::L1Tile)> = std::thread::scope(|sc| {
        let handles: Vec<_> = tiles
            .iter()
            .map(|&k| {
                let a = &a;
                sc.spawn(move || {
                    solve_l1_with(body, lattice, a, k, OutletRule::StandingWater).map(|t| (k, t))
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().expect("a solve"))
            .collect()
    });
    let row_q = |k: u32| 4 * (i32::from(a.rows[k as usize].z_m) - sea_m);
    // Each raster's own pass, by the shard's rule.
    let mut least = None::<i32>;
    for (k, t) in &solved {
        let half = BUFFER_NODES >> 1;
        let (i0, j0) = ((k.tx * tn - half) << shift, (k.ty * tn - half) << shift);
        let side = t.side;
        let node_of = |c: usize| -> Option<u32> {
            let cell = chart_cell(k.face, i0 + (c % side) as i32, j0 + (c / side) as i32, n_r)?;
            Some(lattice.index(
                Face::from_index(cell.face)?,
                cell.i >> shift,
                cell.j >> shift,
            ))
        };
        let cn: Vec<Option<u32>> = (0..side * side).map(node_of).collect();
        let cell_lake: Vec<u32> = cn
            .iter()
            .map(|n| n.filter(|n| nodes.contains(n)).map_or(NO_LAKE, |_| first))
            .collect();
        let cell_row: Vec<i32> = cn.iter().map(|n| n.map_or(i32::MAX, row_q)).collect();
        let passes = raster_passes(&t.quarters, side, &cell_lake, &cell_row, |_| lam);
        if let Some(&(_, p)) = passes.first() {
            println!(
                "  the raster of tile ({}, {}, {}): its pass {:.2} m",
                k.face.index(),
                k.tx,
                k.ty,
                f64::from(p) / 4.0
            );
            least = Some(least.map_or(p, |l: i32| l.min(p)));
        }
    }
    println!(
        "BY RASTER: the lake's fine level {:.2} m",
        f64::from(least.map_or(lam, |l| l.min(lam))) / 4.0
    );
    // The joined squares, the pass and its path.
    let s0 = (tn << shift) as usize;
    let (tx0, ty0) = (
        tiles.iter().map(|t| t.tx).min().expect("a tile"),
        tiles.iter().map(|t| t.ty).min().expect("a tile"),
    );
    let tw = (tiles.iter().map(|t| t.tx).max().expect("a tile") - tx0 + 1) as usize;
    let th = (tiles.iter().map(|t| t.ty).max().expect("a tile") - ty0 + 1) as usize;
    let (w, h) = (tw * s0, th * s0);
    let mut z = vec![i32::MIN; w * h];
    for (k, t) in &solved {
        let m = (t.side - s0) / 2;
        let (ox, oy) = ((k.tx - tx0) as usize * s0, (k.ty - ty0) as usize * s0);
        for y in 0..s0 {
            z[(oy + y) * w + ox..(oy + y) * w + ox + s0]
                .copy_from_slice(&t.quarters[(m + y) * t.side + m..(m + y) * t.side + m + s0]);
        }
    }
    let node_at = |c: usize| {
        let (x, y) = ((c % w) as i32, (c / w) as i32);
        lattice.index(
            face,
            (tx0 * (tn << shift) + x) >> shift,
            (ty0 * (tn << shift) + y) >> shift,
        )
    };
    let mut best = vec![i32::MAX; w * h];
    let mut from = vec![usize::MAX; w * h];
    let mut heap = BinaryHeap::new();
    for (c, b) in best.iter_mut().enumerate() {
        if z[c] != i32::MIN && nodes.contains(&node_at(c)) {
            *b = z[c];
            heap.push(Reverse((z[c], c)));
        }
    }
    let mut end = None;
    while let Some(Reverse((hc, c))) = heap.pop() {
        if hc > best[c] {
            continue;
        }
        if hc >= lam {
            break;
        }
        let k = node_at(c);
        if !nodes.contains(&k) && row_q(k) < lam {
            end = Some((c, hc));
            break;
        }
        let x = c % w;
        for m in [
            (x > 0).then(|| c - 1),
            (x + 1 < w).then(|| c + 1),
            (c >= w).then(|| c - w),
            (c + w < w * h).then(|| c + w),
        ]
        .into_iter()
        .flatten()
        {
            if z[m] == i32::MIN {
                continue;
            }
            let hm = hc.max(z[m]);
            if hm < best[m] {
                best[m] = hm;
                from[m] = c;
                heap.push(Reverse((hm, m)));
            }
        }
    }
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for c in 0..w * h {
        let (x, y) = (c % w, c / w);
        let v = |xx: usize, yy: usize| {
            let q = z[yy.min(h - 1) * w + xx.min(w - 1)];
            if q == i32::MIN {
                z[c] as f32 / 4.0
            } else {
                q as f32 / 4.0
            }
        };
        let shade = (0.6
            + (v(x + 1, y) - v(x.saturating_sub(1), y) + v(x, y + 1) - v(x, y.saturating_sub(1)))
                / 512.0)
            .clamp(0.15, 1.0);
        let px = if z[c] == i32::MIN {
            [0, 0, 0]
        } else if nodes.contains(&node_at(c)) && z[c] < lam {
            [30, 70, 130]
        } else {
            [
                (170.0 * shade) as u8,
                (160.0 * shade) as u8,
                (140.0 * shade) as u8,
            ]
        };
        img.put_pixel(x as u32, y as u32, image::Rgb(px));
    }
    match end {
        Some((c, pass)) => {
            let mut p = c;
            let mut top = (c, z[c]);
            while from[p] != usize::MAX {
                if z[p] > top.1 {
                    top = (p, z[p]);
                }
                img.put_pixel((p % w) as u32, (p / w) as u32, image::Rgb([230, 30, 30]));
                p = from[p];
            }
            img.put_pixel((c % w) as u32, (c / w) as u32, image::Rgb([255, 230, 0]));
            let k = node_at(c);
            println!(
                "JOINED MAP: the pass {:.2} m at cell ({}, {}) of the map (tile {}, {}); the path ends in node {k} whose row stands at {:.0} m",
                f64::from(pass) / 4.0,
                top.0 % w,
                top.0 / w,
                tx0 + (top.0 % w / s0) as i32,
                ty0 + (top.0 / w / s0) as i32,
                f64::from(row_q(k)) / 4.0
            );
        }
        None => println!("JOINED MAP: no path under the lake's level"),
    }
    let path = format!("{out}/lake_{first}_spill.png");
    img.save(&path).expect("the picture saves");
    println!("the map: {path} ({w} × {h} cells of 128 m)");
    // ★ THE RING (B115 item 1): the lake on the joined map at its 8 km level and at its fine level (the least pass by
    // raster), each spread from the lake's own cells through the cells under it, seen from one eye 4 km from the fine
    // shore on ground dry at both levels, 300 m up.
    let fine = least.map_or(lam, |l| l.min(lam));
    let spread = |level: i32| -> Vec<f32> {
        let mut wet = vec![f32::NAN; w * h];
        let mut stack: Vec<usize> = (0..w * h)
            .filter(|&c| z[c] != i32::MIN && z[c] < level && nodes.contains(&node_at(c)))
            .collect();
        for &c in &stack {
            wet[c] = level as f32 / 4.0;
        }
        while let Some(c) = stack.pop() {
            let x = c % w;
            for m in [
                (x > 0).then(|| c - 1),
                (x + 1 < w).then(|| c + 1),
                (c >= w).then(|| c - w),
                (c + w < w * h).then(|| c + w),
            ]
            .into_iter()
            .flatten()
            {
                if wet[m].is_nan() && z[m] != i32::MIN && z[m] < level {
                    wet[m] = level as f32 / 4.0;
                    stack.push(m);
                }
            }
        }
        wet
    };
    let (today, ruled) = (spread(lam), spread(fine));
    let zf: Vec<f32> = z
        .iter()
        .map(|&q| {
            if q == i32::MIN {
                f32::NAN
            } else {
                q as f32 / 4.0
            }
        })
        .collect();
    let wet_n = |v: &[f32]| v.iter().filter(|x| !x.is_nan()).count();
    println!(
        "THE RING: {} cells under water at {:.0} m, {} at {:.0} m (the joined map; today's spread there reaches lower ground)",
        wet_n(&today),
        f64::from(lam) / 4.0,
        wet_n(&ruled),
        f64::from(fine) / 4.0
    );
    // The fine shore nearest the fine water's middle, and an eye 4 km away, dry at both levels.
    let wet_cells: Vec<usize> = (0..w * h).filter(|&c| !ruled[c].is_nan()).collect();
    if wet_cells.is_empty() {
        println!("THE RING: no water at the fine level");
        return;
    }
    let (mx, my) = (
        wet_cells.iter().map(|&c| (c % w) as f64).sum::<f64>() / wet_cells.len() as f64,
        wet_cells.iter().map(|&c| (c / w) as f64).sum::<f64>() / wet_cells.len() as f64,
    );
    let shore = wet_cells
        .iter()
        .copied()
        .filter(|&c| {
            let x = c % w;
            x > 0
                && x + 1 < w
                && c >= w
                && c + w < w * h
                && [c - 1, c + 1, c - w, c + w]
                    .iter()
                    .any(|&m| ruled[m].is_nan() && z[m] != i32::MIN)
        })
        .min_by(|&p, &q| {
            let d = |c: usize| ((c % w) as f64 - mx).powi(2) + ((c / w) as f64 - my).powi(2);
            d(p).total_cmp(&d(q))
        })
        .expect("a shore");
    let (tx, ty) = ((shore % w) as f64, (shore / w) as f64);
    let dist = 4_000.0 / 128.0;
    let eye = (0..8)
        .filter_map(|i| {
            let hd = f64::from(i) * 45.0;
            let (sx, sy) = (hd.to_radians().sin(), -hd.to_radians().cos());
            let (ex, ey) = (tx - sx * dist, ty - sy * dist);
            let inside = ex >= 1.0 && ey >= 1.0 && ex < (w - 2) as f64 && ey < (h - 2) as f64;
            let c = ey as usize * w + ex as usize;
            (inside && today[c].is_nan() && z[c] != i32::MIN).then_some((hd, ex, ey, z[c]))
        })
        .max_by_key(|e| e.3);
    let Some((hd, ex, ey, _)) = eye else {
        println!("THE RING: no dry eye 4 km around the fine shore");
        return;
    };
    for (name, water) in [("today", &today), ("fine", &ruled)] {
        let path = format!("{out}/lake_{first}_side_{name}.png");
        render_view_water(&zf, Some(water), w, 128.0, (ex, ey), 300.0, hd, 8.0, &path);
        println!("  the side view at the {name} level: {path}");
    }
    // ★ FORM (a′) (B115 item 1): the lake's tiles solved again with the lake at its fine level, the spills read again, until
    // the level holds; then the same eye on the new ground.
    let mut level = fine;
    let mut ground = z.clone();
    let dry_at = |l: i32| {
        nodes
            .iter()
            .filter(|&&k| 4 * (i32::from(a.rows[k as usize].z_m) - sea_m) >= l)
            .count()
    };
    println!(
        "(a′): {} of the lake's nodes stand at or over {:.2} m (their floor dry), {} of those end their water",
        dry_at(fine),
        f64::from(fine) / 4.0,
        nodes
            .iter()
            .filter(
                |&&k| 4 * (i32::from(a.rows[k as usize].z_m) - sea_m) >= fine
                    && a.rows[k as usize].receiver_facies & vd_terrain::artifact::RECEIVER_NONE
                        != 0
            )
            .count()
    );
    for round in 1..=6 {
        let levels = BTreeMap::from([(first, level)]);
        let again: Vec<(TileKey, vd_terrain::fine::l1::L1Tile)> = std::thread::scope(|sc| {
            let handles: Vec<_> = tiles
                .iter()
                .map(|&k| {
                    let (a, levels) = (&a, &levels);
                    sc.spawn(move || {
                        vd_terrain::fine::l1::solve_l1_levels(
                            body,
                            lattice,
                            a,
                            k,
                            OutletRule::StandingWater,
                            levels,
                        )
                        .map(|t| (k, t))
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| h.join().expect("a solve"))
                .collect()
        });
        ground = vec![i32::MIN; w * h];
        for (k, t) in &again {
            let m = (t.side - s0) / 2;
            let (ox, oy) = ((k.tx - tx0) as usize * s0, (k.ty - ty0) as usize * s0);
            for y in 0..s0 {
                ground[(oy + y) * w + ox..(oy + y) * w + ox + s0]
                    .copy_from_slice(&t.quarters[(m + y) * t.side + m..(m + y) * t.side + m + s0]);
            }
        }
        let next = again
            .iter()
            .flat_map(|(_, t)| t.spills.iter())
            .filter(|&&(l, _)| l == first)
            .map(|&(_, p)| p)
            .fold(level, i32::min);
        println!(
            "(a′) round {round}: solved at {:.2} m, its spills give {:.2} m",
            f64::from(level) / 4.0,
            f64::from(next) / 4.0
        );
        if next == level {
            break;
        }
        level = next;
    }
    let zf2: Vec<f32> = ground
        .iter()
        .map(|&q| {
            if q == i32::MIN {
                f32::NAN
            } else {
                q as f32 / 4.0
            }
        })
        .collect();
    let mut wet2 = vec![f32::NAN; w * h];
    let mut stack: Vec<usize> = (0..w * h)
        .filter(|&c| ground[c] != i32::MIN && ground[c] < level && nodes.contains(&node_at(c)))
        .collect();
    for &c in &stack {
        wet2[c] = level as f32 / 4.0;
    }
    while let Some(c) = stack.pop() {
        let x = c % w;
        for m in [
            (x > 0).then(|| c - 1),
            (x + 1 < w).then(|| c + 1),
            (c >= w).then(|| c - w),
            (c + w < w * h).then(|| c + w),
        ]
        .into_iter()
        .flatten()
        {
            if wet2[m].is_nan() && ground[m] != i32::MIN && ground[m] < level {
                wet2[m] = level as f32 / 4.0;
                stack.push(m);
            }
        }
    }
    println!(
        "(a′): {} cells under water at {:.2} m",
        wet_n(&wet2),
        f64::from(level) / 4.0
    );
    let path = format!("{out}/lake_{first}_side_resolved.png");
    render_view_water(&zf2, Some(&wet2), w, 128.0, (ex, ey), 300.0, hd, 8.0, &path);
    println!("  the side view on the ground solved again: {path}");
    // From 14 km up, 40° down, over the fine shore (the owner's window view), and the flat ground's share by node: a node
    // square whose dry cells' local relief is under a metre over most of it is a flat plate.
    let high = format!("{out}/lake_{first}_high_resolved.png");
    render_view_water(
        &zf2,
        Some(&wet2),
        w,
        128.0,
        (tx - 40.0, ty - 40.0),
        14_000.0,
        135.0,
        40.0,
        &high,
    );
    println!("  the view from 14 km up: {high}");
    let node_cells = 1usize << shift;
    let mut plates = 0usize;
    for ny in (0..h).step_by(node_cells) {
        for nx in (0..w).step_by(node_cells) {
            let (mut dry, mut flat) = (0usize, 0usize);
            for y in ny + 1..(ny + node_cells).min(h - 1) {
                for x in nx + 1..(nx + node_cells).min(w - 1) {
                    let c = y * w + x;
                    if ground[c] == i32::MIN || !wet2[c].is_nan() {
                        continue;
                    }
                    dry += 1;
                    let relief = [c - 1, c + 1, c - w, c + w]
                        .iter()
                        .map(|&m| (ground[m] - ground[c]).abs())
                        .max()
                        .unwrap_or(0);
                    if relief <= 4 {
                        flat += 1;
                    }
                }
            }
            if dry > node_cells * node_cells / 2 && flat * 2 > dry {
                plates += 1;
            }
        }
    }
    println!(
        "  FLAT PLATES: {plates} node squares mostly dry and flat (within a metre of their neighbours)"
    );
    // THE ROUGHNESS (a playa is smooth, eroded land is not): the mean curvature |z(x±1) + z(y±1) − 4z| / 4 in metres over
    // the dry cells of the lake's floor left dry, against the dry cells of the nodes outside the lake.
    let (mut floor_sum, mut floor_n, mut land_sum, mut land_n) = (0f64, 0usize, 0f64, 0usize);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = y * w + x;
            let around = [c - 1, c + 1, c - w, c + w];
            if ground[c] == i32::MIN
                || !wet2[c].is_nan()
                || around.iter().any(|&m| ground[m] == i32::MIN)
            {
                continue;
            }
            let sum: i64 = around.iter().map(|&m| i64::from(ground[m])).sum();
            let lap = (sum - 4 * i64::from(ground[c])).abs() as f64 / 16.0;
            if nodes.contains(&node_at(c)) {
                floor_sum += lap;
                floor_n += 1;
            } else {
                land_sum += lap;
                land_n += 1;
            }
        }
    }
    println!(
        "  ROUGHNESS: the dry former floor {:.2} m over {floor_n} cells; the land outside the lake {:.2} m over {land_n} cells",
        floor_sum / floor_n.max(1) as f64,
        land_sum / land_n.max(1) as f64
    );
}

/// ★ THE STORED WATER (bench only; the owner's window over the basin lake, 2026-10-19): the finished 128 m tiles a realm's
/// store holds for the tiles `F,TX0,TY0,TX1,TY1` (inclusive) under the fine version `VERSION` (the stored input word), each
/// tile's water levels and their counts, and the square from above (`OUT_DIR/stored_water.png`: the ground shaded, each
/// water level its own colour, a missing tile black). Reads the store `VD_REALM_STORE` names (a copy, never the live file).
fn store_water(body: &BodyDefinition, lattice: &MacroLattice, args: &[String], planet: u64) {
    use vd_sim::io::Store as _;
    use vd_sim::stub::built_store::{FineAt, FineForm, read_fine_record};
    use vd_terrain::fine::level::{FineLevel, LevelGeom};
    use vd_terrain::fine::stitch::MISSING;
    use vd_terrain::fine::tile::{WET_FULL, decode_water_levels};
    let _ = body;
    let store = vd_bins::open_realm_store(&vd_bins::process_env(), RealmId::Planet(planet))
        .expect("the store opens")
        .expect("VD_REALM_STORE names a file");
    let head = vd_sim::stub::built_store::decode_artifact_head(
        &store
            .get(&vd_sim::stub::built_store::artifact_head_key())
            .expect("the store holds a world"),
    )
    .expect("the head decodes");
    let artifact = vd_bins::artifact_store::read_artifact(&store, head.world_tag, lattice)
        .expect("the stored world reads")
        .expect("the stored world is whole");
    let r: Vec<i32> = args
        .get(2)
        .expect("F,TX0,TY0,TX1,TY1")
        .split(',')
        .map(|x| x.trim().parse().expect("an integer"))
        .collect();
    // `VERSION` the fine version; `VERSION,L2` reads the 16 m level's tiles of that 16 m version (tile keys in nodes).
    let versions: Vec<u64> = args
        .get(3)
        .expect("VERSION[,L2_VERSION]")
        .split(',')
        .map(|v| v.parse().expect("a version"))
        .collect();
    let level = if versions.len() > 1 { FineLevel::L2 } else { FineLevel::L1 };
    let out = args.get(4).expect("OUT_DIR");
    std::fs::create_dir_all(out).expect("the output directory");
    let input = {
        use vd_seed::digest::{FNV_OFFSET, fnv1a_u64};
        let d = artifact.digest();
        let l1 = fnv1a_u64(fnv1a_u64(fnv1a_u64(FNV_OFFSET, d[0]), d[1]), versions[0]);
        match versions.get(1) {
            Some(&v2) => fnv1a_u64(l1, (u64::from(level.id()) << 32) | v2),
            None => l1,
        }
    };
    let geom = LevelGeom::of(level, body, lattice).expect("a level");
    let s0 = geom.side() as usize;
    let (tw, th) = ((r[3] - r[1] + 1) as usize, (r[4] - r[2] + 1) as usize);
    let (w, h) = (tw * s0, th * s0);
    let (mut z, mut lv) = (vec![MISSING; w * h], vec![MISSING; w * h]);
    for ty in r[2]..=r[4] {
        for tx in r[1]..=r[3] {
            let at = FineAt {
                form: FineForm::Finished,
                input,
                level: level.id(),
                face: r[0] as u8,
                tx: tx as u32,
                ty: ty as u32,
            };
            let Ok(Some(bytes)) = read_fine_record(&store, &at) else {
                println!("  tile ({}, {tx}, {ty}): none stored", r[0]);
                continue;
            };
            let Some((_, levels)) = decode_water_levels(&bytes, level) else {
                println!(
                    "  tile ({}, {tx}, {ty}): a record that does not decode",
                    r[0]
                );
                continue;
            };
            let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
            let (ox, oy) = ((tx - r[1]) as usize * s0, (ty - r[2]) as usize * s0);
            for y in 0..s0 {
                for x in 0..s0 {
                    let (src, dst) = (y * s0 + x, (oy + y) * w + ox + x);
                    z[dst] = levels.heights[0][src];
                    if levels.wet[0][src] == WET_FULL && levels.level[0][src] != MISSING {
                        lv[dst] = levels.level[0][src];
                        *counts.entry(levels.level[0][src]).or_default() += 1;
                    }
                }
            }
            let list: Vec<String> = counts
                .iter()
                .filter(|&(_, &n)| n > 200)
                .map(|(l, n)| format!("{:.1} m ×{n}", f64::from(*l) / 4.0))
                .collect();
            println!("  tile ({}, {tx}, {ty}): {}", r[0], list.join(", "));
        }
    }
    // The square kept for a closer reading: heights and water (quarter metres, `i32::MIN` where none), row by row.
    for (name, v) in [("stored_z.i32", &z), ("stored_water.i32", &lv)] {
        let bytes: Vec<u8> = v.iter().flat_map(|q| q.to_le_bytes()).collect();
        std::fs::write(format!("{out}/{name}"), bytes).expect("the dump writes");
    }
    // ★ THE WALLS: water cells beside a dry cell whose ground stands under that water's level (four neighbours).
    let mut walls = 0usize;
    for c in 0..w * h {
        if lv[c] == MISSING {
            continue;
        }
        let (x, y) = (c % w, c / w);
        for (nx, ny) in [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)] {
            if nx < w && ny < h {
                let o = ny * w + nx;
                if lv[o] == MISSING && z[o] != MISSING && z[o] < lv[c] {
                    walls += 1;
                }
            }
        }
    }
    println!(
        "the walls: {walls} water-cell sides beside dry ground under the water's level, over {} water cells",
        lv.iter().filter(|&&l| l != MISSING).count()
    );
    let mut levels: Vec<i32> = lv.iter().copied().filter(|&l| l != MISSING).collect();
    levels.sort_unstable();
    levels.dedup();
    let palette = [
        [30, 70, 130],
        [200, 60, 60],
        [60, 160, 60],
        [200, 160, 40],
        [150, 60, 180],
        [40, 170, 190],
    ];
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for c in 0..w * h {
        let (x, y) = (c % w, c / w);
        let px = if z[c] == MISSING {
            [0, 0, 0]
        } else if lv[c] != MISSING {
            palette[levels.binary_search(&lv[c]).unwrap_or(0) % palette.len()]
        } else {
            let v = |xx: usize, yy: usize| {
                let q = z[yy.min(h - 1) * w + xx.min(w - 1)];
                if q == MISSING {
                    z[c] as f32 / 4.0
                } else {
                    q as f32 / 4.0
                }
            };
            let shade = (0.6
                + (v(x + 1, y) - v(x.saturating_sub(1), y) + v(x, y + 1)
                    - v(x, y.saturating_sub(1)))
                    / 512.0)
                .clamp(0.15, 1.0);
            [
                (170.0 * shade) as u8,
                (160.0 * shade) as u8,
                (140.0 * shade) as u8,
            ]
        };
        img.put_pixel(x as u32, y as u32, image::Rgb(px));
    }
    let path = format!("{out}/stored_water.png");
    img.save(&path).expect("the picture saves");
    println!(
        "the levels' colours in order (blue, red, green, gold, violet, teal): {:?} m; {path}",
        levels
            .iter()
            .map(|l| f64::from(*l) / 4.0)
            .collect::<Vec<_>>()
    );
}

/// ★ THE GAME'S LAKE PATH ON THE BENCH (ruling B115; the shard's `fine_source` rule, step for step): the 5 × 5 tiles around
/// `F,TX,TY`, every lake in their domains, each lake's rounds from the shared plain solves (round 0) and its own solves
/// (later rounds) until its level holds, each tile's final solve with its domain's lowered lakes named, the 3 × 3 ring
/// finished with its water and stitched. Prints each lake's level and each finished tile's water levels; draws the square
/// from above (`OUT_DIR/game_water.png`: each water level its own colour) and from 14 km up (`OUT_DIR/game_high.png`).
/// ★ THE LAKE'S VIEWS FROM A BENCH SQUARE'S DUMPS (`water-game`'s `game_z.i32` and `game_water.i32`): the eye over the main
/// lake's water near its middle, facing the nearest shore and the opposite way, 400 m and 4 km over the water.
/// `planet_stand lake-views SEED DIR [DIR …]`; each DIR gets `over_*.png`, every one from the eye the LAST DIR chooses (the
/// same square under several rules, seen from one place).
fn lake_views(args: &[String]) {
    let mut chosen: Option<(f64, f64, f64)> = None;
    for dir in args[2..].iter().rev() {
        let read = |name: &str| -> Vec<i32> {
            std::fs::read(format!("{dir}/{name}"))
                .expect("the dump reads")
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let (z, lv) = (read("game_z.i32"), read("game_water.i32"));
        let w = (z.len() as f64).sqrt() as usize;
        let missing = i32::MIN;
        let mut by_level: BTreeMap<i32, (f64, f64, f64)> = BTreeMap::new();
        for (c, &l) in lv.iter().enumerate() {
            if l != missing {
                let e = by_level.entry(l).or_insert((0.0, 0.0, 0.0));
                e.0 += 1.0;
                e.1 += (c % w) as f64;
                e.2 += (c / w) as f64;
            }
        }
        let Some((&level, &(count, sx, sy))) =
            by_level.iter().max_by(|a, b| a.1.0.total_cmp(&b.1.0))
        else {
            println!("{dir}: no lake");
            continue;
        };
        // The main lake's water cell farthest from every dry cell (a walk out from the dry cells, four neighbours).
        let _ = (sx, sy);
        let mut far = vec![u32::MAX; w * w];
        let mut queue: std::collections::VecDeque<usize> = (0..w * w)
            .filter(|&c| lv[c] == missing && z[c] != missing)
            .collect();
        for &c in &queue {
            far[c] = 0;
        }
        while let Some(c) = queue.pop_front() {
            let (x, y) = (c % w, c / w);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < w && ny < w && far[ny * w + nx] == u32::MAX {
                    far[ny * w + nx] = far[c] + 1;
                    queue.push_back(ny * w + nx);
                }
            }
        }
        let eye = (0..w * w)
            .filter(|&c| lv[c] == level)
            .max_by_key(|&c| (far[c], std::cmp::Reverse(c)))
            .expect("a water cell");
        let (mut ex, mut ey) = ((eye % w) as f64, (eye / w) as f64);
        // The nearest dry cell, ring by ring.
        let mut shore = None;
        'rings: for r in 1..w as i64 {
            for k in -r..=r {
                for (dx, dy) in [(k, -r), (k, r), (-r, k), (r, k)] {
                    let (x, y) = (ex as i64 + dx, ey as i64 + dy);
                    if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < w {
                        let c = y as usize * w + x as usize;
                        if lv[c] == missing && z[c] != missing {
                            shore = Some((x as f64, y as f64));
                            break 'rings;
                        }
                    }
                }
            }
        }
        let Some((sx, sy)) = shore else {
            continue;
        };
        let mut to_shore = (sx - ex).atan2(-(sy - ey)).to_degrees();
        if let Some((x, y, h)) = chosen {
            (ex, ey, to_shore) = (x, y, h);
        } else {
            chosen = Some((ex, ey, to_shore));
        }
        let at = ey as usize * w + ex as usize;
        // The eye's height over the ground under it: over the water where water stands there, else over the ground.
        let depth = if lv[at] == missing {
            0.0
        } else {
            f64::from(lv[at] - z[at]) / 4.0
        };
        let zf: Vec<f32> = z
            .iter()
            .map(|&q| {
                if q == missing {
                    f32::NAN
                } else {
                    q as f32 / 4.0
                }
            })
            .collect();
        let wf: Vec<f32> = lv
            .iter()
            .map(|&q| {
                if q == missing {
                    f32::NAN
                } else {
                    q as f32 / 4.0
                }
            })
            .collect();
        for (name, over, heading, pitch) in [
            ("over_low_shore", 400.0, to_shore, 4.0),
            ("over_low_back", 400.0, to_shore + 180.0, 4.0),
            ("over_high_shore", 4_000.0, to_shore, 25.0),
            ("over_high_back", 4_000.0, to_shore + 180.0, 25.0),
        ] {
            let path = format!("{dir}/{name}.png");
            render_view_water(
                &zf,
                Some(&wf),
                w,
                128.0,
                (ex, ey),
                depth + over,
                heading,
                pitch,
                &path,
            );
        }
        println!(
            "{dir}: the main lake {:.1} m ({count} cells); the eye over ({ex}, {ey}), {depth:.0} m of water, the shore at ({sx}, {sy})",
            f64::from(level) / 4.0
        );
    }
}

/// ★ THE LARGEST DELTA FROM THE SIDE (ruling B121 Part 3's bench): from one square solved under the drowned rule (`OLD`) and
/// the deltas (`NEW`), the largest connected plain the deltas raised to a lake's level (more than 20 m over the old ground),
/// seen from over the water 1.5 km up and 10 km out, the same eye in both. `planet_stand delta-views SEED OLD NEW`.
fn delta_views(args: &[String]) {
    let read = |dir: &str, name: &str| -> Vec<i32> {
        std::fs::read(format!("{dir}/{name}"))
            .expect("the dump reads")
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let (old, new) = (&args[2], &args[3]);
    let (z0, z1, lv1) = (
        read(old, "game_z.i32"),
        read(new, "game_z.i32"),
        read(new, "game_water.i32"),
    );
    let w = (z1.len() as f64).sqrt() as usize;
    let missing = i32::MIN;
    // The raised plain: dry cells over 20 m higher than the old ground, beside a water at their own height.
    let raised: Vec<bool> = (0..w * w)
        .map(|c| z0[c] != missing && z1[c] != missing && lv1[c] == missing && z1[c] - z0[c] > 80)
        .collect();
    let mut comp = vec![u32::MAX; w * w];
    let mut best: (usize, f64, f64) = (0, 0.0, 0.0);
    let mut next = 0u32;
    for s0 in 0..w * w {
        if !raised[s0] || comp[s0] != u32::MAX {
            continue;
        }
        let (mut n, mut sx, mut sy) = (0usize, 0.0, 0.0);
        let mut stack = vec![s0];
        comp[s0] = next;
        while let Some(c) = stack.pop() {
            n += 1;
            sx += (c % w) as f64;
            sy += (c / w) as f64;
            let (x, y) = (c % w, c / w);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < w && ny < w && raised[ny * w + nx] && comp[ny * w + nx] == u32::MAX {
                    comp[ny * w + nx] = next;
                    stack.push(ny * w + nx);
                }
            }
        }
        if n > best.0 {
            best = (n, sx / n as f64, sy / n as f64);
        }
        next += 1;
    }
    let (n, cx, cy) = best;
    if n == 0 {
        println!("no raised plain");
        return;
    }
    // The water cell nearest the plain's middle, and the eye 10 km on past it, over the water.
    let water = (0..w * w)
        .filter(|&c| lv1[c] != missing)
        .min_by(|&a, &b| {
            let d = |c: usize| ((c % w) as f64 - cx).powi(2) + ((c / w) as f64 - cy).powi(2);
            d(a).total_cmp(&d(b))
        })
        .expect("a water cell");
    let (wx, wy) = ((water % w) as f64, (water / w) as f64);
    let (dx, dy) = (wx - cx, wy - cy);
    let len = (dx * dx + dy * dy).sqrt().max(1.0);
    let mut k = 78.0;
    let (mut ex, mut ey) = (wx, wy);
    while k > 0.0 {
        let (x, y) = (wx + dx / len * k, wy + dy / len * k);
        if x >= 0.0
            && y >= 0.0
            && (x as usize) < w
            && (y as usize) < w
            && lv1[y as usize * w + x as usize] != missing
        {
            (ex, ey) = (x, y);
            break;
        }
        k -= 4.0;
    }
    let heading = (cx - ex).atan2(-(cy - ey)).to_degrees();
    for (dir, z, lv) in [
        (old.as_str(), &z0, read(old, "game_water.i32")),
        (new.as_str(), &z1, lv1.clone()),
    ] {
        let at = ey as usize * w + ex as usize;
        let depth = if lv[at] == missing {
            0.0
        } else {
            f64::from(lv[at] - z[at]) / 4.0
        };
        let zf: Vec<f32> = z
            .iter()
            .map(|&q| {
                if q == missing {
                    f32::NAN
                } else {
                    q as f32 / 4.0
                }
            })
            .collect();
        let wf: Vec<f32> = lv
            .iter()
            .map(|&q| {
                if q == missing {
                    f32::NAN
                } else {
                    q as f32 / 4.0
                }
            })
            .collect();
        for (name, over, pitch) in [("delta_low", 400.0, 3.0), ("delta_high", 1_500.0, 9.0)] {
            render_view_water(
                &zf,
                Some(&wf),
                w,
                128.0,
                (ex, ey),
                depth + over,
                heading,
                pitch,
                &format!("{dir}/{name}.png"),
            );
        }
    }
    println!(
        "the largest raised plain: {n} cells ({:.0} km²) about ({cx:.0}, {cy:.0}); the eye over ({ex:.0}, {ey:.0}) facing {heading:.0}°",
        n as f64 * 128.0 * 128.0 / 1e6
    );
}

fn water_game(body: &BodyDefinition, lattice: &MacroLattice, args: &[String]) {
    use vd_terrain::fine::finish::block_keys;
    use vd_terrain::fine::l1::{L1Tile, OutletRule, Waters, solve_l1_levels};
    use vd_terrain::fine::level::{FineLevel, LevelGeom};
    use vd_terrain::fine::stitch::{Field, MISSING};
    use vd_terrain::fine::tile::{WET_FULL, finish_water, raw_field, water_planes};
    let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art")).expect("the world reads");
    // ★ THE DELTAS (ruling B121 Part 3): every step reads the bedrock view — the lakes' laid rock taken off their rows.
    let a = if args.get(5).map(String::as_str) == Some("deltas") {
        let bed = vd_terrain::fine::l1::bedrock_view(&a, lattice);
        let lowered = a
            .rows
            .iter()
            .zip(&bed.rows)
            .filter(|(x, y)| x.z_m != y.z_m)
            .count();
        let joined = a
            .rows
            .iter()
            .zip(&bed.rows)
            .filter(|(x, y)| x.water_m != y.water_m)
            .count();
        println!(
            "the bedrock view: {} records, {lowered} rows lowered, {joined} filled nodes given their lake's word",
            a.lakes.len()
        );
        bed
    } else {
        a
    };
    let t: Vec<i32> = args
        .get(3)
        .expect("F,TX,TY")
        .split(',')
        .map(|x| x.trim().parse().expect("an integer"))
        .collect();
    let out = args.get(4).expect("OUT_DIR");
    std::fs::create_dir_all(out).expect("the output directory");
    // The rule: the game's, or `drowned` (the lake floor design's candidate).
    let rule = match args.get(5).map(String::as_str) {
        Some("drowned") => OutletRule::Drowned,
        Some("graded") => OutletRule::Graded,
        Some("deltas") => OutletRule::Deltas,
        _ => OutletRule::StandingWater,
    };
    println!("the rule: {rule:?}");
    let key = TileKey {
        face: vd_seed::bend::Face::from_index(t[0] as u8).expect("a face"),
        tx: t[1],
        ty: t[2],
    };
    let geom = LevelGeom::of(FineLevel::L1, body, lattice).expect("a level");
    let ring: Vec<TileKey> = block_keys(key, geom).into_iter().flatten().collect();
    let all: BTreeSet<TileKey> = ring
        .iter()
        .flat_map(|&k| block_keys(k, geom).into_iter().flatten())
        .collect();
    let none = BTreeMap::new();
    let waters = Waters::new(lattice, &a, &none);
    let tile_lakes: BTreeMap<TileKey, BTreeSet<u32>> =
        all.iter().map(|&k| (k, waters.domain_lakes(k))).collect();
    let lakes: BTreeSet<u32> = tile_lakes.values().flatten().copied().collect();
    let lake_tiles: BTreeMap<u32, BTreeSet<TileKey>> =
        lakes.iter().map(|&l| (l, waters.lake_tiles(l))).collect();
    let solve_many = |keys: &[TileKey], levels: &BTreeMap<u32, i32>| -> BTreeMap<TileKey, L1Tile> {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let done = std::sync::Mutex::new(BTreeMap::new());
        let workers = std::thread::available_parallelism()
            .map_or(4, |c| c.get())
            .saturating_sub(2)
            .max(1);
        std::thread::scope(|sc| {
            for _ in 0..workers {
                sc.spawn(|| {
                    while let Some(&k) =
                        keys.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
                    {
                        if let Some(t) = solve_l1_levels(body, lattice, &a, k, rule, levels) {
                            done.lock().expect("the map").insert(k, t);
                        }
                    }
                });
            }
        });
        done.into_inner().expect("the map")
    };
    let mut plain_keys: BTreeSet<TileKey> = all.clone();
    for ts in lake_tiles.values() {
        plain_keys.extend(ts.iter().copied());
    }
    let started = std::time::Instant::now();
    let plain = solve_many(&plain_keys.iter().copied().collect::<Vec<_>>(), &none);
    println!(
        "{} lakes in the domains; {} plain solves in {:.0} s",
        lakes.len(),
        plain.len(),
        started.elapsed().as_secs_f64()
    );
    // Each lake's rounds.
    let mut fine: BTreeMap<u32, i32> = BTreeMap::new();
    for &lake in &lakes {
        let word = waters.lake_level_q(lake);
        let spill_of = |tiles: &BTreeMap<TileKey, L1Tile>| {
            lake_tiles[&lake]
                .iter()
                .filter_map(|k| tiles.get(k))
                .flat_map(|t| t.spills.iter())
                .filter(|&&(l, _)| l == lake)
                .map(|&(_, p)| p)
                .min()
        };
        let mut level = word;
        let mut next = spill_of(&plain).map_or(level, |p| p.min(level));
        let mut n = 0;
        while next != level && n < 7 {
            level = next;
            n += 1;
            let again = solve_many(
                &lake_tiles[&lake].iter().copied().collect::<Vec<_>>(),
                &BTreeMap::from([(lake, level)]),
            );
            next = spill_of(&again).map_or(level, |p| p.min(level));
        }
        let settled = next;
        if settled != word {
            fine.insert(lake, settled);
        }
        println!(
            "  lake {lake} ({} tiles): 8 km {:.2} m → {:.2} m after {n} round(s)",
            lake_tiles[&lake].len(),
            f64::from(word) / 4.0,
            f64::from(settled) / 4.0
        );
    }
    // The final solves — and, with `joint` (2026-10-19, the water wall), THE LAKES TOGETHER: a lake whose pass in a final
    // raster (every domain lake at its current level) stands under its level drains to that pass, and the finals are solved
    // again, until no final raster shows a lake passing under its level (Fill–Spill–Merge over all the lakes at once).
    let joint = args.iter().any(|a| a == "joint");
    let mut finals: BTreeMap<TileKey, L1Tile> = BTreeMap::new();
    for pass in 0..8 {
        finals.clear();
        let mut by_levels: BTreeMap<Vec<(u32, i32)>, Vec<TileKey>> = BTreeMap::new();
        for &k in &all {
            let named: Vec<(u32, i32)> = tile_lakes[&k]
                .iter()
                .filter_map(|l| fine.get(l).map(|&v| (*l, v)))
                .collect();
            if named.is_empty() {
                if let Some(t) = plain.get(&k) {
                    finals.insert(k, t.clone());
                }
            } else {
                by_levels.entry(named).or_default().push(k);
            }
        }
        for (named, keys) in &by_levels {
            let levels: BTreeMap<u32, i32> = named.iter().copied().collect();
            finals.extend(solve_many(keys, &levels));
        }
        // ★ THE SPILLS IN THE FINAL SOLVES: a lake whose pass in a final raster stands under its level would drain there.
        let mut lower: BTreeMap<u32, i32> = BTreeMap::new();
        for (k, t) in &finals {
            for &(l, p) in &t.spills {
                let settled = fine
                    .get(&l)
                    .copied()
                    .unwrap_or_else(|| waters.lake_level_q(l));
                if p < settled {
                    println!(
                        "  pass {pass}, final raster ({}, {}): lake {l} passes at {:.2} m, under its level {:.2} m",
                        k.tx,
                        k.ty,
                        f64::from(p) / 4.0,
                        f64::from(settled) / 4.0
                    );
                    let e = lower.entry(l).or_insert(p);
                    *e = (*e).min(p);
                }
            }
        }
        if !joint || lower.is_empty() {
            break;
        }
        fine.extend(lower);
    }
    for &k in &ring {
        if let Some(t) = finals.get(&k).filter(|t| !t.deltas.is_empty()) {
            println!(
                "  deltas of tile ({}, {}): {} mouths, the records' rock {:.3} km³, laid {:.3} km³, no room for {:.3} km³",
                k.tx,
                k.ty,
                t.deltas[0],
                t.deltas[1] as f64 / 1e9,
                t.deltas[2] as f64 / 1e9,
                t.deltas[3] as f64 / 1e9
            );
        }
    }
    // Finish the ring, stitch it.
    let raws: BTreeMap<TileKey, (Field, Field, Field)> = finals
        .iter()
        .map(|(&k, t)| {
            let (level, wet) = water_planes(&t.water, &t.river_share);
            (
                k,
                (
                    raw_field(k, geom, t.side, t.quarters.clone()),
                    raw_field(k, geom, t.side, level),
                    raw_field(k, geom, t.side, wet),
                ),
            )
        })
        .collect();
    let s0 = geom.side() as usize;
    let w = 3 * s0;
    let (mut z, mut lv) = (vec![MISSING; w * w], vec![MISSING; w * w]);
    for &k in &ring {
        let keys = block_keys(k, geom);
        let pick = |plane: usize| -> [Option<&Field>; 9] {
            std::array::from_fn(|i| {
                keys[i].and_then(|kk| raws.get(&kk)).map(|r| match plane {
                    0 => &r.0,
                    1 => &r.1,
                    _ => &r.2,
                })
            })
        };
        let block: [Option<(TileKey, &Field)>; 9] =
            std::array::from_fn(|i| keys[i].and_then(|kk| raws.get(&kk).map(|r| (kk, &r.0))));
        let Some(f) = finish_water(block, pick(1), pick(2), geom) else {
            println!("  tile ({}, {}): no finish", k.tx, k.ty);
            continue;
        };
        let (ox, oy) = (
            (k.tx - key.tx + 1) as usize * s0,
            (k.ty - key.ty + 1) as usize * s0,
        );
        let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
        for y in 0..s0 {
            for x in 0..s0 {
                let (src, dst) = (y * s0 + x, (oy + y) * w + ox + x);
                z[dst] = f.heights[0][src];
                if f.wet[0][src] == WET_FULL && f.level[0][src] != MISSING {
                    lv[dst] = f.level[0][src];
                    *counts.entry(f.level[0][src]).or_default() += 1;
                }
            }
        }
        let list: Vec<String> = counts
            .iter()
            .filter(|&(_, &n)| n > 200)
            .map(|(l, n)| format!("{:.1} m ×{n}", f64::from(*l) / 4.0))
            .collect();
        println!("  finished tile ({}, {}): {}", k.tx, k.ty, list.join(", "));
    }
    // The finished square kept for a closer reading: heights, water (quarter metres, `i32::MIN` where none) and each cell's
    // 8 km node, little-endian words, row by row.
    let shift = cells_per_node_shift(body, lattice, L1_RUNG).expect("the rung nests");
    let dump = |name: &str, v: &mut dyn Iterator<Item = i32>| {
        let bytes: Vec<u8> = v.flat_map(i32::to_le_bytes).collect();
        std::fs::write(format!("{out}/{name}"), bytes).expect("the dump writes");
    };
    dump("game_z.i32", &mut z.iter().copied());
    dump("game_water.i32", &mut lv.iter().copied());
    let (gx0, gy0) = (
        (key.tx - 1) * (TILE_NODES << shift),
        (key.ty - 1) * (TILE_NODES << shift),
    );
    dump(
        "game_node.i32",
        &mut (0..w * w).map(|c| {
            lattice.index(
                key.face,
                (gx0 + (c % w) as i32) >> shift,
                (gy0 + (c / w) as i32) >> shift,
            ) as i32
        }),
    );
    let mut levels: Vec<i32> = lv.iter().copied().filter(|&l| l != MISSING).collect();
    levels.sort_unstable();
    levels.dedup();
    let palette = [
        [30, 70, 130],
        [200, 60, 60],
        [60, 160, 60],
        [200, 160, 40],
        [150, 60, 180],
        [40, 170, 190],
    ];
    let mut img = image::RgbImage::new(w as u32, w as u32);
    for c in 0..w * w {
        let (x, y) = (c % w, c / w);
        let px = if z[c] == MISSING {
            [0, 0, 0]
        } else if lv[c] != MISSING {
            palette[levels.binary_search(&lv[c]).unwrap_or(0) % palette.len()]
        } else {
            let v = |xx: usize, yy: usize| {
                let q = z[yy.min(w - 1) * w + xx.min(w - 1)];
                if q == MISSING {
                    z[c] as f32 / 4.0
                } else {
                    q as f32 / 4.0
                }
            };
            let shade = (0.6
                + (v(x + 1, y) - v(x.saturating_sub(1), y) + v(x, y + 1)
                    - v(x, y.saturating_sub(1)))
                    / 512.0)
                .clamp(0.15, 1.0);
            [
                (170.0 * shade) as u8,
                (160.0 * shade) as u8,
                (140.0 * shade) as u8,
            ]
        };
        img.put_pixel(x as u32, y as u32, image::Rgb(px));
    }
    let path = format!("{out}/game_water.png");
    img.save(&path).expect("the picture saves");
    println!(
        "the levels' colours (blue, red, green, gold, violet, teal): {:?} m; {path}",
        levels
            .iter()
            .map(|l| f64::from(*l) / 4.0)
            .collect::<Vec<_>>()
    );
    let zf: Vec<f32> = z
        .iter()
        .map(|&q| {
            if q == MISSING {
                f32::NAN
            } else {
                q as f32 / 4.0
            }
        })
        .collect();
    let wf: Vec<f32> = lv
        .iter()
        .map(|&q| {
            if q == MISSING {
                f32::NAN
            } else {
                q as f32 / 4.0
            }
        })
        .collect();
    let high = format!("{out}/game_high.png");
    let mid = (w / 2) as f64;
    render_view_water(
        &zf,
        Some(&wf),
        w,
        128.0,
        (mid - 80.0, mid - 80.0),
        14_000.0,
        135.0,
        35.0,
        &high,
    );
    println!("the view from 14 km up: {high}");
    // The tiles south of the lake (the square's lower part), from 14 km up, facing along the rows.
    let south = format!("{out}/game_high_south.png");
    render_view_water(
        &zf,
        Some(&wf),
        w,
        128.0,
        (mid + 190.0, mid + 250.0),
        14_000.0,
        180.0,
        40.0,
        &south,
    );
    println!("the view south from 14 km up: {south}");
    // Low views of two small lakes north of the basin lake (the four-pointed one and the rounded one on today's map), each
    // from 3 km south-west, 1 500 m up.
    for (name, (tx, ty)) in [("star", (680.0, 690.0)), ("round", (640.0, 950.0))] {
        let (ex, ey): (f64, f64) = (tx - 17.0, ty + 17.0);
        let heading = (tx - ex).atan2(-(ty - ey)).to_degrees();
        let path = format!("{out}/game_lake_{name}.png");
        render_view_water(
            &zf,
            Some(&wf),
            w,
            128.0,
            (ex, ey),
            1_500.0,
            heading,
            25.0,
            &path,
        );
        println!("the {name} lake from 1 500 m: {path}");
    }
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
        Some("maps") => {
            let artifact = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art"))
                .expect("the world reads");
            let window = (args.len() >= 9).then(|| {
                [5, 6, 7, 8].map(|k| args[k].parse::<f64>().expect("LAT0 LAT1 LON0 LON1"))
            });
            world_maps(
                &lattice,
                body.radius_m(),
                &artifact,
                args.get(3).expect("OUT_DIR"),
                args.get(4).expect("TAG"),
                window,
            );
        }
        Some("moon-with") => {
            // ★ A BENCH READING, NEVER A WORLD: the home moon's small lattice solved by the game's worker with PLANET's
            // words, so a climate rule is read in minutes before a planet's solve spends hours on it.
            let cores: usize = args.get(2).map_or(8, |s| s.parse().expect("cores"));
            let started = std::time::Instant::now();
            let artifact = vd_bins::artifact_worker::run_solve_on(
                &vd_bins::artifact_worker::SolveJob {
                    body: vd_terrain::home::home_moon(),
                    words: vd_bins::solve_words_of(&charter),
                },
                vd_terrain::par::Cores(cores),
            )
            .expect("the moon solves");
            let mut c: Vec<f64> = artifact
                .rows
                .iter()
                .map(|r| row_celsius(r.temperature))
                .collect();
            c.sort_by(f64::total_cmp);
            println!(
                "the home moon with Planet({planet})'s words, {:.1} s: the temperature rows {:.1} / {:.1} / {:.1} °C (coldest / median / warmest), the charter's surface {:?} mK",
                started.elapsed().as_secs_f64(),
                c[0],
                c[c.len() / 2],
                c[c.len() - 1],
                vd_bins::solve_words_of(&charter).t_surface_mk
            );
        }
        Some("diff") => {
            // ★ TWO WORLDS OF ONE BODY, ROW FIELD BY ROW FIELD: how many nodes differ in each, and the sea.
            let a = vd_bins::artifact_file::read(args.get(2).expect("A.art")).expect("A reads");
            let b = vd_bins::artifact_file::read(args.get(3).expect("B.art")).expect("B reads");
            assert_eq!(a.rows.len(), b.rows.len(), "one body's lattice");
            let count = |f: &dyn Fn(&vd_terrain::artifact::Row) -> i32| {
                a.rows
                    .iter()
                    .zip(&b.rows)
                    .filter(|(x, y)| f(x) != f(y))
                    .count()
            };
            println!(
                "{} nodes; differ: height {}, water {}, receiver and facies {}, discharge {}, temperature {}, rain {}, dryness {}, province {}; the sea {} / {} m; digests [{:#x}, {:#x}] / [{:#x}, {:#x}]",
                a.rows.len(),
                count(&|r| i32::from(r.z_m)),
                count(&|r| i32::from(r.water_m)),
                count(&|r| i32::from(r.receiver_facies)),
                count(&|r| i32::from(r.discharge_log)),
                count(&|r| i32::from(r.temperature)),
                count(&|r| i32::from(r.rain)),
                count(&|r| i32::from(r.aridity)),
                count(&|r| i32::from(r.province)),
                a.sea_m,
                b.sea_m,
                a.digest()[0],
                a.digest()[1],
                b.digest()[0],
                b.digest()[1]
            );
        }
        Some("rivers") => {
            // ★ THE LARGEST RIVERS' MOUTHS (B114; slice W1's carve): each land node whose receiver is the sea, by its
            // discharge (m³/s over the body's own year), with its 128 m tile, for `water1`.
            let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art"))
                .expect("the world reads");
            let year_s = vd_bins::solve_words_of(&charter).year_s as f64;
            let sea = |i: usize| {
                (a.rows[i].receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
                    & vd_terrain::solve::FACIES_SEA
                    != 0
            };
            let mut mouths: Vec<(f64, u32)> = (0..a.rows.len())
                .filter(|&i| !sea(i))
                .filter_map(|i| {
                    let f = a.rows[i].receiver_facies;
                    if f & vd_terrain::artifact::RECEIVER_NONE != 0 {
                        return None;
                    }
                    let r = lattice.neighbours(i as u32)
                        [(f & vd_terrain::artifact::RECEIVER_SLOT_MASK) as usize];
                    (r != vd_terrain::macro_lattice::NO_NODE && sea(r as usize)).then(|| {
                        let q = vd_terrain::fine::l1::decode_class(a.rows[i].discharge_log, 2)
                            as f64
                            / 1_000.0
                            / year_s;
                        (q, i as u32)
                    })
                })
                .collect();
            mouths.sort_by(|x, y| y.0.total_cmp(&x.0));
            for (q, k) in mouths.iter().take(12) {
                let (f, ni, nj) = lattice.split(*k);
                let tn = vd_terrain::fine::l1::TILE_NODES;
                println!(
                    "  a mouth of {q:.0} m³/s at node {k}: the 128 m tile {},{},{}",
                    f.index(),
                    ni / tn,
                    nj / tn
                );
            }
        }
        Some("lakes") => {
            // ★ G-OUTLETS (the one-volume revision §3.2, Q1): every lake — a connected patch of nodes whose water stands
            // over their ground and which are not the sea — CLOSED when none of its nodes passes water on (no receiver),
            // SPILLING when one does; their counts and areas.
            use vd_terrain::artifact::{DRY_M, FACIES_SHIFT, RECEIVER_NONE};
            let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art"))
                .expect("the world reads");
            let n = a.rows.len();
            let sea = |i: usize| {
                (a.rows[i].receiver_facies >> FACIES_SHIFT) & vd_terrain::solve::FACIES_SEA != 0
            };
            let lake = |i: usize| {
                let r = &a.rows[i];
                !sea(i) && r.water_m != DRY_M && r.water_m > r.z_m
            };
            let mut seen = vec![false; n];
            let (mut closed, mut spilling) = ((0usize, 0f64), (0usize, 0f64));
            let mut biggest_spilling: Vec<(f64, usize)> = Vec::new();
            // By kind (closed, spilling), size (1, 2–9, 10+ nodes) and deepest water (<1, 1–9, 10–99, 100+ m).
            let mut classes = [[[0usize; 4]; 3]; 2];
            // Each lake's first shore node and its 128 m tile, for the water's bench (`water1`).
            // (spilling, area m², nodes, deepest m, level m, the shore's 128 m tile)
            type Shore = (bool, f64, usize, i32, i16, (u8, i32, i32));
            let mut shores: Vec<Shore> = Vec::new();
            for start in 0..n {
                if seen[start] || !lake(start) {
                    continue;
                }
                let (mut stack, mut area, mut nodes, mut spills) =
                    (vec![start], 0f64, 0usize, false);
                let mut deepest = 0i32;
                let mut shore: Option<u32> = None;
                seen[start] = true;
                while let Some(i) = stack.pop() {
                    area += lattice.area_m2(i as u32) as f64;
                    nodes += 1;
                    deepest = deepest.max(i32::from(a.rows[i].water_m) - i32::from(a.rows[i].z_m));
                    spills |= a.rows[i].receiver_facies & RECEIVER_NONE == 0;
                    if shore.is_none()
                        && lattice.neighbours(i as u32).iter().any(|&m| {
                            m != vd_terrain::macro_lattice::NO_NODE
                                && !lake(m as usize)
                                && !sea(m as usize)
                        })
                    {
                        shore = Some(i as u32);
                    }
                    for m in lattice.neighbours(i as u32) {
                        if m != vd_terrain::macro_lattice::NO_NODE
                            && !seen[m as usize]
                            && lake(m as usize)
                        {
                            seen[m as usize] = true;
                            stack.push(m as usize);
                        }
                    }
                }
                let size = [2usize, 10].iter().filter(|&&b| nodes >= b).count();
                let deep = [1i32, 10, 100].iter().filter(|&&b| deepest >= b).count();
                classes[usize::from(spills)][size][deep] += 1;
                if let Some(k) = shore {
                    let (f, ni, nj) = lattice.split(k);
                    let tn = vd_terrain::fine::l1::TILE_NODES;
                    shores.push((
                        spills,
                        area,
                        nodes,
                        deepest,
                        a.rows[k as usize].water_m,
                        (f.index(), ni / tn, nj / tn),
                    ));
                }
                if spills {
                    spilling.0 += 1;
                    spilling.1 += area;
                    biggest_spilling.push((area, nodes));
                } else {
                    closed.0 += 1;
                    closed.1 += area;
                }
            }
            biggest_spilling.sort_by(|x, y| y.0.total_cmp(&x.0));
            let land: f64 = (0..n)
                .filter(|&i| !sea(i))
                .map(|i| lattice.area_m2(i as u32) as f64)
                .sum();
            println!(
                "lakes: {} closed ({:.0} km², {:.3} % of the land), {} spilling ({:.0} km², {:.3} % of the land); the largest spilling: {:?}",
                closed.0,
                closed.1 / 1e6,
                100.0 * closed.1 / land,
                spilling.0,
                spilling.1 / 1e6,
                100.0 * spilling.1 / land,
                biggest_spilling
                    .iter()
                    .take(5)
                    .map(|(km2, nodes)| format!("{:.0} km² ({nodes} nodes)", km2 / 1e6))
                    .collect::<Vec<_>>()
            );
            for (k, kind) in ["closed", "spilling"].iter().enumerate() {
                for (z, size) in ["1 node", "2–9 nodes", "10+ nodes"].iter().enumerate() {
                    println!(
                        "  {kind}, {size}: deepest under 1 m {}, 1–9 m {}, 10–99 m {}, 100 m or more {}",
                        classes[k][z][0], classes[k][z][1], classes[k][z][2], classes[k][z][3]
                    );
                }
            }
            shores.sort_by(|x, y| y.1.total_cmp(&x.1));
            for kind in [true, false] {
                for (sp, area, nodes, deepest, level, (f, tx, ty)) in
                    shores.iter().filter(|s| s.0 == kind).take(4)
                {
                    println!(
                        "  {} lake {:.0} km² ({nodes} nodes, deepest {deepest} m, its level {level} m): a shore in the 128 m tile {f},{tx},{ty}",
                        if *sp { "spilling" } else { "closed" },
                        area / 1e6
                    );
                }
            }
        }
        Some("water1") => {
            // ★ B99 STEP 4, SLICE W1 (bench only): the 128 m tile `F,TX,TY` solved under today's outlets and under each
            // standing water's (B105 Q1 (b)), each filled by the same first water map — every water from its own 8 km
            // nodes through the raster cells under its level (the sea at its level, each lake at its word) — with the
            // readings and side views of the shore.
            use vd_terrain::fine::l1::{
                BUFFER_NODES, L1_RUNG, OutletRule, TILE_NODES, cells_per_node_shift, solve_l1_with,
            };
            let artifact = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art"))
                .expect("the world reads");
            let tile: Vec<i32> = args
                .get(3)
                .expect("F,TX,TY")
                .split(',')
                .map(|x| x.trim().parse().expect("an integer"))
                .collect();
            let out = args.get(4).expect("OUT_DIR");
            std::fs::create_dir_all(out).expect("the output directory");
            let face = vd_seed::bend::Face::from_index(tile[0] as u8).expect("a face");
            let key = TileKey {
                face,
                tx: tile[1],
                ty: tile[2],
            };
            let started = std::time::Instant::now();
            let solved: Vec<vd_terrain::fine::l1::L1Tile> = std::thread::scope(|sc| {
                let hs: Vec<_> = [OutletRule::Today, OutletRule::StandingWater]
                    .into_iter()
                    .map(|rule| {
                        let (body, lattice, artifact) = (&body, &lattice, &artifact);
                        sc.spawn(move || {
                            solve_l1_with(body, lattice, artifact, key, rule)
                                .expect("the tile solves")
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().expect("a solve")).collect()
            });
            println!(
                "the 128 m tile {},{},{} solved under both rules in {:.1} s",
                tile[0],
                tile[1],
                tile[2],
                started.elapsed().as_secs_f64()
            );
            let shift = cells_per_node_shift(&body, &lattice, L1_RUNG).expect("nests");
            let n_r = body.ladder().cells_per_edge(L1_RUNG) as i32;
            let half = BUFFER_NODES >> 1;
            let (i0, j0) = (
                (key.tx * TILE_NODES - half) << shift,
                (key.ty * TILE_NODES - half) << shift,
            );
            let side = solved[0].side;
            let sea_m = artifact.sea_m;
            // Each raster cell's 8 km water: the sea at nought, a lake at its word over the sea, or none.
            let level_of = |node: usize| -> Option<f32> {
                let r = &artifact.rows[node];
                if (r.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
                    & vd_terrain::solve::FACIES_SEA
                    != 0
                {
                    Some(0.0)
                } else if r.water_m != vd_terrain::artifact::DRY_M && r.water_m > r.z_m {
                    Some(f32::from(r.water_m) - f32::from(sea_m))
                } else {
                    None
                }
            };
            let cell_level: Vec<Option<f32>> = (0..side * side)
                .map(|k| {
                    let (a, b) = ((k % side) as i32, (k / side) as i32);
                    chart_cell(face, i0 + a, j0 + b, n_r).and_then(|c| {
                        let f = vd_seed::bend::Face::from_index(c.face)?;
                        level_of(lattice.index(f, c.i >> shift, c.j >> shift) as usize)
                    })
                })
                .collect();
            let cell = f64::from(1u32 << L1_RUNG);
            for (name, t) in ["today", "standing_water"].iter().zip(&solved) {
                let z: Vec<f32> = t
                    .quarters
                    .iter()
                    .map(|&q| {
                        if q == i32::MIN {
                            f32::NAN
                        } else {
                            q as f32 / 4.0
                        }
                    })
                    .collect();
                // The first water map: every level spreads from its own cells under it through the cells under it; then
                // the carved rivers' water at their banks (the standing-water rule's tile only).
                let mut water = vec![f32::NAN; side * side];
                let river_cells = t.river.iter().filter(|&&r| r != i32::MIN).count();
                println!("  {name}: {river_cells} cells of carved river");
                let mut levels: Vec<f32> = cell_level.iter().flatten().copied().collect();
                levels.sort_by(f32::total_cmp);
                levels.dedup();
                for &lam in &levels {
                    let mut stack: Vec<usize> = (0..side * side)
                        .filter(|&k| cell_level[k] == Some(lam) && z[k] < lam)
                        .collect();
                    let mut seen = vec![false; side * side];
                    for &k in &stack {
                        seen[k] = true;
                    }
                    while let Some(k) = stack.pop() {
                        if water[k].is_nan() || water[k] < lam {
                            water[k] = lam;
                        }
                        let (x, y) = (k % side, k / side);
                        for m in [
                            (x > 0).then(|| k - 1),
                            (x + 1 < side).then(|| k + 1),
                            (y > 0).then(|| k - side),
                            (y + 1 < side).then(|| k + side),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            if !seen[m] && z[m] < lam {
                                seen[m] = true;
                                stack.push(m);
                            }
                        }
                    }
                }
                for (k, &r) in t.river.iter().enumerate() {
                    if r != i32::MIN {
                        let bank = r as f32 / 4.0;
                        if water[k].is_nan() || water[k] < bank {
                            water[k] = bank;
                        }
                    }
                }
                // The tile's OWN water map (the terrain crate's, `fine::water`), where the rule made one: it must equal the
                // bench's flood above cell for cell.
                if t.water.iter().any(|&w| w != vd_terrain::fine::water::DRY) {
                    let differ = (0..side * side)
                        .filter(|&k| {
                            let own = t.water[k];
                            let bench = water[k];
                            (own == vd_terrain::fine::water::DRY) != bench.is_nan()
                                || (own != vd_terrain::fine::water::DRY
                                    && (own as f32 / 4.0 - bench).abs() > 0.01)
                        })
                        .count();
                    println!(
                        "  {name}: the tile's own water map differs from the bench's flood in {differ} cells"
                    );
                    for (w, &own) in water.iter_mut().zip(&t.water) {
                        *w = if own == vd_terrain::fine::water::DRY {
                            f32::NAN
                        } else {
                            own as f32 / 4.0
                        };
                    }
                }
                // FROM ABOVE: the tile's ground shaded from the north-west, each wet cell blue (a carved river bright).
                {
                    let mut img = image::RgbImage::new(side as u32, side as u32);
                    for y in 0..side {
                        for x in 0..side {
                            let k = y * side + x;
                            let at = |xx: usize, yy: usize| {
                                let v = z[yy.min(side - 1) * side + xx.min(side - 1)];
                                if v.is_finite() { v } else { z[k] }
                            };
                            let dx = at(x + 1, y) - at(x.saturating_sub(1), y);
                            let dy = at(x, y + 1) - at(x, y.saturating_sub(1));
                            let shade = (0.6 + (dx + dy) / (4.0 * cell as f32)).clamp(0.15, 1.0);
                            let c = if t.river[k] != i32::MIN {
                                [40, 120, 255]
                            } else if water[k].is_finite() && water[k] > z[k] {
                                [30, 60, 110]
                            } else {
                                // A narrower river: its share of the cell, a tint over the shaded ground.
                                let a = f32::from(t.river_share[k]) / 255.0;
                                let ground = [170.0 * shade, 160.0 * shade, 140.0 * shade];
                                let blue = [40.0, 120.0, 255.0];
                                [0, 1, 2].map(|i| (ground[i] * (1.0 - a) + blue[i] * a) as u8)
                            };
                            img.put_pixel(x as u32, y as u32, image::Rgb(c));
                        }
                    }
                    img.save(format!("{out}/{name}_above.png"))
                        .expect("the picture saves");
                }
                // The readings: the lake cells (a lake node's cells) under their level that hold water (a floor) and
                // that do not (the ground cut under the level, the water drained over the rim), and the deepest water.
                let lake_cell = |k: usize| cell_level[k].is_some_and(|l| l != 0.0);
                let (mut held, mut cut, mut over, mut deepest) = (0usize, 0usize, 0usize, 0f32);
                for k in 0..side * side {
                    if !lake_cell(k) || !z[k].is_finite() {
                        continue;
                    }
                    let lam = cell_level[k].expect("a lake cell");
                    if water[k].is_finite() {
                        held += 1;
                        deepest = deepest.max(water[k] - z[k]);
                    } else if z[k] < lam {
                        cut += 1;
                    } else {
                        over += 1;
                    }
                }
                let wet = water.iter().filter(|w| w.is_finite()).count();
                println!(
                    "  {name:15}: {} of {} cells wet; the lake nodes' cells: {held} hold their water (deepest {deepest:.1} m), {cut} under the level and dry, {over} over the level; passes {:?}",
                    wet,
                    side * side,
                    t.passes
                );
                // The views: from a dry cell beside the water nearest the middle, toward the water's middle.
                let mid = side as f64 / 2.0;
                let wet_mid = {
                    let (mut sx, mut sy, mut nw) = (0f64, 0f64, 0f64);
                    for (k, w) in water.iter().enumerate() {
                        if w.is_finite() {
                            sx += (k % side) as f64;
                            sy += (k / side) as f64;
                            nw += 1.0;
                        }
                    }
                    (nw > 0.0).then(|| (sx / nw, sy / nw))
                };
                let eye = (0..side * side)
                    .filter(|&k| z[k].is_finite() && !water[k].is_finite())
                    .filter(|&k| {
                        let (x, y) = (k % side, k / side);
                        [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)]
                            .iter()
                            .any(|&(dx, dy)| {
                                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                                nx >= 0
                                    && ny >= 0
                                    && (nx as usize) < side
                                    && (ny as usize) < side
                                    && water[ny as usize * side + nx as usize].is_finite()
                            })
                    })
                    .min_by(|&a, &b| {
                        let d = |k: usize| {
                            let (x, y) = ((k % side) as f64 - mid, (k / side) as f64 - mid);
                            x * x + y * y
                        };
                        d(a).total_cmp(&d(b))
                    });
                if let (Some(e), Some((wx, wy))) = (eye, wet_mid) {
                    let (ex, ey) = ((e % side) as f64 + 0.5, (e / side) as f64 + 0.5);
                    // Heading: 0 up the rows (−y), 90 along them (+x).
                    let heading = (wx - ex).atan2(-(wy - ey)).to_degrees();
                    // ★ THE STAND FOR THE WINDOW: the tile's main LAKE (its most common level other than the sea's), a dry
                    // cell beside it nearest its middle as the eye, its middle as the target — unit directions in the
                    // planet's frame and the eye's ground over the sea, for a spawn pose.
                    if name == &"standing_water" {
                        let unit = |a: f64, b: f64| -> Option<[f64; 3]> {
                            let c = chart_cell(face, i0 + a as i32, j0 + b as i32, n_r)?;
                            let inv_n = vd_recipe::bend::inv_n_of(n_r as u32);
                            let d = vd_recipe::bend::direction(i32::from(c.face), c.i, c.j, inv_n);
                            let v = [0, 1, 2].map(|k| d[k].raw() as f64);
                            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                            Some(v.map(|x| x / l))
                        };
                        let mut counts: std::collections::BTreeMap<i64, usize> = Default::default();
                        for w in water.iter().filter(|w| w.is_finite() && **w != 0.0) {
                            *counts.entry((*w * 4.0) as i64).or_default() += 1;
                        }
                        if let Some((&lq, _)) = counts.iter().max_by_key(|(_, n)| **n) {
                            let lam = lq as f32 / 4.0;
                            let is_lake =
                                |k: usize| water[k].is_finite() && (water[k] - lam).abs() < 0.3;
                            let (mut sx, mut sy, mut nl) = (0f64, 0f64, 0f64);
                            for k in (0..side * side).filter(|&k| is_lake(k)) {
                                sx += (k % side) as f64;
                                sy += (k / side) as f64;
                                nl += 1.0;
                            }
                            // The lake's most open water: the lake cell farthest (four-neighbour steps) from any cell that is not
                            // this lake, by one breadth-first pass from all of those.
                            let mut dist = vec![u32::MAX; side * side];
                            let mut queue = std::collections::VecDeque::new();
                            for (k, d) in dist.iter_mut().enumerate() {
                                if !is_lake(k) {
                                    *d = 0;
                                    queue.push_back(k);
                                }
                            }
                            while let Some(k) = queue.pop_front() {
                                let (x, y) = (k % side, k / side);
                                for (nx, ny) in [
                                    (x.wrapping_sub(1), y),
                                    (x + 1, y),
                                    (x, y.wrapping_sub(1)),
                                    (x, y + 1),
                                ] {
                                    if nx < side && ny < side {
                                        let m = ny * side + nx;
                                        if dist[m] == u32::MAX {
                                            dist[m] = dist[k] + 1;
                                            queue.push_back(m);
                                        }
                                    }
                                }
                            }
                            let open = (0..side * side)
                                .max_by_key(|&k| (dist[k], std::cmp::Reverse(k)))
                                .unwrap_or(0);
                            let (lx, ly) = ((open % side) as f64 + 0.5, (open / side) as f64 + 0.5);
                            println!(
                                "  the lake's most open water: {} cells from any shore",
                                dist[open]
                            );
                            let _ = (sx, sy);
                            let shore = (0..side * side)
                                .filter(|&k| z[k].is_finite() && !water[k].is_finite())
                                .filter(|&k| {
                                    let (x, y) = (k % side, k / side);
                                    [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)].iter().any(
                                        |&(dx, dy)| {
                                            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                                            nx >= 0
                                                && ny >= 0
                                                && (nx as usize) < side
                                                && (ny as usize) < side
                                                && is_lake(ny as usize * side + nx as usize)
                                        },
                                    )
                                })
                                .min_by(|&a, &b| {
                                    let d = |k: usize| {
                                        let (x, y) =
                                            ((k % side) as f64 - lx, (k / side) as f64 - ly);
                                        x * x + y * y
                                    };
                                    d(a).total_cmp(&d(b))
                                });
                            if let Some(e2) = shore {
                                let (sx2, sy2) =
                                    ((e2 % side) as f64 + 0.5, (e2 / side) as f64 + 0.5);
                                if let (Some(eye_d), Some(target_d)) =
                                    (unit(sx2, sy2), unit(lx, ly))
                                {
                                    println!(
                                        "  STAND: the lake at {lam:.1} m over the sea ({nl} cells); eye {eye_d:?} ground {:.1} m; target {target_d:?}",
                                        z[e2]
                                    );
                                }
                            }
                        }
                    }
                    for (view, eye_m, pitch) in [
                        ("walker", 2.0, 2.0),
                        ("low", 150.0, 6.0),
                        ("high", 2_000.0, 20.0),
                    ] {
                        render_view_water(
                            &z,
                            Some(&water),
                            side,
                            cell,
                            (ex, ey),
                            eye_m,
                            heading,
                            pitch,
                            &format!("{out}/{name}_{view}.png"),
                        );
                    }
                }
            }
            println!("pictures in {out}");
        }
        Some("words") => println!("{:?}", vd_bins::solve_words_of(&charter)),
        Some("store-check") => {
            // ★ THE LIVE REFUSAL HUNT (slice W2): the stored finished record of the 128 m tile `F,TX,TY` read back, then
            // replayed as the gateway serves it — one level a record, the coarsest first — into a book made as the
            // client makes it (the head's edge and sea), each step reported.
            use vd_sim::io::Store as _;
            use vd_sim::stub::built_store::{FineAt, FineForm, read_fine_record};
            use vd_terrain::fine::level::FineLevel;
            use vd_terrain::fine::read::FineBook;
            use vd_terrain::fine::tile::{
                decode_water_levels, decode_water_range, input_word_of, water_range_record,
            };
            let store = vd_bins::open_realm_store(&vd_bins::process_env(), RealmId::Planet(planet))
                .expect("the store opens")
                .expect("VD_REALM_STORE names a file");
            let head = vd_sim::stub::built_store::decode_artifact_head(
                &store
                    .get(&vd_sim::stub::built_store::artifact_head_key())
                    .expect("the store holds a world"),
            )
            .expect("the head decodes");
            let artifact = vd_bins::artifact_store::read_artifact(&store, head.world_tag, &lattice)
                .expect("the stored world reads")
                .expect("the stored world is whole");
            let tile: Vec<u32> = args
                .get(2)
                .expect("F,TX,TY")
                .split(',')
                .map(|x| x.trim().parse().expect("an integer"))
                .collect();
            let at = FineAt {
                form: FineForm::Finished,
                input: input_word_of(FineLevel::L1, artifact.digest()),
                level: FineLevel::L1.id(),
                face: tile[0] as u8,
                tx: tile[1],
                ty: tile[2],
            };
            let words = vd_sim::stub::built_store::fine_words(&store, FineForm::Finished);
            println!(
                "this build's input word {:#x}; the stored finished words {:x?}",
                at.input, words
            );
            // The finished tiles stored under this word near the asked tile (keys: family, word, level, face, tx, ty).
            let (from, to) =
                vd_sim::stub::built_store::fine_word_range(FineForm::Finished, at.input);
            let rows = store.range(&from, &to, 1_000_000);
            let mut tiles = std::collections::BTreeSet::new();
            for (k, _) in &rows {
                if k.len() >= 19 {
                    let tx = u32::from_be_bytes([k[11], k[12], k[13], k[14]]);
                    let ty = u32::from_be_bytes([k[15], k[16], k[17], k[18]]);
                    tiles.insert((k[9], k[10], tx, ty));
                }
            }
            let l1: Vec<_> = tiles.iter().filter(|t| t.0 == 1).collect();
            let l2 = tiles.iter().filter(|t| t.0 != 1).count();
            let nearest = l1.iter().min_by_key(|t| {
                (i64::from(t.2) - i64::from(tile[1])).pow(2)
                    + (i64::from(t.3) - i64::from(tile[2])).pow(2)
                    + if u32::from(t.1) == tile[0] {
                        0
                    } else {
                        1_000_000
                    }
            });
            println!(
                "{} part rows under this word: {} 128 m tiles, {} 16 m tiles; the 128 m tile nearest: {nearest:?}",
                rows.len(),
                l1.len(),
                l2
            );
            let record = read_fine_record(&store, &at);
            println!(
                "the stored record: {:?}",
                record.as_ref().map(|r| r.as_ref().map(Vec::len))
            );
            let Ok(Some(record)) = record else { return };
            println!(
                "its form byte {}, count {}, ts {}",
                record[0], record[1], record[2]
            );
            let whole = decode_water_levels(&record, FineLevel::L1);
            println!(
                "decoded whole: {:?}",
                whole.as_ref().map(|(ts, l)| (*ts, l.heights.len()))
            );
            let Some((ts, _)) = whole else { return };
            let mut book = FineBook::with_side(FineLevel::L1, lattice.edge, ts, artifact.sea_m)
                .expect("a client's book");
            let key = TileKey {
                face: vd_seed::bend::Face::from_index(tile[0] as u8).expect("a face"),
                tx: tile[1] as i32,
                ty: tile[2] as i32,
            };
            let top = FineLevel::L1.mean_levels() + 1;
            for lo in (0..top).rev() {
                let range = water_range_record(&record, lo, lo + 1);
                let dec = range
                    .as_ref()
                    .and_then(|r| decode_water_range(r, FineLevel::L1));
                let held = dec
                    .clone()
                    .map(|(_, lo2, lv)| book.insert_water_from(key, lo2, lv));
                println!(
                    "level {lo}: cut {:?} bytes, decoded {:?}, held {held:?}",
                    range.as_ref().map(Vec::len),
                    dec.as_ref()
                        .map(|(t, l, lv)| (*t, *l, lv.heights.len(), lv.heights[0].len()))
                );
            }
        }
        Some("store-get") => {
            // ★ A STORED WORLD OUT TO A FILE, read under the key its own head names, so an older generation's world
            // reads too (the "before" of a world change).
            use vd_sim::io::Store as _;
            let store = vd_bins::open_realm_store(&vd_bins::process_env(), RealmId::Planet(planet))
                .expect("the store opens")
                .expect("VD_REALM_STORE names a file");
            let head = vd_sim::stub::built_store::decode_artifact_head(
                &store
                    .get(&vd_sim::stub::built_store::artifact_head_key())
                    .expect("the store holds a world"),
            )
            .expect("the head decodes");
            let artifact = vd_bins::artifact_store::read_artifact(&store, head.world_tag, &lattice)
                .expect("the stored world reads")
                .expect("the stored world is whole");
            let out = args.get(2).expect("OUT.art");
            vd_bins::artifact_file::write(out, &artifact).expect("the world writes");
            let d = artifact.digest();
            println!(
                "the world under key {:#x} (this build's key {:#x}): digest [{:#x}, {:#x}], the sea {} m; written to {out}",
                head.world_tag,
                vd_terrain::artifact_world_tag(UNIVERSE),
                d[0],
                d[1],
                artifact.sea_m
            );
        }
        Some("store-put") => {
            // ★ A SOLVED WORLD INTO A REALM'S STORE under this build's key, by the shard's own writer
            // (`artifact_store::write_artifact`), then read back as the shard reads it at its boot.
            let mut store =
                vd_bins::open_realm_store(&vd_bins::process_env(), RealmId::Planet(planet))
                    .expect("the store opens")
                    .expect("VD_REALM_STORE names a file");
            let artifact = vd_bins::artifact_file::read(args.get(2).expect("IN.art"))
                .expect("the world reads");
            let tag = vd_terrain::artifact_world_tag(UNIVERSE);
            vd_bins::artifact_store::write_artifact(&mut store, tag, &artifact);
            // The store commits behind the call, so a read in the same opening can still see the older head: closed and
            // opened again, as a shard opens it at its boot.
            drop(store);
            let store = vd_bins::open_realm_store(&vd_bins::process_env(), RealmId::Planet(planet))
                .expect("the store opens again")
                .expect("VD_REALM_STORE names a file");
            let back = vd_bins::artifact_store::read_artifact(&store, tag, &lattice)
                .expect("the stored world reads")
                .expect("the stored world is whole");
            assert_eq!(
                back.digest(),
                artifact.digest(),
                "the store gives back the world it took"
            );
            let d = artifact.digest();
            println!(
                "stored under key {tag:#x}: digest [{:#x}, {:#x}], the sea {} m",
                d[0], d[1], artifact.sea_m
            );
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
        Some("valley") => {
            // ★ A VALLEY FLOOR IN DAYLIGHT (2026-10-14, the owner: "put me into some valleys"): in the tile `F,TX,TY`'s
            // solved 128 m ground, the land cell with the highest walls on both sides of an axis over the ground's change
            // along it — the walls the lowest of the four samples across the axis at 768 m and 1 536 m, less the larger
            // change 768 m along it either way; the spawn on the floor looking along the valley's way down, and a hull
            // 60 m that way as a size reference, nose along the valley.
            use vd_core::glam::{DMat3, DQuat, DVec3};
            let world = args.get(2).expect("WORLD.art");
            let tile: Vec<i32> = args
                .get(3)
                .expect("F,TX,TY")
                .split(',')
                .map(|x| x.trim().parse().expect("an integer"))
                .collect();
            let artifact = vd_bins::artifact_file::read(world).expect("the world reads");
            let key = TileKey {
                face: vd_seed::bend::Face::from_index(tile[0] as u8).expect("a face"),
                tx: tile[1],
                ty: tile[2],
            };
            let l1 = solve_l1(&body, &lattice, &artifact, key).expect("the tile solves");
            let shift = cells_per_node_shift(&body, &lattice, L1_RUNG).expect("nests") as i32;
            let (w, cells) = (l1.side as i32, 1i32 << shift);
            let n7 = (lattice.edge as i32) << shift;
            let (i0, j0) = (
                (key.tx * TILE_NODES - (BUFFER_NODES >> 1)) << shift,
                (key.ty * TILE_NODES - (BUFFER_NODES >> 1)) << shift,
            );
            let h = |x: i32, y: i32| f64::from(l1.quarters[(y * w + x) as usize]) / 4.0;
            let sea = f64::from(artifact.sea_m);
            {
                let mut v: Vec<i32> = l1.quarters.clone();
                v.sort_unstable();
                println!(
                    "the raster: side {w}, {} values; quarters min {} median {} max {}; the sea {sea} m",
                    v.len(),
                    v[0],
                    v[v.len() / 2],
                    v[v.len() - 1]
                );
            }
            let ring = |x: i32, y: i32, r: f64| -> Vec<(f64, f64)> {
                (0..16)
                    .map(|k| {
                        let a = f64::from(k) * std::f64::consts::TAU / 16.0;
                        let (sx, sy) = (
                            x + (r * a.cos()).round() as i32,
                            y + (r * a.sin()).round() as i32,
                        );
                        (a, h(sx, sy))
                    })
                    .collect()
            };
            let inner = (BUFFER_NODES >> 1) * cells;
            let mut best: Option<(f64, i32, i32)> = None;
            let mut top: Vec<(f64, i32, i32, f64, f64, f64)> = Vec::new();
            for y in inner..inner + TILE_NODES * cells {
                for x in inner..inner + TILE_NODES * cells {
                    // The raster stands over the sea.
                    let z = h(x, y);
                    if z <= 2.0 {
                        continue;
                    }
                    let (near, far) = (ring(x, y, 6.0), ring(x, y, 12.0));
                    for k in 0..8usize {
                        let along = [near[k].1 - z, near[k + 8].1 - z];
                        let walls = [
                            near[k + 4].1,
                            near[(k + 12) % 16].1,
                            far[k + 4].1,
                            far[(k + 12) % 16].1,
                        ]
                        .iter()
                        .map(|w| w - z)
                        .fold(f64::MAX, f64::min);
                        let depth = walls - along[0].abs().max(along[1].abs());
                        if best.is_none_or(|b| depth > b.0) {
                            best = Some((depth, x, y));
                        }
                        top.push((depth, x, y, walls, along[0], along[1]));
                    }
                }
            }
            top.sort_by(|a, b| b.0.total_cmp(&a.0));
            top.dedup_by_key(|t| (t.1, t.2));
            for t in top.iter().take(5) {
                println!(
                    "  candidate ({}, {}): score {:.0} m, walls {:.0} m, along {:.0} / {:.0} m",
                    t.1, t.2, t.0, t.3, t.4, t.5
                );
            }
            let (depth, x, y) = best.expect("a land cell");
            let z = h(x, y);
            let rim = ring(x, y, 6.0);
            let &(down_a, down_z) = rim
                .iter()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("a sample");
            let high = rim.iter().map(|s| s.1).fold(f64::MIN, f64::max);
            println!(
                "tile {key:?}: the valley floor at raster cell ({x}, {y}), {z:.1} m over the sea; its walls {depth:.0} m over it at the least; at 768 m the ground {:.0} m (the way down) to {:.0} m",
                down_z - z,
                high - z
            );
            let to_dir = |gx: f64, gy: f64| {
                let a = 2.0 * gx / f64::from(n7) - 1.0;
                let b = 2.0 * gy / f64::from(n7) - 1.0;
                DVec3::from_array(vd_seed::bend::direction(key.face, a, b))
            };
            let (gx, gy) = (f64::from(i0 + x) + 0.5, f64::from(j0 + y) + 0.5);
            let up = to_dir(gx, gy);
            // The check: the middle node's direction by this mapping against the lattice's own.
            let half = TILE_NODES / 2;
            let g = lattice.direction(lattice.index(
                key.face,
                key.tx * TILE_NODES + half,
                key.ty * TILE_NODES + half,
            ));
            let s40 = (1u64 << 40) as f64;
            let node = DVec3::new(
                g[0].raw() as f64 / s40,
                g[1].raw() as f64 / s40,
                g[2].raw() as f64 / s40,
            )
            .normalize();
            let mid = f64::from(((key.tx * TILE_NODES + half) << shift) + cells / 2);
            let mid_y = f64::from(((key.ty * TILE_NODES + half) << shift) + cells / 2);
            println!(
                "the cell mapping's check: the middle node {:.2} m off the lattice's own direction",
                to_dir(mid, mid_y).angle_between(node) * body.radius_m()
            );
            // The way down the valley, on the ground's plane.
            let ahead = to_dir(gx + down_a.cos(), gy + down_a.sin());
            let level = (ahead - up * up.dot(ahead)).normalize();
            let right = level.cross(up).normalize();
            let orient = DQuat::from_mat3(&DMat3::from_cols(right, up, -level)).normalize();
            let r = body.radius_m() + sea + z;
            let eye = up * (r + 1.7);
            let hull_up = (up * r + level * 60.0).normalize();
            let hull = hull_up * (r + 4.0);
            let orbit = vd_bins::home_orbit(UNIVERSE).expect("the home planet's orbit");
            let sun = -vd_physics::celestial::orbital_state(&orbit, 0.0)
                .position
                .normalize();
            println!(
                "the star {:.1}° up at the floor",
                up.dot(sun).clamp(-1.0, 1.0).asin().to_degrees()
            );
            println!(
                "SPAWN 1000=Planet({}):{},{},{}@{},{},{},{}",
                body.seed(),
                eye.x,
                eye.y,
                eye.z,
                orient.x,
                orient.y,
                orient.z,
                orient.w
            );
            println!(
                "BERTH --berth-x-m {} --berth-y-m {} --berth-z-m {} --berth-facing-x {} --berth-facing-y {} --berth-facing-z {} --berth-facing-w {}",
                hull.x, hull.y, hull.z, orient.x, orient.y, orient.z, orient.w
            );
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
        Some("codec") => {
            // ★ THE WATER RECORD'S ROUND TRIP AT THE GAME'S SIZE (slice W2's refusal hunt): a 128 m tile of a lake (its
            // level a quarter metre word far under the sea) and dry hills, encoded, cut, decoded and held — each step
            // reported, so a refusal names its step. No solve.
            use vd_terrain::fine::level::FineLevel;
            use vd_terrain::fine::read::FineBook;
            use vd_terrain::fine::stitch::{Field, MISSING, mean_levels};
            use vd_terrain::fine::tile::{
                WET_FULL, WaterLevels, decode_water_range, encode_water_levels, water_mean_levels,
                water_range_record,
            };
            let geom = vd_terrain::fine::level::LevelGeom::of(FineLevel::L1, &body, &lattice)
                .expect("a level");
            let side = 1usize << geom.ts;
            let heights: Vec<i32> = (0..side * side)
                .map(|c| {
                    let (x, y) = ((c % side) as i32, (c / side) as i32);
                    if x < 200 {
                        -8_400 + y
                    } else {
                        -7_000 + 3 * x + (y % 17) * 40
                    }
                })
                .collect();
            let level: Vec<i32> = heights
                .iter()
                .map(|&h| if h < -8_008 { -8_008 } else { MISSING })
                .collect();
            let wet: Vec<i32> = level
                .iter()
                .enumerate()
                .map(|(c, &l)| {
                    if l != MISSING {
                        WET_FULL
                    } else if c % side == 300 {
                        40
                    } else {
                        0
                    }
                })
                .collect();
            let f = Field {
                gx0: 0,
                gy0: 0,
                w: side,
                h: side,
                v: heights.clone(),
            };
            let n = FineLevel::L1.mean_levels();
            let (lv, wt) = water_mean_levels(&level, &wet, side, n);
            let mut levels = WaterLevels {
                heights: vec![heights],
                level: vec![level],
                wet: vec![wet],
            };
            levels
                .heights
                .extend(mean_levels(&f, 0, 0, side, n).expect("the means"));
            levels.level.extend(lv);
            levels.wet.extend(wt);
            let rec = encode_water_levels(&levels, geom.ts);
            println!(
                "encoded: {:?} bytes (ts {})",
                rec.as_ref().map(Vec::len),
                geom.ts
            );
            let rec = rec.expect("a record");
            let key = TileKey {
                face: vd_seed::bend::Face::from_index(0).expect("a face"),
                tx: 1,
                ty: 1,
            };
            for lo in [0u32, 3] {
                let range = water_range_record(&rec, lo, n + 1);
                let dec = range
                    .as_ref()
                    .and_then(|r| decode_water_range(r, FineLevel::L1));
                let mut book = FineBook::new(FineLevel::L1, &body, &lattice, artifact_sea(&body))
                    .expect("a book");
                let book_ts = book.side_shift();
                let held = dec
                    .clone()
                    .map(|(_, lo2, lv)| book.insert_water_from(key, lo2, lv));
                println!(
                    "range {lo}..{}: cut {:?} bytes, decoded {:?}, held {held:?} (the book's side shift {book_ts})",
                    n + 1,
                    range.as_ref().map(Vec::len),
                    dec.as_ref()
                        .map(|(ts, lo2, lv)| (*ts, *lo2, lv.heights.len()))
                );
            }
        }
        Some("water2") => {
            // ★ SLICE W2 ON THE BENCH (B114): the 128 m tiles around `F,TX,TY` solved under the standing-water rule (the
            // base level, the outlets, the carve, the water map), their middles finished WITH THEIR WATER as the planet's
            // shard will (each finished cell the water of the raster its height came from), held in a book with it, then
            // the 2 × 2 block of 16 m tiles at the tile's middle under the same rule — each drawn from above: the ground
            // shaded, each standing water dark blue, a carved river bright, a narrower river's share a tint.
            use vd_terrain::fine::finish::block_keys;
            use vd_terrain::fine::l1::{OutletRule, solve_l1_with};
            use vd_terrain::fine::l2::solve_l2_with;
            use vd_terrain::fine::read::FineBook;
            use vd_terrain::fine::stitch::{Field, MISSING};
            use vd_terrain::fine::tile::{WET_FULL, finish_water, raw_field, water_planes};
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
            let geom = vd_terrain::fine::level::LevelGeom::of(
                vd_terrain::fine::level::FineLevel::L1,
                &body,
                &lattice,
            )
            .expect("a level");
            let ring: Vec<TileKey> = block_keys(key, geom).into_iter().flatten().collect();
            let mut all: Vec<TileKey> = ring
                .iter()
                .flat_map(|&k| block_keys(k, geom).into_iter().flatten())
                .collect();
            all.sort();
            all.dedup();
            let started = std::time::Instant::now();
            type Raw = (Field, Field, Field);
            let raws: std::collections::BTreeMap<TileKey, Raw> = std::thread::scope(|sc| {
                let handles: Vec<_> = all
                    .iter()
                    .map(|&k| {
                        let (body, lattice, artifact) = (&body, &lattice, &artifact);
                        sc.spawn(move || {
                            solve_l1_with(body, lattice, artifact, k, OutletRule::StandingWater)
                                .map(|t| {
                                    let (level, wet) = water_planes(&t.water, &t.river_share);
                                    (
                                        k,
                                        (
                                            raw_field(k, geom, t.side, t.quarters),
                                            raw_field(k, geom, t.side, level),
                                            raw_field(k, geom, t.side, wet),
                                        ),
                                    )
                                })
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|h| h.join().expect("a solve"))
                    .collect()
            });
            println!(
                "{} 128 m tiles solved in {:.1} s",
                raws.len(),
                started.elapsed().as_secs_f64()
            );
            let mut book = FineBook::new(
                vd_terrain::fine::level::FineLevel::L1,
                &body,
                &lattice,
                artifact.sea_m,
            )
            .expect("a book");
            let mut middle = None;
            for &k in &ring {
                let keys = block_keys(k, geom);
                let pick = |plane: usize| -> [Option<&Field>; 9] {
                    std::array::from_fn(|i| {
                        keys[i].and_then(|kk| raws.get(&kk)).map(|r| match plane {
                            0 => &r.0,
                            1 => &r.1,
                            _ => &r.2,
                        })
                    })
                };
                let block: [Option<(TileKey, &Field)>; 9] = std::array::from_fn(|i| {
                    keys[i].and_then(|kk| raws.get(&kk).map(|r| (kk, &r.0)))
                });
                let levels = finish_water(block, pick(1), pick(2), geom).expect("a tile finishes");
                if k == key {
                    middle = Some(levels.clone());
                }
                assert!(
                    book.insert_water_from(k, 0, levels),
                    "the finished tile is held"
                );
            }
            // FROM ABOVE: a square of heights, levels and shares (quarter metres over the sea).
            let above = |name: &str,
                         side: usize,
                         z: &[i32],
                         level: &[i32],
                         wet: &[i32],
                         cell: f32| {
                let mut img = image::RgbImage::new(side as u32, side as u32);
                for y in 0..side {
                    for x in 0..side {
                        let k = y * side + x;
                        let at = |xx: usize, yy: usize| {
                            let v = z[yy.min(side - 1) * side + xx.min(side - 1)];
                            if v == MISSING {
                                z[k] as f32
                            } else {
                                v as f32 / 4.0
                            }
                        };
                        let dx = at(x + 1, y) - at(x.saturating_sub(1), y);
                        let dy = at(x, y + 1) - at(x, y.saturating_sub(1));
                        let shade = (0.6 + (dx + dy) / (4.0 * cell)).clamp(0.15, 1.0);
                        let ground = [170.0 * shade, 160.0 * shade, 140.0 * shade];
                        let c = if wet[k] == WET_FULL && level[k] != MISSING {
                            [30, 70, 130]
                        } else {
                            let a = wet[k] as f32 / 255.0;
                            let blue = [40.0, 120.0, 255.0];
                            [0, 1, 2].map(|i| (ground[i] * (1.0 - a) + blue[i] * a) as u8)
                        };
                        img.put_pixel(x as u32, y as u32, image::Rgb(c));
                    }
                }
                img.save(format!("{out}/{name}"))
                    .expect("the picture saves");
                let wet_cells = (0..side * side).filter(|&k| wet[k] == WET_FULL).count();
                let tinted = (0..side * side)
                    .filter(|&k| wet[k] > 0 && wet[k] < WET_FULL)
                    .count();
                println!(
                    "  {name}: {wet_cells} cells under water, {tinted} with a narrower river's share"
                );
            };
            let m = middle.expect("the middle tile");
            let s0 = geom.side() as usize;
            above(
                "tile_128m_above.png",
                s0,
                &m.heights[0],
                &m.level[0],
                &m.wet[0],
                128.0,
            );
            // The 2 × 2 block of 16 m tiles at the tile's middle.
            let mid = TILE_NODES / 2;
            let nodes: Vec<(i32, i32)> = [(-1, -1), (0, -1), (-1, 0), (0, 0)]
                .iter()
                .map(|(dx, dy)| {
                    (
                        key.tx * TILE_NODES + mid + dx,
                        key.ty * TILE_NODES + mid + dy,
                    )
                })
                .collect();
            let started = std::time::Instant::now();
            let solved: Vec<Option<vd_terrain::fine::l2::L2Tile>> = std::thread::scope(|sc| {
                let handles: Vec<_> = nodes
                    .iter()
                    .map(|&nd| {
                        let (body, lattice, artifact, book) = (&body, &lattice, &artifact, &book);
                        sc.spawn(move || {
                            solve_l2_with(
                                body,
                                lattice,
                                artifact,
                                book,
                                TileKey {
                                    face: key.face,
                                    tx: nd.0,
                                    ty: nd.1,
                                },
                                OutletRule::StandingWater,
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("a solve"))
                    .collect()
            });
            println!(
                "the four 16 m tiles solved in {:.1} s",
                started.elapsed().as_secs_f64()
            );
            // The four tiles' own squares (the raster less half its buffer each side) side by side.
            if let Some(t0) = solved.first().and_then(Option::as_ref) {
                let shift4 = vd_terrain::fine::l1::cells_per_node_shift(
                    &body,
                    &lattice,
                    vd_terrain::fine::l2::L2_RUNG,
                )
                .expect("nests") as usize;
                let node = 1usize << shift4;
                let half = (t0.side - node) / 2;
                let big = 2 * node;
                let (mut z, mut level, mut wet) = (
                    vec![MISSING; big * big],
                    vec![MISSING; big * big],
                    vec![0i32; big * big],
                );
                for (q, t) in solved.iter().enumerate() {
                    let Some(t) = t else {
                        println!("  a 16 m tile did not solve");
                        continue;
                    };
                    let (ox, oy) = ((q & 1) * node, (q >> 1) * node);
                    for y in 0..node {
                        for x in 0..node {
                            let src = (half + y) * t.side + half + x;
                            let dst = (oy + y) * big + ox + x;
                            z[dst] = t.quarters[src];
                            level[dst] = t.water[src];
                            wet[dst] = if t.water[src] != MISSING {
                                WET_FULL
                            } else {
                                i32::from(t.river_share[src])
                            };
                        }
                    }
                }
                above("tiles_16m_above.png", big, &z, &level, &wet, 16.0);
            }
        }
        Some("water-gates") => water_gates(&body, &lattice, &args),
        Some("l1-probe") => l1_probe(&body, &lattice, &args),
        Some("spill-count") => spill_count(&body, &lattice, &args),
        Some("spill-full") => spill_full(&body, &lattice, &args),
        // ★ WHERE A POINT IS (2026-10-19, the rung pictures): the 8 km node and the 128 m tile under a point of the
        // planet's frame, moved `east_km` and `north_km` along the ground (a nadir picture's offset from its centre).
        // `planet_stand where SEED X,Y,Z [EAST_KM NORTH_KM]`.
        Some("where") => {
            let p: Vec<f64> = args[2]
                .split(',')
                .map(|v| v.parse().expect("a coordinate"))
                .collect();
            let km = |k: usize| args.get(k).map_or(0.0, |v| v.parse::<f64>().expect("km"));
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let u = [p[0] / r, p[1] / r, p[2] / r];
            let e = [-u[1], u[0], 0.0];
            let el = (e[0] * e[0] + e[1] * e[1]).sqrt();
            let e = [e[0] / el, e[1] / el, 0.0];
            let n = [
                u[1] * e[2] - u[2] * e[1],
                u[2] * e[0] - u[0] * e[2],
                u[0] * e[1] - u[1] * e[0],
            ];
            let radius = body.radius_m();
            let (de, dn) = (km(3) * 1_000.0 / radius, km(4) * 1_000.0 / radius);
            let d: Vec<f64> = (0..3).map(|k| u[k] + de * e[k] + dn * n[k]).collect();
            let dl = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let d = [d[0] / dl, d[1] / dl, d[2] / dl];
            let node =
                vd_bins::drainage_gates::node_under(&lattice, vd_core::glam::DVec3::from_array(d));
            let (face, i, j) = lattice.split(node);
            let shift = vd_terrain::fine::level::FineLevel::L1.tile_nodes_shift();
            println!(
                "direction {d:?}: node {node} = {face:?} ({i}, {j}); the 128 m tile {} ({}, {})",
                face as u8,
                i >> shift,
                j >> shift
            );
        }
        Some("dry-ends") => {
            // The 8 km world's DRY ENDS (no water, no receiver): how many, and how they cluster (eight neighbours) — one pit
            // a hollow, or every node of a hollow.
            use vd_terrain::artifact::{DRY_M, FACIES_SHIFT, RECEIVER_NONE};
            let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art"))
                .expect("the world reads");
            let n = a.rows.len();
            let dry_end = |i: usize| {
                let r = &a.rows[i];
                (r.receiver_facies >> FACIES_SHIFT) & vd_terrain::solve::FACIES_SEA == 0
                    && !(r.water_m != DRY_M && r.water_m > r.z_m)
                    && r.receiver_facies & RECEIVER_NONE != 0
            };
            let mut seen = vec![false; n];
            let mut sizes = Vec::new();
            for start in 0..n {
                if seen[start] || !dry_end(start) {
                    continue;
                }
                seen[start] = true;
                let (mut stack, mut size) = (vec![start as u32], 0usize);
                while let Some(i) = stack.pop() {
                    size += 1;
                    for m in lattice.neighbours(i) {
                        if m != NO_NODE && !seen[m as usize] && dry_end(m as usize) {
                            seen[m as usize] = true;
                            stack.push(m);
                        }
                    }
                }
                sizes.push(size);
            }
            sizes.sort_unstable();
            let total: usize = sizes.iter().sum();
            let multi = sizes.iter().filter(|&&s| s > 1).count();
            let in_multi: usize = sizes.iter().filter(|&&s| s > 1).sum();
            println!(
                "{total} dry end nodes in {} clusters; {multi} clusters of more than one node hold {in_multi}; the largest {:?}",
                sizes.len(),
                sizes.iter().rev().take(5).collect::<Vec<_>>()
            );
        }
        // ★ THE 8 km ROWS AROUND A NODE (2026-10-19, the water wall): each node's row, water word and lake id in a square of
        // `2·R + 1` nodes on the node's face. `planet_stand rows SEED WORLD.art NODE [R]`.
        Some("rows") => {
            let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art"))
                .expect("the world reads");
            let node: u32 = args.get(3).expect("NODE").parse().expect("a node");
            let r: i32 = args.get(4).map_or(3, |v| v.parse().expect("a radius"));
            let ids = vd_terrain::fine::spill::lake_ids(&lattice, &a);
            let (face, i0, j0) = lattice.split(node);
            println!(
                "rows around {node} = {face:?} ({i0}, {j0}), sea {} m: z / water / lake",
                a.sea_m
            );
            for j in (j0 - r)..=(j0 + r) {
                let line: Vec<String> = ((i0 - r)..=(i0 + r))
                    .map(|i| {
                        let n = lattice.index(face, i, j) as usize;
                        let row = &a.rows[n];
                        let w = if row.water_m == vd_terrain::artifact::DRY_M {
                            "-".to_string()
                        } else {
                            row.water_m.to_string()
                        };
                        let l = if ids[n] == vd_terrain::fine::spill::NO_LAKE {
                            ".".to_string()
                        } else {
                            ids[n].to_string()
                        };
                        format!("{:>5}/{:>5}/{:>7}", row.z_m, w, l)
                    })
                    .collect();
                println!("j {j}: {}", line.join(" "));
            }
        }
        Some("lake-nodes") => {
            // How many 8 km nodes lie under a lake, and how many more stand beside one (the far water's list).
            use vd_terrain::artifact::{DRY_M, FACIES_SHIFT};
            let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art"))
                .expect("the world reads");
            let lake = |i: usize| {
                let r = &a.rows[i];
                (r.receiver_facies >> FACIES_SHIFT) & vd_terrain::solve::FACIES_SEA == 0
                    && r.water_m != DRY_M
                    && r.water_m > r.z_m
            };
            let under = (0..a.rows.len()).filter(|&i| lake(i)).count();
            let beside = (0..a.rows.len())
                .filter(|&i| {
                    !lake(i)
                        && lattice
                            .neighbours(i as u32)
                            .iter()
                            .any(|&m| m != NO_NODE && lake(m as usize))
                })
                .count();
            println!("{under} nodes under a lake, {beside} beside one");
        }
        Some("spill-lake") => spill_lake(&body, &lattice, &args),
        Some("store-water") => store_water(&body, &lattice, &args, planet),
        Some("water-game") => water_game(&body, &lattice, &args),
        Some("lake-views") => lake_views(&args),
        Some("delta-views") => delta_views(&args),
        Some("cell-dir") => {
            // ★ A BENCH SQUARE'S CELL AS A DIRECTION: `cell-dir PLANET F,TX,TY X,Y` — the unit direction of cell (X, Y) of the
            // 3 × 3 square of 128 m tiles centred on tile (TX, TY) of face F (`water-game`'s dumps), for a window's stand.
            let t: Vec<i32> = args[2].split(',').map(|v| v.parse().expect("an integer")).collect();
            let c: Vec<i32> = args[3].split(',').map(|v| v.parse().expect("an integer")).collect();
            let rung = vd_terrain::fine::l1::L1_RUNG;
            let shift = vd_terrain::fine::l1::cells_per_node_shift(&body, &lattice, rung).expect("the rung nests");
            let n_r = body.ladder().cells_per_edge(rung) as i32;
            let tn = vd_terrain::fine::l1::TILE_NODES;
            let (gi, gj) = ((t[1] - 1) * (tn << shift) + c[0], (t[2] - 1) * (tn << shift) + c[1]);
            let cell = vd_terrain::fine::points::chart_cell(
                vd_seed::bend::Face::from_index(t[0] as u8).expect("a face"),
                gi,
                gj,
                n_r,
            )
            .expect("a cell");
            let g = vd_recipe::bend::direction(i32::from(cell.face), cell.i, cell.j, vd_recipe::bend::inv_n_of(n_r as u32));
            let s = (1u64 << 40) as f64;
            let up = vd_core::glam::DVec3::new(g[0].raw() as f64 / s, g[1].raw() as f64 / s, g[2].raw() as f64 / s)
                .normalize();
            println!("{:?},{:?},{:?}", up.x, up.y, up.z);
        }
        Some("l2") => {
            // ★ THE 16 m LEVEL ON THE BENCH (ruling B95): the 128 m tiles around the tile `F,TX,TY` solved and its middle
            // finished as the planet's shard does, then a 2 × 2 block of 16 m tiles at its middle under them, beside
            // today's ground (the 128 m curve at the same 16 m cells): the numbers and the views.
            use vd_terrain::fine::finish::block_keys;
            use vd_terrain::fine::l2::{L2_RUNG, solve_l2};
            use vd_terrain::fine::read::{FineBook, FineZ};
            use vd_terrain::fine::tile::{finish, raw_field};
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
            let geom = vd_terrain::fine::level::LevelGeom::of(
                vd_terrain::fine::level::FineLevel::L1,
                &body,
                &lattice,
            )
            .expect("a level");
            // The 3 × 3 tiles around `key` finished, each from its own nine raw tiles (25 solves in all).
            let ring: Vec<TileKey> = block_keys(key, geom).into_iter().flatten().collect();
            let mut all: Vec<TileKey> = ring
                .iter()
                .flat_map(|&k| block_keys(k, geom).into_iter().flatten())
                .collect();
            all.sort();
            all.dedup();
            let started = std::time::Instant::now();
            let raws: std::collections::BTreeMap<TileKey, vd_terrain::fine::stitch::Field> =
                std::thread::scope(|sc| {
                    let handles: Vec<_> = all
                        .iter()
                        .map(|&k| {
                            let (body, lattice, artifact) = (&body, &lattice, &artifact);
                            sc.spawn(move || {
                                solve_l1(body, lattice, artifact, k)
                                    .map(|t| (k, raw_field(k, geom, t.side, t.quarters)))
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .filter_map(|h| h.join().expect("a solve"))
                        .collect()
                });
            println!(
                "{} 128 m tiles solved in {:.1} s",
                raws.len(),
                started.elapsed().as_secs_f64()
            );
            let mut book = FineBook::new(
                vd_terrain::fine::level::FineLevel::L1,
                &body,
                &lattice,
                artifact.sea_m,
            )
            .expect("a book");
            for &k in &ring {
                let keys = block_keys(k, geom);
                let block: [Option<(TileKey, &vd_terrain::fine::stitch::Field)>; 9] =
                    std::array::from_fn(|i| keys[i].and_then(|kk| raws.get(&kk).map(|f| (kk, f))));
                let levels = finish(block, geom).expect("a tile finishes");
                assert!(book.insert(k, levels), "the finished tile is held");
            }
            println!("{} finished 128 m tiles held", book.len());
            // The 2 × 2 block of 16 m tiles at the tile's middle (its flow region stays inside the finished tile).
            let mid = TILE_NODES / 2;
            let nodes: Vec<(i32, i32)> = [(-1, -1), (0, -1), (-1, 0), (0, 0)]
                .iter()
                .map(|(dx, dy)| {
                    (
                        key.tx * TILE_NODES + mid + dx,
                        key.ty * TILE_NODES + mid + dy,
                    )
                })
                .collect();
            let started = std::time::Instant::now();
            let solved: Vec<Option<vd_terrain::fine::l2::L2Tile>> = std::thread::scope(|sc| {
                let handles: Vec<_> = nodes
                    .iter()
                    .map(|&nd| {
                        let (body, lattice, artifact, book) = (&body, &lattice, &artifact, &book);
                        sc.spawn(move || {
                            let t0 = std::time::Instant::now();
                            let t = solve_l2(body, lattice, artifact, book, TileKey { face: key.face, tx: nd.0, ty: nd.1 });
                            println!(
                                "  the 16 m tile at node {nd:?}: {:.1} s{}",
                                t0.elapsed().as_secs_f64(),
                                t.as_ref().map_or(" — DID NOT SOLVE".to_owned(), |t| format!(
                                    ", {} points, {} passes, the law's detail median / largest {:.2} / {:.2} m",
                                    t.points, t.passes, t.detail_cm[0] as f64 / 100.0, t.detail_cm[1] as f64 / 100.0
                                ))
                            );
                            t
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("a solve"))
                    .collect()
            });
            println!(
                "the four 16 m tiles solved in {:.1} s (four threads)",
                started.elapsed().as_secs_f64()
            );
            // Each tile's bytes as one word (FNV-1a over its quarters), so two builds of the solve can be compared.
            for (t, nd) in solved.iter().zip(&nodes) {
                if let Some(t) = t {
                    let mut d = 0xcbf2_9ce4_8422_2325u64;
                    for q in &t.quarters {
                        for b in q.to_le_bytes() {
                            d = (d ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
                        }
                    }
                    println!(
                        "  DIGEST node {nd:?}: {d:#018x} ({} samples)",
                        t.quarters.len()
                    );
                }
            }
            // The 16 m square: each tile's own node square (512 × 512 of its 640 × 640) laid side by side.
            let shift4 = cells_per_node_shift(&body, &lattice, L2_RUNG).expect("nests");
            let node4 = 1usize << shift4;
            let w = 2 * node4;
            let mut l2 = vec![f32::NAN; w * w];
            for (t, (dx, dy)) in solved.iter().zip([(0, 0), (1, 0), (0, 1), (1, 1)]) {
                let Some(t) = t else { continue };
                let off = (t.side - node4) / 2;
                for y in 0..node4 {
                    for x in 0..node4 {
                        let q = t.quarters[(y + off) * t.side + x + off];
                        if q != i32::MIN {
                            l2[(dy * node4 + y) * w + dx * node4 + x] = q as f32 / 4.0;
                        }
                    }
                }
            }
            // Today's ground at the same cells: the 128 m curve read at rung 4.
            let n4 = body.ladder().cells_per_edge(L2_RUNG) as i32;
            let (i0, j0) = (
                (nodes[0].0 as usize) << shift4,
                (nodes[0].1 as usize) << shift4,
            );
            let sea = f64::from(artifact.sea_m);
            let mut l1 = vec![f32::NAN; w * w];
            for y in 0..w {
                for x in 0..w {
                    if let Some(c) = chart_cell(key.face, (i0 + x) as i32, (j0 + y) as i32, n4)
                        && let FineZ::Held(z) = book.z(
                            vd_seed::bend::Face::from_index(c.face).expect("a face"),
                            L2_RUNG,
                            L2_RUNG,
                            c.i,
                            c.j,
                        )
                    {
                        let m = z.raw() as f64
                            / (vd_terrain::units::STEPS_PER_M << vd_terrain::units::LENGTH_BITS)
                                as f64;
                        l1[y * w + x] = (m - sea) as f32;
                    }
                }
            }
            let cell = lattice.node_m() / node4 as f64;
            for (name, z) in [("today's 128 m curve", &l1), ("the 16 m ground", &l2)] {
                let mut v: Vec<f32> = z.iter().copied().filter(|x| x.is_finite()).collect();
                v.sort_by(f32::total_cmp);
                if !v.is_empty() {
                    println!(
                        "  {name}: {} cells, lowest {:.1} m, median {:.1} m, highest {:.1} m over the sea",
                        v.len(),
                        v[0],
                        v[v.len() / 2],
                        v[v.len() - 1]
                    );
                }
            }
            // ★ "DO NOT BREAK TODAY'S GROUND": each 128 m cell's mean of the 16 m ground against the same cell's mean of
            // today's 128 m curve (8 × 8 cells of 16 m a 128 m cell), the gap's median, 99th and largest.
            let k = 8usize;
            let mut gaps: Vec<f32> = Vec::new();
            for by in 0..w / k {
                for bx in 0..w / k {
                    let (mut a, mut b, mut n) = (0f64, 0f64, 0u32);
                    for y in by * k..(by + 1) * k {
                        for x in bx * k..(bx + 1) * k {
                            let (u, v) = (l2[y * w + x], l1[y * w + x]);
                            if u.is_finite() && v.is_finite() {
                                a += f64::from(u);
                                b += f64::from(v);
                                n += 1;
                            }
                        }
                    }
                    if n > 0 {
                        gaps.push(((a - b) / f64::from(n)).abs() as f32);
                    }
                }
            }
            gaps.sort_by(f32::total_cmp);
            if !gaps.is_empty() {
                println!(
                    "each 128 m cell's mean, the 16 m ground against today's: off by median {:.1} m, 99th {:.1} m, largest {:.1} m",
                    gaps[gaps.len() / 2],
                    gaps[gaps.len() * 99 / 100],
                    gaps[gaps.len() - 1]
                );
            }
            // The seams between the four 16 m tiles (laid as solved, before any stitch): the step across each join.
            let step = |a: usize, b: usize| (l2[a] - l2[b]).abs();
            let mut steps: Vec<f32> = Vec::new();
            for k in 0..w {
                steps.push(step(k * w + node4 - 1, k * w + node4));
                steps.push(step((node4 - 1) * w + k, node4 * w + k));
            }
            steps.retain(|s| s.is_finite());
            steps.sort_by(f32::total_cmp);
            if steps.is_empty() {
                println!("no 16 m tile solved: nothing to read");
                return;
            }
            println!(
                "the joins between the four 16 m tiles (no stitch yet): the step median {:.1} m, 99th {:.1} m, the largest {:.1} m",
                steps[steps.len() / 2],
                steps[steps.len() * 99 / 100],
                steps[steps.len() - 1]
            );
            // ★ THE STITCH THE GAME MAKES: the four tiles joined by the one cut and the eight-cell blend (`stitch`, the
            // band half a buffer less the blend), as the finished 128 m tiles are; the rest reads the joined ground.
            {
                use vd_terrain::fine::stitch::{BLEND, Field, stitch_columns, stitch_rows};
                let fields: Vec<Option<Field>> = solved
                    .iter()
                    .zip(&nodes)
                    .map(|(t, &(nx, ny))| {
                        t.as_ref().map(|t| {
                            let off = ((t.side - node4) / 2) as i64;
                            Field {
                                gx0: (i64::from(nx) << shift4) - off,
                                gy0: (i64::from(ny) << shift4) - off,
                                w: t.side,
                                h: t.side,
                                v: t.quarters.clone(),
                            }
                        })
                    })
                    .collect();
                if let [Some(nw), Some(ne), Some(sw), Some(se)] = &fields[..] {
                    let off = ((solved[0].as_ref().expect("a tile").side - node4) / 2) as i64;
                    let band = off - BLEND;
                    let gx = i64::from(nodes[1].0) << shift4;
                    let gy = i64::from(nodes[2].1) << shift4;
                    let (top, _) = stitch_columns(nw, ne, gx, band);
                    let (bottom, _) = stitch_columns(sw, se, gx, band);
                    let (joined, _) = stitch_rows(&top, &bottom, gy, band);
                    for y in 0..w {
                        for x in 0..w {
                            let q = joined.at((i0 + x) as i64, (j0 + y) as i64);
                            l2[y * w + x] = if q == i32::MIN {
                                f32::NAN
                            } else {
                                q as f32 / 4.0
                            };
                        }
                    }
                    let step = |a: usize, b: usize| (l2[a] - l2[b]).abs();
                    let mut steps: Vec<f32> = (0..w)
                        .flat_map(|k| {
                            [
                                step(k * w + node4 - 1, k * w + node4),
                                step((node4 - 1) * w + k, node4 * w + k),
                            ]
                        })
                        .filter(|s| s.is_finite())
                        .collect();
                    steps.sort_by(f32::total_cmp);
                    println!(
                        "the joins AFTER THE STITCH: the step median {:.1} m, 99th {:.1} m, the largest {:.1} m",
                        steps[steps.len() / 2],
                        steps[steps.len() * 99 / 100],
                        steps[steps.len() - 1]
                    );
                }
            }
            for (name, z) in [("today_128m_curve", &l1), ("new_16m", &l2)] {
                let (mut r1, mut r5) = (
                    reliefs(z, w, (1_000.0 / cell) as usize),
                    reliefs(z, w, (5_000.0 / cell) as usize),
                );
                summary(&format!("{name}, 1 km"), &mut r1);
                summary(&format!("{name}, 5 km"), &mut r5);
                steep_census(name, z, w, cell);
                let (counts, rb, density, order) = branching(z, w, cell, 0.05);
                println!(
                    "  {name:<40}: BRANCHING (heads 0.05 km²) orders {counts:?}, bifurcation {rb:.2}, density {density:.2} km/km²"
                );
                drainage_map(z, w, cell, &order, &format!("{out}/{name}_above.png"));
                let c = (w as f64 / 2.0, w as f64 * 0.85);
                render_view(
                    z,
                    w,
                    cell,
                    c,
                    1.8,
                    0.0,
                    2.0,
                    &format!("{out}/{name}_walker.png"),
                );
                render_view(
                    z,
                    w,
                    cell,
                    c,
                    60.0,
                    0.0,
                    6.0,
                    &format!("{out}/{name}_low.png"),
                );
                render_view(
                    z,
                    w,
                    cell,
                    c,
                    400.0,
                    0.0,
                    12.0,
                    &format!("{out}/{name}_high.png"),
                );
            }
            println!("pictures in {out}");
        }
        Some("l3") => {
            // ★ THE 4 m LEVEL ON THE BENCH (B5, `docs/investigation/2026-10-14/ground_2to4m_design.md` rev 2): the 128 m tiles
            // around the tile `F,TX,TY` solved and finished; 5 × 5 16 m tiles around its middle node solved and the middle
            // 3 × 3 finished, as the planet's shard does; the 16 m flow record of the middle node; the 2 × 2 4 m tiles in the
            // node's middle; the census beside Earth's 4 m numbers, the detail against today's 16 m curve, the views; the
            // stored unit `UNIT` a metre (4 = quarter metres, today's; 16 = sixteenths).
            use vd_terrain::fine::finish::block_keys;
            use vd_terrain::fine::l2::solve_l2;
            use vd_terrain::fine::l3::{L3_RUNG, TILE_CELLS_SHIFT, flow_record, solve_l3};
            use vd_terrain::fine::level::{FineLevel, LevelGeom};
            use vd_terrain::fine::read::{FineBook, FineZ};
            use vd_terrain::fine::stitch::{BLEND, Field, stitch_columns, stitch_rows};
            use vd_terrain::fine::tile::{finish, raw_field};
            let world = args.get(2).expect("WORLD.art");
            let tile: Vec<i32> = args
                .get(3)
                .expect("F,TX,TY")
                .split(',')
                .map(|x| x.trim().parse().expect("an integer"))
                .collect();
            let out = args.get(4).expect("OUT_DIR");
            let unit: f64 = args.get(5).map_or(4.0, |s| s.parse().expect("a unit"));
            std::fs::create_dir_all(out).expect("the output directory");
            let artifact = vd_bins::artifact_file::read(world).expect("the world reads");
            let face = vd_seed::bend::Face::from_index(tile[0] as u8).expect("a face");
            let key = TileKey {
                face,
                tx: tile[1],
                ty: tile[2],
            };
            let g1 = LevelGeom::of(FineLevel::L1, &body, &lattice).expect("the 128 m level");
            let g2 = LevelGeom::of(FineLevel::L2, &body, &lattice).expect("the 16 m level");
            let started = std::time::Instant::now();
            // 1. The 128 m tiles: the 3 × 3 around `key` finished, each from its nine raw tiles.
            let ring: Vec<TileKey> = block_keys(key, g1).into_iter().flatten().collect();
            let mut all: Vec<TileKey> = ring
                .iter()
                .flat_map(|&k| block_keys(k, g1).into_iter().flatten())
                .collect();
            all.sort();
            all.dedup();
            let raws1: std::collections::BTreeMap<TileKey, vd_terrain::fine::stitch::Field> =
                std::thread::scope(|sc| {
                    let handles: Vec<_> = all
                        .iter()
                        .map(|&k| {
                            let (body, lattice, artifact) = (&body, &lattice, &artifact);
                            sc.spawn(move || {
                                solve_l1(body, lattice, artifact, k)
                                    .map(|t| (k, raw_field(k, g1, t.side, t.quarters)))
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .filter_map(|h| h.join().expect("a solve"))
                        .collect()
                });
            let mut book1 =
                FineBook::new(FineLevel::L1, &body, &lattice, artifact.sea_m).expect("a book");
            for &k in &ring {
                let keys = block_keys(k, g1);
                let block: [Option<(TileKey, &vd_terrain::fine::stitch::Field)>; 9] =
                    std::array::from_fn(|i| keys[i].and_then(|kk| raws1.get(&kk).map(|f| (kk, f))));
                assert!(book1.insert(k, finish(block, g1).expect("a tile finishes")));
            }
            println!(
                "{} 128 m tiles solved, {} finished, in {:.1} s",
                raws1.len(),
                book1.len(),
                started.elapsed().as_secs_f64()
            );
            // 2. The 16 m tiles: 5 × 5 raw around the middle node, the 3 × 3 middle finished.
            let mid = TILE_NODES / 2;
            let (cx, cy) = (key.tx * TILE_NODES + mid, key.ty * TILE_NODES + mid);
            let node = |x: i32, y: i32| TileKey { face, tx: x, ty: y };
            let raw_keys: Vec<TileKey> = (-2..=2)
                .flat_map(|dy| (-2..=2).map(move |dx| (dx, dy)))
                .map(|(dx, dy)| node(cx + dx, cy + dy))
                .collect();
            let started = std::time::Instant::now();
            let raws2: std::collections::BTreeMap<TileKey, vd_terrain::fine::stitch::Field> =
                std::thread::scope(|sc| {
                    let handles: Vec<_> = raw_keys
                        .iter()
                        .map(|&k| {
                            let (body, lattice, artifact, book1) =
                                (&body, &lattice, &artifact, &book1);
                            sc.spawn(move || {
                                solve_l2(body, lattice, artifact, book1, k)
                                    .map(|t| (k, raw_field(k, g2, t.side, t.quarters)))
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .filter_map(|h| h.join().expect("a solve"))
                        .collect()
                });
            let mut book2 =
                FineBook::new(FineLevel::L2, &body, &lattice, artifact.sea_m).expect("a book");
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let k = node(cx + dx, cy + dy);
                    let keys = block_keys(k, g2);
                    let block: [Option<(TileKey, &vd_terrain::fine::stitch::Field)>; 9] =
                        std::array::from_fn(|i| {
                            keys[i].and_then(|kk| raws2.get(&kk).map(|f| (kk, f)))
                        });
                    assert!(book2.insert(k, finish(block, g2).expect("a 16 m tile finishes")));
                }
            }
            println!(
                "{} 16 m tiles solved, {} finished, in {:.1} s",
                raws2.len(),
                book2.len(),
                started.elapsed().as_secs_f64()
            );
            // 3. The 16 m flow record of the middle node.
            let started = std::time::Instant::now();
            let record = flow_record(&body, &lattice, &artifact, &book1, &book2, node(cx, cy))
                .expect("the 16 m flow record");
            println!(
                "the 16 m flow record of node ({cx}, {cy}): {} points in {:.2} s",
                record.points(),
                started.elapsed().as_secs_f64()
            );
            // 4. The 2 × 2 4 m tiles in the node's middle.
            let shift2 = cells_per_node_shift(&body, &lattice, L3_RUNG).expect("nests");
            let per = 1i32 << (shift2 - TILE_CELLS_SHIFT);
            let (lx, ly) = (cx * per + per / 2 - 1, cy * per + per / 2 - 1);
            let l3keys: Vec<TileKey> = [(0, 0), (1, 0), (0, 1), (1, 1)]
                .iter()
                .map(|&(dx, dy)| node(lx + dx, ly + dy))
                .collect();
            // ★ THE SOIL LAW (`VD_L3_SOIL=1`, ruling B99 step 2's bench): each 4 m point the gentler of the river law's
            // slope and the soil law's (K_h Oregon's 0.003 m²/yr, ρr/ρs 2, S_c the level's tan 35°).
            let soil = std::env::var_os("VD_L3_SOIL").map(|_| vd_terrain::fine::refine::SoilLaw {
                kh_m2_yr: SOIL_KH_M2_YR,
                rock_over_soil: ROCK_OVER_SOIL,
            });
            println!(
                "the 4 m law: {}",
                if soil.is_some() {
                    "the river law and the soil law (each point the gentler slope)"
                } else {
                    "the river law alone (today's bench)"
                }
            );
            let started = std::time::Instant::now();
            let solved: Vec<Option<vd_terrain::fine::l3::L3Tile>> = std::thread::scope(|sc| {
                let handles: Vec<_> = l3keys
                    .iter()
                    .map(|&k| {
                        let (body, lattice, record, book2, soil) =
                            (&body, &lattice, &record, &book2, &soil);
                        let sea_m = artifact.sea_m;
                        sc.spawn(move || {
                            let t0 = std::time::Instant::now();
                            let t = solve_l3(body, lattice, sea_m, record, book2, k, soil.as_ref());
                            println!(
                                "  the 4 m tile ({}, {}): {:.1} s{}",
                                k.tx,
                                k.ty,
                                t0.elapsed().as_secs_f64(),
                                t.as_ref().map_or(" — DID NOT SOLVE".to_owned(), |t| format!(
                                    ", {} points, {} passes, {:.1} % of them on the soil law, the law's detail median / largest {:.2} / {:.2} m",
                                    t.points,
                                    t.passes,
                                    100.0 * t.on_soil as f64 / t.points.max(1) as f64,
                                    t.detail_cm[0] as f64 / 100.0,
                                    t.detail_cm[1] as f64 / 100.0
                                ))
                            );
                            t
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("a solve"))
                    .collect()
            });
            println!(
                "the four 4 m tiles solved in {:.1} s (four threads)",
                started.elapsed().as_secs_f64()
            );
            // 5. The joined 4 m ground at the unit, and today's 16 m curve at the same 4 m cells.
            let ts = 1usize << TILE_CELLS_SHIFT;
            let w = 2 * ts;
            let (i0, j0) = (
                (lx as i64) << TILE_CELLS_SHIFT,
                (ly as i64) << TILE_CELLS_SHIFT,
            );
            let fields: Vec<Field> = solved
                .iter()
                .zip(&l3keys)
                .map(|(t, k)| {
                    let t = t.as_ref().expect("every 4 m tile solves");
                    let off = ((t.side - ts) / 2) as i64;
                    Field {
                        gx0: (i64::from(k.tx) << TILE_CELLS_SHIFT) - off,
                        gy0: (i64::from(k.ty) << TILE_CELLS_SHIFT) - off,
                        w: t.side,
                        h: t.side,
                        v: t.metres
                            .iter()
                            .map(|m| m.map_or(i32::MIN, |m| (m * unit + 0.5).floor() as i32))
                            .collect(),
                    }
                })
                .collect();
            let off = ((solved[0].as_ref().expect("a tile").side - ts) / 2) as i64;
            let band = off - BLEND;
            let (gx, gy) = (
                i64::from(l3keys[1].tx) << TILE_CELLS_SHIFT,
                i64::from(l3keys[2].ty) << TILE_CELLS_SHIFT,
            );
            let (top, _) = stitch_columns(&fields[0], &fields[1], gx, band);
            let (bottom, _) = stitch_columns(&fields[2], &fields[3], gx, band);
            let (joined, _) = stitch_rows(&top, &bottom, gy, band);
            let mut l3 = vec![f32::NAN; w * w];
            for y in 0..w {
                for x in 0..w {
                    let q = joined.at(i0 + x as i64, j0 + y as i64);
                    if q != i32::MIN {
                        l3[y * w + x] = (f64::from(q) / unit) as f32;
                    }
                }
            }
            let sea = f64::from(artifact.sea_m);
            let mut l2c = vec![f32::NAN; w * w];
            let unit_z = (vd_terrain::units::STEPS_PER_M << vd_terrain::units::LENGTH_BITS) as f64;
            for y in 0..w {
                for x in 0..w {
                    if let FineZ::Held(z) = book2.z(
                        face,
                        L3_RUNG,
                        L3_RUNG,
                        (i0 + x as i64) as i32,
                        (j0 + y as i64) as i32,
                    ) {
                        l2c[y * w + x] = (z.raw() as f64 / unit_z - sea) as f32;
                    }
                }
            }
            // 6. The readings.
            let step = |z: &[f32], a: usize, b: usize| (z[a] - z[b]).abs();
            let mut joins: Vec<f32> = (0..w)
                .flat_map(|k| {
                    [
                        step(&l3, k * w + ts - 1, k * w + ts),
                        step(&l3, (ts - 1) * w + k, ts * w + k),
                    ]
                })
                .filter(|s| s.is_finite())
                .collect();
            joins.sort_by(f32::total_cmp);
            println!(
                "the joins after the stitch: the step median {:.2} m, 99th {:.2} m, the largest {:.2} m",
                joins[joins.len() / 2],
                joins[joins.len() * 99 / 100],
                joins[joins.len() - 1]
            );
            let mut res: Vec<f64> = l3
                .iter()
                .zip(&l2c)
                .filter(|(a, b)| a.is_finite() && b.is_finite())
                .map(|(a, b)| f64::from(a - b))
                .collect();
            let rms = (res.iter().map(|r| r * r).sum::<f64>() / res.len().max(1) as f64).sqrt();
            res.iter_mut().for_each(|r| *r = r.abs());
            res.sort_by(f64::total_cmp);
            println!(
                "THE 4 m DETAIL against today's 16 m curve: RMS {:.2} m, 99th {:.2} m, largest {:.2} m (Earth's 2 m detail: RMS 0.71 / 0.95 / 1.22 m, 99th 2.45 / 3.11 / 4.19 m)",
                rms,
                res[res.len() * 99 / 100],
                res[res.len() - 1]
            );
            let k4 = 4usize;
            let mut gaps: Vec<f64> = Vec::new();
            for by in 0..w / k4 {
                for bx in 0..w / k4 {
                    let (mut a, mut b, mut n) = (0f64, 0f64, 0u32);
                    for y in by * k4..(by + 1) * k4 {
                        for x in bx * k4..(bx + 1) * k4 {
                            let (u, v) = (l3[y * w + x], l2c[y * w + x]);
                            if u.is_finite() && v.is_finite() {
                                a += f64::from(u);
                                b += f64::from(v);
                                n += 1;
                            }
                        }
                    }
                    if n > 0 {
                        gaps.push(((a - b) / f64::from(n)).abs());
                    }
                }
            }
            gaps.sort_by(f64::total_cmp);
            println!(
                "each 16 m cell's mean, the 4 m ground against today's: off by median {:.2} m, 99th {:.2} m, largest {:.2} m",
                gaps[gaps.len() / 2],
                gaps[gaps.len() * 99 / 100],
                gaps[gaps.len() - 1]
            );
            println!(
                "  EARTH at 4 m: San Gabriel 21.63 / 5.01 / 0.364 % of the links over 35° / 45° / 63°; Oregon 9.38 / 0.515 / 0.0021 %"
            );
            let cell = f64::from(1u32 << L3_RUNG);
            let r_m = body.radius_m();
            let terraced = |z: &[f32]| -> Vec<f32> {
                z.iter()
                    .map(|&h| {
                        if h.is_finite() {
                            let s = r_m + sea + f64::from(h);
                            (vd_terrain::height::terrace_m(&body, s, 0) - r_m - sea) as f32
                        } else {
                            h
                        }
                    })
                    .collect()
            };
            let (l2t, l3t) = (terraced(&l2c), terraced(&l3));
            let tag = if (unit - 4.0).abs() < 1e-9 {
                "q4".to_owned()
            } else {
                format!("u{unit}")
            };
            for (name, z) in [
                ("today_16m_curve".to_owned(), &l2c),
                (format!("new_4m_{tag}"), &l3),
                ("today_16m_curve_terraced".to_owned(), &l2t),
                (format!("new_4m_{tag}_terraced"), &l3t),
            ] {
                let name = name.as_str();
                let mut r1 = reliefs(z, w, (1_000.0 / cell) as usize);
                summary(&format!("{name}, 1 km"), &mut r1);
                steep_census(name, z, w, cell);
                if !name.ends_with("terraced") {
                    let (counts, rb, density, order) = branching(z, w, cell, 0.05);
                    println!(
                        "  {name:<40}: BRANCHING (heads 0.05 km²) orders {counts:?}, bifurcation {rb:.2}, density {density:.2} km/km²"
                    );
                    drainage_map(z, w, cell, &order, &format!("{out}/{name}_above.png"));
                }
                let c = (w as f64 / 2.0, w as f64 * 0.85);
                render_view(
                    z,
                    w,
                    cell,
                    c,
                    1.8,
                    0.0,
                    2.0,
                    &format!("{out}/{name}_walker.png"),
                );
                render_view(
                    z,
                    w,
                    cell,
                    c,
                    30.0,
                    0.0,
                    6.0,
                    &format!("{out}/{name}_low.png"),
                );
                // A walker on the highest ground of the square's south half, looking north across it.
                let (mut best, mut at) = (f32::MIN, c);
                for y in (w / 2)..(w * 15 / 16) {
                    for x in (w / 8)..(w * 7 / 8) {
                        let v = l2c[y * w + x];
                        if v.is_finite() && v > best {
                            (best, at) = (v, (x as f64, y as f64));
                        }
                    }
                }
                render_view(
                    z,
                    w,
                    cell,
                    at,
                    1.8,
                    0.0,
                    4.0,
                    &format!("{out}/{name}_crest.png"),
                );
            }
            println!("pictures in {out}");
        }
        _ => panic!(
            "solve PLANET OUT.art [cores] | maps PLANET WORLD.art OUT_DIR TAG [LAT0 LAT1 LON0 LON1] | words PLANET | store-get PLANET OUT.art | store-put PLANET IN.art | stands PLANET WORLD.art OUT_DIR | forms PLANET WORLD.art F,TX,TY OUT_DIR [EARTH.hgt…]"
        ),
    }
}
