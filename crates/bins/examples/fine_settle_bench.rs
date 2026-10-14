//! ★ THE FINE ROUNDS' SETTLE BENCH (2026-10-13; the owner: "let's optimize it"): every fine round of a 128 m tile ended
//! at `MAX_PASSES` without settling. A COPY of the fine rounds (the game's own coarse start, then the fine domain's
//! rounds re-written here with an instrument and a longer pass limit) logs each pass: how far the node means moved (the
//! settle test reads the largest), how many points changed their receiver, and how many held nodes still move more than
//! the settle bound — so creep (slow decay), a cycle (the same numbers again and again) and a drift can be told apart.
//! Nothing in the game changes. `VD_REALM_STORE=<file> cargo run --release -p vd-bins --example fine_settle_bench --
//! [tile index among the berth's whole tiles] [pass limit]`

use std::collections::BTreeSet;

use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::fine::delaunay::NONE;
use vd_terrain::fine::geometry::locate;
use vd_terrain::fine::l1::{
    COARSE_ROUNDS, COARSE_RUNG, Domain, L1_RUNG, SETTLE_M, TileKey, domain, rounds,
};
use vd_terrain::fine::tree::{accumulate, tree};
use vd_terrain::gf::Gf;

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;
/// The rebuilt method's fine rounds, as the game ran them before ruling B92 (the bench's "classic" copy).
const ROUNDS: usize = 3;
const HOME: u64 = 4_030_111_653_607_004_909;
const BERTH: [f64; 3] = [
    3_273_214.798_924_267_3,
    -2_850_131.846_649_597,
    -4_630_822.805_490_945_5,
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // The tile: an index among the berth's whole tiles, or `F,TX,TY` named outright.
    let named_tile: Option<(u8, u32, u32)> = args.get(1).and_then(|s| {
        let v: Vec<u32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (v.len() == 3).then(|| (v[0] as u8, v[1], v[2]))
    });
    let pick: usize = named_tile.map_or_else(
        || args.get(1).map_or(0, |s| s.parse().expect("an index")),
        |_| 0,
    );
    let limit: usize = args
        .get(2)
        .map_or(200, |s| s.parse().expect("a pass limit"));
    let mode: String = args.get(3).cloned().unwrap_or_else(|| "classic".to_owned());
    let realm = RealmId::Planet(HOME);
    let (rows, _) = vd_bins::home_system_world(UNIVERSE);
    let radius_m = rows
        .iter()
        .find(|r| r.realm == realm)
        .and_then(|r| match r.look {
            Some(vd_core::geometry::Boundary::Shell { r }) => Some(r),
            _ => None,
        })
        .expect("the home planet is round");
    let charter = vd_bins::body_charter(
        UNIVERSE,
        &BTreeSet::from([vd_core::worldgen::HOME_SYSTEM]),
        &BTreeSet::from([vd_core::worldgen::GALAXY]),
        realm,
    )
    .expect("a charter");
    let body = BodyDefinition::from_seed(HOME, radius_m, vd_bins::facts_of_charter(&charter))
        .expect("a ladder");
    let lattice = body.macro_lattice().expect("a lattice");
    let env = vd_bins::process_env();
    let store = vd_bins::open_realm_store(&env, realm)
        .expect("the store opens")
        .expect("VD_REALM_STORE names a file");
    let artifact = vd_bins::artifact_store::read_artifact(
        &store,
        vd_terrain::artifact_world_tag(UNIVERSE),
        &lattice,
    )
    .expect("the stored rows agree with their head")
    .expect("an artifact is stored");
    let radial = (BERTH[0] * BERTH[0] + BERTH[1] * BERTH[1] + BERTH[2] * BERTH[2]).sqrt();
    let dir = [BERTH[0] / radial, BERTH[1] / radial, BERTH[2] / radial];
    let drawn = artifact.drawn_range_m(&lattice);
    let named = vd_terrain::artifact::fine_needs_of_head(
        vd_terrain::fine::level::FineLevel::L1,
        &body,
        (drawn.0, drawn.1, artifact.sea_m),
        dir,
        radial,
        vd_core::geometry::drawable_theta_min_rad(),
    );
    let (face, tx, ty) = named_tile.unwrap_or_else(|| {
        named
            .iter()
            .filter(|(_, l)| *l == 0)
            .nth(pick)
            .expect("a whole tile")
            .0
    });
    let key = TileKey {
        face: Face::from_index(face).expect("a face"),
        tx: tx as i32,
        ty: ty as i32,
    };
    println!("tile ({face}, {tx}, {ty}), pass limit {limit}");
    // THE GAME'S OWN START: the coarse domain's rounds, then the fine start read off them (`solve_l1`, copied).
    let mut coarse = domain(&body, &lattice, &artifact, COARSE_RUNG, key).expect("a coarse domain");
    // ★ THE BASE LEVEL TEST (`lower<D>`, 2026-10-13, the flat plains): every outlet where an 8 km river leaves the domain
    // stands D metres under the lower node's row — a river bed under its node's mean, as Earth's plains' rivers lie under
    // their uplands — on both domains.
    let lower_m: Option<f64> = mode
        .strip_prefix("lower")
        .map(|d| d.parse().expect("lower<D>"));
    if let Some(d) = lower_m {
        lower_exits(&mut coarse, d);
    }
    // ★ THE CHANNEL TEST (`channels<D>`): every 8 km river link inside the domain is a line of outlets — the points within
    // a point's spacing of the link — at a bed D metres under the rows it joins, so the fine streams meet the 8 km
    // rivers anywhere inside the tile (B77's "a drainage network built first").
    let channel_m: Option<f64> = mode
        .strip_prefix("channels")
        .map(|d| d.parse().expect("channels<D>"));
    let radius = body.radius_m();
    if let Some(d) = channel_m {
        let n = channel_outlets(&mut coarse, &lattice, &artifact, radius, 600.0, d);
        println!("the coarse channels: {n} points");
    }
    // ★ A, THE LAWFUL EXIT (`lawexits`, ruling B94 step 1 as the owner started it): each river exit under its row by the
    // channel law's depth (see [`law_exits`]).
    let law = mode == "lawexits";
    if law {
        law_exits(&mut coarse, &lattice, &artifact, radius, false);
    }
    let mut uc = coarse.ref_u0.clone();
    let mut hc = coarse.base.clone();
    let coarse_passes = rounds(&coarse, &mut uc, &mut hc, COARSE_ROUNDS);
    println!(
        "coarse rounds' passes {coarse_passes:?} ({} points)",
        coarse.dirs.len()
    );
    let mut fine = domain(&body, &lattice, &artifact, L1_RUNG, key).expect("a fine domain");
    if let Some(d) = lower_m {
        let n = lower_exits(&mut fine, d);
        println!("the base level: {n} river exits {d} m under their rows");
    }
    if let Some(d) = channel_m {
        let n = channel_outlets(&mut fine, &lattice, &artifact, radius, 150.0, d);
        println!("the fine channels: {n} points, {d} m under the rows");
    }
    if law {
        law_exits(&mut fine, &lattice, &artifact, radius, true);
    }
    let mut u: Vec<Gf> = fine
        .refs
        .iter()
        .zip(&fine.ref_u0)
        .map(|(m, &u0)| coarse.refs.binary_search(m).map_or(u0, |i| uc[i]))
        .collect();
    let mut h = fine.base.clone();
    let mut from = 0u32;
    for ((hp, d), &out) in h.iter_mut().zip(&fine.dirs).zip(&fine.outlet) {
        if out {
            continue;
        }
        if let Some((t, w)) = locate(&coarse.dirs, &coarse.graph, from, d) {
            let v = coarse.graph.tris[t as usize];
            *hp = w[0] * hc[v[0] as usize] + w[1] * hc[v[1] as usize] + w[2] * hc[v[2] as usize];
            from = t;
        }
    }
    println!(
        "fine domain: {} points, {} nodes ({} held)",
        fine.dirs.len(),
        fine.nodes.len(),
        fine.held.iter().filter(|h| **h).count()
    );
    if mode == "pictures" {
        pictures(&fine, &u, &h, face);
        return;
    }
    if lower_m.is_some() | channel_m.is_some() | law {
        let (mut un, mut hn) = (u.clone(), h.clone());
        let rounds_run = vd_terrain::fine::l1::fine_rounds(&fine, &mut un, &mut hn);
        println!("the game's rounds run: {rounds_run}");
        fit(&fine, &hn);
        census(&mode, &fine, &hn);
        relief_of(&fine, &hn, face, &mode);
        return;
    }
    if mode == "census" {
        // ★ THE WALLS AND SPIKES CENSUS (2026-10-13, after the owner's window look showed vertical ribbed walls and a
        // spire near the berth): the rebuilt method (the game before B92) and the game's own fixed rounds from the same
        // start, each read for steep links, walls across the drainage, pits and spikes.
        let (mut uo, mut ho) = (u.clone(), h.clone());
        instrumented_rounds(&fine, &mut uo, &mut ho, ROUNDS, 60);
        census("rebuilt (before B92)", &fine, &ho);
        let (mut un, mut hn) = (u.clone(), h.clone());
        let rounds_run = vd_terrain::fine::l1::fine_rounds(&fine, &mut un, &mut hn);
        println!("the game's rounds run: {rounds_run}");
        census("the game's rounds (B92 amended)", &fine, &hn);
        return;
    }
    let started = std::time::Instant::now();
    if mode == "interleave" {
        interleaved(&fine, &mut u, &mut h, limit);
    } else if mode == "fixed" {
        fixed_network(&fine, &mut u, &mut h, limit, 0);
    } else if let Some(every) = mode.strip_prefix("rebuild") {
        fixed_network(
            &fine,
            &mut u,
            &mut h,
            limit,
            every.parse().expect("rebuildN"),
        );
    } else {
        instrumented_rounds(&fine, &mut u, &mut h, ROUNDS, limit);
    }
    println!(
        "fine part took {:.0} ms",
        started.elapsed().as_secs_f64() * 1e3
    );
    fit(&fine, &h);
    census(&mode, &fine, &h);
}

/// How well each held node's fine mean matches the node's own row (what the rounds exist to reach): the largest and the
/// mean gap in metres.
fn fit(d: &Domain, h: &[Gf]) {
    let nl = d.nodes.len();
    let (mut sum, mut area) = (vec![0.0f64; nl], vec![0.0f64; nl]);
    for ((&k, &hp), &a) in d.pnode.iter().zip(h).zip(&d.geom.area) {
        sum[k as usize] += hp.to_f64() * a.to_f64();
        area[k as usize] += a.to_f64();
    }
    let gaps: Vec<f64> = (0..nl)
        .filter(|&k| d.held[k] && area[k] > 0.0)
        .map(|k| (sum[k] / area[k] - d.node_z[k].to_f64()).abs())
        .collect();
    let max = gaps.iter().copied().fold(0.0, f64::max);
    let mean = gaps.iter().sum::<f64>() / gaps.len().max(1) as f64;
    println!(
        "FIT: the held nodes' fine mean against their own row: max {max:.1} m, mean {mean:.1} m"
    );
}

/// ★ THE LAWFUL EXIT (bench only): each 8 km river that leaves the domain leaves at its BED, under the lower row by
/// `D = HI·Rf`:
/// - `Rf`, the channel relief inside the cell — Whipple & Tucker 1999 (JGR 104:17661) eq. 22,
///   `Rf = ks·ka^−θ·(L^(1−hθ) − xc^(1−hθ))/(1−hθ)`, with the steepness `ks = S·A^θ` read off the 8 km world (S the drop
///   to the receiver over the link, A the node's drainage area = its discharge over its rain), θ = 1/n of the joint law
///   (n = 2.2, so the law's own concavity), Hack's law `A = ka·x^h` (Hack 1957: ka = 6.69 m^0.33, h = 1.67), the channel
///   head at xc = 320 m (Whipple & Tucker's 10⁵ m² head area; their calibration bodies the King Range and Taiwan's
///   Central Range), and L the link's length;
/// - `HI`, the share of that relief under the channel's mean — the same law's steady profile `z ∝ L^(1−p) − x^(1−p)`
///   (p = hθ) averaged along the channel: `HI = (1−p)/(2−p)`, a property of the law, never a reading.
fn law_exits(
    d: &mut Domain,
    lattice: &vd_terrain::macro_lattice::MacroLattice,
    artifact: &vd_terrain::artifact::Artifact,
    radius_m: f64,
    print: bool,
) {
    use vd_terrain::artifact::{RECEIVER_NONE, RECEIVER_SLOT_MASK};
    use vd_terrain::fine::l1::decode_class;
    let (ka, h, xc) = (6.69f64, 1.67f64, 320.0f64);
    let theta = 1.0 / vd_terrain::solve::JOINT_N;
    let p = h * theta;
    let hi = (1.0 - p) / (2.0 - p);
    let unit = |g: [vd_recipe::Gi; 3]| {
        let s = (1u64 << 40) as f64;
        let v = [
            g[0].raw() as f64 / s,
            g[1].raw() as f64 / s,
            g[2].raw() as f64 / s,
        ];
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let local: std::collections::BTreeMap<u32, usize> =
        d.nodes.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let sea = f64::from(artifact.sea_m);
    let mut depths = Vec::new();
    for (li, &k) in d.nodes.iter().enumerate() {
        let f = artifact.rows[k as usize].receiver_facies;
        if f & RECEIVER_NONE != 0 {
            continue;
        }
        let r = lattice.neighbours(k)[(f & RECEIVER_SLOT_MASK) as usize];
        if local.contains_key(&r) {
            continue;
        }
        let (a, b) = (unit(lattice.direction(k)), unit(lattice.direction(r)));
        let mid = {
            let m = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
            let l = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            [m[0] / l, m[1] / l, m[2] / l]
        };
        // The domain's own exit: the node's point nearest the link's midpoint.
        let Some(pt) = (0..d.pnode.len())
            .filter(|&q| d.pnode[q] as usize == li && d.outlet[q])
            .min_by(|&x, &y| {
                let dx = unit(d.dirs[x]);
                let dy = unit(d.dirs[y]);
                let ex =
                    (dx[0] - mid[0]).powi(2) + (dx[1] - mid[1]).powi(2) + (dx[2] - mid[2]).powi(2);
                let ey =
                    (dy[0] - mid[0]).powi(2) + (dy[1] - mid[1]).powi(2) + (dy[2] - mid[2]).powi(2);
                ex.total_cmp(&ey)
            })
        else {
            continue;
        };
        let len = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            * radius_m;
        let (zk, zr) = (
            f64::from(artifact.rows[k as usize].z_m) - sea,
            f64::from(artifact.rows[r as usize].z_m) - sea,
        );
        let slope = ((zk - zr) / len).max(0.0);
        let rain = decode_class(artifact.rows[k as usize].rain, 4) as f64 / 1_000.0;
        let q = decode_class(artifact.rows[k as usize].discharge_log, 2) as f64 / 1_000.0;
        let area = if rain > 0.0 { q / rain } else { 0.0 };
        let ks = slope * area.powf(theta);
        let rf = ks * ka.powf(-theta) * (len.powf(1.0 - p) - xc.powf(1.0 - p)) / (1.0 - p);
        let depth = hi * rf.max(0.0);
        d.base[pt] -= Gf::from_f64(depth);
        depths.push((depth, slope, area, rf));
    }
    if print {
        depths.sort_by(|x, y| x.0.total_cmp(&y.0));
        let n = depths.len();
        let at = |f: f64| depths[((n.max(1) - 1) as f64 * f).round() as usize];
        if n > 0 {
            for (label, f) in [("least", 0.0), ("median", 0.5), ("most", 1.0)] {
                let (dd, s, a, rf) = at(f);
                println!(
                    "LAW EXIT {label}: depth {dd:.1} m (Rf {rf:.0} m, HI {hi:.3}) — slope {:.3} %, area {:.0} km²",
                    s * 100.0,
                    a / 1e6
                );
            }
        }
        println!("the lawful exits: {n} river exits lowered");
    }
}

/// The 8 km river links inside the domain as outlets: each point within `width_m` of a link from a node to its receiver
/// (both in the domain) stands at the rows' interpolation along the link less `depth_m`; returns how many points.
fn channel_outlets(
    d: &mut Domain,
    lattice: &vd_terrain::macro_lattice::MacroLattice,
    artifact: &vd_terrain::artifact::Artifact,
    radius_m: f64,
    width_m: f64,
    depth_m: f64,
) -> usize {
    use vd_terrain::artifact::{RECEIVER_NONE, RECEIVER_SLOT_MASK};
    let unit = |g: [vd_recipe::Gi; 3]| {
        let s = (1u64 << 40) as f64;
        let v = [
            g[0].raw() as f64 / s,
            g[1].raw() as f64 / s,
            g[2].raw() as f64 / s,
        ];
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let local: std::collections::BTreeMap<u32, usize> =
        d.nodes.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let mut of_node: Vec<Vec<usize>> = vec![Vec::new(); d.nodes.len()];
    for (p, &k) in d.pnode.iter().enumerate() {
        of_node[k as usize].push(p);
    }
    let mut marked = 0usize;
    for (li, &k) in d.nodes.iter().enumerate() {
        let f = artifact.rows[k as usize].receiver_facies;
        if f & RECEIVER_NONE != 0 {
            continue;
        }
        let r = lattice.neighbours(k)[(f & RECEIVER_SLOT_MASK) as usize];
        let Some(&lr) = local.get(&r) else {
            continue;
        };
        let (a, b) = (unit(lattice.direction(k)), unit(lattice.direction(r)));
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        let (za, zb) = (d.node_z[li].to_f64(), d.node_z[lr].to_f64());
        for &p in of_node[li].iter().chain(&of_node[lr]) {
            let v = unit(d.dirs[p]);
            let av = [v[0] - a[0], v[1] - a[1], v[2] - a[2]];
            let t = ((av[0] * ab[0] + av[1] * ab[1] + av[2] * ab[2]) / ab2).clamp(0.0, 1.0);
            let c = [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
            let dist = ((v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2) + (v[2] - c[2]).powi(2))
                .sqrt()
                * radius_m;
            if dist <= width_m {
                let bed = Gf::from_f64(za + t * (zb - za) - depth_m);
                if !d.outlet[p] || bed < d.base[p] {
                    marked += usize::from(!d.outlet[p]);
                    d.outlet[p] = true;
                    d.base[p] = bed;
                }
            }
        }
    }
    marked
}

/// Every outlet over the sea (a river's exit) lowered `d` metres; returns how many.
fn lower_exits(d: &mut Domain, by: f64) -> usize {
    let by = Gf::from_f64(by);
    let mut n = 0;
    for p in 0..d.base.len() {
        if d.outlet[p] && d.base[p] > Gf::ZERO {
            d.base[p] -= by;
            n += 1;
        }
    }
    n
}

/// The tile's points rastered to its face's cells (each cell its points' mean), the local relief at 1 km and 5 km over
/// land windows, and the raster written to `$VD_GRID_OUT` (little-endian f32, row by row) for the side view.
fn relief_of(d: &Domain, h: &[Gf], face: u8, name: &str) {
    let on: Vec<usize> = (0..h.len()).filter(|&p| d.cells[p].face == face).collect();
    let (i0, i1) = on.iter().fold((i32::MAX, i32::MIN), |(a, b), &p| {
        (a.min(d.cells[p].i), b.max(d.cells[p].i))
    });
    let (j0, j1) = on.iter().fold((i32::MAX, i32::MIN), |(a, b), &p| {
        (a.min(d.cells[p].j), b.max(d.cells[p].j))
    });
    let (w, ht) = ((i1 - i0 + 1) as usize, (j1 - j0 + 1) as usize);
    let (mut sum, mut cnt) = (vec![0.0f64; w * ht], vec![0u32; w * ht]);
    for &p in &on {
        let at = (d.cells[p].j - j0) as usize * w + (d.cells[p].i - i0) as usize;
        sum[at] += h[p].to_f64();
        cnt[at] += 1;
    }
    let mut z: Vec<f32> = (0..w * ht)
        .map(|k| {
            if cnt[k] > 0 {
                (sum[k] / f64::from(cnt[k])) as f32
            } else {
                f32::NAN
            }
        })
        .collect();
    // A cell no point fell in takes a neighbour's height (the top-down pictures' own fill).
    for _ in 0..4 {
        for y in 0..ht {
            for x in 0..w {
                if z[y * w + x].is_nan() {
                    let near = [
                        (x.wrapping_sub(1), y),
                        (x + 1, y),
                        (x, y.wrapping_sub(1)),
                        (x, y + 1),
                    ];
                    if let Some(v) = near
                        .iter()
                        .filter(|(a, b)| *a < w && *b < ht)
                        .map(|(a, b)| z[b * w + a])
                        .find(|v| !v.is_nan())
                    {
                        z[y * w + x] = v;
                    }
                }
            }
        }
    }
    for win in [8usize, 39] {
        let mut r = Vec::new();
        for wy in 0..ht / win {
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
                    r.push(hi - lo);
                }
            }
        }
        r.sort_by(f32::total_cmp);
        if r.is_empty() {
            continue;
        }
        let q = |f: f64| r[((r.len() - 1) as f64 * f).round() as usize];
        let under = |m: f32| 100.0 * r.iter().filter(|v| **v < m).count() as f64 / r.len() as f64;
        println!(
            "RELIEF {name} at {} m: p10 {:.0} p50 {:.0} p90 {:.0} m, under 20 m {:.1} %, under 50 m {:.1} % ({} windows)",
            win * 128,
            q(0.1),
            q(0.5),
            q(0.9),
            under(20.0),
            under(50.0),
            r.len()
        );
    }
    if let Ok(path) = std::env::var("VD_GRID_OUT") {
        let bytes: Vec<u8> = z.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).expect("the raster writes");
        println!("GRID {path} {w} {ht}");
    }
}

/// ★ THE CENSUS (bench only): over every link of the point graph, the slope |dh| / length — the shares over 35° (the
/// law's repose), 45°, 63° and 79°, the steepest and its drop — and how many of the links over 63° join two points neither
/// of which drains to the other on the drainage of the final heights (a wall across the drainage); plus the pits and the
/// spikes (a point more than 50 m under, or over, every neighbour) with the deepest and the tallest.
fn census(name: &str, d: &Domain, h: &[Gf]) {
    let z: Vec<f64> = h.iter().map(|v| v.to_f64()).collect();
    let t = tree(h, &d.outlet, &d.graph, &d.geom.len);
    let (mut links, mut over) = (0usize, [0usize; 4]);
    let (mut across, mut steepest, mut steepest_drop) = (0usize, 0.0f64, 0.0f64);
    let bounds = [35f64.to_radians().tan(), 1.0, 2.0, 5.0];
    let (mut pits, mut spikes, mut deepest, mut tallest) = (0usize, 0usize, 0.0f64, 0.0f64);
    for p in 0..z.len() {
        let start = d.graph.start[p] as usize;
        let mut lowest = f64::INFINITY;
        let mut highest = f64::NEG_INFINITY;
        for (k, &q) in d.graph.neighbours(p).iter().enumerate() {
            let q = q as usize;
            lowest = lowest.min(z[q]);
            highest = highest.max(z[q]);
            if q <= p {
                continue;
            }
            let len = d.geom.len[start + k].to_f64();
            if len <= 0.0 {
                continue;
            }
            links += 1;
            let drop = (z[p] - z[q]).abs();
            let slope = drop / len;
            for (b, n) in bounds.iter().zip(over.iter_mut()) {
                if slope > *b {
                    *n += 1;
                }
            }
            if slope > 2.0 && t.recv[p] as usize != q && t.recv[q] as usize != p {
                across += 1;
            }
            if slope > steepest {
                steepest = slope;
                steepest_drop = drop;
            }
        }
        if lowest - z[p] > 50.0 {
            pits += 1;
            deepest = deepest.max(lowest - z[p]);
        }
        if z[p] - highest > 50.0 {
            spikes += 1;
            tallest = tallest.max(z[p] - highest);
        }
    }
    let share = |n: usize| 100.0 * n as f64 / links.max(1) as f64;
    println!(
        "CENSUS {name}: {links} links; over 35° {:.2} %, 45° {:.3} %, 63° {:.4} %, 79° {:.4} %; steepest {:.1}° (a {:.0} m \
         drop); over 63° across the drainage {across}; pits {pits} (deepest {deepest:.0} m), spikes {spikes} (tallest \
         {tallest:.0} m); heights {:.0}..{:.0} m",
        share(over[0]),
        share(over[1]),
        share(over[2]),
        share(over[3]),
        steepest.atan().to_degrees(),
        steepest_drop,
        z.iter().copied().fold(f64::INFINITY, f64::min),
        z.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
}

/// ★ THE INTERLEAVED VARIANT (bench only): the uplift corrected after EVERY pass by the same law the rounds apply after
/// their passes, so the correction meets the capture's drift as it happens; ends when a pass moves no held node's mean by
/// more than the settle bound AND the correction moved no rise by more than a part in a thousand.
fn interleaved(d: &Domain, u: &mut [Gf], h: &mut [Gf], limit: usize) {
    use vd_terrain::solve::{JOINT_K_LP, JOINT_N};
    let n = h.len();
    let (k_lp, inv_n, half_n) = (
        Gf::from_f64(JOINT_K_LP),
        Gf::from_f64(1.0 / JOINT_N),
        Gf::from_f64(0.5 * JOINT_N),
    );
    let tan = Gf::from_f64(vd_terrain::body::TAN_REPOSE);
    let tiny = Gf::from_f64(f64::MIN_POSITIVE);
    let nl = d.nodes.len();
    let means = |h: &[Gf]| -> Vec<Gf> {
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
    };
    let mut last = means(h);
    for pass in 1..=limit {
        let ln_u: Vec<Gf> = u.iter().map(|x| x.ln_fine()).collect();
        let up: Vec<Gf> = (0..n)
            .map(|p| {
                let mut s = Gf::ZERO;
                for t in d.wstart[p] as usize..d.wstart[p + 1] as usize {
                    s += d.wval[t] * ln_u[d.wnode[t] as usize];
                }
                s.exp_fine()
            })
            .collect();
        let t = tree(h, &d.outlet, &d.graph, &d.geom.len);
        let q = accumulate(&t, &d.water);
        for &c in &t.order {
            let c = c as usize;
            let r = t.recv[c];
            if d.outlet[c] || r == NONE {
                h[c] = d.base[c];
                continue;
            }
            let s = (up[c] / (k_lp * q[c].greater(tiny)))
                .powf_fine(inv_n)
                .lesser(tan);
            h[c] = h[r as usize] + s * t.link[c];
        }
        let now = means(h);
        let moved = (0..nl)
            .filter(|&k| d.held[k])
            .map(|k| (now[k] - last[k]).abs().to_f64())
            .fold(0.0, f64::max);
        let mut rise_moved = 0.0f64;
        for (k, &mean) in now.iter().enumerate() {
            let r = d.node_ref[k] as usize;
            if d.held[k] && mean > Gf::ZERO && r < u.len() {
                let factor = (d.node_z[k] / mean).powf_fine(half_n);
                rise_moved = rise_moved.max((factor.to_f64() - 1.0).abs());
                u[r] *= factor;
            }
        }
        last = now;
        if pass % 5 == 0 || pass < 6 {
            println!(
                "interleaved pass {pass:3}: moved max {moved:9.3} m, the largest rise correction {rise_moved:.4}"
            );
        }
        if moved <= SETTLE_M && rise_moved <= 1e-3 {
            println!("interleaved: settled at pass {pass}");
            return;
        }
    }
    println!("interleaved: not settled in {limit}");
}

/// The fine rounds of `vd_terrain::fine::l1::rounds`, COPIED, with the instrument and the pass limit as an argument.
fn instrumented_rounds(d: &Domain, u: &mut [Gf], h: &mut [Gf], rounds: usize, limit: usize) {
    use vd_terrain::solve::{JOINT_K_LP, JOINT_N};
    let n = h.len();
    let (k_lp, inv_n, half_n) = (
        Gf::from_f64(JOINT_K_LP),
        Gf::from_f64(1.0 / JOINT_N),
        Gf::from_f64(0.5 * JOINT_N),
    );
    let tan = Gf::from_f64(vd_terrain::body::TAN_REPOSE);
    let settle = Gf::from_f64(SETTLE_M);
    let tiny = Gf::from_f64(f64::MIN_POSITIVE);
    let nl = d.nodes.len();
    let means = |h: &[Gf]| -> Vec<Gf> {
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
    };
    for round in 0..rounds {
        let ln_u: Vec<Gf> = u.iter().map(|x| x.ln_fine()).collect();
        let up: Vec<Gf> = (0..n)
            .map(|p| {
                let mut s = Gf::ZERO;
                for t in d.wstart[p] as usize..d.wstart[p + 1] as usize {
                    s += d.wval[t] * ln_u[d.wnode[t] as usize];
                }
                s.exp_fine()
            })
            .collect();
        let mut last = means(h);
        let mut prev_recv: Option<Vec<u32>> = None;
        let mut prev2_recv: Option<Vec<u32>> = None;
        let mut node_series: Vec<Vec<f64>> = vec![Vec::new(); nl];
        let mut passes = 0;
        let mut settled_at = None;
        while passes < limit {
            passes += 1;
            let t = tree(h, &d.outlet, &d.graph, &d.geom.len);
            let changed = prev_recv
                .as_ref()
                .map_or(n, |p| p.iter().zip(&t.recv).filter(|(a, b)| a != b).count());
            // Of the points that changed, how many went BACK to the receiver they had two passes ago (a two-step cycle).
            let back = match (prev_recv.as_ref(), prev2_recv.as_ref()) {
                (Some(p1), Some(p2)) => (0..n)
                    .filter(|&c| t.recv[c] != p1[c] && t.recv[c] == p2[c])
                    .count(),
                _ => 0,
            };
            let q = accumulate(&t, &d.water);
            for &c in &t.order {
                let c = c as usize;
                let r = t.recv[c];
                if d.outlet[c] || r == NONE {
                    h[c] = d.base[c];
                    continue;
                }
                let s = (up[c] / (k_lp * q[c].greater(tiny)))
                    .powf_fine(inv_n)
                    .lesser(tan);
                h[c] = h[r as usize] + s * t.link[c];
            }
            let now = means(h);
            let deltas: Vec<f64> = (0..nl)
                .filter(|&k| d.held[k])
                .map(|k| (now[k] - last[k]).abs().to_f64())
                .collect();
            let moved = deltas.iter().copied().fold(0.0, f64::max);
            let over = deltas.iter().filter(|&&x| x > SETTLE_M).count();
            let mean_moved = deltas.iter().sum::<f64>() / deltas.len().max(1) as f64;
            println!(
                "round {round} pass {passes:3}: moved max {moved:9.3} m mean {mean_moved:8.3} m, nodes over {SETTLE_M} m {over:3}, receivers changed {changed:7}, flooded {}",
                t.flooded
            );
            for k in 0..nl {
                node_series[k].push(now[k].to_f64());
            }
            println!("    ... of the changed, back to two passes ago: {back}");
            last = now;
            prev2_recv = prev_recv.take();
            prev_recv = Some(t.recv.clone());
            if moved <= settle.to_f64() && settled_at.is_none() {
                settled_at = Some(passes);
                break;
            }
        }
        println!("round {round}: settled at {settled_at:?} of limit {limit}");
        // The worst node over the last twenty passes: its mean, pass by pass.
        let worst = (0..nl)
            .filter(|&k| d.held[k])
            .max_by(|&a, &b| {
                let span = |k: usize| {
                    let s = &node_series[k][node_series[k].len().saturating_sub(20)..];
                    s.iter().copied().fold(f64::MIN, f64::max)
                        - s.iter().copied().fold(f64::MAX, f64::min)
                };
                span(a).total_cmp(&span(b))
            })
            .expect("a held node");
        let tail = &node_series[worst][node_series[worst].len().saturating_sub(20)..];
        println!(
            "round {round}: the worst node {worst}'s mean over the last passes: {:?}",
            tail.iter()
                .map(|x| (x * 10.0).round() / 10.0)
                .collect::<Vec<_>>()
        );
        for (k, &mean) in last.iter().enumerate() {
            let r = d.node_ref[k] as usize;
            if d.held[k] && mean > Gf::ZERO && r < u.len() {
                u[r] *= (d.node_z[k] / mean).powf_fine(half_n);
            }
        }
    }
}

/// ★ THE FIXED NETWORK (bench only; ruling B77's words: "the stream law's analytic solution on a drainage network built
/// first"): the drainage tree built ONCE from the fine start, then each round one exact pass of the joint law's steady
/// state on it (the heights integrated from the outlets) and the uplift corrected by the rounds' own law, until no rise
/// moves by more than a part in a thousand.
fn fixed_network(d: &Domain, u: &mut [Gf], h: &mut [Gf], limit: usize, rebuild_every: usize) {
    use vd_terrain::solve::{JOINT_K_LP, JOINT_N};
    let n = h.len();
    let (k_lp, inv_n, half_n) = (
        Gf::from_f64(JOINT_K_LP),
        Gf::from_f64(1.0 / JOINT_N),
        Gf::from_f64(0.5 * JOINT_N),
    );
    let tan = Gf::from_f64(vd_terrain::body::TAN_REPOSE);
    let tiny = Gf::from_f64(f64::MIN_POSITIVE);
    let nl = d.nodes.len();
    let step_exp = if std::env::var("VD_STEP").is_ok_and(|v| v == "full") {
        Gf::from_f64(JOINT_N)
    } else {
        half_n
    };
    let mut t = tree(h, &d.outlet, &d.graph, &d.geom.len);
    let mut q = accumulate(&t, &d.water);
    println!("fixed network: built once, flooded {}", t.flooded);
    let mut last_means: Option<Vec<f64>> = None;
    for round in 1..=limit {
        if rebuild_every > 0 && round > 1 && (round - 1) % rebuild_every == 0 {
            t = tree(h, &d.outlet, &d.graph, &d.geom.len);
            q = accumulate(&t, &d.water);
        }
        let ln_u: Vec<Gf> = u.iter().map(|x| x.ln_fine()).collect();
        let up: Vec<Gf> = (0..n)
            .map(|p| {
                let mut s = Gf::ZERO;
                for t in d.wstart[p] as usize..d.wstart[p + 1] as usize {
                    s += d.wval[t] * ln_u[d.wnode[t] as usize];
                }
                s.exp_fine()
            })
            .collect();
        for &c in &t.order {
            let c = c as usize;
            let r = t.recv[c];
            if d.outlet[c] || r == NONE {
                h[c] = d.base[c];
                continue;
            }
            let s = (up[c] / (k_lp * q[c].greater(tiny)))
                .powf_fine(inv_n)
                .lesser(tan);
            h[c] = h[r as usize] + s * t.link[c];
        }
        let (mut sum, mut area) = (vec![Gf::ZERO; nl], vec![Gf::ZERO; nl]);
        for ((&k, &hp), &a) in d.pnode.iter().zip(&*h).zip(&d.geom.area) {
            sum[k as usize] += hp * a;
            area[k as usize] += a;
        }
        let mut rise_moved = 0.0f64;
        for k in 0..nl {
            let r = d.node_ref[k] as usize;
            if d.held[k] && area[k] > Gf::ZERO && r < u.len() {
                let mean = sum[k] / area[k];
                if mean > Gf::ZERO {
                    let factor = (d.node_z[k] / mean).powf_fine(step_exp);
                    rise_moved = rise_moved.max((factor.to_f64() - 1.0).abs());
                    u[r] *= factor;
                }
            }
        }
        let now: Vec<f64> = (0..nl)
            .map(|k| {
                if area[k] > Gf::ZERO {
                    (sum[k] / area[k]).to_f64()
                } else {
                    0.0
                }
            })
            .collect();
        let moved = last_means.as_ref().map_or(f64::INFINITY, |l: &Vec<f64>| {
            (0..nl)
                .filter(|&k| d.held[k])
                .map(|k| (now[k] - l[k]).abs())
                .fold(0.0, f64::max)
        });
        last_means = Some(now);
        println!(
            "fixed round {round:3}: the largest rise correction {rise_moved:.5}, the means moved max {moved:.2} m"
        );
        if moved <= SETTLE_M && std::env::var("VD_SETTLE_LAW").is_ok() {
            println!("fixed: settled by the settle law at round {round}");
            return;
        }
    }
    println!("fixed: not settled in {limit}");
}

/// ★ THE PICTURES (bench only): the same tile from the same start by four methods, each a shaded relief of its fine
/// heights (each point drawn in its cell, the cells of the tile's own face; light from the north-west, the same scale for
/// all), side by side in one PNG: the game's rounds today, the fixed network at 20 and 60 rounds, and the network
/// rebuilt every five rounds at 120.
fn pictures(d: &Domain, u0: &[Gf], h0: &[Gf], face: u8) {
    let methods: [(&str, usize, usize); 4] = [
        ("game", 60, usize::MAX),
        ("fixed20", 20, 0),
        ("fixed60", 60, 0),
        ("rebuild5x120", 120, 5),
    ];
    let mut panels = Vec::new();
    for (name, limit, every) in methods {
        let (mut u, mut h) = (u0.to_vec(), h0.to_vec());
        let started = std::time::Instant::now();
        if every == usize::MAX {
            instrumented_rounds(d, &mut u, &mut h, ROUNDS, limit);
        } else {
            fixed_network(d, &mut u, &mut h, limit, every);
        }
        println!(
            "PICTURE {name}: {:.0} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        fit(d, &h);
        panels.push(relief(d, &h, face));
    }
    let (w, ht) = (panels[0].width(), panels[0].height());
    let gap = 8;
    let mut out = image::RgbImage::from_pixel(w * 4 + gap * 3, ht, image::Rgb([255, 255, 255]));
    for (k, p) in panels.iter().enumerate() {
        image::imageops::overlay(&mut out, p, i64::from(k as u32 * (w + gap)), 0);
    }
    let path = std::env::var("VD_PICTURE").unwrap_or_else(|_| "/tmp/fine_methods.png".to_owned());
    out.save(&path).expect("the picture saves");
    println!(
        "PICTURES saved to {path} (left to right: the game today, fixed 20, fixed 60, rebuilt every 5 at 120)"
    );
}

/// One shaded relief: each cell the mean of its points' heights, holes filled from a neighbour, light from the north-west.
fn relief(d: &Domain, h: &[Gf], face: u8) -> image::RgbImage {
    let on: Vec<usize> = (0..h.len()).filter(|&p| d.cells[p].face == face).collect();
    let (i0, i1) = on.iter().fold((i32::MAX, i32::MIN), |(a, b), &p| {
        (a.min(d.cells[p].i), b.max(d.cells[p].i))
    });
    let (j0, j1) = on.iter().fold((i32::MAX, i32::MIN), |(a, b), &p| {
        (a.min(d.cells[p].j), b.max(d.cells[p].j))
    });
    let (w, ht) = ((i1 - i0 + 1) as usize, (j1 - j0 + 1) as usize);
    let (mut sum, mut cnt) = (vec![0.0f64; w * ht], vec![0u32; w * ht]);
    for &p in &on {
        let at = (d.cells[p].j - j0) as usize * w + (d.cells[p].i - i0) as usize;
        sum[at] += h[p].to_f64();
        cnt[at] += 1;
    }
    let mut z: Vec<f64> = (0..w * ht)
        .map(|k| {
            if cnt[k] > 0 {
                sum[k] / f64::from(cnt[k])
            } else {
                f64::NAN
            }
        })
        .collect();
    for _ in 0..4 {
        for y in 0..ht {
            for x in 0..w {
                if z[y * w + x].is_nan() {
                    let near = [
                        (x.wrapping_sub(1), y),
                        (x + 1, y),
                        (x, y.wrapping_sub(1)),
                        (x, y + 1),
                    ];
                    if let Some(v) = near
                        .iter()
                        .filter(|(a, b)| *a < w && *b < ht)
                        .map(|(a, b)| z[b * w + a])
                        .find(|v| !v.is_nan())
                    {
                        z[y * w + x] = v;
                    }
                }
            }
        }
    }
    // The cell's size in metres (rung 7, 128 m) and the light: azimuth 315 degrees, 45 degrees up.
    let cell = 128.0;
    let (az, alt) = (315f64.to_radians(), 45f64.to_radians());
    let mut img = image::RgbImage::new(w as u32, ht as u32);
    for y in 0..ht {
        for x in 0..w {
            let g = |a: usize, b: usize| {
                let v = z[b.min(ht - 1) * w + a.min(w - 1)];
                if v.is_nan() { z[y * w + x] } else { v }
            };
            let dzdx = (g(x + 1, y) - g(x.saturating_sub(1), y)) / (2.0 * cell);
            let dzdy = (g(x, y + 1) - g(x, y.saturating_sub(1))) / (2.0 * cell);
            let slope = (dzdx * dzdx + dzdy * dzdy).sqrt().atan();
            let aspect = dzdy.atan2(-dzdx);
            let shade = alt.sin() * slope.cos() + alt.cos() * slope.sin() * (az - aspect).cos();
            let v = if z[y * w + x].is_nan() {
                0
            } else {
                (shade.clamp(0.0, 1.0) * 255.0) as u8
            };
            img.put_pixel(x as u32, y as u32, image::Rgb([v, v, v]));
        }
    }
    img
}
