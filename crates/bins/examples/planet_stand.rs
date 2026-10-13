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
use vd_terrain::fine::l1::{TILE_NODES, TileKey, solve_l1};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};

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
        _ => panic!("solve PLANET OUT.art [cores] | stands PLANET WORLD.art OUT_DIR"),
    }
}
