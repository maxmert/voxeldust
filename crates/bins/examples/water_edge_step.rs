//! ★ HOW FAR THE WATER'S EDGE MOVES BETWEEN THE NEAR RUNGS (2026-09-22; the owner, after ruling
//! W10: *"It's not only far-sight. When I fly very close over water, it also changes all the
//! time."*).
//!
//! `shore_step` measures the SEA's shore against the sea's one level. This measures the edge of
//! whatever water a column really holds — the sea, or a LAKE at its own level — the way the chunk
//! builder and the morph read it: the cell's own side through `sample_side`, then the water that
//! side names through `column_water`.
//!
//! ★ 2026-09-23 (ruling W16): step 4's DRAWN rivers are RETIRED, so no column holds a stream's own
//! surface any more. The `river` mode still runs, and it must now read NO river water at all —
//! which is the gate that says the stamp is gone.
//!
//! Three numbers per rung pair:
//! * `edge step` — how far the crossing of the ground and its water moves sideways, in metres.
//! * `water flips` — at how many sample points the column HOLDS a stream's surface at one rung and
//!   not at the other. Since ruling W16 no column ever holds one, so this reads ZERO everywhere.
//! * `level step` — the median change of the water's own level, in metres, where both rungs state
//!   one.
//!
//! `cargo run --release -p vd-bins --example water_edge_step -- <sea|lake|river> [dx dy dz]
//! [lines] [half_length_m] [step_m] [rung_lo] [rung_hi]`

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_core::glam::DVec3;
use vd_terrain::artifact::{PyramidField, ZField, column_water, sample_side, sample_z};
use vd_terrain::height::height_field_m;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::solve::{FACIES_LAKE, FACIES_SEA};
use vd_terrain::units::metres_of_q28;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().cloned().unwrap_or_else(|| "sea".to_string());
    let num =
        |k: usize, d: f64| -> f64 { args.get(k).and_then(|s| s.parse::<f64>().ok()).unwrap_or(d) };
    let aim = DVec3::new(num(1, 0.617_270), num(2, -0.437_286), num(3, -0.654_033)).normalize();
    let lines = num(4, 200.0) as usize;
    let half = num(5, 4_000.0);
    let step_m = num(6, 16.0);
    let rung_lo = num(7, 0.0) as u8;
    let rung_hi = num(8, 9.0) as u8;

    let body = home_planet();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let radius = body.radius_m();
    let fine = MacroLattice::of(&body).expect("a lattice");
    let levels = artifact.pyramid.len() as u32;
    let counts = artifact.coast_counts();

    // ★ THE STAND: the node near the aim that best answers the mode — a lake node with the most
    // lake neighbours, the trunk with the largest discharge class, or the aim itself for the sea.
    let face = vd_seed::bend::face_of(aim.to_array());
    let (ta, sa) = vd_seed::bend::face_coords(face, aim.to_array());
    let ci = vd_seed::ladder::index_of(vd_seed::bend::unbend(ta), fine.edge);
    let cj = vd_seed::ladder::index_of(vd_seed::bend::unbend(sa), fine.edge);
    // ★ THE SEARCH'S REACH. A coast or a river is near the belt stand, so a window of sixty nodes
    // (about 490 km) around it is enough. A LAKE is not: the nearest one to the belt stand is
    // further than that window, so the lake search walks the WHOLE globe and takes the node with
    // the most lake neighbours — the biggest lake the solve left.
    let window = 60i32;
    let whole = mode == "lake";
    let faces: Vec<u8> = if whole {
        (0..6).collect()
    } else {
        vec![face.index()]
    };
    let mut best: Option<(i64, u32)> = None;
    for f in faces {
        let face = vd_seed::bend::Face::from_index(f).expect("a face");
        let (jlo, jhi, ilo, ihi) = if whole {
            (0, fine.edge as i32, 0, fine.edge as i32)
        } else {
            (
                (cj - window).max(0),
                (cj + window).min(fine.edge as i32 - 1),
                (ci - window).max(0),
                (ci + window).min(fine.edge as i32 - 1),
            )
        };
        for j in jlo..jhi {
            for i in ilo..ihi {
                let node = fine.index(face, i, j);
                let row = &artifact.rows[node as usize];
                let facies = row.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT;
                let score = match mode.as_str() {
                    "lake" => {
                        if facies & FACIES_LAKE == 0 {
                            continue;
                        }
                        i64::from(
                            fine.neighbours(node)
                                .iter()
                                .filter(|&&m| {
                                    m != vd_terrain::macro_lattice::NO_NODE
                                        && artifact.rows[m as usize].receiver_facies
                                            >> vd_terrain::artifact::FACIES_SHIFT
                                            & FACIES_LAKE
                                            != 0
                                })
                                .count() as u32,
                        )
                    }
                    "river" => {
                        if facies & (FACIES_SEA | FACIES_LAKE) != 0 {
                            continue;
                        }
                        i64::from(row.discharge_log)
                    }
                    // THE SEA'S OWN COAST: the node nearest the aim that the solve marked a coast, so
                    // the lines really cross a shore instead of walking one plain.
                    "coast" => {
                        if facies & vd_terrain::solve::FACIES_COAST == 0 {
                            continue;
                        }
                        -i64::from((i - ci).abs() + (j - cj).abs())
                    }
                    _ => 0,
                };
                if best.is_none_or(|(s, _)| score > s) {
                    best = Some((score, node));
                }
            }
        }
    }
    let (score, node) = best.expect("a node near the aim");
    let stand = if mode == "sea" || score == i64::MIN {
        aim
    } else {
        let d = fine.direction(node);
        let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
        DVec3::new(
            d[0].raw() as f64 / unit,
            d[1].raw() as f64 / unit,
            d[2].raw() as f64 / unit,
        )
        .normalize()
    };
    println!(
        "water_edge_step: mode {mode}, stand node {node} (score {score}), {lines} lines of {:.0} m, step {step_m} m, rungs {rung_lo}..{rung_hi}",
        2.0 * half
    );

    let rungs: Vec<u8> = (rung_lo..=rung_hi).collect();
    let fields: Vec<Box<dyn ZField>> = rungs
        .iter()
        .map(|&r| {
            let level = PyramidField::level_for(&fine, levels, r);
            if level == 0 {
                Box::new(artifact.clone()) as Box<dyn ZField>
            } else {
                Box::new(PyramidField::of(&artifact, level, &counts).expect("a level"))
                    as Box<dyn ZField>
            }
        })
        .collect();

    // The column's ground and its water surface, in metres, the way the chunk builder reads them.
    let read = |field: &dyn ZField, rung: u8, p: DVec3| -> Option<(f64, f64, bool)> {
        let dir = p.to_array();
        let face = vd_seed::bend::face_of(dir);
        let (t, s) = vd_seed::bend::face_coords(face, dir);
        let n0 = body.ladder().cells_per_edge(0);
        let i = vd_seed::ladder::index_of(vd_seed::bend::unbend(t), n0);
        let j = vd_seed::ladder::index_of(vd_seed::bend::unbend(s), n0);
        let lattice = fine.coarser(field.level())?;
        let _ = sample_z(&lattice, field, face, 0, i, j)?;
        // ★ THE SHIPPED PATH'S OWN READ (2026-09-23, ruling W16): the cell's SIDE at this rung,
        // then the water that side names — the very two calls `height_field_m` makes, so the
        // instrument can never measure a water the ground does not hold.
        let cell_i =
            vd_seed::ladder::index_of(vd_seed::bend::unbend(t), body.ladder().cells_per_edge(rung));
        let cell_j =
            vd_seed::ladder::index_of(vd_seed::bend::unbend(s), body.ladder().cells_per_edge(rung));
        let cell_side = sample_side(&fine, field, face, rung, cell_i, cell_j);
        let (water, _, _) = column_water(&body, &lattice, field, face, 0, i, j, cell_side);
        // ★ 2026-09-23 (ruling W16): step 4's drawn lines are RETIRED, so no column holds a
        // stream's own surface any more. A column's water is the row's, and nothing else — which
        // is exactly what the `river` mode must now read: no river water at all.
        let (w_m, stream) = (metres_of_q28(water), false);
        let h = height_field_m(&body, field, dir, rung)?;
        Some((h, w_m, stream))
    };

    let seed = if stand.z.abs() < 0.9 {
        DVec3::Z
    } else {
        DVec3::X
    };
    let east = stand.cross(seed).normalize();
    let north = east.cross(stand).normalize();
    let n = (2.0 * half / step_m) as usize;
    let mut steps: Vec<Vec<f64>> = vec![Vec::new(); rungs.len() - 1];
    let mut flips: Vec<u64> = vec![0; rungs.len() - 1];
    let mut level_steps: Vec<Vec<f64>> = vec![Vec::new(); rungs.len() - 1];
    let mut samples = 0u64;
    // ★ IS THE WATER DRAWN AT ALL? How many sample columns stand UNDER their own water at each
    // rung. A lake whose floor the shore law lifts over its own surface reads zero here, and the
    // sheet over it is buried: the lake is not drawn.
    let mut under: Vec<u64> = vec![0; rungs.len()];
    let mut held: Vec<u64> = vec![0; rungs.len()];
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..lines {
        let off = east * ((next() - 0.5) * half) + north * ((next() - 0.5) * half);
        let ang = next() * std::f64::consts::TAU;
        let dir = (east * ang.cos() + north * ang.sin()).normalize();
        let centre = (stand * radius + off).normalize() * radius;
        let mut gap: Vec<Vec<f64>> = vec![Vec::with_capacity(n); rungs.len()];
        for k in 0..n {
            let p = (centre + dir * ((k as f64) * step_m - half)).normalize();
            let mut row: Vec<Option<(f64, f64, bool)>> = Vec::with_capacity(rungs.len());
            for (ri, r) in rungs.iter().enumerate() {
                row.push(read(&*fields[ri], *r, p));
            }
            samples += 1;
            for ri in 0..rungs.len() - 1 {
                if let (Some(a), Some(b)) = (row[ri], row[ri + 1]) {
                    if a.2 != b.2 {
                        flips[ri] += 1;
                    } else if a.2 {
                        level_steps[ri].push((a.1 - b.1).abs());
                    }
                }
            }
            for (ri, cell) in row.iter().enumerate() {
                if let Some((h, w, stream)) = *cell {
                    if h < w {
                        under[ri] += 1;
                    }
                    if stream {
                        held[ri] += 1;
                    }
                }
                gap[ri].push(cell.map_or(f64::NAN, |(h, w, _)| h - w));
            }
        }
        let first_cross = |v: &[f64]| -> Option<f64> {
            for k in 1..v.len() {
                if v[k - 1].is_nan() || v[k].is_nan() {
                    continue;
                }
                if (v[k - 1] > 0.0) != (v[k] > 0.0) {
                    let t = v[k - 1] / (v[k - 1] - v[k]);
                    return Some(((k - 1) as f64 + t) * step_m);
                }
            }
            None
        };
        let xs: Vec<Option<f64>> = gap.iter().map(|v| first_cross(v)).collect();
        for ri in 0..rungs.len() - 1 {
            if let (Some(a), Some(b)) = (xs[ri], xs[ri + 1]) {
                steps[ri].push((a - b).abs());
            }
        }
    }
    println!("samples per rung: {samples}");
    println!("rung | samples under their own water | samples holding a stream's surface");
    for (ri, r) in rungs.iter().enumerate() {
        println!("{r:>4} | {:>29} | {:>33}", under[ri], held[ri]);
    }
    println!(
        "rung pair | lines | edge step median m | p90 m | max m | water flips | level step median m | cell m"
    );
    for ri in 0..rungs.len() - 1 {
        let mut s = steps[ri].clone();
        s.sort_by(f64::total_cmp);
        let mut l = level_steps[ri].clone();
        l.sort_by(f64::total_cmp);
        let med = |v: &[f64]| {
            if v.is_empty() {
                f64::NAN
            } else {
                v[v.len() / 2]
            }
        };
        println!(
            "{:>3} → {:<3} | {:>5} | {:>18.1} | {:>5.1} | {:>5.1} | {:>11} | {:>19.2} | {}",
            rungs[ri],
            rungs[ri + 1],
            s.len(),
            med(&s),
            if s.is_empty() {
                f64::NAN
            } else {
                s[(s.len() * 9 / 10).min(s.len() - 1)]
            },
            if s.is_empty() {
                f64::NAN
            } else {
                s[s.len() - 1]
            },
            flips[ri],
            med(&l),
            vd_seed::ladder::cell_m(rungs[ri + 1])
        );
    }
}
