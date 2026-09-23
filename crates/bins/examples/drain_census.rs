//! ★ THE DRAIN CENSUS — gate G-DRAIN (ruling B2 step 4, 2026-09-22).
//!
//! **THE QUESTION.** A drained land has FEW CLOSED HOLLOWS: rain that falls anywhere runs to a
//! stream, and a place with nowhere to run is rare. A land shaped by noise alone has a closed
//! hollow wherever two ripples cross, which is what makes it read as dunes from the air. So the
//! gate counts, over the FINE rungs and on three stands, how many columns are LOCAL MINIMA — a
//! column lower than all eight of its neighbours — and how many of those minima stand on a
//! drainage line.
//!
//! **ONE KERNEL, THE SHIPPED ONE.** The instrument never holds two generators: it calls the ONE
//! column kernel (`vd_recipe::plan::column_surface_at`) with the very words a chunk's column pass
//! hands it.
//!
//! ★ **RE-POINTED 2026-09-23 (ruling W16).** Step 4's DRAWN drainage is retired - the stamped
//! tributary lines, the valley profile and the channel trench are gone, and with them the census's
//! before/after pair and its long-axis histogram of the drawn lines. What the gate measures now is
//! the number the owner's law is about: how many CLOSED HOLLOWS the shipped ground holds at the
//! fine rungs, on the three stands. The number is a READING of the shipped field, and a rise in it
//! is a defect in whatever last moved the fine relief.
//!
//! **IT ALSO MEASURES THE COST** of the shipped column pass (`column_field`), so a change to the
//! kernel can never make the ground cheap by accident.
//!
//! `cargo run --release -p vd-bins --example drain_census -- [rung] [blocks]`

use std::time::Instant;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_core::glam::DVec3;
use vd_recipe::Gi;
use vd_recipe::plan::{FieldRead, column_surface_at, site_direction};
use vd_seed::bend::Face;
use vd_terrain::artifact::Artifact;
use vd_terrain::chunk::{CHUNK_EDGE, column_field};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::solve::{FACIES_LAKE, FACIES_SEA};

/// How many columns a sampled block holds each way — one chunk's own edge, so the gather the
/// instrument runs is the gather a chunk runs.
const BLOCK: i32 = CHUNK_EDGE as i32;

/// One stand's reading.
struct Stand {
    name: &'static str,
    dir: DVec3,
}

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    let rung = a.first().map_or(3u8, |v| *v as u8);
    let blocks = a.get(1).map_or(12i32, |v| *v as i32);
    let body = home_planet();
    let started = Instant::now();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    println!(
        "drain_census: solved in {:.1} s; the sea stands at {} m",
        started.elapsed().as_secs_f64(),
        artifact.sea().unwrap_or(0)
    );
    let body = home_planet().with_sea_m(artifact.sea());
    let lattice = MacroLattice::of(&body).expect("a macro lattice");
    // The rung's cell, in whole millimetres, off the body's own ladder.
    let cell_mm = i64::from(vd_seed::ladder::cell_m(rung)) * 1_000;
    println!(
        "the rung is {rung} ({} m cells); each stand samples {blocks}x{blocks} blocks of {BLOCK} columns = {} m a side",
        cell_mm as f64 / 1_000.0,
        (blocks * BLOCK) as f64 * cell_mm as f64 / 1_000.0
    );

    let stands = pick_stands(&artifact, &lattice);
    for st in &stands {
        println!(
            "  stand {:<10} aim {:.6} {:.6} {:.6}  (feed it to `land_stand -- 2000 dx dy dz`)",
            st.name, st.dir.x, st.dir.y, st.dir.z
        );
    }
    for stand in &stands {
        let (face, i0, j0) = cell_of(&body, stand.dir, rung);
        let n = blocks * BLOCK;
        let half = n / 2;
        let (i0, j0) = (i0 - half, j0 - half);
        let n_cells = body.ladder().cells_per_edge(rung) as i32;
        if i0 < 1 || j0 < 1 || i0 + n >= n_cells || j0 + n >= n_cells {
            println!(
                "\n{}: REFUSED — the stand stands too near a face edge for a whole sample",
                stand.name
            );
            continue;
        }
        let charter = body.plan_charter(rung, face);
        // ONE height map of the shipped ground, from the ONE kernel.
        let mut ground = vec![0i64; (n * n) as usize];
        let mut kernel_us = 0.0f64;
        let mut a = 0;
        while a < n {
            let mut b = 0;
            while b < n {
                let (ci, cj) = (i0 + a, j0 + b);
                let dir = site_direction(&charter, i32::from(face.index()), ci, cj);
                let Some(z) =
                    vd_terrain::artifact::sample_z(&lattice, &artifact, face, rung, ci, cj)
                else {
                    b += 1;
                    continue;
                };
                let slope_share = vd_terrain::artifact::slope_share(
                    &vd_terrain::artifact::slope_charter(&body, &lattice, rung),
                    &lattice,
                    &artifact,
                    face,
                    rung,
                    ci,
                    cj,
                )
                .unwrap_or(Gi::ZERO);
                let cell_side =
                    vd_terrain::artifact::sample_side(&lattice, &artifact, face, rung, ci, cj);
                let (water, side, _) = vd_terrain::artifact::column_water(
                    &body, &lattice, &artifact, face, rung, ci, cj, cell_side,
                );
                let t2 = Instant::now();
                let got = column_surface_at(
                    &charter,
                    dir,
                    &FieldRead {
                        z,
                        first: body.first_fine(),
                        slope_share,
                        water,
                        side,
                    },
                );
                kernel_us += t2.elapsed().as_secs_f64() * 1.0e6;
                ground[(a * n + b) as usize] = got.h.raw();
                b += 1;
            }
            a += 1;
        }
        let min_count = minima(&ground, n);
        let deep = depths(&ground, n);
        let inner = ((n - 2) * (n - 2)) as f64;
        println!(
            "\nG-DRAIN, {}: {} columns measured\n  closed local minima {} ({:.3} % of the columns)",
            stand.name,
            (n * n),
            min_count,
            min_count as f64 * 100.0 / inner
        );
        println!(
            "  how deep those hollows are: {} of a tenth of a metre, {} of a metre, {} of ten metres, the deepest {:.2} m",
            deep.0, deep.1, deep.2, deep.3
        );
        let (di, dj, deepest) = deepest_cell(&ground, n);
        println!(
            "  the deepest hollow stands at cell ({}, {}) of face {face:?}, {deepest:.2} m under its lowest neighbour",
            i0 + di,
            j0 + dj
        );
        let columns = f64::from(n * n);
        println!(
            "  the cost: the kernel itself {:.0} ns a column",
            kernel_us * 1_000.0 / columns
        );
    }
    // ★ THE SHIPPED PATH'S OWN COST, on the very function a worker calls.
    let stand = stands.first().expect("a stand");
    let (face, i0, j0) = cell_of(&body, stand.dir, rung);
    let (x, y) = (i0 / BLOCK, j0 / BLOCK);
    let _ = j0;
    let rounds = 8;
    let t = Instant::now();
    let mut k = 0;
    while k < rounds {
        let got = column_field(&body, Some(&artifact), face, rung, x + k, y);
        assert!(got.is_some(), "the stand's chunk builds");
        k += 1;
    }
    let us = t.elapsed().as_secs_f64() * 1.0e6 / f64::from(rounds);
    println!(
        "the shipped column pass with the artifact: {us:.0} us a chunk, {:.0} ns a column",
        us * 1_000.0 / f64::from(BLOCK * BLOCK)
    );
}

/// The `(face, i, j)` cell of a direction at a rung.
fn cell_of(body: &vd_terrain::BodyDefinition, dir: DVec3, rung: u8) -> (Face, i32, i32) {
    let d = [dir.x, dir.y, dir.z];
    let face = vd_seed::bend::face_of(d);
    let (t, s) = vd_seed::bend::face_coords(face, d);
    let n = body.ladder().cells_per_edge(rung);
    (
        face,
        vd_seed::ladder::index_of(vd_seed::bend::unbend(t), n),
        vd_seed::ladder::index_of(vd_seed::bend::unbend(s), n),
    )
}

/// How many columns of a square map are lower than all eight of their neighbours.
fn minima(map: &[i64], n: i32) -> usize {
    let mut count = 0usize;
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            if is_min(map, n, i, j) {
                count += 1;
            }
        }
    }
    count
}

/// ★ A HOLLOW MUST BE A HOLLOW, AND THE FLOOR IS STATED. A column lower than all eight of its
/// neighbours by ONE UNIT of the height word is not a landform: the word carries 2^-28 of a gap
/// step, which is 3 x 10^-11 m, and the arithmetic of any blend leaves such a step wherever two
/// smooth terms cross. A closed hollow counts here only where it stands at least
/// [`HOLLOW_FLOOR_MM`] under its lowest neighbour - a tenth of a metre, which is under the solved
/// row's own whole-metre word and still a hollow a boot would find.
const HOLLOW_FLOOR_MM: i64 = 100;

/// How many hollows stand at least a tenth of a metre, a metre and ten metres under their lowest
/// neighbour, and how deep the deepest one is in metres. A hollow's DEPTH is what says whether it
/// is a pond a player would see or a step the arithmetic left between two smooth terms.
fn depths(map: &[i64], n: i32) -> (usize, usize, usize, f64) {
    let unit = (vd_terrain::units::STEPS_PER_M) << vd_recipe::noise::NOISE_BITS;
    let (mut tenth, mut one, mut ten, mut worst) = (0usize, 0usize, 0usize, 0i64);
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            let here = map[(i * n + j) as usize];
            let mut low = i64::MAX;
            for di in -1..=1 {
                for dj in -1..=1 {
                    if (di == 0) & (dj == 0) {
                        continue;
                    }
                    low = low.min(map[((i + di) * n + (j + dj)) as usize]);
                }
            }
            let deep = low - here;
            if deep <= 0 {
                continue;
            }
            worst = worst.max(deep);
            if deep >= unit / 10 {
                tenth += 1;
            }
            if deep >= unit {
                one += 1;
            }
            if deep >= unit * 10 {
                ten += 1;
            }
        }
    }
    (tenth, one, ten, worst as f64 / unit as f64)
}

/// The deepest hollow's own cell and its depth in metres - so a run that finds one can be told
/// WHERE.
fn deepest_cell(map: &[i64], n: i32) -> (i32, i32, f64) {
    let unit = (vd_terrain::units::STEPS_PER_M) << vd_recipe::noise::NOISE_BITS;
    let (mut bi, mut bj, mut worst) = (0i32, 0i32, 0i64);
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            let here = map[(i * n + j) as usize];
            let mut low = i64::MAX;
            for di in -1..=1 {
                for dj in -1..=1 {
                    if (di == 0) & (dj == 0) {
                        continue;
                    }
                    low = low.min(map[((i + di) * n + (j + dj)) as usize]);
                }
            }
            if low - here > worst {
                worst = low - here;
                bi = i;
                bj = j;
            }
        }
    }
    (bi, bj, worst as f64 / unit as f64)
}

fn is_min(map: &[i64], n: i32, i: i32, j: i32) -> bool {
    let here = map[(i * n + j) as usize];
    let floor = ((HOLLOW_FLOOR_MM * vd_terrain::units::STEPS_PER_M)
        << vd_recipe::noise::NOISE_BITS)
        / 1_000;
    for di in -1..=1 {
        for dj in -1..=1 {
            if (di == 0) & (dj == 0) {
                continue;
            }
            if map[((i + di) * n + (j + dj)) as usize] - here < floor {
                return false;
            }
        }
    }
    true
}

/// ★ THE THREE STANDS (ruling B2): the belt the owner flew, a COAST and a PLAIN, both picked off
/// the census's own map rather than typed in — the coast is a dry node beside a sea node with the
/// greatest discharge near it, and the plain is the land node of the flattest neighbourhood far
/// from any water.
fn pick_stands(artifact: &Artifact, lattice: &MacroLattice) -> Vec<Stand> {
    let n = artifact.node_count();
    let facies: Vec<u8> = artifact
        .rows
        .iter()
        .map(|r| r.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
        .collect();
    let mut coast = (0u32, 0u8);
    let mut plain = (0u32, i32::MAX);
    for node in 0..n as u32 {
        let f = facies[node as usize];
        if (f & FACIES_SEA != 0) | (f & FACIES_LAKE != 0) {
            continue;
        }
        let here = artifact.rows[node as usize].z_m;
        let neigh = lattice.neighbours(node);
        let wet = neigh
            .iter()
            .any(|&m| m != u32::MAX && facies[m as usize] & FACIES_SEA != 0);
        if wet {
            let q = artifact.rows[node as usize].discharge_log;
            if q > coast.1 {
                coast = (node, q);
            }
        } else {
            let mut spread = 0i32;
            let mut dry = true;
            for &m in &neigh {
                if m == u32::MAX {
                    dry = false;
                    break;
                }
                let f = facies[m as usize];
                if (f & FACIES_SEA != 0) | (f & FACIES_LAKE != 0) {
                    dry = false;
                    break;
                }
                spread += (i32::from(artifact.rows[m as usize].z_m) - i32::from(here)).abs();
            }
            // A plain is flat AND standing well over the sea, so it is ground and not a shelf.
            if dry && here > 200 && spread < plain.1 {
                plain = (node, spread);
            }
        }
    }
    let of = |node: u32| -> DVec3 {
        let d = lattice.direction(node);
        DVec3::new(d[0].raw() as f64, d[1].raw() as f64, d[2].raw() as f64).normalize()
    };
    println!(
        "the stands: the belt, the coast at node {} (discharge class {}), the plain at node {} (its ring spreads {} steps)",
        coast.0, coast.1, plain.0, plain.1
    );
    vec![
        Stand {
            name: "the belt",
            dir: DVec3::new(0.617_270, -0.437_286, -0.654_033).normalize(),
        },
        Stand {
            name: "the coast",
            dir: of(coast.0),
        },
        Stand {
            name: "a plain",
            dir: of(plain.0),
        },
    ]
}
