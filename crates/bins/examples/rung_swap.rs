//! ★ WHAT CHANGES AT A RUNG SWAP: THE GROUND'S OWN HEIGHT, AND THE SHADE IT TAKES (2026-09-23; the
//! owner, after ruling W16: *"Now it's way better, but for some of the far-view rungs the change is
//! still visible — not for close or very far view"*).
//!
//! The near rungs read the artifact's ROWS and the far rungs read a PYRAMID LEVEL, whose node is
//! the MEAN of four of the level under it. On the home planet rungs 0 to 9 read the rows, 10 to 14
//! read level 1, 15 level 2, 16 level 3. So three swaps change the FIELD the ground is read from —
//! 9 → 10, 14 → 15, 15 → 16 — and a mean of four is not the recipe's dropped octaves.
//!
//! **THE TWO READINGS, per rung pair, and either can fail.**
//!
//! 1. **THE SHAPE.** Along a grid of directions, the surface at rung `L` and at rung `L + 1`, both
//!    through the shipped reader `vd_terrain::height::height_field_m` — which is the very function
//!    the client's geomorph calls for a vertex with no parent triangle on its radial. The
//!    difference is printed in metres (median, p90, max), in CELLS of the coarser rung, and in
//!    PIXELS at the distance the swap happens (the finer rung's own switch distance). Beside it
//!    stand the two numbers the ladder BELIEVES: `handover_step_m` — the body's own step bound,
//!    read from the OCTAVE TABLE alone — and `sink_m` of the coarser rung, which is how far the
//!    coarser mesh is dropped so that it cannot show through the finer one. A difference over the
//!    sink is a coarse mesh POKING THROUGH a finer one, which is a ring of the wrong ground.
//!
//! 2. **THE SHADE.** For the same ground — one chunk at rung `L + 1` and the four chunks at rung
//!    `L` under it — the mean Lambert term `max(0, n · l)` over the extracted mesh's own smooth
//!    normals, at a STATED sun. The two rungs draw the same ground with the same material, so a
//!    difference in that mean is a difference in BRIGHTNESS, and the eye's working threshold is
//!    2 % of the local mean (Blackwell 1946; Barten 1999; DICOM PS 3.14).
//!
//! `cargo run --release -p vd-bins --example rung_swap -- [rung_lo] [rung_hi] [grid] [sun_deg]`

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client::chunks::{sink_m, smooth_normals};
use vd_client::ladder_view::{handover_step_m, pixel_rad, switch_m};
use vd_core::glam::DVec3;
use vd_seed::bend::{Face, direction};
use vd_terrain::artifact::{PyramidField, ZField};
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};
use vd_terrain::digest::surface_column_field;
use vd_terrain::extract::extract;
use vd_terrain::height::height_field_m;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::lattice::sample_box;
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::position::vertex_position_m;

/// THE BELT the owner flies over, as a unit direction in the planet's frame.
const BELT: [f64; 3] = [0.617_270, -0.437_286, -0.654_033];

/// Whether a direction's column stands under its own water at a rung, and the water's own radius
/// in metres — read the way the chunk's column pass and the morph read it (ruling W16 fault B: the
/// cell's own SIDE through `sample_side`, then the water that side names through `column_water`).
fn wet_at(
    body: &vd_terrain::BodyDefinition,
    fine: &MacroLattice,
    field: &dyn ZField,
    dir: [f64; 3],
    rung: u8,
) -> Option<(bool, f64)> {
    let lattice = fine.coarser(field.level())?;
    let face = vd_seed::bend::face_of(dir);
    let (t, s) = vd_seed::bend::face_coords(face, dir);
    let n0 = body.ladder().cells_per_edge(0);
    let i = vd_seed::ladder::index_of(vd_seed::bend::unbend(t), n0);
    let j = vd_seed::ladder::index_of(vd_seed::bend::unbend(s), n0);
    let cell_i =
        vd_seed::ladder::index_of(vd_seed::bend::unbend(t), body.ladder().cells_per_edge(rung));
    let cell_j =
        vd_seed::ladder::index_of(vd_seed::bend::unbend(s), body.ladder().cells_per_edge(rung));
    let side = vd_terrain::artifact::sample_side(fine, field, face, rung, cell_i, cell_j);
    let (water, _, _) =
        vd_terrain::artifact::column_water(body, &lattice, field, face, 0, i, j, side);
    let ground = height_field_m(body, field, dir, rung)?;
    let water_m = vd_terrain::units::metres_of_q28(water);
    Some((water_m > ground, water_m))
}

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    let body = home_planet();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let levels = artifact.pyramid.len() as u32;
    let counts = artifact.coast_counts();
    let top = body.ladder().rungs - 1;
    let rung_lo = a.first().map_or(8u8, |v| *v as u8);
    let rung_hi = a
        .get(1)
        .map_or(17u8.min(top - 1), |v| (*v as u8).min(top - 1));
    let grid = a.get(2).map_or(40i32, |v| *v as i32);
    let sun_deg = a.get(3).map_or(22.0, |v| *v);
    let field_of = |rung: u8| -> Box<dyn ZField> {
        let level = PyramidField::level_for(&lattice, levels, rung);
        if level == 0 {
            Box::new(artifact.clone())
        } else {
            Box::new(PyramidField::of(&artifact, level, &counts).expect("a level"))
        }
    };
    // THE SUN, stated: `sun_deg` over the belt's own horizon, in the planet's frame.
    let up = DVec3::from_array(BELT).normalize();
    let east = up.cross(DVec3::Z).normalize();
    let sun = (up * sun_deg.to_radians().sin() + east * sun_deg.to_radians().cos()).normalize();
    println!(
        "rung_swap: the home planet, {} rungs, the sea at {} m, {levels} pyramid levels; the sun {sun_deg:.0}° over the belt",
        body.ladder().rungs,
        artifact.sea().unwrap_or(0)
    );

    println!("\nTHE SHAPE — {} directions a rung pair", 6 * grid * grid);
    println!(
        "L -> L+1 | level | cell(L+1) m | the swap m | median m | p90 m | p99 m | max m | max cells | p99 px | max px | the stated step m | sink(L+1) m | over the sink"
    );
    let faces = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];
    for rung in rung_lo..=rung_hi {
        let (fine, coarse) = (field_of(rung), field_of(rung + 1));
        let mut diffs: Vec<f64> = Vec::with_capacity((6 * grid * grid) as usize);
        for face in faces {
            for i in 0..grid {
                for j in 0..grid {
                    let a = (2.0 * (f64::from(i) + 0.5) / f64::from(grid)) - 1.0;
                    let b = (2.0 * (f64::from(j) + 0.5) / f64::from(grid)) - 1.0;
                    let d = direction(face, a, b);
                    let Some(h) = height_field_m(&body, fine.as_ref(), d, rung) else {
                        continue;
                    };
                    let Some(c) = height_field_m(&body, coarse.as_ref(), d, rung + 1) else {
                        continue;
                    };
                    diffs.push((h - c).abs());
                }
            }
        }
        diffs.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        let at = |q: f64| diffs[((diffs.len() as f64 * q) as usize).min(diffs.len() - 1)];
        let (median, p90, p99, max) = (at(0.5), at(0.9), at(0.99), at(1.0));
        let cell = f64::from(vd_seed::ladder::cell_m(rung + 1));
        // ★ THE DISTANCE THE SWAP REALLY HAPPENS AT (ruling W17): the tier rule's own switch
        // distance, never nearer than the floor the handover's own step sets. Before W17 the step
        // read the octave table alone and the floor never bit; the pixels below are read at this
        // distance, so the table says what the eye gets.
        let field = vd_terrain::artifact::field_step_m(&body, &lattice, levels, rung);
        let step = handover_step_m(&body, rung, field);
        let s = switch_m(rung).max(vd_client::ladder_view::switch_floor_m(step));
        let px = |m: f64| (m / s) / pixel_rad();
        let sink = sink_m(&body, rung + 1);
        let over = diffs.iter().filter(|d| **d > sink).count();
        println!(
            "{rung:>3} -> {:>3} | {}->{} | {cell:>11.0} | {s:>8.0} | {median:>8.1} | {p90:>5.1} | {p99:>5.1} | {max:>7.1} | {:>9.2} | {:>6.2} | {:>6.2} | {:>12.1} | {sink:>11.1} | {over:>6} of {}",
            rung + 1,
            fine.level(),
            coarse.level(),
            max / cell,
            px(p99),
            px(max),
            step,
            diffs.len()
        );
    }

    // ★ THE LEVEL'S OWN FOLD (ruling W17): a pyramid node is the MEAN OF FOUR of the level under
    // it, so the height it states differs from each child's by the fold's own SPREAD. That spread
    // is a fact about the SOLVE, folded once by a stated rule — never a drawn number — and it is
    // the half of a handover's step that the body's OCTAVE TABLE cannot know.
    println!("\nTHE LEVELS — how far a pyramid node stands from each of its four children");
    // ★ AND THE LAW THAT STANDS IN FOR IT (ruling W17): the body's own SLOPE REFERENCE — the first
    // fine octave's RMS slope, which every host already holds in the charter — over the distance
    // from a parent node's centre to a child node's, `node / (2·√2)`. The ladder can therefore know
    // a handover's field step with nothing new crossing a realm boundary (SL6).
    let slope_ref = body.slope_ref().raw() as f64 / f64::from(1u32 << vd_recipe::noise::NOISE_BITS);
    println!(
        "the body's slope reference is {slope_ref:.6} (the first fine octave's own RMS slope)"
    );
    println!(
        "level | node m | children | median m | p90 m | p99 m | max m | the slope law: slope_ref x node/(2 root 2) m"
    );
    {
        let mut below: Vec<i16> = artifact.rows.iter().map(|r| r.z_m).collect();
        let mut fine = lattice;
        for level in 1..=levels {
            let Some(coarse) = lattice.coarser(level) else {
                break;
            };
            let means = &artifact.pyramid[level as usize - 1];
            let mut spread: Vec<f64> = Vec::with_capacity(fine.node_count());
            for node in 0..fine.node_count() as u32 {
                let (face, i, j) = fine.split(node);
                let up = coarse.index(face, i >> 1, j >> 1) as usize;
                spread.push(f64::from(
                    (i32::from(means[up]) - i32::from(below[node as usize])).abs(),
                ));
            }
            spread.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let at = |q: f64| spread[((spread.len() as f64 * q) as usize).min(spread.len() - 1)];
            println!(
                "{level:>5} | {:>6.0} | {:>8} | {:>8.1} | {:>5.1} | {:>5.1} | {:>5.1} | {:>42.1}",
                coarse.node_m(),
                spread.len(),
                at(0.5),
                at(0.9),
                at(0.99),
                at(1.0),
                slope_ref * coarse.node_m() / (2.0 * std::f64::consts::SQRT_2)
            );
            below.clone_from(means);
            fine = coarse;
        }
    }

    println!("\nTHE WATER — the side and the level the same directions read at the two rungs");
    println!(
        "L -> L+1 | wet at L | wet at L+1 | columns that FLIP | as a share % | median level step m | max level step m"
    );
    for rung in rung_lo..=rung_hi {
        let (fine_f, coarse_f) = (field_of(rung), field_of(rung + 1));
        let (mut wet_l, mut wet_c, mut flips, mut n) = (0u64, 0u64, 0u64, 0u64);
        let mut level_steps: Vec<f64> = Vec::new();
        for face in faces {
            for i in 0..grid {
                for j in 0..grid {
                    let a = (2.0 * (f64::from(i) + 0.5) / f64::from(grid)) - 1.0;
                    let b = (2.0 * (f64::from(j) + 0.5) / f64::from(grid)) - 1.0;
                    let d = direction(face, a, b);
                    let Some(w0) = wet_at(&body, &lattice, fine_f.as_ref(), d, rung) else {
                        continue;
                    };
                    let Some(w1) = wet_at(&body, &lattice, coarse_f.as_ref(), d, rung + 1) else {
                        continue;
                    };
                    n += 1;
                    wet_l += u64::from(w0.0);
                    wet_c += u64::from(w1.0);
                    flips += u64::from(w0.0 != w1.0);
                    if w0.0 && w1.0 {
                        level_steps.push((w0.1 - w1.1).abs());
                    }
                }
            }
        }
        level_steps.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        let median = level_steps
            .get(level_steps.len() / 2)
            .copied()
            .unwrap_or(0.0);
        let max = level_steps.last().copied().unwrap_or(0.0);
        println!(
            "{rung:>3} -> {:>3} | {wet_l:>8} | {wet_c:>10} | {flips:>17} | {:>12.3} | {median:>19.1} | {max:>16.1}",
            rung + 1,
            flips as f64 * 100.0 / n.max(1) as f64
        );
    }

    println!(
        "\nTHE SHADE — the mean Lambert over the same ground, one coarse chunk and its four children"
    );
    println!(
        "L -> L+1 | chunks L | vertices L | mean Lambert L | sd L | chunks L+1 | vertices L+1 | mean Lambert L+1 | sd L+1 | the step of the means %"
    );
    for rung in rung_lo..=rung_hi {
        let (fine, coarse) = (field_of(rung), field_of(rung + 1));
        // The coarse chunk column under the belt, and the four finer columns under it.
        let dir = DVec3::from_array(BELT).normalize();
        let face = vd_seed::bend::face_of(dir.to_array());
        let (t, s) = vd_seed::bend::face_coords(face, dir.to_array());
        let cells = |r: u8| {
            (
                vd_seed::ladder::index_of(
                    vd_seed::bend::unbend(t),
                    body.ladder().cells_per_edge(r),
                ),
                vd_seed::ladder::index_of(
                    vd_seed::bend::unbend(s),
                    body.ladder().cells_per_edge(r),
                ),
            )
        };
        let (ci, cj) = cells(rung + 1);
        let (cx, cy) = (ci / CHUNK_EDGE as i32, cj / CHUNK_EDGE as i32);
        let lambert = |r: u8, field: &dyn ZField, columns: &[(i32, i32)]| -> (u64, u64, f64, f64) {
            let (mut chunks, mut n, mut sum, mut square) = (0u64, 0u64, 0.0f64, 0.0f64);
            for (x, y) in columns {
                let Some(span) = surface_column_field(&body, field, face, r, *x, *y) else {
                    continue;
                };
                for z in span.lo..=span.hi {
                    let key = ChunkKey {
                        face,
                        rung: r,
                        x: *x,
                        y: *y,
                        z,
                    };
                    let Some(bx) = sample_box(&body, Some(field), key) else {
                        continue;
                    };
                    let mesh = extract(&bx);
                    if mesh.vertices.is_empty() {
                        continue;
                    }
                    chunks += 1;
                    let half = (CHUNK_EDGE / 2) as i16 * vd_terrain::VERTEX_QUANTUM as i16;
                    let origin = vertex_position_m(&body, &bx, [half, half, half]);
                    let vertices: Vec<[f32; 3]> = mesh
                        .vertices
                        .iter()
                        .map(|v| {
                            let p = vertex_position_m(&body, &bx, *v);
                            [
                                (p[0] - origin[0]) as f32,
                                (p[1] - origin[1]) as f32,
                                (p[2] - origin[2]) as f32,
                            ]
                        })
                        .collect();
                    for nrm in smooth_normals(&vertices, &mesh.triangles) {
                        let nn =
                            DVec3::new(f64::from(nrm[0]), f64::from(nrm[1]), f64::from(nrm[2]));
                        if nn.length() < 0.5 {
                            continue;
                        }
                        n += 1;
                        let lit = nn.normalize().dot(sun).max(0.0);
                        sum += lit;
                        square += lit * lit;
                    }
                }
            }
            let mean = sum / n.max(1) as f64;
            (
                chunks,
                n,
                mean,
                (square / n.max(1) as f64 - mean * mean).max(0.0).sqrt(),
            )
        };
        let children: Vec<(i32, i32)> = (0..2)
            .flat_map(|dx| (0..2).map(move |dy| (cx * 2 + dx, cy * 2 + dy)))
            .collect();
        let (fc, fn_, fl, fsd) = lambert(rung, fine.as_ref(), &children);
        let (cc, cn, cl, csd) = lambert(rung + 1, coarse.as_ref(), &[(cx, cy)]);
        let step = (fl - cl).abs() / ((fl + cl) / 2.0).max(1.0e-9) * 100.0;
        println!(
            "{rung:>3} -> {:>3} | {fc:>8} | {fn_:>10} | {fl:>14.4} | {fsd:>8.4} | {cc:>10} | {cn:>12} | {cl:>16.4} | {csd:>10.4} | {step:>16.2}",
            rung + 1
        );
    }
}
