//! ★ WHAT A COLUMN'S CHUNKS HOLD (the owner's coast flight, 2026-09-20): for one chunk column of
//! the home planet, the slices the field span asks, and for each of them the chunk the client
//! builds on the artifact — its triangle count and how many of the column's 62 × 62 core cells its
//! mesh covers — so a hole in the picture is traced to the slice that holds no ground under it.
//! `cargo run --release -p vd-bins --example chunk_probe -- face rung x y [face rung x y ...]`
//! (`face` is the bend's index: 0 +X, 1 −X, 2 +Y, 3 −Y, 4 +Z, 5 −Z).

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_seed::bend::Face;
use vd_terrain::artifact::{PyramidField, ZField};
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey, column_field};
use vd_terrain::digest::{surface_column, surface_column_field};
use vd_terrain::extract::extract_all_edges;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::lattice::sample_box;
use vd_terrain::macro_lattice::MacroLattice;

fn main() {
    let a: Vec<i64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("an integer"))
        .collect();
    assert!(
        a.len() >= 4 && a.len().is_multiple_of(4),
        "face rung x y [...]"
    );
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("a macro lattice");
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let levels = artifact.pyramid.len() as u32;
    let floor = i64::from(body.ladder().floor_m);
    let edge = i64::from(CHUNK_EDGE as i32);
    let counts = artifact.coast_counts();
    let q = vd_terrain::VERTEX_QUANTUM;
    for c in a.chunks(4) {
        let face = Face::from_index(c[0] as u8).expect("a face");
        let (rung, x, y) = (c[1] as u8, c[2] as i32, c[3] as i32);
        let level = PyramidField::level_for(&lattice, levels, rung);
        let pyramid = PyramidField::of(&artifact, level, &counts);
        let field: &dyn ZField = match &pyramid {
            Some(l) => l,
            None => &artifact,
        };
        let slice = |h_m: f64| ((h_m.floor() as i64 - floor) >> rung) / edge;
        let columns = column_field(&body, Some(field), face, rung, x, y);
        let (low, high) = columns.as_ref().map_or((f64::NAN, f64::NAN), |c| {
            (
                vd_terrain::units::metres_of_q28(c.lowest),
                vd_terrain::units::metres_of_q28(c.highest),
            )
        });
        let fs = surface_column_field(&body, field, face, rung, x, y);
        let rs = surface_column(&body, face, rung, x, y);
        println!(
            "column {face:?} rung {rung} ({x}, {y}): level {level}; the core's ground {:.1}..{:.1} m over the ladder radius = slices {}..{}; FIELD span {}; RECIPE span {}..={}",
            low - body.radius_m(),
            high - body.radius_m(),
            slice(low),
            slice(high),
            fs.map_or("none".to_owned(), |s| format!(
                "{}..={} (peak {:.0} m)",
                s.lo,
                s.hi,
                s.peak_m - body.radius_m()
            )),
            rs.lo,
            rs.hi
        );
        let Some(fs) = fs else { continue };
        for z in (fs.lo - 1)..=(fs.hi + 1) {
            let key = ChunkKey {
                face,
                rung,
                x,
                y,
                z,
            };
            let Some(samples) = sample_box(&body, Some(field), key) else {
                println!("  slice {z}: no box (outside the band, or the field not whole)");
                continue;
            };
            let mesh = extract_all_edges(&samples);
            let mut covered = vec![false; CHUNK_EDGE * CHUNK_EDGE];
            for v in &mesh.vertices {
                let (ca, cb) = (i32::from(v[0]).div_euclid(q), i32::from(v[1]).div_euclid(q));
                if (0..CHUNK_EDGE as i32).contains(&ca) && (0..CHUNK_EDGE as i32).contains(&cb) {
                    covered[(cb as usize) * CHUNK_EDGE + ca as usize] = true;
                }
            }
            let n = covered.iter().filter(|c| **c).count();
            println!(
                "  slice {z}{}: {} vertices, {} triangles, the mesh touches {n} of {} core cells ({:.1} %)",
                if (fs.lo..=fs.hi).contains(&z) {
                    " (asked)"
                } else {
                    " (NOT asked)"
                },
                mesh.vertices.len(),
                mesh.triangles.len(),
                CHUNK_EDGE * CHUNK_EDGE,
                100.0 * n as f64 / (CHUNK_EDGE * CHUNK_EDGE) as f64
            );
        }
    }
}
