//! ★ IS THE WATER DRAWN AT ALL, AT EVERY RUNG? (2026-09-23; the owner, flying out: *"at some point
//! water is not visible at all, just land"*; ruling W16 fault C.)
//!
//! **THE QUESTION, AS A NUMBER.** The sea covers 78.6 % of the home planet's surface. A coarse rung
//! is a LOOK of the same field, so the share of the globe's columns that stand UNDER their own
//! water must read about that at EVERY rung. And the sheet drawn over those columns must stand over
//! the ground, not under it.
//!
//! **THE TWO READINGS, per rung, over EVERY chunk of the globe** (the shipped path:
//! `lattice::sample_box`, the very box the chunk builder fills):
//!
//! * `wet` — how many columns stand under their own water, as a share of all columns. This is the
//!   MODEL's own answer, and it must not fall with the rung.
//! * `shallower than a cell's chord` — how many of those wet columns stand nearer to the water than
//!   one CELL's chord dips (`cell² / 4R`): the columns whose ground stood OVER a sheet quad drawn
//!   on the corner columns, the rule ruling W18 retired (the sheet now rides the ground's own
//!   triangles, `position::water_sheet`, and `position::water_sheet_tests` holds the statement
//!   that could fail). The column stays as the reading of how much ground the old rule pushed
//!   through the water at each rung.
//! * `water triangles` — what the shipped builder really makes for the whole globe at that rung,
//!   so the law's own cost is a number too: one triangle per ground triangle that reaches down to
//!   the water plus the client's hide bound.
//!
//! `cargo run --release -p vd-bins --example far_water_share -- [rung_lo] [rung_hi]`

use std::time::Instant;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_seed::bend::Face;
use vd_terrain::artifact::PyramidField;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};
use vd_terrain::extract::extract_all_edges;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::position::water_sheet;
use vd_terrain::units::metres_of_q28;

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    let body = home_planet();
    let started = Instant::now();
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
    let rung_lo = a.first().map_or(9u8, |v| *v as u8);
    let rung_hi = a.get(1).map_or(top, |v| (*v as u8).min(top));
    let radius = body.radius_m();
    println!(
        "far_water_share: solved in {:.1} s; the sea stands at {} m; the top rung is {top}",
        started.elapsed().as_secs_f64(),
        artifact.sea().unwrap_or(0)
    );
    println!(
        "rung |   cell m | columns | wet | wet share % | shallower than a cell's chord (the retired quad) | water triangles | cell dip m"
    );
    for rung in rung_lo..=rung_hi {
        let level = PyramidField::level_for(&lattice, levels, rung);
        let field: Box<dyn vd_terrain::artifact::ZField> = if level == 0 {
            Box::new(artifact.clone())
        } else {
            Box::new(PyramidField::of(&artifact, level, &counts).expect("a level"))
        };
        let cell_m = f64::from(vd_seed::ladder::cell_m(rung));
        // A flat quad of `w` metres is a chord of the sphere: its middle stands `w² / 4R` under it.
        let dip = |w: f64| w * w / (4.0 * radius);
        let cell_dip = dip(cell_m);
        let n = body.ladder().cells_per_edge(rung) as i32;
        let chunks = (n - 1) / CHUNK_EDGE as i32 + 1;
        let (mut columns, mut wet, mut before, mut tris) = (0u64, 0u64, 0u64, 0u64);
        for face in 0..6u8 {
            let face = Face::from_index(face).expect("a face");
            for y in 0..chunks {
                for x in 0..chunks {
                    let key = ChunkKey {
                        face,
                        rung,
                        x,
                        y,
                        z: 0,
                    };
                    let Some(bx) =
                        vd_terrain::lattice::sample_box(&body, Some(field.as_ref()), key)
                    else {
                        continue;
                    };
                    for b in 0..CHUNK_EDGE as i32 {
                        for a in 0..CHUNK_EDGE as i32 {
                            let col = vd_terrain::lattice::SampleBox::column_index(a, b);
                            let water = bx.water[col].max(bx.sea);
                            let ground = bx.surfaces[col];
                            columns += 1;
                            if water <= ground {
                                continue;
                            }
                            wet += 1;
                            let depth = metres_of_q28(water) - metres_of_q28(ground);
                            before += u64::from(depth < cell_dip);
                        }
                    }
                    // The sheet the shipped builder really makes for this box, with the hide bound
                    // the client passes.
                    let hide = body.step_bound_m(rung)
                        + vd_client::chunks::sink_m(&body, rung)
                        + f64::from(vd_seed::ladder::cell_m(rung));
                    let hide = vd_terrain::units::q28_of_metres(hide);
                    let mesh = extract_all_edges(&bx);
                    tris += water_sheet(&body, &bx, &mesh, hide).2.len() as u64;
                }
            }
        }
        println!(
            "{rung:>4} | {cell_m:>8.0} | {columns:>7} | {wet:>7} | {:>11.1} | {before:>48} | {tris:>15} | {cell_dip:>10.3}",
            wet as f64 * 100.0 / columns.max(1) as f64
        );
    }
}
