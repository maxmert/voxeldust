//! ★ THE SPAN AGAINST THE ARTIFACT (the owner's coast flight, 2026-09-20: "the water shores' shapes
//! are constantly changing while I'm flying"): the client picks a column's chunk slices from the
//! RECIPE'S OWN relief (`digest::surface_column`), but the chunk it then builds stands on the
//! ARTIFACT's `Z` plus the fine octaves. Where the two differ by more than the column bound, the
//! wanted slice holds no ground, the chunk builds empty, and the flat sea sheet shows through the
//! hole — land drawn as sea, square by square, differently at every rung. This instrument measures
//! that miss over a square of the home planet around a stand: per rung, how many columns' real
//! surface falls outside the span the client asks for, and how far the two heights stand apart.
//! `cargo run --release -p vd-bins --example span_miss -- [half_width_km] [dx dy dz] [rung_lo rung_hi]`

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_core::glam::DVec3;
use vd_terrain::artifact::{PyramidField, ZField};
use vd_terrain::chunk::CHUNK_EDGE;
use vd_terrain::digest::{
    COLUMN_SAMPLES_PER_EDGE, SAMPLE_GAPS_LOG2, column_bound_m, surface_column, surface_column_field,
};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::{ChunkKey, height};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let half_km: f64 = args
        .first()
        .map_or(150.0, |s| s.parse().expect("the half width in km"));
    let aim = if args.len() >= 4 {
        DVec3::new(
            args[1].parse().expect("dx"),
            args[2].parse().expect("dy"),
            args[3].parse().expect("dz"),
        )
        .normalize()
    } else {
        // The land stand of the coast flights.
        DVec3::new(-0.169_974, -0.983_964, -0.054_071).normalize()
    };
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("the home planet has a macro lattice");
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let levels = artifact.pyramid.len() as u32;
    let sea_m = artifact.sea().unwrap_or(0);
    let floor = i64::from(body.ladder().floor_m);
    let edge = CHUNK_EDGE as i32;
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let face = vd_seed::bend::face_of([aim.x, aim.y, aim.z]);
    let (t, s) = vd_seed::bend::face_coords(face, [aim.x, aim.y, aim.z]);
    let n0 = body.ladder().cells_per_edge(0) as i64;
    let i0 = i64::from(vd_seed::ladder::index_of(
        vd_seed::bend::unbend(t),
        n0 as u32,
    ));
    let j0 = i64::from(vd_seed::ladder::index_of(
        vd_seed::bend::unbend(s),
        n0 as u32,
    ));
    let half_cells = (half_km * 1000.0 / f64::from(vd_seed::ladder::cell_m(0))) as i64;
    println!(
        "span_miss: face {face:?}, centre cell ({i0}, {j0}) of {n0}, ±{half_km} km, sea {sea_m} m over the ladder radius, {levels} pyramid levels"
    );
    println!(
        "rung | columns | recipe misses | miss % | land-as-sea | FIELD misses | field miss % | bound m | worst |real-recipe| m | mean m"
    );
    let step = (edge - 1) >> SAMPLE_GAPS_LOG2;
    let rung_lo: u8 = args
        .get(4)
        .map_or(4, |s| s.parse().expect("the first rung"));
    let rung_hi: u8 = args
        .get(5)
        .map_or(11, |s| s.parse().expect("the last rung"));
    for rung in rung_lo..=rung_hi {
        let level = PyramidField::level_for(&lattice, levels, rung);
        let pyramid = PyramidField::of(&artifact, level);
        let field: &dyn ZField = match &pyramid {
            Some(p) => p,
            None => &artifact,
        };
        let cells_per_column = i64::from(edge) << rung;
        let last = (i64::from(body.ladder().cells_per_edge(rung)) - 1) / i64::from(edge);
        let (x_lo, x_hi) = (
            ((i0 - half_cells) / cells_per_column).max(0),
            ((i0 + half_cells) / cells_per_column).min(last),
        );
        let (y_lo, y_hi) = (
            ((j0 - half_cells) / cells_per_column).max(0),
            ((j0 + half_cells) / cells_per_column).min(last),
        );
        let bound = column_bound_m(&body, rung);
        let mut columns = 0u64;
        let mut misses = 0u64;
        let mut field_misses = 0u64;
        let mut land_as_sea = 0u64;
        let mut worst = 0.0f64;
        let mut sum = 0.0f64;
        let mut samples = 0u64;
        for y in y_lo..=y_hi {
            for x in x_lo..=x_hi {
                let (x, y) = (x as i32, y as i32);
                let span = surface_column(&body, face, rung, x, y);
                let field_span = surface_column_field(&body, field, face, rung, x, y);
                let mut field_miss = false;
                let key = ChunkKey {
                    face,
                    rung,
                    x,
                    y,
                    z: 0,
                };
                let mut column_miss = false;
                let mut any_real_land = false;
                let mut any_slice_holds = false;
                let mut i = 0;
                while i < COLUMN_SAMPLES_PER_EDGE {
                    let mut j = 0;
                    while j < COLUMN_SAMPLES_PER_EDGE {
                        let site = vd_terrain::lattice::site_of(&body, key, i * step, j * step);
                        let dir = vd_terrain::lattice::site_dir(&body, key, site);
                        let d = [
                            dir[0].raw() as f64 / unit,
                            dir[1].raw() as f64 / unit,
                            dir[2].raw() as f64 / unit,
                        ];
                        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        let d = [d[0] / len, d[1] / len, d[2] / len];
                        let recipe_m = height::height_m(&body, d, rung);
                        if let Some(real_m) = height::height_field_m(&body, field, d, rung) {
                            let gap = (real_m - recipe_m).abs();
                            worst = worst.max(gap);
                            sum += gap;
                            samples += 1;
                            let z = ((real_m.floor() as i64 - floor) >> rung) / i64::from(edge);
                            let z = z as i32;
                            let holds = (span.lo..=span.hi).contains(&z);
                            field_miss |= !field_span.is_some_and(|f| (f.lo..=f.hi).contains(&z));
                            any_slice_holds |= holds;
                            column_miss |= !holds;
                            let over_sea = real_m - body.radius_m() > f64::from(sea_m);
                            any_real_land |= over_sea && !holds;
                        }
                        j += 1;
                    }
                    i += 1;
                }
                columns += 1;
                misses += u64::from(column_miss);
                field_misses += u64::from(field_miss);
                land_as_sea += u64::from(any_real_land && !any_slice_holds);
            }
        }
        println!(
            "{rung:>4} | {columns:>7} | {misses:>13} | {:>6.2} | {land_as_sea:>11} | {field_misses:>12} | {:>12.2} | {bound:>7.1} | {worst:>18.1} | {:>6.1}",
            100.0 * misses as f64 / columns.max(1) as f64,
            100.0 * field_misses as f64 / columns.max(1) as f64,
            sum / samples.max(1) as f64
        );
    }
}
