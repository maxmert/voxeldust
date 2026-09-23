//! ★ WHAT THE COAST MASK SAYS UNDER A PIXEL (2026-09-23, ruling W18's leftover patch): for a
//! pilot capture (the entity's position and facing, the 1284 × 720 frame) and a pixel, run the
//! pixel's ray to the sea's sphere, find the fine node under it, and print the mask's side in a
//! window around it at level 0 (the fine nodes) and at the count pyramid's levels 1 to 3 — so a
//! patch of dots in a picture can be read as the mask's own islets or as a drawing fault.
//! `cargo run --release -p vd-bins --example mask_at_pixel -- ex ey ez qx qy qz qw px py`

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client_harness::camera::pilot_capture_camera;
use vd_core::glam::{DQuat, DVec3};
use vd_terrain::artifact::{PyramidField, Side, ZField, coast_side};
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};
use vd_terrain::digest::surface_chunk_z;
use vd_terrain::extract::extract_all_edges;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::lattice::{SampleBox, sample_box};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::position::{WEIGHT_BITS, vertex_position_m};
use vd_terrain::units::metres_of_q28;

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    assert!(a.len() == 9, "ex ey ez qx qy qz qw px py");
    let own = DVec3::new(a[0], a[1], a[2]);
    let orient = DQuat::from_xyzw(a[3], a[4], a[5], a[6]);
    let (px, py) = (a[7], a[8]);
    let body = home_planet();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let counts = artifact.coast_counts();
    let cam = pilot_capture_camera(own, orient, 1284, 720);
    let dir = (cam.unproject(px, py, 1.0) - cam.eye).normalize();
    // The ray against the sea's sphere: the nearer root.
    let r = body.sea_radius_m();
    let b = cam.eye.dot(dir);
    let c = cam.eye.length_squared() - r * r;
    let disc = b * b - c;
    assert!(disc > 0.0, "the pixel's ray misses the sea");
    let t = -b - disc.sqrt();
    let hit = (cam.eye + dir * t).normalize();
    let d = [hit.x, hit.y, hit.z];
    let face = vd_seed::bend::face_of(d);
    let (u, v) = vd_seed::bend::face_coords(face, d);
    let n = lattice.edge;
    let i = vd_seed::ladder::index_of(vd_seed::bend::unbend(u), n);
    let j = vd_seed::ladder::index_of(vd_seed::bend::unbend(v), n);
    println!(
        "pixel ({px}, {py}) hits the sea at {:.0} km along the ray; face {face:?} node ({i}, {j}); a node is {:.0} m",
        t / 1000.0,
        lattice.node_m()
    );
    let glyph = |s: Option<Side>| match s {
        Some(Side::Land) => 'L',
        Some(Side::Sea) => '.',
        Some(Side::Lake) => 'k',
        None => '?',
    };
    for level in 0..=3u32 {
        let half = 24i32;
        let step = 1i32 << level;
        println!(
            "--- level {level} (a glyph is a {} m block), the window is ±{} nodes",
            lattice.node_m() * f64::from(step),
            half * step
        );
        let mut jj = -half;
        while jj <= half {
            let mut line = String::new();
            let mut ii = -half;
            while ii <= half {
                let (ni, nj) = (i + ii * step, j + jj * step);
                let inside = ni >= 0 && nj >= 0 && ni < n as i32 && nj < n as i32;
                let side = if !inside {
                    None
                } else if level == 0 {
                    coast_side(&artifact.coast, lattice.index(face, ni, nj))
                } else {
                    counts.side(level, face, (ni >> level) as u32, (nj >> level) as u32)
                };
                let g = glyph(side);
                line.push(if ii == 0 && jj == 0 { '@' } else { g });
                ii += 1;
            }
            println!("{line}");
            jj += 1;
        }
    }
    // ★ THE DRAWN GROUND AGAINST THE WATER under the same window, per rung: the chunk that holds
    // the hit node, built on the field the client reads at that rung; every ground vertex within
    // the window binned by its height over the water (the vertex's level as the sheet takes it)
    // and named by the mask's word at its nearest node.
    let levels = artifact.pyramid.len() as u32;
    for rung in [13u8, 14, 15] {
        let level = PyramidField::level_for(&lattice, levels, rung);
        let field: Box<dyn ZField> = if level == 0 {
            Box::new(artifact.clone())
        } else {
            Box::new(PyramidField::of(&artifact, level, &counts).expect("a level"))
        };
        let cells_per_node = i64::from(lattice.cells_per_node);
        let cell_of = |node: i32| -> i64 { (i64::from(node) * cells_per_node) >> rung };
        let edge = CHUNK_EDGE as i64;
        let (x, y) = ((cell_of(i) / edge) as i32, (cell_of(j) / edge) as i32);
        let key = ChunkKey {
            face,
            rung,
            x,
            y,
            z: surface_chunk_z(&body, face, rung, x, y),
        };
        let Some(bx) = sample_box(&body, Some(field.as_ref()), key) else {
            println!("rung {rung}: no box for chunk ({x}, {y})");
            continue;
        };
        let mesh = extract_all_edges(&bx);
        let window_m = 24.0 * lattice.node_m();
        // The COLUMNS' own word first — the shore law's output, before the extractor: how many
        // columns in the window stand over their water, and how many of those the mask calls sea.
        let (mut cols, mut cols_over, mut cols_over_sea) = (0u64, 0u64, 0u64);
        for b in 0..CHUNK_EDGE as i32 {
            for a in 0..CHUNK_EDGE as i32 {
                let col = SampleBox::column_index(a, b);
                let dir = bx.dirs[col];
                let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
                let dv = DVec3::new(
                    dir[0].raw() as f64 / unit,
                    dir[1].raw() as f64 / unit,
                    dir[2].raw() as f64 / unit,
                );
                let along = hit.dot(dv).clamp(-1.0, 1.0).acos() * body.radius_m();
                if along > window_m {
                    continue;
                }
                cols += 1;
                let water = bx.water[col].max(bx.sea);
                if bx.surfaces[col] > water {
                    cols_over += 1;
                    let (fu, fv) = vd_seed::bend::face_coords(face, [dv.x, dv.y, dv.z]);
                    let ni = vd_seed::ladder::index_of(vd_seed::bend::unbend(fu), n);
                    let nj = vd_seed::ladder::index_of(vd_seed::bend::unbend(fv), n);
                    if coast_side(&artifact.coast, lattice.index(face, ni, nj)) != Some(Side::Land)
                    {
                        cols_over_sea += 1;
                    }
                }
            }
        }
        println!(
            "rung {rung}: {cols} columns within the window, {cols_over} over their water, {cols_over_sea} of those where the mask says sea or lake"
        );
        let mut bins = [0u64; 5];
        let mut land_bins = [0u64; 5];
        let mut seen = 0u64;
        for v in &mesh.vertices {
            let p = vertex_position_m(&body, &bx, *v);
            let pv = DVec3::new(p[0], p[1], p[2]);
            let ground_r = pv.length();
            let along = hit.dot(pv / ground_r).clamp(-1.0, 1.0).acos() * body.radius_m();
            if along > window_m {
                continue;
            }
            seen += 1;
            let mut group = [0i32; 2];
            for axis in 0..2 {
                group[axis] = (i32::from(v[axis]) >> WEIGHT_BITS).min(CHUNK_EDGE as i32 - 1);
            }
            let mut level_r = 0.0f64;
            for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let col = SampleBox::column_index(group[0] + da, group[1] + db);
                let w = bx.water[col].max(bx.sea);
                level_r = level_r.max(metres_of_q28(w));
            }
            let h = ground_r - level_r;
            let bin = if h < -1.0 {
                0
            } else if h < -0.1 {
                1
            } else if h <= 0.1 {
                2
            } else if h <= 1.0 {
                3
            } else {
                4
            };
            bins[bin] += 1;
            let d = pv / ground_r;
            let (fu, fv) = vd_seed::bend::face_coords(face, [d.x, d.y, d.z]);
            let ni = vd_seed::ladder::index_of(vd_seed::bend::unbend(fu), n);
            let nj = vd_seed::ladder::index_of(vd_seed::bend::unbend(fv), n);
            if coast_side(&artifact.coast, lattice.index(face, ni, nj)) == Some(Side::Land) {
                land_bins[bin] += 1;
            }
        }
        println!(
            "rung {rung} (level {level}, chunk ({x}, {y}) z {}): {seen} vertices within {:.0} km of the hit; height over the water: < -1 m {} | -1..-0.1 {} | ±0.1 m {} | 0.1..1 m {} | > 1 m {}   (of which the mask says LAND: {} | {} | {} | {} | {})",
            key.z,
            window_m / 1000.0,
            bins[0],
            bins[1],
            bins[2],
            bins[3],
            bins[4],
            land_bins[0],
            land_bins[1],
            land_bins[2],
            land_bins[3],
            land_bins[4]
        );
    }
}
