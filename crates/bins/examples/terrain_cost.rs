//! THE GENERATOR's measurements (the voxel foundation, slice 5; ruling V8 addendum: the coarse-answer
//! measurement runs FIRST, before any terrain is drawn), on THE world's home planet, with the real
//! hash — never a proxy bench:
//!
//! 1. the cost of one surface chunk at EVERY rung, and the measurement: the top rung's column pass
//!    costs less than rung 0's (the EXACT half — fewer octaves at every rung — is a unit test in
//!    the generator crate, `every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it`);
//! 2. how many chunks of one NAMED radial column the skips refuse without a cell pass;
//! 3. the boot self-check's cost (eight chunks) and its digest;
//! 4. the home planet's seed and ladder, the same body `vd_bins::home_body` gives the gateway;
//! 5. (slice 6, ruling V10) THE EXTRACTOR: the sample box (the chunk with its halo) and the surface
//!    nets at every rung on the surface chunk of the same column — milliseconds, vertices,
//!    triangles, bytes — the halo's cost against the bare column pass, the WORST case (a 3-D
//!    checkerboard, a synthetic BOUND, never a world), and the SNAP error: the radial distance
//!    between a vertex and the true seed surface, over the surface chunks' vertices.
//!
//! Run in release:
//!
//! ```text
//! cargo run --release -p vd-bins --example terrain_cost
//! ```

use std::process::ExitCode;
use std::time::Instant;

use vd_bins::DEV;
use vd_seed::bend::Face;
use vd_terrain::Gf;
use vd_terrain::chunk::{CHUNK_EDGE, Cell, How, column_field, generate, generate_in};
use vd_terrain::digest::{golden_self_check, self_check_key, surface_chunk_z};
use vd_terrain::extract::extract;
use vd_terrain::height::height_m;
use vd_terrain::lattice::{BOX_CELLS, BOX_EDGE, HALO, SampleBox, Site, sample_box};
use vd_terrain::position::vertex_position_m;
use vd_terrain::strata::Stratum;
use vd_terrain::{ChunkKey, GOLDEN_SELF_CHECK_KEYS};

/// The extractor's budget per chunk of WORKER time — the sample box and the extraction together —
/// set by the owner at 8 ms on 2026-09-08 (ruling V10, after the measurement: a surface chunk 3.3 ms,
/// a cave-dense chunk 6.1 ms). "If that becomes a problem over time, we rethink."
const EXTRACT_BUDGET_US: f64 = 8_000.0;
/// The synthetic worst case (a 3-D checkerboard) as MEASURED at slice 6 on this machine, release:
/// 26 ms, six times the 4 ms budget. It is a BOUND no chunk of THE world reaches; the owner
/// decides the budget question (slice 6 document §13).
const WORST_CASE_MEASURED_US: f64 = 26_000.0;

fn main() -> ExitCode {
    // ONE rule names the home body: the same one the gateway's world identity uses.
    let Some(body) = vd_bins::home_body(DEV.universe_seed) else {
        println!("terrain_cost: REFUSED — the home system holds no planet the recipe accepts");
        return ExitCode::FAILURE;
    };
    println!(
        "terrain_cost: home planet seed {} -> ladder radius {:.0} m, {} rungs, {} octaves, relief {} m",
        body.seed(),
        body.ladder().radius_m(),
        body.ladder().rungs,
        body.octave_count(),
        body.relief_bound_m(0).to_i64_floor()
    );

    // 1. At every rung: the COLUMN pass (the coarse answer — one height per column with the rung's
    //    octaves) and the CELL pass of the surface chunk, timed apart. The cell pass is the same
    //    count of cells at every rung.
    let rounds = 5u32;
    let mut column_us_at = Vec::new();
    let mut rung = 0u8;
    while rung < body.ladder().rungs {
        let (x, y) = (3, 5);
        let z = surface_chunk_z(&body, Face::PosX, rung, x, y);
        let start = Instant::now();
        let mut column = None;
        for _ in 0..rounds {
            column = column_field(&body, Face::PosX, rung, x, y);
        }
        let column_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
        let column = column.expect("in the ladder");
        let start = Instant::now();
        let mut how = How::AboveSurface;
        for _ in 0..rounds {
            how = generate_in(&body, &column, z).expect("in the ladder").how;
        }
        let cells_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
        println!(
            "terrain_cost: rung {rung:>2} ({:>5} m cells) column pass {column_us:>8.1} us ({:>2} octaves), surface chunk cell pass {cells_us:>8.1} us ({how:?})",
            1u32 << rung,
            body.octaves_at(rung).len()
        );
        column_us_at.push(column_us);
        rung += 1;
    }

    // 2. The skips along one radial column at rung 0. The column is NAMED so the numbers can be
    //    re-derived on another machine.
    let chunks_in_band = body.ladder().cells_in_band(0).div_ceil(62) as i32;
    println!("terrain_cost: the radial column measured is face PosY, rung 0, x 11, y 13");
    let mut skipped = 0;
    let mut evaluated = 0;
    let start = Instant::now();
    for z in 0..chunks_in_band {
        let key = ChunkKey {
            face: Face::PosY,
            rung: 0,
            x: 11,
            y: 13,
            z,
        };
        match generate(&body, key).expect("in the ladder").how {
            How::Evaluated => evaluated += 1,
            How::AboveSurface | How::BelowSurface => skipped += 1,
        }
    }
    let column_ms = start.elapsed().as_secs_f64() * 1e3;
    println!(
        "terrain_cost: one radial column at rung 0: {chunks_in_band} chunks, {skipped} skipped, {evaluated} evaluated, {column_ms:.1} ms"
    );

    // 5. THE EXTRACTOR (slice 6). At every rung: the sample box (column pass + cell pass + halo)
    //    and the extraction, timed apart, with the output's size; then the snap error over the
    //    surface chunk's vertices.
    let mut worst_us: f64 = 0.0;
    let mut worst_key = None;
    let mut rung = 0u8;
    let mut snap_errors: Vec<u64> = Vec::new();
    let mut cave_vertices = 0usize;
    // The set measured: at every rung the surface chunk of column (3, 5) of +X; and at rung 0
    // also a seam chunk (the last chunk of +X, partial on the home planet), the corner chunk, and
    // the cave-dense chunk the first measurement found at rung 2.
    let n0 = body.ladder().cells_per_edge(0) as i32;
    let last0 = (n0 - 1) / 62;
    let mut named: Vec<(Face, u8, i32, i32)> = Vec::new();
    while rung < body.ladder().rungs {
        named.push((Face::PosX, rung, 3, 5));
        rung += 1;
    }
    named.push((Face::PosX, 0, last0, 7));
    named.push((Face::PosX, 0, last0, last0));
    let n2 = body.ladder().cells_per_edge(2) as i32;
    named.push((Face::PosX, 2, (n2 - 1) / 62, 7));
    for (face, rung, x, y) in named {
        let z = surface_chunk_z(&body, face, rung, x, y);
        let key = ChunkKey {
            face,
            rung,
            x,
            y,
            z,
        };
        let start = Instant::now();
        let mut samples = None;
        for _ in 0..rounds {
            samples = sample_box(&body, key);
        }
        let box_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
        let samples = samples.expect("in the band");
        let start = Instant::now();
        let mut mesh = None;
        for _ in 0..rounds {
            mesh = Some(extract(&samples));
        }
        let extract_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
        let mesh = mesh.expect("extracted");
        let bytes = mesh.vertices.len() * 6 + mesh.triangles.len() * 12;
        println!(
            "terrain_cost: {face:?} rung {rung:>2} chunk ({x}, {y}): sample box {box_us:>8.1} us, extract {extract_us:>8.1} us, {:>6} vertices, {:>6} triangles, {bytes:>7} bytes",
            mesh.vertices.len(),
            mesh.triangles.len()
        );
        if box_us + extract_us > worst_us {
            worst_us = box_us + extract_us;
            worst_key = Some(key);
        }
        // The snap error at this rung: |r(vertex) − h(dir(vertex))| in cells, over the mesh.
        let cell = f64::from(1u32 << rung);
        for v in &mesh.vertices {
            let p = vertex_position_m(&body, &samples, *v);
            let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            let r = r2.sqrt();
            let dir = [p[0] / r, p[1] / r, p[2] / r];
            let h = height_m(&body, dir, rung);
            // A vertex more than half a cell UNDER the height surface is a cave wall's (the height
            // surface has no vertex there); every other vertex is the surface's own, creases and
            // cliffs included, and its distance to the height surface is the snap error.
            let signed = (r - h) / Gf::from_f64(cell);
            if signed < -Gf::HALF {
                cave_vertices += 1;
            } else {
                snap_errors.push(signed.abs().to_bits());
            }
        }
    }
    // The snap errors as a distribution, in cells of the rung.
    let mut errs: Vec<f64> = snap_errors.iter().map(|b| f64::from_bits(*b)).collect();
    errs.sort_by(f64::total_cmp);
    let at = |q: f64| errs[((errs.len() - 1) as f64 * q) as usize];
    println!(
        "terrain_cost: snap error over {} surface vertices ({cave_vertices} cave vertices set aside), in cells: p50 {:.4}, p90 {:.4}, p99 {:.4}, max {:.4}",
        errs.len(),
        at(0.5),
        at(0.9),
        at(0.99),
        errs[errs.len() - 1]
    );
    // The halo's cost: the bare column pass against the sample box's column work, rung 0.
    let z0 = surface_chunk_z(&body, Face::PosX, 0, 3, 5);
    let k0 = ChunkKey {
        face: Face::PosX,
        rung: 0,
        x: 3,
        y: 5,
        z: z0,
    };
    let start = Instant::now();
    for _ in 0..rounds {
        let _ = generate(&body, k0);
    }
    let bare_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    let start = Instant::now();
    for _ in 0..rounds {
        let _ = sample_box(&body, k0);
    }
    let halo_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    println!(
        "terrain_cost: rung 0 chunk alone {bare_us:.1} us, with its halo {halo_us:.1} us ({:.1} % more)",
        (halo_us / bare_us - 1.0) * 100.0
    );
    // The WORST case: a 3-D checkerboard, every group crossed. A synthetic bound, never a world.
    let mut cells = Vec::with_capacity(BOX_CELLS);
    let mut n = 0usize;
    while n < BOX_CELLS {
        let (a, b, c) = (
            n % BOX_EDGE,
            (n / BOX_EDGE) % BOX_EDGE,
            n / (BOX_EDGE * BOX_EDGE),
        );
        let rock = (a + b + c) % 2 == 0;
        cells.push(Cell {
            stratum: if rock { Stratum::Granite } else { Stratum::Air },
            gap: if rock { -64 } else { 64 },
        });
        n += 1;
    }
    let checker = SampleBox {
        key: k0,
        cells,
        sites: vec![
            Site {
                face: 0,
                i: 0,
                j: 0
            };
            BOX_EDGE * BOX_EDGE
        ],
        dirs: vec![[Gf::ONE, Gf::ZERO, Gf::ZERO]; BOX_EDGE * BOX_EDGE],
    };
    let start = Instant::now();
    let mut worst = None;
    for _ in 0..rounds {
        worst = Some(extract(&checker));
    }
    let checker_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    let worst = worst.expect("extracted");
    println!(
        "terrain_cost: WORST CASE (3-D checkerboard, synthetic) extract {checker_us:.1} us, {} vertices, {} triangles, {} bytes",
        worst.vertices.len(),
        worst.triangles.len(),
        worst.vertices.len() * 6 + worst.triangles.len() * 12
    );
    let _ = (HALO, CHUNK_EDGE);

    // 6. THE CLIENT'S SIDE (slice 7, M7-2 and M7-3): the geometry the engine draws — the extractor's
    //    output around the chunk's origin, with smooth normals — over the columns a ground picture
    //    stands on at rung 0 (9 × 9 columns, the surface chunk and its two neighbours each): chunks
    //    per second on ONE thread and on every core (the threaded workers' shape: one job per
    //    chunk, no shared state), and the bytes per chunk handed to the engine.
    {
        use vd_client::chunks::{chunks_around, geometry_of};
        let r = body.ladder().radius_m();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let keys = chunks_around(&body, [d[0] * r, d[1] * r, d[2] * r], 0, 4);
        let start = Instant::now();
        let mut built = 0usize;
        let mut bytes = 0usize;
        let mut vertices = 0usize;
        let mut triangles = 0usize;
        for key in &keys {
            if let Some(g) = geometry_of(&body, *key) {
                built += 1;
                vertices += g.vertices.len();
                triangles += g.triangles.len();
                bytes += g.vertices.len() * 12 + g.normals.len() * 12 + g.triangles.len() * 12;
            }
        }
        let one_s = start.elapsed().as_secs_f64();
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let per = keys.len().div_ceil(threads).max(1);
        let body_ref = &body;
        let start = Instant::now();
        let built_many: usize = std::thread::scope(|scope| {
            let workers: Vec<_> = keys
                .chunks(per)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .filter(|k| geometry_of(body_ref, **k).is_some())
                            .count()
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|w| w.join().expect("a worker finishes"))
                .sum()
        });
        let many_s = start.elapsed().as_secs_f64();
        assert_eq!(built, built_many, "the threads build the same chunks");
        println!(
            "terrain_cost: client geometry, rung 0, {} chunks of {} keys: one thread {:.0} chunks/s ({:.2} ms each); {threads} threads {:.0} chunks/s ({:.1}x); {:.0} vertices, {:.0} triangles, {:.0} bytes per chunk",
            built,
            keys.len(),
            built as f64 / one_s,
            one_s * 1e3 / built.max(1) as f64,
            built as f64 / many_s,
            one_s / many_s,
            vertices as f64 / built.max(1) as f64,
            triangles as f64 / built.max(1) as f64,
            bytes as f64 / built.max(1) as f64
        );
    }

    // 3. The boot self-check.
    let start = Instant::now();
    let digest = golden_self_check(&body).expect("the home body self-checks");
    let check_ms = start.elapsed().as_secs_f64() * 1e3;
    for entry in GOLDEN_SELF_CHECK_KEYS {
        let key = self_check_key(&body, entry);
        println!("terrain_cost: self-check key {key:?}");
    }
    println!("terrain_cost: boot self-check (8 chunks) {check_ms:.1} ms, digest {digest:#018x}");

    // The gates: measurements with a wide margin, which a loaded machine cannot invert. The exact
    // property (fewer octaves at every rung) is the crate's unit test.
    let (bottom, top) = (column_us_at[0], column_us_at[column_us_at.len() - 1]);
    println!(
        "terrain_cost: the costliest named chunk (sample box + extract) is {worst_key:?} at {worst_us:.1} us against the {EXTRACT_BUDGET_US} us budget"
    );
    assert!(
        top < bottom,
        "the top rung's column pass ({top:.1} us) must cost less than rung 0's ({bottom:.1} us)"
    );
    assert!(
        skipped > evaluated,
        "the skips must refuse most of a column without a cell pass"
    );
    // The budget is the WORKER time per chunk — the sample box and the extraction together — over
    // the named set: the owner's 8 ms, gated as stated.
    assert!(
        worst_us < EXTRACT_BUDGET_US,
        "the costliest named chunk ({worst_us:.1} us) must stay under the {EXTRACT_BUDGET_US} us budget"
    );
    // The checkerboard is a BOUND (every edge of the box crossed: 250 047 vertices, 1.43 million
    // triangles), not a chunk THE world can hold; it is printed, and a regression past twice the
    // bound measured at the slice goes red.
    assert!(
        checker_us < 2.0 * WORST_CASE_MEASURED_US,
        "the worst-case extraction ({checker_us:.1} us) regressed past twice its measured bound"
    );
    println!("terrain_cost: PASS");
    ExitCode::SUCCESS
}
