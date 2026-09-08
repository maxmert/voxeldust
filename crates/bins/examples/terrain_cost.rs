//! THE GENERATOR's measurements (the voxel foundation, slice 5; ruling V8 addendum: the coarse-answer
//! measurement runs FIRST, before any terrain is drawn), on THE world's home planet, with the real
//! hash — never a proxy bench:
//!
//! 1. the cost of one surface chunk at EVERY rung, and the measurement: the top rung's column pass
//!    costs less than rung 0's (the EXACT half — fewer octaves at every rung — is a unit test in
//!    the generator crate, `every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it`);
//! 2. how many chunks of one NAMED radial column the skips refuse without a cell pass;
//! 3. the boot self-check's cost (eight chunks) and its digest;
//! 4. the home planet's seed and ladder, the same body `vd_bins::home_body` gives the gateway.
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
use vd_terrain::chunk::{How, column_field, generate, generate_in};
use vd_terrain::digest::{golden_self_check, self_check_key, surface_chunk_z};
use vd_terrain::{ChunkKey, GOLDEN_SELF_CHECK_KEYS};

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
    assert!(
        top < bottom,
        "the top rung's column pass ({top:.1} us) must cost less than rung 0's ({bottom:.1} us)"
    );
    assert!(
        skipped > evaluated,
        "the skips must refuse most of a column without a cell pass"
    );
    println!("terrain_cost: PASS");
    ExitCode::SUCCESS
}
