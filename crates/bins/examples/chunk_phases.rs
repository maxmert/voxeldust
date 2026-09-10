//! THE PHASES OF ONE CHUNK BUILD (M8-2a, ruling V15): where a worker's time goes on THE world's
//! home planet, per rung — the sample box and the extraction (the generator's own cost, budgeted at
//! 8 ms by ruling V10), each PARENT MESH the geomorph reads (a coarser chunk's whole surface, built
//! on a cache miss), and the whole build with the parents WARM (a hit) and COLD (every parent a
//! miss). MEASURED first on the M8-1 flight: 52–64 ms per chunk on the workers against the 4 ms of
//! M7-2, and this bench names the difference.
//!
//! Run in release:
//!
//! ```text
//! cargo run --release -p vd-bins --example chunk_phases
//! ```

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use vd_bins::DEV;
use vd_client::chunks::{ParentCache, ParentMesh, geometry_with, parent_keys};
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::ChunkKey;
use vd_terrain::digest::surface_chunk_z;
use vd_terrain::extract::extract;
use vd_terrain::lattice::sample_box;

/// How many times each phase runs; the mean is printed.
const ROUNDS: u32 = 5;

fn ms(start: Instant, rounds: u32) -> f64 {
    start.elapsed().as_secs_f64() * 1e3 / f64::from(rounds)
}

fn main() -> ExitCode {
    let Some(body) = vd_bins::home_body(DEV.universe_seed) else {
        println!("chunk_phases: REFUSED — the home system holds no planet the recipe accepts");
        return ExitCode::FAILURE;
    };
    let realm = RealmId::Planet(body.seed());
    println!(
        "chunk_phases: home planet seed {} -> {} rungs; {ROUNDS} rounds per phase, release",
        body.seed(),
        body.ladder().rungs
    );
    println!(
        "{:>4} | {:>11} | {:>7} | {:>10} | {:>9} | {:>9} | {:>9} | {:>9} | bytes (an estimate)",
        "rung", "box+extract", "verts", "parents", "1 parent", "warm", "cold", "warm−box"
    );
    let mut rung = 0u8;
    while rung + 1 < body.ladder().rungs {
        // The named column of the cost bench (`terrain_cost`): face +X, chunk (3, 5) at this rung.
        let (x, y) = (3, 5);
        let key = ChunkKey {
            face: Face::PosX,
            rung,
            x,
            y,
            z: surface_chunk_z(&body, Face::PosX, rung, x, y),
        };
        // 1. The generator's own cost: the sample box and the extraction.
        let start = Instant::now();
        let mut verts = 0;
        for _ in 0..ROUNDS {
            let samples = sample_box(&body, key).expect("in the ladder");
            verts = extract(&samples).vertices.len();
        }
        let extract_ms = ms(start, ROUNDS);
        // 2. One parent mesh: the coarser chunk over this one, built as the cache builds it.
        let parents = parent_keys(&body, key);
        let start = Instant::now();
        let mut parent_bytes = 0;
        for _ in 0..ROUNDS {
            parent_bytes = ParentMesh::build(&body, parents[0]).map_or(0, |m| m.bytes());
        }
        let parent_ms = ms(start, ROUNDS);
        // 3. The whole build, WARM: every parent in the cache.
        let warm = ParentCache::default();
        for p in &parents {
            if let Some(m) = ParentMesh::build(&body, *p) {
                warm.insert(realm, *p, Arc::new(m));
            }
        }
        let start = Instant::now();
        for _ in 0..ROUNDS {
            let _ = geometry_with(&body, realm, key, &warm);
        }
        let warm_ms = ms(start, ROUNDS);
        // 4. The whole build, COLD: a fresh cache each round, every parent a miss.
        let start = Instant::now();
        for _ in 0..ROUNDS {
            let cold = ParentCache::default();
            let _ = geometry_with(&body, realm, key, &cold);
        }
        let cold_ms = ms(start, ROUNDS);
        println!(
            "{rung:>4} | {extract_ms:>8.2} ms | {verts:>7} | {:>10} | {parent_ms:>6.2} ms | \
             {warm_ms:>6.2} ms | {cold_ms:>6.2} ms | {:>6.2} ms | a parent mesh is {} KB",
            parents.len(),
            warm_ms - extract_ms,
            parent_bytes / 1024
        );
        rung += 1;
    }
    println!(
        "chunk_phases: `warm` is what a worker pays when the cache holds every parent (the \
         extraction, the morph targets, the vertex vectors); `cold` adds the parents' builds; \
         `warm−box` is everything past the box and the extraction. Five rounds each, no warm-up: \
         a mean, not a median."
    );
    ExitCode::SUCCESS
}
