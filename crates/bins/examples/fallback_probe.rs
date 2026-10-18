//! ★ THE MORPH FALLBACKS ON THE SOLVED GROUND, WITH EVERY PART HELD (2026-10-01, the picture gate on the
//! solved stands, ruling B28 item 2): solve the home planet, hand a client cache every part, run the
//! client's own descent at a stated eye, and build every wanted chunk by the client's own build with
//! one shared parent cache (`vd_client::chunks::geometry_with`). The fallbacks by rung, against the
//! picture gate's bound (one vertex in a hundred). A count near the live run's says the ground itself
//! makes them; a count far under it says the live landing's order (a chunk built while a tile its
//! parent reads was missing) left them behind.
//! `cargo run --release -p vd-bins --example fallback_probe -- ex ey ez`

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client::artifact_book::ArtifactReceiver;
use vd_client::chunks::{ParentCache, geometry_with};
use vd_client::ladder_view::LadderView;
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::chunk::ChunkKey;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_wire::channels::BulkMsg;

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    assert!(a.len() >= 3, "ex ey ez");
    let eye = [a[0], a[1], a[2]];
    let artifact = run_solve(&SolveJob {
        body: home_planet(),
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let wet = home_planet().with_sea_m(artifact.sea());
    // ★ THE BAND THE HEAD SIZES (ruling B55), as the client's lane takes it from the same head.
    let drawn = artifact.drawn_range_m(&wet.macro_lattice().expect("a lattice"));
    let body = wet.with_band_m(drawn.0, drawn.1).unwrap_or(wet);
    let realm = RealmId::Planet(HOME_PLANET_SEED);
    let mut rx = ArtifactReceiver::default();
    rx.accept(BulkMsg::ArtifactHead {
        realm,
        world_tag: 0,
        version: artifact.version,
        edge: artifact.edge,
        digest: artifact.digest(),
        tiles_per_edge: artifact.tiles_per_edge(),
        levels: artifact.pyramid.len() as u32,
        sea_m: artifact.sea_m,
        coast_parts: 0,
        low_m: drawn.0,
        high_m: drawn.1,
        wind_parts: 0,
    });
    for (k, level) in artifact.pyramid.iter().enumerate().rev() {
        rx.accept(BulkMsg::ArtifactPyramid {
            realm,
            level: k as u32 + 1,
            part: 0,
            parts: 1,
            z_m: level.clone(),
            water_m: Vec::new(),
        });
    }
    for face in Face::ALL {
        for tx in 0..artifact.tiles_per_edge() {
            for ty in 0..artifact.tiles_per_edge() {
                rx.accept(BulkMsg::ArtifactTile {
                    realm,
                    face: face.index(),
                    tx,
                    ty,
                    rows: artifact.tile(face, tx, ty).to_bytes(),
                });
            }
        }
    }
    let cache = rx.book().get(realm).cloned().expect("a cache");
    let mut view = LadderView::default();
    let wanted = view.wanted(&body, eye, Some(&cache));
    let keys: Vec<ChunkKey> = wanted
        .keys
        .iter()
        .map(|k| ChunkKey {
            face: k.face,
            rung: k.rung,
            x: k.x,
            y: k.y,
            z: k.z,
        })
        .collect();
    println!("{} wanted keys", keys.len());
    let parents = ParentCache::default();
    let next = AtomicUsize::new(0);
    let tally: Mutex<BTreeMap<u8, (u64, u64, u64)>> = Mutex::new(BTreeMap::new());
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(key) = keys.get(i) else {
                        break;
                    };
                    let Some(g) = geometry_with(&body, Some(&cache), realm, *key, &parents) else {
                        continue;
                    };
                    let mut t = tally.lock().expect("the tally");
                    let row = t.entry(key.rung).or_insert((0, 0, 0));
                    row.0 += u64::from(g.morph_fallbacks);
                    row.1 += u64::from(g.morph_seam);
                    row.2 += g.vertices.len() as u64;
                }
            });
        }
    });
    let tally = tally.into_inner().expect("the tally");
    let (mut f, mut v) = (0u64, 0u64);
    for (rung, (fallbacks, seam, vertices)) in &tally {
        f += fallbacks;
        v += vertices;
        println!(
            "rung {rung}: {fallbacks} fallbacks of {vertices} vertices ({:.3} %), {seam} on a seam",
            100.0 * *fallbacks as f64 / (*vertices).max(1) as f64
        );
    }
    println!(
        "ALL: {f} fallbacks of {v} vertices ({:.3} %; the gate allows 1 %)",
        100.0 * f as f64 / v.max(1) as f64
    );
}
