//! Diagnosis probe (2026-10-13, send-ahead's measurement: a raw 128 m tile took 6-16 s on one lowered solve thread):
//! solve the raw tiles nearest the hull's berth over the home planet from the realm's STORED artifact, one thread, and
//! time each — the work the shard's fine workers do per tile, for the CPU profiler to split.
//! `VD_REALM_STORE=<file> cargo run --release -p vd-bins --example fine_solve_profile -- [tiles] [threads] [lowered]`
//! With `threads` > 1 the tiles are shared out over that many threads at once (the shard's fine workers' shape); with
//! `lowered` = 1 each thread first takes the shard's solve class (`vd_bins::thread_class`, ruling B91).

use std::collections::BTreeSet;

use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::fine::l1::{TileKey, solve_l1};

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;
const HOME: u64 = 4_030_111_653_607_004_909;
/// The hull's berth over the home planet (the dev cluster's spawn pose, metres in the planet's frame).
const BERTH: [f64; 3] = [
    3_273_214.798_924_267_3,
    -2_850_131.846_649_597,
    -4_630_822.805_490_945_5,
];

fn main() {
    let arg = |n: usize, default: usize| -> usize {
        std::env::args()
            .nth(n)
            .map_or(default, |s| s.parse().expect("a count"))
    };
    let (tiles, threads, lowered) = (arg(1, 4), arg(2, 1).max(1), arg(3, 0) == 1);
    let realm = RealmId::Planet(HOME);
    let (rows, _) = vd_bins::home_system_world(UNIVERSE);
    let radius_m = rows
        .iter()
        .find(|r| r.realm == realm)
        .and_then(|r| match r.look {
            Some(vd_core::geometry::Boundary::Shell { r }) => Some(r),
            _ => None,
        })
        .expect("the home planet is round");
    let charter = vd_bins::body_charter(
        UNIVERSE,
        &BTreeSet::from([vd_core::worldgen::HOME_SYSTEM]),
        &BTreeSet::from([vd_core::worldgen::GALAXY]),
        realm,
    )
    .expect("a charter");
    let body = BodyDefinition::from_seed(HOME, radius_m, vd_bins::facts_of_charter(&charter))
        .expect("a ladder");
    let lattice = body.macro_lattice().expect("a lattice");
    let env = vd_bins::process_env();
    let store = vd_bins::open_realm_store(&env, realm)
        .expect("the store opens")
        .expect("VD_REALM_STORE names a file");
    let artifact = vd_bins::artifact_store::read_artifact(
        &store,
        vd_terrain::artifact_world_tag(UNIVERSE),
        &lattice,
    )
    .expect("the stored rows agree with their head")
    .expect("an artifact is stored");
    let radial = (BERTH[0] * BERTH[0] + BERTH[1] * BERTH[1] + BERTH[2] * BERTH[2]).sqrt();
    let dir = [BERTH[0] / radial, BERTH[1] / radial, BERTH[2] / radial];
    let drawn = artifact.drawn_range_m(&lattice);
    let named = vd_terrain::artifact::fine_needs_of_head(
        vd_terrain::fine::level::FineLevel::L1,
        &body,
        (drawn.0, drawn.1, artifact.sea_m),
        dir,
        radial,
        vd_core::geometry::drawable_theta_min_rad(),
    );
    let keys: Vec<(TileKey, u8)> = named
        .iter()
        .filter(|(_, l)| *l == 0)
        .take(tiles)
        .map(|&((face, tx, ty), level)| {
            (
                TileKey {
                    face: Face::from_index(face).expect("a face"),
                    tx: tx as i32,
                    ty: ty as i32,
                },
                level,
            )
        })
        .collect();
    let wall = std::time::Instant::now();
    let times: Vec<f64> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|w| {
                let (body, lattice, artifact, keys) = (&body, &lattice, &artifact, &keys);
                scope.spawn(move || {
                    if lowered {
                        vd_bins::thread_class::lower_to_solve_class();
                    }
                    let mut times = Vec::new();
                    for &(key, level) in keys.iter().skip(w).step_by(threads) {
                        let started = std::time::Instant::now();
                        let solved = solve_l1(body, lattice, artifact, key);
                        let ms = started.elapsed().as_secs_f64() * 1e3;
                        times.push(ms);
                        println!(
                            "thread {w} tile {key:?} level {level}: {ms:.0} ms, {} points, passes {:?}",
                            solved.as_ref().map_or(0, |t| t.points),
                            solved
                                .as_ref()
                                .map(|t| t.passes.clone())
                                .unwrap_or_default()
                        );
                    }
                    times
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("a worker"))
            .collect()
    });
    let wall_s = wall.elapsed().as_secs_f64();
    println!(
        "{} tiles on {threads} threads (lowered: {lowered}), mean {:.0} ms a tile, wall {wall_s:.1} s, {:.2} tiles a second",
        times.len(),
        times.iter().sum::<f64>() / times.len().max(1) as f64,
        times.len() as f64 / wall_s
    );
}
