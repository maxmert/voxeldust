//! ★ THE WATER SHEET'S TRIANGLES on the coast eye's wanted set (2026-09-21): for the recorded eye,
//! run the ladder's descent over the home planet's whole artifact, build every wanted chunk's box,
//! and count the sheet's triangles per rung under three hide bounds — none (a corner must be wet),
//! the client's own (`step_bound + sink + cell`), and every quad — beside the land's triangles. It
//! says how much the cut removes and where the sheet's cost stands.
//! `cargo run --release -p vd-bins --example sheet_cost -- ex ey ez`

use std::collections::BTreeMap;
use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client::artifact_book::ArtifactReceiver;
use vd_client::chunks::sink_m;
use vd_client::ladder_view::LadderView;
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::chunk::ChunkKey;
use vd_terrain::extract::extract_all_edges;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_terrain::lattice::sample_box;
use vd_terrain::position::water_sheet;
use vd_terrain::units::q28_of_metres;
use vd_wire::channels::BulkMsg;

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    assert!(a.len() >= 3, "ex ey ez");
    let eye = [a[0], a[1], a[2]];
    let body = home_planet();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
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
    });
    for (k, level) in artifact.pyramid.iter().enumerate().rev() {
        rx.accept(BulkMsg::ArtifactPyramid {
            realm,
            level: k as u32 + 1,
            part: 0,
            parts: 1,
            z_m: level.clone(),
            water_m: artifact.pyramid_water[k].clone(),
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
    let lattice = body.macro_lattice().expect("a lattice");
    let mut view = LadderView::default();
    let wanted = view.wanted(&body, eye, Some(&cache));
    println!(
        "rung | hide m | chunks | land triangles | sheet: no hide | client's hide | every quad"
    );
    let mut per_rung: BTreeMap<u8, [u64; 5]> = BTreeMap::new();
    for k in &wanted.keys {
        let Some(field) = cache.field_at_rung(&lattice, k.rung) else {
            continue;
        };
        let key = ChunkKey {
            face: k.face,
            rung: k.rung,
            x: k.x,
            y: k.y,
            z: k.z,
        };
        let Some(samples) = sample_box(&body, Some(&*field), key) else {
            continue;
        };
        let hide_m = body.step_bound_m(k.rung)
            + sink_m(&body, k.rung)
            + f64::from(vd_seed::ladder::cell_m(k.rung));
        let mesh = extract_all_edges(&samples);
        let land = mesh.triangles.len() as u64;
        let none = water_sheet(&body, &samples, &mesh, vd_recipe::Gi::ZERO)
            .2
            .len() as u64;
        let own = water_sheet(&body, &samples, &mesh, q28_of_metres(hide_m))
            .2
            .len() as u64;
        let all = water_sheet(&body, &samples, &mesh, vd_recipe::Gi::new(i64::MAX >> 2))
            .2
            .len() as u64;
        let e = per_rung.entry(k.rung).or_insert([0; 5]);
        e[0] += 1;
        e[1] += land;
        e[2] += none;
        e[3] += own;
        e[4] += all;
    }
    let (mut land_t, mut own_t, mut all_t) = (0, 0, 0);
    for (rung, e) in &per_rung {
        let hide_m = body.step_bound_m(*rung)
            + sink_m(&body, *rung)
            + f64::from(vd_seed::ladder::cell_m(*rung));
        println!(
            "{rung:>4} | {hide_m:>6.0} | {:>6} | {:>14} | {:>14} | {:>13} | {:>10}",
            e[0], e[1], e[2], e[3], e[4]
        );
        land_t += e[1];
        own_t += e[3];
        all_t += e[4];
    }
    println!("total land {land_t}, sheet with the client's hide {own_t}, every quad {all_t}");
}
