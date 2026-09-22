//! ★ THE WANTED SET, REPLAYED (the owner's coast flight, 2026-09-20): run the client's own ladder
//! descent (`vd_client::ladder_view::LadderView::wanted`) for a recorded eye over the home planet
//! with the whole artifact in a client cache, then build every wanted chunk the way the client
//! does and report the HOLE COLUMNS — chunk columns whose every wanted slice builds with no
//! triangle — and, for named columns, exactly which slices the descent asked.
//! `cargo run --release -p vd-bins --example wanted_probe -- ex ey ez [face rung x y ...]`

use std::collections::{BTreeMap, BTreeSet};
use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client::artifact_book::ArtifactReceiver;
use vd_client::ladder_view::LadderView;
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::chunk::ChunkKey;
use vd_terrain::extract::extract_all_edges;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_terrain::lattice::sample_box;
use vd_wire::channels::BulkMsg;

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    assert!(a.len() >= 3, "ex ey ez [face rung x y ...]");
    let eye = [a[0], a[1], a[2]];
    let named: Vec<(Face, u8, i32, i32)> = a[3..]
        .chunks(4)
        .filter(|c| c.len() == 4)
        .map(|c| {
            (
                Face::from_index(c[0] as u8).expect("a face"),
                c[1] as u8,
                c[2] as i32,
                c[3] as i32,
            )
        })
        .collect();
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
        "wanted: {} keys, {} provisional spans of {}",
        wanted.keys.len(),
        view.provisional_spans(),
        view.spans_held()
    );
    let mut per_rung: BTreeMap<u8, usize> = BTreeMap::new();
    for k in &wanted.keys {
        *per_rung.entry(k.rung).or_default() += 1;
    }
    println!("per rung: {per_rung:?}");
    for (face, rung, x, y) in &named {
        let zs: Vec<i32> = wanted
            .keys
            .iter()
            .filter(|k| k.face == *face && k.rung == *rung && k.x == *x && k.y == *y)
            .map(|k| k.z)
            .collect();
        let (span, provisional) = vd_client::ladder_view::column_span(
            &body,
            Some(&cache),
            vd_client::ladder_view::Column {
                face: *face,
                rung: *rung,
                x: *x,
                y: *y,
            },
        );
        println!(
            "column {face:?} rung {rung} ({x}, {y}): wanted slices {zs:?}; column_span {}..={} provisional {provisional}",
            span.lo, span.hi
        );
    }
    // Every wanted chunk, built as the client builds it: which columns hold no ground at all.
    let mut empties = 0usize;
    let mut columns: BTreeMap<(Face, u8, i32, i32), (usize, usize)> = BTreeMap::new();
    let mut field_missing = 0usize;
    for k in &wanted.keys {
        let level =
            vd_terrain::artifact::PyramidField::level_for(&lattice, cache.head.levels, k.rung);
        let field: Option<std::sync::Arc<dyn vd_terrain::artifact::ZField + Send + Sync>> =
            cache.field_at_rung(&lattice, k.rung);
        let Some(field) = field else {
            field_missing += 1;
            continue;
        };
        let key = ChunkKey {
            face: k.face,
            rung: k.rung,
            x: k.x,
            y: k.y,
            z: k.z,
        };
        let entry = columns.entry((k.face, k.rung, k.x, k.y)).or_insert((0, 0));
        entry.0 += 1;
        let Some(samples) = sample_box(&body, Some(&*field), key) else {
            field_missing += 1;
            let _ = level;
            continue;
        };
        let mesh = extract_all_edges(&samples);
        if mesh.triangles.is_empty() {
            empties += 1;
        } else {
            entry.1 += 1;
        }
    }
    let holes: Vec<_> = columns
        .iter()
        .filter(|(_, (asked, with_ground))| *asked > 0 && *with_ground == 0)
        .collect();
    let mut per_rung_holes: BTreeMap<u8, usize> = BTreeMap::new();
    for ((_, rung, _, _), _) in &holes {
        *per_rung_holes.entry(*rung).or_default() += 1;
    }
    println!(
        "built {} chunks: {empties} empty, {field_missing} with no field; {} columns, {} HOLE columns (no wanted slice holds ground) per rung {per_rung_holes:?}",
        wanted.keys.len(),
        columns.len(),
        holes.len()
    );
    let sample: BTreeSet<_> = holes.iter().take(12).map(|(c, _)| **c).collect();
    println!("hole columns (first 12): {sample:?}");
}
