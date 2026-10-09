//! ★ DO THE SHARD'S TILES HOLD WHAT THE CLIENT'S CHUNKS READ? (2026-10-01, the picture gate's seam
//! stand): solve the home planet, hand a client cache every part, run the client's own descent
//! (`LadderView::wanted`) at a stated eye, gather every tile its wanted chunks' stencils read
//! (`tiles_of_chunk`), and hold that set against the tiles the shard names for an occupant at that
//! eye (`ArtifactTiles::tiles_under`, the shipped source). A tile the client reads and the shard does
//! not name is a chunk that waits for as long as the eye stands there.
//! `cargo run --release -p vd-bins --example seam_tiles -- ex ey ez`

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client::artifact_book::ArtifactReceiver;
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
    let lattice = body.macro_lattice().expect("a lattice");
    let len = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
    let dir = [eye[0] / len, eye[1] / len, eye[2] / len];
    let surface = vd_client::chunks::drawn_surface_m(&body, Some(&cache), dir);
    let mut view = LadderView::default();
    let wanted = view.wanted(&body, eye, Some(&cache));
    println!(
        "the eye {:.1} m over the radius, {:.1} m over the drawn ground; {} wanted keys",
        len - body.radius_m(),
        len - surface,
        wanted.keys.len()
    );
    // The tiles the client reads, each with the nearest wanted chunk that reads it: only a chunk
    // whose rung reads the rows (level 0) reads tiles; a coarser rung reads a pyramid level.
    let levels = cache.head.levels;
    let mut read: BTreeMap<(u8, u32, u32), (f64, ChunkKey)> = BTreeMap::new();
    let mut tile_rungs: BTreeMap<u8, usize> = BTreeMap::new();
    for k in &wanted.keys {
        if vd_terrain::artifact::PyramidField::level_for(&lattice, levels, k.rung) != 0 {
            continue;
        }
        *tile_rungs.entry(k.rung).or_default() += 1;
        let key = ChunkKey {
            face: k.face,
            rung: k.rung,
            x: k.x,
            y: k.y,
            z: k.z,
        };
        // The chunk's column centre on the drawn ground, from its face position.
        let n = f64::from(body.ladder().cells_per_edge(key.rung));
        let edge = f64::from(vd_seed::ladder::CHUNK_EDGE as u32);
        let param = |c: i32| -1.0 + 2.0 * (f64::from(c) * edge + edge / 2.0) / n;
        let d_col = vd_seed::bend::direction(key.face, param(key.x), param(key.y));
        let ground = vd_client::chunks::drawn_surface_m(&body, Some(&cache), d_col);
        let d = ((d_col[0] * ground - eye[0]).powi(2)
            + (d_col[1] * ground - eye[1]).powi(2)
            + (d_col[2] * ground - eye[2]).powi(2))
        .sqrt();
        for tile in vd_terrain::artifact::tiles_of_chunk(&lattice, key) {
            let entry = read.entry(tile).or_insert((f64::MAX, key));
            if d < entry.0 {
                *entry = (d, key);
            }
        }
    }
    println!("the wanted chunks that read the rows, by rung: {tile_rungs:?}");
    // The chunk the live run waited on (2026-10-01): is it wanted here, and where does it stand?
    let watched = ChunkKey {
        face: Face::PosX,
        rung: 9,
        x: 47,
        y: 0,
        z: 8,
    };
    let in_set = wanted.keys.iter().any(|k| {
        (k.face, k.rung, k.x, k.y, k.z)
            == (watched.face, watched.rung, watched.x, watched.y, watched.z)
    });
    println!(
        "the live run's waiting chunk {watched:?}: wanted here {in_set}, reads {:?}",
        vd_terrain::artifact::tiles_of_chunk(&lattice, watched)
    );
    let source = vd_bins::artifact_source::artifact_source_of(realm, 0, &body, &Arc::new(artifact))
        .expect("a source");
    let named: BTreeSet<(u8, u32, u32)> = source.tiles_under(dir, len, 0.0).into_iter().collect();
    println!(
        "the client's chunks read {} tiles; the shard names {}",
        read.len(),
        named.len()
    );
    let mut short = 0usize;
    for (tile, (d, key)) in &read {
        if !named.contains(tile) {
            short += 1;
            println!(
                "  NOT NAMED: tile {tile:?}, read by {key:?} at {:.0} km",
                d / 1000.0
            );
        }
    }
    let extra = named.iter().filter(|t| !read.contains_key(t)).count();
    println!("{short} tiles read and not named; {extra} named and not read");
}
