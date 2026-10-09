//! ★ THE FAR GROUND THE CULLING DROPS, MEASURED (ruling B55 item 2, 2026-10-05; the owner: "Agree, go" —
//! the far summits the culling may drop, measured headless before any per-tile range is asked for).
//!
//! The client's descent (`vd_client::ladder_view::LadderView::wanted`) culls a far column when its
//! peak bound stands under the skyline the near ground raises. At the rungs that read a pyramid
//! level the bound reads that level, whose node is the MEAN of four — so a column could be culled
//! though the ground the client would draw there stands in sight. This tool replays the descent for
//! an eye over the home planet with the whole artifact held, and beside it marches the TRUE skyline
//! by brute force: along each azimuth (every 0.1°) the ground at the rung the client draws at that
//! distance (the tier rule's own switch distances, the level that rung reads), through the shipped
//! reader `vd_terrain::height::height_field_m`. A sample is VISIBLE where it stands over every nearer
//! sample of its azimuth; a visible sample whose chunk column the descent wants at none of its own
//! rung and the two beside it is DROPPED ground.
//!
//! `cargo run --release -p vd-bins --example far_cull -- <artifact_file> <stands_file>`: the eyes are
//! the stands file's (`account|label|entry|…`, the entry `account=Planet(seed):x,y,z@q`), the home
//! planet's only.
//!
//! **Example.** From 1 500 m over the plateau's rim a range 180 km away stands over the near ridge;
//! if the descent drops its column, its samples count here, with their distance and how far they
//! stand over the near ridge's line.

use std::collections::BTreeSet;
use vd_client::artifact_book::ArtifactReceiver;
use vd_client::ladder_view::{LadderView, switch_m};
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_seed::bend::{Face, face_coords, face_of, unbend};
use vd_seed::ladder::index_of;
use vd_terrain::artifact::{PyramidField, ZField};
use vd_terrain::chunk::CHUNK_EDGE;
use vd_terrain::height::height_field_m;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet};
use vd_wire::channels::BulkMsg;

/// Azimuths around the eye.
const AZIMUTHS: usize = 3_600;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() >= 2, "<artifact_file> <stands_file>");
    let artifact = vd_bins::artifact_file::read(&args[0]).expect("the artifact file");
    let wet = home_planet().with_sea_m(artifact.sea());
    let lattice = wet.macro_lattice().expect("a lattice");
    let drawn = artifact.drawn_range_m(&lattice);
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
    let levels = artifact.pyramid.len() as u32;
    let counts = artifact.coast_counts();
    let top = body.ladder().rungs - 1;
    let fields: Vec<Box<dyn ZField>> = (0..=top)
        .map(|r| -> Box<dyn ZField> {
            let level = PyramidField::level_for(&lattice, levels, r);
            if level == 0 {
                Box::new(artifact.clone())
            } else {
                Box::new(PyramidField::of(&artifact, level, &counts).expect("a level"))
            }
        })
        .collect();
    // The rung the client draws at a SLANT distance from the eye (the tier rule's switch distances are the eye's own,
    // `ground_radius_m`): the first whose switch distance lies past it.
    let rung_at = |d: f64| -> u8 { (0..top).find(|&r| d < switch_m(r)).unwrap_or(top) };
    let column_of = |dir: [f64; 3], rung: u8| -> (Face, u8, i32, i32) {
        let face = face_of(dir);
        let (t, s) = face_coords(face, dir);
        let n = body.ladder().cells_per_edge(rung);
        let edge = CHUNK_EDGE as i32;
        (
            face,
            rung,
            index_of(unbend(t), n).div_euclid(edge),
            index_of(unbend(s), n).div_euclid(edge),
        )
    };
    println!(
        "far_cull: the home planet, the drawn range {} .. {} m, {} rungs; {AZIMUTHS} azimuths an eye",
        drawn.0,
        drawn.1,
        body.ladder().rungs
    );
    println!(
        "\nthe stand | eye over the ground m | wanted chunks | visible samples | DROPPED samples | dropped's distance km (min / median / max) | dropped's height over the skyline (max, mrad)"
    );
    for line in std::fs::read_to_string(&args[1])
        .expect("the stands file")
        .lines()
    {
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() < 3 || !parts[1].starts_with("home_") {
            continue;
        }
        let pos = parts[2]
            .split(':')
            .nth(1)
            .and_then(|p| p.split('@').next())
            .expect("a position");
        let e: Vec<f64> = pos
            .split(',')
            .map(|x| x.parse().expect("a number"))
            .collect();
        let eye = DVec3::new(e[0], e[1], e[2]);
        let mut view = LadderView::default();
        let wanted = view.wanted(&body, eye.to_array(), Some(&cache));
        let cols: BTreeSet<(Face, u8, i32, i32)> = wanted
            .keys
            .iter()
            .map(|k| (k.face, k.rung, k.x, k.y))
            .collect();
        let up = eye.normalize();
        let ground0 =
            height_field_m(&body, fields[0].as_ref(), up.to_array(), 0).expect("the ground");
        let east = up.cross(DVec3::Z).normalize();
        let north = east.cross(up);
        let radius = body.radius_m();
        let (mut visible, mut dropped) = (0usize, 0usize);
        let mut dropped_d: Vec<f64> = Vec::new();
        let mut over_max = 0.0f64;
        for k in 0..AZIMUTHS {
            let az = (k as f64 + 0.5) / AZIMUTHS as f64 * std::f64::consts::TAU;
            let t = east * az.cos() + north * az.sin();
            let mut sky = f64::NEG_INFINITY;
            let mut s = 50.0;
            while s < 700_000.0 {
                let theta = s / radius;
                let dir = (up * theta.cos() + t * theta.sin()).to_array();
                // The slant distance to the ground near this sample, read once at the finest rung.
                let near = height_field_m(&body, fields[0].as_ref(), dir, 0).unwrap_or(radius);
                let slant = (DVec3::from_array(dir) * near - eye).length();
                let rung = rung_at(slant);
                if let Some(h) =
                    height_field_m(&body, fields[usize::from(rung)].as_ref(), dir, rung)
                {
                    let p = DVec3::from_array(dir) * h;
                    let v = p - eye;
                    let elev = (v.dot(up) / v.length()).asin();
                    if elev >= sky {
                        visible += 1;
                        let (face, _, x, y) = column_of(dir, rung);
                        let covered = [rung.saturating_sub(1), rung, (rung + 1).min(top)]
                            .iter()
                            .any(|&r| {
                                let (f, rr, cx, cy) = if r == rung {
                                    (face, rung, x, y)
                                } else {
                                    column_of(dir, r)
                                };
                                cols.contains(&(f, rr, cx, cy))
                            });
                        if !covered {
                            dropped += 1;
                            dropped_d.push(s);
                            if sky.is_finite() {
                                over_max = over_max.max(elev - sky);
                            }
                        }
                        sky = elev;
                    }
                }
                s += (s / 400.0).max(50.0);
            }
        }
        dropped_d.sort_by(f64::total_cmp);
        let q = |p: f64| {
            if dropped_d.is_empty() {
                0.0
            } else {
                dropped_d[((dropped_d.len() as f64 * p) as usize).min(dropped_d.len() - 1)]
                    / 1_000.0
            }
        };
        println!(
            "{:<28} | {:>9.0} | {:>8} | {:>9} | {:>8} | {:>7.1} / {:>7.1} / {:>7.1} | {:>8.3}",
            parts[1],
            eye.length() - ground0,
            wanted.keys.len(),
            visible,
            dropped,
            q(0.0),
            q(0.5),
            q(1.0),
            over_max * 1_000.0
        );
    }
}
