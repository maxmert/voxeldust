//! ★ THE SEA FLOOR THROUGH THE SHEET (2026-09-23; the owner, from 11 500 km and 17 000 km: a
//! speckled coast at the coarser rung, and a staircase of teeth at rung 16): for each stated eye
//! over the home planet, run the ladder's descent, build every wanted chunk's box and its ground
//! mesh, and count the ground VERTICES whose four columns are all WET and that stand ABOVE the water
//! sheet's own quad over those columns — the ground poking through the water. The sheet's quad is
//! the bilinear of the four corner columns at the water's radius, which is what the client draws
//! per cell; a ground vertex is the recipe's own position law.
//! `cargo run --release -p vd-bins --example sheet_poke -- ex ey ez [ex ey ez ...]`

use std::collections::BTreeMap;
use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client::artifact_book::ArtifactReceiver;
use vd_client::ladder_view::LadderView;
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::chunk::ChunkKey;
use vd_terrain::extract::extract_all_edges;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_terrain::lattice::{SampleBox, sample_box};
use vd_terrain::position::{WEIGHT_BITS, vertex_position_m};
use vd_terrain::units::metres_of_q28;
use vd_wire::channels::BulkMsg;

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    assert!(
        a.len() >= 3 && a.len().is_multiple_of(3),
        "ex ey ez [ex ey ez ...]"
    );
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
        coast_parts: vd_bins::artifact_store::coast_parts_of(&artifact),
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
    let parts = vd_bins::artifact_store::coast_parts_of(&artifact);
    for (part, bits) in artifact
        .coast
        .chunks(vd_bins::artifact_source::COAST_PART_BYTES)
        .enumerate()
    {
        rx.accept(BulkMsg::ArtifactCoast {
            realm,
            part: part as u32,
            parts,
            bits: bits.to_vec(),
        });
    }
    let cache = rx.book().get(realm).cloned().expect("a cache");
    let lattice = body.macro_lattice().expect("a lattice");
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let mut eye_index = 0;
    while eye_index + 2 < a.len() {
        let eye = [a[eye_index], a[eye_index + 1], a[eye_index + 2]];
        eye_index += 3;
        let altitude =
            (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt() - body.radius_m();
        let mut view = LadderView::default();
        let wanted = view.wanted(&body, eye, Some(&cache));
        // Per rung: chunks, ground vertices over four wet columns, those above the sheet's quad,
        // the worst excess in metres, and the wet columns shallower than the cell's own chord dip.
        let mut per_rung: BTreeMap<u8, (u64, u64, u64, f64, u64, u64)> = BTreeMap::new();
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
            let e = per_rung.entry(k.rung).or_insert((0, 0, 0, 0.0, 0, 0));
            e.0 += 1;
            let dir_m = |a: i32, b: i32| -> [f64; 3] {
                let d = samples.dirs[SampleBox::column_index(a, b)];
                [
                    d[0].raw() as f64 / unit,
                    d[1].raw() as f64 / unit,
                    d[2].raw() as f64 / unit,
                ]
            };
            let level_m = |a: i32, b: i32| -> f64 {
                let w = samples.water[SampleBox::column_index(a, b)];
                metres_of_q28(if w > samples.sea { w } else { samples.sea })
            };
            let surface_m = |a: i32, b: i32| -> f64 {
                metres_of_q28(samples.surfaces[SampleBox::column_index(a, b)])
            };
            let mesh = extract_all_edges(&samples);
            let edge = vd_terrain::chunk::CHUNK_EDGE as i32;
            for v in &mesh.vertices {
                let mut group = [0i32; 2];
                let mut frac = [0f64; 2];
                let mut axis = 0;
                while axis < 2 {
                    let q = i32::from(v[axis]);
                    let mut cell = q >> WEIGHT_BITS;
                    if cell > edge - 1 {
                        cell = edge - 1;
                    }
                    group[axis] = cell;
                    frac[axis] = f64::from(q - (cell << WEIGHT_BITS)) / f64::from(1 << WEIGHT_BITS);
                    axis += 1;
                }
                let (ga, gb) = (group[0], group[1]);
                if ga < -1 || gb < -1 || ga + 1 > edge || gb + 1 > edge {
                    continue;
                }
                let corners = [(ga, gb), (ga + 1, gb), (ga, gb + 1), (ga + 1, gb + 1)];
                // The quad's level as the sheet takes it: the highest water among its corners.
                let mut level = 0.0f64;
                let mut all_wet = true;
                for (ca, cb) in corners {
                    let l = level_m(ca, cb);
                    if l > level {
                        level = l;
                    }
                    all_wet &= surface_m(ca, cb) < level_m(ca, cb);
                }
                if !all_wet || level <= 0.0 {
                    continue;
                }
                e.1 += 1;
                let (u, w) = (frac[0], frac[1]);
                let weights = [(1.0 - u) * (1.0 - w), u * (1.0 - w), (1.0 - u) * w, u * w];
                let mut sheet = [0.0f64; 3];
                for (i, (ca, cb)) in corners.into_iter().enumerate() {
                    let d = dir_m(ca, cb);
                    sheet[0] += weights[i] * d[0] * level;
                    sheet[1] += weights[i] * d[1] * level;
                    sheet[2] += weights[i] * d[2] * level;
                }
                let sheet_r =
                    (sheet[0] * sheet[0] + sheet[1] * sheet[1] + sheet[2] * sheet[2]).sqrt();
                let p = vertex_position_m(&body, &samples, *v);
                let ground_r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                let excess = ground_r - sheet_r;
                if excess > 0.0 {
                    e.2 += 1;
                    if excess > e.3 {
                        e.3 = excess;
                    }
                }
            }
            // The columns' own reading: wet columns shallower than one cell's chord dip.
            let cell_m = f64::from(vd_seed::ladder::cell_m(k.rung));
            let dip = cell_m * cell_m / (4.0 * body.radius_m());
            let mut b = 0;
            while b < edge {
                let mut a = 0;
                while a < edge {
                    let l = level_m(a, b);
                    let s = surface_m(a, b);
                    if s < l {
                        e.4 += 1;
                        if l - s < dip {
                            e.5 += 1;
                        }
                    }
                    a += 1;
                }
                b += 1;
            }
        }
        println!("eye altitude {altitude:.0} m");
        println!(
            "rung | chunks | wet-quad vertices | above the sheet | share | worst m | wet columns | shallower than the cell's dip"
        );
        for (rung, e) in &per_rung {
            let share = if e.1 > 0 {
                e.2 as f64 / e.1 as f64
            } else {
                0.0
            };
            println!(
                "{rung:>4} | {:>6} | {:>17} | {:>15} | {:>5.1}% | {:>7.1} | {:>11} | {:>6} ({:.1}%)",
                e.0,
                e.1,
                e.2,
                share * 100.0,
                e.3,
                e.4,
                e.5,
                if e.4 > 0 {
                    e.5 as f64 / e.4 as f64 * 100.0
                } else {
                    0.0
                }
            );
        }
    }
}
