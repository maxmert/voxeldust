//! ★ THE NEAR HAND-OVER STEPS ON THE STORED FINE GROUND (ruling B95; the owner's "measure first", 2026-10-14).
//!
//! The finished 128 m and 16 m tiles around a stand, read back from the home planet's realm store (nothing solved), and
//! at each near rung `r` how far the rung's surface stands from the next coarser rung's surface between that rung's
//! samples — the step a hand-over from `r` to `r + 1` shows — on the 16 m ground and on the 128 m ground alone. Beside
//! them the client's bound for the step (`ladder_view::curve_step_m`) by each level's steepest slope.
//!
//! **Example.** At hull 17's stand on the day-side hills: the 1 m rung against the 2 m rung over every stored 16 m tile
//! near the stand, the median, the 99th and 99.9th percentiles and the largest step in metres.
//!
//! `VD_REALM_STORE=<a copy of the home planet's store> cargo run --release -p vd-bins --example fine_steps -- X,Y,Z`

use std::collections::BTreeSet;
use std::sync::Arc;

use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_sim::stub::built_store::{FineAt, FineForm, read_fine_record};
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::ZField;
use vd_terrain::fine::l1::TileKey;
use vd_terrain::fine::level::FineLevel;
use vd_terrain::fine::read::{FineBook, FineField, FineZ};
use vd_terrain::fine::tile::{decode_levels, input_word_of};
use vd_terrain::units::{LENGTH_BITS, STEPS_PER_M};

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;
const HOME: u64 = 4_030_111_653_607_004_909;
/// The samples a tile gives each rung, along each axis.
const SAMPLES: i64 = 256;

fn main() {
    let eye: Vec<f64> = std::env::args()
        .nth(1)
        .expect("X,Y,Z in the planet's frame")
        .split(',')
        .map(|x| x.trim().parse().expect("a number"))
        .collect();
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
    let radial = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
    let dir = [eye[0] / radial, eye[1] / radial, eye[2] / radial];
    let drawn = artifact.drawn_range_m(&lattice);
    let body = body.with_sea_m(artifact.sea());
    // Both levels' stored finished tiles the stand names.
    let book = |level: FineLevel| {
        let mut book =
            FineBook::new(level, &body, &lattice, artifact.sea_m).expect("the level applies");
        let input = input_word_of(level, artifact.digest());
        let named = vd_terrain::artifact::fine_needs_of_head(
            level,
            &body,
            (drawn.0, drawn.1, artifact.sea_m),
            dir,
            radial,
            vd_core::geometry::drawable_theta_min_rad(),
        );
        for &((f, tx, ty), _) in &named {
            let at = FineAt {
                form: FineForm::Finished,
                input,
                level: level.id(),
                face: f,
                tx,
                ty,
            };
            if let Ok(Some(bytes)) = read_fine_record(&store, &at)
                && let Some((_, levels)) = decode_levels(&bytes, level)
            {
                let key = TileKey {
                    face: Face::from_index(f).expect("a face"),
                    tx: tx as i32,
                    ty: ty as i32,
                };
                assert!(book.insert(key, levels), "a stored tile is held");
            }
        }
        println!(
            "{level:?}: {} of {} named tiles stored",
            book.len(),
            named.len()
        );
        book
    };
    let (l1, l2) = (Arc::new(book(FineLevel::L1)), Arc::new(book(FineLevel::L2)));
    let inner: Arc<dyn ZField + Send + Sync> = Arc::new(artifact);
    let fields = [
        (
            "the 16 m ground",
            FineField {
                inner: Arc::clone(&inner),
                book: Arc::clone(&l1),
                finer: Some(Arc::clone(&l2)),
            },
        ),
        (
            "the 128 m ground",
            FineField {
                inner,
                book: Arc::clone(&l1),
                finer: None,
            },
        ),
    ];
    let unit = (STEPS_PER_M as f64) * f64::from(1u32 << LENGTH_BITS);
    let metres = |z: FineZ| match z {
        FineZ::Held(g) => Some(g.raw() as f64 / unit),
        FineZ::Missing(_) | FineZ::Outside => None,
    };
    let tiles: Vec<TileKey> = l2.keys().collect();
    let l2_rung = FineLevel::L2.rung();
    println!(
        "rung  cell   ground            samples   median    99 %     99.9 %   largest   bound(128 m)  bound(16 m)"
    );
    for r in 0..=5u8 {
        for (name, field) in &fields {
            let per_tile: Vec<Vec<f64>> = std::thread::scope(|scope| {
                let handles: Vec<_> = tiles
                    .iter()
                    .map(|&key| {
                        scope.spawn(move || {
                            // The tile's rung-r cells, sampled on a regular grid of SAMPLES × SAMPLES.
                            let side = 512i64 << (l2_rung - r.min(l2_rung));
                            let side = if r > l2_rung {
                                512 >> (r - l2_rung)
                            } else {
                                side
                            };
                            let (i0, j0) = (i64::from(key.tx) * side, i64::from(key.ty) * side);
                            let stride = (side / SAMPLES).max(1) | 1;
                            let mut out = Vec::new();
                            let mut j = j0 + 1;
                            while j < j0 + side - 1 {
                                let mut i = i0 + 1;
                                while i < i0 + side - 1 {
                                    let at = |rung: u8, a: i64, b: i64| {
                                        metres(
                                            field.fine_z(key.face, rung, rung, a as i32, b as i32),
                                        )
                                    };
                                    let (ci, ti) = (
                                        (i - 1).div_euclid(2),
                                        if i % 2 == 0 { 0.75 } else { 0.25 },
                                    );
                                    let (cj, tj) = (
                                        (j - 1).div_euclid(2),
                                        if j % 2 == 0 { 0.75 } else { 0.25 },
                                    );
                                    if let (Some(z), Some(a), Some(b), Some(c), Some(d)) = (
                                        at(r, i, j),
                                        at(r + 1, ci, cj),
                                        at(r + 1, ci + 1, cj),
                                        at(r + 1, ci, cj + 1),
                                        at(r + 1, ci + 1, cj + 1),
                                    ) {
                                        let coarse = (a * (1.0 - ti) + b * ti) * (1.0 - tj)
                                            + (c * (1.0 - ti) + d * ti) * tj;
                                        out.push((z - coarse).abs());
                                    }
                                    i += stride;
                                }
                                j += stride;
                            }
                            out
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("a tile"))
                    .collect()
            });
            let mut all: Vec<f64> = per_tile.into_iter().flatten().collect();
            all.sort_by(f64::total_cmp);
            let q = |p: f64| all[((all.len() as f64 - 1.0) * p) as usize];
            if all.is_empty() {
                println!("{r:>4}  {name}: no sample");
                continue;
            }
            println!(
                "{r:>4}  {:>4} m  {name:<16} {:>9}  {:>7.3}  {:>7.3}  {:>7.3}  {:>8.3}  {:>12.3}  {:>11.3}",
                1u32 << r,
                all.len(),
                q(0.5),
                q(0.99),
                q(0.999),
                all[all.len() - 1],
                vd_client::ladder_view::curve_step_m(FineLevel::L1, r),
                vd_client::ladder_view::curve_step_m(FineLevel::L2, r),
            );
        }
    }
}
