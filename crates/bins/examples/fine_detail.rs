//! ★ WHERE THE 16 m DETAIL STANDS FAR FROM TODAY'S GROUND (B5's first measurement, 2026-10-14). `fine_steps` found steps
//! of up to 1 338 m between a 16 m cell and its 32 m mean at hull 19's valley, against 52 m at hull 17's hills. This bench
//! reads the same stored tiles and, for every 16 m tile, the DETAIL — the 16 m ground less today's 128 m ground at each
//! 16 m cell — and the steepest 16 m link, and lists the worst tiles with where their worst cell stands.
//!
//! The rest of this header is `fine_steps`' own:
//!
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
//! `VD_REALM_STORE=<a copy of the home planet's store> cargo run --release -p vd-bins --example fine_detail -- X,Y,Z`

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

/// One 16 m tile's readings: its largest detail (the 16 m ground less today's 128 m ground), that cell and today's ground
/// there, its cells over 30 m, its steepest 16 m link and that cell, and its mean detail — all in metres.
struct Row {
    key: TileKey,
    worst: f64,
    at: (i64, i64),
    ground: f64,
    over30: usize,
    steep: f64,
    steep_at: (i64, i64),
    mean: f64,
    /// The tile's 16 m links (each cell with its east and south neighbour) and those over 35°, 45° and 63°; those over
    /// 63° that touch the tile's rim (within 8 cells of its edge).
    links: [usize; 5],
}

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
    let artifact_sea_m = artifact.sea_m;
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
    let [fine, coarse] = &fields;
    let l2_rung = FineLevel::L2.rung();
    let tiles: Vec<TileKey> = l2.keys().collect();
    let rows: Vec<Row> = std::thread::scope(|scope| {
        let handles: Vec<_> = tiles
            .iter()
            .map(|&key| {
                let (fine, coarse) = (&fine.1, &coarse.1);
                scope.spawn(move || {
                    let side = 512i64;
                    let (i0, j0) = (i64::from(key.tx) * side, i64::from(key.ty) * side);
                    let z16 = |i: i64, j: i64| {
                        metres(fine.fine_z(key.face, l2_rung, l2_rung, i as i32, j as i32))
                    };
                    let z128 = |i: i64, j: i64| {
                        metres(coarse.fine_z(key.face, l2_rung, l2_rung, i as i32, j as i32))
                    };
                    let (mut worst, mut at, mut ground, mut over30) = (0.0f64, (0, 0), 0.0, 0usize);
                    let (mut steep, mut steep_at) = (0.0f64, (0, 0));
                    let mut sum = 0.0;
                    let mut n = 0usize;
                    let mut links = [0usize; 5];
                    let bounds = [35f64, 45.0, 63.0].map(|d| d.to_radians().tan() * 16.0);
                    for j in j0..j0 + side {
                        for i in i0..i0 + side {
                            let (Some(a), Some(b)) = (z16(i, j), z128(i, j)) else {
                                continue;
                            };
                            let d = (a - b).abs();
                            sum += d;
                            n += 1;
                            if d > 30.0 {
                                over30 += 1;
                            }
                            if d > worst {
                                (worst, at, ground) = (d, (i - i0, j - j0), b);
                            }
                            for (di, dj) in [(1, 0), (0, 1)] {
                                if let Some(c) = z16(i + di, j + dj) {
                                    let s = (c - a).abs();
                                    links[0] += 1;
                                    for (k, &b) in bounds.iter().enumerate() {
                                        links[k + 1] += usize::from(s > b);
                                    }
                                    let (u, v) = (i - i0, j - j0);
                                    let rim = u < 8 || v < 8 || u >= side - 9 || v >= side - 9;
                                    links[4] += usize::from(s > bounds[2] && rim);
                                    if s > steep {
                                        (steep, steep_at) = (s, (i - i0, j - j0));
                                    }
                                }
                            }
                        }
                    }
                    Row {
                        key,
                        worst,
                        at,
                        ground,
                        over30,
                        steep,
                        steep_at,
                        mean: if n > 0 { sum / n as f64 } else { 0.0 },
                        links,
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a tile"))
            .collect()
    });
    let mut rows = rows;
    rows.sort_by(|a, b| b.worst.total_cmp(&a.worst));
    println!("the sea {} m; {} 16 m tiles", artifact_sea_m, rows.len());
    let mut maxes: Vec<f64> = rows.iter().map(|r| r.worst).collect();
    maxes.sort_by(f64::total_cmp);
    let q = |p: f64| maxes[((maxes.len() as f64 - 1.0) * p) as usize];
    println!(
        "a tile's largest detail: median {:.1} m, 90 % {:.1} m, 99 % {:.1} m, largest {:.1} m; tiles over 30 m: {}, over 100 m: {}",
        q(0.5),
        q(0.9),
        q(0.99),
        maxes[maxes.len() - 1],
        maxes.iter().filter(|&&m| m > 30.0).count(),
        maxes.iter().filter(|&&m| m > 100.0).count()
    );
    let tot = rows.iter().fold([0usize; 5], |mut t, r| {
        for (a, b) in t.iter_mut().zip(r.links) {
            *a += b;
        }
        t
    });
    let pc = |k: usize| 100.0 * tot[k] as f64 / tot[0].max(1) as f64;
    println!(
        "the 16 m links: {} — over 35° {:.2} %, 45° {:.3} %, 63° {:.4} % ({} links, {} of them within 8 cells of a tile's rim)",
        tot[0],
        pc(1),
        pc(2),
        pc(3),
        tot[3],
        tot[4]
    );
    println!(
        "face  tx    ty     largest   at (i, j)     ground    cells>30m  mean|d|  steepest 16 m link  at (i, j)"
    );
    for r in rows.iter().take(25) {
        println!(
            "{:?} {:>5} {:>5}  {:>9.1}  ({:>3}, {:>3})  {:>9.1}  {:>9}  {:>7.2}  {:>9.1}  ({:>3}, {:>3})",
            r.key.face,
            r.key.tx,
            r.key.ty,
            r.worst,
            r.at.0,
            r.at.1,
            r.ground,
            r.over30,
            r.mean,
            r.steep,
            r.steep_at.0,
            r.steep_at.1
        );
    }
}
