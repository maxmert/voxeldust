//! ★ A HULL BESIDE A WALKER ON THE STORED FINE GROUND (2026-10-14, the owner: "including the hull placement"): around a
//! spawn, the hull's place 40–80 m away in 16 directions whose highest ground under its 40 × 12 m footprint stands lowest
//! over the walker's own ground — the hull broadside to the walker — read on the 16 m ground the home planet's store holds
//! (nothing solved). Prints the `BERTH` arguments for `vd-build-ship`, the hull's centre its half height and the stand's
//! metre over that highest ground.
//!
//! **Example.** On the valley floor at (NegY, 19, 65): the hull 40 m to one side on the floor, not 25 m up the slope.
//!
//! `VD_REALM_STORE=<a copy of the home planet's store> cargo run --release -p vd-bins --example valley_hull -- X,Y,Z [dry]`
//! (`dry`: the walker first moved to the nearest dry, gentle place within 8 km, and its eye printed).

use std::collections::BTreeSet;
use std::sync::Arc;

use vd_core::glam::{DMat3, DQuat, DVec3};
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_sim::stub::built_store::{FineAt, FineForm, read_fine_record};
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::ZField;
use vd_terrain::fine::l1::TileKey;
use vd_terrain::fine::level::FineLevel;
use vd_terrain::fine::read::{FineBook, FineField};
use vd_terrain::fine::tile::{decode_levels, input_word_of};

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;
const HOME: u64 = 4_030_111_653_607_004_909;
/// The default hull's half sizes (`vd-build-ship`): across, up, along.
const HALF: [f64; 3] = [6.0, 3.0, 20.0];

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
    let at = DVec3::new(eye[0], eye[1], eye[2]);
    let mut up = at.normalize();
    let drawn = artifact.drawn_range_m(&lattice);
    let sea_r = body.radius_m() + f64::from(artifact.sea_m);
    let body = body.with_sea_m(artifact.sea());
    let book = |level: FineLevel| {
        let mut book =
            FineBook::new(level, &body, &lattice, artifact.sea_m).expect("the level applies");
        let input = input_word_of(level, artifact.digest());
        for ((f, tx, ty), _) in vd_terrain::artifact::fine_needs_of_head(
            level,
            &body,
            (drawn.0, drawn.1, artifact.sea_m),
            up.to_array(),
            at.length(),
            vd_core::geometry::drawable_theta_min_rad(),
        ) {
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
                book.insert(key, levels);
            }
        }
        println!("{level:?}: {} stored tiles held", book.len());
        Arc::new(book)
    };
    let (l1, l2) = (book(FineLevel::L1), book(FineLevel::L2));
    let inner: Arc<dyn ZField + Send + Sync> = Arc::new(artifact);
    let field = FineField {
        inner,
        book: l1,
        finer: Some(l2),
    };
    let ground = |d: DVec3| {
        vd_terrain::height::height_field_m(&body, &field, d.normalize().to_array(), 0)
            .expect("the ground is held")
    };
    let mut own = ground(up);
    println!(
        "the walker's ground {own:.2} m from the centre ({:.1} m over the sea); the spawn {:.2} m over it",
        own - sea_r,
        at.length() - own
    );
    // ★ `dry`: the walker moved to the nearest place within 8 km whose stored floor stands 20 m or more over the sea and
    // slopes under 10° over 16 m (rings every 100 m, 32 ways), the eye 1.8 m over that floor.
    if std::env::args().nth(2).as_deref() == Some("dry") {
        let east = DVec3::Z.cross(up).normalize();
        let north = up.cross(east);
        let gentle = |d: DVec3, h: f64| {
            let e = DVec3::Z.cross(d).normalize();
            let n = d.cross(e);
            [e, -e, n, -n]
                .iter()
                .all(|&w| (ground(d * h + w * 16.0) - h).abs() < 16.0 * 10f64.to_radians().tan())
        };
        for km in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
            let hs: Vec<f64> = (0..32)
                .map(|k| {
                    let a = f64::from(k) * std::f64::consts::TAU / 32.0;
                    ground(
                        (up * own + (east * a.cos() + north * a.sin()) * (km * 1_000.0))
                            .normalize(),
                    ) - sea_r
                })
                .collect();
            println!(
                "at {km} km the floor over the sea: lowest {:.0} m, highest {:.0} m, dry {} of 32",
                hs.iter().copied().fold(f64::MAX, f64::min),
                hs.iter().copied().fold(f64::MIN, f64::max),
                hs.iter().filter(|&&h| h > 0.0).count()
            );
        }
        let found = (1..=80).find_map(|ring| {
            (0..32).find_map(|k| {
                let a = f64::from(k) * std::f64::consts::TAU / 32.0;
                let d = (up * own + (east * a.cos() + north * a.sin()) * (f64::from(ring) * 100.0))
                    .normalize();
                let h = ground(d);
                (h - sea_r >= 20.0 && gentle(d, h)).then_some((d, h, ring))
            })
        });
        let (d, h, ring) = found.expect("a dry, gentle place within 8 km");
        up = d;
        own = h;
        let eye = d * (h + 1.8);
        println!(
            "DRY STAND {} m away: the floor {:.1} m over the sea; the eye {},{},{}",
            ring * 100,
            h - sea_r,
            eye.x,
            eye.y,
            eye.z
        );
    }
    // A tangent frame at the walker.
    let east = DVec3::Z.cross(up).normalize();
    let north = up.cross(east);
    let mut best: Option<(f64, DVec3, DVec3, f64)> = None;
    for dist in [40.0, 60.0, 80.0] {
        for k in 0..16 {
            let a = f64::from(k) * std::f64::consts::TAU / 16.0;
            let out = east * a.cos() + north * a.sin();
            let side = up.cross(out);
            let centre = up * own + out * dist;
            // The footprint: the hull's length across the line to the walker, its width along it, every 4 m.
            let mut high = f64::MIN;
            let (nl, nw) = ((2.0 * HALF[2] / 4.0) as i32, (2.0 * HALF[0] / 4.0) as i32);
            for i in 0..=nl {
                for j in 0..=nw {
                    let p = centre
                        + side * (f64::from(i) * 4.0 - HALF[2])
                        + out * (f64::from(j) * 4.0 - HALF[0]);
                    high = high.max(ground(p));
                }
            }
            let rise = high - own;
            if best.is_none_or(|b| rise < b.0) {
                best = Some((rise, centre, side, high));
            }
        }
    }
    let (rise, centre, side, high) = best.expect("a place");
    let hull_up = centre.normalize();
    let level = (side - hull_up * hull_up.dot(side)).normalize();
    let right = level.cross(hull_up).normalize();
    let orient = DQuat::from_mat3(&DMat3::from_cols(right, hull_up, -level)).normalize();
    let pos = hull_up * (high + 1.0 + HALF[1]);
    println!(
        "the hull {:.0} m from the walker, its highest ground {rise:.1} m over the walker's",
        (centre - up * own).length()
    );
    println!(
        "BERTH --berth-x-m {} --berth-y-m {} --berth-z-m {} --berth-facing-x {} --berth-facing-y {} --berth-facing-z {} --berth-facing-w {}",
        pos.x, pos.y, pos.z, orient.x, orient.y, orient.z, orient.w
    );
}
