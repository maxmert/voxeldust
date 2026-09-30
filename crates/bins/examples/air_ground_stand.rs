//! A diagnostic (not a gate): THE AIR'S GROUND AGAINST THE SOLVED GROUND (2026-09-30; the owner, low over
//! the dry sea floor: "athmosphere is dissapeared"; ruling B25).
//!
//! Before ruling B25 the air pass drew the air as a shell over ONE sphere, the ladder's datum; an eye under
//! it was lifted onto it and a ray under the eye's level met it at once, so the far ground took no haze.
//! Since B25 the air's ground sphere is the SEA the realm states (`vd_client::sky::sky_terms`), and the
//! march reaches every ground under it. This tool puts an eye under that sphere, over the dry sea floor.
//!
//! It reads the home planet's artifact from its realm store (never solved here), prints the datum, the
//! solve's sea, the pyramid's lowest and highest node and the ladder's floor, finds the nearest sea floor
//! at least [`FLOOR_UNDER_M`] under the sea from the owner's stand where the star stands at least
//! [`MIN_SUN_DEG`] up (`vd_bins::air_stand`), and prints two spawn poses on that floor's radial, both
//! facing back toward the owner's stand [`PITCH_DEG`] under the level: the eye [`EYE_OVER_M`] over the
//! floor (under the air's ground), and the eye [`EYE_OVER_M`] over the sea (over it, the control).
//!
//! ```text
//! VD_REALM_STORE=<realm-planet-….redb> cargo run --release -p vd-bins --example air_ground_stand \
//!     -- [first_account] [sun_x sun_y sun_z]
//! ```
//!
//! **Example.** The floor 7 727 m under the sea lies 20 km from the stand: the first pose puts the eye
//! 5.7 km under the air's ground sphere, the second 2 km over it, on the same radial, facing the coast.

use vd_bins::air_stand::{FloorSearch, facing, nearest_floor};
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_terrain::home::home_planet;

/// The owner's stand on foot (the moving-eye gate's walkable stand), in the body frame, metres.
const OWN_STAND_M: [f64; 3] = [
    -5_938_198.569_863_499,
    -1_425_844.222_490_500_2,
    -1_724_752.453_824_647_7,
];
/// The home planet's realm (the recorded stand's `realm`).
const HOME_PLANET: u64 = 4_030_111_653_607_004_909;
/// The star's direction in the body frame at the recorded stand (the stamp's `star.direction_body`).
const SUN_DIR: [f64; 3] = [
    -0.029_593_808_877_100_215,
    -0.999_271_187_545_32,
    -0.024_110_168_350_959_74,
];
/// How far under the sea the chosen floor lies, at the least.
const FLOOR_UNDER_M: f64 = 4_000.0;
/// The eye's height over the floor (stand one) and over the sea (stand two).
const EYE_OVER_M: f64 = 2_000.0;
/// The least star elevation at the chosen floor, so the haze is lit.
const MIN_SUN_DEG: f64 = 10.0;
/// The facing's pitch under the level.
const PITCH_DEG: f64 = 2.0;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let first_account: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1002);
    let sun = match args.get(1..4) {
        Some(v) => DVec3::new(
            v[0].parse().expect("a number"),
            v[1].parse().expect("a number"),
            v[2].parse().expect("a number"),
        ),
        None => DVec3::from_array(SUN_DIR),
    }
    .normalize();

    let env = vd_bins::process_env();
    let store = vd_bins::open_realm_store(&env, RealmId::Planet(0))
        .expect("the store opens")
        .expect("VD_REALM_STORE names the home planet's file");
    let bare = home_planet();
    let lattice = bare.macro_lattice().expect("the home planet has a lattice");
    let artifact = vd_bins::artifact_store::read_artifact(
        &store,
        vd_terrain::artifact_world_tag(vd_bins::DEV.universe_seed),
        &lattice,
    )
    .expect("the stored artifact reads")
    .expect("the store holds an artifact");
    let body = bare.with_sea_m(artifact.sea());
    let datum = body.radius_m();
    let sea = body.sea_radius_m();
    let fine = artifact.pyramid.first();
    println!(
        "the ladder's datum {datum:.1} m; the solve's sea {:?} m over it; the finest pyramid level's \
         nodes from {:?} to {:?} m over it; the relief bound {:.0} m; the ladder's floor {} m ({:.0} m \
         under the datum)",
        artifact.sea(),
        fine.and_then(|l| l.iter().min()),
        fine.and_then(|l| l.iter().max()),
        body.relief_bound_m(0),
        body.ladder().floor_m,
        datum - f64::from(body.ladder().floor_m)
    );

    let own = DVec3::from_array(OWN_STAND_M).normalize();
    let ground = |d: DVec3| -> f64 {
        vd_terrain::height::height_field_m(&body, &artifact, d.to_array(), 0).expect("covered")
    };
    println!(
        "the owner's stand: the ground {:.0} m over the sea, the star {:.1}° up",
        ground(own) - sea,
        sun.dot(own).asin().to_degrees()
    );
    let search = FloorSearch {
        sphere_m: sea,
        depth_m: FLOOR_UNDER_M,
        min_sun_deg: MIN_SUN_DEG,
        bearings: 72,
        step_m: 2_000.0,
        reach_m: 3_000_000.0,
    };
    let Some(floor) = nearest_floor(&ground, own, sun, &search) else {
        println!("NO FLOOR {FLOOR_UNDER_M} m under the sea within the reach, lit");
        return;
    };
    println!(
        "the floor: {:.1} km from the stand, {:.0} m under the sea, the star {:.1}° up",
        floor.along_m / 1000.0,
        sea - floor.ground_m,
        sun.dot(floor.dir).asin().to_degrees()
    );
    let q = facing(floor.dir, floor.back, PITCH_DEG);
    let poses: Vec<String> = [floor.ground_m + EYE_OVER_M, sea + EYE_OVER_M]
        .iter()
        .zip(first_account..)
        .map(|(r, account)| {
            let e = floor.dir * *r;
            println!(
                "account {account}: the eye {:+.0} m over the air's ground sphere (the sea), {:.0} m \
                 over the floor",
                r - sea,
                r - floor.ground_m
            );
            format!(
                "{account}=Planet({HOME_PLANET}):{:?},{:?},{:?}@{:?},{:?},{:?},{:?}",
                e.x, e.y, e.z, q.x, q.y, q.z, q.w
            )
        })
        .collect();
    println!("VD_SPAWN_POSES={}", poses.join(";"));

    // ★ THE INLAND VIEW (2026-09-30, the owner: "how the visibility of the far distance will be above
    // the sea level"): from the same coast stand, the bearing whose ground stays over the sea the
    // farthest, and a pose [`INLAND_EYE_M`] over the ground there looking along it, [`PITCH_DEG`] down.
    let east = DVec3::Z.cross(own).normalize();
    let mut best = (0.0_f64, east);
    for b in 0..72u32 {
        let tangent =
            vd_core::glam::DQuat::from_axis_angle(own, f64::from(b) * 5.0_f64.to_radians()) * east;
        let axis = own.cross(tangent).normalize();
        let mut run_m = 0.0;
        while run_m < INLAND_REACH_M {
            let d = (vd_core::glam::DQuat::from_axis_angle(axis, (run_m + 2_000.0) / sea) * own)
                .normalize();
            if ground(d) <= sea {
                break;
            }
            run_m += 2_000.0;
        }
        if run_m > best.0 {
            best = (run_m, tangent);
        }
    }
    let (run_m, inland) = best;
    let q = facing(own, inland, PITCH_DEG);
    let e = own * (ground(own) + INLAND_EYE_M);
    println!(
        "the inland view: land for {:.0} km along the best bearing; the eye {:.0} m over the ground, \
         {:.0} m over the sea",
        run_m / 1000.0,
        INLAND_EYE_M,
        ground(own) + INLAND_EYE_M - sea
    );
    println!(
        "VD_INLAND_POSE={}=Planet({HOME_PLANET}):{:?},{:?},{:?}@{:?},{:?},{:?},{:?}",
        first_account + 2,
        e.x,
        e.y,
        e.z,
        q.x,
        q.y,
        q.z,
        q.w
    );
}

/// The inland view's eye over the coast's ground, metres: a hill's height (the picture gate's hill).
const INLAND_EYE_M: f64 = 300.0;
/// How far the inland view's land run is read, metres.
const INLAND_REACH_M: f64 = 400_000.0;
