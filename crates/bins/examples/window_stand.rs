//! ★ THE OWNER'S WINDOW STAND ON THE SHARD'S OWN SOLVED GROUND (ruling B63 item 3: the window preview of generator 26).
//! Reads the home planet's artifact from its realm store — the shard's own solve, refused by name if it was solved under
//! another key (`vd_bins::artifact_store::read_artifact`) — writes it as a bench artifact file, finds the walkable stand
//! where the star stands 15° up at the clock's start by the moving-eye gate's own rule (`vd_bins::solved::walkable_stand`),
//! and prints the walker's spawn entry and the `vd-build-ship` arguments for a test hull berthed upright 40 m ahead of the
//! walker, its belly a metre over the highest ground under its footprint. Runs while the cluster is DOWN (a realm's file
//! is held by its shard alone).
//!
//! `cargo run --release -p vd-bins --example window_stand -- <slot work dir> <artifact out file> [hull seq]`
//!
//! **Example.** After the home planet's first solve lands, `window_stand $SLOT $S/home.art 9` prints a
//! `VD_SPAWN_POSES='1000=Planet(…):…'` line for the gateway and the berth's eight numbers for hull 9; the owner logs in
//! on dry land with the star over the shoulder and the hull standing 40 m ahead.

use vd_core::glam::{DMat3, DQuat, DVec3};
use vd_core::pose::RealmId;

/// The walker's eye over the ground, metres (the moving-eye gate's).
const EYE_HEIGHT_M: f64 = 1.8;
/// The star's height over the stand, degrees (the owner's ruling B19: "move the stand to a 15° star").
const SUN_ELEVATION_DEG: f64 = 15.0;
/// The star's azimuth off the walker's nose, degrees (the gate's: over the shoulder).
const SUN_OFF_NOSE_DEG: f64 = 120.0;
/// The hull's centre ahead of the walker, metres (the gate's berth stand-off).
const BERTH_STANDOFF_M: f64 = 40.0;
/// The test hull's half size along its up, metres (`vd-build-ship`'s default 12 × 6 × 40 m box), and its belly's
/// clearance over the highest ground under it.
const HULL_HALF_UP_M: f64 = 3.0;
const HULL_CLEARANCE_M: f64 = 1.0;
/// The hull's half length and half width, metres (the same box), for the ground under its footprint.
const HULL_HALF_LENGTH_M: f64 = 20.0;
const HULL_HALF_WIDTH_M: f64 = 6.0;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        args.len() >= 2,
        "usage: window_stand <slot work dir> <artifact out file> [hull seq]"
    );
    let work = std::path::PathBuf::from(&args[0]);
    let out = &args[1];
    let seq: u64 = args.get(2).map_or(9, |s| s.parse().expect("a hull serial"));
    let universe_seed = vd_bins::DEV.universe_seed;
    let body = vd_bins::home_body(universe_seed).expect("the home planet");
    let planet = RealmId::Planet(body.seed());
    let dir = vd_bins::realm_store_dir(&work);
    let path = vd_bins::realm_store_path(&dir, planet);
    let stamp = vd_bins::durable_stamp(
        &vd_bins::process_env(),
        vd_core::store_stamp::StoreRole::RealmStore,
        vd_core::EpochId(0),
    )
    .expect("the store's label");
    let (store, _) = vd_io_prod::store::RedbStore::open(
        std::path::Path::new(&path),
        vd_io_prod::store::StoreTuning::default(),
        stamp,
    )
    .expect("the home planet's realm store opens (is the cluster down?)");
    let lattice = body.macro_lattice().expect("a lattice");
    let artifact = vd_bins::artifact_store::read_artifact(
        &store,
        vd_terrain::artifact_world_tag(universe_seed),
        &lattice,
    )
    .expect("the stored artifact reads")
    .expect("the home planet's shard has solved and stored its artifact");
    drop(store);
    let digest = artifact.digest();
    println!(
        "window_stand: the home planet's stored artifact, digest [{:#x}, {:#x}], sea {:?} m",
        digest[0],
        digest[1],
        artifact.sea()
    );
    vd_bins::artifact_file::write(out, &artifact).expect("the artifact file is written");
    println!("window_stand: written to {out}");
    // THE BODY THE CLIENT DRAWS (ruling B55), as `vd_bins::solved::solve_home` makes it.
    let wet = body.with_sea_m(artifact.sea());
    let drawn = artifact.drawn_range_m(&wet.macro_lattice().expect("a lattice"));
    let solved = vd_bins::solved::SolvedHome {
        body: wet.with_band_m(drawn.0, drawn.1).unwrap_or(wet),
        artifact,
    };
    // THE STAND: the gate's own rule — the circle where the star stands 15° up at the clock's start, the first
    // walkable dry land along it.
    let orbit = vd_bins::home_orbit(universe_seed).expect("the home planet's orbit");
    let sun = -vd_physics::celestial::orbital_state(&orbit, 0.0)
        .position
        .normalize();
    let along = sun.cross(DVec3::Z).normalize();
    let zenith = (90.0_f64 - SUN_ELEVATION_DEG).to_radians();
    let d0 = (sun * zenith.cos() + along * zenith.sin()).normalize();
    let (d, level, h) = vd_bins::solved::walkable_stand(
        &solved,
        sun,
        d0,
        BERTH_STANDOFF_M + HULL_HALF_LENGTH_M,
        EYE_HEIGHT_M,
        SUN_OFF_NOSE_DEG,
        "window_stand",
    );
    let up = (d - level * level.dot(d)).normalize();
    let right = level.cross(up).normalize();
    let orient = DQuat::from_mat3(&DMat3::from_cols(right, up, -level)).normalize();
    let walker = d * (h + EYE_HEIGHT_M);
    // THE BERTH: 40 m ahead along the walker's nose, its belly a metre over the highest ground under the box.
    let centre_dir = (d * h + level * BERTH_STANDOFF_M).normalize();
    let mut highest = f64::MIN;
    for a in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        for b in [-1.0, 0.0, 1.0] {
            let q = (d * h
                + level * (BERTH_STANDOFF_M + a * HULL_HALF_LENGTH_M)
                + right * (b * HULL_HALF_WIDTH_M))
                .normalize();
            highest = highest.max(solved.ground_m(q, 0));
        }
    }
    let berth = centre_dir * (highest + HULL_HALF_UP_M + HULL_CLEARANCE_M);
    println!(
        "window_stand: the stand {:.4}° N {:.4}° E (planet frame), the ground {:.1} m over the radius {:.1} m (the sea {:.1} m); \
         the berth's ground up to {:.1} m",
        d.z.asin().to_degrees(),
        d.y.atan2(d.x).to_degrees(),
        h - solved.body.radius_m(),
        solved.body.radius_m(),
        solved.body.sea_radius_m() - solved.body.radius_m(),
        highest - solved.body.radius_m()
    );
    println!(
        "VD_SPAWN_POSES='1000=Planet({}):{},{},{}@{},{},{},{}'",
        body.seed(),
        walker.x,
        walker.y,
        walker.z,
        orient.x,
        orient.y,
        orient.z,
        orient.w
    );
    let ship = RealmId::Ship(vd_core::ids::EntityId::pack(
        vd_core::entity_kind::EntityKind::Ship,
        1,
        seq,
        0,
    ));
    println!(
        "vd-build-ship --parent-store {} --ship-store {} --owner 1000 --seq {seq} \
         --max-push-micro-mps2 9000000000000000000 --max-turn-micro-radps2 3000000 \
         --berth-x-m {} --berth-y-m {} --berth-z-m {} \
         --berth-facing-x {} --berth-facing-y {} --berth-facing-z {} --berth-facing-w {}",
        path,
        vd_bins::realm_store_path(&dir, ship),
        berth.x,
        berth.y,
        berth.z,
        orient.x,
        orient.y,
        orient.z,
        orient.w
    );
}
