//! A diagnostic (not a gate): ★ A STAND FOR THE FIRST SURFACE LOOK (ruling B75 item 4, 2026-10-10), lit by the star at the clock's start. The summer's freezing
//! line is lowest where the summer is coldest; on the home planet (a 55° tilt) that is the equator. This reads a cached
//! solve of the home planet (`vd_bins::artifact_file`), finds the highest solved ground within a band of the equator,
//! and prints a walker's spawn entry `eye_m` over its drawn ground, looking level along the ground's steepest fall, so a
//! headless picture shows the mountain's rock and its snow over the line.
//!
//! ```text
//! cargo run --release -p vd-bins --example surface_stand -- <home.art> [eye_m] [max_sin_lat]
//! ```
//!
//! **Example.** `surface_stand home.art 1500 0.12` prints the summit at 6.1 km over the ladder radius 3° south of the
//! equator, the snow line there, and a `VD_SPAWN_POSES='1000=Planet(…)'` line 1.5 km over that summit.

use vd_core::glam::{DMat3, DQuat, DVec3};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        !args.is_empty(),
        "usage: surface_stand <home.art> [eye_m] [max_sin_lat]"
    );
    let eye_m: f64 = args.get(1).map_or(1_500.0, |s| s.parse().expect("metres"));
    let band: f64 = args.get(2).map_or(0.12, |s| s.parse().expect("a sine"));
    let artifact = vd_bins::artifact_file::read(&args[0]).expect("the artifact file reads");
    let body = vd_bins::home_body(vd_bins::DEV.universe_seed).expect("the home planet");
    let wet = body.with_sea_m(artifact.sea());
    let drawn = artifact.drawn_range_m(&wet.macro_lattice().expect("a lattice"));
    let solved = vd_bins::solved::SolvedHome {
        body: wet.with_band_m(drawn.0, drawn.1).unwrap_or(wet),
        artifact,
    };
    let lattice = solved.body.macro_lattice().expect("a lattice");
    let one = f64::from(1u32 << 20) * f64::from(1u32 << 20);
    let dir_of = |node: u32| {
        let d = lattice.direction(node);
        DVec3::new(d[0].raw() as f64, d[1].raw() as f64, d[2].raw() as f64) / one
    };
    // THE STAR at the clock's start (the window stand's own rule): only ground it lights 20° up or more.
    let orbit = vd_bins::home_orbit(vd_bins::DEV.universe_seed).expect("the home planet's orbit");
    let sun = -vd_physics::celestial::orbital_state(&orbit, 0.0)
        .position
        .normalize();
    let lit = 20.0_f64.to_radians().sin();
    let (mut best, mut best_z) = (0u32, i16::MIN);
    for (node, row) in solved.artifact.rows.iter().enumerate() {
        let node = node as u32;
        let n = dir_of(node);
        if row.z_m > best_z && n.z.abs() <= band && n.normalize().dot(sun) >= lit {
            (best, best_z) = (node, row.z_m);
        }
    }
    let d = dir_of(best).normalize();
    let radius = solved.body.ladder().radius_m();
    let sea = solved.artifact.sea().map_or(f64::NAN, f64::from);
    let ground = solved.ground_m(d, 0);
    // The ground's steepest fall from the summit, over one node: the nose looks down it.
    let up = d;
    let east = DVec3::Z.cross(up).normalize();
    let north = up.cross(east);
    let step = lattice.node_m() / radius;
    let fall = (0..16)
        .map(|k| {
            let a = f64::from(k) * std::f64::consts::TAU / 16.0;
            let t = east * a.cos() + north * a.sin();
            (t, ground - solved.ground_m((up + t * step).normalize(), 0))
        })
        .fold((east, f64::MIN), |b, c| if c.1 > b.1 { c } else { b });
    let level = fall.0;
    let right = level.cross(up).normalize();
    let orient = DQuat::from_mat3(&DMat3::from_cols(right, up, -level)).normalize();
    let walker = up * (ground + eye_m);
    println!(
        "surface_stand: the highest solved node within |sin lat| ≤ {band} with the star {:.1}° up: {:.3}° N {:.3}° E, its row {best_z} m, the drawn \
         ground {:.1} m over the ladder radius ({:.1} m over the sea {sea} m); the ground falls {:.0} m over one node \
         ({:.0} m) along the nose",
        d.dot(sun).asin().to_degrees(),
        d.z.asin().to_degrees(),
        d.y.atan2(d.x).to_degrees(),
        ground - radius,
        ground - radius - sea,
        fall.1,
        lattice.node_m(),
    );
    println!(
        "VD_SPAWN_POSES='1000=Planet({}):{},{},{}@{},{},{},{}'",
        solved.body.seed(),
        walker.x,
        walker.y,
        walker.z,
        orient.x,
        orient.y,
        orient.z,
        orient.w
    );
}
