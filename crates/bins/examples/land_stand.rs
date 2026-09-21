//! ★ A STAND OVER LAND (the far-view ship's flights, 2026-09-20): solve the home planet, find the
//! land node nearest a stated direction whose eight neighbours are land too, and print the
//! per-account spawn entry for a stand `height_m` over it with the nose straight down (the picture
//! gate's nadir stand), plus the berth arguments for a hull forty metres beside it. A picture of the
//! land has to be taken over land, and a direction guessed from a globe's picture lands on the sea.
//! `cargo run --release -p vd-bins --example land_stand -- [height_m] [dx dy dz]`

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_core::glam::{DMat3, DQuat, DVec3};
use vd_terrain::artifact::DRY_M;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let height_m: f64 = args
        .first()
        .map_or(410_000.0, |s| s.parse().expect("the height in metres"));
    let aim = if args.len() >= 4 {
        DVec3::new(
            args[1].parse().expect("dx"),
            args[2].parse().expect("dy"),
            args[3].parse().expect("dz"),
        )
        .normalize()
    } else {
        DVec3::new(0.0, -1.0, 0.0)
    };
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("the home planet has a macro lattice");
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let sea = artifact.sea().unwrap_or(0);
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let land = |node: u32| {
        let r = artifact.rows[node as usize];
        r.water_m == DRY_M && i32::from(r.z_m) > sea
    };
    let dir_of = |node: u32| {
        let d = lattice.direction(node);
        DVec3::new(
            d[0].raw() as f64 / unit,
            d[1].raw() as f64 / unit,
            d[2].raw() as f64 / unit,
        )
        .normalize()
    };
    let mut best: Option<(f64, u32)> = None;
    for node in 0..lattice.node_count() as u32 {
        if !land(node)
            || !lattice
                .neighbours(node)
                .iter()
                .all(|&m| m == NO_NODE || land(m))
        {
            continue;
        }
        let dot = dir_of(node).dot(aim);
        if best.is_none_or(|(b, _)| dot > b) {
            best = Some((dot, node));
        }
    }
    let (dot, node) = best.expect("the home planet has land");
    let d = dir_of(node);
    let row = artifact.rows[node as usize];
    // The nadir stand: the nose down the radial; any direction across it serves as up.
    let forward = -d;
    let seed = if d.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
    let right = forward.cross(seed).normalize();
    let up = right.cross(forward);
    let orient = DQuat::from_mat3(&DMat3::from_cols(right, up, -forward)).normalize();
    let pos = d * (body.radius_m() + height_m);
    let berth = pos + right * 40.0;
    println!(
        "land_stand: node {node} at {:.1}° off the aim, {} m over the ladder radius (sea {sea} m), direction {:.6}, {:.6}, {:.6}",
        dot.clamp(-1.0, 1.0).acos().to_degrees(),
        row.z_m,
        d.x,
        d.y,
        d.z
    );
    println!(
        "VD_SPAWN_POSES='1000=Planet({HOME_PLANET_SEED}):{},{},{}@{},{},{},{}'",
        pos.x, pos.y, pos.z, orient.x, orient.y, orient.z, orient.w
    );
    println!(
        "--berth-x-m {} --berth-y-m {} --berth-z-m {}",
        berth.x, berth.y, berth.z
    );
}
