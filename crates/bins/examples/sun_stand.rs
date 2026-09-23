//! ★ A COAST STAND UNDER A LOW SUN, AT FOUR ALTITUDES (2026-09-23, ruling W16 faults D and E).
//!
//! **WHY IT EXISTS.** `land_stand` picks the LAND node nearest an aim, with every neighbour land —
//! so it never stands on a coast, and its first run for this slice landed under a sun **81.7°**
//! up. At the subsolar point a nadir picture is flat-lit: `N · L` is one everywhere, no relief
//! shades, and the frame reads as one grey-brown wash with a contrast of about one per cent. A
//! picture like that can judge neither a ring's brightness nor a shoreline. The owner's own
//! screenshot was taken with the star **19.8°** up.
//!
//! **WHAT IT PICKS.** The COAST nodes — a dry node with a sea neighbour — and among them the one
//! whose elevation under the stated sun is nearest the stated target. Then it prints the NADIR
//! spawn pose (the nose straight down) at each altitude asked for, so one flight script can stand
//! at the same ground from 3 000 km and from 3 km.
//!
//! `cargo run --release -p vd-bins --example sun_stand -- <sun_x> <sun_y> <sun_z> [target_deg]
//! [alt_m ...]`

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_core::glam::{DMat3, DQuat, DVec3};
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::solve::{FACIES_LAKE, FACIES_SEA};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |k: usize, d: f64| -> f64 { a.get(k).and_then(|s| s.parse().ok()).unwrap_or(d) };
    let sun = DVec3::new(num(0, 0.0), num(1, -1.0), num(2, 0.0)).normalize();
    let target = num(3, 20.0).to_radians().sin();
    let alts: Vec<f64> = if a.len() > 4 {
        a[4..]
            .iter()
            .map(|s| s.parse().expect("an altitude"))
            .collect()
    } else {
        vec![3_000_000.0, 300_000.0, 30_000.0, 3_000.0]
    };
    let body = home_planet();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let facies: Vec<u8> = artifact
        .rows
        .iter()
        .map(|r| r.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
        .collect();
    let dir_of = |node: u32| -> DVec3 {
        let d = lattice.direction(node);
        DVec3::new(d[0].raw() as f64, d[1].raw() as f64, d[2].raw() as f64).normalize()
    };
    let mut best: Option<(f64, u32)> = None;
    for node in 0..artifact.node_count() as u32 {
        let f = facies[node as usize];
        if f & (FACIES_SEA | FACIES_LAKE) != 0 {
            continue;
        }
        let wet = lattice
            .neighbours(node)
            .iter()
            .any(|&m| m != u32::MAX && facies[m as usize] & FACIES_SEA != 0);
        if !wet {
            continue;
        }
        let elevation = dir_of(node).dot(sun);
        let miss = (elevation - target).abs();
        if best.is_none_or(|(b, _)| miss < b) {
            best = Some((miss, node));
        }
    }
    let (miss, node) = best.expect("the home planet has a coast");
    let d = dir_of(node);
    let row = artifact.rows[node as usize];
    // The nadir stand: the nose down the radial; any direction across it serves as up.
    let forward = -d;
    let seed = if d.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
    let right = forward.cross(seed).normalize();
    let up = right.cross(forward);
    let orient = DQuat::from_mat3(&DMat3::from_cols(right, up, -forward)).normalize();
    let ground = body.radius_m() + f64::from(row.z_m);
    println!(
        "sun_stand: coast node {node}, the ground {} m over the ladder radius (the sea {} m), the star {:.1}° up ({:.4} off the target), direction {:.6} {:.6} {:.6}",
        row.z_m,
        artifact.sea().unwrap_or(0),
        d.dot(sun).asin().to_degrees(),
        miss,
        d.x,
        d.y,
        d.z
    );
    for alt in alts {
        let p = d * (ground + alt);
        println!(
            "{alt}|1000=Planet({HOME_PLANET_SEED}):{},{},{}@{},{},{},{}",
            p.x, p.y, p.z, orient.x, orient.y, orient.z, orient.w
        );
    }
}
