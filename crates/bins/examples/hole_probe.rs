//! ★ WHAT STANDS UNDER A PIXEL (the owner's coast flight, 2026-09-20): from a recorded frame's eye and
//! camera (the stamp's `eye_body_m` and `camera_body_xyzw`), cast the ray of a pixel to the home
//! planet's artifact surface and print, for every rung the picture draws, the chunk column under
//! the hit, the slices the ladder asks for that column (the field span and the recipe's), the slice
//! the ground actually stands in, and whether an asked slice holds it. A hole in the probe frame
//! (no rung drew the pixel) is explained here: a slice nobody asked, or a chunk that was asked.
//! `cargo run --release -p vd-bins --example hole_probe -- ex ey ez qx qy qz qw px py [fov_deg] [w] [h]`

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_core::glam::{DQuat, DVec3};
use vd_terrain::artifact::{PyramidField, ZField};
use vd_terrain::chunk::CHUNK_EDGE;
use vd_terrain::digest::{surface_column, surface_column_field};
use vd_terrain::height;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    assert!(a.len() >= 9, "ex ey ez qx qy qz qw px py [fov_deg] [w] [h]");
    let eye = DVec3::new(a[0], a[1], a[2]);
    let cam = DQuat::from_xyzw(a[3], a[4], a[5], a[6]).normalize();
    let (px, py) = (a[7], a[8]);
    let fov = a.get(9).copied().unwrap_or(45.0).to_radians();
    let w = a.get(10).copied().unwrap_or(1284.0);
    let h = a.get(11).copied().unwrap_or(720.0);
    let tan_half = (fov / 2.0).tan();
    let x_ndc = (px + 0.5) / w * 2.0 - 1.0;
    let y_ndc = 1.0 - (py + 0.5) / h * 2.0;
    let dir_cam = DVec3::new(x_ndc * tan_half * (w / h), y_ndc * tan_half, -1.0).normalize();
    let ray = (cam * dir_cam).normalize();
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("a macro lattice");
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let levels = artifact.pyramid.len() as u32;
    let sea_m = artifact.sea().unwrap_or(0);
    let floor = i64::from(body.ladder().floor_m);
    let edge = i64::from(CHUNK_EDGE as i32);
    let radius = body.radius_m();
    // March the ray to the finest drawn surface (rung 4 on the rows), then bisect to a metre.
    let ground = |p: DVec3| -> Option<f64> {
        let d = p.normalize();
        height::height_field_m(&body, &artifact, d.to_array(), 4)
    };
    let mut t = 0.0;
    let mut hit = None;
    while t < 2_000_000.0 {
        let p = eye + ray * t;
        let r = p.length();
        if r > radius + 20_000.0 && ray.dot(p) > 0.0 {
            break;
        }
        if let Some(g) = ground(p)
            && r <= g
        {
            let (mut lo, mut hi) = ((t - 50.0).max(0.0), t);
            while hi - lo > 0.5 {
                let m = (lo + hi) / 2.0;
                let pm = eye + ray * m;
                if pm.length() <= ground(pm).unwrap_or(f64::MAX) {
                    hi = m;
                } else {
                    lo = m;
                }
            }
            hit = Some(hi);
            break;
        }
        t += 50.0;
    }
    let Some(t) = hit else {
        println!("pixel ({px}, {py}): the ray meets no ground within 2 000 km");
        return;
    };
    let p = eye + ray * t;
    let d = p.normalize();
    let hit_m = p.length();
    println!(
        "pixel ({px}, {py}): ground {t:.0} m down the ray, radius {:.1} m over the ladder radius (sea {sea_m} m) — {}",
        hit_m - radius,
        if hit_m - radius > f64::from(sea_m) {
            "LAND"
        } else {
            "SEA FLOOR"
        }
    );
    let face = vd_seed::bend::face_of(d.to_array());
    let (fa, fb) = vd_seed::bend::face_coords(face, d.to_array());
    println!(
        "rung | column (face x y) | ground at rung m | slice | FIELD span | RECIPE span | field holds | recipe holds"
    );
    for rung in 4u8..=9 {
        let level = PyramidField::level_for(&lattice, levels, rung);
        let pyramid = PyramidField::of(&artifact, level);
        let field: &dyn ZField = match &pyramid {
            Some(l) => l,
            None => &artifact,
        };
        let n = body.ladder().cells_per_edge(rung);
        let i = i64::from(vd_seed::ladder::index_of(vd_seed::bend::unbend(fa), n));
        let j = i64::from(vd_seed::ladder::index_of(vd_seed::bend::unbend(fb), n));
        let (x, y) = ((i / edge) as i32, (j / edge) as i32);
        let at_rung = height::height_field_m(&body, field, d.to_array(), rung);
        let Some(at_rung) = at_rung else {
            println!("{rung:>4} | {face:?} {x} {y} | (no field row) |");
            continue;
        };
        let slice = ((at_rung.floor() as i64 - floor) >> rung) / edge;
        let fs = surface_column_field(&body, field, face, rung, x, y);
        let rs = surface_column(&body, face, rung, x, y);
        let fh = fs.is_some_and(|s| (i64::from(s.lo)..=i64::from(s.hi)).contains(&slice));
        let rh = (i64::from(rs.lo)..=i64::from(rs.hi)).contains(&slice);
        println!(
            "{rung:>4} | {face:?} {x} {y} | {:>10.1} | {slice:>5} | {} | {}..={} | {fh} | {rh}",
            at_rung - radius,
            fs.map_or("none".to_owned(), |s| format!("{}..={}", s.lo, s.hi)),
            rs.lo,
            rs.hi
        );
    }
}
