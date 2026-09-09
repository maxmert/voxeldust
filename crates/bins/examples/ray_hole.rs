//! RAY THROUGH A HOLE (slice 8 step 3, a diagnostic): rebuild the ground stand of `terrain_pictures`,
//! shoot the pilot camera's ray through a named pixel, march it against the recipe, and ask the
//! ladder view whether the column it lands in is wanted — and if not, why (the reach, the horizon,
//! the peak against the sightline). MEASURED need: 47 pixels of nothing under drawn ground at rows
//! 249–254 of the ground picture survived every crossfade; the probe's neighbours say rung 5 at
//! 13–15 km above and below them.
//!
//! `cargo run --release -p vd-bins --example ray_hole -- 700 251`

use vd_client::chunks::eye_surface;
use vd_client::ladder_view::{
    Column, LadderView, column_under, horizon_m, relief_m, rung_for_distance,
};
use vd_core::glam::DVec3;
use vd_terrain::Gf;

const SUN_ELEVATION_DEG: f64 = 15.0;
const SUN_OFF_NOSE_DEG: f64 = 120.0;
const W: usize = 1284;
const H: usize = 720;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let px: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(700);
    let py: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(251);
    // The stand: the eye's height over the recipe and the nose's tilt below level (the ground
    // stand's 1.8 m and 8° by default; the hill's are 300 m and 15°).
    let eye_height_m: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.8);
    let ground_tilt_deg: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8.0);
    let body = vd_bins::home_body(vd_bins::DEV.universe_seed).expect("the home planet");
    let orbit = vd_bins::home_orbit(vd_bins::DEV.universe_seed).expect("the orbit");
    let sun = -vd_physics::celestial::orbital_state(&orbit, 0.0)
        .position
        .normalize();
    let along = sun.cross(DVec3::Z).normalize();
    let zenith = (90.0_f64 - SUN_ELEVATION_DEG).to_radians();
    let d = (sun * zenith.cos() + along * zenith.sin()).normalize();
    let dir = [Gf::from_f64(d.x), Gf::from_f64(d.y), Gf::from_f64(d.z)];
    let h = vd_terrain::height::height_m(&body, dir, 0).to_f64();
    let toward_sun = (sun - d * sun.dot(d)).normalize();
    let ahead =
        vd_core::glam::DQuat::from_axis_angle(d, SUN_OFF_NOSE_DEG.to_radians()) * toward_sun;
    let t = ground_tilt_deg.to_radians();
    let nose = (ahead * t.cos() - d * t.sin()).normalize();
    let up = (d - nose * nose.dot(d)).normalize();
    let right = nose.cross(up).normalize();
    let basis = vd_core::glam::DMat3::from_cols(right, up, -nose);
    let orient = vd_core::glam::DQuat::from_mat3(&basis).normalize();
    let stand = d * (h + eye_height_m);
    // The pilot camera: the eye 1.6 m up the avatar's own up, looking along its nose (the same
    // expression `pilot_capture_camera` states), at the reference view's field.
    let cam_up = orient * DVec3::Y;
    let eye = stand + cam_up * 1.6;
    let fov_y = vd_core::geometry::REFERENCE_VIEW_FOV_Y_RAD;
    let ground = eye_surface(&body, eye.to_array()).expect("the eye's ground");
    let horizon = horizon_m(ground.surface_m, ground.altitude_m);
    println!(
        "eye {:.1} m over the recipe, horizon {horizon:.0} m, relief bound {:.0} m",
        ground.altitude_m,
        relief_m(&body)
    );
    // The ray through the pixel's centre.
    let forward = orient * DVec3::NEG_Z;
    let u = cam_up;
    let r = forward.cross(u).normalize();
    let tan_half = (fov_y * 0.5).tan();
    let aspect = W as f64 / H as f64;
    let ndc_x = (px as f64 + 0.5) / W as f64 * 2.0 - 1.0;
    let ndc_y = 1.0 - (py as f64 + 0.5) / H as f64 * 2.0;
    let ray = (forward + r * (ndc_x * tan_half * aspect) + u * (ndc_y * tan_half)).normalize();
    let radial = eye.normalize();
    println!(
        "pixel ({px}, {py}): the ray dips {:.3}° below level",
        (-ray.dot(radial)).asin().to_degrees()
    );
    let mut view = LadderView::default();
    let wanted = view.wanted(&body, eye.to_array());
    let wanted_columns: std::collections::BTreeSet<Column> =
        wanted.keys.iter().map(|k| Column::of(*k)).collect();
    // March the ray against the recipe at rungs 0, 5 and 6: where does it land?
    for rung in [0u8, 5, 6] {
        let cell = f64::from(vd_seed::ladder::cell_m(rung));
        let mut t = cell;
        let mut hit: Option<(f64, DVec3)> = None;
        while t < 40_000.0 {
            let p = eye + ray * t;
            let len = p.length();
            let pd = p / len;
            let hh = vd_terrain::height::height_m(
                &body,
                [Gf::from_f64(pd.x), Gf::from_f64(pd.y), Gf::from_f64(pd.z)],
                rung,
            )
            .to_f64();
            if len <= hh {
                hit = Some((t, pd));
                break;
            }
            t += cell.max(1.0);
        }
        match hit {
            None => println!("rung {rung}: the ray meets no ground within 40 km"),
            Some((t, pd)) => {
                let rule = rung_for_distance(t, body.ladder().rungs);
                println!(
                    "rung {rung} field: the ray meets ground at {t:.0} m (the rule's rung there: {rule})"
                );
                let mut l = 0u8;
                while l < body.ladder().rungs {
                    let col = column_under(&body, pd.to_array(), l);
                    if wanted_columns.contains(&col) {
                        println!("   wanted at rung {l}: {col:?}");
                    }
                    l += 1;
                }
                let col = column_under(&body, pd.to_array(), rule);
                let span =
                    vd_terrain::digest::surface_column(&body, col.face, col.rung, col.x, col.y);
                println!(
                    "   the column at the rule's rung {col:?}: wanted {}, span {}..{}, peak {:.1} m over the eye's surface, horizon {}",
                    wanted_columns.contains(&col),
                    span.lo,
                    span.hi,
                    span.peak_m.to_f64() - ground.surface_m,
                    if t <= horizon { "inside" } else { "beyond" }
                );
            }
        }
    }
}
