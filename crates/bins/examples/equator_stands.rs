//! A diagnostic (not a gate): THE WALKER'S STANDS ON THE SOLVED EQUATOR (2026-09-27, the moving-eye
//! gate's stand). The gate's walker stands on the equator where the star is fifteen degrees up and
//! walks 157 m along a nose the sun's rule sets; the server holds its eye at one radius, so the
//! path must be dry and must not rise above the eye. This reads a cached solve of the home planet
//! (`vd_bins::artifact_file`) and scans the whole equator: where it is dry, how long the dry runs
//! are, and at each dry stand how far the ground along the path rises over the stand.
//!
//! ```text
//! cargo run --release -p vd-bins --example equator_stands -- <home.art> [path_m] [step_m]
//! ```
//!
//! **Example.** The sun's stand lies in the sea; the scan prints the nearest dry stand east and
//! west, the share of the equator that is dry, and how many dry stands keep a 157 m walk under a
//! 1.8 m eye.

use vd_core::glam::{DQuat, DVec3};
use vd_terrain::home::home_planet;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .expect("usage: equator_stands <home.art> [path_m] [step_m]");
    let path_m: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(157.0);
    let step_m: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500.0);
    let artifact = vd_bins::artifact_file::read(path).expect("a readable artifact file");
    println!(
        "artifact {path}: digest {:?}, sea {:?} m",
        artifact.digest(),
        artifact.sea()
    );
    let body = home_planet().with_sea_m(artifact.sea());
    let radius_m = body.radius_m();
    let orbit = vd_bins::home_orbit(vd_bins::DEV.universe_seed).expect("the home planet's orbit");
    let sun = -vd_physics::celestial::orbital_state(&orbit, 0.0)
        .position
        .normalize();
    let along = sun.cross(DVec3::Z).normalize();
    let zenith = (90.0_f64 - 15.0).to_radians();
    let d0 = (sun * zenith.cos() + along * zenith.sin()).normalize();
    let ground = |d: DVec3, rung: u8| -> f64 {
        vd_terrain::height::height_field_m(&body, &artifact, d.to_array(), rung).expect("covered")
    };
    let drawn = |d: DVec3, rung: u8| -> f64 {
        vd_terrain::height::drawn_height_field_m(&body, &artifact, d.to_array(), rung)
            .expect("covered")
    };
    let dry = |d: DVec3| drawn(d, 0) <= ground(d, 0);
    println!(
        "the sun's stand: {:.1}° up, drawn {:.0} m over the radius, ground {:.0} m, dry {}",
        sun.dot(d0).asin().to_degrees(),
        drawn(d0, 0) - radius_m,
        ground(d0, 0) - radius_m,
        dry(d0)
    );
    let circumference = std::f64::consts::TAU * radius_m;
    let steps = (circumference / step_m) as i64;
    let (mut dry_n, mut runs, mut run, mut longest) = (0u64, 0u64, 0u64, 0u64);
    let (mut pass_n, mut rises) = (0u64, Vec::new());
    let mut first_dry: Option<(i64, f64)> = None;
    let mut first_pass: Option<(i64, f64, f64)> = None;
    let samples = (path_m / 5.0).ceil() as u64;
    for k in 0..steps {
        // East for the first half of the circle, then west, so "nearest" reads both ways.
        let signed = if k < steps / 2 { k } else { k - steps };
        let phi = signed as f64 * step_m / radius_m;
        let d = (DQuat::from_rotation_z(phi) * d0).normalize();
        if dry(d) {
            dry_n += 1;
            run += 1;
            longest = longest.max(run);
            if first_dry.is_none_or(|(s, _)| signed.abs() < s.abs()) {
                first_dry = Some((signed, sun.dot(d).asin().to_degrees()));
            }
            let toward_sun = (sun - d * sun.dot(d)).normalize();
            let level = DQuat::from_axis_angle(d, 120.0_f64.to_radians()) * toward_sun;
            let h = drawn(d, 0);
            let eye = d * (h + 1.8);
            let mut rise = f64::NEG_INFINITY;
            let mut wet = false;
            for i in 0..=samples {
                let p = eye + level * (i as f64 * 5.0);
                let q = p.normalize();
                wet |= !dry(q);
                rise = rise.max(drawn(q, 0) - (p.length() - 1.8));
            }
            if !wet {
                rises.push(rise);
                if rise <= 1.3 {
                    pass_n += 1;
                    if first_pass.is_none_or(|(s, _, _)| signed.abs() < s.abs()) {
                        first_pass = Some((signed, sun.dot(d).asin().to_degrees(), rise));
                    }
                }
            }
        } else if run > 0 {
            runs += 1;
            run = 0;
        }
    }
    rises.sort_by(f64::total_cmp);
    let q = |f: f64| {
        rises
            .get(((rises.len() as f64 - 1.0) * f) as usize)
            .copied()
    };
    println!(
        "the equator at {step_m} m steps: {steps} stands, {dry_n} dry ({:.1} %), {runs} dry runs, \
         the longest {:.1} km",
        100.0 * dry_n as f64 / steps as f64,
        longest as f64 * step_m / 1000.0
    );
    println!(
        "the nearest dry stand: {:?} (steps signed east, the star's elevation there)",
        first_dry.map(|(s, e)| (s as f64 * step_m / 1000.0, e))
    );
    println!(
        "dry stands whose {path_m} m path stays dry: {}; the path's rise over the stand's ground \
         (quartiles, metres): {:?}",
        rises.len(),
        [q(0.0), q(0.25), q(0.5), q(0.75), q(1.0)]
    );
    println!(
        "dry stands whose path rises at most 1.3 m (under a 1.8 m eye with 0.5 m spare): {pass_n}; \
         the nearest {:?} (km signed east, the star's elevation, the rise)",
        first_pass.map(|(s, e, r)| (s as f64 * step_m / 1000.0, e, r))
    );
    // THE WALKER'S TEST at a stand: dry, and its path along the sun's nose dry and at most 1.3 m
    // over the stand's ground.
    let walkable = |d: DVec3| -> bool {
        if !dry(d) {
            return false;
        }
        let toward_sun = (sun - d * sun.dot(d)).normalize();
        let level = DQuat::from_axis_angle(d, 120.0_f64.to_radians()) * toward_sun;
        let eye = d * (drawn(d, 0) + 1.8);
        (0..=samples).all(|i| {
            let p = eye + level * (i as f64 * 5.0);
            let q = p.normalize();
            dry(q) && drawn(q, 0) <= p.length() - 0.5
        })
    };
    // THE CIRCLE WHERE THE STAR STANDS 15° UP: the sun's stand turned about the sun's axis.
    let ring_r = radius_m * zenith.sin();
    let ring_steps = (std::f64::consts::TAU * ring_r / step_m) as i64;
    let mut best: Option<(f64, f64, f64)> = None;
    let mut ring_dry = 0u64;
    for k in 0..ring_steps {
        let signed = if k < ring_steps / 2 {
            k
        } else {
            k - ring_steps
        };
        let psi = signed as f64 * step_m / ring_r;
        let d = (DQuat::from_axis_angle(sun, psi) * d0).normalize();
        ring_dry += u64::from(dry(d));
        if walkable(d) {
            let km = signed as f64 * step_m / 1000.0;
            if best.is_none_or(|(b, _, _)| km.abs() < b.abs()) {
                best = Some((km, d.z.asin().to_degrees(), sun.dot(d).asin().to_degrees()));
            }
        }
    }
    println!(
        "the 15° circle: {ring_steps} stands, {ring_dry} dry; the nearest walkable {:?} (km along the \
         circle from the sun's stand, its latitude, the star's elevation)",
        best
    );
    // THE BAND ONE DEGREE EITHER SIDE OF THE EQUATOR, on the day side (the star at least 5° up):
    // where a hull's −Z nose is within a degree of level.
    let mut near: Option<(f64, f64, f64, f64)> = None;
    let mut lowest: Option<(f64, f64, f64, f64)> = None;
    let mut band_dry = 0u64;
    for lat_i in -10..=10 {
        let lat = f64::from(lat_i) * 0.1_f64.to_radians();
        for k in 0..steps {
            let signed = if k < steps / 2 { k } else { k - steps };
            let phi = signed as f64 * step_m / radius_m;
            let e = (DQuat::from_rotation_z(phi) * d0).normalize();
            let axis = e.cross(DVec3::Z).normalize();
            let d = (DQuat::from_axis_angle(axis, lat) * e).normalize();
            if sun.dot(d).asin().to_degrees() < 5.0 {
                continue;
            }
            band_dry += u64::from(dry(d));
            if walkable(d) {
                let km = signed as f64 * step_m / 1000.0;
                let elev = sun.dot(d).asin().to_degrees();
                if lowest.is_none_or(|(_, _, e, _)| (elev - 15.0).abs() < (e - 15.0).abs()) {
                    lowest = Some((km, d.z.asin().to_degrees(), elev, drawn(d, 0) - radius_m));
                }
                if near.is_none_or(|(b, _, _, _)| km.abs() < b.abs()) {
                    near = Some((
                        km,
                        d.z.asin().to_degrees(),
                        sun.dot(d).asin().to_degrees(),
                        drawn(d, 0) - radius_m,
                    ));
                }
            }
        }
    }
    println!(
        "the day side within 1° of the equator: {band_dry} dry stands; the nearest walkable {:?} \
         (km east, latitude, the star's elevation, the ground over the radius)",
        near
    );
    println!("the walkable stand there whose star is nearest 15° up: {lowest:?}");
    // ★ THE CAP AROUND THE PLANET FRAME'S +Y (a planted hull's own up): where a hull stands nearly
    // upright, its nose (−Z) within the latitude of level. The nearest walkable day-side stands.
    let mut cap: Vec<(f64, f64, f64)> = Vec::new();
    let cap_step = step_m / radius_m;
    let n = (25.0_f64.to_radians() / cap_step) as i64;
    for i in -n..=n {
        for j in -n..=n {
            let (a, b) = (i as f64 * cap_step, j as f64 * cap_step);
            if (a * a + b * b).sqrt() > 25.0_f64.to_radians() {
                continue;
            }
            // Tilt +Y toward +X by a and toward +Z by b.
            let d = (DQuat::from_rotation_x(b) * DQuat::from_rotation_z(-a) * DVec3::Y).normalize();
            if sun.dot(d).asin().to_degrees() < 5.0 || !walkable(d) {
                continue;
            }
            cap.push((
                d.dot(DVec3::Y).clamp(-1.0, 1.0).acos().to_degrees(),
                d.z.asin().to_degrees(),
                sun.dot(d).asin().to_degrees(),
            ));
        }
    }
    cap.sort_by(|x, y| x.0.total_cmp(&y.0));
    println!(
        "the cap within 25° of +Y: {} walkable day-side stands; the nearest five (degrees from +Y, \
         latitude, the star's elevation): {:?}",
        cap.len(),
        cap.iter().take(5).collect::<Vec<_>>()
    );
}
