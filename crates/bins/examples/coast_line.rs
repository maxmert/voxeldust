//! ★ WHERE THE SHORE STANDS ON THE SCREEN, AND WHETHER FOUR ALTITUDES AGREE (2026-09-23; the
//! owner: *"when I'm flying away too far, it also changes, so I can't trust the far view and be
//! sure that the shape will not change when I will fly closer"*; ruling W16 fault E).
//!
//! **WHAT IT READS.** A NADIR screenshot — the nose straight down — of a coast. Along a stated
//! BEARING from the frame's centre it walks outward and finds the first crossing from WATER to
//! LAND, or the reverse. The two are told apart by the two materials the renderer really uses: the
//! sea is `srgb(0.06, 0.24, 0.42)`, blue, and the ground `srgb(0.55, 0.50, 0.42)`, tan, so a pixel
//! is WATER where its blue channel stands over its red one.
//!
//! **THE NUMBER IS IN METRES OF GROUND, never in pixels**, so four altitudes can be compared. A
//! pixel `r` from the centre stands at an angle `θ = atan(r · tan(fov/2) / (h/2))` off the nadir;
//! the ray from an eye at `D` from the body's centre meets the sphere of radius `R` at a central
//! angle `φ = π − θ − asin(D · sin θ / R)`, and the ground distance is `R · φ`. The fov is the
//! ladder's own (`2 · tan(22.5°) / rows` is `pixel_rad`, so the vertical field is 45°).
//!
//! **THE GATE (ruling W16 fault E).** The shore's ground distance must agree across the altitudes
//! to within ONE CELL of the coarser rung drawn there. Print the numbers; a reader compares them.
//!
//! ★ **AND IT READS AN OBLIQUE FRAME** (2026-09-23, ruling W17). Give it the nose's PITCH below the
//! local horizontal and it walks the frame's own vertical CENTRE COLUMN from the horizon down to
//! the bottom row instead of walking outward from the nadir: a pixel at row `y` stands
//! `θ = (90° − pitch) − atan((y − h/2) · tan 22.5° / (h/2))` off the nadir, and the same sine rule
//! turns that into GROUND METRES. A nadir stand over a coast NODE could not feed the instrument at
//! all (ruling W16 §5a: the shore stood inside the avatar's own ball); an oblique stand aimed along
//! the bearing to the nearest sea puts the shoreline on that very column.
//!
//! `cargo run --release -p vd-bins --example coast_line -- <shot.png> <altitude_m> [bearing_deg]
//! [radius_m] [pitch_deg]`

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = PathBuf::from(args.first().expect("a png to read"));
    let alt_m: f64 = args
        .get(1)
        .expect("the altitude in metres")
        .parse()
        .expect("a number");
    let bearing: f64 = args.get(2).map_or(0.0, |s| s.parse().expect("a number"));
    let radius_m: f64 = args
        .get(3)
        .map_or(6_341_800.0, |s| s.parse().expect("a number"));
    let img = image::open(&path).expect("a png").to_rgb8();
    let (w, h) = img.dimensions();
    let (cx, cy) = (f64::from(w) / 2.0, f64::from(h) / 2.0);
    // The vertical half-field the ladder itself states (`vd_core::geometry`'s pixel_rad).
    let tan_half = 22.5f64.to_radians().tan();
    let half_rows = f64::from(h) / 2.0;
    let eye = radius_m + alt_m;
    let ground_m = |r_px: f64| -> f64 {
        let theta = (r_px * tan_half / half_rows).atan();
        let s = (eye * theta.sin() / radius_m).clamp(-1.0, 1.0);
        // The FAR root of the sine rule is the near side of the sphere.
        let alpha = std::f64::consts::PI - s.asin();
        let phi = std::f64::consts::PI - theta - alpha;
        radius_m * phi.max(0.0)
    };
    let wet = |x: i64, y: i64| -> Option<bool> {
        if x < 0 || y < 0 || x >= i64::from(w) || y >= i64::from(h) {
            return None;
        }
        let p = img.get_pixel(x as u32, y as u32);
        Some(u32::from(p.0[2]) > u32::from(p.0[0]))
    };
    let pitch_deg: f64 = args.get(4).map_or(0.0, |s| s.parse().expect("a number"));
    if pitch_deg > 0.0 {
        // THE OBLIQUE WALK: down the frame's own vertical centre column, from the horizon row to
        // the bottom, in ground metres from the nadir.
        println!(
            "coast_line {}: {w}x{h}, the altitude {alt_m:.0} m, the nose {pitch_deg:.2}° below the local horizontal, the body's radius {radius_m:.0} m",
            path.display()
        );
        let ground_of_row = |row: f64| -> Option<f64> {
            let theta_v = ((row - f64::from(h) / 2.0) * tan_half / half_rows).atan();
            let theta = (90.0 - pitch_deg).to_radians() - theta_v;
            if theta <= 0.0 {
                return None;
            }
            let s = eye * theta.sin();
            if s > radius_m {
                return None;
            }
            let alpha = std::f64::consts::PI - (s / radius_m).asin();
            Some(radius_m * (std::f64::consts::PI - theta - alpha).max(0.0))
        };
        let mut previous: Option<bool> = None;
        let mut crossings = 0;
        let mut row = 0f64;
        while row < f64::from(h) {
            let Some(g) = ground_of_row(row) else {
                row += 1.0;
                previous = None;
                continue;
            };
            let Some(now) = wet(cx.round() as i64, row as i64) else {
                break;
            };
            if let Some(was) = previous
                && was != now
            {
                crossings += 1;
                println!(
                    "  crossing {crossings}: {} at row {row:.0} = {g:.0} m of ground from the nadir",
                    if now {
                        "land → water"
                    } else {
                        "water → land"
                    }
                );
                if crossings >= 8 {
                    break;
                }
            }
            previous = Some(now);
            row += 1.0;
        }
        if crossings == 0 {
            println!("  NO CROSSING down the centre column: the frame is all one side");
        }
        return;
    }
    let (dx, dy) = (bearing.to_radians().cos(), bearing.to_radians().sin());
    let limit = (f64::from(w).powi(2) + f64::from(h).powi(2)).sqrt() / 2.0;
    println!(
        "coast_line {}: {w}x{h}, the altitude {alt_m:.0} m, the bearing {bearing:.0}°, the body's radius {radius_m:.0} m",
        path.display()
    );
    // The HUD lives in the frame's top-left; a walk that starts at the centre and goes outward
    // never reads it unless the bearing points into it, which the caller states.
    let mut previous: Option<bool> = None;
    let mut crossings = 0;
    let mut r = 8.0f64;
    while r < limit {
        let x = (cx + dx * r).round() as i64;
        let y = (cy + dy * r).round() as i64;
        let Some(now) = wet(x, y) else { break };
        if let Some(was) = previous
            && was != now
        {
            crossings += 1;
            println!(
                "  crossing {crossings}: {} at {r:.0} px = {:.0} m of ground from the nadir",
                if now {
                    "land → water"
                } else {
                    "water → land"
                },
                ground_m(r)
            );
            if crossings >= 6 {
                break;
            }
        }
        previous = Some(now);
        r += 1.0;
    }
    if crossings == 0 {
        println!("  NO CROSSING along this bearing: the frame is all one side");
    }
}
