//! ★ THE RINGS' OWN BRIGHTNESS STEP, IN PIXELS (2026-09-23; the owner, from 3 300 km: *"even when
//! shape do not change from far view the moving rings are visible, because for some reason they
//! have different color"*; ruling W16 fault D).
//!
//! **WHAT IT READS.** A NADIR screenshot — the nose straight down — of the ground. The rings are
//! concentric about the point under the eye, which is the middle of the frame, so it reads the MEAN
//! LUMINANCE of concentric ANNULI about that middle. A rung boundary is a circle on screen, so a
//! rung that shades differently from its neighbour shows as a STEP between two neighbouring annuli.
//! The HUD's own rows at the top and the avatar's ball at the centre are drawings over the ground
//! and are left out by a stated count of rows and a stated radius.
//!
//! **THE NUMBER, AND WHAT PASSES.** For each pair of neighbouring annuli it prints the step as a
//! share of the local mean. The eye's own threshold for a luminance step across an edge is about
//! ONE PER CENT under ideal conditions and about two to three per cent in a real picture — Weber's
//! law as measured by Blackwell (1946) and restated for displays by Barten (1999, *Contrast
//! Sensitivity of the Human Eye*); the DICOM greyscale standard (PS 3.14) builds its whole
//! luminance ladder on a just-noticeable difference of about one per cent. The gate below is
//! **2 % of the local mean**, the common working figure, and it is stated here rather than drawn:
//! a step under it is not a ring a pilot can see.
//!
//! ★ **AND IT READS SECTORS, NOT WHOLE CIRCLES.** The ring's own mechanism TILTS the ground
//! radially, so on the sun-facing side of a ring the shade brightens and on the far side it
//! darkens by the same amount — and the mean of a whole annulus CANCELS it. MEASURED: over whole
//! circles the worst step in the 3 000 km nadir picture read 0.155 %, which says nothing. So each
//! annulus is cut into SECTORS about the centre and the step is read in each one; the number
//! reported is the worst over every sector of every pair, and the whole-circle mean is printed
//! beside it.
//!
//! **IT IS A MEASUREMENT, NOT A PROOF OF CAUSE.** A terminator, a coastline or a cloud also make a
//! radial step. So the instrument prints the WHOLE profile, and a ring reads as a step that stands
//! at the same screen radius in every picture of the same altitude.
//!
//! ★ **AND A HORIZON PICTURE IS READ IN ROWS, NOT IN RINGS** (`rows` as the last argument). A NADIR
//! frame of 45° spans only a narrow cone of ground — MEASURED: at a 300 km stand the whole frame
//! holds eye distances of 300 to 360 km, while the nearest residual annulus stands at 489 to
//! 513 km — so a nadir picture CANNOT HOLD A RUNG BOUNDARY AT ALL. The owner's own framing is
//! oblique: the ground runs from under the nose out to the limb, and a rung boundary is then a
//! roughly HORIZONTAL band across the frame. In `rows` mode the instrument reads the mean luminance
//! of each horizontal band of rows and the step between neighbours, by the same 2 % gate.
//!
//! ★ **AND IN `rows` MODE IT NAMES THE BOUNDARY** (2026-09-23, ruling W17). Give it the stand's
//! ALTITUDE and the nose's PITCH below the local horizontal and it turns every row band into the
//! EYE DISTANCE of the ground drawn there — the ray at `θ` off the nadir meets the body's sphere
//! at `E · cos θ − √(R² − E² sin²θ)` — and prints the RUNG the ladder draws at that distance
//! (`vd_client::ladder_view::rung_for_distance`) beside it. A band where the rung changes IS a
//! rung boundary, so a luminance step that stands on one is the ladder's, and a step anywhere else
//! is the landscape's. Nothing is guessed from the picture.
//!
//! `cargo run --release -p vd-bins --example ring_step -- <shot.png> [bands] [share] [hud_rows]
//! [ball_px] [sectors] [rings|rows] [alt_m] [pitch_deg] [radius_m]`

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = PathBuf::from(args.first().expect("a png to read"));
    let annuli: usize = args.get(1).map_or(40, |s| s.parse().expect("a count"));
    let gate: f64 = args.get(2).map_or(0.02, |s| s.parse().expect("a share"));
    // The HUD's own rows at the top of the frame, and the avatar ball at the centre: both are
    // drawings over the ground, and neither is a ring.
    let hud_rows: u32 = args.get(3).map_or(195, |s| s.parse().expect("a row count"));
    let ball_px: f64 = args.get(4).map_or(60.0, |s| s.parse().expect("a radius"));
    let sectors: usize = args.get(5).map_or(8, |s| s.parse().expect("a count"));
    let rows_mode = args.get(6).is_some_and(|m| m == "rows");
    // ★ THE STAND, for the row-to-rung reading: the eye's altitude and the nose's pitch below the
    // local horizontal. Absent, the bands are printed in pixels alone, as they always were.
    let alt_m: Option<f64> = args.get(7).map(|s| s.parse().expect("an altitude"));
    let pitch_deg: f64 = args.get(8).map_or(0.0, |s| s.parse().expect("an angle"));
    let radius_m: f64 = args
        .get(9)
        .map_or(6_341_800.0, |s| s.parse().expect("a radius"));
    let img = image::open(&path).expect("a png").to_rgb8();
    let (w, h) = img.dimensions();
    let lum = |p: &image::Rgb<u8>| {
        // Rec. 709 luma of the sRGB bytes: the number a viewer's eye weighs.
        0.2126 * f64::from(p.0[0]) + 0.7152 * f64::from(p.0[1]) + 0.0722 * f64::from(p.0[2])
    };
    // ★ WATER IS NOT LAND, AND THE SHADE OF THE TWO MAY NOT BE AVERAGED (2026-09-23, ruling W17).
    // The renderer draws the sea `srgb(0.06, 0.24, 0.42)` and the ground `srgb(0.55, 0.50, 0.42)`,
    // so a pixel is WATER where its blue channel stands over its red one — `coast_line`'s own
    // rule. MEASURED before this line: at the 150 km stand a bay filled the middle of the frame
    // and the trend on one side of the 9 -> 10 band was sea while the other was sand, so the
    // boundary read a gap of 84 %, which measures a coastline and not a ladder. The land's shade
    // and the WATER'S OWN SHARE are therefore read apart, and both are printed.
    let wet = |p: &image::Rgb<u8>| u32::from(p.0[2]) > u32::from(p.0[0]);
    // ★ THE ROW-TO-RUNG READING (ruling W17): the ray through a screen row, its meeting with the
    // body's sphere, and the rung the ladder draws at that distance. A row whose ray misses the
    // body is SKY, and the sky is left out of every number below.
    let home = vd_terrain::home::home_planet();
    let rungs = home.ladder().rungs;
    // ★ THE BANDS THE PICTURE REALLY DRAWS (ruling W17): the ladder's own bound, having read the
    // body AND the pyramid level count, so a rung whose handover step floors its switch is judged
    // where it is really drawn.
    let home_bound = {
        let lattice = home
            .macro_lattice()
            .expect("the home planet has a macro lattice");
        let mut levels = 0u32;
        while lattice.coarser(levels + 1).is_some() {
            levels += 1;
        }
        vd_client::ladder_view::AskBound::unbounded().for_body(&home, rungs, levels)
    };
    let tan_half = 22.5f64.to_radians().tan();
    let stand = |row: f64| -> Option<(f64, f64, u8)> {
        let alt = alt_m?;
        let eye = radius_m + alt;
        // The pixel's angle below the view axis, and the axis's own angle off the nadir.
        let theta_v = ((row - f64::from(h) / 2.0) * tan_half / (f64::from(h) / 2.0)).atan();
        let theta = (90.0 - pitch_deg).to_radians() - theta_v;
        if theta <= 0.0 {
            return None;
        }
        let s = eye * theta.sin();
        if s > radius_m {
            // The ray passes over the limb: sky, not ground.
            return None;
        }
        let slant = eye * theta.cos() - (radius_m * radius_m - s * s).sqrt();
        if slant <= 0.0 {
            return None;
        }
        let ground = radius_m
            * (std::f64::consts::PI - theta - (std::f64::consts::PI - (s / radius_m).asin()))
                .max(0.0);
        Some((
            slant,
            ground,
            vd_client::ladder_view::rung_for_distance(slant, rungs),
        ))
    };
    let on_ground = |row: u32| -> bool { alt_m.is_none() || stand(f64::from(row)).is_some() };
    // ★ THE CENTRE IS THE FRAME'S OWN, because the picture is a NADIR stand: the rings are
    // concentric about the point under the eye, which is the middle of the screen.
    let (cx, cy) = (f64::from(w) / 2.0, f64::from(h) / 2.0);
    let radius = f64::from(h) / 2.0;
    let mut sum = vec![0.0f64; annuli * sectors];
    let mut count = vec![0u64; annuli * sectors];
    // ★ THE SAME LUMINANCE BY SINGLE ROW (ruling W17), for the boundary reading below: a band is
    // too coarse to fit a trend through, and the trend is what separates a ring from the air.
    let mut row_sum = vec![0.0f64; h as usize * sectors];
    let mut row_count = vec![0u64; h as usize * sectors];
    let mut row_wet = vec![0u64; h as usize * sectors];
    let mut row_all = vec![0u64; h as usize * sectors];
    for (x, y, p) in img.enumerate_pixels() {
        if y < hud_rows || !on_ground(y) {
            continue;
        }
        let (dx, dy) = (f64::from(x) - cx, f64::from(y) - cy);
        let d = (dx * dx + dy * dy).sqrt();
        if d < ball_px {
            continue;
        }
        // RINGS: the band is the distance from the nadir, the sector the bearing about it.
        // ROWS: the band is the screen ROW, the sector the COLUMN — an oblique picture's rung
        // boundary is a horizontal band, and its own shade changes across the frame.
        let (k, sector) = if rows_mode {
            (
                (f64::from(y) / f64::from(h) * annuli as f64) as usize,
                ((f64::from(x) / f64::from(w) * sectors as f64) as usize).min(sectors - 1),
            )
        } else {
            let turn = dy.atan2(dx) / std::f64::consts::TAU + 0.5;
            (
                (d / radius * annuli as f64) as usize,
                ((turn * sectors as f64) as usize).min(sectors - 1),
            )
        };
        if k >= annuli {
            continue;
        }
        let column = ((f64::from(x) / f64::from(w) * sectors as f64) as usize).min(sectors - 1);
        row_all[y as usize * sectors + column] += 1;
        if wet(p) {
            row_wet[y as usize * sectors + column] += 1;
            continue;
        }
        sum[k * sectors + sector] += lum(p);
        count[k * sectors + sector] += 1;
        row_sum[y as usize * sectors + column] += lum(p);
        row_count[y as usize * sectors + column] += 1;
    }
    println!(
        "ring_step {}: {w}x{h}, {} bands, {sectors} sectors, the HUD's first {hud_rows} rows and the ball's {ball_px:.0} px left out",
        path.display(),
        if rows_mode {
            format!("{annuli} ROW")
        } else {
            format!("{annuli} RING (the nadir at ({cx:.0}, {cy:.0}), out to {radius:.0} px)")
        }
    );
    if alt_m.is_some() {
        println!(
            "the stand: the eye {:.0} m up, the nose {pitch_deg:.2}° below the local horizontal, the body's radius {radius_m:.0} m",
            alt_m.unwrap_or(0.0)
        );
    }
    println!(
        "band | px | eye distance m | ground m | rung | pixels | whole-band luma | whole-band step % | the WORST SECTOR's step % (of {sectors})"
    );
    let mut worst = (0usize, 0.0f64, 0usize);
    let mut over = 0usize;
    let mut previous: Option<Vec<Option<f64>>> = None;
    for k in 0..annuli {
        let means: Vec<Option<f64>> = (0..sectors)
            .map(|q| {
                let n = count[k * sectors + q];
                (n >= 64).then(|| sum[k * sectors + q] / n as f64)
            })
            .collect();
        let held: Vec<f64> = means.iter().flatten().copied().collect();
        if held.len() < sectors {
            previous = Some(means);
            continue;
        }
        let pixels: u64 = (0..sectors).map(|q| count[k * sectors + q]).sum();
        let whole = held.iter().sum::<f64>() / held.len() as f64;
        let (mut whole_step, mut sector_step, mut which) = (0.0f64, 0.0f64, 0usize);
        if let Some(before) = &previous
            && before.iter().all(Option::is_some)
        {
            let prev_whole =
                before.iter().flatten().sum::<f64>() / before.iter().flatten().count() as f64;
            whole_step = (whole - prev_whole).abs() / ((whole + prev_whole) / 2.0).max(1.0e-9);
            for q in 0..sectors {
                let (a, b) = (means[q].unwrap_or(0.0), before[q].unwrap_or(0.0));
                let step = (a - b).abs() / ((a + b) / 2.0).max(1.0e-9);
                if step > sector_step {
                    sector_step = step;
                    which = q;
                }
            }
        }
        if sector_step > worst.1 {
            worst = (k, sector_step, which);
        }
        if sector_step > gate {
            over += 1;
        }
        let at = if rows_mode {
            (k as f64 + 0.5) / annuli as f64 * f64::from(h)
        } else {
            (k as f64 + 0.5) / annuli as f64 * radius
        };
        let (slant, ground, rung) = stand(at).map_or_else(
            || ("-".to_owned(), "-".to_owned(), "-".to_owned()),
            |(s, g, r)| (format!("{s:.0}"), format!("{g:.0}"), format!("{r}")),
        );
        println!(
            "{k:>4} | {at:>4.0} | {slant:>14} | {ground:>8} | {rung:>4} | {:>6} | {whole:>15.2} | {:>17.3} | {:>12.3} (sector {which})",
            pixels,
            whole_step * 100.0,
            sector_step * 100.0
        );
        previous = Some(means);
    }
    let at = if rows_mode {
        (worst.0 as f64 + 0.5) / annuli as f64 * f64::from(h)
    } else {
        (worst.0 as f64 + 0.5) / annuli as f64 * radius
    };
    println!(
        "the worst SECTOR step is {:.3} % at band {} ({at:.0} px, sector {}); {over} of {annuli} bands step past the stated {:.1} %",
        worst.1 * 100.0,
        worst.0,
        worst.2,
        gate * 100.0
    );
    if alt_m.is_none() || !rows_mode {
        return;
    }
    // ★ THE RUNG BOUNDARIES, WITH THE AIR TAKEN OUT (2026-09-23, ruling W17).
    //
    // A step between neighbouring bands measures the LADDER and the AIR in front of it together,
    // and near the horizon the air wins: a band there spans a hundred kilometres of ground, and
    // the aerial perspective alone changes the shade by several per cent from one band to the
    // next. MEASURED on the 20 km stand before this section existed: the worst band step read
    // 6.3 % two bands under the horizon, at no rung boundary at all, and it fell smoothly with
    // distance — the air's own gradient, not a ring.
    //
    // So a boundary is judged against ITS OWN NEIGHBOURHOOD. A straight line is fitted to the
    // luminance of the rows OUTSIDE the crossfade band on each side; both lines are extrapolated
    // to the band's middle row; the GAP between them is the boundary's own step with the air's
    // smooth gradient removed. **A ring is a gap; haze is a slope, and a slope leaves no gap.**
    let row_mean = |row: usize, q: usize| -> Option<f64> {
        let n = row_count[row * sectors + q];
        (n >= 16).then(|| row_sum[row * sectors + q] / n as f64)
    };
    // The screen row whose ground stands nearest a stated eye distance, searched down the frame.
    let row_at = |target: f64| -> Option<usize> {
        let mut best: Option<(f64, usize)> = None;
        for row in (hud_rows as usize)..(h as usize) {
            if let Some((slant, _, _)) = stand(row as f64) {
                let miss = (slant - target).abs();
                if best.is_none_or(|(b, _)| miss < b) {
                    best = Some((miss, row));
                }
            }
        }
        best.map(|(_, r)| r)
    };
    // A least-squares line through the rows of one window, extrapolated to `at`.
    let trend = |rows: &[usize], q: usize, at: f64| -> Option<f64> {
        let held: Vec<(f64, f64)> = rows
            .iter()
            .filter_map(|r| row_mean(*r, q).map(|v| (*r as f64, v)))
            .collect();
        if held.len() < 3 {
            return None;
        }
        let n = held.len() as f64;
        let mx = held.iter().map(|(x, _)| x).sum::<f64>() / n;
        let my = held.iter().map(|(_, y)| y).sum::<f64>() / n;
        let sxy: f64 = held.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
        let sxx: f64 = held.iter().map(|(x, _)| (x - mx) * (x - mx)).sum();
        let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
        Some(my + slope * (at - mx))
    };
    println!(
        "\nTHE RUNG BOUNDARIES in this frame, the air's own gradient taken out (the gate is {:.1} % of the local mean)",
        gate * 100.0
    );
    println!(
        "rung pair | switch m | the band m | the band's rows | window rows | the worst SECTOR gap % | the mean gap % | water FAR side % | water NEAR side %"
    );
    let mut worst_gap = (0u8, 0.0f64);
    let mut boundaries = 0usize;
    let mut over_gate = 0usize;
    for rung in 0..rungs.saturating_sub(1) {
        let band = home_bound.fade_bands(rung + 1, rungs).0;
        let (Some(far), Some(near)) = (row_at(band[1]), row_at(band[0])) else {
            continue;
        };
        // The band must stand INSIDE the read area with room for a window on each side.
        let width = near.saturating_sub(far).max(4);
        if far <= hud_rows as usize + 3 || near + 4 >= h as usize || near <= far {
            continue;
        }
        let window = width.clamp(6, 40);
        // ★ THE WINDOW WALKS OUTWARD UNTIL IT HAS LAND (ruling W17): a band that falls inside a
        // bay has sea on one side of it, and sea is another material. The fit takes the nearest
        // `window` rows OUTSIDE the band that hold land at all, searching up to three windows
        // away; where it cannot find three such rows the boundary is reported as unjudgeable
        // rather than as a ring.
        let has_land = |r: usize| (0..sectors).any(|q| row_count[r * sectors + q] >= 16);
        let far_rows: Vec<usize> = (hud_rows as usize..far)
            .rev()
            .take(window * 3)
            .filter(|r| has_land(*r))
            .take(window)
            .collect();
        let near_rows: Vec<usize> = ((near + 1)..(h as usize))
            .take(window * 3)
            .filter(|r| has_land(*r))
            .take(window)
            .collect();
        let middle = (far + near) as f64 / 2.0;
        let mut gaps: Vec<f64> = Vec::new();
        for q in 0..sectors {
            let (Some(a), Some(b)) = (trend(&far_rows, q, middle), trend(&near_rows, q, middle))
            else {
                continue;
            };
            gaps.push((a - b).abs() / ((a + b) / 2.0).max(1.0e-9));
        }
        // ★ THE WATER'S OWN SHARE on each side of the band: how much of the ground drawn there is
        // under water. A rung that draws water where its neighbour draws land is the shape fault,
        // not a shade fault, and it is a number of its own.
        let share = |rows: &[usize]| -> f64 {
            let (mut w, mut n) = (0u64, 0u64);
            for r in rows {
                for q in 0..sectors {
                    w += row_wet[r * sectors + q];
                    n += row_all[r * sectors + q];
                }
            }
            f64::from(u32::try_from(w).unwrap_or(u32::MAX)) / n.max(1) as f64
        };
        let (far_share, near_share) = (share(&far_rows), share(&near_rows));
        if gaps.is_empty() {
            continue;
        }
        boundaries += 1;
        let top = gaps.iter().fold(0.0f64, |m, g| m.max(*g));
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        if top > worst_gap.1 {
            worst_gap = (rung, top);
        }
        over_gate += usize::from(top > gate);
        println!(
            "{rung:>4} -> {:>2} | {:>8.0} | {:>5.0}..{:>7.0} | {far:>5}..{near:<8} | {:>11} | {:>22.3} | {:>14.3} | {:>13.1} | {:>14.1}",
            rung + 1,
            home_bound.switch_m(rung),
            band[0],
            band[1],
            window,
            top * 100.0,
            mean * 100.0,
            far_share * 100.0,
            near_share * 100.0
        );
    }
    println!(
        "{boundaries} rung boundaries stand in this frame; the worst SECTOR gap is {:.3} % at the {} -> {} swap; {over_gate} past the stated {:.1} %",
        worst_gap.1 * 100.0,
        worst_gap.0,
        worst_gap.0 + 1,
        gate * 100.0
    );
}
