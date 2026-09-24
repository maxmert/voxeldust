//! M1b — THE SKYLINE PREDICTION (the landform arc, ruling V13; `01_reference_target.md` §6.1).
//!
//! From four named stations on the home planet, an eye 373 m over the surface marches 360 bearings out
//! to 540 km through the height field and records the SKYLINE: the largest elevation angle along each
//! bearing. A BREAK is a local maximum of the skyline that stands over the median of the skyline within
//! ±5° of bearing by more than a threshold. Printed for the field TODAY (the shipped octave table) and
//! for the field under the PROPOSED slope spectrum (`03_erosion_rivers.md` §4.2.3: the fine octaves'
//! amplitudes from a rational bump in the octave index, anchored on the angle of repose; the coarse
//! octaves unchanged). The third column adds the RIDGED BAND and the PER-COLUMN ROUGHNESS FACTOR,
//! which together are the field the world ships — so the third column and the TODAY column must
//! print the same numbers. That agreement is a CROSS-CHECK THAT COULD FAIL: this instrument states
//! the shape in plain 64-bit floats and the world computes it in the recipe's fenced integers. The
//! spectrum's own table is reprinted beside the computed amplitudes so a mismatch shows.
//!
//! ★ THE MARCH AND THE BREAK RULE LIVE IN `vd_bins::skyline` (ruling B6 item 2, 2026-09-24): the
//! reference stand (`reference_stand`) reads the same rule, so the two instruments cannot drift
//! apart. They moved there unchanged; this instrument states its own eye height and its own field.
//!
//! ```text
//! cargo run --release -p vd-bins --example skyline_march
//! ```

use std::process::ExitCode;

use vd_bins::DEV;
use vd_bins::skyline::{BEARINGS, Eye, Fan, Ground, M1B_STEPS, THRESHOLDS_DEG, judge, march};
use vd_seed::bend::{Face, direction};
use vd_terrain::height::height_m;

/// The eye's height over its own surface, metres (`01` §6.1).
const EYE_M: f64 = 373.0;

/// The four stations (`01` §1.4a): face and face parameters.
const STATIONS: [(&str, Face, f64, f64); 4] = [
    ("S2-a", Face::PosX, 0.296_580, 0.142_400),
    ("S2-b", Face::PosY, -0.190_120, 0.418_999),
    ("S2-c", Face::PosZ, 0.052_660, -0.342_400),
    ("S2-d", Face::NegX, 0.449_620, 0.235_561),
];

/// The proposed spectrum (`03` §4.2.3): a rational bump in the octave index over the FINE octaves.
const O_PEAK: f64 = 7.0;
const SIGMA: f64 = 1.4;
const S_PEAK: f64 = 0.3998;
/// The first FINE octave (0..FIRST_FINE are the coarse ones the macro field replaces later).
const FIRST_FINE: usize = 4;

/// The MIDDLE BAND the shipped field makes RIDGED (`04` §4.4 rule 2: `1 − |noise|`, so a crest is a
/// LINE instead of a bump). ★ STATED IN METRES, the same pair the generator states
/// (`vd_terrain::body::RIDGE_LO_M` / `RIDGE_HI_M`), so the instrument and the world cannot drift
/// apart: an octave is ridged while its wavelength lies in `[RIDGE_LO_M, RIDGE_HI_M]`. On the home
/// planet that is octaves 5..=9, 12.5 km down to 781 m — the band M1b-2 measured.
const RIDGE_LO_M: u64 = vd_terrain::body::RIDGE_LO_M;
const RIDGE_HI_M: u64 = vd_terrain::body::RIDGE_HI_M;

/// ONE octave's noise along a direction, as a real number: the recipe's own integer kernel with the
/// amplitude of one noise unit, read out at the noise's fraction bits. The instrument needs the raw
/// noise because it re-weights the octaves itself.
fn octave_noise(o: &vd_recipe::height::Octave, dir: [vd_recipe::Gi; 3]) -> f64 {
    // ★ SMOOTH, ALWAYS. The instrument applies the ridge ITSELF below, so the octave it reads back
    // must be the RAW gradient noise: reading a ridged octave here and ridging it again would
    // measure a field nobody ships.
    let unit = vd_recipe::height::Octave::smooth(
        o.seed,
        o.frequency_int,
        o.frequency_frac,
        vd_recipe::Gi::new(1 << vd_recipe::height::AMP_BITS),
    );
    // The word is a NOISE VALUE at 28 fraction bits; it is read as a real number directly, never
    // narrowed to an i32 first (a 32-bit cast would silently wrap a value this instrument may widen).
    vd_recipe::height::relief(&[unit], dir).raw() as f64
        / f64::from(1u32 << vd_recipe::noise::NOISE_BITS)
}

/// The height field under a given amplitude table (the octaves' frequencies and seeds are the body's);
/// `ridged` turns the middle band into ridged noise, re-centred so its amplitude means the same, and
/// `rough` multiplies the FINE octaves by the column's own ROUGHNESS FACTOR (slice 8a stage 3).
///
/// ★ THE FACTOR IS READ THROUGH THE GENERATOR'S OWN DOOR (`vd_terrain::height::roughness_at`), not
/// re-derived here: the instrument states the SHAPE in plain floats and the world computes the same
/// shape in the fenced integers, so the two columns agreeing is a cross-check that could fail.
fn height_with(
    body: &vd_terrain::BodyDefinition,
    amplitudes: &[f64],
    ridged: bool,
    rough: bool,
    bench: bool,
    dir: [f64; 3],
) -> f64 {
    let d = vd_terrain::units::direction_of_unit(dir);
    let r = body.ladder().radius_m();
    let factor = if rough {
        vd_terrain::height::roughness_at(body, dir)
    } else {
        1.0
    };
    let mut h = r;
    for (i, (o, a)) in body.octaves_at(0).iter().zip(amplitudes).enumerate() {
        let lambda = r / vd_terrain::body::octave_frequency(o);
        let in_band = lambda >= RIDGE_LO_M as f64 && lambda <= RIDGE_HI_M as f64;
        let n = octave_noise(o, d);
        let v = if ridged && in_band {
            (1.0 - n.abs()) * 2.0 - 1.0
        } else {
            n
        };
        let m = if i >= FIRST_FINE { factor } else { 1.0 };
        h += a * v * m;
    }
    // ★ THE CAP-ROCK BENCH (slice 8a stage 4), THROUGH THE GENERATOR'S OWN DOOR
    // (`vd_terrain::height::terrace_m`) and never re-derived here — the same rule the roughness
    // factor above follows. The instrument states the octave table in plain floats; the bench is the
    // world's own fenced-integer kernel, so the two agreeing is a cross-check that could fail.
    if bench {
        vd_terrain::height::terrace_m(body, h, 0)
    } else {
        h
    }
}

/// The skyline from a station: the largest elevation angle along each bearing, degrees — the ONE
/// march (`vd_bins::skyline::march`), with the eye `EYE_M` over the field at the station and the
/// field read ON each marched direction.
fn skyline(
    body: &vd_terrain::BodyDefinition,
    field: &dyn Fn([f64; 3]) -> f64,
    d: [f64; 3],
) -> Vec<f64> {
    let eye = Eye {
        dir: d,
        radius_m: field(d) + EYE_M,
    };
    march(
        &eye,
        body.ladder().radius_m(),
        &Fan::full_turn(BEARINGS),
        &M1B_STEPS,
        &mut |dir: [f64; 3], _s: f64| Ground::OnRay(field(dir)),
    )
    .deg
}

fn main() -> ExitCode {
    let Some(body) = vd_bins::home_body(DEV.universe_seed) else {
        println!("skyline_march: REFUSED — the home system holds no planet the recipe accepts");
        return ExitCode::FAILURE;
    };
    let r = body.ladder().radius_m();
    let octaves = body.octaves_at(0);
    println!(
        "skyline_march: home planet {} — ladder radius {:.1} m, {} octaves",
        body.seed(),
        r,
        octaves.len()
    );
    // The amplitude tables: today's, and the proposed spectrum's.
    let today: Vec<f64> = octaves
        .iter()
        .map(vd_terrain::body::octave_amplitude_m)
        .collect();
    let mut proposed = today.clone();
    println!(
        "  the ridged band: [{RIDGE_LO_M} m, {RIDGE_HI_M} m] — an octave is a CREST inside it"
    );
    println!("  octave  wavelength(m)  amplitude today(m)  proposed(m)  slope proposed  kind");
    for (i, o) in octaves.iter().enumerate() {
        let lambda = r / vd_terrain::body::octave_frequency(o);
        if i >= FIRST_FINE {
            let x = ((i as f64) - O_PEAK) * std::f64::consts::LN_2;
            let s = S_PEAK / (1.0 + x * x / (SIGMA * SIGMA));
            proposed[i] = s * lambda / std::f64::consts::TAU;
        }
        let kind = if lambda >= RIDGE_LO_M as f64 && lambda <= RIDGE_HI_M as f64 {
            "ridged"
        } else {
            "smooth"
        };
        println!(
            "  {i:>6}  {lambda:>13.1}  {:>18.1}  {:>11.1}  {:.4}  {kind}",
            today[i],
            proposed[i],
            proposed[i] * std::f64::consts::TAU / lambda
        );
    }
    let fine_rms: f64 = (FIRST_FINE..octaves.len())
        .map(|i| {
            let s = proposed[i] * std::f64::consts::TAU
                / (r / vd_terrain::body::octave_frequency(&octaves[i]));
            s * s
        })
        .sum::<f64>()
        .sqrt();
    println!(
        "  proposed: fine octaves sum {:.0} m of relief {:.0} m; fine RMS slope {fine_rms:.3} (tan 35° = 0.700)",
        proposed[FIRST_FINE..].iter().sum::<f64>(),
        body.relief_bound_m(0)
    );
    let start = std::time::Instant::now();
    for (name, face, a, b) in STATIONS {
        let d = direction(face, a, b);
        let dg = [d[0], d[1], d[2]];
        let h_today = height_m(&body, dg, 0);
        let field_today = |dir: [f64; 3]| height_m(&body, [dir[0], dir[1], dir[2]], 0);
        // The spectrum alone, and the spectrum with the ridge AND the per-column roughness factor —
        // which is what the world now ships, so the third column and the TODAY column must agree.
        let field_new = |dir: [f64; 3]| height_with(&body, &proposed, false, false, false, dir);
        let field_ridged = |dir: [f64; 3]| height_with(&body, &proposed, true, true, true, dir);
        let sky_t = skyline(&body, &field_today, d);
        let sky_n = skyline(&body, &field_new, d);
        let sky_r = skyline(&body, &field_ridged, d);
        let (rise_t, breaks_t) = judge(&sky_t);
        let (rise_n, breaks_n) = judge(&sky_n);
        let (rise_r, breaks_r) = judge(&sky_r);
        let range = |s: &[f64]| {
            let lo = s.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (lo, hi)
        };
        println!(
            "  {name} {face:?} ({a:.6}, {b:.6}): surface {:+.1} m over the ladder radius, \
             roughness factor {:.4}",
            h_today - r,
            vd_terrain::height::roughness_at(&body, dg)
        );
        println!(
            "    TODAY:    largest rise {rise_t:.4}°, skyline range {:.2}°..{:.2}°, breaks/60° at {:?}° = {:?}",
            range(&sky_t).0,
            range(&sky_t).1,
            THRESHOLDS_DEG,
            breaks_t
        );
        println!(
            "    PROPOSED: largest rise {rise_n:.4}°, skyline range {:.2}°..{:.2}°, breaks/60° at {:?}° = {:?}",
            range(&sky_n).0,
            range(&sky_n).1,
            THRESHOLDS_DEG,
            breaks_n
        );
        println!(
            "    +RIDGED [{RIDGE_LO_M} m, {RIDGE_HI_M} m] +ROUGHNESS +BENCH ({:.0} m beds, strength {:.3}): largest rise {rise_r:.4}°, skyline range {:.2}°..{:.2}°, breaks/60° at {:?}° = {:?}",
            body.bed_spacing_m(),
            body.terrace_strength(0),
            range(&sky_r).0,
            range(&sky_r).1,
            THRESHOLDS_DEG,
            breaks_r
        );
    }
    println!(
        "skyline_march: {} stations × {BEARINGS} bearings to {} km, three fields, in {:.1} s",
        STATIONS.len(),
        M1B_STEPS.limit_m / 1000.0,
        start.elapsed().as_secs_f64()
    );
    ExitCode::SUCCESS
}
