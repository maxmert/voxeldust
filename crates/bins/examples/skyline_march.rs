//! M1b — THE SKYLINE PREDICTION (the landform arc, ruling V13; `01_reference_target.md` §6.1).
//!
//! From four named stations on the home planet, an eye 373 m over the surface marches 360 bearings out
//! to 540 km through the height field and records the SKYLINE: the largest elevation angle along each
//! bearing. A BREAK is a local maximum of the skyline that stands over the median of the skyline within
//! ±5° of bearing by more than a threshold. Printed for the field TODAY (the shipped octave table) and
//! for the field under the PROPOSED slope spectrum (`03_erosion_rivers.md` §4.2.3: the fine octaves'
//! amplitudes from a rational bump in the octave index, anchored on the angle of repose; the coarse
//! octaves unchanged; the roughness field at its maximum everywhere, which is the prediction's stated
//! assumption). The spectrum's own table is reprinted beside the computed amplitudes so a mismatch
//! shows.
//!
//! ```text
//! cargo run --release -p vd-bins --example skyline_march
//! ```

use std::process::ExitCode;

use vd_bins::DEV;
use vd_seed::bend::{Face, direction};
use vd_terrain::height::height_m;

/// The eye's height over its own surface, metres (`01` §6.1).
const EYE_M: f64 = 373.0;
/// The march: 50 m steps to 5 km, then 300 m steps to the limit.
const NEAR_STEP_M: f64 = 50.0;
const NEAR_LIMIT_M: f64 = 5_000.0;
const FAR_STEP_M: f64 = 300.0;
const MARCH_LIMIT_M: f64 = 540_000.0;
/// Bearings per full turn, and the half-window (in bearings) of the local median.
const BEARINGS: usize = 360;
const HALF_WINDOW: usize = 5;
/// The break thresholds, degrees.
const THRESHOLDS_DEG: [f64; 4] = [0.05, 0.10, 0.25, 0.50];

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

/// The MIDDLE BAND the proposal makes RIDGED (`03` §4.2.3 / `04`: `1 − |noise|`, so a crest is a
/// line instead of a bump): the octaves whose wavelength lies between the hillslope length and the
/// macro node — here octaves 5..=9 (12.5 km to 781 m).
const RIDGED_BAND: std::ops::RangeInclusive<usize> = 5..=9;

/// ONE octave's noise along a direction, as a real number: the recipe's own integer kernel with the
/// amplitude of one noise unit, read out at the noise's fraction bits. The instrument needs the raw
/// noise because it re-weights the octaves itself.
fn octave_noise(o: &vd_recipe::height::Octave, dir: [vd_recipe::Gi; 3]) -> f64 {
    let unit = vd_recipe::height::Octave {
        amplitude: vd_recipe::Gi::new(1 << vd_recipe::height::AMP_BITS),
        ..*o
    };
    // The word is a NOISE VALUE at 28 fraction bits; it is read as a real number directly, never
    // narrowed to an i32 first (a 32-bit cast would silently wrap a value this instrument may widen).
    vd_recipe::height::relief(&[unit], dir).raw() as f64
        / f64::from(1u32 << vd_recipe::noise::NOISE_BITS)
}

/// The height field under a given amplitude table (the octaves' frequencies and seeds are the body's);
/// `ridged` turns the middle band into ridged noise, re-centred so its amplitude means the same.
fn height_with(
    body: &vd_terrain::BodyDefinition,
    amplitudes: &[f64],
    ridged: bool,
    dir: [f64; 3],
) -> f64 {
    let d = vd_terrain::units::direction_of_unit(dir);
    let mut h = body.ladder().radius_m();
    for (i, (o, a)) in body.octaves_at(0).iter().zip(amplitudes).enumerate() {
        let n = octave_noise(o, d);
        let v = if ridged && RIDGED_BAND.contains(&i) {
            (1.0 - n.abs()) * 2.0 - 1.0
        } else {
            n
        };
        h += a * v;
    }
    h
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The skyline from a station: the largest elevation angle along each bearing, degrees.
fn skyline(
    body: &vd_terrain::BodyDefinition,
    field: &dyn Fn([f64; 3]) -> f64,
    d: [f64; 3],
) -> Vec<f64> {
    let r = body.ladder().radius_m();
    let r_c = field(d) + EYE_M;
    // A tangent basis at the station.
    let helper = if d[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let e1 = normalize(cross(d, helper));
    let e2 = cross(d, e1);
    let mut out = Vec::with_capacity(BEARINGS);
    let mut b = 0;
    while b < BEARINGS {
        let phi = (b as f64) * std::f64::consts::TAU / (BEARINGS as f64);
        let (sp, cp) = phi.sin_cos();
        let t = [
            e1[0] * cp + e2[0] * sp,
            e1[1] * cp + e2[1] * sp,
            e1[2] * cp + e2[2] * sp,
        ];
        let mut best = f64::NEG_INFINITY;
        let mut s = NEAR_STEP_M;
        while s <= MARCH_LIMIT_M {
            let theta = s / r;
            let (st, ct) = theta.sin_cos();
            let dir = normalize([
                d[0] * ct + t[0] * st,
                d[1] * ct + t[1] * st,
                d[2] * ct + t[2] * st,
            ]);
            let r_t = field(dir);
            let elev = (r_t * ct - r_c).atan2(r_t * st);
            if elev > best {
                best = elev;
            }
            s += if s < NEAR_LIMIT_M {
                NEAR_STEP_M
            } else {
                FAR_STEP_M
            };
        }
        out.push(best.to_degrees());
        b += 1;
    }
    out
}

/// The rise of each bearing's skyline over the median of its ±5° window, and the breaks per 60°.
fn judge(sky: &[f64]) -> (f64, [f64; 4]) {
    let n = sky.len();
    let mut largest = f64::NEG_INFINITY;
    let mut rises = vec![0.0; n];
    for b in 0..n {
        let mut window: Vec<f64> = (0..=2 * HALF_WINDOW)
            .map(|k| sky[(b + n + k - HALF_WINDOW) % n])
            .collect();
        window.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        let median = window[HALF_WINDOW];
        rises[b] = sky[b] - median;
        largest = largest.max(rises[b]);
    }
    let mut per_60 = [0.0; 4];
    for (k, t) in THRESHOLDS_DEG.iter().enumerate() {
        let mut count = 0;
        for b in 0..n {
            let local_max = sky[b] > sky[(b + n - 1) % n] && sky[b] > sky[(b + 1) % n];
            if local_max && rises[b] > *t {
                count += 1;
            }
        }
        per_60[k] = f64::from(count) / 6.0;
    }
    (largest, per_60)
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
    println!("  octave  wavelength(m)  amplitude today(m)  proposed(m)  slope proposed");
    for (i, o) in octaves.iter().enumerate() {
        let lambda = r / vd_terrain::body::octave_frequency(o);
        if i >= FIRST_FINE {
            let x = ((i as f64) - O_PEAK) * std::f64::consts::LN_2;
            let s = S_PEAK / (1.0 + x * x / (SIGMA * SIGMA));
            proposed[i] = s * lambda / std::f64::consts::TAU;
        }
        println!(
            "  {i:>6}  {lambda:>13.1}  {:>18.1}  {:>11.1}  {:.4}",
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
        let field_new = |dir: [f64; 3]| height_with(&body, &proposed, false, dir);
        let field_ridged = |dir: [f64; 3]| height_with(&body, &proposed, true, dir);
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
            "  {name} {face:?} ({a:.6}, {b:.6}): surface {:+.1} m over the ladder radius",
            h_today - r
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
            "    +RIDGED {:?}: largest rise {rise_r:.4}°, skyline range {:.2}°..{:.2}°, breaks/60° at {:?}° = {:?}",
            RIDGED_BAND,
            range(&sky_r).0,
            range(&sky_r).1,
            THRESHOLDS_DEG,
            breaks_r
        );
    }
    println!(
        "skyline_march: {} stations × {BEARINGS} bearings to {} km, three fields, in {:.1} s",
        STATIONS.len(),
        MARCH_LIMIT_M / 1000.0,
        start.elapsed().as_secs_f64()
    );
    ExitCode::SUCCESS
}
