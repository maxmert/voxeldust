//! M-C — THE SLOPE HISTOGRAM, AND THE ROUGHNESS FACTOR'S OWN (slice 8a stage 3;
//! `slice_8a_design.md` §4 M-C; `slice_landforms_discussion.md` §8 M9; `04_detail_rungs.md` §4.8).
//!
//! The proxy P2 had no instrument in the tree. This is it. It reads the shipped height field
//! (`vd_terrain::height::height_m`) over a grid of columns on every face of the home planet and
//! prints, at each stated baseline:
//!
//! * the slope histogram of the WHOLE planet — the median and the 95th percentile, in degrees;
//! * the same for the columns the roughness factor calls a PLAIN and for those it calls a RANGE;
//! * the factor's own histogram, so the owner sees what share of the planet is which.
//!
//! **The band the arc asks for:** median slope 3°–12°, p95 25°–40°, and — the half that proves the
//! per-column factor works — the PLAIN's p99 under 5° while the RANGE's p50 sits at 28°–34°. Today
//! (before slice 8a) the whole planet measured 1.6°–2.0° at every baseline from 2 m to 8 km
//! (`02_planet_layout.md` §1.2), a single spike. A single-humped histogram after 8a means the
//! per-column factor is not modulating anything.
//!
//! **The slope, stated exactly.** At a column's direction the instrument takes ONE tangent step of
//! the baseline's own length — `d' = normalise(d + t · b/R)` — and reads the height at both ends.
//! The slope is the rise over the baseline, and the angle is its arc tangent. A tangent step is the
//! same length wherever it stands, which a face parameter is not.
//!
//! ```text
//! cargo run --release -p vd-bins --example slope_histogram
//! ```

use std::process::ExitCode;

use vd_bins::DEV;
use vd_seed::bend::{Face, direction};
use vd_terrain::height::{height_m, roughness_at};

/// Directions per face edge in the sample grid: 6 × 40 × 40 = 9 600 columns.
const GRID: i32 = 40;
/// The baselines the histogram is taken at, in metres — the same span `02` §1.2 measured on.
const BASELINES_M: [f64; 5] = [4.0, 32.0, 256.0, 2_048.0, 8_192.0];
/// A column is a PLAIN at or under this factor and a RANGE at or over the other — the quarter
/// points of the factor's own band, so the two names are stated once and never fitted.
const PLAIN_AT_OR_UNDER: f64 = 0.25;
const RANGE_AT_OR_OVER: f64 = 0.75;
/// The factor histogram's buckets.
const BUCKETS: usize = 10;

fn normalise(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// A unit tangent of a direction: the cross product with the pole, or with `+X` at the pole itself.
fn tangent(d: [f64; 3]) -> [f64; 3] {
    let c = [d[1], -d[0], 0.0];
    let n = (c[0] * c[0] + c[1] * c[1]).sqrt();
    if n < 1e-6 {
        [0.0, 1.0, 0.0]
    } else {
        [c[0] / n, c[1] / n, 0.0]
    }
}

fn percentile(sorted: &[f64], share: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() as f64) * share) as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn report(label: &str, degrees: &mut [f64]) {
    degrees.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    println!(
        "    {label:<8} n {:>6}   p50 {:>6.2}°   p95 {:>6.2}°   p99 {:>6.2}°   max {:>6.2}°",
        degrees.len(),
        percentile(degrees, 0.50),
        percentile(degrees, 0.95),
        percentile(degrees, 0.99),
        percentile(degrees, 1.0)
    );
}

fn main() -> ExitCode {
    let Some(body) = vd_bins::home_body(DEV.universe_seed) else {
        println!("slope_histogram: REFUSED — the home system holds no planet the recipe accepts");
        return ExitCode::FAILURE;
    };
    let radius_m = body.radius_m();
    println!(
        "slope_histogram: home planet {} — radius {:.0} m, {} octaves, Σ|a| {:.2} m, {} columns",
        body.seed(),
        radius_m,
        body.octaves_at(0).len(),
        body.relief_bound_m(0),
        6 * GRID * GRID
    );

    // The columns, their directions and their factors — read once and reused at every baseline.
    let faces = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];
    let mut columns: Vec<([f64; 3], f64)> = Vec::with_capacity(6 * (GRID * GRID) as usize);
    for face in faces {
        let mut i = 0;
        while i < GRID {
            let mut j = 0;
            while j < GRID {
                let a = (2.0 * (f64::from(i) + 0.5) / f64::from(GRID)) - 1.0;
                let b = (2.0 * (f64::from(j) + 0.5) / f64::from(GRID)) - 1.0;
                let d = direction(face, a, b);
                let dg = [d[0], d[1], d[2]];
                columns.push((dg, roughness_at(&body, dg)));
                j += 1;
            }
            i += 1;
        }
    }

    // ★ THE FACTOR'S OWN HISTOGRAM: what share of the planet is a plain and what a range.
    let mut buckets = [0usize; BUCKETS];
    let (mut plain, mut range) = (0usize, 0usize);
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for (_, m) in &columns {
        let b = ((*m * BUCKETS as f64) as usize).min(BUCKETS - 1);
        buckets[b] += 1;
        plain += usize::from(*m <= PLAIN_AT_OR_UNDER);
        range += usize::from(*m >= RANGE_AT_OR_OVER);
        lo = lo.min(*m);
        hi = hi.max(*m);
    }
    let n = columns.len() as f64;
    println!(
        "  the roughness factor over {} columns: lowest {lo:.4}, highest {hi:.4}",
        columns.len()
    );
    print!("    buckets of a tenth:");
    for (b, count) in buckets.iter().enumerate() {
        print!(
            " [{:.1}) {:.1}%",
            b as f64 / 10.0,
            *count as f64 * 100.0 / n
        );
    }
    println!();
    println!(
        "    PLAIN (m ≤ {PLAIN_AT_OR_UNDER}): {:.1}% of the planet；RANGE (m ≥ {RANGE_AT_OR_OVER}): {:.1}%",
        plain as f64 * 100.0 / n,
        range as f64 * 100.0 / n
    );

    // ★ THE SLOPE HISTOGRAM at each baseline, whole planet, plain and range.
    for baseline in BASELINES_M {
        println!("  baseline {baseline:.0} m:");
        let mut all = Vec::with_capacity(columns.len());
        let mut plains = Vec::new();
        let mut ranges = Vec::new();
        for (d, m) in &columns {
            let t = tangent(*d);
            let step = baseline / radius_m;
            let d2 = normalise([d[0] + t[0] * step, d[1] + t[1] * step, d[2] + t[2] * step]);
            let rise = height_m(&body, d2, 0) - height_m(&body, *d, 0);
            let degrees = (rise / baseline).abs().atan().to_degrees();
            all.push(degrees);
            if *m <= PLAIN_AT_OR_UNDER {
                plains.push(degrees);
            }
            if *m >= RANGE_AT_OR_OVER {
                ranges.push(degrees);
            }
        }
        report("whole", &mut all);
        report("plain", &mut plains);
        report("range", &mut ranges);
    }
    println!(
        "slope_histogram: the arc's band — whole p50 3°–12°, whole p95 25°–40°, the PLAIN's p99 under 5°, the RANGE's p50 28°–34°"
    );
    ExitCode::SUCCESS
}
