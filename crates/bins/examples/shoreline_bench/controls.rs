//! ★ THE SYNTHETIC CONTROLS (plan §9; survey §1.8): every shape estimator reads a curve whose answer
//! is known BEFORE it reads our shore. They need no solve and run first. A failed CHECK-D-LINE or
//! CHECK-D-KOCH voids every shape gate of the run.
//!
//! **Example.** A straight coast at 17° reads D = 1.003; a Koch snowflake 1.270; a staircase of
//! 64-column steps leans on the grid with L4 = 0.97 and its 45° turn with L4 = −0.95; eight
//! fractional surfaces of H = 0.5 give the KINK spread the real shore is judged against.

use vd_bins::shore_geometry::zero_contours;
use vd_bins::shore_shape::{
    Point, chord_axes, divider_dimension, fill_polygon, flat_null_l4, four_fold, koch_snowflake,
    line_field, mean_sd, perimeter_area_slope, polyline_length, quantile, rulers,
    slopes_per_octave, squares_field, staircase_field, wm_fbm,
};

use crate::census::{Census, val};

/// What the controls hand the real lines: the KINK bands by the node's size in columns, the
/// lattice bands by the window's side in columns, whether KINK can turn the staircase red at all,
/// and whether the shape gates are void.
pub struct Bands {
    pub kink: Vec<(usize, f64)>,
    pub lattice: Vec<(usize, f64)>,
    pub kink_power: bool,
    pub void: bool,
}

impl Bands {
    #[must_use]
    pub fn kink_at(&self, node_cols: usize) -> Option<f64> {
        self.kink.iter().find(|k| k.0 == node_cols).map(|k| k.1)
    }
    /// The lattice band of a window of `n` columns: the fractional surfaces' own spread of |L4|.
    #[must_use]
    pub fn lattice_at(&self, n: usize) -> Option<f64> {
        self.lattice.iter().find(|k| k.0 == n).map(|k| k.1)
    }
}

/// The fBm field's contour at its median, as lines.
fn fbm_lines(n: usize, h: f64, seed: u64) -> Vec<Vec<Point>> {
    let f = wm_fbm(n, h, seed);
    let mut sorted = f.clone();
    sorted.sort_by(f64::total_cmp);
    let med = sorted[sorted.len() / 2];
    let field: Vec<Option<f64>> = f.iter().map(|v| Some(v - med)).collect();
    zero_contours(n, &field)
}

/// KINK at a node of `node` columns over the lines of four nodes or more.
fn kink(lines: &[Vec<Point>], node: usize) -> f64 {
    let long: Vec<Vec<Point>> = lines
        .iter()
        .filter(|l| polyline_length(l) >= 4.0 * node as f64)
        .cloned()
        .collect();
    if long.is_empty() {
        return f64::NAN;
    }
    let eps = [node as f64 / 2.0, node as f64, 2.0 * node as f64];
    let s = slopes_per_octave(&long, &eps);
    (s[0] - s[1]).abs()
}

/// ★ ALL CONTROLS, on `threads` threads for the fractional surfaces.
pub fn run(c: &mut Census, threads: usize, w2_side: usize) -> Bands {
    c.say("\n== THE CONTROLS (no solve; survey §1.8) ==");
    // CHECK-D-LINE.
    let n = 1_024;
    let lines = zero_contours(n, &line_field(n, 17.0));
    let d_line = divider_dimension(&lines, &rulers(8.0, 102.0, 2));
    let line_ok = (d_line - 1.0).abs() <= 0.01;
    c.judged(
        "CHECK-D-LINE",
        &val(d_line, 4, ""),
        "1.00+-0.01[survey§1.8]",
        line_ok,
    );
    // CHECK-D-KOCH.
    let n = 2_048;
    let poly = koch_snowflake(6, 1_500.0, 1_024.0, 1_024.0);
    let eps = rulers(8.0, 204.0, 4);
    let d_poly = divider_dimension(std::slice::from_ref(&poly), &eps);
    let traced = zero_contours(n, &fill_polygon(&poly, n));
    let longest = traced
        .into_iter()
        .max_by(|a, b| polyline_length(a).total_cmp(&polyline_length(b)))
        .unwrap_or_default();
    let d_traced = divider_dimension(&[longest], &eps);
    let koch_ok = (d_poly - 1.262).abs() <= 0.02 && (d_traced - 1.262).abs() <= 0.02;
    c.judged(
        "CHECK-D-KOCH",
        &format!("POLY{}/TRACED{}", val(d_poly, 4, ""), val(d_traced, 4, "")),
        "1.262+-0.02[survey§1.8]",
        koch_ok,
    );
    // CHECK-D-FBM: three H, eight seeds each, one loop at the median, D against (3 − H)/2.
    let jobs: Vec<(f64, u64)> = [0.3, 0.5, 0.7]
        .iter()
        .flat_map(|&h| (0..8u64).map(move |s| (h, 1_000 + s)))
        .collect();
    let results = parallel(&jobs, threads, |&(h, seed)| {
        let lines = fbm_lines(1_024, h, seed);
        let longest = lines
            .into_iter()
            .max_by(|a, b| polyline_length(a).total_cmp(&polyline_length(b)))
            .unwrap_or_default();
        let one =
            |lo: f64| divider_dimension(std::slice::from_ref(&longest), &rulers(lo, 102.0, 2));
        (h, one(8.0), [one(2.0), one(4.0), one(7.0), one(14.0)])
    });
    for &h in &[0.3, 0.5, 0.7] {
        let ds: Vec<f64> = results.iter().filter(|r| r.0 == h).map(|r| r.1).collect();
        let (m, sd) = mean_sd(&ds);
        let want = (3.0 - h) / 2.0;
        c.judged(
            &format!("CHECK-D-FBM.H{h:.1}"),
            &format!("mean{}/sd{}/8seeds", val(m, 4, ""), val(sd, 4, "")),
            &format!("(3-H)/2={want:.3}-in-mean+-2sd[survey§1.8;C4]"),
            (want - m).abs() <= 2.0 * sd,
        );
        let starts: Vec<String> = (0..4)
            .map(|k| {
                let v: Vec<f64> = results
                    .iter()
                    .filter(|r| r.0 == h)
                    .map(|r| r.2[k])
                    .collect();
                val(mean_sd(&v).0, 3, "")
            })
            .collect();
        c.reading(
            &format!("CHECK-D-FBM-START.H{h:.1}"),
            &format!(
                "from2:{}/4:{}/7:{}/14:{}",
                starts[0], starts[1], starts[2], starts[3]
            ),
            "survey§1.6-trap-1",
        );
    }
    // The KINK band and the tracer's own lean: eight H = 0.5 fields at W2's size (rung 7, node 64
    // columns), and eight at W2's size at rung 9 (a quarter as wide, node 16 columns).
    let seeds: Vec<u64> = (0..8u64).map(|s| 2_000 + s).collect();
    let fields = parallel(&seeds, threads, |&seed| {
        let lines = fbm_lines(w2_side, 0.5, seed);
        let axes = chord_axes(&lines, 8.0);
        let (l4, _) = four_fold(&axes);
        (kink(&lines, 64), l4, axes.len())
    });
    let small = parallel(&seeds, threads, |&seed| {
        kink(&fbm_lines(w2_side / 4, 0.5, seed + 100), 16)
    });
    let k64: Vec<f64> = fields
        .iter()
        .map(|f| f.0)
        .filter(|v| v.is_finite())
        .collect();
    let k16: Vec<f64> = small.iter().copied().filter(|v| v.is_finite()).collect();
    let band64 = quantile(&k64, 0.95);
    let band16 = quantile(&k16, 0.95);
    c.reading(
        "KINK-BAND.node64",
        &format!("q95{}/{}fields", val(band64, 4, ""), k64.len()),
        "8-WM-fBm,H=0.5,W2-size-at-rung-7",
    );
    c.reading(
        "KINK-BAND.node16",
        &format!("q95{}/{}fields", val(band16, 4, ""), k16.len()),
        "8-WM-fBm,H=0.5,W2-size-at-rung-9",
    );
    // ★ THE LATTICE BAND: the flat null treats a line's windows as independent, and they are not
    // (a long line gives many windows at one angle). MEASURED on the first runs: 7 of 8 isotropic
    // fields read over the flat null. So G-SHORE-LATTICE's band is the isotropic fields' own spread
    // of |L4| at the window's size (the survey's "null's spread", §1.7), and the flat null stands
    // beside it. Three sizes: W2 at rungs 7 (1 984 columns, windows of 2 km = 16.8 columns), 9 (496)
    // and 10 (248, windows of 8 columns).
    let mut flat_red = 0usize;
    for (k, f) in fields.iter().enumerate() {
        let null = flat_null_l4(f.2, 2_000, 0.95);
        flat_red += usize::from(f.1.abs() > null);
        c.reading(
            &format!("CHECK-D-FBM-LATTICE.seed{k}"),
            &format!(
                "L4{}/{}windows/flat-null{}",
                val(f.1, 4, ""),
                f.2,
                val(null, 4, "")
            ),
            "an-isotropic-field's-own-lean",
        );
    }
    let lean = |size: usize, window: f64, base: u64| -> f64 {
        let seeds: Vec<u64> = (0..8u64).map(|s| base + s).collect();
        let l4s = parallel(&seeds, threads, |&seed| {
            four_fold(&chord_axes(&fbm_lines(size, 0.5, seed), window))
                .0
                .abs()
        });
        quantile(&l4s, 0.95)
    };
    let lattice = vec![
        (w2_side, lean(w2_side, 16.8, 3_000)),
        (w2_side / 4, lean(w2_side / 4, 8.0, 3_100)),
        (w2_side / 8, lean(w2_side / 8, 8.0, 3_200)),
    ];
    for (n, b) in &lattice {
        c.reading(
            &format!("LATTICE-BAND.n{n}"),
            &format!("q95|L4|{}", val(*b, 4, "")),
            "8-WM-fBm,H=0.5;the-band-G-SHORE-LATTICE-reads",
        );
    }
    c.say(format!(
        "   the flat null is too strict for a traced line: {flat_red} of 8 isotropic fields read over it; G-SHORE-LATTICE is judged against the fields' own spread (LATTICE-BAND)"
    ));
    // CHECK-D-STAIR: a staircase of 64-column steps at 30°, and its 45° turn.
    let n = w2_side;
    let stair = zero_contours(n, &staircase_field(n, 64.0, 30.0, 0.0));
    let turned = zero_contours(n, &staircase_field(n, 64.0, 30.0, 45.0));
    let axes = chord_axes(&stair, 8.0);
    let (l4, _) = four_fold(&axes);
    let (l4t, _) = four_fold(&chord_axes(&turned, 8.0));
    let null = flat_null_l4(axes.len(), 2_000, 0.95).max(lattice[0].1);
    let lattice_red = l4.abs() > null;
    c.judged(
        "CHECK-D-STAIR.LATTICE",
        &format!("L4{}/turned{}", val(l4, 3, ""), val(l4t, 3, "")),
        &format!(
            "must-read-RED(|L4|>{})-and-the-turn-flips-the-sign",
            val(null, 3, "")
        ),
        lattice_red && l4 * l4t < 0.0,
    );
    let ks = kink(&stair, 64);
    let kink_power = ks > band64;
    c.judged(
        "CHECK-D-STAIR.KINK",
        &val(ks, 4, ""),
        &format!("must-read-over-the-fBm-q95({})", val(band64, 4, "")),
        kink_power,
    );
    if !kink_power {
        c.say("   G-COAST-KINK HAS NO POWER here: an 8 km staircase reads inside the isotropic fields' own spread at one node in a 31-node window; every G-COAST-KINK line prints NO-BAND (the lattice lean, G-SHORE-LATTICE, carries the staircase)");
    }
    // CHECK-LAKE-SQUARES.
    let (slope, count) =
        perimeter_area_slope(2_048, &squares_field(2_048, 50, 12.0, 200.0, 7), 10.0);
    c.judged(
        "CHECK-LAKE-SQUARES",
        &format!(
            "slope{}/d{}/{count}squares",
            val(slope, 4, ""),
            val(2.0 * slope, 3, "")
        ),
        "0.50+-0.01;d=1.00-reads-RED-against-Earth's[1.12,1.40]",
        (slope - 0.5).abs() <= 0.01 && !(1.12..=1.40).contains(&(2.0 * slope)),
    );
    let void = !(line_ok && koch_ok);
    if void {
        c.say("######## VOID: CHECK-D-LINE or CHECK-D-KOCH failed; every shape gate prints VOID ########");
    }
    let _ = slopes_per_octave;
    Bands {
        kink: vec![(64, band64), (16, band16)],
        lattice,
        kink_power,
        void,
    }
}

/// A map over `jobs` on `threads` threads, the answers in the jobs' order.
fn parallel<J: Sync, R: Send>(jobs: &[J], threads: usize, f: impl Fn(&J) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: std::sync::Mutex<Vec<Option<R>>> =
        std::sync::Mutex::new((0..jobs.len()).map(|_| None).collect());
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if k >= jobs.len() {
                        break;
                    }
                    let r = f(&jobs[k]);
                    if let Ok(mut o) = out.lock() {
                        o[k] = Some(r);
                    }
                }
            });
        }
    });
    out.into_inner()
        .unwrap_or_default()
        .into_iter()
        .flatten()
        .collect()
}
