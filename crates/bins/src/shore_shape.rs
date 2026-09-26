//! ★ THE SHORE'S SHAPE, AS NUMBERS — the estimators the shoreline bench reads a drawn shore with
//! (`docs/investigation/2026-09-25/shoreline_papers_survey.md` §1.6–§1.8 and §2.3; plan step 4
//! slice 0).
//!
//! **WHY IT EXISTS.** A natural shore is rough in a regular way at every scale, and our drawn shore
//! today is a staircase on the 8 192 m node grid. The survey turns both into lines that can go RED:
//! the four-fold moment of the shore's direction against a flat null (G-SHORE-LATTICE), the change
//! of the length law's slope across one node (G-COAST-KINK), the perimeter–area law of the lakes
//! (G-LAKE-FRACTAL) and its bias-corrected index (G-LAKE-DBC). Every estimator here is checked on
//! a curve whose answer is known BEFORE it reads our shore (§1.8, the failing controls): a straight
//! line, a Koch snowflake, a Weierstrass–Mandelbrot fractional surface, an 8 km staircase and a set
//! of squares.
//!
//! **ONE METHOD** (§1.6 trap 5): the divider's FIRST crossing (the complete perimeter), on a
//! marching-squares contour (`shore_geometry::zero_contours`), in the window's own metric.
//!
//! **Example.** The bench traces the plain's drawn shore at rung 7 and walks it with rulers of 8,
//! 16, 32 … 512 columns. A shore that follows the dunes lengthens by about 15 % each time the ruler
//! halves; a staircase of node squares stops lengthening at 64 columns (one node) and turns at
//! exactly that ruler: its KINK stands far over the spread the fractional surfaces read.

use crate::shore_geometry::{window_bearings, zero_contours};

/// A point of a traced line, in columns (or in any unit the caller states).
pub type Point = (f64, f64);

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE DIVIDER.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ THE DIVIDER'S LENGTH of a polyline with a ruler `eps` (the compass method, FIRST crossing):
/// from the line's first point, step to the first point FARTHER ALONG THE LINE that stands exactly
/// `eps` away, count the steps, and add the straight rest to the line's end. A closed loop repeats
/// its first point at its end. Returns the length and the number of whole steps.
#[must_use]
pub fn divider(line: &[Point], eps: f64) -> (f64, usize) {
    if line.len() < 2 || eps <= 0.0 {
        return (0.0, 0);
    }
    let mut steps = 0usize;
    let mut centre = line[0];
    // The walk's place: the segment `seg` (from `line[seg]` to `line[seg + 1]`) and the parameter
    // `u` along it, from which the next crossing is sought.
    let mut seg = 0usize;
    let mut u0 = 0.0f64;
    loop {
        let mut found: Option<(usize, f64)> = None;
        let mut k = seg;
        while k + 1 < line.len() {
            let (a, b) = (line[k], line[k + 1]);
            let d = (b.0 - a.0, b.1 - a.1);
            let aa = d.0 * d.0 + d.1 * d.1;
            // On the walk's own segment the crossing must lie past the centre; on a later one a
            // crossing exactly at its first point counts (a vertex on the circle), and one a hair
            // past a segment's end is that end.
            let (start, open) = if k == seg {
                (u0, 1.0e-12)
            } else {
                (0.0, -1.0e-12)
            };
            if aa > 0.0 {
                let f = (a.0 - centre.0, a.1 - centre.1);
                let bb = 2.0 * (d.0 * f.0 + d.1 * f.1);
                let cc = f.0 * f.0 + f.1 * f.1 - eps * eps;
                let disc = bb * bb - 4.0 * aa * cc;
                if disc >= 0.0 {
                    let root = disc.sqrt();
                    let r1 = (-bb - root) / (2.0 * aa);
                    let r2 = (-bb + root) / (2.0 * aa);
                    let pick = [r1, r2]
                        .into_iter()
                        .filter(|&r| r > start + open && r <= 1.0 + 1.0e-12)
                        .fold(f64::INFINITY, f64::min);
                    if pick.is_finite() {
                        found = Some((k, pick));
                        break;
                    }
                }
            }
            k += 1;
        }
        match found {
            Some((k, u)) => {
                let (a, b) = (line[k], line[k + 1]);
                let u = u.clamp(0.0, 1.0);
                centre = (a.0 + u * (b.0 - a.0), a.1 + u * (b.1 - a.1));
                steps += 1;
                seg = k;
                u0 = u;
            }
            None => {
                let end = line[line.len() - 1];
                let rest = ((end.0 - centre.0).powi(2) + (end.1 - centre.1).powi(2)).sqrt();
                return (steps as f64 * eps + rest, steps);
            }
        }
    }
}

/// The plain polyline length.
#[must_use]
pub fn polyline_length(line: &[Point]) -> f64 {
    line.windows(2)
        .map(|w| ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt())
        .sum()
}

/// ★ THE LENGTH LAW'S SLOPE per octave of the ruler: `σ_k = ln(L(2ε_k)/L(ε_k)) / ln 2` summed over
/// every line, for the rulers `eps` (each the double of the one before). A line of dimension D
/// reads `σ = 1 − D` at every octave.
#[must_use]
pub fn slopes_per_octave(lines: &[Vec<Point>], eps: &[f64]) -> Vec<f64> {
    let totals: Vec<f64> = eps
        .iter()
        .map(|&e| lines.iter().map(|l| divider(l, e).0).sum())
        .collect();
    totals
        .windows(2)
        .zip(eps.windows(2))
        .map(|(l, e)| (l[1] / l[0]).ln() / (e[1] / e[0]).ln())
        .collect()
}

/// ★ THE DIVIDER DIMENSION by ordinary least squares of `ln L` on `ln ε` over the rulers, summed
/// over every line: `D = 1 − slope`.
#[must_use]
pub fn divider_dimension(lines: &[Vec<Point>], eps: &[f64]) -> f64 {
    let x: Vec<f64> = eps.iter().map(|e| e.ln()).collect();
    let y: Vec<f64> = eps
        .iter()
        .map(|&e| lines.iter().map(|l| divider(l, e).0).sum::<f64>().ln())
        .collect();
    1.0 - ols(&x, &y).0
}

/// The rulers from `lo` to `hi` at `per_octave` steps an octave.
#[must_use]
pub fn rulers(lo: f64, hi: f64, per_octave: u32) -> Vec<f64> {
    let mut out = Vec::new();
    let step = 2f64.powf(1.0 / f64::from(per_octave.max(1)));
    let mut e = lo;
    while e <= hi * (1.0 + 1.0e-9) {
        out.push(e);
        e *= step;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE LATTICE MOMENT.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ THE FOUR-FOLD MOMENT of a set of axes (degrees in `[0, 180)` against the face grid): `L4 =
/// mean cos 4θ` (+1 along the grid's axes, −1 along its diagonals) and `|M4| = |mean e^{4iθ}|`.
/// `(NaN, NaN)` for no axis.
#[must_use]
pub fn four_fold(axes: &[f64]) -> (f64, f64) {
    if axes.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = axes.len() as f64;
    let (mut c, mut s) = (0.0f64, 0.0f64);
    for a in axes {
        let t = 4.0 * a.to_radians();
        c += t.cos();
        s += t.sin();
    }
    (c / n, (c * c + s * s).sqrt() / n)
}

/// The chord windows of every line as axes, a window `window` long (in the lines' unit).
#[must_use]
pub fn chord_axes(lines: &[Vec<Point>], window: f64) -> Vec<f64> {
    lines
        .iter()
        .flat_map(|l| window_bearings(l, window))
        .collect()
}

/// ★ THE FLAT NULL OF `|L4|`: the `q` quantile of `|mean cos 4θ|` over `trials` draws of `count`
/// axes with no direction of their own (a fixed xorshift, so the band is one number every run).
#[must_use]
pub fn flat_null_l4(count: usize, trials: usize, q: f64) -> f64 {
    if count == 0 || trials == 0 {
        return f64::NAN;
    }
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut out = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut c = 0.0f64;
        for _ in 0..count {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
            c += (4.0 * std::f64::consts::PI * u).cos();
        }
        out.push((c / count as f64).abs());
    }
    out.sort_by(f64::total_cmp);
    let k = ((out.len() as f64 * q) as usize).min(out.len() - 1);
    out[k]
}

/// Every point of the lines turned about the origin by `deg` degrees.
#[must_use]
pub fn rotated(lines: &[Vec<Point>], deg: f64) -> Vec<Vec<Point>> {
    let (s, c) = deg.to_radians().sin_cos();
    lines
        .iter()
        .map(|l| {
            l.iter()
                .map(|p| (c * p.0 - s * p.1, s * p.0 + c * p.1))
                .collect()
        })
        .collect()
}

/// Every point of the lines shifted by `(dx, dy)`.
#[must_use]
pub fn shifted(lines: &[Vec<Point>], dx: f64, dy: f64) -> Vec<Vec<Point>> {
    lines
        .iter()
        .map(|l| l.iter().map(|p| (p.0 + dx, p.1 + dy)).collect())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE LAKES: components, loops, areas.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ THE 4-CONNECTED COMPONENTS of a `w × h` grid of keys (`k = b·w + a`), each component of ONE
/// non-zero key. Returns each cell's component (`u32::MAX` for key 0) and the components' cells.
#[must_use]
pub fn components(w: usize, h: usize, key: &[u32]) -> (Vec<u32>, Vec<Vec<usize>>) {
    let mut label = vec![u32::MAX; w * h];
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if key[start] == 0 || label[start] != u32::MAX {
            continue;
        }
        let id = out.len() as u32;
        let k0 = key[start];
        let mut cells = Vec::new();
        label[start] = id;
        stack.push(start);
        while let Some(c) = stack.pop() {
            cells.push(c);
            let (a, b) = (c % w, c / w);
            let mut visit = |n: usize| {
                if key[n] == k0 && label[n] == u32::MAX {
                    label[n] = id;
                    stack.push(n);
                }
            };
            if a > 0 {
                visit(c - 1);
            }
            if a + 1 < w {
                visit(c + 1);
            }
            if b > 0 {
                visit(c - w);
            }
            if b + 1 < h {
                visit(c + w);
            }
        }
        out.push(cells);
    }
    (label, out)
}

/// The signed shoelace area of a closed loop (its first point repeated at its end or not).
#[must_use]
pub fn shoelace(poly: &[Point]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for k in 0..poly.len() {
        let p = poly[k];
        let q = poly[(k + 1) % poly.len()];
        a += p.0 * q.1 - q.0 * p.1;
    }
    0.5 * a
}

/// ★ THE OUTER LOOP of one component: its cells read −1 (with their own drawn value where given)
/// and every other cell +1, traced by marching squares over the component's box grown by one;
/// the closed loop of the greatest area is the outer loop (islands and lakes-in-islands are the
/// smaller loops). Returns the loop in the grid's columns, or `None` where no closed loop exists.
/// `value(cell)` gives a cell's own field (negative inside); a cell outside the component reads
/// `max(value, +ε)` so no other water joins the loop.
#[must_use]
pub fn outer_loop(
    w: usize,
    h: usize,
    cells: &[usize],
    value: &dyn Fn(usize) -> f64,
) -> Option<Vec<Point>> {
    if cells.is_empty() {
        return None;
    }
    let (mut a0, mut b0, mut a1, mut b1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for &c in cells {
        let (a, b) = (c % w, c / w);
        a0 = a0.min(a);
        b0 = b0.min(b);
        a1 = a1.max(a);
        b1 = b1.max(b);
    }
    // The box grown by one column each way; a cell outside the grid reads dry.
    let (ga0, gb0) = (a0 as i64 - 1, b0 as i64 - 1);
    let n = (a1 - a0 + 3).max(b1 - b0 + 3);
    let mut member = std::collections::BTreeSet::new();
    for &c in cells {
        member.insert(c);
    }
    let mut field = vec![Some(1.0f64); n * n];
    for (k, slot) in field.iter_mut().enumerate() {
        let (la, lb) = ((k % n) as i64 + ga0, (k / n) as i64 + gb0);
        if la < 0 || lb < 0 || la >= w as i64 || lb >= h as i64 {
            continue;
        }
        let c = lb as usize * w + la as usize;
        let v = value(c);
        *slot = Some(if member.contains(&c) {
            v.min(-1.0e-6)
        } else {
            v.max(1.0e-6)
        });
    }
    let lines = zero_contours(n, &field);
    let mut best: Option<(f64, Vec<Point>)> = None;
    for l in lines {
        if l.len() < 4 || l.first() != l.last() {
            continue;
        }
        let area = shoelace(&l).abs();
        if best.as_ref().is_none_or(|b| area > b.0) {
            best = Some((area, l));
        }
    }
    best.map(|(_, l)| {
        l.into_iter()
            .map(|p| (p.0 + ga0 as f64, p.1 + gb0 as f64))
            .collect()
    })
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE STATISTICS.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// Ordinary least squares `y = a·x + b`: `(a, b)`; `(NaN, NaN)` under two points.
#[must_use]
pub fn ols(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len().min(y.len());
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for k in 0..n {
        sxy += (x[k] - mx) * (y[k] - my);
        sxx += (x[k] - mx) * (x[k] - mx);
    }
    let a = sxy / sxx;
    (a, my - a * mx)
}

/// ★ KENDALL'S τ-b between two samples (the ties corrected), `O(n²)`.
#[must_use]
pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let (mut conc, mut disc, mut tx, mut ty) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..n {
        for j in i + 1..n {
            let dx = (x[i] - x[j]).signum() * f64::from(u8::from(x[i] != x[j]));
            let dy = (y[i] - y[j]).signum() * f64::from(u8::from(y[i] != y[j]));
            if dx == 0.0 && dy == 0.0 {
                continue;
            } else if dx == 0.0 {
                tx += 1;
            } else if dy == 0.0 {
                ty += 1;
            } else if dx * dy > 0.0 {
                conc += 1;
            } else {
                disc += 1;
            }
        }
    }
    let denom = (((conc + disc + tx) * (conc + disc + ty)) as f64).sqrt();
    if denom == 0.0 {
        f64::NAN
    } else {
        (conc - disc) as f64 / denom
    }
}

/// The `q` quantile of a slice by the index `⌊len·q⌋` (held to the last); `NaN` for none.
#[must_use]
pub fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let k = ((v.len() as f64 * q) as usize).min(v.len() - 1);
    v[k]
}

/// The mean and the standard deviation of a sample.
#[must_use]
pub fn mean_sd(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = values.len() as f64;
    let m = values.iter().sum::<f64>() / n;
    let v = values.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, v.sqrt())
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE SYNTHETIC CONTROLS (§1.8).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// A seed spread over the word (a multiply by the golden ratio's word), never zero: two seeds that
/// differ in their last bit start two different streams (`seed | 1` did not: 2 000 and 2 001 drew
/// one field, MEASURED in the first run).
#[must_use]
pub fn seeded(seed: u64) -> u64 {
    let s = seed.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    if s == 0 { 1 } else { s }
}

/// A fixed xorshift draw in `[0, 1)`.
pub fn draw(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// ★ CHECK-D-LINE's field: the signed distance to a straight line through the square's middle at
/// `bearing` degrees, on `n × n` samples.
#[must_use]
pub fn line_field(n: usize, bearing: f64) -> Vec<Option<f64>> {
    let (s, c) = bearing.to_radians().sin_cos();
    let mid = n as f64 / 2.0;
    (0..n * n)
        .map(|k| {
            let (a, b) = ((k % n) as f64 - mid, (k / n) as f64 - mid);
            Some(-s * a + c * b + 0.37)
        })
        .collect()
}

/// ★ THE KOCH SNOWFLAKE of `depth`, a closed polygon (its first point repeated) of side `side`
/// centred on `(cx, cy)`.
#[must_use]
pub fn koch_snowflake(depth: u32, side: f64, cx: f64, cy: f64) -> Vec<Point> {
    let h = side * 3f64.sqrt() / 2.0;
    let a = (cx - side / 2.0, cy - h / 3.0);
    let b = (cx + side / 2.0, cy - h / 3.0);
    let c = (cx, cy + 2.0 * h / 3.0);
    // Clockwise, so each bump points outward.
    let mut pts = vec![a, c, b, a];
    for _ in 0..depth {
        let mut next = Vec::with_capacity(pts.len() * 4);
        for w in pts.windows(2) {
            let (p, q) = (w[0], w[1]);
            let d = ((q.0 - p.0) / 3.0, (q.1 - p.1) / 3.0);
            let p1 = (p.0 + d.0, p.1 + d.1);
            let p3 = (p.0 + 2.0 * d.0, p.1 + 2.0 * d.1);
            // The bump: the middle third turned by +60° about p1 — to the LEFT of the walk, which
            // is outward for this clockwise loop.
            let (s, co) = 60f64.to_radians().sin_cos();
            let p2 = (p1.0 + co * d.0 - s * d.1, p1.1 + s * d.0 + co * d.1);
            next.extend_from_slice(&[p, p1, p2, p3]);
        }
        next.push(pts[pts.len() - 1]);
        pts = next;
    }
    pts
}

/// ★ A POLYGON FILLED BY SCANLINE (the even-odd rule at each row's sample centre): −1 inside, +1
/// outside, on `n × n` samples.
#[must_use]
pub fn fill_polygon(poly: &[Point], n: usize) -> Vec<Option<f64>> {
    let mut out = vec![Some(1.0); n * n];
    for b in 0..n {
        let y = b as f64;
        let mut xs: Vec<f64> = Vec::new();
        for w in poly.windows(2) {
            let (p, q) = (w[0], w[1]);
            if (p.1 <= y) != (q.1 <= y) {
                xs.push(p.0 + (y - p.1) / (q.1 - p.1) * (q.0 - p.0));
            }
        }
        xs.sort_by(f64::total_cmp);
        for pair in xs.chunks(2) {
            if pair.len() < 2 {
                continue;
            }
            let lo = pair[0].ceil().max(0.0) as usize;
            let hi = (pair[1].floor() as i64).min(n as i64 - 1);
            let mut a = lo as i64;
            while a <= hi {
                out[b * n + a as usize] = Some(-1.0);
                a += 1;
            }
        }
    }
    out
}

/// ★ CHECK-D-FBM's field: a Weierstrass–Mandelbrot sum `Σ_k γ^(−kH) cos(γ^k k₀ (u_{k,d}·x) +
/// φ_{k,d})` with `γ = √2`, four random directions a frequency, wavelengths from the window down
/// to two cells, and seeded draws. It is isotropic by construction (the game's Perlin noise is
/// not a lattice-free null, D9).
#[must_use]
pub fn wm_fbm(n: usize, h: f64, seed: u64) -> Vec<f64> {
    let gamma = 2f64.sqrt();
    let k0 = 2.0 * std::f64::consts::PI / n as f64;
    let mut state = seeded(seed);
    let mut waves: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut k = 0i32;
    loop {
        let freq = k0 * gamma.powi(k);
        if freq > std::f64::consts::PI {
            break;
        }
        let amp = gamma.powf(-f64::from(k) * h);
        for _ in 0..4 {
            let ang = draw(&mut state) * std::f64::consts::TAU;
            let phase = draw(&mut state) * std::f64::consts::TAU;
            waves.push((freq * ang.cos(), freq * ang.sin(), phase, amp));
        }
        k += 1;
    }
    let mut out = vec![0.0f64; n * n];
    for (idx, v) in out.iter_mut().enumerate() {
        let (x, y) = ((idx % n) as f64, (idx / n) as f64);
        *v = waves
            .iter()
            .map(|&(kx, ky, ph, a)| a * (kx * x + ky * y + ph).cos())
            .sum();
    }
    out
}

/// ★ CHECK-D-STAIR's field: the union of `step × step` blocks whose centres stand under a line at
/// `bearing` degrees through the middle, turned by `turn` degrees: −1 inside, +1 outside. A
/// staircase of axis steps at `turn = 0`, of diagonal steps at `turn = 45`.
#[must_use]
pub fn staircase_field(n: usize, step: f64, bearing: f64, turn: f64) -> Vec<Option<f64>> {
    let mid = n as f64 / 2.0;
    let (st, ct) = (-turn).to_radians().sin_cos();
    let slope = bearing.to_radians().tan();
    (0..n * n)
        .map(|k| {
            let (x, y) = ((k % n) as f64 - mid, (k / n) as f64 - mid);
            let (u, v) = (ct * x - st * y, st * x + ct * y);
            let (bu, bv) = (
                ((u / step).floor() + 0.5) * step,
                ((v / step).floor() + 0.5) * step,
            );
            Some(if bv < slope * bu { -1.0 } else { 1.0 })
        })
        .collect()
}

/// ★ CHECK-LAKE-SQUARES's field: `count` axis-aligned squares of random sides from `lo` to `hi`
/// cells laid on a grid without touching: −1 inside, +1 outside, and the squares' cell sets.
#[must_use]
pub fn squares_field(n: usize, count: usize, lo: f64, hi: f64, seed: u64) -> Vec<Option<f64>> {
    let mut out = vec![Some(1.0); n * n];
    let per_row = (count as f64).sqrt().ceil() as usize;
    let slot = n / per_row.max(1);
    let mut state = seeded(seed);
    for k in 0..count {
        let side = (lo + draw(&mut state) * (hi - lo)).min(slot as f64 - 4.0) as usize;
        let (sa, sb) = ((k % per_row) * slot + 2, (k / per_row) * slot + 2);
        for b in sb..(sb + side).min(n) {
            for a in sa..(sa + side).min(n) {
                out[b * n + a] = Some(-1.0);
            }
        }
    }
    out
}

/// ★ THE PERIMETER–AREA SLOPE of a field's closed loops: each closed zero-contour loop of at least
/// `min_side²` in area, the OLS slope of `ln P` on `ln A` (so `d = 2·slope`), and the count.
#[must_use]
pub fn perimeter_area_slope(n: usize, field: &[Option<f64>], min_side: f64) -> (f64, usize) {
    let (mut lx, mut ly) = (Vec::new(), Vec::new());
    for l in zero_contours(n, field) {
        if l.first() != l.last() {
            continue;
        }
        let a = shoelace(&l).abs();
        if a < min_side * min_side {
            continue;
        }
        lx.push(a.ln());
        ly.push(polyline_length(&l).ln());
    }
    (ols(&lx, &ly).0, lx.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ CHECK-D-LINE: a straight coast at 17° reads D = 1.00 ± 0.01 through the tracer.
    #[test]
    fn a_straight_line_reads_one() {
        let n = 1_024;
        let lines = zero_contours(n, &line_field(n, 17.0));
        let d = divider_dimension(&lines, &rulers(8.0, 102.0, 2));
        assert!((d - 1.0).abs() < 0.01, "D {d}");
    }

    /// ★ CHECK-D-KOCH: the snowflake reads 1.262 ± 0.02 on the polygon itself and through the
    /// tracer over rulers of 8 to 204 cells.
    #[test]
    fn the_koch_snowflake_reads_its_dimension() {
        let n = 2_048;
        let poly = koch_snowflake(6, 1_500.0, 1_024.0, 1_024.0);
        let eps = rulers(8.0, 204.0, 4);
        let d_poly = divider_dimension(std::slice::from_ref(&poly), &eps);
        assert!((d_poly - 1.262).abs() < 0.02, "polygon D {d_poly}");
        let lines = zero_contours(n, &fill_polygon(&poly, n));
        let longest = lines
            .into_iter()
            .max_by(|a, b| polyline_length(a).total_cmp(&polyline_length(b)))
            .expect("a loop");
        let d_traced = divider_dimension(&[longest], &eps);
        assert!((d_traced - 1.262).abs() < 0.02, "traced D {d_traced}");
    }

    /// ★ THE LATTICE MOMENT: a staircase of axis steps reads a clearly positive L4, its 45° turn a
    /// clearly negative one; axes turned by 45° read −L4 exactly.
    #[test]
    fn a_staircase_leans_on_the_lattice_and_its_turn_flips_the_sign() {
        let n = 1_024;
        let axis = zero_contours(n, &staircase_field(n, 64.0, 30.0, 0.0));
        let diag = zero_contours(n, &staircase_field(n, 64.0, 30.0, 45.0));
        let (l4a, _) = four_fold(&chord_axes(&axis, 8.0));
        let (l4d, _) = four_fold(&chord_axes(&diag, 8.0));
        assert!(l4a > 0.5, "axis staircase L4 {l4a}");
        assert!(l4d < -0.3, "diagonal staircase L4 {l4d}");
        let turned = rotated(&axis, 45.0);
        let (l4t, _) = four_fold(&chord_axes(&turned, 8.0));
        assert!((l4t + l4a).abs() < 1.0e-9, "{l4t} vs {l4a}");
    }

    /// ★ CHECK-LAKE-SQUARES: fifty squares read a perimeter–area slope of 0.50 ± 0.01.
    #[test]
    fn squares_read_a_slope_of_one_half() {
        let n = 2_048;
        let (slope, count) = perimeter_area_slope(n, &squares_field(n, 50, 12.0, 200.0, 7), 10.0);
        assert_eq!(count, 50);
        assert!((slope - 0.5).abs() < 0.01, "slope {slope}");
    }

    /// The divider on a unit square loop reads its perimeter at a ruler that divides the side.
    #[test]
    fn the_divider_walks_a_square() {
        let sq = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ];
        let (len, steps) = divider(&sq, 5.0);
        assert_eq!(steps, 8);
        assert!((len - 40.0).abs() < 1.0e-9);
    }

    /// Kendall's τ: a monotone pair reads one, a reversed pair minus one.
    #[test]
    fn kendall_reads_order() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert!((kendall_tau(&x, &[2.0, 3.0, 5.0, 9.0]) - 1.0).abs() < 1.0e-12);
        assert!((kendall_tau(&x, &[9.0, 5.0, 3.0, 2.0]) + 1.0).abs() < 1.0e-12);
    }

    /// Two touching blocks of two keys are two components.
    #[test]
    fn components_split_by_key() {
        let key = [1, 1, 2, 2, 0, 0, 2, 0, 1];
        let (_, comps) = components(3, 3, &key);
        assert_eq!(comps.len(), 4);
    }

    /// The flat null of |L4| falls with the count.
    #[test]
    fn the_l4_null_falls_with_the_count() {
        assert!(flat_null_l4(20, 500, 0.95) > flat_null_l4(2_000, 500, 0.95));
    }
}
