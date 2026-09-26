//! ★ SHORE-STEP: HOW FAR THE DRAWN SHORE MOVES SIDEWAYS AT EACH RING SWAP (plan §8.2; T7, W17.1;
//! survey §4.2: every published error measure limits a HEIGHT, and a shore moves sideways by that
//! height over the slope, so a gentle coast is the decisive stand).
//!
//! Transects stand normal to C0's `B = W` contour at rung 7 in W2 (the ground without its fine
//! octaves meeting the water's flat level), at most 128 per water kind, spaced at least 1 km. Along
//! each, the drawn field `v = s − L_c` is sampled bilinearly every quarter cell of each rung; its
//! sign changes are the drawn shore's crossings. Each coarser crossing takes the nearest finer
//! crossing within four coarser cells; the distance, over one finer cell, is the step in pixels at
//! the swap (one pixel there is one cell of the finer rung). A coarser crossing with no finer one
//! near it is a FOLD event, counted apart.
//!
//! **Example.** On the gentle coast a transect crosses the sea's line at rung 9 at +140 m and at
//! rung 10 at +830 m: the 9 → 10 swap moves the drawn shore 690 m, 1.35 pixels of 512 m. The
//! census pools every transect's step per swap and judges p99 ≤ 1 px and max ≤ 2 px.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use vd_bins::shore_geometry::zero_contours;
use vd_bins::shore_rules::{NodeSite, Nodes, Water};

use crate::census::{Census, val};
use crate::eval::{Caches, Checks, ColOut, Ctx, Fold, FoldMap, NVAR, RungOut, Var};
use crate::pyramid::Pyramid;

/// One transect: its centre in rung-0 cells (continuous), its unit normal in metres (along `i`,
/// `j`), its half-length in metres, and its water kind (0 the sea, 1 the lakes).
#[derive(Clone, Copy, Debug)]
pub struct Transect {
    pub c0: (f64, f64),
    pub normal: (f64, f64),
    pub half_m: f64,
    pub kind: usize,
}

/// ★ THE TRANSECTS OF ONE STAND from W2's rung-7 ground: the zero contour of `B − W` for the sea
/// and for each lake level, over the columns whose nearest node stands within two nodes of a wet
/// node of that water, then points along the contours at the stated spacing.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the transects read the window's own words"
)]
pub fn place(
    w2: &RungOut,
    fine: &vd_terrain::macro_lattice::MacroLattice,
    nodes: &Nodes,
    face: vd_seed::bend::Face,
    m_i: f64,
    m_j: f64,
    ch: &vd_bins::shore_rules::NamedChoices,
) -> Vec<Transect> {
    let n = w2.n;
    let rung = w2.rung;
    let cells = f64::from(1u32 << rung);
    // The waters within two nodes of each column's nearest node (memoised per node).
    let mut memo: HashMap<u32, Vec<Water>> = HashMap::new();
    let mut waters_at: Vec<Vec<Water>> = Vec::with_capacity(n * n);
    for k in 0..n * n {
        let (a, b) = ((k % n) as i32 + w2.i0, (k / n) as i32 + w2.j0);
        let site = NodeSite::of(fine, face, rung, a, b);
        let near = site.nearest(fine);
        let list = memo
            .entry(near)
            .or_insert_with(|| {
                let (f, ni, nj) = fine.split(near);
                let mut out: Vec<Water> = Vec::new();
                for dj in -2..=2i64 {
                    for di in -2..=2i64 {
                        let m = vd_terrain::artifact::node_at(
                            fine,
                            f,
                            i64::from(ni) + di,
                            i64::from(nj) + dj,
                        );
                        if let Some(w) = nodes.water_of(m)
                            && !out.contains(&w)
                        {
                            out.push(w);
                        }
                    }
                }
                out
            })
            .clone();
        waters_at.push(list);
    }
    let mut kinds: [Vec<Contour>; 2] = [Vec::new(), Vec::new()];
    // The sea over the whole window; each lake over its own box.
    let mut levels: BTreeMap<i16, (usize, usize, usize, usize)> = BTreeMap::new();
    for (k, list) in waters_at.iter().enumerate() {
        for w in list {
            if let Water::Lake(l) = w {
                let (a, b) = (k % n, k / n);
                let e = levels.entry(*l).or_insert((a, b, a, b));
                e.0 = e.0.min(a);
                e.1 = e.1.min(b);
                e.2 = e.2.max(a);
                e.3 = e.3.max(b);
            }
        }
    }
    let b_at = |k: usize| w2.cols[k].as_ref().map(|c| f64::from(c.shared.b));
    if nodes.sea_m != vd_terrain::artifact::DRY_M {
        let sea = f64::from(nodes.sea_m);
        let field: Vec<Option<f64>> = (0..n * n)
            .map(|k| {
                if waters_at[k].contains(&Water::Sea) {
                    b_at(k).map(|b| b - sea)
                } else {
                    None
                }
            })
            .collect();
        for l in zero_contours(n, &field) {
            kinds[0].push((l, ()));
        }
    }
    for (&level, &(a0, b0, a1, b1)) in &levels {
        let (a0, b0) = (a0.saturating_sub(1), b0.saturating_sub(1));
        let side = (a1 + 2 - a0).max(b1 + 2 - b0).min(n);
        let w = Water::Lake(level);
        let field: Vec<Option<f64>> = (0..side * side)
            .map(|k| {
                let (a, b) = (a0 + k % side, b0 + k / side);
                if a >= n || b >= n {
                    return None;
                }
                let g = b * n + a;
                if waters_at[g].contains(&w) {
                    b_at(g).map(|bb| bb - f64::from(level))
                } else {
                    None
                }
            })
            .collect();
        for l in zero_contours(side, &field) {
            let shifted: Vec<(f64, f64)> = l
                .into_iter()
                .map(|p| (p.0 + a0 as f64, p.1 + b0 as f64))
                .collect();
            kinds[1].push((shifted, ()));
        }
    }
    // Points along the contours, at the spacing, in metres.
    let mut out = Vec::new();
    for (kind, lines) in kinds.iter().enumerate() {
        let metric = |p: (f64, f64)| (p.0 * m_i, p.1 * m_j);
        let total: f64 = lines
            .iter()
            .map(|(l, ())| {
                l.windows(2)
                    .map(|w| {
                        let (p, q) = (metric(w[0]), metric(w[1]));
                        (q.0 - p.0).hypot(q.1 - p.1)
                    })
                    .sum::<f64>()
            })
            .sum();
        if total <= 0.0 {
            continue;
        }
        let spacing = ch.transect_spacing_m.max(total / ch.transects as f64);
        let half_m = if kind == 0 {
            ch.half_sea_nodes
        } else {
            ch.half_lake_nodes
        } * fine.node_m();
        let mut carry = 0.5 * spacing;
        for (l, ()) in lines {
            for w in l.windows(2) {
                let (p, q) = (metric(w[0]), metric(w[1]));
                let seg = (q.0 - p.0).hypot(q.1 - p.1);
                if seg <= 0.0 {
                    continue;
                }
                let mut at = carry;
                while at <= seg {
                    if out.iter().filter(|t: &&Transect| t.kind == kind).count() >= ch.transects {
                        break;
                    }
                    let u = at / seg;
                    let (a, b) = (
                        w[0].0 + u * (w[1].0 - w[0].0),
                        w[0].1 + u * (w[1].1 - w[0].1),
                    );
                    let t = ((q.0 - p.0) / seg, (q.1 - p.1) / seg);
                    out.push(Transect {
                        c0: (
                            (f64::from(w2.i0) + a + 0.5) * cells,
                            (f64::from(w2.j0) + b + 0.5) * cells,
                        ),
                        normal: (-t.1, t.0),
                        half_m,
                        kind,
                    });
                    at += spacing;
                }
                carry = at - seg;
            }
        }
    }
    out
}

/// One contour line and nothing else (a named type, so the kinds' list reads plainly).
type Contour = (Vec<(f64, f64)>, ());

/// The crossings of one transect at one rung for every rule: `t` in metres along the normal.
type Crossings = [Vec<f64>; NVAR];

/// ★ ONE TRANSECT'S CROSSINGS at every rung, coarse to fine: full sampling from `full_from` up, and
/// under it only windows round the coarser rung's crossings.
#[allow(
    clippy::too_many_arguments,
    reason = "the walk reads the stand's own words"
)]
fn walk(
    tr: &Transect,
    ctxs: &[Option<Ctx>],
    fold: &Fold,
    m_i: f64,
    m_j: f64,
    full_from: u8,
    window_cells: f64,
    tail_cells: f64,
    caches: &mut [HashMap<(i32, i32), Option<ColOut>>],
    shared: &mut Caches,
) -> Vec<Option<Crossings>> {
    let rungs = ctxs.len();
    let mut out: Vec<Option<Crossings>> = vec![None; rungs];
    let (mi0, mj0) = (m_i / 128.0, m_j / 128.0);
    for r in (0..rungs).rev() {
        let Some(ctx) = ctxs[r].as_ref() else {
            continue;
        };
        let cell = f64::from(1u32 << r);
        let step = cell / 4.0;
        // The spans of t to sample.
        let spans: Vec<(f64, f64)> = if r as u8 >= full_from || r + 1 >= rungs {
            let tail = tail_cells * f64::from(1u32 << (r + 1));
            vec![(-tr.half_m - tail, tr.half_m + tail)]
        } else {
            let Some(coarse) = out[r + 1].as_ref() else {
                continue;
            };
            let reach = window_cells * f64::from(1u32 << (r + 1));
            let mut spans: Vec<(f64, f64)> = coarse
                .iter()
                .flatten()
                .map(|&x| ((x - reach).max(-tr.half_m), (x + reach).min(tr.half_m)))
                .collect();
            spans.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut merged: Vec<(f64, f64)> = Vec::new();
            for s in spans {
                match merged.last_mut() {
                    Some(m) if s.0 <= m.1 => m.1 = m.1.max(s.1),
                    _ => merged.push(s),
                }
            }
            merged
        };
        let mut xs: Crossings = std::array::from_fn(|_| Vec::new());
        let cache = &mut caches[r];
        let n_cells = ctx.body.ladder().cells_per_edge(ctx.rung) as i32;
        for (lo, hi) in spans {
            let count = ((hi - lo) / step).floor() as i64;
            let mut prev: Option<(f64, [f64; NVAR])> = None;
            for s in 0..=count {
                let t = lo + s as f64 * step;
                let (u, w) = (
                    tr.c0.0 + t * tr.normal.0 / mi0,
                    tr.c0.1 + t * tr.normal.1 / mj0,
                );
                let (x, y) = (u / cell - 0.5, w / cell - 0.5);
                let (ia, jb) = (x.floor() as i32, y.floor() as i32);
                let (fx, fy) = (x - f64::from(ia), y - f64::from(jb));
                let mut vals = [0.0f64; NVAR];
                let mut ok = ia >= 1 && jb >= 1 && ia + 2 < n_cells && jb + 2 < n_cells;
                if ok {
                    'corners: for (da, db, wgt) in [
                        (0, 0, (1.0 - fx) * (1.0 - fy)),
                        (1, 0, fx * (1.0 - fy)),
                        (0, 1, (1.0 - fx) * fy),
                        (1, 1, fx * fy),
                    ] {
                        let Some(v) = column_v(ctx, ia + da, jb + db, fold, cache, shared) else {
                            ok = false;
                            break 'corners;
                        };
                        for k in 0..NVAR {
                            vals[k] += wgt * v[k];
                        }
                    }
                }
                if !ok {
                    prev = None;
                    continue;
                }
                if let Some((pt, pv)) = prev {
                    for k in 0..NVAR {
                        if (pv[k] >= 0.0) != (vals[k] >= 0.0) {
                            let f = pv[k] / (pv[k] - vals[k]);
                            xs[k].push(pt + f * (t - pt));
                        }
                    }
                }
                prev = Some((t, vals));
            }
        }
        out[r] = Some(xs);
    }
    out
}

/// A column's drawn field for every rule, `v = s − L_c` (+1 where no water stands in its 3 × 3).
fn column_v(
    ctx: &Ctx,
    i: i32,
    j: i32,
    fold: &Fold,
    cache: &mut HashMap<(i32, i32), Option<ColOut>>,
    shared: &mut Caches,
) -> Option<[f64; NVAR]> {
    let mut get = |a: i32, b: i32| -> Option<ColOut> {
        *cache.entry((a, b)).or_insert_with(|| {
            let mut ck = Checks::default();
            ctx.eval(a, b, fold, shared, &mut ck)
        })
    };
    let me = get(i, j)?;
    let mut lc = [f32::NAN; NVAR];
    for db in -1..=1 {
        for da in -1..=1 {
            if let Some(o) = get(i + da, j + db) {
                for (best, d) in lc.iter_mut().zip(o.drawn.iter()) {
                    let l = d.l;
                    if !l.is_nan() && (best.is_nan() || l > *best) {
                        *best = l;
                    }
                }
            }
        }
    }
    Some(std::array::from_fn(|k| {
        if lc[k].is_nan() {
            1.0
        } else {
            f64::from(me.drawn[k].s) - f64::from(lc[k])
        }
    }))
}

/// The worst matched step of one swap: its size in pixels, its transect, and the coarser and the
/// finer crossing along it (metres from the transect's centre).
pub type Worst = (f64, usize, f64, f64);

/// Per water kind, rule and swap: the steps in pixels, the matched and the unmatched counts, and
/// the worst matched step.
pub type Steps = BTreeMap<(usize, usize, u8), (Vec<f64>, usize, usize, Worst)>;

/// ★ EVERY TRANSECT OF A STAND, over `threads` threads (each with its own column caches): the steps
/// at every swap for every rule. `tail` (the far window's transects) samples each fully read rung
/// `match_cells` coarser cells past both ends and matches only the coarser crossings inside the
/// transect, so a crossing at an end never takes a far finer one for want of its own.
#[allow(clippy::too_many_arguments, reason = "the stand's own words")]
pub fn measure(
    transects: &[Transect],
    ctxs: &[Option<Ctx>],
    map7: Option<&FoldMap>,
    pyramid: Option<&Pyramid>,
    m_i: f64,
    m_j: f64,
    m_col7: f64,
    ch: &vd_bins::shore_rules::NamedChoices,
    threads: usize,
    tail: bool,
) -> Steps {
    let rungs = ctxs.len();
    let ctx_f = ctxs.get(7).and_then(Option::as_ref);
    let fold = Fold {
        map: map7,
        ctx_f,
        pyramid,
    };
    let next = AtomicUsize::new(0);
    let result: Mutex<Steps> = Mutex::new(BTreeMap::new());
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let mut caches: Vec<HashMap<(i32, i32), Option<ColOut>>> =
                    (0..rungs).map(|_| HashMap::new()).collect();
                let mut shared = Caches::default();
                let mut local: Steps = BTreeMap::new();
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= transects.len() {
                        break;
                    }
                    let tr = &transects[k];
                    let xs = walk(
                        tr,
                        ctxs,
                        &fold,
                        m_i,
                        m_j,
                        ch.full_from,
                        ch.window_cells,
                        if tail { ch.match_cells } else { 0.0 },
                        &mut caches,
                        &mut shared,
                    );
                    for r in 0..rungs.saturating_sub(1) {
                        let (Some(fine), Some(coarse)) = (xs[r].as_ref(), xs[r + 1].as_ref())
                        else {
                            continue;
                        };
                        let px = m_col7 * 2f64.powi(r as i32 - 7);
                        let reach = ch.match_cells * f64::from(1u32 << (r + 1));
                        for v in 0..NVAR {
                            let e = local.entry((tr.kind, v, r as u8)).or_default();
                            for &x in &coarse[v] {
                                if tail && x.abs() > tr.half_m {
                                    continue;
                                }
                                let (near, y) = fine[v].iter().map(|&y| ((y - x).abs(), y)).fold(
                                    (f64::INFINITY, f64::NAN),
                                    |a, b| {
                                        if b.0 < a.0 { b } else { a }
                                    },
                                );
                                if near <= reach {
                                    e.0.push(near / px);
                                    e.1 += 1;
                                    if near / px > e.3.0 {
                                        e.3 = (near / px, k, x, y);
                                    }
                                } else {
                                    e.2 += 1;
                                }
                            }
                        }
                    }
                    // A thread's caches grow with its transects; the far rungs repay them, the
                    // fine ones do not.
                    for c in caches.iter_mut().take(5) {
                        c.clear();
                    }
                }
                if let Ok(mut r) = result.lock() {
                    for (key, (steps, m, u, worst)) in local {
                        let e = r.entry(key).or_default();
                        e.0.extend(steps);
                        e.1 += m;
                        e.2 += u;
                        if worst.0 > e.3.0 {
                            e.3 = worst;
                        }
                    }
                }
            });
        }
    });
    let mut steps = result.into_inner().unwrap_or_default();
    for e in steps.values_mut() {
        e.0.sort_by(f64::total_cmp);
    }
    steps
}

/// ★ THE SHORE-STEP LINES of one stand over the swaps `swaps` (`r` is the swap `r → r + 1`);
/// returns, per rule, the largest step at the swaps 0 → 1 to `last_swap` (the deciding number's
/// own reading) with its p99, over the sea kind.
#[allow(clippy::too_many_arguments, reason = "the stand's own words")]
pub fn print(
    c: &mut Census,
    stand: &str,
    steps: &Steps,
    transects: &[Transect],
    vars: &[Var],
    f_rung: u8,
    last_swap: u8,
    swaps: std::ops::Range<u8>,
) -> BTreeMap<usize, (f64, f64, usize)> {
    let mut decide: BTreeMap<usize, (f64, f64, usize)> = BTreeMap::new();
    for kind in 0..2usize {
        let kname = if kind == 0 { "sea" } else { "lake" };
        for &v in vars {
            let vi = v.idx();
            for r in swaps.clone() {
                if v.nofold() && r < f_rung {
                    continue;
                }
                let Some((s, matched, unmatched, worst)) = steps.get(&(kind, vi, r)) else {
                    continue;
                };
                let p = |q: f64| {
                    if s.is_empty() {
                        f64::NAN
                    } else {
                        s[((s.len() as f64 * q) as usize).min(s.len() - 1)]
                    }
                };
                let (p50, p99, max) = (p(0.5), p(0.99), p(1.0));
                let gate = format!(
                    "SHORE-STEP.{stand}.{kname}.{}.r{r:02}-r{:02}",
                    v.name(),
                    r + 1
                );
                let value = format!(
                    "p50{}/p99{}/max{}px/{matched}matched/{unmatched}fold",
                    val(p50, 3, ""),
                    val(p99, 3, ""),
                    val(max, 3, "")
                );
                let band = "p99<=1px,max<=2px,>=200pairs[T7;W17.1]";
                if *matched < 200 {
                    c.line(&gate, &value, band, "NO-BAND");
                } else {
                    c.judged(&gate, &value, band, p99 <= 1.0 && max <= 2.0);
                }
                // The worst step's own transect, so a failing swap can be probed.
                if worst.0 > 2.0
                    && let Some(t) = transects.get(worst.1)
                {
                    c.say(format!(
                        "   worst {gate}: {:.2} px on transect #{} (centre ({:.0}, {:.0}) rung-0 cells, normal ({:.3}, {:.3})): coarser crossing {:.0} m, finer {:.0} m",
                        worst.0, worst.1, t.c0.0, t.c0.1, t.normal.0, t.normal.1, worst.2, worst.3
                    ));
                }
                if kind == 0 && r < last_swap && max.is_finite() {
                    let e = decide.entry(vi).or_insert((0.0, 0.0, 0));
                    e.0 = e.0.max(max);
                    e.1 = e.1.max(p99);
                    e.2 += matched;
                }
            }
        }
    }
    decide
}
