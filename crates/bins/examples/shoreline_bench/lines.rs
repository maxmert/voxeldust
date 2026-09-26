//! ★ THE LINES, EVERY ONE ON THE DRAWN SET (plan §8.1–§8.3, §8.5): per window, rung and rule; and
//! across the rungs of one window (§8.2's APPEAR, FLICKER and FOLD-RETURN).
//!
//! "Drawn wet" is the sheet's own rule, `s < l` (`vd_terrain::position::water_sheet_sided`), never
//! a rule's own side word; the side word is read only by the lines that compare the two.
//!
//! **Example.** On the plain's W2 at rung 7 under R0, a dry column beside a pond stands 40 m under
//! the pond's level while the sheet over its 3 × 3 stands at that level: G-SEA-OVER-DRY counts it,
//! because the owner would see water over ground the rule calls dry.

use std::collections::VecDeque;

use vd_bins::shore_geometry::zero_contours;
use vd_bins::shore_rules::{NodeSite, Nodes, WORD_POCKET, Water, shore_f64};
use vd_bins::shore_shape::{
    Point, chord_axes, components, flat_null_l4, four_fold, kendall_tau, ols, outer_loop,
    polyline_length, quantile, rotated, shifted, shoelace, slopes_per_octave,
};
use vd_recipe::height::SIDE_LAND;
use vd_seed::bend::Face;
use vd_terrain::macro_lattice::MacroLattice;

use crate::census::{Census, val};
use crate::eval::{BINDS, HALO, NVAR, RAMP, RungOut, Var};

/// What every line of one window's rung needs beside the columns.
pub struct LineCtx<'a> {
    pub stand: &'a str,
    pub window: &'a str,
    pub rung: u8,
    pub face: Face,
    /// The column's metres along `i` and `j` at the window's middle, and their mean.
    pub m_i: f64,
    pub m_j: f64,
    pub m_col: f64,
    /// The extractor's step at this rung, `cell / 128`.
    pub q_r: f64,
    pub sea: Option<f64>,
    pub pocket_stand: bool,
    pub fine: &'a MacroLattice,
    pub nodes: &'a Nodes,
    pub vars: &'a [Var],
    pub f_rung: u8,
    pub wall_rung: bool,
    pub shape_rung: bool,
    pub choices: vd_bins::shore_rules::NamedChoices,
    /// KINK's bands from the fractional surfaces, by the node's size in columns (none where the
    /// staircase control showed the gate has no power).
    pub kink_band: &'a dyn Fn(usize) -> Option<f64>,
    /// G-SHORE-LATTICE's band from the fractional surfaces, by the window's side in columns.
    pub lattice_band: &'a dyn Fn(usize) -> Option<f64>,
    /// The shape estimators are void (a failed CHECK-D-LINE or CHECK-D-KOCH).
    pub shape_void: bool,
}

impl LineCtx<'_> {
    fn gate(&self, g: &str, v: Var) -> String {
        format!(
            "{g}.{}.{}.{}.r{:02}",
            self.stand,
            self.window,
            v.name(),
            self.rung
        )
    }
    fn gate_rung(&self, g: &str) -> String {
        format!("{g}.{}.{}.r{:02}", self.stand, self.window, self.rung)
    }
    /// Whether a variant's line is printed at this rung: the no-fold twins only over rung F.
    fn prints(&self, v: Var) -> bool {
        !v.nofold() || self.rung > self.f_rung
    }
}

/// The pooled lakes of every stand, per rule and rung: `(ln A, ln P)` for G-LAKE-FRACTAL.
pub type Pool = std::collections::BTreeMap<(usize, u8), Vec<(f64, f64)>>;

/// The 4-neighbour pairs `(p, q)` of an `n × n` window, each once.
fn pairs(n: usize) -> impl Iterator<Item = (usize, usize, bool)> {
    (0..n * n).flat_map(move |k| {
        let (a, b) = (k % n, k / n);
        let right = (a + 1 < n).then_some((k, k + 1, false));
        let up = (b + 1 < n).then_some((k, k + n, true));
        right.into_iter().chain(up)
    })
}

/// ★ THE SHEET'S LEVEL OVER EACH COLUMN, `L_c`: the highest standing water over its 3 × 3, and
/// that water's key (plan §6: a vertex's level is the highest of its columns').
fn sheet_level(out: &RungOut, v: usize) -> (Vec<f32>, Vec<u32>) {
    let n = out.n;
    let mut lc = vec![f32::NAN; n * n];
    let mut kc = vec![0u32; n * n];
    for k in 0..n * n {
        let (a, b) = ((k % n) as i64, (k / n) as i64);
        let mut best = f32::NAN;
        let mut key = 0;
        for db in -1..=1i64 {
            for da in -1..=1i64 {
                let (x, y) = (a + da, b + db);
                if x < 0 || y < 0 || x >= n as i64 || y >= n as i64 {
                    continue;
                }
                let Some(c) = out.cols[(y as usize) * n + x as usize].as_ref() else {
                    continue;
                };
                let d = c.drawn[v];
                if !d.l.is_nan() && (best.is_nan() || d.l > best) {
                    best = d.l;
                    key = d.key;
                }
            }
        }
        lc[k] = best;
        kc[k] = key;
    }
    (lc, kc)
}

/// The traced drawn shore of one water kind (the sea: key 1; the lakes: keys over 1) in the
/// window's metric (a unit is one mean column), from `v_c = s − L_c`.
fn traces(
    out: &RungOut,
    v: usize,
    lc: &[f32],
    kc: &[u32],
    sea: bool,
    lx: &LineCtx,
) -> Vec<Vec<Point>> {
    let n = out.n;
    let field: Vec<Option<f64>> = (0..n * n)
        .map(|k| {
            let c = out.cols[k].as_ref()?;
            let mine = if sea { kc[k] == 1 } else { kc[k] > 1 };
            Some(if mine && !lc[k].is_nan() {
                f64::from(c.drawn[v].s) - f64::from(lc[k])
            } else {
                1.0
            })
        })
        .collect();
    let (sx, sy) = (lx.m_i / lx.m_col, lx.m_j / lx.m_col);
    zero_contours(n, &field)
        .into_iter()
        .map(|l| l.into_iter().map(|p| (p.0 * sx, p.1 * sy)).collect())
        .collect()
}

/// ★ THE PER-RUNG CHECK LINES of one window (not per rule).
pub fn check_lines(c: &mut Census, lx: &LineCtx, out: &RungOut, sheet: Option<[usize; 7]>) {
    let k = &out.checks;
    c.judged(
        &lx.gate_rung("CHECK-R0"),
        &format!("{}of{}cols/{}chunks", k.r0_differ, k.r0_cols, k.r0_chunks),
        "0[plan§4.1;the-shipped-path-bit-for-bit]",
        k.r0_differ == 0 && k.r0_cols > 0,
    );
    c.line(
        &lx.gate_rung("CHECK-R0-CONTROL"),
        &format!("{}cols", k.control_r0),
        ">0[the-card's-side-word-must-differ]",
        if k.control_r0 > 0 { "GREEN" } else { "NO-BAND" },
    );
    c.judged(
        &lx.gate_rung("CHECK-CACHE-FIELD"),
        &format!("{}of{}cols", k.cache_differ, k.cache_cols),
        "0[plan§4.1;the-cache's-field]",
        k.cache_differ == 0 && k.cache_cols > 0,
    );
    c.judged(
        &lx.gate_rung("CHECK-CACHE-CONTROL"),
        &format!("{}of{}cols", k.control_cache, k.control_cache_cols),
        ">0[the-next-level-must-differ]",
        k.control_cache > 0,
    );
    c.judged(
        &lx.gate_rung("BASECHECK"),
        &format!("{}of{}cols", k.columns - k.base_disagree, k.columns),
        "=all[drift-alarm]",
        k.base_disagree == 0,
    );
    c.judged(
        &lx.gate_rung("CHECK-FINE-SPLIT"),
        &format!("{}of{}cols", k.fine_split_bad, k.fine_split_checked),
        "0[plan§4.1]",
        k.fine_split_bad == 0,
    );
    c.judged(
        &lx.gate_rung("CHECK-R0-F64"),
        &val(k.f64_max, 6, "m"),
        &format!("<={}m[named]", lx.choices.f64_band_m),
        k.f64_max <= lx.choices.f64_band_m,
    );
    c.judged(
        &lx.gate_rung("CHECK-SHORE-CANDIDATES-OVER-4"),
        &format!("{}cols", k.over_cap),
        "0[finding-23]",
        k.over_cap == 0,
    );
    c.reading(
        &lx.gate_rung("CANDIDATES-HIST"),
        &format!(
            "0:{}/1:{}/2:{}/3:{}/4:{}/5+:{}",
            k.hist[0], k.hist[1], k.hist[2], k.hist[3], k.hist[4], k.hist[5]
        ),
        "a-reading",
    );
    if k.fold_cols > 0 {
        c.reading(
            &lx.gate_rung("FOLD-RULES-DIFFER"),
            &format!("{}of{}cols", k.fold_differ, k.fold_cols),
            "N2's-own-samples:the-record-(most-samples)-vs-the-one-rule-(lake-share,highest-level)[S0-item-4;Q7]",
        );
    }
    if k.unread > 0 {
        c.say(format!(
            "   NOTE: {} columns of the window were not held by the field (the reader answered none)",
            k.unread
        ));
    }
    if let Some([cols, cdiff, verts, vdiff, vties, control, boxes]) = sheet {
        c.judged(
            &lx.gate_rung("CHECK-SHEET-COLUMNS"),
            &format!("{cdiff}of{cols}cols/{vdiff}of{verts}verts({vties}ties)/{boxes}boxes"),
            "0[plan§4.1;water_sheet_sided]",
            cdiff == 0 && vdiff == vties && cols > 0,
        );
        c.judged(
            &lx.gate_rung("CHECK-SHEET-CONTROL"),
            &format!("{control}cols"),
            ">0[the-neighbour-chunk-must-differ]",
            control > 0,
        );
    }
}

/// ★ THE PER-RUNG LINES OF EVERY RULE (plan §8.1, §8.2 WALL/CLAMP/POCKET/PERCHED, §8.3, §8.5).
pub fn rule_lines(c: &mut Census, lx: &LineCtx, out: &RungOut, pool: &mut Pool) {
    let n = out.n;
    let q = lx.q_r;
    // WALL's reference: the raw ground's own neighbour steps in the window.
    let q_wall = if lx.wall_rung {
        let steps: Vec<f64> = pairs(n)
            .filter_map(|(p, qq, _)| {
                let (a, b) = (out.cols[p].as_ref()?, out.cols[qq].as_ref()?);
                Some(f64::from((a.shared.h_raw - b.shared.h_raw).abs()))
            })
            .collect();
        quantile(&steps, lx.choices.wall_quantile)
    } else {
        f64::NAN
    };
    // C0's lattice moments, beside every rule's.
    let mut c0_l4: [f64; 2] = [f64::NAN; 2];
    for &v in lx.vars {
        if !lx.prints(v) {
            continue;
        }
        let vi = v.idx();
        let (lc, kc) = sheet_level(out, vi);
        // ── §8.1 ────────────────────────────────────────────────────────────────────────────
        let (mut mism, mut ctl, mut over_dry, mut sides) = (0usize, 0usize, 0usize, 0usize);
        // The mismatches split: side wet but drawn dry (a tie `s = l` included), side dry but drawn
        // wet; and the first few, for the reader.
        let (mut wet_not_drawn, mut drawn_not_wet, mut ties) = (0usize, 0usize, 0usize);
        let mut first: Vec<String> = Vec::new();
        for (k, col) in out.cols.iter().enumerate() {
            let Some(col) = col else { continue };
            let d = col.drawn[vi];
            if d.side() != d.wet() {
                if d.side() {
                    wet_not_drawn += 1;
                    ties += usize::from(d.s == d.l);
                } else {
                    drawn_not_wet += 1;
                }
                if first.len() < 3 {
                    first.push(format!(
                        "({}, {}) s {:.2} l {:.2} word {} B {:.1}",
                        out.i0 + (k % n) as i32,
                        out.j0 + (k / n) as i32,
                        d.s,
                        d.l,
                        col.shared.word,
                        col.shared.b
                    ));
                }
            }
            mism += usize::from(d.side() != d.wet());
            if v == Var::R0 && !col.shared.hold.is_nan() {
                ctl += usize::from(d.side() != (d.s < col.shared.hold));
            }
            sides += usize::from(d.side());
            if !d.side() && !lc[k].is_nan() && f64::from(d.s) < f64::from(lc[k]) - q {
                over_dry += 1;
            }
        }
        c.judged(
            &lx.gate("CHECK-DRAWN-IS-SIDE", v),
            &format!("{mism}cols(side-wet-drawn-dry:{wet_not_drawn},of-them-ties:{ties}/side-dry-drawn-wet:{drawn_not_wet})"),
            "0[survey§6.2;finding-19]",
            mism == 0,
        );
        if mism > 0 {
            c.say(format!("     the first: {}", first.join("; ")));
        }
        if v == Var::R0 {
            c.reading(
                &lx.gate("CHECK-DRAWN-IS-SIDE-CONTROL", v),
                &format!("{ctl}cols"),
                "l:=hold;must-be-RED-on-the-pockets",
            );
        }
        c.judged(
            &lx.gate("G-SEA-OVER-DRY", v),
            &format!("{over_dry}cols"),
            "0[survey§6.2;finding-2]",
            over_dry == 0,
        );
        // G-SIDE-VS-GROUND.
        let mut svg: Vec<f64> = Vec::new();
        for (p, qq, _) in pairs(n) {
            let (Some(a), Some(b)) = (out.cols[p].as_ref(), out.cols[qq].as_ref()) else {
                continue;
            };
            let (da, db) = (a.drawn[vi], b.drawn[vi]);
            if da.side() == db.side() {
                continue;
            }
            let (w, dr) = if da.side() { (da, db) } else { (db, da) };
            let Some(level) = w.l() else { continue };
            let (gw, gd) = (f64::from(w.g), f64::from(dr.g));
            let t = if gw.is_finite() && gd.is_finite() && gw < 0.0 && gd > 0.0 {
                gw / (gw - gd)
            } else {
                0.5
            };
            let s = f64::from(w.s) + t * (f64::from(dr.s) - f64::from(w.s));
            svg.push((s - level).abs());
        }
        let svg_max = svg.iter().copied().fold(0.0f64, f64::max);
        c.judged(
            &lx.gate("G-SIDE-VS-GROUND", v),
            &format!(
                "max{}/p99{}/{}pairs",
                val(svg_max, 2, "m"),
                val(quantile(&svg, 0.99), 2, "m"),
                svg.len()
            ),
            &format!("<={:.4}m(half-q_r)[finding-1]", 0.5 * q),
            svg_max <= 0.5 * q,
        );
        // ── §8.2 WALL and CLAMP-BINDS ───────────────────────────────────────────────────────
        if lx.wall_rung {
            let (mut shore, mut over, mut bind_pairs, mut bind_over) =
                (0usize, 0usize, 0usize, 0usize);
            let mut ratios = Vec::new();
            let mut per_pair = Vec::new();
            for (p, qq, _) in pairs(n) {
                let (Some(a), Some(b)) = (out.cols[p].as_ref(), out.cols[qq].as_ref()) else {
                    continue;
                };
                let (da, db) = (a.drawn[vi], b.drawn[vi]);
                if da.wet() == db.wet() {
                    continue;
                }
                shore += 1;
                let delta = f64::from((da.s - db.s).abs());
                let raw = f64::from((a.shared.h_raw - b.shared.h_raw).abs());
                ratios.push(delta / q_wall);
                per_pair.push(delta / raw.max(q));
                let high = delta > q_wall;
                over += usize::from(high);
                if (da.flags | db.flags) & BINDS != 0 {
                    bind_pairs += 1;
                    bind_over += usize::from(high);
                }
            }
            let band = lx.choices.wall_band_share;
            let judged = shore >= lx.choices.wall_min_pairs;
            let pass = (over as f64) <= band * shore as f64;
            c.line(
                &lx.gate("WALL", v),
                &format!(
                    "{over}of{shore}pairs(p50{}/p99{}/max{}xQ;Q={})",
                    val(quantile(&ratios, 0.5), 2, ""),
                    val(quantile(&ratios, 0.99), 2, ""),
                    val(quantile(&ratios, 1.0), 2, ""),
                    val(q_wall, 3, "m")
                ),
                &format!(
                    "<={}of-pairs,>={}pairs[finding-19]",
                    band, lx.choices.wall_min_pairs
                ),
                if !judged {
                    "NO-BAND"
                } else if pass {
                    "GREEN"
                } else {
                    "RED"
                },
            );
            c.reading(
                &lx.gate("WALL-PER-PAIR", v),
                &format!(
                    "p50{}/p99{}xraw",
                    val(quantile(&per_pair, 0.5), 2, ""),
                    val(quantile(&per_pair, 0.99), 2, "")
                ),
                "delta/max(raw,q_r)",
            );
            if matches!(v, Var::O1 | Var::O1nf | Var::O1q) {
                let share = bind_pairs as f64 / shore.max(1) as f64;
                let pass_b = (bind_over as f64) <= band * bind_pairs as f64;
                c.line(
                    &lx.gate("CLAMP-BINDS", v),
                    &format!(
                        "{}%pairs/wall{bind_over}of{bind_pairs}",
                        val(100.0 * share, 2, "")
                    ),
                    &format!("wall<={band}of-bound-pairs[survey§6.2]"),
                    if bind_pairs < lx.choices.wall_min_pairs {
                        "NO-BAND"
                    } else if pass_b {
                        "GREEN"
                    } else {
                        "RED"
                    },
                );
            }
        }
        // ── §8.2 G-POCKET-HELD ──────────────────────────────────────────────────────────────
        let pockets_here = out
            .cols
            .iter()
            .flatten()
            .any(|col| col.shared.word == WORD_POCKET);
        if (lx.pocket_stand || pockets_here)
            && let Some(sea) = lx.sea
        {
            let (mut held, mut moved, mut worst, mut ctl2, mut pocket_cols) =
                (0usize, 0usize, 0.0f64, 0usize, 0usize);
            for col in out.cols.iter().flatten() {
                if col.shared.word != WORD_POCKET {
                    continue;
                }
                pocket_cols += 1;
                let d = col.drawn[vi];
                let (s, hr) = (f64::from(d.s), f64::from(col.shared.h_raw));
                held += usize::from(s > sea && hr < sea);
                let m = (s - hr).abs();
                moved += usize::from(m > q);
                worst = worst.max(m);
                if v == Var::R0 {
                    let pre = shore_f64(f64::from(col.shared.b), Some(sea), hr, SIDE_LAND);
                    ctl2 += usize::from(pre > sea && hr < sea);
                }
            }
            c.judged(
                &lx.gate("G-POCKET-HELD", v),
                &format!("{held}of{pocket_cols}cols"),
                "0[ruling-B7]",
                held == 0,
            );
            c.reading(
                &lx.gate("POCKET-FLOOR-MOVED", v),
                &format!("{moved}cols/max{}", val(worst, 1, "m")),
                "|s-h_raw|>q_r",
            );
            if v == Var::R0 {
                c.judged(
                    &lx.gate("G-POCKET-HELD-CONTROL", v),
                    &format!("{ctl2}cols"),
                    ">0[the-pre-B7-rule-must-lift-a-floor;DOC:19cols]",
                    ctl2 > 0,
                );
            }
        }
        // ── §8.2 G-PERCHED ──────────────────────────────────────────────────────────────────
        let wet_keys: Vec<u32> = out
            .cols
            .iter()
            .map(|col| col.as_ref().map_or(0, |c| c.drawn[vi].wet_key()))
            .collect();
        let (_, comps) = components(n, n, &wet_keys);
        let (mut perched, mut perched_cols) = (0usize, 0usize);
        for comp in &comps {
            let key = wet_keys[comp[0]];
            let Some(water) = Water::of_key(key) else {
                continue;
            };
            let any = comp.iter().any(|&k| {
                let (a, b) = ((k % n) as i32, (k / n) as i32);
                let site = NodeSite::of(lx.fine, lx.face, lx.rung, out.i0 + a, out.j0 + b);
                [(0, 0), (1, 0), (0, 1), (1, 1)].iter().any(|&(oi, oj)| {
                    let node =
                        vd_terrain::artifact::node_at(lx.fine, lx.face, site.ni + oi, site.nj + oj);
                    lx.nodes.wet_for(node, water)
                })
            });
            if !any {
                perched += 1;
                perched_cols += comp.len();
            }
        }
        c.judged(
            &lx.gate("G-PERCHED", v),
            &format!("{perched}bodies/{perched_cols}cols"),
            "0[ruling-B7]",
            perched == 0,
        );
        // ── §8.5 READINGS ───────────────────────────────────────────────────────────────────
        let (mut ramp, mut halo, mut two) = (0usize, 0usize, 0usize);
        let mut fade_slope = 0.0f64;
        for col in out.cols.iter().flatten() {
            ramp += usize::from(col.drawn[vi].flags & RAMP != 0);
            halo += usize::from(col.drawn[vi].flags & HALO != 0);
        }
        for (p, qq, along_j) in pairs(n) {
            let (Some(a), Some(b)) = (out.cols[p].as_ref(), out.cols[qq].as_ref()) else {
                continue;
            };
            let m = if along_j { lx.m_j } else { lx.m_i };
            let da = f64::from(a.drawn[vi].s - a.shared.h_raw);
            let db = f64::from(b.drawn[vi].s - b.shared.h_raw);
            fade_slope = fade_slope.max((da - db).abs() / m);
        }
        for bb in 0..n.saturating_sub(1) {
            for aa in 0..n - 1 {
                let ks = [
                    bb * n + aa,
                    bb * n + aa + 1,
                    (bb + 1) * n + aa,
                    (bb + 1) * n + aa + 1,
                ];
                let mut first: Option<u32> = None;
                let mut differ = false;
                for &k in &ks {
                    let Some(col) = out.cols[k].as_ref() else {
                        continue;
                    };
                    let d = col.drawn[vi];
                    if d.l.is_nan() {
                        continue;
                    }
                    match first {
                        None => first = Some(d.key),
                        Some(f) => differ |= f != d.key,
                    }
                }
                two += usize::from(differ);
            }
        }
        // FINE-RMS-NEAR-SHORE: the drawn relief s − B within 1 km of the drawn shore against 2–3
        // nodes away (a walk over the four sides from every shore column).
        let dist = shore_distance(out, vi);
        let node_cols = f64::from(lx.fine.cells_per_node) * lx.m_col / f64::from(1u32 << lx.rung);
        let (mut near, mut far) = (Vec::new(), Vec::new());
        for (k, col) in out.cols.iter().enumerate() {
            let Some(col) = col else { continue };
            if col.drawn[vi].wet() || dist[k] == u32::MAX {
                continue;
            }
            let dm = f64::from(dist[k]) * lx.m_col;
            let r = f64::from(col.drawn[vi].s - col.shared.b);
            if dm <= 1_000.0 {
                near.push(r * r);
            } else if dm >= 2.0 * node_cols && dm <= 3.0 * node_cols {
                far.push(r * r);
            }
        }
        let rms = |v: &[f64]| {
            if v.is_empty() {
                f64::NAN
            } else {
                (v.iter().sum::<f64>() / v.len() as f64).sqrt()
            }
        };
        if matches!(v, Var::N1 | Var::O1 | Var::O1q | Var::N2) {
            c.reading(
                &lx.gate("HOLD-RAMP", v),
                &format!("{ramp}cols"),
                "finding-10",
            );
            let halo_field: Vec<Option<f64>> = out
                .cols
                .iter()
                .map(|col| {
                    col.as_ref().map(|c| {
                        if c.drawn[vi].flags & HALO != 0 {
                            -1.0
                        } else {
                            1.0
                        }
                    })
                })
                .collect();
            let halo_lines: Vec<Vec<Point>> = zero_contours(n, &halo_field);
            let (hl4, _) = four_fold(&chord_axes(&halo_lines, (2_000.0 / lx.m_col).max(8.0)));
            c.reading(
                &lx.gate("HALO-EXTENT", v),
                &format!("{halo}cols/L4{}", val(hl4, 3, "")),
                "finding-16",
            );
            c.reading(
                &lx.gate("FADE-SLOPE-MAX", v),
                &val(fade_slope, 4, ""),
                "|grad(s-h_raw)|;finding-21",
            );
        }
        c.reading(
            &lx.gate("FINE-RMS-NEAR-SHORE", v),
            &format!(
                "{}within1km/{}at2-3nodes",
                val(rms(&near), 1, "m"),
                val(rms(&far), 1, "m")
            ),
            "finding-20",
        );
        // FINE-SLOPE-NEAR-SHORE (the second round, ruling B10 item 1): the drawn ground's RMS
        // slope over dry neighbour pairs within 1 km of the drawn shore, the raw ground's over the
        // same pairs, and the drawn ground's 2–3 nodes away — a flat disc round a lake reads a
        // near slope far under the raw one.
        let (mut s_near, mut raw_near, mut s_far) = (Vec::new(), Vec::new(), Vec::new());
        for (p, qq, along_j) in pairs(n) {
            let (Some(a), Some(b)) = (out.cols[p].as_ref(), out.cols[qq].as_ref()) else {
                continue;
            };
            if a.drawn[vi].wet() || b.drawn[vi].wet() {
                continue;
            }
            let dk = dist[p].min(dist[qq]);
            if dk == u32::MAX {
                continue;
            }
            let m = if along_j { lx.m_j } else { lx.m_i };
            let dm = f64::from(dk) * lx.m_col;
            let ds = f64::from(a.drawn[vi].s - b.drawn[vi].s) / m;
            if dm <= 1_000.0 {
                s_near.push(ds * ds);
                let dr = f64::from(a.shared.h_raw - b.shared.h_raw) / m;
                raw_near.push(dr * dr);
            } else if dm >= 2.0 * node_cols && dm <= 3.0 * node_cols {
                s_far.push(ds * ds);
            }
        }
        // RAISE-OVER-RAW (the second round): how far the rule stands the dry ground OVER the raw
        // ground within three nodes of the drawn shore — a filled lowland reads here.
        let mut raise: Vec<f64> = Vec::new();
        for (k, col) in out.cols.iter().enumerate() {
            let Some(col) = col else { continue };
            if col.drawn[vi].wet() || dist[k] == u32::MAX {
                continue;
            }
            if f64::from(dist[k]) * lx.m_col <= 3.0 * node_cols {
                raise.push(f64::from(col.drawn[vi].s - col.shared.h_raw));
            }
        }
        raise.sort_by(f64::total_cmp);
        let q = |p: f64| {
            if raise.is_empty() {
                f64::NAN
            } else {
                raise[((raise.len() as f64 * p) as usize).min(raise.len() - 1)]
            }
        };
        c.reading(
            &lx.gate("RAISE-OVER-RAW", v),
            &format!(
                "p50{}/p99{}/max{}/{}cols",
                val(q(0.5), 1, "m"),
                val(q(0.99), 1, "m"),
                val(q(1.0), 1, "m"),
                raise.len()
            ),
            "the-second-round",
        );
        c.reading(
            &lx.gate("FINE-SLOPE-NEAR-SHORE", v),
            &format!(
                "{}within1km/{}raw/{}at2-3nodes/{}pairs",
                val(rms(&s_near), 3, ""),
                val(rms(&raw_near), 3, ""),
                val(rms(&s_far), 3, ""),
                s_near.len()
            ),
            "B10-item-1",
        );
        c.reading(
            &lx.gate("G-SHORE-TWO-WATERS", v),
            &format!("{two}cells"),
            "a-reading",
        );
        let _ = sides;
        // ── §8.3 SHAPE ──────────────────────────────────────────────────────────────────────
        if lx.shape_rung {
            shape_lines(c, lx, out, v, &lc, &kc, &mut c0_l4, pool);
        }
    }
}

/// The distance in columns from every column to the nearest drawn shore column (a column with a
/// 4-neighbour of the other drawn side); `u32::MAX` where none is reached.
fn shore_distance(out: &RungOut, vi: usize) -> Vec<u32> {
    let n = out.n;
    let mut dist = vec![u32::MAX; n * n];
    let mut queue = VecDeque::new();
    let wet = |k: usize| out.cols[k].as_ref().map(|c| c.drawn[vi].wet());
    for (k, slot) in dist.iter_mut().enumerate() {
        let Some(w) = wet(k) else { continue };
        let (a, b) = (k % n, k / n);
        let edge = (a > 0 && wet(k - 1).is_some_and(|x| x != w))
            || (a + 1 < n && wet(k + 1).is_some_and(|x| x != w))
            || (b > 0 && wet(k - n).is_some_and(|x| x != w))
            || (b + 1 < n && wet(k + n).is_some_and(|x| x != w));
        if edge {
            *slot = 0;
            queue.push_back(k);
        }
    }
    while let Some(k) = queue.pop_front() {
        let (a, b) = (k % n, k / n);
        let d = dist[k] + 1;
        let mut go = |m: usize| {
            if dist[m] == u32::MAX {
                dist[m] = d;
                queue.push_back(m);
            }
        };
        if a > 0 {
            go(k - 1);
        }
        if a + 1 < n {
            go(k + 1);
        }
        if b > 0 {
            go(k - n);
        }
        if b + 1 < n {
            go(k + n);
        }
    }
    dist
}

/// ★ THE SHAPE LINES (plan §8.3) of one rule at one shape rung of W2.
#[allow(
    clippy::too_many_arguments,
    reason = "the shape lines read the rung's own arrays"
)]
fn shape_lines(
    c: &mut Census,
    lx: &LineCtx,
    out: &RungOut,
    v: Var,
    lc: &[f32],
    kc: &[u32],
    c0_l4: &mut [f64; 2],
    pool: &mut Pool,
) {
    let vi = v.idx();
    let n = out.n;
    let window = (2_000.0 / lx.m_col).max(8.0);
    let node_cols = (lx.fine.cells_per_node >> lx.rung) as usize;
    let void = lx.shape_void;
    for (ki, sea) in [(0usize, true), (1usize, false)] {
        let kind = if sea { "sea" } else { "lake" };
        let lines = traces(out, vi, lc, kc, sea, lx);
        let axes = chord_axes(&lines, window);
        let (l4, m4) = four_fold(&axes);
        if v == Var::C0 {
            c0_l4[ki] = l4;
        }
        let null = flat_null_l4(axes.len(), 2_000, 0.95).max((lx.lattice_band)(n).unwrap_or(0.0));
        let gate = lx.gate(&format!("G-SHORE-LATTICE.{kind}"), v);
        let value = format!(
            "L4{}/M4{}/{}windows/C0-L4{}",
            val(l4, 3, ""),
            val(m4, 3, ""),
            axes.len(),
            val(c0_l4[ki], 3, "")
        );
        let band = format!(
            "|L4|<={}[fBm-spread-or-flat-null;survey§1.7]",
            val(null, 3, "")
        );
        if void {
            c.line(&gate, &value, &band, "VOID");
        } else if axes.len() < 16 {
            c.line(&gate, &value, &band, "NO-BAND");
        } else {
            c.judged(&gate, &value, &band, l4.abs() <= null);
        }
        // CHECK-ROTATE: a 45° turn negates L4; a half-node shift keeps it.
        if !axes.is_empty() {
            let (lr, _) = four_fold(&chord_axes(&rotated(&lines, 45.0), window));
            let half = node_cols as f64 / 2.0;
            let (ls, _) = four_fold(&chord_axes(&shifted(&lines, half, half), window));
            let ok = (lr + l4).abs() < 1.0e-9 && (ls - l4).abs() < 1.0e-9;
            c.judged(
                &lx.gate(&format!("CHECK-ROTATE.{kind}"), v),
                &format!("{}/{}", val(lr + l4, 12, ""), val(ls - l4, 12, "")),
                "0[exact-arithmetic;survey§1.7]",
                ok,
            );
        }
        // G-COAST-KINK: the lines of four nodes or more, rulers 2^k columns.
        let long: Vec<Vec<Point>> = lines
            .iter()
            .filter(|l| polyline_length(l) >= 4.0 * node_cols as f64)
            .cloned()
            .collect();
        let gate = lx.gate(&format!("G-COAST-KINK.{kind}"), v);
        if node_cols >= 16 && !long.is_empty() {
            let eps: Vec<f64> = (3..=12)
                .map(|k| f64::from(1u32 << k))
                .filter(|&e| e <= n as f64 / 10.0 || e <= 2.0 * node_cols as f64)
                .collect();
            let sig = slopes_per_octave(&long, &eps);
            let at = |e: f64| eps.iter().position(|&x| (x - e).abs() < 0.5);
            let kink = match (at(node_cols as f64 / 2.0), at(node_cols as f64)) {
                (Some(a), Some(b)) if b < sig.len() => (sig[a] - sig[b]).abs(),
                _ => f64::NAN,
            };
            let band = (lx.kink_band)(node_cols);
            let value = format!(
                "{}/sigma[{}]",
                val(kink, 3, ""),
                sig.iter()
                    .map(|s| format!("{s:.3}"))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            match band {
                _ if void => c.line(&gate, &value, "q95-fBm", "VOID"),
                Some(b) if kink.is_finite() => c.judged(
                    &gate,
                    &value,
                    &format!("<={}[q95-of-8-WM-fBm,H=0.5;survey§1.8]", val(b, 3, "")),
                    kink <= b,
                ),
                _ => c.line(&gate, &value, "q95-fBm", "NO-BAND"),
            }
        } else {
            c.line(&gate, &format!("{}lines", long.len()), "q95-fBm", "NO-BAND");
        }
        // G-LAKE-CONTOUR / G-SEA-CONTOUR.
        contour_line(c, lx, out, vi, v, sea);
    }
    // G-LAKE-FRACTAL and G-LAKE-DBC.
    let wet_keys: Vec<u32> = out
        .cols
        .iter()
        .map(|col| {
            col.as_ref().map_or(0, |c| {
                let k = c.drawn[vi].wet_key();
                if k > 1 { k } else { 0 }
            })
        })
        .collect();
    let (_, comps) = components(n, n, &wet_keys);
    let value_of = |k: usize| -> f64 {
        match (out.cols[k].as_ref(), lc[k].is_nan()) {
            (Some(col), false) => f64::from(col.drawn[vi].s) - f64::from(lc[k]),
            _ => 1.0,
        }
    };
    let (mut la, mut lp) = (Vec::new(), Vec::new());
    for comp in &comps {
        if (comp.len() as f64).sqrt() < 10.0 {
            continue;
        }
        let Some(lp_loop) = outer_loop(n, n, comp, &value_of) else {
            continue;
        };
        let metric: Vec<Point> = lp_loop
            .iter()
            .map(|p| (p.0 * lx.m_i, p.1 * lx.m_j))
            .collect();
        let (p, a) = (polyline_length(&metric), shoelace(&metric).abs());
        if a > 0.0 && p > 0.0 {
            la.push(a.ln());
            lp.push(p.ln());
        }
    }
    let slope = ols(&la, &lp).0;
    let d = 2.0 * slope;
    let gate = lx.gate("G-LAKE-FRACTAL", v);
    let value = format!("d{}/{}lakes", val(d, 3, ""), la.len());
    if lx.shape_void {
        c.line(&gate, &value, "[1.12,1.40]", "VOID");
    } else if la.len() < 10 {
        c.line(&gate, &value, "[1.12,1.40][survey§2.3;L1-L3]", "NO-BAND");
    } else {
        c.judged(
            &gate,
            &value,
            "[1.12,1.40][survey§2.3;L1-L3]",
            (1.12..=1.40).contains(&d),
        );
    }
    let dbc: Vec<f64> = la
        .iter()
        .zip(&lp)
        .map(|(a, p)| p.exp() / (2.0 * std::f64::consts::PI.sqrt() * (a * d / 2.0).exp()))
        .collect();
    let areas: Vec<f64> = la.iter().map(|a| a.exp()).collect();
    let tau = kendall_tau(&dbc, &areas);
    let gate = lx.gate("G-LAKE-DBC", v);
    if la.len() < 10 || !d.is_finite() {
        c.line(
            &gate,
            &format!("tau{}/{}lakes", val(tau, 3, ""), la.len()),
            "[-0.16,0.12][L1]",
            "NO-BAND",
        );
    } else {
        c.judged(
            &gate,
            &format!("tau{}/{}lakes", val(tau, 3, ""), la.len()),
            "[-0.16,0.12][L1]",
            (-0.16..=0.12).contains(&tau),
        );
    }
    pool.entry((vi, lx.rung))
        .or_default()
        .extend(la.into_iter().zip(lp));
}

/// ★ G-LAKE-CONTOUR / G-SEA-CONTOUR: from each drawn water body, a flood through the four sides
/// over columns whose surface stands under that water's level by more than the extractor's step;
/// a flooded column drawn dry is water the ground's own contour holds and the rule does not draw.
fn contour_line(c: &mut Census, lx: &LineCtx, out: &RungOut, vi: usize, v: Var, sea: bool) {
    let n = out.n;
    let q = lx.q_r;
    let keys: Vec<u32> = out
        .cols
        .iter()
        .map(|col| {
            col.as_ref().map_or(0, |c| {
                let k = c.drawn[vi].wet_key();
                if (k == 1) == sea { k } else { 0 }
            })
        })
        .collect();
    let (label, comps) = components(n, n, &keys);
    let (mut flooded_dry, mut pocket_dry, mut comp_cols, mut worst) =
        (0usize, 0usize, 0usize, 0u32);
    let mut seen = vec![u32::MAX; n * n];
    for (id, comp) in comps.iter().enumerate() {
        let Some(level) = out.cols[comp[0]].as_ref().and_then(|c| c.drawn[vi].l()) else {
            continue;
        };
        comp_cols += comp.len();
        let mut queue: VecDeque<(usize, u32)> = comp.iter().map(|&k| (k, 0)).collect();
        for &k in comp {
            seen[k] = id as u32;
        }
        while let Some((k, d)) = queue.pop_front() {
            let (a, b) = (k % n, k / n);
            let mut next = |m: usize| {
                if seen[m] == id as u32 || label[m] == id as u32 {
                    return;
                }
                let Some(col) = out.cols[m].as_ref() else {
                    return;
                };
                if f64::from(col.drawn[vi].s) < level - q {
                    seen[m] = id as u32;
                    if !col.drawn[vi].wet() {
                        if col.shared.word == WORD_POCKET {
                            pocket_dry += 1;
                        } else {
                            flooded_dry += 1;
                            worst = worst.max(d + 1);
                        }
                    }
                    queue.push_back((m, d + 1));
                }
            };
            if a > 0 {
                next(k - 1);
            }
            if a + 1 < n {
                next(k + 1);
            }
            if b > 0 {
                next(k - n);
            }
            if b + 1 < n {
                next(k + n);
            }
        }
    }
    let name = if sea {
        "G-SEA-CONTOUR"
    } else {
        "G-LAKE-CONTOUR"
    };
    c.judged(
        &lx.gate(name, v),
        &format!(
            "X{flooded_dry}cols/{}%/{}bodies/Hausdorff{}{}",
            val(100.0 * flooded_dry as f64 / comp_cols.max(1) as f64, 2, ""),
            comps.len(),
            val(f64::from(worst) * lx.m_col, 0, "m"),
            if sea {
                format!("/pocket{pocket_dry}cols")
            } else {
                String::new()
            }
        ),
        "X=0[B8-item-1;survey§2.3]",
        flooded_dry == 0,
    );
}

/// ★ THE CROSS-RUNG LINES of one window and one rule (plan §8.2): LAKE-APPEAR at every swap,
/// LAKE-FLICKER at every inner rung, FOLD-RETURN over the window. `keys[r]` holds rung `fine + r`'s
/// drawn wet keys, `dims[r]` its origin and side.
#[allow(clippy::too_many_arguments, reason = "a window's own words")]
pub fn cross_lines(
    c: &mut Census,
    stand: &str,
    window: &str,
    v: Var,
    fine: u8,
    f_rung: u8,
    keys: &[Vec<u32>],
    dims: &[(i32, i32, usize)],
) {
    let rungs = keys.len();
    let comps_of = |r: usize| -> Vec<Vec<usize>> {
        let n = dims[r].2;
        let (_, comps) = components(n, n, &keys[r]);
        comps
            .into_iter()
            .filter(|comp| {
                !comp.iter().any(|&k| {
                    let (a, b) = (k % n, k / n);
                    a == 0 || b == 0 || a + 1 == n || b + 1 == n
                })
            })
            .collect()
    };
    // Whether rung `r2` (coarser) holds a wet column of `key` over any member of `comp` at `r1`.
    let holds = |r1: usize, comp: &[usize], r2: usize, key: u32| -> bool {
        let (i1, j1, n1) = dims[r1];
        let (i2, j2, n2) = dims[r2];
        let k = (r2 - r1) as u32;
        comp.iter().any(|&m| {
            let (a, b) = ((m % n1) as i32 + i1, (m / n1) as i32 + j1);
            let (pa, pb) = ((a >> k) - i2, (b >> k) - j2);
            (0..n2 as i32).contains(&pa)
                && (0..n2 as i32).contains(&pb)
                && keys[r2][pb as usize * n2 + pa as usize] == key
        })
    };
    // LAKE-APPEAR@r-(r+1): a coarser body with no finer wet column of its water under it.
    for r in 0..rungs.saturating_sub(1) {
        let (rf, rc) = (fine + r as u8, fine + r as u8 + 1);
        if v.nofold() && rc <= f_rung {
            continue;
        }
        let comps = comps_of(r + 1);
        let (i1, j1, n1) = dims[r + 1];
        let (i0, j0, n0) = dims[r];
        let mut appear = 0usize;
        let mut appear_any = 0usize;
        for comp in &comps {
            let key = keys[r + 1][comp[0]];
            let mut same = false;
            let mut any = false;
            for &m in comp {
                let (a, b) = ((m % n1) as i32 + i1, (m / n1) as i32 + j1);
                for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (ca, cb) = (2 * a + da - i0, 2 * b + db - j0);
                    if (0..n0 as i32).contains(&ca) && (0..n0 as i32).contains(&cb) {
                        let k0 = keys[r][cb as usize * n0 + ca as usize];
                        same |= k0 == key;
                        any |= k0 != 0;
                    }
                }
                if same {
                    break;
                }
            }
            appear += usize::from(!same);
            appear_any += usize::from(!any);
        }
        c.judged(
            &format!(
                "LAKE-APPEAR.{stand}.{window}.{}.r{rf:02}-r{rc:02}",
                v.name()
            ),
            &format!("{appear}bodies(any-water:{appear_any})/{}", comps.len()),
            "0[W16-item-2;survey§6.2]",
            appear == 0,
        );
    }
    // LAKE-FLICKER@r and FOLD-RETURN.
    let mut returns = 0usize;
    let mut folds = 0usize;
    for (r, rung_keys) in keys.iter().enumerate().take(rungs.saturating_sub(1)) {
        let comps = comps_of(r);
        let mut flicker = 0usize;
        for comp in &comps {
            let key = rung_keys[comp[0]];
            if holds(r, comp, r + 1, key) {
                continue;
            }
            folds += 1;
            if r + 2 < rungs && holds(r, comp, r + 2, key) {
                flicker += 1;
            }
            if (r + 2..rungs).any(|r2| holds(r, comp, r2, key)) {
                returns += 1;
            }
        }
        if r + 2 < rungs {
            let rr = fine + r as u8 + 1;
            if v.nofold() && rr < f_rung {
                continue;
            }
            c.judged(
                &format!("LAKE-FLICKER.{stand}.{window}.{}.r{rr:02}", v.name()),
                &format!("{flicker}bodies"),
                "0[W16;survey§6.2]",
                flicker == 0,
            );
        }
    }
    c.judged(
        &format!("FOLD-RETURN.{stand}.{window}.{}", v.name()),
        &format!("{returns}of{folds}folds"),
        "0[W15.1;survey§6.2]",
        returns == 0,
    );
}

/// The pooled G-LAKE-FRACTAL over every stand, per rule and rung (a reading beside the stands).
pub fn pooled_lines(c: &mut Census, pool: &Pool) {
    for (&(vi, rung), pts) in pool {
        let (la, lp): (Vec<f64>, Vec<f64>) = pts.iter().copied().unzip();
        let d = 2.0 * ols(&la, &lp).0;
        c.reading(
            &format!("G-LAKE-FRACTAL-POOLED.{}.r{rung:02}", Var::ALL[vi].name()),
            &format!("d{}/{}lakes", val(d, 3, ""), la.len()),
            "Earth[1.12,1.40];pooled-over-stands",
        );
    }
    let _ = NVAR;
}
