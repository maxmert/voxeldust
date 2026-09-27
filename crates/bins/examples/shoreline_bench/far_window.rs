//! ★ W4, THE FAR WINDOW (the build design's S0 item 5; rulings B13 and B14): rungs 13–18 of a
//! stand, today's rule beside N2, and N2's fold from THE SAMPLE COUNT PYRAMID under the one naming
//! rule (the only rule a count pyramid carries to the top rung).
//!
//! W4 is 1 024 node squares a side at rung 13 (about 8 400 km on the home planet), aligned to the
//! cells of rung 18 and pulled onto the stand's face. Its lines:
//! * CHECK-PYRAMID-UNIFORM — uniform node cells keyed whole again: the samples that differ from
//!   the one key the maximum principle promises (0).
//! * CHECK-PYRAMID-BAND — the pyramid keyed over the band-held nodes against the rows' (0 quarters).
//! * CHECK-PYRAMID-FOLD — rung-13 columns folded from their 4 096 samples through the column pass
//!   itself, against the pyramid's summed words (0).
//! * per rung: the window's check and rule lines (G-POCKET-HELD among them), and CHECK-SHORE-BAND:
//!   N2 over the band-held nodes against N2 over the rows, bit for bit (0).
//! * across the rungs: LAKE-APPEAR, LAKE-FLICKER and FOLD-RETURN.
//! * per swap 13 → 14 … 17 → 18: WATER-STEP, AREA-NEST and G-RUNG-HOLD, and SHORE-STEP on far
//!   transects (each 128 nodes long a side, so a transect still crosses its shore at rung 18).
//!
//! **Example.** On the gentle coast the 16 → 17 swap swaps a 65 km column for a 131 km one. Under
//! N2 the 131 km column sums the 2²⁰ samples' words of its four children, so it names the water
//! most of them name; WATER-STEP counts the 131 km columns whose water or level still differs from
//! the fold of their four 65 km children's drawn water.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use vd_bins::shore_rules::{WORD_POCKET, Water, fold_one_rule, length_of, metres};
use vd_recipe::terrace::terrace;
use vd_terrain::artifact::node_at;

use crate::census::{Census, val};
use crate::eval::{self, Caches, Checks, Fold, NVAR, RungOut, Var};
use crate::lines::{self, LineCtx, Pool};
use crate::pyramid::Pyramid;
use crate::stands::{self, Stand};
use crate::transects;
use crate::{World, metric};

/// W4's finest and last rung, and its side at the finest (node squares).
const FAR_FINE: u8 = 13;
const FAR_LAST: u8 = 18;
const FAR_N: i32 = 1_024;
/// Every this-many-th uniform cell with a water is keyed whole again (CHECK-PYRAMID-UNIFORM).
const UNIFORM_CHECK_EVERY: usize = 64;
/// CHECK-PYRAMID-FOLD's columns: at most this many coast columns and this many others.
const FOLD_CHECK_COAST: usize = 48;
const FOLD_CHECK_OTHER: usize = 16;
/// The far transects' half length in nodes: four cells of rung 18 (2⁵ nodes each).
const FAR_HALF_NODES: f64 = 128.0;

/// One far column as the swap lines read it, per rule of W4.
#[derive(Clone, Copy, Debug, Default)]
struct FarCol {
    read: bool,
    key: u32,
    s: f32,
    h_raw: f32,
    b: f32,
}

/// One rung of W4 as the swap lines read it.
struct FarRung {
    rung: u8,
    i0: i32,
    j0: i32,
    n: usize,
    cols: Vec<[FarCol; 2]>,
}

impl FarRung {
    fn of(out: &RungOut, vars: &[Var; 2]) -> FarRung {
        FarRung {
            rung: out.rung,
            i0: out.i0,
            j0: out.j0,
            n: out.n,
            cols: out
                .cols
                .iter()
                .map(|col| {
                    std::array::from_fn(|v| match col {
                        Some(o) => {
                            let d = o.drawn[vars[v].idx()];
                            FarCol {
                                read: true,
                                key: d.wet_key(),
                                s: d.s,
                                h_raw: o.shared.h_raw,
                                b: o.shared.b,
                            }
                        }
                        None => FarCol::default(),
                    })
                })
                .collect(),
        }
    }

    fn at(&self, i: i32, j: i32) -> Option<&[FarCol; 2]> {
        let (a, b) = (i - self.i0, j - self.j0);
        let n = self.n as i32;
        ((0..n).contains(&a) && (0..n).contains(&b))
            .then(|| &self.cols[(b * n + a) as usize])
            .filter(|c| c[0].read)
    }
}

/// ★ W4 OF ONE STAND.
pub fn run(c: &mut Census, w: &World, stand: &Stand) {
    let fine = w.body.macro_lattice().expect("a lattice");
    let win = stands::window(
        w.body, "W4", stand.face, FAR_FINE, FAR_LAST, FAR_N, FAR_LAST, stand.far,
    );
    let vars = [Var::R0, Var::N2];
    c.say(format!(
        "\n-- W4 of `{}`: face {:?}, rungs {}..={}, {} columns at rung {} from ({}, {}), pulled ({}, {}) columns to stay on the face; N2 folds from the sample count pyramid under the one naming rule",
        stand.name, win.face, win.fine, win.last, win.n, win.fine, win.i0, win.j0, win.pull.0, win.pull.1
    ));
    // THE PYRAMID, over the rows and over the band-held nodes.
    let q0 = (2 * win.i0, 2 * win.j0);
    let nq = 2 * win.n;
    let (pyr, built) = Pyramid::build(
        &fine,
        w.nodes,
        win.face,
        w.f_rung,
        q0,
        nq,
        &w.ch,
        w.threads,
        UNIFORM_CHECK_EVERY,
    );
    c.say(format!(
        "   W4's pyramid: {nq} x {nq} quarters of rung {} over {} node cells ({} with a water in their block, {} mixed); {} samples keyed in {:.1} s",
        pyr.q_rung, built.cells, built.with_water, built.mixed, built.samples_keyed, built.seconds
    ));
    c.judged(
        &format!("CHECK-PYRAMID-UNIFORM.{}.W4", stand.name),
        &format!(
            "{}samples-differ/{}cells-keyed-whole",
            built.uniform_differ, built.uniform_checked
        ),
        "0[design§3.2:the-held-read's-maximum-principle]",
        built.uniform_differ == 0 && built.uniform_checked > 0,
    );
    c.reading(
        &format!("MIXED-NODE-SQUARES.{}.W4", stand.name),
        &format!(
            "{}of{}squares/{}mixed-cells",
            built.squares_mixed, built.squares, built.mixed
        ),
        "the-node-squares-that-hold-a-quarter-of-a-mixed-cell:the-shipped-unit[B14-item-3]",
    );
    let (pyr_b, built_b) = Pyramid::build(
        &fine,
        w.banded,
        win.face,
        w.f_rung,
        q0,
        nq,
        &w.ch,
        w.threads,
        usize::MAX,
    );
    let q_differ = pyr
        .quarters
        .iter()
        .zip(&pyr_b.quarters)
        .filter(|(a, b)| a != b)
        .count();
    c.judged(
        &format!("CHECK-PYRAMID-BAND.{}.W4", stand.name),
        &format!(
            "{q_differ}of{}quarters/{}mixed-cells-over-the-band",
            pyr.quarters.len(),
            built_b.mixed
        ),
        "0[B14-item-1;design§2.3:the-band's-words-are-the-rows']",
        q_differ == 0 && built_b.mixed == built.mixed,
    );
    drop(pyr_b);
    // THE SKIN'S EXACTNESS on W4 (slice K0a): the one pyramid from the quarter records (level 12)
    // up to the top rung, and the rung-13 windows cut inside it.
    {
        let base = crate::skin::Level {
            level: pyr.q_rung,
            i0: i64::from(pyr.q0.0),
            j0: i64::from(pyr.q0.1),
            n: i64::from(pyr.nq),
            wet: pyr.quarters.iter().map(|q| q.wet).collect(),
            lake: pyr.quarters.iter().map(|q| q.lake).collect(),
        };
        let skin = crate::skin::SkinPyramid::from_base(win.face, base, FAR_LAST);
        crate::skin_lines::exactness(c, &format!("{}.W4", stand.name), &skin, 13..=13);
        // SKIN-FRONT (slice K0c): every front 6 -> 7 ... 17 -> 18 over this window's cells.
        let sea_r = w.body.radius_m() + w.artifact.sea().map_or(0.0, f64::from);
        let mut sources = crate::front::Sources::new(
            w.body,
            win.face,
            w.f_rung,
            &pyr,
            &skin,
            w.ch.skin_sub,
            w.ch.skin_margin,
            w.ch.skin_levels,
        );
        let t = std::time::Instant::now();
        crate::front::front_lines(c, &format!("{}.W4", stand.name), &mut sources, sea_r, 6);
        c.say(format!(
            "   SKIN-FRONT read in {:.1} s",
            t.elapsed().as_secs_f64()
        ));
        // SKIN-TRACK (slice K0c): the design's climbs, each over its own stand's window.
        let t = std::time::Instant::now();
        let f_cell = f64::from(1u32 << w.f_rung);
        let climbs: Vec<crate::front::Climb> = match stand.name {
            // The 5 882 m lake's measured piece: rung-14 column (275, 98) of this window's face.
            "coast" => vec![crate::front::Climb {
                name: "highlakes-piece-7500-60000km",
                at: (275.0 * 128.0 + 64.0, 98.0 * 128.0 + 64.0),
                h0: 7.5e6,
                h1: 6.0e7,
                half: 16,
                step: 2,
            }],
            "gentle" => vec![crate::front::Climb {
                name: "gentle-coast-60-1400km",
                at: (stand.far.0 as f64 / f_cell, stand.far.1 as f64 / f_cell),
                h0: 6.0e4,
                h1: 1.4e6,
                half: 16,
                step: 2,
            }],
            _ => Vec::new(),
        };
        for climb in &climbs {
            crate::front::track_lines(c, &format!("{}.W4", stand.name), &mut sources, sea_r, climb);
        }
        if !climbs.is_empty() {
            c.say(format!(
                "   SKIN-TRACK read in {:.1} s",
                t.elapsed().as_secs_f64()
            ));
        }
        // PHI-HANDOVER (Q5) over the gentle coast; SKIN-MB (Q8) on every window.
        if stand.name == "gentle" {
            let at = (stand.far.0 as f64 / f_cell, stand.far.1 as f64 / f_cell);
            crate::front::handover_lines(c, &format!("{}.W4", stand.name), &mut sources, sea_r, at);
        }
        crate::front::memory_lines(c, &format!("{}.W4", stand.name), &mut sources);
        let at = (stand.far.0 as f64 / f_cell, stand.far.1 as f64 / f_cell);
        crate::front::ladder_memory_lines(c, &format!("{}.W4", stand.name), &sources, w.cache, at);
    }
    fold_check(
        c,
        w,
        stand,
        &win,
        &pyr,
        win.fine,
        FOLD_CHECK_COAST,
        FOLD_CHECK_OTHER,
    );
    fold_check(
        c,
        w,
        stand,
        &win,
        &pyr,
        win.fine + 1,
        FOLD_CHECK_COAST / 4,
        FOLD_CHECK_OTHER / 4,
    );

    // THE RUNGS.
    let mut far: Vec<FarRung> = Vec::new();
    let mut cross: Vec<Vec<Vec<u32>>> = vec![Vec::new(); NVAR];
    let mut dims: Vec<(i32, i32, usize)> = Vec::new();
    let mut far_tr: Vec<transects::Transect> = Vec::new();
    let mut pool = Pool::default();
    let fold = Fold {
        map: None,
        ctx_f: None,
        pyramid: Some(&pyr),
    };
    let pocket = matches!(stand.name, "pocket1" | "pocket2");
    let (mut control_cols, mut control_differ) = (0usize, 0usize);
    let mut far_ch = w.ch;
    far_ch.half_sea_nodes = FAR_HALF_NODES;
    far_ch.half_lake_nodes = FAR_HALF_NODES;
    far_ch.full_from = FAR_FINE;
    for r in win.rungs() {
        let Some(ctx) = w.ctx(win.face, r, false) else {
            c.say(format!("   rung {r}: REFUSED — no read at this rung"));
            continue;
        };
        let (i0, j0, n) = win.at(r);
        let out = eval::eval_window(&ctx, i0, j0, n, &fold, w.threads, true);
        let (m_i, m_j) = metric(&ctx, i0 + n / 2, j0 + n / 2);
        let m_col = 0.5 * (m_i + m_j);
        c.say(format!(
            "   W4 rung {r}: {n} x {n} columns of {m_i:.0} x {m_j:.0} m ({:.0} km a side), level {}; {} columns read in {:.1} s",
            n as f64 * m_col / 1_000.0,
            ctx.reader.level,
            out.checks.columns,
            out.seconds
        ));
        // CHECK-SHORE-BAND: N2 over the band-held nodes, bit for bit.
        if let Some(ctx_b) = w.ctx_on(win.face, r, w.banded) {
            let out_b = eval::eval_window(&ctx_b, i0, j0, n, &fold, w.threads, false);
            let (mut cols, mut differ) = (0usize, 0usize);
            for (a, b) in out.cols.iter().zip(&out_b.cols) {
                match (a, b) {
                    (Some(a), Some(b)) => {
                        cols += 1;
                        let (x, y) = (a.drawn[Var::N2.idx()], b.drawn[Var::N2.idx()]);
                        differ += usize::from(
                            x.s.to_bits() != y.s.to_bits()
                                || x.l.to_bits() != y.l.to_bits()
                                || x.key != y.key
                                || x.flags != y.flags,
                        );
                    }
                    (None, None) => {}
                    _ => differ += 1,
                }
            }
            c.judged(
                &format!("CHECK-SHORE-BAND.{}.W4.r{r:02}", stand.name),
                &format!("{differ}of{cols}cols"),
                "0[B14-item-1;design§2.3:N2-over-the-band-is-N2-over-the-rows]",
                differ == 0 && cols > 0,
            );
        }
        if let Some(ctx_c) = w.ctx_on(win.face, r, w.banded_control) {
            let out_c = eval::eval_window(&ctx_c, i0, j0, n, &fold, w.threads, false);
            for (a, b) in out.cols.iter().zip(&out_c.cols) {
                if let (Some(a), Some(b)) = (a, b) {
                    control_cols += 1;
                    let (x, y) = (a.drawn[Var::N2.idx()], b.drawn[Var::N2.idx()]);
                    control_differ += usize::from(
                        x.s.to_bits() != y.s.to_bits()
                            || x.l.to_bits() != y.l.to_bits()
                            || x.key != y.key
                            || x.flags != y.flags,
                    );
                }
            }
        }
        let lx = LineCtx {
            stand: stand.name,
            window: "W4",
            rung: r,
            face: win.face,
            m_i,
            m_j,
            m_col,
            q_r: f64::from(vd_seed::ladder::cell_m(r)) / 128.0,
            sea: w.artifact.sea().map(f64::from),
            pocket_stand: pocket,
            fine: &fine,
            nodes: w.nodes,
            vars: &vars,
            f_rung: w.f_rung,
            wall_rung: false,
            shape_rung: false,
            choices: w.ch,
            kink_band: &|_| None,
            lattice_band: &|_| None,
            shape_void: false,
        };
        lines::check_lines(c, &lx, &out, None);
        lines::rule_lines(c, &lx, &out, &mut pool);
        // The first side-wet columns N2 draws dry, with their parts (the far rungs' ties).
        let mut shown = 0usize;
        for (k, col) in out.cols.iter().enumerate() {
            let Some(o) = col else {
                continue;
            };
            let d = o.drawn[Var::N2.idx()];
            if d.side() && !d.wet() && shown < 3 {
                shown += 1;
                let (ci, cj) = (i0 + (k % n as usize) as i32, j0 + (k / n as usize) as i32);
                c.say(format!(
                    "   W4 N2 side-wet-drawn-dry at rung {r} ({ci}, {cj}): s {:.4} l {:.4} g_used {:.6} h_raw {:.1} b {:.1} flags {:#04x} key {} words {:?}",
                    d.s,
                    d.l,
                    d.g,
                    o.shared.h_raw,
                    o.shared.b,
                    d.flags,
                    d.key,
                    pyr.words(r, ci, cj)
                ));
            }
        }
        if r == win.fine {
            far_tr = transects::place(&out, &fine, w.nodes, win.face, m_i, m_j, &far_ch);
            c.say(format!(
                "   the far transects: {} over the sea, {} over the lakes, {:.0} km a side",
                far_tr.iter().filter(|t| t.kind == 0).count(),
                far_tr.iter().filter(|t| t.kind == 1).count(),
                FAR_HALF_NODES * fine.node_m() / 1_000.0
            ));
        }
        for v in &vars {
            cross[v.idx()].push(
                out.cols
                    .iter()
                    .map(|col| col.as_ref().map_or(0, |o| o.drawn[v.idx()].wet_key()))
                    .collect(),
            );
        }
        dims.push((i0, j0, n as usize));
        far.push(FarRung::of(&out, &vars));
    }
    c.judged(
        &format!("CHECK-SHORE-BAND-CONTROL.{}.W4", stand.name),
        &format!("{control_differ}of{control_cols}cols-over-rungs-13-18"),
        ">0[S0's-first-band-rule-left-out-the-wet-nodes-at-W-1:the-check-must-see-it]",
        control_differ > 0,
    );
    for v in &vars {
        lines::cross_lines(
            c,
            stand.name,
            "W4",
            *v,
            win.fine,
            w.f_rung,
            &cross[v.idx()],
            &dims,
        );
    }
    let sea = w.artifact.sea().map_or(f64::NAN, f64::from);
    swap_lines(c, stand.name, &far, &vars, sea);
    lid_lines(c, w, stand, &win, &far, sea);
    flicker_examples(
        c,
        &cross[Var::N2.idx()],
        &cross[Var::R0.idx()],
        &dims,
        &pyr,
        win.fine,
    );

    // SHORE-STEP on the far transects.
    if far_tr.is_empty() {
        c.say("   W4 SHORE-STEP: no far transect (no water's contour at rung 13)");
        return;
    }
    let Some(ctx7) = w.ctx(win.face, w.f_rung, false) else {
        return;
    };
    let k7 = u32::from(win.fine - w.f_rung);
    let (c7i, c7j) = ((win.i0 + win.n / 2) << k7, (win.j0 + win.n / 2) << k7);
    let (m7i, m7j) = metric(&ctx7, c7i, c7j);
    drop(ctx7);
    let ctxs: Vec<Option<eval::Ctx>> = (0..=FAR_LAST)
        .map(|r| {
            if r >= FAR_FINE {
                w.ctx(win.face, r, false)
            } else {
                None
            }
        })
        .collect();
    let t = std::time::Instant::now();
    let steps = transects::measure(
        &far_tr,
        &ctxs,
        None,
        Some(&pyr),
        m7i,
        m7j,
        0.5 * (m7i + m7j),
        &far_ch,
        w.threads,
        true,
    );
    c.say(format!(
        "   W4 SHORE-STEP: {} far transects walked in {:.1} s",
        far_tr.len(),
        t.elapsed().as_secs_f64()
    ));
    let _ = transects::print(
        c,
        stand.name,
        &steps,
        &far_tr,
        &vars,
        w.f_rung,
        0,
        FAR_FINE..FAR_LAST,
    );
}

/// ★ CHECK-PYRAMID-FOLD: columns of W4 at rung `rung` folded from their own rung-F samples through
/// the column pass (`Ctx::eval`'s N2 key), against the pyramid's summed words and its one-rule
/// fold: at most `coast_n` columns whose words are mixed and `other_n` others.
#[allow(clippy::too_many_arguments, reason = "the check's own words")]
fn fold_check(
    c: &mut Census,
    w: &World,
    stand: &Stand,
    win: &stands::Window,
    pyr: &Pyramid,
    rung: u8,
    coast_n: usize,
    other_n: usize,
) {
    let Some(ctx_f) = w.ctx(win.face, w.f_rung, false) else {
        return;
    };
    let (i0, j0, n) = win.at(rung);
    let mut coast: Vec<(i32, i32)> = Vec::new();
    let mut other: Vec<(i32, i32)> = Vec::new();
    for j in j0..j0 + n {
        for i in i0..i0 + n {
            if let Some((tot, wet, lake, _)) = pyr.words(rung, i, j) {
                if (wet > 0 && wet < tot) || lake > 0 {
                    coast.push((i, j));
                } else {
                    other.push((i, j));
                }
            }
        }
    }
    let pick = |v: &[(i32, i32)], m: usize| -> Vec<(i32, i32)> {
        let step = v.len().div_ceil(m.max(1)).max(1);
        v.iter().step_by(step).copied().collect()
    };
    let mut picks = pick(&coast, coast_n);
    picks.extend(pick(&other, other_n));
    let k = u32::from(rung - w.f_rung);
    let side = 1i32 << k;
    let next = AtomicUsize::new(0);
    let tally: Mutex<(usize, usize, usize, u64, usize)> = Mutex::new((0, 0, 0, 0, 0));
    std::thread::scope(|sc| {
        for _ in 0..w.threads.max(1) {
            sc.spawn(|| {
                let mut caches = Caches::default();
                let mut ck = Checks::default();
                let (mut checked, mut words_differ, mut fold_differ, mut samples, mut unread) =
                    (0usize, 0usize, 0usize, 0u64, 0usize);
                loop {
                    let q = next.fetch_add(1, Ordering::Relaxed);
                    if q >= picks.len() {
                        break;
                    }
                    let (ci, cj) = picks[q];
                    let mut keys: Vec<u32> = Vec::with_capacity((side * side) as usize);
                    for b in 0..side {
                        for a in 0..side {
                            match ctx_f.eval(
                                (ci << k) + a,
                                (cj << k) + b,
                                &Fold::none(),
                                &mut caches,
                                &mut ck,
                            ) {
                                Some(o) => keys.push(o.keys[3]),
                                None => {
                                    unread += 1;
                                    keys.push(0);
                                }
                            }
                        }
                    }
                    samples += keys.len() as u64;
                    let (mut wet, mut lake, mut top) = (0u64, 0u64, None::<i16>);
                    for &kk in &keys {
                        match Water::of_key(kk) {
                            None => {}
                            Some(Water::Sea) => wet += 1,
                            Some(Water::Lake(l)) => {
                                wet += 1;
                                lake += 1;
                                top = Some(top.map_or(l, |t| t.max(l)));
                            }
                        }
                    }
                    let want = Some((keys.len() as u64, wet, lake, top));
                    words_differ += usize::from(pyr.words(rung, ci, cj) != want);
                    fold_differ +=
                        usize::from(pyr.fold(rung, ci, cj) != Some(fold_one_rule(&keys, 0)));
                    checked += 1;
                }
                if let Ok(mut t) = tally.lock() {
                    t.0 += checked;
                    t.1 += words_differ;
                    t.2 += fold_differ;
                    t.3 += samples;
                    t.4 += unread;
                }
            });
        }
    });
    let (checked, words_differ, fold_differ, samples, unread) =
        tally.into_inner().unwrap_or_default();
    c.judged(
        &format!("CHECK-PYRAMID-FOLD.{}.W4.r{rung:02}", stand.name),
        &format!(
            "{words_differ}words/{fold_differ}folds-differ-of{checked}cols({samples}samples-through-the-column-pass,{unread}unread)"
        ),
        "0[design§3.3:the-record-is-the-column's-own-keys]",
        words_differ == 0 && fold_differ == 0 && unread == 0 && checked > 0,
    );
}

/// ★ THE SWAP LINES of W4 per rule, at every swap `r → r + 1`:
/// * WATER-STEP — the coarser columns whose four finer columns all draw ONE water (or all draw
///   none) and which draw another: water APPEARS where every finer column is dry, VANISHES where
///   every finer column is wet, or CHANGES (another water or level). A coast's own footprint (finer
///   columns of both sides) is the shore step's, never this line's; its coarser column is only
///   read against a three-of-four majority, as a reading.
/// * AREA-NEST — the drawn wet area over the coarser columns' footprint, coarser against finer.
/// * G-RUNG-HOLD — dry pairs whose ground stands under the sea's level at either rung, lifted over
///   the sea at one rung and not at the other by more than the finer rung's extractor step (a
///   floor that jumps at the swap), with the largest jump.
fn swap_lines(c: &mut Census, stand: &str, far: &[FarRung], vars: &[Var; 2], sea: f64) {
    for pair in far.windows(2) {
        let (fine_r, coarse) = (&pair[0], &pair[1]);
        let tag = format!("r{:02}-r{:02}", fine_r.rung, coarse.rung);
        let q_fine = f64::from(vd_seed::ladder::cell_m(fine_r.rung)) / 128.0;
        for (vi, v) in vars.iter().enumerate() {
            let (mut unanimous, mut appear, mut vanish, mut changed) = (0usize, 0, 0, 0);
            let (mut majority, mut majority_flips, mut mixed) = (0usize, 0usize, 0usize);
            let (mut area_c, mut area_f, mut lake_c, mut lake_f) = (0u64, 0u64, 0u64, 0u64);
            let (mut hold_pairs, mut hold_flips, mut hold_visible) = (0usize, 0usize, 0usize);
            let mut hold_max = 0.0f64;
            for b in 0..coarse.n as i32 {
                for a in 0..coarse.n as i32 {
                    let (pi, pj) = (coarse.i0 + a, coarse.j0 + b);
                    let Some(p) = coarse.at(pi, pj) else {
                        continue;
                    };
                    let p = p[vi];
                    let kids: Vec<FarCol> = [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .iter()
                        .filter_map(|&(da, db)| fine_r.at(2 * pi + da, 2 * pj + db))
                        .map(|k| k[vi])
                        .collect();
                    if kids.len() < 4 {
                        continue;
                    }
                    let k0 = kids[0].key;
                    if kids.iter().all(|k| k.key == k0) {
                        unanimous += 1;
                        if p.key != k0 {
                            match (k0 == 0, p.key == 0) {
                                (true, false) => appear += 1,
                                (false, true) => vanish += 1,
                                _ => changed += 1,
                            }
                        }
                    } else {
                        mixed += 1;
                        let top = kids
                            .iter()
                            .map(|k| (kids.iter().filter(|o| o.key == k.key).count(), k.key))
                            .max();
                        if let Some((n, key)) = top
                            && n >= 3
                        {
                            majority += 1;
                            majority_flips += usize::from(p.key != key);
                        }
                    }
                    area_c += 4 * u64::from(p.key != 0);
                    lake_c += 4 * u64::from(p.key > 1);
                    for k in &kids {
                        area_f += u64::from(k.key != 0);
                        lake_f += u64::from(k.key > 1);
                        if p.key == 0
                            && k.key == 0
                            && (f64::from(p.b) < sea || f64::from(k.b) < sea)
                        {
                            hold_pairs += 1;
                            let held =
                                |x: &FarCol| f64::from(x.s) > sea && f64::from(x.h_raw) < sea;
                            if held(&p) != held(k) {
                                hold_flips += 1;
                                let jump = (f64::from(p.s) - f64::from(k.s)).abs();
                                if jump > q_fine {
                                    hold_visible += 1;
                                    hold_max = hold_max.max(jump);
                                }
                            }
                        }
                    }
                }
            }
            let pct = |c: u64, f: u64| {
                if f == 0 {
                    f64::NAN
                } else {
                    100.0 * (c as f64 - f as f64) / f as f64
                }
            };
            let steps = appear + vanish + changed;
            c.judged(
                &format!("WATER-STEP.{stand}.W4.{}.{tag}", v.name()),
                &format!(
                    "{steps}of{unanimous}one-water-footprints({appear}appear/{vanish}vanish/{changed}change)/coast-footprints:{majority_flips}of{majority}against-a-3-of-4-majority({mixed}mixed)"
                ),
                "0[W16:a-rung-boundary-is-invisible;S0-item-5]",
                steps == 0 && unanimous > 0,
            );
            c.reading(
                &format!("AREA-NEST.{stand}.W4.{}.{tag}", v.name()),
                &format!(
                    "wet{}%/lakes{}%({area_c}vs{area_f}and{lake_c}vs{lake_f}finer-columns)",
                    val(pct(area_c, area_f), 2, ""),
                    val(pct(lake_c, lake_f), 2, "")
                ),
                "the-coarser-drawn-area-against-the-finer-over-one-footprint[S0-item-5]",
            );
            c.judged(
                &format!("G-RUNG-HOLD.{stand}.W4.{}.{tag}", v.name()),
                &format!(
                    "{hold_visible}jumps-over-q_r{}/max{}({hold_flips}flips-of{hold_pairs}dry-pairs-under-the-sea's-level)",
                    val(q_fine, 0, "m"),
                    val(hold_max, 0, "m")
                ),
                "0[ruling-W16;plan-step-2-risk-R7]",
                hold_visible == 0,
            );
        }
    }
}

/// The first N2 lake bodies that FLICKER at the second far rung (drawn at rung r, not over any
/// member at r + 1, again at r + 2), with the water words of the columns over one member.
fn flicker_examples(
    c: &mut Census,
    keys: &[Vec<u32>],
    keys_r0: &[Vec<u32>],
    dims: &[(i32, i32, usize)],
    pyr: &Pyramid,
    fine: u8,
) {
    if keys.len() < 3 {
        return;
    }
    let (r1, r2, r3) = (1usize, 2usize, 3usize.min(keys.len() - 1));
    let (i1, j1, n1) = dims[r1];
    let key_in = |set: &[Vec<u32>], r: usize, a: i32, b: i32| -> u32 {
        let (i0, j0, n) = dims[r];
        let (x, y) = (a - i0, b - j0);
        if (0..n as i32).contains(&x) && (0..n as i32).contains(&y) {
            set[r][y as usize * n + x as usize]
        } else {
            0
        }
    };
    let key_at = |r: usize, a: i32, b: i32| key_in(keys, r, a, b);
    let (_, comps) = vd_bins::shore_shape::components(n1, n1, &keys[r1]);
    let mut shown = 0usize;
    for comp in &comps {
        let k0 = keys[r1][comp[0]];
        if k0 <= 1 {
            continue;
        }
        let at = |m: usize| ((m % n1) as i32 + i1, (m / n1) as i32 + j1);
        let held = |rr: usize, sh: u32| {
            comp.iter().any(|&m| {
                let (a, b) = at(m);
                key_at(rr, a >> sh, b >> sh) == k0
            })
        };
        if held(r2, 1) || !held(r3, 2) || shown >= 8 {
            continue;
        }
        shown += 1;
        let (a, b) = at(comp[0]);
        let r = fine + r1 as u8;
        let level = Water::of_key(k0).map_or(0.0, |w| w.level_m(0));
        c.say(format!(
            "   W4 N2 FLICKER at rung {r}: a lake at {level} m, {} columns from ({a}, {b}); rung {r} words {:?}; rung {} over it: key {} words {:?}; rung {}: key {} words {:?}; today's rule there: keys {} / {} / {}",
            comp.len(),
            pyr.words(r, a, b),
            r + 1,
            key_at(r2, a >> 1, b >> 1),
            pyr.words(r + 1, a >> 1, b >> 1),
            r + 2,
            key_at(r3, a >> 2, b >> 2),
            pyr.words(r + 2, a >> 2, b >> 2),
            key_in(keys_r0, r1, a, b),
            key_in(keys_r0, r2, a >> 1, b >> 1),
            key_in(keys_r0, r3, a >> 2, b >> 2)
        ));
    }
}

/// ★ THE LID'S LINES on W4 (slice K0b; the far-water design §5.2 group 6): today's far land under
/// N2 against the lid built from the rows.
/// * HOLD-RETURN — columns whose hold (dry, over the sea, its raw ground under it) flips at a swap
///   and flips back at the next: the land shows, hides and shows; the largest jump in metres, in
///   pixels at the swap, and the Lambert step at a 45° sun.
/// * LID-RETURN — the same walk on the lid: a column whose height rises and falls back by more than
///   the extractor's step at two swaps in a row (the relief's own wobble, beside HOLD-RETURN).
/// * TOP-STEP — the finer surface against the coarser lid over each coarser column (at 13 → 14 the
///   finer is N2's rung-13 surface), in coarser cells: the handover's step (T7: at most one).
/// * TOP-APPROX — the interim lid T~ (the shipped level-1 word for a part-wet node) against the
///   exact lid, in metres and in pixels at the swap.
/// * POCKET-BASIN — level nodes all of a pocket's dry floor whose lid stands over their own
///   terraced mean; control: a lid held over the sea's level.
/// * MIXED-LEVEL-NODES — the part-wet level-1 nodes of the window (the dry word's rows).
fn lid_lines(
    c: &mut Census,
    w: &World,
    stand: &Stand,
    win: &stands::Window,
    far: &[FarRung],
    sea: f64,
) {
    let fine = w.body.macro_lattice().expect("a lattice");
    let Some(ctx) = w.ctx(win.face, FAR_FINE, false) else {
        return;
    };
    let charter = ctx.reader.charter;
    let (i0, j0, n) = win.at(FAR_FINE);
    let lid = crate::lid::Lid::from_rows(
        w.nodes,
        w.artifact,
        &fine,
        win.face,
        (i64::from(i0), i64::from(j0), i64::from(n)),
        FAR_LAST - FAR_FINE,
    );
    let cell = |r: u8| f64::from(vd_seed::ladder::cell_m(r));
    let n2 = 1usize; // FarRung's second slot is N2.
    // The drawn surfaces per rung: rung 13 from N2, rungs 14..18 today's N2 and the lid.
    let height_today = |r: u8, i: i64, j: i64| -> Option<(f64, bool)> {
        let fr = far.iter().find(|f| f.rung == r)?;
        let col = fr.at(i as i32, j as i32)?[n2];
        let held = col.key == 0 && f64::from(col.s) > sea && f64::from(col.h_raw) < sea;
        Some((f64::from(col.s), held))
    };
    let height_lid = |r: u8, i: i64, j: i64| -> Option<f64> {
        if r == FAR_FINE {
            height_today(r, i, j).map(|(s, _)| s)
        } else {
            lid.height(&charter, r - FAR_FINE, i, j, false)
        }
    };
    let lambert = |jump: f64, width: f64| -> f64 {
        let tilt = (jump / width).atan();
        let sun = std::f64::consts::FRAC_PI_4;
        100.0 * ((sun - tilt).cos() / sun.cos() - 1.0).abs()
    };
    // HOLD-RETURN and LID-RETURN, per swap pair r -> r+1 -> r+2.
    for r in FAR_FINE..FAR_LAST.saturating_sub(1) {
        let Some(fr) = far.iter().find(|f| f.rung == r) else {
            continue;
        };
        let (mut returns, mut max_jump, mut lid_returns, mut lid_max) =
            (0usize, 0.0f64, 0usize, 0.0f64);
        let (mut today_returns, mut today_max) = (0usize, 0.0f64);
        let q = cell(r + 1) / 128.0;
        for b in 0..fr.n as i64 {
            for a in 0..fr.n as i64 {
                let (i, j) = (i64::from(fr.i0) + a, i64::from(fr.j0) + b);
                let (Some((s0, h0)), Some((s1, h1)), Some((s2, h2))) = (
                    height_today(r, i, j),
                    height_today(r + 1, i >> 1, j >> 1),
                    height_today(r + 2, i >> 2, j >> 2),
                ) else {
                    continue;
                };
                if h0 != h1 && h2 == h0 {
                    returns += 1;
                    max_jump = max_jump.max((s1 - s0).abs()).max((s2 - s1).abs());
                }
                let (e1, e2) = (s1 - s0, s2 - s1);
                if e1.abs() > q && e2.abs() > q && e1.signum() != e2.signum() {
                    today_returns += 1;
                    today_max = today_max.max(e1.abs().min(e2.abs()));
                }
                let (Some(t0), Some(t1), Some(t2)) = (
                    height_lid(r, i, j),
                    height_lid(r + 1, i >> 1, j >> 1),
                    height_lid(r + 2, i >> 2, j >> 2),
                ) else {
                    continue;
                };
                let (d1, d2) = (t1 - t0, t2 - t1);
                if d1.abs() > q && d2.abs() > q && d1.signum() != d2.signum() {
                    lid_returns += 1;
                    lid_max = lid_max.max(d1.abs().min(d2.abs()));
                }
            }
        }
        let w1 = cell(r + 1);
        c.reading(
            &format!(
                "HOLD-RETURN.{}.W4.N2.r{r:02}-r{:02}-r{:02}",
                stand.name,
                r + 1,
                r + 2
            ),
            &format!(
                "{returns}cols/max{max_jump:.0}m/{:.2}px/lambert{:.1}%",
                max_jump / cell(r),
                lambert(max_jump, w1)
            ),
            "today's-far-land:the-hold-flips-and-flips-back[design-D-Q2]",
        );
        c.reading(
            &format!("LID-RETURN.{}.W4.r{r:02}-r{:02}-r{:02}", stand.name, r + 1, r + 2),
            &format!(
                "{lid_returns}cols/max{lid_max:.0}m(q{q:.0}m);today's-surface:{today_returns}cols/max{today_max:.0}m"
            ),
            "the-relief's-up-and-down-over-two-swaps:the-lid-beside-today's-N2[design-D-Q2]",
        );
    }
    // TOP-STEP at every swap 13 -> 14 ... 17 -> 18.
    for r in FAR_FINE..FAR_LAST {
        let Some(coarse_level) = lid.level(r + 1 - FAR_FINE) else {
            continue;
        };
        let mut steps: Vec<f64> = Vec::new();
        for b in 0..coarse_level.n {
            for a in 0..coarse_level.n {
                let (ci, cj) = (coarse_level.i0 + a, coarse_level.j0 + b);
                let Some(tc) = height_lid(r + 1, ci, cj) else {
                    continue;
                };
                for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    if let Some(tf) = height_lid(r, 2 * ci + da, 2 * cj + db) {
                        steps.push((tc - tf).abs() / cell(r + 1));
                    }
                }
            }
        }
        steps.sort_by(f64::total_cmp);
        let q = |p: f64| {
            steps
                .get(((steps.len() as f64 * p) as usize).min(steps.len().saturating_sub(1)))
                .copied()
                .unwrap_or(f64::NAN)
        };
        let max = steps.last().copied().unwrap_or(f64::NAN);
        c.judged(
            &format!("TOP-STEP.{}.W4.r{r:02}-r{:02}", stand.name, r + 1),
            &format!(
                "p99{:.3}/max{max:.3}coarser-cells/{}pairs",
                q(0.99),
                steps.len()
            ),
            "max<=1-coarser-cell[T7-rule-2]",
            max <= 1.0,
        );
    }
    // TOP-APPROX: T~ against T.
    for k in 1..=(FAR_LAST - FAR_FINE) {
        let Some(lv) = lid.level(k) else {
            continue;
        };
        let mut diffs: Vec<f64> = Vec::new();
        let mut mixed = 0usize;
        for b in 0..lv.n {
            for a in 0..lv.n {
                let (i, j) = (lv.i0 + a, lv.j0 + b);
                mixed += usize::from(lid.mixed(k, i, j) == Some(true));
                if let (Some(t), Some(tt)) = (
                    lid.height(&charter, k, i, j, false),
                    lid.height(&charter, k, i, j, true),
                ) {
                    diffs.push((tt - t).abs());
                }
            }
        }
        diffs.sort_by(f64::total_cmp);
        let q = |p: f64| {
            diffs
                .get(((diffs.len() as f64 * p) as usize).min(diffs.len().saturating_sub(1)))
                .copied()
                .unwrap_or(f64::NAN)
        };
        let max = diffs.last().copied().unwrap_or(f64::NAN);
        let r = FAR_FINE + k;
        c.reading(
            &format!("TOP-APPROX.{}.W4.r{r:02}", stand.name),
            &format!(
                "p99{:.1}m/max{max:.1}m/{:.3}px({}nodes,{mixed}part-wet)",
                q(0.99),
                max / cell(r),
                diffs.len()
            ),
            "the-interim-T~-against-the-exact-lid[design-D-Q3]",
        );
    }
    // POCKET-BASIN, and its control.
    let (mut pocket_nodes, mut lifted, mut control) = (0usize, 0usize, 0usize);
    if let Some(l1) = lid.level(1) {
        for b in 0..l1.n {
            for a in 0..l1.n {
                let (i, j) = (l1.i0 + a, l1.j0 + b);
                let all_pocket = [(0, 0), (1, 0), (0, 1), (1, 1)].iter().all(|&(da, db)| {
                    let node = node_at(&fine, win.face, 2 * i + da, 2 * j + db);
                    w.nodes.word[node as usize] == WORD_POCKET
                });
                if !all_pocket {
                    continue;
                }
                pocket_nodes += 1;
                let Some(t) = lid.height(&charter, 1, i, j, false) else {
                    continue;
                };
                let mut z = 0i64;
                for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let node = node_at(&fine, win.face, 2 * i + da, 2 * j + db);
                    z += i64::from(w.nodes.z_m[node as usize]);
                }
                let r = charter.radius;
                let own = metres(terrace(&charter.terrace, r + length_of(z as f64 / 4.0)) - r);
                lifted += usize::from(t > own + 0.5);
                control += usize::from(t.max(sea) > own + 0.5);
            }
        }
    }
    c.judged(
        &format!("POCKET-BASIN.{}.W4", stand.name),
        &format!("{lifted}of{pocket_nodes}pocket-level-1-nodes-lifted(control-held-over-the-sea:{control})"),
        "0,control>0[design-D§2.6;ruling-B7]",
        lifted == 0 && (pocket_nodes == 0 || control > 0),
    );
    let mut part_wet = 0usize;
    if let Some(l1) = lid.level(1) {
        for b in 0..l1.n {
            for a in 0..l1.n {
                part_wet += usize::from(lid.mixed(1, l1.i0 + a, l1.j0 + b) == Some(true));
            }
        }
    }
    c.reading(
        &format!("MIXED-LEVEL-NODES.{}.W4", stand.name),
        &format!(
            "{part_wet}part-wet-level-1-nodes/{:.1}KB-at-2B",
            part_wet as f64 * 2.0 / 1024.0
        ),
        "the-dry-word's-rows-in-this-window[design-D-Q3]",
    );
}
