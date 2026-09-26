//! ★ W-SEAM, THE CUBE'S SEAM (the build design's S0 item 6). Every stand's window is pulled onto
//! its face (`stands::window`), so no line ever read a column beside a cube edge. W-SEAM stands on
//! the busiest stretch where a coast crosses a cube edge (`shore_rules::edge_coast`) and reads TWO
//! windows flush against that edge, one on each face, under today's rule and N2:
//! * the window's own check and rule lines on each side (`WSA` on the first face, `WSB` on the
//!   partner), CHECK-R0 among them on the chunks that touch the seam;
//! * SEAM-KEY and SEAM-STEP per rung: each edge column against its partner across the seam
//!   (`vd_seed::seam::across`), beside the same edge column against its neighbour one column
//!   inward — a seam that draws like the interior reads alike on both.
//!
//! **Example.** On face −Y's `v−` edge the sea crosses the cube edge 21 times in 32 nodes. At rung
//! 7 the edge column over the water and its partner on the next face must draw the same sea and
//! a surface step no larger than the steps between columns of one face.

use vd_bins::shore_rules::edge_coast;
use vd_seed::bend::Face;
use vd_seed::seam::{Edge, across, partner_of};

use crate::census::{Census, val};
use crate::eval::{self, FAMS, Fold, FoldMap, RungOut, Var};
use crate::lines::{self, LineCtx, Pool};
use crate::{World, metric};

/// The two window sets: (finest rung, last rung, columns a side at the finest).
const SETS: [(u8, u8, i32); 2] = [(3, 8, 1_024), (7, 13, 1_024)];

/// One flush window: its face, the edge it touches, and its origin and side at its finest rung.
#[derive(Clone, Copy, Debug)]
struct Flush {
    face: Face,
    edge: Edge,
    fine: u8,
    i0: i32,
    j0: i32,
    n: i32,
}

impl Flush {
    fn at(&self, r: u8) -> (i32, i32, i32) {
        let k = r - self.fine;
        (self.i0 >> k, self.j0 >> k, self.n >> k)
    }
}

/// The flush window on `face` against `edge` whose along-edge range starts at `a0` (rung `fine`).
fn flush(face: Face, edge: Edge, fine: u8, cells: i32, a0: i32, n: i32) -> Flush {
    let across_0 = if edge.is_plus() { cells - n } else { 0 };
    let (i0, j0) = if edge.is_u() {
        (across_0, a0)
    } else {
        (a0, across_0)
    };
    Flush {
        face,
        edge,
        fine,
        i0,
        j0,
        n,
    }
}

/// The column of window output `out` at face column `(i, j)`.
fn col(out: &RungOut, i: i32, j: i32) -> Option<&eval::ColOut> {
    let (a, b) = (i - out.i0, j - out.j0);
    let n = out.n as i32;
    if (0..n).contains(&a) && (0..n).contains(&b) {
        out.cols[(b * n + a) as usize].as_ref()
    } else {
        None
    }
}

/// ★ W-SEAM.
pub fn run(c: &mut Census, w: &World, pool: &mut Pool) {
    let fine = w.body.macro_lattice().expect("a lattice");
    let ec = edge_coast(w.nodes, &fine);
    let Some((count, face, edge, along_node)) = ec.best else {
        c.say("\n== W-SEAM: no coast crosses a cube edge ==");
        return;
    };
    c.begin_stand("seam");
    c.say(format!(
        "\n==================== W-SEAM: face {face:?}, edge {edge:?}, node {along_node} along it ({count} straddling node pairs within 32 nodes; {} on the globe) ====================",
        ec.pairs
    ));
    let rec = partner_of(face, edge);
    let vars = [Var::R0, Var::N2];
    let cpn = i64::from(fine.cells_per_node);
    for (fine_r, last_r, n) in SETS {
        let cells = w.body.ladder().cells_per_edge(fine_r) as i32;
        // The along range: centred on the stand's node, aligned to the last rung's cells.
        let centre = ((along_node * cpn + cpn / 2) >> fine_r) as i32;
        let align = 1i32 << (last_r - fine_r);
        let a0 = ((centre - n / 2).div_euclid(align) * align).clamp(0, cells - n);
        let wa = flush(face, edge, fine_r, cells, a0, n);
        let wb = flush(rec.dst_face, rec.dst_edge, fine_r, cells, a0, n);
        c.say(format!(
            "\n-- W-SEAM rungs {fine_r}..={last_r}: WSA on {:?} from ({}, {}), WSB on {:?} ({:?}) from ({}, {}), {n} columns a side at rung {fine_r}",
            wa.face, wa.i0, wa.j0, wb.face, wb.edge, wb.i0, wb.j0
        ));
        let mut maps: [Option<FoldMap>; 2] = [None, None];
        for r in fine_r..=last_r {
            let mut outs: Vec<RungOut> = Vec::new();
            for (k, win) in [wa, wb].iter().enumerate() {
                let Some(ctx) = w.ctx(win.face, r, false) else {
                    continue;
                };
                let (i0, j0, nr) = win.at(r);
                let fold = Fold {
                    map: maps[k].as_ref(),
                    ctx_f: None,
                    pyramid: None,
                };
                let out = eval::eval_window(&ctx, i0, j0, nr, &fold, w.threads, true);
                let (m_i, m_j) = metric(&ctx, i0 + nr / 2, j0 + nr / 2);
                let lx = LineCtx {
                    stand: "seam",
                    window: if k == 0 { "WSA" } else { "WSB" },
                    rung: r,
                    face: win.face,
                    m_i,
                    m_j,
                    m_col: 0.5 * (m_i + m_j),
                    q_r: f64::from(vd_seed::ladder::cell_m(r)) / 128.0,
                    sea: w.artifact.sea().map(f64::from),
                    pocket_stand: false,
                    fine: &fine,
                    nodes: w.nodes,
                    vars: &vars,
                    f_rung: w.f_rung,
                    wall_rung: r == fine_r,
                    shape_rung: false,
                    choices: w.ch,
                    kink_band: &|_| None,
                    lattice_band: &|_| None,
                    shape_void: false,
                };
                lines::check_lines(c, &lx, &out, None);
                lines::rule_lines(c, &lx, &out, pool);
                if r == w.f_rung {
                    maps[k] = Some(FoldMap {
                        i0,
                        j0,
                        n: nr,
                        keys: out
                            .cols
                            .iter()
                            .map(|col| col.as_ref().map_or([0; FAMS], |o| o.keys))
                            .collect(),
                    });
                }
                outs.push(out);
            }
            if outs.len() == 2 {
                let cells_r = w.body.ladder().cells_per_edge(r) as i32;
                seam_lines(c, &outs[0], &outs[1], &wa, r, cells_r, &vars);
            }
        }
    }
    c.end_stand();
}

/// ★ SEAM-KEY and SEAM-STEP at one rung: each edge column of WSA against its partner across the
/// seam in WSB, and against its own neighbour one column inward. `cells` is the face's columns a
/// side at rung `r`.
fn seam_lines(
    c: &mut Census,
    a: &RungOut,
    b: &RungOut,
    wa: &Flush,
    r: u8,
    cells: i32,
    vars: &[Var; 2],
) {
    let (i0, j0, n) = wa.at(r);
    let (edge_at, inward) = if wa.edge.is_plus() {
        (cells - 1, cells - 2)
    } else {
        (0, 1)
    };
    for v in vars {
        let vi = v.idx();
        let (mut pairs, mut seam_diff, mut in_diff) = (0usize, 0usize, 0usize);
        let (mut seam_steps, mut in_steps): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        for t in 0..n {
            let along = if wa.edge.is_u() { j0 + t } else { i0 + t };
            let ((ei, ej), (ii, ij)) = if wa.edge.is_u() {
                ((edge_at, along), (inward, along))
            } else {
                ((along, edge_at), (along, inward))
            };
            let (Some(ce), Some(ci)) = (col(a, ei, ej), col(a, ii, ij)) else {
                continue;
            };
            let x = across(wa.face, wa.edge, along, cells);
            let Some(cp) = col(b, x.i, x.j) else {
                continue;
            };
            pairs += 1;
            let (de, dp, di) = (ce.drawn[vi], cp.drawn[vi], ci.drawn[vi]);
            if (t < 2 && vi == Var::R0.idx() && r < 13) || (r == 13 && vi == Var::N2.idx() && t < 6)
            {
                c.say(format!(
                    "   seam pair at rung {r}, along {along}: edge ({ei}, {ej}) b {:.1} h_raw {:.1} word {} | partner {:?} ({}, {}) b {:.1} h_raw {:.1} word {} | inward b {:.1}; R0 s {:.1}/{:.1} l {:.1}/{:.1} key {}/{}; N2 s {:.1}/{:.1} key {}/{}",
                    ce.shared.b,
                    ce.shared.h_raw,
                    ce.shared.word,
                    x.face,
                    x.i,
                    x.j,
                    cp.shared.b,
                    cp.shared.h_raw,
                    cp.shared.word,
                    ci.shared.b,
                    de.s,
                    dp.s,
                    de.l,
                    dp.l,
                    de.wet_key(),
                    dp.wet_key(),
                    ce.drawn[Var::N2.idx()].s,
                    cp.drawn[Var::N2.idx()].s,
                    ce.drawn[Var::N2.idx()].wet_key(),
                    cp.drawn[Var::N2.idx()].wet_key()
                ));
            }
            seam_diff += usize::from(de.wet_key() != dp.wet_key());
            in_diff += usize::from(de.wet_key() != di.wet_key());
            seam_steps.push((f64::from(de.s) - f64::from(dp.s)).abs());
            in_steps.push((f64::from(de.s) - f64::from(di.s)).abs());
        }
        seam_steps.sort_by(f64::total_cmp);
        in_steps.sort_by(f64::total_cmp);
        let q = |s: &[f64], p: f64| {
            if s.is_empty() {
                f64::NAN
            } else {
                s[((s.len() as f64 * p) as usize).min(s.len() - 1)]
            }
        };
        c.reading(
            &format!("SEAM-KEY.seam.{}.r{r:02}", v.name()),
            &format!("{seam_diff}of{pairs}pairs-across/{in_diff}of{pairs}pairs-inward"),
            "the-edge-column's-drawn-water-against-its-partner-across-the-seam,beside-its-inward-neighbour[S0-item-6]",
        );
        c.reading(
            &format!("SEAM-STEP.seam.{}.r{r:02}", v.name()),
            &format!(
                "across:p50{}/p99{}/max{}m;inward:p50{}/p99{}/max{}m",
                val(q(&seam_steps, 0.5), 2, ""),
                val(q(&seam_steps, 0.99), 2, ""),
                val(q(&seam_steps, 1.0), 2, ""),
                val(q(&in_steps, 0.5), 2, ""),
                val(q(&in_steps, 0.99), 2, ""),
                val(q(&in_steps, 1.0), 2, "")
            ),
            "the-drawn-surface's-step-across-the-seam-beside-the-step-inward[S0-item-6]",
        );
    }
}
