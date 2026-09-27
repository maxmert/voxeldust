//! ★ THE SAMPLE COUNT PYRAMID OF A FAR WINDOW (the build design §3.2–§3.4, S0 item 5; ruling B14
//! item 3).
//!
//! A column over rung F folds the rung-F point sides of its whole footprint: 4 096 of them at rung
//! 13 and 2²² at rung 18 on the home planet, so no far column can key its own samples. The pyramid
//! keys each node cell's samples ONCE. A cell whose four corner nodes stand on one side of every
//! water of its block draws ONE key over the whole cell (the held read's maximum principle, §3.2),
//! so it keys one sample; a MIXED cell keys all of its samples. Each QUARTER cell (half a node a
//! side: one column of the rung under the node's rung) keeps three words: its wet samples, its lake
//! samples and the highest lake level among them. A far column sums the quarters under it and
//! names its water by the one naming rule, so every far rung reads one rule on one kind of word.
//!
//! **Example.** On the gentle coast a rung-16 column is 65 km a side: 8 × 8 node squares, 16 × 16
//! quarters. The column sums their words (530 000 of its 1 048 576 samples wet, all of them the
//! sea's), so it draws the sea.

use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use vd_bins::shore_rules::{
    FoldWords, NamedChoices, NodeSite, Nodes, Water, block_index, candidates, dweights,
    fold_by_kernel, one_rule_counts, point_side, weights_f64,
};
use vd_recipe::Gi;
use vd_seed::bend::Face;
use vd_terrain::macro_lattice::MacroLattice;

/// One quarter cell's three words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quarter {
    pub wet: u32,
    pub lake: u32,
    /// The highest lake level among the lake samples, `i16::MIN` for none.
    pub top: i16,
}

impl Default for Quarter {
    fn default() -> Quarter {
        Quarter {
            wet: 0,
            lake: 0,
            top: i16::MIN,
        }
    }
}

impl Quarter {
    /// One more sample of water key `key` (0 for dry).
    fn add(&mut self, key: u32) {
        match Water::of_key(key) {
            None => {}
            Some(Water::Sea) => self.wet += 1,
            Some(Water::Lake(l)) => {
                self.wet += 1;
                self.lake += 1;
                self.top = self.top.max(l);
            }
        }
    }

    /// `n` samples of one key.
    fn uniform(key: u32, n: u32) -> Quarter {
        match Water::of_key(key) {
            None => Quarter::default(),
            Some(Water::Sea) => Quarter {
                wet: n,
                lake: 0,
                top: i16::MIN,
            },
            Some(Water::Lake(l)) => Quarter {
                wet: n,
                lake: n,
                top: l,
            },
        }
    }
}

/// ★ N2's POINT KEY of one rung-F sample: the water its side says wet, 0 for dry. The same calls
/// as the column pass's own key (`Ctx::eval`'s `keys[3]`): the node site, the block, the banded
/// candidates and the point side. CHECK-PYRAMID-FOLD holds the two equal.
#[must_use]
pub fn n2_key(
    fine: &MacroLattice,
    nodes: &Nodes,
    face: Face,
    f_rung: u8,
    i: i32,
    j: i32,
    ch: &NamedChoices,
) -> u32 {
    let site = NodeSite::of(fine, face, f_rung, i, j);
    let block = site.block(fine);
    let (wa, wb) = (weights_f64(site.ta_q), weights_f64(site.tb_q));
    let (dwa, dwb) = (dweights(site.ta), dweights(site.tb));
    let words = if ch.word_band_m.is_finite() {
        *ch
    } else {
        ch.unbanded()
    };
    let cands = candidates(
        nodes, &block, site.ta, site.tb, &wa, &wb, &dwa, &dwb, &words,
    );
    let mut fields = [0.0f64; 8];
    for (k, c) in cands.as_slice().iter().enumerate() {
        fields[k] = c.s;
    }
    match point_side(&cands, &fields, nodes.sea_m) {
        Some((true, g)) => cands.list[g].water.key(),
        _ => 0,
    }
}

/// ★ THE PYRAMID of one far window: the quarters `[q0, q0 + nq)²` of rung `q_rung` on one face.
pub struct Pyramid {
    pub q_rung: u8,
    pub q0: (i32, i32),
    pub nq: i32,
    /// The samples one quarter holds.
    pub per_quarter: u64,
    pub quarters: Vec<Quarter>,
    /// The rung-F samples along a node edge, and half of it.
    pub per_node: i32,
    pub half: i32,
    /// The node cells `[cell0, cell0 + cn)²` the pyramid keyed, each cell's ONE class where it is
    /// uniform (0 dry, 1 sea, 2 lake), [`MIXED`] where it is not.
    pub cell0: (i32, i32),
    pub cn: i32,
    pub uniform: Vec<u8>,
    /// A mixed cell's samples' classes, row by row (the far-water design §5.1: levels 7-12 exact).
    pub mixed_classes: std::collections::HashMap<(i32, i32), Vec<u8>>,
}

/// A node cell whose samples do not share one class.
pub const MIXED: u8 = 255;

/// The skin's class of a water key: 0 dry, 1 the sea, 2 a lake.
#[must_use]
pub fn class_of(key: u32) -> u8 {
    match Water::of_key(key) {
        None => 0,
        Some(Water::Sea) => 1,
        Some(Water::Lake(_)) => 2,
    }
}

/// What building a pyramid measured.
#[derive(Clone, Debug, Default)]
pub struct Built {
    pub cells: usize,
    pub with_water: usize,
    pub mixed: usize,
    /// The node squares of the window that hold a quarter of a mixed cell (the shipped unit).
    pub squares_mixed: usize,
    pub squares: usize,
    pub samples_keyed: u64,
    /// CHECK-PYRAMID-UNIFORM: uniform cells keyed whole again, and their samples that differ.
    pub uniform_checked: usize,
    pub uniform_differ: u64,
    pub seconds: f64,
}

/// One cell's classification and its quarters.
struct CellOut {
    ni: i32,
    nj: i32,
    mixed: bool,
    water: bool,
    quarters: [Quarter; 4],
    keyed: u64,
    checked: bool,
    differ: u64,
    /// The uniform class, or the mixed cell's samples' classes.
    class: u8,
    classes: Option<Vec<u8>>,
}

impl Pyramid {
    /// ★ BUILD the quarters `[q0, q0 + nq)²` on `face` over `threads` threads: each node cell
    /// under them keyed once (one sample where it is uniform, every sample where it is mixed);
    /// every `check_every`-th uniform cell that holds a water keyed whole again as the check.
    #[must_use]
    #[allow(clippy::too_many_arguments, reason = "the window's own words")]
    pub fn build(
        fine: &MacroLattice,
        nodes: &Nodes,
        face: Face,
        f_rung: u8,
        q0: (i32, i32),
        nq: i32,
        ch: &NamedChoices,
        threads: usize,
        check_every: usize,
    ) -> (Pyramid, Built) {
        let t = Instant::now();
        let cpn = fine.cells_per_node;
        assert!(
            cpn.is_power_of_two(),
            "the pyramid reads a power-of-two node"
        );
        let node_rung = cpn.trailing_zeros() as u8;
        let q_rung = node_rung - 1;
        let per_node = (cpn >> f_rung) as i32;
        let half = per_node / 2;
        let cell_of = |q: i32| (q - 1).div_euclid(2);
        let (ci0, ci1) = (cell_of(q0.0), cell_of(q0.0 + nq - 1));
        let (cj0, cj1) = (cell_of(q0.1), cell_of(q0.1 + nq - 1));
        let rows: Vec<i32> = (cj0..=cj1).collect();
        let next = AtomicUsize::new(0);
        let water_seen = AtomicUsize::new(0);
        let results: Mutex<Vec<CellOut>> = Mutex::new(Vec::new());
        let key = |i: i32, j: i32| n2_key(fine, nodes, face, f_rung, i, j, ch);
        std::thread::scope(|sc| {
            for _ in 0..threads.max(1) {
                sc.spawn(|| {
                    let mut local: Vec<CellOut> = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= rows.len() {
                            break;
                        }
                        let nj = rows[k];
                        for ni in ci0..=ci1 {
                            let block = NodeSite {
                                face,
                                ni: i64::from(ni),
                                nj: i64::from(nj),
                                ta_q: Gi::ZERO,
                                tb_q: Gi::ZERO,
                                ta: 0.0,
                                tb: 0.0,
                            }
                            .block(fine);
                            let mut waters: Vec<Water> = Vec::new();
                            for &n in &block {
                                if let Some(b) = nodes.water_of(n)
                                    && !waters.contains(&b)
                                {
                                    waters.push(b);
                                }
                            }
                            let corners = [(0, 0), (1, 0), (0, 1), (1, 1)];
                            let mixed = waters.len() > ch.cap
                                || waters.iter().any(|&b| {
                                    let wet = corners
                                        .iter()
                                        .filter(|&&(oi, oj)| {
                                            nodes.wet_for(block[block_index(oi, oj)], b)
                                        })
                                        .count();
                                    wet > 0 && wet < 4
                                });
                            let (x0, y0) = (per_node * ni + half, per_node * nj + half);
                            let mut out = CellOut {
                                ni,
                                nj,
                                mixed,
                                water: !waters.is_empty(),
                                quarters: [Quarter::default(); 4],
                                keyed: 0,
                                checked: false,
                                differ: 0,
                                class: MIXED,
                                classes: None,
                            };
                            if mixed {
                                let mut classes = vec![0u8; (per_node * per_node) as usize];
                                for b in 0..per_node {
                                    for a in 0..per_node {
                                        let q = ((b / half) * 2 + a / half) as usize;
                                        let k = key(x0 + a, y0 + b);
                                        out.quarters[q].add(k);
                                        classes[(b * per_node + a) as usize] = class_of(k);
                                    }
                                }
                                out.keyed = (per_node * per_node) as u64;
                                out.classes = Some(classes);
                            } else {
                                let k0 = if waters.is_empty() { 0 } else { key(x0, y0) };
                                out.keyed = u64::from(!waters.is_empty());
                                out.class = class_of(k0);
                                out.quarters = [Quarter::uniform(k0, (half * half) as u32); 4];
                                if !waters.is_empty()
                                    && water_seen
                                        .fetch_add(1, Ordering::Relaxed)
                                        .is_multiple_of(check_every.max(1))
                                {
                                    out.checked = true;
                                    for b in 0..per_node {
                                        for a in 0..per_node {
                                            out.differ += u64::from(key(x0 + a, y0 + b) != k0);
                                        }
                                    }
                                }
                            }
                            local.push(out);
                        }
                    }
                    if let Ok(mut r) = results.lock() {
                        r.extend(local);
                    }
                });
            }
        });
        let cells = results.into_inner().unwrap_or_default();
        let cn = ci1 - ci0 + 1;
        let mut uniform = vec![0u8; (cn * cn) as usize];
        let mut mixed_classes = std::collections::HashMap::new();
        let mut quarters = vec![Quarter::default(); (nq * nq) as usize];
        let mut built = Built::default();
        let mut squares: BTreeSet<(i32, i32)> = BTreeSet::new();
        // The node squares of the window: rung-(q_rung + 1) columns [q0 / 2, (q0 + nq) / 2).
        let (s0i, s0j, sn) = (q0.0 / 2, q0.1 / 2, nq / 2);
        for cell in cells {
            let x = ((cell.nj - cj0) * cn + (cell.ni - ci0)) as usize;
            uniform[x] = cell.class;
            if let Some(classes) = cell.classes.clone() {
                mixed_classes.insert((cell.ni, cell.nj), classes);
            }
            let cell = &cell;
            built.cells += 1;
            built.with_water += usize::from(cell.water);
            built.mixed += usize::from(cell.mixed);
            built.samples_keyed += cell.keyed;
            built.uniform_checked += usize::from(cell.checked);
            built.uniform_differ += cell.differ;
            if cell.mixed {
                for (si, sj) in [
                    (cell.ni, cell.nj),
                    (cell.ni + 1, cell.nj),
                    (cell.ni, cell.nj + 1),
                    (cell.ni + 1, cell.nj + 1),
                ] {
                    if (s0i..s0i + sn).contains(&si) && (s0j..s0j + sn).contains(&sj) {
                        squares.insert((si, sj));
                    }
                }
            }
            for (k, q) in cell.quarters.iter().enumerate() {
                let (hi, hj) = ((k % 2) as i32, (k / 2) as i32);
                let (qi, qj) = (2 * cell.ni + 1 + hi, 2 * cell.nj + 1 + hj);
                let (a, b) = (qi - q0.0, qj - q0.1);
                if (0..nq).contains(&a) && (0..nq).contains(&b) {
                    quarters[(b * nq + a) as usize] = *q;
                }
            }
        }
        built.squares_mixed = squares.len();
        built.squares = (sn * sn) as usize;
        built.seconds = t.elapsed().as_secs_f64();
        (
            Pyramid {
                q_rung,
                q0,
                nq,
                per_quarter: (half * half) as u64,
                quarters,
                per_node,
                half,
                cell0: (ci0, cj0),
                cn,
                uniform,
                mixed_classes,
            },
            built,
        )
    }

    /// ★ THE THREE WORDS SUMMED under column `(i, j)` of rung `rung` (over the quarter's rung):
    /// its samples, its wet samples, its lake samples and the highest lake level among them.
    /// `None` where the column's footprint leaves the pyramid.
    #[must_use]
    pub fn words(&self, rung: u8, i: i32, j: i32) -> Option<(u64, u64, u64, Option<i16>)> {
        if rung <= self.q_rung {
            return None;
        }
        let side = 1i32 << (rung - self.q_rung);
        let (a0, b0) = (i * side - self.q0.0, j * side - self.q0.1);
        if a0 < 0 || b0 < 0 || a0 + side > self.nq || b0 + side > self.nq {
            return None;
        }
        let (mut wet, mut lake, mut top) = (0u64, 0u64, i16::MIN);
        for b in b0..b0 + side {
            let row = (b * self.nq) as usize;
            for q in &self.quarters[row + a0 as usize..row + (a0 + side) as usize] {
                wet += u64::from(q.wet);
                lake += u64::from(q.lake);
                top = top.max(q.top);
            }
        }
        let n = self.per_quarter * (side as u64) * (side as u64);
        Some((n, wet, lake, (top != i16::MIN).then_some(top)))
    }

    /// ★ THE CLASS of rung-F sample `(x, y)`: 0 dry, 1 the sea, 2 a lake; `None` outside the cells
    /// the pyramid keyed.
    #[must_use]
    pub fn class_at(&self, x: i64, y: i64) -> Option<u8> {
        let (pn, h) = (i64::from(self.per_node), i64::from(self.half));
        let (ni, nj) = ((x - h).div_euclid(pn), (y - h).div_euclid(pn));
        let (a, b) = (ni - i64::from(self.cell0.0), nj - i64::from(self.cell0.1));
        let cn = i64::from(self.cn);
        if !(0..cn).contains(&a) || !(0..cn).contains(&b) {
            return None;
        }
        let u = self.uniform[(b * cn + a) as usize];
        if u != MIXED {
            return Some(u);
        }
        let classes = self.mixed_classes.get(&(ni as i32, nj as i32))?;
        let (sa, sb) = (x - h - ni * pn, y - h - nj * pn);
        Some(classes[(sb * pn + sa) as usize])
    }

    /// ★ THE SKIN'S WORDS of texel `(i, j)` of level `l` (F ..= the quarter's level): its wet
    /// samples, its lake samples and its samples, counted from the cells' classes. A texel whose
    /// cell is uniform is read in one step.
    #[must_use]
    pub fn texel(&self, f_rung: u8, l: u8, i: i64, j: i64) -> Option<(u64, u64, u64)> {
        if l < f_rung || l > self.q_rung {
            return None;
        }
        let side = 1i64 << (l - f_rung);
        let n = (side * side) as u64;
        let (x0, y0) = (i << (l - f_rung), j << (l - f_rung));
        // A texel of these levels lies in one node cell: read a uniform one in one step.
        let first = self.class_at(x0, y0)?;
        let (pn, h) = (i64::from(self.per_node), i64::from(self.half));
        let (ni, nj) = ((x0 - h).div_euclid(pn), (y0 - h).div_euclid(pn));
        let (a, b) = (ni - i64::from(self.cell0.0), nj - i64::from(self.cell0.1));
        if self.uniform[(b * i64::from(self.cn) + a) as usize] != MIXED {
            let wet = if first > 0 { n } else { 0 };
            let lake = if first == 2 { n } else { 0 };
            return Some((wet, lake, n));
        }
        let (mut wet, mut lake) = (0u64, 0u64);
        for dy in 0..side {
            for dx in 0..side {
                let c = self.class_at(x0 + dx, y0 + dy)?;
                wet += u64::from(c > 0);
                lake += u64::from(c == 2);
            }
        }
        Some((wet, lake, n))
    }

    /// ★ THE FAR FOLD of column `(i, j)` of rung `rung`: the one naming rule on the summed words.
    #[must_use]
    pub fn fold(&self, rung: u8, i: i32, j: i32) -> Option<(bool, Option<Water>)> {
        let (n, wet, lake, top) = self.words(rung, i, j)?;
        Some(one_rule_counts(n, wet, lake, top))
    }

    /// ★ THE KERNEL'S FOLD of column `(i, j)` (ruling B15 item 1): the one naming rule on the words
    /// `kernel_words` reads from the column and its eight neighbours at its rung.
    #[must_use]
    pub fn fold_kernel(
        &self,
        kernel: u8,
        rung: u8,
        i: i32,
        j: i32,
    ) -> Option<(bool, Option<Water>)> {
        self.words(rung, i, j)?;
        let mut words: [[Option<FoldWords>; 3]; 3] = [[None; 3]; 3];
        for (b, row) in words.iter_mut().enumerate() {
            for (a, cell) in row.iter_mut().enumerate() {
                *cell = self.words(rung, i + a as i32 - 1, j + b as i32 - 1);
            }
        }
        fold_by_kernel(kernel, &words)
    }
}
