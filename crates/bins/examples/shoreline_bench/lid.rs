//! ★ THE LID (the far-water design D §2.6, slice K0b): from rung R_P a far column's surface is the
//! mean of what a pilot sees over it — the water's level over its wet fine nodes, the terraced mean
//! ground over its dry fine nodes:
//!
//!   T = (Σ_wet W + n_dry · (terrace(R + z̄_dry) − R)) / n.
//!
//! No side word, fold, hold, stencil or pocket word decides a far height, so no count flips a far
//! column at a swap. The bench builds the lid EXACTLY from the rows (z̄_dry the mean of the dry
//! fine nodes) and in its INTERIM form T~ from the words a client holds without a new datum (for a
//! part-wet level-1 node, the shipped level-1 height word — the mean of ALL four fine nodes — stands
//! for z̄_dry), and reads both against today's far land.
//!
//! **Example.** The second pocket's floor stands 8.2 km under the sea's level. Every level node
//! over the floor is all dry, so its lid is the floor's own terraced mean: a basin at every rung.
//! A node over the pocket's rim holds 30 % sea: its lid stands 30 % of the way from the rim's mean
//! to the sea's level, at every rung, and no swap lifts or drops it.

use vd_bins::shore_rules::{Nodes, length_of, metres};
use vd_recipe::plan::PlanCharter;
use vd_recipe::terrace::terrace;
use vd_seed::bend::Face;
use vd_terrain::artifact::{Artifact, node_at};
use vd_terrain::macro_lattice::MacroLattice;

/// ★ ONE LEVEL of the lid's sums: level-`k` nodes `[i0, i0 + n) × [j0, j0 + n)` (a level-k node
/// holds `4^k` fine nodes).
#[derive(Clone, Debug)]
pub struct LidLevel {
    pub k: u8,
    pub i0: i64,
    pub j0: i64,
    pub n: i64,
    pub n_wet: Vec<u32>,
    /// Σ of the wet fine nodes' water levels, metres.
    pub sum_w: Vec<i64>,
    pub n_dry: Vec<u32>,
    /// Σ of the dry fine nodes' heights, metres (the exact form).
    pub sum_z_dry: Vec<i64>,
    /// Σ over the level-1 nodes under it of n_dry(level-1) × that node's shipped height word (the
    /// interim form T~).
    pub sum_z_interim: Vec<i64>,
}

impl LidLevel {
    fn idx(&self, i: i64, j: i64) -> Option<usize> {
        let (a, b) = (i - self.i0, j - self.j0);
        ((0..self.n).contains(&a) && (0..self.n).contains(&b)).then(|| (b * self.n + a) as usize)
    }

    fn up(&self) -> LidLevel {
        let n = self.n / 2;
        let len = (n * n) as usize;
        let mut out = LidLevel {
            k: self.k + 1,
            i0: self.i0 / 2,
            j0: self.j0 / 2,
            n,
            n_wet: vec![0; len],
            sum_w: vec![0; len],
            n_dry: vec![0; len],
            sum_z_dry: vec![0; len],
            sum_z_interim: vec![0; len],
        };
        for b in 0..n {
            for a in 0..n {
                let k = (b * n + a) as usize;
                for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let c = ((2 * b + db) * self.n + 2 * a + da) as usize;
                    out.n_wet[k] += self.n_wet[c];
                    out.sum_w[k] += self.sum_w[c];
                    out.n_dry[k] += self.n_dry[c];
                    out.sum_z_dry[k] += self.sum_z_dry[c];
                    out.sum_z_interim[k] += self.sum_z_interim[c];
                }
            }
        }
        out
    }
}

/// ★ THE LID of a region of one face: levels 1 ..= top (level 0, the fine nodes, is folded in).
pub struct Lid {
    pub levels: Vec<LidLevel>,
}

impl Lid {
    /// The lid's sums over the fine nodes `[i0, i0 + n)²` of `face` (the origin and the side even
    /// multiples of `2^top`), from the rows and the shipped level-1 words.
    #[must_use]
    pub fn from_rows(
        nodes: &Nodes,
        artifact: &Artifact,
        fine: &MacroLattice,
        face: Face,
        (i0, j0, n): (i64, i64, i64),
        top: u8,
    ) -> Lid {
        let coarse = fine.coarser(1).expect("a level-1 lattice");
        let n1 = n / 2;
        let len = (n1 * n1) as usize;
        let mut l1 = LidLevel {
            k: 1,
            i0: i0 / 2,
            j0: j0 / 2,
            n: n1,
            n_wet: vec![0; len],
            sum_w: vec![0; len],
            n_dry: vec![0; len],
            sum_z_dry: vec![0; len],
            sum_z_interim: vec![0; len],
        };
        for b in 0..n1 {
            for a in 0..n1 {
                let k = (b * n1 + a) as usize;
                let (ci, cj) = (l1.i0 + a, l1.j0 + b);
                for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let node = node_at(fine, face, 2 * ci + da, 2 * cj + db);
                    match nodes.water_of(node) {
                        Some(w) => {
                            l1.n_wet[k] += 1;
                            l1.sum_w[k] += w.level_m(nodes.sea_m).round() as i64;
                        }
                        None => {
                            l1.n_dry[k] += 1;
                            l1.sum_z_dry[k] += i64::from(nodes.z_m[node as usize]);
                        }
                    }
                }
                // The shipped level-1 word: the floored mean of all four fine nodes.
                let word = if (0..i64::from(coarse.edge)).contains(&ci)
                    && (0..i64::from(coarse.edge)).contains(&cj)
                {
                    i64::from(
                        artifact.pyramid[0][coarse.index(face, ci as i32, cj as i32) as usize],
                    )
                } else {
                    // Off the face (the seam cells): the rows' own mean.
                    l1.sum_z_dry[k].div_euclid(i64::from(l1.n_dry[k].max(1)))
                };
                l1.sum_z_interim[k] = i64::from(l1.n_dry[k]) * word;
            }
        }
        let mut levels = vec![l1];
        while levels.last().is_some_and(|l| l.k < top && l.n >= 2) {
            let next = levels.last().map(LidLevel::up);
            levels.extend(next);
        }
        Lid { levels }
    }

    /// Level `k`.
    #[must_use]
    pub fn level(&self, k: u8) -> Option<&LidLevel> {
        self.levels.iter().find(|l| l.k == k)
    }

    /// ★ THE LID of level-`k` node `(i, j)`, metres over the ladder radius: exact (`interim` false)
    /// or T~ (`interim` true).
    #[must_use]
    pub fn height(
        &self,
        charter: &PlanCharter,
        k: u8,
        i: i64,
        j: i64,
        interim: bool,
    ) -> Option<f64> {
        let lv = self.level(k)?;
        let x = lv.idx(i, j)?;
        let n = 1i64 << (2 * u32::from(k));
        let n_dry = i64::from(lv.n_dry[x]);
        let dry = if n_dry == 0 {
            0.0
        } else {
            let sum = if interim {
                lv.sum_z_interim[x]
            } else {
                lv.sum_z_dry[x]
            };
            let mean = sum as f64 / n_dry as f64;
            let r = charter.radius;
            n_dry as f64 * metres(terrace(&charter.terrace, r + length_of(mean)) - r)
        };
        Some((lv.sum_w[x] as f64 + dry) / n as f64)
    }

    /// Whether level-`k` node `(i, j)` is part wet and part dry.
    #[must_use]
    pub fn mixed(&self, k: u8, i: i64, j: i64) -> Option<bool> {
        let lv = self.level(k)?;
        let x = lv.idx(i, j)?;
        Some(lv.n_wet[x] > 0 && lv.n_dry[x] > 0)
    }
}
