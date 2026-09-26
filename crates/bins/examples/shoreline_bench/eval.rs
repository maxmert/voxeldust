//! ★ ONE COLUMN UNDER EVERY RULE, AND A WINDOW OF THEM (plan §4, §5).
//!
//! The shipped column comes through `vd_bins::shore_geometry::RungReader` (the chunk builder's own
//! reads, then the ONE kernel). R0 is that column bit for bit. C0, N1 and O1 are computed from the
//! same column's parts (its ground `B`, its fine half `f`, its raw surface) and the node arrays, by
//! `vd_bins::shore_rules`. A window is read in chunk rows: each chunk row also builds its chunks
//! with `vd_terrain::chunk::column_field` on the artifact's field AND on the client cache's field,
//! so CHECK-R0 and CHECK-CACHE-FIELD read every chunk of every window at every rung.
//!
//! **Example.** The plain's W2 at rung 9 is 496 × 496 columns in nine chunk rows. Each row is one
//! task on one thread: it reads its columns under the seven rules and builds its chunks twice; a
//! column whose shipped surface differs from the reader's by one raw unit turns CHECK-R0 red.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use vd_bins::shore_geometry::{RungReader, ShippedColumn};
use vd_bins::shore_rules::{
    Cand, Cands, Family, Ground, NamedChoices, NodeSite, Nodes, RUNGS_MAX, WORD_POCKET, Water,
    block_index, candidate_for, candidates, contour_c0, dweights, factor_q, fine_terms, finish,
    finish_alone, fold, metres, o1_field, point_side, residuals, shore_f64, term_sum, unit_of_q,
    weights_f64, word_of,
};
use vd_client::artifact_book::SharedField;
use vd_recipe::Gi;
use vd_recipe::height::{OCTAVES_CAP, SIDE_SEA, SIDE_UNKNOWN, shore};
use vd_recipe::noise::NOISE_BITS;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{Side, ZField, column_water};
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey, column_field};

/// The rule variants the bench paints (plan §5): today, the pure contour, N1 and O1 with the fold
/// and without it, and O1 with a quarter-cell clamp.
pub const NVAR: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Var {
    R0,
    C0,
    N1,
    N1nf,
    O1,
    O1nf,
    O1q,
}

impl Var {
    pub const ALL: [Var; NVAR] = [
        Var::R0,
        Var::C0,
        Var::N1,
        Var::N1nf,
        Var::O1,
        Var::O1nf,
        Var::O1q,
    ];

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Var::R0 => "R0",
            Var::C0 => "C0",
            Var::N1 => "N1",
            Var::N1nf => "N1-nofold",
            Var::O1 => "O1",
            Var::O1nf => "O1-nofold",
            Var::O1q => "O1q",
        }
    }

    #[must_use]
    pub fn idx(self) -> usize {
        self as usize
    }

    /// Whether the variant differs from its fold twin only above rung F.
    #[must_use]
    pub fn nofold(self) -> bool {
        matches!(self, Var::N1nf | Var::O1nf)
    }
}

/// The three point-side families: N1, O1 (κ of O1), O1q (κ of O1q).
pub const FAMS: usize = 3;

/// A drawn column's flags.
pub const WET: u8 = 1;
pub const SIDE: u8 = 2;
pub const BINDS: u8 = 4;
pub const RAMP: u8 = 8;
pub const HALO: u8 = 16;

/// ★ ONE COLUMN AS ONE RULE DRAWS IT: the surface `s` and the standing water `l` (metres over the
/// ladder radius, NaN for none), the rule's field of its governing water signed by the side (NaN
/// where the rule has none), the drawn water's key, and the flags.
#[derive(Clone, Copy, Debug, Default)]
pub struct Drawn {
    pub s: f32,
    pub l: f32,
    pub g: f32,
    pub key: u32,
    pub flags: u8,
}

impl Drawn {
    /// Drawn wet: the sheet's own rule, `s < l`.
    #[must_use]
    pub fn wet(&self) -> bool {
        self.flags & WET != 0
    }
    /// The rule's side says wet.
    #[must_use]
    pub fn side(&self) -> bool {
        self.flags & SIDE != 0
    }
    /// The standing water.
    #[must_use]
    pub fn l(&self) -> Option<f64> {
        (!self.l.is_nan()).then_some(f64::from(self.l))
    }
    /// The drawn water's key where the column is drawn wet, else 0.
    #[must_use]
    pub fn wet_key(&self) -> u32 {
        if self.wet() { self.key } else { 0 }
    }
}

/// ★ WHAT EVERY RULE SHARES about one column: the ground `B`, the fine half `f`, the raw surface,
/// R0's hold and the nearest fine node's word.
#[derive(Clone, Copy, Debug, Default)]
pub struct Shared {
    pub b: f32,
    pub f: f32,
    pub h_raw: f32,
    pub hold: f32,
    pub word: u8,
}

/// ★ THE SELF-CHECKS of one window at one rung, summed over its columns and chunks.
#[derive(Clone, Debug, Default)]
pub struct Checks {
    pub columns: usize,
    pub unread: usize,
    pub base_disagree: usize,
    pub fine_split_checked: usize,
    pub fine_split_bad: usize,
    pub f64_max: f64,
    pub r0_cols: usize,
    pub r0_differ: usize,
    pub r0_chunks: usize,
    pub r0_refused: usize,
    pub cache_cols: usize,
    pub cache_differ: usize,
    pub control_r0: usize,
    pub control_cache: usize,
    pub control_cache_cols: usize,
    pub over_cap: usize,
    pub hist: [usize; 6],
    /// The what-if's own time per column (ns), by candidate count 0, 1, 2+.
    pub whatif_ns: [(f64, usize); 3],
}

impl Checks {
    pub fn merge(&mut self, o: &Checks) {
        self.columns += o.columns;
        self.unread += o.unread;
        self.base_disagree += o.base_disagree;
        self.fine_split_checked += o.fine_split_checked;
        self.fine_split_bad += o.fine_split_bad;
        self.f64_max = self.f64_max.max(o.f64_max);
        self.r0_cols += o.r0_cols;
        self.r0_differ += o.r0_differ;
        self.r0_chunks += o.r0_chunks;
        self.r0_refused += o.r0_refused;
        self.cache_cols += o.cache_cols;
        self.cache_differ += o.cache_differ;
        self.control_r0 += o.control_r0;
        self.control_cache += o.control_cache;
        self.control_cache_cols += o.control_cache_cols;
        self.over_cap += o.over_cap;
        for (h, oh) in self.hist.iter_mut().zip(o.hist) {
            *h += oh;
        }
        for (w, ow) in self.whatif_ns.iter_mut().zip(o.whatif_ns) {
            w.0 += ow.0;
            w.1 += ow.1;
        }
    }
}

/// ★ THE READ CONTEXT OF ONE RUNG ON ONE FACE: the shipped reader, the client cache's field of the
/// same rung (and the next level, the cache check's failing control), and the rules' words.
pub struct Ctx<'a> {
    pub body: &'a BodyDefinition,
    pub nodes: &'a Nodes,
    pub reader: RungReader<'a>,
    pub cache_field: Option<SharedField>,
    pub cache_next: Option<SharedField>,
    pub rung: u8,
    pub face: Face,
    pub ch: NamedChoices,
    pub r1: u8,
    pub f_rung: u8,
    pub octave_counts: [usize; RUNGS_MAX],
    pub first: usize,
    pub a_r: f64,
    pub sea: Option<f64>,
    pub sea_radius: Gi,
    pub cpn: u32,
    /// Read R0 alone (W3: the far window; its fold would need a rung-F map over 1 000 km).
    pub r0_only: bool,
}

/// Per-thread caches: the fine terms of the nodes, and the rung-F point keys read on demand.
#[derive(Default)]
pub struct Caches {
    pub node_terms: HashMap<u32, [Gi; OCTAVES_CAP]>,
    pub point_keys: HashMap<(i32, i32), [u32; FAMS]>,
}

/// ★ THE RUNG-F POINT SIDES of a window (the fold's samples): each family's wet water key at each
/// rung-F column of `[i0, i0 + n) × [j0, j0 + n)`, 0 for dry.
pub struct FoldMap {
    pub i0: i32,
    pub j0: i32,
    pub n: i32,
    pub keys: Vec<[u32; FAMS]>,
}

impl FoldMap {
    #[must_use]
    pub fn get(&self, i: i32, j: i32) -> Option<[u32; FAMS]> {
        let (a, b) = (i - self.i0, j - self.j0);
        ((0..self.n).contains(&a) && (0..self.n).contains(&b))
            .then(|| self.keys[(b * self.n + a) as usize])
    }
}

/// Where the fold's samples come from: a window's map, and a rung-F context for the samples it
/// does not hold (the transects).
#[derive(Clone, Copy)]
pub struct Fold<'a> {
    pub map: Option<&'a FoldMap>,
    pub ctx_f: Option<&'a Ctx<'a>>,
}

impl Fold<'_> {
    /// The rung-F point keys of column `(i, j)` of rung F.
    fn keys(&self, i: i32, j: i32, caches: &mut Caches) -> [u32; FAMS] {
        if let Some(k) = self.map.and_then(|m| m.get(i, j)) {
            return k;
        }
        if let Some(k) = caches.point_keys.get(&(i, j)) {
            return *k;
        }
        let k = self
            .ctx_f
            .and_then(|c| {
                let mut ck = Checks::default();
                c.eval(i, j, &Fold::none(), caches, &mut ck).map(|o| o.keys)
            })
            .unwrap_or([0; FAMS]);
        caches.point_keys.insert((i, j), k);
        k
    }

    #[must_use]
    pub fn none() -> Fold<'static> {
        Fold {
            map: None,
            ctx_f: None,
        }
    }
}

/// ★ ONE COLUMN'S OUTPUT: what every rule shares, what each rule draws, the three families' point
/// keys (the fold's samples at rung F), and R0's surface and water radii for CHECK-R0.
#[derive(Clone, Copy, Debug)]
pub struct ColOut {
    pub shared: Shared,
    pub drawn: [Drawn; NVAR],
    pub keys: [u32; FAMS],
    pub surface_q: Gi,
    pub water_q: Gi,
}

const NAN: f32 = f32::NAN;

impl Ctx<'_> {
    /// ★ ONE COLUMN UNDER EVERY RULE. `None` where the field does not hold the column's stencil.
    pub fn eval(
        &self,
        i: i32,
        j: i32,
        fold_src: &Fold,
        caches: &mut Caches,
        ck: &mut Checks,
    ) -> Option<ColOut> {
        let Some(c) = self.reader.column(i, j) else {
            ck.unread += 1;
            return None;
        };
        let t0 = Instant::now();
        let rq = self.reader.charter.radius;
        let b = metres(c.base - rq);
        let f = metres(c.fine);
        let h_raw = metres(c.raw - rq);
        let s0 = metres(c.surface - rq);
        let has_water = c.water != Gi::ZERO;
        let l0 = has_water.then(|| metres(c.water - rq));
        let hold_m = (c.hold != Gi::ZERO).then(|| metres(c.hold - rq));
        let word = word_of(c.side);
        ck.columns += 1;
        ck.base_disagree += usize::from(!c.base_agrees);
        // CHECK-R0-F64: the f64 shore on the same B, hold, raw and side word.
        ck.f64_max = ck
            .f64_max
            .max((shore_f64(b, hold_m, h_raw, c.side_word) - s0).abs());
        // CHECK-R0's failing control: the card's side word on the same column.
        ck.control_r0 += usize::from(shore(c.base, c.hold, c.raw, SIDE_UNKNOWN) != c.surface);
        // R0, bit for bit.
        let key0 = match l0 {
            None => 0,
            Some(l) => {
                if c.side == Some(Side::Lake) && c.water != self.sea_radius {
                    Water::Lake(l.round() as i16).key()
                } else {
                    1
                }
            }
        };
        let r0 = Drawn {
            s: s0 as f32,
            l: l0.map_or(NAN, |l| l as f32),
            g: NAN,
            key: key0,
            flags: if has_water && c.surface < c.water {
                WET
            } else {
                0
            } | if c.side_word == SIDE_SEA { SIDE } else { 0 },
        };
        if self.r0_only {
            return Some(ColOut {
                shared: Shared {
                    b: b as f32,
                    f: f as f32,
                    h_raw: h_raw as f32,
                    hold: hold_m.map_or(NAN, |h| h as f32),
                    word,
                },
                drawn: [r0; NVAR],
                keys: [0; FAMS],
                surface_q: c.surface,
                water_q: c.water,
            });
        }
        // The candidates.
        let fine = &self.reader.fine;
        let site = NodeSite::of(fine, self.face, self.rung, i, j);
        let block = site.block(fine);
        let (wa, wb) = (weights_f64(site.ta_q), weights_f64(site.tb_q));
        let (dwa, dwb) = (dweights(site.ta), dweights(site.tb));
        let cands = candidates(
            self.nodes, &block, site.ta, site.tb, &wa, &wb, &dwa, &dwb, &self.ch,
        );
        ck.hist[cands.waters.min(5)] += 1;
        ck.over_cap += usize::from(cands.waters > self.ch.cap);
        // D7: the dry column's standing water — the land rule, or none on a pocket's floor.
        let d7side = if word == WORD_POCKET {
            Side::Pocket
        } else {
            Side::Land
        };
        let d7q = column_water(
            self.body,
            &self.reader.lattice,
            self.reader.source.field(),
            self.face,
            self.rung,
            i,
            j,
            Some(d7side),
        )
        .water;
        let d7 = (d7q != Gi::ZERO).then(|| metres(d7q - rq));
        let ground = Ground {
            b,
            f,
            h_raw,
            base_q: c.base,
            fine_q: c.fine,
            radius_q: rq,
            d7,
            sea: self.sea,
            a_r: self.a_r,
        };
        // C0.
        let (sc, lc, wc) = contour_c0(&ground, &cands);
        let c0_wet = lc.is_some_and(|l| sc < l);
        let c0 = Drawn {
            s: sc as f32,
            l: lc.map_or(NAN, |l| l as f32),
            g: NAN,
            key: if lc.is_some() {
                wc.map_or(0, Water::key)
            } else {
                0
            },
            flags: if c0_wet { WET | SIDE } else { 0 },
        };
        // The families' fields: N1 reads S; O1 and O1q add the clamped residual.
        let mut fields = [[0.0f64; 8]; FAMS];
        let mut binds = [[false; 8]; FAMS];
        let mut rho: Option<[f64; RUNGS_MAX]> = None;
        if cands.len > 0 {
            let r = self.rho(&c, &block, &wa, &wb, caches, ck);
            for (k, cand) in cands.as_slice().iter().enumerate() {
                fields[0][k] = cand.s;
                for (fi, kappa) in [(1usize, self.ch.kappa_o1), (2, self.ch.kappa_o1q)] {
                    let (g, bnd) =
                        o1_field(cand.s, cand.g, &r, self.rung, self.r1, kappa, self.cpn);
                    fields[fi][k] = g;
                    binds[fi][k] = bnd;
                }
            }
            rho = Some(r);
        }
        let sea_m = self.nodes.sea_m;
        let points: [Option<(bool, usize)>; FAMS] =
            std::array::from_fn(|fi| point_side(&cands, &fields[fi], sea_m));
        let keys: [u32; FAMS] = std::array::from_fn(|fi| match points[fi] {
            Some((true, g)) => cands.list[g].water.key(),
            _ => 0,
        });
        // The fold of the rung-F point sides under this column (rungs over F).
        let folded: Option<[(bool, Option<Water>); FAMS]> = if self.rung > self.f_rung {
            let k = u32::from(self.rung - self.f_rung);
            let (fi0, fj0) = (i << k, j << k);
            let side = 1i32 << k;
            let mut samples: [Vec<u32>; FAMS] =
                std::array::from_fn(|_| Vec::with_capacity((side * side) as usize));
            for sj in 0..side {
                for si in 0..side {
                    let kk = fold_src.keys(fi0 + si, fj0 + sj, caches);
                    for (list, k) in samples.iter_mut().zip(kk) {
                        list.push(k);
                    }
                }
            }
            Some(std::array::from_fn(|fi| fold(&samples[fi], sea_m)))
        } else {
            None
        };
        // Each family under its side: the point side, or the fold.
        let mut family_out = |fam: usize, family: Family, use_fold: bool| -> Drawn {
            let mut ext: Cands = cands;
            let mut fext = fields[fam];
            let mut bext = binds[fam];
            let decision: Option<(bool, usize, f64)> = match (use_fold, folded) {
                (true, Some(fo)) => {
                    let (wet, water) = fo[fam];
                    match (wet, water) {
                        (true, Some(w)) => {
                            let gov = match ext.find(w) {
                                Some(g) => g,
                                None => {
                                    // The fold names a water the column's own block did not keep.
                                    let cand: Cand = candidate_for(
                                        self.nodes, &block, w, site.ta, site.tb, &wa, &wb, &dwa,
                                        &dwb, &self.ch,
                                    );
                                    let r = match rho {
                                        Some(r) => r,
                                        None => {
                                            let r = self.rho(&c, &block, &wa, &wb, caches, ck);
                                            rho = Some(r);
                                            r
                                        }
                                    };
                                    let (g, bnd) = if fam == 0 {
                                        (cand.s, false)
                                    } else {
                                        let kappa = if fam == 1 {
                                            self.ch.kappa_o1
                                        } else {
                                            self.ch.kappa_o1q
                                        };
                                        o1_field(
                                            cand.s, cand.g, &r, self.rung, self.r1, kappa, self.cpn,
                                        )
                                    };
                                    let at = ext.len.min(7);
                                    ext.list[at] = cand;
                                    ext.len = at + 1;
                                    fext[at] = g;
                                    bext[at] = bnd;
                                    at
                                }
                            };
                            Some((true, gov, -fext[gov].abs()))
                        }
                        _ if ext.len > 0 => {
                            let gov = least_distance(&ext, &fext, sea_m);
                            Some((false, gov, fext[gov].abs()))
                        }
                        _ => None,
                    }
                }
                _ => points[fam].map(|(w, g)| (w, g, fext[g])),
            };
            match decision {
                None => {
                    let (s, l) = finish_alone(&ground, &self.ch);
                    Drawn {
                        s: s as f32,
                        l: l.map_or(NAN, |l| l as f32),
                        g: NAN,
                        key: u32::from(l.is_some()),
                        flags: if l.is_some_and(|l| s < l) { WET } else { 0 },
                    }
                }
                Some((wet, gov, g_used)) => {
                    let fin = finish(
                        family,
                        &ground,
                        &self.reader.charter.terrace,
                        &ext,
                        &fext,
                        wet,
                        gov,
                        g_used,
                        &self.ch,
                    );
                    let key = match (wet, fin.l) {
                        (true, _) => ext.list[gov].water.key(),
                        (false, Some(_)) => 1,
                        (false, None) => 0,
                    };
                    let mut flags = 0u8;
                    if fin.l.is_some_and(|l| fin.s < l) {
                        flags |= WET;
                    }
                    if wet {
                        flags |= SIDE;
                    }
                    if fam > 0 && bext[gov] {
                        flags |= BINDS;
                    }
                    if fin.ramp {
                        flags |= RAMP;
                    }
                    if fin.beta > 0.01 {
                        flags |= HALO;
                    }
                    Drawn {
                        s: fin.s as f32,
                        l: fin.l.map_or(NAN, |l| l as f32),
                        g: g_used as f32,
                        key,
                        flags,
                    }
                }
            }
        };
        let n1 = family_out(0, Family::N1, true);
        let n1nf = family_out(0, Family::N1, false);
        let o1 = family_out(1, Family::O1, true);
        let o1nf = family_out(1, Family::O1, false);
        let o1q = family_out(2, Family::O1, true);
        let slot = cands.waters.min(2);
        ck.whatif_ns[slot].0 += t0.elapsed().as_nanos() as f64;
        ck.whatif_ns[slot].1 += 1;
        Some(ColOut {
            shared: Shared {
                b: b as f32,
                f: f as f32,
                h_raw: h_raw as f32,
                hold: hold_m.map_or(NAN, |h| h as f32),
                word,
            },
            drawn: [r0, c0, n1, n1nf, o1, o1nf, o1q],
            keys,
            surface_q: c.surface,
            water_q: c.water,
        })
    }

    /// O1's residuals at a column: the column's own fine terms, the stencil nodes' (cached), and
    /// CHECK-FINE-SPLIT (the terms' sum times the column's factor is the kernel's own fine half).
    fn rho(
        &self,
        c: &ShippedColumn,
        block: &[u32; 36],
        wa: &[f64; 4],
        wb: &[f64; 4],
        caches: &mut Caches,
        ck: &mut Checks,
    ) -> [f64; RUNGS_MAX] {
        let charter = &self.reader.charter;
        let m_q = factor_q(charter, c.dir, c.share);
        let count_r = self.octave_counts[usize::from(self.rung)];
        let col_terms = fine_terms(charter, self.first, count_r, c.dir);
        ck.fine_split_checked += 1;
        ck.fine_split_bad += usize::from(
            term_sum(&col_terms, self.first, count_r).mul_shr(m_q, NOISE_BITS) != c.fine,
        );
        let mut node_terms = [[Gi::ZERO; OCTAVES_CAP]; 16];
        let count0 = self.octave_counts[0];
        for (s, slot) in node_terms.iter_mut().enumerate() {
            let (oi, oj) = ((s % 4) as i64 - 1, (s / 4) as i64 - 1);
            let n = block[block_index(oi, oj)];
            *slot = *caches.node_terms.entry(n).or_insert_with(|| {
                fine_terms(charter, self.first, count0, self.reader.fine.direction(n))
            });
        }
        residuals(
            &col_terms,
            &node_terms,
            &self.octave_counts,
            self.first,
            self.rung,
            self.r1,
            unit_of_q(m_q),
            wa,
            wb,
        )
    }
}

/// ★ A DEVELOPMENT PROBE: one column's parts, its candidates and every rule's drawn words, printed
/// (`--probe FACE:RUNG:I:J`; not a census line).
pub fn probe(ctx: &Ctx, i: i32, j: i32) -> Vec<String> {
    let mut out = Vec::new();
    let Some(c) = ctx.reader.column(i, j) else {
        out.push(format!("probe ({i}, {j}) rung {}: not read", ctx.rung));
        return out;
    };
    let rq = ctx.reader.charter.radius;
    out.push(format!(
        "probe face {:?} rung {} ({i}, {j}): side {:?} word {:?}; B {:.2} m, f {:.2} m, raw {:.2} m, s_R0 {:.2} m, water {}, hold {}",
        ctx.face,
        ctx.rung,
        c.side,
        c.side_word,
        metres(c.base - rq),
        metres(c.fine),
        metres(c.raw - rq),
        metres(c.surface - rq),
        if c.water == Gi::ZERO { "none".to_string() } else { format!("{:.2}", metres(c.water - rq)) },
        if c.hold == Gi::ZERO { "none".to_string() } else { format!("{:.2}", metres(c.hold - rq)) },
    ));
    let fine = &ctx.reader.fine;
    let site = NodeSite::of(fine, ctx.face, ctx.rung, i, j);
    let block = site.block(fine);
    let (wa, wb) = (weights_f64(site.ta_q), weights_f64(site.tb_q));
    let (dwa, dwb) = (dweights(site.ta), dweights(site.tb));
    let cands = candidates(
        ctx.nodes, &block, site.ta, site.tb, &wa, &wb, &dwa, &dwb, &ctx.ch,
    );
    out.push(format!(
        "   node ({}, {}) + ({:.3}, {:.3}); {} waters in the block",
        site.ni, site.nj, site.ta, site.tb, cands.waters
    ));
    for cand in cands.as_slice() {
        let (d, beta) = vd_bins::shore_rules::distance_and_beta(cand, cand.s);
        out.push(format!(
            "   candidate {:?}: W {:.1} m, S {:.2} m, g {:.2} m/node, Omega {:.3}, d {:.3} nodes, beta {:.3}, wet corner {}",
            cand.water, cand.w, cand.s, cand.g, cand.omega, d, beta, cand.wet_corner
        ));
    }
    let mut zs = String::new();
    for row in (0..6).rev() {
        for col in 0..6 {
            let n = block[row * 6 + col];
            zs.push_str(&format!(
                "{:>6}{} ",
                ctx.nodes.z_m[n as usize],
                ["L", "S", "P", "K"][usize::from(ctx.nodes.word[n as usize])]
            ));
        }
        zs.push_str(" | ");
    }
    out.push(format!(
        "   the 6x6 block (rows from +j down; L land S sea P pocket K lake): {zs}"
    ));
    let mut caches = Caches::default();
    let mut ck = Checks::default();
    if let Some(o) = ctx.eval(i, j, &Fold::none(), &mut caches, &mut ck) {
        for v in Var::ALL {
            let d = o.drawn[v.idx()];
            out.push(format!(
                "   {:>10}: s {:.2} m, l {}, G {:.2}, key {}, wet {}, side {}, flags {:#07b}",
                v.name(),
                d.s,
                if d.l.is_nan() {
                    "none".to_string()
                } else {
                    format!("{:.2}", d.l)
                },
                d.g,
                d.key,
                d.wet(),
                d.side(),
                d.flags
            ));
        }
    }
    out
}

/// The candidate with the least distance to its line (a tie: the higher level).
fn least_distance(cands: &Cands, fields: &[f64; 8], sea_m: i16) -> usize {
    let list = cands.as_slice();
    let mut best = 0usize;
    for (k, c) in list.iter().enumerate().skip(1) {
        let dk = fields[k].abs() / c.g;
        let db = fields[best].abs() / list[best].g;
        if dk < db || (dk == db && c.water.level_m(sea_m) > list[best].water.level_m(sea_m)) {
            best = k;
        }
    }
    best
}

/// ★ ONE WINDOW AT ONE RUNG: its columns under every rule, row-major `b · n + a`, and its checks.
pub struct RungOut {
    pub rung: u8,
    pub i0: i32,
    pub j0: i32,
    pub n: usize,
    pub cols: Vec<Option<ColOut>>,
    pub checks: Checks,
    pub seconds: f64,
}

/// One chunk row's work.
struct TaskOut {
    rows: std::ops::Range<i32>,
    cols: Vec<Option<ColOut>>,
    checks: Checks,
}

/// ★ THE WINDOW'S PASS: chunk rows as tasks over `threads` threads (each column a pure function of
/// its address, so the output is the same on any thread count); each task also builds its chunks
/// on the artifact's field and on the cache's (CHECK-R0, CHECK-CACHE-FIELD).
#[must_use]
pub fn eval_window(
    ctx: &Ctx,
    i0: i32,
    j0: i32,
    n: i32,
    fold_src: &Fold,
    threads: usize,
    checks: bool,
) -> RungOut {
    let t = Instant::now();
    let edge = CHUNK_EDGE as i32;
    let tasks: Vec<i32> = (j0.div_euclid(edge)..=(j0 + n - 1).div_euclid(edge)).collect();
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<TaskOut>>> = Mutex::new((0..tasks.len()).map(|_| None).collect());
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let mut caches = Caches::default();
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= tasks.len() {
                        break;
                    }
                    let out = task(ctx, tasks[k], i0, j0, n, fold_src, &mut caches, checks);
                    if let Ok(mut r) = results.lock() {
                        r[k] = Some(out);
                    }
                }
            });
        }
    });
    let mut cols = Vec::with_capacity((n * n) as usize);
    let mut all = Checks::default();
    for out in results
        .into_inner()
        .unwrap_or_default()
        .into_iter()
        .flatten()
    {
        let _ = &out.rows;
        cols.extend(out.cols);
        all.merge(&out.checks);
    }
    RungOut {
        rung: ctx.rung,
        i0,
        j0,
        n: n as usize,
        cols,
        checks: all,
        seconds: t.elapsed().as_secs_f64(),
    }
}

#[allow(clippy::too_many_arguments, reason = "one task's own words")]
fn task(
    ctx: &Ctx,
    y: i32,
    i0: i32,
    j0: i32,
    n: i32,
    fold_src: &Fold,
    caches: &mut Caches,
    checks: bool,
) -> TaskOut {
    let edge = CHUNK_EDGE as i32;
    let rows = j0.max(y * edge)..(j0 + n).min(y * edge + edge);
    let mut ck = Checks::default();
    let mut cols = Vec::with_capacity(rows.len() * n as usize);
    for j in rows.clone() {
        for i in i0..i0 + n {
            cols.push(ctx.eval(i, j, fold_src, caches, &mut ck));
        }
    }
    if checks {
        let field = ctx.reader.source.field();
        for x in i0.div_euclid(edge)..=(i0 + n - 1).div_euclid(edge) {
            let shipped = column_field(ctx.body, Some(field), ctx.face, ctx.rung, x, y);
            let Some(shipped) = shipped else {
                ck.r0_refused += 1;
                continue;
            };
            ck.r0_chunks += 1;
            let cached = ctx.cache_field.as_ref().and_then(|f| {
                column_field(
                    ctx.body,
                    Some(&**f as &dyn ZField),
                    ctx.face,
                    ctx.rung,
                    x,
                    y,
                )
            });
            for b in 0..edge {
                let jj = y * edge + b;
                if !rows.contains(&jj) {
                    continue;
                }
                for a in 0..edge {
                    let ii = x * edge + a;
                    if !(i0..i0 + n).contains(&ii) {
                        continue;
                    }
                    let k = (b * edge + a) as usize;
                    let at = ((jj - rows.start) * n + (ii - i0)) as usize;
                    ck.r0_cols += 1;
                    match &cols[at] {
                        Some(o) => {
                            ck.r0_differ += usize::from(
                                o.surface_q != shipped.columns[k].1
                                    || o.water_q != shipped.water[k],
                            );
                        }
                        None => ck.r0_differ += 1,
                    }
                    if let Some(cf) = &cached {
                        ck.cache_cols += 1;
                        ck.cache_differ += usize::from(
                            cf.columns[k].1 != shipped.columns[k].1
                                || cf.water[k] != shipped.water[k],
                        );
                    }
                }
            }
            // The cache check's failing control: the next level's field on the first chunk only.
            if ck.control_cache_cols == 0
                && let Some(next) = ctx.cache_next.as_ref()
                && let Some(nf) = column_field(
                    ctx.body,
                    Some(&**next as &dyn ZField),
                    ctx.face,
                    ctx.rung,
                    x,
                    y,
                )
            {
                for (p, q) in nf.columns.iter().zip(&shipped.columns) {
                    ck.control_cache_cols += 1;
                    ck.control_cache += usize::from(p.1 != q.1);
                }
            }
        }
    }
    TaskOut {
        rows,
        cols,
        checks: ck,
    }
}

/// ★ CHECK-SHEET-COLUMNS (plan §4.1): one chunk's sample box against the reader on all 64 × 64
/// columns (the halo included), then every ground vertex's side word from `water_sheet_sided`
/// against the bench's own drawn R0 columns; and the control (the neighbour chunk's box differs).
/// Returns `(columns compared, columns differing, vertices compared, vertices differing, vertex
/// ties, control differs, boxes)`.
#[must_use]
pub fn sheet_check(ctx: &Ctx, x: i32, y: i32) -> Option<[usize; 7]> {
    use vd_terrain::lattice::{SampleBox, sample_box};
    use vd_terrain::position::{WaterSide, water_sheet_sided};
    let field = ctx.reader.source.field();
    let cf = column_field(ctx.body, Some(field), ctx.face, ctx.rung, x, y)?;
    let ladder = ctx.body.ladder();
    let cell = f64::from(vd_seed::ladder::cell_m(ctx.rung));
    let z_of = |r: Gi| -> i32 {
        let m = r.raw() as f64 / vd_bins::shore_rules::RAW_PER_M;
        (((m - f64::from(ladder.floor_m)) / cell).floor() as i32).div_euclid(CHUNK_EDGE as i32)
    };
    let (z_lo, z_hi) = (z_of(cf.lowest) - 1, z_of(cf.highest) + 1);
    let edge = CHUNK_EDGE as i32;
    // The reader's R0 drawn word for each box column: 1 under its water, 2 over it, 3 a tie.
    let mut word = vec![0u8; 64 * 64];
    let (mut cols, mut col_diff) = (0usize, 0usize);
    let mut first_box: Option<SampleBox> = None;
    let (mut verts, mut vdiff, mut vties, mut boxes) = (0usize, 0usize, 0usize, 0usize);
    for z in z_lo.max(0)..=z_hi {
        let key = ChunkKey {
            face: ctx.face,
            rung: ctx.rung,
            x,
            y,
            z,
        };
        let Some(sb) = sample_box(ctx.body, Some(field), key) else {
            continue;
        };
        boxes += 1;
        if first_box.is_none() {
            for b in -1..=edge {
                for a in -1..=edge {
                    let col = SampleBox::column_index(a, b);
                    let site = sb.site(a, b);
                    if Face::from_index(site.face) != Some(ctx.face) {
                        continue;
                    }
                    let Some(c) = ctx.reader.column(site.i, site.j) else {
                        continue;
                    };
                    cols += 1;
                    col_diff +=
                        usize::from(c.surface != sb.surfaces[col] || c.water != sb.water[col]);
                    let l = c.water;
                    word[col] = if c.surface < l {
                        1
                    } else if c.surface > l {
                        2
                    } else {
                        3
                    };
                }
            }
            first_box = Some(sb.clone());
        }
        let mesh = vd_terrain::extract(&sb);
        let (_, _, words) = water_sheet_sided(ctx.body, &sb, &mesh, Gi::ZERO);
        for (v, gw) in mesh.vertices.iter().zip(&words) {
            let group = [
                (i32::from(v[0]) >> 8).min(edge - 1),
                (i32::from(v[1]) >> 8).min(edge - 1),
            ];
            let (mut under, mut over, mut tie) = (0, 0, 0);
            for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                match word[SampleBox::column_index(group[0] + da, group[1] + db)] {
                    1 => under += 1,
                    2 => over += 1,
                    _ => tie += 1,
                }
            }
            let expect = if under == 4 {
                WaterSide::Under
            } else if over == 4 {
                WaterSide::Over
            } else {
                WaterSide::Mixed
            };
            verts += 1;
            if expect != gw.side {
                vdiff += 1;
                vties += usize::from(tie > 0);
            }
        }
    }
    // The control: the neighbour chunk's box read at this chunk's columns must differ.
    let control = sample_box(
        ctx.body,
        Some(field),
        ChunkKey {
            face: ctx.face,
            rung: ctx.rung,
            x: x + 1,
            y,
            z: z_lo.max(0),
        },
    )
    .zip(first_box.as_ref())
    .map_or(0, |(nb, ob)| {
        nb.surfaces
            .iter()
            .zip(&ob.surfaces)
            .filter(|(p, q)| p != q)
            .count()
    });
    Some([cols, col_diff, verts, vdiff, vties, control, boxes])
}
