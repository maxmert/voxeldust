//! ★ THE SHORELINE BENCH'S RULES — THE WHAT-IF OF ONE SHORELINE FROM ONE FIELD, AS NUMBERS (plan
//! step 4 slice 0 of `docs/investigation/2026-09-24/terrain_believability_status.md`; ruling B8
//! item 1; `docs/investigation/2026-09-24/shoreline_design.md` §19 "WHAT FOLLOWS";
//! `docs/investigation/2026-09-25/shoreline_papers_survey.md` §6.1).
//!
//! **WHY IT EXISTS.** Two paper rounds of the shoreline design found 47 faults, and each cure added
//! a mechanism the next round broke. So slice 0 is a PROTOTYPE BENCH: it paints the DRAWN water and
//! measures it. This module holds the candidate rules as plain functions of one column's inputs, so
//! the bench (`examples/shoreline_bench.rs`) and slice 1's integer re-measure read ONE reference.
//! Nothing here is a shipped path, and nothing here changes the world.
//!
//! **THE RULES** (survey §6.1; the formulas of the bench's plan §5):
//! * **R0** — today's shipped column (the bench reads it through `shore_geometry::RungReader`).
//! * **C0** — the pure contour: the drawn ground's own waterline, W6's hold off.
//! * **N1** — the node-scale side `S_b` as ONE field, and the ground blended toward `W_b + S_b`
//!   inside the presence.
//! * **O1(κ)** — N1's field plus the ground's own fine residual, each rung's step clamped to κ
//!   coarser cells laterally (κ = 1 is O1, κ = ¼ is O1q), the hold and the fade on that field.
//!
//! **THE UNITS.** Every height here is METRES over the ladder radius `R`, in `f64` (a what-if), except
//! where the recipe has a public integer function: the Catmull-Rom weights, the quintic fade and
//! the terrace are the recipe's own integers, converted at the edge.
//!
//! **THE NAMED CHOICES** (ruling T9: no hidden number) live in [`NamedChoices`], each printed with
//! its default and its source.
//!
//! **Example.** A column on the plain stands 2 km from a pond at 1 200 m. Its 6 × 6 node block holds
//! the pond's four lake nodes, so the pond is its one candidate. The held read of the pond's side
//! field says the column stands 30 m over the pond's line; the slope is 60 m a node, so the column is
//! half a node from the line, its presence is one and its fade 0.5. Under N1 its dunes shrink by
//! half and its surface is held over 1 200 m by a quarter of its blended ground's height; under O1
//! its dunes stay whole and the pond's edge may bend by one coarser cell toward it.

use vd_recipe::Gi;
use vd_recipe::height::{
    OCTAVES_CAP, SIDE_INLAND, SIDE_SEA, SIDE_UNKNOWN, relief_parts_from, roughness_factor,
};
use vd_recipe::macro_field::weights;
use vd_recipe::noise::{NOISE_BITS, NOISE_ONE, fade};
use vd_recipe::plan::PlanCharter;
use vd_recipe::terrace::{Terrace, terrace};
use vd_seed::bend::Face;
use vd_terrain::artifact::{Artifact, DRY_M, Side, coast_side, node_and_fraction, node_at};
use vd_terrain::macro_lattice::MacroLattice;

/// The noise's fraction as a float scale: `2^28`.
pub const NOISE_SCALE: f64 = (1u64 << NOISE_BITS) as f64;
/// One metre in the length format's raw units: 128 gap steps at 28 fraction bits.
pub const RAW_PER_M: f64 = 128.0 * NOISE_SCALE;
/// The most rungs a body has (the address's five bits name 0 to 21).
pub const RUNGS_MAX: usize = 22;
/// The most candidates a column may carry before the cap (the 6 × 6 block's 36 nodes).
const BLOCK: usize = 36;

/// A length at the length format as metres.
#[must_use]
pub fn metres(g: Gi) -> f64 {
    g.raw() as f64 / RAW_PER_M
}

/// Metres as a length at the length format, rounded to the nearest raw unit.
#[must_use]
pub fn length_of(m: f64) -> Gi {
    Gi::new((m * RAW_PER_M).round() as i64)
}

/// A fraction at the noise's bits as a float in `[0, 1]`.
#[must_use]
pub fn unit_of_q(t: Gi) -> f64 {
    t.raw() as f64 / NOISE_SCALE
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE NAMED CHOICES (ruling T9).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ EVERY NUMBER A RULE READS THAT NO LAW STATES, named once, printed in the bench's header with
/// its default and its source (ruling T9: a number no law states is a stated choice, never a hidden
/// constant). A run may override any of them with `--choice NAME=VALUE`, and the header prints the
/// override.
///
/// **Example.** The owner asks what a half in place of W6's quarter does to the plain's walls. The
/// bench runs `--choice k=0.5`; every line of that run carries the changed word in its header.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NamedChoices {
    /// W6's quarter: how far a column stays on its side, as a share of its distance from the water.
    pub k: f64,
    /// D5's `K`: how deep the fine octaves may dig a DRY column that stands under a water.
    pub big_k: f64,
    /// The least slope of the side field, metres per node (the rows' own unit).
    pub g_min: f64,
    /// The candidate cap (finding 23).
    pub cap: usize,
    /// O1's lateral bound in coarser cells.
    pub kappa_o1: f64,
    /// O1q's lateral bound in coarser cells (design §5.6).
    pub kappa_o1q: f64,
    /// The mirror over all sixteen words (1) or the four corners only (0; the bench's second slice).
    pub mirror_all: bool,
    /// The held read as the mean of both pass orders (1), or rows first only (0; finding 11).
    pub held_both_orders: bool,
    /// The fold starts this many rungs under R₁ (F = R₁ − `fold_under`; finding 7, D6).
    pub fold_under: u8,
    /// The f64 shore's band against the integer shore, metres (CHECK-R0-F64).
    pub f64_band_m: f64,
    /// The transects per water kind per stand (SHORE-STEP).
    pub transects: usize,
    /// The least spacing of the transects along the ground's own contour, metres.
    pub transect_spacing_m: f64,
    /// A transect's half-length over the sea and over a lake, in nodes.
    pub half_sea_nodes: f64,
    pub half_lake_nodes: f64,
    /// SHORE-STEP's match radius, in coarser cells (finding 8).
    pub match_cells: f64,
    /// SHORE-STEP samples the whole transect from this rung up; under it only windows of
    /// `window_cells` coarser cells around the coarser rung's crossings (O1's own telescoped bound
    /// holds a finer line within about two coarser cells of the coarser one).
    pub full_from: u8,
    pub window_cells: f64,
    /// WALL's reference quantile of the raw ground's neighbour steps, and its band (a share of the
    /// shore pairs), and the fewest pairs it is judged on (finding 19).
    pub wall_quantile: f64,
    pub wall_band_share: f64,
    pub wall_min_pairs: usize,
    /// The panels' byte budget (the bench refuses to write past it).
    pub panel_budget_bytes: u64,
}

impl Default for NamedChoices {
    fn default() -> NamedChoices {
        NamedChoices {
            k: 0.25,
            big_k: 4.0,
            g_min: 1.0,
            cap: 4,
            kappa_o1: 1.0,
            kappa_o1q: 0.25,
            mirror_all: true,
            held_both_orders: true,
            fold_under: 3,
            f64_band_m: 0.001,
            transects: 256,
            transect_spacing_m: 1_000.0,
            half_sea_nodes: 1.0,
            half_lake_nodes: 0.5,
            match_cells: 4.0,
            full_from: 5,
            window_cells: 8.0,
            wall_quantile: 0.999,
            wall_band_share: 0.001,
            wall_min_pairs: 1_000,
            panel_budget_bytes: 12 * 1024 * 1024,
        }
    }
}

impl NamedChoices {
    /// ★ ONE OVERRIDE `NAME=VALUE`; `Err` names what was refused.
    ///
    /// # Errors
    /// An unknown name or a value that does not parse.
    pub fn set(&mut self, pair: &str) -> Result<(), String> {
        let (name, value) = pair
            .split_once('=')
            .ok_or_else(|| format!("`{pair}` is not NAME=VALUE"))?;
        let f = || -> Result<f64, String> {
            value
                .parse::<f64>()
                .map_err(|_| format!("`{value}` is not a number"))
        };
        match name {
            "k" => self.k = f()?,
            "K" => self.big_k = f()?,
            "g_min" => self.g_min = f()?,
            "cap" => self.cap = (f()? as usize).clamp(1, 8),
            "kappa_o1" => self.kappa_o1 = f()?,
            "kappa_o1q" => self.kappa_o1q = f()?,
            "mirror_all" => self.mirror_all = f()? != 0.0,
            "held_both_orders" => self.held_both_orders = f()? != 0.0,
            "fold_under" => self.fold_under = f()? as u8,
            "f64_band_m" => self.f64_band_m = f()?,
            "transects" => self.transects = f()? as usize,
            "transect_spacing_m" => self.transect_spacing_m = f()?,
            "half_sea_nodes" => self.half_sea_nodes = f()?,
            "half_lake_nodes" => self.half_lake_nodes = f()?,
            "match_cells" => self.match_cells = f()?,
            "full_from" => self.full_from = f()? as u8,
            "window_cells" => self.window_cells = f()?,
            "wall_quantile" => self.wall_quantile = f()?,
            "wall_band_share" => self.wall_band_share = f()?,
            "wall_min_pairs" => self.wall_min_pairs = f()? as usize,
            "panel_budget_bytes" => self.panel_budget_bytes = f()? as u64,
            _ => return Err(format!("no named choice `{name}`")),
        }
        Ok(())
    }

    /// ★ THE HEADER'S LINES: every choice, its value and its source.
    #[must_use]
    pub fn describe(&self) -> Vec<String> {
        vec![
            format!(
                "k = {} (W6's quarter; the owner's word owed, design §15 Q3)",
                self.k
            ),
            format!(
                "K = {} (D5, 1/k; the owner's word owed, design §15 Q4)",
                self.big_k
            ),
            format!(
                "g_min = {} m per node (D4: the rows' own unit)",
                self.g_min
            ),
            "the fade = the recipe's quintic over ONE node of the first-order distance (D4: the read's own reach)".to_string(),
            format!(
                "the mirror = {} (finding 1 fix (a); the corners only is the bench's second slice)",
                if self.mirror_all {
                    "all sixteen words"
                } else {
                    "the four corners"
                }
            ),
            format!(
                "the held read = {} (finding 11)",
                if self.held_both_orders {
                    "the mean of both pass orders"
                } else {
                    "rows first only"
                }
            ),
            "the governing candidate = the lowest-level wet one; if dry, the least d; a tie goes to the higher level (W15.1)".to_string(),
            format!(
                "the candidate cap = {}, the nearest wet node first (finding 23; an overflow is counted, never dropped silently)",
                self.cap
            ),
            format!(
                "F = R1 - {} (finding 7; D6): the rungs from F up take the wet majority of their rung-F point sides, a tie wet (W15.1 item 4), the water with the most samples (a tie the higher level)",
                self.fold_under
            ),
            "the magnitude at a folded rung = the rung's own |G| of the fold's water at the column's centre, signed by the fold's side (this bench, stated)".to_string(),
            format!(
                "O1's clamp: telescoped from R1, c_s = kappa x one cell of rung s+1 in nodes; kappa = {} (O1), {} (O1q, design §5.6) (survey §6.1, finding 3)",
                self.kappa_o1, self.kappa_o1q
            ),
            "the terrace inside the band = faded with beta (one field at the line)".to_string(),
            "A_r = body.relief_bound(r) (D5: a charter fact, an over-bound)".to_string(),
            "a dry column's standing water = today's land rule (D7): the sea where its read stencil holds a sea node, else none; a pocket's floor none".to_string(),
            "the sheet's level over a column L_c = the highest l over its 3 x 3 columns (water_sheet_sided: a vertex's level is the highest of its four columns)".to_string(),
            format!(
                "transects: {} per water kind (the sea; the lakes pooled) per stand — the plan's 128 give about 128 matched pairs, under the 200 its band is judged on — spacing >= {} m along C0's B = W contour at rung 7 in W2; half-length {} node (sea), {} node (lake); sample step cell(r)/4; the whole transect from rung {} up, windows of +-{} coarser cells round the coarser rung's crossings under it",
                self.transects,
                self.transect_spacing_m,
                self.half_sea_nodes,
                self.half_lake_nodes,
                self.full_from,
                self.window_cells
            ),
            format!(
                "the SHORE-STEP match: within {} coarser cells; beyond that it is a fold event (finding 8)",
                self.match_cells
            ),
            format!(
                "WALL: Q_r = the p{} of the raw ground's neighbour steps in the window; band {} of the shore pairs; judged on >= {} pairs (finding 19)",
                self.wall_quantile * 100.0,
                self.wall_band_share,
                self.wall_min_pairs
            ),
            format!(
                "CHECK-R0-F64: the f64 shore against the integer one, max |d| <= {} m",
                self.f64_band_m
            ),
            format!(
                "the panels' budget: {} bytes (picture choices, not laws)",
                self.panel_budget_bytes
            ),
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE NODE ARRAYS AND THE WATERS.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The coast mask's words as bytes: land, the sea, a pocket's dry floor, a lake.
pub const WORD_LAND: u8 = 0;
pub const WORD_SEA: u8 = 1;
pub const WORD_POCKET: u8 = 2;
pub const WORD_LAKE: u8 = 3;

/// ★ THE GLOBAL NODE ARRAYS, built once: each fine node's height, its coast-mask word, and its
/// lake level (the row's `water_m` for a lake word, [`DRY_M`] for any other).
pub struct Nodes {
    pub z_m: Vec<i16>,
    pub word: Vec<u8>,
    pub level_m: Vec<i16>,
    /// The body's sea, whole metres over the ladder radius ([`DRY_M`] for none).
    pub sea_m: i16,
}

impl Nodes {
    /// The arrays of one artifact.
    #[must_use]
    pub fn of(artifact: &Artifact) -> Nodes {
        let n = artifact.node_count();
        let mut z_m = Vec::with_capacity(n);
        let mut word = Vec::with_capacity(n);
        let mut level_m = Vec::with_capacity(n);
        for (k, row) in artifact.rows.iter().enumerate() {
            z_m.push(row.z_m);
            let w = word_of(coast_side(&artifact.coast, k as u32));
            word.push(w);
            level_m.push(if w == WORD_LAKE { row.water_m } else { DRY_M });
        }
        Nodes {
            z_m,
            word,
            level_m,
            sea_m: artifact.sea_m,
        }
    }

    /// Whether node `n` is wet for water `b`: a sea word for the sea, a lake word of the same level
    /// for a lake.
    #[must_use]
    pub fn wet_for(&self, n: u32, b: Water) -> bool {
        let w = self.word[n as usize];
        match b {
            Water::Sea => w == WORD_SEA,
            Water::Lake(l) => (w == WORD_LAKE) & (self.level_m[n as usize] == l),
        }
    }

    /// The water node `n` is wet for, if any.
    #[must_use]
    pub fn water_of(&self, n: u32) -> Option<Water> {
        match self.word[n as usize] {
            WORD_SEA => Some(Water::Sea),
            WORD_LAKE => Some(Water::Lake(self.level_m[n as usize])),
            _ => None,
        }
    }
}

/// A coast-mask side as the bench's byte.
#[must_use]
pub fn word_of(side: Option<Side>) -> u8 {
    match side {
        Some(Side::Sea) => WORD_SEA,
        Some(Side::Pocket) => WORD_POCKET,
        Some(Side::Lake) => WORD_LAKE,
        _ => WORD_LAND,
    }
}

/// ★ ONE WATER (D1): the sea, or one lake level. Two lake nodes of one level are one water.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Water {
    #[default]
    Sea,
    Lake(i16),
}

impl Water {
    /// The water's flat level, metres over the ladder radius.
    #[must_use]
    pub fn level_m(self, sea_m: i16) -> f64 {
        match self {
            Water::Sea => f64::from(sea_m),
            Water::Lake(l) => f64::from(l),
        }
    }

    /// A compact key: 1 for the sea, `2 + (level + 32 768)` for a lake; 0 is kept for "no water".
    #[must_use]
    pub fn key(self) -> u32 {
        match self {
            Water::Sea => 1,
            Water::Lake(l) => 2 + (i32::from(l) + 32_768) as u32,
        }
    }

    /// The water a key names (`None` for 0).
    #[must_use]
    pub fn of_key(key: u32) -> Option<Water> {
        match key {
            0 => None,
            1 => Some(Water::Sea),
            k => Some(Water::Lake((k as i32 - 2 - 32_768) as i16)),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE COLUMN'S PLACE ON THE FINE LATTICE.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ WHERE A COLUMN STANDS ON THE FINE LATTICE: its face, the node at or before its centre along
/// each axis (`node_and_fraction`), and the fractions across the node cell, as the recipe's words
/// and as floats.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSite {
    pub face: Face,
    pub ni: i64,
    pub nj: i64,
    pub ta_q: Gi,
    pub tb_q: Gi,
    pub ta: f64,
    pub tb: f64,
}

impl NodeSite {
    /// The site of column `(i, j)` of `rung` on `face`.
    #[must_use]
    pub fn of(fine: &MacroLattice, face: Face, rung: u8, i: i32, j: i32) -> NodeSite {
        let (ni, ta_q) = node_and_fraction(fine, rung, i);
        let (nj, tb_q) = node_and_fraction(fine, rung, j);
        NodeSite {
            face,
            ni,
            nj,
            ta_q,
            tb_q,
            ta: unit_of_q(ta_q),
            tb: unit_of_q(tb_q),
        }
    }

    /// The nearest node (the one whose square holds the column), as `sample_side` rounds.
    #[must_use]
    pub fn nearest(&self, fine: &MacroLattice) -> u32 {
        let half = Gi::new(1i64 << (NOISE_BITS - 1));
        node_at(
            fine,
            self.face,
            self.ni + i64::from(self.ta_q >= half),
            self.nj + i64::from(self.tb_q >= half),
        )
    }

    /// ★ THE 6 × 6 NODE BLOCK `[ni − 2, ni + 3] × [nj − 2, nj + 3]` through the seam table, index
    /// `(oj + 2) · 6 + (oi + 2)`. It is the block the chunk already holds (design C11).
    #[must_use]
    pub fn block(&self, fine: &MacroLattice) -> [u32; BLOCK] {
        let mut out = [0u32; BLOCK];
        for (k, slot) in out.iter_mut().enumerate() {
            let (oi, oj) = ((k % 6) as i64 - 2, (k / 6) as i64 - 2);
            *slot = node_at(fine, self.face, self.ni + oi, self.nj + oj);
        }
        out
    }
}

/// The block index of offset `(oi, oj)`, each in `−2..=3`.
#[must_use]
pub const fn block_index(oi: i64, oj: i64) -> usize {
    ((oj + 2) * 6 + (oi + 2)) as usize
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE READS: the recipe's Catmull-Rom weights, the held read, the gradient.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The recipe's four Catmull-Rom weights at a fraction, as floats.
#[must_use]
pub fn weights_f64(t: Gi) -> [f64; 4] {
    weights(t).map(|w| w.raw() as f64 / NOISE_SCALE)
}

/// ★ THE DERIVATIVE WEIGHTS `w'(t) = ½[−3t²+4t−1, 9t²−10t, −9t²+8t+1, 3t²−2t]`: the slope of the
/// Catmull-Rom read per node.
#[must_use]
pub fn dweights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    [
        0.5 * (-3.0 * t2 + 4.0 * t - 1.0),
        0.5 * (9.0 * t2 - 10.0 * t),
        0.5 * (-9.0 * t2 + 8.0 * t + 1.0),
        0.5 * (3.0 * t2 - 2.0 * t),
    ]
}

/// One held pass: the weighted sum of four words, held inside its two MIDDLE words.
#[must_use]
pub fn held_pass(p: [f64; 4], w: &[f64; 4]) -> f64 {
    let v = w[0] * p[0] + w[1] * p[1] + w[2] * p[2] + w[3] * p[3];
    let (lo, hi) = if p[1] <= p[2] {
        (p[1], p[2])
    } else {
        (p[2], p[1])
    };
    v.clamp(lo, hi)
}

/// ★ THE HELD READ, ROWS FIRST (D2): each row along the first axis held inside its two middle
/// words, then the four rows along the second axis held inside the two middle rows. `z[row·4 +
/// col]`, the rows along the second axis, the columns along the first.
#[must_use]
pub fn held_rows_first(z: &[f64; 16], wa: &[f64; 4], wb: &[f64; 4]) -> f64 {
    let mut r = [0.0f64; 4];
    for (k, rk) in r.iter_mut().enumerate() {
        *rk = held_pass([z[k * 4], z[k * 4 + 1], z[k * 4 + 2], z[k * 4 + 3]], wa);
    }
    held_pass(r, wb)
}

/// The held read, COLUMNS first: the transpose's rows-first read.
#[must_use]
pub fn held_columns_first(z: &[f64; 16], wa: &[f64; 4], wb: &[f64; 4]) -> f64 {
    let mut q = [0.0f64; 4];
    for (c, qc) in q.iter_mut().enumerate() {
        *qc = held_pass([z[c], z[4 + c], z[8 + c], z[12 + c]], wb);
    }
    held_pass(q, wa)
}

/// ★ THE SYMMETRIC HELD READ (finding 11): the mean of both pass orders, or rows first alone.
/// Both terms stand inside the four corners' range, so the mean keeps the maximum principle; the
/// mean does not change under a transpose, so two faces that swap axes at a seam read one value.
#[must_use]
pub fn held_read(z: &[f64; 16], wa: &[f64; 4], wb: &[f64; 4], both: bool) -> f64 {
    let ab = held_rows_first(z, wa, wb);
    if both {
        0.5 * (ab + held_columns_first(z, wa, wb))
    } else {
        ab
    }
}

/// THE PLAIN Catmull-Rom read of sixteen words (the ground's own read, `sample_z`, in floats).
#[must_use]
pub fn plain_read(z: &[f64; 16], wa: &[f64; 4], wb: &[f64; 4]) -> f64 {
    let mut out = 0.0;
    for (k, wk) in wb.iter().enumerate() {
        let row =
            wa[0] * z[k * 4] + wa[1] * z[k * 4 + 1] + wa[2] * z[k * 4 + 2] + wa[3] * z[k * 4 + 3];
        out += wk * row;
    }
    out
}

/// ★ THE PLAIN READ'S GRADIENT per node along the two axes (the unheld Catmull-Rom's own
/// derivative weights: the held read's slope is zero where it holds).
#[must_use]
pub fn plain_gradient(
    z: &[f64; 16],
    wa: &[f64; 4],
    wb: &[f64; 4],
    dwa: &[f64; 4],
    dwb: &[f64; 4],
) -> (f64, f64) {
    let mut ga = 0.0;
    let mut gb = 0.0;
    for k in 0..4 {
        let row = [z[k * 4], z[k * 4 + 1], z[k * 4 + 2], z[k * 4 + 3]];
        let along = wa[0] * row[0] + wa[1] * row[1] + wa[2] * row[2] + wa[3] * row[3];
        let slope = dwa[0] * row[0] + dwa[1] * row[1] + dwa[2] * row[2] + dwa[3] * row[3];
        ga += wb[k] * slope;
        gb += dwb[k] * along;
    }
    (ga, gb)
}

/// The recipe's quintic fade of a float in `[0, 1]`, through the recipe's own integers.
#[must_use]
pub fn fade_f64(t: f64) -> f64 {
    let q = Gi::new((t.clamp(0.0, 1.0) * NOISE_SCALE).round() as i64);
    fade(q).raw() as f64 / NOISE_SCALE
}

/// ★ D5's λ: `k·g` over the water, `K·g` under it — continuous at zero from both sides.
#[must_use]
pub fn lambda(g: f64, k: f64, big_k: f64) -> f64 {
    if g >= 0.0 { k * g } else { big_k * g }
}

/// ★ THE MIRRORED NODE WORD (D2, finding 1): a node of water `b` is held a metre under its level,
/// a node not of `b` at or under the level stands mirrored over it by its own depth (at least a
/// metre), any other node reads its own height.
#[must_use]
pub fn mirrored(z: f64, mine: bool, w: f64) -> f64 {
    if mine {
        z.min(w - 1.0)
    } else if z <= w {
        w + (w - z).max(1.0)
    } else {
        z
    }
}

/// ★ THE SHORE LAW IN FLOATS (`vd_recipe::height::shore`, re-read for CHECK-R0-F64): `base` and
/// `h` metres over the ladder radius, `water` metres or `None` for no water, `side` the recipe's
/// word.
#[must_use]
pub fn shore_f64(base: f64, water: Option<f64>, h: f64, side: Gi) -> f64 {
    let Some(w) = water else {
        return h;
    };
    let g = base - w;
    if side == SIDE_INLAND && g < 0.0 {
        return h;
    }
    let d = h - w;
    let land = if side == SIDE_UNKNOWN {
        g >= 0.0
    } else {
        side != SIDE_SEA
    };
    let keep = g.abs() * 0.25;
    let toward = if land { d } else { -d };
    let held = toward.max(keep);
    w + if land { held } else { -held }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE CANDIDATES AND THEIR SIDE FIELDS (D1–D4).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ ONE CANDIDATE WATER of one column (D1–D3): its level, its node-scale side field `S_b` and
/// that field's slope, its presence `Ω_b`, whether the column's node cell has a corner node wet for
/// it, and how far its nearest wet node stands (Chebyshev, nodes) — the cap's order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cand {
    pub water: Water,
    pub w: f64,
    pub s: f64,
    pub g: f64,
    pub omega: f64,
    pub wet_corner: bool,
    pub near: f64,
    pub near_node: u32,
}

/// The candidates of one column: at most [`NamedChoices::cap`] of them, and how many waters the
/// block held before the cap.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cands {
    pub list: [Cand; 8],
    pub len: usize,
    pub waters: usize,
}

impl Cands {
    /// The kept candidates.
    #[must_use]
    pub fn as_slice(&self) -> &[Cand] {
        &self.list[..self.len]
    }

    /// The index of water `b` among the kept candidates.
    #[must_use]
    pub fn find(&self, b: Water) -> Option<usize> {
        self.as_slice().iter().position(|c| c.water == b)
    }
}

/// ★ THE CANDIDATES OF ONE COLUMN (D1, D2, D3). `block` is [`NodeSite::block`]; `(ta, tb)` the
/// column's fractions; `wa, wb` the recipe's weights there and `dwa, dwb` their derivatives.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the column's own words, each read once; a struct would only rename them"
)]
pub fn candidates(
    nodes: &Nodes,
    block: &[u32; BLOCK],
    ta: f64,
    tb: f64,
    wa: &[f64; 4],
    wb: &[f64; 4],
    dwa: &[f64; 4],
    dwb: &[f64; 4],
    ch: &NamedChoices,
) -> Cands {
    // The distinct waters of the block, each with its nearest wet node (distance, then index).
    let mut found: [(Water, f64, u32); BLOCK] = [(Water::Sea, 0.0, 0); BLOCK];
    let mut count = 0usize;
    for (k, &n) in block.iter().enumerate() {
        let Some(b) = nodes.water_of(n) else {
            continue;
        };
        let (oi, oj) = ((k % 6) as f64 - 2.0, (k / 6) as f64 - 2.0);
        let d = (oi - ta).abs().max((oj - tb).abs());
        match found[..count].iter_mut().find(|f| f.0 == b) {
            Some(f) => {
                if d < f.1 || (d == f.1 && n < f.2) {
                    f.1 = d;
                    f.2 = n;
                }
            }
            None => {
                found[count] = (b, d, n);
                count += 1;
            }
        }
    }
    let mut out = Cands {
        waters: count,
        ..Cands::default()
    };
    let kept = &mut found[..count];
    kept.sort_by(|x, y| x.1.total_cmp(&y.1).then(x.2.cmp(&y.2)));
    let cap = ch.cap.min(8);
    for &(b, near, near_node) in kept.iter().take(cap) {
        let mut c = candidate_for(nodes, block, b, ta, tb, wa, wb, dwa, dwb, ch);
        c.near = near;
        c.near_node = near_node;
        out.list[out.len] = c;
        out.len += 1;
    }
    out
}

/// ★ ONE WATER'S SIDE FIELD at one column (D2, D3): the mirrored stencil, the held read, the
/// plain read's slope, the presence and the wet corner. [`candidates`] calls it for each water it
/// keeps; the fold calls it for a water its samples name that the column's own block did not keep.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the column's own words, each read once; a struct would only rename them"
)]
pub fn candidate_for(
    nodes: &Nodes,
    block: &[u32; BLOCK],
    b: Water,
    ta: f64,
    tb: f64,
    wa: &[f64; 4],
    wb: &[f64; 4],
    dwa: &[f64; 4],
    dwb: &[f64; 4],
    ch: &NamedChoices,
) -> Cand {
    let w = b.level_m(nodes.sea_m);
    // The mirrored words of the sixteen-node stencil (offsets −1..=2).
    let mut zhat = [0.0f64; 16];
    for (s, slot) in zhat.iter_mut().enumerate() {
        let (oi, oj) = ((s % 4) as i64 - 1, (s / 4) as i64 - 1);
        let n = block[block_index(oi, oj)];
        let z = f64::from(nodes.z_m[n as usize]);
        let corner = (0..=1).contains(&oi) && (0..=1).contains(&oj);
        *slot = if ch.mirror_all || corner {
            mirrored(z, nodes.wet_for(n, b), w)
        } else {
            z
        };
    }
    let s = held_read(&zhat, wa, wb, ch.held_both_orders) - w;
    let (ga, gb) = plain_gradient(&zhat, wa, wb, dwa, dwb);
    let g = ga.hypot(gb).max(ch.g_min);
    // The presence: ω at the four corner nodes, then its bilinear read.
    let omega_at = |ci: i64, cj: i64| -> f64 {
        let mut any = false;
        for oj in cj - 2..=cj + 2 {
            for oi in ci - 2..=ci + 2 {
                any |= nodes.wet_for(block[block_index(oi, oj)], b);
            }
        }
        if any { 1.0 } else { 0.0 }
    };
    let (o00, o10, o01, o11) = (
        omega_at(0, 0),
        omega_at(1, 0),
        omega_at(0, 1),
        omega_at(1, 1),
    );
    let omega = (1.0 - ta) * (1.0 - tb) * o00
        + ta * (1.0 - tb) * o10
        + (1.0 - ta) * tb * o01
        + ta * tb * o11;
    let wet_corner = [(0, 0), (1, 0), (0, 1), (1, 1)]
        .iter()
        .any(|&(oi, oj)| nodes.wet_for(block[block_index(oi, oj)], b));
    Cand {
        water: b,
        w,
        s,
        g,
        omega,
        wet_corner,
        near: f64::INFINITY,
        near_node: u32::MAX,
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// O1: THE GROUND'S OWN FINE RESIDUAL, CLAMPED PER RUNG.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ THE RAW FINE TERMS at a direction: `terms[k]` is fine octave `k`'s own term at a share of ONE
/// (so the roughness factor multiplies nothing), for `k` in `[first, count)`, from the kernel's own
/// public function run over one octave at a time. The sum of the terms over `[first, count)` is the
/// kernel's raw fine sum exactly (integer adds); CHECK-FINE-SPLIT proves it on every column.
#[must_use]
pub fn fine_terms(
    charter: &PlanCharter,
    first: usize,
    count: usize,
    dir: [Gi; 3],
) -> [Gi; OCTAVES_CAP] {
    let mut out = [Gi::ZERO; OCTAVES_CAP];
    for (k, slot) in out
        .iter_mut()
        .enumerate()
        .take(count.min(OCTAVES_CAP))
        .skip(first)
    {
        *slot = relief_parts_from(
            &charter.octaves,
            k,
            k + 1,
            dir,
            &charter.roughness,
            NOISE_ONE,
        )
        .fine;
    }
    out
}

/// The column's roughness factor `m = max(roughness, share)` at the noise's bits — the very word
/// the kernel multiplies its fine half by.
#[must_use]
pub fn factor_q(charter: &PlanCharter, dir: [Gi; 3], share: Gi) -> Gi {
    vd_recipe::cell::greater(roughness_factor(&charter.roughness, dir), share)
}

/// The sum of raw terms over `[first, count)`.
#[must_use]
pub fn term_sum(terms: &[Gi; OCTAVES_CAP], first: usize, count: usize) -> Gi {
    let mut sum = Gi::ZERO;
    for t in terms.iter().take(count.min(OCTAVES_CAP)).skip(first) {
        sum += *t;
    }
    sum
}

/// ★ O1's RESIDUALS `ρ_s = m·Σ_s(x) − m·CR(Σ_s(n))(x)` for `s` in `[r, r1]` (`ρ_{r1} = 0`): the
/// fine sum the rung-`s` column keeps at its own direction, less its plain Catmull-Rom over the
/// sixteen stencil nodes' own sums — the octaves conditioned on the nodes, so no lift.
/// `counts[s]` is rung `s`'s octave count; `node_terms[row·4 + col]` the stencil nodes' raw terms.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the residual reads the column's and the stencil's own words"
)]
pub fn residuals(
    column_terms: &[Gi; OCTAVES_CAP],
    node_terms: &[[Gi; OCTAVES_CAP]; 16],
    counts: &[usize],
    first: usize,
    r: u8,
    r1: u8,
    m: f64,
    wa: &[f64; 4],
    wb: &[f64; 4],
) -> [f64; RUNGS_MAX] {
    let mut rho = [0.0f64; RUNGS_MAX];
    for s in r..r1 {
        let count = counts[usize::from(s)];
        if count <= first {
            continue;
        }
        let at = metres(term_sum(column_terms, first, count));
        let mut words = [0.0f64; 16];
        for (slot, terms) in words.iter_mut().zip(node_terms.iter()) {
            *slot = metres(term_sum(terms, first, count));
        }
        rho[usize::from(s)] = m * at - m * plain_read(&words, wa, wb);
    }
    rho
}

/// ★ O1's FIELD for one candidate: `G = S + Σ_{s=r}^{r1−1} clamp(ρ_s − ρ_{s+1}, ±c_s·g)`, with
/// `c_s = κ · cell(s+1) / node` in nodes; and whether any step's clamp binds.
#[must_use]
pub fn o1_field(
    s_b: f64,
    g_b: f64,
    rho: &[f64; RUNGS_MAX],
    r: u8,
    r1: u8,
    kappa: f64,
    cells_per_node: u32,
) -> (f64, bool) {
    let mut g = s_b;
    let mut binds = false;
    for s in r..r1 {
        let step = rho[usize::from(s)] - rho[usize::from(s) + 1];
        let c = kappa * f64::from(1u32 << (s + 1)) / f64::from(cells_per_node);
        let bound = c * g_b;
        binds |= step.abs() > bound;
        g += step.clamp(-bound, bound);
    }
    (g, binds)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE SIDE AND THE SURFACE.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// A candidate's distance to its line in nodes, and its blend `β = Ω·(1 − fade(min(1, d)))`.
#[must_use]
pub fn distance_and_beta(c: &Cand, g_field: f64) -> (f64, f64) {
    let d = g_field.abs() / c.g;
    (d, c.omega * (1.0 - fade_f64(d.min(1.0))))
}

/// ★ THE POINT SIDE of a column (plan §5.5): wet for the LOWEST-level candidate whose field is
/// under zero (a tie of levels: the sea first); else dry, governed by the candidate with the least
/// distance `d` (a tie: the higher level). `None` where the column has no candidate.
#[must_use]
pub fn point_side(cands: &Cands, fields: &[f64; 8], sea_m: i16) -> Option<(bool, usize)> {
    let list = cands.as_slice();
    if list.is_empty() {
        return None;
    }
    let mut wet: Option<usize> = None;
    for (k, c) in list.iter().enumerate() {
        if fields[k] < 0.0 {
            wet = match wet {
                Some(b) if list[b].w < c.w || (list[b].w == c.w && list[b].water <= c.water) => {
                    Some(b)
                }
                _ => Some(k),
            };
        }
    }
    if let Some(b) = wet {
        return Some((true, b));
    }
    let mut best = 0usize;
    for (k, c) in list.iter().enumerate().skip(1) {
        let dk = fields[k].abs() / c.g;
        let db = fields[best].abs() / list[best].g;
        if dk < db || (dk == db && c.water.level_m(sea_m) > list[best].water.level_m(sea_m)) {
            best = k;
        }
    }
    Some((false, best))
}

/// What one column hands the surface rule: its ground, its fine half, the raw surface, the rung's
/// words for the integer terrace, its dry standing water (D7), the body's sea and the relief bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ground {
    /// `B`: the node-scale ground (the solved `Z` and the coarse octaves), metres over `R`.
    pub b: f64,
    /// `f`: the fine half the kernel added, metres.
    pub f: f64,
    /// `h_raw = terrace(R + B + f) − R`, metres.
    pub h_raw: f64,
    /// `R + B` at the length format (the integer terrace's input with the fine half).
    pub base_q: Gi,
    /// The fine half at the length format.
    pub fine_q: Gi,
    /// The ladder radius at the length format.
    pub radius_q: Gi,
    /// D7: the dry column's standing water, metres, or `None`.
    pub d7: Option<f64>,
    /// The body's sea, metres, or `None` for a dry body.
    pub sea: Option<f64>,
    /// `A_r`: the relief bound at the rung, metres.
    pub a_r: f64,
}

/// Which surface family a rule belongs to: N1 fades the fine half by `β`; O1 keeps it whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    N1,
    O1,
}

/// What [`finish`] answers for one column: the surface `s` and the standing water `l` (metres over
/// `R`), the governing candidate's blend `β`, and whether a lake's dry hold bound the surface inside
/// a cell whose presence of that lake is between nought and one (finding 10's ramp).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Finished {
    pub s: f64,
    pub l: Option<f64>,
    pub beta: f64,
    pub ramp: bool,
}

/// ★ ONE COLUMN'S DRAWN SURFACE AND WATER under N1 or O1 (plan §5.6): the column is `wet` for the
/// governing candidate `gov` (or dry, governed by it), `g_used` that candidate's field signed by
/// the side.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the rule reads the column's ground, its candidates and the side, each once"
)]
pub fn finish(
    family: Family,
    ground: &Ground,
    terrace_rule: &Terrace,
    cands: &Cands,
    fields: &[f64; 8],
    wet: bool,
    gov: usize,
    g_used: f64,
    ch: &NamedChoices,
) -> Finished {
    let c = &cands.list[gov];
    let (_, beta) = distance_and_beta(c, g_used);
    let kept = match family {
        Family::N1 => (1.0 - beta) * metres(ground.fine_q),
        Family::O1 => metres(ground.fine_q),
    };
    let h_far = metres(terrace(terrace_rule, ground.base_q + length_of(kept)) - ground.radius_q);
    let h1 = (1.0 - beta) * h_far + beta * (c.w + g_used);
    if wet {
        return Finished {
            s: c.w - (c.w - h1).max(ch.k * g_used.abs()),
            l: Some(c.w),
            beta,
            ramp: false,
        };
    }
    // The dry holds: every lake candidate by its own blended ground, and the sea by its own.
    let mut lb = f64::NEG_INFINITY;
    let mut ramp_lb = f64::NEG_INFINITY;
    let mut sea_seen = false;
    for (k, b) in cands.as_slice().iter().enumerate() {
        let (_, beta_b) = distance_and_beta(b, fields[k]);
        let b1d = (1.0 - beta_b) * ground.b + beta_b * (b.w + fields[k].abs());
        let hold = match b.water {
            Water::Sea => {
                sea_seen = true;
                b.w + lambda(b1d - b.w, ch.k, ch.big_k)
            }
            Water::Lake(_) => {
                let h = b.w + lambda(b1d - b.w, ch.k, ch.big_k) - (1.0 - b.omega) * ground.a_r;
                if b.omega > 0.0 && b.omega < 1.0 {
                    ramp_lb = ramp_lb.max(h);
                }
                h
            }
        };
        lb = lb.max(hold);
    }
    if !sea_seen && let Some(sea) = ground.sea {
        lb = lb.max(sea + lambda(ground.b - sea, ch.k, ch.big_k));
    }
    Finished {
        s: h1.max(lb),
        l: ground.d7,
        beta,
        ramp: ramp_lb > h1 && ramp_lb >= lb,
    }
}

/// ★ A COLUMN WITH NO CANDIDATE (plan §5.6): `s = max(h_raw, LB_sea)`, `l = D7`.
#[must_use]
pub fn finish_alone(ground: &Ground, ch: &NamedChoices) -> (f64, Option<f64>) {
    let lb = ground.sea.map_or(f64::NEG_INFINITY, |sea| {
        sea + lambda(ground.b - sea, ch.k, ch.big_k)
    });
    (ground.h_raw.max(lb), ground.d7)
}

/// ★ C0, THE PURE CONTOUR (plan §5.7): the raw surface, the level of the LOWEST candidate the
/// raw surface stands under (else D7), and no basin guard.
#[must_use]
pub fn contour_c0(ground: &Ground, cands: &Cands) -> (f64, Option<f64>, Option<Water>) {
    let mut best: Option<&Cand> = None;
    for c in cands.as_slice() {
        if ground.h_raw < c.w && best.is_none_or(|b| c.w < b.w) {
            best = Some(c);
        }
    }
    match best {
        Some(c) => (ground.h_raw, Some(c.w), Some(c.water)),
        None => (ground.h_raw, ground.d7, ground.d7.map(|_| Water::Sea)),
    }
}

/// ★ THE FOLD (plan §5.5, D6 generalised): the wet majority of `n` point sides — wet when the wet
/// samples are at least half (a tie wet, W15.1 item 4) — and the water with the most samples (a
/// tie: the higher level, W15.1's lake-over-sea tie). `samples` yields each sample's water key (0
/// for a dry sample).
#[must_use]
pub fn fold(samples: &[u32], sea_m: i16) -> (bool, Option<Water>) {
    let n = samples.len();
    let mut counts: Vec<(u32, usize)> = Vec::new();
    let mut wet = 0usize;
    for &k in samples {
        if k == 0 {
            continue;
        }
        wet += 1;
        match counts.iter_mut().find(|c| c.0 == k) {
            Some(c) => c.1 += 1,
            None => counts.push((k, 1)),
        }
    }
    if wet * 2 < n || counts.is_empty() {
        return (false, None);
    }
    let level = |k: u32| Water::of_key(k).map_or(f64::NEG_INFINITY, |w| w.level_m(sea_m));
    let mut best = counts[0];
    for &c in &counts[1..] {
        if c.1 > best.1 || (c.1 == best.1 && level(c.0) > level(best.0)) {
            best = c;
        }
    }
    (true, Water::of_key(best.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed xorshift for the tests' random stencils.
    fn rng(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn q(t: f64) -> Gi {
        Gi::new((t * NOISE_SCALE) as i64)
    }

    /// ★ TEST 1: THE HELD READ KEEPS THE MAXIMUM PRINCIPLE over random stencils: it stands inside
    /// the four corners' range. CONTROL: the plain read of C7's example `W+100, W+1, W+1, W+100`
    /// reads `W − 11.4 m` at the half, under every word.
    #[test]
    fn the_held_read_keeps_the_maximum_principle() {
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        for _ in 0..100_000 {
            let mut z = [0.0f64; 16];
            for v in &mut z {
                *v = (rng(&mut s) - 0.5) * 2_000.0;
            }
            let (ta, tb) = (rng(&mut s), rng(&mut s));
            let (wa, wb) = (weights_f64(q(ta)), weights_f64(q(tb)));
            let v = held_read(&z, &wa, &wb, true);
            let corners = [z[5], z[6], z[9], z[10]];
            let lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(
                v >= lo - 1.0e-9 && v <= hi + 1.0e-9,
                "{v} outside [{lo}, {hi}]"
            );
        }
        // The control: C7's example along one axis, the same words in every row.
        let w = 500.0;
        let row = [w + 100.0, w + 1.0, w + 1.0, w + 100.0];
        let mut z = [0.0f64; 16];
        for k in 0..4 {
            z[k * 4..k * 4 + 4].copy_from_slice(&row);
        }
        let (wa, wb) = (weights_f64(q(0.5)), weights_f64(q(0.0)));
        let plain = plain_read(&z, &wa, &wb);
        assert!(
            (plain - (w - 11.375)).abs() < 1.0e-3,
            "the plain read {plain}"
        );
        let held = held_read(&z, &wa, &wb, true);
        assert!(held >= w + 1.0 - 1.0e-9, "the held read {held}");
    }

    /// ★ TEST 2: NODE-LINE CONTINUITY. On the shared edge of two neighbouring cells (the second
    /// cell's `ta = 0` against the first's `ta = 1`) the held read agrees. CONTROL: a clamp to the
    /// four CORNERS instead of each pass's two middle words disagrees on some stencil.
    #[test]
    fn the_held_read_is_continuous_across_a_node_line() {
        let mut s = 0x0F0F_1234_5555_7777u64;
        let mut control_differs = false;
        for _ in 0..20_000 {
            // Five columns of nodes; the first cell reads columns 0..4, the second 1..5.
            let mut grid = [[0.0f64; 5]; 4];
            for row in &mut grid {
                for v in row.iter_mut() {
                    *v = (rng(&mut s) - 0.5) * 1_000.0;
                }
            }
            let tb = rng(&mut s);
            let wb = weights_f64(q(tb));
            let first: [f64; 16] = std::array::from_fn(|k| grid[k / 4][k % 4]);
            let second: [f64; 16] = std::array::from_fn(|k| grid[k / 4][k % 4 + 1]);
            // ta = 1 is the recipe's weights (0, 0, 1, 0) at the limit: the second node exactly.
            let at_one = [0.0, 0.0, 1.0, 0.0];
            let at_zero = weights_f64(Gi::ZERO);
            let a = held_read(&first, &at_one, &wb, true);
            let b = held_read(&second, &at_zero, &wb, true);
            assert!((a - b).abs() < 1.0e-9, "{a} vs {b}");
            // The control: one pass clamped to the four corners of the cell.
            let corner_clamp = |z: &[f64; 16], wa: &[f64; 4]| {
                let v = plain_read(z, wa, &wb);
                let c = [z[5], z[6], z[9], z[10]];
                let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                v.clamp(lo, hi)
            };
            let ca = corner_clamp(&first, &at_one);
            let cb = corner_clamp(&second, &at_zero);
            control_differs |= (ca - cb).abs() > 1.0e-6;
        }
        assert!(control_differs, "the corner clamp never disagreed");
    }

    /// ★ TEST 2b: THE SYMMETRIC READ DOES NOT CHANGE UNDER A TRANSPOSE (an axis-swapping face
    /// seam). CONTROL: the rows-first read alone changes on some stencil.
    #[test]
    fn the_symmetric_read_survives_a_transpose() {
        let mut s = 0xDEAD_BEEF_0000_0001u64;
        let mut control_differs = false;
        for _ in 0..20_000 {
            let mut z = [0.0f64; 16];
            for v in &mut z {
                *v = (rng(&mut s) - 0.5) * 1_000.0;
            }
            let t: [f64; 16] = std::array::from_fn(|k| z[(k % 4) * 4 + k / 4]);
            let (ta, tb) = (rng(&mut s), rng(&mut s));
            let (wa, wb) = (weights_f64(q(ta)), weights_f64(q(tb)));
            let a = held_read(&z, &wa, &wb, true);
            let b = held_read(&t, &wb, &wa, true);
            assert!((a - b).abs() < 1.0e-9, "{a} vs {b}");
            let a1 = held_read(&z, &wa, &wb, false);
            let b1 = held_read(&t, &wb, &wa, false);
            control_differs |= (a1 - b1).abs() > 1.0e-6;
        }
        assert!(control_differs, "the one-order read never changed");
    }

    /// ★ TEST 4: THE f64 SHORE EQUALS THE RECIPE'S `shore()` to within one step of the length
    /// format on random columns and every side word.
    #[test]
    fn the_float_shore_is_the_recipe_shore() {
        let mut s = 0x5151_2323_7777_0101u64;
        let radius = length_of(6_000_000.0);
        for _ in 0..50_000 {
            let base = (rng(&mut s) - 0.5) * 8_000.0;
            let water = (rng(&mut s) - 0.5) * 4_000.0;
            let h = base + (rng(&mut s) - 0.5) * 1_000.0;
            let side = [
                vd_recipe::height::SIDE_UNKNOWN,
                vd_recipe::height::SIDE_LAND,
                vd_recipe::height::SIDE_SEA,
                vd_recipe::height::SIDE_INLAND,
            ][(rng(&mut s) * 4.0) as usize % 4];
            let bq = radius + length_of(base);
            let wq = radius + length_of(water);
            let hq = radius + length_of(h);
            let exact = metres(vd_recipe::height::shore(bq, wq, hq, side) - radius);
            let float = shore_f64(
                metres(bq - radius),
                Some(metres(wq - radius)),
                metres(hq - radius),
                side,
            );
            assert!((exact - float).abs() < 1.0e-6, "{exact} vs {float}");
        }
    }

    /// ★ TEST 5: THE TELESCOPE BOUND `|G_r − G_{r+1}| ≤ c_r·g`: the field of rung `r` departs from
    /// rung `r + 1`'s by at most one clamp.
    #[test]
    fn the_o1_steps_are_bounded_by_one_coarser_cell() {
        let mut s = 0x7777_1111_2222_3333u64;
        let (r1, node) = (10u8, 8_192u32);
        for _ in 0..20_000 {
            let mut rho = [0.0f64; RUNGS_MAX];
            for v in rho.iter_mut().take(usize::from(r1)) {
                *v = (rng(&mut s) - 0.5) * 600.0;
            }
            let (s_b, g_b) = ((rng(&mut s) - 0.5) * 400.0, 1.0 + rng(&mut s) * 300.0);
            for r in 0..r1 {
                let (g_r, _) = o1_field(s_b, g_b, &rho, r, r1, 1.0, node);
                let (g_up, _) = o1_field(s_b, g_b, &rho, r + 1, r1, 1.0, node);
                let c = f64::from(1u32 << (r + 1)) / f64::from(node);
                assert!((g_r - g_up).abs() <= c * g_b + 1.0e-9);
            }
        }
    }

    /// The fold: a tie is wet, the most samples name the water, and a tie of counts takes the
    /// higher level; under half is dry.
    #[test]
    fn the_fold_takes_the_wet_majority() {
        let sea = 100i16;
        let lake = Water::Lake(300).key();
        assert_eq!(fold(&[1, 1, 0, 0], sea), (true, Some(Water::Sea)));
        assert_eq!(fold(&[1, 0, 0, 0], sea), (false, None));
        assert_eq!(fold(&[1, lake, 0, 0], sea), (true, Some(Water::Lake(300))));
        assert_eq!(fold(&[1, 1, lake, 0], sea), (true, Some(Water::Sea)));
    }

    /// A water's key names it again.
    #[test]
    fn a_water_key_round_trips() {
        for w in [
            Water::Sea,
            Water::Lake(-32_000),
            Water::Lake(0),
            Water::Lake(5_983),
        ] {
            assert_eq!(Water::of_key(w.key()), Some(w));
        }
        assert_eq!(Water::of_key(0), None);
    }

    /// Every named choice parses back, and an unknown one is refused.
    #[test]
    fn a_named_choice_is_set_by_name() {
        let mut c = NamedChoices::default();
        c.set("k=0.5").expect("k");
        c.set("kappa_o1=0.5").expect("kappa");
        assert_eq!((c.k, c.kappa_o1), (0.5, 0.5));
        assert!(c.set("nothing=1").is_err());
        assert!(!c.describe().is_empty());
    }
}
