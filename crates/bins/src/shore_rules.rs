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
//! * **N2** (the second round, ruling B10 item 1) — N1's side and N1's node-scale blend, but the
//!   ground's fine half is KEPT near the water: each fine octave is scaled down only inside a RAMP
//!   from the line whose reach comes from that octave's own ground (its RMS height over its RMS
//!   slope), so the long octaves that lift the land fade over kilometres and the walking-scale
//!   ones stay to the water's edge (`n2_form=1`: one ramp for the whole fine half). On request its node words are
//!   held in a band round the water first (`word_band_m`), for the steep coasts' notches. Its
//!   first form (`n2_form=0`: the conditioned residual bent into the room it has) is kept as a
//!   refuted control: the 8 km nodes alias the 3–25 km octaves, so that residual is not texture.
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
use vd_terrain::body::TAN_REPOSE;
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
    /// ★ N2's BAND on the node words, metres: each mirrored word is held within this far of the
    /// water's level before the held read (∞ = N1's own words; the steep coasts' notches). 30 m by
    /// default — a stated choice of the second round, the owner's word owed.
    pub word_band_m: f64,
    /// ★ N2's words as a SIGNED DISTANCE (1): each stencil node's word is the level plus the band
    /// times its distance, in nodes, to the nearest block node of the other side, less a half (a
    /// wet node under, a dry one over) — so a diagonal staircase of wet nodes reads a straight
    /// line. Needs a finite band (metres per node of distance).
    pub word_sdf: bool,
    /// ★ N2's side field as a SMOOTHED MASK (1): `S = band × Σ w·side / Σ w` over the block's
    /// nodes, `side` +1 dry and −1 wet for the water, `w = (1 − r²/R²)³` at the distance `r` in
    /// nodes from the column, `R` = 2 nodes (so a node entering or leaving the block weighs
    /// nought: no seam on a node line). Needs a finite band.
    pub word_kernel: bool,
    /// N2's KNEE: the share of the texture's room it keeps whole before it bends toward the bound.
    pub n2_theta: f64,
    /// N2's texture on the side it cannot cross: bounded by the bank from the line (1), or by the
    /// same room as the side it can cross (0).
    pub n2_free: bool,
    /// N2's bank slope: 2 = the fine half's own measured RMS slope at the rung times the column's
    /// factor (the ground's own slope); 1 = the angle of repose times the factor; 0 = the angle of
    /// repose alone.
    pub n2_slope: u8,
    /// N2's form: 3 = one fade PER OCTAVE in HEIGHT: each octave fades where the column stands
    /// under `n2_cross_sigma` of that octave's RMS heights over the water (where it could carry
    /// the column across); 2 = one ramp per octave over its own reach in metres (its RMS height
    /// over its RMS slope); 1 = one ramp for the whole fine half (its reach from `n2_slope`);
    /// 0 = the first form's bent residual (a refuted control).
    pub n2_form: u8,
    /// N2's form 3: how many of an octave's RMS heights (times the column's factor) the column
    /// must stand over the water before the octave stands whole.
    pub n2_cross_sigma: f64,
    /// N2: a side-dry column with a standing water stands over it by W6's share of its field (1).
    pub n2_side_holds: bool,
    /// N2's reach per octave by the LAW (1): `C_kind × wavelength`, `C_kind` measured once over
    /// unit octaves of the recipe's own noise; or the per-octave statistic over the table (0).
    pub n2_reach_law: bool,
    /// N2's pocket variants (the build design's S0 item 3, Q11): 0 = the record N2; 1 = a
    /// POCKET-word column is exempt from the sea's side hold; 2 = 1, and the standing water's floor
    /// only where the SEA governs; 3 = 1, and that floor relaxed by the sea's presence.
    pub n2_pocket: u8,
    /// N2's fold names its water by THE ONE RULE (1; the build design's S0 item 4, Q7): a lake when
    /// the lake samples are at least half of the wet ones, at the highest lake level among them;
    /// or by the most samples (0, the record).
    pub n2_fold_rule: bool,
    /// N2's DEFAULT RULE (1; the build design §1.2 item 9): a slot for a water with no wet node
    /// in the column's block reads every word as the band's edge over the water, every distance
    /// 3.5 nodes and no presence, on every host alike; or the rows' own heights (0).
    pub n2_default_rule: bool,
    /// N2's WET TIE (1; S0 item 5): a column whose side is wet stands at least one length unit
    /// under its water, so the sheet's own rule (`s < l`) draws it wet. From rung 14 up a column's
    /// centre stands at the middle of its node cell, where a straight coast's clamped words read
    /// `S = 0` exactly and the wet hold is nought; 0 leaves that column drawn dry at its level.
    pub n2_wet_strict: bool,
    /// N2's LONE HOLD (1; S0 item 5): a column with no water in its block, whose side is dry,
    /// stands over its standing water (D7) by `k × band`, as the default rule's slot would; 0
    /// leaves it to the land rule, which lets a far column's ground stand under the sea it reads.
    pub n2_lone_holds: bool,
    /// N2's FOLD KERNEL under the one naming rule (ruling B15 item 1, the far lake flicker): 0 the
    /// column's own footprint; 1 the TENT, the column and its eight neighbours at its own rung
    /// weighted 1-2-1 along each axis ([`tent_words`]); 2 the tent only where the column's own
    /// samples disagree (a column all wet or all dry keeps its own answer); 3 the column's own
    /// footprint, and a wet column with no wet edge neighbour at its rung is dry (no lone column).
    pub n2_fold_kernel: u8,
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
            word_band_m: 30.0,
            word_sdf: false,
            word_kernel: false,
            n2_theta: 0.5,
            n2_free: true,
            n2_slope: 2,
            n2_form: 2,
            n2_cross_sigma: 3.0,
            n2_side_holds: true,
            n2_reach_law: true,
            n2_pocket: 2,
            n2_fold_rule: false,
            n2_default_rule: true,
            n2_wet_strict: true,
            n2_lone_holds: true,
            n2_fold_kernel: 0,
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
            "word_band_m" => self.word_band_m = f()?.max(1.0),
            "word_sdf" => self.word_sdf = f()? != 0.0,
            "word_kernel" => self.word_kernel = f()? != 0.0,
            "n2_theta" => self.n2_theta = f()?.clamp(0.0, 1.0),
            "n2_free" => self.n2_free = f()? != 0.0,
            "n2_slope" => self.n2_slope = (f()? as u8).min(2),
            "n2_form" => self.n2_form = (f()? as u8).min(3),
            "n2_cross_sigma" => self.n2_cross_sigma = f()?.max(0.0),
            "n2_side_holds" => self.n2_side_holds = f()? != 0.0,
            "n2_reach_law" => self.n2_reach_law = f()? != 0.0,
            "n2_pocket" => self.n2_pocket = (f()? as u8).min(3),
            "n2_fold_rule" => self.n2_fold_rule = f()? != 0.0,
            "n2_default_rule" => self.n2_default_rule = f()? != 0.0,
            "n2_wet_strict" => self.n2_wet_strict = f()? != 0.0,
            "n2_lone_holds" => self.n2_lone_holds = f()? != 0.0,
            "n2_fold_kernel" => self.n2_fold_kernel = (f()? as u8).min(3),
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
            format!(
                "N2's side hold = {} (1: a side-dry column that carries a standing water stands over it by k x |its governing field|; the side decides, the sheet never covers it)",
                u8::from(self.n2_side_holds)
            ),
            format!(
                "N2's fold names its water by = {} (1 the one rule: a lake when the lake samples are at least half of the wet ones, at the highest lake level among them — W15.1's lake share, the rule a count pyramid carries to the top rung; 0 the record: the water with the most samples, a tie the higher level; the build design's S0 item 4, Q7)",
                if self.n2_fold_rule { "the one rule" } else { "the most samples" }
            ),
            format!(
                "N2's default rule = {} (1: a slot for a water with no wet node in the column's block reads every word W + band, every distance 3.5 nodes, no presence — the build design §1.2 item 9, so a host with the shore band and a host with the rows read one word; it fires only where the fold names such a water, from rung 14 up; 0: the rows' own heights)",
                u8::from(self.n2_default_rule)
            ),
            format!(
                "N2's wet tie = {} (1: a side-wet column stands at least one length unit under its water, so the sheet draws it wet; 0: a column whose side field reads exactly nought stands AT its level and the sheet draws it dry — from rung 14 up a column's centre is its node cell's middle, where a straight coast's clamped words read nought; S0 item 5)",
                u8::from(self.n2_wet_strict)
            ),
            format!(
                "N2's lone hold = {} (1: a column with no water in its block whose side is dry stands over its standing water by k x band, the default rule's slot; 0: the land rule alone, under which a far column's ground can stand under the sea it reads; S0 item 5)",
                u8::from(self.n2_lone_holds)
            ),
            format!(
                "N2's fold kernel = {} (under the one naming rule: 0 the column's own footprint; 1 the tent, the column and its eight neighbours at its rung weighted 1-2-1 per axis; 2 the tent only where the column's own samples disagree; 3 the own footprint with no lone wet column; ruling B15 item 1, the far lake flicker)",
                self.n2_fold_kernel
            ),
            format!(
                "N2's pocket variant = {} (0 the record; 1 a POCKET-word column exempt from the sea's side hold; 2 = 1 and the standing water's floor only where the sea governs; 3 = 1 and that floor relaxed by the sea's presence; the build design's S0 item 3, Q11)",
                self.n2_pocket
            ),
            format!(
                "N2's reach per octave = {} (1: the law C_kind x wavelength, C_kind measured over unit octaves of the recipe's own noise; 0: 15/8 x RMS height / RMS slope measured over the table's own octaves)",
                if self.n2_reach_law { "the law" } else { "the statistic" }
            ),
            format!(
                "N2's word band = {} m (each mirrored word held within it of the level; inf = N1's words; ruling B10 item 1, the steep coasts' notches; 30 m a stated choice of the second round, the owner's word owed)",
                self.word_band_m
            ),
            format!(
                "N2's words as a signed distance = {} (1: word = level + band x (the distance in nodes to the nearest block node of the other side - 1/2), signed by the side; 0: the band's clamp)",
                u8::from(self.word_sdf)
            ),
            format!(
                "N2's side as a smoothed mask = {} (1: S = band x sum w.side / sum w over the block, side +1 dry -1 wet, w = (1 - r^2/4)^3 within 2 nodes of the column; the slope by central differences of 1/64 node)",
                u8::from(self.word_kernel)
            ),
            format!(
                "N2's form = {} (3 a fade per fine octave in HEIGHT: octave k scaled by the recipe's quintic of |S| over {} x the column's factor x its RMS height at share one; 2 a ramp per fine octave: each octave's term scaled by the recipe's quintic of the first-order distance over its own reach 15/8 x its RMS height / its RMS slope, both measured at share one over the sphere; 1 one ramp for the whole fine half over 15/8 x lift / slope, lift = the column's factor x the fine half's mean at share one at the rung, slope = the bank below; the wet side N1's own in both; 0 the bent residual below, a refuted control)",
                self.n2_form, self.n2_cross_sigma
            ),
            format!(
                "N2's first form: the conditioned residual rho at the rung, kept whole to {} of its room and bent (tanh) toward the room past it; the room on the side it can cross = the node-scale surface's height over W6's hold, on the other side = {} (the solve's TAN_REPOSE {}{} times the first-order distance to the line); faded in by the presence",
                self.n2_theta,
                if self.n2_free {
                    "the bank"
                } else {
                    "the same room"
                },
                TAN_REPOSE,
                match self.n2_slope {
                    0 => "",
                    1 => " times the column's roughness factor,",
                    _ => " (here: the fine half's measured RMS slope at the rung times the factor),",
                }
            ),
        ]
    }

    /// The same choices with N2's word band off: the words N1, O1 and O1q read.
    #[must_use]
    pub fn unbanded(&self) -> NamedChoices {
        NamedChoices {
            word_band_m: f64::INFINITY,
            word_sdf: false,
            word_kernel: false,
            ..*self
        }
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
// THE SHORE BAND (ruling B14 item 1; the build design §2.3): the heights a far host needs.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ WHETHER A NODE'S BANDED WORD NEEDS ITS HEIGHT for a water at level `w` (whole metres). Where
/// this answers no, the mirrored word held within `band` of the level is the band's edge on the
/// node's side: `w − band` for a node wet for the water, `w + band` for any other node.
///
/// **Example.** A land node of the gentle coast 12 m over the sea reads `sea + 12`, so it needs
/// its height. A sea node 1 m under the sea reads `sea − 1` (not the edge `sea − 30`), so it needs
/// its height too. A land node 80 m over the sea reads `sea + 30` without its height.
#[must_use]
pub fn band_needs(z: i32, wet: bool, w: i32, band: i32) -> bool {
    if wet {
        z > w - band
    } else {
        z > w - band && z < w + band
    }
}

/// ★ THE SHORE BAND (ruling B14 item 1): the fine nodes whose height a host needs when it holds
/// only the mask, the lake levels and the band. Node `n` joins where it stands in the sixteen-node
/// stencil of a node cell whose 6 × 6 block holds a node wet for a water `b`, and [`band_needs`]
/// says yes for `b`. The walk reads every regular node cell a column can name (`−1 ..= edge − 1`
/// on each axis of each face, each block through the seam table). The corner phantoms' cells,
/// which reach further from rung 13 up, are the host's exhaustive check (the build's S2).
#[must_use]
pub fn shore_band(nodes: &Nodes, fine: &MacroLattice, band_m: f64) -> Vec<bool> {
    shore_band_by(nodes, fine, band_m, band_needs)
}

/// [`shore_band`] under another membership rule (the bench's failing control reads S0's first
/// rule through it).
#[must_use]
pub fn shore_band_by(
    nodes: &Nodes,
    fine: &MacroLattice,
    band_m: f64,
    needs: fn(i32, bool, i32, i32) -> bool,
) -> Vec<bool> {
    let band = band_m as i32;
    let edge = i64::from(fine.edge);
    let mut in_band = vec![false; nodes.z_m.len()];
    for face in Face::ALL {
        for nj in -1..edge {
            for ni in -1..edge {
                let block = NodeSite {
                    face,
                    ni,
                    nj,
                    ta_q: Gi::ZERO,
                    tb_q: Gi::ZERO,
                    ta: 0.0,
                    tb: 0.0,
                }
                .block(fine);
                let mut waters: [Option<Water>; BLOCK] = [None; BLOCK];
                let mut count = 0usize;
                for &n in &block {
                    if let Some(b) = nodes.water_of(n)
                        && !waters[..count].contains(&Some(b))
                    {
                        waters[count] = Some(b);
                        count += 1;
                    }
                }
                for b in waters[..count].iter().flatten() {
                    let w = b.level_m(nodes.sea_m).round() as i32;
                    for oj in -1..=2i64 {
                        for oi in -1..=2i64 {
                            let n = block[block_index(oi, oj)];
                            let z = i32::from(nodes.z_m[n as usize]);
                            if needs(z, nodes.wet_for(n, *b), w, band) {
                                in_band[n as usize] = true;
                            }
                        }
                    }
                }
            }
        }
    }
    in_band
}

/// ★ WHERE A COAST CROSSES THE CUBE'S EDGES (the build design's S0 item 6): the node pairs that
/// straddle a cube edge with another water on each side (each pair once), per face side, and the
/// busiest stretch of 32 nodes along one side as `(pairs, face, side, node along the side)`.
pub struct EdgeCoast {
    pub pairs: usize,
    pub per_edge: Vec<(Face, vd_seed::seam::Edge, usize)>,
    pub best: Option<(usize, Face, vd_seed::seam::Edge, i64)>,
}

/// ★ THE EDGE COAST of one body's nodes (see [`EdgeCoast`]).
///
/// **Example.** Face −Y's `v−` side: node `(t, 0)` is sea and its partner across the cube edge,
/// `node_at(−Y, t, −1)`, is land, so the coast runs between them: one pair.
#[must_use]
pub fn edge_coast(nodes: &Nodes, fine: &MacroLattice) -> EdgeCoast {
    use vd_seed::seam::Edge;
    let edge = i64::from(fine.edge);
    let mut seen: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
    let mut best: Option<(usize, Face, Edge, i64)> = None;
    let mut per_edge = Vec::new();
    for face in Face::ALL {
        for side in Edge::ALL {
            let pair = |t: i64| -> (u32, u32) {
                let (inside, across) = match side {
                    Edge::UMinus => ((0, t), (-1, t)),
                    Edge::UPlus => ((edge - 1, t), (edge, t)),
                    Edge::VMinus => ((t, 0), (t, -1)),
                    Edge::VPlus => ((t, edge - 1), (t, edge)),
                };
                (
                    node_at(fine, face, inside.0, inside.1),
                    node_at(fine, face, across.0, across.1),
                )
            };
            let coast: Vec<bool> = (0..edge)
                .map(|t| {
                    let (a, b) = pair(t);
                    nodes.water_of(a) != nodes.water_of(b)
                })
                .collect();
            let mut here = 0usize;
            for (t, &x) in coast.iter().enumerate() {
                if x {
                    let (a, b) = pair(t as i64);
                    here += usize::from(seen.insert((a.min(b), a.max(b))));
                }
            }
            let span = 32usize.min(coast.len());
            let mut run = coast.iter().take(span).filter(|&&x| x).count();
            for t in span..coast.len() {
                if run > best.map_or(0, |b| b.0) {
                    best = Some((run, face, side, (t - span / 2) as i64));
                }
                run += usize::from(coast[t]);
                run -= usize::from(coast[t - span]);
            }
            if here > 0 {
                per_edge.push((face, side, here));
            }
        }
    }
    EdgeCoast {
        pairs: seen.len(),
        per_edge,
        best,
    }
}

/// ★ THE NODES A BAND-HOLDING HOST READS: the rows' arrays with every height outside the band
/// replaced by the deepest word, `i16::MIN`. The mirror and the band carry that word to the band's
/// edge on the node's side for every water, which is the word a host that holds only the mask, the
/// lake levels and the band reads there. So N2 read over these arrays draws what such a host
/// draws, and CHECK-SHORE-BAND compares it with N2 read over the rows.
#[must_use]
pub fn band_held(nodes: &Nodes, in_band: &[bool]) -> Nodes {
    Nodes {
        z_m: nodes
            .z_m
            .iter()
            .zip(in_band)
            .map(|(&z, &held)| if held { z } else { i16::MIN })
            .collect(),
        word: nodes.word.clone(),
        level_m: nodes.level_m.clone(),
        sea_m: nodes.sea_m,
    }
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
    /// The node-sampled distance to the water's line, in nodes: each stencil node's distance to
    /// the nearest block node of the other side, less a half, read at the column by the plain
    /// Catmull-Rom (N2's ramps; it cannot blow up where the side field goes flat).
    pub dist_nodes: f64,
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
    // THE DEFAULT RULE (the build design §1.2 item 9): N2's slot for a water with no wet node in
    // the block reads the band's edge over the water for every word, 3.5 nodes for every distance
    // and no presence, so a host that holds the shore band and a host that holds the rows read it
    // alike. Its read: S = +band, g = g_min, Ω = 0.
    if ch.n2_default_rule
        && ch.word_band_m.is_finite()
        && !block.iter().any(|&n| nodes.wet_for(n, b))
    {
        return Cand {
            water: b,
            w,
            s: ch.word_band_m,
            g: ch.g_min,
            omega: 0.0,
            wet_corner: false,
            near: f64::INFINITY,
            near_node: u32::MAX,
            dist_nodes: 3.5,
        };
    }
    let (s_k, g_k) = if ch.word_kernel && ch.word_band_m.is_finite() {
        let side: [f64; BLOCK] = std::array::from_fn(|k| {
            if nodes.wet_for(block[k], b) {
                -1.0
            } else {
                1.0
            }
        });
        let at = |x: f64, y: f64| -> f64 {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for (k, sd) in side.iter().enumerate() {
                let (ki, kj) = ((k % 6) as f64 - 2.0, (k / 6) as f64 - 2.0);
                let r2 = (ki - x).powi(2) + (kj - y).powi(2);
                if r2 < 4.0 {
                    let q = 1.0 - r2 / 4.0;
                    let wk = q * q * q;
                    num += wk * sd;
                    den += wk;
                }
            }
            if den > 0.0 {
                ch.word_band_m * num / den
            } else {
                ch.word_band_m
            }
        };
        let e = 1.0 / 64.0;
        let gx = (at(ta + e, tb) - at(ta - e, tb)) / (2.0 * e);
        let gy = (at(ta, tb + e) - at(ta, tb - e)) / (2.0 * e);
        (Some(at(ta, tb)), gx.hypot(gy))
    } else {
        (None, 0.0)
    };
    // The mirrored words of the sixteen-node stencil (offsets −1..=2).
    let mut zhat = [0.0f64; 16];
    if ch.word_sdf && ch.word_band_m.is_finite() {
        // N2's signed distance over the block: each stencil node's distance, in nodes, to the
        // nearest block node of the other side, less a half.
        let wet: [bool; BLOCK] = std::array::from_fn(|k| nodes.wet_for(block[k], b));
        for (s, slot) in zhat.iter_mut().enumerate() {
            let (oi, oj) = ((s % 4) as i64 - 1, (s / 4) as i64 - 1);
            let mine = wet[block_index(oi, oj)];
            let mut best = f64::INFINITY;
            for (k, &other) in wet.iter().enumerate() {
                if other != mine {
                    let (ki, kj) = ((k % 6) as i64 - 2, (k / 6) as i64 - 2);
                    let dist = ((ki - oi) as f64).hypot((kj - oj) as f64);
                    best = best.min(dist);
                }
            }
            let sdf = if best.is_finite() { best - 0.5 } else { 3.5 };
            *slot = if mine {
                w - ch.word_band_m * sdf
            } else {
                w + ch.word_band_m * sdf
            };
        }
    }
    for (s, slot) in zhat.iter_mut().enumerate() {
        if ch.word_sdf && ch.word_band_m.is_finite() {
            break;
        }
        let (oi, oj) = ((s % 4) as i64 - 1, (s / 4) as i64 - 1);
        let n = block[block_index(oi, oj)];
        let z = f64::from(nodes.z_m[n as usize]);
        let corner = (0..=1).contains(&oi) && (0..=1).contains(&oj);
        let word = if ch.mirror_all || corner {
            mirrored(z, nodes.wet_for(n, b), w)
        } else {
            z
        };
        *slot = word.clamp(w - ch.word_band_m, w + ch.word_band_m);
    }
    let (s, g) = match s_k {
        Some(s) => (s, g_k.max(ch.g_min)),
        None => {
            let s = held_read(&zhat, wa, wb, ch.held_both_orders) - w;
            let (ga, gb) = plain_gradient(&zhat, wa, wb, dwa, dwb);
            (s, ga.hypot(gb).max(ch.g_min))
        }
    };
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
    // The node-sampled distance: each stencil node's distance to the nearest block node of the
    // other side (the block reaches three nodes past the stencil at most), less a half, signed by
    // the side, then the plain read at the column.
    let wet: [bool; BLOCK] = std::array::from_fn(|k| nodes.wet_for(block[k], b));
    let mut sdf = [0.0f64; 16];
    for (st, slot) in sdf.iter_mut().enumerate() {
        let (oi, oj) = ((st % 4) as i64 - 1, (st / 4) as i64 - 1);
        let mine = wet[block_index(oi, oj)];
        let mut best = f64::INFINITY;
        for (k, &other) in wet.iter().enumerate() {
            if other != mine {
                let (ki, kj) = ((k % 6) as i64 - 2, (k / 6) as i64 - 2);
                best = best.min(((ki - oi) as f64).hypot((kj - oj) as f64));
            }
        }
        let d = if best.is_finite() { best - 0.5 } else { 3.5 };
        *slot = if mine { -d } else { d };
    }
    let dist_nodes = plain_read(&sdf, wa, wb).abs();
    Cand {
        water: b,
        w,
        s,
        g,
        omega,
        wet_corner,
        near: f64::INFINITY,
        near_node: u32::MAX,
        dist_nodes,
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
    /// `ρ_r`: the conditioned residual at the rung — the fine half less its node-scale read
    /// (O1's own [`residuals`]), metres; N2's texture. Nought where no candidate asked for it.
    pub rho: f64,
    /// The fine node's size, metres (N2's first-order distance in metres).
    pub node_m: f64,
    /// The column's roughness factor `m` (the word the kernel multiplies its fine half by) as a
    /// float: the fine half's RMS slope is `m × TAN_REPOSE` (N2's bank).
    pub rough: f64,
    /// The fine half's MEAN at this column, metres: `m` times the fine octaves' mean at share one
    /// at the rung (the ridged octaves stand on a positive mean — the lift). N2's ramp reads it.
    pub lift: f64,
    /// The fine octaves' own RMS slope at the rung at share one, along one axis at the rung's own
    /// column (measured over the sphere); times `m` it is the column's texture slope.
    pub slope_mu: f64,
    /// Each fine octave's own part of the fine half at this column, metres (its term times `m`;
    /// nought for an octave the rung does not sum). Their sum is the fine half `f`.
    pub octave_f: [f64; OCTAVES_CAP],
    /// Each fine octave's REACH, metres: `15/8` of its RMS height over its RMS slope, both
    /// measured at share one over the sphere (nought for a coarse octave).
    pub octave_reach: [f64; OCTAVES_CAP],
    /// Each fine octave's RMS height at share one, metres, measured over the sphere (nought for a
    /// coarse octave): times the column's factor, the height it can carry a column by.
    pub octave_rms: [f64; OCTAVES_CAP],
    /// The column's own coast word (its nearest fine node's): N2's pocket variants read it.
    pub word: u8,
}

/// Which surface family a rule belongs to: N1 fades the fine half by `β`; O1 keeps it whole; N2
/// blends the node-scale ground like N1 and keeps the texture, bent ([`n2_surface`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    N1,
    O1,
    N2,
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
    let (d, beta) = distance_and_beta(c, g_used);
    let h1 = if family == Family::N2 && ch.n2_form == 3 && !wet {
        n2_octave_heights(ground, c, beta, g_used, ch.n2_cross_sigma)
    } else if family == Family::N2 && ch.n2_form == 2 && !wet {
        n2_octave_ramps(ground, c, beta, g_used, cands, fields)
    } else if family == Family::N2 && ch.n2_form == 1 && !wet {
        n2_ramp(ground, c, d, beta, g_used, ch)
    } else if family == Family::N2 && ch.n2_form == 0 {
        n2_surface(ground, c, wet, d, beta, g_used, ch)
    } else {
        // N1 and N2's wet side fade the fine half by the blend; O1 keeps it whole.
        let kept = if family == Family::O1 {
            metres(ground.fine_q)
        } else {
            (1.0 - beta) * metres(ground.fine_q)
        };
        let h_far =
            metres(terrace(terrace_rule, ground.base_q + length_of(kept)) - ground.radius_q);
        (1.0 - beta) * h_far + beta * (c.w + g_used)
    };
    if wet {
        let depth = (c.w - h1).max(ch.k * g_used.abs());
        let depth = if family == Family::N2 && ch.n2_wet_strict {
            depth.max(1.0 / RAW_PER_M)
        } else {
            depth
        };
        return Finished {
            s: c.w - depth,
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
    // ★ N2: THE SIDE DECIDES OVER THE STANDING WATER. A column the side calls dry stands over the
    // sea by W6's share of the sea's own field, relaxed by the relief bound as the sea's presence
    // thins out (the lake hold's own rule, finding 10) — so the hold has no edge on a node line;
    // and a column that carries a standing water (D7: the sea in its read stencil) stands over it
    // in full, so the sheet never covers it, whatever the node-scale ground under it (a banded
    // line, or a far rung's coarser ground, may stand seaward of that ground's crossing).
    if family == Family::N2 && ch.n2_side_holds {
        // The pocket variants (Q11): a POCKET-word column keeps the solve's basin, never the sea's
        // side hold (variants 1–3).
        let pocket_exempt = ch.n2_pocket >= 1 && ground.word == WORD_POCKET;
        let mut omega_sea = 0.0f64;
        for (k, b) in cands.as_slice().iter().enumerate() {
            if b.water == Water::Sea {
                omega_sea = omega_sea.max(b.omega);
                if !pocket_exempt {
                    lb = lb.max(b.w + ch.k * fields[k].abs() - (1.0 - b.omega) * ground.a_r);
                }
            }
        }
        if let Some(l) = ground.d7 {
            match ch.n2_pocket {
                // The floor only where the sea governs.
                2 => {
                    if c.water == Water::Sea {
                        lb = lb.max(l + ch.k * g_used.abs());
                    }
                }
                // The floor relaxed by the sea's presence, as the sea slot's own hold is.
                3 => lb = lb.max(l + ch.k * g_used.abs() - (1.0 - omega_sea) * ground.a_r),
                _ => lb = lb.max(l + ch.k * g_used.abs()),
            }
        }
    }
    Finished {
        s: h1.max(lb),
        l: ground.d7,
        beta,
        ramp: ramp_lb > h1 && ramp_lb >= lb,
    }
}

/// ★ N2's BANK SLOPE (`n2_slope`): the fine half's own measured RMS slope at the rung times the
/// column's factor (2, the ground's own slope), the angle of repose times the factor (1), or the
/// angle of repose alone (0).
#[must_use]
pub fn n2_bank_slope(ground: &Ground, ch: &NamedChoices) -> f64 {
    match ch.n2_slope {
        0 => TAN_REPOSE,
        1 => TAN_REPOSE * ground.rough,
        _ => ground.slope_mu * ground.rough,
    }
}

/// The recipe's quintic fade has its steepest slope, 15/8, at the middle of its span.
pub const FADE_PEAK_SLOPE: f64 = 1.875;

/// ★ N2's FADE PER OCTAVE IN HEIGHT — ITS SURFACE ON THE DRY SIDE BEFORE THE HOLDS (ruling B10
/// item 1, the fourth form, the default). "Fade only what can cross", read literally: an octave can
/// carry a column across the water only where the column stands within a few of that octave's own
/// heights of the water. So octave `k` is scaled by the recipe's quintic of `|S| / (σ × m × h_k)`
/// — `|S|` the column's node-scale height over the water, `h_k` the octave's RMS height at share
/// one, `σ` = `n2_cross_sigma` — and stands whole wherever the column stands higher. The reach
/// comes from the ground in HEIGHT: on a steep coast `|S|` grows fast and every octave returns
/// within metres; on a flat plain it grows slowly and the long octaves (which carry the lift) fade
/// over kilometres, while the walking-scale ones, a metre high, stay to the water's edge.
///
/// **Example.** A pond on the plain with a factor of 0.1: the 12.5 km octave (252 m RMS at share
/// one) stands whole only where the column stands 76 m over the pond; the 98 m octave (0.2 m) is
/// whole 6 cm over it — so the ripples reach the water and the long swell has gone from the
/// pond's low shore.
#[must_use]
pub fn n2_octave_heights(ground: &Ground, c: &Cand, beta: f64, g_used: f64, sigma: f64) -> f64 {
    let f = ground.f;
    let node = (1.0 - beta) * (ground.h_raw - f) + beta * (c.w + g_used);
    let height = g_used.abs();
    let mut kept = 0.0f64;
    for (part, rms) in ground.octave_f.iter().zip(ground.octave_rms.iter()) {
        let bound = sigma * ground.rough * rms;
        let phi = if bound > 0.0 {
            fade_f64((height / bound).min(1.0))
        } else {
            1.0
        };
        kept += phi * part;
    }
    node + f - c.omega * (f - kept)
}

/// ★ N2's RAMP PER OCTAVE — ITS SURFACE ON THE DRY SIDE BEFORE THE HOLDS (ruling B10 item 1, the
/// third form). As [`n2_ramp`], but each fine octave climbs from nought at the line
/// over its OWN reach, `15/8 ×` its RMS height over its RMS slope: the long octaves, which carry
/// the lift and the big swells, fade over kilometres, and the short ones — the walking-scale
/// texture — stay whole to within metres of the water. So "fade only what can cross" is read per
/// octave, and each reach comes from the octave's own ground, never from the node square. A rung
/// that has dropped an octave draws it nowhere, so the octaves near the line agree at every rung
/// that keeps them.
///
/// **Example.** On the gentle coast the 25 km octave stands some 300 m high on the shore and
/// fades over about 20 km, at its own gentle slope; the 98 m octave stands a metre high and fades
/// over its last hundred metres, so a pilot landing on the beach sees ripples to the water's edge.
///
/// **THE PRICE, MEASURED** (the census's RAISE-OVER-RAW): fading a long octave toward nought also
/// fills its troughs — on the gentle coast a lowland 12 km from the sea rises 190 m over the raw
/// ground. Fading only the parts over nought was tried and refuted: there the node-scale ground
/// stands under the sea and only the fine half's lift holds the land up, so the land fell flat to
/// the hold. The long octaves are relief the 8 km solve never saw; R2 and R4 own the cure.
///
/// **EVERY NEARBY WATER FADES IT, BY A DISTANCE THAT CANNOT JUMP** (the round's later
/// measurements): each octave's factor is the PRODUCT over the column's candidate waters of
/// `1 − Ω_b·(1 − fade(d_b / reach))`, and `d_b` is the water's first-order distance `|S|/|∇S|`
/// never past its node-sampled distance ([`Cand::dist_nodes`]). The first-order distance alone
/// blows up where the side field goes flat — the edge of the band's plateau — so there the long
/// octaves came back within a few hundred metres and stood a 200 m ridge over the raw ground on
/// the gentle coast; the node-sampled distance grows smoothly there, and the first-order one keeps
/// the fine half nought on the drawn line.
#[must_use]
pub fn n2_octave_ramps(
    ground: &Ground,
    c: &Cand,
    beta: f64,
    g_used: f64,
    cands: &Cands,
    fields: &[f64; 8],
) -> f64 {
    let f = ground.f;
    let node = (1.0 - beta) * (ground.h_raw - f) + beta * (c.w + g_used);
    let mut kept = 0.0f64;
    for (part, reach) in ground.octave_f.iter().zip(ground.octave_reach.iter()) {
        let mut phi = 1.0f64;
        if *reach > 0.0 {
            for (k, b) in cands.as_slice().iter().enumerate() {
                // The first-order distance, never past the node-sampled one: nought on the line,
                // and no jump where the side field goes flat.
                let dist = (fields[k].abs() / b.g).min(b.dist_nodes) * ground.node_m;
                phi *= 1.0 - b.omega * (1.0 - fade_f64((dist / reach).min(1.0)));
            }
        }
        kept += phi * part;
    }
    node + kept
}

/// ★ N2's RAMP — ITS SURFACE ON THE DRY SIDE BEFORE THE HOLDS (ruling B10 item 1, the second
/// form). The ground WITHOUT its fine half, `h_raw − f`, is blended toward `W + S` by N1's own `β`
/// (so the surface meets the level exactly on the line); the fine half `f` — its mean, the LIFT,
/// and its texture together — is then added back scaled by `φ`, the recipe's quintic of the
/// first-order distance to the line over the REACH `15/8 × lift / slope`. So the fine half climbs
/// from nought at the line to whole at the reach, and the ramp's steepest slope is the bank's
/// `slope` ([`n2_bank_slope`]; by default the fine half's own measured RMS slope). The reach
/// comes from the ground: the fine half's mean over its own slope, whatever the roughness, because
/// both scale with `m`. The ramp is faded in by the presence `Ω`, so where the candidate thins out
/// the ground is the raw ground.
///
/// **Example.** A pond on the plain: the raw ground 2 km from it stands 35 m of lift over the
/// node-scale ground. N1 fades that away over a whole node (8 km) and draws a textureless dish;
/// N2 keeps the ground whole to about 2 km from the pond and ramps it down to the pond's level
/// in the last 2 km, the dunes on the ramp shrinking with it.
#[must_use]
pub fn n2_ramp(
    ground: &Ground,
    c: &Cand,
    d: f64,
    beta: f64,
    g_used: f64,
    ch: &NamedChoices,
) -> f64 {
    let f = ground.f;
    let node = (1.0 - beta) * (ground.h_raw - f) + beta * (c.w + g_used);
    let slope = n2_bank_slope(ground, ch);
    let phi = if ground.lift > 0.0 && slope > 0.0 {
        let reach = FADE_PEAK_SLOPE * ground.lift / slope;
        fade_f64((d * ground.node_m / reach).min(1.0))
    } else {
        1.0
    };
    node + (1.0 - c.omega * (1.0 - phi)) * f
}

/// ★ N2's FIRST FORM, A REFUTED CONTROL (`n2_form=0`). The ground WITHOUT its texture,
/// `h_raw − ρ`, is blended toward `W + S` by N1's own `β`; the texture `ρ` is then added back,
/// bent by [`soft_bound`] into the ROOM it has: on the side it can cross, the node-scale surface's
/// height over W6's hold (`W ± k·|S|`), so it never crosses; on the other side a BANK from the
/// line (`slope × d × node`), so it meets the line at nought and never stands as a wall. The
/// bank's slope is the texture's own RMS slope, `m × TAN_REPOSE` (the first run's bank of repose
/// alone stood walls on the belt, whose ground is far gentler than 35°). The bend is faded in by the presence `Ω`, so where the candidate thins out the texture is
/// the raw ground's own.
///
/// **Example.** A column on the plain stands a quarter node (2 km) from a pond at 1 200 m. N1
/// draws it as a smooth rise with no texture. Under N2 its 30 m knoll stays: the knoll's hollows
/// bend before they reach W6's hold over the pond, so they never reach the pond, and its crest may
/// stand as high as the bank allows at 2 km, so it stands whole.
#[must_use]
pub fn n2_surface(
    ground: &Ground,
    c: &Cand,
    wet: bool,
    d: f64,
    beta: f64,
    g_used: f64,
    ch: &NamedChoices,
) -> f64 {
    let rho = ground.rho;
    let node = (1.0 - beta) * (ground.h_raw - rho) + beta * (c.w + g_used);
    let slope = n2_bank_slope(ground, ch);
    let bank = slope * d * ground.node_m;
    let hold = ch.k * g_used.abs();
    let (down, up) = if wet {
        let room = (c.w - hold - node).max(0.0);
        (if ch.n2_free { bank } else { room }, room)
    } else {
        let room = (node - (c.w + hold)).max(0.0);
        (room, if ch.n2_free { bank } else { room })
    };
    let bent = soft_bound(rho, down, up, ch.n2_theta);
    node + (1.0 - c.omega) * rho + c.omega * bent
}

/// ★ THE SOFT BOUND: `x` kept whole up to `theta` of its room (`up` over nought, `down` under
/// it), then bent by a hyperbolic tangent toward the room, which it never passes (a float's
/// tangent reaches one for a very large `x`, so the room itself is reached there); a room of
/// nought gives nought. It is continuous and has a continuous slope, so a bent texture shows no
/// crease where the bend starts.
#[must_use]
pub fn soft_bound(x: f64, down: f64, up: f64, theta: f64) -> f64 {
    let bend = |x: f64, room: f64| -> f64 {
        if room <= 0.0 {
            return 0.0;
        }
        let knee = theta * room;
        if x <= knee {
            return x;
        }
        let span = room - knee;
        if span <= 0.0 {
            return room;
        }
        knee + span * ((x - knee) / span).tanh()
    };
    if x >= 0.0 {
        bend(x, up)
    } else {
        -bend(-x, down)
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

/// ★ THE ONE NAMING RULE (the build design's §1.4 item 2, S0 item 4, Q7): wet when the wet samples
/// are at least half (a tie wet); then a LAKE when the lake samples are at least half of the wet
/// ones (W15.1's lake share), at the HIGHEST lake level among the lake samples; else the sea. It
/// reads only a wet count, a lake count and a level maximum, so a count pyramid carries it from
/// rung 8 to the top rung with one rule — where the record's "most samples" needs a count per
/// water.
///
/// **Example.** A far cell holds 2 048 samples: 700 of the sea, 400 of a lake at 1 200 m and 300 of
/// a lake at 1 180 m. It is wet (1 400 of 2 048); its lake samples are 700 of the 1 400 wet ones, so
/// it names a lake, at 1 200 m.
#[must_use]
pub fn fold_one_rule(samples: &[u32], sea_m: i16) -> (bool, Option<Water>) {
    let (mut wet, mut lake) = (0u64, 0u64);
    let mut top: Option<i16> = None;
    for &k in samples {
        match Water::of_key(k) {
            None => {}
            Some(Water::Sea) => wet += 1,
            Some(Water::Lake(l)) => {
                wet += 1;
                lake += 1;
                top = Some(top.map_or(l, |t| t.max(l)));
            }
        }
    }
    let _ = sea_m;
    one_rule_counts(samples.len() as u64, wet, lake, top)
}

/// A column's fold words: its samples, its wet samples, its lake samples and the highest lake
/// level among them.
pub type FoldWords = (u64, u64, u64, Option<i16>);

/// ★ THE TENT FOLD'S WORDS (ruling B15 item 1, a candidate cure of the far lake flicker): the words
/// of a column and its eight neighbours at one rung, `words[b][a]` for the offset `(a − 1, b − 1)`,
/// weighted 1-2-1 along each axis and summed; a neighbour the host does not hold (`None`) drops out
/// with its weight. The level is the column's own highest lake level where its own footprint holds
/// a lake sample, else the neighbours' highest. [`one_rule_counts`] reads the result.
///
/// **Example.** A one-column piece of a lake's edge is 64 % wet; its lake-side neighbour and one
/// corner are 90 % wet, its other three neighbours and three corners 10 %. The tent reads
/// (4·64 + 2·90 + 6·10 + 1·90 + 3·10) / 16 = 38 % wet, so the piece is not drawn at its own rung
/// and cannot flicker at the next.
#[must_use]
pub fn tent_words(words: &[[Option<FoldWords>; 3]; 3]) -> FoldWords {
    const W: [u64; 3] = [1, 2, 1];
    let (mut n, mut wet, mut lake) = (0u64, 0u64, 0u64);
    let mut top: Option<i16> = None;
    for (b, row) in words.iter().enumerate() {
        for (a, cell) in row.iter().enumerate() {
            if let Some((cn, cw, cl, ct)) = cell {
                let w = W[a] * W[b];
                n += w * cn;
                wet += w * cw;
                lake += w * cl;
                if let Some(t) = ct {
                    top = Some(top.map_or(*t, |x: i16| x.max(*t)));
                }
            }
        }
    }
    let own = words[1][1].and_then(|w| w.3);
    (n, wet, lake, own.or(top))
}

/// ★ THE KERNEL'S WORDS (ruling B15 item 1): kernel 0 reads the column's own words, kernel 1 the
/// tent, kernel 2 the tent only where the column's own samples disagree — a column all wet (of one
/// water) or all dry keeps its own words.
#[must_use]
pub fn kernel_words(kernel: u8, words: &[[Option<FoldWords>; 3]; 3]) -> Option<FoldWords> {
    let own = words[1][1]?;
    let (n, wet, lake, _) = own;
    let unanimous = wet == 0 || (wet == n && (lake == 0 || lake == n));
    Some(match kernel {
        0 => own,
        2 if unanimous => own,
        _ => tent_words(words),
    })
}

/// ★ THE FOLD BY KERNEL (ruling B15 item 1): the one naming rule on the words [`kernel_words`]
/// reads (kernels 0–2), or kernel 3 — the column's own words, and a column that would be wet
/// with no wet edge neighbour at its rung (by the same rule on the neighbour's own words) is dry.
/// `None` where the host does not hold the column's own words.
///
/// **Example.** At rung 14 a 16 km piece of the 5 882 m lake's edge is 64 % wet, and its four
/// edge neighbours are 30 %, 20 %, 40 % and 10 % wet: under kernel 3 it is dry, so it cannot hide
/// at rung 15 and show again at rung 16.
#[must_use]
pub fn fold_by_kernel(
    kernel: u8,
    words: &[[Option<FoldWords>; 3]; 3],
) -> Option<(bool, Option<Water>)> {
    if kernel == 3 {
        let (n, wet, lake, top) = words[1][1]?;
        let own = one_rule_counts(n, wet, lake, top);
        let wet_at = |w: Option<FoldWords>| {
            w.is_some_and(|(n, wet, lake, top)| one_rule_counts(n, wet, lake, top).0)
        };
        let neighbour = wet_at(words[0][1])
            || wet_at(words[1][0])
            || wet_at(words[1][2])
            || wet_at(words[2][1]);
        return Some(if own.0 && !neighbour {
            (false, None)
        } else {
            own
        });
    }
    let (n, wet, lake, top) = kernel_words(kernel, words)?;
    Some(one_rule_counts(n, wet, lake, top))
}

/// The fold words of a list of sample keys.
#[must_use]
pub fn words_of(samples: &[u32]) -> FoldWords {
    let (mut wet, mut lake) = (0u64, 0u64);
    let mut top: Option<i16> = None;
    for &k in samples {
        match Water::of_key(k) {
            None => {}
            Some(Water::Sea) => wet += 1,
            Some(Water::Lake(l)) => {
                wet += 1;
                lake += 1;
                top = Some(top.map_or(l, |t| t.max(l)));
            }
        }
    }
    (samples.len() as u64, wet, lake, top)
}

/// ★ THE ONE NAMING RULE ON COUNTS: `n` samples, `wet` of them wet, `lake` of those a lake's, `top`
/// the highest lake level among the lake samples. A count pyramid sums the three words of the
/// quarters under a far column and reads this; [`fold_one_rule`] counts the samples and reads it.
///
/// **Example.** A rung-16 column over the gentle coast holds 1 048 576 samples: 600 000 wet, 30 000
/// of a tarn at 3 800 m. It is wet (600 000 of 1 048 576), and the tarn's share of the wet samples
/// is under half, so it names the sea.
#[must_use]
pub fn one_rule_counts(n: u64, wet: u64, lake: u64, top: Option<i16>) -> (bool, Option<Water>) {
    if wet == 0 || wet * 2 < n {
        return (false, None);
    }
    match top {
        Some(l) if lake * 2 >= wet => (true, Some(Water::Lake(l))),
        _ => (true, Some(Water::Sea)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ The one naming rule: a tie wet, the lake share, the highest level; the record's rule
    /// names the water with the most samples instead.
    ///
    /// **Example.** 700 sea, 400 of a lake at 1 200 m, 300 of one at 1 180 m, 648 dry: the one
    /// rule names the 1 200 m lake (700 lake samples of 1 400 wet); the record names the sea.
    #[test]
    fn the_one_rule_names_a_lake_by_its_share_and_its_highest_level() {
        let (sea, hi, lo) = (
            Water::Sea.key(),
            Water::Lake(1_200).key(),
            Water::Lake(1_180).key(),
        );
        let mut s: Vec<u32> = Vec::new();
        s.extend(std::iter::repeat_n(sea, 700));
        s.extend(std::iter::repeat_n(hi, 400));
        s.extend(std::iter::repeat_n(lo, 300));
        s.extend(std::iter::repeat_n(0, 648));
        assert_eq!(fold_one_rule(&s, 0), (true, Some(Water::Lake(1_200))));
        assert_eq!(fold(&s, 0), (true, Some(Water::Sea)));
        // A tie of wet and dry is wet; fewer lake samples than half the wet ones names the sea.
        let t = [sea, sea, 0, 0];
        assert_eq!(fold_one_rule(&t, 0), (true, Some(Water::Sea)));
        let u = [sea, sea, sea, hi, 0, 0];
        assert_eq!(fold_one_rule(&u, 0), (true, Some(Water::Sea)));
        let dry = [0, 0, sea];
        assert_eq!(fold_one_rule(&dry, 0), (false, None));
    }

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
        c.set("word_band_m=30").expect("band");
        assert_eq!(c.word_band_m, 30.0);
        assert_eq!(c.unbanded().word_band_m, f64::INFINITY);
        assert_eq!(c.unbanded().k, 0.5);
    }

    /// ★ The tent weighs a column and its eight neighbours 1-2-1 per axis, drops a neighbour the
    /// host does not hold, and keeps the column's own lake level; kernel 2 keeps a unanimous
    /// column's own words.
    ///
    /// **Example.** The doc's piece of a lake's edge: 64 % wet, its lake-side neighbour and one
    /// corner 90 %, the rest 10 % — the tent reads 616 of 1 600 (38.5 %), so the one rule says dry.
    #[test]
    fn the_tent_weighs_the_neighbours_and_keeps_a_unanimous_column() {
        let c = |pct: u64| Some((100u64, pct, 0u64, None));
        let words = [
            [c(10), c(10), c(10)],
            [c(90), c(64), c(10)],
            [c(90), c(10), c(10)],
        ];
        // Weights: the centre 4, the four sides 2, the four corners 1.
        let (n, wet, lake, top) = tent_words(&words);
        assert_eq!(
            (n, wet, lake, top),
            (1_600, 4 * 64 + 2 * 90 + 6 * 10 + 90 + 3 * 10, 0, None)
        );
        assert_eq!(one_rule_counts(n, wet, lake, top), (false, None));
        // A neighbour the host does not hold drops out with its weight.
        let mut partial = words;
        partial[0][0] = None;
        assert_eq!(tent_words(&partial).0, 1_500);
        // The level: the column's own where it holds a lake sample, else the neighbours'.
        let mut lakes = words;
        lakes[1][1] = Some((100, 80, 80, Some(1_200)));
        lakes[1][0] = Some((100, 90, 90, Some(1_300)));
        assert_eq!(tent_words(&lakes).3, Some(1_200));
        lakes[1][1] = Some((100, 0, 0, None));
        assert_eq!(tent_words(&lakes).3, Some(1_300));
        // Kernel 2: a unanimous column keeps its own words; a mixed one reads the tent.
        let mut all_wet = words;
        all_wet[1][1] = Some((100, 100, 0, None));
        assert_eq!(kernel_words(2, &all_wet), Some((100, 100, 0, None)));
        assert_eq!(kernel_words(2, &words), Some(tent_words(&words)));
        assert_eq!(kernel_words(0, &words), words[1][1]);
        assert_eq!(kernel_words(1, &all_wet), Some(tent_words(&all_wet)));
    }

    /// ★ The band holds exactly the words off its edge: every height from `W − 100` to `W + 100`,
    /// wet and not, against the mirror and the clamp themselves.
    ///
    /// **Example.** A sea node 1 m under the sea reads `sea − 1`, not the edge `sea − 30`, so it
    /// joins the band (S0's first rule left it out).
    #[test]
    fn the_band_holds_exactly_the_words_off_the_edge() {
        let (w, band) = (1_200i32, 30i32);
        for z in w - 100..=w + 100 {
            for wet in [false, true] {
                let word = mirrored(f64::from(z), wet, f64::from(w))
                    .clamp(f64::from(w - band), f64::from(w + band));
                let edge = f64::from(if wet { w - band } else { w + band });
                assert_eq!(
                    band_needs(z, wet, w, band),
                    word != edge,
                    "z {z}, wet {wet}"
                );
            }
        }
        assert!(band_needs(w - 1, true, w, band));
    }

    /// ★ The default rule: a water with no wet node in the column's block reads `S = +band`,
    /// `g = g_min`, no presence and 3.5 nodes, whatever the rows hold; with the rule off, the rows'
    /// heights decide.
    ///
    /// **Example.** A far column's fold names the sea, but its 6 × 6 block is all land 10 m over
    /// the sea: under the rule its sea slot reads `+30 m`, not the land's own `+10 m`.
    #[test]
    fn a_water_outside_the_block_reads_the_default_rule() {
        let sea_m = 100i16;
        let nodes = Nodes {
            z_m: vec![sea_m + 10; BLOCK],
            word: vec![WORD_LAND; BLOCK],
            level_m: vec![DRY_M; BLOCK],
            sea_m,
        };
        let block: [u32; BLOCK] = std::array::from_fn(|k| k as u32);
        let (wa, wb) = (weights_f64(Gi::ZERO), weights_f64(Gi::ZERO));
        let (dwa, dwb) = (dweights(0.25), dweights(0.25));
        let ch = NamedChoices::default();
        let c = candidate_for(
            &nodes,
            &block,
            Water::Sea,
            0.25,
            0.25,
            &wa,
            &wb,
            &dwa,
            &dwb,
            &ch,
        );
        assert_eq!(
            (c.s, c.g, c.omega, c.wet_corner, c.dist_nodes),
            (30.0, ch.g_min, 0.0, false, 3.5)
        );
        let off = NamedChoices {
            n2_default_rule: false,
            ..ch
        };
        let d = candidate_for(
            &nodes,
            &block,
            Water::Sea,
            0.25,
            0.25,
            &wa,
            &wb,
            &dwa,
            &dwb,
            &off,
        );
        assert!(
            (d.s - 10.0).abs() < 1e-9,
            "the rows' own +10 m, got {}",
            d.s
        );
    }

    /// ★ The soft bound keeps a texture whole inside its knee, never passes its room, has no
    /// crease at the knee, and gives nought for a room of nought.
    ///
    /// **Example.** A 10 m knoll 2 km from a pond has 40 m of room: it stands whole. A 60 m knoll
    /// there bends to under 40 m and never passes the pond's hold.
    #[test]
    fn the_soft_bound_bends_and_never_passes_its_room() {
        assert_eq!(soft_bound(10.0, 40.0, 40.0, 0.5), 10.0);
        assert_eq!(soft_bound(-10.0, 40.0, 40.0, 0.5), -10.0);
        let big = soft_bound(60.0, 40.0, 40.0, 0.5);
        assert!(big > 20.0 && big < 40.0, "{big}");
        assert!(soft_bound(-1.0e6, 40.0, 5.0, 0.5) >= -40.0);
        assert!(soft_bound(1.0e6, 40.0, 5.0, 0.5) <= 5.0);
        assert!(soft_bound(-45.0, 40.0, 5.0, 0.5) > -40.0);
        assert_eq!(soft_bound(3.0, 40.0, 0.0, 0.5), 0.0);
        assert_eq!(soft_bound(-3.0, 0.0, 40.0, 0.5), 0.0);
        // No crease at the knee: the slope on both sides of it is one.
        let (knee, e) = (20.0, 1.0e-4);
        let below = (soft_bound(knee, 40.0, 40.0, 0.5) - soft_bound(knee - e, 40.0, 40.0, 0.5)) / e;
        let above = (soft_bound(knee + e, 40.0, 40.0, 0.5) - soft_bound(knee, 40.0, 40.0, 0.5)) / e;
        assert!((below - 1.0).abs() < 1.0e-3 && (above - 1.0).abs() < 1.0e-3);
        // A knee at the whole room is a plain clip.
        assert_eq!(soft_bound(60.0, 40.0, 40.0, 1.0), 40.0);
    }

    fn n2_ground(h_raw: f64, rho: f64) -> Ground {
        Ground {
            b: h_raw - rho,
            f: rho,
            h_raw,
            base_q: Gi::ZERO,
            fine_q: Gi::ZERO,
            radius_q: Gi::ZERO,
            d7: None,
            sea: None,
            a_r: 1_000.0,
            rho,
            node_m: 8_192.0,
            rough: 1.0,
            lift: 300.0,
            slope_mu: TAN_REPOSE,
            octave_f: [0.0; OCTAVES_CAP],
            octave_reach: [0.0; OCTAVES_CAP],
            octave_rms: [0.0; OCTAVES_CAP],
            word: WORD_LAND,
        }
    }

    fn pond(omega: f64) -> Cand {
        Cand {
            water: Water::Lake(1_200),
            w: 1_200.0,
            s: 0.0,
            g: 60.0,
            omega,
            wet_corner: false,
            near: 0.0,
            near_node: 0,
            dist_nodes: f64::INFINITY,
        }
    }

    /// ★ N2's ramp meets the pond's level on the line, keeps the ground whole past its reach and
    /// where the pond is absent, and climbs no steeper than its bank in between.
    ///
    /// **Example.** A pond with 300 m of lift on its shore: the ground stands at the pond's level on
    /// the line, whole 804 m out (15/8 × 300 m over the bank of repose), and in between it never
    /// climbs steeper than the bank plus the node-scale slope.
    #[test]
    fn n2_ramp_meets_the_line_and_keeps_the_ground() {
        let ch = NamedChoices::default();
        let g = |f: f64| {
            let mut g = n2_ground(1_600.0 + f, 0.0);
            g.b = 1_600.0;
            g.f = f;
            g
        };
        // On the line: the level, whatever the fine half.
        for f in [-40.0, 0.0, 300.0, 700.0] {
            assert_eq!(n2_ramp(&g(f), &pond(1.0), 0.0, 1.0, 0.0, &ch), 1_200.0);
        }
        // Past the reach (15/8 x 300 / 0.7 = 804 m, under a node): the node part plus the whole
        // fine half.
        let far = n2_ramp(&g(300.0), &pond(1.0), 0.2, 0.0, 12.0, &ch);
        assert!((far - 1_900.0).abs() < 1.0e-9, "{far}");
        // No presence: the raw ground.
        let absent = n2_ramp(&g(300.0), &pond(0.0), 0.01, 0.0, 12.0, &ch);
        assert!((absent - 1_900.0).abs() < 1.0e-9, "{absent}");
        // In between the fine half climbs no steeper than the bank (per metre, with no node slope).
        let mut prev = 0.0;
        let step_m = 4.0;
        for k in 1..300 {
            let d = f64::from(k) * step_m / 8_192.0;
            let s = n2_ramp(&g(300.0), &pond(1.0), d, 0.0, 0.0, &ch) - 1_600.0;
            if k > 1 {
                assert!((s - prev) / step_m <= TAN_REPOSE * 1.001, "k {k}");
            }
            prev = s;
        }
    }

    /// ★ N2's ramp per octave keeps a short octave to near the line and fades a long one far out.
    ///
    /// **Example.** A 1 m ripple with a 50 m reach and a 300 m swell with a 20 km reach: 100 m
    /// from the pond the ripple stands whole and the swell has all but gone; on the line both
    /// are nought.
    #[test]
    fn n2_octave_ramps_keep_the_short_octaves() {
        let mut g = n2_ground(1_600.0 + 301.0, 0.0);
        g.b = 1_600.0;
        g.f = 301.0;
        g.octave_f[5] = 300.0;
        g.octave_reach[5] = 20_000.0;
        g.octave_f[11] = 1.0;
        g.octave_reach[11] = 50.0;
        // One candidate, the pond, its field 60 m a node: `dist_m` metres out it reads
        // 60 × dist_m / 8 192.
        let one = |omega: f64| {
            let mut cands = Cands::default();
            cands.list[0] = pond(omega);
            cands.len = 1;
            cands
        };
        let fields_at = |dist_m: f64| {
            let mut f = [0.0f64; 8];
            f[0] = 60.0 * dist_m / 8_192.0;
            f
        };
        let at = |dist_m: f64| {
            n2_octave_ramps(&g, &pond(1.0), 0.0, 0.0, &one(1.0), &fields_at(dist_m)) - 1_600.0
        };
        assert_eq!(
            n2_octave_ramps(&g, &pond(1.0), 1.0, 0.0, &one(1.0), &fields_at(0.0)),
            1_200.0
        );
        let near = at(100.0);
        assert!((near - 1.0).abs() < 0.01, "{near}");
        let far = at(25_000.0);
        assert!((far - 301.0).abs() < 1.0e-9, "{far}");
        // No presence: whole.
        let absent =
            n2_octave_ramps(&g, &pond(0.0), 0.0, 0.0, &one(0.0), &fields_at(100.0)) - 1_600.0;
        assert!((absent - 301.0).abs() < 1.0e-9, "{absent}");
        // Two ponds, each 1 km off, fade the swell more than one alone: the factor is the product.
        let mut two = one(1.0);
        two.list[1] = pond(1.0);
        two.len = 2;
        let mut f2 = fields_at(1_000.0);
        f2[1] = f2[0];
        let both = n2_octave_ramps(&g, &pond(1.0), 0.0, 0.0, &two, &f2) - 1_600.0;
        let alone = at(1_000.0);
        assert!(both < alone, "{both} {alone}");
    }

    /// ★ N2's fade per octave in height: on the line nought, a short octave whole a few of its
    /// heights over the water, a long one gone there, both whole high over the water.
    ///
    /// **Example.** A 1 m ripple (RMS 0.5 m) and a 300 m swell (RMS 150 m) on a pond's shore with
    /// the factor one: 10 m over the pond the ripple stands whole and the swell has all but gone;
    /// 500 m over it both stand whole.
    #[test]
    fn n2_octave_heights_fade_what_can_cross() {
        let mut g = n2_ground(1_600.0 + 301.0, 0.0);
        g.b = 1_600.0;
        g.f = 301.0;
        g.octave_f[5] = 300.0;
        g.octave_rms[5] = 150.0;
        g.octave_f[11] = 1.0;
        g.octave_rms[11] = 0.5;
        let at = |h: f64| n2_octave_heights(&g, &pond(1.0), 0.0, h, 3.0) - 1_600.0;
        assert_eq!(n2_octave_heights(&g, &pond(1.0), 1.0, 0.0, 3.0), 1_200.0);
        // The ripple whole, and the swell all but gone: 3 cm of its 300 m.
        let low = at(10.0);
        assert!((low - 1.0).abs() < 0.05, "{low}");
        let high = at(500.0);
        assert!((high - 301.0).abs() < 1.0e-9, "{high}");
    }

    /// ★ N2 keeps the ground whole where the pond is absent, meets the pond's level at the line
    /// from both sides, and its texture never crosses the water where the pond is present.
    ///
    /// **Example.** A column 2 km from a pond on the plain keeps its knolls; a column on the line
    /// stands at the pond's level whatever its knoll; a column a node inside the pond stays under
    /// the pond's hold whatever its knoll.
    #[test]
    fn n2_keeps_the_texture_and_never_crosses() {
        let ch = NamedChoices::default();
        // No presence and no blend: the raw ground, texture and all.
        let far = n2_surface(
            &n2_ground(1_530.0, -80.0),
            &pond(0.0),
            false,
            2.0,
            0.0,
            120.0,
            &ch,
        );
        assert!((far - 1_530.0).abs() < 1.0e-9, "{far}");
        // On the line: the level, from the dry side and from the wet side.
        for rho in [-150.0, -5.0, 0.0, 5.0, 150.0] {
            let dry = n2_surface(
                &n2_ground(1_500.0, rho),
                &pond(1.0),
                false,
                0.0,
                1.0,
                0.0,
                &ch,
            );
            let wet = n2_surface(
                &n2_ground(1_500.0, rho),
                &pond(1.0),
                true,
                0.0,
                1.0,
                0.0,
                &ch,
            );
            assert_eq!((dry, wet), (1_200.0, 1_200.0), "rho {rho}");
        }
        // Never across: a dry column stays over W6's hold, a wet one under it, whatever the texture.
        let mut state = 7u64;
        for _ in 0..2_000 {
            let rho = (rng(&mut state) - 0.5) * 600.0;
            let d = rng(&mut state);
            let beta = 1.0 - fade_f64(d);
            let s_abs = 60.0 * d;
            let raw = 1_200.0 + (rng(&mut state) - 0.3) * 800.0;
            let dry = n2_surface(&n2_ground(raw, rho), &pond(1.0), false, d, beta, s_abs, &ch);
            let wet = n2_surface(&n2_ground(raw, rho), &pond(1.0), true, d, beta, -s_abs, &ch);
            let node_dry = (1.0 - beta) * (raw - rho) + beta * (1_200.0 + s_abs);
            if node_dry >= 1_200.0 + ch.k * s_abs {
                assert!(dry >= 1_200.0 + ch.k * s_abs, "dry {dry} rho {rho} d {d}");
            }
            let node_wet = (1.0 - beta) * (raw - rho) + beta * (1_200.0 - s_abs);
            if node_wet <= 1_200.0 - ch.k * s_abs {
                assert!(wet <= 1_200.0 - ch.k * s_abs, "wet {wet} rho {rho} d {d}");
            }
        }
    }
}
