//! ★ THE SHORE'S GEOMETRY, AS NUMBERS (plan step 1 of
//! `docs/investigation/2026-09-24/terrain_believability_status.md`; the owner, 2026-09-24:
//! *"shores are squared like teeth"*).
//!
//! **WHY IT EXISTS.** No gate saw the teeth (the status file, §4 "THE BLIND SPOTS"). This module is
//! the shared half of the teeth instrument (`examples/shore_teeth.rs`). It MEASURES the shipped
//! column and the line the shore draws. It never changes the world.
//!
//! 1. **THE SHIPPED COLUMN** ([`RungReader`]). The chunk builder's own reads for one column, in the
//!    chunk builder's order (`vd_terrain::chunk`, `ColumnRead::column`): the cell's SIDE
//!    (`sample_side`, on the fine lattice), the WATER that side names (`column_water`), the slope
//!    share, the solved `Z`, and then the ONE kernel (`column_surface_at`). Beside the surface it
//!    keeps the GROUND BASE the shore law reads (`radius + z + coarse`, the value
//!    `vd_recipe::plan::column_surface_at` hands to `vd_recipe::height::shore`). The kernel does not
//!    return the base, so the reader computes it from the kernel's own public parts and then CHECKS
//!    it in every column: the base, the fine half, the bench and the shore law must give the
//!    kernel's own surface again ([`ShippedColumn::base_agrees`]). This check runs the kernel's own
//!    last lines again, so it is a DRIFT ALARM, not an independent proof: it goes false the day the
//!    kernel builds its surface from another base.
//! 2. **THE ZERO CONTOUR** ([`zero_contours`]). Marching squares on a square of columns, chained
//!    into lines. The drawn shoreline is the contour of the side word; the ground's own waterline is
//!    the contour of `base − water`.
//! 3. **THE WINDOW BEARINGS** ([`window_bearings`]). A line cut into windows of one stated length,
//!    each window's chord as an angle against the face grid, in the face's own cells — the frame
//!    gate G-GRID states every long axis in (`crate::grid_bias`).
//! 4. **THE FLAT NULL** ([`flat_null_quantile`]). The flatness a line with NO direction of its own
//!    reads at a given count, by simulation — the method ruling W13.3 used for its 1.006 and 1.012.
//!    The statistic is `crate::grid_bias::AxisHistogram::flatness`, never a copy of it.
//! 5. **THE EDGES** ([`sea_edge`], [`lake_edge`]). The coast stand is the pinned one
//!    `crate::drainage_gates::pick_stands` states for every instrument; [`sea_edge`] only turns its
//!    node into the node edge the side word changes on. [`lake_edge`] finds the edge of the largest
//!    lake (a reading; `shore_teeth` pins its lake stand, review finding 9 of plan step 2b).
//!
//! **Example.** A pilot at 120 km over the belt's coast sees a bank run straight for 8 km, turn 90°,
//! and run straight again. The shipped column says: this column is LAND by the coast mask, its
//! ground stands 60 m under the sea, and the shore law holds it 15 m over the water. The zero
//! contour of the side word runs along the node's edge; the zero contour of the ground runs 3 km
//! offshore of it. The window bearings of the first pile into the 0° and 90° bins; those of the
//! second spread over all four.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use vd_core::glam::DVec3;
use vd_recipe::Gi;
use vd_recipe::height::{SIDE_INLAND, SIDE_LAND, SIDE_SEA, relief_parts_from, shore};
use vd_recipe::plan::{FieldRead, PlanCharter, column_surface_at, site_direction};
use vd_recipe::terrace::terrace;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{
    Artifact, CoastCounts, FACIES_SHIFT, PyramidField, Side, SlopeCharter, ZField, column_water,
    sample_side, sample_z, slope_charter, slope_share,
};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::solve::{FACIES_LAKE, FACIES_SEA};

use crate::grid_bias::AxisHistogram;

/// The field a rung's chunk reads: the rows themselves at the fine rungs, or one pyramid level at
/// the far rungs — the choice `PyramidField::level_for` makes for every host.
pub enum RungSource<'a> {
    /// Level zero: the artifact's own rows.
    Rows(&'a Artifact),
    /// Level `k` from one: the pyramid's means, with the body's coast mask and counts.
    Level(PyramidField),
}

impl RungSource<'_> {
    /// The field as the chunk builder holds it.
    #[must_use]
    pub fn field(&self) -> &dyn ZField {
        match self {
            RungSource::Rows(rows) => {
                let field: &dyn ZField = *rows;
                field
            }
            RungSource::Level(level) => level,
        }
    }
}

/// ★ ONE COLUMN AS THE CHUNK BUILDER READS IT. Every length is a radius in gap steps at the length
/// format (`vd_recipe::cell::LENGTH_BITS`): `vd_terrain::metres_of_q28` of a DIFFERENCE of two of
/// them is metres.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShippedColumn {
    /// The column's direction, at the bend's fraction bits.
    pub dir: [Gi; 3],
    /// The cell's side of the coast mask (`sample_side`): land, the sea, a lake, a pocket's dry
    /// floor (2026-09-25), or `None`.
    pub side: Option<Side>,
    /// The side WORD the shore law obeys (`column_water`'s side): `SIDE_LAND`, `SIDE_SEA`,
    /// `SIDE_INLAND` (land with no sea node in its read stencil, 2026-09-24) or `SIDE_UNKNOWN`. A
    /// lake cell whose level names no lake reads the land rule's word here.
    pub side_word: Gi,
    /// The water that STANDS over the column, as a radius: the body's sea, a lake's level, the
    /// nearest row's (no mask), or ZERO for an inland column (2026-09-24, plan step 2 part 3b).
    pub water: Gi,
    /// ★ The water the shore law HOLDS the column against (`column_water`'s `hold`): the sea for
    /// every land column, inland ones included, and the standing water for the rest.
    pub hold: Gi,
    /// ★ THE GROUND BASE the shore law reads: the ladder radius, the solved `Z` and the coarse
    /// octaves the field did not replace — the ground WITHOUT the fine octaves.
    pub base: Gi,
    /// ★ THE SOLVED `Z` the column read (`sample_z` at the rung's level), in gap steps at the
    /// length format: `base` less the ladder radius and the coarse octaves (the shoreline bench,
    /// 2026-09-25).
    pub z: Gi,
    /// ★ THE SLOPE SHARE the column read (`slope_share`), at the noise's fraction bits: the
    /// solved field's own voice in the roughness factor.
    pub share: Gi,
    /// ★ THE FINE HALF the kernel added (`relief_parts_from(..).fine`): the fine octaves' sum
    /// times the column's roughness factor, in the same unit as `base`.
    pub fine: Gi,
    /// ★ THE RAW SURFACE: `terrace(base + fine)`, the kernel's surface BEFORE the shore law. The
    /// shore law turns it into [`ShippedColumn::surface`].
    pub raw: Gi,
    /// The final surface, from the ONE kernel.
    pub surface: Gi,
    /// Whether the base, the fine half, the bench and the shore law give the kernel's own surface
    /// again. A `false` says this reader's base is NOT the kernel's base (the kernel changed), and
    /// every reading against the base is void. A `true` says only that the kernel still builds its
    /// surface the way this reader repeats it: a drift alarm.
    pub base_agrees: bool,
}

impl ShippedColumn {
    /// The side the shore law DRAWS: `Some(true)` for a column held under its water, `Some(false)`
    /// for one held over it, `None` where no side is stated and the ground's own sign decides.
    #[must_use]
    pub fn drawn_wet(&self) -> Option<bool> {
        if self.side_word == SIDE_SEA {
            Some(true)
        } else if (self.side_word == SIDE_LAND) | (self.side_word == SIDE_INLAND) {
            Some(false)
        } else {
            None
        }
    }

    /// Whether the column's water is a LAKE (the cell's own mask word), and not the sea.
    #[must_use]
    pub fn lake(&self) -> bool {
        self.side == Some(Side::Lake)
    }

    /// ★ THE GROUND'S OWN SIDE: over the water the shore law holds it against (or at it) by the base
    /// alone — the rule the shore law runs where no side is stated (`vd_recipe::height::shore`,
    /// `g >= 0`).
    #[must_use]
    pub fn ground_dry(&self) -> bool {
        self.base >= self.hold
    }
}

/// ★ THE CHUNK BUILDER'S READ OF ONE RUNG ON ONE FACE: the field that rung reads, the lattice the
/// heights gather on, the fine lattice the side is read on, and the two charters a chunk draws once.
pub struct RungReader<'a> {
    pub body: &'a BodyDefinition,
    pub face: Face,
    pub rung: u8,
    /// The pyramid level the rung reads: zero for the rows.
    pub level: u32,
    pub source: RungSource<'a>,
    /// The body's own (fine) lattice: the side's lattice at every rung.
    pub fine: MacroLattice,
    /// The level's lattice: the heights' and the water's lattice.
    pub lattice: MacroLattice,
    pub charter: PlanCharter,
    pub slope: SlopeCharter,
    /// The first fine octave: the octaves before it are the ones `Z` replaces.
    pub first: usize,
}

impl<'a> RungReader<'a> {
    /// The read of `rung` on `face`, as `vd_terrain::chunk::column_field` sets it up for a chunk of
    /// that face. `counts` is the body's own coast counts (`Artifact::coast_counts`), folded once.
    /// `None` where the body has no lattice or the level does not exist.
    #[must_use]
    pub fn new(
        body: &'a BodyDefinition,
        artifact: &'a Artifact,
        counts: &Arc<CoastCounts>,
        face: Face,
        rung: u8,
    ) -> Option<RungReader<'a>> {
        let fine = body.macro_lattice()?;
        let levels = artifact.pyramid.len() as u32;
        let level = PyramidField::level_for(&fine, levels, rung);
        let source = if level == 0 {
            RungSource::Rows(artifact)
        } else {
            RungSource::Level(PyramidField::of(artifact, level, counts)?)
        };
        let lattice = fine.coarser(source.field().level())?;
        let slope = slope_charter(body, &lattice, rung);
        Some(RungReader {
            body,
            face,
            rung,
            level,
            source,
            fine,
            lattice,
            charter: body.plan_charter(rung, face),
            slope,
            first: body.first_fine(),
        })
    }

    /// ★ THE READ of `rung` on `face` at a STATED pyramid level (the shoreline bench's one-change
    /// variant R0b, 2026-09-25): the same read as [`RungReader::new`], but the heights gather on
    /// `level` (zero: the rows) whatever `PyramidField::level_for` would pick for the rung. The
    /// side still reads the fine lattice. No shipped path reads a rung at another level: this is
    /// an instrument's what-if, and [`RungReader::new`] is the shipped read. `None` where the body
    /// has no lattice or the level does not exist.
    ///
    /// **Example.** The high lake seen from 1 000 km draws rung 11 on level 1 (16 km means). The
    /// bench reads the same rung-11 columns on level 0 (the rows themselves), so the one change
    /// "the block means" can be tested alone.
    #[must_use]
    pub fn with_level(
        body: &'a BodyDefinition,
        artifact: &'a Artifact,
        counts: &Arc<CoastCounts>,
        face: Face,
        rung: u8,
        level: u32,
    ) -> Option<RungReader<'a>> {
        let fine = body.macro_lattice()?;
        let source = if level == 0 {
            RungSource::Rows(artifact)
        } else {
            RungSource::Level(PyramidField::of(artifact, level, counts)?)
        };
        let lattice = fine.coarser(source.field().level())?;
        let slope = slope_charter(body, &lattice, rung);
        Some(RungReader {
            body,
            face,
            rung,
            level,
            source,
            fine,
            lattice,
            charter: body.plan_charter(rung, face),
            slope,
            first: body.first_fine(),
        })
    }

    /// ★ THE COLUMN at cell `(i, j)` of this rung, which must stand ON the face (`0 ≤ i, j <` the
    /// rung's cells an edge): the chunk builder's reads in the chunk builder's order, then the ONE
    /// kernel, then the base the kernel handed the shore law. `None` where the field does not hold
    /// the column's stencil.
    #[must_use]
    pub fn column(&self, i: i32, j: i32) -> Option<ShippedColumn> {
        let field = self.source.field();
        let side = sample_side(&self.fine, field, self.face, self.rung, i, j);
        let read = column_water(
            self.body,
            &self.lattice,
            field,
            self.face,
            self.rung,
            i,
            j,
            side,
        );
        let (water, hold, side_word) = (read.water, read.hold, read.side);
        let share = slope_share(
            &self.slope,
            &self.lattice,
            field,
            self.face,
            self.rung,
            i,
            j,
        )?;
        let z = sample_z(&self.lattice, field, self.face, self.rung, i, j)?;
        let dir = site_direction(&self.charter, i32::from(self.face.index()), i, j);
        let surface = column_surface_at(
            &self.charter,
            dir,
            &FieldRead {
                z,
                first: self.first,
                slope_share: share,
                water: hold,
                side: side_word,
            },
        )
        .h;
        // The kernel's own two halves, from the kernel's own public function, with the very words
        // the kernel read; the base is the ground the kernel hands the shore law.
        let parts = relief_parts_from(
            &self.charter.octaves,
            self.first,
            self.charter.octave_count.raw() as usize,
            dir,
            &self.charter.roughness,
            share,
        );
        let base = self.charter.radius + z + parts.coarse;
        let raw = terrace(&self.charter.terrace, base + parts.fine);
        let again = shore(base, hold, raw, side_word);
        Some(ShippedColumn {
            dir,
            side,
            side_word,
            water,
            hold,
            base,
            z,
            share,
            fine: parts.fine,
            raw,
            surface,
            base_agrees: again == surface,
        })
    }

    /// The body's own sea as a radius (ZERO for a body with none): what the shore law holds a SEA
    /// column against.
    #[must_use]
    pub fn sea_radius(&self) -> Gi {
        self.charter.sea_radius
    }
}

/// A direction at the bend's fraction bits as a plain unit vector.
#[must_use]
pub fn unit_of(dir: [Gi; 3]) -> DVec3 {
    DVec3::new(
        dir[0].raw() as f64,
        dir[1].raw() as f64,
        dir[2].raw() as f64,
    )
    .normalize()
}

/// ★ THE ZERO CONTOUR of a square of samples, as chained lines. `values` holds `n × n` samples in
/// the order `b · n + a`; a sample at or over zero is on the POSITIVE side (the dry side, the
/// ground over its water), one under zero on the negative side. A point of a line is `(a, b)` in
/// columns of the square: `a` along the face's `i` axis, `b` along its `j` axis. A cell with a
/// `None` corner draws nothing.
///
/// A crossing stands on the cell edge where the sign changes, at the linear interpolation of the
/// two samples. ★ A SADDLE (the four corners alternate) is split by the cell's MEAN: the corners
/// that share the mean's sign are joined through the cell. On a field of ±1 the mean of a saddle
/// is zero, which is the positive side, so the DRY corners join — a stated choice.
///
/// Every crossing point is shared by at most two cells, so the segments chain into open lines and
/// closed loops (a loop repeats its first point at its end).
#[must_use]
pub fn zero_contours(n: usize, values: &[Option<f64>]) -> Vec<Vec<(f64, f64)>> {
    // A horizontal edge (a, b)–(a + 1, b) is 2·(b·n + a); a vertical edge (a, b)–(a, b + 1) is one
    // more.
    let h_edge = |a: usize, b: usize| 2 * (b * n + a);
    let v_edge = |a: usize, b: usize| 2 * (b * n + a) + 1;
    let mut points: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    let mut segments: Vec<(usize, usize)> = Vec::new();
    if n < 2 || values.len() < n * n {
        return Vec::new();
    }
    for b in 0..n - 1 {
        for a in 0..n - 1 {
            let (Some(v0), Some(v1), Some(v2), Some(v3)) = (
                values[b * n + a],
                values[b * n + a + 1],
                values[(b + 1) * n + a + 1],
                values[(b + 1) * n + a],
            ) else {
                continue;
            };
            let pos = [v0 >= 0.0, v1 >= 0.0, v2 >= 0.0, v3 >= 0.0];
            // The four edges: bottom (c0, c1), right (c1, c2), top (c3, c2), left (c0, c3).
            let ids = [
                h_edge(a, b),
                v_edge(a + 1, b),
                h_edge(a, b + 1),
                v_edge(a, b),
            ];
            let ends = [
                (v0, v1, (a, b), (a + 1, b)),
                (v1, v2, (a + 1, b), (a + 1, b + 1)),
                (v3, v2, (a, b + 1), (a + 1, b + 1)),
                (v0, v3, (a, b), (a, b + 1)),
            ];
            let crossing = [
                pos[0] != pos[1],
                pos[1] != pos[2],
                pos[3] != pos[2],
                pos[0] != pos[3],
            ];
            let mut list = [0usize; 4];
            let mut count = 0usize;
            for (e, (&id, &(p, q, from, to))) in ids.iter().zip(ends.iter()).enumerate() {
                if crossing[e] {
                    points
                        .entry(id)
                        .or_insert_with(|| crossing_point(p, q, from, to));
                    list[count] = e;
                    count += 1;
                }
            }
            if count == 2 {
                segments.push((ids[list[0]], ids[list[1]]));
            } else if count == 4 {
                let mean_pos = (v0 + v1 + v2 + v3) / 4.0 >= 0.0;
                if mean_pos == pos[0] {
                    // c0 and c2 join through the cell: the line cuts off c1 and c3.
                    segments.push((ids[0], ids[1]));
                    segments.push((ids[2], ids[3]));
                } else {
                    // c1 and c3 join: the line cuts off c0 and c2.
                    segments.push((ids[0], ids[3]));
                    segments.push((ids[1], ids[2]));
                }
            }
        }
    }
    chain(&points, &segments)
}

/// The crossing of the zero level on the edge from `from` (sample `p`) to `to` (sample `q`); the
/// two samples stand on two sides of zero, so `p − q` is never zero.
fn crossing_point(p: f64, q: f64, from: (usize, usize), to: (usize, usize)) -> (f64, f64) {
    let t = p / (p - q);
    (
        from.0 as f64 + t * (to.0 as f64 - from.0 as f64),
        from.1 as f64 + t * (to.1 as f64 - from.1 as f64),
    )
}

/// The segments chained into lines: every point joins at most two segments, so a walk from any
/// unused segment in both directions finds its whole line.
fn chain(
    points: &BTreeMap<usize, (f64, f64)>,
    segments: &[(usize, usize)],
) -> Vec<Vec<(f64, f64)>> {
    let mut adjacent: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (s, &(p, q)) in segments.iter().enumerate() {
        adjacent.entry(p).or_default().push(s);
        adjacent.entry(q).or_default().push(s);
    }
    let mut used = vec![false; segments.len()];
    let walk = |start: usize, used: &mut Vec<bool>| -> Vec<usize> {
        let mut out = Vec::new();
        let mut at = start;
        loop {
            let next = adjacent
                .get(&at)
                .and_then(|list| list.iter().copied().find(|&s| !used[s]));
            let Some(s) = next else {
                break;
            };
            used[s] = true;
            let (x, y) = segments[s];
            at = if x == at { y } else { x };
            out.push(at);
        }
        out
    };
    let mut lines = Vec::new();
    for s0 in 0..segments.len() {
        if used[s0] {
            continue;
        }
        used[s0] = true;
        let (p, q) = segments[s0];
        let mut back = walk(p, &mut used);
        back.reverse();
        back.push(p);
        back.push(q);
        back.extend(walk(q, &mut used));
        lines.push(
            back.iter()
                .filter_map(|id| points.get(id).copied())
                .collect(),
        );
    }
    lines
}

/// ★ THE WINDOW BEARINGS of one line: the line cut into consecutive windows of `window` columns of
/// ARC length, and each window's CHORD as an angle in degrees in `[0, 180)` against the face's `i`
/// axis — 0° along `i`, 90° along `j`, 45° and 135° the stencil's diagonals. A rest shorter than a
/// window is dropped; a window whose chord has no length (it returns to its start) is dropped.
#[must_use]
pub fn window_bearings(line: &[(f64, f64)], window: f64) -> Vec<f64> {
    let mut out = Vec::new();
    if line.len() < 2 || window <= 0.0 {
        return out;
    }
    let mut start = line[0];
    let mut prev = line[0];
    let mut travelled = 0.0f64;
    let mut k = 1usize;
    while k < line.len() {
        let next = line[k];
        let seg = ((next.0 - prev.0).powi(2) + (next.1 - prev.1).powi(2)).sqrt();
        if seg > 0.0 && travelled + seg >= window {
            let t = (window - travelled) / seg;
            let cut = (
                prev.0 + t * (next.0 - prev.0),
                prev.1 + t * (next.1 - prev.1),
            );
            if let Some(d) = bearing(start, cut) {
                out.push(d);
            }
            start = cut;
            prev = cut;
            travelled = 0.0;
        } else {
            travelled += seg;
            prev = next;
            k += 1;
        }
    }
    out
}

/// The chord from `from` to `to` as an axis in degrees in `[0, 180)`; `None` for no length.
fn bearing(from: (f64, f64), to: (f64, f64)) -> Option<f64> {
    let (da, db) = (to.0 - from.0, to.1 - from.1);
    if da.hypot(db) < 1.0e-9 {
        return None;
    }
    let mut d = db.atan2(da).to_degrees();
    while d < 0.0 {
        d += 180.0;
    }
    while d >= 180.0 {
        d -= 180.0;
    }
    Some(d)
}

/// ★ THE FLAT NULL: the `q` quantile of the flatness (`AxisHistogram::flatness`, the biggest bin
/// over the mean bin) that `count` axes with NO direction of their own read in `bins` bins, over
/// `trials` draws. This is the method ruling W13.3 used (its table: "a perfect router … 1.006 (95th
/// 1.012, measured by simulation)" at 96 804 axes). A flat histogram of few axes reads well over one by
/// chance alone, so a band must come from the count in hand, never from W13.3's count.
///
/// The draw is a fixed xorshift, so the band is the same number on every run.
///
/// ★ THE AXES ARE INDEPENDENT IN THE NULL. The windows of one shoreline are not: a long straight
/// run gives many windows at one angle. So the band is STRICT for a coast of few long lines — a
/// natural coast of few lines can read over it. The ground's own waterline, read in the same square
/// with the same windows, is the control that says how much of a reading is the square's own.
#[must_use]
pub fn flat_null_quantile(count: usize, bins: usize, trials: usize, q: f64) -> f64 {
    quantile_sorted(&flat_null_sorted(count, bins, trials), q)
}

/// The flatness of every draw of [`flat_null_quantile`], SORTED, so a caller can read more than one
/// quantile of one simulation (the method check reads ruling W13.3's median and its 95th). Empty
/// where `count`, `bins` or `trials` is zero.
#[must_use]
pub fn flat_null_sorted(count: usize, bins: usize, trials: usize) -> Vec<f64> {
    if count == 0 || bins == 0 || trials == 0 {
        return Vec::new();
    }
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut flat = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut h = AxisHistogram::new(bins);
        for _ in 0..count {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
            h.add(u * 180.0);
        }
        flat.push(h.flatness());
    }
    flat.sort_by(f64::total_cmp);
    flat
}

/// The `q` quantile of a SORTED slice by the index `⌊len · q⌋` (held to the last), the rule the
/// census examples use (`v[len · 9 / 10]` for the 90th); `NaN` for an empty slice.
#[must_use]
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let k = ((sorted.len() as f64 * q) as usize).min(sorted.len() - 1);
    sorted[k]
}

/// ★ ONE NODE EDGE OF THE COAST MASK: a node the mask calls DRY beside a node it calls WET, on one
/// face. The side word changes at the line between them at every rung from 0 to 13. Where the wet
/// node is a CORNER neighbour ([`sea_edge`] tries the four sides first), the two nodes share only a
/// corner, and [`NodeEdge::middle`] is that corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeEdge {
    pub face: Face,
    /// The dry node's `(i, j)` on the face.
    pub dry: (i32, i32),
    /// The wet node's `(i, j)` on the face: a neighbour of `dry`.
    pub wet: (i32, i32),
    /// The dry node's discharge class (`Row::discharge_log`), the coast pick's own score.
    pub discharge: u8,
}

impl NodeEdge {
    /// ★ THE MIDDLE OF THE EDGE in rung-0 cells of the face: the mean of the two nodes' centres. A
    /// node `k` covers rung-0 cells `[k · size, (k + 1) · size)`, so for two nodes side by side the
    /// middle stands on the first cell of the higher node, and at every rung `r` the cell index
    /// `middle >> r` is the first column past the side's change.
    #[must_use]
    pub fn middle(&self, cells_per_node: u32) -> (i64, i64) {
        let size = i64::from(cells_per_node);
        (
            (i64::from(self.dry.0) + i64::from(self.wet.0) + 1) * size / 2,
            (i64::from(self.dry.1) + i64::from(self.wet.1) + 1) * size / 2,
        )
    }
}

/// Whether `(i, j)` stands at least `margin` nodes inside its face.
fn inside(lattice: &MacroLattice, i: i32, j: i32, margin: i32) -> bool {
    let last = lattice.edge as i32 - 1 - margin;
    (margin..=last).contains(&i) && (margin..=last).contains(&j)
}

/// The first neighbour of `(face, i, j)` ON THE SAME FACE, in the stated order — the four sides
/// first, then the four corners — for which `wet` holds.
fn first_neighbour(
    lattice: &MacroLattice,
    face: Face,
    i: i32,
    j: i32,
    wet: &dyn Fn(u32) -> bool,
) -> Option<(i32, i32)> {
    const ORDER: [(i32, i32); 8] = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    ORDER
        .iter()
        .map(|&(di, dj)| (i + di, j + dj))
        .filter(|&(ni, nj)| inside(lattice, ni, nj, 0))
        .find(|&(ni, nj)| wet(lattice.index(face, ni, nj)))
}

/// ★ THE COAST EDGE AT ONE NODE: the node, and its first SEA neighbour on the same face by the
/// facies bit the coast rule reads (`crate::drainage_gates::rule_picks`), so a stand and its edge
/// agree. The coast itself is NEVER chosen here: the three stands live once, in
/// `crate::drainage_gates`, and this only turns the stand's node into the node edge the side word
/// changes on. `None` where the node has no sea neighbour on its own face.
#[must_use]
pub fn sea_edge(artifact: &Artifact, lattice: &MacroLattice, node: u32) -> Option<NodeEdge> {
    let facies = |m: u32| artifact.rows[m as usize].receiver_facies >> FACIES_SHIFT;
    let (face, i, j) = lattice.split(node);
    let sea = |m: u32| facies(m) & FACIES_SEA != 0;
    let wet = first_neighbour(lattice, face, i, j, &sea)?;
    Some(NodeEdge {
        face,
        dry: (i, j),
        wet,
        discharge: artifact.rows[node as usize].discharge_log,
    })
}

/// ★ THE EDGE OF THE LARGEST LAKE: the lake patches as `examples/lake_census.rs` counts them (lake
/// and not sea by the facies, joined through the eight neighbours), largest first (the first node
/// breaks a tie); in the first patch that has one, the first node by index that stands `margin`
/// nodes inside its face with a DRY node (neither sea nor lake) on one of its four sides. Returns
/// the edge and the patch's size in nodes. `margin` must be at least one, so the four sides stand
/// on the node's own face.
#[must_use]
pub fn lake_edge(
    artifact: &Artifact,
    lattice: &MacroLattice,
    margin: i32,
) -> Option<(NodeEdge, usize)> {
    let n = artifact.node_count();
    let facies = |node: u32| artifact.rows[node as usize].receiver_facies >> FACIES_SHIFT;
    let is_lake = |node: u32| {
        let f = facies(node);
        (f & FACIES_LAKE != 0) & (f & FACIES_SEA == 0)
    };
    let is_dry = |node: u32| {
        let f = facies(node);
        (f & FACIES_LAKE == 0) & (f & FACIES_SEA == 0)
    };
    let mut seen = vec![false; n];
    let mut patches: Vec<Vec<u32>> = Vec::new();
    for start in 0..n as u32 {
        if seen[start as usize] || !is_lake(start) {
            continue;
        }
        seen[start as usize] = true;
        let mut patch = Vec::new();
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            patch.push(node);
            for m in lattice.neighbours(node) {
                if m == vd_terrain::macro_lattice::NO_NODE || seen[m as usize] || !is_lake(m) {
                    continue;
                }
                seen[m as usize] = true;
                queue.push_back(m);
            }
        }
        patch.sort_unstable();
        patches.push(patch);
    }
    // Largest first; a tie goes to the patch whose first node is smaller.
    patches.sort_by(|x, y| y.len().cmp(&x.len()).then(x[0].cmp(&y[0])));
    for patch in &patches {
        for &node in patch {
            let (face, i, j) = lattice.split(node);
            if !inside(lattice, i, j, margin) {
                continue;
            }
            let sides = [(1, 0), (-1, 0), (0, 1), (0, -1)];
            let dry = sides
                .iter()
                .map(|&(di, dj)| (i + di, j + dj))
                .find(|&(ni, nj)| is_dry(lattice.index(face, ni, nj)));
            if let Some(dry) = dry {
                return Some((
                    NodeEdge {
                        face,
                        dry,
                        wet: (i, j),
                        discharge: artifact.rows[lattice.index(face, dry.0, dry.1) as usize]
                            .discharge_log,
                    },
                    patch.len(),
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight shore along `j` at `a = 2.5` makes one line, and all its windows read 90°.
    #[test]
    fn a_straight_shore_is_one_line_at_ninety_degrees() {
        let n = 6;
        let values: Vec<Option<f64>> = (0..n * n)
            .map(|k| Some(if k % n <= 2 { 1.0 } else { -1.0 }))
            .collect();
        let lines = zero_contours(n, &values);
        assert_eq!(lines.len(), 1);
        let line = &lines[0];
        assert_eq!(line.len(), n);
        for p in line {
            assert_eq!(p.0, 2.5);
        }
        let windows = window_bearings(line, 2.0);
        assert_eq!(windows.len(), 2);
        for d in windows {
            assert!((d - 90.0).abs() < 1.0e-9, "a window reads {d}°");
        }
    }

    /// A single wet column inside dry ground is one closed loop of four points and its first again.
    #[test]
    fn a_pond_is_one_closed_loop() {
        let n = 3;
        let mut values = vec![Some(1.0); n * n];
        values[4] = Some(-1.0);
        let lines = zero_contours(n, &values);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].len(), 5);
        assert_eq!(lines[0].first(), lines[0].last());
    }

    /// The flat null falls toward one as the count grows, and stays over one.
    #[test]
    fn the_flat_null_falls_with_the_count() {
        let few = flat_null_quantile(20, 4, 400, 0.95);
        let many = flat_null_quantile(2_000, 4, 400, 0.95);
        assert!(few > many);
        assert!(many > 1.0);
    }
}
