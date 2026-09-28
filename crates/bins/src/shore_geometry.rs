//! ★ THE SHORE'S GEOMETRY, AS NUMBERS (plan step 1 of
//! `docs/investigation/2026-09-24/terrain_believability_status.md`; the owner, 2026-09-24:
//! *"shores are squared like teeth"*).
//!
//! **WHY IT EXISTS.** No gate saw the teeth (the status file, §4 "THE BLIND SPOTS"). This module
//! MEASURES the shipped column and the lines a field draws. It never changes the world.
//!
//! 1. **THE SHIPPED COLUMN** ([`RungReader`]). The chunk builder's own reads for one column, in the
//!    chunk builder's order (`vd_terrain::chunk`, `ColumnRead::column`): the slope share, the solved
//!    `Z`, and then the ONE kernel (`column_surface_at`). Beside the surface it keeps the GROUND
//!    BASE (`radius + z + coarse`) and the fine half. The kernel does not return the base, so the
//!    reader computes it from the kernel's own public parts and then CHECKS it in every column: the
//!    base, the fine half and the bench must give the kernel's own surface again
//!    ([`ShippedColumn::base_agrees`]). This check runs the kernel's own last lines again, so it is
//!    a DRIFT ALARM, not an independent proof: it goes false the day the kernel builds its surface
//!    from another base. ★ Since ruling B22 (2026-09-28) the column is the plain ground: the shore
//!    law is removed, and the side and the water it read are removed with it.
//! 2. **THE ZERO CONTOUR** ([`zero_contours`]). Marching squares on a square of columns, chained
//!    into lines: the contour of a field's sign, such as the ground's height over the solve's sea.
//! 3. **THE WINDOW BEARINGS** ([`window_bearings`]). A line cut into windows of one stated length,
//!    each window's chord as an angle against the face grid, in the face's own cells — the frame
//!    gate G-GRID states every long axis in (`crate::grid_bias`).
//! 4. **THE FLAT NULL** ([`flat_null_quantile`]). The flatness a line with NO direction of its own
//!    reads at a given count, by simulation — the method ruling W13.3 used for its 1.006 and 1.012.
//!    The statistic is `crate::grid_bias::AxisHistogram::flatness`, never a copy of it.
//!
//! **Example.** A pilot at 120 km over the belt's coast sees a bank run straight for 8 km, turn 90°,
//! and run straight again. The shipped column says: this column's ground stands 60 m under the
//! solve's sea, the kernel's surface is that ground, and the base check holds. The zero contour of
//! the ground over the sea runs along the bank; its window bearings pile into the 0° and 90° bins.

use std::collections::BTreeMap;
use std::sync::Arc;

use vd_core::glam::DVec3;
use vd_recipe::Gi;
use vd_recipe::height::relief_parts_from;
use vd_recipe::plan::{FieldRead, PlanCharter, column_surface_at, site_direction};
use vd_recipe::terrace::terrace;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{
    Artifact, CoastCounts, PyramidField, SlopeCharter, ZField, sample_z, slope_charter, slope_share,
};
use vd_terrain::macro_lattice::MacroLattice;

use crate::grid_bias::AxisHistogram;

/// The field a rung's chunk reads: the rows themselves at the fine rungs, or one pyramid level at
/// the far rungs — the choice `PyramidField::level_for` makes for every host.
pub enum RungSource<'a> {
    /// Level zero: the artifact's own rows.
    Rows(&'a Artifact),
    /// Level `k` from one: the pyramid's means.
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
    /// ★ THE GROUND BASE: the ladder radius, the solved `Z` and the coarse octaves the field did
    /// not replace — the ground WITHOUT the fine octaves.
    pub base: Gi,
    /// ★ THE SOLVED `Z` the column read (`sample_z` at the rung's level), in gap steps at the
    /// length format: `base` less the ladder radius and the coarse octaves.
    pub z: Gi,
    /// ★ THE SLOPE SHARE the column read (`slope_share`), at the noise's fraction bits: the
    /// solved field's own voice in the roughness factor.
    pub share: Gi,
    /// ★ THE FINE HALF the kernel added (`relief_parts_from(..).fine`): the fine octaves' sum
    /// times the column's roughness factor, in the same unit as `base`.
    pub fine: Gi,
    /// The surface, from the ONE kernel: `terrace(base + fine)`.
    pub surface: Gi,
    /// Whether the base, the fine half and the bench give the kernel's own surface again. A
    /// `false` says this reader's base is NOT the kernel's base (the kernel changed), and every
    /// reading against the base is void. A `true` says only that the kernel still builds its
    /// surface the way this reader repeats it: a drift alarm.
    pub base_agrees: bool,
}

/// ★ THE CHUNK BUILDER'S READ OF ONE RUNG ON ONE FACE: the field that rung reads, the lattice the
/// heights gather on, and the two charters a chunk draws once.
pub struct RungReader<'a> {
    pub body: &'a BodyDefinition,
    pub face: Face,
    pub rung: u8,
    /// The pyramid level the rung reads: zero for the rows.
    pub level: u32,
    pub source: RungSource<'a>,
    /// The level's lattice: the heights' lattice.
    pub lattice: MacroLattice,
    pub charter: PlanCharter,
    pub slope: SlopeCharter,
    /// The first fine octave: the octaves before it are the ones `Z` replaces.
    pub first: usize,
}

impl<'a> RungReader<'a> {
    /// The read of `rung` on `face`, as `vd_terrain::chunk::column_field` sets it up for a chunk of
    /// that face. `counts` is the body's own coast counts (`Artifact::coast_counts`), folded once,
    /// which a pyramid level carries as the artifact's stored fold. `None` where the body has no
    /// lattice or the level does not exist.
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
            lattice,
            charter: body.plan_charter(rung, face),
            slope,
            first: body.first_fine(),
        })
    }

    /// ★ THE COLUMN at cell `(i, j)` of this rung, which must stand ON the face (`0 ≤ i, j <` the
    /// rung's cells an edge): the chunk builder's reads in the chunk builder's order, then the ONE
    /// kernel, then the base the kernel built its surface on. `None` where the field does not hold
    /// the column's stencil.
    #[must_use]
    pub fn column(&self, i: i32, j: i32) -> Option<ShippedColumn> {
        let field = self.source.field();
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
            },
        )
        .h;
        // The kernel's own two halves, from the kernel's own public function, with the very words
        // the kernel read; the base is the ground the kernel adds the fine half to.
        let parts = relief_parts_from(
            &self.charter.octaves,
            self.first,
            self.charter.octave_count.raw() as usize,
            dir,
            &self.charter.roughness,
            share,
        );
        let base = self.charter.radius + z + parts.coarse;
        let again = terrace(&self.charter.terrace, base + parts.fine);
        Some(ShippedColumn {
            dir,
            base,
            z,
            share,
            fine: parts.fine,
            surface,
            base_agrees: again == surface,
        })
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
