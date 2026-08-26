//! THE CHILD INDEX — "which of my children could hold this point", answered without touching them all.
//!
//! Owns: the derived uniform grid a realm builds ONCE over its own STATIC direct children, and the
//! candidate query the containment fold and the area-of-interest sweep both read.
//!
//! Does NOT own: the verdict. This says only *which children are worth asking*; whether a point is
//! actually inside one is [`crate::geometry::region_verdict`]'s answer and nothing here may
//! anticipate it. Nor does it own MOTION: a child that moves is simply never indexed (see
//! [`ChildIndex::build`]) — this module carries no orbit, no velocity and no way to ask how anything
//! moves (SL4).
//!
//! # Why this exists (SL9)
//!
//! A parent's child count is unbounded: six moons, six hundred stations, a galaxy's hundred and fifty
//! thousand star systems. The containment fold used to test EVERY child against EVERY occupant on
//! EVERY tick, which is `occupants × children` work per tick and is a defect at that width. Sibling
//! realms may not overlap, so **at most one child can hold a point** — the answer is a lookup, and the
//! only reason it was ever a scan is that nobody had written the lookup.
//!
//! # The invariant that makes one cell enough
//!
//! The grid's edge is at least the DIAMETER of the widest indexed child, and a child is registered in
//! EVERY cell its bound overlaps. So a child whose bound covers a point is always registered in that
//! point's own cell, and a POINT query reads exactly one cell — never its neighbours. The cost is
//! bounded on the insert side instead: a child spans at most `2×2×2` cells, because its diameter
//! cannot exceed one edge.
//!
//! ★ THE SEGMENT QUERY RESTS ON THE SAME INVARIANT, READ ONE STEP FURTHER. A child intersecting a
//! segment covers at least one point of it, and is therefore registered in that point's own cell —
//! which lies inside the box spanned by the two endpoint keys. So enumerating that box WHOLE is
//! sufficient. It is sufficient only when it is enumerated whole, which is why
//! [`ChildIndex::candidates_segment`] ABSTAINS on a span wider than one cell per axis instead of
//! returning a subset: a subset would look exactly like a positive "not here", and this index's
//! answers are trusted as positive.
//!
//! # What it degrades to, stated
//!
//! The edge follows the WIDEST child, so children very much smaller than the widest share cells. In
//! the limit — one huge child and many tiny ones packed inside its span — a query returns them all and
//! this is exactly the scan it replaced, no worse and no better. That case is not hypothetical: it is
//! a star system, whose planets sweep bounds comparable to the system itself. It is also the case
//! where the child count is ten. The width and the degeneracy are anti-correlated, which is why one
//! structure serves both, and [`ChildIndex::candidates`] is measured on both rather than argued about.

use crate::pose::{LatticePos, RealmId, Tier};
use std::collections::{BTreeMap, BTreeSet};

/// One indexed child: its realm, the centre of its bound in the PARENT's frame, and a radius that
/// circumscribes that bound. Plain geometry — the producer states it, and nothing here can ask how the
/// child came to be where it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexedChild {
    /// The child realm this entry answers for.
    pub realm: RealmId,
    /// The centre of the child's bound, in the parent's frame.
    pub centre: LatticePos,
    /// A radius, in metres, that fully contains the child's bound INCLUDING its hysteresis band — the
    /// conservative side. A point outside this radius cannot be a member by any edge of the band, so
    /// dropping the child for such a point cannot change a verdict.
    pub radius_m: f64,
}

/// A realm's index over its own static direct children.
///
/// Built once (see [`ChildIndex::build`]); read per occupant per tick. An EMPTY index answers every
/// query with nothing, which is the correct answer for a realm that indexed no child — the caller
/// still evaluates whatever it holds unconditionally.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChildIndex {
    /// The grid edge in metres — DERIVED from the widest indexed child (never chosen), and always a
    /// power of two so the division is exact at every magnitude. Zero for an empty index.
    edge_m: f64,
    /// Cell key → the realms whose bound overlaps that cell. A `BTreeMap` because the sim bans the
    /// default-hasher map, and because deterministic iteration order is worth more here than a hash.
    cells: BTreeMap<[i64; 3], Vec<RealmId>>,
    /// How many DISTINCT children were indexed. Recorded at build rather than counted from `cells`,
    /// where a child straddling a boundary appears in up to eight of them — the difference between
    /// "how many children" and "how much memory" is exactly the kind of number that reads as the
    /// other one if it is derived carelessly.
    children: usize,
    /// The realms this index holds an entry for — see [`ChildIndex::answers_for`].
    indexed: BTreeSet<RealmId>,
}

impl ChildIndex {
    /// Build the index over `children`, whose centres are all expressed in ONE frame at `tier`.
    ///
    /// **Only pass children that do not move.** A moving child's centre is true for one instant, and an
    /// index that stored it would answer with yesterday's position. The caller decides which children
    /// move and simply omits them; they stay unconditional candidates, which is conservative and
    /// therefore always correct. That decision is the "a static flag may decide WHETHER TO RECOMPUTE,
    /// never WHAT ANYONE READS" clause of SL4 — this module never learns which is which.
    #[must_use]
    pub fn build(children: &[IndexedChild], tier: Tier) -> ChildIndex {
        let edge_m = grid_edge_m(children);
        let mut cells: BTreeMap<[i64; 3], Vec<RealmId>> = BTreeMap::new();
        // NOTHING TO INDEX. No children, or none with any extent, means no grid — and a grid with no
        // edge cannot be divided by. `candidates` takes the same early exit for the same reason.
        if edge_m <= 0.0 {
            return ChildIndex {
                edge_m,
                cells,
                children: children.len(),
                indexed: children.iter().map(|c| c.realm).collect(),
            };
        }
        for child in children {
            let lo = cell_key(child.centre, -child.radius_m, edge_m, tier);
            let hi = cell_key(child.centre, child.radius_m, edge_m, tier);
            insert_span(&mut cells, lo, hi, child.realm);
        }
        ChildIndex {
            edge_m,
            cells,
            children: children.len(),
            indexed: children.iter().map(|c| c.realm).collect(),
        }
    }

    /// The realms worth asking about `point` (expressed in the SAME frame and tier the index was built
    /// in). A SUPERSET of the realms that could hold the point, and never a verdict.
    ///
    /// Reads exactly one cell — see the module's invariant. An empty index answers with nothing.
    #[must_use]
    pub fn candidates(&self, point: LatticePos, tier: Tier) -> &[RealmId] {
        if self.edge_m <= 0.0 {
            return &[];
        }
        let key = cell_key(point, 0.0, self.edge_m, tier);
        self.cells.get(&key).map_or(&[], Vec::as_slice)
    }

    /// The realms worth asking about the SEGMENT `p0 → p1`, appended to `out`. Returns `false` when the
    /// index ABSTAINS — the span is wider than the grid can enumerate — and an abstaining answer obliges
    /// the caller to evaluate every realm, which is conservative and therefore always correct.
    ///
    /// WHY THIS EXISTS. [`ChildIndex::candidates`] reads exactly one cell, keyed on one point, and its
    /// answer is trusted as a positive "not here". A subject that travels THROUGH a child within one
    /// tick has neither endpoint in that child's cell, so the child would be skipped before anything
    /// could ask whether the motion crossed it — the swept verdict would be correct and unreachable.
    ///
    /// CORRECTNESS: a child that intersects the segment covers at least one point of it, and is
    /// therefore registered in that point's own cell — which lies inside the axis-aligned box spanned
    /// by the two endpoint keys. So enumerating that box WHOLE is sufficient for any span at all. It is
    /// sufficient only when enumerated whole: a subset would look exactly like a positive "not here".
    ///
    /// WHEN IT ABSTAINS, AND WHY THE RULE CARRIES NO TUNING NUMBER. The box is enumerated unless doing
    /// so would touch MORE GRID CELLS THAN THE INDEX HAS OCCUPIED — at which point the lookup costs
    /// more than handing the whole field back, so it hands the whole field back instead. That is a
    /// cost argument against this index's own size, not a threshold someone picked, and it has the
    /// property the caller wants: a shard with a handful of children abstains freely and loses
    /// nothing, while a shard with a large field — the case this index exists for — enumerates.
    ///
    /// When the two keys are equal the box is one cell and the answer is EXACTLY today's single-cell
    /// answer, so a stationary subject pays today's lookup and nothing more.
    pub fn candidates_segment(
        &self,
        p0: LatticePos,
        p1: LatticePos,
        tier: Tier,
        out: &mut Vec<RealmId>,
    ) -> bool {
        if self.edge_m <= 0.0 {
            // Nothing indexed: the same answer `candidates` gives, and it is an ANSWER, not an abstain.
            return true;
        }
        let k0 = cell_key(p0, 0.0, self.edge_m, tier);
        let k1 = cell_key(p1, 0.0, self.edge_m, tier);
        // How many cells the endpoint box holds. `abs_diff` is total on i64 (it cannot overflow the
        // way a subtraction can), and the two multiplies are CHECKED: a galaxy-scale step spans more
        // cells than any integer holds, and that is an abstain, never a wrap.
        let span = |i: usize| k0[i].abs_diff(k1[i]).saturating_add(1);
        let box_cells = span(0)
            .checked_mul(span(1))
            .and_then(|v| v.checked_mul(span(2)));
        let Some(box_cells) = box_cells else {
            return false;
        };
        if box_cells > self.cells.len() as u64 {
            return false;
        }
        for x in k0[0].min(k1[0])..=k0[0].max(k1[0]) {
            for y in k0[1].min(k1[1])..=k0[1].max(k1[1]) {
                for z in k0[2].min(k1[2])..=k0[2].max(k1[2]) {
                    if let Some(realms) = self.cells.get(&[x, y, z]) {
                        out.extend_from_slice(realms);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        true
    }

    /// Does this index answer for `realm` at all?
    ///
    /// THE DIFFERENCE BETWEEN A MISS AND IGNORANCE, and the whole safety of skipping. A realm the index
    /// holds an entry for, absent from a query's answer, is positively NOT near the point. A realm the
    /// index never indexed (an ancestor, the realm itself, anything that moves) is simply unknown here,
    /// and a caller must evaluate it. Without this, an empty answer would look the same in both cases.
    #[must_use]
    pub fn answers_for(&self, realm: RealmId) -> bool {
        self.indexed.contains(&realm)
    }

    /// How many DISTINCT children this index answers for — the instrument that makes "the lookup holds
    /// the whole field" a measurement rather than a claim.
    #[must_use]
    pub fn indexed_len(&self) -> usize {
        self.children
    }

    /// How many CELL ENTRIES the grid holds — the memory instrument, and the one that would grow if the
    /// "at most two cells per axis" invariant ever broke. Always at least [`ChildIndex::indexed_len`],
    /// and at most eight times it.
    #[must_use]
    pub fn cell_entries(&self) -> usize {
        self.cells.values().map(Vec::len).sum()
    }

    /// The derived grid edge in metres. Zero for an empty index.
    #[must_use]
    pub fn edge_m(&self) -> f64 {
        self.edge_m
    }
}

/// The grid edge: a power of two at least the DIAMETER of the widest child, so no child can span more
/// than two cells on an axis and a point query never needs a neighbour (a SEGMENT query needs the box
/// its endpoint keys span, and abstains when that box would be wider). Zero for no children, and for
/// children that all have a zero radius — both mean "nothing to index", which
/// [`ChildIndex::candidates`] answers with nothing.
fn grid_edge_m(children: &[IndexedChild]) -> f64 {
    let widest = children.iter().fold(0.0_f64, |acc, c| acc.max(c.radius_m));
    next_power_of_two_m(2.0 * widest)
}

/// The smallest power of two that is at least `v`, for a non-negative `v`. Zero and below map to zero
/// (there is nothing to size a grid against). A power of two so that dividing a coordinate by the edge
/// is exact in binary at every magnitude, which is what keeps a cell key from drifting at the rim of a
/// galaxy — the same discipline the coordinate tiers themselves are built on.
fn next_power_of_two_m(v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    let exp = v.log2().ceil();
    exp.exp2()
}

/// The cell a position falls in, `nudge_m` metres from itself on every axis.
///
/// ★ THE POSITION IS NEVER FLATTENED FROM THE FRAME ORIGIN (slice S4). It used to be, and at galaxy
/// magnitudes that flatten rounds by a quarter of a metre — over a hundred times the whole containment
/// band. A key computed from a rounded position can name a cell the child was never registered in, and
/// because this index's answer is trusted as a positive "not here", the caller would then SKIP the one
/// child that actually holds the point.
///
/// The subtraction happens on the integers instead, against a base derived from the position's own
/// cell, so the only f64 that ever appears is a small local displacement.
///
/// `floor` (not truncate) so the grid is uniform across zero — a truncating divide folds the cells
/// either side of the origin onto one, which would quietly make two distant children share a key.
fn cell_key(p: LatticePos, nudge_m: f64, edge_m: f64, tier: Tier) -> [i64; 3] {
    // How many lattice cells make one grid cell. Both edges are powers of two, so this is exact.
    let cells_per_grid = (edge_m / tier.cell_edge_m()).round() as i64;
    let cell = p.cell();
    let off = p.offset();
    // THE ONLY f64 HERE IS A LOCAL DISPLACEMENT — the child's own radius plus the position's sub-cell
    // remainder. Both are small by construction: a normalized position's remainder is under one lattice
    // cell, and a position built as a pure local offset has no cell magnitude to lose in the first
    // place. What is NEVER done is flattening the position's own distance from the frame origin, which
    // at galaxy magnitudes rounds by a quarter of a metre.
    let cells_of = |m: f64| -> i64 { (m / tier.cell_edge_m()).floor() as i64 };
    let key = |axis: i64, off_axis: f64| -> i64 {
        axis.saturating_add(cells_of(nudge_m + off_axis))
            .div_euclid(cells_per_grid)
    };
    [key(cell.x, off.x), key(cell.y, off.y), key(cell.z, off.z)]
}

/// Register `realm` in every cell of the inclusive span `lo..=hi`. The span is at most `2×2×2` by the
/// module's edge invariant; it is written as a general triple loop anyway, because a bound that turned
/// out wider would silently drop cells if the loop assumed the size.
fn insert_span(
    cells: &mut BTreeMap<[i64; 3], Vec<RealmId>>,
    lo: [i64; 3],
    hi: [i64; 3],
    realm: RealmId,
) {
    for x in lo[0]..=hi[0] {
        for y in lo[1]..=hi[1] {
            for z in lo[2]..=hi[2] {
                cells.entry([x, y, z]).or_default().push(realm);
            }
        }
    }
}

/// The cell span of a child, exposed for the tripwire that pins the "at most two cells per axis"
/// invariant the query rests on.
#[must_use]
pub fn axis_span_cells(child: &IndexedChild, tier: Tier, edge_m: f64) -> [i64; 3] {
    let lo = cell_key(child.centre, -child.radius_m, edge_m, tier);
    let hi = cell_key(child.centre, child.radius_m, edge_m, tier);
    [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::CELL_DOMAIN_MAX;
    use glam::{DVec3, I64Vec3};

    fn child(seed: u64, centre_m: DVec3, radius_m: f64) -> IndexedChild {
        IndexedChild {
            realm: RealmId::Planet(seed),
            centre: LatticePos::local(centre_m),
            radius_m,
        }
    }

    const T: Tier = Tier::Fine;

    // ---------------- THE SEGMENT QUERY (slice S5) ----------------

    #[test]
    fn a_child_the_subject_travels_straight_through_is_named_by_the_segment_and_missed_by_the_point()
     {
        // THE WHOLE REASON THE SEGMENT QUERY EXISTS. Without it the swept verdict would be correct
        // and unreachable: the child holds neither endpoint, so a point lookup answers "not here",
        // and that answer is trusted as positive information rather than as ignorance.
        // A REAL FIELD, because that is the only case where the lookup is worth doing at all: a
        // shard with two children abstains freely and loses nothing.
        let mut kids: Vec<IndexedChild> = (0..64)
            .map(|i| child(i, DVec3::new(i as f64 * 40.0, 0.0, 0.0), 4.0))
            .collect();
        // The one the subject flies straight through, sitting between two grid cells' worth of gap.
        kids.push(child(999, DVec3::new(-100.0, 0.0, 0.0), 4.0));
        let ix = ChildIndex::build(&kids, T);
        assert_eq!(
            ix.edge_m(),
            8.0,
            "the grid edge follows the widest child's diameter"
        );
        // The child at -100 m with radius 4 registers in the grid cells covering [-104, -88) m, so
        // the endpoints must sit either side of THAT, not either side of the child's own bound.
        let a = LatticePos::from_metres(DVec3::new(-110.0, 0.0, 0.0), T);
        let b = LatticePos::from_metres(DVec3::new(-85.0, 0.0, 0.0), T);
        // NEITHER ENDPOINT IS IN THE CHILD'S CELL — a point lookup answers "not here", and that
        // answer is trusted as positive information.
        assert_eq!(ix.candidates(a, T), &[] as &[RealmId]);
        assert_eq!(ix.candidates(b, T), &[] as &[RealmId]);
        let mut out = Vec::new();
        assert!(
            ix.candidates_segment(a, b, T, &mut out),
            "a field this size must enumerate rather than abstain"
        );
        assert_eq!(out, vec![RealmId::Planet(999)]);
    }

    #[test]
    fn a_segment_that_has_not_left_its_grid_cell_answers_exactly_what_the_point_answers() {
        // A stationary or slow subject must pay today's lookup and get today's answer, or the whole
        // "no prior means no change" guarantee would stop at the index.
        let ix = ChildIndex::build(
            &[
                child(1, DVec3::new(0.0, 0.0, 0.0), 4.0),
                child(2, DVec3::new(40.0, 0.0, 0.0), 3.0),
            ],
            T,
        );
        for m in [0.0, 1.0, 3.5, 40.0, 41.0, 100.0] {
            let p = LatticePos::from_metres(DVec3::new(m, 0.0, 0.0), T);
            let mut out = Vec::new();
            assert!(ix.candidates_segment(p, p, T, &mut out));
            assert_eq!(out, ix.candidates(p, T).to_vec(), "at {m} m");
        }
    }

    #[test]
    fn a_span_wider_than_one_cell_per_axis_abstains_rather_than_answering_a_subset() {
        // A SUBSET WOULD LOOK EXACTLY LIKE A POSITIVE "NOT HERE", which is the one thing this index
        // must never say by accident. Abstaining hands the decision back to the full evaluation,
        // which is conservative and therefore always correct.
        let ix = ChildIndex::build(&[child(1, DVec3::new(0.0, 0.0, 0.0), 4.0)], T);
        // Two occupied cells, and a span of 200 m across an 8 m grid — enumerating would cost far
        // more than simply handing back the whole field.
        for v in [
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.0, 100.0, 0.0),
            DVec3::new(0.0, 0.0, 100.0),
        ] {
            let mut out = Vec::new();
            assert!(
                !ix.candidates_segment(
                    LatticePos::from_metres(-v, T),
                    LatticePos::from_metres(v, T),
                    T,
                    &mut out
                ),
                "span {v:?} must abstain"
            );
        }
        // AND THE ABSTAIN IS NOT A WRAP. A step of galaxy magnitude spans more grid cells than any
        // integer holds; the multiply is checked, so it abstains rather than answering on garbage.
        let mut out = Vec::new();
        assert!(!ix.candidates_segment(
            LatticePos::at(I64Vec3::splat(-CELL_DOMAIN_MAX), DVec3::ZERO),
            LatticePos::at(I64Vec3::splat(CELL_DOMAIN_MAX), DVec3::ZERO),
            T,
            &mut out
        ));
    }

    #[test]
    fn an_index_with_nothing_in_it_answers_the_segment_rather_than_abstaining() {
        // Nothing indexed is an ANSWER — there is genuinely no child to name — not ignorance. If this
        // abstained, every shard with no static children would fall back to the full scan every tick.
        let ix = ChildIndex::build(&[child(1, DVec3::new(5.0, 0.0, 0.0), 0.0)], T);
        let mut out = Vec::new();
        assert!(ix.candidates_segment(
            LatticePos::from_metres(DVec3::new(-1.0e6, 0.0, 0.0), T),
            LatticePos::from_metres(DVec3::new(1.0e6, 0.0, 0.0), T),
            T,
            &mut out
        ));
        assert!(out.is_empty());
    }

    #[test]
    fn the_segment_union_names_each_child_once_however_many_cells_it_spans() {
        // A child registered in several of the spanned cells must not be returned several times, or
        // the caller would evaluate it repeatedly and the candidate count would stop measuring the
        // field it claims to measure.
        let mut kids = vec![
            child(1, DVec3::new(0.0, 0.0, 0.0), 4.0),
            child(2, DVec3::new(7.0, 0.0, 0.0), 4.0),
        ];
        // Enough elsewhere that the enumeration is worth doing.
        kids.extend((10..40).map(|i| child(i, DVec3::new(i as f64 * 40.0, 0.0, 0.0), 4.0)));
        let ix = ChildIndex::build(&kids, T);
        let mut out = Vec::new();
        assert!(ix.candidates_segment(
            LatticePos::from_metres(DVec3::new(-1.0, 0.0, 0.0), T),
            LatticePos::from_metres(DVec3::new(8.0, 0.0, 0.0), T),
            T,
            &mut out
        ));
        let mut unique = out.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(out, unique, "the union must be de-duplicated");
        assert_eq!(out, vec![RealmId::Planet(1), RealmId::Planet(2)]);
    }

    #[test]
    fn an_empty_index_answers_nothing_and_has_no_edge() {
        let ix = ChildIndex::build(&[], T);
        assert_eq!(ix.edge_m(), 0.0);
        assert_eq!(ix.indexed_len(), 0);
        assert_eq!(
            ix.candidates(LatticePos::from_metres(DVec3::ZERO, T), T),
            &[] as &[RealmId]
        );
    }

    #[test]
    fn children_with_no_extent_index_nothing() {
        // A zero radius means the child covers no point, so there is nothing a query could hit. The
        // edge degenerates to zero and the query short-circuits — the same arm the empty index takes,
        // reached by a different route.
        let ix = ChildIndex::build(&[child(1, DVec3::new(5.0, 0.0, 0.0), 0.0)], T);
        assert_eq!(ix.edge_m(), 0.0);
        assert_eq!(
            ix.candidates(LatticePos::from_metres(DVec3::new(5.0, 0.0, 0.0), T), T),
            &[] as &[RealmId]
        );
    }

    #[test]
    fn the_edge_is_a_power_of_two_at_least_the_widest_childs_diameter() {
        // DERIVED, never chosen: 3 m radius ⇒ 6 m diameter ⇒ 8 m, the next power of two.
        let ix = ChildIndex::build(
            &[
                child(1, DVec3::ZERO, 3.0),
                child(2, DVec3::new(100.0, 0.0, 0.0), 1.0),
            ],
            T,
        );
        assert_eq!(ix.edge_m(), 8.0);
        // And an exact power of two is returned unchanged rather than doubled.
        let exact = ChildIndex::build(&[child(1, DVec3::ZERO, 8.0)], T);
        assert_eq!(exact.edge_m(), 16.0);
    }

    #[test]
    fn a_point_inside_a_child_finds_that_child_in_its_own_cell() {
        let ix = ChildIndex::build(&[child(7, DVec3::new(1000.0, 0.0, 0.0), 10.0)], T);
        assert_eq!(
            ix.candidates(LatticePos::from_metres(DVec3::new(1000.0, 0.0, 0.0), T), T),
            &[RealmId::Planet(7)]
        );
        assert_eq!(
            ix.candidates(LatticePos::from_metres(DVec3::new(1009.0, 0.0, 0.0), T), T),
            &[RealmId::Planet(7)]
        );
    }

    #[test]
    fn a_point_far_from_every_child_finds_nothing() {
        let ix = ChildIndex::build(&[child(7, DVec3::new(1000.0, 0.0, 0.0), 10.0)], T);
        assert_eq!(
            ix.candidates(LatticePos::from_metres(DVec3::new(-1.0e6, 0.0, 0.0), T), T),
            &[] as &[RealmId]
        );
    }

    #[test]
    fn the_grid_is_uniform_across_the_origin() {
        // FLOOR, not truncate. Two children either side of zero, closer together than one edge, must
        // still land in DIFFERENT cells — a truncating divide folds both onto cell 0 and they would
        // answer each other's queries.
        let ix = ChildIndex::build(
            &[
                child(1, DVec3::new(-12.0, 0.0, 0.0), 4.0),
                child(2, DVec3::new(12.0, 0.0, 0.0), 4.0),
            ],
            T,
        );
        assert_eq!(
            ix.candidates(LatticePos::from_metres(DVec3::new(-12.0, 0.0, 0.0), T), T),
            &[RealmId::Planet(1)]
        );
        assert_eq!(
            ix.candidates(LatticePos::from_metres(DVec3::new(12.0, 0.0, 0.0), T), T),
            &[RealmId::Planet(2)]
        );
    }

    #[test]
    fn no_child_spans_more_than_two_cells_on_any_axis() {
        // THE INVARIANT THE ONE-CELL QUERY RESTS ON. The edge is at least the widest diameter, so a
        // child can straddle at most one boundary per axis. If this ever fails, a query must read
        // neighbours and `candidates` becomes wrong — so it is pinned rather than trusted.
        let children = vec![
            child(1, DVec3::ZERO, 7.0),
            child(2, DVec3::new(31.0, -63.0, 127.0), 7.0),
            child(3, DVec3::new(0.5, 0.5, 0.5), 0.25),
        ];
        let ix = ChildIndex::build(&children, T);
        // The widest span over every child, compared as a VALUE. Written this way rather than as an
        // `assert!` carrying `ix.edge_m()` in its message because a message argument is only evaluated
        // on failure, and the coverage gate counts that never-run evaluation as a real miss.
        let widest_axis = children
            .iter()
            .flat_map(|c| axis_span_cells(c, T, ix.edge_m()))
            .max();
        assert_eq!(
            widest_axis,
            Some(2),
            "the one-cell query is sound only while no child straddles more than one boundary per axis"
        );
    }

    #[test]
    fn a_child_straddling_a_cell_boundary_is_found_from_both_sides() {
        // The insert-the-whole-span half of the invariant: registered in every overlapped cell, so a
        // point in EITHER cell finds it with a single-cell read.
        let ix = ChildIndex::build(&[child(9, DVec3::new(0.0, 0.0, 0.0), 3.0)], T);
        assert_eq!(ix.edge_m(), 8.0);
        assert_eq!(
            ix.candidates(LatticePos::from_metres(DVec3::new(-2.0, 0.0, 0.0), T), T),
            &[RealmId::Planet(9)]
        );
        assert_eq!(
            ix.candidates(LatticePos::from_metres(DVec3::new(2.0, 0.0, 0.0), T), T),
            &[RealmId::Planet(9)]
        );
    }

    #[test]
    fn a_wide_field_answers_with_a_handful_while_holding_many() {
        // THE MEASUREMENT SL9 EXISTS FOR, on the shape a galaxy states: many equal children on a
        // lattice, each far smaller than the gap between them. The index HOLDS them all and ANSWERS
        // with at most one — the answer size does not grow with the child count, which is the whole
        // claim, asserted rather than argued.
        const SIDE: i64 = 25; // 15,625 children — well past any width a machine word could hold
        let mut children = Vec::new();
        for x in 0..SIDE {
            for y in 0..SIDE {
                for z in 0..SIDE {
                    let seed = (x * SIDE * SIDE + y * SIDE + z) as u64;
                    children.push(child(
                        seed,
                        DVec3::new(x as f64, y as f64, z as f64) * 1000.0,
                        10.0,
                    ));
                }
            }
        }
        let ix = ChildIndex::build(&children, T);
        assert_eq!(ix.indexed_len(), (SIDE * SIDE * SIDE) as usize);
        // The memory an entry-per-overlapped-cell grid costs, MEASURED rather than assumed: bounded by
        // eight entries per child, which is the invariant's own arithmetic.
        let entries = ix.cell_entries();
        let ceiling = 8 * ix.indexed_len();
        assert_eq!(
            entries.min(ceiling),
            entries,
            "cell entries exceeded eight per child"
        );
        let worst = children
            .iter()
            .map(|c| ix.candidates(c.centre, T).len())
            .max();
        assert_eq!(
            worst,
            Some(1),
            "a query over a wide field must answer with ONE child, not with the field"
        );
    }

    #[test]
    fn a_child_is_still_found_at_a_magnitude_where_flattening_would_lose_the_key() {
        // ★ SLICE S4's SERVER HALF, and the failure it prevents is a SKIP, not a wobble.
        //
        // At the star placement radius a position is about 1.53e18 lattice cells, whose f64 spacing is
        // 256 cells — a quarter of a metre, and over a hundred times the whole containment band. The
        // index used to key on a position flattened from the frame origin, so near a grid boundary the
        // rounded key could name a cell the child was never registered in.
        //
        // And this index's empty answer is TRUSTED: the caller reads it as a positive "not here" and
        // SKIPS the child. So a lost key does not blur a boundary — it makes a realm unenterable.
        //
        // Positions here are built on the lattice, at a magnitude where the old path could not have
        // survived, and every point inside the child must still find it.
        const R_CELLS: i64 = 1_534_955_097_245_569_024;
        let realm = RealmId::Planet(1);
        let centre = LatticePos::at(I64Vec3::new(R_CELLS, 0, 0), DVec3::ZERO);
        let ix = ChildIndex::build(
            &[IndexedChild {
                realm,
                centre,
                radius_m: 1000.0,
            }],
            T,
        );

        // Sweep across the child, in steps that are NOT multiples of the rounding quantum — the same
        // reason the sibling gate in `pose.rs` sweeps them: a whole-metre step is exact even with the
        // defect present, so it would prove nothing.
        for step in [-1_023_701_i64, -517_003, -101, 0, 101, 517_003, 1_023_701] {
            let probe = LatticePos::at(I64Vec3::new(R_CELLS + step, 0, 0), DVec3::ZERO);
            assert!(
                ix.candidates(probe, T).contains(&realm),
                "a point {} cells from the centre must still find the child that holds it",
                step
            );
        }

        // AND THE ANSWER IS STILL A SUPERSET, not everything: a point well outside finds nothing.
        let far = LatticePos::at(I64Vec3::new(R_CELLS + 100_000_000, 0, 0), DVec3::ZERO);
        assert_eq!(ix.candidates(far, T), &[] as &[RealmId]);
    }

    #[test]
    fn the_degenerate_field_answers_with_everything_and_says_so() {
        // THE STATED DEGRADATION, pinned so it can never become a silent surprise: children whose
        // bounds all overlap one another (a star system's planets, whose swept bounds are comparable
        // to the system) share a cell and the query returns them all. That is the scan — correct, and
        // exactly as expensive as before. It is acceptable ONLY because such a realm has few children,
        // which is what the second assert states.
        let children: Vec<IndexedChild> = (0..8)
            .map(|n| child(n, DVec3::new(n as f64, 0.0, 0.0), 100.0))
            .collect();
        let ix = ChildIndex::build(&children, T);
        assert_eq!(
            ix.candidates(LatticePos::from_metres(DVec3::ZERO, T), T)
                .len(),
            children.len()
        );
        assert!(
            children.len() <= 16,
            "the degenerate answer is only affordable while the child count is small"
        );
    }
}
