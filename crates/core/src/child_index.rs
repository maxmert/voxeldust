//! THE CHILD INDEX — "which of my children could hold this point, or meet this line", answered
//! without touching them all.
//!
//! Owns: an R*-tree (the `rstar` crate, owner-approved 2026-09-04) a realm builds over its own
//! STATIC direct children, and the candidate query the containment fold and the area-of-interest
//! sweep both read.
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
//! # Why a tree, and not the grid it replaces
//!
//! The first lookup was a uniform grid whose cell was sized by the WIDEST child. MEASURED on the
//! fifth flight (2026-09-03): the galaxy's widest star system reaches 18 light years, so its cell
//! held 9 640 systems, and every query filtered all of them — 10 ms per hull, and a ceiling of about
//! two flying hulls before the galaxy's tick went stale. A tree has no cell: each child is stored
//! under its OWN box, and a query descends only into the boxes that meet it, so the cost is
//! `log(children) + answers`, never the size of the widest child's neighbourhood. The owner's word
//! (2026-09-04): *"agree that we need to implement rstar"*.
//!
//! # Two stages, and why the second is exact
//!
//! The tree sorts children by an f64 box in metres from the lattice origin. At galaxy magnitudes an
//! f64 metre rounds by the hundred, so every box is PADDED by the rounding at its own magnitude
//! (plus one lattice cell), and a query pads its own line the same way. The box stage is therefore
//! a SUPERSET, never a miss. The second stage tests the child's exact bounding sphere against the
//! asked point or line on the LATTICE ([`LatticePos::delta_m`] is integer-exact), so the answer is
//! the same one the grid gave, and this module's empty answer stays what the fold trusts it to be:
//! a positive "not here".
//!
//! # Determinism
//!
//! A query's order comes from the tree's shape, and the shape from the insertion history. Every
//! answer is therefore SORTED and de-duplicated before it leaves, so a built index and a grown one
//! answer byte-identically.

use crate::glam::DVec3;
use crate::pose::{LatticePos, RealmId, Tier};
use rstar::{AABB, RTree, RTreeObject, SelectionFunction};
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

/// One leaf of the tree: the child's realm, its EXACT bounding sphere on the lattice (the second
/// stage reads these), and the padded f64 box the tree sorts it by (the first stage reads this).
#[derive(Clone, Debug, PartialEq)]
struct Leaf {
    realm: RealmId,
    centre: LatticePos,
    radius_m: f64,
    lo: [f64; 3],
    hi: [f64; 3],
}

impl RTreeObject for Leaf {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(self.lo, self.hi)
    }
}

/// A realm's index over its own static direct children.
///
/// Built once (see [`ChildIndex::build`]), grown and shrunk one child at a time (the ruler switch),
/// read per occupant per tick. An EMPTY index answers every query with nothing, which is the correct
/// answer for a realm that indexed no child — the caller still evaluates whatever it holds
/// unconditionally.
#[derive(Clone, Debug, Default)]
pub struct ChildIndex {
    /// The R*-tree over every child with an extent.
    tree: RTree<Leaf>,
    /// How many DISTINCT children were indexed, extent or not — the difference between "how many
    /// children" and "how many leaves" is exactly the kind of number that reads as the other one if
    /// it is derived carelessly.
    children: usize,
    /// The realms this index holds an entry for — see [`ChildIndex::answers_for`].
    indexed: BTreeSet<RealmId>,
    /// Each child's own leaf, so a removal hands the tree the exact leaf it holds (★ SL9: a removal
    /// costs the child's own path down the tree, never a walk of the field).
    leaves: BTreeMap<RealmId, Leaf>,
}

impl PartialEq for ChildIndex {
    /// Two indexes are equal when they hold the same children with the same spheres. The tree's
    /// SHAPE is deliberately not compared: a built tree and a grown one differ in shape and answer
    /// identically, and it is the answers this module promises.
    fn eq(&self, other: &Self) -> bool {
        (self.children, &self.indexed, &self.leaves)
            == (other.children, &other.indexed, &other.leaves)
    }
}

impl ChildIndex {
    /// Build the index over `children`, whose centres are all expressed in ONE frame at `tier`.
    ///
    /// **Only pass children that do not move.** A moving child's centre is true for one instant, and an
    /// index that stored it would answer with yesterday's position. The caller decides which children
    /// move and simply omits them; they stay unconditional candidates, which is conservative and
    /// therefore always correct. That decision is the "a static flag may decide WHETHER TO RECOMPUTE,
    /// never WHAT ANYONE READS" clause of SL4 — this module never learns which is which.
    ///
    /// The tree is BULK-LOADED, which packs a galaxy's field into a balanced tree in one pass; an
    /// [`insert`](Self::insert) per child would reach the same answers more slowly.
    #[must_use]
    pub fn build(children: &[IndexedChild], tier: Tier) -> ChildIndex {
        let leaves: BTreeMap<RealmId, Leaf> = children
            .iter()
            .filter_map(|c| leaf_of(c, tier).map(|leaf| (c.realm, leaf)))
            .collect();
        ChildIndex {
            tree: RTree::bulk_load(leaves.values().cloned().collect()),
            children: children.len(),
            indexed: children.iter().map(|c| c.realm).collect(),
            leaves,
        }
    }

    /// The realms worth asking about `point` (expressed in the SAME frame and tier the index was built
    /// in). A SUPERSET of the realms that could hold the point, and never a verdict — in practice the
    /// children whose bounding sphere holds the point, which for non-overlapping siblings is at most
    /// one. An empty index answers with nothing.
    #[must_use]
    pub fn candidates(&self, point: LatticePos, tier: Tier) -> Vec<RealmId> {
        let mut out = Vec::new();
        self.candidates_into(point, tier, &mut out);
        out
    }

    /// [`candidates`](Self::candidates) into a caller's buffer: appended, then the whole buffer
    /// sorted and de-duplicated.
    pub fn candidates_into(&self, point: LatticePos, tier: Tier, out: &mut Vec<RealmId>) {
        self.candidates_segment(point, point, tier, out);
    }

    /// The realms whose bounding sphere MEETS the segment `p0 → p1`, appended to `out`, which is then
    /// sorted and de-duplicated. A segment of zero length is the point query.
    ///
    /// WHY THIS EXISTS. A point query's answer is trusted as a positive "not here". A subject that
    /// travels THROUGH a child within one tick has neither endpoint inside it, so the child would be
    /// skipped before anything could ask whether the motion crossed it — the swept verdict would be
    /// correct and unreachable. The same query serves a looker's predictive lead at warp: a line of
    /// hundreds of billions of metres, almost never axis-aligned. ★ THE LONG LEAD (measured on the
    /// fifth flight, 2026-09-03) cost the grid nine seconds a tick when it stepped that line one cell
    /// at a time; the tree descends only the boxes the line passes, so the cost is the tree's depth
    /// plus the answers, never the line's length. There is no abstain: every span is answered.
    pub fn candidates_segment(
        &self,
        p0: LatticePos,
        p1: LatticePos,
        tier: Tier,
        out: &mut Vec<RealmId>,
    ) {
        let a = origin_m(p0, tier);
        let span = p1.separation(p0, tier).metres();
        let b = a + span;
        let pad = slack(a.abs().max(b.abs()).max_element(), tier);
        let sweep = Sweep {
            a: a.to_array(),
            b: b.to_array(),
            pad,
            p0,
            span,
            tier,
        };
        out.extend(
            self.tree
                .locate_with_selection_function(sweep)
                .map(|leaf| leaf.realm),
        );
        out.sort_unstable();
        out.dedup();
    }

    /// Does this index answer for `realm` at all?
    ///
    /// THE DIFFERENCE BETWEEN A MISS AND IGNORANCE, and the whole safety of skipping. A realm the index
    /// holds an entry for, absent from a query's answer, is positively NOT near the point. A realm the
    /// index never indexed (an ancestor, the realm itself, anything that moves) is simply unknown here,
    /// and a caller must evaluate it. Without this, an empty answer would look the same in both cases.
    /// ★ THE CHILDREN WITHIN `radius_m` OF `point` (2026-09-05, the galaxy wedge): every indexed
    /// child whose padded sphere comes within `radius_m` of the point — a superset the caller
    /// tests exactly, never a scan of the roster. The governed ceiling asks this with the approach
    /// horizon (`v × tau`): a child farther than that cannot lower the ceiling, so it is never
    /// visited. Example: a walker in the galaxy at 3e8 m/s with a two-second horizon asks for
    /// the star systems within 6e8 m — a handful, never all 279,380. Sorted, deduplicated,
    /// appended to `out`.
    pub fn candidates_within(
        &self,
        point: LatticePos,
        radius_m: f64,
        tier: Tier,
        out: &mut Vec<RealmId>,
    ) {
        let p = origin_m(point, tier);
        let pad = slack(p.abs().max_element() + radius_m, tier);
        let within = Within {
            p: p.to_array(),
            radius_m,
            pad,
            point,
            tier,
        };
        out.extend(
            self.tree
                .locate_with_selection_function(within)
                .map(|leaf| leaf.realm),
        );
        out.sort_unstable();
        out.dedup();
    }

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

    /// How many LEAVES the tree holds — the memory instrument. At most [`ChildIndex::indexed_len`]:
    /// a child with no extent is counted and answered for, but has no leaf.
    #[must_use]
    pub fn stored_len(&self) -> usize {
        self.tree.size()
    }

    /// ★ ONE CHILD ARRIVES (the ruler switch, slice 2; SL9): insert it without rebuilding the tree.
    /// An already-indexed realm is re-inserted after a remove, so the two cannot double-count.
    ///
    /// Example: the galaxy holds 233,220 star systems in its index. A hull arrives from one of them.
    /// The hull's leaf goes down one path of the tree; nothing else moves.
    pub fn insert(&mut self, child: &IndexedChild, tier: Tier) {
        if self.indexed.contains(&child.realm) {
            self.remove(child.realm);
        }
        if let Some(leaf) = leaf_of(child, tier) {
            self.tree.insert(leaf.clone());
            self.leaves.insert(child.realm, leaf);
        }
        self.children += 1;
        self.indexed.insert(child.realm);
    }

    /// ★ ONE CHILD LEAVES (the ruler switch, slice 2): remove its leaf. A realm the index never held
    /// is a no-op, so a release can be re-driven.
    pub fn remove(&mut self, realm: RealmId) {
        if !self.indexed.remove(&realm) {
            return;
        }
        self.children -= 1;
        // ONLY THE CHILD'S OWN LEAF (SL9): the exact leaf it was inserted with, never a walk.
        if let Some(leaf) = self.leaves.remove(&realm) {
            self.tree.remove(&leaf);
        }
    }
}

/// The selection the tree descends by: a PARENT box is opened when the padded line meets it (the
/// slab test, in f64 metres); a LEAF is answered when the child's exact sphere meets the line, on
/// the lattice. The leaf test never sees the leaf's box, so the padding can only widen the descent,
/// never the answer.
struct Sweep {
    a: [f64; 3],
    b: [f64; 3],
    pad: f64,
    p0: LatticePos,
    span: DVec3,
    tier: Tier,
}

impl SelectionFunction<Leaf> for Sweep {
    fn should_unpack_parent(&self, envelope: &AABB<[f64; 3]>) -> bool {
        segment_meets_box(self.a, self.b, envelope.lower(), envelope.upper(), self.pad)
    }

    fn should_unpack_leaf(&self, leaf: &Leaf) -> bool {
        sphere_meets_segment(leaf.centre, leaf.radius_m, self.p0, self.span, self.tier)
    }
}

/// A child's leaf: its exact sphere, and its box padded by the f64 rounding at its own magnitude.
/// `None` for a child with no extent — a zero radius covers no point, so there is nothing a query
/// could meet, and the built index's posture for it is "counted, never answered".
/// The radius query's selection: a box is opened when the point is within `radius + pad` of it,
/// a leaf is taken when the point is within `radius + the leaf's own radius` of its centre (the
/// centre's offset taken on the lattice — exact at every magnitude).
struct Within {
    p: [f64; 3],
    radius_m: f64,
    pad: f64,
    point: LatticePos,
    tier: Tier,
}

impl SelectionFunction<Leaf> for Within {
    fn should_unpack_parent(&self, envelope: &AABB<[f64; 3]>) -> bool {
        let (lo, hi) = (envelope.lower(), envelope.upper());
        let mut d2 = 0.0;
        for i in 0..3 {
            let gap = (lo[i] - self.p[i]).max(self.p[i] - hi[i]).max(0.0);
            d2 += gap * gap;
        }
        d2 <= (self.radius_m + self.pad) * (self.radius_m + self.pad)
    }

    fn should_unpack_leaf(&self, leaf: &Leaf) -> bool {
        let d = leaf
            .centre
            .separation(self.point, self.tier)
            .metres()
            .length();
        d <= self.radius_m + leaf.radius_m
    }
}

fn leaf_of(child: &IndexedChild, tier: Tier) -> Option<Leaf> {
    if child.radius_m <= 0.0 {
        return None;
    }
    let centre = origin_m(child.centre, tier);
    let reach = child.radius_m + slack(centre.abs().max_element() + child.radius_m, tier);
    Some(Leaf {
        realm: child.realm,
        centre: child.centre,
        radius_m: child.radius_m,
        lo: (centre - DVec3::splat(reach)).to_array(),
        hi: (centre + DVec3::splat(reach)).to_array(),
    })
}

/// A position as f64 metres from the lattice origin — the tree's own ruler, and the ONLY place a
/// position is flattened. The flatten rounds at galaxy magnitudes, which is why every box and every
/// query carries [`slack`], and why the deciding test runs on the lattice instead.
fn origin_m(p: LatticePos, tier: Tier) -> DVec3 {
    p.separation(LatticePos::ORIGIN, tier).metres()
}

/// The padding that makes the f64 box a superset: a few units in the last place at `magnitude_m`
/// (an f64 metre at 1e17 m is about 16 m wide) plus one lattice cell for the sub-cell remainder.
/// Derived from the magnitude, never chosen.
fn slack(magnitude_m: f64, tier: Tier) -> f64 {
    magnitude_m * f64::EPSILON * 8.0 + tier.cell_edge_m()
}

/// Does the sphere (`centre`, `radius_m`) meet the segment `p0 → p0 + span`? The centre's offset from
/// `p0` is taken on the lattice, so this is exact at every magnitude.
fn sphere_meets_segment(
    centre: LatticePos,
    radius_m: f64,
    p0: LatticePos,
    span: DVec3,
    tier: Tier,
) -> bool {
    let c = centre.delta_m(p0, tier);
    let len2 = span.length_squared();
    let t = if len2 > 0.0 {
        (c.dot(span) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (c - span * t).length() <= radius_m
}

/// Does the segment `a → b` pass through the box `lo..=hi` widened by `pad` on every side? The slab
/// test, axis by axis; a segment parallel to an axis is a hit only if it lies inside that slab.
fn segment_meets_box(a: [f64; 3], b: [f64; 3], lo: [f64; 3], hi: [f64; 3], pad: f64) -> bool {
    let mut t_min = 0.0_f64;
    let mut t_max = 1.0_f64;
    for axis in 0..3 {
        let (low, high) = (lo[axis] - pad, hi[axis] + pad);
        let origin = a[axis];
        let dir = b[axis] - origin;
        if dir == 0.0 {
            if origin < low || origin > high {
                return false;
            }
            continue;
        }
        let (t0, t1) = ((low - origin) / dir, (high - origin) / dir);
        t_min = t_min.max(t0.min(t1));
        t_max = t_max.min(t0.max(t1));
        if t_min > t_max {
            return false;
        }
    }
    true
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

    fn at(x: f64, y: f64, z: f64) -> LatticePos {
        LatticePos::from_metres(DVec3::new(x, y, z), T)
    }

    // ---------------- THE SEGMENT QUERY (slice S5) ----------------

    /// ★ 2026-09-05: the radius query returns the children whose padded sphere comes within the
    /// radius of the point — near ones in, far ones out, a child whose sphere overlaps the edge in
    /// — and never a child the index does not hold.
    #[test]
    fn candidates_within_returns_the_children_the_radius_can_reach_and_no_others() {
        let tier = Tier::Fine;
        let at = |x: f64| LatticePos::from_metres(DVec3::new(x, 0.0, 0.0), tier);
        let child = |id: u64, x: f64, r: f64| IndexedChild {
            realm: RealmId::Planet(id),
            centre: at(x),
            radius_m: r,
        };
        let ix = ChildIndex::build(
            &[
                child(1, 100.0, 10.0),   // 90 m away: in
                child(2, 1_000.0, 10.0), // 990 m away: out at 500, in at 1 000
                child(3, 540.0, 50.0),   // its sphere's near edge at 490 m: in at 500
                child(4, -5_000.0, 1.0), // far behind: out
            ],
            tier,
        );
        let mut out = Vec::new();
        ix.candidates_within(at(0.0), 500.0, tier, &mut out);
        assert_eq!(out, vec![RealmId::Planet(1), RealmId::Planet(3)]);
        out.clear();
        ix.candidates_within(at(0.0), 1_000.0, tier, &mut out);
        assert_eq!(
            out,
            vec![RealmId::Planet(1), RealmId::Planet(2), RealmId::Planet(3)]
        );
        out.clear();
        ix.candidates_within(at(0.0), 1.0, tier, &mut out);
        assert!(out.is_empty(), "nothing within a metre");
    }

    #[test]
    fn a_child_the_subject_travels_straight_through_is_named_by_the_segment_and_missed_by_the_point()
     {
        // THE WHOLE REASON THE SEGMENT QUERY EXISTS. Without it the swept verdict would be correct
        // and unreachable: the child holds neither endpoint, so a point lookup answers "not here",
        // and that answer is trusted as positive information rather than as ignorance.
        let mut kids: Vec<IndexedChild> = (0..64)
            .map(|i| child(i, DVec3::new(i as f64 * 40.0, 0.0, 0.0), 4.0))
            .collect();
        // The one the subject flies straight through.
        kids.push(child(999, DVec3::new(-100.0, 0.0, 0.0), 4.0));
        let ix = ChildIndex::build(&kids, T);
        let a = at(-110.0, 0.0, 0.0);
        let b = at(-85.0, 0.0, 0.0);
        // NEITHER ENDPOINT IS INSIDE THE CHILD — a point lookup answers "not here", and that answer
        // is trusted as positive information.
        assert_eq!(ix.candidates(a, T), &[] as &[RealmId]);
        assert_eq!(ix.candidates(b, T), &[] as &[RealmId]);
        let mut out = Vec::new();
        ix.candidates_segment(a, b, T, &mut out);
        assert_eq!(out, vec![RealmId::Planet(999)]);
    }

    #[test]
    fn a_segment_of_zero_length_answers_exactly_what_the_point_answers() {
        // A stationary subject must pay today's lookup and get today's answer, or the whole
        // "no prior means no change" guarantee would stop at the index.
        let ix = ChildIndex::build(
            &[
                child(1, DVec3::new(0.0, 0.0, 0.0), 4.0),
                child(2, DVec3::new(40.0, 0.0, 0.0), 3.0),
            ],
            T,
        );
        for m in [0.0, 1.0, 3.5, 40.0, 41.0, 100.0] {
            let p = at(m, 0.0, 0.0);
            let mut out = Vec::new();
            ix.candidates_segment(p, p, T, &mut out);
            assert_eq!(out, ix.candidates(p, T), "at {m} m");
        }
    }

    #[test]
    fn a_span_of_any_width_is_answered_never_abstained() {
        // THE GRID ABSTAINED on a span wider than one cell per axis, and on a step of galaxy
        // magnitude. The tree answers both: the child on the line is named, the one off it is not.
        let ix = ChildIndex::build(
            &[
                child(1, DVec3::new(0.0, 0.0, 0.0), 4.0),
                child(2, DVec3::new(0.0, 50.0, 0.0), 4.0),
            ],
            T,
        );
        for v in [
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 100.0),
            DVec3::new(100.0, 0.0, 100.0),
        ] {
            let mut out = Vec::new();
            ix.candidates_segment(
                LatticePos::from_metres(-v, T),
                LatticePos::from_metres(v, T),
                T,
                &mut out,
            );
            assert_eq!(out, vec![RealmId::Planet(1)], "span {v:?}");
        }
        // A step across the whole lattice domain: answered, and correctly.
        let mut out = Vec::new();
        ix.candidates_segment(
            LatticePos::at(I64Vec3::splat(-CELL_DOMAIN_MAX), DVec3::ZERO),
            LatticePos::at(I64Vec3::splat(CELL_DOMAIN_MAX), DVec3::ZERO),
            T,
            &mut out,
        );
        assert_eq!(out, vec![RealmId::Planet(1)]);
    }

    #[test]
    fn the_segment_union_names_each_child_once_and_sorted() {
        // A child met by the line is returned once, and the buffer the caller hands in is sorted as
        // a whole — a grown tree and a built one must answer byte-identically.
        let mut kids = vec![
            child(2, DVec3::new(7.0, 0.0, 0.0), 4.0),
            child(1, DVec3::new(0.0, 0.0, 0.0), 4.0),
        ];
        kids.extend((10..40).map(|i| child(i, DVec3::new(i as f64 * 40.0, 0.0, 0.0), 4.0)));
        let ix = ChildIndex::build(&kids, T);
        let mut out = vec![RealmId::Planet(2)];
        ix.candidates_segment(at(-1.0, 0.0, 0.0), at(8.0, 0.0, 0.0), T, &mut out);
        assert_eq!(out, vec![RealmId::Planet(1), RealmId::Planet(2)]);
    }

    #[test]
    fn an_empty_index_answers_nothing_and_holds_nothing() {
        let ix = ChildIndex::build(&[], T);
        assert_eq!(ix.indexed_len(), 0);
        assert_eq!(ix.stored_len(), 0);
        assert_eq!(ix.candidates(at(0.0, 0.0, 0.0), T), &[] as &[RealmId]);
        assert_eq!(ix, ChildIndex::default());
    }

    #[test]
    fn children_with_no_extent_are_counted_and_never_answered() {
        // A zero radius means the child covers no point, so there is nothing a query could hit. The
        // child is answered for (a miss is a miss, not ignorance) but has no leaf.
        let ix = ChildIndex::build(&[child(1, DVec3::new(5.0, 0.0, 0.0), 0.0)], T);
        assert_eq!(ix.indexed_len(), 1);
        assert_eq!(ix.stored_len(), 0);
        assert!(ix.answers_for(RealmId::Planet(1)));
        assert_eq!(ix.candidates(at(5.0, 0.0, 0.0), T), &[] as &[RealmId]);
    }

    #[test]
    fn a_point_inside_a_child_finds_that_child_and_a_point_outside_finds_nothing() {
        let ix = ChildIndex::build(&[child(7, DVec3::new(1000.0, 0.0, 0.0), 10.0)], T);
        assert_eq!(
            ix.candidates(at(1000.0, 0.0, 0.0), T),
            &[RealmId::Planet(7)]
        );
        assert_eq!(
            ix.candidates(at(1009.0, 0.0, 0.0), T),
            &[RealmId::Planet(7)]
        );
        // Inside the box's corner, outside the sphere: the second stage says no.
        assert_eq!(ix.candidates(at(1008.0, 8.0, 0.0), T), &[] as &[RealmId]);
        assert_eq!(ix.candidates(at(-1.0e6, 0.0, 0.0), T), &[] as &[RealmId]);
    }

    #[test]
    fn two_children_either_side_of_the_origin_answer_only_their_own_side() {
        let ix = ChildIndex::build(
            &[
                child(1, DVec3::new(-12.0, 0.0, 0.0), 4.0),
                child(2, DVec3::new(12.0, 0.0, 0.0), 4.0),
            ],
            T,
        );
        assert_eq!(ix.candidates(at(-12.0, 0.0, 0.0), T), &[RealmId::Planet(1)]);
        assert_eq!(ix.candidates(at(12.0, 0.0, 0.0), T), &[RealmId::Planet(2)]);
    }

    #[test]
    fn a_wide_field_answers_with_one_while_holding_many() {
        // THE MEASUREMENT SL9 EXISTS FOR, on the shape a galaxy states: many equal children on a
        // lattice, each far smaller than the gap between them. The index HOLDS them all and ANSWERS
        // with exactly one — the answer size does not grow with the child count, which is the whole
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
        assert_eq!(
            ix.stored_len(),
            ix.indexed_len(),
            "one leaf per child with an extent"
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
        // And a lead across the whole field names exactly the children on its line.
        let mut along = Vec::new();
        ix.candidates_segment(at(-500.0, 0.0, 0.0), at(30_000.0, 0.0, 0.0), T, &mut along);
        assert_eq!(
            along.len(),
            SIDE as usize,
            "one child per lattice column on the x axis"
        );
    }

    #[test]
    fn a_child_is_still_found_at_a_magnitude_where_flattening_would_lose_it() {
        // ★ SLICE S4's SERVER HALF, and the failure it prevents is a SKIP, not a wobble.
        //
        // At the star placement radius a position is about 1.53e18 lattice cells, whose f64 spacing is
        // 256 cells — a quarter of a metre, and over a hundred times the whole containment band. The
        // tree's boxes are f64, so they are padded by that rounding; the deciding sphere test runs on
        // the lattice. This index's empty answer is TRUSTED: the caller reads it as a positive "not
        // here" and SKIPS the child. So a lost key would not blur a boundary — it would make a realm
        // unenterable. Every point inside the child must still find it.
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
        // Steps that are NOT multiples of the rounding quantum: a whole-metre step is exact even with
        // the defect present, so it would prove nothing.
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
    fn overlapping_children_are_all_named_and_the_answer_is_the_overlap_not_the_field() {
        // A star system's planets sweep bounds comparable to the system: at the centre they all
        // overlap, and the query names them all — that is the honest answer, and it is affordable
        // only because such a realm has few children. Away from the overlap the answer shrinks.
        let children: Vec<IndexedChild> = (0..8)
            .map(|n| child(n, DVec3::new(n as f64, 0.0, 0.0), 100.0))
            .collect();
        let ix = ChildIndex::build(&children, T);
        assert_eq!(ix.candidates(at(0.0, 0.0, 0.0), T).len(), children.len());
        // 104 m from the last child's centre and 111 m from the first's: only the last four reach.
        assert_eq!(ix.candidates(at(104.0, 0.0, 0.0), T).len(), 4);
    }

    #[test]
    fn the_slab_test_decides_a_miss_on_the_axis_the_line_crosses_last() {
        // A diagonal line whose x slab and y slab do not overlap in t: a miss decided by the
        // running t bounds, not by an axis-parallel arm.
        let a = [0.0, 0.0, 0.0];
        let b = [100.0, 100.0, 0.0];
        assert!(!segment_meets_box(
            a,
            b,
            [50.0, 0.0, -1.0],
            [60.0, 5.0, 1.0],
            0.0
        ));
        assert!(segment_meets_box(
            a,
            b,
            [50.0, 45.0, -1.0],
            [60.0, 55.0, 1.0],
            0.0
        ));
        // A line parallel to z, inside the x/y slabs, decided by the z slab alone.
        assert!(segment_meets_box(
            [55.0, 50.0, -10.0],
            [55.0, 50.0, 10.0],
            [50.0, 45.0, -1.0],
            [60.0, 55.0, 1.0],
            0.0
        ));
        assert!(!segment_meets_box(
            [55.0, 70.0, -10.0],
            [55.0, 70.0, 10.0],
            [50.0, 45.0, -1.0],
            [60.0, 55.0, 1.0],
            0.0
        ));
    }
}

#[cfg(test)]
mod mutator_tests {
    use super::*;
    use crate::pose::{LatticePos, Tier};
    use glam::DVec3;

    fn child(realm: RealmId, x_m: f64, r_m: f64) -> IndexedChild {
        IndexedChild {
            realm,
            centre: LatticePos::from_metres(DVec3::new(x_m, 0.0, 0.0), Tier::Fine),
            radius_m: r_m,
        }
    }

    #[test]
    fn an_inserted_child_answers_like_a_built_one_and_a_removed_one_answers_nothing() {
        let a = child(RealmId::Planet(1), 0.0, 100.0);
        let b = child(RealmId::Planet(2), 5_000.0, 100.0);
        let built = ChildIndex::build(&[a, b], Tier::Fine);
        let mut grown = ChildIndex::build(&[a], Tier::Fine);
        assert_ne!(grown, built);
        grown.insert(&b, Tier::Fine);
        assert_eq!(grown, built, "an insert reaches the built index exactly");
        let at_b = LatticePos::from_metres(DVec3::new(5_000.0, 0.0, 0.0), Tier::Fine);
        assert_eq!(grown.candidates(at_b, Tier::Fine), &[RealmId::Planet(2)]);
        assert!(grown.answers_for(RealmId::Planet(2)));
        assert_eq!(grown.indexed_len(), 2);
        // Re-inserting the same child does not double-count it, in the count or in the tree.
        grown.insert(&b, Tier::Fine);
        assert_eq!((grown.indexed_len(), grown.stored_len()), (2, 2));
        assert_eq!(grown.candidates(at_b, Tier::Fine), &[RealmId::Planet(2)]);
        // Removing it empties its leaf and its answer; removing it again is a no-op.
        grown.remove(RealmId::Planet(2));
        assert!(grown.candidates(at_b, Tier::Fine).is_empty());
        assert!(!grown.answers_for(RealmId::Planet(2)));
        assert_eq!((grown.indexed_len(), grown.stored_len()), (1, 1));
        grown.remove(RealmId::Planet(2));
        assert_eq!(grown.indexed_len(), 1);
    }

    #[test]
    fn a_query_answers_with_the_children_whose_sphere_meets_it_never_with_a_neighbour() {
        // Three children close together (a 400 m one beside two 50 m ones): a point query inside
        // the first's sphere names it alone; a segment past the second names the second alone; a
        // lead line across all three names the two it actually touches.
        let a = child(RealmId::Planet(1), 100.0, 400.0);
        let b = child(RealmId::Planet(2), 700.0, 50.0);
        let c = IndexedChild {
            realm: RealmId::Planet(3),
            centre: LatticePos::from_metres(DVec3::new(300.0, 600.0, 0.0), Tier::Fine),
            radius_m: 50.0,
        };
        let index = ChildIndex::build(&[a, b, c], Tier::Fine);
        let at = |x: f64, y: f64| LatticePos::from_metres(DVec3::new(x, y, 0.0), Tier::Fine);
        assert_eq!(
            index.candidates(at(150.0, 0.0), Tier::Fine),
            vec![RealmId::Planet(1)]
        );
        // (300, 700) is 728 m from the first (radius 400), 100 m from the third (radius 50): outside all.
        assert!(index.candidates(at(300.0, 700.0), Tier::Fine).is_empty());
        let mut seg = Vec::new();
        index.candidates_segment(at(650.0, 0.0), at(760.0, 0.0), Tier::Fine, &mut seg);
        assert_eq!(seg, vec![RealmId::Planet(2)]);
        let mut along = Vec::new();
        index.candidates_segment(at(-600.0, 0.0), at(900.0, 0.0), Tier::Fine, &mut along);
        assert_eq!(
            along,
            vec![RealmId::Planet(1), RealmId::Planet(2)],
            "the third is 600 m off the line"
        );
    }

    #[test]
    fn a_lead_of_a_hundred_billion_metres_names_the_child_on_the_line_and_leaves_the_one_off_it() {
        // THE LONG LEAD: a hull at warp leads by a line of hundreds of billions of metres. Two
        // children, one on the line and one off it; the answer names the one on the line and
        // finishes at once.
        let on = child(RealmId::Planet(1), 5.0e10, 100.0);
        let off = IndexedChild {
            realm: RealmId::Planet(2),
            centre: LatticePos::from_metres(DVec3::new(5.0e10, 9.0e5, 0.0), Tier::Fine),
            radius_m: 100.0,
        };
        let index = ChildIndex::build(&[on, off], Tier::Fine);
        let p0 = LatticePos::from_metres(DVec3::ZERO, Tier::Fine);
        let p1 = LatticePos::from_metres(DVec3::new(1.0e11, 0.0, 0.0), Tier::Fine);
        let mut out = Vec::new();
        index.candidates_segment(p0, p1, Tier::Fine, &mut out);
        assert_eq!(out, vec![RealmId::Planet(1)]);
        // A line parallel to an axis but outside every box's slab is a miss.
        let above0 = LatticePos::from_metres(DVec3::new(0.0, 5.0e5, 0.0), Tier::Fine);
        let above1 = LatticePos::from_metres(DVec3::new(1.0e11, 5.0e5, 0.0), Tier::Fine);
        let mut none = Vec::new();
        index.candidates_segment(above0, above1, Tier::Fine, &mut none);
        assert!(none.is_empty(), "{none:?}");
    }

    #[test]
    fn a_zero_radius_child_is_inserted_without_a_leaf_and_removed_without_one() {
        let mut flat = ChildIndex::default();
        flat.insert(&child(RealmId::Planet(3), 0.0, 0.0), Tier::Fine);
        assert_eq!((flat.indexed_len(), flat.stored_len()), (1, 0));
        assert!(flat.candidates(LatticePos::ORIGIN, Tier::Fine).is_empty());
        flat.remove(RealmId::Planet(3));
        assert_eq!((flat.indexed_len(), flat.stored_len()), (0, 0));
        // And a child with an extent, into the empty index, answers at once.
        let mut one = ChildIndex::default();
        one.insert(&child(RealmId::Planet(9), 0.0, 100.0), Tier::Fine);
        assert_eq!(
            one.candidates(LatticePos::ORIGIN, Tier::Fine),
            &[RealmId::Planet(9)]
        );
    }
}
