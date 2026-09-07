//! ★ THE SEAM TABLE — how the six faces of the cube-sphere join along their twelve edges.
//!
//! Every face has four edges. Each directed edge (a face and one of its four sides) has exactly one
//! partner: the side of the neighbouring face it shares the cube edge with. The table records, for
//! every one of the 24 directed edges, the partner face, the partner side, and whether the position
//! along the edge runs the other way there. The whole table is GENERATED at compile time from the
//! face basis table, so it can never disagree with it, and it is PINNED by a committed digest so a
//! change to the basis is loud.
//!
//! The table depends on the cube net only — never on a body's size or on the rung — so an exhaustive
//! test on a small face proves it for every body (ruling V6 A1, `G-MAPPING-TABLE`).
//!
//! **Example.** The mesher gathers the apron for the chunk at the top edge of face `+X`. The table
//! says the cells beyond that edge live on face `+Y`, along its `+u` side, running the same way. The
//! gather reads them there, with the axes swapped, and no triangle is emitted into a hole.

use super::bend::{Axis, Face, axis_eq, axis_neg};

/// One of a face's four sides: the side where `i` is least (`UMinus`) or greatest (`UPlus`), and the
/// side where `j` is least (`VMinus`) or greatest (`VPlus`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Edge {
    UMinus = 0,
    UPlus = 1,
    VMinus = 2,
    VPlus = 3,
}

impl Edge {
    /// Every side, in table order.
    pub const ALL: [Edge; 4] = [Edge::UMinus, Edge::UPlus, Edge::VMinus, Edge::VPlus];

    /// The table row of this side.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Whether this side bounds the `i` axis (a `u` side) rather than the `j` axis (a `v` side).
    #[must_use]
    pub const fn is_u(self) -> bool {
        matches!(self, Edge::UMinus | Edge::UPlus)
    }

    /// Whether this side sits at the greatest index (`+1` in face position) rather than the least.
    #[must_use]
    pub const fn is_plus(self) -> bool {
        matches!(self, Edge::UPlus | Edge::VPlus)
    }
}

/// The partner of a directed edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeamRecord {
    /// The face across the cube edge.
    pub dst_face: Face,
    /// The side of that face which is the same cube edge.
    pub dst_edge: Edge,
    /// Whether the along-edge position runs the other way on the partner face.
    pub reversed: bool,
}

/// The outward axis across a face's side: `±u` for a `u` side, `±v` for a `v` side.
const fn side_axis(face: Face, edge: Edge) -> Axis {
    let b = face.basis();
    match edge {
        Edge::UMinus => axis_neg(b.u),
        Edge::UPlus => b.u,
        Edge::VMinus => axis_neg(b.v),
        Edge::VPlus => b.v,
    }
}

/// The along-edge axis of a face's side: `v` for a `u` side, `u` for a `v` side.
const fn along_axis(face: Face, edge: Edge) -> Axis {
    let b = face.basis();
    if edge.is_u() { b.v } else { b.u }
}

/// The side of `face` whose outward axis is `axis`, for an axis that IS one of its four (a
/// neighbour's normal always is). The fourth side is the `else`, so the function is total and has no
/// unreachable arm.
const fn side_toward(face: Face, axis: Axis) -> Edge {
    if axis_eq(side_axis(face, Edge::UMinus), axis) {
        Edge::UMinus
    } else if axis_eq(side_axis(face, Edge::UPlus), axis) {
        Edge::UPlus
    } else if axis_eq(side_axis(face, Edge::VMinus), axis) {
        Edge::VMinus
    } else {
        Edge::VPlus
    }
}

/// The partner of one directed edge, from the basis table alone.
const fn partner(face: Face, edge: Edge) -> SeamRecord {
    // The neighbour is the face whose outward normal is this side's outward axis.
    let out = side_axis(face, edge);
    let dst_face = Face::with_normal(out);
    // On the neighbour, our normal points back across the shared edge, so the neighbour's side is
    // the one whose outward axis is OUR normal.
    let dst_edge = side_toward(dst_face, face.basis().n);
    let reversed = axis_eq(
        along_axis(face, edge),
        axis_neg(along_axis(dst_face, dst_edge)),
    );
    SeamRecord {
        dst_face,
        dst_edge,
        reversed,
    }
}

const fn build() -> [[SeamRecord; 4]; 6] {
    let blank = SeamRecord {
        dst_face: Face::PosX,
        dst_edge: Edge::UMinus,
        reversed: false,
    };
    let mut table = [[blank; 4]; 6];
    let mut f = 0;
    while f < 6 {
        let mut e = 0;
        while e < 4 {
            table[f][e] = partner(Face::ALL[f], Edge::ALL[e]);
            e += 1;
        }
        f += 1;
    }
    table
}

/// THE SEAM TABLE: `SEAM[face][edge]` is the partner of that directed edge.
pub const SEAM: [[SeamRecord; 4]; 6] = build();

/// Whether any record of a table is reversed.
const fn any_reversed(table: &[[SeamRecord; 4]; 6]) -> bool {
    let mut f = 0;
    while f < 6 {
        let mut e = 0;
        while e < 4 {
            if table[f][e].reversed {
                return true;
            }
            e += 1;
        }
        f += 1;
    }
    false
}

// ★ WITH THIS BASIS NO SEAM IS REVERSED: every face's `u` and `v` are positive axes, so an along-edge
// axis is always positive and can never equal the negation of another. The flag stays in the record
// and in the crossing rule (the rule is general and is exercised with a synthetic record), and THIS
// assertion makes a future basis edit that introduces a reversal loud at compile time — which matters
// because a reversal would re-pair every seam on every saved planet (ruling V6, Format A).
const _: () = assert!(!any_reversed(&SEAM));

/// The partner of a directed edge.
#[must_use]
pub const fn partner_of(face: Face, edge: Edge) -> SeamRecord {
    SEAM[face as usize][edge.index()]
}

/// The digest of the table as bytes `[dst_face, dst_edge, reversed]` per record, in table order —
/// the committed pin.
#[must_use]
pub fn table_digest() -> u64 {
    let mut acc = crate::digest::FNV_OFFSET;
    for row in &SEAM {
        for rec in row {
            acc = crate::digest::fnv1a(
                acc,
                &[
                    rec.dst_face.index(),
                    rec.dst_edge.index() as u8,
                    u8::from(rec.reversed),
                ],
            );
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_directed_edge_has_a_partner_that_points_back() {
        // Involution: the partner's partner is the edge itself, with the same reversal.
        for f in Face::ALL {
            for e in Edge::ALL {
                let p = partner_of(f, e);
                let back = partner_of(p.dst_face, p.dst_edge);
                assert_eq!(
                    back.dst_face, f,
                    "{f:?}/{e:?} partner points back to the face"
                );
                assert_eq!(
                    back.dst_edge, e,
                    "{f:?}/{e:?} partner points back to the side"
                );
                assert_eq!(back.reversed, p.reversed, "reversal is symmetric");
                assert_ne!(p.dst_face, f, "a face never joins itself");
                // The along axes are parallel or antiparallel, never anything else.
                let a = along_axis(f, e);
                let b = along_axis(p.dst_face, p.dst_edge);
                let same = axis_eq(a, b);
                let opposite = axis_eq(a, axis_neg(b));
                assert_ne!(
                    same, opposite,
                    "along axes are collinear: exactly one holds"
                );
                assert_eq!(p.reversed, opposite);
            }
        }
    }

    #[test]
    fn every_cube_edge_is_named_exactly_twice() {
        // 24 directed edges pair into 12 cube edges: every record appears once as a source and once
        // as a destination.
        let mut hits = [[0u8; 4]; 6];
        for f in Face::ALL {
            for e in Edge::ALL {
                let p = partner_of(f, e);
                hits[p.dst_face as usize][p.dst_edge.index()] += 1;
            }
        }
        assert!(
            hits.iter().all(|row| row.iter().all(|&h| h == 1)),
            "{hits:?}"
        );
    }

    #[test]
    fn three_faces_meet_at_every_cube_corner() {
        // At a corner, two adjacent sides of a face lead to two different faces, and those two
        // faces are adjacent to each other across the third edge of the corner.
        for f in Face::ALL {
            for (eu, ev) in [
                (Edge::UMinus, Edge::VMinus),
                (Edge::UMinus, Edge::VPlus),
                (Edge::UPlus, Edge::VMinus),
                (Edge::UPlus, Edge::VPlus),
            ] {
                let a = partner_of(f, eu).dst_face;
                let b = partner_of(f, ev).dst_face;
                assert_ne!(a, b, "the two sides of a corner lead to two faces");
                let joined = Edge::ALL.iter().any(|&e| partner_of(a, e).dst_face == b);
                assert!(joined, "{a:?} and {b:?} share the corner's third edge");
            }
        }
    }

    #[test]
    fn two_hand_checked_records_and_the_committed_digest() {
        // +X's +u side (u = +Y) meets +Y; +Y's side toward +X is its v side at +1 (v = +X).
        assert_eq!(
            partner_of(Face::PosX, Edge::UPlus),
            SeamRecord {
                dst_face: Face::PosY,
                dst_edge: Edge::VPlus,
                // along +X/UPlus is v = +Z; along +Y/VPlus is u = +Z: same way.
                reversed: false,
            }
        );
        // +X's −v side (v = +Z, so −Z) meets −Z; −Z's side toward +X is its v side at +1 (v = +X).
        assert_eq!(
            partner_of(Face::PosX, Edge::VMinus),
            SeamRecord {
                dst_face: Face::NegZ,
                dst_edge: Edge::VPlus,
                // along +X/VMinus is u = +Y; along −Z/VPlus is u = +Y: same way.
                reversed: false,
            }
        );
        assert_eq!(
            table_digest(),
            SEAM_DIGEST,
            "the seam table changed; a changed basis re-pairs every seam on every saved planet"
        );
        assert!(Edge::UPlus.is_plus());
        assert!(!Edge::VMinus.is_plus());
        assert!(!Edge::VMinus.is_u());
    }

    /// The table's builders run at COMPILE time, which no coverage tool can see. Run them at runtime
    /// and demand the same table, so a change to the rule is caught by execution, not only by the pin.
    #[test]
    fn the_table_built_at_runtime_equals_the_compile_time_table() {
        assert_eq!(build(), SEAM);
        assert!(!any_reversed(&SEAM), "this basis reverses no seam");
        let mut flipped = SEAM;
        flipped[2][1].reversed = true;
        assert!(any_reversed(&flipped), "a reversed record is detected");
        for f in Face::ALL {
            for e in Edge::ALL {
                assert_eq!(partner(f, e), partner_of(f, e));
            }
        }
        assert_eq!(
            side_toward(Face::PosX, [0, -1, 0]),
            Edge::UMinus,
            "−Y is +X's −u side"
        );
        assert_eq!(side_toward(Face::PosX, [0, 1, 0]), Edge::UPlus);
        assert_eq!(side_toward(Face::PosX, [0, 0, -1]), Edge::VMinus);
        assert_eq!(side_toward(Face::PosX, [0, 0, 1]), Edge::VPlus);
    }

    /// THE COMMITTED PIN of the seam table. Computed once from the basis table and never changed:
    /// a mismatch means the face basis or the pairing rule moved, which re-addresses every planet.
    const SEAM_DIGEST: u64 = 5_642_201_002_378_215_521;
}
