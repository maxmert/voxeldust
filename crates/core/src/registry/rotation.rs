//! THE ROTATION TABLE — the 24 proper rotations of a cube as integer matrices, and the orientation
//! code a record carries (ruling B-4: six bits, the sixth zero and reserved; codes `0..=23`).
//!
//! The 24 are every signed permutation matrix of determinant +1, in ONE frozen order: the enumeration
//! order of `(axis permutation, sign pattern)`, kept by a pinned digest. Rotations compose and invert
//! exactly, so a rotated face mask, template or collider is exact — never a float, never a
//! transcendental.
//!
//! **Example.** A shipwright mirrors a wing built of wedges. Each wedge's mirror image is the same
//! wedge under another of the 24 rotations, and the closure test proves the table has it.

/// One rotation as an integer 3×3 matrix, row-major: `out = m · in`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rotation(pub [[i8; 3]; 3]);

/// A record's orientation code: `0..=23`, an index into [`ORIENTATIONS`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Orientation(u8);

/// The 24 proper rotations, in the frozen enumeration order.
pub const ORIENTATIONS: [Rotation; 24] = build();

const fn build() -> [Rotation; 24] {
    // The six axis permutations, in a fixed order, and the eight sign patterns; keep det = +1.
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut out = [Rotation([[0; 3]; 3]); 24];
    let mut n = 0;
    let mut p = 0;
    while p < 6 {
        let mut s = 0;
        while s < 8 {
            let signs = [
                if s & 1 == 0 { 1i8 } else { -1 },
                if s & 2 == 0 { 1i8 } else { -1 },
                if s & 4 == 0 { 1i8 } else { -1 },
            ];
            let mut m = [[0i8; 3]; 3];
            let mut r = 0;
            while r < 3 {
                m[r][PERMS[p][r]] = signs[r];
                r += 1;
            }
            if det(m) == 1 {
                out[n] = Rotation(m);
                n += 1;
            }
            s += 1;
        }
        p += 1;
    }
    out
}

/// The determinant of an integer 3×3 matrix.
const fn det(m: [[i8; 3]; 3]) -> i32 {
    let a = m[0][0] as i32 * (m[1][1] as i32 * m[2][2] as i32 - m[1][2] as i32 * m[2][1] as i32);
    let b = m[0][1] as i32 * (m[1][0] as i32 * m[2][2] as i32 - m[1][2] as i32 * m[2][0] as i32);
    let c = m[0][2] as i32 * (m[1][0] as i32 * m[2][1] as i32 - m[1][1] as i32 * m[2][0] as i32);
    a - b + c
}

impl Rotation {
    /// The identity.
    pub const IDENTITY: Rotation = Rotation([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);

    /// `self · other`: apply `other` first, then `self`.
    #[must_use]
    pub fn then(self, first: Rotation) -> Rotation {
        let a = self.0;
        let b = first.0;
        let mut m = [[0i8; 3]; 3];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
            }
        }
        Rotation(m)
    }

    /// The inverse: the transpose, because every rotation here is orthogonal.
    #[must_use]
    pub fn inverse(self) -> Rotation {
        let m = self.0;
        Rotation([
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ])
    }

    /// Rotate an integer vector.
    #[must_use]
    pub fn apply(self, v: [i32; 3]) -> [i32; 3] {
        let m = self.0;
        let mut out = [0i32; 3];
        for (r, o) in out.iter_mut().enumerate() {
            *o = i32::from(m[r][0]) * v[0] + i32::from(m[r][1]) * v[1] + i32::from(m[r][2]) * v[2];
        }
        out
    }

    /// Rotate a six-bit face mask (`+X −X +Y −Y +Z −Z` as bits 0..=5): the face a unit axis maps to.
    #[must_use]
    pub fn apply_faces(self, mask: u8) -> u8 {
        let mut out = 0u8;
        for bit in 0..6u8 {
            if mask & (1 << bit) != 0 {
                let axis = usize::from(bit / 2);
                let sign = if bit % 2 == 0 { 1 } else { -1 };
                let mut v = [0i32; 3];
                v[axis] = sign;
                let w = self.apply(v);
                let (axis2, sign2) = if w[0] != 0 {
                    (0u8, w[0])
                } else if w[1] != 0 {
                    (1, w[1])
                } else {
                    (2, w[2])
                };
                let bit2 = axis2 * 2 + u8::from(sign2 < 0);
                out |= 1 << bit2;
            }
        }
        out
    }

    /// This rotation's code in the frozen table; `None` for a matrix that is not a proper rotation of
    /// the cube (a reflection, a scale, a hand-typed mistake) — a refusal, never a wrong code.
    #[must_use]
    pub fn code(self) -> Option<Orientation> {
        ORIENTATIONS
            .iter()
            .position(|r| *r == self)
            .map(|i| Orientation(i as u8))
    }
}

impl Orientation {
    /// No rotation.
    pub const IDENTITY: Orientation = Orientation(0);
    /// The largest code.
    pub const MAX: u8 = 23;

    /// A code from its number; `None` above [`Orientation::MAX`] (a decoder refuses).
    #[must_use]
    pub const fn new(code: u8) -> Option<Orientation> {
        if code <= Orientation::MAX {
            Some(Orientation(code))
        } else {
            None
        }
    }

    /// The number.
    #[must_use]
    pub const fn code(self) -> u8 {
        self.0
    }

    /// The rotation.
    #[must_use]
    pub fn rotation(self) -> Rotation {
        ORIENTATIONS[usize::from(self.0)]
    }
}

/// The digest of the table, in table order — the committed pin of the numbering.
#[must_use]
pub fn table_digest() -> u64 {
    let mut acc = crate::digest::FNV_OFFSET;
    for r in &ORIENTATIONS {
        for row in &r.0 {
            acc = crate::digest::fnv1a(acc, &[row[0] as u8, row[1] as u8, row[2] as u8]);
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_twenty_four_rotations_form_a_group_with_identity_and_inverses() {
        assert_eq!(
            ORIENTATIONS[0],
            Rotation::IDENTITY,
            "code 0 is the identity"
        );
        assert_eq!(
            build(),
            ORIENTATIONS,
            "the compile-time table equals the runtime build"
        );
        let set: std::collections::BTreeSet<_> = ORIENTATIONS.iter().map(|r| r.0).collect();
        assert_eq!(set.len(), 24, "all distinct");
        for a in &ORIENTATIONS {
            assert_eq!(det(a.0), 1, "proper");
            assert!(set.contains(&a.inverse().0), "closed under inverse");
            assert_eq!(a.then(a.inverse()), Rotation::IDENTITY);
            assert_eq!(
                a.code().map(Orientation::rotation),
                Some(*a),
                "the code round-trips"
            );
            for b in &ORIENTATIONS {
                assert!(set.contains(&a.then(*b).0), "closed under composition");
            }
        }
    }

    #[test]
    fn a_quarter_turn_about_z_permutes_axes_and_faces_exactly() {
        let quarter = Rotation([[0, -1, 0], [1, 0, 0], [0, 0, 1]]);
        assert_eq!(quarter.apply([1, 0, 0]), [0, 1, 0], "+X goes to +Y");
        assert_eq!(quarter.apply([0, 1, 0]), [-1, 0, 0], "+Y goes to −X");
        assert_eq!(quarter.apply([0, 0, 1]), [0, 0, 1], "+Z stays");
        // +X face (bit 0) → +Y face (bit 2); −Y face (bit 3) → +X face (bit 0); +Z (bit 4) stays.
        assert_eq!(quarter.apply_faces(0b00_0001), 0b00_0100);
        assert_eq!(quarter.apply_faces(0b00_1000), 0b00_0001);
        assert_eq!(quarter.apply_faces(0b01_0000), 0b01_0000);
        assert_eq!(quarter.apply_faces(0), 0);
        let four = quarter.then(quarter).then(quarter).then(quarter);
        assert_eq!(
            four,
            Rotation::IDENTITY,
            "four quarter turns are the identity"
        );
        assert_ne!(quarter.code(), Some(Orientation::IDENTITY));
        assert!(quarter.code().is_some(), "a quarter turn is in the table");
        // A face that lands on a NEGATIVE axis: +Y (bit 2) goes to −X (bit 1).
        assert_eq!(quarter.apply_faces(0b00_0100), 0b00_0010);
    }

    #[test]
    fn codes_are_six_bits_with_the_top_reserved_and_the_table_is_pinned() {
        assert_eq!(Orientation::new(23).map(Orientation::code), Some(23));
        assert_eq!(
            Orientation::new(24),
            None,
            "a code past the table is refused"
        );
        assert_eq!(Orientation::IDENTITY.rotation(), Rotation::IDENTITY);
        assert_eq!(
            table_digest(),
            ROTATION_DIGEST,
            "the rotation numbering changed: every saved orientation code re-points"
        );
        // A reflection is not a rotation: it gets NO code, never a wrong one (the decode-to-Default
        // ban). A mesher that hands over a mirror matrix is refused.
        assert_eq!(Rotation([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).code(), None);
    }

    /// The plan's gate: every rotation of every face mask is exact. 24 rotations × 64 masks, checked
    /// by two laws that do not restate the implementation: a rotation and its inverse undo each other
    /// on masks, and rotating by a composition equals rotating twice. The bit count never changes.
    #[test]
    fn every_rotation_of_every_face_mask_is_exact() {
        let mut checked = 0u32;
        for a in &ORIENTATIONS {
            for mask in 0u8..64 {
                let turned = a.apply_faces(mask);
                assert_eq!(
                    turned.count_ones(),
                    mask.count_ones(),
                    "{a:?} on {mask:#08b}"
                );
                assert_eq!(
                    a.inverse().apply_faces(turned),
                    mask,
                    "{a:?} undone on {mask:#08b}"
                );
                checked += 1;
            }
            for b in &ORIENTATIONS {
                let m = 0b10_1101u8;
                assert_eq!(a.then(*b).apply_faces(m), a.apply_faces(b.apply_faces(m)));
            }
        }
        assert_eq!(checked, 24 * 64);
    }

    /// THE COMMITTED PIN of the rotation numbering.
    const ROTATION_DIGEST: u64 = 5_946_405_317_151_576_421;
}
