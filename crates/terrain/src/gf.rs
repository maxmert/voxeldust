//! ★ `Gf` — THE FENCED FLOAT (the voxel foundation, slice 5; SL10 clause 4, layer 1 of the fence).
//!
//! ★ **THE SHAPE IS NO LONGER A FLOAT** (ruling F7, 2026-09-12). The recipe computes the world's static
//! shape in [`vd_recipe`]'s integer kernels, and this type has exactly TWO callers left:
//!
//! 1. **THE DRAW of a body from its seed** (`body::BodyDefinition::from_seed`): the shares, caps and
//!    weights the seed states, rounded ONCE into the body's integer charter, on the CPU, once per
//!    body. A kernel never sees them, and a GPU never runs them. (Ruling V13 L12 owes the next step:
//!    the realm stores the charter and states it, so even the draw runs once in the world's life.)
//! 2. **THE DOORS between the recipe and metres** (`units`): four functions, each one IEEE-exact
//!    multiply or divide by a power of two, for a host outside the recipe — a client's mesh buffer, a
//!    picture's caption, a float query about an arbitrary point.
//!
//! Every float either of them computes with is a `Gf`. The type offers exactly the operations
//! IEEE-754 fixes on every target — add, subtract, multiply, divide, negate, square root, floor,
//! truncate, absolute value, comparison — and NOTHING else. The inner number is PRIVATE: there is no
//! `From<Gf> for f64`, no `Deref`, no public field, so code that wants a sine cannot reach the number
//! to call one on it. Values enter as integers (`from_i64`), as bit patterns (`from_bits`), or as
//! plain floats the `vd-seed` leaf computed under the same fence (`from_f64`); they leave as integers
//! or as bit patterns, which is what a digest and a record want.
//!
//! What is deliberately ABSENT: `min` and `max` (the standard library documents them
//! non-deterministic on `+0.0` against `−0.0` — write the comparison, [`Gf::lesser`] and
//! [`Gf::greater`] do), `mul_add` (a fused multiply-add rounds once where two chips round twice),
//! every transcendental, `powi` (a `libm` call on some targets), and `f32` anywhere.
//!
//! **The compile-fail controls** (SL1 clause 5: a structural fence has an observed-failing control):
//!
//! ```compile_fail
//! let x = vd_terrain::Gf::from_i64(2);
//! let inner = x.0; // the field is private: a `Gf` never gives its number away
//! ```
//!
//! ```compile_fail
//! let x = vd_terrain::Gf::from_i64(2);
//! let s = x.sin(); // no such method: the fence offers no transcendental
//! ```
//!
//! ```compile_fail
//! let x = vd_terrain::Gf::from_i64(2);
//! let y: f64 = x.into(); // no `Into<f64>`: nothing escapes to the primitive
//! ```
//!
//! ```compile_fail
//! let x = vd_terrain::Gf::from_i64(2);
//! let m = x.mul_add(x, x); // no fused multiply-add
//! ```
//!
//! **Example.** The home planet's draw states its relief as a share of its radius, as a `Gf`. A
//! contributor who writes `relief * latitude.sin()` gets a compile error, not a moon whose hills
//! differ by a metre between the Mac and the pod.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A float under the fence. `repr(transparent)` only for the C edge a foreign client links.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Gf(f64);

impl Gf {
    pub const ZERO: Gf = Gf(0.0);
    pub const ONE: Gf = Gf(1.0);
    pub const HALF: Gf = Gf(0.5);
    pub const TWO: Gf = Gf(2.0);

    /// A whole number.
    #[must_use]
    pub fn from_i64(v: i64) -> Gf {
        Gf(v as f64)
    }

    /// A whole number that fits exactly.
    #[must_use]
    pub fn from_i32(v: i32) -> Gf {
        Gf(f64::from(v))
    }

    /// A plain float the `vd-seed` leaf computed under the same fence (the bend, the ladder): the
    /// one door in for a value that is not an integer. Never call it on a value any other code made.
    #[must_use]
    pub const fn from_f64(v: f64) -> Gf {
        Gf(v)
    }

    /// A bit pattern (a constant stated as bits is a constant stated exactly).
    #[must_use]
    pub const fn from_bits(bits: u64) -> Gf {
        Gf(f64::from_bits(bits))
    }

    /// THE EXIT for a host outside the fence: the plain float, for a display path (a client's mesh
    /// buffer, `vd_client::chunks::geometry_of`) that is not under the fence. Nothing inside the
    /// crate calls it.
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        self.0
    }

    /// The bit pattern: what a digest folds and a golden set pins.
    #[must_use]
    pub const fn to_bits(self) -> u64 {
        self.0.to_bits()
    }

    /// The floor as a whole number, saturating at the integer range's ends (a `NaN` becomes 0, as
    /// the cast defines).
    #[must_use]
    pub fn to_i64_floor(self) -> i64 {
        self.0.floor() as i64
    }

    /// IEEE-exact on every target.
    #[must_use]
    pub fn sqrt(self) -> Gf {
        Gf(self.0.sqrt())
    }

    #[must_use]
    pub fn floor(self) -> Gf {
        Gf(self.0.floor())
    }

    #[must_use]
    pub fn trunc(self) -> Gf {
        Gf(self.0.trunc())
    }

    #[must_use]
    pub fn abs(self) -> Gf {
        Gf(self.0.abs())
    }

    /// The smaller of two, by the comparison `other < self` — fully determined, unlike `f64::min`.
    #[must_use]
    pub fn lesser(self, other: Gf) -> Gf {
        if other.0 < self.0 { other } else { self }
    }

    /// The larger of two, by the comparison `other > self`.
    #[must_use]
    pub fn greater(self, other: Gf) -> Gf {
        if other.0 > self.0 { other } else { self }
    }

    /// `lo` where below it, `hi` where above it, itself between: three comparisons.
    #[must_use]
    pub fn clamp(self, lo: Gf, hi: Gf) -> Gf {
        if self.0 < lo.0 {
            lo
        } else if self.0 > hi.0 {
            hi
        } else {
            self
        }
    }

    /// Whether the number is finite (a division by zero somewhere upstream is a defect, and a test
    /// asks this so the defect is named rather than folded into a digest).
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

impl Add for Gf {
    type Output = Gf;
    fn add(self, rhs: Gf) -> Gf {
        Gf(self.0 + rhs.0)
    }
}

impl Sub for Gf {
    type Output = Gf;
    fn sub(self, rhs: Gf) -> Gf {
        Gf(self.0 - rhs.0)
    }
}

impl Mul for Gf {
    type Output = Gf;
    fn mul(self, rhs: Gf) -> Gf {
        Gf(self.0 * rhs.0)
    }
}

impl Div for Gf {
    type Output = Gf;
    fn div(self, rhs: Gf) -> Gf {
        Gf(self.0 / rhs.0)
    }
}

impl Neg for Gf {
    type Output = Gf;
    fn neg(self) -> Gf {
        Gf(-self.0)
    }
}

impl AddAssign for Gf {
    fn add_assign(&mut self, rhs: Gf) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Gf {
    fn sub_assign(&mut self, rhs: Gf) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for Gf {
    fn mul_assign(&mut self, rhs: Gf) {
        self.0 *= rhs.0;
    }
}

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE (ruling F7's rule is about the SHIPPED path, not the measurement): a test
    //! states the exact quotient a reciprocal stands for, and a fixture picks its sample columns with a
    //! remainder. Neither runs in a kernel.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states an exact quotient or picks a sample column; never a kernel's path"
    )]
    use super::*;

    #[test]
    fn the_fence_offers_exactly_the_exact_operations_and_they_behave() {
        let two = Gf::from_i64(2);
        let three = Gf::from_i32(3);
        assert_eq!((two + three).to_i64_floor(), 5);
        assert_eq!((three - two).to_i64_floor(), 1);
        assert_eq!((two * three).to_i64_floor(), 6);
        assert_eq!((three / two).to_bits(), Gf::from_f64(1.5).to_bits());
        assert_eq!((-two).to_i64_floor(), -2);
        assert_eq!(Gf::from_i64(9).sqrt().to_i64_floor(), 3);
        assert_eq!(Gf::from_f64(-1.5).floor().to_i64_floor(), -2);
        assert_eq!(Gf::from_f64(-1.5).trunc().to_i64_floor(), -1);
        assert_eq!(
            Gf::from_f64(-1.5).abs().to_bits(),
            Gf::from_f64(1.5).to_bits()
        );
        assert_eq!(Gf::from_bits(Gf::ONE.to_bits()), Gf::ONE);
        assert_eq!(Gf::HALF + Gf::HALF, Gf::ONE);
        assert_eq!(Gf::TWO, two);
        assert_eq!(Gf::default(), Gf::ZERO);
        let mut acc = Gf::ONE;
        acc += Gf::ONE;
        acc -= Gf::HALF;
        acc *= Gf::TWO;
        assert_eq!(acc, Gf::from_i64(3));
        assert!(two < three);
        assert!(Gf::ONE.is_finite());
        assert!(!(Gf::ONE / Gf::ZERO).is_finite());
        assert_eq!(
            Gf::from_f64(f64::NAN).to_i64_floor(),
            0,
            "the cast defines a NaN as 0"
        );
        assert_eq!(Gf::from_f64(1e300).to_i64_floor(), i64::MAX, "saturating");
    }

    #[test]
    fn lesser_greater_and_clamp_are_comparisons_and_keep_the_left_operand_on_a_tie() {
        let pz = Gf::from_f64(0.0);
        let nz = Gf::from_f64(-0.0);
        // The whole reason min/max are banned: on a tie the LEFT operand is returned, always.
        assert_eq!(pz.lesser(nz).to_bits(), pz.to_bits());
        assert_eq!(nz.lesser(pz).to_bits(), nz.to_bits());
        assert_eq!(pz.greater(nz).to_bits(), pz.to_bits());
        assert_eq!(Gf::ONE.lesser(Gf::TWO), Gf::ONE);
        assert_eq!(Gf::TWO.lesser(Gf::ONE), Gf::ONE);
        assert_eq!(Gf::ONE.greater(Gf::TWO), Gf::TWO);
        assert_eq!(Gf::TWO.greater(Gf::ONE), Gf::TWO);
        assert_eq!(Gf::from_i64(-5).clamp(-Gf::ONE, Gf::ONE), -Gf::ONE);
        assert_eq!(Gf::from_i64(5).clamp(-Gf::ONE, Gf::ONE), Gf::ONE);
        assert_eq!(Gf::HALF.clamp(-Gf::ONE, Gf::ONE), Gf::HALF);
    }
}
