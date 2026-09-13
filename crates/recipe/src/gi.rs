//! THE FENCED INTEGER — the only arithmetic type the recipe's kernels use.
//!
//! A 64-bit word with a fixed number of fraction bits the caller keeps in mind (the crate's formats
//! are documented at each kernel). Its operators are exactly the ones that give one answer on every
//! host: add, subtract and multiply WRAP (the GPU wraps; a debug Rust build would panic on the same
//! overflow, so the primitive is touched with `wrapping_*` here and nowhere else); a shift MASKS its
//! amount to the word (WGSL masks, SPIR-V calls a wider shift poison, Rust panics); the bit
//! operations and the comparisons are the same everywhere. What the type does NOT offer, on
//! purpose: division and remainder (WGSL defines `x / 0 == x`, SPIR-V leaves it undefined, Rust
//! panics; the recipe has no division — a reciprocal is computed once and multiplied), negation and
//! absolute value (the most negative word has no negation; the recipe's formats keep every value
//! far from it, and a kernel that needs a magnitude asks for [`Gi::unsigned_abs`], which cannot fail).
//!
//! ★ **THE TWO ROUNDINGS, AND WHICH KERNEL USES WHICH.** A fixed-point multiply must drop the extra
//! fraction bits, and this type offers two ways to drop them. They round DIFFERENTLY on a negative
//! value, and both are fully determined on every host:
//!
//! 1. [`Gi::mul_shr`] — the two-word product — TRUNCATES TOWARD ZERO. It computes the magnitudes'
//!    product, shifts it, and puts the sign back, so `−7 × 5 >> 2` reads `−8` as `−8`, not `−9`.
//! 2. `>>` — the arithmetic shift — FLOORS TOWARD −∞, as the hardware does: `−31 >> 2` is `−8`.
//!
//! **The rule for a kernel.** Use [`Gi::mul_shr`] where the product of two SIGNED words needs the full
//! width before the shift (the bend's Horner steps, the direction times a radius, a reciprocal
//! multiply): a plain `*` would wrap. Use `*` and `>>` where one product already fits a word (the
//! noise's fade and blend, the octave's amplitude, the trilinear weights) and where the value being
//! shifted is a LENGTH the recipe wants floored (the gap's one shift by `rung + the length's bits`).
//! Each kernel states its own choice at its own doc comment, so a reader never has to guess which
//! rounding a line carries; a GPU and a CPU agree on both, which is why either is lawful.
//!
//! ```compile_fail
//! let a = vd_recipe::Gi::new(6);
//! let _ = a / a;
//! ```
//! ```compile_fail
//! let a = vd_recipe::Gi::new(6);
//! let _ = a % a;
//! ```
//! ```compile_fail
//! let a = vd_recipe::Gi::new(6);
//! let _ = -a;
//! ```
//! ```compile_fail
//! let a = vd_recipe::Gi::new(6);
//! let _ = a.abs();
//! ```

use core::ops::{Add, AddAssign, BitAnd, BitOr, BitXor, Mul, Shl, Shr, Sub, SubAssign};

/// The fenced 64-bit integer. The field is private: every operation goes through the impls below.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Gi(i64);

impl Gi {
    /// Zero.
    pub const ZERO: Gi = Gi(0);
    /// One, as a whole number (a kernel scales it to its format with a shift).
    pub const ONE: Gi = Gi(1);

    /// A word from a plain integer.
    #[must_use]
    pub const fn new(v: i64) -> Gi {
        Gi(v)
    }

    /// The word.
    #[must_use]
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// The magnitude as an unsigned word — defined for every value, the most negative included.
    #[must_use]
    pub const fn unsigned_abs(self) -> u64 {
        self.0.unsigned_abs()
    }

    /// Whether the word is negative.
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// `self × other >> shift` through the two-word product: the recipe's multiply for formats whose
    /// product passes 63 bits. The magnitudes are multiplied and shifted and the sign is put back, so
    /// the result is TRUNCATED TOWARD ZERO — not floored, as `>>` is (the module's own doc states the
    /// rule for choosing between the two). `shift` is masked to `1..=127` as
    /// [`crate::wide::shr_wide`] does.
    #[must_use]
    pub fn mul_shr(self, other: Gi, shift: u32) -> Gi {
        let (hi, lo) = crate::wide::mul_wide(self.unsigned_abs(), other.unsigned_abs());
        let m = crate::wide::shr_wide(hi, lo, shift);
        let negative = self.is_negative() != other.is_negative();
        Gi(sign_magnitude(m, negative))
    }
}

/// A magnitude with a sign, as a word: `0 − m` wraps for `m` past the positive range, which the
/// recipe's formats never produce (every product is shifted under 2⁶²).
const fn sign_magnitude(m: u64, negative: bool) -> i64 {
    let m = m as i64;
    if negative { 0i64.wrapping_sub(m) } else { m }
}

impl Add for Gi {
    type Output = Gi;
    fn add(self, rhs: Gi) -> Gi {
        Gi(self.0.wrapping_add(rhs.0))
    }
}

impl AddAssign for Gi {
    fn add_assign(&mut self, rhs: Gi) {
        self.0 = self.0.wrapping_add(rhs.0);
    }
}

impl Sub for Gi {
    type Output = Gi;
    fn sub(self, rhs: Gi) -> Gi {
        Gi(self.0.wrapping_sub(rhs.0))
    }
}

impl SubAssign for Gi {
    fn sub_assign(&mut self, rhs: Gi) {
        self.0 = self.0.wrapping_sub(rhs.0);
    }
}

impl Mul for Gi {
    type Output = Gi;
    fn mul(self, rhs: Gi) -> Gi {
        Gi(self.0.wrapping_mul(rhs.0))
    }
}

impl Shr<u32> for Gi {
    type Output = Gi;
    /// An ARITHMETIC shift right (the sign fills), the amount masked to the word: the result FLOORS
    /// toward −∞, where [`Gi::mul_shr`] truncates toward zero.
    fn shr(self, rhs: u32) -> Gi {
        Gi(self.0.wrapping_shr(rhs))
    }
}

impl Shl<u32> for Gi {
    type Output = Gi;
    /// A shift left, the amount masked to the word; bits shifted past the top are lost (wrap).
    fn shl(self, rhs: u32) -> Gi {
        Gi(self.0.wrapping_shl(rhs))
    }
}

impl BitAnd for Gi {
    type Output = Gi;
    fn bitand(self, rhs: Gi) -> Gi {
        Gi(self.0 & rhs.0)
    }
}

impl BitOr for Gi {
    type Output = Gi;
    fn bitor(self, rhs: Gi) -> Gi {
        Gi(self.0 | rhs.0)
    }
}

impl BitXor for Gi {
    type Output = Gi;
    fn bitxor(self, rhs: Gi) -> Gi {
        Gi(self.0 ^ rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_operators_wrap_and_the_shifts_mask() {
        let max = Gi::new(i64::MAX);
        assert_eq!(max + Gi::ONE, Gi::new(i64::MIN));
        assert_eq!(Gi::new(i64::MIN) - Gi::ONE, max);
        assert_eq!(max * Gi::new(2), Gi::new(-2));
        let mut a = Gi::new(5);
        a += Gi::new(3);
        a -= Gi::ONE;
        assert_eq!(a, Gi::new(7));
        // A shift by 64 is a shift by 0; by 65, by 1; the right shift keeps the sign.
        assert_eq!(Gi::new(-8) >> 1, Gi::new(-4));
        assert_eq!(Gi::new(-8) >> 64, Gi::new(-8));
        assert_eq!(Gi::new(3) << 65, Gi::new(6));
        assert_eq!(Gi::new(1) << 63, Gi::new(i64::MIN));
        assert_eq!(Gi::new(0b1100) & Gi::new(0b1010), Gi::new(0b1000));
        assert_eq!(Gi::new(0b1100) | Gi::new(0b1010), Gi::new(0b1110));
        assert_eq!(Gi::new(0b1100) ^ Gi::new(0b1010), Gi::new(0b0110));
        assert_eq!(Gi::default(), Gi::ZERO);
        assert_eq!(Gi::new(-3).raw(), -3);
        assert!(Gi::new(-3).is_negative());
        assert!(!Gi::ZERO.is_negative());
        assert_eq!(Gi::new(i64::MIN).unsigned_abs(), 1u64 << 63);
    }

    /// ★ THE TWO ROUNDINGS, stated as a measurement: the same quotient, two answers, each fixed.
    #[test]
    fn the_product_truncates_toward_zero_and_the_shift_floors() {
        // −31/4 is −7.75. The shift floors it to −8; the two-word product truncates it to −7.
        assert_eq!(Gi::new(-31) >> 2, Gi::new(-8));
        assert_eq!(Gi::new(-31).mul_shr(Gi::ONE, 2), Gi::new(-7));
        // On a positive value the two agree.
        assert_eq!(Gi::new(31) >> 2, Gi::new(7));
        assert_eq!(Gi::new(31).mul_shr(Gi::ONE, 2), Gi::new(7));
    }

    #[test]
    fn the_two_word_multiply_truncates_toward_zero_with_the_sign_of_the_product() {
        // 3 × 5 at 2 fraction bits each: 15 / 4 = 3.75 → 3; the signs: (−3)(5) → −3, (−3)(−5) → 3.
        let three = Gi::new(3);
        let five = Gi::new(5);
        assert_eq!(three.mul_shr(five, 2), Gi::new(3));
        assert_eq!(Gi::new(-3).mul_shr(five, 2), Gi::new(-3));
        assert_eq!(three.mul_shr(Gi::new(-5), 2), Gi::new(-3));
        assert_eq!(Gi::new(-3).mul_shr(Gi::new(-5), 2), Gi::new(3));
        // A product past 64 bits: 2⁴⁰ × 2⁴⁰ >> 40 = 2⁴⁰.
        let big = Gi::new(1 << 40);
        assert_eq!(big.mul_shr(big, 40), big);
        // The shift's top of the range: (2⁶² × 2⁶²) >> 120 = 2⁴.
        let top = Gi::new(1 << 62);
        assert_eq!(top.mul_shr(top, 120), Gi::new(16));
    }

    #[test]
    fn a_magnitude_with_a_sign_is_a_word() {
        assert_eq!(sign_magnitude(7, false), 7);
        assert_eq!(sign_magnitude(7, true), -7);
        assert_eq!(sign_magnitude(0, true), 0);
    }
}
