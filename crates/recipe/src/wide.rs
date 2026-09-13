//! THE TWO-WORD PRODUCT — a 64 × 64 → 128-bit multiply spelled out from 32-bit halves, and the
//! shift that brings a two-word number back to one.
//!
//! Neither WGSL nor naga's back ends have a widening multiply or an add-with-carry (the research of
//! 2026-09-12), and the GPU has no 128-bit word. So the recipe spells the product out from four
//! 32 × 32 → 64 products and carries, on every host alike — the same code on the CPU, where a
//! `u128` would exist, because one source is the point. MEASURED (bench part 3): a direction at 40
//! fraction bits with this product in four places costs 77 ns a column on one CPU core.

/// The two-word product `(hi, lo)` of two unsigned words.
#[must_use]
pub const fn mul_wide(a: u64, b: u64) -> (u64, u64) {
    let (a0, a1) = (a & 0xFFFF_FFFF, a >> 32);
    let (b0, b1) = (b & 0xFFFF_FFFF, b >> 32);
    // Each partial product of two 32-bit halves fits a word exactly.
    let p00 = a0.wrapping_mul(b0);
    let p01 = a0.wrapping_mul(b1);
    let p10 = a1.wrapping_mul(b0);
    let p11 = a1.wrapping_mul(b1);
    // The middle column: three 32-bit pieces sum under 2³⁴, no wrap.
    let mid = (p00 >> 32)
        .wrapping_add(p01 & 0xFFFF_FFFF)
        .wrapping_add(p10 & 0xFFFF_FFFF);
    let lo = (p00 & 0xFFFF_FFFF) | mid.wrapping_shl(32);
    let hi = p11
        .wrapping_add(p01 >> 32)
        .wrapping_add(p10 >> 32)
        .wrapping_add(mid >> 32);
    (hi, lo)
}

/// `(hi, lo) >> shift` as one word, `shift` masked to `1..=127` (a shift by 0 or by 128 reads as
/// 64: the high word). The caller keeps the result under a word by its format.
#[must_use]
pub const fn shr_wide(hi: u64, lo: u64, shift: u32) -> u64 {
    let s = shift & 127;
    let s = if s == 0 { 64 } else { s };
    if s >= 64 {
        hi >> (s - 64)
    } else {
        (lo >> s) | hi.wrapping_shl(64 - s)
    }
}

/// The two-word sum `(hi, lo) + (h, l)` with the carry between the words; a wrap past the high
/// word is lost, which the recipe's formats never reach.
#[must_use]
pub const fn add_wide(hi: u64, lo: u64, h: u64, l: u64) -> (u64, u64) {
    let (nl, carry) = lo.overflowing_add(l);
    (hi.wrapping_add(h).wrapping_add(carry as u64), nl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_word_product_is_the_wide_multiply() {
        let cases: [(u64, u64); 8] = [
            (0, 0),
            (1, u64::MAX),
            (u64::MAX, u64::MAX),
            (0xFFFF_FFFF, 0xFFFF_FFFF),
            (0x1_0000_0000, 0x1_0000_0000),
            (0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321),
            (1 << 63, 3),
            (0xDEAD_BEEF_CAFE_F00D, 0x0BAD_F00D_DEAD_BEEF),
        ];
        for (a, b) in cases {
            let exact = u128::from(a) * u128::from(b);
            let (hi, lo) = mul_wide(a, b);
            assert_eq!(u128::from(hi) << 64 | u128::from(lo), exact, "{a} × {b}");
        }
    }

    #[test]
    fn the_wide_shift_reads_the_right_bits() {
        let hi = 0x0123_4567_89AB_CDEF;
        let lo = 0xFEDC_BA98_7654_3210;
        let whole = u128::from(hi) << 64 | u128::from(lo);
        let mut s = 1;
        while s < 128 {
            assert_eq!(
                u128::from(shr_wide(hi, lo, s)),
                (whole >> s) & u128::from(u64::MAX),
                "{s}"
            );
            s += 1;
        }
        // 0 and 128 read as 64.
        assert_eq!(shr_wide(hi, lo, 0), hi);
        assert_eq!(shr_wide(hi, lo, 128), hi);
        assert_eq!(shr_wide(hi, lo, 64), hi);
    }

    #[test]
    fn the_wide_sum_carries() {
        assert_eq!(add_wide(0, u64::MAX, 0, 1), (1, 0));
        assert_eq!(add_wide(5, 10, 7, 20), (12, 30));
    }
}
