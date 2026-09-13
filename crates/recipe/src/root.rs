//! THE INTEGER ROOT AND THE RECIPROCAL — the two places the float recipe had a square root and a
//! division, as loops of shifts, compares and subtracts.
//!
//! The root is the textbook bit-by-bit method: 32 steps for a 64-bit word, the same on every host.
//! The reciprocal `floor(2⁶⁰ / len)` for a length at 30 fraction bits is six Newton steps from the
//! seed 1.0 (the length is in `[1, √3]`, so the seed is under `2 / len` and the iteration converges)
//! and then one compare loop that lands it EXACTLY on the floor (a Newton reciprocal step only ever
//! lands below the true value, and every shift floors): MEASURED (bench part 2), without the landing
//! the steps read three units low, and with it the reciprocal is the division's own answer. No `/`
//! anywhere.
//!
//! [`recip_pow2`] is the GENERAL reciprocal the same rule asks for: `floor(2^bits / d)` for any
//! positive word, as the restoring binary division loop — one shift, one compare and one subtract per
//! bit of the numerator. It runs ONCE per body (the radius's reciprocal, which the column bound
//! reads) and ONCE per tube carver (the reciprocal of the segment's squared length, which the
//! projection reads), never per cell, and it names no `/`, so the kernel that holds it compiles for
//! the GPU too.

use crate::gi::Gi;

/// The fraction bits of the reciprocal's length and result.
pub const RECIP_BITS: u32 = 30;

/// ★ `floor(√n)` OF AN UNSIGNED WORD — the textbook restoring method, WRITTEN OUT as its
/// [`ROOT_STEPS`] steps, with no loop and no branch.
///
/// **Why it is written out (MEASURED, 2026-09-13, step G1).** This root used to be two `while`
/// loops. On the card the answer came out ONE STEP SHORT: `isqrt(1)` read 0 (`integer_bench`, the
/// `isqrt_probe` entry point). Reading naga's Metal showed why: rust-gpu carries such a loop's
/// value out in a SPILLED VARIABLE written at the TOP of the body, so the value read after the
/// loop is the value from before the last step. Three loop shapes each read 0 for `isqrt(1)` —
/// the counted `while`, the `loop` that returns from inside, and the branchless body. The octave
/// sum's loop does NOT (its value leaves in a phi, and 3 936 256 columns agree), so the fault is
/// the SHAPE, not the loop. Thirty-two steps in a row have no shape to get wrong, and they are
/// faster on a card besides, where a loop costs a thread every iteration the widest lane takes.
/// With the steps written out the probe reads 0 of 511 words differing.
///
/// **Why there is no branch either.** `fits` is a MASK — all ones where the bit fits under the
/// remainder, all zeros where it does not — so the subtraction and the addition happen always and
/// the word decides their value. One answer on every host, and nothing for a compiler to predict.
///
/// **Example.** A tube carver runs three metres from the pilot's boots. The squared distance in
/// gap steps is 147 456; this root answers 384 gap steps on the shard and 384 on the card, so the
/// passage the player walks into is the passage the card drew.
#[must_use]
pub const fn isqrt(n: u64) -> u64 {
    let (x, res) = (n, 0u64);
    let (x, res) = root_step(x, res, 1u64 << 62);
    let (x, res) = root_step(x, res, 1u64 << 60);
    let (x, res) = root_step(x, res, 1u64 << 58);
    let (x, res) = root_step(x, res, 1u64 << 56);
    let (x, res) = root_step(x, res, 1u64 << 54);
    let (x, res) = root_step(x, res, 1u64 << 52);
    let (x, res) = root_step(x, res, 1u64 << 50);
    let (x, res) = root_step(x, res, 1u64 << 48);
    let (x, res) = root_step(x, res, 1u64 << 46);
    let (x, res) = root_step(x, res, 1u64 << 44);
    let (x, res) = root_step(x, res, 1u64 << 42);
    let (x, res) = root_step(x, res, 1u64 << 40);
    let (x, res) = root_step(x, res, 1u64 << 38);
    let (x, res) = root_step(x, res, 1u64 << 36);
    let (x, res) = root_step(x, res, 1u64 << 34);
    let (x, res) = root_step(x, res, 1u64 << 32);
    let (x, res) = root_step(x, res, 1u64 << 30);
    let (x, res) = root_step(x, res, 1u64 << 28);
    let (x, res) = root_step(x, res, 1u64 << 26);
    let (x, res) = root_step(x, res, 1u64 << 24);
    let (x, res) = root_step(x, res, 1u64 << 22);
    let (x, res) = root_step(x, res, 1u64 << 20);
    let (x, res) = root_step(x, res, 1u64 << 18);
    let (x, res) = root_step(x, res, 1u64 << 16);
    let (x, res) = root_step(x, res, 1u64 << 14);
    let (x, res) = root_step(x, res, 1u64 << 12);
    let (x, res) = root_step(x, res, 1u64 << 10);
    let (x, res) = root_step(x, res, 1u64 << 8);
    let (x, res) = root_step(x, res, 1u64 << 6);
    let (x, res) = root_step(x, res, 1u64 << 4);
    let (x, res) = root_step(x, res, 1u64 << 2);
    let (x, res) = root_step(x, res, 1u64 << 0);
    let _ = x;
    res
}

/// One step of the restoring square root: the bit fits under the remainder, or it does not, and a
/// MASK — never a branch — says which.
const fn root_step(x: u64, res: u64, bit: u64) -> (u64, u64) {
    let sum = res.wrapping_add(bit);
    let fits = 0u64.wrapping_sub((x >= sum) as u64);
    (
        x.wrapping_sub(sum & fits),
        (res >> 1).wrapping_add(bit & fits),
    )
}

/// The steps [`isqrt`] takes, whatever the word: the powers of four a 64-bit word holds, from
/// `4³¹` down to `4⁰`. A step whose bit stands above the root costs nothing — the remainder is
/// smaller than the bit, so the mask is zero and the partial root stays where it is.
pub const ROOT_STEPS: u32 = 32;

/// `floor(2^bits / d)` for a positive `d` under 2⁶² and `bits` at most 63: the restoring binary
/// division loop — one shift, one compare and one subtract per bit of the numerator. A `d` of zero
/// reads as one, so the loop cannot run away (a caller that can hold a degenerate value — a tube
/// carver of no length — tests for it and never reads the answer).
///
/// **Example.** A tube carver runs 431 metres through the rock. Its squared length in gap steps is
/// drawn once when the carver is drawn, and its reciprocal with it; every cell the carver may hollow
/// then projects onto the segment with one two-word product, and no cell divides.
#[must_use]
pub const fn recip_pow2(d: u64, bits: u32) -> u64 {
    let d = if d == 0 { 1 } else { d };
    let mut rem = 0u64;
    let mut q = 0u64;
    let mut i = bits;
    loop {
        // The numerator is 2^bits: the only set bit it has is the one this loop starts at.
        let bit = if i == bits { 1u64 } else { 0u64 };
        rem = rem.wrapping_shl(1) | bit;
        if rem >= d {
            rem = rem.wrapping_sub(d);
            q |= 1u64.wrapping_shl(i);
        }
        if i == 0 {
            return q;
        }
        i -= 1;
    }
}

/// `floor(2⁶⁰ / len)` for `len` at 30 fraction bits in `[2³⁰, 2³¹)` (a length in `[1, 2)`): the
/// reciprocal at 30 fraction bits, exact.
#[must_use]
pub fn recip30(len: Gi) -> Gi {
    let one = Gi::ONE << RECIP_BITS;
    let top = Gi::ONE << (2 * RECIP_BITS);
    let mut r = one;
    let mut step = 0;
    while step < 6 {
        // e = 2⁶⁰ − len·r, the residual at 60 fraction bits, under 2⁶¹ by the range of `len`.
        let e = top - len * r;
        r += (r * (e >> RECIP_BITS)) >> RECIP_BITS;
        step += 1;
    }
    // THE EXACT LANDING. Newton's reciprocal step lands BELOW the true value from any seed
    // (`x' = 1/a − a·(x − 1/a)²`), and every shift here floors, so `r` never passes the floor of
    // 2⁶⁰/len: one loop upward lands it exactly (MEASURED on a sweep in the test below; without
    // it the steps read up to three units low).
    while len * (r + Gi::ONE) <= top {
        r += Gi::ONE;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_root_is_the_floor_of_the_square_root() {
        let cases: [u64; 14] = [
            0,
            1,
            2,
            3,
            4,
            5,
            15,
            16,
            17,
            100,
            380_000,
            1 << 40,
            u64::MAX,
            (1 << 62) + 12345,
        ];
        for n in cases {
            let r = isqrt(n);
            assert!(u128::from(r) * u128::from(r) <= u128::from(n), "{n}");
            assert!(
                (u128::from(r) + 1) * (u128::from(r) + 1) > u128::from(n),
                "{n}"
            );
        }
        // The widest word's root is the widest half-word: the steps and the range agree.
        assert_eq!(isqrt(u64::MAX), (1u64 << ROOT_STEPS) - 1);
        // A sweep of the range the recipe uses: the square sum of a unit direction at 30 bits.
        let mut n = 1u64 << 60;
        while n < 3u64 << 60 {
            let r = isqrt(n);
            assert!(r * r <= n);
            assert!((r + 1) * (r + 1) > n);
            n = n.wrapping_add((1 << 55) + 7919);
        }
    }

    #[test]
    #[allow(clippy::integer_division, reason = "the test states the exact floor")]
    fn the_general_reciprocal_is_the_exact_floor_of_the_division() {
        // The two the recipe asks for: a body's radius in gap steps, and a tube's squared length.
        for (d, bits) in [
            (1u64, 62u32),
            (1, 0),
            (2, 63),
            (815_405_260, 62),
            (5_466_132_837, 62),
            (3, 40),
            (u64::MAX >> 2, 62),
            (7, 10),
        ] {
            let want = ((1u128 << bits) / u128::from(d)) as u64;
            assert_eq!(recip_pow2(d, bits), want, "2^{bits} / {d}");
        }
        // A sweep over the tube carvers' own range: squared gap steps.
        let mut d = 1u64 << 20;
        while d < (1u64 << 36) {
            assert_eq!(
                u128::from(recip_pow2(d, 62)),
                (1u128 << 62) / u128::from(d),
                "{d}"
            );
            d = d.wrapping_add((1 << 28) + 7_919);
        }
        // A degenerate divisor reads as one rather than running away.
        assert_eq!(recip_pow2(0, 10), 1 << 10);
    }

    #[test]
    #[allow(clippy::integer_division, reason = "the test states the exact floor")]
    fn the_reciprocal_is_the_exact_floor() {
        let mut len = 1i64 << RECIP_BITS;
        while len < (1i64 << 31) {
            let r = recip30(Gi::new(len)).raw();
            let exact = (1i128 << 60) / i128::from(len);
            assert_eq!(i128::from(r), exact, "{len}");
            len += (1 << 23) + 104_729;
        }
        // The ends of the range.
        assert_eq!(recip30(Gi::new(1 << 30)).raw(), 1 << 30);
        assert_eq!(
            i128::from(recip30(Gi::new((1 << 31) - 1)).raw()),
            (1i128 << 60) / ((1i128 << 31) - 1)
        );
    }
}
