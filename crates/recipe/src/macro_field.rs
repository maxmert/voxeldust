//! ★ THE MACRO FIELD's READ (the landform arc, slice 8c stage C4; `03_erosion_rivers.md` §5.4):
//! the Catmull-Rom interpolation of sixteen node words at two fractions, on integers — the kernel
//! every column runs to read the eroded height `Z` between four macro nodes. Which sixteen words
//! and which fractions is the host's gather (`vd_terrain::artifact`); the arithmetic is here, ONE
//! SOURCE for the CPU and, later, the card.
//!
//! A Catmull-Rom spline passes through its nodes and is C1 between them — the value and the slope
//! agree from both sides of every node — so the valley the solve cut reads as one smooth valley at
//! every rung, with no step at a node's edge. The weights at a fraction `t` in `[0, 1)`:
//!
//! ```text
//!   w₀ = (−t³ + 2t² − t) / 2      w₁ = (3t³ − 5t² + 2) / 2
//!   w₂ = (−3t³ + 4t² + t) / 2     w₃ = (t³ − t²) / 2
//! ```
//!
//! at [`NOISE_BITS`] fraction bits; the value is the row-by-row sum, each product through the
//! two-word multiply (a node word at the length format times a weight passes 63 bits), truncated
//! toward zero as every `mul_shr` is. One evaluation order, so two hosts cannot disagree.
//!
//! **Example.** A column stands a third of the way from one macro node to the next along both
//! axes. The host gathers the sixteen surrounding nodes' heights; this kernel answers the height
//! under that column, the same word on the planet's shard and on the client that draws it.

use crate::Gi;
use crate::noise::NOISE_BITS;

/// The four Catmull-Rom weights at fraction `t` (at [`NOISE_BITS`]), at the same bits.
#[must_use]
pub fn weights(t: Gi) -> [Gi; 4] {
    let t2 = t.mul_shr(t, NOISE_BITS);
    let t3 = t2.mul_shr(t, NOISE_BITS);
    let one = Gi::new(1 << NOISE_BITS);
    let two = Gi::new(2 << NOISE_BITS);
    let w0 = (t2 + t2 - t3 - t) >> 1;
    let w1 = (t3 + t3 + t3 - t2 * Gi::new(5) + two) >> 1;
    let w2 = (t2 * Gi::new(4) - t3 - t3 - t3 + t) >> 1;
    let w3 = (t3 - t2) >> 1;
    let _ = one;
    [w0, w1, w2, w3]
}

/// The interpolated value of sixteen node words `z[row · 4 + column]` (rows along the second
/// axis, columns along the first, each from one node before the point to two after) at the
/// fractions `ta` (the first axis) and `tb` (the second), both at [`NOISE_BITS`]. The words are
/// at any format; the answer is at the same format.
#[must_use]
pub fn catmull_rom_16(z: &[Gi; 16], ta: Gi, tb: Gi) -> Gi {
    let wa = weights(ta);
    let wb = weights(tb);
    let mut out = Gi::ZERO;
    let mut row = 0;
    while row < 4 {
        let mut along = Gi::ZERO;
        let mut col = 0;
        while col < 4 {
            along += z[row * 4 + col].mul_shr(wa[col], NOISE_BITS);
            col += 1;
        }
        out += along.mul_shr(wb[row], NOISE_BITS);
        row += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE AND MAY READ A FLOAT (ruling F7's rule is about the SHIPPED path): a test
    //! states an exact quotient of the classical weights and reads a fraction as a real number.
    #![allow(
        clippy::integer_division,
        clippy::float_arithmetic,
        reason = "a test states exact quotients and reads a fraction as a float; never a kernel's path"
    )]
    use super::*;

    fn q(v: f64) -> Gi {
        Gi::new((v * f64::from(1u32 << NOISE_BITS)) as i64)
    }

    /// At a node (`t = 0`) the weights are `(0, 1, 0, 0)`; at the half they are the classical
    /// `(−1/16, 9/16, 9/16, −1/16)`; and they sum to one at every fraction.
    #[test]
    fn the_weights_pass_through_the_nodes_and_sum_to_one() {
        let one = 1i64 << NOISE_BITS;
        assert_eq!(
            weights(Gi::ZERO),
            [Gi::ZERO, Gi::new(one), Gi::ZERO, Gi::ZERO]
        );
        let half = weights(q(0.5));
        let expect = [-one / 16, 9 * one / 16, 9 * one / 16, -one / 16];
        for (w, e) in half.iter().zip(expect) {
            assert!((w.raw() - e).abs() <= 2, "{} vs {e}", w.raw());
        }
        let mut t = 0.0;
        while t < 1.0 {
            let w = weights(q(t));
            let sum: i64 = w.iter().map(|g| g.raw()).sum();
            assert!((sum - one).abs() <= 4, "at {t}: {sum}");
            t += 0.05;
        }
    }

    /// A flat field reads flat, a plane reads the plane (Catmull-Rom reproduces linear fields), and
    /// the value at a node is that node's word.
    #[test]
    fn a_plane_is_reproduced_and_a_node_reads_itself() {
        let flat = [Gi::new(1 << 40); 16];
        assert!((catmull_rom_16(&flat, q(0.3), q(0.7)).raw() - (1 << 40)).abs() <= 8);
        // z = 1000·(a + 2b) in some unit, at nodes a, b ∈ {−1, 0, 1, 2}.
        let mut plane = [Gi::ZERO; 16];
        for row in 0..4 {
            for col in 0..4 {
                let (a, b) = (col as i64 - 1, row as i64 - 1);
                plane[row * 4 + col] = Gi::new((a + 2 * b) * 1_000 * (1 << NOISE_BITS));
            }
        }
        let at =
            catmull_rom_16(&plane, q(0.25), q(0.5)).raw() as f64 / f64::from(1u32 << NOISE_BITS);
        assert!((at - (0.25 + 2.0 * 0.5) * 1_000.0).abs() < 0.01, "{at}");
        let node = catmull_rom_16(&plane, Gi::ZERO, Gi::ZERO);
        assert_eq!(node, plane[5]);
    }
}
