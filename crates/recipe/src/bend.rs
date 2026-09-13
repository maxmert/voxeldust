//! THE FACE BEND AT 40 FRACTION BITS — from a cell's `(face, i, j)` to its unit direction from the
//! body's centre, on integers.
//!
//! The float recipe (`vd_seed::bend`) bent a cube face onto the sphere with `W(a) = a(k₁ + a²(k₂ +
//! a²k₃))`, `k₁ = π/4`, then normalised with a square root and three divisions. Here `k₁`, `k₂` and
//! `k₃` are words at 40 fraction bits whose sum is EXACTLY one, so a face edge still lands on the
//! cube edge exactly; the face parameter `(2i + 1)/n_l − 1` is one multiply by a reciprocal computed
//! once per body ([`inv_n_of`]); the products run through the two-word multiply; the square sum is
//! carried at 80 fraction bits in two words; and the normalise is a reciprocal square root — a seed
//! at 30 fraction bits (an integer root of the top word and its exact reciprocal) and ONE Newton
//! step with the residual at 120 bits, because an error of 2⁻³⁰ squares to 2⁻⁶⁰, past the 40 bits
//! kept. MEASURED (bench part 3): identical on the CPU and the GPU on 11 808 768 components; within
//! 0.02 mm of the float bend laterally on every column of the home planet; 77 ns a column.
//!
//! **Example.** Cell (face +X, 1181, 77) of a rung with 10 006 528 cells to an edge: the face
//! parameters are two words near −1, the bend pulls them toward the face's centre by the quintic,
//! the basis places the point beside the +X axis, and the normalise scales it onto the sphere — the
//! same word triple on the moon's shard and on the pilot's GPU.

use crate::gi::Gi;
use crate::root::{RECIP_BITS, isqrt, recip30};
use crate::wide::{add_wide, mul_wide, shr_wide};

/// The direction's fraction bits (F8 decision 1).
pub const DIR_BITS: u32 = 40;
/// One at the direction's format.
pub const DIR_ONE: Gi = Gi::new(1 << DIR_BITS);
/// The fraction bits of a body's cell-count reciprocal ([`inv_n_of`]): a whole word, so the face
/// parameter's error is under half a unit at [`DIR_BITS`] out to the face's edge (a reciprocal at
/// 2⁻⁵⁶ left the edge's cells three millimetres off on the home planet — the first run of the
/// crate's own test).
pub const INV_BITS: u32 = 64;
/// The smallest cell count [`inv_n_of`] will divide by: the reciprocal of two is the most negative
/// word and the reciprocal of one truncates to nothing, so three is the floor that keeps the word a
/// positive number. A real face is at least a chunk wide (62 cells).
pub const INV_MIN_CELLS: u32 = 3;

/// `k₁ = round(π/4 · 2⁴⁰)`.
pub const K1: Gi = Gi::new(863_554_413_089);
/// `k₂ = round(0.15 · 2⁴⁰)`.
pub const K2: Gi = Gi::new(164_926_744_166);
/// `k₃ = 2⁴⁰ − k₁ − k₂`, so the three sum to one exactly.
pub const K3: Gi = Gi::new(71_030_470_521);
const _: () = assert!(K1.raw() + K2.raw() + K3.raw() == 1 << DIR_BITS);

/// A face's basis: the normal, the `u` axis (along `i`) and the `v` axis (along `j`), each an axis
/// with one component of ±1 — the address format's table, as `vd_seed::bend::BASIS` has it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Basis {
    /// The face's outward normal.
    pub n: [i8; 3],
    /// The axis a cell's `i` counts along.
    pub u: [i8; 3],
    /// The axis a cell's `j` counts along.
    pub v: [i8; 3],
}

/// THE FACE BASIS TABLE, by face index: +X, −X, +Y, −Y, +Z, −Z.
pub const BASIS: [Basis; 6] = [
    Basis {
        n: [1, 0, 0],
        u: [0, 1, 0],
        v: [0, 0, 1],
    },
    Basis {
        n: [-1, 0, 0],
        u: [0, 0, 1],
        v: [0, 1, 0],
    },
    Basis {
        n: [0, 1, 0],
        u: [0, 0, 1],
        v: [1, 0, 0],
    },
    Basis {
        n: [0, -1, 0],
        u: [1, 0, 0],
        v: [0, 0, 1],
    },
    Basis {
        n: [0, 0, 1],
        u: [1, 0, 0],
        v: [0, 1, 0],
    },
    Basis {
        n: [0, 0, -1],
        u: [0, 1, 0],
        v: [1, 0, 0],
    },
];

/// The reciprocal of a face's cell count at [`INV_BITS`]: `floor(2⁶⁴ / n_l)`, computed ONCE per body
/// on the CPU when the body's integers are drawn, and stored beside them. This is the recipe's only
/// division, and it never runs in a kernel: a body's charter carries the result.
///
/// THE COUNT IS CLAMPED TO AT LEAST [`INV_MIN_CELLS`], so the function is TOTAL: `2⁶⁴/2` is
/// `i64::MIN` and `2⁶⁴/1` truncates to zero, and either word would bend a face inside out. `2⁶⁴/3`
/// still fits the word, so three is the floor. The clamp is a GUARD, not a path: a face of a body on
/// the ladder never has fewer than one chunk's width of cells (62), because the ladder's own snap
/// keeps every rung at least that wide.
#[must_use]
#[allow(
    clippy::integer_division,
    reason = "computed once per body on the CPU, never in a kernel"
)]
pub const fn inv_n_of(n_l: u32) -> Gi {
    let n = if n_l < INV_MIN_CELLS {
        INV_MIN_CELLS
    } else {
        n_l
    };
    Gi::new(((1u128 << INV_BITS) / (n as u128)) as i64)
}

/// The face parameter of a cell's centre, `(2i + 1)/n_l − 1`, at [`DIR_BITS`]: one multiply by the
/// body's reciprocal and a shift.
#[must_use]
pub fn face_param(i: i32, inv_n: Gi) -> Gi {
    let odd = (Gi::new(i as i64) << 1) + Gi::ONE;
    odd.mul_shr(inv_n, INV_BITS - DIR_BITS) - DIR_ONE
}

/// The forward bend `W(a) = a(k₁ + a²(k₂ + a²k₃))` at [`DIR_BITS`], Horner form, two-word products.
#[must_use]
pub fn bend(a: Gi) -> Gi {
    let a2 = a.mul_shr(a, DIR_BITS);
    let inner = K2 + a2.mul_shr(K3, DIR_BITS);
    let inner2 = K1 + a2.mul_shr(inner, DIR_BITS);
    a.mul_shr(inner2, DIR_BITS)
}

/// The unit direction of cell `(face, i, j)` at [`DIR_BITS`], for a body whose cell-count reciprocal
/// at this rung is `inv_n`. `face` is the face index; an index past 5 reads as −Z.
#[must_use]
pub fn direction(face: u8, i: i32, j: i32, inv_n: Gi) -> [Gi; 3] {
    let basis = BASIS[if (face as usize) < 6 {
        face as usize
    } else {
        5
    }];
    let wa = bend(face_param(i, inv_n));
    let wb = bend(face_param(j, inv_n));
    normalise([
        axis(basis, 0, wa, wb),
        axis(basis, 1, wa, wb),
        axis(basis, 2, wa, wb),
    ])
}

/// `v / |v|` at [`DIR_BITS`], for a vector whose components are at most a few at [`DIR_BITS`]: the
/// reciprocal square root of the square sum, then three two-word products.
///
/// **Example.** At a cube corner a chunk's box holds a PHANTOM column on the corner direction. Its
/// axis sum is `(±1, ±1, ±1)` at [`DIR_BITS`], and this kernel turns it into the unit direction the
/// three faces that meet there all read (`vd_terrain::lattice::site_dir`).
#[must_use]
pub fn normalise(v: [Gi; 3]) -> [Gi; 3] {
    let y = recip_sqrt(v);
    [
        v[0].mul_shr(y, DIR_BITS),
        v[1].mul_shr(y, DIR_BITS),
        v[2].mul_shr(y, DIR_BITS),
    ]
}

/// One component of `n + W(a)·u + W(b)·v`; the axes are −1, 0 or 1.
fn axis(basis: Basis, c: usize, wa: Gi, wb: Gi) -> Gi {
    Gi::new(basis.n[c] as i64) * DIR_ONE
        + wa * Gi::new(basis.u[c] as i64)
        + wb * Gi::new(basis.v[c] as i64)
}

/// `1/|v|` at [`DIR_BITS`] for a vector whose components are under one at [`DIR_BITS`]: the square
/// sum at 80 fraction bits in two words, a seed from the 30-bit root and its exact reciprocal, and
/// one Newton step of the reciprocal square root with the residual carried at 120 bits.
fn recip_sqrt(v: [Gi; 3]) -> Gi {
    // S = Σ v² at 80 fraction bits.
    let (mut s_hi, mut s_lo) = (0u64, 0u64);
    let mut c = 0;
    while c < 3 {
        let m = v[c].unsigned_abs();
        let (h, l) = mul_wide(m, m);
        (s_hi, s_lo) = add_wide(s_hi, s_lo, h, l);
        c += 1;
    }
    // T = S >> 20: S at 60 fraction bits in one word (S < 3·2⁸⁰, so T < 2⁶²).
    let t = shr_wide(s_hi, s_lo, 20);
    // The seed: the 30-bit root and its exact reciprocal, widened to the direction's bits.
    let y0 = recip30(Gi::new(isqrt(t) as i64)) << (DIR_BITS - RECIP_BITS);
    // y0² at 60 fraction bits in one word.
    let (yh, yl) = mul_wide(y0.unsigned_abs(), y0.unsigned_abs());
    let y0sq = shr_wide(yh, yl, 20);
    // P = T × y0² = S·y0² at 120 fraction bits, in two words; E = 2¹²⁰ − P, signed.
    let (ph, pl) = mul_wide(t, y0sq);
    let one120_hi = 1u64 << 56;
    let (e_neg, e_hi, e_lo) = if ph > one120_hi || (ph == one120_hi && pl > 0) {
        (true, ph.wrapping_sub(one120_hi), pl)
    } else {
        let (l, borrow) = 0u64.overflowing_sub(pl);
        (
            false,
            one120_hi.wrapping_sub(ph).wrapping_sub(borrow as u64),
            l,
        )
    };
    // e at 70 fraction bits in one word: |E| ≈ 2⁻³⁰ · 2¹²⁰, so E >> 50 is under 2⁴¹.
    let e70 = Gi::new(shr_wide(e_hi, e_lo, 50) as i64);
    let e70 = if e_neg { Gi::ZERO - e70 } else { e70 };
    // y1 = y0 + y0·e/2: (y0 at 40) × (e at 70) >> 71.
    y0 + y0.mul_shr(e70, 71)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: f64 = (1u64 << DIR_BITS) as f64;

    #[allow(
        clippy::float_arithmetic,
        reason = "the test reads the word as a real number"
    )]
    fn real(g: Gi) -> f64 {
        g.raw() as f64 / ONE
    }

    #[test]
    fn the_constants_sum_to_one_and_the_face_edge_lands_on_the_cube_edge() {
        assert_eq!(K1 + K2 + K3, DIR_ONE);
        assert_eq!(bend(DIR_ONE), DIR_ONE);
        assert_eq!(bend(Gi::ZERO - DIR_ONE), Gi::ZERO - DIR_ONE);
        assert_eq!(bend(Gi::ZERO), Gi::ZERO);
        // Odd: W(−a) = −W(a).
        let a = Gi::new(123_456_789_012);
        assert_eq!(bend(Gi::ZERO - a), Gi::ZERO - bend(a));
    }

    #[test]
    #[allow(
        clippy::float_arithmetic,
        reason = "the test compares against real arithmetic"
    )]
    #[allow(
        clippy::integer_division,
        reason = "the test states the exact reciprocal"
    )]
    fn the_face_parameter_is_the_cell_centre_in_the_face() {
        let n_l = 10_006_528u32;
        let inv_n = inv_n_of(n_l);
        assert_eq!(inv_n.raw(), ((1u128 << 64) / u128::from(n_l)) as i64);
        // The first cell's centre, the middle, the last: (2i + 1)/n_l − 1, within a unit.
        for i in [0i32, 5_003_263, 5_003_264, 10_006_527] {
            let want = (2.0 * f64::from(i) + 1.0) / f64::from(n_l) - 1.0;
            let got = real(face_param(i, inv_n));
            assert!((got - want).abs() < 1.5 / ONE, "{i}: {got} vs {want}");
        }
        // The smallest face a rung can have: 62 cells; the reciprocal fits the word.
        assert_eq!(inv_n_of(62).raw(), ((1u128 << 64) / 62) as i64);
        // ★ THE GUARD IS TOTAL: a count under three reads as three, so the word is always a positive
        // number. (Before the clamp, two gave the most negative word and one gave zero — either one
        // bends a face inside out.)
        let three = ((1u128 << 64) / 3) as i64;
        for n_l in [0u32, 1, 2, 3] {
            assert_eq!(inv_n_of(n_l).raw(), three, "{n_l} cells read as three");
        }
        assert!(inv_n_of(0).raw() > 0);
    }

    #[test]
    #[allow(
        clippy::float_arithmetic,
        reason = "the test compares against real arithmetic"
    )]
    fn a_direction_is_a_unit_vector_on_every_face_and_the_faces_meet_at_the_edge() {
        let n_l = 62u32;
        let inv_n = inv_n_of(n_l);
        let mut face = 0u8;
        while face < 6 {
            for (i, j) in [(0, 0), (30, 30), (61, 0), (0, 61), (61, 61), (7, 40)] {
                let d = direction(face, i, j, inv_n);
                // THE SQUARED LENGTH against one, with a squared tolerance: the float fence bans
                // `hypot` (a platform call), and a square root is not needed to say "this is a unit
                // vector within four units of the last bit".
                let (x, y, z) = (real(d[0]), real(d[1]), real(d[2]));
                let len2 = x * x + y * y + z * z;
                let tol = 8.0 / ONE; // |len² − 1| ≈ 2·|len − 1| near one
                assert!(
                    (len2 - 1.0).abs() < tol,
                    "face {face} ({i}, {j}): |d|² = {len2}"
                );
                // The face's own normal is the largest component, on the right side.
                let b = BASIS[face as usize];
                let mut c = 0;
                while c < 3 {
                    if b.n[c] != 0 {
                        assert!(
                            real(d[c]) * f64::from(b.n[c]) > 0.5,
                            "face {face} ({i}, {j})"
                        );
                    }
                    c += 1;
                }
            }
            face += 1;
        }
        // A face index past the table reads as −Z.
        assert_eq!(direction(9, 3, 4, inv_n), direction(5, 3, 4, inv_n));
        // The centre cell of +X points along +X with a tilt of half a cell.
        let d = direction(0, 31, 31, inv_n);
        assert!(real(d[0]) > 0.999);
    }

    #[test]
    fn the_corner_direction_is_the_axis_sum_made_a_unit_vector() {
        // The cube corner a chunk box's phantom column sits on: (1, 1, 1)/√3 at 40 bits.
        let third = 634_803_334_274i64; // round(2⁴⁰ / √3)
        let d = normalise([DIR_ONE, DIR_ONE, DIR_ONE]);
        let mut c = 0;
        while c < 3 {
            assert!((d[c].raw() - third).abs() <= 2, "[{c}]: {:?}", d[c]);
            c += 1;
        }
        // The signs ride through: the (−u, −v) corner of a minus face.
        let n = normalise([Gi::ZERO - DIR_ONE, Gi::ZERO - DIR_ONE, DIR_ONE]);
        assert_eq!(n[0], Gi::ZERO - d[0]);
        assert_eq!(n[1], Gi::ZERO - d[1]);
        assert_eq!(n[2], d[2]);
        // An axis on its own is already a unit vector.
        assert_eq!(
            normalise([DIR_ONE, Gi::ZERO, Gi::ZERO]),
            [DIR_ONE, Gi::ZERO, Gi::ZERO]
        );
    }

    #[test]
    fn the_reciprocal_square_root_reads_one_over_the_length() {
        // |v| = 1 exactly: y = 1.
        assert_eq!(recip_sqrt([DIR_ONE, Gi::ZERO, Gi::ZERO]), DIR_ONE);
        // |v| = √3: y = 1/√3 at 40 bits, within two units.
        let y = recip_sqrt([DIR_ONE, DIR_ONE, DIR_ONE]).raw();
        let want = 634_803_334_274i64; // round(2⁴⁰ / √3)
        assert!((y - want).abs() <= 2, "{y} vs {want}");
        // |v|² = 2, y = 1/√2.
        let y = recip_sqrt([DIR_ONE, DIR_ONE, Gi::ZERO]).raw();
        let want = 777_472_127_994i64; // round(2⁴⁰ / √2)
        assert!((y - want).abs() <= 2, "{y} vs {want}");
        // THE RESIDUAL ON THE OTHER SIDE: a length a hair over one floors its 30-bit root to one,
        // so the seed reads high and the residual is negative — through the high word (a second
        // component of 2²⁰) and through the equal-high-word, low-word arm (a component of 2¹¹).
        for (eps, want) in [
            (1i64 << 20, 1_099_511_627_776i64),
            (2_048, 1_099_511_627_776),
        ] {
            let y = recip_sqrt([DIR_ONE, Gi::new(eps), Gi::ZERO]).raw();
            assert!((y - want).abs() <= 2, "eps {eps}: {y} vs {want}");
        }
    }
}
