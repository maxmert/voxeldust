//! THE NOISE AT 28 FRACTION BITS — Perlin's gradient noise and value noise on the integer lattice,
//! the recipe's own operations in the recipe's own order, on words.
//!
//! The lattice point arrives at [`NOISE_BITS`] fraction bits: its cube is the arithmetic shift
//! (the floor), its offset the masked fraction. Sixteen gradients (Perlin's twelve cube edges, four
//! repeated so a four-bit draw selects one) with components −1, 0 and 1, so a dot product is adds
//! and subtracts of the offsets; the quintic fade `t³(t(6t − 15) + 10)` in Horner form with each
//! product shifted back once; `a + t(b − a)` for the blend, eight corners in ONE fixed order. Value
//! noise draws a corner's value as the hash's top bits, a shift instead of the float recipe's
//! division by 2⁵³. MEASURED (bench part 1): the octave sum on this noise sits within 0.13 mm mean
//! and 1.06 mm widest of the float recipe over four million columns, and 0 of them differ between
//! the CPU and the GPU; 24 fraction bits left the coarsest octave's eight kilometres of amplitude a
//! five-millimetre error, so 28 it is.

use crate::gi::Gi;
use crate::rng::corner_hash;

/// The noise's fraction bits: the lattice point's, the fade's, the blend's and the result's.
pub const NOISE_BITS: u32 = 28;
/// One at the noise's format.
pub const NOISE_ONE: Gi = Gi::new(1 << NOISE_BITS);
const FRAC_MASK: Gi = Gi::new((1 << NOISE_BITS) - 1);

/// Perlin's sixteen gradients.
const GRADIENTS: [[i8; 3]; 16] = [
    [1, 1, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, 1],
    [0, 1, -1],
    [0, -1, -1],
    [1, 1, 0],
    [-1, 1, 0],
    [0, -1, 1],
    [0, -1, -1],
];

/// The quintic fade `t³(t(6t − 15) + 10)` for `t` in `[0, 1)` at [`NOISE_BITS`], every product
/// shifted back once (each intermediate under 2⁶⁰).
#[must_use]
pub fn fade(t: Gi) -> Gi {
    let t2 = (t * t) >> NOISE_BITS;
    let t3 = (t2 * t) >> NOISE_BITS;
    let inner = Gi::new(6) * t - Gi::new(15) * NOISE_ONE;
    let poly = ((inner * t) >> NOISE_BITS) + Gi::new(10) * NOISE_ONE;
    (t3 * poly) >> NOISE_BITS
}

/// `a + t(b − a)` at [`NOISE_BITS`]: one multiply, one shift, one add, in that order.
#[must_use]
pub fn lerp(a: Gi, b: Gi, t: Gi) -> Gi {
    a + ((t * (b - a)) >> NOISE_BITS)
}

fn dot(g: [i8; 3], dx: Gi, dy: Gi, dz: Gi) -> Gi {
    Gi::new(g[0] as i64) * dx + Gi::new(g[1] as i64) * dy + Gi::new(g[2] as i64) * dz
}

fn gradient(seed: u64, x: i64, y: i64, z: i64) -> [i8; 3] {
    GRADIENTS[(corner_hash(seed, x, y, z) & 15) as usize]
}

/// Gradient noise at a lattice point at [`NOISE_BITS`], in about `[−1, 1]` at [`NOISE_BITS`]: the
/// point's cube is its floor; the eight corners are blended in ONE fixed order (x pairs, then y
/// pairs, then z).
#[must_use]
pub fn noise3(seed: u64, p: [Gi; 3]) -> Gi {
    let x0 = (p[0] >> NOISE_BITS).raw();
    let y0 = (p[1] >> NOISE_BITS).raw();
    let z0 = (p[2] >> NOISE_BITS).raw();
    let dx = p[0] & FRAC_MASK;
    let dy = p[1] & FRAC_MASK;
    let dz = p[2] & FRAC_MASK;
    let u = fade(dx);
    let v = fade(dy);
    let w = fade(dz);
    let one = NOISE_ONE;
    let c000 = dot(gradient(seed, x0, y0, z0), dx, dy, dz);
    let c100 = dot(gradient(seed, x0 + 1, y0, z0), dx - one, dy, dz);
    let c010 = dot(gradient(seed, x0, y0 + 1, z0), dx, dy - one, dz);
    let c110 = dot(gradient(seed, x0 + 1, y0 + 1, z0), dx - one, dy - one, dz);
    let c001 = dot(gradient(seed, x0, y0, z0 + 1), dx, dy, dz - one);
    let c101 = dot(gradient(seed, x0 + 1, y0, z0 + 1), dx - one, dy, dz - one);
    let c011 = dot(gradient(seed, x0, y0 + 1, z0 + 1), dx, dy - one, dz - one);
    let c111 = dot(
        gradient(seed, x0 + 1, y0 + 1, z0 + 1),
        dx - one,
        dy - one,
        dz - one,
    );
    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);
    let y0v = lerp(x00, x10, v);
    let y1v = lerp(x01, x11, v);
    lerp(y0v, y1v, w)
}

/// A hash's top bits as a value in `[0, 1)` at [`NOISE_BITS`]: a shift.
#[must_use]
pub const fn unit_value(h: u64) -> Gi {
    Gi::new((h >> (64 - NOISE_BITS)) as i64)
}

/// Value noise at a lattice point at [`NOISE_BITS`]: each corner draws one value in `[0, 1)` from
/// the hash, blended with the same fade in the same order. Cheaper than gradient noise; the cavern
/// field uses it.
#[must_use]
pub fn value3(seed: u64, p: [Gi; 3]) -> Gi {
    let x0 = (p[0] >> NOISE_BITS).raw();
    let y0 = (p[1] >> NOISE_BITS).raw();
    let z0 = (p[2] >> NOISE_BITS).raw();
    let u = fade(p[0] & FRAC_MASK);
    let v = fade(p[1] & FRAC_MASK);
    let w = fade(p[2] & FRAC_MASK);
    let corner =
        |dx: i64, dy: i64, dz: i64| unit_value(corner_hash(seed, x0 + dx, y0 + dy, z0 + dz));
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fade_and_the_blend_read_their_ends() {
        assert_eq!(fade(Gi::ZERO), Gi::ZERO);
        // t = 1 − 2⁻²⁸: the fade is one within a few units.
        let almost = NOISE_ONE - Gi::ONE;
        assert!((fade(almost) - NOISE_ONE).raw().abs() <= 8);
        // t = ½: the fade is exactly ½ (the quintic is symmetric about the centre).
        let half = NOISE_ONE >> 1;
        assert!((fade(half) - half).raw().abs() <= 2);
        assert_eq!(lerp(Gi::new(100), Gi::new(300), Gi::ZERO), Gi::new(100));
        assert_eq!(lerp(Gi::new(100), Gi::new(300), half), Gi::new(200));
    }

    #[test]
    fn gradient_noise_is_zero_on_the_lattice_and_bounded_between() {
        let seed = 0x5EED;
        // On a lattice corner every offset is zero, so every dot is zero.
        assert_eq!(noise3(seed, [Gi::ZERO, Gi::ZERO, Gi::ZERO]), Gi::ZERO);
        assert_eq!(
            noise3(
                seed,
                [NOISE_ONE * Gi::new(7), NOISE_ONE * Gi::new(-3), NOISE_ONE]
            ),
            Gi::ZERO
        );
        // Between corners the value is bounded by the gradients' reach (|n| < 2) and depends on
        // the seed and the point.
        let bound = NOISE_ONE * Gi::new(2);
        let mut i = 1i64;
        let mut seen_nonzero = false;
        while i < 200 {
            let p = [
                Gi::new(i * 8_191_331),
                Gi::new(-i * 5_432_109),
                Gi::new(i * 3_141_593),
            ];
            let n = noise3(seed, p);
            assert!(n < bound, "{i}: {n:?}");
            assert!(n > Gi::ZERO - bound, "{i}: {n:?}");
            seen_nonzero |= n != Gi::ZERO;
            assert_eq!(n, noise3(seed, p));
            i += 1;
        }
        assert!(seen_nonzero);
        let p = [
            Gi::new(123_456_789),
            Gi::new(987_654_321),
            Gi::new(555_555_555),
        ];
        assert_ne!(noise3(seed, p), noise3(seed ^ 1, p));
        // A negative lattice point floors toward −∞ and reads its cube below.
        let below = [Gi::new(-1), Gi::ZERO, Gi::ZERO];
        assert_eq!((below[0] >> NOISE_BITS).raw(), -1);
        let _ = noise3(seed, below);
    }

    #[test]
    fn value_noise_stays_in_the_unit_interval_and_reads_the_hash() {
        assert_eq!(unit_value(u64::MAX), NOISE_ONE - Gi::ONE);
        assert_eq!(unit_value(0), Gi::ZERO);
        let seed = 77;
        let mut i = 0i64;
        while i < 100 {
            let p = [
                Gi::new(i * 7_777_777),
                Gi::new(i * 1_234_567),
                Gi::new(-i * 999_999),
            ];
            let v = value3(seed, p);
            assert!(v >= Gi::ZERO, "{i}: {v:?}");
            assert!(v < NOISE_ONE, "{i}: {v:?}");
            i += 1;
        }
        // On a corner the value is that corner's own draw.
        let corner = [
            NOISE_ONE * Gi::new(3),
            NOISE_ONE * Gi::new(4),
            NOISE_ONE * Gi::new(5),
        ];
        assert_eq!(value3(seed, corner), unit_value(corner_hash(seed, 3, 4, 5)));
    }
}
