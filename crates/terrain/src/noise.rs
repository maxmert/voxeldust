//! ★ THE NOISE — gradient noise on the integer hash, under the fence (ruling V6 Part D: ~400 vendored
//! lines on our own hash inside the fenced float type; no external noise crate).
//!
//! A point in space lands in a unit cube of the lattice. Each of the cube's eight corners draws a
//! gradient direction from the hash of `(seed, corner)`, so the same corner gives the same gradient on
//! every host for all time. The value is the smooth blend of the eight corner contributions, with the
//! quintic fade `t³(t(6t − 15) + 10)`, a polynomial. Every draw is ONE round of the integer hash
//! (`SplitMix64`, the same avalanche the forest uses) over the corner's folded key; every float step
//! is add, subtract, multiply, divide.
//!
//! **Example.** The hill field asks for the noise at the direction of cell (face 2, 1181, 77) times
//! octave 3's frequency. The point falls in lattice cube (417, −88, 1203); its eight corners' gradients
//! come from the hash; the answer is the same number on the Mac and on the pod.

use crate::gf::Gf;
use vd_seed::rng::SplitMix64;

/// The sixteen gradient directions (Perlin's twelve cube edges, four repeated so a four-bit draw
/// selects one). Every component is −1, 0 or 1, so a dot product is adds and subtracts only.
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

/// The three odd multipliers that fold a corner's coordinates into one key before the avalanche
/// (SplitMix64's own constants, so no second hash enters the tree).
const FOLD_X: u64 = 0x9E37_79B9_7F4A_7C15;
const FOLD_Y: u64 = 0xBF58_476D_1CE4_E5B9;
const FOLD_Z: u64 = 0x94D0_49BB_1331_11EB;

/// The hash of one lattice corner: the seed and the three coordinates folded into one key, then ONE
/// round of the integer hash.
#[must_use]
pub fn corner_hash(seed: u64, x: i64, y: i64, z: i64) -> u64 {
    let key = seed
        ^ (x as u64).wrapping_mul(FOLD_X)
        ^ (y as u64).wrapping_mul(FOLD_Y)
        ^ (z as u64).wrapping_mul(FOLD_Z);
    SplitMix64::new(key).next_u64()
}

/// The gradient a corner draws, as fenced floats.
fn gradient(seed: u64, x: i64, y: i64, z: i64) -> [Gf; 3] {
    let g = GRADIENTS[(corner_hash(seed, x, y, z) & 15) as usize];
    [
        Gf::from_i32(i32::from(g[0])),
        Gf::from_i32(i32::from(g[1])),
        Gf::from_i32(i32::from(g[2])),
    ]
}

/// The quintic fade: `t³(t(6t − 15) + 10)`, Horner form, a polynomial and nothing else.
#[must_use]
pub fn fade(t: Gf) -> Gf {
    let six = Gf::from_i64(6);
    let fifteen = Gf::from_i64(15);
    let ten = Gf::from_i64(10);
    t * t * t * (t * (t * six - fifteen) + ten)
}

/// `a + t(b − a)`: one multiply, two adds, in that order.
#[must_use]
pub fn lerp(a: Gf, b: Gf, t: Gf) -> Gf {
    a + t * (b - a)
}

fn dot(g: [Gf; 3], dx: Gf, dy: Gf, dz: Gf) -> Gf {
    g[0] * dx + g[1] * dy + g[2] * dz
}

/// Gradient noise at a point, in about `[−1, 1]`. The point's lattice cube is its floor; the eight
/// corners are blended in ONE fixed order (x pairs, then y pairs, then z).
#[must_use]
pub fn noise3(seed: u64, p: [Gf; 3]) -> Gf {
    let fx = p[0].floor();
    let fy = p[1].floor();
    let fz = p[2].floor();
    let x0 = fx.to_i64_floor();
    let y0 = fy.to_i64_floor();
    let z0 = fz.to_i64_floor();
    let dx = p[0] - fx;
    let dy = p[1] - fy;
    let dz = p[2] - fz;
    let u = fade(dx);
    let v = fade(dy);
    let w = fade(dz);
    let one = Gf::ONE;
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

/// Value noise: each corner draws one value in `[0, 1)` from the hash, blended with the same fade.
/// Cheaper than gradient noise; the cavern field uses it.
#[must_use]
pub fn value3(seed: u64, p: [Gf; 3]) -> Gf {
    let fx = p[0].floor();
    let fy = p[1].floor();
    let fz = p[2].floor();
    let x0 = fx.to_i64_floor();
    let y0 = fy.to_i64_floor();
    let z0 = fz.to_i64_floor();
    let u = fade(p[0] - fx);
    let v = fade(p[1] - fy);
    let w = fade(p[2] - fz);
    let corner =
        |dx: i64, dy: i64, dz: i64| unit_value(corner_hash(seed, x0 + dx, y0 + dy, z0 + dz));
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

/// A hash's top 53 bits as a float in `[0, 1)`: exact, one division by a power of two.
#[must_use]
pub fn unit_value(h: u64) -> Gf {
    Gf::from_i64((h >> 11) as i64) / Gf::from_i64(1 << 53)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64, z: f64) -> [Gf; 3] {
        [Gf::from_f64(x), Gf::from_f64(y), Gf::from_f64(z)]
    }

    #[test]
    fn the_fade_is_a_smooth_step_and_the_lerp_hits_its_ends() {
        assert_eq!(fade(Gf::ZERO), Gf::ZERO);
        assert_eq!(fade(Gf::ONE), Gf::ONE);
        assert_eq!(
            fade(Gf::HALF),
            Gf::HALF,
            "the quintic is symmetric about the middle"
        );
        assert_eq!(
            lerp(Gf::from_i64(3), Gf::from_i64(7), Gf::ZERO),
            Gf::from_i64(3)
        );
        assert_eq!(
            lerp(Gf::from_i64(3), Gf::from_i64(7), Gf::ONE),
            Gf::from_i64(7)
        );
        assert_eq!(unit_value(0), Gf::ZERO);
        assert!(unit_value(u64::MAX) < Gf::ONE);
    }

    #[test]
    fn the_noise_is_zero_on_the_lattice_bounded_and_deterministic() {
        // On a lattice corner every offset is zero, so every contribution is zero.
        assert_eq!(noise3(7, p(3.0, -2.0, 5.0)), Gf::ZERO);
        let mut lo = Gf::ZERO;
        let mut hi = Gf::ZERO;
        let mut i = 0;
        while i < 4_000 {
            let x = Gf::from_i64(i) * Gf::from_f64(0.173);
            let y = Gf::from_i64(i) * Gf::from_f64(-0.091);
            let z = Gf::from_i64(i) * Gf::from_f64(0.047);
            let n = noise3(7, [x, y, z]);
            assert!(n.is_finite());
            lo = lo.lesser(n);
            hi = hi.greater(n);
            assert_eq!(n, noise3(7, [x, y, z]), "the same point, the same number");
            assert!(value3(7, [x, y, z]) >= Gf::ZERO);
            assert!(value3(7, [x, y, z]) < Gf::ONE);
            i += 1;
        }
        assert!(lo < Gf::from_f64(-0.3), "the field reaches down: {lo:?}");
        assert!(hi > Gf::from_f64(0.3), "the field reaches up: {hi:?}");
        assert!(lo >= -Gf::ONE);
        assert!(hi <= Gf::ONE);
        assert_ne!(
            noise3(7, p(0.3, 0.3, 0.3)),
            noise3(8, p(0.3, 0.3, 0.3)),
            "the seed matters"
        );
        // Every gradient is drawn: the sixteen entries all appear across many corners.
        let mut seen = [false; 16];
        let mut c = 0;
        while c < 2_000 {
            seen[(corner_hash(1, c, c * 3, c * 7) & 15) as usize] = true;
            c += 1;
        }
        assert!(
            seen.iter().all(|s| *s),
            "every gradient direction is reachable"
        );
        // Neighbouring corners never share a hash by construction of the fold (a spot check).
        assert_ne!(corner_hash(1, 0, 0, 0), corner_hash(1, 1, 0, 0));
        assert_ne!(corner_hash(1, 0, 0, 0), corner_hash(1, 0, 1, 0));
        assert_ne!(corner_hash(1, 0, 0, 0), corner_hash(1, 0, 0, 1));
    }

    #[test]
    fn a_known_vector_pins_the_noise_forever() {
        // The composition of the hash and the blend is part of the world identity (Format D): this
        // literal is the bit pattern the home world's hills rest on.
        assert_eq!(noise3(2298, p(0.25, 0.5, 0.75)).to_bits(), NOISE_PIN);
        assert_eq!(value3(2298, p(0.25, 0.5, 0.75)).to_bits(), VALUE_PIN);
    }

    const NOISE_PIN: u64 = 13_827_097_110_060_728_320;
    const VALUE_PIN: u64 = 4_601_792_929_522_320_944;
}
