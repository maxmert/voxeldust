//! ★ THE RECIPE'S UNITS, AND THE DOORS TO METRES (ruling F7: the recipe is integer-only).
//!
//! Every length the recipe holds is a whole number of GAP STEPS — 1/128 of a metre, which is the
//! density byte's own step at the one-metre rung — or a fixed-point fraction of one at the noise's
//! [`vd_recipe::noise::NOISE_BITS`] fraction bits. A radius, a surface, a cave's hollow and a cell
//! centre are all words in that unit, so the server and the client fold the same bytes on every chip
//! and on every GPU.
//!
//! **Metres are an EXIT, never an input to the shape.** A host outside the recipe — the client's mesh
//! buffer, a picture's caption, a collider's query — reads metres through [`metres_of_q28`],
//! [`metres_of_steps`] or [`metres_of_fixed`], and a direction through [`unit_of_direction`]; a float
//! query about an arbitrary point enters through [`q28_of_metres`] and [`direction_of_unit`]. THE
//! DOORS OF THIS MODULE ARE THE WHOLE FLOAT SURFACE of the recipe's path, and each of them is one
//! multiply or divide by a POWER OF TWO under the float fence ([`crate::gf::Gf`]) — which is exact —
//! around ONE unavoidable rounding: **a word wider than 53 bits does not fit a float's mantissa**. A
//! radius at [`LENGTH_BITS`] is about 2⁶⁰ at the home planet, so the door rounds it to the nearest
//! float — about four NANOMETRES of the 6 370 km radius, and never more than one part in 2⁵². The
//! shape never reads the rounded number back: a door's answer goes to a host that already works in
//! floats, and the word stays the truth. [`q28_of_metres`] saturates rather than wrapping on a float
//! too large for the word (the cast's own rule), so a nonsense query is a clamped number, never a
//! negative radius.
//!
//! **Example.** The ground under the pilot's boots stands at 815 406 723 gap steps — 6 370 365.02 m.
//! The shard compares that word with the boots' cell to know what they stand on; the client turns it
//! into a float only to hand a vertex to the graphics card.

use crate::gf::Gf;
use vd_recipe::Gi;
use vd_recipe::bend::DIR_BITS;
use vd_recipe::noise::NOISE_BITS;

/// Gap steps in one metre: the recipe's unit of length, 1/128 m (the leaf's own constant).
pub const STEPS_PER_M: i64 = vd_seed::ladder::STEPS_PER_M;

/// The fraction bits a length carries on the recipe's path: the noise's own, so a surface and an
/// octave sum add without a shift.
pub const LENGTH_BITS: u32 = NOISE_BITS;

/// One whole gap step at [`LENGTH_BITS`].
pub const STEP_ONE: Gi = Gi::new(1 << LENGTH_BITS);

/// Gap steps in one metre at [`LENGTH_BITS`]: the divisor the exit to metres uses, a power of two
/// (2³⁵), so the division is exact.
const STEPS_PER_M_Q: i64 = STEPS_PER_M << LENGTH_BITS;

/// THE EXIT TO METRES: a length in gap steps at [`LENGTH_BITS`], as a plain float.
#[must_use]
pub fn metres_of_q28(length: Gi) -> f64 {
    metres_of_fixed(length, LENGTH_BITS)
}

/// THE EXIT TO METRES for a length in gap steps at `bits` fraction bits — the vertex sum carries its
/// own count ([`crate::position::POSITION_BITS`]). The divisor is a power of two, so the DIVISION is
/// exact; the word itself is rounded to the nearest float where it is wider than a mantissa (the
/// module's own doc measures it).
#[must_use]
pub fn metres_of_fixed(length: Gi, bits: u32) -> f64 {
    (Gf::from_i64(length.raw()) / Gf::from_i64(STEPS_PER_M << bits)).to_f64()
}

/// THE EXIT TO METRES for a length in WHOLE gap steps.
#[must_use]
pub fn metres_of_steps(steps: i64) -> f64 {
    (Gf::from_i64(steps) / Gf::from_i64(STEPS_PER_M)).to_f64()
}

/// THE DOOR IN for a float query about a length in metres: the length in gap steps at
/// [`LENGTH_BITS`], floored. A host that asks "how far above the sea is this?" in metres gets the
/// recipe's own answer about the nearest word. A metre count too large for the word SATURATES at the
/// word's end (the cast's own rule), so a nonsense query never wraps into a negative radius.
#[must_use]
pub fn q28_of_metres(metres: f64) -> Gi {
    Gi::new((Gf::from_f64(metres) * Gf::from_i64(STEPS_PER_M_Q)).to_i64_floor())
}

/// THE DOOR IN for a float query about a DIRECTION: the three components at the bend's
/// [`vd_recipe::bend::DIR_BITS`] fraction bits, floored. The recipe's own directions come from the
/// bend and never through here; this is for a host asking about a point it holds as floats — the
/// ground under a camera, a ray from a pilot's boots.
#[must_use]
pub fn direction_of_unit(dir: [f64; 3]) -> [Gi; 3] {
    let one = Gf::from_i64(1 << DIR_BITS);
    [
        Gi::new((Gf::from_f64(dir[0]) * one).to_i64_floor()),
        Gi::new((Gf::from_f64(dir[1]) * one).to_i64_floor()),
        Gi::new((Gf::from_f64(dir[2]) * one).to_i64_floor()),
    ]
}

/// THE EXIT for a DIRECTION: the three components the recipe holds at the bend's
/// [`vd_recipe::bend::DIR_BITS`] fraction bits, as plain floats. The inverse of
/// [`direction_of_unit`], for a host outside the recipe — a camera's radial, an instrument's shadow
/// of the recipe. The divisor is a power of two, so the division is exact.
#[must_use]
pub fn unit_of_direction(dir: [Gi; 3]) -> [f64; 3] {
    let one = Gf::from_i64(1 << DIR_BITS);
    [
        (Gf::from_i64(dir[0].raw()) / one).to_f64(),
        (Gf::from_i64(dir[1].raw()) / one).to_f64(),
        (Gf::from_i64(dir[2].raw()) / one).to_f64(),
    ]
}

/// ★ THE GREATER AND THE LESSER ARE THE RECIPE'S (ruling F7, step G1): the recipe has no `max`
/// (the float fence's rule, kept for the integer path so one reading order holds everywhere), and
/// the comparison that stands in for it lives with the kernels that the GPU compiles. A tie keeps
/// the LEFT word, here and in a shader alike.
pub use vd_recipe::cell::{greater, lesser};

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
    fn the_exit_to_metres_and_the_doors_in_agree_on_the_unit() {
        // One metre is 128 gap steps; the exit divides by a power of two, so it is exact.
        assert_eq!(metres_of_steps(STEPS_PER_M), 1.0);
        assert_eq!(metres_of_steps(-64), -0.5);
        assert_eq!(metres_of_q28(STEP_ONE * Gi::new(STEPS_PER_M)), 1.0);
        assert_eq!(metres_of_q28(Gi::ZERO), 0.0);
        // The door in and the exit are each other's inverse on a whole step.
        let steps = 815_406_723i64;
        let q = q28_of_metres(metres_of_steps(steps));
        assert_eq!(q, Gi::new(steps) << LENGTH_BITS);
        assert_eq!(metres_of_q28(q), metres_of_steps(steps));
        // A direction the camera holds: the components land at 40 fraction bits.
        let d = direction_of_unit([1.0, -0.5, 0.0]);
        assert_eq!(d[0], Gi::new(1 << DIR_BITS));
        assert_eq!(d[1], Gi::new(-(1 << (DIR_BITS - 1))));
        assert_eq!(d[2], Gi::ZERO);
        assert_eq!(STEPS_PER_M_Q, 1 << 35);
        // The direction's two doors are each other's inverse.
        assert_eq!(unit_of_direction(d), [1.0, -0.5, 0.0]);
        assert_eq!(direction_of_unit(unit_of_direction(d)), d);
    }

    /// ★ THE ONE ROUNDING THE DOORS CARRY, MEASURED: a word wider than a float's 53-bit mantissa is
    /// rounded to the nearest float on its way out — never by more than one part in 2⁵², which is four
    /// nanometres of the home planet's radius — and a metre count too large for the word saturates on
    /// its way in rather than wrapping.
    #[test]
    #[allow(
        clippy::float_arithmetic,
        reason = "the test measures the door's own rounding against the exact real number"
    )]
    fn a_word_wider_than_a_mantissa_rounds_on_the_way_out_and_a_huge_float_saturates_on_the_way_in()
    {
        // The home planet's radius as the recipe holds it: 6 370 353.6 m at 35 fraction bits, which is
        // about 2⁶⁰ — seven bits past a mantissa.
        let radius = Gi::new(815_405_260i64 << LENGTH_BITS);
        assert!(
            radius.raw() > (1i64 << 53),
            "the word is wider than a mantissa"
        );
        let metres = metres_of_q28(radius);
        // The exact real number, from the word's own halves (each under 2⁵³, so each is exact).
        let exact = (radius.raw() >> 20) as f64 / f64::from(1u32 << 15);
        assert!(
            (metres - exact).abs() <= exact / f64::from(1u32 << 26) / f64::from(1u32 << 26),
            "the door rounds by at most one part in 2⁵²: {metres} vs {exact}"
        );
        assert!((metres - 6_370_353.6).abs() < 1.0, "{metres}");
        // A word that FITS a mantissa is exact to the last bit.
        assert_eq!(metres_of_q28(Gi::new(3 << LENGTH_BITS)), 3.0 / 128.0);
        // THE DOOR IN SATURATES: a metre count past the word's end clamps at the largest word, it does
        // not wrap into a negative radius.
        assert_eq!(q28_of_metres(1.0e300), Gi::new(i64::MAX));
        assert_eq!(q28_of_metres(-1.0e300), Gi::new(i64::MIN));
    }

    #[test]
    fn the_greater_and_the_lesser_are_comparisons_and_keep_the_left_word_on_a_tie() {
        assert_eq!(greater(Gi::new(3), Gi::new(7)), Gi::new(7));
        assert_eq!(greater(Gi::new(7), Gi::new(3)), Gi::new(7));
        assert_eq!(greater(Gi::new(3), Gi::new(3)), Gi::new(3));
        assert_eq!(lesser(Gi::new(3), Gi::new(7)), Gi::new(3));
        assert_eq!(lesser(Gi::new(7), Gi::new(3)), Gi::new(3));
        assert_eq!(lesser(Gi::new(3), Gi::new(3)), Gi::new(3));
    }
}
