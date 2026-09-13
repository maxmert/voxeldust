//! THE OCTAVE SUM — the relief along a direction: the live octaves of gradient noise, each at its
//! own frequency, amplitude and seed, summed at the noise's fraction bits and floored ONCE.
//!
//! An octave's frequency (the body's radius over the octave's wavelength: up to 2.1 × 10⁵ cells per
//! unit direction on the home planet) is stored as an INTEGER PART and a FRACTION at [`NOISE_BITS`],
//! so the lattice point is two products of the direction at 30 fraction bits: `d × int` (exact) and
//! `d × frac >> 28`, summed at 30 bits and shifted to the noise's 28. MEASURED (bench part 1): a
//! frequency rounded to 2⁻⁸ moved the coarsest lattice point by a ten-thousandth of a cell, which
//! eight kilometres of amplitude turned into metres; the octave products floored to whole gap steps
//! lost up to a step each, 57 mm over fourteen; an amplitude in whole gap steps cost 3.9 mm an
//! octave. With the frequency exact, the amplitude at [`AMP_BITS`] below the gap step and one floor
//! at the end, the sum sits within 1.06 mm of the float recipe on four million columns.
//!
//! The direction the noise samples is the 40-bit direction shifted to 30 bits: a lateral step of
//! 5.9 mm on the home planet, under the finest octave's wavelength by four orders, and the cell's
//! own centre keeps the 40-bit direction ([`crate::bend`]).

use crate::bend::DIR_BITS;
use crate::gi::Gi;
use crate::noise::{NOISE_BITS, noise3};

/// The direction's fraction bits inside the noise: the bend's 40 shifted down to 30.
pub const SAMPLE_BITS: u32 = 30;
/// The amplitude's fraction bits below the gap step (1/128 of a cell): 1/32 768 m at the metre rung.
pub const AMP_BITS: u32 = 8;
/// The gap steps in one cell — the density byte's unit, pinned against `vd_core`'s registry.
pub const GAP_STEPS_PER_CELL: i64 = 128;
/// The octave table's cap: a body draws at most this many octaves, and a GPU shell holds them in
/// a fixed-size array (a shader has no heap and no runtime-length slice of a local array).
pub const OCTAVES_CAP: usize = 16;

/// One octave of the height field in the recipe's formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Octave {
    /// The octave's own noise seed.
    pub seed: u64,
    /// The frequency's integer part: cells per unit direction.
    pub frequency_int: Gi,
    /// The frequency's fraction at [`NOISE_BITS`].
    pub frequency_frac: Gi,
    /// The amplitude in gap steps at [`AMP_BITS`]: metres × 128 × 256 at the metre rung.
    pub amplitude: Gi,
}

/// The lattice point of one direction component for one octave, at [`NOISE_BITS`].
fn lattice(d30: Gi, o: &Octave) -> Gi {
    (d30 * o.frequency_int + ((d30 * o.frequency_frac) >> NOISE_BITS)) >> (SAMPLE_BITS - NOISE_BITS)
}

/// The relief along a 40-bit direction over `octaves`, in gap steps at [`NOISE_BITS`], UNFLOORED:
/// the caller adds the radius in the same unit and floors once to the gap byte. An index loop,
/// never an iterator: the GPU compiler (rust-gpu) refuses an iterator's pointer arithmetic.
#[must_use]
pub fn relief(octaves: &[Octave], dir: [Gi; 3]) -> Gi {
    let d = sample_direction(dir);
    let mut h = Gi::ZERO;
    let mut k = 0;
    while k < octaves.len() {
        h += octave_term(&octaves[k], d);
        k += 1;
    }
    h
}

/// [`relief`] over the first `count` octaves of a fixed-size table — the form a GPU shell calls,
/// because a shader holds its octaves in a local array and cannot slice it to a runtime length.
/// A `count` past the table reads the whole table.
#[must_use]
pub fn relief_of_table(octaves: &[Octave; OCTAVES_CAP], count: usize, dir: [Gi; 3]) -> Gi {
    let d = sample_direction(dir);
    let n = if count < OCTAVES_CAP {
        count
    } else {
        OCTAVES_CAP
    };
    let mut h = Gi::ZERO;
    let mut k = 0;
    while k < n {
        h += octave_term(&octaves[k], d);
        k += 1;
    }
    h
}

/// The 40-bit direction shifted to the noise's sample bits.
fn sample_direction(dir: [Gi; 3]) -> [Gi; 3] {
    [
        dir[0] >> (DIR_BITS - SAMPLE_BITS),
        dir[1] >> (DIR_BITS - SAMPLE_BITS),
        dir[2] >> (DIR_BITS - SAMPLE_BITS),
    ]
}

/// One octave's term of the sum.
fn octave_term(o: &Octave, d: [Gi; 3]) -> Gi {
    let p = [lattice(d[0], o), lattice(d[1], o), lattice(d[2], o)];
    (o.amplitude * noise3(o.seed, p)) >> AMP_BITS
}

/// A relief at [`NOISE_BITS`] floored to whole gap steps.
#[must_use]
pub fn to_steps(relief: Gi) -> Gi {
    relief >> NOISE_BITS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bend::DIR_ONE;
    use crate::noise::NOISE_ONE;

    fn octave(seed: u64, frequency: f64, amplitude_m: f64) -> Octave {
        #[allow(
            clippy::float_arithmetic,
            reason = "the test states its octave in real numbers"
        )]
        let (fi, ff, a) = (
            frequency.floor() as i64,
            ((frequency - frequency.floor()) * f64::from(1u32 << NOISE_BITS)).round() as i64,
            (amplitude_m * 128.0 * 256.0).round() as i64,
        );
        Octave {
            seed,
            frequency_int: Gi::new(fi),
            frequency_frac: Gi::new(ff),
            amplitude: Gi::new(a),
        }
    }

    #[test]
    fn the_lattice_point_is_the_direction_times_the_frequency() {
        // d = 1, f = 16.5: p = 16.5 at 28 bits.
        let o = octave(1, 16.5, 1.0);
        let one30 = Gi::ONE << SAMPLE_BITS;
        assert_eq!(lattice(one30, &o), Gi::new(33) * (NOISE_ONE >> 1));
        // d = −½, f = 16.5: p = −8.25.
        let p = lattice(Gi::ZERO - (one30 >> 1), &o);
        assert_eq!(p, Gi::ZERO - Gi::new(33) * (NOISE_ONE >> 2));
    }

    #[test]
    fn the_relief_sums_the_octaves_and_floors_once() {
        let octaves = [octave(1, 15.9259, 8011.2287), octave(2, 31.8518, 4111.0007)];
        let dir = [DIR_ONE, Gi::ZERO, Gi::ZERO];
        let h = relief(&octaves, dir);
        // Bounded by the amplitudes' sum (the noise is under 2 in magnitude — under 1 in practice).
        let bound =
            ((octaves[0].amplitude + octaves[1].amplitude) * Gi::new(2)) << (NOISE_BITS - AMP_BITS);
        assert!(h < bound);
        assert!(h > Gi::ZERO - bound);
        // Each octave alone sums to the whole.
        let h0 = relief(&octaves[..1], dir);
        let h1 = relief(&octaves[1..], dir);
        assert_eq!(h, h0 + h1);
        assert_eq!(relief(&[], dir), Gi::ZERO);
        // The floor to steps.
        assert_eq!(to_steps(NOISE_ONE * Gi::new(5) + Gi::ONE), Gi::new(5));
        assert_eq!(to_steps(Gi::ZERO - Gi::ONE), Gi::new(-1));
        // A direction change changes the relief.
        assert_ne!(h, relief(&octaves, [Gi::ZERO, DIR_ONE, Gi::ZERO]));
        // The table form reads the same words: the first two of a full table, and a count past
        // the table reads the whole table.
        let mut table = [octaves[0]; OCTAVES_CAP];
        table[1] = octaves[1];
        assert_eq!(relief_of_table(&table, 2, dir), h);
        assert_eq!(relief_of_table(&table, 0, dir), Gi::ZERO);
        assert_eq!(relief_of_table(&table, 99, dir), relief(&table, dir));
    }
}
