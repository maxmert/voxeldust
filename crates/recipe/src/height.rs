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
use crate::noise::{NOISE_BITS, NOISE_ONE, noise3};

/// The direction's fraction bits inside the noise: the bend's 40 shifted down to 30.
pub const SAMPLE_BITS: u32 = 30;
/// The amplitude's fraction bits below the gap step (1/128 of a cell): 1/32 768 m at the metre rung.
pub const AMP_BITS: u32 = 8;
/// The gap steps in one cell — the density byte's unit, pinned against `vd_core`'s registry.
pub const GAP_STEPS_PER_CELL: i64 = 128;
/// The octave table's cap: a body draws at most this many octaves, and a GPU shell holds them in
/// a fixed-size array (a shader has no heap and no runtime-length slice of a local array).
pub const OCTAVES_CAP: usize = 16;

/// One octave of the height field in the recipe's formats. `repr(C)`: four words in this order,
/// so a GPU reads a table of them in place from a buffer the CPU filled with the same bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
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

/// ★ THE BIOME FIELD's own numbers (ruling F7: the kernel names no substance and no biome; it
/// answers a CODE the host gives meaning to). `repr(C)`, twelve words, so a card reads the row the
/// host wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct BiomeCharter {
    /// The sea's radius, in gap steps at the length format.
    pub sea_radius: Gi,
    /// Above this height over the sea a column is HIGHLAND whatever its climate.
    pub highland_above: Gi,
    /// `floor(2^b / (highland_above + one metre))`, so the height share is one multiply.
    pub highland_recip: Gi,
    /// The shift that takes the reciprocal's own bits down to [`NOISE_BITS`].
    pub highland_shift: Gi,
    /// The two slow noises, each an octave of unit amplitude.
    pub temperature: Octave,
    pub humidity: Octave,
}

/// The biome CODE of a hot dry column.
pub const BIOME_DESERT: Gi = Gi::new(0);
/// The biome CODE of a temperate column.
pub const BIOME_GRASSLAND: Gi = Gi::new(1);
/// The biome CODE of a cold column.
pub const BIOME_TUNDRA: Gi = Gi::new(2);
/// The biome CODE of a column far above the sea.
pub const BIOME_HIGHLAND: Gi = Gi::new(3);

/// The pole component of a direction: `+Z` in the body's own frame — the axis the world's orbits
/// turn about, so ice caps face away from the orbital plane, never into it.
pub const POLE_AXIS: usize = 2;

/// The climate constants, at the noise's fraction bits: how much of the slow temperature noise the
/// climate reads, how much height cools a column, and the three thresholds the biomes stand on.
/// Each is the share times `2^28`, rounded once.
const NOISE_SHARE: Gi = Gi::new(93_952_410); // 0.35
const HEIGHT_SHARE: Gi = Gi::new(80_530_637); // 0.30
const COLD_BELOW: Gi = Gi::new(93_952_410); // 0.35
const WARM_ABOVE: Gi = Gi::new(201_326_592); // 0.75
const DRY_BELOW: Gi = Gi::new(-26_843_546); // -0.10

/// The length format's fraction bits — the same as the noise's, which is why a surface radius and a
/// relief add without a shift.
const LENGTH_BITS: u32 = NOISE_BITS;

/// ★ THE BIOME OF A COLUMN: cold near the poles and high up, dry where the humidity noise says so,
/// and highland where the surface stands far above the sea. `surface` is the column's surface
/// radius in gap steps at the length format — the radius plus [`relief`].
///
/// **Example.** The column under the pilot's boots stands forty metres over the sea at a latitude of
/// a third: its temperature reads over the cold threshold and under the warm one, so the kernel
/// answers grassland and the cell pass reads the grassland row of the charter's strata.
#[must_use]
pub fn biome_of(charter: &BiomeCharter, dir: [Gi; 3], surface: Gi) -> Gi {
    let above_sea = surface - charter.sea_radius;
    if above_sea > charter.highland_above {
        return BIOME_HIGHLAND;
    }
    // The latitude: the pole component's magnitude, at the noise's fraction bits.
    let latitude = Gi::new(dir[POLE_AXIS].unsigned_abs() as i64) >> (DIR_BITS - NOISE_BITS);
    let t_noise = one_octave(&charter.temperature, dir);
    // The height share: the gap steps over the sea against the highland height, one multiply by the
    // charter's reciprocal. Only a column above the sea is cooled by its height. The reciprocal
    // carries its own bits, so the shift leaves the share at the noise's.
    let over_sea = if above_sea > Gi::ZERO {
        above_sea
    } else {
        Gi::ZERO
    };
    let height_share = (over_sea >> LENGTH_BITS)
        .mul_shr(charter.highland_recip, charter.highland_shift.raw() as u32);
    // Warm at the equator, cold at the poles, plus a slow noise; cooler with height.
    let temperature = NOISE_ONE - latitude + t_noise.mul_shr(NOISE_SHARE, NOISE_BITS)
        - height_share.mul_shr(HEIGHT_SHARE, NOISE_BITS);
    if temperature < COLD_BELOW {
        return BIOME_TUNDRA;
    }
    let humidity = one_octave(&charter.humidity, dir);
    if (temperature > WARM_ABOVE) & (humidity < DRY_BELOW) {
        return BIOME_DESERT;
    }
    BIOME_GRASSLAND
}

/// ONE octave's own relief along a direction — a slow climate noise is a single octave, and this is
/// the sum's own body without its loop.
fn one_octave(o: &Octave, dir: [Gi; 3]) -> Gi {
    octave_term(o, sample_direction(dir))
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

    /// A biome charter with no climate noise at all, so a test names a column's temperature by its
    /// latitude and its height alone. The sea stands at a thousand gap steps.
    fn biome_charter() -> BiomeCharter {
        let quiet = Octave {
            seed: 1,
            frequency_int: Gi::ONE,
            frequency_frac: Gi::ZERO,
            amplitude: Gi::ZERO,
        };
        BiomeCharter {
            sea_radius: Gi::new(1_000) << LENGTH_BITS,
            highland_above: Gi::new(400) << LENGTH_BITS,
            // One over (the highland height in gap steps plus a metre), at 62 bits — through the
            // recipe's own reciprocal, because the crate denies a bare divide even in a test.
            highland_recip: Gi::new(crate::root::recip_pow2(401, 62) as i64),
            highland_shift: Gi::new(i64::from(62 - NOISE_BITS)),
            temperature: quiet,
            humidity: quiet,
        }
    }

    /// ★ EVERY BIOME THE KERNEL CAN ANSWER, and the two things that cool a column. The climate
    /// noises are silent here, so each arm is driven by the latitude, the height and the thresholds
    /// alone — a biome that moved would be this test going red, not a hill that looked odd.
    #[test]
    fn the_biome_kernel_reads_the_height_the_latitude_and_the_two_noises() {
        let c = biome_charter();
        let equator = [DIR_ONE, Gi::ZERO, Gi::ZERO];
        let pole = [Gi::ZERO, Gi::ZERO, DIR_ONE];
        // Far above the sea is HIGHLAND whatever the climate.
        assert_eq!(
            biome_of(&c, equator, c.sea_radius + c.highland_above + Gi::ONE),
            BIOME_HIGHLAND
        );
        // At the pole the latitude alone takes the temperature under the cold threshold.
        assert_eq!(biome_of(&c, pole, c.sea_radius), BIOME_TUNDRA);
        // At the equator, at the sea, with a silent humidity noise of zero — which is above the dry
        // threshold — the column is GRASSLAND.
        assert_eq!(biome_of(&c, equator, c.sea_radius), BIOME_GRASSLAND);
        // ★ A DRY HUMIDITY MAKES A DESERT. The gradient noise is ZERO on a lattice point, so the
        // equator's own axis reads no humidity at all whatever the amplitude; a desert wants a
        // direction BETWEEN lattice points where the noise runs negative. The scan finds one, and
        // the noise is a function of the seed, so it finds the same one every run.
        let mut dry = c;
        dry.humidity.amplitude = Gi::new(1 << (AMP_BITS + 6));
        dry.humidity.frequency_int = Gi::new(3);
        let mut deserts = 0;
        let mut grasslands = 0;
        let mut k = 1i64;
        while k < 200 {
            // A direction near the equator, tilted a little off the axis each step.
            let d = crate::bend::normalise([DIR_ONE, Gi::new(k) << (DIR_BITS - 12), Gi::ZERO]);
            let b = biome_of(&dry, d, dry.sea_radius);
            deserts += i32::from(b == BIOME_DESERT);
            grasslands += i32::from(b == BIOME_GRASSLAND);
            k += 1;
        }
        assert_eq!(deserts + grasslands, 199, "a warm equator is never cold");
        assert!(deserts > 0, "the equator holds a dry column somewhere");
        assert!(grasslands > 0, "and a wet one");
        // ★ HEIGHT COOLS A COLUMN: the same direction just under the highland height reads colder
        // than at the sea. The share read ZERO once, when the reciprocal's shift left it a whole
        // number, and every column then ignored its own height.
        let low = biome_of(&c, pole, c.sea_radius - Gi::ONE);
        let high = biome_of(&c, pole, c.sea_radius + c.highland_above - Gi::ONE);
        assert_eq!(low, BIOME_TUNDRA);
        assert_eq!(high, BIOME_TUNDRA);
        // A column UNDER the sea is not cooled by its height: the share is clamped at zero.
        assert_eq!(
            biome_of(&c, equator, c.sea_radius - (Gi::new(300) << LENGTH_BITS)),
            BIOME_GRASSLAND
        );
        // The codes are the four the host names, and no two are the same.
        let all = [BIOME_DESERT, BIOME_GRASSLAND, BIOME_TUNDRA, BIOME_HIGHLAND];
        let mut i = 0;
        while i < all.len() {
            assert_eq!(all[i], Gi::new(i as i64));
            i += 1;
        }
        assert_eq!(POLE_AXIS, 2);
    }

    /// The climate constants are the shares the recipe reads, rounded once to the noise's bits.
    #[test]
    fn the_climate_constants_are_the_shares_at_the_noises_bits() {
        #[allow(
            clippy::float_arithmetic,
            reason = "the test states each share as a real number"
        )]
        let one = f64::from(1u32 << NOISE_BITS);
        for (word, share) in [
            (NOISE_SHARE, 0.35),
            (HEIGHT_SHARE, 0.30),
            (COLD_BELOW, 0.35),
            (WARM_ABOVE, 0.75),
            (DRY_BELOW, -0.10),
        ] {
            #[allow(
                clippy::float_arithmetic,
                reason = "the test states each share as a real number"
            )]
            let want = (share * one).round() as i64;
            assert_eq!(word.raw(), want, "{share}");
        }
        assert_eq!(LENGTH_BITS, NOISE_BITS);
    }
}
