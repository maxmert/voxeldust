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
use crate::noise::{NOISE_BITS, NOISE_ONE, fade, noise3};

/// The direction's fraction bits inside the noise: the bend's 40 shifted down to 30.
pub const SAMPLE_BITS: u32 = 30;
/// The amplitude's fraction bits below the gap step (1/128 of a cell): 1/32 768 m at the metre rung.
pub const AMP_BITS: u32 = 8;
/// The gap steps in one cell — the density byte's unit, pinned against `vd_core`'s registry.
pub const GAP_STEPS_PER_CELL: i64 = 128;
/// The octave table's cap: a body draws at most this many octaves, and a GPU shell holds them in
/// a fixed-size array (a shader has no heap and no runtime-length slice of a local array).
pub const OCTAVES_CAP: usize = 16;

/// ★ AN OCTAVE IS SMOOTH: its term is the gradient noise itself. The word is ZERO, so the mask in
/// [`octave_term`] selects nothing (slice 8a stage 2; `slice_8a_design.md` §2.2).
pub const OCTAVE_SMOOTH: Gi = Gi::new(0);
/// ★ AN OCTAVE IS RIDGED: its term is `1 − |n|`, re-centred, so a crest is a LINE and not a lump —
/// which is what a spur and a gully are. The word is ALL ONES, so the mask selects the whole
/// difference. It is a MASK and never a flag: a kernel must not branch on it
/// (`slice_8a_design.md` §2.2, and the two faults §2.0 records — a value a branch assigns inside a
/// loop read one step stale on the card).
pub const OCTAVE_RIDGED: Gi = Gi::new(-1);

/// ★ AN OCTAVE IS COARSE: the per-column roughness factor does not touch it (8c's macro solve
/// replaces the coarse octaves whole). The word is ZERO, so the mask in [`relief_shaped`] selects
/// nothing (slice 8a stage 3).
pub const OCTAVE_COARSE: Gi = Gi::new(0);
/// ★ AN OCTAVE IS FINE: the per-column roughness factor multiplies it. The word is ALL ONES, so the
/// mask selects the whole term. It is a MASK and never a flag, for the reason `octave_term` states
/// and `relief_shaped` MEASURED: an index compared against a split point inside the sum's loop read
/// wrong on the card.
pub const OCTAVE_FINE: Gi = Gi::new(-1);

/// One octave of the height field in the recipe's formats. `repr(C)`: EIGHT words in this order, so
/// a GPU reads a table of them in place from a buffer the CPU filled with the same bytes. Five words
/// carry the octave and three are explicit padding, which puts the stride at 64 bytes — a power of
/// two the card's addressing likes (slice 8a stage 2).
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
    /// [`OCTAVE_SMOOTH`] or [`OCTAVE_RIDGED`] — an ARITHMETIC MASK, never a flag.
    pub kind: Gi,
    /// [`OCTAVE_COARSE`] or [`OCTAVE_FINE`] — an ARITHMETIC MASK, never a flag (slice 8a stage 3).
    /// The per-column roughness factor multiplies the FINE octaves and no others, and the sum picks
    /// them out with this word rather than by comparing an index against a split point.
    pub fine: Gi,
    /// EXPLICIT PADDING to eight words. No kernel reads it; it stands so the row's stride is 64
    /// bytes on every host and the card reads the row the CPU wrote.
    pub pad: [Gi; 2],
}

impl Octave {
    /// An octave of the four words that describe it, SMOOTH, with its padding zeroed — the shape
    /// every caller had before the `kind` word existed.
    #[must_use]
    pub const fn smooth(seed: u64, frequency_int: Gi, frequency_frac: Gi, amplitude: Gi) -> Octave {
        Octave {
            seed,
            frequency_int,
            frequency_frac,
            amplitude,
            kind: OCTAVE_SMOOTH,
            fine: OCTAVE_COARSE,
            pad: [Gi::ZERO; 2],
        }
    }
}

/// ★ THE PER-COLUMN ROUGHNESS FACTOR's own words — slice 8a stage 3 (`slice_8a_design.md` §1.2 and
/// §2.3; `03_erosion_rivers.md` §4.2.4; ruling V13 L5). ONE downward-only multiplier per column on
/// the FINE octaves alone. It is what makes a plain flat and a range rough out of ONE table:
///
/// ```text
///    r = (one slow octave + 1) / 2                in [0, 1)
///    m = m_min + (1 − m_min) · fade(r)            in [m_min, 1], DOWNWARD ONLY
///    a(o, column) = a(o) · m                      the FINE octaves only
/// ```
///
/// ★ **ONE MULTIPLY PER COLUMN, NOT PER OCTAVE.** `m` has no octave argument, so
/// `Σ_fine a·m·n = m · Σ_fine a·n` and the sum splits at [`Roughness::first_fine`] and multiplies
/// ONCE. Sixteen multiplies become one.
///
/// ★ **DOWNWARD ONLY, WHICH IS WHAT LETS 8c SWAP IT.** `m ≤ 1` always, so the ladder's band — sized
/// on `Σ|a|` at `m = 1` — holds for this placeholder AND for 8c's macro field, which is bounded
/// above by one in the same way. 8c replaces [`Roughness::octave`] with its own field's gradient,
/// bumps the generator's version, and moves not one address.
///
/// `repr(C)`: ten words, so a card reads the row the host wrote.
///
/// **Example.** The pilot walks a craton: its factor reads 0.06, so the fine octaves that make a
/// hillside are a sixteenth of what they are on the orogen a thousand kilometres away, and the
/// plain is a plain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Roughness {
    /// ★ THE PLACEHOLDER (`slice_8a_design.md` §1.2): ONE MORE SLOW OCTAVE of the recipe's own
    /// noise, SMOOTH, of unit amplitude, drawn from its own salt at a continental wavelength — the
    /// band the biome noises already use. 8c replaces this one member with the macro field's own
    /// slope and moves no address, because the band is derived at `m = 1` and both fields obey that
    /// same ceiling.
    pub octave: Octave,
    /// The factor's FLOOR at [`NOISE_BITS`]: what share of a range's fine slopes a craton keeps.
    pub m_min: Gi,
    /// THE INDEX THE FINE BAND BEGINS AT, as the DRAW recorded it. A host reads it (the column
    /// bound's own drift term does); the KERNEL does not — it reads each octave's own
    /// [`Octave::fine`] mask instead, and a test asserts the two agree.
    pub first_fine: Gi,
}

/// The factor from a raw reading in `[0, 1)`: `m = m_min + (1 − m_min)·fade(r)` at [`NOISE_BITS`].
/// Both factors of the product stand under 2²⁸, so a plain multiply and one shift are exact enough
/// — the error is one unit at 28 bits of a gap step, 2.9 × 10⁻¹¹ m (`slice_8a_design.md` §2.8).
fn factor_of(m_min: Gi, r_raw: Gi) -> Gi {
    m_min + (((NOISE_ONE - m_min) * fade(r_raw)) >> NOISE_BITS)
}

/// The raw reading of the roughness field at one column: the slow octave mapped from `[−1, 1]` to
/// `[0, 1)`. `d` is the direction already at the sample bits.
fn roughness_raw(rough: &Roughness, d: [Gi; 3]) -> Gi {
    (octave_term(&rough.octave, d) + NOISE_ONE) >> 1
}

/// ★ THE ROUGHNESS FACTOR OF ONE COLUMN, in `[m_min, 1]` at [`NOISE_BITS`] — the number
/// [`relief_shaped`] multiplies its fine half by, named on its own so a host can measure the
/// planet's own histogram of it (the slope histogram M-C).
#[must_use]
pub fn roughness_factor(rough: &Roughness, dir: [Gi; 3]) -> Gi {
    factor_of(rough.m_min, roughness_raw(rough, sample_direction(dir)))
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

/// ★ THE COLUMN'S RELIEF — [`relief`] with the per-column ROUGHNESS FACTOR on the fine half (slice
/// 8a stage 3; `slice_8a_design.md` §2.3). The octaves before [`Roughness::first_fine`] are summed
/// at full amplitude; the octaves from it on are summed and multiplied ONCE.
///
/// **The bits, and why `mul_shr` and not `* >>`.** The fine sum is gap steps at [`NOISE_BITS`]: on
/// the home planet the fine amplitudes reach 1 532 m, which is `1 532 × 128 × 2²⁸ ≈ 5.3 × 10¹³`, and
/// times a factor at 2²⁸ it is about `1.4 × 10²²` — past 2⁶³. So the product goes through the
/// TWO-WORD [`Gi::mul_shr`], which TRUNCATES toward zero. Stated here so no reader guesses.
///
/// ★ ONE INDEX LOOP AND TWO ADDITIVE ACCUMULATORS, and the fine half is picked out by an ARITHMETIC
/// MASK the OCTAVE ITSELF carries ([`Octave::fine`]) — never by comparing the index against a split
/// point, and never by a second loop that starts where the first one stopped.
///
/// ★ **THE CARD MEASURED BOTH WRONG SHAPES** (slice 8a stage 3, `just gpu-drift` on an Apple M4 Pro
/// through Metal, the boot self-check over 6 815 744 cells):
/// * two loops sharing ONE counter — **3 364 016 differing cells**;
/// * two loops, the second starting at the branch-assigned split — **24 272 differing cells**;
/// * this shape — **ZERO**.
///
/// That is the third entry in the ledger `root.rs` and `cell.rs` began: a shader compiler spells a
/// value a loop carries or a branch assigns differently from the CPU, and only an ADDITIVE
/// accumulator with its mask in the data is safe.
#[must_use]
pub fn relief_shaped(octaves: &[Octave], dir: [Gi; 3], rough: &Roughness) -> Gi {
    let d = sample_direction(dir);
    let m = factor_of(rough.m_min, roughness_raw(rough, d));
    let count = octaves.len();
    let mut whole = Gi::ZERO;
    let mut fine = Gi::ZERO;
    let mut k = 0;
    while k < count {
        let term = octave_term(&octaves[k], d);
        whole += term;
        fine += term & octaves[k].fine;
        k += 1;
    }
    (whole - fine) + fine.mul_shr(m, NOISE_BITS)
}

/// [`relief_shaped`] over the first `count` octaves of a fixed-size table — the form a GPU shell
/// calls, because a shader holds its octaves in a local array and cannot slice it to a runtime
/// length. A `count` past the table reads the whole table.
#[must_use]
pub fn relief_of_table(
    octaves: &[Octave; OCTAVES_CAP],
    count: usize,
    dir: [Gi; 3],
    rough: &Roughness,
) -> Gi {
    let d = sample_direction(dir);
    let n = if count < OCTAVES_CAP {
        count
    } else {
        OCTAVES_CAP
    };
    let m = factor_of(rough.m_min, roughness_raw(rough, d));
    let mut whole = Gi::ZERO;
    let mut fine = Gi::ZERO;
    let mut k = 0;
    while k < n {
        let term = octave_term(&octaves[k], d);
        whole += term;
        fine += term & octaves[k].fine;
        k += 1;
    }
    (whole - fine) + fine.mul_shr(m, NOISE_BITS)
}

/// The 40-bit direction shifted to the noise's sample bits.
fn sample_direction(dir: [Gi; 3]) -> [Gi; 3] {
    [
        dir[0] >> (DIR_BITS - SAMPLE_BITS),
        dir[1] >> (DIR_BITS - SAMPLE_BITS),
        dir[2] >> (DIR_BITS - SAMPLE_BITS),
    ]
}

/// One octave's term of the sum — SMOOTH or RIDGED, chosen by an ARITHMETIC MASK and never by a
/// branch (slice 8a stage 2; `slice_8a_design.md` §2.2).
///
/// A ridged octave's value is `1 − |n|`, doubled and re-centred so the amplitude keeps its meaning:
/// `ridged = ((1 − |n|) << 1) − 1`. There is no shift down, so the transform adds NO rounding — the
/// error budget gains nothing from it — and `|n| ≤ 1` keeps `ridged` inside `[−1, 1]`, so
/// `relief_bound` holds in form.
///
/// **Why a mask and not an `if`.** `root.rs` and `cell.rs` each record a MEASURED fault where the
/// card mis-spelled a value a branch assigned inside a loop. `v = n + (kind & (ridged − n))` is one
/// `and`, one add and one subtract, the accumulator stays `h += term`, and there is no shape for a
/// shader compiler to get wrong.
///
/// **Example.** The home planet's 3.13 km octave is ridged, so the pilot walking the hillside sees a
/// SPUR run down it where the smooth table gave her a lump.
fn octave_term(o: &Octave, d: [Gi; 3]) -> Gi {
    let p = [lattice(d[0], o), lattice(d[1], o), lattice(d[2], o)];
    let n = noise3(o.seed, p);
    let mag = Gi::new(n.unsigned_abs() as i64);
    let ridged = ((NOISE_ONE - mag) << 1) - NOISE_ONE;
    let v = n + (o.kind & (ridged - n));
    (o.amplitude * v) >> AMP_BITS
}

/// ★ THE BIOME FIELD's own numbers (ruling F7: the kernel names no substance and no biome; it
/// answers a CODE the host gives meaning to). `repr(C)`, twenty words — four of its own and two
/// octaves of eight — so a card reads the row the host wrote.
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
        Octave::smooth(seed, Gi::new(fi), Gi::new(ff), Gi::new(a))
    }

    /// An octave of ONE NOISE UNIT: `(1 << AMP_BITS) · n >> AMP_BITS` is `n` exactly, so a test
    /// reads the raw noise back out of it.
    fn unit_octave(seed: u64, frequency: f64) -> Octave {
        Octave {
            amplitude: Gi::new(1 << AMP_BITS),
            ..octave(seed, frequency, 0.0)
        }
    }

    /// `round(0.06 · 2²⁸)` — the floor the generator draws (`vd_terrain::body::M_MIN`), stated here
    /// as the word so the recipe's own tests need no body.
    const M_MIN_Q: Gi = Gi::new(16_106_127);

    /// The same octave, marked FINE: the per-column factor multiplies it.
    fn fine(o: Octave) -> Octave {
        Octave {
            fine: OCTAVE_FINE,
            ..o
        }
    }

    /// A roughness row: the placeholder octave, the floor, and where the fine band begins.
    fn roughness(seed: u64, frequency: f64, m_min: Gi, first_fine: i64) -> Roughness {
        Roughness {
            octave: unit_octave(seed, frequency),
            m_min,
            first_fine: Gi::new(first_fine),
        }
    }

    /// ★ A ROUGHNESS ROW WHOSE FACTOR IS EXACTLY ONE at every column: the floor stands AT one, so
    /// `m = 1 + (1 − 1)·fade(r) = 1` whatever the field says. It is what lets a test compare the
    /// shaped sum with the raw sum word for word.
    fn flat_roughness() -> Roughness {
        roughness(29, 3.5, NOISE_ONE, 0)
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
        let flat = flat_roughness();
        assert_eq!(relief_of_table(&table, 2, dir, &flat), h);
        assert_eq!(relief_of_table(&table, 0, dir, &flat), Gi::ZERO);
        assert_eq!(relief_of_table(&table, 99, dir, &flat), relief(&table, dir));
    }

    /// ★ A RIDGED OCTAVE IS `1 − |n|`, RE-CENTRED (slice 8a stage 2; `slice_8a_design.md` §2.2).
    ///
    /// The test states the transform ITSELF, from the raw noise read back through a smooth octave of
    /// one noise unit, and compares it with what the kernel answers. It could fail three ways: a
    /// ridge that forgot to double lands at half the height; a ridge that shifted down rounds, which
    /// the error budget says it must not; and a ridge that left `[−1, 1]` would break
    /// `relief_bound`, which the ladder's band is sized on.
    ///
    /// RED before the stage: `octave_term` read the noise itself whatever the octave said, so the
    /// ridged term equalled the smooth one.
    #[test]
    fn a_ridged_octave_is_one_minus_the_magnitude_recentred() {
        // The two words are a MASK, not a flag: zero selects nothing, all ones select everything.
        assert_eq!(OCTAVE_SMOOTH, Gi::ZERO);
        assert_eq!(OCTAVE_RIDGED.raw(), -1);
        let smooth = octave(7, 16.5, 8.0);
        let ridged = Octave {
            kind: OCTAVE_RIDGED,
            ..smooth
        };
        // The RAW noise, through an octave of ONE noise unit: `(1 << AMP_BITS) · n >> AMP_BITS` is
        // `n` exactly, whatever the sign, because the low bits are zero.
        let unit = Octave::smooth(
            smooth.seed,
            smooth.frequency_int,
            smooth.frequency_frac,
            Gi::new(1 << AMP_BITS),
        );
        let bound = smooth.amplitude << (NOISE_BITS - AMP_BITS);
        let mut crests = 0;
        let mut flanks = 0;
        let mut differ = 0;
        let mut k = 1i64;
        while k < 400 {
            let dir = crate::bend::normalise([
                DIR_ONE,
                Gi::new(k) << (DIR_BITS - 11),
                Gi::new(400 - k) << (DIR_BITS - 12),
            ]);
            let n = relief(&[unit], dir);
            let mag = Gi::new(n.unsigned_abs() as i64);
            let want = (smooth.amplitude * (((NOISE_ONE - mag) << 1) - NOISE_ONE)) >> AMP_BITS;
            let got = relief(&[ridged], dir);
            let plain = relief(&[smooth], dir);
            assert_eq!(got, want, "the ridged term at k {k}");
            assert_eq!(
                plain,
                (smooth.amplitude * n) >> AMP_BITS,
                "the smooth term at k {k}"
            );
            // The transform stays inside the amplitude, which is what keeps `relief_bound` true and
            // the ladder's band big enough to hold the lifted ground.
            assert!(want <= bound, "over the amplitude at k {k}");
            assert!(want >= Gi::ZERO - bound, "under the amplitude at k {k}");
            crests += i32::from(want > Gi::ZERO);
            flanks += i32::from(want < Gi::ZERO);
            differ += i32::from(got != plain);
            k += 1;
        }
        // A crest stands above the mean and its flank falls below it: the scan meets both, so the
        // one-sided lift is a shape and not an offset.
        assert!(crests > 0, "a crest stands over the mean");
        assert!(flanks > 0, "and a flank falls under it");
        assert!(differ > 0, "the mask changes the term");
    }

    /// ★ THE SUM IS ADDITIVE WITH A RIDGED OCTAVE IN IT — the accumulator shape the card needs
    /// (`slice_8a_design.md` §2.0: a value a BRANCH assigns inside a loop read one step stale on the
    /// card, MEASURED twice). Each octave's term is a function of that octave alone, so the whole sum
    /// equals the terms taken one at a time, in any order, and the table form answers the same.
    #[test]
    fn the_sum_is_additive_with_a_ridged_octave_in_it() {
        let a = octave(1, 15.9259, 8011.2287);
        let b = Octave {
            kind: OCTAVE_RIDGED,
            ..octave(2, 31.8518, 4111.0007)
        };
        let c = Octave {
            kind: OCTAVE_RIDGED,
            ..octave(3, 63.7036, 2109.6)
        };
        let d = octave(4, 127.4072, 1082.5);
        let table = [a, b, c, d];
        let mut k = 0i64;
        while k < 40 {
            let dir = crate::bend::normalise([
                DIR_ONE,
                Gi::new(k) << (DIR_BITS - 9),
                Gi::new(k * 7) << (DIR_BITS - 11),
            ]);
            let whole = relief(&table, dir);
            let one_at_a_time =
                relief(&[a], dir) + relief(&[b], dir) + relief(&[c], dir) + relief(&[d], dir);
            assert_eq!(whole, one_at_a_time, "the sum is additive at k {k}");
            // The order of the table does not change the sum.
            assert_eq!(relief(&[d, c, b, a], dir), whole, "the order at k {k}");
            // A prefix of the table is the prefix of the sum: what a coarse rung reads.
            assert_eq!(
                relief(&table[..2], dir),
                relief(&[a], dir) + relief(&[b], dir),
                "the coarse prefix at k {k}"
            );
            k += 1;
        }
        // The table form the GPU shell calls reads the same words, ridged octaves and all.
        let dir = crate::bend::normalise([DIR_ONE, Gi::ONE << (DIR_BITS - 8), Gi::ZERO]);
        let mut full = [a; OCTAVES_CAP];
        full[1] = b;
        full[2] = c;
        full[3] = d;
        assert_eq!(
            relief_of_table(&full, 4, dir, &flat_roughness()),
            relief(&table, dir)
        );
        // And the mask is what makes the difference: the same table, every octave smooth, answers
        // something else.
        let plain = [
            a,
            Octave {
                kind: OCTAVE_SMOOTH,
                ..b
            },
            Octave {
                kind: OCTAVE_SMOOTH,
                ..c
            },
            d,
        ];
        assert_ne!(relief(&plain, dir), relief(&table, dir));
    }

    /// ★ THE ROUGHNESS FACTOR NEVER LEAVES ITS BAND, AND IT CLIMBS WITH ITS FIELD (slice 8a stage
    /// 3; `slice_8a_design.md` §1.2 and §2.3).
    ///
    /// Four statements, each of which could fail. (1) BOTH ENDS ARE REACHED EXACTLY: a field at its
    /// bottom answers the floor and a field at its top answers ONE — an `m` over one would break
    /// the ladder's band, which is sized at the ceiling, and an `m` under the floor would flatten a
    /// range into a sea bed. (2) The factor is MONOTONIC in its field: a rougher reading never
    /// gives a smoother column, which is what makes the plain and the range two places and not a
    /// speckle. (3) Over a scan of real directions the factor stays inside `[m_min, 1]`. (4) The
    /// scan MEETS both halves of the band, so the field really modulates and is not a constant.
    ///
    /// RED before the stage: neither `Roughness` nor `roughness_factor` existed.
    #[test]
    fn the_roughness_factor_never_leaves_its_band_and_climbs_with_its_field() {
        // The two ends, stated on the pure factor so the test names them and does not hunt for them.
        assert_eq!(
            factor_of(M_MIN_Q, Gi::ZERO),
            M_MIN_Q,
            "the floor is reached"
        );
        assert_eq!(factor_of(M_MIN_Q, NOISE_ONE), NOISE_ONE, "and the ceiling");
        // Monotonic, over the whole reading: 256 steps of the field, never falling, and it rises.
        let mut last = factor_of(M_MIN_Q, Gi::ZERO);
        let mut rose = 0;
        let mut k = 1i64;
        while k <= 256 {
            let r = (NOISE_ONE * Gi::new(k)) >> 8;
            let m = factor_of(M_MIN_Q, r);
            assert!(m >= last, "the factor fell at k {k}: {m:?} under {last:?}");
            rose += i32::from(m > last);
            last = m;
            k += 1;
        }
        assert!(rose > 200, "the factor climbs: {rose} of 256 steps");
        // A scan of real columns: inside the band, and both halves of it met.
        let rough = roughness(31, 4.5, M_MIN_Q, 4);
        let half = (M_MIN_Q + NOISE_ONE) >> 1;
        let (mut low, mut high) = (0, 0);
        let mut k = 1i64;
        while k < 400 {
            let dir = crate::bend::normalise([
                DIR_ONE,
                Gi::new(k) << (DIR_BITS - 11),
                Gi::new(400 - k) << (DIR_BITS - 12),
            ]);
            let m = roughness_factor(&rough, dir);
            assert!(m >= M_MIN_Q, "under the floor at k {k}: {m:?}");
            assert!(m <= NOISE_ONE, "over the ceiling at k {k}: {m:?}");
            low += i32::from(m < half);
            high += i32::from(m >= half);
            k += 1;
        }
        assert!(low > 0, "the scan meets a plain");
        assert!(high > 0, "and a range");
    }

    /// ★ THE FACTOR MULTIPLIES THE FINE HALF ONCE, AND THE SUM STAYS ADDITIVE (slice 8a stage 3;
    /// `slice_8a_design.md` §2.3).
    ///
    /// The accumulator shape the card needs: the whole sum equals the COARSE prefix taken on its own
    /// plus the FINE tail taken on its own and multiplied once. It could fail four ways — a factor
    /// applied per octave (the rounding would differ), a factor applied to the coarse half too, a
    /// split at the wrong index, and a product through a plain multiply, which wraps past 2⁶³ at the
    /// home planet's fine amplitudes.
    ///
    /// RED before the stage: `relief_shaped` did not exist and the table form took no roughness.
    #[test]
    fn the_factor_multiplies_the_fine_half_once_and_the_sum_stays_additive() {
        // The first two octaves are COARSE; the last two are FINE, and one of each is RIDGED, so
        // the two masks are proved independent of one another.
        let a = octave(1, 15.9259, 8011.2287);
        let b = Octave {
            kind: OCTAVE_RIDGED,
            ..octave(2, 31.8518, 4111.0007)
        };
        let c = fine(octave(3, 63.7036, 2109.6));
        let d = fine(Octave {
            kind: OCTAVE_RIDGED,
            ..octave(4, 127.4072, 1082.5)
        });
        let table = [a, b, c, d];
        let rough = roughness(37, 2.5, M_MIN_Q, 2);
        // The KERNEL reads the per-octave mask; the roughness row's own index is the DRAW's record
        // of the same split, and the two must agree — a table whose masks disagreed with it would
        // give a sum the host's own accounting could not reproduce.
        for (i, o) in table.iter().enumerate() {
            let want = if i < rough.first_fine.raw() as usize {
                OCTAVE_COARSE
            } else {
                OCTAVE_FINE
            };
            assert_eq!(o.fine, want, "the mask of octave {i}");
        }
        let mut differ = 0;
        let mut k = 0i64;
        while k < 40 {
            let dir = crate::bend::normalise([
                DIR_ONE,
                Gi::new(k) << (DIR_BITS - 9),
                Gi::new(k * 7) << (DIR_BITS - 11),
            ]);
            let m = roughness_factor(&rough, dir);
            let coarse_sum = relief(&table[..2], dir);
            let fine_sum = relief(&table[2..], dir);
            assert_eq!(
                relief_shaped(&table, dir, &rough),
                coarse_sum + fine_sum.mul_shr(m, NOISE_BITS),
                "the split at k {k}"
            );
            // The card's form reads the same words.
            let mut full = [a; OCTAVES_CAP];
            full[1] = b;
            full[2] = c;
            full[3] = d;
            assert_eq!(
                relief_of_table(&full, 4, dir, &rough),
                relief_shaped(&table, dir, &rough),
                "the table form at k {k}"
            );
            differ += i32::from(relief_shaped(&table, dir, &rough) != relief(&table, dir));
            k += 1;
        }
        assert!(differ > 0, "the factor changes the sum");
        // ★ AT THE FACTOR'S CEILING THE SHAPED SUM IS THE RAW SUM, WORD FOR WORD. That is the
        // property the ladder's band is sized on (`m = 1`), and it is what lets 8c swap the field.
        let dir = crate::bend::normalise([DIR_ONE, Gi::ONE << (DIR_BITS - 8), Gi::ZERO]);
        let flat = flat_roughness();
        assert_eq!(roughness_factor(&flat, dir), NOISE_ONE);
        assert_eq!(relief_shaped(&table, dir, &flat), relief(&table, dir));
        // A table of COARSE octaves only is the raw sum again — what a coarse rung reads once its
        // fine octaves are dropped.
        let coarse_table = [
            a,
            b,
            Octave {
                fine: OCTAVE_COARSE,
                ..c
            },
            Octave {
                fine: OCTAVE_COARSE,
                ..d
            },
        ];
        let all_coarse = roughness(37, 2.5, M_MIN_Q, 9);
        assert_eq!(
            relief_shaped(&coarse_table, dir, &all_coarse),
            relief(&coarse_table, dir)
        );
        let mut full = [a; OCTAVES_CAP];
        full[1] = b;
        full[2] = coarse_table[2];
        full[3] = coarse_table[3];
        assert_eq!(
            relief_of_table(&full, 4, dir, &all_coarse),
            relief(&coarse_table, dir)
        );
        // And a table of FINE octaves only multiplies the WHOLE sum, the other end of the rule.
        let fine_table = [fine(a), fine(b), c, d];
        let all_fine = roughness(37, 2.5, M_MIN_Q, 0);
        let m = roughness_factor(&all_fine, dir);
        assert_eq!(
            relief_shaped(&fine_table, dir, &all_fine),
            relief(&fine_table, dir).mul_shr(m, NOISE_BITS)
        );
    }

    /// A biome charter with no climate noise at all, so a test names a column's temperature by its
    /// latitude and its height alone. The sea stands at a thousand gap steps.
    fn biome_charter() -> BiomeCharter {
        let quiet = Octave::smooth(1, Gi::ONE, Gi::ZERO, Gi::ZERO);
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
