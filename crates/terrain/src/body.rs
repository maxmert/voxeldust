//! ★ THE BODY DEFINITION — everything the recipe needs about one round body, drawn ONCE from the
//! body's seed and its radius: the ladder, the sea level, the relief, the octave table, the strata,
//! the biome and the cave parameters.
//!
//! ★ THE CHARTER IS INTEGERS (ruling F7, 2026-09-12). Every number a kernel reads is a word in the
//! recipe's own fixed-point formats — the octaves as [`vd_recipe::height::Octave`], the radius and
//! the sea in gap steps, the cave band in gap steps, the cavern wavelength's RECIPROCAL so no kernel
//! divides, the cell-count reciprocal of every rung so no kernel divides there either. The DRAW is
//! the one float left in the generator: the seed's shares, caps and weights are read as fenced
//! floats ([`crate::gf::Gf`], IEEE-exact on every target) and ROUNDED ONCE into the charter, here,
//! on the CPU, once per body. A kernel never sees a float. (Ruling V13 L12 owes the next step: the
//! realm STORES the charter and STATES it in its surface statement, so the draw itself runs once in
//! the world's life rather than once per process.)
//!
//! **Example.** The home planet's seed draws its relief, its sea level, its octave table from a long
//! wave down to a 30 m ripple, its topsoil over subsoil over sediment over bedrock, and its cave band.
//! Every host that holds the seed holds this table as the same words (`crate::home` states the planet;
//! the bench `terrain_cost` prints its numbers).

use crate::gf::Gf;
use crate::strata::{Bedrock, StrataTable, Stratum};
use crate::units::{LENGTH_BITS, STEPS_PER_M};
use vd_recipe::Gi;
use vd_recipe::bend::inv_n_of;
use vd_recipe::height::{AMP_BITS, Octave};
use vd_recipe::noise::NOISE_BITS;
use vd_recipe::root::recip_pow2;
use vd_seed::ladder::Ladder;
use vd_seed::rng::{SplitMix64, child_seed};

/// The most octaves a body can have; the table is sized for it. The recipe holds the same cap: a
/// GPU shell keeps the octaves in a fixed-size table of it (`vd_recipe::height::relief_of_table`).
pub const OCTAVES: usize = 16;
const _: () = assert!(OCTAVES == vd_recipe::height::OCTAVES_CAP);
/// The most rungs a body can have: the address's four bits and one (`vd_seed::ladder::RUNG_MAX`).
pub const RUNGS: usize = 16;
/// The coarsest wavelength any body draws, in metres, and the wavelength below which no octave is
/// added. The table can never overflow: the cap halves to under the floor within `OCTAVES` steps.
pub const LONG_WAVE_CAP_M: u64 = 400_000;
pub const SHORT_WAVE_M: u64 = 30;
const _: () = assert!((LONG_WAVE_CAP_M >> OCTAVES) < SHORT_WAVE_M);
const _: () = assert!(RUNGS == vd_seed::ladder::RUNG_MAX as usize + 1);

/// The fraction bits of the RADIUS's reciprocal ([`BodyDefinition::radius_recip`]): a whole word less
/// the headroom a two-word product needs, so the column bound's divide is one multiply.
pub const RADIUS_RECIP_BITS: u32 = 62;
/// The fraction bits of the CAVERN WAVELENGTH's reciprocal: the lattice point it makes is under 2⁵⁰
/// at the largest legal body, and the wavelength's own error is then under one part in 2⁴³.
pub const CAVERN_RECIP_BITS: u32 = 48;
/// The tube carvers' region edge, in metres. A POWER OF TWO, so a point's region index is a shift and
/// not a division (the charter holds the shift).
pub const TUBE_REGION_M: u32 = 512;
const _: () = assert!(TUBE_REGION_M.is_power_of_two());

/// The salts of the seed tree: each part of the body draws from its own stream, so adding a draw to
/// one part never moves another.
mod salt {
    pub const OCTAVES: u64 = 0x5e_ed_01;
    pub const SEA: u64 = 0x5e_ed_02;
    pub const STRATA: u64 = 0x5e_ed_03;
    pub const BIOME: u64 = 0x5e_ed_04;
    pub const CAVERN: u64 = 0x5e_ed_05;
    pub const TUBES: u64 = 0x5e_ed_06;
    pub const NOISE: u64 = 0x5e_ed_07;
}

/// The cave parameters, as the kernels read them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caves {
    /// The cavern field's lattice wavelength in whole metres (every fourth cell at rung 0 is 4 m; the
    /// field is smooth across many cells). The rung test reads it as metres.
    pub(crate) cavern_wavelength_m: u32,
    /// `floor(2^CAVERN_RECIP_BITS / wavelength_m)`: the lattice point is one two-word product.
    pub(crate) cavern_recip: Gi,
    /// The value above which a cell is hollow, at the noise's fraction bits.
    pub(crate) cavern_threshold: Gi,
    /// How many GAP STEPS of hollow one unit of the field above the threshold opens.
    pub(crate) cavern_scale_steps: Gi,
    /// The depth band caves live in, in metres under the surface.
    pub(crate) min_depth_m: u32,
    pub(crate) max_depth_m: u32,
    /// The tube carvers' region edge in metres, the same edge in gap steps, the shift from gap steps
    /// to a region index, and their seed.
    pub(crate) tube_region_m: u32,
    pub(crate) tube_region_steps: Gi,
    pub(crate) tube_region_shift: u32,
    pub(crate) tube_seed: u64,
    /// A tube's radius in gap steps.
    pub(crate) tube_radius_steps: Gi,
}

/// The biome field's parameters. The two slow noises are stated as OCTAVES of the one octave sum, with
/// the amplitude of one noise unit, so the biome reads the field through the same kernel the hills do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiomeField {
    pub(crate) temperature: Octave,
    pub(crate) humidity: Octave,
    /// Above this height over the sea, in gap steps at [`LENGTH_BITS`], a column is highland whatever
    /// its climate.
    pub(crate) highland_above: Gi,
    /// `floor(2^RADIUS_RECIP_BITS / (highland_above_steps + one metre))`: the height share the
    /// temperature reads is one multiply, never a divide.
    pub(crate) highland_recip: Gi,
}

/// One round body, fully described. Every field is read inside the crate only (the refuter's
/// finding 18): a body is DRAWN from a seed by [`BodyDefinition::from_seed`] and never assembled from
/// numbers computed elsewhere, so no number from an unfenced crate can enter the recipe as a body.
/// What a host outside the crate may read is behind the accessors below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyDefinition {
    /// The body's own seed (the realm's).
    pub(crate) seed: u64,
    /// The grid on the ladder.
    pub(crate) ladder: Ladder,
    /// The radius on the ladder, in gap steps at [`LENGTH_BITS`].
    pub(crate) radius: Gi,
    /// The same radius in WHOLE gap steps, which the reciprocal below is the reciprocal of.
    pub(crate) radius_steps: Gi,
    /// `floor(2^RADIUS_RECIP_BITS / radius_steps)`: the column bound's one divide, done once.
    pub(crate) radius_recip: Gi,
    /// The sea's radius: the ladder radius plus the seed's sea offset, in gap steps at
    /// [`LENGTH_BITS`].
    pub(crate) sea_radius: Gi,
    /// The cell-count reciprocal of every rung (`vd_recipe::bend::inv_n_of`), so a direction costs no
    /// divide. Entries past the body's own rungs are zero and never read.
    pub(crate) inv_n: [Gi; RUNGS],
    /// The octaves, coarsest first; only the first `octave_count` are live.
    pub(crate) octaves: [Octave; OCTAVES],
    pub(crate) octave_count: u8,
    /// The strata and the caves and the biomes.
    pub(crate) strata: StrataTable,
    pub(crate) caves: Caves,
    pub(crate) biome: BiomeField,
}

/// A draw in `[lo, hi)` metres as a whole number.
fn draw_m(rng: &mut SplitMix64, lo: u64, hi: u64) -> u32 {
    rng.range_u64(lo, hi) as u32
}

/// A draw in `[0, 1)` as a fenced float: the top 53 bits over 2^53, exact.
fn draw_unit(rng: &mut SplitMix64) -> Gf {
    Gf::from_i64((rng.next_u64() >> 11) as i64) / Gf::from_i64(1 << 53)
}

/// A fenced float ROUNDED to a whole number, half upward: the one rounding between the draw and the
/// charter. The draw is never negative where this is used.
fn rounded(v: Gf) -> i64 {
    (v + Gf::HALF).floor().to_i64_floor()
}

/// A length the draw states in metres, as a word in gap steps at [`LENGTH_BITS`].
fn length_of(metres: Gf) -> Gi {
    Gi::new(rounded(metres * Gf::from_i64(STEPS_PER_M << LENGTH_BITS)))
}

/// A share the draw states in `[0, 1)`, as a word at the noise's fraction bits.
fn share_of(unit: Gf) -> Gi {
    Gi::new(rounded(unit * Gf::from_i64(1 << NOISE_BITS)))
}

/// One noise of the biome field, as an octave of unit amplitude: the wavelength in metres turns into
/// the frequency the one octave sum reads (cells per unit direction = the radius over the wavelength).
fn biome_octave(seed: u64, radius_m: Gf, wavelength_m: u32) -> Octave {
    let (frequency_int, frequency_frac) =
        frequency_of(radius_m / Gf::from_i64(i64::from(wavelength_m)));
    Octave {
        seed,
        frequency_int,
        frequency_frac,
        // One noise unit: the octave sum then reads back the noise itself.
        amplitude: Gi::new(1 << AMP_BITS),
    }
}

/// ★ AN OCTAVE'S FREQUENCY as a real number — cells per unit direction, for a host outside the recipe
/// (a caption, the skyline march's spectrum table). The recipe itself reads the pair.
#[must_use]
pub fn octave_frequency(o: &Octave) -> f64 {
    (Gf::from_i64(o.frequency_int.raw())
        + Gf::from_i64(o.frequency_frac.raw()) / Gf::from_i64(1 << NOISE_BITS))
    .to_f64()
}

/// ★ AN OCTAVE'S AMPLITUDE in metres, for a host outside the recipe. The recipe reads gap steps at
/// [`AMP_BITS`] below the step.
#[must_use]
pub fn octave_amplitude_m(o: &Octave) -> f64 {
    crate::units::metres_of_fixed(o.amplitude, AMP_BITS)
}

/// A frequency the draw states as a fenced float, as the recipe's pair: the integer part and the
/// fraction at the noise's fraction bits. MEASURED (the integer bench, part 1): a frequency rounded
/// to 2⁻⁸ moved the coarsest lattice point by a ten-thousandth of a cell, which eight kilometres of
/// amplitude turned into metres — so the fraction carries the noise's own 28 bits.
fn frequency_of(frequency: Gf) -> (Gi, Gi) {
    let whole = frequency.floor();
    (
        Gi::new(whole.to_i64_floor()),
        Gi::new(rounded((frequency - whole) * Gf::from_i64(1 << NOISE_BITS))),
    )
}

impl BodyDefinition {
    /// The body's own seed (the realm's).
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The grid on the ladder.
    #[must_use]
    pub const fn ladder(&self) -> &Ladder {
        &self.ladder
    }

    /// How many octaves the seed drew (the live prefix of the table).
    #[must_use]
    pub const fn octave_count(&self) -> u8 {
        self.octave_count
    }

    /// The body's radius on the ladder, in metres — for a host outside the recipe (a caption, a
    /// census). The recipe itself reads the word.
    #[must_use]
    pub fn radius_m(&self) -> f64 {
        crate::units::metres_of_q28(self.radius)
    }

    /// The sea's radius in metres, for a host outside the recipe.
    #[must_use]
    pub fn sea_radius_m(&self) -> f64 {
        crate::units::metres_of_q28(self.sea_radius)
    }

    /// The cell-count reciprocal of a rung: what a direction needs instead of a divide. A rung past
    /// the body's own reads the top rung's.
    #[must_use]
    pub(crate) fn inv_n(&self, rung: u8) -> Gi {
        self.inv_n[usize::from(rung.min(self.ladder.rungs - 1))]
    }

    /// The body for `seed` at `look_radius_m`; `None` where the ladder refuses the radius. The band
    /// — the crust under the surface and the room over it — is DERIVED from the relief the seed draws,
    /// the strata and the caves, so the surface, every stratum and every cave fit inside the grid.
    #[must_use]
    pub fn from_seed(seed: u64, look_radius_m: f64) -> Option<BodyDefinition> {
        // The ladder's edge count depends on the radius alone; the band is fixed once the relief is
        // known, below, and the ladder is taken again with it (the same `n` both times).
        let radius_m = Gf::from_f64(Ladder::for_radius(look_radius_m, 1, 1)?.radius_m());

        // The relief: a share of the radius, capped, times a seed factor in [0.5, 1.5).
        let mut octave_rng = SplitMix64::new(child_seed(seed, salt::OCTAVES, 0));
        let relief_share = radius_m * Gf::from_f64(0.004);
        let relief_cap = relief_share.clamp(Gf::from_i64(200), Gf::from_i64(12_000));
        let relief_m = relief_cap * (Gf::HALF + draw_unit(&mut octave_rng));
        // The coarsest wavelength: a quarter to a half of the radius, at least 20 km, at most 400 km.
        let long_share = Gf::from_f64(0.25) + draw_unit(&mut octave_rng) * Gf::from_f64(0.25);
        let long_wave_m = (radius_m * long_share)
            .clamp(Gf::from_i64(20_000), Gf::from_i64(LONG_WAVE_CAP_M as i64));
        // The roughness: how much each finer octave keeps of the one before, in [0.45, 0.55).
        let k_rough = Gf::from_f64(0.45) + draw_unit(&mut octave_rng) * Gf::from_f64(0.10);
        // Halve the wavelength until it is under 30 m or the table is full. The amplitudes are scaled
        // so their SUM is the relief, so the band bound is exact.
        let mut waves = [Gf::ZERO; OCTAVES];
        let mut weights = [Gf::ZERO; OCTAVES];
        let mut count = 0usize;
        let mut wave_m = long_wave_m;
        let mut weight = Gf::ONE;
        let mut weight_sum = Gf::ZERO;
        // Never past the table: the compile-time assertion above proves the cap halves to under the
        // floor within `OCTAVES` steps.
        while wave_m > Gf::from_i64(SHORT_WAVE_M as i64) {
            waves[count] = wave_m;
            weights[count] = weight;
            weight_sum += weight;
            weight *= k_rough;
            wave_m *= Gf::HALF;
            count += 1;
        }
        // THE ONE ROUNDING into the charter: the frequency as an integer part and a 28-bit fraction,
        // the amplitude in gap steps at `AMP_BITS` below the step (1/32 768 m at the metre rung).
        let mut octaves = [Octave {
            seed: 0,
            frequency_int: Gi::ZERO,
            frequency_frac: Gi::ZERO,
            amplitude: Gi::ZERO,
        }; OCTAVES];
        let mut o = 0;
        while o < count {
            let (frequency_int, frequency_frac) = frequency_of(radius_m / waves[o]);
            let amplitude_m = weights[o] * relief_m / weight_sum;
            octaves[o] = Octave {
                seed: child_seed(seed, salt::NOISE, o as u64),
                frequency_int,
                frequency_frac,
                amplitude: Gi::new(rounded(amplitude_m * Gf::from_i64(STEPS_PER_M << AMP_BITS))),
            };
            o += 1;
        }

        // The sea: between 40 % of the relief below the ladder radius and 30 % above it.
        let mut sea_rng = SplitMix64::new(child_seed(seed, salt::SEA, 0));
        let sea_offset =
            (draw_unit(&mut sea_rng) * Gf::from_f64(0.7) - Gf::from_f64(0.4)) * relief_m;
        let sea_radius = length_of(radius_m + sea_offset.floor());

        // The strata.
        let mut strata_rng = SplitMix64::new(child_seed(seed, salt::STRATA, 0));
        let sediments = [Stratum::Sandstone, Stratum::Limestone, Stratum::Shale];
        let strata = StrataTable {
            topsoil_m: draw_m(&mut strata_rng, 1, 4),
            subsoil_m: draw_m(&mut strata_rng, 2, 8),
            sediment_m: draw_m(&mut strata_rng, 20, 80),
            sediment: sediments[strata_rng.range_u64(0, 3) as usize],
            bedrock: Bedrock::ALL[strata_rng.range_u64(0, 5) as usize],
        };

        // The biomes.
        let mut biome_rng = SplitMix64::new(child_seed(seed, salt::BIOME, 0));
        let temperature_seed = biome_rng.next_u64();
        let temperature_wavelength_m = draw_m(&mut biome_rng, 30_000, 120_000);
        let humidity_seed = biome_rng.next_u64();
        let humidity_wavelength_m = draw_m(&mut biome_rng, 20_000, 80_000);
        let highland_above = length_of(relief_m * Gf::from_f64(0.55));
        let biome = BiomeField {
            temperature: biome_octave(temperature_seed, radius_m, temperature_wavelength_m),
            humidity: biome_octave(humidity_seed, radius_m, humidity_wavelength_m),
            highland_above,
            // One metre of floor under the share's divisor, as the float recipe had it, so a body of
            // no relief still reads a share.
            highland_recip: Gi::new(recip_pow2(
                ((highland_above >> LENGTH_BITS) + Gi::new(STEPS_PER_M)).raw() as u64,
                RADIUS_RECIP_BITS,
            ) as i64),
        };

        // The caves: a cavern field in a depth band, and tube carvers in regions.
        let mut cave_rng = SplitMix64::new(child_seed(seed, salt::CAVERN, 0));
        let cave_min_depth_m = draw_m(&mut cave_rng, 8, 30);
        let cavern_wavelength_m = draw_m(&mut cave_rng, 24, 48);
        // ONE rounding: the draw's whole expression, then the charter's word.
        let cavern_threshold =
            share_of(Gf::from_f64(0.62) + draw_unit(&mut cave_rng) * Gf::from_f64(0.08));
        let cavern_scale_steps = Gi::new(20 * STEPS_PER_M);
        let max_depth_m = cave_min_depth_m + draw_m(&mut cave_rng, 200, 400);
        let tube_radius_m = draw_m(&mut cave_rng, 2, 5);
        let tube_region_steps = i64::from(TUBE_REGION_M) * STEPS_PER_M;
        let caves = Caves {
            cavern_wavelength_m,
            cavern_recip: Gi::new(
                recip_pow2(u64::from(cavern_wavelength_m), CAVERN_RECIP_BITS) as i64,
            ),
            cavern_threshold,
            cavern_scale_steps,
            min_depth_m: cave_min_depth_m,
            max_depth_m,
            tube_region_m: TUBE_REGION_M,
            tube_region_steps: Gi::new(tube_region_steps),
            tube_region_shift: tube_region_steps.trailing_zeros(),
            tube_seed: child_seed(seed, salt::TUBES, 0),
            tube_radius_steps: Gi::new(i64::from(tube_radius_m) * STEPS_PER_M),
        };

        // The band: the relief on both sides, the strata and the caves below, and room to stand on the
        // highest peak above. Whole metres, so the ladder floors them to whole cells at every rung.
        let relief_whole = relief_m.floor().to_i64_floor() as u32 + 1;
        let crust_m = relief_whole + strata.max_depth_m() + caves.max_depth_m + 64;
        let above_m = relief_whole + 64;
        let ladder = Ladder::for_radius(look_radius_m, crust_m, above_m)?;

        let mut inv_n = [Gi::ZERO; RUNGS];
        let mut rung = 0usize;
        while rung < usize::from(ladder.rungs) {
            inv_n[rung] = inv_n_of(ladder.cells_per_edge(rung as u8));
            rung += 1;
        }
        let radius = length_of(radius_m);
        let radius_steps = radius >> LENGTH_BITS;

        Some(BodyDefinition {
            seed,
            ladder,
            radius,
            radius_steps,
            radius_recip: Gi::new(recip_pow2(radius_steps.raw() as u64, RADIUS_RECIP_BITS) as i64),
            sea_radius,
            inv_n,
            octaves,
            octave_count: count as u8,
            strata,
            caves,
            biome,
        })
    }

    /// The live octaves at rung `rung`: the coarsest `octave_count − rung`, never fewer than one.
    #[must_use]
    pub fn octaves_at(&self, rung: u8) -> &[Octave] {
        let count = usize::from(self.octave_count);
        let keep = count.saturating_sub(usize::from(rung));
        let keep = if keep == 0 { 1 } else { keep };
        &self.octaves[..keep]
    }

    /// The most the surface can rise or fall from the ladder radius at rung `rung`, in gap steps at
    /// [`LENGTH_BITS`]: the sum of the live amplitudes. An exact bound, because the noise is in
    /// `[−1, 1]`.
    #[must_use]
    pub fn relief_bound(&self, rung: u8) -> Gi {
        let mut sum = Gi::ZERO;
        for o in self.octaves_at(rung) {
            sum += o.amplitude;
        }
        sum << (LENGTH_BITS - AMP_BITS)
    }

    /// The most the surface at rung `rung` can differ from the surface at rung 0, in gap steps at
    /// [`LENGTH_BITS`]: the sum of the dropped amplitudes (the tier-agreement bound).
    #[must_use]
    pub fn dropped_bound(&self, rung: u8) -> Gi {
        self.relief_bound(0) - self.relief_bound(rung)
    }

    /// The relief bound in metres, for a host outside the recipe.
    #[must_use]
    pub fn relief_bound_m(&self, rung: u8) -> f64 {
        crate::units::metres_of_q28(self.relief_bound(rung))
    }

    /// The dropped-octave bound in metres, for a host outside the recipe (the client's crossfade sink
    /// reads it).
    #[must_use]
    pub fn dropped_bound_m(&self, rung: u8) -> f64 {
        crate::units::metres_of_q28(self.dropped_bound(rung))
    }
}

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
    use crate::units::metres_of_q28;
    use vd_recipe::noise::NOISE_ONE;

    fn home() -> BodyDefinition {
        crate::home::home_planet()
    }

    /// The frequency of one octave, reassembled from the charter's pair, at the noise's bits.
    fn frequency_q(o: &Octave) -> Gi {
        (o.frequency_int << NOISE_BITS) + o.frequency_frac
    }

    #[test]
    fn a_body_is_drawn_once_from_its_seed_and_its_octaves_sum_to_its_relief() {
        let m = home();
        assert_eq!(m, home(), "the same seed, the same body");
        assert_ne!(
            m.octaves[0].seed,
            BodyDefinition::from_seed(
                m.seed + 1,
                f64::from_bits(crate::home::HOME_PLANET_RADIUS_BITS)
            )
            .expect("on the ladder")
            .octaves[0]
                .seed
        );
        assert!(
            m.octave_count >= 8,
            "a 20 km wave halves to 30 m in at least eight steps"
        );
        assert!(usize::from(m.octave_count) <= OCTAVES);
        let relief = m.relief_bound_m(0);
        assert!(relief >= 100.0, "{relief}");
        assert!(relief <= 18_000.0, "{relief}");
        // Amplitudes fall and frequencies double, coarsest first.
        let live = m.octaves_at(0);
        let mut o = 1;
        while o < live.len() {
            assert!(live[o].amplitude < live[o - 1].amplitude);
            assert!(frequency_q(&live[o]) > frequency_q(&live[o - 1]));
            o += 1;
        }
        assert!((m.sea_radius_m() - m.radius_m()).abs() <= relief);
        assert!(m.strata.max_depth_m() < 100);
        assert!(m.caves.min_depth_m < m.caves.max_depth_m);
        assert_eq!(m.ladder.rungs, 13, "the home planet's ladder");
        // The band holds the whole relief, the strata and the caves.
        let relief_whole = relief as u32 + 1;
        assert!(m.ladder.band_m >= 2 * relief_whole + m.strata.max_depth_m() + m.caves.max_depth_m);
        assert!(u64::from(m.ladder.floor_m) + u64::from(relief_whole) < m.radius_m() as u64);
        assert_eq!(
            BodyDefinition::from_seed(1, f64::NAN),
            None,
            "the ladder refuses"
        );
        assert_eq!(
            BodyDefinition::from_seed(1, 100.0),
            None,
            "a body smaller than its own crust"
        );
    }

    /// ★ THE CHARTER'S OWN GATE (ruling F7): every word the kernels read is the ROUNDED draw, and the
    /// two reciprocals are the exact floors of the divisions they stand for. MEASURED, not argued.
    #[test]
    fn the_integer_charter_is_the_rounded_draw_and_its_reciprocals_are_exact() {
        let m = home();
        // The radius: the word is the metres to within half a step's 2⁻²⁸, and the whole-step twin is
        // its floor.
        let radius_m = m.radius_m();
        assert!((radius_m - 6_370_353.6).abs() < 1.0, "{radius_m}");
        assert_eq!(m.radius_steps, m.radius >> LENGTH_BITS);
        // The two reciprocals, against the divisions they replace.
        let radius_steps = m.radius_steps.raw();
        assert_eq!(
            i128::from(m.radius_recip.raw()),
            (1i128 << RADIUS_RECIP_BITS) / i128::from(radius_steps)
        );
        assert_eq!(
            i128::from(m.caves.cavern_recip.raw()),
            (1i128 << CAVERN_RECIP_BITS) / i128::from(m.caves.cavern_wavelength_m)
        );
        let d = (m.biome.highland_above >> LENGTH_BITS).raw() + STEPS_PER_M;
        assert_eq!(
            i128::from(m.biome.highland_recip.raw()),
            (1i128 << RADIUS_RECIP_BITS) / i128::from(d)
        );
        // The tube region's edge is a power of two, so the region index is a shift.
        assert_eq!(m.caves.tube_region_m, TUBE_REGION_M);
        assert_eq!(
            m.caves.tube_region_steps,
            Gi::new(i64::from(TUBE_REGION_M) * STEPS_PER_M)
        );
        assert_eq!(
            Gi::ONE << m.caves.tube_region_shift,
            m.caves.tube_region_steps
        );
        // The threshold sits inside the noise's unit interval, and the scale is twenty metres.
        assert!(m.caves.cavern_threshold > Gi::ZERO);
        assert!(m.caves.cavern_threshold < NOISE_ONE);
        assert_eq!(m.caves.cavern_scale_steps, Gi::new(20 * STEPS_PER_M));
        // The two exits an investigation host reads: the frequency and the amplitude as real numbers.
        let coarse = &m.octaves_at(0)[0];
        let f = octave_frequency(coarse);
        assert!((f - (coarse.frequency_int.raw() as f64)).abs() < 1.0, "{f}");
        assert!(f > 1.0, "{f}");
        assert_eq!(
            octave_amplitude_m(coarse),
            metres_of_q28(Gi::new(coarse.amplitude.raw()) << (LENGTH_BITS - AMP_BITS))
        );
        // The biome's noises carry one noise unit of amplitude, so the octave sum reads the noise.
        assert_eq!(m.biome.temperature.amplitude, Gi::new(1 << AMP_BITS));
        assert_eq!(m.biome.humidity.amplitude, Gi::new(1 << AMP_BITS));
        assert_ne!(m.biome.temperature.seed, m.biome.humidity.seed);
        // Every rung of the body has its cell-count reciprocal; none past it does.
        let mut rung = 0u8;
        while rung < m.ladder.rungs {
            assert_eq!(
                m.inv_n(rung),
                inv_n_of(m.ladder.cells_per_edge(rung)),
                "rung {rung}"
            );
            rung += 1;
        }
        assert_eq!(m.inv_n(m.ladder.rungs), m.inv_n(m.ladder.rungs - 1));
        assert_eq!(m.inv_n[usize::from(m.ladder.rungs)], Gi::ZERO);
    }

    /// M-16 restated, part 1, as an EXACT property (the refuter's finding 22): the coarse answer at
    /// every rung sums strictly fewer octaves than the rung below it, down to one. The timing in the
    /// bench is a measurement; this is the gate.
    /// What a host outside the crate may read of a body: the seed, the ladder and the octave count,
    /// and nothing it could write (the refuter's finding 18).
    #[test]
    fn a_body_shows_its_seed_its_ladder_and_its_octave_count_and_nothing_writable() {
        let m = home();
        assert_eq!(m.seed(), crate::home::HOME_PLANET_SEED);
        assert_eq!(m.ladder(), &m.ladder);
        assert_eq!(m.octave_count(), m.octave_count);
        assert_eq!(usize::from(m.octave_count()), m.octaves_at(0).len());
        assert_eq!(m.radius_m(), metres_of_q28(m.radius));
        assert_eq!(m.sea_radius_m(), metres_of_q28(m.sea_radius));
    }

    #[test]
    fn every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it() {
        let m = home();
        for o in m.octaves_at(0) {
            assert!(frequency_q(o) > Gi::ZERO);
            assert!(o.amplitude > Gi::ZERO);
        }
        let mut rung = 1u8;
        while rung < m.ladder.rungs {
            let below = m.octaves_at(rung - 1).len();
            let here = m.octaves_at(rung).len();
            assert!(here < below, "rung {rung}: {here} vs {below}");
            rung += 1;
        }
        assert!(m.octaves_at(m.ladder.rungs - 1).len() < m.octaves_at(0).len());
    }

    #[test]
    fn dropping_octaves_keeps_at_least_one_and_the_bounds_nest() {
        let m = home();
        let count = usize::from(m.octave_count);
        assert_eq!(m.octaves_at(0).len(), count);
        assert_eq!(m.octaves_at(1).len(), count - 1);
        assert_eq!(m.octaves_at(15).len(), 1, "never fewer than one octave");
        assert_eq!(m.octaves_at(200).len(), 1);
        let mut rung = 0u8;
        while rung < 15 {
            assert!(
                m.relief_bound(rung + 1) <= m.relief_bound(rung),
                "the band nests"
            );
            assert!(m.dropped_bound(rung + 1) >= m.dropped_bound(rung));
            rung += 1;
        }
        assert_eq!(m.dropped_bound(0), Gi::ZERO);
        assert_eq!(m.dropped_bound_m(0), 0.0);
        assert!(m.dropped_bound_m(9) > 0.0);
        // A tiny body: the relief cap's low end and a short octave table.
        let rock = BodyDefinition::from_seed(5, 3_000.0).expect("a rock");
        assert!(rock.octave_count >= 1);
        assert!(rock.relief_bound_m(0) >= 100.0);
        // A giant: the cap's high end and a long table.
        let giant = BodyDefinition::from_seed(9, 40_000_000.0).expect("a giant");
        assert!(giant.relief_bound_m(0) <= 18_000.0);
        assert!(
            giant.octave_count >= 12,
            "a 400 km wave halves to 30 m in fourteen steps"
        );
        // The giant's radius is the widest the address can name: its words still fit.
        assert!(giant.radius_steps.raw() < 1 << 33);
        assert_eq!(draw_m(&mut SplitMix64::new(1), 5, 6), 5);
    }
}
