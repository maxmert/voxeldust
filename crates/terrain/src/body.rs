//! ★ THE BODY DEFINITION — everything the recipe needs about one round body, drawn ONCE from the
//! body's seed and its radius: the ladder, the sea level, the relief, the octave table, the strata,
//! the biome and the cave parameters.
//!
//! Every number here is drawn with the integer hash and the fenced arithmetic, so the same seed gives
//! the same body on every host. The radius is an INPUT today: the forest draws it (with its own law)
//! and the ladder snaps it; the generator owning the radius law outright is owed to the collider slice,
//! because it changes every body in THE world and needs its own measurement.
//!
//! **Example.** The home planet's seed draws its relief, its sea level, its octave table from a long
//! wave down to a 30 m ripple, its topsoil over subsoil over sediment over bedrock, and its cave band.
//! Every host that holds the seed holds this table (`crate::home` states the planet; the bench
//! `terrain_cost` prints its numbers).

use crate::gf::Gf;
use crate::strata::{Bedrock, StrataTable, Stratum};
use vd_seed::ladder::Ladder;
use vd_seed::rng::{SplitMix64, child_seed};

/// The most octaves a body can have; the table is sized for it.
pub const OCTAVES: usize = 16;
/// The coarsest wavelength any body draws, in metres, and the wavelength below which no octave is
/// added. The table can never overflow: the cap halves to under the floor within `OCTAVES` steps.
pub const LONG_WAVE_CAP_M: u64 = 400_000;
pub const SHORT_WAVE_M: u64 = 30;
const _: () = assert!(LONG_WAVE_CAP_M / (1 << OCTAVES) < SHORT_WAVE_M);

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

/// One octave of the height field: a wavelength on the surface, an amplitude, and its own noise seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Octave {
    /// Lattice cells per unit of direction: the body's radius divided by this octave's wavelength.
    pub(crate) frequency: Gf,
    /// Metres of relief this octave contributes at most (the noise is in `[−1, 1]`).
    pub(crate) amplitude_m: Gf,
    /// This octave's noise seed.
    pub(crate) seed: u64,
}

impl Octave {
    /// The octave's frequency: the body's radius divided by its wavelength.
    #[must_use]
    pub const fn frequency(&self) -> Gf {
        self.frequency
    }

    /// The octave's amplitude in metres.
    #[must_use]
    pub const fn amplitude_m(&self) -> Gf {
        self.amplitude_m
    }

    /// The octave's noise seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }
}

/// The cave parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Caves {
    /// The cavern field's lattice wavelength in metres (every fourth cell at rung 0 is 4 m; the field
    /// is smooth across many cells).
    pub(crate) cavern_wavelength_m: Gf,
    /// The value above which a cell is hollow, in `(0, 1)`.
    pub(crate) cavern_threshold: Gf,
    /// How many metres of hollow one unit of the field above the threshold opens.
    pub(crate) cavern_scale_m: Gf,
    /// The depth band caves live in, in metres under the surface.
    pub(crate) min_depth_m: u32,
    pub(crate) max_depth_m: u32,
    /// The tube carvers' region edge in metres and their seed.
    pub(crate) tube_region_m: u32,
    pub(crate) tube_seed: u64,
    /// A tube's radius in metres, and its length in regions.
    pub(crate) tube_radius_m: Gf,
}

/// The biome field's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiomeField {
    /// The temperature noise's seed and wavelength in metres.
    pub(crate) temperature_seed: u64,
    pub(crate) temperature_wavelength_m: Gf,
    /// The humidity noise's seed and wavelength in metres.
    pub(crate) humidity_seed: u64,
    pub(crate) humidity_wavelength_m: Gf,
    /// Above this height over the sea, in metres, a column is highland whatever its climate.
    pub(crate) highland_above_m: Gf,
}

/// One round body, fully described. Every field is read inside the crate only (the refuter's
/// finding 18): a body is DRAWN from a seed by [`BodyDefinition::from_seed`] and never assembled from
/// numbers computed elsewhere, so no float from an unfenced crate can enter the recipe as a body.
/// What a host outside the crate may read is behind the accessors below.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyDefinition {
    /// The body's own seed (the realm's).
    pub(crate) seed: u64,
    /// The grid on the ladder.
    pub(crate) ladder: Ladder,
    /// The radius on the ladder, as a fenced float.
    pub(crate) radius_m: Gf,
    /// The sea's radius: the ladder radius plus the seed's sea offset.
    pub(crate) sea_radius_m: Gf,
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
        let mut octaves = [Octave {
            frequency: Gf::ZERO,
            amplitude_m: Gf::ZERO,
            seed: 0,
        }; OCTAVES];
        let mut count = 0usize;
        let mut wave_m = long_wave_m;
        let mut weight = Gf::ONE;
        let mut weight_sum = Gf::ZERO;
        // Never past the table: the compile-time assertion above proves the cap halves to under the
        // floor within `OCTAVES` steps.
        while wave_m > Gf::from_i64(SHORT_WAVE_M as i64) {
            octaves[count] = Octave {
                frequency: radius_m / wave_m,
                amplitude_m: weight,
                seed: child_seed(seed, salt::NOISE, count as u64),
            };
            weight_sum += weight;
            weight *= k_rough;
            wave_m *= Gf::HALF;
            count += 1;
        }
        let mut o = 0;
        while o < count {
            octaves[o].amplitude_m = octaves[o].amplitude_m * relief_m / weight_sum;
            o += 1;
        }

        // The sea: between 40 % of the relief below the ladder radius and 30 % above it.
        let mut sea_rng = SplitMix64::new(child_seed(seed, salt::SEA, 0));
        let sea_offset =
            (draw_unit(&mut sea_rng) * Gf::from_f64(0.7) - Gf::from_f64(0.4)) * relief_m;
        let sea_radius_m = radius_m + sea_offset.floor();

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
        let biome = BiomeField {
            temperature_seed: biome_rng.next_u64(),
            temperature_wavelength_m: Gf::from_i64(i64::from(draw_m(
                &mut biome_rng,
                30_000,
                120_000,
            ))),
            humidity_seed: biome_rng.next_u64(),
            humidity_wavelength_m: Gf::from_i64(i64::from(draw_m(&mut biome_rng, 20_000, 80_000))),
            highland_above_m: relief_m * Gf::from_f64(0.55),
        };

        // The caves: a cavern field in a depth band, and tube carvers in regions.
        let mut cave_rng = SplitMix64::new(child_seed(seed, salt::CAVERN, 0));
        let cave_min_depth_m = draw_m(&mut cave_rng, 8, 30);
        let caves = Caves {
            cavern_wavelength_m: Gf::from_i64(i64::from(draw_m(&mut cave_rng, 24, 48))),
            cavern_threshold: Gf::from_f64(0.62) + draw_unit(&mut cave_rng) * Gf::from_f64(0.08),
            cavern_scale_m: Gf::from_i64(20),
            min_depth_m: cave_min_depth_m,
            max_depth_m: cave_min_depth_m + draw_m(&mut cave_rng, 200, 400),
            tube_region_m: 512,
            tube_seed: child_seed(seed, salt::TUBES, 0),
            tube_radius_m: Gf::from_i64(i64::from(draw_m(&mut cave_rng, 2, 5))),
        };

        // The band: the relief on both sides, the strata and the caves below, and room to stand on the
        // highest peak above. Whole metres, so the ladder floors them to whole cells at every rung.
        let relief_whole = relief_m.floor().to_i64_floor() as u32 + 1;
        let crust_m = relief_whole + strata.max_depth_m() + caves.max_depth_m + 64;
        let above_m = relief_whole + 64;
        let ladder = Ladder::for_radius(look_radius_m, crust_m, above_m)?;

        Some(BodyDefinition {
            seed,
            ladder,
            radius_m,
            sea_radius_m,
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

    /// The most the surface can rise or fall from the ladder radius at rung `rung`: the sum of the
    /// live amplitudes. An exact bound, because the noise is in `[−1, 1]`.
    #[must_use]
    pub fn relief_bound_m(&self, rung: u8) -> Gf {
        let mut sum = Gf::ZERO;
        for o in self.octaves_at(rung) {
            sum += o.amplitude_m;
        }
        sum
    }

    /// The most the surface at rung `rung` can differ from the surface at rung 0: the sum of the
    /// dropped amplitudes (the tier-agreement bound).
    #[must_use]
    pub fn dropped_bound_m(&self, rung: u8) -> Gf {
        self.relief_bound_m(0) - self.relief_bound_m(rung)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home() -> BodyDefinition {
        crate::home::home_planet()
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
        assert!(relief >= Gf::from_i64(100), "{relief:?}");
        assert!(relief <= Gf::from_i64(18_000), "{relief:?}");
        // Amplitudes fall and frequencies double, coarsest first.
        let live = m.octaves_at(0);
        let mut o = 1;
        while o < live.len() {
            assert!(live[o].amplitude_m < live[o - 1].amplitude_m);
            assert!(live[o].frequency > live[o - 1].frequency);
            o += 1;
        }
        assert!((m.sea_radius_m - m.radius_m).abs() <= relief);
        assert!(m.strata.max_depth_m() < 100);
        assert!(m.caves.min_depth_m < m.caves.max_depth_m);
        assert_eq!(m.ladder.rungs, 13, "the home planet's ladder");
        // The band holds the whole relief, the strata and the caves.
        let relief_whole = relief.floor().to_i64_floor() as u32 + 1;
        assert!(m.ladder.band_m >= 2 * relief_whole + m.strata.max_depth_m() + m.caves.max_depth_m);
        assert!(
            u64::from(m.ladder.floor_m) + u64::from(relief_whole)
                < m.radius_m.to_i64_floor() as u64
        );
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
    }

    #[test]
    fn every_rung_sums_strictly_fewer_octaves_than_the_rung_below_it() {
        // The octave's public face (the skyline march reads it): each accessor is the field.
        for o in crate::home::home_planet().octaves_at(0) {
            assert_eq!(o.frequency(), o.frequency);
            assert_eq!(o.amplitude_m(), o.amplitude_m);
            assert_eq!(o.seed(), o.seed);
            assert!(o.frequency() > Gf::ZERO);
        }
        let m = home();
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
                m.relief_bound_m(rung + 1) <= m.relief_bound_m(rung),
                "the band nests"
            );
            assert!(m.dropped_bound_m(rung + 1) >= m.dropped_bound_m(rung));
            rung += 1;
        }
        assert_eq!(m.dropped_bound_m(0), Gf::ZERO);
        // A tiny body: the relief cap's low end and a short octave table.
        let rock = BodyDefinition::from_seed(5, 3_000.0).expect("a rock");
        assert!(rock.octave_count >= 1);
        assert!(rock.relief_bound_m(0) >= Gf::from_i64(100));
        // A giant: the cap's high end and a long table.
        let giant = BodyDefinition::from_seed(9, 40_000_000.0).expect("a giant");
        assert!(giant.relief_bound_m(0) <= Gf::from_i64(18_000));
        assert!(
            giant.octave_count >= 12,
            "a 400 km wave halves to 30 m in fourteen steps"
        );
        assert_eq!(draw_m(&mut SplitMix64::new(1), 5, 6), 5);
    }
}
