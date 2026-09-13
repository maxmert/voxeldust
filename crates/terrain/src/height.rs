//! ★ THE HEIGHT FIELD — where the surface is along one direction, at one rung: the ladder radius plus
//! the sum of the live octaves of noise, ON INTEGERS (ruling F7). At rung `L` the `L` finest octaves
//! are dropped, so the far view is the same hill with the small bumps left out, cheaper by exactly
//! those octaves. The biome of a column reads the same field and two slow noises.
//!
//! **The unit.** [`height`] answers in GAP STEPS at [`crate::units::LENGTH_BITS`] fraction bits — the
//! radius word plus the octave sum, added without a shift because both carry the same bits, and
//! floored ONCE by the density (`chunk::finish_cell`). [`height_m`] and [`biome_at`] are the same two
//! answers for a host that holds its direction as floats: the doors of [`crate::units`] on either side
//! of the integer field, and the only floats this module names.
//!
//! **Example.** Along the direction of the pilot's boots the field at rung 3 is the rung-0 hill
//! without the last three ripples, and the two are never further apart than the dropped amplitudes
//! promise (the test below measures it on the home planet).

use crate::body::{BodyDefinition, RADIUS_RECIP_BITS as RECIP_BITS};
use crate::strata::Biome;
use crate::units::{LENGTH_BITS, direction_of_unit, greater, metres_of_q28, q28_of_metres};
use vd_recipe::Gi;
use vd_recipe::bend::DIR_BITS;
use vd_recipe::height::relief;
use vd_recipe::noise::{NOISE_BITS, NOISE_ONE};

/// The surface's radius along a unit direction at a rung, in GAP STEPS at
/// [`crate::units::LENGTH_BITS`] fraction bits. `dir` carries the bend's 40 fraction bits.
#[must_use]
pub fn height(body: &BodyDefinition, dir: [Gi; 3], rung: u8) -> Gi {
    body.radius + relief(body.octaves_at(rung), dir)
}

/// THE FLOAT SEAM of the height field: the surface's radius in METRES along a direction a host holds
/// as floats — a camera's radial, a ray from a pilot's boots. The direction enters through
/// [`crate::units::direction_of_unit`] and the answer leaves through [`crate::units::metres_of_q28`];
/// the shape between them is the integer recipe's.
#[must_use]
pub fn height_m(body: &BodyDefinition, dir: [f64; 3], rung: u8) -> f64 {
    metres_of_q28(height(body, direction_of_unit(dir), rung))
}

/// THE POLE AXIS of every body: `+Z` in the body's own frame — the axis the world's orbits turn
/// about (`crates/physics/src/celestial.rs`: the perifocal plane is `z = 0`), so ice caps face away
/// from the orbital plane, never into it. An obliquity the parent authors per body is a later
/// slice; until then every body's spin axis is its orbit's axis. Cross-pinned in
/// `crates/bins/tests/home_body_pin.rs`.
pub const POLE_AXIS: usize = 2;

/// The climate constants, at the noise's fraction bits: how much of the slow temperature noise the
/// climate reads, how much height cools a column, and the two thresholds the biomes stand on. Each is
/// `round(share · 2²⁸)` of the share the float recipe read.
const NOISE_SHARE: Gi = Gi::new(93_952_410); // 0.35
const HEIGHT_SHARE: Gi = Gi::new(80_530_637); // 0.30
const COLD_BELOW: Gi = Gi::new(93_952_410); // 0.35
const WARM_ABOVE: Gi = Gi::new(201_326_592); // 0.75
const DRY_BELOW: Gi = Gi::new(-26_843_546); // −0.10

/// The biome of a column: cold near the poles and high up, dry where the humidity noise says so, and
/// highland where the surface stands far above the sea. `surface` is the column's surface radius in
/// gap steps at [`crate::units::LENGTH_BITS`], as [`height`] answers it.
#[must_use]
pub fn biome_of(body: &BodyDefinition, dir: [Gi; 3], surface: Gi) -> Biome {
    let above_sea = surface - body.sea_radius;
    if above_sea > body.biome.highland_above {
        return Biome::Highland;
    }
    // The latitude: the pole component's magnitude, at the noise's fraction bits.
    let latitude = Gi::new(dir[POLE_AXIS].unsigned_abs() as i64) >> (DIR_BITS - NOISE_BITS);
    let t_noise = relief(&[body.biome.temperature], dir);
    // The height share: the gap steps over the sea against the highland height, one multiply by the
    // charter's reciprocal. Only a column above the sea is cooled by its height. The reciprocal
    // carries RADIUS_RECIP_BITS, so the shift leaves the share at the noise's own fraction bits.
    let over_sea_steps = greater(above_sea, Gi::ZERO) >> LENGTH_BITS;
    let height_share = over_sea_steps.mul_shr(body.biome.highland_recip, RECIP_BITS - NOISE_BITS);
    // Warm at the equator, cold at the poles, plus a slow noise; cooler with height.
    let temperature = NOISE_ONE - latitude + t_noise.mul_shr(NOISE_SHARE, NOISE_BITS)
        - height_share.mul_shr(HEIGHT_SHARE, NOISE_BITS);
    if temperature < COLD_BELOW {
        return Biome::Tundra;
    }
    let humidity = relief(&[body.biome.humidity], dir);
    if (temperature > WARM_ABOVE) & (humidity < DRY_BELOW) {
        return Biome::Desert;
    }
    Biome::Grassland
}

/// THE FLOAT SEAM of the biome field: the biome of a column a host names with a float direction and a
/// surface radius in metres.
#[must_use]
pub fn biome_at(body: &BodyDefinition, dir: [f64; 3], surface_m: f64) -> Biome {
    biome_of(body, direction_of_unit(dir), q28_of_metres(surface_m))
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
    use vd_seed::bend::Face;

    fn home() -> BodyDefinition {
        crate::home::home_planet()
    }

    /// The direction of a face position, through the recipe's own bend: the cell nearest `(a, b)` of
    /// a face at rung 0 (the test names positions, the recipe names cells).
    fn dir(face: Face, a: f64, b: f64) -> [Gi; 3] {
        let m = home();
        let n_l = m.ladder().cells_per_edge(0);
        let i = vd_seed::ladder::index_of(a, n_l);
        let j = vd_seed::ladder::index_of(b, n_l);
        vd_seed::bend::direction_q(face, i, j, m.inv_n(0))
    }

    #[test]
    fn the_surface_stays_inside_the_band_and_coarser_rungs_stay_within_the_dropped_bound() {
        let m = home();
        let mut i = 0u32;
        while i < 400 {
            let a = -1.0 + f64::from(i % 20) / 10.0;
            let b = -1.0 + f64::from(i / 20) / 10.0;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            let h0 = height(&m, d, 0);
            assert!(
                Gi::new((h0 - m.radius).unsigned_abs() as i64) <= m.relief_bound(0),
                "inside the band"
            );
            let mut rung = 1u8;
            while rung < m.ladder.rungs {
                let hl = height(&m, d, rung);
                assert!(
                    Gi::new((hl - h0).unsigned_abs() as i64) <= m.dropped_bound(rung),
                    "rung {rung}: {hl:?} vs {h0:?}"
                );
                assert!(Gi::new((hl - m.radius).unsigned_abs() as i64) <= m.relief_bound(rung));
                rung += 1;
            }
            i += 1;
        }
        assert_eq!(
            height(&m, dir(Face::PosX, 0.2, 0.3), 0),
            height(&m, dir(Face::PosX, 0.2, 0.3), 0)
        );
    }

    /// The float seam: the same answer as the integer path, in metres, for a direction a host holds as
    /// floats. The exactness of the unit is `units`'s own test; this one proves the two doors meet.
    #[test]
    fn the_float_seam_reads_the_integer_answer_in_metres() {
        let m = home();
        // A direction a camera holds: the +X axis, which the door in turns into the bend's own word.
        assert_eq!(
            height_m(&m, [1.0, 0.0, 0.0], 0),
            metres_of_q28(height(&m, [Gi::new(1 << DIR_BITS), Gi::ZERO, Gi::ZERO], 0))
        );
        let surface_m = height_m(&m, [1.0, 0.0, 0.0], 0);
        assert_eq!(
            biome_at(&m, [1.0, 0.0, 0.0], surface_m),
            biome_of(
                &m,
                [Gi::new(1 << DIR_BITS), Gi::ZERO, Gi::ZERO],
                q28_of_metres(surface_m)
            )
        );
    }

    #[test]
    fn every_biome_appears_on_the_moon_and_the_poles_are_cold() {
        let m = home();
        let mut seen = [false; 4];
        let mut i = 0u32;
        while i < 2_000 {
            let a = -1.0 + f64::from(i % 40) / 20.0;
            let b = -1.0 + f64::from((i / 40) % 40) / 20.0;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            let h = height(&m, d, 0);
            seen[biome_of(&m, d, h) as usize] = true;
            i += 1;
        }
        // Forced cases, so every arm is driven whatever the seed draws.
        let pole = dir(Face::PosZ, 0.0, 0.0);
        assert_eq!(
            biome_of(&m, pole, m.sea_radius),
            Biome::Tundra,
            "the pole is cold"
        );
        let anywhere = dir(Face::PosX, 0.1, 0.1);
        assert_eq!(
            biome_of(
                &m,
                anywhere,
                m.sea_radius + m.biome.highland_above + Gi::ONE
            ),
            Biome::Highland,
            "far above the sea is highland"
        );
        assert!(seen[Biome::Grassland as usize], "grassland exists");
        assert!(seen[Biome::Tundra as usize]);
        assert!(seen[Biome::Highland as usize], "highland exists");
        // A desert is a warm dry equator cell; find one by scanning the equator of the +X face.
        let mut desert = false;
        let mut j = 0;
        while j < 2_000 {
            let d = dir(Face::PosX, -1.0 + f64::from(j) / 1_000.0, 0.0);
            desert |= biome_of(&m, d, m.sea_radius) == Biome::Desert;
            j += 1;
        }
        assert!(
            desert,
            "the home planet's equator holds a desert somewhere on the +X face"
        );
        // ★ MEASURED: HEIGHT COOLS A COLUMN. The same column at the sea and just under the highland
        // height differs in nothing but its height share, so a column whose biome changes between the
        // two proves the share carries the noise's fraction bits. (It read ZERO on the first run of
        // the integer recipe: the reciprocal's shift left the share a whole number, so every share
        // floored to nothing and height cooled no column at all.)
        let just_under = m.biome.highland_above - Gi::ONE;
        let mut cooled = 0;
        let mut j = 0;
        while j < 2_000 {
            let d = dir(Face::PosX, -1.0 + f64::from(j) / 1_000.0, 0.0);
            if biome_of(&m, d, m.sea_radius) != biome_of(&m, d, m.sea_radius + just_under) {
                cooled += 1;
            }
            j += 1;
        }
        assert!(cooled > 0, "height cools some column: {cooled}");
    }

    /// The climate constants are the shares the float recipe read, rounded once to the noise's bits.
    #[test]
    fn the_climate_constants_are_the_shares_at_the_noises_bits() {
        let one = f64::from(1u32 << NOISE_BITS);
        for (word, share) in [
            (NOISE_SHARE, 0.35),
            (HEIGHT_SHARE, 0.30),
            (COLD_BELOW, 0.35),
            (WARM_ABOVE, 0.75),
            (DRY_BELOW, -0.10),
        ] {
            assert_eq!(word.raw(), (share * one).round() as i64, "{share}");
        }
    }
}
