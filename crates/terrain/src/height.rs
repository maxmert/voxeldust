//! ★ THE HEIGHT FIELD — where the surface is along one direction, at one rung: the ladder radius plus
//! the sum of the live octaves of noise. At rung `L` the `L` finest octaves are dropped, so the far
//! view is the same hill with the small bumps left out, cheaper by exactly those octaves. The biome of
//! a column reads the same field and two slow noises.
//!
//! **Example.** Along the direction of the pilot's boots the field at rung 3 is the rung-0 hill
//! without the last three ripples, and the two are never further apart than the dropped amplitudes
//! promise (the test below measures it on the home planet).

use crate::body::BodyDefinition;
use crate::gf::Gf;
use crate::noise::noise3;
use crate::strata::Biome;

/// The surface's radius along a unit direction at a rung, in metres.
#[must_use]
pub fn height_m(body: &BodyDefinition, dir: [Gf; 3], rung: u8) -> Gf {
    let mut h = body.radius_m;
    for o in body.octaves_at(rung) {
        let p = [
            dir[0] * o.frequency,
            dir[1] * o.frequency,
            dir[2] * o.frequency,
        ];
        h += o.amplitude_m * noise3(o.seed, p);
    }
    h
}

/// THE POLE AXIS of every body: `+Z` in the body's own frame — the axis the world's orbits turn
/// about (`crates/physics/src/celestial.rs`: the perifocal plane is `z = 0`), so ice caps face away
/// from the orbital plane, never into it. An obliquity the parent authors per body is a later
/// slice; until then every body's spin axis is its orbit's axis. Cross-pinned in
/// `crates/bins/tests/home_body_pin.rs`.
pub const POLE_AXIS: usize = 2;

/// The biome of a column: cold near the poles and high up, dry where the humidity noise says so, and
/// highland where the surface stands far above the sea.
#[must_use]
pub fn biome_at(body: &BodyDefinition, dir: [Gf; 3], surface_m: Gf) -> Biome {
    let above_sea = surface_m - body.sea_radius_m;
    if above_sea > body.biome.highland_above_m {
        return Biome::Highland;
    }
    let latitude = dir[POLE_AXIS].abs();
    let t_freq = body.radius_m / body.biome.temperature_wavelength_m;
    let t_noise = noise3(
        body.biome.temperature_seed,
        [dir[0] * t_freq, dir[1] * t_freq, dir[2] * t_freq],
    );
    // Warm at the equator, cold at the poles, plus a slow noise; cooler with height.
    let temperature = Gf::ONE - latitude + t_noise * Gf::from_f64(0.35)
        - above_sea.greater(Gf::ZERO) / (body.biome.highland_above_m + Gf::ONE) * Gf::from_f64(0.3);
    if temperature < Gf::from_f64(0.35) {
        return Biome::Tundra;
    }
    let h_freq = body.radius_m / body.biome.humidity_wavelength_m;
    let humidity = noise3(
        body.biome.humidity_seed,
        [dir[0] * h_freq, dir[1] * h_freq, dir[2] * h_freq],
    );
    if temperature > Gf::from_f64(0.75) && humidity < Gf::from_f64(-0.1) {
        return Biome::Desert;
    }
    Biome::Grassland
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_seed::bend::{Face, direction};

    fn home() -> BodyDefinition {
        crate::home::home_planet()
    }

    fn dir(face: Face, a: f64, b: f64) -> [Gf; 3] {
        let d = direction(face, a, b);
        [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])]
    }

    #[test]
    fn the_surface_stays_inside_the_band_and_coarser_rungs_stay_within_the_dropped_bound() {
        let m = home();
        let mut i = 0u32;
        while i < 400 {
            let a = -1.0 + f64::from(i % 20) / 10.0;
            let b = -1.0 + f64::from(i / 20) / 10.0;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            let h0 = height_m(&m, d, 0);
            assert!(
                (h0 - m.radius_m).abs() <= m.relief_bound_m(0),
                "inside the band"
            );
            let mut rung = 1u8;
            while rung < m.ladder.rungs {
                let hl = height_m(&m, d, rung);
                assert!(
                    (hl - h0).abs() <= m.dropped_bound_m(rung),
                    "rung {rung}: {hl:?} vs {h0:?}"
                );
                assert!((hl - m.radius_m).abs() <= m.relief_bound_m(rung));
                rung += 1;
            }
            i += 1;
        }
        assert_eq!(
            height_m(&m, dir(Face::PosX, 0.2, 0.3), 0),
            height_m(&m, dir(Face::PosX, 0.2, 0.3), 0)
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
            let h = height_m(&m, d, 0);
            seen[biome_at(&m, d, h) as usize] = true;
            i += 1;
        }
        // Forced cases, so every arm is driven whatever the seed draws.
        let pole = dir(Face::PosZ, 0.0, 0.0);
        assert_eq!(
            biome_at(&m, pole, m.sea_radius_m),
            Biome::Tundra,
            "the pole is cold"
        );
        let anywhere = dir(Face::PosX, 0.1, 0.1);
        assert_eq!(
            biome_at(
                &m,
                anywhere,
                m.sea_radius_m + m.biome.highland_above_m + Gf::ONE
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
            desert |= biome_at(&m, d, m.sea_radius_m) == Biome::Desert;
            j += 1;
        }
        assert!(
            desert,
            "the home planet's equator holds a desert somewhere on the +X face"
        );
    }
}
