/// Deterministic star system generation and orbital mechanics.
///
/// Uses the brahe crate (MIT) for Kepler equation solving and
/// orbital element → Cartesian state conversion.
use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::seed::{derive_seed, seed_to_f64, seed_to_range, seed_to_u32};

/// Gravitational constant in m³/(kg·s²).
pub const G: f64 = 6.674e-11;

/// Hill sphere SOI factor: r_soi = a * (m_planet / m_star)^(2/5).
pub const SOI_EXPONENT: f64 = 0.4; // 2/5

/// Keplerian orbital elements for a planet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalElements {
    /// Semi-major axis in meters.
    pub sma: f64,
    /// Eccentricity (0 = circular, <1 = elliptical).
    pub eccentricity: f64,
    /// Inclination in radians.
    pub inclination: f64,
    /// Longitude of ascending node in radians.
    pub raan: f64,
    /// Argument of periapsis in radians.
    pub arg_periapsis: f64,
    /// Mean anomaly at epoch (t=0) in radians.
    pub mean_anomaly_epoch: f64,
}

/// Parameters for a star.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarParams {
    pub mass_kg: f64,
    pub radius_m: f64,
    pub color: [f32; 3],
    pub luminosity: f64,
    /// Gravitational parameter GM (m³/s²).
    pub gm: f64,
}

/// Parameters for a planet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanetParams {
    pub index: u32,
    pub planet_seed: u64,
    pub orbital_elements: OrbitalElements,
    pub mass_kg: f64,
    pub radius_m: f64,
    pub color: [f32; 3],
    /// Gravitational parameter GM (m³/s²).
    pub gm: f64,
    /// Orbital period in seconds.
    pub period: f64,
}

/// Complete star system parameters, generated deterministically from a seed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemParams {
    pub system_seed: u64,
    pub star: StarParams,
    pub planets: Vec<PlanetParams>,
}

/// Lighting information computed by the server, sent to the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingInfo {
    /// Normalized direction from observer toward the star.
    pub sun_direction: DVec3,
    /// Star color (RGB, 0-1).
    pub sun_color: [f32; 3],
    /// Distance-attenuated intensity (0-1).
    pub sun_intensity: f32,
    /// Base ambient light level.
    pub ambient: f32,
}

impl SystemParams {
    /// Generate a complete star system deterministically from a seed.
    pub fn from_seed(system_seed: u64) -> Self {
        let star_seed = derive_seed(system_seed, 0);
        let planet_count_seed = derive_seed(system_seed, 1);

        // Star parameters (Sun-like by default, varied by seed).
        let star_mass = seed_to_range(derive_seed(star_seed, 0), 0.5e30, 4.0e30); // 0.25-2x solar
        let star_radius = seed_to_range(derive_seed(star_seed, 1), 3.5e8, 1.4e9); // 0.5-2x solar
        let star_luminosity = (star_mass / 1.989e30).powf(3.5); // mass-luminosity relation

        // Star color from temperature (approximation).
        let temp_factor = (star_mass / 1.989e30).powf(0.5);
        let star_color = if temp_factor > 1.5 {
            [0.7, 0.8, 1.0] // blue-white
        } else if temp_factor > 0.8 {
            [1.0, 0.95, 0.8] // yellow-white
        } else {
            [1.0, 0.6, 0.3] // orange-red
        };

        let star = StarParams {
            mass_kg: star_mass,
            radius_m: star_radius,
            color: star_color,
            luminosity: star_luminosity,
            gm: G * star_mass,
        };

        // Generate 2-8 planets.
        let planet_count = 2 + seed_to_u32(planet_count_seed, 7); // 2-8

        let mut planets = Vec::with_capacity(planet_count as usize);
        for i in 0..planet_count {
            let planet_seed = derive_seed(system_seed, 100 + i);
            planets.push(generate_planet(i, planet_seed, &star));
        }

        Self {
            system_seed,
            star,
            planets,
        }
    }
}

/// Generate a single planet's parameters.
fn generate_planet(index: u32, planet_seed: u64, star: &StarParams) -> PlanetParams {
    // Semi-major axis: planets spread out logarithmically (Titius-Bode-like).
    let base_sma = 5.0e10; // ~0.33 AU
    let sma = base_sma * (1.5_f64).powf(index as f64 + seed_to_range(derive_seed(planet_seed, 0), 0.0, 1.0));

    let eccentricity = seed_to_range(derive_seed(planet_seed, 1), 0.0, 0.3);
    let inclination = seed_to_range(derive_seed(planet_seed, 2), -0.1, 0.1); // near-planar
    let raan = seed_to_range(derive_seed(planet_seed, 3), 0.0, std::f64::consts::TAU);
    let arg_periapsis = seed_to_range(derive_seed(planet_seed, 4), 0.0, std::f64::consts::TAU);
    let mean_anomaly_epoch = seed_to_range(derive_seed(planet_seed, 5), 0.0, std::f64::consts::TAU);

    // Planet mass: Earth-like range (0.1-10 Earth masses).
    let mass_kg = seed_to_range(derive_seed(planet_seed, 6), 5.97e23, 5.97e25);

    // Planet radius: derived from mass (rough density model).
    let density = seed_to_range(derive_seed(planet_seed, 7), 3000.0, 6000.0); // kg/m³
    let volume = mass_kg / density;
    let radius_m = (3.0 * volume / (4.0 * std::f64::consts::PI)).powf(1.0 / 3.0);

    // Planet color.
    let color_seed = seed_to_u32(derive_seed(planet_seed, 8), 5);
    let color = match color_seed {
        0 => [0.2, 0.4, 0.8],  // blue (ocean)
        1 => [0.3, 0.6, 0.2],  // green (earthlike)
        2 => [0.8, 0.5, 0.2],  // orange (desert)
        3 => [0.6, 0.6, 0.7],  // grey (rocky)
        _ => [0.9, 0.85, 0.7], // pale (ice)
    };

    let gm = G * mass_kg;

    // Orbital period from Kepler's third law: T = 2π√(a³/GM_star).
    let period = std::f64::consts::TAU * (sma.powi(3) / star.gm).sqrt();

    PlanetParams {
        index,
        planet_seed,
        orbital_elements: OrbitalElements {
            sma,
            eccentricity,
            inclination,
            raan,
            arg_periapsis,
            mean_anomaly_epoch,
        },
        mass_kg,
        radius_m,
        color,
        gm,
        period,
    }
}

/// Compute a planet's 3D position at a given time using Kepler's equation.
/// Uses brahe for mean→eccentric anomaly conversion.
pub fn compute_planet_position(planet: &PlanetParams, time_s: f64) -> DVec3 {
    let oe = &planet.orbital_elements;

    // Mean anomaly at time t.
    let mean_motion = std::f64::consts::TAU / planet.period;
    let mean_anomaly = (oe.mean_anomaly_epoch + mean_motion * time_s) % std::f64::consts::TAU;

    use brahe::constants::units::AngleFormat;

    // Solve Kepler's equation: M = E - e*sin(E) → find E.
    let ecc_anomaly = brahe::orbits::keplerian::anomaly_mean_to_eccentric(
        mean_anomaly,
        oe.eccentricity,
        AngleFormat::Radians,
    )
    .expect("Kepler equation failed to converge");

    // Eccentric anomaly → true anomaly.
    let true_anomaly = brahe::orbits::keplerian::anomaly_eccentric_to_true(
        ecc_anomaly,
        oe.eccentricity,
        AngleFormat::Radians,
    );

    // Convert Keplerian elements to Cartesian position using brahe.
    // brahe expects: [sma, ecc, inc, raan, aop, true_anomaly] in radians.
    let koe = nalgebra::SVector::<f64, 6>::new(
        oe.sma,
        oe.eccentricity,
        oe.inclination,
        oe.raan,
        oe.arg_periapsis,
        true_anomaly,
    );

    let state = brahe::coordinates::cartesian::state_koe_to_eci(
        koe,
        AngleFormat::Radians,
    );

    // state = [x, y, z, vx, vy, vz] — we only need position.
    DVec3::new(state[0], state[1], state[2])
}

/// Compute gravitational acceleration at a position from all bodies.
/// Returns acceleration vector in m/s².
pub fn compute_gravity_acceleration(
    position: DVec3,
    star: &StarParams,
    planets: &[PlanetParams],
    planet_positions: &[DVec3],
    time_s: f64,
) -> DVec3 {
    let mut accel = DVec3::ZERO;

    // Star gravity (star is at origin).
    let r_star = position;
    let dist_sq = r_star.length_squared();
    if dist_sq > 1.0 {
        accel -= r_star.normalize() * star.gm / dist_sq;
    }

    // Planet gravity.
    for (planet, &planet_pos) in planets.iter().zip(planet_positions.iter()) {
        let r = position - planet_pos;
        let dist_sq = r.length_squared();
        if dist_sq > 1.0 {
            accel -= r.normalize() * planet.gm / dist_sq;
        }
    }

    accel
}

/// Compute the sphere-of-influence radius for a planet.
/// Uses Hill sphere approximation: r_soi = a * (m_planet / m_star)^(2/5).
pub fn compute_soi_radius(planet: &PlanetParams, star: &StarParams) -> f64 {
    planet.orbital_elements.sma * (planet.mass_kg / star.mass_kg).powf(SOI_EXPONENT)
}

/// Compute the sun direction from an observer position.
/// Returns a normalized vector pointing from the observer toward the star.
/// The star is at origin in system coordinates.
pub fn compute_sun_direction(observer_pos: DVec3) -> DVec3 {
    if observer_pos.length_squared() < 1.0 {
        return DVec3::Y; // fallback if at star
    }
    -observer_pos.normalize()
}

/// Compute lighting info for an observer at a given position.
pub fn compute_lighting(observer_pos: DVec3, star: &StarParams) -> LightingInfo {
    let sun_direction = compute_sun_direction(observer_pos);
    let dist = observer_pos.length();

    // Inverse square falloff normalized to 1.0 at 1 AU (1.496e11 m).
    let au = 1.496e11;
    let intensity = (star.luminosity * au * au / (dist * dist)).min(1.0) as f32;

    LightingInfo {
        sun_direction,
        sun_color: star.color,
        sun_intensity: intensity.max(0.001), // never fully dark
        ambient: 0.08,
    }
}

/// Convert a system-local position to planet-local coordinates.
/// Planet-local: planet center is at origin.
pub fn system_to_planet_local(system_pos: DVec3, planet_pos: DVec3) -> DVec3 {
    system_pos - planet_pos
}

/// Convert planet-local position to system coordinates.
pub fn planet_local_to_system(local_pos: DVec3, planet_pos: DVec3) -> DVec3 {
    local_pos + planet_pos
}

/// Compute a spawn position on a planet's surface (2m above surface).
pub fn surface_spawn_position(planet_radius: f64) -> DVec3 {
    DVec3::new(0.0, planet_radius + 2.0, 0.0) // spawn at "north pole" + 2m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_system_generation() {
        let a = SystemParams::from_seed(42);
        let b = SystemParams::from_seed(42);
        assert_eq!(a.planets.len(), b.planets.len());
        assert_eq!(a.star.mass_kg, b.star.mass_kg);
        for (pa, pb) in a.planets.iter().zip(b.planets.iter()) {
            assert_eq!(pa.orbital_elements.sma, pb.orbital_elements.sma);
            assert_eq!(pa.mass_kg, pb.mass_kg);
        }
    }

    #[test]
    fn planet_count_in_range() {
        for seed in 0..20 {
            let sys = SystemParams::from_seed(seed);
            assert!(
                sys.planets.len() >= 2 && sys.planets.len() <= 8,
                "seed {seed}: {} planets",
                sys.planets.len()
            );
        }
    }

    #[test]
    fn planets_orbit_increases_with_index() {
        let sys = SystemParams::from_seed(42);
        for i in 1..sys.planets.len() {
            assert!(
                sys.planets[i].orbital_elements.sma > sys.planets[i - 1].orbital_elements.sma,
                "planet {} sma not greater than planet {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn kepler_orbit_returns_to_start() {
        let sys = SystemParams::from_seed(42);
        let planet = &sys.planets[0];

        let pos_0 = compute_planet_position(planet, 0.0);
        let pos_half = compute_planet_position(planet, planet.period / 2.0);
        let pos_full = compute_planet_position(planet, planet.period);

        // After one full period, should return near starting position.
        let diff = (pos_full - pos_0).length();
        assert!(
            diff < 1000.0, // within 1km (numerical precision)
            "orbit didn't close: diff = {diff} m"
        );

        // Halfway should be on opposite side.
        let half_diff = (pos_half - pos_0).length();
        assert!(
            half_diff > planet.orbital_elements.sma * 0.5,
            "halfway position too close to start"
        );
    }

    #[test]
    fn soi_radius_reasonable() {
        let sys = SystemParams::from_seed(42);
        for planet in &sys.planets {
            let soi = compute_soi_radius(planet, &sys.star);
            // SOI should be smaller than the orbit but larger than the planet.
            assert!(soi > planet.radius_m, "SOI smaller than planet radius");
            assert!(
                soi < planet.orbital_elements.sma,
                "SOI larger than orbit"
            );
        }
    }

    #[test]
    fn sun_direction_points_toward_star() {
        let observer = DVec3::new(1e11, 0.0, 0.0);
        let sun_dir = compute_sun_direction(observer);
        // Should point toward origin (star).
        assert!(sun_dir.x < 0.0);
        assert!((sun_dir.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn lighting_intensity_decreases_with_distance() {
        let sys = SystemParams::from_seed(42);
        let close = compute_lighting(DVec3::new(1e10, 0.0, 0.0), &sys.star);
        let far = compute_lighting(DVec3::new(1e12, 0.0, 0.0), &sys.star);
        assert!(close.sun_intensity > far.sun_intensity);
    }

    #[test]
    fn coordinate_transforms_invertible() {
        let planet_pos = DVec3::new(1e11, 0.0, 5e10);
        let system_pos = DVec3::new(1e11 + 1000.0, 500.0, 5e10 + 200.0);

        let local = system_to_planet_local(system_pos, planet_pos);
        let back = planet_local_to_system(local, planet_pos);
        assert!((back - system_pos).length() < 1e-6);
    }

    #[test]
    fn gravity_points_toward_star() {
        let sys = SystemParams::from_seed(42);
        let pos = DVec3::new(1e11, 0.0, 0.0);
        let positions: Vec<DVec3> = sys
            .planets
            .iter()
            .map(|p| compute_planet_position(p, 0.0))
            .collect();
        let accel = compute_gravity_acceleration(pos, &sys.star, &sys.planets, &positions, 0.0);
        // Primary gravity should point toward origin (star dominates at this distance).
        assert!(accel.x < 0.0, "gravity should point toward star");
    }
}
