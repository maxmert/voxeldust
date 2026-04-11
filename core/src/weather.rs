//! Deterministic weather simulation — produces a 2D weather map per planet.
//!
//! The weather map drives cloud formation: WHERE clouds appear, their type,
//! and density. All outputs are pure functions of `(planet_seed, game_time,
//! planet_params)` — no random state, identical on all clients.
//!
//! ## Architecture
//!
//! Two layers combined multiplicatively:
//! 1. **Hadley cells** — latitude bands from atmospheric circulation (analytical)
//! 2. **Pressure cells** — deterministic cyclones/anticyclones placed from seed
//!
//! Future: Layer 3 (terrain interaction — mountains, oceans) when heightmap is available.

use glam::DVec3;

use crate::seed::{derive_seed, seed_to_f64, seed_to_range};
use crate::system::PlanetParams;

/// Earth's sidereal rotation rate (radians/second) for scaling Hadley cell count.
const EARTH_ROTATION_RATE: f64 = 7.2921e-5;

/// Weather map resolution (one side of the equirectangular projection).
pub const WEATHER_MAP_WIDTH: u32 = 512;
pub const WEATHER_MAP_HEIGHT: u32 = 256;

/// A pressure cell on the planet surface (cyclone or anticyclone).
/// Position evolves deterministically with game_time.
#[derive(Debug, Clone)]
struct PressureCell {
    /// Base position on unit sphere (from seed, at time=0).
    base_lat: f64,
    base_lon: f64,
    /// Angular radius of influence (radians). Typically 0.1-0.5 (~6°-30°).
    radius: f64,
    /// Strength: negative = low pressure (cyclone, clouds), positive = high pressure (anticyclone, clear).
    strength: f64,
    /// Longitude drift rate (radians/second). Latitude-dependent from Coriolis.
    drift_rate: f64,
}

impl PressureCell {
    /// Get the cell's position at a given time.
    fn position_at(&self, game_time: f64) -> (f64, f64) {
        let lon = self.base_lon + self.drift_rate * game_time;
        // Wrap longitude to [-PI, PI].
        let lon = ((lon + std::f64::consts::PI) % std::f64::consts::TAU) - std::f64::consts::PI;
        (self.base_lat, lon)
    }

    /// Compute this cell's influence at a given lat/lon. Returns [-1, 1]:
    /// negative = cloud-forming, positive = cloud-suppressing.
    fn influence_at(&self, lat: f64, lon: f64, game_time: f64) -> f64 {
        let (cell_lat, cell_lon) = self.position_at(game_time);

        // Angular distance on unit sphere (Haversine-like).
        let dlat = lat - cell_lat;
        let dlon = lon - cell_lon;
        let a = (dlat / 2.0).sin().powi(2)
            + lat.cos() * cell_lat.cos() * (dlon / 2.0).sin().powi(2);
        let angular_dist = 2.0 * a.sqrt().asin();

        // Gaussian falloff from cell center.
        if angular_dist > self.radius * 3.0 {
            return 0.0; // too far, no influence
        }
        let falloff = (-angular_dist * angular_dist / (2.0 * self.radius * self.radius)).exp();
        self.strength * falloff
    }
}

/// Generate deterministic pressure cells for a planet.
fn generate_pressure_cells(planet_seed: u64, cloud_params: &crate::system::CloudParams) -> Vec<PressureCell> {
    // Number of cells scales with planet's weather activity.
    // Use seeds 100-199 for weather cells (avoid collision with other seed indices).
    let num_cells_seed = derive_seed(planet_seed, 100);
    let num_cells = 12 + (seed_to_f64(num_cells_seed) * 18.0) as usize; // 12-30 cells

    let mut cells = Vec::with_capacity(num_cells);

    for i in 0..num_cells {
        let cell_seed = derive_seed(planet_seed, 101 + i as u32);

        // Position: distributed across the planet, biased toward mid-latitudes
        // where weather systems are most active.
        let lat_raw = seed_to_range(derive_seed(cell_seed, 0), -1.0, 1.0);
        // Bias toward mid-latitudes (±30°-60°) where most weather activity occurs.
        let lat = lat_raw.signum() * (lat_raw.abs().sqrt()) * std::f64::consts::FRAC_PI_2;
        let lon = seed_to_range(derive_seed(cell_seed, 1), -std::f64::consts::PI, std::f64::consts::PI);

        // Radius: 0.1-0.5 radians (~6°-30°, ~600-3000km on Earth).
        let radius = seed_to_range(derive_seed(cell_seed, 2), 0.1, 0.5);

        // Strength: mix of cyclones (negative, cloudy) and anticyclones (positive, clear).
        // Slight bias toward cyclones for more cloud activity.
        let strength_raw = seed_to_range(derive_seed(cell_seed, 3), -1.0, 0.8);
        let strength = strength_raw;

        // Drift rate: cells drift eastward at mid-latitudes (Rossby wave propagation).
        // Rate depends on latitude: faster at mid-latitudes, slower near equator/poles.
        let wind_speed = cloud_params.wind_velocity[0].abs() + cloud_params.wind_velocity[2].abs();
        let coriolis_factor = lat.sin().abs().max(0.1);
        let base_drift = wind_speed * 0.001 * coriolis_factor; // slow drift
        let drift_rate = seed_to_range(derive_seed(cell_seed, 4), -base_drift, base_drift * 2.0);

        cells.push(PressureCell {
            base_lat: lat,
            base_lon: lon,
            radius,
            strength,
            drift_rate,
        });
    }

    cells
}

/// Hadley cell cloud probability based on latitude and planet rotation.
///
/// Produces alternating bands of high/low cloud probability matching
/// real atmospheric circulation patterns. Fast-rotating planets get
/// more bands (like Jupiter's stripes).
fn hadley_cloud_probability(latitude: f64, rotation_period: f64) -> f64 {
    let abs_lat = latitude.abs();

    // Rotation rate in radians/second.
    let rotation_rate = std::f64::consts::TAU / rotation_period.max(1.0);

    // Number of Hadley-like cells scales with rotation rate.
    // Earth (24h): 3 cells. Slow rotator (240h): 1 cell. Fast (2.4h): ~7 cells.
    let cell_count = (rotation_rate / EARTH_ROTATION_RATE).sqrt().clamp(1.0, 7.0);

    // Generate alternating high/low bands.
    // Each cell creates a cloudy zone (ascending air) and a clear zone (descending).
    let band_width = std::f64::consts::FRAC_PI_2 / cell_count;
    let band_phase = abs_lat / band_width;

    // Cosine produces smooth alternation: peaks = cloudy (ITCZ-like), troughs = clear (subtropical high).
    let band_value = 0.5 + 0.5 * (band_phase * std::f64::consts::PI).cos();

    // Poles are always drier (descending air).
    let polar_suppression = (abs_lat / std::f64::consts::FRAC_PI_2).powi(2);
    let result = band_value * (1.0 - polar_suppression * 0.7);

    result.clamp(0.0, 1.0)
}

/// Per-texel weather data packed into RGBA8.
#[derive(Debug, Clone, Copy)]
pub struct WeatherTexel {
    /// Cloud coverage (0 = clear, 1 = overcast).
    pub coverage: f32,
    /// Cloud type (0 = stratus, 0.5 = cumulus, 1 = cumulonimbus).
    pub cloud_type: f32,
    /// Precipitation intensity (0 = none, 1 = heavy).
    pub precipitation: f32,
    /// Wind strength modifier (0.5 = calm, 1.5 = stormy).
    pub wind_modifier: f32,
}

/// A 2D weather map in equirectangular projection.
pub struct WeatherMap {
    pub width: u32,
    pub height: u32,
    /// Raw RGBA8 pixel data (width × height × 4 bytes), ready for GPU upload.
    pub data: Vec<u8>,
}

impl WeatherMap {
    /// Generate a deterministic weather map for a planet at a given time.
    ///
    /// The output is identical for the same `(planet_seed, game_time, planet_params)`.
    pub fn generate(planet: &PlanetParams, game_time: f64) -> Self {
        let width = WEATHER_MAP_WIDTH;
        let height = WEATHER_MAP_HEIGHT;
        let clouds = &planet.clouds;

        if !clouds.has_clouds {
            // Planet has no clouds — return empty (all clear).
            return Self {
                width,
                height,
                data: vec![0u8; (width * height * 4) as usize],
            };
        }

        let cells = generate_pressure_cells(planet.planet_seed, clouds);
        let base_coverage = clouds.base_coverage;
        let base_cloud_type = clouds.cloud_type;

        let mut data = vec![0u8; (width * height * 4) as usize];

        for y in 0..height {
            for x in 0..width {
                // Equirectangular: x → longitude [-PI, PI], y → latitude [-PI/2, PI/2].
                let lon = (x as f64 / width as f64) * std::f64::consts::TAU - std::f64::consts::PI;
                let lat = (y as f64 / height as f64) * std::f64::consts::PI - std::f64::consts::FRAC_PI_2;

                // Layer 1: Hadley cell circulation.
                let hadley = hadley_cloud_probability(lat, planet.rotation_period);

                // Layer 2: Pressure cell influence.
                let mut pressure_sum = 0.0_f64;
                for cell in &cells {
                    pressure_sum += cell.influence_at(lat, lon, game_time);
                }
                // Remap pressure influence: negative (cyclone) → more clouds, positive → less.
                // Scale to [0, 1] where 0 = strong anticyclone, 1 = strong cyclone.
                let pressure_factor = (0.5 - pressure_sum * 0.5).clamp(0.0, 1.0);

                // Combine layers: both must agree for clouds to form.
                let raw_coverage = hadley * 0.4 + pressure_factor * 0.6;

                // Apply base_coverage as a global multiplier.
                let coverage = (raw_coverage * base_coverage * 2.0).clamp(0.0, 1.0);

                // Apply a sharp threshold to create distinct cloud/clear boundaries.
                let threshold = 0.35;
                let coverage = smooth_step(threshold, threshold + 0.2, coverage);

                // Cloud type: varies with latitude and coverage intensity.
                // Low latitudes + high coverage → cumulonimbus (tall thunderstorms).
                // High latitudes + moderate coverage → stratus (flat layers).
                // Mid coverage → cumulus (puffy).
                let lat_factor = 1.0 - (lat.abs() / std::f64::consts::FRAC_PI_2);
                let cloud_type = (base_cloud_type * 0.3
                    + coverage * lat_factor * 0.4
                    + pressure_factor * 0.3)
                    .clamp(0.0, 1.0);

                // Precipitation: correlates with coverage intensity and cyclone strength.
                let precipitation = (coverage * (-pressure_sum).max(0.0) * 2.0).clamp(0.0, 1.0);

                // Wind modifier: stronger near cyclone centers.
                let wind_mod = (1.0 + (-pressure_sum).max(0.0) * 0.5).clamp(0.5, 2.0);

                let idx = ((y * width + x) * 4) as usize;
                data[idx] = (coverage as f32 * 255.0) as u8;
                data[idx + 1] = (cloud_type as f32 * 255.0) as u8;
                data[idx + 2] = (precipitation as f32 * 255.0) as u8;
                data[idx + 3] = (wind_mod as f32 / 2.0 * 255.0) as u8;
            }
        }

        Self { width, height, data }
    }
}

fn smooth_step(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weather_map_determinism() {
        // Same inputs must produce identical output.
        let planet = create_test_planet();
        let map1 = WeatherMap::generate(&planet, 100.0);
        let map2 = WeatherMap::generate(&planet, 100.0);
        assert_eq!(map1.data, map2.data, "weather map must be deterministic");
    }

    #[test]
    fn weather_map_different_seeds() {
        let mut planet1 = create_test_planet();
        planet1.planet_seed = 12345;
        let mut planet2 = create_test_planet();
        planet2.planet_seed = 67890;
        let map1 = WeatherMap::generate(&planet1, 100.0);
        let map2 = WeatherMap::generate(&planet2, 100.0);
        assert_ne!(map1.data, map2.data, "different seeds should produce different weather");
    }

    #[test]
    fn weather_map_has_clear_regions() {
        let planet = create_test_planet();
        let map = WeatherMap::generate(&planet, 100.0);
        // Count clear pixels (coverage < 10).
        let clear_count = (0..map.width * map.height)
            .filter(|&i| map.data[(i * 4) as usize] < 10)
            .count();
        let total = (map.width * map.height) as usize;
        let clear_fraction = clear_count as f64 / total as f64;
        assert!(
            clear_fraction > 0.05,
            "at least 5% of planet should be clear sky, got {:.1}%",
            clear_fraction * 100.0
        );
    }

    #[test]
    fn weather_map_has_cloudy_regions() {
        let planet = create_test_planet();
        let map = WeatherMap::generate(&planet, 100.0);
        // Count cloudy pixels (coverage > 200).
        let cloudy_count = (0..map.width * map.height)
            .filter(|&i| map.data[(i * 4) as usize] > 200)
            .count();
        let total = (map.width * map.height) as usize;
        let cloudy_fraction = cloudy_count as f64 / total as f64;
        assert!(
            cloudy_fraction > 0.05,
            "at least 5% of planet should be cloudy, got {:.1}%",
            cloudy_fraction * 100.0
        );
    }

    #[test]
    fn weather_evolves_with_time() {
        let planet = create_test_planet();
        let map_t0 = WeatherMap::generate(&planet, 0.0);
        let map_t1 = WeatherMap::generate(&planet, 3600.0); // 1 hour later
        assert_ne!(map_t0.data, map_t1.data, "weather should change over time");
    }

    fn create_test_planet() -> PlanetParams {
        use crate::system::*;
        PlanetParams {
            index: 0,
            planet_seed: 42,
            orbital_elements: OrbitalElements {
                sma: 1.5e11,
                eccentricity: 0.02,
                inclination: 0.0,
                raan: 0.0,
                arg_periapsis: 0.0,
                mean_anomaly_epoch: 0.0,
            },
            mass_kg: 5.972e24,
            radius_m: 6.371e6,
            color: [0.3, 0.5, 0.8],
            gm: 3.986e14,
            period: 3.156e7,
            rotation_period: 86400.0, // 24 hours
            surface_gravity: 9.81,
            atmosphere: AtmosphereParams {
                has_atmosphere: true,
                atmosphere_height: 100_000.0,
                sea_level_density: 1.225,
                scale_height: 8500.0,
                rayleigh_coeff: [5.5e-6, 13.0e-6, 22.4e-6],
                rayleigh_scale_height: 8000.0,
                mie_coeff: 3.996e-6,
                mie_absorption: 4.4e-7,
                mie_scale_height: 1200.0,
                mie_anisotropy: 0.8,
                ozone_coeff: [0.65e-6, 1.881e-6, 0.085e-6],
                ozone_center_altitude: 25000.0,
                ozone_width: 15000.0,
                weather_mie_multiplier: 1.0,
                weather_sun_occlusion: 1.0,
            },
            clouds: CloudParams {
                has_clouds: true,
                cloud_base_altitude: 1500.0,
                cloud_layer_thickness: 4000.0,
                base_coverage: 0.5,
                density_scale: 1.0,
                cloud_type: 0.5,
                wind_velocity: [10.0, 0.0, 5.0],
                wind_shear: 1.5,
                absorption_factor: 1.0,
                scatter_color: [1.0, 1.0, 1.0],
                weather_scale: 50000.0,
                weather_octaves: 4,
            },
        }
    }
}
