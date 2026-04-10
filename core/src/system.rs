/// Deterministic star system generation and orbital mechanics.
///
/// Uses the brahe crate (MIT) for Kepler equation solving and
/// orbital element → Cartesian state conversion.
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::seed::{derive_seed, seed_to_f64, seed_to_range, seed_to_u32};

/// Gravitational constant in m³/(kg·s²).
pub const G: f64 = 6.674e-11;

/// Normalize an angle to [0, TAU). Rust's `%` on f64 can return negative values.
fn normalize_angle(angle: f64) -> f64 {
    let a = angle % std::f64::consts::TAU;
    if a < 0.0 { a + std::f64::consts::TAU } else { a }
}

/// Solve Kepler's equation: M = E - e*sin(E) for E, given M and e.
/// Newton-Raphson with guaranteed convergence for e < 1.
/// Returns eccentric anomaly in radians.
fn solve_kepler(mean_anomaly: f64, eccentricity: f64) -> f64 {
    let m = normalize_angle(mean_anomaly);
    let e = eccentricity;

    // Initial guess: E = M + e*sin(M) (good for low eccentricity).
    let mut ea = m + e * m.sin();

    // Newton-Raphson: f(E) = E - e*sin(E) - M, f'(E) = 1 - e*cos(E).
    for _ in 0..50 {
        let sin_ea = ea.sin();
        let cos_ea = ea.cos();
        let f = ea - e * sin_ea - m;
        let fp = 1.0 - e * cos_ea;
        if fp.abs() < 1e-15 { break; } // degenerate
        let delta = f / fp;
        ea -= delta;
        if delta.abs() < 1e-12 { break; } // converged
    }
    ea
}

/// Compute true anomaly from eccentric anomaly and eccentricity.
fn eccentric_to_true_anomaly(ecc_anomaly: f64, eccentricity: f64) -> f64 {
    let e = eccentricity;
    let half_e = ecc_anomaly / 2.0;
    // ν = 2 * atan2(sqrt(1+e) * sin(E/2), sqrt(1-e) * cos(E/2))
    let y = (1.0 + e).sqrt() * half_e.sin();
    let x = (1.0 - e).sqrt() * half_e.cos();
    2.0 * y.atan2(x)
}

/// Hill sphere SOI factor: r_soi = a * (m_planet / m_star)^(2/5).
pub const SOI_EXPONENT: f64 = 0.4; // 2/5

/// All gameplay-vs-realistic scale tuning in one place.
/// Switch `SCALE` between `GAMEPLAY` and `REALISTIC` to change mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CelestialScaleConfig {
    /// Celestial time runs this many times faster than real time.
    /// 1.0 = real-time, 12.0 = 1 real second equals 12 celestial seconds.
    pub time_scale: f64,
    /// Base semi-major axis for innermost planet orbit (meters).
    pub base_sma: f64,
    /// Ship spawn offset from nearest planet (meters). Must be < SOI radius.
    pub spawn_offset: f64,
    /// Fallback spawn when no planets exist (meters from star).
    pub fallback_spawn_distance: f64,
    /// Min/max planet sidereal rotation period in celestial seconds.
    pub rotation_period_range: (f64, f64),
}

impl CelestialScaleConfig {
    pub const GAMEPLAY: Self = Self {
        time_scale: 1.0,
        base_sma: 1.0e10,
        spawn_offset: 2.0e7,
        fallback_spawn_distance: 2.0e8,
        rotation_period_range: (3_600.0, 14_400.0), // 1-4 real hours per planet day
    };

    pub const REALISTIC: Self = Self {
        time_scale: 1.0,
        base_sma: 5.0e10,
        spawn_offset: 1.0e8,
        fallback_spawn_distance: 1.0e11,
        rotation_period_range: (36_000.0, 172_800.0), // 10h-48h
    };
}

/// Active scale config. Change this one line to switch modes.
pub const SCALE: CelestialScaleConfig = CelestialScaleConfig::GAMEPLAY;

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

/// Atmosphere parameters for a planet. Includes scattering coefficients for
/// Rayleigh/Mie atmospheric rendering (Hillaire 2020 technique).
/// All optical properties are derived deterministically from the planet seed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmosphereParams {
    /// Whether this planet has a significant atmosphere.
    pub has_atmosphere: bool,
    /// Height above surface where density is effectively zero (meters).
    pub atmosphere_height: f64,
    /// Atmospheric density at sea level (kg/m³). Earth: ~1.225.
    pub sea_level_density: f64,
    /// Scale height — density halves every H*ln(2) meters. Earth: ~8500 m.
    pub scale_height: f64,

    // --- Rayleigh scattering (molecular) ---
    /// Rayleigh scattering coefficients at sea level (1/m), per RGB channel.
    /// Determines sky color: Earth [5.5e-6, 13.0e-6, 22.4e-6] produces blue sky.
    pub rayleigh_coeff: [f64; 3],
    /// Rayleigh density scale height (meters). Earth: ~8000.
    pub rayleigh_scale_height: f64,

    // --- Mie scattering (aerosol/particulate) ---
    /// Mie scattering coefficient at sea level (1/m). Earth: ~3.996e-6.
    pub mie_coeff: f64,
    /// Mie absorption coefficient at sea level (1/m). Earth: ~4.4e-7.
    pub mie_absorption: f64,
    /// Mie density scale height (meters). Earth: ~1200.
    pub mie_scale_height: f64,
    /// Mie asymmetry factor (Henyey-Greenstein g). 0=isotropic, 1=all forward.
    /// Earth: ~0.8. Controls sun halo shape.
    pub mie_anisotropy: f64,

    // --- Ozone absorption (Earth-like planets only) ---
    /// Ozone absorption coefficients (1/m) per RGB. Zero for non-ozone planets.
    pub ozone_coeff: [f64; 3],
    /// Altitude of ozone layer center (meters). Earth: ~25000.
    pub ozone_center_altitude: f64,
    /// Width of ozone layer (meters). Earth: ~15000.
    pub ozone_width: f64,

    // --- Weather hooks (runtime-modulated by future weather system) ---
    /// Mie multiplier for weather effects. 1.0=clear, 5-20=fog/haze.
    pub weather_mie_multiplier: f64,
    /// Sun occlusion from clouds. 1.0=clear sky, 0.0=full overcast.
    pub weather_sun_occlusion: f64,
}

impl AtmosphereParams {
    /// Compute atmospheric density at a given altitude above the surface.
    /// Returns 0.0 if above atmosphere_height or if planet has no atmosphere.
    pub fn density_at_altitude(&self, altitude: f64) -> f64 {
        if !self.has_atmosphere || altitude > self.atmosphere_height || altitude < 0.0 {
            return 0.0;
        }
        self.sea_level_density * (-altitude / self.scale_height).exp()
    }
}

/// Cloud layer parameters for a planet. All values derived deterministically
/// from the planet seed. Controls volumetric cloud rendering (Nubis technique).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudParams {
    /// Whether this planet has volumetric clouds.
    pub has_clouds: bool,
    /// Cloud layer bottom altitude above surface (meters). Earth: ~1500.
    pub cloud_base_altitude: f64,
    /// Cloud layer thickness (meters). Earth: ~3000-5000.
    pub cloud_layer_thickness: f64,
    /// Base cloud coverage (0-1). Modulated spatially by weather noise.
    pub base_coverage: f64,
    /// Cloud density multiplier. Higher = thicker, more opaque.
    pub density_scale: f64,
    /// Cloud type blend: 0=stratus (flat), 0.5=cumulus (puffy), 1=cumulonimbus (towering).
    pub cloud_type: f64,
    /// Wind velocity for cloud scrolling (m/s). Deterministic from seed.
    pub wind_velocity: [f64; 3],
    /// Wind shear: how much upper clouds move faster than lower (multiplier).
    pub wind_shear: f64,
    /// Absorption factor controlling rain cloud darkness. Earth: ~0.5-3.0.
    pub absorption_factor: f64,
    /// Scattering color tint (linear RGB). White for Earth, can be alien colors.
    pub scatter_color: [f32; 3],
    /// Weather spatial variation scale (meters). Controls weather zone size.
    pub weather_scale: f64,
    /// Number of fBm octaves for weather coverage noise (2-6).
    pub weather_octaves: u32,
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
    /// Sidereal rotation period in celestial seconds (for day/night cycle).
    pub rotation_period: f64,
    /// Surface gravity (m/s²). Cached from GM/r².
    pub surface_gravity: f64,
    /// Atmosphere parameters.
    pub atmosphere: AtmosphereParams,
    /// Cloud parameters.
    pub clouds: CloudParams,
}

/// Complete star system parameters, generated deterministically from a seed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemParams {
    pub system_seed: u64,
    pub star: StarParams,
    pub planets: Vec<PlanetParams>,
    pub scale: CelestialScaleConfig,
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
            scale: SCALE,
        }
    }
}

/// Generate a single planet's parameters.
fn generate_planet(index: u32, planet_seed: u64, star: &StarParams) -> PlanetParams {
    // Semi-major axis: planets spread out logarithmically (Titius-Bode-like).
    let base_sma = SCALE.base_sma;
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

    // Sidereal rotation period in celestial seconds (for day/night cycle).
    let rotation_period = seed_to_range(
        derive_seed(planet_seed, 9),
        SCALE.rotation_period_range.0,
        SCALE.rotation_period_range.1,
    );

    let surface_gravity = gm / (radius_m * radius_m);

    // Atmosphere: planets >= 0.3 Earth masses (1.79e24 kg) generally have atmosphere.
    // ~10% of eligible planets are airless (seed-based).
    let earth_mass = 5.972e24;
    let atmo_eligible = mass_kg >= 0.3 * earth_mass;
    let atmo_seed_roll = seed_to_range(derive_seed(planet_seed, 10), 0.0, 1.0);
    let has_atmosphere = atmo_eligible && atmo_seed_roll > 0.1;

    let atmosphere = if has_atmosphere {
        let scale_height = radius_m * seed_to_range(derive_seed(planet_seed, 11), 0.0008, 0.002);
        let sea_level_density = seed_to_range(derive_seed(planet_seed, 12), 0.3, 3.0);
        let atmo_height_factor = seed_to_range(derive_seed(planet_seed, 13), 5.0, 8.0);

        // Rayleigh scattering: base coefficient + channel ratios for sky color variety.
        // Varying the ratio between RGB produces different sky colors per planet.
        let rayleigh_base = seed_to_range(derive_seed(planet_seed, 20), 3.0e-6, 8.0e-6);
        let rayleigh_ratio_g = seed_to_range(derive_seed(planet_seed, 21), 1.8, 3.0);
        let rayleigh_ratio_b = seed_to_range(derive_seed(planet_seed, 22), 3.0, 5.0);
        let rayleigh_coeff = [rayleigh_base, rayleigh_base * rayleigh_ratio_g, rayleigh_base * rayleigh_ratio_b];
        let rayleigh_scale_height = scale_height; // same as thermodynamic scale height

        // Mie scattering: particulate/aerosol. Varies from clear to dusty.
        let mie_coeff = seed_to_range(derive_seed(planet_seed, 23), 1.0e-6, 10.0e-6);
        let mie_absorption = mie_coeff * seed_to_range(derive_seed(planet_seed, 24), 0.05, 0.15);
        let mie_scale_height = seed_to_range(derive_seed(planet_seed, 25), 800.0, 2000.0);
        let mie_anisotropy = seed_to_range(derive_seed(planet_seed, 26), 0.6, 0.95);

        // Ozone: present on ~50% of atmosphere planets (oxygen-rich).
        let has_ozone = seed_to_range(derive_seed(planet_seed, 27), 0.0, 1.0) > 0.5;
        let ozone_coeff = if has_ozone {
            let base = seed_to_range(derive_seed(planet_seed, 28), 0.3e-6, 1.0e-6);
            [base, base * 2.9, base * 0.13]
        } else {
            [0.0; 3]
        };
        let ozone_center_altitude = radius_m * seed_to_range(derive_seed(planet_seed, 29), 0.003, 0.005);
        let ozone_width = ozone_center_altitude * seed_to_range(derive_seed(planet_seed, 30), 0.4, 0.8);

        AtmosphereParams {
            has_atmosphere: true,
            atmosphere_height: scale_height * atmo_height_factor,
            sea_level_density,
            scale_height,
            rayleigh_coeff,
            rayleigh_scale_height,
            mie_coeff,
            mie_absorption,
            mie_scale_height,
            mie_anisotropy,
            ozone_coeff,
            ozone_center_altitude,
            ozone_width,
            weather_mie_multiplier: 1.0,
            weather_sun_occlusion: 1.0,
        }
    } else {
        AtmosphereParams {
            has_atmosphere: false,
            atmosphere_height: 0.0,
            sea_level_density: 0.0,
            scale_height: 1.0, // avoid div-by-zero
            rayleigh_coeff: [0.0; 3],
            rayleigh_scale_height: 1.0,
            mie_coeff: 0.0,
            mie_absorption: 0.0,
            mie_scale_height: 1.0,
            mie_anisotropy: 0.8,
            ozone_coeff: [0.0; 3],
            ozone_center_altitude: 0.0,
            ozone_width: 0.0,
            weather_mie_multiplier: 1.0,
            weather_sun_occlusion: 1.0,
        }
    };

    // Cloud generation: all atmosphere planets have clouds.
    let clouds = if has_atmosphere {
        let scale_height = atmosphere.scale_height;

        // Cloud base altitude: proportional to atmosphere scale height (0.1-0.3×).
        let cloud_base = scale_height * seed_to_range(derive_seed(planet_seed, 41), 0.1, 0.3);
        // Cloud layer thickness: 2-5× base altitude.
        let cloud_thickness = cloud_base * seed_to_range(derive_seed(planet_seed, 42), 2.0, 5.0);
        // Base coverage: moderate to heavy.
        let base_coverage = seed_to_range(derive_seed(planet_seed, 43), 0.3, 0.8);
        // Cloud type: stratus to cumulonimbus.
        let cloud_type = seed_to_range(derive_seed(planet_seed, 44), 0.0, 1.0);
        // Density multiplier.
        let density_scale = seed_to_range(derive_seed(planet_seed, 45), 0.5, 2.0);

        // Wind: speed and direction from seed.
        let wind_speed = seed_to_range(derive_seed(planet_seed, 46), 5.0, 50.0);
        let wind_angle = seed_to_range(derive_seed(planet_seed, 47), 0.0, std::f64::consts::TAU);
        let wind_elevation = seed_to_range(derive_seed(planet_seed, 48), -0.1, 0.1);
        let wind_velocity = [
            wind_speed * wind_angle.cos() * (1.0 - wind_elevation.abs()),
            wind_speed * wind_elevation,
            wind_speed * wind_angle.sin() * (1.0 - wind_elevation.abs()),
        ];
        let wind_shear = seed_to_range(derive_seed(planet_seed, 49), 1.0, 2.5);

        // Absorption and scattering.
        let absorption_factor = seed_to_range(derive_seed(planet_seed, 50), 0.5, 3.0);
        // Scatter color: mostly white, can be tinted by atmosphere composition.
        let tint = seed_to_range(derive_seed(planet_seed, 51), 0.8, 1.0) as f32;
        let scatter_color = [tint, tint, tint]; // near-white, slight warmth variation

        // Weather zone scale: size of weather systems in meters.
        let weather_scale = seed_to_range(derive_seed(planet_seed, 52), 10_000.0, 100_000.0);
        let weather_octaves = 3 + seed_to_u32(derive_seed(planet_seed, 53), 4); // 3-6 octaves

        CloudParams {
            has_clouds: true,
            cloud_base_altitude: cloud_base,
            cloud_layer_thickness: cloud_thickness,
            base_coverage,
            density_scale,
            cloud_type,
            wind_velocity,
            wind_shear,
            absorption_factor,
            scatter_color,
            weather_scale,
            weather_octaves,
        }
    } else {
        CloudParams {
            has_clouds: false,
            cloud_base_altitude: 0.0,
            cloud_layer_thickness: 0.0,
            base_coverage: 0.0,
            density_scale: 0.0,
            cloud_type: 0.0,
            wind_velocity: [0.0; 3],
            wind_shear: 1.0,
            absorption_factor: 1.0,
            scatter_color: [1.0; 3],
            weather_scale: 10_000.0,
            weather_octaves: 3,
        }
    };

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
        rotation_period,
        surface_gravity,
        atmosphere,
        clouds,
    }
}

/// Compute a planet's full Cartesian state (position + velocity) at a given time.
///
/// Position uses brahe's Kepler solver + orbital element → Cartesian conversion.
/// Velocity is computed manually with the correct star GM (brahe hardcodes GM_EARTH
/// which is wrong for our procedural stars).
///
/// `star_gm` is the gravitational parameter of the central star (G * star_mass).
pub fn compute_planet_state(planet: &PlanetParams, star_gm: f64, time_s: f64) -> (DVec3, DVec3) {
    let oe = &planet.orbital_elements;

    // Mean anomaly at time t.
    let mean_motion = std::f64::consts::TAU / planet.period;
    let mean_anomaly = normalize_angle(oe.mean_anomaly_epoch + mean_motion * time_s);

    use brahe::constants::units::AngleFormat;

    // --- Position: use brahe's state_koe_to_eci with MEAN_ANOMALY (not true anomaly). ---
    // Brahe internally solves M → E → ν → position. Passing mean_anomaly is the correct API usage.
    let koe = nalgebra::SVector::<f64, 6>::new(
        oe.sma, oe.eccentricity, oe.inclination, oe.raan, oe.arg_periapsis,
        mean_anomaly, // CORRECT: 6th element is mean anomaly for state_koe_to_eci
    );
    let state = brahe::coordinates::cartesian::state_koe_to_eci(koe, AngleFormat::Radians);
    let position = DVec3::new(state[0], state[1], state[2]);

    // --- Velocity: compute with correct star GM (brahe hardcodes GM_EARTH). ---
    // Solve Kepler ourselves for E → ν, then compute velocity in perifocal frame.
    let ecc_anomaly = solve_kepler(mean_anomaly, oe.eccentricity);
    let true_anomaly = eccentric_to_true_anomaly(ecc_anomaly, oe.eccentricity);

    // --- Perifocal (PQW) frame vectors ---
    // P = periapsis direction, Q = 90° ahead in orbit plane.
    let e = oe.eccentricity;
    let i = oe.inclination;
    let raan = oe.raan;
    let omega = oe.arg_periapsis;

    let p_vec = DVec3::new(
        omega.cos() * raan.cos() - omega.sin() * i.cos() * raan.sin(),
        omega.cos() * raan.sin() + omega.sin() * i.cos() * raan.cos(),
        omega.sin() * i.sin(),
    );
    let q_vec = DVec3::new(
        -omega.sin() * raan.cos() - omega.cos() * i.cos() * raan.sin(),
        -omega.sin() * raan.sin() + omega.cos() * i.cos() * raan.cos(),
        omega.cos() * i.sin(),
    );

    // --- Velocity with correct star GM ---
    // v_perifocal = sqrt(GM/p) * [-sin(ν) * P + (e + cos(ν)) * Q]
    // where p = a(1-e²) is the semi-latus rectum, ν = true anomaly.
    let p_slr = oe.sma * (1.0 - e * e); // semi-latus rectum
    let sqrt_gm_over_p = (star_gm / p_slr).sqrt();
    let v_p = -true_anomaly.sin() * sqrt_gm_over_p;
    let v_q = (e + true_anomaly.cos()) * sqrt_gm_over_p;
    let velocity = p_vec * v_p + q_vec * v_q;

    (position, velocity)
}

/// Compute a planet's 3D position at a given time.
/// Uses brahe's state_koe_to_eci with mean_anomaly (correct 6th element).
pub fn compute_planet_position(planet: &PlanetParams, time_s: f64) -> DVec3 {
    let oe = &planet.orbital_elements;
    let mean_motion = std::f64::consts::TAU / planet.period;
    let mean_anomaly = normalize_angle(oe.mean_anomaly_epoch + mean_motion * time_s);

    use brahe::constants::units::AngleFormat;
    let koe = nalgebra::SVector::<f64, 6>::new(
        oe.sma, oe.eccentricity, oe.inclination, oe.raan, oe.arg_periapsis,
        mean_anomaly, // CORRECT: brahe expects mean anomaly, solves Kepler internally
    );
    let state = brahe::coordinates::cartesian::state_koe_to_eci(koe, AngleFormat::Radians);
    DVec3::new(state[0], state[1], state[2])
}

/// Compute a planet's orbital velocity at a given time (m/s per celestial second).
/// Uses the correct star GM (not brahe's hardcoded GM_EARTH).
///
/// NOTE: This returns dPosition/dCelestialTime. For real-space velocity (m/s per
/// real second), multiply by `time_scale` or use `compute_planet_velocity_realtime`.
pub fn compute_planet_velocity(planet: &PlanetParams, star_gm: f64, time_s: f64) -> DVec3 {
    compute_planet_state(planet, star_gm, time_s).1
}

/// Compute a planet's velocity in real-time system-space coordinates (m/s per real second).
///
/// Accounts for celestial time scale: `real_vel = celestial_vel × time_scale`.
/// Use this for reference frame conversions (patched conics SOI transitions).
pub fn compute_planet_velocity_realtime(
    planet: &PlanetParams,
    star_gm: f64,
    celestial_time: f64,
    time_scale: f64,
) -> DVec3 {
    compute_planet_velocity(planet, star_gm, celestial_time) * time_scale
}

/// Compute a planet's rotation angle at a given celestial time.
/// Returns angle in radians [0, TAU).
pub fn compute_planet_rotation(planet: &PlanetParams, celestial_time_s: f64) -> f64 {
    normalize_angle(std::f64::consts::TAU * celestial_time_s / planet.rotation_period)
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

/// Check if a position is within any planet's atmosphere.
/// Returns `Some((planet_index, altitude_above_surface))` if inside, `None` otherwise.
pub fn check_atmosphere(
    position: DVec3,
    planets: &[PlanetParams],
    planet_positions: &[DVec3],
) -> Option<(usize, f64)> {
    for (i, (planet, &planet_pos)) in planets.iter().zip(planet_positions.iter()).enumerate() {
        if !planet.atmosphere.has_atmosphere {
            continue;
        }
        let dist = (position - planet_pos).length();
        let altitude = dist - planet.radius_m;
        if altitude < planet.atmosphere.atmosphere_height && altitude >= 0.0 {
            return Some((i, altitude));
        }
    }
    None
}

/// Compute atmospheric drag acceleration on a ship.
///
/// Uses the standard drag formula: `a = -0.5 * Cd * (A/m) * ρ * |v|² * v_hat`
/// All ship properties come from the entity, not hardcoded constants.
///
/// `dt` is used for stability clamping (prevents velocity reversal in one tick).
pub fn compute_atmospheric_drag(
    ship_pos: DVec3,
    ship_vel: DVec3,
    planet_pos: DVec3,
    planet: &PlanetParams,
    ship_mass: f64,
    ship_cross_section: f64,
    ship_drag_coeff: f64,
    dt: f64,
) -> DVec3 {
    if !planet.atmosphere.has_atmosphere {
        return DVec3::ZERO;
    }
    let dist = (ship_pos - planet_pos).length();
    let altitude = dist - planet.radius_m;
    let density = planet.atmosphere.density_at_altitude(altitude);
    if density <= 0.0 {
        return DVec3::ZERO;
    }

    let speed = ship_vel.length();
    if speed < 0.01 {
        return DVec3::ZERO;
    }

    // F_drag = 0.5 * rho * v^2 * Cd * A
    let drag_force = 0.5 * density * speed * speed * ship_drag_coeff * ship_cross_section;
    let drag_accel = drag_force / ship_mass;

    // Stability clamp: don't decelerate more than 50% of speed in one tick.
    let max_decel = 0.5 * speed / dt;
    let clamped = drag_accel.min(max_decel);

    // Drag opposes velocity.
    let v_hat = ship_vel / speed;
    -v_hat * clamped
}

// ---------------------------------------------------------------------------
// Orientation-dependent aerodynamics
// ---------------------------------------------------------------------------

use crate::autopilot::ShipPhysicalProperties;

/// Compute orientation-dependent effective cross-section and drag coefficient.
///
/// Projects the velocity vector into the ship's local frame, then weights
/// per-axis cross-sections and Cd values by direction cosines.
/// Returns `(effective_area_m2, effective_cd)`.
pub fn compute_effective_drag_area(
    ship_rotation: DQuat,
    velocity: DVec3,
    props: &ShipPhysicalProperties,
) -> (f64, f64) {
    let speed = velocity.length();
    if speed < 0.01 {
        return (props.cross_section_front, props.cd_front);
    }

    // Velocity direction in ship-local frame
    let v_local = ship_rotation.inverse() * (velocity / speed);

    // Direction cosines (absolute values)
    let ax = v_local.x.abs(); // side contribution
    let ay = v_local.y.abs(); // top/bottom contribution
    let az = v_local.z.abs(); // front/back contribution

    let area = ax * props.cross_section_side
             + ay * props.cross_section_top
             + az * props.cross_section_front;

    let cd = ax * props.cd_side
           + ay * props.cd_top
           + az * props.cd_front;

    (area, cd)
}

/// Compute aerodynamic lift force as acceleration (m/s²).
///
/// Uses flat-plate Newtonian model: `Cl = Cl_max * sin(2*alpha)`.
/// Boxy ships have low Cl_max (~0.6 vs 1.5+ for airfoils).
/// Lift is perpendicular to velocity in the plane of velocity and ship's up vector.
pub fn compute_aerodynamic_lift(
    ship_rotation: DQuat,
    velocity: DVec3,
    density: f64,
    props: &ShipPhysicalProperties,
) -> DVec3 {
    let speed = velocity.length();
    if speed < 1.0 || density < 1e-6 {
        return DVec3::ZERO;
    }

    let v_hat = velocity / speed;

    // Velocity in ship-local frame for angle-of-attack computation
    let v_local = ship_rotation.inverse() * v_hat;
    // AoA in pitch plane (Y-Z plane of ship).
    // Positive AoA = nose above velocity (airflow from below = lift upward).
    // Negate v_local.y because downward velocity component means positive AoA.
    let alpha = (-v_local.y).atan2(-v_local.z);

    const CL_MAX: f64 = 0.6; // boxy ship
    let cl = CL_MAX * (2.0 * alpha).sin();

    // Lift direction: component of ship_up perpendicular to velocity
    let ship_up = ship_rotation * DVec3::Y;
    let lift_dir_raw = ship_up - v_hat * ship_up.dot(v_hat);
    let lift_dir_len = lift_dir_raw.length();
    if lift_dir_len < 1e-8 {
        return DVec3::ZERO;
    }
    let lift_dir = lift_dir_raw / lift_dir_len;

    // L = 0.5 * rho * v^2 * Cl * A_ref
    let lift_force = 0.5 * density * speed * speed * cl * props.cross_section_top;
    let lift_accel = lift_force / props.mass_kg;

    lift_dir * lift_accel
}

/// Compute aerodynamic weathercock torque (world-space, in N*m).
///
/// The center of pressure (CoP) behind the center of mass creates a restoring
/// torque that aligns the ship nose-first into the airflow. Includes angular
/// velocity damping to prevent oscillation.
///
/// Returns world-space torque vector.
pub fn compute_aerodynamic_torque(
    ship_rotation: DQuat,
    velocity: DVec3,
    angular_velocity: DVec3,
    density: f64,
    props: &ShipPhysicalProperties,
) -> DVec3 {
    let speed = velocity.length();
    if speed < 5.0 || density < 1e-6 {
        return DVec3::ZERO;
    }

    // Velocity in ship-local frame
    let v_hat_local = ship_rotation.inverse() * (velocity / speed);

    // Lateral components (deviation from nose-first)
    let lateral_x = v_hat_local.x; // sideslip
    let lateral_y = v_hat_local.y; // angle of attack

    // Dynamic pressure
    let q = 0.5 * density * speed * speed;

    // Restoring torque: lateral aero force at CoP creates moment about CoM.
    // Torque about Y (yaw) from sideslip:
    let yaw_torque = -q * props.cd_side * props.cross_section_side
                     * lateral_x * props.cop_offset_z;
    // Torque about X (pitch) from AoA:
    let pitch_torque = q * props.cd_top * props.cross_section_top
                       * lateral_y * props.cop_offset_z;

    let torque_local = DVec3::new(pitch_torque, yaw_torque, 0.0);

    // Angular damping: resists rotation in atmosphere (prevents oscillation).
    let (ix, iy, _iz) = props.moment_of_inertia();
    let ang_vel_local = ship_rotation.inverse() * angular_velocity;
    let damping_factor = density * speed * props.cop_offset_z.abs()
                        * props.cross_section_side.max(props.cross_section_top);
    let damping_local = DVec3::new(
        -ang_vel_local.x * damping_factor * 0.5,
        -ang_vel_local.y * damping_factor * 0.5,
        -ang_vel_local.z * damping_factor * 0.1, // less roll damping
    );

    // Clamp aero angular acceleration to 2× the ship's RCS capability.
    // This ensures the weathercock is meaningful (real aerodynamic effect) but
    // the ship's RCS can always partially resist it. Scales with ship design:
    // strong RCS → higher clamp, weak RCS → lower clamp. No hardcoded values.
    let max_ang_accel = props.angular_acceleration() * 2.0;
    let total_local = torque_local + damping_local;
    let clamped = DVec3::new(
        total_local.x.clamp(-max_ang_accel * ix, max_ang_accel * ix),
        total_local.y.clamp(-max_ang_accel * iy, max_ang_accel * iy),
        total_local.z, // roll is small, no clamp needed
    );

    ship_rotation * clamped
}

/// Sutton-Graves constant for N₂/O₂ atmospheres (SI units).
const SUTTON_GRAVES_K: f64 = 1.7415e-4;

/// Compute re-entry heating and thermal damage.
///
/// Uses Sutton-Graves stagnation point heat flux: `q = k * sqrt(rho/r_nose) * v³`.
/// Returns `(heat_flux_w_m2, thermal_damage_this_tick)`.
pub fn compute_reentry_heating(
    speed: f64,
    density: f64,
    props: &ShipPhysicalProperties,
    thermal_energy: &mut f64,
    dt: f64,
) -> (f64, f64) {
    if density < 1e-8 || speed < 100.0 {
        // Radiative cooling only (slow exponential decay)
        *thermal_energy = (*thermal_energy - *thermal_energy * 0.02 * dt).max(0.0);
        return (0.0, 0.0);
    }

    // Stagnation point heat flux
    let q_stag = SUTTON_GRAVES_K * (density / props.nose_radius_m).sqrt() * speed.powi(3);

    // Total heating power on front face (40% average factor vs stagnation peak)
    let heating_power = q_stag * props.cross_section_front * 0.4;

    // Radiative cooling: proportional to stored energy^(4/3)
    let energy_fraction = *thermal_energy / props.thermal_capacity_j.max(1.0);
    let cooling_power = props.thermal_emissivity * 50.0 * energy_fraction.powf(1.33)
                       * props.cross_section_front * 1000.0;

    // Net energy change
    let net_power = heating_power - cooling_power;
    *thermal_energy = (*thermal_energy + net_power * dt).max(0.0);

    // Damage: excess energy above capacity
    let damage = if *thermal_energy > props.thermal_capacity_j {
        let excess = *thermal_energy - props.thermal_capacity_j;
        *thermal_energy = props.thermal_capacity_j;
        excess / 1e7 // 1 HP per 10 MJ excess
    } else {
        0.0
    };

    (q_stag, damage)
}

/// Complete aerodynamic computation result for one ship in one tick.
#[derive(Debug, Clone)]
pub struct AerodynamicsResult {
    /// Drag acceleration (m/s², opposes velocity).
    pub drag_accel: DVec3,
    /// Lift acceleration (m/s², perpendicular to velocity).
    pub lift_accel: DVec3,
    /// Aerodynamic torque (N*m, world-space).
    pub aero_torque: DVec3,
    /// Stagnation point heat flux (W/m²).
    pub heat_flux: f64,
    /// Thermal damage this tick (HP).
    pub thermal_damage: f64,
    /// Atmospheric density at ship position (kg/m³).
    pub density: f64,
    /// Altitude above planet surface (meters).
    pub altitude: f64,
}

impl AerodynamicsResult {
    pub const ZERO: Self = Self {
        drag_accel: DVec3::ZERO,
        lift_accel: DVec3::ZERO,
        aero_torque: DVec3::ZERO,
        heat_flux: 0.0,
        thermal_damage: 0.0,
        density: 0.0,
        altitude: f64::INFINITY,
    };
}

/// Compute all aerodynamic effects for one ship in one tick.
///
/// Single density lookup, then drag (orientation-dependent), lift, torque, and heating.
/// Works identically for manual flight and autopilot.
pub fn compute_full_aerodynamics(
    ship_pos: DVec3,
    ship_vel: DVec3,
    ship_rotation: DQuat,
    angular_velocity: DVec3,
    planet_pos: DVec3,
    planet: &PlanetParams,
    props: &ShipPhysicalProperties,
    thermal_energy: &mut f64,
    dt: f64,
) -> AerodynamicsResult {
    if !planet.atmosphere.has_atmosphere {
        return AerodynamicsResult::ZERO;
    }

    let dist = (ship_pos - planet_pos).length();
    let altitude = dist - planet.radius_m;
    let density = planet.atmosphere.density_at_altitude(altitude);

    if density < 1e-10 {
        // No atmosphere — only radiative cooling
        *thermal_energy = (*thermal_energy * (-0.02 * dt).exp()).max(0.0);
        return AerodynamicsResult { altitude, ..AerodynamicsResult::ZERO };
    }

    let speed = ship_vel.length();
    if speed < 0.01 {
        return AerodynamicsResult { density, altitude, ..AerodynamicsResult::ZERO };
    }

    // 1. Orientation-dependent drag
    let (eff_area, eff_cd) = compute_effective_drag_area(ship_rotation, ship_vel, props);
    let drag_force = 0.5 * density * speed * speed * eff_cd * eff_area;
    let drag_accel_mag = (drag_force / props.mass_kg).min(0.5 * speed / dt); // stability clamp
    let drag_accel = -(ship_vel / speed) * drag_accel_mag;

    // 2. Lift (stability clamp matching drag — at hypersonic speeds, unclamped lift
    //    ∝ density×v² can reach tens of thousands of m/s², deflecting the trajectory
    //    out of the atmosphere in a single tick)
    let lift_accel_raw = compute_aerodynamic_lift(ship_rotation, ship_vel, density, props);
    let lift_mag = lift_accel_raw.length();
    let max_lift_accel = 0.5 * speed / dt;
    let lift_accel = if lift_mag > max_lift_accel && lift_mag > 1e-10 {
        lift_accel_raw * (max_lift_accel / lift_mag)
    } else {
        lift_accel_raw
    };

    // 3. Torque (weathercock + damping)
    let aero_torque = compute_aerodynamic_torque(
        ship_rotation, ship_vel, angular_velocity, density, props,
    );

    // 4. Re-entry heating
    let (heat_flux, thermal_damage) = compute_reentry_heating(
        speed, density, props, thermal_energy, dt,
    );

    AerodynamicsResult {
        drag_accel, lift_accel, aero_torque,
        heat_flux, thermal_damage, density, altitude,
    }
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
    fn rotation_period_in_range() {
        for seed in 0..20 {
            let sys = SystemParams::from_seed(seed);
            for planet in &sys.planets {
                assert!(
                    planet.rotation_period >= SCALE.rotation_period_range.0
                        && planet.rotation_period <= SCALE.rotation_period_range.1,
                    "seed {seed}, planet {}: rotation_period {} out of range",
                    planet.index,
                    planet.rotation_period
                );
            }
        }
    }

    #[test]
    fn planet_rotation_deterministic() {
        let sys = SystemParams::from_seed(42);
        let planet = &sys.planets[0];
        let angle_a = compute_planet_rotation(planet, 1000.0);
        let angle_b = compute_planet_rotation(planet, 1000.0);
        assert_eq!(angle_a, angle_b);
        // Different time = different angle
        let angle_c = compute_planet_rotation(planet, 2000.0);
        assert!((angle_a - angle_c).abs() > 1e-10);
    }

    #[test]
    fn scale_config_switchable() {
        // Verify both configs have sane values
        assert_eq!(CelestialScaleConfig::REALISTIC.time_scale, 1.0);
        assert_eq!(CelestialScaleConfig::GAMEPLAY.time_scale, 1.0);
        assert!(CelestialScaleConfig::GAMEPLAY.base_sma < CelestialScaleConfig::REALISTIC.base_sma);
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

    // -- Aerodynamics tests --

    #[test]
    fn effective_drag_area_nose_first() {
        use crate::autopilot::ShipPhysicalProperties;
        let props = ShipPhysicalProperties::starter_ship();
        let rotation = DQuat::IDENTITY;
        // Velocity along -Z (ship's forward axis)
        let vel = DVec3::new(0.0, 0.0, -100.0);
        let (area, cd) = compute_effective_drag_area(rotation, vel, &props);
        assert!((area - props.cross_section_front).abs() < 0.1,
            "nose-first area should be front cross-section, got {area}");
        assert!((cd - props.cd_front).abs() < 0.01,
            "nose-first Cd should be cd_front, got {cd}");
    }

    #[test]
    fn effective_drag_area_broadside() {
        use crate::autopilot::ShipPhysicalProperties;
        let props = ShipPhysicalProperties::starter_ship();
        let rotation = DQuat::IDENTITY;
        // Velocity along +X (broadside)
        let vel = DVec3::new(100.0, 0.0, 0.0);
        let (area, cd) = compute_effective_drag_area(rotation, vel, &props);
        assert!((area - props.cross_section_side).abs() < 0.1,
            "broadside area should be side cross-section, got {area}");
        assert!((cd - props.cd_side).abs() < 0.01,
            "broadside Cd should be cd_side, got {cd}");
    }

    #[test]
    fn drag_nose_vs_broadside_ratio() {
        use crate::autopilot::ShipPhysicalProperties;
        let props = ShipPhysicalProperties::starter_ship();
        let rotation = DQuat::IDENTITY;
        let (area_nose, cd_nose) = compute_effective_drag_area(
            rotation, DVec3::new(0.0, 0.0, -100.0), &props);
        let (area_side, cd_side) = compute_effective_drag_area(
            rotation, DVec3::new(100.0, 0.0, 0.0), &props);
        let drag_ratio = (area_nose * cd_nose) / (area_side * cd_side);
        assert!(drag_ratio < 0.35, "nose drag should be ~30% of broadside, got {:.1}%", drag_ratio * 100.0);
        assert!(drag_ratio > 0.25, "nose drag ratio too low: {:.1}%", drag_ratio * 100.0);
    }

    #[test]
    fn lift_zero_at_zero_aoa() {
        use crate::autopilot::ShipPhysicalProperties;
        let props = ShipPhysicalProperties::starter_ship();
        let rotation = DQuat::IDENTITY;
        // Velocity exactly along ship's forward (-Z): zero angle of attack
        let vel = DVec3::new(0.0, 0.0, -300.0);
        let lift = compute_aerodynamic_lift(rotation, vel, 1.225, &props);
        assert!(lift.length() < 0.1, "lift should be ~0 at zero AoA, got {}", lift.length());
    }

    #[test]
    fn lift_nonzero_at_angle_of_attack() {
        use crate::autopilot::ShipPhysicalProperties;
        let props = ShipPhysicalProperties::starter_ship();
        let rotation = DQuat::IDENTITY;
        // Velocity at ~20 degrees below ship's forward
        let speed = 300.0;
        let alpha = 20.0_f64.to_radians();
        let vel = DVec3::new(0.0, -speed * alpha.sin(), -speed * alpha.cos());
        let lift = compute_aerodynamic_lift(rotation, vel, 1.225, &props);
        assert!(lift.length() > 10.0, "lift should be significant at 20 deg AoA, got {}", lift.length());
        // Lift should be mostly upward (positive Y)
        assert!(lift.y > 0.0, "lift should be upward");
    }

    #[test]
    fn aero_torque_restoring() {
        use crate::autopilot::ShipPhysicalProperties;
        let props = ShipPhysicalProperties::starter_ship();
        let rotation = DQuat::IDENTITY;
        // Velocity at angle to ship: flow coming from +X (sideslip)
        let vel = DVec3::new(200.0, 0.0, -200.0); // 45 degree sideslip
        let ang_vel = DVec3::ZERO;
        let torque = compute_aerodynamic_torque(rotation, vel, ang_vel, 1.225, &props);
        // Torque should try to yaw the ship to face the velocity (restoring)
        assert!(torque.length() > 100.0, "torque should be significant for 45 deg sideslip");
    }

    #[test]
    fn full_aerodynamics_no_atmosphere() {
        use crate::autopilot::ShipPhysicalProperties;
        let sys = SystemParams::from_seed(42);
        let props = ShipPhysicalProperties::starter_ship();
        // Create a planet with no atmosphere
        let mut planet = sys.planets[0].clone();
        planet.atmosphere.has_atmosphere = false;
        let mut thermal = 0.0;
        let result = compute_full_aerodynamics(
            DVec3::new(planet.radius_m + 1000.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, -1000.0),
            DQuat::IDENTITY, DVec3::ZERO,
            DVec3::ZERO, &planet, &props, &mut thermal, 0.05,
        );
        assert!(result.drag_accel.length() < 1e-10, "no drag without atmosphere");
        assert!(result.lift_accel.length() < 1e-10, "no lift without atmosphere");
    }

    #[test]
    fn reentry_heating_accumulates() {
        use crate::autopilot::ShipPhysicalProperties;
        let props = ShipPhysicalProperties::starter_ship();
        let mut thermal_energy = 0.0;
        let density = 0.001; // ~50 km altitude in Earth atmosphere
        let speed = 7000.0; // orbital velocity

        let (flux, _damage) = compute_reentry_heating(speed, density, &props, &mut thermal_energy, 0.05);

        assert!(flux > 1e5, "heat flux should be substantial at orbital speed, got {flux:.0}");
        assert!(thermal_energy > 0.0, "thermal energy should accumulate");
        assert!(thermal_energy < props.thermal_capacity_j, "should not exceed capacity in one tick");
    }
}
