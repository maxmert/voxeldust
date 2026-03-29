/// Procedural galaxy generation: deterministic star positioning from a galaxy seed.
///
/// Stars are placed in a 4-arm spiral galaxy pattern using cylindrical coordinates.
/// The same galaxy_seed always produces the same star field.
use glam::DVec3;

use crate::seed::{derive_seed, derive_system_seed, seed_to_f64, seed_to_range, seed_to_u32};

/// 1 galaxy unit = 1,000,000 blocks.
pub const GALAXY_UNIT_IN_BLOCKS: f64 = 1_000_000.0;

/// Galaxy radius in galaxy units.
pub const GALAXY_RADIUS: f64 = 50_000.0;

/// Vertical thickness of the galactic disk.
pub const DISK_THICKNESS: f64 = 2_000.0;

/// Number of spiral arms.
pub const NUM_ARMS: u32 = 4;

/// How tightly the spiral arms wind (radians per GU of radius).
pub const SPIRAL_TIGHTNESS: f64 = 0.0004;

/// Minimum stars per galaxy.
pub const MIN_STARS: u32 = 200;

/// Maximum stars per galaxy.
pub const MAX_STARS: u32 = 2000;

/// Base SOI radius in galaxy units.
pub const BASE_SOI_RADIUS: f64 = 100.0;

/// SOI scaling factor per unit of luminosity.
pub const SOI_LUMINOSITY_SCALE: f64 = 200.0;

/// Spectral classification of a star.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarClass {
    O, // Blue giant (very rare, very luminous)
    B, // Blue-white (rare)
    A, // White (uncommon)
    F, // Yellow-white
    G, // Yellow (Sun-like)
    K, // Orange
    M, // Red dwarf (most common)
}

impl StarClass {
    /// Derive star class from a seed, weighted by realistic distribution.
    /// M stars are most common (~76%), O stars are rarest (~0.003%).
    fn from_seed(seed: u64) -> Self {
        let v = seed_to_u32(seed, 1000);
        match v {
            0..=0 => StarClass::O,       // 0.1%
            1..=9 => StarClass::B,       // 0.9%
            10..=39 => StarClass::A,     // 3%
            40..=89 => StarClass::F,     // 5%
            90..=159 => StarClass::G,    // 7%
            160..=289 => StarClass::K,   // 13%
            _ => StarClass::M,           // 71%
        }
    }

    /// Base luminosity relative to the Sun (L☉).
    pub fn luminosity(&self) -> f64 {
        match self {
            StarClass::O => 50.0,
            StarClass::B => 20.0,
            StarClass::A => 8.0,
            StarClass::F => 3.0,
            StarClass::G => 1.0,
            StarClass::K => 0.4,
            StarClass::M => 0.08,
        }
    }

    pub fn as_u8(&self) -> u8 {
        match self {
            StarClass::O => 0,
            StarClass::B => 1,
            StarClass::A => 2,
            StarClass::F => 3,
            StarClass::G => 4,
            StarClass::K => 5,
            StarClass::M => 6,
        }
    }
}

/// Information about a single star in the galaxy.
#[derive(Debug, Clone)]
pub struct StarInfo {
    /// Index in the galaxy (0..star_count).
    pub index: u32,
    /// Position in galaxy-unit coordinates.
    pub position: DVec3,
    /// Derived system seed for this star's planetary system.
    pub system_seed: u64,
    /// Spectral class.
    pub star_class: StarClass,
    /// Luminosity relative to Sun.
    pub luminosity: f64,
}

/// The complete star map for a galaxy, generated deterministically from a seed.
pub struct GalaxyMap {
    pub galaxy_seed: u64,
    pub stars: Vec<StarInfo>,
}

impl GalaxyMap {
    /// Generate the complete star map from a galaxy seed.
    pub fn generate(galaxy_seed: u64) -> Self {
        let count = star_count(galaxy_seed);
        let mut stars = Vec::with_capacity(count as usize);

        for i in 0..count {
            stars.push(generate_star(galaxy_seed, i));
        }

        Self {
            galaxy_seed,
            stars,
        }
    }

    /// Find the nearest star to a position.
    pub fn nearest_star(&self, pos: DVec3) -> Option<&StarInfo> {
        self.stars
            .iter()
            .min_by(|a, b| {
                let da = a.position.distance_squared(pos);
                let db = b.position.distance_squared(pos);
                da.partial_cmp(&db).unwrap()
            })
    }

    /// Find all stars within a radius of a position.
    pub fn stars_in_range(&self, pos: DVec3, radius: f64) -> Vec<&StarInfo> {
        let r2 = radius * radius;
        self.stars
            .iter()
            .filter(|s| s.position.distance_squared(pos) < r2)
            .collect()
    }

    /// Check if a position is within any star's sphere of influence.
    /// Returns the star if inside an SOI.
    pub fn check_soi(&self, pos: DVec3) -> Option<&StarInfo> {
        self.stars.iter().find(|star| {
            let soi = system_soi_radius(star);
            pos.distance_squared(star.position) < soi * soi
        })
    }

    /// Get a star by index.
    pub fn get_star(&self, index: u32) -> Option<&StarInfo> {
        self.stars.iter().find(|s| s.index == index)
    }
}

/// Derive star count from galaxy seed.
pub fn star_count(galaxy_seed: u64) -> u32 {
    let h = derive_seed(galaxy_seed, 0xFFFF_FFFF);
    let range = MAX_STARS - MIN_STARS;
    MIN_STARS + (h % range as u64) as u32
}

/// Compute the sphere-of-influence radius for a star system.
pub fn system_soi_radius(star: &StarInfo) -> f64 {
    BASE_SOI_RADIUS + star.luminosity * SOI_LUMINOSITY_SCALE
}

/// Convert a system-local position (in blocks) to galaxy coordinates (in GU).
pub fn system_to_galaxy(star_position: DVec3, system_pos: DVec3) -> DVec3 {
    star_position + system_pos / GALAXY_UNIT_IN_BLOCKS
}

/// Convert a galaxy position (in GU) to system-local coordinates (in blocks).
pub fn galaxy_to_system(star_position: DVec3, galaxy_pos: DVec3) -> DVec3 {
    (galaxy_pos - star_position) * GALAXY_UNIT_IN_BLOCKS
}

/// Generate a single star's info from the galaxy seed and star index.
fn generate_star(galaxy_seed: u64, star_index: u32) -> StarInfo {
    let system_seed = derive_system_seed(galaxy_seed, star_index);

    // Derive sub-seeds for each property.
    let pos_seed = derive_seed(system_seed, 0);
    let class_seed = derive_seed(system_seed, 1);
    let arm_seed = derive_seed(system_seed, 2);
    let height_seed = derive_seed(system_seed, 3);
    let jitter_seed = derive_seed(system_seed, 4);

    let star_class = StarClass::from_seed(class_seed);
    let luminosity = star_class.luminosity() * seed_to_range(derive_seed(system_seed, 5), 0.5, 1.5);

    // Spiral arm positioning.
    let arm_index = seed_to_u32(arm_seed, NUM_ARMS);
    let arm_base_angle = std::f64::consts::TAU * arm_index as f64 / NUM_ARMS as f64;

    // Radial distance: exponential distribution concentrated toward center.
    // Use inverse CDF: r = -ln(1-U) * scale, clamped to galaxy radius.
    let u = seed_to_f64(pos_seed).max(0.001); // avoid log(0)
    let radial = (-u.ln() * GALAXY_RADIUS * 0.2).min(GALAXY_RADIUS);

    // Angular position: arm base + spiral winding + random jitter.
    let spiral_angle = arm_base_angle + radial * SPIRAL_TIGHTNESS;
    let jitter = seed_to_range(jitter_seed, -0.8, 0.8);
    let theta = spiral_angle + jitter;

    // Vertical displacement: thin disk with Gaussian-like distribution.
    // Approximate Gaussian via Box-Muller with two uniform seeds.
    let u1 = seed_to_f64(height_seed).max(0.001);
    let u2 = seed_to_f64(derive_seed(height_seed, 1));
    let gaussian = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    let height = gaussian * DISK_THICKNESS * 0.1; // Most stars near the plane.

    let position = DVec3::new(radial * theta.cos(), height, radial * theta.sin());

    StarInfo {
        index: star_index,
        position,
        system_seed,
        star_class,
        luminosity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_galaxy() {
        let a = GalaxyMap::generate(42);
        let b = GalaxyMap::generate(42);
        assert_eq!(a.stars.len(), b.stars.len());
        for (sa, sb) in a.stars.iter().zip(b.stars.iter()) {
            assert_eq!(sa.position, sb.position);
            assert_eq!(sa.system_seed, sb.system_seed);
            assert_eq!(sa.star_class, sb.star_class);
        }
    }

    #[test]
    fn star_count_in_range() {
        for seed in 0..20 {
            let count = star_count(seed);
            assert!(
                count >= MIN_STARS && count < MAX_STARS,
                "star count {count} out of range for seed {seed}"
            );
        }
    }

    #[test]
    fn positions_within_bounds() {
        let map = GalaxyMap::generate(42);
        for star in &map.stars {
            let r = (star.position.x * star.position.x + star.position.z * star.position.z).sqrt();
            assert!(
                r <= GALAXY_RADIUS * 1.1,
                "star {} at radius {r} exceeds galaxy radius",
                star.index
            );
            assert!(
                star.position.y.abs() < DISK_THICKNESS,
                "star {} height {} exceeds disk thickness",
                star.index,
                star.position.y
            );
        }
    }

    #[test]
    fn soi_detection() {
        let map = GalaxyMap::generate(42);
        let star = &map.stars[0];

        // Position at the star should be inside SOI.
        assert!(map.check_soi(star.position).is_some());

        // Position far away should not.
        let far = DVec3::new(1e9, 1e9, 1e9);
        assert!(map.check_soi(far).is_none());

        // Position just inside SOI edge.
        let soi = system_soi_radius(star);
        let edge = star.position + DVec3::new(soi * 0.99, 0.0, 0.0);
        assert!(map.check_soi(edge).is_some());
    }

    #[test]
    fn coordinate_transforms_invertible() {
        let star_pos = DVec3::new(1000.0, 50.0, -2000.0);
        let system_pos = DVec3::new(500_000.0, 100_000.0, -300_000.0); // in blocks

        let galaxy_pos = system_to_galaxy(star_pos, system_pos);
        let back = galaxy_to_system(star_pos, galaxy_pos);

        assert!(
            (back - system_pos).length() < 1e-6,
            "round-trip error: {back} != {system_pos}"
        );
    }

    #[test]
    fn nearest_star() {
        let map = GalaxyMap::generate(42);
        let star = &map.stars[0];
        let nearby = star.position + DVec3::new(1.0, 0.0, 0.0);
        let nearest = map.nearest_star(nearby).unwrap();
        assert_eq!(nearest.index, star.index);
    }

    #[test]
    fn star_class_distribution() {
        let map = GalaxyMap::generate(42);
        let m_count = map.stars.iter().filter(|s| s.star_class == StarClass::M).count();
        // M stars should be the majority (>50%).
        let ratio = m_count as f64 / map.stars.len() as f64;
        assert!(
            ratio > 0.5,
            "M stars only {:.1}% — expected majority",
            ratio * 100.0
        );
    }

    #[test]
    fn different_seeds_different_galaxies() {
        let a = GalaxyMap::generate(42);
        let b = GalaxyMap::generate(43);
        // Star counts or positions should differ.
        assert!(
            a.stars.len() != b.stars.len()
                || a.stars[0].position != b.stars[0].position,
            "different seeds produced identical galaxies"
        );
    }
}
