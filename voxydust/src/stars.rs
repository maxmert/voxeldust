//! Star field rendering: instanced billboarded quads for galaxy stars.

use glam::DVec3;
use voxeldust_core::galaxy::{GalaxyMap, StarClass, StarInfo};

/// Per-star instance data uploaded to GPU.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StarInstance {
    /// xyz = direction (skybox) or camera-relative position (galaxy), w = apparent_size.
    pub position: [f32; 4],
    /// rgb = spectral color, a = brightness multiplier.
    pub color: [f32; 4],
}

/// Scene-wide uniforms for the star pipeline.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StarSceneUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_right: [f32; 4],
    pub camera_up: [f32; 4],
    pub warp_velocity: [f32; 4], // xyz = dir, w = speed (GU/s)
    pub render_mode: [f32; 4],   // x = 0.0 skybox / 1.0 galaxy, y = streak_factor
}

/// Cached star entry with pre-computed color.
pub struct StarEntry {
    pub galaxy_position: DVec3,
    pub color: [f32; 3],
    pub luminosity: f32,
    pub index: u32,
    pub system_seed: u64,
    pub star_class: u8,
}

/// Star class name for HUD display.
pub fn star_class_name(class: u8) -> &'static str {
    match class {
        0 => "O (Blue Giant)",
        1 => "B (Blue-White)",
        2 => "A (White)",
        3 => "F (Yellow-White)",
        4 => "G (Yellow)",
        5 => "K (Orange)",
        6 => "M (Red Dwarf)",
        _ => "Unknown",
    }
}

/// Maximum number of star instances rendered per frame.
pub const MAX_STAR_INSTANCES: usize = 100_000;

/// Client-side star field state.
pub struct StarField {
    pub catalog: Vec<StarEntry>,
    pub current_star_index: Option<u32>,
    pub instances: Vec<StarInstance>,
}

impl StarField {
    /// Generate the star catalog from a galaxy seed. Deterministic.
    pub fn from_galaxy_seed(galaxy_seed: u64, current_system_seed: u64) -> Self {
        let galaxy_map = GalaxyMap::generate(galaxy_seed);

        let current_star_index = galaxy_map.stars.iter()
            .find(|s| s.system_seed == current_system_seed)
            .map(|s| s.index);

        let catalog: Vec<StarEntry> = galaxy_map.stars.iter().map(|star| {
            StarEntry {
                galaxy_position: star.position,
                color: star.star_class.color(),
                luminosity: star.luminosity as f32,
                index: star.index,
                system_seed: star.system_seed,
                star_class: star.star_class.as_u8(),
            }
        }).collect();

        StarField {
            catalog,
            current_star_index,
            instances: Vec::with_capacity(MAX_STAR_INSTANCES),
        }
    }

    /// Find the star most aligned with a camera direction. Returns index and alignment dot product.
    /// Skips the current star system and stars below the alignment threshold.
    pub fn find_aligned_star(&self, current_star_pos: DVec3, cam_fwd: DVec3, min_dot: f64) -> Option<(u32, f64)> {
        let mut best: Option<(u32, f64)> = None;
        for star in &self.catalog {
            if Some(star.index) == self.current_star_index { continue; }
            let dir = (star.galaxy_position - current_star_pos).normalize();
            let dot = cam_fwd.dot(dir);
            if dot > min_dot && dot > best.map(|b| b.1).unwrap_or(min_dot) {
                best = Some((star.index, dot));
            }
        }
        best
    }

    /// Find the next-best aligned star after the current target (for cycling).
    pub fn find_next_aligned_star(&self, current_star_pos: DVec3, cam_fwd: DVec3, current_target: u32) -> Option<(u32, f64)> {
        // Get current target's alignment.
        let current_dot = self.catalog.iter()
            .find(|s| s.index == current_target)
            .map(|s| cam_fwd.dot((s.galaxy_position - current_star_pos).normalize()))
            .unwrap_or(1.0);

        // Find the next star with slightly lower alignment.
        let mut best: Option<(u32, f64)> = None;
        for star in &self.catalog {
            if Some(star.index) == self.current_star_index { continue; }
            if star.index == current_target { continue; }
            let dir = (star.galaxy_position - current_star_pos).normalize();
            let dot = cam_fwd.dot(dir);
            // Must be less aligned than current target, but most aligned of the remaining.
            if dot < current_dot && dot > 0.3 {
                if dot > best.map(|b| b.1).unwrap_or(0.3) {
                    best = Some((star.index, dot));
                }
            }
        }
        // If no next candidate, wrap around to the best aligned.
        if best.is_none() {
            return self.find_aligned_star(current_star_pos, cam_fwd, 0.3);
        }
        best
    }

    /// Get info about a star by index.
    pub fn get_star(&self, index: u32) -> Option<&StarEntry> {
        self.catalog.iter().find(|s| s.index == index)
    }

    /// Update the current star system (e.g., after warp arrival).
    /// The current star is excluded from star field rendering because
    /// it's rendered as a celestial body by the main pipeline.
    pub fn set_current_system_seed(&mut self, system_seed: u64) {
        self.current_star_index = self.catalog.iter()
            .find(|s| s.system_seed == system_seed)
            .map(|s| s.index);
    }

    /// Update star instances for rendering. Call once per frame.
    ///
    /// - `current_star_pos`: position of the current star system in galaxy units.
    /// - `cam_galaxy_pos`: camera position in galaxy units (for galaxy shard).
    /// - `skybox_mode`: true = system shard (stars at infinity), false = galaxy shard (real positions).
    /// - `targeted_star`: optional star index to render highlighted (larger + brighter).
    pub fn update_instances(
        &mut self,
        current_star_pos: DVec3,
        cam_galaxy_pos: DVec3,
        skybox_mode: bool,
        targeted_star: Option<u32>,
    ) {
        self.instances.clear();

        for star in &self.catalog {
            // Skip the current star system (rendered as a celestial body by the main pipeline).
            if Some(star.index) == self.current_star_index {
                continue;
            }

            let is_targeted = targeted_star == Some(star.index);

            // Compute direction and distance from the reference position.
            let ref_pos = if skybox_mode { current_star_pos } else { cam_galaxy_pos };
            let offset = star.galaxy_position - ref_pos;
            let dist_sq = offset.length_squared();
            if dist_sq < 1e-6 { continue; }
            let dist = dist_sq.sqrt();
            let dir = offset / dist;

            // Apparent magnitude using log scale (like real astronomy).
            let flux = star.luminosity as f64 / dist_sq;
            let mag = -2.5 * flux.max(1e-20).log10();

            let brightness = if is_targeted {
                1.0_f32
            } else {
                (5.0 - mag as f32 * 0.25).clamp(0.1, 1.0)
            };

            // Skip very dim stars.
            if !is_targeted && brightness < 0.12 {
                continue;
            }

            // Billboard size: visible multi-pixel glows. The additive blend
            // shader scales alpha very low (×0.06), so overlap accumulates
            // to a Milky Way band rather than saturating to white.
            let size = if is_targeted {
                4.0_f32
            } else {
                (4.0 - mag as f32 * 0.15).clamp(0.5, 3.0)
            };

            let color = if is_targeted {
                [1.0, 1.0, 1.0, brightness]
            } else {
                [star.color[0], star.color[1], star.color[2], brightness]
            };

            let dir_f32 = dir.as_vec3();
            self.instances.push(StarInstance {
                position: [dir_f32.x as f32, dir_f32.y as f32, dir_f32.z as f32, size],
                color,
            });

            if self.instances.len() >= MAX_STAR_INSTANCES { break; }
        }
    }
}
