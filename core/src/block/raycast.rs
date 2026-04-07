//! DDA (Amanatides & Woo) voxel raycast — traces a ray through a block grid
//! and returns the first solid block hit.
//!
//! Generic over the block lookup via `is_solid` callback, so it works with:
//! - **Server**: `ShipGrid` (authoritative block edit)
//! - **Client**: `ClientChunkCache` (visual block highlight)
//! - **Tests**: any closure

use glam::{IVec3, Vec3};

/// Result of a successful block raycast.
#[derive(Clone, Debug)]
pub struct BlockHit {
    /// World-space position of the block that was hit.
    pub world_pos: IVec3,
    /// Which face of the block the ray entered through.
    /// One of: ±X, ±Y, ±Z (axis-aligned unit vector).
    pub face_normal: IVec3,
    /// Distance from ray origin to the hit point.
    pub distance: f32,
}

/// Cast a ray through a voxel grid using the DDA algorithm.
///
/// Visits exactly the voxels the ray passes through — no skips, no redundant
/// checks. Each step crosses exactly one axis-aligned voxel boundary.
///
/// # Arguments
/// - `origin`: ray start position (f32, continuous coordinates)
/// - `direction`: ray direction (does not need to be normalized, but must not be zero)
/// - `max_distance`: maximum ray length before giving up
/// - `is_solid`: callback that returns true if the block at (x, y, z) is solid
///
/// # Returns
/// `Some(BlockHit)` if a solid block was found within `max_distance`,
/// `None` if the ray passed through only air/empty space.
pub fn raycast(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    is_solid: impl Fn(i32, i32, i32) -> bool,
) -> Option<BlockHit> {
    // Normalize direction for consistent distance computation.
    let dir_len = direction.length();
    if dir_len < 1e-10 {
        return None;
    }
    let dir = direction / dir_len;

    // Current voxel position (floor of origin).
    let mut voxel = IVec3::new(
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    );

    // Step direction: +1 or -1 per axis.
    let step = IVec3::new(
        if dir.x >= 0.0 { 1 } else { -1 },
        if dir.y >= 0.0 { 1 } else { -1 },
        if dir.z >= 0.0 { 1 } else { -1 },
    );

    // t_max: distance along the ray to the next voxel boundary on each axis.
    // t_delta: distance along the ray to cross one full voxel on each axis.
    let mut t_max = Vec3::new(
        t_to_boundary(origin.x, dir.x, step.x),
        t_to_boundary(origin.y, dir.y, step.y),
        t_to_boundary(origin.z, dir.z, step.z),
    );

    let t_delta = Vec3::new(
        if dir.x.abs() > 1e-10 { (1.0 / dir.x).abs() } else { f32::MAX },
        if dir.y.abs() > 1e-10 { (1.0 / dir.y).abs() } else { f32::MAX },
        if dir.z.abs() > 1e-10 { (1.0 / dir.z).abs() } else { f32::MAX },
    );

    // Track which axis was crossed last (for face normal computation).
    // -1 = none yet (initial voxel).
    let mut last_axis: i32 = -1;
    let mut distance = 0.0f32;

    // Check the starting voxel first (player might be inside a block).
    if is_solid(voxel.x, voxel.y, voxel.z) {
        return Some(BlockHit {
            world_pos: voxel,
            face_normal: IVec3::ZERO, // inside the block, no face
            distance: 0.0,
        });
    }

    // DDA traversal loop.
    loop {
        // Step to the next voxel boundary on the axis with smallest t_max.
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                distance = t_max.x;
                voxel.x += step.x;
                t_max.x += t_delta.x;
                last_axis = 0;
            } else {
                distance = t_max.z;
                voxel.z += step.z;
                t_max.z += t_delta.z;
                last_axis = 2;
            }
        } else if t_max.y < t_max.z {
            distance = t_max.y;
            voxel.y += step.y;
            t_max.y += t_delta.y;
            last_axis = 1;
        } else {
            distance = t_max.z;
            voxel.z += step.z;
            t_max.z += t_delta.z;
            last_axis = 2;
        }

        // Check distance limit.
        if distance > max_distance {
            return None;
        }

        // Check if this voxel is solid.
        if is_solid(voxel.x, voxel.y, voxel.z) {
            // Face normal: the axis we crossed, pointing back toward the ray origin.
            let face_normal = match last_axis {
                0 => IVec3::new(-step.x, 0, 0),
                1 => IVec3::new(0, -step.y, 0),
                2 => IVec3::new(0, 0, -step.z),
                _ => IVec3::ZERO,
            };
            return Some(BlockHit {
                world_pos: voxel,
                face_normal,
                distance,
            });
        }
    }
}

/// Compute the parametric distance along a ray to the next voxel boundary
/// on a single axis.
#[inline]
fn t_to_boundary(origin_component: f32, dir_component: f32, step: i32) -> f32 {
    if dir_component.abs() < 1e-10 {
        return f32::MAX; // ray is parallel to this axis
    }
    let boundary = if step > 0 {
        origin_component.floor() + 1.0
    } else {
        origin_component.ceil() - 1.0
    };
    let t = (boundary - origin_component) / dir_component;
    // Clamp to avoid negative t when origin is exactly on a boundary.
    t.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: a 3x3x3 solid cube centered at (5, 5, 5).
    fn is_solid_cube(x: i32, y: i32, z: i32) -> bool {
        x >= 4 && x <= 6 && y >= 4 && y <= 6 && z >= 4 && z <= 6
    }

    #[test]
    fn hit_straight_on() {
        // Ray from (0, 5.5, 5.5) looking in +X direction → hits block (4, 5, 5).
        let hit = raycast(
            Vec3::new(0.0, 5.5, 5.5),
            Vec3::new(1.0, 0.0, 0.0),
            20.0,
            is_solid_cube,
        );
        let hit = hit.expect("should hit");
        assert_eq!(hit.world_pos, IVec3::new(4, 5, 5));
        assert_eq!(hit.face_normal, IVec3::new(-1, 0, 0)); // -X face (ray entered from left)
    }

    #[test]
    fn hit_from_above() {
        // Ray from (5.5, 10, 5.5) looking down → hits block (5, 6, 5).
        let hit = raycast(
            Vec3::new(5.5, 10.0, 5.5),
            Vec3::new(0.0, -1.0, 0.0),
            20.0,
            is_solid_cube,
        );
        let hit = hit.expect("should hit");
        assert_eq!(hit.world_pos, IVec3::new(5, 6, 5));
        assert_eq!(hit.face_normal, IVec3::new(0, 1, 0)); // +Y face (ray entered from above)
    }

    #[test]
    fn miss_completely() {
        // Ray going in +X but above the cube.
        let hit = raycast(
            Vec3::new(0.0, 20.0, 5.5),
            Vec3::new(1.0, 0.0, 0.0),
            100.0,
            is_solid_cube,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn max_distance_stops_search() {
        // Ray going in +X toward the cube but max_distance too short.
        let hit = raycast(
            Vec3::new(0.0, 5.5, 5.5),
            Vec3::new(1.0, 0.0, 0.0),
            2.0, // cube starts at x=4, origin at x=0 → distance ~4
            is_solid_cube,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn inside_solid_block() {
        // Origin is inside a solid block.
        let hit = raycast(
            Vec3::new(5.5, 5.5, 5.5),
            Vec3::new(1.0, 0.0, 0.0),
            10.0,
            is_solid_cube,
        );
        let hit = hit.expect("should hit immediately");
        assert_eq!(hit.world_pos, IVec3::new(5, 5, 5));
        assert_eq!(hit.distance, 0.0);
    }

    #[test]
    fn diagonal_hit() {
        // Ray at 45° angle in XY plane.
        let hit = raycast(
            Vec3::new(0.0, 0.0, 5.5),
            Vec3::new(1.0, 1.0, 0.0),
            20.0,
            is_solid_cube,
        );
        let hit = hit.expect("should hit");
        // Should hit the corner region — exact block depends on DDA stepping.
        assert!(hit.world_pos.x >= 4 && hit.world_pos.x <= 6);
        assert!(hit.world_pos.y >= 4 && hit.world_pos.y <= 6);
    }

    #[test]
    fn hit_from_negative_direction() {
        // Ray from +X looking in -X direction.
        let hit = raycast(
            Vec3::new(10.0, 5.5, 5.5),
            Vec3::new(-1.0, 0.0, 0.0),
            20.0,
            is_solid_cube,
        );
        let hit = hit.expect("should hit");
        assert_eq!(hit.world_pos, IVec3::new(6, 5, 5));
        assert_eq!(hit.face_normal, IVec3::new(1, 0, 0)); // +X face
    }

    #[test]
    fn hit_z_axis() {
        // Ray from -Z looking in +Z.
        let hit = raycast(
            Vec3::new(5.5, 5.5, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            20.0,
            is_solid_cube,
        );
        let hit = hit.expect("should hit");
        assert_eq!(hit.world_pos, IVec3::new(5, 5, 4));
        assert_eq!(hit.face_normal, IVec3::new(0, 0, -1)); // -Z face
    }

    #[test]
    fn empty_world_misses() {
        let hit = raycast(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            100.0,
            |_, _, _| false, // nothing solid
        );
        assert!(hit.is_none());
    }

    #[test]
    fn zero_direction_returns_none() {
        let hit = raycast(
            Vec3::new(5.5, 5.5, 5.5),
            Vec3::ZERO,
            10.0,
            is_solid_cube,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn face_normal_points_toward_ray_origin() {
        // For every axis-aligned approach, the face normal should point
        // back toward where the ray came from.
        let approaches = [
            (Vec3::new(0.0, 5.5, 5.5), Vec3::X, IVec3::NEG_X),   // +X approach → -X normal
            (Vec3::new(10.0, 5.5, 5.5), Vec3::NEG_X, IVec3::X),   // -X approach → +X normal
            (Vec3::new(5.5, 0.0, 5.5), Vec3::Y, IVec3::NEG_Y),    // +Y approach → -Y normal
            (Vec3::new(5.5, 10.0, 5.5), Vec3::NEG_Y, IVec3::Y),   // -Y approach → +Y normal
            (Vec3::new(5.5, 5.5, 0.0), Vec3::Z, IVec3::NEG_Z),    // +Z approach → -Z normal
            (Vec3::new(5.5, 5.5, 10.0), Vec3::NEG_Z, IVec3::Z),   // -Z approach → +Z normal
        ];
        for (origin, dir, expected_normal) in approaches {
            let hit = raycast(origin, dir, 20.0, is_solid_cube)
                .unwrap_or_else(|| panic!("should hit from {:?} dir {:?}", origin, dir));
            assert_eq!(
                hit.face_normal, expected_normal,
                "From {:?} dir {:?}: expected normal {:?}, got {:?}",
                origin, dir, expected_normal, hit.face_normal
            );
        }
    }
}
