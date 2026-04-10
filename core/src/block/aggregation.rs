//! Per-subsystem ship aggregation — computes physical properties from block composition.
//!
//! ## Architecture
//!
//! Aggregation is split into **modular, composable pieces**:
//! - `MassProperties`: mass, center of mass, dimensions — shared by all subsystems
//! - `ThrustProperties`: per-axis thrust + torque from thruster blocks
//! - Future: `PowerProperties`, `ShieldProperties`, `AtmosphereProperties`
//!
//! Each piece is computed independently by a pure function. The shard combines
//! them into the final `ShipPhysicalProperties` struct for the physics pipeline.
//!
//! Functions take `&impl BlockGridView`, NOT `&ShipGrid` — works for ships,
//! stations, and planet bases.

use glam::{DVec3, IVec3};

use super::block_grid::BlockGridView;
use super::block_id::BlockId;
use super::block_meta::BlockOrientation;
use super::registry::BlockRegistry;
use crate::autopilot::ShipPhysicalProperties;

// ---------------------------------------------------------------------------
// Mass aggregation (shared by all subsystems)
// ---------------------------------------------------------------------------

/// Mass properties computed from all blocks in a grid.
#[derive(Clone, Debug)]
pub struct MassProperties {
    /// Total mass from all block densities (kg). 1 block = 1m³.
    pub total_mass_kg: f64,
    /// Center of mass in world-space block coordinates.
    pub center_of_mass: DVec3,
    /// Total non-air block count.
    pub block_count: u32,
    /// Bounding box dimensions (width_x, height_y, length_z) in meters.
    pub dimensions: (f64, f64, f64),
    /// Cross-section areas derived from bounding box.
    pub cross_section_front: f64, // width × height (facing Z)
    pub cross_section_side: f64,  // length × height (facing X)
    pub cross_section_top: f64,   // width × length (facing Y)
}

/// Compute mass properties from all blocks in a grid.
///
/// Iterates every non-air block to sum mass, compute center of mass,
/// and determine bounding box dimensions.
pub fn compute_mass_properties(
    grid: &impl BlockGridView,
    registry: &BlockRegistry,
) -> MassProperties {
    let mut total_mass = 0.0f64;
    let mut weighted_pos = DVec3::ZERO;
    let mut block_count = 0u32;
    let mut min_pos = IVec3::MAX;
    let mut max_pos = IVec3::MIN;

    for (pos, block_id) in grid.iter_blocks() {
        let def = registry.get(block_id);
        let mass = def.density as f64; // kg/m³ × 1m³ = kg

        total_mass += mass;
        // Block center is at pos + 0.5 in each axis.
        let center = DVec3::new(pos.x as f64 + 0.5, pos.y as f64 + 0.5, pos.z as f64 + 0.5);
        weighted_pos += center * mass;
        block_count += 1;

        min_pos = min_pos.min(pos);
        max_pos = max_pos.max(pos);
    }

    let center_of_mass = if total_mass > 0.0 {
        weighted_pos / total_mass
    } else {
        DVec3::ZERO
    };

    let dimensions = if block_count > 0 {
        (
            (max_pos.x - min_pos.x + 1) as f64,
            (max_pos.y - min_pos.y + 1) as f64,
            (max_pos.z - min_pos.z + 1) as f64,
        )
    } else {
        (1.0, 1.0, 1.0)
    };

    MassProperties {
        total_mass_kg: total_mass.max(1.0), // minimum 1 kg to avoid division by zero
        center_of_mass,
        block_count,
        dimensions,
        cross_section_front: dimensions.0 * dimensions.1, // width × height
        cross_section_side: dimensions.2 * dimensions.1,  // length × height
        cross_section_top: dimensions.0 * dimensions.2,   // width × length
    }
}

// ---------------------------------------------------------------------------
// Thrust aggregation (Phase 4)
// ---------------------------------------------------------------------------

/// Thrust capabilities computed from thruster blocks.
#[derive(Clone, Debug)]
pub struct ThrustProperties {
    /// Per-axis maximum thrust (Newtons): [+X, -X, +Y, -Y, +Z, -Z].
    pub thrust_per_axis: [f64; 6],
    /// Maximum torque the ship can produce around each axis (N·m).
    /// Components: (roll around X, pitch around Y, yaw around Z).
    pub max_torque: DVec3,
    /// Number of thruster blocks.
    pub thruster_count: u32,
}

/// Compute thrust properties from thruster block positions and orientations.
///
/// Each thruster contributes:
/// - **Linear thrust** in its facing direction
/// - **Torque** from its offset relative to center of mass (cross product)
///
/// A thruster at the center of mass produces only linear thrust.
/// A thruster at the wingtip produces both linear thrust and torque.
pub fn compute_thrust_properties(
    thrusters: &[(IVec3, BlockId, BlockOrientation)],
    center_of_mass: DVec3,
    registry: &BlockRegistry,
) -> ThrustProperties {
    let mut thrust_per_axis = [0.0f64; 6];
    let mut max_torque = DVec3::ZERO;
    let mut count = 0u32;

    for &(pos, block_id, orientation) in thrusters {
        let props = match registry.thruster_props(block_id) {
            Some(p) => p,
            None => continue,
        };

        let thrust_n = props.thrust_n;
        let facing = orientation.facing();

        // Linear thrust contribution — add to the axis this thruster faces.
        if (facing as usize) < 6 {
            thrust_per_axis[facing as usize] += thrust_n;
        }

        // Torque contribution — cross product of offset from CoM × thrust vector.
        let block_center = DVec3::new(pos.x as f64 + 0.5, pos.y as f64 + 0.5, pos.z as f64 + 0.5);
        let offset = block_center - center_of_mass;
        let thrust_dir = orientation.facing_direction();
        let thrust_vec = thrust_dir * thrust_n;

        // torque = offset × thrust_vector
        // We take abs() because we want the MAXIMUM torque capability per axis,
        // regardless of rotation direction. The actual rotation direction is
        // determined at runtime by pilot input.
        let torque = offset.cross(thrust_vec);
        max_torque += torque.abs();

        count += 1;
    }

    ThrustProperties {
        thrust_per_axis,
        max_torque,
        thruster_count: count,
    }
}

// ---------------------------------------------------------------------------
// Conversion to ShipPhysicalProperties
// ---------------------------------------------------------------------------

/// Combine mass + thrust aggregation into the physics pipeline's
/// `ShipPhysicalProperties` struct.
///
/// This is the bridge between the modular aggregation system and the
/// existing monolithic physics properties struct.
pub fn build_ship_properties(
    mass: &MassProperties,
    thrust: &ThrustProperties,
) -> ShipPhysicalProperties {
    // Total thrust across all axes (for thrust_multiplier scaling).
    let total_thrust: f64 = thrust.thrust_per_axis.iter().sum();

    // thrust_multiplier scales the ENGINE_TIERS base thrust values.
    // If total thrust matches Cruise tier (490.5 MN), multiplier = 1.0.
    let reference_thrust = 490_500_000.0; // Cruise tier base
    let thrust_multiplier = if total_thrust > 0.0 {
        (total_thrust / reference_thrust).max(0.001)
    } else {
        0.0 // no thrusters = no thrust capability
    };

    // Forward thrust = thrusters facing -Z (index 5), because a thruster
    // facing -Z pushes the ship in +Z... wait, convention:
    // A thruster facing +Z has its exhaust going -Z, so it pushes the ship +Z (backward).
    // A thruster facing -Z has its exhaust going +Z, so it pushes the ship -Z (forward).
    //
    // In the ship's coordinate system, forward = -Z.
    // So "forward thrust" = thrust from thrusters facing +Z (index 4),
    // because they push the ship in -Z (forward).
    //
    // Actually, let's reconsider:
    // facing 4 = +Z direction. A thruster oriented facing +Z emits exhaust -Z,
    // pushing the ship +Z. That's BACKWARD (away from -Z forward).
    //
    // facing 5 = -Z direction. A thruster oriented facing -Z emits exhaust +Z,
    // pushing the ship -Z. That's FORWARD.
    //
    // So forward thrust = thrust_per_axis[5] (-Z facing thrusters).
    let max_thrust_forward = thrust.thrust_per_axis[5]; // -Z facing = pushes forward
    let max_thrust_reverse = thrust.thrust_per_axis[4]; // +Z facing = pushes backward

    let max_torque = thrust.max_torque.length().max(1.0); // minimum 1 N·m
    let max_torque_per_axis = DVec3::new(
        thrust.max_torque.x.max(1.0),
        thrust.max_torque.y.max(1.0),
        thrust.max_torque.z.max(1.0),
    );

    ShipPhysicalProperties {
        mass_kg: mass.total_mass_kg,
        cross_section_front: mass.cross_section_front,
        cross_section_side: mass.cross_section_side,
        cross_section_top: mass.cross_section_top,
        cd_front: 1.2, // streamlined
        cd_side: 2.0,  // flat plate
        cd_top: 2.0,   // flat plate
        thrust_multiplier,
        dimensions: mass.dimensions,
        cop_offset_z: mass.dimensions.2 * 0.1, // 10% of length behind CoM
        thermal_capacity_j: mass.total_mass_kg * 50_000.0, // 50 kJ per kg of hull
        thermal_emissivity: 0.8,
        nose_radius_m: (mass.dimensions.0 * mass.dimensions.1).sqrt() / 2.0,
        landing_gear_height: 1.5,
        hull_strength: mass.block_count as f64 * 2.0,
        max_thrust_forward_n: max_thrust_forward,
        max_thrust_reverse_n: max_thrust_reverse,
        max_torque_nm: max_torque,
        max_torque_per_axis,
        thrust_per_axis: thrust.thrust_per_axis,
        available_tiers: 0b11111, // all tiers available (tier thrust × multiplier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockRegistry, ShipGrid, build_starter_ship, StarterShipLayout};

    #[test]
    fn mass_of_starter_ship() {
        let registry = BlockRegistry::new();
        let grid = build_starter_ship(&StarterShipLayout::default_starter());
        let mass = compute_mass_properties(&grid, &registry);

        assert!(mass.total_mass_kg > 1000.0, "starter ship should weigh more than 1t");
        assert!(mass.block_count > 100, "starter ship should have 100+ blocks");
        assert!(mass.dimensions.0 > 5.0, "width should be > 5m");
        assert!(mass.dimensions.1 > 3.0, "height should be > 3m");
    }

    #[test]
    fn center_of_mass_symmetric_ship() {
        let registry = BlockRegistry::new();
        let mut grid = ShipGrid::new();
        // Place 4 blocks symmetrically around origin, all same density.
        grid.set_block(-1, 0, 0, BlockId::HULL_STANDARD);
        grid.set_block(0, 0, 0, BlockId::HULL_STANDARD);
        grid.set_block(-1, 0, 1, BlockId::HULL_STANDARD);
        grid.set_block(0, 0, 1, BlockId::HULL_STANDARD);

        let mass = compute_mass_properties(&grid, &registry);
        // CoM should be at the center of the 4 blocks: x=0.0, y=0.5, z=1.0
        assert!((mass.center_of_mass.x - 0.0).abs() < 0.01, "CoM X should be ~0.0, got {}", mass.center_of_mass.x);
        assert!((mass.center_of_mass.y - 0.5).abs() < 0.01, "CoM Y should be ~0.5");
        assert!((mass.center_of_mass.z - 1.0).abs() < 0.01, "CoM Z should be ~1.0");
    }

    #[test]
    fn empty_grid_safe() {
        let registry = BlockRegistry::new();
        let grid = ShipGrid::new();
        let mass = compute_mass_properties(&grid, &registry);

        assert_eq!(mass.block_count, 0);
        assert!(mass.total_mass_kg >= 1.0, "minimum mass should be 1 kg");
    }

    #[test]
    fn thrust_per_axis() {
        let registry = BlockRegistry::new();
        let com = DVec3::new(0.5, 0.5, 0.5);

        // One thruster facing -Z (forward).
        let thrusters = vec![
            (IVec3::new(0, 0, 2), BlockId::THRUSTER_SMALL_CHEMICAL, BlockOrientation::new(5, 0)),
        ];
        let thrust = compute_thrust_properties(&thrusters, com, &registry);

        assert_eq!(thrust.thruster_count, 1);
        assert!(thrust.thrust_per_axis[5] > 0.0, "-Z axis should have thrust");
        assert_eq!(thrust.thrust_per_axis[0], 0.0, "+X should be zero");
    }

    #[test]
    fn torque_from_offset_thruster() {
        let registry = BlockRegistry::new();
        let com = DVec3::new(5.5, 0.5, 5.5);

        // Thruster at x=10, far from CoM on X axis, facing -Z.
        // offset.x = 10.5 - 5.5 = 5.0, thrust in -Z.
        // torque = offset × thrust = (5, 0, 0) × (0, 0, -T) = (0, 5T, 0)
        // → pitch torque (Y axis).
        let thrusters = vec![
            (IVec3::new(10, 0, 5), BlockId::THRUSTER_SMALL_CHEMICAL, BlockOrientation::new(5, 0)),
        ];
        let thrust = compute_thrust_properties(&thrusters, com, &registry);

        assert!(thrust.max_torque.y > 0.0, "should have pitch torque from offset thruster");
    }

    #[test]
    fn no_torque_at_com() {
        let registry = BlockRegistry::new();
        // Thruster exactly at center of mass.
        let com = DVec3::new(0.5, 0.5, 0.5);
        let thrusters = vec![
            (IVec3::new(0, 0, 0), BlockId::THRUSTER_SMALL_CHEMICAL, BlockOrientation::new(5, 0)),
        ];
        let thrust = compute_thrust_properties(&thrusters, com, &registry);

        // Offset = (0,0,0), so cross product should be zero.
        assert!(thrust.max_torque.length() < 0.01, "thruster at CoM should produce no torque");
    }

    #[test]
    fn build_properties_no_thrusters() {
        let registry = BlockRegistry::new();
        let grid = build_starter_ship(&StarterShipLayout::default_starter());
        let mass = compute_mass_properties(&grid, &registry);
        let thrust = compute_thrust_properties(&[], mass.center_of_mass, &registry);
        let props = build_ship_properties(&mass, &thrust);

        assert_eq!(props.thrust_multiplier, 0.0, "no thrusters = zero thrust multiplier");
        assert_eq!(props.max_thrust_forward_n, 0.0);
    }
}
