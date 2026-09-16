//! ★ THE ONE PLACE where a realm's own body becomes its grid (the voxel foundation, slice 2).
//!
//! The grid family is PER-BODY DATA, never per realm kind (ruling V5): a body the seed makes large
//! and round enough to be a world gets the cube-sphere; a small or irregular body and every built
//! realm get the flat grid. So this function is fed the BODY, not the profile: a round body's look
//! radius, a lump's look radius, or a built realm's sold slot. The caller says which of the three it
//! holds, from the body's own seed data; the capability profile's `VoxelGeometry` is a shard-kind
//! fact for re-home matching and is NOT read here.
//!
//! The extent is a typed value, so a caller cannot hand the round arm a BOUND (the sphere of
//! influence, about two hundred times the look at real scale) where the LOOK belongs. The one fault
//! that could build a grid two hundred times too large is unwritable.
//!
//! **No caller exists yet.** The first consumer is the block store (slice 9), where a shard first
//! needs its grid. Until then this is the derivation, tested alone. The threshold between a round
//! body and a lump is the body definition's (slice 5, the world identity); today the caller decides.
//!
//! **Example.** A moon's shard knows its own look: a shell of 200 km. The grid is the cube-sphere with
//! 311 296 cells per face edge and fourteen rungs. A hull's shard knows its bound: a box of 16 by 8 by 32
//! metres. The grid is the flat box in whole cells, and a thruster one metre aft of the origin sits at
//! `k = −1`. A three-kilometre potato asteroid states its look as a lump and gets a flat grid
//! inscribed in it.

use vd_core::geometry::Boundary;
use vd_core::grid::{BandParams, GridMapping, IdentityGrid, ShellGrid};

/// What a realm states about its own body, for its grid: named so a look and a bound cannot be
/// confused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BodyExtent {
    /// A body the seed made large and round enough to be a world: its LOOK radius.
    Round { look_radius_m: f64 },
    /// A small or irregular seed body: its LOOK radius; the flat grid is inscribed in it.
    Lump { look_radius_m: f64 },
    /// A built realm: the slot it was sold (its BOUND), a box by ruling V6 A3, or a shell on an older
    /// row.
    Built { bound: Boundary },
}

/// The grid of a realm with this body. `None` when the body has no grid: a round body too large for
/// the address (a gas giant), or an extent that is not finite or is absurd.
#[must_use]
pub fn grid_for(extent: BodyExtent) -> Option<GridMapping> {
    match extent {
        BodyExtent::Round { look_radius_m } => {
            ShellGrid::for_body(look_radius_m, BandParams::provisional(look_radius_m))
                .map(GridMapping::CubeSphere)
        }
        BodyExtent::Lump { look_radius_m } => {
            IdentityGrid::inscribed_in_shell(look_radius_m).map(GridMapping::Identity)
        }
        BodyExtent::Built {
            bound: Boundary::Aabb { half } | Boundary::Obb { half, .. },
        } => IdentityGrid::from_half_extent_m([half.x, half.y, half.z]).map(GridMapping::Identity),
        BodyExtent::Built {
            bound: Boundary::Shell { r },
        } => IdentityGrid::inscribed_in_shell(r).map(GridMapping::Identity),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DQuat, DVec3};
    use vd_core::grid::GridDomain;

    #[test]
    fn a_round_body_gets_the_cube_sphere_from_its_look() {
        let g = grid_for(BodyExtent::Round {
            look_radius_m: 200_000.0,
        })
        .expect("moon");
        assert_eq!(
            g.domain(),
            GridDomain::Band {
                cells_per_edge: 311_296,
                floor_m: 189_985,
                band_m: 8_452,
                rungs: 14
            }
        );
    }

    #[test]
    fn a_round_body_too_large_for_the_address_has_no_grid() {
        assert_eq!(
            grid_for(BodyExtent::Round {
                look_radius_m: 1.0e8
            }),
            None
        );
    }

    #[test]
    fn a_lump_gets_a_flat_grid_inscribed_in_its_look() {
        let g = grid_for(BodyExtent::Lump {
            look_radius_m: 3_000.0,
        })
        .expect("potato");
        assert_eq!(
            g.domain(),
            GridDomain::Box {
                half_cells: [1732, 1732, 1732]
            },
            "3 000 / √3 = 1 732.05 → 1 732 cells"
        );
        assert_eq!(
            grid_for(BodyExtent::Lump {
                look_radius_m: f64::NAN
            }),
            None
        );
    }

    #[test]
    fn a_built_realm_gets_the_flat_box_from_its_slot_or_the_inscribed_cube_from_a_shell() {
        let half = DVec3::new(16.0, 8.5, 32.0);
        let boxed = grid_for(BodyExtent::Built {
            bound: Boundary::Aabb { half },
        })
        .expect("hull");
        assert_eq!(
            boxed.domain(),
            GridDomain::Box {
                half_cells: [16, 9, 32]
            }
        );
        let turned = grid_for(BodyExtent::Built {
            bound: Boundary::Obb {
                half,
                orient: DQuat::IDENTITY,
            },
        })
        .expect("station");
        assert_eq!(
            turned.domain(),
            boxed.domain(),
            "an oriented box has the same cells"
        );
        let old = grid_for(BodyExtent::Built {
            bound: Boundary::Shell { r: 10.0 },
        })
        .expect("old slot");
        assert_eq!(
            old.domain(),
            GridDomain::Box {
                half_cells: [5, 5, 5]
            },
            "10 / √3 = 5.77 → five cells"
        );
        assert_eq!(
            grid_for(BodyExtent::Built {
                bound: Boundary::Aabb {
                    half: DVec3::new(f64::INFINITY, 1.0, 1.0)
                }
            }),
            None,
            "a slot nobody could have sold"
        );
    }
}
