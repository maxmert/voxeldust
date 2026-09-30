//! ★ THE AIR'S STANDS OVER A DRY SEA FLOOR (ruling B25, 2026-09-30; the gate G-AIR-UNDER and
//! `examples/air_ground_stand.rs`).
//!
//! The owner flew low over the dry sea floor and the haze vanished: the eye stood under the air
//! model's ground sphere. The cure (the air rests on the sea and reaches every ground) is judged at a
//! stand that puts the eye UNDER the sphere, over solved ground that lies deep under the sea. This
//! finds such ground: from a start direction, along evenly spread bearings, the nearest direction
//! whose solved ground lies at least a stated depth under a stated sphere, where the star stands at
//! least a stated elevation up (so the air is lit). It also states the level direction there back
//! toward the start, so a stand faces the rising ground and the coast.
//!
//! Example: from the owner's coast stand (the star 15° up), the floor 4 136 m under the ladder's
//! datum, 7 727 m under the sea, lies 20 km off; an eye 2 km over that floor stands 5.7 km under the
//! air's ground sphere and looks back at the coast.

use vd_core::glam::{DQuat, DVec3};

/// What the search asks for, and how it looks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloorSearch {
    /// The sphere the floor must lie under, metres from the centre (the air's ground: the sea).
    pub sphere_m: f64,
    /// How far under that sphere the ground must lie, metres.
    pub depth_m: f64,
    /// The least elevation of the star over the floor's horizon, degrees.
    pub min_sun_deg: f64,
    /// The bearings round the start, evenly spread.
    pub bearings: u32,
    /// The step along a bearing, metres of arc on `sphere_m`.
    pub step_m: f64,
    /// The farthest arc searched, metres.
    pub reach_m: f64,
}

/// The floor the search found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AirFloor {
    /// The direction from the body's centre (unit, the body frame).
    pub dir: DVec3,
    /// The ground's radius there, metres from the centre.
    pub ground_m: f64,
    /// The arc from the start, metres.
    pub along_m: f64,
    /// The level direction there, back toward the start (unit).
    pub back: DVec3,
}

/// ★ THE NEAREST DEEP FLOOR from `from` (unit), with the ground's radius along a direction given by
/// `ground`, under the star's direction `sun` (unit). The nearest over every bearing wins; `None`
/// where no bearing finds one within the reach.
#[must_use]
pub fn nearest_floor(
    ground: &dyn Fn(DVec3) -> f64,
    from: DVec3,
    sun: DVec3,
    search: &FloorSearch,
) -> Option<AirFloor> {
    let east = DVec3::Z.cross(from).try_normalize().unwrap_or(DVec3::X);
    let steps = (search.reach_m / search.step_m) as u32;
    let mut best: Option<AirFloor> = None;
    for b in 0..search.bearings {
        let bearing = f64::from(b) * std::f64::consts::TAU / f64::from(search.bearings);
        let tangent = DQuat::from_axis_angle(from, bearing) * east;
        let axis = from.cross(tangent).normalize();
        for s in 1..=steps {
            let along_m = f64::from(s) * search.step_m;
            if best.is_some_and(|f| along_m >= f.along_m) {
                break;
            }
            let dir = (DQuat::from_axis_angle(axis, along_m / search.sphere_m) * from).normalize();
            let ground_m = ground(dir);
            let deep = ground_m <= search.sphere_m - search.depth_m;
            let lit = sun.dot(dir).asin().to_degrees() >= search.min_sun_deg;
            if deep && lit {
                let back = (from - dir * from.dot(dir)).normalize();
                best = Some(AirFloor {
                    dir,
                    ground_m,
                    along_m,
                    back,
                });
                break;
            }
        }
    }
    best
}

/// The facing of a stand at `dir` looking along `back` pitched `pitch_deg` under the level, as the
/// body-frame rotation whose −Z is the nose and whose +Y is up (the spawn poses' convention).
#[must_use]
pub fn facing(dir: DVec3, back: DVec3, pitch_deg: f64) -> DQuat {
    let pitch = pitch_deg.to_radians();
    let forward = (back * pitch.cos() - dir * pitch.sin()).normalize();
    let right = forward.cross(dir).normalize();
    let up = right.cross(forward);
    DQuat::from_mat3(&vd_core::glam::DMat3::from_cols(right, up, -forward)).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: f64 = 6_000_000.0;

    fn search() -> FloorSearch {
        FloorSearch {
            sphere_m: R,
            depth_m: 4_000.0,
            min_sun_deg: 10.0,
            bearings: 8,
            step_m: 2_000.0,
            reach_m: 400_000.0,
        }
    }

    /// A bowl 5 km deep round +X: the ground is at `R` everywhere else.
    fn bowl(d: DVec3) -> f64 {
        if d.dot(DVec3::X) > (100_000.0 / R).cos() {
            R - 5_000.0
        } else {
            R
        }
    }

    #[test]
    fn the_nearest_lit_deep_floor_is_found_and_faces_back() {
        // Start 200 km from the bowl's centre, the sun overhead there.
        let from = (DQuat::from_rotation_z(200_000.0 / R) * DVec3::X).normalize();
        let f = nearest_floor(&bowl, from, from, &search()).expect("the bowl is in reach");
        assert_eq!(f.ground_m, R - 5_000.0);
        // The bowl's rim stands 100 km off its centre: the nearest floor lies 100 km from the start,
        // to within one step.
        assert!((f.along_m - 100_000.0).abs() <= 2_000.0, "{}", f.along_m);
        // The back direction is level at the floor and points toward the start.
        assert!(f.back.dot(f.dir).abs() < 1e-9);
        assert!(f.back.dot(from) > 0.0);
        let q = facing(f.dir, f.back, 2.0);
        let nose = q * DVec3::NEG_Z;
        let pitch = nose.dot(f.dir).asin().to_degrees();
        assert!((pitch + 2.0).abs() < 1e-9, "{pitch}");
        assert!((q * DVec3::Y).dot(f.dir) > 0.99);
    }

    #[test]
    fn a_dark_or_distant_floor_is_not_found() {
        let from = (DQuat::from_rotation_z(200_000.0 / R) * DVec3::X).normalize();
        // The sun under the horizon everywhere near: no lit floor.
        assert_eq!(nearest_floor(&bowl, from, -from, &search()), None);
        // The reach short of the bowl.
        let short = FloorSearch {
            reach_m: 50_000.0,
            ..search()
        };
        assert_eq!(nearest_floor(&bowl, from, from, &short), None);
    }
}
