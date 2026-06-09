//! `StampedPose` / `FrameRef` — the ONE coordinate-frame transfer type
//! (integration resolution: unifies the designs' `ReferenceFrameDef`/`StampedPose`/
//! `FrameRef` into a single shared shape).
//!
//! Frames are time-parameterized: a pose is meaningful only together with the
//! `universe_tick` it was stamped at, and consumers re-evaluate frame functions at
//! their own render/sim instant — never by comparing wall clocks (R6: ~110 km/s
//! orbital velocity makes 1 ms of clock skew ≈ 110 m of error).
//!
//! The SOURCE shard computes a transfer's destination pose and ships it; the dest
//! sanity-bounds and uses it (authority never depends on cross-binary float
//! reproducibility).

use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::ids::{EntityId, UniverseTick};

/// A persistence/ownership realm: the unit of single-writer durable state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RealmId {
    /// A planet's surface world, keyed by its deterministic seed.
    Planet(u64),
    /// A star system's space, keyed by its deterministic seed.
    System(u64),
    /// A ship's interior world, keyed by the ship entity.
    Ship(EntityId),
}

impl core::fmt::Display for RealmId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RealmId::Planet(seed) => write!(f, "planet-{seed:016x}"),
            RealmId::System(seed) => write!(f, "system-{seed:016x}"),
            RealmId::Ship(id) => write!(f, "ship-{id}"),
        }
    }
}

/// The reference frame a pose is expressed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FrameRef {
    /// Planet-centered Cartesian, planet center at origin (surface worlds).
    PlanetCentered { planet_seed: u64 },
    /// A ship's interior grid frame (moves with the hull).
    ShipLocal { ship: EntityId },
    /// Star-system space, star at origin.
    SystemSpace { system_seed: u64 },
    /// Galaxy space: integer light-year cells + f64 offsets keep f64 precision
    /// (the cell layout lands with the galaxy work, P10).
    GalaxySpace,
}

impl FrameRef {
    /// The realm whose owner is authoritative for entities expressed in this frame,
    /// when one exists (GalaxySpace has no single realm owner).
    #[must_use]
    pub fn realm(self) -> Option<RealmId> {
        match self {
            FrameRef::PlanetCentered { planet_seed } => Some(RealmId::Planet(planet_seed)),
            FrameRef::ShipLocal { ship } => Some(RealmId::Ship(ship)),
            FrameRef::SystemSpace { system_seed } => Some(RealmId::System(system_seed)),
            FrameRef::GalaxySpace => None,
        }
    }

    /// A short human-readable label for the player-stats HUD and the `vdctl` location
    /// readout — the player's "where am I", derived from their authoritative frame, NOT a
    /// raw shard id (the client never sees shard processes; this is fence-validated and
    /// changes only on a real cross-realm move). Stub realms have no name yet, so it is
    /// the realm KIND + its seed/id; named planets/ships (P4/P8) refine the text here
    /// without changing the seam.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            FrameRef::PlanetCentered { planet_seed } => format!("Planet {planet_seed}"),
            FrameRef::ShipLocal { ship } => format!("Ship {ship}"),
            FrameRef::SystemSpace { system_seed } => format!("System {system_seed}"),
            FrameRef::GalaxySpace => "Galaxy".to_owned(),
        }
    }
}

/// A pose + motion state bound to one frame at one analytic-clock instant.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StampedPose {
    pub frame: FrameRef,
    pub pos: DVec3,
    pub vel: DVec3,
    pub orient: DQuat,
    pub universe_tick: UniverseTick,
}

impl StampedPose {
    /// A rest pose at a position — the common spawn/test constructor.
    #[must_use]
    pub fn at_rest(frame: FrameRef, pos: DVec3, universe_tick: UniverseTick) -> StampedPose {
        StampedPose {
            frame,
            pos,
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick,
        }
    }

    /// A copy with every non-finite component replaced by a safe default (`pos`/`vel`
    /// per-component → 0, `orient` → identity). Delivered poses ride the wire, so a
    /// corrupt/diverged sender could carry `NaN`/`Inf`; the CLIENT must never feed one
    /// into its render transforms (it would poison the whole scene graph) — so the view
    /// sanitizes HERE, at the single decode-ingress chokepoint, and EVERY downstream
    /// consumer (interpolation, the render snapshot, the DevState rows) inherits finite
    /// values. The "never trust/panic on network input" mandate, applied once.
    #[must_use]
    pub fn sanitized(self) -> StampedPose {
        StampedPose {
            frame: self.frame,
            pos: finite_or_zero(self.pos),
            vel: finite_or_zero(self.vel),
            orient: if self.orient.is_finite() {
                self.orient
            } else {
                DQuat::IDENTITY
            },
            universe_tick: self.universe_tick,
        }
    }

    /// Closed-form ballistic advance under constant acceleration for `dt_s` seconds
    /// (Category A: the ONLY way frozen/transient motion is re-advanced across hosts —
    /// never by re-stepping a physics engine, which is not cross-binary deterministic).
    #[must_use]
    pub fn advanced_ballistic(
        &self,
        accel: DVec3,
        dt_s: f64,
        new_tick: UniverseTick,
    ) -> StampedPose {
        StampedPose {
            frame: self.frame,
            pos: self.pos + self.vel * dt_s + 0.5 * accel * dt_s * dt_s,
            vel: self.vel + accel * dt_s,
            orient: self.orient,
            universe_tick: new_tick,
        }
    }
}

/// A finite scalar (`NaN`/`±Inf` → 0) — the per-component guard for [`StampedPose::sanitized`].
fn finite(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

/// A vector with each non-finite component zeroed (one bad component does not nuke the
/// other two).
fn finite_or_zero(v: DVec3) -> DVec3 {
    DVec3::new(finite(v.x), finite(v.y), finite(v.z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity_kind::EntityKind;

    fn ship_id() -> EntityId {
        EntityId::pack(EntityKind::Ship, 3, 17, 0xABCDEF)
    }

    #[test]
    fn realm_mapping_per_frame() {
        assert_eq!(
            FrameRef::PlanetCentered { planet_seed: 5 }.realm(),
            Some(RealmId::Planet(5))
        );
        assert_eq!(
            FrameRef::SystemSpace { system_seed: 9 }.realm(),
            Some(RealmId::System(9))
        );
        assert_eq!(
            FrameRef::ShipLocal { ship: ship_id() }.realm(),
            Some(RealmId::Ship(ship_id()))
        );
        assert_eq!(FrameRef::GalaxySpace.realm(), None);
    }

    #[test]
    fn label_is_human_readable_per_frame_kind() {
        // The player-facing location string for every frame variant (the HUD / vdctl
        // readout) — kind + seed/id for stub realms; GalaxySpace has no seed.
        assert_eq!(
            FrameRef::PlanetCentered { planet_seed: 5 }.label(),
            "Planet 5"
        );
        assert_eq!(FrameRef::SystemSpace { system_seed: 9 }.label(), "System 9");
        assert_eq!(
            FrameRef::ShipLocal { ship: ship_id() }.label(),
            format!("Ship {}", ship_id())
        );
        assert_eq!(FrameRef::GalaxySpace.label(), "Galaxy");
    }

    #[test]
    fn realm_display_is_unambiguous() {
        assert_eq!(RealmId::Planet(0xAB).to_string(), "planet-00000000000000ab");
        assert_eq!(RealmId::System(0xCD).to_string(), "system-00000000000000cd");
        assert!(
            RealmId::Ship(ship_id())
                .to_string()
                .starts_with("ship-ent-01.")
        );
    }

    #[test]
    fn at_rest_has_zero_motion() {
        let p = StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 1 },
            DVec3::new(1.0, 2.0, 3.0),
            UniverseTick(40),
        );
        assert_eq!(p.vel, DVec3::ZERO);
        assert_eq!(p.orient, DQuat::IDENTITY);
        assert_eq!(p.universe_tick, UniverseTick(40));
    }

    #[test]
    fn sanitized_replaces_non_finite_components_with_safe_defaults() {
        // An all-finite pose passes through unchanged.
        let good = StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 1 },
            pos: DVec3::new(1.0, 2.0, 3.0),
            vel: DVec3::new(-1.0, 0.0, 4.0),
            orient: DQuat::from_rotation_y(0.5),
            universe_tick: UniverseTick(7),
        };
        assert_eq!(good.sanitized(), good);
        // Non-finite pos/vel components are zeroed PER-COMPONENT; a non-finite orient
        // collapses to identity. Frame + tick are preserved.
        let bad = StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 1 },
            pos: DVec3::new(f64::NAN, 2.0, f64::INFINITY),
            vel: DVec3::new(1.0, f64::NEG_INFINITY, 3.0),
            orient: DQuat::from_xyzw(f64::NAN, 0.0, 0.0, 1.0),
            universe_tick: UniverseTick(7),
        };
        let s = bad.sanitized();
        assert_eq!(s.pos, DVec3::new(0.0, 2.0, 0.0));
        assert_eq!(s.vel, DVec3::new(1.0, 0.0, 3.0));
        assert_eq!(s.orient, DQuat::IDENTITY);
        assert_eq!(s.frame, bad.frame);
        assert_eq!(s.universe_tick, UniverseTick(7));
    }

    #[test]
    fn ballistic_advance_is_exact_kinematics() {
        let p0 = StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 1 },
            pos: DVec3::new(0.0, 100.0, 0.0),
            vel: DVec3::new(10.0, 0.0, 0.0),
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        };
        let g = DVec3::new(0.0, -2.0, 0.0);
        let p1 = p0.advanced_ballistic(g, 3.0, UniverseTick(60));
        // x = x0 + v*t; y = y0 + 0.5*a*t^2; v_y = a*t
        assert_eq!(p1.pos, DVec3::new(30.0, 100.0 - 9.0, 0.0));
        assert_eq!(p1.vel, DVec3::new(10.0, -6.0, 0.0));
        assert_eq!(p1.universe_tick, UniverseTick(60));
        assert_eq!(p1.frame, p0.frame);
    }

    #[test]
    fn ballistic_advance_is_composable() {
        // Advancing 2s then 3s equals advancing 5s (closed form, no accumulation drift).
        let p0 = StampedPose {
            frame: FrameRef::GalaxySpace,
            pos: DVec3::new(1.0, 2.0, 3.0),
            vel: DVec3::new(-1.0, 0.5, 2.0),
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        };
        let a = DVec3::new(0.1, -0.2, 0.3);
        let split = p0
            .advanced_ballistic(a, 2.0, UniverseTick(40))
            .advanced_ballistic(a, 3.0, UniverseTick(100));
        let whole = p0.advanced_ballistic(a, 5.0, UniverseTick(100));
        assert!((split.pos - whole.pos).length() < 1e-9);
        assert!((split.vel - whole.vel).length() < 1e-12);
    }

    #[test]
    fn stamped_pose_serde_roundtrip() {
        let p = StampedPose {
            frame: FrameRef::ShipLocal { ship: ship_id() },
            pos: DVec3::new(4.0, 5.0, 6.0),
            vel: DVec3::new(0.1, 0.2, 0.3),
            orient: DQuat::from_xyzw(0.0, 1.0, 0.0, 0.0),
            universe_tick: UniverseTick(77),
        };
        let bytes = postcard::to_allocvec(&p).expect("encode");
        let back: StampedPose = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, p);
    }
}
