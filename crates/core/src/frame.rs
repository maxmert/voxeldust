//! Cross-frame pose re-expression — `transfer_frame` (`docs/design/transfer_protocol.md`
//! §6; audit XFRAME-1). The single most load-bearing piece for every real transition:
//! a player crossing from SystemSpace into a planet's PlanetCentered frame, a boarding
//! pod entering a ship's ShipLocal frame, a debris chunk falling planet-ward.
//!
//! Category-A and PURE: the transform is a closed-form rigid-body composition given the
//! frames' placements; the placements themselves come from the ephemeris (also
//! closed-form `f(seed, universe_tick)`). Authority never depends on cross-binary float
//! reproducibility — the SOURCE shard computes the destination pose and ships it, the
//! dest sanity-bounds it against the orchestrator's Ephemeris Authority.
//!
//! P1 has one live frame (SystemSpace) so every placement is the identity; the algebra
//! is exercised and round-trips. P4 (planets) / P8 (ships) / P10 (warp) supply a real
//! ephemeris-backed [`FrameContext`] — the SIGNATURE landing now is what lets those
//! phases add the function bodies WITHOUT reshaping the frozen `StampedPose`/`FrameRef`.

use glam::{DQuat, DVec3};

use crate::ids::UniverseTick;
use crate::pose::{FrameRef, LatticePos, RealmId, StampedPose, frame_for_realm};

/// Where a frame's origin sits — and how it moves — relative to the COMMON PARENT at a
/// universe tick. A rigid placement: position, velocity, orientation, angular velocity.
/// All four come from the ephemeris closed-form; P1 uses the identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramePlacement {
    /// Parent-frame position of this frame's origin (metres).
    pub origin: DVec3,
    /// Parent-frame velocity of this frame's origin (m/s).
    pub velocity: DVec3,
    /// Rotation taking THIS frame's axes into the parent's axes.
    pub orientation: DQuat,
    /// Angular velocity of this frame in the parent (rad/s) — drives the velocity
    /// transform's Coriolis term for rotating frames (planet surface, spinning ship).
    pub angular_velocity: DVec3,
}

impl FramePlacement {
    /// The identity placement: this frame IS the parent (no offset, no motion).
    #[must_use]
    pub fn identity() -> FramePlacement {
        FramePlacement {
            origin: DVec3::ZERO,
            velocity: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
        }
    }

    /// A non-rotating translated/moving frame (the common celestial case: a planet
    /// orbiting a star carries no spin in SystemSpace until surface frames are added).
    #[must_use]
    pub fn moving(origin: DVec3, velocity: DVec3) -> FramePlacement {
        FramePlacement {
            origin,
            velocity,
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
        }
    }
}

/// Resolves a frame's placement relative to the common parent at a tick — the seam the
/// ephemeris implements. Returning `None` means the frame is unknown to this context
/// (e.g. a realm this shard has no ephemeris for): the caller treats that as a transfer
/// precondition failure, never a silent garbage pose.
pub trait FrameContext {
    fn placement(&self, frame: FrameRef, tick: UniverseTick) -> Option<FramePlacement>;
}

/// Why a frame re-expression could not be computed (a typed precondition failure, never
/// a silent garbage spawn — the R6 km-scale-error class made loud).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    #[error("no ephemeris placement for the source frame at this tick")]
    UnknownSourceFrame,
    #[error("no ephemeris placement for the destination frame at this tick")]
    UnknownDestFrame,
}

/// Re-express `pose` (in its current frame) into `to` using the ephemeris `ctx`. The
/// closed-form rigid-body transform: lift the local pose into the common parent, then
/// into the destination frame, carrying velocity (with the Coriolis term for rotating
/// frames) and orientation.
///
/// Same-frame transfer is the identity (and needs no context). The `universe_tick` is
/// preserved: a pose is meaningful only with the instant it was stamped at.
///
/// # Errors
/// [`FrameError`] if either frame has no placement at the pose's tick.
pub fn transfer_frame(
    pose: &StampedPose,
    to: FrameRef,
    ctx: &impl FrameContext,
) -> Result<StampedPose, FrameError> {
    // A BRANCHLESS generic shim (HR5): look up both placements as straight-line expressions and
    // delegate ALL branching — the same-frame short-circuit, the missing-placement errors, and the
    // transform — to the MONOMORPHIC `transfer_frame_resolved`. So a new `FrameContext`
    // monomorphization (e.g. the always-`Some` `IdentityFrames`) adds ZERO uncovered per-mono branch;
    // the error arms are covered ONCE in the resolved helper. (Same-frame looks the placements up but
    // ignores them — the resolved short-circuit still returns `Ok(*pose)`, unchanged behaviour.)
    transfer_frame_resolved(
        pose,
        to,
        ctx.placement(pose.frame, pose.universe_tick),
        ctx.placement(to, pose.universe_tick),
    )
}

/// The MONOMORPHIC core of [`transfer_frame`]: given the already-looked-up source + dest placements,
/// short-circuit a same-frame transfer, fail LOUD (a typed error, never a garbage pose) on a missing
/// placement, else compose the rigid-body transform. Holding every branch here keeps [`transfer_frame`]
/// a straight-line generic shim so its coverage does not multiply per `FrameContext` instantiation.
fn transfer_frame_resolved(
    pose: &StampedPose,
    to: FrameRef,
    from: Option<FramePlacement>,
    dest: Option<FramePlacement>,
) -> Result<StampedPose, FrameError> {
    if pose.frame == to {
        return Ok(*pose);
    }
    let from = from.ok_or(FrameError::UnknownSourceFrame)?;
    let dest = dest.ok_or(FrameError::UnknownDestFrame)?;

    // 1. Lift the local pose into the common parent frame. (P1-P3: cell is ZERO, so `offset()` is the
    // full frame-local position. The exact-integer cross-cell re-base — lift source cell+offset at the
    // source tier, re-quantize `new_pos` into the DEST frame's cell at its tier unit, replacing the
    // provisional `LatticePos::local(new_pos)` cell-0 write below — lands WITH P4/P5 re-centering, galaxy
    // ly-cells at P10. D-41 plant-item 2 ["exact-integer re-base in transfer_frame"].)
    let local = pose.pos.offset();
    let world_pos = from.origin + from.orientation * local;
    let lever = from.orientation * local;
    let world_vel =
        from.velocity + from.orientation * pose.vel + from.angular_velocity.cross(lever);
    let world_orient = from.orientation * pose.orient;

    // 2. Express the parent-frame pose in the destination frame.
    let inv = dest.orientation.inverse();
    let rel = world_pos - dest.origin;
    let new_pos = inv * rel;
    let new_vel = inv * (world_vel - dest.velocity - dest.angular_velocity.cross(rel));
    let new_orient = inv * world_orient;

    Ok(StampedPose {
        frame: to,
        pos: LatticePos::local(new_pos),
        vel: new_vel,
        orient: new_orient.normalize(),
        universe_tick: pose.universe_tick,
    })
}

/// A tick-independent [`FrameContext`] whose every frame is the IDENTITY placement (origin at the
/// common parent, no motion, no rotation). This is the P1-P3 reality: only `SystemSpace` is live and
/// every placement is the identity (see the module docs), so a cross-realm `transfer_frame` re-expresses
/// the FRAME field while leaving position/velocity/orientation UNCHANGED — exactly what the crossing
/// needs to rebind the authoritative pose into the dest realm's frame today. P4/P8/P10 replace this with
/// the closed-form ephemeris `FrameContext`; because the signature is frozen, that swap adds the real
/// transform WITHOUT reshaping any caller. Total + branchless: `placement` is `Some` for every frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityFrames;

impl FrameContext for IdentityFrames {
    fn placement(&self, _frame: FrameRef, _tick: UniverseTick) -> Option<FramePlacement> {
        Some(FramePlacement::identity())
    }
}

/// Rebind an authoritative pose into the frame of its destination realm — the ONE machinery (HR3) every
/// cross-realm hand-off uses to re-express a source-frame pose so the dest reads it in its OWN frame: the
/// durable crossing (`build_crossing`), the D-37 forward re-home (`build_rehome`), and the D-7 transient
/// batch (`emit_transient_batch`) all funnel through here. Through P1-P3 [`IdentityFrames`] makes this a
/// pure FRAME-field rebind — position/velocity/orientation UNCHANGED, only the frame label flips to the
/// dest realm (the dot stays put while its HUD realm advances). P4/P8/P10 swap in the closed-form
/// ephemeris `FrameContext` for the real transform with NO caller reshape (frozen signature).
///
/// `parent` is threaded as `None` today: no `Area` realm crosses through P3, so `frame_for_realm` is
/// always `Some` for a live (`System`/`Planet`/`Ship`/`Station`) dest. SAFE DEGRADE: a dest realm with no
/// nameable frame (an `Area` lacking its planet-parent provenance, owed at P4+) or a transform error
/// returns the SOURCE-frame pose UNCHANGED (the label lags) — NEVER a dropped hand-off. The proper
/// post-commit abort on a genuinely un-nameable dest is a P4 item; the degrade arm is inert now.
pub fn rebind_pose_to_dest(pose: StampedPose, to_realm: RealmId) -> StampedPose {
    frame_for_realm(to_realm, None)
        .and_then(|dest_frame| transfer_frame(&pose, dest_frame, &IdentityFrames).ok())
        .unwrap_or(pose)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections::DetHashMap;
    use std::collections::BTreeMap;

    /// A static, tick-independent ephemeris for tests and the P1 trivial case: a map
    /// from frame to placement. (P4+ replaces this with a closed-form ephemeris.)
    struct StaticFrames(BTreeMap<FrameRef, FramePlacement>);

    impl FrameContext for StaticFrames {
        fn placement(&self, frame: FrameRef, _tick: UniverseTick) -> Option<FramePlacement> {
            self.0.get(&frame).copied()
        }
    }

    fn sys() -> FrameRef {
        FrameRef::SystemSpace { system_seed: 1 }
    }
    fn planet() -> FrameRef {
        FrameRef::PlanetCentered { planet_seed: 2 }
    }

    fn pose_in(frame: FrameRef, pos: DVec3, vel: DVec3) -> StampedPose {
        StampedPose {
            frame,
            pos: LatticePos::local(pos),
            vel,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(100),
        }
    }

    #[test]
    fn same_frame_is_the_identity_without_a_context() {
        // An empty context (no placements) still returns the pose unchanged.
        let ctx = StaticFrames(BTreeMap::new());
        let p = pose_in(sys(), DVec3::new(1.0, 2.0, 3.0), DVec3::new(0.1, 0.0, 0.0));
        assert_eq!(transfer_frame(&p, sys(), &ctx), Ok(p));
    }

    #[test]
    fn unknown_frames_are_typed_errors_not_garbage() {
        let ctx = StaticFrames(BTreeMap::new());
        let p = pose_in(sys(), DVec3::ZERO, DVec3::ZERO);
        assert_eq!(
            transfer_frame(&p, planet(), &ctx),
            Err(FrameError::UnknownSourceFrame)
        );
        let mut map = BTreeMap::new();
        map.insert(sys(), FramePlacement::identity());
        let ctx = StaticFrames(map);
        assert_eq!(
            transfer_frame(&p, planet(), &ctx),
            Err(FrameError::UnknownDestFrame)
        );
    }

    #[test]
    fn identity_frames_reframes_without_moving_the_pose() {
        // The P1-P3 crossing fix: re-express a pose from SystemSpace{7} into SystemSpace{8} under
        // IdentityFrames — the FRAME field flips but position/velocity/orientation are UNCHANGED (every
        // placement is the identity). This is exactly what rebinds a crossed pose to the dest realm today.
        let from = FrameRef::SystemSpace { system_seed: 7 };
        let to = FrameRef::SystemSpace { system_seed: 8 };
        let p = pose_in(from, DVec3::new(47.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0));
        let got = transfer_frame(&p, to, &IdentityFrames).expect("identity reframe");
        assert_eq!(got.frame, to);
        assert_eq!(got.pos.offset(), p.pos.offset());
        assert_eq!(got.vel, p.vel);
        assert_eq!(got.orient, p.orient);
        assert_eq!(got.universe_tick, p.universe_tick);
    }

    #[test]
    fn rebind_pose_to_dest_flips_the_frame_to_a_nameable_realm() {
        // The Some arm (the live P3 path): a System dest is always nameable, so the pose is re-expressed
        // into SystemSpace{8} — frame flips, position/velocity/orientation unchanged under IdentityFrames.
        let from = FrameRef::SystemSpace { system_seed: 7 };
        let p = pose_in(from, DVec3::new(47.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0));
        let got = rebind_pose_to_dest(p, RealmId::System(8));
        assert_eq!(got.frame, FrameRef::SystemSpace { system_seed: 8 });
        assert_eq!(got.pos.offset(), p.pos.offset());
        assert_eq!(got.vel, p.vel);
        assert_eq!(got.orient, p.orient);
    }

    #[test]
    fn rebind_pose_to_dest_safe_degrades_an_unnameable_dest_to_the_source_pose() {
        // The None fallback arm: an Area realm has no nameable frame WITHOUT its planet parent (threaded as
        // None here, owed at P4), so `frame_for_realm` is None and the pose is returned UNCHANGED — the
        // hand-off is never dropped, only the frame label lags. This exercises `unwrap_or(pose)`.
        let from = FrameRef::SystemSpace { system_seed: 7 };
        let p = pose_in(from, DVec3::new(47.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0));
        let got = rebind_pose_to_dest(p, RealmId::Area(99));
        assert_eq!(
            got, p,
            "an un-nameable dest returns the source pose verbatim"
        );
    }

    #[test]
    fn translated_moving_frame_subtracts_origin_and_velocity() {
        // The planet's origin sits at (1000, 0, 0) in SystemSpace, moving +Y at 5 m/s.
        // A body at rest in SystemSpace at (1000, 0, 0) is at the planet ORIGIN, and in
        // the planet frame moves -Y at 5 m/s (the planet moves out from under it).
        let mut map = BTreeMap::new();
        map.insert(sys(), FramePlacement::identity());
        map.insert(
            planet(),
            FramePlacement::moving(DVec3::new(1000.0, 0.0, 0.0), DVec3::new(0.0, 5.0, 0.0)),
        );
        let ctx = StaticFrames(map);
        let p = pose_in(sys(), DVec3::new(1000.0, 0.0, 0.0), DVec3::ZERO);
        let got = transfer_frame(&p, planet(), &ctx).expect("transformed");
        assert!(
            got.pos.offset().length() < 1e-9,
            "at the planet origin: {:?}",
            got.pos
        );
        assert!(
            (got.vel - DVec3::new(0.0, -5.0, 0.0)).length() < 1e-9,
            "relative velocity: {:?}",
            got.vel
        );
        assert_eq!(got.universe_tick, UniverseTick(100));
        assert_eq!(got.frame, planet());
    }

    #[test]
    fn round_trip_through_a_moving_rotated_frame_is_the_identity() {
        // A->B->A must return the original pose for ANY placement (the algebra is a
        // rigid transform and its inverse).
        let mut map = BTreeMap::new();
        map.insert(sys(), FramePlacement::identity());
        map.insert(
            planet(),
            FramePlacement {
                origin: DVec3::new(3.0, -7.0, 11.0),
                velocity: DVec3::new(0.5, -0.25, 2.0),
                orientation: DQuat::from_rotation_z(0.7) * DQuat::from_rotation_x(0.3),
                angular_velocity: DVec3::new(0.0, 0.0, 0.4),
            },
        );
        let ctx = StaticFrames(map);
        let original = pose_in(
            sys(),
            DVec3::new(13.0, 5.0, -2.0),
            DVec3::new(1.0, -2.0, 0.5),
        );
        let in_planet = transfer_frame(&original, planet(), &ctx).expect("to planet");
        let back = transfer_frame(&in_planet, sys(), &ctx).expect("back to system");
        assert!(
            (back.pos.offset() - original.pos.offset()).length() < 1e-9,
            "pos round-trips: {:?} vs {:?}",
            back.pos,
            original.pos
        );
        assert!(
            (back.vel - original.vel).length() < 1e-9,
            "vel round-trips (Coriolis cancels): {:?} vs {:?}",
            back.vel,
            original.vel
        );
    }

    #[test]
    fn rotation_only_frame_reorients_position_and_velocity() {
        // A frame rotated 90° about Z: a body at +X (1,0,0) in the parent reads as
        // +Y... actually inv(R_z(90))*(+X) = -Y. Verify the orientation composes.
        let mut map = BTreeMap::new();
        map.insert(sys(), FramePlacement::identity());
        map.insert(
            planet(),
            FramePlacement {
                origin: DVec3::ZERO,
                velocity: DVec3::ZERO,
                orientation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
                angular_velocity: DVec3::ZERO,
            },
        );
        let ctx = StaticFrames(map);
        let p = pose_in(sys(), DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO);
        let got = transfer_frame(&p, planet(), &ctx).expect("rotated");
        assert!(
            (got.pos.offset() - DVec3::new(0.0, -1.0, 0.0)).length() < 1e-9,
            "rotated into frame axes: {:?}",
            got.pos
        );
    }

    #[test]
    fn placement_constructors_and_det_hashmap_context_work() {
        // FrameContext can be backed by a DetHashMap too (deterministic ordering is
        // irrelevant for a point lookup, but the seam accepts any map).
        let mut map: DetHashMap<FrameRef, FramePlacement> = DetHashMap::default();
        map.insert(sys(), FramePlacement::identity());
        map.insert(
            planet(),
            FramePlacement::moving(DVec3::new(2.0, 0.0, 0.0), DVec3::ZERO),
        );
        struct DetCtx(DetHashMap<FrameRef, FramePlacement>);
        impl FrameContext for DetCtx {
            fn placement(&self, frame: FrameRef, _t: UniverseTick) -> Option<FramePlacement> {
                self.0.get(&frame).copied()
            }
        }
        let ctx = DetCtx(map);
        // A CROSS-frame transfer invokes the DetCtx placement lookup.
        let p = pose_in(sys(), DVec3::new(2.0, 0.0, 0.0), DVec3::ZERO);
        let got = transfer_frame(&p, planet(), &ctx).expect("cross-frame via DetCtx");
        assert!(got.pos.offset().length() < 1e-9, "at the planet origin");
        assert_eq!(FramePlacement::identity().origin, DVec3::ZERO);
        assert_eq!(
            FramePlacement::moving(DVec3::X, DVec3::Y).velocity,
            DVec3::Y
        );
    }

    #[test]
    fn frame_errors_display() {
        assert_eq!(
            FrameError::UnknownSourceFrame.to_string(),
            "no ephemeris placement for the source frame at this tick"
        );
        assert_eq!(
            FrameError::UnknownDestFrame.to_string(),
            "no ephemeris placement for the destination frame at this tick"
        );
    }
}
