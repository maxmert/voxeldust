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

use std::collections::BTreeMap;

use glam::{DQuat, DVec3, I64Vec3};

use crate::celestial::{OrbitalElements, orbital_state, secs_since_epoch};
use crate::ids::UniverseTick;
use crate::pose::{FrameRef, LatticePos, RealmId, StampedPose, frame_for_realm};

/// Where a frame's origin sits — and how it moves — relative to the COMMON PARENT at a
/// universe tick. A rigid placement: position, velocity, orientation, angular velocity.
/// All four come from the ephemeris closed-form; P1 uses the identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramePlacement {
    /// Integer CELL anchor of this frame's origin in the parent frame (parent tier units) — the
    /// exact-integer coarse part of the origin position, companion to `origin` (its sub-cell f64
    /// residual). Through P3 every placement is at `ZERO` (the ephemeris returns small system-local
    /// origins that ride entirely in `origin`); the field is planted so the S5 cross-cell re-base in
    /// [`transfer_frame`] carries the dest cell instead of the current cell-0 write. Byte-floor: with
    /// `origin_cell == ZERO` the dest pose is exactly `LatticePos::local(new_pos)` as before.
    pub origin_cell: I64Vec3,
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
            origin_cell: I64Vec3::ZERO,
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
            origin_cell: I64Vec3::ZERO,
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

    // 1. Lift the local pose into the common parent frame, reconstructing the FULL source-frame position
    // = source-tier cell metres + the f64 offset, so a non-zero source cell is NOT silently dropped (S0
    // of the floating-origin plan). Through P3 `cell == ZERO`, so this is exactly `offset()` —
    // byte-identical. The origin CELL anchors (`from.origin_cell`/`dest.origin_cell`, `ZERO` through P3)
    // and the DEST-side re-quantization of `new_pos` back into dest-tier cells (which would push the dest
    // pose off cell-0 and change the wire bytes) fold in WITH the P4/P5 re-centering — the dest write
    // stays cell-0 here to hold the byte-floor; galaxy ly-cells at P10. D-41 plant-item 2.
    let local = pose.pos.offset() + pose.pos.cell().as_dvec3() * pose.frame.tier().cell_edge_m();
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

/// A per-shard ephemeris [`FrameContext`] — the moving-frame successor to [`IdentityFrames`]
/// (D-45(a) frame-authority). Each shard is the SOLE AUTHOR of the placements of its DIRECT
/// CHILDREN, expressed in its OWN frame, and evaluates containment in that frame:
/// - **own** → [`FramePlacement::identity`]: this shard IS its own local origin; its boundary is a
///   static shell/box at the origin, never moving under it.
/// - a **moving direct child** → placed LIVE this tick from its orbital elements
///   (`origin`/`velocity` = [`orbital_state`]`(elements, secs_since_epoch(tick, tick_hz))`) — the
///   child's position IN THIS SHARD'S frame, which the shard AUTHORS.
/// - a **static direct child** or the **direct parent** → a stored [`FramePlacement`] (identity for a
///   walk-scale body whose position rides its region `center`; the RECEIVED parent placement
///   otherwise — parent authors it, this shard receives it).
/// - anything else (an ancestor ABOVE the direct parent — grandparent, root — a sibling, an
///   unrelated realm) → `None`.
///
/// **Why grandparent+ → `None` is correct (not a bug):** an entity re-homes to its DIRECT parent
/// the instant it leaves this realm, before it could reach a grandparent boundary; and the
/// root-seeded `container` fold resolves a member-of-nothing subject to the root. So dropping
/// grandparent MEMBERSHIP never changes the container DECISION — only the raw membership bitset,
/// which nothing observes. On the walk-scale forest every neighbourhood frame is stored at the
/// identity (children ride their center; the parent is static), so `LocalFrames` is byte-equivalent
/// to [`IdentityFrames`] for the decision — the proof the FA-1 seam swap gates.
///
/// Two shards at a SHARED boundary read DIFFERENT frames: the parent evaluates a child's moving SOI
/// at the child's AUTHORED position in the parent frame; the child evaluates its OWN boundary as a
/// static shell at the local origin. No cross-host bit-equality is ever invoked (author-and-ship) —
/// the ephemeris SPIKE-6a gate the replicated-by-seed oracle needed is retired.
#[derive(Clone, Debug)]
pub struct LocalFrames {
    own: FrameRef,
    moving: BTreeMap<FrameRef, OrbitalElements>,
    placed: BTreeMap<FrameRef, FramePlacement>,
    tick_hz: f64,
}

impl LocalFrames {
    /// A context for a shard whose realm frame is `own`, advancing at `tick_hz` (the per-shard clock
    /// the ephemeris samples). Register children/parent placements with the builder methods.
    #[must_use]
    pub fn new(own: FrameRef, tick_hz: f64) -> LocalFrames {
        LocalFrames {
            own,
            moving: BTreeMap::new(),
            placed: BTreeMap::new(),
            tick_hz,
        }
    }

    /// Register a MOVING direct child: its placement is derived live from `elements` each tick.
    #[must_use]
    pub fn with_moving_child(mut self, frame: FrameRef, elements: OrbitalElements) -> LocalFrames {
        self.moving.insert(frame, elements);
        self
    }

    /// Register a STATIC direct child or the DIRECT PARENT at a fixed placement (identity for a
    /// walk-scale body whose position rides its region `center`; the received parent placement
    /// otherwise).
    #[must_use]
    pub fn with_placed(mut self, frame: FrameRef, placement: FramePlacement) -> LocalFrames {
        self.placed.insert(frame, placement);
        self
    }
}

impl FrameContext for LocalFrames {
    fn placement(&self, frame: FrameRef, tick: UniverseTick) -> Option<FramePlacement> {
        if self.own == frame {
            return Some(FramePlacement::identity());
        }
        if let Some(elements) = self.moving.get(&frame) {
            let state = orbital_state(elements, secs_since_epoch(tick.0, self.tick_hz));
            return Some(FramePlacement::moving(state.position, state.velocity));
        }
        self.placed.get(&frame).copied()
    }
}

/// Rebind an authoritative pose into the frame of its destination realm — the ONE machinery (HR3) every
/// cross-realm hand-off uses to re-express a source-frame pose so the dest reads it in its OWN frame: the
/// durable crossing (`build_crossing`), the D-37 forward re-home (`build_rehome`), and the D-7 transient
/// batch (`emit_transient_batch`) all funnel through here. The caller supplies the frame context `ctx`: a
/// shard passes its LIVE ephemeris ([`LocalFrames`]) so a crossing INTO a MOVING realm REBASES the position
/// by the dest realm's live placement (`transfer_frame` does the rigid-body transform) — so the source and
/// the dest read the SAME containment and the crossing cannot flap. A context-free caller passes
/// [`IdentityFrames`] for a pure FRAME-field relabel (position UNCHANGED — the dest realm is static, or walk
/// scale where every placement is the identity, so the two are equivalent). Cell-0 today; the exact-integer
/// cross-cell rebase (D-41) lands with P4/P5 re-centering, NO caller reshape (frozen signature).
///
/// `to_parent` supplies the dest realm's PARENT provenance — the one field [`frame_for_realm`] needs to
/// build the lossy arm: an `Area` frame carries `{planet_seed, area_seed}`, so re-expressing a pose into
/// an `Area` requires its enclosing `Planet`. Every other realm kind is a one-field lift and ignores
/// `to_parent`. The caller threads it from the crossing carrier (`CrossingRequest.to_parent`), which the
/// SOURCE detector fills from the container region's `parent` — a deterministic worldgen fact. SAFE
/// DEGRADE: a dest realm with no nameable frame (an `Area` genuinely given without its planet parent) or a
/// transform error returns the SOURCE-frame pose UNCHANGED (the label lags) — NEVER a dropped hand-off.
pub fn rebind_pose_to_dest(
    pose: StampedPose,
    to_realm: RealmId,
    to_parent: Option<RealmId>,
    ctx: &impl FrameContext,
) -> StampedPose {
    frame_for_realm(to_realm, to_parent)
        .and_then(|dest_frame| transfer_frame(&pose, dest_frame, ctx).ok())
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
    fn transfer_preserves_a_nonzero_source_cell_in_the_lifted_position() {
        // S0: the lift reconstructs the FULL source position (cell millimetres + f64 offset), so a
        // non-zero source cell is NOT silently dropped. Under IdentityFrames a cross-frame transfer
        // re-expresses the frame while leaving the position unchanged — so the dest offset carries the
        // source cell's metres.
        let cell = I64Vec3::new(1_024_000, 0, 0); // 1_024_000 × 2⁻¹⁰ m/cell = 1000 m
        let pose = StampedPose {
            frame: sys(),
            pos: LatticePos::at(cell, DVec3::new(0.5, -0.25, 0.75)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(100),
        };
        let out = transfer_frame(&pose, planet(), &IdentityFrames).expect("identity transfer");
        let want = DVec3::new(1000.5, -0.25, 0.75);
        assert!((out.pos.offset() - want).length() < 1e-9);
        // The dest write stays cell-0 (byte-floor: re-quantizing new_pos into dest cells is P4/P5).
        assert_eq!(out.pos.cell(), I64Vec3::ZERO);
    }

    #[test]
    fn transfer_at_cell_zero_is_byte_identical_to_the_pre_lattice_lift() {
        // THE byte-floor: with a cell-0 source (the P3 shipping form) the lift is exactly `offset()`, so
        // the dest pose is unchanged from the pre-S0 behaviour — cell 0, offset = the transformed pos.
        let ctx = StaticFrames({
            let mut m = BTreeMap::new();
            m.insert(sys(), FramePlacement::identity());
            m.insert(
                planet(),
                FramePlacement::moving(DVec3::new(10.0, 0.0, 0.0), DVec3::ZERO),
            );
            m
        });
        let p = pose_in(sys(), DVec3::new(3.0, 4.0, 5.0), DVec3::ZERO);
        let out = transfer_frame(&p, planet(), &ctx).expect("transfer");
        assert_eq!(out.pos.cell(), I64Vec3::ZERO);
        // planet origin at +10x ⇒ dest offset = (3-10, 4, 5).
        assert!((out.pos.offset() - DVec3::new(-7.0, 4.0, 5.0)).length() < 1e-9);
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
        let got = rebind_pose_to_dest(p, RealmId::System(8), None, &IdentityFrames);
        assert_eq!(got.frame, FrameRef::SystemSpace { system_seed: 8 });
        assert_eq!(got.pos.offset(), p.pos.offset());
        assert_eq!(got.vel, p.vel);
        assert_eq!(got.orient, p.orient);
    }

    #[test]
    fn rebind_pose_to_dest_flips_into_an_area_frame_given_its_planet_parent() {
        // The lossy arm: an Area dest IS nameable once its enclosing planet is threaded as `to_parent`, so
        // the pose re-expresses into AreaLocal{planet_seed, area_seed} — the frame flips, position/velocity
        // unchanged under IdentityFrames. This is the Area-label fix: the parent provenance the crossing
        // now carries makes the district frame form (Station/System/Planet needed no parent; Area does).
        let from = FrameRef::PlanetCentered { planet_seed: 7 };
        let p = pose_in(from, DVec3::new(25.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0));
        let got = rebind_pose_to_dest(
            p,
            RealmId::Area(99),
            Some(RealmId::Planet(7)),
            &IdentityFrames,
        );
        assert_eq!(
            got.frame,
            FrameRef::AreaLocal {
                planet_seed: 7,
                area_seed: 99,
            }
        );
        assert_eq!(got.pos.offset(), p.pos.offset());
        assert_eq!(got.vel, p.vel);
    }

    #[test]
    fn rebind_pose_to_dest_safe_degrades_an_unnameable_dest_to_the_source_pose() {
        // The None fallback arm: an Area realm has no nameable frame WITHOUT its planet parent, so
        // `frame_for_realm` is None and the pose is returned UNCHANGED — the hand-off is never dropped,
        // only the frame label lags. This exercises `unwrap_or(pose)` (a caller precondition failure —
        // the detector always supplies the parent for an Area dest today).
        let from = FrameRef::SystemSpace { system_seed: 7 };
        let p = pose_in(from, DVec3::new(47.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0));
        let got = rebind_pose_to_dest(p, RealmId::Area(99), None, &IdentityFrames);
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
                origin_cell: I64Vec3::ZERO,
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
                origin_cell: I64Vec3::ZERO,
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

    // ---- D-45(a) frame-authority FA-0: LocalFrames ----------------------------------

    fn test_elements() -> OrbitalElements {
        OrbitalElements {
            sma: 1.0e6,
            ecc: 0.0,
            inclination: 0.0,
            raan: 0.0,
            arg_periapsis: 0.0,
            mean_anomaly_epoch: 0.0,
            central_mass: 1.989e30,
        }
    }

    #[test]
    fn local_frames_own_is_the_identity() {
        // A shard IS its own local origin: its boundary is static at the origin.
        let ctx = LocalFrames::new(sys(), 20.0);
        assert_eq!(
            ctx.placement(sys(), UniverseTick(5)),
            Some(FramePlacement::identity())
        );
    }

    #[test]
    fn local_frames_moving_child_tracks_the_authored_orbit() {
        let elem = test_elements();
        let ctx = LocalFrames::new(sys(), 20.0).with_moving_child(planet(), elem);
        // The child's placement equals the shard's AUTHORED orbital state at that tick.
        for t in [0u64, 1, 137, 5000] {
            let state = orbital_state(&elem, secs_since_epoch(t, 20.0));
            assert_eq!(
                ctx.placement(planet(), UniverseTick(t)),
                Some(FramePlacement::moving(state.position, state.velocity)),
            );
        }
        // Tick-dependent: the planet is at a different place a tick later (it orbits).
        assert_ne!(
            ctx.placement(planet(), UniverseTick(0)),
            ctx.placement(planet(), UniverseTick(1)),
        );
    }

    #[test]
    fn local_frames_placed_returns_the_stored_placement() {
        // A static child / the received parent placement.
        let station = FrameRef::StationLocal { station_seed: 3 };
        let p = FramePlacement::moving(DVec3::new(4.0, 0.0, 0.0), DVec3::ZERO);
        let ctx = LocalFrames::new(sys(), 20.0).with_placed(station, p);
        assert_eq!(ctx.placement(station, UniverseTick(9)), Some(p));
    }

    #[test]
    fn local_frames_unknown_frame_is_none() {
        // A grandparent (above the direct parent), a sibling, or an unrelated frame -> None
        // (non-member by design; the root-seeded container fold preserves the decision).
        let ctx = LocalFrames::new(sys(), 20.0);
        assert_eq!(ctx.placement(planet(), UniverseTick(1)), None);
    }

    // ---- rehome COORDINATE correctness (successful rehomes + expected coords, incl. floating-point) ----

    #[test]
    fn rehome_system_to_a_moving_planet_lands_at_the_correct_planet_local_coordinate() {
        // A player rehoming System→Planet: rebind_pose_to_dest re-expresses their System pose into the planet
        // frame, REBASED by the planet's LIVE orbital position (LocalFrames derives it from the ephemeris). A
        // player sitting AT the planet centre lands at the planet-local ORIGIN; one offset by delta lands at
        // delta. This is the exact coordinate the dest planet shard must read (author-and-ship, same
        // containment both sides).
        let elem = test_elements();
        let tick = UniverseTick(137);
        let planet_pos = orbital_state(&elem, secs_since_epoch(tick.0, 20.0)).position;
        let ctx = LocalFrames::new(sys(), 20.0).with_moving_child(planet(), elem);
        // AT the planet centre ⇒ planet-local origin.
        let at_centre = StampedPose::at_rest(sys(), planet_pos, tick);
        let landed = rebind_pose_to_dest(at_centre, RealmId::Planet(2), None, &ctx);
        assert_eq!(landed.frame, planet());
        assert!(landed.pos.offset().length() < 1e-6);
        // Offset by delta ⇒ planet-local delta.
        let delta = DVec3::new(10.0, -5.0, 2.0);
        let off = StampedPose::at_rest(sys(), planet_pos + delta, tick);
        let landed_off = rebind_pose_to_dest(off, RealmId::Planet(2), None, &ctx);
        assert!((landed_off.pos.offset() - delta).length() < 1e-6);
    }

    #[test]
    fn a_system_planet_system_round_trip_returns_the_original_world_position() {
        // Rehome IN then OUT: System→Planet then Planet→System must return the ORIGINAL System pose (the frame
        // transform and its inverse compose to the identity) — a player who steps onto a planet and back off
        // is exactly where they started, no drift.
        let elem = test_elements();
        let tick = UniverseTick(42);
        let ctx = LocalFrames::new(sys(), 20.0).with_moving_child(planet(), elem);
        let start = StampedPose::at_rest(sys(), DVec3::new(1.0e6, 2.0e5, -3.0e5), tick);
        let on_planet = rebind_pose_to_dest(start, RealmId::Planet(2), None, &ctx);
        assert_eq!(on_planet.frame, planet());
        let back = rebind_pose_to_dest(on_planet, RealmId::System(1), None, &ctx);
        assert_eq!(back.frame, sys());
        assert!((back.pos.offset() - start.pos.offset()).length() < 1e-6);
    }

    #[test]
    fn a_rehome_is_invariant_to_how_the_source_position_splits_across_cell_and_offset() {
        // Floating-point robustness: the SAME physical System position, expressed either as a cell-0 full
        // offset (today's shipping form) OR as an integer-millimetre CELL + sub-mm residual (the S5 form),
        // rehomes to the SAME planet-local coordinate — the transfer reconstructs the full source position
        // from cell+offset (S0), so how the position is split is invisible to the destination coordinate.
        let elem = test_elements();
        let tick = UniverseTick(7);
        let planet_pos = orbital_state(&elem, secs_since_epoch(tick.0, 20.0)).position;
        let ctx = LocalFrames::new(sys(), 20.0).with_moving_child(planet(), elem);
        let world = planet_pos + DVec3::new(7.0, -3.0, 11.0);
        // Rep 1: cell-0 full offset (the P3 shipping form).
        let flat = StampedPose::at_rest(sys(), world, tick);
        // Rep 2: normalized into integer-mm cells + residual (the S5 form) — the SAME physical position.
        let mut split = flat;
        split.pos = LatticePos::local(world).normalize(sys().tier());
        let a = rebind_pose_to_dest(flat, RealmId::Planet(2), None, &ctx);
        let b = rebind_pose_to_dest(split, RealmId::Planet(2), None, &ctx);
        // Both land at the SAME planet-local coordinate (≈ the 7,-3,11 offset from the centre).
        assert!((a.pos.offset() - b.pos.offset()).length() < 1e-6);
        assert!((a.pos.offset() - DVec3::new(7.0, -3.0, 11.0)).length() < 1e-6);
    }
}
