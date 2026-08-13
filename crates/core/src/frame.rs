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
    /// origins that ride entirely in `origin`), so the two anchors cancel and the dest pose is exactly
    /// `LatticePos::local(new_pos)` as it was — the byte-floor.
    ///
    /// [`transfer_frame`] now SUBTRACTS the two anchors as integers and carries the difference into the
    /// dest pose. It used to read neither field and write cell `0` unconditionally, which quietly
    /// destroyed the anchor on every hop: a pose one anchor-block out came back as ~9.8×10⁸ f64 metres
    /// with the integer truth gone, and a multi-level ladder compounds that loss once per hop. The whole
    /// point of the tiered coordinate is that a planet's surface deals in millimetres however far the
    /// planet sits from anything else, and that only holds if the big part stays an integer.
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
    /// The two frames' origins sit in DIFFERENT integer cells and the destination frame is rotated
    /// relative to the common parent. The cell difference is a count of parent-axis cells; turning it
    /// into the destination's axes is a rotation, and a rotated cell COUNT is not a cell count. Folding
    /// it into f64 metres instead would be the exact loss this whole conversion exists to remove, so it
    /// is refused LOUD rather than silently degraded. Unreachable in production (every placement in the
    /// tree is identity-oriented, and every anchor is `ZERO`); it becomes live only when a spinning
    /// realm is authored more than one cell-block from its parent's origin.
    #[error("the destination frame is rotated and its origin sits in a different integer cell")]
    RotatedFrameAcrossCells,
}

/// Re-express `pose` (in its current frame) into `to` using the ephemeris `ctx`. The
/// closed-form rigid-body transform: lift the local pose into the common parent, then
/// into the destination frame, carrying velocity (with the Coriolis term for rotating
/// frames) and orientation.
///
/// Same-frame transfer is the identity (and needs no context). The `universe_tick` is
/// preserved: a pose is meaningful only with the instant it was stamped at.
///
/// The two origins' INTEGER cell anchors subtract as integers and the difference rides through to the
/// destination pose — the big part of a position never passes through f64, which is what keeps a
/// planet's surface millimetre-exact however far the planet sits from its star.
///
/// # Errors
/// [`FrameError`] if either frame has no placement at the pose's tick, or if the two origins sit in
/// different integer cells while the destination frame is rotated (an integer cell count cannot be
/// rotated into another frame's axes — see [`FrameError::RotatedFrameAcrossCells`]).
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

    // 1. THE LEVER — where the pose sits inside its OWN realm, reconstructing the FULL source-frame
    // position (source-tier cell metres + the f64 offset) so a non-zero source cell is not silently
    // dropped (S0 of the floating-origin plan). Through P3 `cell == ZERO`, so this is exactly
    // `offset()` — byte-identical. This vector is bounded by the SOURCE realm's own extent, so rotating
    // it in f64 metres is exact to far under a micron. The huge magnitudes are NOT here: they are the
    // two origins' integer anchors, and step 3 keeps those integral.
    let local = pose.pos.offset() + pose.pos.cell().as_dvec3() * pose.frame.tier().cell_edge_m();
    let lever = from.orientation * local;
    let world_pos = from.origin + lever;
    let world_vel =
        from.velocity + from.orientation * pose.vel + from.angular_velocity.cross(lever);
    let world_orient = from.orientation * pose.orient;

    // 2. Express the parent-frame pose in the destination frame. Every term here is a small residual:
    // the two origins' sub-cell remainders and the lever, never a universe-scale magnitude.
    let inv = dest.orientation.inverse();
    let rel = world_pos - dest.origin;
    let new_pos = inv * rel;
    let new_vel = inv * (world_vel - dest.velocity - dest.angular_velocity.cross(rel));
    let new_orient = inv * world_orient;

    // 3. THE INTEGER HALF — subtract the two origins' cell anchors AS INTEGERS. This line used to not
    // exist: the dest pose was written with `LatticePos::local`, which pins the cell to ZERO, so every
    // hop threw the anchor away and left the f64 offset carrying the whole distance. Measured on the
    // pre-change code: a pose one anchor-block (10¹² cells ≈ 9.77×10⁸ m) out came back as cell 0 with
    // offset 976_562_645.5 m — the integer truth gone and f64 left holding a nine-digit number, which
    // is precisely the precision cliff the tiered coordinate exists to avoid. Saturating per component
    // for the same reason `LatticePos::normalize` saturates: a hostile or diverged anchor must not
    // PANIC the shard in debug (i.e. across the whole test + coverage suite), and every in-domain
    // anchor is far inside the range so the clamp never fires on a real pose.
    let cell_out = I64Vec3::new(
        from.origin_cell.x.saturating_sub(dest.origin_cell.x),
        from.origin_cell.y.saturating_sub(dest.origin_cell.y),
        from.origin_cell.z.saturating_sub(dest.origin_cell.z),
    );
    // A cell COUNT is measured along the PARENT's axes. If the destination is rotated relative to the
    // parent, expressing that count in the destination's axes is a rotation, and no rotation of a
    // non-zero integer cell count is itself an integer cell count. Refuse it — the alternative is to
    // fold the anchor into f64 metres, which is the loss this function was just fixed to stop doing.
    // EXACT quaternion equality, never an epsilon: a frame that is "nearly" unrotated is rotated, and a
    // tolerance here would silently resume the folding for every slowly-spinning realm.
    if dest.orientation != DQuat::IDENTITY && cell_out != I64Vec3::ZERO {
        return Err(FrameError::RotatedFrameAcrossCells);
    }

    Ok(StampedPose {
        frame: to,
        // A tier change (FINE millimetre cells ↔ COARSE light-year cells at P10) routes through the ONE
        // existing re-quantizer rather than a second copy of that arithmetic. Same tier — every live
        // frame pair today, since only `GalaxySpace` is COARSE and no COARSE pose exists yet — returns
        // `self` bit-for-bit, so this is the exact identity on the shipped path. When the galaxy tier
        // lights up at P10, the unit `origin_cell` is counted in (the PARENT's, which this signature
        // cannot name) has to be settled in the same change that settles `convert_tier`'s own
        // P10-deferred FINE↔COARSE remainder carry; the two are the same open question.
        pos: LatticePos::at(cell_out, new_pos).convert_tier(pose.frame.tier(), to.tier()),
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
/// scale where every placement is the identity, so the two are equivalent). The exact-integer cross-cell
/// rebase (D-41) is live inside [`transfer_frame`] and needed no caller reshape (frozen signature); every
/// placement in the tree is still anchored at cell `ZERO`, so it is inert here today.
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
    use crate::pose::FINE_CELL_EDGE_M;
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
        assert_eq!(
            FrameError::RotatedFrameAcrossCells.to_string(),
            "the destination frame is rotated and its origin sits in a different integer cell"
        );
    }

    /// One anchor block: 10¹² FINE cells × 2⁻¹⁰ m = 976_562_500 m exactly (no rounding — the FINE edge
    /// is a power of two, so the product is exact in f64 and the test's literals are exact too).
    const ANCHOR_CELLS: i64 = 1_000_000_000_000;
    const ANCHOR_M: f64 = 976_562_500.0;

    #[test]
    fn a_transfer_preserves_the_integer_cell_anchor() {
        // THE defect this slice removes. A planet's origin sits one anchor block plus 145 m out inside
        // its star system. The transfer used to write `LatticePos::local(new_pos)`, which pins the cell
        // to ZERO unconditionally, so the anchor was destroyed on every hop and f64 was left holding the
        // whole nine-digit distance. MEASURED on the pre-change code: cell 0, offset 976_562_645.5 m.
        // Now the two anchors subtract as integers and the difference rides through to the dest pose.
        let anchor = I64Vec3::new(ANCHOR_CELLS, 0, 0);
        let mut map = BTreeMap::new();
        // The system IS the common parent here, so its own placement is the identity.
        map.insert(sys(), FramePlacement::identity());
        map.insert(
            planet(),
            FramePlacement {
                origin_cell: anchor,
                origin: DVec3::new(145.0, 0.0, 0.0),
                velocity: DVec3::ZERO,
                orientation: DQuat::IDENTITY,
                angular_velocity: DVec3::ZERO,
            },
        );
        let ctx = StaticFrames(map);
        // A dot standing half a metre from the planet's centre — and itself anchored a block out, to
        // pin down what happens to the POSE's own cell as well as the placement's.
        let pose = StampedPose {
            frame: planet(),
            pos: LatticePos::at(anchor, DVec3::new(0.5, 0.0, 0.0)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(100),
        };
        let got = transfer_frame(&pose, sys(), &ctx).expect("planet -> system");
        assert_eq!(
            got.pos.cell(),
            anchor,
            "the placement's integer anchor SURVIVES the hop (it used to be written as ZERO)"
        );
        // The pose's OWN cell still rides the f64 lever, by design: a lever is bounded by the source
        // realm's own extent, and the conversion is only exact-integer in the two ORIGINS. So the
        // offset is unchanged from the pre-change measurement — that number is the byte floor, not a
        // regression. If poses ever start shipping normalized (all magnitude in the cell) this is the
        // line that has to change, and this assertion is what will say so.
        assert_eq!(got.pos.offset().x, ANCHOR_M + 145.0 + 0.5);
        assert_eq!(got.pos.offset().x, 976_562_645.5);
        assert_eq!(got.frame, sys());

        // GOING BACK DOWN, the anchors subtract the other way and the round trip is EXACT — the same
        // physical point, to the bit, with no accumulated f64 drift. (The split between cell and offset
        // differs on the way back only because the pose's own cell was folded into the lever on the way
        // out; the total is identical, which is what a position means.)
        let back = transfer_frame(&got, planet(), &ctx).expect("system -> planet");
        assert_eq!(back.pos.cell(), I64Vec3::new(-ANCHOR_CELLS, 0, 0));
        let total_out = |p: LatticePos| p.offset().x + p.cell().as_dvec3().x * FINE_CELL_EDGE_M;
        assert_eq!(total_out(back.pos), total_out(pose.pos));
        assert_eq!(total_out(back.pos), ANCHOR_M + 0.5);
    }

    #[test]
    fn a_rotated_destination_refuses_a_cross_cell_transfer_but_allows_a_same_cell_one() {
        // A cell count is measured along the PARENT's axes. Rotating a non-zero count into the
        // destination's axes does not yield a count, so it is refused LOUD rather than folded into f64
        // metres — folding is exactly the loss the slice removes. Unreachable in production today
        // (every placement is identity-oriented and anchored at ZERO); covered here because HR5 wants
        // both sides of the guard, and because a spinning realm authored a block out is a real P8 shape.
        let spun = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let p = pose_in(sys(), DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO);

        // Rotated dest, anchors in DIFFERENT cells -> typed refusal.
        let mut map = BTreeMap::new();
        map.insert(sys(), FramePlacement::identity());
        map.insert(
            planet(),
            FramePlacement {
                origin_cell: I64Vec3::new(ANCHOR_CELLS, 0, 0),
                origin: DVec3::ZERO,
                velocity: DVec3::ZERO,
                orientation: spun,
                angular_velocity: DVec3::ZERO,
            },
        );
        assert_eq!(
            transfer_frame(&p, planet(), &StaticFrames(map)),
            Err(FrameError::RotatedFrameAcrossCells)
        );

        // Rotated dest, anchors in the SAME cell -> the count is zero, nothing needs rotating, allowed.
        let mut map = BTreeMap::new();
        map.insert(sys(), FramePlacement::identity());
        map.insert(
            planet(),
            FramePlacement {
                origin_cell: I64Vec3::ZERO,
                origin: DVec3::ZERO,
                velocity: DVec3::ZERO,
                orientation: spun,
                angular_velocity: DVec3::ZERO,
            },
        );
        let got = transfer_frame(&p, planet(), &StaticFrames(map)).expect("same-cell rotated dest");
        assert_eq!(got.pos.cell(), I64Vec3::ZERO);
        assert!((got.pos.offset() - DVec3::new(0.0, -1.0, 0.0)).length() < 1e-9);
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
