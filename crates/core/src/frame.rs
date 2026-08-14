//! Cross-frame pose re-expression — `transfer_frame` (`docs/design/transfer_protocol.md`
//! §6; audit XFRAME-1). The single most load-bearing piece for every real transition:
//! a player crossing from SystemSpace into a planet's PlanetCentered frame, a boarding
//! pod entering a ship's ShipLocal frame, a debris chunk falling planet-ward.
//!
//! Category-A and PURE: the transform is a closed-form rigid-body composition given the
//! frames' placements. The placements arrive as a [`PlacementBook`] — one anchor's rows
//! at ONE named instant, authored by the one physics writer — so this conversion CANNOT
//! ask how anything moves (SL4): it has no clock to hand an ephemeris and no ephemeris
//! to hand it to. It used to take a `FrameContext` trait whose implementor ran the
//! Kepler solve inside the lookup on a tick it was handed; that trait is deleted — the
//! injection seam it provided is exactly where a solver could hide.
//!
//! Authority never depends on cross-binary float reproducibility — the SOURCE shard
//! computes the destination pose and ships it, the dest sanity-bounds it against the
//! orchestrator's Ephemeris Authority.

use glam::{DQuat, DVec3, I64Vec3};

use crate::ids::UniverseTick;
use crate::placement::PlacementBook;
use crate::pose::{FrameRef, LatticePos, StampedPose};

/// Where a frame's origin sits — and how it moves — relative to the COMMON PARENT at a
/// universe tick. A rigid placement: position, velocity, orientation, angular velocity.
/// All four are authored by the parent's physics writer; P1 uses the identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramePlacement {
    /// Integer CELL anchor of this frame's origin in the parent frame (parent tier units) — the
    /// exact-integer coarse part of the origin position, companion to `origin` (its sub-cell f64
    /// residual). Through P3 every placement is at `ZERO` (the authored rows carry small system-local
    /// origins that ride entirely in `origin`), so the two anchors cancel and the dest pose is exactly
    /// `LatticePos::local(new_pos)` as it was — the byte-floor.
    ///
    /// [`transfer_frame`] SUBTRACTS the two anchors as integers and carries the difference into the
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

/// Why a frame re-expression could not be computed (a typed precondition failure, never
/// a silent garbage spawn — the R6 km-scale-error class made loud).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    #[error("no authored placement for the source frame in this book")]
    UnknownSourceFrame,
    #[error("no authored placement for the destination frame in this book")]
    UnknownDestFrame,
    /// The book speaks at one instant and the pose is stamped at another. A book's rows are true at
    /// exactly its own instant, so converting a differently-stamped pose through them would move the
    /// pose by however far the world swept between the two — silently. Refused LOUD instead, which
    /// makes every consumer's book SELECTION explicit, auditable and testable: a consumer that means
    /// "now" re-stamps its pose to the head book's instant and says so; a consumer that means "then"
    /// selects the book at the pose's own stamp.
    #[error("the placement book speaks at tick {} but the pose is stamped at tick {}", book_at.0, pose_at.0)]
    InstantMismatch {
        book_at: UniverseTick,
        pose_at: UniverseTick,
    },
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

/// Re-express `pose` (in its current frame) into `to` using the authored placement `book`. The
/// closed-form rigid-body transform: lift the local pose into the common parent (the book's anchor),
/// then into the destination frame, carrying velocity (with the Coriolis term for rotating frames)
/// and orientation.
///
/// Same-frame transfer is the identity (and reads no book). The `universe_tick` is preserved: a pose
/// is meaningful only with the instant it was stamped at — and the book must speak at that SAME
/// instant ([`FrameError::InstantMismatch`]), so a conversion can never mix two times.
///
/// MONOMORPHIC, deliberately: the deleted generic-over-`FrameContext` form needed a branchless-shim
/// split (`transfer_frame_resolved`) so its branches would not multiply per monomorphization (HR5).
/// With one concrete parameter type every branch lives here, covered once — and there is no seam left
/// where an implementor could hand the crossing path a clock-reading context.
///
/// The two origins' INTEGER cell anchors subtract as integers and the difference rides through to the
/// destination pose — the big part of a position never passes through f64, which is what keeps a
/// planet's surface millimetre-exact however far the planet sits from its star.
///
/// # Errors
/// [`FrameError`] if the book speaks at a different instant than the pose's stamp, if either frame
/// has no placement in the book, or if the two origins sit in different integer cells while the
/// destination frame is rotated (an integer cell count cannot be rotated into another frame's axes —
/// see [`FrameError::RotatedFrameAcrossCells`]).
pub fn transfer_frame(
    pose: &StampedPose,
    to: FrameRef,
    book: &PlacementBook,
) -> Result<StampedPose, FrameError> {
    if pose.frame == to {
        return Ok(*pose);
    }
    if book.at() != pose.universe_tick {
        return Err(FrameError::InstantMismatch {
            book_at: book.at(),
            pose_at: pose.universe_tick,
        });
    }
    let from = book.of(pose.frame).ok_or(FrameError::UnknownSourceFrame)?;
    let dest = book.of(to).ok_or(FrameError::UnknownDestFrame)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::FINE_CELL_EDGE_M;

    fn sys() -> FrameRef {
        FrameRef::SystemSpace { system_seed: 1 }
    }
    fn planet() -> FrameRef {
        FrameRef::PlanetCentered { planet_seed: 2 }
    }

    /// A book anchored on the SYSTEM at tick 100 (the tick every test pose is stamped at), holding
    /// one row per given child placement.
    fn book(rows: Vec<(FrameRef, FramePlacement)>) -> PlacementBook {
        PlacementBook::new(sys(), UniverseTick(100), rows)
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
    fn same_frame_is_the_identity_and_reads_no_book() {
        // An empty book — at a DIFFERENT instant, even — still returns the pose unchanged: the
        // same-frame short-circuit precedes the instant check because no row is read at all.
        let b = PlacementBook::new(sys(), UniverseTick(9), Vec::new());
        let p = pose_in(sys(), DVec3::new(1.0, 2.0, 3.0), DVec3::new(0.1, 0.0, 0.0));
        assert_eq!(transfer_frame(&p, sys(), &b), Ok(p));
    }

    #[test]
    fn a_book_at_another_instant_is_refused_loud() {
        // The instant discipline: a consumer that wants "now" must re-stamp its pose to the head
        // book's instant, and a consumer that wants "then" must select the book at the pose's stamp —
        // mixing the two is never silent.
        let b = PlacementBook::new(
            sys(),
            UniverseTick(101),
            vec![(planet(), FramePlacement::identity())],
        );
        let p = pose_in(sys(), DVec3::ZERO, DVec3::ZERO);
        assert_eq!(
            transfer_frame(&p, planet(), &b),
            Err(FrameError::InstantMismatch {
                book_at: UniverseTick(101),
                pose_at: UniverseTick(100),
            })
        );
    }

    #[test]
    fn transfer_preserves_a_nonzero_source_cell_in_the_lifted_position() {
        // S0: the lift reconstructs the FULL source position (cell millimetres + f64 offset), so a
        // non-zero source cell is NOT silently dropped. With the dest row at the identity a
        // cross-frame transfer re-expresses the frame while leaving the position unchanged — so the
        // dest offset carries the source cell's metres.
        let cell = I64Vec3::new(1_024_000, 0, 0); // 1_024_000 × 2⁻¹⁰ m/cell = 1000 m
        let pose = StampedPose {
            frame: sys(),
            pos: LatticePos::at(cell, DVec3::new(0.5, -0.25, 0.75)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(100),
        };
        let b = book(vec![(planet(), FramePlacement::identity())]);
        let out = transfer_frame(&pose, planet(), &b).expect("identity transfer");
        let want = DVec3::new(1000.5, -0.25, 0.75);
        assert!((out.pos.offset() - want).length() < 1e-9);
        // The dest write stays cell-0 (byte-floor: re-quantizing new_pos into dest cells is P4/P5).
        assert_eq!(out.pos.cell(), I64Vec3::ZERO);
    }

    #[test]
    fn transfer_at_cell_zero_is_byte_identical_to_the_pre_lattice_lift() {
        // THE byte-floor: with a cell-0 source (the P3 shipping form) the lift is exactly `offset()`, so
        // the dest pose is unchanged from the pre-S0 behaviour — cell 0, offset = the transformed pos.
        let b = book(vec![(
            planet(),
            FramePlacement::moving(DVec3::new(10.0, 0.0, 0.0), DVec3::ZERO),
        )]);
        let p = pose_in(sys(), DVec3::new(3.0, 4.0, 5.0), DVec3::ZERO);
        let out = transfer_frame(&p, planet(), &b).expect("transfer");
        assert_eq!(out.pos.cell(), I64Vec3::ZERO);
        // planet origin at +10x ⇒ dest offset = (3-10, 4, 5).
        assert!((out.pos.offset() - DVec3::new(-7.0, 4.0, 5.0)).length() < 1e-9);
    }

    #[test]
    fn unknown_frames_are_typed_errors_not_garbage() {
        // A pose in a frame the book has no row for (and which is not its anchor): the source is
        // unknowable. A book anchored on the pose's own frame with no row for the dest: the dest is.
        let foreign = PlacementBook::new(planet(), UniverseTick(100), Vec::new());
        let p = pose_in(sys(), DVec3::ZERO, DVec3::ZERO);
        assert_eq!(
            transfer_frame(&p, planet(), &foreign),
            Err(FrameError::UnknownSourceFrame)
        );
        let empty = book(Vec::new());
        assert_eq!(
            transfer_frame(&p, planet(), &empty),
            Err(FrameError::UnknownDestFrame)
        );
    }

    #[test]
    fn an_identity_row_reframes_without_moving_the_pose() {
        // The P1-P3 crossing shape: re-express a pose from the anchor into a child whose authored row
        // is the identity — the FRAME field flips but position/velocity/orientation are UNCHANGED.
        let b = book(vec![(planet(), FramePlacement::identity())]);
        let p = pose_in(sys(), DVec3::new(47.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0));
        let got = transfer_frame(&p, planet(), &b).expect("identity reframe");
        assert_eq!(got.frame, planet());
        assert_eq!(got.pos.offset(), p.pos.offset());
        assert_eq!(got.vel, p.vel);
        assert_eq!(got.orient, p.orient);
        assert_eq!(got.universe_tick, p.universe_tick);
    }

    #[test]
    fn translated_moving_frame_subtracts_origin_and_velocity() {
        // The planet's origin sits at (1000, 0, 0) in SystemSpace, moving +Y at 5 m/s.
        // A body at rest in SystemSpace at (1000, 0, 0) is at the planet ORIGIN, and in
        // the planet frame moves -Y at 5 m/s (the planet moves out from under it).
        let b = book(vec![(
            planet(),
            FramePlacement::moving(DVec3::new(1000.0, 0.0, 0.0), DVec3::new(0.0, 5.0, 0.0)),
        )]);
        let p = pose_in(sys(), DVec3::new(1000.0, 0.0, 0.0), DVec3::ZERO);
        let got = transfer_frame(&p, planet(), &b).expect("transformed");
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
        // rigid transform and its inverse) — through the SAME book both ways.
        let b = book(vec![(
            planet(),
            FramePlacement {
                origin_cell: I64Vec3::ZERO,
                origin: DVec3::new(3.0, -7.0, 11.0),
                velocity: DVec3::new(0.5, -0.25, 2.0),
                orientation: DQuat::from_rotation_z(0.7) * DQuat::from_rotation_x(0.3),
                angular_velocity: DVec3::new(0.0, 0.0, 0.4),
            },
        )]);
        let original = pose_in(
            sys(),
            DVec3::new(13.0, 5.0, -2.0),
            DVec3::new(1.0, -2.0, 0.5),
        );
        let in_planet = transfer_frame(&original, planet(), &b).expect("to planet");
        let back = transfer_frame(&in_planet, sys(), &b).expect("back to system");
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
        let b = book(vec![(
            planet(),
            FramePlacement {
                origin_cell: I64Vec3::ZERO,
                origin: DVec3::ZERO,
                velocity: DVec3::ZERO,
                orientation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
                angular_velocity: DVec3::ZERO,
            },
        )]);
        let p = pose_in(sys(), DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO);
        let got = transfer_frame(&p, planet(), &b).expect("rotated");
        assert!(
            (got.pos.offset() - DVec3::new(0.0, -1.0, 0.0)).length() < 1e-9,
            "rotated into frame axes: {:?}",
            got.pos
        );
    }

    #[test]
    fn placement_constructors_hold_their_fields() {
        assert_eq!(FramePlacement::identity().origin, DVec3::ZERO);
        assert_eq!(FramePlacement::identity().orientation, DQuat::IDENTITY);
        assert_eq!(
            FramePlacement::moving(DVec3::X, DVec3::Y).velocity,
            DVec3::Y
        );
        assert_eq!(FramePlacement::moving(DVec3::X, DVec3::Y).origin, DVec3::X);
    }

    #[test]
    fn frame_errors_display() {
        assert_eq!(
            FrameError::UnknownSourceFrame.to_string(),
            "no authored placement for the source frame in this book"
        );
        assert_eq!(
            FrameError::UnknownDestFrame.to_string(),
            "no authored placement for the destination frame in this book"
        );
        assert_eq!(
            FrameError::InstantMismatch {
                book_at: UniverseTick(7),
                pose_at: UniverseTick(5),
            }
            .to_string(),
            "the placement book speaks at tick 7 but the pose is stamped at tick 5"
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
        // THE defect the floating-origin slice removed. A planet's origin sits one anchor block plus
        // 145 m out inside its star system. The transfer used to write `LatticePos::local(new_pos)`,
        // which pins the cell to ZERO unconditionally, so the anchor was destroyed on every hop and
        // f64 was left holding the whole nine-digit distance. MEASURED on the pre-change code: cell 0,
        // offset 976_562_645.5 m. Now the two anchors subtract as integers and the difference rides
        // through to the dest pose.
        let anchor = I64Vec3::new(ANCHOR_CELLS, 0, 0);
        let b = book(vec![(
            planet(),
            FramePlacement {
                origin_cell: anchor,
                origin: DVec3::new(145.0, 0.0, 0.0),
                velocity: DVec3::ZERO,
                orientation: DQuat::IDENTITY,
                angular_velocity: DVec3::ZERO,
            },
        )]);
        // A dot standing half a metre from the planet's centre — and itself anchored a block out, to
        // pin down what happens to the POSE's own cell as well as the placement's.
        let pose = StampedPose {
            frame: planet(),
            pos: LatticePos::at(anchor, DVec3::new(0.5, 0.0, 0.0)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(100),
        };
        let got = transfer_frame(&pose, sys(), &b).expect("planet -> system");
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
        let back = transfer_frame(&got, planet(), &b).expect("system -> planet");
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
        let b = book(vec![(
            planet(),
            FramePlacement {
                origin_cell: I64Vec3::new(ANCHOR_CELLS, 0, 0),
                origin: DVec3::ZERO,
                velocity: DVec3::ZERO,
                orientation: spun,
                angular_velocity: DVec3::ZERO,
            },
        )]);
        assert_eq!(
            transfer_frame(&p, planet(), &b),
            Err(FrameError::RotatedFrameAcrossCells)
        );

        // Rotated dest, anchors in the SAME cell -> the count is zero, nothing needs rotating, allowed.
        let b = book(vec![(
            planet(),
            FramePlacement {
                origin_cell: I64Vec3::ZERO,
                origin: DVec3::ZERO,
                velocity: DVec3::ZERO,
                orientation: spun,
                angular_velocity: DVec3::ZERO,
            },
        )]);
        let got = transfer_frame(&p, planet(), &b).expect("same-cell rotated dest");
        assert_eq!(got.pos.cell(), I64Vec3::ZERO);
        assert!((got.pos.offset() - DVec3::new(0.0, -1.0, 0.0)).length() < 1e-9);
    }

    // ---- rehome COORDINATE correctness (successful rehomes + expected coords, incl. floating-point) ----

    #[test]
    fn rehome_system_to_a_moving_planet_lands_at_the_correct_planet_local_coordinate() {
        // A player rehoming System→Planet: `transfer_frame` re-expresses their System pose into the
        // planet frame, REBASED by the planet's authored placement at this instant. A player sitting
        // AT the planet centre lands at the planet-local ORIGIN; one offset by delta lands at delta.
        // This is the exact coordinate the dest planet shard must read (author-and-ship, same
        // containment both sides). The placement is a hand-authored row — where it came from (an
        // orbit, a thruster, a hand) is exactly what this conversion must not be able to ask.
        let planet_at = DVec3::new(83.0, 0.0, -12.5);
        let b = book(vec![(
            planet(),
            FramePlacement::moving(planet_at, DVec3::new(0.0, 4.2, 0.0)),
        )]);
        // AT the planet centre ⇒ planet-local origin.
        let at_centre = StampedPose::at_rest(sys(), planet_at, UniverseTick(100));
        let landed = transfer_frame(&at_centre, planet(), &b).expect("placed");
        assert_eq!(landed.frame, planet());
        assert!(landed.pos.offset().length() < 1e-6);
        // Offset by delta ⇒ planet-local delta.
        let delta = DVec3::new(10.0, -5.0, 2.0);
        let off = StampedPose::at_rest(sys(), planet_at + delta, UniverseTick(100));
        let landed_off = transfer_frame(&off, planet(), &b).expect("placed");
        assert!((landed_off.pos.offset() - delta).length() < 1e-6);
    }

    #[test]
    fn a_system_planet_system_round_trip_returns_the_original_world_position() {
        // Rehome IN then OUT: System→Planet then Planet→System must return the ORIGINAL System pose
        // (the frame transform and its inverse compose to the identity) — a player who steps onto a
        // planet and back off is exactly where they started, no drift.
        let b = book(vec![(
            planet(),
            FramePlacement::moving(DVec3::new(1.0e5, -3.0e4, 7.0e3), DVec3::new(1.0, 2.0, 3.0)),
        )]);
        let start =
            StampedPose::at_rest(sys(), DVec3::new(1.0e6, 2.0e5, -3.0e5), UniverseTick(100));
        let on_planet = transfer_frame(&start, planet(), &b).expect("in");
        assert_eq!(on_planet.frame, planet());
        let back = transfer_frame(&on_planet, sys(), &b).expect("out");
        assert_eq!(back.frame, sys());
        assert!((back.pos.offset() - start.pos.offset()).length() < 1e-6);
    }

    #[test]
    fn a_rehome_is_invariant_to_how_the_source_position_splits_across_cell_and_offset() {
        // Floating-point robustness: the SAME physical System position, expressed either as a cell-0
        // full offset (today's shipping form) OR as an integer-millimetre CELL + sub-mm residual (the
        // S5 form), rehomes to the SAME planet-local coordinate — the transfer reconstructs the full
        // source position from cell+offset (S0), so how the position is split is invisible to the
        // destination coordinate.
        let planet_at = DVec3::new(141.0, 9.0, -3.0);
        let b = book(vec![(
            planet(),
            FramePlacement::moving(planet_at, DVec3::ZERO),
        )]);
        let world = planet_at + DVec3::new(7.0, -3.0, 11.0);
        // Rep 1: cell-0 full offset (the P3 shipping form).
        let flat = StampedPose::at_rest(sys(), world, UniverseTick(100));
        // Rep 2: normalized into integer-mm cells + residual (the S5 form) — the SAME physical position.
        let mut split = flat;
        split.pos = LatticePos::local(world).normalize(sys().tier());
        let a = transfer_frame(&flat, planet(), &b).expect("flat");
        let c = transfer_frame(&split, planet(), &b).expect("split");
        // Both land at the SAME planet-local coordinate (≈ the 7,-3,11 offset from the centre).
        assert!((a.pos.offset() - c.pos.offset()).length() < 1e-6);
        assert!((a.pos.offset() - DVec3::new(7.0, -3.0, 11.0)).length() < 1e-6);
    }
}
