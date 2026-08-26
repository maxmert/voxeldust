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
use serde::{Deserialize, Serialize};

use crate::ids::UniverseTick;
use crate::placement::PlacementBook;
use crate::pose::{FrameRef, LatticePos, StampedPose, Tier};

/// THE RUNG A PLACEMENT BOOK'S ROWS ARE AUTHORED IN.
///
/// Two places must agree about this and now agree by construction: [`FramePlacement::moving`], which
/// builds an anchor, and [`transfer_frame`]'s guard, which refuses any conversion whose frames do not both
/// count in it. Before S8 the constructor hard-coded the rung and the guard did not exist, so the agreement
/// was a sentence in a doc comment rather than a fact about the program.
pub(crate) const PLACEMENT_TIER: Tier = Tier::Fine;

/// Where a frame's origin sits — and how it moves — relative to the COMMON PARENT at a
/// universe tick. A rigid placement: position, velocity, orientation, angular velocity.
/// All four are authored by the parent's physics writer; P1 uses the identity.
///
/// Serde-carried since the window lane (owner-approved 2026-08-15/16,
/// `docs/design/window_lane.md` §2.2): the pre-inverted hop row (`vd-wire`
/// `session_flow::HopRow.inv` — "my frame expressed in the child's frame at `at`", authored
/// and inverted by the parent) ships this exact type shard→gateway, so the placement that
/// crosses is the one the frame core already folds — never a second rigid-transform shape.
/// It remains OFF every realm-inbound lane: no `InterShardFlow` arm carries it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FramePlacement {
    /// Integer CELL anchor of this frame's origin in the parent frame (parent tier units) — the
    /// exact-integer coarse part of the origin position, companion to `origin` (its sub-cell f64
    /// residual). NORMALIZED on every producer since the cell activation ([`FramePlacement::moving`],
    /// the motion arms, the generator's region centres): the whole-quantum part of an origin rides
    /// here and `origin` holds the sub-cell remainder.
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
    ///
    /// NORMALIZED since the cell activation (real-scale addendum §A4.8 row 5): the integer anchor
    /// carries the whole-quantum part of the origin and `origin` holds the sub-cell residual —
    /// this constructor used to hard-write `origin_cell: ZERO`, which left every mover's and every
    /// static child's placement bypassing the lattice. FINE tier by fence: no COARSE placement can
    /// exist (the COARSE tier is dormant and its conversion refused — R1); the tier parameter
    /// arrives with P10's COARSE activation.
    #[must_use]
    pub fn moving(origin: DVec3, velocity: DVec3) -> FramePlacement {
        let pos = LatticePos::from_metres(origin, PLACEMENT_TIER);
        FramePlacement {
            origin_cell: pos.cell(),
            origin: pos.offset(),
            velocity,
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
        }
    }

    /// The placement anchor as the one lattice position it is — the raw halves fused back for
    /// separation arithmetic ([`transfer_frame`]'s two anchor reads spell this).
    #[must_use]
    pub fn anchor(&self) -> LatticePos {
        LatticePos::at(self.origin_cell, self.origin)
    }
}

/// Why a frame re-expression could not be computed (a typed precondition failure, never
/// a silent garbage spawn — the R6 km-scale-error class made loud).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    #[error("no authored placement for the source frame in this book")]
    UnknownSourceFrame,
    /// A conversion was asked between frames that do not both count in the PLACEMENT rung.
    ///
    /// ★ WHY THIS IS A REFUSAL AND NOT A CONVERSION (slice S8). This function does its ENTIRE arithmetic
    /// at the SOURCE frame's rung: it adds the pose onto the source anchor and subtracts the destination
    /// anchor, both as integer cells. The anchors come out of the placement book, whose rows are authored
    /// in [`PLACEMENT_TIER`]. The moment those rungs differ, a distance counted in one unit is added onto
    /// an anchor counted in another — and re-stating the FINISHED result in the destination's unit cannot
    /// repair arithmetic that has already happened.
    ///
    /// The comment that used to sit on that final re-statement said a rung change *"still routes through
    /// the ONE existing re-quantizer"*. It routes the OUTPUT, not the arithmetic, and it was the
    /// load-bearing false claim in this file.
    ///
    /// S8 BUILDS THE LADDER; S9 CLIMBS IT. Until the arithmetic is done at the book's own rung, a
    /// cross-rung conversion is refused loudly rather than answered wrongly.
    /// A cross-rung crossing could not re-state a distance in the rung it had to move to.
    ///
    /// ★ THIS REPLACES A BLANKET REFUSAL (slice S9). It used to be `CrossTierCrossingNotBuilt`, which
    /// refused EVERY crossing between two different rungs — correct while the arithmetic ran at one
    /// fixed rung, and wrong the moment the world actually had rungs to cross. S9 does the arithmetic
    /// at the book's own rung, so the general case works and only the genuinely impossible one refuses:
    /// a distance too wide to count in the finer unit it is being asked for. A galaxy-wide separation
    /// has no millimetre count, and saying so is the honest answer.
    #[error("a cross-rung crossing could not be re-stated: {0}")]
    CrossTierCrossing(#[from] crate::pose::TierConversionError),
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
    /// A rotation was asked of a separation whose magnitude exceeds the tier's EXACT rotation
    /// reach ([`crate::pose::rotation_exact_reach_m`] = `cell_edge / f64::EPSILON` ≈ 29.399 AU at
    /// FINE). Rotating a lever of magnitude `L` through f64 costs `L·ε`; inside the reach that cost
    /// is at most ONE cell, so the fold is admitted; beyond it the fold would break the millimetre
    /// grid, so it is refused LOUD (R2 — a spinning realm at galaxy magnitudes is a COARSE-tier
    /// object, and this refusal is a named P10 trigger). Replaces the pre-activation
    /// `RotatedFrameAcrossCells` predicate ("rotated AND `cell != ZERO`"), which the activation
    /// made active-hostile: with live cells essentially every hop is cross-cell, so the old rule
    /// would have refused every rotating realm outright (real-scale addendum §A4.6, H-11).
    /// Unreachable in production today (every placement in the tree is identity-oriented).
    #[error("the frame is rotated and the separation exceeds the tier's exact rotation reach")]
    RotationBeyondExactReach,
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
    // ★ THE ARITHMETIC HAPPENS AT THE BOOK'S OWN RUNG (slice S9 — the climb S8 named and deferred).
    //
    // S8 refused a cross-rung crossing outright, because the whole conversion used to run at the SOURCE
    // frame's rung while both anchors come out of the book: the moment those differ, a distance counted
    // in one unit is added onto an anchor counted in another, and re-stating the FINISHED result cannot
    // repair arithmetic that has already happened. That refusal was correct and it is now unnecessary.
    //
    // The book knows what unit it speaks: its rows are placements its ANCHOR authored, in the anchor's
    // own frame. So the anchor's rung is THE rung of this arithmetic — not a constant, and not the
    // source's. The pose is converted INTO it before anything is added, and the answer is converted OUT
    // of it into the destination's rung at the end. Between those two points every quantity counts in
    // one unit, which is the property the S8 guard was standing in for.
    //
    // SAME RUNG — every crossing in the world today — makes both conversions the exact identity, so the
    // shipped path is unchanged. That is measured, not argued — see
    // `pose::tests::a_same_rung_conversion_is_the_identity_bit_for_bit`, which pins both conversions
    // as the bit-for-bit identity at every rung, including residuals a tidier would have "fixed".
    let tier = book.anchor().tier();

    // 1. LIFT (real-scale addendum §A4.7). The pose's own position is a SEPARATION from its frame
    // origin, so it rotates through the ONE rotation rule (`Separation::rotated`) and never through
    // a bare f64 fold. The SOURCE orientation is applied HERE — a sketch that subtracts anchors
    // only would silently regress the `from.orientation * local` term the shipped code computes
    // (H-07). Identity orientation — every placement THE world generates — is a bit-exact
    // passthrough, so the integer half of the pose survives untouched.
    // IN: the pose's own position is a separation from its frame origin, counted in ITS frame's rung.
    // Converted to the book's rung FIRST, so the rotation, the add and the subtract below all happen in
    // one unit. Converting before rotating (rather than after) is deliberate: going to a coarser rung
    // shrinks the cell count, and the rotation's exact-reach bound is kinder to the smaller number.
    let own = pose
        .pos
        .separation(LatticePos::ORIGIN, pose.frame.tier())
        .convert_tier(tier)
        .map_err(FrameError::CrossTierCrossing)?
        .rotated(from.orientation)?;
    // The metre lever, for the Coriolis term (`ω × r`): needed EXACTLY when the frame spins, and
    // exactly then the rotation reach has already bounded it, so this flatten is exact.
    let lever = own.metres();
    let from_anchor = from.anchor();
    // The pose in the common-parent frame: exact integer add onto the source anchor, unnormalized
    // (a working value — the output normalize below is the one re-bucketing).
    let world = own.from_origin(from_anchor);
    let world_vel =
        from.velocity + from.orientation * pose.vel + from.angular_velocity.cross(lever);
    let world_orient = from.orientation * pose.orient;

    // 2. DESCEND — THE EXACT INTEGER SUBTRACTION. This one `separation` replaces BOTH the f64
    // `world_pos − dest.origin` (catastrophic cancellation at scale) and the separate saturating
    // anchor subtract ("THE INTEGER HALF") — one operation, not two halves. The big part of a
    // position never passes through f64: with identity orientations the whole conversion touches
    // f64 only on two sub-cell residuals (the identity theorem, §A4.7) — no magnitude, however
    // large, can degrade it.
    let dest_anchor = dest.anchor();
    let rel_parent = world.separation(dest_anchor, tier);
    // The parent-axes relative vector in metres, for the destination's Coriolis/velocity terms —
    // bounded by the destination realm's own extent on every lawful conversion.
    let rel_parent_m = rel_parent.metres();
    let inv = dest.orientation.inverse();
    // OUT: back into the DESTINATION frame's own rung, which is the last thing that happens to the
    // integer half. Refining can leave the finer rung's domain and is refused by name rather than
    // wrapped — a galaxy-wide distance genuinely has no millimetre count.
    let rel = rel_parent
        .rotated(inv)?
        .convert_tier(to.tier())
        .map_err(FrameError::CrossTierCrossing)?;
    let new_vel = inv * (world_vel - dest.velocity - dest.angular_velocity.cross(rel_parent_m));
    let new_orient = inv * world_orient;

    Ok(StampedPose {
        frame: to,
        // NORMALIZED output (§A4.7): the integer half carries the whole-quantum part into the
        // destination frame, the residual stays sub-cell — the invariant every downstream consumer
        // (the integrator, the verdict, the wire) now relies on. A tier change (FINE ↔ COARSE at
        // P10) still routes through the ONE existing re-quantizer; same tier — every live frame
        // pair today — is the exact identity there.
        // ★ NO RE-STATEMENT HERE ANY MORE (slice S8). The guard above establishes that the source rung,
        // the destination rung and the book's rung are the same, so the call that used to sit here was
        // provably the identity — removing it is byte-identical by proof rather than by measurement. When
        // S9 makes the arithmetic itself rung-aware, the conversion belongs where the arithmetic is, not
        // stapled onto its output.
        pos: rel.from_origin(LatticePos::ORIGIN).normalize(to.tier()),
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

    /// ★ A CONVERSION BETWEEN LEVELS THAT COUNT DIFFERENTLY NOW WORKS, BOTH WAYS (slice S9).
    ///
    /// This test used to assert the opposite. S8 refused every cross-rung crossing, because the whole
    /// arithmetic ran at the SOURCE level's step while the book's anchors are authored in the ANCHOR's
    /// step: where those differ, a distance counted in one unit was being added onto an anchor counted
    /// in another. The refusal was right, and the cure was never a better guard — it was to do the
    /// arithmetic in the book's own unit.
    ///
    /// **The world needs this, and needs it now.** A star system counts in millimetres and the galaxy
    /// that holds it counts in two-metre steps, so "the galaxy places its own star system" — a dot
    /// leaving home, the most ordinary journey in the game — IS a cross-rung crossing. While it was
    /// refused, nothing could enter or leave the galaxy at all.
    ///
    /// What is asserted is the METRES, in both directions, because that is what a crossing must
    /// preserve. The cells are deliberately NOT asserted: the same point has a different whole-number
    /// count in each unit, which is the entire purpose of having units.
    #[test]
    fn a_conversion_between_levels_that_count_differently_works_both_ways() {
        let galaxy = FrameRef::GalaxySpace { galaxy_seed: 0 };
        let b = book(vec![
            (planet(), FramePlacement::identity()),
            (galaxy, FramePlacement::identity()),
        ]);
        let at = DVec3::new(1.5, -2.5, 0.25);
        // THE SOURCE counts in two-metre steps, the destination in millimetres.
        let down = transfer_frame(&pose_in(galaxy, at, DVec3::ZERO), planet(), &b)
            .expect("a galaxy can place its own star system");
        assert_eq!(down.frame, planet());
        assert_eq!(down.pos.delta_m(LatticePos::ORIGIN, Tier::Fine), at);
        // THE DESTINATION counts in two-metre steps, the source in millimetres — the other direction.
        let up = transfer_frame(&pose_in(planet(), at, DVec3::ZERO), galaxy, &b)
            .expect("a star system can hand its own occupant upward");
        assert_eq!(up.frame, galaxy);
        assert_eq!(up.pos.delta_m(LatticePos::ORIGIN, Tier::Galaxy), at);
        // …and the two are INVERSES, exactly. A crossing that loses a millimetre each way is a seam,
        // and a seam is a defect (SL8) — so this is asserted as bit equality, never a tolerance.
        let round = transfer_frame(&down, galaxy, &b).expect("and back again");
        assert_eq!(round.pos.delta_m(LatticePos::ORIGIN, Tier::Galaxy), at);
        // A same-rung conversion is unaffected, so the two above are about the units and not about
        // this function having become permissive.
        assert!(transfer_frame(&pose_in(planet(), DVec3::X, DVec3::ZERO), sys(), &b).is_ok());
    }

    /// ★ THE ONE CROSS-RUNG CROSSING THAT STILL REFUSES, and it refuses for a real reason: a distance
    /// too wide to count in the finer unit being asked for.
    ///
    /// The galaxy's own lattice reaches 2⁶² cells of two metres. Asked for that in millimetres, the
    /// count would need 2⁷³ — past what the number can hold. There is no right answer, so it is named
    /// rather than wrapped. This is P10's trigger, stated as something that fails.
    #[test]
    fn a_distance_too_wide_for_the_finer_unit_is_refused_by_name() {
        let galaxy = FrameRef::GalaxySpace { galaxy_seed: 0 };
        let b = book(vec![
            (planet(), FramePlacement::identity()),
            (galaxy, FramePlacement::identity()),
        ]);
        // The bound, DERIVED from the two steps rather than typed: refining multiplies the cell count
        // by the ratio between them, so the widest that still fits is the domain over that ratio.
        let ratio = 1_i64 << (Tier::Galaxy.step_exponent() - Tier::Fine.step_exponent());
        let max_cells = crate::pose::CELL_DOMAIN_MAX / ratio;
        let far = StampedPose {
            frame: galaxy,
            pos: LatticePos::at(I64Vec3::new(max_cells + 1, 0, 0), DVec3::ZERO),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(100),
        };
        let refused = transfer_frame(&far, planet(), &b).expect_err("no millimetre count exists");
        assert_eq!(
            refused,
            FrameError::CrossTierCrossing(crate::pose::TierConversionError::BeyondReach {
                from: Tier::Galaxy,
                to: Tier::Fine,
                widest_cells: (max_cells + 1).unsigned_abs(),
                max_cells: max_cells.unsigned_abs(),
            })
        );
        // …and ONE cell narrower is answered, so the refusal is a bound and not a blanket.
        let near = StampedPose {
            pos: LatticePos::at(I64Vec3::new(max_cells, 0, 0), DVec3::ZERO),
            ..far
        };
        assert!(transfer_frame(&near, planet(), &b).is_ok());
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
        assert!((out.pos.delta_m(LatticePos::ORIGIN, planet().tier()) - want).length() < 1e-9);
        // The dest write is NORMALIZED (the activation): the whole-quantum part rides the integer
        // half — 1000.5 m = 1_024_512 cells exactly (the FINE edge is a power of two).
        assert_eq!(out.pos.cell(), I64Vec3::new(1_024_512, -256, 768));
    }

    #[test]
    fn transfer_output_is_normalized_and_value_identical_to_the_pre_lattice_lift() {
        // The activation's output invariant: the dest pose is NORMALIZED (integer half full, sub-cell
        // residual) and its VALUE is exactly the pre-lattice f64 answer — dyadic inputs, so the
        // equality is bit-exact on the flattened metres.
        let b = book(vec![(
            planet(),
            FramePlacement::moving(DVec3::new(10.0, 0.0, 0.0), DVec3::ZERO),
        )]);
        let p = pose_in(sys(), DVec3::new(3.0, 4.0, 5.0), DVec3::ZERO);
        let out = transfer_frame(&p, planet(), &b).expect("transfer");
        // planet origin at +10x ⇒ value (3-10, 4, 5); normalized ⇒ cells (-7168, 4096, 5120), offset 0.
        assert_eq!(out.pos.cell(), I64Vec3::new(-7168, 4096, 5120));
        assert_eq!(
            out.pos.delta_m(LatticePos::ORIGIN, planet().tier()),
            DVec3::new(-7.0, 4.0, 5.0)
        );
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
        // Value-identical (the output is normalized; 47.0 is dyadic so the flatten is bit-exact).
        assert_eq!(
            got.pos.delta_m(LatticePos::ORIGIN, planet().tier()),
            p.pos.delta_m(LatticePos::ORIGIN, sys().tier())
        );
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
            (back.pos.delta_m(LatticePos::ORIGIN, sys().tier())
                - original.pos.delta_m(LatticePos::ORIGIN, sys().tier()))
            .length()
                < 1e-9,
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
            (got.pos.delta_m(LatticePos::ORIGIN, planet().tier()) - DVec3::new(0.0, -1.0, 0.0))
                .length()
                < 1e-9,
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
        // `moving` NORMALIZES: 1 m rides the integer anchor (1024 fine cells), residual zero,
        // value preserved.
        let m = FramePlacement::moving(DVec3::X, DVec3::Y);
        assert_eq!(m.origin_cell, I64Vec3::new(1024, 0, 0));
        assert_eq!(m.origin, DVec3::ZERO);
        assert_eq!(m.anchor().delta_m(LatticePos::ORIGIN, Tier::Fine), DVec3::X);
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
            FrameError::RotationBeyondExactReach.to_string(),
            "the frame is rotated and the separation exceeds the tier's exact rotation reach"
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
        // THE ACTIVATION: the pose's OWN cell now rides INTEGRALLY too — the identity theorem
        // (§A4.7): with identity orientations the whole conversion touches f64 only on sub-cell
        // residuals. The output is normalized: pose anchor + placement anchor + 145.5 m of residual
        // metres, all in the integer half (145.5 m = 148_992 cells exactly).
        assert_eq!(
            got.pos.cell(),
            I64Vec3::new(2 * ANCHOR_CELLS + 148_992, 0, 0),
            "both integer anchors AND the pose's own magnitude ride the integer half"
        );
        assert_eq!(got.pos.offset(), DVec3::ZERO);
        assert_eq!(got.frame, sys());

        // GOING BACK DOWN, the anchors subtract the other way and the round trip is EXACT — the same
        // physical point, to the bit, with no accumulated f64 drift. (The split between cell and offset
        // differs on the way back only because the pose's own cell was folded into the lever on the way
        // out; the total is identical, which is what a position means.)
        let back = transfer_frame(&got, planet(), &b).expect("system -> planet");
        // Normalized both ways: the round trip returns the identical physical point with the whole
        // magnitude in the integer half (ANCHOR_M + 0.5 m = ANCHOR_CELLS + 512 cells exactly).
        assert_eq!(back.pos.cell(), I64Vec3::new(ANCHOR_CELLS + 512, 0, 0));
        assert_eq!(back.pos.offset(), DVec3::ZERO);
        let total_out = |p: LatticePos| p.offset().x + p.cell().as_dvec3().x * FINE_CELL_EDGE_M;
        assert_eq!(total_out(back.pos), total_out(pose.pos));
        assert_eq!(total_out(back.pos), ANCHOR_M + 0.5);
    }

    #[test]
    fn a_rotated_dest_folds_exactly_in_reach_and_refuses_beyond_the_rotation_reach() {
        // THE RESTATED ROTATION LAW (real-scale addendum §A4.6). The old predicate ("rotated AND
        // anchors in different cells" => refuse) became active-hostile the moment cells went live:
        // essentially every hop is cross-cell, so it would have refused every rotating realm. The
        // restated rule derives from the actual cost: rotating a lever of magnitude L costs L*eps,
        // so inside `rotation_exact_reach_m` (2^42 m at FINE) the fold costs at most one cell and
        // is ADMITTED; beyond it, refused LOUD (R2 -- the P10 trigger).
        let spun = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let p = pose_in(sys(), DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO);

        // Rotated dest, anchors a whole block apart but WELL INSIDE the reach -> ADMITTED, and the
        // fold is exact: the pose lands `-anchor` along the rotated axes (a 90-degree turn about z
        // maps the parent -x separation onto +y).
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
        let got = transfer_frame(&p, planet(), &b).expect("in-reach rotated cross-cell dest folds");
        let flat = got.pos.delta_m(LatticePos::ORIGIN, sys().tier());
        // Parent-frame separation: (1 - ANCHOR_M, 0, 0); rotated by spun.inverse() -> (0, ANCHOR_M - 1, 0).
        assert!((flat - DVec3::new(0.0, ANCHOR_M - 1.0, 0.0)).length() < 1e-3);

        // Rotated dest BEYOND the exact rotation reach (2^42 m at FINE = 2^52 cells) -> typed refusal.
        let far = 1_i64 << 53; // 2^53 cells = 2 x the reach
        let b = book(vec![(
            planet(),
            FramePlacement {
                origin_cell: I64Vec3::new(far, 0, 0),
                origin: DVec3::ZERO,
                velocity: DVec3::ZERO,
                orientation: spun,
                angular_velocity: DVec3::ZERO,
            },
        )]);
        assert_eq!(
            transfer_frame(&p, planet(), &b),
            Err(FrameError::RotationBeyondExactReach)
        );

        // Rotated dest at the SAME anchor: nothing large needs rotating -> allowed, exact.
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
        let flat = got.pos.delta_m(LatticePos::ORIGIN, sys().tier());
        assert!((flat - DVec3::new(0.0, -1.0, 0.0)).length() < 1e-9);
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
        assert!(
            landed
                .pos
                .delta_m(LatticePos::ORIGIN, planet().tier())
                .length()
                < 1e-6
        );
        // Offset by delta ⇒ planet-local delta.
        let delta = DVec3::new(10.0, -5.0, 2.0);
        let off = StampedPose::at_rest(sys(), planet_at + delta, UniverseTick(100));
        let landed_off = transfer_frame(&off, planet(), &b).expect("placed");
        assert!(
            (landed_off.pos.delta_m(LatticePos::ORIGIN, planet().tier()) - delta).length() < 1e-6
        );
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
        // Rep 1: raw cell-0 full offset (the pre-activation shipping form — constructible in-crate
        // only, exactly so nothing outside can ship one).
        let mut flat = StampedPose::at_rest(sys(), world, UniverseTick(100));
        flat.pos = LatticePos::local(world);
        // Rep 2: normalized (the shipping form since the activation) — the SAME physical position.
        let split = StampedPose::at_rest(sys(), world, UniverseTick(100));
        let a = transfer_frame(&flat, planet(), &b).expect("flat");
        let c = transfer_frame(&split, planet(), &b).expect("split");
        // Both land at the SAME planet-local coordinate (≈ the 7,-3,11 offset from the centre).
        let flat_m = |p: &StampedPose| p.pos.delta_m(LatticePos::ORIGIN, planet().tier());
        assert!((flat_m(&a) - flat_m(&c)).length() < 1e-6);
        assert!((flat_m(&a) - DVec3::new(7.0, -3.0, 11.0)).length() < 1e-6);
    }

    #[test]
    fn a_frame_placement_roundtrips_postcard_field_complete() {
        // The window lane's hop row (owner-approved 2026-08-15/16, docs/design/window_lane.md §2.2)
        // ships this exact type shard→gateway, so its serde carry is wire contract: every field —
        // integer cell anchor included — survives a postcard roundtrip byte-exactly. A full-width
        // fixture (no zero field) so a dropped/reordered field cannot pass as a lucky zero.
        let full = FramePlacement {
            origin_cell: I64Vec3::new(3, -7, 11),
            origin: DVec3::new(1.5, -2.25, 9.0),
            velocity: DVec3::new(0.5, 4.0, -1.0),
            orientation: DQuat::from_xyzw(0.5, 0.5, 0.5, 0.5),
            angular_velocity: DVec3::new(0.0625, -0.125, 0.25),
        };
        let bytes = postcard::to_allocvec(&full).expect("encode");
        assert_eq!(
            postcard::from_bytes::<FramePlacement>(&bytes).expect("decode"),
            full
        );
        // The identity — the P1..P3 shipping value — roundtrips too.
        let id = FramePlacement::identity();
        let id_bytes = postcard::to_allocvec(&id).expect("encode identity");
        assert_eq!(
            postcard::from_bytes::<FramePlacement>(&id_bytes).expect("decode identity"),
            id
        );
    }
}
