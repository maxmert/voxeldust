//! ★ THE MOTION OF A DRAWN THING (the temporal path's slice T1; `docs/investigation/2026-09-28/
//! temporal_path_design.md` §5): the place where a drawn thing stood in LAST frame's render space,
//! so that the reconstruction finds each pixel in the last frame's picture.
//!
//! **The one rule (§5.1).** The previous place of a drawn thing is its realm's step applied to its
//! place now, then the thing's own change inside the realm.
//! - THE REALM'S STEP `S_r` maps this frame's render space to last frame's, for realm `r`:
//!   `S_r = P_r(last) · P_r(now)⁻¹`, where `P_r(x) = centre + facing · x` is the placement that the
//!   window DREW for the realm (the eye stands at the origin in both frames). A realm that had no
//!   placement last frame takes the identity: its chunks stood nowhere, so nothing moved them.
//! - THE OWN CHANGE of a ground vertex (§5.2): its morph, its sink and its arrival ramp at LAST
//!   frame's band row and clock.
//!
//! **The wrapping trap (§5.2, the design's F6).** The ramp's weight reads the stamp's age as a
//! WRAPPING difference ([`crate::arrival::ramp_weight`]). A chunk that lands this frame carries a
//! stamp one frame NEWER than last frame's clock; read wrapping, its age at last frame's clock is
//! about 2³² ticks, and its weight reads 1 (whole) instead of 0 (on its stand-in). The previous place
//! reads the age SIGNED: a stamp newer than the clock is weight 0, the stand-in's surface. So a
//! landing chunk's previous place is exactly where its stand-in stood (the landing identity), and the
//! history carries across the landing.
//!
//! This module is the rule that the shaders transliterate step for step (the ground's vertex stage
//! and its prepass), in `f64`, so its arms are unit tests. The step is computed in `f64` from the
//! two drawn placements and narrowed ONCE ([`RealmStep::rows`]), as the placement is.
//!
//! **Example.** A hull at 528 m/s passes over the belt, and a rung-9 chunk lands late 1 km beside it.
//! The engine's own rule gives that chunk no previous place in its first frame (about 23 display
//! pixels of error). Here its previous place is the planet's step applied to its stand-in's surface,
//! so it enters the history with no mark.

use vd_core::glam::{DMat3, DQuat, DVec3};

use crate::arrival::{CLOCK_MASK, RampVertex, tag_clock, tag_ramps};

/// A realm's placement as the window drew it: `P(x) = centre + facing · x`, in render space (the
/// eye at the origin).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawnPlacement {
    pub centre: DVec3,
    pub facing: DQuat,
}

impl DrawnPlacement {
    /// The render-space place of the realm's point `x`.
    #[must_use]
    pub fn place(&self, x: DVec3) -> DVec3 {
        self.centre + self.facing * x
    }
}

/// ★ THE REALM'S STEP: this frame's render space to last frame's, for one realm — a rotation and a
/// translation, `S(y) = rotation · y + translation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealmStep {
    pub rotation: DQuat,
    pub translation: DVec3,
}

impl RealmStep {
    /// The step of a realm that did not move (or had no placement last frame).
    pub const IDENTITY: RealmStep = RealmStep {
        rotation: DQuat::IDENTITY,
        translation: DVec3::ZERO,
    };

    /// `S = P(last) · P(now)⁻¹`: a point that the window draws at `y` this frame stood at `S(y)` last
    /// frame. A realm with no placement last frame takes the identity.
    #[must_use]
    pub fn between(last: Option<DrawnPlacement>, now: DrawnPlacement) -> RealmStep {
        last.map_or(RealmStep::IDENTITY, |last| {
            let rotation = last.facing * now.facing.inverse();
            RealmStep {
                rotation,
                translation: last.centre - rotation * now.centre,
            }
        })
    }

    /// Carries a render-space point of this frame into last frame's render space.
    #[must_use]
    pub fn apply(&self, y: DVec3) -> DVec3 {
        self.rotation * y + self.translation
    }

    /// Turns a direction of this frame into last frame's (a chunk carries no scale).
    #[must_use]
    pub fn turn(&self, v: DVec3) -> DVec3 {
        self.rotation * v
    }

    /// The step as the three rows of a 3 × 4 matrix, narrowed ONCE to `f32`: what the band table's
    /// header carries to the shaders. The translation is a difference of two centres computed in
    /// `f64` (a step of metres even when each centre stands thousands of kilometres off), so its
    /// narrowing keeps its precision.
    #[must_use]
    pub fn rows(&self) -> [[f32; 4]; 3] {
        let m = DMat3::from_quat(self.rotation);
        let t = self.translation;
        let row = |r: DVec3, t: f64| [r.x as f32, r.y as f32, r.z as f32, t as f32];
        [row(m.row(0), t.x), row(m.row(1), t.y), row(m.row(2), t.z)]
    }
}

/// ★ THE RAMP'S WEIGHT AT LAST FRAME'S CLOCK, the age read SIGNED: a tag newer than the clock (a
/// chunk that landed after it) reads 0, its stand-in's surface; a tag without the ramp's flag (a
/// chunk that entered whole) reads 1. Otherwise the age over the ramp, clamped to `[0, 1]`, as
/// [`crate::arrival::ramp_weight`] reads it. The signed read of the 31-bit difference holds across
/// the clock's wrap, for any age under half the clock's range (2³⁰ ticks, 12 days).
#[must_use]
pub fn previous_ramp_weight(clock: u32, tag: u32, ramp: u32) -> f64 {
    let diff = clock.wrapping_sub(tag_clock(tag)) & CLOCK_MASK;
    let signed = (diff << 1).cast_signed() >> 1;
    let age = f64::from(signed) / f64::from(ramp.max(1));
    let ramps = u8::from(tag_ramps(tag));
    f64::from(1 - ramps) + f64::from(ramps) * age.clamp(0.0, 1.0)
}

/// ★ THE PREVIOUS PLACED VERTEX (§5.2): the vertex's own place and its radial carried into last
/// frame's render space by its realm's step, and placed there with last frame's band row and ramp
/// weight. Its distance is its own place's length there, because the eye stands at the origin in
/// both frames; its morph and its sinks are the vertex's own.
#[must_use]
pub fn previous_placed(
    vertex: &RampVertex,
    step: &RealmStep,
    last_row: [f32; 4],
    last_weight: f64,
) -> DVec3 {
    RampVertex {
        own: step.apply(vertex.own),
        radial: step.turn(vertex.radial),
        ..*vertex
    }
    .placed(last_row, last_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrival::{RAMP_FLAG, ramp_ticks, ramp_weight, whole};
    use crate::ladder_view::{band_row, fade_bands};

    /// The home planet's ladder is 19 rungs; any rung with a coarser one serves.
    const RUNGS: u8 = 19;

    /// A band row of rung `rung` on the unbounded ladder, with the sink ramp ending at its edge.
    fn row(rung: u8) -> [f32; 4] {
        let (fade_in, fade_out) = fade_bands(rung, RUNGS);
        band_row((fade_in, fade_out), fade_in[1])
    }

    /// Two drawn placements of one realm a frame apart: the planet turned a little and slid under
    /// the eye (the eye walked on, so the centre moved in render space).
    fn two_placements() -> (DrawnPlacement, DrawnPlacement) {
        let last = DrawnPlacement {
            centre: DVec3::new(-5_938_198.5, -1_425_844.2, -1_724_752.4),
            facing: DQuat::from_axis_angle(DVec3::new(0.2, 0.9, -0.3).normalize(), 0.61),
        };
        let now = DrawnPlacement {
            centre: last.centre + DVec3::new(0.35, -0.12, 0.8),
            facing: DQuat::from_axis_angle(DVec3::Y, 2.0e-5) * last.facing,
        };
        (last, now)
    }

    /// A ground vertex 30 % into rung 9's fade-out band, along a line of sight across its radial.
    fn vertex(fine: [f32; 4]) -> RampVertex {
        let (lo, hi) = (f64::from(fine[2]), f64::from(fine[3]));
        let radial = DVec3::new(0.3, -0.4, 0.866_025_403_784_438_6).normalize();
        let side = radial.cross(DVec3::Z).normalize();
        RampVertex {
            own: side * (lo + 0.3 * (hi - lo)),
            radial,
            morph_m: 25.0,
            sink_m: 40.0,
            stand_in_sink_m: 90.0,
        }
    }

    fn close(a: DVec3, b: DVec3, within: f64) -> bool {
        (a - b).length() <= within
    }

    /// ★ THE STEP CARRIES A REALM'S POINT TO WHERE LAST FRAME DREW IT, and a direction likewise:
    /// `S(P_now(x)) = P_last(x)` for every point of the realm.
    #[test]
    fn the_step_carries_a_realm_point_to_where_last_frame_drew_it() {
        let (last, now) = two_placements();
        let step = RealmStep::between(Some(last), now);
        for x in [
            DVec3::ZERO,
            DVec3::new(6_345_363.5, 0.0, 0.0),
            DVec3::new(-120.0, 4_000.0, 77.5),
        ] {
            assert!(close(step.apply(now.place(x)), last.place(x), 1e-6));
        }
        let v = DVec3::new(0.6, 0.0, -0.8);
        assert!(close(step.turn(now.facing * v), last.facing * v, 1e-12));
    }

    /// ★ A SWAP (the design's §6.2: a boarding turns the render frame's axes): the hull's berth faced
    /// 90° off the planet's axes, so between the two frames every realm's drawn placement turned a
    /// quarter and its centre moved by the hull's offset. The step still carries each point of the
    /// planet, and each radial, to where last frame drew it: no reset, no special case.
    #[test]
    fn a_swap_turns_the_axes_and_the_step_carries_every_point() {
        let last = DrawnPlacement {
            centre: DVec3::new(0.0, -6_345_363.5, 0.0),
            facing: DQuat::IDENTITY,
        };
        let turn = DQuat::from_axis_angle(DVec3::Z, std::f64::consts::FRAC_PI_2);
        let now = DrawnPlacement {
            centre: turn * last.centre + DVec3::new(12.0, 3.5, -40.0),
            facing: turn * last.facing,
        };
        let step = RealmStep::between(Some(last), now);
        for x in [
            DVec3::new(0.0, 6_345_363.5, 0.0),
            DVec3::new(1_200.0, 6_345_100.0, -300.0),
        ] {
            assert!(close(step.apply(now.place(x)), last.place(x), 1e-6));
        }
        let radial = DVec3::new(0.0, 1.0, 0.0);
        assert!(close(
            step.turn(now.facing * radial),
            last.facing * radial,
            1e-12
        ));
    }

    /// A realm with no placement last frame takes the identity; a realm drawn at one placement in
    /// both frames steps by nothing.
    #[test]
    fn a_new_realm_takes_the_identity_and_a_still_one_steps_by_nothing() {
        let (last, now) = two_placements();
        assert_eq!(RealmStep::between(None, now), RealmStep::IDENTITY);
        let y = DVec3::new(3.0, -4.0, 12.0);
        assert_eq!(RealmStep::IDENTITY.apply(y), y);
        let still = RealmStep::between(Some(last), last);
        assert!(close(still.apply(y), y, 1e-6));
        assert!(close(still.turn(y), y, 1e-12));
    }

    /// ★ THE STEP IS NARROWED ONCE: the rows applied in `f32` agree with the `f64` step within a
    /// millimetre at a kilometre, although each centre stands 6 400 km off (the difference of the
    /// centres is taken before the narrowing).
    #[test]
    fn the_rows_narrow_the_step_once_and_keep_a_millimetre() {
        let (last, now) = two_placements();
        let step = RealmStep::between(Some(last), now);
        let rows = step.rows();
        let y = DVec3::new(812.0, -140.0, 377.0);
        let yf = [y.x as f32, y.y as f32, y.z as f32];
        let apply = |r: [f32; 4]| r[0] * yf[0] + r[1] * yf[1] + r[2] * yf[2] + r[3];
        let narrow = DVec3::new(
            f64::from(apply(rows[0])),
            f64::from(apply(rows[1])),
            f64::from(apply(rows[2])),
        );
        let exact = step.apply(y);
        assert!(close(narrow, exact, 1e-3), "{narrow} {exact}");
    }

    /// ★ THE PREVIOUS WEIGHT READS THE AGE SIGNED — and THE CONTROL: the wrapping read of
    /// [`ramp_weight`] reads a stamp one frame newer than the clock as whole.
    #[test]
    fn the_previous_weight_reads_the_age_signed_and_the_wrapping_read_is_the_trap() {
        let ramp = ramp_ticks();
        let t = |clock: u32| RAMP_FLAG | clock;
        // A chunk that landed one frame (17 ticks) after last frame's clock: on its stand-in.
        assert_eq!(previous_ramp_weight(5_000, t(5_017), ramp), 0.0);
        assert_eq!(ramp_weight(5_000, t(5_017), ramp), 1.0, "the trap");
        // At its clock, half way, at the end and after: 0, 0.5, 1, 1.
        assert_eq!(previous_ramp_weight(5_000, t(5_000), ramp), 0.0);
        assert_eq!(previous_ramp_weight(5_256, t(5_000), ramp), 0.5);
        assert_eq!(previous_ramp_weight(5_512, t(5_000), ramp), 1.0);
        assert_eq!(previous_ramp_weight(9_000, t(5_000), ramp), 1.0);
        // A chunk that entered whole reads 1 at once, untagged or at its clock.
        assert_eq!(previous_ramp_weight(5_000, 0, ramp), 1.0);
        assert_eq!(previous_ramp_weight(5_000, 5_017, ramp), 1.0);
        // Across the clock's wrap: stamped 100 ticks before it, read 156 ticks after it.
        assert_eq!(previous_ramp_weight(156, t(CLOCK_MASK - 99), ramp), 0.5);
        // Stamped just after the wrap, read just before it: newer than the clock, on its stand-in.
        assert_eq!(previous_ramp_weight(CLOCK_MASK - 5, t(10), ramp), 0.0);
        // A ramp of nought ticks is a ramp of one: no division by nought.
        assert_eq!(previous_ramp_weight(5_001, t(5_000), 0), 1.0);
    }

    /// ★ THE LANDING IDENTITY: a chunk that lands this frame stood on its stand-in's surface last
    /// frame, so its previous place is the stand-in's drawn surface, carried by the step.
    #[test]
    fn a_landing_chunks_previous_place_is_its_stand_ins_surface() {
        let fine = row(9);
        let (last, now) = two_placements();
        let step = RealmStep::between(Some(last), now);
        let v = vertex(fine);
        let a_last = previous_ramp_weight(5_000, RAMP_FLAG | 5_017, ramp_ticks());
        let previous = previous_placed(&v, &step, fine, a_last);
        let carried = RampVertex {
            own: step.apply(v.own),
            radial: step.turn(v.radial),
            ..v
        };
        assert!(close(previous, carried.stand_in_place(fine), 1e-9));
        // Its own place stands elsewhere: the landing is a rise, which the history must not read
        // as one frame's motion.
        assert!(!close(previous, carried.own_place(fine), 1.0));
    }

    /// A still realm, a whole vertex and one band row: the previous place is the place now (no
    /// motion); a band that slid since last frame moves the previous place by last frame's row.
    #[test]
    fn a_still_vertex_does_not_move_and_a_band_slide_moves_it_by_last_frames_row() {
        let fine = row(9);
        let v = vertex(fine);
        let step = RealmStep::IDENTITY;
        assert!(close(
            previous_placed(&v, &step, fine, 1.0),
            v.placed(fine, 1.0),
            1e-9
        ));
        // Last frame's fade-out band stood 5 % farther out: the vertex was less morphed then.
        let (lo, hi) = (fine[2], fine[3]);
        let slid = [
            fine[0],
            fine[1],
            lo + 0.05 * (hi - lo),
            hi + 0.05 * (hi - lo),
        ];
        let previous = previous_placed(&v, &step, slid, 1.0);
        assert!(!close(previous, v.placed(fine, 1.0), 1e-3));
        let d = v.own.length();
        assert!(
            whole(slid, d) > whole(fine, d),
            "last frame the vertex was more whole"
        );
        assert!(close(previous, v.placed(slid, 1.0), 1e-9));
    }

    /// TODAY'S FORMULA (the design's F5), as a fixture: last frame's placement of the vertex moved
    /// by THIS frame's displacement (its morph, sink and ramp now). `carried` is whether the engine
    /// holds a previous placement for the chunk; a chunk spawned this frame has none, and the
    /// engine reads its place now (F7).
    fn todays_previous(
        v: &RampVertex,
        step: &RealmStep,
        carried: bool,
        row_now: [f32; 4],
        weight_now: f64,
    ) -> DVec3 {
        let previous_own = if carried { step.apply(v.own) } else { v.own };
        previous_own + (v.placed(row_now, weight_now) - v.own)
    }

    /// ★ THE CONTROL (the design's T1 control 2): today's formula FAILS the ramp, the band slide and
    /// the landing — each case's exact previous place, which the tests above judge, stands more than
    /// a centimetre from today's.
    #[test]
    fn todays_formula_fails_the_ramp_the_band_slide_and_the_landing() {
        let fine = row(9);
        let v = vertex(fine);
        let ramp = ramp_ticks();
        // THE RAMP: a still realm, a chunk half way up its ramp; last frame it stood 17 ticks lower.
        let (now_clock, stamp) = (5_256, RAMP_FLAG | 5_000);
        let still = RealmStep::IDENTITY;
        let exact = previous_placed(
            &v,
            &still,
            fine,
            previous_ramp_weight(now_clock - 17, stamp, ramp),
        );
        let today = todays_previous(&v, &still, true, fine, ramp_weight(now_clock, stamp, ramp));
        assert!(!close(exact, today, 0.01), "the ramp: {exact} {today}");
        // THE BAND SLIDE: a still realm, a whole vertex, last frame's band 5 % farther out.
        let (lo, hi) = (fine[2], fine[3]);
        let slid = [
            fine[0],
            fine[1],
            lo + 0.05 * (hi - lo),
            hi + 0.05 * (hi - lo),
        ];
        let exact = previous_placed(&v, &still, slid, 1.0);
        let today = todays_previous(&v, &still, true, fine, 1.0);
        assert!(
            !close(exact, today, 0.01),
            "the band slide: {exact} {today}"
        );
        // THE LANDING: the planet stepped under the eye, and a chunk spawned this frame.
        let (last, now) = two_placements();
        let step = RealmStep::between(Some(last), now);
        let exact = previous_placed(
            &v,
            &step,
            fine,
            previous_ramp_weight(5_000, RAMP_FLAG | 5_017, ramp),
        );
        let today = todays_previous(
            &v,
            &step,
            false,
            fine,
            ramp_weight(5_017, RAMP_FLAG | 5_017, ramp),
        );
        assert!(!close(exact, today, 0.01), "the landing: {exact} {today}");
    }
}
