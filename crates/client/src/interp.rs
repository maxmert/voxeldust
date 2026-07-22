//! Game-time-indexed interpolation — the no-prediction render primitive.
//!
//! `lerp_at_game_time` is ported verbatim (with its tests) from the legacy client
//! (`client/src/shard/temporal_interp.rs`): given two snapshots and a target
//! game-time it returns the blended value WITHIN the window and clamps to an
//! endpoint outside it — it NEVER extrapolates. [`EntityTrack`] is the legacy
//! `ShipPoseTrack` re-keyed onto the analytic `universe_tick` timeline and feeding
//! [`StampedPose`] snapshots; one track per `(SubId, EntityId)` lives in the view.
//!
//! No-prediction is structural here: the render sample is a [`RenderPose`] with NO
//! velocity field, and nothing on this path reads `StampedPose::vel` or calls
//! `advanced_ballistic`. A stalled entity FREEZES at its last delivered pose
//! (the clamp), it does not coast.

use glam::{DQuat, DVec3};
use vd_core::UniverseTick;
use vd_core::pose::{FrameRef, StampedPose};

use crate::tick_to_f64;

/// Linear interpolation between two value snapshots indexed on the
/// server-authoritative analytic clock. Returns `current` at/after the window or
/// when the window is degenerate, `prev` at/before it, and the blend between.
/// NEVER extrapolates. `lerp_fn` is called at most once and only on the blend
/// branch (so `slerp` pays its cost only when actually interpolating).
#[inline]
pub fn lerp_at_game_time<T, F>(
    prev: T,
    prev_time: f64,
    current: T,
    current_time: f64,
    target: f64,
    lerp_fn: F,
) -> T
where
    T: Copy,
    F: FnOnce(T, T, f64) -> T,
{
    let span = current_time - prev_time;
    if span <= 0.0 || target >= current_time {
        return current;
    }
    if target <= prev_time {
        return prev;
    }
    let alpha = ((target - prev_time) / span).clamp(0.0, 1.0);
    lerp_fn(prev, current, alpha)
}

/// What the renderer draws for one entity at the render cursor: a pose with NO
/// velocity — the no-prediction firewall, structural. There is no field a caller
/// could read to extrapolate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderPose {
    pub frame: FrameRef,
    pub pos: DVec3,
    pub orient: DQuat,
}

/// A per-entity two-snapshot interpolation buffer keyed on the shared
/// `universe_tick` timeline. `current` is the freshest delivered pose; `prev` the
/// one before it. Sampling between them at the render cursor gives smooth motion
/// at 20 Hz snapshot rate without prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EntityTrack {
    prev: StampedPose,
    current: StampedPose,
    last_universe_tick: UniverseTick,
}

impl EntityTrack {
    /// First sighting: a degenerate window (prev == current) — sampling returns the
    /// pose unchanged until a second snapshot arrives.
    #[must_use]
    pub fn new(first: StampedPose) -> EntityTrack {
        EntityTrack {
            prev: first,
            current: first,
            last_universe_tick: first.universe_tick,
        }
    }

    /// Fold in a delivered pose. A STRICTLY newer `universe_tick` shifts
    /// `current → prev` and advances the window; an equal tick (a sibling chunk of
    /// the same tick) updates `current` in place WITHOUT collapsing the window, so
    /// the interpolation span is preserved (audit finding-25). Stale poses never
    /// reach here — the §6.3 gate drops them upstream.
    pub fn observe(&mut self, pose: StampedPose) {
        if pose.frame != self.current.frame {
            // A FRAME CHANGE (P8 board-ship / P10 warp / P2 cross-frame): `prev` and
            // the new pose live in DIFFERENT frames, so blending their raw
            // coordinates would yield a garbage intermediate. Collapse the window to
            // the new-frame pose — the next sample FREEZES there rather than
            // interpolating across frames. (Re-expressing `prev` into the new frame
            // via `transfer_frame` is the P8/P10 refinement; the seam is `world_pos`.)
            *self = EntityTrack::new(pose);
        } else if pose.universe_tick > self.last_universe_tick {
            self.prev = self.current;
            self.current = pose;
            self.last_universe_tick = pose.universe_tick;
        } else {
            self.current = pose;
        }
    }

    /// The pose to render at `cursor` (a continuous f64 in universe-tick units).
    /// Pos lerps, orientation slerps, and the cursor is clamped to the window —
    /// past the freshest tick the entity FREEZES at `current` (never vel-projected).
    #[must_use]
    pub fn sample(&self, cursor: f64) -> RenderPose {
        let prev_time = tick_to_f64(self.prev.universe_tick);
        let current_time = tick_to_f64(self.current.universe_tick);
        // Blend the frame-local offsets (RenderPose is render-local DVec3). Through P3 cell is ZERO so
        // the offset is the full frame-local position; prev/current share a cell so the lerp is valid.
        // Cross-cell interpolation (rebasing prev into current's cell, or collapsing the window on a
        // same-frame cell change the way `observe` does on a frame change) lands WITH P4/P5 re-centering
        // — the first producer of a non-zero cell — galaxy ly-cells at P10. D-41.
        let pos = lerp_at_game_time(
            self.prev.pos.offset(),
            prev_time,
            self.current.pos.offset(),
            current_time,
            cursor,
            DVec3::lerp,
        );
        let orient = lerp_at_game_time(
            self.prev.orient,
            prev_time,
            self.current.orient,
            current_time,
            cursor,
            DQuat::slerp,
        );
        RenderPose {
            frame: self.current.frame,
            pos,
            orient,
        }
    }

    /// The freshest delivered tick (the window's leading edge).
    #[must_use]
    pub fn newest_tick(self) -> UniverseTick {
        self.current.universe_tick
    }

    /// The frame the entity is currently expressed in (the leading edge) — the basis for
    /// the player-location stat. Frame does not interpolate, so no cursor is needed.
    #[must_use]
    pub fn current_frame(self) -> FrameRef {
        self.current.frame
    }

    /// The freshest delivered pose as a [`RenderPose`] — the LATEST server-shipped position with NO
    /// interpolation (the leading edge). For consumers that want the latest pose without a render cursor
    /// (the realm-box overlay, FA-2c): a slow moving realm box shows its latest streamed placement per
    /// step; render-side cursor interpolation is an FA-5+ smoothness refinement (D-45).
    #[must_use]
    pub fn current_render_pose(self) -> RenderPose {
        RenderPose {
            frame: self.current.frame,
            pos: self.current.pos.offset(),
            orient: self.current.orient,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lerp_vec(a: DVec3, b: DVec3, t: f64) -> DVec3 {
        a.lerp(b, t)
    }

    fn pose_at(tick: u64, x: f64) -> StampedPose {
        StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 1 },
            DVec3::new(x, 0.0, 0.0),
            UniverseTick(tick),
        )
    }

    // ---- ported lerp_at_game_time contract (legacy temporal_interp tests) -------

    #[test]
    fn returns_current_when_target_at_or_after_current_time() {
        let prev = DVec3::ZERO;
        let current = DVec3::new(10.0, 20.0, 30.0);
        assert_eq!(
            lerp_at_game_time(prev, 0.0, current, 1.0, 1.0, lerp_vec),
            current
        );
        assert_eq!(
            lerp_at_game_time(prev, 0.0, current, 1.0, 1.5, lerp_vec),
            current
        );
    }

    #[test]
    fn returns_prev_when_target_at_or_before_prev_time() {
        let prev = DVec3::ZERO;
        let current = DVec3::new(10.0, 20.0, 30.0);
        assert_eq!(
            lerp_at_game_time(prev, 0.0, current, 1.0, 0.0, lerp_vec),
            prev
        );
        assert_eq!(
            lerp_at_game_time(prev, 0.0, current, 1.0, -0.5, lerp_vec),
            prev
        );
    }

    #[test]
    fn returns_current_when_window_is_degenerate() {
        let v = DVec3::new(5.0, 6.0, 7.0);
        assert_eq!(lerp_at_game_time(v, 100.0, v, 100.0, 50.0, lerp_vec), v);
        assert_eq!(lerp_at_game_time(v, 200.0, v, 100.0, 150.0, lerp_vec), v);
    }

    #[test]
    fn midpoint_and_quarter_targets_blend() {
        let prev = DVec3::ZERO;
        let current = DVec3::new(10.0, 20.0, 30.0);
        assert_eq!(
            lerp_at_game_time(prev, 0.0, current, 1.0, 0.5, lerp_vec),
            DVec3::new(5.0, 10.0, 15.0)
        );
        let current4 = DVec3::new(100.0, 200.0, 400.0);
        assert_eq!(
            lerp_at_game_time(prev, 0.0, current4, 4.0, 1.0, lerp_vec),
            DVec3::new(25.0, 50.0, 100.0)
        );
    }

    #[test]
    fn closure_only_called_on_the_blend_branch() {
        use std::cell::Cell;
        let called = Cell::new(0u32);
        let counting = |a: DVec3, b: DVec3, t: f64| -> DVec3 {
            called.set(called.get() + 1);
            a.lerp(b, t)
        };
        let v = DVec3::ONE;
        let _ = lerp_at_game_time(v, 0.0, v, 0.0, 0.0, counting); // degenerate
        let _ = lerp_at_game_time(v, 0.0, v, 1.0, 1.0, counting); // at/after current
        let _ = lerp_at_game_time(v, 0.0, v, 1.0, 0.0, counting); // at/before prev
        assert_eq!(called.get(), 0);
        let _ = lerp_at_game_time(v, 0.0, v, 1.0, 0.5, counting); // blend
        assert_eq!(called.get(), 1);
    }

    // ---- EntityTrack ------------------------------------------------------------

    #[test]
    fn a_fresh_track_freezes_at_its_only_pose() {
        let track = EntityTrack::new(pose_at(10, 5.0));
        // Degenerate window: any cursor returns the pose unchanged.
        assert_eq!(track.sample(10.0).pos, DVec3::new(5.0, 0.0, 0.0));
        assert_eq!(track.sample(7.0).pos, DVec3::new(5.0, 0.0, 0.0));
        assert_eq!(track.newest_tick(), UniverseTick(10));
    }

    #[test]
    fn a_strictly_newer_tick_advances_the_window_and_lerps() {
        let mut track = EntityTrack::new(pose_at(10, 0.0));
        track.observe(pose_at(12, 10.0)); // shift: prev=tick10/x0, current=tick12/x10
        assert_eq!(track.newest_tick(), UniverseTick(12));
        // Cursor at the window midpoint (tick 11) → halfway.
        assert_eq!(track.sample(11.0).pos, DVec3::new(5.0, 0.0, 0.0));
        // Before the window → prev; at/after → current.
        assert_eq!(track.sample(10.0).pos, DVec3::new(0.0, 0.0, 0.0));
        assert_eq!(track.sample(12.0).pos, DVec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn a_same_tick_sibling_updates_in_place_without_collapsing_the_window() {
        let mut track = EntityTrack::new(pose_at(10, 0.0));
        track.observe(pose_at(12, 10.0)); // window [10,12]
        // A sibling at the SAME tick 12 (e.g. a re-observed pose) must NOT collapse
        // the window to [12,12] (which would kill interpolation, audit finding-25).
        track.observe(pose_at(12, 9.0));
        assert_eq!(track.newest_tick(), UniverseTick(12));
        // The window is still [10,12]; tick-11 cursor still lerps (toward the
        // updated current x=9).
        assert_eq!(track.sample(11.0).pos, DVec3::new(4.5, 0.0, 0.0));
    }

    #[test]
    fn a_frame_change_collapses_the_window_instead_of_blending_across_frames() {
        let mut track = EntityTrack::new(pose_at(10, 0.0)); // SystemSpace
        track.observe(pose_at(12, 10.0)); // same frame → window [10, 12]
        // A pose in a DIFFERENT frame arrives.
        let mut other = pose_at(14, 99.0);
        other.frame = FrameRef::PlanetCentered { planet_seed: 5 };
        track.observe(other);
        // The window collapsed: sampling freezes at the new-frame pose (x=99), never
        // a garbage blend between the SystemSpace x and the PlanetCentered x.
        assert_eq!(track.sample(13.0).pos, DVec3::new(99.0, 0.0, 0.0));
        assert_eq!(
            track.sample(14.0).frame,
            FrameRef::PlanetCentered { planet_seed: 5 }
        );
        assert_eq!(track.newest_tick(), UniverseTick(14));
    }

    #[test]
    fn current_frame_tracks_the_leading_edge_pose_frame() {
        let mut track = EntityTrack::new(pose_at(10, 0.0)); // SystemSpace { 1 }
        assert_eq!(
            track.current_frame(),
            FrameRef::SystemSpace { system_seed: 1 }
        );
        let mut other = pose_at(12, 9.0);
        other.frame = FrameRef::PlanetCentered { planet_seed: 5 };
        track.observe(other); // frame change → leading edge is the new frame
        assert_eq!(
            track.current_frame(),
            FrameRef::PlanetCentered { planet_seed: 5 }
        );
    }

    #[test]
    fn no_extrapolation_past_the_newest_tick_under_loss() {
        let mut track = EntityTrack::new(pose_at(10, 0.0));
        track.observe(pose_at(12, 10.0));
        // The stream stalls; the cursor runs WELL past the newest tick. The entity
        // must FREEZE at current (x=10), never project past it on velocity.
        assert_eq!(track.sample(50.0).pos, DVec3::new(10.0, 0.0, 0.0));
        assert_eq!(track.sample(1000.0).pos, DVec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn orientation_slerps_within_the_window() {
        let mut a = StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 1 },
            DVec3::ZERO,
            UniverseTick(10),
        );
        a.orient = DQuat::IDENTITY;
        let mut b = a;
        b.universe_tick = UniverseTick(12);
        b.orient = DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2);
        let mut track = EntityTrack::new(a);
        track.observe(b);
        // Midpoint orientation is between identity and 90°-about-Y, and unit-length.
        let mid = track.sample(11.0).orient;
        assert!(
            (mid.length() - 1.0).abs() < 1e-9,
            "slerp preserves unit length"
        );
        assert!(
            mid.to_axis_angle().1 > 0.0,
            "rotated partway toward current"
        );
    }

    #[test]
    fn two_entities_at_the_same_universe_tick_sample_at_the_same_alpha() {
        // Oscillation guard (audit finding-20): every track keys on the SHARED
        // universe_tick timeline, so two entities advanced over the same tick window
        // sample consistently at one cursor — never a per-source-tick mismatch.
        let mut a = EntityTrack::new(pose_at(10, 0.0));
        let mut b = EntityTrack::new(pose_at(10, 100.0));
        a.observe(pose_at(14, 40.0)); // window [10,14], span 4
        b.observe(pose_at(14, 200.0));
        // Cursor at tick 11 → alpha 0.25 for BOTH.
        assert_eq!(a.sample(11.0).pos, DVec3::new(10.0, 0.0, 0.0)); // 0 + 0.25*40
        assert_eq!(b.sample(11.0).pos, DVec3::new(125.0, 0.0, 0.0)); // 100 + 0.25*100
    }
}
