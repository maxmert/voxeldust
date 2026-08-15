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

use glam::{DQuat, DVec3, I64Vec3};
use vd_core::UniverseTick;
use vd_core::pose::{FrameRef, StampedPose, Tier};

use crate::tick_to_f64;

/// THE ONE PLACE IN THE CLIENT WHERE A LABEL BECOMES A UNIT — and it reads a label the SHIPPER
/// attached to the very value it describes, which is the whole difference between being told and
/// guessing.
///
/// A `LatticePos` counts its integer cell in a unit that depends on the tier of the space the value is
/// measured in: millimetre-ish quanta inside a star system, light-years out at galaxy scale. Get that
/// wrong by one tier and every cell is off by a factor of 10^19 — which is why the client must never
/// pick it. `StampedPose::frame` is the sender's statement of the space its `pos` is measured in: the
/// emitting shard labels its own rows with its own frame, and every relay hop that restates a value
/// re-labels it in the same operation (`vd_core::frame::transfer_frame` writes `frame: to` and routes
/// the cell through `LatticePos::convert_tier`). So the tier read here is a value that was SHIPPED,
/// not one the renderer inferred from context.
///
/// It is called at exactly three ingress points — [`EntityTrack::observe`] (are two delivered poses
/// even commensurable?), [`EntityTrack::sample`] and [`EntityTrack::current_render_pose`] (stamp the
/// unit onto the [`RenderPose`] that carries the value onward). Everything downstream — `world_pos`,
/// `RealmBox::draw_center`, the capture camera, the containment verdicts, the HR6 diagnosis rows —
/// multiplies by the unit it was HANDED and never looks a frame up again.
#[must_use]
pub fn stated_tier(frame: FrameRef) -> Tier {
    frame.tier()
}

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
    /// The space the shipper said this value is measured in — carried for identity/diagnosis
    /// (the player-location readout reads it), NEVER re-consulted for arithmetic. The UNIT that
    /// arithmetic needs is `tier`, stamped beside it at the same instant from the same statement.
    pub frame: FrameRef,
    /// Integer CELL anchor of the rendered position, counted in `tier` units — the exact-integer coarse
    /// part carried alongside the `pos` offset (the tiered-i64 base, D-41). Through P3 it is `ZERO` and
    /// `pos` carries the full frame-local metres (byte-identical to the pre-S1 bare-`DVec3` pose); the
    /// flattening in `world_pos` scales it by `tier`. The interpolation blend rebases into this cell
    /// (see `sample`).
    pub cell: I64Vec3,
    /// Frame-local offset within `cell`. What the renderer draws / `world_pos` consumes.
    pub pos: DVec3,
    pub orient: DQuat,
    /// THE UNIT `cell` IS COUNTED IN, as stated by whoever shipped this value — captured ONCE by
    /// [`stated_tier`] at the moment the delivered pose became a render pose, and multiplied by
    /// downstream without any further lookup.
    ///
    /// It is a field rather than a `frame.tier()` call at each use site because the two are only the
    /// same thing while every label happens to name the space its value is in. Reading the tier at the
    /// point of DRAWING re-opens that question at every call site; reading it once, next to the value
    /// it belongs to, closes it. A drawn point is then `cell * tier.cell_edge_m() + pos` — a
    /// multiplication by a number the client was handed.
    pub tier: Tier,
}

/// How many delivered poses one track retains (SLICE 6).
///
/// WHY A RING AND NOT TWO POSES. The render cursor deliberately sits `buffer_ticks` BEHIND the
/// freshest delivered tick — that delay is what guarantees "there is ALWAYS a newer snapshot to
/// interpolate toward" (see [`crate::tuning::ClientInterpTuning::interp_buffer_ms`]). Two poses span
/// ONE tick, so the cursor was always OLDER than the oldest pose retained, [`lerp_at_game_time`] took
/// its `target <= prev_time` branch every frame, and NOTHING was ever interpolated. The machinery was
/// correct and completely inert.
///
/// THE DERIVATION (asserted by `depth_spans_every_supported_buffer`, never guessed). To bracket the
/// cursor the retained span must EXCEED the buffer. With one pose per tick, `N` poses span `N-1`
/// ticks, so the requirement is `N - 1 > interp_buffer_ms/1000 * tick_hz` for every configuration the
/// server can impose. The client learns `tick_hz` from the wire (`ServerControlMsg::UniverseRate`), so
/// the bound is taken over the SUPPORTED range, not the default: 150 ms at 50 Hz = 7.5 ticks is the
/// worst case, and 12 poses span 11 ticks — comfortably clear with room for a dropped snapshot
/// widening the gaps. 12 × 144 B = 1.7 KB per track; at a thousand entities that is 1.7 MB.
pub const TRACK_POSES: usize = 12;

/// Which branch [`EntityTrack::sample`] takes at a cursor — the SLICE 6 S5 diagnosis signal.
///
/// This exists because the shake was invisible for so long: the interpolation machinery was correct
/// and completely inert, and nothing reported that. A feed sitting permanently on `ClampedOld` means
/// the cursor is falling behind the retained history — exactly the condition that WAS the shake.
/// `ClampedNew` means the feed has stalled and the entity is frozen (required, never coasting).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleWindow {
    /// The cursor sits BETWEEN two retained poses — interpolation is genuinely running.
    Blended,
    /// The cursor precedes the whole ring; the sample clamps to the oldest retained pose.
    ClampedOld,
    /// The cursor is at or past the newest retained pose; the entity is FROZEN there.
    ClampedNew,
}

/// Tally an iterator of [`SampleWindow`]s into `(blended, clamped_old, clamped_new)` — the ONE place
/// the three-way count lives, shared by both feeds so their numbers are directly comparable.
#[must_use]
pub fn census(windows: impl Iterator<Item = SampleWindow>) -> (u32, u32, u32) {
    let mut blended = 0;
    let mut old = 0;
    let mut new = 0;
    for w in windows {
        match w {
            SampleWindow::Blended => blended += 1,
            SampleWindow::ClampedOld => old += 1,
            SampleWindow::ClampedNew => new += 1,
        }
    }
    (blended, old, new)
}

/// A per-entity interpolation buffer keyed on the shared `universe_tick` timeline: a fixed ring of the
/// last [`TRACK_POSES`] delivered poses, newest last. Sampling BETWEEN the two that bracket the render
/// cursor gives smooth motion without prediction — past the newest the entity FREEZES (never coasts),
/// before the oldest it clamps (the history ran out; the client does not invent).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EntityTrack {
    /// The ring. Only the first `len` slots starting at `head` are meaningful.
    poses: [StampedPose; TRACK_POSES],
    /// Index of the OLDEST retained pose.
    head: usize,
    /// How many slots are live (`1..=TRACK_POSES`; never 0 — a track is born with one pose).
    len: usize,
    last_universe_tick: UniverseTick,
}

impl EntityTrack {
    /// First sighting: a degenerate window (prev == current) — sampling returns the
    /// pose unchanged until a second snapshot arrives.
    #[must_use]
    pub fn new(first: StampedPose) -> EntityTrack {
        EntityTrack {
            poses: [first; TRACK_POSES],
            head: 0,
            len: 1,
            last_universe_tick: first.universe_tick,
        }
    }

    /// Ring slot of the `i`-th retained pose, oldest-first (`i < self.len`). Monomorphic, branchless.
    #[must_use]
    fn slot(&self, i: usize) -> StampedPose {
        self.poses[(self.head + i) % TRACK_POSES]
    }

    /// The newest retained pose (the leading edge).
    #[must_use]
    fn newest(&self) -> StampedPose {
        self.slot(self.len - 1)
    }

    /// Append a strictly-newer pose, evicting the oldest once full.
    fn push(&mut self, pose: StampedPose) {
        if self.len < TRACK_POSES {
            self.poses[(self.head + self.len) % TRACK_POSES] = pose;
            self.len += 1;
            return;
        }
        // Full: overwrite the oldest slot and advance the head — it becomes the newest.
        self.poses[self.head] = pose;
        self.head = (self.head + 1) % TRACK_POSES;
    }

    /// Replace the newest pose in place (a sibling chunk of the SAME tick — audit finding-25: this must
    /// NOT collapse the window, so it never touches `head`/`len`).
    fn replace_newest(&mut self, pose: StampedPose) {
        let slot = (self.head + self.len - 1) % TRACK_POSES;
        self.poses[slot] = pose;
    }

    /// Index of the newest retained pose at or before `cursor`, or `None` when the cursor precedes the
    /// whole ring. Monomorphic linear scan from the newest end — the answer is at or near the tail
    /// under normal play (the cursor sits a couple of ticks back), and `TRACK_POSES` is 12.
    #[must_use]
    fn bracket_start(&self, cursor: f64) -> Option<usize> {
        let mut i = self.len;
        while i > 0 {
            i -= 1;
            if tick_to_f64(self.slot(i).universe_tick) <= cursor {
                return Some(i);
            }
        }
        None
    }

    /// Fold in a delivered pose. A STRICTLY newer `universe_tick` shifts
    /// `current → prev` and advances the window; an equal tick (a sibling chunk of
    /// the same tick) updates `current` in place WITHOUT collapsing the window, so
    /// the interpolation span is preserved (audit finding-25).
    ///
    /// THE FRAME-STABILITY GUARD (crossing-render review, CRITICAL finding): a non-strictly-newer
    /// row whose FRAME differs from the newest is IGNORED, never folded. The in-place update exists
    /// for sibling chunks of one tick, which are by definition one shard's one emission — one frame.
    /// A same-or-older-tick row in a DIFFERENT frame is the other side of a crossing's grace window:
    /// the demoted source keeps emitting the avatar (its retained ghost, frozen at the demote pose,
    /// old stamp) on the still-open old sub, and per-sub `frame_id` gates cannot order across subs —
    /// so without this guard every interleaved old-sub row overwrote the newest slot's FRAME,
    /// flipping the delivered location backward per datagram (the forget-space storm + the drawn
    /// avatar snapping between two spaces). The old comment's premise ("stale poses never reach
    /// here — the §6.3 gate drops them upstream") was true per sub and false across two.
    pub fn observe(&mut self, pose: StampedPose) {
        if stated_tier(pose.frame) != stated_tier(self.newest().frame) {
            // A UNIT CHANGE (Fine <-> Coarse, i.e. in-system <-> galaxy at P10). The test is on the
            // UNIT and not on the frame NAME, because what makes two delivered poses blendable is
            // whether their integer cells are counted in the same quantum: the rebase below scales by
            // ONE tier's `cell_edge_m`, so it cannot express a window whose two ends count in different
            // ones. Collapse to the new pose; the next sample FREEZES there rather than producing a
            // number that is a light-year out per cell.
            //
            // SLICE 6 S3 — this used to fire on any FRAME change, which is both unnecessary and
            // actively harmful. Every in-system frame — `PlanetCentered`, `SystemSpace`, `ShipLocal`,
            // `StationLocal`, `AreaLocal` — is `Tier::Fine`, so a re-home that merely restates a pose
            // in a neighbouring realm's frame changes the name and the numbers but not the lattice they
            // are quantized on, and the window is still perfectly blendable. Collapsing there cost one
            // tick with the two-pose track; with the ring it would throw away the WHOLE buffer and
            // stall for its full depth — a fresh hitch at exactly the moment that matters most (a
            // crossing, boarding, warp).
            *self = EntityTrack::new(pose);
        } else if pose.universe_tick > self.last_universe_tick {
            if pose.frame != self.newest().frame {
                // A SPACE CHANGE (the crossing-render slice's one-space model): under the one-space
                // ingress every folded row is stated in the space the client stands in, so the only
                // legitimate frame change left on a track is the OWN avatar's committed crossing —
                // and a crossing is a CUT between two spaces, not a motion inside one. Blending
                // across it lerps numbers measured from two different origins (MEASURED: the drawn
                // own pose read 19.65 m in the old space beside a new-space box at the origin for
                // the whole interpolation-buffer depth after the flip — the ride gate caught it).
                // Collapse to the new pose; the next samples start clean in the new space. The
                // slice-6 no-collapse rule was right when same-tier relabels of OTHER entities
                // reached this fold; the one-space filter now keeps those out upstream.
                *self = EntityTrack::new(pose);
            } else {
                self.push(pose);
                self.last_universe_tick = pose.universe_tick;
            }
        } else if pose.frame == self.newest().frame {
            self.replace_newest(pose);
        }
        // else: a non-newer row in another frame — the old feed's grace-window straggler; ignored
        // (see the frame-stability guard above). The newest pose, its frame, and the window survive.
    }

    /// The pose to render at `cursor` (a continuous f64 in universe-tick units).
    /// Pos lerps, orientation slerps, and the cursor is clamped to the window —
    /// past the freshest tick the entity FREEZES at `current` (never vel-projected).
    #[must_use]
    pub fn sample(&self, cursor: f64) -> RenderPose {
        // SLICE 6: pick the two retained poses that BRACKET the cursor. Before the whole ring (history
        // ran out, or a just-collapsed window) both ends are the oldest — clamp, never invent. At or
        // past the newest both ends are the newest — FREEZE, never coast. In between, `prev` is the
        // newest pose at-or-before the cursor and `current` the one after it, and the blend below runs
        // for real. `lerp_at_game_time` is UNCHANGED; it simply now receives a window that contains the
        // cursor instead of one that is always ahead of it.
        let start = self.bracket_start(cursor);
        let (prev, current) = match start {
            None => (self.slot(0), self.slot(0)),
            Some(i) if i + 1 < self.len => (self.slot(i), self.slot(i + 1)),
            Some(i) => (self.slot(i), self.slot(i)),
        };
        let prev_time = tick_to_f64(prev.universe_tick);
        let current_time = tick_to_f64(current.universe_tick);
        // Rebase-before-lerp (S1): re-express `prev`'s offset in `current`'s CELL before blending, so the
        // offset lerp is continuous ACROSS a cell boundary — `prev_in_cell = prev.offset + (prev.cell −
        // cell)·edge` (raw, NOT re-normalized, so the blend stays continuous). Through P3 both cells are
        // ZERO ⇒ this is exactly `prev.offset()` and the result rides `cell == ZERO` — byte-identical to
        // the pre-S1 offset-only lerp. The result cell is `current`'s, counted in `current`'s stated unit,
        // and `world_pos` later flattens both together; it SUBTRACTS NOTHING (there is no pinned origin
        // any more — the server ships every value already measured from the realm the client stands in).
        // Using ONE end's unit for both ends is sound only because `observe` collapses the window
        // whenever the unit changes, which is why that guard is on the unit rather than on the name.
        let cell = current.pos.cell();
        let tier = stated_tier(current.frame);
        let edge = tier.cell_edge_m();
        let prev_offset = prev.pos.offset() + (prev.pos.cell() - cell).as_dvec3() * edge;
        let pos = lerp_at_game_time(
            prev_offset,
            prev_time,
            current.pos.offset(),
            current_time,
            cursor,
            DVec3::lerp,
        );
        let orient = lerp_at_game_time(
            prev.orient,
            prev_time,
            current.orient,
            current_time,
            cursor,
            DQuat::slerp,
        );
        RenderPose {
            frame: current.frame,
            cell,
            pos,
            orient,
            tier,
        }
    }

    /// Which branch [`EntityTrack::sample`] takes at `cursor` — read-only diagnosis (S5), derived from
    /// the SAME bracket the sample uses so the two can never disagree.
    #[must_use]
    pub fn window_at(&self, cursor: f64) -> SampleWindow {
        match self.bracket_start(cursor) {
            None => SampleWindow::ClampedOld,
            Some(i) if i + 1 < self.len => SampleWindow::Blended,
            Some(_) => SampleWindow::ClampedNew,
        }
    }

    /// The freshest delivered tick (the window's leading edge).
    #[must_use]
    pub fn newest_tick(&self) -> UniverseTick {
        self.newest().universe_tick
    }

    /// The frame the entity is currently expressed in (the leading edge) — the basis for
    /// the player-location stat. Frame does not interpolate, so no cursor is needed.
    #[must_use]
    pub fn current_frame(&self) -> FrameRef {
        self.newest().frame
    }

    /// The freshest delivered pose as a [`RenderPose`] — the LATEST server-shipped position with NO
    /// interpolation (the leading edge). For consumers that want the latest pose without a render cursor
    /// (the realm-box overlay, FA-2c): a slow moving realm box shows its latest streamed placement per
    /// step; render-side cursor interpolation is an FA-5+ smoothness refinement (D-45).
    #[must_use]
    pub fn current_render_pose(&self) -> RenderPose {
        let n = self.newest();
        RenderPose {
            frame: n.frame,
            cell: n.pos.cell(),
            pos: n.pos.offset(),
            orient: n.orient,
            tier: stated_tier(n.frame),
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

    /// THE FRAME-STABILITY GUARD (crossing-render review, the CRITICAL finding's root fix): a
    /// non-strictly-newer row in a DIFFERENT frame is IGNORED — the newest pose, its frame, and the
    /// window all survive. During a crossing's grace window the demoted source keeps emitting the
    /// avatar (its retained ghost, frozen stamp, old frame) on the still-open old sub; before this
    /// guard every such row overwrote the newest slot's FRAME and flipped the delivered location
    /// backward per datagram. A same-frame equal-tick sibling chunk still updates in place.
    #[test]
    fn a_non_newer_row_in_another_frame_is_ignored_never_folded() {
        let planet = FrameRef::PlanetCentered { planet_seed: 7 };
        let mut track = EntityTrack::new(pose_at(10, 1.0));
        // The crossing: a strictly-newer row in the NEW frame advances the window (same tier —
        // the slice-6 no-collapse rule).
        track.observe(StampedPose::at_rest(
            planet,
            DVec3::new(2.0, 0.0, 0.0),
            UniverseTick(11),
        ));
        assert_eq!(track.current_frame(), planet, "the crossing row lands");
        // The old feed's straggler: EQUAL tick, OLD frame — ignored (frame + pose survive).
        track.observe(pose_at(11, 9.0));
        assert_eq!(
            track.current_frame(),
            planet,
            "an equal-tick old-frame row never flips the frame back"
        );
        assert_eq!(track.sample(11.0).pos, DVec3::new(2.0, 0.0, 0.0));
        // OLDER tick, old frame — ignored too (the frozen retained-ghost row).
        track.observe(pose_at(10, 9.0));
        assert_eq!(track.current_frame(), planet);
        // A same-frame equal-tick sibling chunk still updates in place (the replace_newest arm).
        track.observe(StampedPose::at_rest(
            planet,
            DVec3::new(3.0, 0.0, 0.0),
            UniverseTick(11),
        ));
        assert_eq!(track.sample(11.0).pos, DVec3::new(3.0, 0.0, 0.0));
    }

    // ---- SLICE 6 (THE SHAKE): the buffer and the history contradict each other ----

    /// GREEN (slice 6 S2 — this was the RED characterization test; the assertion is now INVERTED).
    /// Same live timing: consecutive delivery, cursor `buffer_ticks` behind the freshest tick. With a
    /// ring deep enough to span the buffer the cursor now falls BETWEEN two retained poses and the
    /// blend runs for real. Before S2 this returned the oldest pose unblended, every frame, forever.
    #[test]
    fn slice6_the_cursor_now_lands_inside_the_window_and_the_blend_actually_runs() {
        let buffer = crate::tuning::ClientInterpTuning::DEFAULT.buffer_ticks();
        // Ten consecutive ticks at 1 m/tick, freshest = 100.
        let mut track = EntityTrack::new(pose_at(91, 91.0));
        for t in 92..=100 {
            track.observe(pose_at(t, t as f64));
        }
        let cursor = 100.0 - buffer; // 97.6
        let drawn = track.sample(cursor).pos;
        assert_eq!(cursor, 97.6);
        // STRICTLY between the poses at 97 and 98 — a real blend, not a clamp to either endpoint.
        assert!(drawn.x > 97.0, "blended past the older endpoint: {drawn}");
        assert!(
            drawn.x < 98.0,
            "blended short of the newer endpoint: {drawn}"
        );
        // Uniform motion => the blend is exactly the cursor.
        assert!((drawn.x - 97.6).abs() < 1e-9, "expected 97.6, got {drawn}");
    }

    /// SLICE 6 S2 — the DERIVATION behind [`TRACK_POSES`], asserted rather than assumed. `N` poses span
    /// `N-1` ticks at one pose per tick; that span must EXCEED the buffer for the cursor to be
    /// bracketed. Checked against the default AND the worst configuration the server can impose over
    /// the wire (150 ms at 50 Hz). If someone raises the buffer or the tick rate past this, THIS fails
    /// — which is the point.
    #[test]
    fn slice6_depth_spans_every_supported_buffer() {
        let span = (TRACK_POSES - 1) as f64;
        let default = crate::tuning::ClientInterpTuning::DEFAULT.buffer_ticks();
        assert!(
            span > default,
            "{span} ticks must exceed the {default}-tick default buffer"
        );
        let worst = crate::tuning::ClientInterpTuning {
            interp_buffer_ms: 150.0,
            tick_hz: 50.0,
        };
        assert_eq!(worst.buffer_ticks(), 7.5);
        assert!(
            span > 7.5,
            "{span} ticks must exceed the 7.5-tick worst case"
        );
    }

    /// SLICE 6 S2 — past the newest retained pose the entity FREEZES (the no-prediction mandate: it
    /// must never coast on velocity).
    #[test]
    fn slice6_a_cursor_past_the_newest_pose_freezes_and_never_coasts() {
        let mut track = EntityTrack::new(pose_at(10, 10.0));
        track.observe(pose_at(11, 11.0));
        assert_eq!(track.sample(11.0).pos, DVec3::new(11.0, 0.0, 0.0));
        assert_eq!(track.sample(50.0).pos, DVec3::new(11.0, 0.0, 0.0));
    }

    /// SLICE 6 S2 — before the whole ring the sample CLAMPS to the oldest retained pose. The history
    /// ran out; the client holds rather than inventing a position it was never told.
    #[test]
    fn slice6_a_cursor_before_the_whole_ring_clamps_to_the_oldest() {
        let mut track = EntityTrack::new(pose_at(10, 10.0));
        track.observe(pose_at(11, 11.0));
        assert_eq!(track.sample(1.0).pos, DVec3::new(10.0, 0.0, 0.0));
    }

    /// SLICE 6 S2 — the ring WRAPS: after more than `TRACK_POSES` observations the oldest are evicted
    /// and the newest survive, with the bracket still correct across the wrap point.
    #[test]
    fn slice6_the_ring_wraps_and_keeps_the_newest_poses() {
        let mut track = EntityTrack::new(pose_at(0, 0.0));
        for t in 1..=(TRACK_POSES as u64 + 5) {
            track.observe(pose_at(t, t as f64));
        }
        let newest = TRACK_POSES as u64 + 5;
        assert_eq!(track.newest_tick(), UniverseTick(newest));
        // The oldest surviving pose is `TRACK_POSES - 1` ticks back; anything older clamps to it.
        let oldest = newest - (TRACK_POSES as u64 - 1);
        assert_eq!(track.sample(0.0).pos, DVec3::new(oldest as f64, 0.0, 0.0));
        // A blend ACROSS the wrap point still interpolates correctly.
        let mid = oldest as f64 + 0.5;
        assert!((track.sample(mid).pos.x - mid).abs() < 1e-9);
    }

    /// SLICE 6 S2 — a sibling chunk of the SAME tick replaces the newest pose IN PLACE and does NOT
    /// collapse the window (audit finding-25). The older history must survive so the blend keeps its
    /// span.
    #[test]
    fn slice6_an_equal_tick_sibling_chunk_replaces_in_place_without_losing_history() {
        let mut track = EntityTrack::new(pose_at(10, 10.0));
        track.observe(pose_at(11, 11.0));
        track.observe(pose_at(11, 99.0)); // same tick, corrected value
        assert_eq!(track.newest_tick(), UniverseTick(11));
        assert_eq!(track.sample(11.0).pos, DVec3::new(99.0, 0.0, 0.0));
        // The tick-10 pose is still there — the window did not collapse.
        assert_eq!(track.sample(10.0).pos, DVec3::new(10.0, 0.0, 0.0));
    }

    /// SLICE 6 S2 — a single-sample track is degenerate: every cursor returns that one pose.
    #[test]
    fn slice6_a_single_sample_track_returns_that_pose_at_any_cursor() {
        let track = EntityTrack::new(pose_at(10, 10.0));
        assert_eq!(track.sample(0.0).pos, DVec3::new(10.0, 0.0, 0.0));
        assert_eq!(track.sample(10.0).pos, DVec3::new(10.0, 0.0, 0.0));
        assert_eq!(track.sample(99.0).pos, DVec3::new(10.0, 0.0, 0.0));
    }

    /// SLICE 6 S2 — WHY DEPTH IS THE FIX, not the ring mechanism. This is the original RED
    /// characterization test, kept: with only TWO poses retained the window still spans one tick and
    /// the cursor still falls behind it, so the sample still clamps to the oldest. The ring changes
    /// nothing on its own — it is the DEPTH that lets the cursor be bracketed. If someone shrinks
    /// `TRACK_POSES` back toward 2, `slice6_depth_spans_every_supported_buffer` fails and this test
    /// explains why that matters.
    #[test]
    fn slice6_a_two_pose_history_still_clamps_which_is_why_depth_is_the_fix() {
        let buffer = crate::tuning::ClientInterpTuning::DEFAULT.buffer_ticks();
        let mut track = EntityTrack::new(pose_at(99, 0.0));
        track.observe(pose_at(100, 10.0));
        let cursor = 100.0 - buffer;
        assert_eq!(cursor, 97.6);
        assert_eq!(
            track.sample(cursor).pos,
            DVec3::new(0.0, 0.0, 0.0),
            "two poses span ONE tick, so a {buffer}-tick cursor is still behind them both",
        );
    }

    /// SLICE 6 S5 — the window classification MATCHES what `sample` actually did. A diagnosis signal
    /// that can disagree with the thing it reports on is worse than none, so both are derived from the
    /// same bracket and this pins that they agree on all three branches.
    #[test]
    fn slice6_the_reported_window_matches_what_sample_actually_did() {
        let mut track = EntityTrack::new(pose_at(10, 10.0));
        for t in 11..=14 {
            track.observe(pose_at(t, t as f64));
        }
        // Blended: strictly inside, and the drawn value is strictly between two poses.
        // (Two asserts, not one `&&` — a short-circuit inside an assert is an uncoverable
        // branch arm, HR5.)
        assert_eq!(track.window_at(12.5), SampleWindow::Blended);
        let drawn = track.sample(12.5).pos.x;
        assert!(drawn > 12.0);
        assert!(drawn < 13.0);
        // Clamped old: before the ring, drawn == the oldest.
        assert_eq!(track.window_at(1.0), SampleWindow::ClampedOld);
        assert_eq!(track.sample(1.0).pos.x, 10.0);
        // Clamped new: at/past the newest, drawn == the newest (frozen).
        assert_eq!(track.window_at(14.0), SampleWindow::ClampedNew);
        assert_eq!(track.window_at(99.0), SampleWindow::ClampedNew);
        assert_eq!(track.sample(99.0).pos.x, 14.0);
    }

    /// SLICE 6 S5 — the census tallies the three classes.
    #[test]
    fn slice6_census_tallies_each_class() {
        let windows = [
            SampleWindow::Blended,
            SampleWindow::ClampedOld,
            SampleWindow::Blended,
            SampleWindow::ClampedNew,
            SampleWindow::Blended,
        ];
        assert_eq!(census(windows.into_iter()), (3, 1, 1));
        assert_eq!(census([].into_iter()), (0, 0, 0));
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
    fn a_tier_change_collapses_the_window_instead_of_blending_across_lattices() {
        let mut track = EntityTrack::new(pose_at(10, 0.0)); // SystemSpace — Fine
        track.observe(pose_at(12, 10.0)); // same frame → window [10, 12]
        // A pose on the COARSE lattice arrives (galaxy scale, P10 warp).
        let mut other = pose_at(14, 99.0);
        other.frame = FrameRef::GalaxySpace;
        track.observe(other);
        // Collapsed: sampling freezes at the new pose (x=99), never a blend between two lattices
        // whose integer cells count in different units.
        assert_eq!(track.sample(13.0).pos, DVec3::new(99.0, 0.0, 0.0));
        assert_eq!(track.sample(14.0).frame, FrameRef::GalaxySpace);
        // …and the sample carries the NEW unit, so whatever draws it scales the cell in light-years and
        // not in millimetres. Before the unit rode the pose, this was re-derived at the drawing site.
        assert_eq!(track.sample(14.0).tier, Tier::Coarse);
        assert_eq!(track.newest_tick(), UniverseTick(14));
    }

    /// THE UNIT IS STAMPED FROM THE LABEL THE SHIPPER PUT ON THIS VALUE, at the one ingress point, and
    /// then travels with it. Both sampling paths do it, so the interpolated pose and the leading-edge
    /// pose (the realm-box overlay reads that one) can never disagree about how big a cell is.
    #[test]
    fn the_stated_unit_rides_every_render_pose_from_both_sampling_paths() {
        let mut track = EntityTrack::new(pose_at(10, 0.0)); // SystemSpace — Fine
        track.observe(pose_at(11, 10.0));
        assert_eq!(track.sample(10.5).tier, Tier::Fine);
        assert_eq!(track.current_render_pose().tier, Tier::Fine);
        // A COARSE-lattice feed stamps Coarse on both.
        let mut coarse = pose_at(12, 5.0);
        coarse.frame = FrameRef::GalaxySpace;
        let coarse_track = EntityTrack::new(coarse);
        assert_eq!(coarse_track.sample(12.0).tier, Tier::Coarse);
        assert_eq!(coarse_track.current_render_pose().tier, Tier::Coarse);
    }

    /// A frame change on a track is a SPACE CHANGE and COLLAPSES the window (the crossing-render
    /// slice's one-space model — this REWRITES the slice-6 "same-tier relabel keeps the window"
    /// contract). Positions are no longer root-absolute: every value is measured from the realm the
    /// session stands in, so two frames on one track means two ORIGINS, and blending across them
    /// lerps numbers from different spaces (MEASURED: the drawn own pose read the old space's 19.65 m
    /// beside a new-space box at the origin for the whole buffer depth after a crossing flip — the
    /// round-trip ride gate caught it). Under the one-space ingress, other entities' relabels never
    /// reach this fold; the own avatar's crossing is the one legitimate frame change, and it is a cut.
    #[test]
    fn a_frame_change_collapses_the_window_to_the_new_space() {
        let mut track = EntityTrack::new(pose_at(10, 10.0)); // SystemSpace — Fine
        track.observe(pose_at(11, 11.0));
        // The crossing: same tier, NEW space, strictly newer — the window collapses to the new pose.
        let mut crossed = pose_at(12, 0.4);
        crossed.frame = FrameRef::PlanetCentered { planet_seed: 5 };
        assert_eq!(
            crossed.frame.tier(),
            FrameRef::SystemSpace { system_seed: 1 }.tier(),
        );
        track.observe(crossed);
        // NO blend across the cut: a cursor BEHIND the crossing row clamps to the new space's pose —
        // never to the old space's numbers, and never to a lerp of the two.
        assert!(
            (track.sample(10.5).pos.x - 0.4).abs() < 1e-9,
            "the old space's history is gone; the sample clamps to the new space"
        );
        assert!((track.sample(12.0).pos.x - 0.4).abs() < 1e-9);
        // The label tracks the leading edge, so the player's location readout flips with the cut.
        assert_eq!(
            track.current_frame(),
            FrameRef::PlanetCentered { planet_seed: 5 }
        );
        // And the window REFILLS in the new space: the next same-frame row blends normally again.
        let mut next = pose_at(13, 1.4);
        next.frame = FrameRef::PlanetCentered { planet_seed: 5 };
        track.observe(next);
        assert!(
            (track.sample(12.5).pos.x - 0.9).abs() < 1e-9,
            "blending resumes inside the new space"
        );
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

    // ---- S1 floating-origin: the cell rides the render pose (project_floating_origin_plan.md) ----

    #[test]
    fn sample_at_cell_zero_carries_a_zero_cell_and_the_plain_offset_blend() {
        // Byte-floor: both poses at cell 0 (the P3 shipping form) ⇒ the rebase is a no-op and the result
        // is the plain offset lerp with cell 0 — unchanged from the pre-S1 bare-DVec3 behaviour.
        let mut track = EntityTrack::new(pose_at(10, 0.0));
        track.observe(pose_at(11, 10.0));
        let mid = track.sample(10.5);
        assert_eq!(mid.cell, I64Vec3::ZERO);
        assert!((mid.pos.x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn sample_rebases_prev_into_currents_cell_so_the_blend_is_continuous_across_a_boundary() {
        use vd_core::pose::{FINE_CELL_EDGE_M, LatticePos};
        // `prev` sits one cell BELOW `current`, near the top of its cell: true position = −1·edge +
        // 0.9·edge = −0.1·edge. `current` is at cell 0, offset 0.1·edge. Rebasing prev into cell 0 gives
        // offset −0.1·edge, so the blend stays continuous across the boundary (no jump). Edge-relative, so
        // it holds at the real FINE edge (2⁻¹⁰ m) — use the constant, never a hard-coded literal.
        let edge = FINE_CELL_EDGE_M;
        let sys = FrameRef::SystemSpace { system_seed: 1 };
        let prev = StampedPose {
            frame: sys,
            pos: LatticePos::at(I64Vec3::new(-1, 0, 0), DVec3::new(0.9 * edge, 0.0, 0.0)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(10),
        };
        let current = StampedPose {
            frame: sys,
            pos: LatticePos::at(I64Vec3::ZERO, DVec3::new(0.1 * edge, 0.0, 0.0)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(11),
        };
        let mut track = EntityTrack::new(prev);
        track.observe(current);
        // At the prev edge → prev rebased into current's cell: offset −0.1·edge, cell 0.
        let at_prev = track.sample(10.0);
        assert_eq!(at_prev.cell, I64Vec3::ZERO);
        assert!((at_prev.pos.x - (-0.1 * edge)).abs() < 1e-12);
        // At the current edge → current's own offset, cell 0.
        let at_current = track.sample(11.0);
        assert_eq!(at_current.cell, I64Vec3::ZERO);
        assert!((at_current.pos.x - 0.1 * edge).abs() < 1e-12);
        // Midpoint → the true geometric midpoint (0.0), no cell-boundary jump.
        let mid = track.sample(10.5);
        assert!((mid.cell.x as f64 * edge + mid.pos.x).abs() < 1e-12);
    }

    #[test]
    fn current_render_pose_carries_the_leading_edge_cell_and_offset() {
        use vd_core::pose::LatticePos;
        let p = StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 1 },
            pos: LatticePos::at(I64Vec3::new(3, -4, 5), DVec3::new(0.25, 0.0, 0.0)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(7),
        };
        let rp = EntityTrack::new(p).current_render_pose();
        assert_eq!(rp.cell, I64Vec3::new(3, -4, 5));
        assert_eq!(rp.pos, DVec3::new(0.25, 0.0, 0.0));
    }
}
