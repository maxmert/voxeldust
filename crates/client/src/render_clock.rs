//! The continuous render clock: a smooth f64 cursor over the integer universe
//! tick, so 20 Hz snapshots render as continuous motion.
//!
//! It anchors on the freshest delivered `universe_tick` AND the wall-time it
//! arrived (fed IN by the render loop as seconds — the lib reads no clock, per the
//! seam), then projects the cursor forward at the configured tick rate, held
//! `buffer_ticks` BEHIND the freshest tick so under normal 20 Hz delivery there is
//! always a newer snapshot to interpolate toward.
//!
//! The anchor TICK only ever advances — a stale or equal tick is ignored. That
//! keeps the anchor monotonic, but NOT the projected cursor under heavy loss: if a
//! run of ticks is lost the cursor coasts forward, and when a fresh tick finally
//! arrives the re-anchor can step the cursor BACK by up to the lost span (a visible
//! micro-stutter). This is never an extrapolation — `EntityTrack::sample` clamps
//! the cursor into each window, so the worst case is a freeze, never a
//! vel-projected guess (the no-prediction mandate holds). Smoothing this with a
//! cursor SLEW (spreading the correction across frames) is a Slice-3 render-tuning
//! refinement, best validated against the real render loop and real packet loss.

use vd_core::UniverseTick;

use crate::tick_to_f64;
use crate::tuning::ClientInterpTuning;

/// Projects the render cursor from the freshest delivered tick.
#[derive(Clone, Copy, Debug)]
pub struct RenderClock {
    tuning: ClientInterpTuning,
    /// `(freshest tick, wall-seconds it was observed)`; `None` until the first tick.
    anchor: Option<(UniverseTick, f64)>,
}

impl RenderClock {
    #[must_use]
    pub fn new(tuning: ClientInterpTuning) -> RenderClock {
        RenderClock {
            tuning,
            anchor: None,
        }
    }

    /// Update the universe-tick rate (learned from the wire via `UniverseRate`) in
    /// place, preserving the current anchor so the cursor stays continuous across the
    /// change. Replaces the lib default with the cluster's actual rate (R1).
    pub fn set_tick_hz(&mut self, hz: f64) {
        self.tuning.tick_hz = hz;
    }

    /// Observe the freshest delivered `universe_tick` at wall-time `now_s`. Re-anchors
    /// ONLY on a strictly newer tick — a stale or equal tick keeps the existing
    /// anchor (the anchor tick only advances). NOTE: this keeps the anchor TICK
    /// monotonic, not the projected cursor under heavy loss — see the module note.
    pub fn observe(&mut self, tick: UniverseTick, now_s: f64) {
        let newer = match self.anchor {
            Some((anchored, _)) => tick > anchored,
            None => true,
        };
        if newer {
            self.anchor = Some((tick, now_s));
        }
    }

    /// The render cursor (f64, universe-tick units) at wall-time `now_s`:
    /// `anchored_tick + elapsed·tick_hz − buffer_ticks`. `None` until the first
    /// snapshot anchors it. The cursor is fed to [`crate::interp::EntityTrack::sample`],
    /// which clamps it into each entity's window (so a cursor before/after a window
    /// freezes that entity — no extrapolation).
    #[must_use]
    pub fn cursor(&self, now_s: f64) -> Option<f64> {
        self.anchor.map(|(tick, anchored_s)| {
            let elapsed = now_s - anchored_s;
            tick_to_f64(tick) + elapsed * self.tuning.tick_hz - self.tuning.buffer_ticks()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock() -> RenderClock {
        RenderClock::new(ClientInterpTuning::DEFAULT) // 120 ms @ 20 Hz → 2.4 tick buffer
    }

    #[test]
    fn no_cursor_before_the_first_tick() {
        assert_eq!(clock().cursor(1.0), None);
    }

    #[test]
    fn cursor_sits_buffer_ticks_behind_the_anchor_and_advances_with_wall_time() {
        let mut c = clock();
        c.observe(UniverseTick(100), 10.0);
        // At the anchor instant: 100 - 2.4 buffer.
        assert_eq!(c.cursor(10.0), Some(97.6));
        // 0.5 s later: + 0.5 * 20 Hz = +10 ticks.
        assert_eq!(c.cursor(10.5), Some(107.6));
    }

    #[test]
    fn a_newer_tick_re_anchors_to_track_the_server() {
        let mut c = clock();
        c.observe(UniverseTick(100), 10.0);
        c.observe(UniverseTick(110), 10.5); // a fresh snapshot arrived
        // Now anchored at (110, 10.5): cursor at 10.5 = 110 - 2.4.
        assert_eq!(c.cursor(10.5), Some(107.6));
    }

    #[test]
    fn a_stale_or_equal_tick_never_rewinds_the_anchor() {
        let mut c = clock();
        c.observe(UniverseTick(100), 10.0);
        // An out-of-order older snapshot, and an equal one: both ignored.
        c.observe(UniverseTick(90), 10.2);
        c.observe(UniverseTick(100), 10.3);
        // Still anchored at (100, 10.0): cursor unchanged from the original anchor.
        assert_eq!(c.cursor(10.0), Some(97.6));
    }
}
