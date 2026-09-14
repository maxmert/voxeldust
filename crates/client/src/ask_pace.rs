//! ★ THE PACE OF THE BOUNDED ASK (ruling F9 item 1, and its adversarial review).
//!
//! The bounded ask has four moving parts that are not geometry at all — they are RATES. How fast
//! the builders can build, how fast the eye is really going, how fast the horizon may slide, and
//! how often the descent and the crossfade's materials may follow it. Each one is a small piece of
//! arithmetic with two arms, and each one was a defect the first time it was written. They live
//! HERE, in the Tier-A library, so every arm is a unit test and not a flight.
//!
//! **Example.** The pilot's hull leaves the berth and pushes to 528 m/s over the home planet.
//! [`Throughput`] reads the three workers' capacity (about 195 chunks a second), [`PeakHold`]
//! reads the eye's true speed out of the lead's sawtooth, and [`AskPace`] slides the deliverable
//! horizon in from 869 m to 491 m over about two seconds. The DESCENT follows only when the
//! horizon has left the ring it last asked for. When the pilot stops, the hold reads zero within a
//! second and the horizon slides back out.

use crate::ladder_view::{ASK_BOUND_SLACK, AskBound};

/// HOW MANY SLOTS A PEAK HOLD KEEPS. The hold reports the largest reading of the last `hold_s`
/// seconds; it keeps that window in eight slots and drops the oldest as time passes, so what it
/// reports covers at least seven eighths of the window and never more than the whole of it. Eight
/// is what makes "a stop reads zero within the hold" true to an eighth of a second at a
/// one-second hold.
pub const PEAK_SLOTS: usize = 8;

/// ★ THE LARGEST READING OF THE LAST `hold_s` SECONDS (ruling F9 item 1; review item 5).
///
/// Why a hold at all. The lead eye FREEZES at the freshest DELIVERED pose and never coasts
/// (SL10 clause 7), so the lead's metres are a SAWTOOTH: zero the instant a row lands, and the
/// whole buffer's travel just before the next one. MEASURED (§16.2): at 528 m/s the lead reads up
/// to 64 m against a buffer of 0.12 s, so the sawtooth's PEAK is the true speed and its MEAN is
/// about half of it. A mean-read speed halves the ask the bound is sizing.
///
/// Why a ring and not a decay. The first cure was an exponential decay with a one-second constant,
/// and a decay NEVER REACHES ZERO: a hull that stopped dead from 528 m/s still read 194 m/s a
/// second later, and the bound went on coarsening ground the pilot stood still on. A ring of slots
/// is a true maximum over a window, so the hull that stops reads zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PeakHold {
    slots: [f64; PEAK_SLOTS],
    /// The moment the newest slot was opened at. `None` before the first reading.
    opened_s: Option<f64>,
    cursor: usize,
}

impl PeakHold {
    /// THE HOLD'S READING: `reading` goes into the newest slot, slots older than `hold_s` are
    /// dropped, and the largest of what is left is the answer. A hold of zero or less holds
    /// nothing and reads the reading itself.
    pub fn read(&mut self, reading: f64, now_s: f64, hold_s: f64) -> f64 {
        if hold_s <= 0.0 {
            self.slots = [0.0; PEAK_SLOTS];
            self.cursor = 0;
            self.slots[0] = reading;
            self.opened_s = Some(now_s);
            return reading;
        }
        let width_s = hold_s / PEAK_SLOTS as f64;
        let opened = self.opened_s.unwrap_or(now_s);
        // How many slots the clock moved on by. A moment that went backwards moves nothing; a gap
        // longer than the whole window empties it.
        let moves = ((now_s - opened) / width_s).floor().max(0.0);
        let steps = if moves >= PEAK_SLOTS as f64 {
            PEAK_SLOTS
        } else {
            moves as usize
        };
        let mut moved = 0usize;
        while moved < steps {
            self.cursor = (self.cursor + 1) % PEAK_SLOTS;
            self.slots[self.cursor] = 0.0;
            moved += 1;
        }
        self.opened_s = Some(opened + steps as f64 * width_s);
        self.slots[self.cursor] = self.slots[self.cursor].max(reading);
        self.slots.iter().copied().fold(0.0f64, f64::max)
    }
}

/// HOW MUCH OF THE AVERAGE ONE READING MAY EVER CARRY: half. The smoothing weighs a reading by the
/// share of the window it covers, so a reading taken after a long idle would carry the whole weight
/// and THROW THE WINDOW AWAY — the ten-second average would become one sample of one frame's luck
/// (review item 10). A reading carries at most half, so the average always keeps half of what it
/// learned. The FIRST reading is the exception: there is nothing to average it with, so it is
/// taken whole.
pub const THROUGHPUT_ALPHA_MAX: f64 = 0.5;

/// ★ THE BUILDERS' CAPACITY, chunks a second, smoothed (ruling F9 item 1).
///
/// The workers' count over the MEAN WALL TIME of a build — never the chunks they happened to
/// finish. On a walk three workers finish two chunks a second because two is all the ladder asks
/// for, and a bound read from two chunks a second would coarsen a walk that is whole today. The
/// mean build time is the ceiling: three workers at 15 ms a build can finish about 200 chunks a
/// second whether or not the ladder wants them.
///
/// ★ AND THE CARD JOINS IT (ruling F9 item 2, LANDED): the capacity is the threads' own PLUS
/// whatever the card delivers inside its time budget ([`crate::card_budget::CardBudget`]), because
/// the bound sizes the ask against everything that can build. The two are kept apart inside — the
/// threads' mean build time must never be averaged with the card's — and added at the reading.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Throughput {
    /// The counters and the moment of the last reading: chunks built, nanoseconds spent, when.
    mark: Option<(u64, u64, f64)>,
    /// The CPU workers' own smoothed capacity, chunks a second.
    value: f64,
    /// The card's capacity as it last stated itself, chunks a second. Zero where no card builds.
    card: f64,
}

impl Throughput {
    /// THE CAPACITY as it stands after this reading. An interval in which nothing was built
    /// carries no reading, and the last capacity stands.
    pub fn read(
        &mut self,
        chunks_built: u64,
        nanos_built: u64,
        now_s: f64,
        workers: usize,
        window_s: f64,
    ) -> f64 {
        let Some((chunks_at, nanos_at, at_s)) = self.mark else {
            self.mark = Some((chunks_built, nanos_built, now_s));
            return self.both();
        };
        let dt = now_s - at_s;
        let chunks = chunks_built.saturating_sub(chunks_at);
        let nanos = nanos_built.saturating_sub(nanos_at);
        let readable = (dt > 0.0) & (chunks > 0) & (nanos > 0) & (window_s > 0.0);
        if !readable {
            return self.both();
        }
        self.mark = Some((chunks_built, nanos_built, now_s));
        let mean_s = nanos as f64 / chunks as f64 / 1.0e9;
        let capacity = workers as f64 / mean_s;
        // A reading that covers `dt` of the window carries that much of the weight, and never more
        // than `THROUGHPUT_ALPHA_MAX` of it; a FIRST reading is taken whole.
        let share = (dt / window_s).clamp(0.0, THROUGHPUT_ALPHA_MAX);
        let alpha = if self.value > 0.0 { share } else { 1.0 };
        self.value += alpha * (capacity - self.value);
        self.both()
    }

    /// ★ THE CARD'S OWN CAPACITY, as the card last stated it (ruling F9 item 2): chunks a second
    /// inside its time budget. It is STATED, never smoothed here, because the card smooths its own
    /// two times already ([`crate::card_budget::CardBudget`]).
    pub fn set_card(&mut self, card_per_s: f64) {
        self.card = card_per_s.max(0.0);
    }

    /// The capacity of BOTH builders as it stands, without taking a reading (the stamp's readout).
    #[must_use]
    pub fn value(&self) -> f64 {
        self.both()
    }

    /// The card's own share of the capacity.
    #[must_use]
    pub fn card_value(&self) -> f64 {
        self.card
    }

    /// ★ THE CPU WORKERS' OWN CAPACITY, without the card (the owner's step after Step 15). The
    /// stand-down rule reads THIS one ([`crate::card_gate::QueueDepth`]), because its question is
    /// what the queue costs WITHOUT the card: a card counted into its own threshold would keep
    /// itself out of every queue it is good at.
    #[must_use]
    pub fn workers_value(&self) -> f64 {
        self.value
    }

    /// Both builders, added: what the bounded ask sizes its horizon against.
    fn both(&self) -> f64 {
        self.value + self.card
    }
}

/// THIS FRAME'S OWN SECONDS: how far a deliverable horizon may slide. The first frame, and a
/// moment that went backwards, read zero — a horizon that has no time does not move.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameClock {
    at_s: Option<f64>,
}

impl FrameClock {
    /// The seconds since the last frame.
    pub fn seconds(&mut self, now_s: f64) -> f64 {
        let dt = self.at_s.map_or(0.0, |last| (now_s - last).max(0.0));
        self.at_s = Some(now_s);
        dt
    }
}

/// ★ ONE REALM'S HORIZON, AND THE THREE RATES THAT READ IT (ruling F9 item 1, the frame bar).
///
/// The horizon the measurement asks for wanders frame to frame; the horizon IN FORCE slides toward
/// it, because a crossfade band that jumps is a pop. Three things then read the horizon in force,
/// at three rates, and that separation is the whole of the frame-bar cure:
///
/// - THE HELD HORIZON moves every frame, by the slew.
/// - THE DRAWN HORIZON — the one the crossfade's materials carry — follows within
///   [`crate::ladder_view::ASK_BOUND_REBIND`], because rewriting a material makes the engine
///   prepare its bind group again.
/// - THE ASKED HORIZON — the one the DESCENT ran on — follows within
///   [`crate::ladder_view::ASK_BOUND_BRACKET`], because the descent is the costliest thing the
///   terrain system does on the main thread (MEASURED, §25.7: 9.7 ms a frame, and following the
///   horizon every frame took it to 18.1 ms).
///
/// The asked horizon carries [`ASK_BOUND_SLACK`], which is DERIVED from those two tolerances, so
/// that what the picture draws always lies inside what the descent asked for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AskPace {
    held: AskBound,
    drawn: AskBound,
    asked: AskBound,
    descent_at_s: Option<f64>,
}

impl AskPace {
    /// The horizon in force: what the slew moves, and what the other two follow.
    #[must_use]
    pub fn held(&self) -> &AskBound {
        &self.held
    }

    /// The horizon the crossfade's materials carry.
    #[must_use]
    pub fn drawn(&self) -> &AskBound {
        &self.drawn
    }

    /// The horizon the last descent ran on, with its slack.
    #[must_use]
    pub fn asked(&self) -> &AskBound {
        &self.asked
    }

    /// ★ THE HORIZON SLIDES UNTIL IT ARRIVES, THEN HOLDS (ruling F9 item 1; review item 1, THE
    /// BLOCKER).
    ///
    /// The hysteresis gates THE DISTANCE FROM THE TARGET, never the per-frame step. The first
    /// writing gated the step, and a step is proportional to the frame's own seconds: at a tenth of
    /// a second a step was adopted, at seven milliseconds it was not. MEASURED by arithmetic: the
    /// step is 0.2678 of a horizon a second at every rung, so above about 54 frames a second EVERY
    /// step was smaller than the half-percent hysteresis, and the horizon never left the tier
    /// rule's own radii at all. On a fast machine the bounded ask silently did nothing, and the
    /// flights that judged it ran at 39 to 47 frames a second — inside the cliff by luck.
    pub fn slew(&mut self, want: &AskBound, rungs: u8, hysteresis: f64, dt_s: f64) {
        if self.held.same_as(want, rungs, hysteresis) {
            return;
        }
        self.held = self.held.slewed_toward(want, rungs, dt_s);
    }

    /// WHETHER THE CROSSFADE'S MATERIALS MUST FOLLOW THE HORIZON NOW — AND THEY DO: a `true`
    /// answer has already moved the drawn horizon to the held one. It answers `true` only when the
    /// held horizon has moved `fraction` from the one the materials carry.
    pub fn take_rebind(&mut self, rungs: u8, fraction: f64) -> bool {
        if self.drawn.same_as(&self.held, rungs, fraction) {
            return false;
        }
        self.drawn = self.held.clone();
        true
    }

    /// WHETHER THE DESCENT MUST FOLLOW THE HORIZON NOW — AND IT DOES: a `true` answer has already
    /// moved the asked horizon to the held one, with its slack, and stamped the moment. It answers
    /// `true` only when the held horizon has left the ring the last descent asked for (`bracket`),
    /// and never more often than `every_s` seconds apart.
    pub fn take_descent(&mut self, rungs: u8, bracket: f64, every_s: f64, now_s: f64) -> bool {
        let too_soon = self.descent_at_s.is_some_and(|last| now_s - last < every_s);
        if too_soon | self.asked.same_as(&self.held, rungs, bracket) {
            return false;
        }
        self.asked = self.held.clone().with_slack(ASK_BOUND_SLACK);
        self.descent_at_s = Some(now_s);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ladder_view::{
        ASK_BOUND_BRACKET, ASK_BOUND_HYSTERESIS, ASK_BOUND_REBIND, AskRate, EYE_HEIGHT_M,
        ask_bound, switch_m,
    };

    /// A rate that binds on the home planet: a hull at 528 m/s over the ground, three workers.
    fn fast() -> AskRate {
        AskRate {
            chunks_per_s: 195.0,
            speed_mps: 528.0,
            altitude_m: EYE_HEIGHT_M,
            chunks_per_column: 2.0,
        }
    }

    const RUNGS: u8 = 12;

    /// ★ THE BLOCKER (review item 1): THE HORIZON ARRIVES AT 144 FRAMES A SECOND, not only at 30.
    ///
    /// The step is proportional to the frame's seconds, so a gate on the STEP refuses every step
    /// above about 54 frames a second and the bound never binds at all. The gate is on the
    /// DISTANCE, so the arrival takes the same WALL TIME at either frame rate.
    #[test]
    fn the_horizon_arrives_at_a_hundred_and_forty_four_frames_a_second_and_at_thirty() {
        let want = ask_bound(RUNGS, fast());
        assert!(
            want.binds(),
            "the fixture must bind for this to mean anything"
        );
        // The moment the horizon first stands within the hysteresis of what is asked, over a
        // fixed thirty seconds of frames. Zero means it never arrived.
        let arrive_s = |dt_s: f64| -> f64 {
            let mut pace = AskPace::default();
            let mut t = 0.0f64;
            let mut arrived = 0.0f64;
            let steps = (30.0 / dt_s) as u32;
            let mut n = 0u32;
            while n < steps {
                pace.slew(&want, RUNGS, ASK_BOUND_HYSTERESIS, dt_s);
                t += dt_s;
                if (arrived == 0.0) && pace.held().same_as(&want, RUNGS, ASK_BOUND_HYSTERESIS) {
                    arrived = t;
                }
                n += 1;
            }
            arrived
        };
        // The fixture's finest rung walks from 869 m to 14 m — the whole ladder's width — at a
        // quarter of a length a second, which is about twelve seconds of sliding.
        let quick = arrive_s(1.0 / 144.0);
        let slow = arrive_s(1.0 / 30.0);
        assert!(
            quick > 0.0,
            "at 144 frames a second the horizon never arrived"
        );
        assert!(
            slow > 0.0,
            "at 30 frames a second the horizon never arrived"
        );
        assert!(
            quick < 15.0,
            "at 144 frames a second the horizon took {quick} s"
        );
        assert!(
            slow < 15.0,
            "at 30 frames a second the horizon took {slow} s"
        );
        // The same wall time either way: the slew is a rate, not a step per frame.
        let spread = (quick - slow).abs() / quick.max(slow);
        assert!(
            spread < 0.2,
            "144 fps took {quick} s and 30 fps took {slow} s"
        );
    }

    /// AND IT HOLDS once it has arrived: a settled horizon stops moving, so the descent and the
    /// materials stop following it.
    #[test]
    fn a_settled_horizon_stops_moving() {
        let want = ask_bound(RUNGS, fast());
        let mut pace = AskPace::default();
        let mut t = 0.0f64;
        while t < 30.0 {
            pace.slew(&want, RUNGS, ASK_BOUND_HYSTERESIS, 1.0 / 144.0);
            t += 1.0 / 144.0;
        }
        assert!(
            pace.held().same_as(&want, RUNGS, ASK_BOUND_HYSTERESIS),
            "it never arrived"
        );
        let settled = pace.held().clone();
        pace.slew(&want, RUNGS, ASK_BOUND_HYSTERESIS, 1.0 / 144.0);
        assert_eq!(&settled, pace.held());
        // And a horizon that has no time does not move either.
        let mut still = AskPace::default();
        still.slew(&want, RUNGS, ASK_BOUND_HYSTERESIS, 0.0);
        assert_eq!(still.held(), &AskBound::unbounded());
    }

    /// ★ THE CURE'S OWN TEST (review item 6): the descent does NOT re-run while the horizon stays
    /// inside the ring it asked for, and DOES when the horizon leaves it. The materials follow at
    /// their own, finer, tolerance — so a slide rewrites the bands several times for one descent.
    #[test]
    fn the_descent_waits_for_the_bracket_and_the_materials_do_not() {
        let want = ask_bound(RUNGS, fast());
        let mut pace = AskPace::default();
        // The first step off the tier rule's radii is a descent: the asked horizon is unbounded.
        pace.slew(&want, RUNGS, ASK_BOUND_HYSTERESIS, 1.0 / 60.0);
        let mut descents = 0u32;
        let mut rebinds = 0u32;
        let mut frames = 0u32;
        let mut t = 0.0f64;
        while t < 30.0 {
            pace.slew(&want, RUNGS, ASK_BOUND_HYSTERESIS, 1.0 / 60.0);
            descents += u32::from(pace.take_descent(RUNGS, ASK_BOUND_BRACKET, 0.0, t));
            rebinds += u32::from(pace.take_rebind(RUNGS, ASK_BOUND_REBIND));
            frames += 1;
            t += 1.0 / 60.0;
        }
        // The whole slide, and the descent ran a few dozen times, never every frame.
        assert!(descents > 0, "the descent never followed the horizon");
        assert!(
            descents * 10 < frames,
            "the descent ran {descents} times over {frames} frames"
        );
        assert!(
            rebinds > descents,
            "the materials ({rebinds}) must follow more often than the descent ({descents})"
        );
        // Settled, neither runs again.
        assert!(
            pace.held().same_as(&want, RUNGS, ASK_BOUND_HYSTERESIS),
            "it never arrived"
        );
        let mut quiet = 0u32;
        let mut u = 0.0f64;
        while u < 4.0 {
            pace.slew(&want, RUNGS, ASK_BOUND_HYSTERESIS, 1.0 / 60.0);
            quiet += u32::from(pace.take_descent(RUNGS, ASK_BOUND_BRACKET, 0.0, 100.0 + u));
            quiet += u32::from(pace.take_rebind(RUNGS, ASK_BOUND_REBIND));
            u += 1.0 / 60.0;
        }
        assert_eq!(quiet, 0);
    }

    /// ★ THE RATE LIMIT'S OWN ARM (review item 7): a nonzero interval refuses a descent that the
    /// bracket alone would allow, and lets it through once the interval has passed. The shipped
    /// value is zero — the bracket paces it — and this is what that lever does when it is not.
    #[test]
    fn the_rate_limit_refuses_a_descent_until_its_interval_has_passed() {
        let want = ask_bound(RUNGS, fast());
        let mut pace = AskPace::default();
        pace.slew(&want, RUNGS, ASK_BOUND_HYSTERESIS, 1.0);
        assert!(pace.take_descent(RUNGS, ASK_BOUND_BRACKET, 0.5, 10.0));
        // The horizon slides on, but the interval has not passed.
        pace.slew(&want, RUNGS, ASK_BOUND_HYSTERESIS, 1.0);
        assert!(!pace.take_descent(RUNGS, ASK_BOUND_BRACKET, 0.5, 10.2));
        // Past the interval, the same move is allowed.
        assert!(pace.take_descent(RUNGS, ASK_BOUND_BRACKET, 0.5, 10.6));
    }

    /// THE ASKED HORIZON CARRIES THE SLACK, and the drawn one never does: the slack is not part of
    /// a bound's identity, so the two read as the same bound.
    #[test]
    fn the_asked_horizon_carries_the_slack_and_the_drawn_one_does_not() {
        let want = ask_bound(RUNGS, fast());
        let mut pace = AskPace::default();
        pace.slew(&want, RUNGS, ASK_BOUND_HYSTERESIS, 1.0);
        assert!(pace.take_descent(RUNGS, ASK_BOUND_BRACKET, 0.0, 1.0));
        assert!(pace.take_rebind(RUNGS, ASK_BOUND_REBIND));
        assert!(pace.asked().same_as(pace.drawn(), RUNGS, 0.0));
        let (_, asked_out) = pace.asked().fade_bands(0, RUNGS);
        let (_, drawn_out) = pace.drawn().fade_bands(0, RUNGS);
        assert!(asked_out[1] > drawn_out[1], "the asked ring must be wider");
    }

    /// ★ A STOP READS ZERO WITHIN THE HOLD (review item 5): the decay it replaced still read
    /// 194 m/s a second after a hull stopped dead from 528 m/s.
    #[test]
    fn a_hull_that_stops_reads_zero_within_the_hold() {
        let mut hold = PeakHold::default();
        let mut t = 0.0f64;
        // Two seconds of a 528 m/s sawtooth: zero on most frames, the peak on one in six.
        while t < 2.0 {
            let reading = if (t * 60.0).round() as i64 % 6 == 0 {
                528.0
            } else {
                0.0
            };
            let peak = hold.read(reading, t, 1.0);
            if t > 1.0 / 60.0 {
                assert!(peak >= 528.0 - 1.0e-9, "the hold lost the peak at {t} s");
            }
            t += 1.0 / 60.0;
        }
        // The hull stops. Within the hold the reading falls to nothing.
        let mut last = f64::MAX;
        while t < 3.05 {
            last = hold.read(0.0, t, 1.0);
            t += 1.0 / 60.0;
        }
        assert_eq!(last, 0.0);
    }

    /// AND THE HOLD'S OWN EDGES: no hold at all reads the reading itself, a moment that went
    /// backwards holds what it had, and a gap longer than the window empties it.
    #[test]
    fn the_hold_reads_its_edges() {
        let mut hold = PeakHold::default();
        assert_eq!(hold.read(528.0, 0.0, 0.0), 528.0);
        assert_eq!(hold.read(3.0, 1.0, 0.0), 3.0);
        let mut ring = PeakHold::default();
        assert_eq!(ring.read(528.0, 10.0, 1.0), 528.0);
        // A moment that went backwards: the peak stands.
        assert_eq!(ring.read(0.0, 9.0, 1.0), 528.0);
        // A gap of a minute: the window is empty and the new reading stands alone.
        assert_eq!(ring.read(4.0, 70.0, 1.0), 4.0);
    }

    /// ★ THE CAPACITY KEEPS ITS WINDOW AFTER AN IDLE (review item 10): a reading taken after a
    /// long gap carries at most half the weight, so the ten-second average is never thrown away
    /// for one sample of one frame's luck.
    #[test]
    fn a_reading_after_a_long_idle_never_replaces_the_whole_average() {
        let mut rate = Throughput::default();
        // The first reading is taken whole: three workers, 15 ms a build, 200 chunks a second.
        assert_eq!(rate.read(0, 0, 0.0, 3, 10.0), 0.0);
        let first = rate.read(100, 100 * 15_000_000, 1.0, 3, 10.0);
        assert!(
            (first - 200.0).abs() < 1.0,
            "the first reading read {first}"
        );
        // A minute of idle, then a reading at half the capacity: the average must not collapse
        // onto it.
        let after = rate.read(200, 100 * 15_000_000 + 100 * 30_000_000, 61.0, 3, 10.0);
        assert!(
            (after - 150.0).abs() < 1.0,
            "a reading of 100 after 200 must land halfway, not on 100: {after}"
        );
        assert_eq!(rate.value(), after);
    }

    /// AND THE CAPACITY'S OWN EDGES: nothing built carries no reading, a window of zero carries
    /// none, and the last capacity stands either way.
    #[test]
    fn a_capacity_reading_that_cannot_be_read_leaves_the_last_one_standing() {
        let mut rate = Throughput::default();
        assert_eq!(rate.read(10, 1_000_000, 0.0, 3, 10.0), 0.0);
        let held = rate.read(20, 2_000_000, 1.0, 3, 10.0);
        assert!(held > 0.0);
        // Nothing built since: the last capacity stands.
        assert_eq!(rate.read(20, 2_000_000, 2.0, 3, 10.0), held);
        // A moment that did not move: no reading.
        assert_eq!(rate.read(40, 4_000_000, 2.0, 3, 10.0), held);
        // A window of zero: no reading.
        assert_eq!(rate.read(60, 6_000_000, 3.0, 3, 0.0), held);
    }

    /// ★ THE CARD COUNTS IN THE SAME CAPACITY (ruling F9 item 2): the bound sizes its horizon
    /// against EVERYTHING that builds, so the reading is the workers' own plus the card's, and the
    /// two never mix inside — the workers' mean build time is theirs alone.
    #[test]
    fn the_workers_own_capacity_is_read_apart_from_the_card() {
        let mut t = Throughput::default();
        t.read(0, 0, 0.0, 3, 10.0);
        t.read(30, 300_000_000, 1.0, 3, 10.0);
        let workers = t.workers_value();
        assert!(
            workers > 0.0,
            "three workers at 10 ms a chunk have a capacity"
        );
        t.set_card(50.0);
        assert_eq!(t.workers_value(), workers, "the card is not the workers'");
        assert_eq!(t.value(), workers + 50.0, "and the bound reads both");
    }

    #[test]
    fn the_capacity_counts_the_workers_and_the_card_together() {
        let mut rate = Throughput::default();
        assert_eq!(rate.read(0, 0, 0.0, 3, 10.0), 0.0);
        let workers = rate.read(100, 100 * 15_000_000, 1.0, 3, 10.0);
        assert!((workers - 200.0).abs() < 1.0, "the workers read {workers}");
        // The card states 73 chunks a second: the horizon grows by exactly that.
        rate.set_card(73.0);
        assert_eq!(rate.card_value(), 73.0);
        assert_eq!(rate.value(), workers + 73.0);
        // A reading that cannot be read still carries the card, and so does the first mark.
        assert_eq!(
            rate.read(100, 100 * 15_000_000, 2.0, 3, 10.0),
            workers + 73.0
        );
        let mut fresh = Throughput::default();
        fresh.set_card(50.0);
        assert_eq!(fresh.read(10, 10, 0.0, 3, 10.0), 50.0);
        // A card that cannot be negative: a card switched off states nothing.
        rate.set_card(-5.0);
        assert_eq!(rate.card_value(), 0.0);
        assert_eq!(rate.value(), workers);
    }

    /// THE FRAME'S OWN SECONDS: the first frame has none, a moment that went backwards has none,
    /// and the rest is the difference.
    #[test]
    fn the_frame_clock_reads_the_difference_and_never_less_than_nothing() {
        let mut clock = FrameClock::default();
        assert_eq!(clock.seconds(100.0), 0.0);
        assert_eq!(clock.seconds(100.25), 0.25);
        assert_eq!(clock.seconds(100.0), 0.0);
        assert_eq!(clock.seconds(100.5), 0.5);
    }

    /// A PACE THAT NEVER BINDS reads the tier rule's own radii at every rung, and asks nothing of
    /// the descent or the materials.
    #[test]
    fn a_covered_ask_moves_nothing() {
        let mut pace = AskPace::default();
        let free = AskBound::unbounded();
        pace.slew(&free, RUNGS, ASK_BOUND_HYSTERESIS, 1.0);
        assert!(!pace.take_descent(RUNGS, ASK_BOUND_BRACKET, 0.0, 1.0));
        assert!(!pace.take_rebind(RUNGS, ASK_BOUND_REBIND));
        assert_eq!(pace.held().switch_m(0), switch_m(0));
        assert_eq!(pace.asked().switch_m(3), switch_m(3));
        assert_eq!(pace.drawn().switch_m(5), switch_m(5));
    }
}
