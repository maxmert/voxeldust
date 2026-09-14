//! ★ THE STAND-DOWN RULE (ruling F9 item 2, the owner's step after Step 15): WHEN the card may
//! take a request at all.
//!
//! The card is a SECOND builder, and a second builder is worth having only where the CPU workers
//! cannot finish the queue IN TIME. The card holds a chunk for a round trip — it submits, the
//! device works, the host maps the answer back — where a CPU worker finishes the same chunk in
//! arithmetic and nothing else. MEASURED (§26.4): on a queue of hundreds the card pays (the band's
//! worst gap at 528 m/s falls from 125 urgent chunks to 87, the queue from 1 442 to 529); on the
//! TAIL OF A STILL STAND, where the queue is two or three deep, the same round trip is pure
//! latency and the hill stand's terrain settled at tick 2 446 against 2 081 with no card — past
//! that stand's own capture tick, so the picture gate went red.
//!
//! Raising the head-of-line rule was tried and REFUSED by the flight: a card that leaves the three
//! most urgent requests alone also stands down on a leg whose queue is six thousand deep, and the
//! slow hull's band went from holding on every frame to 3 394 urgent chunks. A PLACE in the queue
//! is not a measure of a queue's DEPTH.
//!
//! **THE RULE.** The queue is DEEP when it holds more requests than the CPU workers can finish
//! BEFORE THE GROUND REACHES THE SCREEN. It is the same arithmetic the bounded ask already reads
//! ([`crate::ladder_view::AskRate`]): the workers' own capacity, the eye's delivered speed, and
//! the lead the client asks ahead by. Below that depth the card takes nothing and the workers do
//! the whole queue.
//!
//! **A STILL EYE NEVER SEES THE CARD.** An eye that does not move uncovers no new ground, so
//! nothing the queue holds is late, so the workers have all the time there is. A still stand is
//! exactly the stand the card cost, and the rule stands down on it BY NAME, not by a place in a
//! queue.
//!
//! **Example.** The pilot's hull crosses the home planet at 528 m/s. The client asks 61.8 m ahead
//! of the drawn picture, so the ground the queue holds must be on the screen in 0.117 s; three
//! workers at 172 chunks a second finish 20 of them in that time, and the queue holds 501. The
//! queue is deep and the card builds. The pilot then sets the hull down and walks: the lead falls
//! to 0.2 m, the queue to five, and the workers finish fifteen in the same window — the queue is
//! shallow and the card takes nothing. She stops to look at a hill: the eye is still, and the card
//! stands down whatever the queue holds, because no ground is on its way to the screen.

/// ★ HOW MUCH DEEPER THAN THE WORKERS' OWN REACH THE QUEUE MUST STAND: twice.
///
/// The three readings the rule takes all WANDER frame to frame. MEASURED: the builders' capacity
/// reads 170 to 197 chunks a second over one leg (§26.3's own bounded-ask line), and the lead is a
/// sawtooth whose peak is held over a window ([`crate::ask_pace::PeakHold`]). A threshold with no
/// margin flips the card on and off across that wander, and every flip costs the latency of the
/// box that was in flight when it flipped. Twice the workers' reach stands well past the measured
/// wander (a fifth either side would not reach it), and it costs the card nothing where the card
/// is wanted: the 528 m/s leg's queue stands at 501 requests against a threshold of 40.
pub const CARD_DEEP_MARGIN: f64 = 2.0;

/// ★ WHAT THE QUEUE AND THE EYE READ, for the stand-down rule. Every number is one the client
/// already holds for the bounded ask; nothing new is measured, and nothing is derived from a pose
/// the client made up (SL10 clause 7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueueDepth {
    /// The requests the lane holds: submitted and not yet harvested.
    pub pending: usize,
    /// THE CPU WORKERS' OWN CAPACITY, chunks a second — never the card's, because the question is
    /// what the queue costs WITHOUT the card.
    pub workers_per_s: f64,
    /// The eye's speed through the body's frame, metres a second, from delivered poses alone.
    pub speed_mps: f64,
    /// How far ahead of the drawn picture the client asks, metres: the offset from the drawn eye
    /// to the lead eye.
    pub lead_m: f64,
}

impl QueueDepth {
    /// ★ HOW LONG BEFORE THE GROUND REACHES THE SCREEN, seconds — the deadline every pending
    /// request stands under. The eye covers the lead it asks ahead by in `lead / speed` seconds,
    /// and the ground it uncovers there must be built by then.
    ///
    /// `None` where THERE IS NO DEADLINE AT ALL: an eye that does not move uncovers no ground, so
    /// nothing the queue holds is late. A reading that is not a positive, finite number reads as a
    /// still eye, because a deadline nobody can compute may not be one the card acts on.
    #[must_use]
    pub fn deadline_s(&self) -> Option<f64> {
        let moving = self.speed_mps.is_finite() & (self.speed_mps > 0.0);
        let led = self.lead_m.is_finite() & (self.lead_m >= 0.0);
        if !(moving & led) {
            return None;
        }
        Some(self.lead_m / self.speed_mps)
    }

    /// HOW MANY REQUESTS THE CPU WORKERS FINISH before that deadline. Infinite where there is no
    /// deadline, so a still eye's workers always finish in time. Zero where the workers' capacity
    /// is not a measurement yet, so the very first frames of a client — whose queue is the whole
    /// ladder and whose workers have timed nothing — read as deep.
    #[must_use]
    pub fn workers_finish(&self) -> f64 {
        let Some(deadline_s) = self.deadline_s() else {
            return f64::INFINITY;
        };
        let rate = if self.workers_per_s.is_finite() & (self.workers_per_s > 0.0) {
            self.workers_per_s
        } else {
            0.0
        };
        rate * deadline_s
    }

    /// ★ IS THE QUEUE DEEP — may the card take a request at all? The queue must hold more than
    /// [`CARD_DEEP_MARGIN`] times what the workers finish in time.
    #[must_use]
    pub fn deep(&self) -> bool {
        let pending = self.pending as f64;
        pending > CARD_DEEP_MARGIN * self.workers_finish()
    }

    /// ★★ HOW MANY CHUNKS THE CARD MAY HOLD AT ONCE — the rule's second half, and the one the
    /// still stand MEASURED (the hill settled at tick 2 509 against 2 081 with no card, even with
    /// the queue's own depth judged).
    ///
    /// A card that takes a request does not finish it: it hands the bytes to ONE geometry thread
    /// that meshes about ninety chunks a second, and a card free to take whatever the queue offers
    /// builds a BACKLOG in front of that thread. MEASURED: the hill stand's card took 960 chunks,
    /// which is ten seconds of its own geometry stage, where three CPU workers would have finished
    /// the same chunks in under six — so every chunk behind the backlog landed later than it would
    /// have without the card, and the settle moved by exactly that.
    ///
    /// So the card may hold only what IT can finish before the ground reaches the screen:
    /// `deadline ÷ its own stage`. At the 528 m/s leg's 0.117 s deadline and an 11 ms stage that
    /// is ten chunks — which still delivers the card's whole ninety a second, because a pipeline
    /// ten deep at 11 ms a chunk IS ninety a second. It costs the card nothing it could deliver in
    /// time, and it forbids exactly the hoard.
    ///
    /// Zero where the queue is not deep (the card stands down); ONE where the card's stage is not
    /// measured yet, so the first box — the probe that measures the stage — can always be taken.
    #[must_use]
    pub fn card_may_hold(&self, stage_s: f64) -> usize {
        // THE DEADLINE FIRST: a still eye has none, and a card with no deadline holds nothing.
        // (It is also never deep — the depth is measured against this same deadline — so asking
        // the depth first would leave this arm unreachable, and an unreachable arm is not a rule.)
        let Some(deadline_s) = self.deadline_s() else {
            return 0;
        };
        if !self.deep() {
            return 0;
        }
        if !stage_s.is_finite() | (stage_s <= 0.0) {
            return 1;
        }
        ((deadline_s / stage_s).floor() as usize).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 528 m/s leg as the flight measured it (§26.3).
    fn cruise() -> QueueDepth {
        QueueDepth {
            pending: 501,
            workers_per_s: 172.0,
            speed_mps: 528.0,
            lead_m: 61.8,
        }
    }

    #[test]
    fn a_still_eye_never_sees_the_card() {
        let still = QueueDepth {
            pending: 6_000,
            speed_mps: 0.0,
            ..cruise()
        };
        assert_eq!(still.deadline_s(), None);
        assert_eq!(still.workers_finish(), f64::INFINITY);
        assert!(!still.deep());
    }

    #[test]
    fn a_speed_that_is_not_a_number_reads_as_a_still_eye() {
        let broken = QueueDepth {
            speed_mps: f64::NAN,
            ..cruise()
        };
        assert_eq!(broken.deadline_s(), None);
        assert!(!broken.deep());
    }

    #[test]
    fn a_lead_that_is_not_a_number_reads_as_a_still_eye() {
        let broken = QueueDepth {
            lead_m: f64::NAN,
            ..cruise()
        };
        assert_eq!(broken.deadline_s(), None);
        assert!(!broken.deep());
        let backwards = QueueDepth {
            lead_m: -1.0,
            ..cruise()
        };
        assert_eq!(backwards.deadline_s(), None);
    }

    #[test]
    fn the_walk_is_a_shallow_queue() {
        // The walk leg: the lead 0.2 m at 1.4 m/s, three workers at 110 chunks a second, and five
        // requests waiting. The workers finish fifteen before that ground shows.
        let walk = QueueDepth {
            pending: 5,
            workers_per_s: 110.0,
            speed_mps: 1.4,
            lead_m: 0.2,
        };
        let deadline = walk.deadline_s().expect("a moving eye has a deadline");
        assert!((deadline - 0.142_857_142_857_142_86).abs() < 1.0e-12);
        assert!(walk.workers_finish() > 15.0);
        assert!(!walk.deep());
    }

    #[test]
    fn full_cruise_is_a_deep_queue() {
        let deep = cruise();
        let deadline = deep.deadline_s().expect("a moving eye has a deadline");
        assert!((deadline - 0.117_045_454_545_454_55).abs() < 1.0e-12);
        assert!((deep.workers_finish() - 20.131_818_181_818_18).abs() < 1.0e-9);
        assert!(deep.deep());
    }

    #[test]
    fn the_margin_is_the_line_the_queue_must_cross() {
        // A queue at the workers' own reach is not deep; twice it and one more request is.
        let at = QueueDepth {
            pending: 20,
            ..cruise()
        };
        assert!(!at.deep());
        let under = QueueDepth {
            pending: 40,
            ..cruise()
        };
        assert!(!under.deep());
        let over = QueueDepth {
            pending: 41,
            ..cruise()
        };
        assert!(over.deep());
    }

    #[test]
    fn workers_that_have_built_nothing_yet_read_as_a_deep_queue() {
        // The client's own start: the whole ladder is wanted and no build has been timed.
        let start = QueueDepth {
            pending: 1,
            workers_per_s: 0.0,
            ..cruise()
        };
        assert_eq!(start.workers_finish(), 0.0);
        assert!(start.deep());
        let broken = QueueDepth {
            workers_per_s: f64::NAN,
            ..start
        };
        assert_eq!(broken.workers_finish(), 0.0);
        assert!(broken.deep());
    }

    /// ★ THE CARD HOLDS ONLY WHAT IT CAN FINISH IN TIME.
    #[test]
    fn the_card_may_hold_what_its_own_stage_delivers_before_the_deadline() {
        // The 528 m/s leg: a 0.117 s deadline and an 11 ms geometry stage buy ten chunks.
        assert_eq!(cruise().card_may_hold(0.011), 10);
        // A slower stage buys fewer; a stage past the whole deadline still buys one, so the
        // measurement can always be taken.
        assert_eq!(cruise().card_may_hold(0.05), 2);
        assert_eq!(cruise().card_may_hold(0.2), 1);
        // An unmeasured stage buys the probe that measures it.
        assert_eq!(cruise().card_may_hold(0.0), 1);
        assert_eq!(cruise().card_may_hold(f64::NAN), 1);
        // A shallow queue holds nothing, whatever the stage.
        let shallow = QueueDepth {
            pending: 5,
            workers_per_s: 110.0,
            speed_mps: 1.4,
            lead_m: 0.2,
        };
        assert_eq!(shallow.card_may_hold(0.011), 0);
        // And a still eye holds nothing, whatever the queue.
        let still = QueueDepth {
            pending: 6_000,
            speed_mps: 0.0,
            ..cruise()
        };
        assert_eq!(still.card_may_hold(0.011), 0);
    }

    /// A DEEP QUEUE WITH NO DEADLINE cannot happen — the depth is measured against the deadline —
    /// but the hold is written so that it answers zero if it ever did.
    #[test]
    fn a_deep_queue_without_a_deadline_holds_nothing() {
        let no_workers = QueueDepth {
            pending: 1,
            workers_per_s: 0.0,
            speed_mps: 0.0,
            lead_m: 0.0,
        };
        assert!(!no_workers.deep(), "a still eye is never deep");
        assert_eq!(no_workers.card_may_hold(0.011), 0);
    }

    #[test]
    fn an_empty_queue_is_never_deep() {
        let empty = QueueDepth {
            pending: 0,
            workers_per_s: 0.0,
            ..cruise()
        };
        assert!(!empty.deep());
    }
}
