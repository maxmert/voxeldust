//! ★ THE CARD'S TIME BUDGET (ruling F9 item 2: the card as a budgeted second builder).
//!
//! The card builds chunks beside the CPU share, and it must keep the rest of the frame to DRAW.
//! So the frame GRANTS the card a share of its own seconds, and the card spends THE CARD'S OWN
//! MEASURED TIME from that grant. Three readings hold the builder, and this module is the
//! arithmetic over all three:
//!
//! - THE CARD'S OWN TIME for one box — what the card's three compute passes took ON THE CARD, read
//!   from the device's own timestamps where it offers them and from the submit-to-map wall time
//!   where it does not. This is what the BUDGET rations, because this is the time the card cannot
//!   also spend drawing.
//! - THE DISPATCH stage: the card thread's own wall time for one box (the plan, the uploads, the
//!   submit, the wait, the read back).
//! - THE GEOMETRY stage: the second thread's own work for one chunk (the decode and the mesh).
//!
//! The two STAGES are a pipeline, so the builder's own ceiling is the SLOWER of them; the capacity
//! the bounded ask reads is the smaller of that ceiling and what the budget actually pays for.
//!
//! **Example.** The pilot's hull runs at 528 m/s over the home planet at 45 frames a second. A
//! frame lasts 22 ms and grants the card a quarter of it: 5.5 ms. The card's own passes take
//! 1.9 ms a box, so the grant pays for two boxes a frame. The dispatch stage takes 3 ms of its
//! thread and the geometry stage 11 ms of its own, so the pipeline carries 91 chunks a second. The
//! bound reads the smaller of the two.

/// ★ THE DEFAULT SHARE OF A FRAME the card may spend building: a QUARTER (ruling F9 item 2 names
/// it, and names it a knob).
///
/// Why a quarter and not a half. The card draws the picture. MEASURED (`slice_08_integer_bench.md`
/// part 6, and the seam probe's own boxes): one box costs the card about 1.8 ms, and the flight's
/// frame is about 22 ms. A quarter of that frame is 5.5 ms — three boxes — and leaves 16.5 ms,
/// which is above the 12 to 14 ms the renderer's own passes take on the 528 m/s leg. A half would
/// leave 11 ms and the card would be the frame's own wall on the near stands, where the shadow
/// pass alone runs 4 ms.
pub const CARD_BUDGET_FRACTION: f64 = 0.25;

/// OVER HOW MANY BOXES THE CARD'S OWN TIMES ARE SMOOTHED: thirty-two.
///
/// MEASURED reason: two GPU passes over the SAME 1 024 boxes in one bench run read 1 821 ms and
/// 2 074 ms — 13 % apart (`slice_08_integer_bench.md` part 6) — so one box's reading is worth
/// little by itself. Thirty-two boxes is about a second of the card at the quarter budget, which is
/// the same order as the throughput's own window and short enough to follow a rung change (a box
/// at the 1 m rung and one at the 64 m rung cost the card the same, but the GEOMETRY does not).
pub const CARD_SMOOTH_BOXES: f64 = 32.0;

/// ★ WHAT A SHARE OF A FRAME MAY BE: none of it, all of it, or anything between. A knob a person
/// types is clamped here and nowhere else, so a typed `-1` or `5` can never hand the card a
/// negative allowance or the whole machine (review item 8). A reading that is not a number at all
/// reads the default.
#[must_use]
pub fn clamped_fraction(raw: f64) -> f64 {
    if raw.is_nan() {
        return CARD_BUDGET_FRACTION;
    }
    raw.clamp(0.0, 1.0)
}

/// ★ THE CARD'S BUDGET AND ITS THREE MEASURED TIMES (ruling F9 item 2).
///
/// The frame grants; the builder spends. Everything here is arithmetic over smoothed readings, so
/// every arm is a unit test and not a flight.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CardBudget {
    /// THE CARD'S OWN time for one box, smoothed, in seconds — the device's, never the host's.
    per_box_s: f64,
    /// The DISPATCH stage: the card thread's wall time for one box, smoothed, in seconds.
    per_dispatch_s: f64,
    /// The GEOMETRY stage: the second thread's own work for one chunk, smoothed, in seconds.
    per_geometry_s: f64,
    /// The seconds the frames have granted and the builder has not yet spent.
    allowance_s: f64,
    /// The last frame's own seconds and the share it granted: the spend rate is a function of both,
    /// because the anti-banking cap makes it one (see [`CardBudget::spend_rate_per_s`]).
    frame_s: f64,
    fraction: f64,
    /// ★ THE CARD HAS LEFT (review item 4): a dispatch failed, the builder is gone, and the card's
    /// capacity is zero from here on. Nothing turns this back on inside one run.
    detached: bool,
    /// ★ HOW MANY BOXES THE CARD KEEPS IN FLIGHT (the owner's step after Step 15): the dispatch
    /// stage's own wall time is a ROUND TRIP, and `lanes` of them overlap, so the stage carries
    /// `lanes` boxes in the time one box takes. One until the builder says otherwise.
    lanes: f64,
}

impl CardBudget {
    /// THE CARD'S MEASURED TIMES, after one box: `device_s` is what the card's own passes took and
    /// `dispatch_s` what the card thread spent on that box in all. A reading that is not a
    /// positive, finite number is refused, so a clock that went backwards cannot poison an average.
    pub fn note_box(&mut self, device_s: f64, dispatch_s: f64) {
        self.per_box_s = smoothed(self.per_box_s, device_s);
        self.per_dispatch_s = smoothed(self.per_dispatch_s, dispatch_s);
    }

    /// THE GEOMETRY STAGE'S OWN TIME for one chunk (the decode and the mesh), on its own thread.
    pub fn note_geometry(&mut self, geometry_s: f64) {
        self.per_geometry_s = smoothed(self.per_geometry_s, geometry_s);
    }

    /// The card's own measured time for one box, seconds (zero before the first box).
    #[must_use]
    pub fn per_box_s(&self) -> f64 {
        self.per_box_s
    }

    /// The dispatch stage's measured time for one box, seconds.
    #[must_use]
    pub fn per_dispatch_s(&self) -> f64 {
        self.per_dispatch_s
    }

    /// The geometry stage's measured time for one chunk, seconds.
    #[must_use]
    pub fn per_geometry_s(&self) -> f64 {
        self.per_geometry_s
    }

    /// The grant that has not been spent, seconds.
    #[must_use]
    pub fn allowance_s(&self) -> f64 {
        self.allowance_s
    }

    /// ★ THE CARD HAS LEFT: a dispatch failed and the builder is gone. The capacity is zero from
    /// here on and no grant buys anything, so the bounded ask stops sizing its horizon against a
    /// builder that is not there (review item 4).
    pub fn detach(&mut self) {
        self.detached = true;
        self.allowance_s = 0.0;
    }

    /// Whether the card has left.
    #[must_use]
    pub fn is_detached(&self) -> bool {
        self.detached
    }

    /// ★ THE FRAME GRANTS: `fraction` of this frame's own seconds, added to what is unspent. The
    /// allowance never holds more than ONE FRAME'S OWN GRANT (or one box's time, where the grant is
    /// smaller than a box), so a quiet second cannot be banked into a burst that takes a frame away
    /// from the draw — and a machine whose frame grants less than a box still affords one
    /// eventually, which a cap at the grant alone would forbid for ever.
    ///
    /// Returns THE SECONDS THIS FRAME ACTUALLY ADDED, which is what the stamp states: a frame that
    /// granted into a full allowance added nothing, and the stamp says so.
    pub fn grant(&mut self, frame_s: f64, fraction: f64) -> f64 {
        if (frame_s <= 0.0) | (fraction <= 0.0) | self.detached {
            return 0.0;
        }
        self.frame_s = frame_s;
        self.fraction = fraction;
        let grant = frame_s * fraction;
        let cap = grant.max(self.per_box_s);
        let before = self.allowance_s;
        self.allowance_s = (self.allowance_s + grant).min(cap);
        self.allowance_s - before
    }

    /// ★ HOW MANY BOXES THE BUILDER KEEPS IN FLIGHT (the owner's step after Step 15). The builder
    /// states it once, at start. A count below one is read as one: a builder always carries at
    /// least the box it is waiting on.
    pub fn set_lanes(&mut self, lanes: usize) {
        self.lanes = (lanes as f64).max(1.0);
    }

    /// WHAT ONE BOX COSTS THE ALLOWANCE, seconds. Before the first box there is no measured time,
    /// so the whole allowance buys the PROBE that measures one.
    fn cost_s(&self) -> f64 {
        if self.per_box_s > 0.0 {
            self.per_box_s
        } else {
            self.allowance_s
        }
    }

    /// ★ MAY ONE MORE BOX BE DISPATCHED — asked WITHOUT spending (the owner's step after Step 15).
    /// The builder's fill loop peeks before it takes a job out of the queue, so a job is never
    /// held by a card that cannot afford it. The card thread is the only spender, so what a peek
    /// answers is still true when that same thread takes.
    #[must_use]
    pub fn can_take(&self) -> bool {
        let cost = self.cost_s();
        !self.detached & (cost > 0.0) & (self.allowance_s >= cost)
    }

    /// ★ THE BUILDER SPENDS: may one more box be dispatched now? A `true` answer has already taken
    /// the box's own time out of the allowance.
    pub fn take(&mut self) -> bool {
        if !self.can_take() {
            return false;
        }
        self.allowance_s -= self.cost_s();
        true
    }

    /// HOW MANY BOXES A FRAME'S BUDGET ALLOWS, at the card's measured time — the readout the stamp
    /// states, so a flight can say how much of the budget the card used. Zero while nothing is
    /// measured yet.
    #[must_use]
    pub fn boxes_per_frame(&self) -> f64 {
        if (self.frame_s <= 0.0) | (self.fraction <= 0.0) | (self.per_box_s <= 0.0) {
            return 0.0;
        }
        self.frame_s * self.fraction / self.per_box_s
    }

    /// ★ WHAT THE BUDGET ACTUALLY PAYS FOR, boxes a second — THE RATE UNDER THE CAP (review item 2).
    ///
    /// The naive answer is `fraction / per_box`, and it is wrong by up to a factor of two, because
    /// the anti-banking cap ([`CardBudget::grant`]) throws away every grant past one frame's worth.
    /// Two cases, and the cap decides which:
    ///
    /// - A FRAME THAT GRANTS A BOX OR MORE (`g ≥ b`): the allowance is capped at `g`, so a frame
    ///   pays for `floor(g / b)` boxes and the remainder is thrown away at the next grant. Three
    ///   boxes' worth of grant at 2.5 boxes' cost buys TWO boxes a frame, never two and a half.
    /// - A FRAME THAT GRANTS LESS THAN A BOX (`g < b`): the allowance is capped at `b`, so it fills
    ///   over `ceil(b / g)` frames and buys ONE box then. A grant of six tenths of a box buys one
    ///   box every two frames, not 0.6 a frame.
    ///
    /// Zero while nothing is measured, while no frame has granted, and once the card has left.
    #[must_use]
    pub fn spend_rate_per_s(&self) -> f64 {
        if (self.per_box_s <= 0.0) | (self.frame_s <= 0.0) | (self.fraction <= 0.0) | self.detached
        {
            return 0.0;
        }
        let grant = self.frame_s * self.fraction;
        if grant >= self.per_box_s {
            (grant / self.per_box_s).floor() / self.frame_s
        } else {
            1.0 / ((self.per_box_s / grant).ceil() * self.frame_s)
        }
    }

    /// THE PIPELINE'S OWN CEILING, seconds a chunk: the SLOWER of the two stages, because a
    /// pipeline delivers no faster than its slowest stage. Zero until both are measured.
    ///
    /// ★ THE DISPATCH STAGE IS DIVIDED BY THE LANES (the owner's step after Step 15): its wall
    /// time is a ROUND TRIP — the submit, the device's own work, the map home — and `lanes` trips
    /// overlap, so that stage delivers a box every `trip / lanes` seconds. The GEOMETRY stage is
    /// one thread doing arithmetic and is never divided.
    #[must_use]
    pub fn stage_s(&self) -> f64 {
        if (self.per_dispatch_s <= 0.0) | (self.per_geometry_s <= 0.0) {
            return 0.0;
        }
        (self.per_dispatch_s / self.lanes.max(1.0)).max(self.per_geometry_s)
    }

    /// ★ THE CARD'S CAPACITY, chunks a second: the SMALLER of its two ceilings — what its pipeline
    /// can carry ([`CardBudget::stage_s`]) and what its share of every second actually pays for
    /// ([`CardBudget::spend_rate_per_s`]). Zero until both are measured, so an unmeasured card
    /// never widens the bounded ask's horizon, and zero once the card has left.
    #[must_use]
    pub fn capacity_per_s(&self) -> f64 {
        let stage = self.stage_s();
        let spend = self.spend_rate_per_s();
        if (stage <= 0.0) | (spend <= 0.0) {
            return 0.0;
        }
        (1.0 / stage).min(spend)
    }
}

/// One reading into a smoothed average over [`CARD_SMOOTH_BOXES`] readings. The FIRST reading is
/// taken whole (there is nothing to average it with); a reading that is not a positive, finite
/// number is refused, so a clock that went backwards and a clock that read for ever leave the
/// average exactly where it was.
fn smoothed(value: f64, reading: f64) -> f64 {
    if !reading.is_finite() | (reading <= 0.0) {
        return value;
    }
    if value <= 0.0 {
        return reading;
    }
    value + (reading - value) / CARD_SMOOTH_BOXES
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flight's own numbers: a 22 ms frame, a 1.9 ms box on the card, a 3 ms dispatch stage and
    /// an 11 ms geometry stage.
    const FRAME_S: f64 = 0.022;
    const BOX_S: f64 = 0.0019;
    const DISPATCH_S: f64 = 0.003;
    const GEOMETRY_S: f64 = 0.011;

    /// A budget with every time measured and one frame granted, without waiting for thirty-two
    /// readings.
    fn measured() -> CardBudget {
        let mut b = CardBudget::default();
        b.note_box(BOX_S, DISPATCH_S);
        b.note_geometry(GEOMETRY_S);
        b.grant(FRAME_S, CARD_BUDGET_FRACTION);
        b
    }

    /// ★ THE FIRST BOX IS A PROBE: nothing is measured, so the first grant buys one box and the
    /// allowance is spent whole. The box that follows it is paid for at the measured time.
    #[test]
    fn the_first_box_is_the_probe_that_measures_the_rest() {
        let mut budget = CardBudget::default();
        assert!(!budget.take(), "a card with no grant may not build");
        let granted = budget.grant(FRAME_S, CARD_BUDGET_FRACTION);
        assert_eq!(granted, FRAME_S * CARD_BUDGET_FRACTION);
        assert!(budget.take(), "the first grant buys the probe");
        assert_eq!(budget.allowance_s(), 0.0);
        assert!(!budget.take(), "and there is nothing left for a second");
        budget.note_box(BOX_S, DISPATCH_S);
        budget.note_geometry(GEOMETRY_S);
        assert_eq!(budget.per_box_s(), BOX_S);
        assert_eq!(budget.per_dispatch_s(), DISPATCH_S);
        assert_eq!(budget.per_geometry_s(), GEOMETRY_S);
    }

    /// ★ THE BUDGET'S ARITHMETIC: a quarter of a 22 ms frame is 5.5 ms, which buys TWO 1.9 ms boxes
    /// and refuses the third.
    #[test]
    fn a_quarter_of_the_frame_buys_two_boxes_and_refuses_the_third() {
        let mut budget = measured();
        assert!(budget.take(), "the first box fits");
        assert!(budget.take(), "the second box fits");
        assert!(!budget.take(), "the third box is past the frame's share");
        // The readout says the same thing, with the fraction of a box the grant left over.
        let allowed = budget.boxes_per_frame();
        let want = FRAME_S * CARD_BUDGET_FRACTION / BOX_S;
        let close = (allowed - want).abs() < 1.0e-9;
        assert!(close, "the frame's share buys the boxes the readout states");
    }

    /// AND THE NEXT FRAME PAYS FOR THE NEXT BOXES: the grant is per frame, never a bank. A card
    /// that stood idle for a hundred frames holds ONE frame's grant, not a hundred.
    #[test]
    fn a_quiet_card_banks_one_frame_and_never_a_burst() {
        let mut budget = measured();
        let mut frames = 0;
        while frames < 100 {
            budget.grant(FRAME_S, CARD_BUDGET_FRACTION);
            frames += 1;
        }
        assert_eq!(
            budget.allowance_s(),
            FRAME_S * CARD_BUDGET_FRACTION,
            "a hundred quiet frames hold one frame's grant"
        );
        assert!(budget.take(), "which is two boxes");
        assert!(budget.take());
        assert!(!budget.take(), "and never a third");
    }

    /// ★ A FRAME SHORTER THAN ONE BOX still lets the card afford a box: the cap is the LARGER of
    /// the frame's grant and one box, so a fast machine is not a card that never builds. (The cap
    /// at the grant alone was the arm this test refuses.)
    #[test]
    fn a_fast_frame_still_affords_a_box() {
        let mut budget = measured();
        let quick_s = 1.0 / 240.0;
        assert!(
            quick_s * CARD_BUDGET_FRACTION < BOX_S,
            "the fixture must grant less than a box for this to mean anything"
        );
        let mut frames = 0;
        while frames < 20 {
            budget.grant(quick_s, CARD_BUDGET_FRACTION);
            frames += 1;
        }
        assert!(
            budget.take(),
            "the card affords a box at 240 frames a second"
        );
    }

    /// A GRANT OF NOTHING IS NOTHING: no frame, or no share, grants no time — and the switch that
    /// turns the card off is exactly a share of zero.
    #[test]
    fn no_frame_and_no_share_grant_nothing() {
        let mut budget = measured();
        let held = budget.allowance_s();
        assert_eq!(budget.grant(0.0, CARD_BUDGET_FRACTION), 0.0);
        assert_eq!(budget.grant(-1.0, CARD_BUDGET_FRACTION), 0.0);
        assert_eq!(budget.grant(FRAME_S, 0.0), 0.0);
        assert_eq!(budget.grant(FRAME_S, -0.5), 0.0);
        assert_eq!(budget.allowance_s(), held);
        assert_eq!(CardBudget::default().boxes_per_frame(), 0.0);
        let mut no_box = CardBudget::default();
        no_box.grant(FRAME_S, CARD_BUDGET_FRACTION);
        assert_eq!(no_box.boxes_per_frame(), 0.0, "no box is measured yet");
        let mut no_frame = CardBudget::default();
        no_frame.note_box(BOX_S, DISPATCH_S);
        assert_eq!(no_frame.boxes_per_frame(), 0.0, "no frame has granted yet");
    }

    /// ★ THE SPEND RATE UNDER THE CAP (review item 2), at four grant-to-box ratios.
    #[test]
    fn the_spend_rate_is_the_rate_the_cap_allows_and_never_the_ratio() {
        // Two and a bit boxes a frame: the cap pays for TWO, never 2.89.
        let budget = measured();
        let ratio = FRAME_S * CARD_BUDGET_FRACTION / BOX_S;
        assert!((2.0..3.0).contains(&ratio), "the fixture must be 2 < r < 3");
        let want = 2.0 / FRAME_S;
        let two_a_frame = (budget.spend_rate_per_s() - want).abs() < 1.0e-9;
        assert!(two_a_frame, "the cap pays for two boxes a frame, not 2.89");
        // A grant of exactly one box: one box a frame.
        let mut one = CardBudget::default();
        one.note_box(BOX_S, DISPATCH_S);
        one.note_geometry(GEOMETRY_S);
        one.grant(BOX_S, 1.0);
        let exact = (one.spend_rate_per_s() - 1.0 / BOX_S).abs() < 1.0e-6;
        assert!(exact, "a grant of exactly one box buys one box a frame");
        // A grant of six tenths of a box: ONE box every two frames, never 0.6 a frame.
        let mut thin = CardBudget::default();
        thin.note_box(BOX_S, DISPATCH_S);
        thin.note_geometry(GEOMETRY_S);
        thin.grant(FRAME_S, 0.6 * BOX_S / FRAME_S);
        let want_thin = 1.0 / (2.0 * FRAME_S);
        let every_second_frame = (thin.spend_rate_per_s() - want_thin).abs() < 1.0e-9;
        assert!(
            every_second_frame,
            "six tenths of a box is one box in two frames"
        );
        // A grant of a tenth of a box: one box every ten frames.
        let mut thinner = CardBudget::default();
        thinner.note_box(BOX_S, DISPATCH_S);
        thinner.note_geometry(GEOMETRY_S);
        thinner.grant(FRAME_S, 0.1 * BOX_S / FRAME_S);
        let want_thinner = 1.0 / (10.0 * FRAME_S);
        let every_tenth_frame = (thinner.spend_rate_per_s() - want_thinner).abs() < 1.0e-9;
        assert!(
            every_tenth_frame,
            "a tenth of a box is one box in ten frames"
        );
        // And nothing measured, or no frame granted, is no rate at all.
        assert_eq!(CardBudget::default().spend_rate_per_s(), 0.0);
        let mut ungranted = CardBudget::default();
        ungranted.note_box(BOX_S, DISPATCH_S);
        assert_eq!(ungranted.spend_rate_per_s(), 0.0);
    }

    /// ★ AND THE SPEND RATE IS WHAT THE BUDGET REALLY DELIVERS: the arithmetic is checked against a
    /// SIMULATION of a thousand frames of grant-and-take, at both sides of the cap.
    #[test]
    fn the_spend_rate_matches_a_thousand_frames_of_grant_and_take() {
        let run = |frame_s: f64, fraction: f64| -> (f64, f64) {
            let mut budget = CardBudget::default();
            budget.note_box(BOX_S, DISPATCH_S);
            budget.note_geometry(GEOMETRY_S);
            let mut built = 0u32;
            let mut frames = 0u32;
            while frames < 1000 {
                budget.grant(frame_s, fraction);
                while budget.take() {
                    built += 1;
                }
                frames += 1;
            }
            let seconds = f64::from(frames) * frame_s;
            (f64::from(built) / seconds, budget.spend_rate_per_s())
        };
        // Above the cap (2.89 boxes' worth a frame) and below it (0.6 of one).
        let (measured_rate, stated) = run(FRAME_S, CARD_BUDGET_FRACTION);
        let wide = (measured_rate - stated).abs() < 1.0;
        assert!(
            wide,
            "the stated rate is the rate a thousand frames deliver"
        );
        let (thin_rate, thin_stated) = run(FRAME_S, 0.6 * BOX_S / FRAME_S);
        let thin_matches = (thin_rate - thin_stated).abs() < 1.0;
        assert!(thin_matches, "and it is, under the cap as well as over it");
    }

    /// ★ THE CAPACITY IS THE SMALLER OF THE PIPELINE AND THE BUDGET.
    #[test]
    fn the_capacity_is_the_smaller_of_the_pipeline_and_the_budget() {
        let mut budget = measured();
        let stage = budget.stage_s();
        assert_eq!(stage, GEOMETRY_S, "the geometry stage is the slower one");
        let quarter = budget.capacity_per_s();
        assert!(quarter <= 1.0 / stage, "the pipeline is a ceiling");
        assert!(quarter <= budget.spend_rate_per_s(), "so is the budget");
        // The whole frame: the pipeline binds alone.
        budget.grant(FRAME_S, 1.0);
        let pipeline_binds = (budget.capacity_per_s() - 1.0 / GEOMETRY_S).abs() < 1.0e-9;
        assert!(pipeline_binds, "the stage binds at the whole frame");
        // A hundredth of a frame: the budget binds.
        budget.grant(FRAME_S, 0.01);
        assert!(
            budget.capacity_per_s() < 1.0 / GEOMETRY_S,
            "the budget binds when it is thin"
        );
        assert_eq!(budget.capacity_per_s(), budget.spend_rate_per_s());
    }

    /// ★ THE LANES DIVIDE THE ROUND TRIP (the owner's step after Step 15): four boxes in flight
    /// carry four of the dispatch stage's trips at once, so that stage stops being the ceiling.
    #[test]
    fn the_boxes_in_flight_divide_the_dispatch_stage() {
        // A dispatch stage that is the SLOWER one: a 20 ms round trip against an 11 ms geometry.
        let trip_s = 0.020;
        let mut budget = CardBudget::default();
        budget.note_box(BOX_S, trip_s);
        budget.note_geometry(GEOMETRY_S);
        budget.grant(FRAME_S, 1.0);
        assert_eq!(
            budget.stage_s(),
            trip_s,
            "one box in flight: the trip binds"
        );
        budget.set_lanes(2);
        assert_eq!(budget.stage_s(), GEOMETRY_S, "two trips overlap");
        budget.set_lanes(4);
        assert_eq!(
            budget.stage_s(),
            GEOMETRY_S,
            "and the geometry binds from there"
        );
        // A count below one is read as one: a builder always carries the box it waits on.
        budget.set_lanes(0);
        assert_eq!(budget.stage_s(), trip_s);
    }

    /// ★ THE PEEK ANSWERS WHAT THE SPEND WOULD: the builder's fill loop asks before it takes a job
    /// out of the queue, and the card thread is the only spender.
    #[test]
    fn the_peek_answers_what_the_spend_would() {
        let mut budget = CardBudget::default();
        assert!(!budget.can_take(), "no grant, no box");
        budget.note_box(BOX_S, DISPATCH_S);
        budget.note_geometry(GEOMETRY_S);
        assert!(!budget.can_take(), "still no grant");
        budget.grant(FRAME_S, CARD_BUDGET_FRACTION);
        assert!(budget.can_take(), "the grant covers a box");
        assert!(budget.take(), "and the spend agrees");
        assert!(budget.can_take(), "the grant covers a second");
        assert!(budget.take());
        assert!(!budget.can_take(), "the third is past the frame's share");
        assert!(!budget.take());
        // A card that has left answers no, whatever it holds.
        budget.grant(FRAME_S, CARD_BUDGET_FRACTION);
        assert!(budget.can_take());
        budget.detach();
        assert!(!budget.can_take());
    }

    /// AND AN UNMEASURED CARD NEVER WIDENS THE HORIZON: no box, no dispatch, no geometry or no
    /// grant reads zero.
    #[test]
    fn an_unmeasured_card_has_no_capacity() {
        assert_eq!(CardBudget::default().capacity_per_s(), 0.0);
        assert_eq!(CardBudget::default().stage_s(), 0.0);
        let mut dispatch_only = CardBudget::default();
        dispatch_only.note_box(BOX_S, DISPATCH_S);
        dispatch_only.grant(FRAME_S, CARD_BUDGET_FRACTION);
        assert_eq!(dispatch_only.stage_s(), 0.0, "no geometry stage yet");
        assert_eq!(dispatch_only.capacity_per_s(), 0.0);
        let mut geometry_only = CardBudget::default();
        geometry_only.note_geometry(GEOMETRY_S);
        geometry_only.grant(FRAME_S, CARD_BUDGET_FRACTION);
        assert_eq!(geometry_only.stage_s(), 0.0, "no dispatch stage yet");
        assert_eq!(geometry_only.capacity_per_s(), 0.0);
        // And a pipeline with no grant behind it states nothing either.
        let mut ungranted = CardBudget::default();
        ungranted.note_box(BOX_S, DISPATCH_S);
        ungranted.note_geometry(GEOMETRY_S);
        assert_eq!(ungranted.capacity_per_s(), 0.0);
    }

    /// ★ THE CARD THAT LEFT (review item 4): a dispatch failed, and from that moment the card holds
    /// no allowance, takes nothing, is granted nothing and states no capacity — so the bounded ask
    /// stops sizing its horizon against a builder that is not there.
    #[test]
    fn a_detached_card_states_no_capacity_and_takes_nothing() {
        let mut budget = measured();
        assert!(budget.capacity_per_s() > 0.0, "it must build first");
        assert!(!budget.is_detached());
        budget.detach();
        assert!(budget.is_detached());
        assert_eq!(budget.allowance_s(), 0.0);
        assert_eq!(budget.capacity_per_s(), 0.0);
        assert_eq!(budget.spend_rate_per_s(), 0.0);
        assert_eq!(budget.grant(FRAME_S, CARD_BUDGET_FRACTION), 0.0);
        assert!(!budget.take());
    }

    /// ★ THE SMOOTHING keeps what it learned: one slow box moves the average by a thirty-second,
    /// never to the slow box itself, and a reading that is not a positive, finite number is refused
    /// outright (a clock that went backwards, and one that read for ever).
    #[test]
    fn one_slow_box_moves_the_average_by_a_thirty_second() {
        let mut budget = measured();
        budget.note_box(BOX_S * 2.0, DISPATCH_S);
        let want = BOX_S + BOX_S / CARD_SMOOTH_BOXES;
        // The equality is the assertion, so the message carries no CALL: a method in a failure
        // message is a region the coverage gate counts and no passing test can ever reach.
        let moved = (budget.per_box_s() - want).abs() < 1.0e-12;
        assert!(moved, "one slow box moved the average past a thirty-second");
        let before = budget.clone();
        budget.note_box(0.0, -1.0);
        budget.note_geometry(0.0);
        assert_eq!(budget, before, "a reading that is not positive is refused");
        budget.note_box(f64::NAN, f64::INFINITY);
        budget.note_geometry(f64::NAN);
        assert_eq!(budget, before, "and so is one that is not a finite number");
    }

    /// ★ A SHARE A PERSON TYPED IS CLAMPED (review item 8): none, all, or anything between; a
    /// number that is not a number reads the default.
    #[test]
    fn a_typed_share_is_clamped_to_none_all_or_between() {
        assert_eq!(clamped_fraction(0.25), 0.25);
        assert_eq!(clamped_fraction(0.0), 0.0);
        assert_eq!(clamped_fraction(1.0), 1.0);
        assert_eq!(clamped_fraction(-1.0), 0.0);
        assert_eq!(clamped_fraction(5.0), 1.0);
        assert_eq!(clamped_fraction(f64::NAN), CARD_BUDGET_FRACTION);
    }
}
