//! The DurableUniverseClock (`docs/design/identity_persistence.md` §7.2): universe
//! time NEVER rewinds — the structural fix for the old system's epoch reset that
//! desynced every orbital clock on orchestrator restart.
//!
//! Two roles, two types:
//! - [`CeilingClock`] (orchestrator): hands out ticks strictly BELOW a durably
//!   write-ahead-reserved ceiling. A crash resumes AT the persisted ceiling — a
//!   FORWARD jump (safe: Category-A positions are pure functions of time), never a
//!   rewind. Reservations are emitted as actions the wrapper MUST persist before the
//!   clock crosses the previous ceiling — pure logic, no I/O here.
//! - [`FollowerClock`] (every shard): a monotonic clamp over observed sync values; a
//!   sync that would step backward is REJECTED as data (typed outcome, counted by
//!   the wrapper), never applied. Seamlessness never depends on absolute agreement —
//!   residual skew bounds interpolation error, not correctness.

use serde::{Deserialize, Serialize};
use vd_core::{EpochId, UniverseTick};

/// Action the persistence wrapper must complete (fsync) before further `advance`
/// calls can cross the previous ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClockAction {
    /// Durably record `ceiling` as the new write-ahead reservation.
    ReserveCeiling(UniverseTick),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum ClockError {
    #[error("ceiling exhausted: reservation not yet confirmed durable")]
    CeilingExhausted,
    #[error("ceiling confirmation must be monotonic: confirmed {confirmed}, held {held}")]
    NonMonotonicCeiling {
        confirmed: UniverseTick,
        held: UniverseTick,
    },
    #[error("zero reserve chunk is meaningless")]
    ZeroReserveChunk,
}

/// Orchestrator-side: the single writer of universe time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CeilingClock {
    epoch: EpochId,
    current: UniverseTick,
    /// Ticks strictly below this are handed out; durably persisted ahead of use.
    confirmed_ceiling: UniverseTick,
    /// A reservation the wrapper has been asked to persist but has not confirmed.
    pending_ceiling: Option<UniverseTick>,
    /// How far ahead each reservation reaches.
    reserve_chunk: u64,
}

impl CeilingClock {
    /// Genesis: universe time starts at zero with an initial reservation the wrapper
    /// must persist before the first `advance`.
    pub fn genesis(
        epoch: EpochId,
        reserve_chunk: u64,
    ) -> Result<(CeilingClock, ClockAction), ClockError> {
        if reserve_chunk == 0 {
            return Err(ClockError::ZeroReserveChunk);
        }
        let first_ceiling = UniverseTick(reserve_chunk);
        Ok((
            CeilingClock {
                epoch,
                current: UniverseTick(0),
                confirmed_ceiling: UniverseTick(0),
                pending_ceiling: Some(first_ceiling),
                reserve_chunk,
            },
            ClockAction::ReserveCeiling(first_ceiling),
        ))
    }

    /// Crash recovery: resume AT the persisted ceiling — possibly a forward jump
    /// past real progress, NEVER behind it. Immediately reserves the next chunk.
    pub fn recover(
        epoch: EpochId,
        persisted_ceiling: UniverseTick,
        reserve_chunk: u64,
    ) -> Result<(CeilingClock, ClockAction), ClockError> {
        if reserve_chunk == 0 {
            return Err(ClockError::ZeroReserveChunk);
        }
        let next_ceiling = UniverseTick(persisted_ceiling.0.saturating_add(reserve_chunk));
        Ok((
            CeilingClock {
                epoch,
                current: persisted_ceiling,
                confirmed_ceiling: persisted_ceiling,
                pending_ceiling: Some(next_ceiling),
                reserve_chunk,
            },
            ClockAction::ReserveCeiling(next_ceiling),
        ))
    }

    /// The wrapper confirms a reservation became durable. A confirmation AT OR ABOVE
    /// the pending request satisfies it (persisting extra headroom is legal); a
    /// partial confirmation below the request keeps the request outstanding so the
    /// wrapper retries — the pending slot can never wedge reservation emission.
    pub fn confirm_ceiling(&mut self, confirmed: UniverseTick) -> Result<(), ClockError> {
        if confirmed <= self.confirmed_ceiling {
            return Err(ClockError::NonMonotonicCeiling {
                confirmed,
                held: self.confirmed_ceiling,
            });
        }
        self.confirmed_ceiling = confirmed;
        if self
            .pending_ceiling
            .is_some_and(|pending| confirmed >= pending)
        {
            self.pending_ceiling = None;
        }
        Ok(())
    }

    /// Hand out the next tick. Near the ceiling a new reservation action is emitted;
    /// AT an unconfirmed ceiling the clock refuses (the wrapper persists, confirms,
    /// retries) — time never outruns durability.
    pub fn advance(&mut self) -> Result<(UniverseTick, Option<ClockAction>), ClockError> {
        let next = UniverseTick(self.current.0.saturating_add(1));
        if next > self.confirmed_ceiling {
            return Err(ClockError::CeilingExhausted);
        }
        self.current = next;

        // Reserve ahead once we are within half a chunk of the ceiling.
        let headroom = self.confirmed_ceiling.0 - self.current.0;
        let action = if headroom < self.reserve_chunk / 2 && self.pending_ceiling.is_none() {
            let new_ceiling =
                UniverseTick(self.confirmed_ceiling.0.saturating_add(self.reserve_chunk));
            self.pending_ceiling = Some(new_ceiling);
            Some(ClockAction::ReserveCeiling(new_ceiling))
        } else {
            None
        };
        Ok((self.current, action))
    }

    #[must_use]
    pub fn now(&self) -> UniverseTick {
        self.current
    }

    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// The durably-confirmed write-ahead ceiling (D-6): persisted each tick so an orchestrator restart
    /// can [`CeilingClock::recover`] AT it — a forward jump past real progress, never behind it (time
    /// never rewinds). `current <= confirmed_ceiling` always holds, so recovering to the ceiling is safe.
    #[must_use]
    pub fn confirmed_ceiling(&self) -> UniverseTick {
        self.confirmed_ceiling
    }
}

/// What a follower did with an observed sync value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncOutcome {
    /// Moved forward to the observed value.
    Advanced {
        from: UniverseTick,
        to: UniverseTick,
    },
    /// Already at the observed value (sync agreed).
    Unchanged,
    /// The observation was BEHIND our clamp: rejected, counted, never applied.
    RejectedBackward {
        held: UniverseTick,
        observed: UniverseTick,
    },
}

/// Shard-side: a monotonic clamp over the orchestrator's published universe time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FollowerClock {
    epoch: EpochId,
    current: UniverseTick,
}

impl FollowerClock {
    #[must_use]
    pub fn new(epoch: EpochId, initial: UniverseTick) -> FollowerClock {
        FollowerClock {
            epoch,
            current: initial,
        }
    }

    /// Apply a sync observation. Backward slew is structurally impossible: the
    /// outcome is data the wrapper counts on a metric, never an applied regression.
    pub fn observe(&mut self, observed: UniverseTick) -> SyncOutcome {
        match observed.cmp(&self.current) {
            core::cmp::Ordering::Greater => {
                let from = self.current;
                self.current = observed;
                SyncOutcome::Advanced { from, to: observed }
            }
            core::cmp::Ordering::Equal => SyncOutcome::Unchanged,
            core::cmp::Ordering::Less => SyncOutcome::RejectedBackward {
                held: self.current,
                observed,
            },
        }
    }

    #[must_use]
    pub fn now(&self) -> UniverseTick {
        self.current
    }

    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const E: EpochId = EpochId(1);

    #[test]
    fn genesis_reserves_before_first_tick() {
        let (mut clock, action) = CeilingClock::genesis(E, 100).expect("genesis");
        assert_eq!(action, ClockAction::ReserveCeiling(UniverseTick(100)));
        assert_eq!(clock.now(), UniverseTick(0));
        assert_eq!(clock.epoch(), E);
        // Until the wrapper confirms the reservation durable, time cannot start.
        assert_eq!(clock.advance(), Err(ClockError::CeilingExhausted));
        clock.confirm_ceiling(UniverseTick(100)).expect("confirm");
        assert_eq!(clock.advance().expect("tick"), (UniverseTick(1), None));
    }

    #[test]
    fn advance_reserves_ahead_within_half_a_chunk() {
        let (mut clock, _) = CeilingClock::genesis(E, 10).expect("genesis");
        clock.confirm_ceiling(UniverseTick(10)).expect("confirm");
        let mut reservation = None;
        for _ in 0..6 {
            let (_, action) = clock.advance().expect("tick");
            if let Some(a) = action {
                reservation = Some(a);
                break;
            }
        }
        // Headroom drops below 5 after tick 6: a new reservation to 20 is emitted.
        assert_eq!(
            reservation,
            Some(ClockAction::ReserveCeiling(UniverseTick(20)))
        );
        // The reservation is emitted ONCE while pending — not repeated every tick.
        let (_, action) = clock.advance().expect("tick");
        assert_eq!(action, None);
    }

    #[test]
    fn unconfirmed_ceiling_stalls_time_then_confirmation_resumes_it() {
        let (mut clock, _) = CeilingClock::genesis(E, 4).expect("genesis");
        clock.confirm_ceiling(UniverseTick(4)).expect("confirm");
        for _ in 0..4 {
            let _ = clock.advance().expect("tick");
        }
        assert_eq!(clock.now(), UniverseTick(4));
        // At the ceiling with the next reservation pending but unconfirmed: refuse.
        assert_eq!(clock.advance(), Err(ClockError::CeilingExhausted));
        clock.confirm_ceiling(UniverseTick(8)).expect("confirm");
        assert_eq!(clock.advance().expect("tick").0, UniverseTick(5));
    }

    #[test]
    fn recovery_resumes_at_the_ceiling_never_behind() {
        // Crash with ceiling persisted at 100 while real progress was anywhere <= 100:
        // recovery jumps FORWARD to 100. Orbits advance analytically; nothing rewinds.
        let (clock, action) = CeilingClock::recover(E, UniverseTick(100), 50).expect("recover");
        assert_eq!(clock.now(), UniverseTick(100));
        assert_eq!(action, ClockAction::ReserveCeiling(UniverseTick(150)));
    }

    #[test]
    fn ceiling_confirmations_are_monotonic() {
        let (mut clock, _) = CeilingClock::genesis(E, 10).expect("genesis");
        clock.confirm_ceiling(UniverseTick(10)).expect("confirm");
        assert_eq!(
            clock.confirm_ceiling(UniverseTick(10)),
            Err(ClockError::NonMonotonicCeiling {
                confirmed: UniverseTick(10),
                held: UniverseTick(10),
            })
        );
        assert_eq!(
            clock.confirm_ceiling(UniverseTick(5)),
            Err(ClockError::NonMonotonicCeiling {
                confirmed: UniverseTick(5),
                held: UniverseTick(10),
            })
        );
    }

    #[test]
    fn over_confirmation_satisfies_the_pending_request() {
        // The wrapper persisted MORE headroom than asked: the request is satisfied
        // (a literal-match would have wedged reservation emission forever — found
        // by the branch-coverage gate).
        let (mut clock, _) = CeilingClock::genesis(E, 10).expect("genesis");
        clock
            .confirm_ceiling(UniverseTick(15))
            .expect("over-confirm");
        // Pending cleared: advancing near the ceiling emits a FRESH reservation.
        let mut saw_reservation = false;
        for _ in 0..12 {
            if let Ok((_, Some(_))) = clock.advance() {
                saw_reservation = true;
                break;
            }
        }
        assert!(saw_reservation, "reservation emission must not wedge");
    }

    #[test]
    fn partial_confirmation_keeps_the_request_outstanding() {
        let (mut clock, _) = CeilingClock::genesis(E, 10).expect("genesis");
        clock.confirm_ceiling(UniverseTick(10)).expect("confirm");
        // Drive to emit the next reservation (to 20).
        let mut pending_request = None;
        for _ in 0..10 {
            if let Ok((_, Some(ClockAction::ReserveCeiling(c)))) = clock.advance() {
                pending_request = Some(c);
                break;
            }
        }
        assert_eq!(pending_request, Some(UniverseTick(20)));
        // A partial confirmation below the request raises the ceiling but keeps the
        // request outstanding (no new reservation is emitted while it waits).
        clock.confirm_ceiling(UniverseTick(15)).expect("partial");
        let (_, action) = clock.advance().expect("tick");
        assert_eq!(action, None, "request still pending; no duplicate emission");
        // Full confirmation finally clears it.
        clock.confirm_ceiling(UniverseTick(20)).expect("full");
    }

    #[test]
    fn zero_chunk_is_rejected() {
        assert_eq!(
            CeilingClock::genesis(E, 0).map(|_| ()),
            Err(ClockError::ZeroReserveChunk)
        );
        assert_eq!(
            CeilingClock::recover(E, UniverseTick(5), 0).map(|_| ()),
            Err(ClockError::ZeroReserveChunk)
        );
    }

    #[test]
    fn follower_rejects_backward_slew_as_data() {
        let mut follower = FollowerClock::new(E, UniverseTick(50));
        assert_eq!(
            follower.observe(UniverseTick(60)),
            SyncOutcome::Advanced {
                from: UniverseTick(50),
                to: UniverseTick(60)
            }
        );
        assert_eq!(follower.observe(UniverseTick(60)), SyncOutcome::Unchanged);
        assert_eq!(
            follower.observe(UniverseTick(55)),
            SyncOutcome::RejectedBackward {
                held: UniverseTick(60),
                observed: UniverseTick(55)
            }
        );
        assert_eq!(follower.now(), UniverseTick(60), "rejection never applied");
        assert_eq!(follower.epoch(), E);
    }

    #[test]
    fn clock_errors_display() {
        assert_eq!(
            ClockError::CeilingExhausted.to_string(),
            "ceiling exhausted: reservation not yet confirmed durable"
        );
        assert_eq!(
            ClockError::ZeroReserveChunk.to_string(),
            "zero reserve chunk is meaningless"
        );
        assert!(
            ClockError::NonMonotonicCeiling {
                confirmed: UniverseTick(1),
                held: UniverseTick(2)
            }
            .to_string()
            .contains("monotonic")
        );
    }

    proptest! {
        /// THE invariant: under ANY interleaving of advances and confirmations, time
        /// never exceeds the durably confirmed ceiling and never moves backward.
        #[test]
        fn time_never_outruns_durability(
            chunk in 1u64..32,
            ops in proptest::collection::vec(proptest::bool::ANY, 0..128),
        ) {
            let (mut clock, ClockAction::ReserveCeiling(first)) =
                CeilingClock::genesis(E, chunk).expect("genesis");
            let mut pending: Vec<UniverseTick> = vec![first];
            let mut last = clock.now();
            for confirm in ops {
                if confirm {
                    if let Some(c) = pending.pop() {
                        let _ = clock.confirm_ceiling(c);
                    }
                } else if let Ok((now, action)) = clock.advance() {
                    prop_assert!(now > last, "time moved backward");
                    last = now;
                    if let Some(ClockAction::ReserveCeiling(c)) = action {
                        pending.push(c);
                    }
                }
                prop_assert!(clock.now() <= clock.confirmed_ceiling,
                    "time outran the durable ceiling");
            }
        }

        /// The follower clamp is monotone under arbitrary observations.
        #[test]
        fn follower_is_monotone(observations in proptest::collection::vec(0u64..1000, 0..64)) {
            let mut follower = FollowerClock::new(E, UniverseTick(0));
            let mut last = follower.now();
            for obs in observations {
                let _ = follower.observe(UniverseTick(obs));
                prop_assert!(follower.now() >= last);
                last = follower.now();
            }
        }
    }
}
