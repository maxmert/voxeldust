//! `Fence` — THE one linearizability primitive (PLAN.md "The fence").
//!
//! One `Fence(u64)` per directory key, bumped on every ownership change, never reused,
//! never decreasing. It collapses what the old system smeared across `session_gen`,
//! `lease_epoch`, and `version` into a single mechanism with five rules:
//!
//! 1. Every authoritative action and inter-process message carries the fence it was
//!    issued under; receivers reject `fence < highest_seen`.
//! 2. The directory CAS on a fence is the single commit point — authority is derived
//!    from the directory, never from a peer notification.
//! 3. COMMIT and ABORT are each a CAS on the same fence: mutually exclusive by
//!    construction; the loser observes `expected != current` and no-ops.
//! 4. Self-fence-before-grant: a node that loses its lease hard-stops authority before
//!    the orchestrator may reassign.
//! 5. The gateway validates the fence on every snapshot frame, so a stale or
//!    partitioned old owner physically cannot affect the client.
//!
//! The same pattern is reused as `AnchorGen` for `FrameSpace` re-anchoring (HR4).

use serde::{Deserialize, Serialize};

/// A monotonic authority generation. See module docs for the five fence rules.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Fence(pub u64);

impl Fence {
    /// The genesis fence: the value a directory key is born with.
    pub const GENESIS: Fence = Fence(0);

    /// The next generation. Saturating: a u64 fence bumped once per ownership change
    /// cannot wrap in any physically possible deployment, and saturation keeps the
    /// type total rather than panicking.
    #[must_use]
    pub fn next(self) -> Fence {
        Fence(self.0.saturating_add(1))
    }

    /// Rule 1: would a receiver that has seen `highest_seen` reject a message carrying
    /// `self`? Equal fences are CURRENT (accepted); only strictly older are stale.
    #[must_use]
    pub fn is_stale_against(self, highest_seen: Fence) -> bool {
        self < highest_seen
    }

    /// Rule 3: the compare-and-swap decision. Returns the new fence on success
    /// (`expected` matched) or `None` for the CAS loser, which MUST no-op.
    #[must_use]
    pub fn cas_next(current: Fence, expected: Fence) -> Option<Fence> {
        if current == expected {
            Some(current.next())
        } else {
            None
        }
    }
}

impl core::fmt::Display for Fence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fence-{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_then_next_is_monotonic() {
        let f0 = Fence::GENESIS;
        let f1 = f0.next();
        let f2 = f1.next();
        assert!(f0 < f1);
        assert!(f1 < f2);
        assert_eq!(f2, Fence(2));
    }

    #[test]
    fn next_saturates_instead_of_wrapping() {
        assert_eq!(Fence(u64::MAX).next(), Fence(u64::MAX));
    }

    #[test]
    fn staleness_is_strictly_older_only() {
        let seen = Fence(5);
        assert!(Fence(4).is_stale_against(seen), "older is stale");
        assert!(!Fence(5).is_stale_against(seen), "equal is current");
        assert!(!Fence(6).is_stale_against(seen), "newer is never stale");
    }

    #[test]
    fn cas_winner_bumps_loser_noops() {
        let current = Fence(7);
        assert_eq!(Fence::cas_next(current, Fence(7)), Some(Fence(8)));
        assert_eq!(
            Fence::cas_next(current, Fence(6)),
            None,
            "stale expectation loses"
        );
        assert_eq!(
            Fence::cas_next(current, Fence(8)),
            None,
            "future expectation loses"
        );
    }

    #[test]
    fn commit_and_abort_are_mutually_exclusive() {
        // Rule 3: two contenders CAS the same key with the same expectation;
        // exactly one wins, and the loser's retry against the new value also loses.
        let current = Fence(3);
        let commit = Fence::cas_next(current, Fence(3));
        assert_eq!(commit, Some(Fence(4)));
        let after_commit = commit.expect("winner");
        let abort = Fence::cas_next(after_commit, Fence(3));
        assert_eq!(abort, None, "the loser observes the bump and no-ops");
    }

    #[test]
    fn display_and_serde_roundtrip() {
        assert_eq!(Fence(9).to_string(), "fence-9");
        let bytes = postcard::to_allocvec(&Fence(9)).expect("encode");
        assert_eq!(
            postcard::from_bytes::<Fence>(&bytes).expect("decode"),
            Fence(9)
        );
    }

    #[test]
    fn default_is_genesis() {
        assert_eq!(Fence::default(), Fence::GENESIS);
    }
}
