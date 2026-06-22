//! The per-entity authority FSM: `Owned | Ghost | Frozen`
//! (`docs/design/transfer_protocol.md` §1.1).
//!
//! Invariant (cluster-wide, enforced by the directory + the saga's ordered
//! demote-before-promote; asserted by the harness's AUTHORITY-UNIQUE): the set of
//! `Owned` placements is a partition — never two owners, and outside an in-flight
//! saga never zero.
//!
//! Local legality (THIS module): which authority moves a single shard may make, each
//! gated on a fence so a stale instruction (rule 1) is a typed no-op, never applied.
//! Ghosts are kinematic mirrors that NEVER independently integrate physics; Frozen
//! entities accept no input and no integration — the gateway fence makes the freeze
//! enforced, not cooperative.
//!
//! ## WIRED (P2 Slice 1d.4b attached it)
//! This FSM is now the per-entity authority TRUTH on the stub `Dot`: `simulates()` is the
//! `emit_frames` gate, half the `apply_input` gate, and the `topology.rs` oracle held-set.
//! Login AND the transfer-dest both mint `Ghost{GENESIS}` and Promote `Ghost→Owned` via the
//! IDENTICAL machinery (`stub.rs` `flip_grant` for login, `apply_crossing` for the dest); the
//! source self-fence demotes `Owned→Frozen→Ghost` and RETAINS the dot (the first ghost). It is
//! KIND-GENERIC (keys only on Fence/TransferId/TickId — never `EntityKind`), so a future
//! ship/block/signal entity uses the SAME states (no per-kind fork — HR2/HR3).
//!
//! FG-2 single-truth (honest interim): there is exactly ONE authority representation in effect
//! (`simulates()`). The stub's `granted`/`entity_fence` are the strictly-weaker directory-record
//! PREDICATE/poll bookkeeping (which directory op is owed) that 1d.5b's saga-pushed Promote/Demote
//! DELETES — they never re-decide emit/input/oracle. Do NOT build a second authority mechanism
//! beside this (split-brain hides where two representations that look like one disagree).

use serde::{Deserialize, Serialize};
use vd_core::{Fence, TickId, TransferId};

/// A shard-local authority state for one entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Authority {
    /// This shard simulates the entity; frames are emitted under `fence`.
    Owned { fence: Fence },
    /// Read-only kinematic mirror fed by `GhostFlow::Delta` from the owner.
    Ghost {
        source_fence: Fence,
        since_tick: TickId,
    },
    /// Mid-transfer: no input, no integration (saga `transfer` holds the key lock).
    Frozen { transfer: TransferId, fence: Fence },
}

/// Authority transitions a shard may be instructed to make.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorityCmd {
    /// Saga FLUSH: stop simulating; retain state for the final flush.
    Freeze { transfer: TransferId },
    /// Compensation: resume simulating (the fence never moved).
    Thaw { transfer: TransferId },
    /// Directory-confirmed promotion: a Ghost becomes Owned at the NEW fence.
    /// NEVER spawn-on-promote — promotion requires an existing ghost (R3).
    Promote { new_fence: Fence },
    /// Post-commit demotion: the old owner becomes a ghost of the new owner.
    Demote {
        new_owner_fence: Fence,
        at_tick: TickId,
    },
    /// Ghost refresh from a delta (kept for fence/tick discipline).
    GhostRefresh { source_fence: Fence },
}

/// Why a command was refused. Refusals are data — the caller logs/metrics them;
/// nothing is ever silently dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum AuthorityError {
    #[error("stale fence: carried {carried}, held {held}")]
    StaleFence { carried: Fence, held: Fence },
    #[error("illegal transition for current authority state")]
    IllegalTransition,
    #[error("freeze/thaw transfer mismatch")]
    TransferMismatch,
}

impl Authority {
    /// The fence this state holds (ghosts hold their SOURCE's fence).
    #[must_use]
    pub fn fence(&self) -> Fence {
        match self {
            Authority::Owned { fence } | Authority::Frozen { fence, .. } => *fence,
            Authority::Ghost { source_fence, .. } => *source_fence,
        }
    }

    /// May this entity be simulated (integrated, accept input) right now?
    #[must_use]
    pub fn simulates(&self) -> bool {
        matches!(self, Authority::Owned { .. })
    }

    /// Apply a command. Total: every refusal is a typed error, never a panic and
    /// never a silent drop.
    pub fn apply(self, cmd: AuthorityCmd) -> Result<Authority, AuthorityError> {
        match (self, cmd) {
            // Owned -> Frozen: the saga's FLUSH step.
            (Authority::Owned { fence }, AuthorityCmd::Freeze { transfer }) => {
                Ok(Authority::Frozen { transfer, fence })
            }
            // Frozen -> Owned: compensation, same transfer only.
            (Authority::Frozen { transfer, fence }, AuthorityCmd::Thaw { transfer: t }) => {
                if transfer == t {
                    Ok(Authority::Owned { fence })
                } else {
                    Err(AuthorityError::TransferMismatch)
                }
            }
            // Ghost -> Owned: directory-confirmed promotion; the new fence must be
            // strictly newer than the ghost's view of the old owner.
            (Authority::Ghost { source_fence, .. }, AuthorityCmd::Promote { new_fence }) => {
                if source_fence.is_stale_against(new_fence) {
                    Ok(Authority::Owned { fence: new_fence })
                } else {
                    Err(AuthorityError::StaleFence {
                        carried: new_fence,
                        held: source_fence,
                    })
                }
            }
            // Frozen -> Ghost: the old owner demotes AFTER the commit point.
            (
                Authority::Frozen { fence, .. },
                AuthorityCmd::Demote {
                    new_owner_fence,
                    at_tick,
                },
            ) => {
                if fence.is_stale_against(new_owner_fence) {
                    Ok(Authority::Ghost {
                        source_fence: new_owner_fence,
                        since_tick: at_tick,
                    })
                } else {
                    Err(AuthorityError::StaleFence {
                        carried: new_owner_fence,
                        held: fence,
                    })
                }
            }
            // Ghost fence refresh: monotone only.
            (
                Authority::Ghost {
                    source_fence,
                    since_tick,
                },
                AuthorityCmd::GhostRefresh { source_fence: new },
            ) => {
                if new.is_stale_against(source_fence) {
                    Err(AuthorityError::StaleFence {
                        carried: new,
                        held: source_fence,
                    })
                } else {
                    Ok(Authority::Ghost {
                        source_fence: new,
                        since_tick,
                    })
                }
            }
            // Everything else is illegal: an Owned entity cannot be promoted (it IS
            // the owner), a Ghost cannot freeze (it owns nothing), etc.
            _ => Err(AuthorityError::IllegalTransition),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const T: TransferId = TransferId(1);

    #[test]
    fn the_transfer_lifecycle_round_trip() {
        // Source side: Owned -> Frozen -> Ghost (after the dest committed at fence+1).
        let source = Authority::Owned { fence: Fence(5) };
        assert!(source.simulates());
        let frozen = source
            .apply(AuthorityCmd::Freeze { transfer: T })
            .expect("freeze");
        assert!(!frozen.simulates());
        assert_eq!(frozen.fence(), Fence(5));
        let ghost = frozen
            .apply(AuthorityCmd::Demote {
                new_owner_fence: Fence(6),
                at_tick: TickId(100),
            })
            .expect("demote");
        assert_eq!(
            ghost,
            Authority::Ghost {
                source_fence: Fence(6),
                since_tick: TickId(100)
            }
        );

        // Dest side: Ghost -> Owned at the new fence.
        let dest_ghost = Authority::Ghost {
            source_fence: Fence(5),
            since_tick: TickId(90),
        };
        let owned = dest_ghost
            .apply(AuthorityCmd::Promote {
                new_fence: Fence(6),
            })
            .expect("promote");
        assert_eq!(owned, Authority::Owned { fence: Fence(6) });
    }

    #[test]
    fn thaw_restores_ownership_for_the_same_transfer_only() {
        let frozen = Authority::Frozen {
            transfer: T,
            fence: Fence(5),
        };
        assert_eq!(
            frozen.apply(AuthorityCmd::Thaw { transfer: T }),
            Ok(Authority::Owned { fence: Fence(5) })
        );
        assert_eq!(
            frozen.apply(AuthorityCmd::Thaw {
                transfer: TransferId(2)
            }),
            Err(AuthorityError::TransferMismatch)
        );
    }

    #[test]
    fn promotion_requires_a_strictly_newer_fence() {
        let ghost = Authority::Ghost {
            source_fence: Fence(5),
            since_tick: TickId(1),
        };
        assert_eq!(
            ghost.apply(AuthorityCmd::Promote {
                new_fence: Fence(5)
            }),
            Err(AuthorityError::StaleFence {
                carried: Fence(5),
                held: Fence(5)
            })
        );
        assert_eq!(
            ghost.apply(AuthorityCmd::Promote {
                new_fence: Fence(4)
            }),
            Err(AuthorityError::StaleFence {
                carried: Fence(4),
                held: Fence(5)
            })
        );
    }

    #[test]
    fn demotion_requires_the_new_owners_newer_fence() {
        let frozen = Authority::Frozen {
            transfer: T,
            fence: Fence(5),
        };
        assert_eq!(
            frozen.apply(AuthorityCmd::Demote {
                new_owner_fence: Fence(5),
                at_tick: TickId(1)
            }),
            Err(AuthorityError::StaleFence {
                carried: Fence(5),
                held: Fence(5)
            })
        );
    }

    #[test]
    fn ghost_refresh_is_fence_monotone() {
        let ghost = Authority::Ghost {
            source_fence: Fence(5),
            since_tick: TickId(7),
        };
        assert_eq!(
            ghost.apply(AuthorityCmd::GhostRefresh {
                source_fence: Fence(6)
            }),
            Ok(Authority::Ghost {
                source_fence: Fence(6),
                since_tick: TickId(7)
            })
        );
        // Equal is current (rule 1): accepted, unchanged.
        assert_eq!(
            ghost.apply(AuthorityCmd::GhostRefresh {
                source_fence: Fence(5)
            }),
            Ok(ghost)
        );
        assert_eq!(
            ghost.apply(AuthorityCmd::GhostRefresh {
                source_fence: Fence(4)
            }),
            Err(AuthorityError::StaleFence {
                carried: Fence(4),
                held: Fence(5)
            })
        );
    }

    #[test]
    fn illegal_transitions_are_typed_refusals() {
        let owned = Authority::Owned { fence: Fence(1) };
        let ghost = Authority::Ghost {
            source_fence: Fence(1),
            since_tick: TickId(0),
        };
        let frozen = Authority::Frozen {
            transfer: T,
            fence: Fence(1),
        };
        let illegal: Vec<(Authority, AuthorityCmd)> = vec![
            (
                owned,
                AuthorityCmd::Promote {
                    new_fence: Fence(2),
                },
            ),
            (owned, AuthorityCmd::Thaw { transfer: T }),
            (
                owned,
                AuthorityCmd::Demote {
                    new_owner_fence: Fence(2),
                    at_tick: TickId(0),
                },
            ),
            (
                owned,
                AuthorityCmd::GhostRefresh {
                    source_fence: Fence(2),
                },
            ),
            (ghost, AuthorityCmd::Freeze { transfer: T }),
            (ghost, AuthorityCmd::Thaw { transfer: T }),
            (
                ghost,
                AuthorityCmd::Demote {
                    new_owner_fence: Fence(2),
                    at_tick: TickId(0),
                },
            ),
            (frozen, AuthorityCmd::Freeze { transfer: T }),
            (
                frozen,
                AuthorityCmd::Promote {
                    new_fence: Fence(2),
                },
            ),
            (
                frozen,
                AuthorityCmd::GhostRefresh {
                    source_fence: Fence(2),
                },
            ),
        ];
        for (state, cmd) in illegal {
            assert_eq!(
                state.apply(cmd),
                Err(AuthorityError::IllegalTransition),
                "{state:?} x {cmd:?}"
            );
        }
    }

    #[test]
    fn errors_display() {
        assert_eq!(
            AuthorityError::StaleFence {
                carried: Fence(1),
                held: Fence(2)
            }
            .to_string(),
            "stale fence: carried fence-1, held fence-2"
        );
        assert_eq!(
            AuthorityError::IllegalTransition.to_string(),
            "illegal transition for current authority state"
        );
        assert_eq!(
            AuthorityError::TransferMismatch.to_string(),
            "freeze/thaw transfer mismatch"
        );
    }

    fn arb_authority() -> impl Strategy<Value = Authority> {
        prop_oneof![
            (0u64..10).prop_map(|f| Authority::Owned { fence: Fence(f) }),
            (0u64..10, 0u64..10).prop_map(|(f, t)| Authority::Ghost {
                source_fence: Fence(f),
                since_tick: TickId(t)
            }),
            (0u64..10, 0u64..10).prop_map(|(f, x)| Authority::Frozen {
                transfer: TransferId(u128::from(x)),
                fence: Fence(f)
            }),
        ]
    }

    fn arb_cmd() -> impl Strategy<Value = AuthorityCmd> {
        prop_oneof![
            (0u64..10).prop_map(|x| AuthorityCmd::Freeze {
                transfer: TransferId(u128::from(x))
            }),
            (0u64..10).prop_map(|x| AuthorityCmd::Thaw {
                transfer: TransferId(u128::from(x))
            }),
            (0u64..10).prop_map(|f| AuthorityCmd::Promote {
                new_fence: Fence(f)
            }),
            (0u64..10, 0u64..10).prop_map(|(f, t)| AuthorityCmd::Demote {
                new_owner_fence: Fence(f),
                at_tick: TickId(t)
            }),
            (0u64..10).prop_map(|f| AuthorityCmd::GhostRefresh {
                source_fence: Fence(f)
            }),
        ]
    }

    proptest! {
        /// Totality + the fence-monotonicity invariant: a successful command never
        /// DECREASES the fence a state holds.
        #[test]
        fn fences_never_move_backward(state in arb_authority(), cmd in arb_cmd()) {
            let before = state.fence();
            if let Ok(after) = state.apply(cmd) {
                prop_assert!(after.fence() >= before,
                    "fence regressed: {state:?} x {cmd:?} -> {after:?}");
            }
        }

        /// Only Owned simulates, in every reachable state.
        #[test]
        fn only_owned_simulates(state in arb_authority(), cmd in arb_cmd()) {
            if let Ok(after) = state.apply(cmd) {
                prop_assert_eq!(after.simulates(), matches!(after, Authority::Owned { .. }));
            }
        }
    }
}
