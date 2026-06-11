//! The Transfer Saga FSM — THE single owner of a transfer's lifecycle (the cure for
//! R1: the old system had no owner, no timeouts, no compensation, and one lost ack
//! wedged a player forever).
//!
//! PURE: `step(state, event) -> (state', actions)` — no I/O, no clocks, no channels.
//! The orchestrator wraps it with persistence (group-committed WAL at the two
//! crash-critical points) and at-least-once delivery; the harness drives it with a
//! virtual clock and fault injection. Proptests below assert global properties:
//! totality (no event panics in any state), terminality (every run ends `Done` or
//! `Aborted`), and the compensation invariant (a frozen source is ALWAYS thawed or
//! committed — never left frozen).
//!
//! Pipeline (integration resolution of the 3-way COMMIT conflict):
//! ```text
//! [AwaitProvision →] Preparing → Cutting → Freezing → CommittingCas → Swapping
//!        → Demoting → Releasing → Done
//! any pre-CAS failure → Aborting (thaw if frozen, abort the dest) → Aborted
//! ```
//! The directory CAS is the commit point; everything after it is forward-only.
//! Source authority is retained until the CAS wins — a failed/slow destination
//! always returns the player to a live source (B2: never "warp into nothing").

use serde::{Deserialize, Serialize};
use vd_core::entity_kind::DurabilityClass;
use vd_core::{Fence, NodeId, SessionId, TransferId};
use vd_wire::seams::directory::DirectoryKey;
use vd_wire::seams::transfer_control::{PrepareReject, PrepareResult, TransferControl};

/// Immutable per-saga context, fixed at creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SagaCtx {
    pub transfer: TransferId,
    pub session: SessionId,
    pub subject: DirectoryKey,
    /// The fence the subject held when the saga was created — the CAS expectation.
    pub expected_fence: Fence,
    pub source: NodeId,
    pub dest: NodeId,
    /// HR2: the durability class drives the commit fan-out on ONE machinery —
    /// `Durable` rides the full per-entity FSM to a directory CAS; `Transient`
    /// (debris/projectiles) commits via the BATCHED `TransientGo` go-token (one
    /// orchestrator fsync per batch, not per item). Carried here so the persisted
    /// checkpoint and the FSM proptests are class-aware from the first transfer.
    pub class: DurabilityClass,
    /// Warp-class transfers must await destination provisioning first.
    pub needs_provision: bool,
}

/// The saga's phase. Serializable: the two durable checkpoints persist it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SagaState {
    /// Waiting for the destination shard to exist (warp class only).
    AwaitProvision,
    /// `PrepareSubscribe` issued; awaiting ghost-ready / spatial verdict.
    Preparing,
    /// `RequestCut` issued; awaiting the client's confirmed CUT_MARKER seq.
    Cutting,
    /// `FreezeSource` issued; the source applies input through `marker_seq`.
    Freezing { marker_seq: u64 },
    /// The directory CAS is in flight — THE commit point.
    CommittingCas { marker_seq: u64, drained_seq: u64 },
    /// CAS won: the gateway route swap (`CommitAuthority`) is in flight.
    Swapping { new_fence: Fence },
    /// Authority flipped; the source ghost persists until the destination acks
    /// delivered-to-all-observers AND the entity leaves the overlap band
    /// (event-driven demotion — never a tick count).
    Demoting { new_fence: Fence },
    /// `ReleaseSubscribe` issued for the source subscription.
    Releasing { new_fence: Fence },
    /// Terminal success.
    Done { new_fence: Fence },
    /// Compensation in flight: thaw the source if frozen, abort the destination.
    Aborting {
        reason: AbortReason,
        /// `ThawSource` is outstanding (the FreezeSource compensator).
        awaiting_thaw: bool,
        /// `AbortTransfer` (dest teardown) is outstanding.
        awaiting_abort_ack: bool,
    },
    /// Terminal failure: the subject is alive and authoritative on the SOURCE.
    Aborted { reason: AbortReason },
}

/// Why a saga aborted. Typed end-to-end (client feedback derives from this).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbortReason {
    ProvisionFailed,
    ProvisionTimeout,
    PrepareRejected(PrepareReject),
    PrepareTimeout,
    CutTimeout,
    FreezeTimeout,
    /// The CAS lost: someone else moved the fence — this saga no-ops (rule 3).
    CasLost,
    /// Operator/system requested cancellation before the commit point.
    Cancelled,
}

/// Everything the saga can observe. Delivery is at-least-once: every handler is
/// idempotent (duplicates and stale events are no-ops).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SagaEvent {
    ProvisionReady,
    ProvisionFailed,
    Prepared(PrepareResult),
    CutConfirmed {
        marker_seq: u64,
    },
    SourceFrozen {
        drained_seq: u64,
    },
    CasWon {
        new_fence: Fence,
    },
    CasLost {
        current: Fence,
    },
    RouteSwapped,
    /// Dest acked observer delivery AND the entity left the overlap band.
    DemoteComplete,
    Released,
    SourceThawed,
    DestAborted,
    /// The phase's adaptive deadline (k·RTT, hard-capped) elapsed.
    Timeout,
    /// Pre-commit cancellation request.
    Cancel,
}

/// What the wrapper must do after a step. The saga never performs effects itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SagaAction {
    Send(TransferControl),
    /// Issue the directory CAS (commit point for a DURABLE subject). Carries the
    /// expectation; the wrapper fills `transfer`/`new_owner` from the `SagaCtx`.
    IssueCommitCas {
        expected: Fence,
    },
    /// Issue the BATCHED `TransientGo` go-token (commit point for a TRANSIENT
    /// subject) — the same Fence-CAS amortized across a batch (HR2). Present and
    /// classified now; the batched transient flow is driven at P3.
    IssueTransientGo {
        expected: Fence,
    },
    /// Durable checkpoint — exactly two per happy path (after Prepared, at CAS won),
    /// group-committed by the wrapper.
    PersistCheckpoint,
    /// Surface a typed rejection to the client (cosmetic feedback).
    NotifyRejected(AbortReason),
    /// The saga reached a terminal state; the wrapper may GC after tombstoning.
    Tombstone,
}

/// The class-appropriate commit action (HR2 fan-out on ONE FSM): Durable subjects
/// commit via the directory CAS, Transient subjects via the batched go-token.
#[must_use]
fn commit_action(ctx: &SagaCtx) -> SagaAction {
    match ctx.class {
        DurabilityClass::Durable => SagaAction::IssueCommitCas {
            expected: ctx.expected_fence,
        },
        DurabilityClass::Transient => SagaAction::IssueTransientGo {
            expected: ctx.expected_fence,
        },
    }
}

/// The initial state + actions for a freshly created saga.
#[must_use]
pub fn start(ctx: &SagaCtx) -> (SagaState, Vec<SagaAction>) {
    if ctx.needs_provision {
        (SagaState::AwaitProvision, vec![])
    } else {
        (
            SagaState::Preparing,
            vec![SagaAction::Send(TransferControl::PrepareSubscribe {
                transfer: ctx.transfer,
                session: ctx.session,
                dest: ctx.dest,
            })],
        )
    }
}

/// THE transition function. Total over (state, event); unknown combinations are
/// idempotent no-ops (at-least-once delivery makes duplicates routine, not errors).
#[must_use]
pub fn step(ctx: &SagaCtx, state: SagaState, event: SagaEvent) -> (SagaState, Vec<SagaAction>) {
    use SagaAction as A;
    use SagaEvent as E;
    use SagaState as S;

    match (state, event) {
        // ---- provisioning (warp class) ------------------------------------------
        (S::AwaitProvision, E::ProvisionReady) => (
            S::Preparing,
            vec![A::Send(TransferControl::PrepareSubscribe {
                transfer: ctx.transfer,
                session: ctx.session,
                dest: ctx.dest,
            })],
        ),
        (S::AwaitProvision, E::ProvisionFailed) => {
            abort_from_pre_freeze(ctx, AbortReason::ProvisionFailed)
        }
        (S::AwaitProvision, E::Timeout) => {
            abort_from_pre_freeze(ctx, AbortReason::ProvisionTimeout)
        }
        (S::AwaitProvision, E::Cancel) => abort_from_pre_freeze(ctx, AbortReason::Cancelled),

        // ---- prepare --------------------------------------------------------------
        (S::Preparing, E::Prepared(PrepareResult::Ready)) => (
            S::Cutting,
            vec![
                A::PersistCheckpoint, // durable point 1: the dest ghost exists
                A::Send(TransferControl::RequestCut {
                    transfer: ctx.transfer,
                    session: ctx.session,
                }),
            ],
        ),
        (S::Preparing, E::Prepared(PrepareResult::Rejected(reject))) => {
            abort_from_pre_freeze(ctx, AbortReason::PrepareRejected(reject))
        }
        (S::Preparing, E::Timeout) => abort_from_pre_freeze(ctx, AbortReason::PrepareTimeout),
        (S::Preparing, E::Cancel) => abort_from_pre_freeze(ctx, AbortReason::Cancelled),

        // ---- cut ------------------------------------------------------------------
        (S::Cutting, E::CutConfirmed { marker_seq }) => (
            S::Freezing { marker_seq },
            vec![A::Send(TransferControl::FreezeSource {
                transfer: ctx.transfer,
                session: ctx.session,
                marker_seq,
                dest: ctx.dest,
            })],
        ),
        (S::Cutting, E::Timeout) => abort_from_pre_freeze(ctx, AbortReason::CutTimeout),
        (S::Cutting, E::Cancel) => abort_from_pre_freeze(ctx, AbortReason::Cancelled),

        // ---- freeze ----------------------------------------------------------------
        (S::Freezing { marker_seq }, E::SourceFrozen { drained_seq }) => (
            S::CommittingCas {
                marker_seq,
                drained_seq,
            },
            vec![commit_action(ctx)],
        ),
        // Freeze failed/timed out: the source MUST be thawed (the compensator).
        (S::Freezing { .. }, E::Timeout) => abort_with_thaw(ctx, AbortReason::FreezeTimeout),
        (S::Freezing { .. }, E::Cancel) => abort_with_thaw(ctx, AbortReason::Cancelled),

        // ---- the commit point --------------------------------------------------------
        (S::CommittingCas { .. }, E::CasWon { new_fence }) => (
            S::Swapping { new_fence },
            vec![
                A::PersistCheckpoint, // durable point 2: authority flipped
                A::Send(TransferControl::CommitAuthority {
                    transfer: ctx.transfer,
                    session: ctx.session,
                    new_fence,
                }),
            ],
        ),
        // The CAS lost: someone else moved the fence. This saga no-ops and unwinds
        // — the source was never demoted, so thaw it and tear down the dest.
        (S::CommittingCas { .. }, E::CasLost { .. }) => abort_with_thaw(ctx, AbortReason::CasLost),
        // A CAS in flight cannot time out into an abort: it may have WON durably.
        // The wrapper must re-read the directory head and re-deliver CasWon/CasLost
        // (this is why Timeout here re-issues, never aborts).
        (S::CommittingCas { .. }, E::Timeout) => (state, vec![commit_action(ctx)]),

        // ---- post-commit: forward-only -------------------------------------------------
        (S::Swapping { new_fence }, E::RouteSwapped) => (S::Demoting { new_fence }, vec![]),
        // Post-commit timeouts retry the same command (idempotent), never abort:
        // the directory already says the dest owns the subject.
        (S::Swapping { new_fence }, E::Timeout) => (
            S::Swapping { new_fence },
            vec![A::Send(TransferControl::CommitAuthority {
                transfer: ctx.transfer,
                session: ctx.session,
                new_fence,
            })],
        ),
        (S::Demoting { new_fence }, E::DemoteComplete) => (
            S::Releasing { new_fence },
            vec![A::Send(TransferControl::ReleaseSubscribe {
                transfer: ctx.transfer,
                session: ctx.session,
                src: ctx.source,
            })],
        ),
        (S::Releasing { new_fence }, E::Released) => (S::Done { new_fence }, vec![A::Tombstone]),
        (S::Releasing { new_fence }, E::Timeout) => (
            S::Releasing { new_fence },
            vec![A::Send(TransferControl::ReleaseSubscribe {
                transfer: ctx.transfer,
                session: ctx.session,
                src: ctx.source,
            })],
        ),

        // ---- compensation ---------------------------------------------------------------
        (
            S::Aborting {
                reason,
                awaiting_thaw,
                awaiting_abort_ack,
            },
            event,
        ) => step_aborting(reason, awaiting_thaw, awaiting_abort_ack, event),

        // ---- idempotent no-ops -------------------------------------------------------------
        // Terminal states absorb everything; unknown (state, event) pairs are
        // duplicates or stale deliveries under at-least-once — never errors.
        (terminal @ (S::Done { .. } | S::Aborted { .. }), _) => (terminal, vec![]),
        (state, _) => (state, vec![]),
    }
}

/// Abort before any freeze happened: tear down the dest; the source never stopped.
fn abort_from_pre_freeze(ctx: &SagaCtx, reason: AbortReason) -> (SagaState, Vec<SagaAction>) {
    (
        SagaState::Aborting {
            reason,
            awaiting_thaw: false,
            awaiting_abort_ack: true,
        },
        vec![
            SagaAction::Send(TransferControl::AbortTransfer {
                transfer: ctx.transfer,
                session: ctx.session,
            }),
            SagaAction::NotifyRejected(reason),
        ],
    )
}

/// Abort after FreezeSource was issued: the compensator chain MUST thaw the source.
fn abort_with_thaw(ctx: &SagaCtx, reason: AbortReason) -> (SagaState, Vec<SagaAction>) {
    (
        SagaState::Aborting {
            reason,
            awaiting_thaw: true,
            awaiting_abort_ack: true,
        },
        vec![
            SagaAction::Send(TransferControl::ThawSource {
                transfer: ctx.transfer,
                session: ctx.session,
            }),
            SagaAction::Send(TransferControl::AbortTransfer {
                transfer: ctx.transfer,
                session: ctx.session,
            }),
            SagaAction::NotifyRejected(reason),
        ],
    )
}

/// Compensation bookkeeping: both acks must arrive (in any order, any number of
/// times) before the saga is terminally `Aborted`.
fn step_aborting(
    reason: AbortReason,
    awaiting_thaw: bool,
    awaiting_abort_ack: bool,
    event: SagaEvent,
) -> (SagaState, Vec<SagaAction>) {
    let (awaiting_thaw, awaiting_abort_ack) = match event {
        SagaEvent::SourceThawed => (false, awaiting_abort_ack),
        SagaEvent::DestAborted => (awaiting_thaw, false),
        // Timeouts in compensation are handled by the wrapper's retry of the
        // outstanding commands (idempotent); the FSM state is unchanged.
        _ => (awaiting_thaw, awaiting_abort_ack),
    };
    if awaiting_thaw || awaiting_abort_ack {
        (
            SagaState::Aborting {
                reason,
                awaiting_thaw,
                awaiting_abort_ack,
            },
            vec![],
        )
    } else {
        (SagaState::Aborted { reason }, vec![SagaAction::Tombstone])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use vd_core::pose::RealmId;
    use vd_wire::seams::transfer_control::SpatialReject;

    fn ctx(needs_provision: bool) -> SagaCtx {
        ctx_class(needs_provision, DurabilityClass::Durable)
    }

    fn ctx_class(needs_provision: bool, class: DurabilityClass) -> SagaCtx {
        SagaCtx {
            transfer: TransferId(1),
            session: SessionId(2),
            subject: DirectoryKey::Realm(RealmId::System(3)),
            expected_fence: Fence(5),
            source: NodeId(10),
            dest: NodeId(20),
            class,
            needs_provision,
        }
    }

    /// Drive a state through events, collecting all actions.
    fn drive(
        c: &SagaCtx,
        mut state: SagaState,
        events: &[SagaEvent],
    ) -> (SagaState, Vec<SagaAction>) {
        let mut all = Vec::new();
        for &e in events {
            let (next, actions) = step(c, state, e);
            state = next;
            all.extend(actions);
        }
        (state, all)
    }

    #[test]
    fn durable_commit_issues_a_directory_cas() {
        let c = ctx(false); // Durable
        let (state, actions) = drive_from_freeze(&c);
        assert_eq!(
            state,
            SagaState::CommittingCas {
                marker_seq: 9,
                drained_seq: 9,
            }
        );
        assert!(
            actions.contains(&SagaAction::IssueCommitCas { expected: Fence(5) }),
            "Durable commits via the directory CAS: {actions:?}"
        );
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, SagaAction::IssueTransientGo { .. })),
            "no go-token for a Durable subject"
        );
    }

    #[test]
    fn transient_commit_issues_the_batched_go_token() {
        // HR2 fan-out on ONE FSM: a Transient subject reaches the same commit point
        // but issues the batched TransientGo go-token, not a per-entity CAS.
        let c = ctx_class(false, DurabilityClass::Transient);
        let (_state, actions) = drive_from_freeze(&c);
        assert!(
            actions.contains(&SagaAction::IssueTransientGo { expected: Fence(5) }),
            "Transient commits via the batched go-token: {actions:?}"
        );
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, SagaAction::IssueCommitCas { .. })),
            "no per-entity CAS for a Transient subject"
        );
    }

    /// Drive a fresh saga to the commit-issue point and return (state, actions there).
    fn drive_from_freeze(c: &SagaCtx) -> (SagaState, Vec<SagaAction>) {
        let (state, _) = drive(
            c,
            start(c).0,
            &[
                SagaEvent::Prepared(PrepareResult::Ready),
                SagaEvent::CutConfirmed { marker_seq: 9 },
            ],
        );
        // The SourceFrozen step is where the commit action is emitted.
        step(c, state, SagaEvent::SourceFrozen { drained_seq: 9 })
    }

    #[test]
    fn happy_path_reaches_done_with_two_checkpoints() {
        let c = ctx(false);
        let (state, mut actions) = start(&c);
        assert_eq!(state, SagaState::Preparing);
        let (state, rest) = drive(
            &c,
            state,
            &[
                SagaEvent::Prepared(PrepareResult::Ready),
                SagaEvent::CutConfirmed { marker_seq: 17 },
                SagaEvent::SourceFrozen { drained_seq: 17 },
                SagaEvent::CasWon {
                    new_fence: Fence(6),
                },
                SagaEvent::RouteSwapped,
                SagaEvent::DemoteComplete,
                SagaEvent::Released,
            ],
        );
        actions.extend(rest);
        assert_eq!(
            state,
            SagaState::Done {
                new_fence: Fence(6)
            }
        );

        let checkpoints = actions
            .iter()
            .filter(|a| **a == SagaAction::PersistCheckpoint)
            .count();
        assert_eq!(
            checkpoints, 2,
            "exactly two durable points on the happy path"
        );
        assert_eq!(actions.last(), Some(&SagaAction::Tombstone));

        // The full command sequence, in order.
        let sends: Vec<TransferControl> = actions
            .iter()
            .filter_map(|a| match a {
                SagaAction::Send(cmd) => Some(*cmd),
                _ => None,
            })
            .collect();
        assert_eq!(
            sends,
            vec![
                TransferControl::PrepareSubscribe {
                    transfer: c.transfer,
                    session: c.session,
                    dest: c.dest,
                },
                TransferControl::RequestCut {
                    transfer: c.transfer,
                    session: c.session,
                },
                TransferControl::FreezeSource {
                    transfer: c.transfer,
                    session: c.session,
                    marker_seq: 17,
                    dest: c.dest,
                },
                TransferControl::CommitAuthority {
                    transfer: c.transfer,
                    session: c.session,
                    new_fence: Fence(6),
                },
                TransferControl::ReleaseSubscribe {
                    transfer: c.transfer,
                    session: c.session,
                    src: c.source,
                },
            ]
        );
    }

    #[test]
    fn warp_class_awaits_provision_and_aborts_safely_on_failure() {
        let c = ctx(true);
        let (state, actions) = start(&c);
        assert_eq!(state, SagaState::AwaitProvision);
        assert!(actions.is_empty(), "nothing to do until the dest exists");

        // Provisioning failure: the player snaps back to a live source — the saga
        // never touched it (no thaw needed).
        let (state, actions) = step(&c, state, SagaEvent::ProvisionFailed);
        assert_eq!(
            state,
            SagaState::Aborting {
                reason: AbortReason::ProvisionFailed,
                awaiting_thaw: false,
                awaiting_abort_ack: true,
            }
        );
        assert!(actions.contains(&SagaAction::NotifyRejected(AbortReason::ProvisionFailed)));
        let (state, _) = step(&c, state, SagaEvent::DestAborted);
        assert_eq!(
            state,
            SagaState::Aborted {
                reason: AbortReason::ProvisionFailed
            }
        );
    }

    #[test]
    fn provision_success_proceeds_to_prepare() {
        let c = ctx(true);
        let (state, _) = start(&c);
        let (state, actions) = step(&c, state, SagaEvent::ProvisionReady);
        assert_eq!(state, SagaState::Preparing);
        assert_eq!(
            actions,
            vec![SagaAction::Send(TransferControl::PrepareSubscribe {
                transfer: c.transfer,
                session: c.session,
                dest: c.dest,
            })]
        );
    }

    #[test]
    fn spatial_rejection_aborts_with_typed_feedback() {
        let c = ctx(false);
        let (state, _) = start(&c);
        let reject = PrepareReject::Spatial(SpatialReject::Obstructed);
        let (state, actions) = step(
            &c,
            state,
            SagaEvent::Prepared(PrepareResult::Rejected(reject)),
        );
        let expected_reason = AbortReason::PrepareRejected(reject);
        assert!(actions.contains(&SagaAction::NotifyRejected(expected_reason)));
        // No thaw: the source was never frozen.
        let has_thaw = actions
            .iter()
            .any(|a| matches!(a, SagaAction::Send(TransferControl::ThawSource { .. })));
        assert!(!has_thaw);
        let (state, _) = step(&c, state, SagaEvent::DestAborted);
        assert_eq!(
            state,
            SagaState::Aborted {
                reason: expected_reason
            }
        );
    }

    #[test]
    fn freeze_timeout_always_thaws_the_source() {
        let c = ctx(false);
        let (state, _) = drive(
            &c,
            start(&c).0,
            &[
                SagaEvent::Prepared(PrepareResult::Ready),
                SagaEvent::CutConfirmed { marker_seq: 5 },
            ],
        );
        assert_eq!(state, SagaState::Freezing { marker_seq: 5 });
        let (state, actions) = step(&c, state, SagaEvent::Timeout);
        let thaw = SagaAction::Send(TransferControl::ThawSource {
            transfer: c.transfer,
            session: c.session,
        });
        assert!(actions.contains(&thaw), "the compensator MUST fire");
        // Acks may arrive in any order; both are required for terminality.
        let (state, _) = step(&c, state, SagaEvent::DestAborted);
        let (state, _) = step(&c, state, SagaEvent::SourceThawed);
        assert_eq!(
            state,
            SagaState::Aborted {
                reason: AbortReason::FreezeTimeout
            }
        );
    }

    #[test]
    fn cancel_during_freeze_thaws_the_source() {
        // The deterministic twin of the (Freezing, Cancel) arm — the proptests only
        // hit it probabilistically, and coverage must never depend on random draws.
        let c = ctx(false);
        let (state, _) = drive(
            &c,
            start(&c).0,
            &[
                SagaEvent::Prepared(PrepareResult::Ready),
                SagaEvent::CutConfirmed { marker_seq: 5 },
            ],
        );
        let (state, actions) = step(&c, state, SagaEvent::Cancel);
        let thaw = SagaAction::Send(TransferControl::ThawSource {
            transfer: c.transfer,
            session: c.session,
        });
        assert!(actions.contains(&thaw), "the compensator MUST fire");
        let (state, _) = step(&c, state, SagaEvent::DestAborted);
        let (state, _) = step(&c, state, SagaEvent::SourceThawed);
        assert_eq!(
            state,
            SagaState::Aborted {
                reason: AbortReason::Cancelled
            }
        );
    }

    #[test]
    fn cas_loser_unwinds_with_thaw() {
        let c = ctx(false);
        let (state, _) = drive(
            &c,
            start(&c).0,
            &[
                SagaEvent::Prepared(PrepareResult::Ready),
                SagaEvent::CutConfirmed { marker_seq: 9 },
                SagaEvent::SourceFrozen { drained_seq: 9 },
            ],
        );
        let (state, actions) = step(&c, state, SagaEvent::CasLost { current: Fence(99) });
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, SagaAction::Send(TransferControl::ThawSource { .. }))),
            "the frozen source must thaw when the CAS loses"
        );
        let (state, _) = drive(
            &c,
            state,
            &[SagaEvent::SourceThawed, SagaEvent::DestAborted],
        );
        assert_eq!(
            state,
            SagaState::Aborted {
                reason: AbortReason::CasLost
            }
        );
    }

    #[test]
    fn cas_timeout_reissues_never_aborts() {
        // A CAS may have durably WON while its reply was lost: timing out into an
        // abort would risk double-authority. The only legal move is re-issue.
        let c = ctx(false);
        let state = SagaState::CommittingCas {
            marker_seq: 1,
            drained_seq: 1,
        };
        let (next, actions) = step(&c, state, SagaEvent::Timeout);
        assert_eq!(next, state);
        assert_eq!(
            actions,
            vec![SagaAction::IssueCommitCas {
                expected: c.expected_fence
            }]
        );
    }

    #[test]
    fn post_commit_timeouts_retry_forward_only() {
        let c = ctx(false);
        let state = SagaState::Swapping {
            new_fence: Fence(7),
        };
        let (next, actions) = step(&c, state, SagaEvent::Timeout);
        assert_eq!(next, state);
        assert_eq!(
            actions,
            vec![SagaAction::Send(TransferControl::CommitAuthority {
                transfer: c.transfer,
                session: c.session,
                new_fence: Fence(7),
            })]
        );
        let state = SagaState::Releasing {
            new_fence: Fence(7),
        };
        let (next, actions) = step(&c, state, SagaEvent::Timeout);
        assert_eq!(next, state);
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn duplicates_and_stale_events_are_noops() {
        let c = ctx(false);
        // A duplicate Prepared in Cutting state: no-op.
        let state = SagaState::Cutting;
        let (next, actions) = step(&c, state, SagaEvent::Prepared(PrepareResult::Ready));
        assert_eq!(next, state);
        assert!(actions.is_empty());
        // Terminal states absorb everything.
        for terminal in [
            SagaState::Done {
                new_fence: Fence(1),
            },
            SagaState::Aborted {
                reason: AbortReason::CasLost,
            },
        ] {
            let (next, actions) = step(&c, terminal, SagaEvent::Timeout);
            assert_eq!(next, terminal);
            assert!(actions.is_empty());
        }
    }

    // ---- proptests over the whole machine -------------------------------------------

    fn arb_event() -> impl Strategy<Value = SagaEvent> {
        prop_oneof![
            Just(SagaEvent::ProvisionReady),
            Just(SagaEvent::ProvisionFailed),
            Just(SagaEvent::Prepared(PrepareResult::Ready)),
            Just(SagaEvent::Prepared(PrepareResult::Rejected(
                PrepareReject::VersionFloor
            ))),
            (0u64..100).prop_map(|s| SagaEvent::CutConfirmed { marker_seq: s }),
            (0u64..100).prop_map(|s| SagaEvent::SourceFrozen { drained_seq: s }),
            (0u64..10).prop_map(|f| SagaEvent::CasWon {
                new_fence: Fence(f)
            }),
            (0u64..10).prop_map(|f| SagaEvent::CasLost { current: Fence(f) }),
            Just(SagaEvent::RouteSwapped),
            Just(SagaEvent::DemoteComplete),
            Just(SagaEvent::Released),
            Just(SagaEvent::SourceThawed),
            Just(SagaEvent::DestAborted),
            Just(SagaEvent::Timeout),
            Just(SagaEvent::Cancel),
        ]
    }

    proptest! {
        /// Totality: no event sequence panics, and every state the machine visits
        /// keeps the compensation invariant — if FreezeSource was ever sent, then by
        /// the time the machine is terminal, ThawSource or CommitAuthority was sent.
        #[test]
        fn no_sequence_panics_and_frozen_sources_are_never_stranded(
            provision in proptest::bool::ANY,
            events in proptest::collection::vec(arb_event(), 0..64),
        ) {
            let c = ctx(provision);
            let (mut state, mut actions) = start(&c);
            for e in events {
                let (next, acts) = step(&c, state, e);
                state = next;
                actions.extend(acts);
            }
            let froze = actions.iter().any(|a| matches!(
                a, SagaAction::Send(TransferControl::FreezeSource { .. })));
            let resolved = actions.iter().any(|a| matches!(
                a,
                SagaAction::Send(TransferControl::ThawSource { .. })
                    | SagaAction::Send(TransferControl::CommitAuthority { .. })
            ));
            if froze && matches!(state, SagaState::Aborted { .. } | SagaState::Done { .. }) {
                prop_assert!(resolved, "frozen source neither thawed nor committed");
            }
        }

        /// Terminality: feeding the machine its own expected progression events
        /// (whatever state it is in) always reaches Done or Aborted in bounded steps.
        #[test]
        fn driving_with_matching_events_terminates(provision in proptest::bool::ANY) {
            let c = ctx(provision);
            let (mut state, _) = start(&c);
            for _ in 0..16 {
                let event = match state {
                    SagaState::AwaitProvision => SagaEvent::ProvisionReady,
                    SagaState::Preparing => SagaEvent::Prepared(PrepareResult::Ready),
                    SagaState::Cutting => SagaEvent::CutConfirmed { marker_seq: 1 },
                    SagaState::Freezing { .. } => SagaEvent::SourceFrozen { drained_seq: 1 },
                    SagaState::CommittingCas { .. } => SagaEvent::CasWon { new_fence: Fence(9) },
                    SagaState::Swapping { .. } => SagaEvent::RouteSwapped,
                    SagaState::Demoting { .. } => SagaEvent::DemoteComplete,
                    SagaState::Releasing { .. } => SagaEvent::Released,
                    SagaState::Aborting { awaiting_thaw: true, .. } => SagaEvent::SourceThawed,
                    SagaState::Aborting { .. } => SagaEvent::DestAborted,
                    SagaState::Done { .. } | SagaState::Aborted { .. } => break,
                };
                let (next, _) = step(&c, state, event);
                state = next;
            }
            prop_assert!(
                matches!(state, SagaState::Done { .. } | SagaState::Aborted { .. }),
                "did not terminate: {state:?}"
            );
        }

        /// The CAS commit point is a one-way door: after CasWon, no sequence of
        /// events ever produces a ThawSource or AbortTransfer.
        #[test]
        fn post_commit_is_forward_only(
            events in proptest::collection::vec(arb_event(), 0..64),
        ) {
            let c = ctx(false);
            let mut state = SagaState::Swapping { new_fence: Fence(6) };
            let mut actions = Vec::new();
            for e in events {
                let (next, acts) = step(&c, state, e);
                state = next;
                actions.extend(acts);
            }
            let unwound = actions.iter().any(|a| matches!(
                a,
                SagaAction::Send(TransferControl::ThawSource { .. })
                    | SagaAction::Send(TransferControl::AbortTransfer { .. })
            ));
            prop_assert!(!unwound, "post-commit compensation is forbidden");
        }
    }
}
