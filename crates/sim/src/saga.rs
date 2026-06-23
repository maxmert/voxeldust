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
//!        → Demoting → Promoting → Releasing → Done
//! any pre-CAS failure → Aborting (thaw if frozen, abort the dest) → Aborted
//! ```
//! The directory CAS is the commit point; everything after it is forward-only.
//! Source authority is retained until the CAS wins — a failed/slow destination
//! always returns the player to a live source (B2: never "warp into nothing").

use serde::{Deserialize, Serialize};
use vd_core::entity_kind::DurabilityClass;
use vd_core::pose::RealmId;
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
    /// The realms the subject crosses BETWEEN (1d.1): stamped onto the `StubCrossing` envelope the
    /// saga emits to the dest at commit. Carried VERBATIM (the saga never matches on realm — HR3);
    /// the dest finds the adopted dot by entity, so these are faithful provenance, not a key.
    pub from_realm: RealmId,
    pub to_realm: RealmId,
}

/// The saga deadline budget (Slice 2a) — the ONE home for the two timeout thresholds (HR "no magic
/// numbers"; never an inline literal in the producer). Two values because the FSM's `Timeout` splits
/// by phase into two RISK CLASSES:
/// - `redrive_deadline_ticks` — the CHEAP, idempotent re-drive of a POST-COMMIT step (re-emit a
///   Demote/Promote/crossing the consumer re-acks WITHOUT re-effect). Firing "early" on a merely-slow
///   saga costs nothing, so this can be tight (fast recovery from a lost saga-ack).
/// - `abort_deadline_ticks` — the DESTRUCTIVE pre-freeze/freeze/compensation timeout (aborts a
///   possibly-healthy player back to the source). MUST be sized FAR above the deployment's worst-case
///   healthy pre-freeze round-trip, else a slow-but-alive link triggers an abort storm. ⚠️ INTERIM: a
///   fixed tick deadline is NOT a crash-vs-slow discriminator (only a kill emits `NodeUnreachable`);
///   the real trigger is lease-lapse liveness (D-3, owed). Sized per deployment as
///   `ceil(MULT · (round_trip + retry_delay + 2·max_extra_delay + stagger))`, MULT ≥ 1.5.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SagaTuning {
    pub redrive_deadline_ticks: u64,
    pub abort_deadline_ticks: u64,
}

/// DEV/test deadline defaults — the lockstep-capstone-safe values (worst healthy post-commit dwell
/// ≈ 3 ticks ≪ 8; worst healthy pre-freeze dwell ≈ 3 ≪ 24). PRODUCTION sets both explicitly from
/// config (`OrchestratorConfig.saga`, env-driven in the bin) per the false-timeout formula — NEVER a
/// `0` (which would fire every tick = abort storm), which is why `SagaTuning` derives no zero Default.
pub const DEFAULT_REDRIVE_DEADLINE_TICKS: u64 = 8;
pub const DEFAULT_ABORT_DEADLINE_TICKS: u64 = 24;

impl Default for SagaTuning {
    fn default() -> SagaTuning {
        SagaTuning {
            redrive_deadline_ticks: DEFAULT_REDRIVE_DEADLINE_TICKS,
            abort_deadline_ticks: DEFAULT_ABORT_DEADLINE_TICKS,
        }
    }
}

/// A mis-tuned [`SagaTuning`] — rejected LOUD at boot, never a silent abort storm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SagaTuningError {
    #[error(
        "saga deadline must be >= 1 tick (a 0-tick deadline fires EVERY scan = an abort storm)"
    )]
    ZeroDeadline,
    #[error(
        "abort_deadline_ticks ({abort}) must be >= redrive_deadline_ticks ({redrive}): the \
         DESTRUCTIVE abort deadline must never be tighter than the cheap re-drive"
    )]
    AbortTighterThanRedrive { redrive: u64, abort: u64 },
}

impl SagaTuning {
    /// Reject a mis-tuned budget at boot (the bin calls this after reading the env). Both deadlines
    /// must be ≥ 1 (a 0 fires every scan tick — the catastrophic abort storm), and the DESTRUCTIVE
    /// `abort_deadline_ticks` must be ≥ the cheap `redrive_deadline_ticks` (an abort tighter than the
    /// re-drive would abort a healthy saga before it ever re-drove). The bitwise `|` keeps both
    /// zero-check operands covered (HR5). The defaults + every in-process rig satisfy this trivially.
    ///
    /// # Errors
    /// [`SagaTuningError`] for a 0 deadline or an abort-below-redrive ordering.
    pub fn validate(&self) -> Result<(), SagaTuningError> {
        if (self.redrive_deadline_ticks < 1) | (self.abort_deadline_ticks < 1) {
            return Err(SagaTuningError::ZeroDeadline);
        }
        if self.abort_deadline_ticks < self.redrive_deadline_ticks {
            return Err(SagaTuningError::AbortTighterThanRedrive {
                redrive: self.redrive_deadline_ticks,
                abort: self.abort_deadline_ticks,
            });
        }
        Ok(())
    }
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
    /// `FreezeSource` + `FlushSource` issued. The POSE-BEFORE-PROMOTE gate: the commit (CAS) is
    /// reachable only once BOTH conditions land, in either order — `frozen_drained` (the gateway's
    /// `SourceFrozen` input-drain watermark) AND `flushed` (the source shipped its pose via
    /// `SourceFlushed`). A lost flush therefore BLOCKS the commit; the machine can never emit a
    /// poseless crossing (closing the silent-pose-loss hole). 1d.1.
    Freezing {
        marker_seq: u64,
        frozen_drained: Option<u64>,
        flushed: bool,
    },
    /// The directory CAS is in flight — THE commit point.
    CommittingCas { marker_seq: u64, drained_seq: u64 },
    /// CAS won: the gateway route swap (`CommitAuthority`) is in flight.
    Swapping { new_fence: Fence },
    /// Authority flipped; the saga has pushed the ORDERED `Demote` to the source — the
    /// fence-enforced `Owned→Frozen→Ghost` that REPLACES the 1c.8 cooperative poll (D-2).
    /// Awaiting the source's `DemoteAcked` (proof-of-freeze) BEFORE the dest is promoted — the
    /// binding demote-before-promote ordering. `dest_delivered` LATCHES an early `DestDelivered`:
    /// the gateway's standing delivery watermark can fire while still here (the dest is already
    /// delivering from its autonomous adopt-promote, retained in 1d.5b.1), so it is captured now
    /// and carried into `Promoting` — the release gate can never lose it (no park). 1d.5b.1.
    Demoting {
        new_fence: Fence,
        dest_delivered: bool,
    },
    /// `DemoteAcked` landed: the saga pushed the ordered `Promote` to the dest. The source
    /// subscription is RELEASED only once BOTH the dest acked the promote (`promote_acked`) AND
    /// delivered ≥1 frame to every observer (`dest_delivered`) — the seamless no-vanish gate (the
    /// FORK-0a overlap held through the dest's first frame). The two conditions land in either
    /// order; `promoting_advance` fires `ReleaseSubscribe` only when both are true. 1d.5b.1.
    Promoting {
        new_fence: Fence,
        promote_acked: bool,
        dest_delivered: bool,
    },
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
    /// The SOURCE shipped its authoritative pose (the reply to `FlushSource`). The pose itself
    /// rides the runtime wrapper (`LiveSaga.flushed_pose`), NOT this event — the FSM stays
    /// pose-free; `drained_seq` is the source's own drain watermark (carried for observability /
    /// a future cross-check, never the CAS watermark).
    SourceFlushed {
        drained_seq: u64,
    },
    CasWon {
        new_fence: Fence,
    },
    CasLost {
        current: Fence,
    },
    RouteSwapped,
    /// 1d.5b.1 (D-2) — the source acked it demoted `Owned→Frozen→Ghost` and stopped simulating
    /// (proof-of-freeze, the reply to the ordered `Demote`). Drives `Demoting → Promoting` (the
    /// dest promote is reachable ONLY after this — demote-before-promote).
    DemoteAcked,
    /// 1d.5b.1 (D-2) — the dest acked it holds the promote (the reply to the ordered `Promote`).
    /// One half of the `Promoting → Releasing` gate (with `DestDelivered`).
    PromoteAcked,
    /// 1d.5b.1 (D-2) — the gateway's STANDING delivery watermark: the dest delivered ≥1 frame to
    /// every current observer. Latched in `Demoting` (it can arrive before `DemoteAcked`) and the
    /// other half of the `Promoting → Releasing` gate — the source sub releases only AFTER the dest
    /// is delivered+rendered (the seamless no-vanish property).
    DestDelivered,
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
    /// Tell the SOURCE shard to ship the subject's authoritative pose (1d.1). Emitted alongside
    /// `Send(FreezeSource)` on entering `Freezing`. UNIT: the wrapper resolves the target
    /// (`SagaCtx.source`) and the step phase (`FLUSH_SOURCE_STEP`); the FSM holds no transport
    /// detail. Read-only at the source ⇒ no compensator (the abort `ThawSource` is the only undo).
    FlushSource,
    /// Emit the entity-STATE crossing (`StubCrossing`) to the DEST (1d.1), carrying the stored
    /// flushed pose at the post-CAS authority `fence`. Emitted in the `CasWon` batch (forward-only)
    /// and re-emitted on a post-commit `Swapping` timeout — the dest journal dedups by
    /// `(transfer, STUB_CROSSING_STEP)`. Carries `fence` (the new authority fence) so a receiver
    /// can reject a stale crossing (fence rule 1); the wrapper fills pose/realms/entity from the
    /// stored flush + `SagaCtx`.
    EmitCrossing {
        fence: Fence,
    },
    /// Push the ORDERED `Demote` to the SOURCE shard (1d.5b.1, D-2): demote the subject
    /// `Owned→Frozen→Ghost` at the post-CAS `new_owner_fence`. UNIT: the wrapper resolves the
    /// target (`SagaCtx.source`) + the step (`DEMOTE_STEP`); the FSM holds no transport detail.
    /// Pure egress to the source (its `DemoteAck` is the synchronous-free reply) — no compensator
    /// (post-commit, forward-only). Re-emitted on a `Demoting` timeout (idempotent at the source).
    Demote {
        new_owner_fence: Fence,
    },
    /// Push the ORDERED `Promote` to the DEST shard (1d.5b.1, D-2): confirm the subject
    /// `Ghost→Owned` at `new_fence`. Emitted ONLY after `DemoteAcked` (demote-before-promote).
    /// UNIT: the wrapper resolves the target (`SagaCtx.dest`) + the step (`PROMOTE_STEP`). Pure
    /// egress (its `PromoteAck` is the reply). Re-emitted on a `Promoting` timeout (idempotent).
    Promote {
        new_fence: Fence,
    },
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
    /// Clear the directory transfer-lock for the subject (Slice 2a, closes D-1). Emitted ONLY at the
    /// TERMINAL `Aborted` edge (after BOTH compensator acks — so the gateway's `apply_abort` has
    /// pruned the session journal, closing RACE-1: a re-transfer's `PrepareSubscribe` finds no stale
    /// journal to discard). The wrapper executes via `DirectoryCore::abort_clear` (the stale-fence
    /// re-read — the aborter's `expected_fence` is stale by definition). Idempotent (a re-driven
    /// terminal whose lock already cleared is a no-op). A successful COMMIT clears the lock at the CAS;
    /// this is the ABORT counterpart, so an aborted subject can immediately re-transfer.
    ClearTransferLock,
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
            S::Freezing {
                marker_seq,
                frozen_drained: None,
                flushed: false,
            },
            vec![
                A::Send(TransferControl::FreezeSource {
                    transfer: ctx.transfer,
                    session: ctx.session,
                    marker_seq,
                    dest: ctx.dest,
                }),
                A::FlushSource,
            ],
        ),
        (S::Cutting, E::Timeout) => abort_from_pre_freeze(ctx, AbortReason::CutTimeout),
        (S::Cutting, E::Cancel) => abort_from_pre_freeze(ctx, AbortReason::Cancelled),

        // ---- freeze (the pose-before-promote gate) ---------------------------------------
        // Both arms record their condition and defer the COMMIT decision to `freezing_advance`,
        // which fires the CAS only when BOTH conditions have landed (in any order).
        (
            S::Freezing {
                marker_seq,
                flushed,
                ..
            },
            E::SourceFrozen { drained_seq },
        ) => freezing_advance(ctx, marker_seq, Some(drained_seq), flushed),
        (
            S::Freezing {
                marker_seq,
                frozen_drained,
                ..
            },
            E::SourceFlushed { .. },
        ) => freezing_advance(ctx, marker_seq, frozen_drained, true),
        // Freeze failed/timed out: the source MUST be thawed (the compensator). A pending flush
        // needs no undo (it is a read on the source).
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
                    subject: ctx.subject, // carried VERBATIM (Realm or Entity); the dest adopts the Entity
                }),
                // The entity-STATE crossing rides the SAME post-CAS batch (forward-only). The gate
                // above guarantees a pose was flushed, so the wrapper always has one to ship.
                A::EmitCrossing { fence: new_fence },
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
        // The route swapped: push the ORDERED `Demote` to the source (the fence-enforced
        // `Owned→Frozen→Ghost` that REPLACES the 1c.8 poll). `dest_delivered` starts false.
        (S::Swapping { new_fence }, E::RouteSwapped) => (
            S::Demoting {
                new_fence,
                dest_delivered: false,
            },
            vec![A::Demote {
                new_owner_fence: new_fence,
            }],
        ),
        // Post-commit timeouts retry the same command (idempotent), never abort:
        // the directory already says the dest owns the subject.
        (S::Swapping { new_fence }, E::Timeout) => (
            S::Swapping { new_fence },
            vec![
                A::Send(TransferControl::CommitAuthority {
                    transfer: ctx.transfer,
                    session: ctx.session,
                    new_fence,
                    subject: ctx.subject, // idempotent re-send carries the same subject
                }),
                // Re-emit the crossing too (at-least-once for the entity state, same window as the
                // route-swap retry); the dest journal dedups a redelivery.
                A::EmitCrossing { fence: new_fence },
            ],
        ),
        // Latch an early `DestDelivered` so the `Promoting` release gate never loses it. Monotone
        // (idempotent re-arrival). ⚠️ PRODUCTION-DEAD since 1d.5b.3b: with the dest sub announced
        // only at PROMOTE (`SubscriptionReady` relocated to `on_saga_promote`, strict ordering), the
        // dest cannot DELIVER a frame while the saga is still in `Demoting` — `DestDelivered` can
        // only arrive in `Promoting`+. Kept as a DEFENSIVE/harness arm (the
        // `an_early_delivery_in_demoting` unit test injects the event directly; HR5 coverage holds).
        (S::Demoting { new_fence, .. }, E::DestDelivered) => (
            S::Demoting {
                new_fence,
                dest_delivered: true,
            },
            vec![],
        ),
        // DemoteAcked (proof-of-freeze) — the ONLY exit from Demoting: the source is now Ghost, so
        // push the ordered `Promote` to the dest (demote-BEFORE-promote). Carry the latched
        // `dest_delivered` forward so an early delivery still counts toward the release gate.
        (
            S::Demoting {
                new_fence,
                dest_delivered,
            },
            E::DemoteAcked,
        ) => (
            S::Promoting {
                new_fence,
                promote_acked: false,
                dest_delivered,
            },
            vec![A::Promote { new_fence }],
        ),
        // A Demoting timeout RE-EMITS the ordered Demote (idempotent at the source — `self_fence_skipped`),
        // never aborts: post-commit is forward-only. ✅ Slice 2a: the production `Timeout` PRODUCER
        // (`saga_runtime::scan_deadlines`, `now - since >= redrive_deadline_ticks`) now drives this in
        // production — a never-acked Demoting saga RE-DRIVES within one redrive window, never PARKS
        // (the R1 cure for lost saga-acks). A starved DELIVERY watermark is a different wedge (owed, P3).
        (
            S::Demoting {
                new_fence,
                dest_delivered,
            },
            E::Timeout,
        ) => (
            S::Demoting {
                new_fence,
                dest_delivered,
            },
            vec![A::Demote {
                new_owner_fence: new_fence,
            }],
        ),
        // ---- promoting: the seamless no-vanish release gate (PromoteAcked AND DestDelivered) ----
        (
            S::Promoting {
                new_fence,
                dest_delivered,
                ..
            },
            E::PromoteAcked,
        ) => promoting_advance(ctx, new_fence, true, dest_delivered),
        (
            S::Promoting {
                new_fence,
                promote_acked,
                ..
            },
            E::DestDelivered,
        ) => promoting_advance(ctx, new_fence, promote_acked, true),
        // A Promoting timeout RE-EMITS the ordered Promote (idempotent at the dest — promotes_redelivered),
        // never aborts. ✅ Slice 2a: `scan_deadlines` drives this in production (redrive_deadline_ticks).
        // ⚠️ a STARVED delivery watermark (the dest never latches `DeliveredToObservers`) is a DIFFERENT
        // wedge the producer cannot cure — re-driving Promote re-acks PromoteAck but cannot re-arm the
        // standing watermark (owed, P3 — DEFERRED D-36); the producer cures lost saga-ACKS only.
        (
            S::Promoting {
                new_fence,
                promote_acked,
                dest_delivered,
            },
            E::Timeout,
        ) => (
            S::Promoting {
                new_fence,
                promote_acked,
                dest_delivered,
            },
            vec![A::Promote { new_fence }],
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
        ) => step_aborting(ctx, reason, awaiting_thaw, awaiting_abort_ack, event),

        // ---- idempotent no-ops -------------------------------------------------------------
        // Terminal states absorb everything; unknown (state, event) pairs are
        // duplicates or stale deliveries under at-least-once — never errors.
        (terminal @ (S::Done { .. } | S::Aborted { .. }), _) => (terminal, vec![]),
        (state, _) => (state, vec![]),
    }
}

/// The pose-before-promote gate's decision (1d.1), hoisted out of the two `Freezing` arms so the
/// branch is covered ONCE: the CAS (commit) fires only when BOTH the gateway freeze
/// (`frozen_drained`) AND the source flush (`flushed`) have landed — otherwise the saga stays in
/// `Freezing`, recording the condition that just arrived. The watermark threaded into
/// `CommittingCas` is the GATEWAY's `drained_seq` (the input-conservation seq), never the source
/// flush watermark.
fn freezing_advance(
    ctx: &SagaCtx,
    marker_seq: u64,
    frozen_drained: Option<u64>,
    flushed: bool,
) -> (SagaState, Vec<SagaAction>) {
    match (frozen_drained, flushed) {
        (Some(drained_seq), true) => (
            SagaState::CommittingCas {
                marker_seq,
                drained_seq,
            },
            vec![commit_action(ctx)],
        ),
        _ => (
            SagaState::Freezing {
                marker_seq,
                frozen_drained,
                flushed,
            },
            vec![],
        ),
    }
}

/// The seamless-release gate's decision (1d.5b.1), hoisted out of the two `Promoting` arms so the
/// branch is covered ONCE (mirroring `freezing_advance`): `ReleaseSubscribe` fires only when BOTH
/// the dest acked the ordered promote (`promote_acked`) AND the gateway reported the dest delivered
/// to every observer (`dest_delivered`) — otherwise the saga stays in `Promoting`, recording the
/// condition that just arrived. Releasing the source sub only after the dest is delivered+rendered
/// is the seamless no-vanish property (the FORK-0a overlap held through the dest's first frame).
fn promoting_advance(
    ctx: &SagaCtx,
    new_fence: Fence,
    promote_acked: bool,
    dest_delivered: bool,
) -> (SagaState, Vec<SagaAction>) {
    match (promote_acked, dest_delivered) {
        (true, true) => (
            SagaState::Releasing { new_fence },
            vec![SagaAction::Send(TransferControl::ReleaseSubscribe {
                transfer: ctx.transfer,
                session: ctx.session,
                src: ctx.source,
            })],
        ),
        _ => (
            SagaState::Promoting {
                new_fence,
                promote_acked,
                dest_delivered,
            },
            vec![],
        ),
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
    ctx: &SagaCtx,
    reason: AbortReason,
    awaiting_thaw: bool,
    awaiting_abort_ack: bool,
    event: SagaEvent,
) -> (SagaState, Vec<SagaAction>) {
    // TIMEOUT (Slice 2a): the deadline producer RE-DRIVES the still-outstanding compensator command(s)
    // — a saga PARKED in Aborting (a dropped `ThawSource`/`AbortTransfer` ack) would otherwise never
    // reach terminal (the FSM's landing pad was a no-op before 2a; the "wrapper's retry" never
    // existed). State + awaiting flags UNCHANGED; `apply_thaw`/`apply_abort` re-ack idempotently. NO
    // `ClearTransferLock` here — that fires only on the terminal edge below. The two `if`s are SPLIT
    // (not `&&`) so each false arm stays a covered region (HR5).
    if let SagaEvent::Timeout = event {
        let mut actions = Vec::new();
        if awaiting_thaw {
            actions.push(SagaAction::Send(TransferControl::ThawSource {
                transfer: ctx.transfer,
                session: ctx.session,
            }));
        }
        if awaiting_abort_ack {
            actions.push(SagaAction::Send(TransferControl::AbortTransfer {
                transfer: ctx.transfer,
                session: ctx.session,
            }));
        }
        return (
            SagaState::Aborting {
                reason,
                awaiting_thaw,
                awaiting_abort_ack,
            },
            actions,
        );
    }
    let (awaiting_thaw, awaiting_abort_ack) = match event {
        SagaEvent::SourceThawed => (false, awaiting_abort_ack),
        SagaEvent::DestAborted => (awaiting_thaw, false),
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
        // TERMINAL: clear the directory lock (D-1) at the terminal edge — after BOTH compensator acks,
        // so the gateway's `apply_abort` has pruned the session journal (RACE-1 closed), then tombstone.
        (
            SagaState::Aborted { reason },
            vec![SagaAction::ClearTransferLock, SagaAction::Tombstone],
        )
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
            from_realm: RealmId::System(3),
            to_realm: RealmId::System(4),
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

    /// Drive a fresh saga to the commit-issue point and return (state, actions there). The
    /// pose-before-promote gate needs BOTH the freeze and the flush; the SECOND (here the flush)
    /// is where the commit action is emitted.
    fn drive_from_freeze(c: &SagaCtx) -> (SagaState, Vec<SagaAction>) {
        let (state, _) = drive(
            c,
            start(c).0,
            &[
                SagaEvent::Prepared(PrepareResult::Ready),
                SagaEvent::CutConfirmed { marker_seq: 9 },
                SagaEvent::SourceFrozen { drained_seq: 9 },
            ],
        );
        step(c, state, SagaEvent::SourceFlushed { drained_seq: 9 })
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
                SagaEvent::SourceFlushed { drained_seq: 17 },
                SagaEvent::CasWon {
                    new_fence: Fence(6),
                },
                SagaEvent::RouteSwapped,
                // The ordered demote-before-promote tail (1d.5b.1): source demote-ack, then the
                // dest promote-ack + the delivery watermark gate the source-sub release.
                SagaEvent::DemoteAcked,
                SagaEvent::PromoteAcked,
                SagaEvent::DestDelivered,
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

        // 1d.1: the source is told to flush its pose on entering Freezing, and the entity-state
        // crossing is emitted at commit, stamped with the new authority fence.
        assert!(
            actions.contains(&SagaAction::FlushSource),
            "the source is told to flush its pose: {actions:?}"
        );
        assert!(
            actions.contains(&SagaAction::EmitCrossing { fence: Fence(6) }),
            "the crossing is emitted at the new authority fence: {actions:?}"
        );

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
                    subject: c.subject,
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
    fn terminal_abort_clears_the_transfer_lock_via_both_paths() {
        // Slice 2a: the terminal Aborted edge emits ClearTransferLock (+ Tombstone) so the directory
        // lock clears (D-1) — via BOTH the pre-freeze path (no thaw) and the freeze path (thaw first).
        let c = ctx(false);
        let reject = PrepareReject::Spatial(SpatialReject::Obstructed);
        // (a) pre-freeze: Prepared-rejected → AbortTransfer; DestAborted → terminal.
        let (state, _) = step(
            &c,
            start(&c).0,
            SagaEvent::Prepared(PrepareResult::Rejected(reject)),
        );
        let (state, actions) = step(&c, state, SagaEvent::DestAborted);
        assert_eq!(
            state,
            SagaState::Aborted {
                reason: AbortReason::PrepareRejected(reject)
            }
        );
        assert!(
            actions.contains(&SagaAction::ClearTransferLock),
            "pre-freeze terminal abort clears the lock: {actions:?}"
        );
        assert!(actions.contains(&SagaAction::Tombstone));

        // (b) freeze path: freeze timeout → thaw + abort; SourceThawed then DestAborted → terminal.
        let (state, _) = drive(
            &c,
            start(&c).0,
            &[
                SagaEvent::Prepared(PrepareResult::Ready),
                SagaEvent::CutConfirmed { marker_seq: 5 },
                SagaEvent::Timeout, // freeze timeout → abort_with_thaw
                SagaEvent::SourceThawed,
            ],
        );
        let (state, actions) = step(&c, state, SagaEvent::DestAborted);
        assert_eq!(
            state,
            SagaState::Aborted {
                reason: AbortReason::FreezeTimeout
            }
        );
        assert!(
            actions.contains(&SagaAction::ClearTransferLock),
            "freeze-path terminal abort clears the lock too: {actions:?}"
        );
    }

    #[test]
    fn aborting_timeout_re_emits_only_the_outstanding_compensators() {
        // Slice 2a finding #6: a saga PARKED in Aborting (a dropped compensator ack) is re-driven by
        // the deadline producer's Timeout — re-emitting ONLY the still-outstanding command(s), state
        // and awaiting flags unchanged (apply_thaw/apply_abort re-ack idempotently).
        let c = ctx(false);
        let thaw = SagaAction::Send(TransferControl::ThawSource {
            transfer: c.transfer,
            session: c.session,
        });
        let abort = SagaAction::Send(TransferControl::AbortTransfer {
            transfer: c.transfer,
            session: c.session,
        });
        let r = AbortReason::FreezeTimeout;

        // BOTH outstanding: Timeout re-emits both (thaw before abort).
        let both = SagaState::Aborting {
            reason: r,
            awaiting_thaw: true,
            awaiting_abort_ack: true,
        };
        let (state, actions) = step(&c, both, SagaEvent::Timeout);
        assert_eq!(state, both);
        assert_eq!(actions, vec![thaw, abort]);

        // THAW-only (the abort ack already landed): re-emit ThawSource only.
        let thaw_only = SagaState::Aborting {
            reason: r,
            awaiting_thaw: true,
            awaiting_abort_ack: false,
        };
        let (state, actions) = step(&c, thaw_only, SagaEvent::Timeout);
        assert_eq!(state, thaw_only);
        assert_eq!(actions, vec![thaw]);

        // ABORT-only (the thaw ack already landed): re-emit AbortTransfer only.
        let abort_only = SagaState::Aborting {
            reason: r,
            awaiting_thaw: false,
            awaiting_abort_ack: true,
        };
        let (state, actions) = step(&c, abort_only, SagaEvent::Timeout);
        assert_eq!(state, abort_only);
        assert_eq!(actions, vec![abort]);
    }

    #[test]
    fn saga_tuning_validate_rejects_a_mistuned_budget() {
        // Slice 2a hardening (audit wf_75a8d57d): a 0 deadline (the abort-storm trap) or an abort
        // tighter than the re-drive is rejected LOUD at boot — never a silent abort storm.
        assert_eq!(SagaTuning::default().validate(), Ok(()));
        // ZeroDeadline — both operands of the bitwise `|`.
        assert_eq!(
            SagaTuning {
                redrive_deadline_ticks: 0,
                abort_deadline_ticks: 24,
            }
            .validate(),
            Err(SagaTuningError::ZeroDeadline)
        );
        assert_eq!(
            SagaTuning {
                redrive_deadline_ticks: 8,
                abort_deadline_ticks: 0,
            }
            .validate(),
            Err(SagaTuningError::ZeroDeadline)
        );
        // The destructive abort deadline below the cheap re-drive is rejected.
        assert_eq!(
            SagaTuning {
                redrive_deadline_ticks: 8,
                abort_deadline_ticks: 4,
            }
            .validate(),
            Err(SagaTuningError::AbortTighterThanRedrive {
                redrive: 8,
                abort: 4
            })
        );
    }

    #[test]
    fn a_stray_non_timeout_event_in_aborting_is_a_noop() {
        // The match's `_` arm: a stray event (neither SourceThawed/DestAborted nor Timeout) in
        // Aborting leaves state + flags unchanged and emits nothing (an at-least-once duplicate).
        let c = ctx(false);
        let aborting = SagaState::Aborting {
            reason: AbortReason::CutTimeout,
            awaiting_thaw: false,
            awaiting_abort_ack: true,
        };
        let (state, actions) = step(&c, aborting, SagaEvent::Prepared(PrepareResult::Ready));
        assert_eq!(state, aborting);
        assert!(actions.is_empty());
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
        assert_eq!(
            state,
            SagaState::Freezing {
                marker_seq: 5,
                frozen_drained: None,
                flushed: false,
            }
        );
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
    fn cut_confirmed_issues_freeze_and_flush() {
        // 1d.1: entering Freezing issues BOTH FreezeSource (to the gateway) and FlushSource (to
        // the source), and the state carries the two ungated sub-flags.
        let c = ctx(false);
        let (state, _) = drive(
            &c,
            start(&c).0,
            &[SagaEvent::Prepared(PrepareResult::Ready)],
        );
        let (next, acts) = step(&c, state, SagaEvent::CutConfirmed { marker_seq: 5 });
        assert_eq!(
            next,
            SagaState::Freezing {
                marker_seq: 5,
                frozen_drained: None,
                flushed: false,
            }
        );
        assert!(
            acts.contains(&SagaAction::Send(TransferControl::FreezeSource {
                transfer: c.transfer,
                session: c.session,
                marker_seq: 5,
                dest: c.dest,
            }))
        );
        assert!(acts.contains(&SagaAction::FlushSource));
    }

    #[test]
    fn pose_before_promote_gate_requires_both_freeze_and_flush() {
        // The CAS fires ONLY once BOTH the gateway freeze and the source flush land — in either
        // order — so a poseless crossing is unrepresentable. Both stay-arms and both advance-paths
        // are exercised here (deterministically, not by proptest draw).
        let c = ctx(false);
        let (in_freezing, _) = drive(
            &c,
            start(&c).0,
            &[
                SagaEvent::Prepared(PrepareResult::Ready),
                SagaEvent::CutConfirmed { marker_seq: 5 },
            ],
        );

        // ORDER A — freeze first stays; the flush completes the commit.
        let (after_frozen, acts) =
            step(&c, in_freezing, SagaEvent::SourceFrozen { drained_seq: 5 });
        assert_eq!(
            after_frozen,
            SagaState::Freezing {
                marker_seq: 5,
                frozen_drained: Some(5),
                flushed: false,
            }
        );
        assert!(acts.is_empty(), "freeze alone issues no commit");
        let (committed_a, acts) = step(
            &c,
            after_frozen,
            SagaEvent::SourceFlushed { drained_seq: 5 },
        );
        assert_eq!(
            committed_a,
            SagaState::CommittingCas {
                marker_seq: 5,
                drained_seq: 5,
            }
        );
        assert_eq!(
            acts,
            vec![SagaAction::IssueCommitCas {
                expected: c.expected_fence
            }]
        );

        // ORDER B — flush first stays; the freeze completes, and the GATEWAY watermark (7) is the
        // one threaded into CommittingCas, NOT the source flush watermark (99).
        let (after_flush, acts) = step(
            &c,
            in_freezing,
            SagaEvent::SourceFlushed { drained_seq: 99 },
        );
        assert_eq!(
            after_flush,
            SagaState::Freezing {
                marker_seq: 5,
                frozen_drained: None,
                flushed: true,
            }
        );
        assert!(acts.is_empty(), "flush alone issues no commit");
        let (committed_b, acts) = step(&c, after_flush, SagaEvent::SourceFrozen { drained_seq: 7 });
        assert_eq!(
            committed_b,
            SagaState::CommittingCas {
                marker_seq: 5,
                drained_seq: 7,
            },
            "the gateway drained_seq wins, not the source flush watermark"
        );
        assert_eq!(
            acts,
            vec![SagaAction::IssueCommitCas {
                expected: c.expected_fence
            }]
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
                SagaEvent::SourceFlushed { drained_seq: 9 }, // gate: both, to reach CommittingCas
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
            vec![
                SagaAction::Send(TransferControl::CommitAuthority {
                    transfer: c.transfer,
                    session: c.session,
                    new_fence: Fence(7),
                    subject: c.subject,
                }),
                // 1d.1: the crossing is re-emitted alongside the route-swap retry (at-least-once).
                SagaAction::EmitCrossing { fence: Fence(7) },
            ]
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

    #[test]
    fn ordered_demote_before_promote_gates_release_on_both_acks() {
        // 1d.5b.1: the binding ordered demote-before-promote tail. RouteSwapped pushes the Demote;
        // DemoteAcked (proof-of-freeze) ALONE advances to Promoting + pushes the Promote; the source
        // sub releases ONLY after BOTH PromoteAcked AND DestDelivered (here PromoteAcked first).
        let c = ctx(false);
        let (state, acts) = step(
            &c,
            SagaState::Swapping {
                new_fence: Fence(6),
            },
            SagaEvent::RouteSwapped,
        );
        assert_eq!(
            state,
            SagaState::Demoting {
                new_fence: Fence(6),
                dest_delivered: false,
            }
        );
        assert_eq!(
            acts,
            vec![SagaAction::Demote {
                new_owner_fence: Fence(6)
            }],
            "entering Demoting pushes the ordered Demote to the source"
        );

        let (state, acts) = step(&c, state, SagaEvent::DemoteAcked);
        assert_eq!(
            state,
            SagaState::Promoting {
                new_fence: Fence(6),
                promote_acked: false,
                dest_delivered: false,
            }
        );
        assert_eq!(
            acts,
            vec![SagaAction::Promote {
                new_fence: Fence(6)
            }],
            "the demote-ack ALONE advances to Promoting + pushes the Promote (demote-before-promote)"
        );

        // PromoteAcked alone does NOT release — the delivery half is still pending.
        let (state, acts) = step(&c, state, SagaEvent::PromoteAcked);
        assert_eq!(
            state,
            SagaState::Promoting {
                new_fence: Fence(6),
                promote_acked: true,
                dest_delivered: false,
            }
        );
        assert!(acts.is_empty(), "promote-ack alone holds the source sub");

        // DestDelivered completes the gate → ReleaseSubscribe.
        let (state, acts) = step(&c, state, SagaEvent::DestDelivered);
        assert_eq!(
            state,
            SagaState::Releasing {
                new_fence: Fence(6)
            }
        );
        assert_eq!(
            acts,
            vec![SagaAction::Send(TransferControl::ReleaseSubscribe {
                transfer: c.transfer,
                session: c.session,
                src: c.source,
            })]
        );
    }

    #[test]
    fn promoting_gate_completes_in_either_arrival_order() {
        // The reverse order (DestDelivered before PromoteAcked) — covers the OTHER promoting_advance
        // arm, so the two-condition gate is exercised both ways (mirrors the freeze/flush gate).
        let c = ctx(false);
        let promoting = SagaState::Promoting {
            new_fence: Fence(6),
            promote_acked: false,
            dest_delivered: false,
        };
        let (state, acts) = step(&c, promoting, SagaEvent::DestDelivered);
        assert_eq!(
            state,
            SagaState::Promoting {
                new_fence: Fence(6),
                promote_acked: false,
                dest_delivered: true,
            }
        );
        assert!(acts.is_empty(), "delivery alone holds the source sub");
        let (state, acts) = step(&c, state, SagaEvent::PromoteAcked);
        assert_eq!(
            state,
            SagaState::Releasing {
                new_fence: Fence(6)
            }
        );
        assert_eq!(acts.len(), 1, "the second condition fires ReleaseSubscribe");
    }

    #[test]
    fn an_early_delivery_in_demoting_is_latched_and_carried_into_promoting() {
        // DEFENSIVE coverage of the latch carry-forward. ⚠️ COUNTERFACTUAL IN PRODUCTION since
        // 1d.5b.3b: the dest sub is announced only at PROMOTE (strict ordering), so the dest cannot
        // deliver while still in Demoting — this injects `DestDelivered` directly to keep the
        // (Demoting, DestDelivered) latch arm covered. The latch + carry-forward stay correct so the
        // gate is robust if a future reorder ever surfaces an early delivery. Demoting latches it,
        // and DemoteAcked carries it forward so
        // a single later PromoteAcked completes the gate — no lost-delivery park.
        let c = ctx(false);
        let demoting = SagaState::Demoting {
            new_fence: Fence(6),
            dest_delivered: false,
        };
        let (state, acts) = step(&c, demoting, SagaEvent::DestDelivered);
        assert_eq!(
            state,
            SagaState::Demoting {
                new_fence: Fence(6),
                dest_delivered: true,
            },
            "early delivery is latched in Demoting"
        );
        assert!(acts.is_empty());
        let (state, acts) = step(&c, state, SagaEvent::DemoteAcked);
        assert_eq!(
            state,
            SagaState::Promoting {
                new_fence: Fence(6),
                promote_acked: false,
                dest_delivered: true,
            },
            "the latched delivery is carried into Promoting"
        );
        assert_eq!(
            acts,
            vec![SagaAction::Promote {
                new_fence: Fence(6)
            }]
        );
        // PromoteAcked ALONE now completes the gate (delivery already counted) → ReleaseSubscribe.
        let (state, _) = step(&c, state, SagaEvent::PromoteAcked);
        assert_eq!(
            state,
            SagaState::Releasing {
                new_fence: Fence(6)
            }
        );
    }

    #[test]
    fn ordered_tail_timeouts_re_emit_forward_only() {
        // A Demoting timeout re-emits the Demote; a Promoting timeout re-emits the Promote — both
        // idempotent re-drives, never aborts (post-commit is forward-only). ✅ Slice 2a: the
        // `saga_runtime::scan_deadlines` producer drives these in production (the FSM arms here are
        // the landing pads; the producer-driven path is covered in `saga_runtime`'s own tests).
        let c = ctx(false);
        let demoting = SagaState::Demoting {
            new_fence: Fence(6),
            dest_delivered: true,
        };
        let (state, acts) = step(&c, demoting, SagaEvent::Timeout);
        assert_eq!(
            state, demoting,
            "the latched delivery survives the re-drive"
        );
        assert_eq!(
            acts,
            vec![SagaAction::Demote {
                new_owner_fence: Fence(6)
            }]
        );
        let promoting = SagaState::Promoting {
            new_fence: Fence(6),
            promote_acked: true,
            dest_delivered: false,
        };
        let (state, acts) = step(&c, promoting, SagaEvent::Timeout);
        assert_eq!(state, promoting, "the acked flags survive the re-drive");
        assert_eq!(
            acts,
            vec![SagaAction::Promote {
                new_fence: Fence(6)
            }]
        );
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
            (0u64..100).prop_map(|s| SagaEvent::SourceFlushed { drained_seq: s }),
            (0u64..10).prop_map(|f| SagaEvent::CasWon {
                new_fence: Fence(f)
            }),
            (0u64..10).prop_map(|f| SagaEvent::CasLost { current: Fence(f) }),
            Just(SagaEvent::RouteSwapped),
            Just(SagaEvent::DemoteAcked),
            Just(SagaEvent::PromoteAcked),
            Just(SagaEvent::DestDelivered),
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
                    // The pose-before-promote gate needs BOTH conditions: feed the flush first
                    // (stays Freezing), then the freeze advances to the commit.
                    SagaState::Freezing { flushed: false, .. } => {
                        SagaEvent::SourceFlushed { drained_seq: 1 }
                    }
                    SagaState::Freezing { .. } => SagaEvent::SourceFrozen { drained_seq: 1 },
                    SagaState::CommittingCas { .. } => SagaEvent::CasWon { new_fence: Fence(9) },
                    SagaState::Swapping { .. } => SagaEvent::RouteSwapped,
                    // The ordered tail: DemoteAcked leaves Demoting → Promoting; then the two-flag
                    // gate (PromoteAcked then DestDelivered) reaches Releasing (mirrors Freezing).
                    SagaState::Demoting { .. } => SagaEvent::DemoteAcked,
                    SagaState::Promoting {
                        promote_acked: false,
                        ..
                    } => SagaEvent::PromoteAcked,
                    SagaState::Promoting { .. } => SagaEvent::DestDelivered,
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
