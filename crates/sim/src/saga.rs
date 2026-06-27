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
use vd_wire::intershard::TRANSIENT_BATCH_STEP;
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

/// D-3 dead-vs-slow discriminator tuning (ONE reviewed home, beside [`SagaTuning`]). A peer is
/// confirmed dead only after `n_consecutive_unreachable` `NodeUnreachable` notices within
/// `unreachable_window_ticks` with NO intervening successful inbound — so a single recoverable blip
/// toward a HEALTHY peer never confirms it dead (the CSCALE-1 cure). Defaults reproduce today's
/// kill-only behavior (`n_consecutive_unreachable == 1`) so the inert path is byte-identical pre-D-3.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LivenessTuning {
    /// Consecutive `NodeUnreachable` notices (within the window, no intervening ack) required before a
    /// peer's dead bit flips. `1` = today's kill-only stand-in. Prod ≥ 3 (survives a transient blip).
    pub n_consecutive_unreachable: u32,
    /// The window (universe ticks) the consecutive notices must fall within; an older first-notice
    /// resets the run. MUST span `n_consecutive_unreachable` redelivery spacings (else a real dead
    /// peer's notices arrive too far apart to ever confirm) — enforced by `validate`.
    pub unreachable_window_ticks: u64,
    /// The expected redelivery spacing (universe ticks) of `NodeUnreachable` toward a down peer — the
    /// transport's redial/retry cadence (the harness `FaultFabric` retry delay; io-prod's QUIC idle
    /// cadence). Only used by `validate` to cross-check the window covers the confirmation run.
    pub retry_delay_ticks_hint: u64,
}

/// DEV/test liveness default — kill-only-equivalent (`n_consecutive_unreachable == 1`), so a default
/// cluster confirms a permanent kill exactly as the pre-D-3 stand-in did and every existing crash cell
/// stays byte-identical. PRODUCTION sets `n_consecutive_unreachable` ≥ 3 from env (the CSCALE-1 margin).
impl Default for LivenessTuning {
    fn default() -> LivenessTuning {
        LivenessTuning {
            n_consecutive_unreachable: 1,
            unreachable_window_ticks: 64,
            retry_delay_ticks_hint: 2,
        }
    }
}

/// A mis-tuned [`LivenessTuning`] — rejected LOUD at boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LivenessTuningError {
    #[error("n_consecutive_unreachable must be >= 1 (0 would confirm a peer dead with no evidence)")]
    ZeroConsecutive,
    #[error(
        "unreachable_window_ticks ({window}) must be >= n_consecutive_unreachable ({n}) * \
         retry_delay_ticks_hint ({retry}) = {span}: the window must span the whole confirmation run, \
         else a genuinely-dead peer's spaced-out notices reset before they ever confirm"
    )]
    WindowTooTight {
        window: u64,
        n: u32,
        retry: u64,
        span: u64,
    },
}

impl LivenessTuning {
    /// Reject a mis-tuned dead-vs-slow budget at boot. `n_consecutive_unreachable` must be ≥ 1, and
    /// the window must be wide enough to hold `n` notices at the redelivery spacing (else a real dead
    /// peer is never confirmed — a liveness hole, the dual of CSCALE-1's false-confirm).
    ///
    /// # Errors
    /// [`LivenessTuningError`] for a zero confirmation count or a window too tight for the run.
    pub fn validate(&self) -> Result<(), LivenessTuningError> {
        if self.n_consecutive_unreachable < 1 {
            return Err(LivenessTuningError::ZeroConsecutive);
        }
        let span = self
            .retry_delay_ticks_hint
            .saturating_mul(self.n_consecutive_unreachable as u64);
        if self.unreachable_window_ticks < span {
            return Err(LivenessTuningError::WindowTooTight {
                window: self.unreachable_window_ticks,
                n: self.n_consecutive_unreachable,
                retry: self.retry_delay_ticks_hint,
                span,
            });
        }
        Ok(())
    }
}

/// The sub-phase of the D-7d transient post-commit tail ([`SagaState::BatchHandoff`]): which
/// choreography ack the saga is awaiting. Each phase has ONE pending egress the producer re-emits on a
/// `Timeout` (idempotent — the shard journals by `(transfer, step)`), and any phase resolves on a
/// dead-participant notice (the D-7d kill cells). The transient analogue of the durable
/// `Demoting`→`Promoting`→`Releasing` walk, but driven by the shard↔shard adopt-before-drop acks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchHandoffPhase {
    /// Awaiting the dest's `BatchAdopted` (it now holds the batch as the uncounted `Arriving` tier).
    AwaitAdopt,
    /// Awaiting the source's `DropApplied`(release): it flipped `Held→Departing` (now uncounted).
    AwaitRelease,
    /// Awaiting the dest's `DropApplied`(promote-confirm): `Arriving→Held` (the dest now counts it).
    AwaitPromote,
    /// Awaiting the source's `SourceRetired` (the new D-7d ack): it dropped the retained `Departing`
    /// copy — the choreography is complete and the saga may tombstone.
    AwaitComplete,
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
    /// TRANSIENT (D-7) SHORT-PATH commit: the batched `TransientGo` go-token is in flight (HR2 — the
    /// SAME commit point + `CasWon` feedback as the durable CAS, fanned out by `commit_action`).
    /// Entered DIRECTLY by `start` for a Transient subject (skipping Prepare/Cut/Freeze — a
    /// session-less batch has no client cut, no per-subject pose flush). `CasWon` ends it at `Done`
    /// with NO Swapping/Demoting/Promoting: the source→dest set hand-off is the shard↔shard
    /// adopt-before-drop choreography (the runtime's `BatchAdopted`→`TransientDrop`), not an FSM tail.
    /// `step_id` is the batch's `(transfer, step)` idempotency phase ([`TRANSIENT_BATCH_STEP`]).
    BatchCommitting { step_id: u32 },
    /// TRANSIENT (D-7d) POST-COMMIT TAIL: the go-token committed; the saga is NOT yet tombstoned but
    /// owns the adopt-before-drop choreography to its terminal — the transient twin of the durable
    /// `Demoting`/`Promoting` tail (HR2: ONE recovery machinery, the SAME `scan_deadlines` producer
    /// re-drives both). Replaces the D-7a/b/c tombstone-at-commit + ledger-driven handlers: keeping the
    /// saga alive is exactly what lets the producer see a stranded handoff and inject the dead-resolution
    /// (the D-7d kill cells). `new_fence` is the committed go-token fence (the SOLE promote authority,
    /// carried VERBATIM into every choreography egress — re-anchors the dest even with the source dead).
    BatchHandoff {
        phase: BatchHandoffPhase,
        new_fence: Fence,
    },
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
        /// D-37 Slice 2d: `Some(target)` iff this `Promoting` was reached via a forward RE-HOME
        /// (`ReHoming → CasWon`), carrying the LIVE re-home target so a `Promoting` `Timeout`
        /// re-drives the dedicated `A::ReHomeAdopt → target` (the SELF-SUFFICIENT orchestrator
        /// re-drive egress) instead of `A::Promote → ctx.dest` — which is the confirmed-dead
        /// original dest a re-home fired BECAUSE of. `None` for a normal demote-promote `Promoting`
        /// (the Timeout re-emits the ordered Promote). Without this field the re-home ADOPT was a
        /// producer-less phase leaning on transport at-least-once (DEFERRED D-6 precondition 1 / D-37).
        rehome_target: Option<NodeId>,
    },
    /// D-37 forward re-home: the committed owner was permanently KILLED, so the saga is re-targeting the
    /// subject onto a LIVE capability-matched shard `target`. `ReHomeCommit` is in flight (the directory
    /// CAS at `prev_fence` → `prev_fence+1` naming `target`); its `CasWon` advances to `Promoting{target}`
    /// plus the `ReHome` adopt, while `CasLost` (someone else won the key — rule 3) terminates as a clean
    /// no-op. `prev_fence` is the fence the dead owner was committed at (the CAS expectation). Reached from
    /// `Promoting` via the producer-injected `ReHomeTo` once the dead dest is confirmed and the abort budget
    /// elapsed.
    ReHoming { target: NodeId, prev_fence: Fence },
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
    /// D-7d transient post-commit tail (`BatchHandoff`) — the adopt-before-drop acks routed through
    /// `deliver` to advance the phase (replacing the D-7a/b/c ledger-driven read-only handlers). The
    /// dest adopted the batch (uncounted `Arriving`) → drives `AwaitAdopt→AwaitRelease`.
    BatchAdopted,
    /// D-7d — the SOURCE acked its release `DropApplied`(`TRANSIENT_RELEASE_STEP`): `Held→Departing`
    /// (now uncounted) → drives `AwaitRelease→AwaitPromote` (the dest promote is reachable only now —
    /// the source is uncounted BEFORE the dest counts, so the holder set is never `{source, dest}`).
    SourceDropApplied,
    /// D-7d — the DEST acked its promote-confirm `DropApplied`(`TRANSIENT_DROP_STEP`): `Arriving→Held`
    /// (the dest now counts it) → drives `AwaitPromote→AwaitComplete`.
    DestDropApplied,
    /// D-7d — the SOURCE acked it RETIRED the retained `Departing` copy (`TRANSIENT_COMPLETE_STEP`, the
    /// NEW ack `on_release_complete` now emits): the choreography is complete → `AwaitComplete→Done`.
    SourceRetired,
    /// D-7d DEAD-RESOLUTION — `scan_deadlines` learned (via a kill-only `NodeUnreachable`) that the
    /// SOURCE is unreachable while the saga is in `BatchHandoff` AND the redrive deadline elapsed: the
    /// dest already adopted, so self-promote it from the go-token (zero loss). Terminal on first fire.
    SourceUnreachable,
    /// D-7d — the DEST is unreachable mid-`BatchHandoff`: the only promote target is gone, so ABANDON
    /// the source's retained copy as an accounted loss-within-budget. Terminal on first fire.
    DestUnreachable,
    /// The phase's adaptive deadline (k·RTT, hard-capped) elapsed.
    Timeout,
    /// Pre-commit cancellation request.
    Cancel,
    /// D-37 forward re-home: `scan_deadlines` confirmed the committed dest dead (in `Promoting`), the
    /// abort budget elapsed, AND `select_rehome_target` found a LIVE capability-matched `target`. Drives
    /// `Promoting → ReHoming{target}`. The producer carries the chosen `target` (the pure FSM cannot
    /// select — selection needs the roster + liveness); a `None` selection injects nothing (the saga
    /// stays parked, honest, never a forced re-home).
    ReHomeTo { target: NodeId },
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
    /// D-37 forward re-home commit (the SAME single commit point, re-targeted): the wrapper calls
    /// `commit_cas(subject, expected, Shard(target))` — bumping the fence to `expected+1` (the
    /// fence-monotone invariant: a resurrected dead owner holds `< expected+1` and self-fences), clearing
    /// any lock, and naming `target` BEFORE the adopt (so the target is a legitimate owner when the
    /// `ReHome` lands). Feeds `CasWon`/`CasLost` back exactly like `IssueCommitCas` (HR3 one commit
    /// machinery), but to `target` (a live roster shard) rather than the hardcoded `ctx.dest`.
    ReHomeCommit {
        expected: Fence,
        target: NodeId,
    },
    /// D-37 forward re-home adopt: emit the dedicated `InterShardFlow::ReHome` to `target` carrying the
    /// stashed flushed pose (`ReHomeState::PoseOnly`), so the target reconstructs the subject as Owned at
    /// `new_fence` and acks `PromoteAck`. The Some/None build (Entity subject + a stashed pose) lives in
    /// `emit_rehome` (a monomorphic helper, unit-tested both ways) so this arm stays a branchless dispatch.
    ReHomeAdopt {
        new_fence: Fence,
        target: NodeId,
    },
    /// Issue the BATCHED `TransientGo` go-token (commit point for a TRANSIENT
    /// subject) — the same Fence-CAS amortized across a batch (HR2). Present and
    /// classified now; the batched transient flow is driven at P3.
    IssueTransientGo {
        expected: Fence,
    },
    /// D-7d transient tail egress — tell the SOURCE to RELEASE the batch (`Held→Departing`). The wrapper
    /// resolves the target (`SagaCtx.source`) + step (`TRANSIENT_RELEASE_STEP`); `fence` is the committed
    /// go-token fence carried VERBATIM. Emitted on `AwaitAdopt+BatchAdopted`; re-emitted on a `Timeout`
    /// in `AwaitRelease` (idempotent — the source journals by `(transfer, RELEASE_STEP)`).
    EmitTransientRelease {
        fence: Fence,
    },
    /// D-7d — tell the DEST to PROMOTE the batch (`Arriving→Held`, re-anchored to the go-token `fence`).
    /// Target `SagaCtx.dest` + step `TRANSIENT_DROP_STEP`. Emitted on `AwaitRelease+SourceDropApplied`
    /// (the source is uncounted first) OR DIRECTLY on a dead-source resolution (the go-token is the SOLE
    /// authority — legitimate with the source dead). Re-emitted on a `Timeout` in `AwaitPromote`.
    EmitTransientPromote {
        fence: Fence,
    },
    /// D-7d — tell the SOURCE the handoff is complete (`ReleaseComplete`): retire the retained
    /// `Departing` copy + ack `SourceRetired`. Target `SagaCtx.source` + step `TRANSIENT_RELEASE_STEP`.
    /// Emitted on `AwaitPromote+DestDropApplied`; re-emitted on a `Timeout` in `AwaitComplete`.
    EmitReleaseComplete {
        fence: Fence,
    },
    /// D-7d dead-DEST resolution egress — tell the SOURCE to ABANDON the batch (drop the retained
    /// `Departing` copy as an accounted loss-within-budget). Target `SagaCtx.source` + step
    /// `TRANSIENT_ABANDON_STEP`. A PROPER new action (NOT a repurposed `ReleaseComplete`, whose
    /// semantics are "clean retire, no loss"): abandon = "this batch is dead, count the drop".
    EmitTransientAbandon {
        fence: Fence,
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

/// The initial state + actions for a freshly created saga. HR2 fan-out at the FIRST of the two
/// structural class branches (the other is `commit_action`): a `Transient` batch takes the SHORT
/// PATH straight to the commit point; a `Durable` subject takes the full per-entity walk.
#[must_use]
pub fn start(ctx: &SagaCtx) -> (SagaState, Vec<SagaAction>) {
    match ctx.class {
        // TRANSIENT (D-7): the SHORT FSM PATH. A session-less debris batch CANNOT traverse the
        // durable session walk (no client `CUT_MARKER`, no per-subject pose flush, no ordered
        // demote/promote — blocker B), so it commits straight at the batched go-token and is `Done`.
        // `needs_provision` is durable-warp only; a transient batch crosses between live realms.
        DurabilityClass::Transient => (
            SagaState::BatchCommitting {
                step_id: TRANSIENT_BATCH_STEP,
            },
            vec![commit_action(ctx)],
        ),
        // DURABLE: the full per-entity walk (warp class awaits destination provisioning first).
        DurabilityClass::Durable => {
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
    }
}

/// THE transition function. Total over (state, event); unknown combinations are
/// idempotent no-ops (at-least-once delivery makes duplicates routine, not errors).
#[must_use]
pub fn step(ctx: &SagaCtx, state: SagaState, event: SagaEvent) -> (SagaState, Vec<SagaAction>) {
    use BatchHandoffPhase as P;
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

        // ---- transient batch commit (D-7): the SHORT FSM PATH commit point ------------------
        // The batched go-token committed (HR2 — the SAME `CasWon` feedback as the durable CAS): the
        // saga enters the POST-COMMIT TAIL (D-7d), NOT a tombstone. The go-token's commit fence rides
        // `new_fence` (the dest realm-lease fence) and is the SOLE promote authority hereafter. Keeping
        // the saga alive through the adopt-before-drop choreography is what lets `scan_deadlines` see a
        // stranded handoff and resolve it (the D-7d kill cells) — the transient twin of the durable
        // post-commit `Swapping→Demoting→Promoting` tail, ONE recovery machinery (HR2).
        (S::BatchCommitting { .. }, E::CasWon { new_fence }) => (
            S::BatchHandoff {
                phase: P::AwaitAdopt,
                new_fence,
            },
            // NO egress: the dest adopts AUTONOMOUSLY off the source's `TransientBatch` envelope (the
            // fabric redelivers it at-least-once); the saga just awaits the dest's `BatchAdopted`.
            vec![],
        ),
        // A go-token in flight re-drives idempotently on Timeout (the ledger write is keyed by
        // `(transfer, step)` — a re-record is a no-op), NEVER an abort — the transient twin of the
        // CommittingCas re-issue (forward-only at the commit point). ✅ Slice-2a producer drives it.
        (S::BatchCommitting { .. }, E::Timeout) => (state, vec![commit_action(ctx)]),

        // ---- D-7d transient post-commit tail (BatchHandoff): the adopt-before-drop choreography ------
        // PHASE 1→2: the dest ADOPTED (uncounted `Arriving`) → tell the SOURCE to RELEASE (`Held→
        // Departing`). The dest promote does NOT fire yet (gated on the source's `DropApplied`), so the
        // source is uncounted BEFORE the dest counts — the holder set is never `{source, dest}`.
        (
            S::BatchHandoff {
                phase: P::AwaitAdopt,
                new_fence,
            },
            E::BatchAdopted,
        ) => (
            S::BatchHandoff {
                phase: P::AwaitRelease,
                new_fence,
            },
            vec![A::EmitTransientRelease { fence: new_fence }],
        ),
        // PHASE 2→3: the source released (now uncounted) → PROMOTE the dest (`Arriving→Held`).
        (
            S::BatchHandoff {
                phase: P::AwaitRelease,
                new_fence,
            },
            E::SourceDropApplied,
        ) => (
            S::BatchHandoff {
                phase: P::AwaitPromote,
                new_fence,
            },
            vec![A::EmitTransientPromote { fence: new_fence }],
        ),
        // PHASE 3→4: the dest promote-confirmed (it now counts the batch) → tell the SOURCE to RETIRE
        // its retained `Departing` copy (`ReleaseComplete`).
        (
            S::BatchHandoff {
                phase: P::AwaitPromote,
                new_fence,
            },
            E::DestDropApplied,
        ) => (
            S::BatchHandoff {
                phase: P::AwaitComplete,
                new_fence,
            },
            vec![A::EmitReleaseComplete { fence: new_fence }],
        ),
        // PHASE 4→DONE: the source retired the last copy → the choreography is complete, tombstone.
        (
            S::BatchHandoff {
                phase: P::AwaitComplete,
                new_fence,
            },
            E::SourceRetired,
        ) => (S::Done { new_fence }, vec![A::Tombstone]),
        // Timeout re-drive: re-emit the CURRENT phase's pending egress idempotently (the shard journals
        // by `(transfer, step)`, so a re-emit is a no-op). `AwaitAdopt` has NO orchestrator egress to
        // re-drive (the dest adopts off the redelivered envelope), so it falls through to the no-op
        // catch-all — the saga simply keeps awaiting `BatchAdopted`.
        (
            S::BatchHandoff {
                phase: P::AwaitRelease,
                new_fence,
            },
            E::Timeout,
        ) => (state, vec![A::EmitTransientRelease { fence: new_fence }]),
        (
            S::BatchHandoff {
                phase: P::AwaitPromote,
                new_fence,
            },
            E::Timeout,
        ) => (state, vec![A::EmitTransientPromote { fence: new_fence }]),
        (
            S::BatchHandoff {
                phase: P::AwaitComplete,
                new_fence,
            },
            E::Timeout,
        ) => (state, vec![A::EmitReleaseComplete { fence: new_fence }]),
        // D-7d DEAD-RESOLUTION (terminal on FIRST fire — straight to Done, NEVER back to BatchHandoff,
        // structurally avoiding the D-37 forever-bounce). SOURCE dead: every BatchHandoff phase is
        // post-adopt, so the dest holds the batch and the go-token is the SOLE promote authority →
        // self-promote the dest (zero loss). DEST dead: the only promote target is gone → ABANDON the
        // source's retained copy as an accounted loss-within-budget. The go-token is NOT GC'd here (it
        // backs the dest's Held authority + the TRANSIENT-AUTHORITY-HELD oracle; bounded GC is owed
        // D-7d Slice 2, which co-designs the drop-completion signal without breaking the quiescence count).
        (S::BatchHandoff { new_fence, .. }, E::SourceUnreachable) => (
            S::Done { new_fence },
            vec![A::EmitTransientPromote { fence: new_fence }, A::Tombstone],
        ),
        (S::BatchHandoff { new_fence, .. }, E::DestUnreachable) => (
            S::Done { new_fence },
            vec![A::EmitTransientAbandon { fence: new_fence }, A::Tombstone],
        ),

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
                rehome_target: None,
            },
            vec![A::Promote { new_fence }],
        ),
        // D-37 CELL 1 (permanent kill of the SOURCE post-commit, in Demoting): the ordered Demote will
        // NEVER `DemoteAcked` because the source is a confirmed-dead corpse — and its held claim is moot
        // (the dead-aware oracle `verify_authority_unique_excluding` excludes it; the directory already
        // committed authority to the dest at `new_fence`). So SELF-PROMOTE the already-committed live dest:
        // the EXACT structural analogue of the transient `(BatchHandoff, SourceUnreachable)` self-promote
        // above, and of the `DemoteAcked` exit — carrying the latched `dest_delivered` forward so the
        // `Promoting` release gate never loses an early delivery. The wasted `ReleaseSubscribe` to the dead
        // source on the way to Done is tolerated by the forward-only `Releasing` timeout. AUTHORITY-UNIQUE
        // holds (the corpse holds nothing; one live owner at `new_fence`). The PRODUCER (scan_deadlines)
        // only injects this once the source is `is_confirmed_dead` (D-3 evidence-gated) — a live-but-slow
        // source merely re-drives via the Demoting `Timeout` arm below.
        (
            S::Demoting {
                new_fence,
                dest_delivered,
            },
            E::SourceUnreachable,
        ) => (
            S::Promoting {
                new_fence,
                promote_acked: false,
                dest_delivered,
                rehome_target: None,
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
                rehome_target,
                ..
            },
            E::PromoteAcked,
        ) => promoting_advance(ctx, new_fence, true, dest_delivered, rehome_target),
        (
            S::Promoting {
                new_fence,
                promote_acked,
                rehome_target,
                ..
            },
            E::DestDelivered,
        ) => promoting_advance(ctx, new_fence, promote_acked, true, rehome_target),
        // A Promoting timeout RE-EMITS the ordered Promote (idempotent at the dest — promotes_redelivered),
        // never aborts. ✅ Slice 2a: `scan_deadlines` drives this in production (redrive_deadline_ticks).
        // ⚠️ a STARVED delivery watermark (the dest never latches `DeliveredToObservers`) is a DIFFERENT
        // wedge the producer cannot cure — re-driving Promote re-acks PromoteAck but cannot re-arm the
        // standing watermark (owed, P3 — DEFERRED D-36); the producer cures lost saga-ACKS only.
        // ✅ D-37 Slice 2d: a RE-HOMED Promoting (`rehome_target = Some`) re-drives the DEDICATED ReHome
        // ADOPT to the LIVE `target`, NOT `A::Promote` to the confirmed-dead `ctx.dest` (which a re-home
        // fired BECAUSE of) — the SELF-SUFFICIENT orchestrator re-drive egress that stops the re-home adopt
        // depending on transport at-least-once (closes DEFERRED D-6 precondition 1's re-home half / D-37).
        // Re-emission is idempotent at the target (`on_re_home` journal-gates `RE_HOME_STEP` →
        // re_home_redelivered, re-acks PromoteAck). A normal Promoting (`None`) re-emits the ordered Promote.
        (
            S::Promoting {
                new_fence,
                promote_acked,
                dest_delivered,
                rehome_target,
            },
            E::Timeout,
        ) => (
            S::Promoting {
                new_fence,
                promote_acked,
                dest_delivered,
                rehome_target,
            },
            match rehome_target {
                Some(target) => vec![A::ReHomeAdopt { new_fence, target }],
                None => vec![A::Promote { new_fence }],
            },
        ),
        // ---- D-37 forward re-home: the committed owner was permanently KILLED ------------------------
        // The producer (scan_deadlines) confirmed the committed dest dead in Promoting, the abort budget
        // elapsed, AND selected a live capability-matched `target` → re-target the subject onto it. The
        // re-home commit is the SAME single commit point re-pointed (HR3): bump the fence to
        // `new_fence+1` (the fence-monotone invariant — a resurrected dead dest is then strictly stale and
        // self-fences) and name `target` BEFORE the adopt.
        (S::Promoting { new_fence, .. }, E::ReHomeTo { target }) => (
            S::ReHoming {
                target,
                prev_fence: new_fence,
            },
            vec![A::ReHomeCommit {
                expected: new_fence,
                target,
            }],
        ),
        // The re-home CAS won: the directory now names `target` at the bumped fence. Adopt the subject
        // there (the dedicated `ReHome` arm, NOT `Promote` — the target reconstructs state, no ghost to
        // flip) and enter `Promoting{target}`. The release gate (PromoteAcked AND DestDelivered) then
        // drives `Releasing → Done` — NEVER Done-on-promote: a starved `DeliveredToObservers` watermark
        // (no gateway sub to the fresh target until the client re-subscribes) parks here, the D-36 wedge
        // the re-home cannot cure (it cures the AUTHORITY orphan; the client re-route is owed, D-37/D-36).
        (S::ReHoming { target, .. }, E::CasWon { new_fence }) => (
            S::Promoting {
                new_fence,
                promote_acked: false,
                dest_delivered: false,
                // D-37 Slice 2d: carry the LIVE target so a Promoting Timeout re-drives the ReHome
                // adopt HERE, not Promote to the dead dest — the re-home's self-sufficient re-drive.
                rehome_target: Some(target),
            },
            vec![A::ReHomeAdopt { new_fence, target }],
        ),
        // The re-home CAS LOST (rule 3): some other writer moved the fence past `prev_fence` — this saga
        // is the loser and no-ops. The entity is alive at the CAS winner; no compensation (the re-home
        // froze nothing new). Defensive in the single-orchestrator in-process model (the Entity key is
        // unlocked + the reaper leaves it for re-home, so nothing else races it); reachable under
        // N-orchestrator partition (D-32). Terminal.
        (S::ReHoming { .. }, E::CasLost { .. }) => (
            S::Aborted {
                reason: AbortReason::CasLost,
            },
            vec![A::Tombstone],
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
    rehome_target: Option<NodeId>,
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
        // Stay in Promoting recording the condition that just arrived — preserving `rehome_target`
        // so a re-homed saga's later Timeout still re-drives the adopt to the live target (D-37 2d).
        _ => (
            SagaState::Promoting {
                new_fence,
                promote_acked,
                dest_delivered,
                rehome_target,
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
    fn transient_takes_the_short_path_to_the_batched_go_token() {
        // HR2 fan-out at `start` (D-7, the FIRST class branch): a Transient subject SKIPS
        // Prepare/Cut/Freeze and enters BatchCommitting DIRECTLY, issuing the batched TransientGo
        // go-token (NOT a per-entity CAS, NOT the session walk). CasWon enters the post-commit
        // `BatchHandoff` tail (D-7d) — the saga OWNS the adopt-before-drop choreography to its terminal
        // (the transient twin of the durable demote/promote tail), tombstoning only at `SourceRetired`.
        let c = ctx_class(false, DurabilityClass::Transient);
        let (state, actions) = start(&c);
        assert_eq!(
            state,
            SagaState::BatchCommitting {
                step_id: TRANSIENT_BATCH_STEP
            },
            "a Transient subject enters the short path directly from start"
        );
        assert_eq!(
            actions,
            vec![SagaAction::IssueTransientGo { expected: Fence(5) }],
            "the short path issues the batched go-token immediately (no PrepareSubscribe)"
        );

        // CasWon (the go-token committed) → the POST-COMMIT TAIL (D-7d), NOT a tombstone: the saga now
        // owns the adopt-before-drop choreography (the transient twin of the durable demote/promote tail).
        let new_fence = Fence(5).next();
        let (tail, actions) = step(&c, state, SagaEvent::CasWon { new_fence });
        assert_eq!(
            tail,
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitAdopt,
                new_fence
            },
            "CasWon enters the post-commit handoff tail awaiting the dest's BatchAdopted"
        );
        assert_eq!(
            actions,
            vec![],
            "no egress on entering AwaitAdopt — the dest adopts autonomously off the redelivered envelope"
        );

        // The choreography acks walk the tail to Done, each emitting exactly its one egress.
        let (s, acts) = step(&c, tail, SagaEvent::BatchAdopted);
        assert_eq!(
            s,
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitRelease,
                new_fence
            }
        );
        assert_eq!(
            acts,
            vec![SagaAction::EmitTransientRelease { fence: new_fence }]
        );
        let (s, acts) = step(&c, s, SagaEvent::SourceDropApplied);
        assert_eq!(
            s,
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitPromote,
                new_fence
            }
        );
        assert_eq!(
            acts,
            vec![SagaAction::EmitTransientPromote { fence: new_fence }]
        );
        let (s, acts) = step(&c, s, SagaEvent::DestDropApplied);
        assert_eq!(
            s,
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitComplete,
                new_fence
            }
        );
        assert_eq!(
            acts,
            vec![SagaAction::EmitReleaseComplete { fence: new_fence }]
        );
        let (done, acts) = step(&c, s, SagaEvent::SourceRetired);
        assert_eq!(done, SagaState::Done { new_fence });
        assert_eq!(acts, vec![SagaAction::Tombstone]);

        // A go-token Timeout in BatchCommitting re-drives the SAME go-token idempotently (forward-only,
        // never aborts) — the transient twin of the CommittingCas re-issue.
        let (held, actions) = step(&c, state, SagaEvent::Timeout);
        assert_eq!(held, state);
        assert_eq!(
            actions,
            vec![SagaAction::IssueTransientGo { expected: Fence(5) }]
        );

        // Each BatchHandoff phase RE-EMITS its one pending egress on a Timeout (the producer re-drive,
        // idempotent — the shard journals by step). AwaitAdopt re-drives NOTHING (the dest adopts off the
        // redelivered envelope), falling through to the no-op catch-all.
        let await_adopt = SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitAdopt,
            new_fence,
        };
        assert_eq!(
            step(&c, await_adopt, SagaEvent::Timeout),
            (await_adopt, vec![])
        );
        let await_release = SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitRelease,
            new_fence,
        };
        assert_eq!(
            step(&c, await_release, SagaEvent::Timeout),
            (
                await_release,
                vec![SagaAction::EmitTransientRelease { fence: new_fence }]
            )
        );
        let await_promote = SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitPromote,
            new_fence,
        };
        assert_eq!(
            step(&c, await_promote, SagaEvent::Timeout),
            (
                await_promote,
                vec![SagaAction::EmitTransientPromote { fence: new_fence }]
            )
        );
        let await_complete = SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitComplete,
            new_fence,
        };
        assert_eq!(
            step(&c, await_complete, SagaEvent::Timeout),
            (
                await_complete,
                vec![SagaAction::EmitReleaseComplete { fence: new_fence }]
            )
        );
    }

    #[test]
    fn batch_handoff_resolves_terminally_on_a_dead_participant() {
        // D-7d dead-resolution arms (terminal on FIRST fire — straight to Done + Tombstone from ANY
        // phase, NEVER back to BatchHandoff, so the D-37 forever-bounce is structurally impossible). A
        // dead SOURCE self-promotes the dest from the go-token (zero loss); a dead DEST abandons the
        // source's retained copy (accounted loss). Exercised from every phase (the `..` phase wildcard).
        let c = ctx_class(false, DurabilityClass::Transient);
        let nf = Fence(5).next();
        for phase in [
            BatchHandoffPhase::AwaitAdopt,
            BatchHandoffPhase::AwaitRelease,
            BatchHandoffPhase::AwaitPromote,
            BatchHandoffPhase::AwaitComplete,
        ] {
            let st = SagaState::BatchHandoff {
                phase,
                new_fence: nf,
            };
            assert_eq!(
                step(&c, st, SagaEvent::SourceUnreachable),
                (
                    SagaState::Done { new_fence: nf },
                    vec![
                        SagaAction::EmitTransientPromote { fence: nf },
                        SagaAction::Tombstone
                    ]
                ),
                "dead source → self-promote the dest from the go-token, then Done"
            );
            assert_eq!(
                step(&c, st, SagaEvent::DestUnreachable),
                (
                    SagaState::Done { new_fence: nf },
                    vec![
                        SagaAction::EmitTransientAbandon { fence: nf },
                        SagaAction::Tombstone
                    ]
                ),
                "dead dest → abandon the source copy as accounted loss, then Done"
            );
        }
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
    fn liveness_tuning_validate_rejects_a_mistuned_budget() {
        // D-3 Slice 0: the dead-vs-slow confirmation budget. The default (kill-only, n=1) validates;
        // a 0 confirmation count or a window too tight to hold the run is rejected LOUD at boot.
        assert_eq!(LivenessTuning::default().validate(), Ok(()));
        // A prod margin (n=3 within a window spanning 3 retry-spacings) validates.
        assert_eq!(
            LivenessTuning {
                n_consecutive_unreachable: 3,
                unreachable_window_ticks: 64,
                retry_delay_ticks_hint: 2,
            }
            .validate(),
            Ok(())
        );
        // ZeroConsecutive — would confirm a peer dead with no evidence.
        assert_eq!(
            LivenessTuning {
                n_consecutive_unreachable: 0,
                ..LivenessTuning::default()
            }
            .validate(),
            Err(LivenessTuningError::ZeroConsecutive)
        );
        // WindowTooTight — 3 notices at spacing 2 need a 6-tick span; a 4-tick window never confirms.
        assert_eq!(
            LivenessTuning {
                n_consecutive_unreachable: 3,
                unreachable_window_ticks: 4,
                retry_delay_ticks_hint: 2,
            }
            .validate(),
            Err(LivenessTuningError::WindowTooTight {
                window: 4,
                n: 3,
                retry: 2,
                span: 6,
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
                rehome_target: None,
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
                rehome_target: None,
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
            rehome_target: None,
        };
        let (state, acts) = step(&c, promoting, SagaEvent::DestDelivered);
        assert_eq!(
            state,
            SagaState::Promoting {
                new_fence: Fence(6),
                promote_acked: false,
                dest_delivered: true,
                rehome_target: None,
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
                rehome_target: None,
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
    fn a_confirmed_dead_source_in_demoting_self_promotes_the_committed_dest() {
        // D-37 CELL 1: a post-commit Demoting saga whose SOURCE is permanently killed never gets its
        // DemoteAcked (the corpse never replies). Injecting SourceUnreachable self-promotes the
        // already-committed live dest — the structural analogue of the DemoteAcked exit — carrying the
        // latched `dest_delivered` forward so the Promoting release gate never loses an early delivery.
        let c = ctx(false);
        let demoting = SagaState::Demoting {
            new_fence: Fence(6),
            dest_delivered: true,
        };
        let (state, acts) = step(&c, demoting, SagaEvent::SourceUnreachable);
        assert_eq!(
            state,
            SagaState::Promoting {
                new_fence: Fence(6),
                promote_acked: false,
                dest_delivered: true,
                rehome_target: None,
            },
            "a confirmed-dead source self-promotes the committed dest, carrying dest_delivered forward"
        );
        assert_eq!(
            acts,
            vec![SagaAction::Promote {
                new_fence: Fence(6)
            }]
        );
    }

    #[test]
    fn rehome_to_a_live_target_commits_adopts_then_aborts_on_cas_loss() {
        // D-37 CELL 2: a Promoting saga whose committed dest was permanently KILLED re-homes onto a LIVE
        // target. ReHomeTo{target} → ReHoming + ReHomeCommit (the CAS at prev_fence, the fence-monotone
        // bump); CasWon → Promoting at the target + the DEDICATED ReHome adopt; CasLost (rule 3) → a clean
        // Aborted no-op (the entity is alive at the CAS winner — no compensation).
        let c = ctx(false);
        let promoting = SagaState::Promoting {
            new_fence: Fence(6),
            promote_acked: false,
            dest_delivered: false,
            rehome_target: None,
        };
        let (state, acts) = step(&c, promoting, SagaEvent::ReHomeTo { target: NodeId(9) });
        assert_eq!(
            state,
            SagaState::ReHoming {
                target: NodeId(9),
                prev_fence: Fence(6),
            }
        );
        assert_eq!(
            acts,
            vec![SagaAction::ReHomeCommit {
                expected: Fence(6),
                target: NodeId(9),
            }]
        );
        // CasWon: the directory now names the target at the bumped fence → adopt there, re-enter Promoting
        // carrying `rehome_target: Some(target)` (D-37 2d) so a later Timeout re-drives the adopt here.
        let (state, acts) = step(&c, state, SagaEvent::CasWon { new_fence: Fence(7) });
        assert_eq!(
            state,
            SagaState::Promoting {
                new_fence: Fence(7),
                promote_acked: false,
                dest_delivered: false,
                rehome_target: Some(NodeId(9)),
            }
        );
        assert_eq!(
            acts,
            vec![SagaAction::ReHomeAdopt {
                new_fence: Fence(7),
                target: NodeId(9),
            }]
        );
        // CasLost from ReHoming → a clean terminal no-op (rule 3; the entity is alive at the CAS winner).
        let (state, acts) = step(
            &c,
            SagaState::ReHoming {
                target: NodeId(9),
                prev_fence: Fence(6),
            },
            SagaEvent::CasLost { current: Fence(8) },
        );
        assert_eq!(
            state,
            SagaState::Aborted {
                reason: AbortReason::CasLost,
            }
        );
        assert_eq!(acts, vec![SagaAction::Tombstone]);
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
            rehome_target: None,
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

    #[test]
    fn a_rehomed_promoting_timeout_redrives_the_adopt_to_the_live_target() {
        // D-37 Slice 2d: the SELF-SUFFICIENT re-home re-drive egress. A Promoting reached via a forward
        // re-home carries `rehome_target: Some(target)`, so a Timeout re-emits the dedicated `ReHomeAdopt`
        // to the LIVE target — NOT `A::Promote` to the confirmed-dead `ctx.dest` (which the re-home fired
        // BECAUSE of). This is what gives the re-home adopt an orchestrator re-drive instead of leaning on
        // transport at-least-once (DEFERRED D-6 precondition 1 / D-37). Covers the `Some` arm of the Timeout
        // branch (the `None` arm is covered above). The state survives the re-drive unchanged (idempotent).
        let c = ctx(false);
        let rehomed = SagaState::Promoting {
            new_fence: Fence(7),
            promote_acked: false,
            dest_delivered: false,
            rehome_target: Some(NodeId(9)),
        };
        let (state, acts) = step(&c, rehomed, SagaEvent::Timeout);
        assert_eq!(state, rehomed, "the re-home target survives the re-drive");
        assert_eq!(
            acts,
            vec![SagaAction::ReHomeAdopt {
                new_fence: Fence(7),
                target: NodeId(9),
            }],
            "a re-homed Promoting Timeout re-drives the adopt to the live target, not Promote to the dead dest"
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
            class in prop_oneof![
                Just(DurabilityClass::Durable),
                Just(DurabilityClass::Transient),
            ],
            events in proptest::collection::vec(arb_event(), 0..64),
        ) {
            // Both classes (D-7): the Durable walk AND the Transient short path are total + keep the
            // compensation invariant (a Transient never freezes, so the invariant holds vacuously).
            let c = ctx_class(provision, class);
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
        fn driving_with_matching_events_terminates(
            provision in proptest::bool::ANY,
            // D-37: when set, re-home ONCE from the first Promoting (ReHomeTo → ReHoming → CasWon →
            // Promoting), proving the forward-re-home path ALSO reaches a terminal in bounded steps.
            rehome in proptest::bool::ANY,
            class in prop_oneof![
                Just(DurabilityClass::Durable),
                Just(DurabilityClass::Transient),
            ],
        ) {
            let c = ctx_class(provision, class);
            let (mut state, _) = start(&c);
            let mut rehomed = false;
            for _ in 0..16 {
                let event = match state {
                    SagaState::AwaitProvision => SagaEvent::ProvisionReady,
                    SagaState::Preparing => SagaEvent::Prepared(PrepareResult::Ready),
                    SagaState::Cutting => SagaEvent::CutConfirmed { marker_seq: 1 },
                    // The transient short path: CasWon ends BatchCommitting at the post-commit tail
                    // (D-7d), then the adopt-before-drop choreography acks walk it to Done.
                    SagaState::BatchCommitting { .. } => SagaEvent::CasWon { new_fence: Fence(9) },
                    SagaState::BatchHandoff {
                        phase: BatchHandoffPhase::AwaitAdopt,
                        ..
                    } => SagaEvent::BatchAdopted,
                    SagaState::BatchHandoff {
                        phase: BatchHandoffPhase::AwaitRelease,
                        ..
                    } => SagaEvent::SourceDropApplied,
                    SagaState::BatchHandoff {
                        phase: BatchHandoffPhase::AwaitPromote,
                        ..
                    } => SagaEvent::DestDropApplied,
                    SagaState::BatchHandoff {
                        phase: BatchHandoffPhase::AwaitComplete,
                        ..
                    } => SagaEvent::SourceRetired,
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
                    // D-37: re-home ONCE from the first Promoting (the committed dest "died") — proving
                    // the re-home path terminates too. The CasWon below walks ReHoming back to Promoting.
                    SagaState::Promoting {
                        promote_acked: false,
                        ..
                    } if rehome && !rehomed => {
                        rehomed = true;
                        SagaEvent::ReHomeTo { target: NodeId(9) }
                    }
                    SagaState::ReHoming { .. } => SagaEvent::CasWon { new_fence: Fence(9) },
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
