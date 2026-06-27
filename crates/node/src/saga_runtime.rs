//! The Transfer Saga RUNTIME (P2 Slice 1b) — the orchestrator-side wrapper around the
//! pure [`vd_sim::saga`] FSM. It is the ONE owner of a transfer's lifecycle at runtime:
//!
//! - **Per-key serialization**: at most one saga per `DirectoryKey`, enforced by
//!   `DirectoryCore::lock_transfer` (the directory's `in_transfer` field) — a duplicate
//!   trigger is refused, not double-started.
//! - **Drives the gateway**: every `SagaAction::Send(TransferControl)` is serialized onto
//!   the `InterShardFlow::Saga` arm to the session's gateway (the gateway never decides a
//!   transfer; HR1 typed seam).
//! - **The single commit point, DIRECT**: `IssueCommitCas` calls `DirectoryCore::commit_cas`
//!   IN PROCESS — the orchestrator owns `DirectoryRes` on this same single-threaded schedule,
//!   so there is no self-addressed wire round-trip (no tick split, no `ResMut` race). The CAS
//!   outcome (`Won`/`Lost`) is fed straight back into the FSM as the next event.
//! - **Class-aware from line one (HR2)**: a `Transient` subject reaches the same commit point
//!   but issues the batched `TransientGo` go-token (a no-op stub until P3), never a per-entity
//!   CAS — the FSM's `commit_action` fan-out, executed here without a shard-kind branch.
//!
//! The sim thread NEVER awaits: all egress is enqueue-only (`OutboundBox`). At-least-once
//! delivery + adaptive timeouts + the CAS re-read loop land at Slice 2; the client-facing cut
//! cycle landed at Slice 1c/1e. The shard-bound `StubCrossing` state transfer is synthesized at
//! Slice 1d. The ORDERED demote-before-promote tail (1d.5b.1, D-2) pushes `Demote`→source then
//! (on `DemoteAcked`) `Promote`→dest, and releases the source sub only once the dest both acked
//! the promote AND delivered to every observer — the seamless no-vanish gate (`saga.rs` Promoting).
//!
//! ## Slice 2a — the RECOVERY MACHINERY (the R1 cure; closes D-1 + the post-commit park)
//! [`scan_deadlines`] is the production TIMEOUT PRODUCER: a `now - since >= deadline_for(phase)`
//! scan (two thresholds — cheap `redrive` for post-commit, large `abort` for the destructive
//! pre-freeze/compensation phases) that injects `SagaEvent::Timeout`, so a lost saga-ack RE-DRIVES
//! (post-commit) or ABORTS (pre-freeze) instead of PARKING forever. The terminal `Aborted` edge now
//! emits `SagaAction::ClearTransferLock` → [`DirectoryCore::abort_clear`] (the stale-fence head
//! re-read — a CAS-loser's `expected_fence` is stale by definition), so the directory lock CLEARS
//! and an aborted subject can immediately re-transfer (D-1 CLOSED; the two pinned asserts FLIPPED
//! to `None`). ⚠️ HONEST SCOPE: a fixed tick deadline is NOT a crash-vs-slow discriminator (only a
//! permanent `kill` emits `NodeUnreachable`) — a mis-tuned `abort_deadline_ticks` below a deployment's
//! worst-case healthy pre-freeze round-trip WILL false-abort a slow-but-alive saga (the real
//! discriminator is lease-lapse liveness, D-3, owed). The producer cures lost SAGA-ACKS, not a starved
//! standing delivery watermark (a `Promoting` saga whose dest frame delivery is blocked stays
//! half-open — owed, P3) nor an orchestrator CRASH (no durable WAL, D-6). (Caught by Slice-1b audit
//! CPO-1/CPO-2 + Slice-2a design `wf_9f22c70d`.)

use std::collections::{BTreeMap, VecDeque};

use bevy_ecs::prelude::{Res, ResMut, Resource};
use serde::{Deserialize, Serialize};
use vd_core::pose::StampedPose;
use vd_core::{BatchId, EpochId, Fence, NodeId, TransferId, UniverseTick};
use vd_sim::directory::{DirectoryCore, DirectoryTuning};
use vd_sim::io::{Bytes, Inbound, MsgClass, Store};
use vd_sim::runtime::{ClockSample, InboundBox, OutboundBox};
use vd_sim::capability::{CapRequest, ShardProfile};
use vd_sim::saga::{
    self, AbortReason, LivenessTuning, SagaAction, SagaCtx, SagaEvent, SagaState, SagaTuning,
};
use vd_wire::intershard::{
    DEMOTE_STEP, DemoteCmd, FLUSH_SOURCE_STEP, FlushSource, InterShardFlow, PROMOTE_STEP,
    PromoteCmd, RE_HOME_STEP, ReHomeCmd, ReHomeState, STUB_CROSSING_STEP, TRANSFER_SCHEMA_VERSION,
    TRANSIENT_ABANDON_STEP, TRANSIENT_DROP_STEP, TRANSIENT_RELEASE_STEP, TransferAck,
    TransferEnvelope, TransientHandoff, TransitionPayload,
};
use vd_wire::seams::directory::{AuthorityRef, CasOutcome, DirectoryKey, OwnerRecord};
use vd_wire::seams::transfer_control::TransferControlAck;

use crate::orchestrator::{DirectoryRes, UniverseClockRes};
use crate::universe_clock::{CeilingClock, ClockAction};

/// One live (in-flight) transfer's identity for the mid-flight AUTHORITY-UNIQUE oracle (1d.5b.3d):
/// the subject key + the source/dest shards. The oracle EXCUSES the post-CAS, pre-demote window (the
/// source still holds the subject `Owned` at the old fence while the directory already records the
/// dest) ONLY for a subject with a live saga of this exact (source→dest) shape — so a real split-brain
/// is never masked. Typed `DirectoryKey` (NOT the `String` `SagaView`) so the oracle matches Entity keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveTransfer {
    pub subject: DirectoryKey,
    pub source: NodeId,
    pub dest: NodeId,
}

/// The typed key families in the orchestrator's durable [`Store`] (D-6 saga WAL). Each key is a 1-byte
/// family TAG + the postcard-encoded id, so `scan(&[TAG])` recovers exactly one family on rehydrate. The
/// VALUE at each key is SELF-DESCRIBING (carries its own id/record), so rehydrate never parses a key back.
/// The `Directory` family is a DISTINCT prefix (independently splittable into its own io-prod file — the
/// D-32 partitioning seam — without a cross-file atomic transaction).
#[derive(Clone, Copy, Debug)]
enum StoreKey {
    Saga(TransferId),
    BatchGo(BatchId),
    Directory(DirectoryKey),
    Clock,
}

impl StoreKey {
    const SAGA: u8 = 1;
    const BATCH_GO: u8 = 2;
    const DIRECTORY: u8 = 3;
    const CLOCK: u8 = 4;

    /// The store key bytes: family tag + postcard(id). postcard encoding of these small fixed-shape
    /// types is infallible (no I/O); `.expect` is straight-line at this monomorphic site (the panic body
    /// is in stdlib, not a coverable caller branch — HR5, matching the codebase's `to_allocvec().expect`).
    fn bytes(self) -> Vec<u8> {
        let mut k = Vec::with_capacity(16);
        match self {
            StoreKey::Saga(t) => {
                k.push(Self::SAGA);
                k.extend(postcard::to_allocvec(&t).expect("encode TransferId store key"));
            }
            StoreKey::BatchGo(b) => {
                k.push(Self::BATCH_GO);
                k.extend(postcard::to_allocvec(&b).expect("encode BatchId store key"));
            }
            StoreKey::Directory(d) => {
                k.push(Self::DIRECTORY);
                k.extend(postcard::to_allocvec(&d).expect("encode DirectoryKey store key"));
            }
            StoreKey::Clock => k.push(Self::CLOCK),
        }
        k
    }
}

/// The durable snapshot of one live saga (D-6) — exactly the serializable [`LiveSaga`] fields, persisted
/// at `commit_result`'s write-back (the SINGLE point that knows the true QUIESCENT `final_state`: since
/// `IssueCommitCas` is a same-tick DIRECT call, `CommittingCas` is never a quiescent phase, so anchoring
/// only to the `PersistCheckpoint` emit sites would force a stale-`Cutting` restart abort of a maybe-
/// committed transfer). Re-hydrated on an orchestrator restart, then re-driven to terminal by the
/// existing Slice-2a Timeout producer (forward-only past Committed; shards dedup by `(transfer, step)`).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SagaSnapshot {
    ctx: SagaCtx,
    state: SagaState,
    gateway: NodeId,
    since: UniverseTick,
    flushed_pose: Option<StampedPose>,
}

/// The persisted go-token (D-6) — self-describing (carries its own `BatchId`) so rehydrate restores
/// `batch_goes` without parsing keys; one record per batch (the G-TIER `batch_go_writes` decoupling is
/// preserved on restart: rehydrate sets the counter to the map len, it does NOT re-drive the `+= 1`).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct GoTokenSnapshot {
    batch: BatchId,
    fence: Fence,
    source: NodeId,
    dest: NodeId,
}

/// The persisted directory record (D-6) — self-describing `(key, record)` so rehydrate calls
/// `DirectoryCore::restore` without parsing keys.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct DirSnapshot {
    key: DirectoryKey,
    record: OwnerRecord,
}

/// The orchestrator's durable store handle, resource-wrapped (D-6). Holds a `dyn Store` so the backend is
/// swappable behind the seam (MemStore now; redb in io-prod). Present on EVERY orchestrator (genesis uses
/// a fresh `MemStore`); the harness RETAINS the concrete handle to re-attach it to a rebuilt orchestrator
/// (the kill-9 analog — the World's RAM dies, the store's committed log survives).
#[derive(Resource)]
pub struct StoreRes(pub Box<dyn Store + Send + Sync>);

/// One live saga: the pure FSM (ctx + state) plus the wrapper-only routing — the gateway
/// `NodeId` to send `TransferControl` to (resolved at creation from the session's
/// directory record; not the FSM's concern).
struct LiveSaga {
    ctx: SagaCtx,
    state: SagaState,
    gateway: NodeId,
    /// The universe tick of the most recent transition that advanced this saga (set
    /// at creation, refreshed on every write-back). A PARKED saga receives no events,
    /// so `commit_result` is never re-entered for it and `since` stays put — which is
    /// exactly when it entered its current (stuck) state. The admin view reports
    /// `now - since` as staleness so a 2am operator sees how long a saga has been wedged.
    since: UniverseTick,
    /// The source's authoritative pose, stashed when its `SourceFlushed` reply arrives (1d.1) and
    /// read by the `EmitCrossing` executor at/after commit. The pose-before-promote gate guarantees
    /// it is `Some` before the CAS for an Entity subject, so `EmitCrossing` never ships a None pose.
    /// (The TLV state blob joins this at 1d.6; pose-only now.)
    flushed_pose: Option<StampedPose>,
    /// D-3 — the universe tick this saga's DEST was first observed CONFIRMED-DEAD while in `BatchHandoff`
    /// (`None` otherwise). The DESTRUCTIVE dest-abandon (`EmitTransientAbandon`) waits the LARGE
    /// `abort_deadline_ticks` measured FROM HERE — NOT from `since` (which `scan_deadlines` re-arms to
    /// `now` on every fire, making a `now - since >= abort` gate structurally unsatisfiable). RAM-ONLY
    /// (NOT persisted in `SagaSnapshot`): on rehydrate it re-derives `None`, so a restarted orchestrator
    /// re-accrues the abort budget from scratch rather than firing an irreversible abandon immediately.
    dead_observed_since: Option<UniverseTick>,
}

/// A pending create-on-trigger. Enqueued by [`SagaRuntimeRes::start_transfer`] and processed
/// next tick (a stub-shard boundary-crossing request wires the real trigger at Slice 1d).
struct PendingStart {
    ctx: SagaCtx,
    gateway: NodeId,
}

/// D-3 dead-vs-slow EVIDENCE for one peer: how many CONSECUTIVE `NodeUnreachable` notices (no
/// intervening successful inbound) have been observed toward it, and when the run STARTED — the
/// abort-budget anchor, PRESERVED across the run (never re-armed per notice, unlike the saga's `since`).
#[derive(Clone, Copy, Debug)]
struct UnreachEvidence {
    consecutive: u32,
    first_unreachable_tick: UniverseTick,
}

/// D-3 the dead-vs-slow discriminator (the CSCALE-1 cure). Replaces the insert-only / never-cleared
/// `dead_participants` set: a peer is CONFIRMED dead only after `n_consecutive_unreachable`
/// `NodeUnreachable` notices within `unreachable_window_ticks` with NO intervening successful inbound —
/// so a single recoverable blip toward a HEALTHY peer never confirms it dead, and a recovered peer
/// un-marks itself on its next inbound (clear-on-ack). RAM-only: rebuilt EMPTY on rehydrate (the D-6
/// freeze — a restarted orchestrator confirms nobody dead until fresh post-restart notices re-accrue,
/// so an orchestrator outage never mass-orphans). The default tuning (`n == 1`) is kill-equivalent.
#[derive(Default)]
struct LivenessTracker {
    seen: std::collections::BTreeMap<NodeId, UnreachEvidence>,
    tuning: LivenessTuning,
}

impl LivenessTracker {
    fn new(tuning: LivenessTuning) -> LivenessTracker {
        LivenessTracker {
            seen: std::collections::BTreeMap::new(),
            tuning,
        }
    }

    /// A `NodeUnreachable` toward `node` at `now`: EXTEND its consecutive run, or START a fresh run if
    /// the window since the run's first notice has elapsed (a stale run is not evidence of a live death).
    fn record_unreachable(&mut self, node: NodeId, now: UniverseTick) {
        let ev = self.seen.entry(node).or_insert(UnreachEvidence {
            consecutive: 0,
            first_unreachable_tick: now,
        });
        if now.0.saturating_sub(ev.first_unreachable_tick.0) > self.tuning.unreachable_window_ticks {
            *ev = UnreachEvidence {
                consecutive: 1,
                first_unreachable_tick: now,
            };
        } else {
            ev.consecutive = ev.consecutive.saturating_add(1);
        }
    }

    /// ANY successful inbound from `node`: it is alive — clear its evidence (idempotent; the clear-on-ack
    /// that makes a recoverable blip non-fatal).
    fn record_ack(&mut self, node: NodeId) {
        self.seen.remove(&node);
    }

    /// Is `node` CONFIRMED dead at `now`: enough consecutive notices AND the run still within the window
    /// (a too-old run is stale — re-confirmation needs fresh notices). Bitwise `&` keeps both operands
    /// covered with no short-circuit branch (HR5).
    fn is_confirmed_dead(&self, node: NodeId, now: UniverseTick) -> bool {
        let Some(ev) = self.seen.get(&node) else {
            return false;
        };
        let enough = ev.consecutive >= self.tuning.n_consecutive_unreachable;
        let fresh = now.0.saturating_sub(ev.first_unreachable_tick.0)
            <= self.tuning.unreachable_window_ticks;
        enough & fresh
    }

    /// Re-tune the discriminator (the prod config path sets it via `with_tunings`; a test scenario that
    /// needs a specific confirmation margin — e.g. the CSCALE-1 flap cells at `n = 3` — sets it here).
    fn set_tuning(&mut self, tuning: LivenessTuning) {
        self.tuning = tuning;
    }
}

/// The live-saga set + the trigger queue, resource-wrapped (single writer: this schedule).
#[derive(Resource, Default)]
pub struct SagaRuntimeRes {
    sagas: BTreeMap<TransferId, LiveSaga>,
    pending: Vec<PendingStart>,
    /// Typed client-facing rejections awaiting the surfacing system. BOUNDED ring
    /// (`REJECTION_LEDGER_CAP`): the PROPER consumer — the typed rejection→client channel —
    /// lands at Slice 1c.9 (CPO-4) and drains via `std::mem::take` (lifetime ONE cycle,
    /// matching `pending`). Until then this is capped so it can NEVER grow without limit on
    /// the orchestrator at MMO scale (audit ROB-1): on overflow the OLDEST rejections are
    /// shed with a counted + warned drop (`rejections_dropped`) — never an unbounded leak,
    /// never silent.
    pub rejected: Vec<(TransferId, AbortReason)>,
    /// Rejections shed because `rejected` reached `REJECTION_LEDGER_CAP` before the 1c.9
    /// consumer drained it — the loud overflow ALERT (0 in any healthy run).
    pub rejections_dropped: u64,
    /// The deadline budget the `scan_deadlines` producer reads (Slice 2a). Set from
    /// `OrchestratorConfig.saga` in `register_orchestrator`; `Default` is the dev/test value
    /// ([`SagaTuning::default`]) for the in-process rigs.
    tuning: SagaTuning,
    /// THE batched `TransientGo` go-token ledger (HR2/D-7): one record per `BatchId` =
    /// `(commit_fence, source, dest)`. Written by the `IssueTransientGo` executor at the transient
    /// SHORT-PATH commit point — ONE write per batch regardless of item count (G-TIER); the
    /// `BatchAdopted`/`DropApplied` handlers read it to drive the D-7b structural drop-before-promote
    /// (release the source, gate the dest promote, then ReleaseComplete the source). The
    /// `TRANSIENT-AUTHORITY-HELD` oracle cross-checks every Held transient's set
    /// anchor against a committed go-token here — a transient is NEVER a directory `OwnerRecord`
    /// (burst-isolation: a 1000-debris burst writes ZERO directory rows). ⚠️ IN-MEMORY (durability
    /// owed D-6). ⚠️ SAME UNBOUNDED-LEDGER CLASS as `rejected` (audit ROB-1) but NOT yet bounded:
    /// one `(Fence, NodeId, NodeId)` entry per batch EVER committed accumulates for orchestrator
    /// uptime (worst case = total batches committed since start) — UNLIKE `rejected`, this cannot be
    /// a cap-and-shed ring because the `TRANSIENT-AUTHORITY-HELD` oracle needs the live record while
    /// any item is Held. The proper GC fires at the shard's terminal retire (`on_release_complete`),
    /// gated on the drop-completion signal D-7d co-designs (owed D-7d, NOT a timeout). Until then do
    /// not run a long-lived burst soak without monitoring orchestrator RSS. `or_insert` idempotent:
    /// a Slice-2a re-driven go-token re-records the SAME (batch → fence) without duplication.
    batch_goes: BTreeMap<BatchId, (Fence, NodeId, NodeId)>,
    /// THE G-TIER observable (D-7c): a MONOTONIC count of go-token WRITES (incremented once per
    /// emitted `BatchGo`, BEFORE the idempotent `or_insert`). `batch_goes.len()` alone is FALSE-GREEN
    /// for the "write rate scales with BATCH count, not ITEM count" claim — the `or_insert` collapses N
    /// same-key writes to one entry, so a per-item-write regression would leave `len` at 1 yet inflate
    /// THIS counter to N. The G-TIER gate asserts `batch_go_writes == distinct-batch-count` (== 1 for a
    /// 1000-item single batch). The metric is forward-compatible with [[D-6]]'s durable WAL: one
    /// in-memory write per batch is exactly one fsync per batch.
    batch_go_writes: u64,
    /// D-3 the dead-vs-slow discriminator (CSCALE-1 cure): evidence-gated + clearable (was the insert-only
    /// `dead_participants` set). Fed by the `Inbound::NodeUnreachable` arm (`record_unreachable`) + cleared
    /// by ANY successful inbound (`record_ack`); `scan_deadlines` reads `is_confirmed_dead` to resolve a
    /// `BatchHandoff` saga whose source/dest is dead. A single recoverable blip no longer confirms a HEALTHY
    /// peer dead (needs `n_consecutive_unreachable` within the window), and the DESTRUCTIVE dest-abandon is
    /// further gated behind the LARGE `abort_deadline_ticks` from the per-saga `dead_observed_since`.
    liveness: LivenessTracker,
    /// D-7d anti-vacuity observables (orchestrator-side: the dest cannot tell a self-promote from a
    /// normal promote, so the resolution is counted HERE where it is decided). `source_unreachable`
    /// counts dead-SOURCE self-promote resolutions (BatchCommittedAt, zero loss); `dest_unreachable`
    /// counts dead-DEST abandon resolutions (BatchDroppedWithinBudget). Both 0 on every healthy run —
    /// `> 0` is the proof a crash cell's resolution actually fired (not that the happy path completed).
    source_unreachable_resolutions: u64,
    dest_unreachable_resolutions: u64,
    /// D-6 — durable writes STAGED this tick by `commit_result` (saga snapshots + go-tokens, encoded at
    /// the monomorphic call site), drained to the [`StoreRes`] + group-committed ONCE at the end of
    /// `drive_sagas` (the ~1-fsync/tick barrier). `Some(bytes)` = put, `None` = delete (a tombstoned
    /// saga). Held on the runtime (not threaded through `commit_result`) so the persist stays a
    /// straight-line push; the single drain+commit is the only place that touches the store for sagas.
    pending_writes: Vec<(Vec<u8>, Option<Bytes>)>,
    /// D-3 anti-vacuity observable: total `NodeUnreachable` notices fed to the liveness tracker. A
    /// CSCALE-1 flap cell asserts this is `> 0` (the blip was genuinely observed — the path the OLD
    /// insert-only code would have abandoned on) AND that `dest_unreachable_resolutions == 0` (the cure:
    /// a recoverable blip toward a healthy dest no longer abandons the batch). 0 on a no-fault run.
    liveness_notices: u64,
    /// D-3 Slice 4: the universe tick the expiry REAPER last swept, re-armed on fire so the O(directory)
    /// sweep runs at most once per `reaper_interval_ticks` (never per tick — like `scan_deadlines`' `since`).
    last_reap_tick: UniverseTick,
    /// D-3 Slice 4 CAP freeze: the reaper does NOT act before this tick. Set on rehydrate to
    /// `now + recovery_grace_ticks` — a FIXED post-restart freeze (belt-and-suspenders atop the RAM-empty
    /// tracker, which is the PRIMARY freeze: an empty tracker confirms nobody dead until fresh notices
    /// re-accrue). 0 at genesis (a fresh orchestrator has no stale leases to reap, so no freeze needed).
    liveness_quiesced_until: UniverseTick,
    /// D-37 forward re-home target roster: `NodeId → ShardProfile` for the shards a re-home may land on,
    /// set from `OrchestratorConfig.roster` (the cluster builder maps each stub shard to the empty
    /// profile). RAM-only operational config (rebuilt on rehydrate, like `liveness`/`clock_peers`).
    /// `select_rehome_target` reads it for the lowest LIVE capability-matched shard; an EMPTY roster
    /// (`Default`) ⇒ no target ⇒ the saga stays parked (honest), which every legacy rig expects.
    roster: BTreeMap<NodeId, ShardProfile>,
}

/// One batched `TransientGo` go-token emitted by the `IssueTransientGo` executor, COLLECTED by
/// `run_to_quiescence` (which is deliberately denied a `runtime` handle — `commit_result`'s
/// write-back soundness rests on it) and recorded into [`SagaRuntimeRes::batch_goes`] by
/// `commit_result`. `batch` is the `BatchId` (the transient saga's transfer); `fence` is the dest
/// realm-lease fence the batch committed at; `source`/`dest` are the shards the D-7b handoff commands
/// (`TransientRelease`/`TransientDrop`/`ReleaseComplete`) target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchGo {
    batch: BatchId,
    fence: Fence,
    source: NodeId,
    dest: NodeId,
}

/// Hard ceiling on the un-drained rejection ledger (operational param, ONE home — no inline
/// literal). Sized to absorb a large abort burst between 1c.9 drain cycles; beyond it the
/// oldest rejections are shed (counted) rather than grow the orchestrator without bound.
const REJECTION_LEDGER_CAP: usize = 1024;

impl SagaRuntimeRes {
    /// Construct with the deadline budget from config (Slice 2a). The production path
    /// (`register_orchestrator`) calls this with `cfg.saga`; `Default` is the dev/test value.
    #[must_use]
    pub fn with_tuning(tuning: SagaTuning) -> SagaRuntimeRes {
        // The dev/test path: the kill-equivalent liveness default (n == 1) — a permanent kill's first
        // NodeUnreachable confirms, so the existing crash cells keep their behavior.
        Self::with_tunings(tuning, LivenessTuning::default())
    }

    /// The production constructor (`register_orchestrator`): the saga deadline budget + the D-3 dead-vs-slow
    /// confirmation tuning, both from `OrchestratorConfig`. Prod sets `n_consecutive_unreachable >= 3` (the
    /// CSCALE-1 margin) so a single recoverable blip never confirms a healthy peer dead.
    #[must_use]
    pub fn with_tunings(tuning: SagaTuning, liveness: LivenessTuning) -> SagaRuntimeRes {
        SagaRuntimeRes {
            tuning,
            liveness: LivenessTracker::new(liveness),
            ..SagaRuntimeRes::default()
        }
    }

    /// Re-tune the dead-vs-slow discriminator after construction — for a test scenario that needs a
    /// specific confirmation margin (the CSCALE-1 flap cells run `n_consecutive_unreachable == 3` so a
    /// short blip stays below the threshold). Prod configures it via [`with_tunings`] / `OrchestratorConfig`.
    pub fn set_liveness_tuning(&mut self, liveness: LivenessTuning) {
        self.liveness.set_tuning(liveness);
    }

    /// Seed the D-37 re-home target roster (set in `register_orchestrator` from `OrchestratorConfig.roster`,
    /// re-seeded on rehydrate). A test rig calls this to make a re-home target available; the default
    /// (empty) roster makes `select_rehome_target` return `None` (the saga parks — honest).
    pub fn set_roster(&mut self, roster: BTreeMap<NodeId, ShardProfile>) {
        self.roster = roster;
    }

    /// Trigger a new transfer (create-on-trigger). `ctx.subject` MUST already be a directory
    /// record — [`drive_sagas`] asserts `lock_transfer` succeeds (a missing/already-locked
    /// subject refuses the start). `gateway` is the session's gateway (the caller resolves it
    /// from the `Session` directory record).
    ///
    /// 1c CONTRACT (audit CPO-4): a lock refusal is currently a silent drop (acceptable while
    /// the only caller is a test rig). When the 1c rejection-surfacing channel lands, refusals
    /// MUST route through that SAME channel with a typed reason (SubjectLocked /
    /// SubjectUnrecorded) so a shard-initiated trigger (1d) always learns its fate.
    pub fn start_transfer(&mut self, ctx: SagaCtx, gateway: NodeId) {
        self.pending.push(PendingStart { ctx, gateway });
    }

    /// Live saga count (the oracle/admin surface; bounded — terminal sagas are tombstoned).
    #[must_use]
    pub fn live(&self) -> usize {
        self.sagas.len()
    }

    /// Operator-facing view of every live (in-flight) saga — the admin snapshot's
    /// `sagas` list, the 2am `curl` that AAA-1 promised. Bounded: terminal sagas are
    /// tombstoned, so this is the in-flight set only, in deterministic `TransferId`
    /// order (BTreeMap). `SagaState` is `Debug`-only and lives in `vd-sim`, so the
    /// operator string is its `{:?}` (wire cannot depend on sim — admin.rs §SagaView).
    #[must_use]
    pub fn views(&self) -> Vec<vd_wire::admin::SagaView> {
        self.sagas
            .values()
            .map(|live| vd_wire::admin::SagaView {
                transfer: live.ctx.transfer.to_string(),
                state: format!("{:?}", live.state),
                since: live.since,
            })
            .collect()
    }

    /// The subject + source/dest of every LIVE (in-flight) saga — the typed ground truth the
    /// mid-flight AUTHORITY-UNIQUE oracle (1d.5b.3d) keys its W1 transfer-window excuse on. Empty once
    /// every saga is tombstoned, so the POST-QUIESCE oracle is strict (no transfer ⇒ no excuse). In
    /// deterministic `TransferId` order (BTreeMap). Orchestrator-only state — a shard never holds it.
    #[must_use]
    pub fn active_transfers(&self) -> Vec<ActiveTransfer> {
        self.sagas
            .values()
            .map(|live| ActiveTransfer {
                subject: live.ctx.subject,
                source: live.ctx.source,
                dest: live.ctx.dest,
            })
            .collect()
    }

    /// The committed batched-transient go-tokens — the `(BatchId, commit_fence)` ground truth the
    /// `TRANSIENT-AUTHORITY-HELD` oracle cross-checks every Held transient against (D-7). In
    /// deterministic `BatchId` order (BTreeMap). Orchestrator-only state (a shard never holds it).
    #[must_use]
    pub fn batch_goes(&self) -> Vec<(BatchId, Fence)> {
        self.batch_goes
            .iter()
            .map(|(batch, (fence, _src, _dst))| (*batch, *fence))
            .collect()
    }

    /// The MONOTONIC go-token write count (D-7c G-TIER observable): how many `BatchGo`s have been
    /// recorded, regardless of how many collapsed onto the same `BatchId` via `or_insert`. The gate
    /// asserts this equals the distinct-batch count (NOT the item count) — the proof the orchestrator
    /// write rate scales with batch count, never burst size.
    #[must_use]
    pub fn batch_go_writes(&self) -> u64 {
        self.batch_go_writes
    }

    /// D-7d — the count of dead-SOURCE self-promote resolutions (the dest was self-promoted from the
    /// go-token after the source died mid-handoff). 0 on every healthy run; `> 0` proves a SOURCE-kill
    /// cell's resolution fired (anti-vacuity for `BatchCommittedAt`).
    #[must_use]
    pub fn source_unreachable_resolutions(&self) -> u64 {
        self.source_unreachable_resolutions
    }

    /// D-7d — the count of dead-DEST abandon resolutions (the source's retained copy was dropped as an
    /// accounted loss after the dest died mid-handoff). 0 on every healthy run; `> 0` proves a DEST-kill
    /// cell's resolution fired (anti-vacuity for `BatchDroppedWithinBudget`).
    #[must_use]
    pub fn dest_unreachable_resolutions(&self) -> u64 {
        self.dest_unreachable_resolutions
    }

    /// D-3 anti-vacuity: total `NodeUnreachable` notices observed (the CSCALE-1 flap cell asserts `> 0`,
    /// so the blip genuinely exercised the path the old insert-only code would have abandoned on).
    #[must_use]
    pub fn liveness_notices(&self) -> u64 {
        self.liveness_notices
    }
}

/// Map a gateway→saga ack to the FSM event it drives. `CasWon`/`CasLost` are NOT here — they
/// come from the DIRECT `commit_cas`, never the wire.
fn ack_to_event(ack: TransferControlAck) -> SagaEvent {
    match ack {
        TransferControlAck::Prepared { result, .. } => SagaEvent::Prepared(result),
        TransferControlAck::CutConfirmed { marker_seq, .. } => {
            SagaEvent::CutConfirmed { marker_seq }
        }
        TransferControlAck::SourceFrozen { drained_seq, .. } => {
            SagaEvent::SourceFrozen { drained_seq }
        }
        // The gateway's `CommitAuthority` ack = the route swapped (post-commit, forward-only).
        TransferControlAck::Committed { .. } => SagaEvent::RouteSwapped,
        TransferControlAck::SourceThawed { .. } => SagaEvent::SourceThawed,
        TransferControlAck::Aborted { .. } => SagaEvent::DestAborted,
        TransferControlAck::Released { .. } => SagaEvent::Released,
        // 1d.5b.1 (D-2): the ordered demote/promote acks + the standing delivery watermark drive the
        // FSM tail directly — DemoteAcked: Demoting→Promoting; PromoteAcked + DestDelivered: the
        // Promoting→Releasing seamless gate (DestDelivered is also latched early in Demoting).
        TransferControlAck::DemoteAck { .. } => SagaEvent::DemoteAcked,
        TransferControlAck::PromoteAck { .. } => SagaEvent::PromoteAcked,
        TransferControlAck::DeliveredToObservers { .. } => SagaEvent::DestDelivered,
    }
}

/// Build the `StubCrossing` envelope the saga ships to the dest at/after commit (1d.1). Returns
/// `None` (a no-op) for a non-Entity subject or a missing flushed pose — the pose-before-promote
/// gate makes the latter unreachable for an Entity subject. The `fence` is the post-CAS authority
/// fence (fence rule 1: the receiver rejects a stale crossing); `state` is empty (the TLV blob
/// lands at 1d.6). Monomorphic helper so the executor arm stays a branchless dispatch (HR5).
fn build_crossing(
    ctx: &SagaCtx,
    fence: Fence,
    flush_pose: Option<StampedPose>,
    epoch: EpochId,
) -> Option<InterShardFlow> {
    let entity = ctx.subject.transfer_subject_entity()?;
    let pose = flush_pose?;
    Some(InterShardFlow::Transfer(TransferEnvelope {
        transfer_id: ctx.transfer,
        universe_epoch: epoch,
        schema_version: TRANSFER_SCHEMA_VERSION,
        fence,
        step_id: STUB_CROSSING_STEP,
        class: ctx.class,
        payload: TransitionPayload::StubCrossing {
            entity,
            from_realm: ctx.from_realm,
            to_realm: ctx.to_realm,
            pose,
            state: vec![],
        },
    }))
}

/// Stash the source's flushed pose onto its live saga, BEFORE the `SourceFlushed` event is
/// stepped — so an `EmitCrossing` reachable on the same tick reads it. A flush for an unknown /
/// GC'd saga is a stale reply: dropped (the event deliver is also a no-op).
fn stash_flush(runtime: &mut SagaRuntimeRes, transfer: TransferId, pose: StampedPose) {
    if let Some(live) = runtime.sagas.get_mut(&transfer) {
        live.flushed_pose = Some(pose);
    }
}

/// Push the crossing to the dest if one can be built (Entity subject + a stashed pose), else a
/// LOUD no-op. Holds the Some/None branch (covered both ways by unit tests) so the executor arm
/// stays branchless (HR5). For an Entity subject the pose-before-promote gate guarantees a pose,
/// so the None arm is reached only by the non-Entity subjects the FSM proptests drive.
fn emit_crossing(
    ctx: &SagaCtx,
    fence: Fence,
    flush_pose: Option<StampedPose>,
    epoch: EpochId,
    outbox: &mut OutboundBox,
) {
    match build_crossing(ctx, fence, flush_pose, epoch) {
        Some(crossing) => outbox.push_flow(ctx.dest, MsgClass::Saga, &crossing),
        None => tracing::warn!(
            transfer = ctx.transfer.0,
            "EmitCrossing skipped: non-Entity subject or no flushed pose"
        ),
    }
}

/// Build the `ReHome` adopt envelope the saga ships to a re-home `target` (D-37). Returns `None` (a
/// no-op) for a non-Entity subject or a missing flushed pose. A DEDICATED arm (NOT `Promote`): the target
/// RECONSTRUCTS the subject from `ReHomeState::PoseOnly`, having no pre-existing ghost. `new_fence` is the
/// post-CAS authority fence (fence rule 1). Monomorphic helper so the executor arm stays branchless (HR5).
fn build_rehome(
    ctx: &SagaCtx,
    new_fence: Fence,
    flush_pose: Option<StampedPose>,
) -> Option<InterShardFlow> {
    let _entity = ctx.subject.transfer_subject_entity()?;
    let pose = flush_pose?;
    Some(InterShardFlow::ReHome(ReHomeCmd {
        transfer: ctx.transfer,
        subject: ctx.subject,
        new_fence,
        step_id: RE_HOME_STEP,
        state: ReHomeState::PoseOnly(pose),
        source: ctx.source,
    }))
}

/// Push the re-home adopt to `target` if one can be built (Entity subject + a stashed pose), else a LOUD
/// no-op (mirrors `emit_crossing`). Holds the Some/None branch (covered both ways by unit tests) so the
/// executor arm stays a branchless dispatch (HR5).
fn emit_rehome(
    ctx: &SagaCtx,
    new_fence: Fence,
    target: NodeId,
    flush_pose: Option<StampedPose>,
    outbox: &mut OutboundBox,
) {
    match build_rehome(ctx, new_fence, flush_pose) {
        Some(rehome) => outbox.push_flow(target, MsgClass::Saga, &rehome),
        None => tracing::warn!(
            transfer = ctx.transfer.0,
            "ReHomeAdopt skipped: non-Entity subject or no flushed pose"
        ),
    }
}

/// The action executor: run a saga from `(state, actions)` to quiescence, executing each
/// action and feeding any SYNCHRONOUS follow-up event (the direct CAS outcome) back into the
/// FSM. Returns the final state, whether it tombstoned (terminal), and any client rejections.
///
/// ORDERING CONTRACT (audit FF-1): a batch's actions execute IN EMITTED ORDER and the WHOLE
/// batch completes before the next queued event is stepped — so when `CasWon` produces
/// `[PersistCheckpoint, Send(CommitAuthority), EmitCrossing]`, the sends happen before any later
/// event is processed. The FSM still emits at most ONE synchronous-feedback action (the commit)
/// per batch — `FlushSource`/`EmitCrossing` are pure egress (no event fed back) — so the loop
/// depth stays bounded (1d.1 preserves the FF-1 second-feedback-source bound).
///
/// `flush_pose` is the saga's stashed source pose (`Some` once `SourceFlushed` arrived); the
/// `EmitCrossing` arm reads it. `epoch` stamps the emitted crossing envelope.
#[allow(clippy::too_many_arguments)]
fn run_to_quiescence(
    ctx: &SagaCtx,
    gateway: NodeId,
    mut state: SagaState,
    mut actions: Vec<SagaAction>,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
    flush_pose: Option<StampedPose>,
) -> (
    SagaState,
    bool,
    Vec<(TransferId, AbortReason)>,
    Vec<BatchGo>,
) {
    let mut events: VecDeque<SagaEvent> = VecDeque::new();
    let mut tombstone = false;
    let mut rejected = Vec::new();
    let mut batch_gos = Vec::new();
    loop {
        for action in actions {
            match action {
                SagaAction::Send(cmd) => {
                    outbox.push_flow(gateway, MsgClass::Saga, &InterShardFlow::Saga(cmd));
                }
                // Tell the SOURCE to ship the subject's pose (1d.1). Pure egress to ctx.source —
                // no synchronous feedback (FF-1 preserved).
                SagaAction::FlushSource => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::FlushSource(FlushSource {
                            transfer: ctx.transfer,
                            subject: ctx.subject,
                            step_id: FLUSH_SOURCE_STEP,
                        }),
                    );
                }
                // Emit the entity-STATE crossing to the DEST (1d.1) from the stashed pose. The
                // Some/None branch lives in `emit_crossing` (a monomorphic helper, unit-tested both
                // ways), so this arm stays a branchless dispatch (HR5).
                SagaAction::EmitCrossing { fence } => {
                    emit_crossing(ctx, fence, flush_pose, epoch, outbox);
                }
                // Push the ORDERED Demote to the SOURCE shard (1d.5b.1, D-2) — the fence-enforced
                // Owned→Frozen→Ghost that REPLACES the 1c.8 poll. Pure egress to ctx.source (its
                // DemoteAck rides InterShardFlow::SagaAck back); no synchronous feedback (FF-1).
                SagaAction::Demote { new_owner_fence } => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::Demote(DemoteCmd {
                            transfer: ctx.transfer,
                            subject: ctx.subject,
                            new_owner_fence,
                            step_id: DEMOTE_STEP,
                        }),
                    );
                }
                // Push the ORDERED Promote to the DEST shard (1d.5b.1, D-2), emitted only after
                // DemoteAcked (demote-before-promote). Pure egress to ctx.dest (its PromoteAck rides
                // SagaAck back); no synchronous feedback (FF-1).
                SagaAction::Promote { new_fence } => {
                    outbox.push_flow(
                        ctx.dest,
                        MsgClass::Saga,
                        &InterShardFlow::Promote(PromoteCmd {
                            transfer: ctx.transfer,
                            subject: ctx.subject,
                            new_fence,
                            step_id: PROMOTE_STEP,
                            // The dest registers this SOURCE as a ghost-neighbor + drives the
                            // GhostFlow collider feed to it after promoting (1d.5b.3b).
                            source: ctx.source,
                        }),
                    );
                }
                // THE single commit point, called DIRECTLY (this thread owns the directory).
                // The outcome re-enters the FSM immediately — no wire round-trip, no tick split.
                // ⚠️ SCALE (DEFERRED D-32): this in-process DIRECT CAS is correct ONLY while ONE
                // orchestrator owns the WHOLE keyspace. When the directory partitions by region
                // (N-orchestrator scaling), this must ROUTE a CommitCas op to `ctx.subject`'s
                // coordinator (`coordinator_of(key)`), not call `commit_cas` locally.
                SagaAction::IssueCommitCas { expected } => {
                    let outcome =
                        dir.commit_cas(ctx.subject, expected, AuthorityRef::Shard(ctx.dest), now);
                    events.push_back(match outcome {
                        CasOutcome::Won { new_fence } => SagaEvent::CasWon { new_fence },
                        CasOutcome::Lost { current } => SagaEvent::CasLost { current },
                    });
                }
                // D-37 forward re-home commit — the SAME single commit point (D-32 routing caveat applies
                // identically), re-pointed at the live `target` rather than the hardcoded `ctx.dest`. The
                // `cas_next` bump (`expected` → `expected+1`) atomically names `target`, clears the lock,
                // and strictly stales the dead owner's claim (the fence-monotone no-double-owner property).
                // Feeds `CasWon`/`CasLost` back exactly like `IssueCommitCas` (FF-1: one feedback event).
                SagaAction::ReHomeCommit { expected, target } => {
                    let outcome =
                        dir.commit_cas(ctx.subject, expected, AuthorityRef::Shard(target), now);
                    events.push_back(match outcome {
                        CasOutcome::Won { new_fence } => SagaEvent::CasWon { new_fence },
                        CasOutcome::Lost { current } => SagaEvent::CasLost { current },
                    });
                }
                // D-37 forward re-home adopt — emit the dedicated `ReHome` to `target` from the stashed
                // pose. The Some/None branch lives in `emit_rehome` (unit-tested both ways), so this arm
                // stays a branchless dispatch (HR5). Pure egress (the target's `PromoteAck` rides back).
                SagaAction::ReHomeAdopt { new_fence, target } => {
                    emit_rehome(ctx, new_fence, target, flush_pose, outbox);
                }
                // D-7: the batched `TransientGo` go-token IS the transient commit point (HR2 — the
                // SAME `CasWon` feedback as the durable CAS, fanned out by `commit_action`, NEVER a
                // per-entity directory CAS: transients aren't `OwnerRecord`s, so a 1000-debris burst
                // writes ZERO directory rows — burst isolation, G-TIER). Collect the go-token (it is
                // recorded into `runtime.batch_goes` by `commit_result` — `run_to_quiescence` is
                // intentionally denied the runtime handle) and feed `CasWon` back so the SHORT-PATH
                // saga reaches `Done` (FF-1: exactly ONE feedback event, like `IssueCommitCas`). The
                // commit fence is `expected` — the dest realm-lease fence the batch set anchors to.
                SagaAction::IssueTransientGo { expected } => {
                    batch_gos.push(BatchGo {
                        batch: BatchId(ctx.transfer),
                        fence: expected,
                        source: ctx.source,
                        dest: ctx.dest,
                    });
                    events.push_back(SagaEvent::CasWon {
                        new_fence: expected,
                    });
                }
                // D-7d transient post-commit tail egress (replacing the deleted ledger-driven
                // handle_batch_adopted/handle_drop_applied): pure egress to ctx.source/ctx.dest carrying
                // the committed go-token `fence`. The shard journals each by (transfer, step), so a
                // Timeout re-emit (the producer re-drive) is idempotent. FF-1: no synchronous feedback —
                // the shard's ack rides TransferAck back, routed to the FSM in `drive_sagas`.
                SagaAction::EmitTransientRelease { fence } => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::TransientRelease(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_RELEASE_STEP,
                            fence,
                        }),
                    );
                }
                SagaAction::EmitTransientPromote { fence } => {
                    outbox.push_flow(
                        ctx.dest,
                        MsgClass::Saga,
                        &InterShardFlow::TransientDrop(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_DROP_STEP,
                            fence,
                        }),
                    );
                }
                SagaAction::EmitReleaseComplete { fence } => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::ReleaseComplete(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_RELEASE_STEP,
                            fence,
                        }),
                    );
                }
                // D-7d dead-DEST resolution: the source ABANDONS the batch as an accounted loss (the
                // dest died, so the promote target is gone). Pure egress to ctx.source carrying the
                // go-token fence; the source journals `(transfer, TRANSIENT_ABANDON_STEP)`. FF-1.
                SagaAction::EmitTransientAbandon { fence } => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::TransientAbandon(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_ABANDON_STEP,
                            fence,
                        }),
                    );
                }
                // D-6: durability is REAL now — `commit_result` stages the QUIESCENT saga snapshot at
                // EVERY transition (this checkpoint's `Send`/`EmitCrossing` effect included), and the
                // end-of-tick `Store::commit()` barrier flushes it BEFORE the node's flush phase sends
                // the tick's effects (persist-before-effect). So the two emit sites (dest-ghost-exists,
                // authority-flipped) need no per-action work — the write-back + barrier subsume them.
                SagaAction::PersistCheckpoint => {}
                SagaAction::NotifyRejected(reason) => rejected.push((ctx.transfer, reason)),
                // Clear the directory lock at TERMINAL abort (Slice 2a, closes D-1), called DIRECTLY
                // (this thread owns the directory — same as IssueCommitCas). The stale-fence re-read
                // (`abort_clear`); the saga is already terminal so the outcome feeds nothing back into
                // the FSM (Won on the first terminal, Lost on an idempotent re-driven terminal). FF-1.
                SagaAction::ClearTransferLock => {
                    let _ = dir.abort_clear(ctx.subject, ctx.transfer);
                }
                SagaAction::Tombstone => tombstone = true,
            }
        }
        match events.pop_front() {
            Some(event) => {
                let (next, acts) = saga::step(ctx, state, event);
                state = next;
                actions = acts;
            }
            None => break,
        }
    }
    (state, tombstone, rejected, batch_gos)
}

/// Apply one event to an existing saga, then run it to quiescence + persist the result. A
/// stale event for an unknown/GC'd saga is an idempotent no-op (at-least-once delivery).
fn deliver(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
    transfer: TransferId,
    event: SagaEvent,
) {
    let Some(live) = runtime.sagas.get_mut(&transfer) else {
        return;
    };
    let ctx = live.ctx;
    let gateway = live.gateway;
    let flush_pose = live.flushed_pose; // Copy; the EmitCrossing executor reads it
    let (state, actions) = saga::step(&ctx, live.state, event);
    let (final_state, tombstone, rejected, batch_gos) = run_to_quiescence(
        &ctx, gateway, state, actions, dir, outbox, epoch, now, flush_pose,
    );
    commit_result(
        runtime,
        transfer,
        final_state,
        tombstone,
        rejected,
        batch_gos,
        now,
    );
}

/// ROB-1: bound the un-drained rejection ledger to `REJECTION_LEDGER_CAP`, shedding the
/// OLDEST beyond it with a counted, warned drop — never an unbounded leak, never silent.
/// Monomorphic helper (the branch is covered once; `commit_result` stays a straight shim).
fn bound_rejection_ledger(runtime: &mut SagaRuntimeRes) {
    let shed = runtime.rejected.len().saturating_sub(REJECTION_LEDGER_CAP);
    if shed > 0 {
        runtime.rejected.drain(0..shed);
        runtime.rejections_dropped += shed as u64;
        tracing::warn!(
            shed,
            cap = REJECTION_LEDGER_CAP,
            "saga rejection ledger over cap: shed oldest (un-surfaced until Slice 1c.9 CPO-4); \
             never silent"
        );
    }
}

/// Persist a quiescent saga: record rejections + the batched go-tokens, then GC (terminal) or write
/// back the state.
fn commit_result(
    runtime: &mut SagaRuntimeRes,
    transfer: TransferId,
    final_state: SagaState,
    tombstone: bool,
    rejected: Vec<(TransferId, AbortReason)>,
    batch_gos: Vec<BatchGo>,
    now: UniverseTick,
) {
    runtime.rejected.extend(rejected);
    bound_rejection_ledger(runtime); // ROB-1: never let the un-drained ledger grow unbounded
    // D-7: record the batched go-tokens emitted this run (here, NOT in `run_to_quiescence` — which
    // is denied the runtime handle so this write-back stays sound). `or_insert` is idempotent: a
    // Slice-2a re-driven go-token re-records the SAME (batch → fence/source/dest), never a dup.
    for bg in batch_gos {
        // D-7c: count the WRITE (the G-TIER observable) BEFORE the idempotent `or_insert` — so a
        // per-item-write regression inflates this to N even though `or_insert` keeps `len` at 1.
        runtime.batch_go_writes += 1;
        runtime
            .batch_goes
            .entry(bg.batch)
            .or_insert((bg.fence, bg.source, bg.dest));
        // D-6: STAGE the go-token durably (self-describing; one record per batch — preserves the
        // G-TIER decouple, since rehydrate restores `batch_go_writes` from the map len, never re-driving
        // the `+= 1`). Committed at the end-of-tick barrier with the saga snapshots.
        let token = GoTokenSnapshot {
            batch: bg.batch,
            fence: bg.fence,
            source: bg.source,
            dest: bg.dest,
        };
        runtime
            .pending_writes
            .push((StoreKey::BatchGo(bg.batch).bytes(), Some(encode(&token))));
    }
    if tombstone {
        runtime.sagas.remove(&transfer);
        // D-6: a tombstoned saga's durable snapshot is DELETED — rehydrate's Saga-scan must not
        // resurrect it (the matching directory mutation commits in the SAME barrier, so the durable
        // saga set + directory are always consistent post-crash).
        runtime
            .pending_writes
            .push((StoreKey::Saga(transfer).bytes(), None));
    } else {
        // Sound because nothing between the lookup and this write-back can touch
        // `runtime.sagas`: `run_to_quiescence` takes only `dir`/`outbox` (the borrow
        // checker enforces it cannot reach the runtime), and this schedule is
        // single-threaded. If a future refactor hands it the runtime, re-prove this.
        let live = runtime
            .sagas
            .get_mut(&transfer)
            .expect("the saga was present at the start of this run");
        // AAA-1 staleness: refresh `since` ONLY on an ACTUAL state change. `deliver` runs
        // for EVERY matching ack — including at-least-once duplicates / stray acks the FSM
        // absorbs as same-state (`saga.rs`'s `(state, _) => (state, vec![])` no-op arm) — so
        // an unconditional bump would RESET a parked saga's staleness on a redelivery,
        // defeating the very stuck-saga metric this field exists for. The stored `live.state`
        // is still the prior state here (single-threaded schedule; nothing touched it between
        // the lookup and now), so the compare is exact.
        if live.state != final_state {
            live.since = now;
        }
        live.state = final_state;
        // D-6: STAGE the QUIESCENT snapshot (built by copying the just-updated live saga, so `live`'s
        // borrow ends before the disjoint `pending_writes` push — NLL). The end-of-tick barrier commits
        // it before the node's flush phase sends this tick's effects (persist-before-effect).
        let snapshot = SagaSnapshot {
            ctx: live.ctx,
            state: live.state,
            gateway: live.gateway,
            since: live.since,
            flushed_pose: live.flushed_pose,
        };
        runtime
            .pending_writes
            .push((StoreKey::Saga(transfer).bytes(), Some(encode(&snapshot))));
    }
}

/// Encode a durable record to [`Bytes`] (D-6). A BRANCHLESS generic shim (postcard to a `Vec` is
/// infallible for these fixed-shape types; the `.expect` panic body lives in stdlib, not a coverable
/// caller branch — HR5, matching the codebase's `to_allocvec().expect` idiom). Instantiated per record
/// type, each covered by its persist site.
fn encode<T: Serialize>(value: &T) -> Bytes {
    postcard::to_allocvec(value)
        .expect("postcard encodes a durable record")
        .into()
}

/// The orchestrator state recovered from a non-empty durable [`Store`] on restart (D-6).
pub(crate) struct Rehydrated {
    pub clock: CeilingClock,
    pub directory: DirectoryCore,
    pub runtime: SagaRuntimeRes,
}

/// Reconstruct the orchestrator's durable state from the [`Store`] on a kill-9 restart (D-6). Returns
/// `None` when the store is empty — a fresh GENESIS orchestrator (no `Clock` record committed yet). On
/// RECOVER: the clock resumes FORWARD at the persisted ceiling (`CeilingClock::recover` — never rewinds,
/// so `scan_deadlines` stays live); the directory + go-tokens restore verbatim (`batch_go_writes` set to
/// the map len, NOT re-driven through the `+= 1` — the G-TIER decouple survives); each saga re-hydrates
/// with `since` ARMED to 0 so the first `scan_deadlines` fires it and the EXISTING per-phase Timeout
/// producer re-drives it to terminal (forward-only past Committed — NO new recovery path). The directory
/// and the saga set commit in the SAME barrier, so they are always CONSISTENT post-crash (a `Swapping`
/// snapshot pairs a dest-owned directory; a `Freezing` snapshot pairs a source-owned directory) — the
/// Timeout arms reconcile correctly without a bespoke head-re-read (that is the io-prod separate-file future).
pub(crate) fn rehydrate(
    store: &dyn Store,
    reserve_chunk: u64,
    saga_tuning: SagaTuning,
    liveness: LivenessTuning,
    dir_tuning: DirectoryTuning,
) -> Option<Rehydrated> {
    // The Clock family is the genesis-vs-recover discriminator: absent ⇒ no tick ever committed ⇒ genesis.
    let clock_recs = store.scan(&[StoreKey::CLOCK]);
    let (_clock_key, clock_bytes) = clock_recs.first()?;
    let (epoch, ceiling): (EpochId, UniverseTick) =
        postcard::from_bytes(clock_bytes).expect("decode persisted clock ceiling");
    let (mut clock, ClockAction::ReserveCeiling(next)) =
        CeilingClock::recover(epoch, ceiling, reserve_chunk)
            .expect("non-zero reserve chunk on recover");
    clock
        .confirm_ceiling(next)
        .expect("the recovered reservation confirms in-memory exactly once");

    let directory = DirectoryCore::restore(
        dir_tuning,
        store
            .scan(&[StoreKey::DIRECTORY])
            .into_iter()
            .map(|(_k, v)| {
                let snapshot: DirSnapshot =
                    postcard::from_bytes(&v).expect("decode persisted directory record");
                (snapshot.key, snapshot.record)
            }),
    );

    let mut sagas: BTreeMap<TransferId, LiveSaga> = BTreeMap::new();
    for (_k, v) in store.scan(&[StoreKey::SAGA]) {
        let snapshot: SagaSnapshot =
            postcard::from_bytes(&v).expect("decode persisted saga snapshot");
        sagas.insert(
            snapshot.ctx.transfer,
            LiveSaga {
                ctx: snapshot.ctx,
                state: snapshot.state,
                gateway: snapshot.gateway,
                // ARM: `now` jumped forward to the recovered ceiling, so `now - 0 >= deadline` fires the
                // re-drive on the first post-restart `scan_deadlines` tick (deterministic, never wedged).
                since: UniverseTick(0),
                flushed_pose: snapshot.flushed_pose,
                // D-3: NOT persisted — a rehydrated saga re-accrues its abort budget from scratch (so a
                // restart never fires an immediate irreversible abandon; the RAM tracker is also empty).
                dead_observed_since: None,
            },
        );
    }

    let mut batch_goes: BTreeMap<BatchId, (Fence, NodeId, NodeId)> = BTreeMap::new();
    for (_k, v) in store.scan(&[StoreKey::BATCH_GO]) {
        let token: GoTokenSnapshot = postcard::from_bytes(&v).expect("decode persisted go-token");
        batch_goes.insert(token.batch, (token.fence, token.source, token.dest));
    }
    let batch_go_writes = batch_goes.len() as u64;

    // D-3: thread the CONFIGURED liveness tuning through the recover path so a kill-9-rebuilt orchestrator
    // keeps its prod dead-vs-slow margin (n=3), not the kill-equivalent n=1 default. The tracker itself is
    // rebuilt EMPTY (the CAP freeze), but its TUNING must survive the restart.
    let mut runtime = SagaRuntimeRes::with_tunings(saga_tuning, liveness);
    runtime.sagas = sagas;
    runtime.batch_goes = batch_goes;
    runtime.batch_go_writes = batch_go_writes;
    // D-3 Slice 4 CAP freeze: a rebuilt orchestrator does NOT reap for `recovery_grace_ticks` after recover
    // — belt-and-suspenders atop the RAM-empty tracker (the PRIMARY freeze), so even a fast restart cannot
    // race a slow-rejoining live peer into a reap before its first renewal re-lands. Measured from the
    // recovered ceiling (the resumed `now`).
    runtime.liveness_quiesced_until =
        UniverseTick(ceiling.0.saturating_add(dir_tuning.recovery_grace_ticks));

    Some(Rehydrated {
        clock,
        directory,
        runtime,
    })
}

/// Process the create-on-trigger queue: per-key-serialize via `lock_transfer`, `start` the
/// FSM, and run its initial actions to quiescence.
fn process_starts(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
) {
    for PendingStart { ctx, gateway } in std::mem::take(&mut runtime.pending) {
        // GUARD (audit D6-1): a re-trigger for an ALREADY-LIVE transfer is a no-op — never CLOBBER an
        // in-flight saga. The durable path's `lock_transfer` already refuses a second saga (the subject
        // is `in_transfer`-locked), but a TRANSIENT batch takes NO lock, so an unconditional re-insert
        // would overwrite its live `BatchHandoff` saga (resetting the choreography). One key → one saga.
        if runtime.sagas.contains_key(&ctx.transfer) {
            continue;
        }
        // Per-key serialization (DURABLE only): `lock_transfer` sets the directory `in_transfer`.
        // FALSE means the subject is absent OR already transferring — refuse to start (a
        // spurious/duplicate trigger is a no-op, never a second concurrent saga on the same key). A
        // TRANSIENT batch is NOT in the directory (`locks_directory_key(Transient)=false`) — burst
        // isolation is STRUCTURAL: a debris burst takes ZERO directory locks, so it can never
        // serialize against / wedge a concurrent Durable saga (HR2). The short-circuit `&&` is
        // HR5-coverable here: `locks_directory_key` is exercised true (Durable) AND false (Transient),
        // and its true arm's `lock_transfer` is exercised both ways (the durable happy path = ok; the
        // unrecorded-subject refusal = fail) — so every branch of both operands is hit.
        if locks_directory_key(ctx.class) && !dir.lock_transfer(ctx.subject, ctx.transfer) {
            continue;
        }
        let (state, actions) = saga::start(&ctx);
        runtime.sagas.insert(
            ctx.transfer,
            LiveSaga {
                ctx,
                state,
                gateway,
                since: now,
                flushed_pose: None, // filled when the source flushes (after Freezing)
                dead_observed_since: None,
            },
        );
        // Durable start = PrepareSubscribe (no EmitCrossing yet); transient start = the go-token
        // (collected by run_to_quiescence). `None` flush_pose is correct for both at start.
        let (final_state, tombstone, rejected, batch_gos) =
            run_to_quiescence(&ctx, gateway, state, actions, dir, outbox, epoch, now, None);
        commit_result(
            runtime,
            ctx.transfer,
            final_state,
            tombstone,
            rejected,
            batch_gos,
            now,
        );
    }
}

/// Whether a saga of this class takes a directory `in_transfer` LOCK at start (HR2 fan-out, the
/// burst-isolation seam): `Durable` subjects are per-entity `OwnerRecord`s serialized by the
/// directory; `Transient` batches are NOT in the directory at all (held-set anchored, committed by
/// the batched go-token), so a debris burst takes zero locks and can never wedge a Durable saga. An
/// explicit `match` (not `matches!`) so both arms are covered regions (HR5).
#[must_use]
fn locks_directory_key(class: vd_core::entity_kind::DurabilityClass) -> bool {
    match class {
        vd_core::entity_kind::DurabilityClass::Durable => true,
        vd_core::entity_kind::DurabilityClass::Transient => false,
    }
}

/// The deadline a saga's CURRENT phase fires its `Timeout` at (Slice 2a). Pre-freeze/freeze/aborting
/// phases use the LARGE `abort_deadline_ticks` (their Timeout is DESTRUCTIVE — an abort — or re-drives
/// a compensator on the same slow latency profile); committing/post-commit phases use the small
/// `redrive_deadline_ticks` (a cheap idempotent re-drive that may fire "early" at zero correctness
/// cost); terminal phases return `u64::MAX` (never fire). A straight monomorphic match — the ONLY
/// per-phase logic, hoisted out of the phase-agnostic producer scan (HR3).
#[must_use]
fn deadline_for(state: &SagaState, tuning: &SagaTuning) -> u64 {
    match state {
        SagaState::AwaitProvision
        | SagaState::Preparing
        | SagaState::Cutting
        | SagaState::Freezing { .. }
        | SagaState::Aborting { .. } => tuning.abort_deadline_ticks,
        SagaState::CommittingCas { .. }
        | SagaState::BatchCommitting { .. }
        | SagaState::BatchHandoff { .. }
        | SagaState::Swapping { .. }
        | SagaState::Demoting { .. }
        | SagaState::Promoting { .. }
        // D-37: a CAS-in-flight state like CommittingCas. In-process the re-home CAS feedback is
        // synchronous (run_to_quiescence resolves CasWon/CasLost before the tick ends), so a saga never
        // PERSISTS in ReHoming and this deadline is not consulted at runtime; the redrive value is the
        // consistent CAS-in-flight choice (and the D-32 async-routed CAS re-drive home, if it ever persists).
        | SagaState::ReHoming { .. }
        | SagaState::Releasing { .. } => tuning.redrive_deadline_ticks,
        SagaState::Done { .. } | SagaState::Aborted { .. } => u64::MAX,
    }
}

/// The dead-aware deadline event for a due saga (D-7d transient hand-off + D-3 liveness discriminator +
/// D-37 durable re-home), hoisted out of [`scan_deadlines`] so the scan stays a branchless dispatch and
/// every branch is covered in ONE monomorphic place (HR5). A re-drive toward a CONFIRMED-DEAD participant
/// (`is_confirmed_dead` — the evidence-gated run, NOT a single blip) becomes a RESOLUTION instead of
/// looping at a corpse forever; a healthy slow peer (no kill ⇒ no NodeUnreachable run ⇒ never confirmed)
/// only ever re-drives via plain `Timeout`. SOURCE-dead is the cheap NON-destructive path (self-promote
/// the already-committed dest/batch — the demote toward the corpse is moot, its claim excluded by the
/// dead-aware oracle). DEST-dead is the DESTRUCTIVE budget gate: resolve only after `abort_deadline_ticks`
/// measured from `dead_observed_since` (the CSCALE-1 cure — a recoverable blip that clears in time never
/// resolves a healthy dest); `dead_observed_since` is the per-saga budget anchor, NOT `since` (which
/// re-armed to `now`), so the budget accrues across re-drives rather than resetting every fire.
#[allow(clippy::too_many_arguments)] // the dead-aware decision needs the full saga + liveness + roster context
fn rehome_event_for(
    state: &SagaState,
    ctx: &SagaCtx,
    liveness: &LivenessTracker,
    dead_observed_since: &mut Option<UniverseTick>,
    tuning: &SagaTuning,
    now: UniverseTick,
    roster: &BTreeMap<NodeId, ShardProfile>,
    req: &CapRequest,
    subject_owner: NodeId,
) -> SagaEvent {
    match state {
        // D-7d transient batch hand-off: source-dead self-promotes the dest, dest-dead abandons the
        // source's retained copy (DESTRUCTIVE, budget-gated).
        SagaState::BatchHandoff { .. } => {
            if liveness.is_confirmed_dead(ctx.source, now) {
                *dead_observed_since = None;
                SagaEvent::SourceUnreachable
            } else if liveness.is_confirmed_dead(ctx.dest, now) {
                let observed = *dead_observed_since.get_or_insert(now);
                if now.0.saturating_sub(observed.0) >= tuning.abort_deadline_ticks {
                    SagaEvent::DestUnreachable
                } else {
                    SagaEvent::Timeout // cheap re-drive while the abort budget accrues (corpse won't ack)
                }
            } else {
                *dead_observed_since = None; // neither confirmed dead → clear stale budget, re-drive
                SagaEvent::Timeout
            }
        }
        // D-37 CELL 1: a POST-commit `Demoting` saga whose SOURCE is confirmed dead self-promotes the
        // already-committed live dest (the ordered Demote toward the corpse will never ack). NON-destructive
        // — the directory already committed authority to the dest — so the cheap redrive deadline, no
        // budget. A live-but-slow source only re-drives (record_ack clears the evidence before confirmation).
        SagaState::Demoting { .. } => {
            if liveness.is_confirmed_dead(ctx.source, now) {
                SagaEvent::SourceUnreachable
            } else {
                SagaEvent::Timeout
            }
        }
        // D-37 CELL 2: a POST-commit `Promoting` saga whose committed DEST is confirmed dead FORWARD
        // re-homes onto a live capability-matched target — the dead dest can never `PromoteAck`, and
        // (unlike CELL 1's Demoting source-death) there is no already-committed live owner to self-promote
        // (the dest IS the committed owner). DESTRUCTIVE-budget gated exactly like the BatchHandoff dest-dead
        // ladder (the CSCALE-1 cure: `dead_observed_since` so a recoverable blip that clears in time never
        // re-homes a healthy dest). Once past budget, `select_rehome_target` picks the lowest live shard
        // satisfying the subject's caps (`req` — empty for a P3 bare-point Entity, the D-31 KindDef seam):
        // Some(target) ⇒ `ReHomeTo` (the FSM bumps the fence to `target`); None (no capable live shard —
        // whole-pool death) ⇒ `Timeout`, the saga stays PARKED (honest RED, never a forced incapable re-home).
        //
        // ⚠️ The liveness check is on `subject_owner` — the CURRENT directory owner — NOT `ctx.dest` (the
        // original, now-stale dest). After ONE re-home the directory names the LIVE target, so a later fire
        // sees a LIVE owner ⇒ no re-home ⇒ the fence does NOT run away (the saga then merely re-drives /
        // parks on the D-36 starved watermark, owed). Checking `ctx.dest` would re-home EVERY fire (it stays
        // dead forever), bumping the fence past the journal-gated adopt into a FenceMismatch.
        SagaState::Promoting { .. } => {
            if liveness.is_confirmed_dead(subject_owner, now) {
                let observed = *dead_observed_since.get_or_insert(now);
                if now.0.saturating_sub(observed.0) >= tuning.abort_deadline_ticks {
                    match select_rehome_target(req, roster, liveness, now) {
                        Some(target) => SagaEvent::ReHomeTo { target },
                        None => SagaEvent::Timeout, // no capable live target → stay parked
                    }
                } else {
                    SagaEvent::Timeout // cheap re-drive while the abort budget accrues (dead dest won't ack)
                }
            } else {
                *dead_observed_since = None; // dest healthy/recovered → clear stale budget, re-drive
                SagaEvent::Timeout
            }
        }
        // Every other phase (pre-commit + the forward-only Swapping/Releasing tail + the transient ReHoming
        // CAS-in-flight) → the idempotent Timeout re-drive.
        _ => SagaEvent::Timeout,
    }
}

/// D-37 target selection (Slice 1): the LOWEST live `NodeId` in the roster whose profile SATISFIES the
/// subject's required capabilities — the deterministic forward-re-home target. `BTreeMap` iterates
/// ascending, so the first match IS the lowest (a deterministic tie-break the SEED-replay byte-identical
/// canary depends on). HR3: a `ShardProfile` capability match (`satisfies`), NEVER a match on a shard-kind
/// discriminant. Returns `None` when no capable live shard exists (whole-pool death / zero spare capacity)
/// — the caller then leaves the saga PARKED (honest RED, never a forced re-home to a dead or incapable
/// node). Bitwise `&` (HR5, no short-circuit branch gap): both predicates are evaluated for every
/// candidate (`satisfies` on a dead node is harmless). For a P3 bare-point entity `req` is empty, so the
/// selection degenerates to "the lowest live shard"; the capability match future-proofs ship/voxel realms.
fn select_rehome_target(
    req: &CapRequest,
    roster: &BTreeMap<NodeId, ShardProfile>,
    liveness: &LivenessTracker,
    now: UniverseTick,
) -> Option<NodeId> {
    roster
        .iter()
        .find(|(node, profile)| !liveness.is_confirmed_dead(**node, now) & profile.satisfies(req))
        .map(|(node, _)| *node)
}

/// THE Slice 2a TIMEOUT PRODUCER (the R1 cure): inject `SagaEvent::Timeout` into every live saga whose
/// `now - since` reached its phase deadline, so a lost saga-ack RE-DRIVES (post-commit) or ABORTS
/// (pre-freeze) instead of PARKING forever. Phase-agnostic — ONE scan, ONE injected event; the FSM's
/// existing per-phase Timeout arms do all the work. `since` is REFRESHED on fire (BEFORE delivery), so
/// a still-stale saga re-fires at most once per its deadline window, never every tick (no storm).
/// O(live sagas)/tick — matches the existing `views()`/`active_transfers()` scans; a deadline-ordered
/// structure (pop only the due) is a P3/MMO-scale optimization, not built now.
fn scan_deadlines(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
) {
    let tuning = runtime.tuning;
    // Bound the re-home inputs ONCE outside the loop (BTreeMap is not Copy → shared ref; `req` is the
    // subject's required caps — empty for a P3 bare-point Entity, the D-31 KindDef seam, one home no
    // inline). `&runtime.roster` is a sound disjoint partial borrow alongside `runtime.sagas.iter_mut()`,
    // exactly like `&runtime.liveness` below.
    let roster = &runtime.roster;
    let req = CapRequest::default();
    let mut due: Vec<(TransferId, SagaEvent)> = Vec::new();
    for (transfer, live) in runtime.sagas.iter_mut() {
        if now.0.saturating_sub(live.since.0) >= deadline_for(&live.state, &tuning) {
            live.since = now; // re-arm BEFORE delivery — one fire per window, never a per-tick storm
            // The CURRENT directory owner of the subject (the re-home liveness check keys on THIS, not the
            // stale `ctx.dest`, so a re-home fires AT MOST once — the next fire sees the live target). The
            // `ctx.dest` fallback covers a (transient) subject with no directory record. `dir.head` is a
            // shared reborrow disjoint from `runtime.sagas.iter_mut()` (dir is a separate param).
            let subject_owner = dir
                .head(live.ctx.subject)
                .map_or(live.ctx.dest, |r| r.authority.node());
            // The dead-aware deadline event: a re-drive toward a CONFIRMED-DEAD participant becomes a
            // RESOLUTION instead of looping at a corpse forever. ALL branching lives in the monomorphic
            // `rehome_event_for` helper (see its doc) so this scan stays a branchless dispatch (HR5). The
            // borrows are disjoint: `liveness`/`roster` are fields of `runtime` disjoint from `sagas` (a
            // sound partial borrow), and `state`/`ctx`/`dead_observed_since` are disjoint fields of `live`.
            let event = rehome_event_for(
                &live.state,
                &live.ctx,
                &runtime.liveness,
                &mut live.dead_observed_since,
                &tuning,
                now,
                roster,
                &req,
                subject_owner,
            );
            due.push((*transfer, event));
        }
    }
    for (transfer, event) in due {
        // Count the resolution (orchestrator-side anti-vacuity) BEFORE delivery; the resolution arm is
        // terminal-on-first-fire (the saga tombstones), so each resolved batch counts exactly once.
        match event {
            SagaEvent::SourceUnreachable => runtime.source_unreachable_resolutions += 1,
            SagaEvent::DestUnreachable => runtime.dest_unreachable_resolutions += 1,
            _ => {}
        }
        deliver(runtime, dir, outbox, epoch, now, transfer, event);
    }
}

/// Map a `DropApplied` proof-of-apply ack to the [`SagaState::BatchHandoff`] event its `step_id` proves
/// (D-7d). Total over the only three steps `DropApplied` carries — RELEASE (the SOURCE went
/// `Held→Departing`), DROP (the DEST promoted `Arriving→Held`), else COMPLETE (the SOURCE retired its
/// retained `Departing` copy) — each emitted by exactly ONE producer, so all three arms are reachable
/// and the `else` is the COMPLETE case, NOT an unexpected-step catch-all (HR5). The egress that used to
/// live in the old `handle_drop_applied`/`handle_batch_adopted` is now in the saga's `Emit*` executors
/// (the saga owns the choreography — so a stranded handoff is visible to `scan_deadlines`).
#[must_use]
fn drop_applied_event(step_id: u32) -> SagaEvent {
    if step_id == TRANSIENT_RELEASE_STEP {
        SagaEvent::SourceDropApplied
    } else if step_id == TRANSIENT_DROP_STEP {
        SagaEvent::DestDropApplied
    } else {
        SagaEvent::SourceRetired
    }
}

/// D-3 Slice 4: the CAP AND-gate deciding if a directory record may be REAPED. All THREE must hold (never
/// OR): (1) the post-restart quiesce window has elapsed (a rebuilt orchestrator does not reap until its
/// liveness evidence has had time to re-accrue); (2) the lease has LAPSED (necessary, NOT sufficient — a
/// slow renewal is not a death); (3) the owner is CONFIRMED dead (the evidence-gated discriminator — not a
/// single blip). The combination realizes the binding CAP choice: an orchestrator outage FREEZES recovery
/// (the reaper only runs inside a live orchestrator, and the RAM tracker is empty on rehydrate so gate (3)
/// is unsatisfiable until fresh notices re-accrue) — so an outage NEVER mass-orphans. Three monomorphic
/// `if`s (HR5; no short-circuit `&&`); each corner is covered.
#[must_use]
fn should_reap(
    record: &OwnerRecord,
    now: UniverseTick,
    liveness: &LivenessTracker,
    quiesced_until: UniverseTick,
) -> bool {
    if now.0 < quiesced_until.0 {
        return false;
    }
    if record.lease_expires.0 >= now.0 {
        return false;
    }
    if !liveness.is_confirmed_dead(record.authority.node(), now) {
        return false;
    }
    true
}

/// D-3 Slice 4: the orchestrator expiry REAPER. Once per `reaper_interval_ticks` (re-armed via
/// `last_reap_tick`, never per tick — an O(directory) sweep), revoke every lapsed-AND-confirmed-dead lease.
/// SCOPE (the D-37 boundary): FULLY REVOKE a dead `Session` key (the client reconnects via its ResumeTicket
/// — a well-defined path), but LEAVE a dead `Realm`/`Entity`/`Ship` key for D-37's forward re-home —
/// revoking a player's ship realm into `HeldNowhere` would freeze the ship forever (the durable re-home is
/// owed, ledgered). Collect-then-act (the immutable `entries()` borrow ends before the `revoke` mutation).
/// INERT when `reaper_interval_ticks == 0` (the pre-D-3 default). Runs INSIDE the D-6 group-commit barrier
/// (before the directory reconcile) so a revoke is captured by the same tick's reconcile + commit (durable
/// — a reaped record never resurrects on a kill-9). `revoke` refuses a transfer-locked key (a saga owns it).
fn reap_lapsed_leases(runtime: &mut SagaRuntimeRes, dir: &mut DirectoryCore, now: UniverseTick) {
    let interval = dir.tuning().reaper_interval_ticks;
    if (interval == 0) | (now.0.saturating_sub(runtime.last_reap_tick.0) < interval) {
        return;
    }
    runtime.last_reap_tick = now;
    let quiesced_until = runtime.liveness_quiesced_until;
    // SESSION keys only (Realm/Entity/Ship re-home is D-37): collect the reapable ones, then revoke.
    let reapable: Vec<(DirectoryKey, Fence)> = dir
        .entries()
        .filter(|(key, record)| {
            matches!(key, DirectoryKey::Session(_))
                && should_reap(record, now, &runtime.liveness, quiesced_until)
        })
        .map(|(key, record)| (*key, record.fence))
        .collect();
    for (key, fence) in reapable {
        let _ = dir.revoke(key, fence);
    }
}

/// The orchestrator saga-runtime system: process new triggers, FIRE due deadlines (Slice 2a), then
/// drive every live saga forward on the gateway acks delivered this tick. Runs on the orchestrator's
/// single-threaded schedule; the directory CAS is a direct in-process call (no await, no lock across a send).
pub fn drive_sagas(
    inbox: Res<InboundBox>,
    clock: Res<ClockSample>,
    clock_res: Res<UniverseClockRes>,
    mut dir: ResMut<DirectoryRes>,
    mut runtime: ResMut<SagaRuntimeRes>,
    mut outbox: ResMut<OutboundBox>,
    mut store: ResMut<StoreRes>,
) {
    let now = clock.universe_tick;
    let epoch = clock.epoch;
    process_starts(&mut runtime, &mut dir.0, &mut outbox, epoch, now);
    // Slice 2a: fire due deadlines BEFORE the ack loop — a saga that loses its ack this tick still
    // gets its Timeout re-drive/abort next tick (the producer is the R1 backstop, never a wedge).
    scan_deadlines(&mut runtime, &mut dir.0, &mut outbox, epoch, now);
    for msg in &inbox.0 {
        let (class, bytes) = match msg {
            // D-3 CLEAR-ON-ACK: ANY successful inbound from a peer is proof it is alive — clear its
            // unreachable evidence (done at the TOP, BEFORE the class filter, so a peer that only sends
            // `LeaseRenew`/Directory ops — not saga acks — still un-marks itself; this is why no separate
            // `serve_directory` clear is needed: both systems read the same inbox, this one sees it all).
            Inbound::Wire { from, class, bytes } => {
                runtime.liveness.record_ack(*from);
                (class, bytes)
            }
            // D-3: a delivery failure toward `to` — record one unreachable notice. The evidence-gated
            // tracker confirms `to` dead only after `n_consecutive_unreachable` within the window with no
            // intervening ack (so a recoverable blip never confirms a healthy peer); `scan_deadlines`
            // resolves a `BatchHandoff` whose source/dest is CONFIRMED dead.
            Inbound::NodeUnreachable { to, .. } => {
                runtime.liveness.record_unreachable(*to, now);
                runtime.liveness_notices += 1;
                continue;
            }
        };
        if *class != MsgClass::Saga {
            continue;
        }
        // Two saga-driving inbound arms ride MsgClass::Saga to the orchestrator: the gateway's
        // SagaAck (route-swap phases) and the SOURCE's TransferAck::SourceFlushed (the pose). The
        // DEST's crossing ack is decoded-and-dropped (1d.1 — see below). Directory ops are
        // `serve_directory`'s; Saga commands / FlushSource flow OUT, never in.
        match postcard::from_bytes::<InterShardFlow>(bytes) {
            Ok(InterShardFlow::SagaAck(ack)) => {
                deliver(
                    &mut runtime,
                    &mut dir.0,
                    &mut outbox,
                    epoch,
                    now,
                    ack.transfer(),
                    ack_to_event(ack),
                );
            }
            Ok(InterShardFlow::TransferAck(TransferAck::SourceFlushed {
                transfer_id,
                drained_seq,
                pose,
                ..
            })) => {
                // STASH the pose BEFORE stepping the event, so an EmitCrossing reachable on this
                // same tick (once both freeze + flush have landed) reads it.
                stash_flush(&mut runtime, transfer_id, pose);
                deliver(
                    &mut runtime,
                    &mut dir.0,
                    &mut outbox,
                    epoch,
                    now,
                    transfer_id,
                    SagaEvent::SourceFlushed { drained_seq },
                );
            }
            // D-7d adopt-before-drop PHASE 1: the dest ADOPTED the batch (uncounted `Arriving`) → drive
            // the LIVE saga's `BatchHandoff` tail (`AwaitAdopt→AwaitRelease`, emitting `TransientRelease`
            // to the source). Routing through `deliver` (not the old read-only handler) is what keeps the
            // saga alive across the handoff so `scan_deadlines` can resolve a stranded item (D-7d kills).
            Ok(InterShardFlow::TransferAck(TransferAck::BatchAdopted { transfer_id, .. })) => {
                deliver(
                    &mut runtime,
                    &mut dir.0,
                    &mut outbox,
                    epoch,
                    now,
                    transfer_id,
                    SagaEvent::BatchAdopted,
                );
            }
            // D-7d PHASES 2-4: a `DropApplied` proof-of-apply advances the tail by the phase its step
            // proves (`drop_applied_event`): RELEASE → `AwaitRelease→AwaitPromote` (promote the dest),
            // DROP → `AwaitPromote→AwaitComplete` (release-complete the source), COMPLETE → `Done`.
            Ok(InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id,
                step_id,
            })) => {
                deliver(
                    &mut runtime,
                    &mut dir.0,
                    &mut outbox,
                    epoch,
                    now,
                    transfer_id,
                    drop_applied_event(step_id),
                );
            }
            // The DEST's crossing ack: the dest journal is the exactly-once dedup. Release is NOT
            // gated on this ack — it rides the ordered `PromoteAck` (1d.5b.1) instead. The crossing
            // ack is decoded + dropped here — never a phase transition.
            Ok(InterShardFlow::TransferAck(
                TransferAck::Accepted { .. } | TransferAck::Rejected { .. },
            )) => {}
            // Everything else (Ghost / Directory / Saga commands / DirectoryReply / FlushSource)
            // and any decode failure: not a saga-driving inbound here.
            _ => {}
        }
    }
    // D-3 Slice 4: the expiry REAPER runs at the HEAD of the barrier — BEFORE the directory reconcile below
    // — so a revoke it makes is captured by the SAME tick's delete-all-then-put-current reconcile + the
    // single commit (the revoke is durable this tick; a reaped record never resurrects on a kill-9 — the
    // COMP-2 guarantee). It mutates the directory in RAM; the reconcile then persists the post-reap RAM.
    reap_lapsed_leases(&mut runtime, &mut dir.0, now);
    // D-6 GROUP-COMMIT BARRIER: stage every saga/go-token write recorded this tick (drained from
    // `pending_writes`), snapshot the DIRECTORY (the independent key family) + the durable clock ceiling,
    // then ONE `commit()` — the ~1-fsync/tick durability point (io-prod batches it off-tick later). This
    // runs at the END of the schedule, BEFORE the node's flush phase sends `outbox`, so no effect leaves
    // the orchestrator before the state authorizing it is durable (persist-before-effect). The directory
    // reconcile + the saga writes commit in the SAME barrier ⇒ the durable saga set and directory are
    // always CONSISTENT post-crash (a Swapping snapshot ⟺ dest-owned directory; rehydrate never sees a
    // half-state). The clock ceiling is persisted so a rebuild resumes FORWARD (`CeilingClock::recover`).
    for (key, value) in std::mem::take(&mut runtime.pending_writes) {
        match value {
            Some(bytes) => store.0.put(&key, &bytes),
            None => store.0.delete(&key),
        }
    }
    // RECONCILE the durable Directory family with RAM (audit COMP-2): DELETE every durable row, then PUT
    // every CURRENT one. A put-only snapshot only GROWS — a REVOKED record (a logged-out / departed owner
    // removed from RAM via `LeaseRevoke`, `directory.rs` `revoke`) would otherwise be RESURRECTED at its
    // stale owner+fence by `rehydrate`'s `restore` on the next kill-9 = a zombie authority record
    // (split-brain on the exact recover path D-6 cures). The directory is mutated across TWO systems
    // (serve_directory + drive_sagas), so the reconcile lives HERE — the one place with both the store +
    // the directory — rather than threading a per-remove delete through serve_directory (which has no
    // `pending_writes`). The MemStore staged map is last-write-wins, so a still-present key's
    // delete-then-put nets to the put; a vanished key's lone delete stands. O(directory) at
    // single-orchestrator P3; io-prod does incremental (a per-mutation delete co-located with the WAL).
    for (key_bytes, _) in store.0.scan(&[StoreKey::DIRECTORY]) {
        store.0.delete(&key_bytes);
    }
    for (key, record) in dir.0.entries() {
        let snapshot = DirSnapshot {
            key: *key,
            record: *record,
        };
        store
            .0
            .put(&StoreKey::Directory(*key).bytes(), &encode(&snapshot));
    }
    store.0.put(
        &StoreKey::Clock.bytes(),
        &encode(&(clock_res.0.epoch(), clock_res.0.confirmed_ceiling())),
    );
    store.0.commit();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{NodeConfig, build_app};
    use crate::orchestrator::{OrchestratorConfig, register_orchestrator_with_store};
    use vd_core::entity_kind::{DurabilityClass, EntityKind};
    use vd_core::glam::DVec3;
    use vd_core::pose::{FrameRef, RealmId};
    use vd_core::{EntityId, EpochId, Fence, SessionId, UniverseTick};
    use vd_sim::capability::NodeKind;
    use vd_sim::directory::DirectoryTuning;
    use vd_sim::io::Transport;
    use vd_sim::io::mem::{MemHub, MemStore};
    use vd_sim::saga::BatchHandoffPhase;
    use vd_wire::intershard::TRANSIENT_COMPLETE_STEP;
    use vd_wire::seams::directory::{DirectoryKey, DirectoryOp};
    use vd_wire::seams::transfer_control::{
        PrepareReject, PrepareResult, SpatialReject, TransferControl,
    };

    const FROM_REALM: RealmId = RealmId::System(7);
    const TO_REALM: RealmId = RealmId::System(8);

    /// A non-origin pose the source "flushes" — distinct components so a test can confirm the
    /// EmitCrossing carried THIS pose (not a default).
    fn flushed_pose() -> StampedPose {
        StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 7 },
            DVec3::new(1.0, 2.0, 3.0),
            UniverseTick(5),
        )
    }

    const ORCH: NodeId = NodeId(1);
    const SOURCE: NodeId = NodeId(2);
    const DEST: NodeId = NodeId(3);
    const GATEWAY: NodeId = NodeId(4);
    const XFER: TransferId = TransferId(77);
    const SESSION: SessionId = SessionId(5);

    fn subject_eid() -> EntityId {
        EntityId::pack(EntityKind::Player, 1, 7, 3)
    }

    fn subject() -> DirectoryKey {
        DirectoryKey::Entity(subject_eid())
    }

    /// The wire form of one `TransferControl` command as the gateway receives it from the
    /// orchestrator (full-value equality target).
    fn saga_wire(cmd: TransferControl) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(&InterShardFlow::Saga(cmd))
                .expect("encode")
                .into(),
        }
    }

    /// The wire form of the ordered `Demote` as the SOURCE receives it (1d.5b.1).
    fn demote_wire(new_owner_fence: Fence) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(&InterShardFlow::Demote(vd_wire::intershard::DemoteCmd {
                transfer: XFER,
                subject: subject(),
                new_owner_fence,
                step_id: vd_wire::intershard::DEMOTE_STEP,
            }))
            .expect("encode")
            .into(),
        }
    }

    /// The wire form of the ordered `Promote` as the DEST receives it (1d.5b.1).
    fn promote_wire(new_fence: Fence) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(&InterShardFlow::Promote(
                vd_wire::intershard::PromoteCmd {
                    transfer: XFER,
                    subject: subject(),
                    new_fence,
                    step_id: vd_wire::intershard::PROMOTE_STEP,
                    source: SOURCE,
                },
            ))
            .expect("encode")
            .into(),
        }
    }

    fn ctx(class: vd_core::entity_kind::DurabilityClass, expected_fence: Fence) -> SagaCtx {
        SagaCtx {
            transfer: XFER,
            session: SESSION,
            subject: subject(),
            expected_fence,
            source: SOURCE,
            dest: DEST,
            class,
            needs_provision: false,
            from_realm: FROM_REALM,
            to_realm: TO_REALM,
        }
    }

    /// A TRANSIENT batch saga ctx (D-7): the subject is the dest realm (inert provenance — never
    /// enters the directory, since `locks_directory_key(Transient)=false`); `expected_fence` is the
    /// dest realm-lease fence the batched go-token commits at.
    fn transient_ctx(expected_fence: Fence) -> SagaCtx {
        SagaCtx {
            subject: DirectoryKey::Realm(TO_REALM),
            ..ctx(DurabilityClass::Transient, expected_fence)
        }
    }

    /// The wire form of an orchestrator→shard `InterShardFlow` as the source/dest receive it (D-7b:
    /// the structural drop-before-promote `TransientRelease`/`TransientDrop`/`ReleaseComplete`).
    fn flow_inbound(flow: &InterShardFlow) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(flow).expect("encode").into(),
        }
    }

    /// A `TransientHandoff` egress the orchestrator emits (the target asserts it against `flow_inbound`).
    fn handoff(
        arm: fn(TransientHandoff) -> InterShardFlow,
        step_id: u32,
        fence: Fence,
    ) -> InterShardFlow {
        arm(TransientHandoff {
            transfer: XFER,
            step_id,
            fence,
        })
    }

    /// A stepped orchestrator + a hub the test drives the saga through. The `source`/`dest`
    /// endpoints receive the 1d.1 `FlushSource`/`StubCrossing` egress and let the source inject
    /// its `SourceFlushed` pose reply.
    struct Rig {
        hub: MemHub,
        orch: crate::app::ShardNode<vd_sim::io::mem::MemTransport>,
        gateway: vd_sim::io::mem::MemTransport,
        source: vd_sim::io::mem::MemTransport,
        dest: vd_sim::io::mem::MemTransport,
        /// D-6: the RETAINED durable store — survives a `rebuild` (the kill-9 analog), so the rebuilt
        /// orchestrator re-hydrates its in-flight sagas/directory/clock from it.
        store: MemStore,
    }

    /// The shared orchestrator config (so `new` + `rebuild` build the IDENTICAL orchestrator).
    fn orch_config() -> OrchestratorConfig {
        OrchestratorConfig {
            epoch: EpochId(1),
            reserve_chunk: 1024,
            clock_peers: vec![],
            directory: DirectoryTuning {
                lease_ttl_ticks: 10_000,
                ..DirectoryTuning::default()
            },
            saga: SagaTuning::default(),
            liveness: LivenessTuning::default(),
            roster: BTreeMap::new(),
        }
    }

    impl Rig {
        fn new() -> Rig {
            let hub = MemHub::new();
            let store = MemStore::new();
            let mut orch = build_app(
                NodeConfig {
                    node_id: ORCH,
                    kind: NodeKind::Orchestrator,
                },
                hub.register(ORCH, 64),
            );
            let (world, schedule) = orch.parts_mut();
            // D-6: build against a RETAINED store (empty ⇒ genesis); the Rig keeps the handle so
            // `rebuild` can re-attach the SAME committed WAL to a fresh orchestrator.
            register_orchestrator_with_store(
                world,
                schedule,
                &orch_config(),
                Box::new(store.clone()),
            );
            let gateway = hub.register(GATEWAY, 64);
            let source = hub.register(SOURCE, 64);
            let dest = hub.register(DEST, 64);
            Rig {
                hub,
                orch,
                gateway,
                source,
                dest,
                store,
            }
        }

        /// KILL-9 + REBUILD the orchestrator (D-6): drop the World (its in-memory saga/directory/clock
        /// state is GONE), re-attach ORCH's transport on the same hub (peers reachable, in-flight queues
        /// lost), and build a FRESH orchestrator that RECOVERS from the retained durable store. The
        /// SOURCE/DEST/GATEWAY peers are untouched (only the orchestrator died).
        fn rebuild(&mut self) {
            let transport = self.hub.reregister(ORCH, 64);
            let mut orch = build_app(
                NodeConfig {
                    node_id: ORCH,
                    kind: NodeKind::Orchestrator,
                },
                transport,
            );
            let (world, schedule) = orch.parts_mut();
            register_orchestrator_with_store(
                world,
                schedule,
                &orch_config(),
                Box::new(self.store.clone()),
            );
            self.orch = orch; // the old orchestrator World is dropped here — its RAM is lost
        }

        /// Like [`rebuild`](Self::rebuild) but with a specific D-3 liveness tuning in the recover config —
        /// proves a kill-9-rebuilt orchestrator KEEPS its configured dead-vs-slow margin (not the default).
        fn rebuild_with_liveness(&mut self, liveness: LivenessTuning) {
            let transport = self.hub.reregister(ORCH, 64);
            let mut orch = build_app(
                NodeConfig {
                    node_id: ORCH,
                    kind: NodeKind::Orchestrator,
                },
                transport,
            );
            let (world, schedule) = orch.parts_mut();
            register_orchestrator_with_store(
                world,
                schedule,
                &OrchestratorConfig {
                    liveness,
                    ..orch_config()
                },
                Box::new(self.store.clone()),
            );
            self.orch = orch;
        }

        /// Deliver pending sends to the orchestrator, run one tick, then deliver its outbound
        /// replies/commands to the peer endpoints (pump → step → pump).
        fn settle(&mut self) {
            self.hub.pump();
            let _ = self.orch.step_tick();
            self.hub.pump();
        }

        /// Grant a directory record for the subject at `fence` (so `lock_transfer` + the CAS
        /// have something to act on), owned by the source shard.
        fn grant_subject(&mut self, fence: Fence) {
            self.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::Directory(
                            DirectoryOp::LeaseGrant {
                                key: subject(),
                                owner: AuthorityRef::Shard(SOURCE),
                                fence,
                            },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// The SOURCE ships its pose (the reply to `FlushSource`) — the second half of the
        /// pose-before-promote gate.
        fn flush(&mut self) {
            self.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::TransferAck(
                            TransferAck::SourceFlushed {
                                transfer_id: XFER,
                                step_id: FLUSH_SOURCE_STEP,
                                pose: flushed_pose(),
                                drained_seq: 0,
                            },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// Everything the orchestrator has sent the DEST since the last drain (the `StubCrossing`
        /// crossing egress + the ordered `Promote`).
        fn drain_dest(&mut self) -> Vec<Inbound> {
            self.dest.drain_inbound()
        }

        /// Everything the orchestrator has sent the SOURCE since the last drain (the `FlushSource`
        /// request + the ordered `Demote`).
        fn drain_source(&mut self) -> Vec<Inbound> {
            self.source.drain_inbound()
        }

        fn trigger(&mut self, ctx: SagaCtx) {
            self.orch
                .world_mut()
                .resource_mut::<SagaRuntimeRes>()
                .start_transfer(ctx, GATEWAY);
        }

        /// Deliver one gateway ack to the orchestrator and step it.
        fn ack(&mut self, ack: TransferControlAck) {
            self.gateway
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::SagaAck(ack)).expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// The DEST acks `BatchAdopted` (D-7) — it now holds the batch as `Arriving`. Drives the
        /// orchestrator's D-7b adopt-before-drop `TransientRelease` egress to the source.
        fn batch_adopted(&mut self, transfer: TransferId) {
            self.dest
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::TransferAck(
                            TransferAck::BatchAdopted {
                                transfer_id: transfer,
                                step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
                            },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// A shard's `DropApplied` proof-of-apply (D-7b): `TRANSIENT_RELEASE_STEP` = the source
        /// released (gates the dest promote); `TRANSIENT_DROP_STEP` = the dest promote-confirmed
        /// (drives the source `ReleaseComplete`). The orchestrator ignores the sender, so it rides
        /// `self.source` regardless of which shard it models.
        fn drop_applied(&mut self, step_id: u32) {
            self.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::TransferAck(
                            TransferAck::DropApplied {
                                transfer_id: XFER,
                                step_id,
                            },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// Everything the orchestrator has sent the gateway since the last drain, as raw
        /// `Inbound` (full-value equality, no destructuring — so there are no never-taken
        /// arms to leave uncovered; the codebase convention).
        fn drain_gateway(&mut self) -> Vec<Inbound> {
            self.gateway.drain_inbound()
        }

        fn subject_head(&mut self) -> vd_wire::seams::directory::OwnerRecord {
            self.orch
                .world_mut()
                .resource::<DirectoryRes>()
                .0
                .head(subject())
                .expect("subject recorded")
        }

        fn live(&mut self) -> usize {
            self.orch.world_mut().resource::<SagaRuntimeRes>().live()
        }
    }

    #[test]
    fn durable_happy_path_drives_the_gateway_through_the_direct_commit() {
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle(); // process the start → PrepareSubscribe
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::PrepareSubscribe {
                transfer: XFER,
                session: SESSION,
                dest: DEST,
            })]
        );

        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::RequestCut {
                transfer: XFER,
                session: SESSION,
            })]
        );

        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 42,
        });
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::FreezeSource {
                transfer: XFER,
                session: SESSION,
                marker_seq: 42,
                dest: DEST,
            })]
        );

        // The pose-before-promote gate: SourceFrozen alone leaves the saga in Freezing (no CAS).
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 42,
        });
        assert!(
            rig.drain_gateway().is_empty(),
            "SourceFrozen alone does not commit — the gate awaits the source flush"
        );
        // The SOURCE flushes its pose → BOTH gate conditions met → the DIRECT commit_cas wins
        // in-process → CommitAuthority is sent with the NEW fence, all in ONE tick.
        rig.flush();
        let new_fence = Fence(1).next();
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::CommitAuthority {
                transfer: XFER,
                session: SESSION,
                new_fence,
                subject: subject(),
            })]
        );
        // The entity-STATE crossing rode the SAME commit batch to the DEST, carrying the flushed
        // pose at the new authority fence (the 1d.1 deliverable, asserted end-to-end).
        assert_eq!(
            rig.drain_dest(),
            vec![Inbound::Wire {
                from: ORCH,
                class: MsgClass::Saga,
                bytes: postcard::to_allocvec(&InterShardFlow::Transfer(TransferEnvelope {
                    transfer_id: XFER,
                    universe_epoch: EpochId(1),
                    schema_version: TRANSFER_SCHEMA_VERSION,
                    fence: new_fence,
                    step_id: STUB_CROSSING_STEP,
                    class: DurabilityClass::Durable,
                    payload: TransitionPayload::StubCrossing {
                        entity: subject_eid(),
                        from_realm: FROM_REALM,
                        to_realm: TO_REALM,
                        pose: flushed_pose(),
                        state: vec![],
                    },
                }))
                .expect("encode")
                .into(),
            }],
            "the crossing reached the dest with the flushed pose at the new fence"
        );
        // The directory CAS committed: the dest now owns the subject at the new fence.
        let head = rig.subject_head();
        assert_eq!(head.authority, AuthorityRef::Shard(DEST));
        assert_eq!(head.fence, new_fence);

        // Clear the source buffer of the earlier egress (the grant DirectoryReply + the Freezing
        // FlushSource) so the next drain isolates the ordered Demote.
        let _ = rig.drain_source();

        // Committed (the route swapped) advances to Demoting and PUSHES the ordered Demote to the
        // SOURCE (the fence-enforced Owned→Frozen→Ghost, at the new owner fence) — NOT yet a release.
        rig.ack(TransferControlAck::Committed { transfer: XFER });
        assert_eq!(
            rig.drain_source(),
            vec![demote_wire(new_fence)],
            "Demoting pushes the ordered Demote to the source at the new owner fence"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "no ReleaseSubscribe before the dest is promoted + delivered (demote-before-promote)"
        );
        assert_eq!(
            rig.live(),
            1,
            "still live in Demoting, awaiting the demote-ack"
        );

        // The source acks DemoteAck (proof-of-freeze) → Promoting + the ordered Promote pushed to
        // the DEST. The demote-ack ALONE advances (breaking R1) — release is NOT yet due.
        rig.ack(TransferControlAck::DemoteAck { transfer: XFER });
        assert_eq!(
            rig.drain_dest(),
            vec![promote_wire(new_fence)],
            "the demote-ack advances to Promoting and pushes the Promote to the dest"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "promote pushed; the source sub is still held (seamless overlap)"
        );
        assert_eq!(rig.live(), 1);

        // PromoteAck alone holds; DeliveredToObservers completes the seamless gate → ReleaseSubscribe.
        rig.ack(TransferControlAck::PromoteAck { transfer: XFER });
        assert!(
            rig.drain_gateway().is_empty(),
            "promote-ack alone does not release — the delivery half is still pending"
        );
        rig.ack(TransferControlAck::DeliveredToObservers { transfer: XFER });
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::ReleaseSubscribe {
                transfer: XFER,
                session: SESSION,
                src: SOURCE,
            })],
            "promote-ack AND delivery release the source sub (the seamless gate)"
        );
        assert_eq!(rig.live(), 1);

        // Released → Done → Tombstone: the tail closes, live()→0 with the subject settled at DEST.
        rig.ack(TransferControlAck::Released { transfer: XFER });
        assert_eq!(
            rig.live(),
            0,
            "the ordered demote/promote/release tail reaches Done"
        );
    }

    #[test]
    fn early_delivery_in_demoting_is_carried_into_promoting_and_never_lost() {
        // 1d.5b.1 race fix: the gateway's standing watermark (`DeliveredToObservers`) can arrive
        // while the saga is STILL Demoting — the dest is already delivering from its autonomous
        // adopt-promote, before the source's Demote round-trip completes. The FSM LATCHES it in
        // `Demoting.dest_delivered` and carries it into `Promoting`, so the release gate can never
        // lose it (no park). This drives the full Demoting→Promoting→Releasing→Done tail end-to-end.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        let new_fence = Fence(1).next();
        // Inject a live saga already in Demoting (the post-RouteSwapped entry state).
        {
            let c = ctx(DurabilityClass::Durable, Fence(1));
            let mut runtime = rig.orch.world_mut().resource_mut::<super::SagaRuntimeRes>();
            runtime.sagas.insert(
                XFER,
                LiveSaga {
                    ctx: c,
                    state: SagaState::Demoting {
                        new_fence,
                        dest_delivered: false,
                    },
                    gateway: GATEWAY,
                    since: UniverseTick(0),
                    flushed_pose: None,
                    dead_observed_since: None,
                },
            );
        }
        // Delivery arrives FIRST, while still Demoting: latched, NO Promote yet (awaiting the
        // demote-ack — demote-before-promote), NO release.
        rig.ack(TransferControlAck::DeliveredToObservers { transfer: XFER });
        assert!(
            rig.drain_dest().is_empty(),
            "no Promote while still Demoting — the demote-ack has not landed"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "no release while still Demoting"
        );
        assert_eq!(rig.live(), 1);

        // DemoteAck → Promoting (carrying the latched delivery) + the ordered Promote to the dest.
        rig.ack(TransferControlAck::DemoteAck { transfer: XFER });
        assert_eq!(
            rig.drain_dest(),
            vec![promote_wire(new_fence)],
            "the demote-ack advances to Promoting and pushes the Promote"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "delivery already counted, but the promote-ack is still pending"
        );

        // PromoteAck ALONE now completes the gate (the early delivery was carried forward) →
        // ReleaseSubscribe — proving the watermark was never lost across the Demoting→Promoting edge.
        rig.ack(TransferControlAck::PromoteAck { transfer: XFER });
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::ReleaseSubscribe {
                transfer: XFER,
                session: SESSION,
                src: SOURCE,
            })],
            "the carried delivery + the promote-ack release the source sub"
        );
        rig.ack(TransferControlAck::Released { transfer: XFER });
        assert_eq!(rig.live(), 0, "the ordered tail reached Done");
    }

    #[test]
    fn prepare_rejection_aborts_and_tombstones_with_typed_feedback() {
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle();
        let _ = rig.drain_gateway();

        let reject = PrepareReject::Spatial(SpatialReject::Obstructed);
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Rejected(reject),
        });
        // The dest is torn down; the typed rejection is recorded for the client (Slice 1c).
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::AbortTransfer {
                transfer: XFER,
                session: SESSION,
            })]
        );
        assert_eq!(
            rig.orch.world_mut().resource::<SagaRuntimeRes>().rejected,
            vec![(XFER, AbortReason::PrepareRejected(reject))]
        );

        // DestAborted → terminal Aborted → tombstoned (GC'd). The player stayed on the source.
        rig.ack(TransferControlAck::Aborted { transfer: XFER });
        assert_eq!(rig.live(), 0, "terminal saga is tombstoned");
        // The subject's authority never moved (the CAS never ran): source still owns it.
        let head = rig.subject_head();
        assert_eq!(head.authority, AuthorityRef::Shard(SOURCE));
        // ✅ FLIPPED (Slice 2a, D-1): the terminal Aborted edge emits ClearTransferLock →
        // abort_clear, so the lock CLEARS — the subject can immediately re-transfer (no wedge).
        assert_eq!(
            head.in_transfer, None,
            "Slice 2a: terminal abort clears the directory lock (D-1 closed)"
        );
    }

    #[test]
    fn cas_loss_unwinds_with_a_thaw() {
        // The saga expects Fence(1) but the directory moved to Fence(5) (someone else won) →
        // the DIRECT commit_cas LOSES → abort-with-thaw fires (the frozen source must thaw).
        let mut rig = Rig::new();
        rig.grant_subject(Fence(5));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1))); // stale expectation
        rig.settle();
        let _ = rig.drain_gateway();
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 9,
        });
        let _ = rig.drain_gateway();

        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 9,
        });
        rig.flush(); // gate: both freeze + flush, so the CAS actually runs (and loses)
        // A lost CAS unwinds with the compensator chain: ThawSource then AbortTransfer.
        assert_eq!(
            rig.drain_gateway(),
            vec![
                saga_wire(TransferControl::ThawSource {
                    transfer: XFER,
                    session: SESSION,
                }),
                saga_wire(TransferControl::AbortTransfer {
                    transfer: XFER,
                    session: SESSION,
                }),
            ]
        );
        // The authority never moved (CAS lost); the dest is being torn down + source thawed.
        assert_eq!(rig.subject_head().fence, Fence(5));
        rig.ack(TransferControlAck::SourceThawed { transfer: XFER });
        rig.ack(TransferControlAck::Aborted { transfer: XFER });
        assert_eq!(rig.live(), 0);
        // ✅ FLIPPED (Slice 2a, D-1, audit CPO-2): even the CAS-LOSER's terminal abort clears the
        // lock — `abort_clear` RE-READS the head (this saga's expected fence is stale by definition,
        // so a naive `abort_cas(expected)` would lose too) and clears the lock. FENCE-NEUTRAL: the CAS
        // LOST so authority never moved off the source and no crossing was ever emitted (CasLost never
        // reaches Swapping), so the fence stays put — `directory.fence == source.entity_fence` (FENCE-9).
        let head = rig.subject_head();
        assert_eq!(
            head.in_transfer, None,
            "Slice 2a: the CAS-loser's terminal abort clears the lock via the head re-read (D-1)"
        );
        assert_eq!(
            head.fence,
            Fence(5),
            "abort_clear is fence-neutral: the source still owns at Fence(5), no spurious bump (FENCE-9)"
        );
    }

    #[test]
    fn transient_subject_commits_via_the_go_token_not_a_cas() {
        // HR2 SHORT PATH (D-7): a Transient subject is NOT in the directory (no `grant_subject` —
        // burst isolation), takes NO lock, SKIPS Prepare/Cut/Freeze, and commits the batched go-token at
        // START. It records EXACTLY ONE go-token (G-TIER: one write per batch, never per item), NEVER a
        // per-entity CAS, NEVER a `CommitAuthority`. D-7d: the go-token commit enters the POST-COMMIT
        // `BatchHandoff` TAIL (the saga stays LIVE owning the adopt-before-drop choreography — so a
        // stranded handoff is visible to `scan_deadlines`), tombstoning only at `SourceRetired`.
        let mut rig = Rig::new();
        rig.trigger(transient_ctx(Fence(9)));
        rig.settle(); // process_starts: no lock (Transient) → start → BatchCommitting → go-token → CasWon → BatchHandoff{AwaitAdopt}

        assert_eq!(
            rig.live(),
            1,
            "the short-path transient saga enters the BatchHandoff tail (alive until the choreography completes), NOT tombstoned at commit"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "a transient drives NO gateway route-swap (no PrepareSubscribe, no CommitAuthority)"
        );
        // Exactly ONE go-token, at the dest realm-lease fence (G-TIER: one orchestrator write per batch).
        assert_eq!(
            rig.orch
                .world_mut()
                .resource::<SagaRuntimeRes>()
                .batch_goes(),
            vec![(BatchId(XFER), Fence(9))],
            "one batched go-token recorded at the commit fence"
        );
        // D-7c G-TIER observable: ONE write (the write COUNT, not just the deduped map size).
        assert_eq!(
            rig.orch
                .world_mut()
                .resource::<SagaRuntimeRes>()
                .batch_go_writes(),
            1,
            "one go-token WRITE per batch (the per-item-regression probe)"
        );

        // The D-7b structural drop-before-promote, driven through the orchestrator (two ordered
        // round-trips). PHASE 1 — the DEST's BatchAdopted → the orchestrator tells the SOURCE ONLY to
        // RELEASE (not a broadcast); the dest gets NOTHING yet (gated on the source's DropApplied).
        let _ = (rig.drain_source(), rig.drain_dest());
        rig.batch_adopted(XFER);
        assert_eq!(
            rig.drain_source(),
            vec![flow_inbound(&handoff(
                InterShardFlow::TransientRelease,
                TRANSIENT_RELEASE_STEP,
                Fence(9)
            ))],
            "the source is told to RELEASE (Held→Departing) — adopt-before-drop phase 1"
        );
        assert!(
            rig.drain_dest().is_empty(),
            "the dest is NOT promoted until the source releases (no both-held window)"
        );

        // PHASE 2 — the source's DropApplied(RELEASE) → the orchestrator PROMOTES the DEST only.
        rig.drop_applied(TRANSIENT_RELEASE_STEP);
        assert_eq!(
            rig.drain_dest(),
            vec![flow_inbound(&handoff(
                InterShardFlow::TransientDrop,
                TRANSIENT_DROP_STEP,
                Fence(9)
            ))],
            "the dest is told to PROMOTE (Arriving→Held) only after the source released"
        );
        assert!(
            rig.drain_source().is_empty(),
            "no source egress on the promote step"
        );

        // PHASE 3 — the dest's DropApplied(DROP, promote-confirm) → the orchestrator tells the SOURCE
        // to retire the retained Departing copy (ReleaseComplete).
        rig.drop_applied(TRANSIENT_DROP_STEP);
        assert_eq!(
            rig.drain_source(),
            vec![flow_inbound(&handoff(
                InterShardFlow::ReleaseComplete,
                TRANSIENT_RELEASE_STEP,
                Fence(9)
            ))],
            "the dest's promote-confirm retires the source's Departing copy"
        );
        // The saga is STILL live through the handoff (the D-7d tail) — NOT tombstoned at commit.
        assert_eq!(
            rig.live(),
            1,
            "the saga owns the handoff to its terminal (awaiting the source's retire-complete)"
        );

        // PHASE 4 (D-7d) — the source's COMPLETE ack (`on_release_complete`'s `DropApplied`(COMPLETE))
        // drives `SourceRetired` → the tail tombstones. NO further egress (the choreography is done).
        rig.drop_applied(TRANSIENT_COMPLETE_STEP);
        assert_eq!(
            rig.live(),
            0,
            "the source's retire-complete drives the BatchHandoff tail to Done (tombstoned)"
        );
        // Split (NOT `&&`) so neither short-circuit edge is an uncoverable branch (HR5).
        assert!(
            rig.drain_source().is_empty(),
            "no source egress after the handoff completes"
        );
        assert!(
            rig.drain_dest().is_empty(),
            "no dest egress after the handoff completes"
        );
    }

    #[test]
    fn a_transient_handoff_ack_with_no_go_token_is_a_loud_noop() {
        // D-7d defensive: a `BatchAdopted` OR a `DropApplied` for a batch with NO LIVE saga (a
        // stray/duplicate, or a batch that never started / already tombstoned) emits NOTHING — a no-op,
        // never a silent drop and never a panic. Covers `deliver`'s early-return on an absent saga
        // (`runtime.sagas.get_mut` → `None`), the same idempotency that absorbs a redelivered ack after
        // the BatchHandoff tail tombstoned.
        let mut rig = Rig::new();
        let _ = (rig.drain_source(), rig.drain_dest());
        rig.batch_adopted(XFER); // no prior trigger → no live saga for XFER
        assert!(rig.drain_source().is_empty(), "no saga ⇒ no release");
        assert!(rig.drain_dest().is_empty(), "no saga ⇒ no promote");
        rig.drop_applied(TRANSIENT_RELEASE_STEP); // also no saga → deliver early-returns (None arm)
        assert!(
            rig.drain_source().is_empty(),
            "a DropApplied with no go-token is a loud no-op (no source egress)"
        );
        assert!(
            rig.drain_dest().is_empty(),
            "a DropApplied with no go-token is a loud no-op (no dest egress)"
        );
    }

    #[test]
    fn durable_saga_parked_in_committing_is_curl_visible() {
        // AAA-1: a parked saga is CURL-VISIBLE through the real admin snapshot — its transfer id, its
        // actual stuck phase, and a staleness anchor, never silently wedged. (Drives both
        // `SagaRuntimeRes::views` and `admin_snapshot`'s saga-population end to end.) D-7 moved this
        // OFF the old transient-park vehicle: a DURABLE saga whose direct CAS is starved (the granted
        // fence races ahead so the CAS would lose) is the wrong vehicle; instead we hold a durable
        // saga in `Demoting` awaiting its `DemoteAck` — a genuine post-commit park.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle();
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 3,
        });
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 3,
        });
        rig.flush(); // both gate conditions → the DIRECT CAS wins → Swapping
        rig.ack(TransferControlAck::Committed { transfer: XFER }); // → Demoting (parked, awaiting DemoteAck)
        let _ = (rig.drain_gateway(), rig.drain_source());
        assert_eq!(rig.live(), 1);

        let snap = crate::orchestrator::admin_snapshot(rig.orch.world_mut());
        assert_eq!(snap.sagas.len(), 1);
        assert_eq!(snap.sagas[0].transfer, XFER.to_string());
        // The operator sees the REAL parked phase with its fields (the post-CAS authority fence),
        // not a placeholder. Exact equality (no `assert!`-message format arm — HR5 discipline).
        assert_eq!(
            snap.sagas[0].state,
            "Demoting { new_fence: Fence(2), dest_delivered: false }"
        );
        // `since` was set when the saga last advanced and is bounded by the clock.
        assert!(snap.sagas[0].since.0 <= snap.universe_tick);
    }

    #[test]
    fn orchestrator_rehydrates_an_in_flight_durable_saga_across_a_kill_9() {
        // D-6 HEADLINE: a durable saga in flight (Demoting, POST-commit) SURVIVES an orchestrator kill-9.
        // The rebuilt orchestrator re-hydrates it from the durable WAL (NOT vaporized) and the existing
        // Slice-2a Timeout producer re-drives it FORWARD (re-emits the ordered Demote) — never-vanish
        // holds across a restart. Anti-theater: the `rebuild` drops the in-memory World, so the saga can
        // only survive via the durable Store (a no-persist orchestrator would lose it = `live() == 0`).
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle();
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 3,
        });
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 3,
        });
        rig.flush(); // both gate conditions → the DIRECT CAS wins → Swapping (durable at the barrier)
        rig.ack(TransferControlAck::Committed { transfer: XFER }); // → Demoting (durable at the barrier)
        let _ = (rig.drain_gateway(), rig.drain_source());
        assert_eq!(rig.live(), 1, "the saga is live in Demoting pre-crash");
        let before = crate::orchestrator::admin_snapshot(rig.orch.world_mut());
        // Static message (no `{}` format arg — a lazily-evaluated arg is an uncoverable region when the
        // assert passes, HR5); the `starts_with` IS the always-evaluated condition.
        assert!(
            before.sagas[0].state.starts_with("Demoting"),
            "pre-crash state is Demoting"
        );

        // KILL-9 + REBUILD: the in-memory saga set dies with the World; the rebuilt orchestrator RECOVERS
        // its in-flight saga + the committed directory + the clock from the retained WAL.
        rig.rebuild();
        assert_eq!(
            rig.live(),
            1,
            "the in-flight saga SURVIVED the orchestrator kill-9 (re-hydrated from the WAL, not vanished)"
        );
        let after = crate::orchestrator::admin_snapshot(rig.orch.world_mut());
        assert!(
            after.sagas[0].state.starts_with("Demoting"),
            "re-hydrated at the SAME quiescent phase (Demoting)"
        );
        // The directory survived too: the subject's authority committed to the DEST at the CAS fence.
        let head = rig
            .orch
            .world_mut()
            .resource::<DirectoryRes>()
            .0
            .head(subject())
            .expect("the subject's directory record re-hydrated");
        assert_eq!(head.authority, AuthorityRef::Shard(DEST));
        assert_eq!(
            head.fence,
            Fence(2),
            "the committed CAS fence survived the crash"
        );

        // RE-DRIVE: the recovered saga's `since` is armed, so the first post-restart tick fires
        // `scan_deadlines` → Demoting+Timeout re-emits the ordered Demote to the SOURCE (forward-only,
        // the SAME proven producer — no new recovery path). Proves the transfer makes progress post-crash.
        let _ = rig.drain_source();
        rig.settle();
        assert!(
            rig.drain_source().contains(&demote_wire(Fence(2))),
            "the rebuilt orchestrator re-drove the in-flight transfer forward (re-emitted the Demote)"
        );
    }

    #[test]
    fn orchestrator_rehydrates_a_transient_go_token_across_a_kill_9() {
        // D-6 (transient arm): an in-flight transient batch's go-token + its `BatchHandoff` saga survive
        // an orchestrator kill-9. The WAL carries the go-token as ONE record per batch, so rehydrate
        // restores `batch_go_writes` from the map LEN — the G-TIER decouple is preserved on restart, NOT
        // re-driven through the `+= 1` (a regression that re-incremented would read N here, not 1).
        let mut rig = Rig::new();
        rig.trigger(transient_ctx(Fence(9)));
        rig.settle(); // BatchCommitting → the go-token commits (persisted) → BatchHandoff{AwaitAdopt}
        {
            let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
            assert_eq!(
                runtime.live(),
                1,
                "the transient saga is live in the BatchHandoff tail pre-crash"
            );
            assert_eq!(runtime.batch_goes(), vec![(BatchId(XFER), Fence(9))]);
            assert_eq!(runtime.batch_go_writes(), 1);
        }

        rig.rebuild();
        let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
        assert_eq!(
            runtime.live(),
            1,
            "the in-flight transient saga SURVIVED the kill-9 (re-hydrated from the WAL)"
        );
        assert_eq!(
            runtime.batch_goes(),
            vec![(BatchId(XFER), Fence(9))],
            "the committed go-token re-hydrated (the dest's Held authority stays backed)"
        );
        assert_eq!(
            runtime.batch_go_writes(),
            1,
            "batch_go_writes restored from the map len, NOT re-incremented through the +=1 (G-TIER decouple)"
        );
    }

    #[test]
    fn rehydrate_does_not_resurrect_a_revoked_directory_record() {
        // D-6 (audit COMP-2): a directory record REVOKED before a kill-9 (a logged-out / departed owner —
        // the gateway-Bye / stub-departing `LeaseRevoke` path) must NOT be resurrected by rehydrate. The
        // reconcile barrier deletes the durable row; a put-only snapshot would re-install a stale-authority
        // zombie on recover. This exercises the barrier's durable-delete loop + the no-resurrect guarantee.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1)); // SOURCE owns the subject at Fence(1) — durably persisted
        // REVOKE it at its exact fence (the logout / departed path), then settle so the barrier reconciles.
        rig.source
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
                        key: subject(),
                        fence: Fence(1),
                    }))
                    .expect("encode"),
                ),
            )
            .expect("revoke sent");
        rig.settle();
        assert!(
            rig.orch
                .world_mut()
                .resource::<DirectoryRes>()
                .0
                .head(subject())
                .is_none(),
            "the record is gone from RAM after the revoke"
        );

        // KILL-9 + REBUILD: the revoked record must STAY gone (not resurrected by rehydrate's restore).
        rig.rebuild();
        assert!(
            rig.orch
                .world_mut()
                .resource::<DirectoryRes>()
                .0
                .head(subject())
                .is_none(),
            "the revoked record is NOT resurrected by rehydrate (COMP-2: the barrier durably deleted it)"
        );
    }

    #[test]
    fn rehydrate_keeps_the_configured_liveness_margin() {
        // D-3 (re-audit wf_4d2ca7ae): a kill-9-rebuilt orchestrator must KEEP its prod dead-vs-slow margin
        // (n = 3), NOT drop to the kill-equivalent n = 1 default — `rehydrate` threads `cfg.liveness` into
        // the rebuilt runtime. (The tracker is still rebuilt EMPTY — the CAP freeze — but its TUNING survives.)
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1)); // persist a directory record + the clock ceiling to the store
        rig.settle();
        rig.rebuild_with_liveness(LivenessTuning {
            n_consecutive_unreachable: 3,
            unreachable_window_ticks: 64,
            retry_delay_ticks_hint: 2,
        });
        // The recovered runtime confirms a peer dead only after 3 notices (the n = 1 default would confirm
        // on the first) — proving the configured margin survived the kill-9 recover.
        let mut runtime = rig.orch.world_mut().resource_mut::<SagaRuntimeRes>();
        runtime
            .liveness
            .record_unreachable(SOURCE, UniverseTick(1));
        assert!(
            !runtime.liveness.is_confirmed_dead(SOURCE, UniverseTick(1)),
            "the recovered orchestrator kept its configured n = 3 margin, not the n = 1 default"
        );
    }

    // ---- D-3 Slice 4: the expiry reaper + CAP gate -------------------------------------------------

    fn rec(node: NodeId, lease_expires: u64) -> vd_wire::seams::directory::OwnerRecord {
        vd_wire::seams::directory::OwnerRecord {
            authority: AuthorityRef::Shard(node),
            fence: Fence(1),
            lease_expires: UniverseTick(lease_expires),
            in_transfer: None,
        }
    }

    #[test]
    fn should_reap_requires_quiesced_lapsed_and_confirmed_dead() {
        // The CAP AND-gate: reap iff quiesce-elapsed AND lapsed AND confirmed-dead. Covers all 4 corners.
        let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1
        let dead = NodeId(5);
        liveness.record_unreachable(dead, UniverseTick(100)); // confirmed (n = 1)
        let now = UniverseTick(100);
        let quiesced = UniverseTick(50); // window elapsed (now >= quiesced)
        // All three hold → reap (lapsed at 90 < now 100; dead confirmed; quiesce elapsed).
        assert!(should_reap(&rec(dead, 90), now, &liveness, quiesced));
        // (1) NOT past the quiesce freeze → no reap.
        assert!(!should_reap(&rec(dead, 90), now, &liveness, UniverseTick(150)));
        // (2) NOT lapsed (lease_expires >= now) → no reap.
        assert!(!should_reap(&rec(dead, 200), now, &liveness, quiesced));
        // (3) NOT confirmed dead (a node with no unreachable evidence) → no reap.
        assert!(!should_reap(&rec(NodeId(99), 90), now, &liveness, quiesced));
    }

    #[test]
    fn reaper_revokes_a_lapsed_confirmed_dead_session_only() {
        // The reaper FULLY revokes a dead Session key but LEAVES a dead Realm key (D-37 re-homes it, not
        // the reaper — revoking a realm into HeldNowhere would freeze it). A lapsed-but-NOT-confirmed
        // session is left (CAP). reaper_interval active; renew INERT (the reaper does not need the heartbeat).
        let dir_tuning = DirectoryTuning {
            lease_ttl_ticks: 10,
            reaper_interval_ticks: 8,
            ..DirectoryTuning::default()
        };
        let mut dir = DirectoryCore::new(dir_tuning);
        let dead = NodeId(5);
        let live = NodeId(6);
        // Grant at tick 0 ⇒ lease_expires = 10 (lapsed by now = 100).
        let _ = dir.grant(
            DirectoryKey::Session(SessionId(1)),
            AuthorityRef::Gateway(dead),
            Fence(1),
            UniverseTick(0),
        );
        let _ = dir.grant(
            DirectoryKey::Session(SessionId(2)),
            AuthorityRef::Gateway(live),
            Fence(1),
            UniverseTick(0),
        );
        let _ = dir.grant(
            DirectoryKey::Realm(RealmId::System(7)),
            AuthorityRef::Shard(dead),
            Fence(1),
            UniverseTick(0),
        );
        // A lapsed, confirmed-dead session that is MID-TRANSFER (in_transfer-locked): the reaper TRIES
        // to revoke it (should_reap is true) but `revoke` REFUSES a transfer-locked key — the saga owns
        // it, and the saga's own recovery (scan_deadlines) handles the dead participant, not the reaper.
        let _ = dir.grant(
            DirectoryKey::Session(SessionId(3)),
            AuthorityRef::Gateway(dead),
            Fence(1),
            UniverseTick(0),
        );
        assert!(dir.lock_transfer(DirectoryKey::Session(SessionId(3)), TransferId(7)));
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // n = 1
        runtime.liveness.record_unreachable(dead, UniverseTick(50)); // only `dead` is confirmed
        reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(100));
        assert!(
            dir.head(DirectoryKey::Session(SessionId(1))).is_none(),
            "the lapsed, confirmed-dead session is reaped"
        );
        assert!(
            dir.head(DirectoryKey::Session(SessionId(2))).is_some(),
            "a lapsed but NOT-confirmed-dead session is left (CAP — only confirmed deaths reap)"
        );
        assert!(
            dir.head(DirectoryKey::Realm(RealmId::System(7))).is_some(),
            "a dead Realm is LEFT for D-37 forward re-home, never reaped into HeldNowhere"
        );
        assert!(
            dir.head(DirectoryKey::Session(SessionId(3))).is_some(),
            "an in_transfer-locked dead session is NOT reaped — revoke refuses it (the saga owns the key)"
        );
    }

    #[test]
    fn reaper_is_inert_at_zero_interval_and_respects_its_cadence() {
        let dead = NodeId(5);
        // INERT: reaper_interval == 0 never reaps, even a lapsed + confirmed-dead session.
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10,
            ..DirectoryTuning::default() // reaper_interval_ticks = 0
        });
        let _ = dir.grant(
            DirectoryKey::Session(SessionId(1)),
            AuthorityRef::Gateway(dead),
            Fence(1),
            UniverseTick(0),
        );
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        runtime.liveness.record_unreachable(dead, UniverseTick(50));
        reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(10_000));
        assert!(
            dir.head(DirectoryKey::Session(SessionId(1))).is_some(),
            "an inert reaper (interval 0) never reaps"
        );

        // NOT DUE: interval > 0 but the elapsed since the last sweep is below it → no reap this tick.
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10,
            reaper_interval_ticks: 8,
            ..DirectoryTuning::default()
        });
        let _ = dir.grant(
            DirectoryKey::Session(SessionId(1)),
            AuthorityRef::Gateway(dead),
            Fence(1),
            UniverseTick(0),
        );
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        runtime.liveness.record_unreachable(dead, UniverseTick(50));
        runtime.last_reap_tick = UniverseTick(100); // just swept at 100
        reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(103)); // 103 - 100 = 3 < 8 → not due
        assert!(
            dir.head(DirectoryKey::Session(SessionId(1))).is_some(),
            "a sweep below the reaper interval since the last is skipped"
        );
    }

    #[test]
    fn a_re_trigger_of_a_live_transient_saga_does_not_clobber_it() {
        // Audit D6-1: a DUPLICATE trigger for an in-flight transient `BatchId` is a no-op — the live
        // `BatchHandoff` saga + its committed go-token are untouched (one key → one saga, never reset).
        // The durable path is already guarded by `lock_transfer`; this covers the transient path (no lock).
        let mut rig = Rig::new();
        rig.trigger(transient_ctx(Fence(9)));
        rig.settle(); // → BatchHandoff{AwaitAdopt}, go-token committed (batch_go_writes == 1)
        rig.trigger(transient_ctx(Fence(9))); // a spurious/duplicate producer re-trigger
        rig.settle();
        let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
        assert_eq!(
            runtime.live(),
            1,
            "the re-trigger did NOT start a second saga (guarded)"
        );
        assert_eq!(
            runtime.batch_go_writes(),
            1,
            "the live saga was NOT clobbered + its go-token NOT re-committed"
        );
    }

    #[test]
    fn a_stray_ack_to_a_parked_saga_preserves_its_staleness_anchor() {
        // AAA-1-SINCE regression guard: a PARKED saga that receives a duplicate / stray ack the FSM
        // absorbs as same-state must KEEP its `since` — an unconditional bump would reset stuck-saga
        // staleness on every at-least-once redelivery. (Exercises the same-state arm of the
        // `final_state != prior` gate in `commit_result`.) D-7 moved this OFF the old transient
        // CommittingCas park (transients no longer park) onto a DURABLE saga held in `Demoting`
        // awaiting its `DemoteAck`: a stray duplicate `Committed` (route-swap) ack is absorbed there
        // as same-state (`RouteSwapped` is not handled in `Demoting`), the genuine post-commit park.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle();
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 3,
        });
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 3,
        });
        rig.flush(); // both gate conditions → the DIRECT CAS wins → Swapping
        rig.ack(TransferControlAck::Committed { transfer: XFER }); // → Demoting (parked, awaiting DemoteAck)
        let _ = (rig.drain_gateway(), rig.drain_source());
        let since_parked = crate::orchestrator::admin_snapshot(rig.orch.world_mut()).sagas[0].since;

        // Two duplicate Committed acks (at-least-once redelivery of the route-swap ack). The FSM
        // absorbs each as same-state in Demoting (RouteSwapped is not handled there); the clock
        // advances on every step. (Well within the redrive deadline, so the producer does not fire.)
        rig.ack(TransferControlAck::Committed { transfer: XFER });
        rig.ack(TransferControlAck::Committed { transfer: XFER });

        let snap = crate::orchestrator::admin_snapshot(rig.orch.world_mut());
        assert_eq!(
            snap.sagas.len(),
            1,
            "still parked in Demoting, not advanced by stray acks"
        );
        assert_eq!(
            snap.sagas[0].state, "Demoting { new_fence: Fence(2), dest_delivered: false }",
            "state unchanged by the stray acks"
        );
        assert_eq!(
            snap.sagas[0].since.0, since_parked.0,
            "the staleness anchor is NOT reset by a stray same-state ack"
        );
        assert!(
            snap.universe_tick > since_parked.0,
            "the clock DID advance — proving `since` was held, not re-stamped to `now`"
        );
    }

    #[test]
    fn active_transfers_reports_the_live_triple_and_is_empty_when_idle() {
        // 1d.5b.3d: the typed mid-flight ground truth the AUTHORITY-UNIQUE oracle excuses against —
        // each live saga's (subject, source, dest); empty once tombstoned (post-quiesce ⇒ strict).
        let mut rt = SagaRuntimeRes::default();
        assert!(rt.active_transfers().is_empty());
        rt.sagas.insert(
            XFER,
            LiveSaga {
                ctx: ctx(vd_core::entity_kind::DurabilityClass::Durable, Fence(1)),
                state: SagaState::Demoting {
                    new_fence: Fence(2),
                    dest_delivered: false,
                },
                gateway: GATEWAY,
                since: UniverseTick(0),
                flushed_pose: None,
                dead_observed_since: None,
            },
        );
        assert_eq!(
            rt.active_transfers(),
            vec![ActiveTransfer {
                subject: subject(),
                source: SOURCE,
                dest: DEST,
            }],
        );
    }

    #[test]
    fn deadline_for_maps_every_phase_to_its_risk_class() {
        // Slice 2a: pre-freeze/freeze/aborting → the LARGE destructive abort deadline; committing/
        // post-commit → the cheap redrive deadline; terminal → u64::MAX (never fires).
        let t = SagaTuning {
            redrive_deadline_ticks: 8,
            abort_deadline_ticks: 24,
        };
        for s in [
            SagaState::AwaitProvision,
            SagaState::Preparing,
            SagaState::Cutting,
            SagaState::Freezing {
                marker_seq: 0,
                frozen_drained: None,
                flushed: false,
            },
            SagaState::Aborting {
                reason: AbortReason::CutTimeout,
                awaiting_thaw: true,
                awaiting_abort_ack: true,
            },
        ] {
            assert_eq!(deadline_for(&s, &t), 24, "{s:?} uses the abort deadline");
        }
        for s in [
            SagaState::CommittingCas {
                marker_seq: 0,
                drained_seq: 0,
            },
            SagaState::BatchCommitting { step_id: 11 },
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitPromote,
                new_fence: Fence(2),
            },
            SagaState::Swapping {
                new_fence: Fence(2),
            },
            SagaState::Demoting {
                new_fence: Fence(2),
                dest_delivered: false,
            },
            SagaState::Promoting {
                new_fence: Fence(2),
                promote_acked: false,
                dest_delivered: false,
                rehome_target: None,
            },
            SagaState::ReHoming {
                target: NodeId(4),
                prev_fence: Fence(2),
            },
            SagaState::Releasing {
                new_fence: Fence(2),
            },
        ] {
            assert_eq!(deadline_for(&s, &t), 8, "{s:?} uses the redrive deadline");
        }
        assert_eq!(
            deadline_for(
                &SagaState::Done {
                    new_fence: Fence(2)
                },
                &t
            ),
            u64::MAX
        );
        assert_eq!(
            deadline_for(
                &SagaState::Aborted {
                    reason: AbortReason::CutTimeout
                },
                &t
            ),
            u64::MAX
        );
    }

    /// Inject a live saga directly (controlled state + `since`) for the producer tests.
    fn inject_saga(runtime: &mut SagaRuntimeRes, state: SagaState, since: UniverseTick) {
        runtime.sagas.insert(
            XFER,
            LiveSaga {
                ctx: ctx(DurabilityClass::Durable, Fence(1)),
                state,
                gateway: GATEWAY,
                since,
                flushed_pose: None,
                dead_observed_since: None,
            },
        );
    }

    fn flows_to_node(outbox: &OutboundBox, node: NodeId) -> Vec<InterShardFlow> {
        outbox
            .0
            .iter()
            .filter(|(to, _, _)| *to == node)
            .filter_map(|(_, _, b)| postcard::from_bytes(b).ok())
            .collect()
    }

    #[test]
    fn scan_deadlines_re_drives_a_due_saga_re_arms_it_and_skips_a_fresh_one() {
        // THE R1 cure: a post-commit (Demoting) saga whose ack was lost re-drives at the redrive
        // deadline (8); the re-arm (since←now) means it fires at most ONCE per window, never every tick.
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive=8, abort=24
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        });
        inject_saga(
            &mut runtime,
            SagaState::Demoting {
                new_fence: Fence(2),
                dest_delivered: false,
            },
            UniverseTick(0),
        );

        // BELOW the deadline (now=7 < 8): nothing re-driven.
        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(7),
        );
        assert!(outbox.0.is_empty(), "below the deadline: no re-drive");

        // AT the deadline (now=8): Timeout → Demoting re-emits the Demote to the SOURCE.
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8),
        );
        let expected = InterShardFlow::Demote(DemoteCmd {
            transfer: XFER,
            subject: subject(),
            new_owner_fence: Fence(2),
            step_id: DEMOTE_STEP,
        });
        assert!(
            flows_to_node(&outbox, SOURCE).contains(&expected),
            "the deadline producer re-drove the Demote: {:?}",
            outbox.0
        );

        // RE-ARM: `since` refreshed to 8, so the next scan within the window (now=9, 9-8=1 < 8) is silent.
        let mut outbox2 = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox2,
            EpochId(1),
            UniverseTick(9),
        );
        assert!(
            outbox2.0.is_empty(),
            "within the re-armed window: no second re-drive (no per-tick storm)"
        );
    }

    #[test]
    fn scan_deadlines_re_drives_a_parked_aborting_saga() {
        // finding #6: a saga PARKED in Aborting (a dropped compensator ack) is re-driven by the
        // producer — at the LARGE abort deadline (24) — re-emitting BOTH outstanding compensators.
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        });
        inject_saga(
            &mut runtime,
            SagaState::Aborting {
                reason: AbortReason::FreezeTimeout,
                awaiting_thaw: true,
                awaiting_abort_ack: true,
            },
            UniverseTick(0),
        );

        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(23),
        );
        assert!(outbox.0.is_empty(), "below the abort deadline: no re-drive");

        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(24),
        );
        let to_gateway = flows_to_node(&outbox, GATEWAY);
        assert!(
            to_gateway.contains(&InterShardFlow::Saga(TransferControl::ThawSource {
                transfer: XFER,
                session: SESSION,
            })),
            "the producer re-drove ThawSource: {to_gateway:?}"
        );
        assert!(
            to_gateway.contains(&InterShardFlow::Saga(TransferControl::AbortTransfer {
                transfer: XFER,
                session: SESSION,
            })),
            "the producer re-drove AbortTransfer: {to_gateway:?}"
        );
    }

    #[test]
    fn scan_deadlines_self_promotes_a_demoting_saga_with_a_confirmed_dead_source() {
        // D-37 CELL 1: a DUE post-commit `Demoting` saga whose SOURCE is confirmed-dead self-promotes the
        // already-committed live dest — re-driving the ordered Demote toward the corpse would PARK forever.
        // The producer injects SourceUnreachable (non-destructive, the cheap redrive deadline; n==1 confirms
        // on the first NodeUnreachable), the FSM transitions Demoting→Promoting and emits the Promote to the
        // DEST (NOT a Demote toward the dead source). The saga stays LIVE — Promoting awaits PromoteAck +
        // DestDelivered (the Releasing gate, never Done-on-promote: the D-36 starved-watermark caveat holds).
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive=8
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        });
        inject_saga(
            &mut runtime,
            SagaState::Demoting {
                new_fence: Fence(2),
                dest_delivered: false,
            },
            UniverseTick(0),
        );
        runtime.liveness.record_unreachable(SOURCE, UniverseTick(8)); // confirmed dead (n == 1)
        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8),
        );
        assert!(
            flows_to_node(&outbox, DEST).contains(&InterShardFlow::Promote(PromoteCmd {
                transfer: XFER,
                subject: subject(),
                new_fence: Fence(2),
                step_id: PROMOTE_STEP,
                source: SOURCE,
            })),
            "the dead-source Demoting saga self-promotes the committed dest: {:?}",
            outbox.0
        );
        assert!(
            flows_to_node(&outbox, SOURCE).is_empty(),
            "no Demote is re-driven toward the dead source"
        );
        assert_eq!(
            runtime.live(),
            1,
            "still LIVE — Promoting awaits PromoteAck + DestDelivered (never Done-on-promote, D-36)"
        );
        assert_eq!(runtime.source_unreachable_resolutions(), 1);
    }

    #[test]
    fn select_rehome_target_picks_the_lowest_live_capable_shard() {
        use std::collections::BTreeMap;
        use vd_sim::capability::{CapRequest, ShardProfile, VoxelGeometry};
        let empty = ShardProfile::build(CapRequest::default()).expect("empty profile");
        let cartesian = ShardProfile::build(CapRequest {
            voxel: Some(VoxelGeometry::Cartesian),
            ..CapRequest::default()
        })
        .expect("cartesian profile");
        // N2 dead + capable, N3 live but INCAPABLE (empty, no voxel), N4 + N5 live + capable.
        let roster: BTreeMap<NodeId, ShardProfile> = [
            (NodeId(2), cartesian),
            (NodeId(3), empty),
            (NodeId(4), cartesian),
            (NodeId(5), cartesian),
        ]
        .into_iter()
        .collect();
        let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1
        liveness.record_unreachable(NodeId(2), UniverseTick(5)); // N2 confirmed dead
        let req = CapRequest {
            voxel: Some(VoxelGeometry::Cartesian),
            ..CapRequest::default()
        };
        // N2 dead (skip), N3 incapable (skip), N4 live + capable = the LOWEST live capable (< N5).
        assert_eq!(
            select_rehome_target(&req, &roster, &liveness, UniverseTick(5)),
            Some(NodeId(4)),
            "the lowest LIVE, CAPABLE shard is chosen (dead + incapable skipped, deterministic order)"
        );
    }

    #[test]
    fn select_rehome_target_returns_none_when_no_capable_live_shard_and_serves_an_empty_req() {
        use std::collections::BTreeMap;
        use vd_sim::capability::{CapRequest, ShardProfile, VoxelGeometry};
        let empty = ShardProfile::build(CapRequest::default()).expect("empty profile");
        let liveness = LivenessTracker::new(LivenessTuning::default());
        // Empty roster ⇒ None (no target — the caller leaves the saga PARKED, honest RED).
        let none_roster: BTreeMap<NodeId, ShardProfile> = BTreeMap::new();
        assert_eq!(
            select_rehome_target(&CapRequest::default(), &none_roster, &liveness, UniverseTick(0)),
            None
        );
        // A roster of only INCAPABLE shards for a voxel req ⇒ None (never a forced incapable re-home).
        let only_stub: BTreeMap<NodeId, ShardProfile> = [(NodeId(3), empty)].into_iter().collect();
        let voxel_req = CapRequest {
            voxel: Some(VoxelGeometry::Cartesian),
            ..CapRequest::default()
        };
        assert_eq!(
            select_rehome_target(&voxel_req, &only_stub, &liveness, UniverseTick(0)),
            None,
            "no shard satisfies a voxel req ⇒ None"
        );
        // But for the P3 EMPTY (bare-point) req, that same live stub IS the target.
        assert_eq!(
            select_rehome_target(&CapRequest::default(), &only_stub, &liveness, UniverseTick(0)),
            Some(NodeId(3)),
            "an empty bare-point req is satisfied by a live stub shard"
        );
    }

    #[test]
    fn rehome_event_for_promoting_dead_dest_rehomes_past_budget_else_redrives() {
        // D-37 CELL 2 producer — the 4 corners of the Promoting-dead-dest branch (HR5: all covered here in
        // the monomorphic helper). ctx: source=SOURCE(2), dest=DEST(3). abort_deadline=24 (SagaTuning::default).
        let tuning = SagaTuning::default();
        let c = ctx(DurabilityClass::Durable, Fence(1));
        let promoting = SagaState::Promoting {
            new_fence: Fence(2),
            promote_acked: false,
            dest_delivered: false,
            rehome_target: None,
        };
        let empty = ShardProfile::build(CapRequest::default()).expect("empty profile");
        let roster: BTreeMap<NodeId, ShardProfile> = [(NodeId(9), empty)].into_iter().collect();
        let req = CapRequest::default();
        let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1

        // (i) dest HEALTHY (not confirmed dead) → Timeout, and a stale budget is cleared.
        let mut dos = Some(UniverseTick(5));
        let ev = rehome_event_for(&promoting, &c, &liveness, &mut dos, &tuning, UniverseTick(30), &roster, &req, DEST);
        assert_eq!(ev, SagaEvent::Timeout);
        assert_eq!(dos, None, "a healthy dest clears the stale abort budget");

        // Confirm DEST dead (n == 1 → one notice confirms).
        liveness.record_unreachable(DEST, UniverseTick(0));
        // (ii) dest DEAD but WITHIN the abort budget → cheap Timeout re-drive (budget anchored at first fire).
        let mut dos = None;
        let ev = rehome_event_for(&promoting, &c, &liveness, &mut dos, &tuning, UniverseTick(0), &roster, &req, DEST);
        assert_eq!(ev, SagaEvent::Timeout);
        assert_eq!(dos, Some(UniverseTick(0)), "the abort budget anchors on the first dead observation");
        let ev = rehome_event_for(&promoting, &c, &liveness, &mut dos, &tuning, UniverseTick(10), &roster, &req, DEST);
        assert_eq!(ev, SagaEvent::Timeout, "still within the 24-tick budget at tick 10");

        // (iii) dest DEAD, PAST budget, a capable LIVE target exists → ReHomeTo{target}.
        let ev = rehome_event_for(&promoting, &c, &liveness, &mut dos, &tuning, UniverseTick(24), &roster, &req, DEST);
        assert_eq!(ev, SagaEvent::ReHomeTo { target: NodeId(9) }, "past budget → forward re-home to the live target");

        // (iv) dest DEAD, PAST budget, NO capable live target (empty roster) → Timeout (stay PARKED, honest).
        let empty_roster: BTreeMap<NodeId, ShardProfile> = BTreeMap::new();
        let mut dos = Some(UniverseTick(0));
        let ev = rehome_event_for(&promoting, &c, &liveness, &mut dos, &tuning, UniverseTick(24), &empty_roster, &req, DEST);
        assert_eq!(ev, SagaEvent::Timeout, "no capable live target → the saga stays parked (honest)");
    }

    #[test]
    fn deliver_rehome_to_commits_to_the_target_and_emits_the_dedicated_adopt() {
        // D-37 CELL 2 executor: a Promoting saga whose committed dest is now dead, delivered ReHomeTo
        // {target}, re-homes onto the live target. ReHomeCommit re-points commit_cas to the target (the
        // fence-monotone bump 1→2 strictly stales the dead dest AND names the target BEFORE the adopt);
        // CasWon → Promoting + ReHomeAdopt emits the DEDICATED ReHome envelope from the stashed pose.
        // Covers the ReHomeCommit + ReHomeAdopt executor arms + emit_rehome's Some arm end-to-end.
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        });
        // The subject is committed to the (now-dead) DEST at Fence(1) — the re-home CAS expectation.
        let _ = dir.grant(subject(), AuthorityRef::Shard(DEST), Fence(1), UniverseTick(0));
        inject_saga(
            &mut runtime,
            SagaState::Promoting {
                new_fence: Fence(1),
                promote_acked: false,
                dest_delivered: false,
                rehome_target: None,
            },
            UniverseTick(0),
        );
        stash_flush(&mut runtime, XFER, flushed_pose()); // the adopt payload
        let target = NodeId(9);
        let mut outbox = OutboundBox::default();
        deliver(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(5),
            XFER,
            SagaEvent::ReHomeTo { target },
        );
        // The directory now names the live TARGET at the bumped fence (fence-monotone 1 → 2).
        let head = dir.head(subject()).expect("subject still recorded");
        assert_eq!(
            head.authority,
            AuthorityRef::Shard(target),
            "re-home committed authority to the live target"
        );
        assert_eq!(
            head.fence,
            Fence(2),
            "fence-monotone bump (1→2) strictly stales the dead dest"
        );
        // The DEDICATED ReHome adopt (NOT a Promote) was emitted to the target from the stashed pose.
        let expected = InterShardFlow::ReHome(ReHomeCmd {
            transfer: XFER,
            subject: subject(),
            new_fence: Fence(2),
            step_id: RE_HOME_STEP,
            state: ReHomeState::PoseOnly(flushed_pose()),
            source: SOURCE,
        });
        assert!(
            flows_to_node(&outbox, target).contains(&expected),
            "the dedicated ReHome adopt was emitted to the target: {:?}",
            outbox.0
        );
    }

    #[test]
    fn rehome_adopt_is_a_loud_no_op_for_a_non_entity_subject_or_missing_pose() {
        // build_rehome / emit_rehome gate on an Entity subject AND a stashed pose (mirrors build/emit_
        // crossing) so the executor ReHomeAdopt arm stays branchless (HR5). A Realm subject (no
        // transfer_subject_entity) ⇒ None; an Entity subject with NO pose ⇒ None; both ⇒ Some.
        let realm_ctx = SagaCtx {
            subject: DirectoryKey::Realm(RealmId::System(9)),
            ..ctx(DurabilityClass::Durable, Fence(1))
        };
        assert!(
            build_rehome(&realm_ctx, Fence(2), Some(flushed_pose())).is_none(),
            "non-Entity subject ⇒ None"
        );
        let entity_ctx = ctx(DurabilityClass::Durable, Fence(1)); // subject() is an Entity
        assert!(
            build_rehome(&entity_ctx, Fence(2), None).is_none(),
            "missing flushed pose ⇒ None"
        );
        assert!(
            build_rehome(&entity_ctx, Fence(2), Some(flushed_pose())).is_some(),
            "Entity subject + a stashed pose ⇒ Some"
        );
        // emit_rehome's None arm: a non-Entity re-home adopt emits NOTHING (the LOUD no-op).
        let mut outbox = OutboundBox::default();
        emit_rehome(&realm_ctx, Fence(2), NodeId(9), Some(flushed_pose()), &mut outbox);
        assert!(
            outbox.0.is_empty(),
            "a non-Entity re-home adopt emits no envelope"
        );
    }

    #[test]
    fn deliver_rehome_to_aborts_cleanly_when_the_cas_is_lost() {
        // D-37 rule 3: if another writer moved the fence past prev_fence, the re-home CAS LOSES — the saga
        // is the loser and tombstones as a clean no-op (the entity is alive at the CAS winner; no
        // compensation, no ReHome envelope). Covers the ReHomeCommit executor's Lost arm end-to-end.
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        });
        // Drive the head AHEAD (Fence 2 @ SOURCE) of the saga's expectation (Fence 1) — someone else won.
        let _ = dir.grant(subject(), AuthorityRef::Shard(DEST), Fence(1), UniverseTick(0));
        let _ = dir.commit_cas(subject(), Fence(1), AuthorityRef::Shard(SOURCE), UniverseTick(0));
        inject_saga(
            &mut runtime,
            SagaState::Promoting {
                new_fence: Fence(1),
                promote_acked: false,
                dest_delivered: false,
                rehome_target: None,
            },
            UniverseTick(0),
        );
        stash_flush(&mut runtime, XFER, flushed_pose());
        let mut outbox = OutboundBox::default();
        deliver(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(5),
            XFER,
            SagaEvent::ReHomeTo { target: NodeId(9) },
        );
        assert_eq!(runtime.live(), 0, "the re-home loser tombstoned (clean no-op)");
        let head = dir.head(subject()).expect("subject still recorded");
        assert_eq!(
            head.authority,
            AuthorityRef::Shard(SOURCE),
            "the CAS winner still owns the entity"
        );
        assert_eq!(head.fence, Fence(2), "the re-home CAS did not bump (it lost)");
        assert!(
            flows_to_node(&outbox, NodeId(9)).is_empty(),
            "no ReHome adopt is emitted to the target when the CAS is lost"
        );
    }

    #[test]
    fn scan_deadlines_resolves_a_batch_handoff_with_a_dead_participant() {
        // D-7d: a DUE `BatchHandoff` saga whose SOURCE is known-dead self-promotes the dest
        // (SourceUnreachable → EmitTransientPromote + Done, counted); whose DEST is dead abandons the
        // source copy (DestUnreachable → EmitTransientAbandon + Done, counted); with NEITHER dead it just
        // RE-DRIVES (Timeout, still live, uncounted). The redrive deadline is 8 (`SagaTuning::default`).
        let mk = || {
            let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
            inject_saga(
                &mut runtime,
                SagaState::BatchHandoff {
                    phase: BatchHandoffPhase::AwaitPromote,
                    new_fence: Fence(2),
                },
                UniverseTick(0),
            );
            runtime
        };
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        });

        // SOURCE dead → self-promote the dest (TransientDrop to DEST), tombstone, count once. The
        // zero-loss self-promote is NOT abort-gated, so it fires on the first due scan once confirmed
        // (n == 1 in the default liveness tuning → one NodeUnreachable confirms).
        let mut runtime = mk();
        runtime
            .liveness
            .record_unreachable(SOURCE, UniverseTick(8));
        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8),
        );
        assert_eq!(
            runtime.live(),
            0,
            "the dead-source resolution tombstoned the saga"
        );
        assert!(
            flows_to_node(&outbox, DEST).contains(&InterShardFlow::TransientDrop(
                TransientHandoff {
                    transfer: XFER,
                    step_id: TRANSIENT_DROP_STEP,
                    fence: Fence(2),
                }
            )),
            "self-promote to the dest: {:?}",
            outbox.0
        );
        assert_eq!(runtime.source_unreachable_resolutions(), 1);
        assert_eq!(runtime.dest_unreachable_resolutions(), 0);

        // DEST dead → abandon the source copy — but the DESTRUCTIVE abandon (irreversible accounted loss)
        // is gated behind the LARGE abort budget from the FIRST confirmed-dead observation (the CSCALE-1
        // cure): a dest that recovers before the budget elapses must NOT be abandoned.
        let mut runtime = mk();
        runtime.liveness.record_unreachable(DEST, UniverseTick(8));
        let mut outbox = OutboundBox::default();
        // First due scan: dest CONFIRMED dead, but the abort budget has not elapsed → re-drive, not abandon.
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8),
        );
        assert_eq!(
            runtime.live(),
            1,
            "the abandon waits the abort budget — not fired on the first confirmed observation"
        );
        assert_eq!(runtime.dest_unreachable_resolutions(), 0);
        // After `abort_deadline_ticks` from the first observation: the abandon fires.
        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8 + saga::DEFAULT_ABORT_DEADLINE_TICKS),
        );
        assert_eq!(runtime.live(), 0);
        assert!(
            flows_to_node(&outbox, SOURCE).contains(&InterShardFlow::TransientAbandon(
                TransientHandoff {
                    transfer: XFER,
                    step_id: TRANSIENT_ABANDON_STEP,
                    fence: Fence(2),
                }
            )),
            "abandon to the source: {:?}",
            outbox.0
        );
        assert_eq!(runtime.dest_unreachable_resolutions(), 1);
        assert_eq!(runtime.source_unreachable_resolutions(), 0);

        // NEITHER dead → a plain Timeout re-drive: still live, no resolution counted.
        let mut runtime = mk();
        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8),
        );
        assert_eq!(runtime.live(), 1, "neither dead → re-drive, not resolve");
        assert_eq!(runtime.source_unreachable_resolutions(), 0);
        assert_eq!(runtime.dest_unreachable_resolutions(), 0);
    }

    #[test]
    fn liveness_tracker_confirms_only_after_n_consecutive_within_the_window() {
        // D-3 CSCALE-1: a peer is CONFIRMED dead only after `n_consecutive_unreachable` notices within the
        // window — a single blip never confirms. Covers: not-in-seen, not-enough, enough+fresh→confirmed,
        // stale (not-fresh), and clear-on-ack.
        let mut t = LivenessTracker::new(LivenessTuning {
            n_consecutive_unreachable: 3,
            unreachable_window_ticks: 10,
            retry_delay_ticks_hint: 2,
        });
        let n = NodeId(5);
        // Not in the tracker → not confirmed (the None arm).
        assert!(!t.is_confirmed_dead(n, UniverseTick(0)));
        // One notice → consecutive 1 < 3 → not enough.
        t.record_unreachable(n, UniverseTick(1));
        assert!(!t.is_confirmed_dead(n, UniverseTick(1)));
        // Two more within the window → consecutive 3 >= 3 + fresh → confirmed (the existing-entry increment).
        t.record_unreachable(n, UniverseTick(2));
        t.record_unreachable(n, UniverseTick(3));
        assert!(t.is_confirmed_dead(n, UniverseTick(3)));
        // STALE: far past the window from the run's first notice (tick 1) → not fresh → not confirmed.
        assert!(!t.is_confirmed_dead(n, UniverseTick(1 + 11)));
        // CLEAR-ON-ACK: a successful inbound un-marks the recovered peer.
        t.record_ack(n);
        assert!(!t.is_confirmed_dead(n, UniverseTick(3)));
        // record_ack is idempotent (clearing an absent node is a no-op).
        t.record_ack(n);
    }

    #[test]
    fn liveness_tracker_resets_a_run_whose_window_elapsed() {
        // A notice arriving AFTER the window since the run's first notice STARTS a fresh run (consecutive
        // 1), not an extension — a long-ago blip is not evidence of a current death (the RESET arm).
        let mut t = LivenessTracker::new(LivenessTuning {
            n_consecutive_unreachable: 2,
            unreachable_window_ticks: 5,
            retry_delay_ticks_hint: 2,
        });
        let n = NodeId(7);
        t.record_unreachable(n, UniverseTick(1)); // run starts: consecutive 1, first = 1
        t.record_unreachable(n, UniverseTick(2)); // within window: consecutive 2
        assert!(t.is_confirmed_dead(n, UniverseTick(2)));
        // A notice at tick 10 (10 - 1 = 9 > window 5) RESETS the run to consecutive 1, first = 10.
        t.record_unreachable(n, UniverseTick(10));
        assert!(
            !t.is_confirmed_dead(n, UniverseTick(10)),
            "the reset run has only 1 notice (< 2) — a stale run is not a confirmation"
        );
    }

    #[test]
    fn set_liveness_tuning_raises_the_confirmation_threshold() {
        // The test-config setter (used by the CSCALE-1 flap cell to run the prod margin n = 3 in-process):
        // re-tuning to n = 3 means a SINGLE NodeUnreachable no longer confirms a peer dead, where the
        // kill-equivalent default (n = 1) would. A fresh runtime has observed zero notices.
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // n = 1
        assert_eq!(runtime.liveness_notices(), 0);
        runtime.set_liveness_tuning(LivenessTuning {
            n_consecutive_unreachable: 3,
            unreachable_window_ticks: 64,
            retry_delay_ticks_hint: 2,
        });
        runtime
            .liveness
            .record_unreachable(NodeId(9), UniverseTick(1));
        assert!(
            !runtime.liveness.is_confirmed_dead(NodeId(9), UniverseTick(1)),
            "after re-tuning to n = 3, one notice is below the confirmation threshold"
        );
    }

    #[test]
    fn trigger_on_an_unrecorded_subject_refuses_to_start() {
        // Per-key serialization: no directory record → lock_transfer FALSE → no saga created.
        let mut rig = Rig::new();
        rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
        rig.settle();
        assert_eq!(rig.live(), 0, "an unrecorded subject cannot start a saga");
        assert!(rig.drain_gateway().is_empty());
    }

    #[test]
    fn an_ack_for_an_unknown_saga_is_an_idempotent_noop() {
        // A duplicate/stale ack after GC (or for a never-started transfer) is a no-op.
        let mut rig = Rig::new();
        rig.ack(TransferControlAck::Committed {
            transfer: TransferId(999),
        });
        assert_eq!(rig.live(), 0);
        assert!(rig.drain_gateway().is_empty());
    }

    #[test]
    fn ack_to_event_maps_every_ack_phase() {
        let cases: [(TransferControlAck, SagaEvent); 10] = [
            (
                TransferControlAck::Prepared {
                    transfer: XFER,
                    result: PrepareResult::Ready,
                },
                SagaEvent::Prepared(PrepareResult::Ready),
            ),
            (
                TransferControlAck::CutConfirmed {
                    transfer: XFER,
                    marker_seq: 1,
                },
                SagaEvent::CutConfirmed { marker_seq: 1 },
            ),
            (
                TransferControlAck::SourceFrozen {
                    transfer: XFER,
                    drained_seq: 2,
                },
                SagaEvent::SourceFrozen { drained_seq: 2 },
            ),
            (
                TransferControlAck::Committed { transfer: XFER },
                SagaEvent::RouteSwapped,
            ),
            (
                TransferControlAck::SourceThawed { transfer: XFER },
                SagaEvent::SourceThawed,
            ),
            (
                TransferControlAck::Aborted { transfer: XFER },
                SagaEvent::DestAborted,
            ),
            (
                TransferControlAck::Released { transfer: XFER },
                SagaEvent::Released,
            ),
            (
                TransferControlAck::DemoteAck { transfer: XFER },
                SagaEvent::DemoteAcked,
            ),
            (
                TransferControlAck::PromoteAck { transfer: XFER },
                SagaEvent::PromoteAcked,
            ),
            (
                TransferControlAck::DeliveredToObservers { transfer: XFER },
                SagaEvent::DestDelivered,
            ),
        ];
        for (ack, event) in cases {
            assert_eq!(ack_to_event(ack), event);
        }
    }

    #[test]
    fn the_rejection_ledger_is_bounded_with_a_counted_drop() {
        // ROB-1: the un-drained ledger can NEVER grow without limit — beyond the cap the
        // oldest rejections are shed, counted (the loud overflow ALERT), oldest-first.
        let mut runtime = SagaRuntimeRes::default();
        for i in 0..(REJECTION_LEDGER_CAP as u128 + 5) {
            runtime
                .rejected
                .push((TransferId(i), AbortReason::Cancelled));
        }
        bound_rejection_ledger(&mut runtime);
        assert_eq!(runtime.rejected.len(), REJECTION_LEDGER_CAP);
        assert_eq!(runtime.rejections_dropped, 5);
        assert_eq!(
            runtime.rejected.first().expect("non-empty").0,
            TransferId(5),
            "the 5 OLDEST were shed; the newest survive"
        );
    }

    #[test]
    fn emit_crossing_builds_for_an_entity_subject_and_skips_otherwise() {
        let c = ctx(DurabilityClass::Durable, Fence(1));

        // Entity subject + a flushed pose → the crossing is pushed to the DEST, stamped with the
        // given fence, STUB_CROSSING_STEP, and an empty (1d.1) state blob.
        let mut outbox = OutboundBox::default();
        emit_crossing(&c, Fence(2), Some(flushed_pose()), EpochId(1), &mut outbox);
        assert_eq!(
            outbox.0.len(),
            1,
            "an Entity subject with a pose emits a crossing"
        );
        let (to, class, bytes) = &outbox.0[0];
        assert_eq!(*to, DEST);
        assert_eq!(*class, MsgClass::Saga);
        assert_eq!(
            postcard::from_bytes::<InterShardFlow>(bytes).expect("decode"),
            InterShardFlow::Transfer(TransferEnvelope {
                transfer_id: XFER,
                universe_epoch: EpochId(1),
                schema_version: TRANSFER_SCHEMA_VERSION,
                fence: Fence(2),
                step_id: STUB_CROSSING_STEP,
                class: DurabilityClass::Durable,
                payload: TransitionPayload::StubCrossing {
                    entity: subject_eid(),
                    from_realm: FROM_REALM,
                    to_realm: TO_REALM,
                    pose: flushed_pose(),
                    state: vec![],
                },
            })
        );

        // No stashed pose → no-op (the gate makes this unreachable for an Entity subject, but the
        // helper is total — this covers the missing-pose None arm).
        let mut no_pose = OutboundBox::default();
        emit_crossing(&c, Fence(2), None, EpochId(1), &mut no_pose);
        assert!(no_pose.0.is_empty(), "no pose → no crossing");

        // Non-Entity subject (the Realm-subject FSM proptests) → no-op.
        let realm_ctx = SagaCtx {
            subject: DirectoryKey::Realm(RealmId::System(9)),
            ..c
        };
        let mut realm_out = OutboundBox::default();
        emit_crossing(
            &realm_ctx,
            Fence(2),
            Some(flushed_pose()),
            EpochId(1),
            &mut realm_out,
        );
        assert!(
            realm_out.0.is_empty(),
            "a non-Entity subject emits no crossing"
        );
    }

    #[test]
    fn a_flush_for_an_unknown_saga_is_a_noop() {
        // stash_flush's None arm: a SourceFlushed for an unknown / GC'd transfer mutates nothing.
        let mut runtime = SagaRuntimeRes::default();
        stash_flush(&mut runtime, TransferId(999), flushed_pose());
        assert_eq!(runtime.live(), 0, "a stray flush creates/mutates no saga");
    }

    #[test]
    fn a_dest_crossing_ack_and_a_non_saga_arm_are_dropped() {
        // drive_sagas decodes but DROPS the DEST's crossing ack (Accepted) and any non-saga-driving
        // arm (here a Saga command, which the orchestrator only ever SENDS): no saga starts,
        // nothing is emitted.
        let mut rig = Rig::new();
        for flow in [
            InterShardFlow::TransferAck(TransferAck::Accepted {
                transfer_id: XFER,
                step_id: STUB_CROSSING_STEP,
            }),
            InterShardFlow::Saga(TransferControl::RequestCut {
                transfer: XFER,
                session: SESSION,
            }),
        ] {
            rig.gateway
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(postcard::to_allocvec(&flow).expect("encode")),
                )
                .expect("sent");
        }
        rig.settle();
        assert_eq!(
            rig.live(),
            0,
            "neither a crossing ack nor a stray arm starts a saga"
        );
        assert!(
            rig.drain_gateway().is_empty(),
            "nothing is emitted in response"
        );
    }
}
