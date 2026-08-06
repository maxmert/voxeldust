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

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use bevy_ecs::prelude::{Res, ResMut, Resource};
use serde::{Deserialize, Serialize};
use vd_core::frame::{IdentityFrames, rebind_pose_to_dest};
use vd_core::pose::{RealmId, StampedPose};
use vd_core::{BatchId, EpochId, Fence, NodeId, SessionId, TransferId, UniverseTick};
use vd_sim::capability::{CapRequest, ShardProfile};
use vd_sim::directory::{DirectoryCore, DirectoryTuning};
use vd_sim::io::{Bytes, Inbound, MsgClass, Store};
use vd_sim::runtime::{ClockSample, InboundBox, OutboundBox};
use vd_sim::saga::{
    self, AbortReason, BatchHandoffPhase, LivenessTuning, SagaAction, SagaCtx, SagaEvent,
    SagaState, SagaTuning,
};
use vd_wire::intershard::{
    CrossingRequest, DEMOTE_STEP, DemoteCmd, FLUSH_SOURCE_STEP, FlushSource, InterShardFlow,
    PROMOTE_STEP, PromoteCmd, RE_HOME_STEP, RE_SOLICIT_STEP, ReHomeCmd, ReHomeState,
    STUB_CROSSING_STEP, TRANSFER_SCHEMA_VERSION, TRANSIENT_ABANDON_STEP, TRANSIENT_DISCARD_STEP,
    TRANSIENT_DROP_STEP, TRANSIENT_RELEASE_STEP, TransferAck, TransferEnvelope,
    TransientCrossingGrant, TransientCrossingRequest, TransientHandoff, TransitionPayload,
    crossing_transfer_id, namespaced_transfer_id,
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
    /// Slice 3f-D: a durable crossing-abort reply pending a source ack (Mechanism Y).
    AbortReply(TransferId),
    /// RLM Step 5b: one launch-intent record per live realm shard (write-ahead of the launch, so a
    /// rehydrate reconstructs exactly the children the pre-crash orchestrator minted). Self-describing
    /// VALUE (carries its own `NodeId`) — `scan(&[RLM_LAUNCH])` recovers the whole live set without
    /// parsing keys, and `is_alive` reconciles each against backend ground truth. A DISTINCT family from
    /// the D-6 saga WAL so the two never collide even when they share one redb file (the ONE key-space
    /// authority — [`StoreKey::bytes`]).
    RlmLaunch(NodeId),
    /// RLM Step 5b: the F2 monotone allocator high-water (`next_node`/`next_port`) — a SINGLE record
    /// (no payload key, like `Clock`). Persisted BEFORE each mint and NEVER derived from `max(survivors)`,
    /// so a torn-down id's slot is never re-minted even after every survivor's intent is deleted.
    RlmWater,
}

impl StoreKey {
    const SAGA: u8 = 1;
    const BATCH_GO: u8 = 2;
    const DIRECTORY: u8 = 3;
    const CLOCK: u8 = 4;
    /// Slice 3f-D: the crossing-abort-reply family tag (5 — the next free tag after CLOCK=4).
    const ABORT_REPLY: u8 = 5;
    /// RLM Step 5b: the launch-intent family tag (6). D-RLM-2 (the deferred demand-ledger snapshot) will
    /// take the NEXT free tag after these two when/if it lands — the tag space is append-only.
    const RLM_LAUNCH: u8 = 6;
    /// RLM Step 5b: the allocator-high-water family tag (7).
    const RLM_WATER: u8 = 7;

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
            StoreKey::AbortReply(t) => {
                k.push(Self::ABORT_REPLY);
                k.extend(postcard::to_allocvec(&t).expect("encode TransferId store key"));
            }
            StoreKey::RlmLaunch(n) => {
                k.push(Self::RLM_LAUNCH);
                k.extend(postcard::to_allocvec(&n).expect("encode NodeId store key"));
            }
            StoreKey::RlmWater => k.push(Self::RLM_WATER),
        }
        k
    }
}

/// The EXACT durable key bytes for one realm's launch-intent record (`[RLM_LAUNCH] ++ postcard(node)`).
/// Public so [`crate::rlm_spawn::SpawnCore`] stages intents through the SAME encoding the rehydrate scan
/// reads back, AND so the RLM Step-5e process-tier crash gate (`rlm_kill9_spawn.rs`) reopens `launch.redb`
/// and reads back the exact same rows — zero key-byte drift (the encoding lives ONLY in [`StoreKey::bytes`]).
#[must_use]
pub fn rlm_launch_store_key(node: NodeId) -> Vec<u8> {
    StoreKey::RlmLaunch(node).bytes()
}

/// The family prefix that `scan`s exactly the launch-intent records (RLM Step 5b rehydrate). Public so the
/// 5e crash gate can compute the SAME prefix the launch.redb writer pauses on (the mid-fsync crash window)
/// and that the gate scans the recovered ledger by — one encoder, [`StoreKey::bytes`], no drift.
#[must_use]
pub fn rlm_launch_prefix() -> Vec<u8> {
    vec![StoreKey::RLM_LAUNCH]
}

/// The durable key for the F2 allocator high-water (RLM Step 5b) — the single-record family.
#[must_use]
pub(crate) fn rlm_water_store_key() -> Vec<u8> {
    StoreKey::RlmWater.bytes()
}

/// The EXACT durable store-key bytes the group-commit barrier stages for a directory row
/// (`[DIRECTORY_TAG] ++ postcard(key)`). Public so a process-tier crash test (D-6 D-delta) can compute the
/// SAME bytes the barrier will persist for a planted grant — the content-keyed writer pause then fires on
/// exactly that batch, with zero key-byte drift (the encoding lives in ONE place, [`StoreKey::bytes`]).
#[must_use]
pub fn directory_store_key(key: &DirectoryKey) -> Vec<u8> {
    StoreKey::Directory(*key).bytes()
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
    /// D-3 — WHICH participant was first observed CONFIRMED-DEAD and WHEN, while this saga awaits a
    /// destructive resolution (`None` otherwise). A DESTRUCTIVE resolution (dest-abandon, source
    /// discard/self-promote, dest re-home) waits the LARGE `abort_deadline_ticks` measured FROM the tick
    /// here — NOT from `since` (which `scan_deadlines` re-arms to `now` on every fire, making a `now -
    /// since >= abort` gate structurally unsatisfiable). R-6d3c keys it by `NodeId`: a CAUSE-SWITCH (dest
    /// confirmed dead → the dest RECOVERS via `record_ack` → the SOURCE is confirmed dead) RE-ANCHORS to
    /// `now` (`dead_budget_elapsed`), so the new participant's budget is never measured from the OTHER
    /// participant's stale first-dead observation. RAM-ONLY (NOT persisted in `SagaSnapshot`): on rehydrate
    /// it re-derives `None`, so a restarted orchestrator re-accrues the abort budget from scratch rather
    /// than firing an irreversible resolution immediately.
    dead_observed_since: Option<(NodeId, UniverseTick)>,
    /// CA-1 S3/S4 — the OVER-DISCARD SAFETY latch. Monotone `true` once the orchestrator has ANY evidence
    /// the dest adopted this batch (a `BatchAdopted` seen in the inbox). `rehome_event_for` fires the
    /// destructive `SourceUnreachablePreAdopt` discard ONLY when this is `false` — so the discard is a
    /// genuine "the dest never got the batch" loss, never a "the ack is lost/in-flight" false loss. Set in
    /// the PRE-SCAN inbox pass (`latch_adopted_from_inbox`) BEFORE `scan_deadlines`, so a `BatchAdopted`
    /// arriving on the exact budget-maturity tick suppresses the discard THAT tick (latching in `deliver`,
    /// which runs in the post-scan drain, would be one tick too late). RAM-ONLY (NOT in `SagaSnapshot`): re-
    /// derives `false` on rehydrate — safe, because the liveness tracker is EMPTY on rehydrate (the D-6
    /// freeze), so no discard can fire until fresh notices re-accrue, and the dest's per-tick `BatchAdopted`
    /// re-drive re-establishes the latch long before then.
    dest_adopted: bool,
    /// The universe tick this saga was INSERTED — the age anchor for the arrival shield's leak bound.
    /// Deliberately NOT [`LiveSaga::since`], which `scan_deadlines` re-arms to `now` on every re-drive:
    /// anchoring the cap there would refresh the shield forever on exactly the wedged saga the cap exists
    /// to bound. RAM-ONLY (not in [`SagaSnapshot`]) — a rehydrated saga re-derives it to the recovered
    /// clock, the established precedent of `dead_observed_since`/`dest_adopted`, so a restart grants a
    /// fresh budget rather than instantly expiring a recovered arrival.
    opened: UniverseTick,
}

/// One hand-off whose arrival shield ran out of budget — reported so the lift is LOUD. Carries what a
/// 2am operator needs to find it: which transfer, which realm it was holding, what phase it is stuck in,
/// and for how long.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpiredArrival {
    pub transfer: TransferId,
    pub realm: RealmId,
    pub state: String,
    pub age_ticks: u64,
}

/// THE ARRIVAL CLASSIFIER: the realm this saga is currently delivering its subject INTO, or `None` if
/// nobody is on their way anywhere because of it. EXHAUSTIVE with no wildcard — a new saga shape does
/// not compile until someone decides which side of this line it falls on.
///
/// `None` for the three terminal/compensating shapes (a compensating saga sends the subject BACK to the
/// source, so the shield must drop immediately rather than after a timeout), and `None` for the two
/// RE-HOME shapes. A re-home is a recovery hand-off between MACHINES, not a move between PLACES: the
/// standing re-home fabricates its realm fields ([`rehome_ctx`]'s `PLACEHOLDERS` — an entity-derived
/// `System(entity)` that names no real realm) and parks indefinitely by design, so including it would
/// pin one garbage realm alive forever per orphaned entity on the first machine failure; and a crossing
/// that DEGRADED into a re-home is re-targeting away from a destination whose node is already confirmed
/// dead, where the zombie force-reap must be allowed to run.
fn arrival_dest(live: &LiveSaga) -> Option<RealmId> {
    match live.state {
        SagaState::Aborting { .. } | SagaState::Aborted { .. } | SagaState::Done { .. } => None,
        SagaState::ReHoming { .. }
        | SagaState::Promoting {
            rehome_target: Some(_),
            ..
        } => None,
        SagaState::AwaitProvision
        | SagaState::Preparing
        | SagaState::Cutting
        | SagaState::Freezing { .. }
        | SagaState::CommittingCas { .. }
        | SagaState::BatchCommitting { .. }
        | SagaState::BatchHandoff { .. }
        | SagaState::Swapping { .. }
        | SagaState::Demoting { .. }
        | SagaState::Promoting {
            rehome_target: None,
            ..
        }
        | SagaState::Releasing { .. } => Some(live.ctx.to_realm),
    }
}

/// A pending create-on-trigger. Enqueued by [`SagaRuntimeRes::start_transfer`] and processed
/// next tick (a stub-shard boundary-crossing request wires the real trigger at Slice 1d).
struct PendingStart {
    ctx: SagaCtx,
    gateway: NodeId,
}

/// D-37 Slice 3: a STANDING re-home detected by the expiry reaper — an `Entity` key whose committed
/// owner is confirmed-dead, lease lapsed, and key UNLOCKED (no live saga). Enqueued by
/// [`reap_lapsed_leases`] and drained the SAME tick by [`process_rehome_starts`] (a within-barrier
/// hand-off, RAM-only). It is NOT separately WAL'd: the durable artifacts are the locked directory
/// record + the armed re-home saga (both persisted in the same group-commit barrier); a kill-9 in the
/// enqueue→drain window loses only this RAM list, and on reboot the reaper re-detects the still-dead,
/// still-UNLOCKED record (the directory IS the durable trigger) and re-enqueues — idempotent recovery
/// with no new WAL family. Carries only REAL recoverable values (no fabricated pose — HR1: the dead
/// owner's store is sealed; the dot reconstruction is owed Slice 4/D-6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingReHome {
    /// The orphaned `Entity` directory key to recover.
    subject: DirectoryKey,
    /// The confirmed-dead committed owner (the re-home `source`/provenance; the CAS expectation owner).
    dead_owner: NodeId,
    /// The fence the dead owner was committed at — Slice 4's `ReHomeCommit` CAS expectation (`fence+1`).
    prev_fence: Fence,
}

/// Slice 3f-D (Mechanism Y): a durable orchestrator record that a CROSSING-ORIGIN durable saga tombstoned
/// ABORTED (a pre-CAS failure), so the SOURCE's `RequestInFlight` crossing latch must be POSITIVELY cleared
/// — a lost RAM enqueue would otherwise strand the entity's latch forever. Held in
/// [`SagaRuntimeRes::pending_abort_replies`] (keyed by the aborted `TransferId`) AND persisted
/// (`StoreKey::AbortReply`), so an orchestrator restart BETWEEN the abort and the source's ack re-emits
/// `CrossingAborted` via rehydrate → the latch still clears (a RAM-only map would have lost it on the kill).
/// Dropped by the source's `CrossingAbortedAck`, or SELF-REAPED when `source` is confirmed dead (D-37
/// already re-homed the entity → the abort is moot). Self-describing (carries `transfer`), so rehydrate
/// never parses the store key. NO generation/attempt field is needed: the 3f-D1 attempt-stamp already
/// makes each crossing ATTEMPT's id UNIQUE (`crossing_transfer_id(subject, subject_fence, attempt)`), so
/// there is no ABA on the `TransferId` key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PendingAbortReply {
    /// The aborted crossing's `TransferId` — the map key AND the id the source latch keys on.
    pub transfer: TransferId,
    /// The SOURCE shard to re-emit `CrossingAborted` to (and whose death triggers the moot-abort reap).
    pub source: NodeId,
    /// The subject key echoed back in `CrossingAborted` (the source's latch is entity-keyed).
    pub subject: DirectoryKey,
}

/// D-3 dead-vs-slow EVIDENCE for one peer: how many CONSECUTIVE `NodeUnreachable` notices (no
/// intervening successful inbound) have been observed toward it, and when the run STARTED — the
/// abort-budget anchor, PRESERVED across the run (never re-armed per notice, unlike the saga's `since`).
#[derive(Clone, Copy, Debug)]
struct UnreachEvidence {
    consecutive: u32,
    first_unreachable_tick: UniverseTick,
    /// D-3 Strong-AND MONOTONE latch: set true the moment `consecutive` first reaches
    /// `n_consecutive_unreachable`, and held true across subsequent window RESETS (the redial backoff
    /// spaces late notices past the window, so the freshness-gated `is_confirmed_dead` PULSE expires long
    /// before the `should_reap` reassign horizon). Cleared only by `record_ack` removing the whole entry.
    /// This is what lets `should_reap` gate on `is_latched_dead` at `lease_expires + max` without the
    /// pulse having expired — the split-brain deadline can never be defeated by a stale evidence run.
    confirmed_dead_latched: bool,
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
            confirmed_dead_latched: false,
        });
        if now.0.saturating_sub(ev.first_unreachable_tick.0) > self.tuning.unreachable_window_ticks
        {
            // A stale run RESETS its consecutive count + anchor — but the `confirmed_dead_latched` MONOTONE
            // latch is deliberately PRESERVED (only `record_ack` clears it): the redial backoff makes late
            // notices arrive past the window, so an ongoing partition resets its run repeatedly, yet a peer
            // that once reached `n` stays confirmed dead for `should_reap`'s ttl+max horizon.
            ev.consecutive = 1;
            ev.first_unreachable_tick = now;
        } else {
            ev.consecutive = ev.consecutive.saturating_add(1);
        }
        if ev.consecutive >= self.tuning.n_consecutive_unreachable {
            ev.confirmed_dead_latched = true;
        }
    }

    /// ANY successful inbound from `node`: it is alive — clear its evidence, INCLUDING the monotone latch
    /// (the whole entry is removed; idempotent). The clear-on-ack that makes a recoverable blip non-fatal
    /// and un-latches a peer that came back.
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

    /// D-3 Strong-AND: is `node` PERSISTENTLY confirmed dead — the MONOTONE latch (set the first tick its
    /// consecutive run reached `n`, held across window resets until a live inbound clears the whole entry).
    /// Unlike [`is_confirmed_dead`](LivenessTracker::is_confirmed_dead) (a freshness-gated PULSE that expires
    /// once the redial backoff spaces late notices past the window), the latch survives to `should_reap`'s
    /// `lease_expires + max` reassign horizon — so the ttl+max split-brain deadline is never defeated by a
    /// stale evidence run. RAM-only ⇒ empty on rehydrate (the D-6 mass-orphan freeze holds: a restarted
    /// orchestrator latches nobody until fresh post-restart notices re-accrue). No freshness/`now` term —
    /// the latch IS the persistence; the reassign-timing gate lives in `should_reap`.
    fn is_latched_dead(&self, node: NodeId) -> bool {
        self.seen
            .get(&node)
            .is_some_and(|ev| ev.confirmed_dead_latched)
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
    /// D-37 Slice 3: standing re-homes the reaper detected THIS sweep, drained the same tick by
    /// [`process_rehome_starts`]. RAM-only (a within-barrier hand-off): see [`PendingReHome`] for why
    /// no WAL family is needed (the locked record + armed saga are the durable artifacts; the reaper
    /// re-detects on reboot). Always empty BETWEEN ticks (drained at the end of every barrier).
    pending_rehome: Vec<PendingReHome>,
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
    /// R-6d3c anti-vacuity observable: dead-SOURCE PRE-adopt resolutions (`AwaitAdopt` +
    /// `SourceUnreachablePreAdopt`) — the batch is counted lost (discard-to-dest, NOT self-promote).
    /// Distinct from `source_unreachable_resolutions` (post-adopt, zero-loss self-promote) so the two
    /// dead-source causes are never conflated in the ledger. 0 on every healthy run and on a RESTART
    /// recovery; `> 0` is the never-restart cell's proof its resolution fired.
    batch_lost_source_crash: u64,
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
    /// R-4d M3 orchestrator-side shed observable: total `Inbound::SendShed` notices seen. A shed is a
    /// LOCAL transport refusal (a lane hit its retry-buffer byte cap, or an oversize frame) — it says
    /// NOTHING about the peer's liveness, so it is counted here and NEVER fed to `record_unreachable`
    /// (the false-confirm cure). 0 on a healthy run; `> 0` cross-references the mesh `reliable_shed`
    /// counter (an ALERT: a saturated retry buffer / a mis-sized frame).
    sends_shed: u64,
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
    /// Slice 3f-B: durable `CrossingRequest`s that RESOLVED (subject + dest-realm + session all found in
    /// the directory) and STARTED a crossing saga. A monotonic count; a redelivered request for an
    /// already-live crossing is absorbed by `process_starts`' `contains_key`/`lock_transfer` guard and
    /// does NOT re-increment (the count reflects distinct started sagas, not requests seen). 0 until the
    /// first durable boundary crossing lands.
    crossings_started: u64,
    /// Slice 3f-B: durable `CrossingRequest`s whose SUBJECT owner was found but whose dest-`Realm` OR
    /// `Session` head was UNRESOLVED, so no saga could start this tick. Counted-only for now — the
    /// abort-reply egress that clears the source latch is a LATER sub-slice (3f-D); until then a lost
    /// crossing here is re-driven by the source's ongoing boundary detection (the `ReDriven` class).
    crossing_unresolved: u64,
    /// Slice 3f-B: durable `CrossingRequest`s whose SUBJECT had no directory `OwnerRecord` at all (the
    /// entity's authority already moved / was revoked between the source's latch and this resolve). No
    /// owner to reply to ⇒ a counted drop.
    crossing_subject_gone: u64,
    /// Slice 3f-C: transient `TransientCrossingRequest`s whose dest `Realm` head RESOLVED, so a
    /// `TransientCrossingGrant` was emitted back to the transport-origin. A redelivered request re-grants
    /// the SAME deterministic `batch` id (idempotent at the source), so this counts grants EMITTED.
    transient_crossings_granted: u64,
    /// Slice 3f-C: transient `TransientCrossingRequest`s whose dest `Realm` head was UNRESOLVED — no grant
    /// emitted (the source re-requests; the grant is idempotent, so a later-resolving realm still grants).
    transient_dest_unresolved: u64,
    /// Slice 3f-D (Mechanism Y): crossing-origin durable-abort replies pending a source ack, keyed by the
    /// aborted `TransferId`. DURABLE (persisted via `StoreKey::AbortReply` + rehydrated) — the whole point is
    /// crash-durability of the latch-clear obligation, so an orchestrator restart between the abort and the
    /// source's ack re-emits `CrossingAborted` and the source latch still clears. `scan_deadlines` re-emits
    /// per entry (throttled to `redrive_deadline_ticks`) until a `CrossingAbortedAck` drops it (or a
    /// dead-source reap). Empty on `Default`. See [`PendingAbortReply`].
    pending_abort_replies: BTreeMap<TransferId, PendingAbortReply>,
    /// Slice 3f-D: the universe tick the abort-reply RE-EMIT pass last fired (like `scan_deadlines`' per-saga
    /// `since`, but ONE runtime-level cadence gate for the whole pending set) — so the re-emit runs at most
    /// once per `redrive_deadline_ticks`, never a per-tick storm against an alive-but-ack-stalled source. RAM
    /// throttle only (NOT persisted): a rebuilt orchestrator re-emits once immediately on its first post-restart
    /// scan, which is exactly the harmless prompt behaviour rehydrate wants (the source re-acks). 0 at genesis.
    last_abort_reply_emit: UniverseTick,
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
    /// Whether `node` has been LATCHED dead by the D-3 liveness tracker (RLM Step 3e: the reconciler's
    /// BUG-A zombie check reads this to tell a live head from a dead-but-recorded one). Delegates to the
    /// private liveness latch — the ONE authority on confirmed-dead nodes (`should_reap` reads the same).
    #[must_use]
    pub fn is_node_latched_dead(&self, node: NodeId) -> bool {
        self.liveness.is_latched_dead(node)
    }

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

    /// THE ARRIVAL SET — every realm a live hand-off is currently delivering someone INTO, so the
    /// lifecycle reconciler can refuse to shut down a place somebody is on their way to.
    ///
    /// DERIVED fresh from the live sagas on every call, never stored. That is the whole safety argument:
    /// there is no clearing path to forget, no eviction rule, and no second source of truth that could
    /// disagree. When `commit_result` tombstones a saga the realm leaves this set on the same sweep, for
    /// free; when a crossing aborts, [`arrival_dest`] classifies it out immediately rather than after a
    /// timeout; and because the saga snapshot persists its whole context, the set re-forms on the first
    /// sweep after an orchestrator restart. The fact and the thing that needs the fact are the same
    /// object, in the same process, on the same tick — there is no message to lose and no race to win.
    /// `max_age` is the leak bound: a single hand-off may hold its destination for that many ticks and
    /// no longer. `0` DISARMS the shield entirely (the inert default). Expiry is evaluated PER HAND-OFF,
    /// not per realm, so one wedged crossing cannot drag down every other arrival at a busy destination.
    /// Returns the shielded realms plus every hand-off whose budget ran out, so the caller can say so
    /// loudly rather than dropping a player's protection in silence.
    #[must_use]
    pub fn arriving_dest_realms(
        &self,
        now: UniverseTick,
        max_age: u64,
    ) -> (BTreeSet<RealmId>, Vec<ExpiredArrival>) {
        let mut shielded = BTreeSet::new();
        let mut expired = Vec::new();
        if max_age == 0 {
            return (shielded, expired); // disarmed
        }
        for (transfer, live) in &self.sagas {
            let Some(realm) = arrival_dest(live) else {
                continue;
            };
            let age = now.0.saturating_sub(live.opened.0);
            if age > max_age {
                expired.push(ExpiredArrival {
                    transfer: *transfer,
                    realm,
                    state: format!("{:?}", live.state),
                    age_ticks: age,
                });
            } else {
                shielded.insert(realm);
            }
        }
        (shielded, expired)
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

    /// R-6d3c — the count of dead-SOURCE PRE-adopt resolutions (`AwaitAdopt` +
    /// `SourceUnreachablePreAdopt`): the batch was counted lost + discarded-to-dest (NOT self-promoted).
    /// 0 on every healthy run and on a RESTART recovery; `> 0` proves the never-restart cell's resolution
    /// fired (anti-vacuity for the accounted-loss-on-detection).
    #[must_use]
    pub fn batch_lost_source_crash(&self) -> u64 {
        self.batch_lost_source_crash
    }

    /// D-3 anti-vacuity: total `NodeUnreachable` notices observed (the CSCALE-1 flap cell asserts `> 0`,
    /// so the blip genuinely exercised the path the old insert-only code would have abandoned on).
    #[must_use]
    pub fn liveness_notices(&self) -> u64 {
        self.liveness_notices
    }

    /// R-4d M3 orchestrator-side shed observable: total `Inbound::SendShed` notices seen. 0 on a healthy
    /// run; `> 0` = a lane hit its retry cap or an oversize frame was refused (cross-reference the mesh
    /// `reliable_shed` counter). A shed NEVER accrues to the liveness tracker — the regression proof is
    /// that a burst of sheds toward a live peer leaves it un-confirmed-dead.
    #[must_use]
    pub fn sends_shed(&self) -> u64 {
        self.sends_shed
    }

    /// Slice 3f-B — durable crossing sagas STARTED from a resolved `CrossingRequest` (subject, dest-realm,
    /// and session all in the directory). Monotonic; a redelivered request for a live crossing does not
    /// re-count (the `contains_key`/`lock_transfer` guard absorbs it).
    #[must_use]
    pub fn crossings_started(&self) -> u64 {
        self.crossings_started
    }

    /// Slice 3f-D (Mechanism Y) — the number of durable crossing-abort replies STAGED and awaiting a source
    /// `CrossingAbortedAck` (the persisted latch-clear obligations). Bumps at the pre-CAS abort tombstone
    /// (`commit_result`), drops when the source acks (or a dead-source ownership reap in `scan_deadlines`).
    /// The 3g abort/crash-leg observable: `>= 1` mid-flight proves the Mechanism-Y path ran (a non-crossing
    /// abort stages NOTHING); `== 0` after the ack proves the round-trip reaped it; restored from the WAL by
    /// `rehydrate` so a `>= 1` immediately post-restart proves the entry rode the store, not RAM survival.
    #[must_use]
    pub fn pending_abort_replies_len(&self) -> usize {
        self.pending_abort_replies.len()
    }

    /// Slice 3f-B — durable `CrossingRequest`s the subject was known for but the dest-realm OR session head
    /// was unresolved (no saga started this tick; the abort-reply that clears the source latch is 3f-D).
    #[must_use]
    pub fn crossing_unresolved(&self) -> u64 {
        self.crossing_unresolved
    }

    /// Slice 3f-B — durable `CrossingRequest`s whose subject had no directory owner (authority already
    /// moved/revoked); a counted drop with no reply target.
    #[must_use]
    pub fn crossing_subject_gone(&self) -> u64 {
        self.crossing_subject_gone
    }

    /// Slice 3f-C — transient `TransientCrossingGrant`s emitted for a resolved dest realm.
    #[must_use]
    pub fn transient_crossings_granted(&self) -> u64 {
        self.transient_crossings_granted
    }

    /// Slice 3f-C — transient `TransientCrossingRequest`s dropped because the dest realm was unresolved.
    #[must_use]
    pub fn transient_dest_unresolved(&self) -> u64 {
        self.transient_dest_unresolved
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
    // The SOURCE already rebased the flushed pose into the dest realm's live frame (`on_flush_source`), so
    // here it is a no-op relabel (same-frame ⇒ `IdentityFrames` returns it unchanged) — the transfer
    // machinery never reads the ephemeris (HR1). Kept in the funnel so the ONE rebind machinery (HR3) still
    // forms an `Area` dest frame from `to_parent` for a source path that had not pre-rebased.
    let pose = rebind_pose_to_dest(flush_pose?, ctx.to_realm, ctx.to_parent, &IdentityFrames);
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
    epoch: EpochId,
) -> Option<InterShardFlow> {
    let _entity = ctx.subject.transfer_subject_entity()?;
    // No-op relabel: the SOURCE pre-rebased the pose into the dest realm's live frame at flush (`IdentityFrames`
    // returns a same-frame pose unchanged). The rebind stays in the funnel (HR3, one machinery).
    let pose = rebind_pose_to_dest(flush_pose?, ctx.to_realm, ctx.to_parent, &IdentityFrames);
    Some(InterShardFlow::ReHome(ReHomeCmd {
        transfer: ctx.transfer,
        universe_epoch: epoch,
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
    epoch: EpochId,
    outbox: &mut OutboundBox,
) {
    match build_rehome(ctx, new_fence, flush_pose, epoch) {
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
                            // The SOURCE rebases the flushed pose into this dest realm's live frame before
                            // shipping it (the moving-realm crossing fix); the saga already holds the dest.
                            to_realm: ctx.to_realm,
                            to_parent: ctx.to_parent,
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
                    emit_rehome(ctx, new_fence, target, flush_pose, epoch, outbox);
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
                // R-6d3c NEVER-restart resolution: the source died in AwaitAdopt (pre-adopt), so tell the
                // DEST to DISCARD any late-replayed Arriving copy + poison its adopt. Pure egress to
                // ctx.dest carrying the go-token fence; the dest journals `(transfer, TRANSIENT_DISCARD_
                // STEP)`. Ack-FREE — the resolving saga is terminal (mirroring the abandon egress).
                SagaAction::EmitTransientDiscard { fence } => {
                    outbox.push_flow(
                        ctx.dest,
                        MsgClass::Saga,
                        &InterShardFlow::TransientDiscard(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: TRANSIENT_DISCARD_STEP,
                            fence,
                        }),
                    );
                }
                // R-6d3c — the batch-lost-source-crash count is applied in `scan_deadlines` PRE-delivery
                // (beside `source_unreachable_resolutions`), so this executor arm is a straight-line no-op.
                SagaAction::CountBatchLostSourceCrash => {}
                // CA-1 S3 — the AwaitAdopt liveness PROBE to the SOURCE. Pure egress to ctx.source carrying
                // the go-token fence, keyed `(transfer, RE_SOLICIT_STEP)`. Its SEND outcome is the signal: a
                // dead source's send fails → `NodeUnreachable` → `is_confirmed_dead(source)` → the pre-adopt
                // discard becomes reachable. A live source no-ops it. Re-driven every AwaitAdopt Timeout.
                SagaAction::EmitReSolicit { fence } => {
                    outbox.push_flow(
                        ctx.source,
                        MsgClass::Saga,
                        &InterShardFlow::ReSolicitBatch(TransientHandoff {
                            transfer: ctx.transfer,
                            step_id: RE_SOLICIT_STEP,
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

/// Slice 3f-D: is this saga a CROSSING-ORIGIN transfer (a geometric boundary crossing, id-namespaced
/// `0x39` in the `TransferId`'s high byte by [`crossing_transfer_id`])? A STATELESS TAG-CHECK on the id —
/// `0x39` is EXCLUSIVE of the re-home namespace (`0x37`) and the connection-plane's small high-byte-`0x00`
/// ids (see `namespaced_transfer_id`), so the discriminator needs no `SagaSnapshot` field, no version bump,
/// and no id recompute. A branchless monomorphic expression (HR5). Only a crossing-origin durable saga that
/// tombstones ABORTED owes a `PendingAbortReply` (Mechanism Y); every other tombstone is inert here.
#[must_use]
fn is_crossing_origin(ctx: &SagaCtx) -> bool {
    (ctx.transfer.0 >> 120) == 0x39
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
        // Slice 3f-D (Mechanism Y): a CROSSING-ORIGIN (`0x39`) DURABLE saga that tombstoned ABORTED (a
        // pre-CAS failure — source authority never handed off) owes the SOURCE a positive latch-clear, else
        // a lost RAM enqueue strands the entity's `RequestInFlight` latch. Read the live saga's ctx BEFORE
        // the `remove` (the source/subject to reply to), gated on the stateless id tag-check
        // (`is_crossing_origin`) AND the terminal being `Aborted` AND `Durable` class. `.get(..).expect(..)`
        // (NOT `if let Some`) — the saga is GUARANTEED present here (it was live when `run_to_quiescence`
        // produced this terminal, and this single-threaded schedule touched nothing in between), matching the
        // else-branch's `.expect` at the write-back below — so there is no uncoverable `None` region (HR5).
        let aborted = matches!(final_state, SagaState::Aborted { .. });
        let live = runtime
            .sagas
            .get(&transfer)
            .expect("the saga was present at the start of this run");
        let crossing_origin = is_crossing_origin(&live.ctx);
        let durable = live.ctx.class == vd_core::entity_kind::DurabilityClass::Durable;
        if aborted && crossing_origin && durable {
            let reply = PendingAbortReply {
                transfer,
                source: live.ctx.source,
                subject: live.ctx.subject,
            };
            runtime.pending_abort_replies.insert(transfer, reply);
            // Persist (self-describing). Committed at the SAME end-of-tick group-commit barrier as the
            // saga-snapshot DELETE below and the directory `abort_clear` (all ride `pending_writes` /
            // `dirty`), so the abort reply + the tombstone are ATOMIC post-crash — a restart never sees one
            // without the other. The FIRST `CrossingAborted` emit is left to `scan_deadlines` (which owns the
            // outbox + the throttle); Mechanism Y IS "scan re-emits per entry until acked", so a scan-driven
            // first emit is the design (and the first scan sees `last_abort_reply_emit == 0` → emits promptly).
            runtime
                .pending_writes
                .push((StoreKey::AbortReply(transfer).bytes(), Some(encode(&reply))));
        }
        runtime.sagas.remove(&transfer);
        // D-6: a tombstoned saga's durable snapshot is DELETED — rehydrate's Saga-scan must not
        // resurrect it (the matching directory mutation commits in the SAME barrier, so the durable
        // saga set + directory are always consistent post-crash). UNCONDITIONAL (never gated on the
        // crossing check) — every tombstone deletes its snapshot.
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
pub(crate) fn encode<T: Serialize>(value: &T) -> Bytes {
    postcard::to_allocvec(value)
        .expect("postcard encodes a durable record")
        .into()
}

/// The orchestrator state recovered from a non-empty durable [`Store`] on restart (D-6).
pub(crate) struct Rehydrated {
    pub clock: CeilingClock,
    pub directory: DirectoryCore,
    pub runtime: SagaRuntimeRes,
    /// RLM Step 4a — the crash-recovery FREEZE watermark for the realm reconciler: teardown is blocked
    /// until `now` reaches it, so a rebuilt orchestrator (empty RAM demand ledger) never mass-reaps a
    /// still-occupied realm before its parent's `KeepAlive` (or a login demand) re-accrues. Measured from
    /// the RECOVERED ceiling (the resumed `now`) + `RlmTuning::recovery_grace_ticks`; the boot path arms
    /// the reconciler with it. `0` when the RLM tuning is inert (nothing to freeze).
    pub rlm_quiesced_until: UniverseTick,
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
    rlm: vd_sim::rlm::RlmTuning,
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
                dest_adopted: false,
                // The recovered clock IS this restart's `now`, so a rehydrated arrival starts its shield
                // budget fresh here rather than at the `0` that would expire it on the first sweep.
                opened: ceiling,
            },
        );
    }

    let mut batch_goes: BTreeMap<BatchId, (Fence, NodeId, NodeId)> = BTreeMap::new();
    for (_k, v) in store.scan(&[StoreKey::BATCH_GO]) {
        let token: GoTokenSnapshot = postcard::from_bytes(&v).expect("decode persisted go-token");
        batch_goes.insert(token.batch, (token.fence, token.source, token.dest));
    }
    let batch_go_writes = batch_goes.len() as u64;

    // Slice 3f-D (Mechanism Y): restore the durable crossing-abort replies (self-describing — no key
    // parse), so a restart between an abort and the source's ack re-emits `CrossingAborted` on the first
    // post-restart `scan_deadlines` tick and the source latch still clears (a RAM-only map would strand it).
    let mut pending_abort_replies: BTreeMap<TransferId, PendingAbortReply> = BTreeMap::new();
    for (_k, v) in store.scan(&[StoreKey::ABORT_REPLY]) {
        let reply: PendingAbortReply =
            postcard::from_bytes(&v).expect("decode persisted abort-reply");
        pending_abort_replies.insert(reply.transfer, reply);
    }

    // D-3: thread the CONFIGURED liveness tuning through the recover path so a kill-9-rebuilt orchestrator
    // keeps its prod dead-vs-slow margin (n=3), not the kill-equivalent n=1 default. The tracker itself is
    // rebuilt EMPTY (the CAP freeze), but its TUNING must survive the restart.
    let mut runtime = SagaRuntimeRes::with_tunings(saga_tuning, liveness);
    runtime.sagas = sagas;
    runtime.batch_goes = batch_goes;
    runtime.batch_go_writes = batch_go_writes;
    runtime.pending_abort_replies = pending_abort_replies;
    // D-3 Slice 4 CAP freeze: a rebuilt orchestrator does NOT reap for `recovery_grace_ticks` after recover
    // — belt-and-suspenders atop the RAM-empty tracker (the PRIMARY freeze), so even a fast restart cannot
    // race a slow-rejoining live peer into a reap before its first renewal re-lands. Measured from the
    // recovered ceiling (the resumed `now`).
    runtime.liveness_quiesced_until =
        UniverseTick(ceiling.0.saturating_add(dir_tuning.recovery_grace_ticks));

    // RLM Step 4a — the realm reconciler's own crash-recovery freeze, sized from the DEMAND re-accrue
    // cadence (`RlmTuning::recovery_grace_ticks`, a different clock than the liveness renewal above). A
    // rebuilt orchestrator's RAM demand ledger is EMPTY, so without this freeze a shard that re-asserts
    // `Empty` before its parent's `KeepAlive` re-lands would be reaped despite still being occupied.
    // Measured from the RAW recovered `ceiling` (the resumed `now`) — NOT `clock.confirmed_ceiling()`,
    // which is already advanced by `reserve_chunk` at the boot insert site.
    let rlm_quiesced_until = UniverseTick(ceiling.0.saturating_add(rlm.recovery_grace_ticks));

    Some(Rehydrated {
        clock,
        directory,
        runtime,
        rlm_quiesced_until,
    })
}

/// Slice 3f-B — consume a DURABLE `CrossingRequest` from a source shard's boundary detector: resolve the
/// subject's current owner, the dest realm's owner, and the client's gateway (from the `Session` head), then
/// START a durable crossing saga on the EXISTING `start_transfer` entry. The saga's `ctx.transfer` is the id
/// the SOURCE latched — `crossing_transfer_id(req.subject, req.subject_fence)` over the WIRE fence — so the
/// eventual `Demote`/abort terminal carries the exact id the source's `RequestInFlight` latch keys on, even
/// if the head fence advanced between latch and resolve. `ctx.expected_fence` is the CURRENT head fence (the
/// CAS expectation), which is distinct from the id fence by design.
///
/// MONOMORPHIC (no generic body): ALL branching is the single 3-arm `match` on the three head reads (NOT a
/// let-else), so each outcome — start / unresolved / subject-gone — is a covered region (HR5). The
/// abort-reply egress that clears the source latch on an unresolved dest is a LATER sub-slice (3f-D); here an
/// unresolved crossing is only COUNTED (the source re-drives it — the `ReDriven` class), never replied to.
fn handle_crossing_request(
    runtime: &mut SagaRuntimeRes,
    dir: &DirectoryCore,
    _outbox: &mut OutboundBox,
    req: CrossingRequest,
) {
    // The id the source latched — derived from the WIRE fence (`req.subject_fence`) + the source's
    // per-attempt counter (`req.attempt`), NOT the head fence, so a terminal carrying it matches the source
    // latch even if the head advanced, and a post-abort re-cross's fresh attempt gets a distinct id (H2).
    let transfer = crossing_transfer_id(req.subject, req.subject_fence, req.attempt);
    match (
        dir.head(req.subject),
        dir.head(DirectoryKey::Realm(req.to_realm)),
        dir.head(DirectoryKey::Session(req.session)),
    ) {
        // REDELIVERY GUARD (mirrors `process_starts`' `contains_key`): a re-delivered request for a crossing
        // whose saga is ALREADY live is absorbed — no second enqueue, no re-count — so `crossings_started`
        // reflects DISTINCT started sagas, not requests seen. The source re-drives the request every tick the
        // entity stays over the boundary (`ReDriven`), so this arm is the common steady-state case.
        (Some(_subj), Some(_dest_rec), Some(_sess_rec))
            if runtime.sagas.contains_key(&transfer) => {}
        (Some(subj), Some(dest_rec), Some(sess_rec)) => {
            let ctx = SagaCtx {
                transfer,
                session: req.session,
                subject: req.subject,
                // The CURRENT head fence is the CAS expectation (distinct from the WIRE-fence-derived id).
                expected_fence: subj.fence,
                source: subj.authority.node(),
                dest: dest_rec.authority.node(),
                class: vd_core::entity_kind::DurabilityClass::Durable,
                needs_provision: false,
                from_realm: req.from_realm,
                to_realm: req.to_realm,
                // Thread the dest realm's parent provenance the SOURCE detector filled — so `build_crossing`'s
                // `rebind_pose_to_dest` forms an Area frame (the "Area label never flips" fix).
                to_parent: req.to_parent,
            };
            let gateway = sess_rec.authority.node();
            runtime.start_transfer(ctx, gateway);
            runtime.crossings_started += 1;
        }
        // The subject owner is known but the dest realm OR the session route is unresolved: no saga can start
        // this tick. COUNTED only (3f-B is happy-path; the source-latch-clearing abort-reply is 3f-D).
        (Some(_subj), _, _) => runtime.crossing_unresolved += 1,
        // No directory owner for the subject at all (authority already moved/revoked): a counted drop.
        (None, _, _) => runtime.crossing_subject_gone += 1,
    }
}

/// Slice 3f-C — consume a TRANSIENT `TransientCrossingRequest`: resolve the dest realm's owner and GRANT it
/// back to the transport-origin `from`. A transient is NOT a directory `OwnerRecord` (burst isolation — HR2),
/// so `from` (the connection the request arrived on) is the ONLY authoritative reply address; there is no
/// owner lookup for the source. The `batch` id is `crossing_transfer_id(req.subject, req.src_realm_fence)` —
/// deterministic per subject, so a redelivered request re-grants the SAME batch (absorbed at the source's
/// `on_transient_crossing_grant` no-op). The grant is `ReDriven` (the source re-requests) ⇒ plain `push_flow`
/// (Ephemeral), never `Retained`. D-43 #9: the resolved arm ALSO starts the `BatchHandoff` saga (keyed on the
/// same `batch`) so the dest's `BatchAdopted` lands on a live `AwaitAdopt` (see the inline note for why this
/// causes no double-emit).
///
/// MONOMORPHIC (no generic body): the branches are the 2-arm `Option` match plus the plain `if !contains_key`
/// saga-start guard (a `ReDriven` re-request whose saga is already live takes the skip arm) — all HR5-covered.
fn handle_transient_crossing_request(
    dir: &DirectoryCore,
    outbox: &mut OutboundBox,
    runtime: &mut SagaRuntimeRes,
    req: TransientCrossingRequest,
    from: NodeId,
) {
    match dir.head(DirectoryKey::Realm(req.to_realm)) {
        Some(rec) => {
            // Transients carry no per-entity attempt (no durable latch / abort-reply) → attempt 0.
            let batch = crossing_transfer_id(req.subject, req.src_realm_fence, 0);
            outbox.push_flow(
                from,
                MsgClass::Saga,
                &InterShardFlow::TransientCrossingGrant(TransientCrossingGrant {
                    subject: req.subject,
                    dest: rec.authority.node(),
                    to_realm: req.to_realm,
                    dst_realm_fence: rec.fence,
                    batch,
                    // Copy the dest realm's parent VERBATIM so the source can stamp it onto
                    // `TransientStatus::Crossing` for the batch's Area-frame rebind.
                    to_parent: req.to_parent,
                }),
            );
            runtime.transient_crossings_granted += 1;

            // D-43 #9: START the SAME `BatchHandoff` saga the machinery drives (HR2 one-machinery),
            // keyed on `batch` == the id the DEST acks `BatchAdopted` under, so `deliver` routes the
            // adopt onto a LIVE `AwaitAdopt` saga instead of the silent None early-return (the exact
            // symptom: `orch.batch_goes == []`, the dest stuck `Arriving`). This is the HR2/HR3 twin of
            // the durable `handle_crossing_request` (build ctx → `start_transfer`), differing ONLY in
            // `class: Transient` + the inert session/subject/gateway.
            //
            // NO DOUBLE-EMIT: the source emits its batch on the GRANT (the `Held→Crossing` flip in
            // `on_transient_crossing_grant`), NOT on the orchestrator-local go-token (`IssueTransientGo`
            // is wire-silent), so starting the saga at grant time is invisible to the source. The
            // `contains_key` fast-skip mirrors the durable arm — it avoids a redundant `pending` push when
            // the saga is already live from a prior tick's re-request; `process_starts`' own `contains_key`
            // is the AUTHORITATIVE one-saga guard (a transient takes NO directory lock, so a debris burst
            // stays burst-isolated).
            if !runtime.sagas.contains_key(&batch) {
                let ctx = SagaCtx {
                    transfer: batch,
                    // INERT — the transient short-path never reads `ctx.session` (no gateway route-swap).
                    session: SessionId::NONE,
                    // INERT provenance — a transient never enters the directory (`locks_directory_key`
                    // is false), so `subject` is never a CAS/lock key.
                    subject: DirectoryKey::Realm(req.to_realm),
                    // The dest realm-lease fence the batched go-token commits at.
                    expected_fence: rec.fence,
                    // The transport-origin connection = the source shard.
                    source: from,
                    dest: rec.authority.node(),
                    class: vd_core::entity_kind::DurabilityClass::Transient,
                    needs_provision: false,
                    from_realm: req.from_realm,
                    to_realm: req.to_realm,
                    // INERT for the transient batch path: the batch's pose rebind rides `TransientStatus::
                    // Crossing.to_parent` (carried on the GRANT below), never this ctx (the transient saga is
                    // the orchestrator-side `BatchHandoff` choreography, which builds no crossing envelope).
                    to_parent: None,
                };
                // The `gateway` arg is INERT for a Transient (never read, never rendered by `views`) —
                // pass the in-scope `from` rather than fabricate a sentinel NodeId.
                runtime.start_transfer(ctx, from);
            }
        }
        // Unresolved dest realm: emit nothing (the source re-requests; the grant is idempotent, so a
        // later-resolving realm still grants). Counted only.
        None => runtime.transient_dest_unresolved += 1,
    }
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
                dest_adopted: false,
                opened: now,
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
/// R-6d3c — the DESTRUCTIVE-resolution budget elapsed for `node`, keyed by the confirmed-dead
/// PARTICIPANT so a CAUSE-SWITCH re-anchors. A single per-saga `Option<(NodeId, UniverseTick)>` times
/// "how long has THIS participant been observed dead"; if the anchor is unset OR held by a DIFFERENT node
/// (the dest was confirmed dead, then RECOVERED, then the source was confirmed dead) it RE-ANCHORS to
/// `now` — so the new participant's `abort_deadline_ticks` budget is never measured from the OTHER
/// participant's stale first-dead observation (the shared-anchor regression the R-6d3c review caught: a
/// destructive source resolution would otherwise fire before its own budget elapsed). Monomorphic — the
/// re-anchor branch is covered ONCE here so both call sites stay branchless (HR5).
fn dead_budget_elapsed(
    anchor: &mut Option<(NodeId, UniverseTick)>,
    node: NodeId,
    now: UniverseTick,
) -> u64 {
    let observed = match *anchor {
        Some((n, t)) if n == node => t,
        _ => {
            *anchor = Some((node, now));
            now
        }
    };
    now.0.saturating_sub(observed.0)
}

#[allow(clippy::too_many_arguments)] // the dead-aware decision needs the full saga + liveness + roster context
fn rehome_event_for(
    state: &SagaState,
    ctx: &SagaCtx,
    liveness: &LivenessTracker,
    dead_observed_since: &mut Option<(NodeId, UniverseTick)>,
    tuning: &SagaTuning,
    now: UniverseTick,
    roster: &BTreeMap<NodeId, ShardProfile>,
    req: &CapRequest,
    subject_owner: NodeId,
    dest_adopted: bool,
) -> SagaEvent {
    match state {
        // D-7d / R-6d3c transient batch hand-off: dest-dead abandons the source's retained copy; source-
        // dead splits BY PHASE — a POST-adopt phase self-promotes the dest (zero loss), PRE-adopt
        // `AwaitAdopt` discards-to-dest + counts the loss (`SourceUnreachablePreAdopt`, R-6d3c). BOTH the
        // source-dead and dest-dead resolutions are DESTRUCTIVE now (self-promoting an unadopted batch, or
        // discarding+accounting a loss, are both irreversible), so BOTH are budget-gated on
        // `abort_deadline_ticks` measured from `dead_observed_since` — a source that RESTARTS within budget
        // delivers via its durable outbox replay FIRST (the two recoveries race; the budget picks the
        // restart winner). R-6d3c ADDED the source-dead budget-gate (the former immediate self-promote
        // could not lose a race, but the AwaitAdopt discard MUST give the restart a chance).
        SagaState::BatchHandoff { phase, .. } => {
            // CA-1 S3/S4 OVER-DISCARD SAFETY (retires the R-6d4-F2 tripwire). Once the dest has ADOPTED an
            // `AwaitAdopt` batch (`dest_adopted`, latched in the PRE-SCAN inbox pass), the DEST is the
            // holder-of-record: its fate decides, NOT the source's. So DEFER to the dest-death ladder for
            // that case and NEVER take the source-death resolution — because the `EmitReSolicit` probe now
            // makes `is_confirmed_dead(source)` reachable in AwaitAdopt (it was inert before), and letting
            // the source `if` win would (a) over-discard a batch the dest holds when the dest is alive, and
            // (b) WEDGE forever when BOTH are dead (the source arm returns a bare Timeout that the dest-dead
            // `else if` can never reach). Deferring restores the pre-CA-1-S3 terminal: dest alive → re-drive
            // (the drain advances the phase off the latched `BatchAdopted`, then the post-adopt self-promote
            // handles the dead source); dest dead → the budget-gated `DestUnreachable` abandon+tombstone
            // (admin-visible via `dest_unreachable_resolutions`). The source-death resolutions
            // (discard-if-never-adopted / self-promote-if-post-adopt) apply only when the dest is NOT the
            // holder — i.e. `!defer_to_dest`.
            let defer_to_dest = matches!(phase, BatchHandoffPhase::AwaitAdopt) && dest_adopted;
            if liveness.is_confirmed_dead(ctx.source, now) && !defer_to_dest {
                // Keyed by ctx.source so a dest-dead-then-source-dead cause-switch re-anchors (does NOT
                // measure the source's restart-race budget from the dest's stale first-dead observation).
                if dead_budget_elapsed(dead_observed_since, ctx.source, now)
                    >= tuning.abort_deadline_ticks
                {
                    match phase {
                        // PRE-adopt (`!defer_to_dest` ⇒ dest never adopted here): the dest never received
                        // the batch → discard-to-dest + count the source-crash loss.
                        BatchHandoffPhase::AwaitAdopt => {
                            *dead_observed_since = None;
                            SagaEvent::SourceUnreachablePreAdopt
                        }
                        // POST-adopt phase: the dest provably holds the batch → zero-loss self-promote.
                        _ => {
                            *dead_observed_since = None;
                            SagaEvent::SourceUnreachable
                        }
                    }
                } else {
                    SagaEvent::Timeout // cheap re-drive while the restart-race budget accrues
                }
            } else if liveness.is_confirmed_dead(ctx.dest, now) {
                // The dest is confirmed dead — the DESTRUCTIVE budget gate (a recoverable blip that clears
                // in time never abandons a healthy dest). Reached for a post-adopt dest-death AND (via
                // `defer_to_dest`) for the AwaitAdopt-adopted-then-dead double-crash: both abandon the
                // source's retained copy + tombstone (the go-token can no longer promote a dead dest).
                if dead_budget_elapsed(dead_observed_since, ctx.dest, now)
                    >= tuning.abort_deadline_ticks
                {
                    SagaEvent::DestUnreachable
                } else {
                    SagaEvent::Timeout // cheap re-drive while the abort budget accrues (corpse won't ack)
                }
            } else {
                *dead_observed_since = None; // neither confirmed dead (or deferring to a LIVE dest) → re-drive
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
                // Keyed by subject_owner (the committed dead dest); a Promoting saga only ever times this
                // one participant, but the keyed anchor keeps the budget honest if the owner ever changes.
                if dead_budget_elapsed(dead_observed_since, subject_owner, now)
                    >= tuning.abort_deadline_ticks
                {
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

/// D-37 Slice 3: a DETERMINISTIC, replay-stable `TransferId` for an orchestrator-minted STANDING re-home.
/// FNV-1a over the postcard bytes of `(subject, prev_fence)` into the low 120 bits, tagged with a fixed
/// high byte (`0x37`, the D-37 namespace) so it can NEVER collide with a client/gateway-assigned
/// `TransferId` (those are minted small + sequential in the connection plane, never with this tag). Pure +
/// branchless (HR5: no rng, no default hasher — the byte-identical seed-replay canary forbids both); the
/// same orphan + fence always derives the same id, so a re-attempt after a lost RAM enqueue is idempotent.
/// One orphan re-homes at most once per fence (the `in_transfer` lock dedups), so per-`(subject, fence)`
/// uniqueness suffices. Slice 3f-D (L2 DRY): the FNV body is the shared [`namespaced_transfer_id`]
/// primitive; only the `0x37` tag + the 2-arg seed are re-home's own.
fn rehome_transfer_id(subject: DirectoryKey, prev_fence: Fence) -> TransferId {
    let seed = postcard::to_allocvec(&(subject, prev_fence)).expect("encode rehome id seed");
    namespaced_transfer_id(0x37, &seed)
}

/// D-37 Slice 3: build the `SagaCtx` for a STANDING re-home. REAL fields carry the recovery: `subject`
/// (the orphan), `expected_fence = prev_fence` (Slice-4's `ReHomeCommit` CAS expectation, `fence+1`),
/// `source = dead_owner` (the `ReHome` envelope's provenance), `dest = target` (the selected live shard),
/// `class = Durable` (a standing re-home is always a durable per-key recovery — transient batches live and
/// die in one realm, never standing-re-homed). PLACEHOLDER fields — the `ReHome` envelope carries NO
/// session/realm and the Slice-4 adopt sources the entity's session/realm/pose from the RealmId-keyed
/// checkpoint reload (D-6/P7), NOT this ctx — so `session`/`from_realm` are NEVER read by the re-home path;
/// they are derived from the entity for replay-determinism + the shared WAL snapshot shape.
///
/// `to_realm` CAVEAT (frame-rebinding): `build_rehome` now rebinds the flushed pose into `to_realm`'s frame,
/// so `to_realm` IS read WHEN a pose is present. This standing-reaper path is safe because it ALWAYS parks
/// with `flushed_pose: None` (a pre-flush death has no recoverable pose — see `process_rehome_starts`), so
/// the `rebind_pose_to_dest` call is never reached and the entity-derived placeholder stays inert. WHEN
/// Slice-4/P7 sources a REAL pose from the RealmId-keyed checkpoint, it MUST supply the true destination
/// realm here (not the `System(entity)` placeholder) or the pose will be rebound into the wrong frame.
fn rehome_ctx(
    subject: DirectoryKey,
    prev_fence: Fence,
    dead_owner: NodeId,
    target: NodeId,
) -> SagaCtx {
    // Slice 3 only re-homes Entity keys (the reaper leaves Realm/Ship for Slice 4), so this is always Some.
    let entity = subject
        .transfer_subject_entity()
        .expect("a standing re-home subject is an Entity key")
        .0;
    SagaCtx {
        transfer: rehome_transfer_id(subject, prev_fence),
        // PLACEHOLDERS (never read by the re-home — see the doc above): entity-derived for determinism.
        session: SessionId(entity),
        from_realm: RealmId::System(entity as u64),
        to_realm: RealmId::System(entity as u64),
        // REAL recovery fields.
        subject,
        expected_fence: prev_fence,
        source: dead_owner,
        dest: target,
        class: vd_core::entity_kind::DurabilityClass::Durable,
        needs_provision: false,
        // `None`: the standing reaper ALWAYS parks with `flushed_pose: None` (a pre-flush death has no
        // recoverable pose — see the `to_realm` CAVEAT above), so `build_rehome`'s `rebind_pose_to_dest` is
        // never reached and this parent is inert. WHEN Slice-4/P7 sources a REAL pose from the RealmId-keyed
        // checkpoint, it must supply the true dest parent here (with the true `to_realm`) for an Area target.
        to_parent: None,
    }
}

/// D-37 Slice 3: drain the standing re-home queue the reaper filled THIS sweep (a same-tick within-barrier
/// hand-off) and ARM each through the ONE machinery. For each orphan: pick a LIVE capability-matched target
/// (`select_rehome_target`); `None` (whole-pool death / no spare capacity) DROPS the entry — the reaper
/// re-detects the still-unlocked record next sweep and retries once a target appears (interval-paced, never
/// a tight spin, never a forced incapable re-home). `lock_transfer` makes the armed saga the SOLE owner of
/// the key's transfer lifecycle (so the next reaper sweep SKIPS it via the `in_transfer` gate — no
/// re-detection churn); `false` = already locked → skip (one key → one re-home saga). The saga ARMS via
/// `saga::start_rehome` and PARKS in `ReHoming` (CONSERVATIVE Slice-3 split: no `ReHomeCommit`, no adopt —
/// authority stays at the dead owner because a pre-flush death has no recoverable pose until the
/// RealmId-keyed redb lands, D-6/P7/Slice 4). `run_to_quiescence` + `commit_result` PERSIST the parked saga
/// (durable across kill-9; on reboot it rehydrates and the locked record keeps the reaper off it). Runs
/// inside the D-6 group-commit barrier right after the reaper, so the lock + the armed saga are durable the
/// SAME tick. NO fabricated pose (HR1: the dead owner's store is sealed; `flushed_pose: None`).
fn process_rehome_starts(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    epoch: EpochId,
    now: UniverseTick,
) {
    let req = CapRequest::default();
    for PendingReHome {
        subject,
        dead_owner,
        prev_fence,
    } in std::mem::take(&mut runtime.pending_rehome)
    {
        let Some(target) = select_rehome_target(&req, &runtime.roster, &runtime.liveness, now)
        else {
            continue; // no live capable shard → drop; the reaper re-detects + retries next sweep (honest)
        };
        let transfer = rehome_transfer_id(subject, prev_fence);
        if !dir.lock_transfer(subject, transfer) {
            continue; // already locked (a concurrent arm / a prior sweep's saga) → one key, one re-home
        }
        let ctx = rehome_ctx(subject, prev_fence, dead_owner, target);
        let (state, actions) = saga::start_rehome(target, prev_fence);
        runtime.sagas.insert(
            transfer,
            LiveSaga {
                ctx,
                state,
                // No client egress in the re-home tail (the adopt is shard→shard); `dead_owner` is an
                // inert provenance placeholder, never read while parked or in the Slice-4 adopt.
                gateway: dead_owner,
                since: now,
                flushed_pose: None, // HR1: the dead owner's store is sealed; the adopt pose is owed Slice 4
                dead_observed_since: None,
                dest_adopted: false,
                opened: now,
            },
        );
        // Parks immediately (start_rehome emits no actions); run_to_quiescence + commit_result PERSIST the
        // ReHoming snapshot so the armed re-home survives an orchestrator kill-9 (rehydrates parked).
        let (final_state, tombstone, rejected, batch_gos) = run_to_quiescence(
            &ctx, dead_owner, state, actions, dir, outbox, epoch, now, None,
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
                live.dest_adopted,
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
            // R-6d3c: the never-restart PRE-adopt resolution (accounted loss, discard-to-dest). Counted
            // HERE — this `_ => {}` wildcard is the ONE place a missing counter arm would compile green
            // (unlike every other new SagaEvent/SagaAction, which the total FSM/executor match rejects).
            SagaEvent::SourceUnreachablePreAdopt => runtime.batch_lost_source_crash += 1,
            _ => {}
        }
        deliver(runtime, dir, outbox, epoch, now, transfer, event);
    }

    // Slice 3f-D (Mechanism Y): drive the durable crossing-abort replies — a SEPARATE pass (disjoint from
    // `sagas`), run AFTER the `due` drain. Per entry: (1) REAP when the subject NO LONGER belongs to
    // `source` (`dir.head(subject) != source`) — the abort is then MOOT: D-37 has re-homed the dead
    // source's entity, OR it transferred away, so the source's stale crossing latch is irrelevant. This is
    // the SAME "entity left the owner" fact D-37's re-home discharges, NOT the `is_confirmed_dead` PULSE
    // (which flips true on a transient blip AND EXPIRES on recovery — so it would reap a briefly-blipped-
    // then-RECOVERED source's live latch-clear obligation BEFORE re-home ever completes, stranding the
    // entity at the boundary; adversary-review HIGH). A recovered source still owns the subject → head
    // stays `source` → the reply keeps re-emitting until that source acks. (2) else RE-EMIT
    // `CrossingAborted` to the source, THROTTLED to `redrive_deadline_ticks` (cheap-idempotent cadence — an
    // alive-but-ack-stalled source must not draw a per-tick egress). The throttle is ONE runtime-level gate
    // (`last_abort_reply_emit`), so `PendingAbortReply` stays a 3-field record (no per-entry stamp, no ABA).
    // COLLECT-then-APPLY into Vecs so mutation happens after the shared borrow ends (determinism +
    // borrow-safety); `dir` is a disjoint param, so `dir.head` inside the `runtime`-values loop is sound.
    let reemit_due =
        now.0.saturating_sub(runtime.last_abort_reply_emit.0) >= tuning.redrive_deadline_ticks;
    let mut reaped: Vec<TransferId> = Vec::new();
    let mut to_emit: Vec<PendingAbortReply> = Vec::new();
    for reply in runtime.pending_abort_replies.values() {
        let owner = dir.head(reply.subject).map(|r| r.authority.node());
        if owner != Some(reply.source) {
            reaped.push(reply.transfer);
        } else if reemit_due {
            to_emit.push(*reply);
        }
    }
    if !to_emit.is_empty() {
        runtime.last_abort_reply_emit = now;
    }
    for reply in to_emit {
        outbox.push_flow(
            reply.source,
            MsgClass::Saga,
            &InterShardFlow::CrossingAborted(vd_wire::intershard::CrossingAborted {
                subject: reply.subject,
                transfer: reply.transfer,
            }),
        );
    }
    for transfer in reaped {
        runtime.pending_abort_replies.remove(&transfer);
        runtime
            .pending_writes
            .push((StoreKey::AbortReply(transfer).bytes(), None));
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

/// D-3 Slice 4 + Strong-AND split-brain fix: the CAP AND-gate deciding if a directory record may be REAPED.
/// All THREE must hold (never OR): (1) the post-restart quiesce window has elapsed (a rebuilt orchestrator
/// does not reap until its liveness evidence has had time to re-accrue); (2) `now` is past the SPLIT-BRAIN
/// DEADLINE `lease_expires + max_self_fence_grace_ticks` — the upper bound (across the two clock domains,
/// sized by [`THETA_MAX`](vd_sim::directory::THETA_MAX) so it dominates a CPU-throttled holder's self-fence)
/// on WHEN a partitioned holder has provably hard-stopped its own authority. This subsumes the lapse gate
/// (a live lease has `lease_expires >= now`) and is the CORE FIX: the old gate reaped at `lease_expires`
/// (=ttl), while a holder self-fences at `grace > ttl` — a verified ~ttl→grace two-holder split-brain
/// window. Reaping only at ttl+max means the holder is PROVABLY self-fenced first (zero zombie window).
/// (3) the owner is PERSISTENTLY confirmed dead — [`is_latched_dead`](LivenessTracker::is_latched_dead), the
/// MONOTONE latch (NOT the freshness-gated `is_confirmed_dead` pulse, which expires far before ttl+max once
/// the redial backoff spaces notices past the window). The evidence gate that (a) never reaps a slow-but-
/// alive holder and (b) realizes the binding CAP choice: the RAM tracker is empty on rehydrate ⇒ nobody
/// latched ⇒ an orchestrator outage FREEZES recovery, NEVER mass-orphans. Three monomorphic `if`s (HR5; no
/// short-circuit `&&`); each corner is covered.
#[must_use]
fn should_reap(
    record: &OwnerRecord,
    now: UniverseTick,
    liveness: &LivenessTracker,
    quiesced_until: UniverseTick,
    max_self_fence_grace_ticks: u64,
) -> bool {
    if now.0 < quiesced_until.0 {
        return false;
    }
    let reassign_after = record
        .lease_expires
        .0
        .saturating_add(max_self_fence_grace_ticks);
    if now.0 <= reassign_after {
        return false;
    }
    if !liveness.is_latched_dead(record.authority.node()) {
        return false;
    }
    true
}

/// D-3 Slice 4 + D-37 Slice 3: the orchestrator expiry REAPER. Once per `reaper_interval_ticks` (re-armed
/// via `last_reap_tick`, never per tick — an O(directory) sweep), resolve every lapsed-AND-confirmed-dead
/// lease, fanned out by key family:
/// - **`Session`** → FULLY REVOKE (the client reconnects via its ResumeTicket — a well-defined path).
/// - **`Entity` (UNLOCKED)** → enqueue a STANDING re-home (D-37 Slice 3): recovered onto a live shard via
///   [`process_rehome_starts`], NOT revoked — revoking would `HeldNowhere`-strand the entity. A LOCKED
///   `Entity` (a saga already owns the key, e.g. an in-flight CELL-1/2 re-home) is LEFT to that saga.
/// - **`Realm`/`Ship`** → LEFT (the durable Realm/Ship re-home is owed Slice 4; the dead realm is the honest
///   `RealmHeldNowhere` residual — revoking a player's ship realm would freeze the ship forever).
///
/// Collect-then-act (the immutable `entries()` borrow ends before the `revoke`/`pending_rehome` mutations).
/// INERT when `reaper_interval_ticks == 0` (the pre-D-3 default). Runs INSIDE the D-6 group-commit barrier
/// (before the directory reconcile, and immediately before `process_rehome_starts`) so a revoke / a locked +
/// armed re-home is captured by the same tick's reconcile + commit (durable — never resurrects on a kill-9).
/// Whether a LIVE saga already owns this subject key's recovery. The standing re-home MUST consult this,
/// not just `record.in_transfer`: `commit_cas` CLEARS `in_transfer` at the commit point while a
/// POST-commit `Promoting`/`ReHoming` saga lives on (it re-homes a dead committed owner via
/// `scan_deadlines`), so a dead-owner key can be UNLOCKED yet still owned by an in-flight saga. Without
/// this check the reaper would arm a SECOND re-home on that key — a double recovery-arm that steals the
/// lock and leaks a parked saga (audit `wf_3b9eb7f0`). This makes "one re-home arm per key" an ENFORCED
/// invariant (the in_transfer lock alone does not, post-commit). O(live sagas) per reapable key — the
/// reaper is interval-paced; a subject→saga index is the MMO-scale optimization, not built now.
fn subject_has_live_saga(sagas: &BTreeMap<TransferId, LiveSaga>, subject: DirectoryKey) -> bool {
    sagas.values().any(|s| s.ctx.subject == subject)
}

fn reap_lapsed_leases(runtime: &mut SagaRuntimeRes, dir: &mut DirectoryCore, now: UniverseTick) {
    let interval = dir.tuning().reaper_interval_ticks;
    if (interval == 0) | (now.0.saturating_sub(runtime.last_reap_tick.0) < interval) {
        return;
    }
    runtime.last_reap_tick = now;
    let quiesced_until = runtime.liveness_quiesced_until;
    // The split-brain reassign deadline term (D-3 Strong-AND): `should_reap` reaps only PAST
    // `lease_expires + max_self_fence_grace_ticks`, so a partitioned holder has provably self-fenced first.
    let max_grace = dir.tuning().max_self_fence_grace_ticks;
    // ONE pass over the past-deadline-AND-latched-dead leases; collect per family, then act.
    let mut to_revoke: Vec<(DirectoryKey, Fence)> = Vec::new();
    let mut to_rehome: Vec<PendingReHome> = Vec::new();
    for (key, record) in dir.entries() {
        if !should_reap(record, now, &runtime.liveness, quiesced_until, max_grace) {
            continue;
        }
        match key {
            DirectoryKey::Session(_) => to_revoke.push((*key, record.fence)),
            // Standing re-home a dead-owner Entity ONLY when it is BOTH unlocked AND has no live saga: a
            // post-commit `Promoting` saga's key is UNLOCKED (commit_cas cleared the lock) yet still owned
            // by that in-flight saga (which re-homes it itself) — arming a second re-home here would
            // double-arm + leak. Bitwise `&` (HR5: both predicates always evaluated, no short-circuit gap).
            DirectoryKey::Entity(_)
                if record.in_transfer.is_none() & !subject_has_live_saga(&runtime.sagas, *key) =>
            {
                to_rehome.push(PendingReHome {
                    subject: *key,
                    dead_owner: record.authority.node(),
                    prev_fence: record.fence,
                });
            }
            // A LOCKED Entity, an Entity a live saga still owns, a Realm, or a Ship → left for the owning
            // saga / Slice 4 (never double-armed).
            DirectoryKey::Entity(_) | DirectoryKey::Realm(_) | DirectoryKey::Ship(_) => {}
        }
    }
    for (key, fence) in to_revoke {
        let _ = dir.revoke(key, fence);
    }
    runtime.pending_rehome.extend(to_rehome);
}

/// CA-1 S3/S4 PRE-SCAN latch pass. Sets `dest_adopted = true` for every live saga whose dest's
/// `BatchAdopted` is present in THIS tick's inbox, BEFORE `scan_deadlines` runs. This is the intra-tick
/// ordering that makes the AwaitAdopt over-discard guarantee HARD: `scan_deadlines` runs before the inbox
/// drain, so latching in `deliver` (inside the drain) is one tick too late for a `BatchAdopted` that lands
/// on the exact budget-maturity tick — the discard would already have fired + tombstoned. Decodes only
/// enough to spot `BatchAdopted` (the drain re-decodes and acts on the same message); a non-Saga message, a
/// decode failure, a non-adopt arm, or an adopt for an absent/tombstoned saga is a skip (no side effect).
/// The latch is MONOTONE (never cleared) and RAM-only — the dest re-drives `BatchAdopted` every tick it
/// holds `Arriving`, so the latch is re-established every tick the ack lane delivers.
fn latch_adopted_from_inbox(runtime: &mut SagaRuntimeRes, inbox: &InboundBox) {
    for msg in &inbox.0 {
        if let Inbound::Wire {
            class: MsgClass::Saga,
            bytes,
            ..
        } = msg
            && let Ok(InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id, ..
            })) = postcard::from_bytes::<InterShardFlow>(bytes)
            && let Some(live) = runtime.sagas.get_mut(&transfer_id)
        {
            live.dest_adopted = true;
        }
    }
}

/// The orchestrator saga-runtime system: process new triggers, FIRE due deadlines (Slice 2a), then
/// drive every live saga forward on the gateway acks delivered this tick. Runs on the orchestrator's
/// single-threaded schedule; the directory CAS is a direct in-process call (no await, no lock across a send).
pub fn drive_sagas_core(
    inbox: Res<InboundBox>,
    clock: Res<ClockSample>,
    mut dir: ResMut<DirectoryRes>,
    mut runtime: ResMut<SagaRuntimeRes>,
    mut outbox: ResMut<OutboundBox>,
) {
    let now = clock.universe_tick;
    let epoch = clock.epoch;
    process_starts(&mut runtime, &mut dir.0, &mut outbox, epoch, now);
    // CA-1 S3/S4 — latch `dest_adopted` from THIS tick's inbox BEFORE `scan_deadlines` decides any
    // destructive resolution. The scan runs before the ack drain below, so a `BatchAdopted` arriving on the
    // exact budget-maturity tick would otherwise be seen too late (the discard would already have fired +
    // tombstoned). This pre-scan latch makes the AwaitAdopt over-discard guarantee HARD (see the latch doc).
    latch_adopted_from_inbox(&mut runtime, &inbox);
    // Slice 2a: fire due deadlines BEFORE the ack loop — a saga that loses its ack this tick still
    // gets its Timeout re-drive/abort next tick (the producer is the R1 backstop, never a wedge).
    scan_deadlines(&mut runtime, &mut dir.0, &mut outbox, epoch, now);
    for msg in &inbox.0 {
        // `from` is the transport ORIGIN — carried past the class filter because the transient crossing
        // consumer (3f-C) replies to it (a transient has no directory `OwnerRecord`, so the connection it
        // arrived on is the only authoritative reply address). The other saga-driving arms ignore it.
        let (from, class, bytes) = match msg {
            // D-3 CLEAR-ON-ACK: ANY successful inbound from a peer is proof it is alive — clear its
            // unreachable evidence (done at the TOP, BEFORE the class filter, so a peer that only sends
            // `LeaseRenew`/Directory ops — not saga acks — still un-marks itself; this is why no separate
            // `serve_directory` clear is needed: both systems read the same inbox, this one sees it all).
            Inbound::Wire { from, class, bytes } => {
                runtime.liveness.record_ack(*from);
                (*from, class, bytes)
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
            // R-4d M3: a LOCAL send shed says NOTHING about `to`'s liveness — routing it to
            // `record_unreachable` would false-confirm a live-but-ack-stalled peer dead and trip a
            // destructive re-home. Count it and CONTINUE; NEVER touch the liveness tracker, and NEVER
            // clear liveness evidence (a shed is orthogonal to the peer's inbound stream — `record_ack`
            // still fires only on `Inbound::Wire`).
            Inbound::SendShed { .. } => {
                runtime.sends_shed += 1;
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
            // Slice 3f-B: a DURABLE entity crossed a realm boundary — resolve the three heads and START a
            // crossing saga (or count the unresolved/subject-gone outcome). The subject/source/dest/session
            // all come from the directory; the id rides the WIRE fence so it matches the source latch.
            Ok(InterShardFlow::CrossingRequest(req)) => {
                handle_crossing_request(&mut runtime, &dir.0, &mut outbox, req);
            }
            // Slice 3f-C: a TRANSIENT crossed a realm boundary — resolve the dest realm and GRANT it back to
            // the transport-origin `from` (a transient has no directory owner to look up).
            Ok(InterShardFlow::TransientCrossingRequest(req)) => {
                handle_transient_crossing_request(&dir.0, &mut outbox, &mut runtime, req, from);
            }
            // Slice 3f-D (Mechanism Y): the SOURCE acked a crossing-abort reply — drop the pending entry +
            // stage its persist-DELETE (both ride this tick's group-commit barrier, so a kill after the ack
            // never re-emits). `is_some()`-gated so a REDELIVERED ack (the `ReDriven` class re-sends it on
            // every re-delivered `CrossingAborted`) is an idempotent no-op — never a double-DELETE stage.
            Ok(InterShardFlow::CrossingAbortedAck(ack)) => {
                if runtime
                    .pending_abort_replies
                    .remove(&ack.transfer)
                    .is_some()
                {
                    runtime
                        .pending_writes
                        .push((StoreKey::AbortReply(ack.transfer).bytes(), None));
                }
            }
            // Everything else (Ghost / Directory / Saga commands / DirectoryReply / FlushSource /
            // CrossingAborted — the orchestrator EMITS the abort reply, never consumes it) and any decode
            // failure: not a saga-driving inbound here.
            _ => {}
        }
    }
    // D-3 Slice 4: the expiry REAPER runs at the HEAD of the barrier — BEFORE the directory reconcile below
    // — so a revoke it makes lands in the SAME tick's `dirty` delta set and is captured by the incremental
    // reconcile + the single commit (the revoke is durable this tick as a DELETE; a reaped record never
    // resurrects on a kill-9 — the COMP-2 guarantee). It mutates the directory in RAM; the reconcile then
    // drains the delta set the mutation recorded.
    reap_lapsed_leases(&mut runtime, &mut dir.0, now);
    // D-37 Slice 3: ARM the standing re-homes the reaper just enqueued — SAME tick, still inside the
    // barrier, so the `lock_transfer` + the armed parked saga are captured by the reconcile + commit below
    // (durable this tick; on a kill-9 mid-window the RAM queue is lost but the still-dead UNLOCKED record
    // makes the reaper re-detect on reboot — see `PendingReHome`). Must run AFTER the reaper (it drains what
    // the reaper enqueued) and BEFORE the reconcile (so the lock/saga persist this tick).
    process_rehome_starts(&mut runtime, &mut dir.0, &mut outbox, epoch, now);
    // RLM Step 3e: the D-6 GROUP-COMMIT BARRIER was extracted to the [`commit_barrier`] system (below) so
    // the realm-lifecycle reconciler (`reconcile_realm_lifecycle`) can run BETWEEN this core and the ONE
    // fsync, staging its grant/revoke into the SAME `dir.0.dirty` set. The reaper + rehome-arm above STAY
    // here (they run before the reconcile); ONLY the drain/commit moved — the mutation phases are unchanged,
    // so persist-before-effect + the COMP-2 anti-zombie guarantees hold exactly as before.
}

/// The D-6 GROUP-COMMIT BARRIER body (RLM Step 3e extraction). Drains this tick's staged saga/go-token
/// writes (`pending_writes`), then this tick's incremental DIRECTORY deltas (`dir.dirty` — `Some` ⇒ PUT the
/// new snapshot, `None`/DELETE ⇒ the COMP-2 anti-zombie: a revoked/reaped record is deleted durably so
/// `rehydrate` can't resurrect it), then the durable clock ceiling, then ONE `commit()`. The `dirty` set
/// accumulates across `serve_directory` + `drive_sagas_core` + `reconcile_realm_lifecycle`, so this single
/// drain — the one place with both the store and the directory — captures every change this tick.
/// `O(changes)`, not `O(directory)`: a quiescent tick stages nothing.
fn group_commit(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    store: &mut (dyn Store + Send + Sync),
    clock_res: &UniverseClockRes,
) {
    for (key, value) in std::mem::take(&mut runtime.pending_writes) {
        match value {
            Some(bytes) => store.put(&key, &bytes),
            None => store.delete(&key),
        }
    }
    for (key, change) in dir.take_dirty() {
        match change {
            Some(record) => {
                let snapshot = DirSnapshot { key, record };
                store.put(&StoreKey::Directory(key).bytes(), &encode(&snapshot));
            }
            None => store.delete(&StoreKey::Directory(key).bytes()),
        }
    }
    store.put(
        &StoreKey::Clock.bytes(),
        &encode(&(clock_res.0.epoch(), clock_res.0.confirmed_ceiling())),
    );
    store.commit();
}

/// The orchestrator's FINAL chained system (RLM Step 3e): the D-6 group-commit barrier. Runs strictly AFTER
/// `drive_sagas_core` (the reaper + rehome-arm) and `reconcile_realm_lifecycle` (the RLM grant/revoke), so
/// ONE fsync captures every state change this tick and no effect leaves the orchestrator before the state
/// authorizing it is durable (persist-before-effect). Splitting the barrier out is a MOVE of the drain/
/// commit only — the mutation order is unchanged, so the D-6 + COMP-2 guarantees are preserved exactly.
pub fn commit_barrier(
    clock_res: Res<UniverseClockRes>,
    mut dir: ResMut<DirectoryRes>,
    mut runtime: ResMut<SagaRuntimeRes>,
    mut store: ResMut<StoreRes>,
) {
    group_commit(&mut runtime, &mut dir.0, &mut *store.0, &clock_res);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{NodeConfig, build_app};
    use crate::orchestrator::{OrchestratorConfig, register_orchestrator_with_store};
    use vd_core::MsgId;
    use vd_core::entity_kind::{DurabilityClass, EntityKind};
    use vd_core::glam::DVec3;
    use vd_core::pose::{FrameRef, RealmId};
    use vd_core::{EntityId, EpochId, Fence, SessionId, UniverseTick};
    use vd_sim::capability::NodeKind;
    use vd_sim::directory::DirectoryTuning;
    use vd_sim::io::mem::{MemHub, MemStore};
    use vd_sim::io::{ShedReason, Transport};
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

    /// The dest-frame pose a crossing/re-home now SHIPS: [`flushed_pose`] re-expressed into `TO_REALM`'s
    /// frame via the SAME `rebind_pose_to_dest` the builders apply, so the frame flips SystemSpace{7}→{8}
    /// through the P3 identity (position unchanged) and this stays byte-identical to the builder output.
    fn dest_flushed_pose() -> StampedPose {
        // `TO_REALM` is `System(8)` (a one-field frame) — no parent needed to name the dest frame.
        super::rebind_pose_to_dest(flushed_pose(), TO_REALM, None, &IdentityFrames)
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
            // `TO_REALM` is `System(8)` — a one-field frame, no parent needed.
            to_parent: None,
        }
    }

    /// A TRANSIENT batch saga ctx (D-7): the subject is the dest realm (inert provenance — never
    /// enters the directory, since `locks_directory_key(Transient)=false`); `expected_fence` is the
    /// dest realm-lease fence the batched go-token commits at; `session` is the session-less sentinel
    /// (`SessionId::NONE`) — faithful to production, which builds exactly this ctx in
    /// `handle_transient_crossing_request` (D-43 #9). The transient short-path never reads it.
    fn transient_ctx(expected_fence: Fence) -> SagaCtx {
        SagaCtx {
            subject: DirectoryKey::Realm(TO_REALM),
            session: SessionId::NONE,
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
            rlm: vd_sim::rlm::RlmTuning::default(),
        }
    }

    /// A throwaway RLM spawner for the D-6 crash/recover rigs (RLM is INERT in `orch_config` ⇒ never
    /// invoked; it only satisfies the `register_orchestrator_with_store` signature).
    fn test_spawner() -> Box<dyn vd_sim::io::RealmSpawner + Send + Sync> {
        Box::new(vd_sim::io::mem::MemSpawner::new(
            MemHub::new(),
            NodeId(1_000_000),
            8,
        ))
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
                test_spawner(),
                // RLM inert in these D-6 crash/recover rigs — no recovered launches.
                crate::rlm_runtime::LaunchSeed::new(),
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
                test_spawner(),
                // RLM inert in these D-6 crash/recover rigs — no recovered launches.
                crate::rlm_runtime::LaunchSeed::new(),
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
                test_spawner(),
                // RLM inert in these D-6 crash/recover rigs — no recovered launches.
                crate::rlm_runtime::LaunchSeed::new(),
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

        /// Grant an ARBITRARY directory key/owner/fence through the seam (the D-alpha oracle drives
        /// multiple keys + a higher-fence owner replace; `grant_subject` is the `subject()`+SOURCE special
        /// case). Routes through `serve_directory` → `DirectoryCore::grant` like any shard request.
        fn grant_key(&mut self, key: DirectoryKey, owner: AuthorityRef, fence: Fence) {
            self.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::Directory(
                            DirectoryOp::LeaseGrant { key, owner, fence },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// Revoke an arbitrary directory key at its exact fence (the logout / departed path; the D-alpha
        /// oracle uses it to drive the DELETE arm of the incremental reconcile).
        fn revoke_key(&mut self, key: DirectoryKey, fence: Fence) {
            self.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::Directory(
                            DirectoryOp::LeaseRevoke { key, fence },
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

        /// Slice 3f-B: a source shard ships a durable `CrossingRequest` (the boundary-detector egress). The
        /// orchestrator sees it as `Inbound::Wire { from: SOURCE, .. }` (the source is the transport origin).
        fn crossing_request(&mut self, req: CrossingRequest) {
            self.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::CrossingRequest(req))
                            .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// Slice 3f-C: a source shard ships a transient `TransientCrossingRequest`. The reply GRANT routes
        /// back to the transport origin (`SOURCE`), so a test drains it via [`drain_source`](Self::drain_source).
        fn transient_crossing_request(&mut self, req: TransientCrossingRequest) {
            self.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::TransientCrossingRequest(req))
                            .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
        }

        /// The immutable ctx of the ONE live saga keyed on `transfer` — the crossing tests assert its
        /// source/dest/session/transfer are the ones the resolver latched. Panics if absent (the test's
        /// contract is that the saga started).
        fn saga_ctx(&mut self, transfer: TransferId) -> SagaCtx {
            self.orch
                .world_mut()
                .resource::<SagaRuntimeRes>()
                .sagas
                .get(&transfer)
                .expect("saga live")
                .ctx
        }

        /// The `SagaState` of the ONE live saga keyed on `transfer` (the D-43 #9 transient-start test
        /// asserts it is parked in `BatchHandoff{AwaitAdopt}`). Panics if absent.
        fn saga_state(&mut self, transfer: TransferId) -> SagaState {
            self.orch
                .world_mut()
                .resource::<SagaRuntimeRes>()
                .sagas
                .get(&transfer)
                .expect("saga live")
                .state
        }

        /// Read a `SagaRuntimeRes` counter accessor (the crossing outcome counts).
        fn count(&mut self, read: impl Fn(&SagaRuntimeRes) -> u64) -> u64 {
            read(self.orch.world_mut().resource::<SagaRuntimeRes>())
        }
    }

    const CROSSING_SESSION: SessionId = SessionId(9);

    /// A durable `CrossingRequest` for `subject()` crossing FROM_REALM → TO_REALM at `subject_fence`,
    /// carrying `CROSSING_SESSION` (the source-supplied session — 3f-A).
    fn crossing_req(subject_fence: Fence) -> CrossingRequest {
        CrossingRequest {
            subject: subject(),
            from_realm: FROM_REALM,
            to_realm: TO_REALM,
            subject_fence,
            session: CROSSING_SESSION,
            attempt: 0,
            // A concrete parent so the threading through `handle_crossing_request` → `ctx.to_parent` is
            // ASSERTED (crossing_request_resolves_and_starts_the_saga). The value is arbitrary here
            // (TO_REALM=System(8) is nameable regardless); the assertion proves the field is not dropped.
            to_parent: Some(FROM_REALM),
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
                        pose: dest_flushed_pose(),
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
                    dest_adopted: false,
                    opened: UniverseTick(0),
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
    fn orchestrator_rehydrates_a_transient_saga_mid_await_release() {
        // D-43 #9 crash-safety (FIRST-TIME-REACHABLE persisted state): before the fix, a Transient saga
        // was NEVER persisted (the only `start_transfer` caller was the durable crossing handler), so a
        // `BatchHandoff{AwaitRelease}` `SagaSnapshot` becomes reachable only now. Persist a transient saga
        // mid-`AwaitRelease` (post-`BatchAdopted`), kill-9 + rehydrate, and assert it re-inserts under
        // `batch` with `SessionId::NONE` INTACT, re-drives its pending `TransientRelease` on the next tick,
        // and its go-token is restored — locking the newly-reachable transient persistence path.
        let mut rig = Rig::new();
        rig.trigger(transient_ctx(Fence(9)));
        rig.settle(); // → BatchHandoff{AwaitAdopt}, go-token committed
        rig.batch_adopted(XFER); // dest adopts → AwaitRelease, source told to TransientRelease
        {
            let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
            assert_eq!(runtime.live(), 1, "the transient saga is live pre-crash");
            assert_eq!(
                runtime.sagas.get(&XFER).expect("saga live").state,
                SagaState::BatchHandoff {
                    phase: BatchHandoffPhase::AwaitRelease,
                    new_fence: Fence(9),
                },
                "parked mid-AwaitRelease (the newly-reachable persisted transient state)"
            );
            assert_eq!(
                runtime.sagas.get(&XFER).expect("saga live").ctx.session,
                SessionId::NONE,
                "the session-less sentinel is what gets persisted"
            );
        }
        let _ = (rig.drain_source(), rig.drain_dest()); // clear the pre-crash egress

        rig.rebuild();
        {
            let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
            assert_eq!(
                runtime.live(),
                1,
                "the in-flight transient saga SURVIVED the kill-9 mid-AwaitRelease"
            );
            let live = runtime.sagas.get(&XFER).expect("re-inserted under batch");
            assert_eq!(
                live.state,
                SagaState::BatchHandoff {
                    phase: BatchHandoffPhase::AwaitRelease,
                    new_fence: Fence(9),
                },
                "the AwaitRelease phase + go-token fence rehydrated verbatim"
            );
            assert_eq!(
                live.ctx.session,
                SessionId::NONE,
                "the session-less sentinel survived the WAL round-trip intact"
            );
            assert_eq!(
                runtime.batch_goes(),
                vec![(BatchId(XFER), Fence(9))],
                "the committed go-token re-hydrated (the dest's authority stays backed)"
            );
        }

        // The rehydrated saga re-drives on the next tick: `since` is armed to 0, so the first
        // `scan_deadlines` fires a `Timeout` → re-emits the idempotent `TransientRelease` to the source.
        rig.settle();
        assert!(
            rig.drain_source().contains(&flow_inbound(&handoff(
                InterShardFlow::TransientRelease,
                TRANSIENT_RELEASE_STEP,
                Fence(9),
            ))),
            "the rebuilt orchestrator re-drove the in-flight transient handoff (re-emitted TransientRelease)"
        );
    }

    /// D-6 (D-alpha) DIFFERENTIAL ORACLE: assert the durable Directory family produced by the INCREMENTAL
    /// `dirty`-delta reconcile is byte-for-byte identical to a FULL delete-all-then-put-current reconcile.
    /// The full reconcile is, by construction, exactly the encoded snapshot of every CURRENT entry — so the
    /// expected map is computed directly from `entries()`. Any arm that changed a row without staging its
    /// delta (a stale durable snapshot, or a phantom row never deleted) shows up here as a mismatch.
    fn assert_incremental_matches_full_reconcile(rig: &mut Rig) {
        let actual: BTreeMap<Vec<u8>, Vec<u8>> = rig
            .store
            .scan(&[StoreKey::DIRECTORY])
            .into_iter()
            .map(|(k, v)| (k, v.to_vec()))
            .collect();
        let expected: BTreeMap<Vec<u8>, Vec<u8>> = rig
            .orch
            .world_mut()
            .resource::<DirectoryRes>()
            .0
            .entries()
            .map(|(k, r)| {
                (
                    StoreKey::Directory(*k).bytes(),
                    encode(&DirSnapshot {
                        key: *k,
                        record: *r,
                    })
                    .to_vec(),
                )
            })
            .collect();
        assert_eq!(
            actual, expected,
            "incremental reconcile diverged from the full reconcile"
        );
    }

    #[test]
    fn incremental_directory_reconcile_equals_a_full_reconcile_byte_for_byte() {
        // Drive PUT-new / PUT-refresh / PUT-replace / DELETE through the REAL group-commit barrier and pin
        // the differential oracle after each settle; the kill-9 rebuild proves rehydrate round-trips the
        // incremental durable set with no divergence (the load-bearing COMP-2 correctness for D-gamma).
        let other = DirectoryKey::Realm(RealmId::System(7));
        let mut rig = Rig::new();

        // PUT-new, two distinct keys → a multi-row durable family.
        rig.grant_subject(Fence(1));
        rig.grant_key(other, AuthorityRef::Shard(DEST), Fence(1));
        assert_incremental_matches_full_reconcile(&mut rig);

        // PUT-refresh (idempotent re-grant moves the lease) then PUT-replace (higher fence, new owner) —
        // both must OVERWRITE the prior durable snapshot, never leave a stale one behind.
        rig.grant_subject(Fence(1));
        rig.grant_key(subject(), AuthorityRef::Shard(DEST), Fence(7));
        assert_incremental_matches_full_reconcile(&mut rig);

        // DELETE: revoke the second key (the COMP-2 anti-zombie path through the barrier).
        rig.revoke_key(other, Fence(1));
        assert_incremental_matches_full_reconcile(&mut rig);

        // KILL-9 + REBUILD: rehydrate restores from the incremental durable set; entries() still equals it.
        rig.rebuild();
        assert_incremental_matches_full_reconcile(&mut rig);
    }

    #[test]
    fn directory_store_key_is_the_directory_family_tag_plus_postcard() {
        // D-delta: the crash test computes the sentinel pause prefix via this shim; it MUST equal the bytes
        // the barrier stages for a directory row ([DIRECTORY] ++ postcard(key)) or the writer pause misses.
        let key = DirectoryKey::Realm(RealmId::System(42));
        let bytes = directory_store_key(&key);
        assert_eq!(
            bytes[0],
            StoreKey::DIRECTORY,
            "leads with the DIRECTORY family tag"
        );
        assert_eq!(
            &bytes[1..],
            &postcard::to_allocvec(&key).expect("encode")[..],
            "tail is postcard(key)"
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
    fn a_handoff_past_its_whole_budget_is_reported_expired_instead_of_shielded() {
        // THE LEAK BOUND, exercised. Every other test reads the shield with an unbounded budget, which
        // only ever takes the "still shielded" side — so the arm that LIFTS the shield had never run.
        // That arm is the one an operator depends on: it is what distinguishes a hand-off that is slow
        // from one that is wedged, and it is what stops a single wedged crossing pinning a realm alive
        // for the lifetime of the process.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(
            vd_core::entity_kind::DurabilityClass::Durable,
            Fence(1),
        ));
        rig.settle();

        // Same live crossing, same instant — the ONLY difference is the budget it is measured against.
        let (shielded, expired) = rig
            .orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .arriving_dest_realms(UniverseTick(1), u64::MAX);
        assert_eq!(shielded, BTreeSet::from([TO_REALM]), "inside budget: held");
        assert!(expired.is_empty(), "inside budget: nothing to report");

        // A budget of ZERO would prove nothing: that is the DISARMED early return, which reports no
        // arrivals at all. The smallest ARMED budget is what puts a real hand-off past a real deadline.
        let (disarmed, none_reported) = rig
            .orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .arriving_dest_realms(UniverseTick(500), 0);
        // SPLIT, not `a && b`: a short-circuit leaves the right-hand side unevaluated whenever the left is
        // false, so one arm can never be reached and the crate cannot hit 100% (the project's own rule).
        assert!(disarmed.is_empty(), "a zero budget shields nothing");
        assert!(
            none_reported.is_empty(),
            "and reports nothing — disarmed is not expired"
        );

        let (shielded, expired) = rig
            .orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .arriving_dest_realms(UniverseTick(500), 1);
        assert!(
            shielded.is_empty(),
            "past budget the destination is NO LONGER held — the shield lifts, it does not linger"
        );
        assert_eq!(expired.len(), 1, "and the lift is REPORTED, never silent");
        // What a 2am operator needs to find it: which realm was being held, and for how long.
        assert_eq!(expired[0].realm, TO_REALM);
        // The phase is carried so the report names WHERE it wedged. Asserted as non-empty rather than as a
        // specific phase: which one a settled fixture parks in is an implementation detail of the rig, and
        // pinning it would make this test fail for reasons that have nothing to do with the shield.
        assert!(
            !expired[0].state.is_empty(),
            "the phase it is stuck in is carried"
        );

        // THE AGE IS A CLOCK READING, pinned by DIFFERENCE rather than by a magic number or a `>`. A
        // hardcoded age depends on how many ticks the fixture takes to settle; a comparison leaves a false
        // arm nothing can ever reach (the project's own HR5 rule — prefer an equality). Reading the same
        // wedged hand-off a hundred ticks later must report exactly a hundred ticks more, which is the real
        // property: the age tracks the universe clock tick for tick, and cannot be a constant.
        let (_, later) = rig
            .orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .arriving_dest_realms(UniverseTick(600), 1);
        assert_eq!(
            later[0].age_ticks.saturating_sub(expired[0].age_ticks),
            100,
            "the reported age advances with the clock"
        );
    }

    #[test]
    fn rehydrate_restores_the_arrival_shield_set() {
        // RESTART-SAFE WITHOUT EXTRA WORK: the shield needs no durable family of its own, because the
        // saga snapshot already persists the whole context including where the subject is going. A
        // rebuilt orchestrator re-derives the identical arrival set on its first sweep — so a crash
        // mid-crossing cannot leave a landing realm unprotected.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(
            vd_core::entity_kind::DurabilityClass::Durable,
            Fence(1),
        ));
        rig.settle();
        assert_eq!(
            rig.orch
                .world_mut()
                .resource::<SagaRuntimeRes>()
                .arriving_dest_realms(UniverseTick(1), u64::MAX)
                .0,
            BTreeSet::from([TO_REALM]),
            "the live crossing shields its destination before the crash"
        );

        rig.rebuild(); // KILL-9: the World's RAM is gone; only the committed WAL survives
        assert_eq!(
            rig.orch
                .world_mut()
                .resource::<SagaRuntimeRes>()
                .arriving_dest_realms(UniverseTick(1), u64::MAX)
                .0,
            BTreeSet::from([TO_REALM]),
            "and it shields the SAME destination after the rebuild, re-derived from the snapshot"
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
        runtime.liveness.record_unreachable(SOURCE, UniverseTick(1));
        assert!(
            !runtime.liveness.is_confirmed_dead(SOURCE, UniverseTick(1)),
            "the recovered orchestrator kept its configured n = 3 margin, not the n = 1 default"
        );
    }

    #[test]
    fn a_send_shed_is_counted_and_never_confirms_a_live_peer_dead() {
        // R-4d M3 regression (the false-confirm cure): a LOCAL send-shed toward a LIVE peer is a
        // transport backpressure/oversize refusal — it says NOTHING about that peer's liveness. It
        // must be counted (`sends_shed`) and NEVER routed to `record_unreachable`; otherwise a
        // live-but-ack-stalled peer accrues false death evidence and gets destructively re-homed.
        let mut rig = Rig::new();
        // Default margin is n = 1 (one notice confirms). Drive THREE sheds toward DEST — far past the
        // margin. If ANY leaked to `record_unreachable`, DEST would be confirmed dead.
        {
            let mut inbox = rig.orch.world_mut().resource_mut::<InboundBox>();
            inbox.0 = (0..3u64)
                .map(|i| Inbound::SendShed {
                    to: DEST,
                    class: MsgClass::Saga,
                    undelivered: MsgId(i),
                    reason: ShedReason::RetryBufferFull,
                })
                .collect();
        }
        // Run the schedule directly (NOT step_tick — its drain would clobber the seeded inbox, and the
        // MemHub cannot produce a shed): drive_sagas reads the seeded InboundBox in place.
        let (world, schedule) = rig.orch.parts_mut();
        schedule.run(world);

        let now = rig.orch.world_mut().resource::<ClockSample>().universe_tick;
        let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
        assert_eq!(
            runtime.sends_shed(),
            3,
            "every shed is counted on sends_shed"
        );
        assert_eq!(
            runtime.liveness_notices(),
            0,
            "no shed ever incremented liveness_notices — proof none reached record_unreachable"
        );
        assert!(
            !runtime.liveness.is_confirmed_dead(DEST, now),
            "a burst of sheds must NEVER confirm a live peer dead (the false-confirm cure)"
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
    fn should_reap_requires_quiesced_past_deadline_and_latched_dead() {
        // The Strong-AND CAP gate: reap iff quiesce-elapsed AND now PAST `lease_expires + max` AND
        // PERSISTENTLY (latched) confirmed dead. Covers all corners incl the `<=` deadline boundary.
        let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1 ⇒ latch on first notice
        let dead = NodeId(5);
        liveness.record_unreachable(dead, UniverseTick(100)); // latched (consecutive 1 >= n 1)
        let now = UniverseTick(200);
        let quiesced = UniverseTick(50); // window elapsed (now >= quiesced)
        let max = 10u64; // reassign horizon = lease_expires + 10
        // All three hold → reap (now 200 > lease_expires 90 + max 10 = 100; latched dead; quiesce elapsed).
        assert!(should_reap(&rec(dead, 90), now, &liveness, quiesced, max));
        // (1) NOT past the quiesce freeze → no reap.
        assert!(!should_reap(
            &rec(dead, 90),
            now,
            &liveness,
            UniverseTick(250),
            max
        ));
        // (2a) NOT past the reassign deadline (now <= lease_expires + max) → no reap.
        assert!(!should_reap(&rec(dead, 200), now, &liveness, quiesced, max));
        // (2b) EXACTLY at the deadline (now == lease_expires + max = 200) → the `<=` still blocks (boundary).
        assert!(!should_reap(&rec(dead, 190), now, &liveness, quiesced, max));
        // (3) NOT latched dead (a node with no unreachable evidence) → no reap.
        assert!(!should_reap(
            &rec(NodeId(99), 90),
            now,
            &liveness,
            quiesced,
            max
        ));
    }

    #[test]
    fn confirmed_dead_latch_sets_at_n_and_survives_a_window_reset_until_ack() {
        // The MONOTONE latch: set the tick consecutive reaches `n`, PRESERVED across a window reset (which
        // expires the freshness-gated pulse), cleared only by a live inbound. This is what lets `should_reap`
        // gate on it at the ttl+max horizon without the pulse having expired.
        let tuning = LivenessTuning {
            n_consecutive_unreachable: 3,
            unreachable_window_ticks: 10,
            retry_delay_ticks_hint: 2,
        };
        let mut lv = LivenessTracker::new(tuning);
        let node = NodeId(5);
        // Below n → not latched (and the pulse is not yet confirmed either).
        lv.record_unreachable(node, UniverseTick(0));
        lv.record_unreachable(node, UniverseTick(1));
        assert!(!lv.is_latched_dead(node), "2 < n=3: not yet latched");
        assert!(!lv.is_confirmed_dead(node, UniverseTick(1)));
        // The 3rd consecutive within the window → LATCHED (the pulse is also true right now).
        lv.record_unreachable(node, UniverseTick(2));
        assert!(
            lv.is_latched_dead(node),
            "consecutive reached n=3 → latched"
        );
        assert!(lv.is_confirmed_dead(node, UniverseTick(2)));
        // A notice FAR past the window RESETS the run (100 - 2 > window 10 ⇒ consecutive back to 1), so the
        // PULSE expires — but the monotone latch stays set (the whole point).
        lv.record_unreachable(node, UniverseTick(100));
        assert!(
            !lv.is_confirmed_dead(node, UniverseTick(100)),
            "pulse expired: consecutive reset to 1 < n"
        );
        assert!(
            lv.is_latched_dead(node),
            "the monotone latch survives the reset"
        );
        // A live inbound CLEARS the whole entry → un-latched.
        lv.record_ack(node);
        assert!(!lv.is_latched_dead(node), "record_ack un-latches");
    }

    #[test]
    fn should_reap_reaps_at_ttl_plus_max_after_the_pulse_expires_never_before() {
        // The Strong-AND split-brain guarantee + attack-1 orphan cure, together: (a) the reassign deadline is
        // `lease_expires + max` (NOT the old lapse@ttl that raced the holder's `grace` self-fence), so a
        // lapsed+latched owner is NOT reaped anywhere in (lease_expires, lease_expires+max]; (b) at the horizon
        // the freshness-gated PULSE has long expired, but the MONOTONE latch persists, so failover COMPLETES
        // (the key IS reaped) — no permanent orphan.
        let tuning = LivenessTuning {
            n_consecutive_unreachable: 1,
            unreachable_window_ticks: 10,
            retry_delay_ticks_hint: 2,
        };
        let mut lv = LivenessTracker::new(tuning);
        let dead = NodeId(5);
        lv.record_unreachable(dead, UniverseTick(5)); // latched (n = 1)
        let max = 250u64; // cloud @50Hz: reassign horizon = lease_expires + 250
        let rec = rec(dead, 100); // lease_expires = 100 ⇒ horizon = 350
        let quiesced = UniverseTick(0);
        // NOT reaped at the OLD ttl-ish point (100) nor anywhere up to and including the horizon (350).
        assert!(
            !should_reap(&rec, UniverseTick(100), &lv, quiesced, max),
            "old lapse@ttl must NOT reap"
        );
        assert!(!should_reap(&rec, UniverseTick(200), &lv, quiesced, max));
        assert!(
            !should_reap(&rec, UniverseTick(350), &lv, quiesced, max),
            "not reaped AT the horizon (<=)"
        );
        // At horizon+1 the PULSE is long dead (351 - 5 = 346 ≫ window 10) ...
        assert!(
            !lv.is_confirmed_dead(dead, UniverseTick(351)),
            "the pulse has expired"
        );
        // ... but the LATCH persists ⇒ should_reap reaps (failover completes, no orphan).
        assert!(should_reap(&rec, UniverseTick(351), &lv, quiesced, max));
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
    fn reap_in_freezing_orphan_enqueues_a_pending_rehome_and_arms_a_parked_saga() {
        // D-37 Slice 3 (CELL 3): a confirmed-dead + lapsed + UNLOCKED Entity orphan (the post-abort residual
        // of a SOURCE killed in Freezing) is DETECTED by the reaper (ENQUEUED, not revoked) and ARMED by
        // process_rehome_starts — a fresh re-home saga PARKS in ReHoming{target} (the lowest live capable
        // shard), the key is LOCKED (so the next sweep skips it), authority STAYS at the dead owner
        // (conservative — no CAS), and NOTHING is emitted (HR1: no fabricated pose/adopt — owed Slice 4).
        // Proves detection + capability-matched target selection + the durable parked saga + no fabrication.
        // Reverting the reaper Entity arm / process_rehome_starts / start_rehome turns this RED.
        let dead = NodeId(5);
        let target = NodeId(9);
        let entity = DirectoryKey::Entity(subject_eid());
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10,
            reaper_interval_ticks: 8,
            ..DirectoryTuning::default()
        });
        let _ = dir.grant(entity, AuthorityRef::Shard(dead), Fence(3), UniverseTick(0)); // lease_expires = 10
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // n = 1
        runtime.set_roster(
            [(
                target,
                ShardProfile::build(CapRequest::default()).expect("empty profile"),
            )]
            .into_iter()
            .collect(),
        );
        runtime.liveness.record_unreachable(dead, UniverseTick(50)); // dead CONFIRMED (n = 1)

        // REAP: the orphan is ENQUEUED (not revoked); the directory record is untouched.
        reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(100));
        assert_eq!(
            runtime.pending_rehome,
            vec![PendingReHome {
                subject: entity,
                dead_owner: dead,
                prev_fence: Fence(3),
            }],
            "the reaper enqueues the orphan for a standing re-home (does NOT revoke it)"
        );

        // ARM: a fresh re-home saga parks in ReHoming{target}; the key is locked; authority STAYS at corpse.
        let mut outbox = OutboundBox::default();
        process_rehome_starts(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(100),
        );
        assert!(
            runtime.pending_rehome.is_empty(),
            "the queue is drained the same tick (within-barrier hand-off)"
        );
        let transfer = rehome_transfer_id(entity, Fence(3));
        let live = runtime
            .sagas
            .get(&transfer)
            .expect("a fresh re-home saga was armed");
        assert_eq!(
            live.state,
            SagaState::ReHoming {
                target,
                prev_fence: Fence(3),
            },
            "the saga parks in ReHoming at the selected live target"
        );
        let head = dir
            .head(entity)
            .expect("the orphan record is still present");
        assert_eq!(
            head.authority,
            AuthorityRef::Shard(dead),
            "authority STAYS at the dead owner (conservative — no CAS, no HeldNowhere strand)"
        );
        assert_eq!(
            head.fence,
            Fence(3),
            "no fence bump (no ReHomeCommit until Slice 4)"
        );
        assert!(
            head.in_transfer.is_some(),
            "the key is LOCKED by the armed re-home saga (the next reaper sweep skips it)"
        );
        assert!(
            outbox.0.is_empty(),
            "NO fabrication: a parked standing re-home emits no ReHome envelope (the adopt is owed Slice 4)"
        );
    }

    #[test]
    fn reaper_leaves_a_locked_dead_entity_for_its_owning_saga() {
        // D-37 Slice 3: an Entity that is dead + lapsed but IN-TRANSFER-LOCKED (a live saga — e.g. an
        // in-flight CELL-1/2 re-home — owns the key) is NOT standing-re-homed; the owning saga's own recovery
        // handles it. Covers the reaper Entity-arm GUARD false branch (in_transfer.is_some()).
        let dead = NodeId(5);
        let entity = DirectoryKey::Entity(subject_eid());
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10,
            reaper_interval_ticks: 8,
            ..DirectoryTuning::default()
        });
        let _ = dir.grant(entity, AuthorityRef::Shard(dead), Fence(3), UniverseTick(0));
        assert!(
            dir.lock_transfer(entity, TransferId(1)),
            "a live saga locks the key"
        );
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        runtime.liveness.record_unreachable(dead, UniverseTick(50));
        reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(100));
        assert!(
            runtime.pending_rehome.is_empty(),
            "a LOCKED dead Entity is left to its owning saga, never standing-re-homed"
        );
    }

    #[test]
    fn reaper_leaves_a_dead_entity_a_live_post_commit_saga_still_owns() {
        // AUDIT wf_3b9eb7f0 (HIGH): commit_cas CLEARS in_transfer at the commit point, so a POST-commit
        // Promoting saga's key is UNLOCKED yet still owned by that live in-flight saga (which re-homes the
        // dead committed owner ITSELF via scan_deadlines). The reaper must NOT arm a SECOND standing
        // re-home on it — it cross-checks runtime.sagas (`subject_has_live_saga`), not just in_transfer, so
        // "one re-home arm per key" is an ENFORCED invariant. Reverting the `& !subject_has_live_saga`
        // guard turns this RED (the reaper would double-arm + leak a parked saga). Covers the
        // subject_has_live_saga TRUE arm + the guard's has-live-saga false branch.
        let dead = NodeId(5);
        let entity = DirectoryKey::Entity(subject_eid());
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10,
            reaper_interval_ticks: 8,
            ..DirectoryTuning::default()
        });
        // The committed key is UNLOCKED (a post-commit saga — commit_cas cleared in_transfer)...
        let _ = dir.grant(entity, AuthorityRef::Shard(dead), Fence(3), UniverseTick(0));
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        // ...but a LIVE Promoting saga still owns the subject (ctx.subject == entity).
        inject_saga(
            &mut runtime,
            SagaState::Promoting {
                new_fence: Fence(3),
                promote_acked: false,
                dest_delivered: false,
                rehome_target: None,
            },
            UniverseTick(0),
        );
        runtime.liveness.record_unreachable(dead, UniverseTick(50));
        reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(100));
        assert!(
            runtime.pending_rehome.is_empty(),
            "an unlocked dead-owner Entity a LIVE saga still owns is NOT double-armed (cross-checks sagas)"
        );
    }

    #[test]
    fn process_rehome_parks_when_no_live_target() {
        // D-37 Slice 3: with NO live capable shard (empty roster — whole-pool death) select_rehome_target
        // returns None and process_rehome_starts DROPS the entry: no saga, key stays UNLOCKED so the reaper
        // re-detects + retries next sweep (interval-paced, never a forced incapable re-home). Covers None.
        let dead = NodeId(5);
        let entity = DirectoryKey::Entity(subject_eid());
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10,
            reaper_interval_ticks: 8,
            ..DirectoryTuning::default()
        });
        let _ = dir.grant(entity, AuthorityRef::Shard(dead), Fence(3), UniverseTick(0));
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // EMPTY roster (default)
        runtime.liveness.record_unreachable(dead, UniverseTick(50));
        reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(100));
        assert_eq!(
            runtime.pending_rehome.len(),
            1,
            "the orphan was detected + enqueued"
        );
        let mut outbox = OutboundBox::default();
        process_rehome_starts(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(100),
        );
        assert_eq!(
            runtime.live(),
            0,
            "no live target → no saga armed (the entry is dropped, retried next sweep)"
        );
        assert!(
            dir.head(entity)
                .expect("record present")
                .in_transfer
                .is_none(),
            "the key stays UNLOCKED so the reaper re-detects + retries once a target appears"
        );
    }

    #[test]
    fn process_rehome_skips_an_already_locked_key() {
        // D-37 Slice 3: if the orphan key is ALREADY locked (a concurrent arm / a prior sweep's saga) when
        // process_rehome_starts drains it, lock_transfer returns false and the entry is skipped — one key,
        // one re-home saga. Covers the lock-false arm.
        let dead = NodeId(5);
        let target = NodeId(9);
        let entity = DirectoryKey::Entity(subject_eid());
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10,
            reaper_interval_ticks: 8,
            ..DirectoryTuning::default()
        });
        let _ = dir.grant(entity, AuthorityRef::Shard(dead), Fence(3), UniverseTick(0));
        assert!(
            dir.lock_transfer(entity, TransferId(1)),
            "pre-lock by another saga"
        );
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        runtime.set_roster(
            [(
                target,
                ShardProfile::build(CapRequest::default()).expect("empty profile"),
            )]
            .into_iter()
            .collect(),
        );
        runtime.pending_rehome.push(PendingReHome {
            subject: entity,
            dead_owner: dead,
            prev_fence: Fence(3),
        });
        let mut outbox = OutboundBox::default();
        process_rehome_starts(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(100),
        );
        assert_eq!(
            runtime.live(),
            0,
            "the already-locked key is skipped — no second re-home saga"
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
                dest_adopted: false,
                opened: UniverseTick(0),
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

    /// Build a live saga in `state` for the arrival-classifier tests.
    fn live_in(state: SagaState, class: vd_core::entity_kind::DurabilityClass) -> LiveSaga {
        LiveSaga {
            ctx: ctx(class, Fence(1)),
            state,
            gateway: GATEWAY,
            since: UniverseTick(0),
            flushed_pose: None,
            dead_observed_since: None,
            dest_adopted: false,
            opened: UniverseTick(0),
        }
    }

    #[test]
    fn arrival_dest_classifies_every_saga_shape() {
        use vd_core::entity_kind::DurabilityClass::Durable;
        // THE classification table. Every variant of the phase enum appears exactly once, so adding a
        // phase without deciding whether someone is arriving during it breaks this test AND the
        // wildcard-free match. `Done`/`Aborted` are unreachable through the live system (they tombstone
        // in the same barrier they are reached in) — they are covered here or nowhere.
        let arriving = [
            SagaState::AwaitProvision,
            SagaState::Preparing,
            SagaState::Cutting,
            SagaState::Freezing {
                marker_seq: 1,
                frozen_drained: None,
                flushed: false,
            },
            SagaState::CommittingCas {
                marker_seq: 1,
                drained_seq: 1,
            },
            SagaState::BatchCommitting {
                step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
            },
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitAdopt,
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
            SagaState::Releasing {
                new_fence: Fence(2),
            },
        ];
        for state in arriving {
            assert_eq!(
                arrival_dest(&live_in(state, Durable)),
                Some(TO_REALM),
                "somebody is on their way into the destination during {state:?}"
            );
        }

        let not_arriving = [
            // Compensating / terminal: the subject is going back to the source, or is already settled.
            SagaState::Aborting {
                reason: AbortReason::CasLost,
                awaiting_thaw: false,
                awaiting_abort_ack: false,
            },
            SagaState::Aborted {
                reason: AbortReason::CasLost,
            },
            SagaState::Done {
                new_fence: Fence(2),
            },
            // Re-home: a hand-off between MACHINES, not between PLACES.
            SagaState::ReHoming {
                target: DEST,
                prev_fence: Fence(1),
            },
            SagaState::Promoting {
                new_fence: Fence(2),
                promote_acked: false,
                dest_delivered: false,
                rehome_target: Some(DEST),
            },
        ];
        for state in not_arriving {
            assert_eq!(
                arrival_dest(&live_in(state, Durable)),
                None,
                "no arrival is pending during {state:?}"
            );
        }
    }

    #[test]
    fn arriving_dest_realms_excludes_a_parked_rehome_saga() {
        // THE LEAK THAT WOULD HAVE SHIPPED. A standing re-home fabricates its realm fields and parks
        // indefinitely BY DESIGN, so a naive "every live saga's destination" projection would pin one
        // made-up realm alive forever per orphaned entity, starting at the first machine failure.
        let mut rt = SagaRuntimeRes::default();
        rt.sagas.insert(
            XFER,
            LiveSaga {
                ctx: rehome_ctx(subject(), Fence(1), SOURCE, DEST),
                state: SagaState::ReHoming {
                    target: DEST,
                    prev_fence: Fence(1),
                },
                gateway: SOURCE,
                since: UniverseTick(0),
                flushed_pose: None,
                dead_observed_since: None,
                dest_adopted: false,
                opened: UniverseTick(0),
            },
        );
        assert_eq!(
            rt.arriving_dest_realms(UniverseTick(1), u64::MAX).0,
            BTreeSet::new()
        );
    }

    #[test]
    fn arriving_dest_realms_includes_a_transient_batch_dest() {
        // The gap the source-side keep-alive never covered at all: a batch of non-persistent things
        // (dropped items, debris) crossing over has no per-entity latch, so nothing was demanding its
        // destination. The shield is class-blind — ONE machinery, both durability classes (HR2).
        let mut rt = SagaRuntimeRes::default();
        rt.sagas.insert(
            XFER,
            live_in(
                SagaState::BatchHandoff {
                    phase: BatchHandoffPhase::AwaitPromote,
                    new_fence: Fence(2),
                },
                vd_core::entity_kind::DurabilityClass::Transient,
            ),
        );
        assert_eq!(
            rt.arriving_dest_realms(UniverseTick(1), u64::MAX).0,
            BTreeSet::from([TO_REALM]),
            "a transient batch's destination is shielded exactly like a player's"
        );
    }

    #[test]
    fn arriving_dest_realms_is_empty_after_the_saga_tombstones() {
        // The shield has no clearing path to forget because it is derived, not stored: the same
        // `remove` that tombstones the saga retires the shield, in the same barrier, for free.
        let mut rt = SagaRuntimeRes::default();
        rt.sagas.insert(
            XFER,
            live_in(
                SagaState::Promoting {
                    new_fence: Fence(2),
                    promote_acked: false,
                    dest_delivered: false,
                    rehome_target: None,
                },
                vd_core::entity_kind::DurabilityClass::Durable,
            ),
        );
        assert_eq!(
            rt.arriving_dest_realms(UniverseTick(1), u64::MAX).0,
            BTreeSet::from([TO_REALM])
        );
        rt.sagas.remove(&XFER);
        assert_eq!(
            rt.arriving_dest_realms(UniverseTick(1), u64::MAX).0,
            BTreeSet::new()
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
                dest_adopted: false,
                opened: UniverseTick(0),
            },
        );
    }

    fn flows_to_node(outbox: &OutboundBox, node: NodeId) -> Vec<InterShardFlow> {
        outbox
            .0
            .iter()
            .filter(|(to, _, _, _)| *to == node)
            .filter_map(|(_, _, b, _)| postcard::from_bytes(b).ok())
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
            select_rehome_target(
                &CapRequest::default(),
                &none_roster,
                &liveness,
                UniverseTick(0)
            ),
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
            select_rehome_target(
                &CapRequest::default(),
                &only_stub,
                &liveness,
                UniverseTick(0)
            ),
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
        let mut dos = Some((DEST, UniverseTick(5)));
        let ev = rehome_event_for(
            &promoting,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(30),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(ev, SagaEvent::Timeout);
        assert_eq!(dos, None, "a healthy dest clears the stale abort budget");

        // Confirm DEST dead (n == 1 → one notice confirms).
        liveness.record_unreachable(DEST, UniverseTick(0));
        // (ii) dest DEAD but WITHIN the abort budget → cheap Timeout re-drive (budget anchored at first fire).
        let mut dos = None;
        let ev = rehome_event_for(
            &promoting,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(0),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(ev, SagaEvent::Timeout);
        assert_eq!(
            dos,
            Some((DEST, UniverseTick(0))),
            "the abort budget anchors on the first dead observation (keyed by the dead node)"
        );
        let ev = rehome_event_for(
            &promoting,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(10),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(
            ev,
            SagaEvent::Timeout,
            "still within the 24-tick budget at tick 10"
        );

        // (iii) dest DEAD, PAST budget, a capable LIVE target exists → ReHomeTo{target}.
        let ev = rehome_event_for(
            &promoting,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(24),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(
            ev,
            SagaEvent::ReHomeTo { target: NodeId(9) },
            "past budget → forward re-home to the live target"
        );

        // (iv) dest DEAD, PAST budget, NO capable live target (empty roster) → Timeout (stay PARKED, honest).
        let empty_roster: BTreeMap<NodeId, ShardProfile> = BTreeMap::new();
        let mut dos = Some((DEST, UniverseTick(0)));
        let ev = rehome_event_for(
            &promoting,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(24),
            &empty_roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(
            ev,
            SagaEvent::Timeout,
            "no capable live target → the saga stays parked (honest)"
        );
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
        let _ = dir.grant(
            subject(),
            AuthorityRef::Shard(DEST),
            Fence(1),
            UniverseTick(0),
        );
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
            universe_epoch: EpochId(1),
            subject: subject(),
            new_fence: Fence(2),
            step_id: RE_HOME_STEP,
            state: ReHomeState::PoseOnly(dest_flushed_pose()),
            source: SOURCE,
        });
        assert!(
            flows_to_node(&outbox, target).contains(&expected),
            "the dedicated ReHome adopt was emitted to the target: {:?}",
            outbox.0
        );
    }

    #[test]
    fn a_rehomed_promoting_redrives_the_adopt_to_the_live_target_via_scan_deadlines() {
        // D-37 Slice 2d END-TO-END — the SELF-SUFFICIENT re-drive proven through the FULL producer chain
        // (scan_deadlines → rehome_event_for → deliver → FSM Some-arm → emit_rehome), NOT just the FSM unit.
        // A Promoting saga that ALREADY re-homed (`rehome_target: Some(target)`, the directory naming the
        // LIVE target at the bumped fence) is driven past the redrive deadline → rehome_event_for returns
        // Timeout (the owner is alive, keyed on `dir.head` not `ctx.dest`) → the FSM Some-arm emits
        // A::ReHomeAdopt → emit_rehome RE-SENDS the dedicated ReHome to the target from the stashed pose.
        // This proves the ORCHESTRATOR OWNS the adopt re-drive (no transport at-least-once needed) and is
        // the regression guard the CELL-2 crash matrix LACKS: the matrix's FIRST adopt lands over the perfect
        // FaultFabric link so it never runs this arm — reverting the Some-arm to always-Promote leaves the
        // matrix green but REDs this test (it would aim a Promote at the dead `ctx.dest`, not a ReHome at the
        // live target). Fence-monotone holds: re-driving the ADOPT re-bumps nothing (only ReHomeCommit bumps).
        let target = NodeId(9);
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive deadline = 8
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        });
        // POST-re-home directory state: the subject is committed to the LIVE target at the bumped Fence(2).
        let _ = dir.grant(
            subject(),
            AuthorityRef::Shard(target),
            Fence(2),
            UniverseTick(0),
        );
        inject_saga(
            &mut runtime,
            SagaState::Promoting {
                new_fence: Fence(2),
                promote_acked: false,
                dest_delivered: false,
                rehome_target: Some(target),
            },
            UniverseTick(0),
        );
        stash_flush(&mut runtime, XFER, flushed_pose()); // the adopt payload, re-read on every re-drive
        // The producer drives the DUE saga (now=8 >= redrive 8); a LIVE owner ⇒ Timeout (not ReHomeTo).
        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8),
        );
        // The dedicated ReHome adopt was RE-EMITTED to the live target from the stashed pose.
        let expected = InterShardFlow::ReHome(ReHomeCmd {
            transfer: XFER,
            universe_epoch: EpochId(1),
            subject: subject(),
            new_fence: Fence(2),
            step_id: RE_HOME_STEP,
            state: ReHomeState::PoseOnly(dest_flushed_pose()),
            source: SOURCE,
        });
        assert!(
            flows_to_node(&outbox, target).contains(&expected),
            "the re-drive re-sent the dedicated ReHome adopt to the live target: {:?}",
            outbox.0
        );
        // NOT a Promote at the (dead) original dest — the explicit regression guard for the Some-arm.
        assert!(
            flows_to_node(&outbox, DEST).is_empty(),
            "the re-drive does NOT aim a Promote at the dead original dest: {:?}",
            outbox.0
        );
        // Re-driving the ADOPT re-bumps nothing; the saga stays live (forward-only re-drive).
        assert_eq!(
            dir.head(subject()).expect("subject recorded").fence,
            Fence(2),
            "re-driving the adopt does not re-bump the fence (only ReHomeCommit bumps)"
        );
        assert_eq!(runtime.live(), 1, "the re-drive keeps the saga live");
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
            build_rehome(&realm_ctx, Fence(2), Some(flushed_pose()), EpochId(1)).is_none(),
            "non-Entity subject ⇒ None"
        );
        let entity_ctx = ctx(DurabilityClass::Durable, Fence(1)); // subject() is an Entity
        assert!(
            build_rehome(&entity_ctx, Fence(2), None, EpochId(1)).is_none(),
            "missing flushed pose ⇒ None"
        );
        assert!(
            build_rehome(&entity_ctx, Fence(2), Some(flushed_pose()), EpochId(1)).is_some(),
            "Entity subject + a stashed pose ⇒ Some"
        );
        // emit_rehome's None arm: a non-Entity re-home adopt emits NOTHING (the LOUD no-op).
        let mut outbox = OutboundBox::default();
        emit_rehome(
            &realm_ctx,
            Fence(2),
            NodeId(9),
            Some(flushed_pose()),
            EpochId(1),
            &mut outbox,
        );
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
        let _ = dir.grant(
            subject(),
            AuthorityRef::Shard(DEST),
            Fence(1),
            UniverseTick(0),
        );
        let _ = dir.commit_cas(
            subject(),
            Fence(1),
            AuthorityRef::Shard(SOURCE),
            UniverseTick(0),
        );
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
        assert_eq!(
            runtime.live(),
            0,
            "the re-home loser tombstoned (clean no-op)"
        );
        let head = dir.head(subject()).expect("subject still recorded");
        assert_eq!(
            head.authority,
            AuthorityRef::Shard(SOURCE),
            "the CAS winner still owns the entity"
        );
        assert_eq!(
            head.fence,
            Fence(2),
            "the re-home CAS did not bump (it lost)"
        );
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

        // SOURCE dead in a POST-ADOPT phase (AwaitPromote) → self-promote the dest (TransientDrop to
        // DEST), tombstone, count once. R-6d3c budget-gated the source-dead path too (so a source that
        // RESTARTS within budget delivers via its outbox replay FIRST): the first confirmed scan RE-DRIVES,
        // the resolution fires only past `abort_deadline_ticks` from the first observation.
        let mut runtime = mk();
        runtime.liveness.record_unreachable(SOURCE, UniverseTick(8));
        let mut outbox = OutboundBox::default();
        // First due scan: source CONFIRMED dead, but the restart-race budget has not elapsed → re-drive.
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
            "the source self-promote waits the restart-race budget — not fired on the first observation"
        );
        assert_eq!(runtime.source_unreachable_resolutions(), 0);
        // After `abort_deadline_ticks` from the first observation: the self-promote fires.
        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8 + saga::DEFAULT_ABORT_DEADLINE_TICKS),
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
        assert_eq!(
            runtime.batch_lost_source_crash(),
            0,
            "a POST-adopt source death is a zero-loss self-promote, NOT a counted loss"
        );

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
    fn rehome_event_for_await_adopt_source_dead_emits_pre_adopt_past_budget_else_redrives() {
        // R-6d3c producer discrimination (HR5: all corners in the monomorphic helper). A dead SOURCE in
        // BatchHandoff is now BUDGET-gated (so a restart-within-budget wins its outbox-replay race), and
        // the resolution event is PHASE-discriminated: AwaitAdopt → SourceUnreachablePreAdopt (accounted
        // loss); a post-adopt phase → SourceUnreachable (zero-loss self-promote). abort_deadline=24.
        let tuning = SagaTuning::default();
        let c = ctx(DurabilityClass::Transient, Fence(1));
        let await_adopt = SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitAdopt,
            new_fence: Fence(2),
        };
        let empty = ShardProfile::build(CapRequest::default()).expect("empty profile");
        let roster: BTreeMap<NodeId, ShardProfile> = [(NodeId(9), empty)].into_iter().collect();
        let req = CapRequest::default();
        let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1

        // (i) source HEALTHY → Timeout, and a stale budget is cleared.
        let mut dos = Some((SOURCE, UniverseTick(5)));
        let ev = rehome_event_for(
            &await_adopt,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(30),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(ev, SagaEvent::Timeout);
        assert_eq!(dos, None, "a healthy source clears the stale budget");

        // Confirm SOURCE dead (n == 1 → one notice confirms).
        liveness.record_unreachable(SOURCE, UniverseTick(0));
        // (ii) source DEAD but WITHIN the restart-race budget → cheap Timeout re-drive (budget anchored).
        let mut dos = None;
        let ev = rehome_event_for(
            &await_adopt,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(0),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(ev, SagaEvent::Timeout);
        assert_eq!(
            dos,
            Some((SOURCE, UniverseTick(0))),
            "the restart-race budget anchors on the first dead observation (keyed by the dead node)"
        );
        let ev = rehome_event_for(
            &await_adopt,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(10),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(
            ev,
            SagaEvent::Timeout,
            "still within the 24-tick budget at tick 10"
        );

        // (iii) source DEAD, PAST budget, PRE-adopt (AwaitAdopt) → SourceUnreachablePreAdopt (accounted
        // loss, discard-to-dest — NEVER self-promote an empty dest).
        let ev = rehome_event_for(
            &await_adopt,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(24),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(ev, SagaEvent::SourceUnreachablePreAdopt);
        assert_eq!(dos, None, "the resolution clears the budget");

        // (iv) source DEAD, PAST budget, POST-adopt (AwaitPromote) → SourceUnreachable (NOT PreAdopt) —
        // closes the mis-pairing gap (a post-adopt phase self-promotes; it must never discard an adopted
        // batch).
        let post_adopt = SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitPromote,
            new_fence: Fence(2),
        };
        let mut dos = Some((SOURCE, UniverseTick(0)));
        let ev = rehome_event_for(
            &post_adopt,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(24),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(
            ev,
            SagaEvent::SourceUnreachable,
            "a post-adopt phase resolves as a zero-loss self-promote, NOT a PreAdopt discard"
        );

        // (v) CA-1 S3/S4 OVER-DISCARD SAFETY: source DEAD, PAST budget, AwaitAdopt, `dest_adopted = true`,
        // and the dest is ALIVE (not confirmed dead) → `defer_to_dest` skips the source resolution and,
        // since the dest is alive, falls to the neutral re-drive → Timeout, NOT the discard. The batch the
        // dest actually holds is never over-discarded; the same-tick drain advances the phase off the
        // latched `BatchAdopted` and the post-adopt self-promote then handles the dead source. Deferring to
        // a LIVE dest clears the (now-moot) source budget anchor.
        let mut dos = Some((SOURCE, UniverseTick(0)));
        let ev = rehome_event_for(
            &await_adopt,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(24),
            &roster,
            &req,
            DEST,
            true,
        );
        assert_eq!(
            ev,
            SagaEvent::Timeout,
            "dest_adopted + live dest suppresses the pre-adopt discard (defer to the dest, re-drive)"
        );
        assert_eq!(
            dos, None,
            "deferring to a live dest clears the moot source budget anchor"
        );
    }

    #[test]
    fn rehome_event_for_reanchors_the_budget_on_a_source_dest_cause_switch() {
        // R-6d3c budget-gate regression guard (the post-impl review's blocker): `dead_observed_since` is
        // keyed by the confirmed-dead NODE, so a CAUSE-SWITCH (the DEST is confirmed dead → the dest
        // RECOVERS → the SOURCE is confirmed dead) RE-ANCHORS the destructive-resolution budget to `now`
        // instead of measuring the source's restart-race grace from the DEST's stale first-dead tick. A
        // POST-adopt phase (AwaitPromote) is used because its orch->source egress makes
        // `is_confirmed_dead(source)` reachable-NOW — the pre-fix shared anchor would fire early.
        let tuning = SagaTuning::default(); // abort_deadline = 24
        let c = ctx(DurabilityClass::Transient, Fence(1));
        let state = SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitPromote,
            new_fence: Fence(2),
        };
        let empty = ShardProfile::build(CapRequest::default()).expect("empty profile");
        let roster: BTreeMap<NodeId, ShardProfile> = [(NodeId(9), empty)].into_iter().collect();
        let req = CapRequest::default();
        let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1, window = 64
        let mut dos = None;

        // DEST confirmed dead at tick 0 → the budget anchors on DEST, within budget → Timeout.
        liveness.record_unreachable(DEST, UniverseTick(0));
        let ev = rehome_event_for(
            &state,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(0),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(ev, SagaEvent::Timeout);
        assert_eq!(
            dos,
            Some((DEST, UniverseTick(0))),
            "anchored on the dead DEST"
        );

        // CAUSE-SWITCH at tick 30 (already PAST 24 from the DEST's tick-0 anchor): the DEST RECOVERS and
        // the SOURCE is confirmed dead. The shared-anchor BUG would fire SourceUnreachable now (30-0 >= 24);
        // the keyed anchor RE-ANCHORS to (SOURCE, 30) and returns Timeout — the source gets its OWN budget.
        liveness.record_ack(DEST);
        liveness.record_unreachable(SOURCE, UniverseTick(30));
        let ev = rehome_event_for(
            &state,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(30),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(
            ev,
            SagaEvent::Timeout,
            "the cause-switch re-anchors — the source is NOT resolved off the dest's stale budget"
        );
        assert_eq!(
            dos,
            Some((SOURCE, UniverseTick(30))),
            "re-anchored on the newly-dead SOURCE"
        );

        // The source's OWN budget elapses at tick 54 (30 + 24) → the post-adopt self-promote fires.
        let ev = rehome_event_for(
            &state,
            &c,
            &liveness,
            &mut dos,
            &tuning,
            UniverseTick(54),
            &roster,
            &req,
            DEST,
            false,
        );
        assert_eq!(
            ev,
            SagaEvent::SourceUnreachable,
            "past the source's OWN re-anchored budget → the post-adopt self-promote fires"
        );
        assert_eq!(dos, None, "the resolution clears the budget");
    }

    #[test]
    fn scan_deadlines_resolves_an_await_adopt_batch_as_accounted_loss_and_counts_it() {
        // R-6d3c end-to-end producer (M2 — the scan_deadlines counter arm the `_ => {}` wildcard would
        // silently drop): an AwaitAdopt BatchHandoff whose SOURCE is confirmed dead, PAST the restart-race
        // budget, resolves as an ACCOUNTED loss — it emits `TransientDiscard` to the DEST, tombstones, and
        // increments `batch_lost_source_crash` (NOT the zero-loss `source_unreachable_resolutions`). A late
        // `BatchAdopted` for the tombstoned saga is then absorbed as a no-op (M1 — no second resolution).
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive=8, abort=24
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        });
        inject_saga(
            &mut runtime,
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitAdopt,
                new_fence: Fence(2),
            },
            UniverseTick(0),
        );
        runtime.liveness.record_unreachable(SOURCE, UniverseTick(8));

        // First due scan (tick 8): source confirmed dead but the restart-race budget has not elapsed → the
        // (AwaitAdopt, Timeout) re-drive, which CA-1 S3 makes emit the `ReSolicitBatch` liveness PROBE to the
        // SOURCE (the egress that makes `is_confirmed_dead(source)` reachable — proves the FSM arm + executor).
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
            1,
            "the accounted-loss discard waits the restart-race budget"
        );
        assert_eq!(runtime.batch_lost_source_crash(), 0);
        assert!(
            flows_to_node(&outbox, SOURCE).contains(&InterShardFlow::ReSolicitBatch(
                TransientHandoff {
                    transfer: XFER,
                    step_id: RE_SOLICIT_STEP,
                    fence: Fence(2),
                }
            )),
            "AwaitAdopt Timeout emits the source liveness probe: {:?}",
            outbox.0
        );

        // Past `abort_deadline_ticks` from the first observation: the discard-to-dest + count fires.
        let mut outbox = OutboundBox::default();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8 + saga::DEFAULT_ABORT_DEADLINE_TICKS),
        );
        assert_eq!(
            runtime.live(),
            0,
            "the accounted-loss resolution tombstoned the saga"
        );
        assert!(
            flows_to_node(&outbox, DEST).contains(&InterShardFlow::TransientDiscard(
                TransientHandoff {
                    transfer: XFER,
                    step_id: TRANSIENT_DISCARD_STEP,
                    fence: Fence(2),
                }
            )),
            "discard-to-dest (poison the late replay): {:?}",
            outbox.0
        );
        assert_eq!(
            runtime.batch_lost_source_crash(),
            1,
            "the never-restart PRE-adopt loss is counted (M2 — the scan_deadlines counter arm)"
        );
        assert_eq!(
            runtime.source_unreachable_resolutions(),
            0,
            "a PRE-adopt loss is NOT a zero-loss self-promote"
        );

        // M1: a LATE BatchAdopted for the now-tombstoned saga is a no-op — no second resolution, no egress.
        let mut outbox = OutboundBox::default();
        deliver(
            &mut runtime,
            &mut dir,
            &mut outbox,
            EpochId(1),
            UniverseTick(8 + saga::DEFAULT_ABORT_DEADLINE_TICKS + 1),
            XFER,
            SagaEvent::BatchAdopted,
        );
        assert_eq!(runtime.live(), 0, "the tombstoned saga stays gone");
        assert_eq!(
            runtime.batch_lost_source_crash(),
            1,
            "no double-count on the late ack"
        );
        assert!(
            outbox.0.is_empty(),
            "no second egress on the late BatchAdopted"
        );
    }

    #[test]
    fn latch_adopted_from_inbox_latches_only_a_present_sagas_batch_adopted() {
        // CA-1 S3/S4 — the pre-scan latch pass sets `dest_adopted` ONLY for a `BatchAdopted` (in a Saga-class
        // Wire) whose transfer names a LIVE saga. Covers every branch: a non-Saga inbound is skipped; a
        // Saga-class non-`BatchAdopted` arm is skipped; a `BatchAdopted` for an ABSENT saga is a no-op; a
        // `BatchAdopted` for the PRESENT XFER saga latches it.
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        inject_saga(
            &mut runtime,
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitAdopt,
                new_fence: Fence(2),
            },
            UniverseTick(0),
        );
        let batch_adopted = |t| {
            flow_inbound(&InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id: t,
                step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
            }))
        };
        let inbox = InboundBox(vec![
            // (1-false) a non-Saga-Wire inbound → skipped.
            Inbound::NodeUnreachable {
                to: SOURCE,
                class: MsgClass::Saga,
                undelivered: MsgId(0),
            },
            // (2-false) a Saga-class Wire that is NOT a BatchAdopted → skipped.
            demote_wire(Fence(2)),
            // (3-false) a BatchAdopted for an ABSENT saga → no-op (no such saga).
            batch_adopted(TransferId(999)),
            // (all-true) a BatchAdopted for the PRESENT XFER saga → latch it.
            batch_adopted(XFER),
        ]);
        latch_adopted_from_inbox(&mut runtime, &inbox);
        assert!(
            runtime.sagas.get(&XFER).expect("saga present").dest_adopted,
            "the present saga's dest_adopted latched from its BatchAdopted"
        );
    }

    #[test]
    fn pre_scan_latch_suppresses_the_await_adopt_over_discard_on_the_maturity_tick() {
        // CA-1 S3/S4 the HARD over-discard guarantee, end-to-end in the intra-tick order `drive_sagas`
        // performs (latch pre-scan → scan). On the EXACT budget-maturity tick, a racing `BatchAdopted` in the
        // inbox is latched BEFORE `scan_deadlines`, so the destructive pre-adopt discard is SUPPRESSED (the
        // saga survives + keeps probing). The CONTROL (no adopt evidence) fires the discard on the SAME tick
        // — proving the latch is precisely what averts the over-discard, not a budget accident.
        let mk = || {
            let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive=8, abort=24
            inject_saga(
                &mut runtime,
                SagaState::BatchHandoff {
                    phase: BatchHandoffPhase::AwaitAdopt,
                    new_fence: Fence(2),
                },
                UniverseTick(0),
            );
            runtime.liveness.record_unreachable(SOURCE, UniverseTick(8)); // source confirmed dead (n = 1)
            runtime
        };
        let mk_dir = || {
            DirectoryCore::new(DirectoryTuning {
                lease_ttl_ticks: 10_000,
                ..DirectoryTuning::default()
            })
        };
        // The budget anchors on the FIRST due scan and elapses `abort_deadline_ticks` later (dead_budget_
        // elapsed returns 0 on the anchoring call) — so, exactly like the accounted-loss test, an ANCHOR scan
        // at tick 8 precedes the MATURITY scan at 8 + abort_deadline where the discard is reachable.
        let anchor = UniverseTick(8);
        let maturity = UniverseTick(8 + saga::DEFAULT_ABORT_DEADLINE_TICKS);

        // --- WITH a racing BatchAdopted: the pre-scan latch suppresses the discard on the maturity tick ---
        let mut runtime = mk();
        let mut dir = mk_dir();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut OutboundBox::default(),
            EpochId(1),
            anchor,
        );
        let inbox = InboundBox(vec![flow_inbound(&InterShardFlow::TransferAck(
            TransferAck::BatchAdopted {
                transfer_id: XFER,
                step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
            },
        ))]);
        latch_adopted_from_inbox(&mut runtime, &inbox); // the pre-scan pass drive_sagas runs FIRST
        let mut outbox = OutboundBox::default();
        scan_deadlines(&mut runtime, &mut dir, &mut outbox, EpochId(1), maturity);
        assert_eq!(
            runtime.live(),
            1,
            "the racing BatchAdopted latched dest_adopted → the discard is suppressed, the saga survives"
        );
        assert_eq!(runtime.batch_lost_source_crash(), 0, "NO over-discard");
        assert!(
            flows_to_node(&outbox, DEST).is_empty(),
            "no discard (indeed no egress at all) to the dest — the batch it holds is untouched: {:?}",
            outbox.0
        );
        assert!(
            flows_to_node(&outbox, SOURCE).contains(&InterShardFlow::ReSolicitBatch(
                TransientHandoff {
                    transfer: XFER,
                    step_id: RE_SOLICIT_STEP,
                    fence: Fence(2),
                }
            )),
            "the suppressed tick still re-drives the source probe: {:?}",
            outbox.0
        );

        // --- CONTROL: no adopt evidence in the inbox → the GENUINE pre-adopt discard fires the same tick ---
        let mut runtime = mk();
        let mut dir = mk_dir();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut OutboundBox::default(),
            EpochId(1),
            anchor,
        );
        latch_adopted_from_inbox(&mut runtime, &InboundBox::default()); // empty inbox → nothing latched
        let mut outbox = OutboundBox::default();
        scan_deadlines(&mut runtime, &mut dir, &mut outbox, EpochId(1), maturity);
        assert_eq!(
            runtime.live(),
            0,
            "no adopt evidence → the pre-adopt loss discard fires + tombstones"
        );
        assert_eq!(
            runtime.batch_lost_source_crash(),
            1,
            "the accounted loss is counted"
        );
        assert!(
            flows_to_node(&outbox, DEST).contains(&InterShardFlow::TransientDiscard(
                TransientHandoff {
                    transfer: XFER,
                    step_id: TRANSIENT_DISCARD_STEP,
                    fence: Fence(2),
                }
            )),
            "discard-to-dest poisons a late replay: {:?}",
            outbox.0
        );

        // --- DOUBLE-CRASH (dest adopted THEN crashed + source also dead): resolve as the dest-dead abandon,
        // NOT a wedge. This is the regression the post-impl review caught: making the source probe live must
        // NOT let source-death preempt the dest-dead terminal into a perpetual Timeout loop. `defer_to_dest`
        // routes the adopted-then-dead dest to the budget-gated `DestUnreachable` abandon + tombstone.
        let mut runtime = mk();
        runtime.liveness.record_unreachable(DEST, UniverseTick(8)); // the dest is ALSO confirmed dead
        runtime
            .sagas
            .get_mut(&XFER)
            .expect("saga present")
            .dest_adopted = true; // it had adopted before dying
        let mut dir = mk_dir();
        scan_deadlines(
            &mut runtime,
            &mut dir,
            &mut OutboundBox::default(),
            EpochId(1),
            anchor,
        ); // anchors the DEST budget
        let mut outbox = OutboundBox::default();
        scan_deadlines(&mut runtime, &mut dir, &mut outbox, EpochId(1), maturity);
        assert_eq!(
            runtime.live(),
            0,
            "both dead → the dest-dead abandon terminates the saga (no wedge)"
        );
        assert_eq!(
            runtime.dest_unreachable_resolutions(),
            1,
            "counted as a dest-dead resolution (admin-visible), NOT a silent park"
        );
        assert_eq!(
            runtime.batch_lost_source_crash(),
            0,
            "an adopted-then-dead dest is NOT a source-crash discard"
        );
        assert!(
            flows_to_node(&outbox, SOURCE).contains(&InterShardFlow::TransientAbandon(
                TransientHandoff {
                    transfer: XFER,
                    step_id: TRANSIENT_ABANDON_STEP,
                    fence: Fence(2),
                }
            )),
            "abandon-to-source (the promote target is gone): {:?}",
            outbox.0
        );
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
            !runtime
                .liveness
                .is_confirmed_dead(NodeId(9), UniverseTick(1)),
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
        let (to, class, bytes, _) = &outbox.0[0];
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
                    pose: dest_flushed_pose(),
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

    // ── Slice 3f-B — the durable `CrossingRequest` consumer ────────────────────────────────────────

    #[test]
    fn crossing_request_starts_a_tagged_saga() {
        // All THREE heads resolve (subject@F/SOURCE, Realm(to)@DEST, Session@GATEWAY) → a durable crossing
        // saga starts, keyed on the id the SOURCE latched (`crossing_transfer_id(subject, F)` over the WIRE
        // fence), with source/dest/session/gateway resolved from the directory.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1)); // Entity(subject())@F(1) owned by Shard(SOURCE)
        rig.grant_key(
            DirectoryKey::Realm(TO_REALM),
            AuthorityRef::Shard(DEST),
            Fence(5),
        );
        rig.grant_key(
            DirectoryKey::Session(CROSSING_SESSION),
            AuthorityRef::Gateway(GATEWAY),
            Fence(3),
        );
        rig.crossing_request(crossing_req(Fence(1))); // tick N: enqueue the start
        rig.settle(); // tick N+1: process_starts inserts the saga

        let latched = crossing_transfer_id(subject(), Fence(1), 0);
        let ctx = rig.saga_ctx(latched);
        assert_eq!(ctx.transfer, latched);
        assert_eq!(ctx.transfer, crossing_transfer_id(subject(), Fence(1), 0));
        assert_eq!(ctx.source, SOURCE);
        assert_eq!(ctx.dest, DEST);
        assert_eq!(ctx.session, CROSSING_SESSION);
        assert_eq!(ctx.expected_fence, Fence(1));
        assert_eq!(ctx.class, DurabilityClass::Durable);
        assert_eq!(ctx.from_realm, FROM_REALM);
        assert_eq!(ctx.to_realm, TO_REALM);
        // The parent-provenance the SOURCE detector supplied was threaded VERBATIM onto the saga ctx (so
        // `build_crossing`'s `rebind_pose_to_dest` can form an Area frame) — never dropped at the resolver.
        assert_eq!(ctx.to_parent, Some(FROM_REALM));
        assert_eq!(rig.live(), 1);
        assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 1);
        assert_eq!(rig.count(SagaRuntimeRes::crossing_unresolved), 0);
        assert_eq!(rig.count(SagaRuntimeRes::crossing_subject_gone), 0);
    }

    #[test]
    fn crossing_request_unresolved_dest_counted() {
        // Subject + session resolve, but Realm(to) is ABSENT → no saga; counted `crossing_unresolved`.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.grant_key(
            DirectoryKey::Session(CROSSING_SESSION),
            AuthorityRef::Gateway(GATEWAY),
            Fence(3),
        );
        rig.crossing_request(crossing_req(Fence(1)));
        rig.settle();

        assert_eq!(rig.live(), 0);
        assert_eq!(rig.count(SagaRuntimeRes::crossing_unresolved), 1);
        assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 0);
        assert_eq!(rig.count(SagaRuntimeRes::crossing_subject_gone), 0);
    }

    #[test]
    fn crossing_request_unresolved_session_counted() {
        // Subject + dest realm resolve, but Session is ABSENT → no saga; counted `crossing_unresolved`.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.grant_key(
            DirectoryKey::Realm(TO_REALM),
            AuthorityRef::Shard(DEST),
            Fence(5),
        );
        rig.crossing_request(crossing_req(Fence(1)));
        rig.settle();

        assert_eq!(rig.live(), 0);
        assert_eq!(rig.count(SagaRuntimeRes::crossing_unresolved), 1);
        assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 0);
        assert_eq!(rig.count(SagaRuntimeRes::crossing_subject_gone), 0);
    }

    #[test]
    fn crossing_request_vanished_subject_counted() {
        // The subject has NO directory owner (authority already moved/revoked) → `crossing_subject_gone`,
        // even though the dest realm + session are present (the subject arm is checked first).
        let mut rig = Rig::new();
        rig.grant_key(
            DirectoryKey::Realm(TO_REALM),
            AuthorityRef::Shard(DEST),
            Fence(5),
        );
        rig.grant_key(
            DirectoryKey::Session(CROSSING_SESSION),
            AuthorityRef::Gateway(GATEWAY),
            Fence(3),
        );
        rig.crossing_request(crossing_req(Fence(1)));
        rig.settle();

        assert_eq!(rig.live(), 0);
        assert_eq!(rig.count(SagaRuntimeRes::crossing_subject_gone), 1);
        assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 0);
        assert_eq!(rig.count(SagaRuntimeRes::crossing_unresolved), 0);
    }

    #[test]
    fn crossing_request_redelivery_is_one_saga() {
        // A redelivered request for the SAME fenced crossing resolves to ONE saga. Because the first request
        // has already SETTLED (its saga inserted by `process_starts`), the redelivery hits the handler's
        // `contains_key(&transfer)` guard arm → no second `start_transfer`, no second count: `crossings_started`
        // stays 1 and reflects DISTINCT started sagas, not requests seen.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.grant_key(
            DirectoryKey::Realm(TO_REALM),
            AuthorityRef::Shard(DEST),
            Fence(5),
        );
        rig.grant_key(
            DirectoryKey::Session(CROSSING_SESSION),
            AuthorityRef::Gateway(GATEWAY),
            Fence(3),
        );
        rig.crossing_request(crossing_req(Fence(1)));
        rig.settle(); // the saga starts + locks the subject
        rig.crossing_request(crossing_req(Fence(1))); // redelivery
        rig.settle();

        assert_eq!(rig.live(), 1, "the redelivery never spawns a second saga");
        let latched = crossing_transfer_id(subject(), Fence(1), 0);
        assert_eq!(rig.saga_ctx(latched).transfer, latched);
        assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 1);
    }

    // ── Slice 3f-C — the transient `TransientCrossingRequest` consumer ─────────────────────────────

    /// A transient crossing request for `subject()` into TO_REALM at `src_realm_fence`. Carries a concrete
    /// `to_parent` so the orchestrator's VERBATIM copy into the `TransientCrossingGrant` is asserted below.
    fn transient_req(src_realm_fence: Fence) -> TransientCrossingRequest {
        TransientCrossingRequest {
            subject: subject(),
            from_realm: FROM_REALM,
            to_realm: TO_REALM,
            src_realm_fence,
            to_parent: Some(FROM_REALM),
        }
    }

    /// The wire form of the `TransientCrossingGrant` the source expects back from the orchestrator.
    fn grant_wire(grant: TransientCrossingGrant) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(&InterShardFlow::TransientCrossingGrant(grant))
                .expect("encode")
                .into(),
        }
    }

    #[test]
    fn transient_request_grants_resolved_dest() {
        // Realm(to) resolves to DEST@RF → ONE grant back to the transport-origin (SOURCE), carrying the
        // resolved dest, the current realm-lease fence, and the deterministic per-subject batch id.
        let mut rig = Rig::new();
        let realm_fence = Fence(4);
        rig.grant_key(
            DirectoryKey::Realm(TO_REALM),
            AuthorityRef::Shard(DEST),
            realm_fence,
        );
        // Discard the `LeaseGrant`'s directory-reply (seeding artifact) so the drain below is the grant only.
        let _ = rig.drain_source();
        rig.transient_crossing_request(transient_req(Fence(2)));

        let batch = crossing_transfer_id(subject(), Fence(2), 0);
        assert_eq!(
            rig.drain_source(),
            vec![grant_wire(TransientCrossingGrant {
                subject: subject(),
                dest: DEST,
                to_realm: TO_REALM,
                dst_realm_fence: realm_fence,
                batch,
                // The orchestrator copied the request's parent VERBATIM into the grant (asserted by this
                // full-value equality) — so the source can stamp it onto `TransientStatus::Crossing`.
                to_parent: Some(FROM_REALM),
            })]
        );
        assert_eq!(rig.count(SagaRuntimeRes::transient_crossings_granted), 1);
        assert_eq!(rig.count(SagaRuntimeRes::transient_dest_unresolved), 0);
    }

    #[test]
    fn transient_request_grant_starts_a_batchhandoff_awaitadopt_saga() {
        // D-43 #9 (THE fix): the resolved grant ALSO starts the transient `BatchHandoff` saga keyed on
        // `batch`, parked in `AwaitAdopt` awaiting the dest's `BatchAdopted` — so the downstream adopt
        // lands on a LIVE saga (not the silent None early-return). The go-token committed at the dest
        // realm-lease fence, and the grant emit is UNCHANGED (still counted once).
        let mut rig = Rig::new();
        let realm_fence = Fence(4);
        rig.grant_key(
            DirectoryKey::Realm(TO_REALM),
            AuthorityRef::Shard(DEST),
            realm_fence,
        );
        let _ = rig.drain_source(); // discard the LeaseGrant reply (seeding artifact)
        rig.transient_crossing_request(transient_req(Fence(2))); // tick N: grant + enqueue the start
        rig.settle(); // tick N+1: process_starts inserts the BatchHandoff saga → drives to AwaitAdopt

        let batch = crossing_transfer_id(subject(), Fence(2), 0);
        // EXACTLY ONE live saga, under `batch`.
        assert_eq!(rig.live(), 1, "the grant started exactly one saga");
        let ctx = rig.saga_ctx(batch);
        assert_eq!(
            rig.saga_state(batch),
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitAdopt,
                new_fence: realm_fence,
            },
            "parked in AwaitAdopt at the dest realm-lease fence"
        );
        assert_eq!(ctx.class, DurabilityClass::Transient);
        assert_eq!(ctx.source, SOURCE, "source == the transport-origin (from)");
        assert_eq!(ctx.dest, DEST, "dest == the resolved realm owner");
        assert_eq!(
            ctx.session,
            SessionId::NONE,
            "a session-less transient carries the typed sentinel, not a real id"
        );
        // The batched go-token committed at the realm-lease fence (G-TIER: one write per batch).
        assert_eq!(
            rig.orch
                .world_mut()
                .resource::<SagaRuntimeRes>()
                .batch_goes(),
            vec![(BatchId(batch), realm_fence)],
            "the go-token was written (was [] before the fix)"
        );
        // The grant leg is UNCHANGED: still exactly one grant counted.
        assert_eq!(rig.count(SagaRuntimeRes::transient_crossings_granted), 1);
    }

    #[test]
    fn transient_request_unknown_realm_drops() {
        // No Realm(to) record → no grant emitted; counted `transient_dest_unresolved`.
        let mut rig = Rig::new();
        rig.transient_crossing_request(transient_req(Fence(2)));

        assert!(
            rig.drain_source().is_empty(),
            "an unresolved dest realm emits no grant"
        );
        assert_eq!(rig.count(SagaRuntimeRes::transient_dest_unresolved), 1);
        assert_eq!(rig.count(SagaRuntimeRes::transient_crossings_granted), 0);
    }

    #[test]
    fn transient_request_redelivery_regrants_same_batch() {
        // A redelivered request re-grants the SAME deterministic batch id (idempotent at the source).
        let mut rig = Rig::new();
        rig.grant_key(
            DirectoryKey::Realm(TO_REALM),
            AuthorityRef::Shard(DEST),
            Fence(4),
        );
        // Discard the `LeaseGrant`'s directory-reply (seeding artifact) so both drains are the grant only.
        let _ = rig.drain_source();
        rig.transient_crossing_request(transient_req(Fence(2)));
        let first = rig.drain_source();
        rig.transient_crossing_request(transient_req(Fence(2)));
        let second = rig.drain_source();

        assert_eq!(
            first, second,
            "the redelivery re-grants the identical batch"
        );
        assert_eq!(rig.count(SagaRuntimeRes::transient_crossings_granted), 2);
        // D-43 #9 idempotency (the `if !contains_key` FALSE arm): the re-request did NOT start a second
        // saga — one batch, one `BatchHandoff` saga, absorbed by the guard.
        assert_eq!(rig.live(), 1, "one batch, one saga (idempotent re-grant)");
    }

    // ───────────────────────── Slice 3f-D: durable crossing-abort core (Mechanism Y) ─────────────────────

    /// A crossing-origin `TransferId` (id-namespaced `0x39`) the source's latch keys on — the SAME value
    /// `handle_crossing_request` mints from `(subject, subject_fence, attempt)`.
    fn crossing_xfer() -> TransferId {
        crossing_transfer_id(subject(), Fence(1), 0)
    }

    /// Inject a live saga under a SPECIFIC transfer id (its `ctx.transfer` set to `transfer`, source
    /// SOURCE / dest DEST / subject `subject()`) with the given class + state — for the `commit_result`
    /// crossing-abort tests (the id's high byte is the `is_crossing_origin` discriminator).
    fn inject_saga_id(
        runtime: &mut SagaRuntimeRes,
        transfer: TransferId,
        class: DurabilityClass,
        state: SagaState,
    ) {
        let mut ctx = ctx(class, Fence(1));
        ctx.transfer = transfer;
        runtime.sagas.insert(
            transfer,
            LiveSaga {
                ctx,
                state,
                gateway: GATEWAY,
                since: UniverseTick(0),
                flushed_pose: None,
                dead_observed_since: None,
                dest_adopted: false,
                opened: UniverseTick(0),
            },
        );
    }

    /// The terminal `Aborted` state (a pre-CAS failure) a tombstoning saga carries.
    fn aborted_state() -> SagaState {
        SagaState::Aborted {
            reason: AbortReason::CutTimeout,
        }
    }

    /// Count the `pending_writes` entries staging THIS abort-reply key with the given put/delete shape
    /// (`Some` = PUT, `None` = DELETE). Value-equality on the key bytes — no key-parse.
    fn abort_reply_writes(runtime: &SagaRuntimeRes, transfer: TransferId, put: bool) -> usize {
        let key = StoreKey::AbortReply(transfer).bytes();
        runtime
            .pending_writes
            .iter()
            .filter(|(k, v)| *k == key && v.is_some() == put)
            .count()
    }

    #[test]
    fn store_key_abort_reply_tag_is_five_and_distinct() {
        // D0: the AbortReply family tag is 5 and disjoint from every prior family (1..=4).
        let bytes = StoreKey::AbortReply(crossing_xfer()).bytes();
        assert_eq!(bytes[0], 5, "AbortReply tag is 5");
        let tags = [
            StoreKey::Saga(XFER).bytes()[0],
            StoreKey::BatchGo(BatchId(XFER)).bytes()[0],
            StoreKey::Directory(subject()).bytes()[0],
            StoreKey::Clock.bytes()[0],
            StoreKey::AbortReply(XFER).bytes()[0],
        ];
        let mut distinct = tags.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            tags.len(),
            "all five family tags are distinct"
        );
    }

    #[test]
    fn pending_abort_reply_roundtrips() {
        // D0: the self-describing record encodes + decodes byte-identically (all three fields).
        let reply = PendingAbortReply {
            transfer: crossing_xfer(),
            source: SOURCE,
            subject: subject(),
        };
        let bytes = encode(&reply);
        let back: PendingAbortReply =
            postcard::from_bytes(&bytes).expect("decode PendingAbortReply");
        assert_eq!(back, reply);
    }

    #[test]
    fn is_crossing_origin_is_true_only_for_the_crossing_namespace() {
        // D1: the stateless tag-check — `0x39` crossing is EXCLUSIVE of re-home (`0x37`) and the
        // connection-plane's small high-byte-`0x00` ids.
        let mut c = ctx(DurabilityClass::Durable, Fence(1));
        c.transfer = crossing_xfer();
        assert!(is_crossing_origin(&c), "a 0x39 id IS crossing-origin");
        c.transfer = namespaced_transfer_id(0x37, b"rehome"); // re-home namespace
        assert!(
            !is_crossing_origin(&c),
            "a 0x37 re-home id is NOT crossing-origin"
        );
        c.transfer = TransferId(1); // a small connection-plane id (high byte 0x00)
        assert!(
            !is_crossing_origin(&c),
            "a high-byte-0x00 id is NOT crossing-origin"
        );
    }

    #[test]
    fn commit_result_aborted_crossing_inserts_pending_reply_and_stages_both_writes() {
        // D1: a crossing-origin (0x39) DURABLE saga that tombstones ABORTED inserts a PendingAbortReply,
        // stages its PUT, AND stages the UNCONDITIONAL saga-snapshot DELETE — both in the SAME tick's
        // `pending_writes` (co-tick atomicity), keyed on DISTINCT families.
        let mut runtime = SagaRuntimeRes::default();
        let transfer = crossing_xfer();
        inject_saga_id(
            &mut runtime,
            transfer,
            DurabilityClass::Durable,
            aborted_state(),
        );

        commit_result(
            &mut runtime,
            transfer,
            aborted_state(),
            true,
            vec![],
            vec![],
            UniverseTick(0),
        );

        let entry = runtime
            .pending_abort_replies
            .get(&transfer)
            .expect("a pending abort reply was inserted");
        assert_eq!(entry.source, SOURCE);
        assert_eq!(entry.subject, subject());
        assert_eq!(entry.transfer, transfer);
        assert_eq!(
            abort_reply_writes(&runtime, transfer, true),
            1,
            "the AbortReply PUT was staged"
        );
        // The co-tick UNCONDITIONAL saga-snapshot DELETE (a distinct key family) is also staged.
        let saga_delete = StoreKey::Saga(transfer).bytes();
        assert_eq!(
            runtime
                .pending_writes
                .iter()
                .filter(|(k, v)| *k == saga_delete && v.is_none())
                .count(),
            1,
            "the saga-snapshot DELETE rides the same tick"
        );
        assert!(
            !runtime.sagas.contains_key(&transfer),
            "the saga was tombstoned"
        );
    }

    #[test]
    fn commit_result_done_crossing_does_not_insert() {
        // D1: a crossing-origin saga that tombstones DONE (not Aborted) inserts NOTHING — the `aborted`
        // operand is false (the &&-split's second-false case).
        let mut runtime = SagaRuntimeRes::default();
        let transfer = crossing_xfer();
        inject_saga_id(
            &mut runtime,
            transfer,
            DurabilityClass::Durable,
            SagaState::Done {
                new_fence: Fence(2),
            },
        );
        commit_result(
            &mut runtime,
            transfer,
            SagaState::Done {
                new_fence: Fence(2),
            },
            true,
            vec![],
            vec![],
            UniverseTick(0),
        );
        assert!(
            runtime.pending_abort_replies.is_empty(),
            "Done does not owe a reply"
        );
        assert_eq!(abort_reply_writes(&runtime, transfer, true), 0);
    }

    #[test]
    fn commit_result_aborted_rehome_does_not_insert() {
        // D1: a NON-crossing (re-home 0x37) saga that tombstones Aborted inserts NOTHING — the
        // `is_crossing_origin` operand is false (the discriminator-exclusivity / first-false case).
        let mut runtime = SagaRuntimeRes::default();
        let transfer = namespaced_transfer_id(0x37, b"rehome");
        inject_saga_id(
            &mut runtime,
            transfer,
            DurabilityClass::Durable,
            aborted_state(),
        );
        commit_result(
            &mut runtime,
            transfer,
            aborted_state(),
            true,
            vec![],
            vec![],
            UniverseTick(0),
        );
        assert!(
            runtime.pending_abort_replies.is_empty(),
            "a re-home abort owes no crossing reply"
        );
    }

    #[test]
    fn commit_result_aborted_transient_crossing_does_not_insert() {
        // D1: a crossing-namespaced but TRANSIENT saga aborting inserts NOTHING — the `durable` operand is
        // false (the class-gate false arm; a transient carries no per-entity durable latch).
        let mut runtime = SagaRuntimeRes::default();
        let transfer = crossing_xfer();
        inject_saga_id(
            &mut runtime,
            transfer,
            DurabilityClass::Transient,
            aborted_state(),
        );
        commit_result(
            &mut runtime,
            transfer,
            aborted_state(),
            true,
            vec![],
            vec![],
            UniverseTick(0),
        );
        assert!(
            runtime.pending_abort_replies.is_empty(),
            "a transient abort owes no durable reply"
        );
    }

    /// Scan the abort-reply pending set + re-emits with a fresh outbox, at `now`, using the runtime's
    /// tuning (redrive default = 8). Returns the emitted `CrossingAborted` flows to `source`.
    /// Run the abort-reply pass with the subject's directory head owned by `owner` (`None` = no record →
    /// the subject "left the source", which the head-check reaps). 3f-D: the reap keys on ownership, so the
    /// re-emit tests seed `Some(SOURCE)` (still owned → re-emit) and the reap tests seed `None`/`Some(other)`.
    fn scan_abort(
        runtime: &mut SagaRuntimeRes,
        now: UniverseTick,
        owner: Option<NodeId>,
    ) -> Vec<InterShardFlow> {
        let mut dir = DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        });
        if let Some(node) = owner {
            dir.grant(subject(), AuthorityRef::Shard(node), Fence(1), now);
        }
        let mut outbox = OutboundBox::default();
        scan_deadlines(runtime, &mut dir, &mut outbox, EpochId(1), now);
        flows_to_node(&outbox, SOURCE)
    }

    /// Seed one live pending abort reply (no live saga — the saga already tombstoned; this is the
    /// standalone obligation) with `last_abort_reply_emit` armed to 0 (never emitted).
    fn seed_pending_reply(runtime: &mut SagaRuntimeRes, transfer: TransferId) {
        runtime.pending_abort_replies.insert(
            transfer,
            PendingAbortReply {
                transfer,
                source: SOURCE,
                subject: subject(),
            },
        );
    }

    #[test]
    fn scan_deadlines_reemits_a_pending_abort_reply_once_per_cadence() {
        // D2: a live-source pending reply re-emits exactly ONE CrossingAborted to the source per
        // redrive-cadence window (8), and the entry stays (awaiting the ack).
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive = 8
        let transfer = crossing_xfer();
        seed_pending_reply(&mut runtime, transfer);

        // First scan at now=8: last_emit=0, 8-0 >= 8 → emit once (source still owns the subject).
        let emitted = scan_abort(&mut runtime, UniverseTick(8), Some(SOURCE));
        assert_eq!(
            emitted,
            vec![InterShardFlow::CrossingAborted(
                vd_wire::intershard::CrossingAborted {
                    subject: subject(),
                    transfer,
                }
            )],
            "exactly one CrossingAborted re-emitted to the source"
        );
        assert!(
            runtime.pending_abort_replies.contains_key(&transfer),
            "the entry persists until acked"
        );
        assert_eq!(
            runtime.last_abort_reply_emit,
            UniverseTick(8),
            "the cadence gate advanced"
        );
    }

    #[test]
    fn scan_deadlines_throttles_the_abort_reemit_within_the_cadence_window() {
        // D2: within the same redrive window the re-emit is THROTTLED (the not-yet-due arm) — no second
        // egress against an alive-but-ack-stalled source.
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive = 8
        let transfer = crossing_xfer();
        seed_pending_reply(&mut runtime, transfer);
        runtime.last_abort_reply_emit = UniverseTick(6); // last emitted at tick 6

        // now=10: 10-6 = 4 < 8 → NOT due, no emit (source still owns → throttled, not reaped).
        let emitted = scan_abort(&mut runtime, UniverseTick(10), Some(SOURCE));
        assert!(
            emitted.is_empty(),
            "within the window: no re-emit (throttled)"
        );
        assert_eq!(
            runtime.last_abort_reply_emit,
            UniverseTick(6),
            "the gate did not advance"
        );
    }

    #[test]
    fn scan_deadlines_reaps_a_pending_reply_whose_subject_left_the_source() {
        // D2: when the subject NO LONGER belongs to the source (head absent — D-37 re-homed it, or it
        // transferred away), the abort is MOOT → REAPED: the entry is removed, a DELETE staged, ZERO
        // CrossingAborted emitted (the reap arm). `None` owner = no directory record for the subject.
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        let transfer = crossing_xfer();
        seed_pending_reply(&mut runtime, transfer);

        let emitted = scan_abort(&mut runtime, UniverseTick(8), None);
        assert!(
            emitted.is_empty(),
            "a subject that left the source draws no re-emit"
        );
        assert!(
            !runtime.pending_abort_replies.contains_key(&transfer),
            "the moot reply was reaped"
        );
        assert_eq!(
            abort_reply_writes(&runtime, transfer, false),
            1,
            "the reap staged a persist-DELETE"
        );
    }

    #[test]
    fn scan_deadlines_reemits_while_owned_then_reaps_when_the_subject_leaves() {
        // D2: two cadence-spaced scans re-emit twice while the source still owns the subject; then the
        // subject leaves the source (re-homed away → head no longer resolves to SOURCE) and the third scan
        // reaps. Proves the re-emit obligation persists on a LIVE-or-recovering owner and only drains once
        // the entity has genuinely moved (the adversary-review-HIGH fix: not the `is_confirmed_dead` pulse).
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive = 8
        let transfer = crossing_xfer();
        seed_pending_reply(&mut runtime, transfer);

        assert_eq!(
            scan_abort(&mut runtime, UniverseTick(8), Some(SOURCE)).len(),
            1,
            "first re-emit"
        );
        assert_eq!(
            scan_abort(&mut runtime, UniverseTick(16), Some(SOURCE)).len(),
            1,
            "second re-emit"
        );
        // The subject re-homed away from SOURCE (owned by DEST now) → the abort is moot → reaped.
        assert!(
            scan_abort(&mut runtime, UniverseTick(24), Some(DEST)).is_empty(),
            "no re-emit once the subject left the source"
        );
        assert!(
            !runtime.pending_abort_replies.contains_key(&transfer),
            "the third scan reaped the moot reply"
        );
    }

    #[test]
    fn crossing_aborted_ack_drops_the_pending_entry() {
        // D3: the source's `CrossingAbortedAck` drops the pending entry — driven through the full
        // `drive_sagas` inbound path (the new ack arm). Grant the subject's head to SOURCE first, so the
        // ownership-reap does NOT preempt the ack (it would otherwise drain the moot entry before the ack
        // arrives). Then seed the entry, have SOURCE send the ack, and settle.
        let mut rig = Rig::new();
        let transfer = crossing_xfer();
        rig.grant_key(subject(), AuthorityRef::Shard(SOURCE), Fence(1)); // source still owns → no reap
        {
            let mut runtime = rig.orch.world_mut().resource_mut::<SagaRuntimeRes>();
            runtime.pending_abort_replies.insert(
                transfer,
                PendingAbortReply {
                    transfer,
                    source: SOURCE,
                    subject: subject(),
                },
            );
        }
        let ack = |rig: &mut Rig| {
            rig.source
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::CrossingAbortedAck(
                            vd_wire::intershard::CrossingAborted {
                                subject: subject(),
                                transfer,
                            },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("ack sent");
            rig.settle();
        };
        ack(&mut rig); // present → `is_some()`-true arm: drops + stages the DELETE
        assert!(
            rig.orch
                .world_mut()
                .resource::<SagaRuntimeRes>()
                .pending_abort_replies
                .is_empty(),
            "the ack dropped the pending entry"
        );
        ack(&mut rig); // redelivery → `is_some()`-false arm: idempotent no-op (no panic, no double-DELETE)
        assert!(
            rig.orch
                .world_mut()
                .resource::<SagaRuntimeRes>()
                .pending_abort_replies
                .is_empty(),
            "a redelivered ack stays a no-op",
        );
    }

    #[test]
    fn crossing_aborted_ack_is_idempotent_on_redelivery() {
        // D3: a REDELIVERED ack (whose entry is already gone) is a `is_some()`-false no-op — no panic, no
        // double-DELETE stage. Unit-drive the arm twice directly on a runtime.
        let mut runtime = SagaRuntimeRes::default();
        let transfer = crossing_xfer();
        seed_pending_reply(&mut runtime, transfer);

        // First remove: present.
        assert!(runtime.pending_abort_replies.remove(&transfer).is_some());
        runtime
            .pending_writes
            .push((StoreKey::AbortReply(transfer).bytes(), None));
        // Second (redelivery): absent → the guard's false arm is a no-op.
        assert!(
            runtime.pending_abort_replies.remove(&transfer).is_none(),
            "a redelivered ack finds nothing"
        );
        assert_eq!(
            abort_reply_writes(&runtime, transfer, false),
            1,
            "only ONE DELETE staged (the redelivery did not double-stage)"
        );
    }

    #[test]
    fn rehydrate_restores_a_persisted_abort_reply_and_reemits_on_first_scan() {
        // D0 + crash-leg: an AbortReply record persisted before a kill-9 is restored by rehydrate, and the
        // FIRST post-restart scan re-emits CrossingAborted (a RAM-only map would have stranded the source).
        let transfer = crossing_xfer();
        let mut store = MemStore::new();
        // The Clock record is the genesis-vs-recover discriminator — seed it so rehydrate recovers.
        store.put(
            &StoreKey::Clock.bytes(),
            &encode(&(EpochId(1), UniverseTick(0))),
        );
        // The persisted abort-reply obligation (what commit_result staged pre-crash).
        let reply = PendingAbortReply {
            transfer,
            source: SOURCE,
            subject: subject(),
        };
        store.put(&StoreKey::AbortReply(transfer).bytes(), &encode(&reply));
        // COMMIT the staged writes (the group-commit barrier the orchestrator would have run pre-crash) —
        // `scan` reads only the committed set.
        store.commit();

        let recovered = rehydrate(
            &store,
            1024,
            SagaTuning::default(),
            LivenessTuning::default(),
            DirectoryTuning {
                lease_ttl_ticks: 10_000,
                ..DirectoryTuning::default()
            },
            vd_sim::rlm::RlmTuning::default(),
        )
        .expect("the store is non-empty (a Clock record present) → recover");
        let mut runtime = recovered.runtime;
        assert_eq!(
            runtime.pending_abort_replies.get(&transfer),
            Some(&reply),
            "rehydrate restored the persisted abort reply"
        );
        // The first post-restart scan (last_abort_reply_emit == 0) re-emits immediately (source still owns).
        let emitted = scan_abort(&mut runtime, UniverseTick(8), Some(SOURCE));
        assert_eq!(
            emitted,
            vec![InterShardFlow::CrossingAborted(
                vd_wire::intershard::CrossingAborted {
                    subject: subject(),
                    transfer,
                }
            )],
            "the restored obligation re-emits on the first post-restart scan (crash-leg proof)"
        );
    }
}
