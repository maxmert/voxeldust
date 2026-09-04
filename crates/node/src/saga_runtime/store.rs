//! THE DURABLE STORE: the key families, the snapshots, and what a kill-9 restart recovers.
//!
//! Owns: the 1-byte-tagged key families the orchestrator writes under, the serializable snapshot of
//! each live saga, go-token and directory row, and the rehydrate that reconstructs them all on a
//! restart. Every snapshot is SELF-DESCRIBING, so a recovered record needs no side table to be
//! understood.
//!
//! Does NOT own: when anything is written. The group-commit barrier decides that, and it is one
//! barrier per tick — this module only says what the bytes are.

use super::{LiveSaga, PendingAbortReply, SagaRuntimeRes};
use crate::universe_clock::{CeilingClock, ClockAction};
use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use vd_core::pose::StampedPose;
use vd_core::{BatchId, EpochId, Fence, NodeId, TransferId, UniverseTick};
use vd_sim::directory::{DirectoryCore, DirectoryTuning};
use vd_sim::io::{Bytes, Store};
use vd_sim::saga::{LivenessTuning, SagaCtx, SagaState, SagaTuning};
use vd_wire::seams::directory::{DirectoryKey, OwnerRecord};

/// The typed key families in the orchestrator's durable [`Store`] (D-6 saga WAL). Each key is a 1-byte
/// family TAG + the postcard-encoded id, so `scan(&[TAG])` recovers exactly one family on rehydrate. The
/// VALUE at each key is SELF-DESCRIBING (carries its own id/record), so rehydrate never parses a key back.
/// The `Directory` family is a DISTINCT prefix (independently splittable into its own io-prod file — the
/// D-32 partitioning seam — without a cross-file atomic transaction).
#[derive(Clone, Copy, Debug)]
pub(crate) enum StoreKey {
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
    pub(crate) const DIRECTORY: u8 = 3;
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
    pub(crate) fn bytes(self) -> Vec<u8> {
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
pub(crate) struct SagaSnapshot {
    pub(crate) ctx: SagaCtx,
    pub(crate) state: SagaState,
    pub(crate) gateway: NodeId,
    pub(crate) since: UniverseTick,
    pub(crate) flushed_pose: Option<StampedPose>,
    /// The ruler switch, slice 2 — the exterior blob beside the pose (empty for an occupant), so a
    /// restart between the flush and the envelope re-emits the whole exterior, never a poseless one.
    pub(crate) flushed_state: Vec<u8>,
}

/// The persisted go-token (D-6) — self-describing (carries its own `BatchId`) so rehydrate restores
/// `batch_goes` without parsing keys; one record per batch (the G-TIER `batch_go_writes` decoupling is
/// preserved on restart: rehydrate sets the counter to the map len, it does NOT re-drive the `+= 1`).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct GoTokenSnapshot {
    pub(crate) batch: BatchId,
    pub(crate) fence: Fence,
    pub(crate) source: NodeId,
    pub(crate) dest: NodeId,
}

/// The persisted directory record (D-6) — self-describing `(key, record)` so rehydrate calls
/// `DirectoryCore::restore` without parsing keys.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct DirSnapshot {
    pub(crate) key: DirectoryKey,
    pub(crate) record: OwnerRecord,
}

/// The orchestrator's durable store handle, resource-wrapped (D-6). Holds a `dyn Store` so the backend is
/// swappable behind the seam (MemStore now; redb in io-prod). Present on EVERY orchestrator (genesis uses
/// a fresh `MemStore`); the harness RETAINS the concrete handle to re-attach it to a rebuilt orchestrator
/// (the kill-9 analog — the World's RAM dies, the store's committed log survives).
#[derive(Resource)]
pub struct StoreRes(pub Box<dyn Store + Send + Sync>);

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
                flushed_state: snapshot.flushed_state.clone(),
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
