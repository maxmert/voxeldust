//! Production redb-backed [`Store`] — D-6 Slice P3-PERSIST-1 (Store A: the orchestrator's durable
//! directory + saga WAL + clock ceiling).
//!
//! This is the GENERIC redb Store: variance is DATA (the file path + the flat, prefix-tagged keyspace),
//! never CODE — there is NO match on shard kind (HR3). The orchestrator (Store A) is its first consumer;
//! a shard's per-RealmId Store B (block_wal / chunk_snapshot / player_ckpt, P6/P7) reuses this SAME type
//! verbatim (path = the RealmId), so D-6 builds the machinery the realm-persistence layer reuses — never a
//! throwaway. HR1: a shard's redb is PRIVATE + keyed by its RealmId on a persistent volume — never
//! `/tmp/{shard_id}` (the old data-loss bug); single-writer is enforced upstream by the directory owner.
//!
//! C2 (this slice): the fsync is OFF-TICK so the orchestrator's synchronous `step_tick`/`drive_sagas`
//! NEVER blocks on disk. A single named `vd-store-writer` thread owns all writes; `commit()` hands the
//! staged batch over a bounded crossbeam channel and returns. The pipeline depth is ONE by construction
//! (COMMIT-BLOCKS-ON-PRIOR, design `wf_caad488e`, user-decided approach A): `commit()` waits until the
//! PREVIOUS batch is durable before submitting the next — at ~50Hz this is ~always already true (the
//! writer had ~20ms for a tiny batch); under a disk stall it stalls the tick (COUNTED, durability over
//! liveness — the correct back-pressure, and for PvP the right trade: never ship an authority handoff that
//! a crash then loses). Depth-1 bounds crash-loss to ≤1 batch AND makes the per-tick reconcile scan read
//! exactly T-1 — so `scan` stays a plain redb-range (NO in-RAM mirror). The durability watermark reaches
//! the bin via a SIDECAR [`DurabilityHandle`] returned alongside the `Store` from [`RedbStore::open`], so
//! the FROZEN infallible `sim::io::Store` seam + `commit()`'s `()` signature are unchanged.
//!
//! FAIL-LOUD ADAPTER CONTRACT (audit `wf_66cb8f06`): the infallible seam must never degrade a fallible
//! redb fault into a silent wrong answer. A READ fault (`scan`/`is_empty`) PANICS — a silent empty would
//! let `rehydrate` misread a faulting non-empty store as genesis (clock reset + data loss). A WRITE fault
//! is ALL-OR-NOTHING; the writer logs LOUD + does NOT bump the watermark on a fsync error, so the
//! persist-before-effect gate STALLS LOUD rather than ship an unjustified effect (a bounded fsync-retry /
//! durable outbox is the owed refinement). A refusal is never a loss.
//!
//! KNOWN LIMIT — this store is CRASH-durable, not HOST-LOSS-durable (D-47), and its records carry NO format
//! version (D-48). The fsync path above bounds a `kill -9` to ≤1 lost batch, but there is exactly ONE local
//! file and no replication or backup anywhere: lose the disk and the in-flight saga WAL, the durable directory
//! head (the commit point) and the clock ceiling go with it. Worse, `open` cannot distinguish "volume not
//! attached" from a legitimate first boot — an ABSENT store IS genesis by construction — so a rescheduled pod
//! without its volume boots clean and silently recovers nothing. The deploy preconditions (pinned volume per
//! orchestrator identity + a store-identity stamp + an explicit allow-genesis opt-in) and the store format
//! version are HARD requirements before any multi-node deploy; see DEFERRED.md D-47 / D-48.
//!
//! Tier-B (HR5): io-prod is excluded from the 100% region+branch gate; the redb adapter is process-tier
//! covered at a ratcheted floor. The in-process durability tests here give high happy-path coverage; the
//! real-process SIGKILL-mid-fsync crash proof lands with the orchestrator wiring (Slice D).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Receiver, SendError, Sender, TrySendError, bounded};
use redb::{Database, ReadableTableMetadata, TableDefinition};
use tokio::sync::Notify;
use vd_core::store_stamp::StoreStamp;
use vd_sim::io::{Bytes, Store, bytes};

/// The writer retries a failed fsync this many times (a TRANSIENT disk blip recovers) before declaring the
/// durable store unwritable + EXITING (a PERMANENT fault → the persist-before-effect gate fails LOUD via the
/// liveness escape, never a silent hang). The writer still HOLDS the merged batch across retries, so a
/// transient retry loses nothing; only a permanent give-up drops it (loud, and recovery re-drives — the
/// owed durable-outbox refinement would avoid even that re-drive).
const WRITER_FSYNC_MAX_RETRIES: u32 = 3;
/// Backoff between fsync retries (a brief pause so a momentary disk hiccup clears).
const WRITER_FSYNC_RETRY_BACKOFF: Duration = Duration::from_millis(10);
/// The block-on-prior / `flush_blocking` wait re-checks writer LIVENESS at this cadence (the writer's
/// notify handles the prompt happy-path wake; this is only the backstop that detects a DEAD writer so the
/// synchronous orchestrator thread fails loud instead of hanging forever).
const WRITER_WAIT_POLL: Duration = Duration::from_millis(100);

/// The ONE key/value table. The keyspace is the EXACT flat, prefix-tagged `StoreKey` bytes the
/// orchestrator already writes through the [`Store`] seam (one prefix byte per family), so `scan(prefix)`
/// is a redb range over that prefix — byte-for-byte compatible with the `MemStore` tier. The split-ready
/// table-set seam (user decision: single file now, the D-32 Directory-file split later): a future split
/// adds a second `TableDefinition` / `Database` here behind a table-set descriptor, with no change to the
/// keyspace or the consumers, landing its required head-re-read WITH the split (never before).
const KV: TableDefinition<&[u8], &[u8]> = TableDefinition::new("kv");

/// THE FILE'S OWN LABEL, at a key reserved for it (owner ruling 2026-08-24; D-48).
///
/// Tag `0`, chosen deliberately: the typed family tags a consumer allocates start at 1 and are
/// append-only, so 0 can never be reached by that list. This key belongs to the FILE, not to any family
/// — it is the one record written and read by the layer that owns the bytes rather than by the layer
/// that owns their meaning, which is exactly why it can be verified BEFORE any family is scanned.
///
/// It must be checked first because our encoding is positional and not self-describing: two different
/// record shapes decode from the same bytes with no error at all, so there is no "read it and see"
/// available here.
const STAMP_KEY: &[u8] = &[0u8];

/// A staged fsync window: `Some` = put, `None` = delete (last-write-wins per key).
type Batch = BTreeMap<Vec<u8>, Option<Bytes>>;

/// Operational tuning for the redb writer (ONE config struct — no inline magic numbers).
///
/// ★ CLONE ONLY, IN EVERY BUILD (2026-08-28). This used to derive `Copy` when `store-test-hooks` was
/// off and not when it was on, because the feature adds owned fields. A trait that appears and vanishes
/// with a feature makes one build compile and another not, and the one that broke was the one no gate
/// linted: the orchestrator binary passed this by value into an `Fn` closure, which needs `Copy`. That
/// stopped compiling at commit 7a0506e and stayed broken for four commits, which took the SIGKILL
/// durability gate (`just orch-crash`) dark with it — the gate could not build its own binary.
///
/// One trait surface in every build is what stops that recurring. The type is small and is cloned once
/// at boot, so nothing pays for it.
#[derive(Clone, Debug)]
pub struct StoreTuning {
    /// Bounded depth of the staged-batch hand-off channel. The EFFECTIVE in-flight depth is ONE
    /// (commit-blocks-on-prior); this is a small slack so a `send` never blocks the sim thread even
    /// momentarily mid-fsync, and the seam for a future depth>1 mode. NEVER set 0 (a rendezvous channel
    /// would couple the sim thread to the fsync).
    pub writer_channel_depth: usize,
    /// TEST-ONLY (D-6 D-delta; feature `store-test-hooks`, ABSENT from release). When `Some(prefix)`, the
    /// off-tick writer BLOCKS FOREVER before fsyncing the FIRST batch whose key-set contains a key starting
    /// with `prefix` (the planted sentinel) — so the process can be SIGKILLed in a deterministic
    /// submitted-but-pre-fsync window. Content-keyed, NOT seq-keyed: the per-tick Clock-key churn cannot
    /// trip it early; it fires on exactly the batch carrying the sentinel.
    #[cfg(feature = "store-test-hooks")]
    pub pause_on_key_prefix: Option<Vec<u8>>,
    /// TEST-ONLY: the marker file the writer creates the instant it parks — the crash test's HONEST,
    /// decoupled "window is open" signal, written by the WRITER thread itself (independent of the bin loop,
    /// which BLOCKS on the persist-before-effect gate the instant the sentinel batch fails to become durable,
    /// so it can never publish the signal). Existence ⟺ sentinel batch submitted but not yet fsynced.
    #[cfg(feature = "store-test-hooks")]
    pub pause_marker_path: Option<PathBuf>,
    /// TEST-ONLY (R-6d4-B4; feature `store-test-hooks`, ABSENT from release). When `Some(prefix)`, the writer
    /// treats the fsync of the FIRST batch whose key-set contains a key starting with `prefix` as a PERMANENT
    /// fault (`apply_batch` forced to fail) — so after `WRITER_FSYNC_MAX_RETRIES` it `break 'drain`s and DIES,
    /// firing `WriterExitSignal`. Distinct from `pause_on_key_prefix` (which parks forever, hanging Drop's
    /// join): this makes the writer EXIT deterministically, so a durability waiter enrolled on that seq takes
    /// the R-6d4-M death short-circuit (proving the prompt fail-loud, not the timeout backstop).
    #[cfg(feature = "store-test-hooks")]
    pub fail_fsync_on_key_prefix: Option<Vec<u8>>,
    /// TEST-ONLY (R-6d4-B4; feature `store-test-hooks`). Overrides [`WRITER_WAIT_POLL`] for the ASYNC
    /// durability wait ([`DurabilityHandle::wait_durable_through_async`]) — set LARGE (e.g. 10s) so a
    /// mutation that neuters the death `notify_waiters()` visibly HANGS to this timeout, while the live
    /// death-wake short-circuit panics in ~ms: the gap is what makes the prompt-wake assertion mutation-
    /// sensitive (the exact `wf_75a225d0` gap — a 100ms release poll is too tight to distinguish). `None` ⇒
    /// the release const (prod is byte-identical).
    #[cfg(feature = "store-test-hooks")]
    pub wait_poll_override: Option<Duration>,
}

impl Default for StoreTuning {
    fn default() -> Self {
        Self {
            writer_channel_depth: 2,
            #[cfg(feature = "store-test-hooks")]
            pause_on_key_prefix: None,
            #[cfg(feature = "store-test-hooks")]
            pause_marker_path: None,
            #[cfg(feature = "store-test-hooks")]
            fail_fsync_on_key_prefix: None,
            #[cfg(feature = "store-test-hooks")]
            wait_poll_override: None,
        }
    }
}

/// Errors opening the durable store — a typed door, never a silent data path.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// Opening / creating the redb database failed.
    #[error("redb open/create: {0}")]
    Open(String),
    /// A transaction (the genesis table-ensure) failed during open.
    #[error("redb txn: {0}")]
    Txn(String),
    /// Spawning the off-tick writer thread failed.
    #[error("spawn writer: {0}")]
    Spawn(String),
    /// THE SAVED-DATA LABEL REFUSED THIS FILE (owner ruling 2026-08-24, Q1 condition 1; D-48).
    ///
    /// The refusal carries the field that disagreed and BOTH values — a refusal an operator cannot act
    /// on becomes an unofficial delete, which is the loss the refusal exists to prevent.
    #[error("refusing to open {path}: {source}")]
    Stamp {
        /// The file that was refused, so the operator knows which one to act on.
        path: String,
        /// Which field disagreed, and with what.
        source: vd_core::store_stamp::StampRefusal,
    },
}

/// The bin-facing durability watermark (the persist-before-effect gate's read side). Cloneable + `Send`;
/// the orchestrator bin (Slice D) holds it alongside the `Box<dyn Store>` and gates a tick's outbox flush
/// on [`DurabilityHandle::is_durable_through`] so no effect leaves before its authorizing state is durable.
#[derive(Clone)]
pub struct DurabilityHandle {
    last_durable: Arc<AtomicU64>,
    /// The seq of the last batch `commit()` submitted (shared with `RedbStore`). The bin records this after
    /// each `run_schedule` and waits on it before flushing that tick's outbox (persist-before-effect).
    last_submitted: Arc<AtomicU64>,
    fsync_backpressure: Arc<AtomicU64>,
    /// The writer's durability notify — so the bin's [`wait_durable_through`](Self::wait_durable_through)
    /// PARKS (not hot-spins) under a disk stall, sharing the writer's wake.
    durable_cv: Arc<(Mutex<()>, Condvar)>,
    /// R-6d3b F2: the ASYNC twin of `durable_cv` — a `tokio::sync::Notify` the writer `notify_waiters()` on
    /// each `last_durable` bump (AND on writer death), so [`wait_durable_through_async`] yields the tokio
    /// worker instead of parking it. Distinct from `durable_cv` (the sim/main-thread sync park) so the two
    /// wait styles never interfere; both are woken by the same writer, at the same site.
    ///
    /// [`wait_durable_through_async`]: Self::wait_durable_through_async
    durable_notify: Arc<Notify>,
    /// `true` while the off-tick writer lives; `false` once it has EXITED. The handle has no `JoinHandle`,
    /// so this flag is its liveness signal for the fail-loud escape.
    writer_alive: Arc<AtomicBool>,
    /// The ASYNC-wait poll backstop (`WRITER_WAIT_POLL` in release; a test may override it via
    /// `StoreTuning::wait_poll_override` — R-6d4-B4's mutation-sensitivity seam). Read every
    /// `wait_durable_through_async` iteration; keeps that method free of a `#[cfg]` on the const.
    wait_poll: Duration,
}

impl DurabilityHandle {
    /// Whether every batch up to and including `seq` has been fsynced.
    #[must_use]
    pub fn is_durable_through(&self, seq: u64) -> bool {
        self.last_durable.load(Ordering::Acquire) >= seq
    }
    /// The highest fsynced batch sequence.
    #[must_use]
    pub fn durable_through(&self) -> u64 {
        self.last_durable.load(Ordering::Acquire)
    }
    /// The seq of the last batch `commit()` submitted — the bin records it after each tick's `run_schedule`
    /// and gates that tick's outbox flush on it (Slice D parked-flush). `0` before the first commit.
    #[must_use]
    pub fn last_submitted(&self) -> u64 {
        self.last_submitted.load(Ordering::Acquire)
    }
    /// Whether the off-tick writer thread is still alive. `false` once it has EXITED (graceful shutdown or a
    /// permanent fsync fault) — the bin's [`wait_durable_through`](Self::wait_durable_through) uses this to
    /// fail LOUD rather than hang on a dead writer.
    #[must_use]
    pub fn writer_alive(&self) -> bool {
        self.writer_alive.load(Ordering::Acquire)
    }
    /// Block until every batch through `seq` is durable — THE persist-before-effect gate the bin calls
    /// before flushing a tick's outbox. Parks on the writer's notify (no hot-spin); fails LOUD if the writer
    /// died before reaching `seq` (a refusal is never a loss). At ~50Hz with the one-tick defer this is
    /// ~always already durable (returns immediately); a real wait IS the disk-stall back-pressure.
    pub fn wait_durable_through(&self, seq: u64) {
        if self.last_durable.load(Ordering::Acquire) < seq {
            // A bin-side disk stall (the parked-flush gate had to block) — count it so the SAME disk-stall
            // back-pressure is observable via `backpressure_stalls()` whether it lands on commit's
            // block-on-prior or here on the flush gate (review wf_D-gamma LOW).
            self.fsync_backpressure.fetch_add(1, Ordering::Relaxed);
        }
        park_until_durable(seq, &self.last_durable, &self.durable_cv, || {
            !self.writer_alive.load(Ordering::Acquire)
        });
    }

    /// R-6d3b F2: the ASYNC durable-before-send gate — awaits until every batch through `seq` is durable
    /// WITHOUT parking a thread. The mesh `write_batch` phase 3 `.await`s this ONCE PER BATCH on a tokio worker, which is
    /// RELEASED at each `.await` — so N concurrent producer-less durable sends yield N workers rather than
    /// parking N (the `worker_threads(2)` starvation the sync [`wait_durable_through`] would cause on the mesh
    /// I/O pool). ADDITIVE: the sync variant stays THE persist-before-effect gate for the orchestrator's
    /// MAIN-thread caller. Lost-wakeup-free: the `Notified` future is enrolled BEFORE the `last_durable`
    /// re-read, and the writer `notify_waiters()` AFTER its `last_durable.store` (Release) — so a bump racing
    /// between our enroll and await still wakes us. Fails LOUD (panic) if the writer dies before `seq`,
    /// mirroring the sync park's liveness escape — a refusal is never a loss (recovery re-drives).
    pub async fn wait_durable_through_async(&self, seq: u64) {
        if self.last_durable.load(Ordering::Acquire) >= seq {
            return; // fast path: already durable ⇒ no enroll, no await (the ~always case at 50Hz)
        }
        self.fsync_backpressure.fetch_add(1, Ordering::Relaxed); // same disk-stall back-pressure counter
        loop {
            // ENROLL before the re-read (lost-wakeup-free): a `notify_waiters()` after this point but before
            // the `.await` still completes the captured `Notified`.
            let notified = self.durable_notify.notified();
            if self.last_durable.load(Ordering::Acquire) >= seq {
                return;
            }
            // Await either a writer wake (a durable bump OR its death `notify_waiters`) or the poll
            // backstop; on EITHER outcome fall through to the SHARED death-check below. R-6d4-M: a
            // death-wake fails loud IMMEDIATELY here (a durable bump returns at the loop top) — the old
            // `Ok(()) => {}` re-enrolled a FRESH `notified` that the already-fired death-notify would never
            // wake, so death cost a full `WRITER_WAIT_POLL`; the collapsed check makes it prompt.
            let _ = tokio::time::timeout(self.wait_poll, notified).await;
            // A writer that DIED can never bump `last_durable`; fail LOUD rather than await forever (the
            // timeout is the guarantee even if the death-wake were lost; the death-wake makes it prompt).
            // A live-but-slow writer (disk stall, still `writer_alive`) simply re-loops under back-pressure.
            if self.last_durable.load(Ordering::Acquire) < seq
                && !self.writer_alive.load(Ordering::Acquire)
            {
                writer_died_panic(seq);
            }
        }
    }

    /// How many `commit()`s had to STALL on the previous fsync (the back-pressure counter — 0 on a healthy
    /// run; `> 0` means the disk could not keep up with the tick rate, the correct durability-over-liveness
    /// signal, mirroring the `TickPacer` overrun discipline).
    #[must_use]
    pub fn backpressure_stalls(&self) -> u64 {
        self.fsync_backpressure.load(Ordering::Acquire)
    }

    /// TEST-ONLY: a handle already durable through `u64::MAX` with the writer reading as alive, so
    /// `is_durable_through`/`wait_durable_through` fast-return for ANY seq. Lets a `MockOutboxSink` drive the
    /// R-6d3a durable-before-send gate (its `submit_barrier` returns this handle) without a real store/writer.
    #[cfg(any(test, feature = "store-test-hooks"))]
    #[must_use]
    pub fn already_durable() -> DurabilityHandle {
        DurabilityHandle {
            last_durable: Arc::new(AtomicU64::new(u64::MAX)),
            last_submitted: Arc::new(AtomicU64::new(u64::MAX)),
            fsync_backpressure: Arc::new(AtomicU64::new(0)),
            durable_cv: Arc::new((Mutex::new(()), Condvar::new())),
            durable_notify: Arc::new(Notify::new()),
            writer_alive: Arc::new(AtomicBool::new(true)),
            wait_poll: WRITER_WAIT_POLL,
        }
    }

    /// TEST-ONLY (R-6d4-B2 / R-6d4-A): a handle whose `last_submitted` / `last_durable` are DRIVEN BY THE
    /// TEST via the returned atomics — modeling the real peer-writer re-mirror (a `send_durable` bumps
    /// `submitted`; the fsync later bumps `durable`) WITHOUT a real store/writer/fsync. `writer_alive` reads
    /// true, so `wait_durable_through(target)` genuinely PARKS while `durable < target` (the un-durable
    /// window B2's no-premature-gc pin asserts across); the test bumps `durable` to release it for cleanup.
    #[cfg(any(test, feature = "store-test-hooks"))]
    #[must_use]
    pub fn controllable() -> (DurabilityHandle, Arc<AtomicU64>, Arc<AtomicU64>) {
        let last_submitted = Arc::new(AtomicU64::new(0));
        let last_durable = Arc::new(AtomicU64::new(0));
        let handle = DurabilityHandle {
            last_durable: Arc::clone(&last_durable),
            last_submitted: Arc::clone(&last_submitted),
            fsync_backpressure: Arc::new(AtomicU64::new(0)),
            durable_cv: Arc::new((Mutex::new(()), Condvar::new())),
            durable_notify: Arc::new(Notify::new()),
            writer_alive: Arc::new(AtomicBool::new(true)),
            wait_poll: WRITER_WAIT_POLL,
        };
        (handle, last_submitted, last_durable)
    }
}

/// The generic redb-backed [`Store`]. Staged mutations live in RAM; `commit()` hands them to the off-tick
/// writer thread (which owns the redb writes); `scan()` reads the committed redb directly (MVCC concurrent
/// with the writer). Last-write-wins within a window is free — the staged map keys on the byte key.
pub struct RedbStore {
    /// Shared with the writer thread (redb is 1-writer/N-reader MVCC): `scan`/`is_empty` `begin_read` here
    /// while the writer `begin_write`s the SAME file.
    db: Arc<Database>,
    path: PathBuf,
    /// Staged this fsync window. Drained by `commit` into a channel batch; RETAINED (returned to here) if
    /// the channel is disconnected (writer gone) — fail-safe, a refusal is never a loss.
    staged: Batch,
    /// Hand-off to the writer. `Option` so [`Drop`] can drop the ONLY sender first (disconnecting the
    /// channel so the writer drains-then-exits) before joining.
    submit_tx: Option<Sender<(u64, Batch)>>,
    /// Monotone batch sequence (the last value ASSIGNED to a submitted batch).
    next_seq: u64,
    /// The seq of the last SUCCESSFULLY-submitted batch (commit-blocks-on-prior waits for this to be
    /// durable). SHARED with the [`DurabilityHandle`] (`Arc<AtomicU64>`) so the bin reads it without the
    /// store — the sidecar seam that keeps `Store` frozen.
    last_submitted: Arc<AtomicU64>,
    /// The highest fsynced seq (bumped by the writer, Release). The lock-free source of truth.
    last_durable: Arc<AtomicU64>,
    /// Count of commits that stalled on the previous fsync (back-pressure observable).
    fsync_backpressure: Arc<AtomicU64>,
    /// The writer notifies on each `last_durable` bump so block-on-prior / `flush_blocking` PARK (not hot
    /// spin) under a disk stall. The atomic stays the source of truth; the mutex guards only the condvar.
    durable_cv: Arc<(Mutex<()>, Condvar)>,
    /// `true` while the writer lives; set `false` by the writer on its single EXIT. THE liveness signal for
    /// both durability waits (this store's and the handle's, which has no `JoinHandle` to probe). The
    /// `JoinHandle` below stays only for Drop's graceful join.
    writer_alive: Arc<AtomicBool>,
    writer: Option<JoinHandle<()>>,
}

/// Apply a batch in ONE redb WriteTransaction + fsync. ALL-OR-NOTHING: a per-key apply error drops the txn
/// uncommitted (rollback) and returns `false` — never a partial fsync. Returns `true` iff durably committed.
fn apply_batch(db: &Database, batch: &Batch) -> bool {
    let txn = match db.begin_write() {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("vd-store-writer: begin_write failed: {e}");
            return false;
        }
    };
    {
        let mut table = match txn.open_table(KV) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("vd-store-writer: open_table failed: {e}");
                return false; // txn dropped → rollback
            }
        };
        for (key, value) in batch {
            let applied = match value {
                Some(v) => table.insert(key.as_slice(), v.as_ref()).map(|_| ()),
                None => table.remove(key.as_slice()).map(|_| ()),
            };
            if let Err(e) = applied {
                tracing::error!("vd-store-writer: apply failed, aborting batch: {e}");
                return false; // drop table + txn → rollback (all-or-nothing)
            }
        }
    }
    match txn.commit() {
        Ok(()) => true,
        Err(e) => {
            tracing::error!("vd-store-writer: commit fsync failed: {e}");
            false
        }
    }
}

/// R-6d4-M: the SINGLE fail-loud site for a durable-writer death before `seq`, called by BOTH the sync
/// [`park_until_durable`] liveness escape and the async [`RedbStore::wait_durable_through_async`] death
/// short-circuit (one message, `#[cold]`, `-> !`). A refusal is NEVER a loss (recovery rehydrates the last
/// durable state + the producer re-drives idempotently); hanging the send forever is strictly worse.
#[cold]
fn writer_died_panic(seq: u64) -> ! {
    panic!(
        "the durable writer thread died before seq {seq} — refusing to hang the send \
         (a refusal is never a loss; recovery rehydrates + re-drives)"
    );
}

/// Park (not hot-spin) until `last_durable >= target`. The shared core of BOTH durability waits — the
/// `RedbStore`'s own (Drop/tests, liveness via its `JoinHandle`) and the bin-side [`DurabilityHandle`]'s
/// (the persist-before-effect gate, liveness via the shared `writer_alive` flag it can read without the
/// handle). Fast-path the common already-durable case; on a miss, lock + re-check + condvar-wait (the writer
/// notifies under the same lock — no lost wakeup). The `WRITER_WAIT_POLL` timeout is the liveness backstop:
/// on a timeout with no progress AND a dead writer (`is_writer_dead()`), fail LOUD rather than hang the
/// synchronous caller forever — a refusal is never a loss; recovery rehydrates the last durable state + the
/// Slice-2a producer re-drives idempotently.
fn park_until_durable(
    target: u64,
    last_durable: &AtomicU64,
    durable_cv: &(Mutex<()>, Condvar),
    is_writer_dead: impl Fn() -> bool,
) {
    if last_durable.load(Ordering::Acquire) >= target {
        return;
    }
    let (lock, cv) = durable_cv;
    let mut guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
    while last_durable.load(Ordering::Acquire) < target {
        let (g, _res) = cv
            .wait_timeout(guard, WRITER_WAIT_POLL)
            .unwrap_or_else(PoisonError::into_inner);
        guard = g;
        // R-6d4-M: check death on EVERY wake, not only on `res.timed_out()` — the writer's death
        // `notify_all` (WriterExitSignal::drop) wakes us WITHOUT a timeout, so gating the escape on
        // `timed_out()` cost a full extra `WRITER_WAIT_POLL`. A genuine durable bump exits the `while`
        // (no panic); a death-wake with `last_durable < target` fails loud promptly here.
        if last_durable.load(Ordering::Acquire) < target && is_writer_dead() {
            drop(guard);
            writer_died_panic(target);
        }
    }
}

/// The off-tick writer loop: drain a batch (coalescing any already-queued ones — vestigial under depth-1,
/// the seam for a future depth>1 mode), apply+fsync in ONE txn, then bump `last_durable` (Release) +
/// notify. On a fsync error the watermark is NOT bumped → the persist-before-effect gate stalls LOUD,
/// never a silent effect loss (the owed refinement is a bounded retry). Exits when the sender is dropped.
/// Publishes writer DEATH on EVERY exit of `run_writer` — clean return, permanent-fault break, OR a
/// panic-unwind (the latter would otherwise leave `writer_alive == true` forever and hang every waiter past
/// the liveness backstop, since the escape reads the still-`true` flag — review wf_D-gamma LOW). Drop sets
/// the flag false (the handle's only liveness signal) then wakes any parked waiter so the fail-loud escape
/// fires promptly rather than at the next `WRITER_WAIT_POLL`.
struct WriterExitSignal {
    writer_alive: Arc<AtomicBool>,
    durable_cv: Arc<(Mutex<()>, Condvar)>,
    /// R-6d3b F2 (finding E): the async waiters' notify — woken on death too, so an async block-B waiter
    /// fails loud PROMPTLY (not after a full `WRITER_WAIT_POLL`), symmetric with the sync condvar wake.
    durable_notify: Arc<Notify>,
}

impl Drop for WriterExitSignal {
    fn drop(&mut self) {
        self.writer_alive.store(false, Ordering::Release);
        let (lock, cv) = &*self.durable_cv;
        let _g = lock.lock().unwrap_or_else(PoisonError::into_inner);
        cv.notify_all();
        // Wake async waiters on death too (finding E) — `notify_waiters()` needs no runtime handle, so it is
        // sound from this std-thread Drop; the timeout backstop is the guarantee, this just makes it prompt.
        self.durable_notify.notify_waiters();
    }
}

/// TEST-ONLY (feature `store-test-hooks`, ABSENT from release — the D-6 D-delta SIGKILL-mid-fsync proof).
/// If a planted sentinel prefix is configured and the merged batch carries it, announce the open
/// submitted-but-pre-fsync window via the marker file (written by THIS writer thread — the bin loop is
/// already blocked on the persist-before-effect gate and cannot signal), then BLOCK FOREVER. The process
/// is meant to be SIGKILLed here; `park` (looped against spurious wakeups) blocks with no busy-spin.
#[cfg(feature = "store-test-hooks")]
fn maybe_pause_before_fsync(prefix: Option<&[u8]>, marker: Option<&Path>, merged: &Batch) {
    let Some(prefix) = prefix else { return };
    if !merged.keys().any(|k| k.starts_with(prefix)) {
        return;
    }
    if let Some(marker) = marker {
        let _ = std::fs::write(marker, b"paused\n");
    }
    loop {
        std::thread::park();
    }
}

// The writer owns the shared durability atomics/notifies + (under the feature) the three content-keyed test
// hooks (pause / pause-marker / fsync-fault); grouping them into a struct would only move the arg list, so
// the internal spawn fn is allowed its wide signature.
#[allow(clippy::too_many_arguments)]
fn run_writer(
    db: Arc<Database>,
    rx: Receiver<(u64, Batch)>,
    last_durable: Arc<AtomicU64>,
    durable_cv: Arc<(Mutex<()>, Condvar)>,
    durable_notify: Arc<Notify>,
    writer_alive: Arc<AtomicBool>,
    #[cfg(feature = "store-test-hooks")] pause_prefix: Option<Vec<u8>>,
    #[cfg(feature = "store-test-hooks")] pause_marker: Option<PathBuf>,
    #[cfg(feature = "store-test-hooks")] fail_prefix: Option<Vec<u8>>,
) {
    // The guard publishes death on ANY exit below (incl. an unexpected panic-unwind), so the liveness
    // escape can never miss it. Holds its own `durable_cv`/`durable_notify` clones; the loop keeps the
    // originals for the per-batch success notify.
    let _exit = WriterExitSignal {
        writer_alive,
        durable_cv: Arc::clone(&durable_cv),
        durable_notify: Arc::clone(&durable_notify),
    };
    'drain: while let Ok((first_seq, first)) = rx.recv() {
        let mut max_seq = first_seq;
        let mut merged = first;
        while let Ok((seq, batch)) = rx.try_recv() {
            for (k, v) in batch {
                merged.insert(k, v); // last-write-wins across coalesced batches
            }
            max_seq = seq;
        }
        // TEST-ONLY pause BEFORE fsync (content-keyed). Coalescing above is harmless: under depth-1 the
        // sentinel batch is alone, and even a coalesced merge still carries the sentinel prefix ⇒ still pauses.
        #[cfg(feature = "store-test-hooks")]
        maybe_pause_before_fsync(pause_prefix.as_deref(), pause_marker.as_deref(), &merged);
        // Retry the fsync (the writer still HOLDS `merged`, so a TRANSIENT disk blip recovers losing
        // nothing); a PERMANENT fault EXITS the writer so the persist-before-effect gate fails LOUD (the
        // liveness escape in `park_until_durable` panics rather than hang the orchestrator).
        // R-6d4-B4: a content-keyed PERMANENT fsync fault — the merged batch carrying `fail_prefix` is
        // treated as unwritable so the writer exhausts its retries + `break 'drain`s (DIES) deterministically,
        // exercising the death path (vs `pause_prefix`, which parks forever + hangs Drop's join). Computed
        // ONCE per batch outside the retry loop (the fault is permanent for this batch).
        #[cfg(feature = "store-test-hooks")]
        let forced_fault = fail_prefix
            .as_deref()
            .is_some_and(|p| merged.keys().any(|k| k.starts_with(p)));
        #[cfg(not(feature = "store-test-hooks"))]
        let forced_fault = false;
        let mut attempt = 0u32;
        loop {
            if !forced_fault && apply_batch(&db, &merged) {
                last_durable.store(max_seq, Ordering::Release);
                // Wake BOTH wait styles, AFTER the Release store (lost-wakeup-free for both): the sync
                // condvar (orchestrator/main-thread) and the async Notify (mesh block-B, R-6d3b F2).
                let (lock, cv) = &*durable_cv;
                let _g = lock.lock().unwrap_or_else(PoisonError::into_inner);
                cv.notify_all();
                durable_notify.notify_waiters();
                break;
            }
            attempt += 1;
            if attempt >= WRITER_FSYNC_MAX_RETRIES {
                tracing::error!(
                    "vd-store-writer: fsync failed {attempt}x for batch up to seq {max_seq} — the durable \
                     store is unwritable; EXITING so the persist-before-effect gate fails LOUD (panics) \
                     rather than hang. Recovery rehydrates the last durable state + re-drives idempotently."
                );
                break 'drain; // writer dies → the SINGLE exit below marks it dead + wakes waiters
            }
            std::thread::sleep(WRITER_FSYNC_RETRY_BACKOFF);
        }
    }
    // Death (permanent fault OR graceful sender-drop) is published by `_exit`'s Drop here — see
    // `WriterExitSignal`. On a graceful shutdown nobody is waiting; on a permanent fault the gate panics
    // loud (a refusal is never a loss; recovery rehydrates + re-drives).
}

/// Opaque key/value rows read out of a file we refused — see [`scan_refused_for_reaping`]. Named rather
/// than written inline so it is obvious at every use site that these are BYTES, not records: nothing has
/// validated them and nothing may adopt them.
pub type RefusedRows = Vec<(Vec<u8>, Vec<u8>)>;

/// READ A REFUSED FILE'S ROWS ANYWAY — a deliberately narrow door, for ONE purpose.
///
/// # Why this exists, and why it is not a hole
///
/// Refusing a file is right. Refusing the ORCHESTRATOR'S LAUNCH LEDGER and then exiting is not, because
/// children are started into their own process group and are NOT killed when their parent drops them.
/// That ledger is the only record of which ones exist. An orchestrator that refuses it and exits
/// therefore ORPHANS every shard it ever started: they keep running, keep holding their ports, and
/// nothing left alive knows they are there.
///
/// So the refusal stands — nothing is adopted, nothing is written, the process still exits — but the
/// children are reaped on the way out, which needs their pids, which needs this read.
///
/// It is safe because of what it is NOT: read-only, no writer thread, no store handle returned, and it
/// hands back OPAQUE BYTES. A caller that cannot decode them gets nothing, which is the correct answer
/// for a file whose shape really is unknown. It can adopt nothing, because there is nothing to adopt.
///
/// # Errors
/// [`StoreError::Open`] or [`StoreError::Txn`] if the file cannot be read at all. A file with no table
/// yet reads as empty — that is a file with no children in it.
pub fn scan_refused_for_reaping(path: &Path, prefix: &[u8]) -> Result<RefusedRows, StoreError> {
    let db = Database::create(path).map_err(|e| StoreError::Open(e.to_string()))?;
    let txn = db
        .begin_read()
        .map_err(|e| StoreError::Txn(e.to_string()))?;
    let Ok(table) = txn.open_table(KV) else {
        return Ok(Vec::new()); // no table ⇒ no rows ⇒ no children to reap
    };
    let mut out = Vec::new();
    let iter = table
        .range(prefix..)
        .map_err(|e| StoreError::Txn(e.to_string()))?;
    for row in iter {
        let (k, v) = row.map_err(|e| StoreError::Txn(e.to_string()))?;
        if !k.value().starts_with(prefix) {
            break;
        }
        out.push((k.value().to_vec(), v.value().to_vec()));
    }
    Ok(out)
}

/// THE OPERATOR'S EXPLICIT ACT: delete a durable file so the next open starts a new world.
///
/// This is the other half of a refusal. A refusal with no way forward becomes an unofficial `rm -rf`
/// typed under pressure at three in the morning — which deletes more than was meant and leaves no
/// record. So the way forward is named IN the refusal message, it is one file, and it is loud.
///
/// It is deliberately NOT reached by a fallback inside [`RedbStore::open`]. A wipe that could happen
/// automatically would eventually happen automatically, on a file somebody needed.
///
/// # Errors
/// [`StoreError::Open`] if the file exists and cannot be removed. A file that is already absent is not
/// an error — that is the state being asked for.
pub fn wipe_for_genesis(path: &Path) -> Result<(), StoreError> {
    match std::fs::remove_file(path) {
        Ok(()) => {
            tracing::warn!(
                store = %path.display(),
                "DELETED a durable store because genesis was explicitly allowed. Whatever it held is gone."
            );
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(StoreError::Open(format!(
            "deleting {} for genesis: {e}",
            path.display()
        ))),
    }
}

/// Open a durable file, and — only if the caller has explicitly allowed a fresh start — delete and
/// retry when the label refuses it.
///
/// Written as a wrapper around whatever opener the caller passes, so the two different store types
/// share exactly one copy of this decision, and so the decision is visible rather than buried inside
/// either of them.
///
/// The retry happens ONCE and only for a LABEL refusal. A file that cannot be opened for any other
/// reason is not a file to delete.
///
/// # Errors
/// The opener's own error, or the refusal itself when genesis was not allowed.
pub fn open_allowing_genesis<T>(
    path: &Path,
    allow_genesis: bool,
    open: impl Fn(&Path) -> Result<T, StoreError>,
) -> Result<T, StoreError> {
    match open(path) {
        Ok(v) => Ok(v),
        Err(StoreError::Stamp { path: p, source }) => {
            if !allow_genesis {
                return Err(StoreError::Stamp { path: p, source });
            }
            tracing::warn!(
                store = %path.display(),
                refusal = %source,
                "the durable store was refused AND genesis is explicitly allowed — deleting it and \
                 starting a new world here"
            );
            wipe_for_genesis(path)?;
            open(path)
        }
        Err(other) => Err(other),
    }
}

/// Read the file's label, compare it against what this build expects, and write it if the file is new.
///
/// Split out of [`RedbStore::open`] so the decision it delegates to is reachable on its own and so the
/// three outcomes are each named here rather than inlined into a long constructor. The DECISION itself
/// is not here — it is [`vd_core::store_stamp::verify`], in the pure crate, where the coverage gate
/// holds every branch of it.
fn verify_stamp(db: &Database, path: &Path, expected: StoreStamp) -> Result<(), StoreError> {
    let txn = db
        .begin_read()
        .map_err(|e| StoreError::Txn(e.to_string()))?;
    // A table that does not exist yet is a file with nothing in it — which is the genesis case, not an
    // error. Distinguishing "no table" from "a table with no rows" here would be a difference without a
    // meaning, and treating it as a fault would refuse every new file.
    let Ok(table) = txn.open_table(KV) else {
        return write_stamp(db, &expected);
    };
    let found: Option<StoreStamp> = table
        .get(STAMP_KEY)
        .map_err(|e| StoreError::Txn(e.to_string()))?
        .and_then(|v| postcard::from_bytes(v.value()).ok());
    // EMPTY MEANS "HOLDS NOTHING BUT ITS OWN LABEL". Counting the label as content would make a file
    // this build just created look like a file from before labels existed, and refuse it on its second
    // open — which is the failure this whole mechanism would be blamed for.
    let rows = table.len().map_err(|e| StoreError::Txn(e.to_string()))?;
    let stamp_rows = u64::from(found.is_some());
    let empty = rows.saturating_sub(stamp_rows) == 0;
    drop(table);
    drop(txn);

    match vd_core::store_stamp::verify(found, &expected, empty) {
        Err(source) => Err(StoreError::Stamp {
            path: path.display().to_string(),
            source,
        }),
        Ok(vd_core::store_stamp::StampVerdict::Proceed) => Ok(()),
        Ok(vd_core::store_stamp::StampVerdict::WriteAndProceed) => write_stamp(db, &expected),
    }
}

/// Write the label onto a file that has been accepted. Reached from two places — a file with no table at
/// all, and a file with an empty one — so it lives here rather than being written twice.
fn write_stamp(db: &Database, expected: &StoreStamp) -> Result<(), StoreError> {
    let bytes = postcard::to_allocvec(expected)
        .map_err(|e| StoreError::Txn(format!("encoding the store label: {e}")))?;
    let txn = db
        .begin_write()
        .map_err(|e| StoreError::Txn(e.to_string()))?;
    {
        let mut table = txn
            .open_table(KV)
            .map_err(|e| StoreError::Txn(e.to_string()))?;
        table
            .insert(STAMP_KEY, bytes.as_slice())
            .map_err(|e| StoreError::Txn(e.to_string()))?;
    }
    txn.commit().map_err(|e| StoreError::Txn(e.to_string()))?;
    Ok(())
}

impl RedbStore {
    /// Open (or create at genesis) the durable store at `path`, spawning the off-tick writer thread.
    /// Returns the `Store` plus the [`DurabilityHandle`] the bin gates effect-flush on. `Database::create`
    /// opens an existing file or creates a new one; the genesis-vs-recover discriminator stays the CLOCK
    /// family being absent (exposed as [`RedbStore::is_empty`], mirroring `MemStore`).
    /// `expected` is what THIS build believes about the world it is opening — see
    /// [`vd_core::store_stamp`]. It is not optional and there is no unstamped door: a label that is
    /// written but never read is worse than none, and this tree already holds one of those.
    ///
    /// # Errors
    /// [`StoreError::Stamp`] if the file was written by a different world, in different units, for a
    /// different job, or before labels existed at all.
    pub fn open(
        path: impl AsRef<Path>,
        tuning: StoreTuning,
        expected: StoreStamp,
    ) -> Result<(RedbStore, DurabilityHandle), StoreError> {
        let path = path.as_ref().to_path_buf();
        let db = Arc::new(Database::create(&path).map_err(|e| StoreError::Open(e.to_string()))?);
        // THE LABEL, BEFORE ANYTHING ELSE — including before the table-ensure write below.
        //
        // The order here is load-bearing and was found by MEASUREMENT, not by reasoning: the ensure is a
        // write transaction, and running it first changed the bytes of a file we then refused. A refusal
        // that modifies the thing it refuses is the worst of both worlds — the operator is told to go and
        // look at a file we have already touched. So: read first, refuse first, and only write once the
        // file has been accepted.
        verify_stamp(&db, &path, expected)?;

        // Materialize the table on a genesis file so `scan` is a clean empty read — one tiny txn at boot.
        let txn = db
            .begin_write()
            .map_err(|e| StoreError::Txn(e.to_string()))?;
        txn.open_table(KV)
            .map_err(|e| StoreError::Txn(e.to_string()))?;
        txn.commit().map_err(|e| StoreError::Txn(e.to_string()))?;

        let (tx, rx) = bounded::<(u64, Batch)>(tuning.writer_channel_depth);
        let last_durable = Arc::new(AtomicU64::new(0));
        let last_submitted = Arc::new(AtomicU64::new(0));
        let fsync_backpressure = Arc::new(AtomicU64::new(0));
        let durable_cv = Arc::new((Mutex::new(()), Condvar::new()));
        let durable_notify = Arc::new(Notify::new());
        let writer_alive = Arc::new(AtomicBool::new(true));
        let writer = {
            let db = Arc::clone(&db);
            let last_durable = Arc::clone(&last_durable);
            let durable_cv = Arc::clone(&durable_cv);
            let durable_notify = Arc::clone(&durable_notify);
            let writer_alive = Arc::clone(&writer_alive);
            #[cfg(feature = "store-test-hooks")]
            let pause_prefix = tuning.pause_on_key_prefix.clone();
            #[cfg(feature = "store-test-hooks")]
            let pause_marker = tuning.pause_marker_path.clone();
            #[cfg(feature = "store-test-hooks")]
            let fail_prefix = tuning.fail_fsync_on_key_prefix.clone();
            std::thread::Builder::new()
                .name("vd-store-writer".into())
                .spawn(move || {
                    run_writer(
                        db,
                        rx,
                        last_durable,
                        durable_cv,
                        durable_notify,
                        writer_alive,
                        #[cfg(feature = "store-test-hooks")]
                        pause_prefix,
                        #[cfg(feature = "store-test-hooks")]
                        pause_marker,
                        #[cfg(feature = "store-test-hooks")]
                        fail_prefix,
                    )
                })
                .map_err(|e| StoreError::Spawn(e.to_string()))?
        };
        let handle = DurabilityHandle {
            last_durable: Arc::clone(&last_durable),
            last_submitted: Arc::clone(&last_submitted),
            fsync_backpressure: Arc::clone(&fsync_backpressure),
            durable_cv: Arc::clone(&durable_cv),
            durable_notify: Arc::clone(&durable_notify),
            writer_alive: Arc::clone(&writer_alive),
            // R-6d4-B4: honor the test override (LARGE) so the async death-wake mutation gap is robust;
            // release has no override field ⇒ the const, byte-identical.
            #[cfg(feature = "store-test-hooks")]
            wait_poll: tuning.wait_poll_override.unwrap_or(WRITER_WAIT_POLL),
            #[cfg(not(feature = "store-test-hooks"))]
            wait_poll: WRITER_WAIT_POLL,
        };
        let store = RedbStore {
            db,
            path,
            staged: BTreeMap::new(),
            submit_tx: Some(tx),
            next_seq: 0,
            last_submitted,
            last_durable,
            fsync_backpressure,
            durable_cv,
            writer_alive,
            writer: Some(writer),
        };
        Ok((store, handle))
    }

    /// Whether anything is durably committed — the genesis-vs-recover discriminator (parallels
    /// `MemStore::is_empty`). FAIL-LOUD on a read fault (never degrade to `true` = a false genesis).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let txn = self
            .db
            .begin_read()
            .expect("RedbStore::is_empty: durable read fault — refusing to degrade to genesis");
        let table = txn
            .open_table(KV)
            .expect("RedbStore::is_empty: open_table fault — refusing to degrade to genesis");
        let rows = table
            .len()
            .expect("RedbStore::is_empty: len fault — refusing to degrade to genesis");
        // ★ THE FILE'S OWN LABEL IS NOT A RECORD. Every file this build opens carries one from the
        // moment it is created, so counting it would make "does this store hold anything?" answer YES on
        // a store that holds nothing — and the four durability assertions that ask exactly that would
        // fail on an empty file. Those assertions are the durability contract's own control and are not
        // to be weakened; the count is what was wrong.
        let stamp_rows = u64::from(
            table
                .get(STAMP_KEY)
                .expect("RedbStore::is_empty: label read fault — refusing to degrade to genesis")
                .is_some(),
        );
        rows.saturating_sub(stamp_rows) == 0
    }

    /// The store's file path (a reopen in tests / a clean re-attach by the bin).
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The seq of the last submitted batch — the bin records this per tick and gates that tick's outbox
    /// flush on `handle.is_durable_through(seq)` (the persist-before-effect gate, wired in Slice D).
    #[must_use]
    pub fn last_submitted(&self) -> u64 {
        self.last_submitted.load(Ordering::Acquire)
    }

    /// Block until every submitted batch is durable. Used by the tests; the bin uses the
    /// [`DurabilityHandle`]'s own `wait_durable_through` (same park) gating the per-tick outbox flush.
    pub fn flush_blocking(&self) {
        self.wait_durable_through(self.last_submitted.load(Ordering::Acquire));
    }

    /// R-6d3a: submit the staged batch to the writer WITHOUT the block-on-prior wait, returning the assigned
    /// batch seq (the caller awaits it on a cloned [`DurabilityHandle`] OUTSIDE any shared lock — the MF-1
    /// durable-before-send gate), or `None` if nothing was staged (MF-2). Distinct from [`Store::commit`],
    /// which block-on-priors under the caller's lock — the cross-peer-serialization hazard when ONE store is
    /// shared by N concurrent writer tasks (the outbox case). The seq-assign + `try_send` are ATOMIC here
    /// (called under the caller's lock), so the writer drains in strict seq order and durability stays
    /// monotone. The CALLER is responsible for bounding in-flight: the outbox sizes its channel
    /// [`OUTBOX_WRITER_CHANNEL_DEPTH`](crate::outbox::OUTBOX_WRITER_CHANNEL_DEPTH) >> its peer count (each
    /// peer-writer submits at most one un-durable batch before parking on the gate), so a `Full` here is an
    /// impossible-capacity-violation tripwire, NEVER a load-reachable back-pressure path (blocking under the
    /// shared lock is the very hazard this method exists to avoid). An un-fsynced in-channel batch is an
    /// un-SENT frame (the gate holds the send until durable), re-emitted by the restarted source on crash —
    /// so the outbox needs NO depth-1 crash-loss bound, only this per-send gate (MF-3).
    pub(crate) fn submit_nonblocking(&mut self) -> Option<u64> {
        if self.staged.is_empty() {
            return None; // MF-2: an idle / release-only span produces no durable barrier
        }
        let batch = std::mem::take(&mut self.staged);
        let seq = self.next_seq + 1;
        let tx = match &self.submit_tx {
            Some(tx) => tx.clone(),
            None => {
                // No sender (only reachable mid-Drop) → retain (fail-safe) + LOUD, mirroring `commit`.
                tracing::error!(
                    "RedbStore::submit_nonblocking with no writer (post-Drop) — staged batch retained"
                );
                self.staged = batch;
                return None;
            }
        };
        match tx.try_send((seq, batch)) {
            Ok(()) => {
                self.next_seq = seq;
                self.last_submitted.store(seq, Ordering::Release);
                Some(seq)
            }
            Err(TrySendError::Full((_, batch))) => {
                // Depth ≫ peer count ⇒ unreachable under any realistic concurrency; if it EVER fires the
                // capacity assumption was violated. Retain (fail-safe) + fail LOUD rather than block under
                // the caller's shared lock (the MF-1 cross-peer-serialization hazard; a refusal is never a
                // loss — recovery rehydrates + the source re-drives).
                self.staged = batch;
                panic!(
                    "RedbStore::submit_nonblocking: writer channel full — concurrent durable submits \
                     exceeded the channel depth; refusing to block under the shared sink lock"
                );
            }
            Err(TrySendError::Disconnected((_, batch))) => {
                // Writer DIED while the store is live — same fail-loud posture as `commit` (channel gone ⇒
                // un-persistable state). Retain (fail-safe) + panic; recovery rehydrates + re-drives.
                self.staged = batch;
                panic!(
                    "RedbStore::submit_nonblocking: the durable writer thread died (channel disconnected) — \
                     refusing to run on un-persistable state (a refusal is never a loss)"
                );
            }
        }
    }

    /// Park (not hot-spin) until `last_durable >= target`. Fast-path the common already-durable case; on a
    /// miss, lock + re-check + condvar-wait (the writer notifies under the same lock — no lost wakeup).
    ///
    /// LIVENESS ESCAPE (review `wf_186fc41d`): the writer notifies promptly on the happy path, but a writer
    /// that DIED (panicked, or EXITED on a permanent fsync fault) can never bump `last_durable` — so we
    /// `wait_timeout` and, on a timeout with no progress, fail LOUD if the writer has finished, rather than
    /// hang the synchronous orchestrator thread forever. A refusal is never a loss: recovery rehydrates the
    /// last durable state + the Slice-2a producer re-drives idempotently.
    fn wait_durable_through(&self, target: u64) {
        // ONE liveness signal shared with the handle: the `writer_alive` flag the writer clears on its single
        // exit (the `JoinHandle` stays only for Drop's join). The shared park core handles the fast-path, the
        // condvar wait, and the fail-loud escape.
        park_until_durable(target, &self.last_durable, &self.durable_cv, || {
            !self.writer_alive.load(Ordering::Acquire)
        });
    }
}

impl Store for RedbStore {
    fn put(&mut self, key: &[u8], value: &Bytes) {
        self.staged.insert(key.to_vec(), Some(value.clone()));
    }

    fn delete(&mut self, key: &[u8]) {
        self.staged.insert(key.to_vec(), None);
    }

    fn scan(&self, prefix: &[u8]) -> Vec<(Vec<u8>, Bytes)> {
        // FAIL-LOUD on a durable read fault: the seam's `scan` is infallible (Vec, no Result), so a SILENT
        // empty would let `rehydrate` misread a FAULTING non-empty store as genesis. Panic — a refusal is
        // never a loss. (Only a genuine I/O/corruption fault reaches these `expect`s.)
        let txn = self
            .db
            .begin_read()
            .expect("RedbStore::scan: begin_read fault — refusing to degrade to empty");
        let table = txn
            .open_table(KV)
            .expect("RedbStore::scan: open_table fault — refusing to degrade to empty");
        let range = table
            .range(prefix..)
            .expect("RedbStore::scan: range fault — refusing to degrade to empty");
        let mut out = Vec::new();
        for entry in range {
            let (k, v) =
                entry.expect("RedbStore::scan: mid-range read fault — refusing to truncate");
            let key = k.value();
            if !key.starts_with(prefix) {
                break; // sorted: the prefix run has ended
            }
            out.push((key.to_vec(), bytes(v.value().to_vec())));
        }
        out
    }

    fn commit(&mut self) {
        // BLOCK-ON-PRIOR (the depth-1 invariant + the back-pressure for approach A): wait until the
        // PREVIOUS batch is durable before submitting the next. ~always already true at 50Hz; a real wait
        // (counted) only under a disk stall — durability over liveness, the PvP-correct trade. This bounds
        // in-flight to ≤1 batch, so crash-loss is ≤1 batch. (D-alpha made the directory reconcile INCREMENTAL
        // — a `dirty`-delta drain that never reads the store back — so the off-tick writer no longer races a
        // reconcile read: the prior "scan reads T-1" coupling is GONE, leaving crash-loss bounding the sole job.)
        let last_submitted = self.last_submitted.load(Ordering::Acquire);
        if self.last_durable.load(Ordering::Acquire) < last_submitted {
            self.fsync_backpressure.fetch_add(1, Ordering::Relaxed);
            self.wait_durable_through(last_submitted);
        }
        if self.staged.is_empty() {
            return; // idle tick: nothing to submit
        }
        let batch = std::mem::take(&mut self.staged);
        let seq = self.next_seq + 1;
        let tx = match &self.submit_tx {
            Some(tx) => tx.clone(),
            None => {
                // No sender (only reachable mid-Drop) → retain (fail-safe). LOUD: a retained batch with no
                // writer is an alert, never a silent no-op (review wf_186fc41d).
                tracing::error!(
                    "RedbStore: commit with no writer (post-Drop) — staged batch retained"
                );
                self.staged = batch;
                return;
            }
        };
        // Block-on-prior guarantees the prior batch was drained, so the depth slot is free → this send does
        // not block. A disconnected channel (writer DIED) → retain the batch (a refusal is never a loss) +
        // fail LOUD; the next block-on-prior's liveness escape then panics rather than hang.
        match tx.send((seq, batch)) {
            Ok(()) => {
                self.next_seq = seq;
                self.last_submitted.store(seq, Ordering::Release);
            }
            Err(SendError(_)) => {
                // The channel is disconnected ⇒ the writer thread is GONE while the store is live. FAIL
                // LOUD here, do NOT retain-and-return: a silent retain lets the sim believe it persisted
                // (and, if the writer died caught-up, leaves the bin's parked seq stale ⇒ a not-yet-durable
                // effect would flush — the persist-before-effect hole review wf_D-gamma found). The
                // block-on-prior escape only fires when durable-BEHIND, so it cannot be relied on here. A
                // panic is a refusal, never a loss: recovery rehydrates the last durable state + re-drives.
                panic!(
                    "RedbStore: the durable writer thread died (commit channel disconnected) — refusing to \
                     run on un-persistable state (a refusal is never a loss; recovery rehydrates + re-drives)"
                );
            }
        }
    }

    fn flush(&mut self) {
        // RLM Step 5e write-ahead-before-fork: park until EVERY submitted batch (incl. the one `commit`
        // just handed the off-tick writer) has fsynced — `commit` is block-on-PRIOR, so without this the
        // caller's LAST commit is durable only after the NEXT commit. Reuses the existing durability park
        // (same `backpressure_stalls()` accounting, same writer-death liveness escape).
        self.flush_blocking();
    }
}

impl Drop for RedbStore {
    fn drop(&mut self) {
        // Drop the ONLY sender first → the writer drains the channel (fsyncing every pending batch) then
        // sees the disconnect and exits; THEN join. Graceful shutdown loses nothing (improves on a writer
        // whose queued batch would be silently lost on teardown).
        self.submit_tx = None;
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
    }
}

/// The label a test opens a store under. One place, so a test never states a generation by hand — the
/// whole point of the mechanism is that those two numbers are derived and un-stateable.
#[cfg(test)]
fn test_stamp() -> vd_core::store_stamp::StoreStamp {
    vd_core::store_stamp::StoreStamp::new(
        vd_core::store_stamp::StoreRole::Directory,
        0,
        vd_core::EpochId(0),
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A process-unique temp path (tests run in parallel; pid + a counter avoid collisions).
    fn temp_path() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("vd-redbstore-{}-{n}.redb", std::process::id()))
    }

    fn b(v: &[u8]) -> Bytes {
        bytes(v.to_vec())
    }

    fn open(path: &Path) -> (RedbStore, DurabilityHandle) {
        RedbStore::open(path, StoreTuning::default(), test_stamp()).expect("open store")
    }

    #[test]
    fn commit_persists_and_survives_a_reopen() {
        // THE durability barrier: staged writes become durable at commit() + the off-tick fsync, and
        // SURVIVE a reopen — the crash-recovery flow. flush_blocking() awaits the async writer.
        let path = temp_path();
        {
            let (mut s, _h) = open(&path);
            assert!(s.is_empty(), "a fresh store is genesis");
            s.put(b"k1", &b(b"v1"));
            s.put(b"k2", &b(b"v2"));
            assert!(
                s.scan(b"k").is_empty(),
                "staged writes are invisible until commit"
            );
            s.commit();
            s.flush_blocking();
            assert_eq!(s.scan(b"k1"), vec![(b"k1".to_vec(), b(b"v1"))]);
        }
        {
            let (s, _h) = open(&path);
            assert!(!s.is_empty(), "committed state recovered on reopen");
            assert_eq!(
                s.scan(b"k"),
                vec![(b"k1".to_vec(), b(b"v1")), (b"k2".to_vec(), b(b"v2"))],
                "both committed records recovered, ascending"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn store_flush_seam_parks_until_the_committed_batch_is_durable() {
        // RLM 5e D1: the `Store::flush` seam (distinct from `commit`, which is block-on-PRIOR) parks until
        // every already-committed batch is fsync'd — `spawn_realm` calls it so the launch intent is ON DISK
        // before the child forks. After flush, the last submitted batch is durable.
        let path = temp_path();
        let (mut s, h) = open(&path);
        s.put(b"launch-intent", &b(b"v1"));
        s.commit();
        s.flush(); // the new Store trait method → RedbStore::flush → flush_blocking
        assert!(
            h.durable_through() >= h.last_submitted(),
            "flush parked until the committed batch fsync'd (durable caught up to submitted)"
        );
        assert_eq!(
            s.scan(b"launch-intent"),
            vec![(b"launch-intent".to_vec(), b(b"v1"))]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn staged_writes_are_lost_without_commit() {
        // A crash BEFORE commit loses the staged batch — also models crash case 1 (kill before submit).
        let path = temp_path();
        {
            let (mut s, _h) = open(&path);
            s.put(b"k1", &b(b"v1"));
            // no commit; drop models crash-before-submit
        }
        {
            let (s, _h) = open(&path);
            assert!(s.is_empty(), "an uncommitted staged batch is NOT durable");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn last_write_wins_and_delete_nets_within_a_window() {
        let path = temp_path();
        {
            let (mut s, _h) = open(&path);
            s.put(b"k", &b(b"first"));
            s.put(b"k", &b(b"second")); // last write wins
            s.put(b"d", &b(b"x"));
            s.delete(b"d"); // put-then-delete nets to absent
            s.commit();
            s.flush_blocking();
            assert_eq!(s.scan(b"k"), vec![(b"k".to_vec(), b(b"second"))]);
            assert!(s.scan(b"d").is_empty(), "put-then-delete nets to absent");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn scan_filters_by_prefix_ascending() {
        let path = temp_path();
        {
            let (mut s, _h) = open(&path);
            s.put(&[1, 9], &b(b"a"));
            s.put(&[2, 0], &b(b"b"));
            s.put(&[2, 5], &b(b"c"));
            s.put(&[3, 0], &b(b"d"));
            s.commit();
            s.flush_blocking();
            assert_eq!(
                s.scan(&[2]),
                vec![(vec![2, 0], b(b"b")), (vec![2, 5], b(b"c"))],
                "scan returns only the prefix run, ascending"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn delete_removes_a_committed_key_durably() {
        let path = temp_path();
        {
            let (mut s, _h) = open(&path);
            s.put(b"k", &b(b"v"));
            s.commit();
            s.delete(b"k");
            s.commit();
            s.flush_blocking();
            assert!(
                s.scan(b"k").is_empty(),
                "a committed delete removes the key"
            );
        }
        {
            let (s, _h) = open(&path);
            assert!(s.is_empty(), "the delete is durable across reopen");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn commit_with_nothing_staged_is_a_noop() {
        let path = temp_path();
        {
            let (mut s, _h) = open(&path);
            s.commit(); // idle tick: no staged → no submit, no writer wake
            s.flush_blocking();
            assert!(s.is_empty());
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_watermark_advances_per_committed_batch() {
        // C2: the DurabilityHandle watermark trails-then-catches commit. After flush_blocking, the handle
        // reports the latest submitted seq as durable — the persist-before-effect gate's read side.
        let path = temp_path();
        {
            let (mut s, h) = open(&path);
            assert_eq!(h.durable_through(), 0, "genesis: nothing durable");
            s.put(b"a", &b(b"1"));
            s.commit();
            s.flush_blocking();
            assert_eq!(h.durable_through(), s.last_submitted());
            assert!(h.is_durable_through(s.last_submitted()));
            let after_first = s.last_submitted();
            s.put(b"b", &b(b"2"));
            s.commit(); // block-on-prior: could only submit because the 1st is already durable
            s.flush_blocking();
            assert!(
                s.last_submitted() > after_first,
                "the seq advanced monotonically"
            );
            assert_eq!(h.durable_through(), s.last_submitted());
        }
        let _ = std::fs::remove_file(&path);
    }

    /// R-6d3b F2: the async gate FAST-RETURNS when the target seq is already durable — no enroll, no await
    /// (the ~always case at 50Hz). If it hung, the current-thread runtime's `block_on` would never complete.
    #[test]
    fn async_wait_fast_returns_when_already_durable() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(async {
            DurabilityHandle::already_durable()
                .wait_durable_through_async(42)
                .await;
        });
    }

    /// R-6d3b F2: the async gate on a NOT-yet-durable seq enrolls + awaits the writer's `notify_waiters()`
    /// (or fast-returns if the off-tick writer raced ahead) and returns ONLY once the row is fsynced — the
    /// durable-before-send guarantee at the async seam. `submit_nonblocking` submits WITHOUT the sync
    /// block-on-prior wait, so the await genuinely gates on the async writer.
    #[test]
    fn async_wait_returns_after_a_real_submit_is_fsynced() {
        let path = temp_path();
        {
            let (mut s, h) = open(&path);
            s.put(b"k", &b(b"v"));
            let seq = s.submit_nonblocking().expect("a staged put ⇒ submitted");
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("rt");
            rt.block_on(async { h.wait_durable_through_async(seq).await });
            assert!(
                h.is_durable_through(seq),
                "the async wait returns ONLY once the submitted seq is durable"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// R-6d4-B4 (the /goal-audit BINDING GATE on R-6d4-M): a durable writer that DIES (permanent fsync
    /// fault) wakes an enrolled ASYNC durability waiter via `notify_waiters()`, and the R-6d4-M death
    /// short-circuit panics it PROMPTLY — not after the `WRITER_WAIT_POLL` backstop. MUTATION-SENSITIVITY:
    /// `wait_poll_override` is set LARGE (10s), so a neutered death `notify_waiters()` (or a lost R-6d4-M
    /// short-circuit) would HANG the await to that 10s timeout — far past the <2s assert ⇒ RED. This is the
    /// exact `wf_75a225d0` gap (a 100ms release poll was too tight to distinguish wake-vs-timeout). The
    /// content-keyed fsync fault kills the writer DETERMINISTICALLY (vs `pause_on_key_prefix`, which parks
    /// forever + hangs Drop's join). Flips M's death branch from region-floor-tolerated to PROVEN.
    #[cfg(feature = "store-test-hooks")]
    #[test]
    fn async_wait_wakes_and_fails_loud_promptly_on_writer_death() {
        let path = temp_path();
        let prefix = b"\x09DIE".to_vec();
        let tuning = StoreTuning {
            fail_fsync_on_key_prefix: Some(prefix.clone()),
            wait_poll_override: Some(Duration::from_secs(10)), // >> the prompt death-wake ⇒ mutation gap
            ..StoreTuning::default()
        };
        let (mut s, h) = RedbStore::open(&path, tuning, test_stamp())
            .expect("open with the fault + poll-override hooks");

        // Stage a row whose key carries the fault prefix + submit (no sync wait): the writer receives this
        // batch, force-fails its fsync WRITER_FSYNC_MAX_RETRIES times (~30ms), then `break 'drain`s + DIES.
        let mut key = prefix.clone();
        key.extend_from_slice(b"-row");
        s.put(&key, &b(b"v"));
        let seq = s.submit_nonblocking().expect("a staged put ⇒ submitted");

        let h_check = h.clone(); // the task consumes `h`; keep a clone for the post-death durability assert
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("rt");
        let start = std::time::Instant::now();
        let joined = rt.block_on(async move {
            // The enrolled await wakes on the writer's death `notify_waiters` ⇒ the R-6d4-M short-circuit
            // PANICS it; the spawned task surfaces that panic as a JoinError (never returns Ok).
            tokio::spawn(async move { h.wait_durable_through_async(seq).await }).await
        });
        let elapsed = start.elapsed();

        let join_err =
            joined.expect_err("the death short-circuit panics the await (never returns Ok)");
        assert!(
            join_err.is_panic(),
            "the task PANICKED on writer death (not cancelled)"
        );
        let panic = join_err.into_panic();
        let msg = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            msg.contains("died before seq"),
            "the fail-loud death message (not a timeout hang): {msg:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "the death WAKE panics PROMPTLY ({elapsed:?}) — far under the injected 10s poll; a neutered \
             notify_waiters / lost short-circuit would hang to ~10s, so this asserts the WAKE mechanism"
        );
        // The writer DIED forcing the fault ⇒ the batch was never fsynced: a refusal, never a durable lie.
        assert!(
            !h_check.is_durable_through(seq),
            "the faulted batch was never made durable"
        );
        // The writer already EXITED (break 'drain), so Drop's join returns immediately — no hang, no forget.
        drop(s);
        let _ = std::fs::remove_file(&path);
    }

    /// R-6d4-B3: `wait_durable_through_async` YIELDS its tokio worker, so N producer-durability waits parked
    /// on a stalled writer do NOT occupy N workers and starve the mesh recv/ack/accept I/O. On
    /// `worker_threads(2)`: TWO waits are parked (both writers paused pre-fsync ⇒ genuinely un-durable) and a
    /// THIRD "mesh-I/O proxy" task must STILL run to completion. MUTATION-SENSITIVITY: swapping
    /// `wait_durable_through_async` for the SYNC `wait_durable_through` (a blocking condvar park) would occupy
    /// BOTH workers ⇒ the third task never schedules ⇒ the bounded `timeout` fires ⇒ RED.
    #[cfg(feature = "store-test-hooks")]
    #[test]
    fn two_parked_durability_waits_do_not_starve_a_third_worker_task() {
        // Open a store whose writer PARKS pre-fsync on a keyed row, submit that row (no sync wait), and wait
        // for the marker (writer parked) — the row is submitted-but-un-durable, so a wait on its seq blocks.
        fn paused_store(prefix: &[u8]) -> (RedbStore, DurabilityHandle, u64) {
            let path = temp_path();
            let marker = path.with_extension("paused");
            let _ = std::fs::remove_file(&marker);
            let tuning = StoreTuning {
                pause_on_key_prefix: Some(prefix.to_vec()),
                pause_marker_path: Some(marker.clone()),
                ..StoreTuning::default()
            };
            let (mut s, h) =
                RedbStore::open(&path, tuning, test_stamp()).expect("open paused store");
            let mut key = prefix.to_vec();
            key.extend_from_slice(b"row");
            s.put(&key, &b(b"v"));
            let seq = s.submit_nonblocking().expect("a staged put ⇒ submitted");
            for _ in 0..500 {
                if marker.exists() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(marker.exists(), "the writer parked pre-fsync");
            (s, h, seq)
        }

        let (s1, h1, seq1) = paused_store(b"\x09P1");
        let (s2, h2, seq2) = paused_store(b"\x09P2");
        let done1 = std::sync::Arc::new(AtomicBool::new(false));
        let done2 = std::sync::Arc::new(AtomicBool::new(false));
        let ran3 = std::sync::Arc::new(AtomicBool::new(false));

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(async {
            let (d1, d2, r3) = (
                std::sync::Arc::clone(&done1),
                std::sync::Arc::clone(&done2),
                std::sync::Arc::clone(&ran3),
            );
            // Two producer-durability waits — parked forever (their writers are paused pre-fsync). Each
            // YIELDS its worker on every `.await`, so neither pins a worker thread.
            tokio::spawn(async move {
                h1.wait_durable_through_async(seq1).await;
                d1.store(true, Ordering::Release); // unreachable while paused (asserted below)
            });
            tokio::spawn(async move {
                h2.wait_durable_through_async(seq2).await;
                d2.store(true, Ordering::Release);
            });
            // The third "mesh-I/O proxy" task: must be schedulable on a freed worker while the two waits park.
            let (tx, rx) = tokio::sync::oneshot::channel::<()>();
            let t3 = tokio::spawn(async move {
                let _ = rx.await;
                r3.store(true, Ordering::Release);
            });
            tokio::task::yield_now().await; // let the two waits get polled once + park at their `.await`
            tx.send(()).expect("fire the third task");
            tokio::time::timeout(Duration::from_secs(2), t3)
                .await
                .expect(
                    "the third worker task ran despite two parked durability waits (async yields)",
                )
                .expect("t3 joined");
        });

        assert!(
            ran3.load(Ordering::Acquire),
            "the third worker task ran to completion while two durability waits were parked (no starvation)"
        );
        assert!(
            !done1.load(Ordering::Acquire),
            "wait 1 is STILL parked (its writer is paused ⇒ never durable) — not spuriously completed"
        );
        assert!(!done2.load(Ordering::Acquire), "wait 2 is STILL parked");

        std::mem::forget(s1); // parked writers never join
        std::mem::forget(s2);
    }

    #[test]
    fn drop_joins_the_writer_and_flushes_pending() {
        // Graceful shutdown loses nothing: commit WITHOUT flush_blocking, then drop — Drop drops the
        // sender (writer drains) + joins (every pending batch fsynced) before teardown. Reopen sees it.
        let path = temp_path();
        {
            let (mut s, _h) = open(&path);
            s.put(b"k", &b(b"v"));
            s.commit();
            // NO flush_blocking — rely on Drop to join + fsync the pending batch.
        }
        {
            let (s, _h) = open(&path);
            assert!(
                !s.is_empty(),
                "Drop joined the writer + fsynced the pending batch (graceful shutdown loses nothing)"
            );
            assert_eq!(s.scan(b"k"), vec![(b"k".to_vec(), b(b"v"))]);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_handle_reports_submitted_durable_and_liveness() {
        // Slice D: the bin gates each tick's outbox flush on the SIDECAR handle. Exercise the new surface —
        // last_submitted (the parked seq), wait_durable_through (the persist-before-effect park), and
        // writer_alive (the fail-loud liveness, which flips false once the writer exits on Drop's join).
        let path = temp_path();
        let (mut s, h) = open(&path);
        assert_eq!(h.last_submitted(), 0, "nothing submitted at genesis");
        assert!(h.writer_alive(), "the writer is spawned");
        s.put(b"\x03k", &b(b"v"));
        s.commit();
        let seq = h.last_submitted();
        assert_eq!(
            seq, 1,
            "one batch submitted, watermark visible through the handle"
        );
        h.wait_durable_through(seq); // parks until durable (or returns immediately on the fast path)
        assert!(h.is_durable_through(seq), "durable after the wait");
        assert_eq!(h.durable_through(), seq);
        drop(s); // Drop joins the writer → it exits → liveness flips false (handle keeps the shared flag)
        assert!(
            !h.writer_alive(),
            "the writer is gone after the store drops"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn redbstore_is_byte_identical_to_memstore_under_the_store_seam() {
        // The RedbStore ARM of the D-alpha differential oracle. The node oracle proves the directory
        // reconcile LOGIC is correct on a faithful Store (MemStore); this proves RedbStore IS faithful —
        // byte-for-byte the same put/delete/commit/scan semantics as the proptested MemStore — so the two
        // compose to incremental-reconcile == full-reconcile ON REDB. Drives a reconcile-shaped op sequence
        // (PUTs, within-window last-write-wins, DELETEs, multi-prefix, an idle commit) against BOTH and
        // compares each prefix scan after every window; a reopen proves the equivalence survives a kill-9.
        use vd_sim::io::mem::MemStore;
        // One staged op: a key + (Some=put / None=delete). Aliased so the window literal is not a
        // clippy::type_complexity violation.
        type KvOp<'a> = (&'a [u8], Option<&'a [u8]>);
        let path = temp_path();
        let (mut redb, h) = open(&path);
        let mut mem = MemStore::new();
        let windows: &[&[KvOp]] = &[
            &[
                (b"\x03a", Some(b"v1")),
                (b"\x03b", Some(b"v2")),
                (b"\x01s", Some(b"saga")),
            ],
            &[(b"\x03a", Some(b"v1b")), (b"\x03a", Some(b"v1c"))], // last-write-wins on \x03a
            &[(b"\x03b", None)],                                   // delete
            &[],                                                   // idle commit (nothing staged)
            &[(b"\x03c", Some(b"v3")), (b"\x01s", None)], // a put + a delete across prefixes
        ];
        for window in windows {
            for (k, v) in *window {
                match v {
                    Some(val) => {
                        redb.put(k, &b(val));
                        mem.put(k, &b(val));
                    }
                    None => {
                        redb.delete(k);
                        mem.delete(k);
                    }
                }
            }
            redb.commit();
            mem.commit();
            h.wait_durable_through(h.last_submitted());
            for prefix in [b"\x03".as_slice(), b"\x01".as_slice()] {
                assert_eq!(
                    redb.scan(prefix),
                    mem.scan(prefix),
                    "RedbStore diverged from MemStore after a window"
                );
            }
        }
        // REOPEN (the kill-9 round-trip): the durable RedbStore state still equals MemStore's final state.
        drop(redb);
        let (redb2, _h2) = open(&path);
        for prefix in [b"\x03".as_slice(), b"\x01".as_slice()] {
            assert_eq!(
                redb2.scan(prefix),
                mem.scan(prefix),
                "RedbStore lost or changed durable state across a reopen"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// D-6 D-delta: the feature-gated writer-pause hook (the SIGKILL-mid-fsync proof's mechanism). A
    /// NON-sentinel batch fsyncs normally; the FIRST batch carrying the sentinel prefix parks the writer
    /// BEFORE fsync (marker written, batch submitted-but-never-durable) — the deterministic crash window.
    #[cfg(feature = "store-test-hooks")]
    #[test]
    fn the_writer_pauses_before_fsync_only_on_the_sentinel_batch() {
        let path = temp_path();
        let marker = path.with_extension("paused");
        let _ = std::fs::remove_file(&marker);
        let tuning = StoreTuning {
            pause_on_key_prefix: Some(b"\x09SENT".to_vec()),
            pause_marker_path: Some(marker.clone()),
            ..StoreTuning::default()
        };
        let (mut s, h) =
            RedbStore::open(&path, tuning, test_stamp()).expect("open with pause hook");

        // A non-sentinel batch fsyncs normally — no pause, no marker.
        s.put(b"\x01a", &b(b"v"));
        s.commit();
        h.wait_durable_through(h.last_submitted());
        assert!(
            !marker.exists(),
            "a non-sentinel batch must NOT pause the writer"
        );
        assert_eq!(s.scan(b"\x01a"), vec![(b"\x01a".to_vec(), b(b"v"))]);

        // The sentinel batch: submitted (last_submitted bumps) but the writer parks BEFORE fsync, so it
        // never becomes durable, and the marker appears (the crash test's decoupled observable).
        s.put(b"\x09SENTINEL", &b(b"x"));
        s.commit();
        let submitted = h.last_submitted();
        let mut parked = false;
        for _ in 0..500 {
            if marker.exists() {
                parked = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            parked,
            "the writer parked on the sentinel batch (marker written)"
        );
        assert!(
            !h.is_durable_through(submitted),
            "the sentinel batch is submitted but NOT durable (paused pre-fsync)"
        );
        assert!(
            h.durable_through() < submitted,
            "durable watermark lags the parked batch"
        );

        // The parked writer NEVER returns, so Drop's join would hang — leak the store (the process exits
        // and reaps the thread; temp_path() is unique so the held file lock collides with nothing).
        std::mem::forget(s);
        let _ = std::fs::remove_file(&marker);
    }
}

#[cfg(test)]
mod stamp_gate {
    //! THE REFUSAL, PROVEN END TO END on a real file (owner ruling 2026-08-24 Q1 condition 1).
    //!
    //! The decision itself is unit-tested in `vd-core`, where every branch is held by the coverage gate.
    //! What is proven HERE is the part that crate cannot see: that the label is really written, really
    //! read back, really compared BEFORE anything else, and that a refused file is left ALONE.
    use super::*;
    use vd_core::EpochId;
    use vd_core::store_stamp::{StampRefusal, StoreRole, StoreStamp};

    /// A process-unique temp path, matching the sibling module's idiom (tests run in parallel).
    fn temp_path(tag: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("vd-stamp-{tag}-{}-{n}.redb", std::process::id()))
    }

    fn stamp_for(seed: u64) -> StoreStamp {
        StoreStamp::new(StoreRole::Directory, seed, EpochId(0), &[1.0])
    }

    #[test]
    fn a_store_written_under_one_world_is_refused_under_another_and_left_untouched() {
        // THE GATE THIS SLICE EXISTS FOR, and the half that matters most is the SECOND assertion. A
        // refusal that damaged the file would be worse than no refusal at all: the operator would have
        // lost the thing the message told them to go and look at.
        let path = temp_path("s");
        {
            let (mut store, _dur) =
                RedbStore::open(&path, StoreTuning::default(), stamp_for(1)).expect("first open");
            store.put(b"k", &bytes(b"v".to_vec()));
            store.commit();
            store.flush_blocking();
        }
        let before = std::fs::read(&path).expect("read the file back");

        let err = RedbStore::open(&path, StoreTuning::default(), stamp_for(2))
            .err()
            .expect("a store from another world must be refused");
        match err {
            StoreError::Stamp { source, .. } => assert_eq!(
                source,
                StampRefusal::UniverseSeed {
                    found: 1,
                    expected: 2
                }
            ),
            other => panic!("expected a label refusal, got {other:?}"),
        }

        let after = std::fs::read(&path).expect("read the file again");
        assert_eq!(
            before, after,
            "a refused file must be left byte-for-byte alone — the operator has to be able to go and \
             look at the thing the refusal named"
        );
    }

    #[test]
    fn the_same_world_reopens() {
        // The other side of the same coin: the ordinary restart must not be refused, or the mechanism
        // would be blamed for the outage it exists to prevent.
        let path = temp_path("s");
        {
            let (mut store, _dur) =
                RedbStore::open(&path, StoreTuning::default(), stamp_for(1)).expect("first open");
            store.put(b"k", &bytes(b"v".to_vec()));
            store.commit();
            store.flush_blocking();
        }
        let (store, _dur) =
            RedbStore::open(&path, StoreTuning::default(), stamp_for(1)).expect("reopen");
        assert_eq!(
            store.scan(b"k").len(),
            1,
            "the row written before the restart is still there"
        );
    }

    #[test]
    fn a_file_written_before_labels_existed_is_refused_never_adopted() {
        // THE OWNER'S RULING: refuse, do not convert. Simulated by writing a row through a store and
        // then removing its label — which is exactly the shape of a pre-label file: records, no label.
        let path = temp_path("s");
        {
            let (mut store, _dur) =
                RedbStore::open(&path, StoreTuning::default(), stamp_for(1)).expect("first open");
            store.put(b"k", &bytes(b"v".to_vec()));
            store.commit();
            store.flush_blocking();
            store.delete(STAMP_KEY);
            store.commit();
            store.flush_blocking();
        }
        let err = RedbStore::open(&path, StoreTuning::default(), stamp_for(1))
            .err()
            .expect("an unlabelled file that holds data must be refused");
        match err {
            StoreError::Stamp { source, .. } => {
                assert_eq!(source, StampRefusal::UnstampedAndNotEmpty);
            }
            other => panic!("expected a label refusal, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_unlabelled_file_is_labelled_and_opens() {
        // The genesis case, which must NOT be refused — and the label must actually be written, or the
        // second open would refuse a file this build itself created.
        let path = temp_path("s");
        {
            let (store, _dur) =
                RedbStore::open(&path, StoreTuning::default(), stamp_for(3)).expect("genesis open");
            assert!(
                store.is_empty(),
                "a fresh store holds no RECORDS — its label is not one"
            );
        }
        let (store, _dur) =
            RedbStore::open(&path, StoreTuning::default(), stamp_for(3)).expect("second open");
        assert!(store.is_empty());
    }

    #[test]
    fn an_outbox_cannot_be_opened_as_a_saga_log() {
        // WHY THE ROLE IS IN THE LABEL. Our encoding is positional and not self-describing: opening one
        // family's file as another's yields RECORDS, not an error. This is the refusal that stops it.
        let path = temp_path("s");
        {
            let (mut store, _dur) = RedbStore::open(
                &path,
                StoreTuning::default(),
                StoreStamp::new(StoreRole::Outbox, 1, EpochId(0), &[1.0]),
            )
            .expect("open as an outbox");
            store.put(b"k", &bytes(b"v".to_vec()));
            store.commit();
            store.flush_blocking();
        }
        let err = RedbStore::open(&path, StoreTuning::default(), stamp_for(1))
            .err()
            .expect("an outbox opened as a directory must be refused");
        match err {
            StoreError::Stamp { source, .. } => assert_eq!(
                source,
                StampRefusal::Role {
                    found: "outbox",
                    expected: "directory+saga"
                }
            ),
            other => panic!("expected a label refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_moved_coordinate_unit_refuses_every_existing_store() {
        // THE UNIT GATE — the one this whole mechanism was built for. A store written under one set of
        // coordinate units must be refused by a build that counts in another, because a position read
        // under the wrong unit is silently wrong by the ratio between them and nothing would crash.
        //
        // The shipped generation is DERIVED from the tier table, so this drives the refusal with a
        // hand-built label carrying a different generation — which is precisely what slice S8 will make
        // the real table produce when it re-values the coarse step.
        let path = temp_path("s");
        {
            let (mut store, _dur) =
                RedbStore::open(&path, StoreTuning::default(), stamp_for(1)).expect("first open");
            store.put(b"k", &bytes(b"v".to_vec()));
            store.commit();
            store.flush_blocking();
        }
        let mut moved = stamp_for(1);
        moved.coordinate_generation ^= 1;
        let err = RedbStore::open(&path, StoreTuning::default(), moved)
            .err()
            .expect("a store counted in other units must be refused");
        match err {
            StoreError::Stamp { source, .. } => assert!(
                matches!(source, StampRefusal::CoordinateGeneration { .. }),
                "the refusal must name the UNIT, not something downstream of it: {source:?}"
            ),
            other => panic!("expected a label refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_refused_store_is_wiped_only_when_genesis_is_explicitly_allowed() {
        // THE OTHER HALF OF A REFUSAL. Without a way forward, a refusal becomes an unofficial delete
        // typed under pressure — which removes more than was meant and leaves no record. With one, the
        // operator has a named, single-file, loud act.
        //
        // BOTH ARMS, because the dangerous one is the default: a wipe that could happen without being
        // asked for would eventually happen without being asked for, on a file somebody needed.
        let path = temp_path("genesis");
        {
            let (mut store, _dur) =
                RedbStore::open(&path, StoreTuning::default(), stamp_for(1)).expect("first open");
            store.put(b"k", &bytes(b"v".to_vec()));
            store.commit();
            store.flush_blocking();
        }
        let before = std::fs::read(&path).expect("read the file");

        // NOT ALLOWED: the refusal stands and the file is untouched.
        let err = open_allowing_genesis(&path, false, |p| {
            RedbStore::open(p, StoreTuning::default(), stamp_for(2))
        })
        .err()
        .expect("refused without the opt-in");
        assert!(matches!(err, StoreError::Stamp { .. }));
        assert_eq!(
            before,
            std::fs::read(&path).expect("read again"),
            "a refusal without the opt-in must not touch the file"
        );

        // ALLOWED: the file is deleted and a new world starts here.
        let (store, _dur) = open_allowing_genesis(&path, true, |p| {
            RedbStore::open(p, StoreTuning::default(), stamp_for(2))
        })
        .expect("the opt-in starts a new world");
        assert!(
            store.is_empty(),
            "the new world starts empty — the old one was deleted, which is what was asked for"
        );
    }

    #[test]
    fn the_wipe_is_not_reached_by_any_error_except_the_label() {
        // A file that cannot be opened for some OTHER reason is not a file to delete. Driven with an
        // opener that fails for a different reason, against a file that exists: it must survive.
        let path = temp_path("other-error");
        std::fs::write(&path, b"not a database").expect("plant a file");
        let before = std::fs::read(&path).expect("read the file");

        let err = open_allowing_genesis(&path, true, |_p| {
            Err::<(), _>(StoreError::Open("a fault that is not a label".to_owned()))
        })
        .expect_err("the opener's own error is returned");
        assert!(matches!(err, StoreError::Open(_)));
        assert_eq!(
            before,
            std::fs::read(&path).expect("read again"),
            "only a LABEL refusal may lead to a delete"
        );
    }

    #[test]
    fn wiping_a_file_that_is_not_there_is_not_an_error() {
        // The state being asked for is "no file". A path that already has none is in that state.
        let path = temp_path("absent");
        wipe_for_genesis(&path).expect("a path with no file is already in the asked-for state");
    }

    #[test]
    fn the_refusal_names_the_file_so_an_operator_can_act_on_it() {
        // A refusal an operator cannot act on becomes an unofficial delete, which is the loss the
        // refusal exists to prevent. So the message must name WHICH file, not merely what was wrong.
        let path = temp_path("named");
        {
            let (mut store, _dur) =
                RedbStore::open(&path, StoreTuning::default(), stamp_for(1)).expect("first open");
            store.put(b"k", &bytes(b"v".to_vec()));
            store.commit();
            store.flush_blocking();
        }
        let err = RedbStore::open(&path, StoreTuning::default(), stamp_for(2))
            .err()
            .expect("refused");
        let text = err.to_string();
        assert!(
            text.contains(&path.display().to_string()),
            "the refusal must name the file: {text}"
        );
        assert!(
            text.contains('1') && text.contains('2'),
            "and both values: {text}"
        );
    }
}
