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
//! Tier-B (HR5): io-prod is excluded from the 100% region+branch gate; the redb adapter is process-tier
//! covered at a ratcheted floor. The in-process durability tests here give high happy-path coverage; the
//! real-process SIGKILL-mid-fsync crash proof lands with the orchestrator wiring (Slice D).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, SendError, Sender, bounded};
use redb::{Database, ReadableTableMetadata, TableDefinition};
use vd_sim::io::{Bytes, Store, bytes};

/// The ONE key/value table. The keyspace is the EXACT flat, prefix-tagged `StoreKey` bytes the
/// orchestrator already writes through the [`Store`] seam (one prefix byte per family), so `scan(prefix)`
/// is a redb range over that prefix — byte-for-byte compatible with the `MemStore` tier. The split-ready
/// table-set seam (user decision: single file now, the D-32 Directory-file split later): a future split
/// adds a second `TableDefinition` / `Database` here behind a table-set descriptor, with no change to the
/// keyspace or the consumers, landing its required head-re-read WITH the split (never before).
const KV: TableDefinition<&[u8], &[u8]> = TableDefinition::new("kv");

/// A staged fsync window: `Some` = put, `None` = delete (last-write-wins per key).
type Batch = BTreeMap<Vec<u8>, Option<Bytes>>;

/// Operational tuning for the redb writer (ONE config struct — no inline magic numbers).
#[derive(Clone, Copy, Debug)]
pub struct StoreTuning {
    /// Bounded depth of the staged-batch hand-off channel. The EFFECTIVE in-flight depth is ONE
    /// (commit-blocks-on-prior); this is a small slack so a `send` never blocks the sim thread even
    /// momentarily mid-fsync, and the seam for a future depth>1 mode. NEVER set 0 (a rendezvous channel
    /// would couple the sim thread to the fsync).
    pub writer_channel_depth: usize,
}

impl Default for StoreTuning {
    fn default() -> Self {
        Self {
            writer_channel_depth: 2,
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
}

/// The bin-facing durability watermark (the persist-before-effect gate's read side). Cloneable + `Send`;
/// the orchestrator bin (Slice D) holds it alongside the `Box<dyn Store>` and gates a tick's outbox flush
/// on [`DurabilityHandle::is_durable_through`] so no effect leaves before its authorizing state is durable.
#[derive(Clone)]
pub struct DurabilityHandle {
    last_durable: Arc<AtomicU64>,
    fsync_backpressure: Arc<AtomicU64>,
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
    /// How many `commit()`s had to STALL on the previous fsync (the back-pressure counter — 0 on a healthy
    /// run; `> 0` means the disk could not keep up with the tick rate, the correct durability-over-liveness
    /// signal, mirroring the `TickPacer` overrun discipline).
    #[must_use]
    pub fn backpressure_stalls(&self) -> u64 {
        self.fsync_backpressure.load(Ordering::Acquire)
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
    /// The seq of the last SUCCESSFULLY-submitted batch (commit-blocks-on-prior waits for this to be durable).
    last_submitted: u64,
    /// The highest fsynced seq (bumped by the writer, Release). The lock-free source of truth.
    last_durable: Arc<AtomicU64>,
    /// Count of commits that stalled on the previous fsync (back-pressure observable).
    fsync_backpressure: Arc<AtomicU64>,
    /// The writer notifies on each `last_durable` bump so block-on-prior / `flush_blocking` PARK (not hot
    /// spin) under a disk stall. The atomic stays the source of truth; the mutex guards only the condvar.
    durable_cv: Arc<(Mutex<()>, Condvar)>,
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

/// The off-tick writer loop: drain a batch (coalescing any already-queued ones — vestigial under depth-1,
/// the seam for a future depth>1 mode), apply+fsync in ONE txn, then bump `last_durable` (Release) +
/// notify. On a fsync error the watermark is NOT bumped → the persist-before-effect gate stalls LOUD,
/// never a silent effect loss (the owed refinement is a bounded retry). Exits when the sender is dropped.
fn run_writer(
    db: Arc<Database>,
    rx: Receiver<(u64, Batch)>,
    last_durable: Arc<AtomicU64>,
    durable_cv: Arc<(Mutex<()>, Condvar)>,
) {
    while let Ok((first_seq, first)) = rx.recv() {
        let mut max_seq = first_seq;
        let mut merged = first;
        while let Ok((seq, batch)) = rx.try_recv() {
            for (k, v) in batch {
                merged.insert(k, v); // last-write-wins across coalesced batches
            }
            max_seq = seq;
        }
        if apply_batch(&db, &merged) {
            last_durable.store(max_seq, Ordering::Release);
            let (lock, cv) = &*durable_cv;
            let _g = lock.lock().unwrap_or_else(PoisonError::into_inner);
            cv.notify_all();
        }
        // else: logged LOUD in apply_batch; watermark NOT bumped → the gate stalls (no silent effect loss).
    }
}

impl RedbStore {
    /// Open (or create at genesis) the durable store at `path`, spawning the off-tick writer thread.
    /// Returns the `Store` plus the [`DurabilityHandle`] the bin gates effect-flush on. `Database::create`
    /// opens an existing file or creates a new one; the genesis-vs-recover discriminator stays the CLOCK
    /// family being absent (exposed as [`RedbStore::is_empty`], mirroring `MemStore`).
    pub fn open(
        path: impl AsRef<Path>,
        tuning: StoreTuning,
    ) -> Result<(RedbStore, DurabilityHandle), StoreError> {
        let path = path.as_ref().to_path_buf();
        let db = Arc::new(Database::create(&path).map_err(|e| StoreError::Open(e.to_string()))?);
        // Materialize the table on a genesis file so `scan` is a clean empty read — one tiny txn at boot.
        let txn = db.begin_write().map_err(|e| StoreError::Txn(e.to_string()))?;
        txn.open_table(KV).map_err(|e| StoreError::Txn(e.to_string()))?;
        txn.commit().map_err(|e| StoreError::Txn(e.to_string()))?;

        let (tx, rx) = bounded::<(u64, Batch)>(tuning.writer_channel_depth);
        let last_durable = Arc::new(AtomicU64::new(0));
        let fsync_backpressure = Arc::new(AtomicU64::new(0));
        let durable_cv = Arc::new((Mutex::new(()), Condvar::new()));
        let writer = {
            let db = Arc::clone(&db);
            let last_durable = Arc::clone(&last_durable);
            let durable_cv = Arc::clone(&durable_cv);
            std::thread::Builder::new()
                .name("vd-store-writer".into())
                .spawn(move || run_writer(db, rx, last_durable, durable_cv))
                .map_err(|e| StoreError::Spawn(e.to_string()))?
        };
        let handle = DurabilityHandle {
            last_durable: Arc::clone(&last_durable),
            fsync_backpressure: Arc::clone(&fsync_backpressure),
        };
        let store = RedbStore {
            db,
            path,
            staged: BTreeMap::new(),
            submit_tx: Some(tx),
            next_seq: 0,
            last_submitted: 0,
            last_durable,
            fsync_backpressure,
            durable_cv,
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
        table
            .len()
            .expect("RedbStore::is_empty: len fault — refusing to degrade to genesis")
            == 0
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
        self.last_submitted
    }

    /// Block until every submitted batch is durable. Used by `Drop` (graceful shutdown) + the tests; the
    /// bin uses the non-blocking [`DurabilityHandle`] watermark instead (never blocks the tick thread).
    pub fn flush_blocking(&self) {
        self.wait_durable_through(self.last_submitted);
    }

    /// Park (not hot-spin) until `last_durable >= target`. Fast-path the common already-durable case; on a
    /// miss, lock + re-check + condvar-wait (the writer notifies under the same lock — no lost wakeup).
    fn wait_durable_through(&self, target: u64) {
        if self.last_durable.load(Ordering::Acquire) >= target {
            return;
        }
        let (lock, cv) = &*self.durable_cv;
        let mut guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
        while self.last_durable.load(Ordering::Acquire) < target {
            guard = cv.wait(guard).unwrap_or_else(PoisonError::into_inner);
        }
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
            let (k, v) = entry.expect("RedbStore::scan: mid-range read fault — refusing to truncate");
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
        // in-flight to ≤1 batch, so crash-loss is ≤1 batch AND the per-tick reconcile scan reads exactly T-1.
        if self.last_durable.load(Ordering::Acquire) < self.last_submitted {
            self.fsync_backpressure.fetch_add(1, Ordering::Relaxed);
            self.wait_durable_through(self.last_submitted);
        }
        if self.staged.is_empty() {
            return; // idle tick: nothing to submit
        }
        let batch = std::mem::take(&mut self.staged);
        let seq = self.next_seq + 1;
        let tx = match &self.submit_tx {
            Some(tx) => tx.clone(),
            None => {
                self.staged = batch; // no sender (shutdown) → retain (fail-safe)
                return;
            }
        };
        // Block-on-prior guarantees the prior batch was drained, so the depth slot is free → this send does
        // not block. A disconnected channel (writer gone) → retain the batch (a refusal is never a loss).
        match tx.send((seq, batch)) {
            Ok(()) => {
                self.next_seq = seq;
                self.last_submitted = seq;
            }
            Err(SendError((_, b))) => self.staged = b,
        }
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
        RedbStore::open(path, StoreTuning::default()).expect("open store")
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
            assert!(s.scan(b"k").is_empty(), "a committed delete removes the key");
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
}
