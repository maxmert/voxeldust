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
//! C1 (this slice): a SYNCHRONOUS backend — `commit()` performs the redb WriteTransaction + fsync inline.
//! Correct + durable + contract-tested across a real reopen. It is NOT yet wired into the orchestrator, so
//! the inline fsync blocks no tick. C2 lands the OFF-TICK fsync writer thread + the `last_durable`
//! persist-before-effect gate (the design's threading model) BEFORE the orchestrator wiring (Slice D), so
//! the sim thread never blocks on disk.
//!
//! Tier-B (HR5): io-prod is excluded from the 100% region+branch gate; the redb adapter is process-tier
//! covered at a ratcheted floor. The in-process durability tests here give high happy-path coverage; the
//! SIGKILL-mid-fsync crash proof lands with the process tier (Slice D).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use redb::{Database, ReadableTableMetadata, TableDefinition};
use vd_sim::io::{Bytes, Store, bytes};

/// The ONE key/value table. The keyspace is the EXACT flat, prefix-tagged `StoreKey` bytes the
/// orchestrator already writes through the [`Store`] seam (one prefix byte per family), so `scan(prefix)`
/// is a redb range over that prefix — byte-for-byte compatible with the `MemStore` tier. The split-ready
/// table-set seam (user decision: single file now, the D-32 Directory-file split later): a future split
/// adds a second `TableDefinition` / `Database` here behind a table-set descriptor, with no change to the
/// keyspace or the consumers, landing its required head-re-read WITH the split (never before).
const KV: TableDefinition<&[u8], &[u8]> = TableDefinition::new("kv");

/// Errors opening or committing the durable store — a typed door, never a silent data path.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// Opening / creating the redb database failed.
    #[error("redb open/create: {0}")]
    Open(String),
    /// A transaction (the genesis table-ensure) failed during open.
    #[error("redb txn: {0}")]
    Txn(String),
}

/// The generic redb-backed [`Store`]. Holds the staged (this-fsync-window) mutations in RAM and the
/// committed state in the redb file; `commit()` applies the staged batch to redb in ONE WriteTransaction
/// (the atomic group-commit fsync); `scan()` reads the committed redb. Last-write-wins within a window is
/// free — the staged map keys on the byte key, so repeated `put`/`delete` at one key keep only the last.
pub struct RedbStore {
    db: Database,
    path: PathBuf,
    /// Staged this fsync window: `Some` = put, `None` = delete. Applied + cleared atomically at `commit`;
    /// RETAINED (never cleared) on any commit error so a transient disk fault retries, never silently
    /// drops the batch (fail-safe: a refusal is never a loss).
    staged: BTreeMap<Vec<u8>, Option<Bytes>>,
}

impl RedbStore {
    /// Open (or create at genesis) the durable store at `path`. `Database::create` opens an existing file
    /// or creates a new one; the genesis-vs-recover discriminator stays the CLOCK family being absent
    /// (the rehydrate gate, unchanged) — exposed here as [`RedbStore::is_empty`], mirroring `MemStore`.
    pub fn open(path: impl AsRef<Path>) -> Result<RedbStore, StoreError> {
        let path = path.as_ref().to_path_buf();
        let db = Database::create(&path).map_err(|e| StoreError::Open(e.to_string()))?;
        // Materialize the table on a genesis file so `scan` is a clean empty read, not a missing-table
        // error — one tiny write txn at boot only.
        let txn = db.begin_write().map_err(|e| StoreError::Txn(e.to_string()))?;
        txn.open_table(KV).map_err(|e| StoreError::Txn(e.to_string()))?;
        txn.commit().map_err(|e| StoreError::Txn(e.to_string()))?;
        Ok(RedbStore {
            db,
            path,
            staged: BTreeMap::new(),
        })
    }

    /// Whether anything is durably committed — the genesis-vs-recover discriminator (parallels
    /// `MemStore::is_empty`): a genesis store has an empty KV table. Staged-but-uncommitted does not count.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let Ok(txn) = self.db.begin_read() else {
            return true;
        };
        let Ok(table) = txn.open_table(KV) else {
            return true;
        };
        table.len().unwrap_or(0) == 0
    }

    /// The store's file path (a reopen in tests / a clean re-attach by the bin).
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
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
        let Ok(txn) = self.db.begin_read() else {
            return Vec::new();
        };
        let Ok(table) = txn.open_table(KV) else {
            return Vec::new();
        };
        let Ok(range) = table.range(prefix..) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for entry in range {
            let Ok((k, v)) = entry else {
                break;
            };
            let key = k.value();
            if !key.starts_with(prefix) {
                break; // sorted: the prefix run has ended
            }
            out.push((key.to_vec(), bytes(v.value().to_vec())));
        }
        out
    }

    fn commit(&mut self) {
        if self.staged.is_empty() {
            return; // idle tick: nothing staged → no write txn, no fsync
        }
        let txn = match self.db.begin_write() {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("RedbStore: begin_write failed, staged batch retained: {e}");
                return; // staged retained → retried next commit (fail-safe)
            }
        };
        {
            let mut table = match txn.open_table(KV) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!("RedbStore: open_table failed, staged batch retained: {e}");
                    return; // txn dropped (rolled back); staged retained
                }
            };
            for (key, value) in &self.staged {
                let applied = match value {
                    Some(v) => table.insert(key.as_slice(), v.as_ref()).map(|_| ()),
                    None => table.remove(key.as_slice()).map(|_| ()),
                };
                if let Err(e) = applied {
                    tracing::error!("RedbStore: apply failed for one key: {e}");
                }
            }
        }
        match txn.commit() {
            Ok(()) => self.staged.clear(), // durable → window closed
            Err(e) => tracing::error!("RedbStore: commit fsync failed, staged retained: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A process-unique temp path (tests run in parallel; the pid + a counter avoid collisions; a fresh
    /// pid each run avoids stale files).
    fn temp_path() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("vd-redbstore-{}-{n}.redb", std::process::id()))
    }

    fn b(v: &[u8]) -> Bytes {
        bytes(v.to_vec())
    }

    #[test]
    fn commit_persists_and_survives_a_reopen() {
        // THE durability barrier: staged writes become durable at commit() and SURVIVE a reopen — the
        // crash-recovery flow MemStore can only model via a retained handle; here it is a real file.
        let path = temp_path();
        {
            let mut s = RedbStore::open(&path).expect("open store");
            assert!(s.is_empty(), "a fresh store is genesis");
            s.put(b"k1", &b(b"v1"));
            s.put(b"k2", &b(b"v2"));
            assert!(
                s.scan(b"k").is_empty(),
                "staged writes are invisible until commit"
            );
            s.commit();
            assert_eq!(s.scan(b"k1"), vec![(b"k1".to_vec(), b(b"v1"))]);
        }
        {
            let s = RedbStore::open(&path).expect("open store");
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
        // A crash BEFORE commit loses the staged batch — the fsync window the kill-9 cells crash across.
        let path = temp_path();
        {
            let mut s = RedbStore::open(&path).expect("open store");
            s.put(b"k1", &b(b"v1"));
            // no commit; drop models crash-before-fsync
        }
        {
            let s = RedbStore::open(&path).expect("open store");
            assert!(s.is_empty(), "an uncommitted staged batch is NOT durable");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn last_write_wins_and_delete_nets_within_a_window() {
        let path = temp_path();
        {
            let mut s = RedbStore::open(&path).expect("open store");
            s.put(b"k", &b(b"first"));
            s.put(b"k", &b(b"second")); // last write wins
            s.put(b"d", &b(b"x"));
            s.delete(b"d"); // put-then-delete in one window nets to absent
            s.commit();
            assert_eq!(s.scan(b"k"), vec![(b"k".to_vec(), b(b"second"))]);
            assert!(s.scan(b"d").is_empty(), "put-then-delete nets to absent");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn scan_filters_by_prefix_ascending() {
        let path = temp_path();
        {
            let mut s = RedbStore::open(&path).expect("open store");
            s.put(&[1, 9], &b(b"a"));
            s.put(&[2, 0], &b(b"b"));
            s.put(&[2, 5], &b(b"c"));
            s.put(&[3, 0], &b(b"d"));
            s.commit();
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
            let mut s = RedbStore::open(&path).expect("open store");
            s.put(b"k", &b(b"v"));
            s.commit();
            s.delete(b"k");
            s.commit();
            assert!(s.scan(b"k").is_empty(), "a committed delete removes the key");
        }
        {
            let s = RedbStore::open(&path).expect("open store");
            assert!(s.is_empty(), "the delete is durable across reopen");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn commit_with_nothing_staged_is_a_noop() {
        let path = temp_path();
        {
            let mut s = RedbStore::open(&path).expect("open store");
            s.commit(); // idle tick: no staged → no-op, no panic
            assert!(s.is_empty());
        }
        let _ = std::fs::remove_file(&path);
    }
}
