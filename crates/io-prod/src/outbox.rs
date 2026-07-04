//! R-6d1 — the durable OUTBOX storage primitive: a per-node, redb-backed, retain-until-acked mirror of the
//! in-RAM `ReliableLaneSender.retry` window, keyed `(peer, class, incarnation, seq)`.
//!
//! WHY (D-6 #1, the AwaitAdopt source-crash residual): the redelivering transport (R-1..R-5) holds unacked
//! reliable frames in an IN-RAM retry buffer replayed on reconnect. If the SENDER PROCESS crashes, that RAM
//! buffer is lost, so a PRODUCER-LESS reliable one-shot (an `AwaitAdopt` `TransientBatch`, a band-exit
//! `GhostFlow::Despawn`) is silently lost with no re-driving producer. This module is the durable tier: a
//! write-through mirror of the RAM `retry` that survives the crash and is replayed on boot.
//!
//! THIS SLICE (R-6d1) is the STORAGE PRIMITIVE ONLY — the [`OutboxKey`] encoding + the [`OutboxSink`] seam +
//! the redb-backed [`NodeOutbox`] impl + its persistence tests. It is not yet wired to the [`ReliableLaneSender`]
//! FSM (the write-through/delete-through) nor to the bins (opening a per-node store) — those are R-6d2/R-6d3.
//! The value stored is OPAQUE already-framed bytes (HR1: never a decoded `World`); this module never inspects
//! the frame. Backed by the SAME `RedbStore` as the orchestrator (no new dependency — the transactional-outbox
//! pattern over the existing store).
//!
//! [`ReliableLaneSender`]: crate::mesh

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use vd_core::ids::NodeId;
use vd_sim::io::{Bytes, Durability, MsgClass, SendError, Store, Transport, bytes};

use crate::store::{DurabilityHandle, RedbStore, StoreError, StoreTuning};

/// R-6d3b: the ONE durable outbox sink per node, shared (cloned `Arc`) by every mesh per-peer writer task AND
/// by the boot-replay path. The `Mutex` is held ONLY across brief `await`-free/fsync-free store ops (a stage
/// plus a non-blocking submit in the writer's block A; a snapshot or gc in [`replay_outbox`]) — NEVER across
/// a durability WAIT (that parks/awaits on a cloned [`DurabilityHandle`] outside the lock), so no thread holds
/// the lock while waiting on the peer-writers. `None` (bins pre-R-6d3b-2b) = the inert byte-identical path.
pub type SharedOutbox = Arc<Mutex<Box<dyn OutboxSink + Send>>>;

/// The FIRST key byte of every outbox row — a private key-family tag so an outbox scan (`scan(&[OUTBOX_TAG])`)
/// never collides with another table user of the same redb file (the outbox may share a node's store with
/// future per-node families). `0x4F` = ASCII `'O'` (outbox).
pub(crate) const OUTBOX_TAG: u8 = 0x4F;

/// The fixed on-disk key width: `tag(1) + peer(8) + class(1) + incarnation(8) + seq(8)`.
pub(crate) const OUTBOX_KEY_LEN: usize = 1 + 8 + 1 + 8 + 8;

/// The 1-byte value-envelope version prefixing every stored frame (mirrors the DEFERRED.md #3 WAL-versioning
/// discipline): a format change bumps this and boot-replay decode quarantines a mismatched row rather than
/// mis-framing it. Bump on ANY change to the stored value layout.
pub(crate) const OUTBOX_FORMAT_VERSION: u8 = 1;

/// R-6d3a (MF-3): the outbox store's off-tick-writer hand-off channel depth. Unlike the orchestrator's
/// depth-2 store (which block-on-priors under its SINGLE sim-thread caller to bound crash-loss to ≤1 batch),
/// the outbox is fed by N CONCURRENT mesh peer-writer tasks sharing ONE store. Each submits at most one
/// un-durable batch before parking on the durable-before-send gate, so the peak in-flight is bounded by the
/// live peer count; a depth far above any realistic node's peer fan-out makes the writer channel FULL an
/// impossible-capacity tripwire (`submit_nonblocking` panics rather than block under the shared sink lock —
/// the MF-1 cross-peer-serialization hazard) instead of a load-reachable path. The outbox needs NO depth-1
/// crash-loss bound: an un-fsynced in-channel batch is an un-SENT frame (the gate withholds the wire send
/// until durable), re-emitted by the restarted source — so a deeper channel never widens loss, only the
/// harmless un-sent set. ONE home for the value (HR "no magic numbers").
pub const OUTBOX_WRITER_CHANNEL_DEPTH: usize = 256;

/// The STABLE durable byte for a `MsgClass` — deliberately NOT the implicit enum discriminant (a variant
/// REORDER must never silently re-map on-disk keys). Exhaustive: adding a `MsgClass` variant fails to compile
/// until it is given a stable byte here (a forced decision, not a silent default). Round-trips via
/// [`class_from_byte`]; both are pinned by an exhaustive golden test.
fn class_to_byte(class: MsgClass) -> u8 {
    match class {
        MsgClass::Control => 0,
        MsgClass::Saga => 1,
        MsgClass::Snapshot => 2,
        MsgClass::Input => 3,
        MsgClass::Membership => 4,
        MsgClass::GhostReliable => 5,
        MsgClass::GhostDelta => 6,
    }
}

/// The inverse of [`class_to_byte`]; `None` for an unknown byte (a corrupt / future-format key).
fn class_from_byte(b: u8) -> Option<MsgClass> {
    Some(match b {
        0 => MsgClass::Control,
        1 => MsgClass::Saga,
        2 => MsgClass::Snapshot,
        3 => MsgClass::Input,
        4 => MsgClass::Membership,
        5 => MsgClass::GhostReliable,
        6 => MsgClass::GhostDelta,
        _ => return None,
    })
}

/// Read a big-endian `u64` from an exactly-8-byte slice (the caller has length-checked the whole key).
fn be_u64(s: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(s);
    u64::from_be_bytes(a)
}

/// The durable identity of one retained reliable frame: the SAME `(peer, class, incarnation, seq)` tuple the
/// receiver dedups on, so a replay is idempotent end-to-end. Encoded BIG-ENDIAN so the byte order matches the
/// numeric order — a redb prefix scan therefore returns rows in `(peer, class, incarnation, seq)` ASCENDING
/// order, exactly the order a fresh lane replays them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutboxKey {
    pub(crate) peer: NodeId,
    pub(crate) class: MsgClass,
    pub(crate) incarnation: u64,
    pub(crate) seq: u64,
}

impl OutboxKey {
    /// Encode to the fixed 26-byte big-endian key.
    fn to_bytes(self) -> [u8; OUTBOX_KEY_LEN] {
        let mut k = [0u8; OUTBOX_KEY_LEN];
        k[0] = OUTBOX_TAG;
        k[1..9].copy_from_slice(&self.peer.0.to_be_bytes());
        k[9] = class_to_byte(self.class);
        k[10..18].copy_from_slice(&self.incarnation.to_be_bytes());
        k[18..26].copy_from_slice(&self.seq.to_be_bytes());
        k
    }

    /// Decode from a key slice; `None` for the wrong length, a foreign tag, or an unknown class byte (so a
    /// non-outbox / corrupt row is skipped, never mis-decoded).
    fn from_bytes(b: &[u8]) -> Option<OutboxKey> {
        if b.len() != OUTBOX_KEY_LEN || b[0] != OUTBOX_TAG {
            return None;
        }
        let class = class_from_byte(b[9])?;
        Some(OutboxKey {
            peer: NodeId(be_u64(&b[1..9])),
            class,
            incarnation: be_u64(&b[10..18]),
            seq: be_u64(&b[18..26]),
        })
    }
}

/// Prefix `framed` with the format-version envelope for storage.
fn encode_value(framed: &[u8]) -> Bytes {
    let mut v = Vec::with_capacity(1 + framed.len());
    v.push(OUTBOX_FORMAT_VERSION);
    v.extend_from_slice(framed);
    bytes(v)
}

/// Strip the format-version envelope; `None` on an unknown version or an empty value (a format mismatch /
/// corruption signal — the boot-replay path (R-6d3) decides quarantine-vs-fail-loud, this layer just reports
/// the row as undecodable).
fn decode_value(value: &[u8]) -> Option<Vec<u8>> {
    match value.split_first() {
        Some((&OUTBOX_FORMAT_VERSION, rest)) => Some(rest.to_vec()),
        _ => None,
    }
}

/// The durable retain-until-acked sink behind the reliable lane. A write-through mirror: `retain` on send,
/// `release` on ack-retire, `commit` to fsync the staged batch, `scan_all` to rehydrate on boot, `gc_below`
/// to sweep a previous incarnation's rows after a fresh-incarnation replay. Object-safe so the FSM (R-6d2)
/// can hold `Option<&mut dyn OutboxSink>` — ONE FSM, a durable backend behind ONE seam (HR3).
pub trait OutboxSink: Send {
    /// Stage a retained frame (opaque already-framed bytes) under its key. Durable only after [`commit`].
    ///
    /// [`commit`]: OutboxSink::commit
    fn retain(&mut self, key: &OutboxKey, framed: &[u8]);

    /// Stage the removal of a retired (acked) frame. Durable only after [`commit`].
    ///
    /// [`commit`]: OutboxSink::commit
    fn release(&mut self, key: &OutboxKey);

    /// Fsync the staged retains/releases (one durability barrier per flush — batched, not per-frame).
    fn commit(&mut self);

    /// Every currently-retained `(key, framed)` in ASCENDING key order (the boot replay order). Undecodable
    /// rows (foreign tag / bad version) are skipped.
    fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)>;

    /// Stage the removal of every retained row whose incarnation is strictly BELOW `incarnation` — the
    /// post-replay sweep of a prior process incarnation's window. Durable only after [`commit`].
    ///
    /// [`commit`]: OutboxSink::commit
    fn gc_below(&mut self, incarnation: u64);

    /// R-6d3a durable-before-send GATE (MF-1/MF-2/MF-3): submit the CURRENTLY-STAGED batch WITHOUT any
    /// blocking wait, returning `(seq, handle)` — the submitted batch seq and a cloned durability watermark
    /// the caller awaits (`handle.wait_durable_through(seq)`) OUTSIDE the shared sink lock, so the row is on
    /// disk BEFORE its frame hits the wire. Returns `None` if nothing was staged (an idle / release-only
    /// span — the caller must NOT then treat the send as durable-gated; on the durable path a `retain` always
    /// precedes ⇒ `Some`). Non-blocking under the lock is the whole point: `commit` block-on-priors and would
    /// serialize every other peer-writer sharing this sink. Waiting on `seq` alone suffices — durability is
    /// monotone (the writer drains in seq order), so it subsumes every prior batch.
    fn submit_barrier(&mut self) -> Option<(u64, DurabilityHandle)>;

    /// R-6d3b-2: a clone of the durability watermark handle — so [`replay_outbox`] can capture it under the
    /// brief snapshot lock and then FENCE (wait for the fresh re-mirrored rows to be submitted + durable)
    /// OUTSIDE the lock. A test/mock sink returns an always-durable handle (its rows are never really staged).
    fn durability(&self) -> DurabilityHandle;
}

/// The redb-backed [`OutboxSink`] — one per node, opened at boot. Holds the [`DurabilityHandle`] both to keep
/// the store's off-tick writer thread alive AND to make [`commit`](OutboxSink::commit) block until the staged
/// batch is DURABLE (the outbox is the durable-before-send gate — a retained frame must be on disk before its
/// QUIC send, so a crash in the send window replays it on boot; R-6d3 wires that ordering into the writer).
pub struct NodeOutbox {
    store: RedbStore,
    durability: DurabilityHandle,
}

impl NodeOutbox {
    /// Open (or create at genesis) the durable outbox at `path`, spawning the store's off-tick writer.
    ///
    /// R-6d3a (MF-3): the outbox forces its writer channel to [`OUTBOX_WRITER_CHANNEL_DEPTH`] regardless of
    /// the caller's tuning — the multi-producer non-blocking submit needs slack well above the peer count so
    /// `submit_nonblocking` never blocks under the shared sink lock. Test-only tuning fields (the SIGKILL
    /// pause hooks) pass through unchanged.
    pub fn open(path: impl AsRef<Path>, mut tuning: StoreTuning) -> Result<NodeOutbox, StoreError> {
        tuning.writer_channel_depth = OUTBOX_WRITER_CHANNEL_DEPTH;
        let (store, durability) = RedbStore::open(path, tuning)?;
        Ok(NodeOutbox { store, durability })
    }
}

impl OutboxSink for NodeOutbox {
    fn retain(&mut self, key: &OutboxKey, framed: &[u8]) {
        self.store.put(&key.to_bytes(), &encode_value(framed));
    }

    fn release(&mut self, key: &OutboxKey) {
        self.store.delete(&key.to_bytes());
    }

    fn commit(&mut self) {
        // Submit the staged batch (block-on-prior), THEN block until it is durable — so `scan_all` reads it
        // and the durable-before-send gate (R-6d3) holds. `commit` submits at most one batch, so waiting on
        // the current `last_submitted` waits for exactly this batch.
        self.store.commit();
        self.durability
            .wait_durable_through(self.store.last_submitted());
    }

    fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)> {
        let mut out = Vec::new();
        for (k, v) in self.store.scan(&[OUTBOX_TAG]) {
            if let Some(key) = OutboxKey::from_bytes(&k)
                && let Some(framed) = decode_value(&v)
            {
                out.push((key, framed));
            }
        }
        out
    }

    fn gc_below(&mut self, incarnation: u64) {
        let stale: Vec<Vec<u8>> = self
            .store
            .scan(&[OUTBOX_TAG])
            .into_iter()
            .filter_map(|(k, _)| OutboxKey::from_bytes(&k).map(|key| (k, key)))
            .filter(|(_, key)| key.incarnation < incarnation)
            .map(|(k, _)| k)
            .collect();
        for k in stale {
            self.store.delete(&k);
        }
    }

    fn submit_barrier(&mut self) -> Option<(u64, DurabilityHandle)> {
        // Non-blocking submit (no block-on-prior under the caller's shared lock — MF-1/MF-3); the seq is the
        // caller's gate target, awaited on the cloned handle OUTSIDE the lock. `None` ⇒ nothing staged (MF-2).
        self.store
            .submit_nonblocking()
            .map(|seq| (seq, self.durability.clone()))
    }

    fn durability(&self) -> DurabilityHandle {
        self.durability.clone()
    }
}

/// R-6d3b-2: a momentarily-FULL live lane during boot replay retries this many times before it is declared
/// wedged. Boot replay out-paces the async peer-writer drain on a small `outbound_capacity`; a bounded retry
/// absorbs that, and exhausting it is a real fault (the drain is stuck) ⇒ fail loud. ONE home (HR no-magic-#).
const REPLAY_SEND_MAX_RETRIES: u32 = 10_000;
/// The backoff between replay-send retries AND the count-fence poll — the bin is on its own thread pre-`build_app`,
/// so a `std::thread::sleep` here parks only that thread (never a tokio worker).
const REPLAY_POLL_BACKOFF: Duration = Duration::from_millis(1);
/// The total boot-replay fence deadline: if the fresh re-mirrored rows are not all SUBMITTED within this, a
/// peer-writer is stuck (never scheduled) ⇒ fail LOUD (leaving the prior window un-gc'd = SAFE), never hang.
const REPLAY_FENCE_DEADLINE: Duration = Duration::from_secs(30);

/// R-6d3b-2 boot-replay failure — a durable row is NEVER silently dropped on the recovery path.
#[derive(Debug, PartialEq, Eq)]
pub enum ReplayError {
    /// A retained row could not be decoded to a payload (a corrupt / version-mismatched value).
    Undecodable,
    /// A retained row's peer is not in the transport's route book — a permanent can't-route.
    Unroutable { peer: NodeId },
    /// A live lane stayed full past [`REPLAY_SEND_MAX_RETRIES`] — the drain is wedged.
    LaneStuck { peer: NodeId },
    /// The fresh re-mirrored rows did not all submit within [`REPLAY_FENCE_DEADLINE`] — a peer-writer is stuck.
    FenceTimeout { submitted: u64, expected: u64 },
}

/// Decode a stored (envelope already stripped by `scan_all`) framed `ReliableFrame` back to its PAYLOAD bytes.
/// Boot replay re-sends the PAYLOAD (the send path re-frames it at the fresh incarnation/epoch/seq), NEVER the
/// stored frame — re-sending the stored frame would double-frame + carry the stale incarnation.
fn decode_value_payload(framed: &[u8]) -> Option<Bytes> {
    vd_wire::framing::decode_frame::<crate::ReliableFrame>(framed)
        .ok()
        .map(|(rf, _)| bytes(rf.bytes))
}

/// Enqueue one replay send, tolerating a momentarily-full LIVE lane (bounded retry). A route-book miss is
/// caught by the caller's pre-send membership check, so a `QueueFull` HERE is definitionally a transient full
/// lane; exhausting the retry is a wedged drain ⇒ `LaneStuck` (loud). NEVER a silent `let _ =` drop.
fn send_durable_with_retry(
    transport: &mut dyn Transport,
    peer: NodeId,
    class: MsgClass,
    payload: Bytes,
) -> Result<(), ReplayError> {
    let mut pay = payload;
    for _ in 0..REPLAY_SEND_MAX_RETRIES {
        match transport.send_durable(peer, class, pay, Durability::Retained) {
            Ok(_) => return Ok(()),
            Err(SendError::QueueFull(returned)) => {
                std::thread::sleep(REPLAY_POLL_BACKOFF);
                pay = returned; // reuse the returned payload — no re-clone
            }
        }
    }
    Err(ReplayError::LaneStuck { peer })
}

/// R-6d3b-2 BOOT REPLAY (the ONE home): re-drive every retained outbox row through `transport.send_durable`
/// (so a producer-less one-shot lost to a source crash is redelivered), then GC the prior incarnation's
/// window. Called by shard/gateway `main` at boot BEFORE `build_app` consumes the transport; the peer-writer
/// tasks are already live (spawned by `spawn_mesh`), so the enqueued sends are drained + re-mirrored at the
/// FRESH `new_incarnation` (ascending scan order ⇒ the fresh lane assigns seq 0,1,2.. ⇒ `classify_reliable`
/// Reset-then-Accept, no gap). Returns the number of rows re-driven.
///
/// LOCK DISCIPLINE (deadlock-free, R-6d3b-2 correction): the shared `Mutex` is held ONLY for the brief snapshot
/// (scan + handle + base) and the brief atomic gc — NEVER across the sends or the fence. If replay held the
/// lock across the sends, a replayed `send_durable` would enqueue to a peer-writer whose block A re-`.lock()`s
/// the SAME mutex to re-mirror ⇒ it blocks its tokio worker, and the fence would wait forever for a durability
/// bump only that blocked writer can make (a certain deadlock). Here the peer-writers re-mirror FREELY.
///
/// FENCE (no premature-gc, R-6d3b-2 correction): the fence is COUNT-anchored, not high-water-anchored — it
/// waits `last_submitted >= base + N` (each replay send = exactly one submit; a previously-retained row
/// re-frames IDENTICALLY so it never sheds `Unframable` ⇒ N is exact) THEN `wait_durable_through(base + N)`.
/// Waiting on the high-water alone would let the fence pass with only k<N fresh rows submitted (the peer-
/// writers are async), and the subsequent `gc_below` would then sweep the OLD rows whose fresh copy never
/// landed = D-6 #1 loss. (The ONE way a fresh row could fail to submit is a `BufferFull` shed — a >retry_cap
/// single-(peer,class) backlog, unreachable at real peer counts; that is fail-SAFE: no submit ⇒ the fence
/// never reaches `base + N` ⇒ `FenceTimeout` ⇒ the `?` skips gc ⇒ the old rows survive for the next boot,
/// NO loss.)
///
/// # Errors
/// A row whose peer is not in `peers` ([`ReplayError::Unroutable`]); a live lane wedged full
/// ([`ReplayError::LaneStuck`]); an undecodable row ([`ReplayError::Undecodable`]); or a peer-writer that
/// never submits its fresh row within [`REPLAY_FENCE_DEADLINE`] ([`ReplayError::FenceTimeout`]). On any error
/// the gc is SKIPPED (the `?` bails first) so a not-yet-redelivered row's data is never swept.
pub fn replay_outbox(
    shared: &SharedOutbox,
    transport: &mut dyn Transport,
    peers: &BTreeMap<NodeId, SocketAddr>,
    new_incarnation: u64,
) -> Result<usize, ReplayError> {
    // (1) ONE brief lock: snapshot the retained rows + a DurabilityHandle clone + the base submit watermark.
    let (rows, durability, base) = {
        let g = shared.lock().unwrap_or_else(PoisonError::into_inner);
        let dh = g.durability();
        let base = dh.last_submitted();
        (g.scan_all(), dh, base)
    };
    if rows.is_empty() {
        return Ok(0); // genesis / already-drained: nothing to replay, nothing to gc
    }
    let n = rows.len() as u64;

    // (2) NO LOCK: decode payload + pre-send route-membership check + send each. The peer-writers acquire the
    // shared lock FREELY to re-mirror (no deadlock — this thread holds none). `?` bails before the fence/gc on
    // any un-routable / undecodable / wedged row, so its data is never swept.
    for (key, framed) in &rows {
        if !peers.contains_key(&key.peer) {
            return Err(ReplayError::Unroutable { peer: key.peer });
        }
        let payload = decode_value_payload(framed).ok_or(ReplayError::Undecodable)?;
        send_durable_with_retry(transport, key.peer, key.class, payload)?;
    }

    // (3) NO LOCK: the COUNT-anchored fence. Wait until all N fresh rows have SUBMITTED (each send = one +1 to
    // last_submitted; at boot pre-`build_app` the replay sends are the ONLY submits), bounded fail-loud so a
    // stuck-never-scheduled peer-writer refuses rather than hangs; THEN wait for durability through that seq
    // (the writer's own death backstop panics loud if it dies). Only after this is gc safe.
    // `checked_add`: a REAL `NodeOutbox` base grows from real submits (never near u64::MAX), but a test/mock
    // sink whose `durability()` is `already_durable()` returns `base = u64::MAX` — refuse LOUD rather than
    // wrap to a bogus target (which would pass the fence instantly ⇒ gc without real durability).
    let target = base.checked_add(n).expect(
        "replay_outbox: durability watermark saturated (already_durable / not a real NodeOutbox) — base + N \
         overflow; replay must run against a real store",
    );
    let start = Instant::now();
    while durability.last_submitted() < target {
        if start.elapsed() >= REPLAY_FENCE_DEADLINE {
            return Err(ReplayError::FenceTimeout {
                submitted: durability.last_submitted(),
                expected: target,
            });
        }
        std::thread::sleep(REPLAY_POLL_BACKOFF);
    }
    durability.wait_durable_through(target);

    // (4) ONE brief lock (atomic): sweep the strictly-lower prior incarnation window + fsync — STRICTLY after
    // every fresh row is durable (HIGH-3). Kept in ONE guard so it is one observable atomic gc step.
    {
        let mut g = shared.lock().unwrap_or_else(PoisonError::into_inner);
        g.gc_below(new_incarnation);
        g.commit();
    }
    Ok(rows.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_CLASSES: [MsgClass; 7] = [
        MsgClass::Control,
        MsgClass::Saga,
        MsgClass::Snapshot,
        MsgClass::Input,
        MsgClass::Membership,
        MsgClass::GhostReliable,
        MsgClass::GhostDelta,
    ];

    fn key(peer: u64, class: MsgClass, inc: u64, seq: u64) -> OutboxKey {
        OutboxKey {
            peer: NodeId(peer),
            class,
            incarnation: inc,
            seq,
        }
    }

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vd-outbox-{tag}-{}-{:p}.redb",
            std::process::id(),
            &tag
        ))
    }

    struct TempOutbox {
        path: std::path::PathBuf,
    }
    impl Drop for TempOutbox {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn class_byte_round_trips_for_every_variant() {
        // The golden pin: a variant reorder / addition must keep the durable bytes stable (or force a
        // deliberate remap). Exhaustive over ALL 7 classes, bytes 0..=6 distinct.
        let mut seen = std::collections::BTreeSet::new();
        for c in ALL_CLASSES {
            let b = class_to_byte(c);
            assert!(
                seen.insert(b),
                "class byte {b} is not unique across MsgClass"
            );
            assert_eq!(class_from_byte(b), Some(c), "round-trip {c:?}");
        }
        assert_eq!(seen.len(), ALL_CLASSES.len(), "every class mapped");
        assert_eq!(class_from_byte(7), None, "an unknown byte decodes to None");
        assert_eq!(class_from_byte(255), None);
    }

    #[test]
    fn key_round_trips_and_encodes_big_endian_ascending() {
        let k = key(0x0102_0304_0506_0708, MsgClass::Saga, 42, 7);
        let bytes = k.to_bytes();
        assert_eq!(bytes.len(), OUTBOX_KEY_LEN);
        assert_eq!(bytes[0], OUTBOX_TAG);
        assert_eq!(&bytes[1..9], &0x0102_0304_0506_0708u64.to_be_bytes());
        assert_eq!(bytes[9], class_to_byte(MsgClass::Saga));
        assert_eq!(OutboxKey::from_bytes(&bytes), Some(k));

        // Big-endian ⇒ lexicographic byte order == numeric (peer, class, incarnation, seq) order.
        assert!(key(1, MsgClass::Saga, 5, 0).to_bytes() < key(1, MsgClass::Saga, 5, 1).to_bytes());
        assert!(key(1, MsgClass::Saga, 5, 9).to_bytes() < key(1, MsgClass::Saga, 6, 0).to_bytes());
        assert!(
            key(1, MsgClass::Control, 9, 9).to_bytes() < key(2, MsgClass::Control, 0, 0).to_bytes()
        );
    }

    #[test]
    fn key_from_bytes_rejects_malformed() {
        assert_eq!(OutboxKey::from_bytes(&[]), None, "empty");
        assert_eq!(
            OutboxKey::from_bytes(&[0u8; OUTBOX_KEY_LEN - 1]),
            None,
            "short"
        );
        assert_eq!(
            OutboxKey::from_bytes(&[0u8; OUTBOX_KEY_LEN + 1]),
            None,
            "long"
        );
        let mut wrong_tag = key(1, MsgClass::Saga, 1, 1).to_bytes();
        wrong_tag[0] = 0x00;
        assert_eq!(OutboxKey::from_bytes(&wrong_tag), None, "foreign tag");
        let mut bad_class = key(1, MsgClass::Saga, 1, 1).to_bytes();
        bad_class[9] = 200;
        assert_eq!(
            OutboxKey::from_bytes(&bad_class),
            None,
            "unknown class byte"
        );
    }

    #[test]
    fn value_envelope_round_trips_and_rejects_bad_version() {
        let framed = b"opaque-reliable-frame-bytes";
        let v = encode_value(framed);
        assert_eq!(v[0], OUTBOX_FORMAT_VERSION);
        assert_eq!(decode_value(&v), Some(framed.to_vec()));
        assert_eq!(decode_value(&[]), None, "empty value");
        assert_eq!(
            decode_value(&[OUTBOX_FORMAT_VERSION + 1, 1, 2]),
            None,
            "wrong version"
        );
        // An empty payload at the right version is a valid (zero-length) frame.
        assert_eq!(decode_value(&[OUTBOX_FORMAT_VERSION]), Some(Vec::new()));
    }

    #[test]
    fn retain_scan_release_round_trip() {
        let path = temp_path("rt");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");

        let k1 = key(1, MsgClass::Saga, 10, 0);
        let k2 = key(1, MsgClass::Saga, 10, 1);
        ob.retain(&k1, b"frame-0");
        ob.retain(&k2, b"frame-1");
        ob.commit();

        let scanned = ob.scan_all();
        assert_eq!(
            scanned,
            vec![(k1, b"frame-0".to_vec()), (k2, b"frame-1".to_vec())],
            "both retained, ascending, payloads version-stripped"
        );

        ob.release(&k1);
        ob.commit();
        assert_eq!(
            ob.scan_all(),
            vec![(k2, b"frame-1".to_vec())],
            "released k1 is gone; k2 remains"
        );
    }

    #[test]
    fn submit_barrier_gives_monotone_durable_seqs_and_none_on_an_empty_span() {
        // R-6d3a: the durable-before-send gate primitive — a retain then `submit_barrier` hands back the STORE
        // batch seq (monotone, > 0) + a cloned handle; awaiting it makes the row durable (readable by
        // `scan_all`). Distinct submits get ASCENDING seqs (so a single wait on `seq` subsumes every prior,
        // MF-3). An empty span (nothing staged) returns `None` (MF-2). This is the non-blocking multi-producer
        // path — the channel is forced to `OUTBOX_WRITER_CHANNEL_DEPTH` by `open`, so it never blocks.
        let path = temp_path("barrier");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");

        let k1 = key(1, MsgClass::Saga, 7, 0);
        ob.retain(&k1, b"f0");
        let (s1, h1) = ob.submit_barrier().expect("a staged retain ⇒ Some");
        assert!(s1 > 0, "the store batch seq is non-zero (never aliased to genesis)");
        h1.wait_durable_through(s1);
        assert_eq!(
            ob.scan_all(),
            vec![(k1, b"f0".to_vec())],
            "the row is durable once the barrier seq is fsynced (durable-before-send holds)"
        );

        let k2 = key(2, MsgClass::GhostReliable, 7, 0);
        ob.retain(&k2, b"f1");
        let (s2, h2) = ob.submit_barrier().expect("a staged retain ⇒ Some");
        assert!(s2 > s1, "monotone ascending store seq across submits (waiting on s2 subsumes s1)");
        h2.wait_durable_through(s2);

        // An empty span (nothing staged) ⇒ None: block B must NOT treat it as durable-gated (MF-2).
        assert!(
            ob.submit_barrier().is_none(),
            "no stage ⇒ no barrier"
        );
    }

    #[test]
    fn decode_value_payload_recovers_the_payload_and_rejects_garbage() {
        // R-6d3b-2: a stored (envelope-stripped) framed ReliableFrame decodes back to its PAYLOAD (not the
        // frame) — replay re-sends the payload so the send path re-frames at the fresh incarnation/seq.
        let framed = vd_wire::framing::encode_frame(&crate::ReliableFrame {
            from: NodeId(1),
            class: MsgClass::Saga,
            incarnation: 5,
            epoch: 9,
            seq: 3,
            bytes: b"payload".to_vec(),
        })
        .expect("encode");
        assert_eq!(
            decode_value_payload(&framed).as_deref(),
            Some(&b"payload"[..]),
            "the payload is recovered (frame header stripped)"
        );
        assert_eq!(
            decode_value_payload(&[0u8; 2]),
            None,
            "an undecodable value ⇒ None (never a mis-decode)"
        );
    }

    /// A `Transport` that fails `send_durable` with `QueueFull` its first `fails_left` calls, then delivers —
    /// exercises `send_durable_with_retry`'s bounded retry + payload-reuse deterministically (no real QUIC).
    struct FlakyTransport {
        fails_left: u32,
        sent: Vec<(NodeId, MsgClass, Vec<u8>)>,
    }
    impl Transport for FlakyTransport {
        fn send_durable(
            &mut self,
            to: NodeId,
            class: MsgClass,
            bytes: Bytes,
            _durability: Durability,
        ) -> Result<vd_core::MsgId, SendError> {
            if self.fails_left > 0 {
                self.fails_left -= 1;
                return Err(SendError::QueueFull(bytes)); // returns the payload for the retry to reuse
            }
            self.sent.push((to, class, bytes.to_vec()));
            Ok(vd_core::MsgId(0))
        }
        fn drain_inbound(&mut self) -> Vec<vd_sim::io::Inbound> {
            Vec::new()
        }
        fn local_id(&self) -> NodeId {
            NodeId(1)
        }
    }

    #[test]
    fn send_durable_with_retry_retries_a_full_lane_then_succeeds() {
        // R-6d3b-2 finding B: a momentarily-full live lane is retried (NOT swallowed); the returned payload is
        // reused (no re-clone) and delivered once the lane drains.
        let mut t = FlakyTransport {
            fails_left: 3,
            sent: Vec::new(),
        };
        assert_eq!(
            send_durable_with_retry(&mut t, NodeId(2), MsgClass::Saga, bytes(vec![7])),
            Ok(())
        );
        assert_eq!(
            t.sent,
            vec![(NodeId(2), MsgClass::Saga, vec![7])],
            "delivered exactly once after 3 QueueFull retries, payload intact"
        );
    }

    /// A valid framed `ReliableFrame` value (as `scan_all` returns, envelope-stripped) carrying `payload`.
    fn framed_row(payload: &[u8]) -> Vec<u8> {
        vd_wire::framing::encode_frame(&crate::ReliableFrame {
            from: NodeId(1),
            class: MsgClass::Saga,
            incarnation: 1,
            epoch: u32::MAX,
            seq: 0,
            bytes: payload.to_vec(),
        })
        .expect("encode")
    }

    /// A dummy loopback addr for a `peers` book entry (never dialed — these tests bail before any send).
    fn dummy_addr() -> SocketAddr {
        "127.0.0.1:9".parse().expect("addr")
    }

    #[test]
    fn replay_outbox_bails_on_an_unroutable_row_without_gc() {
        // R-6d3b-2 finding B / boot-safety: a retained row whose peer is NOT in the route book is a permanent
        // can't-route ⇒ `Unroutable` LOUD, and the gc is SKIPPED (the `?` bails first) so the prior-incarnation
        // rows SURVIVE for the next boot — a durable row is never swept because it could not be re-driven.
        let path = temp_path("unroutable");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
        ob.retain(&key(9, MsgClass::Saga, 1, 0), &framed_row(b"x")); // peer 9
        ob.commit();
        let shared: SharedOutbox = Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>));
        let peers: BTreeMap<NodeId, SocketAddr> = [(NodeId(2), dummy_addr())].into(); // NO peer 9
        let mut t = FlakyTransport {
            fails_left: 0,
            sent: Vec::new(),
        };
        assert_eq!(
            replay_outbox(&shared, &mut t, &peers, 2),
            Err(ReplayError::Unroutable { peer: NodeId(9) })
        );
        assert!(t.sent.is_empty(), "bailed before any send");
        let remaining = shared
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .scan_all();
        assert_eq!(remaining.len(), 1, "the un-routable row SURVIVES — gc was skipped");
        assert_eq!(remaining[0].0.incarnation, 1, "the prior-incarnation row is intact");
    }

    #[test]
    fn replay_outbox_bails_on_an_undecodable_row_without_gc() {
        // Boot-safety: a retained row whose value is not a decodable frame ⇒ `Undecodable` LOUD, gc SKIPPED,
        // the row SURVIVES (never mis-decoded, never swept).
        let path = temp_path("undecodable");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
        ob.retain(&key(2, MsgClass::Saga, 1, 0), b"not-a-valid-frame"); // garbage value, peer 2 (routable)
        ob.commit();
        let shared: SharedOutbox = Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>));
        let peers: BTreeMap<NodeId, SocketAddr> = [(NodeId(2), dummy_addr())].into();
        let mut t = FlakyTransport {
            fails_left: 0,
            sent: Vec::new(),
        };
        assert_eq!(
            replay_outbox(&shared, &mut t, &peers, 2),
            Err(ReplayError::Undecodable)
        );
        assert!(t.sent.is_empty(), "bailed at decode, before any send");
        assert_eq!(
            shared
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .scan_all()
                .len(),
            1,
            "the undecodable row SURVIVES — gc was skipped"
        );
    }

    #[test]
    fn scan_all_is_ascending_across_peers_and_classes() {
        let path = temp_path("order");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
        // Insert out of order; scan must come back ascending by (peer, class, incarnation, seq).
        let keys = [
            key(2, MsgClass::Saga, 1, 0),
            key(1, MsgClass::GhostReliable, 1, 0),
            key(1, MsgClass::Saga, 2, 0),
            key(1, MsgClass::Saga, 1, 5),
            key(1, MsgClass::Saga, 1, 0),
        ];
        for (i, k) in keys.iter().enumerate() {
            ob.retain(k, format!("f{i}").as_bytes());
        }
        ob.commit();
        let got: Vec<OutboxKey> = ob.scan_all().into_iter().map(|(k, _)| k).collect();
        let mut want = keys.to_vec();
        want.sort_by_key(|k| k.to_bytes());
        assert_eq!(got, want, "scan_all is key-ascending");
    }

    #[test]
    fn gc_below_sweeps_only_strictly_lower_incarnations() {
        let path = temp_path("gc");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
        ob.retain(&key(1, MsgClass::Saga, 5, 0), b"old");
        ob.retain(&key(1, MsgClass::Saga, 5, 1), b"old");
        ob.retain(&key(1, MsgClass::Saga, 6, 0), b"fresh");
        ob.commit();

        ob.gc_below(6);
        ob.commit();
        let got: Vec<OutboxKey> = ob.scan_all().into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            got,
            vec![key(1, MsgClass::Saga, 6, 0)],
            "incarnation 5 swept, incarnation 6 (== the floor, not <) kept"
        );
    }

    #[test]
    fn retained_frames_survive_a_reopen() {
        // The load-bearing property: a committed retain is durable across a process restart (drop + reopen
        // the SAME file). RedbStore::Drop graceful-joins the writer, so the committed batch is on disk.
        let path = temp_path("persist");
        let _g = TempOutbox { path: path.clone() };
        let k = key(7, MsgClass::GhostReliable, 3, 0);
        {
            let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
            ob.retain(&k, b"despawn-envelope");
            ob.commit();
        } // dropped ⇒ writer flushed + joined
        let ob2 = NodeOutbox::open(&path, StoreTuning::default()).expect("reopen");
        assert_eq!(
            ob2.scan_all(),
            vec![(k, b"despawn-envelope".to_vec())],
            "the retained frame survived the reopen"
        );
    }
}
