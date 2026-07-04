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
use vd_sim::io::{Bytes, Durability, MsgClass, SendError, Store, bytes};

use crate::mesh::ReplayTransport;
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
    /// WHOLE prior-window sweep. RETAINED for a future admin/whole-window use; NOT used by [`replay_outbox`]
    /// (which sweeps only the re-driven keys — see [`gc_replayed`]). Durable only after [`commit`].
    ///
    /// [`commit`]: OutboxSink::commit
    /// [`gc_replayed`]: OutboxSink::gc_replayed
    fn gc_below(&mut self, incarnation: u64);

    /// R-6d3b-2b (the F1 no-loss fix): stage the removal of EXACTLY these keys — the rows RE-DRIVEN this boot
    /// (now durable at the fresh incarnation, so the prior-incarnation copy is redundant). A row NOT in this
    /// set — a QUARANTINED roster-gone/undecodable row, or a version-mismatched row `scan_all` filtered — is
    /// RETAINED (never swept because it could NOT be re-driven; the D-6 #1 no-loss invariant). This REPLACES
    /// the blunt `gc_below(new_incarnation)`, which could not tell a re-driven row from a skipped one and would
    /// sweep a recoverable roster-gone row. Durable only after [`commit`].
    ///
    /// [`commit`]: OutboxSink::commit
    fn gc_replayed(&mut self, replayed_keys: &[OutboxKey]);

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

    fn gc_replayed(&mut self, replayed_keys: &[OutboxKey]) {
        // Delete EXACTLY the re-driven keys — a straight-line loop over the caller's set (the caller has
        // already excluded quarantined/undecodable rows). Durable at the caller's `commit`.
        for key in replayed_keys {
            self.store.delete(&key.to_bytes());
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

/// R-6d3b-2 boot-replay REFUSE-TO-BOOT failure — a transient/infra pathology that must fail LOUD (gc is
/// SKIPPED, so every retained row survives for the next boot). Distinct from a QUARANTINE (a poison/roster-gone
/// row is RETAINED + loud-counted and the boot PROCEEDS — see [`ReplayCounts`]); a durable row is NEVER
/// silently dropped on the recovery path.
#[derive(Debug, PartialEq, Eq)]
pub enum ReplayError {
    /// A live lane stayed full past [`REPLAY_SEND_MAX_RETRIES`] — the drain is wedged.
    LaneStuck { peer: NodeId },
    /// The peer's send lane is DEAD (its `peer_writer` task exited ⇒ the mpsc is closed) while the peer is
    /// still in the roster — an infra pathology (R-6d3b-2b RC-2b), fast-failed rather than a ~10s corpse-spin.
    LaneDead { peer: NodeId },
    /// The fresh re-mirrored rows did not all submit within [`REPLAY_FENCE_DEADLINE`] — a peer-writer is stuck.
    FenceTimeout { submitted: u64, expected: u64 },
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayError::LaneStuck { peer } => {
                write!(f, "boot replay: the lane to peer {} stayed full past the retry cap (drain wedged)", peer.0)
            }
            ReplayError::LaneDead { peer } => {
                write!(f, "boot replay: the lane to peer {} is dead (its writer task exited)", peer.0)
            }
            ReplayError::FenceTimeout { submitted, expected } => write!(
                f,
                "boot replay: the durability fence timed out ({submitted}/{expected} fresh rows submitted) — a peer-writer is stuck"
            ),
        }
    }
}
impl std::error::Error for ReplayError {}

/// R-6d3b-2b: the outcome of a boot replay — every retained row is accounted (RC-2a: no silent loss).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ReplayCounts {
    /// Rows re-sent (block A durable, fence passed) then gc'd (the redundant prior-incarnation copy).
    pub replayed: usize,
    /// Rows QUARANTINED — RETAINED on disk (NOT swept), loud-counted: a roster-gone peer (recoverable once
    /// the book is corrected) or an undecodable frame (on-disk corruption, preserved for admin forensics).
    pub quarantined: usize,
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
    transport: &mut dyn ReplayTransport,
    peer: NodeId,
    class: MsgClass,
    payload: Bytes,
) -> Result<(), ReplayError> {
    let mut pay = payload;
    for _ in 0..REPLAY_SEND_MAX_RETRIES {
        match transport.send_durable(peer, class, pay, Durability::Retained) {
            Ok(_) => return Ok(()),
            Err(SendError::QueueFull(returned)) => {
                // RC-2b: distinguish a DEAD lane (the peer_writer task exited ⇒ the mpsc is closed) from a
                // transiently-full one. A dead lane can NEVER drain ⇒ fail FAST (LaneDead) rather than spin
                // ~REPLAY_SEND_MAX_RETRIES×backoff (~10s) on a corpse before LaneStuck. `send_durable`'s
                // frozen-seam contract is unchanged — the disambiguation is an out-of-band `lane_alive` probe.
                if !transport.lane_alive(peer) {
                    return Err(ReplayError::LaneDead { peer });
                }
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
/// DISPOSITION (RC-2a, R-6d3b-2b): a poison/roster-gone row is QUARANTINED — RETAINED on disk + loud-counted,
/// the boot PROCEEDS (a single bad row never wedges the node). ONLY a transient infra pathology (a wedged or
/// dead lane, a stuck peer-writer) refuses-to-boot LOUD (gc SKIPPED ⇒ every row survives). gc sweeps ONLY the
/// re-driven keys ([`OutboxSink::gc_replayed`]) — a NOT-re-driven row (quarantined, or version-mismatch-filtered
/// by `scan_all`) is RETAINED (the D-6 #1 no-loss invariant; a durable row is never swept because it could not
/// be re-driven). The fence counts the RE-DRIVEN rows only, not `rows.len()` — a quarantined row never submits.
///
/// # Errors
/// [`ReplayError::LaneStuck`] (a live lane wedged full past the retry cap), [`ReplayError::LaneDead`] (a peer's
/// writer task died while still in the roster), or [`ReplayError::FenceTimeout`] (a peer-writer never submitted
/// within [`REPLAY_FENCE_DEADLINE`]). On any error the gc is SKIPPED (the `?` bails first). A roster-gone peer
/// or an undecodable frame is NOT an error — it is quarantined (see [`ReplayCounts::quarantined`]).
pub fn replay_outbox(
    shared: &SharedOutbox,
    transport: &mut dyn ReplayTransport,
    peers: &BTreeMap<NodeId, SocketAddr>,
) -> Result<ReplayCounts, ReplayError> {
    // (1) ONE brief lock: snapshot the retained rows + a DurabilityHandle clone + the base submit watermark.
    let (rows, durability, base) = {
        let g = shared.lock().unwrap_or_else(PoisonError::into_inner);
        let dh = g.durability();
        let base = dh.last_submitted();
        (g.scan_all(), dh, base)
    };
    if rows.is_empty() {
        return Ok(ReplayCounts::default()); // genesis / already-drained: nothing to replay, nothing to gc
    }

    // (2) NO LOCK: decode payload + pre-send route-membership check + send each. The peer-writers acquire the
    // shared lock FREELY to re-mirror (no deadlock — this thread holds none). A roster-gone peer or an
    // undecodable row is QUARANTINED (RETAINED + loud-counted, continue) — the boot PROCEEDS; only a wedged/
    // dead lane `?`-bails before the fence/gc (so a not-yet-redelivered row's data is never swept). The exact
    // re-driven keys are collected for `gc_replayed` — a quarantined/skipped row is NEVER in that set.
    let mut counts = ReplayCounts::default();
    let mut replayed_keys: Vec<OutboxKey> = Vec::new();
    for (key, framed) in &rows {
        if !peers.contains_key(&key.peer) {
            tracing::warn!(
                peer = key.peer.0,
                class = ?key.class,
                incarnation = key.incarnation,
                seq = key.seq,
                "boot replay QUARANTINE: a retained row's peer is no longer in the route book (roster diff) — \
                 RETAINED for the next boot (recoverable once the roster is corrected); the boot PROCEEDS"
            );
            counts.quarantined += 1;
            continue;
        }
        let Some(payload) = decode_value_payload(framed) else {
            tracing::warn!(
                peer = key.peer.0,
                class = ?key.class,
                incarnation = key.incarnation,
                seq = key.seq,
                "boot replay QUARANTINE: a retained row is undecodable (on-disk frame corruption) — RETAINED \
                 (unrecoverable but preserved for admin forensics); the boot PROCEEDS"
            );
            counts.quarantined += 1;
            continue;
        };
        send_durable_with_retry(transport, key.peer, key.class, payload)?; // LaneStuck/LaneDead ?-bail (loud)
        replayed_keys.push(*key);
        counts.replayed += 1;
    }

    // Nothing re-driven ⇒ nothing to gc; every row was quarantined + RETAINED. No sweep at all.
    if counts.replayed == 0 {
        return Ok(counts);
    }

    // (3) NO LOCK: the COUNT-anchored fence over the RE-DRIVEN rows ONLY (a quarantined row never submits, so
    // fencing on `rows.len()` would `FenceTimeout` on any boot that quarantined a row). Wait until all
    // `replayed` fresh rows have SUBMITTED (each send = one +1 to last_submitted; at boot pre-`build_app` the
    // replay sends are the ONLY submits), bounded fail-loud so a stuck peer-writer refuses rather than hangs;
    // THEN wait for durability through that seq (the writer's own death backstop panics loud if it dies).
    // `checked_add`: a REAL `NodeOutbox` base grows from real submits (never near u64::MAX); a test/mock sink
    // whose `durability()` is `already_durable()` returns `base = u64::MAX` — refuse LOUD rather than wrap.
    let target = base.checked_add(counts.replayed as u64).expect(
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

    // (4) ONE brief lock (atomic): sweep ONLY the re-driven keys + fsync — STRICTLY after every fresh row is
    // durable (HIGH-3). A quarantined/skipped row is NOT in `replayed_keys` ⇒ RETAINED (F1 no-loss). Kept in
    // ONE guard so it is one observable atomic gc step.
    {
        let mut g = shared.lock().unwrap_or_else(PoisonError::into_inner);
        g.gc_replayed(&replayed_keys);
        g.commit();
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_sim::io::Transport; // FlakyTransport impls Transport (+ ReplayTransport via super::*)

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

    /// A `ReplayTransport` that fails `send_durable` with `QueueFull` its first `fails_left` calls, then
    /// delivers — exercises `send_durable_with_retry`'s bounded retry + payload-reuse deterministically (no
    /// real QUIC). `lane_alive` is the RC-2b probe: `false` makes the first `QueueFull` fast-fail `LaneDead`.
    struct FlakyTransport {
        fails_left: u32,
        lane_alive: bool,
        sent: Vec<(NodeId, MsgClass, Vec<u8>)>,
    }
    impl FlakyTransport {
        fn new(fails_left: u32) -> Self {
            FlakyTransport {
                fails_left,
                lane_alive: true, // the common case: a live lane that is momentarily full
                sent: Vec::new(),
            }
        }
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
    impl ReplayTransport for FlakyTransport {
        fn lane_alive(&self, _peer: NodeId) -> bool {
            self.lane_alive
        }
    }

    #[test]
    fn send_durable_with_retry_retries_a_full_lane_then_succeeds() {
        // R-6d3b-2 finding B: a momentarily-full LIVE lane is retried (NOT swallowed); the returned payload is
        // reused (no re-clone) and delivered once the lane drains.
        let mut t = FlakyTransport::new(3); // lane_alive = true
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

    #[test]
    fn send_durable_with_retry_fast_fails_on_a_dead_lane() {
        // RC-2b: a DEAD lane (peer_writer gone ⇒ lane_alive=false) fast-fails LaneDead on the FIRST QueueFull
        // — no ~10s corpse-spin. Wall-clock is a single poll; assert it is far below the retry-cap duration.
        let mut t = FlakyTransport::new(u32::MAX);
        t.lane_alive = false;
        let start = Instant::now();
        assert_eq!(
            send_durable_with_retry(&mut t, NodeId(2), MsgClass::Saga, bytes(vec![7])),
            Err(ReplayError::LaneDead { peer: NodeId(2) })
        );
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "a dead lane fast-fails, not a corpse-spin"
        );
    }

    /// A dummy loopback addr for a `peers` book entry (never dialed — the mock quarantine tests never send).
    fn dummy_addr() -> SocketAddr {
        "127.0.0.1:9".parse().expect("addr")
    }

    #[test]
    fn replay_outbox_quarantines_an_unroutable_row_retains_it_and_proceeds() {
        // RC-2a: a retained row whose peer is no longer in the route book (roster diff) is QUARANTINED —
        // RETAINED for the next boot (recoverable once the roster is fixed) + counted; the boot PROCEEDS (no
        // wedge). With only this row, `replayed == 0` ⇒ no fence, no gc ⇒ the row SURVIVES on disk.
        let path = temp_path("unroutable");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
        ob.retain(&key(9, MsgClass::Saga, 1, 0), b"any-value"); // peer 9 (not in peers)
        ob.commit();
        let shared: SharedOutbox = Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>));
        let peers: BTreeMap<NodeId, SocketAddr> = [(NodeId(2), dummy_addr())].into(); // NO peer 9
        let mut t = FlakyTransport::new(0);
        assert_eq!(
            replay_outbox(&shared, &mut t, &peers),
            Ok(ReplayCounts {
                replayed: 0,
                quarantined: 1
            })
        );
        assert!(t.sent.is_empty(), "quarantined before any send");
        let remaining = shared
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .scan_all();
        assert_eq!(remaining.len(), 1, "the roster-gone row SURVIVES — never swept");
        assert_eq!(remaining[0].0.incarnation, 1, "the prior-incarnation row is intact");
    }

    #[test]
    fn replay_outbox_quarantines_an_undecodable_row_and_retains_it() {
        // RC-2a: a routable row whose value is not a decodable frame (on-disk corruption) is QUARANTINED —
        // RETAINED (unrecoverable but preserved for forensics), counted; the boot PROCEEDS; the row is never
        // mis-decoded nor swept.
        let path = temp_path("undecodable");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
        ob.retain(&key(2, MsgClass::Saga, 1, 0), b"not-a-valid-frame"); // garbage value, peer 2 (routable)
        ob.commit();
        let shared: SharedOutbox = Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>));
        let peers: BTreeMap<NodeId, SocketAddr> = [(NodeId(2), dummy_addr())].into();
        let mut t = FlakyTransport::new(0);
        assert_eq!(
            replay_outbox(&shared, &mut t, &peers),
            Ok(ReplayCounts {
                replayed: 0,
                quarantined: 1
            })
        );
        assert!(t.sent.is_empty(), "quarantined at decode, before any send");
        assert_eq!(
            shared
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .scan_all()
                .len(),
            1,
            "the undecodable row SURVIVES — never swept"
        );
    }

    /// A valid framed `ReliableFrame` value (as `scan_all` returns it, envelope-stripped) carrying `payload`.
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

    #[test]
    fn replay_outbox_refuses_to_boot_on_a_dead_lane_without_gc() {
        // RC-2a/RC-2b: a routable+decodable row whose lane is DEAD ⇒ `LaneDead` (refuse-to-boot, loud); the
        // `?` bails BEFORE the fence/gc, so the row SURVIVES for the next boot (never swept). Fast (no spin).
        let path = temp_path("deadlane");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
        ob.retain(&key(2, MsgClass::Saga, 1, 0), &framed_row(b"x")); // routable + decodable
        ob.commit();
        let shared: SharedOutbox = Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>));
        let peers: BTreeMap<NodeId, SocketAddr> = [(NodeId(2), dummy_addr())].into();
        let mut t = FlakyTransport::new(u32::MAX);
        t.lane_alive = false; // the lane is dead ⇒ fast-fail LaneDead on the first QueueFull
        assert_eq!(
            replay_outbox(&shared, &mut t, &peers),
            Err(ReplayError::LaneDead { peer: NodeId(2) })
        );
        assert_eq!(
            shared
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .scan_all()
                .len(),
            1,
            "refuse-to-boot skips gc ⇒ the row SURVIVES"
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
