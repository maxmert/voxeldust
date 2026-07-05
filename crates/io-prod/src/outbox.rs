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
// `Ord`/`Hash` derived (R-6d4-A): the natural tuple order (peer, class, incarnation, seq) MATCHES the
// big-endian `to_bytes` byte order (see `to_bytes`/`from_bytes` below), so a `BTreeMap<OutboxKey, _>` scans
// in the SAME ascending order as the redb range — letting the replay-proptest model key on the struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

    /// TEST-ONLY (R-6d4-D; feature `store-test-hooks`, ABSENT from release): plant + DURABLY write ONE retained
    /// `ReliableFrame` outbox row, so the process-tier SIGKILL-restart proof can seed a durable-and-unacked row
    /// before the kill (the boot-2 replay then re-drives it). Builds+frames INTERNALLY (keeps `ReliableFrame`
    /// pub(crate) — no bin touches it) and `commit()`s, which BLOCKS until fsynced (outbox.rs `commit`), so the
    /// row is on disk on return. `from` is carried for honesty (the stored frame's `from` is inert — replay
    /// re-frames from the decoded PAYLOAD only). Returns the durable batch seq (the caller asserts
    /// `durability().is_durable_through(seq)` as a fail-loud belt-and-suspenders).
    #[cfg(feature = "store-test-hooks")]
    #[must_use]
    pub fn seed_reliable_row(
        &mut self,
        from: NodeId,
        peer: NodeId,
        class: MsgClass,
        incarnation: u64,
        seq: u64,
        payload: &[u8],
    ) -> u64 {
        let framed = frame_reliable(from, class, incarnation, seq, payload);
        self.retain(&OutboxKey { peer, class, incarnation, seq }, &framed);
        self.commit(); // submit + wait_durable_through ⇒ durable-on-return (no pause hook)
        self.durability.last_submitted()
    }
}

/// R-6d4-D/A: the ONE home that frames a `ReliableFrame` for a durable outbox row — used by the process-tier
/// seed seam ([`NodeOutbox::seed_reliable_row`]) and the in-process proptest (`framed_row` delegates). Frames
/// at `epoch = u32::MAX` (the write-path re-stamp): replay re-frames from the decoded PAYLOAD, so the stored
/// `from`/`epoch` are inert — this is purely the on-disk-row builder. Gated to the two test surfaces.
#[cfg(any(test, feature = "store-test-hooks"))]
fn frame_reliable(from: NodeId, class: MsgClass, incarnation: u64, seq: u64, payload: &[u8]) -> Vec<u8> {
    vd_wire::framing::encode_frame(&crate::ReliableFrame {
        from,
        class,
        incarnation,
        epoch: u32::MAX,
        seq,
        bytes: payload.to_vec(),
    })
    .expect("encode reliable frame")
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

/// R-6d4-C: the replay retry/deadline bounds, made INJECTABLE so the `LaneStuck`/`FenceTimeout` error arms
/// fire FAST in unit tests (the release values are ~10s retry-cap / 30s fence — a real wedged drain / stuck
/// peer-writer, far too slow for a unit test). [`ReplayLimits::release`] IS the module consts, so the prod
/// path is byte-identical (`replay_outbox` delegates with it); this is a pure test-speed seam, NOT a prod
/// knob (no env/config surface). The BRANCH structure is unchanged — only the loop bound + deadline differ —
/// so Tier coverage of the arms is preserved.
#[derive(Clone, Copy)]
struct ReplayLimits {
    max_send_retries: u32,
    poll_backoff: Duration,
    fence_deadline: Duration,
}
impl ReplayLimits {
    /// The production bounds (the module consts). `replay_outbox` uses exactly this — no behavior change.
    const fn release() -> Self {
        ReplayLimits {
            max_send_retries: REPLAY_SEND_MAX_RETRIES,
            poll_backoff: REPLAY_POLL_BACKOFF,
            fence_deadline: REPLAY_FENCE_DEADLINE,
        }
    }
}

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
    limits: ReplayLimits,
) -> Result<(), ReplayError> {
    let mut pay = payload;
    for _ in 0..limits.max_send_retries {
        match transport.send_durable(peer, class, pay, Durability::Retained) {
            Ok(_) => return Ok(()),
            Err(SendError::QueueFull(returned)) => {
                // RC-2b: distinguish a DEAD lane (the peer_writer task exited ⇒ the mpsc is closed) from a
                // transiently-full one. A dead lane can NEVER drain ⇒ fail FAST (LaneDead) rather than spin
                // ~max_send_retries×backoff (~10s) on a corpse before LaneStuck. `send_durable`'s
                // frozen-seam contract is unchanged — the disambiguation is an out-of-band `lane_alive` probe.
                if !transport.lane_alive(peer) {
                    return Err(ReplayError::LaneDead { peer });
                }
                std::thread::sleep(limits.poll_backoff);
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
    // The prod entry: the release retry/deadline bounds (byte-identical to the pre-R-6d4-C body). The
    // `_with_limits` seam only shortens those bounds so the LaneStuck/FenceTimeout arms are unit-testable fast.
    replay_outbox_with_limits(shared, transport, peers, ReplayLimits::release())
}

/// R-6d4-C: [`replay_outbox`] with injectable [`ReplayLimits`] — the ONE body; the public wrapper passes the
/// release bounds. Tests pass a tiny retry-cap + short fence-deadline so the wedged-lane / stuck-writer arms
/// fire in milliseconds instead of ~10s / 30s.
fn replay_outbox_with_limits(
    shared: &SharedOutbox,
    transport: &mut dyn ReplayTransport,
    peers: &BTreeMap<NodeId, SocketAddr>,
    limits: ReplayLimits,
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
        send_durable_with_retry(transport, key.peer, key.class, payload, limits)?; // LaneStuck/LaneDead ?-bail
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
        if start.elapsed() >= limits.fence_deadline {
            return Err(ReplayError::FenceTimeout {
                submitted: durability.last_submitted(),
                expected: target,
            });
        }
        std::thread::sleep(limits.poll_backoff);
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
            send_durable_with_retry(
                &mut t,
                NodeId(2),
                MsgClass::Saga,
                bytes(vec![7]),
                ReplayLimits::release()
            ),
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
            send_durable_with_retry(
                &mut t,
                NodeId(2),
                MsgClass::Saga,
                bytes(vec![7]),
                ReplayLimits::release()
            ),
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
        // Delegate to the ONE frame-encode home (R-6d4-D DRY) so the test row + the process-tier seed row
        // are byte-identical builders.
        super::frame_reliable(NodeId(1), MsgClass::Saga, 1, 0, payload)
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

    #[cfg(feature = "store-test-hooks")]
    #[test]
    fn seed_reliable_row_writes_one_durable_scannable_row() {
        // R-6d4-D: the process-tier seed seam (vd-outbox-testnode calls it) plants ONE durable, decodable,
        // scannable row and returns durable-on-commit. Exercised in-crate so `coverage-io-prod-hooks`
        // instruments the seam + `frame_reliable` (the process-tier test that USES it lives in vd-bins, which
        // io-prod's own coverage gate never compiles — "REAL not theater": cover the seam where it is defined).
        let path = temp_path("seed");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
        let seq = ob.seed_reliable_row(NodeId(1), NodeId(2), MsgClass::Saga, 5, 0, b"hello");
        assert!(ob.durability().is_durable_through(seq), "the seeded row is durable-on-return");
        let rows = ob.scan_all();
        assert_eq!(rows.len(), 1, "exactly the one seeded row");
        assert_eq!(rows[0].0, key(2, MsgClass::Saga, 5, 0), "under its (peer,class,incarnation,seq) key");
        assert_eq!(
            decode_value_payload(&rows[0].1).as_deref(),
            Some(&b"hello"[..]),
            "the framed row decodes back to the seeded payload (replay re-frames from THIS payload)"
        );
    }

    // ---- R-6d4-C: the replay_outbox ERROR ARMS (F-C, the R-6d3b-2b review left them uncovered) ----------

    /// A mock [`OutboxSink`] with a real committed `rows` set (scanned + swept by `gc_replayed`) and a
    /// TEST-CONTROLLED `durability` handle. C1 uses `already_durable()` (base = `u64::MAX`) to drive the
    /// count-fence overflow arm; B2 uses a `controllable()` handle so the durability watermark parks the
    /// replay fence mid-flight (the no-premature-gc window). `scan_all` returns the committed rows so replay
    /// reaches the send loop; `gc_replayed` REALLY removes the swept keys (so B2's gc-not-run assert is honest).
    struct MockOutboxSink {
        rows: Vec<(OutboxKey, Vec<u8>)>,
        durability: DurabilityHandle,
    }
    impl OutboxSink for MockOutboxSink {
        fn retain(&mut self, _key: &OutboxKey, _framed: &[u8]) {}
        fn release(&mut self, _key: &OutboxKey) {}
        fn commit(&mut self) {}
        fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)> {
            self.rows.clone()
        }
        fn gc_below(&mut self, _incarnation: u64) {}
        fn gc_replayed(&mut self, replayed_keys: &[OutboxKey]) {
            self.rows.retain(|(k, _)| !replayed_keys.contains(k));
        }
        fn submit_barrier(&mut self) -> Option<(u64, DurabilityHandle)> {
            Some((1, self.durability.clone()))
        }
        fn durability(&self) -> DurabilityHandle {
            self.durability.clone()
        }
    }

    #[test]
    #[should_panic(expected = "saturated")]
    fn replay_outbox_overflow_base_plus_n_panics_loud() {
        // The count-fence `base.checked_add(counts.replayed).expect(...)` (outbox.rs) refuses LOUD rather than
        // wrap when `base == u64::MAX` — only reachable against a mock/`already_durable` sink, never a real
        // store. One routable+decodable row is re-driven (replayed = 1) ⇒ `u64::MAX + 1` overflows ⇒ panic.
        let mock = MockOutboxSink {
            rows: vec![(key(2, MsgClass::Saga, 1, 0), framed_row(b"x"))],
            durability: DurabilityHandle::already_durable(), // last_submitted() == u64::MAX
        };
        let shared: SharedOutbox = Arc::new(Mutex::new(Box::new(mock) as Box<dyn OutboxSink + Send>));
        let peers: BTreeMap<NodeId, SocketAddr> = [(NodeId(2), dummy_addr())].into();
        let mut t = FlakyTransport::new(0); // sends OK ⇒ replayed = 1 ⇒ the fence computes base + 1
        let _ = replay_outbox(&shared, &mut t, &peers); // panics inside the expect (caught by should_panic)
    }

    #[test]
    fn replay_error_display_and_error_impls() {
        // Every ReplayError variant Displays a distinct, actionable message + coerces to &dyn std::error::Error
        // (so the bin `?`-propagation at boot has a real Error). Exact strings (no matches! false arm).
        let stuck = ReplayError::LaneStuck { peer: NodeId(7) };
        let dead = ReplayError::LaneDead { peer: NodeId(7) };
        let timeout = ReplayError::FenceTimeout {
            submitted: 3,
            expected: 5,
        };
        assert_eq!(
            stuck.to_string(),
            "boot replay: the lane to peer 7 stayed full past the retry cap (drain wedged)"
        );
        assert_eq!(
            dead.to_string(),
            "boot replay: the lane to peer 7 is dead (its writer task exited)"
        );
        assert_eq!(
            timeout.to_string(),
            "boot replay: the durability fence timed out (3/5 fresh rows submitted) — a peer-writer is stuck"
        );
        // Coerce each to the trait object the bin boot path relies on.
        for e in [stuck, dead, timeout] {
            let dyn_err: &dyn std::error::Error = &e;
            assert!(!dyn_err.to_string().is_empty(), "the Error impl renders");
        }
    }

    /// Tiny, fast replay bounds so the wedged-lane / stuck-writer arms fire in milliseconds (the release
    /// bounds are ~10s / 30s). ONLY the loop cap + deadline differ; the branch structure is identical.
    fn fast_limits() -> ReplayLimits {
        ReplayLimits {
            max_send_retries: 5,
            poll_backoff: Duration::from_millis(1),
            fence_deadline: Duration::from_millis(50),
        }
    }

    #[test]
    fn replay_outbox_lane_stuck_refuses_boot_without_gc() {
        // A LIVE lane (lane_alive = true) that stays FULL past the retry cap ⇒ LaneStuck (refuse-to-boot,
        // loud). The `?` bails BEFORE the fence/gc ⇒ the row SURVIVES for the next boot (no silent sweep).
        let path = temp_path("lanestuck");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
        ob.retain(&key(2, MsgClass::Saga, 1, 0), &framed_row(b"x")); // routable + decodable
        ob.commit();
        let shared: SharedOutbox = Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>));
        let peers: BTreeMap<NodeId, SocketAddr> = [(NodeId(2), dummy_addr())].into();
        let mut t = FlakyTransport::new(u32::MAX); // always QueueFull, lane_alive = true ⇒ retries then LaneStuck
        assert_eq!(
            replay_outbox_with_limits(&shared, &mut t, &peers, fast_limits()),
            Err(ReplayError::LaneStuck { peer: NodeId(2) })
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
    fn replay_outbox_fence_timeout_refuses_boot_without_gc() {
        // The row SENDS (the mock transport records it) but never SUBMITS back to the store (a real
        // MeshTransport would re-mirror; FlakyTransport does not), so `last_submitted` never reaches base + 1
        // ⇒ FenceTimeout past the short deadline (refuse-to-boot, loud). The `?` bails before gc ⇒ the row
        // SURVIVES. This is the count-fence deadline arm no other test reaches (the real-QUIC test passes it).
        let path = temp_path("fencetimeout");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
        ob.retain(&key(2, MsgClass::Saga, 1, 0), &framed_row(b"x"));
        ob.commit();
        let base = {
            let g = ob;
            let dh = g.durability();
            let b = dh.last_submitted();
            let shared: SharedOutbox = Arc::new(Mutex::new(Box::new(g) as Box<dyn OutboxSink + Send>));
            let peers: BTreeMap<NodeId, SocketAddr> = [(NodeId(2), dummy_addr())].into();
            let mut t = FlakyTransport::new(0); // sends OK (records) but never re-mirrors ⇒ no new submit
            let err = replay_outbox_with_limits(&shared, &mut t, &peers, fast_limits());
            assert_eq!(
                err,
                Err(ReplayError::FenceTimeout {
                    submitted: b,
                    expected: b + 1,
                }),
                "one row re-driven but never re-submitted ⇒ the fence times out at base + 1"
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
            b
        };
        let _ = base;
    }

    /// R-6d4-B1 (the DEFERRED.md F1-pin, deterministic — the writer-pause hook, NOT a latency race): a
    /// durable outbox row is INVISIBLE to `scan_all` until its block-B durability wait completes. Opens a
    /// `NodeOutbox` whose store writer PARKS pre-fsync on the retained row's exact key (`pause_on_key_prefix
    /// = key.to_bytes()`); while the marker proves the writer is parked (submitted-but-pre-fsync), `scan_all`
    /// must NOT contain the row and the handle must report it NOT durable. MUTATION-SENSITIVE: a submit that
    /// fsynced SYNCHRONOUSLY (breaking durable-before-send) would make the row visible while the marker is
    /// present ⇒ the `scan_all().is_empty()` assert flips RED.
    #[cfg(feature = "store-test-hooks")]
    #[test]
    fn a_row_is_not_visible_to_scan_all_until_block_b_waits() {
        let path = temp_path("orderpin");
        let _g = TempOutbox { path: path.clone() };
        let marker = path.with_extension("paused");
        let _ = std::fs::remove_file(&marker);
        let k = key(2, MsgClass::Saga, 5, 0);
        let tuning = StoreTuning {
            pause_on_key_prefix: Some(k.to_bytes().to_vec()), // park on THIS row's exact durable key
            pause_marker_path: Some(marker.clone()),
            ..StoreTuning::default()
        };
        let mut ob = NodeOutbox::open(&path, tuning).expect("open with the pause hook");
        ob.retain(&k, &framed_row(b"x"));
        let (seq, h) = ob.submit_barrier().expect("a staged retain ⇒ submitted");

        // Poll the marker to existence — the deterministic "submitted-but-pre-fsync window open" edge (the
        // writer writes it the instant it parks), NOT a wall-clock sleep-guess.
        let mut parked = false;
        for _ in 0..500 {
            if marker.exists() {
                parked = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(parked, "the writer parked pre-fsync on the submitted row (marker written)");
        assert!(!h.is_durable_through(seq), "submitted but NOT durable (paused pre-fsync)");
        assert!(h.durable_through() < seq, "the durable watermark lags the parked row");
        assert!(
            ob.scan_all().is_empty(),
            "durable-before-send ORDERING: a submitted-but-pre-fsync row is INVISIBLE to scan_all until \
             block B's durability wait completes (a synchronous-fsync submit would surface it here ⇒ RED)"
        );

        std::mem::forget(ob); // the parked writer never returns ⇒ Drop's join would hang
        let _ = std::fs::remove_file(&marker);
    }

    /// R-6d4-B2 (MUTATION-PROOF no-premature-gc): `replay_outbox`'s gc of the swept prior-incarnation copy is
    /// fenced behind the DURABILITY of the re-driven rows, NOT merely their submission. A capture transport
    /// bumps `submitted` (block A re-mirror) so the count-fence passes, but `durable` is held back (block B
    /// parked); WHILE the replay is parked at `wait_durable_through`, the 2 recoverable prior rows MUST still
    /// be present. MUTATION: moving gc before the durability wait (or fencing on submission) would sweep them
    /// here while un-durable = the exact D-6 #1 loss ⇒ the "still present" assert flips RED. Deterministic
    /// (test-driven watermarks, no fsync/timing race). A plain `#[test]` (`controllable()` needs no feature)
    /// so it runs in the default `coverage-io-prod` gate.
    #[test]
    fn gc_never_runs_before_the_replayed_rows_are_durable_mutation_proof() {
        use std::sync::atomic::{AtomicU64, Ordering};

        /// Models the real peer-writer re-mirror: each `send_durable` bumps the shared `submitted` watermark
        /// (block A), NEVER `durable` (block B's fsync is what the test controls) — so the replay's count-
        /// fence passes but `wait_durable_through` parks: the no-premature-gc window.
        struct RemirrorTransport {
            submitted: Arc<AtomicU64>,
        }
        impl Transport for RemirrorTransport {
            fn send_durable(
                &mut self,
                _to: NodeId,
                _class: MsgClass,
                _bytes: Bytes,
                _d: Durability,
            ) -> Result<vd_core::MsgId, SendError> {
                self.submitted.fetch_add(1, Ordering::Release);
                Ok(vd_core::MsgId(0))
            }
            fn drain_inbound(&mut self) -> Vec<vd_sim::io::Inbound> {
                Vec::new()
            }
            fn local_id(&self) -> NodeId {
                NodeId(1)
            }
        }
        impl ReplayTransport for RemirrorTransport {
            fn lane_alive(&self, _peer: NodeId) -> bool {
                true
            }
        }

        let (handle, submitted, durable) = DurabilityHandle::controllable();
        let mock = MockOutboxSink {
            rows: vec![
                (key(2, MsgClass::Saga, 1, 0), framed_row(b"a")),
                (key(2, MsgClass::Saga, 1, 1), framed_row(b"b")),
            ],
            durability: handle,
        };
        let shared: SharedOutbox = Arc::new(Mutex::new(Box::new(mock) as Box<dyn OutboxSink + Send>));
        let peers: BTreeMap<NodeId, SocketAddr> = [(NodeId(2), dummy_addr())].into();

        // Replay on a background thread: it re-drives both rows (submitted → 2), the count-fence passes
        // (2 >= base + 2), then PARKS at `wait_durable_through(2)` because `durable` is still 0.
        let shared_bg = Arc::clone(&shared);
        let sub_bg = Arc::clone(&submitted);
        let replay = std::thread::spawn(move || {
            let mut t = RemirrorTransport { submitted: sub_bg };
            replay_outbox(&shared_bg, &mut t, &peers)
        });

        // Wait until both re-mirrors submitted (the fence is reached) — the replay is now parked pre-durable.
        for _ in 0..1000 {
            if submitted.load(Ordering::Acquire) >= 2 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            submitted.load(Ordering::Acquire),
            2,
            "both rows re-mirrored (block A submit) — the count-fence is reached"
        );
        assert!(
            durable.load(Ordering::Acquire) < 2,
            "but they are NOT yet durable — the replay parks at block B (the anti-vacuity guard)"
        );
        assert_eq!(
            shared
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .scan_all()
                .len(),
            2,
            "no-premature-gc: the prior-incarnation rows SURVIVE until the re-driven rows are DURABLE (moving \
             gc before the durability wait would sweep them here while un-durable = the D-6 #1 loss ⇒ RED)"
        );

        // Release the parked replay: bump `durable` past the fence ⇒ wait_durable_through returns ⇒ gc runs.
        durable.store(2, Ordering::Release);
        let counts = replay.join().expect("replay thread joins").expect("replay ok");
        assert_eq!(counts.replayed, 2, "both rows were re-driven");
        assert_eq!(
            shared
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .scan_all()
                .len(),
            0,
            "AFTER durability, gc swept the now-redundant prior-incarnation rows"
        );
    }

    // ============================ R-6d4-A: the both-ends-restart replay proptest ============================
    // Drives the REAL `replay_outbox` through arbitrary crash/restart/redeliver interleavings and cross-checks
    // disk state against a HAND-MAINTAINED `Ref` computed independently (an agreement assert is load-bearing).
    // The dedup oracle drives the REAL `classify_reliable` receiver ladder (mesh.rs) so a SEQ-DEDUP regression
    // (a re-delivered seq accepted twice) turns it RED — see `witness_redeliver_of_a_delivered_row_is_deduped`.
    // SCOPE (honest, /goal wf_0b8c712a): this proptest proves the OUTBOX/REPLAY side end-to-end (no-loss,
    // no-orphan, accounting, source-idempotence) + the seq-dedup INTEGRATION; the receiver's FULL verdict
    // ladder (the A1 incarnation-reset, epoch, and Gap arms) is proven separately by mesh.rs's
    // `classify_reliable` unit tests — the oracle feeds each row's STORED incarnation, so it does not exercise
    // A1. Vetted design of record: scripts/r6d4a_vetted_design.md (wf_ec22b736).
    mod replay_proptest {
        use super::super::*; // the outbox module surface (OutboxKey/OutboxSink/replay_outbox/…)
        use super::{ALL_CLASSES, dummy_addr, fast_limits, framed_row};
        use crate::mesh::recv_test_hooks::RecvCell;
        use proptest::prelude::*;
        use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
        use std::cell::Cell;
        use std::collections::{BTreeMap, BTreeSet};
        use std::rc::Rc;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::{Arc, Mutex, PoisonError};
        use vd_sim::io::{Inbound, Transport};

        const ROSTER: [u64; 2] = [1, 2]; // the fixed route book; NodeId(9) is the off-roster (quarantine) peer

        /// The reference-model sink: a two-set RAM/disk crash boundary (committed = durable/survives-crash;
        /// staged = RAM/lost-on-crash), mirroring `NodeOutbox` + `RedbStore` (validated by the differential-
        /// vs-redb witness below). `Clone` so a BootReplay can run `replay_outbox` against a fresh wrapping.
        #[derive(Clone)]
        struct ModelOutboxSink {
            committed: BTreeMap<OutboxKey, Vec<u8>>,
            staged: BTreeMap<OutboxKey, Option<Vec<u8>>>,
            durability: DurabilityHandle,
        }
        impl OutboxSink for ModelOutboxSink {
            fn retain(&mut self, key: &OutboxKey, framed: &[u8]) {
                self.staged.insert(*key, Some(framed.to_vec()));
            }
            fn release(&mut self, key: &OutboxKey) {
                self.staged.insert(*key, None);
            }
            fn commit(&mut self) {
                for (k, v) in std::mem::take(&mut self.staged) {
                    match v {
                        Some(framed) => {
                            self.committed.insert(k, framed);
                        }
                        None => {
                            self.committed.remove(&k);
                        }
                    }
                }
            }
            fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)> {
                self.committed.iter().map(|(k, v)| (*k, v.clone())).collect()
            }
            fn gc_below(&mut self, incarnation: u64) {
                self.committed.retain(|k, _| k.incarnation >= incarnation);
            }
            fn gc_replayed(&mut self, replayed_keys: &[OutboxKey]) {
                for k in replayed_keys {
                    self.committed.remove(k);
                }
            }
            fn submit_barrier(&mut self) -> Option<(u64, DurabilityHandle)> {
                if self.staged.is_empty() {
                    None
                } else {
                    Some((1, self.durability.clone()))
                }
            }
            fn durability(&self) -> DurabilityHandle {
                self.durability.clone()
            }
        }

        /// The receiver-integration oracle: drives the REAL `classify_reliable` (mesh.rs) per `(peer,class)`,
        /// keyed on the full `OutboxKey`. `accept` returns true iff the ladder delivered (Accept/Reset). The
        /// SEQ-DEDUP arm is asserted here (a re-delivered seq ⇒ `false`, `witness_redeliver...`); the A1
        /// incarnation-reset / epoch / Gap arms are proven by mesh.rs's `classify_reliable` unit tests (the
        /// oracle feeds each row's STORED incarnation, so it does not drive A1 — scope note above).
        struct DedupLedger {
            states: BTreeMap<(NodeId, MsgClass), RecvCell>,
            delivered: BTreeSet<OutboxKey>,
            arrivals: usize,
        }
        impl DedupLedger {
            fn new() -> Self {
                DedupLedger {
                    states: BTreeMap::new(),
                    delivered: BTreeSet::new(),
                    arrivals: 0,
                }
            }
            fn accept(&mut self, k: OutboxKey) -> bool {
                self.arrivals += 1;
                let cell = self.states.entry((k.peer, k.class)).or_insert_with(RecvCell::fresh);
                // The outbox re-frames at epoch = u32::MAX (framed_row / the write-path re-stamp), so the
                // receiver sees that epoch; the incarnation/seq are the OutboxKey's.
                if cell.accept(k.incarnation, u32::MAX, k.seq) {
                    self.delivered.insert(k)
                } else {
                    false
                }
            }
        }

        /// A capture transport modeling the mesh block-A re-mirror + block-B fsync synchronously: each
        /// `send_durable` bumps BOTH the shared `submitted` (block A) and `durable` (fsync) watermarks (so
        /// `replay_outbox`'s count-fence passes + `wait_durable_through` fast-returns, no thread/park) and
        /// records the send. `fail_after`/`lane_alive` drive the LaneStuck/LaneDead arms.
        struct CaptureTransport {
            submitted: Arc<AtomicU64>,
            durable: Arc<AtomicU64>,
            lane_alive: bool,
            fail_after: Option<usize>,
            calls: usize,
            sent: Vec<(NodeId, MsgClass, Vec<u8>)>,
        }
        impl Transport for CaptureTransport {
            fn send_durable(
                &mut self,
                to: NodeId,
                class: MsgClass,
                bytes: Bytes,
                _d: Durability,
            ) -> Result<vd_core::MsgId, SendError> {
                if self.fail_after.is_some_and(|k| self.calls >= k) {
                    return Err(SendError::QueueFull(bytes)); // a full lane (lane_alive drives Stuck-vs-Dead)
                }
                self.calls += 1;
                self.submitted.fetch_add(1, Ordering::Release);
                self.durable.fetch_add(1, Ordering::Release);
                self.sent.push((to, class, bytes.to_vec()));
                Ok(vd_core::MsgId(0))
            }
            fn drain_inbound(&mut self) -> Vec<Inbound> {
                Vec::new()
            }
            fn local_id(&self) -> NodeId {
                NodeId(1)
            }
        }
        impl ReplayTransport for CaptureTransport {
            fn lane_alive(&self, _peer: NodeId) -> bool {
                self.lane_alive
            }
        }

        /// A fresh `(handle, transport)` sharing ONE `(submitted, durable)` atomic pair (INV-8): the sink's
        /// durability handle + the transport MUST bump the same atomics or the fence math desyncs.
        fn fresh_capture_pair(
            fail_after: Option<usize>,
            lane_alive: bool,
        ) -> (DurabilityHandle, CaptureTransport) {
            let (handle, submitted, durable) = DurabilityHandle::controllable();
            (
                handle,
                CaptureTransport {
                    submitted,
                    durable,
                    lane_alive,
                    fail_after,
                    calls: 0,
                    sent: Vec::new(),
                },
            )
        }

        #[derive(Clone, Debug)]
        enum Op {
            Retain {
                peer: u64,
                class: usize,
                seq: u64,
                payload: u8,
                garbage: bool,
            },
            Commit,
            Ack {
                idx: usize,
            },
            SourceCrash,
            BootReplay,
            DestRedeliver {
                idx: usize,
            },
        }

        fn op_strategy() -> impl Strategy<Value = Op> {
            prop_oneof![
                4 => (
                    prop_oneof![3 => Just(1u64), 3 => Just(2u64), 1 => Just(9u64)],
                    0usize..ALL_CLASSES.len(),
                    0u64..4,
                    any::<u8>(),
                    prop::bool::weighted(0.15), // occasionally an undecodable payload (quarantine arm)
                ).prop_map(|(peer, class, seq, payload, garbage)| Op::Retain { peer, class, seq, payload, garbage }),
                2 => Just(Op::Commit),
                1 => (0usize..8).prop_map(|idx| Op::Ack { idx }),
                1 => Just(Op::SourceCrash),
                2 => Just(Op::BootReplay),
                1 => (0usize..8).prop_map(|idx| Op::DestRedeliver { idx }),
            ]
        }

        fn roster() -> BTreeMap<NodeId, SocketAddr> {
            ROSTER.iter().map(|p| (NodeId(*p), dummy_addr())).collect()
        }

        /// Apply an op sequence to BOTH the model sink (code-driven) and the independent `Ref`, asserting the
        /// invariants after every op. `cov` records which hard arms fired (the anti-vacuity floor).
        fn model_check(ops: &[Op], cov: &Cell<[bool; 4]>) {
            let mut model = ModelOutboxSink {
                committed: BTreeMap::new(),
                staged: BTreeMap::new(),
                durability: DurabilityHandle::already_durable(), // placeholder; re-set per BootReplay
            };
            // Independent reference (maintained BY HAND, never by calling the sink):
            let mut committed_ref: BTreeMap<OutboxKey, Vec<u8>> = BTreeMap::new();
            let mut staged_ref: BTreeMap<OutboxKey, Option<Vec<u8>>> = BTreeMap::new();
            let mut incarnation: u64 = 0;
            let mut ledger = DedupLedger::new();
            let roster = roster();

            let mut flags = cov.get();
            for op in ops {
                match op {
                    Op::Retain {
                        peer,
                        class,
                        seq,
                        payload,
                        garbage,
                    } => {
                        let k = OutboxKey {
                            peer: NodeId(*peer),
                            class: ALL_CLASSES[*class],
                            incarnation,
                            seq: *seq,
                        };
                        let framed = if *garbage {
                            vec![0xFEu8, 0xFF, 0x00] // not a decodable ReliableFrame ⇒ quarantine on replay
                        } else {
                            framed_row(&[*payload])
                        };
                        model.retain(&k, &framed);
                        staged_ref.insert(k, Some(framed));
                    }
                    Op::Commit => {
                        model.commit();
                        for (k, v) in std::mem::take(&mut staged_ref) {
                            match v {
                                Some(f) => {
                                    committed_ref.insert(k, f);
                                }
                                None => {
                                    committed_ref.remove(&k);
                                }
                            }
                        }
                    }
                    Op::Ack { idx } => {
                        let keys: Vec<OutboxKey> =
                            committed_ref.keys().chain(staged_ref.keys()).copied().collect();
                        if !keys.is_empty() {
                            let k = keys[*idx % keys.len()];
                            model.release(&k);
                            staged_ref.insert(k, None);
                        }
                    }
                    Op::SourceCrash => {
                        if !staged_ref.is_empty() {
                            flags[0] = true; // crash-with-nonempty-staged actually taken
                        }
                        // RAM lost, disk kept: staged cleared, committed unchanged; a restart = fresh incarnation.
                        model.staged.clear();
                        staged_ref.clear();
                        incarnation += 1;
                    }
                    Op::BootReplay => {
                        // A boot is a process (re)START: the RAM staged window is GONE — `replay_outbox`
                        // runs on a freshly-opened store (committed/durable only). Clearing staged here is
                        // the load-bearing fidelity (else the replay's final `commit` would drain a
                        // pre-boot uncommitted retain back in — the exact divergence the fuzzer caught).
                        model.staged.clear();
                        staged_ref.clear();
                        let rows_scanned = committed_ref.len();
                        // Ref predicts replay INDEPENDENTLY, ascending key order (BTreeMap == redb scan order):
                        let mut expected: Vec<OutboxKey> = Vec::new();
                        let mut quarantined = 0usize;
                        for (k, framed) in &committed_ref {
                            let routable = roster.contains_key(&k.peer);
                            let decodable = decode_value_payload(framed).is_some();
                            if routable && decodable {
                                expected.push(*k);
                            } else {
                                quarantined += 1;
                            }
                        }
                        // Drive the REAL replay against a fresh wrapping sharing the capture pair's atomics.
                        let (handle, mut transport) = fresh_capture_pair(None, true);
                        model.durability = handle;
                        let shared: SharedOutbox =
                            Arc::new(Mutex::new(Box::new(model.clone()) as Box<dyn OutboxSink + Send>));
                        let counts =
                            replay_outbox(&shared, &mut transport, &roster).expect("healthy replay ok");
                        // Extract the post-gc committed back (the code's gc is authoritative).
                        model.committed = shared
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .scan_all()
                            .into_iter()
                            .collect();
                        // INV-4 ACCOUNTING (RC-2a no-silent-loss):
                        assert_eq!(counts.replayed, expected.len(), "replayed == routable+decodable rows");
                        assert_eq!(counts.quarantined, quarantined, "quarantined == off-roster + undecodable");
                        assert_eq!(
                            counts.replayed + counts.quarantined,
                            rows_scanned,
                            "every scanned row is either re-driven or quarantined (none silently vanishes)"
                        );
                        // Positional: the transport saw EXACTLY the expected deliveries, in order.
                        let want_sent: Vec<(NodeId, MsgClass, Vec<u8>)> = expected
                            .iter()
                            .map(|k| {
                                let pay = decode_value_payload(&committed_ref[k]).expect("decodable");
                                (k.peer, k.class, pay.to_vec())
                            })
                            .collect();
                        assert_eq!(transport.sent, want_sent, "capture.sent == expected deliveries, in order");
                        // Ref: re-driven keys leave committed; quarantined keys STAY (no-orphan — verified
                        // against the sink's actual post-gc committed by the INV-1 equality below).
                        for k in &expected {
                            committed_ref.remove(k);
                        }
                        let redrove_after_crash = incarnation > 0 && !expected.is_empty();
                        for k in &expected {
                            ledger.accept(*k);
                        }
                        if redrove_after_crash {
                            flags[1] = true;
                        }
                        if quarantined > 0 {
                            flags[3] = true;
                        }
                    }
                    Op::DestRedeliver { idx } => {
                        let ks: Vec<OutboxKey> = ledger.delivered.iter().copied().collect();
                        if !ks.is_empty() {
                            let k = ks[*idx % ks.len()];
                            let before = ledger.delivered.len();
                            assert!(!ledger.accept(k), "a redelivery of a delivered row is deduped");
                            assert_eq!(ledger.delivered.len(), before, "dedup adds no new effect");
                            flags[2] = true;
                        }
                    }
                }
                // INV-1 NO-LOSS + INV-5 (independent scan_all): the model's durable set equals the hand-
                // maintained committed reference after EVERY op.
                assert_eq!(model.committed, committed_ref, "sink.committed == Ref.committed (no-loss)");
                assert_eq!(
                    model.scan_all(),
                    committed_ref
                        .iter()
                        .map(|(k, v)| (*k, v.clone()))
                        .collect::<Vec<_>>(),
                    "scan_all agrees with the reference (ascending)"
                );
                // NO-ORPHAN is subsumed by the INV-1 equality above: Ref keeps quarantined rows in committed
                // while removing exactly the re-driven ones; a gc that swept a quarantined/non-scanned row
                // (the pre-F1 blunt gc_below) would make model.committed lose it ⇒ the equality goes RED.
            }
            // INV-3 (receiver): arrivals never under-count distinct effects.
            assert!(ledger.arrivals >= ledger.delivered.len(), "arrivals >= distinct effects");
            cov.set(flags);
        }

        #[test]
        fn both_ends_restart_replay_is_no_loss_no_orphan_exactly_once() {
            let cov = Rc::new(Cell::new([false; 4]));
            let cov_run = Rc::clone(&cov);
            let mut runner = TestRunner::new_with_rng(
                Config {
                    cases: 1024,
                    ..Config::default()
                },
                TestRng::from_seed(RngAlgorithm::ChaCha, &[0x6du8; 32]),
            );
            if let Err(e) = runner.run(&prop::collection::vec(op_strategy(), 0..=48), move |ops| {
                model_check(&ops, &cov_run);
                Ok(())
            }) {
                panic!("proptest failed; minimal case: {e:?}");
            }
            let f = cov.get();
            assert!(f[0], "hit ≥1 SourceCrash with a non-empty staged set");
            assert!(f[1], "hit ≥1 BootReplay-after-crash that re-drove a row");
            assert!(f[2], "hit ≥1 DestRedeliver of an already-delivered row");
            assert!(f[3], "hit ≥1 BootReplay that quarantined a row");
        }

        // ---- hand-written witnesses (deterministic regressions kept alongside the fuzzer) ----

        #[test]
        fn witness_source_crash_drops_staged_keeps_committed() {
            let cov = Cell::new([false; 4]);
            model_check(
                &[
                    Op::Retain { peer: 1, class: 0, seq: 0, payload: 1, garbage: false },
                    Op::Commit,
                    Op::Retain { peer: 1, class: 0, seq: 1, payload: 2, garbage: false }, // staged
                    Op::SourceCrash,
                ],
                &cov,
            );
            assert!(cov.get()[0], "the crash dropped a non-empty staged set");
        }

        #[test]
        fn witness_boot_replay_after_crash_redrives_and_is_idempotent() {
            let cov = Cell::new([false; 4]);
            model_check(
                &[
                    Op::Retain { peer: 1, class: 0, seq: 0, payload: 7, garbage: false },
                    Op::Commit,
                    Op::SourceCrash,
                    Op::BootReplay, // re-drives the durable row (committed empty after)
                    Op::BootReplay, // idempotent: nothing left ⇒ replayed 0 (asserted via accounting)
                ],
                &cov,
            );
            assert!(cov.get()[1], "a BootReplay-after-crash re-drove a row");
        }

        #[test]
        fn witness_redeliver_of_a_delivered_row_is_deduped() {
            let cov = Cell::new([false; 4]);
            model_check(
                &[
                    Op::Retain { peer: 2, class: 0, seq: 0, payload: 3, garbage: false },
                    Op::Commit,
                    Op::BootReplay,               // delivers the row once
                    Op::DestRedeliver { idx: 0 }, // a duplicate ⇒ deduped (asserted inside)
                ],
                &cov,
            );
            assert!(cov.get()[2], "the redelivery was deduped");
        }

        #[test]
        fn witness_quarantine_off_roster_row() {
            let cov = Cell::new([false; 4]);
            model_check(
                &[
                    Op::Retain { peer: 9, class: 0, seq: 0, payload: 5, garbage: false }, // off-roster
                    Op::Commit,
                    Op::BootReplay, // quarantined + retained (asserted via accounting)
                ],
                &cov,
            );
            assert!(cov.get()[3], "the off-roster row was quarantined");
        }

        #[test]
        fn witness_crash_mid_replay_loses_nothing() {
            // The sharpest: a FIRST replay that dies partway (LaneStuck) skips gc (refuse-to-boot) ⇒ NO partial
            // sweep; a FRESH replay re-drives BOTH, and the row delivered twice across boots dedups to ONE effect.
            let roster = roster();
            let k1 = OutboxKey { peer: NodeId(2), class: MsgClass::Saga, incarnation: 0, seq: 0 };
            let k2 = OutboxKey { peer: NodeId(2), class: MsgClass::Saga, incarnation: 0, seq: 1 };
            let mut committed: BTreeMap<OutboxKey, Vec<u8>> = BTreeMap::new();
            committed.insert(k1, framed_row(&[1]));
            committed.insert(k2, framed_row(&[2]));
            let mut ledger = DedupLedger::new();

            // First replay: fail_after=1 ⇒ re-drives k1, then the k2 send stays QueueFull past the retry cap.
            let (handle, mut t1) = fresh_capture_pair(Some(1), true);
            let sink1 = ModelOutboxSink { committed: committed.clone(), staged: BTreeMap::new(), durability: handle };
            let shared1: SharedOutbox = Arc::new(Mutex::new(Box::new(sink1) as Box<dyn OutboxSink + Send>));
            let err = replay_outbox_with_limits(&shared1, &mut t1, &roster, fast_limits());
            assert_eq!(err, Err(ReplayError::LaneStuck { peer: NodeId(2) }), "the partial replay refuses to boot");
            assert_eq!(
                shared1.lock().unwrap_or_else(PoisonError::into_inner).scan_all().len(),
                2,
                "refuse-to-boot skips gc ⇒ BOTH rows survive (no partial sweep of the re-driven k1)"
            );

            // Fresh replay (healthy): re-drives BOTH; k1 is delivered a SECOND time across the two boots.
            let (handle2, mut t2) = fresh_capture_pair(None, true);
            let sink2 = ModelOutboxSink { committed, staged: BTreeMap::new(), durability: handle2 };
            let shared2: SharedOutbox = Arc::new(Mutex::new(Box::new(sink2) as Box<dyn OutboxSink + Send>));
            let counts = replay_outbox(&shared2, &mut t2, &roster).expect("healthy replay ok");
            assert_eq!(counts.replayed, 2, "the fresh replay re-drove both rows");
            // Feed the ledger in send order across BOTH boots: k1 (boot1), then k1,k2 (boot2).
            ledger.accept(k1);
            ledger.accept(k1); // the cross-boot re-delivery of k1 ⇒ deduped
            ledger.accept(k2);
            assert_eq!(ledger.delivered, [k1, k2].into_iter().collect(), "exactly-once across a crash-mid-replay");
            assert_eq!(ledger.arrivals, 3, "3 arrivals (k1 twice), 2 distinct effects");
        }

        /// Differential-vs-redb (MF-3): pin the model sink's staged/committed semantics to the REAL
        /// `NodeOutbox` on a fixed prefix — so the fast reference model cannot silently drift from redb.
        #[test]
        fn model_sink_matches_a_real_node_outbox() {
            let path = super::temp_path("model-diff");
            let _g = super::TempOutbox { path: path.clone() };
            let mut real = NodeOutbox::open(&path, StoreTuning::default()).expect("open");
            let mut model = ModelOutboxSink {
                committed: BTreeMap::new(),
                staged: BTreeMap::new(),
                durability: DurabilityHandle::already_durable(),
            };
            let k = |p, s| OutboxKey { peer: NodeId(p), class: MsgClass::Saga, incarnation: 1, seq: s };
            let steps: &[(OutboxKey, Option<u8>)] = &[
                (k(1, 0), Some(10)),
                (k(1, 1), Some(11)),
                (k(2, 0), Some(20)),
                (k(1, 0), None), // release
            ];
            for (key, val) in steps {
                match val {
                    Some(p) => {
                        let f = framed_row(&[*p]);
                        real.retain(key, &f);
                        model.retain(key, &f);
                    }
                    None => {
                        real.release(key);
                        model.release(key);
                    }
                }
                real.commit();
                model.commit();
                assert_eq!(real.scan_all(), model.scan_all(), "model tracks NodeOutbox after each commit");
            }
            // Cover the trait surface the replay proptest does not drive (submit_barrier / gc_below), keeping
            // the reference model honest to those NodeOutbox semantics too.
            assert!(model.submit_barrier().is_none(), "empty staged ⇒ no barrier (matches NodeOutbox)");
            model.retain(&k(3, 0), &framed_row(&[9]));
            assert!(model.submit_barrier().is_some(), "a staged retain ⇒ a barrier");
            model.commit();
            model.gc_below(2); // every row here is incarnation 1 ⇒ swept by the incarnation-2 floor
            assert!(model.scan_all().is_empty(), "gc_below(2) swept the incarnation-1 rows");
        }
    }
}
