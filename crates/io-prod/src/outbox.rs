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

use std::path::Path;

use vd_core::ids::NodeId;
use vd_sim::io::{Bytes, MsgClass, Store, bytes};

use crate::store::{DurabilityHandle, RedbStore, StoreError, StoreTuning};

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
    pub fn open(path: impl AsRef<Path>, tuning: StoreTuning) -> Result<NodeOutbox, StoreError> {
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
