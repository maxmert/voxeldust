//! The connection-plane channel families (HR1 taxonomy 2 of 2;
//! `docs/design/connection_plane.md` §1.1/§6.1).
//!
//! Every byte between client and gateway is exactly one of these closed types — no
//! `Raw`, no `Other`. The old single ~50-variant `ServerMsg` god-enum is replaced by
//! per-channel enums whose reliability/ordering matches their QUIC carrier:
//!
//! | family            | carrier            | reliability        |
//! |-------------------|--------------------|--------------------|
//! | [`ControlMsg`]    | 1 bidi stream      | reliable, ordered  |
//! | [`InputDatagram`] | datagrams C→G      | unreliable, seq    |
//! | [`BulkMsg`]       | uni per sub G→C    | reliable, paced    |
//! | [`EventMsg`]      | uni per sub G→C    | reliable           |
//! | [`SnapshotDatagram`] | datagrams G→C   | unreliable         |
//!
//! Transfer internals (`ShardRedirect`/`ShardHandoff`/observer messages of the old
//! system) DO NOT EXIST on the client wire: the client sees at most a cosmetic
//! notice. v1 payloads are deliberately minimal; growth is additive under
//! `proto_minor` negotiation.

use serde::{Deserialize, Serialize};
use vd_core::pose::{FrameRef, StampedPose};
use vd_core::{EntityId, EpochId, Fence, SessionId, TickId, TransferId, UniverseTick};

use crate::seams::tickets::{LoginTicket, ResumeTicket};
use crate::version::ProtoVersion;

/// A client-facing render-layer subscription id: monotonically increasing per
/// session, NEVER reused (stale in-flight datagrams are dropped by id mismatch).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SubId(pub u32);

/// Why a transfer was rejected, surfaced to the client as feedback
/// ("hatch obstructed"), never as a stall.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferRejectReason {
    SpatiallyObstructed,
    DestinationUnavailable,
    VersionFloor,
}

/// Client → gateway control messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClientControlMsg {
    Hello {
        version: ProtoVersion,
        login: LoginTicket,
    },
    Resume {
        version: ProtoVersion,
        ticket: ResumeTicket,
    },
    /// Confirms the in-band CUT_MARKER was emitted at `marker_seq`.
    CutEmitted {
        transfer: TransferId,
        marker_seq: u64,
    },
    Pong {
        nonce: u64,
    },
    Bye,
}

/// Gateway → client control messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ServerControlMsg {
    Welcome {
        version: ProtoVersion,
        session: SessionId,
        session_fence: Fence,
        epoch: EpochId,
    },
    SubscriptionOpened {
        sub: SubId,
        frame: FrameRef,
    },
    SubscriptionClosing {
        sub: SubId,
    },
    /// Exactly one sub renders authoritatively for an entity (invariant A1).
    AuthorityChanged {
        entity: EntityId,
        sub: SubId,
    },
    /// Ask the client to emit the in-band CUT_MARKER on its input flow.
    RequestCut {
        transfer: TransferId,
    },
    CutConfirmed {
        transfer: TransferId,
        marker_seq: u64,
    },
    /// Cosmetic transfer feedback (warp progress stretches the masking visual).
    TransferCosmetic {
        transfer: TransferId,
        progress_pct: u8,
    },
    TransferRejected {
        transfer: TransferId,
        reason: TransferRejectReason,
    },
    Ping {
        nonce: u64,
    },
    Close {
        reason: String,
    },
    /// The cluster's universe-tick rate (Hz), so the client drives its render cursor
    /// at the server's rate instead of a hard-coded guess. Appended trailing variant
    /// (proto_minor 1), emitted only to a peer that negotiated minor >= 1. (An
    /// APPENDED VARIANT is the only postcard-safe additive shape — a new struct field
    /// is NOT, even with `#[serde(default)]`; postcard is non-self-describing.)
    UniverseRate {
        tick_hz: u32,
    },
}

/// The 20 Hz client input frame (latest-wins; loss = skip a tick, never a wedge).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputDatagram {
    /// Strictly increasing per session; the gateway dedups and routes by seq
    /// relative to the transfer cut marker.
    pub seq: u64,
    /// The ONE input that must not be lost: triple-sent + CONTROL-confirmed.
    pub is_cut_marker: bool,
    pub client_tick: TickId,
    /// Movement axes in [-1, 1] (forward, strafe, vertical).
    pub movement: [f32; 3],
    /// Look delta (yaw, pitch) in radians.
    pub look: [f32; 2],
    /// Momentary action bits (jump, interact, ...); discrete WORLD-MUTATING actions
    /// ride reliable channels instead (v1.1).
    pub action_bits: u32,
}

impl InputDatagram {
    /// Every float component is finite — the AUTHORITATIVE-ingress gate (the server
    /// twin of the client's delivered-pose `sanitized()` chokepoint): a forged/corrupt
    /// NaN or Inf in `movement`/`look` would otherwise integrate into the shard's
    /// authoritative pose and STICK (NaN propagates through every subsequent tick),
    /// fanning out to every observer. Never trust network input — a shard MUST discard
    /// (and count) a non-finite datagram instead of integrating it.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.movement.iter().all(|c| c.is_finite()) && self.look.iter().all(|c| c.is_finite())
    }
}

/// Reliable per-subscription bulk data (chunk payloads land at P4/P6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BulkMsg {
    /// Opaque, content-addressed bulk blob; concrete chunk schemas arrive with
    /// terrain (P4) as additive variants.
    Blob { kind: BulkKind, bytes: Vec<u8> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BulkKind {
    ChunkSnapshot,
    ChunkDelta,
    Catalog,
}

/// Reliable per-subscription discrete gameplay events.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EventMsg {
    Notice { text: String },
    EntityRemoved { entity: EntityId },
}

/// One entity's state inside a snapshot frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntitySnap {
    pub entity: EntityId,
    pub pose: StampedPose,
}

/// The 20 Hz per-subscription world-state datagram.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SnapshotDatagram {
    pub sub: SubId,
    /// Monotonic per-sub frame counter (stale frames dropped).
    pub frame_id: u64,
    /// The SENDER's local sim tick (there is NO global sim tick).
    pub source_tick: TickId,
    /// The analytic clock value this frame's poses are stamped against.
    pub universe_tick: UniverseTick,
    pub entities: Vec<EntitySnap>,
}

/// The fixed per-datagram overhead (sub + frame_id + source_tick + universe_tick +
/// the entities-vec length prefix) — a generous postcard upper bound. Subtracted from
/// the budget so a packed chunk's ENCODED size stays under the datagram MTU.
const SNAPSHOT_HEADER_BUDGET: usize = 40;

/// The conservative datagram budget every shard's snapshot partitioning must stay at
/// or below: derived from the IPv6 minimum-MTU QUIC datagram floor minus headroom for
/// the gateway's `sub_id` re-tag and QUIC/UDP overhead. A configured budget above this
/// risks the path MTU; the bin asserts it at boot (never-silent, audit GW-1 §6.3).
pub const CONSERVATIVE_DATAGRAM_BUDGET: usize = 1200;

/// Split a world's entities into chunks each of which encodes to <= `budget_bytes` as a
/// `SnapshotDatagram` (`connection_plane.md` §6.3: oversize snapshots are partitioned BY
/// CONTENT into independent self-contained datagrams). THE shared, shard-agnostic
/// partitioner — every shard calls this so a full-world snapshot never silently exceeds
/// the QUIC datagram MTU (audit GW-1). Each returned chunk is non-empty (a single entity
/// larger than the budget still ships alone, where the transport's counted-drop guard
/// catches the impossible case). Greedy, deterministic, O(n) encodes.
#[must_use]
pub fn partition_entities(entities: &[EntitySnap], budget_bytes: usize) -> Vec<Vec<EntitySnap>> {
    let body_budget = budget_bytes.saturating_sub(SNAPSHOT_HEADER_BUDGET).max(1);
    let mut chunks: Vec<Vec<EntitySnap>> = Vec::new();
    let mut current: Vec<EntitySnap> = Vec::new();
    let mut current_bytes = 0usize;
    for snap in entities {
        let snap_bytes = postcard::to_allocvec(snap)
            .map(|v| v.len())
            .unwrap_or(SNAPSHOT_HEADER_BUDGET);
        // Start a new chunk if this entity would overflow the body budget — unless the
        // chunk is empty (one oversize entity still ships alone).
        if !current.is_empty() && current_bytes + snap_bytes > body_budget {
            chunks.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current.push(*snap);
        current_bytes += snap_bytes;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// What a client does with one arriving [`SnapshotDatagram`], given the
/// subscription it currently holds and the highest `frame_id` it has applied for
/// that sub.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotVerdict {
    /// Apply the frame's entities and advance the per-sub high-water to `frame_id`.
    Apply,
    /// Drop: the datagram targets a subscription the client does not hold (a stale
    /// in-flight frame after a re-subscribe).
    DropForeignSub,
    /// Drop: a STRICTLY-older `frame_id` — a late straggler from a past tick.
    DropStale,
}

/// THE §6.3 snapshot staleness gate, shared by every snapshot consumer (the
/// in-process `ScriptedClient` and the real client) so they cannot drift: a
/// strictly-older `frame_id` is stale, but an EQUAL `frame_id` is a sibling chunk
/// of the current tick (a partitioned multi-datagram snapshot) and is APPLIED —
/// each chunk self-contained, latest-wins. A foreign sub drops.
#[must_use]
pub fn classify_snapshot(
    held_sub: Option<SubId>,
    high_water: Option<u64>,
    snap_sub: SubId,
    snap_frame_id: u64,
) -> SnapshotVerdict {
    if held_sub != Some(snap_sub) {
        return SnapshotVerdict::DropForeignSub;
    }
    if high_water.is_some_and(|hw| snap_frame_id < hw) {
        return SnapshotVerdict::DropStale;
    }
    SnapshotVerdict::Apply
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::entity_kind::EntityKind;
    use vd_core::glam::DVec3;

    fn eid() -> EntityId {
        EntityId::pack(EntityKind::Player, 1, 1, 1)
    }

    #[test]
    fn classify_snapshot_applies_siblings_and_drops_stale_or_foreign() {
        let held = SubId(0);
        // No sub held yet, or a different sub: foreign.
        assert_eq!(
            classify_snapshot(None, None, held, 5),
            SnapshotVerdict::DropForeignSub
        );
        assert_eq!(
            classify_snapshot(Some(SubId(9)), Some(5), held, 5),
            SnapshotVerdict::DropForeignSub
        );
        // Held sub, no high-water yet: apply.
        assert_eq!(
            classify_snapshot(Some(held), None, held, 5),
            SnapshotVerdict::Apply
        );
        // Newer frame: apply (advances the tick).
        assert_eq!(
            classify_snapshot(Some(held), Some(5), held, 6),
            SnapshotVerdict::Apply
        );
        // EQUAL frame: a sibling chunk of the current tick — apply (latest-wins).
        assert_eq!(
            classify_snapshot(Some(held), Some(5), held, 5),
            SnapshotVerdict::Apply
        );
        // Strictly older: stale.
        assert_eq!(
            classify_snapshot(Some(held), Some(5), held, 4),
            SnapshotVerdict::DropStale
        );
    }

    #[test]
    fn control_messages_roundtrip() {
        let msgs = vec![
            ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: SessionId(5),
                session_fence: Fence(1),
                epoch: EpochId(2),
            },
            ServerControlMsg::SubscriptionOpened {
                sub: SubId(3),
                frame: FrameRef::SystemSpace { system_seed: 9 },
            },
            ServerControlMsg::SubscriptionClosing { sub: SubId(3) },
            ServerControlMsg::AuthorityChanged {
                entity: eid(),
                sub: SubId(4),
            },
            ServerControlMsg::RequestCut {
                transfer: TransferId(7),
            },
            ServerControlMsg::CutConfirmed {
                transfer: TransferId(7),
                marker_seq: 99,
            },
            ServerControlMsg::TransferCosmetic {
                transfer: TransferId(7),
                progress_pct: 50,
            },
            ServerControlMsg::TransferRejected {
                transfer: TransferId(7),
                reason: TransferRejectReason::SpatiallyObstructed,
            },
            ServerControlMsg::Ping { nonce: 1 },
            ServerControlMsg::Close {
                reason: "test".into(),
            },
            ServerControlMsg::UniverseRate { tick_hz: 50 },
        ];
        for msg in msgs {
            let bytes = postcard::to_allocvec(&msg).expect("encode");
            let back: ServerControlMsg = postcard::from_bytes(&bytes).expect("decode");
            assert_eq!(back, msg);
        }
    }

    #[test]
    fn appended_variant_is_additive_a_prior_variant_decodes_unchanged() {
        // The minor-1 additive shape: bytes a minor-0 sender produced (any variant
        // BEFORE UniverseRate) still decode unchanged on a minor-1 decoder — appending
        // a trailing variant never shifts a prior variant's discriminant or framing.
        let prior = ServerControlMsg::Close {
            reason: "bye".into(),
        };
        let bytes = postcard::to_allocvec(&prior).expect("encode");
        let back: ServerControlMsg = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, prior);
        // And the new variant itself is a clean self-contained message.
        let rate = ServerControlMsg::UniverseRate { tick_hz: 20 };
        let rate_bytes = postcard::to_allocvec(&rate).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ServerControlMsg>(&rate_bytes).expect("decode"),
            rate
        );
    }

    #[test]
    fn client_control_roundtrip() {
        let msgs = vec![
            ClientControlMsg::CutEmitted {
                transfer: TransferId(1),
                marker_seq: 12,
            },
            ClientControlMsg::Pong { nonce: 4 },
            ClientControlMsg::Bye,
        ];
        for msg in msgs {
            let bytes = postcard::to_allocvec(&msg).expect("encode");
            let back: ClientControlMsg = postcard::from_bytes(&bytes).expect("decode");
            assert_eq!(back, msg);
        }
    }

    #[test]
    fn input_finite_gate_catches_nan_and_inf_in_either_field() {
        let finite = InputDatagram {
            seq: 1,
            is_cut_marker: false,
            client_tick: TickId(1),
            movement: [1.0, -1.0, 0.0],
            look: [0.1, -0.2],
            action_bits: 0,
        };
        assert!(finite.is_finite());
        // NaN in movement trips the gate (the look arm short-circuits — both arms below).
        let mut bad = finite;
        bad.movement[1] = f32::NAN;
        assert!(!bad.is_finite());
        // Inf in look trips the gate with movement finite (covers the second arm).
        let mut bad = finite;
        bad.look[0] = f32::INFINITY;
        assert!(!bad.is_finite());
        // -Inf likewise.
        let mut bad = finite;
        bad.look[1] = f32::NEG_INFINITY;
        assert!(!bad.is_finite());
    }

    #[test]
    fn input_and_snapshot_datagrams_roundtrip() {
        let input = InputDatagram {
            seq: 42,
            is_cut_marker: true,
            client_tick: TickId(7),
            movement: [1.0, 0.0, -1.0],
            look: [0.1, -0.2],
            action_bits: 0b101,
        };
        let bytes = postcard::to_allocvec(&input).expect("encode");
        assert_eq!(
            postcard::from_bytes::<InputDatagram>(&bytes).expect("decode"),
            input
        );

        let snap = SnapshotDatagram {
            sub: SubId(1),
            frame_id: 100,
            source_tick: TickId(50),
            universe_tick: UniverseTick(2000),
            entities: vec![EntitySnap {
                entity: eid(),
                pose: StampedPose::at_rest(
                    FrameRef::PlanetCentered { planet_seed: 3 },
                    DVec3::new(1.0, 2.0, 3.0),
                    UniverseTick(2000),
                ),
            }],
        };
        let bytes = postcard::to_allocvec(&snap).expect("encode");
        assert_eq!(
            postcard::from_bytes::<SnapshotDatagram>(&bytes).expect("decode"),
            snap
        );
    }

    #[test]
    fn bulk_and_event_roundtrip() {
        let bulk = BulkMsg::Blob {
            kind: BulkKind::ChunkSnapshot,
            bytes: vec![1, 2, 3],
        };
        let bytes = postcard::to_allocvec(&bulk).expect("encode");
        assert_eq!(
            postcard::from_bytes::<BulkMsg>(&bytes).expect("decode"),
            bulk
        );

        let ev = EventMsg::EntityRemoved { entity: eid() };
        let bytes = postcard::to_allocvec(&ev).expect("encode");
        assert_eq!(
            postcard::from_bytes::<EventMsg>(&bytes).expect("decode"),
            ev
        );
    }

    #[test]
    fn sub_ids_are_ordered_and_never_conflated_with_entities() {
        assert!(SubId(1) < SubId(2));
        // Type-level separation: this is a compile-time property; the test documents it.
        let _sub: SubId = SubId(1);
        let _entity: EntityId = eid();
    }

    fn snap(n: u32) -> EntitySnap {
        EntitySnap {
            entity: EntityId::pack(EntityKind::Player, 1, u64::from(n), n),
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 1 },
                DVec3::new(f64::from(n), 0.0, 0.0),
                UniverseTick(10),
            ),
        }
    }

    #[test]
    fn partition_keeps_every_chunk_under_budget_and_loses_nothing() {
        let entities: Vec<EntitySnap> = (0..50).map(snap).collect();
        // A budget that fits only a handful of entities forces several chunks.
        let budget = 300;
        let chunks = partition_entities(&entities, budget);
        assert!(chunks.len() > 1, "a 50-entity world must partition");
        // Every chunk encodes (as a real SnapshotDatagram) within the budget...
        let mut seen = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(!chunk.is_empty(), "no empty chunk");
            let datagram = SnapshotDatagram {
                sub: SubId(0),
                frame_id: i as u64,
                source_tick: TickId(1),
                universe_tick: UniverseTick(10),
                entities: chunk.clone(),
            };
            let encoded = postcard::to_allocvec(&datagram).expect("encode").len();
            assert!(
                encoded <= budget,
                "chunk {i} encodes to {encoded} > budget {budget}"
            );
            seen.extend(chunk.iter().map(|s| s.entity));
        }
        // ...and the union of chunks is exactly the input (no loss, no duplication).
        let original: Vec<EntityId> = entities.iter().map(|s| s.entity).collect();
        assert_eq!(seen, original);
    }

    #[test]
    fn the_conservative_budget_leaves_room_for_a_header() {
        // A single entity always fits the conservative budget (the partitioner never
        // produces a chunk the transport must drop in normal operation).
        let one = partition_entities(&[snap(0)], CONSERVATIVE_DATAGRAM_BUDGET);
        assert_eq!(one.len(), 1);
        let encoded = postcard::to_allocvec(&SnapshotDatagram {
            sub: SubId(0),
            frame_id: 0,
            source_tick: TickId(0),
            universe_tick: UniverseTick(0),
            entities: one[0].clone(),
        })
        .expect("encode")
        .len();
        assert!(encoded < CONSERVATIVE_DATAGRAM_BUDGET);
    }

    #[test]
    fn partition_edge_cases() {
        // Empty input -> no chunks.
        assert_eq!(partition_entities(&[], 1000), Vec::<Vec<EntitySnap>>::new());
        // A roomy budget keeps everything in one chunk.
        let entities: Vec<EntitySnap> = (0..5).map(snap).collect();
        assert_eq!(partition_entities(&entities, 100_000).len(), 1);
        // A degenerate budget still ships one entity per chunk (the transport's
        // counted-drop guard catches a truly-impossible single oversize entity).
        let chunks = partition_entities(&entities, 1);
        assert_eq!(chunks.len(), 5);
        assert!(chunks.iter().all(|c| c.len() == 1));
    }
}
