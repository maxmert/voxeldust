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

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::entity_kind::EntityKind;
    use vd_core::glam::DVec3;

    fn eid() -> EntityId {
        EntityId::pack(EntityKind::Player, 1, 1, 1)
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
        ];
        for msg in msgs {
            let bytes = postcard::to_allocvec(&msg).expect("encode");
            let back: ServerControlMsg = postcard::from_bytes(&bytes).expect("decode");
            assert_eq!(back, msg);
        }
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
}
