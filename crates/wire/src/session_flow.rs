//! Gateway ↔ shard session-scoped traffic (the THIRD closed taxonomy, landing with
//! its first consumer per the incremental-freeze rule; `docs/design/
//! connection_plane.md` §M0, `identity_persistence.md` §input-carries-fence).
//!
//! Distinct from both client families (the gateway terminates those) and
//! `InterShardFlow` (the gateway is not a shard). Binding rules encoded here:
//!
//! - Routing is ALWAYS by in-frame `SessionId` + `Fence` — never by source address
//!   (R2). The shard drops input whose fence is below its highest-seen for that
//!   session (the stale-gateway drop branch exists from day one).
//! - The gateway NEVER decodes world state: snapshot payloads cross as opaque
//!   postcard bytes of [`crate::channels::SnapshotDatagram`]; the gateway's only
//!   touches are the fence compare (drop stale) and the byte-level `sub_id` re-tag
//!   ([`retag_snapshot_sub`]) — the 20 Hz hot path is a header rewrite, not a decode.
//! - Client input crosses verbatim: the gateway reads ONLY the leading `seq` varint
//!   ([`peek_input_seq`]) for dedup and forwards the original bytes unmodified.

use serde::{Deserialize, Serialize};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::{AccountId, EntityId, Fence, SessionId, TickId};

use crate::channels::RealmShape;

use crate::channels::SubId;
use crate::seams::directory::DirectoryKey;

/// Gateway → shard session control and input.
///
/// `PartialEq` but NOT `Eq`: `AttachSession` now carries a spawn pose, and a position is made of floats.
/// Nothing compares these for total equality — the tests that compare them want "are these the same
/// bytes on the wire", which is what `PartialEq` (and, where it matters, a postcard round-trip) answers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GatewayToShard {
    /// Attach a logged-in session: the shard spawns (or re-binds) the avatar and
    /// replies [`ShardToGateway::SessionAttached`]. Idempotent per (session, fence):
    /// a duplicate attach at the same fence re-sends the existing attachment.
    AttachSession {
        session: SessionId,
        /// The Session-key fence the gateway holds authority under.
        fence: Fence,
        account: AccountId,
        /// WHERE TO PUT THE AVATAR, already measured from the receiving realm's own centre and stamped
        /// with that realm's frame. `None` ⇒ the shard births it at its own origin, at rest.
        ///
        /// The DESCENT — stepping down the lineage subtracting each realm's placement in turn — is right,
        /// and each subtraction being one realm's own placement is right. WHO RUNS IT is a KNOWN OPEN
        /// BREACH, recorded here rather than defended: today the gateway runs the whole walk over its own
        /// copy of the forest, which is one party holding every parent's placement of every child. Only a
        /// parent may hold where its child sits, so each step belongs in the shard that authored it — the
        /// galaxy shard handing the system shard a point in the system's frame, the system shard handing
        /// the planet shard a point in the planet's frame, and the planet accepting it and computing
        /// nothing. What blocks the move is BOOTSTRAP ORDERING, not disagreement: at the instant the
        /// descent must answer, not one shard of the home lineage is running, because the demand that
        /// spins them up is what the same walk decides. `vd-connection-plane`'s
        /// `nothing_of_the_home_lineage_is_running_at_the_instant_the_login_descent_must_answer` asserts
        /// exactly that, so the blocker is a measurement and not an excuse. It closes when a home is
        /// stored as (lineage, pose in that realm's own frame) — the P7 durable per-realm store — or when
        /// the ambient root becomes permanently resident and the descent demands as it steps. That is an
        /// owner-visible call and is deliberately not made here.
        ///
        /// This is NOT the earlier defect, which was worse and is fixed: the shard held a copy of the same
        /// account→position map, in UNIVERSE-ROOT coordinates, and "converted" it by handing it to the
        /// frame machinery with an identity context — which changed the label and moved no number. A
        /// player stored three metres above a planet was planted three metres from the STAR. The receiving
        /// shard now checks the frame and REFUSES a pose that is not measured in its own, rather than
        /// wearing it, and that terminal behaviour is right and stays whoever ends up running the descent.
        spawn: Option<StampedPose>,
    },
    /// One client input datagram, forwarded VERBATIM (`input_bytes` is the postcard
    /// [`crate::channels::InputDatagram`] exactly as the client sent it).
    SessionInput {
        session: SessionId,
        fence: Fence,
        input_bytes: Vec<u8>,
    },
    /// Detach (logout/disconnect): the shard despawns the avatar.
    DetachSession { session: SessionId, fence: Fence },
    /// Open a STATE-FREE provisional input-landing slot on a TRANSFER DESTINATION shard so
    /// it can ACCEPT (not drop as `UnknownSession`) the `seq > marker_seq` input the gateway
    /// buffered during the cut and drains here at commit (integration.json #1: "dest applies
    /// gateway-buffered post-marker frames after commit"). The dest mints a provisional dot
    /// (NOT granted, NOT rendered, NO `SessionAttached` reply — the source still owns the
    /// client, R2) and seeds its input dedup watermark to `resume_from_seq` (= `marker_seq`)
    /// so the resume batch is non-vacuously deduped (a `seq <= marker` replay is rejected;
    /// `marker+1..` apply in order). Sent at `CommitAuthority` (`resume_from_seq = marker_seq`)
    /// — the gateway buffers locally during the cut, so the dest needs nothing until the
    /// commit drain. The slot's grant→owned-entity promotion (render, ghost, directory record)
    /// is 1d (D-27); in 1c.5 the slot is `input_active` but never granted.
    ///
    /// 1c.8: `subject` carries the transfer SUBJECT (forwarded VERBATIM from
    /// [`crate::seams::transfer_control::TransferControl::CommitAuthority`]). The dest ADOPTS
    /// the transferred avatar by extracting the `Entity` from it (its provisional dot's entity
    /// becomes the SUBJECT id, not a fresh mint, so the dest's directory adopt-grant lands on
    /// the record the CAS moved). A non-`Entity` subject is a counted no-op (no adopt) — the
    /// extraction NEVER panics. The dest learns `new_fence` from its own directory HeadRead
    /// (pull-through, the single source of truth), so it is NOT carried here.
    OpenInputSlot {
        session: SessionId,
        fence: Fence,
        account: AccountId,
        /// The dest treats `seq <= resume_from_seq` as already-applied (at the source).
        resume_from_seq: u64,
        /// The transfer subject the dest adopts (the `Entity` becomes the dot's id).
        subject: DirectoryKey,
    },
}

/// Shard → gateway session replies and world frames.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ShardToGateway {
    /// The avatar exists; `entity` was minted by THIS shard (`EntityId::pack` —
    /// never time-derived) and `realm_fence` is the realm-authority fence the
    /// shard stamps on every frame (the gateway's accepted fence).
    SessionAttached {
        session: SessionId,
        entity: EntityId,
        frame: FrameRef,
        realm_fence: Fence,
    },
    /// One world frame: `snapshot_bytes` is an opaque postcard
    /// [`crate::channels::SnapshotDatagram`]. The gateway compares `realm_fence`
    /// against its accepted fence (drops stale — fence rule 5), re-tags the sub id
    /// at byte level, and forwards. It never decodes the payload.
    Frame {
        realm_fence: Fence,
        source_tick: TickId,
        snapshot_bytes: Vec<u8>,
    },
    /// The avatar is gone (detach completed).
    SessionDetached { session: SessionId },
    /// A transfer-DESTINATION shard has ADOPTED the crossing entity and is ready to be
    /// READ by this session (Track R / 1d.2b). The gateway opens a SECOND per-session sub on
    /// the dest at `realm_fence` and re-points the avatar's render authority to it
    /// (`AuthorityChanged{entity, dest_sub}`) — the read-plane analog of the write-plane
    /// `CommitAuthority`. APPENDED variant (the only postcard-safe additive shape — postcard
    /// is non-self-describing): a prior arm's discriminant/framing is unchanged. Emitted by the
    /// dest at its grant-flip→`Adopted` (where `realm_fence`/`frame` are first legitimately
    /// owned), NOT a new `InterShardFlow` arm (HR1 — it rides this already-reviewed seam).
    /// `realm_fence` is the DEST realm fence (NOT the per-entity CAS `new_fence`).
    SubscriptionReady {
        session: SessionId,
        entity: EntityId,
        frame: FrameRef,
        realm_fence: Fence,
    },
    /// One REALM-placement frame (D-45(a) realm-unification FA-2c): `realm_snapshot_bytes` is an opaque
    /// postcard [`crate::channels::RealmSnapshotDatagram`] carrying the shard's authored placements for the
    /// moving REALMS it parents (an orbiting planet/station/ship box). The gateway forwards it to the shard's
    /// subscribers as a `MsgClass::RealmSnapshot` datagram (the render-plane twin of [`Frame`]); it never
    /// decodes the payload. A SEPARATE arm from [`Frame`] so a realm datagram is dispatched to the client's
    /// `RealmScene` consumer, not its entity-snapshot path. APPENDED variant (postcard-safe additive shape —
    /// a prior arm's discriminant/framing is unchanged). Emitted only when the shard parents >=1 moving child
    /// (EMPTY at walk/static scale ⇒ never sent).
    RealmFrame {
        realm_fence: Fence,
        source_tick: TickId,
        realm_snapshot_bytes: Vec<u8>,
    },
    /// The per-OBSERVER incremental render-scene update (VU AoI, proto_minor 6): the realms among THIS
    /// shard's own children that ENTERED (`added`) or LEFT (`removed`) `observer`'s AoI this tick, driven by
    /// the per-observer hysteresis (S0). `observer` is the DURABLE player id (survives re-home) — the gateway
    /// maps it to the live session and forwards a [`crate::channels::ServerControlMsg::RealmSceneDelta`] to
    /// that client, gated on its negotiated minor >= 6. UNLIKE [`Frame`]/[`RealmFrame`] (opaque forwarded
    /// bytes fanned to ALL subscribers), this is a TYPED, per-observer routed message the gateway decodes.
    /// APPENDED variant (postcard-safe additive shape — a prior arm's discriminant/framing is unchanged).
    /// Emitted only when a child crosses `observer`'s AoI edge (EMPTY sets are never sent).
    RealmSceneDelta {
        observer: AccountId,
        added: Vec<RealmShape>,
        removed: Vec<RealmId>,
    },
    /// THE REMOVE MESSAGE's mesh leg (proto_minor 14, D-4(a)): this shard PERMANENTLY stopped
    /// emitting `entity` (a leaver's retained ghost tore down at band-exit; a detach completed at
    /// the directory), so every client still subscribed here must evict its track — the reliable
    /// signal a pure-renderer client needs because absence from a datagram is deliberately never an
    /// eviction. The gateway fans it to the shard's subscribers as
    /// [`crate::channels::ServerControlMsg::Event`] (typed, decoded, per-session minor-gated), each
    /// checked against ITS per-shard accepted fence (rule 5 — a demoted old owner's stale removal
    /// is dropped and counted, like a stale frame). `at` is the shard's universe tick at removal —
    /// carried through to the client's resurrect guard. APPENDED variant (postcard-safe additive
    /// shape — a prior arm's discriminant/framing is unchanged).
    EntityRemoved {
        realm_fence: Fence,
        entity: EntityId,
        at: vd_core::UniverseTick,
    },
}

impl ShardToGateway {
    /// The opaque snapshot payload, when this is a `Frame` (test/tooling sugar —
    /// the gateway's hot path never owns the enum, only the bytes).
    #[must_use]
    pub fn into_snapshot_bytes(self) -> Option<Vec<u8>> {
        match self {
            ShardToGateway::Frame { snapshot_bytes, .. } => Some(snapshot_bytes),
            ShardToGateway::SessionAttached { .. }
            | ShardToGateway::SessionDetached { .. }
            | ShardToGateway::SubscriptionReady { .. }
            | ShardToGateway::RealmFrame { .. }
            | ShardToGateway::RealmSceneDelta { .. }
            | ShardToGateway::EntityRemoved { .. } => None,
        }
    }

    /// The opaque realm-placement payload, when this is a [`ShardToGateway::RealmFrame`] (the render-plane
    /// twin of [`into_snapshot_bytes`](Self::into_snapshot_bytes)) — test/tooling sugar.
    #[must_use]
    pub fn into_realm_snapshot_bytes(self) -> Option<Vec<u8>> {
        match self {
            ShardToGateway::RealmFrame {
                realm_snapshot_bytes,
                ..
            } => Some(realm_snapshot_bytes),
            ShardToGateway::Frame { .. }
            | ShardToGateway::SessionAttached { .. }
            | ShardToGateway::SessionDetached { .. }
            | ShardToGateway::SubscriptionReady { .. }
            | ShardToGateway::RealmSceneDelta { .. }
            | ShardToGateway::EntityRemoved { .. } => None,
        }
    }
}

/// Errors from the byte-level header operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HeaderError {
    #[error("buffer truncated inside the leading varint")]
    TruncatedVarint,
    #[error("varint exceeds the maximum width for its field")]
    VarintTooWide,
    #[error("buffer ended before the is_cut_marker byte")]
    Truncated,
    #[error("is_cut_marker byte is not a canonical bool (0x00/0x01)")]
    BadBool,
}

/// Read one unsigned LEB128 varint (postcard's integer encoding) from the front of
/// `bytes`. Returns (value, encoded_len). `max_bytes` bounds the field width
/// (10 for u64, 5 for u32).
fn read_varint(bytes: &[u8], max_bytes: usize) -> Result<(u64, usize), HeaderError> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    for (i, &b) in bytes.iter().enumerate() {
        if i >= max_bytes {
            return Err(HeaderError::VarintTooWide);
        }
        value |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return Ok((value, i + 1));
        }
        shift += 7;
    }
    Err(HeaderError::TruncatedVarint)
}

/// Write one unsigned LEB128 varint.
fn write_varint(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Read the `seq` of a postcard-encoded [`crate::channels::InputDatagram`] WITHOUT
/// decoding the rest — `seq: u64` is its first field. This is the gateway's input
/// hot path (dedup by seq); conformance with the real codec is property-tested.
///
/// # Errors
/// [`HeaderError`] if the buffer doesn't start with a valid u64 varint.
pub fn peek_input_seq(bytes: &[u8]) -> Result<u64, HeaderError> {
    read_varint(bytes, 10).map(|(v, _)| v)
}

/// Read `(seq, is_cut_marker)` of a postcard-encoded [`crate::channels::InputDatagram`]
/// WITHOUT decoding the rest: `seq: u64` (varint) then `is_cut_marker: bool` (one
/// `0x00`/`0x01` byte). The gateway's cut-marker observer peeks these two head fields
/// instead of full-decoding every input mid-transfer (D-24 SCALE-CUTDECODE-1).
///
/// CONTRACT: agreement with the real codec is on the `(seq, is_cut_marker)` PREFIX only;
/// the tail (`client_tick`/`movement`/`look`/`action_bits`) is deliberately NOT validated
/// — `on_cut_marker` never reads it. The bool byte IS validated (postcard rejects non-0/1),
/// so the peek is no more permissive than the decode it replaces on the fields it reads.
///
/// # Errors
/// [`HeaderError`] if the seq varint is malformed, the bool byte is missing
/// ([`HeaderError::Truncated`]), or the bool byte is not canonical ([`HeaderError::BadBool`]).
pub fn peek_is_cut_marker(bytes: &[u8]) -> Result<(u64, bool), HeaderError> {
    let (seq, used) = read_varint(bytes, 10)?;
    let &flag = bytes.get(used).ok_or(HeaderError::Truncated)?;
    let is_cut_marker = decode_cut_bool(flag)?;
    Ok((seq, is_cut_marker))
}

/// Decode the postcard bool byte that follows the seq varint (canonical `0x00`/`0x01`).
/// Monomorphic so [`peek_is_cut_marker`] stays a straight-line shim and these arms are
/// covered once over `u8` (the branchless-generic-shim discipline, HR5).
fn decode_cut_bool(byte: u8) -> Result<bool, HeaderError> {
    match byte {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(HeaderError::BadBool),
    }
}

/// Rewrite the leading `sub: SubId(u32)` of a postcard-encoded
/// [`crate::channels::SnapshotDatagram`] WITHOUT decoding the rest — `sub` is its
/// first field; everything after it is copied verbatim. This is the gateway's
/// snapshot hot path (re-tag + forward); conformance is property-tested.
///
/// # Errors
/// [`HeaderError`] if the buffer doesn't start with a valid u32 varint.
pub fn retag_snapshot_sub(bytes: &[u8], sub: SubId) -> Result<Vec<u8>, HeaderError> {
    let (_, old_len) = read_varint(bytes, 5)?;
    let tail = &bytes[old_len..];
    let mut out = Vec::with_capacity(5 + tail.len());
    write_varint(u64::from(sub.0), &mut out);
    out.extend_from_slice(tail);
    Ok(out)
}

/// Peek the `frame_id: u64` of a postcard-encoded [`crate::channels::SnapshotDatagram`] WITHOUT
/// decoding the rest — it is the SECOND field, right after `sub: SubId(u32)`. The gateway's
/// per-observer delivery watermark (1d.5a) reads this off each forwarded dest frame to advance the
/// `delivered` high-water (the body `sub` is irrelevant — the watermark keys on the gateway's own
/// per-session `entry.sub`). Conformance is property-tested against the real codec.
///
/// # Errors
/// [`HeaderError`] if the buffer is not a valid `sub` u32 varint followed by a `frame_id` u64 varint.
pub fn peek_snapshot_frame_id(bytes: &[u8]) -> Result<u64, HeaderError> {
    let (_, sub_len) = read_varint(bytes, 5)?; // skip the leading `sub: SubId(u32)`
    let (frame_id, _) = read_varint(&bytes[sub_len..], 10)?;
    Ok(frame_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::{EntitySnap, InputDatagram, SnapshotDatagram};
    use proptest::prelude::*;
    use vd_core::pose::StampedPose;
    use vd_core::{UniverseTick, glam::DVec3};

    fn input(seq: u64) -> InputDatagram {
        input_with(seq, false)
    }

    fn input_with(seq: u64, is_cut_marker: bool) -> InputDatagram {
        InputDatagram {
            seq,
            is_cut_marker,
            client_tick: TickId(3),
            movement: [0.5, 0.0, -1.0],
            look: [0.1, 0.2],
            action_bits: 7,
        }
    }

    fn snapshot(sub: SubId) -> SnapshotDatagram {
        SnapshotDatagram {
            sub,
            frame_id: 41,
            source_tick: TickId(9),
            universe_tick: UniverseTick(100),
            entities: vec![EntitySnap {
                entity: EntityId(7),
                pose: StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 1 },
                    DVec3::new(1.0, 2.0, 3.0),
                    UniverseTick(100),
                ),
            }],
        }
    }

    #[test]
    fn session_flow_roundtrips() {
        let g2s = vec![
            GatewayToShard::AttachSession {
                session: SessionId(1),
                fence: Fence(2),
                account: AccountId(3),
                // The spawn pose is measured in the RECEIVING realm's frame, never the root's — a pose
                // wearing `SystemSpace{0}` here would be the universe-absolute the shard used to be
                // handed and silently relabel. Round-tripping a `Some` proves the field survives.
                spawn: Some(StampedPose::at_rest(
                    FrameRef::PlanetCentered { planet_seed: 7 },
                    DVec3::new(3.0, 0.0, 0.0),
                    UniverseTick(0),
                )),
            },
            // The `None` arm — a login with nothing stored — must round-trip too (it is the byte-identical
            // default every rig takes).
            GatewayToShard::AttachSession {
                session: SessionId(1),
                fence: Fence(2),
                account: AccountId(3),
                spawn: None,
            },
            GatewayToShard::SessionInput {
                session: SessionId(1),
                fence: Fence(2),
                input_bytes: postcard::to_allocvec(&input(5)).expect("encode input"),
            },
            GatewayToShard::DetachSession {
                session: SessionId(1),
                fence: Fence(2),
            },
            GatewayToShard::OpenInputSlot {
                session: SessionId(1),
                fence: Fence(2),
                account: AccountId(3),
                resume_from_seq: 42,
                subject: DirectoryKey::Entity(EntityId(9)),
            },
        ];
        for msg in g2s {
            let bytes = postcard::to_allocvec(&msg).expect("encode");
            assert_eq!(
                postcard::from_bytes::<GatewayToShard>(&bytes).expect("decode"),
                msg
            );
        }
        let s2g = vec![
            ShardToGateway::SessionAttached {
                session: SessionId(1),
                entity: EntityId(9),
                frame: FrameRef::SystemSpace { system_seed: 4 },
                realm_fence: Fence(1),
            },
            ShardToGateway::Frame {
                realm_fence: Fence(1),
                source_tick: TickId(8),
                snapshot_bytes: postcard::to_allocvec(&snapshot(SubId(0))).expect("encode snap"),
            },
            ShardToGateway::SessionDetached {
                session: SessionId(1),
            },
            ShardToGateway::SubscriptionReady {
                session: SessionId(1),
                entity: EntityId(9),
                frame: FrameRef::SystemSpace { system_seed: 8 },
                realm_fence: Fence(2),
            },
            ShardToGateway::RealmFrame {
                realm_fence: Fence(1),
                source_tick: TickId(8),
                realm_snapshot_bytes: vec![7, 8, 9],
            },
        ];
        for msg in s2g {
            let bytes = postcard::to_allocvec(&msg).expect("encode");
            assert_eq!(
                postcard::from_bytes::<ShardToGateway>(&bytes).expect("decode"),
                msg
            );
        }
    }

    #[test]
    fn into_snapshot_bytes_extracts_frames_only() {
        let frame = ShardToGateway::Frame {
            realm_fence: Fence(1),
            source_tick: TickId(2),
            snapshot_bytes: vec![1, 2, 3],
        };
        assert_eq!(frame.into_snapshot_bytes(), Some(vec![1, 2, 3]));
        let attached = ShardToGateway::SessionAttached {
            session: SessionId(1),
            entity: EntityId(2),
            frame: FrameRef::SystemSpace { system_seed: 3 },
            realm_fence: Fence(1),
        };
        assert_eq!(attached.into_snapshot_bytes(), None);
        let detached = ShardToGateway::SessionDetached {
            session: SessionId(1),
        };
        assert_eq!(detached.into_snapshot_bytes(), None);
        let ready = ShardToGateway::SubscriptionReady {
            session: SessionId(1),
            entity: EntityId(2),
            frame: FrameRef::SystemSpace { system_seed: 3 },
            realm_fence: Fence(2),
        };
        assert_eq!(ready.into_snapshot_bytes(), None);
        // A RealmFrame is NOT an entity Frame — `into_snapshot_bytes` must not extract it.
        let realm_frame = ShardToGateway::RealmFrame {
            realm_fence: Fence(1),
            source_tick: TickId(2),
            realm_snapshot_bytes: vec![7, 8, 9],
        };
        assert_eq!(realm_frame.into_snapshot_bytes(), None);
        // A RealmSceneDelta is a typed per-observer message, never an opaque frame.
        let scene_delta = ShardToGateway::RealmSceneDelta {
            observer: AccountId(5),
            added: Vec::new(),
            removed: Vec::new(),
        };
        assert_eq!(scene_delta.into_snapshot_bytes(), None);
    }

    #[test]
    fn into_realm_snapshot_bytes_extracts_realm_frames_only() {
        // FA-2c: the render-plane twin of `into_snapshot_bytes` — only a RealmFrame yields its payload.
        let realm_frame = ShardToGateway::RealmFrame {
            realm_fence: Fence(1),
            source_tick: TickId(2),
            realm_snapshot_bytes: vec![4, 5, 6],
        };
        assert_eq!(realm_frame.into_realm_snapshot_bytes(), Some(vec![4, 5, 6]));
        // Every other arm (including the entity Frame) yields None.
        let entity_frame = ShardToGateway::Frame {
            realm_fence: Fence(1),
            source_tick: TickId(2),
            snapshot_bytes: vec![1, 2, 3],
        };
        assert_eq!(entity_frame.into_realm_snapshot_bytes(), None);
        let attached = ShardToGateway::SessionAttached {
            session: SessionId(1),
            entity: EntityId(2),
            frame: FrameRef::SystemSpace { system_seed: 3 },
            realm_fence: Fence(1),
        };
        assert_eq!(attached.into_realm_snapshot_bytes(), None);
        let detached = ShardToGateway::SessionDetached {
            session: SessionId(1),
        };
        assert_eq!(detached.into_realm_snapshot_bytes(), None);
        let ready = ShardToGateway::SubscriptionReady {
            session: SessionId(1),
            entity: EntityId(2),
            frame: FrameRef::SystemSpace { system_seed: 3 },
            realm_fence: Fence(2),
        };
        assert_eq!(ready.into_realm_snapshot_bytes(), None);
        let scene_delta = ShardToGateway::RealmSceneDelta {
            observer: AccountId(5),
            added: Vec::new(),
            removed: Vec::new(),
        };
        assert_eq!(scene_delta.into_realm_snapshot_bytes(), None);
    }

    #[test]
    fn realm_scene_delta_shard_to_gateway_round_trips() {
        use vd_core::geometry::Boundary;
        use vd_core::glam::DVec3;
        use vd_core::pose::LatticePos;
        // VU AoI (proto_minor 6): a per-observer delta — one realm ENTERED the observer's AoI (its shape),
        // one LEFT (its id) — round-trips, and is NOT an opaque forwarded frame (both extractors decline it).
        let delta = ShardToGateway::RealmSceneDelta {
            observer: AccountId(5),
            added: vec![RealmShape {
                realm: RealmId::Planet(7),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                center: LatticePos::local(DVec3::new(20.0, 0.0, 0.0)),
                shape: Boundary::Shell { r: 10.0 },
                parent: Some(RealmId::System(7)),
            }],
            removed: vec![RealmId::Planet(8)],
        };
        let bytes = postcard::to_allocvec(&delta).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ShardToGateway>(&bytes).expect("decode"),
            delta
        );
        assert_eq!(delta.clone().into_snapshot_bytes(), None);
        assert_eq!(delta.into_realm_snapshot_bytes(), None);
    }

    #[test]
    fn peek_matches_the_real_codec_on_boundary_values() {
        for seq in [0u64, 1, 127, 128, 16_383, 16_384, u64::MAX] {
            let bytes = postcard::to_allocvec(&input(seq)).expect("encode");
            assert_eq!(peek_input_seq(&bytes), Ok(seq));
        }
    }

    #[test]
    fn retag_matches_the_real_codec_on_boundary_values() {
        for (old, new) in [(0u32, 1u32), (127, 128), (16_384, 0), (u32::MAX, 5)] {
            let bytes = postcard::to_allocvec(&snapshot(SubId(old))).expect("encode");
            let retagged = retag_snapshot_sub(&bytes, SubId(new)).expect("retag");
            let decoded: SnapshotDatagram = postcard::from_bytes(&retagged).expect("decode");
            let mut expected = snapshot(SubId(old));
            expected.sub = SubId(new);
            assert_eq!(decoded, expected, "only the sub changed");
        }
    }

    #[test]
    fn header_errors_are_typed() {
        // Truncated: a continuation bit with nothing after it.
        assert_eq!(
            peek_input_seq(&[0x80]),
            Err(HeaderError::TruncatedVarint),
            "{}",
            HeaderError::TruncatedVarint
        );
        assert_eq!(peek_input_seq(&[]), Err(HeaderError::TruncatedVarint));
        // Too wide: 11 continuation bytes exceed a u64's 10-byte maximum.
        assert_eq!(
            peek_input_seq(&[0x80; 11]),
            Err(HeaderError::VarintTooWide),
            "{}",
            HeaderError::VarintTooWide
        );
        // A u32 sub field is at most 5 bytes wide.
        assert_eq!(
            retag_snapshot_sub(&[0x80; 6], SubId(0)),
            Err(HeaderError::VarintTooWide)
        );
        assert_eq!(
            retag_snapshot_sub(&[], SubId(0)),
            Err(HeaderError::TruncatedVarint)
        );
        // peek_snapshot_frame_id: empty, a sub then a truncated frame_id varint, and a too-wide sub.
        assert_eq!(
            peek_snapshot_frame_id(&[]),
            Err(HeaderError::TruncatedVarint)
        );
        assert_eq!(
            peek_snapshot_frame_id(&[0x01, 0x80]),
            Err(HeaderError::TruncatedVarint)
        );
        assert_eq!(
            peek_snapshot_frame_id(&[0x80; 6]),
            Err(HeaderError::VarintTooWide)
        );
    }

    #[test]
    fn peek_is_cut_marker_reads_seq_and_canonical_bool() {
        // W2: seq varint then a canonical bool byte (0x05 = seq 5).
        assert_eq!(peek_is_cut_marker(&[0x05, 0x00]), Ok((5, false)));
        assert_eq!(peek_is_cut_marker(&[0x05, 0x01]), Ok((5, true)));
    }

    #[test]
    fn peek_is_cut_marker_typed_errors() {
        // W3: malformed at the seq stage (reuses read_varint's errors).
        assert_eq!(peek_is_cut_marker(&[]), Err(HeaderError::TruncatedVarint));
        assert_eq!(
            peek_is_cut_marker(&[0x80]),
            Err(HeaderError::TruncatedVarint)
        );
        assert_eq!(
            peek_is_cut_marker(&[0x80; 11]),
            Err(HeaderError::VarintTooWide)
        );
        // W4: the seq is a valid varint but the bool byte is MISSING. `.get(used)` (not
        // indexing) is what keeps this from panicking — including when a multi-byte varint
        // consumes the whole buffer (used == len).
        assert_eq!(peek_is_cut_marker(&[0x05]), Err(HeaderError::Truncated));
        assert_eq!(
            peek_is_cut_marker(&[0x80, 0x01]),
            Err(HeaderError::Truncated)
        );
        // W5: the bool byte is present but non-canonical. This is the ONLY thing covering
        // the BadBool arm — the via-codec proptest can never emit a 0x02. Document that the
        // real codec rejects it too, so the peek is no more permissive than the decode.
        assert_eq!(peek_is_cut_marker(&[0x05, 0x02]), Err(HeaderError::BadBool));
        assert_eq!(peek_is_cut_marker(&[0x00, 0xFF]), Err(HeaderError::BadBool));
        assert!(postcard::from_bytes::<InputDatagram>(&[0x00, 0xFF, 0, 0, 0, 0, 0, 0, 0]).is_err());
    }

    proptest! {
        /// The byte-level peek agrees with the real codec for EVERY seq.
        #[test]
        fn peek_agrees_with_postcard(seq in any::<u64>()) {
            let bytes = postcard::to_allocvec(&input(seq)).expect("encode");
            prop_assert_eq!(peek_input_seq(&bytes), Ok(seq));
        }

        /// The cut-marker peek agrees with the real codec on the (seq, is_cut_marker)
        /// PREFIX for EVERY seq and BOTH flag values (two-sided conformance, both bool arms).
        #[test]
        fn peek_is_cut_marker_agrees_with_postcard(seq in any::<u64>(), flag in any::<bool>()) {
            let bytes = postcard::to_allocvec(&input_with(seq, flag)).expect("encode");
            prop_assert_eq!(peek_is_cut_marker(&bytes), Ok((seq, flag)));
        }

        /// The byte-level re-tag agrees with decode-modify-encode for EVERY pair of
        /// sub ids and arbitrary frame ids (the tail is preserved bit-exactly).
        #[test]
        fn retag_agrees_with_postcard(old in any::<u32>(), new in any::<u32>(), frame_id in any::<u64>()) {
            let mut snap = snapshot(SubId(old));
            snap.frame_id = frame_id;
            let bytes = postcard::to_allocvec(&snap).expect("encode");
            let retagged = retag_snapshot_sub(&bytes, SubId(new)).expect("retag");
            snap.sub = SubId(new);
            let reference = postcard::to_allocvec(&snap).expect("encode reference");
            prop_assert_eq!(retagged, reference, "byte-identical to a full re-encode");
        }

        /// The frame-id peek reads the SAME `frame_id` a full decode would, for arbitrary `sub` +
        /// `frame_id` (the 1d.5a watermark trusts the wire layout, never the body decode).
        #[test]
        fn peek_frame_id_agrees_with_postcard(sub in any::<u32>(), frame_id in any::<u64>()) {
            let mut snap = snapshot(SubId(sub));
            snap.frame_id = frame_id;
            let bytes = postcard::to_allocvec(&snap).expect("encode");
            prop_assert_eq!(peek_snapshot_frame_id(&bytes).expect("peek"), frame_id);
        }
    }
}
