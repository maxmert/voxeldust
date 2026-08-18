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

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use vd_core::frame::FramePlacement;
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId, TickId, UniverseTick};

use crate::channels::RealmShape;

use crate::channels::{RealmSnap, SubId};
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
    /// THE WINDOW LANE's subscription open (gateway → shard, mesh minor 16; SL6 ask APPROVED —
    /// owner 2026-08-15/16, `docs/design/window_lane.md` §1.1/§2.3/§4.5): "serve the picture for
    /// `scope`" on the window id the gateway minted. Carries NO account and NO pose — an
    /// [`WindowScope::Occupants`]/[`WindowScope::Child`] scope is information-equivalent to the
    /// SL7 occupancy bit that already crosses (SL2 intact). RELIABLE control with DERIVED
    /// keep-alive semantics: the subscriber re-asserts it on a derived cadence, and the shard
    /// drops a window not refreshed within the derived TTL of **2 beats + 1** (owner law 3(a),
    /// `docs/design/owner_decisions_2026-08-15.md` item 3: retention is FOREVER DERIVED — at
    /// least two cadences plus one, never a free literal), so a dead subscriber can never leak a
    /// fan. Idempotent per `window` (a re-open refreshes the TTL). APPENDED variant (postcard-safe
    /// additive shape — a prior arm's discriminant/framing is unchanged). LIVE since Slice A: the
    /// gateway derives + opens the MINIMAL per-session chain windows (own-realm `Occupants`, and
    /// `Child(c)` on the parent where its own routing state already names it — the FULL chain
    /// derivation is Slice B), and the shard-side registry consumes it (`vd-sim` `OpenWindows`).
    WindowOpen {
        window: WindowId,
        scope: WindowScope,
    },
    /// THE WINDOW LANE's subscription close (gateway → shard, mesh minor 16; same SL6 approval as
    /// [`GatewayToShard::WindowOpen`]). RELIABLE control, idempotent (closing an unknown window is
    /// a counted no-op); the derived keep-alive TTL above is the crash backstop — this message is
    /// only the polite fast path. APPENDED variant (postcard-safe additive shape). LIVE since
    /// Slice A (the gateway closes windows that stop being derivable — structurally, zero
    /// sessions ⇒ zero windows; the shard registry consumes it).
    WindowClose { window: WindowId },
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
    /// ★TOMBSTONE (window lane Slice C2, minor 19; owner-approved 2026-08-16 —
    /// docs/design/window_lane.md §5 RULINGS) — THE OLD REALM DATAGRAM is DELETED. It carried an
    /// opaque [`crate::channels::RealmSnapshotDatagram`] of a shard's authored child placements,
    /// which the gateway fanned to that shard's subscribers untouched. Since the minor-18 flag day
    /// the client's ONE scene author has been the gateway's composed feed, so this lane's last
    /// consumer was the Slice-B parity comparator; with the comparator's subject (the inter-realm
    /// cascade) deleted, the measurement is discharged and the lane is deleted with it — never
    /// disabled (§4.5 Topic 5). Its successor is [`ShardToGateway::WindowFrame`] on an
    /// [`WindowScope::Occupants`] window, which `window_lane.md` §2.9 step 3 names as subsuming
    /// this emit: TYPED rows, one universe stamp, attested per sender. The variant REMAINS because
    /// postcard discriminants are positional and may never be renumbered; discriminant 4 is
    /// reserved forever; nothing produces it, and a received frame is counted, never served.
    /// Do not revive.
    RealmFrame {
        realm_fence: Fence,
        source_tick: TickId,
        realm_snapshot_bytes: Vec<u8>,
    },
    /// ★TOMBSTONE (window lane Slice C2, minor 19; owner-approved 2026-08-16 —
    /// docs/design/window_lane.md §5 RULINGS) — THE PER-OBSERVER SHAPE PUSH is DELETED. It shipped
    /// the OUTLINES entering and leaving one observer's band, which made a parent the author of its
    /// children's LOOK (SL3's violation, D-LANE-4). `window_lane.md` §2.9 shrinks this emit to the
    /// ids-only membership verdict: [`ShardToGateway::WindowMembership`] states WHICH children the
    /// parent's SL7 fold admits, and each realm states its OWN look on
    /// [`ShardToGateway::WindowBody`]. The variant REMAINS because postcard discriminants are
    /// positional and may never be renumbered; its discriminant is reserved forever; nothing
    /// produces it, and a received frame is counted, never served. Do not revive.
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
    /// THE WINDOW LANE's per-tick unit of one window level (shard → gateway, mesh minor 16;
    /// SL6 ask APPROVED — owner 2026-08-15/16, `docs/design/window_lane.md` §1.1/§2.2/§4.5): the
    /// stating realm's FULL direct-child placement roster (dormant children included — markers are
    /// never membership-filtered) plus, for a [`WindowScope::Child`] window, ONE pre-inverted hop
    /// row. ONE message per tick per open window, everything inside stamped at the one `at` —
    /// intra-level same-tickness is by construction, never by matching. Cadence: the realm-lane
    /// tick (20 Hz), stamped off the follower clock. Classification: **FireAndForget /
    /// Unreliable**, full-state latest-wins per tick (owner law 3(b),
    /// `docs/design/owner_decisions_2026-08-15.md` item 3: only FULL STATE rides a drop lane — a
    /// delta here would be a defect, not a tuning choice); a lost frame self-heals next tick.
    /// Attestation (fail-closed, measured now / refused at cloud mTLS — owner item 11 pattern):
    /// the receiver drops + counts any frame whose sender node is not the `ShardRoster` head for
    /// the stating realm ([`window_sender_is_head`]), and any stale `realm_fence` (zombie guard).
    /// APPENDED variant (postcard-safe additive shape). Since window lane Slice C2 this is the
    /// ONLY per-tick placement statement that leaves a realm at all: the shard emits one frame
    /// per tick per open window (its authored rows, built once), and the gateway ATTESTS
    /// fail-closed (unknown window / forged sender dropped + counted) before INGESTING it into
    /// the composer (`window_rows_ingested`) that serves the client's one scene feed.
    WindowFrame {
        realm_fence: Fence,
        /// The subscription id the receiving side minted at [`GatewayToShard::WindowOpen`].
        window: WindowId,
        /// The ONE universe-tick stamp for everything inside this frame.
        at: UniverseTick,
        /// `Some` on a [`WindowScope::Child`] window (the author's own body expressed in that
        /// child's frame at `at`); `None` on the observer's-own-level [`WindowScope::Occupants`]
        /// window (an occupant already stands in the author's own frame — there is no hop).
        /// Boxed for enum-size hygiene ONLY (a full rigid placement is ~150 B and would balloon
        /// every holder of this enum): a `Box` is serde-transparent, so the WIRE SHAPE is exactly
        /// the design's `Option<HopRow>` — byte-identical either way.
        hop: Option<Box<HopRow>>,
        /// The TYPED authored child rows, in the sender's own frame (no nested
        /// serialize-inside-serialize — the judge fix; same row type as the realm-lane feed).
        rows: Vec<RealmSnap>,
    },
    /// THE WINDOW LANE's look/marker lane (shard → gateway, mesh minor 16; same SL6 approval as
    /// [`ShardToGateway::WindowFrame`]): one body statement about `subject` — the two nested
    /// [`BodyStmt`] kinds are structurally exclusive (a look CANNOT carry a position, a marker
    /// CANNOT carry a look; the third pixel source is unrepresentable in the types). Cadence:
    /// send-on-change + on-open, NEVER per-tick. Classification: **ReDriven, reliable** (rides
    /// the session-reply lane) — mirroring `InterShardFlow::ShardRoster`'s reasoning: a lost look
    /// is an invisible realm at exactly the no-flicker moment, so it is never Unreliable; and its
    /// sender re-drives it from live state, so it needs no durable outbox (not producer-less).
    /// Attestation (fail-closed): sender must be the roster head for its own realm, AND the
    /// subject rule of [`window_body_admissible`] holds — `SelfLook` only about the sender's own
    /// realm (SL3: a realm draws itself), `Marker` only about the sender's DIRECT children (the
    /// owner-ruled photometric datum for a sleeping child, R4; superseded by data presence the
    /// moment the child states its own look). A mis-authored body is dropped + counted, never
    /// patched. APPENDED variant (postcard-safe additive shape). The emitter is LIVE since
    /// Slice A (send-on-change + on-open; the look from the realm's OWN boot extent, the marker
    /// bags from the boot roster's photometrics); the receiving engine is Slice B.
    WindowBody {
        realm_fence: Fence,
        window: WindowId,
        /// The realm this statement is ABOUT (the sender itself, or one of its direct children).
        subject: RealmId,
        stmt: BodyStmt,
        /// When the author stated it (send-on-change: NOT a per-tick stamp; the newest wins).
        authored_at: UniverseTick,
    },
    /// THE WINDOW LANE's membership verdict (shard → gateway, mesh minor 16; same SL6 approval):
    /// the parent's OWN SL7 band/hysteresis decision — which of its direct children are inside
    /// the interest band of an occupant it holds (or of an occupied child standing proxy) —
    /// shipped as ids only, so the receiver NEVER re-derives AoI. Scope: membership gates BODIES
    /// and live-child INTERIOR windows only; marker/placement rows always ship (the full roster —
    /// stars stay in the sky by construction). Cadence: the AoI cadence (the same
    /// `aoi_recheck_cadence` the SL7 bit beats on). Classification: **ReDriven, reliable** — a
    /// lost delta would desynchronize the drawn set until the next edge, so it rides the
    /// session-reply lane like [`ShardToGateway::WindowBody`]. Attestation (fail-closed): sender
    /// must be the roster head for the stating realm ([`window_sender_is_head`]). APPENDED
    /// variant (postcard-safe additive shape). The emitter is LIVE since Slice A (the verdict
    /// diffed per window out of the ONE existing `aoi_decide` fold — per-dot for `Occupants`,
    /// the occupied-child proxy fold for `Child` scopes); the receiving engine is Slice B.
    WindowMembership {
        window: WindowId,
        added: Vec<RealmId>,
        removed: Vec<RealmId>,
    },
    /// THE Q2 PARENT-RELAY's forward leg (mesh minor 17; owner-approved 2026-08-16 —
    /// `docs/design/owner_decisions_2026-08-15.md` addendum + `docs/design/window_lane.md` §5
    /// RULINGS): a live CHILD's VERBATIM self-authored window statements, relayed one hop through
    /// its parent to a subscriber holding a window ON THE PARENT. The parent's only lawful acts
    /// are FORWARD or DROP: `statements` is the SEALED byte blob it received on
    /// [`crate::intershard::InterShardFlow::WindowRelay`], copied here unopened (no store-merge,
    /// no re-state, no read — the parent-side holder keeps bytes, never values), exactly the
    /// sealed-payload discipline the realm cascade established. The receiver alone decodes it
    /// ([`open_relay_statements`]) and admits every inner statement against the CHILD's identity
    /// with the SAME predicates the direct lanes use: [`window_body_admissible`] with the child
    /// as the stating realm, the child vouched by the window author's own attested roster, and
    /// `child_fence` (the child's own fence at authoring, INTACT end-to-end) as the zombie guard.
    /// This is how a waking realm's self-look reaches an OUTSIDE observer before the observer
    /// enters it (the §2.8 marker⇒look handover) while "am I observed from outside" stays
    /// unrepresentable in every realm (Q2's rationale). Classification mirrors
    /// [`ShardToGateway::WindowBody`]: **ReDriven, reliable** (the child re-drives from live
    /// state; a lost relay is an invisible realm at exactly the no-flicker moment G-HANDOVER
    /// measures). APPENDED variant (postcard-safe additive shape).
    WindowRelayed {
        /// The FORWARDING parent's realm fence (the window's author — the hop the relay rode);
        /// the receiver's usual stale-fence zombie guard for the forwarder itself.
        realm_fence: Fence,
        window: WindowId,
        /// The stating CHILD realm, copied from the relay envelope as-received.
        child: RealmId,
        /// The CHILD's own realm fence at authoring — forwarded INTACT (never the parent's).
        child_fence: Fence,
        /// The child's OWN sealed statements, byte-for-byte as received (postcard
        /// `Vec<RelayedStatement>`).
        statements: Vec<u8>,
        /// THE SEALED INTERIOR FORWARD (mesh minor 20, APPENDED; owner-approved 2026-08-17 —
        /// docs/design/look_horizon.md RULINGS + §2 ASK A): the child's own held GRANDCHILD
        /// batches, forwarded sealed and unopened exactly as the child shipped them
        /// ([`crate::intershard::InteriorRelay`] — each carries the grandchild's own fence
        /// outside its seal). The gateway is the first and only opener: it vouches each named
        /// grandchild against the child's own attested roster, orders its fence, and admits
        /// ONLY the author's own picture (a `SelfLook` about itself) — the batch's `Level` and
        /// markers describe depth-3 subjects no row can exist for, dropped + counted as the
        /// lawful filter (`window_relay_interior_filtered`, expected NON-zero), while an
        /// unrostered grandchild is a violation (`window_relay_interior_unvouched`, asserted 0
        /// on a lawful flight).
        interior: Vec<crate::intershard::InteriorRelay>,
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
            | ShardToGateway::EntityRemoved { .. }
            | ShardToGateway::WindowFrame { .. }
            | ShardToGateway::WindowBody { .. }
            | ShardToGateway::WindowMembership { .. }
            | ShardToGateway::WindowRelayed { .. } => None,
        }
    }

    /// The opaque realm-placement payload, when this is a [`ShardToGateway::RealmFrame`] (the render-plane
    /// twin of [`into_snapshot_bytes`](Self::into_snapshot_bytes)) — test/tooling sugar for a
    /// ★TOMBSTONED arm (window lane Slice C2, minor 19): nothing produces a `RealmFrame` any more,
    /// and this exists so the reserved discriminant keeps a readable shape.
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
            | ShardToGateway::EntityRemoved { .. }
            | ShardToGateway::WindowFrame { .. }
            | ShardToGateway::WindowBody { .. }
            | ShardToGateway::WindowMembership { .. }
            | ShardToGateway::WindowRelayed { .. } => None,
        }
    }
}

// ===== THE WINDOW LANE's types (mesh minor 16; SL6 ask APPROVED — owner 2026-08-15/16,
// `docs/design/window_lane.md` §1.1/§2.2/§2.3/§4.5; rulings record:
// `docs/design/owner_decisions_2026-08-15.md`, 2026-08-16 addendum) ==========================

/// A window subscription id, minted by the SUBSCRIBING side at [`GatewayToShard::WindowOpen`]
/// (monotone per subscriber, never reused — the [`crate::channels::SubId`] discipline): every
/// window-lane row names the subscription it answers, so a straggler from a closed window is
/// dropped by id mismatch, never guessed at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WindowId(pub u64);

/// What a [`GatewayToShard::WindowOpen`] asks the shard to serve (`docs/design/window_lane.md`
/// §2.3 — the typed scope that dissolved the `child: None` overload, judge hole H7).
///
/// **There is deliberately NO `Observed` variant** (owner Q2 ruling, 2026-08-16 — see
/// `docs/design/window_lane.md` §5 RULINGS and DEFERRED.md D-WINDOW-2): watching a live realm
/// from OUTSIDE rides the PARENT RELAY — the parent forwards its live children's self-authored
/// statements verbatim (fence + attestation intact; no store, no merge, no read) — so "am I
/// observed from outside" stays UNREPRESENTABLE in every realm. The direct window is the
/// ledgered per-realm upgrade taken ONLY on a measured G-HANDOVER failure, via a fresh SL6 ask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowScope {
    /// Serve the picture for MY OWN occupants (the observer's-own-level window; frames carry
    /// `hop: None`). Information-equivalent to the occupancy bit that already crosses (SL2).
    Occupants,
    /// Serve the picture for the occupants under my DIRECT child (frames carry the hop row for
    /// that child). Same occupancy-bit equivalence as `Occupants` — the child realm itself is
    /// told nothing.
    Child(RealmId),
}

/// ONE hop of the observer chain (`docs/design/window_lane.md` §2.2, R1): the AUTHOR's own frame
/// expressed in `child`'s frame at the enclosing [`ShardToGateway::WindowFrame::at`] — a full
/// rigid transform, PRE-INVERTED by the author, who authors that child's placement (SL1's
/// "conversion in the parent" held hop-by-hop; the one inversion happens at the author, nowhere
/// downstream). INV-BODY-AT-ORIGIN (named invariant, pinned in Slice A): a realm's own body sits
/// at its own frame origin, so `inv`'s origin IS "the author's body in the child's frame". This
/// type exists ONLY on the shard→gateway leg — no `InterShardFlow` arm carries it, so the type
/// system keeps a reversed placement out of every realm (a realm can never hear where it is).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HopRow {
    /// The direct child whose frame `inv` is expressed in.
    pub child: RealmId,
    /// "My frame expressed in the child's frame at `at`" — pre-inverted by the author. A rotated
    /// cross-cell inversion inherits `transfer_frame`'s refusal semantics (dropped + counted;
    /// owed with P10 cell math — the measurement pin lands with Slice A, never argued).
    pub inv: FramePlacement,
}

/// A body statement's two STRUCTURALLY EXCLUSIVE kinds (`docs/design/window_lane.md` §2.2 — the
/// type graft that makes a third pixel source unrepresentable): a look cannot carry a position
/// (no such field EXISTS), a marker cannot carry a look. Body selection downstream is a presence
/// gate — self-look if one was received (only a running realm can ship one), else the parent's
/// marker — THE DRAW LAW by absence of data, never an if-running flag.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BodyStmt {
    /// The subject's OWN look: outline + display tags as a canonical [`vd_core::tlv`] blob
    /// (TAG_LOOK; tags grow additively forever, unknown tags are skipped). Legal ONLY when the
    /// subject IS the sender's own realm — a realm states a look about ITSELF alone
    /// (SL3/R7/D-LANE-4; enforced by [`window_body_admissible`], fail-closed).
    SelfLook { bag: Vec<u8> },
    /// The parent-authored point-of-light datum for a SLEEPING direct child: photometric scalars
    /// as a canonical [`vd_core::tlv`] blob (TAG_LUMA), drawn from the same seed stream that
    /// generated the child (`vd-physics` worldgen's per-system draw, pinned f(seed) values).
    /// Legal ONLY when the subject is one of the sender's DIRECT children (the owner-ruled R4
    /// bend of SL3, named in the ask; superseded by data presence the instant the child states
    /// its own look). NO look field exists here — a marker can never carry an outline.
    Marker { luma: Vec<u8> },
}

/// THE WINDOW-LANE ADMISSION RULE for placement frames and membership verdicts
/// (`docs/design/window_lane.md` §2.2, fail-closed): the sender node must BE the `ShardRoster`
/// head for the stating realm. Pure — the resolved head comes in as an argument (no I/O, no
/// directory read here); `None` (no head resolved) refuses, because fail-closed means an
/// unattestable row is dropped + counted, never served on faith (owner item 11 pattern:
/// measured now, refused at cloud mTLS).
#[must_use]
pub fn window_sender_is_head(sender: NodeId, roster_head: Option<NodeId>) -> bool {
    roster_head == Some(sender)
}

/// THE WINDOW-LANE ADMISSION RULE for body statements (`docs/design/window_lane.md` §2.2,
/// fail-closed; applied ON TOP of [`window_sender_is_head`] for the sender's own realm): the two
/// authorships are structurally exclusive — [`BodyStmt::SelfLook`] is legal iff the subject IS
/// the sender's own realm (a realm draws itself, SL3), [`BodyStmt::Marker`] is legal iff the
/// subject is one of the sender's DIRECT children (the roster check). Pure — the sender's realm
/// and its direct-child set come in as arguments (no I/O). A `false` is a mis-authored body:
/// dropped + counted (`window_misauthored_body`), never patched.
#[must_use]
pub fn window_body_admissible(
    stmt: &BodyStmt,
    subject: RealmId,
    sender_realm: RealmId,
    sender_children: &BTreeSet<RealmId>,
) -> bool {
    match stmt {
        BodyStmt::SelfLook { .. } => subject == sender_realm,
        BodyStmt::Marker { .. } => sender_children.contains(&subject),
    }
}

/// ONE statement inside a sealed Q2-relay batch (mesh minor 17; owner-approved 2026-08-16 —
/// `docs/design/owner_decisions_2026-08-15.md` addendum + `docs/design/window_lane.md` §5
/// RULINGS): exactly the payloads the child's own direct window lanes carry, minus the
/// subscription id (subscriber-side state the AUTHOR never holds). Only the AUTHORING child
/// builds these ([`seal_relay_statements`]) and only the final receiver decodes them
/// ([`open_relay_statements`]); the relaying parent holds the sealed bytes and structurally
/// cannot re-state a value it never sees.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RelayedStatement {
    /// The child's own body statement — the [`ShardToGateway::WindowBody`] payload verbatim.
    /// Admitted against the CHILD's identity by [`window_body_admissible`]: a `SelfLook` only
    /// about the child itself, a `Marker` only about the child's own direct children.
    Body {
        subject: RealmId,
        stmt: BodyStmt,
        authored_at: UniverseTick,
    },
    /// The child's own authored interior level — the [`ShardToGateway::WindowFrame`] payload
    /// with `hop: None` (an own-level statement: the child states its interior in its OWN frame;
    /// the outside observer's hop TO the child is the parent's placement row, which the receiver
    /// already holds from the parent's own window). Doubles as the child's attested roster the
    /// batch's markers are vouched against.
    Level {
        at: UniverseTick,
        rows: Vec<RealmSnap>,
    },
}

/// THE LOOK CARRIER's ARITY (look_horizon.md §3.2/§3.3.2, owner Q3 ruling 2026-08-17): how many
/// LEVELS below a window's author the landed carrier can serve a subject's presence — the
/// author's own statements name its direct children (1), and ONE relayed child batch (the Q2
/// relay: held sealed, forwarded once) names that child's own children (2). There is no deeper
/// field on the wire — a third level is UNREPRESENTABLE, which is what makes this a bound and
/// not a promise; deepening it is an edit to this reviewed file. The boot fence
/// (`vd_physics::worldgen::guard_visibility_climb_bounded`) refuses any world whose MEASURED
/// visibility climb exceeds this number, and the build-admission fence refuses any candidate
/// placement that would need more. Per the Q3 ruling: the arity STAYS 2; the near-real-scale
/// world re-solve is the scheduled cure (its first gate run must include
/// `measure_visibility_climb`); 3 only if that measurement demands it.
pub const LOOK_CARRIER_ARITY: usize = 2;

/// Seal a batch of self-authored statements for the Q2 relay (author-side ONLY — the one lawful
/// builder). The bytes ride [`crate::intershard::InterShardFlow::WindowRelay`] up and
/// [`ShardToGateway::WindowRelayed`] out, unopened in between.
#[must_use]
pub fn seal_relay_statements(statements: &[RelayedStatement]) -> Vec<u8> {
    postcard::to_allocvec(statements).expect("closed wire enums serialize infallibly")
}

/// Open a sealed relay batch (final-receiver-side ONLY). The relaying parent never calls this —
/// its holder keeps bytes, never values (forward-or-drop, Q2's "no read").
///
/// # Errors
/// The postcard decode error when the blob is not a well-formed statement batch (counted by the
/// caller as an undecodable relay, dropped — fail-closed).
pub fn open_relay_statements(bytes: &[u8]) -> Result<Vec<RelayedStatement>, postcard::Error> {
    postcard::from_bytes(bytes)
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
        // VU AoI (proto_minor 6): a per-observer delta — one realm ENTERED the observer's AoI (its shape),
        // one LEFT (its id) — round-trips, and is NOT an opaque forwarded frame (both extractors decline it).
        // The shape is the flag-day PURE SELF-DESCRIPTION (no position field exists, minor 18).
        let delta = ShardToGateway::RealmSceneDelta {
            observer: AccountId(5),
            added: vec![RealmShape {
                realm: RealmId::Planet(7),
                frame: FrameRef::SystemSpace { system_seed: 7 },
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

    // ===== THE WINDOW LANE (mesh minor 16) — pins, roundtrips, admission rules ============

    /// One typed authored child row for the window fixtures — head ≠ tail (the child's own frame
    /// vs the authoring parent's frame), every field non-default so a dropped field cannot pass
    /// as a lucky zero.
    fn window_row() -> RealmSnap {
        RealmSnap {
            realm: RealmId::Planet(7),
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 1 },
                DVec3::new(20.0, -3.0, 5.0),
                UniverseTick(100),
            ),
        }
    }

    /// One full-width pre-inverted hop row (no zero field, a real rotation) — the author's own
    /// frame expressed in the child's frame.
    fn hop_row() -> HopRow {
        HopRow {
            child: RealmId::Planet(7),
            inv: FramePlacement {
                origin_cell: vd_core::glam::I64Vec3::new(1, -2, 3),
                origin: DVec3::new(-20.0, 3.0, -5.0),
                velocity: DVec3::new(0.25, -0.5, 1.0),
                orientation: vd_core::glam::DQuat::from_xyzw(0.5, 0.5, 0.5, 0.5),
                angular_velocity: DVec3::new(0.0, 0.125, 0.0),
            },
        }
    }

    /// EVERY `ShardToGateway` arm, one fixture each, in declaration order — the session lane's
    /// per-release surface set (the `intershard_closed::every_arm` discipline brought in-crate).
    fn every_shard_to_gateway_arm() -> Vec<ShardToGateway> {
        vec![
            ShardToGateway::SessionAttached {
                session: SessionId(1),
                entity: EntityId(9),
                frame: FrameRef::SystemSpace { system_seed: 4 },
                realm_fence: Fence(1),
            },
            ShardToGateway::Frame {
                realm_fence: Fence(1),
                source_tick: TickId(8),
                snapshot_bytes: vec![1, 2, 3],
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
            ShardToGateway::RealmSceneDelta {
                observer: AccountId(5),
                added: Vec::new(),
                removed: vec![RealmId::Planet(8)],
            },
            ShardToGateway::EntityRemoved {
                realm_fence: Fence(1),
                entity: EntityId(9),
                at: UniverseTick(11),
            },
            // The window lane (mesh minor 16) — the `Some(hop)` arm of a Child-scope frame.
            ShardToGateway::WindowFrame {
                realm_fence: Fence(3),
                window: WindowId(2),
                at: UniverseTick(100),
                hop: Some(Box::new(hop_row())),
                rows: vec![window_row()],
            },
            ShardToGateway::WindowBody {
                realm_fence: Fence(3),
                window: WindowId(2),
                subject: RealmId::System(4),
                stmt: BodyStmt::SelfLook { bag: vec![9, 9] },
                authored_at: UniverseTick(100),
            },
            ShardToGateway::WindowMembership {
                window: WindowId(2),
                added: vec![RealmId::Planet(7)],
                removed: vec![RealmId::Planet(8)],
            },
            // The Q2 relay forward leg (mesh minor 17) — sealed statements ride as-received;
            // mesh minor 20 appends the sealed interior forward (a grandchild's own batch,
            // fenced outside its seal, forwarded verbatim).
            ShardToGateway::WindowRelayed {
                realm_fence: Fence(3),
                window: WindowId(2),
                child: RealmId::Planet(7),
                child_fence: Fence(9),
                statements: seal_relay_statements(&[RelayedStatement::Body {
                    subject: RealmId::Planet(7),
                    stmt: BodyStmt::SelfLook { bag: vec![8, 8] },
                    authored_at: UniverseTick(101),
                }]),
                interior: vec![crate::intershard::InteriorRelay {
                    child: RealmId::Area(3),
                    child_fence: Fence(11),
                    own: seal_relay_statements(&[RelayedStatement::Body {
                        subject: RealmId::Area(3),
                        stmt: BodyStmt::SelfLook { bag: vec![7, 7] },
                        authored_at: UniverseTick(102),
                    }]),
                }],
            },
        ]
    }

    /// EVERY `GatewayToShard` arm, one fixture each, in declaration order.
    fn every_gateway_to_shard_arm() -> Vec<GatewayToShard> {
        vec![
            GatewayToShard::AttachSession {
                session: SessionId(1),
                fence: Fence(2),
                account: AccountId(3),
                spawn: None,
            },
            GatewayToShard::SessionInput {
                session: SessionId(1),
                fence: Fence(2),
                input_bytes: vec![5],
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
            // The window lane (mesh minor 16) — the Child scope carries a realm id.
            GatewayToShard::WindowOpen {
                window: WindowId(2),
                scope: WindowScope::Child(RealmId::Planet(7)),
            },
            GatewayToShard::WindowClose {
                window: WindowId(2),
            },
        ]
    }

    /// THE POSITIONAL PIN for BOTH session-lane enums (the `intershard_closed` tombstone
    /// discipline): postcard writes a variant's DECLARED index as the leading varint, so
    /// reordering — or deleting — an arm re-labels every later arm ON THE WIRE while every
    /// same-build roundtrip stays green. Declaration order stated ONCE as data (wildcard-free,
    /// so a new arm must take a pinned index to compile), asserted against the real first byte,
    /// and the fixture set must span the whole contiguous index space (no vacuous pass).
    #[test]
    fn every_session_flow_arm_encodes_its_declared_discriminant_index() {
        fn s2g_index(msg: &ShardToGateway) -> u8 {
            match msg {
                ShardToGateway::SessionAttached { .. } => 0,
                ShardToGateway::Frame { .. } => 1,
                ShardToGateway::SessionDetached { .. } => 2,
                ShardToGateway::SubscriptionReady { .. } => 3,
                ShardToGateway::RealmFrame { .. } => 4,
                ShardToGateway::RealmSceneDelta { .. } => 5,
                ShardToGateway::EntityRemoved { .. } => 6,
                // The window lane holds 7/8/9 (mesh minor 16) forever.
                ShardToGateway::WindowFrame { .. } => 7,
                ShardToGateway::WindowBody { .. } => 8,
                ShardToGateway::WindowMembership { .. } => 9,
                // The Q2 relay forward leg holds 10 (mesh minor 17) forever.
                ShardToGateway::WindowRelayed { .. } => 10,
            }
        }
        fn g2s_index(msg: &GatewayToShard) -> u8 {
            match msg {
                GatewayToShard::AttachSession { .. } => 0,
                GatewayToShard::SessionInput { .. } => 1,
                GatewayToShard::DetachSession { .. } => 2,
                GatewayToShard::OpenInputSlot { .. } => 3,
                // The window control lane holds 4/5 (mesh minor 16) forever.
                GatewayToShard::WindowOpen { .. } => 4,
                GatewayToShard::WindowClose { .. } => 5,
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for msg in every_shard_to_gateway_arm() {
            let bytes = postcard::to_allocvec(&msg).expect("encode");
            assert_eq!(bytes[0], s2g_index(&msg));
            seen.insert(bytes[0]);
        }
        assert_eq!(seen.len(), 11);
        assert_eq!(seen.first().copied(), Some(0));
        assert_eq!(seen.last().copied(), Some(10));
        let mut seen = std::collections::BTreeSet::new();
        for msg in every_gateway_to_shard_arm() {
            let bytes = postcard::to_allocvec(&msg).expect("encode");
            assert_eq!(bytes[0], g2s_index(&msg));
            seen.insert(bytes[0]);
        }
        assert_eq!(seen.len(), 6);
        assert_eq!(seen.first().copied(), Some(0));
        assert_eq!(seen.last().copied(), Some(5));
    }

    /// The NESTED positional pins for the window lane's two payload enums — the same
    /// reorder/deletion hole the outer table closes, at the nested level ([`BodyStmt`] and
    /// [`WindowScope`] each lead with their own declared index).
    #[test]
    fn window_nested_enums_encode_their_declared_discriminant_indices() {
        // Standalone: the nested enum's own leading byte IS its declared index.
        let look = BodyStmt::SelfLook { bag: vec![1] };
        let marker = BodyStmt::Marker { luma: vec![2] };
        assert_eq!(postcard::to_allocvec(&look).expect("encode")[0], 0);
        assert_eq!(postcard::to_allocvec(&marker).expect("encode")[0], 1);
        let occupants = WindowScope::Occupants;
        let child = WindowScope::Child(RealmId::Planet(7));
        assert_eq!(postcard::to_allocvec(&occupants).expect("encode")[0], 0);
        assert_eq!(postcard::to_allocvec(&child).expect("encode")[0], 1);
        // In context: `WindowOpen{window: WindowId(1), scope}` puts the scope tag at byte 2
        // (outer tag 4, then the one-byte WindowId varint) — the nested index rides the real
        // message exactly where the declaration says.
        let open_occ = GatewayToShard::WindowOpen {
            window: WindowId(1),
            scope: WindowScope::Occupants,
        };
        let open_child = GatewayToShard::WindowOpen {
            window: WindowId(1),
            scope: WindowScope::Child(RealmId::Planet(7)),
        };
        assert_eq!(
            postcard::to_allocvec(&open_occ).expect("encode")[..3],
            [4, 1, 0]
        );
        assert_eq!(
            postcard::to_allocvec(&open_child).expect("encode")[..3],
            [4, 1, 1]
        );
    }

    /// Same-build roundtrips for EVERY arm of both enums (the new window arms ride the same
    /// fixture set as the frozen ones), plus the shapes the surface set does not carry: the
    /// `hop: None` own-level frame, the `Marker` body, the `Occupants` scope, and an EMPTY
    /// membership delta (byte-cheap-when-empty, never a decode fault).
    #[test]
    fn window_lane_arms_roundtrip_postcard() {
        for msg in every_shard_to_gateway_arm() {
            let bytes = postcard::to_allocvec(&msg).expect("encode");
            assert_eq!(
                postcard::from_bytes::<ShardToGateway>(&bytes).expect("decode"),
                msg
            );
        }
        for msg in every_gateway_to_shard_arm() {
            let bytes = postcard::to_allocvec(&msg).expect("encode");
            assert_eq!(
                postcard::from_bytes::<GatewayToShard>(&bytes).expect("decode"),
                msg
            );
        }
        let own_level = ShardToGateway::WindowFrame {
            realm_fence: Fence(3),
            window: WindowId(2),
            at: UniverseTick(100),
            hop: None, // the Occupants-scope frame: no hop — the observer stands in this frame
            rows: vec![window_row()],
        };
        let marker = ShardToGateway::WindowBody {
            realm_fence: Fence(3),
            window: WindowId(2),
            subject: RealmId::Planet(7),
            stmt: BodyStmt::Marker { luma: vec![4, 2] },
            authored_at: UniverseTick(100),
        };
        let empty_membership = ShardToGateway::WindowMembership {
            window: WindowId(2),
            added: Vec::new(),
            removed: Vec::new(),
        };
        let open_occupants = GatewayToShard::WindowOpen {
            window: WindowId(2),
            scope: WindowScope::Occupants,
        };
        for msg in [own_level, marker, empty_membership] {
            let bytes = postcard::to_allocvec(&msg).expect("encode");
            assert_eq!(
                postcard::from_bytes::<ShardToGateway>(&bytes).expect("decode"),
                msg
            );
        }
        let bytes = postcard::to_allocvec(&open_occupants).expect("encode");
        assert_eq!(
            postcard::from_bytes::<GatewayToShard>(&bytes).expect("decode"),
            open_occupants
        );
        // The Box on `hop` is a MEASURED no-op on the wire (never-assume): a boxed hop row
        // encodes byte-identically to the bare row, so the wire shape is the design's
        // `Option<HopRow>` exactly.
        assert_eq!(
            postcard::to_allocvec(&Box::new(hop_row())).expect("encode boxed"),
            postcard::to_allocvec(&hop_row()).expect("encode bare"),
        );
    }

    /// The additive-decode discipline (mesh minor 16): the window arms are APPENDED, so bytes a
    /// minor-15 sender produced (any prior variant — here the newest prior arm on each enum)
    /// still decode unchanged, and each new variant is a clean self-contained message. A trailing
    /// variant never shifts a prior variant's discriminant or framing.
    #[test]
    fn window_lane_is_additive_minor_16_and_prior_variants_decode_unchanged() {
        let prior_s2g = ShardToGateway::EntityRemoved {
            realm_fence: Fence(1),
            entity: EntityId(9),
            at: UniverseTick(11),
        };
        let bytes = postcard::to_allocvec(&prior_s2g).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ShardToGateway>(&bytes).expect("decode"),
            prior_s2g
        );
        let prior_g2s = GatewayToShard::OpenInputSlot {
            session: SessionId(1),
            fence: Fence(2),
            account: AccountId(3),
            resume_from_seq: 42,
            subject: DirectoryKey::Entity(EntityId(9)),
        };
        let bytes = postcard::to_allocvec(&prior_g2s).expect("encode");
        assert_eq!(
            postcard::from_bytes::<GatewayToShard>(&bytes).expect("decode"),
            prior_g2s
        );
        // The extractor sugar declines every window arm (they are typed rows, never opaque
        // frames) — the exhaustive matches stay honest.
        for msg in every_shard_to_gateway_arm().into_iter().skip(7) {
            assert_eq!(msg.clone().into_snapshot_bytes(), None);
            assert_eq!(msg.into_realm_snapshot_bytes(), None);
        }
    }

    /// The frame/membership admission rule (`window_lane.md` §2.2): the sender must BE the
    /// resolved roster head — a mismatched head refuses, and an UNRESOLVED head refuses too
    /// (fail-closed: unattestable is dropped, never served on faith).
    #[test]
    fn window_sender_is_head_admits_only_the_resolved_head() {
        use vd_core::NodeId;
        assert!(window_sender_is_head(NodeId(1002), Some(NodeId(1002))));
        assert!(!window_sender_is_head(NodeId(1002), Some(NodeId(1004))));
        assert!(!window_sender_is_head(NodeId(1002), None));
    }

    /// The body admission rule (`window_lane.md` §2.2): `SelfLook` only about the sender's own
    /// realm; `Marker` only about a DIRECT child. Both answers of BOTH arms driven — including
    /// the cross cases (a look about a child, a marker about the sender itself) that a lazier
    /// fixture would leave unrun.
    #[test]
    fn window_body_admissible_enforces_the_two_exclusive_authorships() {
        let sender = RealmId::System(4);
        let child = RealmId::Planet(7);
        let stranger = RealmId::Planet(8);
        let children: BTreeSet<RealmId> = [child].into_iter().collect();
        let look = BodyStmt::SelfLook { bag: vec![1] };
        let marker = BodyStmt::Marker { luma: vec![2] };
        // A realm states a look ONLY about itself.
        assert!(window_body_admissible(&look, sender, sender, &children));
        assert!(!window_body_admissible(&look, child, sender, &children));
        // A marker ONLY about a direct child — never about itself, never about a stranger.
        assert!(window_body_admissible(&marker, child, sender, &children));
        assert!(!window_body_admissible(&marker, sender, sender, &children));
        assert!(!window_body_admissible(
            &marker, stranger, sender, &children
        ));
    }

    /// The Q2 relay's sealed batch (mesh minor 17): seal → open is the identity for every
    /// statement kind; the nested discriminants are pinned (Body 0 / Level 1 forever); and a
    /// malformed blob opens to a typed error, never a guess (the fail-closed drop the receiver
    /// counts). The admission of an opened Body statement is the SAME predicate the direct lane
    /// uses, applied with the CHILD as the stating realm — asserted here so the "existing
    /// attestation predicates against the child's identity" wording stays a measurement.
    #[test]
    fn relay_statements_seal_open_verbatim_and_pin_their_nested_discriminants() {
        let child = RealmId::Planet(7);
        let grandchild = RealmId::Area(3);
        let body = RelayedStatement::Body {
            subject: child,
            stmt: BodyStmt::SelfLook { bag: vec![1, 2] },
            authored_at: UniverseTick(100),
        };
        let marker = RelayedStatement::Body {
            subject: grandchild,
            stmt: BodyStmt::Marker { luma: vec![3] },
            authored_at: UniverseTick(100),
        };
        let level = RelayedStatement::Level {
            at: UniverseTick(100),
            rows: vec![window_row()],
        };
        // Nested positional pins: a reorder would relabel sealed bytes already in flight.
        assert_eq!(postcard::to_allocvec(&body).expect("encode")[0], 0);
        assert_eq!(postcard::to_allocvec(&level).expect("encode")[0], 1);
        // Seal → open is the identity (the parent forwards these bytes UNOPENED in between).
        let batch = vec![body.clone(), marker.clone(), level.clone()];
        let sealed = seal_relay_statements(&batch);
        assert_eq!(open_relay_statements(&sealed).expect("open"), batch);
        // A malformed blob refuses loud (the receiver's undecodable-relay drop).
        assert!(open_relay_statements(&[0xFF, 0xFF, 0xFF]).is_err());
        // The opened statements admit under the DIRECT lane's predicate with the CHILD as the
        // stating realm: its self-look about itself, its marker about ITS OWN direct child.
        let childs_children: BTreeSet<RealmId> = [grandchild].into_iter().collect();
        // The SAME (stmt, subject) values the batch above carries, restated as data (HR5: a
        // destructure of a value constructed three lines up has an unreachable refusal arm).
        assert!(window_body_admissible(
            &BodyStmt::SelfLook { bag: vec![1, 2] },
            child,
            child,
            &childs_children
        ));
        assert!(window_body_admissible(
            &BodyStmt::Marker { luma: vec![3] },
            grandchild,
            child,
            &childs_children
        ));
        // A relayed look about anything but the child itself is refused by the same predicate.
        assert!(!window_body_admissible(
            &BodyStmt::SelfLook { bag: vec![1] },
            grandchild,
            child,
            &childs_children
        ));
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
