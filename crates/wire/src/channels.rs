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
use vd_core::geometry::Boundary;
use vd_core::pose::{FrameRef, RealmId, StampedPose};
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
    /// "This entity is YOUR avatar" — the PURE-RENDERER own-entity signal (proto_minor 2),
    /// emitted only to a peer that negotiated minor >= 2. It replaces `AuthorityChanged` as
    /// the "this is your avatar" signal for a node-AGNOSTIC client: it names ONLY the entity,
    /// never a `sub` / owning node, so a pure-renderer client learns which entity to center on
    /// WITHOUT ever learning which shard simulates it (the server re-homes authority invisibly;
    /// the client just renders whatever authoritative coordinate streams in for that entity).
    /// Appended trailing variant — every prior variant decodes unchanged (postcard additive rule).
    OwnEntity {
        entity: EntityId,
    },
    /// THE COMPOSED SCENE LEVEL (the window lane's flag day, proto_minor 18 —
    /// `docs/design/window_lane.md` §2.4; owner-approved 2026-08-15/16, items 1/9/10): the COMPLETE
    /// drawable scene for this session, stacked from attested per-realm window statements at ONE
    /// universe tick and expressed in the ORIGIN realm's frame. The client draws it and computes
    /// nothing — every row arrives ready (pose + bag), so a row is drawable the instant it lands
    /// (no "shape before first pose row" gap exists, at login or crossing). Sent on (re)login and
    /// on every origin change (a crossing); the per-tick POSITIONS stream on the unreliable
    /// [`RealmSnapshotDatagram`]; incremental membership/body changes ride [`RealmSceneDelta`].
    /// Reshaped IN PLACE at the flag day (owner item 9: zero deployed clients ⇒ no shims, no
    /// dual-decode; [`crate::version::PROTO_MINOR_FLOOR`] moved so no pre-flag-day peer is served).
    RealmRegistry {
        /// THE ORIGIN MARKER (§2.7, owner item 9's explicit successor of the deleted `pin`): the
        /// realm this scene is composed in — the observer's standing realm. Stamped at the
        /// composition point, ON the level itself (no separate origin message exists that could
        /// desync from it). The origin realm never appears as a row: it draws from its own look
        /// AT the origin.
        origin: RealmId,
        /// The origin's epoch: bumps on every origin/chain change. The client swaps scenes
        /// ATOMICALLY on the bump — the old scene renders until the new level lands; early
        /// new-epoch datagrams are held one beat (§2.7). This REPLACES the client's old
        /// `forget_space` inference.
        origin_epoch: u64,
        /// The composed rows — the full drawn set at one tick.
        rows: Vec<SceneRow>,
    },
    /// The INCREMENTAL composed-scene update (reliable): rows that ENTERED the drawn set (`added`
    /// — complete [`SceneRow`]s, drawable on arrival) and realms that LEFT it (`removed`). Rides
    /// the same origin marker + epoch as [`RealmRegistry`]; a delta whose epoch is not the
    /// client's current one is refused (it belongs to a scene the client no longer — or does not
    /// yet — hold). Emitted only on a real membership/body change. Reshaped IN PLACE at the
    /// proto_minor-18 flag day (same owner citation as [`RealmRegistry`]).
    RealmSceneDelta {
        /// The same origin marker as [`RealmRegistry`] (§2.7) — re-carried on every delta so the
        /// reliable scene lane is self-describing.
        origin: RealmId,
        /// The same epoch as [`RealmRegistry`]; the client applies a delta only at its current
        /// epoch.
        origin_epoch: u64,
        added: Vec<SceneRow>,
        removed: Vec<RealmId>,
    },
    /// A reliable discrete event for this client (proto_minor 14) — TODAY only
    /// [`EventMsg::EntityRemoved`], the per-entity eviction a pure-renderer client cannot derive
    /// (its tracks are keyed by `EntityId` and a sub close deliberately evicts nothing); P9's
    /// gameplay signal deliveries ride this SAME arm as appended `EventMsg` variants (D-4: one
    /// carrier, two consumers, built once). Emitted only to a peer that negotiated minor >= 14.
    /// Appended trailing variant (postcard additive rule).
    Event(EventMsg),
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
    /// ride reliable channels instead (v1.1). That owed reliable client→shard discrete-action arm
    /// has TWO consumers — P6 block-edit-forward AND P11 PvP fire-registration (DEFERRED D-39.1);
    /// it is build-once shared infra, never a per-feature fork (HR3).
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

/// Reliable per-subscription discrete gameplay events (proto_minor 14 — the first minor that ever
/// EMITS one; the enum was declared unroutable since P1.5, so reshaping `EntityRemoved` below was
/// lawful: no producer existed, no peer ever negotiated a wire that carried it). Rides the reliable
/// Control lane inside [`ServerControlMsg::Event`]; P9's gameplay signals append variants HERE.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EventMsg {
    Notice {
        text: String,
    },
    /// THE REMOVE MESSAGE (D-4(a), owner-picked 2026-08-13): the server-authoritative "this entity
    /// left your view". The client evicts the entity's track on THIS and on nothing else — absence
    /// from a datagram is never an eviction (the deliberate reliable-signal-only drop rule).
    ///
    /// `at` is the emitting shard's universe tick at the removal — the client's RESURRECT GUARD: the
    /// removal races the unreliable snapshot lane, so a straggler row for this entity with a stamp
    /// `<= at` is refused (it predates the removal), while a NEWER row is a genuine return (the
    /// player flew back) and clears the guard. Without it a reordered datagram would silently
    /// re-create the track this message just evicted — frozen forever, the exact defect again.
    EntityRemoved {
        entity: EntityId,
        at: UniverseTick,
    },
}

/// One entity's state inside a snapshot frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntitySnap {
    pub entity: EntityId,
    pub pose: StampedPose,
}

/// One realm's PURE SELF-DESCRIPTION — its id, its own frame, its boundary and its parent link.
/// A shape says WHAT a realm looks like and NEVER where it is (the proto_minor-18 flag day,
/// `docs/design/window_lane.md` §2.4; owner-approved 2026-08-15 item 9): the one-meaning law.
/// WHERE a realm sits is exactly one party's statement — its parent's placement row — and rides
/// the placement lanes ([`RealmSnap`]) or arrives already composed ([`SceneRow`]). This type used
/// to carry a `center` whose meaning was "measured from whoever this message is addressed to",
/// which made one field mean a different number on every hop and required a chain of restating
/// parties to keep it true; the field is DELETED, not zeroed — a field that must be re-derived on
/// every hop is an invitation to re-derive it wrongly. THE CURE for the old known violator (a
/// multi-level shard emitting two frames with no hop between them) is structural: with no
/// position field, a shape cannot state a position in anyone's frame, so the violation is
/// unrepresentable.
///
/// Post-flag-day consumers are the still-running mesh scene lanes only
/// (`ShardToGateway::RealmSceneDelta`, [`crate::intershard::ChildSceneSet`]) — the client never
/// receives this type any more ([`SceneRow`] is the client's row). Those lanes are deleted whole
/// in Slice C2 and this type retires with them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealmShape {
    pub realm: RealmId,
    /// THIS realm's OWN frame — the frame its occupants and its own children are measured in.
    pub frame: FrameRef,
    pub shape: Boundary,
    pub parent: Option<RealmId>,
}

/// ONE ROW OF THE COMPOSED SCENE (proto_minor 18, `docs/design/window_lane.md` §2.4 — the VU
/// streaming contract realized: *a pose + a bag of signals*). The row arrives ready to draw: the
/// pose is already expressed in the level's ORIGIN frame at one stamp, and the bag carries the
/// row's whole appearance as tagged TLV (`vd_core::look` codec). The client applies zero
/// transforms and branches on NO realm kind — only on data presence:
///
/// - `TAG_LOOK` present  ⇒ a BODY: the realm's OWN self-authored outline (a running realm draws
///   itself — THE DRAW LAW, owner decision 10).
/// - `TAG_LOOK` absent + `TAG_LUMA` present ⇒ a MARKER: the parent-authored point-of-light datum
///   for a sleeping realm (photometric scalars, never an outline).
/// - Unknown tags are SKIPPED — signals extend forever with zero client change.
///
/// A third pixel source is unrepresentable: the two tags come from wire types
/// ([`crate::session_flow::BodyStmt`]) that structurally cannot carry each other's payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneRow {
    pub realm: RealmId,
    /// Hierarchy identity only — which realm authored this row's placement. Never a position.
    pub parent: Option<RealmId>,
    /// The row's ready-to-draw pose in the ORIGIN frame, stamp explicit PER ROW: held strata
    /// lawfully carry an older stamp than their level-mates (`window_lane.md` §2.6.4), declared
    /// here rather than hidden. `pose.frame` names the origin frame, which also states the tier
    /// (the drawn unit) explicitly — no receiver-side tier inference exists.
    pub pose: StampedPose,
    /// The tagged skip-unknown TLV bag (`vd_core::look`): the row's appearance. Empty ⇒ the
    /// realm is tracked but NOT drawn (a missing statement means the thing is not drawn — every
    /// pixel has exactly one lawful author, by construction).
    pub bag: Vec<u8>,
}

/// One REALM's authored placement inside a realm-snapshot frame (the frame-authority
/// observer feed, D-45(a) realm-unification FA-2a). A realm is NOT an entity — it is
/// keyed by [`RealmId`] (the render layer's `RealmScene` is `RealmId`-keyed), and
/// [`FrameRef::realm`] is a LOSSY inverse, so a moving realm's box cannot be recovered
/// from an [`EntitySnap`]'s `pose.frame`. Its parent shard AUTHORS this pose each tick
/// ({input signals} + {ambient physics} → pose; a passive orbiting body is the
/// zero-signal degenerate case) and SHIPS it to observers as a latest-wins,
/// FireAndForget row — never acked, always re-derivable (kept STRICTLY separate from the
/// child-shard authority feed). EVERY direct child ships a row, static and moving alike
/// (owner Q3, the placement arc): "movers only" was a motion test deciding what the feed
/// ships, which SL4 forbids — so a walk/static forest with children now puts rows on the
/// wire too (the old zero-bytes floor was that filter's artifact, re-baselined with it).
///
/// SELF-DESCRIBING PLACEMENT EDGE (proto_minor 8). A row is the graph edge `(head, tail, value)`:
/// `frame` is the HEAD — the CHILD's own frame, the frame that realm's occupants are measured in;
/// `pose.frame` is the TAIL — the frame the value is measured in; `pose.pos`/`pose.vel` are the value,
/// that realm's centre expressed in the tail's frame. Head and tail therefore DIFFER on every real row;
/// a row where they are equal is a realm claiming to be its own parent.
///
/// THE TAIL IS WHOEVER IS SHIPPING THE ROW, which for a row a shard authors and emits directly is that
/// shard's own frame — the child's parent — and stays so all the way to the client. On the observation
/// cascade ([`crate::intershard::RealmCascade`]) it is instead the frame of the realm the row is being
/// shipped DOWN to, because each level restates the value from that child's centre before sending it,
/// and it does so precisely so the party at the bottom holds one space rather than two. The head never
/// moves: it is a property of the row's own realm, not of the hop.
///
/// The head is carried rather than derived because [`FrameRef::realm`] is a LOSSY inverse: an
/// `AreaLocal { planet_seed, area_seed }` collapses to `RealmId::Area(area_seed)`, and no receiver can
/// invert that without already knowing which planet holds the area — i.e. without the very hierarchy it
/// is trying to build. Carrying the head means a receiver needs no join against the reliable shape lane,
/// no per-session scene mirror, and no ordering guarantee between the two lanes: the unreliable pose row
/// is complete on its own. The cost is one discriminant byte plus one or two u64 varints, on rows bounded
/// by MOVING DIRECT CHILDREN — tens — never by entities. MEASURED, encoding the real type: **2 B/row** for
/// a `PlanetCentered` head and **3 B/row** for the two-seed `AreaLocal` head at the small seeds the shipped
/// forest generates, and **21 B/row** at full-width `u64` seeds, which is the true worst case (an earlier
/// estimate of "≤19 B" here was two bytes short — two 10-byte varints plus the discriminant).
///
/// THE HEAD RIDES THE CLIENT-FACING DATAGRAM TOO, deliberately, and the reason is that there is no other
/// datagram to keep it off. The parent-to-parent hop and the client feed are ONE byte stream: a level
/// receives a [`crate::intershard::RealmCascade`], and a level with nobody standing on it forwards those
/// exact bytes to its own gateway as `ShardToGateway::RealmFrame` without decoding them. Giving the client
/// a leaner row would mean the bottom level decoding every row, stripping a field and re-encoding — the
/// one thing a leaf is defined by not doing, and the property `cascade_rows_converted == 0` is the
/// measurement of. The client itself reads `realm` and `pose` and ignores the head; it pays two bytes a row
/// on rows counted in tens, in exchange for the leaf staying a pass-through. Judged against re-splitting the
/// type, that is the cheaper side.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealmSnap {
    pub realm: RealmId,
    /// The edge HEAD: the CHILD realm's OWN frame (see the type docs). Not derivable from `realm`.
    pub frame: FrameRef,
    /// The edge TAIL + VALUE: `pose.frame` is the frame of whoever is shipping this row, `pose.pos` this
    /// realm's centre measured in it (see the type docs for why that is the authoring parent on one lane
    /// and the receiving child on the other).
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

/// The per-subscription REALM-placement datagram (D-45(a) realm-unification FA-2a): the
/// twin of [`SnapshotDatagram`] carrying a shard's authored placements for the RENDERABLE
/// REALMS it parents (a moving planet/station/ship box), keyed by [`RealmId`] not
/// [`EntityId`]. Shares [`SnapshotDatagram`]'s staleness discipline (`sub` + monotone
/// `frame_id`, latest-wins, unreliable) and is partitioned by the SAME MTU budget. A shard
/// emits this ONLY when it parents ≥1 moving/renderable child, so at walk/static scale
/// (every placement identity ⇒ no moving child) NOTHING is sent — zero bytes, byte-identical
/// to the pre-plant wire. FA-2c wires the emit; this is the frozen SHAPE plant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealmSnapshotDatagram {
    pub sub: SubId,
    /// Monotonic per-sub frame counter (stale frames dropped) — the realm feed's own
    /// counter, independent of the entity snapshot's `frame_id`. On the composed client feed
    /// (proto_minor 18) this is the per-session monotone counter the connection plane stamps —
    /// lawful because a composed row is a NEW row it authors from attested inputs — so the
    /// client's staleness gate collapses to ONE feed counter (single author) plus the epoch.
    pub frame_id: u64,
    /// The SENDER's local sim tick (there is NO global sim tick).
    pub source_tick: TickId,
    /// The analytic clock value this frame's placements are authored against.
    pub universe_tick: UniverseTick,
    /// The scene epoch these rows belong to (proto_minor 18, `docs/design/window_lane.md`
    /// §2.4/§2.7) — the same counter [`ServerControlMsg::RealmRegistry`] carries. The client
    /// applies rows only at its CURRENT epoch: an older epoch is a straggler from the previous
    /// scene (dropped + counted `stale_epoch_rows`), a NEWER epoch raced its own level on the
    /// reliable lane and is held one beat (§2.7). On the still-running shard-authored mesh
    /// lanes (dead in Slice C2 with their lanes) this field is stamped 0 — those bytes reach
    /// no client any more; only the composed feed carries a live epoch.
    pub origin_epoch: u64,
    pub realms: Vec<RealmSnap>,
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
    partition_rows(entities, budget_bytes)
}

/// Split a shard's authored REALM placements into datagram-sized chunks — the [`RealmSnapshotDatagram`]
/// twin of [`partition_entities`] (D-45(a) FA-2c), sharing the SAME greedy MTU budget so a realm feed
/// never exceeds the datagram MTU (audit GW-1). EMPTY in ⇒ EMPTY out (a walk-scale shard authors no
/// moving child ⇒ no realm datagram is ever sent).
#[must_use]
pub fn partition_realms(realms: &[RealmSnap], budget_bytes: usize) -> Vec<Vec<RealmSnap>> {
    partition_rows(realms, budget_bytes)
}

/// The shared greedy MTU partitioner for a snapshot row type (`EntitySnap` / `RealmSnap`, DRY). A
/// BRANCHLESS generic shim (HR5): it maps each row to its encoded size and slices by the chunk ranges
/// [`chunk_boundaries`] computes — ALL the bin-packing branching lives in that MONOMORPHIC helper, so a
/// new row-type monomorphization adds ZERO uncovered per-mono branch.
fn partition_rows<T: Serialize + Copy>(rows: &[T], budget_bytes: usize) -> Vec<Vec<T>> {
    let sizes: Vec<usize> = rows
        .iter()
        .map(|r| {
            postcard::to_allocvec(r)
                .map(|v| v.len())
                .unwrap_or(SNAPSHOT_HEADER_BUDGET)
        })
        .collect();
    chunk_boundaries(&sizes, budget_bytes, SNAPSHOT_HEADER_BUDGET)
        .into_iter()
        .map(|(s, e)| rows[s..e].to_vec())
        .collect()
}

/// Greedy MTU bin-packing over the ENCODED SIZES of a row list: the `[start, end)` chunk index ranges
/// that keep each chunk's body at or under the budget (one oversize row still ships alone). MONOMORPHIC
/// (takes only `&[usize]`), so ALL the partition branching is covered ONCE here (HR5) and every typed
/// `partition_*` shim stays branchless. `header` is the fixed per-datagram overhead subtracted first.
fn chunk_boundaries(sizes: &[usize], budget_bytes: usize, header: usize) -> Vec<(usize, usize)> {
    let body_budget = budget_bytes.saturating_sub(header).max(1);
    let mut bounds: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    let mut current_bytes = 0usize;
    for (i, &sz) in sizes.iter().enumerate() {
        // Start a new chunk if this row would overflow the body budget — unless the current chunk is
        // empty (`i > start`; one oversize row still ships alone).
        if i > start && current_bytes + sz > body_budget {
            bounds.push((start, i));
            start = i;
            current_bytes = 0;
        }
        current_bytes += sz;
    }
    if start < sizes.len() {
        bounds.push((start, sizes.len()));
    }
    bounds
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

/// The strictly-older staleness predicate shared by every latest-wins frame gate (D-45(a) FA-2c): a
/// `frame_id` STRICTLY below the per-feed `high_water` is a late straggler (stale); an EQUAL id is a
/// sibling chunk of the CURRENT frame (a partitioned multi-datagram snapshot) and is NOT stale — each
/// chunk self-contained, latest-wins. Extracted so the entity [`classify_snapshot`] and the realm-feed
/// gate (`vd_client::RealmView`) share the ONE `<` comparison and can never drift (DRY; §6.3).
#[must_use]
pub fn is_stale(high_water: Option<u64>, frame_id: u64) -> bool {
    high_water.is_some_and(|hw| frame_id < hw)
}

/// THE §6.3 snapshot staleness gate, shared by every snapshot consumer (the
/// in-process `ScriptedClient` and the real client) so they cannot drift: a
/// strictly-older `frame_id` is stale ([`is_stale`]), but an EQUAL `frame_id` is a sibling chunk
/// of the current tick (a partitioned multi-datagram snapshot) and is APPLIED —
/// each chunk self-contained, latest-wins. A sub the client does not hold drops.
///
/// `held_subs` is the SET of subscriptions the client currently holds (Track R / 1d.2d):
/// during a cross-shard transfer the client holds BOTH the source and dest subs for the
/// overlap window, so a frame on EITHER is admitted (the render layer composites the avatar
/// to ONE sub via `AuthorityChanged`/`DeliveredView`). A frame on a sub NOT in the set is a
/// stale in-flight datagram for a closed/never-opened sub and drops.
#[must_use]
pub fn classify_snapshot(
    held_subs: &std::collections::BTreeSet<SubId>,
    high_water: Option<u64>,
    snap_sub: SubId,
    snap_frame_id: u64,
) -> SnapshotVerdict {
    if !held_subs.contains(&snap_sub) {
        return SnapshotVerdict::DropForeignSub;
    }
    if is_stale(high_water, snap_frame_id) {
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
    fn is_stale_is_the_strictly_older_predicate() {
        // The shared latest-wins scalar (FA-2c): no high-water ⇒ never stale; strictly-older ⇒ stale;
        // EQUAL (a sibling chunk of the current frame) and NEWER ⇒ fresh.
        assert!(!is_stale(None, 0));
        assert!(is_stale(Some(5), 4));
        assert!(!is_stale(Some(5), 5));
        assert!(!is_stale(Some(5), 6));
    }

    #[test]
    fn classify_snapshot_applies_siblings_and_drops_stale_or_foreign() {
        use std::collections::BTreeSet;
        let held = SubId(0);
        let none: BTreeSet<SubId> = BTreeSet::new();
        let one: BTreeSet<SubId> = BTreeSet::from([held]);
        let other: BTreeSet<SubId> = BTreeSet::from([SubId(9)]);
        // No sub held yet, or a set without this sub: foreign.
        assert_eq!(
            classify_snapshot(&none, None, held, 5),
            SnapshotVerdict::DropForeignSub
        );
        assert_eq!(
            classify_snapshot(&other, Some(5), held, 5),
            SnapshotVerdict::DropForeignSub
        );
        // Held sub, no high-water yet: apply.
        assert_eq!(
            classify_snapshot(&one, None, held, 5),
            SnapshotVerdict::Apply
        );
        // Newer frame: apply (advances the tick).
        assert_eq!(
            classify_snapshot(&one, Some(5), held, 6),
            SnapshotVerdict::Apply
        );
        // EQUAL frame: a sibling chunk of the current tick — apply (latest-wins).
        assert_eq!(
            classify_snapshot(&one, Some(5), held, 5),
            SnapshotVerdict::Apply
        );
        // Strictly older: stale.
        assert_eq!(
            classify_snapshot(&one, Some(5), held, 4),
            SnapshotVerdict::DropStale
        );
        // TWO held subs (the transfer overlap): a frame on EITHER is admitted.
        let two: BTreeSet<SubId> = BTreeSet::from([SubId(0), SubId(1)]);
        assert_eq!(
            classify_snapshot(&two, None, SubId(0), 1),
            SnapshotVerdict::Apply
        );
        assert_eq!(
            classify_snapshot(&two, None, SubId(1), 1),
            SnapshotVerdict::Apply
        );
        // A third sub not in the held set still drops.
        assert_eq!(
            classify_snapshot(&two, None, SubId(2), 1),
            SnapshotVerdict::DropForeignSub
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
            ServerControlMsg::OwnEntity { entity: eid() },
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
    fn own_entity_is_additive_minor_2_and_prior_variants_decode_unchanged() {
        // The minor-2 additive shape: OwnEntity is appended AFTER UniverseRate, so bytes a
        // minor<2 sender produced (any variant BEFORE OwnEntity, including the minor-1
        // UniverseRate itself) still decode unchanged on a minor-2 decoder — appending a
        // trailing variant never shifts a prior variant's discriminant or framing.
        let prior = ServerControlMsg::UniverseRate { tick_hz: 50 };
        let bytes = postcard::to_allocvec(&prior).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ServerControlMsg>(&bytes).expect("decode"),
            prior
        );
        // The new variant itself is a clean self-contained message.
        let own = ServerControlMsg::OwnEntity { entity: eid() };
        let own_bytes = postcard::to_allocvec(&own).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ServerControlMsg>(&own_bytes).expect("decode"),
            own
        );
    }

    #[test]
    fn event_is_additive_minor_14_and_prior_variants_decode_unchanged() {
        // Appended AFTER RealmSceneDelta ⇒ a minor<14 sender's bytes (any prior variant, here
        // OwnEntity) still decode unchanged on a minor-14 decoder — a trailing variant never
        // shifts a prior discriminant/framing.
        let prior = ServerControlMsg::OwnEntity { entity: eid() };
        let prior_bytes = postcard::to_allocvec(&prior).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ServerControlMsg>(&prior_bytes).expect("decode"),
            prior
        );
        // The new variant itself is a clean self-contained message, payload intact.
        let ev = ServerControlMsg::Event(EventMsg::EntityRemoved {
            entity: eid(),
            at: UniverseTick(9),
        });
        let ev_bytes = postcard::to_allocvec(&ev).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ServerControlMsg>(&ev_bytes).expect("decode"),
            ev
        );
    }

    /// One full-width composed row for the scene-message fixtures: every field non-default
    /// (a dropped field cannot pass as a lucky zero), the bag a real `vd_core::look` blob.
    fn scene_row() -> SceneRow {
        use vd_core::glam::DVec3;
        SceneRow {
            realm: RealmId::Planet(7),
            parent: Some(RealmId::System(7)),
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 7 },
                DVec3::new(20.0, -3.0, 5.0),
                UniverseTick(100),
            ),
            bag: vd_core::look::look_bag(&Boundary::Shell { r: 10.0 }),
        }
    }

    #[test]
    fn the_composed_level_roundtrips_with_its_origin_marker_and_empty_is_render_neutral() {
        // The proto_minor-18 flag day (window_lane.md §2.4, owner item 9: reshape IN PLACE — no
        // shims, no dual-decode; the floor moved so no pre-flag-day peer is ever served): the
        // level carries the origin marker + epoch ON itself and complete drawable rows.
        let reg = ServerControlMsg::RealmRegistry {
            origin: RealmId::System(7),
            origin_epoch: 3,
            rows: vec![scene_row()],
        };
        let bytes = postcard::to_allocvec(&reg).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ServerControlMsg>(&bytes).expect("decode"),
            reg
        );
        // An EMPTY level round-trips (the byte-cheap-when-empty precedent — a session whose
        // windows have not confirmed yet gets a well-formed empty scene, never a decode fault).
        let empty = ServerControlMsg::RealmRegistry {
            origin: RealmId::System(7),
            origin_epoch: 1,
            rows: Vec::new(),
        };
        let empty_bytes = postcard::to_allocvec(&empty).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ServerControlMsg>(&empty_bytes).expect("decode"),
            empty
        );
        // The row's bag is the REAL look codec: TAG_LOOK decodes back to the outline (presence
        // IS the body/marker gate — the client never branches on a realm kind). Read off the
        // fixture directly (HR5: destructuring the value constructed above would carry an
        // unreachable refusal arm).
        assert_eq!(
            vd_core::look::look_of(&scene_row().bag),
            Ok(Boundary::Shell { r: 10.0 })
        );
    }

    #[test]
    fn the_composed_delta_roundtrips_at_its_epoch_and_empty_is_neutral() {
        // The reliable incremental half of the composed lane (same flag day, same citation): a
        // delta names its origin + epoch so the client can refuse one from a scene it no longer
        // (or does not yet) hold.
        let delta = ServerControlMsg::RealmSceneDelta {
            origin: RealmId::System(7),
            origin_epoch: 3,
            added: vec![scene_row()],
            removed: vec![RealmId::Planet(8)],
        };
        let bytes = postcard::to_allocvec(&delta).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ServerControlMsg>(&bytes).expect("decode"),
            delta
        );
        // The EMPTY delta round-trips (the codec is total — though the server never SENDS one).
        let empty = ServerControlMsg::RealmSceneDelta {
            origin: RealmId::System(7),
            origin_epoch: 3,
            added: Vec::new(),
            removed: Vec::new(),
        };
        let empty_bytes = postcard::to_allocvec(&empty).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ServerControlMsg>(&empty_bytes).expect("decode"),
            empty
        );
    }

    #[test]
    fn a_realm_shape_is_pure_self_description_with_no_position_field() {
        // The flag-day cure is STRUCTURAL: the type has no field that could state a position in
        // anyone's frame (window_lane.md §2.4; owner item 9). A shape round-trips as identity +
        // geometry + parent link and nothing else.
        let shape = RealmShape {
            realm: RealmId::Planet(7),
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            shape: Boundary::Shell { r: 10.0 },
            parent: Some(RealmId::System(7)),
        };
        let bytes = postcard::to_allocvec(&shape).expect("encode");
        assert_eq!(
            postcard::from_bytes::<RealmShape>(&bytes).expect("decode"),
            shape
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
    fn realm_snapshot_datagram_roundtrips_and_is_empty_at_static_scale() {
        // FA-2a: the RealmId-keyed observer carrier round-trips a moving-realm placement
        // (two DISTINCT RealmId arms + poses in their own frames), and the STATIC-scale case
        // (no moving/renderable child) carries an EMPTY realm list — the byte-identity plant.
        let populated = RealmSnapshotDatagram {
            sub: SubId(4),
            frame_id: 77,
            source_tick: TickId(9),
            universe_tick: UniverseTick(3000),
            origin_epoch: 2,
            realms: vec![
                RealmSnap {
                    realm: RealmId::Planet(7),
                    frame: FrameRef::PlanetCentered { planet_seed: 7 },
                    pose: StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 7 },
                        DVec3::new(1.496e11, 0.0, 0.0),
                        UniverseTick(3000),
                    ),
                },
                RealmSnap {
                    realm: RealmId::Station(3),
                    frame: FrameRef::StationLocal { station_seed: 3 },
                    pose: StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 7 },
                        DVec3::new(0.0, 2.0e8, 0.0),
                        UniverseTick(3000),
                    ),
                },
            ],
        };
        let bytes = postcard::to_allocvec(&populated).expect("encode");
        let back = postcard::from_bytes::<RealmSnapshotDatagram>(&bytes).expect("decode");
        assert_eq!(back, populated);

        // proto_minor 8: the row is a self-describing edge, so the HEAD survives the wire on its own
        // and — on a real parent/child pair — DIFFERS from the tail. This is the assertion that catches
        // somebody "helpfully" restoring a same-frame label: a row whose head equals its tail is a realm
        // claiming to be its own parent, and a receiver reading it would compose the placement twice.
        assert_eq!(
            back.realms[0].frame,
            FrameRef::PlanetCentered { planet_seed: 7 },
            "the HEAD is the CHILD's own frame — not derivable from `realm` (FrameRef::realm is lossy)"
        );
        assert_eq!(
            back.realms[0].pose.frame,
            FrameRef::SystemSpace { system_seed: 7 },
            "the TAIL is the authoring PARENT's own frame"
        );
        assert_ne!(
            back.realms[0].frame, back.realms[0].pose.frame,
            "head and tail differ on every real row: a planet is not its own star system"
        );
        assert_ne!(back.realms[1].frame, back.realms[1].pose.frame);

        // The zero-signal / static-scale case: an empty realm list still round-trips (and is
        // what a walk-scale shard would build — FA-2c never SENDS it, so zero bytes on the wire).
        let empty = RealmSnapshotDatagram {
            sub: SubId(4),
            frame_id: 78,
            source_tick: TickId(10),
            universe_tick: UniverseTick(3001),
            origin_epoch: 2,
            realms: Vec::new(),
        };
        let bytes = postcard::to_allocvec(&empty).expect("encode");
        assert_eq!(
            postcard::from_bytes::<RealmSnapshotDatagram>(&bytes).expect("decode"),
            empty
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

        let ev = EventMsg::EntityRemoved {
            entity: eid(),
            at: UniverseTick(77),
        };
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

    fn realm_snap(n: u32) -> RealmSnap {
        RealmSnap {
            realm: RealmId::Planet(u64::from(n)),
            frame: FrameRef::PlanetCentered {
                planet_seed: u64::from(n),
            },
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 7 },
                DVec3::new(f64::from(n) * 1.0e9, 0.0, 0.0),
                UniverseTick(10),
            ),
        }
    }

    #[test]
    fn partition_realms_chunks_under_budget_and_is_empty_for_no_realms() {
        // FA-2c: the realm partitioner shares the entity partitioner's MTU discipline (DRY —
        // `partition_rows`/`chunk_boundaries`). Empty in ⇒ empty out (a walk-scale shard authors no moving
        // realm ⇒ no datagram); a tight budget forces >1 chunk, each a real RealmSnapshotDatagram within
        // budget, losing no realm; a roomy budget keeps one chunk.
        assert_eq!(partition_realms(&[], 1000), Vec::<Vec<RealmSnap>>::new());
        let realms: Vec<RealmSnap> = (0u32..40).map(realm_snap).collect();
        let budget = 300;
        let chunks = partition_realms(&realms, budget);
        assert!(chunks.len() > 1, "a 40-realm system must partition");
        let mut seen = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(!chunk.is_empty(), "no empty chunk");
            let datagram = RealmSnapshotDatagram {
                sub: SubId(0),
                frame_id: i as u64,
                source_tick: TickId(1),
                universe_tick: UniverseTick(10),
                origin_epoch: 0,
                realms: chunk.clone(),
            };
            let encoded = postcard::to_allocvec(&datagram).expect("encode").len();
            assert!(
                encoded <= budget,
                "chunk {i} encodes to {encoded} > budget {budget}"
            );
            seen.extend(chunk.iter().map(|s| s.realm));
        }
        assert_eq!(seen, realms.iter().map(|s| s.realm).collect::<Vec<_>>());
        assert_eq!(partition_realms(&realms, 100_000).len(), 1);
    }
}
