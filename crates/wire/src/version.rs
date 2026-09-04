//! Wire protocol version negotiation (`docs/design/connection_plane.md` §6.2).
//!
//! `Hello{proto_major, proto_minor}` is exchanged once per connection; the connection
//! is then PINNED to the negotiated version for its whole life — a rolling upgrade
//! drains old connections, it never changes schema mid-connection.
//!
//! - `proto_major` must match exactly; a mismatch is a typed refusal, never a guess.
//! - `proto_minor` is forward/backward compatible by ONE mechanism: APPENDED ENUM
//!   VARIANTS gated on the negotiated minor (a sender never emits a variant the
//!   peer's minor predates). NOTE: adding a trailing FIELD to a struct/variant is
//!   NOT additive in postcard — postcard is non-self-describing, so `#[serde(default)]`
//!   does NOT help (a new decoder hits `DeserializeUnexpectedEnd` on old bytes; an
//!   old decoder desyncs on the trailing bytes). New data rides a new trailing
//!   variant, never a new field.

use serde::{Deserialize, Serialize};

/// Breaking-change generation of the whole wire contract.
pub const PROTO_MAJOR: u16 = 1;
/// Additive revision within the major — minor 1 added `ServerControlMsg::UniverseRate`;
/// minor 2 added `ServerControlMsg::OwnEntity` (the pure-renderer own-entity signal);
/// minor 3 appended `to_parent: Option<RealmId>` to the shard↔orch crossing carriers
/// (`CrossingRequest`/`TransientCrossingRequest`/`TransientCrossingGrant`) — the parent-provenance the
/// "Area label never flips" fix threaded to a consumer (`rebind_pose_to_dest`) that has since been
/// DELETED (D-PLACE-1): the field is now a ★DEAD carried-but-unread tombstone (the receiver forms an
/// `Area` frame from its own roster), owed a flag-day removal (D-WIRE-1). NOTE these are
/// `InterShardFlow` (mesh) carriers, whose whole cluster runs ONE build in
/// dev-greenfield; the field-append is version-visible here for the release-conformance ledger, not for a
/// mixed-minor mesh negotiation (which does not exist — the negotiated minor gates the CLIENT↔gateway
/// `ServerControlMsg` variants).
///
/// minor 4 appended `InterShardFlow::ShardPresence` — the RLM reactive greeting (a demand-spawned shard
/// greets its gateway so the mesh learns the return connection, replacing the dev address pre-book). Like
/// minor 3 this is a mesh carrier (one cluster build), version-visible for the release-conformance ledger.
/// (The RLM `PeerLocate`/`PeerLocated` address-lookup pair, if it lands, takes minor 5 — it was earlier
/// pencilled at 4, but the greeting lands first and `PeerLocate` remains user-gated + may be declined.)
///
/// minor 5 appended `ServerControlMsg::RealmRegistry` — the wire-delivered realm render-scene (VU): the gateway
/// ships the AoI-scoped realm SHAPES so a fully-agnostic client draws its world from the STREAM ALONE, retiring
/// the `--realm-boxes` boot file as the networked source. (This took the minor slot earlier pencilled for
/// `PeerLocate`, which remains user-gated + may be declined.)
///
/// minor 6 appended `ServerControlMsg::RealmSceneDelta` + `ShardToGateway::RealmSceneDelta` — the INCREMENTAL
/// realm render-scene (VU AoI): the scene now FOLLOWS the client's view continuously (a realm streams IN as it
/// enters AoI, OUT as it leaves), so login/walk/warp are all "AoI membership changed". `RealmRegistry` remains
/// the cold-start / warp re-anchor snapshot; the delta is the per-observer add/remove on top. The
/// `ShardToGateway` arm is a mesh carrier (one cluster build), version-visible for the release-conformance
/// ledger; the `ServerControlMsg` arm is the negotiated client-facing variant (gated on minor >= 6).
///
/// minor 7 (floating-origin A5 — THE FLIP) changes the VALUE SEMANTICS with no shape change: entity and realm
/// positions are now ROOT-ABSOLUTE (the server composes and ships finished positions; the client range-reduces
/// against a server-told render origin instead of composing). `ServerControlMsg::{RealmRegistry,RealmSceneDelta}`
/// additively carry that origin (`pin`/`pin_abs`/`anchor_epoch`). This MUST refuse a mismatched peer loudly:
/// client and server are separate binaries, and a stale minor-6 client would compose an already-absolute pose
/// and render every orbit twice — a teleport that grows with the orbit. Not optional (verifier JUMP-7).
///
/// minor 8 (the frame-anchored placement edge) appends `frame: FrameRef` to `RealmSnap` — the edge HEAD
/// (the CHILD realm's own frame) next to the existing tail (`pose.frame`, the authoring parent's frame)
/// and value. A struct FIELD append is NOT postcard-additive (see the module header above), so this is a
/// FLAG DAY: a minor-7 decoder meeting a minor-8 realm datagram does not lose one field, it desyncs the
/// rest of that row and every row after it. That is why minor 8 introduces [`PROTO_MINOR_FLOOR`] and
/// [`ProtoVersion::negotiate`] REFUSES below it. It also discharges what minor 7 promised and never
/// implemented: minor 7 changed position SEMANTICS with no shape change and said a mismatched peer must
/// be refused loudly, but no refusal was ever written — so a stale client silently rendered every orbit
/// twice instead of being told to update.
///
/// minor 8 also REMOVES `pin_abs` AND `anchor_epoch` from `RealmRegistry` and `RealmSceneDelta`. Those
/// messages used to hand the client the pin realm's own absolute position so it could subtract it from
/// every incoming position, plus a counter that bumped whenever that anchor moved. There are no absolute
/// positions any more — no realm is entitled to know where it sits, so nothing can state one — and every
/// position now arrives already measured from the centre of the realm it is being described to, each
/// level's subtraction made by the parent that authored the placement. `pin_abs` is dropped rather than
/// kept at zero: a field that must always be zero is an invitation to refill it, and refilling it would
/// restore the double subtraction. `anchor_epoch` goes with it because an anchor epoch is only meaningful
/// to a receiver that RE-DERIVES its scene when the anchor moves; this receiver derives nothing, reads
/// neither field, and was shipped a hard-coded `0` at every emit site for its whole life. The remaining
/// `pin` is kept: it NAMES a space and carries no number, which costs one enum tag and re-seeds nothing.
/// Both removals ride the same flag day as the `RealmSnap` field append, so they need no separate floor.
///
/// THE DELIBERATE KEEP, recorded because it is the decision and not an oversight: `RealmSnap.frame` — the
/// field that forced this floor — STAYS on the client-facing datagram even though no client reads it
/// (`RealmView::on_realm_snapshot` reads `realm` and `pose`). The reason is that there is no separate
/// client-facing datagram to keep it off: a `RealmSnapshotDatagram` travels parent→child as
/// `InterShardFlow::RealmCascade` and the receiving level forwards those EXACT BYTES to its own gateway as
/// `ShardToGateway::RealmFrame` without decoding them. A leaner client row would make the bottom level
/// decode every row, strip a field and re-encode — the one thing a leaf is defined by not doing, and what
/// its `cascade_rows_converted == 0` measures. The head is what the parent-to-parent hop uses to drop the
/// recipient's own row (head == the child being shipped to) and it is not derivable from `realm`
/// (`FrameRef::realm` is lossy). MEASURED price on the real type: 2 B per row for a `PlanetCentered` head,
/// 3 B for an `AreaLocal` one at shipped-forest seeds, 21 B at full-width `u64` seeds — on rows bounded by
/// MOVING DIRECT CHILDREN, tens, never by entities.
/// If it is ever to leave the client lane, it leaves as a split row type with its own minor and its own
/// flag day — not by being quietly dropped here.
/// **9** appends `InterShardFlow::ShardRoster` — the orchestrator telling a router which nodes the
/// ownership record shows holding a realm, so node class stops being inferred from a transfer's claim or
/// asserted by the node itself. Purely a server↔server addition: no client-facing message changed, so the
/// floor below does NOT move and every existing client negotiates exactly as before. Retroactively
/// owner-approved 2026-08-15 (docs/design/owner_decisions_2026-08-15.md item 7).
/// **10** appends `InterShardFlow::ChildLive` (the SL7 occupancy bit, child→parent — Step 5 slice A)
/// and `InterShardFlow::RealmObservation` (a live child's own authored rows, one hop up, for the parent
/// to restate and re-fan — the "planets freeze when I exit the system" cure, owner-approved 2026-08-13).
/// Purely server↔server additions again: no client-facing message changed, the floor does not move.
/// **11** appends `InterShardFlow::RealmShapeObservation` (a live child's interior OUTLINES, one hop up
/// — the STATIC half of the minor-10 observation lane, so a neighbour realm's interior gets scene boxes
/// for the minor-10 rows to animate: the "approaching a star, its planets never appear" cure) and
/// `InterShardFlow::ChildSceneSet` (the per-LIVE-CHILD rekey of the down-reflected sibling scene — the
/// occupant-keyed `ProxySceneSet` stops being emitted; its `AccountId` leaves the wire). Purely
/// server↔server again: the client keeps receiving the same `RealmSceneDelta`, the floor does not move.
/// `RealmShapeObservation` is retroactively owner-approved 2026-08-15
/// (docs/design/owner_decisions_2026-08-15.md item 7) — the shape lane is INTERIM: its content evolves
/// to self-authored looks with the observer chain.
/// **12** TOMBSTONES `InterShardFlow::OccupantInterest` and `InterShardFlow::ProxySceneSet` (Step 5
/// slice D): the per-occupant lanes are DELETED — no producer, no consumer; a received frame counts
/// undecodable. The variants and their payload structs REMAIN because postcard discriminants are
/// positional and may never be renumbered; the discriminants are reserved forever. Nothing is appended
/// and no client-facing message changed, so the floor does not move. The tombstones are owner-approved
/// 2026-08-12 (docs/design/step5_sl7_lane_deletion.md §8 answer 1: "the four pose-carrying arms
/// tombstone").
/// **13** TOMBSTONES `InterShardFlow::EntityInterest` and `InterShardFlow::EntityCascade` (Step 5
/// slice E, owner-approved): the entity lane is DELETED — occupant poses no longer cross a realm
/// boundary at steady state (SL2); a bystander sees the occupied realm itself as its occupants'
/// proxy (SL7). Same tombstone discipline as minor 12: shapes kept, discriminants reserved forever,
/// received frames count undecodable. A client's own-shard feed (the untouched client lane) is
/// unchanged, so the floor does not move.
/// **14** appends THE REMOVE MESSAGE (D-4(a), owner-picked): `ServerControlMsg::Event(EventMsg)` —
/// the reliable per-entity eviction (`EventMsg::EntityRemoved{entity, at}`) a pure-renderer client
/// needs because a sub close deliberately evicts nothing and absence from a datagram never does —
/// plus its mesh leg `ShardToGateway::EntityRemoved{realm_fence, entity, at}` (one cluster build,
/// ledger-visible). `EventMsg` itself was reshaped in place (`EntityRemoved` gained `at`, the
/// client's resurrect guard) — LAWFUL, uniquely, because the enum had been declared but unroutable
/// since P1.5: no producer ever existed, so no negotiated wire ever carried its old shape. Emitted
/// only to a peer that negotiated minor >= 14; older peers keep the frozen-figure gap this message
/// closes (the slice E accepted-loss escalation). The floor does not move.
/// **15** (Step 5 slice F): `GhostFlow::Spawn` and `GhostFlow::Delta` are TOMBSTONED — the dest→
/// source ghost pose feed is DELETED (every write was a foreign-frame pose into a promotable dot:
/// the §4u corruption at its root) — and `GhostFlow::SpawnV2` is APPENDED: the same take-over proof
/// with the pose gone. The retained ghost emits only while the hand-off hold is open; a bystander's
/// leaver VANISHES at hold closure (the minor-14 remove message, retimed). Mesh-only (one cluster
/// build, ledger-visible); no client-facing message changed, so the floor does not move. SpawnV2 and
/// the Spawn/Delta tombstones are owner-approved 2026-08-12 (docs/design/step5_sl7_lane_deletion.md
/// §8 answer 1).
/// **16** appends THE WINDOW LANE's skeleton (Slice 0 — nothing moves yet: no producer, no consumer):
/// `ShardToGateway::WindowFrame` (per-tick typed child rows + the pre-inverted hop row, one stamp,
/// FireAndForget/Unreliable full-state latest-wins), `ShardToGateway::WindowBody` (the typed
/// look/marker statement, ReDriven/reliable — a lost look is an invisible realm at the no-flicker
/// moment), `ShardToGateway::WindowMembership` (the parent's SL7 verdict as ids, ReDriven on the AoI
/// cadence), and `GatewayToShard::WindowOpen`/`WindowClose` (reliable control; a fan not refreshed
/// within the DERIVED TTL of 2 beats + 1 dies shard-side). `WindowScope` is `Occupants | Child` ONLY —
/// the `Observed` variant NEVER ships (Q2 = parent relay). Mesh-only (one cluster build,
/// ledger-visible): no client-facing message changed, so the floor does not move; the owner-mandated
/// flag day (RealmShape loses `center`, the scene messages reshape, floor moves) is Slice C1's OWN
/// later minor, not this one. The lane is owner-approved 2026-08-15/16, docs/design/window_lane.md §1.1 + §4.5
/// (the five-topic walk + Q1/Q2/Q3 rulings; in-repo record:
/// docs/design/owner_decisions_2026-08-15.md, 2026-08-16 addendum).
/// **17** — THE SIBLING-INTERIOR CARRIER (window lane Slice C1, owner-approved 2026-08-16:
/// docs/design/owner_decisions_2026-08-15.md addendum + docs/design/window_lane.md §5 RULINGS —
/// the Q2 = parent-relay ruling made carrier-real). TOMBSTONES `InterShardFlow::
/// RealmShapeObservation` (disc 32 reserved forever; the interim shape lane's content EVOLVED,
/// exactly as its minor-11 entry promised; variant + payload struct remain, classifications
/// frozen, a received frame counts `undecodable` at its real carrier) and APPENDS its successor
/// pair: `InterShardFlow::WindowRelay` (disc 34 — child→parent, the child's VERBATIM
/// self-authored window statements as SEALED bytes + the child's own fence; the parent's whole
/// lawful vocabulary is forward-or-drop) and `ShardToGateway::WindowRelayed` (disc 10 — the
/// forward leg to a subscriber holding a window on the parent, admitted against the CHILD's
/// identity by the existing predicates). Mesh-only (one cluster build, ledger-visible): no
/// client-facing message changed, the floor does not move.
/// **18** — THE FLAG DAY (window lane Slice C1, `docs/design/window_lane.md` §2.4; owner-approved
/// 2026-08-15/16 — items 1/9/10 of docs/design/owner_decisions_2026-08-15.md + the five-topic
/// walk). ONE owner-mandated in-place reshape (zero deployed clients ⇒ no shims, no dual-decode):
/// `RealmShape` loses `center` (a shape is pure self-description — the one-meaning law);
/// `ServerControlMsg::RealmRegistry` becomes the composed LEVEL `{origin, origin_epoch, rows:
/// Vec<SceneRow>}` with `pin` DELETED (the origin marker is its lawful successor, owner item 9);
/// `ServerControlMsg::RealmSceneDelta` becomes `{origin, origin_epoch, added, removed}`;
/// `SceneRow` is born (a pose in the ORIGIN frame + a tagged skip-unknown TLV bag — the VU
/// streaming contract realized); `RealmSnapshotDatagram` gains `origin_epoch`. Because in-place
/// reshapes are NOT postcard-additive, [`PROTO_MINOR_FLOOR`] moves to 18: a pre-flag-day peer is
/// refused loudly, never served bytes it would mis-frame.
/// **19** — THE DELETION (window lane Slice C2, `docs/design/window_lane.md` §2.5 + §2.9;
/// owner-approved 2026-08-16 — docs/design/window_lane.md §5 RULINGS, including the Q3 ruling
/// that retires the SL1 self-placement filter BY AMENDMENT, together with the lane it guarded,
/// never by silent removal). The four old inter-realm SCENERY lanes are now producer-less and
/// TOMBSTONED: `InterShardFlow::RealmCascade` (disc 26 — the parent's down-cascade of restated
/// rows), `InterShardFlow::RealmObservation` (disc 31 — the child's up-ship the parent restated
/// and re-fanned), `InterShardFlow::ChildSceneSet` (disc 33 — the parent's down-reflect of
/// outlines, the SL1 filter's lane); `RealmShapeObservation` (disc 32) was tombstoned at minor 17.
/// TWO shard→gateway carriers go with them, their producers deleted in the same commit:
/// `ShardToGateway::RealmFrame` (disc 4 — the old per-tick realm datagram; the composed lane has
/// been the client's only scene author since the minor-18 flag day, and its `Occupants`
/// `WindowFrame` subsumes this emit per §2.9 step 3) and `ShardToGateway::RealmSceneDelta` (the
/// per-dot shape push, which shrinks to the ids-only `WindowMembership` verdict per §2.9).
/// Same tombstone discipline as minors 12/13/15/17 throughout: variants AND payload structs
/// remain, discriminants reserved forever, both exhaustive classifications frozen, a received
/// frame counts `undecodable` at its REAL carrier. ZERO arms added; sealing strictly increases —
/// no realm-to-realm scenery data exists any more, and no realm-inbound message type carries a
/// placement or a centre at all (the structural successor to the deleted filter, pinned in
/// `crates/wire/tests/intershard_closed.rs`). Mesh-only (one cluster build, ledger-visible): no
/// client-facing message changed, so the floor does not move.
/// **20** — THE SEALED INTERIOR FORWARD (look horizon slice 3, owner-approved 2026-08-17 —
/// docs/design/look_horizon.md RULINGS + §2 ASK A). A grandchild's OWN sealed picture now
/// travels TWO hops instead of one, opened by no realm on the way: `InterShardFlow::WindowRelay`
/// renames `statements` → `own` IN PLACE (same type, same position — postcard-inert) and
/// APPENDS `interior: Vec<InteriorRelay>` (each entry a held direct child's own sealed batch,
/// VERBATIM, with that child's own fence outside the seal — the zombie-guard graft);
/// `ShardToGateway::WindowRelayed` appends the same `interior` field. The depth bound is the
/// TYPE: `InteriorRelay` has no `interior` field, so a third level is unrepresentable and the
/// carrier arity stays [`crate::session_flow::LOOK_CARRIER_ARITY`] = 2 (the owner's Q3 ruling).
/// The forward gate is the sender's own SL7 in-band verdict (§3.4.5 — the membership gate);
/// the gateway vouches each grandchild against the child's own attested roster, orders its
/// fence, and admits ONLY the author's own picture. Mesh-only (one cluster build,
/// ledger-visible): no client-facing message changed, so the floor does not move. Dark until
/// the slice-4 interest bit wakes the deep realms.
/// **21** — THE INTEREST BIT (look horizon slice 4; Q1 APPROVED, owner-approved 2026-08-17 Q1 —
/// docs/design/look_horizon.md §2 ASK B). ONE new arm, `InterShardFlow::RealmInterest`
/// (disc 35): one byte, two lawful values, parent → ONE direct child, direct to the child's
/// head node on the route the parent already resolves (`ChildRealmNodes`) — never further,
/// never sideways, never through the orchestrator. The first realm-inbound arm since the
/// minor-19 deletion, and it carries NO placement, no centre, no identity, no direction, no
/// count (the `intershard_closed` absence pin covers it by construction of the type). It ends
/// the landed Q2 rationale clause ("am I observed from outside stays unrepresentable in every
/// realm") by the owner's EXPLICIT amendment — a ruling change stated as one, not smuggled.
/// Fail-closed at the receiver (mis-route / unattested sender / stale fence / unlawful value —
/// refused + counted, mirroring the SL7 bit's admission), held under the derived retain TTL,
/// decaying to `0` — "nobody is watching" — on silence. Mesh-only (one cluster build,
/// ledger-visible): no client-facing message changed, so the floor does not move.
/// **22** — THE STAR REALM (celestial taxonomy arc T2; owner-approved 2026-08-19 — rulings D +
/// E of the taxonomy design's §2, docs/design/DEFERRED.md D-TAX rows): ONE new lawful
/// discriminant, `RealmId::Star` (disc 5, APPENDED) with its `FrameRef::StarCentered` (disc 6,
/// APPENDED) — the star becomes a body-bearing child realm of its system, bounded by the
/// dust-sublimation radius, drawn at its own photosphere. NO new message, NO new field, NO
/// `InterShardFlow` arm (the closed-wire goldens are asserted unchanged); the discriminant
/// rides `SceneRow.realm`/`RealmRegistry`, which already cross. postcard writes variant
/// indices as varints, so appending at 5/6 leaves every existing encoding byte-identical —
/// the exact route Station (3) and Area (4) took. THE FLOOR RISES WITH IT (18 → 22, one
/// signature — ruling D verbatim: "raise the floor, we don't have old clients, no need for
/// any checks"): postcard is non-self-describing, so a pre-22 peer meeting a `Star`
/// discriminant fails the whole `RealmRegistry` message, and the two lawful exits were a
/// silently starless sky (a per-session filter) or a loud refusal — the owner chose the
/// refusal, with no filter machinery built.
/// **23** — THE COORDINATE UNIT ENTERS THE HANDSHAKE (slice S3; owner-approved 2026-08-24, Q1
/// condition 2 — `docs/design/owner_decisions_2026-08-24.md`). `ProtoVersion` gains
/// `coordinate_generation`, folded at compile time over the coordinate tier table
/// ([`vd_core::store_stamp::coordinate_generation`]).
///
/// WHY IT IS A FLAG DAY AND NOT AN APPEND. The version rides INSIDE `Hello`, and postcard is
/// positional: a field added to a struct both ends decode is not additive, it re-labels every
/// byte after it. THE FLOOR RISES WITH IT (22 → 23) for that reason alone — there is no
/// serving a pre-23 peer, because its `Hello` does not have the field and would be read as
/// though it did.
///
/// WHY IT EXISTS AT ALL. Slice S8 re-values the galaxy's coordinate step. Two builds that
/// disagree about how many metres one integer step is exchange positions that decode
/// perfectly and are wrong by the ratio between the units — a factor of about a thousand.
/// Nothing crashes, nothing is logged, and a player is simply somewhere else. The refusal has
/// to happen at the handshake, because the first thing a session does is exchange a position.
///
/// The value is DERIVED, never typed, so it moves on exactly the change it exists for and
/// cannot be forgotten. The same fold stamps every durable file (slice S1), so a store and a
/// peer can never disagree about which world they are in.
/// **24** — THE HOP IS THE AUTHORED PLACEMENT (owner-approved 2026-09-02 —
/// `docs/design/owner_decisions_2026-09-02_reach.md` R1/R4, "implement this design"). `HopRow` on
/// `ShardToGateway::WindowFrame` carries the CHILD'S PLACEMENT IN THE AUTHOR'S FRAME, at the
/// author's own step, instead of the author's frame pre-inverted into the child's step. Same
/// type, same bytes, opposite meaning — so it is a FLAG DAY on the mesh, not an append.
/// WHY. The pre-inverted form could not be stated at galaxy scale: a galaxy's origin in a star
/// system's millimetre step has no lattice count, so the galaxy shard refused its own hop every
/// tick and the observer chain never reached the galaxy level. The authored form is representable
/// on every hop in the world, because a parent contains its child. The gateway now inverts once,
/// inside the one routine that already maps every roster row, and lifts the observer's own zero up
/// the chain to place the sky in the galaxy's frame — stated as ONE new field,
/// `RealmSnapshotDatagram.sky_anchor` (an `Option<StampedPose>` in the catalogue's frame, before
/// `realms`), by which the client places its one star cloud and never rebuilds it. A struct field
/// is positional in postcard, so that half is a client-facing flag day too. THE FLOOR RISES WITH IT
/// (23 → 24): a peer on either side of this line composes a wrong picture with no error, and one
/// cluster build is the only deployment that exists (the 2026-08-19 ruling D posture, unchanged).
/// **25** — THE EXTERIOR CROSSING REQUEST (the ruler switch, slice 1; owner-approved 2026-09-03 —
/// `docs/design/realm_crossing_plan_2026-09-02.md`). ONE new arm, `InterShardFlow::ExteriorCrossingRequest`
/// (disc 38, APPENDED): a parent asks the orchestrator to move a driven child's EXTERIOR — the
/// authorship of the child's placement, keyed by its `Ship` key — to its own parent or to one of its
/// direct children, after its swept verdict decided the child left it. The third request arm beside
/// the durable and the transient one; it carries no session because an exterior has no client input
/// to cut. MEASURED cause: a hull flown to the end of a star system's millimetre ruler, twice,
/// because a driven child was never a subject of the containment scan. Mesh-only (one cluster build,
/// ledger-visible): no client-facing message changed, so the floor does not move.
/// The same minor carries the rest of the ruler switch's mesh: `TransferAck::SourceFlushed` APPENDS
/// `state: Vec<u8>` (the exterior blob, a mesh flag day for a positional struct — one cluster build)
/// and `InterShardFlow::LineageStated` (disc 39, APPENDED): the one new datum the owner approved, a
/// parent stating an adopted child's lineage to it.
/// **26** — THE SKY FOLLOWS THE HULL (the ruler switch, slice 5; owner-approved 2026-09-03 —
/// `docs/design/realm_crossing_plan_2026-09-02.md` §3.9). ONE new arm, `InterShardFlow::ExteriorMoved`
/// (disc 40, APPENDED): at the exterior CAS the orchestrator tells every session gateway a moved
/// child's new coord, so a session aboard the hull re-derives its chain, its windows move with the
/// hull and its origin epoch bumps. Mesh-only (one cluster build): no client-facing message changed,
/// so the floor does not move.
/// **27** — THE PEER BOOK (D-RLM-6 mechanism C, owner-approved 2026-07-25 by the judge panel decision and
/// again on 2026-09-03 under "implement all"; built 2026-09-03 for the ruler
/// switch's inward leg — `docs/design/realm_crossing_plan_2026-09-02.md` slice 7). TWO new arms on the
/// Membership class: `InterShardFlow::PeerLocate` (disc 41, a node asks its clock's source where a node it
/// has no lane for listens) and `InterShardFlow::PeerLocated` (disc 42, the orchestrator answers from its
/// launch ledger; sixteen octets and a port). Mesh-only: no client-facing message changed, so the floor
/// does not move.
/// **28** — THE HAND-OVER NAMES ITS NODE (owner-approved 2026-09-04, item 4 of the flight plan):
/// `ExteriorMoved` APPENDS `parent_node`, the commit's destination node, so a gateway opens the new
/// parent's window at once. Mesh-only, one cluster build; the floor does not move.
/// **29** — THE REACH (owner-approved: ruling 2026-09-02 R6, the rule of 2026-09-04): `ReachStated` (discriminant
/// 43), a child's visibility reach by size and by light, to its parent, on change. Mesh-only.
pub const PROTO_MINOR: u16 = 29;

/// The OLDEST minor this build will hold a conversation at. Below it, [`ProtoVersion::negotiate`]
/// refuses outright instead of negotiating down.
///
/// The sender-gates-new-variants rule makes a lower negotiated minor safe only while every change since
/// is an APPENDED ENUM VARIANT the sender can withhold. Minor 18 is not that: it is the window lane's
/// owner-mandated flag day — client-facing payloads RESHAPED IN PLACE (`RealmShape` lost a field, the
/// two scene messages changed shape, the realm datagram gained a field), and postcard is
/// non-self-describing, so there is no gating a reshape out of bytes both ends must agree on
/// byte-for-byte. A pre-18 peer therefore cannot be served at all, and the honest failure is a Close
/// naming the floor rather than a stream that decodes into garbage. (The floor's first move was 8 —
/// the `RealmSnap.frame` field append — for the identical reason.) Raise it only alongside a change of
/// the same kind, and say in the ledger above which change forced it.
///
/// **22** (the current floor): the minor-22 `RealmId::Star` discriminant — owner-approved
/// 2026-08-19 (ruling D): a pre-22 peer meeting a `Star` row fails the WHOLE `RealmRegistry`
/// message (postcard is non-self-describing), and the owner ruled the loud refusal over a
/// per-session starless filter: "raise the floor, we don't have old clients, no need for any
/// checks".
/// **23** (the current floor): the minor-23 coordinate-unit field inside `Hello` — see the
/// ledger entry above. A pre-23 peer's `Hello` lacks the field, and a positional encoding
/// reads the bytes after it as though it were there, so the peer cannot be served at all. The
/// owner's standing posture on this class (ruling D, 2026-08-19) is the loud refusal rather
/// than a filter: there are no deployed clients to protect.
pub const PROTO_MINOR_FLOOR: u16 = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtoVersion {
    pub major: u16,
    pub minor: u16,
    /// ★ THE UNIT THIS PEER COUNTS POSITIONS IN (slice S3; owner ruling 2026-08-24 Q1 condition 2).
    ///
    /// Folded over the coordinate tier table, so it moves when — and only when — a unit moves. Two
    /// builds that disagree about how many metres one integer step is would exchange positions that
    /// LOOK valid and are wrong by the ratio between them. Nothing crashes. Nobody is told. A player is
    /// simply somewhere else.
    ///
    /// This is why it is negotiated rather than logged: the refusal must happen BEFORE a position is
    /// exchanged, and a position is exchanged immediately.
    pub coordinate_generation: u64,
}

impl ProtoVersion {
    pub const CURRENT: ProtoVersion = ProtoVersion {
        major: PROTO_MAJOR,
        minor: PROTO_MINOR,
        // Folded at COMPILE TIME from the same table the durable label folds, so the two can never
        // disagree about what this build believes — and so no caller can state it.
        coordinate_generation: vd_core::store_stamp::coordinate_generation(),
    };

    /// A peer speaking THIS build's coordinate unit at a stated major and minor.
    ///
    /// The unit is not a parameter, and that is the point: a caller that could state it could state it
    /// wrongly, and the one value this negotiation exists to protect would become the one value a test
    /// could fake. To drive a MISMATCH, build the struct literally — which reads as the deliberate act
    /// it is.
    #[must_use]
    pub const fn speaking(major: u16, minor: u16) -> ProtoVersion {
        ProtoVersion {
            major,
            minor,
            coordinate_generation: vd_core::store_stamp::coordinate_generation(),
        }
    }

    /// Can we talk to a peer at `theirs`? Major must match; the conversation is then conducted at the
    /// LOWER minor (the sender-gates-new-variants rule) — but never below [`PROTO_MINOR_FLOOR`].
    ///
    /// The floor is checked AFTER the min, so it catches BOTH directions: an old peer talking to this
    /// build, and this build talking to an old peer. Negotiating down past a field-append would leave
    /// both ends encoding a shape the other cannot parse, and postcard would not report it as a bad
    /// field — it would mis-frame the rest of the stream. Refusing is the only honest answer.
    #[must_use]
    pub fn negotiate(self, theirs: ProtoVersion) -> Option<ProtoVersion> {
        if self.major != theirs.major {
            return None;
        }
        let minor = self.minor.min(theirs.minor);
        if minor < PROTO_MINOR_FLOOR {
            return None;
        }
        // ★ THE UNIT IS NOT NEGOTIABLE (slice S3). Major and minor may differ and still talk — that is
        // what a floor and a minimum are for. A UNIT cannot: there is no lower unit two peers can agree
        // to speak, because every position either means metres or it does not. So this is an equality,
        // and it refuses.
        if self.coordinate_generation != theirs.coordinate_generation {
            return None;
        }
        Some(ProtoVersion {
            major: self.major,
            minor,
            coordinate_generation: self.coordinate_generation,
        })
    }

    /// Why [`negotiate`](Self::negotiate) refused `theirs` — the exact sentence a refused peer is
    /// Closed with. Two causes, and they need different words: a major mismatch means "different
    /// protocol generation, this build cannot help you"; a minor below the floor means "same
    /// generation, but your positions would be framed wrong — update". Reported as one message they
    /// look like the same fault, and the operator chases the wrong one. The floor sentence is BUILT from
    /// [`PROTO_MINOR_FLOOR`] rather than spelled out, so raising the floor cannot leave the message
    /// quoting the old number. Cold path (one refused connection), so the allocation is free.
    #[must_use]
    pub fn refusal_reason(self, theirs: ProtoVersion) -> String {
        if self.major != theirs.major {
            "incompatible protocol major version".to_owned()
        } else if self.coordinate_generation != theirs.coordinate_generation {
            // NAMES THE FIELD AND BOTH VALUES (Q1 condition 3). A refusal an operator cannot act on is
            // how a cluster gets deleted instead of diagnosed, and this is the one refusal whose cause
            // is invisible from the outside: both peers are healthy, both are the same version, and
            // every position either of them sends is wrong.
            format!(
                "coordinate units differ: this build counts positions under generation {} and yours \
                 under {}. A position exchanged between them would be silently wrong by the ratio \
                 between the two units, so the connection is refused instead.",
                self.coordinate_generation, theirs.coordinate_generation
            )
        } else {
            format!(
                "protocol minor below the floor ({PROTO_MINOR_FLOOR}): the scene is server-composed from v{PROTO_MAJOR}.{PROTO_MINOR_FLOOR}"
            )
        }
    }
}

impl core::fmt::Display for ProtoVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "v{}.{}", self.major, self.minor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_major_negotiates_to_lower_minor() {
        // The min still wins — but only ABOVE the floor, so this reads at floor+3 / floor+5 rather
        // than the old 3 / 5 (both of which are now refused outright).
        let a = ProtoVersion::speaking(1, PROTO_MINOR_FLOOR + 3);
        let b = ProtoVersion::speaking(1, PROTO_MINOR_FLOOR + 5);
        assert_eq!(
            a.negotiate(b),
            Some(ProtoVersion::speaking(1, PROTO_MINOR_FLOOR + 3))
        );
        assert_eq!(
            b.negotiate(a),
            Some(ProtoVersion::speaking(1, PROTO_MINOR_FLOOR + 3))
        );
    }

    #[test]
    fn major_mismatch_refuses() {
        let a = ProtoVersion::speaking(1, 0);
        let b = ProtoVersion::speaking(2, 0);
        assert_eq!(a.negotiate(b), None);
    }

    #[test]
    fn a_minor_below_the_floor_is_refused_in_both_directions_with_its_own_reason() {
        // Minor 8 appended a FIELD to `RealmSnap`. postcard cannot skip it and a sender cannot gate it
        // out, so a pre-8 peer is not served at a lower minor — it is refused. Before the floor existed
        // this negotiated down happily and the peer went on to mis-frame every realm datagram it decoded.
        let ours = ProtoVersion::CURRENT;
        let below = ProtoVersion::speaking(PROTO_MAJOR, PROTO_MINOR_FLOOR - 1);
        assert_eq!(
            ours.negotiate(below),
            None,
            "an old peer offering us a pre-flag-day minor"
        );
        assert_eq!(
            below.negotiate(ours),
            None,
            "and the same seen from its side"
        );
        // The refusal SENTENCE distinguishes the two causes; reported identically they send an operator
        // hunting a version generation mismatch when the real answer is "your client is stale".
        assert_eq!(
            ours.refusal_reason(below),
            "protocol minor below the floor (24): the scene is server-composed from v1.24"
        );
        assert_eq!(
            ours.refusal_reason(ProtoVersion::speaking(PROTO_MAJOR + 1, PROTO_MINOR)),
            "incompatible protocol major version"
        );
    }

    #[test]
    fn a_peer_counting_positions_in_another_unit_is_refused_and_told_which_unit() {
        // ★ SLICE S3's WHOLE PRODUCT. Slice S8 re-values the galaxy's coordinate step. From that day,
        // two builds can be the same version, both healthy, both talking — and every position between
        // them wrong by about a thousand. Nothing crashes. Nothing is logged. A player is somewhere
        // else.
        //
        // RED BEFORE THIS SLICE: nothing in the handshake compared a unit at all, and a step change
        // moves ZERO bytes — no golden reddens, no decode fails, nothing goes red by itself. Every
        // safeguard here had to be built deliberately, so it is asserted deliberately.
        let ours = ProtoVersion::CURRENT;
        let theirs = ProtoVersion {
            // The SAME protocol, the SAME minor. Only the unit differs — which is exactly the state
            // that is invisible from the outside and is why this must be a refusal.
            coordinate_generation: ours.coordinate_generation ^ 1,
            ..ours
        };

        assert_eq!(
            ours.negotiate(theirs),
            None,
            "a peer counting positions in another unit must be refused — there is no lower unit two \
             builds can agree to speak"
        );
        assert_eq!(
            theirs.negotiate(ours),
            None,
            "and refused from the other side too"
        );

        // AND IT MUST SAY WHICH FIELD, WITH BOTH VALUES (Q1 condition 3). This is the one refusal
        // whose cause is invisible without being told: both peers are healthy and the same version.
        let reason = ours.refusal_reason(theirs);
        assert!(
            reason.contains("coordinate units differ"),
            "the refusal must name the cause: {reason}"
        );
        // SPLIT, not joined with `&&`: a short-circuit hides one side's false branch from the coverage
        // gate, which is a rule this project learned the hard way and writes down.
        assert!(
            reason.contains(&ours.coordinate_generation.to_string()),
            "the refusal must carry OUR unit, or an operator cannot act on it: {reason}"
        );
        assert!(
            reason.contains(&theirs.coordinate_generation.to_string()),
            "and THEIRS, or they cannot tell which side is behind: {reason}"
        );
        // NOT the floor sentence — that would send somebody hunting a version split that is not there.
        assert!(
            !reason.contains("below the floor"),
            "wrong cause reported: {reason}"
        );
    }

    #[test]
    fn the_unit_in_the_handshake_is_the_same_one_that_stamps_a_durable_file() {
        // ONE FOLD, TWO USERS. A store and a peer must never disagree about which world they are in —
        // so the protocol's unit and the saved-data label's unit are the same value, not two values
        // that happen to match today.
        assert_eq!(
            ProtoVersion::CURRENT.coordinate_generation,
            vd_core::store_stamp::coordinate_generation(),
            "the handshake and the durable label must fold the SAME coordinate table"
        );
    }

    #[test]
    fn current_is_self_compatible_and_displays() {
        assert_eq!(
            PROTO_MINOR, 29,
            "minor 29 is THE REACH (owner ruling 2026-09-02 R6, built 2026-09-04): ReachStated (disc 43), \
             a child's reach by size and by light to its parent, on change; \
             minor 28 is THE HAND-OVER NAMES ITS NODE (owner-approved 2026-09-04): ExteriorMoved carries \
             the new parent's node, so a gateway opens the window at once; \
             minor 27 is THE PEER BOOK (D-RLM-6 mechanism C, built 2026-09-03): PeerLocate (disc 41) and \
             PeerLocated (disc 42), a node asking where an unbooked node listens and the orchestrator's \
             answer; \
             minor 26 is THE SKY FOLLOWS THE HULL (the ruler switch, slice 5; owner-approved 2026-09-03): the \
             ExteriorMoved arm (disc 40) — the orchestrator tells every session gateway a moved child's new \
             coord at the exterior CAS; \
             minor 25 is THE RULER SWITCH's mesh (owner-approved 2026-09-03): the ExteriorCrossingRequest \
             arm (disc 38), the exterior blob appended to SourceFlushed, and the LineageStated arm \
             (disc 39) — a parent tells an adopted child its lineage; \
             minor 24 is THE HOP AS THE AUTHORED PLACEMENT (owner ruling 2026-09-02 R1/R4): HopRow \
             carries the child's placement in the author's frame at the author's step, not the \
             author's frame pre-inverted into the child's step — the inverted form has no lattice \
             count at galaxy scale, so the galaxy shard refused its own hop and no chain ever \
             reached the galaxy; the gateway inverts once where it already maps every row, and lifts \
             the observer's zero up the chain to place the sky. Same bytes, opposite meaning: a mesh \
             flag day, and the floor rises to 24 with it; \
             minor 23 is THE COORDINATE UNIT IN THE HANDSHAKE (slice S3; owner-approved 2026-08-24, \
             Q1 condition 2): ProtoVersion gains coordinate_generation, folded at COMPILE TIME over \
             the coordinate tier table — the same fold that stamps every durable file (slice S1), so \
             a store and a peer can never disagree about which world they are in. A FLAG DAY, not an \
             append: the version rides inside Hello and postcard is positional, so the field re-labels \
             every byte after it and THE FLOOR RISES WITH IT (22 → 23). It exists because slice S8 \
             re-values the galaxy's coordinate step, and two builds that disagree about how many \
             metres one integer step is exchange positions that decode perfectly and are wrong by the \
             ratio between the units — nothing crashes, nothing is logged, and a player is simply \
             somewhere else; \
             minor 22 is THE STAR REALM (celestial taxonomy arc T2; owner-approved 2026-08-19 \
             ruling D): RealmId::Star (disc 5) + FrameRef::StarCentered (disc 6) APPENDED — one \
             lawful discriminant inside SceneRow.realm, zero new messages, zero InterShardFlow \
             arms; the floor rises to 22 in the same signature (no old clients, no filter \
             machinery — a pre-22 peer would mis-frame the whole RealmRegistry); \
             minor 21 is THE INTEREST BIT (look horizon slice 4; Q1 APPROVED, owner-approved \
             2026-08-17 Q1 — docs/design/look_horizon.md §2 ASK B): ONE new arm, \
             InterShardFlow::RealmInterest (disc 35) — one byte, two lawful values, parent → ONE \
             direct child on the already-resolved head route; ends the Q2 rationale clause \
             (am-I-observed stays unrepresentable) by the owner's explicit amendment; fail-closed \
             admission mirroring the SL7 bit; decays to 0 on silence. Mesh-only, floor unmoved; \
             minor 20 is THE SEALED INTERIOR FORWARD (look horizon slice 3, owner-approved \
             2026-08-17 — docs/design/look_horizon.md RULINGS + §2 ASK A): \
             InterShardFlow::WindowRelay renames `statements` → `own` in place (postcard-inert) \
             and APPENDS interior: Vec<InteriorRelay> (each a held direct child's own sealed \
             batch VERBATIM, its own fence outside the seal); ShardToGateway::WindowRelayed \
             appends the same field. The depth bound is the TYPE — InteriorRelay has no interior \
             field, so a third level is unrepresentable and LOOK_CARRIER_ARITY stays 2 (the Q3 \
             ruling). Mesh-only, so the floor does not move; \
             minor 19 is THE DELETION (window lane Slice C2, owner-approved 2026-08-16 — \
             window_lane.md §5 RULINGS, Q3 retiring the SL1 self-placement filter BY AMENDMENT \
             together with the lane it guarded): the old inter-realm scenery lanes are \
             producer-less and TOMBSTONED — InterShardFlow::RealmCascade (26), RealmObservation \
             (31), ChildSceneSet (33) — beside RealmShapeObservation (32, minor 17); and their two \
             shard→gateway carriers go with them, producers deleted — ShardToGateway::RealmFrame \
             (4, subsumed by the Occupants WindowFrame per §2.9 step 3) and \
             ShardToGateway::RealmSceneDelta (the per-dot shape push, shrunk to the ids-only \
             WindowMembership verdict). Zero arms added; mesh-only, so the floor does not move; \
             minor 18 is THE FLAG DAY (window lane Slice C1, owner-approved 2026-08-15/16 items \
             1/9/10): RealmShape lost `center` (pure self-description), RealmRegistry became the \
             composed level {{origin, origin_epoch, rows: Vec<SceneRow>}} with `pin` deleted (the \
             origin marker is its lawful successor), RealmSceneDelta became {{origin, origin_epoch, \
             added, removed}}, SceneRow was born (pose in the ORIGIN frame + tagged skip-unknown \
             TLV bag), RealmSnapshotDatagram gained origin_epoch — in-place reshapes, so the floor \
             moved to 18; \
             minor 17 tombstoned InterShardFlow::RealmShapeObservation (its interim content EVOLVED \
             as the minor-11 entry promised — owner-approved 2026-08-16, \
             owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS) and appended the \
             Q2-relay pair: InterShardFlow::WindowRelay (sealed verbatim child statements, one hop \
             up, forward-or-drop) + ShardToGateway::WindowRelayed (the forward leg, admitted \
             against the CHILD's identity); \
             minor 16 appended THE WINDOW LANE's skeleton (owner-approved 2026-08-15/16, \
             docs/design/window_lane.md §1.1 + §4.5): ShardToGateway::WindowFrame/WindowBody/\
             WindowMembership + GatewayToShard::WindowOpen/WindowClose, WindowScope = Occupants | \
             Child ONLY (Q2 = parent relay; Observed never ships); \
             minor 15 TOMBSTONED the ghost pose feed (GhostFlow::Spawn + Delta) and appended \
             GhostFlow::SpawnV2; minor 14 appended the REMOVE MESSAGE (D-4(a)); minor 13 TOMBSTONED \
             the entity lane (EntityInterest + EntityCascade); minor 12 TOMBSTONED the per-occupant \
             lanes (OccupantInterest + ProxySceneSet); minor 11 appended RealmShapeObservation + \
             ChildSceneSet; minor 10 appended ChildLive + RealmObservation; minor 9 appended \
             ShardRoster; minor 8 appended the edge HEAD (RealmSnap.frame), dropped pin_abs + \
             anchor_epoch, and introduced PROTO_MINOR_FLOOR; minor 7 floating-origin A5; minor 6 \
             RealmSceneDelta, minor 5 RealmRegistry, minor 4 ShardPresence, minor 3 to_parent, \
             minor 2 OwnEntity, minor 1 UniverseRate"
        );
        assert_eq!(
            PROTO_MINOR_FLOOR, 24,
            "the floor tracks the last break a peer cannot be served across, and minor 24 IS one \
             (owner-approved 2026-09-02, R1/R4): HopRow keeps its bytes and reverses its meaning — \
             the child's placement in the author's frame where the author's frame in the child's \
             used to be — so a peer on either side of the line composes a WRONG picture with no \
             error at all. There is no serving such a peer, and the standing posture on this \
             class is the loud refusal rather than a filter: one cluster build, no shims, no \
             checks (the 2026-08-19 ruling D posture). Do not raise this again except alongside a \
             change of the same kind, named in the ledger."
        );
        assert_eq!(
            ProtoVersion::CURRENT.negotiate(ProtoVersion::CURRENT),
            Some(ProtoVersion::CURRENT)
        );
        assert_eq!(ProtoVersion::CURRENT.to_string(), "v1.29");
        // These USED to negotiate (17/16 fully; 8 as the previous floor). They are now refused:
        // the sender-gates-variants rule only covers appended VARIANTS, and minor 18 reshaped
        // payloads in place. This flip IS the proof the floor is live — asserting `Some` here is
        // what would ship a stale client a scene stream it decodes into garbage.
        for stale in [17u16, 16, 8, 0] {
            assert_eq!(
                ProtoVersion::CURRENT.negotiate(ProtoVersion::speaking(1, stale)),
                None,
                "a minor-{stale} peer is below the floor"
            );
        }
    }

    /// THE APPROVAL-CITATION GATE (owner ruling 2026-08-15, docs/design/owner_decisions_2026-08-15.md
    /// item 7): every version-ledger minor entry from 9 upward must carry an owner citation — a new
    /// wire minor documented without one FAILS THE BUILD. The ledger IS the doc comment on
    /// [`PROTO_MINOR`], so the gate reads this file's own source (the same discipline as the
    /// router-converter scan below) and parses the entries off their real structure: an entry starts
    /// at a doc line opening with `**N**` (the style every entry since 9 uses) or `minor N` (the
    /// pre-9 style, kept so the parser sees the whole ledger and the completeness check below stays
    /// honest). ACCEPTED MARKERS — the minimal set the entries actually use: `owner-approved` (which
    /// "retroactively owner-approved" contains) and `owner-picked`. Two asserts, both load-bearing:
    /// (1) COMPLETENESS — minors 9..=PROTO_MINOR each own exactly one parseable entry, in order, so a
    /// bump cannot dodge the gate by writing an entry the parser cannot see (or none at all);
    /// (2) CITATION — no gated entry lacks a marker.
    ///
    /// The helpers are exercised on named examples for BOTH answers of every arm (HR5: a green tree
    /// alone would leave the guilty arms unrun).
    fn ledger_entry_start(line: &str) -> Option<u16> {
        let text = line.trim_start().strip_prefix("/// ")?;
        if let Some(rest) = text.strip_prefix("**") {
            let (num, _) = rest.split_once("**")?;
            return num.parse().ok();
        }
        if let Some(rest) = text.strip_prefix("minor ") {
            let first = rest.split_whitespace().next()?;
            return first.parse().ok();
        }
        None
    }

    /// The contiguous doc block sitting directly on `pub const PROTO_MINOR` — the ledger itself.
    fn ledger_block(source: &str) -> Vec<String> {
        let lines: Vec<&str> = source.lines().collect();
        let const_ix = lines
            .iter()
            .position(|l| l.starts_with("pub const PROTO_MINOR:"))
            .expect("version.rs declares PROTO_MINOR");
        let block: Vec<String> = lines[..const_ix]
            .iter()
            .rev()
            .take_while(|l| l.trim_start().starts_with("///"))
            .map(|l| (*l).to_owned())
            .collect();
        block.into_iter().rev().collect()
    }

    /// Split the ledger block into `(minor, entry text)` rows. Lines before the first entry are the
    /// ledger preamble and belong to no entry.
    fn ledger_entries(block: &[String]) -> Vec<(u16, String)> {
        let mut entries: Vec<(u16, String)> = Vec::new();
        for line in block {
            if let Some(minor) = ledger_entry_start(line) {
                entries.push((minor, line.clone()));
            } else if let Some((_, text)) = entries.last_mut() {
                text.push('\n');
                text.push_str(line);
            }
        }
        entries
    }

    /// The citation classifier — true means the entry lacks every accepted marker.
    fn uncited(entry: &str) -> bool {
        !entry.contains("owner-approved") && !entry.contains("owner-picked")
    }

    #[test]
    fn ledger_entry_start_reads_both_entry_styles_and_refuses_the_rest() {
        // The two real styles.
        assert_eq!(ledger_entry_start("/// **9** appends ShardRoster"), Some(9));
        assert_eq!(
            ledger_entry_start("/// minor 4 appended ShardPresence"),
            Some(4)
        );
        // Refusals, one per arm: not a doc line; a doc line that starts no entry; bold that is
        // not a number; an unterminated bold opener; a non-numeric minor; a bare "minor ".
        assert_eq!(ledger_entry_start("pub const PROTO_MINOR: u16 = 15;"), None);
        assert_eq!(ledger_entry_start("/// the floor does not move."), None);
        assert_eq!(ledger_entry_start("/// **bold** emphasis"), None);
        assert_eq!(ledger_entry_start("/// **unterminated"), None);
        assert_eq!(ledger_entry_start("/// minor tweak to prose"), None);
        assert_eq!(ledger_entry_start("/// minor "), None);
    }

    #[test]
    fn ledger_entries_attach_continuations_and_skip_the_preamble() {
        let block: Vec<String> = [
            "/// Additive revision within the major — a preamble line owned by no entry.",
            "/// **9** appends a lane",
            "/// and this wrapped line belongs to entry 9.",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        assert_eq!(
            ledger_entries(&block),
            vec![(
                9,
                "/// **9** appends a lane\n/// and this wrapped line belongs to entry 9."
                    .to_owned()
            )]
        );
    }

    #[test]
    fn uncited_recognises_each_marker_and_flags_a_bare_entry() {
        assert!(!uncited(
            "/// **9** ... retroactively owner-approved 2026-08-15"
        ));
        assert!(!uncited("/// **14** ... (D-4(a), owner-picked)"));
        assert!(uncited(
            "/// **16** appends a lane with no approval on record"
        ));
    }

    #[test]
    fn every_ledger_minor_from_9_up_carries_an_owner_citation() {
        let source = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/version.rs"),
        )
        .expect("the crate's own source is readable");
        let entries = ledger_entries(&ledger_block(&source));
        let gated: Vec<(u16, String)> = entries.into_iter().filter(|(n, _)| *n >= 9).collect();
        // COMPLETENESS: every minor from 9 to current owns exactly one parseable entry, in order.
        // A new PROTO_MINOR bump whose ledger entry the parser cannot see fails HERE, so the
        // citation assert below can never be dodged by malformed (or missing) documentation.
        let minors: Vec<u16> = gated.iter().map(|(n, _)| *n).collect();
        let expected: Vec<u16> = (9..=PROTO_MINOR).collect();
        assert_eq!(
            minors, expected,
            "the version ledger must hold ONE parseable entry per minor from 9 to PROTO_MINOR"
        );
        // CITATION: an entry without an owner marker is a wire change nobody approved.
        let offenders: Vec<u16> = gated
            .iter()
            .filter(|(_, text)| uncited(text))
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(
            offenders,
            Vec::<u16>::new(),
            "every wire minor from 9 up needs an owner citation in its ledger entry \
             (owner ruling 2026-08-15; accepted markers: owner-approved / owner-picked)"
        );
    }

    // The old router-converter doc scan that lived here ("no doc line may name the gateway and
    // a conversion in one breath") guarded the REJECTED world-wide-graph model and is REPLACED,
    // per docs/design/window_lane.md §2.6.1 (ledgered with the owner citation on the minor-17/18
    // entries above): the structural guard is now the dependency gate
    // `tests/tests/crate_isolation.rs::the_window_composer_cannot_name_a_motion_or_generate_a_world`
    // — the connection plane is BUILD-UNABLE to evaluate a placement, so the contract may state
    // the flag-day truth plainly: the composed level's rows are stacked at one tick by the
    // observer's connection process from attested statements, and the client draws them.

    #[test]
    fn serde_roundtrip() {
        let bytes = postcard::to_allocvec(&ProtoVersion::CURRENT).expect("encode");
        let back: ProtoVersion = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, ProtoVersion::CURRENT);
    }
}
