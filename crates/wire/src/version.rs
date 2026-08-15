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
pub const PROTO_MINOR: u16 = 16;

/// The OLDEST minor this build will hold a conversation at. Below it, [`ProtoVersion::negotiate`]
/// refuses outright instead of negotiating down.
///
/// The sender-gates-new-variants rule makes a lower negotiated minor safe only while every change since
/// is an APPENDED ENUM VARIANT the sender can withhold. Minor 8 is not that: it appended a FIELD to a
/// struct that rides an unreliable datagram, and postcard is non-self-describing, so there is no gating
/// a field out of a shape both ends must agree on byte-for-byte. A pre-8 peer therefore cannot be served
/// at all, and the honest failure is a Close naming the floor rather than a stream that decodes into
/// garbage. This is the protocol's first floor; raise it only alongside a change of the same kind, and
/// say in the ledger above which change forced it.
pub const PROTO_MINOR_FLOOR: u16 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtoVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtoVersion {
    pub const CURRENT: ProtoVersion = ProtoVersion {
        major: PROTO_MAJOR,
        minor: PROTO_MINOR,
    };

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
        Some(ProtoVersion {
            major: self.major,
            minor,
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
        } else {
            format!(
                "protocol minor below the floor ({PROTO_MINOR_FLOOR}): positions are frame-anchored from v{PROTO_MAJOR}.{PROTO_MINOR_FLOOR}"
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
        let a = ProtoVersion {
            major: 1,
            minor: PROTO_MINOR_FLOOR + 3,
        };
        let b = ProtoVersion {
            major: 1,
            minor: PROTO_MINOR_FLOOR + 5,
        };
        assert_eq!(
            a.negotiate(b),
            Some(ProtoVersion {
                major: 1,
                minor: PROTO_MINOR_FLOOR + 3
            })
        );
        assert_eq!(
            b.negotiate(a),
            Some(ProtoVersion {
                major: 1,
                minor: PROTO_MINOR_FLOOR + 3
            })
        );
    }

    #[test]
    fn major_mismatch_refuses() {
        let a = ProtoVersion { major: 1, minor: 0 };
        let b = ProtoVersion { major: 2, minor: 0 };
        assert_eq!(a.negotiate(b), None);
    }

    #[test]
    fn a_minor_below_the_floor_is_refused_in_both_directions_with_its_own_reason() {
        // Minor 8 appended a FIELD to `RealmSnap`. postcard cannot skip it and a sender cannot gate it
        // out, so a pre-8 peer is not served at a lower minor — it is refused. Before the floor existed
        // this negotiated down happily and the peer went on to mis-frame every realm datagram it decoded.
        let ours = ProtoVersion::CURRENT;
        let below = ProtoVersion {
            major: PROTO_MAJOR,
            minor: PROTO_MINOR_FLOOR - 1,
        };
        assert_eq!(
            ours.negotiate(below),
            None,
            "an old peer offering us minor 7"
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
            "protocol minor below the floor (8): positions are frame-anchored from v1.8"
        );
        assert_eq!(
            ours.refusal_reason(ProtoVersion {
                major: PROTO_MAJOR + 1,
                minor: PROTO_MINOR,
            }),
            "incompatible protocol major version"
        );
    }

    #[test]
    fn current_is_self_compatible_and_displays() {
        assert_eq!(
            PROTO_MINOR, 16,
            "minor 16 appended THE WINDOW LANE's skeleton (owner-approved 2026-08-15/16, \
             docs/design/window_lane.md §1.1 + §4.5): ShardToGateway::WindowFrame/WindowBody/\
             WindowMembership + GatewayToShard::WindowOpen/WindowClose, WindowScope = Occupants | \
             Child ONLY (Q2 = parent relay; Observed never ships) — mesh-only, Slice 0, nothing \
             moves yet, the floor stays; \
             minor 15 TOMBSTONED the ghost pose feed (GhostFlow::Spawn + Delta — the foreign-frame \
             pose writer died with it) and appended GhostFlow::SpawnV2, the pose-free take-over \
             proof; minor 14 appended the REMOVE MESSAGE (ServerControlMsg::Event carrying \
             EventMsg::EntityRemoved(entity, at) + the ShardToGateway::EntityRemoved mesh leg — \
             the reliable per-entity eviction + the client resurrect guard, D-4(a)); \
             minor 13 TOMBSTONED the entity lane (EntityInterest + EntityCascade, Step 5 slice E — \
             occupant poses no longer cross a realm boundary at steady state; the occupied realm is \
             its occupants' proxy; no producer, no consumer, discriminants reserved forever, received \
             frames count undecodable); minor 12 TOMBSTONED the per-occupant lanes (OccupantInterest \
             + ProxySceneSet — no \
             producer, no consumer, discriminants reserved forever, received frames count \
             undecodable); minor 11 appended `InterShardFlow::RealmShapeObservation` (a live child's interior OUTLINES \
             one hop up — the static half of the observation lane, the approaching-star invisible-planets \
             cure) and `InterShardFlow::ChildSceneSet` (the per-live-child rekey of the down-reflected \
             sibling scene; the occupant-keyed ProxySceneSet stops being emitted). \
             SERVER-TO-SERVER ONLY again, so the floor stays where it is; minor 10 appended \
             `InterShardFlow::ChildLive` (the SL7 occupancy bit, child→parent, Step 5 \
             slice A) and `InterShardFlow::RealmObservation` (a live child's own authored rows one hop up, \
             for the parent to restate and re-fan — the exit-the-system frozen-planets cure); minor 9 appended \
             `InterShardFlow::ShardRoster` (which nodes the ownership record shows holding a realm); minor 8 appended the edge HEAD (`RealmSnap.frame`) and KEEPS it on the client lane (the cascade bytes ARE the client feed, re-fanned unopened), dropped BOTH the render-origin `pin_abs` and the re-anchor `anchor_epoch` from the scene messages (every position arrives measured from the realm it is described to, and nothing downstream re-derives), and, because a field append is not postcard-additive, introduced PROTO_MINOR_FLOOR; minor 7 made positions root-absolute + server-told the render origin (floating-origin A5); minor 6 appended RealmSceneDelta, minor 5 RealmRegistry, minor 4 ShardPresence, minor 3 to_parent on the crossing carriers, minor 2 OwnEntity, minor 1 UniverseRate"
        );
        assert_eq!(
            PROTO_MINOR_FLOOR, 8,
            "THEY HAVE NOW PARTED COMPANY, exactly as the previous version of this assertion predicted: \
             minor 9 appends a VARIANT on a server-to-server arm, so a minor-8 client is fully correct \
             and must not be refused. The floor tracks the last CLIENT-VISIBLE break — the minor-8 field \
             append — not every bump. Raising it here would turn an internal addition into a flag day for \
             every client, which is the opposite of what an append-only wire is for."
        );
        assert_eq!(
            ProtoVersion::CURRENT.negotiate(ProtoVersion::CURRENT),
            Some(ProtoVersion::CURRENT)
        );
        assert_eq!(ProtoVersion::CURRENT.to_string(), "v1.16");
        // These three USED to negotiate down and be welcomed (minor 7 fully, minor 1 without the
        // minor-2 OwnEntity, minor 0 without that AND UniverseRate). They are now refused: the
        // sender-gates-variants rule only covers appended VARIANTS, and minor 8 appended a FIELD.
        // This flip IS the proof the floor is live — asserting `Some` here is what shipped a stale
        // client a `RealmSnap` stream it would decode into garbage.
        for stale in [7u16, 1, 0] {
            assert_eq!(
                ProtoVersion::CURRENT.negotiate(ProtoVersion {
                    major: 1,
                    minor: stale
                }),
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

    /// THE CONTRACT MAY NOT NAME THE ROUTER AND A CONVERSION IN ONE BREATH.
    ///
    /// This crate's doc comments ARE the wire specification — the last three model changes were each
    /// first believed because a comment said so. `RealmShape::center` used to define itself by WHERE its
    /// value was re-expressed and by WHICH party did it — naming the router as that party in the same
    /// sentence — which gave one field two meanings keyed on sender and re-seeded the router-composes
    /// model in every reader. Deleting that sentence is not enough; nothing stopped it coming back.
    ///
    /// The rule is deliberately blunt and polarity-blind: no doc line in `src/` may mention the gateway AND
    /// a conversion in the same breath, not even to deny it. The wire contract describes WHAT a field means
    /// and who states it; a router's relationship to arithmetic is the router's own documentation. Scanning
    /// the DIRECTORY rather than a fixed include list means a file added later is covered too.
    /// The classifier, kept apart from the scan so BOTH its answers are exercised by a named example
    /// rather than only by whatever happens to be in the tree (a green scan exercises the "no" answer
    /// alone, which would leave the "yes" answer — the one that has to work — never run).
    fn names_the_router_as_a_converter(line: &str) -> bool {
        // Affirmative or negated, singular or plural — the point is that the two ideas never share a line.
        // Stems, not whole words, and this matters: "convert" does NOT contain "conversion", so a list of
        // whole verbs let the exact sentence that started all this through. Found by the positive example
        // below failing, not by reading the list.
        const CONVERSION: [&str; 8] = [
            "conver",
            "compos",
            "re-express",
            "rewrit",
            "restat",
            "subtract",
            "pin space",
            "pin-space",
        ];
        let trimmed = line.trim_start();
        if !trimmed.starts_with("///") && !trimmed.starts_with("//!") {
            return false;
        }
        let lower = line.to_lowercase();
        if !lower.contains("gateway") {
            return false;
        }
        CONVERSION.iter().any(|w| lower.contains(w))
    }

    #[test]
    fn no_doc_line_in_this_crate_names_the_gateway_as_a_converter() {
        // Both answers, on named examples. The second is a REAL line from `ShardToGateway::Frame` — the
        // sentence this whole run's acceptance rests on — so the rule provably does not forbid the
        // contract's own truthful statement about the router.
        assert!(names_the_router_as_a_converter(
            "/// the gateway performs that conversion with the SAME walk"
        ));
        assert!(!names_the_router_as_a_converter(
            "    /// at byte level, and forwards. It never decodes the payload."
        ));

        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        let mut stack = vec![src.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("the crate's own src/ is readable") {
                let path = entry.expect("a readable dir entry").path();
                // Everything under src/ is source; recursing rather than listing means a module added
                // later (or a new `seams/` sibling) is scanned without anyone remembering to add it.
                if path.is_dir() {
                    stack.push(path);
                } else {
                    files.push(path);
                }
            }
        }
        // Non-vacuity: a scan that found nothing would pass for the wrong reason forever. (A plain
        // literal message — an expression inside a passing assert's message is a region no run ever
        // evaluates, HR5.)
        assert!(
            files.len() >= 5,
            "the scan found almost no source files — it is measuring nothing"
        );
        let offences: Vec<String> = files.iter().flat_map(|p| scan_file(p)).collect();
        assert_eq!(
            offences,
            Vec::<String>::new(),
            "the wire contract names the gateway alongside a conversion. Say what the FIELD means and \
             who states it; where a router does or does not do arithmetic belongs in the router's own docs"
        );
    }

    /// One offence, formatted — the shape the scan reports in, exercised by a NAMED example so a
    /// clean tree (where the scan loop pushes nothing) still covers the formatter.
    fn offence_line(path: &std::path::Path, n: usize, line: &str) -> String {
        format!(
            "{}:{}: {}",
            path.file_name().expect("a named file").to_string_lossy(),
            n + 1,
            line.trim()
        )
    }

    /// Every offence of ONE file — the scan's per-file body, extracted so a manufactured guilty file
    /// covers the offence path a clean tree can never take.
    fn scan_file(path: &std::path::Path) -> Vec<String> {
        let text = std::fs::read_to_string(path).expect("a readable source file");
        text.lines()
            .enumerate()
            .filter(|(_, line)| names_the_router_as_a_converter(line))
            .map(|(n, line)| offence_line(path, n, line))
            .collect()
    }

    #[test]
    fn the_offence_report_names_file_one_based_line_and_trimmed_text() {
        assert_eq!(
            offence_line(
                std::path::Path::new("src/channels.rs"),
                3,
                "  guilty line  "
            ),
            "channels.rs:4: guilty line",
        );
        // The per-file scan, against a manufactured GUILTY file — the arm a clean tree cannot take.
        let dir = std::env::temp_dir().join(format!("vd-wire-scan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let guilty = dir.join("guilty.rs");
        std::fs::write(
            &guilty,
            "fn ok() {}\n/// the gateway converts every centre it relays\n",
        )
        .expect("write");
        assert_eq!(
            scan_file(&guilty),
            vec!["guilty.rs:2: /// the gateway converts every centre it relays".to_owned()],
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn serde_roundtrip() {
        let bytes = postcard::to_allocvec(&ProtoVersion::CURRENT).expect("encode");
        let back: ProtoVersion = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, ProtoVersion::CURRENT);
    }
}
