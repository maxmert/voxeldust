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
/// (`CrossingRequest`/`TransientCrossingRequest`/`TransientCrossingGrant`) so an `Area` dest's frame
/// forms — the parent-provenance the re-home threads to `rebind_pose_to_dest` (the "Area label never
/// flips" fix). NOTE these are `InterShardFlow` (mesh) carriers, whose whole cluster runs ONE build in
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
pub const PROTO_MINOR: u16 = 7;

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

    /// Can we talk to a peer at `theirs`? Major must match; the conversation is then
    /// conducted at the LOWER minor (the sender-gates-new-variants rule).
    #[must_use]
    pub fn negotiate(self, theirs: ProtoVersion) -> Option<ProtoVersion> {
        if self.major != theirs.major {
            return None;
        }
        Some(ProtoVersion {
            major: self.major,
            minor: self.minor.min(theirs.minor),
        })
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
        let a = ProtoVersion { major: 1, minor: 3 };
        let b = ProtoVersion { major: 1, minor: 5 };
        assert_eq!(a.negotiate(b), Some(ProtoVersion { major: 1, minor: 3 }));
        assert_eq!(b.negotiate(a), Some(ProtoVersion { major: 1, minor: 3 }));
    }

    #[test]
    fn major_mismatch_refuses() {
        let a = ProtoVersion { major: 1, minor: 0 };
        let b = ProtoVersion { major: 2, minor: 0 };
        assert_eq!(a.negotiate(b), None);
    }

    #[test]
    fn current_is_self_compatible_and_displays() {
        assert_eq!(
            PROTO_MINOR, 7,
            "minor 7 made positions root-absolute + server-told the render origin (floating-origin A5); minor 6 appended RealmSceneDelta, minor 5 RealmRegistry, minor 4 ShardPresence, minor 3 to_parent on the crossing carriers, minor 2 OwnEntity, minor 1 UniverseRate"
        );
        assert_eq!(
            ProtoVersion::CURRENT.negotiate(ProtoVersion::CURRENT),
            Some(ProtoVersion::CURRENT)
        );
        assert_eq!(ProtoVersion::CURRENT.to_string(), "v1.7");
        // Sender-gates-variants: talking to an older minor-1 peer negotiates DOWN to
        // minor 1, so the gateway withholds the minor-2 OwnEntity variant (falling back to
        // the retained-ghost/AuthorityChanged path). An even-older minor-0 peer negotiates
        // down to minor 0, withholding BOTH UniverseRate and OwnEntity.
        let old1 = ProtoVersion { major: 1, minor: 1 };
        assert_eq!(
            ProtoVersion::CURRENT.negotiate(old1),
            Some(ProtoVersion { major: 1, minor: 1 })
        );
        let old0 = ProtoVersion { major: 1, minor: 0 };
        assert_eq!(
            ProtoVersion::CURRENT.negotiate(old0),
            Some(ProtoVersion { major: 1, minor: 0 })
        );
    }

    #[test]
    fn serde_roundtrip() {
        let bytes = postcard::to_allocvec(&ProtoVersion::CURRENT).expect("encode");
        let back: ProtoVersion = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, ProtoVersion::CURRENT);
    }
}
