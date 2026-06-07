//! Wire protocol version negotiation (`docs/design/connection_plane.md` §6.2).
//!
//! `Hello{proto_major, proto_minor}` is exchanged once per connection; the connection
//! is then PINNED to the negotiated version for its whole life — a rolling upgrade
//! drains old connections, it never changes schema mid-connection.
//!
//! - `proto_major` must match exactly; a mismatch is a typed refusal, never a guess.
//! - `proto_minor` is forward/backward compatible by construction: postcard structs
//!   evolve additively via `#[serde(default)]` on new trailing fields, and enums via
//!   appended variants gated on the negotiated minor (a sender never emits a variant
//!   the peer's minor predates).

use serde::{Deserialize, Serialize};

/// Breaking-change generation of the whole wire contract.
pub const PROTO_MAJOR: u16 = 1;
/// Additive revision within the major.
pub const PROTO_MINOR: u16 = 0;

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
            ProtoVersion::CURRENT.negotiate(ProtoVersion::CURRENT),
            Some(ProtoVersion::CURRENT)
        );
        assert_eq!(ProtoVersion::CURRENT.to_string(), "v1.0");
    }

    #[test]
    fn serde_roundtrip() {
        let bytes = postcard::to_allocvec(&ProtoVersion::CURRENT).expect("encode");
        let back: ProtoVersion = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, ProtoVersion::CURRENT);
    }
}
