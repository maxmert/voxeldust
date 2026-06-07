//! Credential SHAPES (`docs/design/identity_persistence.md` §2; integration
//! resolution: Ed25519 login validated at the gateway, HMAC session/resume tickets
//! gateway-only, SHARDS NEVER SEE TICKETS). The cryptographic validation lands with
//! P0.8; the wire shapes are frozen here so both sides build against one contract.
//!
//! R7 lesson encoded in the types: the session credential is random, signed,
//! expiring, revocable — and is NOT the cross-shard principal (`SessionId` is,
//! attested over mTLS by the saga, never by the client).

use serde::{Deserialize, Serialize};
use vd_core::{AccountId, EpochId, Fence, SessionId, UniverseTick};

/// Presented once in `Hello`: the strong login identity (Ed25519 over the payload,
/// minted by the auth service).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoginTicket {
    pub account: AccountId,
    pub epoch: EpochId,
    /// Single-use login nonce (replay rejection).
    pub nonce: u64,
    /// Ed25519 signature over (account, epoch, nonce) — validated by the gateway
    /// against the auth-service public key (P0.8).
    pub signature: Vec<u8>,
}

/// The in-session claims the gateway mints and is the ONLY validator of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionClaims {
    pub session: SessionId,
    pub account: AccountId,
    pub epoch: EpochId,
    /// Bumped on logout/ban/credential change: outstanding tickets die mid-session.
    pub validity_epoch: u32,
    /// Absolute expiry against the analytic clock (sliding renewal mints FRESH
    /// tickets on the CURRENT key — old keys drain on schedule).
    pub expires: UniverseTick,
    /// Which gateway-cluster HMAC key signed this (rotation).
    pub key_id: u32,
}

/// Reconnect credential. Single-use: the directory stores the current
/// `resume_nonce`; a successful resume CAS-bumps it, invalidating every prior ticket.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResumeTicket {
    pub claims: SessionClaims,
    /// The session-ownership fence at mint time; adoption CAS-bumps it and the old
    /// gateway self-fences (split-brain adoption is structurally impossible).
    pub session_fence: Fence,
    pub resume_nonce: u64,
    /// HMAC-SHA256 over the payload under the cluster gateway key (P0.8).
    pub hmac: [u8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claims() -> SessionClaims {
        SessionClaims {
            session: SessionId(1),
            account: AccountId(2),
            epoch: EpochId(3),
            validity_epoch: 4,
            expires: UniverseTick(1000),
            key_id: 5,
        }
    }

    #[test]
    fn login_ticket_roundtrips() {
        let t = LoginTicket {
            account: AccountId(7),
            epoch: EpochId(1),
            nonce: 99,
            signature: vec![1; 64],
        };
        let bytes = postcard::to_allocvec(&t).expect("encode");
        assert_eq!(
            postcard::from_bytes::<LoginTicket>(&bytes).expect("decode"),
            t
        );
    }

    #[test]
    fn resume_ticket_roundtrips_with_exact_hmac_width() {
        let t = ResumeTicket {
            claims: claims(),
            session_fence: Fence(6),
            resume_nonce: 42,
            hmac: [9; 32],
        };
        let bytes = postcard::to_allocvec(&t).expect("encode");
        let back: ResumeTicket = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, t);
        assert_eq!(back.hmac.len(), 32);
    }

    #[test]
    fn claims_are_copy_and_comparable() {
        let a = claims();
        let b = a; // Copy
        assert_eq!(a, b);
        let mut c = a;
        c.validity_epoch += 1;
        assert_ne!(a, c, "validity bump invalidates");
    }
}
