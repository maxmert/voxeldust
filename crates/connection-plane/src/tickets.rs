//! Ticket CRYPTOGRAPHY — mint and validate the credential shapes frozen in
//! `vd_wire::seams::tickets` (`docs/design/identity_persistence.md` §2).
//!
//! This module living in `connection-plane` is itself the trust design: the gateway
//! is the ONLY validator of tickets, and shards never depend on this crate — so a
//! shard accepting a ticket is structurally impossible, not merely forbidden
//! (R7's triple-overloaded token cannot be rebuilt by accident).
//!
//! - `LoginTicket`: Ed25519, minted by the auth service, verified here against the
//!   auth service's public key.
//! - `ResumeTicket`: HMAC-SHA256 under a rotating gateway-cluster key ring; the MAC
//!   covers `key_id`, so a key-confusion swap breaks the MAC.
//! - All payloads are domain-separated; constant-time MAC comparison via `Mac::verify`.
//!
//! Replay state (single-use login nonces, the directory's `resume_nonce`) is the
//! CALLER's: this module checks the nonce it is HANDED against the ticket — pulling
//! the current nonce from the directory is gateway session logic (P1/P3).

use std::collections::BTreeMap;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use vd_core::{AccountId, EpochId, Fence, UniverseTick};
use vd_wire::seams::tickets::{LoginTicket, ResumeTicket, SessionClaims};

type HmacSha256 = Hmac<Sha256>;

/// Gateway-cluster HMAC keys are exactly this wide.
pub const HMAC_KEY_BYTES: usize = 32;
/// Ed25519 keys are exactly this wide.
pub const ED25519_KEY_BYTES: usize = 32;

/// Domain-separation prefixes: a MAC/signature minted for one purpose can never
/// validate as another (and a future v2 payload can never collide with v1).
const RESUME_DOMAIN: &[u8] = b"vd-resume-ticket-v1";
const LOGIN_DOMAIN: &[u8] = b"vd-login-ticket-v1";

/// The rotating gateway-cluster key ring. Minting always uses the CURRENT key;
/// validation accepts any key still installed — rotation is `install_key` (new id)
/// then `promote` (mint on it) then, after old tickets drain, `retire` (old id).
#[derive(Clone, Debug)]
pub struct GatewayKeyRing {
    keys: BTreeMap<u32, [u8; HMAC_KEY_BYTES]>,
    current: u32,
}

/// Key-ring management failures (operator errors, loud and typed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum KeyRingError {
    #[error("key id {0} is already installed")]
    DuplicateKeyId(u32),
    #[error("key id {0} is not installed")]
    UnknownKeyId(u32),
    #[error("key id {0} is the current minting key and cannot be retired")]
    RetireCurrent(u32),
}

impl GatewayKeyRing {
    /// A ring with one installed, current key.
    #[must_use]
    pub fn new(current_id: u32, key: [u8; HMAC_KEY_BYTES]) -> GatewayKeyRing {
        let mut keys = BTreeMap::new();
        keys.insert(current_id, key);
        GatewayKeyRing {
            keys,
            current: current_id,
        }
    }

    /// Install a new key (not yet minting). Duplicate ids are rejected: silently
    /// replacing key material would orphan every outstanding ticket on that id.
    ///
    /// # Errors
    /// [`KeyRingError::DuplicateKeyId`] if `id` is already installed.
    pub fn install_key(&mut self, id: u32, key: [u8; HMAC_KEY_BYTES]) -> Result<(), KeyRingError> {
        if self.keys.contains_key(&id) {
            return Err(KeyRingError::DuplicateKeyId(id));
        }
        self.keys.insert(id, key);
        Ok(())
    }

    /// Switch minting to an installed key (validation keeps accepting the others).
    ///
    /// # Errors
    /// [`KeyRingError::UnknownKeyId`] if `id` is not installed.
    pub fn promote(&mut self, id: u32) -> Result<(), KeyRingError> {
        if !self.keys.contains_key(&id) {
            return Err(KeyRingError::UnknownKeyId(id));
        }
        self.current = id;
        Ok(())
    }

    /// Remove a drained key; tickets minted under it become invalid.
    ///
    /// # Errors
    /// [`KeyRingError::RetireCurrent`] for the current minting key,
    /// [`KeyRingError::UnknownKeyId`] if `id` is not installed.
    pub fn retire(&mut self, id: u32) -> Result<(), KeyRingError> {
        if id == self.current {
            return Err(KeyRingError::RetireCurrent(id));
        }
        if self.keys.remove(&id).is_none() {
            return Err(KeyRingError::UnknownKeyId(id));
        }
        Ok(())
    }

    #[must_use]
    pub fn current_key_id(&self) -> u32 {
        self.current
    }

    fn key(&self, id: u32) -> Option<&[u8; HMAC_KEY_BYTES]> {
        self.keys.get(&id)
    }
}

/// Typed resume-ticket rejections. Validation order: key lookup → MAC (nothing past
/// the MAC is trusted) → expiry → revocation epoch → single-use nonce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ResumeReject {
    #[error("ticket key id {0} is not in the ring")]
    UnknownKey(u32),
    #[error("ticket MAC verification failed")]
    BadMac,
    #[error("ticket expired at {expires} (now {now})")]
    Expired {
        expires: UniverseTick,
        now: UniverseTick,
    },
    #[error("ticket validity epoch {got} is revoked (current {current})")]
    Revoked { got: u32, current: u32 },
    #[error("resume nonce is not the directory-current nonce (single-use)")]
    NonceMismatch,
}

/// The exact byte string the resume MAC covers (key_id rides inside `claims`, so it
/// is covered too — key-confusion swaps break the MAC).
fn resume_payload(claims: &SessionClaims, session_fence: Fence, resume_nonce: u64) -> Vec<u8> {
    let body = postcard::to_allocvec(&(claims, session_fence, resume_nonce))
        .expect("plain-integer claims serialize infallibly");
    let mut payload = Vec::with_capacity(RESUME_DOMAIN.len() + body.len());
    payload.extend_from_slice(RESUME_DOMAIN);
    payload.extend_from_slice(&body);
    payload
}

fn mac_under(key: &[u8; HMAC_KEY_BYTES], payload: &[u8]) -> HmacSha256 {
    let mut mac =
        HmacSha256::new_from_slice(key).expect("HMAC-SHA256 accepts any 32-byte key length");
    mac.update(payload);
    mac
}

/// Mint a `ResumeTicket` under the ring's CURRENT key. The handed-in claims'
/// `key_id` is overwritten — the ring, not the caller, decides the signing key.
#[must_use]
pub fn mint_resume(
    ring: &GatewayKeyRing,
    mut claims: SessionClaims,
    session_fence: Fence,
    resume_nonce: u64,
) -> ResumeTicket {
    claims.key_id = ring.current;
    let key = ring
        .key(ring.current)
        .expect("the current key is always installed (retire refuses it)");
    let payload = resume_payload(&claims, session_fence, resume_nonce);
    let mac = mac_under(key, &payload).finalize().into_bytes();
    ResumeTicket {
        claims,
        session_fence,
        resume_nonce,
        hmac: mac.into(),
    }
}

/// Validate a presented `ResumeTicket`. `current_nonce` is the directory's stored
/// single-use `resume_nonce`; `current_validity_epoch` is the account's revocation
/// epoch — the caller pulls both through (hints never decide authority).
///
/// # Errors
/// A typed [`ResumeReject`]; nothing about the ticket is trusted unless `Ok`.
pub fn validate_resume(
    ring: &GatewayKeyRing,
    ticket: &ResumeTicket,
    now: UniverseTick,
    current_validity_epoch: u32,
    current_nonce: u64,
) -> Result<SessionClaims, ResumeReject> {
    let key_id = ticket.claims.key_id;
    let Some(key) = ring.key(key_id) else {
        return Err(ResumeReject::UnknownKey(key_id));
    };
    let payload = resume_payload(&ticket.claims, ticket.session_fence, ticket.resume_nonce);
    if mac_under(key, &payload).verify_slice(&ticket.hmac).is_err() {
        return Err(ResumeReject::BadMac);
    }
    if now >= ticket.claims.expires {
        return Err(ResumeReject::Expired {
            expires: ticket.claims.expires,
            now,
        });
    }
    if ticket.claims.validity_epoch != current_validity_epoch {
        return Err(ResumeReject::Revoked {
            got: ticket.claims.validity_epoch,
            current: current_validity_epoch,
        });
    }
    if ticket.resume_nonce != current_nonce {
        return Err(ResumeReject::NonceMismatch);
    }
    Ok(ticket.claims)
}

/// Typed login-ticket rejections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LoginReject {
    #[error("auth-service verifying key bytes are not a valid Ed25519 point")]
    MalformedKey,
    #[error("ticket signature is not 64 bytes of Ed25519 signature")]
    MalformedSignature,
    #[error("ticket signature does not verify under the auth-service key")]
    BadSignature,
}

/// The exact byte string the login signature covers.
fn login_payload(account: AccountId, epoch: EpochId, nonce: u64) -> Vec<u8> {
    let body = postcard::to_allocvec(&(account, epoch, nonce))
        .expect("plain-integer login fields serialize infallibly");
    let mut payload = Vec::with_capacity(LOGIN_DOMAIN.len() + body.len());
    payload.extend_from_slice(LOGIN_DOMAIN);
    payload.extend_from_slice(&body);
    payload
}

/// Mint a `LoginTicket` (the AUTH SERVICE's half, here for tests and for the auth
/// bin to call — Ed25519 signing is deterministic, no RNG enters).
#[must_use]
pub fn mint_login(
    signing_key: &[u8; ED25519_KEY_BYTES],
    account: AccountId,
    epoch: EpochId,
    nonce: u64,
) -> LoginTicket {
    let key = SigningKey::from_bytes(signing_key);
    let signature = key.sign(&login_payload(account, epoch, nonce));
    LoginTicket {
        account,
        epoch,
        nonce,
        signature: signature.to_bytes().to_vec(),
    }
}

/// Validate a presented `LoginTicket` against the auth service's public key.
/// Login-nonce replay tracking is the caller's (single-use against durable state).
///
/// # Errors
/// A typed [`LoginReject`]; the ticket asserts nothing unless `Ok`.
pub fn validate_login(
    verifying_key: &[u8; ED25519_KEY_BYTES],
    ticket: &LoginTicket,
) -> Result<AccountId, LoginReject> {
    let Ok(key) = VerifyingKey::from_bytes(verifying_key) else {
        return Err(LoginReject::MalformedKey);
    };
    let Ok(signature) = Signature::from_slice(&ticket.signature) else {
        return Err(LoginReject::MalformedSignature);
    };
    let payload = login_payload(ticket.account, ticket.epoch, ticket.nonce);
    if key.verify(&payload, &signature).is_err() {
        return Err(LoginReject::BadSignature);
    }
    Ok(ticket.account)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use vd_core::SessionId;

    const KEY_A: [u8; 32] = [0x11; 32];
    const KEY_B: [u8; 32] = [0x22; 32];

    fn claims() -> SessionClaims {
        SessionClaims {
            session: SessionId(0xCAFE),
            account: AccountId(0xBEEF),
            epoch: EpochId(1),
            validity_epoch: 3,
            expires: UniverseTick(1_000),
            key_id: 999, // mint overwrites this
        }
    }

    fn valid_ticket(ring: &GatewayKeyRing) -> ResumeTicket {
        mint_resume(ring, claims(), Fence(5), 42)
    }

    #[test]
    fn mint_then_validate_roundtrips_and_pins_the_key_id() {
        let ring = GatewayKeyRing::new(1, KEY_A);
        let ticket = valid_ticket(&ring);
        assert_eq!(ticket.claims.key_id, 1, "ring decides the key, not caller");
        let validated =
            validate_resume(&ring, &ticket, UniverseTick(500), 3, 42).expect("valid ticket");
        assert_eq!(validated.session, SessionId(0xCAFE));
        assert_eq!(validated.account, AccountId(0xBEEF));
    }

    #[test]
    fn unknown_key_id_is_rejected_before_anything_else() {
        let ring = GatewayKeyRing::new(1, KEY_A);
        let mut ticket = valid_ticket(&ring);
        ticket.claims.key_id = 7;
        let err = validate_resume(&ring, &ticket, UniverseTick(500), 3, 42).expect_err("rejected");
        assert_eq!(err, ResumeReject::UnknownKey(7));
    }

    #[test]
    fn any_tampered_claim_breaks_the_mac() {
        let ring = GatewayKeyRing::new(1, KEY_A);
        let base = valid_ticket(&ring);
        let tampered: Vec<ResumeTicket> = vec![
            ResumeTicket {
                claims: SessionClaims {
                    session: SessionId(0xDEAD),
                    ..base.claims
                },
                ..base.clone()
            },
            ResumeTicket {
                claims: SessionClaims {
                    account: AccountId(0xD00D),
                    ..base.claims
                },
                ..base.clone()
            },
            ResumeTicket {
                claims: SessionClaims {
                    expires: UniverseTick(9_999),
                    ..base.claims
                },
                ..base.clone()
            },
            ResumeTicket {
                claims: SessionClaims {
                    validity_epoch: 4,
                    ..base.claims
                },
                ..base.clone()
            },
            ResumeTicket {
                session_fence: Fence(6),
                ..base.clone()
            },
            ResumeTicket {
                resume_nonce: 43,
                ..base.clone()
            },
        ];
        for ticket in tampered {
            let err = validate_resume(&ring, &ticket, UniverseTick(500), 4, 43)
                .expect_err("tampered ticket rejected");
            assert_eq!(err, ResumeReject::BadMac);
        }
    }

    #[test]
    fn expiry_revocation_and_nonce_are_checked_in_order() {
        let ring = GatewayKeyRing::new(1, KEY_A);
        let ticket = valid_ticket(&ring);
        // Expired: now AT expires is already dead (no off-by-one grace).
        let err = validate_resume(&ring, &ticket, UniverseTick(1_000), 3, 42).expect_err("expired");
        assert_eq!(
            err,
            ResumeReject::Expired {
                expires: UniverseTick(1_000),
                now: UniverseTick(1_000),
            }
        );
        // Revoked: validity epoch moved (ban/logout bites mid-session).
        let err = validate_resume(&ring, &ticket, UniverseTick(500), 4, 42).expect_err("revoked");
        assert_eq!(err, ResumeReject::Revoked { got: 3, current: 4 });
        // Single-use: the directory nonce moved on (a prior resume consumed it).
        let err = validate_resume(&ring, &ticket, UniverseTick(500), 3, 43).expect_err("replayed");
        assert_eq!(err, ResumeReject::NonceMismatch);
    }

    #[test]
    fn rotation_old_tickets_drain_then_die_on_retire() {
        let mut ring = GatewayKeyRing::new(1, KEY_A);
        let old_ticket = valid_ticket(&ring);
        ring.install_key(2, KEY_B).expect("install");
        ring.promote(2).expect("promote");
        assert_eq!(ring.current_key_id(), 2);
        // Old ticket still validates while key 1 remains installed (drain window).
        let ok = validate_resume(&ring, &old_ticket, UniverseTick(500), 3, 42);
        assert_eq!(ok, Ok(old_ticket.claims));
        // New mints carry the new key id.
        let new_ticket = valid_ticket(&ring);
        assert_eq!(new_ticket.claims.key_id, 2);
        // Retire the old key: outstanding old tickets are now invalid.
        ring.retire(1).expect("retire drained key");
        let err =
            validate_resume(&ring, &old_ticket, UniverseTick(500), 3, 42).expect_err("retired key");
        assert_eq!(err, ResumeReject::UnknownKey(1));
    }

    #[test]
    fn key_ring_management_errors_are_typed() {
        let mut ring = GatewayKeyRing::new(1, KEY_A);
        assert_eq!(
            ring.install_key(1, KEY_B),
            Err(KeyRingError::DuplicateKeyId(1))
        );
        assert_eq!(ring.promote(9), Err(KeyRingError::UnknownKeyId(9)));
        assert_eq!(ring.retire(1), Err(KeyRingError::RetireCurrent(1)));
        assert_eq!(ring.retire(9), Err(KeyRingError::UnknownKeyId(9)));
    }

    #[test]
    fn key_ring_errors_display() {
        assert_eq!(
            KeyRingError::DuplicateKeyId(1).to_string(),
            "key id 1 is already installed"
        );
        assert_eq!(
            KeyRingError::UnknownKeyId(2).to_string(),
            "key id 2 is not installed"
        );
        assert_eq!(
            KeyRingError::RetireCurrent(3).to_string(),
            "key id 3 is the current minting key and cannot be retired"
        );
    }

    #[test]
    fn resume_rejects_display() {
        assert_eq!(
            ResumeReject::UnknownKey(7).to_string(),
            "ticket key id 7 is not in the ring"
        );
        assert_eq!(
            ResumeReject::BadMac.to_string(),
            "ticket MAC verification failed"
        );
        assert_eq!(
            ResumeReject::Expired {
                expires: UniverseTick(10),
                now: UniverseTick(11)
            }
            .to_string(),
            "ticket expired at ut-10 (now ut-11)"
        );
        assert_eq!(
            ResumeReject::Revoked { got: 1, current: 2 }.to_string(),
            "ticket validity epoch 1 is revoked (current 2)"
        );
        assert_eq!(
            ResumeReject::NonceMismatch.to_string(),
            "resume nonce is not the directory-current nonce (single-use)"
        );
    }

    const SIGNING_KEY: [u8; 32] = [0x42; 32];

    fn verifying_key_bytes() -> [u8; 32] {
        SigningKey::from_bytes(&SIGNING_KEY)
            .verifying_key()
            .to_bytes()
    }

    #[test]
    fn login_mint_then_validate_roundtrips() {
        let ticket = mint_login(&SIGNING_KEY, AccountId(7), EpochId(1), 99);
        let account = validate_login(&verifying_key_bytes(), &ticket).expect("valid login");
        assert_eq!(account, AccountId(7));
    }

    #[test]
    fn tampered_login_fields_fail_signature() {
        let base = mint_login(&SIGNING_KEY, AccountId(7), EpochId(1), 99);
        let tampered = vec![
            LoginTicket {
                account: AccountId(8),
                ..base.clone()
            },
            LoginTicket {
                epoch: EpochId(2),
                ..base.clone()
            },
            LoginTicket {
                nonce: 100,
                ..base.clone()
            },
        ];
        for ticket in tampered {
            let err = validate_login(&verifying_key_bytes(), &ticket).expect_err("tampered");
            assert_eq!(err, LoginReject::BadSignature);
        }
    }

    #[test]
    fn malformed_key_and_signature_are_distinct_rejections() {
        let ticket = mint_login(&SIGNING_KEY, AccountId(7), EpochId(1), 99);
        // Uniform 0x02 bytes decode to a y-coordinate with no curve point
        // (decompression fails) — a genuinely malformed key, not just a wrong one.
        let err = validate_login(&[0x02; 32], &ticket).expect_err("bad key");
        assert_eq!(err, LoginReject::MalformedKey);

        let short_sig = LoginTicket {
            signature: vec![1, 2, 3],
            ..ticket.clone()
        };
        let err = validate_login(&verifying_key_bytes(), &short_sig).expect_err("short sig");
        assert_eq!(err, LoginReject::MalformedSignature);

        // A different auth service's signature does not verify.
        let foreign = mint_login(&[0x77; 32], AccountId(7), EpochId(1), 99);
        let err = validate_login(&verifying_key_bytes(), &foreign).expect_err("foreign signer");
        assert_eq!(err, LoginReject::BadSignature);
    }

    #[test]
    fn login_rejects_display() {
        assert_eq!(
            LoginReject::MalformedKey.to_string(),
            "auth-service verifying key bytes are not a valid Ed25519 point"
        );
        assert_eq!(
            LoginReject::MalformedSignature.to_string(),
            "ticket signature is not 64 bytes of Ed25519 signature"
        );
        assert_eq!(
            LoginReject::BadSignature.to_string(),
            "ticket signature does not verify under the auth-service key"
        );
    }

    proptest! {
        /// ANY corruption of the MAC bytes is rejected (constant-time comparison
        /// still rejects everything that is not the exact MAC).
        #[test]
        fn any_mac_corruption_is_rejected(byte_index in 0usize..32, flip in 1u8..=255) {
            let ring = GatewayKeyRing::new(1, KEY_A);
            let mut ticket = valid_ticket(&ring);
            ticket.hmac[byte_index] ^= flip;
            let err = validate_resume(&ring, &ticket, UniverseTick(500), 3, 42)
                .expect_err("corrupted MAC rejected");
            prop_assert_eq!(err, ResumeReject::BadMac);
        }

        /// A ticket minted under one key never validates under a ring holding a
        /// DIFFERENT key at the same id (key material is what matters, not the id).
        #[test]
        fn wrong_key_material_is_rejected(key_byte in 0u8..=254) {
            let mint_ring = GatewayKeyRing::new(1, KEY_A);
            let ticket = valid_ticket(&mint_ring);
            let mut other = [key_byte; 32];
            other[0] = other[0].wrapping_add(1);
            let validate_ring = GatewayKeyRing::new(1, other);
            let err = validate_resume(&validate_ring, &ticket, UniverseTick(500), 3, 42)
                .expect_err("wrong key rejected");
            prop_assert_eq!(err, ResumeReject::BadMac);
        }
    }
}
