//! HMAC-SHA256 sign/verify for cross-shard signal authentication.
//!
//! # Threat model
//!
//! A foreign shard may attempt to inject a `SignalBroadcastBatch` with a
//! valid `(channel_name, scope, value)` triple but the wrong content — e.g.,
//! re-publishing Alice's `radio.fleet-alpha` channel with their own bogus
//! coordinates. Without authentication, scope-class enforcement and replay
//! defense alone don't catch this: a fresh seq + recent timestamp passes
//! the pre-Phase-3 gates.
//!
//! HMAC-SHA256 closes the forgery hole: every authenticated `SignalBroadcastEntry`
//! carries an `auth_tag` computed from the channel's 32-byte signature
//! (Phase 3 grant-bearing channels) or a per-grant key (Phase 3 grant lookup).
//! A receiver who shares the key can verify; an adversary without the key
//! cannot construct a tag that survives `hmac_verify`.
//!
//! # Tag construction (canonicalization)
//!
//! ```text
//! tag = HMAC-SHA256(
//!     key,
//!     channel_name_len_le_u16
//!     || channel_name_bytes
//!     || scope_code_u8
//!     || frequency_le_u32
//!     || value_type_u8
//!     || value_bits_le_u32        // f32::to_bits()
//!     || timestamp_ms_le_u64
//!     || sender_shard_id_le_u64
//!     || seq_le_u64
//!     || grant_id_le_u64
//! )[..16]                          // truncated to 128 bits
//! ```
//!
//! Truncation to 128 bits is plenty for game-scale unforgeability and halves
//! on-wire overhead vs full SHA-256. (Birthday-collision threshold is
//! 2⁶⁴ messages — far past any realistic adversary budget for a single
//! channel.)
//!
//! Length-prefixing `channel_name` defeats canonicalization-confusion
//! attacks where two different (name, value) pairs hash the same when
//! concatenated. Little-endian ordering is fixed across all platforms so
//! the same bytes produce the same tag regardless of host endianness.
//!
//! # Constant-time comparison
//!
//! [`hmac_verify`] uses [`subtle::ConstantTimeEq`] for the final byte-vs-byte
//! comparison so a verifier doesn't leak partial-match information to a
//! timing-side-channel attacker. Critical because verification happens on
//! the receive hot path — a few thousand verifies per second per shard at
//! peak.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

/// Truncated tag length on the wire. 128 bits is the security floor for
/// HMAC-SHA256 use in this game-scale threat model — anything below 80
/// bits would be vulnerable to brute force, anything above 160 bits
/// wastes wire overhead.
pub const HMAC_TAG_LEN: usize = 16;

/// Compute the canonical signal HMAC tag.
///
/// `value_bits` is `f32::to_bits()` — the bit pattern, not the float — so
/// floats with the same bit pattern always produce the same tag regardless
/// of the receiver's float-comparison semantics (NaN payloads, signaling
/// vs quiet, etc.).
#[inline]
pub fn hmac_sign(
    key: &[u8; 32],
    channel_name: &str,
    scope_code: u8,
    frequency: u32,
    value_type: u8,
    value_bits: u32,
    timestamp_ms: u64,
    sender_shard_id: u64,
    seq: u64,
    grant_id: u64,
) -> [u8; HMAC_TAG_LEN] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");

    // Length-prefixed channel name (defeats canonicalization-confusion).
    let name_bytes = channel_name.as_bytes();
    debug_assert!(
        name_bytes.len() <= u16::MAX as usize,
        "channel name longer than 64 KiB — refuse to sign"
    );
    let name_len_le = (name_bytes.len() as u16).to_le_bytes();
    mac.update(&name_len_le);
    mac.update(name_bytes);

    mac.update(&[scope_code]);
    mac.update(&frequency.to_le_bytes());
    mac.update(&[value_type]);
    mac.update(&value_bits.to_le_bytes());
    mac.update(&timestamp_ms.to_le_bytes());
    mac.update(&sender_shard_id.to_le_bytes());
    mac.update(&seq.to_le_bytes());
    mac.update(&grant_id.to_le_bytes());

    let full = mac.finalize().into_bytes();
    let mut out = [0u8; HMAC_TAG_LEN];
    out.copy_from_slice(&full[..HMAC_TAG_LEN]);
    out
}

/// Verify a tag in constant time. Returns `true` iff the tag matches —
/// constant time across all `Err` and `Ok` paths so an attacker can't
/// learn which prefix of bits matched by timing the response.
#[inline]
pub fn hmac_verify(
    key: &[u8; 32],
    channel_name: &str,
    scope_code: u8,
    frequency: u32,
    value_type: u8,
    value_bits: u32,
    timestamp_ms: u64,
    sender_shard_id: u64,
    seq: u64,
    grant_id: u64,
    tag: &[u8],
) -> bool {
    if tag.len() != HMAC_TAG_LEN {
        return false;
    }
    let expected = hmac_sign(
        key,
        channel_name,
        scope_code,
        frequency,
        value_type,
        value_bits,
        timestamp_ms,
        sender_shard_id,
        seq,
        grant_id,
    );
    expected.ct_eq(tag).into()
}

/// HMAC tag for a `ShardMsg::SignalSubscribe` request. Distinct from the
/// publish-path tag construction so a captured publish tag can't be
/// replayed as a subscribe request and vice versa.
///
/// Canonical input:
/// ```text
/// channel_name_len_le_u16 || channel_name_bytes
///   || subscriber_shard_id_le_u64
///   || grant_id_le_u64
///   || nonce_le_u64
///   || timestamp_ms_le_u64
///   || valid_until_tick_le_u64
/// ```
///
/// `nonce` is generated fresh per request — the receiver can keep a
/// per-(grant, subscriber) "last accepted nonce" check to prevent replay
/// of subscribe requests within the timestamp window.
#[inline]
pub fn hmac_sign_subscribe_request(
    key: &[u8; 32],
    channel_name: &str,
    subscriber_shard_id: u64,
    grant_id: u64,
    nonce: u64,
    timestamp_ms: u64,
    valid_until_tick: u64,
) -> [u8; HMAC_TAG_LEN] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    let name_bytes = channel_name.as_bytes();
    debug_assert!(name_bytes.len() <= u16::MAX as usize);
    mac.update(&(name_bytes.len() as u16).to_le_bytes());
    mac.update(name_bytes);
    mac.update(&subscriber_shard_id.to_le_bytes());
    mac.update(&grant_id.to_le_bytes());
    mac.update(&nonce.to_le_bytes());
    mac.update(&timestamp_ms.to_le_bytes());
    mac.update(&valid_until_tick.to_le_bytes());
    let full = mac.finalize().into_bytes();
    let mut out = [0u8; HMAC_TAG_LEN];
    out.copy_from_slice(&full[..HMAC_TAG_LEN]);
    out
}

/// Constant-time verify of a subscribe-request tag.
#[inline]
pub fn hmac_verify_subscribe_request(
    key: &[u8; 32],
    channel_name: &str,
    subscriber_shard_id: u64,
    grant_id: u64,
    nonce: u64,
    timestamp_ms: u64,
    valid_until_tick: u64,
    tag: &[u8],
) -> bool {
    if tag.len() != HMAC_TAG_LEN {
        return false;
    }
    let expected = hmac_sign_subscribe_request(
        key,
        channel_name,
        subscriber_shard_id,
        grant_id,
        nonce,
        timestamp_ms,
        valid_until_tick,
    );
    expected.ct_eq(tag).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_key() -> [u8; 32] {
        let mut k = [0u8; 32];
        for (i, b) in k.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(13);
        }
        k
    }

    #[test]
    fn sign_verify_roundtrip() {
        let key = fixture_key();
        let tag = hmac_sign(&key, "alice.beacon", 1, 0, 1, 0.5_f32.to_bits(), 1000, 42, 7, 0);
        assert!(hmac_verify(
            &key,
            "alice.beacon",
            1,
            0,
            1,
            0.5_f32.to_bits(),
            1000,
            42,
            7,
            0,
            &tag,
        ));
    }

    #[test]
    fn distinct_keys_produce_distinct_tags() {
        let key1 = fixture_key();
        let mut key2 = fixture_key();
        key2[0] ^= 1;
        let tag1 = hmac_sign(&key1, "ch", 1, 0, 1, 0, 0, 0, 0, 0);
        let tag2 = hmac_sign(&key2, "ch", 1, 0, 1, 0, 0, 0, 0, 0);
        assert_ne!(tag1, tag2);
    }

    #[test]
    fn tampered_value_rejects() {
        let key = fixture_key();
        let tag = hmac_sign(&key, "ch", 1, 0, 1, 100u32, 1000, 42, 7, 0);
        // Same metadata, different value bits — verify must reject.
        assert!(!hmac_verify(&key, "ch", 1, 0, 1, 101u32, 1000, 42, 7, 0, &tag));
    }

    #[test]
    fn tampered_seq_rejects() {
        let key = fixture_key();
        let tag = hmac_sign(&key, "ch", 1, 0, 1, 0, 1000, 42, 7, 0);
        assert!(!hmac_verify(&key, "ch", 1, 0, 1, 0, 1000, 42, 8, 0, &tag));
    }

    #[test]
    fn tampered_channel_name_rejects() {
        let key = fixture_key();
        let tag = hmac_sign(&key, "alice.beacon", 1, 0, 1, 0, 1000, 42, 7, 0);
        // Receiver claims a different channel name — drop.
        assert!(!hmac_verify(&key, "bob.beacon", 1, 0, 1, 0, 1000, 42, 7, 0, &tag));
    }

    #[test]
    fn tampered_scope_rejects() {
        let key = fixture_key();
        // Sign as ShortRange (1), verify as LongRange (2) — drop. Defends
        // against a relay that lies about the scope to bypass spatial
        // filtering.
        let tag = hmac_sign(&key, "ch", 1, 0, 1, 0, 1000, 42, 7, 0);
        assert!(!hmac_verify(&key, "ch", 2, 0, 1, 0, 1000, 42, 7, 0, &tag));
    }

    #[test]
    fn wrong_tag_length_rejects() {
        let key = fixture_key();
        // Empty / wrong-length tags must reject without panicking.
        assert!(!hmac_verify(&key, "ch", 1, 0, 1, 0, 1000, 42, 7, 0, &[]));
        assert!(!hmac_verify(&key, "ch", 1, 0, 1, 0, 1000, 42, 7, 0, &[0u8; 8]));
        assert!(!hmac_verify(&key, "ch", 1, 0, 1, 0, 1000, 42, 7, 0, &[0u8; 32]));
    }

    // -- Subscribe-path HMAC -------------------------------------------------

    #[test]
    fn subscribe_request_sign_verify_roundtrip() {
        let key = fixture_key();
        let tag = hmac_sign_subscribe_request(
            &key,
            "alice.lights",
            /*subscriber=*/ 99,
            /*grant=*/ 42,
            /*nonce=*/ 0xDEAD_BEEF,
            /*ts=*/ 1000,
            /*valid_until=*/ 2000,
        );
        assert!(hmac_verify_subscribe_request(
            &key, "alice.lights", 99, 42, 0xDEAD_BEEF, 1000, 2000, &tag,
        ));
    }

    #[test]
    fn subscribe_request_distinct_from_publish_tag() {
        // A publish tag and a subscribe-request tag for the "same" inputs
        // produce different bytes — defends against an adversary who
        // captured a publish tag and tries to replay it as a subscribe.
        let key = fixture_key();
        let pub_tag = hmac_sign(&key, "ch", 1, 0, 1, 0, 1000, 99, 7, 42);
        let sub_tag = hmac_sign_subscribe_request(&key, "ch", 99, 42, 0, 1000, 2000);
        assert_ne!(pub_tag, sub_tag);
    }

    #[test]
    fn subscribe_request_tampered_grant_rejects() {
        let key = fixture_key();
        let tag = hmac_sign_subscribe_request(&key, "ch", 99, 42, 0, 1000, 2000);
        // Same inputs but grant_id changed — verify rejects.
        assert!(!hmac_verify_subscribe_request(&key, "ch", 99, /*grant=*/ 43, 0, 1000, 2000, &tag));
    }

    #[test]
    fn subscribe_request_tampered_lease_rejects() {
        // An adversary trying to extend their own lease arbitrarily by
        // changing valid_until_tick after capture.
        let key = fixture_key();
        let tag = hmac_sign_subscribe_request(&key, "ch", 99, 42, 0, 1000, 2000);
        assert!(!hmac_verify_subscribe_request(&key, "ch", 99, 42, 0, 1000, /*until=*/ 99999, &tag));
    }

    #[test]
    fn name_length_prefix_disambiguates_concatenation() {
        // Without length-prefixing, sign("ab", "cd...") and sign("a", "bcd...")
        // would produce the same tag if the concatenation overlapped. With
        // length-prefixing, the lengths differ, so the tags differ.
        let key = fixture_key();
        let tag_ab = hmac_sign(&key, "ab", 1, 0, 1, 0, 0, 0, 0, 0);
        let tag_a = hmac_sign(&key, "a", 1, 0, 1, 0, 0, 0, 0, 0);
        // Of course the channel names differ outright too — this just
        // confirms our canonicalization includes the name length.
        assert_ne!(tag_ab, tag_a);
    }
}
