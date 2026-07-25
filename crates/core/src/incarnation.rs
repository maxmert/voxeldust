//! `IncarnationCookie` (RLM Step 5c) — a per-launch nonce that identifies ONE incarnation of a realm
//! shard across an orchestrator restart.
//!
//! The problem it solves: after the orchestrator process is killed and rebuilt, it inherits shards it
//! launched pre-crash but no longer holds OS `Child` handles for — and the operating system may have
//! recycled a dead shard's process id onto a COMPLETELY UNRELATED program. A bare pid is therefore not a
//! safe identity. The spawner mints a fresh random cookie BEFORE each fork, persists it in the write-ahead
//! launch intent, and (Step 5e) has the shard echo it on an admin probe — so a rebuilt orchestrator can
//! ask "are you really MY incarnation X?" and never be fooled by a coincidental pid match.
//!
//! This type is PURE DATA (no rng, no clock — vd-core is deterministic): it is MINTED by the launch
//! backend (`vd-bins`, where std entropy is allowed) and only STORED/encoded here. It rides the durable
//! store as opaque bytes and the `VD_INCARNATION_COOKIE` env transport as a fixed 32-char lowercase-hex
//! string (like [`crate::realm_path::RealmPath::to_env_string`], it can never drift from a wire form). The
//! cookie is NOT a secret — the pid-reuse guard relies on the true child HOLDING its probe port, not on
//! cookie secrecy — so a plain nonce is sufficient.

use crate::realm_path::{HEX_LOWER, hex_digit};
use serde::{Deserialize, Serialize};

/// The fixed hex width of a cookie's env-string form: a `u128` is 16 bytes → 32 lowercase-hex chars.
const COOKIE_HEX_LEN: usize = 32;

/// A per-launch incarnation nonce (128 bits — wide enough that two incarnations never collide in
/// practice). Opaque: the kernel compares and stores it, never interprets the bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IncarnationCookie(pub u128);

impl IncarnationCookie {
    /// Encode as the `VD_INCARNATION_COOKIE` env transport: exactly [`COOKIE_HEX_LEN`] lowercase-hex
    /// chars of the big-endian bytes. Table-driven (no `write!` `Result` arm to leave uncovered — HR5);
    /// round-trips exactly via [`IncarnationCookie::from_env_string`].
    #[must_use]
    pub fn to_env_string(&self) -> String {
        let bytes = self.0.to_be_bytes();
        let mut s = String::with_capacity(COOKIE_HEX_LEN);
        for b in bytes {
            s.push(HEX_LOWER[(b >> 4) as usize]);
            s.push(HEX_LOWER[(b & 0x0f) as usize]);
        }
        s
    }

    /// Decode a [`to_env_string`](IncarnationCookie::to_env_string) form. FAILS LOUD on any malformation
    /// — a boot misconfiguration must never silently give a shard the wrong identity. Monomorphic body
    /// (all branching here, HR5).
    ///
    /// # Errors
    /// [`IncarnationCookieError::WrongLength`] unless the string is exactly [`COOKIE_HEX_LEN`] chars;
    /// [`IncarnationCookieError::NotHex`] on any character outside `[0-9a-f]`.
    pub fn from_env_string(s: &str) -> Result<IncarnationCookie, IncarnationCookieError> {
        let raw = s.as_bytes();
        if raw.len() != COOKIE_HEX_LEN {
            return Err(IncarnationCookieError::WrongLength(raw.len()));
        }
        let mut value: u128 = 0;
        for &c in raw {
            let nibble = hex_digit(c).ok_or(IncarnationCookieError::NotHex)?;
            value = (value << 4) | u128::from(nibble);
        }
        Ok(IncarnationCookie(value))
    }
}

/// A malformed `VD_INCARNATION_COOKIE` — rejected LOUD at boot so a shard never boots with a wrong or
/// truncated identity (which would defeat the Step-5e pid-reuse guard).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IncarnationCookieError {
    /// The hex string is not exactly [`COOKIE_HEX_LEN`] characters.
    #[error("VD_INCARNATION_COOKIE hex must be {COOKIE_HEX_LEN} chars, got {0}")]
    WrongLength(usize),
    /// A character outside the lowercase-hex alphabet `[0-9a-f]`.
    #[error("VD_INCARNATION_COOKIE contains a non-hex character")]
    NotHex,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_string_round_trips_including_extremes() {
        for raw in [
            0u128,
            1,
            0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF,
            u128::MAX,
        ] {
            let cookie = IncarnationCookie(raw);
            let s = cookie.to_env_string();
            assert_eq!(s.len(), COOKIE_HEX_LEN, "fixed width for {raw:#x}");
            assert_eq!(
                IncarnationCookie::from_env_string(&s),
                Ok(cookie),
                "round-trip for {raw:#x}"
            );
        }
    }

    #[test]
    fn zero_and_max_have_the_expected_hex() {
        assert_eq!(IncarnationCookie(0).to_env_string(), "0".repeat(32));
        assert_eq!(IncarnationCookie(u128::MAX).to_env_string(), "f".repeat(32));
    }

    #[test]
    fn distinct_cookies_compare_unequal() {
        // HR5(d): exercise BOTH equality arms (the `!=` false arm is otherwise uncovered).
        assert_eq!(IncarnationCookie(7), IncarnationCookie(7));
        assert_ne!(IncarnationCookie(7), IncarnationCookie(8));
    }

    #[test]
    fn from_env_string_rejects_wrong_length_loud() {
        // Too short and too long (both hit the length gate before any hex parse).
        assert_eq!(
            IncarnationCookie::from_env_string(""),
            Err(IncarnationCookieError::WrongLength(0))
        );
        assert_eq!(
            IncarnationCookie::from_env_string(&"a".repeat(31)),
            Err(IncarnationCookieError::WrongLength(31))
        );
        assert_eq!(
            IncarnationCookie::from_env_string(&"a".repeat(33)),
            Err(IncarnationCookieError::WrongLength(33))
        );
    }

    #[test]
    fn from_env_string_rejects_non_hex_loud() {
        // Correct length (32) but a non-hex char — reaches the per-digit reject (valid digits first, then
        // the invalid `z`, so the NotHex arm is hit AFTER the accept arms are exercised).
        let mut s = "a".repeat(31);
        s.push('z');
        assert_eq!(
            IncarnationCookie::from_env_string(&s),
            Err(IncarnationCookieError::NotHex)
        );
    }
}
