//! Cross-shard character state preservation.
//!
//! Wraps the body/head/turn-in-place state into a forward-compatible
//! blob so a player handoff (ship → planet, planet → system EVA, etc.)
//! preserves the head pose and any in-flight turn animation.
//!
//! Wire format: `bincode` for compact binary encoding, no compression
//! (the blob is well under 64 bytes — zstd would add framing overhead
//! and a runtime dep without saving anything). The blob is carried by
//! [`crate::handoff::PlayerHandoff::character_state`] and gated by the
//! `schema_version` field so older shards on `version = 0/1` ignore it.
//!
//! `SCHEMA_VERSION` for the body/head extension is **2**; bump to 3
//! only for an incompatible field rename or removal. Adding optional
//! tag-keyed fields (Stamina, EquipmentLoad, ActiveItem,
//! DamageResistance per [`super::hooks::CharacterComponentTag`]) does
//! NOT bump the version — extend the [`CharacterStateBlob`] struct in
//! place with `#[serde(default)]` so older blobs decode cleanly.

use super::body_head::TurnInPlace;
use serde::{Deserialize, Serialize};

/// Schema version stamped into [`crate::handoff::PlayerHandoff::schema_version`]
/// when this blob is present.
///
/// - `0` = pre-KCC legacy.
/// - `1` = KCC migration (blob reserved but always empty).
/// - `2` = body/head/turn extension (this blob).
pub const SCHEMA_VERSION: u16 = 2;

/// Body / head / turn snapshot transferred across shards.
///
/// All angles in radians (tangent-frame). Velocities are NOT included —
/// they live on the existing [`crate::handoff::PlayerHandoff::velocity`]
/// field, which is independent of the body/head pose.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct CharacterStateBlob {
    pub body_yaw: f32,
    pub head_yaw: f32,
    pub head_pitch: f32,
    /// Locomotion state at handoff time; mapped to/from
    /// [`super::locomotion::LocomotionState`] via its `as u8` repr.
    pub locomotion: u8,
    /// Horizontal speed at handoff time (m/s); drives the receiving
    /// shard's first-frame walk/run blend.
    pub locomotion_speed: f32,
    /// `Some` if the player was mid-turn at handoff time. Receiving
    /// shard re-inserts [`TurnInPlace`] so the animation continues
    /// across the boundary.
    #[serde(default)]
    pub turn: Option<TurnInPlace>,
}

/// Encode a blob into the wire-format bytes carried by
/// [`crate::handoff::PlayerHandoff::character_state`].
pub fn encode(blob: &CharacterStateBlob) -> Vec<u8> {
    bincode::serialize(blob).expect("CharacterStateBlob has no non-serializable fields")
}

/// Decode a blob; returns `Default` on any parse error so a malformed
/// blob never blocks a handoff (the player still spawns, just with the
/// neutral pose). Logs the error at debug level for diagnostics.
pub fn decode(bytes: &[u8]) -> CharacterStateBlob {
    if bytes.is_empty() {
        return CharacterStateBlob::default();
    }
    match bincode::deserialize::<CharacterStateBlob>(bytes) {
        Ok(b) => b,
        Err(_) => CharacterStateBlob::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_all_fields() {
        let original = CharacterStateBlob {
            body_yaw: 1.234,
            head_yaw: -0.5,
            head_pitch: 0.8,
            locomotion: 1,
            locomotion_speed: 4.5,
            turn: Some(TurnInPlace {
                target_body_yaw: 2.7,
                t: 0.42,
            }),
        };
        let bytes = encode(&original);
        let decoded = decode(&bytes);
        assert_eq!(original, decoded);
    }

    #[test]
    fn round_trip_no_turn() {
        let original = CharacterStateBlob {
            body_yaw: 0.0,
            head_yaw: 0.0,
            head_pitch: 0.0,
            locomotion: 0,
            locomotion_speed: 0.0,
            turn: None,
        };
        let bytes = encode(&original);
        assert_eq!(original, decode(&bytes));
    }

    #[test]
    fn empty_bytes_decode_to_default() {
        let blob = decode(&[]);
        assert_eq!(blob, CharacterStateBlob::default());
    }

    #[test]
    fn malformed_bytes_decode_to_default_not_panic() {
        let blob = decode(&[0xff; 4]);
        assert_eq!(blob, CharacterStateBlob::default());
    }
}
