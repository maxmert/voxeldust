//! The `DevState` diagnosis substrate (HR6): the client's DECODED DELIVERED world
//! view + lifecycle + honesty counters, as a pure serde type SHARED by the client
//! (which builds it) and `vdctl` (which decodes it). The composited render rows are
//! exactly what pixels would show; the counters distinguish never-welcomed /
//! starved / live without exposing interpolation internals (P2 reshapes those).

use serde::{Deserialize, Serialize};

/// The client lifecycle phase (mirrors the client's `ClientPhase`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevPhase {
    Connecting,
    AwaitingWelcome,
    AwaitingSubscription,
    Active,
    Closed,
}

/// One rendered entity: its id (canonical `Display`) and composited world pose.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevEntityRow {
    pub entity: String,
    pub pos: [f64; 3],
    /// Composited world orientation (`x,y,z,w` — glam `DQuat` component order). Always
    /// finite (sanitized like `pos`); EVERY row carries it (no per-row `is_own` flag), so
    /// the look-at closed loop reads the OWN row's orient as the current facing. Through
    /// P3 this is a full world-space rotation (`orient * -Z` = world-forward), so the
    /// closed-loop nav needs no separate "up".
    pub orient: [f64; 4],
    pub authoritative_sub: u32,
}

/// The (P2) transfer view — empty-but-present so P2 transfer diagnosis is purely
/// additive (a new variant), never a reshape of this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DevTransferView {
    /// No transfer in flight (the only P1.5 state).
    None,
}

/// The decoded, delivered client state — wire truth, the agent's diagnosis surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevState {
    pub phase: DevPhase,
    /// Canonical `Display` of the session id (a string — JSON cannot hold a u128).
    pub session: Option<String>,
    /// Canonical `Display` of this client's own entity (from `AuthorityChanged`).
    pub own_entity: Option<String>,
    /// The player's LOCATION — a human-readable realm label (e.g. "System 7", later
    /// named planets/ships) derived from the authoritative FrameRef of the own entity,
    /// NOT a raw shard id (the client never sees shard processes). `None` until the own
    /// entity has a delivered pose. Fence-validated; changes only on a real cross-realm
    /// move. The player-stats-HUD source + a P2 transfer-test hook.
    pub location: Option<String>,
    /// The continuous render cursor (universe-tick units); `None` before the first
    /// snapshot. Always finite (the builder sanitizes), so this state JSON-encodes.
    pub render_cursor: Option<f64>,
    /// The freshest APPLIED universe tick (integer, run-stable + join-independent);
    /// `None` before the first snapshot. This — NOT the session-relative
    /// `snapshots_applied` count — is what `screenshot --at-tick` aligns on, so a
    /// capture lands on the same world state across runs.
    pub universe_tick: Option<u64>,
    /// The composited render rows — each entity once, exactly what pixels show.
    pub entities: Vec<DevEntityRow>,
    // The honesty counters, classified so an agent reads them right: a nonzero FAULT
    // counter is a real problem; THROUGHPUT/BENIGN ones are not.
    /// THROUGHPUT: snapshots accepted by the §6.3 gate (proves frames are landing).
    pub snapshots_applied: u64,
    /// BENIGN: snapshots dropped as foreign-sub or strictly-stale (the gate working).
    pub stale_frames_dropped: u64,
    /// THROUGHPUT: input datagrams that rode the wire.
    pub sent_input_count: u64,
    /// FAULT: undecodable control/snapshot payloads from the gateway.
    pub decode_errors: u64,
    /// BENIGN (forward-compat): messages of a class/variant not relevant in P1.5,
    /// tolerated without a crash (e.g. a P2 control variant).
    pub ignored: u64,
    /// FAULT: messages from a non-gateway peer — the one-connection invariant tripped.
    pub foreign_peer_drops: u64,
    /// FAULT: delivered poses carrying a non-finite (NaN/Inf) component, sanitized at
    /// ingress. Nonzero means a sender (gateway/shard) is shipping corrupt floats — a
    /// real fault, surfaced (not silently fixed).
    pub nonfinite_poses: u64,
    /// THROUGHPUT: dev-control actions dequeued and applied (incl. Close/Reset).
    pub dev_commands_applied: u64,
    /// FAULT (overload): dev commands shed because the bounded mailbox was full.
    pub dev_commands_dropped: u64,
    pub transfer: DevTransferView,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn sample() -> DevState {
        DevState {
            phase: DevPhase::Active,
            session: Some("sess-1".to_owned()),
            own_entity: Some("ent-7".to_owned()),
            location: Some("System 7".to_owned()),
            render_cursor: Some(101.5),
            universe_tick: Some(101),
            entities: vec![DevEntityRow {
                entity: "ent-7".to_owned(),
                pos: [1.0, 2.0, 3.0],
                orient: [0.0, 0.0, 0.0, 1.0],
                authoritative_sub: 0,
            }],
            snapshots_applied: 4,
            stale_frames_dropped: 1,
            sent_input_count: 9,
            decode_errors: 0,
            ignored: 0,
            foreign_peer_drops: 0,
            nonfinite_poses: 0,
            dev_commands_applied: 2,
            dev_commands_dropped: 0,
            transfer: DevTransferView::None,
        }
    }

    #[test]
    fn devstate_roundtrips_through_json_with_string_ids() {
        let state = sample();
        let json = serde_json::to_string(&state).expect("encode");
        // ids are strings (u128 cannot ride a JSON number).
        assert!(json.contains("\"session\":\"sess-1\""));
        assert!(json.contains("\"location\":\"System 7\""));
        // The orientation quat rides each row (x,y,z,w) — the identity here.
        assert!(json.contains("\"orient\":[0.0,0.0,0.0,1.0]"));
        assert!(json.contains("\"transfer\":{\"kind\":\"none\"}"));
        let back: DevState = serde_json::from_str(&json).expect("decode");
        assert_eq!(back, state);
    }

    #[test]
    fn phase_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&DevPhase::AwaitingSubscription).expect("encode"),
            "\"awaiting_subscription\""
        );
    }
}
