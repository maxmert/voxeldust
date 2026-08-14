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
    /// INERT since the pure-renderer collapse (S6): a node-AGNOSTIC client no longer has a
    /// per-entity authoritative sub (it renders every entity latest-wins by `EntityId`, never
    /// learning which node owns it). Retained as a stable diagnostic field (always the inert
    /// `vd_client::view::RENDERED_SUB` = 0) so `vdctl`/process-parity decode the row unchanged.
    pub authoritative_sub: u32,
}

/// One DRAWN realm box (VU diagnosis): its realm id (canonical `Debug`) and the center it would
/// render at (its `RealmBox` centre flattened to metres through the one `draw_center` chokepoint —
/// there is nothing to reduce it against, because the shard chain already measured it from the realm
/// this session stands in, the same space [`DevEntityRow::pos`] is in, so a test may compare the two
/// directly). This is
/// the render-plane twin of [`DevEntityRow`]: it exposes WHICH realms the client is
/// drawing and WHERE, so a headless test can prove the streamed scene is present AND
/// moving (a planet whose center changes across ticks is orbiting; a static container
/// shell never moves). Always finite (sanitized).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevRealmBox {
    pub realm: String,
    pub center: [f64; 3],
    /// SHAKE DIAGNOSIS — the newest universe tick this realm's pose feed has delivered; `None` for a
    /// box the feed never streamed (it is sitting at its boot placement). Read against
    /// [`DevState::entity_feed_newest_tick`], this is what distinguishes the two candidate causes of a
    /// wobbling horizon: a box whose tick TRACKS the entity feed is being drawn from the same moment
    /// as the player, so any residual wobble is a render-path fault; a box whose tick drifts against
    /// it is being authored by a shard whose sense of universe time is running independently. It also
    /// separates a FROZEN box (tick stops advancing) from a live one, so a smoothness gate cannot be
    /// satisfied by a box that simply stopped updating.
    pub newest_tick: Option<u64>,
}

/// SLICE 6 S5 — how one feed's tracks classified at the render cursor. The shake was invisible for a
/// long time because nothing reported that the interpolation machinery, though correct, never ran: a
/// feed permanently on `clamped_old` means the cursor is falling behind the retained history, which IS
/// that condition. A healthy live feed reads mostly `blended`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DevWindowCensus {
    /// Tracks genuinely interpolating (the cursor sits between two delivered poses).
    pub blended: u32,
    /// Tracks whose cursor precedes their whole history — clamped to the oldest pose.
    pub clamped_old: u32,
    /// Tracks at or past their newest pose — FROZEN (required; the client never coasts).
    pub clamped_new: u32,
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
    /// Canonical `Display` of this client's own entity (from `ServerControlMsg::OwnEntity` — the
    /// node-agnostic own-avatar signal; the client never learns which node owns it).
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
    /// SHAKE DIAGNOSIS — the newest universe tick the ENTITY pose feed has delivered (across all
    /// tracks). `None` before the first snapshot.
    ///
    /// WHY BOTH FEEDS' TICKS ARE REPORTED SEPARATELY. A player standing on a planet sees two things
    /// that must agree: their own body, composed by the shard that owns the realm they are in, and the
    /// ground under them, whose placement is authored by that realm's PARENT and relayed down. Each
    /// shard advances its own sense of universe time only when a clock sync ARRIVES from the
    /// orchestrator — there is no local per-tick advance — so the two numbers can drift apart, and the
    /// relative motion that drift produces is visible as a wobbling horizon. Reporting both, next to
    /// [`DevState::render_cursor`], is what tells a wobble caused by the render path apart from one
    /// caused by two clocks. Measure before building: the fix spans four crates on a ~50-minute build.
    pub entity_feed_newest_tick: Option<u64>,
    /// SHAKE DIAGNOSIS — the newest universe tick the REALM pose feed has delivered (across all
    /// streamed realms). `None` before the first realm frame. See
    /// [`DevState::entity_feed_newest_tick`] for why this is reported alongside it.
    pub realm_feed_newest_tick: Option<u64>,
    /// The composited render rows — each entity once, exactly what pixels show.
    pub entities: Vec<DevEntityRow>,
    /// SLICE 6 S5 — how the ENTITY feed's tracks classified at the reported cursor.
    pub entity_windows: DevWindowCensus,
    /// SLICE 6 S5 — how the REALM feed's placements classified at the same cursor.
    pub realm_windows: DevWindowCensus,
    /// SLICE 6 S5 — the two feeds' newest ticks, differenced (entity minus realm). `None` until both
    /// have delivered. This is the arrival skew whose interaction with un-cursored drawing WAS the
    /// shake; after the fix it should be small and, more importantly, harmless — both feeds are read
    /// at one cursor regardless.
    pub feed_skew_ticks: Option<i64>,
    /// The DRAWN realm boxes (VU) — each realm the client is currently rendering, with
    /// its composited center. Proves the streamed render-scene is present (a `Planet`
    /// box appears once its `RealmSceneDelta` lands) and LIVE (its center moves as the
    /// realm-pose feed overlays the orbit). Empty at walk scale (no realm scene streams).
    pub realm_boxes: Vec<DevRealmBox>,
    // The honesty counters, classified so an agent reads them right: a nonzero FAULT
    // counter is a real problem; THROUGHPUT/BENIGN ones are not.
    /// THROUGHPUT: snapshots accepted by the §6.3 gate (proves frames are landing).
    pub snapshots_applied: u64,
    /// THROUGHPUT: REALM frames accepted by the gate (FA-2c) — proves the moving-realm feed is landing
    /// (a `WaitUntil{RealmFramesApplied >= 1}` closed-loop e2e polls this). 0 at walk scale (no movers).
    pub realm_frames_applied: u64,
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
    /// FAULT at steady state / BENIGN right after a removal: entity rows refused by the client's
    /// RESURRECT GUARD (a straggler for an entity the reliable remove message already evicted).
    /// A brief burst after a removal is the unreliable lane draining; GROWTH at steady state means
    /// a shard keeps emitting an entity it told this client was removed — OR the guard is wrongly
    /// armed and the entity is permanently undrawable (the frozen/invisible-figure class). Surfaced
    /// here (audit :304) so `vdctl state` can tell "the shard stopped emitting" from "the client
    /// refused every row it sent".
    pub resurrect_rows_dropped: u64,
    /// BENIGN during a crossing's grace window / FAULT at steady state: rows skipped by the
    /// ONE-SPACE rule, summed across the entity AND realm feeds — a pose stated in a space other
    /// than the one this client stands in. Steady-state growth means a shard ships entities/realms
    /// in a space its observer does not stand in.
    pub foreign_space_rows: u64,
    /// BENIGN during a crossing's grace window: own-entity rows dropped as the ECHO (the old home's
    /// still-draining copy of the leaver in the space just left). Growth OUTSIDE a crossing means
    /// the echo filter is eating live rows.
    pub echo_rows_dropped: u64,
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
            // Distinct from each other AND from `universe_tick`, so a round-trip that dropped or
            // transposed one of the two feed ticks cannot still pass.
            entity_feed_newest_tick: Some(102),
            realm_feed_newest_tick: Some(100),
            entity_windows: Default::default(),
            realm_windows: Default::default(),
            feed_skew_ticks: None,
            entities: vec![DevEntityRow {
                entity: "ent-7".to_owned(),
                pos: [1.0, 2.0, 3.0],
                orient: [0.0, 0.0, 0.0, 1.0],
                authoritative_sub: 0,
            }],
            realm_boxes: vec![DevRealmBox {
                realm: "Planet(7)".to_owned(),
                center: [10.0, 0.0, 0.0],
                newest_tick: Some(100),
            }],
            snapshots_applied: 4,
            realm_frames_applied: 3,
            stale_frames_dropped: 1,
            sent_input_count: 9,
            decode_errors: 0,
            ignored: 0,
            foreign_peer_drops: 0,
            nonfinite_poses: 0,
            // Distinct non-zero values so a round-trip that dropped or transposed one of the three
            // row-drop counters cannot still pass (the same discipline as the feed ticks above).
            resurrect_rows_dropped: 5,
            foreign_space_rows: 6,
            echo_rows_dropped: 7,
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
        // The drawn realm box rides its realm id + composited center.
        assert!(json.contains("\"realm\":\"Planet(7)\""));
        assert!(json.contains("\"transfer\":{\"kind\":\"none\"}"));
        // The three row-drop honesty counters ride the surface (audit :304 — a wrongly-armed
        // resurrect guard must be VISIBLE to `vdctl state`), each with its distinct sample value.
        assert!(json.contains("\"resurrect_rows_dropped\":5"));
        assert!(json.contains("\"foreign_space_rows\":6"));
        assert!(json.contains("\"echo_rows_dropped\":7"));
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
