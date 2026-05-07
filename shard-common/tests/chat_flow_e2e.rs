//! Phase A4 — end-to-end chat flow tests.
//!
//! These tests stitch together the channel-keyed media plumbing
//! (Phase A1) with the Antenna media bridge systems (Phase A2) to
//! prove the full chat path works without spinning up a real cluster.
//! Each scenario sets up the receiving-side pipeline (channel table,
//! grants registry, replay window, channel-media buffer) and walks a
//! frame through every stage:
//!
//! 1. Same-shard chat (no Antenna) — Player A's Terminal publish
//!    pushes to a Local channel; Player B's Terminal subscribed to
//!    the same channel reads it.
//! 2. Cross-shard chat via Antenna pair — Player A on shard 1 has a
//!    TX-Antenna wired to a Local channel + Radio frequency; Player
//!    B on shard 2 has an RX-Antenna on the same frequency mirroring
//!    to a local channel; Player B's Terminal reads it.
//! 3. Forged HMAC drops — same as scenario 2 but the cross-shard
//!    frame's auth_tag is mangled; receiver rejects.
//! 4. Uni-directional antennas — TX-only and RX-only antennas
//!    function as one-way bridges.

use voxeldust_core::media::{
    hmac_sign_media, payload_kind, ChannelMediaBuffer, MediaFrame, MediaOutboundQueue,
    MediaPayload, MediaReplayRegistry, MediaTarget,
};
use voxeldust_core::signal::components::{
    AntennaRxSide, AntennaState, AntennaTxSide, TerminalState,
};
use voxeldust_core::signal::grants::GrantsRegistry;
use voxeldust_core::signal::types::{ChannelMergeStrategy, SignalScope};
use voxeldust_core::signal::SignalChannelTable;

const SHARD_A: u64 = 1;
const SHARD_B: u64 = 2;

fn now_ms() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Test 1: same-shard chat (no Antenna)
// ---------------------------------------------------------------------------

#[test]
fn same_shard_terminal_to_terminal_chat_works_without_antenna() {
    // Two Terminals on the same shard, both wired to "ship.chat".
    // Player A types; Player B (subscribed to the same channel) sees
    // the message in its scrollback. No Antenna involved — the
    // channel-keyed media buffer is the only routing.
    let mut table = SignalChannelTable::default();
    let chat_id = table.resolve_or_create(
        "ship.chat",
        SignalScope::Local,
        ChannelMergeStrategy::LastWrite,
        99,
    );
    let chat_signature = table.get_by_id(chat_id).unwrap().signature;

    // Simulate terminal_publish: Player A produces a frame on chat_id.
    let mut buffer = ChannelMediaBuffer::new();
    let now = now_ms();
    let body = "hello from A";
    let auth_tag = hmac_sign_media(
        &chat_signature,
        "ship.chat",
        payload_kind::TEXT,
        0,
        now,
        SHARD_A,
        1,
        0,
        body.as_bytes(),
    );
    let frame = MediaFrame {
        channel_name: "ship.chat".into(),
        grant_id: 0,
        sequence: 1,
        timestamp_ms: now,
        payload: MediaPayload::Text(body.into()),
        auth_tag: auth_tag.to_vec(),
    };
    buffer.push(chat_id, frame);

    // Simulate terminal_subscribe: Player B reads the channel slot.
    let mut player_b = TerminalState {
        subscribe_channel: Some(chat_id),
        publish_channel: Some(chat_id),
        recent_lines: Vec::new(),
        max_lines: 64,
        owner_session: 0,
        next_sequence: 0,
        active: true,
    };
    for f in buffer.frames_for(chat_id) {
        if let MediaPayload::Text(text) = &f.payload {
            player_b.push_line(text.clone());
        }
    }

    assert_eq!(player_b.recent_lines, vec!["hello from A".to_string()]);
}

#[test]
fn channel_isolation_means_unrelated_terminals_dont_see_each_others_messages() {
    // Two channels on the same shard. Terminal-on-A subscribes to
    // ch_a; a publish on ch_b must NOT show up in A's scrollback.
    let mut table = SignalChannelTable::default();
    let cid_a = table.resolve_or_create(
        "ch.a",
        SignalScope::Local,
        ChannelMergeStrategy::LastWrite,
        99,
    );
    let cid_b = table.resolve_or_create(
        "ch.b",
        SignalScope::Local,
        ChannelMergeStrategy::LastWrite,
        99,
    );
    let mut buffer = ChannelMediaBuffer::new();
    let now = now_ms();
    let frame_b = MediaFrame {
        channel_name: "ch.b".into(),
        grant_id: 0,
        sequence: 1,
        timestamp_ms: now,
        payload: MediaPayload::Text("not for A".into()),
        auth_tag: Vec::new(),
    };
    buffer.push(cid_b, frame_b);

    // Subscriber on ch_a reads ONLY ch_a.
    let mut subscriber = TerminalState {
        subscribe_channel: Some(cid_a),
        publish_channel: None,
        recent_lines: Vec::new(),
        max_lines: 64,
        owner_session: 0,
        next_sequence: 0,
        active: true,
    };
    for f in buffer.frames_for(cid_a) {
        if let MediaPayload::Text(text) = &f.payload {
            subscriber.push_line(text.clone());
        }
    }
    assert!(
        subscriber.recent_lines.is_empty(),
        "ch_b frames must not reach a ch_a subscriber"
    );
}

// ---------------------------------------------------------------------------
// Test 2: cross-shard chat via Antenna pair (open Radio)
// ---------------------------------------------------------------------------

#[test]
fn cross_shard_chat_via_antenna_pair_open_radio_works_end_to_end() {
    // Set up two shards, each with the bridged Radio channel registered
    // (this is what apply_antenna_config does on each side at config
    // time). Walk a frame through:
    //
    //   Shard A: Terminal publish → ChannelMediaBuffer[local_a]
    //          → Antenna TX scans local_a + ships to MediaOutboundQueue
    //            with RadioFrequency target
    //   Wire:    MediaFrame addressed to bridged Radio channel name,
    //            HMAC-signed with that channel's signature
    //   Shard B: drain_quic invokes try_push_remote_media which validates
    //            HMAC and pushes to ChannelMediaBuffer[bridged_b]
    //          → Antenna RX mirror copies to ChannelMediaBuffer[local_b]
    //          → Terminal subscribe reads local_b
    const FREQ: u32 = 100;
    let bridged_name = format!("ship.chat__radio_out_{}", FREQ);

    // ---- Shard A side ----
    let mut table_a = SignalChannelTable::default();
    let local_a = table_a.resolve_or_create(
        "ship.chat",
        SignalScope::Local,
        ChannelMergeStrategy::LastWrite,
        99,
    );
    // The TX-side bridged Radio channel is created at apply time.
    let bridged_a = table_a.resolve_or_create(
        &bridged_name,
        SignalScope::Radio { frequency: FREQ },
        ChannelMergeStrategy::LastWrite,
        99,
    );
    let bridged_a_signature = table_a.get_by_id(bridged_a).unwrap().signature;

    // ---- Shard B side ----
    // The RX-side antenna creates a matching bridged Radio channel
    // (must use the same name + same signature for HMAC verify to
    // succeed — in production the signature is established by an
    // out-of-band exchange or by both sides deriving it deterministi-
    // cally from a shared seed; here we copy explicitly to simulate
    // the post-handshake state).
    let mut table_b = SignalChannelTable::default();
    let bridged_b = table_b.resolve_or_create(
        &bridged_name,
        SignalScope::Radio { frequency: FREQ },
        ChannelMergeStrategy::LastWrite,
        99,
    );
    table_b.get_by_id_mut(bridged_b).unwrap().signature = bridged_a_signature;
    let local_b = table_b.resolve_or_create(
        "ship.chat",
        SignalScope::Local,
        ChannelMergeStrategy::LastWrite,
        99,
    );

    // ---- Step 1: Player A's terminal_publish writes to local_a ----
    let mut buffer_a = ChannelMediaBuffer::new();
    let now = now_ms();
    let body = "hello from A";
    // (terminal_publish uses the LOCAL channel's signature; for this
    // test we don't care about its tag — we only follow the frame
    // through Antenna TX which re-signs with the bridged channel.)
    let local_frame = MediaFrame {
        channel_name: "ship.chat".into(),
        grant_id: 0,
        sequence: 1,
        timestamp_ms: now,
        payload: MediaPayload::Text(body.into()),
        auth_tag: Vec::new(),
    };
    buffer_a.push(local_a, local_frame.clone());

    // ---- Step 2: Antenna TX bridge scans local_a and produces an
    // outbound frame stamped for bridged_name. We re-create the
    // signing logic from antenna_publish_media here directly. ----
    let outbound_frame = {
        let payload_bytes = body.as_bytes();
        let tag = hmac_sign_media(
            &bridged_a_signature,
            &bridged_name,
            payload_kind::TEXT,
            0,
            now,
            SHARD_A,
            1, // tx.next_sequence
            0,
            payload_bytes,
        );
        MediaFrame {
            channel_name: bridged_name.clone(),
            grant_id: 0,
            sequence: 1,
            timestamp_ms: now,
            payload: MediaPayload::Text(body.into()),
            auth_tag: tag.to_vec(),
        }
    };
    let mut outbound_queue = MediaOutboundQueue::new();
    outbound_queue.enqueue(MediaTarget::RadioFrequency(FREQ), outbound_frame.clone());
    let drained = outbound_queue.drain();
    assert_eq!(drained.len(), 1);

    // ---- Step 3: Shard B receives the frame, runs ingress validation. ----
    let grants_b = GrantsRegistry::default();
    let mut replay_b = MediaReplayRegistry::new();
    let mut buffer_b = ChannelMediaBuffer::new();
    voxeldust_core::media::try_push_remote_media(
        &table_b,
        &grants_b,
        &mut replay_b,
        &mut buffer_b,
        SHARD_A,
        now,
        outbound_frame,
    )
    .expect("valid bridged frame must accept");

    // Frame should have landed in bridged_b's slot.
    assert_eq!(
        buffer_b.frames_for(bridged_b).len(),
        1,
        "frame must land in bridged channel after ingress"
    );

    // ---- Step 4: Antenna RX mirrors bridged_b → local_b. We re-
    // create the mirror logic from antenna_rx_media_mirror here. ----
    let local_b_signature = table_b.get_by_id(local_b).unwrap().signature;
    let mirrored_frame = {
        let payload_bytes = body.as_bytes();
        let tag = hmac_sign_media(
            &local_b_signature,
            "ship.chat",
            payload_kind::TEXT,
            0,
            now,
            SHARD_B,
            1, // sequence preserved from cross-shard
            0,
            payload_bytes,
        );
        MediaFrame {
            channel_name: "ship.chat".into(),
            grant_id: 0,
            sequence: 1,
            timestamp_ms: now,
            payload: MediaPayload::Text(body.into()),
            auth_tag: tag.to_vec(),
        }
    };
    buffer_b.push(local_b, mirrored_frame);

    // ---- Step 5: Player B's terminal_subscribe reads local_b. ----
    let mut player_b = TerminalState {
        subscribe_channel: Some(local_b),
        publish_channel: Some(local_b),
        recent_lines: Vec::new(),
        max_lines: 64,
        owner_session: 0,
        next_sequence: 0,
        active: true,
    };
    for f in buffer_b.frames_for(local_b) {
        if let MediaPayload::Text(text) = &f.payload {
            player_b.push_line(text.clone());
        }
    }
    assert_eq!(player_b.recent_lines, vec!["hello from A".to_string()]);
}

// ---------------------------------------------------------------------------
// Test 3: forged HMAC dropped at ingress
// ---------------------------------------------------------------------------

#[test]
fn cross_shard_chat_drops_forged_hmac() {
    // Same setup as Test 2, but the frame's auth_tag is garbage. The
    // receiver's try_push_remote_media must reject before the frame
    // touches the bridged channel buffer.
    const FREQ: u32 = 200;
    let bridged_name = format!("comms.chat__radio_out_{}", FREQ);

    let mut table_a = SignalChannelTable::default();
    let bridged_a = table_a.resolve_or_create(
        &bridged_name,
        SignalScope::Radio { frequency: FREQ },
        ChannelMergeStrategy::LastWrite,
        99,
    );
    let bridged_signature_a = table_a.get_by_id(bridged_a).unwrap().signature;

    let mut table_b = SignalChannelTable::default();
    let bridged_b = table_b.resolve_or_create(
        &bridged_name,
        SignalScope::Radio { frequency: FREQ },
        ChannelMergeStrategy::LastWrite,
        99,
    );
    table_b.get_by_id_mut(bridged_b).unwrap().signature = bridged_signature_a;

    let now = now_ms();
    let attacker_frame = MediaFrame {
        channel_name: bridged_name,
        grant_id: 0,
        sequence: 1,
        timestamp_ms: now,
        // Attacker doesn't have the channel signature; can't compute
        // a valid HMAC. Garbage tag.
        payload: MediaPayload::Text("evil".into()),
        auth_tag: vec![0xFF; 16],
    };
    let grants_b = GrantsRegistry::default();
    let mut replay_b = MediaReplayRegistry::new();
    let mut buffer_b = ChannelMediaBuffer::new();
    let result = voxeldust_core::media::try_push_remote_media(
        &table_b,
        &grants_b,
        &mut replay_b,
        &mut buffer_b,
        SHARD_A,
        now,
        attacker_frame,
    );
    assert!(result.is_err(), "forged HMAC must reject at ingress");
    assert!(
        buffer_b.is_empty(),
        "rejected frame must NOT enter the channel buffer"
    );
}

// ---------------------------------------------------------------------------
// Test 4: uni-directional antennas (TX-only, RX-only)
// ---------------------------------------------------------------------------

#[test]
fn antenna_state_supports_tx_only_beacon() {
    // TX-only: an Antenna with rx=None is valid; antenna_publish_media
    // forwards from local_a, antenna_rx_media_mirror is a no-op.
    let antenna = AntennaState {
        tx: Some(AntennaTxSide {
            local_channel_id: None, // resolved at apply time
            frequency: 50,
            grant_id: None,
            remote_shard_id: None,
            next_sequence: 0,
        }),
        rx: None,
        owner_session: 99,
        active: true,
    };
    assert!(antenna.has_any_side());
    assert!(antenna.tx.is_some());
    assert!(antenna.rx.is_none());
}

#[test]
fn antenna_state_supports_rx_only_listener() {
    // RX-only: an Antenna with tx=None is valid; antenna_publish_media
    // is a no-op for it, antenna_rx_media_mirror picks up bridged frames.
    let antenna = AntennaState {
        tx: None,
        rx: Some(AntennaRxSide {
            local_channel_id: None, // resolved at apply time
            bridged_channel_id: None,
            frequency: 75,
            grant_id: None,
            remote_shard_id: None,
            lease_until_tick: 0,
        }),
        owner_session: 99,
        active: true,
    };
    assert!(antenna.has_any_side());
    assert!(antenna.tx.is_none());
    assert!(antenna.rx.is_some());
}

#[test]
fn antenna_state_with_no_sides_is_invalid() {
    // An Antenna with both sides None is invalid — apply rejects it
    // (NoSideConfigured) and runtime systems should treat it as a
    // no-op. has_any_side() is the predicate the apply path checks.
    let antenna = AntennaState {
        tx: None,
        rx: None,
        owner_session: 99,
        active: true,
    };
    assert!(!antenna.has_any_side());
}

// ---------------------------------------------------------------------------
// Test 5: replay defense across the bridge
// ---------------------------------------------------------------------------

#[test]
fn cross_shard_chat_rejects_replay() {
    // After Player B accepts the first frame, a literal replay (same
    // (channel, sender, seq, ts)) must reject. This guards against an
    // attacker capturing the wire packet and re-injecting it.
    const FREQ: u32 = 300;
    let bridged_name = format!("test.chat__radio_out_{}", FREQ);

    let mut table_a = SignalChannelTable::default();
    let bridged_a = table_a.resolve_or_create(
        &bridged_name,
        SignalScope::Radio { frequency: FREQ },
        ChannelMergeStrategy::LastWrite,
        99,
    );
    let bridged_signature = table_a.get_by_id(bridged_a).unwrap().signature;

    let mut table_b = SignalChannelTable::default();
    let bridged_b = table_b.resolve_or_create(
        &bridged_name,
        SignalScope::Radio { frequency: FREQ },
        ChannelMergeStrategy::LastWrite,
        99,
    );
    table_b.get_by_id_mut(bridged_b).unwrap().signature = bridged_signature;

    let now = now_ms();
    let body = "captured";
    let tag = hmac_sign_media(
        &bridged_signature,
        &bridged_name,
        payload_kind::TEXT,
        0,
        now,
        SHARD_A,
        7,
        0,
        body.as_bytes(),
    );
    let frame = MediaFrame {
        channel_name: bridged_name,
        grant_id: 0,
        sequence: 7,
        timestamp_ms: now,
        payload: MediaPayload::Text(body.into()),
        auth_tag: tag.to_vec(),
    };

    let grants_b = GrantsRegistry::default();
    let mut replay_b = MediaReplayRegistry::new();
    let mut buffer_b = ChannelMediaBuffer::new();
    voxeldust_core::media::try_push_remote_media(
        &table_b,
        &grants_b,
        &mut replay_b,
        &mut buffer_b,
        SHARD_A,
        now,
        frame.clone(),
    )
    .expect("first delivery accepts");
    let result = voxeldust_core::media::try_push_remote_media(
        &table_b,
        &grants_b,
        &mut replay_b,
        &mut buffer_b,
        SHARD_A,
        now,
        frame,
    );
    assert!(
        matches!(
            result,
            Err(voxeldust_core::media::MediaIngressDenied::Replay)
        ),
        "replayed frame must reject"
    );
}

// Plugin-init smoke test would require a real ShardIdentity fixture +
// peer registry, which lives behind the harness's full cluster setup.
// The component-level tests above cover Phase A1's contracts; the
// per-shard integration tests in the harness cluster cover plugin wiring.
