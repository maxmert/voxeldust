//! End-to-end media pipeline tests.
//!
//! Exercises the full Phase 5.1/5.2/5.2.B substrate through the
//! production code paths — same FB serializer, same HMAC verify,
//! same replay window — without spinning up actual shards. Each
//! test simulates one publisher + one receiver, both holding the
//! channel's signature, and walks a frame across them through the
//! wire format.
//!
//! # What this verifies
//!
//! * `MediaFrame` round-trips losslessly through `ShardMsg`'s FB
//!   serializer + deserializer.
//! * `try_push_remote_media` ACCEPTS a correctly-signed frame.
//! * `try_push_remote_media` REJECTS forged tags, stale timestamps,
//!   replay attempts, and frames for unknown channels.
//! * `IncomingMediaBuffer` collects only validated frames.
//!
//! # What this CANNOT verify (requires gameplay layer that doesn't
//! exist yet)
//!
//! * Block-kind producers (Mic, Camera, KeyboardTerminal) — none
//!   exist. The publisher side is "any code that calls
//!   `MediaPublishQueue::enqueue`."
//! * Block-kind consumers (Speaker, Display, TextDisplay) — none
//!   exist. The subscriber side is "any code that reads
//!   `IncomingMediaBuffer.frames`."
//! * Tablet UI for configuring channels, frequencies, grants on
//!   media-bearing blocks — deferred (Phase 5.x).
//! * Codec deps for audio (Opus) / video (H.264 / AV1) / image —
//!   deferred.
//!
//! Until those land, in-game testing of media is not possible. The
//! tests here prove the wire + auth + routing substrate is correct
//! and ready for those layers to plug into.

use std::time::SystemTime;

use glam::DVec3;
use voxeldust_core::media::{
    hmac_sign_media, payload_kind, ChannelMediaBuffer, MediaFrame, MediaIngressDenied,
    MediaPayload, MediaReplayRegistry,
};
use voxeldust_core::shard_message::{MediaBroadcastBatchData, ShardMsg};
use voxeldust_core::signal::channel::SignalChannelTable;
use voxeldust_core::signal::grants::GrantsRegistry;
use voxeldust_core::signal::types::{ChannelMergeStrategy, SignalScope};

const SENDER_SHARD_ID: u64 = 7;
const RECEIVER_OWNER_ID: u64 = 42;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build a fresh receiver-side state with a single channel pre-
/// registered. Returns the table, an empty registry + replay state,
/// and the channel's signature (which both ends hold; in production
/// distributed via grant or out-of-band channel-key sharing).
fn fixture_receiver(channel: &str) -> (SignalChannelTable, GrantsRegistry, MediaReplayRegistry, [u8; 32]) {
    let mut table = SignalChannelTable::default();
    let id = table.resolve_or_create(
        channel,
        SignalScope::Local,
        ChannelMergeStrategy::LastWrite,
        RECEIVER_OWNER_ID,
    );
    let signature = table.get_by_id(id).unwrap().signature;
    (
        table,
        GrantsRegistry::default(),
        MediaReplayRegistry::new(),
        signature,
    )
}

/// Build a signed text frame ready to ship.
fn signed_text_frame(channel: &str, body: &str, sequence: u64, key: &[u8; 32]) -> MediaFrame {
    let timestamp_ms = now_ms();
    let payload_bytes = body.as_bytes();
    let tag = hmac_sign_media(
        key,
        channel,
        payload_kind::TEXT,
        0, // codec=0 for text
        timestamp_ms,
        SENDER_SHARD_ID,
        sequence,
        0, // grant_id=0: open-channel HMAC keyed by signature
        payload_bytes,
    );
    MediaFrame {
        channel_name: channel.into(),
        grant_id: 0,
        sequence,
        timestamp_ms,
        payload: MediaPayload::Text(body.into()),
        auth_tag: tag.to_vec(),
    }
}

#[test]
fn signed_text_frame_round_trips_through_full_wire_path() {
    // 1. Receiver sets up the channel + holds its signature.
    let (table, grants, mut replay, key) = fixture_receiver("alice.intercom");

    // 2. Publisher synthesizes a HMAC-signed text frame.
    let frame = signed_text_frame("alice.intercom", "approach corridor A-7", 1, &key);

    // 3. Publisher wraps the frame in a `MediaBroadcastBatch` and
    //    serializes via the same FB encoder ship-shard would use.
    let outbound = ShardMsg::MediaBroadcastBatch(MediaBroadcastBatchData {
        source_shard_id: SENDER_SHARD_ID,
        source_position: DVec3::ZERO,
        frames: vec![frame],
    });
    let bytes = outbound.serialize();

    // 4. Receiver deserializes (same path the QUIC drain runs).
    let inbound = ShardMsg::deserialize(&bytes).expect("FB round-trip");
    let ShardMsg::MediaBroadcastBatch(batch) = inbound else {
        panic!("unexpected variant after round-trip");
    };

    // 5. Receiver runs the validated-ingress path frame-by-frame.
    let mut buffer = ChannelMediaBuffer::new();
    let now = now_ms();
    for frame in batch.frames {
        match voxeldust_core::media::try_push_remote_media(
            &table,
            &grants,
            &mut replay,
            &mut buffer,
            SENDER_SHARD_ID,
            now,
            frame,
        ) {
            Ok(_channel_id) => {}
            Err(reason) => panic!("valid frame rejected: {reason:?}"),
        }
    }

    // 6. Verify the buffer holds the frame, fully decoded, on the
    // resolved channel id.
    let cid = table.resolve("alice.intercom").unwrap();
    let frames = buffer.frames_for(cid);
    assert_eq!(frames.len(), 1);
    match &frames[0].payload {
        MediaPayload::Text(s) => assert_eq!(s, "approach corridor A-7"),
        other => panic!("expected Text payload, got {other:?}"),
    }
    assert_eq!(frames[0].channel_name, "alice.intercom");
    assert_eq!(frames[0].sequence, 1);
}

#[test]
fn forged_tag_is_rejected_at_ingress() {
    let (table, grants, mut replay, _key) = fixture_receiver("alice.intercom");
    let now = now_ms();

    // Attacker tries to inject a frame without the channel signature.
    let attacker_frame = MediaFrame {
        channel_name: "alice.intercom".into(),
        grant_id: 0,
        sequence: 1,
        timestamp_ms: now,
        payload: MediaPayload::Text("evil command".into()),
        auth_tag: vec![0xAB; 16], // garbage tag — won't survive verify
    };

    let mut buffer = ChannelMediaBuffer::new();
    let result = voxeldust_core::media::try_push_remote_media(
        &table,
        &grants,
        &mut replay,
        &mut buffer,
        SENDER_SHARD_ID,
        now,
        attacker_frame,
    );
    assert_eq!(result.unwrap_err(), MediaIngressDenied::AuthFailed);
}

#[test]
fn replay_of_already_seen_frame_rejected() {
    let (table, grants, mut replay, key) = fixture_receiver("alice.intercom");
    let frame = signed_text_frame("alice.intercom", "hi", 5, &key);
    let now = now_ms();
    let mut buffer = ChannelMediaBuffer::new();

    // First receipt — accept.
    voxeldust_core::media::try_push_remote_media(
        &table,
        &grants,
        &mut replay,
        &mut buffer,
        SENDER_SHARD_ID,
        now,
        frame.clone(),
    )
    .expect("first receipt should accept");

    // Replay the SAME frame (attacker captured + retransmitted).
    let result = voxeldust_core::media::try_push_remote_media(
        &table,
        &grants,
        &mut replay,
        &mut buffer,
        SENDER_SHARD_ID,
        now,
        frame,
    );
    assert_eq!(result.unwrap_err(), MediaIngressDenied::Replay);
}

#[test]
fn frame_for_unknown_channel_rejected() {
    let (table, grants, mut replay, _key) = fixture_receiver("alice.intercom");
    let mut buffer = ChannelMediaBuffer::new();
    // Channel name doesn't resolve — receiver never registered it.
    let unknown_key = [0u8; 32]; // doesn't matter, won't be reached
    let frame = signed_text_frame("bob.unknown", "hi", 1, &unknown_key);
    let result = voxeldust_core::media::try_push_remote_media(
        &table,
        &grants,
        &mut replay,
        &mut buffer,
        SENDER_SHARD_ID,
        now_ms(),
        frame,
    );
    assert_eq!(result.unwrap_err(), MediaIngressDenied::UnknownChannel);
}

#[test]
fn stale_timestamp_rejected() {
    use voxeldust_core::signal::channel::REPLAY_TIMESTAMP_WINDOW_MS;
    let (table, grants, mut replay, key) = fixture_receiver("alice.intercom");

    // Frame timestamped 1_000_000ms; receiver clock is well past
    // REPLAY_TIMESTAMP_WINDOW_MS later.
    let frame_ts = 1_000_000;
    let now = frame_ts + REPLAY_TIMESTAMP_WINDOW_MS + 5_000;
    let payload = b"ancient";
    let tag = hmac_sign_media(
        &key,
        "alice.intercom",
        payload_kind::TEXT,
        0,
        frame_ts,
        SENDER_SHARD_ID,
        1,
        0,
        payload,
    );
    let frame = MediaFrame {
        channel_name: "alice.intercom".into(),
        grant_id: 0,
        sequence: 1,
        timestamp_ms: frame_ts,
        payload: MediaPayload::Text("ancient".into()),
        auth_tag: tag.to_vec(),
    };

    let mut buffer = ChannelMediaBuffer::new();
    let result = voxeldust_core::media::try_push_remote_media(
        &table,
        &grants,
        &mut replay,
        &mut buffer,
        SENDER_SHARD_ID,
        now,
        frame,
    );
    assert_eq!(result.unwrap_err(), MediaIngressDenied::StaleOrFutureTimestamp);
}

#[test]
fn multi_frame_batch_round_trip_with_replay_protection() {
    // Realistic scenario: a publisher ships 3 frames in one batch.
    // Receiver accepts all 3 first time, rejects every replay.
    let (table, grants, mut replay, key) = fixture_receiver("fleet.voice");
    let now = now_ms();
    let f1 = signed_text_frame("fleet.voice", "alpha", 1, &key);
    let f2 = signed_text_frame("fleet.voice", "bravo", 2, &key);
    let f3 = signed_text_frame("fleet.voice", "charlie", 3, &key);

    let batch = ShardMsg::MediaBroadcastBatch(MediaBroadcastBatchData {
        source_shard_id: SENDER_SHARD_ID,
        source_position: DVec3::ZERO,
        frames: vec![f1.clone(), f2.clone(), f3.clone()],
    });
    let bytes = batch.serialize();
    let ShardMsg::MediaBroadcastBatch(round_trip) = ShardMsg::deserialize(&bytes).unwrap()
    else {
        panic!("variant changed");
    };

    let mut buffer = ChannelMediaBuffer::new();
    for frame in round_trip.frames {
        voxeldust_core::media::try_push_remote_media(
            &table,
            &grants,
            &mut replay,
            &mut buffer,
            SENDER_SHARD_ID,
            now,
            frame,
        )
        .expect("first pass: every frame valid");
    }
    let cid = table.resolve("fleet.voice").unwrap();
    assert_eq!(buffer.frames_for(cid).len(), 3);

    // Second pass: same batch re-delivered. Every frame must reject
    // as a replay.
    for frame in &[f1, f2, f3] {
        let result = voxeldust_core::media::try_push_remote_media(
            &table,
            &grants,
            &mut replay,
            &mut buffer,
            SENDER_SHARD_ID,
            now,
            frame.clone(),
        );
        assert_eq!(result.unwrap_err(), MediaIngressDenied::Replay);
    }
}

/// Phase A1: producer-side dispatch via the new `MediaOutboundQueue`.
/// The block-kind publisher (Antenna TX bridge / Terminal in the
/// direct-shard case) enqueues a `(MediaTarget, MediaFrame)` pair;
/// `media_publish_remote` resolves the target into peer shard ids and
/// ships per-peer batches via QUIC. Tests the queue contract — the
/// Bevy system + QUIC dispatch live behind a real cluster fixture.
#[test]
fn publisher_side_enqueues_for_dispatch() {
    use voxeldust_core::media::{MediaOutboundQueue, MediaTarget};
    use voxeldust_core::shard_types::ShardId;

    let mut queue = MediaOutboundQueue::new();
    let (table, _grants, _replay, key) = fixture_receiver("fleet.voice");
    let _ = table; // unused on producer side; kept to clarify the contract.

    let frame = signed_text_frame("fleet.voice", "transmission", 1, &key);
    let target = MediaTarget::Shard(ShardId(99));
    queue.enqueue(target, frame.clone());
    queue.enqueue(
        MediaTarget::RadioFrequency(100),
        signed_text_frame("fleet.voice", "out", 2, &key),
    );

    let drained = queue.drain();
    assert_eq!(drained.len(), 2);
    assert!(matches!(drained[0].0, MediaTarget::Shard(ShardId(99))));
    assert!(matches!(drained[1].0, MediaTarget::RadioFrequency(100)));
    assert_eq!(drained[0].1.sequence, 1);
    assert_eq!(drained[1].1.sequence, 2);
    assert!(queue.is_empty(), "drain leaves the queue empty");
}
