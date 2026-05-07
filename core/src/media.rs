//! Phase 5.1 — media (text/audio/video/image) channels.
//!
//! Communication devices — voice radio, video phones, security
//! cameras + displays, in-world text terminals, intercoms — are
//! first-class for a Star Citizen × Minecraft game. The signal
//! infrastructure already gave us channels, scopes, auth/grants, and
//! cross-shard routing; media is a parallel **data plane** that
//! shares that entire **control plane**.
//!
//! # Same control plane, different data plane
//!
//! - **Channels**: media frames carry a `channel_name` string. The
//!   same `ChannelKey { owner_id, path }` namespacing used by signals
//!   applies — `<owner>.<block-uid>.<role>` defaults work identically.
//! - **Scopes**: `Local`, `ShortRange`, `LongRange`, `Radio`. A
//!   ShortRange voice channel within a 10 km CB-radio radius is one
//!   line of config. A Radio audio channel routes through the same
//!   galaxy-shard `signal_relay` we already built.
//! - **Auth**: HMAC-SHA256 keyed by grant. `auth_tag` covers a hash
//!   of the payload (not the full body) — verify cost is O(hash)
//!   regardless of frame size.
//! - **Tablet inbox**: media subscriptions surface in the same
//!   Connections hub as signal subscriptions.
//!
//! # What's different
//!
//! - **Payload shape**: `MediaPayload` carries `Vec<u8>` per frame
//!   (text bytes, encoded audio frame, encoded video frame,
//!   compressed image). Vastly different size per kind — text frames
//!   are ~tens of bytes, video keyframes can be ~25 KB.
//! - **Transport**: text frames batch fine through the existing
//!   `SignalBroadcastBatch` shape. Audio (~600 B/frame at 50 ms) and
//!   video (~5–25 KB/frame at 30 fps) want a dedicated QUIC stream
//!   per active subscription, pull-driven from publisher to
//!   subscriber. Transport is a deferred slice (`Phase 5.2+`); this
//!   module ships the frame/codec/payload type system + the wire
//!   format needed for that future work.
//! - **Codec selection**: codecs are identified by stable `u8`
//!   ordinals on the wire (matches `SignalProperty::as_ordinal`'s
//!   evolution scheme). Mismatched-codec subscriptions fail at
//!   config time in the tablet UI — no runtime transcoding burden
//!   on the server. Subscribers must support the publisher's
//!   codec.
//!
//! # What this slice DOES ship
//!
//! - `MediaPayload` enum + codec/format ordinal enums.
//! - `MediaFrame` Rust type carrying the full per-frame metadata.
//! - `MediaBroadcastBatch` wire shape (FB schema + Rust serde) —
//!   parallel to `SignalBroadcastBatch` for text + low-bitrate use.
//! - `ShardMsg::MediaBroadcastBatch` variant for cross-shard
//!   routing.
//! - Round-trip tests for every payload variant.
//!
//! # What this slice deliberately defers
//!
//! - Per-subscription QUIC stream provisioning (audio/video).
//! - Codec library deps: `opus = "0.3"` for audio, AV1/H.264 via
//!   `dav1d`/H.264 lib for video, `image = "0.25"` for image. None
//!   are imported until a slice actually needs them.
//! - Block kinds: Mic/Speaker/Camera/Display/Intercom/Keyboard
//!   Terminal/TextDisplay. These plug into the existing
//!   `BlockKindSignalSchema` extended with `BlockKindMediaSchema`.
//! - Tablet UI: media playback widgets (audio level meter, video
//!   frame display, text scrollback).

use std::collections::HashMap;

use bevy_ecs::prelude::Resource;

use crate::shard_types::ShardId;
use crate::signal::channel::{ChannelId, ReplayReject, ReplayWindow, REPLAY_TIMESTAMP_WINDOW_MS};
use crate::signal::grants::GrantsRegistry;
use crate::signal::SignalChannelTable;

#[allow(unused_imports)]
use ReplayReject as _; // silence "unused" — referenced in error-conversion docs

// ---------------------------------------------------------------------------
// Codec / format ordinals — wire-stable, never reuse a value.
// ---------------------------------------------------------------------------

/// Audio codec ordinal. New codecs are added at the next available
/// value; existing values are never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AudioCodec {
    /// Opus — narrowband / wideband / fullband, configured per
    /// channel via `codec_metadata`. Opus is the baseline audio
    /// codec for VoIP at MVP scale.
    Opus = 0,
}

/// Video codec ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum VideoCodec {
    /// H.264 baseline profile. Compatible with hardware decoders on
    /// every modern GPU; conservative MVP choice.
    H264Baseline = 0,
    /// AV1. Better compression at given quality but heavier
    /// software-decode on older hardware. Reserved for a future
    /// upgrade path.
    Av1 = 1,
}

/// Still-image format ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ImageFormat {
    /// PNG — lossless, larger files. Use for terminal screenshots
    /// and lossless screen captures.
    Png = 0,
    /// JPEG — lossy, smaller files. Use for camera snapshots / map
    /// thumbnails.
    Jpeg = 1,
}

impl AudioCodec {
    pub fn as_ordinal(self) -> u8 {
        self as u8
    }
    pub fn from_ordinal(o: u8) -> Option<Self> {
        match o {
            0 => Some(Self::Opus),
            _ => None,
        }
    }
}

impl VideoCodec {
    pub fn as_ordinal(self) -> u8 {
        self as u8
    }
    pub fn from_ordinal(o: u8) -> Option<Self> {
        match o {
            0 => Some(Self::H264Baseline),
            1 => Some(Self::Av1),
            _ => None,
        }
    }
}

impl ImageFormat {
    pub fn as_ordinal(self) -> u8 {
        self as u8
    }
    pub fn from_ordinal(o: u8) -> Option<Self> {
        match o {
            0 => Some(Self::Png),
            1 => Some(Self::Jpeg),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Payload kinds — top-level discriminator.
// ---------------------------------------------------------------------------

/// Payload-kind ordinal carried on the wire. The `MediaPayload` enum
/// preserves these via its discriminant naming convention.
pub mod payload_kind {
    pub const TEXT: u8 = 0;
    pub const AUDIO: u8 = 1;
    pub const VIDEO: u8 = 2;
    pub const IMAGE: u8 = 3;
}

/// What a media frame carries. Tagged by `MediaFrame::payload_kind`
/// on the wire; the variant fields recompose into the Rust enum on
/// deserialization.
#[derive(Clone, Debug, PartialEq)]
pub enum MediaPayload {
    /// UTF-8 text. ≤ 4 KB per frame at MVP scale (chat lines,
    /// terminal output, notification toasts). No codec — `codec`
    /// field is 0.
    Text(String),
    /// Encoded audio frame. Codec-specific bytes (e.g., Opus
    /// container). `samples` = sample count in this frame,
    /// `sample_rate` = Hz.
    Audio {
        codec: AudioCodec,
        frame: Vec<u8>,
        samples: u32,
        sample_rate: u32,
    },
    /// Encoded video frame. Codec-specific bytes (e.g., H.264 NAL
    /// units). `keyframe = true` marks I-frames the decoder can
    /// resync from.
    Video {
        codec: VideoCodec,
        frame: Vec<u8>,
        width: u16,
        height: u16,
        keyframe: bool,
    },
    /// Encoded still image (PNG / JPEG bytes).
    Image {
        format: ImageFormat,
        data: Vec<u8>,
        width: u16,
        height: u16,
    },
}

impl MediaPayload {
    pub fn payload_kind(&self) -> u8 {
        match self {
            Self::Text(_) => payload_kind::TEXT,
            Self::Audio { .. } => payload_kind::AUDIO,
            Self::Video { .. } => payload_kind::VIDEO,
            Self::Image { .. } => payload_kind::IMAGE,
        }
    }

    /// Codec ordinal. 0 for Text. The publisher and every subscriber
    /// must agree on the codec for a given channel — mismatch is a
    /// config-time error in the tablet UI, never a runtime transcode.
    pub fn codec_ordinal(&self) -> u8 {
        match self {
            Self::Text(_) => 0,
            Self::Audio { codec, .. } => codec.as_ordinal(),
            Self::Video { codec, .. } => codec.as_ordinal(),
            Self::Image { format, .. } => format.as_ordinal(),
        }
    }
}

// ---------------------------------------------------------------------------
// MediaFrame — full per-frame metadata + payload.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ECS data-plane buffers (Phase 5.2).
//
// Mirrors the signal pipeline's `IncomingSignalBuffer` shape so any
// Bevy system already familiar with the signal layer can read media
// the same way. Two resources:
//
//   * `IncomingMediaBuffer` — frames that arrived this tick from any
//     source (cross-shard QUIC ingest, local block publishers).
//     Subscribers iterate it each tick. Cleared at the START of each
//     tick by `media_ingest_clear` — so frames are always for the
//     current tick, never stale.
//
//   * `MediaPublishQueue` — frames the local shard wants to ship to
//     other shards via `MediaBroadcastBatch`. Drained at the END of
//     each tick by `media_publish_remote`, batched per target shard,
//     dispatched as a single QUIC message per (target, source).
//
// Data plane only. The signal pipeline's `SignalChannelTable` keeps
// owning channel identity, scope, grants, and replay state — media
// reuses the same control plane via channel-name resolution. A
// channel can carry signal values, media frames, or both; they do
// not conflict because they live in separate value planes.
// ---------------------------------------------------------------------------

/// Per-channel pending media frames for the current tick.
///
/// Replaces the prior flat `IncomingMediaBuffer` with a channel-keyed
/// data plane. Every media frame on this shard — whether produced
/// locally by `terminal_publish`, mirrored from a Radio bridge by
/// `antenna_rx_media_mirror`, or arrived cross-shard via
/// `try_push_remote_media` — lives here keyed by `ChannelId`. Readers
/// (Terminal subscribers, Antenna TX bridges) look up by channel and
/// see exactly the frames published to that channel this tick.
///
/// Tick-scoped: cleared at the start of each tick by the pipeline's
/// `Ingest` phase. A subscriber that misses its read window doesn't
/// see the frame next tick — same semantics as the signal channel
/// `pending` map.
///
/// **Why per-channel keying matters**: the original design routed
/// media by `target_shard_id` directly. That made same-shard chat
/// (publisher and subscriber on one shard) impossible — there was no
/// loopback. Going through a channel id makes loopback automatic: the
/// publisher writes to channel C, and any subscriber on C reads it,
/// regardless of which shard each lives on.
#[derive(Resource, Default, Debug)]
pub struct ChannelMediaBuffer {
    pending: HashMap<ChannelId, smallvec::SmallVec<[MediaFrame; 2]>>,
}

impl ChannelMediaBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop every frame from every channel. Called once per tick at
    /// the start of `MediaSet::Ingest`.
    pub fn clear(&mut self) {
        self.pending.clear();
    }

    /// Append `frame` to `channel`'s pending list. Order-preserving so
    /// multi-frame bursts (a player typing fast) arrive in the order
    /// they were produced.
    pub fn push(&mut self, channel: ChannelId, frame: MediaFrame) {
        self.pending.entry(channel).or_default().push(frame);
    }

    /// Read this tick's frames for `channel`. Returns an empty slice
    /// when nothing has been pushed (so callers can iterate freely
    /// without an Option dance).
    pub fn frames_for(&self, channel: ChannelId) -> &[MediaFrame] {
        self.pending
            .get(&channel)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Iterate every (channel_id, frame) pair pending this tick.
    /// Used by Antenna TX bridges to scan multiple subscribed channels
    /// in one pass.
    pub fn iter(&self) -> impl Iterator<Item = (ChannelId, &MediaFrame)> {
        self.pending
            .iter()
            .flat_map(|(id, frames)| frames.iter().map(move |f| (*id, f)))
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Total frame count across all channels — for metrics / tests.
    pub fn total_frames(&self) -> usize {
        self.pending.values().map(|v| v.len()).sum()
    }
}

/// Where a media frame is headed when leaving this shard.
///
/// Replaces the bare `ShardId` target of the prior `MediaPublishQueue`
/// with a routing-aware enum. Direct shard targeting still works (used
/// by keyed Local-via-grant publishes), but Radio frequencies are now
/// first-class — the dispatcher resolves the peer set from the
/// frequency at send time, so the same enqueue site doesn't have to
/// know who's listening.
#[derive(Clone, Debug)]
pub enum MediaTarget {
    /// Direct delivery to a specific peer shard. Used by the
    /// pre-Phase-D path where a Terminal/grant addressed a specific
    /// `target_shard_id`.
    Shard(ShardId),
    /// Open Radio broadcast on `frequency`. Fan-out resolved at
    /// dispatch time against the peer registry's frequency-interest
    /// map (Phase 3F.9.C `ShipFrequencyInterest`). Receivers whose
    /// Antenna RX is wired to the same frequency accept; others drop.
    RadioFrequency(u32),
}

/// Outbound media queue. Antenna TX bridges + direct shard publishers
/// enqueue here; `media_publish_remote` drains at the end of each
/// tick and ships via QUIC `MediaBroadcastBatch`.
///
/// Replaces the prior `MediaPublishQueue` (Vec<(ShardId, MediaFrame)>).
/// The new shape unlocks frequency-routed broadcast without forcing
/// every producer to know the peer set.
#[derive(Resource, Default, Debug)]
pub struct MediaOutboundQueue {
    pub pending: Vec<(MediaTarget, MediaFrame)>,
}

impl MediaOutboundQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&mut self, target: MediaTarget, frame: MediaFrame) {
        self.pending.push((target, frame));
    }

    pub fn drain(&mut self) -> Vec<(MediaTarget, MediaFrame)> {
        std::mem::take(&mut self.pending)
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

/// One full media frame on the wire.
///
/// Carries the same auth / sequencing / timestamping fields as the
/// signal pipeline's `SignalBroadcastEntry` so the existing replay-
/// window + HMAC verify logic can be reused unchanged at the
/// receiver. The only meaningful structural difference is the
/// payload field — `MediaPayload` instead of a numeric scalar.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaFrame {
    pub channel_name: String,
    /// Capability lookup id. Identical semantics to
    /// `SignalBroadcastEntry::grant_id`: receiver looks up the grant
    /// in `GrantsRegistry`, verifies `auth_tag` against `grant.key`.
    pub grant_id: u64,
    /// Per-(channel, sender) monotonic sequence — drives the
    /// receiver's replay window for media just as for signals.
    pub sequence: u64,
    /// UNIX wall-clock millis at the publisher.
    pub timestamp_ms: u64,
    pub payload: MediaPayload,
    /// HMAC-SHA256 truncated to 16 bytes. The auth input is
    /// `(canonical_header || sha256(payload_bytes))` so verify cost
    /// stays O(hash) instead of O(frame_size) — important for large
    /// video frames.
    pub auth_tag: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Phase 5.2.B — HMAC sign/verify + cross-shard ingress validation.
//
// Mirrors `signal::auth::{hmac_sign, hmac_verify}` for media frames.
// The auth input differs in one critical way: media payloads can be
// large (25 KB video keyframes), so the HMAC input includes a SHA-256
// HASH of the payload bytes rather than the bytes themselves. Verify
// cost stays O(hash) regardless of frame size.
// ---------------------------------------------------------------------------

/// Truncated tag length on the wire — same as signals (HMAC-SHA256
/// truncated to 16 bytes).
pub const MEDIA_HMAC_TAG_LEN: usize = 16;

/// Compute the canonical media HMAC tag.
///
/// ```text
/// payload_hash = SHA-256(payload_data)
/// tag = HMAC-SHA256(
///     key,
///     channel_name_len_le_u16
///     || channel_name_bytes
///     || payload_kind_u8
///     || codec_u8
///     || timestamp_ms_le_u64
///     || sender_shard_id_le_u64
///     || sequence_le_u64
///     || grant_id_le_u64
///     || payload_hash         // 32 bytes
/// )[..16]
/// ```
///
/// Length-prefixing the channel name defeats canonicalization
/// confusion. Hashing the payload first bounds verify cost
/// regardless of frame size.
#[inline]
pub fn hmac_sign_media(
    key: &[u8; 32],
    channel_name: &str,
    payload_kind: u8,
    codec: u8,
    timestamp_ms: u64,
    sender_shard_id: u64,
    sequence: u64,
    grant_id: u64,
    payload_data: &[u8],
) -> [u8; MEDIA_HMAC_TAG_LEN] {
    use hmac::{Hmac, Mac};
    use sha2::{Digest, Sha256};
    type HmacSha256 = Hmac<Sha256>;

    let payload_hash = Sha256::digest(payload_data);

    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    let name_bytes = channel_name.as_bytes();
    debug_assert!(
        name_bytes.len() <= u16::MAX as usize,
        "channel name longer than 64 KiB — refuse to sign"
    );
    let name_len_le = (name_bytes.len() as u16).to_le_bytes();
    mac.update(&name_len_le);
    mac.update(name_bytes);
    mac.update(&[payload_kind, codec]);
    mac.update(&timestamp_ms.to_le_bytes());
    mac.update(&sender_shard_id.to_le_bytes());
    mac.update(&sequence.to_le_bytes());
    mac.update(&grant_id.to_le_bytes());
    mac.update(&payload_hash);

    let full = mac.finalize().into_bytes();
    let mut out = [0u8; MEDIA_HMAC_TAG_LEN];
    out.copy_from_slice(&full[..MEDIA_HMAC_TAG_LEN]);
    out
}

/// Verify a media tag in constant time. Same single-purpose verify
/// path as signals — `subtle::ConstantTimeEq` guards against
/// timing-side-channel partial-match leaks.
#[inline]
pub fn hmac_verify_media(
    key: &[u8; 32],
    channel_name: &str,
    payload_kind: u8,
    codec: u8,
    timestamp_ms: u64,
    sender_shard_id: u64,
    sequence: u64,
    grant_id: u64,
    payload_data: &[u8],
    tag: &[u8],
) -> bool {
    use subtle::ConstantTimeEq;
    if tag.len() != MEDIA_HMAC_TAG_LEN {
        return false;
    }
    let expected = hmac_sign_media(
        key,
        channel_name,
        payload_kind,
        codec,
        timestamp_ms,
        sender_shard_id,
        sequence,
        grant_id,
        payload_data,
    );
    expected.ct_eq(tag).into()
}

// ---------------------------------------------------------------------------
// Replay registry — per-(channel, sender) sliding window for media.
// ---------------------------------------------------------------------------

/// Key for the media replay map: (channel id, sender shard id).
/// Channel id is local to the receiving shard; sender shard id is
/// the cluster-wide source identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MediaReplayKey {
    pub channel_id: ChannelId,
    pub sender_shard_id: u64,
}

/// Per-(channel, sender) sliding-window replay defense for media.
/// Reuses [`signal::channel::ReplayWindow`] — same 64-entry bitmap +
/// timestamp freshness logic. Media uses its own keyspace so signal
/// and media seqs don't conflict on the same channel.
#[derive(Resource, Default, Debug)]
pub struct MediaReplayRegistry {
    windows: HashMap<MediaReplayKey, ReplayWindow>,
}

impl MediaReplayRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to record a (channel, sender, seq, ts) tuple. Returns
    /// `Ok(())` on accept, `Err(reason)` on reject. Mirrors the
    /// signal channel's per-(channel, sender) replay logic.
    pub fn try_accept(
        &mut self,
        channel_id: ChannelId,
        sender_shard_id: u64,
        seq: u64,
        timestamp_ms: u64,
        now_ms: u64,
    ) -> Result<(), ReplayReject> {
        let key = MediaReplayKey {
            channel_id,
            sender_shard_id,
        };
        let window = self.windows.entry(key).or_default();
        window.try_accept(seq, timestamp_ms, now_ms)
    }

    /// Drop replay state for a (channel, sender) pair. Used when the
    /// channel is removed or the sender disconnects.
    pub fn forget(&mut self, channel_id: ChannelId, sender_shard_id: u64) {
        self.windows.remove(&MediaReplayKey {
            channel_id,
            sender_shard_id,
        });
    }

    pub fn entry_count(&self) -> usize {
        self.windows.len()
    }
}

// ---------------------------------------------------------------------------
// try_push_remote_media — receive-side validation for cross-shard frames.
// ---------------------------------------------------------------------------

/// Reasons `try_push_remote_media` rejects a frame. Same shape as
/// signal's `RemoteIngressDenied` — caller logs at debug + drops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaIngressDenied {
    /// Channel name doesn't resolve in `SignalChannelTable`. Anti-
    /// griefing — denies arbitrary-name flooding.
    UnknownChannel,
    /// HMAC verification failed: either the grant doesn't authorize
    /// this channel, the key is wrong, or the tag was forged.
    AuthFailed,
    /// Timestamp drift outside `REPLAY_TIMESTAMP_WINDOW_MS` —
    /// either stale (replay attempt) or future (clock skew).
    StaleOrFutureTimestamp,
    /// Sequence number was already seen for this (channel, sender).
    Replay,
}

/// Validate + accept (or reject) a cross-shard media frame.
///
/// Performs, in order:
///   1. Channel-name resolution — drops unknown channels.
///   2. Grant lookup (`grant_id != 0`) or open-broadcast check
///      (`grant_id == 0` → channel must have `auth = None` to skip
///      HMAC; otherwise reject).
///   3. HMAC verify against the lookup key.
///   4. Replay window check on `(channel, sender)`.
///
/// On success the (validated) frame is pushed directly into
/// [`ChannelMediaBuffer`] keyed by the channel's resolved `ChannelId`.
/// Subscribers (`terminal_subscribe`, `antenna_publish_media`) read it
/// from there during their respective phases of the same tick.
///
/// Returns `Ok(channel_id)` on success so callers can stamp metrics
/// or log without re-resolving the name.
pub fn try_push_remote_media(
    table: &SignalChannelTable,
    grants: &GrantsRegistry,
    replay: &mut MediaReplayRegistry,
    media_buffer: &mut ChannelMediaBuffer,
    sender_shard_id: u64,
    now_ms: u64,
    frame: MediaFrame,
) -> Result<ChannelId, MediaIngressDenied> {
    // 1. Channel must exist locally.
    let channel_id = table
        .resolve(&frame.channel_name)
        .ok_or(MediaIngressDenied::UnknownChannel)?;

    // 2. Look up the HMAC key. For grant-bearing frames the key
    //    comes from the grant; for grant_id=0 frames the channel's
    //    own signature is the key (matches signal's
    //    `try_push_remote` policy — channels always have an HMAC
    //    identity, the question is only whether a grant overrides
    //    it for cross-shard authorization).
    let key: [u8; 32] = if frame.grant_id != 0 {
        *grants
            .check_publish(frame.grant_id, channel_id, now_ms)
            .ok_or(MediaIngressDenied::AuthFailed)?
    } else {
        let channel = table
            .get_by_id(channel_id)
            .ok_or(MediaIngressDenied::UnknownChannel)?;
        channel.signature
    };

    // 3. Freshness window check before HMAC (cheaper, skips the
    //    expensive verify on stale timestamps).
    let drift = if now_ms >= frame.timestamp_ms {
        now_ms - frame.timestamp_ms
    } else {
        frame.timestamp_ms - now_ms
    };
    if drift > REPLAY_TIMESTAMP_WINDOW_MS {
        return Err(MediaIngressDenied::StaleOrFutureTimestamp);
    }

    // 4. HMAC verify. Constant-time comparison via `hmac_verify_media`.
    let payload_kind = frame.payload.payload_kind();
    let codec = frame.payload.codec_ordinal();
    let payload_bytes = match &frame.payload {
        MediaPayload::Text(s) => s.as_bytes().to_vec(),
        MediaPayload::Audio { frame, .. } => frame.clone(),
        MediaPayload::Video { frame, .. } => frame.clone(),
        MediaPayload::Image { data, .. } => data.clone(),
    };
    if !hmac_verify_media(
        &key,
        &frame.channel_name,
        payload_kind,
        codec,
        frame.timestamp_ms,
        sender_shard_id,
        frame.sequence,
        frame.grant_id,
        &payload_bytes,
        &frame.auth_tag,
    ) {
        return Err(MediaIngressDenied::AuthFailed);
    }

    // 5. Replay window — last gate. HMAC verify already passed, so
    //    we know the frame was authored by a legitimate publisher.
    //    The replay check ensures THIS particular frame (by seq)
    //    hasn't been seen before.
    if replay
        .try_accept(channel_id, sender_shard_id, frame.sequence, frame.timestamp_ms, now_ms)
        .is_err()
    {
        return Err(MediaIngressDenied::Replay);
    }

    // 6. Push to the per-channel buffer. Subscribers + Antenna TX
    //    bridges read from here during their phase of this same tick.
    media_buffer.push(channel_id, frame);
    Ok(channel_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_kind_ordinals_are_distinct_and_stable() {
        // The ordinals are part of the wire format. Existing
        // payloads must keep their values across schema evolutions.
        assert_eq!(MediaPayload::Text("".into()).payload_kind(), 0);
        assert_eq!(
            MediaPayload::Audio {
                codec: AudioCodec::Opus,
                frame: vec![],
                samples: 0,
                sample_rate: 0,
            }
            .payload_kind(),
            1
        );
        assert_eq!(
            MediaPayload::Video {
                codec: VideoCodec::H264Baseline,
                frame: vec![],
                width: 0,
                height: 0,
                keyframe: false,
            }
            .payload_kind(),
            2
        );
        assert_eq!(
            MediaPayload::Image {
                format: ImageFormat::Png,
                data: vec![],
                width: 0,
                height: 0,
            }
            .payload_kind(),
            3
        );
    }

    #[test]
    fn audio_codec_round_trip() {
        for c in [AudioCodec::Opus] {
            let o = c.as_ordinal();
            assert_eq!(AudioCodec::from_ordinal(o), Some(c));
        }
        // Unknown ordinal returns None — never panics on bad input.
        assert!(AudioCodec::from_ordinal(99).is_none());
    }

    #[test]
    fn video_codec_round_trip() {
        for c in [VideoCodec::H264Baseline, VideoCodec::Av1] {
            assert_eq!(VideoCodec::from_ordinal(c.as_ordinal()), Some(c));
        }
        assert!(VideoCodec::from_ordinal(99).is_none());
    }

    #[test]
    fn image_format_round_trip() {
        for f in [ImageFormat::Png, ImageFormat::Jpeg] {
            assert_eq!(ImageFormat::from_ordinal(f.as_ordinal()), Some(f));
        }
        assert!(ImageFormat::from_ordinal(99).is_none());
    }

    #[test]
    fn codec_ordinal_for_text_is_zero() {
        let t = MediaPayload::Text("hello".into());
        assert_eq!(t.codec_ordinal(), 0);
    }

    fn fixture_text_frame(channel: &str, body: &str) -> MediaFrame {
        MediaFrame {
            channel_name: channel.into(),
            grant_id: 0,
            sequence: 1,
            timestamp_ms: 0,
            payload: MediaPayload::Text(body.into()),
            auth_tag: vec![],
        }
    }

    #[test]
    fn channel_media_buffer_clear_resets_to_empty() {
        let mut buf = ChannelMediaBuffer::new();
        buf.push(ChannelId(0), fixture_text_frame("ch", "hello"));
        buf.push(ChannelId(0), fixture_text_frame("ch", "world"));
        assert_eq!(buf.total_frames(), 2);
        buf.clear();
        assert!(buf.is_empty(), "clear must drop every frame");
    }

    #[test]
    fn channel_media_buffer_isolates_by_channel_id() {
        // Frames pushed to one channel must not appear when reading
        // another channel's slot. Per-channel keying is the whole point
        // of this resource.
        let mut buf = ChannelMediaBuffer::new();
        buf.push(ChannelId(0), fixture_text_frame("a", "first"));
        buf.push(ChannelId(1), fixture_text_frame("b", "second"));
        let a = buf.frames_for(ChannelId(0));
        let b = buf.frames_for(ChannelId(1));
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(buf.frames_for(ChannelId(99)).len(), 0);
    }

    #[test]
    fn channel_media_buffer_preserves_push_order_within_channel() {
        // Order within a channel matters for chat — "hi" before "bye"
        // must arrive in that order at every subscriber.
        let mut buf = ChannelMediaBuffer::new();
        buf.push(ChannelId(0), fixture_text_frame("ch", "hi"));
        buf.push(ChannelId(0), fixture_text_frame("ch", "bye"));
        let frames = buf.frames_for(ChannelId(0));
        match (&frames[0].payload, &frames[1].payload) {
            (MediaPayload::Text(a), MediaPayload::Text(b)) => {
                assert_eq!(a, "hi");
                assert_eq!(b, "bye");
            }
            _ => panic!("expected text frames"),
        }
    }

    #[test]
    fn outbound_queue_drain_returns_and_empties() {
        let mut q = MediaOutboundQueue::new();
        q.enqueue(MediaTarget::Shard(ShardId(7)), fixture_text_frame("a", "x"));
        q.enqueue(MediaTarget::RadioFrequency(42), fixture_text_frame("b", "y"));
        assert_eq!(q.len(), 2);
        let drained = q.drain();
        assert_eq!(drained.len(), 2);
        assert!(q.is_empty(), "drain leaves the queue empty");
        assert!(matches!(drained[0].0, MediaTarget::Shard(ShardId(7))));
        assert!(matches!(drained[1].0, MediaTarget::RadioFrequency(42)));
    }

    #[test]
    fn outbound_queue_drain_on_empty_returns_empty_vec() {
        let mut q = MediaOutboundQueue::new();
        let drained = q.drain();
        assert!(drained.is_empty());
    }

    // ---------------- HMAC sign/verify ------------------

    #[test]
    fn hmac_round_trips_on_matching_key() {
        let key = [0x42; 32];
        let payload = b"approach corridor A-7";
        let tag = hmac_sign_media(&key, "alice.intercom", 0, 0, 1_700_000, 7, 1, 42, payload);
        assert!(hmac_verify_media(
            &key,
            "alice.intercom",
            0, 0, 1_700_000, 7, 1, 42,
            payload,
            &tag,
        ));
    }

    #[test]
    fn hmac_rejects_wrong_key() {
        let key1 = [0x11; 32];
        let key2 = [0x22; 32];
        let tag = hmac_sign_media(&key1, "ch", 0, 0, 0, 0, 0, 0, b"x");
        assert!(!hmac_verify_media(&key2, "ch", 0, 0, 0, 0, 0, 0, b"x", &tag));
    }

    #[test]
    fn hmac_rejects_payload_tampering() {
        let key = [0x42; 32];
        let tag = hmac_sign_media(&key, "ch", 0, 0, 0, 0, 0, 0, b"hello");
        // Modify payload after signing — must fail.
        assert!(!hmac_verify_media(&key, "ch", 0, 0, 0, 0, 0, 0, b"world", &tag));
    }

    #[test]
    fn hmac_rejects_channel_name_tampering() {
        let key = [0x42; 32];
        let tag = hmac_sign_media(&key, "alice.lights", 0, 0, 0, 0, 0, 0, b"x");
        // Same key + payload but different channel — must fail.
        assert!(!hmac_verify_media(&key, "alice.locks", 0, 0, 0, 0, 0, 0, b"x", &tag));
    }

    #[test]
    fn hmac_rejects_truncated_tag() {
        let key = [0x42; 32];
        let tag = hmac_sign_media(&key, "ch", 0, 0, 0, 0, 0, 0, b"x");
        // 15-byte slice is not a valid tag length.
        assert!(!hmac_verify_media(&key, "ch", 0, 0, 0, 0, 0, 0, b"x", &tag[..15]));
    }

    // ---------------- MediaReplayRegistry ------------------

    #[test]
    fn replay_registry_accepts_first_then_rejects_duplicate() {
        let mut r = MediaReplayRegistry::new();
        let cid = ChannelId(0);
        // First seq: accept.
        assert!(r.try_accept(cid, 1, 100, 1000, 1000).is_ok());
        // Same seq from same sender: replay — reject.
        assert!(r.try_accept(cid, 1, 100, 1000, 1000).is_err());
    }

    #[test]
    fn replay_registry_isolates_per_sender_and_per_channel() {
        let mut r = MediaReplayRegistry::new();
        let cid_a = ChannelId(0);
        let cid_b = ChannelId(1);
        // seq=1 on ch_a from sender 7 — accept.
        assert!(r.try_accept(cid_a, 7, 1, 100, 100).is_ok());
        // seq=1 on ch_a from DIFFERENT sender 8 — also accept.
        assert!(r.try_accept(cid_a, 8, 1, 100, 100).is_ok());
        // seq=1 on ch_b from sender 7 — also accept (different
        // channel keyspace).
        assert!(r.try_accept(cid_b, 7, 1, 100, 100).is_ok());
    }

    // ---------------- try_push_remote_media end-to-end ------------------

    fn fixture_table_with_channel(name: &str) -> SignalChannelTable {
        use crate::signal::types::{ChannelMergeStrategy, SignalScope};
        let mut t = SignalChannelTable::default();
        t.resolve_or_create(name, SignalScope::Local, ChannelMergeStrategy::LastWrite, 42);
        t
    }

    #[test]
    fn try_push_remote_media_accepts_signed_grantless_frame() {
        // No grant: HMAC keyed by the channel's signature.
        let table = fixture_table_with_channel("alice.intercom");
        let cid = table.resolve("alice.intercom").unwrap();
        let key = table.get_by_id(cid).unwrap().signature;
        let grants = GrantsRegistry::default();
        let mut replay = MediaReplayRegistry::new();
        let mut buffer = ChannelMediaBuffer::new();

        let payload = b"approach corridor A-7";
        let now = 1_700_000_000_000_u64;
        let tag = hmac_sign_media(
            &key,
            "alice.intercom",
            payload_kind::TEXT,
            0,
            now,
            7, // sender shard id
            1, // sequence
            0, // grant_id
            payload,
        );
        let frame = MediaFrame {
            channel_name: "alice.intercom".into(),
            grant_id: 0,
            sequence: 1,
            timestamp_ms: now,
            payload: MediaPayload::Text(String::from_utf8(payload.to_vec()).unwrap()),
            auth_tag: tag.to_vec(),
        };
        let result = try_push_remote_media(&table, &grants, &mut replay, &mut buffer, 7, now, frame.clone());
        assert!(result.is_ok(), "valid frame must accept; got {:?}", result);
    }

    #[test]
    fn try_push_remote_media_rejects_unknown_channel() {
        let table = fixture_table_with_channel("alice.intercom");
        let grants = GrantsRegistry::default();
        let mut replay = MediaReplayRegistry::new();
        let mut buffer = ChannelMediaBuffer::new();
        let frame = MediaFrame {
            channel_name: "bob.unknown".into(),
            grant_id: 0,
            sequence: 1,
            timestamp_ms: 0,
            payload: MediaPayload::Text("x".into()),
            auth_tag: vec![],
        };
        let err = try_push_remote_media(&table, &grants, &mut replay, &mut buffer, 7, 0, frame).unwrap_err();
        assert_eq!(err, MediaIngressDenied::UnknownChannel);
    }

    #[test]
    fn try_push_remote_media_rejects_forged_tag() {
        let table = fixture_table_with_channel("alice.intercom");
        let grants = GrantsRegistry::default();
        let mut replay = MediaReplayRegistry::new();
        let mut buffer = ChannelMediaBuffer::new();
        let now = 1_700_000_000_000_u64;
        let frame = MediaFrame {
            channel_name: "alice.intercom".into(),
            grant_id: 0,
            sequence: 1,
            timestamp_ms: now,
            payload: MediaPayload::Text("evil".into()),
            // 16 bytes of attacker garbage — won't survive the
            // constant-time compare.
            auth_tag: vec![0xFF; 16],
        };
        let err = try_push_remote_media(&table, &grants, &mut replay, &mut buffer, 7, now, frame).unwrap_err();
        assert_eq!(err, MediaIngressDenied::AuthFailed);
    }

    #[test]
    fn try_push_remote_media_rejects_replay() {
        let table = fixture_table_with_channel("alice.intercom");
        let cid = table.resolve("alice.intercom").unwrap();
        let key = table.get_by_id(cid).unwrap().signature;
        let grants = GrantsRegistry::default();
        let mut replay = MediaReplayRegistry::new();
        let mut buffer = ChannelMediaBuffer::new();
        let now = 1_700_000_000_000_u64;
        let payload = b"hi";
        let tag = hmac_sign_media(&key, "alice.intercom", payload_kind::TEXT, 0, now, 7, 5, 0, payload);
        let frame = MediaFrame {
            channel_name: "alice.intercom".into(),
            grant_id: 0,
            sequence: 5,
            timestamp_ms: now,
            payload: MediaPayload::Text("hi".into()),
            auth_tag: tag.to_vec(),
        };
        // First accept — fresh.
        try_push_remote_media(&table, &grants, &mut replay, &mut buffer, 7, now, frame.clone()).unwrap();
        // Replay of the SAME (channel, sender, seq) — reject.
        let err = try_push_remote_media(&table, &grants, &mut replay, &mut buffer, 7, now, frame).unwrap_err();
        assert_eq!(err, MediaIngressDenied::Replay);
    }

    #[test]
    fn try_push_remote_media_rejects_stale_timestamp() {
        let table = fixture_table_with_channel("ch");
        let cid = table.resolve("ch").unwrap();
        let key = table.get_by_id(cid).unwrap().signature;
        let grants = GrantsRegistry::default();
        let mut replay = MediaReplayRegistry::new();
        let mut buffer = ChannelMediaBuffer::new();
        // Frame timestamped now=1_000_000; we receive at
        // now=1_000_000 + REPLAY_TIMESTAMP_WINDOW_MS + 1000 (well
        // outside the freshness window).
        let frame_ts = 1_000_000_u64;
        let now = frame_ts + REPLAY_TIMESTAMP_WINDOW_MS + 1000;
        let tag = hmac_sign_media(&key, "ch", payload_kind::TEXT, 0, frame_ts, 7, 1, 0, b"x");
        let frame = MediaFrame {
            channel_name: "ch".into(),
            grant_id: 0,
            sequence: 1,
            timestamp_ms: frame_ts,
            payload: MediaPayload::Text("x".into()),
            auth_tag: tag.to_vec(),
        };
        let err = try_push_remote_media(&table, &grants, &mut replay, &mut buffer, 7, now, frame).unwrap_err();
        assert_eq!(err, MediaIngressDenied::StaleOrFutureTimestamp);
    }

    #[test]
    fn forget_drops_window_so_future_seqs_in_old_range_accepted() {
        let mut r = MediaReplayRegistry::new();
        let cid = ChannelId(0);
        r.try_accept(cid, 1, 100, 1000, 1000).unwrap();
        r.forget(cid, 1);
        assert_eq!(r.entry_count(), 0);
        // After forget, the old seq accepts again.
        assert!(r.try_accept(cid, 1, 100, 1000, 1000).is_ok());
    }

    #[test]
    fn codec_ordinal_per_payload_kind() {
        let a = MediaPayload::Audio {
            codec: AudioCodec::Opus,
            frame: vec![],
            samples: 480,
            sample_rate: 48_000,
        };
        assert_eq!(a.codec_ordinal(), AudioCodec::Opus.as_ordinal());

        let v = MediaPayload::Video {
            codec: VideoCodec::Av1,
            frame: vec![],
            width: 1920,
            height: 1080,
            keyframe: true,
        };
        assert_eq!(v.codec_ordinal(), VideoCodec::Av1.as_ordinal());

        let i = MediaPayload::Image {
            format: ImageFormat::Jpeg,
            data: vec![],
            width: 256,
            height: 256,
        };
        assert_eq!(i.codec_ordinal(), ImageFormat::Jpeg.as_ordinal());
    }
}
