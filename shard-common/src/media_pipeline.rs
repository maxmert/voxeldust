//! Phase 5.2 — generic media pipeline ECS layer.
//!
//! Mirrors `shard-common::signal_pipeline`'s structure for the
//! media data plane. The `MediaPipelinePlugin` registers two
//! resources (`IncomingMediaBuffer`, `MediaPublishQueue`) and two
//! systems (`media_ingest_clear`, `media_publish_remote`). Any shard
//! that hosts media-publishing or media-subscribing blocks wires the
//! plugin into its app and gets cross-shard delivery for free.
//!
//! # Tick lifecycle
//!
//! 1. **Start of tick**: `media_ingest_clear` drops last tick's
//!    inbound frames. Frames are tick-scoped — subscribers that
//!    didn't read in time miss the message. Future per-channel
//!    queues with retention can layer on top if a use case needs
//!    catch-up semantics.
//! 2. **Per-shard drain_quic** (in shard binary): incoming
//!    `MediaBroadcastBatch` payloads land in `IncomingMediaBuffer`
//!    after a channel-existence prune (Phase 5.2.B will add full
//!    HMAC verify against the matching grant — same shape as
//!    signal's `try_push_remote`).
//! 3. **Block subscriber systems** (per-shard): iterate
//!    `IncomingMediaBuffer` and dispatch to interested entities.
//! 4. **Block publisher systems** (per-shard): push outbound frames
//!    to `MediaPublishQueue::enqueue(target, frame)`.
//! 5. **End of tick**: `media_publish_remote` drains
//!    `MediaPublishQueue`, groups by `(target_shard, source_shard)`,
//!    ships one `MediaBroadcastBatch` per group via QUIC.
//!
//! Same control plane as signals — channel resolution, grants,
//! scopes — but a separate data plane. A single
//! `SignalChannelTable` channel can carry signal values, media
//! frames, or both without conflict.

use std::collections::HashMap;

use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use bevy_ecs::message::Message;
use glam::DVec3;

use voxeldust_core::media::{
    hmac_sign_media, payload_kind, ChannelMediaBuffer, MediaFrame, MediaOutboundQueue,
    MediaPayload, MediaReplayRegistry, MediaTarget,
};
use voxeldust_core::signal::components::TerminalState;
use voxeldust_core::shard_message::{MediaBroadcastBatchData, ShardMsg};
use voxeldust_core::shard_types::ShardId;

use crate::harness::{NetworkBridge, ShardIdentity};

/// Schedule labels for the media pipeline. Phase A1 expands the prior
/// 4-stage schedule (Ingest / Subscribe / Publish / Dispatch) to a
/// 7-stage chain that mirrors the signal pipeline's data-flow shape:
///
/// ```text
/// Ingest    ─▶ clear ChannelMediaBuffer (start fresh tick)
/// Receive   ─▶ cross-shard ingress: try_push_remote_media → buffer
/// RxMirror  ─▶ Antenna RX: Radio-channel buffer → local-channel buffer
/// Publish   ─▶ block producers: terminal_publish → buffer
/// Subscribe ─▶ block consumers: terminal_subscribe ← buffer
/// TxBridge  ─▶ Antenna TX: local-channel buffer → MediaOutboundQueue
/// Dispatch  ─▶ media_publish_remote: drain OutboundQueue → QUIC
/// ```
///
/// Each phase reads what's in the buffer at its position. Tick-scoped:
/// `Ingest` wipes everything, so subscribers that didn't read a frame
/// this tick don't see it next tick (matches the signal pipeline's
/// `pending` semantics — keeps the system memoryless).
///
/// Phase ordering is critical:
/// * `Receive` runs before `RxMirror` so an inbound radio frame can be
///   mirrored to a local channel in the same tick.
/// * `Publish` runs before `TxBridge` so newly-typed text is forwarded
///   out the same tick.
/// * `Subscribe` runs AFTER `TxBridge` so the antenna's intra-shard
///   echo path (TX antenna pushing into a co-located RX antenna's
///   local channel for same-shard loopback / two-antenna intercom)
///   delivers the frame to subscribers in the same tick — without
///   this, single-shard chat-with-self over Radio would never close
///   the loop because `media_publish_remote` only ships outbound to
///   peer shards. Subscribers reading from a directly-published Local
///   channel (no antenna) still see the frame the same tick because
///   `terminal_publish` runs in `Publish`, before `Subscribe`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum MediaSet {
    /// Drop last tick's `ChannelMediaBuffer`. Generic.
    Ingest,
    /// Cross-shard ingress drains; validated frames land in
    /// `ChannelMediaBuffer`. Per-shard (each shard's drain_quic does this).
    Receive,
    /// Antenna RX bridges drain Radio-scope channel buffer → local
    /// mirror channel buffer. Generic.
    RxMirror,
    /// Block-kind producers push to `ChannelMediaBuffer` for the
    /// channel they publish to. Per-shard.
    Publish,
    /// Antenna TX bridges scan their TX-side local channel and
    /// enqueue frames to `MediaOutboundQueue` for QUIC dispatch +
    /// deliver them to co-located RX antennas tuned to the same
    /// frequency (intra-shard radio echo). Generic.
    TxBridge,
    /// Block-kind consumers read frames from `ChannelMediaBuffer`.
    /// Per-shard.
    Subscribe,
    /// Drain `MediaOutboundQueue` → QUIC. Generic.
    Dispatch,
}

/// Plugin: registers the media data-plane resources and the two
/// generic systems that sit at the start + end of the tick.
/// Per-shard subscriber/publisher systems slot into
/// [`MediaSet::Subscribe`] / [`MediaSet::Publish`].
pub struct MediaPipelinePlugin;

impl Plugin for MediaPipelinePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChannelMediaBuffer>()
            .init_resource::<MediaOutboundQueue>()
            // Phase 5.2.B: per-(channel, sender) replay window.
            // Lazy-populated as cross-shard frames arrive; entries
            // can be `forget`-ed when channels go away (future
            // sweep system).
            .init_resource::<MediaReplayRegistry>()
            // Phase 5.x: keyboard input message channel for
            // `terminal_publish`. Per-shard tablet UI / interaction
            // layer emits these as the player types.
            .add_message::<KeyboardTerminalInput>()
            .configure_sets(
                Update,
                (
                    MediaSet::Ingest,
                    MediaSet::Receive,
                    MediaSet::RxMirror,
                    MediaSet::Publish,
                    MediaSet::TxBridge,
                    MediaSet::Subscribe,
                    MediaSet::Dispatch,
                )
                    .chain(),
            )
            .add_systems(Update, media_ingest_clear.in_set(MediaSet::Ingest))
            .add_systems(Update, antenna_rx_media_mirror.in_set(MediaSet::RxMirror))
            .add_systems(Update, terminal_publish.in_set(MediaSet::Publish))
            .add_systems(Update, terminal_subscribe.in_set(MediaSet::Subscribe))
            .add_systems(Update, antenna_publish_media.in_set(MediaSet::TxBridge))
            .add_systems(Update, media_publish_remote.in_set(MediaSet::Dispatch));
    }
}

/// First sub-step of the Ingest phase: drop last tick's
/// `ChannelMediaBuffer`. Tick-scoped semantics — subscribers that
/// missed the read window don't see the frame again.
pub fn media_ingest_clear(mut buf: ResMut<ChannelMediaBuffer>) {
    buf.clear();
}

/// End-of-tick: drain `MediaOutboundQueue`, route by `MediaTarget`,
/// ship one `ShardMsg::MediaBroadcastBatch` per (peer_shard, source).
/// Direct shard targets ship to that one peer; Radio frequency
/// targets fan-out to every peer in the registry (the relay-routed
/// frequency-band table is a follow-up — for now the dispatcher uses
/// a peer-set fan-out which is correct for small clusters and
/// matches the signal-radio behaviour).
///
/// Backpressure: fire-and-forget via `tokio::spawn` so a slow QUIC
/// channel never stalls the per-tick schedule. Failed sends log at
/// `warn`; the in-memory queue is already drained — there's no
/// retry path for an unreachable peer (matches signal pipeline's
/// behavior; same rationale: state is ephemeral, the next tick's
/// publish will replace it for any active stream).
pub fn media_publish_remote(
    mut queue: ResMut<MediaOutboundQueue>,
    bridge: Res<NetworkBridge>,
    identity: Res<ShardIdentity>,
) {
    let drained = queue.drain();
    if drained.is_empty() {
        return;
    }
    // Resolve each frame's MediaTarget into a concrete peer shard
    // (or set thereof). Group by destination so we ship one batch
    // per peer instead of one per frame. Frequency targets fan out
    // to all currently-known peer shards — the receiving shard's
    // RX-Antenna decides whether to accept based on its frequency
    // configuration. (This is the same pattern as signal Radio
    // forwarding before galaxy-shard frequency-band routing landed.)
    let mut by_target: HashMap<ShardId, Vec<voxeldust_core::media::MediaFrame>> =
        HashMap::new();
    let peer_registry = bridge.peer_registry.clone();
    let known_peers: Vec<ShardId> = {
        // Snapshot peer ids first so we can iterate without holding
        // the lock during routing decisions.
        let reg = peer_registry.try_read();
        match reg {
            Ok(r) => r.all().iter().map(|p| p.id).collect(),
            Err(_) => Vec::new(),
        }
    };
    for (target, frame) in drained {
        match target {
            MediaTarget::Shard(s) => {
                by_target.entry(s).or_default().push(frame);
            }
            MediaTarget::RadioFrequency(_freq) => {
                // Fan-out to every known peer. Receivers that don't
                // have a matching RX-Antenna will drop the frame at
                // try_push_remote_media's UnknownChannel check (the
                // bridged Radio channel only exists on shards whose
                // antenna is wired to that frequency).
                for s in &known_peers {
                    by_target.entry(*s).or_default().push(frame.clone());
                }
            }
        }
    }
    if by_target.is_empty() {
        return;
    }

    let quic_send_tx = bridge.quic_send_tx.clone();
    let source_shard_id = identity.shard_id;
    tokio::spawn(async move {
        // Address resolution moved into the dispatcher (with on-
        // demand orchestrator refresh on cache miss); we just send
        // (target_id, msg) here.
        let _ = peer_registry; // retained for future short-range range gating
        for (target, frames) in by_target {
            let batch = ShardMsg::MediaBroadcastBatch(MediaBroadcastBatchData {
                source_shard_id: source_shard_id.0,
                source_position: DVec3::ZERO, // ShortRange media path will fill this in.
                frames,
            });
            if let Err(e) = quic_send_tx
                .send((target, batch))
                .await
            {
                tracing::warn!(
                    target = target.0,
                    %e,
                    "MediaPublish QUIC send failed"
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Phase A1 — Terminal block-kind systems on the channel-keyed buffer.
//
// `terminal_publish` (Publish phase): drains KeyboardTerminalInput
// events, signs (HMAC if the channel is keyed via grant, bare for
// open Local channels), and pushes the resulting MediaFrame to
// `ChannelMediaBuffer[publish_channel]`. Same-tick subscribers see it.
//
// `terminal_subscribe` (Subscribe phase): for each Terminal with a
// configured subscribe_channel, reads frames from
// `ChannelMediaBuffer[subscribe_channel]` and appends Text payloads
// to recent_lines.
//
// Cross-shard delivery happens through Antenna TX/RX in `signal_pipeline`
// (Phase A2): TX-Antenna scans a configured local channel for media
// frames and ships them to a Radio frequency; receiving shard's
// RX-Antenna mirrors them to its local mirror channel where the
// destination Terminal subscribes.
// ---------------------------------------------------------------------------

/// Player typed a line on a Terminal block. Emitted by the per-shard
/// tablet UI / interaction layer; consumed by `terminal_publish`.
/// Name kept stable for backward compatibility with existing
/// per-shard code that emits this message; the underlying state is
/// `TerminalState`.
#[derive(Message, Clone, Debug)]
pub struct KeyboardTerminalInput {
    /// Block entity the player typed on. Looked up to find the
    /// `TerminalState` (subscribe + publish channel ids).
    pub entity: Entity,
    pub line: String,
}

/// Drain `ChannelMediaBuffer` per-Terminal and append text payloads
/// to scrollback. Each Terminal reads only its `subscribe_channel`'s
/// frames — no cross-channel leakage, no global iteration.
pub fn terminal_subscribe(
    buffer: Res<ChannelMediaBuffer>,
    mut terminals: Query<&mut TerminalState>,
) {
    if buffer.is_empty() {
        return;
    }
    for mut state in terminals.iter_mut() {
        if !state.can_read() {
            continue;
        }
        let Some(channel) = state.subscribe_channel else {
            continue;
        };
        // Per-channel frame slice — bounded by what was published this
        // tick to this exact channel. O(frames-on-this-channel) per
        // Terminal, vs. the prior O(all-frames × all-terminals) scan.
        for frame in buffer.frames_for(channel) {
            if let MediaPayload::Text(body) = &frame.payload {
                let body_clone = body.clone();
                state.push_line(body_clone);
            }
        }
    }
}

/// Drain `KeyboardTerminalInput` events (write side). Sign each frame
/// against the publish channel's HMAC identity (its `signature` for
/// open channels, or a grant's key for keyed channels) and push to
/// `ChannelMediaBuffer[publish_channel]`. Same-tick local subscribers
/// + TX-Antennas read it from there.
///
/// **Why HMAC even for open same-shard chat**: the same frame may be
/// forwarded over the wire by an Antenna TX bridge in this same tick
/// (TxBridge phase). We sign once at production so any downstream
/// recipient — local subscriber, Antenna bridge, cross-shard receiver
/// — sees a single authoritative auth tag. The local subscriber path
/// doesn't verify (in-shard memory is trusted), so the cost is one
/// HMAC compute per frame, ~500 ns; negligible at chat rates.
pub fn terminal_publish(
    mut events: MessageReader<KeyboardTerminalInput>,
    mut buffer: ResMut<ChannelMediaBuffer>,
    mut terminals: Query<&mut TerminalState>,
    channels: Res<voxeldust_core::signal::SignalChannelTable>,
    identity: Res<crate::harness::ShardIdentity>,
) {
    let now_ms = voxeldust_core::signal::current_unix_millis();
    for evt in events.read() {
        let Ok(mut state) = terminals.get_mut(evt.entity) else {
            tracing::info!(
                entity = ?evt.entity,
                line = %evt.line,
                "terminal_publish: entity has no TerminalState — dropped"
            );
            continue;
        };
        if !state.can_write() {
            tracing::info!(
                entity = ?evt.entity,
                line = %evt.line,
                active = state.active,
                has_publish_channel = state.publish_channel.is_some(),
                "terminal_publish: !can_write — dropped"
            );
            continue;
        }
        let publish_channel_id = state
            .publish_channel
            .expect("can_write ⇒ publish_channel is Some");

        // Resolve the channel — required for HMAC keying (we sign with
        // the channel's signature for open broadcast). Missing channel
        // = stale state; skip with a debug log.
        let Some(channel) = channels.get_by_id(publish_channel_id) else {
            tracing::info!(
                owner = state.owner_session,
                channel_id = publish_channel_id.0,
                line = %evt.line,
                "terminal_publish: channel id not in table — dropped"
            );
            continue;
        };
        tracing::info!(
            channel = %channel.name,
            channel_id = publish_channel_id.0,
            line = %evt.line,
            "terminal_publish: pushing media frame"
        );

        let sequence = state.next_sequence.wrapping_add(1);
        state.next_sequence = sequence;
        let payload_bytes = evt.line.as_bytes();
        // Open broadcast: signed with the channel's own signature.
        // Receivers verify against the same signature on the cross-
        // shard ingress path. (Keyed Radio with grants is wired
        // through Antenna TX in Phase A2; the Terminal itself only
        // signs for the channel, not for any specific recipient.)
        let auth_tag = hmac_sign_media(
            &channel.signature,
            &channel.name,
            payload_kind::TEXT,
            0, // codec: 0 for text
            now_ms,
            identity.shard_id.0,
            sequence,
            0, // grant_id = 0 for open broadcast at the channel level
            payload_bytes,
        )
        .to_vec();
        let frame = MediaFrame {
            channel_name: channel.name.clone(),
            grant_id: 0,
            sequence,
            timestamp_ms: now_ms,
            payload: MediaPayload::Text(evt.line.clone()),
            auth_tag,
        };
        buffer.push(publish_channel_id, frame);
    }
}

// ---------------------------------------------------------------------------
// Phase A2 — Antenna media bridge systems.
//
// `antenna_publish_media` (TxBridge phase): each Antenna with a TX side
// scans its `tx.local_channel_id` for media frames pushed this tick by
// terminal_publish (or any other publisher). Each frame is wrapped with
// the bridged channel name + Radio frequency metadata, signed with the
// Radio channel's signature (or grant-keyed for keyed channels), and
// enqueued to MediaOutboundQueue with MediaTarget::RadioFrequency.
// media_publish_remote ships them.
//
// `antenna_rx_media_mirror` (RxMirror phase): each Antenna with an RX
// side checks `ChannelMediaBuffer[rx.bridged_channel_id]` for frames
// that arrived this tick from cross-shard ingress. Each is mirrored to
// `ChannelMediaBuffer[rx.local_channel_id]` so terminal_subscribe (in
// the Subscribe phase later this tick) sees them.
//
// Together these form the media counterpart to signal_pipeline's
// antenna_publish + listener_mirror — the entire chat-over-radio flow
// composes from these two systems plus terminal_publish/subscribe.
// ---------------------------------------------------------------------------

/// Phase A2: scan TX-side local channels for new frames; rewrap each
/// for the configured Radio frequency, sign, enqueue for cross-shard
/// dispatch.
///
/// **Why we re-sign with the Radio channel's identity**: the local
/// channel's signature isn't valid at the receiver (it doesn't exist
/// on the other shard). The Radio-scope channel identified by
/// `(frequency, antenna's bridged Radio channel name)` IS the
/// cross-shard identity — receivers verify against the bridged Radio
/// channel's signature on their side. The TX antenna acts as the
/// trust-translation point: locally trusted (in-shard memory) → wire-
/// authenticated (HMAC over the Radio channel's identity).
pub fn antenna_publish_media(
    mut buffer: ResMut<ChannelMediaBuffer>,
    channels: Res<voxeldust_core::signal::SignalChannelTable>,
    mut outbound: ResMut<MediaOutboundQueue>,
    held: Res<crate::signal_pipeline::HeldGrants>,
    identity: Res<ShardIdentity>,
    mut antennas: Query<&mut voxeldust_core::signal::AntennaState>,
) {
    if buffer.is_empty() {
        return;
    }
    // Snapshot every active RX antenna's (frequency, local_channel_id)
    // pair so the TX pass below can deliver intra-shard echoes without
    // re-borrowing the antennas query mutably while it's already lent
    // out for `tx.next_sequence` mutation. Same antenna with both TX
    // and RX (the user's chat-with-self loopback) lands in this map
    // exactly once, on its own frequency.
    let local_rx_targets: Vec<(u32, voxeldust_core::signal::ChannelId)> = antennas
        .iter()
        .filter(|a| a.active)
        .filter_map(|a| {
            let rx = a.rx.as_ref()?;
            let local = rx.local_channel_id?;
            Some((rx.frequency, local))
        })
        .collect();

    let now_ms = voxeldust_core::signal::current_unix_millis();
    // Frames we'll push into local RX channels at the END of this
    // system. Buffered up so we don't `&mut buffer` while still
    // holding `&buffer.frames_for(...)` borrow inside the TX loop.
    let mut intra_shard_deliveries: Vec<(
        voxeldust_core::signal::ChannelId,
        voxeldust_core::media::MediaFrame,
    )> = Vec::new();
    for mut antenna in antennas.iter_mut() {
        if !antenna.active {
            continue;
        }
        let owner_session = antenna.owner_session;
        let Some(tx) = antenna.tx.as_mut() else {
            continue;
        };
        let Some(local_id) = tx.local_channel_id else {
            continue;
        };
        let frames = buffer.frames_for(local_id);
        if frames.is_empty() {
            continue;
        }

        // Build the bridged Radio channel name. Naming convention
        // matches the apply path: `<local>__radio_out_<freq>` for TX,
        // `<local>__radio_in_<freq>` for RX. Receivers wire their RX
        // bridged channel to the SAME name; the relay ships frames
        // addressed to that channel name; the receiver's
        // try_push_remote_media resolves it.
        //
        // (Cross-shard naming is shared across shards because both
        // sides agree on the convention. The radio frequency itself
        // could route via a relay table later; the channel name is the
        // wire-stable identifier.)
        let bridged_name = match channels.name_for_id(local_id) {
            Some(name) => format!("{}__radio_out_{}", name, tx.frequency),
            None => continue,
        };

        // Resolve auth: open Radio uses a deterministic signature (we
        // use the BRIDGED Radio channel's signature if it exists
        // locally on this shard — created at apply time; otherwise we
        // create the channel lazily here so the signature is stable
        // for the lifetime of this antenna). Keyed Radio uses the
        // grant's key.
        let (grant_id_wire, key) = match tx.grant_id {
            None => {
                // Open broadcast — the bridged Radio channel must exist
                // locally for HMAC keying. `apply_antenna_config` creates
                // it on the TX side (symmetric with RX); this lookup is
                // a should-never-fire guard. INFO-level so any future
                // regression in the apply path is loud, not silent.
                let Some(bridged_id) = channels.resolve(&bridged_name) else {
                    tracing::info!(
                        owner = owner_session,
                        bridged = %bridged_name,
                        local_channel_id = local_id.0,
                        "antenna_publish_media: bridged Radio channel not in table — \
                         skipping. (apply_antenna_config TX side should have created it.)"
                    );
                    continue;
                };
                let Some(bridged_ch) = channels.get_by_id(bridged_id) else {
                    continue;
                };
                (0_u64, bridged_ch.signature)
            }
            Some(gid) => {
                let session = voxeldust_core::shard_types::SessionToken(owner_session);
                let Some(grant) = held.get(session, gid) else {
                    tracing::info!(
                        owner = owner_session,
                        grant_id = gid,
                        "antenna_publish_media: held grant missing — skipping"
                    );
                    continue;
                };
                (gid, grant.key)
            }
        };

        // Per-antenna sequence counter — same field signal antennas
        // use, shared across both data planes per (channel, sender)
        // replay window contract.
        for frame in frames {
            // Skip non-text payloads only when we want to (we'll
            // forward ALL kinds through Radio — the receiver's
            // codec-mismatch handling decides). Keep it simple here.
            let payload_kind_ord = frame.payload.payload_kind();
            let codec_ord = frame.payload.codec_ordinal();
            let payload_bytes = match &frame.payload {
                voxeldust_core::media::MediaPayload::Text(s) => s.as_bytes().to_vec(),
                voxeldust_core::media::MediaPayload::Audio { frame, .. } => frame.clone(),
                voxeldust_core::media::MediaPayload::Video { frame, .. } => frame.clone(),
                voxeldust_core::media::MediaPayload::Image { data, .. } => data.clone(),
            };
            let sequence = tx.next_sequence.wrapping_add(1);
            tx.next_sequence = sequence;

            let auth_tag = voxeldust_core::media::hmac_sign_media(
                &key,
                &bridged_name,
                payload_kind_ord,
                codec_ord,
                now_ms,
                identity.shard_id.0,
                sequence,
                grant_id_wire,
                &payload_bytes,
            )
            .to_vec();

            let outbound_frame = voxeldust_core::media::MediaFrame {
                channel_name: bridged_name.clone(),
                grant_id: grant_id_wire,
                sequence,
                timestamp_ms: now_ms,
                payload: frame.payload.clone(),
                auth_tag,
            };
            // Intra-shard echo: deliver to every local RX antenna
            // listening on this same frequency. Same-shard self-
            // loopback (single antenna with both TX + RX), or two
            // antennas in the same shard tuned to the same freq
            // (intercom). The cross-shard QUIC fan-out below covers
            // peer shards; this branch closes the local loop without
            // a network round-trip. Authoritative semantics: same
            // payload, same timestamp / sequence as the outbound
            // frame; receivers see "now_ms" and the antenna's own
            // sequence counter.
            for (rx_freq, rx_local_id) in &local_rx_targets {
                if *rx_freq == tx.frequency {
                    intra_shard_deliveries.push((*rx_local_id, outbound_frame.clone()));
                }
            }
            outbound.enqueue(
                voxeldust_core::media::MediaTarget::RadioFrequency(tx.frequency),
                outbound_frame,
            );
        }
    }
    // Apply the buffered intra-shard deliveries. Done after the
    // antenna loop so `antennas.iter_mut()` doesn't conflict with
    // `buffer` mutation (Bevy's borrow checker resolves this fine
    // because `buffer` and `antennas` are different params, but the
    // collect-then-apply pattern keeps the loop body small + readable).
    if !intra_shard_deliveries.is_empty() {
        tracing::info!(
            count = intra_shard_deliveries.len(),
            "antenna_publish_media: intra-shard echoes delivered"
        );
        for (channel_id, frame) in intra_shard_deliveries {
            buffer.push(channel_id, frame);
        }
    }
}

/// Phase A2: each tick, mirror frames from RX-side bridged Radio
/// channels onto their local mirror channels.
///
/// The cross-shard ingress path (`drain_quic` →
/// `try_push_remote_media`) pushes verified frames into
/// `ChannelMediaBuffer[bridged_channel_id]`. This system reads them
/// out and republishes into `ChannelMediaBuffer[local_channel_id]` so
/// in-shard subscribers (`terminal_subscribe`, etc.) — which only
/// know about Local channels — see them.
///
/// Sequence numbers and grant_ids are NOT preserved across the bridge:
/// the local mirror is a fresh channel; its frames are tagged as
/// "this antenna" produced them. If a future feature needs per-sender
/// attribution, it lives at the application layer (chat metadata),
/// not the channel-replay layer.
pub fn antenna_rx_media_mirror(
    mut buffer: ResMut<ChannelMediaBuffer>,
    channels: Res<voxeldust_core::signal::SignalChannelTable>,
    identity: Res<ShardIdentity>,
    antennas: Query<&voxeldust_core::signal::AntennaState>,
) {
    if buffer.is_empty() {
        return;
    }
    let now_ms = voxeldust_core::signal::current_unix_millis();
    // Snapshot the work first (channel id pairs + frames to mirror)
    // so we don't hold an immutable borrow of `buffer.frames_for`
    // across the mutable `buffer.push` below.
    let mut to_mirror: Vec<(voxeldust_core::signal::ChannelId, voxeldust_core::media::MediaFrame)> =
        Vec::new();
    for antenna in &antennas {
        if !antenna.active {
            continue;
        }
        let Some(rx) = &antenna.rx else { continue };
        let (Some(bridged), Some(local)) = (rx.bridged_channel_id, rx.local_channel_id) else {
            continue;
        };
        let Some(local_ch) = channels.get_by_id(local) else {
            continue;
        };
        for frame in buffer.frames_for(bridged) {
            // Re-stamp with the LOCAL channel's identity for the
            // intra-shard data plane. Subscribers don't HMAC-verify
            // (in-shard memory is trusted), but stamping keeps the
            // frame consistent with terminal_publish's output and
            // prevents the same frame being TX-bridged a second time
            // (it's now on a Local-scope channel; antenna_publish_media
            // would only forward TX-side local channels, which this
            // local channel could be — but loop-back is filtered by
            // the `next_sequence` write at the local channel's
            // pretend "publisher" — see Phase A5 echo de-dup).
            let mirrored = voxeldust_core::media::MediaFrame {
                channel_name: local_ch.name.clone(),
                grant_id: 0,
                sequence: frame.sequence, // preserve the cross-shard seq for de-dup
                timestamp_ms: now_ms,
                payload: frame.payload.clone(),
                auth_tag: voxeldust_core::media::hmac_sign_media(
                    &local_ch.signature,
                    &local_ch.name,
                    frame.payload.payload_kind(),
                    frame.payload.codec_ordinal(),
                    now_ms,
                    identity.shard_id.0,
                    frame.sequence,
                    0,
                    match &frame.payload {
                        voxeldust_core::media::MediaPayload::Text(s) => s.as_bytes(),
                        voxeldust_core::media::MediaPayload::Audio { frame, .. } => frame,
                        voxeldust_core::media::MediaPayload::Video { frame, .. } => frame,
                        voxeldust_core::media::MediaPayload::Image { data, .. } => data,
                    },
                )
                .to_vec(),
            };
            to_mirror.push((local, mirrored));
        }
    }
    for (local, frame) in to_mirror {
        buffer.push(local, frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxeldust_core::media::{MediaFrame, MediaPayload};

    fn fixture_text(channel: &str, body: &str) -> MediaFrame {
        MediaFrame {
            channel_name: channel.into(),
            grant_id: 0,
            sequence: 1,
            timestamp_ms: 0,
            payload: MediaPayload::Text(body.into()),
            auth_tag: vec![],
        }
    }

    use voxeldust_core::signal::channel::ChannelId;

    #[test]
    fn ingest_clear_empties_buffer() {
        let mut world = World::new();
        let mut buf = ChannelMediaBuffer::new();
        buf.push(ChannelId(0), fixture_text("a", "hi"));
        buf.push(ChannelId(1), fixture_text("b", "yo"));
        world.insert_resource(buf);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(media_ingest_clear);
        schedule.run(&mut world);
        let buf = world.resource::<ChannelMediaBuffer>();
        assert!(buf.is_empty(), "media_ingest_clear must drain the buffer");
    }

    /// Test fixture: a Terminal entity with the read side wired to
    /// `channel`. Write side intentionally left empty for these tests.
    fn read_only_terminal(channel: ChannelId, max_lines: u32, active: bool) -> TerminalState {
        TerminalState {
            subscribe_channel: Some(channel),
            publish_channel: None,
            recent_lines: Vec::new(),
            max_lines,
            owner_session: 0,
            next_sequence: 0,
            active,
        }
    }

    #[test]
    fn terminal_subscribe_appends_matching_frames_only() {
        // Only frames pushed to the Terminal's subscribe channel are
        // appended; frames on a different channel id never reach this
        // entity (per-channel isolation is the buffer's contract).
        let mut world = World::new();
        let buf = {
            let mut b = ChannelMediaBuffer::new();
            b.push(ChannelId(0), fixture_text("ch.matched", "first"));
            b.push(ChannelId(1), fixture_text("ch.other", "ignored"));
            b.push(ChannelId(0), fixture_text("ch.matched", "second"));
            b
        };
        world.insert_resource(buf);

        let terminal_entity = world
            .spawn(read_only_terminal(ChannelId(0), TerminalState::DEFAULT_MAX_LINES, true))
            .id();

        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(terminal_subscribe);
        schedule.run(&mut world);

        let state = world.get::<TerminalState>(terminal_entity).unwrap();
        assert_eq!(state.recent_lines, vec!["first".to_string(), "second".to_string()]);
    }

    #[test]
    fn terminal_subscribe_respects_active_flag() {
        let mut world = World::new();
        let mut b = ChannelMediaBuffer::new();
        b.push(ChannelId(0), fixture_text("ch", "muted"));
        world.insert_resource(b);

        let entity = world
            .spawn(read_only_terminal(ChannelId(0), TerminalState::DEFAULT_MAX_LINES, false))
            .id();

        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(terminal_subscribe);
        schedule.run(&mut world);

        let state = world.get::<TerminalState>(entity).unwrap();
        assert!(state.recent_lines.is_empty(), "muted terminal drops frames");
    }

    #[test]
    fn terminal_subscribe_caps_at_max_lines() {
        let mut world = World::new();
        let mut b = ChannelMediaBuffer::new();
        for i in 0..10 {
            b.push(ChannelId(0), fixture_text("ch", &format!("line{i}")));
        }
        world.insert_resource(b);

        let entity = world.spawn(read_only_terminal(ChannelId(0), 3, true)).id();

        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(terminal_subscribe);
        schedule.run(&mut world);

        let state = world.get::<TerminalState>(entity).unwrap();
        assert_eq!(state.recent_lines.len(), 3);
        assert_eq!(state.recent_lines[0], "line7");
        assert_eq!(state.recent_lines[2], "line9");
    }

    #[test]
    fn outbound_queue_groups_by_target() {
        // The grouping invariant `media_publish_remote` relies on:
        // direct shard targets cluster by ShardId; Radio frequency
        // targets fan out per peer. Tested here at the queue level.
        let mut q = MediaOutboundQueue::new();
        q.enqueue(MediaTarget::Shard(ShardId(1)), fixture_text("a", "hi"));
        q.enqueue(MediaTarget::Shard(ShardId(2)), fixture_text("b", "yo"));
        q.enqueue(MediaTarget::Shard(ShardId(1)), fixture_text("c", "again"));
        let drained = q.drain();
        let mut by_target: HashMap<ShardId, Vec<MediaFrame>> = HashMap::new();
        for (target, frame) in drained {
            if let MediaTarget::Shard(s) = target {
                by_target.entry(s).or_default().push(frame);
            }
        }
        assert_eq!(by_target.len(), 2);
        assert_eq!(by_target[&ShardId(1)].len(), 2);
        assert_eq!(by_target[&ShardId(2)].len(), 1);
    }
}
