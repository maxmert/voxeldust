use std::net::SocketAddr;
use std::sync::Arc;

use glam::{DQuat, DVec3};
use tracing::{info, warn};

use voxeldust_core::client_message::ServerMsg;
use voxeldust_core::handoff::{self, HandoffAccepted, PlayerHandoff, SpawnPose};
use voxeldust_core::shard_message::ShardMsg;
use voxeldust_core::shard_types::{ShardId, ShardInfo};

use crate::peer_registry::PeerShardRegistry;
use crate::quic_transport::QuicTransport;

/// Default value of `ShardHandoff.source_demote_after_ticks`. Matches
/// the wire-protocol default (`protocol/voxeldust.fbs:1510`). 15 ticks
/// at the 20 Hz server cadence ≈ 750 ms — long enough that the
/// destination's first authoritative tick reaches every observer
/// before the source ghost goes away, short enough that the demoted
/// connection doesn't waste server bandwidth.
pub const DEFAULT_SOURCE_DEMOTE_TICKS: u8 = 15;

/// Describes what kind of shard a player should be handed off to.
#[derive(Debug, Clone)]
pub struct HandoffTarget {
    pub shard_type: voxeldust_core::shard_types::ShardType,
    pub planet_seed: Option<u64>,
    pub system_seed: Option<u64>,
    pub target_shard_id: Option<ShardId>,
}

/// Sends a player handoff to a target shard via QUIC.
pub async fn send_handoff(
    transport: &Arc<QuicTransport>,
    peer_registry: &PeerShardRegistry,
    target_shard_id: ShardId,
    handoff: PlayerHandoff,
) -> Result<(), String> {
    let peer_addr = peer_registry
        .quic_addr(target_shard_id)
        .ok_or_else(|| format!("no QUIC address for shard {target_shard_id}"))?;

    info!(
        session = handoff.session_token.0,
        target = %target_shard_id,
        player = %handoff.player_name,
        "sending player handoff"
    );

    let msg = ShardMsg::PlayerHandoff(handoff);
    transport
        .send(target_shard_id, peer_addr, &msg)
        .await
        .map_err(|e| format!("QUIC send failed: {e}"))?;

    Ok(())
}

/// Build the `ServerMsg` the source shard sends to the client after a
/// `HandoffAccepted` arrives from the destination shard. Returns:
///
///   * [`ServerMsg::ShardHandoff`] when `accepted.observer_promoted`
///     is `true` — the destination has the player's session as a
///     pre-connected observer and has already promoted that
///     observer's TCP into its primary client entry. The client
///     promotes its existing secondary connection in place — true
///     zero-RTT seamless transition (no fresh TCP handshake, no
///     chunk re-stream, no PlayerInput gap).
///   * [`ServerMsg::ShardRedirect`] otherwise — legacy fresh-
///     handshake path. Client tears down the old primary, opens a
///     new TCP+UDP pair to the destination, performs `Connect` →
///     `JoinResponse`. ~5-50 ms TCP handshake gap.
///
/// `spawn_pose` is the post-transition camera pose the client
/// should render at on its first frame after the switch. For the
/// `ShardRedirect` path it lands in `spawn_pose: Option<SpawnPose>`;
/// for the `ShardHandoff` path it splits into the three explicit
/// `handoff_position`/`_velocity`/`_rotation` fields. Defaults to
/// `accepted.spawn_pose` when `None`; callers that compute a more
/// authoritative pose (e.g. system-shard's `authoritative_spawn`
/// from a fresh ship-index lookup) pass `Some(...)` to override.
///
/// This helper is the single source of truth for the
/// ShardHandoff-vs-ShardRedirect choice — every per-shard
/// `HandoffAccepted` consumer (ship-shard, system-shard,
/// planet-shard) calls into here so the invariant
/// "observer_promoted=true ⇒ ShardHandoff" can never silently
/// regress in just one of them.
pub fn build_post_handoff_redirect(
    accepted: &HandoffAccepted,
    target_peer: &ShardInfo,
    spawn_pose: Option<SpawnPose>,
) -> ServerMsg {
    let pose = spawn_pose.or_else(|| accepted.spawn_pose.clone());
    let target_tcp_addr = target_peer.endpoint.tcp_addr.to_string();
    let target_udp_addr = target_peer.endpoint.udp_addr.to_string();
    let target_shard_type = target_peer.shard_type as u8;

    if accepted.observer_promoted {
        let (handoff_position, handoff_rotation, handoff_velocity) = match &pose {
            Some(sp) => (sp.position, sp.rotation, sp.velocity),
            None => (DVec3::ZERO, DQuat::IDENTITY, DVec3::ZERO),
        };
        ServerMsg::ShardHandoff(handoff::ShardHandoff {
            session_token: accepted.session_token,
            promote_shard_type: target_shard_type,
            promote_shard_id: accepted.target_shard,
            target_tcp_addr,
            target_udp_addr,
            source_demote_after_ticks: DEFAULT_SOURCE_DEMOTE_TICKS,
            handoff_position,
            handoff_velocity,
            handoff_rotation,
        })
    } else {
        ServerMsg::ShardRedirect(handoff::ShardRedirect {
            session_token: accepted.session_token,
            target_tcp_addr,
            target_udp_addr,
            shard_id: accepted.target_shard,
            target_shard_type,
            spawn_pose: pose,
        })
    }
}

/// Sends a HandoffAccepted confirmation back to the source shard.
pub async fn send_handoff_accepted(
    transport: &Arc<QuicTransport>,
    peer_registry: &PeerShardRegistry,
    source_shard_id: ShardId,
    accepted: HandoffAccepted,
) -> Result<(), String> {
    let peer_addr = peer_registry
        .quic_addr(source_shard_id)
        .ok_or_else(|| format!("no QUIC address for shard {source_shard_id}"))?;

    let msg = ShardMsg::HandoffAccepted(accepted);
    transport
        .send(source_shard_id, peer_addr, &msg)
        .await
        .map_err(|e| format!("QUIC send failed: {e}"))?;

    Ok(())
}
