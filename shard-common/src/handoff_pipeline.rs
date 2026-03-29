use std::net::SocketAddr;
use std::sync::Arc;

use tracing::{info, warn};

use voxeldust_core::handoff::{HandoffAccepted, PlayerHandoff};
use voxeldust_core::shard_message::ShardMsg;
use voxeldust_core::shard_types::ShardId;

use crate::peer_registry::PeerShardRegistry;
use crate::quic_transport::QuicTransport;

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
