use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::UdpSocket;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use voxeldust_core::shard_message::ShardMsg;

use crate::registry::ShardRegistry;

/// Listens for UDP heartbeats from shards.
///
/// Each heartbeat is a FlatBuffer `ShardMessage` with `ShardHeartbeat` payload,
/// preceded by a 4-byte big-endian length prefix.
pub async fn run_heartbeat_listener(
    addr: SocketAddr,
    registry: Arc<RwLock<ShardRegistry>>,
    cancel: CancellationToken,
) {
    let socket = UdpSocket::bind(addr)
        .await
        .expect("failed to bind heartbeat UDP socket");
    info!(%addr, "heartbeat listener ready");

    let mut buf = vec![0u8; 65536];

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                info!("heartbeat listener shutting down");
                return;
            }
            result = socket.recv_from(&mut buf) => {
                match result {
                    Ok((len, src)) => {
                        handle_heartbeat_packet(&buf[..len], src, &registry).await;
                    }
                    Err(e) => {
                        warn!(%e, "heartbeat recv error");
                    }
                }
            }
        }
    }
}

async fn handle_heartbeat_packet(
    data: &[u8],
    src: SocketAddr,
    registry: &Arc<RwLock<ShardRegistry>>,
) {
    // Expect 4-byte length prefix + payload.
    if data.len() < 4 {
        warn!(%src, len = data.len(), "heartbeat packet too small");
        return;
    }

    let len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if data.len() < 4 + len {
        warn!(%src, expected = len, actual = data.len() - 4, "heartbeat packet truncated");
        return;
    }

    let payload = &data[4..4 + len];
    match ShardMsg::deserialize(payload) {
        Ok(ShardMsg::Heartbeat(hb)) => {
            debug!(shard_id = %hb.shard_id, players = hb.player_count, tick_ms = hb.tick_ms, "heartbeat received");
            let mut reg = registry.write().await;
            reg.update_heartbeat(&hb);
        }
        Ok(other) => {
            warn!(%src, msg_type = ?std::mem::discriminant(&other), "unexpected message type on heartbeat channel");
        }
        Err(e) => {
            warn!(%src, %e, "failed to deserialize heartbeat");
        }
    }
}
