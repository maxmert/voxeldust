use std::net::SocketAddr;
use std::time::Duration;

use tokio::net::UdpSocket;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use voxeldust_core::shard_message::ShardMsg;
use voxeldust_core::shard_types::{ShardHeartbeat, ShardId};

/// Periodically sends heartbeat to the orchestrator via UDP.
/// The address supports DNS names (resolved via tokio).
pub async fn run_heartbeat_sender(
    shard_id: ShardId,
    orchestrator_heartbeat_addr: String,
    metrics_fn: Box<dyn Fn() -> (f32, f32, u32, u32) + Send>, // (tick_ms, p99, players, chunks)
    cancel: CancellationToken,
) {
    // Resolve the address (supports DNS names like K8s service names).
    let resolved_addr: SocketAddr = match tokio::net::lookup_host(&orchestrator_heartbeat_addr).await {
        Ok(mut addrs) => match addrs.next() {
            Some(addr) => addr,
            None => {
                warn!(addr = %orchestrator_heartbeat_addr, "heartbeat address resolved to nothing");
                return;
            }
        },
        Err(e) => {
            warn!(%e, addr = %orchestrator_heartbeat_addr, "failed to resolve heartbeat address");
            return;
        }
    };
    let socket = match UdpSocket::bind("0.0.0.0:0").await {
        Ok(s) => s,
        Err(e) => {
            warn!(%e, "failed to bind heartbeat sender socket");
            return;
        }
    };

    let mut interval = tokio::time::interval(Duration::from_secs(2));

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                info!("heartbeat sender shutting down");
                return;
            }
            _ = interval.tick() => {
                let (tick_ms, p99_tick_ms, player_count, chunk_count) = metrics_fn();

                let msg = ShardMsg::Heartbeat(ShardHeartbeat {
                    shard_id,
                    tick_ms,
                    p99_tick_ms,
                    player_count,
                    chunk_count,
                });

                let payload = msg.serialize();
                let len_bytes = (payload.len() as u32).to_be_bytes();

                let mut packet = Vec::with_capacity(4 + payload.len());
                packet.extend_from_slice(&len_bytes);
                packet.extend_from_slice(&payload);

                match socket.send_to(&packet, resolved_addr).await {
                    Ok(_) => {
                        debug!(shard_id = %shard_id, "heartbeat sent");
                    }
                    Err(e) => {
                        warn!(%e, "failed to send heartbeat");
                    }
                }
            }
        }
    }
}
