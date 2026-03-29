//! Client networking: connect to gateway, handle redirect, receive WorldState.

use std::net::SocketAddr;
use std::sync::Arc;


use glam::{DQuat, DVec3};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::mpsc;
use tracing::{info, warn};

use voxeldust_core::client_message::{
    ClientMsg, JoinResponseData, PlayerInputData, ServerMsg, WorldStateData,
};
use voxeldust_core::handoff::ShardRedirect;

/// Events from the network thread to the render thread.
pub enum NetEvent {
    Connected {
        shard_type: u8,
        reference_position: DVec3,
        reference_rotation: DQuat,
        game_time: f64,
        system_seed: u64,
    },
    WorldState(WorldStateData),
    Transitioning,
    Disconnected(String),
}

/// Run the network loop on a tokio runtime.
pub async fn run_network(
    gateway_addr: SocketAddr,
    player_name: String,
    event_tx: mpsc::UnboundedSender<NetEvent>,
    input_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<PlayerInputData>>>,
    direct: Option<String>,
) {
    // Determine shard TCP/UDP addresses.
    let (shard_tcp_addr, shard_udp_addr) = if let Some(ref direct_addr) = direct {
        let parts: Vec<&str> = direct_addr.split(',').collect();
        let tcp: SocketAddr = parts[0].parse().expect("bad direct tcp addr");
        let udp: SocketAddr = if parts.len() > 1 {
            parts[1].parse().expect("bad direct udp addr")
        } else {
            SocketAddr::new(tcp.ip(), tcp.port() + 1)
        };
        info!(%tcp, %udp, "direct shard connection");
        (tcp, udp)
    } else {
        info!(%gateway_addr, "connecting to gateway");
        let redirect = match connect_to_gateway(gateway_addr, &player_name).await {
            Ok(r) => r,
            Err(e) => {
                let _ = event_tx.send(NetEvent::Disconnected(format!("gateway error: {e}")));
                return;
            }
        };
        info!(tcp = %redirect.target_tcp_addr, udp = %redirect.target_udp_addr, "received shard redirect");

        let tcp: SocketAddr = match redirect.target_tcp_addr.parse() {
            Ok(a) => a,
            Err(e) => { let _ = event_tx.send(NetEvent::Disconnected(format!("bad tcp: {e}"))); return; }
        };
        let udp: SocketAddr = match redirect.target_udp_addr.parse() {
            Ok(a) => a,
            Err(e) => { let _ = event_tx.send(NetEvent::Disconnected(format!("bad udp: {e}"))); return; }
        };
        (tcp, udp)
    };

    // Connect to shard.
    info!(%shard_tcp_addr, "connecting to shard");
    let jr = match connect_to_shard(shard_tcp_addr, &player_name).await {
        Ok(jr) => jr,
        Err(e) => {
            let _ = event_tx.send(NetEvent::Disconnected(format!("shard connect error: {e}")));
            return;
        }
    };

    info!(shard_type = jr.shard_type, "joined shard");
    let _ = event_tx.send(NetEvent::Connected {
        shard_type: jr.shard_type,
        reference_position: jr.reference_position,
        reference_rotation: jr.reference_rotation,
        game_time: jr.game_time,
        system_seed: jr.system_seed,
    });

    // UDP loop.
    let udp = match UdpSocket::bind("0.0.0.0:0").await {
        Ok(s) => s,
        Err(e) => { let _ = event_tx.send(NetEvent::Disconnected(format!("udp: {e}"))); return; }
    };

    // Send initial hole-punch.
    let hello = build_input(&empty_input());
    let _ = udp.send_to(&hello, shard_udp_addr).await;
    info!(%shard_udp_addr, "UDP hole-punch sent");

    let udp = Arc::new(udp);
    let udp_send = udp.clone();

    // Input sender: reads from main thread, sends at 20Hz.
    let send_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(50));
        loop {
            interval.tick().await;

            let input = {
                let mut rx = input_rx.lock().await;
                // Drain to latest input. For pilot yaw/pitch rate, latest-wins is correct
                // since the rate represents the current desired turn speed, not a delta.
                let mut latest = empty_input();
                while let Ok(i) = rx.try_recv() {
                    latest = i;
                }
                latest
            };

            let pkt = build_input(&input);
            let _ = udp_send.send_to(&pkt, shard_udp_addr).await;
        }
    });

    // WorldState receiver.
    let mut buf = vec![0u8; 65536];
    loop {
        match udp.recv_from(&mut buf).await {
            Ok((len, _)) => {
                if len < 4 { continue; }
                let msg_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                if len < 4 + msg_len { continue; }
                match ServerMsg::deserialize(&buf[4..4 + msg_len]) {
                    Ok(ServerMsg::WorldState(ws)) => {
                        let _ = event_tx.send(NetEvent::WorldState(ws));
                    }
                    _ => {}
                }
            }
            Err(e) => { warn!(%e, "UDP recv error"); break; }
        }
    }

    send_handle.abort();
}

fn build_input(input: &PlayerInputData) -> Vec<u8> {
    let msg = ClientMsg::PlayerInput(input.clone());
    let data = msg.serialize();
    let mut pkt = Vec::with_capacity(4 + data.len());
    pkt.extend_from_slice(&(data.len() as u32).to_be_bytes());
    pkt.extend_from_slice(&data);
    pkt
}

async fn connect_to_gateway(
    addr: SocketAddr,
    player_name: &str,
) -> Result<ShardRedirect, Box<dyn std::error::Error + Send + Sync>> {
    let mut stream = TcpStream::connect(addr).await?;
    send_msg(&mut stream, &ClientMsg::Connect { player_name: player_name.to_string() }).await?;
    let response = recv_server_msg(&mut stream).await?;
    match response {
        ServerMsg::ShardRedirect(r) => Ok(r),
        other => Err(format!("expected ShardRedirect, got {:?}", std::mem::discriminant(&other)).into()),
    }
}

async fn connect_to_shard(
    addr: SocketAddr,
    player_name: &str,
) -> Result<JoinResponseData, Box<dyn std::error::Error + Send + Sync>> {
    let mut stream = TcpStream::connect(addr).await?;
    send_msg(&mut stream, &ClientMsg::Connect { player_name: player_name.to_string() }).await?;
    let response = recv_server_msg(&mut stream).await?;
    match response {
        ServerMsg::JoinResponse(jr) => Ok(jr),
        other => Err(format!("expected JoinResponse, got {:?}", std::mem::discriminant(&other)).into()),
    }
}

async fn send_msg(stream: &mut TcpStream, msg: &ClientMsg) -> Result<(), std::io::Error> {
    let data = msg.serialize();
    let len = (data.len() as u32).to_be_bytes();
    stream.write_all(&len).await?;
    stream.write_all(&data).await?;
    stream.flush().await?;
    Ok(())
}

async fn recv_server_msg(stream: &mut TcpStream) -> Result<ServerMsg, Box<dyn std::error::Error + Send + Sync>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    Ok(ServerMsg::deserialize(&buf)?)
}

fn empty_input() -> PlayerInputData {
    PlayerInputData {
        movement: [0.0; 3],
        look_yaw: 0.0,
        look_pitch: 0.0,
        jump: false,
        fly_toggle: false,
        speed_tier: 0,
        action: 0,
        block_type: 0,
        tick: 0,
    }
}
