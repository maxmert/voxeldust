//! Client networking: connect to gateway, handle redirect, receive WorldState.
//! Supports mid-gameplay shard transitions via ShardRedirect on TCP.

use std::net::SocketAddr;
use std::sync::Arc;

use glam::{DQuat, DVec3};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::mpsc;
use tracing::{info, warn};

use voxeldust_core::client_message::{
    ClientMsg, PlayerInputData, ServerMsg, WorldStateData,
};
use voxeldust_core::handoff::{ShardPreConnect, ShardRedirect};

/// Events from the network thread to the render thread.
pub enum NetEvent {
    Connected {
        shard_type: u8,
        reference_position: DVec3,
        reference_rotation: DQuat,
        game_time: f64,
        system_seed: u64,
        galaxy_seed: u64,
    },
    WorldState(WorldStateData),
    /// A secondary shard has been pre-connected for rendering.
    SecondaryConnected {
        shard_type: u8,
        seed: u64,
        reference_position: DVec3,
        reference_rotation: DQuat,
    },
    /// WorldState from a secondary shard (for composite rendering).
    SecondaryWorldState(WorldStateData),
    /// Galaxy world state from secondary UDP (warp travel position for star parallax).
    GalaxyWorldState(voxeldust_core::client_message::GalaxyWorldStateData),
    /// Primary shard is changing (ShardRedirect received).
    Transitioning,
    Disconnected(String),
}

/// Run the network loop on a tokio runtime. Handles shard transitions.
pub async fn run_network(
    gateway_addr: SocketAddr,
    player_name: String,
    event_tx: mpsc::UnboundedSender<NetEvent>,
    input_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<PlayerInputData>>>,
    direct: Option<String>,
) {
    // Resolve initial shard addresses (via gateway or direct).
    let (mut shard_tcp_addr, mut shard_udp_addr) = if let Some(ref direct_addr) = direct {
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
        info!(tcp = %redirect.target_tcp_addr, udp = %redirect.target_udp_addr,
            "received shard redirect");
        match (redirect.target_tcp_addr.parse(), redirect.target_udp_addr.parse()) {
            (Ok(tcp), Ok(udp)) => (tcp, udp),
            _ => {
                let _ = event_tx.send(NetEvent::Disconnected("bad redirect addrs".into()));
                return;
            }
        }
    };

    // Main shard connection loop — reconnects on ShardRedirect.
    loop {
        info!(%shard_tcp_addr, "connecting to shard");
        let (tcp_stream, jr) = match connect_to_shard_full(shard_tcp_addr, &player_name).await {
            Ok(r) => r,
            Err(e) => {
                let _ = event_tx.send(NetEvent::Disconnected(format!("shard connect: {e}")));
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
            galaxy_seed: jr.galaxy_seed,
        });

        // Set up UDP for this shard.
        let udp = match UdpSocket::bind("0.0.0.0:0").await {
            Ok(s) => s,
            Err(e) => {
                let _ = event_tx.send(NetEvent::Disconnected(format!("udp: {e}")));
                return;
            }
        };
        let hello = build_input(&empty_input());
        let _ = udp.send_to(&hello, shard_udp_addr).await;
        info!(%shard_udp_addr, "UDP hole-punch sent");

        let udp = Arc::new(udp);

        // Channel to signal shutdown to child tasks.
        let (cancel_tx, _) = tokio::sync::broadcast::channel::<()>(1);

        // Input sender task (20Hz).
        let udp_send = udp.clone();
        let input_rx_clone = input_rx.clone();
        let mut cancel_input = cancel_tx.subscribe();
        let send_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(50));
            let mut last_sent = empty_input();
            let mut ticks_since_send: u32 = 0;
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let input = {
                            let mut rx = input_rx_clone.lock().await;
                            let mut latest = empty_input();
                            while let Ok(i) = rx.try_recv() { latest = i; }
                            latest
                        };
                        // Suppress unchanged input; send keepalive every 1s (20 ticks).
                        if input != last_sent || ticks_since_send >= 20 {
                            let pkt = build_input(&input);
                            let _ = udp_send.send_to(&pkt, shard_udp_addr).await;
                            last_sent = input;
                            ticks_since_send = 0;
                        } else {
                            ticks_since_send += 1;
                        }
                    }
                    _ = cancel_input.recv() => { return; }
                }
            }
        });

        // UDP WorldState receiver task.
        let event_tx_udp = event_tx.clone();
        let udp_recv = udp.clone();
        let mut cancel_udp = cancel_tx.subscribe();
        let mut recv_handle = tokio::spawn(async move {
            let mut buf = vec![0u8; 65536];
            loop {
                tokio::select! {
                    result = udp_recv.recv_from(&mut buf) => {
                        match result {
                            Ok((len, _)) => {
                                if len < 4 { continue; }
                                let msg_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                                if len < 4 + msg_len { continue; }
                                let decoded = match voxeldust_core::wire_codec::decode(&buf[4..4 + msg_len]) {
                                    Ok(d) => d,
                                    Err(_) => continue,
                                };
                                if let Ok(ServerMsg::WorldState(ws)) = ServerMsg::deserialize(&decoded) {
                                    let _ = event_tx_udp.send(NetEvent::WorldState(ws));
                                }
                            }
                            Err(e) => { warn!(%e, "UDP recv error"); return; }
                        }
                    }
                    _ = cancel_udp.recv() => { return; }
                }
            }
        });

        // TCP listener — monitors for ShardRedirect or ShardPreConnect.
        let event_tx_tcp = event_tx.clone();
        let player_name_tcp = player_name.clone();
        let cancel_tx_for_tcp = cancel_tx.clone();
        let (redirect_tx, mut redirect_rx) = mpsc::channel::<ShardRedirect>(1);
        let mut cancel_tcp = cancel_tx.subscribe();
        let tcp_handle = tokio::spawn(async move {
            let mut stream = tcp_stream;
            let mut keepalive_interval = tokio::time::interval(std::time::Duration::from_secs(60));
            keepalive_interval.tick().await; // skip first immediate tick
            loop {
                tokio::select! {
                    _ = keepalive_interval.tick() => {
                        // Send a tiny TCP message to prevent proxy idle timeout.
                        // A zero-length frame is ignored by the server's message parser.
                        let _ = stream.write_all(&0u32.to_be_bytes()).await;
                        let _ = stream.flush().await;
                    }
                    result = recv_server_msg(&mut stream) => {
                        match result {
                            Ok(ServerMsg::ShardRedirect(r)) => {
                                info!(target_tcp = %r.target_tcp_addr, "received ShardRedirect");
                                let _ = redirect_tx.send(r).await;
                                return;
                            }
                            Ok(ServerMsg::ShardPreConnect(pc)) => {
                                info!(shard_type = pc.shard_type, seed = pc.seed,
                                    tcp = %pc.tcp_addr, udp = %pc.udp_addr,
                                    "received ShardPreConnect — opening secondary connection");

                                // Spawn secondary connection to the planet shard.
                                // UDP-only: no TCP connect (that would trigger a player spawn
                                // on the planet shard). All metadata comes from ShardPreConnect.
                                let sec_event_tx = event_tx_tcp.clone();
                                let mut sec_cancel = cancel_tx_for_tcp.subscribe();
                                let sec_pc = pc;
                                tokio::spawn(async move {
                                    let sec_udp: SocketAddr = match sec_pc.udp_addr.parse() {
                                        Ok(a) => a,
                                        Err(e) => { warn!(%e, "bad ShardPreConnect udp addr"); return; }
                                    };

                                    let _ = sec_event_tx.send(NetEvent::SecondaryConnected {
                                        shard_type: sec_pc.shard_type,
                                        seed: sec_pc.seed,
                                        reference_position: sec_pc.reference_position,
                                        reference_rotation: sec_pc.reference_rotation,
                                    });

                                    // Open secondary UDP.
                                    let udp = match UdpSocket::bind("0.0.0.0:0").await {
                                        Ok(s) => s,
                                        Err(e) => { warn!(%e, "secondary UDP bind failed"); return; }
                                    };
                                    let hello = build_input(&empty_input());
                                    let _ = udp.send_to(&hello, sec_udp).await;
                                    info!(%sec_udp, "secondary UDP hole-punch sent");

                                    // Receive loop: forward WorldState as SecondaryWorldState.
                                    let mut buf = vec![0u8; 65536];
                                    loop {
                                        tokio::select! {
                                            result = udp.recv_from(&mut buf) => {
                                                match result {
                                                    Ok((len, _)) => {
                                                        if len < 4 { continue; }
                                                        let msg_len = u32::from_be_bytes(
                                                            [buf[0], buf[1], buf[2], buf[3]]) as usize;
                                                        if len < 4 + msg_len { continue; }
                                                        let decoded = match voxeldust_core::wire_codec::decode(
                                                            &buf[4..4 + msg_len]) {
                                                            Ok(d) => d,
                                                            Err(_) => continue,
                                                        };
                                                        match ServerMsg::deserialize(&decoded) {
                                                            Ok(ServerMsg::WorldState(ws)) => {
                                                                let _ = sec_event_tx.send(
                                                                    NetEvent::SecondaryWorldState(ws));
                                                            }
                                                            Ok(ServerMsg::GalaxyWorldState(gws)) => {
                                                                let _ = sec_event_tx.send(
                                                                    NetEvent::GalaxyWorldState(gws));
                                                            }
                                                            _ => {}
                                                        }
                                                    }
                                                    Err(e) => {
                                                        warn!(%e, "secondary UDP recv error");
                                                        return;
                                                    }
                                                }
                                            }
                                            _ = sec_cancel.recv() => { return; }
                                        }
                                    }
                                });
                            }
                            Ok(_) => { /* ignore other TCP messages */ }
                            Err(e) => {
                                warn!(%e, "TCP read error");
                                return;
                            }
                        }
                    }
                    _ = cancel_tcp.recv() => { return; }
                }
            }
        });

        // Wait for either a redirect or a disconnect.
        tokio::select! {
            redirect = redirect_rx.recv() => {
                if let Some(r) = redirect {
                    let _ = event_tx.send(NetEvent::Transitioning);
                    // Cancel all tasks for current shard.
                    let _ = cancel_tx.send(());
                    send_handle.abort();
                    recv_handle.abort();
                    tcp_handle.abort();

                    // Parse new shard addresses and loop back to reconnect.
                    match (r.target_tcp_addr.parse(), r.target_udp_addr.parse()) {
                        (Ok(tcp), Ok(udp)) => {
                            shard_tcp_addr = tcp;
                            shard_udp_addr = udp;
                            info!(%shard_tcp_addr, "transitioning to new shard");
                            continue;
                        }
                        _ => {
                            let _ = event_tx.send(NetEvent::Disconnected("bad redirect addrs".into()));
                            return;
                        }
                    }
                }
                // redirect_rx closed without value — fall through to disconnect.
                let _ = cancel_tx.send(());
                send_handle.abort();
                tcp_handle.abort();
                let _ = event_tx.send(NetEvent::Disconnected("redirect channel closed".into()));
                return;
            }
            _ = &mut recv_handle => {
                // UDP died — disconnect.
                let _ = cancel_tx.send(());
                send_handle.abort();
                tcp_handle.abort();
                let _ = event_tx.send(NetEvent::Disconnected("connection lost".into()));
                return;
            }
        }
    }
}

fn build_input(input: &PlayerInputData) -> Vec<u8> {
    let msg = ClientMsg::PlayerInput(input.clone());
    let data = msg.serialize();
    let mut pkt = Vec::new();
    voxeldust_core::wire_codec::encode(&data, &mut pkt);
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

/// Connect to a shard, returning both the TCP stream (kept alive for redirect
/// monitoring) and the JoinResponse.
async fn connect_to_shard_full(
    addr: SocketAddr,
    player_name: &str,
) -> Result<(TcpStream, voxeldust_core::client_message::JoinResponseData), Box<dyn std::error::Error + Send + Sync>> {
    let mut stream = TcpStream::connect(addr).await?;

    // Set TCP nodelay for low latency.
    let _ = stream.set_nodelay(true);

    send_msg(&mut stream, &ClientMsg::Connect { player_name: player_name.to_string() }).await?;
    let response = recv_server_msg(&mut stream).await?;
    match response {
        ServerMsg::JoinResponse(jr) => Ok((stream, jr)),
        other => Err(format!("expected JoinResponse, got {:?}", std::mem::discriminant(&other)).into()),
    }
}

async fn send_msg(stream: &mut TcpStream, msg: &ClientMsg) -> Result<(), std::io::Error> {
    let data = msg.serialize();
    let mut buf = Vec::new();
    voxeldust_core::wire_codec::encode(&data, &mut buf);
    stream.write_all(&buf).await?;
    stream.flush().await?;
    Ok(())
}

async fn recv_server_msg(stream: &mut TcpStream) -> Result<ServerMsg, Box<dyn std::error::Error + Send + Sync>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    let decoded = voxeldust_core::wire_codec::decode(&buf)
        .map_err(|e| format!("wire decode: {e}"))?;
    Ok(ServerMsg::deserialize(&decoded)?)
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
