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
    BlockEditData, ChunkDeltaData, ChunkSnapshotData, ClientMsg, PlayerInputData, ServerMsg,
    WorldStateData,
};
use voxeldust_core::handoff::{ShardPreConnect, ShardRedirect};

/// Events from the network thread to the render thread.
pub enum NetEvent {
    Connected {
        shard_type: u8,
        seed: u64,
        reference_position: DVec3,
        reference_rotation: DQuat,
        game_time: f64,
        system_seed: u64,
        galaxy_seed: u64,
        /// Our player's id on this shard. Client must filter
        /// `WorldState.players[]` by this to avoid picking up someone
        /// else's position (system-shard EVA broadcasts include all
        /// players, not just the recipient).
        player_id: u64,
    },
    WorldState(WorldStateData),
    /// A secondary shard has been pre-connected for rendering.
    SecondaryConnected {
        shard_type: u8,
        seed: u64,
        reference_position: DVec3,
        reference_rotation: DQuat,
    },
    /// A secondary shard's connection has ended (for any reason: server-sent
    /// `ShardDisconnectNotify`, replacement by a new pre-connect for the
    /// same key, primary-transition cancel, session end, or UDP error).
    /// The main thread must release the chunk source associated with `seed`
    /// so its GPU buffers are freed — otherwise chunks accumulate across
    /// every ship transition.
    SecondaryDisconnected {
        seed: u64,
    },
    /// WorldState from a secondary shard (for composite rendering).
    /// `shard_type` identifies which secondary the WS came from —
    /// necessary because multiple secondaries (SYSTEM, PLANET) can be
    /// connected simultaneously and their WorldStates interleave. The
    /// client uses this to route the WS to the correct per-type slot
    /// so Transitioning's `ws.secondary.take()` picks the secondary
    /// matching the redirect target, not whichever sent last.
    SecondaryWorldState {
        shard_type: u8,
        ws: WorldStateData,
    },
    /// Galaxy world state from secondary UDP (warp travel position for star parallax).
    GalaxyWorldState(voxeldust_core::client_message::GalaxyWorldStateData),
    /// Block signal config state from server (config UI).
    BlockConfigState(voxeldust_core::signal::config::BlockSignalConfig),
    /// Seat bindings from server (when player enters a seat).
    SeatBindingsNotify(voxeldust_core::client_message::SeatBindingsNotifyData),
    /// Sub-grid block assignments from server (mechanical mount membership).
    SubGridAssignmentUpdate(voxeldust_core::client_message::SubGridAssignmentData),
    /// Full chunk snapshot received (initial sync or resync).
    ChunkSnapshot(ChunkSnapshotData),
    /// Incremental block changes to a chunk.
    ChunkDelta(ChunkDeltaData),
    /// Chunk snapshot from a secondary (observer) connection.
    SecondaryChunkSnapshot {
        seed: u64,
        data: ChunkSnapshotData,
    },
    /// Chunk delta from a secondary (observer) connection.
    SecondaryChunkDelta {
        seed: u64,
        data: ChunkDeltaData,
    },
    /// Sub-grid assignments from a secondary (observer) connection.
    SecondarySubGridAssignment {
        seed: u64,
        data: voxeldust_core::client_message::SubGridAssignmentData,
    },
    /// Primary shard is changing (ShardRedirect received).
    /// `target_shard_type` disambiguates which open secondary to promote when
    /// more than one is live (e.g., Ship + Galaxy secondaries). `255` = legacy
    /// redirect with no type hint (falls back to last-connected secondary).
    ///
    /// `spawn_pose` (when present) carries the authoritative
    /// post-transition camera pose, computed server-side. The client uses
    /// this directly for its first rendered frame after the primary switch,
    /// so there is zero client-side prediction of position/rotation across
    /// the handoff.
    Transitioning {
        target_shard_type: u8,
        spawn_pose: Option<voxeldust_core::handoff::SpawnPose>,
    },
    /// Full star catalogue for the galaxy — sent by the galaxy
    /// shard once per connect. Authoritative; no per-tick update.
    StarCatalog(voxeldust_core::client_message::StarCatalogData),
    Disconnected(String),
}

/// Run the network loop on a tokio runtime. Handles shard transitions.
pub async fn run_network(
    gateway_addr: SocketAddr,
    player_name: String,
    event_tx: mpsc::UnboundedSender<NetEvent>,
    input_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<PlayerInputData>>>,
    block_edit_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<BlockEditData>>>,
    tcp_out_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<Vec<u8>>>>,
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

    // Session-level cancel: fires only on genuine session end (Disconnected),
    // NOT on `ShardRedirect`. Scene-context secondaries (System/Galaxy) bind
    // their lifetime to this so they survive every primary reconnect — without
    // it, stars/bodies/AOI blip off on every shard transition.
    let (session_cancel_tx, _) = tokio::sync::broadcast::channel::<()>(1);

    // Session-level secondary registry. Keyed by (shard_type, shard_id).
    // Scene-context entries (type 1/3) persist across primary reconnects;
    // per-primary entries (type 0/2) are drained when a `ShardRedirect` is
    // processed inside the TCP listener. Shared across the outer reconnect
    // loop via Arc<Mutex<…>>.
    let active_secondaries: Arc<std::sync::Mutex<
        std::collections::HashMap<(u8, u64), tokio::sync::broadcast::Sender<()>>
    >> = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

    // Main shard connection loop — reconnects on ShardRedirect.
    loop {
        info!(%shard_tcp_addr, %shard_udp_addr, "connecting to shard");
        let (tcp_stream, jr) = match connect_to_shard_full(shard_tcp_addr, &player_name).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(%shard_tcp_addr, %e, "connect_to_shard_full FAILED — session will disconnect");
                let _ = event_tx.send(NetEvent::Disconnected(format!("shard connect: {e}")));
                return;
            }
        };

        info!(shard_type = jr.shard_type, "joined shard");
        let _ = event_tx.send(NetEvent::Connected {
            shard_type: jr.shard_type,
            seed: jr.seed,
            reference_position: jr.reference_position,
            reference_rotation: jr.reference_rotation,
            game_time: jr.game_time,
            system_seed: jr.system_seed,
            galaxy_seed: jr.galaxy_seed,
            player_id: jr.player_id,
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

        // TCP outbound channel: block/sub-block edits and config updates
        // are sent reliably via TCP (not UDP) to guarantee delivery.
        // The sender lives in the ECS world (NetworkChannels.tcp_out_tx);
        // the receiver is consumed here for writing to the TCP stream.
        let tcp_out_rx_clone = tcp_out_rx.clone();

        // Input sender task (20Hz via UDP) + block edit drain → TCP.
        let udp_send = udp.clone();
        let input_rx_clone = input_rx.clone();
        let _block_edit_rx_clone = block_edit_rx.clone(); // kept for future UDP fallback
        let _tcp_out_rx_for_edit = tcp_out_rx.clone();
        let mut cancel_input = cancel_tx.subscribe();
        let send_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(50));
            let mut last_sent = empty_input();
            let mut ticks_since_send: u32 = 0;
            let mut ticks_total: u32 = 0;
            info!(%shard_udp_addr, "input sender task started");
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        ticks_total += 1;
                        let input = {
                            let mut rx = input_rx_clone.lock().await;
                            let mut latest = empty_input();
                            while let Ok(i) = rx.try_recv() { latest = i; }
                            latest
                        };
                        // Suppress unchanged input; send keepalive every 1s (20 ticks).
                        let movement_snapshot = input.movement;
                        let changed = input != last_sent;
                        if changed || ticks_since_send >= 20 {
                            let pkt = build_input(&input);
                            let send_result = udp_send.send_to(&pkt, shard_udp_addr).await;
                            if changed {
                                if let Err(e) = send_result {
                                    tracing::warn!(%shard_udp_addr, %e, "UDP input send failed");
                                }
                            }
                            last_sent = input;
                            ticks_since_send = 0;
                        } else {
                            ticks_since_send += 1;
                        }
                        // Heartbeat every ~5s so we can verify the task is alive.
                        if ticks_total % 100 == 0 {
                            tracing::info!(
                                %shard_udp_addr,
                                ticks_total,
                                movement = ?movement_snapshot,
                                "input sender heartbeat"
                            );
                        }
                    }
                    _ = cancel_input.recv() => {
                        info!("input sender task cancelled");
                        return;
                    }
                }
            }
        });

        // UDP WorldState receiver task.
        let event_tx_udp = event_tx.clone();
        let udp_recv = udp.clone();
        let primary_udp_addr = shard_udp_addr;
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
                            Err(e) => {
                                // On Linux, an ICMP port-unreachable received in response
                                // to a prior UDP send causes the NEXT recv_from to fail
                                // with ECONNREFUSED on an UNCONNECTED socket — this is
                                // how we'd silently lose the primary UDP path if the
                                // server's peer tracking got confused.
                                let kind = e.kind();
                                warn!(%e, ?kind, %primary_udp_addr, "primary UDP recv error — giving up on this socket");
                                return;
                            }
                        }
                    }
                    _ = cancel_udp.recv() => {
                        info!(%primary_udp_addr, "primary UDP recv task cancelled");
                        return;
                    }
                }
            }
        });

        // TCP listener — monitors for ShardRedirect or ShardPreConnect.
        let event_tx_tcp = event_tx.clone();
        let player_name_tcp = player_name.clone();
        // `primary_cancel_tx` kills only per-primary subordinate tasks (send/recv/tcp
        // and ship/planet secondaries). Scene-context secondaries bind to
        // `session_cancel_tx_for_tcp` which does NOT fire on `ShardRedirect`.
        let primary_cancel_tx = cancel_tx.clone();
        let session_cancel_tx_for_tcp = session_cancel_tx.clone();
        let secondaries_for_tcp = active_secondaries.clone();
        let (redirect_tx, mut redirect_rx) = mpsc::channel::<ShardRedirect>(1);
        let mut cancel_tcp = cancel_tx.subscribe();
        let tcp_handle = tokio::spawn(async move {
            let (tcp_read, mut tcp_write) = tokio::io::split(tcp_stream);
            let mut tcp_read = tokio::io::BufReader::new(tcp_read);
            let mut keepalive_interval = tokio::time::interval(std::time::Duration::from_secs(60));
            keepalive_interval.tick().await; // skip first immediate tick

            // Active secondary connections keyed by (shard_type, shard_id) are
            // stored in `secondaries_for_tcp` (Arc<Mutex<HashMap<…>>>) shared
            // with the outer session. Multiple secondaries of the SAME type
            // coexist (e.g., 3 nearby ships); for planet shards shard_id is 0.
            //
            // Shard type discriminants (matching core/src/client_message.rs::shard_type):
            //   PLANET=0, SYSTEM=1, SHIP=2, GALAXY=3.
            //
            // System (1) and Galaxy (3) secondaries are "scene context" — they
            // carry celestial bodies, long-range AOI entities, and warp parallax.
            // They are always-on for the session duration, exempt from
            // MAX_SECONDARIES, and bind their cancel to `session_cancel_tx`
            // so they survive primary reconnects.
            const MAX_SECONDARIES: usize = 4;
            let is_scene_context = |shard_type: u8| shard_type == 1 || shard_type == 3;

            loop {
                tokio::select! {
                    // Outbound: send block/sub-block edits via TCP (reliable).
                    pkt = async {
                        let mut rx = tcp_out_rx_clone.lock().await;
                        rx.recv().await
                    } => {
                        if let Some(pkt) = pkt {
                            let _ = tcp_write.write_all(&pkt).await;
                            let _ = tcp_write.flush().await;
                        }
                    }
                    _ = keepalive_interval.tick() => {
                        let _ = tcp_write.write_all(&0u32.to_be_bytes()).await;
                        let _ = tcp_write.flush().await;
                    }
                    result = recv_server_msg(&mut tcp_read) => {
                        match result {
                            Ok(ServerMsg::ShardRedirect(r)) => {
                                info!(target_tcp = %r.target_tcp_addr, "received ShardRedirect");
                                // Scene-context secondaries (System/Galaxy) survive the
                                // primary transition: they're bound to `session_cancel_tx`
                                // (NOT `primary_cancel_tx`), and we leave their entries
                                // in the shared map. Only ship/planet entries are torn
                                // down here.
                                let doomed: Vec<(u8, u64)> = {
                                    let map = secondaries_for_tcp.lock().unwrap();
                                    map.keys()
                                        .filter(|(st, _)| !is_scene_context(*st))
                                        .copied()
                                        .collect()
                                };
                                for key in doomed {
                                    let cancel = {
                                        let mut map = secondaries_for_tcp.lock().unwrap();
                                        map.remove(&key)
                                    };
                                    if let Some(cancel) = cancel {
                                        let _ = cancel.send(());
                                        info!(
                                            shard_type = key.0,
                                            shard_id = key.1,
                                            "cancelled non-scene secondary on ShardRedirect"
                                        );
                                    }
                                }
                                let _ = redirect_tx.send(r).await;
                                return;
                            }
                            Ok(ServerMsg::ShardPreConnect(pc)) => {
                                info!(shard_type = pc.shard_type, seed = pc.seed,
                                    tcp = %pc.tcp_addr, udp = %pc.udp_addr,
                                    "received ShardPreConnect — opening secondary connection");

                                // Key: (shard_type, shard_id). For planets, shard_id = 0.
                                let sec_key = (pc.shard_type, pc.shard_id);

                                // Cancel existing secondary with the SAME key (shared map).
                                let prev_cancel = {
                                    let mut map = secondaries_for_tcp.lock().unwrap();
                                    map.remove(&sec_key)
                                };
                                if let Some(old_cancel) = prev_cancel {
                                    let _ = old_cancel.send(());
                                    info!(shard_type = pc.shard_type, shard_id = pc.shard_id,
                                        "cancelled old secondary for replacement");
                                }

                                // Enforce maximum simultaneous secondaries, counting
                                // only Ship/Planet secondaries (System+Galaxy are exempt).
                                if !is_scene_context(pc.shard_type) {
                                    loop {
                                        let evict_key = {
                                            let map = secondaries_for_tcp.lock().unwrap();
                                            let count = map
                                                .keys()
                                                .filter(|(st, _)| !is_scene_context(*st))
                                                .count();
                                            if count < MAX_SECONDARIES {
                                                break;
                                            }
                                            map.keys()
                                                .find(|(st, _)| !is_scene_context(*st))
                                                .copied()
                                        };
                                        let Some(key) = evict_key else { break };
                                        let cancel = {
                                            let mut map = secondaries_for_tcp.lock().unwrap();
                                            map.remove(&key)
                                        };
                                        if let Some(cancel) = cancel {
                                            let _ = cancel.send(());
                                            info!(shard_type = key.0, shard_id = key.1,
                                                "evicted secondary — max reached");
                                        }
                                    }
                                }

                                // Dedicated cancel token for this secondary.
                                let (sec_cancel_tx, _) = tokio::sync::broadcast::channel::<()>(1);
                                {
                                    let mut map = secondaries_for_tcp.lock().unwrap();
                                    map.insert(sec_key, sec_cancel_tx.clone());
                                }

                                // Open observer TCP connection for chunk data.
                                if let Ok(tcp_addr) = pc.tcp_addr.parse::<SocketAddr>() {
                                    let tcp_event_tx = event_tx_tcp.clone();
                                    let tcp_cancel = sec_cancel_tx.subscribe();
                                    let tcp_seed = pc.seed;
                                    tokio::spawn(async move {
                                        connect_observer_tcp(tcp_addr, tcp_seed, tcp_event_tx, tcp_cancel).await;
                                    });
                                }

                                let sec_event_tx = event_tx_tcp.clone();
                                let sec_shard_type = pc.shard_type;
                                let mut sec_cancel_own = sec_cancel_tx.subscribe();
                                // Scene-context secondaries (System/Galaxy) bind their
                                // second cancel source to `session_cancel_tx` so they
                                // survive primary transitions. Per-primary secondaries
                                // (Ship/Planet) bind to `primary_cancel_tx` and die
                                // with the current primary — matching the old behavior.
                                let mut sec_cancel_parent = if is_scene_context(pc.shard_type) {
                                    session_cancel_tx_for_tcp.subscribe()
                                } else {
                                    primary_cancel_tx.subscribe()
                                };
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
                                    // If bind fails after `SecondaryConnected` was sent,
                                    // main has already allocated a chunk source for this
                                    // seed — emit `SecondaryDisconnected` to clean it up.
                                    let udp = match UdpSocket::bind("0.0.0.0:0").await {
                                        Ok(s) => s,
                                        Err(e) => {
                                            warn!(%e, "secondary UDP bind failed");
                                            let _ = sec_event_tx.send(
                                                NetEvent::SecondaryDisconnected { seed: sec_pc.seed });
                                            return;
                                        }
                                    };
                                    let hello = build_input(&empty_input());
                                    let _ = udp.send_to(&hello, sec_udp).await;
                                    info!(%sec_udp, "secondary UDP hole-punch sent");

                                    // Receive loop.
                                    //
                                    // Every exit from this loop emits `SecondaryDisconnected`
                                    // so the main thread frees the chunk source's GPU
                                    // buffers. The UDP task is always spawned for every
                                    // secondary (the observer TCP task is optional), so
                                    // this is the one reliable disconnect signal per seed.
                                    let sec_seed = sec_pc.seed;
                                    let mut buf = vec![0u8; 65536];
                                    let exit_reason: &str = loop {
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
                                                                    NetEvent::SecondaryWorldState {
                                                                        shard_type: sec_shard_type,
                                                                        ws,
                                                                    });
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
                                                        break "udp error";
                                                    }
                                                }
                                            }
                                            _ = sec_cancel_own.recv() => {
                                                info!("secondary connection cancelled (replaced)");
                                                break "cancel_own";
                                            }
                                            _ = sec_cancel_parent.recv() => { break "cancel_parent"; }
                                        }
                                    };
                                    info!(seed = sec_seed, reason = exit_reason,
                                        "secondary UDP task exiting — emitting SecondaryDisconnected");
                                    let _ = sec_event_tx.send(NetEvent::SecondaryDisconnected {
                                        seed: sec_seed,
                                    });
                                });
                            }
                            Ok(ServerMsg::BlockConfigState(config)) => {
                                let _ = event_tx_tcp.send(NetEvent::BlockConfigState(config));
                            }
                            Ok(ServerMsg::SeatBindingsNotify(data)) => {
                                let _ = event_tx_tcp.send(NetEvent::SeatBindingsNotify(data));
                            }
                            Ok(ServerMsg::ChunkSnapshot(cs)) => {
                                let _ = event_tx_tcp.send(NetEvent::ChunkSnapshot(cs));
                            }
                            Ok(ServerMsg::ChunkDelta(cd)) => {
                                let _ = event_tx_tcp.send(NetEvent::ChunkDelta(cd));
                            }
                            Ok(ServerMsg::SubGridAssignmentUpdate(data)) => {
                                let _ = event_tx_tcp.send(NetEvent::SubGridAssignmentUpdate(data));
                            }
                            Ok(ServerMsg::StarCatalog(data)) => {
                                let _ = event_tx_tcp.send(NetEvent::StarCatalog(data));
                            }
                            Ok(ServerMsg::ShardDisconnectNotify(dn)) => {
                                let key = (dn.shard_type, dn.seed);
                                let cancel = {
                                    let mut map = secondaries_for_tcp.lock().unwrap();
                                    map.remove(&key)
                                };
                                if let Some(cancel) = cancel {
                                    let _ = cancel.send(());
                                    info!(shard_type = dn.shard_type, seed = dn.seed,
                                        "secondary disconnected via ShardDisconnectNotify");
                                }
                                // TODO: notify main thread to remove chunk source.
                            }
                            Ok(_) => { /* ignore other TCP messages */ }
                            Err(e) => {
                                warn!(%e, "primary TCP read error — exiting TCP task");
                                return;
                            }
                        }
                    }
                    _ = cancel_tcp.recv() => {
                        info!("primary TCP task cancelled");
                        return;
                    }
                }
            }
        });

        // Wait for either a redirect or a disconnect.
        tokio::select! {
            redirect = redirect_rx.recv() => {
                if let Some(r) = redirect {
                    info!(
                        target_tcp = %r.target_tcp_addr,
                        target_udp = %r.target_udp_addr,
                        target_shard_type = r.target_shard_type,
                        "processing redirect: tearing down primary tasks"
                    );
                    let _ = event_tx.send(NetEvent::Transitioning {
                        target_shard_type: r.target_shard_type,
                        spawn_pose: r.spawn_pose.clone(),
                    });
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
                            info!(%shard_tcp_addr, %shard_udp_addr, "transitioning to new shard");
                            continue;
                        }
                        _ => {
                            // Genuine disconnect — session_cancel_tx fires so
                            // scene-context secondaries are cleaned up.
                            let _ = session_cancel_tx.send(());
                            let _ = event_tx.send(NetEvent::Disconnected("bad redirect addrs".into()));
                            return;
                        }
                    }
                }
                // redirect_rx closed without value — fall through to disconnect.
                // This happens when the TCP read loop exited (TCP error or EOF).
                warn!(%shard_tcp_addr, "TCP read task exited without sending redirect — treating as disconnect");
                let _ = cancel_tx.send(());
                let _ = session_cancel_tx.send(());
                send_handle.abort();
                tcp_handle.abort();
                let _ = event_tx.send(NetEvent::Disconnected("redirect channel closed".into()));
                return;
            }
            _ = &mut recv_handle => {
                // UDP died — genuine disconnect.
                warn!(%shard_udp_addr, "primary UDP recv task exited — treating as disconnect");
                let _ = cancel_tx.send(());
                let _ = session_cancel_tx.send(());
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

fn build_block_edit(edit: &BlockEditData) -> Vec<u8> {
    let msg = ClientMsg::BlockEditRequest(edit.clone());
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

/// Open a TCP connection to a secondary shard as an observer.
/// Sends ObserverConnect instead of Connect, receives chunk data.
async fn connect_observer_tcp(
    addr: SocketAddr,
    seed: u64,
    event_tx: mpsc::UnboundedSender<NetEvent>,
    mut cancel_rx: tokio::sync::broadcast::Receiver<()>,
) {
    let mut stream = match TcpStream::connect(addr).await {
        Ok(s) => s,
        Err(e) => {
            warn!(%e, %addr, "observer TCP connect failed");
            return;
        }
    };
    let _ = stream.set_nodelay(true);

    // Send ObserverConnect instead of Connect.
    if let Err(e) = send_msg(&mut stream, &ClientMsg::ObserverConnect {
        observer_name: format!("observer_{}", seed),
    }).await {
        warn!(%e, "failed to send ObserverConnect");
        return;
    }

    info!(%addr, seed, "observer TCP connected — receiving chunks");

    // Read loop: receive ChunkSnapshots, ChunkDeltas, SubGridAssignmentUpdates.
    loop {
        tokio::select! {
            result = recv_server_msg(&mut stream) => {
                match result {
                    Ok(ServerMsg::ChunkSnapshot(cs)) => {
                        let _ = event_tx.send(NetEvent::SecondaryChunkSnapshot { seed, data: cs });
                    }
                    Ok(ServerMsg::ChunkDelta(cd)) => {
                        let _ = event_tx.send(NetEvent::SecondaryChunkDelta { seed, data: cd });
                    }
                    Ok(ServerMsg::SubGridAssignmentUpdate(sg)) => {
                        let _ = event_tx.send(NetEvent::SecondarySubGridAssignment { seed, data: sg });
                    }
                    Ok(ServerMsg::StarCatalog(data)) => {
                        // Galaxy shard is opened as a secondary for
                        // scene-context rendering — its StarCatalog
                        // arrives on the observer TCP, not the
                        // primary. Forward it as the shared
                        // `NetEvent::StarCatalog` so client-side
                        // systems see one path regardless of which
                        // connection delivered the bytes.
                        let _ = event_tx.send(NetEvent::StarCatalog(data));
                    }
                    Ok(_) => {} // ignore other messages
                    Err(e) => {
                        info!(%e, seed, "observer TCP ended");
                        return;
                    }
                }
            }
            _ = cancel_rx.recv() => {
                info!(seed, "observer TCP cancelled");
                return;
            }
        }
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

async fn recv_server_msg(stream: &mut (impl tokio::io::AsyncRead + Unpin)) -> Result<ServerMsg, Box<dyn std::error::Error + Send + Sync>> {
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
        orbit_stabilizer_toggle: false,
        speed_tier: 0,
        action: 0,
        block_type: 0,
        tick: 0,
        thrust_limiter: 0.75,
        roll: 0.0,
        cruise: false,
        atmo_comp: false,
        seat_values: Vec::new(),
        actions_bits: 0,
    }
}
