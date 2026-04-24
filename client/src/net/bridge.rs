//! Bevy plugin bridging `network::run_network` (Tokio, mpsc) → Bevy ECS.
//!
//! ```text
//!                 ┌──────────────────┐
//!                 │ Tokio worker     │   (dedicated std::thread, owns a
//!                 │  run_network()   │    multi-threaded Tokio runtime)
//!                 └─┬───────────▲────┘
//!                   │NetEvent   │ClientMsg
//!  mpsc::UnboundedSender        │
//!                   ▼           │
//! ┌────────────────────────────┐│
//! │ NetworkBridge resource     │├── held by main thread
//! │  rx: Mutex<Receiver<...>>  ││
//! │  input_tx: Sender<...>     ││
//! └────────────┬───────────────┘│
//!              │drain_network_events (runs in NetworkBridgeSet)
//!              ▼ (MessageWriter<GameEvent>)
//!    Bevy systems act on events
//! ```
//!
//! `NetworkBridgeSet` is explicitly placed at the start of each frame so
//! that downstream systems (ShardRegistrySet, WorldStateIngestSet, …)
//! observe every NetEvent in the same frame it arrived. This eliminates
//! the "first-frame dropped chunks" race the voxydust-next audit
//! documented.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use tokio::sync::{mpsc, Mutex as AsyncMutex};

use voxeldust_core::client_message::{BlockEditData, PlayerInputData};

use crate::net::network::{run_network, NetEvent};

/// System set for the bridge drain. Downstream plugins order themselves
/// `.after(NetworkBridgeSet)` so they see a consistent event stream and
/// fresh summary resources every frame.
#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct NetworkBridgeSet;

/// Plugin entry point.
pub struct NetworkPlugin {
    pub gateway: SocketAddr,
    pub player_name: String,
}

impl Plugin for NetworkPlugin {
    fn build(&self, app: &mut App) {
        // run_network expects three outbound channels — one per message
        // category the protocol separates:
        //   * input_rx        : PlayerInputData, sent over UDP each tick
        //   * block_edit_rx   : BlockEditData, sent over TCP (reliable)
        //   * tcp_out_rx      : raw Vec<u8> for any other reliable TCP msg
        let (event_tx, event_rx) = mpsc::unbounded_channel::<NetEvent>();
        let (input_tx, input_rx) = mpsc::unbounded_channel::<PlayerInputData>();
        let (block_edit_tx, block_edit_rx) = mpsc::unbounded_channel::<BlockEditData>();
        let (tcp_out_tx, tcp_out_rx) = mpsc::unbounded_channel::<Vec<u8>>();

        let input_rx = Arc::new(AsyncMutex::new(input_rx));
        let block_edit_rx = Arc::new(AsyncMutex::new(block_edit_rx));
        let tcp_out_rx = Arc::new(AsyncMutex::new(tcp_out_rx));

        let gateway = self.gateway;
        let name = self.player_name.clone();
        std::thread::Builder::new()
            .name("client-net".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::error!("failed to build tokio runtime: {e}");
                        return;
                    }
                };
                rt.block_on(async move {
                    run_network(
                        gateway,
                        name,
                        event_tx,
                        input_rx,
                        block_edit_rx,
                        tcp_out_rx,
                        None,
                    )
                    .await;
                });
            })
            .expect("spawn client-net thread");

        app.insert_resource(NetworkBridge {
            rx: Mutex::new(event_rx),
        })
        .insert_resource(InputSender { tx: input_tx })
        .insert_resource(BlockEditSender { tx: block_edit_tx })
        .insert_resource(TcpSender { tx: tcp_out_tx })
        .init_resource::<NetConnection>()
        .init_resource::<NetSecondaries>()
        .add_message::<GameEvent>()
        .configure_sets(Update, NetworkBridgeSet)
        .add_systems(Update, drain_network_events.in_set(NetworkBridgeSet));
    }
}

/// Receiver end of the mpsc channel — wrapped in a `Mutex` because Bevy
/// `Resource`s must be `Send + Sync` and `mpsc::UnboundedReceiver` is only
/// `Send`.
#[derive(Resource)]
pub struct NetworkBridge {
    rx: Mutex<mpsc::UnboundedReceiver<NetEvent>>,
}

/// UDP-bound player input channel. Camera/movement systems push into this
/// once per frame; `run_network` pulls and serializes into the UDP socket.
#[derive(Resource, Clone)]
pub struct InputSender {
    pub tx: mpsc::UnboundedSender<PlayerInputData>,
}

/// TCP-bound block-edit channel. Gameplay interactions (place/break blocks)
/// push `BlockEditData` here; reliably delivered by `run_network`.
#[derive(Resource, Clone)]
pub struct BlockEditSender {
    pub tx: mpsc::UnboundedSender<BlockEditData>,
}

/// Generic reliable TCP out — pre-serialized flatbuffer bytes pushed here
/// are sent on the primary shard's TCP connection. Used by subsystems that
/// build their own message types (signal config, seat bindings, etc.)
/// without going through the narrow BlockEdit path.
#[derive(Resource, Clone)]
pub struct TcpSender {
    pub tx: mpsc::UnboundedSender<Vec<u8>>,
}

/// Summary of the current network state for HUD / systems to observe.
/// Updated whenever a `Connected` event fires.
#[derive(Resource, Default, Debug, Clone)]
pub struct NetConnection {
    pub connected: bool,
    pub shard_type: u8,
    pub seed: u64,
    pub player_id: u64,
    pub system_seed: u64,
    pub galaxy_seed: u64,
    pub game_time: f64,
    pub last_status: String,
}

/// Bevy `Message` newtype forwarding the full `NetEvent` enum into ECS.
/// Wrapping preserves **every** network event verbatim — primary and
/// secondary shard messages, transitions/promotions, chunk snapshots
/// and deltas, galaxy world state, config state, seat bindings,
/// sub-grid assignments — without the bridge making semantic decisions
/// about which ones matter. Gameplay systems pattern-match on the inner
/// `NetEvent`.
#[derive(Message)]
pub struct GameEvent(pub NetEvent);

/// Drains the mpsc receiver into `Events<GameEvent>` and keeps the
/// `NetConnection` + `NetSecondaries` summary resources up to date.
fn drain_network_events(
    bridge: Res<NetworkBridge>,
    mut conn: ResMut<NetConnection>,
    mut secondaries: ResMut<NetSecondaries>,
    mut out: MessageWriter<GameEvent>,
) {
    let mut rx = match bridge.rx.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    loop {
        let event = match rx.try_recv() {
            Ok(ev) => ev,
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                conn.connected = false;
                conn.last_status = "network thread died".into();
                break;
            }
        };

        // Update summary resources in lockstep with the NetEvent stream
        // BEFORE forwarding, so ECS systems observing GameEvent this frame
        // see a consistent `NetConnection` / `NetSecondaries` snapshot.
        match &event {
            NetEvent::Connected {
                shard_type,
                seed,
                game_time,
                system_seed,
                galaxy_seed,
                player_id,
                ..
            } => {
                conn.connected = true;
                conn.shard_type = *shard_type;
                conn.seed = *seed;
                conn.player_id = *player_id;
                conn.system_seed = *system_seed;
                conn.galaxy_seed = *galaxy_seed;
                conn.game_time = *game_time;
                conn.last_status = format!("connected to shard_type {shard_type}");
            }
            NetEvent::Disconnected(reason) => {
                conn.connected = false;
                conn.last_status = format!("disconnected: {reason}");
            }
            NetEvent::Transitioning { target_shard_type, .. } => {
                conn.last_status = format!("transitioning → shard_type {target_shard_type}");
            }
            NetEvent::WorldState(ws) => {
                conn.game_time = ws.game_time;
            }
            NetEvent::SecondaryConnected { shard_type, seed, .. } => {
                secondaries.active.insert(*seed, *shard_type);
            }
            NetEvent::SecondaryDisconnected { seed } => {
                secondaries.active.remove(seed);
            }
            _ => {}
        }

        out.write(GameEvent(event));
    }
}

/// Tracks which secondary shards are currently pre-connected for
/// compositing. Keyed by shard seed; value is shard_type. Populated by
/// `drain_network_events`; consumed by ShardRegistryPlugin in Phase 3.
#[derive(Resource, Default, Debug, Clone)]
pub struct NetSecondaries {
    pub active: std::collections::HashMap<u64, u8>,
}
