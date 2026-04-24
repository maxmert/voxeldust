//! Bevy plugin bridging `network::run_network` (Tokio, mpsc) to the Bevy
//! World. Phase 3 substance of the migration plan — turns voxydust-next from
//! a demo scaffold into an actual client that connects to a voxeldust
//! dev-cluster and reports live game state.
//!
//! Architecture:
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
//!              │drain_network_events
//!              ▼ (MessageWriter<GameEvent>)
//!    Bevy systems act on events
//! ```
//!
//! The plugin:
//! - Spawns the Tokio runtime + `run_network` task on startup.
//! - Every Update frame, drains the `NetEvent` receiver and re-emits each one
//!   as a Bevy `Message` for ECS systems to consume.
//! - Provides a `NetSender` resource so gameplay systems (camera input,
//!   interactions) can send `ClientMsg` back to the shard.
//! - Exposes `NetConnection` — a lightweight status resource mirroring the
//!   current connection (shard type, seed, player id, game_time).

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use tokio::sync::{mpsc, Mutex as AsyncMutex};

use voxeldust_core::client_message::{BlockEditData, PlayerInputData};

use crate::network::{run_network, NetEvent};
use crate::shard_transition::ShardTransitionSet;

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
        // We own the senders as Bevy resources so gameplay systems can push
        // into them each frame.
        let (event_tx, event_rx) = mpsc::unbounded_channel::<NetEvent>();
        let (input_tx, input_rx) = mpsc::unbounded_channel::<PlayerInputData>();
        let (block_edit_tx, block_edit_rx) = mpsc::unbounded_channel::<BlockEditData>();
        let (tcp_out_tx, tcp_out_rx) = mpsc::unbounded_channel::<Vec<u8>>();

        let input_rx = Arc::new(AsyncMutex::new(input_rx));
        let block_edit_rx = Arc::new(AsyncMutex::new(block_edit_rx));
        let tcp_out_rx = Arc::new(AsyncMutex::new(tcp_out_rx));

        // Spawn the Tokio runtime on a dedicated thread so it can drive
        // network I/O independently of Bevy's scheduler.
        let gateway = self.gateway;
        let name = self.player_name.clone();
        std::thread::Builder::new()
            .name("voxydust-net".into())
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
                        None, // direct=None; always go through the gateway
                    )
                    .await;
                });
            })
            .expect("spawn voxydust-net thread");

        app.insert_resource(NetworkBridge {
            rx: Mutex::new(event_rx),
        })
        .insert_resource(InputSender { tx: input_tx })
        .insert_resource(BlockEditSender { tx: block_edit_tx })
        .insert_resource(TcpSender { tx: tcp_out_tx })
        .init_resource::<NetConnection>()
        .init_resource::<NetSecondaries>()
        .add_message::<GameEvent>()
        // `drain_network_events` MUST run before `ShardTransitionSet` so
        // that `NetEvent::Connected` / `::Transitioning` / `::WorldState`
        // are visible to `handle_connected` / `handle_transitioning` in
        // the same frame they arrive. Without this ordering Bevy can
        // schedule the drain *after* the shard-transition set within
        // a frame, which causes:
        //
        //   Frame N: shard_transition runs (sees nothing), then drain
        //            writes Connected + ChunkSnapshots, then chunk_stream
        //            reads the chunks → drops them because `primary_seed`
        //            is still None.
        //   Frame N+1: shard_transition finally sees Connected → sets
        //            `primary_seed`. But the chunk reader's cursor has
        //            already moved past the snapshots from frame N, so
        //            those chunks are lost forever.
        //
        // Observable symptom: "ship disappeared" — the cockpit interior
        // meshes never rendered because their initial snapshots were
        // dropped in the first-frame race. Ordering drain before the
        // shard-transition set collapses the two stages into one frame.
        .add_systems(Update, drain_network_events.before(ShardTransitionSet));
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

/// Bevy `Message` newtype that forwards the full `NetEvent` enum into the
/// ECS. Wrapping preserves **every** network event verbatim — primary and
/// secondary shard messages, transitions/promotions, chunk snapshots and
/// deltas, galaxy world state, config state, seat bindings, sub-grid
/// assignments — without the bridge making semantic decisions about which
/// ones matter. Gameplay systems pattern-match on the inner `NetEvent`.
///
/// This design preserves the multi-shard + promotion semantics of the
/// server-authoritative protocol: a `Transitioning { new_tcp, new_udp,
/// next_seed, … }` event from the network thread reaches ECS unchanged, and
/// the shard-selection logic stays entirely on the server.
#[derive(Message)]
pub struct GameEvent(pub NetEvent);

/// Drains the mpsc receiver into `Events<GameEvent>` and keeps the
/// `NetConnection` resource up to date. The summary resource mirrors the
/// latest primary-shard state; secondaries have their own tracking state
/// added as gameplay systems need it (see `NetSecondaries` below).
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
                conn.last_status = format!("connected to shard {shard_type}");
            }
            NetEvent::Disconnected(reason) => {
                conn.connected = false;
                conn.last_status = format!("disconnected: {reason}");
            }
            NetEvent::Transitioning { .. } => {
                // Server-driven promotion in flight. Exact target info lives
                // in the NetEvent; gameplay systems read it from the
                // forwarded GameEvent. We just flag the resource so the HUD
                // can render a "transitioning…" banner.
                conn.last_status = "transitioning to new shard".into();
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
            // All remaining variants (SecondaryWorldState, GalaxyWorldState,
            // chunk snapshots / deltas, sub-grid updates, block-config,
            // seat bindings) are passed through unchanged. Gameplay systems
            // responsible for each subsystem pattern-match on the inner
            // NetEvent to react.
            _ => {}
        }

        out.write(GameEvent(event));
    }
}

/// Tracks which secondary shards are currently pre-connected for dual
/// compositing. Mirrors the protocol-level set maintained by `run_network`:
/// keyed by shard seed, value is the shard type (1=PLANET, 2=SHIP, …).
/// Updated by `drain_network_events` from `NetEvent::Secondary{Connected,
/// Disconnected}`. Gameplay systems read this to decide which chunk-source
/// mapping applies to a given incoming snapshot.
#[derive(Resource, Default, Debug, Clone)]
pub struct NetSecondaries {
    pub active: std::collections::HashMap<u64, u8>,
}
