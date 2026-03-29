mod config;
mod router;
mod session;

use std::sync::Arc;

use clap::Parser;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tracing::{info, warn};

use voxeldust_core::client_message::{ClientMsg, ServerMsg};
use voxeldust_core::handoff::ShardRedirect;
use voxeldust_core::seed::{derive_galaxy_seed, derive_system_seed};
use voxeldust_core::shard_types::SessionToken;

use crate::config::GatewayConfig;
use crate::router::Router as ShardRouter;
use crate::session::SessionStore;

fn main() {
    let config = GatewayConfig::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(run(config));
}

async fn run(config: GatewayConfig) {
    info!(listen = %config.listen_addr, orchestrator = %config.orchestrator_url, "gateway starting");

    let router = Arc::new(ShardRouter::new(config.orchestrator_url.clone()));
    let session_store = Arc::new(SessionStore::open(&config.db_path).expect("failed to open session db"));

    let listener = TcpListener::bind(config.listen_addr)
        .await
        .expect("failed to bind gateway TCP");

    info!(addr = %config.listen_addr, "gateway ready, accepting clients");

    loop {
        match listener.accept().await {
            Ok((stream, peer_addr)) => {
                let router = router.clone();
                let session_store = session_store.clone();
                // Derive the default system seed from universe hierarchy.
                let galaxy_seed = derive_galaxy_seed(config.universe_seed, config.default_galaxy_index);
                let default_system_seed = derive_system_seed(galaxy_seed, config.default_star_index);
                tokio::spawn(async move {
                    if let Err(e) =
                        handle_client(stream, peer_addr, &router, &session_store, default_system_seed)
                            .await
                    {
                        warn!(%peer_addr, %e, "client handling failed");
                    }
                });
            }
            Err(e) => {
                warn!(%e, "TCP accept error");
            }
        }
    }
}

async fn handle_client(
    mut stream: tokio::net::TcpStream,
    peer_addr: std::net::SocketAddr,
    router: &ShardRouter,
    session_store: &SessionStore,
    default_system_seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read length-prefixed ClientMessage.
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;

    if len > 65536 {
        return Err(format!("message too large: {len}").into());
    }

    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;

    let msg = ClientMsg::deserialize(&buf)?;

    let player_name = match msg {
        ClientMsg::Connect { ref player_name } => player_name.clone(),
        _ => return Err("expected Connect message".into()),
    };

    info!(%peer_addr, %player_name, "client connecting via gateway");

    // Check for existing session — validate shard is still alive.
    if let Some(session) = session_store.find_by_name(&player_name) {
        if router.validate_shard(session.last_shard_id).await {
            info!(%player_name, shard_id = session.last_shard_id.0, "resuming existing session");

            let redirect = ServerMsg::ShardRedirect(ShardRedirect {
                session_token: session.session_token,
                target_tcp_addr: session.target_tcp_addr,
                target_udp_addr: session.target_udp_addr,
                shard_id: session.last_shard_id,
            });

            send_server_msg(&mut stream, &redirect).await?;
            return Ok(());
        } else {
            info!(%player_name, shard_id = session.last_shard_id.0, "stale session — shard not Ready, reprovisioning");
            session_store.remove(&player_name);
        }
    }

    // Provision a ship shard for this player (ensures system shard exists too).
    let shard_info = router.find_shard_for_player(default_system_seed, &player_name).await?;

    // Generate session token.
    let session_token = SessionToken(rand_u64());

    // Save session.
    session_store.save(session::SessionRecord {
        session_token,
        player_name: player_name.clone(),
        last_shard_id: shard_info.id,
        target_tcp_addr: shard_info.endpoint.tcp_addr.to_string(),
        target_udp_addr: shard_info.endpoint.udp_addr.to_string(),
    });

    // Send redirect to client.
    let redirect = ServerMsg::ShardRedirect(ShardRedirect {
        session_token,
        target_tcp_addr: shard_info.endpoint.tcp_addr.to_string(),
        target_udp_addr: shard_info.endpoint.udp_addr.to_string(),
        shard_id: shard_info.id,
    });

    send_server_msg(&mut stream, &redirect).await?;
    info!(%player_name, shard_id = %shard_info.id, "client redirected");

    Ok(())
}

async fn send_server_msg(
    stream: &mut tokio::net::TcpStream,
    msg: &ServerMsg,
) -> Result<(), std::io::Error> {
    let data = msg.serialize();
    let len_bytes = (data.len() as u32).to_be_bytes();
    stream.write_all(&len_bytes).await?;
    stream.write_all(&data).await?;
    stream.flush().await?;
    Ok(())
}

fn rand_u64() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let s = RandomState::new();
    let mut h = s.build_hasher();
    h.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );
    h.finish()
}
