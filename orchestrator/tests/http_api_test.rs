use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use voxeldust_core::shard_types::*;

/// Test helper: start orchestrator HTTP server on a random port and return the URL.
async fn start_test_orchestrator() -> (String, Arc<RwLock<orchestrator::registry::ShardRegistry>>) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.redb");

    let registry = orchestrator::registry::ShardRegistry::open(&db_path).unwrap();
    let registry = Arc::new(RwLock::new(registry));
    let provisioner = Arc::new(orchestrator::provisioner::LocalProvisioner::new());

    let app = orchestrator::http_api::build_router(registry.clone(), provisioner, 1000000);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Give server a moment to start.
    tokio::time::sleep(Duration::from_millis(50)).await;

    (format!("http://{addr}"), registry)
}

fn test_endpoint(port_base: u16) -> ShardEndpoint {
    ShardEndpoint {
        tcp_addr: format!("127.0.0.1:{}", port_base).parse::<SocketAddr>().unwrap(),
        udp_addr: format!("127.0.0.1:{}", port_base + 1).parse::<SocketAddr>().unwrap(),
        quic_addr: format!("127.0.0.1:{}", port_base + 2).parse::<SocketAddr>().unwrap(),
    }
}

#[tokio::test]
async fn register_and_list_shards() {
    let (url, _registry) = start_test_orchestrator().await;
    let client = reqwest::Client::new();

    // Register a planet shard.
    let info = ShardInfo {
        id: ShardId(1),
        shard_type: ShardType::Planet,
        state: ShardState::Ready,
        endpoint: test_endpoint(7777),
        planet_seed: Some(42),
        sectors: Some(vec![0, 1, 2, 3, 4, 5]),
        system_seed: None,
        ship_id: None,
        galaxy_seed: None,
        host_shard_id: None,
        launch_args: vec![],
    };

    // Use serde to get the correct JSON format for ShardInfo.
    #[derive(serde::Serialize)]
    struct RegisterRequest {
        #[serde(flatten)]
        info: ShardInfo,
        launch_config: Option<()>,
    }

    let resp = client
        .post(format!("{url}/register"))
        .json(&RegisterRequest {
            info: info.clone(),
            launch_config: None,
        })
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "register failed: {}", resp.status());

    // List shards.
    let resp = client.get(format!("{url}/shards")).send().await.unwrap();
    assert!(resp.status().is_success());

    let body: serde_json::Value = resp.json().await.unwrap();
    let shards = body["shards"].as_array().unwrap();
    assert_eq!(shards.len(), 1);
}

#[tokio::test]
async fn lookup_shard_by_id() {
    let (url, registry) = start_test_orchestrator().await;
    let client = reqwest::Client::new();

    // Register directly via registry.
    {
        let mut reg = registry.write().await;
        reg.register(
            ShardInfo {
                id: ShardId(5),
                shard_type: ShardType::System,
                state: ShardState::Ready,
                endpoint: test_endpoint(9777),
                planet_seed: None,
                sectors: None,
                system_seed: Some(100),
                ship_id: None,
            galaxy_seed: None,
            host_shard_id: None,
                launch_args: vec![],
            },
            None,
        );
    }

    // Lookup by ID.
    let resp = client.get(format!("{url}/shard/5")).send().await.unwrap();
    assert!(resp.status().is_success());

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["system_seed"], 100);

    // Lookup non-existent.
    let resp = client.get(format!("{url}/shard/999")).send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}

#[tokio::test]
async fn find_planet_shard_by_seed() {
    let (url, registry) = start_test_orchestrator().await;
    let client = reqwest::Client::new();

    {
        let mut reg = registry.write().await;
        reg.register(
            ShardInfo {
                id: ShardId(1),
                shard_type: ShardType::Planet,
                state: ShardState::Ready,
                endpoint: test_endpoint(7777),
                planet_seed: Some(42),
                sectors: Some(vec![0, 1, 2]),
                system_seed: None,
                ship_id: None,
            galaxy_seed: None,
            host_shard_id: None,
                launch_args: vec![],
            },
            None,
        );
        reg.register(
            ShardInfo {
                id: ShardId(2),
                shard_type: ShardType::Planet,
                state: ShardState::Ready,
                endpoint: test_endpoint(7780),
                planet_seed: Some(42),
                sectors: Some(vec![3, 4, 5]),
                system_seed: None,
                ship_id: None,
            galaxy_seed: None,
            host_shard_id: None,
                launch_args: vec![],
            },
            None,
        );
    }

    let resp = client.get(format!("{url}/planet/42")).send().await.unwrap();
    assert!(resp.status().is_success());

    let body: serde_json::Value = resp.json().await.unwrap();
    let shards = body["shards"].as_array().unwrap();
    assert_eq!(shards.len(), 2);
}

#[tokio::test]
async fn find_system_shard_by_seed() {
    let (url, registry) = start_test_orchestrator().await;
    let client = reqwest::Client::new();

    {
        let mut reg = registry.write().await;
        reg.register(
            ShardInfo {
                id: ShardId(10),
                shard_type: ShardType::System,
                state: ShardState::Ready,
                endpoint: test_endpoint(9777),
                planet_seed: None,
                sectors: None,
                system_seed: Some(200),
                ship_id: None,
            galaxy_seed: None,
            host_shard_id: None,
                launch_args: vec![],
            },
            None,
        );
    }

    let resp = client
        .get(format!("{url}/system/200"))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // Querying a system with no shard triggers on-demand provisioning.
    // In tests, the LocalProvisioner can't find the binary, so it returns 500.
    let resp = client
        .get(format!("{url}/system/999"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 500);
}

#[tokio::test]
async fn heartbeat_via_udp_updates_registry() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.redb");

    let registry = orchestrator::registry::ShardRegistry::open(&db_path).unwrap();
    let registry = Arc::new(RwLock::new(registry));

    // Pre-register a shard in Provisioning state.
    {
        let mut reg = registry.write().await;
        reg.register(
            ShardInfo {
                id: ShardId(1),
                shard_type: ShardType::Planet,
                state: ShardState::Provisioning,
                endpoint: test_endpoint(7777),
                planet_seed: Some(42),
                sectors: Some(vec![0, 1, 2, 3, 4, 5]),
                system_seed: None,
                ship_id: None,
            galaxy_seed: None,
            host_shard_id: None,
                launch_args: vec![],
            },
            None,
        );
    }

    // Start heartbeat listener on random port.
    // Bind first to get the address, then pass it to the listener.
    // The listener will bind internally, so we need to pick a free port.
    let temp_socket = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
    let hb_addr = temp_socket.local_addr().unwrap();
    drop(temp_socket); // Free the port for the listener.

    let cancel = tokio_util::sync::CancellationToken::new();
    let hb_registry = registry.clone();
    let hb_cancel = cancel.clone();
    tokio::spawn(async move {
        orchestrator::heartbeat::run_heartbeat_listener(hb_addr, hb_registry, hb_cancel).await;
    });

    // Give the listener time to bind.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Send a heartbeat.
    use voxeldust_core::shard_message::ShardMsg;
    let msg = ShardMsg::Heartbeat(ShardHeartbeat {
        shard_id: ShardId(1),
        tick_ms: 45.0,
        p99_tick_ms: 48.0,
        player_count: 5,
        chunk_count: 200,
    });

    let payload = msg.serialize();
    let mut packet = Vec::with_capacity(4 + payload.len());
    packet.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    packet.extend_from_slice(&payload);

    let sender = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    sender.send_to(&packet, hb_addr).await.unwrap();

    // Wait for processing.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify registry updated.
    let reg = registry.read().await;
    let entry = reg.get(ShardId(1)).unwrap();
    assert_eq!(entry.info.state, ShardState::Ready); // transitioned from Provisioning
    assert!(entry.last_metrics.is_some());
    assert_eq!(entry.last_metrics.as_ref().unwrap().player_count, 5);

    cancel.cancel();
}
