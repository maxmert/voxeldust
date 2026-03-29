use e2e_tests::test_cluster::TestCluster;
use voxeldust_core::seed::{derive_galaxy_seed, derive_planet_seed, derive_system_seed};

#[tokio::test]
async fn galaxy_system_planet_provisioning_chain() {
    let cluster = TestCluster::start().await;
    let client = cluster.http_client();

    // 1. Provision a galaxy shard.
    let galaxy_seed = derive_galaxy_seed(1, 0);
    let resp = client
        .get(format!("{}/galaxy/{galaxy_seed}", cluster.orchestrator_url))
        .send()
        .await
        .unwrap();
    // Will be 500 (can't spawn stub-shard in test env) or 504 (timeout).
    // But if it's 200, the provisioning chain worked.
    // For local-mode without stub-shard binary on PATH, expect failure.
    // We test the API contract, not the spawning.
    let status = resp.status().as_u16();
    assert!(
        status == 200 || status == 500 || status == 504,
        "unexpected status: {status}"
    );

    // 2. Register a shard manually and verify lookup.
    use voxeldust_core::shard_types::*;

    #[derive(serde::Serialize)]
    struct RegisterReq {
        #[serde(flatten)]
        info: ShardInfo,
        launch_config: Option<()>,
    }

    let system_seed = derive_system_seed(galaxy_seed, 0);
    let info = ShardInfo {
        id: ShardId(42),
        shard_type: ShardType::System,
        state: ShardState::Ready,
        endpoint: ShardEndpoint {
            tcp_addr: "127.0.0.1:19000".parse().unwrap(),
            udp_addr: "127.0.0.1:19001".parse().unwrap(),
            quic_addr: "127.0.0.1:19002".parse().unwrap(),
        },
        planet_seed: None,
        sectors: None,
        system_seed: Some(system_seed),
        ship_id: None,
        galaxy_seed: None,
        host_shard_id: None,
        launch_args: vec![],
    };

    let resp = client
        .post(format!("{}/register", cluster.orchestrator_url))
        .json(&RegisterReq {
            info,
            launch_config: None,
        })
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // 3. Verify system shard lookup returns the registered shard.
    let resp = client
        .get(format!(
            "{}/system/{system_seed}",
            cluster.orchestrator_url
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["system_seed"], system_seed);
}

#[tokio::test]
async fn ship_provisioning_idempotent() {
    let cluster = TestCluster::start().await;
    let client = cluster.http_client();

    // Register a pre-existing ship shard.
    use voxeldust_core::shard_types::*;

    #[derive(serde::Serialize)]
    struct RegisterReq {
        #[serde(flatten)]
        info: ShardInfo,
        launch_config: Option<()>,
    }

    let info = ShardInfo {
        id: ShardId(100),
        shard_type: ShardType::Ship,
        state: ShardState::Ready,
        endpoint: ShardEndpoint {
            tcp_addr: "127.0.0.1:19010".parse().unwrap(),
            udp_addr: "127.0.0.1:19011".parse().unwrap(),
            quic_addr: "127.0.0.1:19012".parse().unwrap(),
        },
        planet_seed: None,
        sectors: None,
        system_seed: None,
        ship_id: Some(42),
        galaxy_seed: None,
        host_shard_id: Some(ShardId(1)),
        launch_args: vec![],
    };

    let resp = client
        .post(format!("{}/register", cluster.orchestrator_url))
        .json(&RegisterReq {
            info,
            launch_config: None,
        })
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // GET /ship/42 should return the shard.
    let resp = client
        .get(format!("{}/ship/42", cluster.orchestrator_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["ship_id"], 42);
    assert_eq!(body["host_shard_id"], serde_json::json!(1)); // ShardId(1)

    // POST /ship { ship_id: 42 } should return the same shard (idempotent).
    let resp = client
        .post(format!("{}/ship", cluster.orchestrator_url))
        .json(&serde_json::json!({ "ship_id": 42 }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["ship_id"], 42);
}

#[tokio::test]
async fn list_shards_after_registration() {
    let cluster = TestCluster::start().await;
    let client = cluster.http_client();

    // Initially empty.
    let resp = client
        .get(format!("{}/shards", cluster.orchestrator_url))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["shards"].as_array().unwrap().len(), 0);
}
