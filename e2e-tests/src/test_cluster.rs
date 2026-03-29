use std::net::SocketAddr;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// A local test cluster with orchestrator and gateway.
/// Starts both as child processes on random ports.
/// Kills them on drop.
pub struct TestCluster {
    orchestrator: Child,
    gateway: Child,
    pub orchestrator_url: String,
    pub gateway_addr: SocketAddr,
}

impl TestCluster {
    /// Start a test cluster. Orchestrator uses --local-mode (child processes, no K8s).
    /// Both bind to random ports.
    pub async fn start() -> Self {
        // Find free ports.
        let orch_http_port = free_port();
        let orch_hb_port = free_port();
        let gw_port = free_port();

        let orchestrator_url = format!("http://127.0.0.1:{orch_http_port}");

        let orchestrator = Command::new(cargo_bin("orchestrator"))
            .args([
                "--http-addr",
                &format!("127.0.0.1:{orch_http_port}"),
                "--heartbeat-addr",
                &format!("127.0.0.1:{orch_hb_port}"),
                "--db-path",
                &format!("/tmp/voxeldust-test-{orch_http_port}.redb"),
                "--provisioner-mode",
                "local",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to start orchestrator");

        // Derive default system seed from universe hierarchy.
        let galaxy_seed = voxeldust_core::seed::derive_galaxy_seed(1, 0);
        let system_seed = voxeldust_core::seed::derive_system_seed(galaxy_seed, 0);

        let gateway = Command::new(cargo_bin("gateway"))
            .args([
                "--listen-addr",
                &format!("127.0.0.1:{gw_port}"),
                "--orchestrator-url",
                &orchestrator_url,
                "--universe-seed",
                "1",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to start gateway");

        let gateway_addr: SocketAddr = format!("127.0.0.1:{gw_port}").parse().unwrap();

        // Wait for orchestrator to be ready.
        let client = reqwest::Client::new();
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if client.get(&format!("{orchestrator_url}/shards")).send().await.is_ok() {
                break;
            }
        }

        Self {
            orchestrator,
            gateway,
            orchestrator_url,
            gateway_addr,
        }
    }

    /// HTTP client for orchestrator API queries.
    pub fn http_client(&self) -> reqwest::Client {
        reqwest::Client::new()
    }
}

impl Drop for TestCluster {
    fn drop(&mut self) {
        let _ = self.orchestrator.kill();
        let _ = self.gateway.kill();
        let _ = self.orchestrator.wait();
        let _ = self.gateway.wait();
    }
}

/// Find a cargo binary in the workspace target directory.
fn cargo_bin(name: &str) -> std::path::PathBuf {
    // From the e2e-tests manifest dir, go up to workspace root, then into target/debug/
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap();
    let path = workspace_root.join("target").join("debug").join(name);
    assert!(
        path.exists(),
        "binary not found: {}. Run `cargo build -p {name}` first.",
        path.display()
    );
    path
}

/// Find a free TCP port by binding to port 0.
fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}
