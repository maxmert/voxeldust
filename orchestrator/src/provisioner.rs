use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{info, warn};

use voxeldust_core::shard_types::ShardId;

use crate::registry::LaunchConfig;

#[derive(Debug, Error)]
pub enum ProvisionError {
    #[error("failed to start process: {0}")]
    SpawnFailed(#[from] std::io::Error),
    #[error("shard {0} not found")]
    NotFound(ShardId),
    #[error("kubernetes API error: {0}")]
    KubeError(String),
    #[error("no ports available in the dynamic range")]
    NoPortsAvailable,
}

/// Trait for starting/stopping shard processes.
/// Object-safe via boxed futures.
pub trait ShardProvisioner: Send + Sync {
    fn start_shard(
        &self,
        config: &LaunchConfig,
    ) -> Pin<Box<dyn Future<Output = Result<ShardId, ProvisionError>> + Send + '_>>;

    fn stop_shard(
        &self,
        id: ShardId,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProvisionError>> + Send + '_>>;
}

/// Starts shards as local child processes (for --local-mode development).
pub struct LocalProvisioner {
    children: Mutex<HashMap<ShardId, Child>>,
    next_id: AtomicU64,
}

impl LocalProvisioner {
    pub fn new() -> Self {
        Self {
            children: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1000),
        }
    }
}

impl ShardProvisioner for LocalProvisioner {
    fn start_shard(
        &self,
        config: &LaunchConfig,
    ) -> Pin<Box<dyn Future<Output = Result<ShardId, ProvisionError>> + Send + '_>> {
        let config = config.clone();
        Box::pin(async move {
            let id = ShardId(self.next_id.fetch_add(1, Ordering::Relaxed));

            info!(
                shard_id = %id,
                binary = %config.binary,
                args = ?config.args,
                "starting local shard process"
            );

            let child = Command::new(&config.binary)
                .args(&config.args)
                .arg("--shard-id")
                .arg(id.0.to_string())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .spawn()?;

            self.children.lock().await.insert(id, child);
            Ok(id)
        })
    }

    fn stop_shard(
        &self,
        id: ShardId,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProvisionError>> + Send + '_>> {
        Box::pin(async move {
            let mut children = self.children.lock().await;
            if let Some(mut child) = children.remove(&id) {
                info!(shard_id = %id, "stopping local shard process");
                if let Err(e) = child.kill().await {
                    warn!(shard_id = %id, %e, "failed to kill shard process");
                }
            }
            Ok(())
        })
    }
}
