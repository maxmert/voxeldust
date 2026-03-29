use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use k8s_openapi::api::core::v1::Pod;
use kube::api::{Api, DeleteParams, PostParams};
use kube::Client;
use tokio::sync::Mutex;
use tracing::{info, warn};

use voxeldust_core::shard_types::ShardId;

use crate::provisioner::{ProvisionError, ShardProvisioner};
use crate::registry::LaunchConfig;

/// Configuration for the K8s provisioner.
#[derive(Debug, Clone)]
pub struct KubernetesProvisionerConfig {
    /// K8s namespace for shard pods.
    pub namespace: String,
    /// Container image for shard pods.
    pub shard_image: String,
    /// Start of port range for dynamic shards (4 ports per shard).
    pub port_range_start: u16,
    /// End of port range (exclusive).
    pub port_range_end: u16,
    /// Orchestrator HTTP URL as seen from within the cluster.
    pub orchestrator_url: String,
    /// Orchestrator heartbeat UDP address as seen from within the cluster.
    pub orchestrator_heartbeat_addr: String,
    /// Host to advertise in shard endpoints (e.g. "127.0.0.1" for k3d).
    pub advertise_host: String,
}

struct ShardPodInfo {
    pod_name: String,
    base_port: u16,
}

/// Creates and manages shard pods via the Kubernetes API.
pub struct KubernetesProvisioner {
    client: Client,
    config: KubernetesProvisionerConfig,
    pods_api: Api<Pod>,
    next_id: AtomicU64,
    active_shards: Mutex<HashMap<ShardId, ShardPodInfo>>,
    port_pool: Mutex<Vec<u16>>,
}

impl KubernetesProvisioner {
    pub async fn new(config: KubernetesProvisionerConfig) -> Result<Self, ProvisionError> {
        let client =
            Client::try_default()
                .await
                .map_err(|e| ProvisionError::KubeError(e.to_string()))?;
        let pods_api: Api<Pod> = Api::namespaced(client.clone(), &config.namespace);

        // Build port pool: each shard gets 4 consecutive ports.
        let mut port_pool = Vec::new();
        let mut port = config.port_range_start;
        while port + 3 < config.port_range_end {
            port_pool.push(port);
            port += 4;
        }
        // Reverse so we pop from the end (LIFO = lower ports first).
        port_pool.reverse();

        info!(
            namespace = %config.namespace,
            image = %config.shard_image,
            port_slots = port_pool.len(),
            "kubernetes provisioner initialized"
        );

        Ok(Self {
            client,
            config,
            pods_api,
            next_id: AtomicU64::new(1000),
            active_shards: Mutex::new(HashMap::new()),
            port_pool: Mutex::new(port_pool),
        })
    }

    fn build_pod(&self, id: ShardId, config: &LaunchConfig, base_port: u16) -> Pod {
        let tcp_port = base_port;
        let udp_port = base_port + 1;
        let quic_port = base_port + 2;
        let healthz_port = base_port + 3;
        let pod_name = format!("shard-{}", id.0);

        // Build container args: LaunchConfig args + injected infra args.
        let mut args = config.args.clone();
        args.extend([
            "--shard-id".to_string(),
            id.0.to_string(),
            "--tcp-port".to_string(),
            tcp_port.to_string(),
            "--udp-port".to_string(),
            udp_port.to_string(),
            "--quic-port".to_string(),
            quic_port.to_string(),
            "--healthz-port".to_string(),
            healthz_port.to_string(),
            "--orchestrator".to_string(),
            self.config.orchestrator_url.clone(),
            "--orchestrator-heartbeat".to_string(),
            self.config.orchestrator_heartbeat_addr.clone(),
            "--advertise-host".to_string(),
            self.config.advertise_host.clone(),
        ]);

        let labels = serde_json::json!({
            "app": "voxeldust-shard",
            "shard-id": id.0.to_string(),
            "shard-type": format!("{}", config.shard_type),
        });

        serde_json::from_value(serde_json::json!({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "namespace": self.config.namespace,
                "labels": labels,
            },
            "spec": {
                "hostNetwork": true,
                "dnsPolicy": "ClusterFirstWithHostNet",
                "restartPolicy": "Never",
                "containers": [{
                    "name": "shard",
                    "image": match config.shard_type {
                        voxeldust_core::shard_types::ShardType::System => "voxeldust-system-shard:latest",
                        voxeldust_core::shard_types::ShardType::Planet => "voxeldust-planet-shard:latest",
                        voxeldust_core::shard_types::ShardType::Ship => "voxeldust-ship-shard:latest",
                        _ => &self.config.shard_image, // galaxy, etc. fall back to stub
                    },
                    "imagePullPolicy": "Never",
                    "args": args,
                    "ports": [
                        { "containerPort": tcp_port, "protocol": "TCP" },
                        { "containerPort": udp_port, "protocol": "UDP" },
                        { "containerPort": quic_port, "protocol": "UDP" },
                        { "containerPort": healthz_port, "protocol": "TCP" },
                    ],
                    "livenessProbe": {
                        "httpGet": {
                            "path": "/healthz",
                            "port": healthz_port as i32,
                        },
                        "initialDelaySeconds": 5,
                        "periodSeconds": 5,
                    },
                    "resources": {
                        "limits": {
                            "memory": "256Mi",
                            "cpu": "500m",
                        },
                        "requests": {
                            "memory": "128Mi",
                            "cpu": "100m",
                        },
                    },
                }],
            },
        }))
        .expect("failed to build pod spec")
    }
}

impl ShardProvisioner for KubernetesProvisioner {
    fn start_shard(
        &self,
        config: &LaunchConfig,
    ) -> Pin<Box<dyn Future<Output = Result<ShardId, ProvisionError>> + Send + '_>> {
        let config = config.clone();
        Box::pin(async move {
            let id = ShardId(self.next_id.fetch_add(1, Ordering::Relaxed));

            // Allocate a port from the pool.
            let base_port = {
                let mut pool = self.port_pool.lock().await;
                pool.pop().ok_or(ProvisionError::NoPortsAvailable)?
            };

            let pod = self.build_pod(id, &config, base_port);
            let pod_name = format!("shard-{}", id.0);

            info!(
                shard_id = %id,
                pod = %pod_name,
                tcp_port = base_port,
                "creating shard pod"
            );

            self.pods_api
                .create(&PostParams::default(), &pod)
                .await
                .map_err(|e| ProvisionError::KubeError(e.to_string()))?;

            self.active_shards.lock().await.insert(
                id,
                ShardPodInfo {
                    pod_name,
                    base_port,
                },
            );

            Ok(id)
        })
    }

    fn stop_shard(
        &self,
        id: ShardId,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProvisionError>> + Send + '_>> {
        Box::pin(async move {
            let mut active = self.active_shards.lock().await;
            if let Some(pod_info) = active.remove(&id) {
                info!(shard_id = %id, pod = %pod_info.pod_name, "deleting shard pod");

                if let Err(e) = self
                    .pods_api
                    .delete(&pod_info.pod_name, &DeleteParams::default())
                    .await
                {
                    warn!(shard_id = %id, %e, "failed to delete pod");
                }

                // Return port to pool.
                self.port_pool.lock().await.push(pod_info.base_port);
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_pool_allocation() {
        // Verify port pool is built correctly.
        let mut pool = Vec::new();
        let start = 10000u16;
        let end = 10020u16;
        let mut port = start;
        while port + 3 < end {
            pool.push(port);
            port += 4;
        }
        // 10000, 10004, 10008, 10012, 10016 = 5 slots for 20 ports
        assert_eq!(pool.len(), 5);
        assert_eq!(pool[0], 10000);
        assert_eq!(pool[4], 10016);
    }

    #[test]
    fn port_pool_100_range() {
        let mut pool = Vec::new();
        let start = 10000u16;
        let end = 10100u16;
        let mut port = start;
        while port + 3 < end {
            pool.push(port);
            port += 4;
        }
        // 100 ports / 4 per shard = 25 slots
        assert_eq!(pool.len(), 25);
    }
}
