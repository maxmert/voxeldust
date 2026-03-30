use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use tracing::info;
use voxeldust_core::shard_types::{ShardId, ShardInfo, ShardState};

/// Routes clients to the appropriate shard by querying the orchestrator.
pub struct Router {
    orchestrator_url: String,
    client: reqwest::Client,
}

#[derive(serde::Deserialize)]
struct ShardResponse {
    #[serde(flatten)]
    info: ShardInfo,
}

impl Router {
    pub fn new(orchestrator_url: String) -> Self {
        Self {
            orchestrator_url,
            client: reqwest::Client::new(),
        }
    }

    /// Find or provision a ship shard for a new player.
    /// 1. Ensures the system shard exists (provisions on demand)
    /// 2. Provisions a ship shard with the system shard as host
    /// 3. Returns the ship shard endpoint
    pub async fn find_shard_for_player(
        &self,
        system_seed: u64,
        player_name: &str,
    ) -> Result<ShardInfo, Box<dyn std::error::Error>> {
        // Step 1: Ensure system shard exists.
        let system_url = format!("{}/system/{}", self.orchestrator_url, system_seed);
        let resp = self.client.get(&system_url).send().await?;

        let system_shard: ShardInfo = if resp.status().is_success() {
            let body: ShardResponse = resp.json().await?;
            body.info
        } else {
            return Err(format!("failed to provision system shard (status {})", resp.status()).into());
        };

        info!(system_shard_id = system_shard.id.0, "system shard ready");

        // Step 2: Provision ship shard for this player.
        let ship_id = hash_player_name(player_name);

        // Check if ship shard already exists.
        let ship_url = format!("{}/ship/{}", self.orchestrator_url, ship_id);
        let resp = self.client.get(&ship_url).send().await?;

        if resp.status().is_success() {
            let body: ShardResponse = resp.json().await?;
            if body.info.state == ShardState::Ready {
                info!(ship_id, shard_id = body.info.id.0, "existing ship shard found");
                return Ok(body.info);
            }
        }

        // Provision new ship shard.
        let provision_url = format!("{}/ship", self.orchestrator_url);
        let resp = self.client
            .post(&provision_url)
            .json(&serde_json::json!({
                "ship_id": ship_id,
                "host_shard_id": system_shard.id.0,
                "system_seed": system_seed,
            }))
            .send()
            .await?;

        if resp.status().is_success() {
            let body: ShardResponse = resp.json().await?;
            info!(ship_id, shard_id = body.info.id.0, "ship shard provisioned");
            Ok(body.info)
        } else {
            Err(format!("failed to provision ship shard (status {})", resp.status()).into())
        }
    }

    /// Validate that a shard is still alive and Ready.
    pub async fn validate_shard(&self, shard_id: ShardId) -> bool {
        let url = format!("{}/shard/{}", self.orchestrator_url, shard_id.0);
        match self.client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<ShardResponse>().await {
                    Ok(body) => body.info.state == ShardState::Ready,
                    Err(_) => false,
                }
            }
            _ => false,
        }
    }
}

/// Deterministic ship_id from player name.
fn hash_player_name(name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}
