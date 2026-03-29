use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

use voxeldust_core::shard_types::*;

use crate::provisioner::ShardProvisioner;
use crate::registry::{LaunchConfig, ShardRegistry};

/// Shared application state for axum handlers.
struct AppState {
    registry: Arc<RwLock<ShardRegistry>>,
    provisioner: Arc<dyn ShardProvisioner>,
}

pub fn build_router(
    registry: Arc<RwLock<ShardRegistry>>,
    provisioner: Arc<dyn ShardProvisioner>,
) -> Router {
    let state = Arc::new(AppState {
        registry,
        provisioner,
    });

    Router::new()
        .route("/register", post(register_shard))
        .route("/shards", get(list_shards))
        .route("/shard/{id}", get(get_shard))
        .route("/planet/{seed}", get(find_planet_shard))
        .route("/system/{seed}", get(find_system_shard))
        .route("/galaxy/{seed}", get(find_galaxy_shard))
        .route("/ship/{ship_id}", get(find_ship_shard))
        .route("/ship", post(provision_ship_shard))
        .with_state(state)
}

// -- Request/Response types --

#[derive(Deserialize)]
struct RegisterRequest {
    #[serde(flatten)]
    info: ShardInfo,
    launch_config: Option<LaunchConfig>,
}

#[derive(Serialize)]
struct RegisterResponse {
    shard_id: u64,
    status: String,
}

#[derive(Serialize)]
struct ShardResponse {
    #[serde(flatten)]
    info: ShardInfo,
}

#[derive(Serialize)]
struct ShardsResponse {
    shards: Vec<ShardInfo>,
}

// -- Handlers --

async fn register_shard(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterRequest>,
) -> (StatusCode, Json<RegisterResponse>) {
    let id = req.info.id;
    info!(shard_id = %id, shard_type = %req.info.shard_type, "registering shard");

    let mut reg = state.registry.write().await;
    reg.register(req.info, req.launch_config);

    (
        StatusCode::OK,
        Json(RegisterResponse {
            shard_id: id.0,
            status: "registered".to_string(),
        }),
    )
}

async fn list_shards(
    State(state): State<Arc<AppState>>,
) -> Json<ShardsResponse> {
    let reg = state.registry.read().await;
    let shards = reg.list().into_iter().cloned().collect();
    Json(ShardsResponse { shards })
}

async fn get_shard(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<ShardResponse>, StatusCode> {
    let reg = state.registry.read().await;
    match reg.get(ShardId(id)) {
        Some(entry) => Ok(Json(ShardResponse {
            info: entry.info.clone(),
        })),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn find_planet_shard(
    State(state): State<Arc<AppState>>,
    Path(seed): Path<u64>,
) -> Result<Json<ShardsResponse>, StatusCode> {
    // Check if a Ready shard already exists for this planet.
    {
        let reg = state.registry.read().await;
        let shards: Vec<ShardInfo> = reg.find_by_planet(seed).into_iter()
            .filter(|s| s.state == ShardState::Ready)
            .cloned().collect();
        if !shards.is_empty() {
            return Ok(Json(ShardsResponse { shards }));
        }
    }

    // No shard — provision one.
    info!(seed, "no planet shard found, provisioning on demand");
    let launch_config = LaunchConfig {
        binary: "planet-shard".to_string(),
        args: vec![
            "--seed".to_string(),
            seed.to_string(),
        ],
        shard_type: ShardType::Planet,
    };

    let shard_id = state
        .provisioner
        .start_shard(&launch_config)
        .await
        .map_err(|e| {
            warn!(%e, "failed to provision planet shard");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    info!(shard_id = %shard_id, seed, "planet shard provisioning started");

    // Wait for the shard to register and reach Ready (up to 30s).
    match wait_for_shard_ready(&state.registry, shard_id, Duration::from_secs(30)).await {
        Some(info) => Ok(Json(ShardsResponse {
            shards: vec![info],
        })),
        None => {
            warn!(shard_id = %shard_id, "shard did not become ready in time");
            Err(StatusCode::GATEWAY_TIMEOUT)
        }
    }
}

async fn find_system_shard(
    State(state): State<Arc<AppState>>,
    Path(seed): Path<u64>,
) -> Result<Json<ShardResponse>, StatusCode> {
    // Check if a Ready shard already exists for this system.
    {
        let reg = state.registry.read().await;
        if let Some(info) = reg.find_by_system(seed) {
            if info.state == ShardState::Ready {
                return Ok(Json(ShardResponse {
                    info: info.clone(),
                }));
            }
            // Shard exists but not Ready (stopped/draining) — re-provision below.
        }
    }

    // No Ready shard — provision one.
    info!(seed, "no system shard found, provisioning on demand");
    let launch_config = LaunchConfig {
        binary: "system-shard".to_string(),
        args: vec![
            "--seed".to_string(),
            seed.to_string(),
        ],
        shard_type: ShardType::System,
    };

    let shard_id = state
        .provisioner
        .start_shard(&launch_config)
        .await
        .map_err(|e| {
            warn!(%e, "failed to provision system shard");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    info!(shard_id = %shard_id, seed, "system shard provisioning started");

    // Wait for the shard to register and reach Ready (up to 30s).
    match wait_for_shard_ready(&state.registry, shard_id, Duration::from_secs(30)).await {
        Some(info) => Ok(Json(ShardResponse { info })),
        None => {
            warn!(shard_id = %shard_id, "shard did not become ready in time");
            Err(StatusCode::GATEWAY_TIMEOUT)
        }
    }
}

async fn find_galaxy_shard(
    State(state): State<Arc<AppState>>,
    Path(seed): Path<u64>,
) -> Result<Json<ShardResponse>, StatusCode> {
    // Check if a shard already exists for this galaxy.
    {
        let reg = state.registry.read().await;
        if let Some(info) = reg.find_by_galaxy(seed) {
            return Ok(Json(ShardResponse {
                info: info.clone(),
            }));
        }
    }

    // No shard — provision one.
    info!(seed, "no galaxy shard found, provisioning on demand");
    let launch_config = LaunchConfig {
        binary: "stub-shard".to_string(),
        args: vec![
            "--shard-type".to_string(),
            "galaxy".to_string(),
            "--seed".to_string(),
            seed.to_string(),
        ],
        shard_type: ShardType::Galaxy,
    };

    let shard_id = state
        .provisioner
        .start_shard(&launch_config)
        .await
        .map_err(|e| {
            warn!(%e, "failed to provision galaxy shard");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    info!(shard_id = %shard_id, seed, "galaxy shard provisioning started");

    match wait_for_shard_ready(&state.registry, shard_id, Duration::from_secs(30)).await {
        Some(info) => Ok(Json(ShardResponse { info })),
        None => {
            warn!(shard_id = %shard_id, "galaxy shard did not become ready in time");
            Err(StatusCode::GATEWAY_TIMEOUT)
        }
    }
}

/// Lookup an existing ship shard by ship_id.
async fn find_ship_shard(
    State(state): State<Arc<AppState>>,
    Path(ship_id): Path<u64>,
) -> Result<Json<ShardResponse>, StatusCode> {
    let reg = state.registry.read().await;
    match reg.find_by_ship(ship_id) {
        Some(info) => Ok(Json(ShardResponse {
            info: info.clone(),
        })),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// Request body for POST /ship.
#[derive(Deserialize)]
struct ProvisionShipRequest {
    /// Unique ship identifier (assigned by the system shard that owns the ship entity).
    ship_id: u64,
    /// The shard managing this ship's exterior (system or planet shard).
    host_shard_id: Option<u64>,
}

/// Provision a new ship shard on demand.
/// Called by the system shard when a player boards a ship that doesn't have a shard yet.
async fn provision_ship_shard(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProvisionShipRequest>,
) -> Result<Json<ShardResponse>, StatusCode> {
    // Check if a Ready shard already exists.
    {
        let reg = state.registry.read().await;
        if let Some(info) = reg.find_by_ship(req.ship_id) {
            if info.state == ShardState::Ready {
                return Ok(Json(ShardResponse {
                    info: info.clone(),
                }));
            }
            // Shard exists but not Ready — re-provision below.
        }
    }

    info!(ship_id = req.ship_id, host = ?req.host_shard_id, "provisioning ship shard on demand");
    let mut args = vec![
        "--ship-id".to_string(),
        req.ship_id.to_string(),
    ];
    if let Some(host_id) = req.host_shard_id {
        args.push("--host-shard".to_string());
        args.push(host_id.to_string());
    }
    let launch_config = LaunchConfig {
        binary: "ship-shard".to_string(),
        args,
        shard_type: ShardType::Ship,
    };

    let shard_id = state
        .provisioner
        .start_shard(&launch_config)
        .await
        .map_err(|e| {
            warn!(%e, "failed to provision ship shard");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    info!(shard_id = %shard_id, ship_id = req.ship_id, "ship shard provisioning started");

    match wait_for_shard_ready(&state.registry, shard_id, Duration::from_secs(30)).await {
        Some(info) => Ok(Json(ShardResponse { info })),
        None => {
            warn!(shard_id = %shard_id, "ship shard did not become ready in time");
            Err(StatusCode::GATEWAY_TIMEOUT)
        }
    }
}

/// Polls the registry until a shard reaches Ready state or timeout expires.
async fn wait_for_shard_ready(
    registry: &Arc<RwLock<ShardRegistry>>,
    shard_id: ShardId,
    timeout: Duration,
) -> Option<ShardInfo> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut interval = tokio::time::interval(Duration::from_millis(500));

    loop {
        interval.tick().await;

        if tokio::time::Instant::now() >= deadline {
            return None;
        }

        let reg = registry.read().await;
        if let Some(entry) = reg.get(shard_id) {
            if entry.info.state == ShardState::Ready {
                return Some(entry.info.clone());
            }
        }
    }
}
