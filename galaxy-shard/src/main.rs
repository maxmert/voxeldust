use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use clap::Parser;
use glam::{DQuat, DVec3};
use tokio::sync::mpsc;
use tracing::{info, warn};

use voxeldust_core::autopilot::{FlightPhase, WARP_ACCELERATION_GU, WARP_MAX_SPEED_GU};
use voxeldust_core::client_message::{GalaxyWorldStateData, JoinResponseData, ServerMsg};
use voxeldust_core::ecs::{
    self, PlayerId, PlayerName, Position, Rotation, ShipId, SourceSystemShard, Velocity, WarpState,
};
use voxeldust_core::galaxy::{system_outer_radius, GalaxyMap, GALAXY_UNIT_IN_BLOCKS};
use voxeldust_core::handoff::{self, GalaxyHandoffContext, ShardPreConnect};
use voxeldust_core::shard_message::{
    LightingInfoData, ShardMsg, ShipPositionUpdate, SystemSceneUpdateData,
};
use voxeldust_core::shard_types::{SessionToken, ShardId, ShardType};
use voxeldust_shard_common::client_listener;
use voxeldust_shard_common::harness::{NetworkBridge, ShardHarness, ShardHarnessConfig};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "galaxy-shard",
    about = "Voxeldust galaxy shard — interstellar warp travel"
)]
struct Args {
    #[arg(long)]
    shard_id: u64,
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    orchestrator: String,
    #[arg(long, default_value = "127.0.0.1:9090")]
    orchestrator_heartbeat: String,
    #[arg(long, default_value = "11777")]
    tcp_port: u16,
    #[arg(long, default_value = "11778")]
    udp_port: u16,
    #[arg(long, default_value = "11779")]
    quic_port: u16,
    #[arg(long, default_value = "11081")]
    healthz_port: u16,
    #[arg(long)]
    advertise_host: Option<String>,
}

// ---------------------------------------------------------------------------
// Resources (galaxy-shard-specific)
// ---------------------------------------------------------------------------

/// Immutable galaxy map generated from seed.
#[derive(Resource)]
struct GalaxyMapResource(GalaxyMap);

/// Galaxy seed for deterministic generation.
#[derive(Resource)]
struct GalaxySeed(u64);

/// This shard's identity.
#[derive(Resource)]
struct ShardIdentity(ShardId);

/// System shards already provisioned: star_index → shard_id.
#[derive(Resource, Default)]
struct ProvisionedSystems(HashMap<u32, ShardId>);

/// Star indices currently being provisioned (in-flight HTTP requests).
#[derive(Resource, Default)]
struct ProvisioningInFlight(HashSet<u32>);

/// Channel for receiving async provisioning results.
#[derive(Resource)]
struct ProvisionReceiver(mpsc::Receiver<ProvisionResult>);

/// Channel sender for async provisioning tasks to return results.
#[derive(Resource, Clone)]
struct ProvisionSender(mpsc::Sender<ProvisionResult>);

/// Result of an async system shard provisioning request.
struct ProvisionResult {
    star_index: u32,
    shard_id: ShardId,
}

/// Orchestrator URL for provisioning requests.
#[derive(Resource)]
struct OrchestratorUrl(String);

/// Shared HTTP client for provisioning (connection pooled).
#[derive(Resource)]
struct HttpClient(reqwest::Client);

/// Ship shard references: ship_id → (shard_id, quic_addr).
/// Kept as a resource instead of per-entity so we can update it atomically
/// from the peer registry without querying every entity.
#[derive(Resource, Default)]
struct ShipShardDirectory(HashMap<u64, (ShardId, SocketAddr)>);

// ---------------------------------------------------------------------------
// Events (galaxy-shard-specific, not shared across shards)
// ---------------------------------------------------------------------------

/// A new TCP client connected.
#[derive(Message)]
struct ClientConnectedEvent {
    session_token: SessionToken,
    tcp_write: Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>,
}

/// A PlayerHandoff arrived via QUIC for a new warp ship.
#[derive(Message)]
struct WarpHandoffEvent {
    handoff: handoff::PlayerHandoff,
    source_shard: ShardId,
    ship_shard_id: Option<ShardId>,
}

/// A WarpAutopilotCommand arrived: change target or disengage.
#[derive(Message)]
struct WarpAutopilotEvent {
    ship_id: u64,
    target_star_index: u32,
}

/// Provisioning completed: a system shard is ready.
#[derive(Message)]
struct ProvisionCompletedEvent {
    star_index: u32,
    shard_id: ShardId,
}

// ---------------------------------------------------------------------------
// System sets for ordering
// ---------------------------------------------------------------------------

/// Schedule ordering: Bridge → Spawn → Physics → Detection → Broadcast.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum GalaxySet {
    /// Drain async channels into ECS events.
    Bridge,
    /// Spawn/despawn entities from events.
    Spawn,
    /// Warp physics integration.
    Physics,
    /// SOI detection and handoff.
    Detection,
    /// Broadcast state to clients and ship shards.
    Broadcast,
    /// Periodic diagnostics.
    Diagnostics,
}

// ---------------------------------------------------------------------------
// Bridge systems (drain async → ECS events)
// ---------------------------------------------------------------------------

fn drain_connects(
    mut bridge: ResMut<NetworkBridge>,
    mut connect_events: MessageWriter<ClientConnectedEvent>,
) {
    for _ in 0..16 {
        let event = match bridge.connect_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        let conn = event.connection;
        if let Ok(mut reg) = bridge.client_registry.try_write() {
            reg.register(&conn);
        }
        connect_events.write(ClientConnectedEvent {
            session_token: conn.session_token,
            tcp_write: conn.tcp_write.clone(),
        });
        info!(
            player = %conn.player_name,
            session = conn.session_token.0,
            "player joined galaxy shard"
        );
    }
}

fn drain_quic(
    mut bridge: ResMut<NetworkBridge>,
    mut handoff_events: MessageWriter<WarpHandoffEvent>,
    mut warp_ap_events: MessageWriter<WarpAutopilotEvent>,
    tick: Res<ecs::TickCounter>,
) {
    for _ in 0..32 {
        let queued = match bridge.quic_msg_rx.try_recv() {
            Ok(q) => q,
            Err(_) => break,
        };
        match queued.msg {
            ShardMsg::PlayerHandoff(h) => {
                let ship_shard_id = h.target_ship_shard_id;
                let source_shard = h.source_shard;
                handoff_events.write(WarpHandoffEvent {
                    handoff: h,
                    source_shard,
                    ship_shard_id,
                });
            }
            ShardMsg::HandoffAccepted(a) => {
                info!(
                    session = a.session_token.0,
                    target = a.target_shard.0,
                    "handoff accepted by destination system"
                );
            }
            ShardMsg::ShipControlInput(_ctrl) => {
                // During warp cruise the galaxy shard runs autonomous physics.
                // Pilot input is acknowledged but not applied (warp autopilot).
                // Future: allow manual course corrections.
            }
            ShardMsg::WarpAutopilotCommand(cmd) => {
                warp_ap_events.write(WarpAutopilotEvent {
                    ship_id: cmd.ship_id,
                    target_star_index: cmd.target_star_index,
                });
            }
            ShardMsg::AutopilotCommand(_) => {
                // Planet autopilot not supported during warp — ignore.
            }
            other => {
                if tick.0 % 100 == 0 {
                    warn!(
                        "galaxy shard received unexpected QUIC message: {:?}",
                        std::mem::discriminant(&other)
                    );
                }
            }
        }
    }
}

fn drain_provisions(
    mut rx: ResMut<ProvisionReceiver>,
    mut events: MessageWriter<ProvisionCompletedEvent>,
) {
    for _ in 0..8 {
        match rx.0.try_recv() {
            Ok(result) => {
                events.write(ProvisionCompletedEvent {
                    star_index: result.star_index,
                    shard_id: result.shard_id,
                });
            }
            Err(_) => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Spawn / process events
// ---------------------------------------------------------------------------

fn process_connect_events(
    mut events: MessageReader<ClientConnectedEvent>,
    galaxy_seed: Res<GalaxySeed>,
    galaxy_map: Res<GalaxyMapResource>,
    tick: Res<ecs::TickCounter>,
) {
    for event in events.read() {
        let token = event.session_token;
        let gs = galaxy_seed.0;
        let tcp_write = event.tcp_write.clone();
        // Compute game time from tick (approximation — the actual celestial_time
        // would come from epoch, but for JoinResponse it only needs to be close).
        let game_time = tick.0 as f64 * 0.05;

        // Snapshot the full star catalogue once per connect so the
        // client can spawn the starfield immediately. The galaxy map
        // is deterministic-from-seed, so a single TCP send at connect
        // time is authoritative — no per-tick re-send needed.
        let catalog = galaxy_catalog_message(gs, &galaxy_map);

        tokio::spawn(async move {
            let jr = ServerMsg::JoinResponse(JoinResponseData {
                seed: gs,
                planet_radius: 0,
                player_id: token.0,
                spawn_position: DVec3::ZERO,
                spawn_rotation: DQuat::IDENTITY,
                spawn_forward: DVec3::NEG_Z,
                session_token: token,
                shard_type: 3, // Galaxy
                galaxy_seed: gs,
                system_seed: 0,
                game_time,
                reference_position: DVec3::ZERO,
                reference_rotation: DQuat::IDENTITY,
            });
            let mut writer = tcp_write.lock().await;
            let _ = client_listener::send_tcp_msg(&mut *writer, &jr).await;
            let _ = client_listener::send_tcp_msg(&mut *writer, &catalog).await;
        });
    }
}

/// Build a `ServerMsg::StarCatalog` from the galaxy map. Called once
/// per client connect; the catalog is deterministic-from-seed and
/// immutable, so no per-tick updates are needed.
fn galaxy_catalog_message(
    galaxy_seed: u64,
    galaxy_map: &Res<GalaxyMapResource>,
) -> ServerMsg {
    use voxeldust_core::client_message::{StarCatalogData, StarCatalogEntryData};
    let stars = galaxy_map
        .0
        .stars
        .iter()
        .enumerate()
        .map(|(i, s)| StarCatalogEntryData {
            index: i as u32,
            position: s.position,
            system_seed: s.system_seed,
            star_class: s.star_class as u8,
            luminosity: s.luminosity as f32,
        })
        .collect();
    ServerMsg::StarCatalog(StarCatalogData {
        galaxy_seed,
        stars,
    })
}

fn spawn_warp_ships(
    mut commands: Commands,
    mut events: MessageReader<WarpHandoffEvent>,
    galaxy_map: Res<GalaxyMapResource>,
    mut ship_dir: ResMut<ShipShardDirectory>,
) {
    for event in events.read() {
        let h = &event.handoff;

        // Start at the departure star's galaxy position.
        let position_gu = if let Some(ref ctx) = h.galaxy_context {
            ctx.star_position
        } else {
            h.position
        };

        let origin_star = h
            .galaxy_context
            .as_ref()
            .map(|c| c.star_index)
            .unwrap_or(0);

        let target_star_index = h.warp_target_star_index.unwrap_or(0);

        // Target star position for distance tracking.
        let target_pos = galaxy_map
            .0
            .get_star(target_star_index)
            .map(|s| s.position)
            .unwrap_or(DVec3::ZERO);
        let initial_distance = (target_pos - position_gu).length();

        // Use the ship's actual rotation from the handoff.
        // WarpAlign in the system shard already aligned toward the target.
        let warp_rotation = h.rotation;

        let ship_id = h.session_token.0;

        // Track ship shard if provided.
        if let Some(ss_id) = event.ship_shard_id {
            // We don't have the QUIC addr yet — it'll be resolved from peer registry
            // during broadcast. Store with placeholder addr.
            ship_dir
                .0
                .entry(ship_id)
                .or_insert((ss_id, SocketAddr::from(([0, 0, 0, 0], 0))));
        }

        info!(
            player = %h.player_name,
            origin = origin_star,
            target = target_star_index,
            distance = format!("{:.0} GU", initial_distance),
            pos = format!(
                "({:.1},{:.1},{:.1})",
                position_gu.x, position_gu.y, position_gu.z
            ),
            "ship entered galaxy — warp cruise (ease-in from v=0)"
        );

        commands.spawn((
            ShipId(ship_id),
            PlayerId(h.session_token),
            PlayerName(h.player_name.clone()),
            Position(position_gu),
            Velocity(DVec3::ZERO), // Start from zero for smooth parallax ease-in
            Rotation(warp_rotation),
            WarpState {
                origin_star_index: origin_star,
                target_star_index,
                phase: FlightPhase::WarpCruise,
                initial_distance_gu: initial_distance,
                preconnect_sent: false,
            },
            SourceSystemShard(event.source_shard),
        ));
    }
}

fn process_warp_autopilot_events(
    mut events: MessageReader<WarpAutopilotEvent>,
    mut ships: Query<(&ShipId, &Position, &mut Velocity, &mut WarpState)>,
    galaxy_map: Res<GalaxyMapResource>,
) {
    for event in events.read() {
        for (ship_id, pos, mut vel, mut warp) in &mut ships {
            if ship_id.0 != event.ship_id {
                continue;
            }
            if event.target_star_index == 0xFFFFFFFF {
                // Disengage warp.
                warp.phase = FlightPhase::WarpArrival;
                vel.0 = DVec3::ZERO;
                info!(ship_id = event.ship_id, "warp disengaged by player");
            } else {
                // Change target star mid-flight.
                if let Some(target_star) = galaxy_map.0.get_star(event.target_star_index) {
                    let new_distance = (target_star.position - pos.0).length();
                    warp.target_star_index = event.target_star_index;
                    warp.phase = FlightPhase::WarpCruise;
                    warp.initial_distance_gu = new_distance;
                    warp.preconnect_sent = false;
                    info!(
                        ship_id = event.ship_id,
                        new_target = event.target_star_index,
                        "warp target changed mid-flight"
                    );
                }
            }
            break;
        }
    }
}

fn process_provision_completed(
    mut events: MessageReader<ProvisionCompletedEvent>,
    mut provisioned: ResMut<ProvisionedSystems>,
    mut in_flight: ResMut<ProvisioningInFlight>,
) {
    for event in events.read() {
        in_flight.0.remove(&event.star_index);
        provisioned.0.insert(event.star_index, event.shard_id);
        info!(
            star_index = event.star_index,
            shard_id = event.shard_id.0,
            "system shard provisioned for destination star"
        );
    }
}

// ---------------------------------------------------------------------------
// Physics
// ---------------------------------------------------------------------------

const DT: f64 = 0.05; // 20Hz tick

fn warp_physics(
    mut ships: Query<(&mut Position, &mut Velocity, &mut WarpState, &ShipId)>,
    galaxy_map: Res<GalaxyMapResource>,
) {
    for (mut pos, mut vel, mut warp, ship_id) in &mut ships {
        if warp.phase != FlightPhase::WarpCruise && warp.phase != FlightPhase::WarpDecelerate {
            continue;
        }

        let target_pos = match galaxy_map.0.get_star(warp.target_star_index) {
            Some(s) => s.position,
            None => continue,
        };

        let to_target = target_pos - pos.0;
        let distance = to_target.length();
        let direction = if distance > 1e-6 {
            to_target / distance
        } else {
            DVec3::NEG_Z
        };

        match warp.phase {
            FlightPhase::WarpCruise => {
                // Accelerate toward target.
                let accel = direction * WARP_ACCELERATION_GU;
                vel.0 += accel * DT;

                // Cap speed: max_safe_speed = sqrt(2*a*d) * 0.95 safety margin.
                // Ensures the ship can always stop before the target.
                let max_safe_speed = (distance * 2.0 * WARP_ACCELERATION_GU).sqrt() * 0.95;
                let effective_max = max_safe_speed.min(WARP_MAX_SPEED_GU);
                let speed = vel.0.length();
                if speed > effective_max {
                    vel.0 = vel.0.normalize() * effective_max;
                }
                let speed = vel.0.length();

                // Velocity Verlet position update.
                pos.0 += vel.0 * DT + 0.5 * accel * DT * DT;

                // Transition to braking when remaining distance <= braking distance.
                let braking_dist = speed * speed / (2.0 * WARP_ACCELERATION_GU);
                if distance < braking_dist {
                    warp.phase = FlightPhase::WarpDecelerate;
                    info!(
                        ship_id = ship_id.0,
                        distance = format!("{:.1} GU", distance),
                        speed = format!("{:.2} GU/s", speed),
                        braking = format!("{:.1} GU", braking_dist),
                        "warp deceleration started"
                    );
                }
            }
            FlightPhase::WarpDecelerate => {
                // Brake: decelerate toward zero velocity.
                let speed = vel.0.length();
                if speed > 0.01 {
                    let brake_dir = -vel.0.normalize();
                    let brake_accel = brake_dir * WARP_ACCELERATION_GU;
                    let new_vel = vel.0 + brake_accel * DT;
                    // Don't overshoot: if velocity direction reversed, stop.
                    if new_vel.dot(vel.0) < 0.0 {
                        vel.0 = DVec3::ZERO;
                    } else {
                        vel.0 = new_vel;
                    }
                }
                pos.0 += vel.0 * DT;
            }
            _ => {}
        }

        // Rotation: preserve the handoff rotation from WarpAlign.
        // The ship was already aligned toward the target in the system shard.
        // During warp travel the direction barely changes, so snapping rotation
        // every tick would cause visual jumps on the client.
    }
}

// ---------------------------------------------------------------------------
// SOI detection and arrival handoff
// ---------------------------------------------------------------------------

fn soi_detection(
    mut commands: Commands,
    ships: Query<(
        Entity,
        &ShipId,
        &PlayerId,
        &PlayerName,
        &Position,
        &Velocity,
        &Rotation,
        &WarpState,
        &SourceSystemShard,
    )>,
    galaxy_map: Res<GalaxyMapResource>,
    provisioned: Res<ProvisionedSystems>,
    mut in_flight: ResMut<ProvisioningInFlight>,
    shard_identity: Res<ShardIdentity>,
    galaxy_seed: Res<GalaxySeed>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
    provision_tx: Res<ProvisionSender>,
    orch_url: Res<OrchestratorUrl>,
    http_client: Res<HttpClient>,
    ship_dir: Res<ShipShardDirectory>,
) {
    // Run every ~1 second (20 ticks).
    if tick.0 % 20 != 0 {
        return;
    }

    for (entity, ship_id, player_id, player_name, pos, vel, rot, warp, origin_system) in &ships {
        let target_star = match galaxy_map.0.get_star(warp.target_star_index) {
            Some(s) => s,
            None => continue,
        };

        let distance = (pos.0 - target_star.position).length();
        let soi = system_outer_radius(target_star);
        let speed = vel.0.length();

        // "Effectively arrived": speed cap asymptotically approaches target but
        // never reaches it. Detect near-stop in WarpCruise.
        let effectively_arrived = speed < 2.0 && warp.phase == FlightPhase::WarpCruise;

        // --- Provision destination system shard when approaching ---
        let needs_provision =
            (distance < soi * 5.0 || effectively_arrived) && !warp.preconnect_sent;
        if needs_provision {
            if provisioned.0.contains_key(&warp.target_star_index) {
                // Already provisioned — handled in preconnect below.
            } else if !in_flight.0.contains(&warp.target_star_index) {
                // Kick off async provisioning.
                let url = format!(
                    "{}/system/{}?galaxy_seed={}&star_index={}",
                    orch_url.0, target_star.system_seed, galaxy_seed.0, warp.target_star_index
                );
                let tx = provision_tx.0.clone();
                let client = http_client.0.clone();
                let star_index = warp.target_star_index;
                in_flight.0.insert(star_index);
                tokio::spawn(async move {
                    match client.get(&url).send().await {
                        Ok(resp) => {
                            if let Ok(body) = resp.json::<serde_json::Value>().await {
                                let sid = body["id"].as_u64().unwrap_or(0);
                                if sid > 0 {
                                    let _ = tx
                                        .send(ProvisionResult {
                                            star_index,
                                            shard_id: ShardId(sid),
                                        })
                                        .await;
                                } else {
                                    warn!(star_index, "provision response missing shard id");
                                }
                            }
                        }
                        Err(e) => {
                            warn!(star_index, %e, "failed to provision system shard");
                        }
                    }
                });
                info!(
                    star_index = warp.target_star_index,
                    system_seed = target_star.system_seed,
                    "provisioning system shard for destination star"
                );
            }
        }

        // --- Send ShardPreConnect when destination is provisioned but not yet sent ---
        if needs_provision && !warp.preconnect_sent {
            if let Some(&dest_shard_id) = provisioned.0.get(&warp.target_star_index) {
                if let Ok(reg) = bridge.peer_registry.try_read() {
                    if let Some(peer_info) = reg.get(dest_shard_id) {
                        let pc = ServerMsg::ShardPreConnect(ShardPreConnect {
                            shard_type: 1,
                            tcp_addr: peer_info.endpoint.tcp_addr.to_string(),
                            udp_addr: peer_info.endpoint.udp_addr.to_string(),
                            seed: target_star.system_seed,
                            planet_index: 0,
                            reference_position: DVec3::ZERO,
                            reference_rotation: DQuat::IDENTITY,
                            shard_id: dest_shard_id.0,
                        });
                        let creg = bridge.client_registry.clone();
                        let token = player_id.0;
                        tokio::spawn(async move {
                            let r = creg.read().await;
                            let _ = r.send_tcp(token, &pc).await;
                        });
                        // Mark preconnect sent on the entity.
                        // We use commands because we only have immutable query access.
                        commands
                            .entity(entity)
                            .insert(PreconnectSent);
                        info!(ship_id = ship_id.0, "sent ShardPreConnect for destination system");
                    }
                }
            }
        }

        // --- Handoff when within SOI proximity or effectively arrived ---
        let ready_for_handoff =
            (distance < soi * 3.0 && speed < 1.0) || (effectively_arrived && speed < 0.5);
        if !ready_for_handoff {
            continue;
        }

        let dest_shard_id = match provisioned.0.get(&warp.target_star_index) {
            Some(&id) => id,
            None => continue,
        };

        // Build handoff. The galaxy shard doesn't know the destination system's
        // real size. Send the ship's forward direction as a hint — the destination
        // system shard computes the actual arrival position from its own SystemParams.
        //
        // Arrival velocity: seeded at a modest sub-orbital speed along the
        // forward direction rather than zero. Emerging from warp stationary
        // feels unphysical and loses momentum continuity. 200 m/s is below
        // escape velocity around a typical star — destination system physics
        // will shape the actual trajectory via gravity.
        let ship_forward = rot.0 * DVec3::NEG_Z;
        const ARRIVAL_SPEED_MPS: f64 = 200.0;
        let handoff_data = handoff::PlayerHandoff {
            session_token: player_id.0,
            player_name: player_name.0.clone(),
            position: ship_forward, // unit vector hint, not a position
            velocity: ship_forward * ARRIVAL_SPEED_MPS,
            rotation: rot.0,
            forward: ship_forward,
            fly_mode: false,
            speed_tier: 0,
            grounded: false,
            health: 100.0,
            shield: 100.0,
            source_shard: shard_identity.0,
            source_tick: tick.0,
            target_star_index: Some(warp.target_star_index),
            galaxy_context: Some(GalaxyHandoffContext {
                galaxy_seed: galaxy_seed.0,
                star_index: warp.target_star_index,
                star_position: target_star.position,
            }),
            target_planet_seed: None,
            target_planet_index: None,
            target_ship_id: None,
            target_ship_shard_id: None,
            ship_system_position: None,
            ship_rotation: None,
            game_time: tick.0 as f64 * DT,
            warp_target_star_index: None,
            warp_velocity_gu: None,
            target_system_eva: false,
            schema_version: 1,
            character_state: Vec::new(),
        };

        // Send PlayerHandoff to destination system shard.
        let handoff_sent = if let Ok(reg) = bridge.peer_registry.try_read() {
            if let Some(peer_info) = reg.get(dest_shard_id) {
                let _ = bridge.quic_send_tx.try_send((
                    dest_shard_id,
                    peer_info.endpoint.quic_addr,
                    ShardMsg::PlayerHandoff(handoff_data),
                ));
                true
            } else {
                warn!(
                    dest = dest_shard_id.0,
                    "destination system shard not in peer registry — handoff dropped"
                );
                false
            }
        } else {
            warn!("peer registry lock failed during warp arrival handoff");
            false
        };

        if !handoff_sent {
            continue; // Retry next detection tick.
        }

        // Send HostSwitch to ship shard — switch host from galaxy to destination.
        if let Some(&(ship_sid, _)) = ship_dir.0.get(&ship_id.0) {
            send_host_switch(
                ship_id.0,
                ship_sid,
                dest_shard_id,
                warp.target_star_index,
                &galaxy_map.0,
                &bridge,
            );
        }

        // Tell the client to tear down the ORIGIN system secondary — we've
        // arrived at the destination system and its secondary is live.
        // The origin system's "shrink-to-dot" window was during cruise; at
        // arrival we no longer need to stream its celestial bodies.
        //
        // The client's active_secondaries map is keyed by (shard_type, shard_id),
        // so we pass the origin system shard's id in the `seed` field (the
        // existing wire protocol reuses that field as the key value).
        let _ = galaxy_map.0.get_star(warp.origin_star_index); // sanity: origin star exists.
        let disconnect = ServerMsg::ShardDisconnectNotify(handoff::ShardDisconnectNotify {
            shard_type: 1, // System
            seed: origin_system.0.0,
        });
        {
            let creg = bridge.client_registry.clone();
            let token = player_id.0;
            tokio::spawn(async move {
                let r = creg.read().await;
                let _ = r.send_tcp(token, &disconnect).await;
            });
        }

        info!(
            player = %player_name.0,
            target = warp.target_star_index,
            "ship arrived at destination — handoff sent"
        );

        // Despawn the entity. apply_deferred ensures this is visible to subsequent systems.
        commands.entity(entity).despawn();
    }
}

/// Marker: ShardPreConnect has been sent for this ship's destination.
/// Applied by soi_detection via Commands, visible after apply_deferred.
#[derive(Component)]
struct PreconnectSent;

/// Sync PreconnectSent marker back to the WarpState component.
/// Runs after apply_deferred following soi_detection.
fn sync_preconnect_flag(
    mut ships: Query<&mut WarpState, Added<PreconnectSent>>,
) {
    for mut warp in &mut ships {
        warp.preconnect_sent = true;
    }
}

fn send_host_switch(
    ship_id: u64,
    ship_shard_id: ShardId,
    dest_shard_id: ShardId,
    target_star_index: u32,
    galaxy_map: &GalaxyMap,
    bridge: &NetworkBridge,
) {
    let Ok(reg) = bridge.peer_registry.try_read() else {
        return;
    };

    // Look up ship shard QUIC address.
    let ship_quic = reg.get(ship_shard_id).map(|s| s.endpoint.quic_addr);
    let dest_ep = reg.get(dest_shard_id);

    if let (Some(quic_addr), Some(dest_info)) = (ship_quic, dest_ep) {
        let dest_system_seed = galaxy_map
            .get_star(target_star_index)
            .map(|s| s.system_seed)
            .unwrap_or(0);
        let host_switch = ShardMsg::HostSwitch(voxeldust_core::shard_message::HostSwitchData {
            ship_id,
            new_host_shard_id: dest_shard_id,
            new_host_quic_addr: dest_info.endpoint.quic_addr.to_string(),
            new_host_tcp_addr: dest_info.endpoint.tcp_addr.to_string(),
            new_host_udp_addr: dest_info.endpoint.udp_addr.to_string(),
            new_host_shard_type: 1, // System
            seed: dest_system_seed,
        });
        let _ = bridge
            .quic_send_tx
            .try_send((ship_shard_id, quic_addr, host_switch));
        info!(
            ship_id,
            new_host = dest_shard_id.0,
            "sent HostSwitch to ship shard for warp arrival"
        );
    }
}

// ---------------------------------------------------------------------------
// Broadcast
// ---------------------------------------------------------------------------

fn update_ship_shard_directory(
    ships: Query<&ShipId>,
    bridge: Res<NetworkBridge>,
    mut ship_dir: ResMut<ShipShardDirectory>,
) {
    // Refresh ship shard addresses from peer registry (needed for QUIC sends).
    let Ok(reg) = bridge.peer_registry.try_read() else {
        return;
    };
    let ship_shards: Vec<_> = reg
        .find_by_type(ShardType::Ship)
        .iter()
        .map(|s| (s.id, s.endpoint.quic_addr, s.ship_id))
        .collect();

    for ship_id in &ships {
        let entry = ship_dir.0.get_mut(&ship_id.0);
        if let Some(entry) = entry {
            // Update QUIC address if we have it.
            if let Some((_, addr, _)) = ship_shards.iter().find(|(sid, _, _)| *sid == entry.0) {
                entry.1 = *addr;
            }
        } else {
            // Discover ship shard from peer registry.
            if let Some((sid, addr, _)) = ship_shards
                .iter()
                .find(|(_, _, sid)| *sid == Some(ship_id.0))
            {
                ship_dir.0.insert(ship_id.0, (*sid, *addr));
            }
        }
    }
}

fn broadcast_galaxy_world_state(
    ships: Query<(
        &ShipId,
        &Position,
        &Velocity,
        &Rotation,
        &WarpState,
    )>,
    galaxy_map: Res<GalaxyMapResource>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
) {
    for (_ship_id, pos, vel, _rot, warp) in &ships {
        let target_star = galaxy_map.0.get_star(warp.target_star_index);
        let distance = target_star
            .map(|s| (s.position - pos.0).length())
            .unwrap_or(0.0);
        let speed = vel.0.length();
        let eta = if speed > 0.01 { distance / speed } else { 0.0 };

        let gws = ServerMsg::GalaxyWorldState(GalaxyWorldStateData {
            tick: tick.0,
            ship_position: pos.0,
            ship_velocity: vel.0,
            ship_rotation: _rot.0,
            warp_phase: match warp.phase {
                FlightPhase::WarpCruise => 22,
                FlightPhase::WarpDecelerate => 23,
                FlightPhase::WarpArrival => 24,
                _ => 20,
            },
            eta_seconds: eta,
            origin_star_index: warp.origin_star_index,
            target_star_index: warp.target_star_index,
        });
        if bridge.broadcast_tx.try_send(gws).is_err() {
            tracing::warn!("GalaxyWorldState broadcast dropped — channel full");
        }
    }
}

fn broadcast_scene_to_ship_shards(
    ships: Query<(&ShipId, &Position, &Velocity, &Rotation, &WarpState)>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
    ship_dir: Res<ShipShardDirectory>,
) {
    for (ship_id, _pos, vel, rot, _warp) in &ships {
        let (shard_id, quic_addr) = match ship_dir.0.get(&ship_id.0) {
            Some(&(sid, addr)) if addr.port() != 0 => (sid, addr),
            _ => continue, // Ship shard not yet discovered
        };

        // Scene update: empty bodies (client renders stars via StarField in galaxy mode).
        // Only lighting is sent for ambient interstellar illumination.
        let scene = SystemSceneUpdateData {
            game_time: tick.0 as f64 * DT,
            bodies: vec![],
            ships: vec![],
            lighting: LightingInfoData {
                sun_direction: DVec3::new(0.0, -1.0, 0.0),
                sun_color: [0.5, 0.5, 0.6],
                sun_intensity: 0.2,
                ambient: 0.08,
            },
        };

        let _ = bridge
            .quic_send_tx
            .try_send((shard_id, quic_addr, ShardMsg::SystemSceneUpdate(scene)));

        // Per-ship position update for the ship shard's WorldState.
        let pos_update = ShardMsg::ShipPositionUpdate(ShipPositionUpdate {
            ship_id: ship_id.0,
            position: DVec3::ZERO,
            velocity: vel.0 * GALAXY_UNIT_IN_BLOCKS,
            rotation: rot.0,
            angular_velocity: DVec3::ZERO,
            autopilot: None,
            in_atmosphere: false,
            atmosphere_planet_index: -1,
            gravity_acceleration: DVec3::ZERO,
            atmosphere_density: 0.0,
        });
        let _ = bridge
            .quic_send_tx
            .try_send((shard_id, quic_addr, pos_update));
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

fn tick_counter(mut tick: ResMut<ecs::TickCounter>) {
    tick.0 += 1;
}

fn log_state(
    ships: Query<(&ShipId, &PlayerName, &Position, &Velocity, &WarpState)>,
    tick: Res<ecs::TickCounter>,
    provisioned: Res<ProvisionedSystems>,
) {
    if tick.0 % 100 != 0 || tick.0 == 0 {
        return;
    }

    let ship_count = ships.iter().count();
    info!(
        tick = tick.0,
        ships = ship_count,
        provisioned = provisioned.0.len(),
        "galaxy shard status"
    );
    for (ship_id, name, pos, vel, warp) in &ships {
        info!(
            ship_id = ship_id.0,
            player = %name.0,
            phase = ?warp.phase,
            speed = format!("{:.2} GU/s", vel.0.length()),
            pos = format!(
                "({:.1},{:.1},{:.1})",
                pos.0.x, pos.0.y, pos.0.z
            ),
            "warp ship status"
        );
    }
}

// ---------------------------------------------------------------------------
// App construction
// ---------------------------------------------------------------------------

fn build_app(
    galaxy_seed: u64,
    shard_id: ShardId,
    orchestrator_url: String,
) -> App {
    let galaxy_map = GalaxyMap::generate(galaxy_seed);
    info!(
        galaxy_seed,
        star_count = galaxy_map.stars.len(),
        "galaxy map generated"
    );

    let (provision_tx, provision_rx) = mpsc::channel(32);

    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("failed to build HTTP client");

    let mut app = App::new();

    // Resources.
    app.insert_resource(GalaxyMapResource(galaxy_map));
    app.insert_resource(GalaxySeed(galaxy_seed));
    app.insert_resource(ShardIdentity(shard_id));
    app.insert_resource(ProvisionedSystems::default());
    app.insert_resource(ProvisioningInFlight::default());
    app.insert_resource(ProvisionReceiver(provision_rx));
    app.insert_resource(ProvisionSender(provision_tx));
    app.insert_resource(OrchestratorUrl(orchestrator_url));
    app.insert_resource(HttpClient(http_client));
    app.insert_resource(ShipShardDirectory::default());
    app.insert_resource(ecs::TickCounter::default());

    // Events.
    app.add_message::<ClientConnectedEvent>();
    app.add_message::<WarpHandoffEvent>();
    app.add_message::<WarpAutopilotEvent>();
    app.add_message::<ProvisionCompletedEvent>();

    // System ordering via SystemSets.
    app.configure_sets(
        Update,
        (
            GalaxySet::Bridge,
            GalaxySet::Spawn,
            GalaxySet::Physics,
            GalaxySet::Detection,
            GalaxySet::Broadcast,
            GalaxySet::Diagnostics,
        )
            .chain(),
    );

    // Bridge systems: drain async channels → ECS events.
    app.add_systems(
        Update,
        (drain_connects, drain_quic, drain_provisions).in_set(GalaxySet::Bridge),
    );

    // Spawn/process: convert events into entities and component updates.
    app.add_systems(
        Update,
        (
            process_connect_events,
            spawn_warp_ships,
            process_warp_autopilot_events,
            process_provision_completed,
        )
            .in_set(GalaxySet::Spawn),
    );

    // apply_deferred between Spawn and Physics so newly spawned entities are queryable.
    app.add_systems(
        Update,
        bevy_ecs::schedule::ApplyDeferred
            .after(GalaxySet::Spawn)
            .before(GalaxySet::Physics),
    );

    // Physics: warp acceleration/deceleration.
    app.add_systems(Update, warp_physics.in_set(GalaxySet::Physics));

    // Detection: SOI detection, provisioning, handoff.
    app.add_systems(Update, soi_detection.in_set(GalaxySet::Detection));

    // apply_deferred between Detection and Broadcast so despawned entities
    // and PreconnectSent markers are applied.
    app.add_systems(
        Update,
        (
            bevy_ecs::schedule::ApplyDeferred,
            sync_preconnect_flag,
        )
            .chain()
            .after(GalaxySet::Detection)
            .before(GalaxySet::Broadcast),
    );

    // Broadcast: send state to clients and ship shards.
    app.add_systems(
        Update,
        (
            update_ship_shard_directory,
            broadcast_galaxy_world_state,
            broadcast_scene_to_ship_shards,
        )
            .chain()
            .in_set(GalaxySet::Broadcast),
    );

    // Diagnostics: tick counter + periodic logging.
    app.add_systems(
        Update,
        (tick_counter, log_state)
            .chain()
            .in_set(GalaxySet::Diagnostics),
    );

    app
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let shard_id = ShardId(args.shard_id);
    let galaxy_seed = args.seed;
    let bind = "0.0.0.0";
    let config = ShardHarnessConfig {
        shard_id,
        shard_type: ShardType::Galaxy,
        tcp_addr: format!("{bind}:{}", args.tcp_port).parse().unwrap(),
        udp_addr: format!("{bind}:{}", args.udp_port).parse().unwrap(),
        quic_addr: format!("{bind}:{}", args.quic_port).parse().unwrap(),
        orchestrator_url: args.orchestrator.clone(),
        orchestrator_heartbeat_addr: args.orchestrator_heartbeat,
        healthz_addr: format!("{bind}:{}", args.healthz_port).parse().unwrap(),
        planet_seed: None,
        system_seed: None,
        ship_id: None,
        galaxy_seed: Some(galaxy_seed),
        host_shard_id: None,
        advertise_host: args.advertise_host,
    };

    info!(shard_id = args.shard_id, galaxy_seed, "galaxy shard starting");

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let harness = ShardHarness::new(config);
        let app = build_app(galaxy_seed, shard_id, args.orchestrator);

        info!("galaxy shard ECS app built, starting harness");
        harness.run_ecs(app).await;
    });
}
