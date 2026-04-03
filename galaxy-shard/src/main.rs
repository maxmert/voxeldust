use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use clap::Parser;
use glam::{DQuat, DVec3};
use tracing::{info, warn};

use voxeldust_core::autopilot::{FlightPhase, WARP_ACCELERATION_GU, WARP_MAX_SPEED_GU};
use voxeldust_core::client_message::{
    GalaxyWorldStateData, JoinResponseData, ServerMsg,
};
use voxeldust_core::galaxy::{
    galaxy_to_system, system_outer_radius, GalaxyMap, GALAXY_UNIT_IN_BLOCKS,
};
use voxeldust_core::handoff::{self, GalaxyHandoffContext, ShardPreConnect};
use voxeldust_core::shard_message::ShardMsg;
use voxeldust_core::shard_types::{SessionToken, ShardId, ShardType};
use voxeldust_shard_common::client_listener;
use voxeldust_shard_common::harness::{ShardHarness, ShardHarnessConfig};

#[derive(Parser, Debug)]
#[command(name = "galaxy-shard", about = "Voxeldust galaxy shard — interstellar warp travel")]
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

/// Ship traveling through interstellar space during warp.
#[derive(Clone)]
struct WarpShipState {
    ship_id: u64,
    session_token: SessionToken,
    player_name: String,
    position_gu: DVec3,
    velocity_gu: DVec3,
    rotation: DQuat,
    origin_star_index: u32,
    target_star_index: u32,
    warp_phase: FlightPhase,
    source_system_shard: ShardId,
    /// Whether ShardPreConnect has been sent for the destination system.
    preconnect_sent: bool,
}

/// Galaxy shard state. Manages ships traveling between star systems.
struct GalaxyState {
    galaxy_seed: u64,
    galaxy_map: GalaxyMap,
    shard_id: ShardId,
    ships: HashMap<u64, WarpShipState>,
    tick_count: u64,
    celestial_time: f64,
    /// System shards already provisioned: star_index → shard_id.
    provisioned_systems: HashMap<u32, ShardId>,
    /// Star indices currently being provisioned (in-flight HTTP requests).
    provisioning_in_flight: HashSet<u32>,
}

impl GalaxyState {
    fn new(galaxy_seed: u64, shard_id: ShardId) -> Self {
        let galaxy_map = GalaxyMap::generate(galaxy_seed);
        info!(
            galaxy_seed,
            star_count = galaxy_map.stars.len(),
            "galaxy map generated"
        );
        Self {
            galaxy_seed,
            galaxy_map,
            shard_id,
            ships: HashMap::new(),
            tick_count: 0,
            celestial_time: 0.0,
            provisioned_systems: HashMap::new(),
            provisioning_in_flight: HashSet::new(),
        }
    }
}

/// Physics workspace: data extracted from GalaxyState for lock-free computation.
struct GalaxyPhysicsWorkspace {
    ships: HashMap<u64, WarpShipState>,
    tick_count: u64,
}

const DT: f64 = 0.05; // 20Hz tick

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
        let mut harness = ShardHarness::new(config);

        let state = Arc::new(Mutex::new(GalaxyState::new(galaxy_seed, shard_id)));
        let client_registry = harness.client_registry.clone();
        let orchestrator_url = args.orchestrator.clone();

        // Take channels.
        let mut connect_rx = std::mem::replace(
            &mut harness.connect_rx,
            tokio::sync::mpsc::unbounded_channel().1,
        );
        let mut quic_msg_rx = std::mem::replace(
            &mut harness.quic_msg_rx,
            tokio::sync::mpsc::unbounded_channel().1,
        );
        let broadcast_tx = harness.broadcast_tx.clone();
        let quic_send_tx = harness.quic_send_tx.clone();
        let peer_registry = harness.peer_registry.clone();
        let universe_epoch = harness.epoch_arc();

        // Channel for async system shard provisioning results.
        let (provision_tx, mut provision_rx) =
            tokio::sync::mpsc::channel::<(u32, ShardId, SocketAddr, SocketAddr)>(32);

        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("failed to build HTTP client");

        // ------------------------------------------------------------------
        // System 1: Drain client TCP connections.
        // ------------------------------------------------------------------
        let state_connect = state.clone();
        let registry_connect = client_registry.clone();
        harness.add_system("drain_connects", move || {
            for _ in 0..16 {
                let event = match connect_rx.try_recv() {
                    Ok(e) => e,
                    Err(_) => break,
                };
                let conn = event.connection;
                let st = state_connect.lock().unwrap();
                let token = conn.session_token;

                if let Ok(mut reg) = registry_connect.try_write() {
                    reg.register(&conn);
                }

                let tcp_stream = conn.tcp_stream.clone();
                let game_time = st.celestial_time;
                let gs = st.galaxy_seed;
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
                    let mut stream = tcp_stream.lock().await;
                    let _ = client_listener::send_tcp_msg(&mut stream, &jr).await;
                });

                info!(
                    player = %conn.player_name,
                    session = token.0,
                    "player joined galaxy shard"
                );
            }
        });

        // ------------------------------------------------------------------
        // System 2: Drain QUIC messages — PlayerHandoff from system shards.
        // ------------------------------------------------------------------
        let state_quic = state.clone();
        harness.add_system("drain_quic", move || {
            for _ in 0..32 {
                let msg = match quic_msg_rx.try_recv() {
                    Ok(m) => m,
                    Err(_) => break,
                };
                match msg {
                    ShardMsg::PlayerHandoff(h) => {
                        let mut st = state_quic.lock().unwrap();

                        // Extract warp context.
                        let target_star = h.warp_target_star_index.unwrap_or(0);
                        let velocity_gu = h.warp_velocity_gu.unwrap_or(DVec3::ZERO);

                        // Convert position to GU using galaxy context.
                        let position_gu = if let Some(ref ctx) = h.galaxy_context {
                            voxeldust_core::galaxy::system_to_galaxy(
                                ctx.star_position,
                                h.position,
                            )
                        } else {
                            // Fallback: position is already in GU.
                            h.position
                        };

                        let origin_star = h
                            .galaxy_context
                            .as_ref()
                            .map(|c| c.star_index)
                            .unwrap_or(0);

                        let ship = WarpShipState {
                            ship_id: h.session_token.0, // Use session as ship ID in galaxy
                            session_token: h.session_token,
                            player_name: h.player_name.clone(),
                            position_gu,
                            velocity_gu,
                            rotation: h.rotation,
                            origin_star_index: origin_star,
                            target_star_index: target_star,
                            warp_phase: FlightPhase::WarpCruise,
                            source_system_shard: h.source_shard,
                            preconnect_sent: false,
                        };

                        info!(
                            player = %h.player_name,
                            origin = origin_star,
                            target = target_star,
                            pos = format!("({:.1},{:.1},{:.1})", position_gu.x, position_gu.y, position_gu.z),
                            "ship entered galaxy — warp cruise"
                        );

                        st.ships.insert(ship.ship_id, ship);
                    }
                    ShardMsg::HandoffAccepted(a) => {
                        info!(
                            session = a.session_token.0,
                            target = a.target_shard.0,
                            "handoff accepted by destination system"
                        );
                    }
                    other => {
                        let st = state_quic.lock().unwrap();
                        if st.tick_count % 100 == 0 {
                            warn!("galaxy shard received unexpected QUIC message: {:?}",
                                  std::mem::discriminant(&other));
                        }
                    }
                }
            }
        });

        // ------------------------------------------------------------------
        // System 3: Drain provisioning results.
        // ------------------------------------------------------------------
        let state_provision = state.clone();
        harness.add_system("drain_provisions", move || {
            for _ in 0..8 {
                let (star_index, shard_id, _tcp_addr, _udp_addr) =
                    match provision_rx.try_recv() {
                        Ok(r) => r,
                        Err(_) => break,
                    };
                let mut st = state_provision.lock().unwrap();
                st.provisioning_in_flight.remove(&star_index);
                st.provisioned_systems.insert(star_index, shard_id);
                info!(star_index, shard_id = shard_id.0, "system shard provisioned for destination star");
            }
        });

        // ------------------------------------------------------------------
        // System 4: Warp physics — extract-compute-writeback.
        // ------------------------------------------------------------------
        let state_physics = state.clone();
        let galaxy_map_arc = Arc::new(GalaxyMap::generate(galaxy_seed));
        harness.add_system("warp_physics", move || {
            let mut st = state_physics.lock().unwrap();

            // Extract.
            let mut ws = GalaxyPhysicsWorkspace {
                ships: st.ships.clone(),
                tick_count: st.tick_count,
            };

            // Compute (no lock held).
            drop(st);

            for ship in ws.ships.values_mut() {
                if ship.warp_phase != FlightPhase::WarpCruise
                    && ship.warp_phase != FlightPhase::WarpDecelerate
                {
                    continue;
                }

                let target_star = galaxy_map_arc.get_star(ship.target_star_index);
                let target_pos = match target_star {
                    Some(s) => s.position,
                    None => continue,
                };

                let to_target = target_pos - ship.position_gu;
                let distance = to_target.length();
                let direction = if distance > 1e-6 {
                    to_target / distance
                } else {
                    DVec3::NEG_Z
                };

                let target_soi = target_star
                    .map(|s| system_outer_radius(s))
                    .unwrap_or(100.0);

                match ship.warp_phase {
                    FlightPhase::WarpCruise => {
                        // Accelerate toward target, clamped to max speed.
                        let accel = direction * WARP_ACCELERATION_GU;
                        ship.velocity_gu += accel * DT;
                        let speed = ship.velocity_gu.length();
                        if speed > WARP_MAX_SPEED_GU {
                            ship.velocity_gu =
                                ship.velocity_gu.normalize() * WARP_MAX_SPEED_GU;
                        }
                        ship.position_gu +=
                            ship.velocity_gu * DT + 0.5 * accel * DT * DT;

                        // Transition to decelerate when within braking distance.
                        // Braking distance = v² / (2*a).
                        let braking_dist =
                            speed * speed / (2.0 * WARP_ACCELERATION_GU);
                        if distance < braking_dist + target_soi * 2.0 {
                            ship.warp_phase = FlightPhase::WarpDecelerate;
                            info!(
                                ship_id = ship.ship_id,
                                distance,
                                speed,
                                "warp deceleration started"
                            );
                        }
                    }
                    FlightPhase::WarpDecelerate => {
                        // Brake: decelerate toward zero velocity relative to target.
                        let speed = ship.velocity_gu.length();
                        if speed > 0.01 {
                            let brake_dir = -ship.velocity_gu.normalize();
                            let brake_accel = brake_dir * WARP_ACCELERATION_GU;
                            let new_vel = ship.velocity_gu + brake_accel * DT;
                            // Don't overshoot: if velocity direction reversed, stop.
                            if new_vel.dot(ship.velocity_gu) < 0.0 {
                                ship.velocity_gu = DVec3::ZERO;
                            } else {
                                ship.velocity_gu = new_vel;
                            }
                        }
                        ship.position_gu += ship.velocity_gu * DT;
                    }
                    _ => {}
                }

                // Update rotation to face travel direction.
                if ship.velocity_gu.length_squared() > 0.01 {
                    let fwd = ship.velocity_gu.normalize();
                    let up = DVec3::Y;
                    let right = fwd.cross(up).normalize();
                    let corrected_up = right.cross(fwd).normalize();
                    // Build rotation from basis vectors (ship faces -Z in local space).
                    ship.rotation = DQuat::from_mat3(&glam::DMat3::from_cols(
                        right,
                        corrected_up,
                        -fwd,
                    ));
                }
            }

            // Writeback.
            let mut st = state_physics.lock().unwrap();
            st.ships = ws.ships;
            st.tick_count += 1;

            // Update celestial time from epoch.
            let epoch_ms = st
                .celestial_time; // Will be overwritten below from epoch.
            drop(st);
        });

        // ------------------------------------------------------------------
        // System 5: SOI detection — check if ships entered destination system.
        // ------------------------------------------------------------------
        let state_soi = state.clone();
        let galaxy_map_soi = Arc::new(GalaxyMap::generate(galaxy_seed));
        let orch_url_soi = orchestrator_url.clone();
        let http_soi = http_client.clone();
        let provision_tx_soi = provision_tx.clone();
        let quic_send_soi = quic_send_tx.clone();
        let peer_reg_soi = peer_registry.clone();
        let client_reg_soi = client_registry.clone();
        harness.add_system("soi_detection", move || {
            let mut st = state_soi.lock().unwrap();
            if st.tick_count % 20 != 0 {
                return; // Run every ~1 second.
            }

            // Phase 1: Collect actions needed (immutable scan).
            enum SoiAction {
                SendPreConnect { ship_id: u64, dest_shard: ShardId, system_seed: u64, token: SessionToken },
                Provision { star_index: u32, system_seed: u64 },
                Handoff { ship_id: u64, dest_shard: ShardId, system_pos: DVec3, system_vel: DVec3,
                          token: SessionToken, player_name: String, rotation: DQuat,
                          target_star_index: u32, star_position: DVec3 },
            }
            let mut actions = Vec::new();
            let mut arrived_ships = Vec::new();

            for ship in st.ships.values() {
                let target_star = match galaxy_map_soi.get_star(ship.target_star_index) {
                    Some(s) => s,
                    None => continue,
                };

                let distance = (ship.position_gu - target_star.position).length();
                let soi = system_outer_radius(target_star);

                // PreConnect when approaching (2× SOI).
                if distance < soi * 2.0 && !ship.preconnect_sent {
                    if let Some(&dest_shard_id) = st.provisioned_systems.get(&ship.target_star_index) {
                        actions.push(SoiAction::SendPreConnect {
                            ship_id: ship.ship_id,
                            dest_shard: dest_shard_id,
                            system_seed: target_star.system_seed,
                            token: ship.session_token,
                        });
                    } else if !st.provisioning_in_flight.contains(&ship.target_star_index) {
                        actions.push(SoiAction::Provision {
                            star_index: ship.target_star_index,
                            system_seed: target_star.system_seed,
                        });
                    }
                }

                // Ship entered SOI — handoff.
                if distance < soi && ship.velocity_gu.length() < 1.0 {
                    if let Some(&dest_shard_id) = st.provisioned_systems.get(&ship.target_star_index) {
                        let system_pos = galaxy_to_system(target_star.position, ship.position_gu);
                        let system_vel = ship.velocity_gu * GALAXY_UNIT_IN_BLOCKS;
                        actions.push(SoiAction::Handoff {
                            ship_id: ship.ship_id,
                            dest_shard: dest_shard_id,
                            system_pos,
                            system_vel,
                            token: ship.session_token,
                            player_name: ship.player_name.clone(),
                            rotation: ship.rotation,
                            target_star_index: ship.target_star_index,
                            star_position: target_star.position,
                        });
                        arrived_ships.push(ship.ship_id);
                    }
                }
            }

            // Phase 2: Apply actions (mutable).
            let shard_id = st.shard_id;
            let galaxy_seed_local = st.galaxy_seed;
            let celestial_time = st.celestial_time;
            let tick_count = st.tick_count;

            for action in actions {
                match action {
                    SoiAction::SendPreConnect { ship_id, dest_shard, system_seed, token } => {
                        if let Ok(reg) = peer_reg_soi.try_read() {
                            if let Some(peer_info) = reg.get(dest_shard) {
                                let pc = ServerMsg::ShardPreConnect(ShardPreConnect {
                                    shard_type: 1,
                                    tcp_addr: peer_info.endpoint.tcp_addr.to_string(),
                                    udp_addr: peer_info.endpoint.udp_addr.to_string(),
                                    seed: system_seed,
                                    planet_index: 0,
                                    reference_position: DVec3::ZERO,
                                    reference_rotation: DQuat::IDENTITY,
                                });
                                let creg = client_reg_soi.clone();
                                tokio::spawn(async move {
                                    let r = creg.read().await;
                                    let _ = r.send_tcp(token, &pc).await;
                                });
                                if let Some(ship) = st.ships.get_mut(&ship_id) {
                                    ship.preconnect_sent = true;
                                }
                                info!(ship_id, "sent ShardPreConnect for destination system");
                            }
                        }
                    }
                    SoiAction::Provision { star_index, system_seed } => {
                        let url = format!("{}/system/{}", orch_url_soi, system_seed);
                        let tx = provision_tx_soi.clone();
                        let client = http_soi.clone();
                        st.provisioning_in_flight.insert(star_index);
                        tokio::spawn(async move {
                            match client.get(&url).send().await {
                                Ok(resp) => {
                                    if let Ok(body) = resp.json::<serde_json::Value>().await {
                                        if let Some(info) = body.get("info") {
                                            let sid = info["id"].as_u64().unwrap_or(0);
                                            let tcp: SocketAddr = info["endpoint"]["tcp_addr"]
                                                .as_str().unwrap_or("127.0.0.1:9777")
                                                .parse().unwrap_or_else(|_| "127.0.0.1:9777".parse().unwrap());
                                            let udp: SocketAddr = info["endpoint"]["udp_addr"]
                                                .as_str().unwrap_or("127.0.0.1:9778")
                                                .parse().unwrap_or_else(|_| "127.0.0.1:9778".parse().unwrap());
                                            let _ = tx.send((star_index, ShardId(sid), tcp, udp)).await;
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!(star_index, %e, "failed to provision system shard");
                                }
                            }
                        });
                        info!(star_index, system_seed, "provisioning system shard for destination star");
                    }
                    SoiAction::Handoff { ship_id, dest_shard, system_pos, system_vel,
                                         token, player_name, rotation, target_star_index, star_position } => {
                        let handoff = handoff::PlayerHandoff {
                            session_token: token,
                            player_name: player_name.clone(),
                            position: system_pos,
                            velocity: system_vel,
                            rotation,
                            forward: rotation * DVec3::NEG_Z,
                            fly_mode: false,
                            speed_tier: 0,
                            grounded: false,
                            health: 100.0,
                            shield: 100.0,
                            source_shard: shard_id,
                            source_tick: tick_count,
                            target_star_index: Some(target_star_index),
                            galaxy_context: Some(GalaxyHandoffContext {
                                galaxy_seed: galaxy_seed_local,
                                star_index: target_star_index,
                                star_position,
                            }),
                            target_planet_seed: None,
                            target_planet_index: None,
                            target_ship_id: None,
                            target_ship_shard_id: None,
                            ship_system_position: None,
                            ship_rotation: None,
                            game_time: celestial_time,
                            warp_target_star_index: None,
                            warp_velocity_gu: None,
                        };

                        if let Ok(reg) = peer_reg_soi.try_read() {
                            if let Some(peer_info) = reg.get(dest_shard) {
                                let _ = quic_send_soi.try_send((
                                    dest_shard, peer_info.endpoint.quic_addr,
                                    ShardMsg::PlayerHandoff(handoff),
                                ));
                            }
                        }

                        // Send ShardRedirect to client.
                        if let Ok(reg) = peer_reg_soi.try_read() {
                            if let Some(peer_info) = reg.get(dest_shard) {
                                let redirect = ServerMsg::ShardRedirect(handoff::ShardRedirect {
                                    session_token: token,
                                    target_tcp_addr: peer_info.endpoint.tcp_addr.to_string(),
                                    target_udp_addr: peer_info.endpoint.udp_addr.to_string(),
                                    shard_id: dest_shard,
                                });
                                let creg = client_reg_soi.clone();
                                tokio::spawn(async move {
                                    let r = creg.read().await;
                                    let _ = r.send_tcp(token, &redirect).await;
                                });
                            }
                        }

                        info!(player = %player_name, target = target_star_index,
                              "ship arrived at destination — handoff sent");
                    }
                }
            }

            for id in arrived_ships {
                st.ships.remove(&id);
            }
        });

        // ------------------------------------------------------------------
        // System 6: Broadcast GalaxyWorldState to connected clients.
        // ------------------------------------------------------------------
        let state_broadcast = state.clone();
        let galaxy_map_bc = Arc::new(GalaxyMap::generate(galaxy_seed));
        harness.add_system("broadcast_udp", move || {
            let st = state_broadcast.lock().unwrap();
            if st.ships.is_empty() {
                return;
            }

            // For each connected ship, send its galaxy world state.
            for ship in st.ships.values() {
                let target_star = galaxy_map_bc.get_star(ship.target_star_index);
                let distance = target_star
                    .map(|s| (s.position - ship.position_gu).length())
                    .unwrap_or(0.0);
                let speed = ship.velocity_gu.length();
                let eta = if speed > 0.01 {
                    distance / speed
                } else {
                    0.0
                };

                let gws = ServerMsg::GalaxyWorldState(GalaxyWorldStateData {
                    tick: st.tick_count,
                    ship_position: ship.position_gu,
                    ship_velocity: ship.velocity_gu,
                    ship_rotation: ship.rotation,
                    warp_phase: match ship.warp_phase {
                        FlightPhase::WarpCruise => 22,
                        FlightPhase::WarpDecelerate => 23,
                        FlightPhase::WarpArrival => 24,
                        _ => 20,
                    },
                    eta_seconds: eta,
                    origin_star_index: ship.origin_star_index,
                    target_star_index: ship.target_star_index,
                });
                let _ = broadcast_tx.try_send(gws);
            }
        });

        // ------------------------------------------------------------------
        // System 7: Periodic logging.
        // ------------------------------------------------------------------
        let state_log = state.clone();
        harness.add_system("log_state", move || {
            let st = state_log.lock().unwrap();
            if st.tick_count % 100 == 0 && st.tick_count > 0 {
                info!(
                    tick = st.tick_count,
                    ships = st.ships.len(),
                    provisioned = st.provisioned_systems.len(),
                    "galaxy shard status"
                );
                for ship in st.ships.values() {
                    info!(
                        ship_id = ship.ship_id,
                        player = %ship.player_name,
                        phase = ?ship.warp_phase,
                        speed = format!("{:.2} GU/s", ship.velocity_gu.length()),
                        pos = format!("({:.1},{:.1},{:.1})", ship.position_gu.x, ship.position_gu.y, ship.position_gu.z),
                        "warp ship status"
                    );
                }
            }
        });

        // ------------------------------------------------------------------
        // Run the harness tick loop.
        // ------------------------------------------------------------------
        info!("galaxy shard systems registered, starting tick loop");
        harness.run().await;
    });
}
