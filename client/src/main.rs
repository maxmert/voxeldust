//! voxeldust bevy client — phase 3: shard registry + plugin trait.
//!
//! Boots Bevy + bevy_egui, brings up `NetworkPlugin` (Phase 2),
//! `ShardRegistryPlugin` + `WorldStateIngestPlugin` (Phase 3). Concrete
//! shard-type plugins (SHIP, SYSTEM, PLANET, GALAXY) land in Phases 6-9;
//! until then the registry logs `"no plugin for shard_type=N"` for every
//! shard the server streams — expected behavior, proves the plumbing.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    render::view::Hdr,
    window::WindowResolution,
};
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use clap::Parser;

mod camera;
mod chunk;
mod config;
mod config_panel;
mod focus;
mod hud;
mod input;
mod interaction;
mod kernel;
mod lighting;
mod net;
mod remote;
mod seat;
mod shard;
mod shard_types;
mod subgrid;

use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use camera::PlayerSyncPlugin;
use chunk::{ChunkStorageCache, ChunkStreamPlugin};
use input::InputPlugin;
use interaction::{BlockHighlightPlugin, BlockRaycastPlugin, BlockTarget, InteractionPlugin};
use kernel::cli::Cli;
use net::{
    GameEvent, NetConnection, NetEvent, NetSecondaries, NetworkBridgeSet, NetworkPlugin,
};
use seat::SeatBindingsPlugin;
use shard::{
    CameraWorldPos, PrimaryShard, Secondaries, ShardOrigin, ShardOriginPlugin,
    ShardRegistryPlugin, ShardTransitionPlugin, ShardTypeRegistry, WorldStateIngestPlugin,
};

fn main() {
    // Install tracing-subscriber so the network task's structured logs
    // actually render; Bevy's bevy_log plugin only configures its own
    // default subscriber, but we want the tokio side's info! calls too.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    tracing::info!(gateway = %cli.gateway, name = %cli.name, "client starting");

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "voxeldust".into(),
                    resolution: WindowResolution::new(1280, 720),
                    ..default()
                }),
                ..default()
            })
            // Bevy ships its own log subscriber; disable it so
            // tracing-subscriber above owns stdout.
            .disable::<bevy::log::LogPlugin>(),
    )
    // Disable ambient entirely. Space has no atmosphere → no sky
    // bounce → no ambient term. The ONLY illumination should come
    // from the directional sun (and, future: in-ship interior
    // lights / emissive blocks). We MUST insert this with
    // `brightness: 0.0` because Bevy's default is `brightness: 80.0`
    // — without our explicit override, that default kicks in and
    // would wash the scene with 80 cd/m² of fake ambient fill,
    // exactly the "uniformly highlighted" look you reported.
    .insert_resource(bevy::light::GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 0.0,
        affects_lightmapped_meshes: true,
    })
    .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.04)))
    .add_plugins(EguiPlugin::default())
    .add_plugins(FrameTimeDiagnosticsPlugin::default())
    // GameConfig registers first — every later plugin reads from it.
    .add_plugins(config::GameConfigPlugin)
    .add_plugins(NetworkPlugin {
        gateway: cli.gateway,
        player_name: cli.name.clone(),
    })
    .add_plugins(ShardRegistryPlugin)
    .add_plugins(ShardTransitionPlugin)
    .add_plugins(WorldStateIngestPlugin)
    .add_plugins(PlayerSyncPlugin)
    .add_plugins(ShardOriginPlugin)
    .add_plugins(ChunkStreamPlugin)
    .add_plugins(SeatBindingsPlugin)
    .add_plugins(InputPlugin)
    .add_plugins(BlockRaycastPlugin)
    .add_plugins(BlockHighlightPlugin)
    .add_plugins(InteractionPlugin)
    .add_plugins(subgrid::SubGridPlugin)
    .add_plugins(remote::RemoteEntitiesPlugin)
    .add_plugins(config_panel::ConfigPanelPlugin)
    .add_plugins(focus::FocusInteractionPlugin)
    .add_plugins(hud::HudPlugin)
    .add_plugins(lighting::SolarLightPlugin)
    .init_resource::<config_panel::PendingConfigShard>();

    // Concrete shard-type plugins — each registers into ShardTypeRegistry
    // via its own Bevy plugin. Adding a future shard-type (DEBRIS,
    // STATION, …) is a one-line addition in `shard_types::register_all`.
    shard_types::register_all(&mut app);

    app.insert_resource(ClearColor(Color::BLACK))
        .insert_resource(cli)
        .add_systems(Startup, (setup_camera, grab_cursor))
        .add_systems(
            Update,
            (
                log_network_events.after(NetworkBridgeSet),
                toggle_cursor_grab,
            ),
        )
        .add_systems(EguiPrimaryContextPass, boot_hud)
        .run();
}

/// Grab cursor on startup so mouse-look captures by default. Esc
/// releases. Phase 21's InputMode state machine will upgrade this to
/// a proper Game / UiPanel / Animating / MenuFocus state machine.
fn grab_cursor(mut cursors: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    if let Ok(mut c) = cursors.single_mut() {
        c.grab_mode = CursorGrabMode::Locked;
        c.visible = false;
    }
}

fn toggle_cursor_grab(
    keys: Res<bevy::prelude::ButtonInput<KeyCode>>,
    mut cursors: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        if let Ok(mut c) = cursors.single_mut() {
            let locked = c.grab_mode != CursorGrabMode::None;
            c.grab_mode = if locked {
                CursorGrabMode::None
            } else {
                CursorGrabMode::Locked
            };
            c.visible = locked;
        }
    }
}

#[derive(Component)]
pub struct MainCamera;

fn setup_camera(mut commands: Commands) {
    // Camera stays at identity. Phase 4's ShardOriginPlugin rebases every
    // shard's ChunkSource relative to the camera each frame; the camera
    // itself never moves in Bevy coordinates.
    // Far-plane at 2 × 10⁶ m lets stars + celestial bodies at their
    // far-field clamp radius (1 × 10⁶ m) render without being culled
    // by the default `far = 1000` plane. Near plane stays tight
    // (0.05 m) so the held tablet at 0.52 m doesn't have z-fighting.
    let projection = Projection::Perspective(PerspectiveProjection {
        near: 0.05,
        far: 2.0e6,
        ..default()
    });
    commands.spawn((
        Camera3d::default(),
        projection,
        // Diagnostic baseline: HDR + AgX disabled so we can verify
        // the directional light is actually being applied. With LDR
        // rendering, default exposure, default ambient (80 cd/m²),
        // and ~10 000 lux directional sun, lit faces should roll up
        // toward ~1.0 (clipped to white) and shaded faces stay near
        // mid-gray (~0.5). If THIS shows visible lit/shaded
        // contrast on the ship, the issue is that AgX + the
        // 100 000 lux sun was producing rolloff that compressed all
        // faces into a similar bright value. If it does NOT show
        // contrast, the directional light is failing to reach the
        // chunks (material setup, normals, or a missing render
        // component).
        //
        // Re-add Hdr + Tonemapping::AgX after we confirm the light
        // is working.
        Transform::IDENTITY,
        MainCamera,
    ));
}

/// Phase-1 boot HUD with a connection-status line from Phase 2.
fn boot_hud(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    cli: Res<Cli>,
    conn: Res<NetConnection>,
    secondaries: Res<NetSecondaries>,
    primary: Res<PrimaryShard>,
    shard_registry: Res<ShardTypeRegistry>,
    shard_secondaries: Res<Secondaries>,
    source_index: Res<crate::shard::registry::SourceIndex>,
    camera_world: Res<CameraWorldPos>,
    shard_origins: Query<&ShardOrigin>,
    storage: Res<ChunkStorageCache>,
    primary_ws_res: Res<crate::shard::PrimaryWorldState>,
    sub_block_tool: Res<crate::interaction::dispatch::SubBlockTool>,
    hud_focus: Res<crate::hud::HudFocusState>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("client")
        .default_pos([8.0, 8.0])
        .resizable(false)
        .show(ctx, |ui| {
            if let Some(fps) = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
            {
                ui.label(format!("FPS: {:.0}", fps));
            }
            ui.label(format!("gateway: {}", cli.gateway));
            ui.label(format!("name: {}", cli.name));
            ui.separator();
            if conn.connected {
                ui.label(format!(
                    "connected: shard_type={} seed=0x{:x} player={}",
                    conn.shard_type, conn.seed, conn.player_id
                ));
            } else {
                ui.label("disconnected");
            }
            ui.label(format!("status: {}", conn.last_status));
            ui.label(format!(
                "net-level secondaries: {}",
                secondaries.active.len()
            ));
            ui.separator();
            ui.label(format!(
                "primary runtime: {}",
                primary
                    .current
                    .as_ref()
                    .map(|k| k.to_string())
                    .unwrap_or_else(|| "none".into()),
            ));
            ui.label(format!(
                "secondary runtimes: {}",
                shard_secondaries.runtimes.len()
            ));
            let reg_names: Vec<String> = shard_registry
                .names()
                .into_iter()
                .map(|(t, n)| format!("{}={}", t, n))
                .collect();
            ui.label(format!(
                "registered plugins: [{}]",
                if reg_names.is_empty() {
                    "none".to_string()
                } else {
                    reg_names.join(", ")
                }
            ));
            ui.separator();
            ui.label(format!(
                "camera_world: ({:+.1}, {:+.1}, {:+.1})",
                camera_world.pos.x, camera_world.pos.y, camera_world.pos.z
            ));
            let origin_count = shard_origins.iter().count();
            ui.label(format!("shard origins tracked: {}", origin_count));
            ui.label(format!("chunks cached: {}", storage.entries.len()));
            // Primary shard-type HUD summary via ShardTypePlugin trait.
            if let Some(primary_key) = primary.current {
                if let Some(&entity) = source_index.by_shard.get(&primary_key) {
                    if let Some(plugin) = shard_registry.get(primary_key.shard_type) {
                        let shard_origin = shard_origins
                            .get(entity)
                            .map(|o| o.origin)
                            .unwrap_or_default();
                        let ctx = crate::shard::plugin::HudSummaryCtx {
                            shard_origin,
                            primary_ws: primary_ws_res.latest.as_ref(),
                            camera_world: camera_world.pos,
                        };
                        let summary = plugin.hud_summary(&ctx);
                        if !summary.is_empty() {
                            ui.separator();
                            ui.label(format!("— {} —", plugin.name()));
                            for (k, v) in &summary {
                                ui.label(format!("{}: {}", k, v));
                            }
                        }
                    }
                }
            }
            ui.separator();
            if sub_block_tool.active {
                let ty_label = match sub_block_tool.element_type {
                    0 => "PowerWire",
                    1 => "SignalWire",
                    10 => "Rail",
                    30 => "Pipe",
                    40 => "Ladder",
                    42 => "SurfaceLight",
                    43 => "Cable",
                    50 => "Bracket",
                    60 => "HudPanel",
                    _ => "unknown",
                };
                ui.label(format!(
                    "SUB-BLOCK TOOL ACTIVE — T to exit  rot={}  type={} ({})",
                    sub_block_tool.rotation, sub_block_tool.element_type, ty_label,
                ));
                ui.label("1-9: select type    RMB: place    LMB: remove    R: rotate");
            } else {
                ui.label("T: sub-block tool    F: tablet    E: interact");
            }
            if hud_focus.focused_tile.is_some() {
                if hud_focus.active {
                    ui.label("TABLET CURSOR: ON — LMB clicks • Tab → free-move");
                } else {
                    ui.label("TABLET CURSOR: OFF (free-move) — Tab → cursor mode");
                }
            }
        });
    Ok(())
}

/// Development-only observer logging key NetEvents. Production consumers
/// will land in Phase 3 (shard registry) and later.
fn log_network_events(mut reader: MessageReader<GameEvent>) {
    for GameEvent(ev) in reader.read() {
        match ev {
            NetEvent::Connected { shard_type, seed, player_id, .. } => {
                tracing::info!(%shard_type, %seed, %player_id, "net: connected");
            }
            NetEvent::Disconnected(reason) => tracing::warn!(%reason, "net: disconnected"),
            NetEvent::Transitioning { target_shard_type, .. } => {
                tracing::info!(%target_shard_type, "net: shard promotion in flight");
            }
            NetEvent::SecondaryConnected { shard_type, seed, .. } => {
                tracing::info!(%shard_type, %seed, "net: secondary pre-connect");
            }
            NetEvent::SecondaryDisconnected { seed } => {
                tracing::info!(%seed, "net: secondary disconnected");
            }
            NetEvent::WorldState(ws) => {
                // Rate-limit: one line per 100 ticks (~5 s at 20 Hz).
                static TICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n % 100 == 0 {
                    tracing::info!(
                        bodies = ws.bodies.len(),
                        entities = ws.entities.len(),
                        game_time = ws.game_time,
                        "net: worldstate tick"
                    );
                }
            }
            NetEvent::SecondaryWorldState { shard_type, .. } => {
                static TICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n % 200 == 0 {
                    tracing::info!(%shard_type, "net: secondary worldstate tick");
                }
            }
            NetEvent::ChunkSnapshot(cs) => {
                tracing::debug!(len = cs.data.len(), "net: chunk snapshot");
            }
            NetEvent::SecondaryChunkSnapshot { seed, data } => {
                tracing::debug!(%seed, len = data.data.len(), "net: secondary chunk snapshot");
            }
            _ => {}
        }
    }
}
