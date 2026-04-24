//! voxydust-next — Bevy-powered voxeldust client.
//!
//! Migration plan: `/Users/maxim/.claude/plans/bevy-render-migration.md`.
//!
//! This phase (4 in the plan) brings up Bevy's production-grade atmosphere
//! scattering on top of the Phase-1 scaffold:
//!
//! - `Atmosphere::earthlike()` with a `ScatteringMedium` default — Hillaire
//!   2020 LUTs (transmittance, multiscatter, sky-view, aerial-perspective)
//!   managed by `bevy_pbr::AtmospherePlugin`. Replaces ALL the handwritten
//!   atmosphere code we've been porting piece by piece.
//! - `AtmosphereEnvironmentMapLight` — the atmosphere drives IBL (ambient
//!   lighting + specular probe) so shadow sides of objects pick up realistic
//!   sky colour.
//! - `VolumetricFog` + `VolumetricLight` — light shafts / god rays when the
//!   sun shines through the atmosphere.
//! - `ScreenSpaceReflections` — reflective surfaces pick up the sky.
//! - Free-fly camera (WASD + mouse look) so we can actually walk through the
//!   atmosphere and verify limb glow, horizon gradient, dawn/dusk.
//!
//! Scene scale: 1 Bevy unit = 1 kilometre. Earth bottom_radius = 6 360 km.
//! Camera and scene geometry live near origin; atmosphere spheres are far
//! below under y ≈ -6 360 000 m (handled by the scene-units-to-m conversion).

use std::f32::consts::{FRAC_PI_2, PI};

use bevy::{
    anti_alias::fxaa::Fxaa,
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    input::mouse::MouseMotion,
    light::{
        light_consts::lux, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder,
        DirectionalLightShadowMap, GlobalAmbientLight,
    },
    pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium, ScreenSpaceReflections},
    post_process::bloom::Bloom,
    prelude::*,
    render::view::Hdr,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow, WindowResolution},
};
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};

mod block_highlight;
mod block_interaction;
mod block_raycast;
mod camera_frame;
mod celestial;
mod chunk_stream;
mod chunks;
mod input_system;
mod net_plugin;
mod network;
mod player_sync;
mod remote_entities;
mod seat_bindings;
mod shard_origin;
mod shard_transition;
mod stars;
mod weather;
use block_highlight::BlockHighlightPlugin;
use block_interaction::BlockInteractionPlugin;
use block_raycast::BlockRaycastPlugin;
use camera_frame::CameraFramePlugin;
use celestial::CelestialPlugin;
use chunk_stream::ChunkStreamPlugin;
use chunks::TerrainDemoPlugin;
use input_system::InputPlugin;
use net_plugin::{GameEvent, NetworkPlugin};
use player_sync::{NetworkedPlayerCamera, PlayerSyncPlugin};
use remote_entities::RemoteEntitiesPlugin;
use seat_bindings::SeatBindingsPlugin;
use shard_origin::ShardOriginPlugin;
use shard_transition::ShardTransitionPlugin;
use stars::StarfieldPlugin;
use weather::{WeatherPlugin, WeatherState};

#[derive(clap::Parser, Debug, Clone)]
#[command(name = "voxydust-next", about = "Bevy-powered voxeldust client")]
struct Cli {
    /// Gateway address.
    #[arg(long, default_value = "127.0.0.1:7777")]
    gateway: std::net::SocketAddr,

    /// Display name announced to the server.
    #[arg(long, default_value = "Player")]
    name: String,

    /// Offline mode — skip the network connection and just render the
    /// procedural demo scene. Useful when no dev-cluster is running.
    #[arg(long)]
    offline: bool,
}

fn main() {
    use clap::Parser;
    let cli = Cli::parse();

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "voxydust-next".into(),
            resolution: WindowResolution::new(1280, 720),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(EguiPlugin::default())
    .add_plugins(FrameTimeDiagnosticsPlugin::default())
    .insert_resource(DirectionalLightShadowMap { size: 4096 })
    .insert_resource(ClearColor(Color::BLACK))
    .add_systems(Startup, grab_cursor)
    .add_systems(Update, toggle_cursor)
    .add_systems(EguiPrimaryContextPass, hud_system);

    if cli.offline {
        // Offline mode: full demo stack (procedural terrain, water, stars,
        // weather, local sun). None of this runs in networked mode.
        tracing::info!("voxydust-next starting in offline mode");
        app.add_plugins(StarfieldPlugin)
            .add_plugins(WeatherPlugin)
            .add_plugins(TerrainDemoPlugin)
            .insert_resource(GlobalAmbientLight::NONE)
            .add_systems(Startup, (setup_camera, setup_scene, setup_sun))
            .add_systems(Update, (advance_sun, camera_free_fly));
    } else {
        // Networked mode: **everything** scene-side is server-authoritative.
        // No demo geometry, no procedural atmosphere, no local sun, no
        // starfield (galaxy_seed → star catalog is Phase 4 work), no
        // client-side weather. The scene contains exactly what the shards
        // stream: chunks via ChunkStreamPlugin, camera pose via
        // PlayerSyncPlugin, sun direction/intensity via network_lighting.
        app.add_plugins(NetworkPlugin {
            gateway: cli.gateway,
            player_name: cli.name.clone(),
        })
        // Register ShardTransitionPlugin BEFORE any subsystem that reads
        // its resources (ChunkStreamPlugin, PlayerSyncPlugin): Bevy
        // resolves `init_resource` on plugin build, so a consumer whose
        // build runs first would observe the resource as missing and
        // default-construct it separately, desynchronising state.
        .add_plugins(ShardTransitionPlugin)
        .add_plugins(InputPlugin)
        .add_plugins(SeatBindingsPlugin)
        .add_plugins(CameraFramePlugin)
        .add_plugins(ChunkStreamPlugin)
        .add_plugins(BlockRaycastPlugin)
        .add_plugins(BlockHighlightPlugin)
        .add_plugins(BlockInteractionPlugin)
        .add_plugins(PlayerSyncPlugin)
        .add_plugins(CelestialPlugin)
        .add_plugins(RemoteEntitiesPlugin)
        .add_plugins(ShardOriginPlugin)
        // Interior ambient: 800 lux (bright-office equivalent). Legacy
        // uses voxel-light-propagation for emissive-block illumination; we
        // stand in with a constant term until Phase 6 ports that compute
        // pass. Paired with the camera's ev100 = 11, surfaces render as
        // readable mid-grey rather than blown-out white.
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.9, 0.95, 1.0),
            brightness: 800.0,
            affects_lightmapped_meshes: true,
        })
        .add_systems(Startup, (setup_camera_networked, setup_sun_networked))
        .add_systems(Update, (log_network_events, network_lighting));
    }

    app.run();
}

/// Sun for networked mode: shadows on, illuminance off (0) by default —
/// the `network_lighting` system below drives its direction + intensity
/// from `WorldState.lighting` every frame. When the server emits `None`
/// lighting (ship interior, pre-connection), the sun stays off.
fn setup_sun_networked(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: 0.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 10.0, 0.0).looking_at(Vec3::NEG_Y, Vec3::Z),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 20.0,
            maximum_distance: 1200.0,
            num_cascades: 4,
            ..default()
        }
        .build(),
        Sun,
    ));
}

/// Drive the directional light from `WorldState.lighting`. Server-authoritative:
/// direction, intensity, colour all come from the shard's view of the local
/// star. On ship-interior / deep-space shards where `ws.lighting` is `None`,
/// we zero the sun intensity.
fn network_lighting(
    mut events: MessageReader<GameEvent>,
    mut q: Query<(&mut DirectionalLight, &mut Transform), With<Sun>>,
) {
    for GameEvent(ev) in events.read() {
        if let network::NetEvent::WorldState(ws) = ev {
            let Ok((mut light, mut tf)) = q.single_mut() else { continue };
            if let Some(lighting) = &ws.lighting {
                let dir = glam::DVec3::new(
                    lighting.sun_direction.x,
                    lighting.sun_direction.y,
                    lighting.sun_direction.z,
                )
                .normalize();
                let world_dir = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32);
                // Bevy's DirectionalLight direction = transform's -Z.
                // Point it at the sun so "-Z = toward the sun" -> camera at
                // origin looking along -world_dir gives the right shading.
                *tf = Transform::from_xyz(0.0, 0.0, 0.0).looking_to(-world_dir, Vec3::Y);
                // Map `lighting.intensity` (0..1-ish multiplier on raw sun)
                // into lux. RAW_SUNLIGHT ≈ 75000 lux but we're getting a
                // post-atmospheric value from the server, so 25k lux is the
                // "bright noon overcast" equivalent that reads correctly
                // without atmosphere IBL in the picture.
                let intensity = lighting.sun_intensity as f32;
                light.illuminance = (25_000.0 * intensity).max(0.0);
                light.color = Color::srgb(
                    lighting.sun_color[0],
                    lighting.sun_color[1],
                    lighting.sun_color[2],
                );
            } else {
                light.illuminance = 0.0;
            }
        }
    }
}

/// Development-time observer: prints key network events to stdout so we can
/// verify the bridge from CLI without a HUD overlay. Uses the raw NetEvent
/// forwarded inside `GameEvent(NetEvent)` — no semantic translation here,
/// so every protocol event (including secondary shards, promotions, galaxy
/// state) is observable by the ECS.
fn log_network_events(mut reader: MessageReader<GameEvent>) {
    use network::NetEvent;
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
                tracing::info!(%shard_type, %seed, "net: secondary pre-connect established");
            }
            NetEvent::SecondaryDisconnected { seed } => {
                tracing::info!(%seed, "net: secondary disconnected");
            }
            NetEvent::WorldState(ws) => {
                // Rate-limit worldstate logs to every ~100 ticks (~5 s).
                static TICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n % 100 == 0 {
                    tracing::info!(
                        players = ws.players.len(),
                        bodies = ws.bodies.len(),
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
                tracing::info!(len = cs.data.len(), "net: chunk snapshot (primary)");
            }
            NetEvent::SecondaryChunkSnapshot { seed, data } => {
                tracing::info!(%seed, len = data.data.len(), "net: chunk snapshot (secondary)");
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Marker on the player camera entity.
#[derive(Component)]
struct PlayerCamera;

/// Simple free-fly state — yaw/pitch accumulators for mouse look.
#[derive(Component, Default)]
struct FreeFly {
    yaw: f32,
    pitch: f32,
    /// Movement speed in scene units per second (= km/s at this scale).
    speed: f32,
}

/// Marker on the directional sun so we can rotate it every frame.
#[derive(Component)]
struct Sun;

// ---------------------------------------------------------------------------
// Startup systems
// ---------------------------------------------------------------------------

/// Camera setup used in **offline** mode — demo planet surface, atmosphere,
/// volumetric-ish scene. Networked mode uses `setup_camera_networked`
/// instead, which is space-ready (no atmosphere, no IBL from a sky we
/// don't have).
/// **Offline** camera — demo planet scene, uses Bevy's atmosphere stack.
/// Networked mode uses `setup_camera_networked` which is space-ready and
/// never attaches planet-atmosphere bits; an atmosphere is re-attached
/// dynamically if/when the client detects a nearby atmosphere-bearing planet
/// (legacy voxydust's `find_atmosphere_planet` equivalent — Phase 4+ work).
fn setup_camera(mut commands: Commands, mut mediums: ResMut<Assets<ScatteringMedium>>) {
    let medium = mediums.add(ScatteringMedium::default());

    commands.spawn((
        Camera3d::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AgX,
        Atmosphere::earthlike(medium),
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight::default(),
        Bloom::NATURAL,
        ScreenSpaceReflections::default(),
        Fxaa::default(),
        Transform::from_xyz(0.0, 40.0, 60.0).looking_at(Vec3::new(0.0, 10.0, 0.0), Vec3::Y),
        PlayerCamera,
        NetworkedPlayerCamera,
        FreeFly {
            yaw: 0.0,
            pitch: -0.35,
            speed: 20.0,
        },
    ));
}

/// **Networked** camera. No planet-assuming components; pure
/// post-processing stack with legacy-voxydust-matched exposure.
fn setup_camera_networked(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Hdr,
        // ev100 = 11 — halfway between legacy's ev100 = 13 (calibrated for
        // RAW_SUNLIGHT + voxel light propagation) and the "too dark without
        // voxel light" floor. Keeps interiors readable under modest ambient.
        Exposure { ev100: 11.0 },
        Tonemapping::AgX,
        Bloom::NATURAL,
        Fxaa::default(),
        Transform::from_xyz(0.0, 0.0, 0.0),
        PlayerCamera,
        NetworkedPlayerCamera,
    ));
}

fn setup_sun(mut commands: Commands) {
    // Shadow cascades tuned for terrain-scale scene (1 Bevy unit = 1 metre).
    // Cascade layout covers close detail → horizon gradient:
    //   cascade 0: 0 → 20 m    (characters, local blocks)
    //   cascade 1: 20 → 80 m   (nearby terrain)
    //   cascade 2: 80 → 300 m  (mid-field)
    //   cascade 3: 300 → 1200 m (horizon)
    let cascade_config = CascadeShadowConfigBuilder {
        first_cascade_far_bound: 20.0,
        maximum_distance: 1200.0,
        num_cascades: 4,
        ..default()
    }
    .build();

    // DirectionalLight with RAW_SUNLIGHT illuminance — this is the correct
    // input to atmospheric scattering because the atmosphere itself does the
    // post-scattering dimming.
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        // VolumetricLight removed alongside the fog slab — it only does
        // anything inside a FogVolume.
        Transform::from_xyz(0.6, 0.7, 0.3).looking_at(Vec3::ZERO, Vec3::Y),
        cascade_config,
        Sun,
    ));

    // FogVolume removed alongside VolumetricFog on the camera.
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Water surface. Spans the full terrain grid at the voxel water line.
    //
    // Z-fighting fix: the terrain generator in `chunks.rs` places sand
    // blocks that extend up to y=8 at the shoreline. If the water plane
    // sits at exactly y=8 it competes with those sand block tops for depth
    // and produces the flickering-shore artifact. Dropping the plane to
    // y=7.5 places it firmly inside the water volume (below any solid
    // block at the shore) while still being above the stone lakebed.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(2000.0, 2000.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.04, 0.08, 0.11, 1.0),
            perceptual_roughness: 0.02,
            metallic: 0.0,
            reflectance: 0.5,
            ..default()
        })),
        Transform::from_xyz(0.0, 7.5, 0.0),
    ));

    // Landmark sphere mesh reused across several spawns.
    let sphere = meshes.add(Sphere::new(1.0).mesh().ico(5).unwrap());

    // Hero sphere — sits a few metres in front of the camera origin so we
    // can immediately verify PBR shading under atmospheric lighting.
    commands.spawn((
        Mesh3d(sphere.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.22, 0.24),
            perceptual_roughness: 0.35,
            ..default()
        })),
        Transform::from_xyz(0.0, 32.0, 0.0),
    ));

    // Emissive beacon — exercises Bloom. 5-metre-radius sphere on a hilltop
    // glowing brightly enough to visibly bloom against the sky.
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(5.0).mesh().ico(4).unwrap())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.9, 0.7),
            emissive: LinearRgba::rgb(40.0, 32.0, 24.0),
            ..default()
        })),
        Transform::from_xyz(80.0, 55.0, -70.0),
    ));

    // Chrome sphere — shows atmosphere IBL reflection + screen-space reflections.
    commands.spawn((
        Mesh3d(sphere.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.98, 0.98, 0.98),
            perceptual_roughness: 0.04,
            metallic: 1.0,
            ..default()
        })),
        Transform::from_xyz(-30.0, 34.0, -20.0)
            .with_scale(Vec3::splat(3.0)),
    ));

    // Distant sphere near the far horizon — aerial perspective should
    // desaturate it into the horizon colour.
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(15.0).mesh().ico(5).unwrap())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.55, 0.5),
            perceptual_roughness: 0.7,
            ..default()
        })),
        Transform::from_xyz(180.0, 45.0, 180.0),
    ));
}

// ---------------------------------------------------------------------------
// Interactive systems
// ---------------------------------------------------------------------------

fn grab_cursor(mut cursors: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    if let Ok(mut cursor) = cursors.single_mut() {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
}

fn toggle_cursor(
    keys: Res<ButtonInput<KeyCode>>,
    mut cursors: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        if let Ok(mut cursor) = cursors.single_mut() {
            let locked = cursor.grab_mode != CursorGrabMode::None;
            cursor.grab_mode = if locked {
                CursorGrabMode::None
            } else {
                CursorGrabMode::Locked
            };
            cursor.visible = locked;
        }
    }
}

fn camera_free_fly(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut mouse: MessageReader<MouseMotion>,
    cursors: Query<&CursorOptions, With<PrimaryWindow>>,
    mut q: Query<(&mut Transform, &mut FreeFly), With<PlayerCamera>>,
) {
    let Ok((mut tf, mut fly)) = q.single_mut() else { return };
    let Ok(cursor) = cursors.single() else { return };

    // Only read mouse when cursor is grabbed — otherwise user is in a menu.
    if cursor.grab_mode != CursorGrabMode::None {
        let sensitivity = 0.003;
        for ev in mouse.read() {
            fly.yaw -= ev.delta.x * sensitivity;
            fly.pitch -= ev.delta.y * sensitivity;
        }
        fly.pitch = fly.pitch.clamp(-FRAC_PI_2 * 0.98, FRAC_PI_2 * 0.98);
    } else {
        mouse.clear();
    }
    tf.rotation = Quat::from_axis_angle(Vec3::Y, fly.yaw)
        * Quat::from_axis_angle(Vec3::X, fly.pitch);

    // Movement in the local frame. Shift = sprint (10× speed).
    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) { dir += *tf.forward(); }
    if keys.pressed(KeyCode::KeyS) { dir -= *tf.forward(); }
    if keys.pressed(KeyCode::KeyD) { dir += *tf.right(); }
    if keys.pressed(KeyCode::KeyA) { dir -= *tf.right(); }
    if keys.pressed(KeyCode::Space) { dir += Vec3::Y; }
    if keys.pressed(KeyCode::ControlLeft) { dir -= Vec3::Y; }
    let speed = if keys.pressed(KeyCode::ShiftLeft) { fly.speed * 20.0 } else { fly.speed };
    let step = dir.normalize_or_zero() * speed * time.delta_secs();
    tf.translation += step;
}

/// Debug HUD — FPS, camera, sun, exposure readout. Same `egui` crate we've
/// always used; the integration path is bevy_egui instead of the handwritten
/// egui_wgpu glue that the old voxydust carries.
fn hud_system(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    cams: Query<(&Transform, &Exposure), With<PlayerCamera>>,
    sun: Query<&Transform, (With<Sun>, Without<PlayerCamera>)>,
    // Weather is an offline-mode-only resource. Wrap in Option so the
    // networked build (which doesn't add WeatherPlugin) still renders HUD.
    weather: Option<ResMut<WeatherState>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("voxydust-next")
        .default_pos([8.0, 8.0])
        .resizable(false)
        .show(ctx, |ui| {
            if let Some(fps) = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
            {
                ui.label(format!("FPS: {:.0}", fps));
            }
            if let Ok((tf, exp)) = cams.single() {
                let p = tf.translation;
                ui.label(format!("pos: ({:.2}, {:.2}, {:.2}) m", p.x, p.y, p.z));
                ui.label(format!("exposure ev100: {:.2}", exp.ev100));
            }
            if let Ok(sun_tf) = sun.single() {
                let f = sun_tf.forward();
                let alt = f.y.asin().to_degrees();
                ui.label(format!("sun altitude: {:.1}°", alt));
            }
            if let Some(mut weather) = weather {
                ui.separator();
                ui.label("Weather");
                ui.add(
                    egui::Slider::new(&mut weather.precip, 0.0..=1.0)
                        .text("precipitation")
                        .clamp_to_range(true),
                );
                ui.horizontal(|ui| {
                    ui.label("wind");
                    ui.add(egui::DragValue::new(&mut weather.wind.x).speed(0.1));
                    ui.add(egui::DragValue::new(&mut weather.wind.z).speed(0.1));
                });
            }
            ui.separator();
            ui.label("WASD + mouse: fly");
            ui.label("Space/Ctrl: up/down");
            ui.label("Shift: sprint");
            ui.label("T: fast-forward sun");
            ui.label("Esc: toggle cursor");
        });
    Ok(())
}

fn advance_sun(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut q: Query<&mut Transform, With<Sun>>,
) {
    // Hold `T` to fast-forward the sun through a full day — lets us verify
    // sunrise / noon / sunset / twilight / night behaviour interactively.
    let speed = if keys.pressed(KeyCode::KeyT) { 0.5 } else { 0.02 };
    let Ok(mut tf) = q.single_mut() else { return };
    tf.rotate_around(
        Vec3::ZERO,
        Quat::from_axis_angle(Vec3::Z, -speed * time.delta_secs()),
    );
    // Keep a very gentle azimuthal drift so the sun isn't on a rail.
    tf.rotate_around(
        Vec3::ZERO,
        Quat::from_axis_angle(Vec3::Y, 0.01 * time.delta_secs() * PI),
    );
}
