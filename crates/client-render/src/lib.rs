//! `vd-client-render` (Tier-B, P1.5 Slice 3 T4) — the windowed Bevy client.
//!
//! This is the ONLY place wgpu/winit/Bevy live. It is a pure CONSUMER of the
//! renderer-free core: it reads an immutable [`RenderSnapshot`] the core thread
//! publishes lock-free (via `ArcSwap`) and writes injected [`InputAction`]s into the
//! SAME bounded mailbox `vdctl` feeds — so a windowed keypress is byte-identical to a
//! dev-control command, and the sim/netcode runs at a deterministic 20 Hz on its own
//! thread while this window interpolates smoothly at the display refresh rate.
//!
//! No prediction: poses come from [`RenderSnapshot::rendered`] (delivered + interpolated,
//! never extrapolated). Mouse-look turns the LOCAL camera immediately (a view-only
//! response); the rendered ENTITY orientation stays server-delivered. ALL branching
//! logic lives in `vd-client-harness` (Tier-A); this crate is the glue, proven by the
//! human-in-the-loop window + (T6) `G-RENDER-SMOKE`, not by `llvm-cov`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::SyncSender;
use std::time::Instant;

use arc_swap::ArcSwap;
use bevy::app::AppExit;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use vd_client::net::ClientPhase;
use vd_client::render_snapshot::RenderSnapshot;
use vd_client::view::world_pos;
use vd_client_harness::camera::FollowCamera;
use vd_client_harness::input_map::{MovementKeys, mouse_look};
use vd_core::EntityId;
use vd_core::glam::DVec3;
use vd_devproto::InputAction;

// ---- render tuning (named consts; no inline magic numbers) ----------------------
const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;
const DOT_RADIUS: f32 = 0.5;
/// Reference-scene extents so motion is VISIBLE in the empty stub world (P1.5 has no
/// terrain): a ground plate + a ring of distinct landmark pillars for parallax. Pure
/// render scaffolding — replaced wholesale by real terrain at P4, never extended.
const GROUND_HALF: f32 = 250.0;
const LANDMARK_RING_RADIUS: f32 = 25.0;
const LANDMARK_HEIGHT: f32 = 6.0;
const LANDMARK_COLORS: [Color; 8] = [
    Color::srgb(0.9, 0.3, 0.3),
    Color::srgb(0.9, 0.6, 0.2),
    Color::srgb(0.9, 0.9, 0.2),
    Color::srgb(0.3, 0.9, 0.3),
    Color::srgb(0.2, 0.8, 0.8),
    Color::srgb(0.3, 0.5, 0.95),
    Color::srgb(0.6, 0.3, 0.9),
    Color::srgb(0.9, 0.3, 0.7),
];

/// The handles the bin wires into the window (constructed on the main thread before
/// `run_window`). Every field is `Send + Sync`, so the Bevy resource is plain.
pub struct RenderHandles {
    /// The lock-free render snapshot, published every step by the core thread.
    pub snapshot: Arc<ArcSwap<RenderSnapshot>>,
    /// The input mailbox — the SAME `SyncSender` the dev-control listener feeds.
    pub input: SyncSender<InputAction>,
    /// The SHARED shed-command counter (`DevState::dev_commands_dropped`): the window
    /// bumps it on a dropped input, so BOTH producers feed one observable back-pressure
    /// metric (the "back-pressure is never silent" rule).
    pub dropped: Arc<AtomicU64>,
    /// Liveness flag the core thread flips false when its loop EXITS (gateway Close /
    /// panic). The window emits `AppExit` when it sees false — bidirectional shutdown.
    pub core_alive: Arc<AtomicBool>,
    /// The shared monotonic epoch the core anchored its render clock on (so this
    /// thread computes the display cursor in the same timeline).
    pub started_at: Instant,
}

/// The bin's handles, held as a Bevy resource (read by every system).
#[derive(Resource)]
struct Net {
    snapshot: Arc<ArcSwap<RenderSnapshot>>,
    input: SyncSender<InputAction>,
    dropped: Arc<AtomicU64>,
    core_alive: Arc<AtomicBool>,
    started_at: Instant,
}

/// The LOCAL first-person camera state (turns immediately on mouse-look; the entity
/// orient stays server-delivered) + the last movement keys we sent (latest-wins on the
/// wire, so we only resend on change).
#[derive(Resource)]
struct CameraState {
    cam: FollowCamera,
    last_movement: MovementKeys,
}

/// The map from a delivered entity to its spawned Bevy dot entity.
#[derive(Resource, Default)]
struct DotEntities(BTreeMap<EntityId, Entity>);

/// Shared dot render assets (one sphere mesh; own/other materials) built once at setup.
#[derive(Resource)]
struct DotAssets {
    mesh: Handle<Mesh>,
    own: Handle<StandardMaterial>,
    other: Handle<StandardMaterial>,
}

/// Marker: a rendered dot (an authoritative delivered entity).
#[derive(Component)]
struct Dot;

/// Marker: the follow camera.
#[derive(Component)]
struct FollowCam;

/// Marker: the stats-HUD text node.
#[derive(Component)]
struct StatsHud;

/// Run the windowed client. BLOCKS on the calling thread until the window closes, so the
/// bin MUST call this on the MAIN thread (winit requires the event loop there) with the
/// core loop on a separate thread.
pub fn run_window(handles: RenderHandles) {
    tracing::info!("windowed client starting (Bevy {}x{})", WINDOW_W, WINDOW_H);
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.02, 0.03, 0.06)))
        .insert_resource(Net {
            snapshot: handles.snapshot,
            input: handles.input,
            dropped: handles.dropped,
            core_alive: handles.core_alive,
            started_at: handles.started_at,
        })
        .insert_resource(CameraState {
            cam: FollowCamera::new(DVec3::Y),
            last_movement: MovementKeys::default(),
        })
        .init_resource::<DotEntities>()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Voxeldust — dev client".into(),
                resolution: (WINDOW_W, WINDOW_H).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (input_system, sync_world, update_hud, exit_when_core_stops),
        )
        .run();
}

/// Spawn the camera, key light, reference scene, shared dot assets, and the HUD.
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // First-person follow camera (positioned each frame by `sync_world`).
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.6, 0.0).looking_at(Vec3::NEG_Z, Vec3::Y),
        FollowCam,
    ));
    // A single key light (dots are also emissive, so they read even unlit).
    commands.spawn((
        DirectionalLight {
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(30.0, 60.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Reference ground plate (a thin slab — motion reference for the empty stub world).
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(GROUND_HALF * 2.0, 0.2, GROUND_HALF * 2.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.06, 0.08, 0.11),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.1, 0.0),
    ));
    // Landmark pillars in a ring — distinct colors give parallax as the player walks.
    let pillar = meshes.add(Cuboid::new(1.0, LANDMARK_HEIGHT, 1.0));
    for (i, color) in LANDMARK_COLORS.iter().enumerate() {
        let angle = i as f32 / LANDMARK_COLORS.len() as f32 * std::f32::consts::TAU;
        commands.spawn((
            Mesh3d(pillar.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: *color,
                ..default()
            })),
            Transform::from_xyz(
                LANDMARK_RING_RADIUS * angle.cos(),
                LANDMARK_HEIGHT * 0.5,
                LANDMARK_RING_RADIUS * angle.sin(),
            ),
        ));
    }

    // Shared dot assets (built once, reused per spawned dot).
    commands.insert_resource(DotAssets {
        mesh: meshes.add(Sphere::new(DOT_RADIUS)),
        own: materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.9, 1.0),
            emissive: LinearRgba::rgb(0.0, 0.6, 0.8),
            ..default()
        }),
        other: materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.8, 0.35),
            emissive: LinearRgba::rgb(0.12, 0.10, 0.0),
            ..default()
        }),
    });

    // Stats HUD (top-left, multi-line native UI text; the default font is built in).
    commands.spawn((
        Text::new("connecting…"),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.85, 0.92, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        StatsHud,
    ));
}

/// Keyboard → held movement (resent only on change; latest-wins on the wire); mouse →
/// the LOCAL camera turn (immediate) AND a `Look` delta to the server. Both ride the
/// shared mailbox — byte-identical to a `vdctl` injection.
fn input_system(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<AccumulatedMouseMotion>,
    net: Res<Net>,
    mut camera: ResMut<CameraState>,
) {
    let movement = MovementKeys {
        forward: keys.pressed(KeyCode::KeyW),
        back: keys.pressed(KeyCode::KeyS),
        left: keys.pressed(KeyCode::KeyA),
        right: keys.pressed(KeyCode::KeyD),
        up: keys.pressed(KeyCode::Space),
        down: keys.pressed(KeyCode::ShiftLeft),
    };
    if movement != camera.last_movement {
        if net.input.try_send(movement.move_action()).is_err() {
            net.dropped.fetch_add(1, Ordering::Relaxed);
        }
        camera.last_movement = movement;
    }

    let delta = mouse.delta;
    if delta != Vec2::ZERO {
        // The ONE shared mapping (pixels → look delta); the camera and the server apply
        // the SAME values, so the local view and the delivered orient agree.
        if let InputAction::Look(look) = mouse_look(delta.x, delta.y) {
            camera
                .cam
                .apply_look(f64::from(look[0]), f64::from(look[1]));
            if net.input.try_send(InputAction::Look(look)).is_err() {
                net.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Sync the dot entities to the delivered+interpolated snapshot (spawn/move/despawn) and
/// place the first-person camera at the own entity's eye.
#[allow(clippy::too_many_arguments)]
fn sync_world(
    net: Res<Net>,
    camera: Res<CameraState>,
    assets: Res<DotAssets>,
    mut dots: ResMut<DotEntities>,
    mut commands: Commands,
    mut dot_tf: Query<&mut Transform, With<Dot>>,
    mut cam_tf: Query<&mut Transform, (With<FollowCam>, Without<Dot>)>,
) {
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let own = snap.own_entity();
    let rendered = snap.rendered(now_s);

    let mut seen: BTreeSet<EntityId> = BTreeSet::new();
    let mut own_world: Option<DVec3> = None;
    for (id, _sub, pose) in &rendered {
        seen.insert(*id);
        let world = world_pos(pose); // the frame-eval seam (identity in P1.5)
        if Some(*id) == own {
            own_world = Some(world);
        }
        match dots.0.get(id) {
            // Existing dot: move it (available from the frame after it was spawned).
            Some(&entity) => {
                if let Ok(mut transform) = dot_tf.get_mut(entity) {
                    transform.translation = world.as_vec3();
                }
            }
            // New dot: spawn it with the right material (own highlighted).
            None => {
                let material = if Some(*id) == own {
                    assets.own.clone()
                } else {
                    assets.other.clone()
                };
                let entity = commands
                    .spawn((
                        Mesh3d(assets.mesh.clone()),
                        MeshMaterial3d(material),
                        Transform::from_translation(world.as_vec3()),
                        Dot,
                    ))
                    .id();
                dots.0.insert(*id, entity);
            }
        }
    }
    // Despawn dots that are no longer delivered.
    dots.0.retain(|id, entity| {
        if seen.contains(id) {
            true
        } else {
            commands.entity(*entity).despawn();
            false
        }
    });

    // First-person camera at the own entity's eye, looking along the LOCAL camera basis.
    if let Some(mut transform) = cam_tf.iter_mut().next()
        && let Some(own_pos) = own_world
    {
        let eye = camera.cam.eye(own_pos);
        let target = eye + camera.cam.forward();
        *transform = Transform::from_translation(eye.as_vec3())
            .looking_at(target.as_vec3(), camera.cam.up.as_vec3());
    }
}

/// Update the stats HUD from the delivered snapshot (wire truth) — phase, the player's
/// LOCATION (realm), own entity, position, and the visible count.
fn update_hud(net: Res<Net>, mut hud: Query<&mut Text, With<StatsHud>>) {
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let rendered = snap.rendered(now_s);
    let own = snap.own_entity();
    let own_pos = rendered
        .iter()
        .find(|(id, _, _)| Some(*id) == own)
        .map(|(_, _, pose)| pose.pos);
    let location = snap.location().unwrap_or_else(|| "—".to_owned());
    let entity = own.map(|e| e.to_string()).unwrap_or_else(|| "—".to_owned());
    let pos = own_pos.map_or_else(
        || "—".to_owned(),
        |p| format!("{:.1}, {:.1}, {:.1}", p.x, p.y, p.z),
    );
    let text = format!(
        "VOXELDUST — dev client\nstatus:   {}\nlocation: {location}\nentity:   {entity}\nposition: {pos}\nvisible:  {}",
        phase_label(snap.phase()),
        rendered.len(),
    );
    if let Some(mut node) = hud.iter_mut().next() {
        node.0 = text;
    }
}

/// Close the window when the core loop has stopped — a gateway `Close` drove the session
/// to `Closed`, or the worker thread died/panicked (its drop-guard flips `core_alive`).
/// The worker->window half of the bidirectional shutdown, so a server-initiated
/// disconnect tears the window down instead of leaving a frozen zombie window.
fn exit_when_core_stops(net: Res<Net>, mut exit: MessageWriter<AppExit>) {
    if !net.core_alive.load(Ordering::Relaxed) {
        exit.write(AppExit::Success);
    }
}

/// A player-facing label for the client lifecycle phase.
fn phase_label(phase: ClientPhase) -> &'static str {
    match phase {
        ClientPhase::Connecting => "connecting",
        ClientPhase::AwaitingWelcome => "authenticating",
        ClientPhase::AwaitingSubscription => "subscribing",
        ClientPhase::Active => "live",
        ClientPhase::Closed => "closed",
    }
}
